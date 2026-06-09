"""Full UCA evaluation across Part-1 + Part-2 + Part-3 (173 videos).

Run:
    python tests/test_uca_all_parts.py --out result_uca_173.json

YOLO results are cached per-video in _pipeline_output/uca_eval/.
LLM config: adaptive frames + few-shot (same as v2).
"""
from __future__ import annotations
import argparse, base64, json, os, re, sys, time, traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.node.uca_prompts import UCA_SYSTEM_PROMPT  # noqa

UCA_TEST_JSON = (PROJECT_ROOT / "_data" / "Surveillance-Video-Understanding"
                 / "UCF Annotation" / "json" / "UCFCrime_Test.json")
DATA_ROOT    = PROJECT_ROOT / "_data"
PIPELINE_OUT = PROJECT_ROOT / "_pipeline_output" / "uca_eval"
PIPELINE_OUT.mkdir(parents=True, exist_ok=True)

PRICE_IN_PER_1K  = float(os.environ.get("QWEN_PRICE_IN",  "0.012"))
PRICE_OUT_PER_1K = float(os.environ.get("QWEN_PRICE_OUT", "0.036"))

PART_CATS = {
    "Part-1": ["Abuse", "Arrest", "Arson", "Assault"],
    "Part-2": ["Burglary", "Explosion", "Fighting"],
    "Part-3": ["RoadAccidents", "Robbery", "Shooting"],
}


# ── utils ─────────────────────────────────────────────────────────────────────
def find_all_videos(gt: dict) -> list[tuple[str, Path, str]]:
    """Return (name, mp4_path, part) for every GT video present on disk."""
    out = []
    for name in gt:
        for part, cats in PART_CATS.items():
            part_dir = DATA_ROOT / f"Anomaly-Videos-{part}"
            for cat in cats:
                p = part_dir / cat / f"{name}.mp4"
                if p.exists():
                    out.append((name, p, part))
                    break
            else:
                continue
            break
    return sorted(out, key=lambda x: (x[2], x[0]))


def temporal_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

_TOKEN_RE = re.compile(r"[a-z0-9]+")
def _tokens(s): return _TOKEN_RE.findall(s.lower())
def token_f1(pred, gt):
    from collections import Counter
    p, g = _tokens(pred), _tokens(gt)
    if not p or not g: return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if not overlap: return 0.0
    pr, rc = overlap/len(p), overlap/len(g)
    return 2*pr*rc/(pr+rc)

def greedy_match(pred_ts, pred_sent, gt_ts, gt_sent):
    used = set(); out = []
    for gi, gts in enumerate(gt_ts):
        best_i, best_iou = -1, 0.0
        for pi, pts in enumerate(pred_ts):
            if pi in used: continue
            iou = temporal_iou(tuple(pts), tuple(gts))
            if iou > best_iou: best_iou, best_i = iou, pi
        if best_i >= 0: used.add(best_i)
        out.append((gi, best_i, best_iou,
                    token_f1(pred_sent[best_i], gt_sent[gi]) if best_i >= 0 else 0.0))
    return out

def adaptive_frame_count(dur, base=12):
    if dur < 30:  return base
    if dur < 90:  return max(base, 18)
    if dur < 180: return max(base, 24)
    return max(base, 32)

def sample_frames_b64(video_path, n=12, max_edge=640):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps
    if total <= 0: cap.release(); raise RuntimeError("empty video")
    idxs = [int(i*(total-1)/max(1,n-1)) for i in range(n)]
    imgs = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok: continue
        h, w = frame.shape[:2]; scale = min(1.0, max_edge/max(h,w))
        if scale < 1.0: frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2: imgs.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode())
    cap.release()
    return imgs, duration

def run_yolo_pipeline(video_path: Path) -> dict:
    base = video_path.stem
    ev_json = PIPELINE_OUT / f"{base}_events.json"
    if ev_json.exists():
        return json.loads(ev_json.read_text(encoding="utf-8"))
    from video.factory.processors.event_track_pipeline import run_pipeline, save_pipeline_output
    events, clips, meta = run_pipeline(str(video_path), conf=0.25, iou=0.25,
        motion_threshold=5.0, min_clip_duration=1.0, max_static_duration=30.0,
        tracker="botsort_reid")
    save_pipeline_output(events, clips, meta, PIPELINE_OUT)
    return {"events": events, "meta": meta, "clips": clips}

def summarize_yolo_events(pipeline_out, max_lines=40):
    events = pipeline_out.get("events") or []
    lines = []
    for ev in events[:max_lines]:
        t0 = ev.get("start_sec") or ev.get("start") or 0.0
        t1 = ev.get("end_sec") or ev.get("end") or t0
        cls = ev.get("class_name") or ev.get("cls") or "obj"
        tid = ev.get("track_id") or ev.get("id") or "-"
        lines.append(f"  - [{float(t0):.1f}-{float(t1):.1f}s] {cls}#{tid}")
    if len(events) > max_lines: lines.append(f"  ... (+{len(events)-max_lines} more)")
    return "\n".join(lines) if lines else "(no tracked objects)"

def call_qwen_vl(client, model, video_name, duration, frames_b64, yolo_evidence):
    text_prompt = (
        f"Video name: {video_name}\nDuration: {duration:.2f} seconds\n\n"
        f"YOLO tracking evidence (class#track_id with time range):\n{yolo_evidence}\n\n"
        f"I am attaching {len(frames_b64)} evenly sampled frames from this video "
        "in chronological order (frame i is at time ≈ i/(N-1) * duration).\n\n"
        "Produce UCA-format JSON with fields: video_name, duration, timestamps, sentences.\n"
        "timestamps[i] = [start_sec, end_sec], 0.1s precision, 0 <= start < end <= duration.\n"
        "sentences[i] describes WHO / WHAT / WHERE for that segment in ONE English sentence.\n"
        "Return ONLY the JSON object, no markdown fence, no commentary."
    )
    content = [{"type": "text", "text": text_prompt}]
    for url in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": url}})
    resp = client.chat.completions.create(
        model=model, temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": UCA_SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
    )
    text = (resp.choices[0].message.content or "").replace("```json","").replace("```","").strip()
    payload = json.loads(text)
    u = getattr(resp, "usage", None)
    usage = {"prompt_tokens": int(getattr(u,"prompt_tokens",0) or 0),
             "completion_tokens": int(getattr(u,"completion_tokens",0) or 0),
             "total_tokens": int(getattr(u,"total_tokens",0) or 0)}
    return payload, usage

def _aggregate(per, total_usage, elapsed):
    ok = [r for r in per if "error" not in r]
    agg = {"n_total": len(per), "n_ok": len(ok), "n_failed": len(per)-len(ok),
           "elapsed_sec": round(elapsed, 1)}
    for k in ("mean_tIoU","recall@0.3","recall@0.5","recall@0.7","mean_tokenF1"):
        vals = [r[k] for r in ok if k in r]
        agg[k] = round(sum(vals)/len(vals),4) if vals else None
    agg["token_usage"] = dict(total_usage)
    cost = (total_usage["prompt_tokens"]*PRICE_IN_PER_1K +
            total_usage["completion_tokens"]*PRICE_OUT_PER_1K) / 1000.0
    agg["estimated_cost_cny"] = round(cost, 4)
    # per-part breakdown
    for part in ("Part-1","Part-2","Part-3"):
        part_ok = [r for r in ok if r.get("part") == part]
        if not part_ok: continue
        agg[f"{part}_n"] = len(part_ok)
        for k in ("mean_tIoU","recall@0.5"):
            vals = [r[k] for r in part_ok if k in r]
            agg[f"{part}_{k}"] = round(sum(vals)/len(vals),4) if vals else None
    return agg


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",   default="result_uca_173.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--resume", action="store_true", help="skip videos already in --out file")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv; load_dotenv(PROJECT_ROOT / ".env")
    except Exception: pass
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key: print("[FATAL] DASHSCOPE_API_KEY missing"); return 2
    model = args.model or os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus")
    from openai import OpenAI
    client = OpenAI(api_key=api_key,
                    base_url=os.environ.get("DASHSCOPE_URL",
                                            "https://dashscope.aliyuncs.com/compatible-mode/v1"))

    gt_all = json.loads(UCA_TEST_JSON.read_text(encoding="utf-8"))
    all_videos = find_all_videos(gt_all)
    print(f"[INFO] Found {len(all_videos)} videos across Part-1/2/3 | model={model}")
    from collections import Counter
    part_counts = Counter(p for _,_,p in all_videos)
    for k,v in sorted(part_counts.items()): print(f"  {k}: {v}")

    # resume support
    per: list[dict] = []
    total_usage = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    done_names: set[str] = set()
    out_path = PROJECT_ROOT / args.out
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        per = existing.get("per_video", [])
        done_names = {r["video"] for r in per}
        for r in per:
            if "usage" in r:
                for k in total_usage: total_usage[k] += r["usage"].get(k,0)
        print(f"[INFO] resuming: {len(done_names)} already done, {len(all_videos)-len(done_names)} remaining")

    t_start = time.time()
    remaining = [(n,p,pt) for n,p,pt in all_videos if n not in done_names]

    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] torch device={dev} {torch.cuda.get_device_name(0) if dev=='cuda' else ''}")
    except Exception: pass

    for idx, (name, vpath, part) in enumerate(remaining, 1):
        gt = gt_all[name]
        r: dict = {"video": name, "path": str(vpath), "part": part}
        total_done = len(done_names) + idx
        print(f"\n[{total_done}/{len(all_videos)}] {name} ({part})")
        t0 = time.time()
        try:
            yolo_out = run_yolo_pipeline(vpath)
            r["num_yolo_events"] = len(yolo_out.get("events",[]))
            print(f"  YOLO: {r['num_yolo_events']} events ({time.time()-t0:.1f}s)")

            gt_dur = float(gt.get("duration") or 0)
            n_frames = adaptive_frame_count(gt_dur)
            frames, duration = sample_frames_b64(vpath, n=n_frames)
            r["n_frames_used"] = len(frames)
            r["duration_video"] = round(duration, 2)
            r["duration_gt"] = gt.get("duration")

            yolo_text = summarize_yolo_events(yolo_out)
            t1 = time.time()
            pred, usage = call_qwen_vl(client, model, name,
                                       gt.get("duration") or duration,
                                       frames, yolo_text)
            r["llm_sec"] = round(time.time()-t1, 1)
            r["usage"] = usage
            for k in total_usage: total_usage[k] += usage[k]
            print(f"  LLM: {usage} ({r['llm_sec']}s)")

            m = greedy_match(pred.get("timestamps",[]), pred.get("sentences",[]),
                             gt["timestamps"], gt["sentences"])
            ious = [x[2] for x in m]; f1s = [x[3] for x in m]
            n = max(len(gt["timestamps"]),1)
            r.update({
                "num_gt": len(gt["timestamps"]),
                "num_pred": len(pred.get("timestamps",[])),
                "mean_tIoU": round(sum(ious)/n, 4),
                "recall@0.3": round(sum(ious and [1 for i in ious if i>=0.3] or [0])/n, 4),
                "recall@0.5": round(sum(1 for i in ious if i>=0.5)/n, 4),
                "recall@0.7": round(sum(1 for i in ious if i>=0.7)/n, 4),
                "mean_tokenF1": round(sum(f1s)/n, 4),
                "pred": pred,
                "gt_timestamps": gt["timestamps"],
                "gt_sentences": gt["sentences"],
            })
            print(f"  tIoU={r['mean_tIoU']}  R@0.5={r['recall@0.5']}  tokF1={r['mean_tokenF1']}")
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
            r["traceback"] = traceback.format_exc(limit=3)
            print(f"  [ERROR] {r['error']}")
        per.append(r)
        done_names.add(name)

        # incremental save
        agg = _aggregate(per, total_usage, time.time()-t_start)
        out_path.write_text(
            json.dumps({"model": model, "aggregate": agg, "per_video": per},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    agg = _aggregate(per, total_usage, time.time()-t_start)
    print(f"\n{'='*70}")
    print(f"DONE  n={agg['n_ok']}/{agg['n_total']}  "
          f"tIoU={agg['mean_tIoU']}  R@0.5={agg['recall@0.5']}  "
          f"cost=CNY{agg['estimated_cost_cny']}")
    for part in ("Part-1","Part-2","Part-3"):
        if f"{part}_mean_tIoU" in agg:
            print(f"  {part}: n={agg[f'{part}_n']}  "
                  f"tIoU={agg[f'{part}_mean_tIoU']}  R@0.5={agg[f'{part}_recall@0.5']}")
    print(f"[saved] {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
