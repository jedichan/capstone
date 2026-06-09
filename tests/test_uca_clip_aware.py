"""Clip-aware frame sampling evaluation.

Compares two frame sampling strategies on the same 46 Part-1 videos:
  uniform  : sample N frames uniformly from the full video  (current behavior)
  clip_aware: sample N frames proportionally from motion clip_segments only

Both use the same YOLO cache, same few-shot prompt, same YOLO text evidence.
The only difference is WHERE frames are sampled from.

This makes the token saving from motion-slicing real and measurable,
not just theoretical.

Run:
    python tests/test_uca_clip_aware.py --mode clip_aware --out result_clip_aware.json
    (uniform baseline already exists as result_uca_46_v2.json)
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
PART1_ROOT   = PROJECT_ROOT / "_data" / "Anomaly-Videos-Part-1"
PIPELINE_OUT = PROJECT_ROOT / "_pipeline_output" / "uca_eval"

PRICE_IN_PER_1K  = float(os.environ.get("QWEN_PRICE_IN",  "0.012"))
PRICE_OUT_PER_1K = float(os.environ.get("QWEN_PRICE_OUT", "0.036"))


# ── utils ─────────────────────────────────────────────────────────────────────
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


# ── frame sampling ────────────────────────────────────────────────────────────
def sample_frames_uniform(video_path: Path, n: int, max_edge: int = 640):
    """Current behavior: N frames uniformly from the full video."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps
    if total <= 0: cap.release(); raise RuntimeError("empty video")
    idxs = [int(i*(total-1)/max(1,n-1)) for i in range(n)]
    imgs = _read_frames(cap, idxs, max_edge)
    cap.release()
    return imgs, duration, None  # active_sec=None means full video


def sample_frames_clip_aware(video_path: Path, n: int,
                              clips_json: Path, max_edge: int = 640):
    """New: sample N frames proportionally from motion clip_segments only.

    If no clips found or total clip duration == 0, fall back to uniform sampling.
    Returns (imgs_b64, video_duration, active_sec).
    active_sec = total duration of clip segments used.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"cannot open {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps
    if total <= 0: cap.release(); raise RuntimeError("empty video")

    # load clips
    clips = []
    if clips_json.exists():
        raw = json.loads(clips_json.read_text(encoding="utf-8"))
        clips = [(float(c["start_sec"]), float(c["end_sec"]))
                 for c in raw.get("clip_segments", [])
                 if c["end_sec"] > c["start_sec"]]

    active_sec = sum(e - s for s, e in clips)

    # fall back to uniform if no useful clips
    if not clips or active_sec < 1.0:
        idxs = [int(i*(total-1)/max(1,n-1)) for i in range(n)]
        imgs = _read_frames(cap, idxs, max_edge)
        cap.release()
        return imgs, duration, active_sec or duration

    # allocate frames per clip proportionally to clip duration
    frame_indices: list[int] = []
    for s, e in clips:
        clip_dur = e - s
        # how many frames for this clip (at least 1)
        n_clip = max(1, round(n * clip_dur / active_sec))
        f_start = int(s * fps)
        f_end   = min(int(e * fps), total - 1)
        if f_start >= f_end:
            frame_indices.append(f_start)
            continue
        step_idxs = [int(f_start + i*(f_end - f_start)/max(1, n_clip-1))
                     for i in range(n_clip)]
        frame_indices.extend(step_idxs)

    # trim to exactly n frames, keeping temporal order
    frame_indices = sorted(set(frame_indices))
    if len(frame_indices) > n:
        # evenly subsample to n
        keep = [frame_indices[int(i*(len(frame_indices)-1)/max(1,n-1))] for i in range(n)]
        frame_indices = sorted(set(keep))

    imgs = _read_frames(cap, frame_indices, max_edge)
    cap.release()
    return imgs, duration, active_sec


def _read_frames(cap, idxs: list[int], max_edge: int) -> list[str]:
    import cv2
    imgs = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None: continue
        h, w = frame.shape[:2]
        scale = min(1.0, max_edge / max(h, w))
        if scale < 1.0: frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            imgs.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode())
    return imgs


def summarize_yolo_events(ev_json: Path, max_lines=40) -> str:
    events = json.loads(ev_json.read_text(encoding="utf-8")).get("events", [])
    lines = []
    for ev in events[:max_lines]:
        t0 = ev.get("start_sec") or ev.get("start") or 0.0
        t1 = ev.get("end_sec")   or ev.get("end")   or t0
        cls = ev.get("class_name") or ev.get("cls") or "obj"
        tid = ev.get("track_id")   or ev.get("id")  or "-"
        lines.append(f"  - [{float(t0):.1f}-{float(t1):.1f}s] {cls}#{tid}")
    if len(events) > max_lines: lines.append(f"  ... (+{len(events)-max_lines} more)")
    return "\n".join(lines) if lines else "(no tracked objects)"


def call_qwen_vl(client, model, video_name, duration, frames_b64, yolo_evidence):
    text_prompt = (
        f"Video name: {video_name}\nDuration: {duration:.2f} seconds\n\n"
        f"YOLO tracking evidence (class#track_id with time range):\n{yolo_evidence}\n\n"
        f"I am attaching {len(frames_b64)} frames sampled from the motion-active segments "
        "of this video in chronological order.\n\n"
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
    for k in ("mean_tIoU","recall@0.3","recall@0.5","recall@0.7","mean_tokenF1",
              "mean_prompt_tokens","mean_active_sec","mean_coverage_ratio"):
        vals = [r[k] for r in ok if k in r]
        agg[k] = round(sum(vals)/len(vals), 4) if vals else None
    agg["token_usage"] = dict(total_usage)
    cost = (total_usage["prompt_tokens"]*PRICE_IN_PER_1K +
            total_usage["completion_tokens"]*PRICE_OUT_PER_1K) / 1000.0
    agg["estimated_cost_cny"] = round(cost, 4)
    return agg


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["uniform","clip_aware"], default="clip_aware")
    ap.add_argument("--out",  default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    out_file = args.out or f"results/result_{args.mode}.json"

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
    cats = ["Abuse","Arrest","Arson","Assault"]
    pairs = sorted([(name, PART1_ROOT / c / f"{name}.mp4")
                    for name in gt_all for c in cats
                    if (PART1_ROOT / c / f"{name}.mp4").exists()])

    print(f"[INFO] mode={args.mode}  videos={len(pairs)}  model={model}")

    per = []; total_usage = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    t_start = time.time()

    for idx, (name, vpath) in enumerate(pairs, 1):
        gt = gt_all[name]
        r = {"video": name, "path": str(vpath), "mode": args.mode}
        print(f"\n[{idx}/{len(pairs)}] {name}")
        try:
            # YOLO always from cache
            ev_json    = PIPELINE_OUT / f"{name}_events.json"
            clips_json = PIPELINE_OUT / f"{name}_clips.json"
            if not ev_json.exists():
                raise FileNotFoundError(f"no YOLO cache for {name}")

            yolo_text = summarize_yolo_events(ev_json)
            r["num_yolo_events"] = len(json.loads(ev_json.read_text(encoding="utf-8")).get("events",[]))

            gt_dur   = float(gt.get("duration") or 0)
            n_frames = adaptive_frame_count(gt_dur)

            if args.mode == "clip_aware":
                frames, duration, active_sec = sample_frames_clip_aware(
                    vpath, n_frames, clips_json)
                r["active_sec"]       = round(active_sec, 2)
                r["coverage_ratio"]   = round(active_sec / duration, 4) if duration > 0 else 1.0
            else:
                frames, duration, _ = sample_frames_uniform(vpath, n_frames)
                r["active_sec"]     = round(duration, 2)   # full video
                r["coverage_ratio"] = 1.0

            r["n_frames_used"]    = len(frames)
            r["duration_video"]   = round(duration, 2)
            print(f"  Frames: {len(frames)} | active={r['active_sec']}s "
                  f"/ total={r['duration_video']}s ({r['coverage_ratio']*100:.1f}% coverage)")

            t1 = time.time()
            pred, usage = call_qwen_vl(client, model, name,
                                       gt.get("duration") or duration,
                                       frames, yolo_text)
            r["llm_sec"] = round(time.time()-t1, 1)
            r["usage"]   = usage
            r["mean_prompt_tokens"] = usage["prompt_tokens"]
            for k in total_usage: total_usage[k] += usage[k]
            print(f"  LLM: {usage} ({r['llm_sec']}s)")

            m = greedy_match(pred.get("timestamps",[]), pred.get("sentences",[]),
                             gt["timestamps"], gt["sentences"])
            ious = [x[2] for x in m]; f1s = [x[3] for x in m]
            n = max(len(gt["timestamps"]), 1)
            r.update({
                "num_gt": len(gt["timestamps"]),
                "num_pred": len(pred.get("timestamps",[])),
                "mean_tIoU":    round(sum(ious)/n, 4),
                "recall@0.3":   round(sum(1 for i in ious if i>=0.3)/n, 4),
                "recall@0.5":   round(sum(1 for i in ious if i>=0.5)/n, 4),
                "recall@0.7":   round(sum(1 for i in ious if i>=0.7)/n, 4),
                "mean_tokenF1": round(sum(f1s)/n, 4),
                "mean_active_sec":      r["active_sec"],
                "mean_coverage_ratio":  r["coverage_ratio"],
                "pred": pred,
            })
            print(f"  tIoU={r['mean_tIoU']}  R@0.5={r['recall@0.5']}  "
                  f"tokens={usage['prompt_tokens']}")
        except Exception as e:
            r["error"]     = f"{type(e).__name__}: {e}"
            r["traceback"] = traceback.format_exc(limit=3)
            print(f"  [ERROR] {r['error']}")
        per.append(r)

        agg = _aggregate(per, total_usage, time.time()-t_start)
        out_path = PROJECT_ROOT / out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"model": model, "mode": args.mode, "aggregate": agg, "per_video": per},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"DONE  mode={args.mode}  n={agg['n_ok']}/{agg['n_total']}")
    print(f"  tIoU={agg['mean_tIoU']}  R@0.5={agg['recall@0.5']}  "
          f"cost=CNY{agg['estimated_cost_cny']}")
    print(f"  mean_prompt_tokens={agg['mean_prompt_tokens']}")
    print(f"  mean_coverage_ratio={agg['mean_coverage_ratio']} "
          f"(mean_active_sec={agg['mean_active_sec']}s)")
    print(f"[saved] {PROJECT_ROOT / out_file}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
