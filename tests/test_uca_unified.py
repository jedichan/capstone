"""Unified UCA evaluation script — all parts, all configs, all models.

Config options (--config):
  full            : YOLO evidence + few-shot  [default]
  no_yolo         : frames only, no YOLO evidence
  no_fewshot      : YOLO evidence, no few-shot examples
  llm_only        : frames only, no YOLO, no few-shot  (same as raw_llm)
  clip_aware      : YOLO evidence + few-shot, frames sampled from motion clips only
  clip_no_fewshot : clip-aware frames + YOLO evidence, no few-shot examples
  clip_no_yolo    : clip-aware frames + few-shot, no YOLO evidence

Parts (--parts, comma-separated):
  1,2,3  [default: all three]

Examples:
  # Full pipeline, all 173 videos, qwen3.6-plus
  python tests/test_uca_unified.py --config full --parts 1,2,3 --model qwen3.6-plus --out results/full_173_plus.json

  # Ablation: LLM only
  python tests/test_uca_unified.py --config llm_only --parts 1,2,3 --model qwen3.6-plus --out results/llm_only_173_plus.json

  # Clip-aware
  python tests/test_uca_unified.py --config clip_aware --parts 1,2,3 --model qwen3.6-plus --out results/clip_aware_173_plus.json
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

UCA_SYSTEM_PROMPT_NOFEWSHOT = (
    "You are a surveillance-video dense captioning assistant. "
    "Your job is to describe a video as a list of non-overlapping or lightly-overlapping events "
    "in the UCA (UCF-Crime Annotation) format. "
    "Rules:\n"
    "1. Output a single JSON object, no markdown fences, no explanation.\n"
    "2. Timestamps are in seconds with 0.1s precision; start < end; end <= duration.\n"
    "3. sentences[i] corresponds to timestamps[i]; lengths MUST match.\n"
    "4. Each sentence is ONE English sentence, target ~20 words.\n"
    "5. Do not invent objects or actions not present in the input evidence.\n"
    "6. Keep sentences factual and surveillance-oriented.\n"
    "7. Describe subjects generically; use simple surveillance phrases and verbs.\n"
)


# ── helpers ───────────────────────────────────────────────────────────────────
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

def optimal_match(pred_ts, pred_sent, gt_ts, gt_sent):
    """Hungarian-optimal matching (maximises sum of tIoU over all GT segments)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    n_g, n_p = len(gt_ts), len(pred_ts)
    if n_g == 0 or n_p == 0:
        return [(gi, -1, 0.0, 0.0) for gi in range(n_g)]
    # cost matrix: negated tIoU (scipy minimises)
    cost = np.zeros((n_g, n_p))
    for gi, gts in enumerate(gt_ts):
        for pi, pts in enumerate(pred_ts):
            cost[gi, pi] = -temporal_iou(tuple(pts), tuple(gts))
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned = {r: c for r, c in zip(row_ind, col_ind)}
    out = []
    for gi in range(n_g):
        if gi in assigned:
            pi = assigned[gi]
            iou = -cost[gi, pi]
            tf1 = token_f1(pred_sent[pi], gt_sent[gi]) if pi < len(pred_sent) else 0.0
            out.append((gi, pi, iou, tf1))
        else:
            out.append((gi, -1, 0.0, 0.0))
    return out

# keep old name as alias so existing call-sites still work
greedy_match = optimal_match

def adaptive_frame_count(dur, base=12):
    if dur < 30:  return base
    if dur < 90:  return max(base, 18)
    if dur < 180: return max(base, 24)
    return max(base, 32)


# ── frame sampling ─────────────────────────────────────────────────────────────
def _encode_frame(frame, max_edge: int) -> str | None:
    import cv2
    h, w = frame.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok: return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

def _read_frames_at(cap, idxs: list[int], max_edge: int) -> list[str]:
    import cv2
    imgs = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None: continue
        enc = _encode_frame(frame, max_edge)
        if enc: imgs.append(enc)
    return imgs

def sample_frames_uniform(video_path: Path, n: int, max_edge=640):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps
    if total <= 0: cap.release(); raise RuntimeError("empty video")
    idxs = [int(i*(total-1)/max(1,n-1)) for i in range(n)]
    imgs = _read_frames_at(cap, idxs, max_edge)
    cap.release()
    return imgs, duration, duration   # active_sec = full duration

def sample_frames_clip_aware(video_path: Path, n: int, clips_json: Path, max_edge=640):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"cannot open {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps
    if total <= 0: cap.release(); raise RuntimeError("empty video")

    clips = []
    if clips_json.exists():
        raw = json.loads(clips_json.read_text(encoding="utf-8"))
        clips = [(float(c["start_sec"]), float(c["end_sec"]))
                 for c in raw.get("clip_segments", [])
                 if c["end_sec"] > c["start_sec"]]
    active_sec = sum(e-s for s, e in clips)

    if not clips or active_sec < 1.0:
        idxs = [int(i*(total-1)/max(1,n-1)) for i in range(n)]
        imgs = _read_frames_at(cap, idxs, max_edge)
        cap.release()
        return imgs, duration, duration

    frame_indices: list[int] = []
    for s, e in clips:
        n_clip = max(1, round(n * (e-s) / active_sec))
        f0, f1 = int(s*fps), min(int(e*fps), total-1)
        if f0 >= f1: frame_indices.append(f0); continue
        frame_indices.extend(
            int(f0 + i*(f1-f0)/max(1,n_clip-1)) for i in range(n_clip))

    frame_indices = sorted(set(frame_indices))
    if len(frame_indices) > n:
        frame_indices = [frame_indices[int(i*(len(frame_indices)-1)/max(1,n-1))]
                         for i in range(n)]
        frame_indices = sorted(set(frame_indices))

    imgs = _read_frames_at(cap, frame_indices, max_edge)
    cap.release()
    return imgs, duration, active_sec


# ── YOLO helpers ──────────────────────────────────────────────────────────────
def run_yolo_pipeline(video_path: Path):
    base = video_path.stem
    ev_json = PIPELINE_OUT / f"{base}_events.json"
    if ev_json.exists():
        return json.loads(ev_json.read_text(encoding="utf-8"))
    from video.factory.processors.event_track_pipeline import run_pipeline, save_pipeline_output
    events, clips, meta = run_pipeline(str(video_path), conf=0.25, iou=0.25,
        motion_threshold=5.0, min_clip_duration=1.0,
        max_static_duration=30.0, tracker="botsort_reid")
    save_pipeline_output(events, clips, meta, PIPELINE_OUT)
    return {"events": events, "meta": meta, "clips": clips}

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


# ── LLM ───────────────────────────────────────────────────────────────────────
def call_qwen_vl(client, model, video_name, duration,
                 frames_b64, yolo_evidence, system_prompt, clip_aware=False):
    evidence_block = (
        f"YOLO tracking evidence (class#track_id with time range):\n{yolo_evidence}\n\n"
        if yolo_evidence else ""
    )
    frame_desc = ("motion-active segments of this " if clip_aware
                  else "evenly sampled frames from this ")
    text_prompt = (
        f"Video name: {video_name}\nDuration: {duration:.2f} seconds\n\n"
        f"{evidence_block}"
        f"I am attaching {len(frames_b64)} frames from the {frame_desc}video "
        "in chronological order.\n\n"
        "Produce UCA-format JSON with fields: video_name, duration, timestamps, sentences.\n"
        "timestamps[i] = [start_sec, end_sec], 0.1s precision, 0 <= start < end <= duration.\n"
        "sentences[i] describes WHO / WHAT / WHERE in ONE English sentence.\n"
        "Return ONLY the JSON object, no markdown fence, no commentary."
    )
    content = [{"type": "text", "text": text_prompt}]
    for url in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": url}})
    resp = client.chat.completions.create(
        model=model, temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user",   "content": content}],
    )
    text = (resp.choices[0].message.content or "").replace("```json","").replace("```","").strip()
    payload = _safe_parse_json(text)
    u = getattr(resp, "usage", None)
    usage = {"prompt_tokens":    int(getattr(u,"prompt_tokens",0)    or 0),
             "completion_tokens":int(getattr(u,"completion_tokens",0) or 0),
             "total_tokens":     int(getattr(u,"total_tokens",0)      or 0)}
    return payload, usage


def _safe_parse_json(text: str) -> dict:
    """Parse JSON, attempting repair if truncated."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # attempt 1: truncate to last complete array element by finding last ']'
    # and close the object
    repaired = text
    # drop incomplete last sentence/timestamp entry
    for closer in ('"}]', '"]', ']'):
        idx = repaired.rfind(closer)
        if idx != -1:
            candidate = repaired[:idx + len(closer)]
            # close the outer object if needed
            open_b = candidate.count('{') - candidate.count('}')
            candidate += '}' * max(0, open_b)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    # attempt 2: extract with regex
    import re
    m = re.search(r'"timestamps"\s*:\s*(\[.*?\])', text, re.S)
    ts = json.loads(m.group(1)) if m else []
    m2 = re.search(r'"sentences"\s*:\s*(\[.*?\])', text, re.S)
    sents = json.loads(m2.group(1)) if m2 else []
    m3 = re.search(r'"duration"\s*:\s*([0-9.]+)', text)
    dur = float(m3.group(1)) if m3 else 0.0
    m4 = re.search(r'"video_name"\s*:\s*"([^"]+)"', text)
    vname = m4.group(1) if m4 else ""
    if ts or sents:
        n = min(len(ts), len(sents))
        return {"video_name": vname, "duration": dur,
                "timestamps": ts[:n], "sentences": sents[:n]}
    raise ValueError(f"Cannot parse JSON response: {text[:200]}")


# ── aggregate ─────────────────────────────────────────────────────────────────
def _aggregate(per, total_usage, elapsed):
    ok = [r for r in per if "error" not in r]
    agg = {"n_total": len(per), "n_ok": len(ok),
           "n_failed": len(per)-len(ok), "elapsed_sec": round(elapsed, 1)}
    for k in ("mean_tIoU","recall@0.3","recall@0.5","recall@0.7","mean_tokenF1",
              "coverage_ratio"):
        vals = [r[k] for r in ok if k in r]
        agg[k] = round(sum(vals)/len(vals), 4) if vals else None
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
    ap.add_argument("--config", default="full",
                    choices=["full","no_yolo","no_fewshot","llm_only","clip_aware",
                             "clip_no_fewshot","clip_no_yolo"])
    ap.add_argument("--parts",  default="1,2,3",
                    help="comma-separated part numbers, e.g. 1,2,3")
    ap.add_argument("--model",  default=None)
    ap.add_argument("--out",    default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # resolve parts
    part_keys = [f"Part-{p.strip()}" for p in args.parts.split(",")]

    # default output name
    parts_str = args.parts.replace(",","")
    out_file = args.out or f"results/result_{args.config}_parts{parts_str}.json"

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

    # config-specific settings
    use_yolo   = args.config not in ("no_yolo", "llm_only", "clip_no_yolo")
    use_fewshot = args.config not in ("no_fewshot", "llm_only", "clip_no_fewshot")
    clip_aware  = args.config in ("clip_aware", "clip_no_fewshot", "clip_no_yolo")
    sys_prompt  = UCA_SYSTEM_PROMPT if use_fewshot else UCA_SYSTEM_PROMPT_NOFEWSHOT

    # find videos
    gt_all = json.loads(UCA_TEST_JSON.read_text(encoding="utf-8"))
    all_videos: list[tuple[str, Path, str]] = []
    for part in part_keys:
        cats = PART_CATS.get(part, [])
        part_dir = DATA_ROOT / f"Anomaly-Videos-{part}"
        for name in gt_all:
            for cat in cats:
                p = part_dir / cat / f"{name}.mp4"
                if p.exists():
                    all_videos.append((name, p, part)); break
    all_videos = sorted(all_videos, key=lambda x: (x[2], x[0]))

    print(f"[INFO] config={args.config}  parts={part_keys}  model={model}")
    print(f"       use_yolo={use_yolo}  use_fewshot={use_fewshot}  clip_aware={clip_aware}")
    print(f"       videos={len(all_videos)}  out={out_file}")
    from collections import Counter
    for k,v in sorted(Counter(p for _,_,p in all_videos).items()): print(f"  {k}: {v}")

    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] torch device={dev} {torch.cuda.get_device_name(0) if dev=='cuda' else ''}")
    except Exception: pass

    # resume
    per: list[dict] = []
    total_usage = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    done_names: set[str] = set()
    out_path = PROJECT_ROOT / out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        per = existing.get("per_video", [])
        done_names = {r["video"] for r in per}
        for r in per:
            if "usage" in r:
                for k in total_usage: total_usage[k] += r["usage"].get(k,0)
        print(f"[INFO] resuming: {len(done_names)} done, {len(all_videos)-len(done_names)} remaining")

    t_start = time.time()
    remaining = [(n,p,pt) for n,p,pt in all_videos if n not in done_names]

    for idx, (name, vpath, part) in enumerate(remaining, 1):
        gt = gt_all[name]
        r: dict = {"video": name, "path": str(vpath), "part": part, "config": args.config}
        total_done = len(done_names) + idx
        print(f"\n[{total_done}/{len(all_videos)}] {name} ({part})")
        t0 = time.time()
        try:
            # YOLO (from cache or run)
            yolo_text = ""
            ev_json    = PIPELINE_OUT / f"{name}_events.json"
            clips_json = PIPELINE_OUT / f"{name}_clips.json"

            if use_yolo or clip_aware:
                if not ev_json.exists():
                    yolo_out = run_yolo_pipeline(vpath)
                    r["num_yolo_events"] = len(yolo_out.get("events",[]))
                else:
                    r["num_yolo_events"] = len(
                        json.loads(ev_json.read_text(encoding="utf-8")).get("events",[]))
                print(f"  YOLO: {r.get('num_yolo_events',0)} events ({time.time()-t0:.1f}s)")
                if use_yolo:
                    yolo_text = summarize_yolo_events(ev_json)

            # frame sampling
            gt_dur   = float(gt.get("duration") or 0)
            n_frames = adaptive_frame_count(gt_dur)

            if clip_aware:
                frames, duration, active_sec = sample_frames_clip_aware(
                    vpath, n_frames, clips_json)
            else:
                frames, duration, active_sec = sample_frames_uniform(vpath, n_frames)

            r["n_frames_used"]  = len(frames)
            r["duration_video"] = round(duration, 2)
            r["active_sec"]     = round(active_sec, 2)
            r["coverage_ratio"] = round(active_sec/duration, 4) if duration > 0 else 1.0
            print(f"  Frames: {len(frames)} | coverage={r['coverage_ratio']*100:.1f}%")

            # LLM
            t1 = time.time()
            pred, usage = call_qwen_vl(
                client, model, name, gt.get("duration") or duration,
                frames, yolo_text, sys_prompt, clip_aware=clip_aware)
            r["llm_sec"] = round(time.time()-t1, 1)
            r["usage"]   = usage
            for k in total_usage: total_usage[k] += usage[k]
            print(f"  LLM: {usage} ({r['llm_sec']}s)")

            # metrics
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
                "pred": pred,
            })
            print(f"  tIoU={r['mean_tIoU']}  R@0.5={r['recall@0.5']}  tokF1={r['mean_tokenF1']}")
        except Exception as e:
            r["error"]     = f"{type(e).__name__}: {e}"
            r["traceback"] = traceback.format_exc(limit=3)
            print(f"  [ERROR] {r['error']}")
        per.append(r)
        done_names.add(name)

        # incremental save
        agg = _aggregate(per, total_usage, time.time()-t_start)
        out_path.write_text(
            json.dumps({"model": model, "config": args.config,
                        "parts": part_keys, "aggregate": agg, "per_video": per},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    agg = _aggregate(per, total_usage, time.time()-t_start)
    print(f"\n{'='*70}")
    print(f"DONE  config={args.config}  n={agg['n_ok']}/{agg['n_total']}")
    print(f"  tIoU={agg['mean_tIoU']}  R@0.5={agg['recall@0.5']}  cost=CNY{agg['estimated_cost_cny']}")
    for part in part_keys:
        if f"{part}_n" in agg:
            print(f"  {part}: n={agg[f'{part}_n']}  "
                  f"tIoU={agg[f'{part}_mean_tIoU']}  R@0.5={agg[f'{part}_recall@0.5']}")
    print(f"[saved] {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
