"""Ablation study on 46 Part-1 UCA Test videos.

Three configs (all reuse YOLO cache, only LLM calls differ):
  full      : YOLO evidence + few-shot examples  (= v2, skip re-running, load from file)
  no_yolo   : frames only, NO YOLO evidence in prompt
  no_fewshot: YOLO evidence, but system prompt WITHOUT few-shot examples

Run:
    python tests/test_uca_ablation.py --config no_yolo   --out result_ablation_no_yolo.json
    python tests/test_uca_ablation.py --config no_fewshot --out result_ablation_no_fewshot.json
"""
from __future__ import annotations
import argparse, base64, json, os, re, sys, time, traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.node.uca_prompts import UCA_SYSTEM_PROMPT  # noqa: E402

UCA_TEST_JSON = (PROJECT_ROOT / "_data" / "Surveillance-Video-Understanding"
                 / "UCF Annotation" / "json" / "UCFCrime_Test.json")
PART1_ROOT   = PROJECT_ROOT / "_data" / "Anomaly-Videos-Part-1"
PIPELINE_OUT = PROJECT_ROOT / "_pipeline_output" / "uca_eval"

PRICE_IN_PER_1K  = float(os.environ.get("QWEN_PRICE_IN",  "0.012"))
PRICE_OUT_PER_1K = float(os.environ.get("QWEN_PRICE_OUT", "0.036"))

# ── stripped system prompt (no few-shot examples) ────────────────────────────
UCA_SYSTEM_PROMPT_NOFEWSHOT = (
    "You are a surveillance-video dense captioning assistant. "
    "Your job is to describe a video as a list of non-overlapping or lightly-overlapping events "
    "in the UCA (UCF-Crime Annotation) format. "
    "Rules:\n"
    "1. Output a single JSON object, no markdown fences, no explanation.\n"
    "2. Timestamps are in seconds with 0.1s precision; start < end; end <= duration.\n"
    "3. sentences[i] corresponds to timestamps[i]; lengths MUST match.\n"
    "4. Each sentence is ONE English sentence, target ~20 words, matching the UCA style.\n"
    "5. Do not invent objects or actions that were not present in the input evidence.\n"
    "6. Keep sentences factual and surveillance-oriented (no speculation, no emotions, no judgment).\n"
    "7. Describe subjects generically (\"a man\", \"two men\", \"a white car\"); "
    "use simple surveillance phrases (\"in the middle of the road\", \"behind the counter\"); "
    "chain actions with simple verbs (walk, stand, push, fall, pick up, run, enter, leave).\n"
)


# ── utils (copied from main eval) ────────────────────────────────────────────
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
    if dur < 30: return base
    if dur < 90: return max(base, 18)
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

def summarize_yolo_events(ev_json_path, max_lines=40):
    events = json.loads(Path(ev_json_path).read_text(encoding="utf-8")).get("events", [])
    lines = []
    for ev in events[:max_lines]:
        t0 = ev.get("start_sec") or ev.get("start") or 0.0
        t1 = ev.get("end_sec") or ev.get("end") or t0
        cls = ev.get("class_name") or ev.get("cls") or "obj"
        tid = ev.get("track_id") or ev.get("id") or "-"
        lines.append(f"  - [{float(t0):.1f}-{float(t1):.1f}s] {cls}#{tid}")
    if len(events) > max_lines: lines.append(f"  ... (+{len(events)-max_lines} more)")
    return "\n".join(lines) if lines else "(no tracked objects)"


# ── LLM call ─────────────────────────────────────────────────────────────────
def call_qwen_vl(client, model, video_name, duration, frames_b64,
                 yolo_evidence, system_prompt):
    text_prompt = (
        f"Video name: {video_name}\nDuration: {duration:.2f} seconds\n\n"
        + (f"YOLO tracking evidence (class#track_id with time range):\n{yolo_evidence}\n\n"
           if yolo_evidence else "")
        + f"I am attaching {len(frames_b64)} evenly sampled frames from this video "
        "in chronological order.\n\n"
        "Produce UCA-format JSON with fields: video_name, duration, timestamps, sentences.\n"
        "timestamps[i] = [start_sec, end_sec], 0.1s precision.\n"
        "Return ONLY the JSON object, no markdown fence, no commentary."
    )
    content = [{"type": "text", "text": text_prompt}]
    for url in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": url}})
    resp = client.chat.completions.create(
        model=model, temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
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


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["no_yolo","no_fewshot","llm_only"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    out_file = args.out or f"results/result_ablation_{args.config}.json"

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

    # same 46 videos as v2
    cats = ["Abuse","Arrest","Arson","Assault"]
    pairs = []
    for name in gt_all:
        for c in cats:
            p = PART1_ROOT / c / f"{name}.mp4"
            if p.exists(): pairs.append((name, p)); break
    pairs = sorted(pairs)

    # config-specific settings
    use_yolo   = (args.config not in ("no_yolo", "llm_only"))
    sys_prompt = UCA_SYSTEM_PROMPT_NOFEWSHOT if args.config in ("no_fewshot", "llm_only") else UCA_SYSTEM_PROMPT
    print(f"[INFO] ablation={args.config}  videos={len(pairs)}  model={model}")
    print(f"       use_yolo={use_yolo}  few_shot={'yes' if sys_prompt is UCA_SYSTEM_PROMPT else 'no'}")

    per = []; total_usage = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    t_start = time.time()

    for idx, (name, vpath) in enumerate(pairs, 1):
        gt = gt_all[name]
        r = {"video": name, "path": str(vpath)}
        print(f"\n[{idx}/{len(pairs)}] {name}")
        try:
            # YOLO always cached — just load or skip
            yolo_text = ""
            if use_yolo:
                ev_path = PIPELINE_OUT / f"{name}_events.json"
                if ev_path.exists():
                    yolo_text = summarize_yolo_events(ev_path)
                    r["num_yolo_events"] = len(json.loads(ev_path.read_text(encoding="utf-8")).get("events",[]))
                else:
                    print(f"  [WARN] no cached YOLO for {name}, running pipeline")
                    from video.factory.processors.event_track_pipeline import run_pipeline, save_pipeline_output
                    events, clips, meta = run_pipeline(str(vpath), conf=0.25, iou=0.25,
                        motion_threshold=5.0, min_clip_duration=1.0,
                        max_static_duration=30.0, tracker="botsort_reid")
                    save_pipeline_output(events, clips, meta, PIPELINE_OUT)
                    yolo_text = "\n".join(
                        f"  - [{float(ev.get('start_sec',0)):.1f}-{float(ev.get('end_sec',0)):.1f}s] "
                        f"{ev.get('class_name','obj')}#{ev.get('track_id','-')}"
                        for ev in events[:40])
                    r["num_yolo_events"] = len(events)

            gt_dur = float(gt.get("duration") or 0)
            n_frames = adaptive_frame_count(gt_dur)
            frames, duration = sample_frames_b64(vpath, n=n_frames)
            r["n_frames_used"] = len(frames); r["duration_video"] = round(duration, 2)

            t1 = time.time()
            pred, usage = call_qwen_vl(client, model, name,
                                       gt.get("duration") or duration,
                                       frames, yolo_text, sys_prompt)
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
                "recall@0.3": round(sum(1 for i in ious if i>=0.3)/n, 4),
                "recall@0.5": round(sum(1 for i in ious if i>=0.5)/n, 4),
                "recall@0.7": round(sum(1 for i in ious if i>=0.7)/n, 4),
                "mean_tokenF1": round(sum(f1s)/n, 4),
                "pred": pred,
            })
            print(f"  tIoU={r['mean_tIoU']}  R@0.5={r['recall@0.5']}  tokF1={r['mean_tokenF1']}")
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
            r["traceback"] = traceback.format_exc(limit=3)
            print(f"  [ERROR] {r['error']}")
        per.append(r)

        # incremental save
        ok = [x for x in per if "error" not in x]
        agg = {"n_total": len(per), "n_ok": len(ok), "n_failed": len(per)-len(ok),
               "elapsed_sec": round(time.time()-t_start, 1)}
        for k in ("mean_tIoU","recall@0.3","recall@0.5","recall@0.7","mean_tokenF1"):
            vals = [x[k] for x in ok if k in x]
            agg[k] = round(sum(vals)/len(vals),4) if vals else None
        agg["token_usage"] = dict(total_usage)
        cost = (total_usage["prompt_tokens"]*PRICE_IN_PER_1K +
                total_usage["completion_tokens"]*PRICE_OUT_PER_1K) / 1000.0
        agg["estimated_cost_cny"] = round(cost, 4)
        out_path = PROJECT_ROOT / out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"model": model, "config": args.config, "aggregate": agg, "per_video": per},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] saved {PROJECT_ROOT / out_file}")
    print(f"  mean_tIoU={agg['mean_tIoU']}  R@0.5={agg['recall@0.5']}  cost=CNY{agg['estimated_cost_cny']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
