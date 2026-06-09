"""UCA 完整评测: YOLO本地跟踪(CUDA) + qwen3.6-plus 多模态 + token 统计.

流程
----
对每个视频:
    1) 用项目 pipeline (video.factory.coordinator) 跑 YOLO+BoT-SORT 跟踪, 得到 events
    2) 从视频中均匀采样 N 张帧, base64 编码
    3) 组装多模态 prompt: 系统提示(UCA) + 帧图像 + YOLO 事件证据, 调 qwen3.6-plus
    4) 解析 JSON 输出, 与 UCA GT 比较 (tIoU + token-F1 + schema)
    5) 累加 usage.prompt_tokens / completion_tokens
最后:
    输出逐视频指标 + 总体聚合 + token 用量 + 估算费用

运行
----
    python tests/test_uca_full_evaluation.py --num 46 --frames 12 --out result_uca_46.json

预设视频: Part-1 与 UCA Test 的 46 个交集.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.node.uca_prompts import UCA_SYSTEM_PROMPT, UCA_OUTPUT_SCHEMA  # noqa: E402

UCA_TEST_JSON = (
    PROJECT_ROOT / "_data" / "Surveillance-Video-Understanding"
    / "UCF Annotation" / "json" / "UCFCrime_Test.json"
)
PART1_ROOT = PROJECT_ROOT / "_data" / "Anomaly-Videos-Part-1"
PIPELINE_OUT = PROJECT_ROOT / "_pipeline_output" / "uca_eval"
PIPELINE_OUT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# qwen3.6-plus 定价 (阿里云百炼 2025): 输入 0.012/千 tokens, 输出 0.036/千 tokens (CNY)
# 视觉 tokens 同输入价. 若实际有调整, 最后报告以 response.usage 为准.
PRICE_IN_PER_1K = float(os.environ.get("QWEN_PRICE_IN", "0.012"))
PRICE_OUT_PER_1K = float(os.environ.get("QWEN_PRICE_OUT", "0.036"))


# --------------------------------------------------------------------------- #
# 数据 & 指标                                                                   #
# --------------------------------------------------------------------------- #
def load_uca_gt() -> dict[str, dict[str, Any]]:
    with UCA_TEST_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_part1_videos(gt: dict[str, dict[str, Any]]) -> list[tuple[str, Path]]:
    """返回 (video_name, mp4_path) 列表: Part-1 与 UCA Test 的交集."""
    cats = ["Abuse", "Arrest", "Arson", "Assault"]
    out: list[tuple[str, Path]] = []
    for name in gt:
        for c in cats:
            p = PART1_ROOT / c / f"{name}.mp4"
            if p.exists():
                out.append((name, p))
                break
    return sorted(out)


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


def token_f1(pred: str, gt: str) -> float:
    p, g = _tokens(pred), _tokens(gt)
    if not p or not g:
        return 0.0
    from collections import Counter
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    pr, rc = overlap / len(p), overlap / len(g)
    return 2 * pr * rc / (pr + rc)


def greedy_match(pred_ts, pred_sent, gt_ts, gt_sent):
    used: set[int] = set()
    out = []
    for gi, gts in enumerate(gt_ts):
        best_i, best_iou = -1, 0.0
        for pi, pts in enumerate(pred_ts):
            if pi in used:
                continue
            iou = temporal_iou(tuple(pts), tuple(gts))
            if iou > best_iou:
                best_iou, best_i = iou, pi
        if best_i >= 0:
            used.add(best_i)
            out.append((gi, best_i, best_iou, token_f1(pred_sent[best_i], gt_sent[gi])))
        else:
            out.append((gi, -1, 0.0, 0.0))
    return out


def validate_uca(payload: dict[str, Any], duration: float) -> list[str]:
    errs = []
    for k in ("video_name", "duration", "timestamps", "sentences"):
        if k not in payload:
            errs.append(f"missing:{k}")
    ts = payload.get("timestamps") or []
    sents = payload.get("sentences") or []
    if len(ts) != len(sents):
        errs.append(f"len_mismatch:{len(ts)}!={len(sents)}")
    for i, t in enumerate(ts):
        try:
            s, e = float(t[0]), float(t[1])
            if not (0 <= s < e <= duration + 0.5):
                errs.append(f"ts[{i}]_oob:{s}-{e}/{duration}")
        except Exception:
            errs.append(f"ts[{i}]_bad")
    return errs


# --------------------------------------------------------------------------- #
# 视频处理                                                                     #
# --------------------------------------------------------------------------- #
def run_yolo_pipeline(video_path: Path, cache_dir: Path) -> dict[str, Any]:
    """跑 YOLO+track, 返回 {events:list, meta:dict}. 有缓存直接读."""
    base = video_path.stem
    ev_json = cache_dir / f"{base}_events.json"
    if ev_json.exists():
        with ev_json.open("r", encoding="utf-8") as f:
            return json.load(f)

    # 直接调 run_pipeline, 避免 coordinator 触发 langchain import
    from video.factory.processors.event_track_pipeline import run_pipeline, save_pipeline_output

    events, clips, meta = run_pipeline(
        str(video_path),
        conf=0.25,
        iou=0.25,
        motion_threshold=5.0,
        min_clip_duration=1.0,
        max_static_duration=30.0,
        tracker="botsort_reid",
    )
    save_pipeline_output(events, clips, meta, cache_dir)
    return {"events": events, "meta": meta, "clips": clips}


def adaptive_frame_count(duration_sec: float, base: int = 12) -> int:
    """根据视频时长自适应: <30s=base, 30-90s=18, 90-180s=24, >=180s=32."""
    if duration_sec < 30:
        return base
    if duration_sec < 90:
        return max(base, 18)
    if duration_sec < 180:
        return max(base, 24)
    return max(base, 32)


def sample_frames_b64(video_path: Path, n: int = 12, max_edge: int = 640) -> tuple[list[str], float]:
    """均匀采 n 张帧, 返回 (base64_data_urls, duration_sec)."""
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps > 0 else 0.0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"empty video: {video_path}")

    idxs = [int(i * (total - 1) / max(1, n - 1)) for i in range(n)]
    imgs_b64: list[str] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, max_edge / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok2:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        imgs_b64.append(f"data:image/jpeg;base64,{b64}")
    cap.release()
    return imgs_b64, duration


def summarize_yolo_events(pipeline_out: dict[str, Any], max_lines: int = 40) -> str:
    """把 YOLO events 压成紧凑文本喂给 LLM 作为 object-level 证据."""
    events = pipeline_out.get("events") or []
    lines = []
    for ev in events[:max_lines]:
        t0 = ev.get("start_sec") or ev.get("start") or ev.get("t_start") or 0.0
        t1 = ev.get("end_sec") or ev.get("end") or ev.get("t_end") or t0
        cls = ev.get("class_name") or ev.get("cls") or ev.get("label") or "obj"
        tid = ev.get("track_id") or ev.get("id") or "-"
        lines.append(f"  - [{float(t0):.1f}-{float(t1):.1f}s] {cls}#{tid}")
    if len(events) > max_lines:
        lines.append(f"  ... (+{len(events)-max_lines} more)")
    if not lines:
        return "(no tracked objects)"
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM 调用                                                                     #
# --------------------------------------------------------------------------- #
def call_qwen_vl(
    client, model: str, video_name: str, duration: float,
    frames_b64: list[str], yolo_evidence: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    text_prompt = (
        f"Video name: {video_name}\n"
        f"Duration: {duration:.2f} seconds\n\n"
        f"YOLO tracking evidence (class#track_id with time range):\n{yolo_evidence}\n\n"
        f"I am attaching {len(frames_b64)} evenly sampled frames from this video "
        "in chronological order (frame i is at time ≈ i/(N-1) * duration).\n\n"
        "Produce UCA-format JSON with fields: video_name, duration, timestamps, sentences.\n"
        "timestamps[i] = [start_sec, end_sec], 0.1s precision, 0 <= start < end <= duration.\n"
        "sentences[i] describes WHO / WHAT / WHERE for that segment in ONE English sentence.\n"
        "Return ONLY the JSON object, no markdown fence, no commentary."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    for url in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": url}})

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": UCA_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    text = resp.choices[0].message.content or ""
    text = text.replace("```json", "").replace("```", "").strip()
    payload = json.loads(text)

    usage_obj = getattr(resp, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }
    return payload, usage


# --------------------------------------------------------------------------- #
# 主流程                                                                       #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=46)
    ap.add_argument("--frames", type=int, default=12, help="基准帧数 (短视频). 长视频会自适应增加")
    ap.add_argument("--adaptive-frames", action="store_true", default=True,
                    help="按视频时长自适应帧数 (默认开启)")
    ap.add_argument("--fixed-frames", action="store_true",
                    help="强制固定帧数, 关闭自适应")
    ap.add_argument("--model", type=str, default=None, help="override DASHSCOPE_CHAT_MODEL")
    ap.add_argument("--out", type=str, default="results/result_uca_full.json")
    ap.add_argument("--skip-yolo", action="store_true", help="跳过 YOLO, 仅用 qwen 看帧")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[FATAL] DASHSCOPE_API_KEY missing")
        return 2
    model = args.model or os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus")
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=api_key, base_url=os.environ.get(
        "DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))

    gt_all = load_uca_gt()
    pairs = find_part1_videos(gt_all)[: args.num]
    print(f"[INFO] {len(pairs)} videos to process | model={model} | frames/video={args.frames}")

    # torch device info
    try:
        import torch  # type: ignore
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] torch={torch.__version__} device={dev}"
              f" {torch.cuda.get_device_name(0) if dev=='cuda' else ''}")
    except Exception:
        print("[WARN] torch not available, YOLO will fail")

    per: list[dict[str, Any]] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    t_start = time.time()

    for idx, (name, vpath) in enumerate(pairs, 1):
        gt = gt_all[name]
        r: dict[str, Any] = {"video": name, "path": str(vpath)}
        print(f"\n[{idx}/{len(pairs)}] {name}")
        t0 = time.time()
        try:
            yolo_out = {"events": []}
            if not args.skip_yolo:
                yolo_out = run_yolo_pipeline(vpath, PIPELINE_OUT)
                r["num_yolo_events"] = len(yolo_out.get("events", []))
                print(f"  YOLO: {r['num_yolo_events']} events ({time.time()-t0:.1f}s)")

            t1 = time.time()
            gt_dur = float(gt.get("duration") or 0)
            use_adaptive = (not args.fixed_frames)
            n_frames = adaptive_frame_count(gt_dur, base=args.frames) if use_adaptive else args.frames
            frames, duration = sample_frames_b64(vpath, n=n_frames)
            r["n_frames_used"] = len(frames)
            r["duration_video"] = round(duration, 2)
            r["duration_gt"] = gt.get("duration")
            print(f"  Frames: {len(frames)} sampled (adaptive={use_adaptive}), duration={duration:.1f}s")

            yolo_text = summarize_yolo_events(yolo_out) if not args.skip_yolo else "(skipped)"
            pred, usage = call_qwen_vl(
                client, model, name,
                gt.get("duration") or duration,
                frames, yolo_text,
            )
            r["llm_sec"] = round(time.time() - t1, 1)
            r["usage"] = usage
            for k in total_usage:
                total_usage[k] += usage[k]
            print(f"  LLM: {usage} ({r['llm_sec']}s)")

            errs = validate_uca(pred, gt["duration"])
            r["schema_errors"] = errs
            pairs_match = greedy_match(
                pred.get("timestamps", []), pred.get("sentences", []),
                gt["timestamps"], gt["sentences"],
            )
            ious = [p[2] for p in pairs_match]
            f1s = [p[3] for p in pairs_match]
            n = max(len(gt["timestamps"]), 1)
            r["num_gt"] = len(gt["timestamps"])
            r["num_pred"] = len(pred.get("timestamps", []))
            r["mean_tIoU"] = round(sum(ious) / n, 4)
            r["recall@0.3"] = round(sum(1 for i in ious if i >= 0.3) / n, 4)
            r["recall@0.5"] = round(sum(1 for i in ious if i >= 0.5) / n, 4)
            r["recall@0.7"] = round(sum(1 for i in ious if i >= 0.7) / n, 4)
            r["mean_tokenF1"] = round(sum(f1s) / n, 4)
            r["pred"] = pred
            print(f"  tIoU={r['mean_tIoU']}  R@0.5={r['recall@0.5']}  tokF1={r['mean_tokenF1']}")
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
            r["traceback"] = traceback.format_exc(limit=3)
            print(f"  [ERROR] {r['error']}")
        per.append(r)

        # 每个视频后增量落盘, 防中断丢失
        out_path = PROJECT_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        agg = _aggregate(per, total_usage, t_start)
        out_path.write_text(json.dumps(
            {"model": model, "frames_per_video": args.frames,
             "aggregate": agg, "per_video": per},
            ensure_ascii=False, indent=2), encoding="utf-8")

    agg = _aggregate(per, total_usage, time.time() - t_start)
    print("\n" + "=" * 70)
    print(f"AGGREGATE: {json.dumps(agg, ensure_ascii=False, indent=2)}")
    print("=" * 70)
    return 0


def _aggregate(per: list[dict[str, Any]], total_usage: dict[str, int], elapsed: float) -> dict[str, Any]:
    ok = [r for r in per if "error" not in r]
    agg: dict[str, Any] = {
        "n_total": len(per),
        "n_ok": len(ok),
        "n_failed": len(per) - len(ok),
        "elapsed_sec": round(elapsed if elapsed < 1e6 else time.time() - elapsed, 1),
    }
    for k in ("mean_tIoU", "recall@0.3", "recall@0.5", "recall@0.7", "mean_tokenF1"):
        vals = [r[k] for r in ok if k in r]
        agg[k] = round(sum(vals) / len(vals), 4) if vals else None
    agg["token_usage"] = dict(total_usage)
    cost_cny = (total_usage["prompt_tokens"] * PRICE_IN_PER_1K
                + total_usage["completion_tokens"] * PRICE_OUT_PER_1K) / 1000.0
    agg["estimated_cost_cny"] = round(cost_cny, 4)
    return agg


if __name__ == "__main__":
    raise SystemExit(main())
