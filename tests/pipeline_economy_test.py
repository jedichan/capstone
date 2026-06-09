"""Surveillance Video Pipeline Economy Test
===========================================
Measures how much computation the motion-filter saves on long static videos.

Pipeline stages (mock-based, drop-in replaceable with real models):
  Frame → Motion Filter → YOLO Detector → Event Slicer → LLM Refiner

Usage:
    python tests/pipeline_economy_test.py --video path/to/video.mp4
    python tests/pipeline_economy_test.py --video path/to/video.mp4 --threshold 20 --sanity
    python tests/pipeline_economy_test.py --video path/to/video.mp4 --stride 5 --max-frames 50000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. Counter dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineCounter:
    total_frames:    int   = 0
    skipped_frames:  int   = 0   # motion filter said "static" → skip
    yolo_calls:      int   = 0   # frames that passed motion filter
    event_slices:    int   = 0   # complete event buffers flushed to LLM
    llm_calls:       int   = 0   # actual LLM invocations
    elapsed_seconds: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mock modules  (same signatures as real pipeline — swap in-place)
# ─────────────────────────────────────────────────────────────────────────────
def mock_motion_filter(frame: np.ndarray,
                       prev_frame: np.ndarray | None,
                       threshold: float = 25.0) -> bool:
    """Frame-difference motion filter.

    Returns True  → frame is STATIC  → skip downstream stages.
    Returns False → frame has motion → continue to YOLO.

    Uses mean absolute error (MAE) of grayscale difference.
    threshold: MAE below this value → considered static.
    """
    if prev_frame is None:
        return False  # always process the first frame

    gray_cur  = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mae = float(np.mean(np.abs(gray_cur - gray_prev)))
    return mae < threshold   # True = static = skip


def mock_yolo_detect(frame: np.ndarray) -> list[dict[str, Any]]:
    """Mock YOLO detector.

    Returns 0–3 detections.  30% chance of empty result (no object visible).
    Each detection: {label, confidence, bbox: [x, y, w, h]}

    Hook: replace with real YOLO call, e.g.:
        results = yolo_model(frame)
        return [{"label": r.label, "confidence": r.conf, "bbox": r.bbox} for r in results]
    """
    LABELS = ["person", "car", "bicycle", "motorcycle", "truck"]

    if random.random() < 0.30:          # 30% chance → no detection
        return []

    n = random.randint(1, 3)
    h, w = frame.shape[:2]
    detections = []
    for _ in range(n):
        bw = random.randint(w // 10, w // 3)
        bh = random.randint(h // 10, h // 3)
        bx = random.randint(0, w - bw)
        by = random.randint(0, h - bh)
        detections.append({
            "label":      random.choice(LABELS),
            "confidence": round(random.uniform(0.5, 0.99), 2),
            "bbox":       [bx, by, bw, bh],
        })
    return detections


def mock_llm_refine(event_slice: list[dict[str, Any]]) -> dict[str, Any]:
    """Mock LLM refiner.

    Receives a list of per-frame detection dicts collected during an event window.
    Returns a structured summary.

    Hook: replace with real VLM call, e.g.:
        frames_b64 = [encode_frame(f["frame"]) for f in event_slice]
        return call_qwen_vl(client, model, frames_b64, ...)
    """
    labels  = [d["label"] for frame_det in event_slice
               for d in frame_det.get("detections", [])]
    unique  = list(set(labels))
    summary = (f"Observed {len(labels)} object(s) across "
               f"{len(event_slice)} frame(s): {', '.join(unique) or 'none'}.")
    return {
        "summary":      summary,
        "entity_count": len(unique),
        "frame_count":  len(event_slice),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main test function
# ─────────────────────────────────────────────────────────────────────────────
def run_economy_test(
    video_path: str,
    motion_threshold: float = 25.0,
    stride: int = 1,           # process every Nth frame (for long videos)
    max_frames: int = 0,       # 0 = no limit; set e.g. 50000 for quick test
    event_gap_frames: int = 30,# flush event buffer after this many consecutive static frames
) -> tuple[dict[str, Any], list[int], list[int]]:
    """Run the full mock pipeline on a video and return economy metrics.

    Returns:
        (result_dict, skipped_indices, active_indices)
    """
    counter = PipelineCounter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_raw  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_raw / fps
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[VIDEO] {Path(video_path).name}")
    print(f"        {duration_s:.0f}s ({duration_s/3600:.2f}h)  "
          f"{total_raw} frames  {fps:.1f}fps  {width}×{height}")
    if stride > 1:
        print(f"        stride={stride} → processing ~{total_raw//stride} frames")
    if max_frames:
        print(f"        max_frames={max_frames}")
    print()

    prev_frame      : np.ndarray | None = None
    event_buffer    : list[dict]        = []   # accumulates active frame detections
    static_run      : int               = 0    # consecutive static frames since last active
    skipped_indices : list[int]         = []
    active_indices  : list[int]         = []

    t_start = time.time()
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # stride: skip frames cheaply at read level
        if stride > 1 and frame_idx % stride != 0:
            frame_idx += 1
            continue

        counter.total_frames += 1   # ← total_frames +1

        # ── Stage 1: Motion Filter ──────────────────────────────────────────
        is_static = mock_motion_filter(frame, prev_frame, threshold=motion_threshold)
        prev_frame = frame.copy()

        if is_static:
            counter.skipped_frames += 1          # ← skipped_frames +1
            skipped_indices.append(frame_idx)
            static_run += 1

            # flush event buffer if static gap is long enough
            if static_run >= event_gap_frames and event_buffer:
                _flush_event_buffer(event_buffer, counter)
                event_buffer = []

            frame_idx += 1
            processed += 1
            if max_frames and processed >= max_frames:
                break
            continue

        # ── Stage 2: YOLO Detector ──────────────────────────────────────────
        static_run = 0
        counter.yolo_calls += 1                  # ← yolo_calls +1
        active_indices.append(frame_idx)
        detections = mock_yolo_detect(frame)

        # ── Stage 3: Event Slicer ───────────────────────────────────────────
        event_buffer.append({
            "frame_idx":  frame_idx,
            "detections": detections,
        })

        frame_idx += 1
        processed += 1

        # progress every 5000 processed frames
        if processed % 5000 == 0:
            elapsed = time.time() - t_start
            pct = 100 * processed / max(total_raw // stride, 1)
            print(f"  [{processed:>7} frames | {pct:4.1f}% | {elapsed:.0f}s] "
                  f"skipped={counter.skipped_frames} yolo={counter.yolo_calls} "
                  f"llm={counter.llm_calls}")

        if max_frames and processed >= max_frames:
            break

    # flush remaining event buffer
    if event_buffer:
        _flush_event_buffer(event_buffer, counter)

    cap.release()
    counter.elapsed_seconds = round(time.time() - t_start, 2)

    # ── Derived metrics ──────────────────────────────────────────────────────
    tf = max(counter.total_frames, 1)
    result = {
        **asdict(counter),
        # video metadata
        "video_path":      video_path,
        "video_duration_s": round(duration_s, 1),
        "video_fps":       fps,
        "stride":          stride,
        "motion_threshold": motion_threshold,
        # 4 economy metrics
        "frame_skip_rate":  round(counter.skipped_frames / tf, 4),
        "yolo_savings":     round(1 - counter.yolo_calls / tf, 4),
        "llm_call_rate":    round(counter.llm_calls / max(counter.event_slices, 1), 4),
        "throughput_fps":   round(tf / counter.elapsed_seconds, 2),
        # token savings extrapolation (32 frames × 275 tokens per LLM call)
        "frames_per_llm_call":   round(tf / max(counter.llm_calls, 1), 1),
        "estimated_token_saving_pct": round(counter.skipped_frames / tf * 100, 1),
    }
    return result, skipped_indices, active_indices


def _flush_event_buffer(buffer: list[dict], counter: PipelineCounter):
    """Flush a completed event buffer → one LLM call."""
    counter.event_slices += 1   # ← event_slices +1
    _result = mock_llm_refine(buffer)
    counter.llm_calls += 1      # ← llm_calls +1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Report
# ─────────────────────────────────────────────────────────────────────────────
def print_report(result: dict):
    skip_rate = result["frame_skip_rate"]
    bar_len   = 40
    filled    = int(bar_len * skip_rate)
    bar       = "█" * filled + "░" * (bar_len - filled)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║         PIPELINE ECONOMY REPORT                     ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Video     : {Path(result['video_path']).name[:38]:<38} ║")
    print(f"║  Duration  : {result['video_duration_s']:.0f}s "
          f"({result['video_duration_s']/3600:.2f}h){'':<22} ║")
    print(f"║  Threshold : motion_threshold = {result['motion_threshold']:<21} ║")
    print(f"║  Stride    : {result['stride']:<40} ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Total frames processed : {result['total_frames']:>10,}{'':<15} ║")
    print(f"║  Skipped (static)       : {result['skipped_frames']:>10,}{'':<15} ║")
    print(f"║  YOLO calls             : {result['yolo_calls']:>10,}{'':<15} ║")
    print(f"║  Event slices           : {result['event_slices']:>10,}{'':<15} ║")
    print(f"║  LLM calls              : {result['llm_calls']:>10,}{'':<15} ║")
    print(f"║  Elapsed                : {result['elapsed_seconds']:>10.1f}s{'':<14} ║")
    print(f"║  Throughput             : {result['throughput_fps']:>10.1f} fps{'':<12} ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  ECONOMY METRICS                                     ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Frame skip rate  : {skip_rate*100:5.1f}%  [{bar}] ║")
    print(f"║  YOLO savings     : {result['yolo_savings']*100:5.1f}%  "
          f"(frames not sent to YOLO){'':<4} ║")
    print(f"║  LLM call rate    : {result['llm_call_rate']:5.2f}   "
          f"(LLM calls per event slice){'':<3} ║")
    print(f"║  Est. token saving: {result['estimated_token_saving_pct']:5.1f}%  "
          f"vs naive full-video sampling{'':<1} ║")
    print("╠══════════════════════════════════════════════════════╣")

    if skip_rate >= 0.80:
        print("║  ✅  PASS  — skip rate ≥ 80%: motion filter effective   ║")
    elif skip_rate >= 0.50:
        print("║  ⚠️  WARN  — skip rate 50–80%: moderate static content  ║")
    else:
        print("║  ❌  WARN  — skip rate < 50%: video is mostly active    ║")

    print("╚══════════════════════════════════════════════════════╝")
    print()


def save_report(result: dict, output_path: str = "economy_report.json"):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sanity Check
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check(video_path: str,
                 skip_frame_indices: list[int],
                 active_frame_indices: list[int],
                 sample_n: int = 10,
                 stride: int = 1):
    """Save sample frames from skipped and active sets for manual inspection."""
    sanity_dir = Path("sanity")
    skip_dir   = sanity_dir / "skipped"
    act_dir    = sanity_dir / "active"
    skip_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    def _save_samples(indices: list[int], out_dir: Path, label: str):
        if not indices:
            print(f"  [{label}] no frames to sample"); return
        samples = random.sample(indices, min(sample_n, len(indices)))
        samples.sort()
        for fi in samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi * stride)
            ok, frame = cap.read()
            if not ok: continue
            path = out_dir / f"frame_{fi:08d}.jpg"
            cv2.imwrite(str(path), frame)
        print(f"  [{label}] saved {len(samples)} frames → {out_dir}")

    print("\n[Sanity Check]")
    _save_samples(skip_frame_indices,   skip_dir, "skipped")
    _save_samples(active_frame_indices, act_dir,  "active")
    cap.release()
    print(f"  Open ./sanity/ to verify skip logic visually.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Surveillance pipeline economy test — measures motion-filter savings.")
    ap.add_argument("--video",      required=True,
                    help="Path to surveillance video (.mp4)")
    ap.add_argument("--threshold",  type=float, default=25.0,
                    help="Motion MAE threshold (default: 25). Higher = more frames skipped.")
    ap.add_argument("--stride",     type=int, default=1,
                    help="Process every Nth frame (default: 1). Use 5–10 for 10-hr videos.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Cap total frames processed (0 = no limit). Use 50000 for quick test.")
    ap.add_argument("--sanity",     action="store_true",
                    help="Save sample skipped/active frames to ./sanity/ for visual check.")
    ap.add_argument("--out",        default="results/economy_report.json",
                    help="Output JSON path (default: results/economy_report.json)")
    args = ap.parse_args()

    if not Path(args.video).exists():
        print(f"[ERROR] video not found: {args.video}"); sys.exit(1)

    result, skipped_idx, active_idx = run_economy_test(
        video_path=args.video,
        motion_threshold=args.threshold,
        stride=args.stride,
        max_frames=args.max_frames,
    )

    print_report(result)
    save_report(result, args.out)

    if args.sanity:
        sanity_check(args.video, skipped_idx, active_idx, stride=args.stride)


if __name__ == "__main__":
    main()
