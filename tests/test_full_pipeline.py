"""End-to-end pipeline test: video → YOLO events → clip-aware frames → LLM description.

Stages
──────
  1. YOLO11m + BoT-SORT tracking  →  raw events + clip segments
  2. Clip-aware frame sampling     →  key frames per active segment
  3. qwen3.6-plus            →  natural-language event description

Output: results/pipeline_<video_stem>.json

Usage
─────
  # 1.56-hour parking lot video, process all clips
  python tests/test_full_pipeline.py \\
      --video "_data/Surveillance Videos Dataset/7-3-day7Time082652PM100031PMCam4.mp4"

  # Quick test: only first 5 clips, 6 frames each
  python tests/test_full_pipeline.py \\
      --video "_data/Surveillance Videos Dataset/7-3-day7Time082652PM100031PMCam4.mp4" \\
      --max-clips 5 --frames 6

  # 10-hour video, first 10 clips
  python tests/test_full_pipeline.py \\
      --video "_data/Surveillance Videos Dataset/13-1-day13Time120000AM100012AMCam4.mp4" \\
      --max-clips 10 --frames 6
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# ── project imports ────────────────────────────────────────────────────────────
from video.factory.processors.event_track_pipeline import run_pipeline

# ── slim event serialisation (inlined to avoid langchain_core dependency) ─────
_EV_SHORT = {
    "motion_segment": "mv", "presence_after_motion": "pam",
    "appearance": "app",    "disappearance": "dis",
    "presence_static": "sta",
}
_SLIM_LEGEND = (
    "# ev: mv=moving pam=still_after_move app=appear dis=disappear sta=static"
    " | id=track_id | cls=class | s/e=sec | b0/b1=bbox_xyxy\n"
)

def _compact_events_str(events: list[dict]) -> str:
    slim = [{
        "ev":  _EV_SHORT.get(str(e.get("event_type", "")), "?"),
        "id":  e.get("track_id"),
        "cls": e.get("class_name"),
        "s":   round(float(e.get("start_time", 0)), 1),
        "e":   round(float(e.get("end_time",   0)), 1),
        "b0":  [int(v) for v in (e.get("start_bbox_xyxy") or [0,0,0,0])],
        "b1":  [int(v) for v in (e.get("end_bbox_xyxy")   or [0,0,0,0])],
    } for e in events]
    return _SLIM_LEGEND + json.dumps(slim, separators=(",", ":"), ensure_ascii=False)

# ── LLM client ────────────────────────────────────────────────────────────────
from openai import OpenAI

_llm_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set — check your .env file")
        _llm_client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get(
                "DASHSCOPE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
    return _llm_client


# ── frame sampling ─────────────────────────────────────────────────────────────

def _encode_frame(frame, max_edge: int = 640) -> str | None:
    h, w = frame.shape[:2]
    scale = min(1.0, max_edge / max(h, w, 1))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _merge_windows(windows: list[tuple[float, float]], gap: float = 0.5) -> list[tuple[float, float]]:
    """Merge time windows that are closer than `gap` seconds."""
    if not windows:
        return []
    ws = sorted(windows)
    merged = [list(ws[0])]
    for s, e in ws[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# event types that carry actual motion (static presence is not worth sampling densely)
_MOTION_EV_TYPES = {"motion_segment", "appearance", "disappearance"}


def adaptive_frame_count(
    clip_events: list[dict],
    clip_start: float,
    clip_end: float,
    min_frames: int = 4,
    max_frames: int = 16,
    fps_rate: float = 0.5,         # base: 1 frame per 2 seconds of motion
    subject_bonus: int = 2,        # extra frames per unique subject beyond first
) -> int:
    """Compute how many frames to sample based on motion content.

    Formula:
        n = clip(fps_rate × total_motion_sec + subject_bonus × (unique_ids - 1),
                 min_frames, max_frames)

    Examples:
        3s motion, 1 subject  → max(4, round(0.5×3 + 0))  = 4  (minimum)
        20s motion, 3 subjects→ max(4, round(0.5×20 + 4)) = 14
        60s motion, 5 subjects→ capped at max_frames=16
    """
    motion_events = [e for e in clip_events
                     if e.get("event_type") in _MOTION_EV_TYPES]

    if not motion_events:
        # No motion events → clip is static → use minimum
        return min_frames

    # Total motion duration (deduplicated via window merge)
    windows = [(float(e["start_time"]), float(e["end_time"])) for e in motion_events]
    merged  = _merge_windows(windows)
    total_motion_sec = sum(e - s for s, e in merged)

    # Unique subjects (track_ids) in this clip
    unique_ids = len({e.get("track_id") for e in motion_events if e.get("track_id") is not None})

    n = round(fps_rate * total_motion_sec) + subject_bonus * max(0, unique_ids - 1)
    return max(min_frames, min(max_frames, n))


def sample_frames_from_events(
    video_path: str,
    clip_events: list[dict],
    clip_start: float,
    clip_end: float,
    n_frames: int,
    max_edge: int = 640,
) -> list[str]:
    """Sample frames ONLY from motion-active windows defined by YOLO events.

    Why: a 30s clip may have only 7s of actual motion.
    Sampling uniformly from all 30s wastes 23/30 frames on static background.
    This function allocates frames proportionally to each motion window's duration.

    Fallback: if no motion events exist, samples uniformly from [clip_start, clip_end].
    """
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build motion windows from mv / app / dis events
    motion_events = [e for e in clip_events if e.get("event_type") in _MOTION_EV_TYPES]
    if motion_events:
        raw_windows = [(float(e["start_time"]), float(e["end_time"])) for e in motion_events]
        windows = _merge_windows(raw_windows)
    else:
        # Fallback: sample full clip uniformly
        windows = [(clip_start, clip_end)]

    active_sec = sum(e - s for s, e in windows)
    if active_sec < 0.1:
        windows    = [(clip_start, clip_end)]
        active_sec = clip_end - clip_start

    # Allocate frame budget proportionally to each window's duration
    frame_indices: list[int] = []
    for w_start, w_end in windows:
        w_dur   = w_end - w_start
        n_win   = max(1, round(n_frames * w_dur / active_sec))
        f0      = max(0, int(w_start * fps))
        f1      = min(total - 1, int(w_end * fps))
        if f1 <= f0:
            frame_indices.append(f0)
            continue
        step = (f1 - f0) / max(n_win - 1, 1)
        frame_indices += [min(f1, int(f0 + i * step)) for i in range(n_win)]

    # Deduplicate + sort; trim to n_frames
    frame_indices = sorted(set(frame_indices))[:n_frames]

    imgs: list[str] = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        enc = _encode_frame(frame, max_edge)
        if enc:
            imgs.append(enc)
    cap.release()
    return imgs


def adaptive_frame_count_long(
    motion_sec: float,
    unique_ids: int,
    min_frames: int = 4,
    max_frames: int = 16,
) -> int:
    """Logarithmic frame count for long events.

    Linear grows too fast for long durations; log scale gives diminishing returns:
        5s  → 4   (min)
        30s → 7
        120s→ 10
        600s→ 14
        3600s→ 16  (max)
    """
    import math
    if motion_sec <= 0:
        return min_frames
    # log2 scale: every doubling of duration adds ~1.5 frames
    n = round(min_frames + 2.5 * math.log2(max(1.0, motion_sec / 5.0)))
    # subject bonus
    n += max(0, unique_ids - 1)
    return max(min_frames, min(max_frames, n))


def chunk_long_event(
    video_path: str,
    clip_events: list[dict],
    clip_start: float,
    clip_end: float,
    chunk_sec: float = 60.0,
    frames_per_chunk: int = 8,
    motion_thresh: float = 3.0,
    max_edge: int = 640,
) -> list[dict]:
    """Split a long clip into fixed-size chunks; skip static chunks.

    Returns a list of chunk dicts:
        {start_sec, end_sec, frames_b64, events, skipped}

    Static chunks (mean frame-diff MAE < motion_thresh) are marked skipped=True
    and have frames_b64=[] so the caller can skip the LLM call for them.

    Usage: replace a single call_llm() for a long clip with:
        chunks = chunk_long_event(...)
        results = [call_llm(...chunk...) for chunk in chunks if not chunk['skipped']]
    """
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = clip_end - clip_start
    n_chunks = max(1, int(duration / chunk_sec) + (1 if duration % chunk_sec > 1.0 else 0))
    overlap  = min(5.0, chunk_sec * 0.1)   # 10% overlap so events on boundaries aren't missed

    chunks: list[dict] = []
    for i in range(n_chunks):
        c_start = clip_start + i * chunk_sec - (overlap if i > 0 else 0.0)
        c_end   = min(clip_end, clip_start + (i + 1) * chunk_sec + overlap)
        c_dur   = c_end - c_start

        # events that overlap this chunk
        c_events = [
            e for e in clip_events
            if float(e.get("end_time",   0)) >= c_start
            and float(e.get("start_time", 0)) <= c_end
        ]

        # quick static check: compare a few evenly spaced frames via MAE
        check_indices = [
            max(0, min(total-1, int((c_start + c_dur * t) * fps)))
            for t in [0.1, 0.3, 0.5, 0.7, 0.9]
        ]
        check_frames = []
        for fi in check_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            if ok and frm is not None:
                import cv2 as _cv2
                check_frames.append(_cv2.cvtColor(frm, _cv2.COLOR_BGR2GRAY).astype(float))

        is_static = False
        if len(check_frames) >= 2:
            import numpy as _np
            maes = [_np.mean(_np.abs(check_frames[j] - check_frames[j-1]))
                    for j in range(1, len(check_frames))]
            is_static = (sum(maes) / len(maes)) < motion_thresh

        if is_static:
            chunks.append({
                "start_sec": c_start, "end_sec": c_end,
                "events": c_events,   "frames_b64": [],
                "skipped": True,      "reason": "static",
            })
            continue

        # sample frames from motion windows within this chunk
        frames_b64 = sample_frames_from_events(
            video_path, c_events, c_start, c_end, frames_per_chunk, max_edge
        )
        chunks.append({
            "start_sec": c_start, "end_sec": c_end,
            "events": c_events,   "frames_b64": frames_b64,
            "skipped": not frames_b64, "reason": "no_frames" if not frames_b64 else None,
        })

    cap.release()
    return chunks


# ── LLM call ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a surveillance-video analyst. "
    "You receive key frames from a clip and a list of YOLO-detected events "
    "(compact JSON: ev=event_type, id=track_id, cls=class, s/e=start/end seconds, b0/b1=bbox). "
    "Describe what happened in this clip as a concise incident report. "
    "Include: subjects (class + rough appearance), actions, timing, and scene location. "
    "Output a JSON object with these fields:\n"
    "  summary   : 2-4 sentence English description of the clip\n"
    "  events    : list of {time_range:[s,e], subject:str, action:str, location:str}\n"
    "  risk_level: 'none' | 'low' | 'medium' | 'high'\n"
    "  notes     : any anomalies or uncertainties\n"
    "Output ONLY the JSON — no markdown, no prose outside the object."
)


def call_llm(
    model: str,
    frames_b64: list[str],
    clip_events: list[dict],
    clip_start: float,
    clip_end: float,
    video_name: str,
) -> dict[str, Any]:
    """Send frames + compact YOLO events to the VLM; return parsed JSON dict."""
    client = _get_client()

    events_text = _compact_events_str(clip_events) if clip_events else "(no YOLO events in this clip)"

    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Video: {video_name}\n"
                f"Clip: {clip_start:.1f}s – {clip_end:.1f}s "
                f"(duration {clip_end - clip_start:.1f}s)\n\n"
                f"YOLO events:\n{events_text}\n\n"
                f"Key frames ({len(frames_b64)} frames, uniformly sampled from the clip):"
            ),
        }
    ]
    for b64 in frames_b64:
        user_content.append({"type": "image_url", "image_url": {"url": b64}})

    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    )

    raw = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens":     resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens":      resp.usage.total_tokens,
    }

    # parse JSON
    try:
        # strip optional markdown fences
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0]
        parsed = json.loads(text.strip())
    except Exception:
        parsed = {"raw_response": raw, "parse_error": True}

    return {"llm_output": parsed, "usage": usage}


# ── main pipeline ──────────────────────────────────────────────────────────────

def run_full_pipeline(
    video_path: str,
    model: str = "qwen3.6-plus",
    max_clips: int = 0,
    frames_per_clip: int = 8,
    max_edge: int = 640,
    out_path: str = "",
    conf: float = 0.25,
    iou: float  = 0.25,
) -> dict[str, Any]:

    video_path = str(Path(video_path).resolve())
    video_name = Path(video_path).name
    out_path   = out_path or str(
        PROJECT_ROOT / "results" / f"pipeline_{Path(video_path).stem}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  VIDEO  : {video_name}")
    print(f"  MODEL  : {model}")
    print(f"  OUTPUT : {out_path}")
    print(f"{'='*60}\n")

    # ── Stage 1: YOLO pipeline ─────────────────────────────────────────────────
    print("[1/3] Running YOLO + tracking…")
    t0 = time.time()
    events, clips, meta = run_pipeline(
        video_path,
        model_path="11m",
        conf=conf,
        iou=iou,
        motion_threshold=5.0,
        min_clip_duration=1.0,
        max_static_duration=30.0,
    )
    yolo_sec = time.time() - t0

    print(f"      → {len(events)} events, {len(clips)} clip segments  ({yolo_sec:.1f}s)")
    print(f"      FPS: {meta['fps']:.1f}  Frames: {meta['total_frames']:,}  Tracks: {meta['num_tracks']}")

    if not clips:
        print("[WARN] No clip segments found — video may be entirely static.")
        result = {
            "video": video_name,
            "meta": meta,
            "clips_processed": 0,
            "clip_results": [],
        }
        Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    clips_to_run = clips[:max_clips] if max_clips > 0 else clips
    print(f"\n[2/3] Processing {len(clips_to_run)}/{len(clips)} clips with LLM ({model})…\n")

    clip_results: list[dict] = []
    total_tokens = 0
    total_cost   = 0.0
    PRICE_IN  = float(os.environ.get("QWEN_PRICE_IN",  "0.012"))  # ¥ per 1k tokens
    PRICE_OUT = float(os.environ.get("QWEN_PRICE_OUT", "0.036"))

    for ci, clip in enumerate(clips_to_run):
        clip_start = float(clip["start_sec"])
        clip_end   = float(clip["end_sec"])
        dur        = clip_end - clip_start

        # events that overlap this clip
        clip_events = [
            e for e in events
            if float(e.get("end_time", 0)) >= clip_start
            and float(e.get("start_time", 0)) <= clip_end
        ]

        # ── Strategy: long clip → chunk; short clip → single call ───────────
        LONG_CLIP_THRESH = 60.0   # clips longer than this get chunked

        print(f"  [{ci+1:>3}/{len(clips_to_run)}] clip [{clip_start:.1f}s – {clip_end:.1f}s]  "
              f"dur={dur:.1f}s  events={len(clip_events)}", end="  ", flush=True)

        t1 = time.time()

        if dur > LONG_CLIP_THRESH:
            # ── Stage 2+3: chunked path for long clips ───────────────────────
            print(f"→ chunked", flush=True)
            chunks = chunk_long_event(
                video_path, clip_events, clip_start, clip_end,
                chunk_sec=LONG_CLIP_THRESH, frames_per_chunk=frames_per_clip,
            )
            active_chunks = [c for c in chunks if not c["skipped"]]
            skipped_chunks = len(chunks) - len(active_chunks)
            print(f"       chunks={len(chunks)}  active={len(active_chunks)}  static_skipped={skipped_chunks}")

            chunk_llm_results = []
            for chk in active_chunks:
                try:
                    r = call_llm(
                        model=model,
                        frames_b64=chk["frames_b64"],
                        clip_events=chk["events"],
                        clip_start=chk["start_sec"],
                        clip_end=chk["end_sec"],
                        video_name=video_name,
                    )
                    chunk_llm_results.append({"chunk": chk, **r})
                    usage = r["usage"]
                    cost  = (usage["prompt_tokens"] * PRICE_IN
                             + usage["completion_tokens"] * PRICE_OUT) / 1000
                    total_tokens += usage["total_tokens"]
                    total_cost   += cost
                    risk = r["llm_output"].get("risk_level", "?")
                    print(f"         [{chk['start_sec']:.0f}s-{chk['end_sec']:.0f}s] "
                          f"risk={risk} tokens={usage['total_tokens']} ¥{cost:.3f}")
                except Exception as e:
                    print(f"         [{chk['start_sec']:.0f}s-{chk['end_sec']:.0f}s] [ERROR] {e}")

            elapsed = time.time() - t1
            # merge chunk summaries
            all_summaries = [r["llm_output"].get("summary", "") for r in chunk_llm_results]
            all_evs       = [ev for r in chunk_llm_results
                             for ev in r["llm_output"].get("events", [])]
            risk_order    = ["high", "medium", "low", "none", "unknown"]
            risks         = [r["llm_output"].get("risk_level", "none") for r in chunk_llm_results]
            merged_risk   = next((rv for rv in risk_order if rv in risks), "none")

            clip_results.append({
                "clip_index":    ci,
                "start_sec":     clip_start,
                "end_sec":       clip_end,
                "duration_sec":  dur,
                "n_yolo_events": len(clip_events),
                "chunked":       True,
                "n_chunks":      len(chunks),
                "active_chunks": len(active_chunks),
                "llm_output": {
                    "summary":    " | ".join(s for s in all_summaries if s),
                    "events":     all_evs,
                    "risk_level": merged_risk,
                    "notes":      f"Chunked into {len(chunks)} segments; {skipped_chunks} static skipped.",
                },
                "usage": {"total_tokens": sum(r["usage"]["total_tokens"] for r in chunk_llm_results)},
            })
            print(f"       → merged risk={merged_risk}  total ¥{sum(r['usage']['total_tokens'] for r in chunk_llm_results)*0.024/1000:.3f}  ({elapsed:.1f}s)")
            continue

        # ── Stage 2: motion-aware frame sampling (short clip) ────────────────
        motion_events = [e for e in clip_events if e.get("event_type") in _MOTION_EV_TYPES]
        total_motion  = sum(float(e["end_time"]) - float(e["start_time"]) for e in motion_events)
        unique_ids    = len({e.get("track_id") for e in motion_events if e.get("track_id") is not None})
        n_frames      = adaptive_frame_count_long(total_motion, unique_ids, max_frames=frames_per_clip)

        print(f"frames={n_frames}", end="  ", flush=True)
        frames_b64 = sample_frames_from_events(
            video_path, clip_events, clip_start, clip_end, n_frames, max_edge
        )
        if not frames_b64:
            print("→ [SKIP] no frames")
            continue

        # ── Stage 3: LLM (single call) ───────────────────────────────────────
        try:
            llm_result = call_llm(
                model=model,
                frames_b64=frames_b64,
                clip_events=clip_events,
                clip_start=clip_start,
                clip_end=clip_end,
                video_name=video_name,
            )
            elapsed = time.time() - t1
            usage = llm_result["usage"]
            cost = (usage["prompt_tokens"] * PRICE_IN
                    + usage["completion_tokens"] * PRICE_OUT) / 1000
            total_tokens += usage["total_tokens"]
            total_cost   += cost

            risk = llm_result["llm_output"].get("risk_level", "?")
            print(f"→ risk={risk}  tokens={usage['total_tokens']}  ¥{cost:.3f}  ({elapsed:.1f}s)")

            clip_results.append({
                "clip_index":  ci,
                "start_sec":   clip_start,
                "end_sec":     clip_end,
                "duration_sec": dur,
                "n_yolo_events": len(clip_events),
                "n_frames_sent": len(frames_b64),
                **llm_result,
            })

        except Exception as e:
            elapsed = time.time() - t1
            print(f"→ [ERROR] {type(e).__name__}: {e}  ({elapsed:.1f}s)")
            clip_results.append({
                "clip_index": ci,
                "start_sec":  clip_start,
                "end_sec":    clip_end,
                "error": str(e),
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    ok_clips  = [r for r in clip_results if "error" not in r]
    err_clips = [r for r in clip_results if "error" in r]

    risk_counts: dict[str, int] = {}
    for r in ok_clips:
        risk = r.get("llm_output", {}).get("risk_level", "unknown")
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    print(f"\n{'='*60}")
    print(f"[3/3] SUMMARY")
    print(f"  Clips processed : {len(clips_to_run)}")
    print(f"  OK / Error      : {len(ok_clips)} / {len(err_clips)}")
    print(f"  Total tokens    : {total_tokens:,}")
    print(f"  Total cost      : ¥{total_cost:.3f}")
    print(f"  Risk breakdown  : {risk_counts}")
    print(f"  YOLO events     : {len(events)} total across {len(clips)} clips")
    print(f"{'='*60}\n")

    result = {
        "video":           video_name,
        "model":           model,
        "meta":            meta,
        "total_clips":     len(clips),
        "clips_processed": len(clips_to_run),
        "ok":              len(ok_clips),
        "errors":          len(err_clips),
        "total_tokens":    total_tokens,
        "total_cost_cny":  round(total_cost, 4),
        "risk_breakdown":  risk_counts,
        "clip_results":    clip_results,
    }

    Path(out_path).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {out_path}")
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full pipeline test: YOLO → clip-aware frames → LLM description"
    )
    ap.add_argument("--video",      required=True,
                    help="Path to surveillance video (.mp4)")
    ap.add_argument("--model",      default="qwen3.6-plus",
                    help="VLM model name (default: qwen3.6-plus)")
    ap.add_argument("--max-clips",  type=int, default=0,
                    help="Max clips to send to LLM (0 = all). Use 3-5 for quick test.")
    ap.add_argument("--frames",     type=int, default=8,
                    help="Frames to sample per clip (default: 8)")
    ap.add_argument("--conf",       type=float, default=0.25,
                    help="YOLO confidence threshold (default: 0.25)")
    ap.add_argument("--iou",        type=float, default=0.25,
                    help="YOLO NMS IoU threshold (default: 0.25)")
    ap.add_argument("--out",        default="",
                    help="Output JSON path (default: results/pipeline_<stem>.json)")
    args = ap.parse_args()

    if not Path(args.video).exists():
        print(f"[ERROR] video not found: {args.video}")
        sys.exit(1)

    run_full_pipeline(
        video_path=args.video,
        model=args.model,
        max_clips=args.max_clips,
        frames_per_clip=args.frames,
        out_path=args.out,
        conf=args.conf,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
