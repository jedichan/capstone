"""agent/test/mevid_test/run_custom_eval.py
===========================================
Custom MEVID evaluation using pre-selected questions from sampled_10.json.

Uses cached pipeline output — no re-running YOLO/OSNet/topology.
Evaluates:
  - yes/no accuracy
  - temporal IoU (for yes questions with expected_time)

Usage:
  conda activate capstone
  python agent/test/mevid_test/run_custom_eval.py
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("custom_eval")

# ── Config ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PIPELINE_CACHE = ROOT / "_cache" / "mevid_pipeline" / "13-50_pipeline.json"
REFINED_CACHE = ROOT / "_cache" / "mevid_pipeline" / "13-50_refined.json"
APPEARANCE_CACHE = ROOT / "_cache" / "mevid_pipeline" / "13-50_appearance_refined.json"
SAMPLED_JSON = HERE / "sampled_10.json"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.6-plus"

# DashScope pricing (per 1K tokens)
PRICE_IN = 0.012   # prompt
PRICE_OUT = 0.036  # completion

SYSTEM_PROMPT = """You are a video analysis assistant analyzing multi-camera surveillance footage.

You will receive:
- Video frames from one camera, evenly sampled across the 5-minute clip
- Tracker context with detected persons, their appearance descriptions, and cross-camera matches

RULES:
1. Answer "yes" or "no" followed by a time range in brackets.
   For yes: "yes, [M:SS-M:SS]" (e.g., "yes, [1:50-2:01]") — Give a REASONABLE time window (~30s)
   For no:  "no"
2. Use the frames as PRIMARY evidence. The tracker context provides hints.
3. For appearance: match descriptions loosely (e.g. "beige" ≈ "tan" ≈ "light brown", "dark" ≈ "black").
   If the context describes a person matching the question AND frames confirm, answer yes with the time.
4. For event (exit/enter/direction): use tracker hints to find the right window, verify in frames.
5. For cross-camera: check if the SAME person entity appears in both cameras within overlapping time.
6. For negative (cross-camera verification): be CONSERVATIVE. Only answer yes if there is CLEAR evidence
   that exactly the same person appears in both cameras. Differences in clothing details = answer no.
7. Always provide a time range with yes answers. The time range should span the relevant video segment."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_time(t_str: str) -> float | None:
    """Parse M:SS or MM:SS to seconds."""
    parts = t_str.strip().split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + float(parts[1])
        except (ValueError, TypeError):
            return None
    return None


def parse_time_range(t_str: str) -> tuple[float | None, float | None]:
    """Parse 'start-end' to (start_sec, end_sec)."""
    if not t_str or t_str.upper() == "N/A":
        return None, None
    t_str = t_str.strip().replace("[", "").replace("]", "")
    if "-" not in t_str:
        return None, None
    parts = t_str.split("-")
    if len(parts) >= 2:
        return parse_time(parts[0]), parse_time(parts[1])
    return None, None


def parse_answer(raw: str) -> tuple[str, tuple[float, float] | None]:
    """Parse VLM answer to (label, (start, end)|None).
    
    Handles formats:
      - "yes, [1:50-2:01]" (M:SS)
      - "yes, [33s-36s]" (seconds)
      - "yes, [1.5-3.2]" (decimal seconds)
      - "no"
    """
    raw = raw.strip()
    low = raw.lower()
    
    # Try M:SS format first: [M:SS-M:SS] or [MM:SS-MM:SS]
    time_pat_colon = re.findall(r'\[?(\d{1,2}:\d{2}(?:\.\d+)?)\s*-\s*(\d{1,2}:\d{2}(?:\.\d+)?)\]?', raw)
    if time_pat_colon:
        s = parse_time(time_pat_colon[0][0])
        e = parse_time(time_pat_colon[0][1])
        if s is not None and e is not None and e > s:
            trange = (s, e)
            if low.startswith("yes"):
                return "yes", trange
            return "no", None
    
    # Try seconds format: [Xs-Xs] or [X.X-X.X]
    time_pat_sec = re.findall(r'\[?(\d+(?:\.\d+)?)\s*s\s*-\s*(\d+(?:\.\d+)?)\s*s\]?', raw)
    if time_pat_sec:
        try:
            s = float(time_pat_sec[0][0])
            e = float(time_pat_sec[0][1])
            if e > s:
                if low.startswith("yes"):
                    return "yes", (s, e)
        except (ValueError, TypeError):
            pass
    
    # No time range found
    if low.startswith("yes"):
        return "yes", None
    if low.startswith("no"):
        return "no", None
    return "unknown", None


def compute_temporal_iou(pred: tuple[float, float], exp: tuple[float, float],
                         padding_sec: float = 30.0) -> float:
    """Compute intersection over union for two time ranges.
    
    Expands predicted range by ±padding_sec to account for detection granularity.
    """
    if pred is None or exp is None:
        return 0.0
    p_start = max(0.0, pred[0] - padding_sec)
    p_end = pred[1] + padding_sec
    intersection = max(0.0, min(p_end, exp[1]) - max(p_start, exp[0]))
    union = max(p_end, exp[1]) - min(p_start, exp[0])
    return intersection / union if union > 0 else 0.0


def _encode_crop_b64(crop) -> str | None:
    """Encode a BGR crop to base64 JPEG."""
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return base64.b64encode(buf).decode("ascii")


def _bbox_area(ev: dict) -> float:
    bbox = ev.get("start_bbox_xyxy") or ev.get("end_bbox_xyxy")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return 0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x2 - x1) * (y2 - y1)


def sample_frames(video_path: str, num_frames: int = 10) -> list[str]:
    """Sample evenly-spaced frames from a video, return base64 strings."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total_frames <= 0:
        cap.release()
        return []
    
    out: list[str] = []
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        encoded = _encode_crop_b64(frame)
        if encoded:
            timestamp = int(idx / fps)
            tm = f"{timestamp // 60}:{timestamp % 60:02d}"
            out.append((encoded, idx, tm))
    cap.release()
    return out


def build_context(pipeline_data: dict, video_id: str, question: str,
                  refined_data: dict | None = None,
                  appearance_data: dict | None = None) -> str:
    """Build text context from pipeline events for the VLM."""
    cam_match = re.search(r'(G\d{3})', video_id)
    if not cam_match:
        return ""
    cam_id = cam_match.group(1)
    
    per_cam = {
        cr["camera_id"]: cr
        for cr in pipeline_data.get("per_camera", [])
    }
    cam_data = per_cam.get(cam_id, {})
    events = cam_data.get("events", [])
    
    mentioned_cams = set(re.findall(r'G\d{3}', question.upper())) - {cam_id}
    
    lines = []
    lines.append(f"Camera: {cam_id}")
    lines.append(f"Total person tracks: {len([e for e in events if e.get('object_class', 'person') == 'person'])}")
    
    # ── Section 1: Appearance Descriptions (most important) ──
    appearance_lines = []
    if appearance_data:
        for cam_name, cam_ref in appearance_data.get("per_camera", {}).items():
            if cam_name == cam_id:
                for re_ev in cam_ref.get("events", []):
                    color = re_ev.get("object_color", "")
                    notes = re_ev.get("appearance_notes", "")
                    entity = re_ev.get("entity_hint", "")
                    t_start = re_ev.get("start_time", 0)
                    t_end = re_ev.get("end_time", 0)
                    t_str = f"{int(t_start//60)}:{int(t_start%60):02d}-{int(t_end//60)}:{int(t_end%60):02d}"
                    if color or notes:
                        appearance_lines.append(
                            f"  [{t_str}] {entity}: {color} — {notes}"
                        )
    
    if appearance_lines:
        lines.append("")
        lines.append("=== Appearance Descriptions (from pipeline) ===")
        lines.extend(appearance_lines)
    
    # Also include appearances of mentioned cameras
    if mentioned_cams and appearance_data:
        for cam_name, cam_ref in appearance_data.get("per_camera", {}).items():
            if cam_name in mentioned_cams:
                lines.append(f"  -- Camera {cam_name} --")
                for re_ev in cam_ref.get("events", []):
                    color = re_ev.get("object_color", "")
                    notes = re_ev.get("appearance_notes", "")
                    entity = re_ev.get("entity_hint", "")
                    t_start = re_ev.get("start_time", 0)
                    t_end = re_ev.get("end_time", 0)
                    t_str = f"{int(t_start//60)}:{int(t_start%60):02d}-{int(t_end//60)}:{int(t_end%60):02d}"
                    if color or notes:
                        lines.append(f"    [{t_str}] {entity}: {color} — {notes}")
    
    # ── Section 2: Cross-camera Entity Matches ──
    global_entities = pipeline_data.get("global_entities", [])
    if mentioned_cams and global_entities:
        lines.append("")
        lines.append("=== Cross-camera Entity Matches ===")
        for ge in global_entities:
            ge_cams = set(a["camera_id"] for a in ge["appearances"])
            if cam_id in ge_cams and any(mc in ge_cams for mc in mentioned_cams):
                apps = [(a["camera_id"], a["start_time"], a["end_time"])
                        for a in ge["appearances"]]
                apps_str = ", ".join(
                    f"{c}={int(s//60)}:{int(s%60):02d}-{int(e//60)}:{int(e%60):02d}"
                    for c, s, e in sorted(apps)
                )
                lines.append(f"  {ge['global_entity_id']}: appears in {apps_str}")
    
    # ── Section 3: Event Timeline (track-level, for time hints) ──
    person_events = [e for e in events if e.get("object_class", "person") == "person"]
    if person_events:
        lines.append("")
        lines.append("=== Event Timeline ===")
        for e in person_events[:12]:
            tid = e.get("track_id", "?")
            st = e.get("start_time", 0)
            et = e.get("end_time", 0)
            t_str = f"{int(st//60)}:{int(st%60):02d}-{int(et//60)}:{int(et%60):02d}"
            text = e.get("event_text", "")
            if refined_data:
                for r_cam_name, r_cam_ref in refined_data.get("per_camera", {}).items():
                    if r_cam_name == cam_id:
                        for re_ev in r_cam_ref.get("events", []):
                            if (abs(float(re_ev.get("start_time", 0)) - float(st)) < 0.1 and
                                str(re_ev.get("track_id")) == str(tid)):
                                text = re_ev.get("event_text", text)
                                break
            if text:
                lines.append(f"  [{t_str}] track={tid}: {text}")
            else:
                lines.append(f"  [{t_str}] track={tid}")
    
    return "\n".join(lines)


def call_vlm(client: OpenAI, frames_data: list, question: str, context: str) -> dict:
    """Single VLM call: frames + context + question → answer."""
    user_content: list[dict] = []
    
    # Add the context first
    full_prompt = f"[Tracker Context]\n{context}\n\n[Question]\n{question}\n\nAnswer with format: 'yes, [time-time]' or 'no'"
    if context:
        user_content.append({"type": "text", "text": full_prompt})
    else:
        user_content.append({"type": "text", "text": question})
    
    # Add frames as images
    for encoded, idx, tm in frames_data:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        })
    
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    elapsed = time.time() - t0
    raw = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        "total_tokens": resp.usage.total_tokens if resp.usage else 0,
    }
    return {"raw": raw, "usage": usage, "elapsed_sec": round(elapsed, 2)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("ERROR: DASHSCOPE_API_KEY not set in .env")
        sys.exit(1)
    
    # Load questions
    with open(SAMPLED_JSON, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {SAMPLED_JSON}")
    
    # Load pipeline cache
    if not PIPELINE_CACHE.exists():
        print(f"ERROR: Pipeline cache not found at {PIPELINE_CACHE}")
        print("Run test_mevid_full.py first to generate cache")
        sys.exit(1)
    with open(PIPELINE_CACHE, encoding="utf-8") as f:
        pipeline_data = json.load(f)
    print(f"Loaded pipeline cache: {len(pipeline_data.get('global_entities', []))} entities, "
          f"{len(pipeline_data.get('merged_events', []))} merged events")
    
    # Load refined cache if available
    refined_data = None
    if REFINED_CACHE.exists():
        with open(REFINED_CACHE, encoding="utf-8") as f:
            refined_data = json.load(f)
        print(f"Loaded refined cache")
    
    # Load appearance cache if available
    appearance_data = None
    if APPEARANCE_CACHE.exists():
        with open(APPEARANCE_CACHE, encoding="utf-8") as f:
            appearance_data = json.load(f)
        print(f"Loaded appearance cache")
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    results: list[dict] = []
    total_prompt = total_completion = 0
    total_cost = 0.0
    correct = wrong = skipped = 0
    
    # IoU metrics
    iou_values: list[float] = []
    iou_correct = 0  # answer correct AND IoU >= 0.3
    IOU_THRESHOLD = 0.3
    
    print("\n" + "=" * 60)
    print("Custom MEVID Evaluation")
    print("=" * 60)
    
    for i, q in enumerate(questions):
        video_id = q["video_id"]
        question = q["question"]
        expected = q["expected"]
        exp_time_str = q.get("expected_time", "")
        category = q["category"]
        difficulty = q["difficulty"]
        exp_t_start, exp_t_end = parse_time_range(exp_time_str)
        
        case_id = f"CUSTOM_{i+1:04d}"
        
        # Find video file
        video_path = DATA_DIR / f"{video_id}.avi"
        if not video_path.exists():
            print(f"{case_id}: SKIP - video not found: {video_path}")
            results.append({
                "case_id": case_id, "video_id": video_id,
                "question": question, "expected": expected,
                "predicted": "skip", "correct": False,
                "category": category, "difficulty": difficulty,
                "error": "video_not_found",
            })
            skipped += 1
            continue
        
        # Sample frames
        frames_data = sample_frames(str(video_path), num_frames=10)
        if not frames_data:
            print(f"{case_id}: SKIP - no frames extracted")
            skipped += 1
            continue
        
        # Build context
        context = build_context(pipeline_data, video_id, question,
                                refined_data, appearance_data)
        
        print(f"\n[{case_id}] ({category}/{difficulty})")
        print(f"  Q: {question[:100]}")
        print(f"  Expected: {expected}, time: {exp_time_str}")
        print(f"  Context: {len(context.splitlines())} lines")
        
        # Call VLM
        try:
            vlm_resp = call_vlm(client, frames_data, question, context)
        except Exception as e:
            print(f"  ✗ API error: {e}")
            results.append({
                "case_id": case_id, "video_id": video_id,
                "question": question, "expected": expected,
                "predicted": "error", "correct": False,
                "category": category, "difficulty": difficulty,
                "error": str(e),
            })
            skipped += 1
            continue
        
        # Parse answer
        predicted_label, predicted_range = parse_answer(vlm_resp["raw"])
        is_correct = predicted_label == expected
        
        # Compute temporal IoU
        temporal_iou = 0.0
        if expected == "yes" and predicted_label == "yes" and predicted_range and exp_t_start and exp_t_end:
            temporal_iou = compute_temporal_iou(predicted_range, (exp_t_start, exp_t_end))
        iou_values.append(temporal_iou)
        
        iou_pass = is_correct and (temporal_iou >= IOU_THRESHOLD if expected == "yes" else True)
        if iou_pass and expected == "yes":
            iou_correct += 1
        
        mark = "✓" if is_correct else "✗"
        print(f"  {mark} predicted={predicted_label} (expected={expected})")
        print(f"  raw: {vlm_resp['raw'][:80]}")
        if predicted_range:
            print(f"  pred_range: {predicted_range[0]:.0f}s-{predicted_range[1]:.0f}s")
        if temporal_iou > 0:
            print(f"  temporal IoU: {temporal_iou:.3f}")
        
        if is_correct:
            correct += 1
        else:
            wrong += 1
        
        # Token usage
        usage = vlm_resp["usage"]
        total_prompt += usage["prompt_tokens"]
        total_completion += usage["completion_tokens"]
        cost = (usage["prompt_tokens"] * PRICE_IN + usage["completion_tokens"] * PRICE_OUT) / 1000
        total_cost += cost
        
        results.append({
            "case_id": case_id,
            "video_id": video_id,
            "question": question,
            "expected": expected,
            "expected_time": exp_time_str,
            "predicted": predicted_label,
            "predicted_range": list(predicted_range) if predicted_range else None,
            "temporal_iou": round(temporal_iou, 4),
            "iout_pass": iou_pass,
            "correct": is_correct,
            "category": category,
            "difficulty": difficulty,
            "raw_answer": vlm_resp["raw"],
            "usage": usage,
            "cost_cny": round(cost, 5),
            "elapsed_sec": vlm_resp["elapsed_sec"],
        })
    
    # ── Summary ────────────────────────────────────────────────────────────────
    n_eval = correct + wrong
    accuracy = correct / n_eval if n_eval > 0 else 0
    
    # Per-category
    by_cat: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r["predicted"] == "skip" or r["predicted"] == "error":
            continue
        cat = r["category"]
        by_cat[cat]["total"] += 1
        if r["correct"]:
            by_cat[cat]["correct"] += 1
    
    # IoU stats
    yes_iou_values = [v for v in iou_values if v > 0]
    mean_iou = sum(yes_iou_values) / len(yes_iou_values) if yes_iou_values else 0
    yes_questions = sum(1 for q in questions if q["expected"] == "yes")
    iou_pass_rate = iou_correct / yes_questions if yes_questions > 0 else 0
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for cat in sorted(by_cat):
        d = by_cat[cat]
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0
        print(f"  {cat:20s}: {d['correct']:3d}/{d['total']:<3d} ({acc:.1%})")
    
    print(f"\n  Overall accuracy : {correct}/{n_eval} = {accuracy:.1%}")
    print(f"  Skipped          : {skipped}")
    
    # IoU summary
    print(f"\n  Temporal IoU (yes questions only):")
    print(f"    Mean IoU       : {mean_iou:.3f}")
    print(f"    IoU >= {IOU_THRESHOLD} pass : {iou_correct}/{yes_questions} = {iou_pass_rate:.1%}")
    print(f"    Individual IoU : {[round(v, 3) for v in yes_iou_values]}")
    
    print(f"\n  Prompt tokens    : {total_prompt:,}")
    print(f"  Completion tokens: {total_completion:,}")
    print(f"  VLM cost         : ¥{total_cost:.4f} CNY")
    
    # Write results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"custom_eval_{timestamp}.json"
    summary = {
        "timestamp": timestamp,
        "n_questions": len(questions),
        "n_evaluated": n_eval,
        "n_correct": correct,
        "n_wrong": wrong,
        "n_skipped": skipped,
        "accuracy": round(accuracy, 4),
        "per_category": {
            cat: {
                "correct": d["correct"],
                "total": d["total"],
                "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0,
            }
            for cat, d in by_cat.items()
        },
        "temporal_iou": {
            "mean": round(mean_iou, 4),
            "iou_threshold": IOU_THRESHOLD,
            "iou_pass": iou_correct,
            "iou_pass_rate": round(iou_pass_rate, 4),
            "values": [round(v, 4) for v in yes_iou_values],
        },
        "token_usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        },
        "cost_cny": round(total_cost, 4),
        "cases": results,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Result saved → {result_path}")


if __name__ == "__main__":
    main()
