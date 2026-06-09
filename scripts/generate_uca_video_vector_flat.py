"""Generate UCA agent vector-flat seeds from video-pipeline outputs.

This bridges the UCA video evaluation harness into the agent/RAG harness:

    test_uca_unified.py result JSON
    -> *_events_vector_flat.json
    -> SQLite + Chroma
    -> agent e2e evaluation

When a video result JSON is supplied, predicted dense-caption events are used
as the primary searchable records. If no result JSON is supplied, the script can
fall back to cached YOLO event JSONs under ``_pipeline_output/uca_eval``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_UCA_GT = ROOT / "_data" / "Surveillance-Video-Understanding" / "UCF Annotation" / "json" / "UCFCrime_Test.json"
DEFAULT_PIPELINE_DIR = ROOT / "_pipeline_output" / "uca_eval"
DEFAULT_OUT_DIR = ROOT / "agent" / "test" / "generated" / "ucfcrime_video_events_vector_flat"

from video.indexing.search_enrichment import enrich_event_for_search


OBJECT_RULES = [
    ("police", ["police", "officer", "uniform"]),
    ("car", ["car", "vehicle", "truck", "suv", "van"]),
    ("person", ["man", "woman", "people", "person", "adult", "lady", "customer", "staff"]),
    ("child", ["baby", "child", "kid"]),
    ("dog", ["dog", "puppy"]),
    ("bicycle", ["bicycle", "bike", "tricycle"]),
]

ZONE_RULES = [
    ("road", ["road", "street", "intersection", "sidewalk"]),
    ("room", ["room", "wall", "door", "sofa", "table", "bed"]),
    ("store", ["store", "supermarket", "counter", "shelf", "shop"]),
    ("yard", ["yard", "parking", "driveway"]),
]

COLORS = [
    "black", "white", "blue", "red", "green", "yellow", "grey", "gray",
    "silver", "pink", "purple", "brown", "beige", "orange", "dark", "light",
]

STOPWORDS = {
    "the", "and", "with", "from", "into", "then", "there", "this", "that",
    "were", "was", "are", "for", "while", "after", "before", "video",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _clean_sentence(text: Any) -> str:
    out = str(text or "").replace("##", " ").strip()
    out = re.sub(r"\s+", " ", out)
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _infer_object_type(text: str) -> str:
    lower = text.lower()
    for label, needles in OBJECT_RULES:
        if any(token in lower for token in needles):
            return label
    return "unknown"


def _infer_color(text: str) -> str:
    lower = text.lower().replace("-", " ")
    for phrase, color in [
        ("light grey", "light_grey"),
        ("light gray", "light_grey"),
        ("dark grey", "dark_grey"),
        ("dark gray", "dark_grey"),
        ("silver grey", "silver_grey"),
        ("silver gray", "silver_grey"),
    ]:
        if phrase in lower:
            return color
    for color in COLORS:
        if re.search(rf"\b{re.escape(color)}\b", lower):
            return "grey" if color == "gray" else color
    return "unknown"


def _infer_scene_zone(text: str) -> str:
    lower = text.lower()
    for label, needles in ZONE_RULES:
        if any(token in lower for token in needles):
            return label
    return "unknown"


def _keywords(text: str, max_items: int = 10) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in re.findall(r"\b[a-z][a-z0-9_]{2,}\b", text.lower().replace("-", "_")):
        if token in STOPWORDS or token in seen:
            continue
        out.append(token)
        seen.add(token)
        if len(out) >= max_items:
            break
    return out


def _duration_lookup(transcript_json: Path) -> dict[str, float]:
    if not transcript_json.exists():
        return {}
    payload = _load_json(transcript_json)
    if not isinstance(payload, dict):
        return {}
    return {
        str(video_id): float(item.get("duration") or 0.0)
        for video_id, item in payload.items()
        if isinstance(item, dict)
    }


def _event_from_prediction(video_id: str, duration: float, idx: int, time_pair: Any, sentence: Any) -> dict[str, Any] | None:
    if not (isinstance(time_pair, (list, tuple)) and len(time_pair) >= 2):
        return None
    start = _safe_float(time_pair[0])
    end = _safe_float(time_pair[1])
    if start is None or end is None or end <= start:
        return None
    text = _clean_sentence(sentence)
    if not text:
        return None
    event_text = f"From {start:.1f}s to {end:.1f}s, {text[0].lower() + text[1:] if len(text) > 1 else text.lower()}"
    event = {
        "video_id": video_id,
        "clip_start_sec": 0.0,
        "clip_end_sec": duration,
        "start_time": start,
        "end_time": end,
        "object_type": _infer_object_type(text),
        "object_color": _infer_color(text),
        "appearance_notes": f"Video-predicted event; {text}",
        "scene_zone": _infer_scene_zone(text),
        "event_text": event_text,
        "keywords": _keywords(text),
        "start_bbox_xyxy": [None, None, None, None],
        "end_bbox_xyxy": [None, None, None, None],
        "entity_hint": f"video_pred_segment_{idx}",
        "source": "uca_video_prediction",
    }
    return enrich_event_for_search(event)


def _event_from_yolo(video_id: str, duration: float, idx: int, raw: dict[str, Any]) -> dict[str, Any] | None:
    start = _safe_float(raw.get("start_time"))
    end = _safe_float(raw.get("end_time"))
    if start is None or end is None or end <= start:
        return None
    obj = str(raw.get("class_name") or raw.get("object_type") or "unknown")
    track_id = raw.get("track_id", idx)
    desc = str(raw.get("description_for_llm") or raw.get("motion_description") or "").strip()
    text = desc or f"{obj} track {track_id} is active in the scene."
    event_text = f"From {start:.1f}s to {end:.1f}s, {text}"
    event = {
        "video_id": video_id,
        "clip_start_sec": 0.0,
        "clip_end_sec": duration,
        "start_time": start,
        "end_time": end,
        "object_type": obj,
        "object_color": "unknown",
        "appearance_notes": f"YOLO motion evidence; {text}",
        "scene_zone": "unknown",
        "event_text": event_text,
        "keywords": _keywords(f"{obj} {text}"),
        "start_bbox_xyxy": raw.get("start_bbox_xyxy") or [None, None, None, None],
        "end_bbox_xyxy": raw.get("end_bbox_xyxy") or [None, None, None, None],
        "entity_hint": f"track_{track_id}",
        "track_id": track_id,
        "source": "uca_yolo_cache",
    }
    return enrich_event_for_search(event)


def _load_video_result_events(result_json: Path, durations: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    payload = _load_json(result_json)
    out: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("per_video", []) if isinstance(payload, dict) else []:
        if not isinstance(row, dict) or row.get("error"):
            continue
        video_id = str(row.get("video") or "").strip()
        pred = row.get("pred") or {}
        timestamps = pred.get("timestamps") or []
        sentences = pred.get("sentences") or []
        duration = float(row.get("duration_video") or durations.get(video_id) or 0.0)
        events: list[dict[str, Any]] = []
        for idx, (time_pair, sentence) in enumerate(zip(timestamps, sentences), start=1):
            event = _event_from_prediction(video_id, duration, idx, time_pair, sentence)
            if event:
                events.append(event)
        if events:
            out[video_id] = events
    return out


def _load_yolo_events(pipeline_dir: Path, durations: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(pipeline_dir.glob("*_events.json")):
        video_id = path.name.removesuffix("_events.json")
        payload = _load_json(path)
        duration = float((payload.get("meta") or {}).get("duration") or durations.get(video_id) or 0.0)
        events: list[dict[str, Any]] = []
        for idx, raw in enumerate(payload.get("events", []), start=1):
            if not isinstance(raw, dict):
                continue
            event = _event_from_yolo(video_id, duration, idx, raw)
            if event:
                events.append(event)
        if events:
            out[video_id] = events
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UCA video-derived vector-flat seeds")
    parser.add_argument("--video-result-json", default="", help="Result JSON from tests/test_uca_unified.py")
    parser.add_argument("--transcript-json", default=str(DEFAULT_UCA_GT))
    parser.add_argument("--pipeline-dir", default=str(DEFAULT_PIPELINE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--include-yolo", action="store_true", help="Append cached YOLO motion events to predicted VLM events")
    parser.add_argument("--yolo-only", action="store_true", help="Use cached YOLO events even without VLM result JSON")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    durations = _duration_lookup(Path(args.transcript_json).resolve())

    by_video: dict[str, list[dict[str, Any]]] = {}
    source = ""
    if args.video_result_json and not args.yolo_only:
        result_path = Path(args.video_result_json).resolve()
        if not result_path.exists():
            raise FileNotFoundError(f"Video result JSON not found: {result_path}")
        by_video = _load_video_result_events(result_path, durations)
        source = str(result_path)
        print(f"[uca-video-seed] Loaded video predictions for {len(by_video)} videos from {result_path}")

    if args.include_yolo or args.yolo_only or not by_video:
        yolo_by_video = _load_yolo_events(Path(args.pipeline_dir).resolve(), durations)
        print(f"[uca-video-seed] Loaded YOLO cache for {len(yolo_by_video)} videos")
        if args.yolo_only or not by_video:
            by_video = yolo_by_video
            source = str(Path(args.pipeline_dir).resolve())
        else:
            for video_id, events in yolo_by_video.items():
                if video_id in by_video:
                    by_video[video_id].extend(events)

    if not by_video:
        raise RuntimeError("No UCA video events found. Run tests/test_uca_unified.py first or use --yolo-only with cached events.")

    manifest_items: list[dict[str, Any]] = []
    written = skipped = 0
    for video_id, events in sorted(by_video.items()):
        out_path = out_dir / f"{video_id}_events_vector_flat.json"
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        duration = durations.get(video_id) or max((float(e.get("end_time") or 0.0) for e in events), default=0.0)
        payload = {
            "video_id": video_id,
            "duration": duration,
            "events": sorted(events, key=lambda e: float(e.get("start_time") or 0.0)),
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_items.append({
            "video_id": video_id,
            "duration": duration,
            "event_count": len(payload["events"]),
            "output_path": str(out_path),
        })
        written += 1

    manifest = {
        "source": source,
        "output_dir": str(out_dir),
        "video_count": len(by_video),
        "written": written,
        "skipped": skipped,
        "items": manifest_items,
    }
    manifest_path = out_dir.parent / "ucfcrime_video_events_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[uca-video-seed] Done. written={written} skipped={skipped} out={out_dir}")
    print(f"[uca-video-seed] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
