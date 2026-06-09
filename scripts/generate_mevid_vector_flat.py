"""
scripts/generate_mevid_vector_flat.py
--------------------------------------
Converts MEVID pipeline cache → per-video *_events_vector_flat.json seed files
required by ragas_eval_runner.py (Chroma + SQLite).

Run AFTER test_mevid_full.py has populated the pipeline cache, OR let this
script run the pipeline inline with --run-pipeline.

Output: agent/test/data/events_vector_flat/<video_id>_events_vector_flat.json
        (one file per unique video_id across all 6 slots / ~48 files)

Usage
-----
    # Requires pipeline cache to already exist
    python scripts/generate_mevid_vector_flat.py

    # Run pipeline first, then convert (needs videos in _data/mevid_slots/)
    python scripts/generate_mevid_vector_flat.py --run-pipeline --reid-device cuda

    # Single slot
    python scripts/generate_mevid_vector_flat.py --slot 16-35

    # Force overwrite existing output files
    python scripts/generate_mevid_vector_flat.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from video.indexing.search_enrichment import enrich_event_for_search  # noqa: E402
from video.factory.appearance_refinement_runner import (  # noqa: E402
    AppearanceRefinementConfig,
    run_appearance_refinement_for_pipeline,
)
from video.factory.scene_profile_runner import (  # noqa: E402
    SceneProfileConfig,
    run_scene_profiles_for_pipeline,
)

PIPELINE_CACHE_DIR = ROOT / "_cache" / "mevid_pipeline"
OUT_DIR = ROOT / "agent" / "test" / "data" / "events_vector_flat"

# ── Slot → camera → stem ──────────────────────────────────────────────────────
SLOT_CAMERAS: dict[str, dict[str, str]] = {
    "11-20": {
        "G329": "2018-03-11.11-20-00.11-25-00.admin.G329.r13",
        "G330": "2018-03-11.11-20-00.11-25-00.school.G330.r13",
        "G336": "2018-03-11.11-20-00.11-25-00.school.G336.r13",
        "G419": "2018-03-11.11-20-00.11-25-00.school.G419.r13",
        "G420": "2018-03-11.11-20-00.11-25-00.school.G420.r13",
        "G421": "2018-03-11.11-20-00.11-25-00.school.G421.r13",
        "G423": "2018-03-11.11-20-00.11-25-00.school.G423.r13",
        "G508": "2018-03-11.11-20-00.11-25-00.bus.G508.r13",
    },
    "11-55": {
        "G299": "2018-03-11.11-55-00.12-00-00.school.G299.r13",
        "G328": "2018-03-11.11-55-00.12-00-00.school.G328.r13",
        "G330": "2018-03-11.11-55-00.12-00-00.school.G330.r13",
        "G419": "2018-03-11.11-55-00.12-00-00.school.G419.r13",
        "G420": "2018-03-11.11-55-00.12-00-00.school.G420.r13",
        "G506": "2018-03-11.11-55-00.12-00-00.bus.G506.r13",
        "G508": "2018-03-11.11-55-00.12-00-00.bus.G508.r13",
    },
    "13-50": {
        "G328": "2018-03-11.13-50-01.13-55-01.school.G328.r13",
        "G329": "2018-03-11.13-50-01.13-55-01.admin.G329.r13",
        "G339": "2018-03-11.13-50-01.13-55-01.school.G339.r13",
        "G421": "2018-03-11.13-50-01.13-55-01.school.G421.r13",
        "G424": "2018-03-11.13-50-01.13-55-01.school.G424.r13",
        "G506": "2018-03-11.13-50-01.13-55-01.bus.G506.r13",
        "G508": "2018-03-11.13-50-01.13-55-01.bus.G508.r13",
    },
    "14-20": {
        "G328": "2018-03-11.14-20-01.14-25-01.school.G328.r13",
        "G339": "2018-03-11.14-20-01.14-25-01.school.G339.r13",
        "G419": "2018-03-11.14-20-01.14-25-01.school.G419.r13",
        "G421": "2018-03-11.14-20-01.14-25-01.school.G421.r13",
        "G423": "2018-03-11.14-20-01.14-25-01.school.G423.r13",
        "G505": "2018-03-11.14-20-01.14-25-01.bus.G505.r13",
        "G506": "2018-03-11.14-20-01.14-25-01.bus.G506.r13",
        "G508": "2018-03-11.14-20-01.14-25-01.bus.G508.r13",
    },
    "16-20": {
        "G326": "2018-03-11.16-20-01.16-25-01.admin.G326.r13",
        "G328": "2018-03-11.16-20-01.16-25-01.school.G328.r13",
        "G329": "2018-03-11.16-20-01.16-25-01.admin.G329.r13",
        "G336": "2018-03-11.16-20-01.16-25-01.school.G336.r13",
        "G419": "2018-03-11.16-20-01.16-25-01.school.G419.r13",
        "G420": "2018-03-11.16-20-01.16-25-01.school.G420.r13",
        "G506": "2018-03-11.16-20-01.16-25-01.bus.G506.r13",
        "G508": "2018-03-11.16-20-01.16-25-01.bus.G508.r13",
    },
    "16-35": {
        "G326": "2018-03-11.16-35-01.16-40-01.admin.G326.r13",
        "G328": "2018-03-11.16-35-01.16-40-01.school.G328.r13",
        "G329": "2018-03-11.16-35-01.16-40-01.admin.G329.r13",
        "G336": "2018-03-11.16-35-01.16-40-01.school.G336.r13",
        "G339": "2018-03-11.16-35-01.16-40-01.school.G339.r13",
        "G419": "2018-03-11.16-35-01.16-40-01.school.G419.r13",
        "G420": "2018-03-11.16-35-01.16-40-01.school.G420.r13",
        "G423": "2018-03-11.16-35-01.16-40-00.school.G423.r13",
        "G506": "2018-03-11.16-35-01.16-40-01.bus.G506.r13",
        "G638": "2018-03-11.16-35-01.16-40-01.school.G638.r13",
    },
}

# Location token in stem → scene_zone
_LOCATION_ZONE = {"admin": "road", "school": "yard", "bus": "road"}

# Common colors to look for in event text
_COLORS = [
    "red", "blue", "green", "black", "white", "yellow",
    "orange", "grey", "gray", "brown", "purple", "pink",
    "dark", "light",
]

_APPEARANCE_TERMS = [
    "hoodie", "jacket", "coat", "shirt", "t-shirt", "pants", "trousers",
    "jeans", "shorts", "skirt", "dress", "bag", "backpack", "handbag",
    "hat", "cap", "hood", "scarf", "fur", "collar", "long", "sleeve",
]

_ACTION_TERMS = [
    "walking", "standing", "moving", "leaving", "entering", "exit", "exiting",
    "left", "right", "up", "down", "stationary", "visible", "side",
]

_COLOR_ALIASES = {
    "gray": "grey",
    "light gray": "light_grey",
    "light grey": "light_grey",
    "dark gray": "dark_grey",
    "dark grey": "dark_grey",
    "silver gray": "silver_grey",
    "silver grey": "silver_grey",
}

_STOPWORDS = {
    "a", "an", "the", "is", "in", "at", "to", "of", "and", "or",
    "from", "with", "for", "by", "on", "into", "that", "this",
    "was", "were", "has", "have", "had", "be", "been", "are",
    "its", "it", "he", "she", "they", "person", "camera",
    "unknown",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zone_from_stem(stem: str) -> str:
    """Infer scene_zone from location token in video stem."""
    for loc, zone in _LOCATION_ZONE.items():
        if f".{loc}." in stem:
            return zone
    return "unknown"


def _extract_color(text: str) -> str:
    tl = text.lower()
    for phrase, alias in _COLOR_ALIASES.items():
        if phrase in tl:
            return alias
    for c in _COLORS:
        if c in tl:
            return _COLOR_ALIASES.get(c, c)
    return "unknown"


def _extract_keywords(text: str, max_k: int = 6) -> list[str]:
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    seen: dict[str, None] = {}
    for w in words:
        if w not in _STOPWORDS:
            seen[w] = None
    return list(seen)[:max_k]


def _normalize_keywords(value: object, fallback_text: str = "", max_k: int = 6) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            token = str(item).strip().lower().replace(" ", "_")
            if token and token not in seen:
                out.append(token)
                seen.add(token)
        if out:
            return out[:max_k]
    if isinstance(value, str) and value.strip():
        out = []
        seen = set()
        for token in re.split(r"[,;/\s]+", value.strip().lower()):
            token = token.strip().replace(" ", "_")
            if token and token not in seen and token not in _STOPWORDS:
                out.append(token)
                seen.add(token)
        if out:
            return out[:max_k]
    return _extract_keywords(fallback_text, max_k=max_k)


def _is_vehicle_class(value: object) -> bool:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return raw in {"car", "vehicle", "bus", "truck", "motorcycle", "bicycle"}


def _appearance_keywords(text: str) -> list[str]:
    tl = text.lower().replace("-", " ")
    out: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.strip().lower().replace(" ", "_")
        if token and token != "unknown" and token not in seen:
            out.append(token)
            seen.add(token)

    for phrase, alias in _COLOR_ALIASES.items():
        if phrase in tl:
            add(alias)
    for c in _COLORS:
        if re.search(rf"\b{re.escape(c)}\b", tl):
            add(_COLOR_ALIASES.get(c, c))
    for term in _APPEARANCE_TERMS:
        term_re = term.replace("-", r"[-\s]?")
        if re.search(rf"\b{term_re}\b", tl):
            add(term.replace("-", "_"))

    # Useful compounds for natural-language clothing queries.
    if ("light grey" in tl or "light gray" in tl) and "hoodie" in tl:
        add("light_grey_hoodie")
    if ("dark" in tl or "black" in tl) and "coat" in tl:
        add("dark_coat")
    if ("black" in tl or "dark" in tl) and "hood" in tl:
        add("hood_up")
    if "fur" in tl and ("hood" in tl or "collar" in tl):
        add("fur_trimmed_hood")
    for term in _ACTION_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", tl):
            add(term)
    return out


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _is_unknown_text(value: object) -> bool:
    txt = _clean_text(value).lower()
    return txt in {"", "unknown", "n/a", "none", "null"}


def _append_unique(items: list[str], extra: list[str], max_len: int = 12) -> list[str]:
    seen = set(items)
    for token in extra:
        token = str(token).strip().lower().replace(" ", "_")
        if token and token != "unknown" and token not in seen:
            items.append(token)
            seen.add(token)
        if len(items) >= max_len:
            break
    return items


def _append_sentence(base: str, sentence: str) -> str:
    base = str(base or "").strip()
    sentence = str(sentence or "").strip()
    if not sentence:
        return base
    if sentence.lower().rstrip(".") in base.lower():
        return base
    if not base:
        return sentence
    return f"{base.rstrip()} {sentence}"


def _append_appearance_text(
    event_text: str,
    appearance: str,
    color: str,
    action_summary: str = "",
    movement_direction: str = "",
    exit_side: str = "",
) -> str:
    text = str(event_text or "").strip()
    if appearance and not _is_unknown_text(appearance):
        text = _append_sentence(text, f"Appearance: {appearance.rstrip('.')}.")
    if color and color.lower() != "unknown":
        readable_color = color.replace("_", " ")
        if readable_color.lower() not in text.lower():
            text = _append_sentence(text, f"Color: {readable_color}.")
    if action_summary and action_summary.lower() != "unknown":
        text = _append_sentence(text, f"Action: {action_summary.rstrip('.')}.")
    if movement_direction and movement_direction.lower() not in {"unknown", "stationary"}:
        text = _append_sentence(text, f"Movement direction: {movement_direction.lower()}.")
    if exit_side and exit_side.lower() not in {"unknown", "stationary"}:
        text = _append_sentence(text, f"Exit side: {exit_side.lower()}.")
    return text


def _build_cross_camera_descriptors(apps: list[dict]) -> tuple[str, str, str]:
    ordered = sorted(
        apps,
        key=lambda a: float(a.get("start_time", 0.0)),
    )
    chain = " -> ".join(
        f"{a.get('camera_id')} track#{a.get('track_id')} "
        f"{_fmt_sec(float(a.get('start_time', 0.0)))}-"
        f"{_fmt_sec(float(a.get('end_time', 0.0)))}"
        for a in ordered
    )
    cameras = [str(a.get("camera_id") or "").strip() for a in ordered if str(a.get("camera_id") or "").strip()]
    dedup_cameras = list(dict.fromkeys(cameras))
    cams = ", ".join(dedup_cameras)
    cam_sequence = " -> ".join(dedup_cameras)
    return chain, cams, cam_sequence


def _build_multi_camera_event_text(
    *,
    cls: str,
    camera_id: str,
    track_id: object,
    start_sec: float,
    end_sec: float,
    entity_id: str,
    appearance: str,
    color: str,
    action_summary: str,
    movement_direction: str,
    exit_side: str,
    refined_text: str,
    chain: str,
    cam_sequence: str,
) -> str:
    parts = [
        f"Cross-camera {cls} event in camera {camera_id} (track #{track_id}, {_fmt_sec(start_sec)}-{_fmt_sec(end_sec)})."
    ]
    if entity_id:
        parts.append(f"Entity: {entity_id}.")
    if appearance and not _is_unknown_text(appearance):
        parts.append(f"Appearance: {appearance.rstrip('.')}.")
    elif color and color.lower() != "unknown":
        parts.append(f"Color: {color.replace('_', ' ')}.")
    if action_summary and not _is_unknown_text(action_summary):
        parts.append(f"Action: {action_summary.rstrip('.')}.")
    if movement_direction and movement_direction.lower() not in {"unknown", "stationary"}:
        parts.append(f"Movement direction: {movement_direction.lower()}.")
    if exit_side and exit_side.lower() not in {"unknown", "stationary"}:
        parts.append(f"Exit side: {exit_side.lower()}.")
    if cam_sequence:
        parts.append(f"Camera sequence: {cam_sequence}.")
    if chain:
        parts.append(f"Trajectory: {chain}.")
    elif refined_text and not _is_unknown_text(refined_text):
        parts.append(f"Observed: {refined_text.rstrip('.')}.")
    return " ".join(parts)


def _build_multi_camera_appearance_notes(
    *,
    cls: str,
    camera_id: str,
    start_sec: float,
    end_sec: float,
    entity_id: str,
    appearance: str,
    color: str,
    action_summary: str,
    movement_direction: str,
    cams: str,
) -> str:
    parts: list[str] = []
    if appearance and not _is_unknown_text(appearance):
        parts.append(appearance.rstrip("."))
    else:
        subject_parts: list[str] = []
        if color and color.lower() != "unknown":
            subject_parts.append(color.replace("_", " "))
        if cls:
            subject_parts.append(cls)
        if subject_parts:
            parts.append(" ".join(subject_parts))
    if action_summary and not _is_unknown_text(action_summary):
        parts.append(f"action {action_summary.rstrip('.')}")
    if movement_direction and movement_direction.lower() not in {"unknown", "stationary"}:
        parts.append(f"moving {movement_direction.lower()}")
    if entity_id:
        parts.append(f"entity {entity_id}")
    if cams:
        parts.append(f"across {cams}")
    parts.append(f"local {camera_id} {_fmt_sec(start_sec)}-{_fmt_sec(end_sec)}")
    return "; ".join(part for part in parts if part)


def _scene_keywords(text: str) -> list[str]:
    words = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", str(text or "").lower().replace("-", " "))
    out: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in _STOPWORDS:
            continue
        token = word.replace(" ", "_")
        if token not in seen:
            out.append(token)
            seen.add(token)
        if len(out) >= 8:
            break
    return out


def _bbox_center_from_event(ev: dict) -> tuple[float, float] | None:
    box = ev.get("start_bbox_xyxy") or ev.get("end_bbox_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
    except Exception:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _extract_bbox(value: object) -> list[float | None] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    out: list[float | None] = []
    has_value = False
    for item in value[:4]:
        if item is None:
            out.append(None)
            continue
        try:
            out.append(float(item))
            has_value = True
        except Exception:
            out.append(None)
    return out if has_value else None


def _bbox_or_nulls(*values: object) -> list[float | None]:
    for value in values:
        bbox = _extract_bbox(value)
        if bbox is not None:
            return bbox
    return [None, None, None, None]


def _scene_context_for_event(ev: dict, scene_profile: dict | None) -> tuple[str, list[str]]:
    if not scene_profile:
        return "", []
    zones = scene_profile.get("zones") if isinstance(scene_profile.get("zones"), dict) else {}
    summary = _clean_text(scene_profile.get("scene_summary"))
    width = float(scene_profile.get("frame_width") or 0.0)
    height = float(scene_profile.get("frame_height") or 0.0)
    center = _bbox_center_from_event(ev)

    zone_labels: list[str] = []
    if center and width > 0 and height > 0:
        x, y = center
        if x < width / 3.0:
            zone_labels.append("left")
        elif x > width * 2.0 / 3.0:
            zone_labels.append("right")
        else:
            zone_labels.append("center")
        if y < height / 3.0:
            zone_labels.append("top")
        elif y > height * 2.0 / 3.0:
            zone_labels.append("bottom")

    zone_texts = []
    for label in dict.fromkeys(zone_labels):
        text = _clean_text(zones.get(label))
        if text:
            zone_texts.append(f"{label}: {text}")

    parts = []
    if summary:
        parts.append(f"Scene: {summary}.")
    if zone_texts:
        parts.append(f"Local zone: {'; '.join(zone_texts)}.")
    context = " ".join(parts)
    keywords = list(scene_profile.get("keywords") or [])
    keywords = _append_unique(keywords, zone_labels, max_len=16)
    keywords = _append_unique(keywords, _scene_keywords(context), max_len=16)
    return context, keywords


def _event_key_from_time(ev: dict) -> tuple[float, float, str]:
    return (
        round(float(ev.get("start_time", 0.0)), 1),
        round(float(ev.get("end_time", 0.0)), 1),
        str(ev.get("object_type") or ev.get("class_name") or "").lower(),
    )


def _merge_refined_events(existing: dict | None, incoming: dict) -> dict:
    """Merge clip-level and crop-appearance refinements for the same track/event."""
    if not existing:
        return dict(incoming)

    merged = dict(existing)
    incoming_text = _clean_text(incoming.get("event_text") or incoming.get("description"))
    existing_text = _clean_text(merged.get("event_text") or merged.get("description"))

    incoming_appearance = _clean_text(incoming.get("appearance_notes"))
    incoming_color = _clean_text(incoming.get("object_color"))
    incoming_action = _clean_text(incoming.get("action_summary"))
    incoming_direction = _clean_text(incoming.get("movement_direction"))
    incoming_exit_side = _clean_text(incoming.get("exit_side"))
    if incoming_color.lower() == "unknown":
        incoming_color = ""

    if incoming_appearance and not _is_unknown_text(incoming_appearance):
        merged["appearance_notes"] = incoming_appearance
        merged["event_text"] = _append_appearance_text(
            existing_text or incoming_text,
            incoming_appearance,
            incoming_color,
            incoming_action,
            incoming_direction,
            incoming_exit_side,
        )
    elif incoming_text and not existing_text:
        merged["event_text"] = incoming_text

    if incoming_color:
        merged["object_color"] = incoming_color
    if incoming_action and incoming_action.lower() != "unknown":
        merged["action_summary"] = incoming_action
    if incoming_direction and incoming_direction.lower() != "unknown":
        merged["movement_direction"] = incoming_direction
    if incoming_exit_side and incoming_exit_side.lower() != "unknown":
        merged["exit_side"] = incoming_exit_side

    merged["keywords"] = _append_unique(
        _normalize_keywords(merged.get("keywords"), existing_text, max_k=16),
        _append_unique(
            _normalize_keywords(incoming.get("keywords"), incoming_text, max_k=16),
            _appearance_keywords(" ".join([incoming_action, incoming_direction, incoming_exit_side])),
            max_len=18,
        ),
        max_len=18,
    )

    for field in ("entity_hint", "video_id", "camera_id", "track_id", "clip_start_sec", "clip_end_sec"):
        if not merged.get(field) and incoming.get(field):
            merged[field] = incoming[field]
    for field in ("start_bbox_xyxy", "end_bbox_xyxy"):
        if _extract_bbox(merged.get(field)) is None and _extract_bbox(incoming.get(field)) is not None:
            merged[field] = list(_extract_bbox(incoming.get(field)) or [])
    return merged


def _fmt_sec(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


def _get_video_duration(video_path: Path) -> float:
    """Get duration via OpenCV; fall back to 300.0 (5-min MEVID clip)."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0 and fps > 0:
            return n / fps
    except Exception:
        pass
    return 300.0  # default 5-minute MEVID clip


def _filter_seed_events(events: list[dict[str, Any]], min_duration_sec: float) -> tuple[list[dict[str, Any]], int]:
    if min_duration_sec <= 0:
        return list(events), 0
    kept = [
        event
        for event in events
        if float(event.get("end_time", event.get("start_time", 0.0)))
        - float(event.get("start_time", 0.0))
        >= min_duration_sec
    ]
    return kept, len(events) - len(kept)


# ── Core conversion ───────────────────────────────────────────────────────────

def pipeline_events_to_vector_flat(
    video_id: str,
    camera_id: str,
    events: list[dict],
    global_entities: list[dict],
    duration: float,
    refined_events: list[dict] | None = None,
    scene_profile: dict | None = None,
    seed_mode: str = "multi_camera",
) -> dict:
    """Convert pipeline CameraResult events → vector_flat dict."""
    single_camera_mode = seed_mode == "single_camera"

    def normalize_search_object_type(value: object) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"person", "people", "pedestrian"}:
            return "person"
        if _is_vehicle_class(raw):
            return "car"
        return raw or "person"

    def normalize_search_keywords(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values:
            token = str(item or "").strip().lower().replace(" ", "_").replace("-", "_")
            if not token or token == "unknown":
                continue
            if token in {"people", "pedestrian"}:
                token = "person"
            elif token in {"vehicle", "bus", "truck", "motorcycle", "bicycle"}:
                token = "car"
            if token not in seen:
                out.append(token)
                seen.add(token)
        return out

    # Build track_id → global_entity_id lookup plus entity trajectory text for
    # cross-camera RAG. The agent needs this even when LLM refinement is disabled.
    track_to_entity: dict[int, str] = {}
    entity_to_apps: dict[str, list[dict]] = {}
    for ge in global_entities:
        entity_id = str(ge.get("global_entity_id") or "")
        if entity_id:
            entity_to_apps[entity_id] = list(ge.get("appearances", []))
        for app in ge.get("appearances", []):
            if app.get("camera_id") == camera_id:
                tid = app.get("track_id")
                if tid is not None:
                    track_to_entity[int(tid)] = ge["global_entity_id"]

    # Build refined event lookup. New refined outputs include track_id; older caches may
    # not, so keep a time/class fallback for compatibility.
    refined_by_track: dict[int, dict] = {}
    refined_by_time: dict[tuple[float, float, str], dict] = {}
    if refined_events:
        for ev in refined_events:
            tid = ev.get("track_id")
            if tid is not None:
                try:
                    tid_int = int(tid)
                    refined_by_track[tid_int] = _merge_refined_events(refined_by_track.get(tid_int), ev)
                except Exception:
                    pass
            key = _event_key_from_time(ev)
            refined_by_time[key] = _merge_refined_events(refined_by_time.get(key), ev)

    scene_zone = _zone_from_stem(video_id)

    flat_events: list[dict] = []
    for idx, ev in enumerate(events, start=1):
        t_start = float(ev.get("start_time", 0.0))
        t_end   = float(ev.get("end_time",   t_start + 1.0))
        cls     = normalize_search_object_type(ev.get("class_name") or ev.get("object_type") or "person")
        tid     = ev.get("track_id", idx)
        entity_id = track_to_entity.get(int(tid) if isinstance(tid, (int, float)) else idx, "")

        refined_event = None
        if isinstance(tid, (int, float)):
            refined_event = refined_by_track.get(int(tid))
        if refined_event is None:
            refined_event = refined_by_time.get(_event_key_from_time(ev))

        # Prefer LLM-refined fields; fall back to generated pipeline text.
        refined_text = ""
        refined_color = ""
        refined_appearance = ""
        refined_action = ""
        refined_direction = ""
        refined_exit_side = ""
        refined_keywords: list[str] = []
        refined_entity_hint = ""
        if refined_event:
            refined_text = _clean_text(refined_event.get("event_text") or refined_event.get("description"))
            refined_color = _clean_text(refined_event.get("object_color"))
            if refined_color.lower() == "unknown":
                refined_color = ""
            refined_appearance = _clean_text(refined_event.get("appearance_notes"))
            if _is_unknown_text(refined_appearance):
                refined_appearance = ""
            refined_action = _clean_text(refined_event.get("action_summary"))
            if _is_unknown_text(refined_action):
                refined_action = ""
            refined_direction = _clean_text(refined_event.get("movement_direction"))
            if _is_unknown_text(refined_direction):
                refined_direction = ""
            refined_exit_side = _clean_text(refined_event.get("exit_side"))
            if _is_unknown_text(refined_exit_side):
                refined_exit_side = ""
            refined_keywords = _normalize_keywords(refined_event.get("keywords"), refined_text)
            refined_entity_hint = _clean_text(refined_event.get("entity_hint"))

        refined_text = refined_text or ev.get("event_text", "")

        apps_for_entity: list[dict] = []
        cross_camera_text = ""
        cross_camera_chain = ""
        cross_camera_cams = ""
        cross_camera_sequence = ""
        if entity_id and not single_camera_mode:
            apps_for_entity = list(entity_to_apps.get(entity_id, []))
            if apps_for_entity:
                cross_camera_chain, cross_camera_cams, cross_camera_sequence = _build_cross_camera_descriptors(apps_for_entity)
                cross_camera_text = (
                    f" cross-camera same-person candidate {entity_id}; "
                    f"seen in cameras {cross_camera_cams}; trajectory {cross_camera_chain}."
                )

        if single_camera_mode and cls == "person":
            # Prefer LLM clip-refined text; fall back to generic template.
            if refined_text:
                event_text = refined_text
            else:
                event_text = (
                    f"From {t_start:.1f}s to {t_end:.1f}s, person "
                    f"(track #{tid}) is visible in camera {camera_id} "
                    f"({_fmt_sec(t_start)}-{_fmt_sec(t_end)})."
                )
        elif not single_camera_mode:
            event_text = _build_multi_camera_event_text(
                cls=cls,
                camera_id=camera_id,
                track_id=tid,
                start_sec=t_start,
                end_sec=t_end,
                entity_id=entity_id,
                appearance=refined_appearance,
                color=refined_color,
                action_summary=refined_action,
                movement_direction=refined_direction,
                exit_side=refined_exit_side,
                refined_text=refined_text,
                chain=cross_camera_chain,
                cam_sequence=cross_camera_sequence,
            )
        elif refined_text:
            event_text = refined_text
        else:
            event_text = (
                f"From {t_start:.1f}s to {t_end:.1f}s, {cls} "
                f"(track #{tid}) detected in camera {camera_id} "
                f"({_fmt_sec(t_start)}–{_fmt_sec(t_end)})"
            )
        if cross_camera_text and cross_camera_text not in event_text:
            event_text += cross_camera_text

        scene_context, scene_keywords = _scene_context_for_event(ev, scene_profile)
        if single_camera_mode and scene_context and scene_context not in event_text:
            event_text = _append_sentence(event_text, scene_context)

        if single_camera_mode:
            event_text = _append_appearance_text(
                event_text,
                refined_appearance,
                refined_color,
                refined_action,
                refined_direction,
                refined_exit_side,
            )

        if refined_color:
            color = refined_color
        else:
            color = _extract_color(" ".join([refined_appearance, refined_text]))
        keywords = [
            k for k in (refined_keywords or _extract_keywords(" ".join([event_text, refined_appearance])))
            if str(k).strip().lower() != "unknown"
        ]
        if single_camera_mode:
            blocked = {"cross_camera", "same_person"}
            if entity_id:
                blocked.add(str(entity_id).lower())
            keywords = [
                k for k in keywords
                if str(k).strip().lower() not in blocked
                and not str(k).strip().lower().startswith("person_global_")
            ]
        if camera_id.lower() not in keywords:
            keywords = [camera_id.lower()] + keywords
        keywords = _append_unique(
            keywords,
            _appearance_keywords(" ".join([refined_appearance, refined_text, event_text])),
        )
        keywords = _append_unique(
            keywords,
            _appearance_keywords(" ".join([refined_action, refined_direction, refined_exit_side])),
        )
        if single_camera_mode:
            keywords = _append_unique(keywords, scene_keywords)
        if entity_id and not single_camera_mode:
            for token in ("cross_camera", "same_person", entity_id.lower()):
                if token not in keywords:
                    keywords.append(token)
        keywords = normalize_search_keywords(keywords)

        motion_desc = ev.get("motion_description", "")
        if not single_camera_mode:
            appearance_notes = _build_multi_camera_appearance_notes(
                cls=cls,
                camera_id=camera_id,
                start_sec=t_start,
                end_sec=t_end,
                entity_id=entity_id,
                appearance=refined_appearance,
                color=refined_color or color,
                action_summary=refined_action,
                movement_direction=refined_direction,
                cams=cross_camera_cams,
            )
        elif refined_appearance:
            appearance_notes = refined_appearance
        elif motion_desc:
            appearance_notes = f"High activity; {motion_desc}"
        elif refined_text and not _is_unknown_text(refined_text):
            appearance_notes = f"Observed event; {refined_text}"
        else:
            appearance_notes = (
                f"Observed event; {cls} present from "
                f"{_fmt_sec(t_start)} to {_fmt_sec(t_end)}"
            )
        entity_hint = f"track_{tid}"
        if refined_entity_hint and not single_camera_mode:
            entity_hint = refined_entity_hint
        elif entity_id and not single_camera_mode:
            entity_hint = f"{entity_id}_{camera_id}"

        start_bbox = _bbox_or_nulls(
            refined_event.get("start_bbox_xyxy") if refined_event else None,
            ev.get("start_bbox_xyxy"),
        )
        end_bbox = _bbox_or_nulls(
            refined_event.get("end_bbox_xyxy") if refined_event else None,
            ev.get("end_bbox_xyxy"),
        )

        record = {
            "video_id":       video_id,
            "camera_id":      camera_id,
            "track_id":       tid,
            "entity_hint":    entity_hint,
            "clip_start_sec": 0.0,
            "clip_end_sec":   duration,
            "start_time":     t_start,
            "end_time":       t_end,
            "object_type":    cls,
            "object_color":   color,
            "appearance_notes": appearance_notes,
            "action_summary": refined_action or None,
            "movement_direction": refined_direction or None,
            "exit_side": refined_exit_side or None,
            "scene_profile_context": scene_context or None,
            "scene_zone":     scene_zone,
            "event_text":     event_text,
            "keywords":       keywords,
            "start_bbox_xyxy": start_bbox,
            "end_bbox_xyxy":   end_bbox,
            # cross-camera context
            "global_entity_id": None if single_camera_mode else (entity_id or None),
            "seed_mode": seed_mode,
        }
        flat_events.append(
            enrich_event_for_search(
                record,
                camera_id=camera_id,
                global_entity_id="" if single_camera_mode else entity_id,
                trajectory_text=cross_camera_text.strip(),
            )
        )

    return {
        "video_id": video_id,
        "duration": duration,
        "camera_id": camera_id,
        "events": flat_events,
    }


# ── Inline pipeline runner (used by --run-pipeline) ───────────────────────────

def _env_flag(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def run_and_cache_pipeline(
    slot: str,
    video_dir: Path,
    reid_device: str,
    *,
    precision_matching: bool = True,
) -> dict:
    """Import and run the multi-camera pipeline; save cache."""
    from video.factory.multi_camera_coordinator import run_multi_camera_pipeline
    from video.core.schema.multi_camera import CrossCameraConfig

    cam_map = SLOT_CAMERAS[slot]
    camera_videos = {
        cam: str(video_dir / f"{stem}.avi")
        for cam, stem in cam_map.items()
        if (video_dir / f"{stem}.avi").exists()
    }
    if not camera_videos:
        print(f"  [{slot}] No videos found — skipping")
        return {}

    if precision_matching:
        config = CrossCameraConfig(
            max_transition_sec=180.0,
            embedding_threshold=0.70,
            cross_camera_min_score=0.72,
            person_only=True,
            cross_camera_overlap_tolerance_sec=2.0,
            topology_weight_reid=0.65,
            topology_weight_topo=0.35,
            max_global_entity_cameras=4,
            max_global_entity_tracks=4,
        )
    else:
        config = CrossCameraConfig(
            max_transition_sec=300.0,
            embedding_threshold=0.60,
            cross_camera_min_score=0.60,
            person_only=True,
        )
    mode = "precision" if precision_matching else "standard"
    print(f"  [{slot}] Running pipeline on {list(camera_videos)} ({mode} matching) …")
    t0 = time.time()
    mc = run_multi_camera_pipeline(
        camera_videos=camera_videos,
        config=config,
        reid_device=reid_device,
        num_crops=5,
        model_path="11m",
        conf=0.25,
        iou=0.25,
    )

    # Serialise (skip ndarray fields)
    per_cam = [
        {
            "camera_id": cr.camera_id,
            "events":    cr.events,
            "clips":     cr.clips,
            "meta":      cr.meta,
        }
        for cr in mc.per_camera
    ]
    entities = [
        {
            "global_entity_id": ge.global_entity_id,
            "appearances": [
                {"camera_id": a.camera_id, "track_id": a.track_id,
                 "start_time": a.start_time, "end_time": a.end_time,
                 "confidence": a.confidence}
                for a in ge.appearances
            ],
        }
        for ge in mc.global_entities
    ]
    result = {
        "slot": slot,
        "elapsed_sec": round(time.time() - t0, 1),
        "cameras_run": list(camera_videos),
        "global_entities": entities,
        "per_camera": per_cam,
        "merged_events": mc.merged_events,
    }
    PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = PIPELINE_CACHE_DIR / f"{slot}_pipeline.json"
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [{slot}] Done in {result['elapsed_sec']}s — cached to {cache}")
    return result


def _load_dotenv_if_available() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except Exception:
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _run_or_load_appearance_refinement(
    *,
    slot: str,
    pipeline: dict,
    cam_map: dict[str, str],
    video_dir: Path,
    cache_path: Path,
    force: bool,
    max_entities: int,
    max_apps_per_entity: int,
    crops_per_app: int,
) -> dict:
    camera_to_video = {
        cam_id: video_dir / f"{stem}.avi"
        for cam_id, stem in cam_map.items()
        if (video_dir / f"{stem}.avi").exists()
    }
    camera_video_stems = {
        cam_id: stem
        for cam_id, stem in cam_map.items()
    }
    cfg = AppearanceRefinementConfig.from_env(
        cache_path=cache_path,
        force=force,
        max_entities=max_entities,
        max_apps_per_entity=max_apps_per_entity,
        crops_per_app=crops_per_app,
    )
    return run_appearance_refinement_for_pipeline(
        slot=slot,
        pipeline=pipeline,
        camera_to_video=camera_to_video,
        camera_video_stems=camera_video_stems,
        config=cfg,
    )


def _run_or_load_scene_profiles(
    *,
    slot: str,
    cam_map: dict[str, str],
    video_dir: Path,
    cache_path: Path,
    force: bool,
) -> dict:
    camera_to_video = {
        cam_id: video_dir / f"{stem}.avi"
        for cam_id, stem in cam_map.items()
        if (video_dir / f"{stem}.avi").exists()
    }
    camera_video_stems = {
        cam_id: stem
        for cam_id, stem in cam_map.items()
    }
    cfg = SceneProfileConfig.from_env(cache_path=cache_path, force=force)
    return run_scene_profiles_for_pipeline(
        slot=slot,
        camera_to_video=camera_to_video,
        camera_video_stems=camera_video_stems,
        config=cfg,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate *_events_vector_flat.json seed files from MEVID pipeline cache"
    )
    ap.add_argument("--slot", default="", help="Process one slot only, e.g. 16-35")
    ap.add_argument("--video-dir", default="_data/mevid_slots",
                    help="Directory with downloaded .avi files (needed for duration + --run-pipeline)")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="Output directory for vector flat JSON files")
    ap.add_argument("--min-event-duration-sec", type=float, default=1.0,
                    help="Drop events shorter than this (seconds) before writing multi-camera seeds")
    ap.add_argument("--run-pipeline", action="store_true",
                    help="Run YOLO+ReID pipeline if cache is missing (slow, ~30 min/slot)")
    ap.add_argument("--force-pipeline-cache", action="store_true",
                    help="Re-run YOLO+ReID and replace the slot pipeline cache even if it already exists")
    ap.add_argument("--precision-matching", action="store_true",
                    help="Use conservative cross-camera matching thresholds (default; kept for compatibility)")
    ap.add_argument("--legacy-matching", action="store_true",
                    help="Use the older looser cross-camera matching thresholds")
    ap.add_argument("--reid-device", default="cpu",
                    help="ReID device when --run-pipeline is set (cpu | cuda)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files")
    ap.add_argument("--appearance-refine", action="store_true",
                    help="Run/load crop-based person appearance refinement and merge it into vector seeds")
    ap.add_argument("--force-appearance-refine", action="store_true",
                    help="Rebuild the cached crop-based appearance refinement for the slot")
    ap.add_argument("--appearance-max-entities", type=int, default=0,
                    help="Max global person entities to appearance-refine per slot (0 = all)")
    ap.add_argument("--appearance-max-apps-per-entity", type=int, default=6,
                    help="Max camera appearances sampled per global entity")
    ap.add_argument("--appearance-crops-per-app", type=int, default=1,
                    help="Person crop images sampled per appearance")
    ap.add_argument("--scene-profile", action="store_true",
                    help="Run/load camera-level scene profiling and merge scene zone text into vector seeds")
    ap.add_argument("--force-scene-profile", action="store_true",
                    help="Rebuild cached camera-level scene profiles for the slot")
    ap.add_argument("--seed-mode", choices=["multi_camera", "single_camera"], default="multi_camera",
                    help="multi_camera keeps cross-camera entity chains; single_camera keeps only local camera evidence")
    args = ap.parse_args()

    _load_dotenv_if_available()

    video_dir = ROOT / args.video_dir
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slots = [args.slot] if args.slot else list(SLOT_CAMERAS)

    written = skipped = errors = 0

    for slot in slots:
        cache_file = PIPELINE_CACHE_DIR / f"{slot}_pipeline.json"
        refined_file = PIPELINE_CACHE_DIR / f"{slot}_refined.json"
        appearance_refined_file = PIPELINE_CACHE_DIR / f"{slot}_appearance_refined.json"
        scene_profile_file = PIPELINE_CACHE_DIR / f"{slot}_scene_profiles.json"

        # Load or generate pipeline cache
        precision_matching = (
            not args.legacy_matching
            and not _env_flag("MEVID_LEGACY_MATCHING")
        )
        if cache_file.exists() and not args.force_pipeline_cache:
            pipeline = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"[{slot}] Loaded pipeline cache: "
                  f"{len(pipeline.get('per_camera', []))} cameras, "
                  f"{len(pipeline.get('global_entities', []))} entities")
        elif args.run_pipeline:
            reason = "Force rebuild requested" if cache_file.exists() else "No cache found"
            print(f"[{slot}] {reason} — running pipeline …")
            pipeline = run_and_cache_pipeline(
                slot,
                video_dir,
                args.reid_device,
                precision_matching=precision_matching,
            )
            if not pipeline:
                errors += 1
                continue
        else:
            print(f"[{slot}] ✗  No pipeline cache at {cache_file}")
            print(f"         Run: python tests/test_mevid_full.py --slot {slot} --video-dir {args.video_dir}")
            print(f"         Or:  python scripts/generate_mevid_vector_flat.py --slot {slot} --run-pipeline")
            errors += 1
            continue

        # Load refined events if available
        refined_by_cam: dict[str, list[dict]] = {}
        if refined_file.exists():
            refined = json.loads(refined_file.read_text(encoding="utf-8"))
            for cam_id, cam_data in refined.get("per_camera", {}).items():
                refined_by_cam[cam_id] = cam_data.get("events", [])
            print(f"[{slot}] Loaded LLM-refined events for {len(refined_by_cam)} cameras")
        if args.appearance_refine:
            try:
                appearance_refined = _run_or_load_appearance_refinement(
                    slot=slot,
                    pipeline=pipeline,
                    cam_map=SLOT_CAMERAS[slot],
                    video_dir=video_dir,
                    cache_path=appearance_refined_file,
                    force=args.force_appearance_refine,
                    max_entities=args.appearance_max_entities,
                    max_apps_per_entity=args.appearance_max_apps_per_entity,
                    crops_per_app=args.appearance_crops_per_app,
                )
                loaded = 0
                for cam_id, cam_data in appearance_refined.get("per_camera", {}).items():
                    refined_by_cam.setdefault(cam_id, [])
                    refined_by_cam[cam_id].extend(cam_data.get("events", []))
                    loaded += 1
                print(f"[{slot}] Loaded/generated crop-based appearance refinement for {loaded} cameras")
            except Exception as exc:
                print(f"[{slot}] ✗  Appearance refinement failed: {exc}")
                errors += 1
                continue
        elif appearance_refined_file.exists():
            appearance_refined = json.loads(appearance_refined_file.read_text(encoding="utf-8"))
            loaded = 0
            for cam_id, cam_data in appearance_refined.get("per_camera", {}).items():
                refined_by_cam.setdefault(cam_id, [])
                refined_by_cam[cam_id].extend(cam_data.get("events", []))
                loaded += 1
            print(f"[{slot}] Loaded crop-based appearance refinement for {loaded} cameras")

        scene_profiles_by_cam: dict[str, dict] = {}
        if args.scene_profile:
            try:
                scene_profiles = _run_or_load_scene_profiles(
                    slot=slot,
                    cam_map=SLOT_CAMERAS[slot],
                    video_dir=video_dir,
                    cache_path=scene_profile_file,
                    force=args.force_scene_profile,
                )
                scene_profiles_by_cam = {
                    str(cam_id): profile
                    for cam_id, profile in scene_profiles.get("per_camera", {}).items()
                }
                print(f"[{slot}] Loaded/generated scene profiles for {len(scene_profiles_by_cam)} cameras")
            except Exception as exc:
                print(f"[{slot}] ✗  Scene profiling failed: {exc}")
                errors += 1
                continue
        elif scene_profile_file.exists():
            scene_profiles = json.loads(scene_profile_file.read_text(encoding="utf-8"))
            scene_profiles_by_cam = {
                str(cam_id): profile
                for cam_id, profile in scene_profiles.get("per_camera", {}).items()
            }
            print(f"[{slot}] Loaded cached scene profiles for {len(scene_profiles_by_cam)} cameras")

        global_entities = pipeline.get("global_entities", [])
        per_camera = {cr["camera_id"]: cr for cr in pipeline.get("per_camera", [])}

        cam_map = SLOT_CAMERAS[slot]
        for cam_id, stem in cam_map.items():
            if cam_id not in per_camera:
                print(f"  [{slot}/{cam_id}] not in pipeline cache — skipped")
                continue

            out_file = out_dir / f"{stem}_events_vector_flat.json"
            if out_file.exists() and not args.force:
                print(f"  [{slot}/{cam_id}] already exists — skip (use --force to overwrite)")
                skipped += 1
                continue

            cam_result = per_camera[cam_id]
            events     = cam_result.get("events", [])
            seed_events, dropped_short = _filter_seed_events(
                events,
                float(args.min_event_duration_sec),
            )

            # Get video duration
            video_path = video_dir / f"{stem}.avi"
            duration   = _get_video_duration(video_path) if video_path.exists() else 300.0

            flat = pipeline_events_to_vector_flat(
                video_id        = stem,
                camera_id       = cam_id,
                events          = seed_events,
                global_entities = global_entities,
                duration        = duration,
                refined_events  = refined_by_cam.get(cam_id),
                scene_profile   = scene_profiles_by_cam.get(cam_id),
                seed_mode       = args.seed_mode,
            )

            out_file.write_text(
                json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if dropped_short > 0:
                print(f"  [{slot}/{cam_id}] filtered {dropped_short} short events (< {args.min_event_duration_sec:.2f}s)")
            print(f"  [{slot}/{cam_id}] ✓  {len(flat['events'])} events → {out_file.name}")
            written += 1

    print(f"\nDone.  Written={written}  Skipped={skipped}  Errors={errors}")
    if written > 0:
        print(f"Output directory: {out_dir}")
        print("\nNext steps:")
        print("  1. Add OPENAI_API_KEY to .env")
        print("  2. python agent/test/ragas_eval_runner.py \\")
        print("       --xlsx-path agent/test/data/agent_test_mevid.xlsx \\")
        print(f"      --seed-dir {out_dir} \\")
        print("       --prepare-subset-db \\")
        print("       --limit 100")


if __name__ == "__main__":
    main()
