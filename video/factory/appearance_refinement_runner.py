"""Crop-based person appearance refinement for multi-camera pipeline output.

The output deliberately reuses the existing vector event fields so downstream
RAG/agent code can benefit from richer appearance text without a schema change.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from video.factory.person_crop_sampler import (
    sample_person_crops_for_appearances,
    sample_person_crops_from_events,
)

logger = logging.getLogger(__name__)


def _refine_concurrency() -> int:
    """Max concurrent appearance-VLM calls (I/O-bound). Env: VIDEO_REFINE_CONCURRENCY."""
    try:
        return max(1, int(os.getenv("VIDEO_REFINE_CONCURRENCY", "4")))
    except (TypeError, ValueError):
        return 4


@dataclass
class AppearanceRefinementConfig:
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-vl-flash"
    temperature: float = 0.0
    max_tokens: int = 260
    max_entities: int = 0
    max_apps_per_entity: int = 6
    crops_per_app: int = 1
    cache_path: Path | None = None
    force: bool = False

    @classmethod
    def from_env(cls, **overrides: Any) -> "AppearanceRefinementConfig":
        values = {
            "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
            "base_url": os.getenv("DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "model": os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3-vl-flash"),
        }
        values.update(overrides)
        return cls(**values)


def _fmt_sec(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    return f"{minutes}:{sec:02d}"


def _json_object_from_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start:end + 1])
    raise ValueError(f"No JSON object found in response: {text[:200]}")


def appearance_keywords_from_text(text: str) -> list[str]:
    tl = text.lower().replace("-", " ")
    terms = [
        "hoodie", "jacket", "coat", "shirt", "pants", "trousers", "jeans",
        "bag", "backpack", "handbag", "hat", "cap", "hood", "scarf", "fur",
        "collar", "white", "black", "grey", "gray", "dark", "light", "beige",
        "brown", "red", "blue", "green",
    ]
    out: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.strip().lower().replace(" ", "_").replace("gray", "grey")
        if token and token != "unknown" and token not in seen:
            out.append(token)
            seen.add(token)

    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", tl):
            add(term)
    if ("light grey" in tl or "light gray" in tl) and "hoodie" in tl:
        add("light_grey_hoodie")
    if "beige" in tl and ("jacket" in tl or "coat" in tl):
        add("beige_jacket")
    if ("dark" in tl or "black" in tl) and "coat" in tl:
        add("dark_coat")
    if ("dark" in tl or "black" in tl) and "hood" in tl:
        add("hood_up")
    if "fur" in tl and ("hood" in tl or "collar" in tl):
        add("fur_trimmed_hood")
    return out[:12]


def color_from_appearance(text: str) -> str:
    tl = text.lower().replace("-", " ")
    for phrase, color in [
        ("light grey", "light_grey"),
        ("light gray", "light_grey"),
        ("dark grey", "dark_grey"),
        ("dark gray", "dark_grey"),
        ("silver grey", "silver_grey"),
        ("silver gray", "silver_grey"),
    ]:
        if phrase in tl:
            return color
    for color in ["beige", "white", "black", "grey", "gray", "dark", "light", "brown", "red", "blue", "green"]:
        if re.search(rf"\b{color}\b", tl):
            return "grey" if color == "gray" else color
    return "unknown"


def _bbox_center(event: dict[str, Any], key: str) -> tuple[float, float] | None:
    box = event.get(key)
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
    except Exception:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _infer_motion_from_events(events: list[dict[str, Any]]) -> dict[str, str]:
    if not events:
        return {"action_summary": "visible in the scene", "movement_direction": "unknown", "exit_side": "unknown"}

    ordered = sorted(events, key=lambda e: float(e.get("start_time", 0.0)))
    first = ordered[0]
    last = ordered[-1]
    start_center = _bbox_center(first, "start_bbox_xyxy") or _bbox_center(first, "end_bbox_xyxy")
    end_center = _bbox_center(last, "end_bbox_xyxy") or _bbox_center(last, "start_bbox_xyxy")
    if not start_center or not end_center:
        return {"action_summary": "visible in the scene", "movement_direction": "unknown", "exit_side": "unknown"}

    dx = end_center[0] - start_center[0]
    dy = end_center[1] - start_center[1]
    if abs(dx) < 20 and abs(dy) < 20:
        direction = "stationary"
        action = "standing or moving slightly"
    elif abs(dx) >= abs(dy):
        direction = "right" if dx > 0 else "left"
        action = f"moving toward the {direction} side"
    else:
        direction = "down" if dy > 0 else "up"
        action = f"moving toward the {direction} side"
    return {
        "action_summary": action,
        "movement_direction": direction,
        "exit_side": direction if direction != "stationary" else "unknown",
    }


def _append_unique_keywords(base: Any, extra: list[str], max_len: int = 16) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(base, list):
        candidates = base
    elif isinstance(base, str):
        candidates = re.split(r"[,;/\s]+", base)
    else:
        candidates = []
    for item in list(candidates) + list(extra):
        token = str(item).strip().lower().replace(" ", "_")
        if token and token != "unknown" and token not in seen:
            out.append(token)
            seen.add(token)
        if len(out) >= max_len:
            break
    return out


def _append_semantics_to_event_text(
    text: Any,
    notes: str,
    color: str,
    action_summary: str = "",
    movement_direction: str = "",
    exit_side: str = "",
) -> str:
    base = str(text or "").strip()
    parts: list[str] = []
    if notes and notes.lower() != "unknown" and notes.lower() not in base.lower():
        parts.append(f"Appearance: {notes}.")
    if color and color.lower() != "unknown" and color.lower().replace("_", " ") not in base.lower():
        parts.append(f"Color: {color.replace('_', ' ')}.")
    if action_summary and action_summary.lower() != "unknown" and action_summary.lower() not in base.lower():
        parts.append(f"Action: {action_summary}.")
    if movement_direction and movement_direction.lower() not in {"", "unknown", "stationary"}:
        if f"toward the {movement_direction.lower()}" not in base.lower():
            parts.append(f"Movement direction: {movement_direction.lower()}.")
    if exit_side and exit_side.lower() not in {"", "unknown", "stationary"}:
        if f"exit side: {exit_side.lower()}" not in base.lower():
            parts.append(f"Exit side: {exit_side.lower()}.")
    if not base:
        return " ".join(parts).strip()
    if parts:
        return f"{base.rstrip()} {' '.join(parts)}"
    return base


def _call_appearance_vlm(
    client: OpenAI,
    crops_b64: list[str],
    entity_id: str,
    trajectory: str,
    config: AppearanceRefinementConfig,
) -> dict[str, Any]:
    user_content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            "Look at the person in the provided crop images. "
            "Return strict JSON with exactly these three fields:\n"
            "{\n"
            '  "object_color": "dominant upper-body color word (e.g. dark, red, light, white, black)",\n'
            '  "appearance_notes": "clothing description: upper-body color+type, lower-body color+type, '
            'accessories if visible, rough build (e.g. dark blue hoodie, grey pants, backpack, medium build)",\n'
            '  "action_summary": "visible action only, e.g. walking right, standing, leaving frame, unknown",\n'
            '  "keywords": ["color/clothing tokens for retrieval"]\n'
            "}\n"
            "Rules: use concrete visible words only; if a detail is unclear say unknown for that part; "
            "do not invent colors or clothing not visible in the images."
        ),
    }]
    for b64 in crops_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    resp = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You produce compact person appearance metadata for RAG. "
                    "Return only valid JSON. Do not include markdown."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=float(config.temperature),
        max_tokens=int(config.max_tokens),
    )
    parsed = _json_object_from_text(resp.choices[0].message.content or "{}")
    appearance = str(parsed.get("appearance_notes") or "").strip()
    action_summary = str(parsed.get("action_summary") or "").strip()
    color = str(parsed.get("object_color") or "").strip().lower().replace(" ", "_")
    if not color or color == "unknown":
        color = color_from_appearance(appearance)
    keywords = parsed.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    normalized = [str(k).strip().lower().replace(" ", "_") for k in keywords if str(k).strip()]
    for token in appearance_keywords_from_text(" ".join([appearance, color])):
        if token not in normalized:
            normalized.append(token)
    return {
        "object_color": color or "unknown",
        "appearance_notes": appearance or "unknown",
        "action_summary": action_summary or "visible in the scene",
        "keywords": [k for k in normalized if k and k != "unknown"][:12],
    }


def merge_refined_appearance(base: dict[str, Any] | None, appearance: dict[str, Any] | None) -> dict[str, Any]:
    """Append appearance-refined events to a clip-refined result."""
    if not base:
        return appearance or {}
    if not appearance:
        return base
    merged = json.loads(json.dumps(base))
    merged.setdefault("per_camera", {})
    for cam_id, cam_data in appearance.get("per_camera", {}).items():
        dst = merged["per_camera"].setdefault(cam_id, {"events": []})
        dst.setdefault("events", [])
        dst["events"].extend(cam_data.get("events", []))
    return merged


def _person_events_by_track(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        obj = str(event.get("class_name") or event.get("object_type") or "").lower()
        if obj not in {"person", "people", "pedestrian"}:
            continue
        tid = event.get("track_id")
        if tid is None:
            continue
        try:
            by_track[int(tid)].append(event)
        except Exception:
            continue
    return by_track


def run_appearance_refinement_for_events(
    *,
    video_path: str | Path,
    events: list[dict[str, Any]],
    video_id: str | None = None,
    camera_id: str | None = None,
    config: AppearanceRefinementConfig | None = None,
    base_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run crop-based appearance refinement for one camera/video.

    If ``base_events`` is provided (e.g. from a prior vector-refine pass), the
    appearance fields (object_color, appearance_notes, keywords) are patched into
    copies of those events so that scene_zone, event_text, bbox and other fields
    are preserved.  Without base_events a minimal new event is created per track.

    The returned payload is vector-refinement shaped::

        {"video_id": "...", "events": [...]}
    """
    cfg = config or AppearanceRefinementConfig.from_env()
    if cfg.cache_path and cfg.cache_path.exists() and not cfg.force:
        logger.info("Loading cached single-camera appearance refinement from %s", cfg.cache_path)
        return json.loads(cfg.cache_path.read_text(encoding="utf-8"))
    if not cfg.api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for appearance refinement")

    video_path = Path(video_path)
    out_video_id = video_id or video_path.name

    # Build track_id → base event list from the prior refine output (if any)
    track_to_base: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if base_events:
        for ev in base_events:
            tid = ev.get("track_id")
            if tid is not None:
                try:
                    track_to_base[int(tid)].append(ev)
                except Exception:
                    pass

    by_track = _person_events_by_track(events)
    track_items = sorted(
        by_track.items(),
        key=lambda item: min(float(e.get("start_time", 0.0)) for e in item[1]),
    )
    if cfg.max_entities > 0:
        track_items = track_items[:cfg.max_entities]

    timeout_sec = float(os.getenv("VIDEO_VLM_TIMEOUT_SEC", "180"))
    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=timeout_sec,
        max_retries=int(os.getenv("VIDEO_VLM_MAX_RETRIES", "1")),
    )
    t0 = time.time()

    def _process_track(tid: int, track_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Refine one person track; returns the produced event records (may be empty)."""
        crops = sample_person_crops_from_events(
            video_path,
            track_events,
            max_tracks=1,
            crops_per_track=max(1, cfg.crops_per_app),
        )
        if not crops:
            return []

        start = min(float(e.get("start_time", 0.0)) for e in track_events)
        end = max(float(e.get("end_time", start)) for e in track_events)
        entity_id = f"track_{tid}"
        motion = _infer_motion_from_events(track_events)
        trajectory = f"{camera_id or 'camera'} track#{tid} {_fmt_sec(start)}-{_fmt_sec(end)}"

        t_track = time.time()
        try:
            appearance = _call_appearance_vlm(client, crops, entity_id, trajectory, cfg)
        except Exception as exc:
            logger.warning("appearance refine failed for track_%s: %s", tid, exc)
            print(f"[appearance] track {tid}: FAILED in {time.time() - t_track:.1f}s ({exc})", flush=True)
            return []

        notes = appearance["appearance_notes"]
        color = appearance["object_color"]
        action_summary = appearance.get("action_summary") or motion["action_summary"]
        movement_direction = motion["movement_direction"]
        exit_side = motion["exit_side"]
        keywords = list(appearance["keywords"])
        for token in ("person", "appearance"):
            if token not in keywords:
                keywords.append(token)
        for token in (action_summary, movement_direction, exit_side):
            token = str(token or "").strip().lower().replace(" ", "_")
            if token and token not in {"unknown", "stationary"} and token not in keywords:
                keywords.append(token)
        keywords = [k for k in keywords if k and k != "unknown"][:14]

        track_events_out: list[dict[str, Any]] = []
        base_list = track_to_base.get(tid)
        if base_list:
            # Patch appearance fields into copies of existing refined events
            for base_ev in base_list:
                patched = dict(base_ev)
                patched["object_color"] = color
                patched["appearance_notes"] = notes
                patched["action_summary"] = action_summary
                patched["movement_direction"] = movement_direction
                patched["exit_side"] = exit_side
                patched["event_text"] = _append_semantics_to_event_text(
                    patched.get("event_text") or patched.get("description"),
                    notes,
                    color,
                    action_summary,
                    movement_direction,
                    exit_side,
                )
                patched["keywords"] = _append_unique_keywords(patched.get("keywords"), keywords)
                track_events_out.append(patched)
        else:
            # No prior refine — build a minimal new event
            track_events_out.append({
                "video_id": out_video_id,
                "camera_id": camera_id,
                "track_id": tid,
                "entity_hint": entity_id,
                "clip_start_sec": start,
                "clip_end_sec": end,
                "start_time": start,
                "end_time": end,
                "object_type": "person",
                "object_color": color,
                "appearance_notes": notes,
                "action_summary": action_summary,
                "movement_direction": movement_direction,
                "exit_side": exit_side,
                "scene_zone": "unknown",
                "event_text": _append_semantics_to_event_text(
                    f"{entity_id} from {_fmt_sec(start)} to {_fmt_sec(end)}.",
                    notes,
                    color,
                    action_summary,
                    movement_direction,
                    exit_side,
                ),
                "keywords": keywords,
                "start_bbox_xyxy": None,
                "end_bbox_xyxy": None,
            })
        print(
            f"[appearance] track {tid}: color={color}, {len(track_events_out)} events "
            f"in {time.time() - t_track:.1f}s",
            flush=True,
        )
        return track_events_out

    # Person tracks are independent and each makes a remote VLM call → run
    # concurrently. Order is irrelevant (result is sorted by start_time below).
    refined_events: list[dict[str, Any]] = []
    workers = min(_refine_concurrency(), max(1, len(track_items)))
    print(
        f"[appearance] start: {len(track_items)} person tracks, "
        f"model={cfg.model}, concurrency={workers}",
        flush=True,
    )
    if workers <= 1 or len(track_items) <= 1:
        for tid, track_events in track_items:
            try:
                refined_events.extend(_process_track(tid, track_events))
            except Exception as exc:  # noqa: BLE001
                logger.warning("appearance refine failed for track_%s: %s", tid, exc)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_tid = {
                executor.submit(_process_track, tid, track_events): tid
                for tid, track_events in track_items
            }
            for future, tid in future_to_tid.items():
                try:
                    refined_events.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("appearance refine failed for track_%s: %s", tid, exc)

    result = {
        "video_id": out_video_id,
        "camera_id": camera_id,
        "elapsed_sec": round(time.time() - t0, 1),
        "events": sorted(refined_events, key=lambda e: float(e.get("start_time", 0.0))),
    }
    if cfg.cache_path:
        cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_appearance_refinement_for_pipeline(
    *,
    slot: str,
    pipeline: dict[str, Any],
    camera_to_video: dict[str, str | Path],
    camera_video_stems: dict[str, str] | None = None,
    config: AppearanceRefinementConfig | None = None,
) -> dict[str, Any]:
    """Run crop-based appearance refinement for all global person entities."""
    cfg = config or AppearanceRefinementConfig.from_env()
    if cfg.cache_path and cfg.cache_path.exists() and not cfg.force:
        logger.info("[%s] Loading cached appearance refinement from %s", slot, cfg.cache_path)
        return json.loads(cfg.cache_path.read_text(encoding="utf-8"))
    if not cfg.api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for appearance refinement")

    per_cam = {
        cr.get("camera_id"): cr
        for cr in pipeline.get("per_camera", [])
        if cr.get("camera_id")
    }
    camera_to_events = {
        str(cam_id): list(cam_data.get("events", []))
        for cam_id, cam_data in per_cam.items()
    }
    timeout_sec = float(os.getenv("VIDEO_VLM_TIMEOUT_SEC", "180"))
    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=timeout_sec,
        max_retries=int(os.getenv("VIDEO_VLM_MAX_RETRIES", "1")),
    )
    per_cam_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entities = list(pipeline.get("global_entities", []))
    if cfg.max_entities > 0:
        entities = entities[:cfg.max_entities]

    t0 = time.time()

    def _process_entity(ge: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """Refine one global entity; returns (camera_id, event) pairs (may be empty)."""
        entity_id = str(ge.get("global_entity_id") or "")
        apps = sorted(ge.get("appearances", []), key=lambda a: float(a.get("start_time", 0.0)))
        if not entity_id or not apps:
            return []

        trajectory = " -> ".join(
            f"{a.get('camera_id')} track#{a.get('track_id')} "
            f"{_fmt_sec(float(a.get('start_time', 0.0)))}-{_fmt_sec(float(a.get('end_time', 0.0)))}"
            for a in apps
        )
        crops = sample_person_crops_for_appearances(
            camera_to_video=camera_to_video,
            camera_to_events=camera_to_events,
            appearances=apps,
            max_apps=cfg.max_apps_per_entity,
            crops_per_app=cfg.crops_per_app,
        )
        if not crops:
            return []
        t_ent = time.time()
        try:
            appearance = _call_appearance_vlm(client, crops, entity_id, trajectory, cfg)
        except Exception as exc:
            logger.warning("[%s] appearance refine failed for %s: %s", slot, entity_id, exc)
            print(f"[appearance] {entity_id}: FAILED in {time.time() - t_ent:.1f}s ({exc})", flush=True)
            return []

        color = appearance["object_color"]
        notes = appearance["appearance_notes"]
        action_summary = appearance.get("action_summary") or "visible in the scene"
        keywords = list(appearance["keywords"])
        for token in ("person", "appearance", "cross_camera", "same_person", entity_id.lower()):
            if token not in keywords:
                keywords.append(token)
        event_text = (
            f"{entity_id} appeared across cameras: {trajectory}. "
            f"Appearance: {notes}. Action: {action_summary}."
        )
        entity_events: list[tuple[str, dict[str, Any]]] = []
        for app in apps:
            cam_id = str(app.get("camera_id") or "")
            if not cam_id:
                continue
            cam_events = [
                e for e in camera_to_events.get(cam_id, [])
                if str(e.get("track_id")) == str(app.get("track_id"))
            ]
            motion = _infer_motion_from_events(cam_events)
            app_event_text = _append_semantics_to_event_text(
                event_text,
                notes,
                color,
                action_summary or motion["action_summary"],
                motion["movement_direction"],
                motion["exit_side"],
            )
            app_keywords = list(keywords)
            for token in (motion["action_summary"], motion["movement_direction"], motion["exit_side"]):
                token = str(token or "").strip().lower().replace(" ", "_")
                if token and token not in {"unknown", "stationary"} and token not in app_keywords:
                    app_keywords.append(token)
            entity_events.append((cam_id, {
                "video_id": (camera_video_stems or {}).get(cam_id, ""),
                "camera_id": cam_id,
                "track_id": app.get("track_id"),
                "entity_hint": entity_id,
                "clip_start_sec": float(app.get("start_time", 0.0)),
                "clip_end_sec": float(app.get("end_time", 0.0)),
                "start_time": float(app.get("start_time", 0.0)),
                "end_time": float(app.get("end_time", 0.0)),
                "object_type": "person",
                "object_color": color,
                "appearance_notes": notes,
                "action_summary": action_summary or motion["action_summary"],
                "movement_direction": motion["movement_direction"],
                "exit_side": motion["exit_side"],
                "scene_zone": "unknown",
                "event_text": app_event_text,
                "keywords": app_keywords[:16],
                "start_bbox_xyxy": None,
                "end_bbox_xyxy": None,
            }))
        print(
            f"[appearance] {entity_id}: color={color}, {len(entity_events)} cam-events "
            f"in {time.time() - t_ent:.1f}s",
            flush=True,
        )
        return entity_events

    # Global entities are independent → refine concurrently; merge into the
    # per-camera buckets on the main thread to avoid shared-dict races.
    workers = min(_refine_concurrency(), max(1, len(entities)))
    print(
        f"[appearance] start: {len(entities)} global entities, "
        f"model={cfg.model}, concurrency={workers}",
        flush=True,
    )
    if workers <= 1 or len(entities) <= 1:
        for ge in entities:
            for cam_id, event in _process_entity(ge):
                per_cam_events[cam_id].append(event)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_eid = {
                executor.submit(_process_entity, ge): str(ge.get("global_entity_id") or "")
                for ge in entities
            }
            for future, eid in future_to_eid.items():
                try:
                    for cam_id, event in future.result():
                        per_cam_events[cam_id].append(event)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] appearance refine failed for %s: %s", slot, eid, exc)

    result = {
        "slot": slot,
        "elapsed_sec": round(time.time() - t0, 1),
        "per_camera": {
            cam_id: {"events": sorted(events, key=lambda e: float(e.get("start_time", 0.0)))}
            for cam_id, events in per_cam_events.items()
        },
    }
    if cfg.cache_path:
        cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
