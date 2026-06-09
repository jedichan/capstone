"""Best-effort person crop sampling from tracked event bboxes.

This module is intentionally small and dependency-light so both evaluation code
and production video refinement can reuse the same crop selection behavior.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2


def encode_crop_b64(crop: Any, max_edge: int = 320, jpeg_quality: int = 78) -> str | None:
    """Encode one image crop as base64 JPEG for VLM input."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        crop = cv2.resize(
            crop,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return None
    return base64.b64encode(buf).decode()


def sample_person_crops_from_events(
    video_path: str | Path,
    events: list[dict[str, Any]],
    max_tracks: int = 6,
    crops_per_track: int = 2,
    min_crop_hw: tuple[int, int] = (36, 18),
    padding: float = 0.20,
) -> list[str]:
    """Sample person crops from tracked event bboxes, capped by track.

    Bad bboxes, missing videos, tiny crops, and read failures are skipped. The
    caller gets an empty list instead of an exception because crop evidence is an
    enhancement, not a hard requirement for the pipeline.
    """
    person_events = [
        e for e in events
        if str(e.get("class_name") or e.get("object_type") or "").lower()
        in {"person", "people", "pedestrian"}
    ]

    by_track: dict[int, list[dict[str, Any]]] = {}
    for event in person_events:
        tid = event.get("track_id")
        if tid is None:
            continue
        try:
            by_track.setdefault(int(tid), []).append(event)
        except Exception:
            continue

    def bbox_area(event: dict[str, Any]) -> float:
        areas: list[float] = []
        for key in ("start_bbox_xyxy", "end_bbox_xyxy"):
            bbox = event.get(key)
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        return max(areas) if areas else 0.0

    ranked_tracks = sorted(
        by_track.items(),
        key=lambda item: max((bbox_area(e) for e in item[1]), default=0.0),
        reverse=True,
    )[:max_tracks]
    if not ranked_tracks:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    if vid_h <= 0 or vid_w <= 0:
        cap.release()
        return []

    crops: list[str] = []
    for _tid, track_events in ranked_tracks:
        for event in sorted(track_events, key=bbox_area, reverse=True)[:crops_per_track]:
            for t_key, b_key in (("start_time", "start_bbox_xyxy"), ("end_time", "end_bbox_xyxy")):
                bbox = event.get(b_key)
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                bw, bh = x2 - x1, y2 - y1
                if bw < min_crop_hw[1] or bh < min_crop_hw[0]:
                    continue
                px, py = bw * padding, bh * padding
                xi1 = max(0, int(x1 - px))
                yi1 = max(0, int(y1 - py))
                xi2 = min(vid_w, int(x2 + px))
                yi2 = min(vid_h, int(y2 + py))
                if xi2 <= xi1 or yi2 <= yi1:
                    continue
                t = float(event.get(t_key, event.get("start_time", 0.0)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                encoded = encode_crop_b64(frame[yi1:yi2, xi1:xi2])
                if encoded:
                    crops.append(encoded)
                break
    cap.release()
    return crops


def sample_person_crops_for_appearances(
    camera_to_video: dict[str, str | Path],
    camera_to_events: dict[str, list[dict[str, Any]]],
    appearances: list[dict[str, Any]],
    max_apps: int = 6,
    crops_per_app: int = 1,
) -> list[str]:
    """Sample crops for specific global-entity camera appearances."""
    crops: list[str] = []
    for app in appearances[:max_apps]:
        cam_id = str(app.get("camera_id") or "")
        if not cam_id:
            continue
        video_path = camera_to_video.get(cam_id)
        if not video_path:
            continue
        tid = app.get("track_id")
        events = [
            e for e in camera_to_events.get(cam_id, [])
            if tid is None or e.get("track_id") == tid
        ]
        if not events:
            continue
        crops.extend(
            sample_person_crops_from_events(
                video_path,
                events,
                max_tracks=1,
                crops_per_track=crops_per_app,
            )
        )
    return crops
