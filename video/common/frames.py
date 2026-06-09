"""Video frame sampling, resolution, and event bbox normalization (OpenCV; shared by refinement and other modules)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FrameSample:
    t_sec: float
    jpg_base64: str


@dataclass
class PersonCrop:
    """Single person crop image (for Re-ID)."""

    t_sec: float
    camera_id: str
    track_id: int
    image_array: np.ndarray
    jpg_base64: str


def _encode_bgr_to_jpg_base64(img_bgr, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def sample_frames_uniform(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int = 12,
    resize_width: int = 768,
) -> list[FrameSample]:
    """Uniformly sample frames in [start_sec, end_sec] and encode each as base64 JPEG."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        cap.release()
        return []

    times = [start_sec + (i + 0.5) * duration / num_frames for i in range(num_frames)]
    out: list[FrameSample] = []

    for t in times:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        if resize_width and w > resize_width:
            new_w = resize_width
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        out.append(FrameSample(t_sec=float(t), jpg_base64=_encode_bgr_to_jpg_base64(frame)))

    cap.release()
    return out


def get_video_size(video_path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError("Cannot read video resolution")
    return w, h


def _bbox_center_norm(xyxy: list[float], w: int, h: int) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    return float(cx), float(cy)


def extract_person_crops(
    video_path: str,
    track: dict[str, Any],
    camera_id: str = "",
    num_crops: int = 5,
    resize: tuple[int, int] = (256, 128),
    min_crop_hw: tuple[int, int] = (32, 16),
) -> list[PersonCrop]:
    """Sample N frames uniformly from track time_xyxy, crop by bbox, resize to Re-ID input size.

    Args:
        resize: (height, width) — typical Re-ID input 256x128.
    """
    time_xyxy: list[tuple[float, list[float]]] = track.get("time_xyxy", [])
    if not time_xyxy:
        return []

    n = min(num_crops, len(time_xyxy))
    step = max(1, len(time_xyxy) // n)
    indices = [i * step for i in range(n)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    crops: list[PersonCrop] = []
    track_id: int = track.get("track_id", -1)

    for idx in indices:
        t_sec, xyxy = time_xyxy[idx]
        frame_no = int(round(t_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        x1 = max(0, int(xyxy[0]))
        y1 = max(0, int(xyxy[1]))
        x2 = min(vid_w, int(xyxy[2]))
        y2 = min(vid_h, int(xyxy[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        crop_h = y2 - y1
        crop_w = x2 - x1
        if crop_h < min_crop_hw[0] or crop_w < min_crop_hw[1]:
            continue

        crop_img = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop_img, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)

        crops.append(PersonCrop(
            t_sec=float(t_sec),
            camera_id=camera_id,
            track_id=track_id,
            image_array=crop_resized,
            jpg_base64=_encode_bgr_to_jpg_base64(crop_resized),
        ))

    cap.release()
    return crops


def enrich_events_with_normalized_location(raw_events: list[dict[str, Any]], w: int, h: int) -> list[dict[str, Any]]:
    """
    Add normalized location fields to raw_events for LLM merge constraints by relative scale.
    When bbox exists: start_center_norm / end_center_norm / start_bbox_norm / end_bbox_norm
    """
    out: list[dict[str, Any]] = []
    for e in raw_events:
        e2 = dict(e)
        for key in ("start_bbox_xyxy", "end_bbox_xyxy"):
            if key in e2 and isinstance(e2[key], list) and len(e2[key]) == 4:
                x1, y1, x2, y2 = [float(v) for v in e2[key]]
                e2[key] = [x1, y1, x2, y2]
                e2[key.replace("_xyxy", "_norm")] = [x1 / w, y1 / h, x2 / w, y2 / h]
                cx, cy = _bbox_center_norm([x1, y1, x2, y2], w, h)
                e2[key.replace("_bbox_xyxy", "_center_norm")] = [cx, cy]
        out.append(e2)
    return out


def crop_bgr_at_time_xyxy(
    video_path: str,
    t_sec: float,
    xyxy: list[float],
) -> np.ndarray | None:
    """Crop bbox region at time t_sec; return BGR ndarray or None on failure."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_no = int(round(float(t_sec) * float(fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or vid_h <= 0 or vid_w <= 0:
        return None
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(vid_w, int(xyxy[2]))
    y2 = min(vid_h, int(xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def coarse_color_label_from_bgr(img_bgr: np.ndarray) -> str:
    """Very coarse color bucket for hard filtering (only blocks merge when clearly different)."""
    if img_bgr.size == 0:
        return "unknown"
    b, g, r = [float(x) for x in img_bgr.reshape(-1, 3).mean(axis=0)]
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 60:
        return "black"
    if mx > 210 and (mx - mn) < 25:
        return "white"
    if (mx - mn) < 18 and mx > 120:
        return "silver_gray"
    if r > g + 40 and r > b + 40:
        return "red"
    if b > r + 40 and b > g + 40:
        return "blue"
    return "dark"


# Backward-compatible name (values are now English labels).
coarse_color_from_bgr = coarse_color_label_from_bgr
