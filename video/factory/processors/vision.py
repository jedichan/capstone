"""
Detection and tracking: YOLO + Ultralytics track (BoT-SORT / ByteTrack, etc.).
See README for factory/processors/vision.py responsibilities.
"""

import shutil
from pathlib import Path

import torch
import cv2
from ultralytics import YOLO

from video.common.paths import botsort_reid_config_path, yolo_model_dir


def _resolve_device(device: str) -> str:
    """Pick a runtime device; fall back to CPU when MPS/CUDA is unusable."""
    if device == "mps":
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                torch.zeros(1).to("mps")
                return "mps"
        except Exception:
            pass
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device

_BOTSORT_REID_YAML = botsort_reid_config_path()
_YOLO_MODEL_DIR = yolo_model_dir()
_YOLO11M_LOCAL = _YOLO_MODEL_DIR / "yolo11m.pt"
DEFAULT_TARGET_CLASSES: list[str] = [
    "person",
    "car",
]


def resolve_tracker(tracker: str) -> tuple[str, str]:
    """
    Resolve a short name to an Ultralytics tracker path or built-in name.
    Returns: (tracker argument for model.track, human-readable label).
    """
    t = tracker.strip().lower()
    if t in ("bytetrack", "byte", "bt"):
        return "bytetrack.yaml", "ByteTrack"
    if t in ("botsort", "bot", "botsort_noreid"):
        return "botsort.yaml", "BoT-SORT(official default, with_reid=False)"
    if t in ("botsort_reid", "bot_reid", "default", "botsort+reid"):
        if not _BOTSORT_REID_YAML.is_file():
            return "botsort.yaml", "BoT-SORT(fallback: config/trackers/botsort_reid.yaml missing)"
        return str(_BOTSORT_REID_YAML), "BoT-SORT+ReID(config/trackers/botsort_reid.yaml)"
    p = Path(tracker)
    if p.is_file():
        return str(p.resolve()), str(p.resolve())
    return tracker, tracker


def resolve_model(model: str) -> tuple[str, str]:
    """
    Resolve model path: supports aliases n/s/m/l/x or any .pt / custom weight path.
    Returns: (path passed to YOLO(), display name written to meta).
    """
    raw = model.strip()
    key = raw.lower()
    aliases: dict[str, str] = {
        "n": "yolov8n.pt",
        "nano": "yolov8n.pt",
        "s": "yolov8s.pt",
        "small": "yolov8s.pt",
        "m": str(_YOLO11M_LOCAL),
        "medium": str(_YOLO11M_LOCAL),
        "11m": str(_YOLO11M_LOCAL),
        "yolo11m": str(_YOLO11M_LOCAL),
        "yolov11m": str(_YOLO11M_LOCAL),
        "l": "yolov8l.pt",
        "large": "yolov8l.pt",
        "x": "yolov8x.pt",
        "xlarge": "yolov8x.pt",
    }
    if key in aliases:
        p = aliases[key]
        return p, p
    return raw, raw


def run_yolo_track_on_video(
    video_path: str,
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    tracker: str = "botsort_reid",
    target_classes: list[str] | None = None,
    save_annotated_video: bool = False,
    annotated_video_path: str | None = None,
):
    """
    Run YOLO + tracking on video (default BoT-SORT+ReID); collect per-frame detections.
    Returns: (fps, total_frames, per-frame detections, tracker_name).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    model_pt, _ = resolve_model(model_path)
    if model_pt == str(_YOLO11M_LOCAL):
        _YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if not _YOLO11M_LOCAL.is_file():
            # Trigger ultralytics download, then copy weights into repo _model for reproducibility.
            downloaded = YOLO("yolo11m.pt")
            src = Path(getattr(downloaded, "ckpt_path", "") or "")
            if src.is_file():
                shutil.copy2(src, _YOLO11M_LOCAL)
            del downloaded
        model_pt = str(_YOLO11M_LOCAL)
    model = YOLO(model_pt)
    tracker_cfg, tracker_name = resolve_tracker(tracker)
    effective_classes = list(target_classes) if target_classes is not None else list(DEFAULT_TARGET_CLASSES)
    classes_ids: list[int] | None = None
    if effective_classes:
        name_to_id = {str(v).lower(): int(k) for k, v in model.names.items()}
        classes_ids = []
        for c in effective_classes:
            key = str(c).strip().lower()
            if key in name_to_id:
                classes_ids.append(name_to_id[key])
        if not classes_ids:
            raise ValueError(f"No matching target classes: {effective_classes}")
    writer: cv2.VideoWriter | None = None
    if save_annotated_video:
        if annotated_video_path is None:
            p = Path(video_path)
            annotated_video_path = str(p.with_name(p.stem + "_tracked.mp4"))

    results = model.track(
        source=video_path,
        conf=conf,
        iou=iou,
        classes=classes_ids,
        persist=True,
        stream=True,
        verbose=False,
        tracker=tracker_cfg,
        device=_resolve_device("cuda" if torch.cuda.is_available() else "mps"),
    )

    frame_detections: list[list[tuple[int | None, str, float, list[float]]]] = []
    for r in results:
        if save_annotated_video:
            plotted = r.plot()
            if writer is None:
                h, w = plotted.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(annotated_video_path), fourcc, float(fps), (w, h))
            writer.write(plotted)

        row: list[tuple[int | None, str, float, list[float]]] = []
        if r.boxes is not None:
            for box in r.boxes:
                tid = int(box.id[0]) if box.id is not None else None
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                c = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                row.append((tid, cls_name, c, [x1, y1, x2, y2]))
        frame_detections.append(row)

    if writer is not None:
        writer.release()

    return fps, total_frames, frame_detections, tracker_name
