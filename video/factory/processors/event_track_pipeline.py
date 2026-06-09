"""
Pipeline entry: chains vision (detect+track) and analyzer (event slicing), writes JSON.
See vision.py / analyzer.py for algorithms.

CLI: video.factory.coordinator.cli_run_video_events or
python -m video.factory.coordinator video ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video.factory.processors.analyzer import aggregate_tracks, slice_events
from video.factory.processors.vision import DEFAULT_TARGET_CLASSES, resolve_model, run_yolo_track_on_video


def run_pipeline(
    video_path: str,
    model_path: str = "11m",
    conf: float = 0.5,
    iou: float = 0.45,
    motion_threshold: float = 5.0,
    min_clip_duration: float = 1.0,
    max_static_duration: float = 30.0,
    motion_window_sec: float = 1.5,
    motion_window_sum_threshold: float = 20.0,
    motion_segment_pad_sec: float = 0.8,
    tracker: str = "botsort_reid",
    target_classes: list[str] | None = None,
    save_annotated_video: bool = False,
    annotated_video_path: str | None = None,
    camera_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, Any]]:
    """Main path: video → YOLO track → aggregate tracks → slice events."""
    model_resolved, _ = resolve_model(model_path)
    effective_target_classes = list(target_classes) if target_classes is not None else list(DEFAULT_TARGET_CLASSES)
    fps, total_frames, frame_detections, tracker_label = run_yolo_track_on_video(
        video_path,
        model_path=model_resolved,
        conf=conf,
        iou=iou,
        tracker=tracker,
        target_classes=effective_target_classes,
        save_annotated_video=save_annotated_video,
        annotated_video_path=annotated_video_path,
    )
    tracks = aggregate_tracks(fps, frame_detections)
    events, clip_segments = slice_events(
        tracks,
        fps,
        frame_detections,
        motion_threshold=motion_threshold,
        min_clip_duration=min_clip_duration,
        max_static_duration=max_static_duration,
        motion_window_sec=motion_window_sec,
        motion_window_sum_threshold=motion_window_sum_threshold,
        motion_segment_pad_sec=motion_segment_pad_sec,
    )

    meta: dict[str, Any] = {
        "video_path": str(Path(video_path).resolve()),
        "fps": fps,
        "total_frames": total_frames,
        "num_tracks": len(tracks),
        "num_events": len(events),
        "num_clips": len(clip_segments),
        "tracker": tracker_label,
        "model": model_resolved,
        "model_input": model_path.strip(),
        "target_classes": effective_target_classes,
        "conf": conf,
        "iou": iou,
    }

    if camera_id is not None:
        meta["camera_id"] = camera_id
        for ev in events:
            ev["camera_id"] = camera_id

    return events, clip_segments, meta


def save_pipeline_output(
    events: list[dict[str, Any]],
    clip_segments: list[dict[str, float]],
    meta: dict[str, Any],
    out_dir: str | Path,
    base_name: str | None = None,
) -> tuple[Path, Path]:
    """Write events and clip segments to JSON. Returns (events path, clips path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if base_name is None:
        base_name = Path(meta["video_path"]).stem

    events_path = out_dir / f"{base_name}_events.json"
    clips_path = out_dir / f"{base_name}_clips.json"

    payload_events = {"meta": meta, "events": events}
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(payload_events, f, ensure_ascii=False, indent=2)

    payload_clips = {"meta": meta, "clip_segments": clip_segments}
    with open(clips_path, "w", encoding="utf-8") as f:
        json.dump(payload_clips, f, ensure_ascii=False, indent=2)

    return events_path, clips_path


if __name__ == "__main__":
    from video.factory.coordinator import cli_run_video_events

    cli_run_video_events()
