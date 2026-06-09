"""Single-camera semantic video pipeline.

This module is the formal entry point for running one video as a
single-camera retrieval source: detection/tracking, optional semantic
refinement, scene profiling, and vector-flat seed generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.generate_mevid_vector_flat import _get_video_duration, pipeline_events_to_vector_flat
from video.factory.appearance_refinement_runner import (
    AppearanceRefinementConfig,
    run_appearance_refinement_for_events,
)
from video.factory.processors.event_track_pipeline import run_pipeline, save_pipeline_output
from video.factory.refinement_runner import RefineEventsConfig, run_refine_events_from_files
from video.factory.scene_profile_runner import SceneProfileConfig, run_scene_profiles_for_pipeline


DEFAULT_SINGLE_CAMERA_CLASSES = ("person", "car")


@dataclass(slots=True)
class SingleCameraPipelineConfig:
    """Configuration for one-video single-camera vector seed generation."""

    model: str = "11m"
    conf: float = 0.35
    iou: float = 0.35
    tracker: str = "botsort_reid"
    target_classes: tuple[str, ...] | None = DEFAULT_SINGLE_CAMERA_CLASSES
    min_event_duration_sec: float = 1.0
    semantic_refine: bool = False
    clip_refine: bool = False
    appearance_refine: bool = False
    scene_profile: bool = False
    clip_refine_model: str = "qwen3-vl-flash"
    clip_refine_min_frames: int = 4
    clip_refine_max_frames: int = 12
    appearance_model: str | None = None
    crops_per_track: int = 2
    max_tracks: int = 0
    force: bool = False


def infer_camera_id_from_stem(stem: str) -> str:
    """Infer a camera id such as G423 from a MEVID-like filename stem."""
    for part in reversed(stem.split(".")):
        if part.upper().startswith("G") and part[1:].isdigit():
            return part.upper()
    return "CAM"


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


def run_single_camera_semantic_pipeline(
    video_path: str | Path,
    *,
    out_dir: str | Path,
    seed_out_dir: str | Path,
    camera_id: str | None = None,
    video_id: str | None = None,
    config: SingleCameraPipelineConfig | None = None,
) -> dict[str, Any]:
    """Run one video and write a single-camera vector-flat seed.

    Returns a manifest-like dict containing output paths, counts, and the
    generated vector-flat payload. The vector seed is always written as
    ``<video_id>_events_vector_flat.json`` under ``seed_out_dir``.
    """
    cfg = config or SingleCameraPipelineConfig()
    video = Path(video_path).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    stem = video_id or video.stem
    cam = (camera_id or infer_camera_id_from_stem(stem)).strip().upper() or "CAM"
    pipeline_out = Path(out_dir).expanduser().resolve()
    seed_dir = Path(seed_out_dir).expanduser().resolve()
    pipeline_out.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_path = seed_dir / f"{stem}_events_vector_flat.json"
    if seed_path.exists() and not cfg.force:
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
        return {
            "video_id": stem,
            "camera_id": cam,
            "seed_path": str(seed_path),
            "vector_event_count": len(payload.get("events", [])),
            "flat": payload,
            "skipped_existing": True,
        }

    events, clips, meta = run_pipeline(
        str(video),
        model_path=cfg.model,
        conf=float(cfg.conf),
        iou=float(cfg.iou),
        motion_threshold=5.0,
        min_clip_duration=1.0,
        max_static_duration=30.0,
        tracker=cfg.tracker,
        camera_id=cam,
        target_classes=list(cfg.target_classes) if cfg.target_classes else None,
    )
    events_path, clips_path = save_pipeline_output(events, clips, meta, pipeline_out)

    refined_events: list[dict[str, Any]] | None = None
    scene_profile_payload: dict[str, Any] | None = None
    run_clip = bool(cfg.semantic_refine or cfg.clip_refine)
    run_appearance = bool(cfg.semantic_refine or cfg.appearance_refine)
    run_scene = bool(cfg.semantic_refine or cfg.scene_profile)

    clip_refine_path: Path | None = None
    if run_clip:
        clip_cfg = RefineEventsConfig(
            mode="vector",
            frames_per_sec=0.1,
            min_frames=max(1, int(cfg.clip_refine_min_frames)),
            max_frames=max(1, int(cfg.clip_refine_max_frames)),
            model=cfg.clip_refine_model,
            temperature=0.1,
            min_event_duration_sec=float(cfg.min_event_duration_sec),
        )
        clip_refine_path = run_refine_events_from_files(events_path, clips_path, clip_cfg)
        clip_payload = json.loads(Path(clip_refine_path).read_text(encoding="utf-8"))
        refined_events = list(clip_payload.get("events", []))

    appearance_path: Path | None = None
    if run_appearance:
        appearance_path = pipeline_out / f"{stem}_appearance_refined.json"
        app_cfg = AppearanceRefinementConfig.from_env(
            model=cfg.appearance_model or AppearanceRefinementConfig.from_env().model,
            crops_per_app=max(1, int(cfg.crops_per_track)),
            max_entities=max(0, int(cfg.max_tracks)),
            cache_path=appearance_path,
            force=bool(cfg.force),
        )
        refined = run_appearance_refinement_for_events(
            video_path=video,
            events=events,
            video_id=stem,
            camera_id=cam,
            config=app_cfg,
            base_events=refined_events,
        )
        refined_events = list(refined.get("events", []))

    scene_profile_path: Path | None = None
    if run_scene:
        scene_profile_path = pipeline_out / f"{stem}_scene_profile.json"
        scene_cfg = SceneProfileConfig.from_env(cache_path=scene_profile_path, force=bool(cfg.force))
        scene_payload = run_scene_profiles_for_pipeline(
            slot="single-video",
            camera_to_video={cam: video},
            camera_video_stems={cam: stem},
            config=scene_cfg,
        )
        scene_profile_payload = (scene_payload.get("per_camera") or {}).get(cam)

    seed_events, dropped_short = _filter_seed_events(events, float(cfg.min_event_duration_sec))
    flat = pipeline_events_to_vector_flat(
        video_id=stem,
        camera_id=cam,
        events=seed_events,
        global_entities=[],
        duration=_get_video_duration(video),
        refined_events=refined_events,
        scene_profile=scene_profile_payload,
        seed_mode="single_camera",
    )
    seed_path.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "video_id": stem,
        "camera_id": cam,
        "video_path": str(video),
        "events_path": str(events_path),
        "clips_path": str(clips_path),
        "clip_refine_path": str(clip_refine_path) if clip_refine_path else None,
        "appearance_refine_path": str(appearance_path) if appearance_path else None,
        "scene_profile_path": str(scene_profile_path) if scene_profile_path else None,
        "seed_path": str(seed_path),
        "raw_event_count": len(events),
        "clip_count": len(clips),
        "seed_event_count": len(seed_events),
        "dropped_short_event_count": dropped_short,
        "vector_event_count": len(flat.get("events", [])),
        "semantic_refine": bool(cfg.semantic_refine),
        "clip_refine": run_clip,
        "appearance_refine": run_appearance,
        "scene_profile": run_scene,
        "flat": flat,
        "skipped_existing": False,
    }
    manifest_path = pipeline_out / f"{stem}_single_camera_manifest.json"
    manifest_for_file = dict(manifest)
    manifest_for_file.pop("flat", None)
    manifest_path.write_text(json.dumps(manifest_for_file, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
