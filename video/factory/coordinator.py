"""
Offline pipeline orchestration: video → event/clip JSON → optional LLM refinement.

- In-process: run_video_to_events, run_refine_events
- Persistence / JSON: video.factory.pipeline_outputs (video_events_as_json_dicts, refined_events_as_dict, etc.)
- CLI: cli_run_video_events, cli_run_refine_events, or python -m video.factory.coordinator video|refine ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from video.common.paths import pipeline_output_dir
from video.factory.appearance_refinement_runner import (
    AppearanceRefinementConfig,
    run_appearance_refinement_for_events,
)
from video.factory.processors.event_track_pipeline import run_pipeline, save_pipeline_output
from video.factory.refinement_runner import RefineEventsConfig, run_refine_events_from_files
from video.factory.scene_profile_runner import SceneProfileConfig, run_scene_profiles_for_pipeline
from video.factory.single_camera_pipeline import (
    DEFAULT_SINGLE_CAMERA_CLASSES,
    SingleCameraPipelineConfig,
    run_single_camera_semantic_pipeline,
)


def run_video_to_events(
    video_path: str,
    out_dir: str | Path | None = None,
    save: bool = True,
    **run_kwargs: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, float]],
    dict[str, Any],
    tuple[Path, Path] | None,
]:
    """
    Run full tracking + event segmentation. When save=True, writes *_events.json / *_clips.json;
    default out_dir is repo root pipeline_output/.

    Returns: (events, clip_segments, meta, saved_paths or None)
    """
    events, clip_segments, meta = run_pipeline(video_path, **run_kwargs)
    saved: tuple[Path, Path] | None = None
    if save:
        od = Path(out_dir) if out_dir is not None else pipeline_output_dir()
        saved = save_pipeline_output(events, clip_segments, meta, od)
    return events, clip_segments, meta, saved


def run_refine_events(
    events_json: str | Path,
    clips_json: str | Path,
    config: RefineEventsConfig | None = None,
) -> Path:
    """LLM-refine pipeline JSON; returns path to output JSON."""
    return run_refine_events_from_files(events_json, clips_json, config)


def run_refine_appearance_events(
    events_json: str | Path,
    config: AppearanceRefinementConfig | None = None,
    base_json: str | Path | None = None,
) -> Path:
    """Crop-refine person appearance from one *_events.json file.

    If ``base_json`` is given (path to a *_events_vector_flat.json), appearance
    fields are patched into those events rather than creating new ones, so
    scene_zone, event_text and bbox from the prior refine pass are preserved.
    When omitted the function auto-detects a sibling *_events_vector_flat.json.
    """
    events_path = Path(events_json)
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    video_path = meta.get("video_path")
    if not video_path:
        raise ValueError(f"{events_path} does not contain meta.video_path")
    video_id = Path(video_path).name
    camera_id = meta.get("camera_id")
    cfg = config or AppearanceRefinementConfig.from_env()
    if cfg.cache_path is None:
        cfg.cache_path = events_path.with_name(events_path.stem + "_appearance_refined.json")

    # Load base events for merging
    base_events: list[dict[str, Any]] | None = None
    if base_json is None:
        # Auto-detect sibling vector_flat file
        auto = events_path.with_name(events_path.stem + "_vector_flat.json")
        if auto.exists():
            base_json = auto
    if base_json is not None:
        base_payload = json.loads(Path(base_json).read_text(encoding="utf-8"))
        base_events = list(base_payload.get("events", []))

    result = run_appearance_refinement_for_events(
        video_path=video_path,
        events=list(payload.get("events", [])),
        video_id=video_id,
        camera_id=camera_id,
        config=cfg,
        base_events=base_events,
    )
    out_path = cfg.cache_path
    assert out_path is not None
    if not out_path.exists():
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _add_video_cli_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("video", nargs="?", default="VIRAT_S_000200_00_000100_000171.mp4", help="Path to input video")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: pipeline_output/ under repo root)",
    )
    p.add_argument(
        "--tracker",
        type=str,
        default="botsort_reid",
        help="botsort_reid | botsort | bytetrack | path to a .yaml tracker config",
    )
    p.add_argument("--model", "-m", type=str, default="11m", help="YOLO weights (default yolo11m in _model/)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.25, help="Detection NMS IoU threshold")
    p.add_argument(
        "--classes",
        type=str,
        default="person,car",
        help="Comma-separated class filter (default: person + car; bags/items remain VLM appearance attributes)",
    )
    p.add_argument("--save-video", action="store_true", help="Write annotated video with boxes + track_id")
    p.add_argument("--save-video-path", type=str, default=None, help="Output path for annotated video")


def _parse_class_list(raw: str) -> tuple[str, ...] | None:
    if not raw.strip():
        return None
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _run_video_cli_namespace(args: argparse.Namespace) -> None:
    out = Path(args.out_dir) if args.out_dir else pipeline_output_dir()
    target_classes = [x.strip() for x in args.classes.split(",") if x.strip()] if args.classes else None
    print(
        f"Running: {args.video} | model={args.model} conf={args.conf} iou={args.iou} | tracker={args.tracker}"
    )
    events, clip_segments, meta, paths = run_video_to_events(
        args.video,
        out_dir=out,
        save=True,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        motion_threshold=5.0,
        min_clip_duration=1.0,
        max_static_duration=30.0,
        tracker=args.tracker,
        target_classes=target_classes,
        save_annotated_video=bool(args.save_video),
        annotated_video_path=args.save_video_path,
    )
    assert paths is not None
    events_path, clips_path = paths
    print(f"Meta: {json.dumps(meta, ensure_ascii=False, indent=2)}")
    print(f"Events saved: {events_path} ({len(events)} rows)")
    print(f"Clip segments saved: {clips_path} ({len(clip_segments)} segments)")
    print("\nFirst 3 events:")
    print(json.dumps(events[:3], ensure_ascii=False, indent=2))
    print("\nFirst 5 clip segments (start_sec, end_sec):")
    print(json.dumps(clip_segments[:5], ensure_ascii=False, indent=2))


def cli_run_video_events(argv: Sequence[str] | None = None) -> None:
    """CLI: same as legacy pipeline_video_events.py (plus --out-dir)."""
    parser = argparse.ArgumentParser(description="Video → YOLO+track → events + clip time ranges")
    _add_video_cli_args(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    _run_video_cli_namespace(args)


def _add_refine_cli_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--events", required=True, help="Path to *_events.json")
    p.add_argument("--clips", required=True, help="Path to *_clips.json")
    p.add_argument(
        "--mode",
        type=str,
        default="vector",
        choices=["full", "vector"],
        help="full=rich structure; vector=minimal retrieval events (default)",
    )
    p.add_argument("--clip-index", type=int, default=None, help="Process only one clip segment index")
    p.add_argument("--num-frames", type=int, default=0,
                   help="Fixed frame count (0=adaptive via --frames-per-sec)")
    p.add_argument("--frames-per-sec", type=float, default=0.1,
                   help="Adaptive: frames per second (default 0.1 ≈ one frame per 10s)")
    p.add_argument("--min-frames", type=int, default=6,
                   help="Adaptive: minimum frames (default 6)")
    p.add_argument("--max-frames", type=int, default=48,
                   help="Adaptive: maximum frames (default 48; caps LLM cost)")
    p.add_argument("--model", type=str, default="qwen3-vl-flash", help="OpenAI multimodal model name")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-time-adjust-sec", type=float, default=0.5)
    p.add_argument("--merge-location-iou", type=float, default=0.9)
    p.add_argument("--merge-center-dist-px", type=float, default=30.0)
    p.add_argument("--merge-location-norm-diff", type=float, default=0.10)
    p.add_argument("--min-event-duration-sec", type=float, default=1.0, help="Drop events shorter than this (seconds) before LLM")


def _add_appearance_cli_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--events", required=True, help="Path to *_events.json")
    p.add_argument("--base", type=str, default=None, help="Path to *_events_vector_flat.json to merge into (auto-detected if omitted)")
    p.add_argument("--model", type=str, default=None, help="VLM model for crop appearance refinement")
    p.add_argument("--max-tracks", type=int, default=0, help="Debug cap for person tracks (0 = all)")
    p.add_argument("--crops-per-track", type=int, default=2)
    p.add_argument("--force", action="store_true", help="Re-run even if output exists")
    p.add_argument("--out", type=str, default=None, help="Output JSON path")


def _add_semantic_cli_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--events", required=True, help="Path to *_events.json")
    p.add_argument("--clips", required=True, help="Path to *_clips.json")
    p.add_argument("--camera", default="", help="Camera id override")
    p.add_argument("--clip-model", default="qwen3-vl-flash", help="VLM model for clip/slice event refinement")
    p.add_argument("--appearance-model", default=None, help="VLM model for crop appearance refinement")
    p.add_argument("--min-frames", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=24)
    p.add_argument("--crops-per-track", type=int, default=2)
    p.add_argument("--max-tracks", type=int, default=0)
    p.add_argument("--min-event-duration-sec", type=float, default=1.0)
    p.add_argument("--force", action="store_true")


def _add_single_camera_cli_args(p: argparse.ArgumentParser) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument("video", help="Path to one input video")
    p.add_argument("--camera", default="", help="Camera id override, e.g. G423")
    p.add_argument("--video-id", default="", help="Video id/stem override used for output file names")
    p.add_argument(
        "--out-dir",
        default=str(repo_root / "results" / "single_camera_pipeline"),
        help="Where pipeline/refine/manifest JSON files are saved",
    )
    p.add_argument(
        "--seed-out-dir",
        default=str(repo_root / "agent" / "test" / "data" / "events_vector_flat"),
        help="Where the final *_events_vector_flat.json seed is saved",
    )
    p.add_argument("--model", default="11m")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.35)
    p.add_argument("--tracker", default="botsort_reid")
    p.add_argument(
        "--classes",
        default=",".join(DEFAULT_SINGLE_CAMERA_CLASSES),
        help="Comma-separated class filter. Default is person + car.",
    )
    p.add_argument("--min-event-duration-sec", type=float, default=1.0)
    p.add_argument(
        "--semantic-refine",
        action="store_true",
        help="Run clip/slice refine + crop appearance refine + scene profile",
    )
    p.add_argument("--clip-refine", action="store_true", help="Run only clip/slice event refinement")
    p.add_argument("--appearance-refine", action="store_true", help="Run only crop appearance refinement")
    p.add_argument("--scene-profile", action="store_true", help="Run only camera scene profiling")
    p.add_argument("--clip-refine-model", default="qwen3-vl-flash")
    p.add_argument("--clip-refine-min-frames", type=int, default=4)
    p.add_argument("--clip-refine-max-frames", type=int, default=24)
    p.add_argument("--appearance-model", default=None)
    p.add_argument("--crops-per-track", type=int, default=2)
    p.add_argument("--max-tracks", type=int, default=0)
    p.add_argument("--force", action="store_true")


def _run_refine_cli_namespace(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    cfg = RefineEventsConfig(
        mode=args.mode,
        clip_index=args.clip_index,
        num_frames=int(args.num_frames),
        frames_per_sec=float(args.frames_per_sec),
        min_frames=int(args.min_frames),
        max_frames=int(args.max_frames),
        model=args.model,
        temperature=float(args.temperature),
        max_time_adjust_sec=float(args.max_time_adjust_sec),
        merge_location_iou_threshold=float(args.merge_location_iou),
        merge_center_dist_px=float(args.merge_center_dist_px),
        merge_location_norm_diff=float(args.merge_location_norm_diff),
        min_event_duration_sec=float(args.min_event_duration_sec),
    )
    out_path = run_refine_events_from_files(args.events, args.clips, cfg)
    print(f"refined events saved to: {out_path}")


def cli_run_refine_events(argv: Sequence[str] | None = None) -> None:
    """CLI: same as legacy langchain_refine_events.py (loads dotenv first)."""
    parser = argparse.ArgumentParser(description="LangChain + ChatGPT multimodal refinement for events")
    _add_refine_cli_args(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    _run_refine_cli_namespace(args)


def _run_appearance_cli_namespace(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    cfg = AppearanceRefinementConfig.from_env(
        model=args.model or AppearanceRefinementConfig.from_env().model,
        max_entities=int(args.max_tracks),
        crops_per_app=int(args.crops_per_track),
        force=bool(args.force),
        cache_path=Path(args.out) if args.out else None,
    )
    out_path = run_refine_appearance_events(args.events, cfg, base_json=args.base or None)
    print(f"appearance-refined events saved to: {out_path}")


def cli_run_refine_appearance(argv: Sequence[str] | None = None) -> None:
    """CLI: crop-based single-camera person appearance refinement."""
    parser = argparse.ArgumentParser(description="Crop-based appearance refinement for *_events.json")
    _add_appearance_cli_args(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    _run_appearance_cli_namespace(args)


def _run_semantic_cli_namespace(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    events_path = Path(args.events)
    clips_path = Path(args.clips)
    events_payload = json.loads(events_path.read_text(encoding="utf-8"))
    meta = events_payload.get("meta", {})
    video_path = meta.get("video_path")
    if not video_path:
        raise ValueError(f"{events_path} does not contain meta.video_path")
    video_path = str(video_path)
    camera_id = args.camera or meta.get("camera_id") or "CAM"
    stem = Path(video_path).stem

    print("[semantic-refine] Running clip/slice event refinement ...")
    clip_cfg = RefineEventsConfig(
        mode="vector",
        frames_per_sec=0.1,
        min_frames=max(1, int(args.min_frames)),
        max_frames=max(1, int(args.max_frames)),
        model=args.clip_model,
        temperature=0.1,
        min_event_duration_sec=float(args.min_event_duration_sec),
    )
    clip_path = run_refine_events_from_files(events_path, clips_path, clip_cfg)
    clip_payload = json.loads(Path(clip_path).read_text(encoding="utf-8"))
    base_events = list(clip_payload.get("events", []))
    print(f"[semantic-refine] Clip refined events={len(base_events)}: {clip_path}")

    print("[semantic-refine] Running crop appearance refinement ...")
    appearance_path = events_path.with_name(events_path.stem + "_semantic_refined.json")
    app_cfg = AppearanceRefinementConfig.from_env(
        model=args.appearance_model or AppearanceRefinementConfig.from_env().model,
        max_entities=max(0, int(args.max_tracks)),
        crops_per_app=max(1, int(args.crops_per_track)),
        cache_path=appearance_path,
        force=bool(args.force),
    )
    appearance = run_appearance_refinement_for_events(
        video_path=video_path,
        events=list(events_payload.get("events", [])),
        video_id=stem,
        camera_id=camera_id,
        config=app_cfg,
        base_events=base_events,
    )
    print(f"[semantic-refine] Appearance+event refined events={len(appearance.get('events', []))}: {appearance_path}")

    print("[semantic-refine] Running scene profile ...")
    scene_path = events_path.with_name(events_path.stem + "_scene_profile.json")
    scene_cfg = SceneProfileConfig.from_env(cache_path=scene_path, force=bool(args.force))
    scene = run_scene_profiles_for_pipeline(
        slot="single-video",
        camera_to_video={camera_id: video_path},
        camera_video_stems={camera_id: stem},
        config=scene_cfg,
    )
    print(f"[semantic-refine] Scene profile: {scene_path}")

    manifest_path = events_path.with_name(events_path.stem + "_semantic_manifest.json")
    manifest = {
        "video_id": stem,
        "camera_id": camera_id,
        "clip_refine_path": str(clip_path),
        "semantic_refine_path": str(appearance_path),
        "scene_profile_path": str(scene_path),
        "scene_profile_cameras": list((scene.get("per_camera") or {}).keys()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[semantic-refine] Manifest: {manifest_path}")


def _run_single_camera_cli_namespace(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    cfg = SingleCameraPipelineConfig(
        model=args.model,
        conf=float(args.conf),
        iou=float(args.iou),
        tracker=args.tracker,
        target_classes=_parse_class_list(args.classes),
        min_event_duration_sec=float(args.min_event_duration_sec),
        semantic_refine=bool(args.semantic_refine),
        clip_refine=bool(args.clip_refine),
        appearance_refine=bool(args.appearance_refine),
        scene_profile=bool(args.scene_profile),
        clip_refine_model=args.clip_refine_model,
        clip_refine_min_frames=int(args.clip_refine_min_frames),
        clip_refine_max_frames=int(args.clip_refine_max_frames),
        appearance_model=args.appearance_model,
        crops_per_track=int(args.crops_per_track),
        max_tracks=int(args.max_tracks),
        force=bool(args.force),
    )
    result = run_single_camera_semantic_pipeline(
        args.video,
        out_dir=args.out_dir,
        seed_out_dir=args.seed_out_dir,
        camera_id=args.camera or None,
        video_id=args.video_id or None,
        config=cfg,
    )
    print("[single-camera] Done")
    print(f"  video_id: {result['video_id']}")
    print(f"  camera_id: {result['camera_id']}")
    print(f"  raw events: {result.get('raw_event_count', 'existing')}")
    print(f"  vector events: {result.get('vector_event_count', 0)}")
    if result.get("dropped_short_event_count") is not None:
        print(f"  dropped short events: {result.get('dropped_short_event_count')}")
    if result.get("manifest_path"):
        print(f"  manifest: {result['manifest_path']}")
    print(f"  vector seed: {result['seed_path']}")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry: python -m video.factory.coordinator video|refine|appearance|semantic-refine|single-camera ..."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Offline video pipeline: video (track+events) or refine (LLM)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_v = sub.add_parser("video", help="YOLO+track → events + clips JSON")
    _add_video_cli_args(p_v)
    p_r = sub.add_parser("refine", help="Refine *_events.json + *_clips.json")
    _add_refine_cli_args(p_r)
    p_a = sub.add_parser("appearance", help="Crop-refine person appearance from *_events.json")
    _add_appearance_cli_args(p_a)
    p_s = sub.add_parser("semantic-refine", help="Run clip refine + crop appearance + scene profile")
    _add_semantic_cli_args(p_s)
    p_sc = sub.add_parser("single-camera", help="Run one-video single-camera pipeline and write vector-flat seed")
    _add_single_camera_cli_args(p_sc)
    args = parser.parse_args(argv)
    if args.cmd == "video":
        _run_video_cli_namespace(args)
    elif args.cmd == "refine":
        _run_refine_cli_namespace(args)
    elif args.cmd == "appearance":
        _run_appearance_cli_namespace(args)
    elif args.cmd == "semantic-refine":
        _run_semantic_cli_namespace(args)
    else:
        _run_single_camera_cli_namespace(args)


if __name__ == "__main__":
    main()
