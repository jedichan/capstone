"""
Generate one MEVID single-video vector seed from the video pipeline.

This is useful when evaluating one camera/time segment as a single-camera
system, without relying on the multi-camera slot cache.

Example:
    python scripts/generate_mevid_single_video_vector_flat.py --slot 13-50 --camera G421 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from scripts.generate_mevid_vector_flat import (  # noqa: E402
    OUT_DIR,
    SLOT_CAMERAS,
    _get_video_duration,
    pipeline_events_to_vector_flat,
)
from video.factory.appearance_refinement_runner import (  # noqa: E402
    AppearanceRefinementConfig,
    run_appearance_refinement_for_events,
)
from video.factory.refinement_runner import RefineEventsConfig, run_refine_events_from_files  # noqa: E402
from video.factory.scene_profile_runner import SceneProfileConfig, run_scene_profiles_for_pipeline  # noqa: E402
from video.factory.coordinator import run_video_to_events  # noqa: E402


def _resolve_video(args: argparse.Namespace) -> tuple[Path, str, str]:
    if args.video_path:
        video_path = Path(args.video_path).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        stem = video_path.stem
        camera_id = args.camera.strip().upper() if args.camera else _camera_from_stem(stem)
        return video_path, stem, camera_id

    slot = args.slot.strip()
    camera_id = args.camera.strip().upper()
    if not slot or not camera_id:
        raise ValueError("Use either --video-path, or both --slot and --camera")
    if slot not in SLOT_CAMERAS:
        raise ValueError(f"Unknown slot {slot!r}. Available: {sorted(SLOT_CAMERAS)}")
    cam_map = SLOT_CAMERAS[slot]
    if camera_id not in cam_map:
        raise ValueError(f"Camera {camera_id!r} not in slot {slot}. Available: {sorted(cam_map)}")

    stem = cam_map[camera_id]
    video_path = Path(args.video_dir).expanduser().resolve() / f"{stem}.avi"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path, stem, camera_id


def _camera_from_stem(stem: str) -> str:
    parts = stem.split(".")
    for part in reversed(parts):
        if part.upper().startswith("G") and part[1:].isdigit():
            return part.upper()
    return "CAM"


def _parse_classes(raw: str) -> list[str] | None:
    if not raw.strip():
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one MEVID video and generate one vector-flat seed")
    parser.add_argument("--slot", default="13-50", help="MEVID slot used with --camera")
    parser.add_argument("--camera", default="", help="Camera id, e.g. G421")
    parser.add_argument("--video-path", default="", help="Direct path to one .avi/.mp4 file")
    parser.add_argument("--video-dir", default=str(ROOT / "_data" / "mevid_slots"))
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output seed directory")
    parser.add_argument(
        "--pipeline-out-dir",
        default=str(ROOT / "results" / "mevid_single_video_pipeline"),
        help="Where raw *_events.json and *_clips.json are saved",
    )
    parser.add_argument("--model", default="11m")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--tracker", default="botsort_reid")
    parser.add_argument(
        "--classes",
        default="person,car",
        help="Detection classes. Default is person + car; bags/items are handled as VLM appearance attributes.",
    )
    parser.add_argument(
        "--min-event-duration-sec",
        type=float,
        default=1.0,
        help="Drop shorter events from the vector seed only; raw pipeline JSON keeps all events",
    )
    parser.add_argument(
        "--appearance-refine",
        action="store_true",
        help="Run crop-based VLM appearance refinement before writing the vector seed",
    )
    parser.add_argument(
        "--semantic-refine",
        action="store_true",
        help="Run the full semantic refinement bundle: clip refine + crop appearance + scene profile",
    )
    parser.add_argument(
        "--clip-refine",
        action="store_true",
        help="Run clip/slice-level VLM event refinement before crop appearance refinement",
    )
    parser.add_argument("--clip-refine-model", default="gpt-5.4-mini")
    parser.add_argument("--clip-refine-max-frames", type=int, default=24)
    parser.add_argument("--clip-refine-min-frames", type=int, default=4)
    parser.add_argument("--scene-profile", action="store_true", help="Run camera-level scene profiling")
    parser.add_argument("--appearance-model", default=None)
    parser.add_argument("--crops-per-track", type=int, default=2)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Overwrite existing seed/refine outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video_path, stem, camera_id = _resolve_video(args)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_path = out_dir / f"{stem}_events_vector_flat.json"
    if seed_path.exists() and not args.force:
        print(f"[single-video] Seed already exists: {seed_path}")
        print("               Use --force to rebuild it.")
        return

    pipeline_out = Path(args.pipeline_out_dir).expanduser().resolve()
    pipeline_out.mkdir(parents=True, exist_ok=True)

    print(f"[single-video] Running pipeline: {video_path.name} camera={camera_id}")
    events, clips, meta, saved = run_video_to_events(
        str(video_path),
        out_dir=pipeline_out,
        save=True,
        model_path=args.model,
        conf=float(args.conf),
        iou=float(args.iou),
        tracker=args.tracker,
        camera_id=camera_id,
        target_classes=_parse_classes(args.classes),
        motion_threshold=5.0,
        min_clip_duration=1.0,
        max_static_duration=30.0,
    )
    events_path, clips_path = saved or (None, None)
    print(f"[single-video] Raw events={len(events)} clips={len(clips)}")
    if events_path:
        print(f"[single-video] Raw events JSON: {events_path}")
    if clips_path:
        print(f"[single-video] Raw clips JSON : {clips_path}")

    refined_events: list[dict] | None = None
    scene_profile: dict | None = None
    run_clip_refine = bool(args.semantic_refine or args.clip_refine)
    run_appearance = bool(args.semantic_refine or args.appearance_refine)
    run_scene = bool(args.semantic_refine or args.scene_profile)

    if run_clip_refine:
        if not events_path or not clips_path:
            raise RuntimeError("clip refine requires saved *_events.json and *_clips.json")
        clip_refine_path = pipeline_out / f"{stem}_events_vector_flat.json"
        if clip_refine_path.exists() and not args.force:
            print(f"[single-video] Loading existing clip/slice refinement: {clip_refine_path}")
        else:
            print("[single-video] Running clip/slice event refinement ...")
            cfg = RefineEventsConfig(
                mode="vector",
                frames_per_sec=0.1,
                min_frames=max(1, int(args.clip_refine_min_frames)),
                max_frames=max(1, int(args.clip_refine_max_frames)),
                model=args.clip_refine_model,
                temperature=0.1,
                min_event_duration_sec=float(args.min_event_duration_sec),
            )
            clip_refine_path = run_refine_events_from_files(events_path, clips_path, cfg)
        clip_payload = json.loads(Path(clip_refine_path).read_text(encoding="utf-8"))
        refined_events = list(clip_payload.get("events", []))
        print(f"[single-video] Clip refined events={len(refined_events)}: {clip_refine_path}")

    if run_appearance:
        refine_path = pipeline_out / f"{stem}_appearance_refined.json"
        cfg = AppearanceRefinementConfig.from_env(
            model=args.appearance_model or AppearanceRefinementConfig.from_env().model,
            crops_per_app=max(1, int(args.crops_per_track)),
            max_entities=max(0, int(args.max_tracks)),
            cache_path=refine_path,
            force=bool(args.force),
        )
        print("[single-video] Running crop appearance refinement ...")
        refined = run_appearance_refinement_for_events(
            video_path=video_path,
            events=events,
            video_id=stem,
            camera_id=camera_id,
            config=cfg,
            base_events=refined_events,
        )
        refined_events = list(refined.get("events", []))
        print(f"[single-video] Appearance refined events={len(refined_events)}: {refine_path}")

    if run_scene:
        scene_path = pipeline_out / f"{stem}_scene_profile.json"
        cfg = SceneProfileConfig.from_env(cache_path=scene_path, force=bool(args.force))
        print("[single-video] Running scene profile ...")
        scene_payload = run_scene_profiles_for_pipeline(
            slot=args.slot or "single-video",
            camera_to_video={camera_id: video_path},
            camera_video_stems={camera_id: stem},
            config=cfg,
        )
        scene_profile = (scene_payload.get("per_camera") or {}).get(camera_id)
        print(f"[single-video] Scene profile: {scene_path}")

    seed_events = events
    if args.min_event_duration_sec > 0:
        before = len(seed_events)
        seed_events = [
            event for event in seed_events
            if float(event.get("end_time", event.get("start_time", 0.0)))
            - float(event.get("start_time", 0.0)) >= args.min_event_duration_sec
        ]
        print(
            "[single-video] Seed duration filter: "
            f"kept={len(seed_events)} dropped={before - len(seed_events)} "
            f"threshold={args.min_event_duration_sec:.2f}s"
        )

    duration = _get_video_duration(video_path)
    flat = pipeline_events_to_vector_flat(
        video_id=stem,
        camera_id=camera_id,
        events=seed_events,
        global_entities=[],
        duration=duration,
        refined_events=refined_events,
        scene_profile=scene_profile,
        seed_mode="single_camera",
    )
    seed_path.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[single-video] Vector seed written: {seed_path}")
    print(f"[single-video] Vector events={len(flat.get('events', []))}")


if __name__ == "__main__":
    main()
