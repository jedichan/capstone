"""Video pipeline service — wraps multi-camera coordinator and DB builders."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _PROJECT_ROOT / "agent"
for _p in (str(_PROJECT_ROOT), str(_AGENT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def ingest_from_seeds(
    seeds_dir: str,
    namespace: str,
    output_base: str,
    reset_existing: bool = True,
) -> dict[str, Any]:
    """Build SQLite + Chroma databases from pre-generated events_vector_flat files.

    Args:
        seeds_dir: Directory containing *_events_vector_flat.json files.
        namespace: Chroma collection namespace prefix.
        output_base: Directory where sqlite / chroma will be written.
        reset_existing: Whether to drop and recreate existing collections.

    Returns:
        Dict with sqlite_path, chroma_path, counts, elapsed_ms.
    """
    from agent.db.chroma_builder import ChromaBuildConfig, ChromaIndexBuilder
    from agent.db.sqlite_builder import SQLiteBuildConfig, SQLiteDatabaseBuilder
    from agent.db.config import (
        CHROMA_CHILD_SUFFIX,
        CHROMA_EVENT_SUFFIX,
        CHROMA_GLOBAL_ENTITY_SUFFIX,
        CHROMA_PARENT_SUFFIX,
    )

    t0 = time.perf_counter()
    seeds_path = Path(seeds_dir)
    seed_files = sorted(seeds_path.glob("*_events_vector_flat.json"))
    if not seed_files:
        raise ValueError(f"No *_events_vector_flat.json files found in seeds_dir '{seeds_dir}'")

    out_dir = Path(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = out_dir / "episodic_events.sqlite"
    chroma_path = out_dir / "chroma"

    child_col = f"{namespace}_{CHROMA_CHILD_SUFFIX}"
    parent_col = f"{namespace}_{CHROMA_PARENT_SUFFIX}"
    event_col = f"{namespace}_{CHROMA_EVENT_SUFFIX}"
    global_entity_col = f"{namespace}_{CHROMA_GLOBAL_ENTITY_SUFFIX}"

    sqlite_result = SQLiteDatabaseBuilder(
        SQLiteBuildConfig(
            db_path=sqlite_path,
            reset_existing=reset_existing,
            generate_init_prompt=False,
        )
    ).build(seed_files=seed_files)

    chroma_result = ChromaIndexBuilder(
        ChromaBuildConfig(
            chroma_path=chroma_path,
            child_collection=child_col,
            parent_collection=parent_col,
            event_collection=event_col,
            global_entity_collection=global_entity_col,
            reset_existing=reset_existing,
        )
    ).build(seed_files=seed_files)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "status": "ok",
        "namespace": namespace,
        "sqlite_path": str(sqlite_path),
        "chroma_path": str(chroma_path),
        "seed_file_count": len(seed_files),
        "sqlite_rows": sqlite_result.get("inserted_rows", 0),
        "chroma_child_records": chroma_result.get("child_record_count", 0),
        "chroma_ge_records": chroma_result.get("global_entity_record_count", 0),
        "elapsed_ms": elapsed_ms,
    }


def ingest_from_single_video(
    video_path: str,
    namespace: str,
    output_base: str,
    camera_id: str | None = None,
    conf: float = 0.35,
    iou: float = 0.35,
    target_classes: list[str] | tuple[str, ...] | None = None,
    clip_refine: bool = False,
    clip_refine_model: str = "qwen3-vl-flash",
    appearance_refine: bool = False,
    scene_profile: bool = False,
    reset_existing: bool = True,
) -> dict[str, Any]:
    """Run single-camera pipeline (no Re-ID / cross-camera matching) and build databases.

    Args:
        video_path: Absolute path to the video file.
        namespace: Chroma collection namespace prefix.
        output_base: Directory for pipeline cache, seeds, sqlite, chroma.
        camera_id: Camera ID string; inferred from filename if None.
        conf: YOLO detection confidence threshold.
        iou: YOLO tracking IoU threshold.
        clip_refine: Run LLM clip semantic refinement (event text).
        target_classes: YOLO class names to track. Defaults to the pipeline default.
        appearance_refine: Run VLM appearance refinement (colour/clothing).
        scene_profile: Run scene profile generation.
        reset_existing: Whether to reset existing DB.

    Returns:
        Same shape as ingest_from_seeds plus single-camera manifest fields.
    """
    from video.factory.single_camera_pipeline import (
        SingleCameraPipelineConfig,
        run_single_camera_semantic_pipeline,
    )

    t0 = time.perf_counter()
    out_dir = Path(output_base)
    pipeline_out = out_dir / "pipeline_cache"
    seeds_dir = out_dir / "seeds"
    pipeline_out.mkdir(parents=True, exist_ok=True)
    seeds_dir.mkdir(parents=True, exist_ok=True)

    cfg = SingleCameraPipelineConfig(
        conf=conf,
        iou=iou,
        target_classes=tuple(target_classes) if target_classes else None,
        clip_refine=clip_refine,
        clip_refine_model=clip_refine_model,
        appearance_refine=appearance_refine,
        scene_profile=scene_profile,
        force=reset_existing,
    )

    manifest = run_single_camera_semantic_pipeline(
        video_path=video_path,
        out_dir=pipeline_out,
        seed_out_dir=seeds_dir,
        camera_id=camera_id,
        config=cfg,
    )

    db_result = ingest_from_seeds(
        seeds_dir=str(seeds_dir),
        namespace=namespace,
        output_base=str(out_dir / "runtime"),
        reset_existing=reset_existing,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        **db_result,
        "video_id": manifest["video_id"],
        "camera_id": manifest["camera_id"],
        "raw_event_count": manifest.get("raw_event_count", 0),
        "seed_event_count": manifest.get("seed_event_count", 0),
        "clip_refine": manifest.get("clip_refine", False),
        "appearance_refine": manifest.get("appearance_refine", False),
        "scene_profile": manifest.get("scene_profile", False),
        "pipeline_cache_path": str(pipeline_out),
        "elapsed_ms": elapsed_ms,
    }


def ingest_from_videos(
    camera_videos: dict[str, str],
    namespace: str,
    output_base: str,
    conf: float = 0.40,
    iou: float = 0.40,
    target_classes: list[str] | tuple[str, ...] | None = None,
    reid_device: str = "cpu",
    refine: bool = False,
    refine_model: str = "qwen3-vl-flash",
    appearance_refine: bool = False,
    campus_topology: bool = False,
    reset_existing: bool = True,
) -> dict[str, Any]:
    """Run full multi-camera pipeline and build databases.

    Args:
        camera_videos: {"G329": "/path/to/G329.avi", ...}
        namespace: Chroma collection namespace prefix.
        output_base: Directory for pipeline cache, seeds, sqlite, chroma.
        conf: YOLO detection confidence threshold.
        iou: YOLO tracking IoU threshold.
        target_classes: YOLO class names to track. Defaults to person/car when omitted.
        reid_device: Device for OSNet Re-ID embedder.
        refine: Whether to run LLM event refinement (Phase 5).
        refine_model: LLM model name for refinement.
        campus_topology: If True, inject the MEVID campus warm-start topology prior
            (school G299/G328/G330/G419/G420 ↔ bus G506/G508).  Replaces the default
            cold-start linear-decay heuristic with a GMM fitted on known transit-time
            ranges, so implausibly fast or slow cross-zone jumps are penalised.
            Only meaningful for MEVID campus footage; leave False for other datasets.
        reset_existing: Whether to reset existing DB.

    Returns:
        Same shape as ingest_from_seeds plus pipeline_cache_path.
    """
    import json
    from video.factory.multi_camera_coordinator import (
        run_multi_camera_pipeline,
        multi_camera_output_to_dict,
    )
    from video.core.schema.multi_camera import CrossCameraConfig

    t0 = time.perf_counter()
    out_dir = Path(output_base)
    cache_dir = out_dir / "pipeline_cache"
    seeds_dir = out_dir / "seeds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # Build topology prior (warm-start) if requested
    topology_prior = None
    if campus_topology:
        from video.core.models.camera_topology import CameraTopologyPrior
        topology_prior = CameraTopologyPrior.mevid_campus(
            cameras=list(camera_videos.keys()),
            max_transit_sec=300.0,
        )

    # Stage 1-4: multi-camera pipeline
    config = CrossCameraConfig(
        max_transition_sec=180.0,
        embedding_threshold=0.70,
        cross_camera_min_score=0.72,
        cross_camera_overlap_tolerance_sec=2.0,
        topology_weight_reid=0.65,
        topology_weight_topo=0.35,
        max_global_entity_cameras=4,
        max_global_entity_tracks=4,
        person_only=True,
    )
    pipeline_output = run_multi_camera_pipeline(
        camera_videos=camera_videos,
        config=config,
        reid_device=reid_device,
        target_classes=list(target_classes) if target_classes else None,
        conf=conf,
        iou=iou,
        topology_prior=topology_prior,
    )
    pipeline_dict = multi_camera_output_to_dict(pipeline_output)
    cache_file = cache_dir / "pipeline_output.json"
    cache_file.write_text(
        json.dumps(pipeline_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Stage 5 (optional): LLM text refinement
    # NOTE: refine_multi_camera_output returns a per-camera dict, NOT a MultiCameraOutput.
    # Do NOT reassign pipeline_output — patch pipeline_dict["events"] in-place instead.
    if refine:
        from video.factory.refinement_runner import refine_multi_camera_output, RefineEventsConfig
        refined_per_cam: dict = refine_multi_camera_output(
            pipeline_output,
            config=RefineEventsConfig(mode="vector", model=refine_model),
        )
        # Build lookup: (camera_id, track_id) → refined event fields
        refined_lookup: dict[tuple[str, str], dict] = {}
        for _cam_id, _cam_data in refined_per_cam.items():
            for _rev in _cam_data.get("events", []):
                _tid = str(_rev.get("track_id") or "")
                refined_lookup[(_cam_id, _tid)] = _rev
        # Patch event_text / event_summary into pipeline_dict events
        for ev in pipeline_dict.get("events", []):
            _key = (str(ev.get("camera_id") or ""), str(ev.get("track_id") or ""))
            _rev = refined_lookup.get(_key)
            if _rev:
                ev["event_text"] = _rev.get("event_text") or ev.get("event_text", "")
                ev["description_for_llm"] = _rev.get("event_text") or ev.get("description_for_llm", "")
                if _rev.get("entity_hint"):
                    ev.setdefault("global_entity_id", _rev["entity_hint"])
        (cache_dir / "pipeline_refined.json").write_text(
            json.dumps({"refined_per_cam": refined_per_cam}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Stage 6 (optional): VLM appearance refinement (crop-based colour/clothing)
    appearance_result: dict = {}
    if appearance_refine:
        from collections import defaultdict
        from video.factory.appearance_refinement_runner import (
            AppearanceRefinementConfig,
            run_appearance_refinement_for_pipeline,
        )
        cam_events_dict: dict = defaultdict(list)
        for ev in pipeline_dict.get("events", []):
            cam = ev.get("camera_id")
            if cam:
                cam_events_dict[cam].append(ev)
        pipeline_with_per_cam = dict(pipeline_dict)
        pipeline_with_per_cam["per_camera"] = [
            {"camera_id": cam, "events": evs}
            for cam, evs in cam_events_dict.items()
        ]
        meta_cameras: dict = (pipeline_dict.get("meta") or {}).get("cameras") or {}
        camera_video_stems = {cam: Path(p).name for cam, p in meta_cameras.items()}
        appearance_result = run_appearance_refinement_for_pipeline(
            slot=namespace,
            pipeline=pipeline_with_per_cam,
            camera_to_video=meta_cameras,
            camera_video_stems=camera_video_stems,
            config=AppearanceRefinementConfig.from_env(
                model=refine_model,
                cache_path=cache_dir / "appearance_refined.json",
                force=reset_existing,
                max_apps_per_entity=4,
                crops_per_app=2,
            ),
        )

    # Convert pipeline output → events_vector_flat seeds
    seed_files = _pipeline_output_to_seeds(
        pipeline_output, pipeline_dict, seeds_dir, namespace,
        appearance_result=appearance_result,
    )

    # Build DB
    db_result = ingest_from_seeds(
        seeds_dir=str(seeds_dir),
        namespace=namespace,
        output_base=str(out_dir / "runtime"),
        reset_existing=reset_existing,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {**db_result, "pipeline_cache_path": str(cache_file), "elapsed_ms": elapsed_ms}


def _pipeline_output_to_seeds(
    pipeline_output: Any,
    pipeline_dict: dict[str, Any],
    seeds_dir: Path,
    namespace: str,
    appearance_result: dict | None = None,
) -> list[Path]:
    """Convert MultiCameraOutput → per-camera events_vector_flat.json seed files.

    Supports both legacy format (top-level cameras/merged_events/per_camera) and
    new format (meta.cameras / events flat list with camera_id per event).
    """
    import json

    global_entities: list[dict] = pipeline_dict.get("global_entities") or []

    # Build GE lookup — supports both old format (camera_ids, track_ids fields) and
    # new format (appearances list with camera_id / track_id per entry).
    ge_lookup: dict[str, dict] = {}
    for ge in global_entities:
        geid = ge.get("global_entity_id")
        if not geid:
            continue
        # Normalise appearances → list of {camera_id, track_id, ...}
        appearances: list[dict] = ge.get("appearances") or []
        # Derive camera_ids list from appearances if not already present
        if appearances and not ge.get("camera_ids"):
            ge = dict(ge)  # shallow copy so we don't mutate original
            ge["camera_ids"] = [a["camera_id"] for a in appearances if a.get("camera_id")]
        ge_lookup[geid] = ge

    # Support both old (merged_events) and new (events) key names
    merged_events: list[dict] = (
        pipeline_dict.get("merged_events")
        or pipeline_dict.get("events")
        or []
    )

    # Group events by camera
    cam_events: dict[str, list[dict]] = {}
    for ev in merged_events:
        cam = str(ev.get("camera_id") or "").strip()
        if cam:
            cam_events.setdefault(cam, []).append(ev)

    # Support both old (top-level cameras) and new (meta.cameras) locations
    cameras: dict[str, str] = (
        pipeline_dict.get("cameras")
        or (pipeline_dict.get("meta") or {}).get("cameras")
        or {}
    )

    # Build appearance lookup: (camera_id, entity_hint) → appearance event
    app_lookup: dict[tuple[str, str], dict] = {}
    if appearance_result:
        per_cam_app = appearance_result.get("per_camera", {})
        for _cam_id, _cam_data in per_cam_app.items():
            for _app_ev in _cam_data.get("events", []):
                _hint = str(_app_ev.get("entity_hint") or "")
                if _hint:
                    app_lookup[(str(_cam_id), _hint)] = _app_ev

    seed_paths: list[Path] = []
    for cam_id, video_path in cameras.items():
        video_id = Path(video_path).name
        events_for_cam = cam_events.get(cam_id, [])

        flat_events: list[dict] = []
        for ev in events_for_cam:
            entity_hint = str(ev.get("global_entity_id") or ev.get("track_id") or "").strip()
            ge = ge_lookup.get(entity_hint, {})
            ge_camera_ids = ge.get("camera_ids") or []
            cross_camera_ids = [c for c in ge_camera_ids if c != cam_id] if ge_camera_ids else []

            # Merge appearance refinement data if available
            app_ev = app_lookup.get((cam_id, entity_hint), {})

            flat_ev = {
                "video_id": video_id,
                "camera_id": cam_id,
                "track_id": str(ev.get("track_id") or ""),
                "entity_hint": entity_hint,
                "start_time": ev.get("start_time"),
                "end_time": ev.get("end_time"),
                "clip_start_sec": ev.get("start_time"),
                "clip_end_sec": ev.get("end_time"),
                "object_type": str(ev.get("class_name") or "person").lower(),
                "object_color": str(app_ev.get("object_color") or ge.get("object_color") or ""),
                "object_color_en": str(app_ev.get("object_color") or ge.get("object_color") or ""),
                "scene_zone_en": str(ev.get("scene_zone") or ""),
                "event_type": str(ev.get("event_type") or ""),
                "appearance_notes": str(app_ev.get("appearance_notes") or ge.get("appearance_notes") or ""),
                "appearance_notes_en": str(app_ev.get("appearance_notes") or ge.get("appearance_notes") or ""),
                "event_text_en": str(app_ev.get("event_text") or ev.get("description_for_llm") or ev.get("event_text") or ""),
                "event_summary_en": str(app_ev.get("event_text") or ev.get("description_for_llm") or ""),
                "keywords": list(dict.fromkeys(
                    (ev.get("keywords") or []) + (app_ev.get("keywords") or []) + (ge.get("keywords") or [])
                ))[:16],
                "cross_camera_ids": cross_camera_ids,
            }
            flat_events.append(flat_ev)

        safe_name = f"{namespace}_{cam_id}_events_vector_flat.json"
        seed_path = seeds_dir / safe_name
        seed_path.write_text(
            json.dumps({"video_id": video_id, "events": flat_events}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        seed_paths.append(seed_path)

    return seed_paths
