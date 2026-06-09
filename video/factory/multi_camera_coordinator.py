"""Multi-camera cross-view orchestration: per-camera track → crops+Re-ID → match → merge."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from video.common.frames import PersonCrop, extract_person_crops
from video.core.models.reid_embedder import ReIDEmbedder
from video.core.schema.multi_camera import (
    CameraResult,
    CrossCameraConfig,
    GlobalEntity,
    MultiCameraOutput,
)
from video.factory.processors.analyzer import aggregate_tracks, slice_events
from video.factory.processors.cross_camera_matcher import match_across_cameras
from video.factory.processors.vision import resolve_model, run_yolo_track_on_video

logger = logging.getLogger(__name__)


def _same_camera_gap_sec(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_s, a_e = float(a["start_time"]), float(a["end_time"])
    b_s, b_e = float(b["start_time"]), float(b["end_time"])
    if min(a_e, b_e) >= max(a_s, b_s):
        return 0.0
    return min(abs(b_s - a_e), abs(a_s - b_e))


def _bbox_center(xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_center_distance(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return float("inf")
    ac = _bbox_center(a)
    bc = _bbox_center(b)
    dx = ac[0] - bc[0]
    dy = ac[1] - bc[1]
    return float((dx * dx + dy * dy) ** 0.5)


def _track_bbox_at_start(track: dict[str, Any]) -> list[float] | None:
    time_xyxy = list(track.get("time_xyxy") or [])
    if not time_xyxy:
        return None
    return [float(x) for x in time_xyxy[0][1]]


def _track_bbox_at_end(track: dict[str, Any]) -> list[float] | None:
    time_xyxy = list(track.get("time_xyxy") or [])
    if not time_xyxy:
        return None
    return [float(x) for x in time_xyxy[-1][1]]


def _is_vehicle_track(track: dict[str, Any]) -> bool:
    return str(track.get("class_name") or "").lower() in {"car", "vehicle", "bus", "truck", "motorcycle", "bicycle"}


def _can_stitch_vehicle_tracks(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    max_gap_sec: float,
    max_center_dist_px: float = 80.0,
    min_iou: float = 0.35,
) -> bool:
    if not (_is_vehicle_track(a) and _is_vehicle_track(b)):
        return False
    a_s, a_e = float(a["start_time"]), float(a["end_time"])
    b_s, b_e = float(b["start_time"]), float(b["end_time"])
    overlap = min(a_e, b_e) - max(a_s, b_s)
    if overlap > 0:
        return False
    gap = _same_camera_gap_sec(a, b)
    if gap > max_gap_sec:
        return False
    a_box = _track_bbox_at_end(a if a_e <= b_s else b)
    b_box = _track_bbox_at_start(b if a_e <= b_s else a)
    if a_box is None or b_box is None:
        return False
    if _bbox_iou(a_box, b_box) >= min_iou:
        return True
    return _bbox_center_distance(a_box, b_box) <= max_center_dist_px


def _stitch_same_camera_fragments(
    camera_result: CameraResult,
    config: CrossCameraConfig,
) -> None:
    """
    Same-camera rules:
    - Same track_id is one target (native tracker semantics).
    - For different person track_ids with small gap, stitch with Re-ID if sim >= threshold.
    - For different vehicle track_ids with small gap and matching geometry, stitch conservatively.
    """
    person_tracks = [t for t in camera_result.tracks if t.get("class_name") == "person"]
    vehicle_tracks = [t for t in camera_result.tracks if _is_vehicle_track(t)]
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(person_tracks)):
        for j in range(i + 1, len(person_tracks)):
            a, b = person_tracks[i], person_tracks[j]
            a_tid, b_tid = int(a["track_id"]), int(b["track_id"])
            if a_tid == b_tid:
                continue
            # Skip pairs with temporal overlap — two simultaneously active tracks in the
            # same camera MUST be different people; stitching them is always wrong.
            a_s, a_e = float(a["start_time"]), float(a["end_time"])
            b_s, b_e = float(b["start_time"]), float(b["end_time"])
            overlap = min(a_e, b_e) - max(a_s, b_s)
            if overlap > 0:
                continue  # simultaneously visible → different people, never stitch
            gap = _same_camera_gap_sec(a, b)
            if gap > config.same_camera_max_gap_sec:
                continue
            emb_a = camera_result.person_embeddings.get(a_tid)
            emb_b = camera_result.person_embeddings.get(b_tid)
            if emb_a is None or emb_b is None:
                continue
            sim = float(emb_a @ emb_b)
            if sim >= config.same_camera_reid_threshold:
                union(a_tid, b_tid)

    for i in range(len(vehicle_tracks)):
        for j in range(i + 1, len(vehicle_tracks)):
            a, b = vehicle_tracks[i], vehicle_tracks[j]
            a_tid, b_tid = int(a["track_id"]), int(b["track_id"])
            if a_tid == b_tid:
                continue
            if _can_stitch_vehicle_tracks(
                a,
                b,
                max_gap_sec=config.same_camera_max_gap_sec,
            ):
                union(a_tid, b_tid)

    if not parent:
        return

    tid_map: dict[int, int] = {}
    for t in camera_result.tracks:
        tid = int(t["track_id"])
        tid_map[tid] = find(tid) if tid in parent else tid

    # Rewrite track_id in tracks/events to canonical root id.
    for t in camera_result.tracks:
        tid = int(t["track_id"])
        t["track_id"] = tid_map.get(tid, tid)
    for e in camera_result.events:
        tid = e.get("track_id")
        if tid is None:
            continue
        e["track_id"] = tid_map.get(int(tid), int(tid))

    # Merge embeddings and crops by canonical tid.
    merged_emb: dict[int, list[np.ndarray]] = {}
    for tid, emb in camera_result.person_embeddings.items():
        root = tid_map.get(int(tid), int(tid))
        merged_emb.setdefault(root, []).append(emb)
    camera_result.person_embeddings = {}
    for root, embs in merged_emb.items():
        v = np.mean(np.stack(embs, axis=0), axis=0)
        n = np.linalg.norm(v)
        camera_result.person_embeddings[root] = v / n if n > 0 else v

    merged_crops: dict[int, list[PersonCrop]] = {}
    for tid, crops in camera_result.person_crops.items():
        root = tid_map.get(int(tid), int(tid))
        merged_crops.setdefault(root, []).extend(crops)
    camera_result.person_crops = merged_crops


def _load_config_yaml(path: str | Path) -> CrossCameraConfig:
    """Load CrossCameraConfig fields from a YAML file."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return CrossCameraConfig(**{
        k: v for k, v in data.items()
        if k in CrossCameraConfig.__dataclass_fields__
    })


def _process_single_camera(
    camera_id: str,
    video_path: str,
    embedder: ReIDEmbedder,
    num_crops: int = 5,
    **pipeline_kwargs: Any,
) -> CameraResult:
    """Stage-1: detect+track one stream, extract person crops and Re-ID embeddings.

    Uses vision/analyzer primitives; one track pass yields tracks and events.
    """
    model_path = pipeline_kwargs.get("model_path", "n")
    model_resolved, _ = resolve_model(model_path)
    fps, total_frames, frame_detections, tracker_label = run_yolo_track_on_video(
        video_path,
        model_path=model_resolved,
        conf=pipeline_kwargs.get("conf", 0.25),
        iou=pipeline_kwargs.get("iou", 0.45),
        tracker=pipeline_kwargs.get("tracker", "botsort_reid"),
        target_classes=pipeline_kwargs.get(
            "target_classes",
            ["person", "car"],
        ),
    )
    all_tracks = aggregate_tracks(fps, frame_detections)
    events, clips = slice_events(
        all_tracks, fps, frame_detections,
        motion_threshold=pipeline_kwargs.get("motion_threshold", 5.0),
        min_clip_duration=pipeline_kwargs.get("min_clip_duration", 1.0),
        max_static_duration=pipeline_kwargs.get("max_static_duration", 30.0),
    )

    meta: dict[str, Any] = {
        "video_path": str(Path(video_path).resolve()),
        "fps": fps,
        "total_frames": total_frames,
        "num_tracks": len(all_tracks),
        "num_events": len(events),
        "num_clips": len(clips),
        "tracker": tracker_label,
        "model": model_resolved,
        "camera_id": camera_id,
    }
    for ev in events:
        ev["camera_id"] = camera_id

    person_crops: dict[int, list[PersonCrop]] = {}
    person_embeddings: dict[int, np.ndarray] = {}

    for t in all_tracks:
        if t.get("class_name") != "person":
            continue
        tid = t["track_id"]
        crops = extract_person_crops(video_path, t, camera_id=camera_id, num_crops=num_crops)
        if crops:
            person_crops[tid] = crops
            imgs = [c.image_array for c in crops]
            emb = embedder.embed_crops(imgs)
            person_embeddings[tid] = emb.mean(axis=0)
            norm = np.linalg.norm(person_embeddings[tid])
            if norm > 0:
                person_embeddings[tid] /= norm

    result = CameraResult(
        camera_id=camera_id,
        video_path=video_path,
        tracks=all_tracks,
        events=events,
        clips=clips,
        meta=meta,
        person_crops=person_crops,
        person_embeddings=person_embeddings,
    )
    return result


def run_multi_camera_pipeline(
    camera_videos: dict[str, str],
    config: CrossCameraConfig | None = None,
    embedder: ReIDEmbedder | None = None,
    reid_config_file: str | None = None,
    reid_weights: str | None = None,
    reid_device: str = "cpu",
    num_crops: int = 5,
    use_llm_verify: bool = False,
    llm_model: str = "qwen3-vl-flash",
    topology_prior_path: str | None = None,
    **pipeline_kwargs: Any,
) -> MultiCameraOutput:
    """Main entry for multi-camera cross-view tracking.

    Args:
        camera_videos: {"cam1": "/path/cam1.mp4", "cam2": "/path/cam2.mp4", ...}
        config: Cross-camera hyperparameters; defaults if None.
        embedder: Re-ID model; created from reid_* if None.
        topology_prior_path: Path to a saved :class:`CameraTopologyPrior` JSON.
            If the file exists it is loaded; if None a fresh prior is initialised
            from the camera list and updated online during this run.
    """
    from video.core.models.camera_topology import CameraTopologyPrior

    if config is None:
        config = CrossCameraConfig()
    if embedder is None:
        embedder = ReIDEmbedder(
            config_file=reid_config_file,
            weights=reid_weights,
            device=reid_device,
        )

    # Load or initialise topology prior
    camera_ids = list(camera_videos.keys())
    if topology_prior_path and Path(topology_prior_path).exists():
        topology_prior = CameraTopologyPrior.load(topology_prior_path)
        logger.info("Loaded topology prior from %s", topology_prior_path)
    else:
        topology_prior = CameraTopologyPrior(
            cameras=camera_ids,
            max_transit_sec=config.max_transition_sec * 10,  # generous upper bound
        )
        logger.info(
            "Initialised fresh topology prior for cameras: %s", camera_ids
        )

    # Stage 1: per camera
    per_camera: list[CameraResult] = []
    for cam_id, vpath in camera_videos.items():
        logger.info("Processing camera %s: %s", cam_id, vpath)
        result = _process_single_camera(
            cam_id, vpath, embedder, num_crops=num_crops, **pipeline_kwargs,
        )
        _stitch_same_camera_fragments(result, config)
        per_camera.append(result)

    # Stage 2: cross-camera match (with topology prior)
    llm_verify_fn = None
    if use_llm_verify:
        from video.core.models.event_refinement_llm import verify_person_match_with_llm
        from functools import partial
        llm_verify_fn = partial(verify_person_match_with_llm, model=llm_model)

    global_entities = match_across_cameras(
        per_camera, config, embedder, llm_verify_fn,
        topology_prior=topology_prior,
    )

    # Persist updated topology prior for future runs
    if topology_prior_path:
        topology_prior.save(topology_prior_path)
        logger.info("Topology prior persisted to %s", topology_prior_path)

    # Stage 3: merge events
    entity_lookup = _build_entity_lookup(global_entities)
    merged_events = _merge_events(per_camera, entity_lookup)

    return MultiCameraOutput(
        cameras=dict(camera_videos),
        config=config,
        global_entities=global_entities,
        per_camera=per_camera,
        merged_events=merged_events,
    )


def _build_entity_lookup(
    entities: list[GlobalEntity],
) -> dict[tuple[str, int], str]:
    """Map (camera_id, track_id) → global_entity_id."""
    lookup: dict[tuple[str, int], str] = {}
    for ent in entities:
        for app in ent.appearances:
            lookup[(app.camera_id, app.track_id)] = ent.global_entity_id
    return lookup


def _merge_events(
    per_camera: list[CameraResult],
    entity_lookup: dict[tuple[str, int], str],
) -> list[dict[str, Any]]:
    """Merge events from all cameras; add global_entity_id when matched."""
    merged: list[dict[str, Any]] = []
    for cam in per_camera:
        for ev in cam.events:
            ev_copy = dict(ev)
            ev_copy.setdefault("camera_id", cam.camera_id)
            tid = ev.get("track_id")
            if tid is not None:
                key = (cam.camera_id, tid)
                gid = entity_lookup.get(key)
                if gid is not None:
                    ev_copy["global_entity_id"] = gid
            merged.append(ev_copy)
    merged.sort(key=lambda e: e.get("start_time", 0.0))
    return merged


# ------------------------------------------------------------------
# JSON serialization
# ------------------------------------------------------------------

def multi_camera_output_to_dict(output: MultiCameraOutput) -> dict[str, Any]:
    """Serialize MultiCameraOutput to a JSON-compatible dict."""
    return {
        "meta": {
            "cameras": output.cameras,
            "cross_camera_config": asdict(output.config),
        },
        "global_entities": [
            {
                "global_entity_id": ent.global_entity_id,
                "appearances": [asdict(a) for a in ent.appearances],
            }
            for ent in output.global_entities
        ],
        "events": output.merged_events,
    }


def save_multi_camera_output(
    output: MultiCameraOutput,
    out_path: str | Path,
) -> Path:
    """Write multi-camera result to a JSON file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = multi_camera_output_to_dict(output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Multi-camera output saved to %s", out_path)
    return out_path
