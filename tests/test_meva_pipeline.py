"""MEVA multi-camera end-to-end pipeline test.

Runs the full pipeline on MEVA KF1 synchronized video clips:
  YOLO11m + BoT-SORT+ReID  →  cross-camera matching  →  topology prior update

Dataset structure expected:
  <meva-root>/
    2018-03-07.10-00-00.10-05-00.admin.G329.r13.avi
    2018-03-07.10-00-00.10-05-00.hospital.G436.r13.avi
    ...

Video filename format:
  <date>.<hh-mm-ss>.<hh-mm-ss>.<location>.<camera>.r13.avi

Usage
-----
python tests/test_meva_pipeline.py \\
    --meva-dir  _data/meva/2018-03-07 \\
    --cameras   G329 G336 G436 G509 \\
    --date      2018-03-07 \\
    --start-hour 10 \\
    --duration-min 15 \\
    --topology-prior results/mevid_topology.json \\
    --out results/meva_multicam.json \\
    --max-clips-per-cam 1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\."
    r"(\w+)\.(G\d+)\.r\d+\.avi",
    re.IGNORECASE,
)


def _parse_meva_filename(path: Path):
    """Parse MEVA filename → (date, start_hms, end_hms, location, camera_id)."""
    m = _FNAME_RE.search(path.name)
    if not m:
        return None
    date, t_start, t_end, location, cam = m.groups()
    def to_sec(hms: str) -> int:
        h, mi, s = hms.split("-")
        return int(h) * 3600 + int(mi) * 60 + int(s)
    return {
        "date":      date,
        "t_start":   t_start,
        "t_end":     t_end,
        "start_sec": to_sec(t_start),
        "end_sec":   to_sec(t_end),
        "location":  location,
        "camera_id": cam,
        "path":      path,
    }


def discover_clips(meva_dir: Path, cameras: list[str]) -> dict[str, list[dict]]:
    """Discover and sort clips per camera."""
    per_cam: dict[str, list[dict]] = defaultdict(list)
    for avi in sorted(meva_dir.glob("*.avi")):
        info = _parse_meva_filename(avi)
        if info is None:
            continue
        if cameras and info["camera_id"] not in cameras:
            continue
        per_cam[info["camera_id"]].append(info)
    for cam in per_cam:
        per_cam[cam].sort(key=lambda x: x["start_sec"])
    return dict(per_cam)


# ---------------------------------------------------------------------------
# Per-camera processing (YOLO + events + embeddings)
# ---------------------------------------------------------------------------

def process_camera_clips(
    camera_id: str,
    clips: list[dict],
    max_clips: int,
    pipeline_kwargs: dict[str, Any],
    embedder: Any,
    num_crops: int = 5,
) -> "CameraResult":
    """Run Stage-1 pipeline on sequential clips for one camera.

    Clips are processed in order with time offsets so that track timestamps
    are continuous across the full 15-minute recording window.
    """
    from video.factory.multi_camera_coordinator import _process_single_camera
    from video.core.schema.multi_camera import CameraResult
    import numpy as np

    selected = clips[:max_clips]
    logger.info(
        "Camera %s: processing %d clips (%s -> %s)",
        camera_id, len(selected),
        selected[0]["t_start"], selected[-1]["t_end"],
    )

    # Process each clip individually then merge
    merged_tracks: list[dict] = []
    merged_events: list[dict] = []
    merged_clips: list[dict] = []
    merged_crops: dict[int, list] = {}
    merged_embs: dict[int, list] = {}
    first_meta: dict = {}

    global_track_offset = 0
    fps_used = 25.0

    for clip_idx, clip in enumerate(selected):
        video_path = str(clip["path"])
        time_offset = clip["start_sec"]
        logger.info(
            "  [%d/%d] %s (offset=%ds)",
            clip_idx + 1, len(selected), clip["path"].name, time_offset,
        )

        try:
            cam_result = _process_single_camera(
                camera_id=camera_id,
                video_path=video_path,
                embedder=embedder,
                num_crops=num_crops,
                **pipeline_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Clip %s failed: %s", clip["path"].name, exc)
            continue

        fps_used = cam_result.meta.get("fps", 25.0)
        if not first_meta:
            first_meta = dict(cam_result.meta)
            first_meta["camera_id"] = camera_id

        # Shift track/event timestamps by clip's wall-clock offset
        # Also shift track IDs to avoid collisions across clips
        tid_map: dict[int, int] = {}
        for t in cam_result.tracks:
            old_tid = int(t["track_id"])
            new_tid = old_tid + global_track_offset
            tid_map[old_tid] = new_tid
            tc = dict(t)
            tc["track_id"] = new_tid
            tc["start_time"] = float(tc["start_time"]) + time_offset
            tc["end_time"]   = float(tc["end_time"])   + time_offset
            tc["camera_id"]  = camera_id
            merged_tracks.append(tc)

        for e in cam_result.events:
            ec = dict(e)
            ec["start_time"] = float(ec.get("start_time", 0)) + time_offset
            ec["end_time"]   = float(ec.get("end_time", 0))   + time_offset
            ec["camera_id"]  = camera_id
            old_tid = ec.get("track_id")
            if old_tid is not None:
                ec["track_id"] = tid_map.get(int(old_tid), int(old_tid) + global_track_offset)
            merged_events.append(ec)

        for c in cam_result.clips:
            cc = dict(c)
            cc["start"] = float(cc.get("start", 0)) + time_offset
            cc["end"]   = float(cc.get("end", 0))   + time_offset
            merged_clips.append(cc)

        # Re-map crops and embeddings
        for old_tid, crops in cam_result.person_crops.items():
            new_tid = tid_map.get(int(old_tid), int(old_tid) + global_track_offset)
            merged_crops.setdefault(new_tid, []).extend(crops)

        for old_tid, emb in cam_result.person_embeddings.items():
            new_tid = tid_map.get(int(old_tid), int(old_tid) + global_track_offset)
            if new_tid not in merged_embs:
                merged_embs[new_tid] = []
            merged_embs[new_tid].append(emb)

        # Advance track ID offset past the highest ID used in this clip
        if cam_result.tracks:
            max_tid = max(int(t["track_id"]) for t in cam_result.tracks)
            global_track_offset += max_tid + 1

    # Average embeddings per merged track
    import numpy as np
    final_embs: dict[int, np.ndarray] = {}
    for tid, embs in merged_embs.items():
        stacked = np.stack(embs, axis=0)
        mean = stacked.mean(axis=0)
        norm = np.linalg.norm(mean)
        final_embs[tid] = mean / norm if norm > 0 else mean

    meta = first_meta or {
        "camera_id": camera_id,
        "fps": fps_used,
        "clips_processed": len(selected),
    }
    meta["num_tracks"] = len({t["track_id"] for t in merged_tracks})
    meta["num_events"] = len(merged_events)

    result = CameraResult(
        camera_id=camera_id,
        video_path=str(selected[0]["path"]) if selected else "",
        tracks=merged_tracks,
        events=merged_events,
        clips=merged_clips,
        meta=meta,
        person_crops=merged_crops,
        person_embeddings=final_embs,
    )
    logger.info(
        "Camera %s: %d tracks, %d events, %d person embeddings",
        camera_id,
        meta["num_tracks"],
        meta["num_events"],
        len(final_embs),
    )
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_meva_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    meva_dir = Path(args.meva_dir)
    cameras  = args.cameras
    out_path = Path(args.out)
    topo_path = args.topology_prior

    t0 = time.time()

    # ------------------------------------------------------------------ #
    # 1. Discover clips                                                    #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 1: Discover clips ===")
    per_cam = discover_clips(meva_dir, cameras)
    if not per_cam:
        logger.error("No MEVA clips found in %s for cameras %s", meva_dir, cameras)
        return {"error": "no clips"}

    for cam, clips in per_cam.items():
        sizes_mb = [c["path"].stat().st_size / 1e6 for c in clips]
        logger.info(
            "  %s: %d clips, %.0f-%.0f s, total %.0f MB",
            cam, len(clips),
            clips[0]["start_sec"], clips[-1]["end_sec"],
            sum(sizes_mb),
        )

    # ------------------------------------------------------------------ #
    # 2. Load Re-ID embedder                                              #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 2: Load Re-ID embedder ===")
    from video.core.models.reid_embedder import ReIDEmbedder
    embedder = ReIDEmbedder(
        config_file=getattr(args, "reid_config", None),
        weights=getattr(args, "reid_weights", None),
        device=getattr(args, "device", "cpu"),
    )

    # ------------------------------------------------------------------ #
    # 3. Load or initialise topology prior                                 #
    # ------------------------------------------------------------------ #
    from video.core.models.camera_topology import CameraTopologyPrior
    if topo_path and Path(topo_path).exists():
        topology_prior = CameraTopologyPrior.load(topo_path)
        logger.info("Loaded topology prior from %s: %s", topo_path, topology_prior)
    else:
        topology_prior = CameraTopologyPrior(
            cameras=cameras,
            max_transit_sec=600.0,
        )
        logger.info("Initialised fresh topology prior for cameras: %s", cameras)

    # ------------------------------------------------------------------ #
    # 4. Per-camera YOLO + tracking                                        #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 3: Per-camera tracking ===")
    pipeline_kwargs = dict(
        model_path=getattr(args, "model", "m"),
        conf=getattr(args, "conf", 0.25),
        iou=getattr(args, "iou", 0.25),
        tracker=getattr(args, "tracker", "botsort_reid"),
        motion_threshold=5.0,
        min_clip_duration=1.0,
        max_static_duration=30.0,
    )

    from video.factory.multi_camera_coordinator import _stitch_same_camera_fragments
    from video.core.schema.multi_camera import CrossCameraConfig

    config = CrossCameraConfig(
        max_transition_sec=120.0,
        topology_weight_reid=0.55,
        topology_weight_topo=0.45,
    )

    per_camera_results = []
    for cam_id, clips in sorted(per_cam.items()):
        cam_result = process_camera_clips(
            camera_id=cam_id,
            clips=clips,
            max_clips=args.max_clips_per_cam,
            pipeline_kwargs=pipeline_kwargs,
            embedder=embedder,
            num_crops=5,
        )
        _stitch_same_camera_fragments(cam_result, config)
        per_camera_results.append(cam_result)

    # ------------------------------------------------------------------ #
    # 5. Cross-camera matching with topology prior                         #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 4: Cross-camera matching ===")
    from video.factory.processors.cross_camera_matcher import match_across_cameras

    global_entities = match_across_cameras(
        per_camera=per_camera_results,
        config=config,
        embedder=embedder,
        topology_prior=topology_prior,
    )

    logger.info("Found %d global entities (cross-camera persons)", len(global_entities))
    for ent in global_entities[:10]:
        cams_seen = [a.camera_id for a in ent.appearances]
        times = [(a.start_time, a.end_time) for a in ent.appearances]
        logger.info(
            "  %s: appeared on %s at times %s",
            ent.global_entity_id, cams_seen,
            [(f"{s:.0f}s", f"{e:.0f}s") for s, e in times],
        )

    # ------------------------------------------------------------------ #
    # 6. Save updated topology prior                                       #
    # ------------------------------------------------------------------ #
    updated_topo_path = out_path.parent / "meva_topology_updated.json"
    topology_prior.save(updated_topo_path)
    logger.info("Updated topology prior saved to %s", updated_topo_path)
    topo_table = topology_prior.transition_table()
    top_pairs = topology_prior.most_connected_pairs(top_k=10)

    # ------------------------------------------------------------------ #
    # 7. Build summary                                                     #
    # ------------------------------------------------------------------ #
    elapsed = time.time() - t0

    # Per-camera stats
    cam_stats = []
    for cr in per_camera_results:
        n_person_tracks = sum(1 for t in cr.tracks if t.get("class_name") == "person")
        cam_stats.append({
            "camera_id":       cr.camera_id,
            "location":        cr.meta.get("location", "?"),
            "num_tracks":      cr.meta.get("num_tracks", 0),
            "num_person_tracks": n_person_tracks,
            "num_events":      cr.meta.get("num_events", 0),
            "num_embeddings":  len(cr.person_embeddings),
        })

    # Cross-camera entities with details
    entities_out = []
    for ent in global_entities:
        entities_out.append({
            "global_entity_id": ent.global_entity_id,
            "n_cameras": len(ent.appearances),
            "cameras_seen": [a.camera_id for a in ent.appearances],
            "appearances": [
                {
                    "camera_id":  a.camera_id,
                    "start_time": round(a.start_time, 1),
                    "end_time":   round(a.end_time, 1),
                    "confidence": round(a.confidence, 3),
                }
                for a in ent.appearances
            ],
        })

    result = {
        "dataset":        "MEVA-KF1",
        "date":           "2018-03-07",
        "time_window":    "10:00-10:15",
        "cameras":        cameras,
        "clips_per_cam":  args.max_clips_per_cam,
        "elapsed_sec":    round(elapsed, 1),
        "per_camera":     cam_stats,
        "n_global_entities": len(global_entities),
        "global_entities": entities_out[:50],  # top 50
        "topology_top_pairs": top_pairs,
        "topology_table": topo_table,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)

    # Console summary
    total_tracks = sum(s["num_tracks"] for s in cam_stats)
    total_persons = sum(s["num_person_tracks"] for s in cam_stats)
    multicam = sum(1 for e in entities_out if e["n_cameras"] >= 2)

    print("\n" + "=" * 65)
    print("MEVA Multi-Camera Pipeline Summary")
    print("=" * 65)
    print(f"  Cameras       : {len(cameras)}  ({', '.join(cameras)})")
    print(f"  Time window   : 10:00-10:15 ({args.max_clips_per_cam} clips/cam)")
    print(f"  Elapsed       : {elapsed:.0f}s")
    print()
    print("  Per-camera stats:")
    for s in cam_stats:
        print(
            f"    {s['camera_id']}  tracks={s['num_tracks']}  "
            f"persons={s['num_person_tracks']}  "
            f"embeddings={s['num_embeddings']}"
        )
    print()
    print(f"  Cross-camera entities : {len(global_entities)}")
    print(f"  Multi-camera persons  : {multicam}  (seen on >=2 cameras)")
    print()
    if top_pairs:
        print("  Topology prior (top pairs after update):")
        for p in top_pairs[:5]:
            mean_t = p["mean_transit_sec"]
            print(
                f"    {p['cam_a']:4s} -> {p['cam_b']:4s}  "
                f"n={p['n_observations']:3d}  "
                f"mean={mean_t:.1f}s  fitted={p['fitted']}"
                if mean_t is not None else
                f"    {p['cam_a']:4s} -> {p['cam_b']:4s}  n={p['n_observations']}"
            )
    print("=" * 65)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEVA multi-camera pipeline test")
    p.add_argument("--meva-dir", required=True,
                   help="Directory containing MEVA .avi clips")
    p.add_argument("--cameras", nargs="+", default=["G329", "G336", "G436", "G509"],
                   help="Camera IDs to include")
    p.add_argument("--max-clips-per-cam", type=int, default=1,
                   help="Max clips per camera (1 = first 5-min clip only)")
    p.add_argument("--topology-prior", default=None,
                   help="Path to existing topology prior JSON (e.g. from MEVID)")
    p.add_argument("--model", default="m",
                   help="YOLO model size (n/s/m/l/x) or full path")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou",  type=float, default=0.25)
    p.add_argument("--tracker", default="botsort_reid")
    p.add_argument("--reid-weights", default=None)
    p.add_argument("--reid-config",  default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", default="results/meva_multicam.json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_meva_pipeline(args)
