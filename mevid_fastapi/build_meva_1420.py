"""Build meva_1420 database — 14:20 slot, all cameras, full refine pipeline.

Steps:
  1. YOLO + Re-ID + cross-camera matching  (warm-start campus topology)
  2. VLM text refinement (refine_multi_camera_output)
  3. VLM appearance refinement (run_appearance_refinement_for_pipeline)
  4. Generate seeds
  5. Build SQLite + Chroma

Usage (from project root, capstone conda env):
    conda activate capstone
    cd D:\\Capstone\\surveillance-video-query-client-release
    python mevid_fastapi\\build_meva_1420.py

Skip VLM steps (fast mode — YOLO+Re-ID only):
    python mevid_fastapi\\build_meva_1420.py --no-refine
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RELEASE = _HERE.parent
_LLM = _RELEASE.parent / "CS32-1-surveillance-video-query-llm"
_AGENT = _RELEASE / "agent"
for _p in (str(_RELEASE), str(_LLM), str(_AGENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env
for _env in (_RELEASE / ".env", _LLM / ".env"):
    if _env.exists():
        for _line in _env.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
        break

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Build meva_1420 database")
parser.add_argument("--no-refine", action="store_true",
                    help="Skip VLM text & appearance refinement (fast mode)")
args = parser.parse_args()
RUN_REFINE = not args.no_refine

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR  = _LLM / "_data" / "mevid_slots"
NAMESPACE  = "meva_1420"
OUTPUT_DIR = _HERE / "data" / "runtime" / NAMESPACE
SLOT       = "14-20"
MODEL      = os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3-vl-flash")

CAMERA_VIDEOS = {
    cam: str(VIDEO_DIR / f"{stem}.avi")
    for cam, stem in {
        "G328": "2018-03-11.14-20-01.14-25-01.school.G328.r13",
        "G339": "2018-03-11.14-20-01.14-25-01.school.G339.r13",
        "G419": "2018-03-11.14-20-01.14-25-01.school.G419.r13",
        "G421": "2018-03-11.14-20-01.14-25-01.school.G421.r13",
        "G423": "2018-03-11.14-20-01.14-25-01.school.G423.r13",
        "G505": "2018-03-11.14-20-01.14-25-01.bus.G505.r13",
        "G506": "2018-03-11.14-20-01.14-25-01.bus.G506.r13",
        "G508": "2018-03-11.14-20-01.14-25-01.bus.G508.r13",
    }.items()
    if (VIDEO_DIR / f"{stem}.avi").exists()
}

CACHE_DIR  = OUTPUT_DIR / "pipeline_cache"
SEEDS_DIR  = OUTPUT_DIR / "seeds"
DB_DIR     = OUTPUT_DIR / "runtime"

for d in (CACHE_DIR, SEEDS_DIR, DB_DIR):
    d.mkdir(parents=True, exist_ok=True)


def sep(title: str) -> None:
    print(f"\n{'='*60}\n{title}")


def _run_with_heartbeat(label: str, fn, interval_sec: float = 20.0):
    """Run a long step and print periodic progress heartbeats."""
    t0 = time.perf_counter()
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(interval_sec):
            elapsed = time.perf_counter() - t0
            print(f"  [{label}] running... elapsed={elapsed:.1f}s", flush=True)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    try:
        return fn()
    finally:
        stop.set()
        th.join(timeout=0.2)


def _print_event_counts(pipeline_dict: dict) -> None:
    events = pipeline_dict.get("events", []) or []
    by_cam: dict[str, int] = defaultdict(int)
    for ev in events:
        cam = str(ev.get("camera_id", ""))
        if cam:
            by_cam[cam] += 1
    if by_cam:
        print("  Events by camera:")
        for cam in sorted(by_cam):
            print(f"    {cam}: {by_cam[cam]}")


# ── STEP 1: Multi-camera pipeline (YOLO + Re-ID + cross-camera) ───────────────
sep(f"STEP 1 — YOLO + Re-ID + Cross-Camera  [{SLOT}]")
print(f"  Cameras : {list(CAMERA_VIDEOS.keys())}")
print(f"  Model   : {MODEL}")
print(f"  Refine  : {RUN_REFINE}")

from video.factory.multi_camera_coordinator import (
    run_multi_camera_pipeline,
    multi_camera_output_to_dict,
)
from video.core.schema.multi_camera import CrossCameraConfig
from video.core.models.camera_topology import CameraTopologyPrior

_topology_prior = CameraTopologyPrior.mevid_campus(
    cameras=list(CAMERA_VIDEOS.keys()),
    max_transit_sec=300.0,
)
print(f"  Topology: {_topology_prior}")

t0 = time.perf_counter()
pipeline_output = _run_with_heartbeat(
    "STEP1 YOLO+ReID",
    lambda: run_multi_camera_pipeline(
        camera_videos=CAMERA_VIDEOS,
        config=CrossCameraConfig(
            embedding_threshold=0.70,
            cross_camera_min_score=0.72,
            topology_weight_reid=0.65,
            topology_weight_topo=0.35,
            max_global_entity_cameras=4,
            max_global_entity_tracks=4,
            person_only=True,
            same_camera_reid_threshold=0.80,
        ),
        model_path="11m",
        conf=0.40,
        iou=0.40,
        reid_device="cpu",
        num_crops=5,
        topology_prior=_topology_prior,
    ),
)
pipeline_dict = multi_camera_output_to_dict(pipeline_output)
CACHE_DIR.joinpath("pipeline_output.json").write_text(
    json.dumps(pipeline_dict, ensure_ascii=False, indent=2), encoding="utf-8"
)
ge_count = len(pipeline_dict.get("global_entities", []))
ev_count = len(pipeline_dict.get("events", []))
print(f"  Done in {time.perf_counter()-t0:.1f}s — global_entities={ge_count}, events={ev_count}")
_print_event_counts(pipeline_dict)

# ── STEP 2: VLM text refinement ───────────────────────────────────────────────
if RUN_REFINE:
    sep("STEP 2 — VLM Text Refinement")
    from video.factory.refinement_runner import RefineEventsConfig, refine_multi_camera_output

    t0 = time.perf_counter()
    refined_per_cam = _run_with_heartbeat(
        "STEP2 TextRefine",
        lambda: refine_multi_camera_output(
            pipeline_output,
            config=RefineEventsConfig(mode="vector", model=MODEL),
        ),
    )
    # refine_multi_camera_output returns per-cam dict, NOT MultiCameraOutput.
    # Patch refined event text back into pipeline_dict in-place.
    for _cam_id, _cam_data in refined_per_cam.items():
        for _rev in _cam_data.get("events", []):
            _tid = str(_rev.get("track_id") or "")
            for ev in pipeline_dict.get("events", []):
                if str(ev.get("camera_id") or "") == _cam_id and str(ev.get("track_id") or "") == _tid:
                    ev["event_text"] = _rev.get("event_text") or ev.get("event_text", "")
                    ev["description_for_llm"] = _rev.get("event_text") or ev.get("description_for_llm", "")
                    break
    CACHE_DIR.joinpath("pipeline_refined.json").write_text(
        json.dumps({"refined_per_cam": refined_per_cam}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    _print_event_counts(pipeline_dict)
else:
    sep("STEP 2 — VLM Text Refinement  [SKIPPED]")

# ── STEP 3: Appearance refinement ─────────────────────────────────────────────
if RUN_REFINE:
    sep("STEP 3 — Appearance Refinement (crop-based VLM)")
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

    t0 = time.perf_counter()
    print(f"  Entities to refine: {len(pipeline_dict.get('global_entities', []))}")
    appearance_result = _run_with_heartbeat(
        "STEP3 AppearanceRefine",
        lambda: run_appearance_refinement_for_pipeline(
            slot=SLOT,
            pipeline=pipeline_with_per_cam,
            camera_to_video=meta_cameras,
            camera_video_stems=camera_video_stems,
            config=AppearanceRefinementConfig.from_env(
                model=MODEL,
                cache_path=CACHE_DIR / "appearance_refined.json",
                force=False,
                max_apps_per_entity=4,
                crops_per_app=2,
            ),
        ),
    )
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
else:
    sep("STEP 3 — Appearance Refinement  [SKIPPED]")
    appearance_result = {}

# ── STEP 4: Generate seeds ─────────────────────────────────────────────────────
sep("STEP 4 — Generate Seeds")
from mevid_fastapi.pipeline_service import _pipeline_output_to_seeds, ingest_from_seeds

# Build appearance lookup
app_lookup: dict = {}
per_cam_app = appearance_result.get("per_camera", {})
for cam_id, cam_data in per_cam_app.items():
    for ev in cam_data.get("events", []):
        app_lookup[(str(cam_id), str(ev.get("entity_hint", "")))] = ev

seed_paths = _pipeline_output_to_seeds(None, pipeline_dict, SEEDS_DIR, NAMESPACE)

total_enriched = 0
for seed_path in seed_paths:
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    enriched = 0
    for ev in seed.get("events", []):
        cam  = str(ev.get("camera_id", ""))
        hint = str(ev.get("entity_hint", ""))
        if not hint:
            continue
        app_ev = app_lookup.get((cam, hint))
        if not app_ev:
            continue
        ev["object_color"]        = app_ev.get("object_color", "")
        ev["object_color_en"]     = ev["object_color"]
        ev["appearance_notes"]    = app_ev.get("appearance_notes", "")
        ev["appearance_notes_en"] = ev["appearance_notes"]
        ev["event_text_en"]       = app_ev.get("event_text", ev.get("event_text_en", ""))
        ev["event_summary_en"]    = ev["event_text_en"]
        ev["keywords"] = list(dict.fromkeys(
            (ev.get("keywords") or []) + (app_ev.get("keywords") or [])
        ))[:16]
        enriched += 1
    total_enriched += enriched
    seed_path.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ev = len(seed.get("events", []))
    print(f"  {seed_path.name}: {n_ev} events, {enriched} appearance-enriched")

print(f"\n  Total enriched: {total_enriched}")

# ── STEP 5: Build SQLite + Chroma ─────────────────────────────────────────────
sep("STEP 5 — Build SQLite + Chroma")
t0 = time.perf_counter()
result = ingest_from_seeds(
    seeds_dir=str(SEEDS_DIR),
    namespace=NAMESPACE,
    output_base=str(DB_DIR),
    reset_existing=True,
)
print(f"  Done in {time.perf_counter()-t0:.1f}s")
print(f"  sqlite_rows    = {result['sqlite_rows']}")
print(f"  chroma_child   = {result['chroma_child_records']}")
print(f"  chroma_ge      = {result['chroma_ge_records']}")

sep("DONE")
print(f"  SQLite : {result['sqlite_path']}")
print(f"  Chroma : {result['chroma_path']}")
print()
print("  Select in server:")
print(f"    namespace   : {NAMESPACE}")
print(f"    sqlite_path : {result['sqlite_path']}")
print(f"    chroma_path : {result['chroma_path']}")
