"""tests/run_junwen_benchmark.py
================================
Multi-camera benchmark test — JunwenChen 16:20-16:25 slot (8 cameras).

Annotator: JunwenChen  |  Slot: 2018-03-11  16:20–16:25
Cameras: G326 G328 G329 G336 G419 G420 G506 G508
Questions: tests/junwen_16_20_mcq50.json (50 questions, ground-truth from xlsx)

Pipeline: ingest_from_videos() — full multi-camera pipeline with Re-ID matching.

Steps
-----
1. Check / download the 8 video files from MEVID S3.
2. Run multi-camera pipeline (YOLO + ReID + cross-camera matching + LLM refinement).
3. Select the resulting DB in the agent.
4. Evaluate 50 MCQ questions (with camera-id filtering ON).
5. Print comparison table and save JSON results.

Usage
-----
  # Full run (pipeline + eval):
  C:\\Users\\17809\\anaconda3\\envs\\capstone\\python.exe tests/run_junwen_benchmark.py

  # Skip pipeline (DB already built, re-run eval only):
  C:\\Users\\17809\\anaconda3\\envs\\capstone\\python.exe tests/run_junwen_benchmark.py --skip-pipeline

  # Skip download check (assume videos are present):
  C:\\Users\\17809\\anaconda3\\envs\\capstone\\python.exe tests/run_junwen_benchmark.py --skip-download
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
FASTAPI = ROOT / "mevid_fastapi"
AGENT   = ROOT / "agent"
for _p in (str(ROOT), str(AGENT), str(FASTAPI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Load .env ─────────────────────────────────────────────────────────────────
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

os.environ.setdefault(
    "AGENT_MCQ_JUDGE_MODEL",
    os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus"),
)

# ── CLI flags ─────────────────────────────────────────────────────────────────
SKIP_PIPELINE = "--skip-pipeline" in sys.argv
SKIP_DOWNLOAD = "--skip-download" in sys.argv or SKIP_PIPELINE

# ── Config ────────────────────────────────────────────────────────────────────
SLOT        = "16-20"
NAMESPACE   = "mevid_junwen_1620"
VIDEO_DIR   = ROOT / "_data" / "mevid_slots"
DB_DIR      = FASTAPI / "data" / "runtime" / NAMESPACE
QUESTIONS   = ROOT / "tests" / "junwen_16_20_mcq50.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# S3 base URL for MEVID dataset
_S3_BASE = "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-11/16"

# 8 cameras in JunwenChen's 16:20 slot
CAMERAS: dict[str, dict] = {
    "G326": {
        "filename": "2018-03-11.16-20-01.16-25-01.admin.G326.r13.avi",
        "url": f"{_S3_BASE}/2018-03-11.16-20-01.16-25-01.admin.G326.r13.avi",
    },
    "G328": {
        "filename": "2018-03-11.16-20-01.16-25-01.school.G328.r13.avi",
        "url": f"{_S3_BASE}/2018-03-11.16-20-01.16-25-01.school.G328.r13.avi",
    },
    "G329": {
        "filename": "2018-03-11.16-20-01.16-25-01.admin.G329.r13.avi",
        "url": f"{_S3_BASE}/2018-03-11.16-20-01.16-25-01.admin.G329.r13.avi",
    },
    "G336": {
        "filename": "2018-03-11.16-20-01.16-25-01.school.G336.r13.avi",
        "url": f"{_S3_BASE}/2018-03-11.16-20-01.16-25-01.school.G336.r13.avi",
    },
    "G419": {
        "filename": "2018-03-11.16-20-01.16-25-01.school.G419.r13.avi",
        "url": f"{_S3_BASE}/2018-03-11.16-20-01.16-25-01.school.G419.r13.avi",
    },
    "G420": {
        "filename": "2018-03-11.16-20-01.16-25-01.school.G420.r13.avi",
        "url": f"{_S3_BASE}/2018-03-11.16-20-01.16-25-01.school.G420.r13.avi",
    },
    "G506": {
        "filename": "2018-03-11.16-20-01.16-25-01.bus.G506.r13.avi",
        "url": f"{_S3_BASE}/2018-03-11.16-20-01.16-25-01.bus.G506.r13.avi",
    },
    "G508": {
        "filename": "2018-03-11.16-20-01.16-25-01.bus.G508.r13.avi",
        "url": f"{_S3_BASE}/2018-03-11.16-20-01.16-25-01.bus.G508.r13.avi",
    },
}


def _sep(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)


# ── STEP 1: Check / download videos ──────────────────────────────────────────
_sep("STEP 1 — VIDEO FILES")

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
camera_videos: dict[str, str] = {}

for cam_id, info in CAMERAS.items():
    path = VIDEO_DIR / info["filename"]
    camera_videos[cam_id] = str(path)

    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {cam_id}: {info['filename']} ({size_mb:.1f} MB)")
        continue

    if SKIP_DOWNLOAD:
        print(f"  ✗ {cam_id}: NOT FOUND (--skip-download active, will fail pipeline)")
        continue

    print(f"  ↓ {cam_id}: downloading {info['filename']} ...")
    t_dl = time.perf_counter()
    try:
        import urllib.request
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(info["url"], str(tmp))
        tmp.rename(path)
        elapsed = round(time.perf_counter() - t_dl, 1)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"    done in {elapsed}s — {size_mb:.1f} MB")
    except Exception as exc:
        print(f"    ERROR downloading {cam_id}: {exc}")
        print(f"    Manual URL: {info['url']}")

missing = [cam for cam, p in camera_videos.items() if not Path(p).exists()]
if missing:
    print(f"\n  WARNING: Missing videos: {missing}")
    if not SKIP_PIPELINE:
        print("  Continuing — missing cameras will be skipped by pipeline.")

present_videos = {cam: p for cam, p in camera_videos.items() if Path(p).exists()}
print(f"\n  Videos ready: {len(present_videos)}/{len(CAMERAS)}")


# ── STEP 2: Multi-camera pipeline ─────────────────────────────────────────────
_sep("STEP 2 — MULTI-CAMERA PIPELINE")

import pipeline_service  # noqa: E402

if SKIP_PIPELINE:
    print("  --skip-pipeline: skipping ingestion")
else:
    if len(present_videos) < 2:
        print("  ERROR: Need at least 2 videos for multi-camera pipeline. Exiting.")
        sys.exit(1)

    print(f"  Running multi-camera pipeline on {len(present_videos)} cameras:")
    for cam, p in present_videos.items():
        print(f"    {cam}: {Path(p).name}")
    print()

    t_pipe = time.perf_counter()
    try:
        result_pipe = pipeline_service.ingest_from_videos(
            camera_videos=present_videos,
            namespace=NAMESPACE,
            output_base=str(DB_DIR),
            conf=0.35,
            iou=0.35,
            reid_device="cpu",
            refine=True,                # LLM event text refinement
            refine_model=os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus"),
            reset_existing=True,
        )
        elapsed_pipe = round(time.perf_counter() - t_pipe, 1)
        print(f"\n  Pipeline done in {elapsed_pipe}s ({elapsed_pipe/60:.1f} min)")
        print(f"  sqlite_rows   = {result_pipe.get('sqlite_rows', 0)}")
        print(f"  chroma_child  = {result_pipe.get('chroma_child_records', 0)}")
        print(f"  seed_files    = {result_pipe.get('seed_file_count', 0)}")
    except Exception as exc:
        import traceback
        print(f"  Pipeline ERROR: {exc}")
        traceback.print_exc()
        print("\n  Attempting to continue with existing DB if present...")


# ── STEP 3: Select DB ─────────────────────────────────────────────────────────
_sep("STEP 3 — SELECT DATABASE")

from service import agent_service           # noqa: E402
from eval_mcq_service import run_mcq_eval   # noqa: E402

sqlite = DB_DIR / "runtime" / "episodic_events.sqlite"
chroma = DB_DIR / "runtime" / "chroma"

if not sqlite.exists():
    print(f"  ERROR: DB not found at {sqlite}")
    print("  Did the pipeline finish? Or run with --skip-pipeline if DB already exists.")
    sys.exit(1)

agent_service.select_database(
    sqlite_path=str(sqlite),
    chroma_path=str(chroma),
    namespace=NAMESPACE,
)
print(f"  DB selected: {NAMESPACE}")
print(f"  sqlite: {sqlite}")


# ── STEP 4: MCQ Evaluation ────────────────────────────────────────────────────
_sep("STEP 4 — MCQ EVALUATION (camera-filter ON)")

def _query_fn(q: str) -> dict:
    return agent_service.run_query(
        query=q, thread_id=None, user_id="mcq_junwen",
        include_rows=True, include_node_trace=False, top_k_rows=5,
    )

t_eval = time.perf_counter()
result = run_mcq_eval(
    questions_file=str(QUESTIONS),
    agent_query_fn=_query_fn,
    use_camera_filter=True,
)
eval_elapsed = round(time.perf_counter() - t_eval, 1)

print(f"\n  Overall: accuracy={result['accuracy']:.1%}  "
      f"correct={result['correct']}/{result['total']}  "
      f"mean_iou={result['mean_iou']:.3f}  ({eval_elapsed}s)")
print()
print(f"  {'Category':<10}  {'Acc':>6}  {'IoU':>6}  {'Correct':>10}")
print("  " + "-" * 40)
for cat, s in result.get("by_category", {}).items():
    print(f"  {cat:<10}  {s['accuracy']:>5.1%}  {s['mean_iou']:>6.3f}  "
          f"  {s['correct']:>2}/{s['total']}")


# ── STEP 5: Per-question breakdown ───────────────────────────────────────────
_sep("STEP 5 — PER-QUESTION BREAKDOWN")

print(f"\n  ✓ Correct ({result['correct']}):")
for c in result.get("cases", []):
    if c["correct"]:
        print(f"    {c['case_id']:<6} ({c['category']:<8})  cam={c.get('camera_id','?'):<5}  "
              f"expected={c['expected']} got={c['predicted']}")

print(f"\n  ✗ Wrong ({result['total'] - result['correct']}):")
for c in result.get("cases", []):
    if not c["correct"]:
        print(f"    {c['case_id']:<6} ({c['category']:<8})  cam={c.get('camera_id','?'):<5}  "
              f"expected={c['expected']} got={c['predicted']}")

# Pull camera_id from questions file for display
try:
    qs_by_id = {q["id"]: q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}
    for c in result.get("cases", []):
        c.setdefault("camera_id", qs_by_id.get(c["case_id"], {}).get("camera_id", ""))
except Exception:
    pass


# ── STEP 6: Save results ──────────────────────────────────────────────────────
out_path = RESULTS_DIR / "junwen_1620_benchmark_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n  Results saved: {out_path}")
