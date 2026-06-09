"""tests/expand_db_full.py
===========================
Expand MEVID database to ALL locally-available camera-slot combinations,
then re-run the 50-question MCQ benchmark with camera-id filtering ON.

ANNOTATOR → SLOT MAPPING (from teammate_test_writing_guide.md):
  Person 1 (Cz)           → 11-20 : G329 G330 G336 G419 G420 G421 G423 G508
  Person 2 (Shunyu)       → 11-55 : (videos NOT downloaded — skip)
  Person 3 (MinqiangYao)  → 13-50 : G329 G421 G424 G328 G506 G339 G508   ← base DB
  Person 4 (QuanjieSheng) → 14-20 : G328 G419 G339 G423 G421 G508 G506
  Person 5 (JunwenChen)   → 16-20 : G326 G328 G336 G419 G420 G506 G508
  Person 6 (副本)          → 16-35 : G326 G328 G329 G336 G339 G419 G420 G423 G506 G638

mevid_extended (14 seeds already built):
  13-50 : G329 G506 G508 G328 G421 G424 G339
  16-20 : G336 G419 G420 G326
  16-35 : G423 G638
  11-20 : G330

This script adds everything still available on disk into mevid_full.

Tiered ingestion — use --tier <1|2|3|all> to ingest incrementally:
  Tier 1  (11-20 additions, Cz's slot — highest expected impact)
      G329 G336 G419 G420 G421 G423 G508
  Tier 2  (14-20, QuanjieSheng's slot)
      G419 G423 G421 G508 G506 G328 G339
  Tier 3  (16-20 gap cameras + 16-35 gap cameras)
      16-20: G328 G329 G506 G508
      16-35: G328 G329 G336 G339 G419 G420 G326 G506

Usage (run from repo root):
    C:\\Users\\17809\\anaconda3\\envs\\capstone\\python.exe tests/expand_db_full.py
    C:\\Users\\17809\\anaconda3\\envs\\capstone\\python.exe tests/expand_db_full.py --skip-pipeline
    C:\\Users\\17809\\anaconda3\\envs\\capstone\\python.exe tests/expand_db_full.py --tier 1
    C:\\Users\\17809\\anaconda3\\envs\\capstone\\python.exe tests/expand_db_full.py --tier 1 --skip-pipeline
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
FASTAPI = ROOT / "mevid_fastapi"
AGENT   = ROOT / "agent"
for p in (str(ROOT), str(AGENT), str(FASTAPI)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Load .env ─────────────────────────────────────────────────────────────────
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

os.environ.setdefault(
    "AGENT_MCQ_JUDGE_MODEL",
    os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus"),
)

# ── CLI flags ─────────────────────────────────────────────────────────────────
SKIP_PIPELINE = "--skip-pipeline" in sys.argv

_tier_arg = "all"
for _i, _a in enumerate(sys.argv[1:], 1):
    if _a.startswith("--tier"):
        _parts = _a.split("=", 1)
        if len(_parts) == 2:
            _tier_arg = _parts[1].strip()
        elif _i < len(sys.argv) - 1:
            _tier_arg = sys.argv[_i + 1].strip()

# ── Paths ─────────────────────────────────────────────────────────────────────
_VIDEO_DIR  = ROOT / "_data" / "mevid_slots"
_NS_EXT     = "mevid_extended"   # source: 14-camera DB already built
_NS_FULL    = "mevid_full"       # destination: all available cameras
_DB_EXT     = FASTAPI / "data" / "runtime" / _NS_EXT
_DB_FULL    = FASTAPI / "data" / "runtime" / _NS_FULL

# ── New cameras to ingest (not yet in mevid_extended) ─────────────────────────
# Format: (camera_id, slot_label, video_path)
# Tier 1 — Cz's 11-20 slot (highest priority: G336×7 questions, G329×5 questions)
_TIER1 = [
    ("G329", "11-20", _VIDEO_DIR / "2018-03-11.11-20-00.11-25-00.admin.G329.r13.avi"),
    ("G336", "11-20", _VIDEO_DIR / "2018-03-11.11-20-00.11-25-00.school.G336.r13.avi"),
    ("G419", "11-20", _VIDEO_DIR / "2018-03-11.11-20-00.11-25-00.school.G419.r13.avi"),
    ("G420", "11-20", _VIDEO_DIR / "2018-03-11.11-20-00.11-25-00.school.G420.r13.avi"),
    ("G421", "11-20", _VIDEO_DIR / "2018-03-11.11-20-00.11-25-00.school.G421.r13.avi"),
    ("G423", "11-20", _VIDEO_DIR / "2018-03-11.11-20-00.11-25-00.school.G423.r13.avi"),
    ("G508", "11-20", _VIDEO_DIR / "2018-03-11.11-20-00.11-25-00.bus.G508.r13.avi"),
]

# Tier 2 — QuanjieSheng's 14-20 slot
_TIER2 = [
    ("G419", "14-20", _VIDEO_DIR / "2018-03-11.14-20-01.14-25-01.school.G419.r13.avi"),
    ("G423", "14-20", _VIDEO_DIR / "2018-03-11.14-20-01.14-25-01.school.G423.r13.avi"),
    ("G421", "14-20", _VIDEO_DIR / "2018-03-11.14-20-01.14-25-01.school.G421.r13.avi"),
    ("G508", "14-20", _VIDEO_DIR / "2018-03-11.14-20-01.14-25-01.bus.G508.r13.avi"),
    ("G506", "14-20", _VIDEO_DIR / "2018-03-11.14-20-01.14-25-01.bus.G506.r13.avi"),
    ("G328", "14-20", _VIDEO_DIR / "2018-03-11.14-20-01.14-25-01.school.G328.r13.avi"),
    ("G339", "14-20", _VIDEO_DIR / "2018-03-11.14-20-01.14-25-01.school.G339.r13.avi"),
]

# Tier 3 — gap fills for 16-20 (JunwenChen) and 16-35 (副本)
_TIER3 = [
    # 16-20 cameras missing from mevid_extended
    ("G328", "16-20", _VIDEO_DIR / "2018-03-11.16-20-01.16-25-01.school.G328.r13.avi"),
    ("G329", "16-20", _VIDEO_DIR / "2018-03-11.16-20-01.16-25-01.admin.G329.r13.avi"),
    ("G506", "16-20", _VIDEO_DIR / "2018-03-11.16-20-01.16-25-01.bus.G506.r13.avi"),
    ("G508", "16-20", _VIDEO_DIR / "2018-03-11.16-20-01.16-25-01.bus.G508.r13.avi"),
    # 16-35 cameras missing from mevid_extended
    ("G328", "16-35", _VIDEO_DIR / "2018-03-11.16-35-01.16-40-01.school.G328.r13.avi"),
    ("G329", "16-35", _VIDEO_DIR / "2018-03-11.16-35-01.16-40-01.admin.G329.r13.avi"),
    ("G336", "16-35", _VIDEO_DIR / "2018-03-11.16-35-01.16-40-01.school.G336.r13.avi"),
    ("G339", "16-35", _VIDEO_DIR / "2018-03-11.16-35-01.16-40-01.school.G339.r13.avi"),
    ("G419", "16-35", _VIDEO_DIR / "2018-03-11.16-35-01.16-40-01.school.G419.r13.avi"),
    ("G420", "16-35", _VIDEO_DIR / "2018-03-11.16-35-01.16-40-01.school.G420.r13.avi"),
    ("G326", "16-35", _VIDEO_DIR / "2018-03-11.16-35-01.16-40-01.admin.G326.r13.avi"),
    ("G506", "16-35", _VIDEO_DIR / "2018-03-11.16-35-01.16-40-01.bus.G506.r13.avi"),
]

_TIER_MAP = {
    "1":   _TIER1,
    "2":   _TIER1 + _TIER2,
    "3":   _TIER1 + _TIER2 + _TIER3,
    "all": _TIER1 + _TIER2 + _TIER3,
}
NEW_CAMERAS = _TIER_MAP.get(_tier_arg, _TIER1 + _TIER2 + _TIER3)


def _sep(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)


# ── STEP 1: Copy mevid_extended seeds → mevid_full ───────────────────────────
_sep(f"STEP 1 — COPY mevid_extended SEEDS → {_NS_FULL}")

src_seeds = _DB_EXT / "seeds"
dst_seeds = _DB_FULL / "seeds"
dst_seeds.mkdir(parents=True, exist_ok=True)

if src_seeds.exists():
    for f in sorted(src_seeds.glob("*_events_vector_flat.json")):
        dest = dst_seeds / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            print(f"  copied : {f.name}")
        else:
            print(f"  exists : {f.name}")
else:
    print(f"  WARNING: source seeds dir not found: {src_seeds}")
    print("  Starting from scratch in mevid_full")

n_seeds = len(list(dst_seeds.glob("*_events_vector_flat.json")))
print(f"  Total seeds in {_NS_FULL}: {n_seeds}")


# ── STEP 2: Build initial DB from copied seeds ────────────────────────────────
_sep("STEP 2 — BUILD INITIAL DB FROM SEEDS")

import pipeline_service  # noqa: E402

if not SKIP_PIPELINE:
    seed_files = list(dst_seeds.glob("*_events_vector_flat.json"))
    if seed_files:
        res0 = pipeline_service.ingest_from_seeds(
            seeds_dir=str(dst_seeds),
            namespace=_NS_FULL,
            output_base=str(_DB_FULL / "runtime"),
            reset_existing=True,
        )
        print(f"  Initial DB: sqlite_rows={res0.get('sqlite_rows', 0)}  "
              f"chroma_child={res0.get('chroma_child_records', 0)}")
    else:
        print("  No seeds found — will build from first ingest call")
else:
    print("  --skip-pipeline: skipping build")


# ── STEP 3: Ingest new camera-slot combinations ───────────────────────────────
_sep(f"STEP 3 — INGEST NEW CAMERAS  (tier={_tier_arg})")

if SKIP_PIPELINE:
    print("  --skip-pipeline: skipping ingestion")
else:
    missing = [(c, s, p) for c, s, p in NEW_CAMERAS if not p.exists()]
    if missing:
        print(f"  WARNING: {len(missing)} video files missing on disk:")
        for c, s, p in missing:
            print(f"    {c} ({s}): {p.name}")

    cam_list = [(c, s, p) for c, s, p in NEW_CAMERAS if p.exists()]
    print(f"  Will ingest {len(cam_list)} camera-slot combinations\n")

    total_t0 = time.perf_counter()
    for idx, (cam_id, slot, video_path) in enumerate(cam_list):
        print(f"  [{idx+1}/{len(cam_list)}] {cam_id} ({slot}) — {video_path.name}")
        t0 = time.perf_counter()
        try:
            res = pipeline_service.ingest_from_single_video(
                video_path=str(video_path),
                namespace=_NS_FULL,
                output_base=str(_DB_FULL),
                camera_id=cam_id,
                conf=0.35,
                iou=0.35,
                clip_refine=True,
                clip_refine_model=os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus"),
                appearance_refine=True,
                scene_profile=False,
                reset_existing=True,
            )
            elapsed = round(time.perf_counter() - t0, 1)
            print(f"    done in {elapsed}s — "
                  f"raw_events={res.get('raw_event_count', 0)}  "
                  f"seed_events={res.get('seed_event_count', 0)}")
        except Exception as exc:
            import traceback
            print(f"    ERROR: {exc}")
            traceback.print_exc()

    total_elapsed = round(time.perf_counter() - total_t0, 1)
    n_seeds_final = len(list(dst_seeds.glob("*_events_vector_flat.json")))
    print(f"\n  All done in {total_elapsed}s ({total_elapsed/60:.1f} min)")
    print(f"  Total seeds in {_NS_FULL}: {n_seeds_final}")


# ── STEP 4: Run MCQ eval with camera-id filter ───────────────────────────────
_sep("STEP 4 — MCQ EVAL (camera-filter ON)")

from service import agent_service           # noqa: E402
from eval_mcq_service import run_mcq_eval   # noqa: E402

_DB_FULL_RUNTIME = _DB_FULL / "runtime"
sqlite = _DB_FULL_RUNTIME / "episodic_events.sqlite"
chroma = _DB_FULL_RUNTIME / "chroma"

if not sqlite.exists():
    print(f"  ERROR: DB not found at {sqlite}")
    sys.exit(1)

agent_service.select_database(
    sqlite_path=str(sqlite),
    chroma_path=str(chroma),
    namespace=_NS_FULL,
)
print(f"  Selected DB: {_NS_FULL}")
print(f"  Seeds available: {len(list(dst_seeds.glob('*_events_vector_flat.json')))}")


def _query_fn(q: str) -> dict:
    return agent_service.run_query(
        query=q, thread_id=None, user_id="mcq_eval_full",
        include_rows=True, include_node_trace=False, top_k_rows=5,
    )


t0_eval = time.perf_counter()
result = run_mcq_eval(
    questions_file=str(ROOT / "tests" / "mevid_xlsx" / "mcq_questions_50.json"),
    agent_query_fn=_query_fn,
    use_camera_filter=True,
)
eval_elapsed = round(time.perf_counter() - t0_eval, 1)

print(f"\n  {_NS_FULL} MCQ (camera-filter=ON): "
      f"accuracy={result['accuracy']:.1%}  "
      f"mean_iou={result['mean_iou']:.3f}  "
      f"({eval_elapsed}s)")
print("  By category:")
for cat, s in result.get("by_category", {}).items():
    print(f"    {cat:<10} acc={s['accuracy']:.1%}  iou={s['mean_iou']:.3f}  "
          f"({s['correct']}/{s['total']})")


# ── STEP 5: Comparison table ──────────────────────────────────────────────────
_sep("STEP 5 — COMPARISON vs PREVIOUS RESULTS")

PREV = {
    # Baseline: mevid_refined (6 cameras, no camera filter)
    "Baseline (6-cam, no filter)": {
        "accuracy": 0.24, "mean_iou": 0.1417,
        "by_category": {
            "Search":   {"correct": 5,  "total": 12, "accuracy": 0.417, "mean_iou": 0.146},
            "Temporal": {"correct": 3,  "total": 10, "accuracy": 0.30,  "mean_iou": 0.119},
            "Event":    {"correct": 3,  "total": 10, "accuracy": 0.30,  "mean_iou": 0.000},
            "Activity": {"correct": 0,  "total": 10, "accuracy": 0.00,  "mean_iou": 0.159},
            "Counting": {"correct": 0,  "total": 5,  "accuracy": 0.00,  "mean_iou": 0.000},
            "Anomaly":  {"correct": 1,  "total": 3,  "accuracy": 0.333, "mean_iou": 0.000},
        },
    },
    # Extended DB (14 cameras, no filter)
    "Extended (14-cam, no filter)": {
        "accuracy": 0.30, "mean_iou": 0.1346,
        "by_category": {
            "Search":   {"correct": 2,  "total": 12, "accuracy": 0.167, "mean_iou": 0.093},
            "Temporal": {"correct": 4,  "total": 10, "accuracy": 0.40,  "mean_iou": 0.196},
            "Event":    {"correct": 4,  "total": 10, "accuracy": 0.40,  "mean_iou": 0.201},
            "Activity": {"correct": 4,  "total": 10, "accuracy": 0.40,  "mean_iou": 0.184},
            "Counting": {"correct": 0,  "total": 5,  "accuracy": 0.00,  "mean_iou": 0.000},
            "Anomaly":  {"correct": 1,  "total": 3,  "accuracy": 0.333, "mean_iou": 0.000},
        },
    },
    # Extended DB (14 cameras, camera filter ON)
    "Extended (14-cam, cam-filter)": {
        "accuracy": 0.36, "mean_iou": 0.081,
        "by_category": {
            "Search":   {"correct": 3,  "total": 12, "accuracy": 0.25,  "mean_iou": 0.045},
            "Temporal": {"correct": 5,  "total": 10, "accuracy": 0.50,  "mean_iou": 0.123},
            "Event":    {"correct": 4,  "total": 10, "accuracy": 0.40,  "mean_iou": 0.111},
            "Activity": {"correct": 5,  "total": 10, "accuracy": 0.50,  "mean_iou": 0.108},
            "Counting": {"correct": 0,  "total": 5,  "accuracy": 0.00,  "mean_iou": 0.000},
            "Anomaly":  {"correct": 1,  "total": 3,  "accuracy": 0.333, "mean_iou": 0.000},
        },
    },
}

cats = ("Search", "Temporal", "Event", "Activity", "Counting", "Anomaly")
col_names = list(PREV.keys()) + [f"Full ({_tier_arg}-tier, cam-filter)"]

# Header
header = f"  {'Category':<10}"
for col in col_names:
    short = col[:18]
    header += f"  {short:>18}"
print(header)
print("  " + "-" * (10 + len(col_names) * 20))

for cat in cats:
    row = f"  {cat:<10}"
    for col, pdata in PREV.items():
        acc = pdata["by_category"].get(cat, {}).get("accuracy", 0)
        row += f"  {acc:>17.1%}"
    curr_acc = result.get("by_category", {}).get(cat, {}).get("accuracy", 0)
    curr_iou = result.get("by_category", {}).get(cat, {}).get("mean_iou", 0)
    row += f"  {curr_acc:>14.1%} ({curr_iou:.3f})"
    print(row)

print("  " + "-" * (10 + len(col_names) * 20))
overall_row = f"  {'Overall':<10}"
for col, pdata in PREV.items():
    overall_row += f"  {pdata['accuracy']:>17.1%}"
overall_row += f"  {result['accuracy']:>14.1%} ({result['mean_iou']:.3f})"
print(overall_row)

# ── Save results ──────────────────────────────────────────────────────────────
out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"mcq50_full_tier{_tier_arg}_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n  Results saved to: {out_path}")

# Per-question breakdown for wrong answers
print("\n  Wrong answers:")
for c in result.get("cases", []):
    if not c["correct"]:
        print(f"    {c['case_id']} ({c['category']:<8}) "
              f"expected={c['expected']} got={c['predicted']}")
