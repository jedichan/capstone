"""Expand MEVID database from 6 cameras to 14 cameras.

Copies existing seed files from mevid_refined (6 cameras, 13-50-01 slot),
then ingests 8 additional cameras from the appropriate recording slots so
that all cameras referenced in the 50-question benchmark have DB coverage.

New cameras added:
  G339  — 13-50-01 (same slot as base DB)
  G336  — 16-20-01 (most annotation-consistent slot with G336+G506+G508)
  G420  — 16-20-01
  G326  — 16-20-01
  G419  — 16-20-01
  G423  — 16-35-01 (only slot available; also has G638)
  G638  — 16-35-01
  G330  — 11-20-00 (only slot with G330)

G299 is not available locally — those 2 questions remain unanswerable.

Usage:
    cd mevid_fastapi
    python expand_db_coverage.py [--skip-pipeline]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_AGENT = _ROOT / "agent"
for p in (str(_ROOT), str(_AGENT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load .env
env_file = _ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

os.environ.setdefault(
    "AGENT_MCQ_JUDGE_MODEL",
    os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3-vl-flash"),
)

SKIP_PIPELINE = "--skip-pipeline" in sys.argv

# ── Paths ──────────────────────────────────────────────────────────────────────
_VIDEO_DIR = _ROOT / "_data" / "mevid_slots"
_NS_BASE    = "mevid_refined"       # source: 6 cameras already processed
_NS_EXT    = "mevid_extended"       # destination: 14 cameras
_DB_BASE   = _HERE / "data" / "runtime" / _NS_BASE
_DB_EXT    = _HERE / "data" / "runtime" / _NS_EXT

_OUTPUT_EXT = str(_DB_EXT)

# New cameras to ingest (camera_id → video file)
_NEW_CAMERAS = {
    # Same slot as base DB — consistent timestamps
    "G339": _VIDEO_DIR / "2018-03-11.13-50-01.13-55-01.school.G339.r13.avi",
    # 16-20-01 — best overlap with the annotation set (has G336, G420, G419, G326, G506, G508)
    "G336": _VIDEO_DIR / "2018-03-11.16-20-01.16-25-01.school.G336.r13.avi",
    "G420": _VIDEO_DIR / "2018-03-11.16-20-01.16-25-01.school.G420.r13.avi",
    "G326": _VIDEO_DIR / "2018-03-11.16-20-01.16-25-01.admin.G326.r13.avi",
    "G419": _VIDEO_DIR / "2018-03-11.16-20-01.16-25-01.school.G419.r13.avi",
    # 16-35-01 — G423 and G638 only available here
    "G423": _VIDEO_DIR / "2018-03-11.16-35-01.16-40-00.school.G423.r13.avi",
    "G638": _VIDEO_DIR / "2018-03-11.16-35-01.16-40-01.school.G638.r13.avi",
    # 11-20-00 — only slot with G330
    "G330": _VIDEO_DIR / "2018-03-11.11-20-00.11-25-00.school.G330.r13.avi",
}

def _sep(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)


# ── STEP 1: Copy existing seeds ────────────────────────────────────────────────
_sep("STEP 1 — COPY EXISTING SEEDS (mevid_refined → mevid_extended)")

src_seeds = _DB_BASE / "seeds"
dst_seeds = _DB_EXT / "seeds"
dst_seeds.mkdir(parents=True, exist_ok=True)

copied = 0
if src_seeds.exists():
    for f in sorted(src_seeds.glob("*_events_vector_flat.json")):
        dest = dst_seeds / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            print(f"  copied: {f.name}")
            copied += 1
        else:
            print(f"  already exists: {f.name}")
            copied += 1
else:
    print(f"  WARNING: source seeds dir not found: {src_seeds}")

print(f"  Total seeds in extended dir: {len(list(dst_seeds.glob('*_events_vector_flat.json')))}")


# ── STEP 2: Build initial DB from copied seeds ─────────────────────────────────
_sep("STEP 2 — BUILD INITIAL DB FROM COPIED SEEDS")

import pipeline_service  # noqa: E402

if list(dst_seeds.glob("*_events_vector_flat.json")):
    res0 = pipeline_service.ingest_from_seeds(
        seeds_dir=str(dst_seeds),
        namespace=_NS_EXT,
        output_base=str(_DB_EXT / "runtime"),
        reset_existing=True,
    )
    print(f"  Initial DB built: sqlite_events={res0.get('sqlite_event_count',0)} "
          f"  chroma_docs={res0.get('chroma_doc_count',0)}")
else:
    print("  No seeds to build from — will rely on first ingest_from_single_video call")


# ── STEP 3: Ingest new cameras ─────────────────────────────────────────────────
_sep("STEP 3 — INGEST NEW CAMERAS")

if SKIP_PIPELINE:
    print("  --skip-pipeline passed, skipping to STEP 4")
else:
    # Verify files exist
    missing_vids = [cam for cam, p in _NEW_CAMERAS.items() if not p.exists()]
    if missing_vids:
        print(f"  WARNING: missing video files for: {missing_vids}")
        print("  These cameras will be skipped")

    total_t0 = time.perf_counter()
    cam_list = [(c, p) for c, p in _NEW_CAMERAS.items() if p.exists()]
    for idx, (cam_id, video_path) in enumerate(cam_list):
        print(f"\n  [{idx+1}/{len(cam_list)}] {cam_id} — {video_path.name}")
        t0 = time.perf_counter()
        try:
            res = pipeline_service.ingest_from_single_video(
                video_path=str(video_path),
                namespace=_NS_EXT,
                output_base=_OUTPUT_EXT,
                camera_id=cam_id,
                conf=0.35,
                iou=0.35,
                clip_refine=True,
                clip_refine_model=os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3-vl-flash"),
                appearance_refine=True,
                scene_profile=False,
                reset_existing=True,  # rebuilds DB from ALL accumulated seeds each time
            )
            elapsed = round(time.perf_counter() - t0, 1)
            print(f"    done in {elapsed}s — "
                  f"raw_events={res.get('raw_event_count',0)}  "
                  f"seed_events={res.get('seed_event_count',0)}")
        except Exception as exc:
            import traceback
            print(f"    ERROR: {exc}")
            traceback.print_exc()

    total_elapsed = round(time.perf_counter() - total_t0, 1)
    print(f"\n  All cameras done in {total_elapsed}s total  "
          f"({total_elapsed/60:.1f} min)")

    # Show final seed count
    n_seeds = len(list(dst_seeds.glob("*_events_vector_flat.json")))
    print(f"  Total seed files in extended DB: {n_seeds}")


# ── STEP 4: Run MCQ eval on extended DB ────────────────────────────────────────
_sep("STEP 4 — MCQ EVAL ON EXTENDED DB")

from service import agent_service       # noqa: E402
from eval_mcq_service import run_mcq_eval  # noqa: E402

_DB_EXT_RUNTIME = _DB_EXT / "runtime"
sqlite = _DB_EXT_RUNTIME / "episodic_events.sqlite"
chroma = _DB_EXT_RUNTIME / "chroma"

if not sqlite.exists():
    print(f"  ERROR: extended DB not found at {sqlite}")
    sys.exit(1)

agent_service.select_database(
    sqlite_path=str(sqlite),
    chroma_path=str(chroma),
    namespace=_NS_EXT,
)
print(f"  Selected DB: {_NS_EXT}")

def _query_fn(q: str) -> dict:
    return agent_service.run_query(
        query=q, thread_id=None, user_id="mcq_eval_ext",
        include_rows=True, include_node_trace=False, top_k_rows=5,
    )

t0 = time.perf_counter()
ext_result = run_mcq_eval(questions_file=None, agent_query_fn=_query_fn)
elapsed = round(time.perf_counter() - t0, 1)

print(f"\n  Extended DB MCQ: accuracy={ext_result['accuracy']:.1%}  "
      f"mean_iou={ext_result['mean_iou']:.3f}  ({elapsed}s)")
print("  By category:")
for cat, s in ext_result.get("by_category", {}).items():
    print(f"    {cat:<10} acc={s['accuracy']:.1%}  iou={s['mean_iou']:.3f}  "
          f"({s['correct']}/{s['total']})")


# ── STEP 5: Camera coverage breakdown ─────────────────────────────────────────
_sep("STEP 5 — COVERAGE BREAKDOWN")

# Load questions to check camera coverage
qs_file = _ROOT / "tests" / "mevid_xlsx" / "mcq_questions_50.json"
qs = json.load(open(qs_file, encoding="utf-8"))

COVERED_CAMS = {"G329", "G506", "G508", "G328", "G424", "G421",  # original 6
                "G339", "G336", "G420", "G326", "G419", "G423", "G638", "G330"}  # 8 new
UNCOVERED_CAMS = {"G299"}  # not downloaded

covered_qs = [q for q in qs if q.get("camera_id","") in COVERED_CAMS or not q.get("camera_id")]
uncovered_qs = [q for q in qs if q.get("camera_id","") in UNCOVERED_CAMS]

def _acc(cases: list[dict], q_ids: set) -> str:
    sub = [c for c in ext_result.get("cases",[]) if c["case_id"] in q_ids]
    if not sub:
        return "—"
    correct = sum(1 for c in sub if c["correct"])
    return f"{correct}/{len(sub)} ({correct/len(sub):.1%})"

covered_ids = {q["id"] for q in covered_qs}
uncovered_ids = {q["id"] for q in uncovered_qs}
other_ids = {q["id"] for q in qs} - covered_ids - uncovered_ids

print(f"\n  Cameras covered by DB ({len(COVERED_CAMS)}):  {len(covered_qs)} questions  acc={_acc(ext_result.get('cases',[]), covered_ids)}")
print(f"  Cameras NOT available (G299):        {len(uncovered_qs)} questions  acc={_acc(ext_result.get('cases',[]), uncovered_ids)}")
print(f"  Cross-camera / no camera filter:     {len(other_ids)} questions  acc={_acc(ext_result.get('cases',[]), other_ids)}")

# Comparison with previous result
_sep("COMPARISON vs PREVIOUS RESULTS")
PREV_RESULT = {
    "total": 50, "correct": 12, "accuracy": 0.24, "mean_iou": 0.1417,
    "by_category": {
        "Search":   {"correct": 5,  "total": 12, "accuracy": 0.417, "mean_iou": 0.146},
        "Temporal": {"correct": 3,  "total": 10, "accuracy": 0.30,  "mean_iou": 0.119},
        "Event":    {"correct": 3,  "total": 10, "accuracy": 0.30,  "mean_iou": 0.000},
        "Activity": {"correct": 0,  "total": 10, "accuracy": 0.00,  "mean_iou": 0.159},
        "Counting": {"correct": 0,  "total": 5,  "accuracy": 0.00,  "mean_iou": 0.000},
        "Anomaly":  {"correct": 1,  "total": 3,  "accuracy": 0.333, "mean_iou": 0.000},
    }
}

print(f"\n  {'Category':<10}  {'Before':>8}  {'After':>8}  {'Delta':>8}  {'tIoU Before':>12}  {'tIoU After':>10}")
print("  " + "-" * 70)
for cat in ("Search","Temporal","Event","Activity","Counting","Anomaly"):
    prev = PREV_RESULT["by_category"].get(cat, {})
    curr = ext_result.get("by_category", {}).get(cat, {})
    p_acc = prev.get("accuracy", 0)
    c_acc = curr.get("accuracy", 0)
    delta = c_acc - p_acc
    sign = "+" if delta >= 0 else ""
    print(f"  {cat:<10}  {p_acc:>7.1%}  {c_acc:>7.1%}  {sign}{delta:>7.1%}  "
          f"{prev.get('mean_iou',0):>11.3f}  {curr.get('mean_iou',0):>9.3f}")
print("  " + "-" * 70)
p_oa = PREV_RESULT["accuracy"]
c_oa = ext_result.get("accuracy",0)
delta_oa = c_oa - p_oa
sign_oa = "+" if delta_oa >= 0 else ""
print(f"  {'Overall':<10}  {p_oa:>7.1%}  {c_oa:>7.1%}  {sign_oa}{delta_oa:>7.1%}  "
      f"{PREV_RESULT['mean_iou']:>11.3f}  {ext_result.get('mean_iou',0):>9.3f}")

# Save result
out_path = _ROOT / "results" / "mcq50_extended_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(ext_result, f, ensure_ascii=False, indent=2)
print(f"\n  Results saved to: {out_path}")
