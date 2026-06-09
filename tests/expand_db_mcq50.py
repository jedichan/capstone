"""tests/expand_db_mcq50.py
===========================
Expand the MEVID database from 6 cameras to 14 cameras, then re-run the
50-question MCQ benchmark to measure the accuracy improvement.

EXISTING DB  (mevid_refined, 13-50 slot):
  G329, G506, G508, G328, G421, G424

NEW CAMERAS added:
  G339  — 13-50 slot  (same session as base DB)
  G336  — 16-20 slot  (JunwenChen's annotation set includes G336+G506+G508)
  G420  — 16-20 slot
  G326  — 16-20 slot
  G419  — 16-20 slot
  G423  — 16-35 slot  (not available in 16-20)
  G638  — 16-35 slot  (only available here)
  G330  — 11-20 slot  (only available here)

  G299 — not downloaded, those 2 questions remain unanswerable.

Usage:
    cd D:\\Capstone\\CS32-1-surveillance-video-query-llm
    python tests/expand_db_mcq50.py                # full run
    python tests/expand_db_mcq50.py --skip-pipeline  # eval only (DB already built)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
FASTAPI    = ROOT / "mevid_fastapi"
AGENT      = ROOT / "agent"
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

SKIP_PIPELINE    = "--skip-pipeline"    in sys.argv
CAMERA_FILTER    = "--camera-filter"    in sys.argv   # add camera_id to query + post-filter rows
COMPARE_BOTH     = "--compare-both"     in sys.argv   # run both modes and compare

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR  = ROOT / "_data" / "mevid_slots"
NS_BASE    = "mevid_refined"
NS_EXT     = "mevid_extended"
DB_BASE    = FASTAPI / "data" / "runtime" / NS_BASE
DB_EXT     = FASTAPI / "data" / "runtime" / NS_EXT

# 8 new cameras  →  (camera_id, video_stem, slot_label)
NEW_CAMERAS = [
    # Same 13-50 slot — consistent timestamps with base DB
    ("G339", "2018-03-11.13-50-01.13-55-01.school.G339.r13.avi",  "13-50"),
    # 16-20 slot — best match for annotations with G336+G420+G419+G326
    ("G336", "2018-03-11.16-20-01.16-25-01.school.G336.r13.avi",  "16-20"),
    ("G420", "2018-03-11.16-20-01.16-25-01.school.G420.r13.avi",  "16-20"),
    ("G326", "2018-03-11.16-20-01.16-25-01.admin.G326.r13.avi",   "16-20"),
    ("G419", "2018-03-11.16-20-01.16-25-01.school.G419.r13.avi",  "16-20"),
    # 16-35 slot — G423 and G638 only available here
    ("G423", "2018-03-11.16-35-01.16-40-00.school.G423.r13.avi",  "16-35"),
    ("G638", "2018-03-11.16-35-01.16-40-01.school.G638.r13.avi",  "16-35"),
    # 11-20 slot — G330 only available here
    ("G330", "2018-03-11.11-20-00.11-25-00.school.G330.r13.avi",  "11-20"),
]

QUESTIONS_FILE = ROOT / "tests" / "mevid_xlsx" / "mcq_questions_50.json"

# Previous benchmark-B result (for comparison table)
PREV = {
    "accuracy": 0.24, "mean_iou": 0.1417,
    "by_category": {
        "Search":   {"correct": 5,  "total": 12, "accuracy": 0.417, "mean_iou": 0.146},
        "Temporal": {"correct": 3,  "total": 10, "accuracy": 0.30,  "mean_iou": 0.119},
        "Event":    {"correct": 3,  "total": 10, "accuracy": 0.30,  "mean_iou": 0.000},
        "Activity": {"correct": 0,  "total": 10, "accuracy": 0.00,  "mean_iou": 0.159},
        "Counting": {"correct": 0,  "total": 5,  "accuracy": 0.00,  "mean_iou": 0.000},
        "Anomaly":  {"correct": 1,  "total": 3,  "accuracy": 0.333, "mean_iou": 0.000},
    },
}


def sep(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print()


# ══════════════════════════════════════════════════════════════════
# STEP 1 — Copy existing seeds into new namespace
# ══════════════════════════════════════════════════════════════════
sep("STEP 1  Copy existing seeds  mevid_refined → mevid_extended")

src_seeds = DB_BASE / "seeds"
dst_seeds = DB_EXT  / "seeds"
dst_seeds.mkdir(parents=True, exist_ok=True)

for f in sorted(src_seeds.glob("*_events_vector_flat.json")):
    dst = dst_seeds / f.name
    if not dst.exists():
        shutil.copy2(f, dst)
        print(f"  copied : {f.name}")
    else:
        print(f"  exists : {f.name}")

n_seeds = len(list(dst_seeds.glob("*_events_vector_flat.json")))
print(f"\n  seeds in {NS_EXT}: {n_seeds}")


# ══════════════════════════════════════════════════════════════════
# STEP 2 — Build initial DB from copied seeds
# ══════════════════════════════════════════════════════════════════
sep("STEP 2  Build initial DB from copied seeds")

import pipeline_service  # noqa: E402

res0 = pipeline_service.ingest_from_seeds(
    seeds_dir=str(dst_seeds),
    namespace=NS_EXT,
    output_base=str(DB_EXT / "runtime"),
    reset_existing=True,
)
print(f"  sqlite events : {res0.get('sqlite_event_count', 0)}")
print(f"  chroma docs   : {res0.get('chroma_doc_count', 0)}")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — Ingest the 8 new cameras
# ══════════════════════════════════════════════════════════════════
sep("STEP 3  Ingest new cameras")

if SKIP_PIPELINE:
    print("  --skip-pipeline → skipping to STEP 4")
else:
    missing = [(c, f, s) for c, f, s in NEW_CAMERAS if not (VIDEO_DIR / f).exists()]
    if missing:
        print("  WARNING – missing video files (will skip):")
        for c, f, _ in missing:
            print(f"    {c}: {f}")

    runnable = [(c, VIDEO_DIR / f, s) for c, f, s in NEW_CAMERAS if (VIDEO_DIR / f).exists()]
    total_t0 = time.perf_counter()

    for idx, (cam_id, video_path, slot) in enumerate(runnable, 1):
        print(f"  [{idx}/{len(runnable)}] {cam_id}  slot={slot}  {video_path.name}")
        t0 = time.perf_counter()
        try:
            res = pipeline_service.ingest_from_single_video(
                video_path=str(video_path),
                namespace=NS_EXT,
                output_base=str(DB_EXT),
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
            print(f"      done in {elapsed}s  "
                  f"raw={res.get('raw_event_count', 0)}  "
                  f"seed={res.get('seed_event_count', 0)}")
        except Exception as exc:
            import traceback
            print(f"      ERROR: {exc}")
            traceback.print_exc()

    total_elapsed = round(time.perf_counter() - total_t0, 1)
    n_seeds_final = len(list(dst_seeds.glob("*_events_vector_flat.json")))
    print(f"\n  All cameras done in {total_elapsed}s ({total_elapsed/60:.1f} min)")
    print(f"  Total seed files: {n_seeds_final}")


# ══════════════════════════════════════════════════════════════════
# STEP 4 — MCQ eval on extended DB
# ══════════════════════════════════════════════════════════════════
sep("STEP 4  MCQ eval on extended DB  (50 questions)")

from service import agent_service       # noqa: E402
from eval_mcq_service import run_mcq_eval  # noqa: E402

sqlite = DB_EXT / "runtime" / "episodic_events.sqlite"
chroma = DB_EXT / "runtime" / "chroma"

if not sqlite.exists():
    sys.exit(f"Extended DB not found: {sqlite}\nRun without --skip-pipeline first.")

agent_service.select_database(
    sqlite_path=str(sqlite),
    chroma_path=str(chroma),
    namespace=NS_EXT,
)
print(f"  DB: {NS_EXT}  ({sqlite})")

def _query(q: str) -> dict:
    return agent_service.run_query(
        query=q, thread_id=None, user_id="mcq_ext",
        include_rows=True, include_node_trace=False, top_k_rows=5,
    )

if COMPARE_BOTH:
    print("  Running WITHOUT camera filter ...")
    t0 = time.perf_counter()
    result_no_filter = run_mcq_eval(
        questions_file=str(QUESTIONS_FILE), agent_query_fn=_query,
        use_camera_filter=False,
    )
    print(f"  no-filter:    accuracy={result_no_filter['accuracy']:.1%}  "
          f"mean_iou={result_no_filter['mean_iou']:.3f}  ({round(time.perf_counter()-t0,1)}s)")

    print("  Running WITH camera filter ...")
    t0 = time.perf_counter()
    result = run_mcq_eval(
        questions_file=str(QUESTIONS_FILE), agent_query_fn=_query,
        use_camera_filter=True,
    )
    elapsed = round(time.perf_counter() - t0, 1)
    print(f"  with-filter:  accuracy={result['accuracy']:.1%}  "
          f"mean_iou={result['mean_iou']:.3f}  ({elapsed}s)")
else:
    t0 = time.perf_counter()
    result = run_mcq_eval(
        questions_file=str(QUESTIONS_FILE), agent_query_fn=_query,
        use_camera_filter=CAMERA_FILTER,
    )
    elapsed = round(time.perf_counter() - t0, 1)
    mode = "camera-filter=ON" if CAMERA_FILTER else "camera-filter=OFF"
    print(f"\n  [{mode}] accuracy = {result['accuracy']:.1%}   mean_iou = {result['mean_iou']:.3f}   ({elapsed}s)")


# ══════════════════════════════════════════════════════════════════
# STEP 5 — Comparison table
# ══════════════════════════════════════════════════════════════════
sep("STEP 5  Comparison: before vs after DB expansion")

CATS = ("Search", "Temporal", "Event", "Activity", "Counting", "Anomaly")

result_nf = result_no_filter if COMPARE_BOTH else result
result_wf = result

hdr_nf = "No-filter" if COMPARE_BOTH else ("w/filter" if CAMERA_FILTER else "After")
hdr_wf = "w/filter"  if COMPARE_BOTH else ""

if COMPARE_BOTH:
    print(f"  {'Category':<10}  {'Before':>7}  {hdr_nf:>9}  {hdr_wf:>9}  "
          f"{'tIoU(nf)':>10}  {'tIoU(wf)':>10}")
    print("  " + "─" * 72)
    for cat in CATS:
        p   = PREV["by_category"].get(cat, {})
        cnf = result_nf.get("by_category", {}).get(cat, {})
        cwf = result_wf.get("by_category", {}).get(cat, {})
        print(f"  {cat:<10}  {p.get('accuracy',0):>6.1%}  "
              f"{cnf.get('accuracy',0):>8.1%}  {cwf.get('accuracy',0):>8.1%}  "
              f"{cnf.get('mean_iou',0):>10.3f}  {cwf.get('mean_iou',0):>9.3f}  "
              f"({cwf.get('correct',0)}/{cwf.get('total',0)})")
    print("  " + "─" * 72)
    print(f"  {'Overall':<10}  {PREV['accuracy']:>6.1%}  "
          f"{result_nf.get('accuracy',0):>8.1%}  {result_wf.get('accuracy',0):>8.1%}  "
          f"{PREV['mean_iou']:>10.3f}  {result_wf.get('mean_iou',0):>9.3f}  "
          f"({result_wf.get('correct',0)}/{result_wf.get('total',0)})")
else:
    print(f"  {'Category':<10}  {'Before':>7}  {hdr_nf:>7}  {'Δ':>6}  "
          f"{'tIoU Before':>11}  {'tIoU After':>10}")
    print("  " + "─" * 66)
    for cat in CATS:
        p  = PREV["by_category"].get(cat, {})
        c  = result.get("by_category", {}).get(cat, {})
        pa = p.get("accuracy", 0)
        ca = c.get("accuracy", 0)
        d  = ca - pa
        sign = "+" if d >= 0 else ""
        print(f"  {cat:<10}  {pa:>6.1%}  {ca:>6.1%}  {sign}{d:>5.1%}  "
              f"{p.get('mean_iou', 0):>11.3f}  {c.get('mean_iou', 0):>9.3f}  "
              f"({c.get('correct',0)}/{c.get('total',0)})")
    print("  " + "─" * 66)
    prev_oa = PREV["accuracy"]
    curr_oa = result.get("accuracy", 0)
    delta_oa = curr_oa - prev_oa
    sign_oa = "+" if delta_oa >= 0 else ""
    print(f"  {'Overall':<10}  {prev_oa:>6.1%}  {curr_oa:>6.1%}  "
          f"{sign_oa}{delta_oa:>5.1%}  "
          f"{PREV['mean_iou']:>11.3f}  {result.get('mean_iou', 0):>9.3f}  "
          f"({result.get('correct',0)}/{result.get('total',0)})")

# Camera coverage breakdown
print()
qs = json.load(open(QUESTIONS_FILE, encoding="utf-8"))
COVERED   = {"G329","G506","G508","G328","G424","G421",
             "G339","G336","G420","G326","G419","G423","G638","G330"}
UNCOVERED = {"G299"}

cases = {c["case_id"]: c for c in result.get("cases", [])}
def _cov_acc(ids):
    sub = [cases[i] for i in ids if i in cases]
    if not sub: return "—"
    ok = sum(1 for c in sub if c["correct"])
    return f"{ok}/{len(sub)} ({ok/len(sub):.1%})"

covered_ids   = {q["id"] for q in qs if q.get("camera_id","") in COVERED}
uncovered_ids = {q["id"] for q in qs if q.get("camera_id","") in UNCOVERED}
other_ids     = {q["id"] for q in qs} - covered_ids - uncovered_ids

print(f"  Cameras in DB ({len(COVERED)}):       {len(covered_ids):>2} questions  {_cov_acc(covered_ids)}")
print(f"  Cameras unavailable (G299):  {len(uncovered_ids):>2} questions  {_cov_acc(uncovered_ids)}")
print(f"  Cross-camera/no-filter:      {len(other_ids):>2} questions  {_cov_acc(other_ids)}")

# Save
suffix = "_camfilter" if CAMERA_FILTER else ("_compare" if COMPARE_BOTH else "")
out = ROOT / "results" / f"mcq50_extended{suffix}_result.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n  Saved → {out}")
if COMPARE_BOTH:
    out_nf = ROOT / "results" / "mcq50_extended_nofilter_result.json"
    with open(out_nf, "w", encoding="utf-8") as f:
        json.dump(result_no_filter, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {out_nf}")
