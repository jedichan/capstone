"""端到端评测驱动脚本：通过 mevid_fastapi HTTP 接口完成视频导入 → agent 评估。

Usage:
    # Terminal 1: start server
    cd mevid_fastapi && uvicorn main:app --port 8765

    # Terminal 2: run eval
    python mevid_fastapi/run_mevid_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8765"
TIMEOUT = 120.0  # seconds per request

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS_DIR = _PROJECT_ROOT / "agent" / "test" / "mevid_test" / "events_vector_flat"
NAMESPACE = "mevid_eval"
OUTPUT_DIR = _PROJECT_ROOT / "mevid_fastapi" / "data" / "runtime" / NAMESPACE


def _post(path: str, payload: dict) -> dict:
    url = f"{BASE_URL}{path}"
    resp = httpx.post(url, json=payload, timeout=TIMEOUT)
    if resp.status_code >= 400:
        print(f"[ERROR] {path} → {resp.status_code}: {resp.text[:400]}")
        sys.exit(1)
    return resp.json()


def _get(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    resp = httpx.get(url, timeout=30.0)
    if resp.status_code >= 400:
        print(f"[ERROR] {path} → {resp.status_code}: {resp.text[:400]}")
        sys.exit(1)
    return resp.json()


def main() -> None:
    t_total = time.perf_counter()

    # ── Step 0: Health check ──────────────────────────────────────────────────
    print("[1/3] Health check ...")
    health = _get("/healthz")
    print(f"      status={health['status']}")

    # ── Step 1: Ingest seeds → build DB ──────────────────────────────────────
    print(f"\n[2/3] Ingest seeds from {SEEDS_DIR} ...")
    ingest_result = _post(
        "/api/v1/ingest/seeds",
        {
            "seeds_dir": str(SEEDS_DIR),
            "namespace": NAMESPACE,
            "reset_existing": True,
        },
    )
    print(f"      seed_files={ingest_result['seed_file_count']}")
    print(f"      sqlite_rows={ingest_result['sqlite_rows']}")
    print(f"      chroma_child={ingest_result['chroma_child_records']}")
    print(f"      chroma_ge={ingest_result['chroma_ge_records']}")
    print(f"      elapsed={ingest_result['elapsed_ms']}ms")
    print(f"      sqlite → {ingest_result['sqlite_path']}")
    print(f"      chroma → {ingest_result['chroma_path']}")

    # ── Step 2: Run eval suite ────────────────────────────────────────────────
    print("\n[3/3] Running eval suite (10 questions) ...")
    eval_result = _post("/api/v1/eval/run", {"questions_file": None})

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = round((time.perf_counter() - t_total) * 1000 / 1000, 1)
    print("\n" + "=" * 60)
    print("EVAL RESULTS (via mevid_fastapi)")
    print("=" * 60)
    n_yes = len(eval_result.get("individual_iou", []))
    print(f"  Answer Accuracy : {eval_result['correct']}/{eval_result['total']} = {eval_result['accuracy']*100:.1f}%")
    print(f"  Top-Hit Rate    : {eval_result['top_hit_total']}/{eval_result['total']} = {eval_result['top_hit_rate']*100:.1f}%")
    print(f"  Mean IoU        : {eval_result['mean_iou']:.3f}  (yes={n_yes} questions, threshold={eval_result.get('iou_threshold', 0.15)})")
    print(f"  IoU >= 0.15     : {eval_result.get('n_iou_pass', 0)}/{eval_result['total']} = {eval_result.get('iou_pass_rate', 0)*100:.1f}%")
    print(f"  Individual IoU  : {[round(v,3) for v in eval_result.get('individual_iou', [])]}")
    print()
    print("  Per category:")
    for cat, s in eval_result["by_category"].items():
        print(f"    {cat:15s}: answer={s['correct']}/{s['total']} ({s['accuracy']*100:.0f}%)  top-hit={s['top_hit']}/{s['total']}")
    print()
    print("  Case details:")
    for c in eval_result["cases"]:
        tick = "✅" if c["correct"] else "❌"
        th = "✅" if c["top_hit"] else "❌"
        iou_str = f"IoU={c['iou']:.3f}" if c["expected"] == "yes" else "     —    "
        print(f"    {c['case_id']} [{c['category']:12s}] {tick}  top-hit={th}  {iou_str}  {c['question'][:55]}")
    print()
    print(f"  Total wall time : {total_elapsed}s")
    print("=" * 60)

    # Save result JSON
    out_path = OUTPUT_DIR / "fastapi_eval_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Result saved → {out_path}")


if __name__ == "__main__":
    main()
