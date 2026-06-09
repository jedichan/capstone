#!/usr/bin/env python3
"""Chunked RAGAS eval runner for Part4 — batches by video.

Usage (from agent/test/)::

    python run_chunks.py                          # full run
    python run_chunks.py --max-chunks 1           # smoke test
    python run_chunks.py --chunk-size 5            # smaller batches
    python run_chunks.py --no-progress             # redirect-friendly

Environment variables honoured::

    CHUNK_SIZE        default chunk size (overridden by --chunk-size)
    AGENT_BUILD_VIDEO_COLLECTION  set to 1 per chunk automatically

Optional retrieval backend (no LLM when ``id_based``): chunk-level same-video precision + RAGAS ID recall::

    python run_chunks.py --retrieval-metrics-backend id_based ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
AGENT_TEST_DIR = ROOT_DIR / "agent" / "test"
DEFAULT_XLSX = AGENT_TEST_DIR / "data" / "agent_test.xlsx"
DEFAULT_SEED_DIR = (
    AGENT_TEST_DIR / "generated" / "datasets" / "ucfcrime_events_vector_flat"
)
DEFAULT_OUTPUT_ROOT = AGENT_TEST_DIR / "generated" / "ragas_eval_p4_chunks"
RAGAS_RUNNER = AGENT_TEST_DIR / "ragas_eval_runner.py"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "10"))

GARBAGE_VIDEO_PREFIXES: tuple[str, ...] = (
    "!?",             # garbled cell data: !?4h?!
    "RoadAccidents",  # UCFCrime leakage
    "多摄像头",         # Chinese garbled text
)


def _tqdm(iterable=None, /, **kwargs):
    """Thin wrapper so tqdm is optional and --no-progress is respected."""
    try:
        from tqdm import tqdm as _tqdm_impl

        return _tqdm_impl(iterable, **kwargs) if iterable is not None else _tqdm_impl(**kwargs)
    except ImportError:
        if iterable is not None:
            return iterable

        class _Fake:
            def update(self, n=1):
                pass

            def set_description(self, desc):
                pass

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return _Fake()


# ---------------------------------------------------------------------------
# Case extraction
# ---------------------------------------------------------------------------


def _extract_cases(xlsx_path: Path) -> list[dict[str, Any]]:
    """Read Part4 cases from xlsx via the existing agent_test_importer."""
    import importlib

    sys.path.insert(0, str(AGENT_TEST_DIR))
    importer_mod = importlib.import_module("agent_test_importer")
    AgentTestImportConfig = importer_mod.AgentTestImportConfig
    AgentTestDatasetImporter = importer_mod.AgentTestDatasetImporter

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        config = AgentTestImportConfig(
            xlsx_path=xlsx_path,
            output_dir=tmpdir_path,
            sqlite_path=tmpdir_path / "agent_test_eval.sqlite",
            normalized_json_path=tmpdir_path / "agent_test_normalized.json",
            report_json_path=tmpdir_path / "agent_test_import_report.json",
            retrieval_view_path=tmpdir_path / "agent_test_retrieval_eval.json",
            e2e_view_path=tmpdir_path / "agent_test_e2e_eval.json",
            generation_view_path=tmpdir_path / "agent_test_generation_eval.json",
            reset_existing=True,
            include_sheets=["Part4"],
        )
        report = AgentTestDatasetImporter(config).build()
        cases = json.loads(config.normalized_json_path.read_text(encoding="utf-8"))
        filtered = [case for case in cases if case.get("e2e_ready") == 1]
        return filtered


# ---------------------------------------------------------------------------
# Video grouping & filtering
# ---------------------------------------------------------------------------


def _is_garbage_video(video_id: str) -> bool:
    vid = video_id.strip()
    if not vid:
        return True
    if not vid.startswith("Normal_Videos"):
        return True
    for prefix in GARBAGE_VIDEO_PREFIXES:
        if vid.startswith(prefix):
            return True
    return False


def _group_cases_by_video(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group cases by video_id, filtering out garbage video_ids.

    Returns an ordered dict (insertion order) so chunking is deterministic.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        vid = str(case.get("video_id", "")).strip()
        if _is_garbage_video(vid):
            continue
        grouped.setdefault(vid, []).append(case)
    return grouped


# ---------------------------------------------------------------------------
# Seed resolution (bidirectional mapping)
# ---------------------------------------------------------------------------


def resolve_seed_file(seed_dir: Path, video_id: str) -> Path | None:
    """Find the *_events_vector_flat.json seed file for a video_id.

    Handles bidirectional naming mismatch:
    - xlsx ``Normal_Videos_924_x264`` → disk ``Normal_Videos924_x264``
    - xlsx ``Normal_Videos594_x264``  → disk ``Normal_Videos594_x264``  (exact)
    """
    vid = video_id.strip()
    # 1. Exact match
    seed_file = seed_dir / f"{vid}_events_vector_flat.json"
    if seed_file.is_file():
        return seed_file

    # 2. Normal_Videos_X → Normal_VideosX  (strip underscore before number)
    if vid.startswith("Normal_Videos_"):
        alt = vid.replace("Normal_Videos_", "Normal_Videos", 1)
        alt_file = seed_dir / f"{alt}_events_vector_flat.json"
        if alt_file.is_file():
            return alt_file

    # 3. Normal_VideosX → Normal_Videos_X  (add underscore before number)
    if vid.startswith("Normal_Videos") and not vid.startswith("Normal_Videos_"):
        suffix = vid[len("Normal_Videos"):]
        if suffix and suffix[0].isdigit():
            alt = f"Normal_Videos_{suffix}"
            alt_file = seed_dir / f"{alt}_events_vector_flat.json"
            if alt_file.is_file():
                return alt_file

    return None


def prepare_chunk_seeds(
    video_ids: list[str],
    seed_dir: Path,
    chunk_seeds_dir: Path,
) -> tuple[list[Path], list[str]]:
    """Copy resolved seed files into chunk_seeds_dir.

    Returns (resolved_paths, missing_video_ids).
    """
    chunk_seeds_dir.mkdir(parents=True, exist_ok=True)
    resolved: list[Path] = []
    missing: list[str] = []

    for vid in video_ids:
        src = resolve_seed_file(seed_dir, vid)
        if src is None:
            missing.append(vid)
            continue
        dst = chunk_seeds_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        resolved.append(dst)

    return resolved, missing


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def build_chunks(
    grouped: dict[str, list[dict[str, Any]]],
    chunk_size: int,
) -> list[tuple[list[str], list[dict[str, Any]]]]:
    """Split grouped videos into chunks of chunk_size videos each."""
    video_ids = list(grouped.keys())
    chunks: list[tuple[list[str], list[dict[str, Any]]]] = []
    for i in range(0, len(video_ids), chunk_size):
        batch_vids = video_ids[i : i + chunk_size]
        batch_cases: list[dict[str, Any]] = []
        for vid in batch_vids:
            batch_cases.extend(grouped[vid])
        chunks.append((batch_vids, batch_cases))
    return chunks


# ---------------------------------------------------------------------------
# ragas_eval_runner invocation
# ---------------------------------------------------------------------------


def _write_case_ids_file(case_ids: list[str], path: Path) -> None:
    path.write_text("\n".join(case_ids) + "\n", encoding="utf-8")


def _extract_summary(output_dir: Path) -> dict[str, Any]:
    """Extract the summary dict from summary_report.json."""
    summary_path = output_dir / "summary_report.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_chunk(
    chunk_index: int,
    video_ids: list[str],
    cases: list[dict[str, Any]],
    *,
    seed_dir: Path,
    output_root: Path,
    xlsx_path: Path,
    no_progress: bool = False,
    disable_sql: bool = False,
    ragas_contexts_filter: str = "none",
    retrieval_metrics_backend: str = "llm",
) -> dict[str, Any]:
    """Execute one chunk: prepare seeds, run ragas_eval_runner, collect metrics."""
    chunk_label = f"chunk{chunk_index:02d}"
    chunk_output = output_root / chunk_label
    chunk_seeds = output_root / f"{chunk_label}_seeds"
    chunk_log = output_root / f"{chunk_label}.log"
    case_ids_file = output_root / f"{chunk_label}_case_ids.txt"

    # --- Prepare seeds ---
    resolved_seeds, missing = prepare_chunk_seeds(video_ids, seed_dir, chunk_seeds)
    if missing:
        print(
            f"[{chunk_label}] WARNING: {len(missing)} video(s) have no seed file "
            f"and will be skipped: {missing[:5]}",
            file=sys.stderr,
        )

    if not resolved_seeds:
        print(f"[{chunk_label}] ERROR: no seed files resolved, skipping chunk", file=sys.stderr)
        return {
            "chunk": chunk_index,
            "videos": len(video_ids),
            "cases": len(cases),
            "error": "no_seed_files",
        }

    # --- Write case IDs ---
    case_ids = [str(c["case_id"]) for c in cases]
    _write_case_ids_file(case_ids, case_ids_file)

    # --- Build ragas_eval_runner command ---
    cmd = [
        sys.executable,
        "-u",
        str(RAGAS_RUNNER),
        "--xlsx-path",
        str(xlsx_path),
        "--output-dir",
        str(chunk_output),
        "--seed-dir",
        str(chunk_seeds),
        "--case-ids-file",
        str(case_ids_file),
        "--prepare-subset-db",
        "--include-sheets",
        "Part4",
    ]
    if ragas_contexts_filter and ragas_contexts_filter != "none":
        cmd.extend(["--ragas-contexts-filter", ragas_contexts_filter])
    if retrieval_metrics_backend and retrieval_metrics_backend != "llm":
        cmd.extend(["--retrieval-metrics-backend", retrieval_metrics_backend])
    env = os.environ.copy()
    env["AGENT_BUILD_VIDEO_COLLECTION"] = "1"
    if disable_sql:
        env["AGENT_DISABLE_SQL_BRANCH"] = "1"

    print(
        f"[{chunk_label}] Running {len(video_ids)} videos, {len(cases)} cases",
        flush=True,
    )
    print(f"[{chunk_label}] Seeds in: {chunk_seeds}", flush=True)
    print(f"[{chunk_label}] Output in: {chunk_output}", flush=True)
    print(
        f"[{chunk_label}] Subprocess log (stdout+stderr): {chunk_log}",
        file=sys.stderr,
        flush=True,
    )

    # --- Run ---
    start = time.monotonic()
    env["PYTHONUNBUFFERED"] = "1"
    with open(chunk_log, "w", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(AGENT_TEST_DIR),
        )
        proc.wait()
    elapsed = time.monotonic() - start

    if proc.returncode != 0:
        print(
            f"[{chunk_label}] FAILED (exit code {proc.returncode}, "
            f"elapsed {elapsed:.0f}s). See {chunk_log}",
            file=sys.stderr,
        )
        return {
            "chunk": chunk_index,
            "videos": len(video_ids),
            "cases": len(cases),
            "error": f"exit_code_{proc.returncode}",
            "elapsed_s": round(elapsed, 1),
        }

    # --- Extract metrics ---
    summary = _extract_summary(chunk_output)
    retrieval = summary.get("retrieval_summary", {})
    generation = summary.get("generation_summary", {})
    temporal = summary.get("temporal_summary", {})
    localization = summary.get("localization_summary", {})
    e2e = summary.get("end_to_end_summary", {})

    def _pct(val: Any) -> float:
        try:
            return round(float(val), 4)
        except (TypeError, ValueError):
            return 0.0

    return {
        "chunk": chunk_index,
        "videos": len(video_ids),
        "cases": len(cases),
        "top_hit": _pct(summary.get("top_hit_rate")),
        "precision": _pct(retrieval.get("context_precision_avg")),
        "recall": _pct(retrieval.get("context_recall_avg")),
        "factual_corr": _pct(generation.get("factual_correctness_avg")),
        "custom_corr": _pct(generation.get("custom_correctness_avg")),
        "e2e": _pct(e2e.get("ragas_e2e_score_avg")),
        "iou_avg": _pct(temporal.get("time_range_overlap_iou_avg")),
        "vid_match_avg": _pct(localization.get("video_match_score_avg")),
        "latency_s": round(summary.get("avg_latency_ms", 0) / 1000, 1),
        "elapsed_s": round(elapsed, 1),
        "error_count": summary.get("error_summary", {}).get("graph_error_cases", 0),
    }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "chunk",
    "videos",
    "cases",
    "top_hit",
    "precision",
    "recall",
    "factual_corr",
    "custom_corr",
    "e2e",
    "iou_avg",
    "vid_match_avg",
    "latency_s",
]


def write_results_csv(rows: list[dict[str, Any]], output_root: Path) -> None:
    csv_path = output_root / "results.csv"
    csv_path.write_text(
        ",".join(_CSV_COLUMNS) + "\n",
        encoding="utf-8",
    )
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        for row in rows:
            writer.writerow(row)
    print(f"\nResults written to {csv_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunked RAGAS eval for Part4 — batches by video",
    )
    parser.add_argument(
        "--xlsx-path",
        type=Path,
        default=DEFAULT_XLSX,
        help="Path to agent_test.xlsx",
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=DEFAULT_SEED_DIR,
        help="Directory with *_events_vector_flat.json seed files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory for chunk results",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Videos per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Only run the first N chunks (0 = all)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress tqdm progress bars",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chunk plan without executing",
    )
    parser.add_argument(
        "--disable-sql",
        action="store_true",
        help="Set AGENT_DISABLE_SQL_BRANCH=1 (hybrid-only ablation)",
    )
    parser.add_argument(
        "--ragas-contexts-filter",
        type=str,
        default="none",
        choices=["none", "same_expected_video"],
        help="Forwarded to ragas_eval_runner.py --ragas-contexts-filter (default: none)",
    )
    parser.add_argument(
        "--retrieval-metrics-backend",
        type=str,
        default="llm",
        choices=["llm", "id_based"],
        help="Forwarded to ragas_eval_runner.py --retrieval-metrics-backend (default: llm)",
    )
    args = parser.parse_args()

    # --- Validate inputs ---
    if not args.xlsx_path.exists():
        print(f"ERROR: xlsx not found: {args.xlsx_path}", file=sys.stderr)
        sys.exit(1)
    if not args.seed_dir.is_dir():
        print(f"ERROR: seed dir not found: {args.seed_dir}", file=sys.stderr)
        sys.exit(1)

    # --- Extract cases ---
    print("Reading Part4 cases from xlsx ...", flush=True)
    cases = _extract_cases(args.xlsx_path)
    print(f"  Total e2e-ready cases: {len(cases)}", flush=True)

    # --- Group by video ---
    grouped = _group_cases_by_video(cases)
    print(f"  Valid videos: {len(grouped)}", flush=True)
    video_ids = list(grouped.keys())

    # --- Build chunks ---
    chunks = build_chunks(grouped, args.chunk_size)
    print(f"  Chunks: {len(chunks)} (size={args.chunk_size} videos each)", flush=True)

    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
        print(f"  Limited to first {args.max_chunks} chunk(s)", flush=True)

    if args.dry_run:
        total_cases = 0
        for i, (vids, batch_cases) in enumerate(chunks, start=1):
            total_cases += len(batch_cases)
            print(f"  chunk{i:02d}: {len(vids)} videos, {len(batch_cases)} cases", flush=True)
        print(f"  Total: {total_cases} cases across {len(chunks)} chunks", flush=True)
        return

    # --- Prepare output root ---
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {args.output_root}", flush=True)

    # --- Run chunks ---
    results: list[dict[str, Any]] = []
    pbar = _tqdm(
        total=len(chunks),
        desc="Chunks",
        disable=args.no_progress,
        unit="chunk",
    )

    for i, (vids, batch_cases) in enumerate(chunks, start=1):
        pbar.set_description(f"Chunk {i:02d}/{len(chunks):02d}")
        result = run_chunk(
            chunk_index=i,
            video_ids=vids,
            cases=batch_cases,
            seed_dir=args.seed_dir,
            output_root=args.output_root,
            xlsx_path=args.xlsx_path,
            no_progress=args.no_progress,
            disable_sql=args.disable_sql,
            ragas_contexts_filter=args.ragas_contexts_filter,
            retrieval_metrics_backend=args.retrieval_metrics_backend,
        )
        results.append(result)
        pbar.update(1)
        # Write CSV incrementally so partial results survive crashes
        write_results_csv(results, args.output_root)

    pbar.close()

    # --- Final summary ---
    passed = sum(1 for r in results if "error" not in r)
    failed = sum(1 for r in results if "error" in r)
    print(f"\nDone. {passed}/{len(results)} chunks passed, {failed} failed.", flush=True)
    write_results_csv(results, args.output_root)


if __name__ == "__main__":
    main()
