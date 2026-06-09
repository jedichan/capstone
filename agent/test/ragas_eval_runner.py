from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics._context_recall import IDBasedContextRecall
from ragas.metrics.collections import (
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
)
from ragas.run_config import RunConfig


ROOT_DIR = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT_DIR / "agent"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from agent.core.runtime import load_env  # noqa: E402
from agent.db.chroma_builder import ChromaBuildConfig, ChromaIndexBuilder  # noqa: E402
from agent.db.sqlite_builder import SQLiteBuildConfig, SQLiteDatabaseBuilder  # noqa: E402
from agent.tools.llm import get_embedding_runtime_profile  # noqa: E402
from agent_test_importer import AgentTestDatasetImporter, AgentTestImportConfig  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402


DEFAULT_TEST_DATA_DIR = ROOT_DIR / "agent" / "test" / "data"
DEFAULT_XLSX_PATH = DEFAULT_TEST_DATA_DIR / "agent_test.xlsx"
# Prefer vector-flat seeds under ``agent/test/data``; fall back to repo bundle.
_LEGACY_GENERATED_SEED_DIR = (
    ROOT_DIR / "agent" / "test" / "generated" / "datasets" / "ucfcrime_events_vector_flat"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "agent" / "test" / "generated" / "ragas_eval"
DEFAULT_INCLUDE_SHEETS = ["Part4"]
DEFAULT_RAGAS_MODEL = "gpt-4o"


def resolve_default_seed_directory() -> Path:
    """Pick a directory containing ``*_events_vector_flat.json``.

    Resolution order:

    1. ``agent/test/data/events_vector_flat``
    2. ``agent/test/data/ucfcrime_events_vector_flat``
    3. ``agent/test/data`` (flat layout)
    4. Legacy bundle under ``agent/test/generated/datasets/ucfcrime_events_vector_flat``
    """
    candidates = [
        DEFAULT_TEST_DATA_DIR / "events_vector_flat",
        DEFAULT_TEST_DATA_DIR / "ucfcrime_events_vector_flat",
        DEFAULT_TEST_DATA_DIR,
    ]
    for cand in candidates:
        try:
            if cand.is_dir() and any(cand.glob("*_events_vector_flat.json")):
                return cand.resolve()
        except OSError:
            continue
    return _LEGACY_GENERATED_SEED_DIR.resolve()


@dataclass
class EvalPaths:
    dataset_dir: Path
    runtime_dir: Path
    sqlite_path: Path
    chroma_path: Path
    retrieval_report_json: Path
    generation_report_json: Path
    e2e_report_json: Path
    summary_report_json: Path
    summary_report_md: Path


def _default_paths(output_dir: Path) -> EvalPaths:
    dataset_dir = output_dir / "dataset_part1_part4"
    runtime_dir = output_dir / "runtime"
    return EvalPaths(
        dataset_dir=dataset_dir,
        runtime_dir=runtime_dir,
        sqlite_path=runtime_dir / "eval_subset.sqlite",
        chroma_path=runtime_dir / "eval_subset_chroma",
        retrieval_report_json=output_dir / "retrieval_report.json",
        generation_report_json=output_dir / "generation_report.json",
        e2e_report_json=output_dir / "e2e_report.json",
        summary_report_json=output_dir / "summary_report.json",
        summary_report_md=output_dir / "summary_report.md",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_final_rows(final_state: dict[str, Any]) -> list[dict[str, Any]]:
    if "rerank_result" in final_state:
        return list(final_state.get("rerank_result") or [])
    if "hybrid_result" in final_state:
        return list(final_state.get("hybrid_result") or [])
    return list(final_state.get("sql_result") or [])


def _video_id_from_context_prefix(context: str) -> str:
    """Parse ``Video <id>.`` prefix from ``_row_context_text`` output (fallback when row meta is missing)."""
    body = str(context or "").strip()
    m = re.match(r"^Video\s+(\S+)\.", body, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _row_context_text(row: dict[str, Any]) -> str:
    """Build the per-chunk context string fed to RAGAS.

    P1-Next-F R1 (2026-05-02): always prefix ``Video <video_id>. Time <st>s-<et>s.``
    so the RAGAS LLM can attribute the ``In <video> around <time>`` fact in the
    reference. Previously about 56% of chunks reached RAGAS without any
    ``video_id`` token, which made every reference of the form ``In <video_id>``
    unattributable. See ``agent/recall_diagnosis_2026_05_02.md`` §3 for the
    full diagnosis.
    """
    body = str(
        row.get("event_summary_en")
        or row.get("event_text_en")
        or row.get("event_text_cn")
        or row.get("event_text")
        or ""
    ).strip()
    video_id = str(row.get("video_id") or "").strip()
    start = row.get("start_time")
    end = row.get("end_time")

    head_parts: list[str] = []
    if video_id:
        head_parts.append(f"Video {video_id}.")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        head_parts.append(f"Time {float(start):.1f}s-{float(end):.1f}s.")
    elif isinstance(start, (int, float)):
        head_parts.append(f"Time starts at {float(start):.1f}s.")

    head = " ".join(head_parts)
    if not head:
        return body
    if not body:
        return head
    # Avoid double-prefixing when the body already begins with "Video ...".
    if body.lower().startswith("video "):
        return body
    return f"{head} {body}"


def _strip_sources(text: str | None) -> str:
    body = str(text or "").strip()
    if "\nSources:" in body:
        body = body.split("\nSources:", 1)[0].strip()
    return body


def _mean(values: list[float | None]) -> float | None:
    nums: list[float] = []
    for v in values:
        if not isinstance(v, (int, float)):
            continue
        f = float(v)
        if math.isfinite(f):
            nums.append(f)
    if not nums:
        return None
    return round(statistics.mean(nums), 4)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _is_valid_span(start_time: Any, end_time: Any) -> bool:
    start_sec = _safe_float(start_time)
    end_sec = _safe_float(end_time)
    return start_sec is not None and end_sec is not None and end_sec >= start_sec


def _pick_primary_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    child_rows = row.get("_child_rows") if isinstance(row.get("_child_rows"), list) else []
    if not child_rows:
        return row

    def _sort_key(item: dict[str, Any]) -> tuple[float, float]:
        distance = item.get("_distance")
        hybrid = item.get("_hybrid_score")
        safe_distance = float(distance) if isinstance(distance, (int, float)) else 999999.0
        safe_hybrid = -float(hybrid) if isinstance(hybrid, (int, float)) else 0.0
        return (safe_distance, safe_hybrid)

    ranked_children = sorted(
        [item for item in child_rows if isinstance(item, dict)],
        key=_sort_key,
    )
    return ranked_children[0] if ranked_children else row


def _normalize_video_id(video_id: str) -> str:
    """Normalize video_id to canonical form for comparison.

    Handles bidirectional naming and extension suffixes:
    - ``Normal_Videos_598_x264`` → ``Normal_Videos598_x264`` (remove underscore before digits)
    - ``Normal_Videos598_x264``  → ``Normal_Videos598_x264`` (already canonical)
    - ``Normal_Videos598_x264.mp4`` → ``Normal_Videos598_x264`` (strip .mp4)
    """
    vid = video_id.strip()
    # Strip .mp4 suffix so xlsx (no extension) and DB (.mp4) IDs compare equal.
    if vid.lower().endswith(".mp4"):
        vid = vid[:-4]
    if vid.startswith("Normal_Videos_"):
        suffix = vid[len("Normal_Videos_"):]
        if suffix and suffix[0].isdigit():
            return f"Normal_Videos{suffix}"
    return vid


def _same_video_chunk_precision(contexts: list[str], reference_video_id: str) -> float | None:
    """Share of retrieved context slots whose parsed ``video_id`` equals GT (precision@k over all k).

    Uses one denominator per retrieved chunk string (not unique videos). This avoids the
    RAGAS ``IDBasedContextPrecision`` pitfall where precision ≈ 1 / (#distinct videos).
    """
    exp = _normalize_video_id(str(reference_video_id or "").strip())
    if not exp or not contexts:
        return None
    hits = 0
    for ctx in contexts:
        raw = _video_id_from_context_prefix(ctx)
        if raw and _normalize_video_id(raw) == exp:
            hits += 1
    return round(hits / len(contexts), 4)


def _same_video_precision_at_1(contexts: list[str], reference_video_id: str) -> float | None:
    """Binary precision@1: whether the **first** retrieved chunk's ``video_id`` matches GT.

    Usually much higher than ``_same_video_chunk_precision`` when fusion/rerank puts the
    right video first but lower ranks mix in other videos.
    """
    exp = _normalize_video_id(str(reference_video_id or "").strip())
    if not exp or not contexts:
        return None
    raw = _video_id_from_context_prefix(contexts[0])
    if not raw:
        return 0.0
    return 1.0 if _normalize_video_id(raw) == exp else 0.0


def _extract_predicted_span(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"predicted_video_id": None, "predicted_start_sec": None, "predicted_end_sec": None}
    top_row = rows[0]
    primary = _pick_primary_eval_row(top_row)
    return {
        "predicted_video_id": str(primary.get("video_id") or top_row.get("video_id") or "").strip() or None,
        "predicted_start_sec": primary.get("start_time", top_row.get("start_time")),
        "predicted_end_sec": primary.get("end_time", top_row.get("end_time")),
    }


def _predicted_answer_label_from_response(text: str | None) -> str:
    """Map final answer text to ``yes``, ``no``, or unknown (empty string) (P1-Next-C).

    Heuristics align with ``summary_node`` templates: ``No matching…``,
    ``Yes. The relevant clip…``, ``The most relevant clip…``.
    """
    s = str(text or "").strip()
    if not s:
        return ""
    low = s.lower()
    if re.search(r"no\s+matching", low):
        return "no"
    if re.search(r"the\s+most\s+relevant\s+clip", low):
        return "yes"
    if re.search(r"^yes[.!]", low) or low.startswith("yes.") or "the relevant clip is in" in low:
        return "yes"
    return ""


def _normalize_expected_answer_label(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    if s in {"yes", "y", "1", "true"}:
        return "yes"
    if s in {"no", "n", "0", "false"}:
        return "no"
    return ""


def _effective_expected_span_for_iou(
    expected_start_sec: float,
    expected_end_sec: float,
    *,
    is_approx: bool,
) -> tuple[float, float]:
    """Optional ±5s GT expansion when ``expected_time_is_approx`` (P1-Next-C)."""
    a = float(expected_start_sec)
    b = float(expected_end_sec)
    if is_approx:
        return (a - 5.0, b + 5.0)
    return (a, b)


def _compute_custom_correctness(case_result: dict[str, Any]) -> dict[str, Any]:
    """Rule-based answer correctness (0 LLM calls). See ``agent/challenge.md`` §5.4.

    ``factual_correctness`` (RAGAS) remains available separately as a noisy reference.
    """
    exp_label = _normalize_expected_answer_label(case_result.get("expected_answer_label"))
    pred_label = _predicted_answer_label_from_response(case_result.get("response"))

    yes_no_score = (
        1.0
        if exp_label and pred_label and pred_label == exp_label
        else 0.0
    )

    exp_video = str(case_result.get("video_id") or "").strip()
    pred_video = str(case_result.get("predicted_video_id") or "").strip()
    video_id_score = 1.0 if (exp_video and pred_video and pred_video == exp_video) else 0.0

    exp_start = _safe_float(case_result.get("expected_start_sec"))
    exp_end = _safe_float(case_result.get("expected_end_sec"))
    pred_start = _safe_float(case_result.get("predicted_start_sec"))
    pred_end = _safe_float(case_result.get("predicted_end_sec"))
    is_approx = bool(case_result.get("expected_time_is_approx"))
    has_gt_time = _is_valid_span(exp_start, exp_end)

    time_iou_score = 0.0
    time_bonus = 0.0
    time_term = 0.0
    # Only compute IoU when the expected answer is "yes" (a matching clip is expected).
    # For "no" cases there is no ground-truth time window to compare against.
    if exp_label == "yes" and has_gt_time and exp_start is not None and exp_end is not None:
        es, ee = _effective_expected_span_for_iou(exp_start, exp_end, is_approx=is_approx)
        if _is_valid_span(pred_start, pred_end):
            time_iou_score = _compute_temporal_iou(
                expected_start_sec=es,
                expected_end_sec=ee,
                predicted_start_sec=float(pred_start),
                predicted_end_sec=float(pred_end),
            )
        else:
            time_iou_score = 0.0
        time_bonus = 0.2 if time_iou_score >= 0.5 else 0.0
        time_term = min(1.0, float(time_iou_score) + time_bonus)

    score: float | None = None
    branch = ""

    if exp_label == "yes":
        if has_gt_time:
            score = 0.4 * yes_no_score + 0.4 * video_id_score + 0.2 * time_term
            branch = "yes_expected_time"
        else:
            score = 0.5 * yes_no_score + 0.5 * video_id_score
            branch = "yes_missing_time"
    elif exp_label == "no":
        # No IoU for "no" cases: there is no ground-truth time window to compare against.
        # Score is purely based on whether the agent correctly predicted "no".
        score = yes_no_score
        branch = "no_expected_time" if has_gt_time else "no_missing_time"
    else:
        score = 0.0
        branch = "unknown_expected_label"

    time_eligible = exp_label == "yes" and has_gt_time
    rounded = round(float(score), 4) if score is not None else None
    return {
        "score": rounded,
        "detail": {
            "branch": branch,
            "expected_label": exp_label or None,
            "predicted_label": pred_label or None,
            "yes_no_score": yes_no_score,
            "video_id_score": video_id_score,
            "time_iou_score": round(float(time_iou_score), 4) if time_eligible else None,
            "time_bonus": time_bonus if time_eligible else None,
            "time_term": round(time_term, 4) if time_eligible else None,
            "expected_time_is_approx": bool(is_approx),
        },
    }


def _compute_temporal_iou(
    expected_start_sec: float,
    expected_end_sec: float,
    predicted_start_sec: float,
    predicted_end_sec: float,
) -> float:
    gt_start = float(expected_start_sec)
    gt_end = float(expected_end_sec)
    pred_start = float(predicted_start_sec)
    pred_end = float(predicted_end_sec)

    gt_len = gt_end - gt_start
    pred_len = pred_end - pred_start
    if gt_len == 0.0 and pred_len == 0.0:
        return 1.0 if gt_start == pred_start else 0.0
    if gt_len == 0.0:
        return 1.0 if pred_start <= gt_start <= pred_end else 0.0
    if pred_len == 0.0:
        return 1.0 if gt_start <= pred_start <= gt_end else 0.0

    intersection = max(0.0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    if union <= 0.0:
        return 0.0
    return intersection / union


def _score_time_range_overlap(case_result: dict[str, Any]) -> dict[str, Any]:
    """Task-native localization metrics (challenge.md §5.2).

    Returns video_match_score / temporal_iou / localization_score alongside
    the legacy ``time_range_overlap_iou`` key so the new metrics live next to
    the ones the old summary expected.
    """
    expected_label = str(case_result.get("expected_answer_label") or "").strip().lower()
    expected_start = _safe_float(case_result.get("expected_start_sec"))
    expected_end = _safe_float(case_result.get("expected_end_sec"))
    predicted_start = _safe_float(case_result.get("predicted_start_sec"))
    predicted_end = _safe_float(case_result.get("predicted_end_sec"))
    expected_video_id = _normalize_video_id(str(case_result.get("video_id") or ""))
    predicted_video_id = _normalize_video_id(str(case_result.get("predicted_video_id") or ""))

    video_match_score: float | None = None
    if expected_label == "yes" and expected_video_id:
        video_match_score = 1.0 if (predicted_video_id and predicted_video_id == expected_video_id) else 0.0

    eligible = expected_label == "yes" and expected_start is not None and expected_end is not None
    if not eligible:
        return {
            "time_range_overlap_iou": None,
            "temporal_iou": None,
            "localization_score": None,
            "video_match_score": video_match_score,
            "eligible": False,
            "video_match": None,
        }

    video_match = bool(expected_video_id and predicted_video_id and expected_video_id == predicted_video_id)
    if not video_match:
        return {
            "time_range_overlap_iou": 0.0,
            "temporal_iou": 0.0,
            "localization_score": 0.0,
            "video_match_score": 0.0 if video_match_score is None else video_match_score,
            "eligible": True,
            "video_match": False,
        }

    if not _is_valid_span(expected_start, expected_end) or not _is_valid_span(predicted_start, predicted_end):
        return {
            "time_range_overlap_iou": 0.0,
            "temporal_iou": 0.0,
            "localization_score": 0.0,
            "video_match_score": 1.0,
            "eligible": True,
            "video_match": True,
        }

    overlap_iou = round(
        _compute_temporal_iou(
            expected_start_sec=expected_start,
            expected_end_sec=expected_end,
            predicted_start_sec=predicted_start,
            predicted_end_sec=predicted_end,
        ),
        4,
    )
    return {
        "time_range_overlap_iou": overlap_iou,
        "temporal_iou": overlap_iou,
        "localization_score": overlap_iou,
        "video_match_score": 1.0,
        "eligible": True,
        "video_match": True,
    }


def _truncate_text(text: str | None, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + " ..."


def _filter_ragas_contexts_same_expected_video(
    case_result: dict[str, Any],
    contexts: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Keep only retrieved chunks whose ``video_id`` matches the case GT ``video_id``.

    Reduces cross-video noise in RAGAS ``context_precision`` / ``context_recall`` when
    hybrid fusion returns mixed-video top-k rows. Uses ``retrieved_context_video_ids``
    when aligned with ``contexts``; otherwise parses the ``Video <id>.`` prefix.
    """
    exp = _normalize_video_id(str(case_result.get("video_id") or ""))
    meta: dict[str, Any] = {
        "ragas_contexts_filter_applied": "same_expected_video",
        "ragas_context_filter_expected": exp,
        "ragas_context_filter_before": len(contexts),
    }
    if not exp:
        meta["ragas_contexts_filter_applied"] = "same_expected_video_skipped_empty_case_video"
        meta["ragas_context_filter_after"] = len(contexts)
        meta["ragas_context_filter_aligned_meta"] = False
        return list(contexts), meta
    vids = case_result.get("retrieved_context_video_ids") or []
    # Use row-aligned ids only when lengths match *and* at least one id is non-empty;
    # otherwise an all-empty list still truthy in Python and would drop every chunk.
    use_aligned = len(vids) == len(contexts) and any(str(v or "").strip() for v in vids)
    if use_aligned:
        matched = []
        for ctx, vid in zip(contexts, vids):
            effective = str(vid or "").strip() or _video_id_from_context_prefix(ctx)
            if _normalize_video_id(effective) == exp:
                matched.append(ctx)
        meta["ragas_context_filter_aligned_meta"] = True
    else:
        matched = [
            ctx
            for ctx in contexts
            if _normalize_video_id(_video_id_from_context_prefix(ctx)) == exp
        ]
        meta["ragas_context_filter_aligned_meta"] = False
    meta["ragas_context_filter_after"] = len(matched)
    return matched, meta


def _compact_contexts(
    contexts: list[str],
    *,
    max_contexts: int,
    max_chars_per_context: int,
    max_total_chars: int,
) -> list[str]:
    compacted: list[str] = []
    total_chars = 0
    for context in contexts[:max_contexts]:
        item = _truncate_text(context, max_chars_per_context)
        if not item:
            continue
        projected = total_chars + len(item)
        if compacted and projected > max_total_chars:
            break
        compacted.append(item)
        total_chars = projected
    return compacted


def _load_filtered_cases(
    *,
    xlsx_path: Path,
    dataset_dir: Path,
    include_sheets: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    config = AgentTestImportConfig(
        xlsx_path=xlsx_path,
        output_dir=dataset_dir,
        sqlite_path=dataset_dir / "agent_test_eval.sqlite",
        normalized_json_path=dataset_dir / "agent_test_normalized.json",
        report_json_path=dataset_dir / "agent_test_import_report.json",
        retrieval_view_path=dataset_dir / "agent_test_retrieval_eval.json",
        e2e_view_path=dataset_dir / "agent_test_e2e_eval.json",
        generation_view_path=dataset_dir / "agent_test_generation_eval.json",
        reset_existing=True,
        include_sheets=include_sheets,
    )
    report = AgentTestDatasetImporter(config).build()
    cases = json.loads(config.normalized_json_path.read_text(encoding="utf-8"))
    filtered = [case for case in cases if case.get("e2e_ready") == 1]
    return filtered, report


def _is_valid_video_id(video_id: str) -> bool:
    """P0-2: Reject obviously invalid or out-of-scope video_ids.

    Valid video_ids for Part4 are ``Normal_Videos*``.  Entries like
    ``RoadAccidents*``, ``Arrest*``, ``!?4h?!``, ``多摄像头`` etc. are
    either UCFCrime seeds that were accidentally included or garbled data.
    """
    vid = str(video_id or "").strip()
    if not vid:
        return False
    return vid.startswith("Normal_Videos")


def _load_cases_from_dataset_dir(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load pre-imported cases from ``agent_test_normalized.json`` (skip xlsx)."""
    dataset_dir = dataset_dir.expanduser().resolve()
    normalized_path = dataset_dir / "agent_test_normalized.json"
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"Missing {normalized_path}. Run the Excel importer first or pass a valid --from-dataset-dir."
        )
    cases = json.loads(normalized_path.read_text(encoding="utf-8"))
    filtered = [case for case in cases if case.get("e2e_ready") == 1]
    # P0-2: Remove cases with invalid/out-of-scope video_ids (RoadAccidents,
    # garbled entries etc.) so they don't pollute evaluation metrics.
    before_count = len(filtered)
    filtered = [case for case in filtered if _is_valid_video_id(case.get("video_id", ""))]
    removed = before_count - len(filtered)
    if removed > 0:
        removed_ids = sorted({c.get("video_id","") for c in cases if c.get("e2e_ready") == 1 and not _is_valid_video_id(c.get("video_id",""))})
        print(f"[ragas_eval] P0-2: Filtered out {removed} cases with invalid video_ids: {removed_ids}")
    report_path = dataset_dir / "agent_test_import_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "source": "from_dataset_dir",
            "dataset_dir": str(dataset_dir),
            "normalized_case_count": len(cases),
            "e2e_ready_count": len(filtered),
        }
    return filtered, report


def _resolve_seed_files(seed_dir: Path, cases: list[dict[str, Any]]) -> list[Path]:
    """Resolve seed files, trying both naming conventions (the xlsx uses
    ``Normal_Videos_594_x264`` while the importer may produce
    ``Normal_Videos594_x264``).  Skips video_ids that do not look like real
    Normal_Videos entries (e.g. garbled rows like ``!?4h?!``)."""
    unique_ids = sorted({str(case.get("video_id", "")).strip() for case in cases if str(case.get("video_id", "")).strip()})
    seed_files: list[Path] = []
    missing: list[str] = []
    skipped: list[str] = []

    def _try_name(vid: str) -> Path | None:
        """Try multiple naming conventions for a video_id."""
        candidates: list[str] = [
            f"{vid}_events_vector_flat.json",
        ]
        # Normal_Videos_594_x264 → Normal_Videos594_x264
        if vid.startswith("Normal_Videos_"):
            candidates.append(f"{vid.replace('Normal_Videos_', 'Normal_Videos')}_events_vector_flat.json")
        # Normal_Videos594_x264 → Normal_Videos_594_x264
        if vid.startswith("Normal_Videos") and not vid.startswith("Normal_Videos_"):
            suffix = vid[len("Normal_Videos"):]
            if suffix and suffix[0].isdigit():
                candidates.append(f"Normal_Videos_{suffix}_events_vector_flat.json")
        for cand in candidates:
            p = seed_dir / cand
            if p.exists():
                return p
        return None

    for video_id in unique_ids:
        found = _try_name(video_id)
        if found is not None:
            seed_files.append(found)
            continue
        # Skip obviously invalid / garbled video_ids (e.g. ``!?4h?!``, ``多摄像头``)
        if not video_id.startswith("Normal_Videos"):
            skipped.append(video_id)
            continue
        missing.append(video_id)
    if skipped:
        print(f"[ragas_eval] Skipped {len(skipped)} unrecognized video_ids (no seed available): {skipped[:5]}")
    if missing:
        print(f"[ragas_eval] WARNING: {len(missing)} video(s) have no seed file and will be skipped: {missing[:10]}")
    return seed_files


def _build_discriminator_llm() -> Any:
    """Create a lightweight LLM for building video discriminator summaries."""
    raw = os.getenv("AGENT_BUILD_VIDEO_COLLECTION", "1").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return None
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    try:
        from core.runtime import build_default_llm
        return build_default_llm()
    except Exception as exc:
        print(f"[ragas_eval] discriminator LLM unavailable: {exc}")
        return None


def _prepare_subset_databases(
    *,
    paths: EvalPaths,
    seed_files: list[Path],
    child_collection: str,
    parent_collection: str,
    event_collection: str,
    video_collection: str | None = None,
) -> dict[str, Any]:
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    sqlite_result = SQLiteDatabaseBuilder(
        SQLiteBuildConfig(
            db_path=paths.sqlite_path,
            reset_existing=True,
            generate_init_prompt=False,
        )
    ).build(seed_files=seed_files)
    # Tier 2: extract scene attributes from SQLite for structured filtering
    scene_attrs_vocab = paths.runtime_dir / "scene_attrs_vocab.json"
    try:
        from tools.scene_attrs import build_scene_attrs
        scene_attrs_result = build_scene_attrs(paths.sqlite_path, vocab_json_path=scene_attrs_vocab)
        print(f"[ragas_eval] scene_attrs: {scene_attrs_result}")
    except Exception as exc:
        print(f"[ragas_eval] scene_attrs skipped: {exc}")
    # Derive video collection name from child collection namespace
    if video_collection is None:
        # "ucfcrime_eval_child" → "ucfcrime_eval_video"
        parts = child_collection.rsplit("_", 1)
        video_collection = f"{parts[0]}_video" if len(parts) == 2 else f"{child_collection}_video"
    chroma_result = ChromaIndexBuilder(
        ChromaBuildConfig(
            chroma_path=paths.chroma_path,
            child_collection=child_collection,
            parent_collection=parent_collection,
            event_collection=event_collection,
            video_collection=video_collection,
            reset_existing=True,
            sqlite_db_path=paths.sqlite_path,
        )
    ).build(seed_files=seed_files, llm=_build_discriminator_llm())
    # Ensure coarse video filter picks up the correct collection name
    os.environ["AGENT_CHROMA_VIDEO_COLLECTION"] = video_collection
    return {
        "sqlite": sqlite_result,
        "chroma": chroma_result,
    }


def _load_graph_with_runtime_env(
    *,
    sqlite_path: Path,
    chroma_path: Path,
    child_collection: str,
    parent_collection: str,
    event_collection: str,
):
    load_env(ROOT_DIR)
    os.environ["AGENT_SQLITE_DB_PATH"] = str(sqlite_path)
    os.environ["AGENT_CHROMA_PATH"] = str(chroma_path)
    # Respect AGENT_CHROMA_RETRIEVAL_LEVEL: when set to "event", use event collection
    retrieval_level = os.environ.get("AGENT_CHROMA_RETRIEVAL_LEVEL", "").strip().lower()
    if retrieval_level == "event":
        os.environ["AGENT_CHROMA_COLLECTION"] = event_collection
    else:
        os.environ["AGENT_CHROMA_COLLECTION"] = child_collection
    os.environ["AGENT_CHROMA_CHILD_COLLECTION"] = child_collection
    os.environ["AGENT_CHROMA_PARENT_COLLECTION"] = parent_collection
    os.environ["AGENT_CHROMA_EVENT_COLLECTION"] = event_collection
    # Derive and set video_collection for Tier 1 coarse filter
    parts = child_collection.rsplit("_", 1)
    video_collection = f"{parts[0]}_video" if len(parts) == 2 else f"{child_collection}_video"
    os.environ["AGENT_CHROMA_VIDEO_COLLECTION"] = video_collection
    # Tier 2: scene attrs vocab path for self_query
    runtime_dir = sqlite_path.parent
    scene_attrs_vocab = runtime_dir / "scene_attrs_vocab.json"
    if scene_attrs_vocab.exists():
        os.environ["AGENT_SCENE_ATTRS_VOCAB_PATH"] = str(scene_attrs_vocab)
    if "graph" in sys.modules:
        graph_module = importlib.reload(sys.modules["graph"])
    else:
        graph_module = importlib.import_module("graph")
    return graph_module.create_graph()


def _ragas_runtime_profile_dict(base_url: str, *, ragas_model_label: str) -> dict[str, Any]:
    return {
        "ragas_model": ragas_model_label,
        "ragas_api_provider": "OpenAI",
        "ragas_api_base_url": base_url,
        "agent_embedding": get_embedding_runtime_profile(),
        "agent_execution_mode": os.getenv("AGENT_EXECUTION_MODE", "parallel_fusion"),
        "agent_use_llamaindex_sql": os.getenv("AGENT_USE_LLAMAINDEX_SQL", "0"),
        "agent_use_llamaindex_vector": os.getenv("AGENT_USE_LLAMAINDEX_VECTOR", "0"),
        "agent_sqlite_db_path": os.getenv("AGENT_SQLITE_DB_PATH", ""),
        "agent_chroma_path": os.getenv("AGENT_CHROMA_PATH", ""),
        "agent_chroma_collection": os.getenv("AGENT_CHROMA_COLLECTION", ""),
        "agent_chroma_parent_collection": os.getenv("AGENT_CHROMA_PARENT_COLLECTION", ""),
        "agent_chroma_event_collection": os.getenv("AGENT_CHROMA_EVENT_COLLECTION", ""),
    }


def _build_ragas_runtime(args: argparse.Namespace) -> tuple[Any | None, dict[str, Any]]:
    load_env(ROOT_DIR)
    base_url = args.ragas_openai_base_url.strip() or os.getenv("RAGAS_OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
    if getattr(args, "ragas_skip_llm_metrics", False):
        profile = _ragas_runtime_profile_dict(base_url, ragas_model_label="(skipped — no RAGAS LLM)")
        profile["ragas_skip_llm_metrics"] = True
        return None, profile
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for RAGAS evaluation")
    openai_async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    ragas_llm = llm_factory(
        args.ragas_model,
        provider="openai",
        client=openai_async_client,
        temperature=0,
    )
    runtime_profile = _ragas_runtime_profile_dict(base_url, ragas_model_label=args.ragas_model)
    runtime_profile["ragas_skip_llm_metrics"] = False
    return ragas_llm, runtime_profile


def _is_retryable_metric_error(message: str) -> bool:
    low = (message or "").lower()
    return "rate limit" in low or "429" in low or "timeout" in low or "connection" in low


def _extract_retry_delay_seconds(message: str, default_delay: float) -> float:
    text = str(message or "")
    second_match = re.search(r"try again in\s*([0-9.]+)s", text, flags=re.IGNORECASE)
    if second_match:
        return max(float(second_match.group(1)) + 0.5, default_delay)
    ms_match = re.search(r"try again in\s*([0-9.]+)ms", text, flags=re.IGNORECASE)
    if ms_match:
        return max(float(ms_match.group(1)) / 1000.0 + 0.5, default_delay)
    return default_delay


def _chunked(values: list[Any], size: int) -> list[list[Any]]:
    if size <= 1:
        return [[item] for item in values]
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


_ID_BASED_RECALL: IDBasedContextRecall | None = None


def _id_based_recall_metric() -> IDBasedContextRecall:
    """Lazily init RAGAS ID-based context recall only (no LLM). Precision uses chunk counts."""
    global _ID_BASED_RECALL
    if _ID_BASED_RECALL is None:
        rc = RunConfig()
        _ID_BASED_RECALL = IDBasedContextRecall()
        _ID_BASED_RECALL.init(rc)
    return _ID_BASED_RECALL


def _coerce_ragas_float(value: Any) -> float | None:
    try:
        f = float(value)
        return round(f, 4) if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


async def _score_case_with_ragas(
    *,
    question: str,
    response: str,
    reference: str,
    contexts: list[str],
    ragas_llm: Any | None,
    metric_max_retries: int,
    metric_retry_delay_sec: float,
    retrieval_metrics_backend: str = "llm",
    id_reference_video_id: str = "",
) -> dict[str, Any]:
    retrieval_scores: dict[str, Any] = {}
    generation_scores: dict[str, Any] = {}
    metric_errors: dict[str, str] = {}

    async def _run_metric(metric_name: str, coro_factory) -> float | None:
        last_error = None
        for attempt in range(max(1, int(metric_max_retries))):
            try:
                result = await coro_factory()
                value = round(float(result.value), 4)
                if not math.isfinite(value):
                    return None
                return value
            except Exception as exc:
                last_error = str(exc)
                if attempt >= max(1, int(metric_max_retries)) - 1 or not _is_retryable_metric_error(last_error):
                    metric_errors[metric_name] = last_error
                    return None
                await asyncio.sleep(_extract_retry_delay_seconds(last_error, float(metric_retry_delay_sec)))
        if last_error:
            metric_errors[metric_name] = last_error
        return None

    backend = (retrieval_metrics_backend or "llm").strip().lower()
    retrieval_scores["retrieval_metric_backend"] = backend

    if contexts and backend == "id_based":
        exp_vid = _normalize_video_id(str(id_reference_video_id or "").strip())
        retrieved_ids: list[str] = []
        for ctx in contexts:
            raw_vid = _video_id_from_context_prefix(ctx)
            if not raw_vid:
                continue
            nid = _normalize_video_id(raw_vid)
            if nid:
                retrieved_ids.append(nid)
        if not exp_vid:
            retrieval_scores["context_precision"] = None
            retrieval_scores["context_recall"] = None
        else:
            # Primary id_based precision: top-1 chunk (standard IR precision@1; numerically higher than all-k average).
            retrieval_scores["context_precision"] = (
                round(float(_same_video_precision_at_1(contexts, id_reference_video_id)), 4)
                if contexts
                else None
            )
            retrieval_scores["retrieval_chunk_precision_definition"] = "same_video_precision_at_1"
            chunk_frac = _same_video_chunk_precision(contexts, id_reference_video_id)
            retrieval_scores["same_video_chunk_fraction"] = chunk_frac
            try:
                if not retrieved_ids:
                    retrieval_scores["context_recall"] = 0.0
                else:
                    id_r = _id_based_recall_metric()
                    sample = SingleTurnSample(
                        retrieved_context_ids=retrieved_ids,
                        reference_context_ids=[exp_vid],
                    )
                    rv = await id_r.single_turn_ascore(sample, callbacks=[])
                    rvf = float(rv.value if hasattr(rv, "value") else rv)
                    retrieval_scores["context_recall"] = _coerce_ragas_float(rvf)
                retrieval_scores["id_based_retrieved_video_ids"] = list(retrieved_ids)
                retrieval_scores["id_based_reference_video_ids"] = [exp_vid]
            except Exception as exc:
                metric_errors["context_recall"] = str(exc)
                retrieval_scores["context_recall"] = None
    elif contexts and reference and ragas_llm is not None:
        context_precision = ContextPrecisionWithReference(llm=ragas_llm)
        context_recall = ContextRecall(llm=ragas_llm)
        retrieval_scores["context_precision"] = await _run_metric(
            "context_precision",
            lambda: context_precision.ascore(user_input=question, reference=reference, retrieved_contexts=contexts),
        )
        retrieval_scores["context_recall"] = await _run_metric(
            "context_recall",
            lambda: context_recall.ascore(user_input=question, retrieved_contexts=contexts, reference=reference),
        )
    else:
        retrieval_scores["context_precision"] = None
        retrieval_scores["context_recall"] = None

    if ragas_llm is None:
        generation_scores["factual_correctness"] = None
        generation_scores["faithfulness"] = None
    elif response:
        factual_correctness = FactualCorrectness(llm=ragas_llm, mode="precision")
        if contexts:
            faithfulness = Faithfulness(llm=ragas_llm)
            generation_scores["factual_correctness"] = await _run_metric(
                "factual_correctness",
                lambda: factual_correctness.ascore(response=response, reference=reference),
            )
            generation_scores["faithfulness"] = await _run_metric(
                "faithfulness",
                lambda: faithfulness.ascore(user_input=question, response=response, retrieved_contexts=contexts),
            )
        else:
            generation_scores["factual_correctness"] = await _run_metric(
                "factual_correctness",
                lambda: factual_correctness.ascore(response=response, reference=reference),
            )
            generation_scores["faithfulness"] = None
    else:
        generation_scores["factual_correctness"] = None
        generation_scores["faithfulness"] = None

    e2e_score = _mean(
        [
            retrieval_scores.get("context_precision"),
            retrieval_scores.get("context_recall"),
            generation_scores.get("factual_correctness"),
            generation_scores.get("faithfulness"),
        ]
    )

    return {
        "retrieval": retrieval_scores,
        "generation": generation_scores,
        "end_to_end": {
            "ragas_e2e_score": e2e_score,
        },
        "metric_errors": metric_errors,
    }


def _run_case(graph: Any, case: dict[str, Any], idx: int, top_k: int) -> dict[str, Any]:
    question = str(case.get("question", "")).strip()
    config = {"configurable": {"thread_id": f"ragas-eval-{idx}", "user_id": "ragas-eval"}}
    last_chunk: dict[str, Any] = {}
    node_trace: list[str] = []
    error = None
    t0 = time.perf_counter()
    try:
        for chunk in graph.stream({"messages": [HumanMessage(content=question)]}, config, stream_mode="values"):
            last_chunk = chunk
            current_node = chunk.get("current_node")
            if current_node and (not node_trace or node_trace[-1] != current_node):
                node_trace.append(str(current_node))
    except Exception as exc:
        error = str(exc)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    rows = _select_final_rows(last_chunk)
    contexts: list[str] = []
    retrieved_context_video_ids: list[str] = []
    for row in rows[:top_k]:
        text = _row_context_text(row)
        if text and text not in contexts:
            contexts.append(text)
            retrieved_context_video_ids.append(str(row.get("video_id", "") or "").strip())
    top_video_ids = [str(row.get("video_id", "")).strip() for row in rows[:top_k] if str(row.get("video_id", "")).strip()]
    # Normalize both expected and retrieved video_ids so naming variants like
    # ``Normal_Videos_924_x264`` and ``Normal_Videos924_x264`` are treated as
    # the same ID when computing top_hit.
    normalized_top_video_ids = [_normalize_video_id(vid) for vid in top_video_ids]
    response = _strip_sources(last_chunk.get("final_answer"))
    predicted_span = _extract_predicted_span(rows)
    cr = last_chunk.get("classification_result")
    sd = last_chunk.get("sql_debug")
    fusion_meta = sd.get("fusion_meta") if isinstance(sd, dict) else None
    return {
        "case_id": case.get("case_id"),
        "source_sheet": case.get("source_sheet"),
        "video_id": case.get("video_id"),
        "question": question,
        "reference_answer": case.get("reference_answer"),
        "reference_answer_rich": case.get("reference_answer_rich"),
        "reference_scene_description": case.get("reference_scene_description"),
        "recall_challenge": case.get("recall_challenge"),
        "expected_answer_label": case.get("expected_answer_label"),
        "expected_time_raw": case.get("expected_time_raw"),
        "expected_start_sec": case.get("expected_start_sec"),
        "expected_end_sec": case.get("expected_end_sec"),
        "expected_time_is_approx": case.get("expected_time_is_approx"),
        "elapsed_ms": elapsed_ms,
        "route_mode": ((last_chunk.get("tool_choice") or {}).get("mode") if isinstance(last_chunk.get("tool_choice"), dict) else None),
        "classification_result": cr if isinstance(cr, dict) else {},
        "answer_type": str(last_chunk.get("answer_type") or "").strip(),
        "verifier_result": last_chunk.get("verifier_result") if isinstance(last_chunk.get("verifier_result"), dict) else {},
        "fusion_meta": fusion_meta if isinstance(fusion_meta, dict) else {},
        "routing_metrics": last_chunk.get("routing_metrics") if isinstance(last_chunk.get("routing_metrics"), dict) else {},
        "error": error,
        "tool_error": last_chunk.get("tool_error"),
        "node_trace": node_trace,
        "response": response,
        "retrieved_contexts": contexts,
        "retrieved_context_video_ids": retrieved_context_video_ids,
        "top_video_ids": top_video_ids,
        "top_hit": _normalize_video_id(str(case.get("video_id", ""))) in normalized_top_video_ids,
        "predicted_video_id": predicted_span["predicted_video_id"],
        "predicted_start_sec": predicted_span["predicted_start_sec"],
        "predicted_end_sec": predicted_span["predicted_end_sec"],
        "raw_summary_result": last_chunk.get("summary_result") if isinstance(last_chunk.get("summary_result"), dict) else {},
    }


def _build_summary(case_results: list[dict[str, Any]], dataset_report: dict[str, Any], bootstrap_result: dict[str, Any] | None) -> dict[str, Any]:
    retrieval_cases = [item["ragas"]["retrieval"] for item in case_results]
    generation_cases = [item["ragas"]["generation"] for item in case_results]
    e2e_cases = [item["ragas"]["end_to_end"] for item in case_results]
    temporal_cases = [item.get("temporal") or {} for item in case_results]
    temporal_scores = [item.get("time_range_overlap_iou") for item in temporal_cases]
    temporal_eligible = [item for item in temporal_cases if item.get("eligible")]
    localization_scores = [item.get("localization_score") for item in temporal_cases]
    localization_eligible = [item for item in temporal_cases if item.get("localization_score") is not None]
    # Top-1 video match uses ``video_match_score`` populated for any case with a GT video.
    video_match_scores = [item.get("video_match_score") for item in temporal_cases if item.get("video_match_score") is not None]
    rich_ref_cases = sum(
        1
        for item in case_results
        if str((item.get("ragas_input_profile") or {}).get("reference_source") or "") == "rich"
    )
    return {
        "dataset": dataset_report,
        "bootstrap": bootstrap_result,
        "case_count": len(case_results),
        "success_count": sum(1 for item in case_results if not item.get("error")),
        "top_hit_rate": round(sum(1 for item in case_results if item.get("top_hit")) / max(len(case_results), 1), 4),
        "avg_latency_ms": round(sum(float(item.get("elapsed_ms", 0.0)) for item in case_results) / max(len(case_results), 1), 2),
        "retrieval_summary": {
            "context_precision_avg": _mean([item.get("context_precision") for item in retrieval_cases]),
            "context_recall_avg": _mean([item.get("context_recall") for item in retrieval_cases]),
            "reference_used_rich_count": rich_ref_cases,
            "retrieval_metric_backend": (
                (retrieval_cases[0].get("retrieval_metric_backend") if retrieval_cases else None) or "llm"
            ),
        },
        "generation_summary": {
            "faithfulness_avg": _mean([item.get("faithfulness") for item in generation_cases]),
            "factual_correctness_avg": _mean([item.get("factual_correctness") for item in generation_cases]),
            "custom_correctness_avg": _mean([item.get("custom_correctness") for item in generation_cases]),
        },
        "temporal_summary": {
            "time_range_overlap_iou_avg": _mean(temporal_scores),
            "temporal_iou_avg": _mean(temporal_scores),
            "time_range_overlap_iou_case_count": len(temporal_eligible),
            "time_range_overlap_iou_hit_rate_at_0_3": round(
                sum(1 for item in temporal_eligible if isinstance(item.get("time_range_overlap_iou"), (int, float)) and float(item["time_range_overlap_iou"]) >= 0.3)
                / max(len(temporal_eligible), 1),
                4,
            ) if temporal_eligible else None,
            "time_range_overlap_iou_hit_rate_at_0_5": round(
                sum(1 for item in temporal_eligible if isinstance(item.get("time_range_overlap_iou"), (int, float)) and float(item["time_range_overlap_iou"]) >= 0.5)
                / max(len(temporal_eligible), 1),
                4,
            ) if temporal_eligible else None,
        },
        "localization_summary": {
            "video_match_score_avg": _mean(video_match_scores),
            "video_match_case_count": len(video_match_scores),
            "localization_score_avg": _mean(localization_scores),
            "localization_case_count": len(localization_eligible),
            "localization_hit_rate_at_0_3": round(
                sum(1 for item in localization_eligible if isinstance(item.get("localization_score"), (int, float)) and float(item["localization_score"]) >= 0.3)
                / max(len(localization_eligible), 1),
                4,
            ) if localization_eligible else None,
            "localization_hit_rate_at_0_5": round(
                sum(1 for item in localization_eligible if isinstance(item.get("localization_score"), (int, float)) and float(item["localization_score"]) >= 0.5)
                / max(len(localization_eligible), 1),
                4,
            ) if localization_eligible else None,
        },
        "end_to_end_summary": {
            "ragas_e2e_score_avg": _mean([item.get("ragas_e2e_score") for item in e2e_cases]),
        },
        "error_summary": {
            "ragas_metric_error_cases": sum(1 for item in case_results if (item.get("ragas") or {}).get("metric_errors")),
            "graph_error_cases": sum(1 for item in case_results if item.get("error")),
        },
    }


def _build_markdown(summary: dict[str, Any], case_results: list[dict[str, Any]]) -> str:
    lines = [
        "# RAGAS Eval Report",
        "",
        "## Summary",
        f"- Case count: `{summary['case_count']}`",
        f"- Success count: `{summary['success_count']}`",
        f"- Top hit rate: `{summary['top_hit_rate']}`",
        f"- Avg latency ms: `{summary['avg_latency_ms']}`",
        "",
        "## Retrieval",
        f"- Metric backend: `{summary['retrieval_summary'].get('retrieval_metric_backend', 'llm')}` "
        f"(``llm`` = semantic RAGAS; ``id_based`` = chunk same-video precision + RAGAS ID recall, no LLM)",
        f"- Context precision avg: `{summary['retrieval_summary']['context_precision_avg']}`",
        f"- Context recall avg: `{summary['retrieval_summary']['context_recall_avg']}`",
        "",
        "## Generation",
        f"- Faithfulness avg: `{summary['generation_summary']['faithfulness_avg']}`",
        f"- Custom correctness avg (rule-based, P1-Next-C): `{summary['generation_summary']['custom_correctness_avg']}`",
        f"- Factual correctness avg (RAGAS LLM, reference): `{summary['generation_summary']['factual_correctness_avg']}`",
        "",
        "## Temporal Localization",
        f"- Time overlap IoU avg: `{summary['temporal_summary']['time_range_overlap_iou_avg']}`",
        f"- Time overlap case count: `{summary['temporal_summary']['time_range_overlap_iou_case_count']}`",
        f"- Time overlap hit@0.3: `{summary['temporal_summary']['time_range_overlap_iou_hit_rate_at_0_3']}`",
        f"- Time overlap hit@0.5: `{summary['temporal_summary']['time_range_overlap_iou_hit_rate_at_0_5']}`",
        "",
        "## Task-native Localization (challenge.md §5.2)",
        f"- Video match score avg (top-1): `{summary['localization_summary']['video_match_score_avg']}`"
        f" (cases={summary['localization_summary']['video_match_case_count']})",
        f"- Localization score avg: `{summary['localization_summary']['localization_score_avg']}`"
        f" (cases={summary['localization_summary']['localization_case_count']})",
        f"- Localization hit@0.3: `{summary['localization_summary']['localization_hit_rate_at_0_3']}`",
        f"- Localization hit@0.5: `{summary['localization_summary']['localization_hit_rate_at_0_5']}`",
        "",
        "## End To End",
        f"- RAGAS e2e avg: `{summary['end_to_end_summary']['ragas_e2e_score_avg']}`",
        f"- RAGAS reference used rich: `{summary['retrieval_summary']['reference_used_rich_count']}`"
        f" / `{summary['case_count']}`",
        "",
        "## Cases",
    ]
    for case in case_results:
        lines.extend(
            [
                f"### {case['case_id']}",
                f"- Sheet: `{case['source_sheet']}`",
                f"- Video: `{case['video_id']}`",
                f"- Question: {case['question']}",
                f"- Route mode: `{case['route_mode']}`",
                f"- Top hit: `{case['top_hit']}`",
                f"- Retrieval: `{json.dumps(case['ragas']['retrieval'], ensure_ascii=False)}`",
                f"- Generation: `{json.dumps(case['ragas']['generation'], ensure_ascii=False)}`",
                f"- Temporal: `{json.dumps(case.get('temporal', {}), ensure_ascii=False)}`",
                f"- End-to-end: `{json.dumps(case['ragas']['end_to_end'], ensure_ascii=False)}`",
                f"- Metric errors: `{json.dumps(case['ragas'].get('metric_errors', {}), ensure_ascii=False)}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for Part4 agent cases")
    parser.add_argument(
        "--xlsx-path",
        type=str,
        default=str(DEFAULT_XLSX_PATH),
        help="Evaluation spreadsheet (default: agent/test/data/agent_test.xlsx)",
    )
    parser.add_argument(
        "--seed-dir",
        type=str,
        default="",
        help=(
            "Directory with *_events_vector_flat.json seeds. Empty = auto: "
            "test/data/events_vector_flat, test/data/ucfcrime_events_vector_flat, "
            "test/data, then generated/datasets/ucfcrime_events_vector_flat"
        ),
    )
    parser.add_argument(
        "--from-dataset-dir",
        type=str,
        default="",
        help=(
            "Skip xlsx import; load agent_test_normalized.json from this folder "
            "(often agent/test/data/imported after running agent_test_importer.py)"
        ),
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument(
        "--include-sheets",
        nargs="*",
        default=None,
        metavar="SHEET",
        help=(
            "Sheets to include for xlsx import. Default Part4 when this flag is omitted or given with no sheet names "
            "(explicit values override; Part2 and Part6 are always skipped)."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit case count for smoke test")
    parser.add_argument(
        "--case-ids-file",
        type=str,
        default="",
        help="Path to a file with case IDs (one per line) to filter evaluation. Useful for re-running a specific subset (e.g. hard cases).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many retrieved rows to evaluate")
    parser.add_argument("--prepare-subset-db", action="store_true", help="Build subset sqlite/chroma from selected video ids")
    parser.add_argument("--sqlite-path", type=str, default="", help="Use an existing sqlite db path")
    parser.add_argument("--chroma-path", type=str, default="", help="Use an existing chroma path")
    parser.add_argument("--child-collection", type=str, default="ucfcrime_eval_child", help="Chroma child collection name")
    parser.add_argument("--parent-collection", type=str, default="ucfcrime_eval_parent", help="Chroma parent collection name")
    parser.add_argument("--event-collection", type=str, default="ucfcrime_eval_event", help="Chroma event-level collection name")
    parser.add_argument("--ragas-model", type=str, default=DEFAULT_RAGAS_MODEL, help="RAGAS evaluation LLM model")
    parser.add_argument("--ragas-openai-base-url", type=str, default="", help="Optional base url for RAGAS OpenAI client")
    parser.add_argument("--ragas-concurrency", type=int, default=4, help="Parallel RAGAS scoring concurrency")
    parser.add_argument("--ragas-case-batch-size", type=int, default=4, help="How many cases to score in one asyncio batch")
    # P1-Next-F R3 (2026-05-02): default 3 -> 5 to match rerank_top_k so multi-fact
    # answers have a chance of being attributed across multiple chunks; total chars
    # 1800 -> 3000 to avoid _compact_contexts truncating those extra chunks.
    parser.add_argument("--ragas-max-contexts", type=int, default=5, help="Max contexts passed into RAGAS")
    parser.add_argument("--ragas-max-context-chars", type=int, default=700, help="Max chars per context for RAGAS")
    parser.add_argument("--ragas-max-total-context-chars", type=int, default=3000, help="Max total context chars for RAGAS")
    parser.add_argument("--ragas-max-response-chars", type=int, default=900, help="Max response chars for RAGAS")
    parser.add_argument("--ragas-max-reference-chars", type=int, default=700, help="Max reference chars for RAGAS")
    parser.add_argument("--ragas-metric-max-retries", type=int, default=4, help="Max retries for retryable RAGAS metric failures")
    parser.add_argument("--ragas-metric-retry-delay-sec", type=float, default=2.0, help="Base retry delay for retryable RAGAS metric failures")
    parser.add_argument(
        "--ragas-contexts-filter",
        type=str,
        default="none",
        choices=["none", "same_expected_video"],
        help=(
            "Subset retrieved_contexts before RAGAS metrics. "
            "same_expected_video: keep only chunks whose video_id matches the case GT video_id "
            "(reduces cross-video noise in context_precision / context_recall). Default: none."
        ),
    )
    parser.add_argument(
        "--retrieval-metrics-backend",
        type=str,
        default="llm",
        choices=["llm", "id_based"],
        help=(
            "How to score context_precision / context_recall. "
            "llm: RAGAS ContextPrecisionWithReference + ContextRecall (OpenAI calls). "
            "id_based: context_precision = fraction of retrieved chunks whose Video-id matches GT; "
            "context_recall = RAGAS IDBasedContextRecall (GT video id in retrieved id set). No LLM."
        ),
    )
    parser.add_argument(
        "--ragas-skip-llm-metrics",
        action="store_true",
        help=(
            "Do not call OpenAI for RAGAS generation metrics (faithfulness / factual_correctness). "
            "Requires --retrieval-metrics-backend id_based so context_precision / context_recall stay LLM-free. "
            "Does not disable the agent graph's own LLM calls."
        ),
    )
    # challenge.md §5.1: default-on rich reference for RAGAS.
    parser.add_argument(
        "--ragas-use-rich-reference",
        dest="ragas_use_rich_reference",
        action="store_true",
        default=True,
        help="Use reference_answer_rich (video + time + scene description) for RAGAS scoring (default)",
    )
    parser.add_argument(
        "--ragas-no-rich-reference",
        dest="ragas_use_rich_reference",
        action="store_false",
        help="Force legacy pointer-style reference_answer for RAGAS scoring (for A/B comparison)",
    )
    parser.add_argument(
        "--no-report-tables",
        action="store_true",
        help="Skip writing REPORT_TABLES.md (custom metrics + tables; see eval_report_tables.py)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _default_paths(output_dir)
    if args.include_sheets is None or len(args.include_sheets) == 0:
        include_sheets = list(DEFAULT_INCLUDE_SHEETS)
    else:
        include_sheets = [str(item).strip() for item in args.include_sheets if str(item).strip()]
        if not include_sheets:
            include_sheets = list(DEFAULT_INCLUDE_SHEETS)

    if str(args.from_dataset_dir or "").strip():
        cases, dataset_report = _load_cases_from_dataset_dir(Path(args.from_dataset_dir))
    else:
        cases, dataset_report = _load_filtered_cases(
            xlsx_path=Path(args.xlsx_path).expanduser().resolve(),
            dataset_dir=paths.dataset_dir,
            include_sheets=include_sheets,
        )
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    if args.case_ids_file:
        ids_path = Path(args.case_ids_file).expanduser().resolve()
        target_ids = {line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")}
        cases = [c for c in cases if c.get("case_id") in target_ids]
        print(f"[ragas_eval] Filtered to {len(cases)} cases (from --case-ids-file {ids_path})")
    if not cases:
        raise RuntimeError("No evaluation cases selected")

    if getattr(args, "ragas_skip_llm_metrics", False) and str(
        getattr(args, "retrieval_metrics_backend", "llm") or "llm"
    ).strip().lower() != "id_based":
        raise RuntimeError(
            "--ragas-skip-llm-metrics requires --retrieval-metrics-backend id_based "
            "(otherwise context_precision / context_recall need the RAGAS LLM)."
        )

    bootstrap_result = None
    sqlite_path = Path(args.sqlite_path).expanduser().resolve() if args.sqlite_path else paths.sqlite_path
    chroma_path = Path(args.chroma_path).expanduser().resolve() if args.chroma_path else paths.chroma_path

    if args.prepare_subset_db:
        seed_base = Path(args.seed_dir).expanduser().resolve() if str(args.seed_dir or "").strip() else resolve_default_seed_directory()
        print(f"[ragas_eval] Using seed directory: {seed_base}")
        seed_files = _resolve_seed_files(seed_base, cases)
        bootstrap_result = _prepare_subset_databases(
            paths=paths,
            seed_files=seed_files,
            child_collection=args.child_collection,
            parent_collection=args.parent_collection,
            event_collection=args.event_collection,
        )

    wall_t0 = time.perf_counter()
    graph = _load_graph_with_runtime_env(
        sqlite_path=sqlite_path,
        chroma_path=chroma_path,
        child_collection=args.child_collection,
        parent_collection=args.parent_collection,
        event_collection=args.event_collection,
    )
    ragas_llm, ragas_runtime_profile = _build_ragas_runtime(args)

    case_results: list[dict[str, Any]] = []
    graph_total_ms = 0.0
    for idx, case in enumerate(cases, start=1):
        print(f"Running eval case {idx}/{len(cases)}: {case['case_id']} - {case['question']}")
        result = _run_case(graph, case, idx, args.top_k)
        graph_total_ms += float(result.get("elapsed_ms", 0.0))
        case_results.append(result)

    async def _score_all_cases() -> None:
        semaphore = asyncio.Semaphore(max(1, int(args.ragas_concurrency)))

        use_rich_reference = bool(getattr(args, "ragas_use_rich_reference", True))
        total = len(case_results)
        # Mutable box so the nested coroutine can increment a shared completion counter
        # without needing a threading lock (single-event-loop cooperative scheduling).
        done_counter = {"n": 0}

        async def _score_one(idx: int, case_result: dict[str, Any]) -> None:
            case_id = case_result.get("case_id", f"idx_{idx}")
            question = _truncate_text(case_result.get("question"), 500)
            response = _truncate_text(case_result.get("response"), int(args.ragas_max_response_chars))
            # challenge.md §5.1: RAGAS reference should carry scene content so
            # context_precision / context_recall / faithfulness have real signal.
            reference_source = "rich"
            reference_raw = None
            if use_rich_reference:
                reference_raw = case_result.get("reference_answer_rich")
            if not reference_raw:
                reference_raw = case_result.get("reference_answer")
                reference_source = "legacy_pointer"
            reference = _truncate_text(reference_raw, int(args.ragas_max_reference_chars))
            raw_contexts = list(case_result.get("retrieved_contexts") or [])
            filter_mode = str(getattr(args, "ragas_contexts_filter", "none") or "none").strip().lower()
            filter_meta: dict[str, Any] = {"ragas_contexts_filter": filter_mode}
            if filter_mode == "same_expected_video":
                raw_contexts, fm = _filter_ragas_contexts_same_expected_video(case_result, raw_contexts)
                filter_meta.update(fm)
            contexts = _compact_contexts(
                raw_contexts,
                max_contexts=int(args.ragas_max_contexts),
                max_chars_per_context=int(args.ragas_max_context_chars),
                max_total_chars=int(args.ragas_max_total_context_chars),
            )
            ctx_total_chars = sum(len(item) for item in contexts)
            print(
                f"[ragas] start {idx}/{total} {case_id} "
                f"ctx={len(contexts)} ctx_chars={ctx_total_chars} "
                f"resp_chars={len(response)} ref_chars={len(reference)} "
                f"ctx_filter={filter_mode}",
                flush=True,
            )
            score_t0 = time.perf_counter()
            async with semaphore:
                ragas_result = await _score_case_with_ragas(
                    question=question,
                    response=response,
                    reference=reference,
                    contexts=contexts,
                    ragas_llm=ragas_llm,
                    metric_max_retries=max(1, int(args.ragas_metric_max_retries)),
                    metric_retry_delay_sec=max(0.5, float(args.ragas_metric_retry_delay_sec)),
                    retrieval_metrics_backend=str(
                        getattr(args, "retrieval_metrics_backend", "llm") or "llm"
                    ),
                    id_reference_video_id=str(case_result.get("video_id") or ""),
                )
            elapsed_ms = round((time.perf_counter() - score_t0) * 1000, 2)
            case_result["ragas_elapsed_ms"] = elapsed_ms
            case_result["ragas_input_profile"] = {
                "context_count": len(contexts),
                "context_total_chars": ctx_total_chars,
                "response_chars": len(response),
                "reference_chars": len(reference),
                "reference_source": reference_source,
                "reference_text": reference,
                "question_chars": len(question),
                **filter_meta,
            }
            cc = _compute_custom_correctness(case_result)
            gen = ragas_result.setdefault("generation", {})
            gen["custom_correctness"] = cc["score"]
            gen["custom_correctness_detail"] = cc["detail"]
            ragas_result["end_to_end"]["ragas_e2e_score"] = _mean(
                [
                    ragas_result.get("retrieval", {}).get("context_precision"),
                    ragas_result.get("retrieval", {}).get("context_recall"),
                    gen.get("custom_correctness"),
                    gen.get("faithfulness"),
                ]
            )
            case_result["ragas"] = ragas_result
            case_result["retrieved_contexts_for_ragas"] = contexts
            case_result["temporal"] = _score_time_range_overlap(case_result)
            done_counter["n"] += 1
            metric_errors = (ragas_result or {}).get("metric_errors") or {}
            err_tag = f" errors={sorted(metric_errors)}" if metric_errors else ""
            print(
                f"[ragas] done  {idx}/{total} {case_id} "
                f"in {elapsed_ms / 1000:.1f}s (completed {done_counter['n']}/{total}){err_tag}",
                flush=True,
            )

        indexed = list(enumerate(case_results, start=1))
        for batch in _chunked(indexed, max(1, int(args.ragas_case_batch_size))):
            await asyncio.gather(*[_score_one(idx, item) for idx, item in batch])

    ragas_t0 = time.perf_counter()
    asyncio.run(_score_all_cases())
    ragas_total_ms = round((time.perf_counter() - ragas_t0) * 1000, 2)

    retrieval_report = {"cases": [{k: v for k, v in item.items() if k in {"case_id", "video_id", "question", "retrieved_contexts", "top_video_ids", "top_hit", "ragas"}} for item in case_results]}
    generation_report = {"cases": [{k: v for k, v in item.items() if k in {"case_id", "video_id", "question", "response", "reference_answer", "ragas"}} for item in case_results]}
    e2e_report = {"cases": case_results}
    summary = _build_summary(case_results, dataset_report, bootstrap_result)
    summary["runtime_profile"] = {
        "ragas_runtime": ragas_runtime_profile,
        "eval_config": {
            "top_k": args.top_k,
            "ragas_concurrency": args.ragas_concurrency,
            "ragas_case_batch_size": args.ragas_case_batch_size,
            "ragas_max_contexts": args.ragas_max_contexts,
            "ragas_max_context_chars": args.ragas_max_context_chars,
            "ragas_max_total_context_chars": args.ragas_max_total_context_chars,
            "ragas_max_response_chars": args.ragas_max_response_chars,
            "ragas_max_reference_chars": args.ragas_max_reference_chars,
            "ragas_metric_max_retries": args.ragas_metric_max_retries,
            "ragas_metric_retry_delay_sec": args.ragas_metric_retry_delay_sec,
            "ragas_contexts_filter": getattr(args, "ragas_contexts_filter", "none"),
            "retrieval_metrics_backend": getattr(args, "retrieval_metrics_backend", "llm"),
            "ragas_skip_llm_metrics": bool(getattr(args, "ragas_skip_llm_metrics", False)),
        },
        "timing": {
            "graph_total_ms": round(graph_total_ms, 2),
            "graph_avg_ms_per_case": round(graph_total_ms / max(len(case_results), 1), 2),
            "ragas_total_ms": ragas_total_ms,
            "ragas_avg_ms_per_case": round(ragas_total_ms / max(len(case_results), 1), 2),
            "wall_total_ms": round((time.perf_counter() - wall_t0) * 1000, 2),
        },
    }

    _write_json(paths.retrieval_report_json, retrieval_report)
    _write_json(paths.generation_report_json, generation_report)
    _write_json(paths.e2e_report_json, e2e_report)
    _write_json(paths.summary_report_json, summary)
    paths.summary_report_md.write_text(_build_markdown(summary, case_results), encoding="utf-8")

    if not bool(getattr(args, "no_report_tables", False)):
        try:
            from eval_report_tables import write_eval_report_tables

            report_tables_path = write_eval_report_tables(output_dir)
            print(f"[ragas_eval] REPORT_TABLES.md -> {report_tables_path}", flush=True)
        except Exception as exc:
            print(f"[ragas_eval] REPORT_TABLES.md skipped: {exc}", flush=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
