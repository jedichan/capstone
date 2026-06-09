"""Eval service — runs sampled questions through the agent and reports metrics."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_DEFAULT_QUESTIONS = _HERE.parent / "agent" / "test" / "mevid_test" / "sampled_10.json"


def _parse_yes_no(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"^[\s`*_>-]+", "", t)
    if t.startswith("yes") or t.startswith("是"):
        return "yes"
    if t.startswith("no") or t.startswith("否"):
        return "no"
    return "unknown"


def _parse_time(t_str: str) -> float | None:
    parts = str(t_str).strip().split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + float(parts[1])
        except (ValueError, TypeError):
            return None
    try:
        return float(str(t_str).replace("s", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_time_range(t_str: str) -> tuple[float | None, float | None]:
    if not t_str or str(t_str).upper() == "N/A":
        return None, None
    t_str = str(t_str).strip().replace("[", "").replace("]", "")
    if "-" not in t_str:
        return None, None
    parts = t_str.split("-")
    if len(parts) >= 2:
        return _parse_time(parts[0]), _parse_time(parts[1])
    return None, None


def _extract_time_from_rows(
    top_rows: list[dict], expected_video_id: str = ""
) -> tuple[float, float] | None:
    """Extract time range from top retrieved rows, preferring expected video_id.
    Mirrors run_agent_eval.py _extract_time_from_rows exactly.
    """
    if expected_video_id:
        for r in top_rows:
            if str(r.get("video_id", "")).strip() == expected_video_id:
                st, et = r.get("start_time"), r.get("end_time")
                if st is not None and et is not None:
                    try:
                        s, e = float(st), float(et)
                        if e > s:
                            return (s, e)
                    except (ValueError, TypeError):
                        continue
    for r in top_rows:
        st, et = r.get("start_time"), r.get("end_time")
        if st is not None and et is not None:
            try:
                s, e = float(st), float(et)
                if e > s:
                    return (s, e)
            except (ValueError, TypeError):
                continue
    return None


def _extract_time_from_answer(answer: str) -> tuple[float, float] | None:
    m = re.search(r"\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]", answer)
    if m:
        s, e = _parse_time(m.group(1)), _parse_time(m.group(2))
        if s is not None and e is not None:
            return (s, e)
    m = re.search(r"(\d+\.?\d*)\s*s\s*-\s*(\d+\.?\d*)\s*s", answer)
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)))
        except (ValueError, TypeError):
            pass
    return None


def _compute_temporal_iou(
    pred: tuple[float, float] | None,
    exp: tuple[float, float] | None,
    padding_sec: float = 30.0,
) -> float:
    if pred is None or exp is None:
        return 0.0
    if any(v is None for v in (*pred, *exp)):
        return 0.0
    ps, pe = pred[0] - padding_sec, pred[1] + padding_sec
    es, ee = exp
    inter = max(0.0, min(pe, ee) - max(ps, es))
    union = max(pe, ee) - min(ps, es)
    return inter / union if union > 0 else 0.0


def run_eval(
    questions_file: str | None,
    agent_query_fn,  # callable(query: str) -> dict
) -> dict[str, Any]:
    """Run evaluation questions through agent_query_fn and return summary.

    Args:
        questions_file: Path to sampled JSON file; uses default sampled_10.json if None.
        agent_query_fn: Function that takes a query string and returns a dict with 'answer' key.
    """
    qfile = Path(questions_file) if questions_file else _DEFAULT_QUESTIONS
    if not qfile.exists():
        raise FileNotFoundError(f"Questions file not found: {qfile}")

    questions: list[dict] = json.loads(qfile.read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("Questions file is empty")

    total_t0 = time.perf_counter()
    cases: list[dict] = []
    category_stats: dict[str, dict] = {}

    for idx, q in enumerate(questions, start=1):
        question = q["question"]
        expected = q["expected"]
        expected_video_id = q.get("video_id", "")
        category = q.get("category", "unknown")
        expected_time_str = q.get("expected_time", "N/A")

        t0 = time.perf_counter()
        result = agent_query_fn(question)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Use final_answer (full text including time markers) for IoU; answer (stripped) for yes/no
        final_answer_text = str(result.get("final_answer") or result.get("answer") or "").strip()
        answer_raw = result.get("answer", "")
        predicted = _parse_yes_no(answer_raw)
        correct = predicted == expected

        # Top-hit: check if expected video_id appears in any of the top rows — matches run_agent_eval.py
        top_rows = result.get("rows", [])
        top_video_ids = [str(r.get("video_id") or "").strip() for r in top_rows if r.get("video_id")]
        top_hit = expected_video_id in top_video_ids if expected_video_id else False

        # IoU (only for yes-expected questions) — mirrors run_agent_eval.py exactly
        exp_start, exp_end = _parse_time_range(expected_time_str)
        exp_range = (exp_start, exp_end) if exp_start is not None else None
        # Try to extract from final_answer text first, then fall back to top rows
        pred_range = _extract_time_from_answer(final_answer_text)
        if pred_range is None:
            pred_range = _extract_time_from_rows(top_rows, expected_video_id)
        iou = _compute_temporal_iou(pred_range, exp_range) if expected == "yes" else 0.0
        iou_pass = correct and (iou >= 0.15 if expected == "yes" else True)

        cat_stats = category_stats.setdefault(
            category, {"total": 0, "correct": 0, "top_hit": 0, "iou_pass": 0}
        )
        cat_stats["total"] += 1
        if correct:
            cat_stats["correct"] += 1
        if top_hit:
            cat_stats["top_hit"] += 1
        if iou_pass:
            cat_stats["iou_pass"] += 1

        cases.append({
            "case_id": f"Q{idx:02d}",
            "category": category,
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "top_hit": top_hit,
            "iou": round(iou, 4),
            "iou_pass": iou_pass,
            "elapsed_ms": elapsed_ms,
            "answer_raw": answer_raw[:300],
        })

    total_elapsed = round((time.perf_counter() - total_t0) * 1000, 2)

    # Aggregate — mirrors run_agent_eval.py metric calculations
    yes_ious = [c["iou"] for c in cases if c["expected"] == "yes"]
    mean_iou = sum(yes_ious) / len(yes_ious) if yes_ious else 0.0
    n_iou_pass = sum(1 for c in cases if c.get("iou_pass", False))
    total_correct = sum(1 for c in cases if c["correct"])
    total_top_hit = sum(1 for c in cases if c["top_hit"])
    n = len(cases)

    by_category = {
        cat: {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0.0,
            "top_hit": s["top_hit"],
            "top_hit_rate": round(s["top_hit"] / s["total"], 4) if s["total"] else 0.0,
            "iou_pass": s["iou_pass"],
        }
        for cat, s in category_stats.items()
    }

    return {
        "total": n,
        "correct": total_correct,
        "accuracy": round(total_correct / n, 4) if n else 0.0,
        "top_hit_total": total_top_hit,
        "top_hit_rate": round(total_top_hit / n, 4) if n else 0.0,
        "mean_iou": round(mean_iou, 4),
        "iou_threshold": 0.15,
        "n_iou_pass": n_iou_pass,
        "iou_pass_rate": round(n_iou_pass / n, 4) if n else 0.0,
        "individual_iou": [round(c["iou"], 4) for c in cases if c["expected"] == "yes"],
        "by_category": by_category,
        "cases": cases,
        "elapsed_total_ms": total_elapsed,
    }
