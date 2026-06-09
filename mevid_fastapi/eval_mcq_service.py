"""Multiple-choice QA evaluation service — ForeSea-style A/B/C/D format.

Two-stage evaluation:
  Stage 1 — Retrieval: Send the question to the LangGraph agent to fetch
             relevant clips and event descriptions from the vector DB.
  Stage 2 — Reasoning: Feed the retrieved context + MCQ choices to a judge
             LLM (Qwen via DashScope) that selects the correct letter.

This mirrors ForeSea's evaluation methodology (6 subtasks, MCQ accuracy +
temporal IoU) adapted to the MEVID multi-camera surveillance dataset.

Metrics reported:
  - Overall Accuracy (A/B/C/D correct letter)
  - Per-category Accuracy  (Search, Temporal, Event, Activity, Counting, Anomaly)
  - Mean temporal IoU (questions that have expected_time)
  - IoU pass rate (tIoU >= 0.15)
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_DEFAULT_MCQ_QUESTIONS = (
    _HERE.parent / "agent" / "test" / "mevid_test" / "mcq_questions.json"
)

_CATEGORY_DISPLAY = {
    "search": "Search",
    "temporal": "Temporal",
    "event": "Event",
    "activity": "Activity",
    "counting": "Counting",
    "anomaly": "Anomaly",
}

# Categories where retrieval must be restricted to a single specific camera
_SINGLE_CAM_CATEGORIES = {"search", "temporal", "activity"}
# Categories where retrieval spans multiple candidate cameras (choices are camera IDs)
_MULTI_CAM_CATEGORIES  = {"event"}


def _extract_choice_cameras(choices: dict[str, str]) -> list[str]:
    """Return camera IDs from choices when ALL choice values are camera IDs (e.g. 'G336').

    Event and cross-camera questions have choices like {"A": "G339", "B": "G506", ...}.
    Search/Activity questions have appearance/direction text — those return [].
    """
    import re
    pat = re.compile(r"^G\d+$", re.IGNORECASE)
    cams = [v.strip() for v in choices.values() if pat.match(v.strip())]
    return cams if len(cams) == len(choices) else []


# ── Category-specific judge guidance ─────────────────────────────────────────
# Each hint is injected into the judge prompt so the LLM knows how to reason
# about the specific sub-task type (description matching, time ranges, etc.).

_CATEGORY_JUDGE_HINTS: dict[str, str] = {
    "search": (
        "This is a person-search question. Match the description in the question "
        "to appearance notes in the retrieved clips. Focus on dominant clothing colour "
        "(dark/light/black/white), garment type (long coat vs jacket), and key accessories "
        "(bag, backpack, hat). 'Dark upper-body clothing' can mean a dark coat or jacket. "
        "Do NOT require word-for-word match — treat 'long black coat' and 'dark upper-body "
        "clothing' as equivalent. "
        "Prioritise the Agent summary if it names specific appearance features."
    ),
    "temporal": (
        "This is a temporal localisation question. The answer is a time window in MM:SS format. "
        "Look at the start_time and end_time fields in the retrieved clips and convert seconds "
        "to MM:SS (e.g., 70 s → 0:01:10, 125 s → 0:02:05). "
        "Find the clip whose time range contains the moment described in the question. "
        "If the Agent summary states a specific timestamp, convert it and match to a choice. "
        "Choose the option whose range brackets the relevant moment."
    ),
    "event": (
        "This is a cross-camera tracking question: which camera did the person appear in "
        "BEFORE or AFTER a named reference camera? "
        "Use the RETRIEVED CLIPS as primary evidence — the agent may have retrieved clips from "
        "the wrong cameras, so do NOT rely on the Agent summary alone if it conflicts with "
        "the clip data. "
        "Look for cross-camera trajectory phrases such as 'across G420, G299, G508', "
        "'moved from G339 to G328', or 'entity ... across G*'. "
        "Find the camera that immediately precedes or follows the reference camera in the "
        "trajectory sequence, and match it to the choices."
    ),
    "activity": (
        "This is a movement-activity question about the exit direction of a specific person. "
        "The question specifies a camera and a time window (e.g. 'around 0:01'). "
        "Find the retrieved clip from that camera whose timestamp matches the time window. "
        "Check the clip's event description for EXACT fields: "
        "'Exit side: left/right/up/down' or 'Movement direction: left/right/up/down'. "
        "These map directly to the choices. "
        "If no exact field is found, look for phrases: "
        "'walks down/downstairs' → Down; 'walks up/upstairs' → Up; "
        "'moving left' / 'walks left' → Left; 'moving right' / 'walks right' → Right. "
        "Use the timestamp to distinguish among multiple clips from the same camera."
    ),
    "counting": (
        "This is a counting question. The Agent summary usually states the count directly "
        "(e.g., 'I found 3 matching persons'). Match the stated number to the numeric range "
        "or exact value in the choices. If the Agent says 'approximately N', pick the closest. "
        "Retrieved clips can also be counted directly if the Agent summary is vague."
    ),
    "anomaly": (
        "This is an anomaly-detection question asking which camera-to-camera transition "
        "does NOT appear in the surveillance footage. "
        "Check each choice transition (e.g., G420→G339) against the retrieved clips: if a "
        "clip shows travel from camera A to camera B, that transition exists. "
        "The anomalous (correct) answer is the transition absent from all retrieved clips. "
        "If multiple are absent, choose the one least consistent with the known routes."
    ),
}

# ── Judge LLM ─────────────────────────────────────────────────────────────────

def _make_judge_llm():
    """Return a lightweight ChatOpenAI client for MCQ judging (Qwen via DashScope)."""
    from langchain_openai import ChatOpenAI

    model = os.environ.get(
        "AGENT_MCQ_JUDGE_MODEL",
        os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3-vl-flash"),
    )
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    dashscope_url = os.environ.get(
        "DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if model.lower().startswith("qwen") and dashscope_key:
        return ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=dashscope_key,
            base_url=dashscope_url,
        )
    # Fallback — whatever OpenAI key is in the environment
    return ChatOpenAI(model=model, temperature=0.0)


def _build_context_from_rows(rows: list[dict]) -> str:
    """Summarise retrieved rows into a text context for the judge."""
    parts: list[str] = []
    for i, r in enumerate(rows[:8], 1):
        cam = r.get("camera_id") or r.get("video_id") or ""
        t_start = r.get("start_time")
        t_end = r.get("end_time")
        time_str = ""
        if t_start is not None and t_end is not None:
            try:
                s_min, s_sec = divmod(int(float(t_start)), 60)
                e_min, e_sec = divmod(int(float(t_end)), 60)
                time_str = f" [{s_min}:{s_sec:02d}-{e_min}:{e_sec:02d}]"
            except (ValueError, TypeError):
                pass
        event_text = (
            r.get("event_text_en")
            or r.get("event_summary_en")
            or r.get("description_for_llm")
            or ""
        )
        appearance = r.get("appearance_notes_en") or r.get("object_color_en") or ""
        scene = r.get("scene_zone_en") or ""
        line = f"[Clip {i}] Camera {cam}{time_str}"
        if event_text:
            line += f" — {event_text}"
        if appearance:
            line += f" | Appearance: {appearance}"
        if scene:
            line += f" | Zone: {scene}"
        parts.append(line)
    return "\n".join(parts) if parts else "(no retrieved clips)"


def _judge_answer(
    question: str,
    choices: dict[str, str],
    agent_answer: str,
    rows: list[dict],
    category: str = "",
    llm=None,
) -> str:
    """Call the judge LLM to pick A/B/C/D based on retrieved context.

    The prompt is structured as:
      1. Role + category-specific task guidance
      2. Agent summary (high-confidence reasoned answer — listed first for emphasis)
      3. Retrieved clips (supporting evidence)
      4. Question + choices
    """
    if llm is None:
        llm = _make_judge_llm()

    context = _build_context_from_rows(rows)
    hint = _CATEGORY_JUDGE_HINTS.get(category.lower(), "")
    choices_text = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))

    # Build prompt: guidance → agent summary → clips → question → choices
    prompt_parts = [
        "You are a surveillance video analyst. "
        "Use the information below to answer the multiple-choice question. "
        "Reply with ONLY the single letter (A, B, C, or D) — nothing else.",
    ]
    if hint:
        prompt_parts.append(f"\nTask guidance: {hint}")
    prompt_parts.append(f"\nAgent summary: {agent_answer or '(none)'}")
    prompt_parts.append(f"\nRetrieved clips:\n{context}")
    prompt_parts.append(f"\nQuestion: {question}")
    prompt_parts.append(f"\n{choices_text}")
    prompt_parts.append("\nYour answer (one letter only):")

    prompt = "\n".join(prompt_parts)

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        return _parse_mcq_answer(str(response.content or "").strip())
    except Exception:  # noqa: BLE001
        return "unknown"


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_mcq_answer(text: str) -> str:
    """Extract the first A/B/C/D letter from a string."""
    t = str(text or "").strip()
    m = re.match(r"^\s*([A-D])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(
        r"\b(?:answer(?:\s+is)?|choice(?:\s+is)?|option(?:\s+is)?)[:\s]+([A-D])\b",
        t,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
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
    t_str = str(t_str).strip().replace("[", "").replace("]", "").replace("–", "-")
    if "-" not in t_str:
        return None, None
    parts = t_str.split("-")
    if len(parts) >= 2:
        return _parse_time(parts[0]), _parse_time(parts[1])
    return None, None


def _extract_time_from_rows(
    top_rows: list[dict], expected_video_id: str = ""
) -> tuple[float, float] | None:
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


# ── Main eval runner ──────────────────────────────────────────────────────────

def run_mcq_eval(
    questions_file: str | None,
    agent_query_fn,  # callable(query: str) -> dict
    use_camera_filter: bool = False,
) -> dict[str, Any]:
    """Run MCQ evaluation with two-stage retrieval + judge reasoning.

    Stage 1 — Retrieval: each question is sent as-is to the agent, which
    performs hybrid retrieval and returns the top-k clips with metadata.

    Stage 2 — Judging: a separate LLM call (Qwen judge) reads the retrieved
    clips + agent summary, then selects the correct A/B/C/D choice.

    Args:
        questions_file: Path to JSON with MCQ questions; uses built-in if None.
        agent_query_fn: Callable accepting a query string, returning a dict
            with at least {'answer': str, 'rows': list[dict]}.
        use_camera_filter: Retained for backward compatibility; camera-aware
            filtering is now always applied based on question category:
              - search/temporal/activity → "camera_id:<cam> <question>"
              - event                    → "camera_ids:<cam1>,<cam2>,... <question>"
              - counting/anomaly         → no filter (full-collection retrieval)

    Returns:
        Summary dict: overall accuracy, mean tIoU, IoU pass rate, by_category,
        per-case breakdown, elapsed_total_ms.
    """
    qfile = Path(questions_file) if questions_file else _DEFAULT_MCQ_QUESTIONS
    if not qfile.exists():
        raise FileNotFoundError(f"MCQ questions file not found: {qfile}")

    questions: list[dict] = json.loads(qfile.read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("MCQ questions file is empty")

    # Lazy-init the judge LLM once for all questions
    judge_llm = _make_judge_llm()

    total_t0 = time.perf_counter()
    cases: list[dict] = []
    category_stats: dict[str, dict] = {}

    for q in questions:
        qid = q.get("id", f"Q{len(cases)+1:02d}")
        category = q.get("category", "unknown")
        question_text = q["question"]
        choices: dict[str, str] = q["choices"]
        correct_answer = q["answer"].strip().upper()
        expected_video_id = q.get("video_id", "")
        expected_time_str = q.get("expected_time", "N/A")

        t0 = time.perf_counter()

        # ── Stage 1: retrieval ─────────────────────────────────────────────
        camera_id   = q.get("camera_id", "").strip()
        cat_lower   = category.lower()

        # Build a camera-aware retrieval query so the query functions can
        # apply DB-level filtering (ChromaDB where / SQLite AND camera_id).
        #
        # Encoding format:
        #   single camera  →  "camera_id:G336 <question>"
        #   multi  camera  →  "camera_ids:G339,G506,G420,G423 <question>"
        #   no filter      →  "<question>"
        if cat_lower in _SINGLE_CAM_CATEGORIES and camera_id:
            retrieval_query = f"cat:{cat_lower} camera_id:{camera_id} {question_text}"
        elif cat_lower in _MULTI_CAM_CATEGORIES and camera_id:
            choice_cams = _extract_choice_cameras(choices)
            # Include the reference camera so the agent can also find the
            # person at the reference point before/after which we reason.
            all_cams = list(dict.fromkeys([camera_id] + choice_cams))
            if all_cams:
                retrieval_query = f"cat:{cat_lower} camera_ids:{','.join(all_cams)} {question_text}"
            else:
                retrieval_query = f"cat:{cat_lower} {question_text}"
        else:
            retrieval_query = f"cat:{cat_lower} {question_text}"

        result = agent_query_fn(retrieval_query)
        agent_answer = str(result.get("answer") or "").strip()
        top_rows: list[dict] = result.get("rows", [])

        # ── Stage 2: judge picks A/B/C/D ──────────────────────────────────
        # Pass the category so the judge uses task-specific reasoning guidance.
        # We do NOT post-filter rows by camera here: agent rows may encode the
        # camera in video_id rather than camera_id, and removing cross-camera
        # trajectory rows hurts Event/Temporal reasoning.
        predicted = _judge_answer(
            question=question_text,
            choices=choices,
            agent_answer=agent_answer,
            rows=top_rows,
            category=category,
            llm=judge_llm,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        correct = predicted == correct_answer

        # Temporal IoU from retrieved rows
        exp_start, exp_end = _parse_time_range(expected_time_str)
        exp_range = (exp_start, exp_end) if exp_start is not None and exp_end is not None else None
        pred_range = _extract_time_from_rows(top_rows, expected_video_id)
        iou = _compute_temporal_iou(pred_range, exp_range) if exp_range is not None else 0.0
        iou_pass = correct and (iou >= 0.15 if exp_range is not None else True)

        cat_stats = category_stats.setdefault(
            category,
            {"total": 0, "correct": 0, "iou_sum": 0.0, "iou_count": 0, "iou_pass": 0},
        )
        cat_stats["total"] += 1
        if correct:
            cat_stats["correct"] += 1
        if exp_range is not None:
            cat_stats["iou_sum"] += iou
            cat_stats["iou_count"] += 1
        if iou_pass:
            cat_stats["iou_pass"] += 1

        cases.append(
            {
                "case_id": qid,
                "category": category,
                "question": question_text,
                "choices": choices,
                "expected": correct_answer,
                "predicted": predicted,
                "correct": correct,
                "iou": round(iou, 4),
                "iou_pass": iou_pass,
                "elapsed_ms": elapsed_ms,
                "agent_answer": agent_answer[:300],
            }
        )

    total_elapsed = round((time.perf_counter() - total_t0) * 1000, 2)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(cases)
    total_correct = sum(1 for c in cases if c["correct"])
    n_iou_pass = sum(1 for c in cases if c.get("iou_pass", False))

    # Mean IoU only over questions that have ground-truth time ranges
    timed_ious = [
        c["iou"]
        for c in cases
        if _parse_time_range(
            next(
                (qq["expected_time"] for qq in questions if qq.get("id") == c["case_id"]),
                "N/A",
            )
        )[0] is not None
    ]
    mean_iou = sum(timed_ious) / len(timed_ious) if timed_ious else 0.0

    by_category: dict[str, dict] = {}
    for cat, s in category_stats.items():
        by_category[_CATEGORY_DISPLAY.get(cat, cat)] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0.0,
            "mean_iou": round(s["iou_sum"] / s["iou_count"], 4) if s["iou_count"] else 0.0,
            "iou_pass": s["iou_pass"],
        }

    return {
        "total": n,
        "correct": total_correct,
        "accuracy": round(total_correct / n, 4) if n else 0.0,
        "mean_iou": round(mean_iou, 4),
        "iou_threshold": 0.15,
        "n_iou_pass": n_iou_pass,
        "iou_pass_rate": round(n_iou_pass / n, 4) if n else 0.0,
        "individual_iou": [round(c["iou"], 4) for c in cases],
        "by_category": by_category,
        "cases": cases,
        "elapsed_total_ms": total_elapsed,
    }
