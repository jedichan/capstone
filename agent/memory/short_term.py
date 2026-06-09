"""
Short-term conversation memory for the Capstone agent.

Stores only user query + summary_node final answer per turn, keyed by
(thread_id, user_id) in an in-memory dict. All intermediate results
(rewritten_query, sql_result, hybrid_result, rerank_result, etc.) are
discarded.

Only active when AGENT_ENABLE_SHORT_TERM_MEMORY=1 (default off), so
existing test pipelines are completely unaffected.
"""

import os
import threading
import time
from typing import Dict, List, Optional


def _is_enabled() -> bool:
    raw = os.getenv("AGENT_ENABLE_SHORT_TERM_MEMORY", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _max_turns() -> int:
    try:
        return max(1, int(os.getenv("AGENT_MEMORY_MAX_TURNS", "5")))
    except (ValueError, TypeError):
        return 5


# Key: f"{thread_id}:{user_id}", Value: list of turn dicts
_memory_store: Dict[str, List[Dict]] = {}
_lock = threading.Lock()


def _make_key(thread_id: str, user_id: str) -> str:
    return f"{thread_id}:{user_id}"


def add_turn(thread_id: str, user_id: str, query: str, answer: str) -> None:
    """Store one conversation turn. Only the query + summary answer are kept."""
    if not _is_enabled():
        return
    key = _make_key(thread_id, user_id)
    with _lock:
        turns = _memory_store.get(key, [])
        turns.append({"query": query, "answer": answer, "timestamp": time.time()})
        _memory_store[key] = turns[-_max_turns():]


def get_history(thread_id: str, user_id: str) -> List[Dict]:
    """Return the conversation history for a session, newest last."""
    if not _is_enabled():
        return []
    key = _make_key(thread_id, user_id)
    with _lock:
        return list(_memory_store.get(key, []))


def _format_turn(idx: int, turn: Dict) -> str:
    q = str(turn.get("query", "")).strip()
    a = str(turn.get("answer", "")).strip()
    return f"  Turn {idx}:\n    Q: {q}\n    A: {a}"


def format_history_for_prompt(
    thread_id: str,
    user_id: str,
    max_turns: Optional[int] = None,
) -> str:
    """Format previous turns as a prompt-friendly block, newest last."""
    turns = get_history(thread_id, user_id)
    if not turns:
        return ""

    limit = max_turns if isinstance(max_turns, int) and max_turns > 0 else len(turns)
    subset = turns[-limit:]

    lines = ["Conversation history (previous turns in this session):"]
    for i, turn in enumerate(subset, start=1):
        lines.append(_format_turn(i, turn))
    return "\n".join(lines)


def clear_session(thread_id: str, user_id: str) -> None:
    """Drop all memory for a session."""
    key = _make_key(thread_id, user_id)
    with _lock:
        _memory_store.pop(key, None)
