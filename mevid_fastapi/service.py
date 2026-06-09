"""Agent service — manages LangGraph graph lifecycle and DB selection."""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _PROJECT_ROOT / "agent"
for _p in (str(_PROJECT_ROOT), str(_AGENT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import graph as graph_module  # noqa: E402
from tools.bm25_index import BM25Index  # noqa: E402


def _load_env() -> None:
    """Load .env from project root (API keys, model settings, etc.)."""
    env_file = _PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _strip_sources(text: str | None) -> str:
    body = str(text or "").strip()
    if "\nSources:" in body:
        body = body.split("\nSources:", 1)[0].strip()
    return body


def _fmt_sec(sec: Any) -> str:
    try:
        s = max(0, int(round(float(sec))))
        h, rem = divmod(s, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    except Exception:
        return "?"


def _parse_video_start(video_id: str | None) -> "datetime | None":
    """Extract wall-clock start time from video_id.

    Format: YYYY-MM-DD.HH-MM-SS.HH-MM-SS.<scene>.<cam>.<run>
    e.g.  2018-03-11.16-20-01.16-25-01.admin.G326.r13  → 16:20:01
    Returns a datetime object (date part included) or None if unparseable.
    """
    if not video_id:
        return None
    try:
        from datetime import datetime, timedelta
        parts = str(video_id).split(".")
        # parts[0] = date, parts[1] = start HH-MM-SS
        date_str = parts[0]           # "2018-03-11"
        time_str = parts[1]           # "16-20-01"
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H-%M-%S")
    except Exception:
        return None


def _abs_timestamp(video_id: str | None, rel_sec: Any) -> str:
    """Return wall-clock time string for a relative offset inside a video.

    e.g. video_id='2018-03-11.16-20-01…', rel_sec=18.3  →  '16:20:19'
    Falls back to _fmt_sec(rel_sec) if parsing fails.
    """
    t0 = _parse_video_start(video_id)
    if t0 is None:
        return _fmt_sec(rel_sec)
    try:
        from datetime import timedelta
        t = t0 + timedelta(seconds=float(rel_sec))
        return t.strftime("%H:%M:%S")
    except Exception:
        return _fmt_sec(rel_sec)


def _count_sql_filters(query: str) -> tuple[str | None, str, list[Any], str]:
    cam_match = re.search(r"\b(G\d{3})\b", query, re.IGNORECASE)
    camera_id = cam_match.group(1).upper() if cam_match else None

    low = query.lower()
    object_clause = ""
    object_params: list[Any] = []
    object_label = "tracked object(s)"
    if re.search(r"\b(person|people|pedestrian|human)\b", low):
        object_clause = " AND LOWER(object_type) = ?"
        object_params = ["person"]
        object_label = "different people"
    elif re.search(r"\b(cars?|vehicles?)\b", low):
        object_clause = " AND LOWER(object_type) IN ('car', 'truck', 'bus')"
        object_label = "different vehicle(s)"
    elif re.search(r"\btrucks?\b", low):
        object_clause = " AND LOWER(object_type) = ?"
        object_params = ["truck"]
        object_label = "different truck(s)"
    return camera_id, object_clause, object_params, object_label


def _try_count_sql(query: str, sqlite_path: str) -> str:
    """Run a COUNT query against SQLite for count-type questions.

    Extracts camera_id from the query (e.g. G326) if present, otherwise
    returns counts grouped by camera.
    """
    if not sqlite_path or not Path(sqlite_path).exists():
        return ""
    try:
        camera_id, object_clause, object_params, object_label = _count_sql_filters(query)
        conn = sqlite3.connect(sqlite_path, timeout=5)
        if camera_id:
            row = conn.execute(
                "SELECT COUNT(DISTINCT COALESCE(global_entity_id, track_id)) "
                f"FROM episodic_events WHERE UPPER(camera_id) = ?{object_clause}",
                [camera_id, *object_params],
            ).fetchone()
            conn.close()
            n = row[0] if row else 0
            return f"{n} {object_label} detected in camera {camera_id}."
        else:
            rows = conn.execute(
                "SELECT camera_id, COUNT(DISTINCT COALESCE(global_entity_id, track_id)) "
                f"FROM episodic_events WHERE 1=1{object_clause} GROUP BY camera_id ORDER BY 2 DESC LIMIT 8",
                object_params,
            ).fetchall()
            conn.close()
            if rows:
                parts = [f"{cam or '?'}: {n}" for cam, n in rows]
                return "Counts by camera — " + ", ".join(parts) + "."
    except Exception:
        pass
    return ""


def _count_evidence_rows(query: str, sqlite_path: str, limit: int = 8) -> list[dict[str, Any]]:
    """Return representative rows for SQL count answers so the UI can explain the count."""
    if not sqlite_path or not Path(sqlite_path).exists():
        return []
    try:
        conn = sqlite3.connect(sqlite_path, timeout=5)
        conn.row_factory = sqlite3.Row
        col_rows = conn.execute("PRAGMA table_info(episodic_events)").fetchall()
        cols = {str(r["name"]) for r in col_rows}

        def _expr(*names: str, default: str = "''") -> str:
            existing = [n for n in names if n in cols]
            if not existing:
                return default
            if len(existing) == 1:
                return existing[0]
            return "COALESCE(" + ", ".join(existing) + ")"

        event_text_expr = _expr("event_text", "event_text_en", "event_summary_en", "vector_doc_text")
        app_expr = _expr("appearance_notes", "appearance_notes_en")
        color_expr = _expr("object_color", "object_color_en")
        zone_expr = _expr("scene_zone", "scene_zone_en")
        move_expr = _expr("movement_direction")
        obj_expr = _expr("object_type")
        gid_expr = _expr("global_entity_id", "entity_hint")
        tid_expr = _expr("track_id")
        cam_expr = _expr("camera_id")
        video_expr = _expr("video_id")
        st_expr = _expr("start_time", default="0")
        et_expr = _expr("end_time", default="0")

        camera_id, object_clause, object_params, _ = _count_sql_filters(query)
        where = "WHERE 1=1"
        params: list[Any] = []
        if camera_id:
            where += " AND UPPER(camera_id) = ?"
            params.append(camera_id)
        where += object_clause
        params.extend(object_params)
        rows = conn.execute(
            f"""
            SELECT
                COALESCE({gid_expr}, 'track_' || CAST({tid_expr} AS TEXT)) AS entity_key,
                MIN({video_expr}) AS video_id,
                MIN({cam_expr}) AS camera_id,
                MIN({st_expr}) AS start_time,
                MAX({et_expr}) AS end_time,
                MIN({tid_expr}) AS track_id,
                MIN({gid_expr}) AS global_entity_id,
                MIN({event_text_expr}) AS event_text,
                MIN({app_expr}) AS appearance_notes,
                MIN({obj_expr}) AS object_type,
                MIN({color_expr}) AS object_color,
                MIN({zone_expr}) AS scene_zone,
                MIN({move_expr}) AS movement_direction
            FROM episodic_events
            {where}
            GROUP BY entity_key
            ORDER BY MIN(start_time)
            LIMIT ?
            """,
            [*params, int(limit)],
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _rewrite_natural(
    query: str,
    concise_answer: str,
    rows: list[dict],
    answer_type: str,
    sqlite_path: str,
) -> str:
    """Post-process the eval-style concise answer into a conversational response.

    Called only when response_style == 'natural'. The LangGraph graph output
    is unchanged; this is purely an API-layer rewrite step.
    """
    extra = ""
    if answer_type == "count":
        extra = _try_count_sql(query, sqlite_path)

    clip_lines: list[str] = []
    evidence_limit = int(os.environ.get("AGENT_NATURAL_ANSWER_EVIDENCE_ROWS", "8"))
    for r in rows[:evidence_limit]:
        cam = r.get("camera_id") or r.get("video_id") or ""
        vid_id = r.get("video_id") or ""
        st, et = r.get("start_time"), r.get("end_time")
        text = (
            r.get("event_text_en")
            or r.get("event_summary_en")
            or r.get("event_text")
            or r.get("event_text_cn")
            or r.get("appearance_notes_en")
            or r.get("appearance_notes")
            or ""
        )
        if not text:
            # Build a readable fallback evidence snippet for count/list rows.
            parts: list[str] = []
            obj = (r.get("object_type") or "").strip()
            color = (r.get("object_color") or "").strip()
            zone = (r.get("scene_zone") or "").strip()
            move = (r.get("movement_direction") or "").strip()
            gid = (r.get("global_entity_id") or "").strip()
            tid = r.get("track_id")
            if obj and color and color.lower() != "unknown":
                parts.append(f"{color} {obj}")
            elif obj:
                parts.append(obj)
            if zone:
                parts.append(f"in {zone}")
            if move:
                parts.append(f"moving {move}")
            if gid:
                parts.append(f"entity {gid}")
            elif tid is not None:
                parts.append(f"track {tid}")
            text = ", ".join(parts) if parts else "detected person event"
        if not (cam or text):
            continue
        if st is not None:
            abs_st = _abs_timestamp(vid_id, st)
            abs_et = _abs_timestamp(vid_id, et) if et is not None else None
            time_str = f" [{abs_st}–{abs_et}]" if abs_et else f" [{abs_st}]"
        else:
            time_str = ""
        clip_lines.append(f"  • Camera {cam}{time_str}: {text}")
    clip_ctx = "\n".join(clip_lines) if clip_lines else "  (no detailed clip evidence available)"

    prompt = (
        "You are a helpful surveillance monitoring assistant. "
        "Rewrite the brief answer below into a natural, conversational English response (2–5 sentences). "
        "Use Markdown formatting to highlight key details:\n"
        "- **Bold** camera IDs (e.g. **G326**), wall-clock timestamps (e.g. **16:20:19–16:20:25**), and notable appearance details "
        "(e.g. **dark jacket**, **red suitcase**).\n"
        "- Use a bullet list if there are multiple distinct people or events.\n"
        "- The timestamps in the evidence are already converted to real wall-clock times (HH:MM:SS) — use them directly.\n"
        "- Use specific camera IDs, timestamps, and event descriptions from the evidence. "
        "Do NOT mention 'clip IDs', 'event_id', 'vector distance', or internal system terms. "
        "If the answer is 'No matching clip is expected', say clearly that no relevant footage was found.\n\n"
        f"User question: {query}\n"
        f"Brief answer: {concise_answer}\n"
        + (f"Count data: {extra}\n" if extra else "")
        + f"Evidence:\n{clip_ctx}\n\n"
        "Write a clear, natural Markdown-formatted response:"
    )

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage as HM

        model = (
            os.environ.get("AGENT_NATURAL_ANSWER_MODEL")
            or os.environ.get("OPENAI_CHAT_MODEL")
            or "gpt-5.4"
        )
        kwargs = {
            "model": model,
            "temperature": float(os.environ.get("AGENT_NATURAL_ANSWER_TEMPERATURE", "0")),
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "max_tokens": 320,
        }
        base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        llm = ChatOpenAI(**kwargs)
        resp = llm.invoke([HM(content=prompt)])
        result = str(resp.content or "").strip()
        return result if result else concise_answer
    except Exception:
        return concise_answer


class AgentService:
    def __init__(self) -> None:
        self._graph = None
        self._db: dict[str, str] = {}  # sqlite_path, chroma_path, namespace

    @property
    def graph(self):
        if self._graph is None:
            self._graph = graph_module.create_graph()
        return self._graph

    def select_database(
        self,
        *,
        sqlite_path: str,
        chroma_path: str,
        namespace: str,
    ) -> None:
        """Point the agent at a specific database and invalidate caches.

        Mirrors the env-var setup used in run_agent_eval.py exactly:
          AGENT_SQLITE_DB_PATH, AGENT_CHROMA_PATH, AGENT_CHROMA_NAMESPACE,
          AGENT_CHROMA_COLLECTION (legacy), AGENT_CHROMA_CHILD_COLLECTION,
          AGENT_CHROMA_PARENT_COLLECTION, AGENT_CHROMA_EVENT_COLLECTION,
          AGENT_CHROMA_RETRIEVAL_LEVEL=child
        """
        from agent.db.config import (
            CHROMA_CHILD_SUFFIX,
            CHROMA_EVENT_SUFFIX,
            CHROMA_PARENT_SUFFIX,
        )

        _load_env()  # pick up API keys / model settings from .env

        child_col = f"{namespace}_{CHROMA_CHILD_SUFFIX}"
        parent_col = f"{namespace}_{CHROMA_PARENT_SUFFIX}"
        event_col = f"{namespace}_{CHROMA_EVENT_SUFFIX}"

        os.environ["AGENT_SQLITE_DB_PATH"] = sqlite_path
        os.environ["AGENT_CHROMA_PATH"] = chroma_path
        os.environ["AGENT_CHROMA_NAMESPACE"] = namespace
        # Both old and new collection env vars — matches run_agent_eval.py
        os.environ["AGENT_CHROMA_COLLECTION"] = child_col
        os.environ["AGENT_CHROMA_CHILD_COLLECTION"] = child_col
        os.environ["AGENT_CHROMA_PARENT_COLLECTION"] = parent_col
        os.environ["AGENT_CHROMA_EVENT_COLLECTION"] = event_col
        os.environ["AGENT_CHROMA_RETRIEVAL_LEVEL"] = "child"

        BM25Index.clear_cache()
        self._graph = None
        self._db = {
            "sqlite_path": sqlite_path,
            "chroma_path": chroma_path,
            "namespace": namespace,
        }

    @property
    def db_info(self) -> dict[str, str]:
        return dict(self._db)

    @property
    def database_selected(self) -> bool:
        return bool(self._db)

    def run_query(
        self,
        *,
        query: str,
        thread_id: str | None,
        user_id: str,
        include_rows: bool,
        include_node_trace: bool,
        top_k_rows: int = 8,
        response_style: str = "concise",
        camera_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        question = str(query).strip()
        if not question:
            raise ValueError("query must not be empty")
        if not self.database_selected:
            raise ValueError("Select a database before querying (call POST /api/v1/database/select)")

        count_answer = _try_count_sql(question, self._db.get("sqlite_path", ""))
        if count_answer and re.search(r"\b(how many|count|number of)\b", question, re.IGNORECASE):
            count_rows = _count_evidence_rows(
                question,
                self._db.get("sqlite_path", ""),
                limit=max(int(top_k_rows or 8), 8),
            )
            display_answer = count_answer
            if response_style == "natural":
                display_answer = _rewrite_natural(
                    query=question,
                    concise_answer=count_answer,
                    rows=count_rows,
                    answer_type="count",
                    sqlite_path=self._db.get("sqlite_path", ""),
                )
            return {
                "query": question,
                "answer": display_answer,
                "final_answer": count_answer,
                "answer_type": "count",
                "node_trace": ["sqlite_count"] if include_node_trace else [],
                "elapsed_ms": 0.0,
                "verifier_result": {
                    "decision": "skipped",
                    "confidence": 0.0,
                    "reason": "answer_type=count",
                    "mode": "skipped",
                },
                "rows": count_rows[:top_k_rows] if include_rows else [],
            }

        resolved_thread_id = thread_id or f"mevid-{uuid.uuid4().hex[:12]}"
        config = {"configurable": {"thread_id": resolved_thread_id, "user_id": user_id}}
        messages: list = [HumanMessage(content=question)]
        # Build initial state — camera_filter constrains retrieval to specific cameras
        initial_state: dict[str, Any] = {"messages": messages}
        if camera_ids:
            initial_state["camera_filter"] = [c.strip().upper() for c in camera_ids if c.strip()]

        last_chunk: dict[str, Any] = {}
        node_trace: list[str] = []
        t0 = time.perf_counter()
        for chunk in self.graph.stream(initial_state, config, stream_mode="values"):
            last_chunk = chunk
            current_node = chunk.get("current_node")
            if include_node_trace and current_node and (not node_trace or node_trace[-1] != current_node):
                node_trace.append(str(current_node))
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        answer = _strip_sources(last_chunk.get("final_answer"))
        answer_type = str(last_chunk.get("answer_type") or "").strip()
        if count_answer and (
            answer_type == "count"
            or re.search(r"\b(how many|count|number of)\b", question, re.IGNORECASE)
        ):
            answer = count_answer
        # Mirror _final_rows from run_agent_eval.py
        all_rows: list[dict] = []
        for key in ("rerank_result", "merged_result", "hybrid_result", "sql_result"):
            candidate = last_chunk.get(key)
            if isinstance(candidate, list):
                all_rows = [r for r in candidate if isinstance(r, dict)]
                break

        # Natural language post-processing — only for UI requests, never for eval
        if response_style == "natural":
            answer = _rewrite_natural(
                query=question,
                concise_answer=answer,
                rows=all_rows,
                answer_type=answer_type,
                sqlite_path=self._db.get("sqlite_path", ""),
            )

        return {
            "query": question,
            "answer": answer,
            "final_answer": str(last_chunk.get("final_answer") or "").strip(),
            "answer_type": answer_type,
            "node_trace": node_trace,
            "elapsed_ms": elapsed_ms,
            "verifier_result": last_chunk.get("verifier_result") if isinstance(last_chunk.get("verifier_result"), dict) else {},
            "rows": all_rows[:top_k_rows] if include_rows else [],
        }


agent_service = AgentService()
