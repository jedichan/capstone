import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DEFAULT_SQLITE_PATH
from .schema import (
    CREATE_TABLE_SQL,
    FTS5_CREATE_SQL_LIST,
    FTS5_REBUILD_SQL,
    INDEX_SQL_LIST,
    INSERT_COLUMNS,
)


logger = logging.getLogger(__name__)
AGENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INIT_PROMPT_MD = AGENT_DIR / "init" / "agent_init_prompt.md"
DEFAULT_INIT_PROMPT_JSON = AGENT_DIR / "init" / "agent_init_profile.json"


class DatabaseBuildError(RuntimeError):
    pass


@dataclass
class SQLiteBuildConfig:
    db_path: Path = DEFAULT_SQLITE_PATH
    reset_existing: bool = False
    generate_init_prompt: bool = True
    init_prompt_md_path: Path = DEFAULT_INIT_PROMPT_MD
    init_prompt_json_path: Path = DEFAULT_INIT_PROMPT_JSON
    schema_version: str = "v2_hybrid_sql_vector_complementary"


class SQLiteDatabaseBuilder:
    def __init__(self, config: SQLiteBuildConfig) -> None:
        self.config = config
        self.db_path = config.db_path

    def build(self, seed_files: list[Path] | None = None) -> dict[str, Any]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.reset_existing and self.db_path.exists():
            logger.info("Reset existing db: %s", self.db_path)
            self.db_path.unlink()

        profile = self._new_init_profile()
        fts_state: dict[str, Any] = {"enabled": False, "row_count": 0}
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._configure_connection(conn)
                self._create_schema(conn)
                self._create_indexes(conn)
                fts_state["enabled"] = self._create_fts5(conn)
                inserted = 0
                if seed_files:
                    for seed_file in seed_files:
                        rows, file_profile = self._load_seed_rows(seed_file)
                        self._merge_profile(profile, file_profile)
                        inserted += self._insert_rows(conn, rows)
                if fts_state["enabled"] and inserted > 0:
                    fts_state["row_count"] = self._rebuild_fts5(conn)
                conn.commit()
        except Exception as exc:
            logger.exception("Failed to build sqlite db")
            raise DatabaseBuildError(f"Build failed for {self.db_path}: {exc}") from exc

        output = {
            "db_path": str(self.db_path),
            "seed_files": [str(x) for x in (seed_files or [])],
            "inserted_rows": inserted,
            "fts5_enabled": fts_state["enabled"],
            "fts5_row_count": fts_state["row_count"],
        }
        if self.config.generate_init_prompt:
            artifacts = self._write_init_prompt_artifacts(profile)
            output.update(artifacts)
        return output

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.execute(CREATE_TABLE_SQL)

    @staticmethod
    def _create_indexes(conn: sqlite3.Connection) -> None:
        for sql in INDEX_SQL_LIST:
            conn.execute(sql)

    @staticmethod
    def _create_fts5(conn: sqlite3.Connection) -> bool:
        """Create the ``episodic_events_fts`` virtual table + sync triggers.

        Returns ``True`` if the index is available after the call. Falls back
        gracefully (returns ``False``) when the host SQLite was compiled
        without FTS5 -- callers should detect this by inspecting
        ``sqlite_master`` and route around the lexical index.
        """

        try:
            for stmt in FTS5_CREATE_SQL_LIST:
                conn.execute(stmt)
            return True
        except sqlite3.OperationalError as exc:
            logger.warning(
                "FTS5 not available on this SQLite build (%s); falling back to LIKE-only text search.",
                exc,
            )
            return False

    @staticmethod
    def _rebuild_fts5(conn: sqlite3.Connection) -> int:
        """Reseed the FTS5 index from the underlying table after bulk inserts.

        Triggers keep the index in sync incrementally, but the documented
        ``rebuild`` call after large seed insertions is a belt-and-braces
        guarantee that the index matches the data.
        """

        try:
            conn.execute(FTS5_REBUILD_SQL)
            cur = conn.execute("SELECT count(*) FROM episodic_events_fts;")
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 rebuild failed (%s); index may be stale.", exc)
            return 0

    def _load_seed_rows(self, seed_file: Path) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
        if not seed_file.exists():
            raise DatabaseBuildError(f"Seed file not found: {seed_file}")

        payload = json.loads(seed_file.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        profile = SQLiteDatabaseBuilder._new_init_profile()

        def append_event(event: dict[str, Any], fallback_video_id: str | None = None) -> None:
            if not isinstance(event, dict):
                return
            SQLiteDatabaseBuilder._collect_prompt_tokens(profile, event)
            start_bbox = event.get("start_bbox_xyxy") or [None, None, None, None]
            end_bbox = event.get("end_bbox_xyxy") or [None, None, None, None]
            start_time = event.get("start_time")
            end_time = event.get("end_time")
            duration_sec = None
            if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                duration_sec = float(end_time) - float(start_time)

            keywords = event.get("keywords") or []
            if isinstance(keywords, str):
                keywords_list = self._tokenize_keywords(keywords)
            elif isinstance(keywords, list):
                keywords_list = [str(x).strip() for x in keywords if str(x).strip()]
            else:
                keywords_list = []
            if not keywords_list:
                cls = event.get("class_name") or ""
                etype = event.get("event_type") or ""
                keywords_list = [k for k in [cls, etype] if k]

            appearance_notes = event.get("appearance_notes_en") or event.get("appearance_notes")
            event_text = (
                event.get("event_text_en")
                or event.get("event_text_cn")
                or event.get("event_text")
                or event.get("description_for_llm")
            )
            event_summary = (
                event.get("event_summary_en")
                or event.get("event_summary_cn")
                or event.get("event_summary")
                or event_text
            )
            is_stationary = 1 if "stationary" in str(appearance_notes or "").lower() else 0
            video_id = event.get("video_id") or fallback_video_id
            if not video_id:
                video_id = event.get("camera_id") or "multi_camera"

            entity_hint = event.get("entity_hint")
            if not entity_hint:
                cls_hint = event.get("class_name") or "object"
                tid = event.get("track_id")
                entity_hint = f"{cls_hint}_{tid}" if tid is not None else None

            # --- Global entity id derivation for multi-camera tracking ---
            # When entity_hint looks like a global/cross-camera entity
            # (e.g. "person_global_1"), populate global_entity_id so the
            # cross-camera index can group the same real-world object across
            # different camera views.
            global_entity_id = event.get("global_entity_id")
            if not global_entity_id and entity_hint:
                global_entity_id = SQLiteDatabaseBuilder._derive_global_entity_id(entity_hint)

            object_type = event.get("object_type") or event.get("class_name")
            cross_camera_ids_raw = event.get("cross_camera_ids")
            if isinstance(cross_camera_ids_raw, list):
                cross_camera_ids = [str(c).strip() for c in cross_camera_ids_raw if str(c).strip()]
            elif isinstance(cross_camera_ids_raw, str) and cross_camera_ids_raw.strip():
                cross_camera_ids = [part.strip() for part in cross_camera_ids_raw.split(",") if part.strip()]
            else:
                cross_camera_ids = []

            rows.append(
                {
                    "video_id": video_id,
                    "camera_id": event.get("camera_id"),
                    "global_entity_id": global_entity_id,
                    "track_id": str(event.get("track_id")) if event.get("track_id") is not None else None,
                    "entity_hint": entity_hint,
                    "clip_start_sec": event.get("clip_start_sec"),
                    "clip_end_sec": event.get("clip_end_sec"),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_sec": duration_sec,
                    "object_type": object_type,
                    "object_color_en": event.get("object_color_en") or event.get("object_color"),
                    "scene_zone_en": event.get("scene_zone_en") or event.get("scene_zone"),
                    "motion_level": event.get("motion_level"),
                    "event_type": event.get("event_type"),
                    "is_stationary": is_stationary,
                    "start_bbox_x1": start_bbox[0] if len(start_bbox) > 0 else None,
                    "start_bbox_y1": start_bbox[1] if len(start_bbox) > 1 else None,
                    "start_bbox_x2": start_bbox[2] if len(start_bbox) > 2 else None,
                    "start_bbox_y2": start_bbox[3] if len(start_bbox) > 3 else None,
                    "end_bbox_x1": end_bbox[0] if len(end_bbox) > 0 else None,
                    "end_bbox_y1": end_bbox[1] if len(end_bbox) > 1 else None,
                    "end_bbox_x2": end_bbox[2] if len(end_bbox) > 2 else None,
                    "end_bbox_y2": end_bbox[3] if len(end_bbox) > 3 else None,
                    "appearance_notes_en": appearance_notes,
                    "event_text_en": event_text,
                    "event_summary_en": event_summary,
                    "keywords_json": json.dumps(keywords_list, ensure_ascii=False),
                    "semantic_tags_json": json.dumps({"keywords": keywords_list}, ensure_ascii=False),
                    "vector_doc_text": self._build_vector_doc_text(
                        video_id=str(video_id),
                        camera_id=str(event.get("camera_id") or ""),
                        global_entity_id=str(global_entity_id or ""),
                        cross_camera_ids=cross_camera_ids,
                        start_time=start_time,
                        end_time=end_time,
                        object_type=str(object_type or ""),
                        object_color=str(event.get("object_color_en") or event.get("object_color") or ""),
                        appearance_notes=str(appearance_notes or ""),
                        event_text=str(event_text or event_summary or ""),
                    ),
                    "vector_ref_id": f"{video_id}:{entity_hint or event.get('track_id') or 'na'}:{start_time}:{end_time}",
                    "source_format": "events_vector_flat",
                    "schema_version": self.config.schema_version,
                    "metadata_json": json.dumps(event, ensure_ascii=False),
                }
            )

        if isinstance(payload, dict):
            # --- Camera-grouped multi-camera format ---
            # { "G328": {"video_id": "...", "events": [...]}, "G339": {...} }
            if SQLiteDatabaseBuilder._is_camera_grouped(payload):
                for __camera_id, camera_entry in payload.items():
                    if not isinstance(camera_entry, dict):
                        continue
                    fallback = str(camera_entry.get("video_id", "")).strip() or str(__camera_id)
                    for event in camera_entry.get("events") or []:
                        if isinstance(event, dict):
                            append_event(event, fallback)
            elif isinstance(payload.get("events"), list):
                fallback = str(payload.get("video_id", "")).strip() or None
                for event in payload["events"]:
                    append_event(event, fallback)
            else:
                append_event(payload)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and isinstance(item.get("events"), list):
                    fallback = str(item.get("video_id", "")).strip() or None
                    for event in item["events"]:
                        append_event(event, fallback)
                elif isinstance(item, dict):
                    append_event(item)
        else:
            raise DatabaseBuildError(f"Unsupported seed JSON format: {seed_file}")

        # Post-pass: enrich vector text with cross-camera trajectory summary for
        # rows that share the same global entity id.
        self._augment_rows_with_global_entity_context(rows)
        return rows, profile

    @staticmethod
    def _build_vector_doc_text(
        *,
        video_id: str,
        camera_id: str,
        global_entity_id: str,
        cross_camera_ids: list[str],
        start_time: Any,
        end_time: Any,
        object_type: str,
        object_color: str,
        appearance_notes: str,
        event_text: str,
    ) -> str:
        parts: list[str] = []
        if event_text:
            parts.append(event_text.strip())
        parts.append(f"video_id={video_id}")
        if camera_id:
            parts.append(f"camera_id={camera_id}")
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            parts.append(f"time={float(start_time):.1f}-{float(end_time):.1f}s")
        if object_type:
            parts.append(f"object={object_type}")
        if object_color:
            parts.append(f"color={object_color}")
        if appearance_notes:
            parts.append(f"appearance={appearance_notes}")
        if global_entity_id:
            parts.append(f"global_entity_id={global_entity_id}")
        if cross_camera_ids:
            parts.append("cross_camera_ids=" + ",".join(cross_camera_ids))
        return " | ".join([p for p in parts if p])

    @staticmethod
    def _augment_rows_with_global_entity_context(rows: list[dict[str, Any]]) -> None:
        # Build per-global-entity camera/time summary
        by_ge: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            ge = str(row.get("global_entity_id") or "").strip()
            if ge:
                by_ge.setdefault(ge, []).append(row)

        ge_summary: dict[str, str] = {}
        for ge, ge_rows in by_ge.items():
            cameras: list[str] = []
            per_cam_time: dict[str, tuple[float, float]] = {}
            for r in ge_rows:
                cam = str(r.get("camera_id") or "").strip()
                if cam and cam not in cameras:
                    cameras.append(cam)
                st = r.get("start_time")
                et = r.get("end_time")
                if cam and isinstance(st, (int, float)) and isinstance(et, (int, float)):
                    prev = per_cam_time.get(cam)
                    if prev is None:
                        per_cam_time[cam] = (float(st), float(et))
                    else:
                        per_cam_time[cam] = (min(prev[0], float(st)), max(prev[1], float(et)))
            cam_parts: list[str] = []
            for cam, (st, et) in per_cam_time.items():
                cam_parts.append(f"{cam}:{st:.1f}-{et:.1f}s")
            summary = f"global_entity {ge} trajectory cameras={','.join(cameras)}"
            if cam_parts:
                summary += " timeline=" + "; ".join(cam_parts)
            ge_summary[ge] = summary

        # Append summary into vector_doc_text to strengthen cross-camera retrieval
        for row in rows:
            ge = str(row.get("global_entity_id") or "").strip()
            if not ge:
                continue
            summary = ge_summary.get(ge, "")
            if not summary:
                continue
            cur = str(row.get("vector_doc_text") or "").strip()
            if summary in cur:
                continue
            row["vector_doc_text"] = f"{cur} | {summary}" if cur else summary

    @staticmethod
    def _insert_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        columns_sql = ", ".join(INSERT_COLUMNS)
        placeholders = ", ".join(["?"] * len(INSERT_COLUMNS))
        conn.executemany(
            f"INSERT INTO episodic_events ({columns_sql}) VALUES ({placeholders})",
            [
                tuple(row.get(col) for col in INSERT_COLUMNS)
                for row in rows
            ],
        )
        return len(rows)

    @staticmethod
    def _new_init_profile() -> dict[str, set[str]]:
        return {
            "keywords": set(),
            "object_types": set(),
            "object_colors": set(),
        }

    @staticmethod
    def _tokenize_keywords(raw_keywords: Any) -> list[str]:
        if isinstance(raw_keywords, list):
            return [str(item).strip() for item in raw_keywords if str(item).strip()]
        if isinstance(raw_keywords, str):
            text = raw_keywords.replace("|", ",").replace(";", ",")
            return [part.strip() for part in text.split(",") if part.strip()]
        return []

    # --- Multi-camera format detection & global entity derivation ---

    _GLOBAL_ENTITY_PATTERNS: tuple[str, ...] = (
        "_global_",       # person_global_1, vehicle_global_2, etc.
        "global_entity_", # alternative naming
    )

    @staticmethod
    def _is_camera_grouped(payload: dict[str, Any]) -> bool:
        """Detect multi-camera format: {camera_id: {video_id, events}, ...}.

        Heuristic: payload is a dict, does NOT have a direct "events" key,
        and at least one value is a dict that contains an "events" list.
        """
        if not isinstance(payload, dict):
            return False
        if "events" in payload:
            return False  # single-video format: {"video_id": ..., "events": [...]}
        # Check that at least one value looks like a camera entry
        for val in payload.values():
            if isinstance(val, dict) and isinstance(val.get("events"), list):
                return True
        return False

    @staticmethod
    def _derive_global_entity_id(entity_hint: str) -> str | None:
        """Derive ``global_entity_id`` from ``entity_hint`` when it matches a
        known global/cross-camera naming pattern (e.g. ``person_global_1``).

        Returns the entity_hint as-is when it looks like a global entity;
        returns None otherwise so the caller falls back to per-camera tracking.
        """
        if not entity_hint:
            return None
        hint_lower = entity_hint.lower()
        for pattern in SQLiteDatabaseBuilder._GLOBAL_ENTITY_PATTERNS:
            if pattern in hint_lower:
                return entity_hint
        return None

    @staticmethod
    def _collect_prompt_tokens(profile: dict[str, set[str]], event: dict[str, Any]) -> None:
        object_type = str(event.get("object_type") or event.get("class_name") or "").strip().lower()
        if object_type:
            profile["object_types"].add(object_type)

        color = str(event.get("object_color_en") or event.get("object_color") or "").strip().lower()
        if color:
            profile["object_colors"].add(color)

        for kw in SQLiteDatabaseBuilder._tokenize_keywords(event.get("keywords")):
            profile["keywords"].add(kw.lower())
        if not event.get("keywords"):
            cls = str(event.get("class_name") or "").strip().lower()
            etype = str(event.get("event_type") or "").strip().lower()
            for kw in (cls, etype):
                if kw:
                    profile["keywords"].add(kw)

    @staticmethod
    def _merge_profile(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
        for key in ("keywords", "object_types", "object_colors"):
            target[key].update(source.get(key, set()))

    def _write_init_prompt_artifacts(self, profile: dict[str, set[str]]) -> dict[str, Any]:
        md_path = self.config.init_prompt_md_path
        json_path = self.config.init_prompt_json_path
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        object_types = sorted(profile["object_types"])
        object_colors = sorted(profile["object_colors"])
        keywords = sorted(profile["keywords"])

        prompt_payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "db_path": str(self.db_path),
            "object_types": object_types,
            "object_colors": object_colors,
            "keywords": keywords,
            "counts": {
                "object_types": len(object_types),
                "object_colors": len(object_colors),
                "keywords": len(keywords),
            },
        }
        json_path.write_text(json.dumps(prompt_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        md_content = [
            "# Agent Initialization Prompt\n\n",
            "## Purpose\n",
            "- This prompt is generated during the `seed json -> sqlite` build process.\n",
            "- It provides a quick pre-retrieval judgment context for router/sub-agent.\n\n",
            "## Prompt Template\n",
            "Use the following context before retrieval:\n\n",
            "```text\n",
            "You are the retrieval pre-check module.\n",
            "Known object types: " + (", ".join(object_types) if object_types else "N/A") + "\n",
            "Known object colors: " + (", ".join(object_colors) if object_colors else "N/A") + "\n",
            "Known keywords: " + (", ".join(keywords) if keywords else "N/A") + "\n\n",
            "Quick judgment rules:\n",
            "1. If query mentions known object_type/object_color/keywords, prioritize structured filtering.\n",
            "2. If query terms are mostly outside the known vocabulary, use semantic/hybrid retrieval.\n",
            "3. Preserve original query text; do not fabricate unseen labels.\n",
            "```\n\n",
            "## Metadata\n",
            f"- Generated at: `{prompt_payload['generated_at']}`\n",
            f"- Source db: `{self.db_path}`\n",
            f"- object_types count: `{len(object_types)}`\n",
            f"- object_colors count: `{len(object_colors)}`\n",
            f"- keywords count: `{len(keywords)}`\n",
        ]
        md_path.write_text("".join(md_content), encoding="utf-8")

        return {
            "init_prompt_md_path": str(md_path),
            "init_prompt_json_path": str(json_path),
            "init_prompt_counts": prompt_payload["counts"],
        }
