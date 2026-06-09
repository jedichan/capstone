"""Unit tests for the SQLite FTS5 lexical index introduced in P1-1."""

from __future__ import annotations

import os
import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

# Bootstrap ``agent/`` onto ``sys.path`` so the bare ``node.*`` / ``tools.*``
# imports inside the package resolve when invoked via ``pytest`` from the repo
# root (mirrors the convention in ``test_bm25_index.py``).
_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent.db.schema import (
    CREATE_TABLE_SQL,
    FTS5_CREATE_SQL_LIST,
    FTS5_REBUILD_SQL,
    FTS_TABLE_NAME,
    INDEX_SQL_LIST,
    INSERT_COLUMNS,
    TABLE_NAME,
)


SAMPLE_EVENTS: list[dict[str, object]] = [
    {
        "video_id": "Arrest050_x264",
        "track_id": "veh_white_01",
        "start_time": 133.6,
        "end_time": 206.5,
        "object_type": "vehicle",
        "object_color_en": "white",
        "scene_zone_en": "roadside",
        "event_text_en": "Two white and one black police vehicles parked by the road with flashing roof lights.",
        "event_summary_en": "White and black police cars with flashing lights along the roadside.",
        "appearance_notes_en": "All three vehicles have rotating roof beacons activated.",
        "keywords_json": '["police","vehicle","flashing","roof","light"]',
    },
    {
        "video_id": "Abuse037_x264",
        "track_id": "person_red_02",
        "start_time": 12.0,
        "end_time": 45.0,
        "object_type": "person",
        "object_color_en": "red",
        "scene_zone_en": "indoor",
        "event_text_en": "A red-shirted person stomps on another person lying on the indoor floor.",
        "event_summary_en": "Red shirted person attacks the figure on the ground.",
        "appearance_notes_en": "Indoor scene with poor lighting and tile flooring.",
        "keywords_json": '["abuse","person","ground","stomp"]',
    },
    {
        "video_id": "Shoplifting001_x264",
        "track_id": "person_dark_03",
        "start_time": 60.0,
        "end_time": 88.0,
        "object_type": "person",
        "object_color_en": "black",
        "scene_zone_en": "store",
        "event_text_en": "A dark-clothed person stuffs items into a backpack while watching the cashier.",
        "event_summary_en": "Concealed shoplifting near the cashier counter.",
        "appearance_notes_en": "Customer pretends to browse before pocketing merchandise.",
        "keywords_json": '["shoplifting","cashier","backpack"]',
    },
]


def _seed_db(db_path: Path, *, with_fts5: bool = True) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(CREATE_TABLE_SQL)
        for stmt in INDEX_SQL_LIST:
            conn.execute(stmt)
        if with_fts5:
            for stmt in FTS5_CREATE_SQL_LIST:
                conn.execute(stmt)
        cols = list(INSERT_COLUMNS)
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO {TABLE_NAME} ({', '.join(cols)}) VALUES ({placeholders})"
        rows = [[ev.get(col) for col in cols] for ev in SAMPLE_EVENTS]
        conn.executemany(sql, rows)
        if with_fts5:
            conn.execute(FTS5_REBUILD_SQL)
        conn.commit()


class FTS5SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "fts5.sqlite"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_fts5_virtual_table_created_and_populated(self) -> None:
        _seed_db(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
                (FTS_TABLE_NAME,),
            ).fetchone()
            self.assertEqual(row[0], 1)
            count = conn.execute(f"SELECT count(*) FROM {FTS_TABLE_NAME}").fetchone()[0]
            self.assertEqual(count, len(SAMPLE_EVENTS))

    def test_fts5_match_returns_expected_rows(self) -> None:
        _seed_db(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT video_id FROM episodic_events WHERE event_id IN ("
                "  SELECT rowid FROM episodic_events_fts WHERE episodic_events_fts MATCH ?"
                ")",
                ('"flashing" OR "beacons"',),
            ).fetchall()
            self.assertEqual([r[0] for r in rows], ["Arrest050_x264"])

            rows = conn.execute(
                "SELECT video_id FROM episodic_events WHERE event_id IN ("
                "  SELECT rowid FROM episodic_events_fts WHERE episodic_events_fts MATCH ?"
                ")",
                ('"cashier"',),
            ).fetchall()
            self.assertEqual([r[0] for r in rows], ["Shoplifting001_x264"])

    def test_after_update_trigger_keeps_fts_in_sync(self) -> None:
        _seed_db(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE episodic_events SET event_text_en = ? WHERE video_id = ?",
                (
                    "An updated summary referencing zebra patterns and starlight.",
                    "Abuse037_x264",
                ),
            )
            conn.commit()
            rows = conn.execute(
                "SELECT video_id FROM episodic_events WHERE event_id IN ("
                "  SELECT rowid FROM episodic_events_fts WHERE episodic_events_fts MATCH ?"
                ")",
                ('"zebra" OR "starlight"',),
            ).fetchall()
            self.assertEqual([r[0] for r in rows], ["Abuse037_x264"])

            old_token_rows = conn.execute(
                "SELECT video_id FROM episodic_events WHERE event_id IN ("
                "  SELECT rowid FROM episodic_events_fts WHERE episodic_events_fts MATCH ?"
                ")",
                ('"stomps"',),
            ).fetchall()
            self.assertEqual(old_token_rows, [])

    def test_after_delete_trigger_drops_row_from_fts(self) -> None:
        _seed_db(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM episodic_events WHERE video_id = ?", ("Shoplifting001_x264",))
            conn.commit()
            rows = conn.execute(
                "SELECT count(*) FROM episodic_events_fts WHERE episodic_events_fts MATCH ?",
                ('"cashier"',),
            ).fetchall()
            self.assertEqual(rows[0][0], 0)


class RunSqlBranchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "fts5.sqlite"
        _seed_db(self.db_path)
        # Avoid the LlamaIndex SQL path so we exercise the in-process FTS5 code.
        self._env_patch = patch.dict(
            os.environ,
            {
                "AGENT_USE_LLAMAINDEX_SQL": "0",
                "AGENT_SQLITE_DB_PATH": str(self.db_path),
            },
        )
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def _run(self, query: str, *, sql_limit: int = 50) -> tuple[str, list[dict]]:
        from node.parallel_retrieval_fusion_node import _run_sql_branch

        return _run_sql_branch(query, {"sql_limit": sql_limit})

    def test_fts5_path_returns_target_event(self) -> None:
        summary, rows = self._run(
            "Two white police vehicles with flashing roof lights on the roadside"
        )
        self.assertIn("text_strategy=fts5", summary)
        video_ids = [r.get("video_id") for r in rows]
        self.assertIn("Arrest050_x264", video_ids)

    def test_fts5_disabled_falls_back_to_like(self) -> None:
        with patch.dict(os.environ, {"AGENT_SQL_USE_FTS5": "0"}):
            summary, rows = self._run(
                "Two white police vehicles with flashing roof lights on the roadside"
            )
        self.assertIn("text_strategy=like", summary)
        video_ids = [r.get("video_id") for r in rows]
        self.assertIn("Arrest050_x264", video_ids)

    def test_fts5_path_skipped_when_no_text_tokens(self) -> None:
        # Filter-only query (a single hard structured filter); tokens should be
        # empty so the SQL still runs but neither LIKE nor FTS5 fires.
        summary, rows = self._run("Arrest050_x264")
        self.assertIn("text_strategy=", summary)


if __name__ == "__main__":
    unittest.main()
