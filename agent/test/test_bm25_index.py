"""Unit tests for the full-corpus BM25 index introduced in P1-2."""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# Allow bare ``node.*`` / ``tools.*`` imports inside the agent package, mirroring
# the runtime convention (everything under ``agent/`` is invoked with
# ``agent/`` on ``sys.path`` -- e.g. ``cd agent && python ...``).
_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent.db.schema import CREATE_TABLE_SQL, INDEX_SQL_LIST, INSERT_COLUMNS, TABLE_NAME
from agent.tools.bm25_index import BM25Index, reciprocal_rank_fuse


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
    },
    {
        "video_id": "Arrest050_x264",
        "track_id": "officer_blue_04",
        "start_time": 215.0,
        "end_time": 290.0,
        "object_type": "officer",
        "object_color_en": "blue",
        "scene_zone_en": "roadside",
        "event_text_en": "Two officers handcuff a suspect next to a parked patrol car on the roadside.",
        "event_summary_en": "Roadside arrest with patrol vehicle visible.",
        "appearance_notes_en": "Officers in blue uniforms work together to restrain the suspect.",
    },
    {
        "video_id": "Vandalism045_x264",
        "track_id": "person_grey_05",
        "start_time": 5.0,
        "end_time": 28.0,
        "object_type": "person",
        "object_color_en": "grey",
        "scene_zone_en": "parking_lot",
        "event_text_en": "A person smashes a parked sedan window with a metal bar in the parking lot.",
        "event_summary_en": "Vandalism: parked sedan window smashed.",
        "appearance_notes_en": "Lone individual in grey jacket performs the act.",
    },
]


def _seed_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(CREATE_TABLE_SQL)
        for stmt in INDEX_SQL_LIST:
            conn.execute(stmt)
        cols = list(INSERT_COLUMNS)
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO {TABLE_NAME} ({', '.join(cols)}) VALUES ({placeholders})"
        rows = []
        for ev in SAMPLE_EVENTS:
            row = [ev.get(col) for col in cols]
            rows.append(row)
        conn.executemany(sql, rows)
        conn.commit()


class BM25IndexTests(unittest.TestCase):
    def setUp(self) -> None:
        BM25Index.clear_cache()
        self._tmpdir = TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "events.sqlite"
        _seed_db(self.db_path)
        self.index = BM25Index(self.db_path)

    def tearDown(self) -> None:
        BM25Index.clear_cache()
        self._tmpdir.cleanup()

    def test_stats_reports_corpus_dimensions(self) -> None:
        stats = self.index.stats()
        self.assertEqual(stats["doc_count"], len(SAMPLE_EVENTS))
        self.assertGreater(stats["vocab_size"], 20)
        self.assertGreater(stats["avgdl"], 5.0)

    def test_lexical_query_ranks_distinctive_event_first(self) -> None:
        # ``flashing roof lights`` should pin the patrol-cars event to the top
        # because the vocabulary appears nowhere else in the corpus.
        rows = self.index.search("flashing roof lights on parked vehicles", top_k=3)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["video_id"], "Arrest050_x264")
        self.assertEqual(rows[0]["track_id"], "veh_white_01")
        self.assertGreater(rows[0]["_bm25"], 0.0)
        self.assertEqual(rows[0]["_source_type"], "bm25")

    def test_short_unique_token_still_recoverable(self) -> None:
        rows = self.index.search("shoplifting cashier counter", top_k=3)
        self.assertEqual(rows[0]["video_id"], "Shoplifting001_x264")

    def test_filter_prunes_corpus_before_scoring(self) -> None:
        rows = self.index.search(
            "person on the ground",
            top_k=5,
            filters={"video_id": "Abuse037_x264"},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["video_id"], "Abuse037_x264")

    def test_filter_with_no_match_returns_empty(self) -> None:
        rows = self.index.search(
            "person",
            top_k=5,
            filters={"video_id": "DoesNotExist"},
        )
        self.assertEqual(rows, [])

    def test_query_of_only_stopwords_returns_empty(self) -> None:
        rows = self.index.search("the and from with this", top_k=3)
        self.assertEqual(rows, [])

    def test_cache_reuses_index_until_mtime_changes(self) -> None:
        first = self.index._load_index()
        second = self.index._load_index()
        self.assertIs(first, second)

        # Insert a new row -> mtime bumps -> cache key changes -> rebuild.
        with sqlite3.connect(self.db_path) as conn:
            cols = list(INSERT_COLUMNS)
            placeholders = ", ".join(["?"] * len(cols))
            sql = f"INSERT INTO {TABLE_NAME} ({', '.join(cols)}) VALUES ({placeholders})"
            extra = {
                "video_id": "Robbery099_x264",
                "track_id": "person_purple_99",
                "start_time": 10.0,
                "end_time": 22.0,
                "object_type": "person",
                "object_color_en": "purple",
                "scene_zone_en": "atm",
                "event_text_en": "Masked robber threatens an ATM customer with a knife.",
                "event_summary_en": "Armed ATM robbery.",
                "appearance_notes_en": "Black mask, purple hoodie.",
            }
            conn.execute(sql, [extra.get(col) for col in cols])
            conn.commit()

        third = self.index._load_index()
        self.assertIsNot(first, third)
        self.assertEqual(len(third["documents"]), len(SAMPLE_EVENTS) + 1)


class ReciprocalRankFuseTests(unittest.TestCase):
    def test_intersection_outranks_singletons(self) -> None:
        vector = [{"event_id": 1}, {"event_id": 2}, {"event_id": 3}]
        bm25 = [{"event_id": 3}, {"event_id": 1}, {"event_id": 4}]
        fused = reciprocal_rank_fuse([vector, bm25], top_k=4)
        ids = [row["event_id"] for row in fused]
        # ``1`` and ``3`` appear in both lists at rank 1 / rank 3 and rank 2 /
        # rank 1 respectively -> they fuse above the singletons.
        self.assertEqual(ids[:2], [1, 3])
        self.assertIn(2, ids)
        self.assertIn(4, ids)

    def test_top_k_truncates(self) -> None:
        vector = [{"event_id": i} for i in range(1, 11)]
        bm25 = [{"event_id": i} for i in range(11, 21)]
        fused = reciprocal_rank_fuse([vector, bm25], top_k=5)
        self.assertEqual(len(fused), 5)
        self.assertEqual(fused[0]["_fused_rank"], 1)

    def test_missing_id_field_falls_back_to_object_identity(self) -> None:
        a = {"event_id": None, "video_id": "a"}
        b = {"event_id": None, "video_id": "b"}
        fused = reciprocal_rank_fuse([[a], [b]], top_k=2)
        ids = sorted(row["video_id"] for row in fused)
        self.assertEqual(ids, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
