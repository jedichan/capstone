import unittest
from unittest.mock import patch
from pathlib import Path

from agent.db.config import get_graph_sqlite_db_path
from agent.tools.sql_debug_utils import build_text2sql_plan, run_guided_sql_candidate, run_relaxed_sql_fallback


ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_SQLITE = ROOT_DIR / "agent" / "test" / "generated" / "ragas_eval_latest" / "runtime" / "eval_subset.sqlite"


class PureSqlFallbackTests(unittest.TestCase):
    def test_text2sql_plan_keeps_multi_entity_case_soft(self) -> None:
        plan = build_text2sql_plan(
            user_query="Is there a clip of a car running over a black dog on the road?",
            db_path=EVAL_SQLITE,
        )
        hard_fields = [item["field"] for item in plan["hard_filters"]]
        self.assertIn("scene_zone_en", hard_fields)
        self.assertNotIn("object_color_en", hard_fields)
        self.assertNotIn("object_type", hard_fields)

    def test_guided_sql_candidate_ranks_part1_0002_to_top(self) -> None:
        plan = build_text2sql_plan(
            user_query="Is there a clip of a car running over a black dog on the road?",
            db_path=EVAL_SQLITE,
        )
        result = run_guided_sql_candidate(
            user_query="Is there a clip of a car running over a black dog on the road?",
            plan=plan,
            limit=5,
            db_path=EVAL_SQLITE,
        )
        self.assertGreater(len(result["rows"]), 0)
        self.assertEqual(result["rows"][0].get("video_id"), "Abuse037_x264")

    def test_recovers_part1_0002_when_strict_sql_returns_empty(self) -> None:
        result = run_relaxed_sql_fallback(
            user_query="Is there a clip of a car running over a black dog on the road?",
            limit=10,
            db_path=EVAL_SQLITE,
        )
        self.assertGreater(len(result["rows"]), 0)
        self.assertTrue(any(row.get("video_id") == "Abuse037_x264" for row in result["rows"]))

    def test_recovers_part1_0011_when_strict_sql_returns_empty(self) -> None:
        result = run_relaxed_sql_fallback(
            user_query="Is there a clip of a caregiver repeatedly hitting a white-haired elderly person on the head while they sit on a sofa?",
            limit=10,
            db_path=EVAL_SQLITE,
        )
        self.assertGreater(len(result["rows"]), 0)
        self.assertTrue(any(row.get("video_id") == "Abuse040_x264" for row in result["rows"]))

    def test_text2sql_plan_removes_junk_terms_for_person_query(self) -> None:
        plan = build_text2sql_plan(
            user_query="Did you see any person in the database?",
            db_path=EVAL_SQLITE,
        )
        self.assertIn("person", plan["soft_terms"])
        self.assertNotIn("did", plan["soft_terms"])
        self.assertNotIn("database", plan["soft_terms"])

    def test_default_sqlite_path_prefers_ucfcrime_dataset_without_env_override(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("AGENT_SQLITE_DB_PATH", None)
            resolved = get_graph_sqlite_db_path()
        self.assertTrue(str(resolved).endswith("agent/test/generated/datasets/ucfcrime_eval.sqlite"))


if __name__ == "__main__":
    unittest.main()
