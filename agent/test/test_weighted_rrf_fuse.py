import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "agents" / "shared" / "fusion_engine.py"
spec = importlib.util.spec_from_file_location("fusion_engine", MODULE_PATH)
fusion_engine = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(fusion_engine)
weighted_rrf_fuse = fusion_engine.weighted_rrf_fuse


class WeightedRrfFuseTests(unittest.TestCase):
    def test_overlap_rows_preserve_sql_and_hybrid_fields(self) -> None:
        sql_rows = [
            {"event_id": 1, "video_id": "A", "event_summary_en": "sql summary", "_source_type": "sql"},
        ]
        hybrid_rows = [
            {"event_id": 1, "video_id": "A", "event_text_en": "hybrid text", "_hybrid_score": 0.9, "_source_type": "hybrid"},
        ]

        rows, meta = weighted_rrf_fuse(sql_rows, hybrid_rows, label="mixed", limit=5)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get("_source_type"), "fused")
        self.assertEqual(rows[0].get("event_summary_en"), "sql summary")
        self.assertEqual(rows[0].get("event_text_en"), "hybrid text")
        self.assertEqual(meta.get("overlap_count"), 1)
        self.assertTrue(rows[0].get("_fusion_trace", {}).get("seen_in_sql"))
        self.assertTrue(rows[0].get("_fusion_trace", {}).get("seen_in_hybrid"))


if __name__ == "__main__":
    unittest.main()
