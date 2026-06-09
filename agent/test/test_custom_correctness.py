"""Unit tests for P1-Next-C ``custom_correctness`` (rule-based, no LLM)."""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT_DIR / "agent"
TEST_DIR = AGENT_DIR / "test"
for _p in (TEST_DIR, AGENT_DIR, ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ragas_eval_runner as rer  # noqa: E402


class PredictedLabelTests(unittest.TestCase):
    def test_no_matching(self) -> None:
        self.assertEqual(
            rer._predicted_answer_label_from_response("No matching clip is expected."),
            "no",
        )

    def test_most_relevant_yes(self) -> None:
        self.assertEqual(
            rer._predicted_answer_label_from_response(
                "The most relevant clip is in Arrest043, around 0:01:00 - 0:01:30."
            ),
            "yes",
        )

    def test_yes_relevant_clip(self) -> None:
        self.assertEqual(
            rer._predicted_answer_label_from_response(
                "Yes. The relevant clip is in Arrest043, around 0:01:00 - 0:01:30."
            ),
            "yes",
        )

    def test_empty_unknown(self) -> None:
        self.assertEqual(rer._predicted_answer_label_from_response(""), "")


class CustomCorrectnessFormulaTests(unittest.TestCase):
    def test_yes_all_match(self) -> None:
        out = rer._compute_custom_correctness(
            {
                "expected_answer_label": "yes",
                "video_id": "V1",
                "response": "The most relevant clip is in V1, around 0:00:10 - 0:00:20.",
                "predicted_video_id": "V1",
                "expected_start_sec": 10.0,
                "expected_end_sec": 20.0,
                "predicted_start_sec": 10.0,
                "predicted_end_sec": 20.0,
                "expected_time_is_approx": 0,
            }
        )
        self.assertEqual(out["score"], 1.0)

    def test_yes_missing_time_video_only(self) -> None:
        out = rer._compute_custom_correctness(
            {
                "expected_answer_label": "yes",
                "video_id": "V1",
                "response": "The most relevant clip is in V1.",
                "predicted_video_id": "V1",
                "expected_start_sec": None,
                "expected_end_sec": None,
                "predicted_start_sec": None,
                "predicted_end_sec": None,
                "expected_time_is_approx": 0,
            }
        )
        self.assertEqual(out["detail"]["branch"], "yes_missing_time")
        self.assertEqual(out["score"], 1.0)

    def test_yes_wrong_video(self) -> None:
        out = rer._compute_custom_correctness(
            {
                "expected_answer_label": "yes",
                "video_id": "V1",
                "response": "The most relevant clip is in V2.",
                "predicted_video_id": "V2",
                "expected_start_sec": 10.0,
                "expected_end_sec": 20.0,
                "predicted_start_sec": 10.0,
                "predicted_end_sec": 20.0,
                "expected_time_is_approx": 0,
            }
        )
        self.assertEqual(out["detail"]["yes_no_score"], 1.0)
        self.assertEqual(out["detail"]["video_id_score"], 0.0)
        self.assertLess(out["score"] or 0.0, 1.0)

    def test_no_branch_skips_video_weight(self) -> None:
        out = rer._compute_custom_correctness(
            {
                "expected_answer_label": "no",
                "video_id": "V1",
                "response": "No matching clip is expected.",
                "predicted_video_id": "V999",
                "expected_start_sec": 10.0,
                "expected_end_sec": 20.0,
                "predicted_start_sec": 10.0,
                "predicted_end_sec": 20.0,
                "expected_time_is_approx": 0,
            }
        )
        self.assertEqual(out["detail"]["branch"], "no_expected_time")
        self.assertEqual(out["score"], 1.0)
        # IoU / time fields must be None for "no" cases — there is no GT time window.
        self.assertIsNone(out["detail"]["time_iou_score"])
        self.assertIsNone(out["detail"]["time_bonus"])
        self.assertIsNone(out["detail"]["time_term"])

    def test_no_branch_ignores_mismatched_time_prediction(self) -> None:
        """Bug fix: when expected is "no", even mismatched time spans must not
        penalize the score.  A correct "no" prediction should always score 1.0
        regardless of whatever time window the agent happened to output."""
        out = rer._compute_custom_correctness(
            {
                "expected_answer_label": "no",
                "video_id": "V1",
                "response": "No matching clip is expected.",
                "predicted_video_id": "V999",
                "expected_start_sec": 10.0,
                "expected_end_sec": 20.0,
                # Predicted time span has zero IoU with GT — but since the
                # answer is "no", IoU must not be computed at all.
                "predicted_start_sec": 50.0,
                "predicted_end_sec": 60.0,
                "expected_time_is_approx": 0,
            }
        )
        self.assertEqual(out["score"], 1.0)
        self.assertEqual(out["detail"]["branch"], "no_expected_time")
        self.assertEqual(out["detail"]["yes_no_score"], 1.0)
        self.assertIsNone(out["detail"]["time_iou_score"])

    def test_approx_tolerance_improves_iou(self) -> None:
        tight = rer._compute_custom_correctness(
            {
                "expected_answer_label": "yes",
                "video_id": "V1",
                "response": "The most relevant clip is in V1.",
                "predicted_video_id": "V1",
                "expected_start_sec": 100.0,
                "expected_end_sec": 110.0,
                "predicted_start_sec": 93.0,
                "predicted_end_sec": 96.0,
                "expected_time_is_approx": 0,
            }
        )
        loose = rer._compute_custom_correctness(
            {
                "expected_answer_label": "yes",
                "video_id": "V1",
                "response": "The most relevant clip is in V1.",
                "predicted_video_id": "V1",
                "expected_start_sec": 100.0,
                "expected_end_sec": 110.0,
                "predicted_start_sec": 93.0,
                "predicted_end_sec": 96.0,
                "expected_time_is_approx": 1,
            }
        )
        self.assertLess((tight["detail"] or {}).get("time_iou_score") or 0.0, (loose["detail"] or {}).get("time_iou_score") or 0.0)


if __name__ == "__main__":
    unittest.main()
