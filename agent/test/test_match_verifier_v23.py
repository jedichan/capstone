"""Unit tests for the P1-7 v2.3 verifier (multi-candidate span re-selection).

Coverage:
  - LLM picks a non-top-1 chunk -> span_source = "rerank_reselected"
  - LLM picks the top-1 chunk -> span_source = "candidate_top_row"
  - LLM call fails / disabled -> heuristic best-chunk fallback path
  - Out-of-range LLM index -> falls back to heuristic
  - Same-video grouping (PART1_0011 style: all candidates from one video)
  - Cross-video diversity slot is honoured
  - answer_type != "existence" -> pass-through (verifier_result has decision="skipped")
  - rows == [] -> mismatch verdict, no candidates considered
  - Legacy AGENT_VERIFIER_RESELECT_SPAN=0 -> single-row v1 path
  - Downstream summary_node consumes verifier's re-selected span
    via _build_factual_summary
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT_DIR / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from node import match_verifier_node as mv  # noqa: E402
from node.summary_node import _build_factual_summary  # noqa: E402


class _StubLLM:
    """Captures the prompt and returns a canned reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_prompt: str | None = None
        self.invoke_count = 0

    def bind(self, **_kwargs):  # pragma: no cover - LangChain runtime API
        return self

    def invoke(self, messages, *, config=None):
        del config
        self.invoke_count += 1
        for msg in messages:
            content = getattr(msg, "content", None)
            if content and "Return JSON only" not in str(content)[:80]:
                self.last_prompt = str(content)
                break
        return mock.Mock(content=self.reply)


def _row(
    *,
    video_id: str = "Abuse040_x264",
    entity_hint: str = "segment_2",
    start: float = 26.6,
    end: float = 40.3,
    summary: str = "The woman in blue clothes took the wheelchair and pushed it to the side of the sofa.",
    distance: float = 0.12,
) -> dict:
    return {
        "video_id": video_id,
        "entity_hint": entity_hint,
        "event_id": f"{video_id}:{entity_hint}",
        "start_time": start,
        "end_time": end,
        "event_summary_en": summary,
        "_distance": distance,
    }


def _abuse040_top5() -> list[dict]:
    """The five same-video candidates from PART1_0011 in the 50-case eval.
    Order matches actual rerank_result top-5: top-1 is segment_2 (wheelchair,
    no hitting), the four hitting segments are below it.
    """
    return [
        _row(entity_hint="segment_2", start=26.6, end=40.3,
             summary="The woman in blue clothes took the wheelchair and pushed it to the side of the sofa."),
        _row(entity_hint="segment_4", start=46.4, end=58.7,
             summary="The woman next to the sofa hit the person on the sofa on the head and pushed it again."),
        _row(entity_hint="segment_6", start=70.1, end=85.8,
             summary="The white-haired woman on the sofa stood up with her wheelchair, then pushed the wheelchair towards the door."),
        _row(entity_hint="segment_3", start=40.3, end=44.4,
             summary="The woman on the sofa wanted to sit up, but the woman standing next to her hit her on the head."),
        _row(entity_hint="segment_5", start=58.7, end=69.1,
             summary="The woman in the blue top hit the white-haired woman on the head, then turned the wheelchair to her side."),
    ]


class _BaseVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env_patch = mock.patch.dict(
            os.environ,
            {
                "AGENT_VERIFIER_RESELECT_SPAN": "1",
                "AGENT_MATCH_VERIFIER_USE_LLM": "1",
                "AGENT_VERIFIER_CANDIDATE_LIMIT": "8",
                "AGENT_VERIFIER_CROSS_VIDEO_TOP_N": "2",
            },
            clear=False,
        )
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()


class LLMReselectionTests(_BaseVerifierTest):
    def test_llm_picks_non_top1_marks_rerank_reselected(self) -> None:
        rows = _abuse040_top5()
        # LLM picks index 4 (segment_5, the actually-matching hitting chunk)
        llm = _StubLLM(reply='{"best_chunk_index": 4, "decision": "exact", "confidence": 0.92, "reason": "segment_5 hits white-haired woman on the head, matches query"}')
        node = mv.create_match_verifier_node(llm=llm)

        state = {
            "answer_type": "existence",
            "rerank_result": rows,
            "user_query": "Is there a clip of a caregiver hitting a white-haired elderly person on the head?",
        }
        out = node(state, config={}, store=None)

        vr = out["verifier_result"]
        self.assertEqual(vr["span_source"], "rerank_reselected")
        self.assertEqual(vr["best_chunk_index"], 4)
        self.assertEqual(vr["candidate_count"], 5)
        self.assertEqual(vr["decision"], "exact")
        # The re-selected span should reflect segment_5's time window, not segment_2's
        self.assertAlmostEqual(vr["start_time"], 58.7)
        self.assertAlmostEqual(vr["end_time"], 69.1)
        self.assertIn("hit the white-haired woman", vr["primary_summary"])

    def test_llm_picks_top1_marks_candidate_top_row(self) -> None:
        rows = _abuse040_top5()
        llm = _StubLLM(reply='{"best_chunk_index": 0, "decision": "exact", "confidence": 0.9, "reason": "top-1 already matches"}')
        node = mv.create_match_verifier_node(llm=llm)
        out = node(
            {"answer_type": "existence", "rerank_result": rows, "user_query": "any query"},
            config={},
            store=None,
        )
        vr = out["verifier_result"]
        self.assertEqual(vr["span_source"], "candidate_top_row")
        self.assertEqual(vr["best_chunk_index"], 0)

    def test_llm_picks_partial_decision_propagates(self) -> None:
        rows = _abuse040_top5()
        llm = _StubLLM(reply='{"best_chunk_index": 1, "decision": "partial", "confidence": 0.7, "reason": "subject role differs slightly"}')
        node = mv.create_match_verifier_node(llm=llm)
        out = node(
            {"answer_type": "existence", "rerank_result": rows, "user_query": "any"},
            config={},
            store=None,
        )
        self.assertEqual(out["verifier_result"]["decision"], "partial")
        self.assertEqual(out["verifier_result"]["span_source"], "rerank_reselected")

    def test_prompt_includes_all_candidates(self) -> None:
        rows = _abuse040_top5()
        llm = _StubLLM(reply='{"best_chunk_index": 0, "decision": "exact", "confidence": 0.9, "reason": "x"}')
        node = mv.create_match_verifier_node(llm=llm)
        node(
            {"answer_type": "existence", "rerank_result": rows, "user_query": "q"},
            config={},
            store=None,
        )
        prompt = llm.last_prompt or ""
        # All 5 same-video candidates should be present, indices [0]..[4]
        for idx in range(5):
            self.assertIn(f"[{idx}]", prompt)
        self.assertIn("CANDIDATES:", prompt)


class FallbackPathTests(_BaseVerifierTest):
    def test_llm_out_of_range_index_falls_back_to_heuristic(self) -> None:
        rows = _abuse040_top5()
        llm = _StubLLM(reply='{"best_chunk_index": 99, "decision": "exact", "confidence": 0.9, "reason": "x"}')
        node = mv.create_match_verifier_node(llm=llm)
        out = node(
            {"answer_type": "existence", "rerank_result": rows, "user_query": "caregiver hitting head"},
            config={},
            store=None,
        )
        # Heuristic kicks in -> verdict mode is "heuristic"
        self.assertEqual(out["verifier_result"]["mode"], "heuristic")

    def test_llm_invalid_json_falls_back_to_heuristic(self) -> None:
        rows = _abuse040_top5()
        llm = _StubLLM(reply="this is not JSON at all")
        node = mv.create_match_verifier_node(llm=llm)
        out = node(
            {"answer_type": "existence", "rerank_result": rows, "user_query": "caregiver hitting head"},
            config={},
            store=None,
        )
        self.assertEqual(out["verifier_result"]["mode"], "heuristic")

    def test_llm_disabled_uses_heuristic(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_MATCH_VERIFIER_USE_LLM": "0"}):
            rows = _abuse040_top5()
            llm = _StubLLM(reply="should not be called")
            node = mv.create_match_verifier_node(llm=llm)
            out = node(
                {"answer_type": "existence", "rerank_result": rows, "user_query": "caregiver hitting elderly head"},
                config={},
                store=None,
            )
            self.assertEqual(out["verifier_result"]["mode"], "heuristic")
            self.assertEqual(llm.invoke_count, 0)


class CandidateCollectionTests(_BaseVerifierTest):
    def test_same_video_chunks_dominate_when_top_k_homogeneous(self) -> None:
        rows = _abuse040_top5()
        candidates = mv._collect_candidates(rows, candidate_limit=8, cross_video_top_n=2)
        self.assertEqual(len(candidates), 5)
        for c in candidates:
            self.assertEqual(c["video_id"], "Abuse040_x264")

    def test_cross_video_top_n_keeps_diversity(self) -> None:
        mixed = [
            _row(video_id="Abuse040_x264", entity_hint="segment_2", start=26.6, end=40.3),
            _row(video_id="Other_video", entity_hint="segment_1", start=10.0, end=20.0),
            _row(video_id="Abuse040_x264", entity_hint="segment_4", start=46.4, end=58.7),
            _row(video_id="Yet_another", entity_hint="segment_1", start=5.0, end=12.0),
            _row(video_id="Yet_another", entity_hint="segment_3", start=20.0, end=30.0),
        ]
        candidates = mv._collect_candidates(mixed, candidate_limit=8, cross_video_top_n=2)
        # 2 same-video (Abuse040) + 2 cross-video (Other_video, Yet_another)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[0]["video_id"], "Abuse040_x264")  # rerank top-1 still index 0
        videos = [c["video_id"] for c in candidates]
        self.assertEqual(videos.count("Abuse040_x264"), 2)
        self.assertIn("Other_video", videos)
        self.assertIn("Yet_another", videos)

    def test_cross_video_top_n_zero_keeps_only_same_video(self) -> None:
        mixed = [
            _row(video_id="Abuse040_x264", entity_hint="segment_2"),
            _row(video_id="Other_video", entity_hint="segment_1"),
        ]
        candidates = mv._collect_candidates(mixed, candidate_limit=8, cross_video_top_n=0)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["video_id"], "Abuse040_x264")

    def test_candidate_limit_caps_pool(self) -> None:
        many = [_row(entity_hint=f"segment_{i}") for i in range(20)]
        candidates = mv._collect_candidates(many, candidate_limit=5, cross_video_top_n=0)
        self.assertEqual(len(candidates), 5)


class HeuristicReselectionTests(_BaseVerifierTest):
    def test_heuristic_picks_chunk_with_max_token_overlap(self) -> None:
        rows = _abuse040_top5()
        idx, verdict = mv._heuristic_select_best_chunk(
            "caregiver hit elderly white-haired woman on the head", rows
        )
        # heuristic should prefer one of the hitting chunks (segment_3/4/5).
        # Indices in _abuse040_top5: 0=segment_2 (wheelchair), 1=segment_4 (hit head),
        # 2=segment_6 (wheelchair), 3=segment_3 (hit head), 4=segment_5 (hit head).
        self.assertNotEqual(idx, 0)  # not the wheelchair-only segment_2
        self.assertNotEqual(idx, 2)  # not segment_6 (wheelchair only)
        self.assertIn(verdict["decision"], {"exact", "partial"})

    def test_heuristic_no_match_yields_mismatch(self) -> None:
        rows = _abuse040_top5()
        idx, verdict = mv._heuristic_select_best_chunk("yellow car parking lot", rows)
        self.assertEqual(verdict["decision"], "mismatch")


class PassThroughTests(_BaseVerifierTest):
    def test_non_existence_answer_type_skips(self) -> None:
        node = mv.create_match_verifier_node(llm=_StubLLM(reply=""))
        out = node(
            {"answer_type": "list", "rerank_result": _abuse040_top5(), "user_query": "list all clips"},
            config={},
            store=None,
        )
        self.assertEqual(out["verifier_result"]["decision"], "skipped")

    def test_empty_rows_yields_mismatch_no_candidates(self) -> None:
        node = mv.create_match_verifier_node(llm=_StubLLM(reply=""))
        out = node(
            {"answer_type": "existence", "rerank_result": [], "user_query": "any"},
            config={},
            store=None,
        )
        self.assertEqual(out["verifier_result"]["decision"], "mismatch")
        self.assertEqual(out["verifier_result"]["reason"], "no_rows")


class StrictAttributeVerifierTests(_BaseVerifierTest):
    def test_semantic_binary_appearance_conflict_returns_mismatch(self) -> None:
        node = mv.create_match_verifier_node(llm=_StubLLM(reply='{"best_chunk_index": 0, "decision": "exact", "confidence": 0.95, "reason": "time matches"}'))
        row = _row(
            video_id="2018-03-11.14-20-01.14-25-01.school.G423.r13",
            start=8.0,
            end=13.0,
            summary="A person with black coat, black pants, no bag exits from the right side.",
        )
        out = node(
            {
                "answer_type": "semantic",
                "rerank_result": [row],
                "user_query": "In the 14-20 video for camera G423, around 0:08-0:13, is there a person with bright red jacket visible?",
            },
            config={},
            store=None,
        )
        vr = out["verifier_result"]
        self.assertEqual(vr["decision"], "mismatch")
        self.assertEqual(vr["mode"], "strict_rule")
        self.assertIn("appearance_conflict", vr["reason"])

    def test_semantic_binary_direction_conflict_returns_mismatch(self) -> None:
        node = mv.create_match_verifier_node(llm=_StubLLM(reply='{"best_chunk_index": 0, "decision": "exact", "confidence": 0.95, "reason": "time matches"}'))
        row = _row(
            video_id="2018-03-11.14-20-01.14-25-01.school.G423.r13",
            start=8.0,
            end=13.0,
            summary="A person with black coat exits from the right side.",
        )
        out = node(
            {
                "answer_type": "semantic",
                "rerank_result": [row],
                "user_query": "In the 14-20 video for camera G423, around 0:08-0:13, did a person exit from the left side?",
            },
            config={},
            store=None,
        )
        vr = out["verifier_result"]
        self.assertEqual(vr["decision"], "mismatch")
        self.assertEqual(vr["mode"], "strict_rule")
        self.assertIn("direction_conflict", vr["reason"])

    def test_matching_direction_keeps_positive_verdict(self) -> None:
        node = mv.create_match_verifier_node(llm=_StubLLM(reply='{"best_chunk_index": 0, "decision": "exact", "confidence": 0.95, "reason": "matches"}'))
        row = _row(
            video_id="2018-03-11.14-20-01.14-25-01.school.G423.r13",
            start=8.0,
            end=13.0,
            summary="A person with black coat exits from the right side.",
        )
        out = node(
            {
                "answer_type": "semantic",
                "rerank_result": [row],
                "user_query": "In the 14-20 video for camera G423, around 0:08-0:13, did a person exit from the right side?",
            },
            config={},
            store=None,
        )
        self.assertEqual(out["verifier_result"]["decision"], "exact")

    def test_generic_dark_clothing_does_not_reject_black_coat(self) -> None:
        node = mv.create_match_verifier_node(llm=_StubLLM(reply='{"best_chunk_index": 0, "decision": "exact", "confidence": 0.95, "reason": "matches"}'))
        row = _row(
            video_id="2018-03-11.14-20-01.14-25-01.school.G423.r13",
            start=8.0,
            end=13.0,
            summary="Appearance: dark upper-body clothing, dark lower-body clothing, backpack visible.",
        )
        out = node(
            {
                "answer_type": "existence",
                "rerank_result": [row],
                "user_query": "In the 14-20 video for camera G423, around 0:08-0:13, is there a person with black coat visible?",
            },
            config={},
            store=None,
        )
        self.assertEqual(out["verifier_result"]["decision"], "exact")

    def test_missing_direction_detail_but_time_match_keeps_yes(self) -> None:
        node = mv.create_match_verifier_node(llm=_StubLLM(reply='{"best_chunk_index": 0, "decision": "mismatch", "confidence": 0.8, "reason": "no direction words"}'))
        row = _row(
            video_id="2018-03-11.13-50-01.13-55-01.admin.G329.r13",
            start=110.0,
            end=116.0,
            summary="Appearance: dark upper-body clothing, dark lower-body clothing, medium build.",
        )
        out = node(
            {
                "answer_type": "unknown",
                "rerank_result": [row],
                "user_query": "In the 13-50 video for camera G329, around 1:50-2:01, did a person exit from the right side?",
            },
            config={},
            store=None,
        )
        vr = out["verifier_result"]
        self.assertEqual(vr["decision"], "exact")
        self.assertEqual(vr["mode"], "strict_rule")
        self.assertIn("time_window_match", vr["reason"])

    def test_time_window_preferred_before_attribute_check(self) -> None:
        node = mv.create_match_verifier_node(llm=_StubLLM(reply='{"best_chunk_index": 0, "decision": "exact", "confidence": 0.95, "reason": "red candidate"}'))
        rows = [
            _row(
                video_id="2018-03-11.14-20-01.14-25-01.school.G423.r13",
                start=197.5,
                end=200.3,
                summary="Appearance: red upper-body clothing, dark lower-body pants.",
            ),
            _row(
                video_id="2018-03-11.14-20-01.14-25-01.school.G423.r13",
                start=8.0,
                end=13.0,
                summary="Appearance: dark upper-body clothing, dark lower-body clothing.",
            ),
        ]
        out = node(
            {
                "answer_type": "existence",
                "rerank_result": rows,
                "user_query": "In the 14-20 video for camera G423, around 0:08-0:13, is there a person with bright red jacket visible?",
            },
            config={},
            store=None,
        )
        vr = out["verifier_result"]
        self.assertEqual(vr["decision"], "mismatch")
        self.assertAlmostEqual(vr["start_time"], 8.0)
        self.assertIn("appearance_conflict", vr["reason"])


class LegacyPathTests(_BaseVerifierTest):
    def test_reselect_off_uses_v1_single_row_path(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_VERIFIER_RESELECT_SPAN": "0"}):
            rows = _abuse040_top5()
            llm = _StubLLM(reply='{"decision": "mismatch", "confidence": 0.9, "reason": "wheelchair, not hitting"}')
            node = mv.create_match_verifier_node(llm=llm)
            out = node(
                {"answer_type": "existence", "rerank_result": rows, "user_query": "caregiver hitting"},
                config={},
                store=None,
            )
            vr = out["verifier_result"]
            self.assertEqual(vr["span_source"], "candidate_top_row")
            self.assertEqual(vr["best_chunk_index"], 0)
            self.assertEqual(vr["candidate_count"], 1)
            # span fields come from the rerank top-1 (segment_2)
            self.assertAlmostEqual(vr["start_time"], 26.6)


class DownstreamConsumptionTests(unittest.TestCase):
    """Verify summary_node._build_factual_summary picks up the verifier's
    re-selected span when span_source == 'rerank_reselected'."""

    def test_factual_summary_uses_verifier_span_when_reselected(self) -> None:
        rows = [
            {
                "video_id": "Abuse040_x264",
                "start_time": 26.6,
                "end_time": 40.3,
                "event_summary_en": "wheelchair scene",
            }
        ]
        verifier_result = {
            "decision": "exact",
            "video_id": "Abuse040_x264",
            "start_time": 58.7,
            "end_time": 69.1,
            "primary_summary": "hit white-haired woman on the head",
            "span_source": "rerank_reselected",
            "best_chunk_index": 4,
            "candidate_count": 5,
        }
        out = _build_factual_summary(
            rows, "Is there a clip of a caregiver hitting?", verifier_result=verifier_result
        )
        # Time window should reflect verifier's segment_5 (round(58.7)=59 - round(69.1)=69),
        # not segment_2's (round(26.6)=27)
        self.assertIn("0:00:59", out)
        self.assertIn("0:01:09", out)
        # Make sure segment_2's time is NOT in the output
        self.assertNotIn("0:00:27", out)

    def test_factual_summary_uses_top_row_when_span_source_is_top_row(self) -> None:
        rows = [
            {
                "video_id": "Abuse040_x264",
                "start_time": 26.6,
                "end_time": 40.3,
                "event_summary_en": "wheelchair scene",
            }
        ]
        verifier_result = {
            "decision": "exact",
            "video_id": "Abuse040_x264",
            "start_time": 26.6,
            "end_time": 40.3,
            "span_source": "candidate_top_row",
            "best_chunk_index": 0,
            "candidate_count": 5,
        }
        out = _build_factual_summary(rows, "any?", verifier_result=verifier_result)
        # Should use top-row's span (round(26.6)=27, round(40.3)=40)
        self.assertIn("0:00:27", out)
        self.assertIn("0:00:40", out)

    def test_factual_summary_no_verifier_falls_back_to_top_row(self) -> None:
        rows = [
            {
                "video_id": "Abuse040_x264",
                "start_time": 26.6,
                "end_time": 40.3,
                "event_summary_en": "wheelchair scene",
            }
        ]
        out = _build_factual_summary(rows, "any?", verifier_result=None)
        self.assertIn("0:00:27", out)
        self.assertIn("0:00:40", out)

    def test_factual_summary_empty_rows_returns_no_match_string(self) -> None:
        out = _build_factual_summary([], "any?", verifier_result=None)
        self.assertEqual(out, "No matching clip is expected.")

    @mock.patch.dict(os.environ, {"AGENT_ENABLE_EXISTENCE_GROUNDER": "1"}, clear=False)
    def test_p1_7_grounder_on_mismatch_rerank_returns_no_match_not_yes(self) -> None:
        """P1-7 follow-up: mismatch + rerank_reselected must not emit Yes + wrong clip."""
        rows = [
            {
                "video_id": "Arrest043_x264",
                "start_time": 10.0,
                "end_time": 20.0,
                "event_summary_en": "scene",
            }
        ]
        verifier_result = {
            "decision": "mismatch",
            "video_id": "Arrest046_x264",
            "start_time": 1.0,
            "end_time": 5.0,
            "span_source": "rerank_reselected",
            "best_chunk_index": 2,
            "candidate_count": 5,
        }
        out = _build_factual_summary(
            rows, "Is there a clip of X?", verifier_result=verifier_result
        )
        self.assertEqual(out, "No matching clip is expected.")

    @mock.patch.dict(os.environ, {"AGENT_ENABLE_EXISTENCE_GROUNDER": "0"}, clear=False)
    def test_p1_7_grounder_off_mismatch_rerank_still_uses_verifier_span(self) -> None:
        """Grounder OFF: keep legacy Yes + verifier span for rows>0."""
        rows = [
            {
                "video_id": "Arrest043_x264",
                "start_time": 10.0,
                "end_time": 20.0,
                "event_summary_en": "scene",
            }
        ]
        verifier_result = {
            "decision": "mismatch",
            "video_id": "Arrest046_x264",
            "start_time": 58.7,
            "end_time": 69.1,
            "span_source": "rerank_reselected",
            "best_chunk_index": 4,
            "candidate_count": 5,
        }
        out = _build_factual_summary(
            rows, "Is there a clip of X?", verifier_result=verifier_result
        )
        self.assertIn("Arrest046_x264", out)
        self.assertNotIn("No matching clip is expected.", out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
