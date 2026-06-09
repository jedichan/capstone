"""Unit tests for the P1-Next-A summary_node bail-out tightening.

Covered truth-table (combination of rows / grounder / answer_type / verifier
decision / strict-flag / LLM output):

    rows == []                                 -> 'No matching clip is expected.' (early return)
    bail_out_strict=False                      -> legacy behaviour, any path may emit bail-out
    rows>0 + grounder OFF + LLM 'No matching'  -> demoted to factual fallback
    rows>0 + grounder OFF + LLM free-text      -> canonicalised to factual fallback
    rows>0 + grounder OFF + LLM 'Yes. ...'     -> kept as-is
    rows>0 + grounder ON  + existence + verifier=mismatch + LLM 'No matching'
                                              -> kept (grounder-sanctioned bail-out)
    rows>0 + grounder ON  + existence + verifier=exact    + LLM 'No matching'
                                              -> demoted to factual fallback
    rows>0 + grounder ON  + answer_type=list  + LLM 'No matching'
                                              -> demoted (only existence + mismatch may bail out)
    LLM prompt assertion: when rows>0, the prompt must NOT contain
        'answer exactly: No matching clip is expected.'
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

from node import summary_node as summary_module  # noqa: E402  (sys.path mutation)
from node.summary_node import (  # noqa: E402
    _allow_no_match_decision,
    _canonicalize_summary,
    _normalize_summary_output,
    create_summary_node,
)


_NO_MATCH = "No matching clip is expected."


class _StubLLM:
    """Captures the prompt sent by ``summary_node`` and returns a canned reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_prompt: str | None = None

    def bind(self, **_kwargs):  # pragma: no cover - matches LangChain runtime API
        return self

    def invoke(self, messages, *, config=None):
        del config
        # ``messages`` is a list of LangChain message objects; the second one is
        # the HumanMessage carrying our prompt.
        for msg in messages:
            content = getattr(msg, "content", None)
            if content and content != "Return only the final English summary text.":
                self.last_prompt = str(content)
                break
        return mock.Mock(content=self.reply)


def _row(video_id: str = "Abuse037_x264", start: float = 6.0, end: float = 22.0) -> dict:
    return {
        "video_id": video_id,
        "event_id": 1,
        "start_time": start,
        "end_time": end,
        "event_summary_en": "A car runs over a black dog on the road.",
        "_distance": 0.12,
    }


def _state_with_rows(
    *,
    rows: list[dict],
    answer_type: str = "",
    verifier_decision: str = "",
    raw_final_answer: str = "",
    user_query: str = "Is there a clip of a black dog on the road?",
) -> dict:
    state: dict = {
        "rerank_result": rows,
        "user_query": user_query,
        "original_user_query": user_query,
        "rewritten_query": user_query,
        "answer_type": answer_type,
        "raw_final_answer": raw_final_answer,
    }
    if verifier_decision:
        state["verifier_result"] = {"decision": verifier_decision}
    return state


class AllowNoMatchDecisionTests(unittest.TestCase):
    def test_empty_rows_always_allows(self) -> None:
        self.assertTrue(
            _allow_no_match_decision(
                rows=[],
                answer_type="existence",
                verifier_decision="exact",
                grounder_enabled=False,
                bail_out_strict=True,
            )
        )

    def test_strict_off_always_allows(self) -> None:
        self.assertTrue(
            _allow_no_match_decision(
                rows=[_row()],
                answer_type="list",
                verifier_decision="",
                grounder_enabled=False,
                bail_out_strict=False,
            )
        )

    def test_grounder_mismatch_existence_allows(self) -> None:
        self.assertTrue(
            _allow_no_match_decision(
                rows=[_row()],
                answer_type="existence",
                verifier_decision="mismatch",
                grounder_enabled=True,
                bail_out_strict=True,
            )
        )

    def test_grounder_off_with_rows_forbids(self) -> None:
        self.assertFalse(
            _allow_no_match_decision(
                rows=[_row()],
                answer_type="existence",
                verifier_decision="mismatch",
                grounder_enabled=False,
                bail_out_strict=True,
            )
        )

    def test_grounder_on_but_not_existence_forbids(self) -> None:
        self.assertFalse(
            _allow_no_match_decision(
                rows=[_row()],
                answer_type="list",
                verifier_decision="mismatch",
                grounder_enabled=True,
                bail_out_strict=True,
            )
        )

    def test_grounder_on_existence_but_exact_forbids(self) -> None:
        self.assertFalse(
            _allow_no_match_decision(
                rows=[_row()],
                answer_type="existence",
                verifier_decision="exact",
                grounder_enabled=True,
                bail_out_strict=True,
            )
        )


class NormalizeSummaryOutputTests(unittest.TestCase):
    def test_no_match_demoted_when_disallowed(self) -> None:
        self.assertEqual(
            _normalize_summary_output(_NO_MATCH, "FALLBACK", allow_no_match=False),
            "FALLBACK",
        )

    def test_no_match_kept_when_allowed(self) -> None:
        self.assertEqual(
            _normalize_summary_output(_NO_MATCH, "FALLBACK", allow_no_match=True),
            _NO_MATCH,
        )

    def test_no_match_with_extra_text_demoted(self) -> None:
        self.assertEqual(
            _normalize_summary_output(
                _NO_MATCH + " But there is a similar clip.",
                "FALLBACK",
                allow_no_match=False,
            ),
            "FALLBACK",
        )


class CanonicalizeSummaryTests(unittest.TestCase):
    def test_rows_present_strict_demotes_no_match_to_factual(self) -> None:
        rows = [_row()]
        result = _canonicalize_summary(
            _NO_MATCH,
            fallback="FALLBACK",
            rows=rows,
            query="is there a clip?",
            answer_type="list",
            verifier_decision="",
            grounder_enabled=False,
            bail_out_strict=True,
        )
        self.assertNotEqual(result, _NO_MATCH)
        self.assertIn("Abuse037_x264", result)
        self.assertTrue(
            result.startswith("Yes. The relevant clip is in ")
            or result.startswith("The most relevant clip is in ")
        )

    def test_rows_present_grounder_mismatch_keeps_no_match(self) -> None:
        rows = [_row()]
        result = _canonicalize_summary(
            _NO_MATCH,
            fallback="FALLBACK",
            rows=rows,
            query="is there a clip?",
            answer_type="existence",
            verifier_decision="mismatch",
            grounder_enabled=True,
            bail_out_strict=True,
        )
        self.assertEqual(result, _NO_MATCH)

    def test_legacy_strict_off_keeps_no_match(self) -> None:
        rows = [_row()]
        result = _canonicalize_summary(
            _NO_MATCH,
            fallback="FALLBACK",
            rows=rows,
            query="is there a clip?",
            answer_type="list",
            verifier_decision="",
            grounder_enabled=False,
            bail_out_strict=False,
        )
        self.assertEqual(result, _NO_MATCH)

    def test_freeform_text_canonicalised_to_factual(self) -> None:
        rows = [_row()]
        result = _canonicalize_summary(
            "There is a black dog on the road being hit by a white car.",
            fallback="FALLBACK",
            rows=rows,
            query="is there a clip?",
            answer_type="list",
        )
        self.assertTrue(
            result.startswith("Yes. The relevant clip is in ")
            or result.startswith("The most relevant clip is in ")
        )
        self.assertIn("Abuse037_x264", result)

    @mock.patch.dict(os.environ, {"AGENT_ENABLE_EXISTENCE_GROUNDER": "1"}, clear=False)
    def test_p1_7_llm_yes_demoted_when_mismatch_rerank(self) -> None:
        """P1-7 follow-up: LLM Yes-line must not override grounder mismatch + rerank_reselected."""
        rows = [_row()]
        vr = {
            "decision": "mismatch",
            "video_id": "Arrest046_x264",
            "start_time": 1.0,
            "end_time": 2.0,
            "span_source": "rerank_reselected",
        }
        yes_line = "Yes. The relevant clip is in Arrest046_x264, around 0:00:01 - 0:00:02."
        result = _canonicalize_summary(
            yes_line,
            fallback="SHOULD_NOT_USE",
            rows=rows,
            query="Is there a clip?",
            answer_type="existence",
            verifier_decision="mismatch",
            grounder_enabled=True,
            bail_out_strict=True,
            verifier_result=vr,
        )
        self.assertEqual(result, _NO_MATCH)


class SummaryNodeIntegrationTests(unittest.TestCase):
    """Integration: invoke the actual ``summary_node`` callable with a stub LLM."""

    def setUp(self) -> None:
        # Force strict bail-out on regardless of test environment.
        self._env_patch = mock.patch.dict(
            os.environ,
            {
                "AGENT_SUMMARY_BAIL_OUT_STRICT": "1",
                "AGENT_ENABLE_EXISTENCE_GROUNDER": "0",
            },
            clear=False,
        )
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()

    def test_rows_present_llm_says_no_match_demoted(self) -> None:
        llm = _StubLLM(reply=_NO_MATCH)
        node = create_summary_node(llm=llm)
        state = _state_with_rows(rows=[_row()], answer_type="list")
        out = node(state, config={}, store=None)
        self.assertNotIn(_NO_MATCH.rstrip("."), out["summary_result"]["summary"])
        # Final answer must contain the video_id from the factual fallback.
        self.assertIn("Abuse037_x264", out["final_answer"])

    def test_prompt_omits_no_match_clause_when_rows_present(self) -> None:
        llm = _StubLLM(
            reply="Yes. The relevant clip is in Abuse037_x264, around 0:00:06 - 0:00:22."
        )
        node = create_summary_node(llm=llm)
        state = _state_with_rows(rows=[_row()], answer_type="existence")
        node(state, config={}, store=None)
        prompt = llm.last_prompt or ""
        self.assertNotIn(
            "answer exactly: No matching clip is expected.",
            prompt,
            msg="rows>0 prompts must drop the bail-out instruction (P1-Next-A)",
        )
        self.assertIn(
            "Do not return 'No matching clip is expected.' when results are provided",
            prompt,
        )

    def test_empty_rows_takes_early_return_path(self) -> None:
        llm = _StubLLM(reply="ignored")
        node = create_summary_node(llm=llm)
        state = _state_with_rows(rows=[], user_query="non-existent")
        out = node(state, config={}, store=None)
        self.assertEqual(out["summary_result"]["summary"], _NO_MATCH)
        self.assertIn("empty_result_fallback", out["summary_result"]["style"])

    def test_legacy_strict_off_keeps_llm_no_match(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_SUMMARY_BAIL_OUT_STRICT": "0"}):
            llm = _StubLLM(reply=_NO_MATCH)
            node = create_summary_node(llm=llm)
            state = _state_with_rows(rows=[_row()], answer_type="list")
            out = node(state, config={}, store=None)
            self.assertEqual(out["summary_result"]["summary"], _NO_MATCH)

    def test_grounder_on_existence_mismatch_keeps_no_match(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_ENABLE_EXISTENCE_GROUNDER": "1"}):
            llm = _StubLLM(reply=_NO_MATCH)
            node = create_summary_node(llm=llm)
            state = _state_with_rows(
                rows=[_row()],
                answer_type="existence",
                verifier_decision="mismatch",
            )
            out = node(state, config={}, store=None)
            self.assertEqual(out["summary_result"]["summary"], _NO_MATCH)

    def test_grounder_on_existence_exact_demotes_llm_no_match(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_ENABLE_EXISTENCE_GROUNDER": "1"}):
            llm = _StubLLM(reply=_NO_MATCH)
            node = create_summary_node(llm=llm)
            state = _state_with_rows(
                rows=[_row()],
                answer_type="existence",
                verifier_decision="exact",
            )
            out = node(state, config={}, store=None)
            self.assertNotEqual(out["summary_result"]["summary"], _NO_MATCH)
            self.assertIn("Abuse037_x264", out["final_answer"])


class SummaryNodeNoLLMFallbackTests(unittest.TestCase):
    """When llm=None, the node should still respect bail_out_strict."""

    def test_llm_none_with_rows_returns_factual_fallback(self) -> None:
        node = create_summary_node(llm=None)
        state = _state_with_rows(rows=[_row()], answer_type="list")
        out = node(state, config={}, store=None)
        # The no-LLM path uses factual_fallback verbatim; with rows>0 that is
        # the canonical "Yes. ..." sentence, never the bail-out string.
        self.assertNotEqual(out["summary_result"]["summary"], _NO_MATCH)
        self.assertIn("Abuse037_x264", out["summary_result"]["summary"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
