"""Unit tests for `extract_text_tokens_for_sql` after the P0-4 stopword cleanup.

Focus areas:
  - content words previously blacklisted (person, car, moving, area, clothed)
    now survive, either as direct tokens or deduped through filter_terms
  - plurals are normalized so the SQL LIKE path does not search on both
    "persons" and "person"
  - genuine function words (the, are, is, with, from, for, this, that, ...)
    are still stripped
  - emitted list is deduped and bounded by the limit cap
"""

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT_DIR / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from node.retrieval_contracts import (  # noqa: E402  (must come after sys.path mutation)
    _PLURAL_TO_SINGULAR,
    SQL_TOKEN_STOPWORDS,
    extract_structured_filters,
    extract_text_tokens_for_sql,
)


class ExtractTextTokensForSqlTests(unittest.TestCase):
    def test_function_words_are_stripped(self) -> None:
        tokens = extract_text_tokens_for_sql(
            "Is there a clip of someone in the database please?",
            filters={},
        )
        for fn_word in ("the", "there", "this", "that", "with", "from", "for", "clip", "database", "please"):
            self.assertNotIn(fn_word, tokens, f"{fn_word!r} should have been stripped")

    def test_moving_and_area_survive(self) -> None:
        """`moving` and `area` were previously blacklisted; they are the signal."""
        tokens = extract_text_tokens_for_sql(
            "moving person near the parking area",
            filters={},
        )
        self.assertIn("moving", tokens)
        self.assertIn("area", tokens)

    def test_content_word_dedups_with_filter_terms(self) -> None:
        query = "Show me a red car in the parking area."
        filters = extract_structured_filters(query)
        tokens = extract_text_tokens_for_sql(query, filters)
        # `red`, `car`, `parking` already become filter_terms, so they
        # should NOT leak back into the LIKE-token list.
        self.assertNotIn("red", tokens)
        self.assertNotIn("car", tokens)
        self.assertNotIn("parking", tokens)
        # `area` has no structured filter, must survive as a LIKE token
        self.assertIn("area", tokens)

    def test_plural_normalizes_to_filter_singular(self) -> None:
        """`persons` must not sneak past filter_terms dedup as a separate token."""
        query = "Are there any persons lying on the ground?"
        filters = extract_structured_filters(query)
        self.assertEqual(filters.get("object_type"), "person")
        tokens = extract_text_tokens_for_sql(query, filters)
        self.assertNotIn("persons", tokens)
        self.assertNotIn("person", tokens)
        self.assertIn("lying", tokens)
        self.assertIn("ground", tokens)

    def test_cars_plural_is_singularized_and_deduped(self) -> None:
        query = "I saw two cars near the parking."
        filters = extract_structured_filters(query)
        self.assertEqual(filters.get("object_type"), "car")
        tokens = extract_text_tokens_for_sql(query, filters)
        self.assertNotIn("cars", tokens)
        self.assertNotIn("car", tokens)

    def test_part1_0020_police_stomping_case(self) -> None:
        """Regression target for PART1_0020 / 0026 style UCFCrime prompts."""
        query = (
            "Is there a clip of a police officer stomping on and beating a "
            "person lying on the ground in a yard?"
        )
        filters = extract_structured_filters(query)
        tokens = extract_text_tokens_for_sql(query, filters)
        # We expect the strongest signal tokens to survive (order / cap aware).
        expected_signal = {"police", "officer", "stomping", "beating", "lying", "ground", "yard"}
        surviving = expected_signal.intersection(tokens)
        # At least 4 of the 7 signal words must survive the 6-token cap.
        self.assertGreaterEqual(
            len(surviving),
            4,
            f"expected >=4 of {expected_signal} in {tokens}",
        )
        # `person` was folded into filter_terms, must not duplicate.
        self.assertNotIn("person", tokens)
        self.assertLessEqual(len(tokens), 6)

    def test_dedup_within_single_query(self) -> None:
        tokens = extract_text_tokens_for_sql(
            "person person persons person running running",
            filters={"object_type": "person"},
        )
        self.assertNotIn("person", tokens)
        self.assertNotIn("persons", tokens)
        self.assertEqual(tokens.count("running"), 1)

    def test_short_tokens_are_length_filtered(self) -> None:
        tokens = extract_text_tokens_for_sql("is at on by to of a an in", filters={})
        self.assertEqual(tokens, [])

    def test_stopwords_table_does_not_contain_content_words(self) -> None:
        """Guard against future regressions where someone re-adds content words."""
        for banned in ("person", "car", "moving", "area", "clothed", "officer", "ground"):
            self.assertNotIn(
                banned,
                SQL_TOKEN_STOPWORDS,
                f"stopword set should not contain content word {banned!r}",
            )

    def test_plural_map_has_expected_entries(self) -> None:
        self.assertEqual(_PLURAL_TO_SINGULAR["persons"], "person")
        self.assertEqual(_PLURAL_TO_SINGULAR["cars"], "car")
        self.assertEqual(_PLURAL_TO_SINGULAR["people"], "person")
        self.assertEqual(_PLURAL_TO_SINGULAR["children"], "child")
        self.assertEqual(_PLURAL_TO_SINGULAR["officers"], "officer")

    def test_light_grey_hoodie_query_gets_phrase_expansion(self) -> None:
        query = "Did a person with light grey hoodie appear in camera G424 and then appear again in camera G421?"
        filters = extract_structured_filters(query)
        tokens = extract_text_tokens_for_sql(query, filters)
        self.assertIn("light grey hoodie", tokens)
        self.assertIn("light_grey_hoodie", tokens)
        self.assertIn("hoodie", tokens)

    def test_hood_up_query_gets_canonical_terms(self) -> None:
        query = "Did a person with dark jacket (hood up) from camera G329 also appear in camera G421?"
        filters = extract_structured_filters(query)
        tokens = extract_text_tokens_for_sql(query, filters)
        self.assertIn("hood up", tokens)
        self.assertIn("hood_up", tokens)
        self.assertIn("hoodie", tokens)

    def test_long_coat_query_expands_to_dark_coat_alias(self) -> None:
        query = "Did a person with dark long coat from camera G329 also appear in camera G424?"
        filters = extract_structured_filters(query)
        tokens = extract_text_tokens_for_sql(query, filters)
        self.assertIn("dark long coat", tokens)
        self.assertIn("dark coat", tokens)
        self.assertIn("long coat", tokens)
        self.assertIn("dark_coat", tokens)


if __name__ == "__main__":
    unittest.main()
