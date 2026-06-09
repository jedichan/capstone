"""Full-corpus BM25 index over the SQLite ``episodic_events`` table.

This replaces the legacy "subset BM25" that re-tokenised the documents already
returned by Chroma vector search. The legacy approach made IDF effectively
constant (the document frequency was computed only over the ~30 docs that
vector retrieval had already pre-selected), so it never contributed real
lexical signal.

The implementation here:

* Loads every event row from SQLite once per process (cached by ``(db_path,
  mtime)``) so subsequent calls are O(query_terms * postings) instead of an
  O(corpus) read.
* Builds a textbook BM25Okapi index over the concatenation of the
  ``event_text_en``, ``event_summary_en`` and ``appearance_notes_en`` fields,
  reusing the SQL token stopwords / plural-singular map from
  ``retrieval_contracts`` so SQL and BM25 channels see the same vocabulary.
* Supports optional metadata pre-pruning (e.g. by ``video_id`` /
  ``object_color_en``) before BM25 scoring, mirroring what Chroma does with its
  ``where`` clause.
* Returns rows in the same shape as ``ChromaGateway.search`` so the hybrid
  branch can RRF-fuse vector and BM25 ranks without further normalisation.

The legacy ``_bm25_scores`` helpers in ``db_access.py`` and
``llamaindex_adapter.py`` are removed in the same change-set; this module is
their full-corpus replacement.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

# Reuse the SQL-side stopword / singularization knobs so the lexical channel
# stays in lock-step with ``extract_text_tokens_for_sql``.
from node.retrieval_contracts import (
    _PLURAL_TO_SINGULAR,
    SQL_TOKEN_STOPWORDS,
)


_DEFAULT_TABLE_NAME = "episodic_events"

# BM25Okapi defaults; tunable but the textbook values work well for short
# event-summary corpora.
_BM25_K1 = 1.5
_BM25_B = 0.75

# Minimum token length retained after lower-casing. Shorter tokens (e.g. "a",
# "to", "of") are pure noise for our domain.
_MIN_TOKEN_LEN = 3

# Columns we project from SQLite to assemble per-document text + metadata.
_TEXT_COLUMNS: tuple[str, ...] = (
    "event_text_en",
    "event_summary_en",
    "appearance_notes_en",
)
_META_COLUMNS: tuple[str, ...] = (
    "event_id",
    "video_id",
    "track_id",
    "entity_hint",
    "start_time",
    "end_time",
    "object_type",
    "object_color_en",
    "scene_zone_en",
    "event_text_en",
    "event_summary_en",
    "appearance_notes_en",
)


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str | None, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase + alphanumeric tokenization with stopword + length filtering.

    ``drop_stopwords`` is True by default for query tokenisation; it should be
    False when indexing documents so that stopwords still contribute to the
    document length / avgdl statistics in a sensible way (we still drop them
    after counting because empty postings would inflate IDF for nothing).
    """

    if not text:
        return []
    raw = text.lower()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.findall(raw):
        if len(match) < _MIN_TOKEN_LEN:
            continue
        token = _PLURAL_TO_SINGULAR.get(match, match)
        if drop_stopwords and token in SQL_TOKEN_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _row_text(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for col in _TEXT_COLUMNS:
        value = row.get(col)
        if value:
            parts.append(str(value))
    return " ".join(parts)


@dataclass
class _BM25Document:
    """In-memory representation of a single indexed event."""

    doc_idx: int
    metadata: dict[str, Any]
    token_counts: Counter[str]
    doc_len: int


class BM25Index:
    """Full-corpus BM25 index over the SQLite events table.

    Parameters
    ----------
    db_path:
        Path to the SQLite database produced by ``agent/db/sqlite_builder.py``.
    table_name:
        Defaults to ``episodic_events`` (matches the schema in
        ``agent/db/schema.py``).

    Notes
    -----
    The first call to :meth:`search` for a given ``(db_path, mtime)`` builds
    the index synchronously; subsequent calls reuse the cached index until the
    underlying file changes. Because the cache is keyed by mtime, rebuilding
    the SQLite database (e.g. ``agent/db/manage_graph_db.py reset``)
    automatically invalidates stale state on the next query.
    """

    _CACHE_LOCK = threading.Lock()
    _CACHE: dict[str, dict[str, Any]] = {}

    def __init__(self, db_path: str | os.PathLike[str], *, table_name: str = _DEFAULT_TABLE_NAME) -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name

    # ------------------------------------------------------------------ public
    def search(
        self,
        query: str,
        *,
        top_k: int = 30,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not (query or "").strip():
            return []
        index = self._load_index()
        if not index["documents"]:
            return []

        q_terms = _tokenize(query, drop_stopwords=True)
        if not q_terms:
            return []
        # Deduplicate query terms to avoid double-counting when a user repeats a
        # word; weighting is already encoded by the term postings.
        q_terms = list(dict.fromkeys(q_terms))

        candidate_indices = self._apply_filters(index, filters)
        if candidate_indices is not None and not candidate_indices:
            return []

        df = index["df"]
        idf = index["idf"]
        avgdl = index["avgdl"] or 1.0
        documents: list[_BM25Document] = index["documents"]

        # Score by walking only the postings of query terms that exist in the
        # corpus -- skip empty postings to avoid the O(N) per-term penalty.
        candidate_set: set[int] | None = candidate_indices
        scores: dict[int, float] = {}
        for term in q_terms:
            if term not in df:
                continue
            postings = index["postings"].get(term)
            if not postings:
                continue
            term_idf = idf[term]
            for doc_idx, term_freq in postings:
                if candidate_set is not None and doc_idx not in candidate_set:
                    continue
                doc = documents[doc_idx]
                denom = term_freq + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (doc.doc_len / avgdl))
                if denom <= 0:
                    continue
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_idf * (term_freq * (_BM25_K1 + 1.0)) / denom

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: int(top_k)]
        out: list[dict[str, Any]] = []
        for doc_idx, score in ranked:
            doc = documents[doc_idx]
            row = dict(doc.metadata)
            row["_bm25"] = float(score)
            row["_distance"] = None  # BM25 has no cosine distance; fusion uses ranks.
            row["_source_type"] = "bm25"
            row.setdefault(
                "event_summary_en",
                row.get("event_summary_en") or row.get("event_text_en"),
            )
            row["event_text"] = row.get("event_text_en") or row.get("event_summary_en")
            out.append(row)
        return out

    # ------------------------------------------------------------------ utils
    def stats(self) -> dict[str, Any]:
        """Diagnostic snapshot of the loaded index (used by tests)."""

        index = self._load_index()
        return {
            "doc_count": len(index["documents"]),
            "vocab_size": len(index["df"]),
            "avgdl": index["avgdl"],
            "db_path": str(self.db_path),
        }

    @classmethod
    def clear_cache(cls) -> None:
        """Drop the process-level cache; primarily for tests."""

        with cls._CACHE_LOCK:
            cls._CACHE.clear()

    # ------------------------------------------------------------- internals
    def _cache_key(self) -> str | None:
        if not self.db_path.exists():
            return None
        try:
            mtime = self.db_path.stat().st_mtime_ns
        except OSError:
            return None
        return f"{self.db_path.resolve()}::{self.table_name}::{mtime}"

    def _load_index(self) -> dict[str, Any]:
        key = self._cache_key()
        if key is None:
            # File missing -> empty index (callers degrade gracefully).
            return {
                "documents": [],
                "df": Counter(),
                "idf": {},
                "avgdl": 0.0,
                "postings": {},
            }
        with self._CACHE_LOCK:
            cached = self._CACHE.get(key)
            if cached is not None:
                return cached
        index = self._build_index()
        with self._CACHE_LOCK:
            self._CACHE[key] = index
        return index

    def _build_index(self) -> dict[str, Any]:
        select_cols = ", ".join(dict.fromkeys(_META_COLUMNS))
        sql = f"SELECT {select_cols} FROM {self.table_name}"
        documents: list[_BM25Document] = []
        df: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = {}
        total_len = 0
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(sql).fetchall()
            except sqlite3.OperationalError:
                return {
                    "documents": [],
                    "df": Counter(),
                    "idf": {},
                    "avgdl": 0.0,
                    "postings": {},
                }

        for doc_idx, row in enumerate(rows):
            row_dict = {key: row[key] for key in row.keys()}
            tokens = _tokenize(_row_text(row_dict), drop_stopwords=True)
            doc_len = len(tokens)
            total_len += doc_len
            counts = Counter(tokens)
            documents.append(_BM25Document(doc_idx=doc_idx, metadata=row_dict, token_counts=counts, doc_len=doc_len))
            for term, freq in counts.items():
                df[term] += 1
                postings.setdefault(term, []).append((doc_idx, freq))

        n = len(documents)
        avgdl = (total_len / n) if n else 0.0
        # Standard BM25Okapi IDF; +1 floor to keep weights non-negative for
        # terms that hit nearly every doc.
        idf = {term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

        return {
            "documents": documents,
            "df": df,
            "idf": idf,
            "avgdl": avgdl,
            "postings": postings,
        }

    @staticmethod
    def _apply_filters(index: dict[str, Any], filters: Mapping[str, Any] | None) -> set[int] | None:
        if not filters:
            return None
        documents: list[_BM25Document] = index["documents"]
        keep: set[int] = set()
        # ``filters`` mirrors the metadata filter dict passed to ChromaGateway:
        # ``{"object_color_en": "red", "video_id": "Arrest050_x264"}``. We do
        # case-insensitive equality on string fields and direct equality
        # otherwise -- intentionally lenient because callers already pass
        # canonicalised values from ``extract_structured_filters``.
        for doc in documents:
            ok = True
            for key, expected in filters.items():
                actual = doc.metadata.get(key)
                if isinstance(expected, str) and isinstance(actual, str):
                    if actual.lower() != expected.lower():
                        ok = False
                        break
                else:
                    if actual != expected:
                        ok = False
                        break
            if ok:
                keep.add(doc.doc_idx)
        return keep


def _normalize_event_id(event_id: Any) -> Any:
    """Normalize event_id for cross-branch matching (P0-1).

    SQL branch returns ``int`` (e.g. 42), Chroma/BM25 may return ``str``
    (e.g. "42").  Coerce numeric values to ``int`` so they compare equal
    regardless of source.
    """
    if event_id is None:
        return None
    try:
        return int(float(event_id))
    except (ValueError, TypeError):
        return event_id


def reciprocal_rank_fuse(
    ranked_lists: Iterable[Iterable[Mapping[str, Any]]],
    *,
    top_k: int = 10,
    rrf_k: int = 60,
    id_key: str = "event_id",
) -> list[dict[str, Any]]:
    """Reciprocal-Rank Fusion over multiple ranked lists.

    Each input list is expected to be already sorted in descending relevance.
    Items are deduplicated by ``id_key`` (defaults to ``event_id`` which is
    stable across SQL/Chroma/BM25 sources because it comes from the same
    SQLite event table). The returned dict contains the union of all source
    rows merged shallowly (later sources fill in missing keys), plus the
    fused ``_fused_rank`` and ``_fused_score`` telemetry.
    """

    fused: dict[Any, dict[str, Any]] = {}
    rank_records: dict[Any, list[tuple[int, int]]] = {}
    for source_idx, ranked in enumerate(ranked_lists):
        for rank, item in enumerate(ranked, start=1):
            ident = _normalize_event_id(item.get(id_key))
            if ident is None:
                ident = id(item)
            if ident not in fused:
                fused[ident] = dict(item)
                rank_records[ident] = []
            else:
                # Backfill any missing keys from later sources without
                # overwriting earlier ones.
                for k, v in item.items():
                    fused[ident].setdefault(k, v)
            rank_records[ident].append((source_idx, rank))

    scored: list[tuple[float, Any]] = []
    for ident, records in rank_records.items():
        score = sum(1.0 / (rrf_k + rank) for _, rank in records)
        scored.append((score, ident))
    scored.sort(key=lambda item: item[0], reverse=True)

    out: list[dict[str, Any]] = []
    for fused_rank, (score, ident) in enumerate(scored[: int(top_k)], start=1):
        row = fused[ident]
        row["_fused_rank"] = fused_rank
        row["_fused_score"] = float(score)
        row["_source_ranks"] = list(rank_records[ident])
        out.append(row)
    return out
