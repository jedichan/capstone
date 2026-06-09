"""Tier 2 zero-LLM scene attribute extraction.

Reads ``object_type``, ``scene_zone_en``, ``object_color_en`` from the
``episodic_events`` SQLite table, maps them to ``has_X`` boolean attributes,
computes corpus IDF, and writes a ``video_scene_attrs`` table + a vocab JSON
for the self-query node.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Map SQL field values → has_X attribute names
SQL_TO_ATTR: dict[str, dict[str, str]] = {
    "object_type": {
        "car": "has_car",
        "person": "has_person",
        "child": "has_child",
    },
    "scene_zone_en": {
        "road": "has_road",
        "store": "has_store",
        "room": "has_room",
        "bedside": "has_bedside",
    },
    "object_color_en": {
        "black": "has_black",
        "white": "has_white",
        "red": "has_red",
        "blue": "has_blue",
        "silver": "has_silver",
        "gray": "has_gray",
        "green": "has_green",
        "yellow": "has_yellow",
        "purple": "has_purple",
        "pink": "has_pink",
    },
}

# Query aliases for self_query LLM — maps attr_name → list of natural-language
# synonyms that might appear in a user query.
QUERY_ALIASES: dict[str, list[str]] = {
    "has_car": ["car", "vehicle", "automobile", "sedan", "truck", "driving"],
    "has_person": ["person", "people", "man", "woman", "individual", "someone", "customer"],
    "has_child": ["child", "kid", "baby", "infant"],
    "has_road": ["road", "street", "highway", "roadside", "intersection"],
    "has_store": ["store", "shop", "retail", "convenience store", "supermarket"],
    "has_room": ["room", "indoor room", "chamber", "office"],
    "has_bedside": ["bedside", "bed", "bedroom"],
    "has_black": ["black", "dark"],
    "has_white": ["white", "light colored"],
    "has_red": ["red"],
    "has_blue": ["blue"],
    "has_silver": ["silver", "silver-gray"],
    "has_gray": ["gray", "grey"],
    "has_green": ["green"],
    "has_yellow": ["yellow"],
    "has_purple": ["purple"],
    "has_pink": ["pink"],
}

CREATE_ATTRS_TABLE = """
CREATE TABLE IF NOT EXISTS video_scene_attrs (
    video_id TEXT NOT NULL,
    attr_name TEXT NOT NULL,
    idf REAL NOT NULL,
    PRIMARY KEY (video_id, attr_name)
);
"""


def build_scene_attrs(db_path: Path, vocab_json_path: Path | None = None) -> dict[str, Any]:
    """Extract per-video scene attributes, compute IDF, write to DB and vocab JSON.

    Returns a summary dict with counts for logging.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(CREATE_ATTRS_TABLE)
    conn.execute("DELETE FROM video_scene_attrs")

    # 1. Count total videos
    total_videos: int = conn.execute(
        "SELECT COUNT(DISTINCT video_id) FROM episodic_events"
    ).fetchone()[0]
    if total_videos == 0:
        conn.close()
        return {"total_videos": 0, "attrs_written": 0}

    # 2. Collect per-video attributes
    per_video: dict[str, set[str]] = {}
    for sql_field, mapping in SQL_TO_ATTR.items():
        rows = conn.execute(
            f"SELECT DISTINCT video_id, LOWER({sql_field}) as val "
            f"FROM episodic_events WHERE {sql_field} IS NOT NULL AND {sql_field} != '' "
            f"AND LOWER({sql_field}) != 'unknown'"
        ).fetchall()
        for video_id, val in rows:
            attr = mapping.get(val.strip())
            if attr:
                per_video.setdefault(video_id, set()).add(attr)

    # 3. Compute document frequency per attribute
    df: dict[str, int] = {}
    for attrs in per_video.values():
        for a in attrs:
            df[a] = df.get(a, 0) + 1

    # 4. Compute IDF: log(N/df) / log(N), normalised to [0,1]
    idf: dict[str, float] = {}
    log_n = math.log(total_videos)
    for attr_name, doc_freq in df.items():
        idf[attr_name] = math.log(total_videos / doc_freq) / log_n if log_n > 0 else 0.0

    # 5. Write to video_scene_attrs
    written = 0
    for video_id, attrs in per_video.items():
        for attr_name in attrs:
            conn.execute(
                "INSERT OR REPLACE INTO video_scene_attrs (video_id, attr_name, idf) VALUES (?, ?, ?)",
                (video_id, attr_name, idf.get(attr_name, 0.0)),
            )
            written += 1
    conn.commit()

    # 6. Export vocab JSON for self_query
    vocab = {}
    for attr_name, idf_val in sorted(idf.items()):
        vocab[attr_name] = {
            "idf": round(idf_val, 4),
            "query_aliases": QUERY_ALIASES.get(attr_name, [attr_name]),
        }

    if vocab_json_path:
        vocab_json_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_json_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    conn.close()
    _log.info(
        "scene_attrs: %d videos, %d attrs written, %d unique attr types (N=%d)",
        len(per_video), written, len(idf), total_videos,
    )
    return {
        "total_videos": total_videos,
        "videos_with_attrs": len(per_video),
        "attrs_written": written,
        "unique_attrs": len(idf),
        "vocab_path": str(vocab_json_path) if vocab_json_path else None,
    }


def load_vocab(vocab_json_path: Path) -> dict[str, dict[str, Any]]:
    """Load the scene attribute vocabulary for self_query prompts."""
    if not vocab_json_path.exists():
        return {}
    return json.loads(vocab_json_path.read_text(encoding="utf-8"))


def query_video_attrs(db_path: Path, video_id: str) -> dict[str, float]:
    """Return {attr_name: idf} for a single video (used at boost time)."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT attr_name, idf FROM video_scene_attrs WHERE video_id = ?", (video_id,)
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}
