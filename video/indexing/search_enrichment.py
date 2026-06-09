"""Search-oriented enrichment for video event records.

These helpers keep the external event schema stable while making ``event_text``,
``appearance_notes``, and ``keywords`` more useful for text/vector retrieval.
"""

from __future__ import annotations

import re
from typing import Any

COLOR_ALIASES = {
    "gray": "grey",
    "light gray": "light_grey",
    "light grey": "light_grey",
    "dark gray": "dark_grey",
    "dark grey": "dark_grey",
    "silver gray": "silver_grey",
    "silver grey": "silver_grey",
}

COLORS = [
    "red", "blue", "green", "black", "white", "yellow", "orange",
    "grey", "gray", "brown", "purple", "pink", "dark", "light", "beige",
]

APPEARANCE_TERMS = [
    "hoodie", "jacket", "coat", "shirt", "t-shirt", "pants", "trousers",
    "jeans", "shorts", "skirt", "dress", "bag", "backpack", "handbag",
    "hat", "cap", "hood", "scarf", "fur", "collar", "long", "sleeve",
]

UNKNOWN_TEXT = {"", "unknown", "n/a", "none", "null"}


def is_unknown_text(value: object) -> bool:
    return str(value or "").strip().lower() in UNKNOWN_TEXT


def append_unique(items: list[str], extra: list[str], max_len: int = 16) -> list[str]:
    seen = {str(item).strip().lower() for item in items}
    for token in extra:
        token = str(token).strip().lower().replace(" ", "_")
        if token and token != "unknown" and token not in seen:
            items.append(token)
            seen.add(token)
        if len(items) >= max_len:
            break
    return items


def appearance_keywords(text: str) -> list[str]:
    tl = text.lower().replace("-", " ")
    out: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.strip().lower().replace(" ", "_").replace("gray", "grey")
        if token and token != "unknown" and token not in seen:
            out.append(token)
            seen.add(token)

    for phrase, alias in COLOR_ALIASES.items():
        if phrase in tl:
            add(alias)
    for color in COLORS:
        if re.search(rf"\b{re.escape(color)}\b", tl):
            add(COLOR_ALIASES.get(color, color))
    for term in APPEARANCE_TERMS:
        term_re = term.replace("-", r"[-\s]?")
        if re.search(rf"\b{term_re}\b", tl):
            add(term.replace("-", "_"))

    if ("light grey" in tl or "light gray" in tl) and "hoodie" in tl:
        add("light_grey_hoodie")
    if "beige" in tl and ("jacket" in tl or "coat" in tl):
        add("beige_jacket")
    if ("dark" in tl or "black" in tl) and "coat" in tl:
        add("dark_coat")
    if ("dark" in tl or "black" in tl) and "hood" in tl:
        add("hood_up")
    if "fur" in tl and ("hood" in tl or "collar" in tl):
        add("fur_trimmed_hood")
    return out


def normalize_keywords(value: object, fallback_text: str = "", max_k: int = 8) -> list[str]:
    out: list[str] = []
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, str) and value.strip():
        candidates = re.split(r"[,;/\s]+", value.strip())
    else:
        candidates = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", fallback_text.lower())

    seen: set[str] = set()
    for item in candidates:
        token = str(item).strip().lower().replace(" ", "_").replace("-", "_")
        if token and token != "unknown" and token not in seen:
            out.append(token)
            seen.add(token)
        if len(out) >= max_k:
            break
    return out


def enrich_event_for_search(
    event: dict[str, Any],
    *,
    camera_id: str = "",
    global_entity_id: str = "",
    trajectory_text: str = "",
    max_keywords: int = 16,
) -> dict[str, Any]:
    """Return an event copy with RAG-friendly text and keyword fields."""
    enriched = dict(event)
    event_text = str(enriched.get("event_text") or enriched.get("description") or "").strip()
    appearance = str(enriched.get("appearance_notes") or "").strip()
    color = str(enriched.get("object_color") or "").strip().lower().replace(" ", "_")

    if is_unknown_text(appearance):
        appearance = ""
    if color == "unknown":
        color = ""

    if global_entity_id and trajectory_text and trajectory_text not in event_text:
        event_text = (
            f"{event_text} cross-camera same-person candidate {global_entity_id}; "
            f"{trajectory_text}."
        ).strip()

    keywords = normalize_keywords(enriched.get("keywords"), " ".join([event_text, appearance]))
    if camera_id and camera_id.lower() not in keywords:
        keywords.insert(0, camera_id.lower())
    keywords = append_unique(
        keywords,
        appearance_keywords(" ".join([appearance, event_text, color])),
        max_len=max_keywords,
    )
    if global_entity_id:
        keywords = append_unique(
            keywords,
            ["cross_camera", "same_person", global_entity_id.lower()],
            max_len=max_keywords,
        )
        if appearance and "same-person candidate" not in appearance:
            appearance = f"{appearance}; same-person candidate {global_entity_id}".strip()

    enriched["event_text"] = event_text
    enriched["appearance_notes"] = appearance or enriched.get("appearance_notes") or ""
    enriched["object_color"] = color or enriched.get("object_color") or "unknown"
    enriched["keywords"] = keywords[:max_keywords]
    return enriched
