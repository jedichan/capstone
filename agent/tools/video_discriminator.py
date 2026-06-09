"""Per-video discriminator summary for Tier 1 video-level retrieval.

Generates a concise, distinctive description of each video's spatial layout,
recurring objects, and unique features.  The summary is embedded into a dedicated
Chroma ``video`` collection so that the coarse stage of two-stage retrieval can
quickly identify the top-3 candidate videos before fine-grained chunk search.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

_log = logging.getLogger(__name__)

DISCRIMINATOR_SYSTEM_PROMPT = (
    "You are a video scene analyst. Given a list of events from a surveillance video, "
    "write a concise paragraph (3-5 sentences) describing what makes this video "
    "**visually distinct** from other surveillance videos. Focus on:\n"
    "- Spatial layout: where are counters, doors, windows, furniture, unique fixtures?\n"
    "- Recurring objects: what objects appear repeatedly (vehicles, specific furniture, signage)?\n"
    "- Unique features: anything that only appears in THIS video (e.g. fish tank, blue trash bin, "
    "  specific floor pattern, distinctive camera angle)?\n"
    "- Lighting and environment: indoor/outdoor, time of day, weather if visible.\n\n"
    "Use concrete nouns. Do NOT describe individual events or people's actions. "
    "Output ONLY the paragraph, no JSON, no markup."
)

DISCRIMINATOR_USER_PROMPT = (
    "Video: {video_id}\n"
    "Number of events: {event_count}\n"
    "Duration: {duration:.0f}s\n\n"
    "Event summaries:\n{event_summaries}"
)

# LLM call is expensive -- cache per video_id so repeated chroma_builder runs don't re-spend.
_DISCRIMINATOR_CACHE: dict[str, str] = {}


def _discriminator_cache_enabled() -> bool:
    raw = os.getenv("AGENT_DISCRIMINATOR_CACHE", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def generate_video_discriminator(
    *,
    video_id: str,
    events: list[dict[str, Any]],
    llm: Any,
) -> str:
    """Generate a distinctive summary for one video.

    Parameters
    ----------
    video_id:
        Normalised video identifier (e.g. ``Normal_Videos_924_x264``).
    events:
        Normalised per-event dicts as produced by
        ``ChromaIndexBuilder._normalize_events``.
    llm:
        A LangChain-compatible chat model.
    """
    if _discriminator_cache_enabled() and video_id in _DISCRIMINATOR_CACHE:
        return _DISCRIMINATOR_CACHE[video_id]

    duration = 0.0
    lines: list[str] = []
    for ev in events:
        st = ev.get("start_time")
        et = ev.get("end_time")
        if isinstance(et, (int, float)):
            duration = max(duration, float(et))
        txt = ev.get("event_text", "")
        if txt:
            lines.append(f"  [{st or '?'}s-{et or '?'}s] {txt}")

    event_summaries = "\n".join(lines[:30])  # cap at 30 to stay within context window
    if len(lines) > 30:
        event_summaries += f"\n  ... ({len(lines) - 30} more events omitted)"

    prompt = DISCRIMINATOR_USER_PROMPT.format(
        video_id=video_id,
        event_count=len(events),
        duration=duration,
        event_summaries=event_summaries,
    )

    try:
        response = llm.invoke(
            [SystemMessage(content=DISCRIMINATOR_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        text = str(response.content).strip() if hasattr(response, "content") else str(response).strip()
        # Strip any accidental markdown / code fences
        for fence in ("```json", "```"):
            text = text.replace(fence, "")
        text = text.strip()
    except Exception as exc:
        _log.warning("discriminator LLM failed for %s: %s", video_id, exc)
        text = f"Video {video_id}. {len(events)} surveillance events."

    if _discriminator_cache_enabled():
        _DISCRIMINATOR_CACHE[video_id] = text
    return text
