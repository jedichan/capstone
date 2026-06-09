"""Structured protocols for events, clips, and LLM outputs."""

from video.core.schema.refined_event_llm import (
    RefinedAllClipsPayload,
    RefinedEntity,
    RefinedEvent,
    RefinedEventsPayload,
    VectorAllClipsPayload,
    VectorEvent,
    VectorEventsPayload,
)

__all__ = [
    "RefinedEntity",
    "RefinedEvent",
    "RefinedEventsPayload",
    "RefinedAllClipsPayload",
    "VectorEvent",
    "VectorEventsPayload",
    "VectorAllClipsPayload",
]
