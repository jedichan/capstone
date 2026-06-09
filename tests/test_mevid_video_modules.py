"""Focused tests for MEVID video-module helpers added during e2e evaluation."""

from __future__ import annotations

import numpy as np
import cv2

from video.factory.appearance_refinement_runner import (
    _person_events_by_track,
    appearance_keywords_from_text,
    color_from_appearance,
    merge_refined_appearance,
)
from video.factory.person_crop_sampler import sample_person_crops_from_events
from video.indexing.search_enrichment import enrich_event_for_search


def test_appearance_keyword_and_color_normalization() -> None:
    text = "Light gray hoodie, dark pants, black hood up"

    assert color_from_appearance(text) == "light_grey"
    keywords = appearance_keywords_from_text(text)

    assert "light_grey_hoodie" in keywords
    assert "hood_up" in keywords
    assert "pants" in keywords


def test_merge_refined_appearance_appends_camera_events() -> None:
    base = {"per_camera": {"G328": {"events": [{"event_text": "base"}]}}}
    appearance = {"per_camera": {"G328": {"events": [{"event_text": "appearance"}]}}}

    merged = merge_refined_appearance(base, appearance)

    assert [e["event_text"] for e in merged["per_camera"]["G328"]["events"]] == [
        "base",
        "appearance",
    ]


def test_single_camera_person_events_group_by_track() -> None:
    events = [
        {"class_name": "person", "track_id": 1, "start_time": 0.0},
        {"class_name": "car", "track_id": 2, "start_time": 0.0},
        {"object_type": "person", "track_id": "1", "start_time": 1.0},
        {"object_type": "person", "track_id": 3, "start_time": 2.0},
    ]

    grouped = _person_events_by_track(events)

    assert sorted(grouped) == [1, 3]
    assert len(grouped[1]) == 2


def test_enrich_event_for_search_adds_retrieval_terms() -> None:
    event = {
        "event_text": "person_global_6 appeared in G339",
        "appearance_notes": "light grey hoodie, dark pants",
        "object_color": "unknown",
        "keywords": ["person"],
    }

    enriched = enrich_event_for_search(
        event,
        camera_id="G339",
        global_entity_id="person_global_6",
        trajectory_text="seen in cameras G508, G339",
    )

    assert "g339" in enriched["keywords"]
    assert "light_grey_hoodie" in enriched["keywords"]
    assert "cross_camera" in enriched["keywords"]
    assert "same_person" in enriched["keywords"]
    assert "seen in cameras G508, G339" in enriched["event_text"]


def test_sample_person_crops_from_events_reads_track_bbox(tmp_path) -> None:
    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (96, 96),
    )
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    frame[24:72, 32:64] = (220, 220, 220)
    for _ in range(3):
        writer.write(frame)
    writer.release()

    events = [{
        "class_name": "person",
        "track_id": 1,
        "start_time": 0.0,
        "end_time": 0.2,
        "start_bbox_xyxy": [30, 20, 66, 76],
        "end_bbox_xyxy": [30, 20, 66, 76],
    }]

    crops = sample_person_crops_from_events(video_path, events, max_tracks=1, crops_per_track=1)

    assert len(crops) == 1
    assert isinstance(crops[0], str)
    assert len(crops[0]) > 50
