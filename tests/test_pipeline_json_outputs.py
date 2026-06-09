"""
Unit tests for video pipeline JSON output helpers.

- video_events_*: inject _run_pipeline mock; no cv2/YOLO required
- refined_*: run with mocked LLM when pydantic, langchain, cv2 are installed; else skip

From repo root:
  python -m unittest tests.test_pipeline_json_outputs -v
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch


def _refine_stack_available() -> bool:
    for mod in ("pydantic", "langchain_core", "langchain_openai", "cv2"):
        try:
            __import__(mod)
        except ImportError:
            return False
    return True


class TestVideoEventsJsonDicts(unittest.TestCase):
    def test_video_events_as_json_dicts_shape(self) -> None:
        from video.factory.pipeline_outputs import video_events_as_json_dicts

        mock_run = MagicMock(
            return_value=(
                [
                    {
                        "event_type": "motion_segment",
                        "track_id": 1,
                        "class_name": "car",
                        "start_time": 0.0,
                        "end_time": 1.0,
                    }
                ],
                [{"start_sec": 0.0, "end_sec": 2.0}],
                {
                    "video_path": "/data/test.mp4",
                    "fps": 25.0,
                    "total_frames": 100,
                    "num_tracks": 1,
                    "num_events": 1,
                    "num_clips": 1,
                    "tracker": "BoT-SORT+ReID",
                    "model": "yolov8n.pt",
                    "model_input": "n",
                    "conf": 0.25,
                    "iou": 0.25,
                },
            )
        )

        events_doc, clips_doc = video_events_as_json_dicts(
            "/data/test.mp4", _run_pipeline=mock_run, model_path="n"
        )

        self.assertIn("meta", events_doc)
        self.assertIn("events", events_doc)
        self.assertEqual(events_doc["meta"]["video_path"], "/data/test.mp4")
        self.assertEqual(len(events_doc["events"]), 1)
        self.assertIn("meta", clips_doc)
        self.assertIn("clip_segments", clips_doc)
        self.assertEqual(clips_doc["clip_segments"][0]["end_sec"], 2.0)

        json.dumps(events_doc, ensure_ascii=False)
        json.dumps(clips_doc, ensure_ascii=False)
        mock_run.assert_called_once()

    def test_video_events_as_json_strings(self) -> None:
        from video.factory.pipeline_outputs import video_events_as_json_strings

        mock_run = MagicMock(return_value=([], [], {"video_path": "/x.mp4", "fps": 1.0}))
        s1, s2 = video_events_as_json_strings("/x.mp4", indent=None, _run_pipeline=mock_run)
        self.assertIsInstance(s1, str)
        self.assertIsInstance(s2, str)
        self.assertIn("events", json.loads(s1))


@unittest.skipUnless(
    _refine_stack_available(),
    "Refine tests require: pydantic, langchain_core, langchain_openai, cv2",
)
class TestRefinedEventsJson(unittest.TestCase):
    @patch("video.factory.refinement_runner.refine_vector_events_with_llm")
    @patch("video.factory.refinement_runner.sample_frames_uniform")
    @patch("video.factory.refinement_runner.get_video_size")
    def test_refined_vector_dict_for_db(
        self,
        mock_size: MagicMock,
        mock_frames: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        from video.core.schema.refined_event_llm import VectorEvent, VectorEventsPayload
        from video.factory.pipeline_outputs import refined_events_as_dict, refined_events_as_json_str
        from video.factory.refinement_runner import RefineEventsConfig

        mock_size.return_value = (640, 480)
        mock_frames.return_value = [MagicMock(t_sec=0.5, jpg_base64="e30=")]
        mock_llm.return_value = VectorEventsPayload(
            video_id="test.mp4",
            clip_start_sec=0.0,
            clip_end_sec=2.0,
            events=[
                VectorEvent(
                    video_id="test.mp4",
                    clip_start_sec=0.0,
                    clip_end_sec=2.0,
                    start_time=0.0,
                    end_time=1.0,
                    object_type="car",
                    object_color="black",
                    appearance_notes="",
                    scene_zone="road",
                    event_text="Black car at the intersection",
                    keywords=["car", "road"],
                )
            ],
        )

        events_document = {
            "meta": {"video_path": "/fake/test.mp4"},
            "events": [
                {
                    "track_id": 1,
                    "class_name": "car",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "start_bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
                    "end_bbox_xyxy": [12.0, 10.0, 52.0, 50.0],
                }
            ],
        }
        clips_document = {"clip_segments": [{"start_sec": 0.0, "end_sec": 2.0}]}

        cfg = RefineEventsConfig(mode="vector", num_frames=4)
        out = refined_events_as_dict(events_document, clips_document, cfg)

        self.assertEqual(out["video_id"], "test.mp4")
        self.assertEqual(len(out["events"]), 1)
        self.assertEqual(out["events"][0]["object_type"], "car")
        json.dumps(out, ensure_ascii=False)

        s = refined_events_as_json_str(events_document, clips_document, cfg, indent=None)
        self.assertIn("intersection", s)

    @patch("video.factory.refinement_runner.refine_vector_events_with_llm")
    @patch("video.factory.refinement_runner.sample_frames_uniform")
    @patch("video.factory.refinement_runner.get_video_size")
    def test_refined_from_json_files_as_dict(
        self,
        mock_size: MagicMock,
        mock_frames: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        import tempfile
        from pathlib import Path

        from video.core.schema.refined_event_llm import VectorEventsPayload
        from video.factory.pipeline_outputs import refined_events_from_json_files_as_dict
        from video.factory.refinement_runner import RefineEventsConfig

        mock_size.return_value = (100, 100)
        mock_frames.return_value = [MagicMock(t_sec=0.0, jpg_base64="e30=")]
        mock_llm.return_value = VectorEventsPayload(
            video_id="f.mp4",
            clip_start_sec=0.0,
            clip_end_sec=1.0,
            events=[],
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "demo_events.json").write_text(
                json.dumps(
                    {
                        "meta": {"video_path": str(root / "f.mp4")},
                        "events": [
                            {
                                "track_id": 1,
                                "start_time": 0.0,
                                "end_time": 0.5,
                                "class_name": "person",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "f.mp4").write_bytes(b"")
            (root / "demo_clips.json").write_text(
                json.dumps({"clip_segments": [{"start_sec": 0.0, "end_sec": 1.0}]}),
                encoding="utf-8",
            )

            out = refined_events_from_json_files_as_dict(
                root / "demo_events.json",
                root / "demo_clips.json",
                RefineEventsConfig(mode="vector"),
            )
            self.assertEqual(out["video_id"], "f.mp4")


if __name__ == "__main__":
    unittest.main()
