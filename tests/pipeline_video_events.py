"""
Convenience entry to run the video pipeline locally (same as repo-root pipeline_video_events.py).

From repo root (set your video path):
  python tests/pipeline_video_events.py /path/to/video.mp4 -m n

First positional arg is `video`; other flags match coordinator CLI (--tracker, --out-dir, ...).
"""

from video.factory.coordinator import cli_run_video_events

if __name__ == "__main__":
    cli_run_video_events()
