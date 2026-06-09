"""
Legacy module name: re-exports refinement types and runner; CLI lives in coordinator (lazy import avoids YOLO/cv2).

Use: video.factory.coordinator.run_refine_events or refinement_runner.run_refine_events_from_files.
"""

from __future__ import annotations

from typing import Sequence

from video.factory.refinement_runner import RefineEventsConfig, run_refine_events_from_files


def cli_run_refine_events(argv: Sequence[str] | None = None) -> None:
    """Delegate to coordinator (loads cv2-heavy deps only when running CLI)."""
    from video.factory.coordinator import cli_run_refine_events as _cli

    _cli(argv)


main = cli_run_refine_events

__all__ = [
    "RefineEventsConfig",
    "run_refine_events_from_files",
    "cli_run_refine_events",
    "main",
]


if __name__ == "__main__":
    cli_run_refine_events()
