"""Convenience wrapper for the MEVID video-only evaluation harness.

This keeps the recommended command stable while the underlying test harness
continues to evolve.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MEVID video-only QA evaluation")
    parser.add_argument("--slot", default="13-50")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--video-dir", default="_data/mevid_slots")
    parser.add_argument("--reid-device", default="cpu")
    parser.add_argument("--with-clip-refine", action="store_true")
    parser.add_argument("--appearance-refine", dest="appearance_refine", action="store_true", default=True)
    parser.add_argument("--no-appearance-refine", dest="appearance_refine", action="store_false")
    parser.add_argument("--force-appearance-refine", action="store_true")
    args, passthrough = parser.parse_known_args()

    cmd = [
        sys.executable,
        str(ROOT / "tests" / "test_mevid_full.py"),
        "--slot",
        args.slot,
        "--limit",
        str(args.limit),
        "--video-dir",
        args.video_dir,
        "--reid-device",
        args.reid_device,
    ]
    if not args.with_clip_refine:
        cmd.append("--no-refine")
    if args.appearance_refine:
        cmd.append("--appearance-refine")
    if args.force_appearance_refine:
        cmd.append("--force-appearance-refine")
    cmd.extend(passthrough)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
