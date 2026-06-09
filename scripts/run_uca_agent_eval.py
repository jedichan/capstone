"""Convenience wrapper for the UCA video + agent e2e evaluation harness."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_SEED_DIR = ROOT / "agent" / "test" / "generated" / "ucfcrime_video_events_vector_flat"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UCA video + agent e2e evaluation")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force-seed", action="store_true")
    parser.add_argument(
        "--video-result-json",
        default="",
        help="Use test_uca_unified.py predictions as agent seeds.",
    )
    parser.add_argument(
        "--force-video-seed",
        action="store_true",
        help="Regenerate video-derived vector seeds before running agent e2e.",
    )
    parser.add_argument(
        "--include-yolo",
        action="store_true",
        help="Append cached YOLO motion events to video-prediction seeds.",
    )
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Build agent seeds only from cached YOLO events.",
    )
    parser.add_argument(
        "--xlsx-cases",
        action="store_true",
        help="Use agent_test.xlsx cases instead of UCA transcript auto-cases.",
    )
    parser.add_argument("--include-sheets", nargs="*", default=["Part4"])
    args, passthrough = parser.parse_known_args()
    seed_dir = DEFAULT_VIDEO_SEED_DIR if (args.video_result_json or args.yolo_only) else None

    if seed_dir is not None:
        seed_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "generate_uca_video_vector_flat.py"),
            "--out-dir",
            str(seed_dir),
        ]
        if args.video_result_json:
            seed_cmd.extend(["--video-result-json", args.video_result_json])
        if args.force_video_seed:
            seed_cmd.append("--force")
        if args.include_yolo:
            seed_cmd.append("--include-yolo")
        if args.yolo_only:
            seed_cmd.append("--yolo-only")
        rc = subprocess.call(seed_cmd, cwd=ROOT)
        if rc != 0:
            return rc

    cmd = [
        sys.executable,
        str(ROOT / "tests" / "test_uca_video_agent_e2e.py"),
        "--limit",
        str(args.limit),
        "--top-k",
        str(args.top_k),
    ]
    if seed_dir is not None:
        cmd.extend(["--seed-dir", str(seed_dir)])
    if args.xlsx_cases:
        cmd.extend(["--include-sheets", *args.include_sheets])
    else:
        cmd.append("--auto-cases")
    if args.force_seed:
        cmd.append("--force-seed")
    cmd.extend(passthrough)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
