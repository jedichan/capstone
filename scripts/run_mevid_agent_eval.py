"""Convenience wrapper for the MEVID video + agent e2e evaluation harness."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MEVID video + agent e2e evaluation")
    parser.add_argument("--slot", default="13-50")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--sample-mode", default="balanced", choices=["balanced", "stratified", "first"])
    parser.add_argument("--top-k", type=int, default=5)
    args, passthrough = parser.parse_known_args()

    cmd = [
        sys.executable,
        str(ROOT / "tests" / "test_mevid_video_agent_e2e.py"),
        "--slot",
        args.slot,
        "--limit",
        str(args.limit),
        "--sample-mode",
        args.sample_mode,
        "--top-k",
        str(args.top_k),
    ]
    cmd.extend(passthrough)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
