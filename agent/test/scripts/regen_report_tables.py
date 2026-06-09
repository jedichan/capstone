#!/usr/bin/env python3
"""CLI wrapper: regenerate ``REPORT_TABLES.md`` without re-running the graph.

Usage::

    cd /path/to/Capstone
    python agent/test/scripts/regen_report_tables.py --output-dir agent/test/generated/<run_tag>/

See ``agent/test/eval_report_tables.py`` for the implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_TEST_DIR = _ROOT / "agent" / "test"
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from eval_report_tables import main  # noqa: E402

if __name__ == "__main__":
    main()
