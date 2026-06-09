"""Build a slot-only test database from existing seeds (14-20 only).

Usage:
    python mevid_fastapi\\build_meva_1420_test_db.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mevid_fastapi.pipeline_service import ingest_from_seeds
from agent.db.sqlite_builder import SQLiteBuildConfig, SQLiteDatabaseBuilder


RUNTIME_ROOT = ROOT / "data" / "runtime"
SRC_NAMESPACE = "mevid_extended"
DST_NAMESPACE = "meva_1420_test"
SLOT_KEY = "14-20"

SRC_SEEDS = RUNTIME_ROOT / SRC_NAMESPACE / "seeds"
DST_BASE = RUNTIME_ROOT / DST_NAMESPACE
DST_SEEDS = DST_BASE / "seeds"
DST_RUNTIME = DST_BASE / "runtime"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sqlite-only",
        action="store_true",
        help="Build only SQLite DB (skip Chroma embeddings).",
    )
    args = ap.parse_args()

    if not SRC_SEEDS.exists():
        raise FileNotFoundError(f"Source seeds not found: {SRC_SEEDS}")

    DST_SEEDS.mkdir(parents=True, exist_ok=True)
    DST_RUNTIME.mkdir(parents=True, exist_ok=True)

    # Clean old slot seeds in destination.
    for old in DST_SEEDS.glob("*_events_vector_flat.json"):
        old.unlink()

    selected = sorted(SRC_SEEDS.glob(f"*{SLOT_KEY}*_events_vector_flat.json"))
    if not selected:
        raise ValueError(f"No seeds matched slot '{SLOT_KEY}' in {SRC_SEEDS}")

    for fp in selected:
        shutil.copy2(fp, DST_SEEDS / fp.name)

    print(f"[slot-db] copied {len(selected)} seed files for slot {SLOT_KEY}")
    for fp in selected:
        print(f"  - {fp.name}")

    if args.sqlite_only:
        sqlite_path = DST_RUNTIME / "episodic_events.sqlite"
        seed_files = sorted(DST_SEEDS.glob("*_events_vector_flat.json"))
        sqlite_result = SQLiteDatabaseBuilder(
            SQLiteBuildConfig(
                db_path=sqlite_path,
                reset_existing=True,
                generate_init_prompt=False,
            )
        ).build(seed_files=seed_files)
        print("\n[slot-db] sqlite-only build complete")
        print(f"  namespace   : {DST_NAMESPACE}")
        print(f"  sqlite_path : {sqlite_path}")
        print(f"  sqlite_rows : {sqlite_result.get('inserted_rows', 0)}")
        print("  chroma_path : (not built)")
        return

    result = ingest_from_seeds(
        seeds_dir=str(DST_SEEDS),
        namespace=DST_NAMESPACE,
        output_base=str(DST_RUNTIME),
        reset_existing=True,
    )

    print("\n[slot-db] build complete")
    print(f"  namespace   : {result['namespace']}")
    print(f"  sqlite_path : {result['sqlite_path']}")
    print(f"  chroma_path : {result['chroma_path']}")
    print(f"  sqlite_rows : {result['sqlite_rows']}")
    print(f"  chroma_child: {result['chroma_child_records']}")
    print(f"  chroma_ge   : {result['chroma_ge_records']}")


if __name__ == "__main__":
    main()
