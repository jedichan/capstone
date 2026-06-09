"""
scripts/download_mevid_slots.py
--------------------------------
Downloads the MEVID-v1 video clips for all 6 annotated time slots.

Base URL:
  https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-11/{HH}/{stem}.avi

Files are saved to:  _data/mevid_slots/{stem}.avi

Usage:
    python scripts/download_mevid_slots.py
    python scripts/download_mevid_slots.py --dry-run      # print URLs only
    python scripts/download_mevid_slots.py --slot 16-35   # one slot only
    python scripts/download_mevid_slots.py --workers 4    # parallel downloads
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

# ── All video stems per slot  (from merge_and_generate.SLOT_CAMERAS) ─────────
SLOT_CAMERAS: dict[str, dict[str, str]] = {
    "11-20": {
        "G329": "2018-03-11.11-20-00.11-25-00.admin.G329.r13",
        "G330": "2018-03-11.11-20-00.11-25-00.school.G330.r13",
        "G336": "2018-03-11.11-20-00.11-25-00.school.G336.r13",
        "G419": "2018-03-11.11-20-00.11-25-00.school.G419.r13",
        "G420": "2018-03-11.11-20-00.11-25-00.school.G420.r13",
        "G421": "2018-03-11.11-20-00.11-25-00.school.G421.r13",
        "G423": "2018-03-11.11-20-00.11-25-00.school.G423.r13",
        "G508": "2018-03-11.11-20-00.11-25-00.bus.G508.r13",
    },
    "11-55": {
        "G299": "2018-03-11.11-55-00.12-00-00.school.G299.r13",
        "G328": "2018-03-11.11-55-00.12-00-00.school.G328.r13",
        "G330": "2018-03-11.11-55-00.12-00-00.school.G330.r13",
        "G419": "2018-03-11.11-55-00.12-00-00.school.G419.r13",
        "G420": "2018-03-11.11-55-00.12-00-00.school.G420.r13",
        "G506": "2018-03-11.11-55-00.12-00-00.bus.G506.r13",
        "G508": "2018-03-11.11-55-00.12-00-00.bus.G508.r13",
    },
    "13-50": {
        "G328": "2018-03-11.13-50-01.13-55-01.school.G328.r13",
        "G329": "2018-03-11.13-50-01.13-55-01.admin.G329.r13",
        "G339": "2018-03-11.13-50-01.13-55-01.school.G339.r13",
        "G421": "2018-03-11.13-50-01.13-55-01.school.G421.r13",
        "G424": "2018-03-11.13-50-01.13-55-01.school.G424.r13",
        "G506": "2018-03-11.13-50-01.13-55-01.bus.G506.r13",
        "G508": "2018-03-11.13-50-01.13-55-01.bus.G508.r13",
    },
    "14-20": {
        "G328": "2018-03-11.14-20-01.14-25-01.school.G328.r13",
        "G339": "2018-03-11.14-20-01.14-25-01.school.G339.r13",
        "G419": "2018-03-11.14-20-01.14-25-01.school.G419.r13",
        "G421": "2018-03-11.14-20-01.14-25-01.school.G421.r13",
        "G423": "2018-03-11.14-20-01.14-25-01.school.G423.r13",
        "G505": "2018-03-11.14-20-01.14-25-01.bus.G505.r13",
        "G506": "2018-03-11.14-20-01.14-25-01.bus.G506.r13",
        "G508": "2018-03-11.14-20-01.14-25-01.bus.G508.r13",
    },
    "16-20": {
        "G326": "2018-03-11.16-20-01.16-25-01.admin.G326.r13",
        "G328": "2018-03-11.16-20-01.16-25-01.school.G328.r13",
        "G329": "2018-03-11.16-20-01.16-25-01.admin.G329.r13",
        "G336": "2018-03-11.16-20-01.16-25-01.school.G336.r13",
        "G419": "2018-03-11.16-20-01.16-25-01.school.G419.r13",
        "G420": "2018-03-11.16-20-01.16-25-01.school.G420.r13",
        "G506": "2018-03-11.16-20-01.16-25-01.bus.G506.r13",
        "G508": "2018-03-11.16-20-01.16-25-01.bus.G508.r13",
    },
    "16-35": {
        "G326": "2018-03-11.16-35-01.16-40-01.admin.G326.r13",
        "G328": "2018-03-11.16-35-01.16-40-01.school.G328.r13",
        "G329": "2018-03-11.16-35-01.16-40-01.admin.G329.r13",
        "G336": "2018-03-11.16-35-01.16-40-01.school.G336.r13",
        "G339": "2018-03-11.16-35-01.16-40-01.school.G339.r13",
        "G419": "2018-03-11.16-35-01.16-40-01.school.G419.r13",
        "G420": "2018-03-11.16-35-01.16-40-01.school.G420.r13",
        "G423": "2018-03-11.16-35-01.16-40-00.school.G423.r13",
        "G506": "2018-03-11.16-35-01.16-40-01.bus.G506.r13",
        "G638": "2018-03-11.16-35-01.16-40-01.school.G638.r13",
    },
}

BASE_URL = "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-11"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "_data" / "mevid_slots"


def stem_to_url(stem: str) -> str:
    # Extract start-time hour from stem: 2018-03-11.16-35-01.… → "16"
    parts = stem.split(".")
    start_hhmm = parts[1]          # e.g. "16-35-01"
    hour = start_hhmm.split("-")[0]  # "16"
    return f"{BASE_URL}/{hour}/{stem}.avi"


def download_one(stem: str, out_dir: Path, dry_run: bool) -> tuple[str, str]:
    """Download one file.  Returns (stem, status)."""
    url = stem_to_url(stem)
    dest = out_dir / f"{stem}.avi"

    if dry_run:
        return stem, f"DRY  {url}"

    if dest.exists() and dest.stat().st_size > 100_000:
        return stem, f"SKIP (already exists, {dest.stat().st_size // 1024} KB)"

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        tmp = dest.with_suffix(".avi.tmp")
        urlretrieve(url, tmp)
        tmp.rename(dest)
        size_kb = dest.stat().st_size // 1024
        elapsed = time.time() - t0
        return stem, f"OK   {size_kb:,} KB  ({elapsed:.1f}s)"
    except Exception as e:
        return stem, f"FAIL {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",  default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--slot",     default="",   help="Only download one slot, e.g. 16-35")
    ap.add_argument("--dry-run",  action="store_true", help="Print URLs, don't download")
    ap.add_argument("--workers",  type=int, default=3,  help="Parallel download workers (default 3)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    slots = {args.slot: SLOT_CAMERAS[args.slot]} if args.slot else SLOT_CAMERAS
    if args.slot and args.slot not in SLOT_CAMERAS:
        print(f"Unknown slot '{args.slot}'. Valid: {list(SLOT_CAMERAS)}")
        sys.exit(1)

    # Build flat list of stems to download
    stems: list[str] = []
    seen: set[str] = set()
    for slot, cam_map in slots.items():
        for cam, stem in cam_map.items():
            if stem not in seen:
                stems.append(stem)
                seen.add(stem)

    print(f"Files to download: {len(stems)}  →  {out_dir}")
    if args.dry_run:
        print("(DRY RUN — URLs only)\n")
        for stem in stems:
            print(stem_to_url(stem))
        return

    ok = fail = skip = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, stem, out_dir, False): stem for stem in stems}
        total = len(futures)
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            stem, status = fut.result()
            done += 1
            tag = status[:4]
            if tag == "OK  ":   ok   += 1
            elif tag == "SKIP": skip += 1
            else:               fail += 1
            print(f"[{done:2d}/{total}] {stem}  →  {status}")

    print(f"\nDone.  OK={ok}  SKIP={skip}  FAIL={fail}")
    if fail:
        print("NOTE: Some downloads failed.  Check if the video stems are correct "
              "by running with --dry-run and verifying URLs against the MEVID URL list.")


if __name__ == "__main__":
    main()
