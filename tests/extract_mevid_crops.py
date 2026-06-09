"""Selective MEVID-v1 bbox crop extractor.

The full bbox_test.tgz contains ~1.18 million JPEG crops (~13.9 GB compressed).
Extracting everything is impractical.  This script:

  1. Reads the annotation zip to discover all test tracklets
     (track_test_info.txt: start_frame, end_frame, identity, occasion, camera).
  2. Reads test_name.txt to build a filename index mapping
     (identity, occasion, camera, tracklet_id) → list[frame filenames].
  3. For each tracklet selects ``--num-crops`` evenly-spaced frames.
  4. Streams the tgz ONCE and extracts ONLY the ~N×num_crops target files.

Result:
  <out-dir>/
    <identity>/
      <identity>O<occasion>C<camera>T<tracklet>F<frame>.jpg
      ...

Runtime: ~15-30 min for the full tgz stream (sequential read).
Output size: ~200-500 MB (8,770 files × 25-50 KB).

Usage
-----
python tests/extract_mevid_crops.py \\
    --tgz   _data/mevid-v1-bbox-test.tgz \\
    --annot _data/mevid-v1-annotation-data.zip \\
    --out   _data/mevid-crops \\
    --num-crops 5
"""

from __future__ import annotations

import argparse
import re
import tarfile
import time
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"(\d{4})O(\d{3})C(\d{3})T(\d{3})F(\d{5})\.jpg", re.IGNORECASE
)


def _parse_mevid_filename(fname: str):
    """Parse '0205O005C505T000F00000.jpg' → (identity, occasion, camera, tracklet, frame)."""
    m = _FNAME_RE.search(fname)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


# ---------------------------------------------------------------------------
# Load annotation index
# ---------------------------------------------------------------------------

def _find_annotation_file(annot_root: Path, fragment: str) -> Path | None:
    if annot_root.is_file():
        return None
    return next((p for p in annot_root.rglob("*") if p.is_file() and fragment in p.name), None)


def _read_annotation_text(annot_path: Path, fragment: str) -> str:
    """Read an annotation text file from a zip archive or extracted directory."""
    if annot_path.is_file() and annot_path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(annot_path) as zf:
            names = zf.namelist()
            name = next(n for n in names if fragment in n)
            return zf.read(name).decode("utf-8")

    match = _find_annotation_file(annot_path, fragment)
    if match is None:
        raise FileNotFoundError(f"{fragment} not found under {annot_path}")
    return match.read_text(encoding="utf-8")


def load_tracklets_from_annot(annot_zip: Path) -> list[dict]:
    """Parse track_test_info.txt → list of tracklet dicts.

    Columns: start_frame  end_frame  identity  occasion  camera
    """
    raw = _read_annotation_text(annot_zip, "track_test_info")

    tracklets = []
    local_counts: dict[int, int] = defaultdict(int)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [float(v) for v in line.split()]
        if len(vals) < 5:
            continue
        identity = int(vals[2])
        occasion = int(vals[3])
        camera = int(vals[4])
        key = identity
        tracklet_id = local_counts[key]
        local_counts[key] += 1
        tracklets.append({
            "start_frame": int(vals[0]),
            "end_frame":   int(vals[1]),
            "identity":    identity,
            "occasion":    occasion,
            "camera":      camera,
            "tracklet_id":  tracklet_id,
        })
    return tracklets


def load_all_test_filenames(annot_zip: Path) -> list[str]:
    """Read test_name.txt → list of bare filenames."""
    raw = _read_annotation_text(annot_zip, "test_name")
    return [l.strip() for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Build target file set
# ---------------------------------------------------------------------------

def build_target_set(
    tracklets: list[dict],
    all_filenames: list[str],
    num_crops: int,
) -> set[str]:
    """Return the set of JPEG filenames (bare, without directory) to extract.

    Strategy per tracklet:
      1. Find all filenames that match (identity, occasion, camera).
      2. Group by tracklet_id.
      3. Pick one canonical tracklet_id (the one with most frames).
      4. From that tracklet select num_crops evenly-spaced frames.
    """
    print(f"Building filename index from {len(all_filenames):,} entries…")
    # Index: (identity, occasion, camera, tracklet_id) → sorted list of (frame_id, fname)
    idx: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
    for fname in all_filenames:
        parsed = _parse_mevid_filename(fname)
        if parsed is None:
            continue
        identity, occasion, camera, tracklet_id, frame_id = parsed
        key = (identity, occasion, camera, tracklet_id)
        idx[key].append((frame_id, fname))

    # Sort frame lists
    for key in idx:
        idx[key].sort(key=lambda x: x[0])

    print(f"Filename index: {len(idx):,} (identity, occasion, camera, tracklet) groups")

    targets: set[str] = set()
    missing = 0

    for tkl in tracklets:
        ident, occ, cam = tkl["identity"], tkl["occasion"], tkl["camera"]
        tracklet_id = tkl["tracklet_id"]
        key = (ident, occ, cam, tracklet_id)
        if key not in idx:
            missing += 1
            continue
        frames = idx[key]

        # Evenly spaced selection
        n = len(frames)
        step = max(1, n // num_crops)
        selected = frames[::step][:num_crops]
        for _, fname in selected:
            targets.add(fname)

    print(
        f"Target files selected: {len(targets):,}  "
        f"(tracklets with no match: {missing}/{len(tracklets)})"
    )
    return targets


# ---------------------------------------------------------------------------
# Stream-extract from tgz
# ---------------------------------------------------------------------------

def stream_extract(
    tgz_path: Path,
    targets: set[str],
    out_dir: Path,
) -> int:
    """Stream the tgz once and extract only files whose basename is in targets.

    Returns number of files extracted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining = set(targets)
    extracted = 0
    scanned = 0
    t0 = time.time()

    print(f"\nStreaming {tgz_path.name} (this may take 15-30 min)…")
    print(f"Looking for {len(remaining):,} target files\n")

    with tarfile.open(tgz_path, "r:gz") as tf:
        for member in tf:
            scanned += 1
            if scanned % 100_000 == 0:
                elapsed = time.time() - t0
                pct = 100 * extracted / max(1, len(targets))
                print(
                    f"  Scanned {scanned:,}  |  extracted {extracted:,}/{len(targets):,}"
                    f"  ({pct:.1f}%)  |  {elapsed:.0f}s elapsed"
                )

            if not member.isfile():
                continue

            # bare filename (no directory)
            basename = Path(member.name).name
            if basename not in remaining:
                continue

            # Determine output subdirectory from identity (first 4 chars)
            parsed = _parse_mevid_filename(basename)
            if parsed is None:
                continue
            identity = parsed[0]
            dest_dir = out_dir / f"{identity:04d}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / basename

            if dest_path.exists():
                remaining.discard(basename)
                extracted += 1
                continue

            # Extract file content
            f = tf.extractfile(member)
            if f is None:
                continue
            with open(dest_path, "wb") as out_f:
                out_f.write(f.read())

            remaining.discard(basename)
            extracted += 1

            if not remaining:
                print("  All target files extracted early — stopping stream.")
                break

    elapsed = time.time() - t0
    print(f"\nExtraction complete: {extracted:,} files in {elapsed:.1f}s")
    if remaining:
        print(f"WARNING: {len(remaining):,} files not found in tgz")
    return extracted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Selective MEVID crop extractor")
    p.add_argument("--tgz",   required=True, help="Path to mevid-v1-bbox-test.tgz")
    p.add_argument("--annot", required=True, help="Path to mevid-v1-annotation-data.zip")
    p.add_argument("--out",   required=True, help="Output directory for extracted crops")
    p.add_argument("--num-crops", type=int, default=5,
                   help="Frames to extract per tracklet (default: 5)")
    p.add_argument("--dry-run", action="store_true",
                   help="Build target set but do NOT extract (show statistics only)")
    args = p.parse_args()

    tgz_path   = Path(args.tgz)
    annot_path = Path(args.annot)
    out_dir    = Path(args.out)

    # Step 1: Load annotations
    print("=== Step 1: Load tracklet annotations ===")
    tracklets = load_tracklets_from_annot(annot_path)
    print(f"Test tracklets: {len(tracklets):,}")
    identities = len({t["identity"] for t in tracklets})
    cameras    = len({t["camera"]   for t in tracklets})
    print(f"Identities: {identities}   Cameras: {cameras}")

    # Step 2: Load filenames index
    print("\n=== Step 2: Load test_name index ===")
    all_filenames = load_all_test_filenames(annot_path)
    print(f"Total test frames: {len(all_filenames):,}")

    # Step 3: Build target set
    print("\n=== Step 3: Build target file set ===")
    targets = build_target_set(tracklets, all_filenames, num_crops=args.num_crops)
    estimated_mb = len(targets) * 0.035  # ~35 KB per crop
    print(f"Estimated output size: ~{estimated_mb:.0f} MB")

    if args.dry_run:
        print("\n[dry-run] Skipping extraction.")
        return

    # Step 4: Stream-extract
    print("\n=== Step 4: Stream-extract from tgz ===")
    n = stream_extract(tgz_path, targets, out_dir)
    print(f"\nDone. {n:,} crops saved to {out_dir}")
    print(f"\nNow run the evaluation:")
    print(f"  python tests/test_mevid_evaluation.py \\")
    print(f"      --annot-root {annot_path} \\")
    print(f"      --bbox-root  {out_dir} \\")
    print(f"      --out        results/mevid_eval.json \\")
    print(f"      --topo-out   results/mevid_topology.json")


if __name__ == "__main__":
    main()
