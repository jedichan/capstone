"""MEVID-v1 evaluation script for cross-camera Re-ID + topology prior.

Dataset layout expected after extraction
-----------------------------------------
MEVID-v1 (Motion-based Extended Video Identity Dataset) provides:
  - ``mevid-v1-annotation-data/``  — identity labels, tracklet metadata
  - ``mevid-v1-bbox-test/``        — per-frame bounding-box annotations
                                     (format: sequence, cam, person_id, frame, x, y, w, h)

The script:
  1. Parses the annotation files → ground-truth cross-camera identity table.
  2. Extracts person crops from bbox annotations using our ``extract_person_crops``
     primitive (requires video files to be present under ``--video-root``).
  3. Encodes crops with our Re-ID embedder and runs cross-camera matching.
  4. Evaluates rank-1 / rank-5 / mAP (standard video Re-ID protocol).
  5. Trains and reports the ``CameraTopologyPrior`` learned from confirmed matches.

Usage
-----
python tests/test_mevid_evaluation.py \\
    --annot-root   _data/mevid-v1-annotation-data \\
    --bbox-root    _data/mevid-v1-bbox-test \\
    --video-root   _data/mevid-videos \\
    --reid-weights _model/osnet_ain_x1_0_msmt17.pth \\
    --out          results/mevid_eval.json \\
    --topo-out     results/mevid_topology.json

If ``--video-root`` is absent, the script falls back to "embedding-only" mode
using pre-extracted crops from the bbox tgz (JPEG crops expected at
``--bbox-root/<person_id>/c<cam_id>/<seq_id>/<frame>.jpg``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import zipfile
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation parsing helpers  (MEVID-v1 specific)
# ---------------------------------------------------------------------------
# MEVID-v1 annotation zip contains:
#   track_test_info.txt  — 5 columns (scientific notation):
#     start_frame  end_frame  identity_id  occasion_id  camera_id
#   query_IDX.txt — row indices (0-based) that are query tracklets
#   test_name.txt — all JPEG filenames (1.18 M entries)
#
# Crop directory structure (after extract_mevid_crops.py):
#   <crops_dir>/<identity:04d>/<identity:04d>O<occasion:03d>C<camera:03d>T<tkl:03d>F<frame:05d>.jpg

def _read_zip_txt(zip_path: Path, fname_fragment: str) -> str:
    """Read a text file from a zip or extracted annotation directory."""
    if zip_path.is_file() and zip_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(zip_path) as zf:
            match = next((n for n in zf.namelist() if fname_fragment in n), None)
            if match is None:
                raise FileNotFoundError(f"{fname_fragment} not found in {zip_path}")
            return zf.read(match).decode("utf-8", errors="replace")

    match = next(
        (p for p in zip_path.rglob("*") if p.is_file() and fname_fragment in p.name),
        None,
    )
    if match is None:
        raise FileNotFoundError(f"{fname_fragment} not found under {zip_path}")
    return match.read_text(encoding="utf-8", errors="replace")


def parse_mevid_tracklets(annot_zip: Path) -> tuple[list[dict[str, Any]], set[int]]:
    """Parse track_test_info.txt and query_IDX.txt from the annotation zip.

    Returns
    -------
    tracklets:
        List of dicts with keys:
          person_id (str "0205"), cam_id (str "505"), occasion_id (str "005"),
          start_frame (int), end_frame (int), tracklet_idx (int)
    query_indices:
        Set of row indices (0-based) that belong to the query set.
    """
    raw_info = _read_zip_txt(annot_zip, "track_test_info")
    raw_query = _read_zip_txt(annot_zip, "query_IDX")

    tracklets: list[dict[str, Any]] = []
    local_counts: dict[int, int] = defaultdict(int)
    for idx, line in enumerate(raw_info.splitlines()):
        line = line.strip()
        if not line:
            continue
        vals = [float(v) for v in line.split()]
        if len(vals) < 5:
            continue
        person_int = int(vals[2])
        occasion_int = int(vals[3])
        cam_int = int(vals[4])
        key = person_int
        local_tracklet_id = local_counts[key]
        local_counts[key] += 1
        tracklets.append({
            "tracklet_idx":  idx,
            "start_frame":   int(vals[0]),
            "end_frame":     int(vals[1]),
            "person_id":     f"{person_int:04d}",   # "0205"
            "occasion_id":   f"{occasion_int:03d}",   # "005"
            "cam_id":        f"{cam_int:03d}",   # "505"
            "local_tracklet_id": f"{local_tracklet_id:03d}",
        })

    query_indices: set[int] = set()
    for line in raw_query.splitlines():
        line = line.strip()
        if line:
            query_indices.add(int(float(line)))

    logger.info(
        "MEVID annotations: %d tracklets, %d query, %d gallery",
        len(tracklets), len(query_indices), len(tracklets) - len(query_indices),
    )
    return tracklets, query_indices


def _parse_mevid_annotation_zip(annot_zip: Path) -> dict[str, Any]:
    """Backward-compat wrapper: returns fps and basic metadata."""
    return {"fps_default": 25.0}  # MEVID-v1 is 25 fps


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------

def _extract_crop_from_video(
    video_path: Path,
    frame_idx: int,
    bbox: tuple[int, int, int, int],  # x, y, w, h
    max_edge: int = 256,
) -> np.ndarray | None:
    """Extract a single person crop from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    x, y, w, h = bbox
    x, y = max(0, x), max(0, y)
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    # Resize maintaining aspect ratio
    scale = max_edge / max(crop.shape[:2])
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return crop


def _load_crops_from_bbox_dir(
    crops_root: Path,
    tracklet: dict[str, Any],
    num_crops: int = 5,
) -> list[np.ndarray]:
    """Load JPEG crops from the MEVID-v1 crops directory.

    Directory structure produced by extract_mevid_crops.py::

        <crops_root>/<person_id>/<person_id>O<occasion>C<camera>T<tkl>F<frame>.jpg

    Matches any tracklet T-index for the given (person, occasion, camera).
    """
    person_id  = tracklet["person_id"]   # "0205"
    occasion   = tracklet["occasion_id"] # "005"
    cam_id     = tracklet["cam_id"]      # "505"

    person_dir = crops_root / person_id
    if not person_dir.exists():
        return []

    # Glob frames for this exact local tracklet T-number. T is local to
    # person_id, not the global row index.
    local_tracklet_id = tracklet.get("local_tracklet_id")
    if local_tracklet_id:
        prefix = f"{person_id}O{occasion}C{cam_id}T{local_tracklet_id}F"
        jpgs = sorted(person_dir.glob(f"{prefix}*.jpg"))
    else:
        prefix = f"{person_id}O{occasion}C{cam_id}T"
        jpgs = sorted(person_dir.glob(f"{prefix}*F*.jpg"))
    if not jpgs:
        return []

    step = max(1, len(jpgs) // num_crops)
    selected = jpgs[::step][:num_crops]
    crops = []
    for jp in selected:
        img = cv2.imread(str(jp))
        if img is not None:
            crops.append(img)
    return crops


# ---------------------------------------------------------------------------
# Re-ID evaluation metrics
# ---------------------------------------------------------------------------

def _compute_ap(query_id: str, ranked_ids: list[str]) -> float:
    """Average Precision for one query."""
    n_rel = 0
    ap = 0.0
    for i, pid in enumerate(ranked_ids, start=1):
        if pid == query_id:
            n_rel += 1
            ap += n_rel / i
    if n_rel == 0:
        return 0.0
    return ap / max(1, ranked_ids.count(query_id))


def compute_reid_metrics(
    query_embeddings: list,   # (person_id, cam_id, emb[, tracklet_idx])
    gallery_embeddings: list,
    ranks: list[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Rank-K CMC and mAP.

    Standard protocol: query vs gallery, exclude same-camera same-identity.

    Parameters
    ----------
    query_embeddings:
        List of (person_id, cam_id, L2-normalised embedding[, tracklet_idx]).
    gallery_embeddings:
        Same format.
    ranks:
        CMC cutoffs to compute.
    """
    q_ids = np.array([q[0] for q in query_embeddings])
    q_cams = np.array([q[1] for q in query_embeddings])
    q_embs = np.stack([q[2] for q in query_embeddings])

    g_ids  = np.array([g[0] for g in gallery_embeddings])
    g_cams = np.array([g[1] for g in gallery_embeddings])
    g_embs = np.stack([g[2] for g in gallery_embeddings])

    # Cosine similarity matrix (both already L2-normalised)
    sim_matrix = q_embs @ g_embs.T  # (n_query, n_gallery)

    ap_list: list[float] = []
    rank_hits: dict[int, int] = {k: 0 for k in ranks}

    for qi in range(len(query_embeddings)):
        qid = q_ids[qi]
        qcam = q_cams[qi]

        sims = sim_matrix[qi].copy()
        # Mask same-camera same-identity (junk)
        junk_mask = (g_ids == qid) & (g_cams == qcam)
        sims[junk_mask] = -2.0  # push to end

        sorted_idx = np.argsort(-sims)
        valid_g_ids = g_ids[sorted_idx]
        valid_g_cams = g_cams[sorted_idx]

        # Remove junk from ranking list for evaluation
        ranked = [
            (gid, gcam)
            for gid, gcam in zip(valid_g_ids, valid_g_cams)
            if not (gid == qid and gcam == qcam)
        ]
        ranked_ids = [r[0] for r in ranked]

        # CMC
        for k in ranks:
            if qid in ranked_ids[:k]:
                rank_hits[k] += 1

        # AP
        ap_list.append(_compute_ap(qid, ranked_ids))

    n_q = len(query_embeddings)
    metrics: dict[str, float] = {f"rank_{k}": rank_hits[k] / n_q for k in ranks}
    metrics["mAP"] = float(np.mean(ap_list))
    return metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_tracklet_embeddings(
    tracklets: list[dict[str, Any]],
    bbox_root: Path,
    video_root: Path | None,
    embedder: Any,
    num_crops: int = 5,
    fps: float = 25.0,
) -> list[tuple[str, str, np.ndarray, int]]:
    """Encode every tracklet into a single Re-ID embedding.

    Returns list of (person_id, cam_id, embedding, tracklet_idx).
    """
    results: list[tuple[str, str, np.ndarray, int]] = []
    n_no_crops = 0
    for tkl in tracklets:
        pid = tkl["person_id"]
        cam = tkl["cam_id"]
        tkl_idx = tkl.get("tracklet_idx", -1)

        # Load pre-extracted JPEG crops from crops directory
        crops: list[np.ndarray] = _load_crops_from_bbox_dir(bbox_root, tkl, num_crops=num_crops)

        if not crops:
            n_no_crops += 1
            continue

        emb = embedder.embed_crops(crops)
        mean_emb = emb.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm

        results.append((pid, cam, mean_emb, tkl_idx))

    logger.info(
        "Encoded %d / %d tracklets (no crops: %d)",
        len(results), len(tracklets), n_no_crops,
    )
    return results


def split_query_gallery(
    embeddings: list[tuple[str, str, np.ndarray]],
    query_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list, list]:
    """Split embeddings into query / gallery by person_id stratification."""
    rng = np.random.default_rng(seed)
    by_person: dict[str, list] = defaultdict(list)
    for item in embeddings:
        by_person[item[0]].append(item)

    query, gallery = [], []
    for pid, items in by_person.items():
        # Ensure each person appears in both sets
        cams = list({item[1] for item in items})
        if len(cams) < 2:
            gallery.extend(items)
            continue
        rng.shuffle(cams)
        n_q = max(1, int(len(cams) * query_fraction))
        query_cams = set(cams[:n_q])
        for item in items:
            if item[1] in query_cams:
                query.append(item)
            else:
                gallery.append(item)

    return query, gallery


def split_query_gallery_mevid(
    embeddings: list[tuple[str, str, np.ndarray]],
    tracklets: list[dict[str, Any]],
    query_indices: set[int],
) -> tuple[list, list]:
    """Split using official MEVID query_IDX.txt assignments.

    Each embedding is tagged with its tracklet_idx.  Embeddings whose
    tracklet_idx is in query_indices become the query set; all others gallery.
    Falls back to camera-stratified split if no query_indices provided.
    """
    if not query_indices:
        return split_query_gallery(embeddings)

    # Build tracklet_idx → person_id, cam_id lookup
    idx_map = {t["tracklet_idx"]: t for t in tracklets}

    query: list = []
    gallery: list = []
    for emb_tuple in embeddings:
        pid, cam, emb, *rest = emb_tuple
        tkl_idx = rest[0] if rest else None
        if tkl_idx is not None and tkl_idx in query_indices:
            query.append((pid, cam, emb))
        else:
            gallery.append((pid, cam, emb))

    # Ensure no empty sets
    if not query or not gallery:
        return split_query_gallery(embeddings)
    return query, gallery


def learn_topology_from_tracklets(
    tracklets: list[dict[str, Any]],
    embeddings: list[tuple[str, str, np.ndarray]],
    cameras: list[str],
    fps: float = 25.0,
    max_transit_sec: float = 600.0,
) -> Any:
    """Build a CameraTopologyPrior from ground-truth tracklet transitions.

    Uses confirmed same-identity cross-camera appearances (GT transitions).
    """
    from video.core.models.camera_topology import CameraTopologyPrior

    prior = CameraTopologyPrior(cameras=cameras, max_transit_sec=max_transit_sec)

    # Group tracklets by person_id
    by_person: dict[str, list[dict]] = defaultdict(list)
    for tkl in tracklets:
        by_person[tkl["person_id"]].append(tkl)

    transitions: list[tuple[str, str, float]] = []
    for pid, tkls in by_person.items():
        sorted_tkls = sorted(tkls, key=lambda t: t["start_frame"])
        for i in range(len(sorted_tkls) - 1):
            a = sorted_tkls[i]
            b = sorted_tkls[i + 1]
            if a["cam_id"] == b["cam_id"]:
                continue
            end_sec_a = a["end_frame"] / fps
            start_sec_b = b["start_frame"] / fps
            delta_t = start_sec_b - end_sec_a
            if 0 < delta_t <= max_transit_sec:
                transitions.append((a["cam_id"], b["cam_id"], delta_t))

    prior.observe_batch(transitions)
    logger.info(
        "Topology prior learned from %d GT transitions across %d persons",
        len(transitions), len(by_person),
    )
    return prior


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_mevid_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    annot_root = Path(args.annot_root)
    bbox_root = Path(args.bbox_root)
    video_root = Path(args.video_root) if args.video_root else None
    out_path = Path(args.out)
    topo_out_path = Path(args.topo_out) if args.topo_out else None

    # ------------------------------------------------------------------ #
    # 1. Parse annotations                                                 #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 1: Parse annotations ===")

    # annot_root can be a .zip file or extracted directory
    if str(annot_root).endswith(".zip") and annot_root.is_file():
        annot_zip = annot_root
    else:
        # Look for zip inside the directory
        candidates = list(annot_root.glob("*.zip"))
        annot_zip = candidates[0] if candidates else annot_root

    fps = 25.0  # MEVID-v1 fixed fps

    # Parse MEVID-v1 tracklets from annotation zip
    tracklets, query_indices = parse_mevid_tracklets(annot_zip)

    if not tracklets:
        logger.error("No tracklets found in annotation zip: %s", annot_zip)
        return {"error": "no tracklets found", "annot_zip": str(annot_zip)}

    cameras = sorted({t["cam_id"] for t in tracklets})
    n_identities = len({t["person_id"] for t in tracklets})
    logger.info("Cameras: %s | Identities: %d | Tracklets: %d", cameras, n_identities, len(tracklets))

    # ------------------------------------------------------------------ #
    # 2. Encode with Re-ID embedder                                        #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 2: Build Re-ID embeddings ===")
    from video.core.models.reid_embedder import ReIDEmbedder

    embedder = ReIDEmbedder(
        config_file=args.reid_config,
        weights=args.reid_weights,
        device=args.device,
    )

    tracklet_embeddings = build_tracklet_embeddings(
        tracklets=tracklets,
        bbox_root=bbox_root,
        video_root=video_root,
        embedder=embedder,
        num_crops=args.num_crops,
        fps=fps,
    )

    if len(tracklet_embeddings) < 10:
        logger.error(
            "Too few embeddings (%d). Ensure crops are accessible.", len(tracklet_embeddings)
        )
        return {"error": "insufficient embeddings", "n_encoded": len(tracklet_embeddings)}

    # ------------------------------------------------------------------ #
    # 3. Re-ID evaluation (Rank-1/5, mAP)                                 #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 3: Re-ID evaluation ===")
    # Use official MEVID query/gallery split (query_IDX.txt)
    query, gallery = split_query_gallery_mevid(tracklet_embeddings, tracklets, query_indices)
    logger.info("Query: %d  Gallery: %d", len(query), len(gallery))

    reid_metrics = compute_reid_metrics(query, gallery, ranks=[1, 5, 10])
    logger.info(
        "Re-ID | Rank-1=%.3f  Rank-5=%.3f  mAP=%.3f",
        reid_metrics["rank_1"], reid_metrics["rank_5"], reid_metrics["mAP"],
    )

    # ------------------------------------------------------------------ #
    # 4. Camera topology prior                                             #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 4: Camera topology prior ===")
    topology_prior = learn_topology_from_tracklets(
        tracklets=tracklets,
        embeddings=tracklet_embeddings,
        cameras=cameras,
        fps=fps,
        max_transit_sec=args.max_transit_sec,
    )
    topo_table = topology_prior.transition_table()
    top_pairs = topology_prior.most_connected_pairs(top_k=10)

    logger.info("Top cross-camera transition pairs:")
    for pair in top_pairs[:5]:
        mean_t = pair["mean_transit_sec"]
        logger.info(
            "  %s → %s: n=%d, mean_Δt=%.1fs, fitted=%s",
            pair["cam_a"], pair["cam_b"], pair["n_observations"],
            mean_t if mean_t is not None else float("nan"),
            pair["fitted"],
        )

    if topo_out_path:
        topology_prior.save(topo_out_path)
        logger.info("Topology prior saved to %s", topo_out_path)

    # ------------------------------------------------------------------ #
    # 5. Topology-augmented Re-ID evaluation                               #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 5: Topology-augmented matching ===")
    # Simulate: for each query tracklet, rerank gallery using topo score
    # Score = 0.55 * cosine + 0.45 * topology_score(qcam, gcam, Δt)
    # Δt estimated from GT tracklet time boundaries

    # Build (person_id, cam_id) → (start_sec, end_sec) lookup
    tracklet_time: dict[tuple[str, str], tuple[float, float]] = {}
    for t in tracklets:
        key = (t["person_id"], t["cam_id"])
        # Keep earliest start / latest end when multiple tracklets per person+cam
        s = t["start_frame"] / fps
        e = t["end_frame"] / fps
        if key not in tracklet_time:
            tracklet_time[key] = (s, e)
        else:
            prev_s, prev_e = tracklet_time[key]
            tracklet_time[key] = (min(prev_s, s), max(prev_e, e))

    def _find_time(person_id: str, cam_id: str) -> tuple[float, float]:
        return tracklet_time.get((person_id, cam_id), (0.0, 0.0))

    q_ids_arr = [q[0] for q in query]
    q_cams_arr = [q[1] for q in query]
    q_embs_arr = np.stack([q[2] for q in query])
    g_ids_arr = [g[0] for g in gallery]
    g_cams_arr = [g[1] for g in gallery]
    g_embs_arr = np.stack([g[2] for g in gallery])

    cosine_matrix = q_embs_arr @ g_embs_arr.T  # (n_q, n_g)

    ap_topo_list: list[float] = []
    rank_topo_hits: dict[int, int] = {1: 0, 5: 0, 10: 0}

    for qi in range(len(query)):
        qid, qcam = q_ids_arr[qi], q_cams_arr[qi]
        q_start, q_end = _find_time(qid, qcam)

        topo_scores = np.zeros(len(gallery))
        for gi in range(len(gallery)):
            gcam = g_cams_arr[gi]
            g_start, _ = _find_time(g_ids_arr[gi], gcam)
            delta_t = abs(g_start - q_end)
            topo_scores[gi] = topology_prior.score(qcam, gcam, delta_t)

        combined = 0.55 * cosine_matrix[qi] + 0.45 * topo_scores
        # Mask same-cam same-id junk
        for gi in range(len(gallery)):
            if g_ids_arr[gi] == qid and g_cams_arr[gi] == qcam:
                combined[gi] = -2.0

        sorted_idx = np.argsort(-combined)
        ranked_ids = [g_ids_arr[i] for i in sorted_idx
                      if not (g_ids_arr[i] == qid and g_cams_arr[i] == qcam)]

        for k in [1, 5, 10]:
            if qid in ranked_ids[:k]:
                rank_topo_hits[k] += 1
        ap_topo_list.append(_compute_ap(qid, ranked_ids))

    n_q = len(query)
    topo_reid_metrics = {f"rank_{k}": rank_topo_hits[k] / n_q for k in [1, 5, 10]}
    topo_reid_metrics["mAP"] = float(np.mean(ap_topo_list))

    logger.info(
        "Topo-augmented Re-ID | Rank-1=%.3f  Rank-5=%.3f  mAP=%.3f",
        topo_reid_metrics["rank_1"], topo_reid_metrics["rank_5"], topo_reid_metrics["mAP"],
    )

    delta_rank1 = topo_reid_metrics["rank_1"] - reid_metrics["rank_1"]
    delta_map = topo_reid_metrics["mAP"] - reid_metrics["mAP"]
    logger.info(
        "Δ(topo vs baseline) | ΔRank-1=%+.3f  ΔmAP=%+.3f", delta_rank1, delta_map
    )

    # ------------------------------------------------------------------ #
    # 6. Save results                                                      #
    # ------------------------------------------------------------------ #
    result = {
        "dataset": "MEVID-v1",
        "n_cameras": len(cameras),
        "cameras": cameras,
        "n_identities": n_identities,
        "n_tracklets": len(tracklets),
        "n_encoded_tracklets": len(tracklet_embeddings),
        "n_query": len(query),
        "n_gallery": len(gallery),
        "reid_baseline": reid_metrics,
        "reid_topology_augmented": topo_reid_metrics,
        "topology_delta": {"rank_1": delta_rank1, "mAP": delta_map},
        "topology_top_pairs": top_pairs,
        "topology_table": topo_table,
        "fps_used": fps,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)

    # Console summary
    print("\n" + "=" * 65)
    print("MEVID-v1 Evaluation Summary")
    print("=" * 65)
    print(f"  Cameras        : {len(cameras)}  |  Identities: {n_identities}")
    print(f"  Tracklets      : {len(tracklets)}  |  Encoded: {len(tracklet_embeddings)}")
    print()
    print("  Re-ID (Re-ID only)")
    print(f"    Rank-1 = {reid_metrics['rank_1']:.4f}  Rank-5 = {reid_metrics['rank_5']:.4f}  mAP = {reid_metrics['mAP']:.4f}")
    print()
    print("  Re-ID + Topology Prior")
    print(f"    Rank-1 = {topo_reid_metrics['rank_1']:.4f}  Rank-5 = {topo_reid_metrics['rank_5']:.4f}  mAP = {topo_reid_metrics['mAP']:.4f}")
    print()
    print(f"  Δ Rank-1 = {delta_rank1:+.4f}   Δ mAP = {delta_map:+.4f}")
    print("=" * 65)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEVID-v1 cross-camera Re-ID + topology evaluation")
    p.add_argument("--annot-root", required=True,
                   help="Path to mevid-v1-annotation-data directory (or .zip)")
    p.add_argument("--bbox-root", required=True,
                   help="Path to mevid-v1-bbox-test directory (or .tgz)")
    p.add_argument("--video-root", default=None,
                   help="Optional: root containing MEVID video files for crop extraction")
    p.add_argument("--reid-weights", default=None,
                   help="Path to Re-ID model weights (.pth)")
    p.add_argument("--reid-config", default=None,
                   help="Path to Re-ID model config YAML")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for Re-ID inference")
    p.add_argument("--num-crops", type=int, default=5,
                   help="Crops per tracklet for embedding")
    p.add_argument("--max-transit-sec", type=float, default=600.0,
                   help="Hard cutoff for transit time in topology prior")
    p.add_argument("--out", default="results/mevid_eval.json",
                   help="Output JSON path")
    p.add_argument("--topo-out", default="results/mevid_topology.json",
                   help="Path to save learned topology prior JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_mevid_evaluation(args)
