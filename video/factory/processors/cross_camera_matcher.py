"""Cross-camera person matching: time filtering → Re-ID embedding → global assignment (greedy + Union-Find).

Scoring formula (with topology prior injected):
  score = topology_weight_reid  * cosine_similarity
        + topology_weight_topo  * topology_score(cam_a, cam_b, Δt)

Without a topology prior the system falls back to the legacy heuristic:
  score = 0.70 * cosine + 0.30 * time_window_score(Δt)
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from video.core.schema.multi_camera import (
    CameraAppearance,
    CameraResult,
    CrossCameraConfig,
    GlobalEntity,
    MatchVerification,
)

if TYPE_CHECKING:
    from video.core.models.reid_embedder import ReIDEmbedder
    from video.core.models.camera_topology import CameraTopologyPrior

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Time constraint helpers
# ------------------------------------------------------------------

def _time_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Overlap length (seconds) of two intervals; 0 if none."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _time_gap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Gap (seconds) between two intervals; 0 if overlapping."""
    if _time_overlap(a_start, a_end, b_start, b_end) > 0:
        return 0.0
    return min(abs(b_start - a_end), abs(a_start - b_end))


def passes_time_constraint(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    config: CrossCameraConfig,
) -> bool:
    """Whether two tracks satisfy time constraints from max_transition_sec / min_overlap_sec."""
    a_s, a_e = track_a["start_time"], track_a["end_time"]
    b_s, b_e = track_b["start_time"], track_b["end_time"]

    overlap = _time_overlap(a_s, a_e, b_s, b_e)
    if overlap > 0:
        return overlap >= config.min_overlap_sec

    gap = _time_gap(a_s, a_e, b_s, b_e)
    return gap <= config.max_transition_sec


# ------------------------------------------------------------------
# Candidate pair generation
# ------------------------------------------------------------------

_MIN_TRACK_DURATION_SEC = 2.0  # tracks shorter than this have noisy embeddings


def _person_tracks(cam: CameraResult) -> list[dict[str, Any]]:
    """Return person tracks that are long enough to have reliable Re-ID embeddings."""
    return [
        t for t in cam.tracks
        if t.get("class_name") == "person"
        and (float(t.get("end_time", 0)) - float(t.get("start_time", 0)))
            >= _MIN_TRACK_DURATION_SEC
    ]


def build_candidate_pairs(
    per_camera: list[CameraResult],
    config: CrossCameraConfig,
) -> list[tuple[CameraResult, dict[str, Any], CameraResult, dict[str, Any]]]:
    """Enumerate all cross-camera (person) track pairs that pass time filtering."""
    pairs: list[tuple[CameraResult, dict, CameraResult, dict]] = []
    for cam_i, cam_j in itertools.combinations(per_camera, 2):
        tracks_i = _person_tracks(cam_i) if config.person_only else cam_i.tracks
        tracks_j = _person_tracks(cam_j) if config.person_only else cam_j.tracks
        for ti in tracks_i:
            for tj in tracks_j:
                # Reject cross-camera pairs whose time windows overlap by more than
                # a small tolerance.  A person cannot be in two different cameras
                # simultaneously; overlaps beyond the tolerance indicate distinct
                # individuals.  A short tolerance (≤ 5 s) accommodates tracker
                # boundary jitter when a person is visible at the edge of two
                # camera fields-of-view at the same moment.
                a_s = float(ti["start_time"]); a_e = float(ti["end_time"])
                b_s = float(tj["start_time"]); b_e = float(tj["end_time"])
                overlap = min(a_e, b_e) - max(a_s, b_s)
                if overlap > config.cross_camera_overlap_tolerance_sec:
                    continue  # genuine simultaneous presence → different people
                if passes_time_constraint(ti, tj, config):
                    pairs.append((cam_i, ti, cam_j, tj))
    return pairs


# ------------------------------------------------------------------
# Similarity scoring
# ------------------------------------------------------------------

def score_candidate_pairs(
    pairs: list[tuple[CameraResult, dict, CameraResult, dict]],
    embedder: "ReIDEmbedder",
    config: CrossCameraConfig,
    topology_prior: "CameraTopologyPrior | None" = None,
) -> list[tuple[CameraResult, dict, CameraResult, dict, float]]:
    """Cross-camera score combining Re-ID cosine with temporal topology.

    With *topology_prior* injected (recommended)::

        score = config.topology_weight_reid * cosine
              + config.topology_weight_topo * topology_score(cam_a, cam_b, Δt)

    Legacy fallback (no topology prior)::

        score = 0.70 * cosine + 0.30 * time_window_score(Δt)

    Parameters
    ----------
    pairs:
        Candidate pairs from :func:`build_candidate_pairs`.
    embedder:
        Re-ID model (used only to verify embedding availability; embeddings
        are pre-computed and stored in ``CameraResult.person_embeddings``).
    config:
        Matching hyper-parameters, including ``topology_weight_*`` fields.
    topology_prior:
        Optional :class:`~video.core.models.camera_topology.CameraTopologyPrior`.
        When provided, replaces the hardcoded time-window heuristic with a
        learned per-pair transit-time distribution.
    """

    def _legacy_time_window_score(gap_sec: float) -> float:
        if gap_sec <= 30.0:
            return 1.0
        if gap_sec <= 120.0:
            return 0.7
        if gap_sec <= 300.0:
            return 0.3
        return 0.0

    scored: list[tuple[CameraResult, dict, CameraResult, dict, float]] = []
    for cam_i, ti, cam_j, tj in pairs:
        emb_i = cam_i.person_embeddings.get(ti["track_id"])
        emb_j = cam_j.person_embeddings.get(tj["track_id"])
        if emb_i is None or emb_j is None:
            continue
        cosine = float(emb_i @ emb_j)
        # Hard gate: reject pairs below embedding_threshold regardless of topology score
        if cosine < config.embedding_threshold:
            continue
        gap = _time_gap(
            float(ti["start_time"]), float(ti["end_time"]),
            float(tj["start_time"]), float(tj["end_time"]),
        )

        if topology_prior is not None:
            # Topology-aware scoring: directed A→B and B→A, take the best
            topo_ab = topology_prior.score(cam_i.camera_id, cam_j.camera_id, gap)
            topo_ba = topology_prior.score(cam_j.camera_id, cam_i.camera_id, gap)
            topo_score = max(topo_ab, topo_ba)
            score = (
                config.topology_weight_reid * cosine
                + config.topology_weight_topo * topo_score
            )
        else:
            # Legacy fallback
            tw = _legacy_time_window_score(gap)
            score = 0.70 * cosine + 0.30 * tw

        scored.append((cam_i, ti, cam_j, tj, score))
    scored.sort(key=lambda x: x[4], reverse=True)
    return scored


# ------------------------------------------------------------------
# Global assignment (greedy or Hungarian)
# ------------------------------------------------------------------

def _greedy_assign(
    scored: list[tuple[CameraResult, dict, CameraResult, dict, float]],
    threshold: float,
) -> list[tuple[str, int, str, int, float]]:
    """Multi-way merge: keep pairs above threshold for Union-Find.

    Unlike one-to-one greedy matching, a track may appear in multiple high-score pairs.
    When one person appears on 3+ cameras, pairs cam1↔cam2, cam1↔cam3, cam2↔cam3 can all be kept;
    Union-Find merges them into one GlobalEntity.

    Within a camera pair (cam_a, cam_b), each track keeps only its best opponent
    to reduce many-to-one errors inside that pair.

    Returns list of (cam_id_a, track_id_a, cam_id_b, track_id_b, score).
    """
    # best_for_cam_pair[(cam_a, tid_a, cam_b)] = best (tid_b, score) for that directed triple
    best: dict[tuple[str, int, str], tuple[int, float]] = {}
    for cam_i, ti, cam_j, tj, score in scored:
        if score < threshold:
            break
        cid_i, tid_i = cam_i.camera_id, ti["track_id"]
        cid_j, tid_j = cam_j.camera_id, tj["track_id"]
        key_ij = (cid_i, tid_i, cid_j)
        key_ji = (cid_j, tid_j, cid_i)
        prev_ij = best.get(key_ij)
        if prev_ij is None or score > prev_ij[1]:
            best[key_ij] = (tid_j, score)
        prev_ji = best.get(key_ji)
        if prev_ji is None or score > prev_ji[1]:
            best[key_ji] = (tid_i, score)

    # Dedup undirected pairs
    seen: set[frozenset[tuple[str, int]]] = set()
    assignments: list[tuple[str, int, str, int, float]] = []
    for (cid_a, tid_a, cid_b), (tid_b, score) in best.items():
        pair_key = frozenset([(cid_a, tid_a), (cid_b, tid_b)])
        if pair_key in seen:
            continue
        seen.add(pair_key)
        assignments.append((cid_a, tid_a, cid_b, tid_b, score))
    return assignments


def _validate_entity_members(
    members: list[tuple[str, int]],
    track_lookup: dict[tuple[str, int], dict[str, Any]],
) -> list[list[tuple[str, int]]]:
    """(Kept for compatibility — no longer called directly.)"""
    return [members]


def _build_global_entities(
    assignments: list[tuple[str, int, str, int, float]],
    per_camera: list[CameraResult],
    config: CrossCameraConfig | None = None,
) -> list[GlobalEntity]:
    """Build global entities using score-ordered greedy merging.

    Replaces Union-Find to prevent chain-merging.  Processes assignments from
    highest score to lowest; a merge is allowed only when the combined entity
    would still have AT MOST ONE track per camera.  This enforces the physical
    constraint (one person cannot be in two places in the same camera at once)
    during construction rather than as a post-processing split.

    This means:
    - High-confidence matches are committed first.
    - A noisy "bridge" track cannot chain two unrelated people together once
      one of its cameras is already claimed by a high-score match.
    """
    # Build track_lookup: for merged tracks (same track_id after same-cam stitching),
    # record the merged time span (earliest start, latest end) so entity appearances
    # have accurate timestamps.
    track_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for cam in per_camera:
        for t in cam.tracks:
            key = (cam.camera_id, int(t["track_id"]))
            existing = track_lookup.get(key)
            if existing is None:
                track_lookup[key] = dict(t)  # shallow copy so we can mutate times
            else:
                # Expand the time span to cover all fragments merged into this id
                existing["start_time"] = min(float(existing["start_time"]), float(t["start_time"]))
                existing["end_time"]   = max(float(existing["end_time"]),   float(t["end_time"]))

    confidence_map: dict[tuple[str, int, str, int], float] = {}
    for cam_a, tid_a, cam_b, tid_b, score in assignments:
        confidence_map[(cam_a, tid_a, cam_b, tid_b)] = score

    # assignments is already sorted highest-score first (from _greedy_assign)
    # entity_of: track_key → entity_id
    entity_of: dict[tuple[str, int], int] = {}
    # entity_cams: entity_id → {cam_id: track_key}
    entity_cams: dict[int, dict[str, tuple[str, int]]] = {}
    next_eid: list[int] = [0]

    def new_entity(key_a: tuple[str, int], key_b: tuple[str, int]) -> int:
        eid = next_eid[0]; next_eid[0] += 1
        entity_cams[eid] = {key_a[0]: key_a, key_b[0]: key_b}
        entity_of[key_a] = eid
        entity_of[key_b] = eid
        return eid

    def add_to_entity(eid: int, key: tuple[str, int]) -> None:
        entity_cams[eid][key[0]] = key
        entity_of[key] = eid

    def merge_entities(eid_dst: int, eid_src: int) -> None:
        for cam, key in entity_cams[eid_src].items():
            entity_cams[eid_dst][cam] = key
            entity_of[key] = eid_dst
        del entity_cams[eid_src]

    def entity_within_caps(keys: list[tuple[str, int]]) -> bool:
        if config is None:
            return True
        max_cameras = int(getattr(config, "max_global_entity_cameras", 0) or 0)
        max_tracks = int(getattr(config, "max_global_entity_tracks", 0) or 0)
        if max_cameras > 0 and len({cam for cam, _tid in keys}) > max_cameras:
            return False
        if max_tracks > 0 and len(set(keys)) > max_tracks:
            return False
        return True

    def time_compatible_with_entity(eid: int, key: tuple[str, int]) -> bool:
        if config is None:
            return True
        incoming = track_lookup.get(key)
        if incoming is None:
            return True
        tolerance = float(getattr(config, "cross_camera_overlap_tolerance_sec", 5.0))
        max_gap = float(getattr(config, "max_transition_sec", 30.0))
        in_start = float(incoming["start_time"])
        in_end = float(incoming["end_time"])
        for existing_key in entity_cams[eid].values():
            if existing_key[0] == key[0]:
                continue
            existing = track_lookup.get(existing_key)
            if existing is None:
                continue
            ex_start = float(existing["start_time"])
            ex_end = float(existing["end_time"])
            if _time_overlap(in_start, in_end, ex_start, ex_end) > tolerance:
                return False
            if _time_gap(in_start, in_end, ex_start, ex_end) > max_gap:
                return False
        return True

    def can_add_to_entity(eid: int, key: tuple[str, int]) -> bool:
        keys = list(entity_cams[eid].values()) + [key]
        return entity_within_caps(keys) and time_compatible_with_entity(eid, key)

    def can_merge_entities(eid_a: int, eid_b: int) -> bool:
        keys = list(entity_cams[eid_a].values()) + list(entity_cams[eid_b].values())
        if not entity_within_caps(keys):
            return False
        for key in entity_cams[eid_b].values():
            if not time_compatible_with_entity(eid_a, key):
                return False
        return True

    for cam_a, tid_a, cam_b, tid_b, _score in assignments:
        key_a = (cam_a, tid_a)
        key_b = (cam_b, tid_b)
        eid_a = entity_of.get(key_a)
        eid_b = entity_of.get(key_b)

        if eid_a is None and eid_b is None:
            new_entity(key_a, key_b)

        elif eid_a is not None and eid_b is None:
            if cam_b not in entity_cams[eid_a] and can_add_to_entity(eid_a, key_b):
                add_to_entity(eid_a, key_b)

        elif eid_a is None and eid_b is not None:
            if cam_a not in entity_cams[eid_b] and can_add_to_entity(eid_b, key_a):
                add_to_entity(eid_b, key_a)

        else:
            if eid_a == eid_b:
                continue  # already in same entity
            # Merge only if no camera conflict between the two entities
            cams_a = set(entity_cams[eid_a].keys())
            cams_b = set(entity_cams[eid_b].keys())
            if not (cams_a & cams_b) and can_merge_entities(eid_a, eid_b):
                merge_entities(eid_a, eid_b)
            # else: conflict — skip this assignment

    entities: list[GlobalEntity] = []
    for idx, (eid, cam_track_map) in enumerate(
        sorted(entity_cams.items()), start=1
    ):
        appearances: list[CameraAppearance] = []
        for cam_id, (cam_id2, tid) in cam_track_map.items():
            t = track_lookup.get((cam_id, tid))
            if t is None:
                continue
            conf_vals = [
                v for (a, ta, b, tb), v in confidence_map.items()
                if (a == cam_id and ta == tid) or (b == cam_id and tb == tid)
            ]
            appearances.append(CameraAppearance(
                camera_id=cam_id,
                track_id=tid,
                start_time=t["start_time"],
                end_time=t["end_time"],
                confidence=max(conf_vals) if conf_vals else 0.0,
            ))
        appearances.sort(key=lambda a: a.start_time)
        entities.append(GlobalEntity(
            global_entity_id=f"person_global_{idx}",
            appearances=appearances,
        ))

    logger.info("Global entities (score-ordered greedy): %d", len(entities))
    return entities


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def match_across_cameras(
    per_camera: list[CameraResult],
    config: CrossCameraConfig,
    embedder: "ReIDEmbedder",
    llm_verify_fn: Callable[..., MatchVerification] | None = None,
    topology_prior: "CameraTopologyPrior | None" = None,
) -> list[GlobalEntity]:
    """Cross-camera matching entrypoint.

    1. Candidate pairs under time constraints
    2. Re-ID cosine + topology scoring
    3. (Optional) top-K LLM verification on borderline cosine
    4. Greedy assignment → GlobalEntity list
    5. Online topology update: feed confirmed match transit times back to prior
    """
    pairs = build_candidate_pairs(per_camera, config)
    logger.info("Candidate pairs after time filter: %d", len(pairs))

    scored = score_candidate_pairs(pairs, embedder, config, topology_prior=topology_prior)
    logger.info("Scored pairs: %d", len(scored))

    if llm_verify_fn is not None and config.llm_verify_top_k > 0:
        # VLM only for borderline cosine in [min, max]
        border_cases: list[tuple[CameraResult, dict, CameraResult, dict, float]] = []
        for item in scored:
            cam_i, ti, cam_j, tj, score = item
            emb_i = cam_i.person_embeddings.get(ti["track_id"])
            emb_j = cam_j.person_embeddings.get(tj["track_id"])
            if emb_i is None or emb_j is None:
                continue
            cosine = float(emb_i @ emb_j)
            if config.llm_verify_cosine_min <= cosine <= config.llm_verify_cosine_max:
                border_cases.append(item)
        top_k = border_cases[: config.llm_verify_top_k]
        verified: list[tuple[CameraResult, dict, CameraResult, dict, float]] = []
        for cam_i, ti, cam_j, tj, sim in top_k:
            crops_i = cam_i.person_crops.get(ti["track_id"], [])
            crops_j = cam_j.person_crops.get(tj["track_id"], [])
            if not crops_i or not crops_j:
                verified.append((cam_i, ti, cam_j, tj, sim))
                continue
            result: MatchVerification = llm_verify_fn(crops_i[0], crops_j[0])
            if result.is_match:
                boosted = min(1.0, sim * 0.7 + result.confidence * 0.3)
                verified.append((cam_i, ti, cam_j, tj, boosted))
            else:
                logger.info(
                    "LLM rejected: cam=%s tid=%d ↔ cam=%s tid=%d (%s)",
                    cam_i.camera_id, ti["track_id"],
                    cam_j.camera_id, tj["track_id"],
                    result.reasoning,
                )
        verified_keys = {(c1.camera_id, t1["track_id"], c2.camera_id, t2["track_id"]) for c1, t1, c2, t2, _ in top_k}
        remaining = [
            x for x in scored
            if (x[0].camera_id, x[1]["track_id"], x[2].camera_id, x[3]["track_id"]) not in verified_keys
        ]
        scored = sorted(verified + remaining, key=lambda x: x[4], reverse=True)

    assignments = _greedy_assign(scored, config.cross_camera_min_score)
    logger.info("Assignments: %d", len(assignments))

    # ------------------------------------------------------------------ #
    # Online topology prior update                                         #
    # Feed confirmed match transit times back into the prior so subsequent #
    # videos benefit from accumulated real-world transit statistics.       #
    # ------------------------------------------------------------------ #
    if topology_prior is not None and assignments:
        track_lookup: dict[tuple[str, int], dict[str, Any]] = {
            (cam.camera_id, t["track_id"]): t
            for cam in per_camera
            for t in cam.tracks
        }
        new_transitions: list[tuple[str, str, float]] = []
        for cam_a, tid_a, cam_b, tid_b, score in assignments:
            t_a = track_lookup.get((cam_a, tid_a))
            t_b = track_lookup.get((cam_b, tid_b))
            if t_a is None or t_b is None:
                continue
            # Direction: earlier track → later track
            end_a = float(t_a["end_time"])
            start_b = float(t_b["start_time"])
            end_b = float(t_b["end_time"])
            start_a = float(t_a["start_time"])
            if end_a <= start_b:
                delta_t = start_b - end_a
                new_transitions.append((cam_a, cam_b, delta_t))
            elif end_b <= start_a:
                delta_t = start_a - end_b
                new_transitions.append((cam_b, cam_a, delta_t))
            # Overlapping tracks: skip (same-time, not a transit)
        if new_transitions:
            topology_prior.observe_batch(new_transitions)
            logger.info(
                "Topology prior updated with %d new transitions.", len(new_transitions)
            )

    return _build_global_entities(assignments, per_camera, config)
