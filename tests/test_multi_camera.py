"""Unit tests for cross-camera matching (mocked; no YOLO / OpenCV / FastReID)."""

from __future__ import annotations

import unittest

import numpy as np

from video.core.schema.multi_camera import (
    CameraAppearance,
    CameraResult,
    CrossCameraConfig,
    GlobalEntity,
    MatchVerification,
)
from video.factory.processors.cross_camera_matcher import (
    _greedy_assign,
    _build_global_entities,
    build_candidate_pairs,
    match_across_cameras,
    passes_time_constraint,
    score_candidate_pairs,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_track(tid: int, cls: str, start: float, end: float) -> dict:
    return {
        "track_id": tid,
        "class_name": cls,
        "start_time": start,
        "end_time": end,
    }


def _make_cam(
    cam_id: str,
    tracks: list[dict],
    embeddings: dict[int, np.ndarray] | None = None,
) -> CameraResult:
    return CameraResult(
        camera_id=cam_id,
        video_path=f"/fake/{cam_id}.mp4",
        tracks=tracks,
        events=[],
        clips=[],
        person_crops={},
        person_embeddings=embeddings or {},
    )


class _FakeEmbedder:
    """Stub that satisfies the ReIDEmbedder protocol used by score_candidate_pairs."""

    def embed_crops(self, crops: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def cosine_similarity(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
        return feats_a @ feats_b.T


# ------------------------------------------------------------------
# Tests: time constraint
# ------------------------------------------------------------------

class TestTimeConstraint(unittest.TestCase):
    def setUp(self):
        self.config = CrossCameraConfig(max_transition_sec=10.0, min_overlap_sec=0.0)

    def test_overlapping_tracks_pass(self):
        ta = _make_track(1, "person", 0, 20)
        tb = _make_track(2, "person", 15, 35)
        self.assertTrue(passes_time_constraint(ta, tb, self.config))

    def test_sequential_within_threshold(self):
        ta = _make_track(1, "person", 0, 20)
        tb = _make_track(2, "person", 25, 40)
        self.assertTrue(passes_time_constraint(ta, tb, self.config))

    def test_sequential_exceeds_threshold(self):
        ta = _make_track(1, "person", 0, 20)
        tb = _make_track(2, "person", 50, 60)
        self.assertFalse(passes_time_constraint(ta, tb, self.config))

    def test_exact_boundary(self):
        ta = _make_track(1, "person", 0, 10)
        tb = _make_track(2, "person", 20, 30)
        self.assertTrue(passes_time_constraint(ta, tb, self.config))

    def test_min_overlap_enforced(self):
        config = CrossCameraConfig(max_transition_sec=10.0, min_overlap_sec=5.0)
        ta = _make_track(1, "person", 0, 20)
        tb = _make_track(2, "person", 18, 35)
        self.assertFalse(passes_time_constraint(ta, tb, config))

        tb2 = _make_track(3, "person", 10, 35)
        self.assertTrue(passes_time_constraint(ta, tb2, config))


# ------------------------------------------------------------------
# Tests: candidate pair building
# ------------------------------------------------------------------

class TestCandidatePairs(unittest.TestCase):
    def test_person_only_filter(self):
        config = CrossCameraConfig(max_transition_sec=30, person_only=True)
        cam1 = _make_cam("c1", [_make_track(1, "person", 0, 10), _make_track(2, "car", 0, 10)])
        cam2 = _make_cam("c2", [_make_track(3, "person", 5, 15), _make_track(4, "car", 5, 15)])
        pairs = build_candidate_pairs([cam1, cam2], config)
        track_ids = {(ti["track_id"], tj["track_id"]) for _, ti, _, tj in pairs}
        self.assertEqual(track_ids, {(1, 3)})

    def test_no_pairs_when_time_exceeds(self):
        config = CrossCameraConfig(max_transition_sec=5, person_only=True)
        cam1 = _make_cam("c1", [_make_track(1, "person", 0, 10)])
        cam2 = _make_cam("c2", [_make_track(2, "person", 50, 60)])
        pairs = build_candidate_pairs([cam1, cam2], config)
        self.assertEqual(len(pairs), 0)


# ------------------------------------------------------------------
# Tests: similarity scoring
# ------------------------------------------------------------------

class TestSimilarityScoring(unittest.TestCase):
    def test_score_sorted_descending(self):
        dim = 128
        emb_a = np.random.randn(dim).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)
        emb_b_similar = emb_a + np.random.randn(dim).astype(np.float32) * 0.1
        emb_b_similar /= np.linalg.norm(emb_b_similar)
        emb_c_different = np.random.randn(dim).astype(np.float32)
        emb_c_different /= np.linalg.norm(emb_c_different)

        cam1 = _make_cam("c1", [_make_track(1, "person", 0, 10)], {1: emb_a})
        cam2 = _make_cam("c2", [
            _make_track(2, "person", 5, 15),
            _make_track(3, "person", 5, 15),
        ], {2: emb_b_similar, 3: emb_c_different})

        config = CrossCameraConfig(max_transition_sec=30)
        pairs = build_candidate_pairs([cam1, cam2], config)
        scored = score_candidate_pairs(pairs, _FakeEmbedder(), config)  # type: ignore[arg-type]

        self.assertEqual(len(scored), 2)
        self.assertGreaterEqual(scored[0][4], scored[1][4])


# ------------------------------------------------------------------
# Tests: greedy assignment
# ------------------------------------------------------------------

class TestGreedyAssign(unittest.TestCase):
    def test_basic_assignment(self):
        dim = 128
        emb = np.random.randn(dim).astype(np.float32)
        emb /= np.linalg.norm(emb)

        cam1 = _make_cam("c1", [_make_track(1, "person", 0, 10)], {1: emb})
        cam2 = _make_cam("c2", [_make_track(2, "person", 5, 15)], {2: emb})

        scored = [(cam1, cam1.tracks[0], cam2, cam2.tracks[0], 0.9)]
        assignments = _greedy_assign(scored, threshold=0.5)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0][:4], ("c1", 1, "c2", 2))

    def test_below_threshold_rejected(self):
        cam1 = _make_cam("c1", [_make_track(1, "person", 0, 10)])
        cam2 = _make_cam("c2", [_make_track(2, "person", 5, 15)])
        scored = [(cam1, cam1.tracks[0], cam2, cam2.tracks[0], 0.3)]
        assignments = _greedy_assign(scored, threshold=0.5)
        self.assertEqual(len(assignments), 0)

    def test_no_duplicate_assignment(self):
        """Each track keeps only the best opponent within a camera pair."""
        cam1 = _make_cam("c1", [_make_track(1, "person", 0, 10)])
        cam2 = _make_cam("c2", [
            _make_track(2, "person", 5, 15),
            _make_track(3, "person", 5, 15),
        ])
        scored = [
            (cam1, cam1.tracks[0], cam2, cam2.tracks[0], 0.9),
            (cam1, cam1.tracks[0], cam2, cam2.tracks[1], 0.85),
        ]
        assignments = _greedy_assign(scored, threshold=0.5)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0][3], 2)


# ------------------------------------------------------------------
# Tests: global entity building
# ------------------------------------------------------------------

class TestGlobalEntityBuilding(unittest.TestCase):
    def test_two_cameras_one_entity(self):
        cam1 = _make_cam("c1", [_make_track(1, "person", 0, 10)])
        cam2 = _make_cam("c2", [_make_track(2, "person", 12, 20)])
        assignments = [("c1", 1, "c2", 2, 0.9)]
        entities = _build_global_entities(assignments, [cam1, cam2])
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(entities[0].appearances), 2)
        self.assertEqual(entities[0].appearances[0].camera_id, "c1")
        self.assertEqual(entities[0].appearances[1].camera_id, "c2")

    def test_transitive_merge_three_cameras(self):
        """c1-1 ↔ c2-2, c2-2 ↔ c3-3 → one merged entity."""
        cam1 = _make_cam("c1", [_make_track(1, "person", 0, 10)])
        cam2 = _make_cam("c2", [_make_track(2, "person", 12, 20)])
        cam3 = _make_cam("c3", [_make_track(3, "person", 22, 30)])
        assignments = [
            ("c1", 1, "c2", 2, 0.9),
            ("c2", 2, "c3", 3, 0.85),
        ]
        entities = _build_global_entities(assignments, [cam1, cam2, cam3])
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(entities[0].appearances), 3)


# ------------------------------------------------------------------
# Tests: full match_across_cameras pipeline (mocked embedder)
# ------------------------------------------------------------------

class TestMatchAcrossCameras(unittest.TestCase):
    def test_end_to_end_with_mock(self):
        dim = 128
        rng = np.random.RandomState(42)
        emb_person = rng.randn(dim).astype(np.float32)
        emb_person /= np.linalg.norm(emb_person)
        emb_person_noisy = emb_person + rng.randn(dim).astype(np.float32) * 0.05
        emb_person_noisy /= np.linalg.norm(emb_person_noisy)
        emb_other = rng.randn(dim).astype(np.float32)
        emb_other /= np.linalg.norm(emb_other)

        cam1 = _make_cam("c1", [_make_track(1, "person", 0, 10)], {1: emb_person})
        cam2 = _make_cam("c2", [
            _make_track(2, "person", 15, 25),
            _make_track(3, "person", 15, 25),
        ], {2: emb_person_noisy, 3: emb_other})

        config = CrossCameraConfig(max_transition_sec=20, embedding_threshold=0.5)
        entities = match_across_cameras([cam1, cam2], config, _FakeEmbedder())  # type: ignore[arg-type]

        matched_tids = set()
        for ent in entities:
            for a in ent.appearances:
                matched_tids.add((a.camera_id, a.track_id))
        self.assertIn(("c1", 1), matched_tids)
        self.assertIn(("c2", 2), matched_tids)

    def test_no_match_when_all_filtered_by_time(self):
        dim = 128
        emb = np.random.randn(dim).astype(np.float32)
        emb /= np.linalg.norm(emb)
        cam1 = _make_cam("c1", [_make_track(1, "person", 0, 10)], {1: emb})
        cam2 = _make_cam("c2", [_make_track(2, "person", 100, 110)], {2: emb})

        config = CrossCameraConfig(max_transition_sec=5, embedding_threshold=0.5)
        entities = match_across_cameras([cam1, cam2], config, _FakeEmbedder())  # type: ignore[arg-type]
        self.assertEqual(len(entities), 0)


if __name__ == "__main__":
    unittest.main()
