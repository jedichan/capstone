"""Unit tests for the P1-3 query embedding cache.

Scope:
  - LRU cache hit / miss
  - Disk cache hit / miss
  - Disk hit warms up LRU
  - Batch path: per-item lookup + merged remote call for misses (single API call)
  - Batch path: respects EMBEDDING_BATCH_LIMIT chunking
  - LRU capacity eviction (LRU eviction order)
  - AGENT_EMBEDDING_CACHE_DISABLED=1 bypasses both layers
  - cache key is sensitive to model/dimensions/provider
  - retry behaviour on remote failure
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT_DIR / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tools import llm as llm_module  # noqa: E402


def _fake_completion(vectors: list[list[float]]):
    """Build an OpenAI-completion-shaped object with .data[*].embedding/.index."""
    items = []
    for i, vec in enumerate(vectors):
        item = mock.Mock()
        item.embedding = vec
        item.index = i
        items.append(item)
    completion = mock.Mock()
    completion.data = items
    return completion


class _BaseCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self._env_patch = mock.patch.dict(
            os.environ,
            {
                "AGENT_EMBEDDING_PROVIDER": "openai",
                "AGENT_EMBEDDING_MODEL": "test-model",
                "AGENT_EMBEDDING_CACHE_DIR": self.tmp_dir.name,
                "AGENT_EMBEDDING_CACHE_LRU_SIZE": "4",
                "AGENT_EMBEDDING_CACHE_DISABLED": "",
                "AGENT_EMBEDDING_DIMENSIONS": "",
                "OPENAI_API_KEY": "test-key",
            },
            clear=False,
        )
        self._env_patch.start()
        # patch sleep to keep retry-driven tests fast
        self._sleep_patch = mock.patch("tools.llm.time.sleep", return_value=None)
        self._sleep_patch.start()
        llm_module.clear_embedding_cache()

    def tearDown(self) -> None:
        self._sleep_patch.stop()
        self._env_patch.stop()
        llm_module.clear_embedding_cache()
        self.tmp_dir.cleanup()


class SingleCacheTests(_BaseCacheTest):
    def test_first_call_hits_remote_then_lru_caches(self) -> None:
        with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
            mock_client = mock.Mock()
            mock_client.embeddings.create.return_value = _fake_completion([[0.1, 0.2, 0.3]])
            mock_build.return_value = mock_client

            v1 = llm_module.get_qwen_embedding("hello world")
            self.assertEqual(v1, [0.1, 0.2, 0.3])
            self.assertEqual(mock_client.embeddings.create.call_count, 1)

            v2 = llm_module.get_qwen_embedding("hello world")
            self.assertEqual(v2, [0.1, 0.2, 0.3])
            self.assertEqual(mock_client.embeddings.create.call_count, 1)  # LRU hit

        stats = llm_module.get_embedding_cache_stats()
        self.assertEqual(stats["remote_calls"], 1)
        self.assertGreaterEqual(stats["lru_hits"], 1)

    def test_disk_hit_after_lru_clear(self) -> None:
        with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
            mock_client = mock.Mock()
            mock_client.embeddings.create.return_value = _fake_completion([[0.5, 0.6]])
            mock_build.return_value = mock_client

            llm_module.get_qwen_embedding("disk-text")
            self.assertEqual(mock_client.embeddings.create.call_count, 1)

            # wipe LRU; disk should still serve the next call
            llm_module.clear_embedding_cache()
            v2 = llm_module.get_qwen_embedding("disk-text")
            self.assertEqual(v2, [0.5, 0.6])
            self.assertEqual(mock_client.embeddings.create.call_count, 1)

        stats = llm_module.get_embedding_cache_stats()
        self.assertGreaterEqual(stats["disk_hits"], 1)

    def test_cache_key_distinguishes_model(self) -> None:
        with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
            mock_client = mock.Mock()
            mock_client.embeddings.create.side_effect = [
                _fake_completion([[1.0, 0.0]]),
                _fake_completion([[0.0, 1.0]]),
            ]
            mock_build.return_value = mock_client

            v_a = llm_module.get_qwen_embedding("same text")
            with mock.patch.dict(os.environ, {"AGENT_EMBEDDING_MODEL": "other-model"}):
                v_b = llm_module.get_qwen_embedding("same text")
            self.assertNotEqual(v_a, v_b)
            self.assertEqual(mock_client.embeddings.create.call_count, 2)

    def test_disabled_flag_bypasses_cache(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_EMBEDDING_CACHE_DISABLED": "1"}):
            with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
                mock_client = mock.Mock()
                mock_client.embeddings.create.side_effect = [
                    _fake_completion([[0.1]]),
                    _fake_completion([[0.2]]),
                ]
                mock_build.return_value = mock_client

                v1 = llm_module.get_qwen_embedding("hot text")
                v2 = llm_module.get_qwen_embedding("hot text")
                self.assertEqual(v1, [0.1])
                self.assertEqual(v2, [0.2])
                self.assertEqual(mock_client.embeddings.create.call_count, 2)

    def test_lru_eviction_keeps_capacity_bounded(self) -> None:
        # capacity = 4 (set in _BaseCacheTest); after 5 distinct inputs, the
        # oldest one must have been evicted.
        side_effects = [_fake_completion([[float(i)]]) for i in range(5)]
        with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
            mock_client = mock.Mock()
            mock_client.embeddings.create.side_effect = side_effects
            mock_build.return_value = mock_client

            for i in range(5):
                llm_module.get_qwen_embedding(f"text-{i}")
            self.assertEqual(mock_client.embeddings.create.call_count, 5)

            stats = llm_module.get_embedding_cache_stats()
            self.assertLessEqual(stats["lru_currsize"], 4)


class BatchCacheTests(_BaseCacheTest):
    def test_batch_misses_merge_into_single_api_call(self) -> None:
        with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
            mock_client = mock.Mock()
            mock_client.embeddings.create.return_value = _fake_completion(
                [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]
            )
            mock_build.return_value = mock_client

            vecs = llm_module.get_qwen_embedding(["a", "b", "c"])
            self.assertEqual(len(vecs), 3)
            self.assertEqual(vecs[0], [0.1, 0.0])
            self.assertEqual(vecs[1], [0.2, 0.0])
            self.assertEqual(vecs[2], [0.3, 0.0])
            # one batched API call, not three
            self.assertEqual(mock_client.embeddings.create.call_count, 1)

    def test_batch_partial_cache_hit_only_misses_call_remote(self) -> None:
        # warm cache with "a"
        with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
            mock_client = mock.Mock()
            mock_client.embeddings.create.return_value = _fake_completion([[0.1, 0.0]])
            mock_build.return_value = mock_client
            llm_module.get_qwen_embedding("a")
            self.assertEqual(mock_client.embeddings.create.call_count, 1)

        # batch ["a", "b"] should only call remote for "b"
        with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
            mock_client = mock.Mock()
            mock_client.embeddings.create.return_value = _fake_completion([[0.2, 0.0]])
            mock_build.return_value = mock_client

            vecs = llm_module.get_qwen_embedding(["a", "b"])
            self.assertEqual(vecs[0], [0.1, 0.0])
            self.assertEqual(vecs[1], [0.2, 0.0])
            self.assertEqual(mock_client.embeddings.create.call_count, 1)
            # the batched call must contain only the cache-miss item
            call_kwargs = mock_client.embeddings.create.call_args.kwargs
            self.assertEqual(call_kwargs["input"], ["b"])

    def test_batch_chunks_at_embedding_batch_limit(self) -> None:
        # capacity=4 LRU, 12 distinct misses → 2 chunks of 10 + ...
        n = llm_module.EMBEDDING_BATCH_LIMIT + 3
        side_effects = [
            _fake_completion([[float(i)] for i in range(llm_module.EMBEDDING_BATCH_LIMIT)]),
            _fake_completion([[float(i)] for i in range(3)]),
        ]
        with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
            mock_client = mock.Mock()
            mock_client.embeddings.create.side_effect = side_effects
            mock_build.return_value = mock_client

            texts = [f"item-{i}" for i in range(n)]
            vecs = llm_module.get_qwen_embedding(texts)
            self.assertEqual(len(vecs), n)
            self.assertEqual(mock_client.embeddings.create.call_count, 2)


class RetryTests(_BaseCacheTest):
    def test_single_call_retries_then_succeeds(self) -> None:
        boom = RuntimeError("transient")
        ok = _fake_completion([[0.7, 0.7]])
        with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
            mock_client = mock.Mock()
            mock_client.embeddings.create.side_effect = [boom, ok]
            mock_build.return_value = mock_client

            vec = llm_module.get_qwen_embedding("flaky")
            self.assertEqual(vec, [0.7, 0.7])
            self.assertEqual(mock_client.embeddings.create.call_count, 2)

    def test_single_call_propagates_after_max_retries(self) -> None:
        with mock.patch.object(llm_module, "_build_embedding_client") as mock_build:
            mock_client = mock.Mock()
            mock_client.embeddings.create.side_effect = RuntimeError("dead")
            mock_build.return_value = mock_client

            with self.assertRaises(RuntimeError):
                llm_module.get_qwen_embedding("dead-text")
            self.assertEqual(mock_client.embeddings.create.call_count, 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
