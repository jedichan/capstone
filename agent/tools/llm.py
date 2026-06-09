import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

EMBEDDING_BATCH_LIMIT = 10

_LOG = logging.getLogger(__name__)


def _embedding_provider() -> str:
    return os.environ.get("AGENT_EMBEDDING_PROVIDER", "openai").strip().lower()


def _embedding_model() -> str:
    provider = _embedding_provider()
    default_model = "text-embedding-3-large" if provider == "openai" else "text-embedding-v3"
    return os.environ.get("AGENT_EMBEDDING_MODEL", default_model).strip()


def _embedding_dimensions() -> int | None:
    raw = os.environ.get("AGENT_EMBEDDING_DIMENSIONS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def _build_embedding_client() -> OpenAI:
    provider = _embedding_provider()
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip(),
    )


def get_embedding_runtime_profile() -> dict:
    return {
        "provider": _embedding_provider(),
        "model": _embedding_model(),
        "dimensions": _embedding_dimensions(),
    }


# ---------------------------------------------------------------------------
# P1-3 query embedding cache (in-memory LRU + optional disk cache + retry)
# ---------------------------------------------------------------------------

_EMBEDDING_LRU_DEFAULT_SIZE = 2048


def _lru_size() -> int:
    raw = os.environ.get("AGENT_EMBEDDING_CACHE_LRU_SIZE", str(_EMBEDDING_LRU_DEFAULT_SIZE)).strip()
    try:
        value = int(raw)
    except Exception:
        return _EMBEDDING_LRU_DEFAULT_SIZE
    return max(1, value)


_EMBEDDING_LRU: "OrderedDict[str, list[float]]" = OrderedDict()
_EMBEDDING_LRU_LOCK = threading.Lock()
_EMBEDDING_CACHE_STATS = {
    "lru_hits": 0,
    "disk_hits": 0,
    "remote_calls": 0,
    "errors": 0,
}
_EMBEDDING_STATS_LOCK = threading.Lock()


def _cache_disabled() -> bool:
    return os.environ.get("AGENT_EMBEDDING_CACHE_DISABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _disk_cache_dir() -> Optional[Path]:
    raw = os.environ.get("AGENT_EMBEDDING_CACHE_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return path


def _cache_key(text: str) -> str:
    profile = get_embedding_runtime_profile()
    raw = f"{profile['provider']}|{profile['model']}|{profile['dimensions']}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _bump(stat_key: str, delta: int = 1) -> None:
    with _EMBEDDING_STATS_LOCK:
        _EMBEDDING_CACHE_STATS[stat_key] = _EMBEDDING_CACHE_STATS.get(stat_key, 0) + delta


def _lru_get(key: str) -> Optional[list[float]]:
    with _EMBEDDING_LRU_LOCK:
        value = _EMBEDDING_LRU.get(key)
        if value is None:
            return None
        _EMBEDDING_LRU.move_to_end(key)
        return list(value)


def _lru_put(key: str, vec: list[float]) -> None:
    with _EMBEDDING_LRU_LOCK:
        _EMBEDDING_LRU[key] = list(vec)
        _EMBEDDING_LRU.move_to_end(key)
        capacity = _lru_size()
        while len(_EMBEDDING_LRU) > capacity:
            _EMBEDDING_LRU.popitem(last=False)


def _load_from_disk(text: str) -> Optional[list[float]]:
    base = _disk_cache_dir()
    if base is None:
        return None
    fp = base / f"{_cache_key(text)}.json"
    if not fp.exists():
        return None
    try:
        with fp.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        vec = payload.get("vector") if isinstance(payload, dict) else payload
        if not isinstance(vec, list):
            return None
        return [float(x) for x in vec]
    except Exception:
        return None


def _save_to_disk(text: str, vec: list[float]) -> None:
    base = _disk_cache_dir()
    if base is None:
        return
    fp = base / f"{_cache_key(text)}.json"
    payload = {
        "vector": vec,
        "model": _embedding_model(),
        "dimensions": _embedding_dimensions(),
        "provider": _embedding_provider(),
    }
    try:
        # write to temp then rename to avoid corrupting partial writes
        tmp = fp.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp.replace(fp)
    except Exception:
        pass


def _lookup_cached(text: str) -> Optional[list[float]]:
    """Return cached vector from LRU (preferred) or disk; None on miss.
    A disk hit is promoted into the LRU."""
    if _cache_disabled():
        return None
    key = _cache_key(text)
    vec = _lru_get(key)
    if vec is not None:
        _bump("lru_hits")
        return vec
    disk = _load_from_disk(text)
    if disk is not None:
        _bump("disk_hits")
        _lru_put(key, disk)
        return disk
    return None


def _store_cached(text: str, vec: list[float]) -> None:
    if _cache_disabled():
        return
    _lru_put(_cache_key(text), vec)
    _save_to_disk(text, vec)


def _embed_single_remote_with_retry(text: str, max_retries: int = 3) -> list[float]:
    """Single-text remote call with exponential backoff (0.5s, 1s, 2s)."""
    last_err: Optional[Exception] = None
    delay = 0.5
    for attempt in range(max_retries):
        try:
            client = _build_embedding_client()
            model = _embedding_model()
            dimensions = _embedding_dimensions()
            request: dict = {"model": model, "input": text, "encoding_format": "float"}
            if dimensions is not None:
                request["dimensions"] = dimensions
            completion = client.embeddings.create(**request)
            _bump("remote_calls")
            return [float(x) for x in completion.data[0].embedding]
        except Exception as exc:
            last_err = exc
            _bump("errors")
            if attempt >= max_retries - 1:
                break
            _LOG.warning(
                "Embedding API call failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt + 1, max_retries, exc, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 4.0)
    assert last_err is not None
    raise last_err


def _embed_batch_remote_with_retry(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """Batch remote call (already chunked to <= EMBEDDING_BATCH_LIMIT)."""
    last_err: Optional[Exception] = None
    delay = 0.5
    for attempt in range(max_retries):
        try:
            client = _build_embedding_client()
            model = _embedding_model()
            dimensions = _embedding_dimensions()
            request: dict = {"model": model, "input": texts, "encoding_format": "float"}
            if dimensions is not None:
                request["dimensions"] = dimensions
            completion = client.embeddings.create(**request)
            _bump("remote_calls")
            sorted_data = sorted(completion.data, key=lambda x: x.index)
            return [[float(x) for x in item.embedding] for item in sorted_data]
        except Exception as exc:
            last_err = exc
            _bump("errors")
            if attempt >= max_retries - 1:
                break
            _LOG.warning(
                "Batch embedding API call failed (attempt %d/%d, batch_size=%d): %s; retrying in %.1fs",
                attempt + 1, max_retries, len(texts), exc, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 4.0)
    assert last_err is not None
    raise last_err


def clear_embedding_cache() -> None:
    """Clear the in-memory LRU cache (disk cache untouched). Used by tests."""
    with _EMBEDDING_LRU_LOCK:
        _EMBEDDING_LRU.clear()
    with _EMBEDDING_STATS_LOCK:
        for k in _EMBEDDING_CACHE_STATS:
            _EMBEDDING_CACHE_STATS[k] = 0


def get_embedding_cache_stats() -> dict:
    """Return a snapshot of cache stats and current LRU occupancy."""
    with _EMBEDDING_STATS_LOCK:
        snapshot = dict(_EMBEDDING_CACHE_STATS)
    with _EMBEDDING_LRU_LOCK:
        snapshot["lru_currsize"] = len(_EMBEDDING_LRU)
    snapshot["lru_maxsize"] = _lru_size()
    return snapshot


def get_qwen_embedding(text: Union[str, List[str]]) -> Union[list[float], List[list[float]]]:
    """
    调用当前配置的 embedding API 获取文本向量，支持单条或批量传入。
    如果是字符串，返回单条 vector；如果是列表，返回 vector 列表。

    P1-3 改造：
      - 单条：先查 LRU → 查磁盘 → 命中提升到 LRU；miss 走 retry-aware remote 调用，结果写两层 cache。
      - 批量：先逐项查 LRU/磁盘；miss 集合按 ``EMBEDDING_BATCH_LIMIT`` 分批一次性合并 API 调用，结果回写两层 cache。
      - 单失败时指数退避（0.5s → 1s → 2s）最多 3 次。

    Env knobs:
      AGENT_EMBEDDING_CACHE_DISABLED=1   bypass both layers (debug only)
      AGENT_EMBEDDING_CACHE_DIR=path     enable persistent disk cache (per cache_key)
      AGENT_EMBEDDING_CACHE_LRU_SIZE=N   LRU capacity (default 2048)
    """
    try:
        load_dotenv()
    except Exception:
        pass

    if isinstance(text, str):
        cached = _lookup_cached(text)
        if cached is not None:
            return cached
        vec = _embed_single_remote_with_retry(text)
        _store_cached(text, vec)
        return vec

    # Batch path
    results: list[Optional[list[float]]] = [None] * len(text)
    miss_indices: list[int] = []
    for idx, t in enumerate(text):
        cached = _lookup_cached(t)
        if cached is not None:
            results[idx] = cached
        else:
            miss_indices.append(idx)

    if miss_indices:
        miss_texts = [text[i] for i in miss_indices]
        for chunk_start in range(0, len(miss_texts), EMBEDDING_BATCH_LIMIT):
            chunk = miss_texts[chunk_start : chunk_start + EMBEDDING_BATCH_LIMIT]
            chunk_indices = miss_indices[chunk_start : chunk_start + EMBEDDING_BATCH_LIMIT]
            chunk_vecs = _embed_batch_remote_with_retry(chunk)
            for local_idx, abs_idx in enumerate(chunk_indices):
                vec = chunk_vecs[local_idx]
                results[abs_idx] = vec
                _store_cached(text[abs_idx], vec)

    return [vec if vec is not None else [] for vec in results]


if __name__ == "__main__":
    test_text = "银灰色轿车在停车位区域长期静止停放"
    print(f"正在测试生成向量: {test_text}")
    try:
        vec = get_qwen_embedding(test_text)
        print(f"✅ 生成成功，向量维度: {len(vec)}")
        print(f"向量前5维: {vec[:5]}")
        vec2 = get_qwen_embedding(test_text)
        print(f"second call returned same shape: {len(vec) == len(vec2)}")
        print(f"cache stats: {get_embedding_cache_stats()}")
    except Exception as e:
        print(f"❌ 生成失败: {e}")
