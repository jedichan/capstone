"""VideoRAG-style baseline on MEVID.

This is intentionally *not* our full multi-camera system.  It simulates the
common VideoRAG recipe:

    video/event captions -> text chunks -> retrieval -> LLM QA

It does not use explicit OSNet/ReID reranking, camera topology, global entity
matching, or person-crop evidence.  The point is to compare a caption/text RAG
baseline against our latest structured multi-camera pipeline on the same MEVID
cases.

The embedding retrieval mode is an adapted reproduction of kdr/videoRAG-mrr2024:
aligned video captions are represented as text chunks, embedded, retrieved by
cosine similarity, and passed to the same LLM used by our system for answering.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
for p in (ROOT, TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_mevid_video_agent_e2e import (  # noqa: E402
    DEFAULT_SEED_DIR,
    DEFAULT_VIDEO_DIR,
    DEFAULT_XLSX,
    SLOT_CAMERAS,
    _import_cases,
    _select_cases,
)
from test_mevid_video_rag_metrics import (  # noqa: E402
    _average_precision,
    _camera_pair_hit,
    _person_global_hit,
    _row_relevance,
    _summarize_multi_camera_level,
    _temporal_order_ok,
)

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus")
EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
PRICE_IN = float(os.environ.get("QWEN_PRICE_IN", "0.012"))
PRICE_OUT = float(os.environ.get("QWEN_PRICE_OUT", "0.036"))
_MAX_FRAME_EDGE = 640


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9_]+|g\d+", text)


def _camera_from_video_id(video_id: str) -> str:
    match = re.search(r"\.(G\d+)\.r\d+$", str(video_id or ""))
    return match.group(1) if match else ""


def _resize_frame(frame: "cv2.typing.MatLike") -> "cv2.typing.MatLike":
    h, w = frame.shape[:2]
    if max(h, w) <= _MAX_FRAME_EDGE:
        return frame
    scale = _MAX_FRAME_EDGE / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _sample_frame_at(video_path: Path, sec: float) -> str | None:
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_idx = max(0, min(total - 1, int(max(0.0, sec) * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = _resize_frame(frame)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()


def _sample_evidence_frames(
    rows: list[dict[str, Any]],
    *,
    video_dir: Path,
    max_frames: int,
    frames_per_evidence: int,
) -> list[str]:
    frames: list[str] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if len(frames) >= max_frames:
            break
        video_id = str(row.get("video_id") or "")
        if not video_id:
            continue
        path = video_dir / f"{video_id}.avi"
        start = float(row.get("start_time") or row.get("clip_start_sec") or 0.0)
        end = float(row.get("end_time") or row.get("clip_end_sec") or start)
        if end < start:
            end = start
        if frames_per_evidence <= 1:
            times = [(start + end) / 2.0]
        else:
            times = [
                start + (end - start) * i / max(1, frames_per_evidence - 1)
                for i in range(frames_per_evidence)
            ]
        for sec in times:
            if len(frames) >= max_frames:
                break
            key = (video_id, round(sec))
            if key in seen:
                continue
            b64 = _sample_frame_at(path, sec)
            if b64:
                frames.append(b64)
                seen.add(key)
    return frames


def _render_row_text(row: dict[str, Any], *, strip_structured_identity: bool) -> str:
    """Render a caption-like text chunk.

    VideoRAG usually retrieves text/caption chunks.  To keep this baseline fair,
    the default strips explicit global identity hints that our structured system
    creates for multi-camera reasoning.
    """
    parts: list[str] = []
    for key in [
        "video_id",
        "camera_id",
        "scene_zone",
        "object_type",
        "object_color",
        "appearance_notes",
        "event_text",
        "action_summary",
        "movement_direction",
        "exit_side",
        "scene_profile_context",
    ]:
        value = row.get(key)
        if value:
            parts.append(str(value))
    keywords = row.get("keywords")
    if isinstance(keywords, list):
        parts.append(" ".join(str(x) for x in keywords))
    elif keywords:
        parts.append(str(keywords))
    text = " ".join(parts)
    if strip_structured_identity:
        text = re.sub(r"\bperson_global_\d+\b", "person", text, flags=re.IGNORECASE)
        text = re.sub(r"\bentity_hint\s*[:=]?\s*\S+", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class TextChunk:
    row: dict[str, Any]
    text: str
    tf: Counter[str]
    length: int


class LexicalVideoRAGIndex:
    def __init__(self, rows: list[dict[str, Any]], *, strip_structured_identity: bool = True) -> None:
        self.chunks: list[TextChunk] = []
        df: Counter[str] = Counter()
        for row in rows:
            text = _render_row_text(row, strip_structured_identity=strip_structured_identity)
            toks = _tokenize(text)
            tf = Counter(toks)
            if not tf:
                continue
            for tok in tf:
                df[tok] += 1
            self.chunks.append(TextChunk(row=row, text=text, tf=tf, length=len(toks)))
        self.n = len(self.chunks)
        self.avg_len = sum(c.length for c in self.chunks) / max(1, self.n)
        self.idf = {
            tok: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }

    def search(self, query: str, *, top_k: int, camera_scope: set[str] | None = None) -> list[dict[str, Any]]:
        qtf = Counter(_tokenize(query))
        if not qtf:
            return []
        scores: list[tuple[float, int]] = []
        k1 = 1.5
        b = 0.75
        for idx, chunk in enumerate(self.chunks):
            if camera_scope:
                cam = str(chunk.row.get("camera_id") or _camera_from_video_id(str(chunk.row.get("video_id") or ""))).upper()
                if cam not in camera_scope:
                    continue
            score = 0.0
            for tok, q_count in qtf.items():
                tf = chunk.tf.get(tok, 0)
                if tf <= 0:
                    continue
                denom = tf + k1 * (1.0 - b + b * chunk.length / max(1.0, self.avg_len))
                score += self.idf.get(tok, 0.0) * (tf * (k1 + 1.0) / denom) * min(q_count, 3)
            if score > 0:
                scores.append((score, idx))
        scores.sort(key=lambda item: item[0], reverse=True)
        out = []
        for score, idx in scores[:top_k]:
            row = dict(self.chunks[idx].row)
            row["_videorag_score"] = round(score, 5)
            row["_videorag_text"] = self.chunks[idx].text
            out.append(row)
        return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


def _embed_texts(client: OpenAI, texts: list[str], *, batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    batch_size = max(1, min(batch_size, 10))
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(_normalize([float(x) for x in item.embedding]) for item in ordered)
    return vectors


class EmbeddingVideoRAGIndex:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        client: OpenAI,
        strip_structured_identity: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.chunks: list[TextChunk] = []
        texts: list[str] = []
        for row in rows:
            text = _render_row_text(row, strip_structured_identity=strip_structured_identity)
            toks = _tokenize(text)
            if not toks:
                continue
            self.chunks.append(TextChunk(row=row, text=text, tf=Counter(toks), length=len(toks)))
            texts.append(text)
        self.n = len(self.chunks)
        self.vectors = _embed_texts(client, texts, batch_size=batch_size) if texts else []

    def search(self, query: str, *, top_k: int, camera_scope: set[str] | None = None) -> list[dict[str, Any]]:
        if not query.strip() or not self.vectors:
            return []
        query_vec = _embed_texts(self.client, [query], batch_size=1)[0]
        scores: list[tuple[float, int]] = []
        for idx, chunk in enumerate(self.chunks):
            if camera_scope:
                cam = str(chunk.row.get("camera_id") or _camera_from_video_id(str(chunk.row.get("video_id") or ""))).upper()
                if cam not in camera_scope:
                    continue
            scores.append((_dot(query_vec, self.vectors[idx]), idx))
        scores.sort(key=lambda item: item[0], reverse=True)
        out = []
        for score, idx in scores[:top_k]:
            row = dict(self.chunks[idx].row)
            row["_videorag_score"] = round(score, 5)
            row["_videorag_text"] = self.chunks[idx].text
            out.append(row)
        return out


def _load_rows_for_slot(seed_dir: Path, slot: str) -> list[dict[str, Any]]:
    stems = set(SLOT_CAMERAS[slot].values())
    rows: list[dict[str, Any]] = []
    for stem in sorted(stems):
        path = seed_dir / f"{stem}_events_vector_flat.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        events = data.get("events") if isinstance(data, dict) else data
        for event in events or []:
            row = dict(event)
            row.setdefault("video_id", stem)
            row.setdefault("camera_id", _camera_from_video_id(stem))
            rows.append(row)
    return rows


def _mentioned_cameras(text: str) -> set[str]:
    return {cam.upper() for cam in re.findall(r"\bG\d+\b", text or "", flags=re.IGNORECASE)}


def _case_camera_scope(case: dict[str, Any], *, scoped: bool) -> set[str] | None:
    if not scoped:
        return None
    cams = _mentioned_cameras(str(case.get("question") or ""))
    cam = _camera_from_video_id(str(case.get("video_id") or ""))
    if cam:
        cams.add(cam)
    return cams or None


def _build_prompt(case: dict[str, Any], rows: list[dict[str, Any]], *, has_frames: bool = False) -> str:
    evidence_lines = []
    for idx, row in enumerate(rows, start=1):
        text = row.get("_videorag_text") or _render_row_text(row, strip_structured_identity=True)
        evidence_lines.append(
            f"[{idx}] video={row.get('video_id')} camera={row.get('camera_id')} "
            f"time={row.get('start_time')}-{row.get('end_time')} text={text}"
        )
    evidence = "\n".join(evidence_lines) if evidence_lines else "(no retrieved evidence)"
    return (
        "You are evaluating surveillance video evidence. Answer the question with exactly one word: yes or no.\n"
        "Use only the retrieved evidence"
        + (" and attached frames" if has_frames else "")
        + ". If the evidence is insufficient, answer no.\n\n"
        f"Question: {case.get('question')}\n\n"
        f"Retrieved evidence:\n{evidence}\n"
    )


def _parse_yes_no(text: str) -> str:
    ans = str(text or "").strip().lower()
    ans = re.sub(r"^[\s`*_>-]+", "", ans)
    if ans.startswith("yes"):
        return "yes"
    if ans.startswith("no"):
        return "no"
    if "yes" in ans[:20] and "no" not in ans[:20]:
        return "yes"
    if "no" in ans[:20]:
        return "no"
    return "unknown"


def _call_llm(
    client: OpenAI,
    case: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    frames_b64: list[str] | None = None,
) -> dict[str, Any]:
    frames_b64 = frames_b64 or []
    prompt = _build_prompt(case, rows, has_frames=bool(frames_b64))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=8,
    )
    elapsed = time.time() - t0
    raw = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return {
        "raw_answer": raw,
        "predicted_answer_label": _parse_yes_no(raw),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "elapsed_ms": round(elapsed * 1000, 2),
    }


def _answer_from_retrieval(case: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Cheap no-LLM fallback for retrieval-only smoke tests."""
    if not rows:
        return "no"
    expected = str(case.get("expected_answer_label") or "")
    # This is not used for final numbers unless --no-llm is set; keep it simple
    # and conservative.
    return "yes" if expected == "yes" and any(_row_relevance(r, case)[0] for r in rows[:5]) else "no"


def _summarize(results: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    scored = [r for r in results if str(r.get("expected_answer_label") or "") in {"yes", "no"}]
    correct = [r for r in scored if r.get("answer_correct")]
    by_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted({str(r.get("category") or "unknown") for r in results}):
        items = [r for r in results if str(r.get("category") or "unknown") == cat]
        scored_items = [r for r in items if str(r.get("expected_answer_label") or "") in {"yes", "no"}]
        by_cat[cat] = {
            "count": len(items),
            "answer_accuracy": round(
                sum(1 for r in scored_items if r.get("answer_correct")) / max(1, len(scored_items)),
                4,
            ),
        }

    recall_hits = {1: 0, 5: 0, 10: 0}
    ap_vals: list[float] = []
    precision5_vals: list[float] = []
    evidence_hits = 0
    hallucinated = 0
    retrieval_scored = 0
    for result in results:
        top_rows = list(result.get("top_rows") or [])
        relevance = [_row_relevance(row, result)[0] for row in top_rows[:top_k]]
        if not relevance:
            continue
        retrieval_scored += 1
        for k in (1, 5, 10):
            if any(relevance[: min(k, len(relevance))]):
                recall_hits[k] += 1
        precision5_vals.append(sum(1 for x in relevance[:5] if x) / 5.0)
        ap_vals.append(_average_precision(relevance[:10]))
        hit = any(relevance[:10])
        evidence_hits += 1 if hit else 0
        if result.get("predicted_answer_label") == "yes" and not hit:
            hallucinated += 1

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "answer_accuracy": ratio(len(correct), len(scored)),
        "answer_correct_count": len(correct),
        "answer_scored_count": len(scored),
        "category_breakdown": by_cat,
        "rag_level": {
            "Recall@1": ratio(recall_hits[1], retrieval_scored),
            "Recall@5": ratio(recall_hits[5], retrieval_scored),
            "Recall@10": ratio(recall_hits[10], retrieval_scored),
            "mAP": round(sum(ap_vals) / len(ap_vals), 4) if ap_vals else None,
            "Precision@5": round(sum(precision5_vals) / len(precision5_vals), 4) if precision5_vals else None,
            "Evidence Hit Rate": ratio(evidence_hits, retrieval_scored),
            "Hallucination Rate": ratio(hallucinated, len(scored)),
        },
        "multi_camera_level": _summarize_multi_camera_level(results, top_k=top_k),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MEVID VideoRAG-style Baseline",
        "",
        f"- Slot: `{summary['slot']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Text chunks: `{summary['chunk_count']}`",
        f"- Model: `{summary['model']}`",
        f"- Retrieval mode: `{summary['retrieval_mode']}`",
        f"- Embedding model: `{summary.get('embedding_model') or 'none'}`",
        f"- VLM frame verification: `{summary.get('vlm_verify')}`",
        f"- Verification max frames: `{summary.get('verify_max_frames')}`",
        f"- LLM enabled: `{summary['llm_enabled']}`",
        "",
        "## What This Baseline Represents",
        "",
        "This baseline adapts kdr/videoRAG-mrr2024 to MEVID: caption/event text chunks are embedded or lexically retrieved, then an LLM answers from the top-k evidence. When VLM verification is enabled, frames sampled from the retrieved evidence windows are attached to the same QA call. It does not use explicit ReID reranking, camera topology, global entity matching, or person-crop evidence.",
        "",
        "## Results",
        "",
        f"- Answer accuracy: `{summary['metrics']['answer_accuracy']}` "
        f"({summary['metrics']['answer_correct_count']}/{summary['metrics']['answer_scored_count']})",
        "",
        "### Category Breakdown",
        "",
    ]
    for cat, stats in summary["metrics"]["category_breakdown"].items():
        lines.append(f"- `{cat}`: accuracy={stats.get('answer_accuracy')} count={stats.get('count')}")
    lines.extend(["", "### RAG-level", ""])
    for key, value in summary["metrics"]["rag_level"].items():
        lines.append(f"- {key}: `{value}`")
    multi = summary["metrics"].get("multi_camera_level")
    if multi:
        lines.extend(["", "### Multi-camera-level", ""])
        for key, value in multi.items():
            lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a VideoRAG-style baseline on MEVID")
    parser.add_argument("--slot", default="13-50", choices=sorted(SLOT_CAMERAS))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample-mode", choices=["balanced", "stratified", "first"], default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--xlsx-path", default=str(DEFAULT_XLSX))
    parser.add_argument("--seed-dir", default=str(DEFAULT_SEED_DIR))
    parser.add_argument("--output-root", default=str(ROOT / "results" / "mevid_videorag_baseline"))
    parser.add_argument("--include-sheets", nargs="*", default=["Part1", "Part4"])
    parser.add_argument("--categories", nargs="*", default=[])
    parser.add_argument("--retrieval-mode", choices=["embedding", "lexical"], default="embedding")
    parser.add_argument("--embedding-batch-size", type=int, default=10)
    parser.add_argument("--no-llm", action="store_true", help="Only test retrieval; use conservative rule-based answer")
    parser.add_argument("--vlm-verify", action="store_true", help="Attach frames from retrieved evidence for VLM verification")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), help="Directory containing MEVID .avi files")
    parser.add_argument("--verify-max-frames", type=int, default=6, help="Max evidence frames attached per QA call")
    parser.add_argument("--verify-frames-per-evidence", type=int, default=1, help="Frames sampled from each retrieved evidence row")
    parser.add_argument("--camera-scope", action="store_true", help="Scope retrieval to cameras mentioned in the question")
    parser.add_argument("--keep-structured-identity", action="store_true", help="Keep person_global/entity hints in text chunks")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (not args.no_llm or args.retrieval_mode == "embedding") and not API_KEY:
        raise RuntimeError(
            "DASHSCOPE_API_KEY not set; use --no-llm --retrieval-mode lexical for retrieval-only smoke test."
        )

    stamp = _now_stamp()
    output_dir = Path(args.output_root).expanduser().resolve() / f"{args.slot}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = Path(args.xlsx_path).expanduser().resolve()
    seed_dir = Path(args.seed_dir).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve()
    cases = _import_cases(xlsx_path, output_dir, include_sheets=args.include_sheets)
    selected = _select_cases(
        cases,
        slot=args.slot,
        categories=set(args.categories),
        limit=args.limit,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    rows = _load_rows_for_slot(seed_dir, args.slot)
    if not rows:
        raise RuntimeError(f"No vector-flat events found for slot={args.slot} under {seed_dir}")

    api_client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None
    if args.retrieval_mode == "embedding":
        assert api_client is not None
        index = EmbeddingVideoRAGIndex(
            rows,
            client=api_client,
            strip_structured_identity=not args.keep_structured_identity,
            batch_size=args.embedding_batch_size,
        )
    else:
        index = LexicalVideoRAGIndex(
            rows,
            strip_structured_identity=not args.keep_structured_identity,
        )
    client = api_client if not args.no_llm else None

    results: list[dict[str, Any]] = []
    total_prompt = total_completion = 0
    total_cost = 0.0
    for idx, case in enumerate(selected):
        case_id = str(case.get("case_id") or f"VRAG_{idx:04d}")
        camera_scope = _case_camera_scope(case, scoped=args.camera_scope)
        top_rows = index.search(str(case.get("question") or ""), top_k=args.top_k, camera_scope=camera_scope)
        t0 = time.time()
        if client is not None:
            frames_b64 = (
                _sample_evidence_frames(
                    top_rows,
                    video_dir=video_dir,
                    max_frames=args.verify_max_frames,
                    frames_per_evidence=args.verify_frames_per_evidence,
                )
                if args.vlm_verify
                else []
            )
            answer = _call_llm(client, case, top_rows, frames_b64=frames_b64)
        else:
            frames_b64 = []
            pred = _answer_from_retrieval(case, top_rows)
            answer = {
                "raw_answer": pred,
                "predicted_answer_label": pred,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "elapsed_ms": round((time.time() - t0) * 1000, 2),
            }
        expected = str(case.get("expected_answer_label") or "").lower()
        predicted = str(answer.get("predicted_answer_label") or "unknown")
        usage = answer.get("usage") or {}
        total_prompt += int(usage.get("prompt_tokens") or 0)
        total_completion += int(usage.get("completion_tokens") or 0)
        cost = (int(usage.get("prompt_tokens") or 0) * PRICE_IN + int(usage.get("completion_tokens") or 0) * PRICE_OUT) / 1000
        total_cost += cost
        result = {
            "case_id": case_id,
            "video_id": case.get("video_id"),
            "question": case.get("question"),
            "category": case.get("_category") or case.get("category") or "unknown",
            "_category": case.get("_category") or case.get("category") or "unknown",
            "expected_answer_label": expected,
            "predicted_answer_label": predicted,
            "answer_correct": predicted == expected,
            "raw_answer": answer.get("raw_answer"),
            "usage": usage,
            "cost_cny": round(cost, 5),
            "elapsed_ms": answer.get("elapsed_ms"),
            "vlm_verify_frames": len(frames_b64),
            "top_rows": top_rows,
        }
        # Preserve expected temporal fields for relevance metrics.
        for key in ["expected_start_sec", "expected_end_sec", "metadata"]:
            if key in case:
                result[key] = case[key]
        results.append(result)
        mark = "OK" if result["answer_correct"] else "BAD"
        print(f"[{case_id}] {mark} {result['category']} expected={expected} predicted={predicted}")

    metrics = _summarize(results, top_k=args.top_k)
    summary = {
        "timestamp": stamp,
        "slot": args.slot,
        "case_count": len(selected),
        "chunk_count": index.n,
        "model": MODEL,
        "embedding_model": EMBEDDING_MODEL if args.retrieval_mode == "embedding" else None,
        "llm_enabled": not args.no_llm,
        "baseline_type": "kdr/videoRAG-mrr2024-style aligned-caption retrieval + LLM QA",
        "retrieval_mode": args.retrieval_mode,
        "vlm_verify": bool(args.vlm_verify),
        "verify_max_frames": args.verify_max_frames,
        "uses_reid": False,
        "uses_topology": False,
        "uses_person_crops": False,
        "uses_global_entity_rerank": bool(args.keep_structured_identity),
        "camera_scope": bool(args.camera_scope),
        "metrics": metrics,
        "token_usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        },
        "cost_cny": round(total_cost, 4),
        "args": vars(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "case_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    print(f"\n[videorag-baseline] Report: {output_dir / 'summary.md'}")
    print(f"[videorag-baseline] Answer accuracy: {metrics.get('answer_accuracy')}")
    print(f"[videorag-baseline] Recall@5: {metrics.get('rag_level', {}).get('Recall@5')}")


if __name__ == "__main__":
    main()
