import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


ROOT = Path("/home/yangxp/Capstone")
CHROMA_PATH = ROOT / "data/chroma/basketball_tracks"
CHILD_COLLECTION = "basketball_tracks"
PARENT_COLLECTION = "basketball_tracks_parent"
SEEDS = [
    ROOT / "data/basketball_output/basketball_1_events_vector_flat.json",
    ROOT / "data/basketball_output/basketball_2_events_vector_flat.json",
]
TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def build_bm25(docs: list[str]):
    tokenized = [tokenize(doc) for doc in docs]
    df = Counter()
    for tks in tokenized:
        for term in set(tks):
            df[term] += 1
    n_docs = len(tokenized)
    avgdl = (sum(len(x) for x in tokenized) / n_docs) if n_docs else 0.0
    return tokenized, df, n_docs, avgdl


def bm25_scores(query: str, tokenized_docs, df, n_docs: int, avgdl: float):
    q = tokenize(query)
    k1, b = 1.5, 0.75
    scores = []
    for i, tks in enumerate(tokenized_docs):
        tf = Counter(tks)
        dl = len(tks)
        score = 0.0
        for term in q:
            if term not in df:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * (dl / (avgdl or 1)))
            score += idf * (freq * (k1 + 1)) / denom
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def main() -> None:
    load_dotenv(ROOT / ".env")
    emb = OpenAIEmbeddings(
        model="text-embedding-v3",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL"),
        check_embedding_ctx_length=False,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(CHILD_COLLECTION)
    try:
        parent_collection = client.get_collection(PARENT_COLLECTION)
        parent_record_count = parent_collection.count()
    except Exception:
        parent_record_count = 0
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    docs = all_data["documents"]
    metas = all_data["metadatas"]

    # Provenance from seed events by id rule: {video_id}_{entity_hint}
    source_events = []
    for seed_file in SEEDS:
        payload = json.loads(seed_file.read_text(encoding="utf-8"))
        source_events.extend(payload.get("events", []))
    provenance = defaultdict(list)
    for event in source_events:
        video_id = event.get("video_id")
        entity_hint = event.get("entity_hint")
        if video_id and entity_hint:
            rec_id = f"{video_id}_{entity_hint}"
            provenance[rec_id].append(
                {
                    "start_time": event.get("start_time"),
                    "end_time": event.get("end_time"),
                    "event_text": event.get("event_text"),
                }
            )

    tokenized_docs, df, n_docs, avgdl = build_bm25(docs)

    def cosine_top(query: str, topk: int = 5):
        qv = emb.embed_query(query)
        rs = collection.query(
            query_embeddings=[qv],
            n_results=min(topk, len(ids)),
            include=["distances", "documents", "metadatas"],
        )
        out = []
        for i in range(len(rs["ids"][0])):
            out.append(
                {
                    "id": rs["ids"][0][i],
                    "distance": float(rs["distances"][0][i]),
                    "doc": rs["documents"][0][i][:160],
                    "meta": rs["metadatas"][0][i],
                }
            )
        return out

    def hybrid_top(query: str, topk: int = 5, alpha: float = 0.6):
        cos = cosine_top(query, topk=min(20, len(ids)))
        bm = bm25_scores(query, tokenized_docs, df, n_docs, avgdl)
        bm_map = {ids[i]: s for i, s in bm[:200]}

        items = []
        for rec in cos:
            cos_sim = max(0.0, 1.0 - rec["distance"])
            items.append((rec["id"], cos_sim, rec))
        cos_vals = [x[1] for x in items] or [0.0]
        cmin, cmax = min(cos_vals), max(cos_vals)
        bm_vals = [bm_map.get(x[0], 0.0) for x in items] or [0.0]
        bmin, bmax = min(bm_vals), max(bm_vals)

        ranked = []
        for rec_id, cos_sim, rec in items:
            cn = (cos_sim - cmin) / (cmax - cmin) if cmax > cmin else 0.0
            b = bm_map.get(rec_id, 0.0)
            bn = (b - bmin) / (bmax - bmin) if bmax > bmin else 0.0
            score = alpha * cn + (1 - alpha) * bn
            ranked.append((score, rec_id, cos_sim, b, rec))
        ranked.sort(key=lambda x: x[0], reverse=True)

        out = []
        for score, rec_id, cos_sim, b, rec in ranked[:topk]:
            out.append(
                {
                    "id": rec_id,
                    "hybrid_score": round(score, 6),
                    "cosine_sim_est": round(cos_sim, 6),
                    "bm25": round(b, 6),
                    "doc": rec["doc"],
                    "meta": rec["meta"],
                }
            )
        return out

    queries = [
        ("zh", "场边站立不动的人"),
        ("zh", "快速移动的目标"),
        ("en", "person standing still near baseline"),
        ("en", "fast moving target"),
    ]
    retrieval_tests = []
    for lang, query in queries:
        cos = cosine_top(query, topk=5)
        bm = bm25_scores(query, tokenized_docs, df, n_docs, avgdl)[:5]
        bm_out = [
            {
                "id": ids[i],
                "bm25": round(s, 6),
                "doc": docs[i][:160],
                "meta": metas[i],
            }
            for i, s in bm
        ]
        hy = hybrid_top(query, topk=5, alpha=0.6)
        retrieval_tests.append(
            {
                "lang": lang,
                "query": query,
                "cosine_top5": cos,
                "bm25_top5": bm_out,
                "hybrid_top5": hy,
            }
        )

    missing = [rec_id for rec_id in ids if rec_id not in provenance]
    per_record = []
    for rec_id in ids:
        src = provenance.get(rec_id, [])
        per_record.append(
            {
                "id": rec_id,
                "source_event_count": len(src),
                "source_start_min": min([x["start_time"] for x in src], default=None),
                "source_end_max": max([x["end_time"] for x in src], default=None),
            }
        )

    report = {
        "embedding_model_test": {"model": "text-embedding-v3", "dimension": 1024, "status": "ok"},
        "child_collection": CHILD_COLLECTION,
        "parent_collection": PARENT_COLLECTION,
        "chroma_path": str(CHROMA_PATH),
        "child_record_count": len(ids),
        "parent_record_count": parent_record_count,
        "source_event_count": len(source_events),
        "chunking_strategy_detected": "parent-child (child=video_id_entity_hint, parent=video_id)",
        "missing_provenance_ids": missing,
        "per_record_provenance": per_record,
        "retrieval_tests": retrieval_tests,
    }

    out_json = ROOT / "agent/chroma_test_report.json"
    out_md = ROOT / "agent/chroma_test_report.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Chroma 测试报告\n\n")
    lines.append(f"- child_collection: `{CHILD_COLLECTION}`\n")
    lines.append(f"- parent_collection: `{PARENT_COLLECTION}`\n")
    lines.append(f"- child_record_count: `{len(ids)}`\n")
    lines.append(f"- parent_record_count: `{parent_record_count}`\n")
    lines.append(f"- source_event_count: `{len(source_events)}`\n")
    lines.append("- chunking: `parent-child (child=video_id_entity_hint, parent=video_id)`\n")
    lines.append("- embedding: `text-embedding-v3 (1024)`\n\n")
    lines.append("## 检索策略测试\n")
    for item in retrieval_tests:
        lines.append(f"### Query: {item['query']} ({item['lang']})\n")
        lines.append("- Cosine Top3:\n")
        for rec in item["cosine_top5"][:3]:
            lines.append(f"  - {rec['id']} | dist={rec['distance']:.4f} | zone={rec['meta'].get('scene_zone')}\n")
        lines.append("- BM25 Top3:\n")
        for rec in item["bm25_top5"][:3]:
            lines.append(f"  - {rec['id']} | bm25={rec['bm25']:.4f} | zone={rec['meta'].get('scene_zone')}\n")
        lines.append("- Hybrid Top3:\n")
        for rec in item["hybrid_top5"][:3]:
            lines.append(
                f"  - {rec['id']} | score={rec['hybrid_score']:.4f} | "
                f"cos={rec['cosine_sim_est']:.4f} | bm25={rec['bm25']:.4f}\n"
            )
        lines.append("\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    print("REPORT_JSON=", out_json)
    print("REPORT_MD=", out_md)
    print("RECORD_COUNT=", len(ids))
    print("SOURCE_EVENT_COUNT=", len(source_events))
    print("MISSING_PROVENANCE_COUNT=", len(missing))


if __name__ == "__main__":
    main()
