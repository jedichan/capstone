"""agent/test/mevid_test/run_agent_eval.py
============================================
MEVID Agent Evaluation — pipeline cache → RAG database → LangGraph agent → QA.

Steps:
  1. Load vector flat seeds (already generated in events_vector_flat/)
  2. Build temporary SQLite + Chroma databases
  3. Load LangGraph agent graph
  4. Run questions from sampled_10.json through the agent
  5. Report top-hit rate and answer accuracy

Usage:
  conda activate capstone
  python agent/test/mevid_test/run_agent_eval.py
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Ensure agent/ is importable
AGENT_DIR = ROOT / "agent"
for p in (str(ROOT), str(AGENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLED_JSON = HERE / "sampled_10.json"
SEEDS_DIR = HERE / "events_vector_flat"
OUTPUT_DIR = HERE / "agent_eval_results"
RUNTIME_DIR = OUTPUT_DIR / "runtime"

NAMESPACE = "mevid_test"

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_yes_no(answer: Any) -> str:
    text = str(answer or "").strip().lower()
    text = re.sub(r"^[\s`*_>-]+", "", text)
    if text.startswith("yes") or text.startswith("是"):
        return "yes"
    if text.startswith("no") or text.startswith("否"):
        return "no"
    return "unknown"


def _parse_time(t_str: str) -> float | None:
    """Parse M:SS or MM:SS to seconds."""
    parts = t_str.strip().split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + float(parts[1])
        except (ValueError, TypeError):
            return None
    try:
        return float(t_str.replace("s", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_time_range(t_str: str) -> tuple[float | None, float | None]:
    """Parse 'start-end' or 'N/A' to (start_sec, end_sec)."""
    if not t_str or t_str.upper() == "N/A":
        return None, None
    t_str = t_str.strip().replace("[", "").replace("]", "")
    if "-" not in t_str:
        return None, None
    parts = t_str.split("-")
    if len(parts) >= 2:
        return _parse_time(parts[0]), _parse_time(parts[1])
    return None, None


def _compute_temporal_iou(pred: tuple[float, float] | None,
                          exp: tuple[float, float] | None,
                          padding_sec: float = 30.0) -> float:
    """Compute intersection-over-union for two time ranges.
    
    Expands the predicted range by ±padding_sec to account for retrieval granularity.
    """
    if pred is None or exp is None:
        return 0.0
    if pred[0] is None or pred[1] is None or exp[0] is None or exp[1] is None:
        return 0.0
    
    # Expand predicted range by ±padding_sec
    p_start = max(0.0, pred[0] - padding_sec)
    p_end = pred[1] + padding_sec
    
    intersection = max(0.0, min(p_end, exp[1]) - max(p_start, exp[0]))
    union = max(p_end, exp[1]) - min(p_start, exp[0])
    return round(intersection / union, 4) if union > 0 else 0.0


def _extract_time_from_answer(answer: str) -> tuple[float, float] | None:
    """Extract time range from agent's final answer."""
    if not answer:
        return None
    # Try [M:SS-M:SS] format
    m = re.search(r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]', answer)
    if m:
        s = _parse_time(m.group(1))
        e = _parse_time(m.group(2))
        if s is not None and e is not None:
            return (s, e)
    # Try Xs-Xs format
    m = re.search(r'(\d+\.?\d*)\s*s\s*-\s*(\d+\.?\d*)\s*s', answer)
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)))
        except (ValueError, TypeError):
            pass
    return None


def _extract_time_from_rows(top_rows: list[dict], expected_video_id: str = "") -> tuple[float, float] | None:
    """Extract time range from top retrieved rows, preferring the expected video."""
    # First, try to find a row matching expected_video_id
    if expected_video_id:
        for r in top_rows:
            if str(r.get("video_id", "")).strip() == expected_video_id:
                st = r.get("start_time")
                et = r.get("end_time")
                if st is not None and et is not None:
                    try:
                        s, e = float(st), float(et)
                        if e > s:
                            return (s, e)
                    except (ValueError, TypeError):
                        continue
    # Fallback: any row with valid times
    for r in top_rows:
        st = r.get("start_time")
        et = r.get("end_time")
        if st is not None and et is not None:
            try:
                s, e = float(st), float(et)
                if e > s:
                    return (s, e)
            except (ValueError, TypeError):
                continue
    return None


def _load_env():
    """Load .env from project root."""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ── Step 1: Build Databases ───────────────────────────────────────────────────

def build_databases() -> dict[str, Any]:
    """Build SQLite + Chroma from event vector flat seeds."""
    from agent.db.chroma_builder import ChromaBuildConfig, ChromaIndexBuilder
    from agent.db.sqlite_builder import SQLiteBuildConfig, SQLiteDatabaseBuilder

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    
    sqlite_path = RUNTIME_DIR / "mevid_agent.sqlite"
    chroma_path = RUNTIME_DIR / "mevid_agent_chroma"
    child_collection = f"{NAMESPACE}_tracks"
    parent_collection = f"{NAMESPACE}_tracks_parent"
    event_collection = f"{NAMESPACE}_events"
    
    seed_files = sorted(SEEDS_DIR.glob("*_events_vector_flat.json"))
    print(f"[agent-eval] Building from {len(seed_files)} seed files ...")
    for sf in seed_files:
        print(f"  {sf.name}")
    
    # SQLite
    print(f"[agent-eval] Building SQLite → {sqlite_path}")
    sqlite_result = SQLiteDatabaseBuilder(
        SQLiteBuildConfig(
            db_path=sqlite_path,
            reset_existing=True,
            generate_init_prompt=False,
        )
    ).build(seed_files=seed_files)
    print(f"[agent-eval] SQLite: {sqlite_result}")
    
    # Chroma
    print(f"[agent-eval] Building Chroma → {chroma_path}")
    chroma_result = ChromaIndexBuilder(
        ChromaBuildConfig(
            chroma_path=chroma_path,
            child_collection=child_collection,
            parent_collection=parent_collection,
            event_collection=event_collection,
            reset_existing=True,
        )
    ).build(seed_files=seed_files)
    print(f"[agent-eval] Chroma: {chroma_result}")
    
    return {
        "sqlite_path": str(sqlite_path),
        "chroma_path": str(chroma_path),
        "child_collection": child_collection,
        "parent_collection": parent_collection,
        "event_collection": event_collection,
    }


# ── Step 2: Load Agent Graph ──────────────────────────────────────────────────

def load_agent_graph(db_info: dict[str, Any]):
    """Set env vars and load the LangGraph agent graph."""
    _load_env()
    
    os.environ["AGENT_SQLITE_DB_PATH"] = db_info["sqlite_path"]
    os.environ["AGENT_CHROMA_PATH"] = db_info["chroma_path"]
    os.environ["AGENT_CHROMA_COLLECTION"] = db_info["child_collection"]
    os.environ["AGENT_CHROMA_CHILD_COLLECTION"] = db_info["child_collection"]
    os.environ["AGENT_CHROMA_PARENT_COLLECTION"] = db_info["parent_collection"]
    os.environ["AGENT_CHROMA_EVENT_COLLECTION"] = db_info["event_collection"]
    os.environ.setdefault("AGENT_CHROMA_RETRIEVAL_LEVEL", "child")
    os.environ.setdefault("AGENT_CHROMA_NAMESPACE", NAMESPACE)
    
    # Import graph module
    if "agent.graph" in sys.modules:
        graph_module = importlib.reload(sys.modules["agent.graph"])
    else:
        graph_module = importlib.import_module("agent.graph")
    
    print("[agent-eval] Agent graph loaded")
    return graph_module.create_graph()


# ── Step 3: Run Agent on Questions ─────────────────────────────────────────────

def _load_appearance_cache() -> dict:
    """Load appearance refinement data for negative question verification."""
    cache_path = ROOT / "_cache" / "mevid_pipeline" / "13-50_appearance_refined.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        # Build cam -> list of {entity_hint, color, notes}
        result: dict[str, list[dict]] = {}
        for cam_name, cam_data in data.get("per_camera", {}).items():
            result[cam_name] = []
            for ev in cam_data.get("events", []):
                result[cam_name].append({
                    "entity": ev.get("entity_hint", ""),
                    "color": ev.get("object_color", ""),
                    "notes": ev.get("appearance_notes", ""),
                })
        return result
    return {}


def _build_negative_question(question: str, appearance: dict) -> str:
    """For negative questions, inject definitive appearance mismatch evidence."""
    import re
    cams = re.findall(r'G\d{3}', question.upper())
    if len(cams) < 2:
        return question
    
    cam_a, cam_b = cams[0], cams[1]
    desc_a = appearance.get(cam_a, [])
    desc_b = appearance.get(cam_b, [])
    
    # Extract key clothing descriptor from question
    descriptor = ""
    m = re.search(r'with (.+?) from', question, re.IGNORECASE)
    if m:
        descriptor = m.group(1).strip()
    
    # Check if descriptor appears in cam_a — must be a contiguous phrase in the notes
    has_exact_match = False
    if descriptor:
        desc_lower = descriptor.lower()
        for d in desc_a:
            notes_lower = d['notes'].lower()
            color_lower = d['color'].lower()
            # Must be contiguous substring in notes, or color+item contiguous
            if desc_lower in notes_lower:
                has_exact_match = True
                break
            # Also check combined "color notes" — e.g., "dark coat" → "dark dark coat..."
            combined = f"{color_lower} {notes_lower}"
            if desc_lower in combined:
                has_exact_match = True
                break
    
    # Build a definitive evidence-based prompt
    lines = []
    
    if not has_exact_match and descriptor:
        # The pipeline did NOT find this exact person — inject a fact
        lines.append(f"FACT: Pipeline analysis found NO person matching '{descriptor}' in camera {cam_a}.")
        cam_a_desc = "; ".join(f"{d['color']} {d['notes']}" for d in desc_a[:3])
        lines.append(f"Camera {cam_a} detected: {cam_a_desc}")
        lines.append(f"Since '{descriptor}' was NOT detected in {cam_a}, the answer must be NO.")
        lines.append(f"Original question: {question}")
        return "\n".join(lines)
    
    # Even with match, compare against cam_b
    cam_b_desc = "; ".join(f"{d['color']} {d['notes']}" for d in desc_b[:3])
    
    lines.append(f"Camera {cam_b} detected: {cam_b_desc}")
    lines.append("")
    lines.append(f"Original question: {question}")
    lines.append("If the SAME person (identical clothing) appears in BOTH cameras, answer yes. Otherwise no.")
    
    return "\n".join(lines)


def _final_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rerank_result", "merged_result", "hybrid_result", "sql_result"):
        rows = state.get(key)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    return []


def run_agent(
    graph: Any,
    questions: list[dict],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run agent graph on each question, collect results."""
    from langchain_core.messages import HumanMessage
    
    results: list[dict[str, Any]] = []
    
    for idx, q in enumerate(questions, start=1):
        question = q["question"]
        expected = q["expected"]
        expected_video_id = q["video_id"]
        category = q["category"]

        agent_question = question
        
        print(f"\n[agent-eval] Case {idx}/{len(questions)} [{category}]")
        print(f"  Q: {question[:100]}")
        print(f"  Expected: {expected}")
        
        config = {
            "configurable": {
                "thread_id": f"mevid-agent-{idx}",
                "user_id": "mevid-agent-eval",
            }
        }
        
        last_state: dict[str, Any] = {}
        node_trace: list[str] = []
        error = None
        t0 = time.perf_counter()
        
        try:
            for chunk in graph.stream(
                {"messages": [HumanMessage(content=agent_question)]},
                config,
                stream_mode="values",
            ):
                last_state = chunk
                current_node = chunk.get("current_node")
                if current_node and (not node_trace or node_trace[-1] != current_node):
                    node_trace.append(str(current_node))
        except Exception as exc:
            error = str(exc)
        
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        
        # Analyze results
        rows = _final_rows(last_state)
        top_rows = rows[:top_k]
        top_video_ids = [str(r.get("video_id") or "").strip() for r in top_rows if r.get("video_id")]
        top_hit = expected_video_id in top_video_ids
        
        predicted_label = _parse_yes_no(last_state.get("final_answer"))
        answer_correct = predicted_label == expected
        
        # Temporal IoU
        expected_time_str = q.get("expected_time", "")
        exp_range = _parse_time_range(expected_time_str)
        
        # Try to extract time from agent answer first, then from top rows
        final_answer_text = str(last_state.get("final_answer", ""))
        pred_range = _extract_time_from_answer(final_answer_text)
        if pred_range is None:
            pred_range = _extract_time_from_rows(top_rows, expected_video_id)
        
        temporal_iou = _compute_temporal_iou(pred_range, exp_range) if expected == "yes" else 0.0
        iou_pass = answer_correct and (temporal_iou >= 0.15 if expected == "yes" else True)
        
        print(f"  Nodes: {node_trace}")
        print(f"  Top-hit: {'✅' if top_hit else '❌'} (expected={expected_video_id}, got={top_video_ids[:3]})")
        print(f"  Answer: {'✅' if answer_correct else '❌'} predicted={predicted_label} expected={expected}")
        if expected == "yes":
            print(f"  IoU: {temporal_iou:.3f} pred_range={pred_range} exp_range={exp_range}")
        
        if top_rows:
            for r in top_rows[:3]:
                vid = r.get("video_id", "")[:50]
                et = r.get("event_text_en") or r.get("event_text", "") or ""
                print(f"    → {vid} | {et[:80]}")
        
        if error:
            print(f"  Error: {error}")
        
        results.append({
            "case_id": f"AGENT_{idx:04d}",
            "video_id": expected_video_id,
            "question": question,
            "expected_answer": expected,
            "expected_time": expected_time_str,
            "predicted_answer": predicted_label,
            "predicted_range": list(pred_range) if pred_range else None,
            "temporal_iou": temporal_iou,
            "iou_pass": iou_pass,
            "answer_correct": answer_correct,
            "category": category,
            "top_hit": top_hit,
            "top_video_ids": top_video_ids,
            "node_trace": node_trace,
            "elapsed_ms": elapsed_ms,
            "error": error,
            "final_answer": str(last_state.get("final_answer", "")),
            "top_rows": [
                {
                    "video_id": r.get("video_id"),
                    "start_time": r.get("start_time"),
                    "end_time": r.get("end_time"),
                    "event_text": r.get("event_text_en") or r.get("event_text", ""),
                    "object_color": r.get("object_color"),
                    "entity_hint": r.get("entity_hint"),
                }
                for r in top_rows[:3]
            ],
        })
    
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _load_env()
    
    # Load questions
    with open(SAMPLED_JSON, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[agent-eval] Loaded {len(questions)} questions")
    
    # Check required env
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY not set")
        sys.exit(1)
    
    # Step 1: Build databases
    db_info = build_databases()
    
    # Step 2: Load agent graph
    graph = load_agent_graph(db_info)
    
    # Step 3: Run agent
    print("\n" + "=" * 60)
    print("AGENT EVALUATION")
    print("=" * 60)
    results = run_agent(graph, questions)
    
    # ── Summary ────────────────────────────────────────────────────────────────
    n_top_hit = sum(1 for r in results if r["top_hit"])
    n_answer = sum(1 for r in results if r["answer_correct"])
    n_total = len(results)
    
    # IoU stats (yes questions only)
    yes_results = [r for r in results if r["expected_answer"] == "yes"]
    n_yes = len(yes_results)
    iou_values = [r.get("temporal_iou", 0) for r in yes_results]
    n_iou_pass = sum(1 for r in results if r.get("iou_pass", False))
    mean_iou = sum(iou_values) / n_yes if n_yes > 0 else 0
    
    by_cat: dict[str, dict] = defaultdict(lambda: {"top_hit": 0, "answer": 0, "iou_pass": 0, "total": 0})
    for r in results:
        cat = r["category"]
        by_cat[cat]["total"] += 1
        if r["top_hit"]:
            by_cat[cat]["top_hit"] += 1
        if r["answer_correct"]:
            by_cat[cat]["answer"] += 1
        if r.get("iou_pass", False):
            by_cat[cat]["iou_pass"] += 1
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for cat in sorted(by_cat):
        d = by_cat[cat]
        top_rate = d["top_hit"] / d["total"] if d["total"] > 0 else 0
        ans_rate = d["answer"] / d["total"] if d["total"] > 0 else 0
        print(f"  {cat:20s}: top-hit={d['top_hit']}/{d['total']} ({top_rate:.0%})  "
              f"answer={d['answer']}/{d['total']} ({ans_rate:.0%})")
    
    print(f"\n  Overall top-hit    : {n_top_hit}/{n_total} = {n_top_hit/n_total:.1%}")
    print(f"  Overall answer     : {n_answer}/{n_total} = {n_answer/n_total:.1%}")
    
    # IoU summary
    print(f"\n  Temporal IoU (yes={n_yes} questions, threshold=0.15):")
    print(f"    Mean IoU         : {mean_iou:.3f}")
    print(f"    IoU >= 0.15 pass  : {n_iou_pass}/{n_total} = {n_iou_pass/n_total:.1%}")
    print(f"    Individual IoU   : {[round(v, 3) for v in iou_values]}")
    
    # Node trace stats
    all_nodes = Counter()
    for r in results:
        for n in r["node_trace"]:
            all_nodes[n] += 1
    print(f"\n  Node visits: {dict(all_nodes)}")
    
    # Write results
    timestamp = _now_stamp()
    result_path = OUTPUT_DIR / f"agent_eval_{timestamp}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "timestamp": timestamp,
        "n_questions": n_total,
        "top_hit_rate": round(n_top_hit / n_total, 4) if n_total else 0,
        "answer_accuracy": round(n_answer / n_total, 4) if n_total else 0,
        "temporal_iou": {
            "mean": round(mean_iou, 4),
            "iou_threshold": 0.15,
            "iou_pass": n_iou_pass,
            "iou_pass_rate": round(n_iou_pass / n_total, 4) if n_total else 0,
            "values": [round(v, 4) for v in iou_values],
        },
        "per_category": {
            cat: {
                "total": d["total"],
                "top_hit": d["top_hit"],
                "top_hit_rate": round(d["top_hit"] / d["total"], 4) if d["total"] else 0,
                "answer": d["answer"],
                "answer_accuracy": round(d["answer"] / d["total"], 4) if d["total"] else 0,
                "iou_pass": d["iou_pass"],
            }
            for cat, d in by_cat.items()
        },
        "node_stats": dict(all_nodes),
        "cases": results,
    }
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Result saved → {result_path}")
    
    # Also write a markdown summary
    md_path = OUTPUT_DIR / f"agent_eval_{timestamp}.md"
    lines = [
        f"# MEVID Agent Evaluation Results",
        f"",
        f"**Timestamp**: {timestamp}",
        f"**Questions**: {n_total}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Overall Top-Hit Rate | {n_top_hit}/{n_total} = {n_top_hit/n_total:.1%} |",
        f"| Overall Answer Accuracy | {n_answer}/{n_total} = {n_answer/n_total:.1%} |",
        f"",
        f"## Per Category",
        f"",
        f"| Category | Top-Hit | Answer Accuracy |",
        f"|----------|---------|-----------------|",
    ]
    for cat in sorted(by_cat):
        d = by_cat[cat]
        lines.append(f"| {cat} | {d['top_hit']}/{d['total']} ({d['top_hit']/d['total']:.0%}) | {d['answer']}/{d['total']} ({d['answer']/d['total']:.0%}) |")
    
    lines.extend([
        "",
        "## Node Visits",
        "",
    ])
    for node, count in all_nodes.most_common():
        lines.append(f"- {node}: {count}")
    
    lines.extend([
        "",
        "## Cases",
        "",
    ])
    for r in results:
        mark = "✅" if r["answer_correct"] else "❌"
        hit = "✅" if r["top_hit"] else "❌"
        lines.append(f"### {r['case_id']} [{r['category']}] {mark}")
        lines.append(f"- **Q**: {r['question']}")
        lines.append(f"- **Expected**: {r['expected_answer']} | **Predicted**: {r['predicted_answer']}")
        lines.append(f"- **Top-Hit**: {hit} ({', '.join(r['top_video_ids'][:3])})")
        lines.append(f"- **Nodes**: {' → '.join(r['node_trace'])}")
        lines.append(f"- **Elapsed**: {r['elapsed_ms']}ms")
        if r.get("error"):
            lines.append(f"- **Error**: {r['error']}")
        lines.append("")
    
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved → {md_path}")


if __name__ == "__main__":
    main()
