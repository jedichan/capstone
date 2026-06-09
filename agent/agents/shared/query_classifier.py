import json
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

# NOTE: P1-6 extends the label enum with ``multi_hop`` and attaches structured
# ``signals`` to every classification payload so downstream fusion/verifier
# nodes can reason about evidence rather than phrasing. P1-7 adds
# ``answer_type`` so the grounder (match_verifier_node) can be enabled
# selectively for existence-style questions.

QUERY_CLASSIFICATION_OUTPUT_SCHEMA = {
    "title": "query_classification",
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["structured", "semantic", "mixed", "multi_hop"],
        },
        "answer_type": {
            "type": "string",
            "enum": ["existence", "list", "description", "count", "unknown"],
        },
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "multi_camera": {
            "type": "boolean",
            "description": "Whether the user is asking about cross-camera / multi-camera behavior (e.g. 'appear in camera A and then camera B', 'which cameras did this person appear in')",
        },
    },
    "required": ["label", "answer_type", "confidence", "reason", "multi_camera"],
}


LABEL_STRUCTURED = "structured"
LABEL_SEMANTIC = "semantic"
LABEL_MIXED = "mixed"
LABEL_MULTI_HOP = "multi_hop"
ALL_LABELS = {LABEL_STRUCTURED, LABEL_SEMANTIC, LABEL_MIXED, LABEL_MULTI_HOP}

ANSWER_TYPE_EXISTENCE = "existence"
ANSWER_TYPE_LIST = "list"
ANSWER_TYPE_DESCRIPTION = "description"
ANSWER_TYPE_COUNT = "count"
ANSWER_TYPE_UNKNOWN = "unknown"
ALL_ANSWER_TYPES = {
    ANSWER_TYPE_EXISTENCE,
    ANSWER_TYPE_LIST,
    ANSWER_TYPE_DESCRIPTION,
    ANSWER_TYPE_COUNT,
    ANSWER_TYPE_UNKNOWN,
}


# Lightweight, domain-agnostic cue dictionaries. Kept intentionally small so
# they function as *signals* (presence/absence) rather than a BOW classifier.
_METADATA_ENUM_HITS = (
    # objects
    "person", "people", "pedestrian", "car", "truck", "vehicle", "bike",
    "bicycle", "motorcycle", "dog", "animal",
    # colors
    "black", "white", "red", "blue", "dark", "light", "gray", "grey",
    # scene zones
    "parking", "sidewalk", "baseline", "court", "bleachers", "road",
    "hallway", "door", "entrance",
)

_RELATION_CUES = (
    "near", "around", "onto", "against", "beside", "behind", "in front of",
    "similar", "like", "toward", "approach", "moving", "walking",
    "running", "turning", "pulling", "throwing", "hitting", "scanning",
    "entering", "exiting", "leaving",
)

_MULTI_STEP_CUES = (
    " then ", " after ", " before ", " while ", " and then ",
    " followed by ", " subsequently ", ", then ", ", and then ",
    " afterwards ", " meanwhile ",
)

_EXISTENCE_CUES = (
    "is there", "are there", "did you see", "do you see", "do you have",
    "have you seen", "any clip", "any video", "any footage", "any scene",
    "any shot", "有没有", "是否存在", "是否有",
    # P2-3: extended existence patterns — many "no" cases start with these
    "does the video record", "does the surveillance", "does the camera",
    "can you find", "can you see", "can you locate",
    "is a scene", "is there a scene", "is there a clip",
    "is there any", "are there any", "was there",
    "search for a segment", "search for the footage",
    "find a segment", "find the footage", "find if",
)

_LIST_CUES = (
    "list", "show me", "find all", "give me all", "enumerate",
)

_COUNT_CUES = (
    "how many", "count", "number of",
)

_MULTI_CAMERA_CUES = (
    # 中文 — 多摄像头开放式查询
    "跨摄像头", "跨镜头", "多个摄像头", "不同摄像头",
    "各个摄像头", "每个摄像头", "哪个摄像头", "哪些摄像头",
    "出现在几个", "经过哪些", "去过哪些",
    "从哪个摄像头", "在哪些镜头", "各个镜头",
    # 英文 — 多摄像头开放式查询
    "across cameras", "across camera", "all cameras", "every camera",
    "different cameras", "each camera", "which cameras", "what cameras",
    "multiple cameras", "cross-camera", "cross camera",
    # 英文 — cross_camera 验证问题 (appear in A and then B)
    "appear in", "and then appear", "appear again",
    "also appear", "from camera", "then in camera",
    "first in", "then in",
)


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in ALL_LABELS else LABEL_MIXED


def _normalize_answer_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in ALL_ANSWER_TYPES else ANSWER_TYPE_UNKNOWN


def _normalize_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        conf = 0.5
    return max(0.0, min(1.0, conf))


def _contains_any(haystack: str, needles: Tuple[str, ...]) -> List[str]:
    return [needle.strip() for needle in needles if needle in haystack]


def _collect_signals(text: str) -> Dict[str, List[str]]:
    low = (text or "").strip().lower()
    padded = f" {low} "
    metadata_hits = sorted(
        {
            token
            for token in _METADATA_ENUM_HITS
            if re.search(rf"\b{re.escape(token)}\b", low)
        }
    )
    relation_cues = sorted({cue.strip() for cue in _RELATION_CUES if cue in padded})
    multi_step_cues = sorted({cue.strip() for cue in _MULTI_STEP_CUES if cue in padded})
    existence_cues = sorted({cue for cue in _EXISTENCE_CUES if cue in low})
    multi_camera_cues = sorted(
        {cue.strip() for cue in _MULTI_CAMERA_CUES if cue in low}
    )
    return {
        "metadata_hits": metadata_hits,
        "relation_cues": relation_cues,
        "multi_step_cues": multi_step_cues,
        "existence_cues": existence_cues,
        "multi_camera_cues": multi_camera_cues,
    }


def _infer_answer_type(text: str) -> str:
    low = (text or "").strip().lower()
    if any(cue in low for cue in _EXISTENCE_CUES) or low.startswith((
        "is ", "are ", "was ", "were ", "does ", "do ", "did ",
        "can you ", "could you ",  # P2-3: "Can you find..." is existence
    )):
        return ANSWER_TYPE_EXISTENCE
    if any(cue in low for cue in _COUNT_CUES):
        return ANSWER_TYPE_COUNT
    if any(cue in low for cue in _LIST_CUES):
        return ANSWER_TYPE_LIST
    if low.endswith("?") or low.startswith(("what ", "describe ", "explain ")):
        return ANSWER_TYPE_DESCRIPTION
    return ANSWER_TYPE_UNKNOWN


def _label_from_signals(signals: Dict[str, List[str]]) -> Tuple[str, float, str]:
    metadata_n = len(signals.get("metadata_hits", []))
    relation_n = len(signals.get("relation_cues", []))
    multi_step_n = len(signals.get("multi_step_cues", []))

    if multi_step_n >= 1 and (relation_n >= 1 or metadata_n >= 2):
        return LABEL_MULTI_HOP, 0.8, f"multi_step_cues={multi_step_n}"
    if metadata_n >= 1 and relation_n == 0 and multi_step_n == 0:
        return LABEL_STRUCTURED, 0.78, f"metadata_hits={metadata_n}"
    if relation_n >= 1 and metadata_n == 0:
        return LABEL_SEMANTIC, 0.78, f"relation_cues={relation_n}"
    if metadata_n >= 1 and relation_n >= 1:
        return LABEL_MIXED, 0.7, f"metadata_hits={metadata_n}, relation_cues={relation_n}"
    return LABEL_MIXED, 0.45, "no strong signals"


def _detect_multi_camera(signals: Dict[str, List[str]]) -> bool:
    """快速判断是否有多摄像头意图。"""
    return len(signals.get("multi_camera_cues", [])) > 0


def _compat_signal_counts(signals: Dict[str, List[str]], label: str) -> Dict[str, int]:
    # Legacy consumers expected ``signals.structured`` / ``signals.semantic`` as
    # ints; keep them for backward-compat alongside the new list-valued cues.
    return {
        "structured": 1 if label in {LABEL_STRUCTURED, LABEL_MIXED} else 0,
        "semantic": 1 if label in {LABEL_SEMANTIC, LABEL_MIXED, LABEL_MULTI_HOP} else 0,
    }


def _fallback_result(reason: str, label: str = LABEL_MIXED, text: str = "") -> Dict[str, Any]:
    safe_label = _normalize_label(label)
    signals = _collect_signals(text)
    return {
        "label": safe_label,
        "answer_type": _infer_answer_type(text),
        "confidence": 0.35,
        "reason": reason,
        "signals": {**signals, **_compat_signal_counts(signals, safe_label)},
        "multi_camera": _detect_multi_camera(signals),
    }


def _fast_path_classification(text: str) -> Dict[str, Any] | None:
    low = (text or "").strip().lower()
    if not low:
        return None

    signals = _collect_signals(low)
    metadata_n = len(signals["metadata_hits"])
    relation_n = len(signals["relation_cues"])
    multi_step_n = len(signals["multi_step_cues"])

    # Only take the fast path when signals are *unambiguous*. The old fast-path
    # relied on fragile keyword co-occurrence (e.g. "did you see" + "court")
    # which misrouted existence questions. The new rule: if there's a clear
    # majority signal, return directly; otherwise defer to the LLM.
    if multi_step_n >= 1 and (relation_n >= 1 or metadata_n >= 2):
        label, confidence, reason = LABEL_MULTI_HOP, 0.82, f"fast-path multi-step ({multi_step_n} cues)"
    elif metadata_n >= 2 and relation_n == 0 and multi_step_n == 0:
        label, confidence, reason = LABEL_STRUCTURED, 0.82, f"fast-path metadata ({metadata_n} enum hits)"
    elif relation_n >= 2 and metadata_n == 0:
        label, confidence, reason = LABEL_SEMANTIC, 0.82, f"fast-path relation ({relation_n} cues)"
    else:
        return None

    return {
        "label": label,
        "answer_type": _infer_answer_type(low),
        "confidence": confidence,
        "reason": reason,
        "signals": {**signals, **_compat_signal_counts(signals, label)},
        "multi_camera": _detect_multi_camera(signals),
    }


def classify_query(query: str, llm: Any = None, config: Any = None) -> Dict[str, Any]:
    text = (query or "").strip()
    if not text:
        return _fallback_result("empty query fallback", text=text)
    fast_path = _fast_path_classification(text)
    if fast_path is not None:
        return fast_path

    if llm is None:
        try:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model_name": os.getenv(
                    "AGENT_CHAT_MODEL",
                    os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4"),
                ),
                "temperature": 0.0,
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
            base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
            if base_url:
                kwargs["base_url"] = base_url
            llm = ChatOpenAI(**kwargs)
        except Exception:
            return _fallback_result("llm init failed", text=text)

    signals = _collect_signals(text)
    prompt = (
        "You are a video-retrieval query classifier. Return JSON matching the schema.\n\n"
        "label ∈ {structured, semantic, mixed, multi_hop}:\n"
        "  - structured: can be answered by metadata/enum filters alone "
        "(object type, color, scene zone). Example: 'black car on the road'.\n"
        "  - semantic: requires visual/relational understanding or free-text description "
        "(near, around, similar, motion pattern, abstract behavior).\n"
        "  - mixed: contains BOTH concrete metadata tokens AND semantic concepts. "
        "Example: 'a person carrying a box near the door'.\n"
        "  - multi_hop: requires sequencing or composition across multiple events "
        "(A then B, while A happened B, after A found B).\n\n"
        "answer_type ∈ {existence, list, description, count, unknown}:\n"
        "  - existence: yes/no question ('is there', 'did you see', 'have you seen').\n"
        "  - list: request to show/list/enumerate matching clips.\n"
        "  - description: request to explain or summarise a clip.\n"
        "  - count: asks how many / number of.\n"
        "  - unknown: cannot determine the answer type.\n\n"
        "multi_camera ∈ {true, false}:\n"
        "  - true: the user is asking about cross-camera / multi-camera behavior. "
        "This includes questions like 'appear in camera A and then camera B', "
        "'which cameras did this person appear in', 'also appear in camera', etc.\n"
        "  - false: the question is about a single camera view.\n\n"
        "Classify by the INFORMATION TYPE needed, not sentence mood. "
        "An existence question like 'is there a black car on the road?' is "
        "structured because it only needs metadata filters, no semantic reasoning.\n\n"
        f"User query: {text}\n"
        f"Detected signals: {signals}\n"
    )
    try:
        if hasattr(llm, "with_structured_output"):
            model = llm.with_structured_output(QUERY_CLASSIFICATION_OUTPUT_SCHEMA)
            result = model.invoke(
                [SystemMessage(content="Return JSON only."), HumanMessage(content=prompt)],
                config=config,
            )
            payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        else:
            raw = llm.invoke(
                [SystemMessage(content="Return JSON only."), HumanMessage(content=prompt)],
                config=config,
            )
            text_out = raw.content if hasattr(raw, "content") else str(raw)
            text_out = text_out.replace("```json", "").replace("```", "").strip()
            payload = json.loads(text_out)
    except Exception:
        return _fallback_result("llm classify failed", text=text)

    label = _normalize_label(payload.get("label"))
    answer_type = _normalize_answer_type(payload.get("answer_type")) or _infer_answer_type(text)
    if answer_type == ANSWER_TYPE_UNKNOWN:
        answer_type = _infer_answer_type(text)
    confidence = _normalize_confidence(payload.get("confidence"))
    reason = str(payload.get("reason", "llm classifier")).strip() or "llm classifier"
    multi_camera = bool(payload.get("multi_camera")) or _detect_multi_camera(signals)
    return {
        "label": label,
        "answer_type": answer_type,
        "confidence": confidence,
        "reason": reason,
        "signals": {**signals, **_compat_signal_counts(signals, label)},
        "multi_camera": multi_camera,
    }


def classify_mode_from_label(label: str) -> str:
    # Retained for legacy-router compatibility.
    if label == LABEL_STRUCTURED:
        return "pure_sql"
    if label in {LABEL_SEMANTIC, LABEL_MULTI_HOP}:
        return "hybrid_search"
    return os.getenv("AGENT_MIXED_COMPAT_MODE", "hybrid_search")
