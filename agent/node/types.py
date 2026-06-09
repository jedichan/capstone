import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from db.config import (
    get_graph_chroma_child_collection,
    get_graph_chroma_collection,
    get_graph_chroma_event_collection,
    get_graph_chroma_namespace,
    get_graph_chroma_parent_collection,
    get_graph_chroma_path,
    get_graph_chroma_retrieval_level,
    get_graph_lancedb_path,
    get_graph_sqlite_db_path,
)


class ParsedQuestion(TypedDict):
    event: Optional[str]
    color: Optional[str]
    time: Optional[str]
    move: Optional[bool]


class ToolChoice(TypedDict, total=False):
    mode: str
    sql_needed: bool
    hybrid_needed: bool
    sub_queries: Dict[str, Any]


class SearchResult(TypedDict):
    source: str
    rows: List[Dict[str, Any]]
    error: Optional[str]


class ReflectionResult(TypedDict, total=False):
    feedback: Optional[str]
    quality_score: float
    needs_retry: bool
    optimized: bool
    can_continue: bool
    errors: list[Any]


class QueryQuadruple(TypedDict, total=False):
    object: List[str]
    color: List[str]
    location: List[str]
    event: str
    confidence: float
    source: str


class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    original_user_query: str
    user_query: str
    rewritten_query: str
    optimized_query: str
    parsed_question: ParsedQuestion
    tool_choice: ToolChoice
    event_list: List[str]
    meta_list: List[Dict[str, Any]]
    sql_result: List[Dict[str, Any]]
    hybrid_result: List[Dict[str, Any]]
    rerank_result: List[Dict[str, Any]]
    tool_error: Optional[str]
    route: str
    retry_count: int
    thought: str
    final_answer: str
    reflection_result: ReflectionResult
    syntax_valid: bool
    syntax_error: Optional[str]
    current_node: str
    cot_context: List[Dict[str, Any]]  # CoT 中间状态
    force_reset: bool                  # 强制重置状态标志
    query_quadruple: QueryQuadruple
    routing_metrics: Dict[str, Any]
    search_config: Dict[str, Any]
    sql_plan: Dict[str, Any]
    sql_debug: Dict[str, Any]
    metrics: Dict[str, Any]
    classification_result: Dict[str, Any]
    answer_type: str
    verifier_result: Dict[str, Any]
    self_query_result: Dict[str, Any]
    search_explain: str
    raw_final_answer: str
    summary_result: Dict[str, Any]
    camera_filter: Optional[List[str]]  # camera_ids injected by eval harness


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return str(content)


def normalize_text_value(value: Any) -> str:
    return str(value).strip() if value is not None else ""


class InputValidator:
    """Unified input validator and extractor"""
    
    @staticmethod
    def extract_latest_query(state: AgentState) -> str:
        """
        Extract the latest raw user query from the state.
        Rules:
        1. Always prioritize extracting from the last HumanMessage in the messages list.
        2. If there are no messages, fall back to original_user_query, then user_query.
        3. Perform basic length and validity truncation.
        """
        query = ""
        messages = state.get("messages", [])
        
        if messages:
            # Prioritize reverse search for the last human message
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = content_to_text(msg.content)
                    break
                # Compatible with dictionary format messages
                elif isinstance(msg, dict) and msg.get("type") == "human" and "content" in msg:
                    query = content_to_text(msg["content"])
                    break
                elif isinstance(msg, dict) and "content" in msg and not msg.get("type"):
                    # Minimal fallback
                    query = content_to_text(msg["content"])
                    break

        # If extraction from messages fails, fallback to user_query field
        if not query:
            query = state.get("original_user_query", "") or state.get("user_query", "")
            
        return InputValidator.sanitize_query(query)

    @staticmethod
    def resolve_active_query(state: AgentState) -> str:
        """
        Resolve the best executable query for downstream retrieval/routing nodes.
        Priority:
        1. optimized_query from reflection retry
        2. rewritten_query from self-query preprocessing
        3. user_query persisted in state
        4. latest raw user query
        """
        retry_count = int(state.get("retry_count", 0) or 0)
        reflection_result = state.get("reflection_result", {})
        if retry_count > 0 and isinstance(reflection_result, dict) and reflection_result.get("needs_retry", False):
            optimized_query = InputValidator.sanitize_query(state.get("optimized_query", ""))
            if optimized_query:
                return optimized_query

        rewritten_query = InputValidator.sanitize_query(state.get("rewritten_query", ""))
        if rewritten_query:
            return rewritten_query

        persisted_query = InputValidator.sanitize_query(state.get("user_query", ""))
        if persisted_query:
            return persisted_query

        return InputValidator.extract_latest_query(state)

    @staticmethod
    def sanitize_query(query: Any) -> str:
        """
        Security protection: prevent injection, limit length, clean illegal characters
        """
        if not query:
            return ""
            
        # Unify to string
        clean_query = str(query).strip()
        
        # 1. Length limit (prevent ultra-long text attacks)
        max_length = 500
        if len(clean_query) > max_length:
            clean_query = clean_query[:max_length]
            
        # 2. Basic security filtering
        return clean_query.strip()


class StateResetter:
    """State resetter: used to clean up old temporary state before processing a new query"""
    
    # Fields that are temporary results generated by the current query and need to be reset on the next query
    EPHEMERAL_FIELDS = {
        "rewritten_query": "",
        "optimized_query": "",
        "parsed_question": {},
        "tool_choice": {},
        "event_list": [],
        "meta_list": [],
        "sql_result": [],
        "hybrid_result": [],
        "rerank_result": [],
        "tool_error": None,
        "route": "",
        "retry_count": 0,
        "thought": "",
        "final_answer": "",
        "reflection_result": {},
        "syntax_valid": True,
        "syntax_error": None,
        "current_node": "",
        "cot_context": [],
        "query_quadruple": {},
        "routing_metrics": {},
        "search_config": {},
        "sql_plan": {},
        "sql_debug": {},
        "metrics": {},
        "classification_result": {},
        "answer_type": "",
        "verifier_result": {},
        "self_query_result": {},
        "search_explain": "",
        "raw_final_answer": "",
        "summary_result": {},
        # camera_filter is intentionally NOT here — it is an input parameter
        # set once by the eval harness before graph execution and must survive
        # the StateResetter.reset_ephemeral_state() call in self_query_node.
    }

    @staticmethod
    def is_new_query(state: AgentState) -> bool:
        """
        Detect if it is a completely new user query:
        When the latest message in the messages list is a HumanMessage, and it is different from the user_query cached in the current state, it indicates a new round of dialogue has started.
        Or if an explicit reset flag is set in the state.
        """
        if state.get("force_reset", False):
            return True
            
        latest_msg_text = ""
        messages = state.get("messages", [])
        if messages:
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    latest_msg_text = content_to_text(msg.content)
                    break
                elif isinstance(msg, dict) and msg.get("type") == "human":
                    latest_msg_text = content_to_text(msg.get("content", ""))
                    break
        
        latest_msg_text = InputValidator.sanitize_query(latest_msg_text)
        current_query = state.get("user_query", "")
        
        # If the latest human message is not empty and is different from the currently stored user_query, it's a new query
        if latest_msg_text and latest_msg_text != current_query:
            return True
            
        return False

    @staticmethod
    def reset_ephemeral_state(state: AgentState, user_query: str) -> dict[str, Any]:
        """
        Generate a reset dictionary. This dictionary should be merged into the current AgentState.
        Restore all temporary results to default values, and update user_query at the same time.
        """
        reset_updates = {k: v for k, v in StateResetter.EPHEMERAL_FIELDS.items()}
        reset_updates["original_user_query"] = user_query
        reset_updates["user_query"] = user_query
        reset_updates["force_reset"] = False # Clear force reset flag
        
        return reset_updates


def question_to_meta_and_event(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    meta_list: list[dict[str, Any]] = []
    event_list: list[str] = []
    event = normalize_text_value(payload.get("event"))
    if event and event.lower() != "null":
        event_list.append(event)
    color = normalize_text_value(payload.get("color"))
    if color and color.lower() != "null":
        meta_list.append({"field": "object_color_en", "op": "contains", "value": color})
    object_value = payload.get("object")
    object_candidates: list[str] = []
    if isinstance(object_value, list):
        object_candidates = [normalize_text_value(item) for item in object_value]
    else:
        object_candidates = [normalize_text_value(object_value)]
    object_alias = {
        "car": "car",
        "truck": "truck",
        "person": "person",
        "pedestrian": "person",
        "bike": "bike",
        "bicycle": "bike",
        "motorcycle": "motorcycle",
    }
    object_tokens: list[str] = []
    for item in object_candidates:
        if not item or item.lower() == "null":
            continue
        normalized = object_alias.get(item.lower(), object_alias.get(item, item))
        if normalized and normalized not in object_tokens:
            object_tokens.append(normalized)
    for token in object_tokens:
        meta_list.append({"field": "object_type", "op": "contains", "value": token})
    location = normalize_text_value(payload.get("location"))
    if location and location.lower() != "null":
        meta_list.append({"field": "scene_zone_en", "op": "contains", "value": location})
        
    if payload.get("move") is False:
        meta_list.append({"field": "appearance_notes_en", "op": "contains", "value": "stationary"})
    return meta_list, event_list


def default_db_path() -> Path:
    return get_graph_lancedb_path()


def default_sqlite_db_path() -> Path:
    return get_graph_sqlite_db_path()


def default_chroma_path() -> Path:
    return get_graph_chroma_path()


def default_chroma_collection() -> str:
    return get_graph_chroma_collection()


def default_chroma_parent_collection() -> str:
    return get_graph_chroma_parent_collection()


def default_chroma_event_collection() -> str:
    return get_graph_chroma_event_collection()


def default_chroma_retrieval_level() -> str:
    return get_graph_chroma_retrieval_level()


def default_chroma_namespace() -> str:
    return get_graph_chroma_namespace()


def existence_grounder_enabled() -> bool:
    """Shared flag: whether the existence-grounder (match_verifier_node)
    may rewrite Yes/No responses for existence queries.

    Mirrors the former per-file ``_existence_grounder_enabled`` helpers;
    keeping a single source of truth avoids drift between answer_node
    and summary_node."""
    raw = os.getenv("AGENT_ENABLE_EXISTENCE_GROUNDER", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    payload = {"event": "enter", "color": "red", "time": "today", "move": None}
    meta_list, event_list = question_to_meta_and_event(payload)
    print("event_list:", event_list)
    print("meta_list:", meta_list)
