import json
import re
from typing import Any

from lightingRL.prompt_registry import (
    SELF_QUERY_SYSTEM_PROMPT_KEY,
    SELF_QUERY_USER_PROMPT_KEY,
    get_prompt_template,
    render_prompt,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from .types import AgentState, InputValidator, StateResetter

SELF_QUERY_OUTPUT_SCHEMA = {
    "title": "self_query_rewrite",
    "type": "object",
    "properties": {
        "rewritten_query": {"type": "string"},
        "user_need": {"type": "string"},
        "intent_label": {"type": "string", "enum": ["structured", "semantic", "mixed"]},
        "retrieval_focus": {"type": "string", "enum": ["structured", "semantic", "mixed"]},
        "key_constraints": {"type": "array", "items": {"type": "string"}},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
        "reasoning_summary": {"type": "string"},
        "confidence": {"type": "number"},
        "expansion_terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete, observable alternative phrasing / keywords that help retrieval surface relevant chunks even when the original query uses abstract language",
        },
        "scene_constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attr_name": {"type": "string"},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["attr_name", "weight"],
            },
            "description": "Scene attributes relevant to this query, selected from the provided vocabulary. Only include attrs that are clearly implied by the query. Weight=0.9 for direct mention, 0.7 for strong inference, 0.5 for weak inference. Leave empty if no attrs are relevant.",
        },
    },
    "required": [
        "rewritten_query",
        "user_need",
        "intent_label",
        "retrieval_focus",
        "key_constraints",
        "ambiguities",
        "reasoning_summary",
        "confidence",
    ],
}


def _fallback_self_query(raw_query: str) -> dict[str, Any]:
    return {
        "rewritten_query": raw_query,
        "user_need": "Find relevant retrieval results from the user's request.",
        "intent_label": "mixed",
        "retrieval_focus": "mixed",
        "key_constraints": [],
        "ambiguities": [],
        "reasoning_summary": "Fallback to the original query because self-query preprocessing was unavailable.",
        "confidence": 0.35,
        "expansion_terms": [],
        "scene_constraints": [],
    }


def _normalize_query_text(raw_query: str) -> str:
    text = InputValidator.sanitize_query(raw_query)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return text.strip()


def _fast_path_self_query(raw_query: str) -> dict[str, Any] | None:
    normalized = _normalize_query_text(raw_query)
    if not normalized:
        return _fallback_self_query(raw_query)

    low = normalized.lower()
    tokens = re.findall(r"[a-z0-9_]+", low)
    structured_markers = [
        "did you see",
        "are there",
        "show me",
        "how many",
        "list",
        "person",
        "car",
        "dark",
        "parking",
        "database",
    ]
    semantic_markers = [
        "near",
        "around",
        "similar",
        "moving",
        "left bleachers",
        "sidewalk",
    ]
    if len(tokens) <= 12 or any(marker in low for marker in structured_markers + semantic_markers):
        if any(marker in low for marker in semantic_markers):
            intent = "semantic"
        elif any(marker in low for marker in structured_markers) or len(tokens) <= 2:
            intent = "structured"
        else:
            intent = "mixed"
        return {
            "rewritten_query": normalized,
            "user_need": f"Retrieve video results that satisfy: {normalized}",
            "intent_label": intent,
            "retrieval_focus": intent,
            "key_constraints": [token for token in tokens[:6]],
            "ambiguities": [],
            "reasoning_summary": "Fast-path preprocessing kept the original meaning and only normalized surface noise.",
            "confidence": 0.8,
            "expansion_terms": [],
            "scene_constraints": [],
        }
    return None


def _load_scene_vocab_list() -> str:
    """Build a human-readable list of scene attributes + aliases from the vocab JSON."""
    import os
    from pathlib import Path
    vocab_path = os.environ.get("AGENT_SCENE_ATTRS_VOCAB_PATH", "").strip()
    if not vocab_path:
        return "(no scene vocabulary available)"
    try:
        from tools.scene_attrs import load_vocab
        vocab = load_vocab(Path(vocab_path))
    except Exception:
        return "(scene vocabulary unavailable)"
    lines = []
    for attr_name, info in sorted(vocab.items()):
        aliases = ", ".join(info.get("query_aliases", [attr_name]))
        lines.append(f"  {attr_name}: {aliases}")
    return "Available scene attributes (with query synonyms):\n" + "\n".join(lines)


def create_self_query_node(llm: Any = None):
    def self_query_node(state: AgentState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
        del store
        raw_query = InputValidator.extract_latest_query(state)
        reset_updates = {}
        if StateResetter.is_new_query(state):
            reset_updates = StateResetter.reset_ephemeral_state(state, raw_query)

        fast_path = _fast_path_self_query(raw_query)
        if fast_path is not None:
            result = fast_path
        elif not raw_query:
            result = _fallback_self_query(raw_query)
        else:
            prompt = render_prompt(SELF_QUERY_USER_PROMPT_KEY, raw_query=raw_query)
            try:
                system_prompt = get_prompt_template(SELF_QUERY_SYSTEM_PROMPT_KEY).format(
                    scene_vocab_list=_load_scene_vocab_list(),
                )
            except Exception:
                # Fallback: remove the placeholder if format fails
                system_prompt = get_prompt_template(SELF_QUERY_SYSTEM_PROMPT_KEY).replace("{scene_vocab_list}", "(no scene vocabulary available)")
            try:
                if hasattr(llm, "with_structured_output"):
                    model = llm.with_structured_output(SELF_QUERY_OUTPUT_SCHEMA)
                    response = model.invoke(
                        [SystemMessage(content=system_prompt), HumanMessage(content=prompt)],
                        config=config,
                    )
                    result = response.model_dump() if hasattr(response, "model_dump") else dict(response)
                else:
                    raw = llm.invoke(
                        [SystemMessage(content=system_prompt), HumanMessage(content=prompt)],
                        config=config,
                    )
                    payload = raw.content if hasattr(raw, "content") else str(raw)
                    payload = payload.replace("```json", "").replace("```", "").strip()
                    result = json.loads(payload)
            except Exception:
                result = _fallback_self_query(raw_query)

        rewritten_query = _normalize_query_text(result.get("rewritten_query", raw_query)) or _normalize_query_text(raw_query)
        expansion_terms = result.get("expansion_terms") or []
        scene_constraints = result.get("scene_constraints") or []
        # Append expansion terms to the rewritten query so retrieval can use them
        expanded_query = rewritten_query
        if expansion_terms:
            expansion_suffix = " | " + " | ".join(str(t).strip() for t in expansion_terms if str(t).strip())
            expanded_query = rewritten_query + expansion_suffix
        print(
            "[SELF_QUERY_DEBUG] "
            + json.dumps(
                {
                    "raw_query": raw_query,
                    "rewritten_query": rewritten_query,
                    "expansion_terms": expansion_terms,
                    "scene_constraints": scene_constraints,
                    "intent_label": result.get("intent_label", "mixed"),
                    "retrieval_focus": result.get("retrieval_focus", "mixed"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return {
            **reset_updates,
            "original_user_query": raw_query,
            "user_query": raw_query,
            "rewritten_query": expanded_query,
            "self_query_result": {
                **result,
                "rewritten_query": expanded_query,
                "base_rewritten_query": rewritten_query,
                "expansion_terms": expansion_terms,
                "scene_constraints": scene_constraints,
                "original_user_query": raw_query,
            },
            "current_node": "self_query_node",
            "thought": f"SelfQuery: focus={result.get('retrieval_focus', 'mixed')}, intent={result.get('intent_label', 'mixed')}, scene_attrs={len(scene_constraints)}",
            "messages": [AIMessage(content=f"Self-query rewrite complete: {rewritten_query}")],
        }

    return self_query_node
