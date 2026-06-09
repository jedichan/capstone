from typing import Any


SELF_QUERY_SYSTEM_PROMPT_KEY = "rewrite.self_query.system"
SELF_QUERY_USER_PROMPT_KEY = "rewrite.self_query.user"


_PROMPTS: dict[str, str] = {
    SELF_QUERY_SYSTEM_PROMPT_KEY: (
        "You rewrite user queries for a video surveillance retrieval system. "
        "Preserve the original intent and all factual constraints, but make the "
        "wording more retrieval-friendly.\n\n"
        "EXPANSION TERMS (\"expansion_terms\"):\n"
        "If the query uses abstract, transformed, or uncommon language, generate "
        "3-5 concrete, observable phrasings that would appear directly in video "
        "annotation text.  Leave empty when the query is already concrete.\n"
        "Good to expand: 'logistics operation' → 'carrying boxes', "
        "'rectangular container' → 'box' / 'crate', "
        "'elderly individual' → 'old man' / 'old woman'.\n"
        "Do NOT expand: concrete nouns (car, truck, door, counter), "
        "simple actions (walking, standing, running), or explicit descriptions.\n\n"
        "SCENE ATTRIBUTE VOCABULARY:\n"
        "{scene_vocab_list}\n\n"
        "SCENE CONSTRAINTS (\"scene_constraints\"):\n"
        "Select ONLY attributes directly implied by the query, not every possible "
        "attribute.  Use weight=0.9 when the attribute is explicitly mentioned "
        "('blue shirt' → has_blue=0.9), weight=0.7 when strongly implied but not "
        "stated ('gas station' → has_car=0.7, has_road=0.7), weight=0.5 for weak "
        "inference.  Leave the array empty if NO attributes are clearly implied.\n\n"
        "OUTPUT: a JSON object with keys: rewritten_query, expansion_terms, "
        "retrieval_focus, intent_label, scene_constraints."
    ),
    SELF_QUERY_USER_PROMPT_KEY: (
        "Original query:\n{raw_query}\n\n"
        "Rewrite this query for retrieval.  Preserve the user's intent and keep "
        "all factual details (object type, color, location, action).  Return JSON."
    ),
}


def get_prompt_template(prompt_key: str) -> str:
    return _PROMPTS.get(prompt_key, "")


def render_prompt(prompt_key: str, **kwargs: Any) -> str:
    template = get_prompt_template(prompt_key)
    try:
        return template.format(**kwargs)
    except Exception:
        return template
