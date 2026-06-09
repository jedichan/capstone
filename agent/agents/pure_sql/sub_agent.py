import json
import logging

from langchain_core.messages import ToolMessage

from agents.shared import run_react_sub_agent
from tools.llamaindex_adapter import run_llamaindex_sql_query, use_llamaindex_sql
from tools.sql_tools import execute_dynamic_sql, inspect_column_enum_values, inspect_database_schema

_logger = logging.getLogger(__name__)

_JSON_START = "[SQL_JSON_START]"
_JSON_END = "[SQL_JSON_END]"

PURE_SQL_AGENT_PROMPT = """
You are an advanced SQL database query assistant responsible for answering user questions directly from the SQLite database.
Your task is to convert natural language queries into correct SQL queries and return the results.

【CRITICAL INSTRUCTIONS】
1. The current model/system DOES NOT support time-dimension queries (e.g., "last week", "10 seconds", "long time", "morning"). Ignore any time-related constraints in the user's prompt.
2. Your query results will be processed by downstream systems. Therefore, when using `execute_dynamic_sql`, you MUST include at least `video_id` and `event_summary_en` in the SELECT clause (or use `*`), so the downstream system can read the event description.
3. Even for simple queries (like COUNT), do not just query a single field. You must bring complete event information.
4. `object_type` is stored in English (e.g., car, person). Colors and scene zones are also in English.

【AVAILABLE TOOLS】
1. `inspect_database_schema`: Get the column names and types of a table (default is episodic_events). Call this first when you don't know the schema.
2. `inspect_column_enum_values`: Check distinct values in a column. Use this to resolve synonym/fuzzy matching problems — e.g., check what colors, object types, or zones actually exist in the database before building WHERE clauses.
3. `execute_dynamic_sql`: Execute dynamically generated SQL SELECT queries and return results. Only SELECT is allowed.

【WORKFLOW】
1. Analyze the user's intent — identify concrete filters (type, color, zone, video_id) vs. descriptive text.
2. Call `inspect_database_schema` to confirm column names and types.
3. For filters involving specific colors, object types, or zones that might not match exactly, call `inspect_column_enum_values` to calibrate values.
4. Build the SQL query: use exact WHERE clauses (lower(col) = 'value') for enum fields, and ensure the SELECT clause contains `video_id` and `event_summary_en`.
5. If execution fails with an error (e.g., 'no such column'), read the error, re-check the schema, and try again with corrected column names.
6. In your final response, summarize the results in natural English. Do NOT output raw JSON — just a plain-text summary of what was found.
"""


def _extract_sql_result(response: dict) -> tuple[str, list[dict]]:
    """Extract the natural-language summary and structured rows from a ReAct
    agent response.

    Relies on ``[SQL_JSON_START]`` / ``[SQL_JSON_END]`` markers emitted by
    ``execute_dynamic_sql`` for reliable JSON extraction.  Falls back to the
    legacy heuristic when markers are absent.
    """
    summary = response["messages"][-1].content
    raw_rows: list[dict] = []
    for msg in reversed(response["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "execute_dynamic_sql":
            content = msg.content
            # Primary path: marker-delimited JSON
            if _JSON_START in content and _JSON_END in content:
                try:
                    json_str = content.split(_JSON_START, 1)[1].split(_JSON_END, 1)[0]
                    raw_rows = json.loads(json_str)
                except Exception as exc:
                    _logger.warning("Failed to parse marker-delimited JSON: %s", exc)
            # Legacy fallback: heuristic extraction
            elif any(keyword in content for keyword in ("Execution successful", "Execution Successful", "执行成功", "{")):
                try:
                    if ":\n" in content:
                        json_str = content.split(":\n", 1)[1]
                    else:
                        json_str = content[content.find("[") :]
                    raw_rows = json.loads(json_str)
                except Exception:
                    pass
            break
    return summary, raw_rows


def run_pure_sql_sub_agent(user_query: str, llm):
    if use_llamaindex_sql():
        return run_llamaindex_sql_query(user_query)
    return run_react_sub_agent(
        user_query=user_query,
        llm=llm,
        tools=[inspect_database_schema, inspect_column_enum_values, execute_dynamic_sql],
        system_prompt=PURE_SQL_AGENT_PROMPT,
        result_extractor=_extract_sql_result,
        recursion_limit=10,
    )
