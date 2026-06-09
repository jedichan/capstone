import json
import sqlite3
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from node.types import default_sqlite_db_path
from tools.sql_debug_utils import extract_where_clause, find_unknown_sql_columns, get_sqlite_table_columns, log_sql_debug

_JSON_START_MARKER = "[SQL_JSON_START]"
_JSON_END_MARKER = "[SQL_JSON_END]"

@tool
def inspect_database_schema(table_name: str = "episodic_events") -> str:
    """
    Inspect the schema of a specific table in the SQLite database.
    Returns the column names and their data types.
    """
    db_path = default_sqlite_db_path()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        conn.close()
        
        if not columns:
            return f"Table '{table_name}' not found."
            
        schema_info = [f"- {col[1]} ({col[2]})" for col in columns]
        return f"Schema for '{table_name}':\n" + "\n".join(schema_info)
    except Exception as e:
        return f"Failed to inspect schema: {str(e)}"

@tool
def inspect_column_enum_values(column_name: str, table_name: str = "episodic_events") -> str:
    """
    Inspect the distinct values present in a specific column.
    Useful for checking exact terms (e.g., what colors or zones actually exist in the database).
    """
    db_path = default_sqlite_db_path()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Get all distinct non-empty values for the column
        cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL AND {column_name} != ''")
        values = [str(row[0]) for row in cursor.fetchall()]
        conn.close()
        
        if not values:
            return f"Column '{column_name}' has no data or does not exist."
            
        return f"Distinct values for '{column_name}':\n" + ", ".join(values)
    except Exception as e:
        return f"Failed to inspect column values: {str(e)}"

@tool
def execute_dynamic_sql(sql_query: str) -> str:
    """
    Execute a dynamically generated SQL SELECT query and return the results.
    WARNING: Always include video_id and event_summary_en in the SELECT clause. Time queries are NOT supported.
    """
    db_path = default_sqlite_db_path()
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Optimize for read-heavy concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA cache_size=-8000")
        conn.execute("PRAGMA query_only=ON")
        cursor = conn.cursor()
        schema_columns = get_sqlite_table_columns(db_path)

        # Simple defense: only allow SELECT queries
        if not sql_query.strip().upper().startswith("SELECT"):
            conn.close()
            return "Error: Only SELECT queries are allowed."

        # P2-5: Auto-append LIMIT if missing to prevent unbounded scans
        normalized = sql_query.rstrip(" ;\t\n")
        if "limit" not in normalized.lower():
            normalized = normalized + " LIMIT 200"

        where_clause = extract_where_clause(normalized)
        unknown_columns = find_unknown_sql_columns(normalized, schema_columns)
        log_sql_debug(
            "execute_dynamic_sql",
            db_path=str(db_path),
            final_sql=normalized,
            where_clause=where_clause,
            unknown_columns=unknown_columns,
        )

        # Set a query timeout via progress handler (sqlite3 lacks native timeout)
        _start = sqlite3.time.time() if hasattr(sqlite3, 'time') else __import__('time').time()
        _timeout = 30.0

        def _progress_handler():
            return 0 if (__import__('time').time() - _start) < _timeout else 1

        conn.set_progress_handler(_progress_handler, 1000)
        cursor.execute(normalized)
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return "Execution successful, but no results returned (0 records)."

        columns = rows[0].keys()
        result_list = [dict(zip(columns, row)) for row in rows]

        # Truncate at SQL level to avoid fetching massive result sets
        if len(result_list) > 50:
            result_list = result_list[:50]
            truncation_msg = " (Truncated server-side to 50 records)"
        else:
            truncation_msg = ""

        conn.close()

        result_json = json.dumps(result_list, ensure_ascii=False, indent=2)
        # Wrap JSON in markers so _extract_sql_result can parse it reliably
        return (
            f"Execution successful, returned {len(result_list)} records{truncation_msg}:\n"
            f"{_JSON_START_MARKER}\n{result_json}\n{_JSON_END_MARKER}"
        )
    except Exception as e:
        return f"SQL Execution Failed: {str(e)}"
