import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lancedb

try:
    from .llm import get_qwen_embedding
except Exception:
    from llm import get_qwen_embedding


class SQLVideoSearchTool:
    """LanceDB-based video retrieval tool.

    This class first applies metadata filtering, then performs vector
    retrieval for event queries.
    """

    def __init__(
        self,
        db_path: str | Path,
        table_name: str = "episodic_events",
        embedder: Callable[[str], list[float]] | None = None,
    ) -> None:
        """Initializes the LanceDB search tool.

        Args:
            db_path: LanceDB directory path.
            table_name: LanceDB table name.
            embedder: Embedding function for event text.
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.embedder = embedder or get_qwen_embedding

    def _escape_identifier(self, field: str) -> str:
        """Escapes and validates a field name for where clauses.

        Args:
            field: Field name in metadata filter.

        Returns:
            Safe field name.

        Raises:
            ValueError: If the field name is invalid.
        """
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field):
            raise ValueError(f"Invalid field name: {field}")
        return field

    def _escape_literal(self, value: Any) -> str:
        """Converts Python values to Lance SQL literals.

        Args:
            value: Input literal value.

        Returns:
            SQL literal string.
        """
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value).replace("'", "''")
        return f"'{text}'"

    def _metadata_item_to_expr(self, item: Any) -> str:
        """Converts one metadata item to a where expression.

        Args:
            item: Metadata condition item.

        Returns:
            One where expression segment.

        Raises:
            ValueError: If metadata item format is invalid.
        """
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            field = self._escape_identifier(str(item.get("field", "")))
            op = str(item.get("op", "==")).lower()
            value = item.get("value")
            if op in {"=", "=="}:
                return f"{field} = {self._escape_literal(value)}"
            if op in {"!=", "<>"}:
                return f"{field} != {self._escape_literal(value)}"
            if op in {">", ">=", "<", "<="}:
                return f"{field} {op} {self._escape_literal(value)}"
            if op == "contains":
                value_text = str(value).replace("'", "''")
                return f"{field} LIKE '%{value_text}%'"
            raise ValueError(f"Unsupported metadata operator: {op}")
        if isinstance(item, (list, tuple)) and len(item) == 3:
            field = self._escape_identifier(str(item[0]))
            op = str(item[1]).lower()
            value = item[2]
            return self._metadata_item_to_expr({"field": field, "op": op, "value": value})
        raise ValueError("Metadata item must be str, dict, or 3-length tuple/list")

    def _build_where_expr(self, metadata_filters: list[Any]) -> str | None:
        """Builds LanceDB where expression from metadata list.

        Args:
            metadata_filters: Metadata conditions list.

        Returns:
            Combined where expression or None.
        """
        expr_list = [self._metadata_item_to_expr(item) for item in metadata_filters if item is not None]
        if not expr_list:
            return None
        return " AND ".join(f"({expr})" for expr in expr_list)

    def _open_table(self):
        """Opens LanceDB table.

        Returns:
            Lance table object.
        """
        db = lancedb.connect(str(self.db_path))
        return db.open_table(self.table_name)

    def _match_metadata_item(self, row: dict[str, Any], item: Any) -> bool:
        """Checks whether one row matches one metadata condition item.

        Args:
            row: One row dictionary.
            item: Metadata condition item.

        Returns:
            True if row matches the condition.
        """
        if not isinstance(item, dict):
            return True
        field = str(item.get("field", ""))
        op = str(item.get("op", "==")).lower()
        value = item.get("value")
        row_value = row.get(field)
        if op in {"=", "=="}:
            return row_value == value
        if op in {"!=", "<>"}:
            return row_value != value
        if op == "contains":
            return str(value) in str(row_value)
        if op == ">":
            return row_value is not None and row_value > value
        if op == ">=":
            return row_value is not None and row_value >= value
        if op == "<":
            return row_value is not None and row_value < value
        if op == "<=":
            return row_value is not None and row_value <= value
        return True

    def _metadata_only_search(self, metadata_filters: list[Any], candidate_limit: int) -> list[dict[str, Any]]:
        """Executes metadata-only retrieval.

        Args:
            metadata_filters: Metadata conditions list.
            candidate_limit: Candidate retrieval upper limit.

        Returns:
            Metadata-filtered rows.
        """
        table = self._open_table()
        where_expr = self._build_where_expr(metadata_filters)
        frame = table.to_pandas()
        records = frame.to_dict(orient="records")
        if where_expr is None or not metadata_filters:
            return records[:candidate_limit]
        filtered: list[dict[str, Any]] = []
        for row in records:
            if all(self._match_metadata_item(row, item) for item in metadata_filters):
                filtered.append(row)
                if len(filtered) >= candidate_limit:
                    break
        return filtered

    def search(
        self,
        metadata_filters: list[Any],
        event_queries: list[str],
        candidate_limit: int = 80,
        top_k_per_event: int = 20,
    ) -> list[dict[str, Any]]:
        """Executes retrieval by metadata filter first, then vector search.

        Args:
            metadata_filters: Metadata conditions list.
            event_queries: Event query text list.
            candidate_limit: Candidate retrieval upper limit.
            top_k_per_event: Vector top-k for each event query.

        Returns:
            Retrieved and merged rows.
        """
        if not self.db_path.exists():
            return []
        table = self._open_table()
        where_expr = self._build_where_expr(metadata_filters)
        if not event_queries:
            return self._metadata_only_search(metadata_filters, candidate_limit)
        merged: dict[Any, dict[str, Any]] = {}
        for event_text in event_queries:
            query_vector = self.embedder(event_text)
            builder = table.search(query_vector)
            if where_expr:
                try:
                    builder = builder.where(where_expr)
                except Exception:
                    pass
            rows = builder.limit(top_k_per_event).to_list()
            for row in rows:
                event_id = row.get("event_id")
                distance = float(row.get("_distance", 1e9))
                if event_id not in merged or distance < float(merged[event_id].get("_distance", 1e9)):
                    merged[event_id] = row
        ranked = sorted(merged.values(), key=lambda x: float(x.get("_distance", 1e9)))
        return ranked[:candidate_limit]


if __name__ == "__main__":
    db_path = Path("/home/yangxp/Capstone/data/lancedb")
    tool = SQLVideoSearchTool(db_path=db_path)

    metadata_filters = [
        {"field": "object_color_cn", "op": "contains", "value": "红"},
        {"field": "start_time", "op": ">=", "value": 0},
    ]
    event_queries = ["红色目标进入画面"]

    results = tool.search(
        metadata_filters=metadata_filters,
        event_queries=event_queries,
        candidate_limit=5,
        top_k_per_event=10,
    )
    print("vector+metadata result count:", len(results))
    if results:
        print("top1:", results[0].get("event_id"), results[0].get("video_id"), results[0].get("_distance"))

    metadata_only_results = tool.search(
        metadata_filters=metadata_filters,
        event_queries=[],
        candidate_limit=3,
    )
    print("metadata-only result count:", len(metadata_only_results))
    if metadata_only_results:
        print("metadata top1:", metadata_only_results)
