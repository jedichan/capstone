# tools 模块说明

## 目录职责
- 封装数据库与检索工具能力，供子 agent 通过工具调用。

## 文件职责
- `sql_tools.py`：SQL schema 探测、枚举值探测、动态 SQL 执行。
- `hybrid_tools.py`：时间锚点与混合检索工具。
- `db_access.py`：LanceDB/SQLite gateway。
- `llm.py`、`summarize.py`、`verify.py`、`playback.py`、`py2sql.py`：辅助能力。

## 对外接口（典型）
- `inspect_database_schema(...)`
- `inspect_column_enum_values(...)`
- `execute_dynamic_sql(...)`
- `get_temporal_anchor(...)`
- `dynamic_weighted_vector_search(...)`

## 依赖关系
- 被 `agents/*/sub_agent.py` 调用。
- 依赖 `node.types` 中数据库路径配置函数。

## 约定
- 工具函数返回结构需稳定，避免影响 ReAct 工具解析。
- 工具层不维护图状态，状态写回由节点层负责。
