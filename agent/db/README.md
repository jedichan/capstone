# db 模块说明

## 目录职责
- 管理结构化数据库构建、路径配置、切换与测试脚本。

## 文件职责
- `config.py`：SQLite/LanceDB 路径配置读取与持久化。
- `schema.py`：SQLite 表结构与索引真源。
- `sqlite_builder.py`：建表、索引、seed 导入、初始化提示词生成。
- `chroma_builder.py`：构建 Chroma 父子索引，child 保持 track-level，parent 按 video_id 聚合。
- `manage_graph_db.py`：CLI 入口（build/switch）。
- `chorma_test_runner.py`：Chroma 检索策略测试脚本。

## 对外接口（典型）
- `python -m agent.db.manage_graph_db build ...`
- `python -m agent.db.manage_graph_db build-chroma ...`
- `python -m agent.db.manage_graph_db switch ...`
- `SQLiteDatabaseBuilder.build(seed_files=...)`

## 当前 Chroma 索引策略
- `child collection`：保留当前 `track-level` 方案，record id 为 `{video_id}_{entity_hint}`。
- `parent collection`：`video-level` 聚合，record id 为 `{video_id}`。
- `event collection`：P0-2 新增 `event-level` 细粒度层，record id 为 `{video_id}:{entity_hint}:{start}:{end}`。
- `child metadata`：显式写入 `parent_id=video_id`，用于父子关联。
- 在线检索默认读取 `child collection`，`parent collection` 作为层级召回入口，`event collection` 提供时间精确定位。

## Chroma Collection Namespace
- All three Chroma collections share a dataset-level namespace controlled by `AGENT_CHROMA_NAMESPACE`.
- Default namespace is `basketball`, producing `basketball_tracks`, `basketball_tracks_parent`, `basketball_events`.
- Example: `AGENT_CHROMA_NAMESPACE=ucfcrime` yields `ucfcrime_tracks`, `ucfcrime_tracks_parent`, `ucfcrime_events`.
- Priority order for each collection getter:
  1. Explicit full env name (`AGENT_CHROMA_CHILD_COLLECTION`, `AGENT_CHROMA_PARENT_COLLECTION`, `AGENT_CHROMA_EVENT_COLLECTION`).
  2. `AGENT_CHROMA_NAMESPACE` + fixed suffix (`_tracks` / `_tracks_parent` / `_events`).
  3. Built-in default (`basketball_*`).
- `AGENT_CHROMA_RETRIEVAL_LEVEL` is orthogonal: it selects `child` vs `event` for the single-collection online lookup, independent of the namespace.
- Persist via CLI: `python -m agent.db.manage_graph_db switch --chroma-namespace <name>`.
- Rollback to historical behavior: unset `AGENT_CHROMA_NAMESPACE` or set it back to `basketball`.

## 依赖关系
- `node/types.py` 通过 `config.py` 读取默认库路径。
- `test` 与运维脚本通过 `manage_graph_db.py` 执行数据库管理。

## 约定
- 字段变更优先修改 `schema.py` 与 `sqlite_builder.py` 的映射。
- 建库失败必须保留可定位日志。
