# agents 模块说明

## 目录职责
- 聚合各子 agent 的装配与执行实现。
- 明确划分子域：`pure_sql`、`hybrid_search`、`shared`。

## 文件职责
- `__init__.py`：导出聚合入口（component builder）。
- `pure_sql/component.py`：`pure_sql_node` 的装配组件定义。
- `pure_sql/sub_agent.py`：Pure SQL 子 agent 执行实现。
- `hybrid_search/component.py`：`hybrid_search_node` 的装配组件定义。
- `hybrid_search/sub_agent.py`：Hybrid 子 agent 执行实现。
- `shared/component_spec.py`：共用组件协议 `NodeComponent`。
- `shared/react_executor.py`：共用 ReAct 执行器。

## 对外接口
- `build_pure_sql_node(llm)`
- `build_hybrid_search_node(llm)`
- `run_pure_sql_sub_agent(user_query: str, llm)`
- `run_hybrid_sub_agent(user_query: str, llm)`
- `run_react_sub_agent(...)`

## 依赖关系
- 依赖 `node/*_node.py` 的工厂函数（装配层）。
- 依赖 `tools/*.py` 作为子 agent 工具执行层。
- 被 `graph_builder.py` 与 `node/*_node.py` 调用。

## 约定
- 子 agent 逻辑优先放在 `agents/<domain>/sub_agent.py`。
- 共用能力统一放入 `agents/shared`，避免跨域复制。
- 目录间尽量单向依赖：`component -> node`，`node -> sub_agent`，禁止循环导入。

## 并行融合方案
- 执行模式：`parallel_fusion`（P1-5 / P3-3 清理后唯一模式，2026-05-02 移除 `legacy_router`）。
- 分类器：`agents/shared/query_classifier.py`（规则优先，兜底 `mixed/semantic`）。
- 融合器：`agents/shared/fusion_engine.py`（Weighted RRF，不直接做异构分数求和）。
- 选型理由：SQL 与 Hybrid 分数不同量纲，采用排名融合更稳定且便于降级。
- 超时配置：`AGENT_PARALLEL_BRANCH_TIMEOUT_SEC`（一路超时自动降级为另一路）。

## 分类器样例（英文）
- `Did anyone stay near the left bleachers for a long time?` -> `semantic`
- `Show me all dark-clothed persons.` -> `structured`
- `Who moved across center-right court around 40 to 51 seconds?` -> `mixed`
- `Find events where a person was standing still near the baseline.` -> `semantic`
- `How many person tracks are there in basketball_2.mp4?` -> `structured`
- `Was there someone near the sidewalk after the first movement?` -> `mixed`
- `List records with object_type = person and object_color = dark.` -> `structured`
- `Can you find a person pacing around the bleachers?` -> `semantic`
- `Show person events between 0s and 30s in basketball_2.mp4.` -> `structured`
- `I am looking for someone who briefly crossed the center court area.` -> `semantic`

## 边界兜底
- 空查询：默认 `semantic`
- 超短查询（<=2 token）：默认 `semantic`
- 歧义查询（结构化/语义信号都弱）：默认 `mixed`
