# node 模块说明

## 目录职责
- 承载图节点实现（自查询、分类、并行检索融合、verifier、回答、摘要）。

## 文件职责
- `self_query_node.py`：原始 query 改写。
- `query_classification_node.py`：查询分类（structured/semantic/mixed）+ answer_type。
- `parallel_retrieval_fusion_node.py`：并行执行 SQL 与 Hybrid，并执行加权 RRF 融合。
- `match_verifier_node.py`：existence-grounder（advisory，默认开启，`AGENT_DISABLE_VERIFIER_NODE=1` 可关）。
- `answer_node.py`：最终回答组装；grounder ON 时输出结构化 Yes/No。
- `summary_node.py`：摘要 LLM + bail-out 收紧（P1-Next-A）。
- `pure_sql_node.py`：pure_sql 子 agent 实现（保留作为 sub-agent，不再被默认图引用）。
- `hybrid_search_node.py`：hybrid_search 子 agent 实现（同上）。
- `retrieval_contracts.py`：跨节点的检索数据契约 + 共享 helper。
- `types.py`：`AgentState` 与状态工具函数（`StateResetter`）。
- 其余 `#*.py` / `#*.deprecated`：冻结历史文件，不参与主链路。

## 对外接口（主链路）
- `create_self_query_node(llm)`
- `create_query_classification_node(llm)`
- `create_parallel_retrieval_fusion_node(llm)`
- `create_match_verifier_node(llm=None)`
- `final_answer_node(state, config, store)`
- `create_summary_node(llm=None)`

## 依赖关系
- 被 `graph_builder.py` 调用。
- `pure_sql_node` / `hybrid_search_node` 依赖 `agents/*/sub_agent.py`，作为 sub-agent 工具被外部引用。
- 依赖 `types.py` 提供统一状态定义。

## 约定
- 节点函数签名保持 `node(state, config, store) -> dict`。
- 节点只做编排与状态写回，不内嵌工具实现细节。

## 已删除（P1-5 / P3-3，2026-05-02）
- `tool_router_node.py` / `router_prompts.py` / `reflection_node.py` /
  `cot_engine.py` / `query_optimizer.py` / `error_classifier.py`：
  原 `legacy_router` 路径节点；并行融合稳定后整体废弃。
