# Routing Rules

## Default Runtime Semantics

The runtime mode is `parallel_fusion` (the only mode after the
P1-5 / P3-3 cleanup on 2026-05-02; the previous `legacy_router`
fallback was removed once the parallel-fusion path stabilised).

- Both retrieval paths run:
  - `pure_sql`
  - `hybrid`
- Query classification outputs:
  - `structured`
  - `semantic`
  - `mixed`
- The label is used to bias fusion weights, not to disable one branch.

## Default Graph Contracts

- `self_query_node` runs before all retrieval nodes.
- `query_classification_node` produces `classification_result` and
`answer_type`.
- `parallel_retrieval_fusion_node` produces:
  - `sql_result`
  - `hybrid_result`
  - `rerank_result`
  - `sql_debug.fusion_meta`
- `match_verifier_node` (advisory) produces `verifier_result`; can be
skipped via `AGENT_DISABLE_VERIFIER_NODE=1`.
- `final_answer_node` produces a grounded draft answer (or a structured
Yes / Likely-yes / No answer when
`AGENT_ENABLE_EXISTENCE_GROUNDER=1`).
- `summary_node` produces the final user-facing answer and appends
minimal citations.

## Reliability

In the default graph, reliability comes from:

- dual-path execution
- branch timeout handling
- degradation fallback (with `fusion_meta.degraded_reason` for
observability)
- fusion guardrails such as `structured_zero_guardrail`
- Weighted RRF fusion biased by `classification_result.signals`
- the optional `match_verifier_node` when running
`answer_type=existence` queries