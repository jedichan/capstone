# RAGAS 调整路径文档

## 文档目的
- 汇总 `agent/test/generated` 下最近一批评测报告。
- 提炼当前链路的主要瓶颈、已有有效动作和不建议继续投入的方向。
- 形成一条可执行的后续调整路径，便于后续实验按同一口径推进。

## 汇总范围
- 近期小样本运行：
  - `agent/test/generated/runs/ragas_eval_gpt4o_c3/summary_report.md`
  - `agent/test/generated/runs/ragas_eval_gpt4o_c1/summary_report.md`
  - `agent/test/generated/runs/ragas_eval_gpt4o_agent_embed_probe/summary_report.md`
- 已有对比实验：
  - `agent/test/generated/ragas_eval_compare_tuned/compare_report.md`
  - `agent/test/generated/ragas_eval_rerank_check/compare_report.md`
- 最近主链路结果：
  - `agent/test/generated/ragas_eval_latest/summary_report.md`
  - `agent/test/generated/ragas_eval_top30_current_chain_rrf_fix/summary_report.md`

## 近期报告结论

### 1. 近期小样本运行说明回答端仍不稳定
- `ragas_eval_gpt4o_c1`：
  - `top_hit_rate=0.75`
  - `context_precision_avg=0.5`
  - `context_recall_avg=0.5`
  - `faithfulness_avg=0.3333`
  - `factual_correctness_avg=0.0425`
  - `ragas_e2e_score_avg=0.2656`
- `ragas_eval_gpt4o_c3`：
  - `top_hit_rate=0.75`
  - `faithfulness_avg=0.4405`
  - `factual_correctness_avg=0.0425`
  - `ragas_e2e_score_avg=0.2938`
- 结论：
  - `c3` 比 `c1` 有小幅提升，主要体现在 `faithfulness` 和 `e2e`。
  - 但 `factual_correctness` 基本没有改善，说明问题不只是“召回到了没有”，还包括“回答如何约束到证据”。

### 2. `agent_embed_probe` 说明 embedding 不是唯一主因
- `ragas_eval_gpt4o_agent_embed_probe` 的 `context_precision_avg/context_recall_avg=1.0`，但 `factual_correctness_avg` 仍是 `0.0`。
- 结论：
  - 单纯把检索命中做高，不会自动带来生成质量提升。
  - 当前更像是“检索命中 + 回答落地方式 + 证据约束”三者同时影响结果。

### 3. 调优后 `LlamaIndex` 主链路优于基线，但代价是时延
- `ragas_eval_compare_tuned/compare_report.md` 显示：
  - `top_hit_rate: 0.8 -> 0.9`
  - `ragas_e2e_score_avg: 0.2242 -> 0.2546`
  - `factual_correctness_avg: 0.014 -> 0.026`
  - `avg_latency_ms: 10642.07 -> 14748.17`
- 结论：
  - `LlamaIndex SQL + Vector + fallback` 方向是有效的，应继续保留。
  - 当前收益已经验证，但时延成本明显，后续要避免继续扩大候选规模。

### 4. `rerank` 有价值，但放大候选池不是好方向
- `ragas_eval_rerank_check/compare_report.md` 显示：
  - 接入默认 `rerank` 后：
    - `faithfulness_avg: 0.2201 -> 0.3125`
    - `context_recall_avg: 0.375 -> 0.3889`
    - `ragas_e2e_score_avg: 0.2345 -> 0.2667`
  - 放大候选池后：
    - `top_hit_rate: 0.9 -> 0.8`
    - `ragas_e2e_score_avg: 0.2667 -> 0.2508`
- 结论：
  - `rerank` 本身是正收益动作。
  - `rerank_candidate_limit=20` 这类较小候选池更接近当前最优点。
  - 单纯堆候选数量不会稳定提升效果，反而会拉高时延并扰乱排序。

### 5. 最近主链路结果表明“局部有效，全量仍弱”
- `ragas_eval_latest/summary_report.md`：
  - `case_count=10`
  - `top_hit_rate=0.7`
  - `factual_correctness_avg=0.55`
  - `answer_relevancy_avg=0.1283`
  - `faithfulness_avg=0.1667`
  - `ragas_e2e_score_avg=0.279`
- `ragas_eval_top30_current_chain_rrf_fix/summary_report.md`：
  - `case_count=30`
  - `top_hit_rate=0.4`
  - `context_precision_avg=0.1833`
  - `context_recall_avg=0.15`
  - `factual_correctness_avg=0.2667`
  - `time_range_overlap_iou_avg=0.1333`
- 结论：
  - 在 `10-case` 上已有部分 case 能答对，但回答相关性和证据一致性仍不稳。
  - 到 `30-case` 后，检索覆盖和时间定位明显掉队，说明现有收益还没有稳定泛化。

## 持续暴露的问题

### 1. 负样本与 `pure_sql` 失败仍反复出现
- `PART1_0002` 在多份近期报告中持续 `Top hit=False`。
- `PART1_0011` 在 `ragas_eval_latest` 中也仍未命中。
- 说明：
  - `pure_sql` 路径对否定型或细节约束型问题仍然脆弱。
  - 当前 fallback 只能兜底一部分 case，不能替代查询理解本身的改进。

### 2. 命中了也不等于答对
- 典型现象：
  - `Top hit=True`，但 `context_precision/context_recall` 仍为 `0`
  - 或 `factual_correctness` 很低
- 说明：
  - 当前存在“命中同视频但片段不对”“证据有了但回答写偏了”两类问题。

### 3. 时间定位能力仍明显不足
- `30-case` 报告里：
  - `time_range_overlap_iou_avg=0.1333`
  - `hit@0.3=0.125`
  - `hit@0.5=0.125`
- 说明：
  - 当前不仅要找对视频，还要更稳定地找对片段。
  - 时间段定位不足会直接拖累 `retrieval` 和 `end-to-end`。

### 4. 时延已经接近上限，不适合继续堆复杂度
- 当前多组结果都表明：
  - 更大的候选池不会带来稳定收益。
  - `rerank` 和 `LlamaIndex` 带来的时延已经较明显。
- 说明：
  - 后续应该做“定向优化”，而不是继续增加全局搜索空间。

## 已验证有效的方向
- 保留 `LlamaIndex SQL + Vector` 双链路。
- 保留 `SQL 0 结果 fallback` 到旧 SQL。
- 保留 `Vector filter` 放宽与旧 `Chroma` fallback。
- 保留轻量 `rerank`，但不继续扩大候选池。
- 保留 `RAGAS` 单 case / 小 batch 评分与重试策略，`ragas_metric_error_cases=0` 这一点已经稳定。

## 不建议继续优先投入的方向
- 不建议继续单纯放大 `rerank` 前候选池。
- 不建议把主要精力放在只换 embedding 上。
- 不建议在缺少 case 归因前继续叠加新的全局模块。

## 建议调整路径

### 阶段 1：先补检索覆盖，再谈回答优化
- 目标：
  - 先把 `top_hit_rate` 和 `context_recall` 稳住。
- 优先动作：
  - 针对持续失败 case 建立小清单，优先看 `PART1_0002`、`PART1_0011` 这一类重复失败样本。
  - 区分 `pure_sql` 失败、`hybrid_search` 失败、命中错片段三种类型。
  - 对否定型、细粒度约束型问题补 query rewrite 或 query decomposition 规则。
  - 对同视频内片段混淆 case，收紧 clip 级排序特征，而不是扩 candidate 数量。
- 建议观察指标：
  - `top_hit_rate`
  - `context_recall_avg`
  - `time_range_overlap_iou_avg`

### 阶段 2：把回答约束到证据上
- 目标：
  - 改善“命中了但答偏了”的问题。
- 优先动作：
  - 强化回答模板，要求先给结论，再给证据片段与理由。
  - 对证据不足 case 明确输出保守结论，避免过度补全。
  - 把 `citation` / clip 证据约束继续前置到最终回答。
  - 结合现有 `match_verifier` 结果，减少与证据不一致的自由生成。
- 建议观察指标：
  - `factual_correctness_avg`
  - `faithfulness_avg`
  - `answer_relevancy_avg`

### 阶段 3：专项修时间定位
- 目标：
  - 提升命中片段而不是只命中视频。
- 优先动作：
  - 复盘 `temporal_iou=0` 但 `video_match=true` 的 case。
  - 检查 parent projection 后是否损失了最优 child clip。
  - 对最终输出 clip 增加“时间段一致性”排序因子。
- 建议观察指标：
  - `time_range_overlap_iou_avg`
  - `hit@0.3`
  - `hit@0.5`

### 阶段 4：最后再收时延
- 目标：
  - 在不回退效果的前提下控制平均耗时。
- 优先动作：
  - 维持当前较优的小候选池 `rerank` 配置。
  - 优先裁剪无收益步骤，而不是裁剪已证明有效的 `rerank`。
  - 对 `pure_sql` 和 `hybrid_search` 分开统计时延，避免只看总平均值。
- 建议观察指标：
  - `avg_latency_ms`
  - 各 route mode 的分布时延

## 推荐实验顺序
- 第 1 轮：
  - 固定当前较优 `rerank` 配置，做失败 case 归因。
- 第 2 轮：
  - 只改 query rewrite / query decomposition / clip 排序，不同时改 prompt。
- 第 3 轮：
  - 在检索稳定后，再改最终回答模板与 verifier 收敛逻辑。
- 第 4 轮：
  - 用 `10-case -> 50-case -> 全量` 的顺序逐步放大验证，避免被小样本误导。

## 当前建议结论
- 主方向继续沿着“`LlamaIndex` 双链路 + 轻量 `rerank` + fallback + verifier 收敛”推进。
- 近期最该优先解决的不是“再加更多模块”，而是：
  - 重复失败 case 的定向归因
  - 命中后回答与证据不一致
  - 时间段定位偏差
- 如果下一轮只能做一件事，优先做“失败 case 归因表 + 针对性检索改动”，因为这一步同时会提升 `top_hit`、`context_recall` 和时间定位的可解释性。
