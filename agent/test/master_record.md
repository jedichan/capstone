# agent/test 总记录文档

## 文档目的
- 作为 `agent/test` 目录下的统一总记录入口。
- 记录数据来源、评测配置、模型与 API、优化方案、时间测试、风险、结论和后续计划。
- 后续所有重要实验与变更，优先追加到本文件，再视情况链接到细分报告。

## 当前目录内的关键记录文件
- 工作过程记录：`agent/test/work.md`
- 今日任务维护：`agent/test/todo.md`
- 运行时间预估报告：`agent/test/generated/ragas_runtime_estimate_report.md`
- `GPT-4o` 优化与时间报告：`agent/test/generated/ragas_gpt4o_optimization_report.md`
- 评测输出目录：
- `agent/test/generated/ragas_eval_smoke/`
- `agent/test/generated/ragas_eval_estimate_4_top3/`
- `agent/test/generated/ragas_eval_gpt4o_probe/`
- `agent/test/generated/ragas_eval_gpt4o_c1/`
- `agent/test/generated/ragas_eval_gpt4o_c3/`
- `agent/test/generated/ragas_eval*/`

## 数据与输入口径
- 原始转录源：
- `agent/test/data/UCFCrime_Test.json`
- 标准化事件产物：
- `agent/test/generated/ucfcrime_events_vector_flat/*.json`
- 评测标签源：
- `agent/test/agent_test.xlsx`
- 当前评测只纳入：
- `Part1`
- `Part4`

## 当前评测入口
- 统一评测脚本：
- `agent/test/ragas_eval_runner.py`
- 当前评测维度：
- `retrieval`
- `generation`
- `end-to-end`

## 模型与 API 记录

### Agent 主图模型
- 模型：`qwen3.6-plus`
- API：`DashScope` OpenAI-compatible API
- 代码入口：`agent/core/runtime.py`

### RAGAS 评测模型
- 当前目标配置：
- 模型：`gpt-4o`
- API：`OpenAI API`
- 鉴权：读取环境变量 `OPENAI_API_KEY`
- 当前固定端点：`https://api.openai.com/v1`
- 代码入口：`agent/test/ragas_eval_runner.py`

### RAGAS embedding
- 当前目标配置：
- 模型：`text-embedding-3-small`
- API：`OpenAI API`
- 鉴权：读取环境变量 `OPENAI_API_KEY`
- 当前固定端点：`https://api.openai.com/v1`

## 评测优化设计
- 并行方案：
- 按 case 并发执行 `RAGAS` 打分
- 通过 `--ragas-concurrency` 控制并发度
- 上下文优化：
- 通过 `--ragas-max-contexts` 限制传入上下文条数
- 通过 `--ragas-max-context-chars` 限制单条上下文长度
- 通过 `--ragas-max-total-context-chars` 限制总上下文长度
- 通过 `--ragas-max-response-chars` 限制回答长度
- 通过 `--ragas-max-reference-chars` 限制参考答案长度
- 指标优化：
- `AnswerRelevancy` 支持降低 `strictness`
- `FactualCorrectness` 当前使用较省 token 的 `precision` 口径
- 稳定性优化：
- metric 级异常单独记录，不让单个 metric 直接中断整轮

## 追加记录区

### 2026-05-01
- 已建立 `RAGAS` 统一评测流程
- 已完成 `Part1 + Part4` 数据过滤
- 已完成 `smoke test`
- 已完成第一版运行时间预估
- 已完成：
- `RAGAS` 评测模型切换到 `gpt-4o`
- `RAGAS embedding` 切换到 `text-embedding-3-small`
- 新增 `RAGAS` 并行打分参数：`--ragas-concurrency`
- 新增上下文裁剪参数：
- `--ragas-max-contexts`
- `--ragas-max-context-chars`
- `--ragas-max-total-context-chars`
- `--ragas-max-response-chars`
- `--ragas-max-reference-chars`
- 新增指标轻量化参数：
- `--answer-relevancy-strictness`
- 当前默认测试配置：
- `top_k = 3`
- `ragas_concurrency = 3`
- `answer_relevancy_strictness = 2`
- 已完成时间测试：
- `concurrency=1`：`real 90.30s / 4 cases`
- `concurrency=3`：`real 69.50s / 4 cases`
- 加速结果：
- 总墙钟时间优化 `23.03%`
- `RAGAS` 打分阶段优化 `49.85%`
- 当前推荐配置：
- `GPT-4o + OpenAI API + concurrency=3 + top_k=3`
