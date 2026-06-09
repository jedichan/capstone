# agent_test 导入工作记录

## 目标
- 解析 `agent/test/data` 下的原始转录文件。
- 将原始转录标准化为接近 `events_vector_flat` 的 JSON。
- 建立统一导入 pipeline，并输出符合当前项目数据库导入格式的产物。
- 保留 `agent_test.xlsx` 的评测标签清洗结果，供后续评测流程使用。

## 阶段记录

### 阶段 1：数据摸底与格式确认
- 时间：2026-05-01
- 已完成内容：
- 初始误判数据源为 `agent/test/agent_test.xlsx`。
- 已更正：真正的原始转录数据源位于 `agent/test/data/UCFCrime_Test.json`。
- `agent_test.xlsx` 本质上更适合作为后续评测标签数据，而不是原始事件 seed。
- 确认现有下游数据库导入习惯是“先标准化，再由 builder 写入 SQLite/Chroma”。
- 确认 workbook 中存在多种格式：
- `Part1`、`Part4`、`Part5`：标准表格型，`Video_id` 存在跨行继承。
- `Part3`：混合型，前半段是块状描述，后半段是标准行。
- `Part2`、`Part6`：当前为空。
- 初步结论：
- 需要建设一个专门的评测数据导入模块。
- 需要同时输出统一 JSON 和本地 SQLite，方便后续 `E2E / RAGAS / generation eval` 复用。

### 阶段 2：解析与清洗规则落地
- 时间：2026-05-01
- 已完成内容：
- 新增导入脚本：`agent/test/agent_test_importer.py`
- 解析策略：
- `Part1`、`Part4`、`Part5` 按标准表格处理，支持 `Video_id` 跨行继承。
- `Part3` 同时支持：
- 前半段块状描述样本：从“问题 / 考察能力 / 预期召回难点 / 难度”中组装 case。
- 后半段标准结构样本：直接按行解析。
- 清洗规则：
- 统一生成 `case_id/source_sheet/source_row/video_id/question/...`
- 统一归一化 `expected_answer_label` 为 `yes/no/unknown`
- 统一归一化 `difficulty_level` 为 `easy/medium/hard/unknown`
- 统一解析 `expected_time_raw -> expected_start_sec/expected_end_sec`
- 自动去除问题字段中的 `问题：` 前缀
- 自动补充 `reference_answer`
- 自动标记下游可用性：
- `retrieval_ready`
- `e2e_ready`
- `generation_ready`

### 阶段 3：导入 pipeline 与产物验证
- 时间：2026-05-01
- 已完成内容：
- 已执行导入命令：
```bash
python /home/yangxp/Capstone/agent/test/agent_test_importer.py --reset
```
- 已产出文件：
- `agent/test/generated/agent_test_normalized.json`
- `agent/test/generated/agent_test_eval.sqlite`
- `agent/test/generated/agent_test_import_report.json`
- `agent/test/generated/agent_test_retrieval_eval.json`
- `agent/test/generated/agent_test_e2e_eval.json`
- `agent/test/generated/agent_test_generation_eval.json`
- 导入结果：
- 总 case 数：`632`
- `Part1`：`312`
- `Part3`：`8`
- `Part4`：`156`
- `Part5`：`156`
- 下游可用统计：
- `retrieval_ready_count = 632`
- `e2e_ready_count = 471`
- `generation_ready_count = 471`
- 验证结果：
- `sqlite` 已成功写入 `632` 条记录
- `Part3` 的块状样本已成功拆解为结构化 case
- 题目文本前缀已清理，适合直接交给后续评测模块

## 当前结论
- 当前已经形成两条可复用的数据链路：
- 评测标签链路：`agent_test.xlsx -> agent/test/generated/agent_test_*.json`
- 原始转录链路：`UCFCrime_Test.json -> agent/test/generated/ucfcrime_events_vector_flat/*.json`
- 后续 `E2E / RAGAS / generation eval` 可以直接消费 `generated` 目录下的标准化产物。
- `agent_test.xlsx` 中缺失标准答案或难度的样本会落到 `unknown`，这是标签层原始数据缺口，不是导入失败。

### 阶段 4：原始转录导入链路更正
- 时间：2026-05-01
- 已完成内容：
- 已确认 `agent/test/data/UCFCrime_Test.json` 是按 `video_id -> duration/timestamps/sentences` 组织的原始转录。
- 新增转录导入脚本：`agent/test/ucfcrime_transcript_importer.py`
- 标准化目标：
- 输出结构尽量贴近现有 `events_vector_flat`：
- 顶层为 `{video_id, duration, events}`
- 每条事件包含：
- `video_id`
- `clip_start_sec`
- `clip_end_sec`
- `start_time`
- `end_time`
- `object_type`
- `object_color`
- `appearance_notes`
- `scene_zone`
- `event_text`
- `keywords`
- `start_bbox_xyxy`
- `end_bbox_xyxy`
- `entity_hint`
- 当前字段策略：
- 时间、文本来自原始转录
- `object_type/object_color/scene_zone/keywords` 使用轻量规则从句子中推断
- bbox 当前未知，统一写为 `null`
- `entity_hint` 按 `segment_{index}` 生成

### 阶段 5：转录标准化与数据库导入验证
- 时间：2026-05-01
- 已完成内容：
- 已执行转录标准化：
```bash
python /home/yangxp/Capstone/agent/test/ucfcrime_transcript_importer.py
```
- 已产出：
- 目录：`agent/test/generated/ucfcrime_events_vector_flat/`
- 清单：`agent/test/generated/ucfcrime_events_manifest.json`
- 产物统计：
- 视频数：`310`
- 事件总数：`4331`
- 抽样验证：
- `Abuse037_x264_events_vector_flat.json` 已与 `basketball_*_events_vector_flat.json` 对齐为相近结构
- 已验证可被当前 `sqlite_builder` 直接导入
- 小样本导入验证：`2` 个视频，共 `10` 条事件
- 全量导入验证：`310` 个视频，共 `4331` 条事件

### 阶段 6：端到端评测流程搭建
- 时间：2026-05-01
- 已完成内容：
- 当前 `xlsx` 评测标签已收敛为仅纳入 `Part1` 和 `Part4`
- `agent_test_importer.py` 已支持 `--include-sheets`
- 过滤后评测数据规模：
- 总 case 数：`468`
- 其中 `Part1`：`312`
- 其中 `Part4`：`156`
- `e2e_ready_count = 310`
- 已新增统一评测脚本：`agent/test/ragas_eval_runner.py`
- 评测框架分为三部分：
- `retrieval`：`ContextPrecisionWithReference`、`ContextRecall`
- `generation`：`Faithfulness`、`AnswerRelevancy`、`FactualCorrectness`
- `end-to-end`：对上述 `RAGAS` 指标做统一聚合，输出 `ragas_e2e_score`
- 评测脚本能力：
- 可复用 `UCFCrime` 标准化事件数据
- 可按 case 的 `video_id` 自动构建最小 `SQLite/Chroma` 子集库
- 可执行图级查询并抓取检索上下文、最终回答、路由信息
- 可输出 `JSON + Markdown` 报告

### 阶段 7：RAGAS smoke test 验证
- 时间：2026-05-01
- 已完成内容：
- 已执行最小验证命令：
```bash
python /home/yangxp/Capstone/agent/test/ragas_eval_runner.py \
  --include-sheets Part1 Part4 \
  --limit 2 \
  --prepare-subset-db \
  --output-dir /home/yangxp/Capstone/agent/test/generated/ragas_eval_smoke
```
- smoke test 数据范围：
- 样本数：`2`
- 涉及视频：`Abuse037_x264`
- 自动构建的子集库：
- `SQLite` 插入行数：`5`
- `Chroma child record count = 5`
- `Chroma parent record count = 1`
- 评测结果摘要：
- `success_count = 2`
- `top_hit_rate = 0.5`
- `context_precision_avg = 0.5`
- `context_recall_avg = 0.5`
- `faithfulness_avg = 0.4`
- `answer_relevancy_avg = 0.377`
- `factual_correctness_avg = 0.0`
- `ragas_e2e_score_avg = 0.3554`
- 已产出报告：
- `agent/test/generated/ragas_eval_smoke/retrieval_report.json`
- `agent/test/generated/ragas_eval_smoke/generation_report.json`
- `agent/test/generated/ragas_eval_smoke/e2e_report.json`
- `agent/test/generated/ragas_eval_smoke/summary_report.json`
- `agent/test/generated/ragas_eval_smoke/summary_report.md`
