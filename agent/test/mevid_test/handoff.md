# MEVID 评测与 Agent 交接说明（handoff）

> 文件名：`handoff.md`（与口语中的 “handoff” 一致；若习惯 `hanoff.md` 可自行复制或软链。）

本文档面向接手本目录评测、Agent 行为与后续改动的同事，概括目标、目录结构、运行方式、已做结论与**明确不建议**的捷径。

---

## 1. 目标与范围

- **业务目标**：在 MEVID 多摄像头场景下，验证「流水线 + RAG + LangGraph Agent」对**带时间**问题的回答质量，并与 **VLM-only** 基线对比。
- **本目录范围**：`agent/test/mevid_test/` 内的小样本（10 题）、缓存、评测脚本与报告；**不**替代主仓库文档，但可作为本实验的单一入口。

更细的流程说明见同目录 **`test_process.md`**；指标与逐题表见 **`todo.md`**。

---

## 2. 目录结构（要点）

| 路径 | 说明 |
|------|------|
| `data/` | 6 段 × 13–50 机位 `.avi` 等原始/切片视频 |
| `sampled_10.json` | 10 道精选题（含 `expected_time`、`expected_video_id` 等，与流水线对齐） |
| `_cache/mevid_pipeline/` | 流水线/精修/外观缓存（如 `13-50_*.json`），由 `tests/test_mevid_full.py` 等生成 |
| `events_vector_flat/*.json` | 由流水线 + 外观构建的扁平事件，供 Agent RAG 入库（约 691 条） |
| `run_custom_eval.py` | **VLM-only**：读 `sampled_10.json` + 多份缓存，DashScope VLM，yes/no + 时间 IoU |
| `run_agent_eval.py` | **Agent**：SQLite + Chroma、`AGENT_*` 环境、LangGraph、top-hit / 准确率 / IoU |
| `agent_eval_results/` | 评测 JSON/MD 与 `runtime/` 下本地 DB |
| `todo.md` | 状态、Q&A 表、IoU 分档、已知问题摘要 |
| `test_process.md` | 流水线与评测步骤说明 |

---

## 3. 环境与前置

- 全项目约定：**在 `conda activate capstone` 环境下**运行 Python 与项目命令。
- Agent 评测依赖：Chroma、SQLite、LangGraph、以及仓库内 Agent 图与节点（尤其 **`agent/node/match_verifier_node.py`**）。
- VLM 评测依赖：DashScope 等（见脚本内配置与 `todo.md`）。

---

## 4. 如何运行（典型命令）

在仓库根目录（或按脚本内 `ROOT` 约定）执行，且已 `conda activate capstone`：

```bash
# VLM-only 评测（示例，具体参数以脚本为准）
python agent/test/mevid_test/run_custom_eval.py

# Agent 评测（会写 agent_eval_results/ 与 runtime/）
python agent/test/mevid_test/run_agent_eval.py
```

流水线缓存若缺失，需先按 `test_process.md` / `tests/test_mevid_full.py` 跑通生成 `_cache/mevid_pipeline/` 再评。

---

## 5. 评测逻辑要点（已落地）

### 5.1 时间 IoU

- 预测区间与标注区间比较时，对**预测区间**做了 **±30 秒 padding**，再算 IoU（与 `run_custom_eval` / `run_agent_eval` 中实现一致）。
- **Agent IoU**：应用 **按题匹配 `expected_video_id` 对应行** 的预测，避免「全局 top-1 被某一机位（如 G328）霸占」导致 IoU 失真。
- 报告中常同时给出多档阈值（如 IoU@0.15 / 0.30 / 0.50）；`todo.md` 中有汇总。

### 5.2 指标叙事（量级，以 `todo.md` 为准）

- Agent：回答准确率约 **70%** 量级，top-hit 约 **80%**；Mean IoU 在修正行匹配与 padding 后约 **0.25** 量级；IoU@0.15 约一半、@0.50 很低（个别题如 Q4 可较高）。
- VLM-only：在外观相关问题上弱于 Agent 路径（历史结论，以最新跑分为准）。

---

## 6. 已知问题与根因（接手重点）

### 6.1 负例（Negative）Q8–Q10 仍判 **yes**

- **现象**：负例样本上 Agent 仍倾向预测「有匹配」。
- **根因方向**（文档化结论，非脚本补丁可根治）：**多摄像头路径下 `match_verifier_node`（v2.4）** 在「全局实体已覆盖问题涉及的机位」时，逻辑上**强依赖** verifier 给出 `exact`/`partial`；流水线若产生**假阳性的全局实体**， verifier 易走到「有匹配」，从而整体答 **yes**。
- **已验证无效或不足**：仅在 `run_agent_eval.py` 侧做外观注入、「FACT: no match」类启发式，**未能**修复负例；需在 **Agent 节点 / 流水线质量** 上动真格。

### 6.2 不建议的「昏招」（产品/科学上应拒绝）

- **关闭 verifier** 或绕过多摄像头路径，仅为刷评测数字——会掩盖真实失败模式，**不建议**。
- 为负例单独「关多摄像头」等**与线上一致性相悖**的 hack——**不建议**。

---

## 7. 建议的后续工作（按优先级）

1. **负例与多摄像头一致性**  
   - 在 **`agent/node/match_verifier_node.py`**（及与 `graph_builder`、query classifier 的衔接）上设计**正规**方案：例如在「机位已覆盖」时仍要求**外观/查询与证据强对齐**再允许 `exact`/`partial`；或为「跨机位否定」单独分支；需与产品/负责人对齐口径。  
   - 同步评估：减少流水线中**错误的全局 Re-ID 合并**（假全局实体）从源头降低误触发。

2. **流水线**（可选）  
   - 更强 Re-ID（如 OSNet）减少假全局实体 → 重建 `events_vector_flat` → 重跑 Agent 评测 → 更新 `todo.md`。

3. **回归**  
   - 任何 Agent 或 verifier 改动后：重跑 `run_agent_eval.py`（及如需对比的 `run_custom_eval.py`），更新 `todo.md` 表格与结论段。

---

## 8. 关键代码引用（便于跳转）

| 主题 | 路径 |
|------|------|
| 多摄像头匹配与 verifier | `agent/node/match_verifier_node.py` |
| 图与节点 wiring | `agent/graph_builder.py` |
| Agent 侧评测与 IoU/负例实验 | `agent/test/mevid_test/run_agent_eval.py` |
| VLM 侧评测 | `agent/test/mevid_test/run_custom_eval.py` |

---

## 9. 交接检查清单

- [ ] 已阅读 `test_process.md` 与 `todo.md`  
- [ ] 能在 `capstone` 环境下跑通至少一种评测脚本  
- [ ] 理解负例问题与「勿关 verifier / 勿绕多摄像头」的边界  
- [ ] 下一版改动有明确的评测回归命令与 `todo.md` 更新责任人  

---

*文档生成目的：交接；若与 `todo.md` 中最新数字冲突，以最新一次评测输出与 `todo.md` 为准。*
