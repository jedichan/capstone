# MEVID Video Pipeline Evaluation

## 1. 什么是 MEVID

MEVID（Motion-based Extended Video Identity Dataset）是一个**多摄像头跨摄像头行人重识别（Re-ID）** 基准数据集。本项目基于此数据集搭建了一条完整的视频理解 + 问答评测流水线。

### 数据集结构

- **时隙（slots）**：6 个 5 分钟时间窗口（`11-20`、`11-55`、`13-50`、`16-35`、`16-45`、`17-00`），每个 slot 包含 6-10 路同步摄像头视频
- **摄像头类型**：admin（G329）、school（G330/G336/G419/G420/G421/G423/G299/G328）、bus（G508/G506）三种场景
- **标注数据**：
  - `mevid-v1-annotation-data/` — 身份标注、tracklet 元数据（track_test_info.txt、query_IDX.txt、test_name.txt）
  - `mevid-v1-bbox-test/` — 逐帧 bbox 标注（约 118 万张行人裁剪）
  - `mevid_slots/` — 实际 `.avi` 视频文件，按 slot 和 camera 组织
- **测试用例**：`agent/test/data/agent_test_mevid.xlsx`，包含 5 类 yes/no 问题：
  - `existence` — 某人是否存在
  - `appearance` — 人物外观（穿着、颜色等）
  - `event` — 行为事件（行走方向、动作等）
  - `cross_camera` — 跨摄像头重识别（同一人在不同摄像头中）
  - `negative` — 否定问题（预期答案为 no）

---

## 2. 6 阶段评测流水线

入口文件 `tests/test_mevid_full.py`，分为 6 个阶段：

| 阶段 | 组件 | 说明 | 成本 |
|------|------|------|------|
| ① | YOLO11m + BoT-SORT | 每路摄像头独立进行行人检测与跟踪 | 本地，无 API |
| ② | OSNet x1.0 (torchreid) | 提取每人外观的 Re-ID 嵌入向量 | 本地，无 API |
| ③ | CameraTopologyPrior | 学习摄像头之间的行人转移时间分布 | 本地，无 API |
| ④ | 跨摄像头贪心匹配 + 并查集 | 将多摄像头的 track 合并为全局实体（global entities） | 本地，无 API |
| ⑤ | LLM 事件细化（可选） | 用 DashScope (qwen3.6-plus) 对事件描述进行细化 | ~¥2-10 CNY |
| ⑤b | 外观细化（可选） | 对实体人物进行 crop-based appearance 描述（颜色、穿着等） | ~¥ 少量 |
| ⑥ | VLM QA 评测 | 将采样帧 + pipeline 上下文送入 VLM，回答 yes/no 问题 | ~¥2 CNY/100 例 |

### Phase 1-4：多摄像头 Pipeline（本地，无 API 消耗）

核心入口函数 `run_pipeline_for_slot()`：

- 读取 slot 对应的 camera → video 映射
- 构造 `CrossCameraConfig`：
  - `embedding_threshold=0.63` — OSNet 跨摄像头余弦相似度阈值
  - `cross_camera_min_score=0.58` — 综合匹配分数阈值
  - `min_overlap_sec=1.0` / `max_transition_sec=180.0` — 时间窗口约束
  - `topology_weight_reid=0.80` / `topology_weight_topo=0.20` — 相似度权重分配
  - `conf=0.40` / `iou=0.40` — YOLO 检测置信度和跟踪 IOU 阈值
- 调用 `run_multi_camera_pipeline()` 执行检测→跟踪→嵌入→拓扑→匹配全流程
- 结果缓存到 `_cache/mevid_pipeline/{slot}_pipeline.json`

### Phase 5：LLM 事件细化

入口函数 `run_refinement_for_slot()`：

- 从 pipeline cache 读取已有结果，避免重新跑 YOLO+OSNet
- 构造 `RefineEventsConfig(mode="vector", frames_per_sec=0.5, min_frames=4, max_frames=12)`
- 调用 `refine_multi_camera_output()` 使用 DashScope 对事件描述进行细化
- 结果缓存到 `_cache/mevid_pipeline/{slot}_refined.json`

### Phase 5b：外观细化

入口函数 `run_appearance_refinement_for_slot()`：

- 对 pipeline 输出的 global entities 进行 crop-based 外观描述
- 生成 `object_color`、`appearance_notes`、`keywords` 等字段
- 结果缓存到 `_cache/mevid_pipeline/{slot}_appearance_refined.json`

### Phase 6：VLM QA 评测

核心函数 `call_vlm()`：

- 从视频中采样帧（base64 编码）
- 构造 VLM 上下文（pipeline 输出的跟踪信息 + 事件描述）
- 使用 DashScope (qwen3.6-plus) 进行 yes/no 判断
- 对于 cross_camera 类问题，额外采样跨摄像头帧和行人裁剪帧
- 答案解析：以 "yes"/"no" 开头匹配，否则标记为 "unknown"
- 结果输出到 `results/mevid_full_*.json`
- 支持断点续跑（`--resume` 参数，通过 `mevid_full_resume.json` checkpoint）

评测指标：
- 按类别（existence / appearance / event / cross_camera / negative）统计准确率
- 按难度（easy / medium / hard）统计准确率
- Token 消耗与费用统计

---

## 3. 三条评测路径

| 路径 | 入口脚本 | 说明 |
|------|----------|------|
| A: 纯视频 QA | `tests/test_mevid_full.py` / `scripts/run_mevid_video_eval.py` | 运行完整 6 阶段 pipeline + VLM 问答，不涉及 agent |
| B: 生成向量种子 | `scripts/generate_mevid_vector_flat.py` | 将 pipeline cache 转换为 `*_events_vector_flat.json`，供 Chroma 向量检索使用 |
| C: 视频+Agent 端到端 | `tests/test_mevid_video_agent_e2e.py` / `scripts/run_mevid_agent_eval.py` | 先跑 B，再构建 SQLite+Chroma 数据库，加载 LangGraph agent，评测 top-hit rate 和答案准确率 |
| D: Re-ID 基准 | `tests/test_mevid_evaluation.py` + `tests/extract_mevid_crops.py` | 独立的跨摄像头 Re-ID 评测（Rank-1/5、mAP） |

### 路径 A: 纯视频 QA

```
tests/test_mevid_full.py
├── Phase 1-4: YOLO + OSNet + Topology + Matching
├── Phase 5: LLM event refinement (可选)
├── Phase 5b: Appearance refinement (可选)
└── Phase 6: VLM QA evaluation
```

### 路径 B: 生成向量种子

入口 `scripts/generate_mevid_vector_flat.py`（约 656 行）：
- 读取 pipeline cache 中的 global_entities、merged_events
- 为每个事件生成 enriched 文本描述（cross-camera context、appearance keywords、entity hints）
- 输出到 `agent/test/data/events_vector_flat/{video_id}_events_vector_flat.json`
- 也可内联运行 pipeline（不需要提前缓存）

### 路径 C: 视频+Agent 端到端

入口 `tests/test_mevid_video_agent_e2e.py`（约 670 行）：
1. 从 xlsx 导入测试用例
2. 运行 generate_mevid_vector_flat 生成向量种子
3. 构建临时 SQLite + Chroma 数据库
4. 加载 LangGraph agent 图
5. 逐 case 运行 agent 查询
6. 评测指标：
   - **Top-hit rate**: 检索是否召回了正确的视频
   - **Answer accuracy**: Agent 最终回答是否正确
   - **Category breakdown**: 按类别统计
7. 输出 `results/mevid_agent_e2e/{slot}_{timestamp}/`
   - `summary.json` / `summary.md` — 汇总报告
   - `case_results.json` — 逐 case 详细结果
   - `selected_cases.json` — 选中的用例列表

---

## 4. 数据流

```
MEVID .avi 视频 (每 slot 每 camera)
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Phase 1-4: YOLO + OSNet + Topology + Matching   │
│  输出: _cache/mevid_pipeline/{slot}_pipeline.json │
└──────────────────────────────────────────────────┘
       │
       ├── Phase 5: LLM refinement (可选)
       │   输出: {slot}_refined.json
       │
       ├── Phase 5b: appearance refinement (可选)
       │   输出: {slot}_appearance_refined.json
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Phase 6: VLM QA                                 │
│  输入: Excel 用例 + 采样帧 + pipeline 上下文      │
│  输出: results/mevid_full_*.json                  │
│  指标: accuracy by category                       │
└──────────────────────────────────────────────────┘
       │
       ▼ (供 agent 路径使用)
generate_mevid_vector_flat.py
  → agent/test/data/events_vector_flat/{video_id}_events_vector_flat.json
       │
       ▼
test_mevid_video_agent_e2e.py
  → SQLite + Chroma DB
  → LangGraph agent
  → results/mevid_agent_e2e/{slot}_{timestamp}/
  指标: top-hit rate + answer accuracy
```

---

## 5. 相关文件清单

### 核心评测脚本

| 文件 | 用途 |
|------|------|
| `tests/test_mevid_full.py` | 主评测入口 — 6 阶段 pipeline + VLM QA（1599 行） |
| `tests/test_mevid_video_agent_e2e.py` | 视频+Agent 端到端评测（670 行） |
| `tests/test_mevid_evaluation.py` | MEVID-v1 Re-ID 基准评测（684 行） |
| `tests/test_mevid_qa.py` | VLM-only QA（不使用完整多摄像头 pipeline） |
| `tests/test_mevid_video_modules.py` | 模块烟雾测试 |
| `tests/extract_mevid_crops.py` | 选择性提取 MEVID 行人裁剪 |

### 便捷封装

| 文件 | 用途 |
|------|------|
| `scripts/run_mevid_video_eval.py` | `test_mevid_full.py` 的 CLI 封装 |
| `scripts/run_mevid_agent_eval.py` | `test_mevid_video_agent_e2e.py` 的 CLI 封装 |
| `scripts/generate_mevid_vector_flat.py` | 将 pipeline cache 转为向量种子（656 行） |

### 支撑模块

| 文件 | 用途 |
|------|------|
| `video/factory/multi_camera_coordinator.py` | 多摄像头 pipeline 编排（YOLO+OSNet+topology+matching） |
| `video/factory/person_crop_sampler.py` | 行人裁剪采样 |
| `video/factory/appearance_refinement_runner.py` | 基于裁剪的外观细化 |
| `video/indexing/search_enrichment.py` | 服装/颜色关键词规范化，用于 RAG 检索 |
| `video/core/schema/multi_camera.py` | CrossCameraConfig 配置定义 |
| `agent/db/config.py` | 数据库配置，支持 `mevid` 命名空间 |

---

## 6. 常用命令

### 纯视频 QA

```bash
# 单个 slot，跳过 LLM 细化，启用外观细化
python tests/test_mevid_full.py --slot 13-50 --limit 40 --no-refine --appearance-refine

# 所有 6 个 slot，启用 LLM 细化
python tests/test_mevid_full.py --video-dir _data/mevid_slots --refine

# 断点续跑
python tests/test_mevid_full.py --video-dir _data/mevid_slots --resume

# 使用 GPU 加速 Re-ID
python tests/test_mevid_full.py --slot 13-50 --reid-device cuda
```

### Agent 端到端评测

```bash
# 单个 slot
python scripts/run_mevid_agent_eval.py --slot 13-50 --limit 40

# 所有 slot
python scripts/run_mevid_agent_eval.py --limit 100
```

### 生成向量种子

```bash
python scripts/generate_mevid_vector_flat.py --slot 13-50
```

### Re-ID 基准评测

```bash
python tests/test_mevid_evaluation.py
```

---

## 7. 费用估算

| 场景 | 阶段 | 预估费用 |
|------|------|----------|
| 纯视频 QA（无细化） | Phase 1-4 + Phase 6 | ~¥2 CNY（100 例 QA） |
| 纯视频 QA（含细化） | Phase 1-6 | ~¥10 CNY |
| Agent E2E | Phase 1-6 + Agent | ~¥10-15 CNY |
| 所有前置阶段 | Phase 1-4 | 免费（纯本地计算） |
