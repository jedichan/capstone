# 测试步骤 — `run_chunks.py`

> 基于 Part4 的分块端到端评测，按视频分批、每批独立 seed 子集。

---

## 前置条件


| 依赖          | 路径                                                                                    |
| ----------- | ------------------------------------------------------------------------------------- |
| 题目 xlsx     | `agent/test/data/agent_test.xlsx`（Part4 sheet）                                        |
| seed 文件     | `agent/test/generated/datasets/ucfcrime_events_vector_flat/*_events_vector_flat.json` |
| eval runner | `agent/test/ragas_eval_runner.py`                                                     |
| 工作目录        | `agent/test/`                                                                         |


---

## 常用命令

```bash
cd agent/test

# 1. 预览分块计划（不执行）
python run_chunks.py --dry-run

# 2. 冒烟测试：只跑第 1 批
python run_chunks.py --max-chunks 1

# 3. 全量运行（所有 5 批）
python run_chunks.py

# 4. 消融实验：关闭 SQL 分支
python run_chunks.py --max-chunks 1 --disable-sql

# 5. 小批次（5 视频/批）
python run_chunks.py --chunk-size 5

# 6. 后台运行 + 日志
nohup python run_chunks.py > run_chunks_stdout.log 2>&1 &
```

---

## 参数说明


| 参数               | 默认值                                              | 说明                                      |
| ---------------- | ------------------------------------------------ | --------------------------------------- |
| `--max-chunks N` | 0（=全部）                                           | 只跑前 N 批                                 |
| `--chunk-size N` | 10                                               | 每批视频数                                   |
| `--disable-sql`  | off                                              | 设 `AGENT_DISABLE_SQL_BRANCH=1`，纯 hybrid |
| `--no-progress`  | off                                              | 关闭 tqdm 进度条                             |
| `--dry-run`      | —                                                | 仅打印分块计划                                 |
| `--xlsx-path`    | `data/agent_test.xlsx`                           | 题目文件                                    |
| `--seed-dir`     | `generated/datasets/ucfcrime_events_vector_flat` | seed 目录                                 |
| `--output-root`  | `generated/ragas_eval_p4_chunks`                 | 输出根目录                                   |


---

## 数据流

```
xlsx (Part4) → 读题 → 按 video_id 分组 → 过滤垃圾 → 切片
                                                  ↓
每批:  复制 seed → 写 case_ids.txt → 调 ragas_eval_runner.py
                                                  ↓
results.csv ← 提取 summary_report.json 指标
```

### 垃圾视频过滤

自动跳过 video_id 不满足条件的行：

- 不包含 `Normal_Videos` 前缀
- 以 `!?`、`RoadAccidents`、`多摄像头` 开头

### seed 双向映射

xlsx 和磁盘文件命名可能不一致，脚本自动处理两种格式：

- `Normal_Videos_924_x264` ↔ `Normal_Videos924_x264`

---

## 输出结构

```
agent/test/generated/ragas_eval_p4_chunks/
├── results.csv                  # 汇总表
├── chunk01/
│   ├── summary_report.json      # 主要指标
│   ├── summary_report.md
│   ├── e2e_report.json          # 逐 case 明细
│   ├── generation_report.json
│   ├── retrieval_report.json
│   ├── REPORT_TABLES.md
│   ├── dataset_part1_part4/     # 导入的标准化数据
│   └── runtime/                 # 子库 (sqlite + chroma)
├── chunk01_seeds/               # 本批 seed 副本
├── chunk01_case_ids.txt         # 本批 case ID 列表
├── chunk01.log                  # 本批运行日志
├── chunk02/ ...
└── ...
```

---

## `results.csv` 字段


| 字段              | 含义                        |
| --------------- | ------------------------- |
| `chunk`         | 批次编号                      |
| `videos`        | 视频数                       |
| `cases`         | 题目数                       |
| `top_hit`       | Top-5 检索命中率               |
| `precision`     | `context_precision_avg`   |
| `recall`        | `context_recall_avg`      |
| `factual_corr`  | `factual_correctness_avg` |
| `custom_corr`   | `custom_correctness_avg`  |
| `e2e`           | `ragas_e2e_score_avg`     |
| `iou_avg`       | 时间窗口 IoU 均值               |
| `vid_match_avg` | 视频匹配分均值                   |
| `latency_s`     | 平均延迟（秒）                   |


---

## 进度查看

```bash
# 看当前跑到哪个 case
grep "Running eval case" generated/ragas_eval_p4_chunks/chunk01.log | tail -1

# 看 RAGAS 评分进度
grep "ragas.*done" generated/ragas_eval_p4_chunks/chunk01.log | tail -1

# 看进程是否还活着
ps -ef | grep run_chunks
```

---

## 分难度汇总

```bash
python3 << 'EOF'
import json
from pathlib import Path

chunk = Path("generated/ragas_eval_p4_chunks/chunk01")
norm = json.loads((chunk / "dataset_part1_part4/agent_test_normalized.json").read_text())
diff = {c["case_id"]: c["difficulty_level"] for c in norm}
e2e = json.loads((chunk / "e2e_report.json").read_text())

for level in ["easy", "medium", "hard"]:
    cases = [c for c in e2e["cases"] if diff.get(c["case_id"]) == level]
    n = len(cases)
    if not n: continue
    top_hit = sum(1 for c in cases if c.get("top_hit")) / n
    prec = sum((c.get("ragas",{}).get("retrieval",{}) or {}).get("context_precision",0) or 0 for c in cases) / n
    rec  = sum((c.get("ragas",{}).get("retrieval",{}) or {}).get("context_recall",0) or 0 for c in cases) / n
    e2e_s = sum((c.get("ragas",{}).get("end_to_end",{}) or {}).get("ragas_e2e_score",0) or 0 for c in cases) / n
    iou = sum((c.get("temporal",{}) or {}).get("temporal_iou",0) or 0 for c in cases) / n
    print(f"{level:<8} n={n}  top_hit={top_hit:.4f}  prec={prec:.4f}  rec={rec:.4f}  e2e={e2e_s:.4f}  iou={iou:.4f}")
EOF
```

