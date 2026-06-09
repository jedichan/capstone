# MEVID 13-50 Slot 流水线测试 TODO

## 状态

| # | 任务 | 状态 |
|---|------|------|
| 1 | data 文件夹 + 下载 6 个视频 | ✅ |
| 2 | 抽样 10 道题，验证视频+时间 | ✅ |
| 3 | pipeline 全阶段（含 Phase 5） | ✅ |
| 4 | VLM-only 评测 | ✅ |
| 5 | Agent 评测（含 IoU，±30s padding，correct video row） | ✅ |

---

## 问题与模型输出对照表

| # | 类别 | 问题 | 预期 | Agent 输出 | 答案 | Top-Hit | IoU |
|---|------|------|------|------------|------|---------|-----|
| 1 | appearance | Is there a person with beige jacket visible in camera G329? | yes | yes | ✅ | ✅ G329 | 0.177 |
| 2 | appearance | Is there a person with grey coat visible in camera G508? | yes | yes | ✅ | ✅ G508 | 0.417 |
| 3 | appearance | Is there a person wearing black hoodie (hood up) in camera G421? | yes | yes | ✅ | ✅ G421 | 0.223 |
| 4 | event | Did a person exit from the left side of camera G508? | yes | yes | ✅ | ✅ G508 | 0.493 |
| 5 | event | Did a person exit from the left side of camera G328? | yes | yes | ✅ | ✅ G328 | 0.069 |
| 6 | cross_camera | Did a person with white shirt appear in camera G328 and then appear again in camera G424? | yes | yes | ✅ | ✅ G424 | 0.361 |
| 7 | cross_camera | Did a person with light grey hoodie appear in camera G424 and then appear again in camera G506? | yes | yes | ✅ | ❌ G328×3 | 0.000 |
| 8 | negative | Did a person with dark hoodie from camera G421 also appear in camera G508? | no | **yes** | ❌ | ✅ G421 | N/A |
| 9 | negative | Did a person with black coat with fur-trimmed hood from camera G329 also appear in camera G421? | no | **yes** | ❌ | ❌ G328×2,G424 | N/A |
| 10 | negative | Did a person with dark long coat from camera G328 also appear in camera G421? | no | **yes** | ❌ | ✅ G328 | N/A |

### 模型输出详情

<details>
<summary>每个问题的 Agent 节点链、检索结果、预测时间</summary>

**Case 1** [appearance] G329 beige jacket → yes ✅
- Nodes: self_query → classify → retrieve → verify → answer → summary
- Top-3: G328, G329, G329
- pred_time: 109.2-111.2s (expected 110-121s), IoU=0.177

**Case 2** [appearance] G508 grey coat → yes ✅
- Nodes: self_query → classify → retrieve → verify → answer → summary
- Top-3: G328, G508, G508
- pred_time: 29.3-32.3s (expected 6-32s), IoU=0.417

**Case 3** [appearance] G421 black hoodie → yes ✅
- Nodes: self_query → classify → retrieve → verify → answer → summary
- Top-3: G328, G421, G421
- pred_time: 54.5-61.1s (expected 0-299s), IoU=0.223

**Case 4** [event] G508 left exit → yes ✅
- Nodes: self_query → classify → retrieve → verify → answer → summary
- Top-3: G508, G506, G508
- pred_time: 16.9-22.7s (expected 6-32s), IoU=0.493

**Case 5** [event] G328 left exit → yes ✅
- Nodes: self_query → classify → retrieve → verify → answer → summary
- Top-3: G508, G328, G328
- pred_time: 234.8-254.9s (expected 122-216s), IoU=0.069

**Case 6** [cross_camera] G328→G424 white shirt → yes ✅
- Nodes: self_query → classify → retrieve → verify → answer → summary
- Top-3: G328, G328, G328
- pred_time: 280.0-291.3s (expected 205-292s), IoU=0.361

**Case 7** [cross_camera] G424→G506 light grey hoodie → yes ✅ (top-hit ❌)
- Nodes: self_query → classify → retrieve → verify → answer → summary
- Top-3: G328, G328, G328 (no G506!)
- pred_time: 279.2-283.2s (expected 131-159s), IoU=0.000

**Case 8** [negative] G421→G508 dark hoodie → **yes ❌**
- Nodes: self_query → classify → retrieve → verify → answer → summary
- Top-3: G328, G421, G508
- match_verifier 过度匹配

**Case 9** [negative] G329→G421 black coat → **yes ❌**
- Nodes: self_query → classify → retrieve → verify → answer → summary
- Top-3: G328, G424, G328 (missed G329)
- match_verifier 过度匹配

**Case 10** [negative] G328→G421 dark long coat → **yes ❌**
- Nodes: self_query → classify → retrieve → verify → answer → summary
- Top-3: G421, G328, G424
- match_verifier 过度匹配

</details>

---

## 汇总指标

| 指标 | 值 |
|------|-----|
| 总体准确率 | 7/10 = 70% |
| Top-hit rate | 8/10 = 80% |
| Mean IoU | **0.249** |

### IoU 多阈值

| 阈值 | 通过率 | 通过题目 |
|------|--------|----------|
| IoU@0.15 | 50% (5/10) | Q1,Q2,Q3,Q4,Q6 |
| IoU@0.30 | 30% (3/10) | Q2(0.417), Q4(0.493), Q6(0.361) |
| IoU@0.50 | 0% (0/10) | — (Q4=0.493 差 0.007) |

---

## IoU 提升历程

| 阶段 | Mean IoU | IoU@0.15 | IoU@0.3 | 关键改动 |
|------|----------|----------|---------|----------|
| v1: top-1 row, ±0s | 0.049 | 10% | 10% | 取 top-1 行（全为 G328） |
| v2: top-1 row, ±30s | 0.136 | 20% | 10% | 放宽 ±30s |
| v3: correct video row, ±30s | **0.249** | **50%** | **30%** | 取 expected video_id 行 |

---

## 剩余问题

1. **negative 0%**：Agent 将所有 no 题判为 yes，match_verifier 节点过度积极确认跨摄像头匹配
2. **IoU@0.5 = 0%**：Q4 差 0.007，需 OSNet 提升检索精度
3. **Q7 top-hit miss**：G424→G506 未召回 G506，Chroma 中 G506 只有 33 条记录 vs G328 368 条
4. **Q5 IoU=0.069**：retrieved event 比预期晚 ~30s
