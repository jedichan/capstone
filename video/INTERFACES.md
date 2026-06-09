# Video 模块接口文档

本文档记录 `video/` 模块对外暴露的核心接口，按子目录组织。
仅覆盖本人负责的部分（视觉感知 + 多摄像头追踪 + 评估）。

---

## 目录结构

```
video/
├── core/
│   ├── models/                    # 算法模型层
│   │   ├── camera_topology.py     # 相机拓扑先验（GMM + 冷启动）
│   │   ├── reid_embedder.py       # OSNet/MobileNetV2 Re-ID 特征提取
│   │   └── event_refinement_llm.py
│   └── schema/
│       ├── multi_camera.py        # 多摄像头数据结构
│       └── refined_event_llm.py
├── factory/                       # 编排层
│   ├── coordinator.py             # 单摄像头管线
│   ├── multi_camera_coordinator.py# 多摄像头管线
│   └── processors/
│       ├── vision.py              # YOLO 检测 + 跟踪
│       ├── cross_camera_matcher.py# 跨摄像头身份匹配
│       └── analyzer.py / captioner.py / event_track_pipeline.py
├── ingestion/                     # 数据接入层（视频/JSON 加载）
├── indexing/                      # 索引层（向量库 / 图存储）
└── common/                        # 工具
```

---

## 1. `video.core.models.camera_topology`

### `CameraTopologyPrior`

学习并维护一个 **directed-pair → 通行时间分布** 的先验。  
对有 ≥5 次观测的相机对拟合 3-component GMM；其余使用线性衰减冷启动。

```python
from video.core.models.camera_topology import CameraTopologyPrior

prior = CameraTopologyPrior(
    cameras=["301", "329", "336"],
    max_transit_sec=600.0,    # 硬截止
    min_obs_for_gmm=5,
)

# 在线观测一次跨摄像头转移
prior.observe(cam_a="336", cam_b="639", delta_t=4.2)

# 批量灌入（评估场景常用）
prior.observe_batch([("336", "639", 4.2), ("639", "336", 5.1), ...])

# 给定一对相机和时间间隔 → 返回 [0, 1] 概率分数
s = prior.score(cam_a="336", cam_b="639", delta_t=5.0)

# 序列化
prior.save("results/topology.json")
prior2 = CameraTopologyPrior.load("results/topology.json")
```

**关键方法**

| 方法 | 用途 |
|------|------|
| `observe(a, b, dt)` | 记录单次确认转移 |
| `observe_batch(triples)` | 批量观测；自动重新拟合 |
| `score(a, b, dt) → float` | **核心评分接口**，[0,1] |
| `expected_transit_sec(a, b)` | 估计平均通行时间 |
| `transition_table()` | 序列化全部相机对统计 |
| `most_connected_pairs(top_k)` | 排序观测最多的相机对 |
| `save / load` | JSON 持久化 |
| `CameraTopologyPrior.from_confirmed_matches(...)` | 从 GT 转移直接构造 |

---

## 2. `video.core.models.reid_embedder`

### `ReIDEmbedder`

行人外观 Re-ID 特征提取器。后端自动选择：
- `torchreid_osnet`（默认，512-d，~2.2M 参数，OSNet x1.0）
- `torchvision` MobileNetV2（fallback，1280-d）

```python
from video.core.models.reid_embedder import ReIDEmbedder
import cv2

embedder = ReIDEmbedder(device="cuda")          # GPU 推理

crops = [cv2.imread("p1.jpg"), cv2.imread("p2.jpg")]
feats = embedder.embed_crops(crops)              # (N, dim) L2-normalized

sim = ReIDEmbedder.cosine_similarity(feats, feats)  # (N, N)
```

**配置点**：`_OSNET_VARIANT` 类变量可切换 `osnet_x0_5` / `osnet_x0_75` / `osnet_x1_0`。

---

## 3. `video.core.schema.multi_camera`

多摄像头管线的纯数据结构（dataclass）。

```python
PersonCrop          # 单帧 crop（image_array, jpg_base64, t_sec, camera_id, track_id）
CameraAppearance    # 一个全局 entity 在某摄像头的出现窗口
GlobalEntity        # 跨摄像头合并后的全局 ID + 多次 appearances
CameraResult        # 单摄像头 Stage-1 全部输出（tracks, events, embeddings, crops）
CrossCameraConfig   # 匹配超参（threshold, transition_sec, weight_reid/topo）
MatchVerification   # 可选 LLM 复核结果
MultiCameraOutput   # 整个 pipeline 最终输出
```

### `CrossCameraConfig` 关键参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `max_transition_sec` | 30.0 | 候选对最大时间间隔 |
| `embedding_threshold` | 0.65 | cosine 准入阈值 |
| `cross_camera_min_score` | 0.65 | **组合得分准入阈值（τ）** |
| `topology_weight_reid` | 0.55 | cosine 在组合得分中的权重 |
| `topology_weight_topo` | 0.45 | topology 在组合得分中的权重 |
| `same_camera_max_gap_sec` | 3.0 | 同摄像头碎片缝合时间窗 |
| `same_camera_reid_threshold` | 0.80 | 同摄像头碎片缝合相似度阈值 |
| `llm_verify_cosine_min/max` | 0.65 / 0.80 | borderline 区间触发 VLM 复核 |

---

## 4. `video.factory.processors.vision`

YOLO 检测 + 跟踪的顶层入口。

```python
from video.factory.processors.vision import run_yolo_track_on_video

result = run_yolo_track_on_video(
    video_path="cam1.mp4",
    model="yolo11m",          # 自动解析为绝对路径
    tracker="bytetrack",
    device="cuda",            # 自动 fallback 到 cpu
    conf=0.25,
    classes=[0],              # COCO person
)
# result: tracks, frames_meta, fps, ...
```

---

## 5. `video.factory.processors.cross_camera_matcher`

跨摄像头身份合并核心算法。

```python
from video.factory.processors.cross_camera_matcher import match_across_cameras

global_entities = match_across_cameras(
    cameras=[cam_result_a, cam_result_b, cam_result_c],
    config=CrossCameraConfig(cross_camera_min_score=0.65),
    topology_prior=prior,        # 可选；提供后启用 0.55*cos + 0.45*topo
    embedder=embedder,           # 可选；用于 same-camera fragment stitching
    llm_verifier=None,           # 可选；borderline 区间触发 VLM
)
```

**算法管线**：
1. `build_candidate_pairs` — 时间约束筛选
2. `score_candidate_pairs` — 组合评分（cosine + topology）
3. `_greedy_assign` — 按分数贪心分配
4. `_build_global_entities` — Union-Find 聚类 + 在线 topology 更新

---

## 6. `video.factory.multi_camera_coordinator`

多摄像头管线总入口。

```python
from video.factory.multi_camera_coordinator import (
    run_multi_camera_pipeline, save_multi_camera_output
)

output = run_multi_camera_pipeline(
    video_paths={"cam1": "a.mp4", "cam2": "b.mp4"},
    config_yaml="config/multi_cam.yaml",  # 可选
    topology_prior_path="results/topology.json",  # 可选；存在则加载，结束保存
    device="cuda",
    save_crops=False,
)

save_multi_camera_output(output, "results/multicam.json")
```

**内部步骤**：
1. 各摄像头独立跑 Stage-1（YOLO + 跟踪 + Re-ID embedding）
2. 同摄像头内碎片缝合（`_stitch_same_camera_fragments`）
3. 跨摄像头匹配生成 `GlobalEntity`
4. 合并事件输出（带 `global_entity_id`）

---

## 7. 评估脚本（`tests/`）

| 脚本 | 用途 |
|------|------|
| `tests/extract_mevid_crops.py` | 从 MEVID-v1 bbox tar 解压 → JPG 训练数据 |
| `tests/test_mevid_evaluation.py` | MEVID Re-ID 标准评估（Rank-1/5/10, mAP）；含 topology 增益 |
| `tests/eval_multicam_entity.py` | 跨摄像头 entity-level Precision/Recall/F1/Purity |
| `tests/test_camera_topology.py` | CameraTopologyPrior 单元测试 |
| `tests/test_meva_pipeline.py` | MEVA KF1 端到端 smoke test |
| `tests/test_multi_camera.py` | 多摄像头管线集成测试（mock + 小数据集） |
| `tests/pipeline_economy_test.py` | 长视频运动门控经济性测试 |
| `tests/test_motion_coverage.py` | 运动覆盖率统计（46 视频） |
| `tests/test_event_track_grounding.py` | YOLO 事件 vs LLM 时序定位对比 |
| `tests/test_uca_*` | UCA 时序定位评估族（46/173 子集，COCO/TSGV 指标，ablation） |

---

## 评分公式（核心）

```
score(track_i, track_j) = topology_weight_reid * cosine(emb_i, emb_j)
                        + topology_weight_topo * topology_prior.score(cam_i, cam_j, |Δt|)
```

默认权重 `0.55 / 0.45`，可在 `CrossCameraConfig` 覆盖。

`τ = cross_camera_min_score`（默认 0.65）控制召回 / 精度的折衷点。
