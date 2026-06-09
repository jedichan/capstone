# shared 模块说明

## 职责
- 提供并行融合模式的共用能力：
- 查询分类器（`query_classifier.py`）
- 融合引擎（`fusion_engine.py`）
- ReAct 执行器（`react_executor.py`）

## 分类器实现与理由
- 方案：规则优先（Rule-based），不依赖额外模型推理。
- 理由：低延迟、可解释、可快速调参；适合当前篮球数据域。
- 标签：`structured` / `semantic` / `mixed`。

## 融合实现与理由
- 融合方式：`Weighted RRF`（加权排名融合）。
- 理由：两路分数不同量纲，直接分数求和不稳定；RRF 对异构检索更稳健。
- 归一化策略：不做分值归一化，改用排名归一化（`1/(k+rank)`）。

## 失败降级策略
- 一路失败/超时：降级为另一路结果（`degraded=true`）。
- 两路都失败：返回空结果并设置 `tool_error`。

## 边界兜底
- 空查询：`semantic`
- 超短查询（<=2 token）：`semantic`
- 歧义查询：`mixed`
