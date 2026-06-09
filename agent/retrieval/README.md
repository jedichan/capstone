# retrieval 模块说明

## 目录职责
- 存放检索实验实现与检索器抽象。

## 文件职责
- `event_retriever.py`：SQLite + sqlite-vec 的检索实现（实验/旁路）。

## 对外接口（当前文件内）
- `EventRetriever(...)`
- `retrieve(...)`

## 依赖关系
- 与主图链路相对独立，当前不直接接入 `graph_builder.py`。

## 约定
- 该目录的实现默认视为实验能力，接入主链路前需补回归测试。
