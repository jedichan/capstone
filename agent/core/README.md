# core 模块说明

## 目录职责
- 承载运行时共用能力（环境装载、默认模型构建）。

## 文件职责
- `runtime.py`：提供 `load_env()` 与 `build_default_llm()`。
- `__init__.py`：导出 core 入口函数。

## 对外接口
- `load_env(project_root: Path | None = None) -> None`
- `build_default_llm() -> ChatOpenAI`

## 依赖关系
- 被 `graph.py` 调用。
- 不依赖业务节点实现。

## 约定
- `core` 只放运行时基础设施，不放检索业务逻辑。
