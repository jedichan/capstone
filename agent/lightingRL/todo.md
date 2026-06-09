# lightingRL Todo

- 更新时间: `2026-05-01`
- 维护规则: 分为 `计划任务` 与 `交付任务` 两组，完成后持续更新状态。

## 计划任务

- [x] 建立 `agent/lightingRL` 工作目录
- [x] 盘点当前 `agent` 默认执行链、prompt 入口、评测入口
- [x] 调研 `Agent-lightning/LightningRL` 的 prompt 优化与 RL 训练能力
- [x] 明确第一阶段训练目标只针对当前默认主链路
- [x] 输出完整接入方案到 `plan.md`
- [ ] 将当前 prompt 从节点代码中抽离为可注入的 `prompt registry`
- [ ] 建立 `agent_test` 到 `Agent-lightning dataset` 的转换脚本
- [ ] 建立基于 `RAGAS + 路由命中 + 延迟约束` 的 reward 适配层
- [ ] 建立 `Trainer.dev()` 干跑脚本验证 traces 与 reward
- [ ] 建立 `APO` prompt 训练脚本并完成首轮 baseline 对比
- [ ] 视首轮结果决定是否进入 `VERL` 模型权重训练

## 交付任务

- [x] 交付 `work.md`
- [x] 交付 `todo.md`
- [x] 交付 `plan.md`
- [x] 交付当前仓库与框架的调研结论
- [x] 交付 prompt 优先级排序与不建议训练项
- [x] 交付分阶段接入路线
- [ ] 交付 `prompt registry` 原型代码
- [ ] 交付 `dataset builder` 原型代码
- [ ] 交付 `reward adapter` 原型代码
- [ ] 交付 `APO` 实验脚本
- [ ] 交付 baseline 与优化后对比报告

## 当前结论

- 第一优先级: `summary prompt`
- 第二优先级: `query classification prompt`
- 第三优先级: `self_query prompt`
- 备注: 原 `legacy_router` 下的 init prompt / router prompts 已于 P1-5 / P3-3 （2026-05-02）随节点删除，不再需要纳入
