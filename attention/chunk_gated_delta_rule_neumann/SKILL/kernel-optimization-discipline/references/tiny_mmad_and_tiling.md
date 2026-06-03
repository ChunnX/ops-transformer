# Tiny MMAD 与 Tiling 性能

当性能分析涉及小矩阵 MatMul/MMAD、`MatmulImpl`、host tiling、AIC/AIV 指标、固定流水开销时读取本文件。

## 主要陷阱

理论 MAC 减少不等于 `aic_mac_time` 下降。tiny MatMul/MMAD 常被固定成本支配：

- `SetSingleShape` / `IterateAll` / `End` 调度；
- GM 到 L1/L0 搬运；
- cube 启动和尾块利用率低；
- Fixpipe 或 MTE3 写回；
- 通过 GM 读写形成 round trip；
- 每个小 block 周围的 scalar setup 循环。

## Tiling 检查

- 区分 host 主 block/chunk 和内部子 block/tile。
- 确认每个 Stage/MatmulImpl 真实读取哪个 tiling data 字段。
- 如果一套通用 tiling 驱动多个 tiny shape，先确认瓶颈不是 copy/sync/writeback 固定开销，再考虑 micro tiling 字段。

## 优化优先级

1. 减少 tiny call 数量和 GM round trip。
2. 在目标允许时保持数据 resident。
3. 布局兼容后合并同 stride load/store。
4. 最后再调 tiny MatMul/MMAD 的 micro tiling。

## 汇报要求

汇报性能候选时说明收益来源：

- MMAD 调用更少；
- MTE2/MTE3/Fixpipe 更少；
- scalar setup/cleanup 更少；
- tiling 更合适；
- 或仅是短序列噪声级改善。
