# Matmul Tiling Performance Skill

用于分析 AscendC `MatmulImpl`、tiny MatMul、host tiling、AIC/AIV 指标与性能瓶颈。触发场景：用户问 C16/C64 为什么慢、tiling 是否正确、AIC MAC/FixPipe/MTE2/Scalar 为什么放大、如何优化 Neumann 求逆。

## 核心判断

理论 MAC 减少不等于 `aic_mac_time(us)` 减少。当前 Neumann 慢点主要是：

```text
MatmulImpl + GM workspace + FIX/MTE2 串 tiny MatMul
```

每个 tiny MatMul 都有固定成本：

- `SetSingleShape/IterateAll/End` 调度成本
- GM→L1/L0 的 MTE2 搬运
- Cube 计算启动和尾块低利用率
- FixPipe/MTE3 写回 GM
- 下一轮再从 GM 读回，形成串行依赖

## Chunk 与子块概念

- `C16` / `C64`：host tiling 主 `chunkSize`。
- `1616` / `3232`：Neumann 求逆内部子矩阵大小。
- `C64_1616` / `C64_3232` 主 chunk 仍是 64，不应把 host `chunkSize` 切到 16 或 32。

## Tiling 检查规则

检查 host：`op_host/chunk_gated_delta_rule_neumann_tiling.cpp`

- 主 `C=16` 方案：`int64_t c = 16`。
- C64 子块方案：`int64_t c = 64`。
- 当前只有一套 `matmulTilingFp32` 时，16×16/32×32 tiny MatMul 仍复用通用 FP32 tiling。

检查 kernel：`op_kernel/chunk_gated_delta_rule_neumann.h`

- stage1/2/3 当前都用 `tiling_->matmulTilingFp32` 初始化。
- 如果新增 micro tiling，需要新增 tiling data 字段并让对应 Stage1 MatmulImpl 使用正确字段。

## Micro Tiling 优化方向

可尝试增加多套 tiling 字段：

- `matmulTilingFp32C16`：用于 `16x16x16`。
- `matmulTilingFp32C32`：用于 `32x32x32` 和 `32x32x16`。
- `matmulTilingFp32C64`：用于主 64 matmul。

预期收益：降低 tiny MatMul 的不合适切分和 buffer 配置成本。风险：如果 `MatmulImpl` 固定启动/FixPipe/GM 往返占主导，收益有限。

## Fused AIC Inverse 状态

截至当前三种方案：都没有真正 fused kernel。

- `C16_1616_Neumann`: 只把 `INV += INV@Lpow` 迁到 AIC/FixPipe atomic add，仍落 GM。
- `C64_1616_NeumannCompose`: 16×16 子块和 compose 仍通过多次 `MatmulImpl` + GM workspace。
- `C64_3232_NeumannCompose`: 32×32 子块和 compose 仍通过多次 `MatmulImpl` + GM workspace。

真正 fused 的目标：让 `L/power/INV` 尽量驻留 L1/L0/register，round 间不落 GM，最终只写一次 `INV`。

## 优化优先级

1. 减少 tiny MatMul 调用次数：优先 32×32 而不是 16×16。
2. 尝试 micro tiling：给 16/32/64 分别配 tiling。
3. 减少 GM workspace 往返：删除不用的 tmp，避免 AIV 往返，能 atomic add 就 atomic add。
4. 研究真正 fused AIC inverse：若 MatmulImpl 不支持跨 round 保持 L0/L1，就需要更底层 micro-kernel。

## Fused AIC Plan

三方案融合实现计划见：`docs/2026-04-30/FUSED_AIC_INVERSE_THREE_SCHEMES_PLAN.md`。

执行该计划时必须保持三方案口径：

- `C16_1616_Neumann`
- `C64_1616_NeumannCompose`
- `C64_3232_NeumannCompose`

不要把 `sync_reduced/tail_specialized/hybrid` 等过程标签当成最终方案。

## 当前范围约束

按用户最新要求，近期 fused 优化只覆盖 Stage1 Neumann inverse。暂时不要修改 Stage2/Stage3，也不要把 Stage3 融合作为当前计划的一部分。
