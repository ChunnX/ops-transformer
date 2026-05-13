# 2026-04-30 三方案 Fused AIC Inverse 实施计划

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 目标

对三种最终方案分别实现“更融合”的 AIC inverse 路径，减少当前 `MatmulImpl + GM workspace + FIX/MTE2` 串 tiny MatMul 的固定开销。

**当前范围约束**：暂时只优化 Stage1 内 Neumann inverse 相关路径；Stage2 和 Stage3 暂不改动、不重构、不纳入本轮收益目标。

三种最终方案口径固定为：

| 方案 | 主 chunk | 求逆结构 | 当前状态 |
|---|---:|---|---|
| `C16_1616_Neumann` | 16 | 直接 16×16 Neumann | AIC/FixPipe atomic add，但 `L/power/INV` 仍落 GM |
| `C64_1616_NeumannCompose` | 64 | 16×16 Neumann 子块 + compose | 多次 `MatmulImpl` + GM workspace |
| `C64_3232_NeumannCompose` | 64 | 32×32 Neumann 子块 + compose，短尾可 16×16 | 多次 `MatmulImpl` + GM workspace |

## 当前瓶颈判断

当前三方案都不是真正 fused kernel。共同瓶颈是：

```text
round i: GM -> MTE2 -> Cube(MatmulImpl) -> FixPipe/MTE3 -> GM
round i+1: 再从 GM 读回
```

这导致：

- tiny MatMul 调用次数多，`aic_scalar_time` 放大。
- 每轮 power/INV 中间结果落 GM，`aic_mte2_time` 与 `aic_fixpipe_time` 放大。
- `16×16×16` 和 `32×32×32` 的单次 Cube 工作量太小，固定调度成本无法摊薄。
- 理论 MAC 降低无法直接转化为 wall time 降低。

## Fused 的定义分层

### Level 0：当前实现

- 多次 `MatmulImpl`。
- 输入输出均为 GM tensor。
- 中间结果落 `L_powWsGm_`、`attnWsGm_`、`tmpWsGm_`、`INVWsGm_`。

### Level 1：MatmulImpl 级轻融合

目标：不改变底层 matmul API，只减少不必要 GM buffer 和 AIV 往返。

可做项：

- 用 FixPipe atomic add 替代 AIV `INV += tmp`。
- 删除不用的 `tmpWsGm_`。
- 对 16/32/64 建多套 micro tiling。
- 合并无用 `SyncFixToMte2()`，只在下一次 MTE2 读依赖前同步。

预期：低风险，收益中低；不能根除 GM 往返。

### Level 2：L1/TSCM/VECCALC resident 半融合

目标：尝试让 Matmul 输入或输出从片上 buffer 承接，减少部分 GM 往返。

可调研项：

- 参考 FA/grad 中 `MatmulType<TPosition::TSCM>`、`TPosition::VECCALC`、`GetTensorC()`、`IterateAll<false>(localTensor...)` 用法。
- 尝试把某一轮 matmul 输出放到 local/TSCM，再直接作为下一轮输入。
- 验证 AscendC 当前 arch32/910_93 是否允许 stage1 小矩阵链路跨 matmul 复用片上输出。

风险：API 约束大，可能需要改 MatmulType、tiling、buffer 管理，编译和精度风险高。

### Level 3：专用 AIC micro-kernel 真融合

目标：绕开通用 `MatmulImpl` 的多次 GM 输入输出形态，对固定小矩阵 inverse 写专用 AIC 流程。

理想数据流：

```text
Load L block once
Initialize INV = I - L
for round in Neumann rounds:
    power = power @ power
    INV += INV @ power
Store final INV once
```

关键要求：

- `L/power/INV/tmp` 主要驻留 L1/L0/register。
- round 间不落 GM。
- 最后只写最终 `INV` 到 GM，供后续 `INV @ k_cumdecay` 和 `INV @ v_beta` 使用。

风险：实现成本最高；需要确认是否能直接控制 Cube MMA tile 和片上 buffer 生命周期。

## 三方案实施计划

## 阶段 A：共用基础设施

### A1. 建立方案 selector

目标：每个最终方案有明确 selector，不混用中间名字。

建议文件：

- `chunk_gated_delta_rule_neumann_stage1_C16_1616_FusedAIC.h`
- `chunk_gated_delta_rule_neumann_stage1_C64_1616_FusedAIC.h`
- `chunk_gated_delta_rule_neumann_stage1_C64_3232_FusedAIC.h`

要求：

- 原有最终方案文件先保留为 baseline。
- selector 只在 `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` 切换。
- 每次切换必须同步 host `chunkSize` 与 golden chunk：C16 用 16，C64 用 64。

### A2. micro tiling 字段

新增 tiling data 字段：

- `matmulTilingFp32C16`
- `matmulTilingFp32C32`
- `matmulTilingFp32C64`

host 侧分别 `SetShape/SetOrgShape/SetFixSplit` 获取 tiling。

目标：先验证 Level 1 micro tiling 是否有收益。

### A3. profiling 固化

每个阶段只保留三方案最终目录：

- `chunk_gdr_c16_1616_fused_all_shapes`
- `chunk_gdr_c64_1616_fused_all_shapes`
- `chunk_gdr_c64_3232_fused_all_shapes`

测试必须用 `msprof`，流程遵循 `SKILL/performance-profiling-msprof/SKILL.md`。

## 阶段 B：C16_1616_Neumann 融合

### B1. Level 1 清理和验证

- 删除未使用 `tmpWsGm_`，host workspace 从 `L_pow + tmp + INV` 缩到 `L_pow + INV`，若当前逻辑不再需要 tmp。
- 保留 AIC/FixPipe atomic add。
- 减少 `SyncFixToMte2()` 到真正依赖点。
- 使用 `matmulTilingFp32C16`。

验证：

- `CGDR_GOLDEN_CHUNK_SIZE=16 bash test_run.sh compare`
- 32-shape msprof，对比 `chunk_gdr_prefill_all_shapes_aic_accum`。

预期收益：

- `S=16` 小幅改善。
- 长序列仍受 chunk 数增多和 tiny MatMul 调用次数限制。

### B2. Level 2/3 PoC

优先在 16×16 上做 PoC，因为矩阵固定、round 少、验证快。

PoC 成功标准：

- 16×16 inverse 单块输出和当前 C16 baseline 精度一致。
- `aic_mte2_time`、`aic_fixpipe_time` 明显下降。
- 若 wall time 没下降，说明 Matmul 调度或后续 stage 是主瓶颈。

## 阶段 C：C64_1616_NeumannCompose 融合

### C1. 子块内复用 C16 fused primitive

64×64 下三角矩阵按 16×16 分块：

- 对角块：复用 C16 16×16 inverse primitive。
- 非对角块：通过 block compose 求 lower blocks。

优化重点：

- 减少 16×16 primitive 的调用成本。
- compose 阶段尽量不要 `tmp -> GM -> read tmp`。
- 对 lower-left blocks 尝试用 atomic add 或片上临时 buffer。

风险：

- 16×16 子块数量多，若 primitive 仍依赖 MatmulImpl，固定开销仍大。
- C64_1616 理论上更细粒度，但当前 profiling 已证明它比 3232 差，除非 fused primitive 真能显著降低 tiny 调用成本，否则收益有限。

### C2. 是否继续投入判定

如果 C16 fused primitive 未能让单块 `aic_mte2/fixpipe/scalar` 明显下降，则 C64_1616 不应继续深挖，优先转 C64_3232。

## 阶段 D：C64_3232_NeumannCompose 融合

### D1. 32×32 primitive 优先

原因：当前最终性能最好的是 3232，说明减少 tiny MatMul 调用次数有效。

优化项：

- 使用 `matmulTilingFp32C32`。
- 32×32 Neumann rounds 内减少 GM 中间结果。
- compose 64 from 32 时减少 `tmpWsGm_` 往返。
- 保留 hybrid tail，但不要把 hybrid 作为独立最终方案。

### D2. Tail 策略

按 `validLen`：

- `<=16`: 16×16 primitive。
- `17..32`: 32×32 primitive。
- `33..48`: 32×32 + 16×16 + compose。
- `49..64`: 两个 32×32 + compose。

验证重点：

- `S=16` 不能因为分支和同步开销变慢。
- `S=64` 是最关键的短序列收益判断。
- `S>=512` 关注平均 `aic_scalar/mte2/fixpipe` 是否下降。

## 可优化收益点清单（本轮只执行 Stage1 项）

| 优化点 | 适用方案 | 预期收益 | 风险 |
|---|---|---:|---|
| 16/32/64 多套 micro tiling | 三方案 | 中低 | 低 |
| 删除不用 `tmpWsGm_` / workspace 缩减 | C16、部分 C64 | 低 | 中，需要同步 host offset |
| 减少 FIX_MTE2 同步点 | 三方案 | 中 | 中，错了会读旧数据 |
| 32×32 优先替代 16×16 | C64 | 中高 | 低，已有 profiling 支持 |
| FP16/BF16 inverse 输入 | 三方案 | 中高 | 高，精度与 cast 路径风险 |
| 真 fused AIC primitive | 三方案 | 高 | 高，需 API/底层 micro-kernel 能力 |
| query/key 连续化减少 GM 往返 | 三方案 | 中 | 中高，影响 stage1 数据布局 |
| Stage3 `qkt -> mask -> tmpGM -> matmul` 融合 | 暂缓 | 中 | 本轮明确不动 Stage3，仅记录为远期方向 |
| 按 S/validLen 动态路径 | C64_3232 | 中 | 中，分支调度可能抵消收益 |

## 推荐实施顺序

1. 先做 `C64_3232_NeumannCompose` 的 Level 1：micro tiling + compose 减 GM + 同步点压缩。
2. 同时做 `C16_1616_Neumann` 的 Level 1 清理，作为 fused primitive PoC 的基线。
3. 如果 C16 primitive 能证明片上 fused 有收益，再回头做 `C64_1616_NeumannCompose`。
4. 最后再评估 FP16/BF16 inverse 和真正 micro-kernel。

## 本轮明确不做

- 不修改 `op_kernel/chunk_gated_delta_rule_neumann_stage2.h`。
- 不修改 `op_kernel/chunk_gated_delta_rule_neumann_stage3.h`。
- 不改变 Stage2/Stage3 的 matmul tiling、workspace 布局、mask 流程。
- 不把 Stage3 融合作为本轮性能收益目标；相关想法只保留为远期记录。

## 成功标准

最低标准：

- 三方精度全量通过。
- 每个方案都有 32-shape msprof 完整数据。
- 新方案至少在 `S=16/S=64` 不退化。

理想标准：

- `C64_3232_NeumannCompose` 平均从 `1.778x` normal 降到 `<=1.3x`。
- `S=64` 从 `1.391x` 降到接近或优于 `1.0x`。
- `aic_mte2_time` 与 `aic_fixpipe_time` ratio 明显下降。
