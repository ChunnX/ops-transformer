# 2026-04-30 Fused / FP16 / INV@(I+P) 分阶段执行计划

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 总目标

围绕三个最终方案做 Stage1 Neumann inverse 优化，不改 Stage2/Stage3：

1. `C16_1616_Neumann`
2. `C64_1616_NeumannCompose`
3. `C64_3232_NeumannCompose`

每个阶段必须闭环：

```text
实现 -> 全量三方精度通过 -> seq 遍历性能测试 -> 阶段总结文档
```

不满足精度门槛的实现不得进入性能对比，也不得作为下一阶段基线。

## 精度门槛

用户确认 FP16 精度标准为0.08，即：

```text
atol = 0.08
rtol = 0.08
```

该门槛用于 FP16 方案和后续 `INV@(I+P)` 方案的三方精度判断。FP32 fused 方案原则上应继续达到既有全量三方精度通过标准；若默认脚本阈值调整为 `0.08/0.08` 后仍需全部通过。

精度检查项保持三方脚本现有口径：

- `CPU golden vs ChunkGDR out`
- `CPU golden vs ChunkGDR state`
- `CPU golden vs ChunkGDRNeumann out`
- `CPU golden vs ChunkGDRNeumann state`
- `ChunkGDR vs ChunkGDRNeumann out`
- `ChunkGDR vs ChunkGDRNeumann state`

CPU golden 逻辑不改。

## 性能测试口径

性能测试只做所有 `seq` 遍历，不要求完整 `B × S × D × H` 全量矩阵。

建议固定典型维度，遍历：

```text
S = 16, 64, 128, 256, 512, 1024, 2048
```

性能必须继续使用现有 `msprof` warmup + icache/dcache flush pipeline，不使用 torch event benchmark。

每阶段输出：

- `performance_compare.csv`
- `performance_raw.csv`
- 阶段总结 Markdown

## 阶段 1：三方案 fusedMM

### 目标

先做 fused，看性能收益。此阶段不引入 FP16，不做 `INV@(I+P)` 公式改写。

目标是让 Neumann/compose 中间矩阵尽量留在 AIC 片上路径，减少 GM workspace 往返：

- 优先减少 `L_powWsGm_ / attnWsGm_ / tmpWsGm_` 的落 GM 次数。
- 优先让 C64 compose 的 `tmp_lower` 留 local。
- 最终 `INV` 如可行，直接喂 `INV @ gBK` 和 `INV @ vBeta`。

### 方案覆盖

- `stage1_C16_FP32_CVInverse.h`
- `stage1_C64_1616_FP32_CVInverse.h`
- `stage1_C64_3232_FP32_CVInverse.h`

### 通过标准

- 三个方案全量三方精度全部通过。
- 完成 seq 遍历性能测试。
- 输出 `docs/2026-04-30/FUSED_STAGE1_PERFORMANCE_ANALYSIS.md`。

## 阶段 2：三方案 FP16

### 目标

在阶段 1 fused 基础上实现真实 FP16/BF16 Neumann 输入链路，形成 `3 × 2` 对比：

```text
C16_FP32_fused      vs C16_FP16_fused
C64_1616_FP32_fused vs C64_1616_FP16_fused
C64_3232_FP32_fused vs C64_3232_FP16_fused
```

### 关键要求

- `stage1_C16_FP16_CVInverse.h` 必须是真 half/bf16 输入 Matmul，不允许继续使用 `float/float/float MatmulType` 冒充 FP16。
- 若 final `INV` 为 half/bf16，而 `gBK/vBeta` 仍为 float，需要二选一：
  - 将 final `INV` cast 回 FP32 后做后续 Matmul。
  - 同步生成 `gBK/vBeta` half/bf16 副本，后续 Matmul 也走 half/bf16 输入。

### 新增文件

- `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_CVInverse.h`
- `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_CVInverse.h`

### 通过标准

- FP16 三个方案全量三方精度全部满足 `atol=rtol=0.08`。
- 完成 seq 遍历性能测试。
- 输出 `docs/2026-04-30/FP16_STAGE1_PERFORMANCE_ANALYSIS.md`。

## 阶段 3：`INV += INV@P` 改写为 `INV@(I+P)`

### 目标

在阶段 2 的最优基线上尝试公式改写：

```text
INV_new = INV_old + INV_old @ P
        = INV_old @ (I + P)
```

其中 `P` 为 `L²/L⁴/L⁸/...`。实现时需要给 `P` 写单位对角形成 `I+P`，并用 ping-pong buffer 避免同一 tensor 同时读写。

### 回退规则

若出现任一情况，先停在当前阶段正向定位和修复；只有明确放弃当前候选或需要重建干净工作面时，才恢复到阶段 2 最优实现：

- 全量三方精度不能满足 `atol=rtol=0.08`。
- seq 遍历性能没有稳定收益。
- `aic_scalar / aic_mte2 / aic_fixpipe` 指标没有改善或退化明显。
- 实现引入额外同步/拷贝导致总耗时上升。

### 通过标准

- 全量三方精度满足 `atol=rtol=0.08`。
- 完成 seq 遍历性能测试。
- 输出 `docs/2026-04-30/INV_FACTOR_STAGE1_PERFORMANCE_ANALYSIS.md`，明确保留或回退。

## 文档要求

每阶段总结文档必须包含：

- 修改了哪些文件。
- 当前 selector 和 host `chunkSize`。
- 精度命令和结果。
- 性能 profiling 目录和 CSV 路径。
- 按 `S` 聚合的耗时比。
- AIC/AIV 指标变化，至少包括 `aic_mac_time`、`aic_mte2_time`、`aic_fixpipe_time`、`aic_scalar_time`、`aiv_vec_time`。
- 本阶段保留/回退结论。

## 执行顺序

1. 阶段 1：三方案 fusedMM。
2. 阶段 2：三方案 FP16，形成 `3 × 2` 性能对比。
3. 阶段 3：尝试 `INV@(I+P)`，无收益则回退。
