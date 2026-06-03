# 2026-05-06 C16 Final RHS Resident Plan

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 背景

此前 `fulllocal/C16_FP16` 只让 16x16 Neumann inverse 内部 resident：最终 `INV` 仍写回 `INVWsGm_`，随后 `INV @ gBK` 与 `INV @ vBeta` 通过 GM `AICProcess` 消费。

本轮目标是让 final RHS 的 `INV` operand 直接消费 half L1/TSCM 中的 resident `INV`，避免 `INV -> GM -> AICProcess` 的回读链路。

## 实现

- active selector：`fulllocal/C16_FP16/chunkSize=16`。
- active 文件：`op_kernel/fulllocal/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocal.h`。
- 开关：
  - `FP16_L1_RESIDENT_INVERSE=true`
  - `FP16_L1_RESIDENT_RHS=true`
  - `FP16_L1_WRITEBACK_INV_FOR_DEBUG=false`
- final RHS：
  - `INV @ gBK` 使用 `ResidentInvRhsGm(aicHalfL1TileInv_, gBKWsGm_, outKCumdecayGm_, dk_)`
  - `INV @ vBeta` 使用 `ResidentInvRhsGm(aicHalfL1TileInv_, vBetaWsGm_, outVInnerGm_, dv_)`
- resident 生命周期：
  - `Neumann16InverseBlock()` 完成后保留 `aicHalfL1TileInv_`。
  - 两路 RHS 完成后再恢复 raw MMAD 状态与 Stage1 Matmul local buffers。
- 安全回退：
  - `UseResidentRhsForCurrentShape()` 要求 `dk` 与 `dv` 均为 16 的倍数。
  - 非对齐维度写回 `INVWsGm_` 并回退 GM `AICProcess` final RHS。

## 验证

- 构建：通过。
- 安装：通过。
- 全量三方精度：`20 passed in 5.96s`。
- 非 16 倍数回退 smoke：`B=1,S=32,nk=4,nv=4,dk=48,dv=80,seed=41`，通过。

## 性能

RHS resident 四点 Neumann/normal ratio：

| S | ratio |
|---|---:|
| 16 | 1.109x |
| 64 | 2.423x |
| 125 | 2.671x |
| 2048 | 5.026x |

相比 no-RHS-resident C16 fulllocal，RHS resident 的 Neumann 绝对耗时为 `1.018x / 1.091x / 1.077x / 1.244x`，没有收益。

## 结论

本轮完成了 final RHS 的 `INV` operand resident/raw 化，但性能变慢。原因是 RHS operand 仍从 FP32 GM workspace 进入，需要额外转换为 half L1/TSCM tile，再做 raw MMAD 和 Fixpipe 写回。

后续若继续追性能，应把 `gBK/vBeta` 生成端也改成可被 AIC raw MMAD 直接消费的片上 half L1/TSCM 数据，或减少 `LoadGmFloatTileToHalfL1ByMmad()` 的转换成本。
