# 2026-05-10 P3L1/P8A Full Fusion Roadmap

## 2026-05-11 覆盖规则

- 后续执行以 `todo.md` 为主；本文件保留历史路线和判断依据。
- 精度失败必须在当前目标上正向定位和修复，不能把回滚到 PASS baseline 当成任务完成。
- Packed MMAD 是 P7/P8 的目标，不是冗余；优化对象是重复搬运、copy-out、layout-copy、identity 构造和同步，不是把 packed 计算退回逐 `16x16`。
- `identity64` 目标是第一次有效生成后直接在 L1 复用，但必须先定义 raw-MMAD buffer epoch / `pipe_->Reset()` 生命周期；不能只加一个跨 epoch 的 prepared flag。
- L0C copy-out 若被硬件/API 边界卡住，可以记录为后置项，但不算 PASS，后续仍要单独解决或明确放弃。
- 所有计划完成并通过 gate 后，先保存新 baseline，再用 `msprof` 实测或 simulator/仿真分析寻找新的优化点。

## 当前基线

- Active path: `p3l1_p8a`
- Precision:
  - `S=64` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.23s`。
- Performance:
  - `tests/pytest/prof_out/p3l1_p8a_current_baseline_20260510/performance_compare.csv`
  - 当前 neumann 比 normal 慢 `1.715x ~ 3.231x`。
- Baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_baseline.h`

## 判断

用户提出的方向总体合理，但需要拆成可验证的阶段。

核心原因：

- 当前 compact 16x16 到 packed 64x64 已经逻辑正确，但靠 identity-MMAD 搬运桥实现，性能不理想。
- 当前 P8 compose 还会从 GM 读 lower-left 初始块，不是纯 L1 fuse。
- 当前最终 inverse 仍走 GM half -> AIV cast -> BF16 `attnWsGm_`，不是 AIC direct BF16 writeback。
- final RHS fusion 还会碰到 `gBK/vBeta` 生成时序和 D 维 tiling，必须放在 inverse full-resident 之后。

## 分阶段计划

### Phase 0: 固定当前 correctness baseline

- 已完成。
- 保留：
  - active Stage1: `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
  - rollback baseline: `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_baseline.h`
- 当前版本只作为 precision baseline，不作为 performance baseline。

### Phase 1: lower blocks 预加载到 packed L1

目标：

- 在 `C64InverseBy16x16Neumann()` 开始时，将 P8 compose 需要的 lower-left 初始块预先放到 `aicHalfL1P3Result_`。
- 优先最小加载：
  - `(16,0)` 16x16，用于 top 32x32 compose。
  - `(48,32)` 16x16，用于 bottom 32x32 compose。
  - `(32,0)` 32x32，用于 64x64 compose。
- 可选简单实现：
  - 加载完整 lower-triangular 64x64 相关块，先以正确性为主。

保持不变：

- P3 仍可先在 compact 16x16 中计算，再复制到 packed L1。
- 最终仍写 GM half，再由 AIV cast BF16。
- final RHS 不动。

验收：

- 删除 `P8Compose32From16SingleDirect()` / `P8Compose64From32Direct32()` 内部 lower-left GM load。
- `S=64 / 256 / 1024` smoke PASS。
- 当前 64-aligned compare PASS。
- compare PASS 后跑 performance smoke，对比 Phase 0。

### Phase 2: 去掉 P3 diagonal 的冗余 GM 写回

目标：

- P3 diagonal inverse 只写 packed L1。
- 等 P8 compose 完成后，再统一写最终 inverse 到 GM half。

保持不变：

- final AIV half->BF16 cast 仍保留。
- final RHS 不动。

验收：

- smoke/compare 同 Phase 1。
- 重点观察 `state` 是否仍稳定，以及 `out` 误差是否继续放大。

### Phase 3: P3 改为 P7-style packed landing

目标：

- 让 P3 diagonal inverse 直接以 packed 64x64/P7 对角布局落点生成或写入。
- 减少当前 compact->packed identity-MMAD 搬运桥。

实现方向：

- 参考 `ref/ref.cpp::CopyDiagonalFractalsL1ToL0()` 的 diagonal fractal 抽取思想，但保持本仓库 lower-triangular 方向。
- 优先只替换 diagonal landing，不同时改 P8 compose 公式。

验收：

- smoke/compare 同 Phase 1。
- 若 precision PASS，再做 performance smoke。

### Phase 4: AIC direct BF16 writeback

目标：

- AIC 直接把最终 inverse 写到 `attnWsGm_` BF16。
- 跳过 `INVInitHalfWsGm_ -> AIV -> attnWsGm_` 的最终转换链。

风险点：

- upper triangle 必须为 0。
- tail 行列当前仍未恢复，Phase 4 只覆盖 64-aligned case。
- AIC/AIV/final RHS flag 必须重新梳理：final RHS 需要等待 AIC 写 `attnWsGm_` 完成。

验收：

- smoke/compare 同 Phase 1。
- 重点对比 final RHS 输出，不使用 GM dump 做真值。

### Phase 5: final RHS fusion

目标：

- 保留 inverse/attn 在 L1/L0，加载 RHS:
  - `gBKWsGm_`
  - `vBetaWsGm_`
- 在同一 Stage1 内完成:
  - `attn @ gBK -> outKCumdecayGm_`
  - `attn @ vBeta -> outVInnerGm_`

前置条件：

- Phase 1-4 precision 和性能边界明确。
- `gBK/vBeta` 的 AIV 生成时序已经确认，不与 AIC 消费冲突。
- D 维 tiling 和 L1 容量单独设计。

验收：

- 先只做 `D=64`、64-aligned。
- 再扩展到 `D=128`。
- 最后恢复更宽 compare/performance set。

## 固定规则

- 一次只改一个数据流。
- 每阶段必须经过：
  1. build
  2. install
  3. `S=64 / 256 / 1024` smoke
  4. 当前 64-aligned compare
  5. compare PASS 后再跑 performance smoke
- 非 64-aligned tail case 暂不恢复。
- final RHS fusion 前，不改 Stage2/Stage3。
- 不使用 GM dump/tensor roundtrip 做真值。
