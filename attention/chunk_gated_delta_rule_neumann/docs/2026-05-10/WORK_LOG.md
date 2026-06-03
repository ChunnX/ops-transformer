# 2026-05-10 工作日志

## 当前 active 与验证

- Active selector:
  - `op_kernel/selected/stage1_selected.h`
  - `op_kernel/selected/stage1_selected_config.h`
- Active Stage1:
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- 当前版本定位：
  - Phase 6b active path：group load 两个 diagonal `16x16` block 到 packed L1，再分别执行两个已验证的 `16x16` P7 direct solves。
  - P3 diagonal inverse 结果落到 `aicHalfL1P3Result_` packed 64x64 tile。
  - `32x32 from 16x16` compose 走 L1-resident。
  - `64x64 from 32x32` compose 走 L1-resident。
  - final RHS 已走 Stage1 内 AIC raw-MMAD fused path，直接消费 `aicHalfL1ResidentInv_`。
  - 当前通过边界不是 strict `32x32/64x64` group-MMAD diagonal solver；失败的 strict helper 未保存为 baseline。

## build / install / precision

- build PASS:
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=3.173828e-02`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=4.882812e-04`
- 当前 64-aligned compare set PASS:
  - `16 passed in 6.23s`

## performance smoke

- 性能输出:
  - `tests/pytest/prof_out/p3l1_p8a_current_baseline_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/p3l1_p8a_current_baseline_20260510/performance_raw.csv`
- 第一次尝试带 `--aiv-metrics=ArithmeticUtilization` 失败：
  - 当前 `msprof` 不支持 `--aiv-metrics`。
  - 未计入性能结果。
- 使用仓库默认 `msprof --application=... --task-time=on` 重跑成功。

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 632.780 | 1085.400 | 1.715 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1674.540 | 4547.520 | 2.716 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5034.700 | 16189.480 | 3.216 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24408.040 | 78863.120 | 3.231 | normal |

结论：

- 当前 `p3l1_p8a` 是 precision/correctness baseline，不是 performance win。
- 回退主因符合预期：当前仍有 identity-MMAD 搬运桥、packed/compact 回读、GM half 写回、AIV half->BF16 cast 和 lower-left GM on-demand load。
- 后续优化不能把当前版本描述为性能收益，只能作为可回退正确性边界。

## baseline 保存

- 已保存当前 precision-passed active Stage1:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_baseline.h`
- 内容与 active Stage1 一致，仅文件名和 include guard 改为 baseline 名称。
- selector 未切换，仍保持当前 active `p3l1_p8a`。

## 用户方向评估

- “P3 结果直接放到 packed 64x64 对角位置”是正确方向；当前逻辑上已经这么做，但实现通过 compact->packed identity-MMAD 桥完成，后续应改成 P7-style packed landing，减少桥接搬运。
- “开始加载 64x64 或只加载必要 lower blocks 到 L1”是正确方向；这能去掉当前 16->32 / 32->64 compose 中 lower-left 从 GM on-demand load 的边界。
- “AIC 直接把最终 inverse 以 BF16 写 GM”可行，但需要先保证 packed L1 中 upper/tail 为 0，并修改 AIC/AIV 同步，让 final RHS 等 AIC 写 `attnWsGm_` 完成。
- “进一步融合 final RHS”是更大阶段，需要等 inverse residency 和 direct BF16 writeback 都通过后再做；涉及 `gBK/vBeta` 生成时序、D 维 tiling、L1 容量和 final matmul 类型。

## Phase 1: P8 lower-left packed L1 preload

变更范围：

- Active Stage1:
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- 新增 `LoadP8LowerLeftToPackedL1()`：
  - 在 `C64InverseBy16x16Neumann()` 中，调用 compose 前把 lower-left block 从 `INVInitHalfWsGm_` 预加载到 `aicHalfL1P3Result_` packed 64x64 对应位置。
- 删除 active compose 内部 lower-left GM on-demand load：
  - `P8Compose32From16SingleDirect()`
  - `P8Compose64From32Direct32()`
- 未改：
  - P3 diagonal 求逆公式。
  - P8 lower-triangular compose 公式。
  - final half GM writeback。
  - AIV half->BF16 `attnWsGm_`。
  - final RHS `AICProcess(attnWsGm_, ...)`。

验证：

- build PASS:
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=3.173828e-02`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=4.882812e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.747369e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.172006e-03`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=6.738281e-02`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.747369e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.137391e-03`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=6.738281e-02`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
- 当前 64-aligned compare set PASS:
  - `16 passed in 6.20s`

性能：

- 输出：
  - `tests/pytest/prof_out/p3l1_p8a_phase1_lower_l1_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/p3l1_p8a_phase1_lower_l1_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 670.940 | 1109.900 | 1.654 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1653.220 | 4650.420 | 2.813 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5051.640 | 16363.220 | 3.239 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24375.340 | 79850.940 | 3.276 | normal |

结论：

- Phase 1 target path correctness PASS。
- performance 没有收益；相对早上的 `p3l1_p8a_current_baseline_20260510`，neumann 绝对耗时略慢。
- 这说明仅把 lower-left GM load 前移到 compose 前，不足以消掉主要性能瓶颈；后续仍应继续去掉 P3 diagonal 中间 GM 写回和 packed/compact identity-MMAD 桥。

## Phase 2: defer P3 diagonal half GM writeback

变更范围：

- Active Stage1:
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- 删除 `Neumann16InverseBlock()` 内每个 `16x16` diagonal inverse 的立即 half GM 写回。
- `C64InverseBy16x16Neumann()` 保持 P8 compose 使用 packed L1 resident 数据，最后只调用一次 `StoreP3DiagonalResultsToGm(paraIdx)` 写回 diagonal half GM。
- 未改：
  - P8 lower-triangular compose 公式。
  - P8 compose 结果 half GM 写回。
  - AIV half->BF16 `attnWsGm_`。
  - final RHS `AICProcess(attnWsGm_, ...)`。

验证：

- build PASS:
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=3.173828e-02`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=4.882812e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.747369e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.172006e-03`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=6.738281e-02`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.747369e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.137391e-03`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=6.738281e-02`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
- 当前 64-aligned compare set PASS:
  - `16 passed in 6.16s`

性能：

- 输出：
  - `tests/pytest/prof_out/p3l1_p8a_phase2_defer_diag_gm_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/p3l1_p8a_phase2_defer_diag_gm_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 657.860 | 1129.620 | 1.717 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1664.260 | 4741.440 | 2.849 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5033.300 | 16783.600 | 3.335 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24378.520 | 81284.340 | 3.334 | normal |

结论：

- Phase 2 target path correctness PASS。
- performance 没有收益；相比 Phase 1 和当前 baseline，neumann 绝对耗时整体仍偏慢。
- 仅推迟 P3 diagonal half GM 写回不足以减少主要瓶颈；后续继续按计划削减 compact->packed identity-MMAD bridge。

## Phase 3: P3 packed landing

变更范围：

- Active Stage1:
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- 新增：
  - `HalfL1InvAddBlockToPacked()`
  - `HalfL1NeumannCompactBlockToPacked()`
- `Neumann16InverseBlock()` 最后一轮 `inv + inv @ power` 直接从 L0C Fixpipe 到 `aicHalfL1P3Result_` packed 64x64 对角位置。
- P3 diagonal 路径不再调用 `CopyHalfL1CompactBlock(aicHalfL1TileInv_, ...)`。
- 未改：
  - P8 lower-triangular compose 公式。
  - P8 compose 结果 half GM 写回。
  - AIV half->BF16 `attnWsGm_`。
  - final RHS `AICProcess(attnWsGm_, ...)`。

验证：

- build PASS:
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=3.173828e-02`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=4.882812e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.747369e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.172006e-03`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=6.738281e-02`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.747369e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.137391e-03`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=6.738281e-02`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
- 当前 64-aligned compare set PASS:
  - `16 passed in 6.25s`

性能：

- 输出：
  - `tests/pytest/prof_out/p3l1_p8a_phase3_packed_landing_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/p3l1_p8a_phase3_packed_landing_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 624.800 | 1130.780 | 1.810 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1640.100 | 4684.680 | 2.856 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5061.200 | 16542.140 | 3.268 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24441.040 | 80860.060 | 3.308 | normal |

结论：

- Phase 3 target path correctness PASS。
- performance 仍不是收益；P3 packed landing 本身没有改变主要耗时。
- 下一步按计划进入 Phase 4a：AIC direct BF16 writeback 到 `attnWsGm_`，final RHS 仍保持 BF16 GM 消费。

## Phase 4a: AIC direct BF16 GM writeback

变更范围：

- Active Stage1:
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- 新增 / 启用：
  - `CopyHalfL1MatrixToGmBf16ByMmad()`
  - `CopyPackedHalfL1BlockToGmBf16()`
  - `StorePackedL1InverseToBf16Attn()`
- `C64InverseBy16x16Neumann()` 在 P8 compose 后由 AIC 从 `aicHalfL1P3Result_` packed L1 直接写 BF16 `attnWsGm_`。
- AIV 侧原 `CopyInvHalfToBf16Attn(i)` 保留为注释，不再执行 half GM -> AIV BF16 cast。
- final RHS 未改：
  - `AICProcess(attnWsGm_, gBKWsGm_, outKCumdecayGm_, ...)`
  - `AICProcess(attnWsGm_, vBetaWsGm_, outVInnerGm_, ...)`

失败与修复：

- 初始 full `64x64` BF16 GM writeback 失败 `S=64`。
- 改为 `32x32` block writeback 后误差降低，但仍失败。
- AIC 从 half GM full block 再写 BF16 GM 也失败。
- 最终通过方案：
  - 按 `16x16` block 写 BF16 GM。
  - `colBase <= rowBase` 的 lower-triangular block 从 packed L1 写出。
  - upper-triangular `16x16` block 显式写 0，避免旧值 / 无效块污染 final RHS。

验证：

- build PASS:
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.747369e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.172006e-03`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.747369e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.137391e-03`
- 当前 64-aligned compare set:
  - 首次全量 compare 有一个边界波动：`varlen_b3_64_128_256`，`out max_abs=9.664921e-02`，state PASS。
  - 该 case 单独 rerun PASS：`out max_abs=5.614135e-02`。
  - 全量 rerun PASS：`16 passed in 6.16s`。

性能：

- 输出：
  - `tests/pytest/prof_out/p3l1_p8a_phase4a_aic_bf16_gm_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/p3l1_p8a_phase4a_aic_bf16_gm_20260510/performance_raw.csv`

| case | neumann/normal | faster |
|---|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 1.648 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 2.783 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 3.293 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 3.338 | normal |

结论：

- Phase 4a target path correctness PASS。
- AIC direct BF16 GM writeback 可作为通过边界，但仍不是 full fuse：final RHS 仍读 BF16 GM。
- performance 仍没有收益；后续 Phase 4b 只作为 Phase 5 RHS fusion 输入准备，不能描述为完成 fused RHS。

## Phase 4b: resident L1 inverse bridge

变更范围：

- Active Stage1:
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- 新增 / 启用：
  - `aicHalfL1ResidentInv_`
  - `StorePackedL1InverseToResidentL1()`
- `C64InverseBy16x16Neumann()` 现在同时执行：
  - `StoreP3DiagonalResultsToGm(paraIdx)`
  - `StorePackedL1InverseToBf16Attn(paraIdx)`
  - `StorePackedL1InverseToResidentL1(paraIdx)`
- `RestoreRawMmadState()` / `RestoreStage1MatmulLocalBuffers()` 后移到 inverse loop 后、RHS matmul 前。
- 未改：
  - Phase 4a 的 BF16 GM bridge 仍保留。
  - final RHS 仍保持 `AICProcess(attnWsGm_, gBKWsGm_, outKCumdecayGm_, ...)` 和 `AICProcess(attnWsGm_, vBetaWsGm_, outVInnerGm_, ...)`。

验证：

- build PASS:
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64 / 256 / 1024` smoke PASS。
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.16s`

性能：

- 输出：
  - `tests/pytest/prof_out/p3l1_p8a_phase4b_resident_l1_bridge_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/p3l1_p8a_phase4b_resident_l1_bridge_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 646.560 | 1726.360 | 2.670 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1641.560 | 8586.420 | 5.231 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5038.720 | 30883.640 | 6.129 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24241.860 | 150779.160 | 6.220 | normal |

结论：

- Phase 4b target path correctness PASS。
- resident copy 是 Phase 5 的输入准备，不是性能收益；相对 Phase 4a 退化明显。
- Phase 5 必须单独验证 resident inverse 是否能直接消费 RHS，不能把 Phase 4b 描述成 final RHS fused。

## Phase 5: final RHS fusion target PASS

目标：

- final RHS 不再读 BF16 GM `attnWsGm_`。
- AIC 在 Stage1 内直接消费 `aicHalfL1ResidentInv_` resident inverse。
- RHS operand 使用 half `gBK/vBeta` workspace，输出仍写 BF16 GM `outKCumdecayGm_` / `outVInnerGm_`。

完成：

- AIV 侧新增 half `gBKHalfWsGm_` / `vBetaHalfWsGm_` 写出，同时保留旧 BF16 `gBKWsGm_` / `vBetaWsGm_`。
- AIC RHS 从旧：
  - `AICProcess(attnWsGm_, gBKWsGm_, outKCumdecayGm_, ...)`
  - `AICProcess(attnWsGm_, vBetaWsGm_, outVInnerGm_, ...)`
  改为：
  - `FusedRhsAICProcess(i, gBKHalfWsGm_[...], outKCumdecayGm_[...])`
  - `FusedRhsAICProcess(i, vBetaHalfWsGm_[...], outVInnerGm_[...])`
- 修复 `StorePackedL1InverseToResidentL1()`：
  - 旧版整块 identity-MMAD copy 不能正确形成 resident L1 lower-triangular layout。
  - 当前按 lower-triangular `16x16` block 从 `aicHalfL1P3Result_` 复制到 `aicHalfL1ResidentInv_`，upper triangle 保持 0。
- 修复 raw-MMAD RHS rectangular output：
  - `tileCols = min(64, shape.n - colBase)`。
  - 新增 rectangular load/accumulate，B operand 按 `kSize x nSize` copy，支持 `dk/dv=32` 边界。

验证：

- build PASS:
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- 当前 64-aligned compare set PASS:
  - `16 passed in 6.16s`

性能：

- 输出：
  - `tests/pytest/prof_out/p3l1_p8a_phase5_fused_rhs_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/p3l1_p8a_phase5_fused_rhs_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 645.740 | 1219.660 | 1.889 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1662.000 | 5106.940 | 3.073 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5031.180 | 21165.860 | 4.207 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24268.560 | 103557.700 | 4.267 | normal |

结论：

- Phase 5 target full RHS fusion correctness PASS。
- 性能仍不是收益；后续 Phase 6 只能在 Phase 5 PASS 回退点上单独替换 P3/P7 diagonal 求解路径。

## Phase 6a: P7-style single 16x16 direct solve

目标：

- 只替换 P3 diagonal 单个 `16x16` 求解路径。
- 不同时修改 P8 compose、resident copy、final RHS fusion。
- 当前阶段仍不是 group-packed 求解；Phase 6b 单独做。

完成：

- `Neumann16InverseBlock()` 不再走旧 compact 求解链路。
- 当前流程：
  - 清空 `aicHalfL1TileInv_` / `aicHalfL1TilePower_` / `aicHalfL1Scratch_`。
  - `LoadGmHalfDiagonalBlockToPackedL1(paraIdx, base, 0, INVERSE_SHAPE)`。
  - `P7HalfL1DiagonalNeumannBlock(INVERSE_SHAPE)`。
  - `CopyHalfL1PackedBlockToCompact(..., aicHalfL1P3Result_, ..., base, base, INVERSE_SHAPE)` 落到 packed 64x64 对角位置。
- 已保存 precision baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_phase6a_baseline.h`

验证：

- build PASS:
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- 当前 64-aligned compare set PASS:
  - `16 passed in 6.14s`

性能：

- 输出：
  - `tests/pytest/prof_out/p3l1_p8a_phase6a_p7_single16_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/p3l1_p8a_phase6a_p7_single16_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 663.240 | 1272.240 | 1.918 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1656.640 | 5444.260 | 3.286 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5067.000 | 22330.760 | 4.407 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24259.440 | 109692.100 | 4.522 | normal |

结论：

- Phase 6a target path correctness PASS。
- 性能仍不是收益；相对 Phase 5 略退化。
- Phase 6b 已在后续单独通过 build -> install -> `S=64 / 256 / 1024` -> compare -> performance gate。

## Phase 6b: group load + two 16 P7 solves target PASS

目标：

- 去掉 P3 外层逐块旧入口，把 diagonal block 的加载统一收敛到 group-packed 入口。
- 保持已验证的 `16x16` P7 direct solve 作为求解内核，不启用失败的 strict `32x32/64x64` group-MMAD diagonal solver。

失败尝试记录：

- strict full `32x32` group solve 失败：
  - `S=32`：out max_abs `1.022196e+00`，state max_abs `4.903520e-01`。
- one-shot `64x64` packed diagonal-only solve 失败：
  - `S=16`：out max_abs `9.514923e-02`，state max_abs `1.818329e-01`。
  - `S=64`：out max_abs `1.059731e-01`，state max_abs `5.440532e-03`。
- group load + two `16x16` solve 在 cleanup 前 diagnostic `S=32` PASS：
  - out max_abs `3.173694e-03`，state max_abs `9.614527e-04`。

完成：

- `C64InverseBy16x16Neumann()` 调用：
  - `P7PackedDiagonalNeumannGroupToP3Result(paraIdx, 0, C64_BLOCK32)`。
  - valid 时对 bottom group 调用 `P7PackedDiagonalNeumannGroupToP3Result(paraIdx, C64_BLOCK32, C64_BLOCK32)`。
  - `validLen > C64_BLOCK32 && <= C64_BLOCK32 + INVERSE_SHAPE` 时保留 `Neumann16InverseBlock(paraIdx, C64_BLOCK32)`。
- `P7PackedDiagonalNeumannGroupToP3Result()` 保持 group load，然后执行两个 proven `16x16` P7 direct solves，并落到 `aicHalfL1P3Result_`。
- 清理失败 one-shot 64 helper：
  - `HalfL1InvAddDiagonalOnlyPackedBlockToPacked()` 已删除。
  - `P7HalfL1DiagonalNeumannBlockToPacked()` 已删除。
  - `P7PackedDiagonalNeumann64ToP3Result()` 已删除。
- `RawMmadHalfL1DiagonalOnlyAccumulateTile()` 的 zero path 恢复为 `blockSize * blockSize` compact L0 load，不再使用 `chunkSize_ * chunkSize_` full-tile zero。

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.51s`

性能：

- 输出：
  - `tests/pytest/prof_out/p3l1_p8a_phase6b_group_load_two16_clean_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/p3l1_p8a_phase6b_group_load_two16_clean_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 653.520 | 1243.280 | 1.902 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1650.620 | 5415.560 | 3.281 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5083.560 | 22518.000 | 4.430 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24242.760 | 110376.060 | 4.553 | normal |

baseline 保存：

- 已保存 clean target PASS active Stage1：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_phase6b_baseline.h`
- 内容与 active Stage1 一致。

结论：

- Phase 6b target path correctness PASS，且所有 planned phases 已完成。
- Phase 6b 不是性能收益；normal 仍更快。
- strict group diagonal-only raw-MMAD body 仍是后续风险点，不得把失败尝试描述为 baseline。

## Cleanup: active probe / tilingdata / workspace 收口

目标：

- 删除 active `p3l1_p8a` 中不再使用的 probe、旧 fallback 和旧 matmul tilingdata。
- 保留 Phase 6b 已通过 target path，不改 Stage2/Stage3/QK/gBK/vBeta/final RHS 语义。

完成：

- active Stage1：
  - 删除旧 `StageOneFp16*` wrapper 和 `mmFp16L1*` 成员。
  - 删除 scalar packed probe、旧 GM direct compose fallback、旧 AIV half->BF16 cast 注释块。
  - 删除旧 `INVWsGm_` FP32 final inverse workspace 绑定。
- top-level kernel：
  - `stageOneOp_` 只接收 `stage1MT_`。
  - 只保留 `stage1MT_ / stage2MT_ / stage3MT_`。
- tiling：
  - `ChunkGatedDeltaRuleNeumannTilingData` 只保留 `matmulTilingBf16`。
  - `DoMatmulTiling()` 只生成 `matmulTilingBf16`。
  - host workspace 中 Neumann inverse 部分收口为 FP16 `L_pow` + FP16 `INVInit` + FP32 `tmp`。
- baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_phase6b_baseline.h`
  - 与 active Stage1 仅 `\file` 和 include guard 不同。
- 新增文档：
  - `docs/2026-05-10/P3L1_P8A_PHASE6B_INVERSE_FLOW.md`

静态扫描：

- active 文件、top-level kernel、tilingdata、host tiling、Phase 6b baseline 中无以下旧 active 残留：
  - `StageOneFp16`
  - `matmulTilingFp32C16`
  - `matmulTilingFp16`
  - `Probe`
  - `INVWsGm_`
  - old P8 GM direct fallback helper 名称
- 全 repo 扫描仍会在历史 baseline / 非 active 实验文件 / skill 文档中命中旧名字；这些不属于当前 selector active path。

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.14s`

## TilingData field rename and performance retest

背景：

- cleanup 后 `ChunkGatedDeltaRuleNeumannTilingData` 只保留一个 matmul tiling 字段。
- 该字段原名 `matmulTilingFp32` 是历史遗留名：
  - HEAD 原始版本中它确实按 `DT_FLOAT / DT_FLOAT / DT_FLOAT` 生成，旁边还有多个 `Fp32C16/Fp32C32/Fp16*` micro tiling 字段。
  - 当前 cleanup 后 host `DoMatmulTiling()` 已改成 `DT_BFLOAT16 / DT_BFLOAT16 / DT_BFLOAT16`，Stage1/2/3 的 high-level `MatmulType` 也是 BF16 GM matmul。
  - 当前 FP16 只用于 Stage1 inverse 的 raw-MMAD 内部 half L1/L0/GM workspace，不再通过 `TCubeTiling` 字段驱动。

变更：

- `matmulTilingFp32` 重命名为 `matmulTilingBf16`。
- 注释改为 `BF16 GM matmul: bf16 x bf16 -> bf16`。
- 同步更新 active/top-level/baseline/非 active 实验文件中对公共 `TilingData` 字段的引用，避免以后切 selector 编译失败。

静态扫描：

- `op_host` / `op_kernel` 中不再命中 `matmulTilingFp32`。
- 当前仅命中 `matmulTilingBf16`。

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.21s`

性能：

- 输出：
  - `tests/pytest/prof_out/p3l1_p8a_tiling_bf16_rename_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/p3l1_p8a_tiling_bf16_rename_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 651.500 | 1147.760 | 1.762 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1682.700 | 5220.240 | 3.102 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5063.300 | 21901.740 | 4.326 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24376.640 | 107070.300 | 4.392 | normal |

结论：

- `matmulTilingFp32` 是原始历史命名；当前 dtype 已不是 FP32，应按 BF16 命名。
- 不应改成 FP16：当前公共 high-level matmul 仍是 BF16 GM，FP16 只存在于 inverse raw-MMAD 内部数据流。
- 本次是命名/注释收口，不是计算优化；性能仍是 normal 更快。

## Packed64 diagonal inverse Phase 0: 静态资源与别名确认

目标：

- 从 `todo.md` 的 Phase 0 开始，只确认 packed64 diagonal inverse 新资源模型，不改数学。
- 本阶段只要求 build gate，不跑 smoke / compare / performance。

静态确认：

- Active selector 仍是 `p3l1_p8a`：
  - `op_kernel/selected/stage1_selected.h`
  - `op_kernel/selected/stage1_selected_config.h`
- Active Stage1 仍是：
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- host / selector chunk size 仍为 `64`。
- `AIC_LOW_L1_BYTES = AIC_LOW_L1_TILE_ELEMS * 12 * sizeof(half)`，其中一个 packed64 half tile 为 `64 * 64` half。
- L1/TSCM 别名确认：
  - `result64 = aicHalfL1P3Result_`，当前已经承载最终 packed 64x64 inverse。
  - `diagPower64 = aicHalfL1TilePower_`，当前用于 diagonal power / compose output scratch，Phase 1 可转为 packed64 diagonal power 容器。
  - `lowerWork64 = aicHalfL1Scratch_`，当前用于 compose tmp / copy scratch，Phase 2 可转为 lower-left work tile。
  - `aicHalfL1TileInv_` 暂留 compact fallback/debug work，不作为新主线长期结果。
  - `aicHalfL1ResidentInv_` 仍占 `MAX_PARALLEL_NUM` 个 packed64 tile，不在 Phase 0/1 调整，避免影响 fused RHS。
- L0C 静态分区确认：
  - `AIC_RAW_L0_C_BYTES = 64KB`。
  - 一个 `64x64 FP32 C` tile 为 `64 * 64 * sizeof(float) = 16KB`。
  - 可逻辑分为 4 个分区：`cInv/cPower/cComposeTmp/cFree`。
  - 当前 helper 的 `Mmad()` / `Fixpipe()` 已按 `LocalTensor<float> cTile` 入参传递，后续可用 `aicL0C_[offset]` 传入带 offset 的 C tile。
- 当前 P8 compose 仍从 `aicHalfL1P3Result_` 同时读取 lower / topInv / bottomInv；Phase 2 才改 lower-left 输入到 `lowerWork64`。

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`

结论：

- Phase 0 PASS。
- 未改 kernel 数学，未进入 smoke / compare / performance。
- 下一步可进入 Phase 1：packed64 容器零初始化 + diagonal GM 定点搬入，仍沿用 proven `16x16` solve。

## Packed64 diagonal inverse Phase 1: P7-style packed64 初始化和 GM 定点搬入

目标：

- 把 diagonal inverse 入口改成 packed64 容器模型：
  - `result64 = aicHalfL1P3Result_`
  - `diagPower64 = aicHalfL1TilePower_`
  - `lowerWork64 = aicHalfL1Scratch_`
- 每个 para 先清零 `result64 / diagPower64 / lowerWork64`，再从 GM 定点搬入四个 diagonal `16x16` block。
- 本阶段不启用 packed64 one-shot 求逆，仍沿用 proven `16x16` diagonal Neumann solve。

实现：

- 新增 packed64 diagonal 输入搬运：
  - `LoadPacked64DiagonalInputBlock()`
  - `LoadPacked64DiagonalInputsToL1()`
- 新增 `P7SolvePacked64DiagonalBlockToResult()`：
  - 从 `result64[(base,base)]` 抽取 compact `16x16` inv 到 `aicHalfL1TileInv_`。
  - 从 `diagPower64[(base,base)]` 抽取 compact `16x16` power 到 `lowerWork64` 前半段。
  - 使用 `lowerWork64` 后半段作为 compact scratch，调用 proven `16x16` solve。
  - 将结果写回 `result64[(base,base)]`。
- `P7HalfL1DiagonalNeumannBlock()` 保持旧入口，同时抽出 `P7HalfL1DiagonalNeumannBlockWithBuffers()`，避免把 `diagPower64` 的 packed64 容器当成 compact power 直接覆盖。
- `C64InverseBy16x16Neumann()` 入口由旧 group-load 流程改为：
  - `LoadPacked64DiagonalInputsToL1()`
  - 按有效长度逐个 `P7SolvePacked64DiagonalBlockToResult()`
  - P8 compose / BF16 GM 写回 / resident L1 / fused RHS 保持旧语义。

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.16s`

性能：

- 输出：
  - `tests/pytest/prof_out/packed64_phase1_p7_style_diag_load_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_phase1_p7_style_diag_load_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 637.200 | 1191.720 | 1.870 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1636.200 | 5326.280 | 3.255 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5053.400 | 22205.920 | 4.394 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24234.940 | 108242.120 | 4.466 | normal |

baseline 保存：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase1_baseline.h`
- 与 active Stage1 仅 `\file` 和 include guard 不同。

结论：

- Phase 1 target path correctness PASS。
- 本阶段证明 packed64 容器零初始化和 diagonal GM 定点搬入可作为后续 Phase 2/3 起点。
- performance 仍不是收益；normal 更快。
- 尚未启用 packed64 one-shot diagonal solver，不能描述为 Phase 3 完成。

## Packed64 diagonal inverse Phase 2: lower-left 输入单独搬入 lowerWork64

目标：

- 让 compose 的 lower-left 输入和 diagonal inverse result 分离：
  - top/bottom inverse 继续读 `result64 = aicHalfL1P3Result_`。
  - lower-left blocks 改为读 `lowerWork64 = aicHalfL1Scratch_`。
- 本阶段只改 P8 lower-left 输入来源，不改 diagonal solve，不改 compose 公式，不改 compose output / BF16 GM / resident L1 / fused RHS。

实现：

- 新增 `LoadP8LowerLeftToLowerWork64()`：
  - `INVInitHalfWsGm_ -> lowerWork64[(16,0)]`。
  - `INVInitHalfWsGm_ -> lowerWork64[(48,32)]`。
  - `INVInitHalfWsGm_ -> lowerWork64[(32,0)]`。
- `P8Compose32From16SingleDirect()` / `P8Compose64From32Direct32()` 的 first MMAD 改为：
  - `lowerWork64 @ topInv`。
  - 其中 lower 从 `aicHalfL1Scratch_` packed64 对应 offset 读。
  - top/bottom inverse 仍从 `aicHalfL1P3Result_` 读。
- `P8` compose tmp 继续落到 `aicHalfL1TileInv_`。
- `P8` compose output 继续先落到 `aicHalfL1TilePower_`，再复制回 `aicHalfL1P3Result_`。

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=6.722955e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.09s`

性能：

- 输出：
  - `tests/pytest/prof_out/packed64_phase2_lowerwork64_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_phase2_lowerwork64_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 654.120 | 1193.700 | 1.825 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1680.500 | 5286.740 | 3.146 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5052.940 | 21956.540 | 4.345 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24390.500 | 108854.580 | 4.463 | normal |

baseline 保存：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase2_baseline.h`
- 与 active Stage1 仅 `\file` 和 include guard 不同。

结论：

- Phase 2 target path correctness PASS。
- performance 仍不是收益；相比 Phase 1，`S=64` 略好但大 S 基本持平或略慢。
- 当前 active 仍是 packed64 容器 + proven `16x16` solve；尚未启用 packed64 one-shot diagonal solver。
- 下一步进入 Phase 3 时，只改 packed64 diagonal-only solve，不同时改 compose output 或 RHS。

## Packed64 diagonal inverse Phase 3: strict packed64 one-shot 跳过

背景：

- strict packed64 diagonal-only / one-shot group solve 仍未打通。
- 用户明确允许：如果打包太难，先跳过 Phase 3，先做 Phase 4~7。
- 因此本轮没有把 strict packed64 one-shot diagonal solver 描述为完成，也没有保存 Phase 3 baseline。

当前判断：

- active path 继续使用 proven `16x16` diagonal solve。
- 后续 Phase 4~7 的完成边界均是在该 proven `16x16` diagonal solve 主线上成立。
- 若后续继续追 strict packed64 one-shot，需要单独开新阶段并重新走 build/install、`S=64/256/1024`、compare、performance gate。

## Packed64 diagonal inverse Phase 4: L0C accum on proven 16x16 path

目标：

- 按用户指令跳过 strict packed64 one-shot 后，先在 proven `16x16` diagonal solve path 内减少 repeated `inv @ I`。
- 只改 diagonal inverse 内部 L0C 使用，不改 P8 compose、BF16 GM 写回、resident L1 copy 或 final RHS 数学语义。

完成：

- `P7HalfL1DiagonalNeumannBlockWithBuffers()` 改为使用两个 L0C 逻辑区：
  - `cInv = aicL0C_`
  - `cPower = aicL0C_[chunkSize_ * chunkSize_]`
- `inv @ I` 初始化只做一次，后续按 power update 累积。

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64 / 256 / 1024` smoke PASS。
- 当前 64-aligned compare set PASS。

性能：

- 输出：
  - `tests/pytest/prof_out/packed64_phase4_l0c_accum_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_phase4_l0c_accum_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 672.180 | 1195.060 | 1.778 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1663.200 | 5278.000 | 3.173 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5072.940 | 22038.980 | 4.344 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24226.560 | 107300.500 | 4.429 | normal |

baseline 保存：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase4_baseline.h`

结论：

- Phase 4 correctness PASS。
- 性能仍不是收益；normal 更快。

## Packed64 diagonal inverse Phase 5: lower-left preload on proven 16x16 path

目标：

- 继续在 proven `16x16` diagonal solve 主线上推进，减少 P8 compose 前后的 on-demand lower-left load 暴露。
- 不改 diagonal solve 数学，不改 final RHS 数学语义。

完成：

- compact diagonal scratch 从 `aicHalfL1Scratch_[256]` 改到 `aicHalfL1TileInv_[256]`。
- 新增 `PreloadP8LowerLeftBlocksToLowerWork64(paraIdx)`。
- lower-left blocks 在 diagonal solve 之前/期间准备到 `lowerWork64`，P8 compose 不再做 on-demand lower-left load。

验证：

- build/install PASS。
- `S=64 / 256 / 1024` smoke PASS。
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.13s`

性能：

- 输出：
  - `tests/pytest/prof_out/packed64_phase5_lower_preload_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_phase5_lower_preload_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 639.780 | 1196.000 | 1.869 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1664.260 | 5216.320 | 3.134 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5054.160 | 21865.640 | 4.326 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24339.940 | 106934.520 | 4.393 | normal |

baseline 保存：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase5_baseline.h`

结论：

- Phase 5 correctness PASS。
- 性能仍不是收益；normal 更快。

## Packed64 diagonal inverse Phase 6: compose output direct result64

目标：

- 去掉 P8 compose output 的 compact scratch 暴露，直接 Fixpipe 回 `result64 = aicHalfL1P3Result_`。
- 不改变 lower-triangular compose 公式，不改变 final RHS 数学语义。

完成：

- `P8Compose32From16SingleDirect()`：
  - `16+16 -> 32` output 直接写 `result64[(16,0)]` / `result64[(48,32)]`。
- `P8Compose64From32Direct32()`：
  - `32+32 -> 64` output 直接写 `result64[(32,0)]`。
- 删除同一 L0C 结果同时写 packed L1 和 half GM 的尝试；该尝试曾导致 `S=64` timeout。
- Phase 6 active 只写 packed L1，half GM diagonal 暴露留到 Phase 7 处理。

验证：

- build/install PASS。
- `S=64 / 256 / 1024` smoke PASS。
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.23s`

性能：

- 输出：
  - `tests/pytest/prof_out/packed64_phase6_compose_direct_result_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_phase6_compose_direct_result_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 634.560 | 991.420 | 1.562 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1666.880 | 3768.140 | 2.261 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5068.340 | 16669.820 | 3.289 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24419.380 | 81385.580 | 3.333 | normal |

baseline 保存：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase6_baseline.h`

结论：

- Phase 6 correctness PASS。
- 仍不是 performance win，但相对 Phase 5 明显改善。

## Packed64 diagonal inverse Phase 7: skip diagonal half GM writeback

目标：

- Phase 6 后，P8 compose 的最终 inverse 已在 `result64 = aicHalfL1P3Result_`。
- BF16 `attnWsGm_` 写回和 resident L1 copy 都直接消费 `result64`。
- 因此移除不再被 active compose/RHS 依赖的 diagonal half GM writeback。

完成：

- `C64InverseBy16x16Neumann()` 不再调用：
  - `StoreP3DiagonalResultsToGm(paraIdx)`
- 保留：
  - `StorePackedL1InverseToBf16Attn(paraIdx)`
  - `StorePackedL1InverseToResidentL1(paraIdx)`
- 未改 Stage2 / Stage3 / QK / `gBK/vBeta` 生成语义 / final RHS 数学语义。

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.14s`

性能：

- 输出：
  - `tests/pytest/prof_out/packed64_phase7_skip_diag_half_gm_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_phase7_skip_diag_half_gm_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 632.920 | 931.800 | 1.472 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1661.680 | 3556.340 | 2.140 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5058.740 | 15692.420 | 3.102 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24284.740 | 76904.560 | 3.167 | normal |

baseline 保存：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase7_baseline.h`
- 内容与 active Stage1 一致。

结论：

- Phase 7 correctness PASS。
- normal 仍更快，但 Phase 7 是当前 packed64 progressive 主线的最新安全回退点。
- 当时 strict packed64 one-shot diagonal solve 仍未完成；本轮按用户指令跳过 Phase 3 后完成 Phase 4~7。
- 后续已完成 strict one-shot placement 定位，但因大 shape 性能退化未采纳为 active，见本文末尾 `Strict packed64 one-shot diagonal solve placement localization`。

## Packed64 diagonal inverse Phase 8: AIC/AIV sync overlap

目标：

- 在不改变 Stage1 final RHS 数学语义的前提下，减少 AIV 等 AIC inverse / BF16 写回的暴露时间。
- 只调整 Stage1 inverse/RHS 附近的跨核同步顺序，不改 Stage2 / Stage3 / QK / `gBK/vBeta` 生成语义。

尝试与取舍：

- Phase 8a：删除 `StorePackedL1InverseToBf16Attn(paraIdx)`。
  - build/install PASS，`S=64/256` PASS。
  - `S=1024` FAIL，out/state 接近全 0。
  - 结论：BF16 `attnWsGm_` 写回仍不能删除，或该写回仍有必要副作用；已恢复。
- Phase 8b：AIV 发 `0x7` 后先做 `GBKCompute`，再等 AIC 的 `0xd`，最后发 `0x6`。
  - build/install PASS。
  - `S=64/256/1024` smoke PASS。
  - compare PASS：`16 passed in 6.25s`。
  - performance：neumann/normal 约 `1.360 / 1.974 / 2.816 / 2.868`。
- Phase 8c：`GBKCompute` 后先发 `0x6`，再等 `0xd`。
  - build/install PASS。
  - `S=64/256/1024` smoke PASS。
  - compare PASS：`16 passed in 6.14s`。
  - performance：neumann/normal 约 `1.397 / 1.975 / 2.798 / 2.851`。
  - 长序列略好于 8b，但 `S=64` 明显回退。
- Phase 8d：`GBKCompute` 后发 `0x6`，继续做 `VBetaCompute`、发 `0x5`、做 `QPrimeCompute`，最后在等待 `0xc` 前消费 `0xd`。
  - 这样 AIC 可以开始第一段 `attn @ k_cumdecay`，AIV 同时准备第二段 `v_beta` 和下一阶段 `q_prime`。
  - `0xd` 仍在下一轮 workspace 复用前消费，保持 AIC inverse / BF16 写回完成边界。

最终 active：

- 保留 Phase 8d。
- AIV 当前顺序：
  - `GBKCompute(...)`
  - `CrossCoreSetFlag(0x6)`
  - `VBetaCompute(...)`
  - `CrossCoreSetFlag(0x5)`
  - `QPrimeCompute(...)`
  - `CrossCoreWaitFlag(0xd)`
  - `CrossCoreWaitFlag(0xc)`

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.24s`

性能：

- 输出：
  - `tests/pytest/prof_out/packed64_phase8d_defer_wait0d_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_phase8d_defer_wait0d_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 677.440 | 893.920 | 1.320 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1664.220 | 3310.080 | 1.989 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5084.800 | 14208.820 | 2.794 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24353.900 | 69045.740 | 2.835 | normal |

baseline 保存：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase8_sync_overlap_baseline.h`
- 内容与 active Stage1 一致。

结论：

- Phase 8 correctness PASS。
- 相对 Phase 7，四档比例从 `1.472 / 2.140 / 3.102 / 3.167` 改善到 `1.320 / 1.989 / 2.794 / 2.835`。
- normal 仍更快，主要剩余瓶颈仍在 diagonal inverse raw-MMAD/compose 的额外标量、MTE/Fixpipe 调度，以及尚未完成的 strict packed64 one-shot diagonal solve。

## Packed64 diagonal inverse Phase 9 attempt: skip BF16 upper zero writeback

目标：

- 在保留 lower-triangular BF16 `attnWsGm_` 写回的前提下，跳过 upper-triangular zero blocks 写回，减少 `StorePackedL1InverseToBf16Attn()` 中的冗余 GM 写。

尝试：

- 移除 `StorePackedL1InverseToBf16Attn()` 内的 upper-triangle zero scratch 写回分支。
- 保留 lower-triangle BF16 block 写回。
- 保留 resident L1 copy 和 final RHS 数学语义。

验证结果：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS。
- `S=256` smoke PASS。
- `S=1024` smoke FAIL：
  - `CPU vs ChunkGDRNeumann out max_abs=1.118436e-01`，actual 接近 `9.9e-07`。
  - `CPU vs ChunkGDRNeumann state max_abs=1.557053e-01`，actual 接近 `-8.5e-07`。

处理：

- 该尝试已回滚，恢复：
  - `ZeroHalfL1(aicHalfL1Scratch_, INVERSE_SHAPE * INVERSE_SHAPE)`。
  - upper-triangle `CopyHalfL1MatrixToGmBf16ByMmad(... attnWsGm_[gmOffset] ...)`。
- 回滚后重新 build/install，并补跑 `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`

结论：

- BF16 upper-triangle zero writeback 或完整 BF16 writeback 的副作用仍被跨 chunk correctness 依赖。
- 不保存 Phase 9 baseline。
- 后续不要重复删除 BF16 写回或跳过 upper zero writeback，除非先重做 `attnWsGm_` 生命周期/清零边界设计。

## Packed64 diagonal inverse Phase 10: drop redundant 0xd wait/set

目标：

- 在 Phase 8d 同步流水基础上，删除已经被 `0xc` 覆盖的冗余跨核同步。
- 不改 final RHS 数学语义，不改 Stage2/Stage3/QK/`gBK/vBeta` 生成语义。

改动：

- 删除 AIC inverse 后的 `CrossCoreSetFlag<0x2, PIPE_FIX>(0xd)`。
- 删除 AIV 在 `CrossCoreWaitFlag(0xc)` 前的 `CrossCoreWaitFlag(0xd)`。
- 保留 `0xc`：AIC 在两段 `FusedRhsAICProcess()` 都完成后发出，表示共享 workspace 已读完，下一批 parallel chunk 可以复用。

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64 / 256 / 1024` smoke PASS。
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.15s`

性能：

- 输出：
  - `tests/pytest/prof_out/packed64_phase10_drop_wait0d_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_phase10_drop_wait0d_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 646.720 | 891.820 | 1.379 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1641.780 | 3284.520 | 2.001 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5086.680 | 14039.840 | 2.760 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24376.900 | 68838.660 | 2.824 | normal |

baseline 保存：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase10_drop_wait0d_baseline.h`
- 内容与 active Stage1 一致。

结论：

- Phase 10 correctness PASS。
- 相对 Phase 8/9 回退点，Phase 10 是同步冗余清理；Neumann 绝对耗时略有改善，但 ratio 受 normal 波动影响，不描述为稳定性能收益。

## Packed64 diagonal inverse Phase 11 attempt: skip RestoreRawMmadState

目标：

- 尝试删除 inverse 后、RHS 前的 `RestoreRawMmadState()`，因为 `FusedRhsAICProcess()` 入口会调用 `PrepareRawMmadState()`。

验证：

- build PASS。
- install PASS。
- `S=64 / 256 / 1024` smoke PASS。
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.13s`

性能：

- 输出：
  - `tests/pytest/prof_out/packed64_phase11_skip_restore_raw_state_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_phase11_skip_restore_raw_state_20260510/performance_raw.csv`

| case | normal us | neumann us | neumann/normal | faster |
|---|---:|---:|---:|---|
| `prefill_b8_s64_d64_h16_hkv4` | 668.200 | 895.540 | 1.340 | normal |
| `prefill_b8_s512_d64_h16_hkv4` | 1679.800 | 3322.000 | 1.978 | normal |
| `prefill_b8_s2048_d128_h16_hkv4` | 5060.040 | 14169.940 | 2.800 | normal |
| `prefill_b8_s10240_d128_h16_hkv4` | 24342.280 | 69272.900 | 2.846 | normal |

处理：

- 与 Phase 10 比，四档 neumann 绝对耗时均略慢：
  - `891.820 -> 895.540`
  - `3284.520 -> 3322.000`
  - `14039.840 -> 14169.940`
  - `68838.660 -> 69272.900`
- 已恢复 `RestoreRawMmadState()`，active 回到 Phase 10 baseline。
- 恢复后重新 build/install PASS，`S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`

结论：

- Phase 11 correctness PASS 但 performance 退化，不保存 baseline。
- 后续不要继续按“删除 raw-MMAD 状态恢复”推进，除非有新的 profile 证据表明状态恢复本身是瓶颈。

## Packed64 diagonal inverse Phase 12 attempt: skip RHS full-tile zero-init

目标：

- 尝试删除 `FusedRhsAICProcess()` 内完整 `64` 列 RHS tile 的 `aicHalfL1TilePower_` 清零。
- 初始判断是完整 tile 会被 `LoadGmHalfBlockToPackedHalfL1WithStride()` 覆盖，尾列 `tileCols < 64` 仍保留清零以保护 padding。

改动：

- 将：
  - `ZeroHalfL1(aicHalfL1TilePower_, chunkSize_ * chunkSize_)`
- 改为：
  - `if (tileCols < chunkSize_) ZeroHalfL1(...)`

验证结果：

- build PASS。
- install PASS。
- `S=64` smoke PASS。
- `S=256` smoke PASS。
- `S=1024` smoke FAIL：
  - `CPU vs ChunkGDRNeumann out max_abs=1.109401e-01`，actual 接近 `-2.55e-05`。
  - `CPU vs ChunkGDRNeumann state max_abs=1.557053e-01`，actual 接近 `-8.53e-07`。

处理：

- 已恢复每个 RHS tile 前的：
  - `ZeroHalfL1(aicHalfL1TilePower_, chunkSize_ * chunkSize_)`
- 回滚后 build/install PASS。
- 回滚后 `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`

结论：

- RHS operand L1 zero-init 不能按“完整 tile 被覆盖”直接删除。
- 该清零很可能还承担 MTE1/raw-MMAD operand 可见性、padding/lifetime 或同步副作用。
- 不保存 Phase 12 baseline，不继续叠加任何基于该假设的优化。

## Strict packed64 one-shot diagonal solve placement localization

目标：

- 定位 strict packed64 one-shot diagonal solve 的三个边界：
  - L1->L0 full packed placement。
  - Fixpipe packed L1 offset。
  - identity placement 与 `initC/accumulateC` 顺序。

实现尝试：

- 在 `C64InverseBy16x16Neumann()` 中临时加入 `P7Packed64DiagonalNeumannOneShotToResult()`：
  - `result64 = aicHalfL1P3Result_` 作为 diagonal inverse/result。
  - `diagPower64 = aicHalfL1TilePower_` 作为 power 递推容器。
  - 构造 packed64 identity 后执行 full `64x64` diagonal-only raw-MMAD。
- 先试 guarded 版本：
  - `PACKED64_ONE_SHOT_RUNTIME_PARA_MASK = 0xF`
- 再移除 magic mask，仅保留 runtime para guard：
  - `validLenBatch_[paraIdx] >= chunkSize_ && paraIdx < paraNum_`

正确性结论：

- `stageOneParaNum = 4`，所以旧 `0xF` mask 不过滤真实 `paraIdx`。
- 无 mask para guard 版本 correctness gate PASS：
  - build PASS。
  - install PASS。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：
    - `16 passed in 6.15s`
- 这说明当前 strict one-shot 的 L1->L0 placement、Fixpipe offset、identity placement 对 64-aligned correctness 已足够正确。

性能：

- 输出：
  - `tests/pytest/prof_out/packed64_strict_oneshot_guarded_20260510/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_strict_oneshot_para_guard_20260510/performance_compare.csv`
- para guard 版本 neumann/normal：
  - `1.319 / 2.091 / 2.961 / 3.034`
- Phase 10 对照：
  - `tests/pytest/prof_out/packed64_phase10_drop_wait0d_20260510/performance_compare.csv`
  - `1.379 / 2.001 / 2.760 / 2.824`

处理：

- strict one-shot 在 `S=64` ratio 略好，但大 shape 相对 Phase 10 退化。
- 不保存 one-shot baseline，不作为 active 优化路径。
- active Stage1 已恢复为 Phase 10 baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase10_drop_wait0d_baseline.h`
- 恢复后验证：
  - active 与 Phase 10 baseline diff 为空。
  - `OneShot` / `ShouldUsePacked64` / `PACKED64_ONE_SHOT` 搜索无命中。
  - build PASS。
  - install PASS。
  - `S=1024` smoke PASS：
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
  - 当前 64-aligned compare set PASS：
    - `16 passed in 6.14s`

结论：

- strict packed64 one-shot 是 placement/offset/identity 的 correctness localization PASS，不是 active performance adoption。
- 后续如果继续追 one-shot，优化目标应集中在降低 full-64 identity 构造和 power/update 开销，而不是继续证明 placement。

## Strict packed64 Phase A1/A2 inherited baseline

状态：

- 进入本轮时 active selector 已确认：
  - `op_kernel/selected/stage1_selected.h` -> `../p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
  - `stage1_selected_config.h`: `p3l1_p8a`, `C64_1616_FP16`, chunk 64
- active Stage1 与 A1/A2 baseline 字节一致：
  - SHA256 `970a1a1f78c4aa0de72cd8a1057d3a5e88f7219c4358aaa9184e9c9f6a35826f`

A1：

- 完成 identity64 生命周期优化。
- `identity64` 使用独立 `aicHalfL1Identity64_`，通过 `packed64IdentityPrepared_` 缓存。
- `PrepareHalfL1IdentityByMmad(16/32/64)` 的普通 identity cache 不覆盖 `identity64`。
- 验证：
  - build/install PASS。
  - `S=64/256/768/1024` smoke PASS。
  - compare PASS：`16 passed in 6.15s`。
- baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA1_identity_once_baseline.h`
- performance：
  - `tests/pytest/prof_out/packed64_phaseA1_identity_once_20260510/performance_compare.csv`
  - ratio 约 `1.376 / 1.954 / 2.785 / 2.831`

A2：

- 失败候选：
  - 尝试将第一轮改成 `result64 @ (I + P^2)`，`S=64` runtime timeout。
  - 已回滚，不保存为 baseline。
- 收口结论：
  - 当前 A1 已只有一次 `INV @ I` bootstrap，后两轮不重复 `INV @ I`。
  - A2 作为 no-code 审计 phase，active 与 A1 baseline 字节一致。
- 验证：
  - build/install PASS。
  - `S=64/256/1024` smoke PASS。
  - compare PASS：`16 passed in 6.12s`。
- baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA2_no_repeat_inv_i_baseline.h`

## Strict packed64 Phase A3: constexpr helper / offset constantization

目标：

- 只做 strict one-shot helper 模板化和 offset 常量化。
- 不改 P8 compose、BF16 GM 写回、resident L1 RHS、buffer 生命周期。

改动：

- 固化 `packedSize=64`、`blockSize=64`、`fractal=16`、diagonal bases `{0,16,32,48}`。
- 新增 strict packed64 专用 helper：
  - `FixpipePacked64FullToHalfL1()`
  - `FixpipePacked64Diag16ToHalfL1<base>()`
  - `LoadPacked64DiagOnlyToL0A()`
  - `LoadPacked64DiagOnlyToL0B()`
  - `MmadPacked64DiagOnly()`
  - `RawMmadPacked64DiagOnly()`
  - `HalfL1PowerPacked64DiagOnlyOnC()`
- `P7Packed64DiagonalNeumannOneShotToResult()` 改用上述专用 helper。
- `EnsurePacked64Identity()` 用 compile-time bases 写入四个 `16x16` diagonal identity blocks。
- 编译修正：
  - 第一版全局 `constexpr` offset 函数被 AscendC 编译器视作 host function，AICORE 内调用失败。
  - 已改为全局常量和模板内 `constexpr` offset，build PASS。

验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS：
  - `16 passed in 6.11s`

Performance：

- 输出：
  - `tests/pytest/prof_out/packed64_phaseA3_constexpr_helper_20260510/performance_compare.csv`
- ratio neumann/normal：
  - `S=64`: `1.379`
  - `S=512`: `1.931`
  - `S=2048`: `2.703`
  - `S=10240`: `2.742`

baseline：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA3_constexpr_helper_baseline.h`
- active 与 baseline SHA256：
  - `86c59770e43a5454013e4f84e75098c43fa164a45ba0094c7768944ed487f5fb`

## Strict packed64 Phase A4: L1 lifetime ownership audit

目标：

- 只解决 `diagScratch64` 与 `lowerWork64` 生命周期归属。
- 不做 preload overlap，不改 P8 公式，不改 RHS。

审计结论：

- 当前 active 已满足 A4 ownership：
  - `result64 = aicHalfL1P3Result_`
  - `diagPower64 = aicHalfL1TilePower_`
  - `diagScratch64 = aicHalfL1TileInv_`
  - `lowerWork64 = aicHalfL1Scratch_`
  - `identity64 = aicHalfL1Identity64_`
- `LoadPacked64DiagonalInputsToL1()` 清零 `aicHalfL1Scratch_`，随后 `PreloadP8LowerLeftBlocksToLowerWork64()` 将 lower-left blocks 写入 `lowerWork64`。
- P7 strict one-shot 只把 `aicHalfL1TileInv_` 当 `diagScratch64`，不写 `aicHalfL1Scratch_`。
- P8 compose 读取 `aicHalfL1Scratch_` 作为 lower-left 输入。

处理：

- A4 为 no-code ownership phase。
- 保存内容与 A3 相同的 baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA4_l1_lifetime_baseline.h`
  - SHA256 `86c59770e43a5454013e4f84e75098c43fa164a45ba0094c7768944ed487f5fb`

下一步：

- Phase A5 在 A4 ownership 已明确的基础上，才开始做 lower-left GM->L1 preload 与 P7 one-shot 计算重叠。

## Strict packed64 Phase A5: lower-left preload overlap

目标：

- 在 A4 已分离 `diagScratch64` 与 `lowerWork64` 生命周期后，将 P8 lower-left GM->L1 搬运发起到 P7 one-shot 计算窗口内。
- 不改 P8 lower-triangular 公式，不改 P8 packing。

改动：

- 新增 no-wait preload helper：
  - `LoadGmHalfBlockToPackedHalfL1NoWait()`
  - `LoadP8LowerLeftToLowerWork64NoWait()`
  - `PreloadP8LowerLeftBlocksToLowerWork64NoWaitFull64()`
- 保留 sync preload helper：
  - `PreloadP8LowerLeftBlocksToLowerWork64Sync()`
- full64 strict one-shot path 中：
  - 先发起 `lowerWork64[(16,0)/(48,32)/(32,0)]` no-wait preload。
  - 再执行 `P7Packed64DiagonalNeumannOneShotToResult()`。
  - P8 compose 前调用 `SyncMte2ToMte1()`。
- fallback/tail 路径仍使用同步 preload。

验证：

- build PASS。
- install PASS。
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=4.041666e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS：
  - `16 passed in 6.21s`

Performance：

- 输出：
  - `tests/pytest/prof_out/packed64_phaseA5_lower_preload_overlap_20260510/performance_compare.csv`
- ratio neumann/normal：
  - `S=64`: `1.312`
  - `S=512`: `1.918`
  - `S=2048`: `2.713`
  - `S=10240`: `2.743`

baseline：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA5_lower_preload_overlap_baseline.h`
- SHA256：
  - `96915b7bfc63290fc5c3c4f7941fbe918fd5a3bb37465a5f294da6b17e6d0bfe`

下一步：

- Phase A6 只迁移 P8 第一层 `(16,0)` 和 `(48,32)` 两个 independent `16x16` lower-triangular compose 到 packed `32x32` MMAD。
- 第二层 `32x32 -> 64x64` compose 和 RHS 生命周期不在 A6 修改。

## Strict packed64 Phase A6: P8 first-layer packed32 compose

目标：

- 只迁移 P8 第一层 `(16,0)` 和 `(48,32)` 两个 independent `16x16` lower-triangular compose。
- 合成 packed `32x32` 两步 MMAD：
  - `tmp0 = lowerWork64[(16,0)] @ result64[(0,0)]`
  - `tmp1 = lowerWork64[(48,32)] @ result64[(32,32)]`
  - `out0 = result64[(16,16)] @ tmp0`
  - `out1 = result64[(48,48)] @ tmp1`
- 第二层 `32x32 -> 64x64` compose、BF16 GM 写回、resident RHS 不在本阶段修改。

改动：

- 新增 packed32 helper：
  - `LoadPacked16BlockToL0Offset()`
  - `LoadP8FirstLayerLowerToPacked32L0A()`
  - `LoadP8FirstLayerTopInvToPacked32L0B()`
  - `LoadP8FirstLayerBottomInvToPacked32L0A()`
  - `LoadPacked32DiagOnlyToL0B()`
  - `MmadPacked32()`
  - `RawMmadP8FirstLayerLowerTopPacked32()`
  - `RawMmadP8FirstLayerBottomTmpPacked32()`
  - `P8ComposeFirstLayerPacked32Direct()`
- full64 strict one-shot path 中，将两个 `P8Compose32From16SingleDirect()` 替换为 `P8ComposeFirstLayerPacked32Direct()`。
- fallback/tail 路径仍保留原 `16x16` compose。
- 修补点：packed32 output copy 后重置 `preparedHalfIdentityBlockSize_ = 0`，避免 copy helper 内部 16x16 identity 状态影响后续 helper。

验证：

- build PASS。
- install PASS。
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=2.295491e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS：
  - `16 passed in 6.12s`

Performance：

- 输出：
  - `tests/pytest/prof_out/packed64_phaseA6_p8_first_layer_packed32_20260510/performance_compare.csv`
- ratio neumann/normal：
  - `S=64`: `1.377`
  - `S=512`: `2.120`
  - `S=2048`: `2.964`
  - `S=10240`: `3.023`

baseline：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA6_p8_first_layer_packed32_baseline.h`
- SHA256：
  - `4e90424f3120b2f7ef80d7dab075c3da79ff28933ddba40c883a669910cfef21`

结论：

- A6 correctness gate 全过，可作为 strict packed64 当前回退点。
- performance 大 shape 退化，A6 不是性能收口；继续按 A7/A8/A9 推进。
- Phase A7 静态检查显示当前 `P8Compose64From32Direct32()` 已是 `lowerWork64[(32,0)] @ result64[(0,0)]`、`result64[(32,32)] @ tmp`，output 直接写 `result64[(32,0)]`，可作为 no-code 收口 phase 独立验证。

## Strict packed64 Phase A7: P8 second-layer packed compose

目标：

- 收口 P8 第二层 `32x32 -> 64x64` packed compose。
- 保持 lower-triangular 公式：
  - `tmp = lowerWork64[(32,0)] @ result64[(0,0)]`
  - `out = result64[(32,32)] @ tmp`
- output 直接写 `result64[(32,0)]`。
- 不改 P7，不改 BF16 GM 写回，不改 RHS。

处理：

- A7 为 no-code structure phase。
- 静态检查 `P8Compose64From32Direct32()`：
  - 第一段 `RawMmadPackedHalfL1AccumulateTile(aicHalfL1Scratch_, aicHalfL1P3Result_, ..., C64_BLOCK32, 0, 0, 0, C64_BLOCK32, ...)` 对应 `lowerWork64[(32,0)] @ result64[(0,0)]`。
  - 第一段结果 Fixpipe 到 `aicHalfL1TileInv_` 作为 `tmp`。
  - 第二段 `RawMmadPackedHalfL1AccumulateTile(aicHalfL1P3Result_, aicHalfL1TileInv_, ..., C64_BLOCK32, C64_BLOCK32, 0, 0, C64_BLOCK32, ...)` 对应 `result64[(32,32)] @ tmp`。
  - 第二段结果 `CopyL0CMatrixToPackedHalfL1(..., C64_BLOCK32, 0, C64_BLOCK32, C64_BLOCK32, chunkSize_, ...)` 直接写 `result64[(32,0)]`。
- active 与 A6 baseline 字节一致。

验证：

- build PASS。
- install PASS。
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=2.454019e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS：
  - `16 passed in 6.13s`

Performance：

- 输出：
  - `tests/pytest/prof_out/packed64_phaseA7_p8_second_layer_packed_20260510/performance_compare.csv`
- ratio neumann/normal：
  - `S=64`: `1.466`
  - `S=512`: `2.109`
  - `S=2048`: `2.978`
  - `S=10240`: `3.017`

baseline：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA7_p8_second_layer_packed_baseline.h`
- SHA256：
  - `4e90424f3120b2f7ef80d7dab075c3da79ff28933ddba40c883a669910cfef21`

结论：

- A7 correctness gate 全过，可作为 A8 的回退点。
- A7 没有代码变化，不是性能收益；A8 只做 BF16 GM / resident L1 inverse 写回生命周期定位。

## Strict packed64 Phase A8: BF16 / resident L1 lifecycle

目标：

- 不直接删除 BF16 `attnWsGm_` inverse 写回。
- 明确 lower 写回、upper zero 写回、resident L1 copy、fused RHS、跨 chunk workspace 复用的依赖关系。

结论：

- `FusedRhsAICProcess()` 的 inverse 输入是 `aicHalfL1ResidentInv_`，不是 `attnWsGm_`。
- `StorePackedL1InverseToBf16Attn()` 仍必须在 fused RHS 前执行；它不只是最终 GM 内容，还承担当前 pipeline/lifecycle 可见性副作用。
- 已失败且不要重复的尝试：
  - 删除完整 BF16 GM 写回：`S=1024` FAIL。
  - 跳过 upper zero BF16 writeback：`S=1024` FAIL。
  - 跳过 lower BF16 writeback、只保留 upper zero：`S=1024` FAIL。
  - 将完整 BF16 写回延后到两段 fused RHS 之后：`S=1024` FAIL。

验证：

- build PASS。
- install PASS。
- `S=64 / 256 / 1024` smoke PASS。
- compare PASS：
  - `16 passed in 6.14s`

Performance：

- 输出：
  - `tests/pytest/prof_out/packed64_phaseA8_bf16_lifecycle_20260510/performance_compare.csv`
- ratio neumann/normal：
  - `S=64`: `1.474`
  - `S=512`: `2.128`
  - `S=2048`: `2.970`
  - `S=10240`: `3.017`

baseline：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA8_bf16_lifecycle_baseline.h`
- SHA256：
  - `4e90424f3120b2f7ef80d7dab075c3da79ff28933ddba40c883a669910cfef21`

## Strict packed64 Phase A9: BF16 pre-RHS required

目标：

- 在 A8 生命周期结论基础上确认 resident L1-only RHS 消费状态。
- 不改 final RHS 数学语义。

处理：

- A9 为 no-code lifecycle conclusion。
- active 从 A8 baseline 恢复后验证。
- RHS inverse 消费已经是 resident L1；但 BF16 GM pre-RHS 写回被确认仍为 correctness 所需 side effect，不能安全削减。

验证：

- build PASS。
- install PASS。
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.160189e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS：
  - `16 passed in 6.76s`

Performance：

- 输出：
  - `tests/pytest/prof_out/packed64_phaseA9_bf16_prerhs_required_20260510/performance_compare.csv`
- ratio neumann/normal：
  - `S=64`: `1.468`
  - `S=512`: `2.125`
  - `S=2048`: `2.941`
  - `S=10240`: `3.004`

baseline：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA9_bf16_prerhs_required_baseline.h`
- SHA256：
  - `4e90424f3120b2f7ef80d7dab075c3da79ff28933ddba40c883a669910cfef21`

## Strict packed64 Phase A10: cleanup baseline

目标：

- 清理阶段性命名和路径表达，确认 A0-A10 全部完成。
- 不改变 P7/P8/RHS 数学。

改动：

- `ENABLE_STRICT_PACKED64_ONESHOT_A0` 重命名为 `ENABLE_STRICT_PACKED64_ONESHOT`。
- full64 strict packed64 one-shot 分支保持 active。
- tail/非 full64 fallback 分支按未启用 case 保护保留，不删除。

验证：

- build PASS。
- install PASS。
- `S=64` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=2.805576e-02`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS：
  - `16 passed in 7.17s`

Performance：

- 输出：
  - `tests/pytest/prof_out/packed64_phaseA10_cleanup_20260510/performance_compare.csv`
- ratio neumann/normal：
  - `S=64`: `1.408`
  - `S=512`: `2.150`
  - `S=2048`: `2.980`
  - `S=10240`: `2.984`

baseline：

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA10_cleanup_baseline.h`
- SHA256：
  - `35c7b0817a0b81271a5d47f631c4f41bd409fc2c2b38c15433e9e288e157701b`

结论：

- A0-A10 correctness 收口完成。
- A10 仍明显慢于 Phase 10，不能作为性能收口；需要继续 A11+。
- profile 显示 AIC scalar 放大最明显，大 shape 约 `4.6x` normal；MTE1/MTE2/Fixpipe 也有 `1.4x~2.2x` 放大。
