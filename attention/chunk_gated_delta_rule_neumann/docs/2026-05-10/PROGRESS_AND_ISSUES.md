# 2026-05-10 进度与问题

## 已完成

- 当前 `p3l1_p8a` active path 已重新 build/install。
- `S=64` targeted smoke PASS。
- 当前 64-aligned compare set PASS：
  - `16 passed in 6.23s`
- 已完成性能 smoke：
  - `tests/pytest/prof_out/p3l1_p8a_current_baseline_20260510/performance_compare.csv`
- 已保存 precision baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_baseline.h`
- Phase 1 已完成并通过 correctness gate：
  - P8 lower-left blocks 预加载到 `aicHalfL1P3Result_` packed L1。
  - active `P8Compose32From16SingleDirect()` / `P8Compose64From32Direct32()` 内部不再做 lower-left GM on-demand load。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.20s`。
  - 性能输出：`tests/pytest/prof_out/p3l1_p8a_phase1_lower_l1_20260510/performance_compare.csv`
- Phase 2 已完成并通过 correctness gate：
  - P3 diagonal inverse 不再每个 `16x16` 立即 half GM 写回。
  - P8 compose 后只执行一次 `StoreP3DiagonalResultsToGm(paraIdx)`。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.16s`。
  - 性能输出：`tests/pytest/prof_out/p3l1_p8a_phase2_defer_diag_gm_20260510/performance_compare.csv`
- Phase 3 已完成并通过 correctness gate：
  - P3 diagonal 最后一轮 `inv + inv @ power` 直接 Fixpipe 到 `aicHalfL1P3Result_` packed 64x64 对角位置。
  - P3 diagonal 路径不再经过 compact->packed identity-MMAD 桥。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.25s`。
  - 性能输出：`tests/pytest/prof_out/p3l1_p8a_phase3_packed_landing_20260510/performance_compare.csv`
- Phase 4a 已完成并通过 correctness gate：
  - AIC 从 `aicHalfL1P3Result_` packed L1 直接写 BF16 `attnWsGm_`。
  - final RHS 仍读 BF16 GM，未融合。
  - 最终采用 `16x16` block 写回：lower-triangular block 从 packed L1 写出，upper-triangular block 显式写 0。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set 初次有 `varlen_b3_64_128_256` 边界波动，单独 rerun PASS；随后全量 rerun PASS：`16 passed in 6.16s`。
  - 性能输出：`tests/pytest/prof_out/p3l1_p8a_phase4a_aic_bf16_gm_20260510/performance_compare.csv`
- Phase 4b 已完成并通过 correctness gate：
  - 新增 `aicHalfL1ResidentInv_`，把 packed L1 inverse 复制到 resident L1。
  - Phase 4a 的 BF16 GM bridge 保留，final RHS 仍读 BF16 GM。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.16s`。
  - 性能输出：`tests/pytest/prof_out/p3l1_p8a_phase4b_resident_l1_bridge_20260510/performance_compare.csv`
- Phase 5 已完成并通过 target correctness gate：
  - final RHS 改为 AIC raw-MMAD fused path，直接消费 resident L1 inverse。
  - `StorePackedL1InverseToResidentL1()` 修复为 lower-triangular `16x16` block 复制。
  - RHS raw-MMAD 修复 rectangular output tile，覆盖 `dk/dv=32` 边界。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.16s`。
  - 性能输出：`tests/pytest/prof_out/p3l1_p8a_phase5_fused_rhs_20260510/performance_compare.csv`
- Phase 6a 已完成并通过 target correctness gate：
  - 单个 `16x16` P3 diagonal block 改为 P7-style direct solve。
  - 结果落到 `aicHalfL1P3Result_` packed 64x64 对角位置。
  - 未启用 group-packed 求解；Phase 6b 后续已单独完成。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.14s`。
  - 性能输出：`tests/pytest/prof_out/p3l1_p8a_phase6a_p7_single16_20260510/performance_compare.csv`
- Phase 6b 已完成并通过 target correctness gate：
  - group load 两个 diagonal `16x16` block 到 packed L1。
  - 分别执行两个 proven `16x16` P7 direct solves，并落到 `aicHalfL1P3Result_` packed 64x64 对角位置。
  - 清理失败 one-shot `64x64` helper，active path 不保留该失败目标尝试。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.51s`。
  - 性能输出：`tests/pytest/prof_out/p3l1_p8a_phase6b_group_load_two16_clean_20260510/performance_compare.csv`
  - 已保存 baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_phase6b_baseline.h`

## 当前问题

- 当前版本没有性能收益：
  - `S=64` case neumann/normal = `1.715`
  - `S=512` case neumann/normal = `2.716`
  - `S=2048` case neumann/normal = `3.216`
  - `S=10240` case neumann/normal = `3.231`
- 当前 active 已是 Phase 6b target correctness PASS，且继承 Phase 5 full RHS fusion：
  - P8 compose 的 diagonal/top/bottom inverse 已从 packed L1 读。
  - final RHS 已直接消费 resident L1 inverse，不再读 BF16 `attnWsGm_`。
  - RHS operand 使用 half `gBKHalfWsGm_` / `vBetaHalfWsGm_`，输出仍写 BF16 GM。
  - P3 diagonal 入口已改为 group load + two `16x16` P7 direct solves。
- Cleanup 后 active path 已收口：
  - active Stage1 不再保留 scalar probe、旧 `StageOneFp16*` wrapper、旧 GM compose fallback、旧 `INVWsGm_`。
  - tilingdata 只保留 `matmulTilingBf16`。
  - host inverse workspace 只保留 FP16 `L_pow`、FP16 `INVInit`、FP32 `tmp`。
  - 清理后重新验证：build/install PASS，`S=64 / 256 / 1024` smoke PASS，当前 compare set PASS：`16 passed in 6.14s`。
  - 当前求逆流程文档：`docs/2026-05-10/P3L1_P8A_PHASE6B_INVERSE_FLOW.md`。
- `matmulTilingFp32` 已更名为 `matmulTilingBf16`：
  - 旧名是原始 FP32 主链路的历史遗留名。
  - 当前 host 生成的是 `DT_BFLOAT16 / DT_BFLOAT16 / DT_BFLOAT16`，Stage1/2/3 公共 high-level matmul 也是 BF16 GM matmul。
  - 不应改成 FP16；FP16 只用于 Stage1 inverse raw-MMAD 内部 half L1/L0/GM 数据流。
  - 更名后重新验证：build/install PASS，`S=64 / 256 / 1024` smoke PASS，当前 compare set PASS：`16 passed in 6.21s`。
  - 性能输出：`tests/pytest/prof_out/p3l1_p8a_tiling_bf16_rename_20260510/performance_compare.csv`
- TilingData 重命名后性能仍无收益：
  - `S=64` case neumann/normal = `1.762`
  - `S=512` case neumann/normal = `3.102`
  - `S=2048` case neumann/normal = `4.326`
  - `S=10240` case neumann/normal = `4.392`
  - 本次是命名/注释收口，不是计算优化；normal 仍更快。
- Phase 1 没有带来性能收益：
  - `S=64` case neumann/normal = `1.654`
  - `S=512` case neumann/normal = `2.813`
  - `S=2048` case neumann/normal = `3.239`
  - `S=10240` case neumann/normal = `3.276`
  - 与 `p3l1_p8a_current_baseline_20260510` 相比，neumann 绝对耗时略慢。
- Phase 2 也没有带来性能收益：
  - `S=64` case neumann/normal = `1.717`
  - `S=512` case neumann/normal = `2.849`
  - `S=2048` case neumann/normal = `3.335`
  - `S=10240` case neumann/normal = `3.334`
  - 仅推迟 P3 diagonal half GM 写回不足以减少主耗时。
- Phase 3 也没有带来性能收益：
  - `S=64` case neumann/normal = `1.810`
  - `S=512` case neumann/normal = `2.856`
  - `S=2048` case neumann/normal = `3.268`
  - `S=10240` case neumann/normal = `3.308`
  - P3 packed landing 没有改变主要耗时。
- Phase 4a 也没有带来性能收益：
  - `S=64` case neumann/normal = `1.648`
  - `S=512` case neumann/normal = `2.783`
  - `S=2048` case neumann/normal = `3.293`
  - `S=10240` case neumann/normal = `3.338`
  - AIC direct BF16 GM writeback 已替代 AIV half->BF16 cast，但 final RHS 仍读 GM，整体仍不是 full fuse。
- Phase 4b 明显退化：
  - `S=64` case neumann/normal = `2.670`
  - `S=512` case neumann/normal = `5.231`
  - `S=2048` case neumann/normal = `6.129`
  - `S=10240` case neumann/normal = `6.220`
  - resident copy 只是 Phase 5 输入准备，不能作为性能完成点。
- Phase 5 没有带来性能收益：
  - `S=64` case neumann/normal = `1.889`
  - `S=512` case neumann/normal = `3.073`
  - `S=2048` case neumann/normal = `4.207`
  - `S=10240` case neumann/normal = `4.267`
  - Phase 5 是 target correctness PASS 和 full RHS fusion 边界，但性能仍慢于 normal。
- Phase 6a 没有带来性能收益：
  - `S=64` case neumann/normal = `1.918`
  - `S=512` case neumann/normal = `3.286`
  - `S=2048` case neumann/normal = `4.407`
  - `S=10240` case neumann/normal = `4.522`
  - Phase 6a 只是单 `16x16` direct solve correctness 边界，尚未去掉 group 维度上的逐块调度。
- Phase 6b 没有带来性能收益：
  - `S=64` case neumann/normal = `1.902`
  - `S=512` case neumann/normal = `3.281`
  - `S=2048` case neumann/normal = `4.430`
  - `S=10240` case neumann/normal = `4.553`
  - Phase 6b 是 group load + two `16x16` solve 的 target correctness 边界，不是 strict `32x32/64x64` group-MMAD diagonal solver。
- strict group diagonal-only raw-MMAD body 仍失败：
  - full `32x32` group solve：`S=32` out max_abs `1.022196e+00`，state max_abs `4.903520e-01`。
  - one-shot `64x64` packed diagonal-only：`S=16` out max_abs `9.514923e-02`，state max_abs `1.818329e-01`；`S=64` out max_abs `1.059731e-01`，state max_abs `5.440532e-03`。
  - 该失败实现未保存为 baseline；后续若继续，应优先查 L1->L0/Fixpipe/identity placement。
- 当前 `out` 误差明显高于 step 3：
  - smoke/compare 中常见 `~3e-02` 到 `~7.5e-02`。
  - 虽然当前阈值内 PASS，但后续每一步必须继续观察。

## 判断

- 用户提出的下一步方向合理：
  1. 已完成：先把 lower-triangular 初始块 materialize 到 packed L1，去掉 P8 compose 的 GM lower-left load。
  2. 下一步：去掉 P3 diagonal 的中间 GM 写回，只保留最终 half GM 写回。
  3. 再把 P3 改成 P7-style packed layout，减少 compact->packed identity-MMAD 搬运桥。
  4. 已完成：再改 AIC 直接写 BF16 `attnWsGm_`，去掉 GM half -> AIV cast。
  5. 已完成：Phase 5 final RHS fusion，AIC 直接消费 resident inverse。
  6. 已完成：Phase 6a 只替换 P3 diagonal 单 `16x16` 求解为 P7-style direct solve，不同时改 P8/RHS。
  7. 已完成：Phase 6b 把 P3 diagonal 入口改成 group load + two `16x16` P7 solves，并直接落到 packed 位置。
- 当前 planned phases 已全部完成。通过版本不是性能收益，也不是 strict group-MMAD diagonal solver 完成。

## 后续风险

1. 当前可回退点是 Phase 6b baseline：
   - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_phase6b_baseline.h`
2. 若继续追 strict group diagonal-only raw-MMAD，必须作为新阶段单独 gate：
   - build/install。
   - `S=64 / 256 / 1024` smoke。
   - 当前 64-aligned compare set。
   - performance smoke。
3. 首要排查方向：
   - diagonal-only group raw-MMAD 的 L1->L0 placement。
   - `CopyHalfL1DiagonalFractalsToL0()` / zero-fill 后的 operand layout。
   - Fixpipe `CFG_NZ` stride/offset。
   - identity operand placement 与 `initC/accumulateC` 顺序。
4. 全 repo 搜索旧 probe / 旧 tiling 名称时，仍会命中历史 baseline、非 active 实验文件和 skill 文档；判断当前状态时必须以 selector 指向的 active Stage1、top-level kernel、tilingdata、host tiling 为准。

## Packed64 diagonal inverse Phase 0 进展

- Phase 0 静态资源与别名确认已完成，build gate PASS。
- 当前别名：
  - `result64 = aicHalfL1P3Result_`
  - `diagPower64 = aicHalfL1TilePower_`
  - `lowerWork64 = aicHalfL1Scratch_`
  - `aicHalfL1TileInv_` 保留为 fallback/debug compact work。
- `AIC_RAW_L0_C_BYTES = 64KB`，可静态容纳 4 个 `64x64 FP32` C tile 分区；当前 raw-MMAD / Fixpipe helper 已支持传入 `LocalTensor<float>`，后续可用 `aicL0C_[offset]` 选择分区。
- Phase 0 未改数学、未跑 smoke / compare / performance。
- 下一步进入 Phase 1：packed64 容器零初始化 + diagonal GM 定点搬入，仍沿用 proven `16x16` solve。

## Packed64 diagonal inverse Phase 1 进展

- Phase 1 已完成并通过 target correctness gate：
  - build/install PASS。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.16s`。
- 当前安全回退点更新为：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase1_baseline.h`
- 性能输出：
  - `tests/pytest/prof_out/packed64_phase1_p7_style_diag_load_20260510/performance_compare.csv`
- Phase 1 没有带来性能收益：
  - `S=64` case neumann/normal = `1.870`
  - `S=512` case neumann/normal = `3.255`
  - `S=2048` case neumann/normal = `4.394`
  - `S=10240` case neumann/normal = `4.466`
- 当前 active 仍只是 packed64 容器初始化 + proven `16x16` solve；尚不是 packed64 one-shot diagonal solver。
- Phase 2 应只改 lower-left 输入来源到 `lowerWork64`，不同时改 diagonal solve 或 compose writeback。

## Packed64 diagonal inverse Phase 2 进展

- Phase 2 已完成并通过 target correctness gate：
  - build/install PASS。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.09s`。
- 当前安全回退点更新为：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase2_baseline.h`
- 性能输出：
  - `tests/pytest/prof_out/packed64_phase2_lowerwork64_20260510/performance_compare.csv`
- Phase 2 没有带来性能收益：
  - `S=64` case neumann/normal = `1.825`
  - `S=512` case neumann/normal = `3.146`
  - `S=2048` case neumann/normal = `4.345`
  - `S=10240` case neumann/normal = `4.463`
- 当前 active 仍是 packed64 容器 + proven `16x16` solve；尚不是 packed64 one-shot diagonal solver。
- Phase 3 strict packed64 one-shot diagonal solve 已按用户指令跳过，不能描述为完成。

## Packed64 diagonal inverse Phase 4~7 进展

- Phase 3 处理：
  - 用户允许“如果打包太难，就先跳过 phase3，先做 phase4~7”。
  - 本轮没有完成 strict packed64 one-shot diagonal solver，也没有保存 Phase 3 baseline。
  - 后续若继续追 strict packed64 one-shot，必须作为新阶段单独 gate。
- Phase 4 已完成并通过 correctness gate：
  - 在 proven `16x16` diagonal solve path 内使用 `cInv/cPower` L0C 分区，减少 repeated `inv @ I`。
  - build/install PASS。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS。
  - 性能输出：`tests/pytest/prof_out/packed64_phase4_l0c_accum_20260510/performance_compare.csv`
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase4_baseline.h`
- Phase 5 已完成并通过 correctness gate：
  - compact diagonal scratch 移到 `aicHalfL1TileInv_`。
  - 新增 `PreloadP8LowerLeftBlocksToLowerWork64(paraIdx)`，compose 不再 on-demand load lower-left。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.13s`。
  - 性能输出：`tests/pytest/prof_out/packed64_phase5_lower_preload_20260510/performance_compare.csv`
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase5_baseline.h`
- Phase 6 已完成并通过 correctness gate：
  - P8 compose output 直接 Fixpipe 到 `result64 = aicHalfL1P3Result_`：
    - `16+16 -> 32` 写 `(16,0)` / `(48,32)`。
    - `32+32 -> 64` 写 `(32,0)`。
  - 移除同一 L0C 同时写 packed L1 和 half GM 的失败尝试；该尝试曾导致 `S=64` timeout。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.23s`。
  - 性能输出：`tests/pytest/prof_out/packed64_phase6_compose_direct_result_20260510/performance_compare.csv`
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase6_baseline.h`
- Phase 7 已完成并通过 correctness gate：
  - `C64InverseBy16x16Neumann()` 不再调用 `StoreP3DiagonalResultsToGm(paraIdx)`。
  - 保留 BF16 `attnWsGm_` 写回和 resident L1 copy，final RHS 数学语义不变。
  - build/install PASS。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.14s`。
  - 性能输出：`tests/pytest/prof_out/packed64_phase7_skip_diag_half_gm_20260510/performance_compare.csv`
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase7_baseline.h`
- Phase 8 已完成并通过 correctness gate：
  - 保留 BF16 `attnWsGm_` 写回；删除该写回的尝试在 `S=1024` 失败，未保留。
  - AIV 侧同步改为 `GBK -> set 0x6 -> VBeta -> set 0x5 -> QPrime -> wait 0xd -> wait 0xc`，用 AIC inverse/RHS 空档掩盖 AIV 后续准备。
  - final RHS 数学语义不变，Stage2/Stage3/QK/`gBK/vBeta` 生成语义不变。
  - build/install PASS。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.24s`。
  - 性能输出：`tests/pytest/prof_out/packed64_phase8d_defer_wait0d_20260510/performance_compare.csv`
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase8_sync_overlap_baseline.h`
- Phase 9 尝试跳过 BF16 upper-triangle zero writeback，`S=64/256` PASS 但 `S=1024` FAIL，已回滚，不保存 baseline。
- Phase 10 已完成并通过 correctness gate：
  - 删除冗余 `0xd` cross-core set/wait。
  - `0xc` 仍作为 AIC 两段 fused RHS 完成、共享 workspace 可复用的边界。
  - build/install PASS。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.15s`。
  - 性能输出：`tests/pytest/prof_out/packed64_phase10_drop_wait0d_20260510/performance_compare.csv`
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase10_drop_wait0d_baseline.h`
- Phase 11 尝试跳过 `RestoreRawMmadState()`：
  - build/install PASS。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.13s`。
  - 性能输出：`tests/pytest/prof_out/packed64_phase11_skip_restore_raw_state_20260510/performance_compare.csv`
  - 四档 neumann 绝对耗时均略慢于 Phase 10，已回滚到 Phase 10 baseline。
  - 回滚后 build/install PASS，`S=1024` smoke PASS。
- Phase 12 尝试跳过 RHS full-tile operand L1 zero-init：
  - build/install PASS。
  - `S=64 / 256` smoke PASS。
  - `S=1024` smoke FAIL，`out/state` 接近全 0。
  - 已恢复每个 RHS tile 前的 `ZeroHalfL1(aicHalfL1TilePower_, chunkSize_ * chunkSize_)`。
  - 回滚后 build/install PASS，`S=1024` smoke PASS。
- strict packed64 one-shot diagonal solve placement 定位：
  - guarded 版本和无 mask para guard 版本均 correctness PASS。
  - `stageOneParaNum = 4`，`0xF` magic mask 不过滤真实 `paraIdx`；正确 guard 是 `validLenBatch_[paraIdx] >= chunkSize_ && paraIdx < paraNum_`。
  - 无 mask para guard 版本 build/install PASS，`S=64 / 256 / 1024` smoke PASS，compare `16 passed in 6.15s`。
  - correctness 说明 L1->L0 full packed placement、Fixpipe packed L1 offset、identity placement 已定位通过。
  - performance 输出：`tests/pytest/prof_out/packed64_strict_oneshot_para_guard_20260510/performance_compare.csv`。
  - ratio `1.319 / 2.091 / 2.961 / 3.034`，大 shape 相对 Phase 10 退化；已恢复 Phase 10 baseline。
  - 恢复后 active 与 Phase 10 baseline diff 为空，build/install PASS，`S=1024` smoke PASS，compare `16 passed in 6.14s`。

## 当前 packed64 progressive 状态

- 当前安全回退点：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA4_l1_lifetime_baseline.h`
- 当前 active path：
  - strict packed `64x64` one-shot diagonal solve active。
  - A3 已完成 strict one-shot helper 模板化和 offset 常量化。
  - A4 已完成 `diagScratch64` / `lowerWork64` ownership 审计。
  - P8 lower-left preload 到 `lowerWork64`。
  - P8 compose output 直接回 `result64`。
  - 不再写 diagonal half GM。
  - BF16 `attnWsGm_` 写回和 resident L1 copy 仍保留。
- AIC/AIV 同步已做 Phase 8 流水掩盖，并在 Phase 10 删除冗余 `0xd` set/wait。
- Phase 10 性能仍不是收益：
  - `S=64` case neumann/normal = `1.379`
  - `S=512` case neumann/normal = `2.001`
  - `S=2048` case neumann/normal = `2.760`
  - `S=10240` case neumann/normal = `2.824`
  - normal 仍更快；Phase 10 是历史 correctness-safe 同步清理，不是稳定性能收益。

## Strict packed64 A0-A5 最新状态

- Phase A0 已完成：
  - strict packed64 one-shot 重新接入 active target path。
  - build/install PASS，`S=64/256/1024` smoke PASS，compare PASS。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA0_strict_oneshot_baseline.h`
- Phase A1 已完成：
  - `identity64` 改为独立 `aicHalfL1Identity64_`，`packed64IdentityPrepared_` 生命周期缓存。
  - build/install PASS，`S=64/256/768/1024` smoke PASS，compare `16 passed in 6.15s`。
  - performance：`tests/pytest/prof_out/packed64_phaseA1_identity_once_20260510/performance_compare.csv`。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA1_identity_once_baseline.h`
- Phase A2 已完成：
  - A2 为 no-code audit phase；A1 已只有一次 `INV @ I` bootstrap，后续 rounds 不重复 `INV @ I`。
  - 曾尝试 `result64 @ (I + P^2)`，`S=64` timeout，已回滚，不作为主线。
  - build/install PASS，`S=64/256/1024` smoke PASS，compare `16 passed in 6.12s`。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA2_no_repeat_inv_i_baseline.h`
- Phase A3 已完成：
  - 新增 strict packed64 专用 L0 diagonal load、MMAD、Fixpipe helper。
  - 固化 `64/16/{0,16,32,48}`，P7 数学顺序不变。
  - build/install PASS，`S=64/256/1024` smoke PASS，compare `16 passed in 6.11s`。
  - performance：`tests/pytest/prof_out/packed64_phaseA3_constexpr_helper_20260510/performance_compare.csv`。
  - ratio `1.379 / 1.931 / 2.703 / 2.742`。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA3_constexpr_helper_baseline.h`
- Phase A4 已完成：
  - no-code ownership phase。
  - `diagScratch64 = aicHalfL1TileInv_`，`lowerWork64 = aicHalfL1Scratch_`，P7 scratch 与 P8 lower-left 生命周期已分离。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA4_l1_lifetime_baseline.h`
  - SHA256 与 A3 baseline 相同：`86c59770e43a5454013e4f84e75098c43fa164a45ba0094c7768944ed487f5fb`
- Phase A5 已完成：
  - full64 strict one-shot path 中，P7 前发起 lower-left no-wait preload，P8 前等待；fallback/tail 仍走 sync preload。
  - P8 公式和 P8 packing 未改。
  - build/install PASS，`S=64/256/1024` smoke PASS，compare `16 passed in 6.21s`。
  - performance：`tests/pytest/prof_out/packed64_phaseA5_lower_preload_overlap_20260510/performance_compare.csv`。
  - ratio `1.312 / 1.918 / 2.713 / 2.743`。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA5_lower_preload_overlap_baseline.h`
  - SHA256：`96915b7bfc63290fc5c3c4f7941fbe918fd5a3bb37465a5f294da6b17e6d0bfe`

## 后续建议

1. 下一步按 todo.md 进入 Phase A8：
   - A7 已作为 no-code structure phase 独立验证当前第二层 `32x32 -> 64x64` compose 收口。
   - A8 只做 RHS / BF16 GM inverse 写回生命周期定位，不直接删 BF16 GM 写回。
2. 新阶段仍必须按 build/install、`S=64 -> 256 -> 1024`、compare、performance 独立验证。
3. 当前 A7 是 strict packed64 correctness-safe 回退点，不是 performance win；strict packed64 one-shot 已是 active target path。
4. 已试过的同步顺序不要重复：
   - 删除 BF16 `attnWsGm_` 写回：`S=1024` FAIL。
   - `GBK -> wait 0xd -> set 0x6`：PASS，比例约 `1.360 / 1.974 / 2.816 / 2.868`。
   - `GBK -> set 0x6 -> wait 0xd`：PASS，比例约 `1.397 / 1.975 / 2.798 / 2.851`。
   - 当前保留 `GBK -> set 0x6 -> VBeta -> set 0x5 -> QPrime -> wait 0xd -> wait 0xc`：PASS，比例约 `1.320 / 1.989 / 2.794 / 2.835`。
5. Phase 9 尝试跳过 BF16 upper-triangle zero writeback 已失败并回滚：
   - build/install PASS，`S=64/256` PASS，`S=1024` FAIL。
   - 失败现象是 `out/state` 接近全 0，说明 upper zero writeback 或完整 BF16 writeback side effect 仍被跨 chunk correctness 依赖。
   - 回滚到 Phase 8 baseline 后 install + `S=1024` smoke PASS。
   - 后续不要重复删除 BF16 写回或跳过 upper zero writeback，除非先重做 `attnWsGm_` 生命周期/清零边界设计。
6. Phase 11 尝试跳过 `RestoreRawMmadState()` 已回滚：
   - correctness gate PASS，但四档 neumann 绝对耗时均略慢于 Phase 10。
   - 已恢复 `RestoreRawMmadState()`，回到 Phase 10 baseline；恢复后 build/install + `S=1024` smoke PASS。
   - 后续不要继续按“删除 raw-MMAD 状态恢复”作为优化方向。
7. Phase 12 尝试跳过 RHS full-tile operand L1 zero-init 已失败并回滚：
   - `S=64/256` PASS，`S=1024` FAIL，失败形态为 `out/state` 接近全 0。
   - 已恢复 RHS tile 前 `ZeroHalfL1(aicHalfL1TilePower_, chunkSize_ * chunkSize_)`；恢复后 build/install + `S=1024` smoke PASS。
   - 后续不要直接删除该清零；若要优化，需先证明 `LoadGmHalfBlockToPackedHalfL1WithStride()` 覆盖、MTE1 可见性和 raw-MMAD operand 生命周期。

## Strict packed64 A6 completion

- Phase A6 已完成：
  - first-layer P8 compose 迁入 packed32 两步 MMAD。
  - 第二层 compose、BF16 GM 写回、resident RHS 未改。
  - build/install PASS，`S=64/256/1024` smoke PASS，compare `16 passed in 6.12s`。
  - performance：`tests/pytest/prof_out/packed64_phaseA6_p8_first_layer_packed32_20260510/performance_compare.csv`。
  - ratio `1.377 / 2.120 / 2.964 / 3.023`，normal 仍更快，且大 shape 相比 A5 退化。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA6_p8_first_layer_packed32_baseline.h`。
  - SHA256：`4e90424f3120b2f7ef80d7dab075c3da79ff28933ddba40c883a669910cfef21`。

## Strict packed64 A7 completion

- Phase A7 已完成：
  - no-code structure phase；active 与 A6 baseline 字节一致。
  - 当前 `P8Compose64From32Direct32()` 已满足第二层 packed compose：
    - `lowerWork64[(32,0)] @ result64[(0,0)]`
    - `result64[(32,32)] @ tmp`
    - output 直接写 `result64[(32,0)]`
  - build/install PASS，`S=64/256/1024` smoke PASS，compare `16 passed in 6.13s`。
  - performance：`tests/pytest/prof_out/packed64_phaseA7_p8_second_layer_packed_20260510/performance_compare.csv`。
  - ratio `1.466 / 2.109 / 2.978 / 3.017`，normal 仍更快。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA7_p8_second_layer_packed_baseline.h`。
  - SHA256：`4e90424f3120b2f7ef80d7dab075c3da79ff28933ddba40c883a669910cfef21`。

## Strict packed64 A8-A10 completion

- Phase A8 已完成：
  - lifecycle/no-code phase。
  - `FusedRhsAICProcess()` 读 `aicHalfL1ResidentInv_`，RHS inverse 消费已经是 resident L1。
  - BF16 `attnWsGm_` pre-RHS 写回仍为 correctness 所需 side effect。
  - 删除完整 BF16 写回、跳过 upper zero、跳过 lower blocks、或延后写回到 RHS 后，均已在 `S=1024` 失败，不作为 baseline。
  - build/install PASS，`S=64/256/1024` smoke PASS，compare `16 passed in 6.14s`。
  - performance：`tests/pytest/prof_out/packed64_phaseA8_bf16_lifecycle_20260510/performance_compare.csv`。
  - ratio `1.474 / 2.128 / 2.970 / 3.017`。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA8_bf16_lifecycle_baseline.h`。
- Phase A9 已完成：
  - no-code lifecycle conclusion。
  - active 恢复到 A8 baseline 后完整 gate。
  - 结论：RHS 保持 resident L1 消费，但 BF16 GM pre-RHS 写回不能安全削减。
  - build/install PASS，`S=64/256/1024` smoke PASS，compare `16 passed in 6.76s`。
  - performance：`tests/pytest/prof_out/packed64_phaseA9_bf16_prerhs_required_20260510/performance_compare.csv`。
  - ratio `1.468 / 2.125 / 2.941 / 3.004`。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA9_bf16_prerhs_required_baseline.h`。
- Phase A10 已完成：
  - 代码收口仅做稳定命名：`ENABLE_STRICT_PACKED64_ONESHOT_A0` -> `ENABLE_STRICT_PACKED64_ONESHOT`。
  - full64 strict packed64 one-shot 分支保持 active；tail/非 full64 fallback 分支按未启用 case 保护保留。
  - build/install PASS，`S=64/256/1024` smoke PASS，compare `16 passed in 7.17s`。
  - performance：`tests/pytest/prof_out/packed64_phaseA10_cleanup_20260510/performance_compare.csv`。
  - ratio `1.408 / 2.150 / 2.980 / 2.984`。
  - baseline：`op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA10_cleanup_baseline.h`。
  - SHA256：`35c7b0817a0b81271a5d47f631c4f41bd409fc2c2b38c15433e9e288e157701b`。

## 当前问题和 A11+ 方向

- A0-A10 correctness 已收口，但 A10 仍慢于 Phase 10：
  - Phase 10 ratio `1.379 / 2.001 / 2.760 / 2.824`。
  - A10 ratio `1.408 / 2.150 / 2.980 / 2.984`。
- A10 不是性能 baseline，不能停止在 A10。
- A10 profile 中放大最明显的是 AIC scalar：
  - `S=2048` scalar ratio 约 `4.59x`。
  - `S=10240` scalar ratio 约 `4.62x`。
- MTE/Fixpipe 也仍有明显放大：
  - MTE1 约 `2.16x`。
  - MTE2 约 `1.45x`。
  - Fixpipe 约 `1.57x~1.60x`。
- Phase A11 应只选一个优化点，优先审：
  - `P7Packed64DiagonalNeumannOneShotToResult()` 内每轮 sync/Fixpipe 可见性。
  - P8 packed helper 的 per-call state reset 和 copy helper。
  - `StorePackedL1InverseToBf16Attn()` 的 side-effect 保留前提下能否重排 pipe/cache 操作。
- 禁止重复尝试：
  - 删除 BF16 pre-RHS 写回。
  - 跳过 upper zero BF16 writeback。
  - 跳过 RHS operand L1 zero-init。
  - 删除 `RestoreRawMmadState()`。
