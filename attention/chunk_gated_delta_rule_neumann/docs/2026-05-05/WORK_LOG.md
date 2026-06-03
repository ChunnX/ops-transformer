# 2026-05-05 工作日志

## 当前求逆实现与耗时口径

- 用户询问当前求逆如何实现、主要耗时是什么。
- 当前默认 selector 为 `C64_3232_FP32_CVInverse`：`op_kernel/chunk_gated_delta_rule_neumann_stage1.h` include `chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h`。
- Stage1 中 AIV 负责生成 `L_powWsGm_=L` 和 `INVWsGm_=I-L`，AIC 在 `C64InverseBy32x32Neumann()` 中做 Neumann inverse。
- 主 chunk 仍为 `C=64`，求逆结构是 C64 拆成 32x32；每个 32x32 再用两个 16x16 对角块和 compose 路径组合。短尾 `validLen<=16` 直接走 16x16 shortcut。
- 16x16 Neumann block 执行三轮：`L^2/L^4/L^8`；32x32 block 执行四轮：`L^2/L^4/L^8/L^16`。power 使用 `RawMmadLowerTriangularBlock()`，只写下三角并显式清零上三角。
- `INV += INV@P` 使用 `RawMmadInvAddBlock()`：先 `INV@I` 用 `cmatrixInitVal=true` 把 old INV 种到 L0C/CMatrix，再 `INV@P` 用 `cmatrixInitVal=false` 在同一 CMatrix 累加，最后普通 Fixpipe 写回 `INVWsGm_`。
- compose 使用 raw MMAD：`Compose32From16()` 和 `Compose64From32()` 计算 lower-left block。
- final RHS 乘法 `INV @ k_cumdecay` 和 `INV @ v_beta` 仍走 FP32 GM `MatmulImpl`，不属于当前 raw fused inverse 验收。

## 主要耗时

- 最新 lower-triangular power smoke：`tests/pytest/prof_out/chunk_gdr_c64_3232_lower_tri_power_smoke/performance_compare.csv`。
- `B=8,S=16,D=64,Hq=16,Hkv=4`：normal `621.64us`，Neumann `822.32us`，ratio `1.323x`。AIC scalar `1.579x`，MTE2 `1.447x`，Fixpipe `1.362x`，MAC time 仅 `1.272x`。
- `B=8,S=64,D=64,Hq=16,Hkv=4`：normal `653.44us`，Neumann `1087.22us`，ratio `1.664x`。AIC scalar `2.630x`，MTE1 `2.965x`，MTE2 `2.923x`，Fixpipe `2.402x`，MAC time `2.381x`。
- 结论：主要耗时不是理论 MAC 本身，而是 raw tiny MMAD 多次调用带来的固定开销、GM/L1/L0 搬运、Fixpipe 写回、cache flush/DCCI、AIC/AIV 同步和 workspace 往返。
- 因为更新后的 `INV` 下一轮仍需要作为 A 输入，当前实现每轮仍是 `L0C -> Fixpipe -> GM -> LoadData -> L0A/B`，没有做到跨 round L1/TSCM resident。
- 后续性能方向仍是减少 GM 往返：`L0C accumulator -> Fixpipe -> L1/TSCM -> LoadData`，但需要先解决 compact NZ/stride/layout 正确性。

## 纯 cube 核内口径澄清

- 用户确认“现在求逆还不是纯 cube 核内”。结论：是的，当前还不是全程纯 cube kernel-resident。
- 已完成的是 `INV += INV@P` 的加法语义从 AIV/标量/Fixpipe atomic add 转为底层 MMAD CMatrix/L0C 累加。
- 仍然不纯的原因：
  - 初始 `L` 和 `I-L` 由 AIV 生成并写 GM。
  - 每轮 power/inv-add 结果都需要 `L0C -> Fixpipe -> GM` 写回。
  - 下一轮再从 GM 经 `GM -> L1 -> L0A/B -> Mmad` 读回。
  - final RHS `INV@k_cumdecay` 和 `INV@v_beta` 仍走 FP32 GM `MatmulImpl`。
- 因此当前是 raw MMAD/CMatrix 累加版 Neumann inverse，不是真正全程 L0/L1/TSCM resident 的纯 cube 融合核。

## 与核内 atomic add 到 GM 的区别

- 用户询问当前 CMatrix 累加和 kernel atomic add GM 有什么区别。
- 旧的 AIC/Fixpipe atomic-add GM 路线：
  - 先做 `tmp = INV @ P`，MMAD 只负责生成乘积。
  - 再通过 Fixpipe/Matmul atomic add 写 GM：`INVWsGm += tmp`。
  - 加法发生在 GM 写回阶段，依赖 GM atomic add 语义。
- 当前 CMatrix/MMAD 累加路线：
  - 先做 `INV @ I`，用 `cmatrixInitVal=true` 把 old `INV` 种到 L0C。
  - 再做 `INV @ P`，用 `cmatrixInitVal=false` 在同一个 L0C/CMatrix accumulator 中累加。
  - Fixpipe 只把已经完成加法的 updated `INV` 普通写回 GM，不再承担加法。
- 二者共同点：
  - 当前仍然需要把 updated `INV` 写回 GM，下一轮再读回 L0A/L0B。
  - 因此都还不是全程 L1/TSCM/L0 resident。
- 本质差异：
  - atomic add GM 是 `乘法结果写 GM 时加到旧 GM 值上`。
  - CMatrix 累加是 `MMAD 在 L0C accumulator 里完成 old_INV + INV@P`，GM 只是存储更新后的结果。

## 下一步：updated INV 写到 L1/TSCM

- 用户确认下一步是否应把累加结果搬运到 L1。
- 结论：方向正确，但精确说是把 L0C 中的 updated `INV` 通过 Fixpipe 写到 AIC 片上可再次 `LoadData` 的 L1/TSCM buffer，而不是写回 GM。
- 目标数据流：
  - 当前：`L0C -> Fixpipe -> GM -> DataCopy/LoadData -> L1/L0A/B`
  - 目标：`L0C -> Fixpipe -> L1/TSCM -> LoadData -> L0A/B`
- 这样仍然不是把 L0C 直接当下一轮 A/B，因为 L0C 不能直接作为 MMAD 输入；但能删掉 GM 往返和 cache flush/DCCI。
- 下一步需要先验证两个问题：
  1. Fixpipe 写到 L1/TSCM 的目标 layout、stride、对齐是否能被下一轮 `LoadData` 正确消费。
  2. 32x32/16x16 子块在 compact NZ 布局中的 offset 是否正确，尤其是 `Compose32From16()` / `Compose64From32()` 的 lower-left block。

## 今日完成算子计划

- 用户要求说明后台任务并写今天完算子计划。
- 后台检查结果：没有仍在运行的 `build.sh`、`pytest`、`msprof`、`opc`、`ninja/gmake/cmake` 任务；`jobs -l` 为空。
- 新增计划文档：`docs/2026-05-05/L0C_TO_L1_TSCM_INVERSE_COMPLETION_PLAN.md`。
- 今日计划聚焦 `L0C -> Fixpipe -> L1/TSCM -> LoadData` 最小正确闭环，先做 16x16 tile 数值探针，再扩到 32x32 compose，最后全量 compare 和 S16/S64 smoke。

## 用户约束更新：final RHS 暂不修改

- 用户明确 final RHS 可以先不用改。
- 今日后续实现范围收窄为 Neumann inverse 矩阵生成 resident 链路。
- `INV @ k_cumdecay` / `INV @ v_beta` 继续保持 FP32 GM `MatmulImpl`，避免与 L1/TSCM inverse 闭环调试混在一起。

## 5.05 计划执行与子代理结论

- 用户要求按 `docs/2026-05-05/L0C_TO_L1_TSCM_INVERSE_COMPLETION_PLAN.md` 开子代理执行并完成算子。
- 子代理只读检索结论一致：
  - 仓库中只找到 `common/include/kernel/l0c_to_l1_iterator.h` 的 `int32_t L0C -> half L1` 样例，没有 `float L0C -> float L1/TSCM -> LoadData<float>` 闭环样例。
  - `C64_3232_FP16_LocalTscm_CVInverse` PoC 走的是 half TSCM local Matmul 和标量 `AddLocalMatrix()`，且 final RHS 也改为 half 路线，不能作为默认 FP32 RHS 稳定消费端。
  - 默认 `C64_3232_FP32_CVInverse` 的 Neumann inverse 仍是 raw MMAD/CMatrix 累加后写回 `INVWsGm_`，final RHS 继续走 FP32 GM `MatmulImpl`。

## FP32 L0C -> L1/TSCM 尝试与硬限制

- 尝试在默认 `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h` 中增加 FP32 resident 链路：
  - 新增 `aicL1Inv_` 作为 64x64 FP32 NZ resident cache。
  - 增加 `CopyL0CMatrixToL1Nz()`，目标是 `Fixpipe<float, float, CFG_NZ>(LocalTensor<float>, L0C)`。
  - 增加 `LoadInvGmToL1Resident()` 和 `RawMmadInvAddBlockResident()`，让下一轮 `INV@P` 从 L1 resident `INV` 读 A。
- 构建失败后确认根因不是 layout 小错，而是 CANN/架构类型支持限制：
  - `dav_c220/kernel_operator_fixpipe_v2_impl.h` 的 L0C->L1 检查不支持 `float -> float`。
  - 当前 `ascend910_93` / DAV2201 可用方向是 L0C float 到 half/bfloat16/int8/uint8 等低精度 L1，不支持 FP32 resident cache。
- 结论：5.05 计划中的 FP32 `L0C -> Fixpipe -> L1/TSCM -> LoadData<float>` 默认实现不能在当前设备架构上落地。

## 默认线收敛处理

- 为保证算子可交付，已从默认 `C64_3232_FP32_CVInverse` 中移除不可编译的 FP32 resident 路径：
  - 删除 `aicL1Inv_`、`MatrixNzOffset()`、`CopyL0CMatrixToL1Nz()`、`LoadInvGmToL1Resident()`。
  - 删除 `RawMmadAccumulateTileL1InvA()`、`RawMmadInvAddOutputTileResident()`、`StoreRawMmadInvAddRowToL1()`、`RawMmadInvAddBlockResident()`。
  - `Neumann16InverseBlock()` / `Neumann32InverseBlock()` 恢复为已验证的 `RawMmadInvAddBlock()`：L0C/CMatrix 中完成 `old_INV + INV@P` 后普通 Fixpipe 写回 GM。
- final RHS 保持用户要求的稳定路径：
  - `INVWsGm_ @ gBKWsGm_ -> outKCumdecayGm_`
  - `INVWsGm_ @ vBetaWsGm_ -> outVInnerGm_`
  - 两者均继续使用 FP32 GM `MatmulImpl`。

## 验证结果

- 单算子构建通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j16`
- 双算子构建通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 安装通过：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 全量三方精度通过：
  - `source /home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer/bin/set_env.bash && CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.22s`

## 后续可行方向

- 若继续坚持片上 resident，需要改为非 FP32 resident 路线：
  - half/bfloat16 L1 resident inverse PoC，并重新做精度门槛验证。
  - 或等待/切换到支持 `float L0C -> float L1` 的架构能力后再实现 FP32 resident。
- 当前默认算子保持稳定 `C64_3232_FP32_CVInverse`：Neumann inverse 矩阵生成已完成 CMatrix/MMAD 累加，但跨 round 仍有 GM 中转。

## L0C -> L1 支持口径纠正与布局讨论

- 用户纠正“0C 搬运到 L1 不可能不支持”。重新收窄结论：
  - `L0C -> L1` 这条通路是支持的。
  - 当前架构限制点是 `float L0C -> float L1/TSCM` 不支持；可支持的方向包括 `float L0C -> half/bfloat16_t L1` 等低精度写回。
- 对 Neumann resident inverse 的可行数据流应写成：
  - `float L0C accumulator -> Fixpipe<half/bfloat16_t, float, CFG_NZ> -> L1 NZ`
  - 后续若继续 raw `Mmad`，需要再经 `LoadData -> L0A/L0B`，且 A/B 侧加载参数不同。
- 布局口径：
  - `CFG_NZ` 表示 Fixpipe 写到目的 L1 的输出格式，不应理解为“原样保留 L0C 内部物理排布”。
  - L0C 是 C/output accumulator 的内部布局；Fixpipe 到 L1 后得到的是 C 逻辑矩阵的 NZ 输出格式。
  - 若该矩阵下一轮作为左矩阵 A 使用，需要用 A 侧 `LoadData` 语义从 L1 NZ 读入 L0A；若作为右矩阵 B 使用，需要用 B 侧 `LoadData` 语义读入 L0B，不能用同一组参数混用。
  - raw 路线里这一步不是“再转成 NZ”，而是“把已在 L1 的 NZ/C-output layout 按 A/B bank 需要重新解释并载入 L0A/L0B”。历史 FP32 raw 修复也证明 A/B 侧布局参数必须拆开处理。
- 因此 half/bfloat16 resident PoC 的首要风险不是通路不存在，而是 `Fixpipe CFG_NZ` 输出、compact tile offset、`LoadData` 到 L0A/L0B 的 transpose/stride 是否匹配。

## 当前状态复核与子代理结论汇总

- 用户询问“现在是什么情况”，重新核对工作区当前状态：
  - 默认 selector 仍为 `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` include `chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h`。
  - 默认文件中 `FP16_L1_RESIDENT_INVERSE`、`FP16_L1_RESIDENT_SCALAR_PROBE`、`FP16_L1_ROUNDTRIP_THEN_RAW_PROBE`、`FP16_L1_ROUNDTRIP_MMAD_WRITEBACK_PROBE` 均为 `false`。
  - 当前默认路径仍是稳定 FP32 GM 中转：AIV 生成 `L/INV`，AIC 用 raw MMAD/CMatrix 做 Neumann `INV += INV@P`，updated `INV` 写回 GM，final RHS 仍走 FP32 GM `MatmulImpl`。
- 子代理只读结论补齐：
  - `L0C -> L1/TSCM` 应使用 `Fixpipe<half, float, CFG_NZ>` 或 `Fixpipe<bfloat16_t, float, CFG_NZ>` 这类低精度目标；CANN 头文件中未发现 `CFG_ZN`。
  - `CFG_NZ` 写到 L1/TSCM 后是 C/output 侧 NZ layout；作为下一轮 A 输入时，L0A load 倾向 `ifTranspose=false`，作为 B 输入时，L0B load 倾向 `ifTranspose=true`。
  - S=16 纯 CPU 模拟显示 FP16 inverse 量化本身误差约 `1e-3` 以内，不足以解释早前 NPU `1e-1` 级失败；优先怀疑 TSCM compact layout、offset、writeback probe 或临时调试分支。
- 当前下一步建议保持不变：先做隔离的 16x16 `L0C -> Fixpipe CFG_NZ -> half/bfloat16 L1/TSCM -> LoadData -> MMAD` 数值探针，探针通过后再接入完整 Neumann resident 分支；默认稳定线暂不打开 resident 开关。

## Baseline/Halflocal/Fulllocal 版本拆分与验证

- 用户纠正 baseline 口径：baseline 不是远端旧 GM+AIV-add 版本，而是最早精度通过的 `3*2` 方案，AIV 只负责初始 `L` / `I-L` 构造，后续 Neumann 轮次通过 AIC `MatmulImpl` / Fixpipe atomic add 实现 `INV += INV@P`。
- 已核对 `op_kernel/baseline`：
  - `C16_FP32/FP16` 主流程调用 `NeumannMmulAddCompute()`，内部 `AICProcess(INVWsGm_, power, INVWsGm_, ..., enableAtomicAdd=true)`。
  - `C64_1616_FP32/FP16` 设置 `RAW_MMAD_USE_MATMULIMPL_FALLBACK=true`，`InvAdd16()` 回到 MatmulImpl atomic add。
  - `C64_3232_FP32/FP16` 的 16/32 子块 Neumann round 使用 `AICBlockProcess(..., true)` 完成 atomic add。
- 已拆分 `op_kernel/baseline`、`op_kernel/halflocal`、`op_kernel/fulllocal`，并通过 `op_kernel/selected/stage1_selected.h` 和 `selected/stage1_selected_config.h` 控制当前编译版本。
- 已确认 `op_kernel` / `op_host` / `tests/pytest` 中无 `CVInverse` / `cvinverse` 命名残留；历史 docs/AGENTS 中旧记录仍保留历史语境。
- 修复 `halflocal` 和 `fulllocal` 的 C16 文件：删除从 C64_3232 复制来的 `Compose64From32()`、`Compose32From16()`、`Inverse32By16Compose()` 与重复 `C16InverseBy16x16Neumann()`，仅保留单 16x16 raw/L0C 累加路径；补 `INVERSE_TAIL16_SHAPE = INVERSE_SHAPE` 供 raw helper 编译使用。
- 修复 `tests/pytest/run_stage1_variant_matrix.sh`：source CANN custom env 前临时关闭 `set -u`，避免 `ASCEND_CUSTOM_OPP_PATH` 未预置时中断。

## 精度验证结果

- baseline `3*2` 全部构建、安装、三方精度通过：
  - `C16_FP32`、`C16_FP16`、`C64_1616_FP32`、`C64_1616_FP16`、`C64_3232_FP32`、`C64_3232_FP16`。
  - 每个 variant 均执行 `bash tests/pytest/test_run.sh compare`，结果均为 `20 passed`。
- halflocal `3*2` 全部构建、安装、三方精度通过：
  - `C16_FP32`、`C16_FP16`、`C64_1616_FP32`、`C64_1616_FP16`、`C64_3232_FP32`、`C64_3232_FP16`。
  - 每个 variant 均执行 `bash tests/pytest/test_run.sh compare`，结果均为 `20 passed`。
- fulllocal 目录已创建并保留对应文件，当前没有作为精度/performance 交付目标；后续继续用于 L0C->L1/TSCM resident PoC。

## S=16/64/125/2048 性能结果

- original normal 只 profile 一次，作为所有 baseline/halflocal variant 的共同对照：`tests/pytest/prof_out/stage1_original_s_selected_20260505/`。
- 每个 baseline/halflocal variant 只 profile `ChunkGDRNeumann`，再用 normal root 合并生成 `performance_compare.csv`。
- 汇总 CSV：`tests/pytest/prof_out/stage1_baseline_halflocal_s_selected_summary_20260505.csv`。
- 文档表：`docs/2026-05-05/BASELINE_HALFLOCAL_PERFORMANCE_SUMMARY.md`。
- 结果概要（Neumann/normal task duration ratio）：
  - baseline/C16_FP32: `1.142x / 2.262x / 2.491x / 4.117x`。
  - baseline/C16_FP16: `1.103x / 2.213x / 2.480x / 4.110x`。
  - baseline/C64_1616_FP32: `5.267x / 31.423x / 43.902x / 133.261x`。
  - baseline/C64_1616_FP16: `5.254x / 31.369x / 43.919x / 133.273x`。
  - baseline/C64_3232_FP32: `1.290x / 1.967x / 2.237x / 2.847x`。
  - baseline/C64_3232_FP16: `1.303x / 1.933x / 2.266x / 2.846x`。
  - halflocal/C16_FP32: `1.038x / 2.171x / 2.382x / 4.054x`。
  - halflocal/C16_FP16: `1.089x / 2.131x / 2.392x / 4.089x`。
  - halflocal/C64_1616_FP32: `5.240x / 31.341x / 43.869x / 133.229x`。
  - halflocal/C64_1616_FP16: `5.215x / 31.325x / 43.870x / 133.238x`。
  - halflocal/C64_3232_FP32: `1.305x / 1.651x / 1.873x / 2.332x`。
  - halflocal/C64_3232_FP16: `1.383x / 1.647x / 1.864x / 2.319x`。
- 当前最佳被测方案是 `halflocal/C64_3232_FP32`，四点平均 `1.790x`；normal 在所有被测点仍更快。
- `halflocal/C16_FP16/S=2048` 首次 msprof warmup 触发一次 `507015` L0A read/write conflict；单独重跑成功，retry 结果已合入最终表。

## 本轮状态更新与 fulllocal 单点计划

- 用户要求先更新当前状态并写好后续计划，同时解释 `C64_1616` 为什么相比其他方案慢得多。
- 重新核对当前 selector：
  - `op_kernel/selected/stage1_selected_config.h`：`CGDR_NEUMANN_SELECTED_SUITE="halflocal"`，`CGDR_NEUMANN_SELECTED_VARIANT="C64_3232_FP32"`，chunk size `64`。
  - `op_kernel/selected/stage1_selected.h` include `../halflocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_halflocal.h`。
- 后台任务复核：当前无仍在运行的 `build.sh`、`pytest`、`msprof`、`opc`、`ninja/gmake/cmake` 或 variant matrix 任务。
- source stage1 `.h` 命名复核：`op_kernel` / `op_host` / `tests/pytest` 中无 `CVInverse` / `cvinverse` 残留；历史文档中的旧名字只保留历史语境。
- 当前交付状态：
  - baseline `3*2` 已全量精度通过。
  - halflocal `3*2` 已全量精度通过。
  - `S=16/64/125/2048` 的 baseline/halflocal/original normal 性能表已生成。
  - fulllocal 目前只是候选目录和 PoC 文件，尚未形成通过精度的 suite。
- 新增计划文档：`docs/2026-05-05/FULLLOCAL_SINGLE_POINT_BREAKTHROUGH_PLAN.md`。
- 后续单点选择 `fulllocal/C64_3232_FP16`：
  - 先做 16x16 `L0C -> Fixpipe CFG_NZ -> half/bfloat16 L1/TSCM -> LoadData -> MMAD` 布局探针。
  - 探针通过后接入 `validLen <= 16` 的 Neumann 路径。
  - 再扩到 32x32 compose 和 C64 full path。
  - 单点 compare 通过前不复制到 `3*2`，也不改 baseline/halflocal。

## C64_1616 性能异常分析

- 最新四点数据中，`halflocal/C64_1616_FP32` 的 Neumann/normal task ratio 为 `5.240x / 31.341x / 43.869x / 133.229x`，而 `halflocal/C64_3232_FP32` 为 `1.305x / 1.651x / 1.873x / 2.332x`。
- profiling 指标说明这不是单纯 MAC 算多：
  - `C64_1616_FP32` 的 AIC MAC ratio 仅约 `1.087x / 1.609x / 1.609x / 1.224x`。
  - 但 task duration ratio 达到 `5.240x / 31.341x / 43.869x / 133.229x`。
  - `S>=64` 时 AIC scalar 约 `6.7x~7.5x`，AIV scalar/等待侧指标被放大到 `131x+`、`148x+`、`412x+`，说明主要时间耗在调度、同步、等待和 tiny call 固定开销上。
- 代码侧原因：
  - `C64_1616` 一个 C64 chunk 会执行 4 个 `Neumann16InverseBlock()`，再做 2 次 `Compose32From16()` 和 1 次 `Compose64From32()`。
  - 当前 `C64_1616` 仍有保守 fallback：`RAW_MMAD_USE_MATMULIMPL_POWER_FALLBACK=true`、`RAW_MMAD_USE_MATMULIMPL_COMPOSE_FALLBACK=true`、final fallback 也为 `true`。
  - 因此 power/compose 仍反复进入 `MatmulImpl + GM workspace + Fixpipe/MTE2` 链路，叠加 `SyncFixToMte2()`、cache flush、DCCI 等固定成本。
- 结论：`C64_1616` 慢的核心不是 16x16 理论 FLOPs，而是 16x16 子块把 MatmulImpl/raw MMAD 的固定成本按 block 和 round 放大；序列越长，按 chunk 重复次数越多，ratio 越失控。后续追性能应先做 `C64_3232` fulllocal 单点，`C64_1616` 需要先去掉 power/compose fallback 和过量同步后才值得继续比较。

## FP16 与 FP32 性能口径复核

- 用户询问为什么 FP16 比 FP32 慢。
- 复核 `halflocal/C64_3232_FP16` 与 `halflocal/C64_3232_FP32` 的 stage1 文件：
  - `diff -u` 显示两者除文件名和 header guard 外没有实际逻辑差异。
  - 两者核心 MatmulType 仍保留 `float` GM ND 路线，`FP16_L1_RESIDENT_INVERSE=false`，默认没有启用 half L1 resident inverse。
  - final RHS `INV @ k_cumdecay` / `INV @ v_beta` 也仍是 FP32 GM `MatmulImpl`。
- 因此当前 `C64_3232_FP16` 名义上是 FP16 variant，但不是已启用的真实 FP16 求逆计算路径。
- 性能数据也支持这个判断：
  - `halflocal/C64_3232_FP32`：`1.305x / 1.651x / 1.873x / 2.332x`，平均 `1.790x`。
  - `halflocal/C64_3232_FP16`：`1.383x / 1.647x / 1.864x / 2.319x`，平均 `1.803x`。
  - 只有 `S=16` 明显更慢；`S=64/125/2048` 实际略快或基本持平。
- 结论：不能解读为“FP16 算法比 FP32 慢”。当前差异主要来自 msprof 单次测量波动、构建/调度/cache 状态，以及 `S=16` 时固定开销占比过大；真实 FP16 收益要等 half/bfloat16 resident path 打通后再判断。

## FP16 最小修改实验结果

- 用户确认可以先试“简单修改”的 FP16 搬运/累加路径，看能有多少差异。
- 已在 `op_kernel/fulllocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_fulllocal.h` 仅打开 `FP16_L1_RESIDENT_INVERSE=true` 做最小实验。
- 结果：构建通过，custom run 包成功生成并安装。
- 精度：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 失败，20/20 case 全部不通过。
- 结论：单独打开 FP16 resident 开关不能直接得到可用的 FP16 搬运/累加版本；至少还需要补 layout、LoadData 语义、compact offset 或更完整的 resident 链路实现。实验后源码开关已恢复为 `false`，避免保留失败态。

## 最终回归与推送准备

- 按用户确认清理 fulllocal：`op_kernel/fulllocal` 仅保留单点突破文件 `chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_fulllocal.h`，其它未构建/未通过精度的 fulllocal 候选不再保留。
- 当前 selector 已恢复到稳定交付线：`halflocal/C64_3232_FP32`，`CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`。
- 重新构建双算子包通过：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
- 重新安装 custom OPP 到 `/home/developer/Ascend/cann-9.0.0/opp` 成功。
- 稳定线全量三方精度复跑通过：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.15s`。

## 历史 profiling 输出清理

- 按用户追加要求，将历史 tracked `tests/pytest/prof_out` profiling 原始输出删除纳入单独提交。
- 本次只 stage `git add -u tests/pytest/prof_out` 捕获的已跟踪删除，不纳入当前本地未跟踪的新 profiling 目录。
- 目的：远端仓库不再携带早期 `PROF_*` 原始采样目录和旧 `performance_compare.csv/performance_raw.csv`，后续保留精简汇总 CSV 和文档口径。

## selector 回退口径澄清

- 用户询问“回退固定为 selector 切回 `halflocal/C64_3232_FP32`”的含义。
- 已澄清：selector 是编译期入口，具体由 `op_kernel/selected/stage1_selected.h` 和 `stage1_selected_config.h` 控制。
- fulllocal 实验失败时，只切 selector 到 `halflocal/C64_3232_FP32`，再重新 build/install/compare；不是运行时动态切换，也不是修改 baseline/halflocal 文件内容。
- 选择该版本作为固定回退点的原因：它是当前 baseline/halflocal 中综合最好的精度通过稳定版本，S=16/64/125/2048 ratio 为 `1.305x / 1.651x / 1.873x / 2.332x`。
