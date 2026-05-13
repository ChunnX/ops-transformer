# 2026-05-06 工作日志

## FP16 C16/C64_1616 fulllocal 复验

- 用户恢复 `TilingData` 结构体后，当前优化范围收敛到两个目标：`fulllocal/C16_FP16` 与 `fulllocal/C64_1616_FP16`。
- 当前 active selector 停在 `fulllocal/C64_1616_FP16`：
  - `op_kernel/selected/stage1_selected.h` include `../fulllocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocal.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocal"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C64_1616_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`
- 两个 sub-agent 做了只读复核：
  - tiling data 审计结论：恢复后的 `ChunkGatedDeltaRuleNeumannTilingData` 与 host tiling 填充字段一致；切换 C16/C64_1616 时必须同步切 selector include 和 `chunkSize`，否则 host/kernel tiling 会错位。
  - 修改边界审计结论：当前 C16/C64_1616 fulllocal 的有效改动仍集中在 Stage1 inverse；final RHS、`gBK/vBeta`、QK、Stage2、Stage3 未被 resident/raw 化。
- `fulllocal/C16_FP16` 复验：
  - 构建：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16` 通过。
  - 安装：`bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet` 成功。
  - 精度：`CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.01s`。
  - smoke 性能：`tests/pytest/prof_out/fulllocal_c16_fp16_smoke_20260506/performance_compare.csv`，`S=16` Neumann/normal `1.129x`，`S=64` `2.184x`。
- `fulllocal/C64_1616_FP16` 复验：
  - `FP16_L1_RESIDENT_INVERSE=true`，`FP16_L1_RESIDENT_COMPOSE16=true`，`FP16_L1_RESIDENT_COMPOSE32=false`。
  - 构建：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16` 通过。
  - 安装：`bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet` 成功。
  - 精度：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.20s`。
  - after-restore smoke 性能：`tests/pytest/prof_out/fulllocal_c64_1616_fp16_smoke_after_tiling_restore_20260506/performance_compare.csv`，`S=16` Neumann/normal `1.234x`，`S=64` `1.722x`。
- 本次 smoke 没有按单方案放大性能规则裁剪无关 `TilingData/MatmulTiling` 字段；这些数据只作为简单性能观察，不能替代后续裁剪后的正式性能结论。
- 静态扫描确认未重新引入 `FP16_L1_RESIDENT_RHS`、`ResidentInvRhs`、`rawFinalScratch`、`FinalMmad`、`gBKHalf`、`vBetaHalf`、`TailQK`、`ClearQk` 等非求逆链路符号；final RHS 仍为 `AICProcess(INVWsGm_, gBKWsGm_/vBetaWsGm_, ...)`。

## fulllocal 单点恢复与完成

- 用户要求继续 `docs/2026-05-05/FULLLOCAL_SINGLE_POINT_BREAKTHROUGH_PLAN.md`，开 sub-agent 并完成 fulllocal。
- 已启动 sub-agent 做只读复核；结论与本地检查一致：当前 selector 已指向 `fulllocal/C64_3232_FP16`，但 `Compose32From16()` / `Compose64From32()` 的 resident 分支缺失 scalar full-L1 backfill。
- 未修改 `baseline/` 或 `halflocal/`，继续把它们作为稳定参考和回退线。

## 代码修复

- 修复文件：`op_kernel/fulllocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_fulllocal.h`。
- 在 `Compose32From16()` 的 resident 分支中，第二次 `RawMmadBlock()` 写回 lower-left block 后恢复：
  - `CopyGmFloatMatrixToHalfL1(INVWsGm_[MatrixOffset(bottom, top)], aicHalfL1Inv_, ..., bottom, top)`。
- 在 `Compose64From32()` 的 resident 分支中，第二次 `RawMmadBlock()` 写回 64 lower-left block 后恢复：
  - `CopyGmFloatMatrixToHalfL1(INVWsGm_[MatrixOffset(INVERSE_SHAPE, 0)], aicHalfL1Inv_, ..., INVERSE_SHAPE, 0)`。
- 修复原因：删除这两处 backfill 后全量 compare 退化到多 case 失败；恢复后回到此前可通过状态。

## 构建安装

- 构建命令：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 构建结果：成功生成 `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`。
- 安装命令：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 安装结果：`SUCCESS`。

## 精度验证

- `S=16` smoke 通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext python3 tests/pytest/compare_chunk_gdr_three_way.py --seqlen 16 --batch-size 1 --nk 4 --nv 4 --dk 64 --dv 64 --seed 26`
  - 结果：`PASSED`。
- `S=17` smoke 通过：
  - 同脚本，`--seqlen 17`。
  - 结果：`PASSED`，覆盖 `Compose32From16()` 路径。
- 第一轮全量 compare：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`19 passed, 1 failed`，失败 case 为 `b1_t129_tail_d64`，表现为 out 单点大偏移，state 通过。
- 隔离复跑失败 case：
  - `--seqlen 129 --batch-size 1 --nk 4 --nv 4 --dk 64 --dv 64 --seed 33`
  - 结果：`PASSED`。
- 第二轮全量 compare：
  - 同全量 compare 命令。
  - 结果：`20 passed in 6.17s`。

## 性能结论

- fulllocal 已有四个指定点 msprof 结果：
  - `tests/pytest/prof_out/fulllocal_C64_3232_FP16_selected_s_20260506/performance_compare.csv`
- Neumann/normal task duration ratio：
  - `B=8,S=16,D=64,Hq=16,Hkv=4`: `1.307x`
  - `B=8,S=64,D=64,Hq=16,Hkv=4`: `1.957x`
  - `B=8,S=125,D=64,Hq=16,Hkv=4`: `2.286x`
  - `B=8,S=2048,D=128,Hq=16,Hkv=4`: `3.011x`
- 结论：`fulllocal/C64_3232_FP16` 单点精度完成，但性能仍慢于 normal，也没有优于此前最佳 `halflocal/C64_3232_FP32`。

## 阻塞与决策

- direct 32x32 resident 路线不纳入当前完成口径：
  - 直接走 `Neumann32InverseBlock()` 曾在 `S=17` 出现大误差。
  - 尝试 `LoadData2DParamsV2` 后出现 `nan`。
  - 当前稳定主线保留 16x16 resident Neumann + compose。
- 当前 selector 保持 fulllocal：
  - suite: `fulllocal`
  - variant: `C64_3232_FP16`
  - chunk size: `64`
- fulllocal 完成口径：单点实现、构建安装、smoke、全量 compare、指定性能点均已闭环；性能未达突破，作为后续优化问题记录。

## 当前 fulllocal 实现口径复核

- 用户询问当前 fulllocal 如何实现，已按代码重新复核 selector、主流程和关键 helper。
- 当前 `op_kernel/selected/stage1_selected.h` 指向 `fulllocal/C64_3232_FP16`，`FP16_L1_RESIDENT_INVERSE=true`，probe 开关均为 `false`。
- AIV 仍负责生成初始 `L_powWsGm_=L` 与 `INVWsGm_=I-L`；AIC 在 flag `0x7` 后执行 `C64InverseBy32x32Neumann()`。
- fulllocal 的核心是 16x16 resident Neumann：
  - 从 FP32 GM 通过 `RawMmadAccumulateTile(gm, I)` 搬到 L0C，再 `Fixpipe<half,float,CFG_NZ>` 写入 half L1/TSCM compact buffer。
  - 在 half L1/TSCM 中用 `LoadData2D -> L0A/L0B -> Mmad -> L0C -> Fixpipe half L1` 做 `L^2/L^4/L^8` 和 `INV += INV@P`。
  - `INV += INV@P` 不是标量加法，而是 `INV@I` 初始化 L0C，再 `INV@P` 用 `cmatrixInitVal=false` 累加到同一 L0C。
  - 最终 16x16 inverse 通过 half L1 `@ I` 写回 FP32 GM，供后续 compose 和 final RHS 使用。
- 32/64 组合当前走 16x16 resident + GM raw compose：
  - `Inverse32By16Compose()` 对两个 16x16 对角块分别 resident 求逆，再 `Compose32From16()` 组合 lower-left。
  - `validLen > 32` 时对第二个 32 block 重复，并通过 `Compose64From32()` 组合 64 lower-left。
  - compose 后保留 `CopyGmFloatMatrixToHalfL1()` backfill；删除该回填会导致 compare 退化。
- final RHS 仍不是 fulllocal：
  - `INV @ k_cumdecay` 和 `INV @ v_beta` 仍走 FP32 GM `AICProcess/MatmulImpl`。
  - 因此当前 fulllocal 完成的是 inverse 子流程的 half L1 resident 化，不是整个 Stage1 全程片上 resident。

## fulllocal 性能不如 halflocal 的原因

- 用户询问为什么 fulllocal 性能还没有 halflocal 好，已对齐四点性能表和代码路径。
- 结论：`S=16` 基本持平；`S>=64` fulllocal 明显慢于 halflocal，主要不是 MAC，而是为了 half L1/TSCM resident 引入了额外的 GM->L0C->half L1、half L1->L0A/B、Fixpipe half L1、identity 构造和 compose backfill。
- 同四点 Neumann/normal ratio：
  - halflocal/C64_3232_FP32：`1.305x / 1.651x / 1.873x / 2.332x`
  - fulllocal/C64_3232_FP16：`1.307x / 1.957x / 2.286x / 3.011x`
- 关键指标：
  - `S=64` AIC scalar：halflocal `303.115us`，fulllocal `533.904us`。
  - `S=125` AIC scalar：halflocal `541.096us`，fulllocal `1009.908us`。
  - `S=2048` AIC scalar：halflocal `8294.395us`，fulllocal `15529.760us`。
- 代码原因：
  - fulllocal 的 `Neumann16InverseBlock()` 每个 16x16 block 都执行 `LoadGmFloatBlockToHalfL1ByMmad()` 两次，把 `L` 和 `INV` 从 FP32 GM 搬到 half L1。
  - 每轮 `HalfL1PowerBlock()` / `HalfL1InvAddBlock()` 都需要 `LoadData2D -> Mmad -> Fixpipe<half,float,CFG_NZ> -> SyncFixToMte1`。
  - `CopyHalfL1MatrixToGmFloatByMmad()` 还要用 half L1 `@ I` 写回 FP32 GM，供 compose 和 final RHS 继续消费。
  - `Compose32From16()` / `Compose64From32()` 当前仍走 GM raw MMAD，并额外执行 `CopyGmFloatMatrixToHalfL1()` backfill；该 backfill 当前对精度稳定必要，但会增加 AIC scalar/DCCI/同步开销。
- 因此当前 fulllocal 不是减少 GM 的纯收益，而是“16x16 inverse 局部 resident + 仍然写回 GM + GM compose/final RHS”，收益被额外 layout 转换和同步抵消并反超。

## 局部 resident 口径解释

- 用户询问“局部 resident 怎么理解”。
- 当前定义：只有某一小段计算链路的中间数据留在片上 L1/TSCM/L0 中连续消费，离开这段链路后仍会写回 GM，并由后续阶段重新从 GM 读取。
- 当前 fulllocal 的局部 resident 范围是单个 16x16 Neumann block 内部：
  - `L/INV` 从 GM 搬到 half L1/TSCM。
  - `power` 和 `INV += INV@P` 在 half L1/TSCM、L0A/L0B、L0C 之间循环。
  - 16x16 block 算完后仍写回 `INVWsGm_`。
- 不属于 resident 的部分：
  - AIV 初始 `L/I-L` 生成仍在 GM。
  - 32/64 compose 仍走 GM raw MMAD。
  - final RHS `INV @ k_cumdecay` / `INV @ v_beta` 仍走 FP32 GM MatmulImpl。
- 因此“局部 resident”不是“全程不出片上”，而是“block 内部短链路不出片上，block 边界和后续消费仍出 GM”。

## C16 复用 fulllocal resident 的判断

- 用户询问当前 fulllocal 是否可以用于 `C16_FP16` / `C16_FP32`，以及 C16 是否就完全 resident。
- 结论：16x16 resident Neumann 内核可以复用到 C16，且 C16 会天然去掉 `Compose32From16()` / `Compose64From32()`，因此比 C64_3232 更接近 resident。
- 但按当前 Stage1 主流程，C16 仍不会自动成为完全 resident：
  - AIV 初始 `L/I-L` 仍写 GM。
  - 16x16 inverse 算完后当前实现仍写回 `INVWsGm_`。
  - final RHS `INV @ k_cumdecay` / `INV @ v_beta` 仍用 `AICProcess(INVWsGm_, ...)` 从 GM 读。
- 因此当前可实现的是 “C16 inverse 子流程完全 resident / 无 compose”，不是 “Stage1 完全 resident”。
- 若要把 C16 做成真正完全 resident，需要让 final RHS 直接消费 half L1/TSCM 中的 `INV`，或至少在写回 GM 前完成 `INV @ gBK` 与 `INV @ vBeta`；这会引入新的 half/FP32 精度和 Matmul tiling 验证。

## C16 fulllocal 迁移执行

- 用户明确要求先把 fulllocal resident inverse 迁移到 `C16` 的 Stage1 实现看收益，`finalRHS` 暂不处理。
- 已开 sub-agent 做只读复核；结论与本地检查一致：
  - 当前 selector 已切到 `fulllocal/C16_FP16`。
  - `final RHS` 仍通过 `AICProcess(INVWsGm_, gBKWsGm_, ...)` 和 `AICProcess(INVWsGm_, vBetaWsGm_, ...)` 从 GM 读取 `INV`。
  - 当前实现只迁移 inverse 子流程，不迁移 RHS 消费端。
- 当前 selector：
  - `op_kernel/selected/stage1_selected.h` include `../fulllocal/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocal.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocal"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C16_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=16`

## C16 fulllocal 实现边界

- 当前实现文件：`op_kernel/fulllocal/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocal.h`。
- 已禁用失败的 `FP32_SCALAR_RESIDENT_INVERSE` 路径，保留 `FP16_L1_RESIDENT_INVERSE=true`。
- `Neumann16InverseBlock()` 当前流程：
  - `PrepareHalfL1IdentityByMmad(INVERSE_SHAPE)` 准备 half L1/TSCM identity。
  - `LoadGmFloatBlockToHalfL1ByMmad()` 分别把 `L_powWsGm_` 和 `INVWsGm_` 的 16x16 block 搬到 half L1/TSCM。
  - `HalfL1NeumannCompactBlock(INVERSE_SHAPE)` 在 half L1/TSCM + L0A/L0B/L0C 内完成 `L^2/L^4/L^8` 与 `INV += INV@P`。
  - `CopyHalfL1MatrixToGmFloatByMmad()` 将最终 inverse 写回 `INVWsGm_`。
- 为避免 raw resident 路径污染后续 MMAD 状态，已在 Stage1 AIC 入口和 resident inverse 前后使用 `PrepareRawMmadState()`。
- `final RHS` 暂不 resident：`INV @ k_cumdecay` 与 `INV @ v_beta` 仍走 FP32 GM MatmulImpl 路径。

## C16 fulllocal 验证

- 构建通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 安装通过：
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 精度通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 5.98s`

## C16 fulllocal 选点性能

- 性能目录：
  - `tests/pytest/prof_out/stage1_fulllocal_C16_FP16_s_selected_20260506/performance_compare.csv`
- 四个选点 Neumann/normal task duration ratio：
  - `B=8,S=16,D=64,Hq=16,Hkv=4`: `1.084x`
  - `B=8,S=64,D=64,Hq=16,Hkv=4`: `2.140x`
  - `B=8,S=125,D=64,Hq=16,Hkv=4`: `2.503x`
  - `B=8,S=2048,D=128,Hq=16,Hkv=4`: `4.076x`
- 与 `halflocal/C16_FP16` 同 shape 对比：
  - `S=16`: fulllocal/halflocal `1.000x`
  - `S=64`: fulllocal/halflocal `1.007x`
  - `S=125`: fulllocal/halflocal `1.011x`
  - `S=2048`: fulllocal/halflocal `0.996x`
- 结论：
  - C16 去掉了 C64 compose 的额外成本，但当前只是 inverse 局部 resident，收益仍被 GM->half L1 搬运、Fixpipe、identity 准备和最终写回 GM 抵消。
  - 相比 C16 halflocal，整体基本持平，未形成明确性能突破。
  - 后续若继续追 C16，需要把 `final RHS` 改为直接消费 resident `INV`，否则 inverse 算完仍要写回 GM，再由 RHS 从 GM 读。

## Fast inverse 参考分析

- 用户提供 `gdn-tri-inverse` fast inverse 文章，要求参考其 Ascend 910B 深度实现，判断当前 `fulllocal` 是否可借鉴。
- 已阅读：
  - `https://github.com/huawei-csl/gdn-tri-inverse/blob/0.1.0/markdown/fast_inverse_blog/fast_inverse_zh.md#deep-dive-on-ascend-910b-implementations`
  - `https://github.com/huawei-csl/pto-kernels/blob/v0.1.2/csrc/kernel/kernel_tri_inv_rec_unroll.cpp`
- 核心判断：
  - fast inverse 的收益来自入口/出口 GM、中间 L1/L0 resident、结构化 diagonal/odd-even block 搬运、`TMATMUL_ACC`、双缓冲和搬运/计算重叠。
  - 当前 `fulllocal` 只完成 16x16 Neumann block 内部的 half L1/TSCM resident，AIV 初始 `L/I-L`、16x16 输出、C64 compose、final RHS 仍大量依赖 GM。
  - 因此当前 `fulllocal` 可以参考 fast inverse，但尚未复现其关键 dataflow。
- 对当前性能问题的判断：
  - 可优化方向与 fast inverse 一致：减少 `MatmulImpl + GM workspace + FIX/MTE2` 串 tiny MatMul 的固定开销。
  - 但只优化 16x16 inverse 内部不足以带来净收益，C16 fulllocal 与 halflocal 基本持平已经验证这一点。
  - 需要把 resident 边界继续向外扩：先做 C16 final RHS resident，再做 C64_3232 compose resident，最后再考虑双缓冲/多 tile pipeline。
- 新增分析文档：
  - `docs/2026-05-06/FAST_INVERSE_REFERENCE_ANALYSIS.md`

## C16 final RHS resident/raw 化

- 用户明确要求下一步必须让 `INV @ gBK` 和 `INV @ vBeta` 直接消费 half L1/TSCM 中的 resident `INV`，即把 final RHS 一起 resident/raw 化。
- 已开 sub-agent 做只读复核；结论与本地检查一致：
  - active selector 为 `fulllocal/C16_FP16/chunkSize=16`。
  - `INV` operand 已经不再从 `INVWsGm_` 读回，而是通过 `aicHalfL1TileInv_` 直接进入 final RHS raw MMAD。
  - RHS operand `gBKWsGm_` / `vBetaWsGm_` 仍来自 FP32 GM workspace，并在 `ResidentInvRhsGm()` 中经 raw MMAD 转成 half L1 tile 后与 resident `INV` 相乘。
- active 实现文件：
  - `op_kernel/fulllocal/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocal.h`
- 关键实现：
  - 启用 `FP16_L1_RESIDENT_RHS=true`，`FP16_L1_WRITEBACK_INV_FOR_DEBUG=false`。
  - `ProcessOneChunk()` 中两路 final RHS 改为：
    - `ResidentInvRhsGm(aicHalfL1TileInv_, gBKWsGm_, outKCumdecayGm_, dk_)`
    - `ResidentInvRhsGm(aicHalfL1TileInv_, vBetaWsGm_, outVInnerGm_, dv_)`
  - `Neumann16InverseBlock()` 在 RHS resident 可用时不再把最终 `INV` 写回 `INVWsGm_`，而是保留在 `aicHalfL1TileInv_` 给两路 RHS 直接消费。
  - 两路 RHS 完成后再执行 `RestoreRawMmadState()` 和 `RestoreStage1MatmulLocalBuffers()`，避免提前 `TPipe::Reset()` 破坏 resident buffer。
  - 新增 `UseResidentRhsForCurrentShape()`：仅当 `dk` 与 `dv` 均为 16 的倍数时启用 RHS resident；否则先写回 `INVWsGm_` 并回退到原 GM `AICProcess` final RHS，避免固定 16 列 tile 处理非对齐维度导致越界或脏列。

## C16 final RHS resident 验证

- 构建通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 安装通过：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 全量三方精度通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 5.96s`
- 非 16 倍数维度回退 smoke 通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext python3 tests/pytest/compare_chunk_gdr_three_way.py --batch-size 1 --seqlen 32 --nk 4 --nv 4 --dk 48 --dv 80 --seed 41`
  - 结果：`PASSED`
- 稳定性哨兵 `b4_t32_h16_h4_d64` 在本轮全量 compare 中通过；此前曾出现一次 out 偶发失败，后续若继续改 resident/raw 路径仍应保留该 case。

## C16 final RHS resident 性能

- 性能目录：
  - `tests/pytest/prof_out/stage1_fulllocal_C16_FP16_rhs_resident_s_selected_20260506/performance_compare.csv`
- 四个选点 Neumann/normal task duration ratio：
  - `S=16`: `1.109x`
  - `S=64`: `2.423x`
  - `S=125`: `2.671x`
  - `S=2048`: `5.026x`
- 对比不做 RHS resident 的 C16 fulllocal：
  - no-RHS-resident ratio：`1.084x / 2.140x / 2.503x / 4.076x`
  - RHS-resident Neumann 绝对耗时相对 no-RHS-resident：`1.018x / 1.091x / 1.077x / 1.244x`
- 对比 `halflocal/C16_FP16`：
  - halflocal ratio：`1.089x / 2.131x / 2.392x / 4.089x`
- 当前结论：
  - 功能上已经做到 final RHS 的 `INV` operand 直接消费 half L1/TSCM resident `INV`。
  - 性能没有收益，反而更慢；主要新增成本来自 RHS FP32 GM workspace 到 half L1/TSCM 的 `LoadGmFloatTileToHalfL1ByMmad()` 转换，以及随之增加的 AIC scalar、MTE1/MTE2、Fixpipe 开销。
  - 这说明只把 `INV` operand resident 化还不够；若要继续追收益，需要减少 RHS operand 的 GM->half L1 转换成本，或者让 `gBK/vBeta` 在生成端也直接产出可被 raw MMAD 消费的 half L1/TSCM tile。

## RHS 转换口径澄清

- 用户追问为什么要用 MMAD 做转换，搬运时是否可以顺便转换。
- 澄清：
  - 当前所谓 `FP32 GM -> half L1/TSCM` 转换不是单纯搬运，而是同时做 dtype cast 和 cube 可消费的 L1 NZ layout materialize。
  - 普通 MTE/DataCopy 搬运可以搬字节、做 padding，部分 TLoad 路径可做 ND2NZ layout，但不能直接把 FP32 GM 数值 cast 成 half L1/TSCM operand。
  - Vector `Cast` 可以做 FP32->half，但结果在 UB/ND 语义下，不能直接得到当前 `LoadData2D` 到 L0A/L0B 所需的 half L1 NZ tile；而 final RHS 发生在 AIC 分支，跨 AIV/UB 后仍需要同步和中转。
  - 当前实现用 `RHS_float_GM @ I_float_GM -> L0C_float`，再 `Fixpipe<half, float, CFG_NZ>` 写 half L1/TSCM，是为了复用已验证的 L0C->L1 dtype+layout 写回能力。
- 结论：MMAD 转换是当前可工作的绕路，不是最终理想路径；更好的方向是让 `gBK/vBeta` 生成端直接产出 half L1/TSCM/NZ 可消费 tile，或找到支持 FP32 GM -> half L1 NZ 的直接 TLoad/搬运接口。

## C64 fast inverse 适配计划

- 用户要求基于 fast inverse 的 diagonal / odd-even block packing 思路写一个适配 C64 的计划。
- 新增计划文档：
  - `docs/2026-05-06/C64_FAST_INVERSE_ADAPTATION_PLAN.md`
- 计划口径：
  - 优先适配 `C64_3232_NeumannCompose`，暂不优先 `C64_1616`。
  - 不直接复制 C16 RHS resident；C16 final RHS resident 已验证功能可行但性能更慢，根因是 RHS FP32 GM -> half L1/TSCM 转换成本。
  - C64 首要目标是消除 `Compose32From16()` / `Compose64From32()` 的 GM raw MMAD 与 scalar full-L1 backfill。
- 分阶段安排：
  1. 审计 64x64 full half L1/TSCM block offset。
  2. 适配 diagonal 16x16 fractal/block packing。
  3. resident 化 `Compose32From16()`。
  4. resident 化 `Compose64From32()`。
  5. final writeback 先保守回 GM，RHS resident 后置。
  6. 在 compose resident 有收益后，再尝试 packed diagonal 16x16 Neumann round 批处理。
- 验收口径：
  - 每阶段先跑 `S=16/17/33/64/125` smoke。
  - 全量 `CGDR_GOLDEN_CHUNK_SIZE=64 ... test_run.sh compare` 必须 `20 passed`。
  - 性能至少不慢于当前 `fulllocal/C64_3232_FP16`，目标接近或优于 `halflocal/C64_3232_FP32`。

## C64 当前代码优化复核

- 用户要求继续看当前 C64 代码，考虑是否还有其它收益点，例如 ping-pong 搬运 C64 矩阵。
- 复核文件：
  - `op_kernel/fulllocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_fulllocal.h`
  - `op_kernel/halflocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_halflocal.h`
  - `op_kernel/halflocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_halflocal.h`
  - `op_kernel/selected/stage1_selected.h`
  - `op_kernel/selected/stage1_selected_config.h`
- 最新 selector 复核：当前默认入口为 `halflocal/C16_FP16`、`chunkSize=16`；今天 selector 曾多次切换，后续测试前必须重新确认真实入口。
- 关键发现：
  - 当前 C64 文件已经有 `HalfL1ComposeBlock()`，`Compose32From16()` / `Compose64From32()` 的 resident 分支调用该 helper。
  - 但 `HalfL1ComposeBlock()` 仍以 GM block 为输入输出：`lowerGm/topDiagGm/bottomDiagGm -> half L1 tile -> GM dstGm`。
  - 因此当前不是 full C64 matrix resident，而是每个 block 局部 resident。
  - `aicHalfL1Inv_` / `aicHalfL1Power_` 两个 64x64 full half L1 buffer 已分配，但主路径没有用它们保存 C64 全矩阵状态。
  - `aicHalfL0AAlt_` / `aicHalfL0BAlt_` 已存在，但当前只是顺序使用两套 L0 buffer，没有真正做到 MTE1 搬运与 MMAD 的异步 overlap。
- 对 FP32/FP16 C64 文件的口径：
  - `C64_3232_FP32_halflocal` 的 `FP16_L1_RESIDENT_INVERSE=false`，仍是稳定 GM/raw MMAD 基线。
  - `C64_3232_FP16_halflocal` 的 `FP16_L1_RESIDENT_INVERSE=true`，但只是 block 局部 half L1 resident，仍有 GM 输入/输出边界。
- 收益判断：
  1. 优先做 identity cache，避免每个 16/32 block 重复 `PrepareHalfL1IdentityByMmad()`。
  2. 其次做 C64 full half L1 staging，把 `INV` / `power` 的 C64 状态保留在 `aicHalfL1Inv_` / `aicHalfL1Power_`，compose 从 full L1 取 block，不再 GM 往返。
  3. 再做 local compose：把 `HalfL1ComposeBlock()` 的输入/输出从 GM block 改成 full half L1 block。
  4. 真正的 L0 ping-pong overlap 后置，因为当前 helper 内部同步很重，且主要成本仍是 GM->half L1 转换、identity、Fixpipe/MTE。
  5. packed diagonal 16x16 批量 MMAD 有潜在收益，但 layout 风险更高，应在 full L1 staging/local compose 后做。
- 新增复核文档：
  - `docs/2026-05-06/C64_CURRENT_CODE_OPTIMIZATION_REVIEW.md`

## C64 第一阶段优化口径修正

- 用户明确：其它收益点可以作为后续优化点，第一步必须按网页 fast inverse 方法进行优化。
- 已将 `docs/2026-05-06/C64_FAST_INVERSE_ADAPTATION_PLAN.md` 的第一阶段修正为网页 MXR 主线：
  1. 先做 `CopyDiagonalFractalsL1ToL0()` 对应的 diagonal fractal/block packing。
  2. 对 C64 内四个 `16x16` 对角块批量执行 MCH/Neumann inverse trick，减少逐块 tiny MMAD 调度。
  3. 结果写回 C64 full half L1/TSCM 对角位置。
  4. 第二阶段再做 `CopyOddOrEvenBlocksL1ToL0()` 对应的 odd/even block packing 和 MBH compose。
- 明确后续优化点：
  - `identity cache`
  - full L1 staging 细节优化
  - 真正 L0A/L0B ping-pong overlap
  - RHS resident
  - 减少 `TPipe::Reset()` / Matmul re-init
- 当前决策：第一步不再写成 identity cache 或泛化 ping-pong，而是网页方法的 diagonal packing + packed 16x16 inverse。

## 最新回退状态：只保留求逆修改

- 用户明确要求不要扩大修改面，参考 `op_kernel/origin/chunk_gated_delta_rule_neumann_stage1.h`，把前面误扩到 final RHS、`gBK/vBeta`、QK 等链路的改动退回；当前只保留求逆相关改动。
- 当前真实 selector 已恢复为稳定精度线：
  - `op_kernel/selected/stage1_selected.h` -> `../halflocal/chunk_gated_delta_rule_neumann_stage1_C16_FP16_halflocal.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="halflocal"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C16_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=16`
- 静态残留扫描无命中：
  - `FP16_L1_RESIDENT_RHS`
  - `ResidentInvRhs`
  - `UseResidentRhsForCurrentShape`
  - `rawFinalScratch`
  - `FinalMmad64ByN`
  - `TailQKComputeAiv`
  - `ClearQkMatrix`
  - `gBKHalf`
  - `vBetaHalf`
- final RHS 已恢复为原稳定 FP32 GM MatmulImpl 消费端：
  - `AICProcess(INVWsGm_, gBKWsGm_, outKCumdecayGm_, ...)`
  - `AICProcess(INVWsGm_, vBetaWsGm_, outVInnerGm_, ...)`
- `C16_FP16` 当前是真 FP16 inverse 路线：`FP16_L1_RESIDENT_INVERSE=true`，通过 `HalfL1NeumannCompactBlock()` 在 half L1/TSCM + L0A/L0B/L0C 内完成求逆子流程。
- 当前安装包和 selector 已恢复到 `halflocal/C16_FP16`；全量精度结果为：
  - `CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.03s`
- 额外验证情况：
  - `baseline/C64_1616_FP16`、`halflocal/C64_1616_FP16`、`baseline/C64_3232_FP16`、`halflocal/C64_3232_FP16`、`baseline/C16_FP16`、`halflocal/C16_FP16` 均已完成构建检查。
  - `halflocal/C64_3232_FP16` 曾临时切换并安装测试，`CGDR_GOLDEN_CHUNK_SIZE=64` 全量 compare 结果为 `15 failed, 5 passed`；失败集中在需要完整 C64 compose 的 case，短尾/小块部分可过。
- 当前判断：
  - C16 FP16 inverse 已可作为当前可交付的精度通过线。
  - C64 FP16 compose 仍有布局/同步/数学顺序问题，不能作为默认 selector。
  - 下一步若继续 C64，应只查 `HalfL1ComposeBlock()`、`Compose32From16()`、`Compose64From32()`，不能再改 final RHS 或 `gBK/vBeta`。

## 状态矩阵澄清

- 当前 active / installed：
  - `halflocal/C16_FP16`，`chunkSize=16`。
  - 构建、安装、全量 compare 已通过：`20 passed in 6.03s`。
- 已通过精度：
  - `halflocal/C16_FP16`：通过，是真 FP16 inverse 路径，final RHS 仍为 FP32 GM。
- 已构建但未作为通过线：
  - `baseline/C16_FP16`
  - `baseline/C64_1616_FP16`
  - `halflocal/C64_1616_FP16`
  - `baseline/C64_3232_FP16`
  - `halflocal/C64_3232_FP16`
- 已明确失败：
  - `halflocal/C64_3232_FP16`：构建和安装可过，但 `CGDR_GOLDEN_CHUNK_SIZE=64` compare 为 `15 failed, 5 passed`；失败集中在完整 C64 compose。
- 尚未重新验收：
  - `C64_1616_FP16` baseline/halflocal 在本次 true FP16 inverse 修改后只确认构建，不计入精度通过。
  - `baseline/C16_FP16` 尚未作为最终 selector 全量 compare，因此不计入当前安装通过线。
- 本次求逆相关修改：
  - FP16 文件启用 `FP16_L1_RESIDENT_INVERSE=true`。
  - 增加 `LoadGmFloatBlockToHalfL1ByMmad()`：FP32 GM block 经 raw MMAD 和 `Fixpipe<half, float, CFG_NZ>` 进入 half L1/TSCM。
  - 修正 `CopyHalfL1MatrixToL0A/B()`：按 16x16 fractal block 循环 `LoadData`，而不是把 half L1 当连续平铺矩阵直接加载。
  - 增加 `HalfL1NeumannCompactBlock()`：在 compact half L1/TSCM tile 内完成 `L^2/L^4/L^8` 和 `INV += INV@P`。
  - C16 路径：`L/INV` 从 GM 进入 half L1/TSCM，求逆完成后写回 `INVWsGm_`，final RHS 继续原 FP32 GM `AICProcess`。
- C64_3232 路径：额外增加 `HalfL1ComposeBlock()`，尝试把 compose 的 block 乘法也改到 half L1/TSCM；该路径目前精度未通过。

## TilingData 性能测试规则补充

- 用户指出当前 `TilingData` / `MatmulTiling` 结构体过多，会影响单个方案放大性能测试的口径。
- 已将规则写入 `AGENTS.md`：测试单个方案的性能时，必须注释或裁剪其它方法的 tiling data 结构体，只保留当前 selector 实际使用的字段。
- 后续跑单方案性能前必须同步检查：
  - `op_kernel/*_tiling_data.h`
  - `op_host/*_tiling.cpp`
  - workspace size
  - kernel constructor 参数
  - `stage1_selected_config.h`
- 性能结论必须记录当时保留了哪些 TilingData 结构体；未裁剪无关结构体的数据不能作为可信性能结论。

## FP16 精度问题定位补充

- 用户指出 FP16 精度绝对可以通过，要求明确当前问题出现在哪里。
- 当前判断：问题不是 FP16 数值精度天然不行，而是 `C64_3232_FP16` 的实现点有误。
- 已通过路径：
  - `C16_FP16`：单个 16x16 half L1/TSCM resident inverse，全量 compare 通过。
  - `C64_3232_FP16` 的短尾/小块 case 有通过，说明 16x16 resident inverse 主体可用。
- 失败集中点：
  - `C64_3232_FP16` 的完整 C64 compose，尤其 `Compose64From32()` 调用的 `HalfL1ComposeBlock(..., blockSize=32)`。
- 关键嫌疑：
  1. `HalfL1ComposeBlock()` 中 32x32 compact NZ half L1 tile 通过 `CopyHalfL1MatrixToL0A/B()` 进入 L0A/L0B 时，当前 `LoadData2DParams` 对 32x32 多 fractal tile 的 A/B 解释可能不对；16x16 退化场景能过，32x32 会暴露布局问题。
  2. `C64_3232_FP16` 的 `LoadGmFloatBlockToHalfL1ByMmad()` 当前缺少和 `C16_FP16` 一致的 `CleanFloatMatrixCacheLines(gm, ...)`，compose 读取刚写回的 GM block 时可能存在可见性风险。
  3. `HalfL1ComposeBlock()` 的数学顺序与旧 FP32 compose 一致，仍是 `lower @ topDiag` 后 `bottomDiag @ tmp`，因此公式本身不是第一嫌疑。
- 下一步建议：
  - 只回退 `Compose64From32()` 为旧 `RawMmadBlock()`，保留 16x16 FP16 inverse，先确认全量 compare 是否恢复。
  - 若恢复，说明问题严格在 32x32 half L1 compose。
  - 再分别验证 `LoadGmFloatBlockToHalfL1ByMmad()` 32x32 roundtrip、`HalfL1PowerBlock()` 32x32、`CopyHalfL1MatrixToGmFloatByMmad()` 32x32。

## C64_3232_FP16 halflocal 修复验证

- 按二分方案修改 `op_kernel/halflocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_halflocal.h`：
  - `LoadGmFloatBlockToHalfL1ByMmad()` 增加 `CleanFloatMatrixCacheLines(gm, blockSize, blockSize, chunkSize_)`，保证从 GM 读刚写回 block 前先清理/失效 cache。
  - `Compose32From16()` 回退为旧 `RawMmadBlock()` 两段 compose。
  - `Compose64From32()` 回退为旧 `RawMmadBlock()` 两段 compose。
  - 保留 `FP16_L1_RESIDENT_INVERSE=true` 和 16x16/32x32 diagonal inverse 的 half L1/TSCM resident 实现。
- selector 已切到：
  - `halflocal/C64_3232_FP16`
  - `chunkSize=64`
- 构建通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 安装通过：
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 全量三方精度通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.27s`
- 结论：
  - FP16 inverse 本体可以通过。
  - 之前 `15 failed, 5 passed` 的问题来自 half L1/TSCM compose 路径，不是 FP16 求逆精度本身。
  - 当前通过版本仍不是 full FP16 resident compose；compose 暂时回退 GM/raw MMAD，后续若继续优化 compose，应单独验证 32x32 half L1 tile 的 A/B LoadData 布局。

## 16:33 C64_3232_FP16 half L1 compose 重新打开并通过

- 当前 selector：
  - `op_kernel/selected/stage1_selected.h` -> `../halflocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_halflocal.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="halflocal"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C64_3232_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`
- 本轮只修改求逆/compose 路径，未改 final RHS、`gBK/vBeta`、QK、Stage2、Stage3。
- 关键代码修改：
  - `FP16_L1_RESIDENT_INVERSE=true` 保持 FP16 half L1/TSCM inverse。
  - 新增独立开关 `FP16_L1_RESIDENT_COMPOSE16=true`、`FP16_L1_RESIDENT_COMPOSE32=true`，避免用 inverse 总开关同时控制 compose，便于二分。
  - `Compose32From16()` 在开关打开时调用 `HalfL1ComposeBlock(..., blockSize=16)`。
  - `Compose64From32()` 在开关打开时调用 `HalfL1ComposeBlock(..., blockSize=32)`。
  - `LoadGmFloatBlockToHalfL1ByMmad()` 保留 `CleanFloatMatrixCacheLines(gm, blockSize, blockSize, chunkSize_)`，读取刚由 Fixpipe/GM 写回的 compose block 前清理/失效 cache。
- 验证过程：
  - 默认关闭 compose probe 时，构建安装后 `CGDR_GOLDEN_CHUNK_SIZE=64 ... test_run.sh compare` 为 `20 passed in 6.28s`，确认诊断开关不改变稳定线。
  - 只打开 `FP16_L1_RESIDENT_COMPOSE16` 后，`S=31` 和 `S=64` CLI smoke 均 PASS，说明 16x16 half L1 compose 可用。
  - 继续打开 `FP16_L1_RESIDENT_COMPOSE32` 后，`S=64` CLI smoke PASS。
  - 最终全量三方精度：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.27s`。
- 最新结论：
  - `halflocal/C64_3232_FP16` 现在不是 raw compose 回退线，而是 FP16 inverse + FP16 half L1 compose 通过线。
  - 之前 `15 failed, 5 passed` 不代表 FP16 精度不行；当前验证表明在修正 half L1/L0 fractal copy 并保证 GM cache 可见性的基础上，完整 C64 compose 也能通过。
  - final RHS 仍保持原 FP32 GM `AICProcess(INVWsGm_, gBKWsGm_/vBetaWsGm_, ...)`，本轮未重新引入 RHS resident。
- 本轮未跑 msprof；若后续测单方案放大性能，需先按 AGENTS 规则裁剪无关 TilingData/MatmulTiling，并记录保留字段。

## fulllocalnew 直接搬运计划

- 用户指出 `attention/chunk_gated_delta_rule` 新代码已大改，性能约为原 normal 写法的 `1.02x~1.5x`，建议 Neumann 侧不要在旧 Stage1 上继续补丁式移植，而是先直接搬运新代码，再在新代码基础上实现 FP16 Neumann fulllocal。
- 已用 `git diff -- attention/chunk_gated_delta_rule` 核对 normal 新代码改动范围：1 个 host tiling cpp 和 5 个 kernel header。
- 新增计划文档：`docs/2026-05-06/FULLLOCALNEW_DIRECT_MIGRATION_PLAN.md`。
- 计划口径：
  - 迁移 normal 新 BF16 主链路到 Neumann wrapper、Stage2、Stage3、host tiling。
  - 新 Stage1 只放入 `op_kernel/fulllocalnew/`，实现 `C16_FP16` 与 `C64_1616_FP16`。
  - 不在算子头尾做全局 dtype 转换；只在 Stage1 inverse 局部做 BF16/FP32/FP16 桥接。
  - final RHS 先使用新代码 BF16 Matmul，不做 RHS resident/raw 化。
  - `baseline/`、`halflocal/`、现有 `fulllocal/` 不作为本轮实验落点。

## fulllocalnew/C16_FP16 实现进度补充

- `fulllocalnew/C16_FP16` 已经从计划进入实现调试阶段，当前 selector 指向：
  - `op_kernel/selected/stage1_selected.h` -> `../fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocalnew.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocalnew"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C16_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=16`
- 已完成构建安装：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 安装后全量 compare 当前结果为 `16 passed, 4 failed`。
- 剩余失败均表现为 `out` 失败、`state` 通过；因此当前调试重点从 inverse 本体转向 Stage1/Stage3 输出链路。
- 失败 case：
  - `b4_t32_h16_h4_d64`
  - `b2_t96_h8_h2_d128`
  - `b1_t257_tail_d128`
  - `b2_t160_h32_h8_d64`
- 下一步优先做失败 case 复跑和 `value=0` 隔离，再决定是否调整 `outQkGm_` cache flush 或 `qPrime/vInner` 链路。
