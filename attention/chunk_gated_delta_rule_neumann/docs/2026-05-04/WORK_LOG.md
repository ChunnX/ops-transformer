# 2026-05-04 工作日志

## 本轮目标

- 用户要求确认当前进度，并继续实现计划。
- 先核对真实代码状态、已有性能数据和当日文档，避免继续沿错误状态推进。

## 状态核对

- 当前工作目录：`/mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann`
- host 主 chunk 仍为 `chunkSize=64`，对应 C64 子块方案。
- 接手时 active selector 指向 `C64_1616_FP32_CVInverse` raw 探针线：
  - `RAW_MMAD_USE_MATMULIMPL_FALLBACK = false`
  - `RAW_MMAD_USE_MATMULIMPL_POWER_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_COMPOSE_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_FINAL_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK = true`
  - `RAW_MMAD_INVADD_USE_POWER_PLUS_IDENTITY = true`
  - `RAW_MMAD_INVADD_POST_STORE_SYNC_MODE = 2`
- 该线已通过精度，但性能探针不可接受：
  - `chunk_gdr_c64_1616_raw_invadd_power_plus_i_probe`
  - `S=64` Neumann/normal `31.405x`
  - `S=512` Neumann/normal `129.931x`
- 当日旧文档一度写成“power/compose/final 均 raw 且性能待测”，与真实源码开关和 profiling 结论不一致，需要纠正。

## 决策

- 停止继续做 `C64_1616` raw inv-add 的同步、DCCI、identity 写入等微调。
- 原因：
  - `INV@(I+P)`、减少 DCCI、减少 identity 写入等探针没有改变主耗时。
  - profiling 中 wall time 主要表现为巨大 AIV scalar/wait 时间和固定调度开销，继续微调 raw inv-add 不会解决主瓶颈。
- 默认实现切回 `C64_3232_NeumannCompose` 的 hybrid 线：
  - 该线已有 32 shape 全量 msprof 结果，是当前三种最终方案里最好的 C64 版本。
  - 后续结构性优化也优先基于 `C64_3232` 做 local/TSCM PoC，避免在 `C64_1616` 的高子块调用数上继续放大固定开销。

## 代码调整

- 修改 `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`：
  - 默认 include 从 `C64_1616_FP32_CVInverse` 切到 `C64_3232_FP32_CVInverse`。
- 修改 `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h`：
  - 补齐与主 kernel 构造函数兼容的 FP16 Matmul 类型别名。
  - 增加忽略额外 Matmul 参数的 `Stage1` 构造重载，保证主 kernel 的统一构造路径可编译。
- 未删除 `C64_1616` raw MMAD 探针代码；该线保留作后续参考和单独实验，不作为默认实现。

## 验证

- 构建：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - 结果：通过，生成 `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`
- 安装：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 结果：`SUCCESS`
- 全量三方精度：
  - `source /home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer/bin/set_env.bash`
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.17s`

## 当前结论

- 当前默认实现已恢复为 `C64_3232_NeumannCompose` hybrid 线。
- 当前精度无阻塞项。
- `C64_1616` raw inv-add 探针线保留，但不再作为继续优化主线。
- 下一步不是继续 raw inv-add 微调，而是做 `C64_3232` local/TSCM PoC，目标是减少 `MatmulImpl + GM workspace + FIX/MTE2` 串联开销。

## 下一步计划

- 新增计划文档：`docs/2026-05-04/C64_3232_LOCAL_TSCM_POC_PLAN.md`
- 计划优先级：
  1. 以当前默认 `C64_3232_FP32_CVInverse` 为稳定 baseline。
  2. 新增独立 local/TSCM PoC selector，不直接污染默认实现。
  3. 先用已有 half/TSCM 本地 Matmul 范式验证片上中转收益。
  4. 精度通过后只跑少量 msprof probe；probe 有收益再跑 32 shape 全量性能。

## 本轮继续实现记录

- 接手时 selector 实际指向 `chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_LocalTscm_CVInverse.h`，不是稳定 `C64_3232_FP32_CVInverse`。
- `C64_3232_FP16_LocalTscm_CVInverse` 已经有初步 PoC 代码，但构造函数只接收 9 个 Matmul 参数；主 kernel 当前统一传入 10 个参数，缺少额外 `StageOneMT &stage1MT_C64_` 兼容位。
- 本轮修改：
  - 在 `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_LocalTscm_CVInverse.h` 的 `Stage1` 构造函数中增加一个忽略的 `StageOneMT &` 参数，适配主 kernel 10 参数构造。
  - 保留 PoC 文件，不删除，以便后续继续调试 local/TSCM 路线。
- PoC 构建安装验证：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 结果：构建和安装通过。
- PoC 精度验证：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - 结果：`17 failed, 3 passed in 6.30s`
  - 通过 case：短尾/小块相关 case 仍可落在 `0.08/0.08` 阈值内。
  - 失败特征：`C=64` 主路径和多 batch/GQA/长序列 case 大面积失败，默认 case out/state 出现约 `2.0+` 量级 max_abs，说明 PoC 不能进入性能 probe。
- 初步判断：
  - 失败集中在 `C64_3232` local/TSCM inverse/compose 主路径。
  - 主要疑点是把 64x64 GM 矩阵整体 `Nd2Nz` 到 TSCM 后，再用 32x32 local Matmul 直接按子块 offset 读取，和 local Matmul 对 compact 32x32 NZ tile 的布局期望不一致。
  - 另一个风险是 32x32 half local Neumann 相比已验证过的 16x16 local/TSCM 路线数值误差更大。
- 稳定线恢复：
  - selector 已切回 `#include "chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h"`。
  - PoC include 改为注释，作为独立实验入口保留。
- 稳定线最终验证：
  - 重新构建、安装后运行 `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.33s`

## 本轮结论

- 当前默认实现保持稳定 `C64_3232_FP32_CVInverse`，精度通过。
- `C64_3232_FP16_LocalTscm_CVInverse` 已推进到“可编译、可运行”，但精度未过，不应跑 msprof 性能。
- 后续继续 local/TSCM 时，应先做 compact 32x32/16x16 TSCM tile 搬运与子块读取校验，或从已验证过的 `C64_1616_FP16_CVInverse` 16x16 local/TSCM 路线逐步升级，而不是直接对 64x64 NZ 大矩阵按 32x32 offset 做 local matmul。

## 状态确认

- 用户询问当前情况。
- 当前默认 selector 仍应保持 `C64_3232_FP32_CVInverse`，这是最后一次已验证通过的稳定线。
- `C64_3232_FP16_LocalTscm_CVInverse` 仅作为 PoC 保留，当前阻塞在精度，不能进入 msprof 性能测试。
- 下一步实现重点不再是 raw inv-add 微调，而是先验证 compact TSCM/NZ 子块布局，再恢复 local/TSCM inverse/compose。

## local/TSCM 累加位置说明

- 用户询问 local/TSCM 累加在哪里。
- `C64_3232_FP16_LocalTscm_CVInverse` PoC 的 local/TSCM Matmul 在 `AICBlockProcessFp16L1()`：
  - 输入输出均为 `LocalTensor<half>`，Matmul 类型是 `StageOneFp16L1MT`。
  - 通过 `mm.IterateAll(z, enableAtomicAdd ? 1 : 0)` 写入 TSCM local 输出。
- 但当前 PoC 的 Neumann `INV += INV @ Lpow` 并没有依赖 local Matmul atomic add：
  - `NeumannLocalInverseBlock()` 先把 `INV@Lpow` 写入 `power` 或 `scratch`。
  - 随后调用 `AddLocalMatrix(inv, power/scratch, blockSize)`。
  - `AddLocalMatrix()` 是 AIC 标量循环：从 TSCM local tensor 读 `dst/src`，相加后 `SetValue()` 写回 `dst`。
- 结论：当前 local/TSCM 累加点是 `AddLocalMatrix()`，不是 `AICBlockProcessFp16L1(..., enableAtomicAdd=true)`；这也是 PoC 仍有明显 scalar 固定开销的原因之一。

## local/TSCM 累加方式确认

- 用户追问是否“靠标量累加而不是 Matmul 结果矩阵累加”。
- 结论：是的，当前 PoC 的 `INV += INV@Lpow` 不是 Matmul C 矩阵累加，也不是 local atomic add。
- 当前路径：
  - `AICBlockProcessFp16L1()` 先将 `INV@Lpow` 写到 TSCM local `power` 或 `scratch`。
  - `AddLocalMatrix()` 再用 AIC 标量循环逐元素读 `inv` 与乘积矩阵，相加后写回 `inv`。
- 代码里虽然 `AICBlockProcessFp16L1()` 保留了 `enableAtomicAdd` 参数，但当前 Neumann round 调用没有启用它。
- 后续可尝试的替代方向：
  - 用 `INV@(I+P)` 形式替代 `INV + INV@P`，避免完整矩阵逐元素加法。
  - 或验证 local/TSCM Matmul 输出累加能力，但要先确认 in-place alias、half TSCM C 输出和精度行为。

## 底层 MMAD 累加参数口径

- 用户确认最终仍要通过底层 API 的 C 矩阵累加参数控制，而不是长期依赖标量循环。
- 已复核现有参考：
  - `AscendC::MmadParams(..., unitFlag, cmatrixSource, cmatrixInitVal)`
  - `cmatrixInitVal` 对应当前 raw helper 里的 `initC`：第一次 `true` 初始化/清空 L0C，后续 `false` 表示继续累加到已有 L0C。
  - `cmatrixSource=true` 用于从额外 C/bias tensor 作为 C 初值来源，参考 `mla_mma.h` 中带 `biasTensor` 的 `Mmad()` overload。
- 当前 `C64_1616_FP32_CVInverse` raw 路线已经用过 L0C accumulator 方式：
  - 先 `Mmad(INV, I, initC=true)` 把 old `INV` 种到 L0C。
  - 再 `Mmad(INV, P, initC=false)` 得到 `old_INV + INV@P`。
- local/TSCM PoC 当前阶段可以先保留标量循环确保精度正确并跑性能对比，但最终优化目标应回到 `MmadParams` 的 `cmatrixInitVal=false` 或 `cmatrixSource=true` 路线。

## 为什么之前类似思路还没做好

- 用户追问：既然之前做过类似 L0C accumulator 思路，为什么还没做好。
- 结论：之前做通的是局部 raw MMAD 累加动作，不是完整 local/TSCM fused inverse。
- 已做部分：
  - 在 `C64_1616_FP32_CVInverse` raw 路线里，用 `Mmad(INV, I, initC=true)` 先把 old `INV` 种到 L0C。
  - 再用 `Mmad(INV, P, initC=false)` 在同一个 L0C accumulator 上得到 `old_INV + INV@P`。
  - 该思路能证明底层 `cmatrixInitVal=false` 累加语义可用。
- 没做好的部分：
  - L0C accumulator 结果仍通过 Fixpipe 写回 GM，下一轮再从 GM/L1/L0A/L0B 读，不是真正片上 resident。
  - final RHS 乘法 `INV @ k_cumdecay` / `INV @ v_beta` 当时仍保留 `MatmulImpl` fallback；raw final 曾导致 `out` 精度失败。
  - `C64_1616` 子块数多，raw inv-add 性能 probe 仍极慢，说明仅替换局部累加无法消掉大量调度、同步和搬运开销。
  - 当前 `C64_3232_FP16_LocalTscm` 又遇到 64x64 NZ 大矩阵按 32x32 offset 读取的布局/精度问题。
- 下一步正确拆法：
  - 先用标量循环版把 local/TSCM 数据驻留路径和 compact tile 布局打通，保证精度。
  - 再把 `AddLocalMatrix()` 替换为底层 `Mmad` C 初值/累加路径。
  - 替换时必须处理“updated INV 如何作为下一轮 A 输入”的问题：L0C 不能直接作为下一轮 A/B，仍需 Fixpipe 到 TSCM/L1/VECCALC 后再 LoadData，或改写成 `INV@(I+P)`/ping-pong 形式减少往返。

## L1/TSCM 承接与标量正确基线

- 用户追问“累加完搬运到 L1 不行吗，为什么一定到 GM”。
- 结论：可以搬到 L1/TSCM，不必须到 GM。
  - 当前到 GM 是因为 `MatmulImpl` 默认使用 GM ND tensor，便于接下一轮 matmul 和做精度二分。
  - 真正目标仍应是 `L0C accumulator -> Fixpipe -> L1/TSCM`，再作为后续 `LoadData/Mmad` 的输入。
  - 难点是 Fixpipe 到 L1/TSCM 后的 compact NZ/stride/layout 是否能被下一轮 `LoadData` 正确消费，不是硬件上只能到 GM。
- 按用户要求先实现“标量循环精度正确版本”，本轮没有继续在混乱的 `C64_1616_FP16` 实验线上叠变量，而是切回已验证稳定的 `C64_3232_FP32_CVInverse`。
- 本轮新增方案文档：
  - `docs/2026-05-04/C64_3232_SCALAR_FORWARD_BASELINE_PLAN.md`

## 标量前代实现

- 修改 `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`：
  - selector 切到 `chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h`。
- 修改 `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h`：
  - 新增 `DEBUG_SCALAR_FORWARD_GM_INVERSE = true`。
  - 新增 `C64InverseByScalarForwardGm()`，直接从 `L_powWsGm_` 中的下三角 `L` 做前代递推：
    - `INV[row,row]=1`
    - `INV[row,col] = -L[row,col] - sum(L[row,k] * INV[k,col])`
  - final 仍保留原 FP32 GM `MatmulImpl(INVWsGm_, gBKWsGm_/vBetaWsGm_)`，用于隔离 inverse 公式正确性。
- 这个版本仍经过 GM，不是最终 local/TSCM fused；它的用途是建立可对齐的精度基线。

## 标量前代验证

- 构建 Neumann 单算子包通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j16`
- 安装通过：
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- Neumann-only smoke 三个 case 全部 PASS：
  - `S=16,nk=1,nv=1,dk=dv=16`
  - `S=17,nk=1,nv=1,dk=dv=16`
  - `S=64,nk=4,nv=4,dk=dv=128`
- 第一次全量 compare 失败原因不是精度，而是只打包了 Neumann，normal `ChunkGDR` 符号缺失：
  - `aclnnChunkGatedDeltaRule ... not in libopapi.so`
- 重新打包双算子通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 双算子安装后全量三方精度通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - `20 passed in 6.16s`
  - 多数 case 中 `ChunkGDR vs ChunkGDRNeumann` out/state `max_abs=0`。

## 标量前代性能对比

- 按用户要求在精度通过后才跑性能。
- 使用 `msprof --application ... --task-time=on` 做最小两 case 对比：
  - 输出目录：`tests/pytest/prof_out/chunk_gdr_c64_3232_scalar_forward_gm_smoke/`
  - 汇总：`performance_compare.csv`、`performance_raw.csv`
- 结果：
  - `B=8,S=16,D=64,Hq=16,Hkv=4`：normal `633.08us`，scalar Neumann `1628.66us`，Neumann/normal `2.573x`。
  - `B=8,S=64,D=64,Hq=16,Hkv=4`：normal `637.12us`，scalar Neumann `15199.28us`，Neumann/normal `23.856x`。
- 性能结论：
  - 标量前代递推只适合做正确性基线，不适合作为最终实现。
  - `S=64` 的 AIC scalar time 从 normal `113.649us` 放大到 scalar Neumann `11284.828us`，说明必须回到底层 MMAD/L0C 累加。

## 后续方向更新

- local/TSCM 不应长期用标量 `AddLocalMatrix()` 做累加。
- 下一步应按这个顺序推进：
  1. 用标量 local 前代验证 L1/TSCM resident 的数据搬运和 compact tile layout。
  2. 精度通过后，将逐元素标量累加替换为底层 `MmadParams(..., cmatrixSource, cmatrixInitVal)`。
  3. `cmatrixInitVal=false` 对应当前 raw helper 的 `initC=false`，表示在已有 L0C accumulator 上继续累加。
  4. 若使用 `cmatrixSource=true`，需验证 C/bias tensor 作为 C 初值来源的 dtype/layout。
  5. 全量 compare 通过后再跑 msprof；精度失败版本不做性能测量。

## CMatrix/MMAD 累加实现更新

- 按用户要求将当前默认 `C64_3232_FP32_CVInverse` 的 Neumann inverse 主线切回底层 CMatrix/MMAD 累加，不再保留标量前代作为默认或 debug 分支。
- 当前 selector 保持：
  - `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`
  - `#include "chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h"`
- 当前 `INV += INV @ P` 的实现位置：
  - `RawMmadInvAddBlock()`
  - `RawMmadInvAddOutputTile()`
  - `StoreRawMmadInvAddRow()`
- 累加机制：
  - 第一拍用 `RawMmadAccumulateTile(..., initC=true, ...)`，对应 `MmadParams(..., cmatrixSource=false, cmatrixInitVal=true)`，初始化/清零 L0C。
  - 先执行 `INV @ I`，把 old `INV` 种到 L0C accumulator。
  - 后续对 `INV @ P` 使用 `initC=false`，对应 `cmatrixInitVal=false`，直接在同一 L0C/CMatrix 上累加。
  - 最终按 row tile 一次性 `Fixpipe` 回 `INVWsGm_`，避免旧 `INV` 被过早覆盖后影响同一行后续输出 tile。
- 已移除当前默认文件中的标量前代残留：
  - 删除 `DEBUG_SCALAR_FORWARD_GM_INVERSE`
  - 删除 `C64InverseByScalarForwardGm()`
  - 删除未使用的 `rawInvAddScratchGm_` 映射和成员
  - 删除未使用的 `RAW_LOAD3DV2_NO_FMATRIX_CONFIG`、`CeilDivLocal()`
- 保留 host 侧 `rawFinalScratchSz`，因为 `C64_1616_FP32_CVInverse` 的 raw final scratch 仍可能依赖这段 workspace；本轮不做全局 workspace 收缩，避免影响其它变体。
- 重新构建双算子包通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 重新安装通过：
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 全量三方精度通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - `20 passed in 6.22s`
- 本轮未重新运行 msprof；当前完成范围是 Neumann inverse 矩阵生成中的 CMatrix/MMAD 累加正确性。

## 当前状态回复

- 用户询问“现在什么情况”。
- 当前默认线仍是 `C64_3232_FP32_CVInverse`，selector 未再切到 local/TSCM PoC。
- Neumann inverse 中 `INV += INV@P` 已由底层 `MmadParams(..., cmatrixSource=false, cmatrixInitVal=initC)` 控制 L0C/CMatrix 累加。
- 当前不是标量循环累加，也不是 Fixpipe atomic add 累加。
- 最新验证仍是构建安装后全量三方 compare `20 passed in 6.22s`。
- 尚未重新跑 msprof；性能数据仍需要后续单独测。

## CMatrix 累加后续探针与回退

- 复跑当前 CMatrix/MMAD 累加默认线精度：
  - `source /home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer/bin/set_env.bash && CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.19s`，`b1_t257_tail_d128` 未复现先前的 `2.5e-2` out 抬高。
- 对 lower-triangular power 探针做 S16/S64 msprof smoke：
  - 输出目录：`tests/pytest/prof_out/chunk_gdr_c64_3232_lower_tri_power_smoke/`
  - `B=8,S=16,D=64,Hq=16,Hkv=4`：normal `621.64us`，Neumann `822.32us`，Neumann/normal `1.323x`。
  - `B=8,S=64,D=64,Hq=16,Hkv=4`：normal `653.44us`，Neumann `1087.22us`，Neumann/normal `1.664x`。
  - 对比 identity-once smoke：S16 从 `1.267x` 退到 `1.323x`，S64 从 `1.642x` 退到 `1.664x`，性能略差。
- 尝试回退 lower-triangular power 为全 tile `RawMmadBlock()`：
  - 构建安装后全量 compare 仍按宽阈值 PASS，但多个 case 出现 `1e-2` 级 out 漂移。
  - 典型现象：`b2_t48_dk128_dv32` Neumann out `max_abs=1.428411e-02`，`b2_t160_h32_h8_d64` Neumann out `max_abs=4.792955e-03`。
  - 结论：power workspace 上三角不能依赖全 tile raw MMAD 的残留/无效区域，必须显式保持下三角结构并清零上三角。
- 尝试在 `RawMmadInvAddOutputTile()` 中对 `INV@P` 做 k-range 裁剪：
  - 仍然使用 `INV@I + initC=true` 种 old INV，再用 `initC=false` 做 CMatrix 累加。
  - 构建安装后全量 compare 按宽阈值 PASS，但出现更明显的 out 漂移：`b1_t129_tail_d64` `1.738328e-02`、`varlen_b3_16_33_80` `2.385536e-02`。
  - 该裁剪已回退，不能进入默认线。
- 最终保留策略：
  - `power` 计算使用 `RawMmadLowerTriangularBlock()`，只写下三角 tile，并用 `WriteZeroUpperTriangleTiles()` 显式清零 power 上三角。
  - `INV += INV@P` 使用全 tile `RawMmadInvAddBlock()`，每个 output tile 先 `INV@I` 种 old INV，再 `INV@P` 用 `cmatrixInitVal=false` 在 L0C/CMatrix 内累加。
  - `StoreRawMmadInvAddRow()` 仍按整行写回，避免同一行后续 output tile 读到被覆盖后的 `INV`。
- 最终重新构建、安装并复跑全量三方精度：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.15s`。

## CMatrix 参数语义收口

- 对当前默认 `C64_3232_FP32_CVInverse` 做低风险语义化收口，不改变默认算法路径。
- 新增/明确 raw MMAD CMatrix 语义常量：
  - `RAW_MMAD_CMATRIX_SOURCE_L0C=false`
  - `RAW_MMAD_INIT_C=true`
  - `RAW_MMAD_ACCUMULATE_C=false`
- 将 raw helper 里的 `initC` 参数改名为 `initCMatrix`，直接对应 `MmadParams(..., cmatrixSource, cmatrixInitVal)` 的 `cmatrixInitVal`。
- 在 `MakeRawMmadParams()` 和 `RawMmadInvAddOutputTile()` 附近补充注释，明确当前 `INV += INV@P` 不是标量循环，也不是 Fixpipe atomic add：
  - `INV @ I` 使用 `cmatrixInitVal=true` 初始化 L0C/CMatrix。
  - `INV @ P` 使用 `cmatrixInitVal=false` 在同一 L0C/CMatrix 上累加。
  - 最后只做普通 Fixpipe 写回 updated `INV`。
- 保留 `RawMmadLowerTriangularBlock()` 作为 power 默认实现，并在代码中写明 full-tile power 曾导致 `1e-2` 级 out 漂移，不能作为默认策略。

### 验证

- 重新构建双算子包通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 重新安装通过：
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 全量三方精度通过：
  - `source /home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer/bin/set_env.bash && CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.14s`

### 当前结论

- 默认线仍是 `C64_3232_FP32_CVInverse`。
- `INV += INV@P` 已按用户要求由底层 CMatrix/MMAD 累加控制。
- final RHS 的 `INV @ k_cumdecay` / `INV @ v_beta` 仍走 FP32 GM `MatmulImpl`，未混同为本轮验收范围。
- 本轮没有重新跑 msprof；性能结论仍沿用前一次 lower-triangular power smoke。
