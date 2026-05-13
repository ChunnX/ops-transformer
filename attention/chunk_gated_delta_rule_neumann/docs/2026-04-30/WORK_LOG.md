# 2026-04-30 WORK_LOG

## 当前进度快照
- `chunkSize` 已切到 64，当前 Stage1 selector 指向 `chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`。
- `C64_1616_FP32_CVInverse` 已实现：AIV 只生成初始 `L`/`I-L`，AIC 内完成 16x16 Neumann、32/64 block compose；全量三方精度已通过。
- `C64_3232_FP32_CVInverse` 已实现：AIC 内完成 32x32 Neumann 4 轮倍增，并 compose 64；随后加入 `validLen <= 16` 的 16x16 tail shortcut。
- `C64_3232` 与 hybrid 历史版本已构建安装成功；当前默认安装态已切回 `C64_1616`，`CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare` 全量 20 cases 通过。

## 性能进度
- 已有 `C64_1616` smoke：S=16 略快于 normal，S=64 明显慢于 normal。
- 已有纯 `C64_3232` smoke：S=64 相比 16x16 方案改善明显，但仍慢于 normal；S=16 比 normal 略慢。
- hybrid smoke 被中断：已采集 S=16 normal/neumann 和 S=64 normal，缺 S=64 neumann，因此完整 summary 未完成。

## 下一步
- 补跑 hybrid 的 S=64 neumann 或重跑 S=16/S=64 smoke，得到完整性能表。
- 如果 hybrid S=16/S=64 均优于纯 32x32，则保留 hybrid；否则按 profiling 指标继续调度/同步优化。

## FlashKDA 对比分析
- 汇总当前 C64_1616、C64_3232、C64_3232 hybrid partial 性能结果。
- 明确当前 CANN C=16 慢的主因不是理论 MAC 增多，而是 tiny MatMul 在通用 MatmulImpl 上固定开销、GM workspace 往返、MTE/FixPipe/Scalar 占比过高。
- 对比 FlashKDA 后确认：FlashKDA 的收益来自围绕 C=16 的整体 kernel 设计，包括 warp-level fused inverse、K1/K2 拆分、TMA/shared memory 复用、FP16 inverse、register-file transpose，而不是单纯 chunk=16。
- 新增分析文档：`/mnt/workspace/gitCode/reference/FlashKDA/FLASHKDA_VS_CANN_CGDR_NEUMANN_ANALYSIS_20260430.md`。

## Profiling 与 Tiling 分析补充
- 完成 `C64_3232_hybrid` 全量 32 shape msprof 批测，结果保存到 `tests/pytest/prof_out/chunk_gdr_c64_3232_hybrid_all_shapes/performance_compare.csv` 与 `performance_raw.csv`。
- 汇总各方案性能：`C16_AICAccum` 平均 `3.015x`，`C64_1616` 平均 `2.382x`，`C64_1616_sync_reduced` 平均 `2.339x`，`C64_1616_tail_specialized` 平均 `2.353x`，`C64_3232_hybrid` 平均 `1.778x`。
- 结论：`C64_3232_hybrid` 是当前最好方案，但仍整体慢于 normal；只有 `S=16` 上 3/4 case 略快，`S>=64` 仍慢。
- 确认 tiling 问题：host 侧 normal 与 Neumann 都是 `chunkSize=64`、`P_NUM=2`，Neumann 仍只有一套 FP32 `matmulTilingFp32`，没有为 `16x16/32x32` tiny MatMul 单独生成 tiling。
- 完成文档：`docs/2026-04-30/PERFORMANCE_AND_TILING_ANALYSIS.md`，并生成各方案 CSV 表格 `docs/2026-04-30/PERFORMANCE_*.csv`。
- 优化方向收敛：优先减少 `MatmulImpl` tiny 调用和 GM workspace 往返；单靠公式减少理论 MAC 不足以赢过 normal。

## C16 Tiling 与 C64 最终性能确认
- 复核历史文档和 git 已提交版本：主 `C=16` 方案当时 host tiling 已切到 `chunkSize=16`；当前工作区改为 `chunkSize=64` 是为了 C64 1616/3232 子块方案。
- 澄清 fused AIC inverse：C16 方案已把 `INV += INV@Lpow` 从 AIV 改为 AIC/FixPipe atomic add，但中间 `L_pow/attn/INV` 仍落 GM，并非真正片上 fused inverse。
- 生成 C64 最终性能表：`docs/2026-04-30/C64_FINAL_PERFORMANCE_TABLE.md`。
- 确认 `tests/pytest/prof_out/chunk_gdr_c64_1616_sync_reduced_all_shapes` 是当前用于最终对比的 C64 16x16 子块方案；`tests/pytest/prof_out/chunk_gdr_c64_3232_hybrid_all_shapes` 是当前 C64 32x32 hybrid 方案。

## C64 表格说明补充
- 按用户要求补充 `docs/2026-04-30/C64_FINAL_PERFORMANCE_TABLE.md`：明确每个方案含义、profiling 目录和对应 CSV 文件用途。

## 三种最终方案口径固化
- 用户确认后续总结只保留三种最终方案：`C16_1616_Neumann`、`C64_1616_NeumannCompose`、`C64_3232_NeumannCompose`。
- 明确中间优化过程（如 `sync_reduced`、`tail_specialized`、`cacheline`、`hybrid`）不作为独立最终方案，只作为对应最终方案的实现演进或细节。
- 更新 `AGENTS.md`：新增 `Current Summary: 三种最终方案口径`，替换过期的 AIC 初始矩阵当前计划。
- 新增 skill：`SKILL/neumann-scheme-summary/SKILL.md`，固化方案命名、profiling 目录、CSV 口径、已知性能结论和汇报模板。

## Profiling 与 Docs 清理
- 保留三种最终方案 profiling：`chunk_gdr_prefill_all_shapes_aic_accum`、`chunk_gdr_c64_1616_sync_reduced_all_shapes`、`chunk_gdr_c64_3232_hybrid_all_shapes`。
- 删除中间 smoke、cacheline、tail_specialized、current、postbuild 等 profiling 目录，避免后续总结混淆。
- 删除 `PERFORMANCE_*` 中间汇总文档，只保留 `C64_FINAL_*` 最终表、`WORK_LOG.md`、`PROGRESS_AND_ISSUES.md`。
- 评估后续值得沉淀的 skill：性能 profiling/msprof 方法、Matmul tiny-matmul 性能分析、CANN workspace/tiling 修改检查清单。

## Skill 补充与 Fused Kernel 状态
- 新增 `SKILL/performance-profiling-msprof/SKILL.md`：固化 msprof warmup、cache flush、批测、summary 规则。
- 新增 `SKILL/matmul-tiling-performance/SKILL.md`：固化 tiny MatMul、MatmulImpl、GM workspace、FIX/MTE2 性能瓶颈分析。
- 新增 `SKILL/workspace-tiling-checklist/SKILL.md`：固化 chunkSize/workspace/stageWorkspace/selector/golden 一致性检查。
- 明确 fused kernel 状态：三种最终方案都还没有真正 fused kernel；C16 只是 AIC/FixPipe atomic add，C64 1616/3232 仍是多次 MatmulImpl + GM workspace 串联。

## 三方案 Fused AIC Inverse 计划
- 新增计划文档：`docs/2026-04-30/FUSED_AIC_INVERSE_THREE_SCHEMES_PLAN.md`。
- 计划按 Level 1/2/3 分层：先做 MatmulImpl 级轻融合和 micro tiling，再评估 L1/TSCM/VECCALC resident，最后才考虑专用 AIC micro-kernel。
- 推荐顺序：优先 `C64_3232_NeumannCompose`，并行清理 `C16_1616_Neumann` 作为 fused primitive PoC；只有 C16 primitive 证明收益后再深挖 `C64_1616_NeumannCompose`。
- 收益点清单已写入计划：micro tiling、删除 tmp workspace、减少 FIX_MTE2、32×32 优先、FP16/BF16 inverse、query/key 连续化、Stage3 融合等。

## Fused 计划范围收紧
- 用户明确 Stage2/Stage3 暂时不要动。
- 已更新 `FUSED_AIC_INVERSE_THREE_SCHEMES_PLAN.md`：本轮只优化 Stage1 Neumann inverse，Stage2/Stage3 不改动、不纳入收益目标。


## C16 与 C64_1616 Micro Tiling 实现验证
- 新增 Stage1 专用 micro tiling：host 生成 `matmulTilingFp32C16`、`matmulTilingFp32C32`、`matmulTilingFp32C64`，kernel 初始化 `stage1MT_C16_` 与 `stage1MT_C32_`；Stage2/Stage3 仍使用原 `matmulTilingFp32`。
- `C16_1616_Neumann` 已用 `matmulTilingFp32C16` 执行 Neumann 轮次；`INV += INV@Lpow` 保持 AIC/FixPipe atomic add 路径。
- `C64_1616_NeumannCompose` 已用 `matmulTilingFp32C16` 执行 16x16 子块 Neumann，用 `matmulTilingFp32C32` 执行 32x32 compose；64 chunk 内仍是 `MatmulImpl + GM workspace`，不是完整 fused kernel。
- 验证结果：`C16` 临时切 `chunkSize=16` 后构建安装通过，`CGDR_GOLDEN_CHUNK_SIZE=16 bash test_run.sh compare` 20/20 PASS。
- 验证结果：默认切回 `C64_1616`、`chunkSize=64` 后构建安装通过，`CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare` 20/20 PASS。
- 边界确认：本轮未修改 `op_kernel/chunk_gated_delta_rule_neumann_stage2.h` 和 `op_kernel/chunk_gated_delta_rule_neumann_stage3.h`。

## Micro Tiling 性能结果
- 完成 `C64_1616_NeumannCompose` micro tiling 32 shape msprof 批测，结果保存到 `tests/pytest/prof_out/chunk_gdr_c64_1616_micro_tiling_all_shapes/performance_compare.csv`。
- `C64_1616` 结果：Neumann/normal 平均 `2.4697x`，`S=16` 平均 `1.0747x`，`S=64` 平均 `1.9157x`，`S=512` 平均 `3.0049x`；32/32 case 仍是 normal 更快。
- `C64_1616` 指标：`aic_mac_time` 平均 `1.4938x`，但 `aic_scalar_time` 平均 `3.1849x`、`aic_mte2` 平均 `2.5796x`、`aic_fixpipe` 平均 `1.9693x`；AIV vec 降到 `0.4033x` 但不足以抵消 AIC 固定开销。
- 完成 `C16_1616_Neumann` micro tiling 32 shape msprof 批测，结果保存到 `tests/pytest/prof_out/chunk_gdr_c16_micro_tiling_all_shapes/performance_compare.csv`。
- `C16` 结果：Neumann/normal 平均 `3.1128x`，`S=16` 平均 `1.0232x`，`S=64` 平均 `2.1566x`，`S>=512` 约 `4.05x~4.13x`；32/32 case 仍是 normal 更快。
- 横向结论：C16 只在 `S=16` 比 C64_1616 更接近 normal，`S>=64` 全面弱于 C64_1616；micro tiling 没有根除 `MatmulImpl + GM workspace + FIX/MTE2` 串联瓶颈。

## 随路 Cast 与 Fused Inverse 可行性确认
- 复查 CANN 9.0.0 本地头文件：`TMatmul.hpp` 的 `Mmad` dtype 组合只支持 `half/half -> float`、`bfloat16_t/bfloat16_t -> float`、`float/float -> float`、`int8/int8 -> int32`，不支持 `float` 输入在进入 Cube 时随路 cast 成 `half`。
- 复查 `DataCopy/LoadData`：GM/L1/L0 输入侧常规接口模板要求源/目的 dtype 一致，未找到 `float GM/L1 -> half L0` 的随路 cast 路径。
- 复查 `TMov/TStore/Fixpipe`：L0C 输出侧支持 `float` 累加结果写出到 `half/bfloat16_t` 时做 F32->F16/BF16 quant/cast；这只能用于输出降精度，不能解决 FP32 输入直接做 FP16 MM。
- 因此当前 `stage1_C16_FP16_CVInverse.h` 不能靠随路 cast 低成本变成真正 FP16 输入 MM；若要 FP16/BF16 inverse，必须显式生成 half/bf16 副本（UB/GM/L1）再用 half/bf16 Matmul。
- 进一步确认 high-level `Matmul` 支持 `TPosition::VECCALC` 本地输入/输出，PFA 中已有 `mm1 -> VECCALC -> mm2` 的串联范式；这为后续减少 GM workspace 往返提供方向，但需要新增本地 MatmulType、host tiling 和 VECCALC buffer，不能只改现有 `GM float` MatmulType 达成 true fused。

## Fused MM、L0C 复用与 FP16 方案分析
- 新增分析文档：`docs/2026-04-30/FUSED_MM_L0C_REUSE_FP16_ANALYSIS.md`。
- 结论：跨 round 不能把 L0C/CO1 直接作为下一次 Matmul 的 A/B；L0C 只适合同一 Matmul 内部 K 分块累加。Neumann 下一轮需要把结果作为输入，至少要经 `Fixpipe` 到 `VECCALC/TSCM/L1/GM` 再 `LoadData`。
- 可落地的 MM 顺序优化：把 `INV += INV @ P` 改写成 `INV @ (I + P)`，对 `P=L²/L⁴/L⁸` 写单位对角，使用 `INV` ping-pong，删除 atomic add 和 `tmp` add 语义。
- 片上复用优先级：最终 `INV` 不落 GM，直接喂 `INV @ gBK` 和 `INV @ vBeta`；其次让 `P2/P4/P8/F2/F4/F8` 在 `VECCALC/TSCM` ping-pong；C64 compose 的 `tmp_lower` 也应留 local。
- FP16 路线判断：值得在 `C16_FP16_CVInverse.h` 先验证，但必须是真 half/bf16 输入 Matmul；若 final `INV` 为 half，而 `gBK/vBeta` 仍为 float，则后续 Matmul dtype 不匹配，需要 cast 回 FP32 或同步生成 half/bf16 RHS。

## 分阶段执行计划与 FP16 精度门槛
- 新增阶段计划：`docs/2026-04-30/FUSED_FP16_STAGED_EXECUTION_PLAN.md`。
- 用户确认 FP16 精度标准为`0.08`，即 `atol=0.08`、`rtol=0.08`。
- 已将 `tests/pytest/compare_chunk_gdr_three_way.py` 默认 `atol/rtol` 恢复为 `0.08`，后续阶段实现必须在`0.08/0.08` 门槛下全量三方精度通过后才能跑性能。
- 执行顺序固化为：先三方案 fusedMM，再三方案 FP16 做 `3×2` 对比，最后尝试 `INV@(I+P)`；若最后阶段无收益则回退。

## FP16 方案2：C64_1616 阶段闭环
- 按用户要求在阶段1后进入 FP16 方案2，默认 selector 切到 `chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_CVInverse.h`。
- 实现真实 FP16 inverse 链路：`L/INV/tmp/gBK/vBeta` 显式生成 half workspace，Neumann 内部使用 `half×half->half`，final `INV@rhs` 使用 `half×half->float`。
- 新增并使用 `matmulTilingFp16Out/Fp16C16/Fp16C32`；Stage2/Stage3 未修改。
- 先安装 `ascend910b` 包导致普通 `ChunkGatedDeltaRule` 缺少当前设备 binary，随后按本机要求重新用 `--soc=ascend910_93` 构建双算子包并安装。
- 全量三方精度通过：`CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare`，20/20 PASS，阈值 `atol=rtol=0.08`。
- 完成 B=8 seq-only msprof sweep，结果保存在 `tests/pytest/prof_out/chunk_gdr_phase2_c64_1616_fp16_seq_sweep/performance_compare.csv`。
- 性能结论：8/8 case normal 更快，FP16 Neumann 平均 `2.494x` normal，最好 `S=16` 为 `1.146x`，最差 `S=512,D=64` 为 `3.160x`。
- 判断：FP16 dtype 本身没有解决主瓶颈，当前瓶颈仍是 `MatmulImpl + GM half workspace + FIX/MTE2` 串 tiny MatMul 的固定开销。
- 阶段总结文档：`docs/2026-04-30/FP16_STAGE2_C64_1616_PERFORMANCE_ANALYSIS.md`。

## CO1/L0C 累加语义确认
- 用户提出：既然 C 矩阵即 MMAD 结果矩阵可以设置在 CO1，为什么不能在 L0C 累加。
- 复查 CANN 9.0.0 high-level Matmul 接口：`Iterate(enPartialSum)` 支持把同一个 Matmul 的 K 分块结果累加到 CO1；`Iterate(bool enPartialSum, LocalTensor<T>& localCmatrix)` 也明确 localCmatrix 是 CO1 上用于存放 Iterate 结果的内存。
- 关键区别：这是同一 Matmul 内部 partial-sum 累加，不是跨 Neumann round 的通用矩阵寄存器。下一轮 Matmul 的 A/B 输入仍必须进入 L0A/L0B，不能直接从 CO1/L0C 作为 A/B。
- 对当前 Neumann 公式，`P2=L@L`、`P4=P2@P2`、`INV@P` 都是不同 Matmul；上一轮输出要作为下一轮 A/B 时，至少需要经过 `Fixpipe -> VECCALC/TSCM/L1/GM -> LoadData -> L0A/L0B`。
- 可利用方向：在同一个 Matmul 的 K 维 splitK 上使用 `enPartialSum=true`，或用 `GetTensorC(LocalTensor)` 将结果搬到 L1/TSCM/VECCALC 后片上复用；不能依赖 CO1 直接跨 round 保存并参与下一次 MMAD。

## CO1 Bias 路径纠正
- 用户纠正：可以定制 bias 矩阵乘，并指定 bias 是 L0C/CO1 的矩阵。
- 复查底层 AscendC：`Mmad(dst, a, b, bias, params)` 的 bias tensor 支持 `CO1` 或 `C2`；当 bias position 是 `CO1/L0C` 时，底层将 `cmatrixSource=false`，表示 C 初始矩阵来自 L0C。
- 因此上一条“不能在 L0C 跨 MatMul 累加”的结论需要收窄：不能通过 high-level `MatmulImpl + GM workspace` 自动完成，但可以绕开 high-level Matmul，手写 raw `LoadData + Mmad` micro-kernel，把上一轮 CO1 结果作为下一次 MMAD 的 bias/初始 C。
- 对 Neumann 的直接价值：`INV += INV @ P` 可尝试实现为 `Mmad(outL0C, invL0A, pL0B, invBiasL0C, params)`，避免 `tmp` 落 GM/L1 后再加，减少 FixPipe/MTE2/GM 往返。
- 仍需注意：bias 只能作为 C 初值，不能作为 A/B；若下一轮要计算 `P4=P2@P2`，`P2` 仍需进入 L0A/L0B，除非同步保留/重载到 L0A/L0B。因此该路径主要优化 `INV += INV@P` 累加，不自动解决所有跨 round 输入复用。

## Raw MMAD Fused 交接总结
- 新增交接文档：`docs/2026-04-30/RAW_MMAD_FUSED_HANDOFF.md`，记录当前 raw MMAD fused 任务、代码状态、挂点、验证命令和下一步二分计划。
- 当前默认 selector 仍为 `C64_1616_FP32_CVInverse`，host `chunkSize=64`；`RawMmadInvAdd16()` 当前临时 no-op，可返回但精度不正确。
- 已确认当前挂点集中在 `RawMmadInvAdd16()` 内部 raw GM/L1/L0/MMAD/Fixpipe 链路，外层 C64_1616 流程、pytest、CPU golden 和 cross-core flag 不是主要问题。
- 用户指定 raw MMAD 可参考 `attention/common/op_kernel/arch32/fia_block_cube_nonquant_gqa.h`，下一步应按 FIA 的事件链和参数二分恢复。
