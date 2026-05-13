# 2026-04-30 PROGRESS_AND_ISSUES

## 已完成
- `chunkSize=64` host tiling 修改并构建通过。
- `C64_1616_FP32_CVInverse`、`C64_3232_FP32_CVInverse`、`C64_3232_hybrid` 方案实现并通过全量三方精度 20/20。
- 完成 `C64_3232_hybrid` 全量 32 shape msprof 批测，生成 `performance_compare.csv` 与 `performance_raw.csv`。
- 汇总 C16/C64 多方案性能表：`docs/2026-04-30/PERFORMANCE_AND_TILING_ANALYSIS.md` 与 `PERFORMANCE_*.csv`。
- 已完成 Stage1 micro tiling 基础设施：新增 `matmulTilingFp32C16/C32/C64`，Stage1 inverse 可按 16/32 子块使用专用 tiling；Stage2/Stage3 不变。

## 当前结论
- 当前最佳是 `C64_3232_hybrid`，平均耗时为 normal 的 `1.778x`，明显优于 `C64_1616` 的 `2.382x`，但仍未超过 normal。
- `S=16` 上 `C64_3232_hybrid` 平均 `0.994x`，3/4 case 略快；`S>=64` 仍慢，`S=64` 平均 `1.391x`。
- 慢点主要是 `MatmulImpl + GM workspace + FIX/MTE2` 串 tiny MatMul 的固定开销，而不是理论 MAC 总量。

## 当前问题
- Neumann 内部 `16x16/32x32` matmul 仍通过通用 `MatmulImpl` 和 GM workspace 串联，AIC scalar/MTE2/FixPipe 开销放大。
- 当前 micro tiling 只解决 tiling 配置颗粒度，尚未消除 `MatmulImpl + GM workspace + FIX/MTE2` 串联。
- `stage1_C16_FP16_CVInverse.h` 名称仍不代表真实 FP16 实现，后续不能基于该文件做 FP16 性能结论。

## 下一步建议
- micro tiling 已完成并通过精度；下一步应做 msprof 全量性能验证，优先看 `aic_scalar/mte2/fixpipe` 是否下降。
- 清理未使用 GM workspace，例如不再需要的 `tmpWsGm_` 路径，降低空间和潜在搬运。
- 评估能否绕开通用 `MatmulImpl` 做 AIC fused inverse pipeline；否则 FIX/MTE2 串联瓶颈很难根除。
- FP16/BF16 inverse 路线需要单独实现并重跑精度，不应复用当前伪 FP16 文件做判断。

## 追加确认
- 主 `C=16` 方案历史上已切 host `chunkSize=16`；当前 `chunkSize=64` 是 C64 子块方案需要。
- C16 的 AIC atomic add 不是完整 fused inverse：仍有 GM workspace 往返和多次 MatmulImpl 调用。
- C64 最终表已生成：`docs/2026-04-30/C64_FINAL_PERFORMANCE_TABLE.md`，对应 CSV 为 `C64_FINAL_SCHEME_SUMMARY.csv`、`C64_FINAL_BY_S.csv`、`C64_FINAL_CASE_DETAIL.csv`。
- 已补充 `C64_FINAL_PERFORMANCE_TABLE.md` 中的方案解释与 CSV 对应关系，便于后续直接查表。
- 已固化后续总结口径：只按三种最终方案归类，中间优化过程不再作为最终方案单列；详见 `AGENTS.md` 和 `SKILL/neumann-scheme-summary/SKILL.md`。
- 已清理多余 profiling 和中间性能 docs；当前 profiling 只保留三种最终方案对应目录。
- 后续可新增 skill：`performance-profiling-msprof`、`matmul-tiling-performance`、`workspace-tiling-checklist`。
- 已新增三个 skill：`performance-profiling-msprof`、`matmul-tiling-performance`、`workspace-tiling-checklist`。
- 已确认 fused kernel 尚未适配三种最终方案；当前实现均未做到 L1/L0/register resident 的 fused inverse。
- 已新增三方案 fused AIC inverse 实施计划：`docs/2026-04-30/FUSED_AIC_INVERSE_THREE_SCHEMES_PLAN.md`；下一阶段优先做 `C64_3232_NeumannCompose` Level 1 和 `C16_1616_Neumann` fused primitive PoC。
- Fused 计划范围已收紧：当前只做 Stage1 Neumann inverse；Stage2/Stage3 暂不动。


## 本轮实现验证
- 当前默认实现：`chunkSize=64` + `chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`。
- `C16_1616_Neumann`：临时切 selector 和 host `chunkSize=16` 构建安装通过，`CGDR_GOLDEN_CHUNK_SIZE=16 bash test_run.sh compare` 20/20 PASS。
- `C64_1616_NeumannCompose`：默认 selector 和 host `chunkSize=64` 构建安装通过，`CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare` 20/20 PASS。
- Stage2/Stage3 未修改；本轮只动 Stage1 selector/实现、host tiling data 与 Stage1 matmul 初始化。
- 已完成：按 msprof warmup + cache flush 批测 `C16` 与 `C64_1616` micro tiling 性能。
- `C64_1616` micro tiling CSV：`tests/pytest/prof_out/chunk_gdr_c64_1616_micro_tiling_all_shapes/performance_compare.csv`；平均 Neumann/normal `2.4697x`，`S=16` `1.0747x`，`S=64` `1.9157x`。
- `C16` micro tiling CSV：`tests/pytest/prof_out/chunk_gdr_c16_micro_tiling_all_shapes/performance_compare.csv`；平均 Neumann/normal `3.1128x`，`S=16` `1.0232x`，`S=64` `2.1566x`。
- 当前判断：micro tiling 后 C16 只改善短序列 `S=16`，长序列因 chunk 数增加导致 AIC scalar/MTE/FixPipe 固定开销更高；C64_1616 是这两者中更合理的后续优化基线。

## 随路 Cast 结论
- 已确认输入侧没有可直接使用的 FP32->FP16 随路 cast；Cube `Mmad` 不支持 `float` 输入自动降到 `half` 输入。
- L0C/Fixpipe 输出侧支持 F32->F16/BF16 cast，但只适合输出落半精度，不适合当前 FP32 inverse 输入链路。
- 真正 fused inverse 的可行方向是 `VECCALC`/片上 local Matmul 串联或 raw `LoadData/Mmad/Fixpipe` micro-kernel；当前 `MatmulImpl + GM workspace` 不能通过“打开随路 cast”解决。

## Fused MM / L0C 复用分析
- 已新增 `docs/2026-04-30/FUSED_MM_L0C_REUSE_FP16_ANALYSIS.md`，记录 FP16、L0C 复用、MM 顺序调整和四方案落地顺序。
- 当前结论：L0C 不能跨 Neumann round 直接作为下一次 Matmul A/B；可复用层级应优先选择 `VECCALC/TSCM/L1`。
- 可优先实施：把 `INV += INV @ P` 改写为 `INV @ (I + P)`，用 ping-pong buffer 消除 atomic add 和 `tmpWsGm_` add 语义。
- FP16 方案先只在 `C16_FP16_CVInverse.h` 验证；若精度和 `aic_mac/mte2/fixpipe` 指标有收益，再扩展到 `C64_1616_FP16_CVInverse.h` 与 `C64_3232_FP16_CVInverse.h`。

## 阶段计划与精度门槛
- 已新增 `docs/2026-04-30/FUSED_FP16_STAGED_EXECUTION_PLAN.md`。
- FP16 和后续实验的精度门槛按用户要求统一为`0.08`：`atol=0.08`、`rtol=0.08`。
- 已将三方精度脚本默认阈值恢复为 `0.08/0.08`；每阶段必须全量三方精度通过后才能进入 seq 遍历性能测试。
- 阶段顺序：三方案 fusedMM -> 三方案 FP16 `3×2` 对比 -> `INV@(I+P)` 尝试与回退判断。

## FP16 方案2进度
- 已完成 `C64_1616_FP16_CVInverse` 实现：chunkSize=64，16×16 Neumann 子块 + compose，half workspace，final `half×half->float`。
- 已完成构建安装：本机最终使用 `--soc=ascend910_93` 双算子包；`ascend910b` 包在当前设备上会导致普通算子 binary 缺失。
- 已完成全量三方精度：`CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare`，20/20 PASS。
- 已完成 seq-only msprof：`tests/pytest/prof_out/chunk_gdr_phase2_c64_1616_fp16_seq_sweep/performance_compare.csv`。
- 当前性能结果：FP16 方案2平均 `neumann/normal=2.494x`，所有 seq sweep case 均慢于 normal。
- 当前问题：FP16 只降低了数据类型宽度，未消除 GM workspace、MatmulImpl tiny 调度、FIX/MTE2 和 AIC scalar 固定开销。
- 下一步建议：优先实现并验证 `C64_3232_FP16_CVInverse`，同时评估减少 GM 中间落地；如果 3232 FP16 仍无收益，应转向 `VECCALC/TSCM` local fused 或 `INV@(I+P)` 公式改写。

## CO1/L0C 累加确认
- 已确认 CO1/L0C 可用于同一 Matmul 内部 partial-sum 累加，但不能作为跨 Neumann round 的 A/B 输入缓存。
- 当前 Neumann 需要的是跨 Matmul 的结果复用，因此仍需从 L0C 经 Fixpipe 搬到 L1/TSCM/VECCALC/GM，再 LoadData 进 L0A/L0B。
- 后续优化重点不应是“把 INV 长驻 L0C 后直接参与下一次 MMAD”，而应是减少 GM 落地、优先用 L1/TSCM/VECCALC local Matmul 或公式改写降低中间搬运。

## CO1 Bias 路径修正
- 已修正判断：raw `Mmad` 支持 CO1/L0C bias，即可把 L0C 矩阵作为下一次 MMAD 的 C 初值参与累加。
- 可优先验证的优化点：将 `INV += INV@P` 从 `Matmul -> tmp -> Add/atomic` 改成 raw `Mmad(..., bias=INV_L0C)`，直接在 CO1 完成累加。
- 限制仍存在：CO1 bias 只能作为 C 初值，不能作为 A/B 输入；`P2/P4/P8` 作为下一轮 A/B 时仍需要进入 L0A/L0B。
- 下一步建议：基于 C16 最小 primitive 写 raw MMAD PoC，先验证 `C = A@B + C_l0c` 精度与性能，再迁入 C64_1616/C64_3232。

## Raw MMAD Fused 当前交接
- 已新增 `docs/2026-04-30/RAW_MMAD_FUSED_HANDOFF.md`，用于切换窗口后恢复上下文。
- 当前默认实现是 `chunkSize=64` + `C64_1616_FP32_CVInverse`。
- `RawMmadInvAdd16()` 当前临时 no-op：kernel 可返回，但 `INV += INV@P` 未执行，精度不正确。
- 当前阻塞点：raw `GM/L1/L0/MMAD/Fixpipe` 链路一恢复真实执行即挂，需要参考 FIA raw MMAD 写法逐段二分。
- 下一步：先验证 `GM->L1`、`L1->L0A/B`、`Mmad`、`Fixpipe` 单段链路，再恢复 CO1/L0C bias 累加。
