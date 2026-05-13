# 2026-05-07 工作日志

## fulllocalnew/C16_FP16 迁移进度复核

- 本次对话按用户询问“现在是什么进度”复核 `docs/2026-05-06/FULLLOCALNEW_DIRECT_MIGRATION_PLAN.md` 的实际执行状态。
- 当前 active selector 已切到 `fulllocalnew/C16_FP16`：
  - `op_kernel/selected/stage1_selected.h` include `../fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocalnew.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocalnew"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C16_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=16`
- `fulllocalnew` 迁移已完成到 C16 单点调试阶段：
  - Step 1：`op_kernel/fulllocalnew/` 已建立。
  - Step 2：normal 新 BF16 主链路已迁移到 Neumann wrapper、Stage2、Stage3、utils、host tiling。
  - Step 3：`C16_FP16` fulllocalnew 已接入 selector，并完成构建安装。
- 最近一次构建安装已成功：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 最近一次全量 compare 状态：
  - 命令：`CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`16 passed, 4 failed`
  - 失败 case：`b4_t32_h16_h4_d64`、`b2_t96_h8_h2_d128`、`b1_t257_tail_d128`、`b2_t160_h32_h8_d64`
- 失败特征：
  - 剩余失败均为 `out` 失败，`state` 通过。
  - 这说明 FP16 inverse/state update 主路径大概率已基本正确，当前问题更集中在 Stage1 输出链路或后续 attention output 链路。
- 当前主要怀疑点：
  - Stage1 `outQkGm_` / `qkt` 生成与 cache flush/order。
  - `qPrime`、`vInner` 或 Stage3 `qkt + vInner -> out` 的数据链路。
  - `fulllocalnew` 中 `FlushBf16MatrixCacheLines(outQkGm_)` 与 normal 新实现的差异需要二分验证。
- 已确认的一处回退：
  - `GammaCompute()` 中错误尝试已恢复为 `Sub(...)` 后 `Exp(...)`，不再使用 `Div(...)`。

## 下一步建议

- 先跑失败 case 的隔离复现，不直接进入 C64。
- 做 `value=0` 隔离测试：
  - 若 `out` 仍差，优先查 `qPrime` / Stage2 initial state 相关路径。
  - 若 `out` 不差，优先查 `vInner` 或 Stage3 `qkt @ vInner`。
- 对比 new normal Stage1 与 `fulllocalnew/C16_FP16`：
  - `ParaChunkAIC()` qkt matmul和 cache flush。
  - `QPrimeCompute()`。
  - `GBKCompute()` / `VBetaCompute()`。
  - `GCumExpCompute()` 写入 `gCumExp_` 的逻辑。
- C16 达到 `20 passed` 前，不建议推进 `fulllocalnew/C64_1616_FP16`。

## 失败误差复测

- 按用户询问“精度失败的误差是多少”，复跑 4 个此前记录的失败 case：
  - `b4_t32_h16_h4_d64`：FAIL，`CPU vs ChunkGDRNeumann out max_abs=2.009693e+00`，`state max_abs=7.709693e-01`。
  - `b2_t96_h8_h2_d128`：PASS，`CPU vs ChunkGDRNeumann out max_abs=2.467379e-03`，`state max_abs=1.070023e-03`。
  - `b1_t257_tail_d128`：PASS，`CPU vs ChunkGDRNeumann out max_abs=3.983974e-03`，`state max_abs=9.540543e-04`。
  - `b2_t160_h32_h8_d64`：FAIL，`CPU vs ChunkGDRNeumann out max_abs=4.880881e+00`，`state max_abs=1.461763e-01`。
- 随后重新执行完整 compare：
  - 命令：`CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`3 failed, 17 passed in 6.02s`
- 完整 compare 失败误差：
  - `b4_t32_h16_h4_d64`：`out max_abs=8.349202e-01`，`state max_abs=1.512490e+00`。
  - `varlen_b3_16_33_80`：`out max_abs=2.254932e+00`，`state PASS max_abs=1.377270e-03`。
  - `b2_t160_h32_h8_d64`：`out max_abs=3.340153e+00`，`state max_abs=1.842076e-01`。
- 结论：
  - 失败 case 集合和误差幅度存在复跑差异，说明当前问题不是稳定的小数值偏差，而更像并行 chunk scratch 覆盖、同步、cache flush/order 或未初始化数据导致的偶发大误差。

## fulllocalnew/C16_FP16 继续定位记录

- 时间：2026-05-07 10:32 CST。
- 复现高风险 case `b2_t160_h32_h8_d64` 3 次：
  - repeat 0：`CPU vs ChunkGDRNeumann out max_abs=1.183619e+00`，`state PASS max_abs=1.457855e-03`。
  - repeat 1：`out max_abs=2.157468e+00`，`state PASS max_abs=1.457855e-03`。
  - repeat 2：`out max_abs=2.303060e+00`，`state PASS max_abs=1.457855e-03`。
- 当前安装包与源码状态匹配：selector 仍是 `fulllocalnew/C16_FP16/chunkSize=16`。
- 已排除/弱化的方向：
  - `QPrimeCompute()` 后新增 `MTE3->MTE2` 同步和 BF16 cache flush 不能稳定修复；失败仍集中在 `out`。
  - 临时将 Stage3 atomic add 绕成 AIC 写临时 `attnInter_`、AIV 再加到 `out` 后，高风险 case 仍不通过；Stage3 atomic add 不是直接根因。该诊断改动已撤回。
- 当前判断：
  - `state` 多数稳定 PASS，说明 Stage1 inverse 到 `kCumDecay/vInner/kg/finalState` 的递推主链路基本可用。
  - `out` 随机大离群仍存在，最可疑的是只影响 output 的链路：Stage2 base out 写回与 Stage3 atomic add 的可见性、Stage1 `qkt` 读写、或 raw FP16 inverse 后残留 cube/cache 状态影响后续 matmul。
- 下一步优先级：
  - 做最小诊断：绕开 raw FP16 resident inverse 写回，改用稳定 AIV/normal 逆矩阵写回路径，判断随机大误差是否消失。
  - 若 bypass 后稳定，则聚焦 `C16InverseBy16x16Neumann()` 内 raw MMAD/Fixpipe/cache 状态恢复。
  - 若 bypass 后仍失败，则继续隔离 `outQkGm_` 与 Stage2/Stage3 out 累加可见性。

## 范围调整：放弃 C16，聚焦 C64_1616

- 用户明确新边界：
  - 不再修改 Stage2/Stage3。
  - Stage1 中除求逆相关路径外不再修改其它部分。
  - 放弃 `C16` 方案，后续只关注 `C64_1616` 方案。
- 当前工作区真实 selector 仍停在 `fulllocalnew/C16_FP16/chunkSize=16`，不是目标 `C64_1616`：
  - `op_kernel/selected/stage1_selected.h` include `../fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocalnew.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocalnew"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C16_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=16`
- 目标 `fulllocal/C64_1616_FP16` 的最近可信状态来自 2026-05-06 复验：
  - selector 当时为 `fulllocal/C64_1616_FP16/chunkSize=64`。
  - 修改边界审计结论：只改 Stage1 inverse；final RHS、`gBK/vBeta`、QK、Stage2、Stage3 未被 resident/raw 化。
  - 全量精度：`CGDR_GOLDEN_CHUNK_SIZE=64 ... test_run.sh compare`，`20 passed in 6.20s`。
  - smoke 性能：`S=16` Neumann/normal `1.234x`，`S=64` `1.722x`。
  - smoke 未裁剪无关 `TilingData/MatmulTiling` 字段，只能作为观察，不作为正式性能结论。
- `fulllocalnew/C64_1616_FP16` 尚未建立对应 Stage1 文件；`op_kernel/fulllocalnew/` 当前只有 `C16_FP16` 文件。
- 后续若继续实现，应先切回并验证 `fulllocal/C64_1616_FP16` 稳定基线，然后只在其求逆相关 helper 上迭代；不得继续沿 `fulllocalnew/C16` 调 Stage2/Stage3 或非求逆链路。

## origin 基线对齐与 fulllocalnew/C64_1616 验证

- 按用户要求先跑通 `op_kernel/origin` 基线，基线使用 `attention/chunk_gated_delta_rule` 最新 BF16 主链路代码。
- `origin` baseline 验证状态：
  - selector 曾切到 `op_kernel/origin/chunk_gated_delta_rule_neumann_stage1.h`。
  - `CGDR_NEUMANN_SELECTED_SUITE="origin"`，`CGDR_NEUMANN_SELECTED_VARIANT="latest_gdr_baseline"`，`CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`。
  - 构建：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
  - 安装：`bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`。
  - 验证：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash test_run.sh compare`。
  - 结果：`20 passed in 6.23s`；`ChunkGDR vs ChunkGDRNeumann` 的 `out/state max_abs` 为 `0`，说明 Neumann wrapper 基线与最新 GDR 对齐。
- 之后切到 `fulllocalnew/C64_1616_FP16/chunkSize=64`：
  - `op_kernel/selected/stage1_selected.h` include `../fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocalnew.h`。
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocalnew"`。
  - `CGDR_NEUMANN_SELECTED_VARIANT="C64_1616_FP16"`。
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`。
- 用户明确边界：新版主 Matmul 仍是 BF16，不能改；求逆 matmul 需要 FP16。
- 本轮 C64 修改遵守该边界：
  - `StageOneMT` 仍为 `bfloat16_t/bfloat16_t -> bfloat16_t`，`key@key`、`query@key`、`attn@gBK`、`attn@vBeta` 均保持新版 BF16 matmul。
  - 只在 `C64InverseBy16x16Neumann()` 内做 FP16 half L1/TSCM resident Neumann/compose。
  - AIV 生成 FP32 `L_powWsGm_` / `INVWsGm_` scratch，AIC 用 FP16 resident 求逆，AIC 完成后 AIV 将 FP32 `INVWsGm_` cast 到 BF16 `attnWsGm_`，供后续 BF16 final matmul 使用。
  - 未修改 Stage2、Stage3，也未修改 Stage1 非求逆链路、`gBK/vBeta` 生成或 final RHS matmul dtype。
- 修复过程：
  - 初始 `fulllocalnew/C64_1616_FP16` 构建安装可过，但全量 compare 为 `19 failed, 1 passed`。
  - 最小 `S=16` case 失败：`ChunkGDR vs ChunkGDRNeumann out max_abs=1.146660e-01`，`state max_abs=1.349649e-01`。
  - 默认 `S=64` case 失败：`ChunkGDR vs ChunkGDRNeumann out max_abs=2.458496e-01`，`state max_abs=1.530762e-01`。
  - 问题集中在 inverse 结果写入 BF16 `attnWsGm_` 的路径；将 FP32 inverse 到 BF16 attn 的转换从 AIC raw MMAD/Fixpipe 转写改为 AIV vector cast，并在 AIC FP32 GM 写回后补齐 `FIX->S` 可见性同步。
- 修复后验证：
  - 构建成功：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
  - 安装成功：`bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`。
  - `S=16` smoke PASS：`ChunkGDR vs ChunkGDRNeumann out max_abs=4.882812e-04`，`state max_abs=4.882812e-04`。
  - 默认 `S=64` smoke PASS：`ChunkGDR vs ChunkGDRNeumann out max_abs=4.882812e-04`，`state max_abs=9.765625e-04`。
  - 全量 compare PASS：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash test_run.sh compare`，结果 `20 passed in 6.26s`。
- 当前结论：
  - `origin/latest_gdr_baseline` 已跑通。
  - `fulllocalnew/C64_1616_FP16` 已跑通精度，且保持新版 BF16 final matmul不变。
  - 当前尚未做性能 profiling；下一步如继续，应在此精度通过点上做 C64 性能 smoke 或按单方案规则裁剪 tiling 后做放大性能。

## fulllocalnew/C64_1616_FP16 典型 shape 性能对比

- 按用户要求跑典型 shape 性能对比，使用 `msprof --application=... --task-time=on`。
- 当前 selector：`fulllocalnew/C64_1616_FP16/chunkSize=64`。
- 输出目录：`tests/pytest/prof_out/fulllocalnew_c64_1616_fp16_typical_20260507/`。
- 命令：
  - `cd tests/pytest && TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext python3 run_chunk_gdr_performance_msprof.py --output ./prof_out/fulllocalnew_c64_1616_fp16_typical_20260507 --ops both --case prefill_b8_s16_d64_h16_hkv4 --case prefill_b8_s64_d64_h16_hkv4 --case prefill_b8_s125_d64_h16_hkv4 --case prefill_b8_s2048_d128_h16_hkv4`
- 汇总文件：
  - `performance_compare.csv`
  - `performance_raw.csv`
- 典型 shape 结果：

| case | normal task(us) | neumann task(us) | neumann/normal | AIC ratio | AIC MAC ratio |
|---|---:|---:|---:|---:|---:|
| `B8,S16,D64,H16,Hkv4` | 656.30 | 751.02 | 1.144x | 1.145x | 1.600x |
| `B8,S64,D64,H16,Hkv4` | 649.90 | 1284.58 | 1.977x | 1.985x | 3.167x |
| `B8,S125,D64,H16,Hkv4` | 838.60 | 1943.62 | 2.318x | 2.321x | 2.667x |
| `B8,S2048,D128,H16,Hkv4` | 5073.28 | 19028.64 | 3.751x | 3.752x | 0.920x |

- 结论：
  - 4/4 典型 shape 仍是 normal 更快。
  - `S=16` ratio 为 `1.144x`，短序列开销接近但仍慢。
  - `S=64/125/2048` ratio 分别为 `1.977x/2.318x/3.751x`，长序列劣化明显。
  - `S=2048` 的 AIC MAC ratio 小于 1，但总时长仍 `3.751x`，说明当前瓶颈不是纯 MAC，而是 scalar、MTE/Fixpipe、同步和 C64 inverse/compose 固定开销放大。
- 注意：
  - 本轮是典型 shape smoke 性能对比，未裁剪无关 `TilingData/MatmulTiling` 字段；不能作为最终单方案放大性能结论。

## baseline/C64_1616_FP16 inverse 与 KV MM 数据流复核

- 当前 active selector 指向 `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_baseline.h`，配置为 `baseline / C64_1616_FP16 / chunkSize=64`。
- 当前 inverse 实现口径：
  - AIV 从 `kkWsGm_` 生成 FP32 `L_powWsGm_` 和 FP32 `INVWsGm_ = I - L` scratch。
  - AIC 对 4 个 16x16 对角块分别从 FP32 GM 搬入 half L1/TSCM，执行 3 轮 Neumann：`power = power @ power` 与 `inv = inv + inv @ power`，该 16x16 内循环在 half L1/TSCM + L0A/L0B/L0C 内完成。
  - 每个 16x16 对角块完成后仍写回 FP32 `INVWsGm_`。
  - `Compose32From16()` 当前使用 half L1/TSCM 局部 compose，但输入仍从 FP32 `INVWsGm_` 读、输出仍写回 FP32 `INVWsGm_`。
  - `Compose64From32()` 当前 `FP16_L1_RESIDENT_COMPOSE32=false`，走 FP32 GM raw MMAD：先写 `tmpWsGm_`，再写回 lower-left `INVWsGm_`。
  - AIC inverse 完成后，AIV 再从 FP32 `INVWsGm_` 读全量 64x64 inverse，vector cast 成 BF16 并写入 `attnWsGm_`。
- 后续 KV MM 口径：
  - `attn @ gBK` 仍是 BF16 `AICProcess(attnWsGm_, gBKWsGm_, outKCumdecayGm_, ...)`。
  - `attn @ vBeta` 仍是 BF16 `AICProcess(attnWsGm_, vBetaWsGm_, outVInnerGm_, ...)`。
  - `gBKWsGm_` 与 `vBetaWsGm_` 由 AIV 生成后写 GM，两个 final matmul 再从 GM 读。
- GM 往返结论：
  - 当前不是 full-local resident；只有每个 16x16 Neumann inner loop 是片上 resident。
  - 对 inverse 主矩阵而言，最终结果至少经历：AIC 写 FP32 `INVWsGm_`、AIV 读 `INVWsGm_`、AIV 写 BF16 `attnWsGm_`、AIC 为 `attn@gBK` 读一次 `attnWsGm_`、AIC 为 `attn@vBeta` 再读一次 `attnWsGm_`。
  - 因此从 inverse 结果到两个 KV MM，仍有 2 个 GM materialization 点：FP32 `INVWsGm_` 与 BF16 `attnWsGm_`；`attnWsGm_` 被两个 final matmul 各读一次。

## C64_1616 后续优化顺序整理

- 按用户指出修正第一优先级：AIV 从 `kkWsGm_` 做 mask/beta/gamma 后，应直接生成 FP16 `L` 与 FP16 `I-L` 初始 scratch，而不是先写 FP32 `L_powWsGm_` / `INVWsGm_` 再由 AIC 通过 identity MMAD 转 half L1/TSCM。
- 结合 2026-05-06 两份 fast inverse 文档，新增整理文档：
  - `docs/2026-05-07/C64_1616_FAST_INVERSE_OPTIMIZATION_ORDER.md`
- 新排序按当前 `baseline/C64_1616_FP16` 约束重排，近期只建议推进：
  1. AIV 直接生成 FP16 `L/I-L` 初始 scratch。
  2. 缓存 half identity，减少重复 GM identity 准备。
  3. inverse 内部尽量 FP16 GM 化，减少 FP32 `INVWsGm_` 中间态。
- 更难项后置：
  - C64 full half L1/TSCM staging。
  - Compose32/Compose64 full-L1 resident。
  - AIC 直接写 BF16 `attnWsGm_`。
  - packed diagonal 16x16 Neumann。
  - odd/even packed MBH compose。
  - 真正 L0 ping-pong pipeline。
  - KVMM resident 或 RHS 片上生成。
- 口径保持：
  - 不修改 Stage2/Stage3。
  - Stage1 只改求逆相关路径。
  - 新版主 matmul 与 final RHS 仍保持 BF16。

## RHS resident 口径修正

- 用户指出当前 `gBK/vBeta` 本来就是 BF16；复核后确认该指出正确。
- 修正 `C64_1616_FAST_INVERSE_OPTIMIZATION_ORDER.md` 中 P10 表述：
  - 当前新版 baseline 的 `gBKWsGm_` / `vBetaWsGm_` 是 BF16 GM，不应沿用旧 C16 PoC 的“FP32 RHS GM 转 half L1/TSCM”口径。
  - KVMM resident 的难点应表述为：BF16 GM RHS 仍需变成 AIC raw MMAD 可消费的片上 L1/NZ operand，或者 AIV 生成端直接产出片上 tile；这会涉及 AIV/AIC 生命周期、同步和当前禁止修改的 `gBK/vBeta` 生成链路。
  - 因此近期仍保持 final RHS BF16 `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)` 不变。

## KVMM resident 口径再修正

- 用户追问“本来也要搬运 `gBK/vBeta` 做矩阵乘，有什么区别”，该判断成立。
- 进一步修正 P10：
  - 当前 BF16 GM x GM `MatmulImpl` 本来就搬运 RHS，resident KVMM 不应把 RHS GM 搬运视为新增成本。
  - 正确收益点是 LHS resident：省掉 final inverse/attn 写入 BF16 `attnWsGm_`，以及两个 final matmul 分别从 GM 读取 `attnWsGm_`。
  - 首选方向应是 `A: resident/L1, B: BF16 GM, C: BF16 GM`；`gBK/vBeta` 仍按当前 BF16 GM 消费。
  - 只有当实现被迫将 BF16 RHS 额外转成 half L1/NZ 时，才会产生额外 RHS 转换成本；这不是首选路线。

## P5 与 P7/P8 顺序修正

- 用户询问 P5 是否可直接用 P7/P8 代替；结论是可以。
- 已修正 `C64_1616_FAST_INVERSE_OPTIMIZATION_ORDER.md`：
  - P5 改为“朴素 Compose32/Compose64 full-L1 resident”，只作为 correctness baseline 或 fallback。
  - fast inverse 主线优先走 P7 packed diagonal 16x16 Neumann 和 P8 odd/even packed MBH compose。
  - 若开发时间有限，P4 full L1 staging 后可跳过 P5，直接做 P7/P8；P5 只在需要隔离 full/compact offset、compose 数学顺序或 tail 问题时使用。

## 计划复审修正

- 重新用当前 `baseline/C64_1616_FP16` 代码审视计划，确认以下事实：
  - 当前 final RHS 实际是 BF16 `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)`，不是旧文档里的 `AICProcess(INVWsGm_, ...)`。
  - 当前 `L_powWsGm_` / `INVWsGm_` / `tmpWsGm_` 是 FP32 GM，`attnWsGm_` / `gBKWsGm_` / `vBetaWsGm_` 是 BF16 GM。
  - 当前 host `stageWorkspaceSz` 额外 Neumann workspace 按 3 块 FP32 `c*c` 计算。
- 已修正计划：
  - P1 不再假设 `half GM -> half L1/TSCM NZ` 直接搬运已可用，改为必须先做入口消费路径探针。
  - P1 明确 workspace 需要同步重算，不能只改 kernel `GlobalTensor` 类型。
  - `PROGRESS_AND_ISSUES.md` 中 selector 下一步从旧 `fulllocalnew` 口径修正为当前 `baseline/C64_1616_FP16`。

## 优化验证 Gate 约束

- 用户明确要求：做完一个优化就验证一次精度，精度通过才能进行下个优化。
- 已写入 `C64_1616_FAST_INVERSE_OPTIMIZATION_ORDER.md`：
  - 每次只打开一个优化开关或只合入一个数据流变化。
  - 必须 build/install 后先跑 smoke 精度，再跑全量 compare。
  - 全量 compare 通过后才能进入下一个优化。
  - 如果 smoke 或 full compare 失败，必须停在当前优化点正向定位和修复，不允许继续叠加后续优化；恢复 PASS baseline 只能用于明确放弃候选或重建干净工作面。
  - 未通过精度的优化不得作为后续优化基线。

## baseline/C64_1616_FP16 基线复验

- 按用户要求，在开始 P1 优化前先复验当前基线。
- selector 状态：
  - `op_kernel/selected/stage1_selected.h` include `../baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_baseline.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="baseline"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C64_1616_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`
- 构建：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - 结果：成功，生成 `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`。
- 安装：
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 结果：`SUCCESS`。
- 全量精度：
  - `cd tests/pytest && CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash test_run.sh compare`
  - 结果：`20 passed in 6.21s`。
- 结论：当前 `baseline/C64_1616_FP16/chunkSize=64` 可作为 P1 前的干净精度基线。

## P1 AIV 直接生成 FP16 L/I-L 初始 scratch

- 按 `C64_1616_FAST_INVERSE_OPTIMIZATION_ORDER.md` 执行第一个优化项，只合入 P1，不叠加 P2。
- 当前 selector：
  - `op_kernel/selected/stage1_selected.h` include `../fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocalnew.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocalnew"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C64_1616_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`
- 实现要点：
  - host `stageWorkspaceSz` 改为 FP16 `L_pow` + FP16 初始 `INV` + FP32 `tmp` + FP32 final `INV`。
  - AIV 从 `kkWsGm_` 直接生成 FP16 `L_powHalfWsGm_` 与 FP16 `INVInitHalfWsGm_`。
  - AIC 通过 `LoadGmHalfBlockToHalfL1()` 消费 FP16 scratch，后续 16x16 Neumann/compose 仍在 half L1/TSCM + L0 中执行。
  - final `INVWsGm_` 仍写 FP32，AIV 再 cast 到 BF16 `attnWsGm_`；final RHS BF16 `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)` 未修改。
- 定位与修复：
  - 初始全量 compare 出现 `2 failed, 18 passed`，失败为 `dims32_b2_t64` 和 `varlen_b4_5_16_31_64`，均表现为 `out` 大误差而 `state` 通过。
  - 复现发现 `dims32` 单独跑通过，但前四个 case 顺序执行时失败。
  - 32x32 `MaterializeInitialInvBlock()` 改为四个 16x16 materialize，避免 32x32 half GM -> L1/NZ 直接搬运在 compose 边界产生污染。
  - AscendC 不允许 `__aicore__` 递归调用，已将递归拆分为非递归 `MaterializeInitialInv16Block()` helper。
- 构建与安装：
  - `bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - 结果：成功。
  - `bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 结果：`SUCCESS`。
- smoke 精度：
  - 前四个顺序 case `default_b1_t64_d128 / varlen_b2_17_65 / gqa_b2_nk2_nv4 / dims32_b2_t64` 全部 PASS。
  - 历史失败 case `varlen_b4_5_16_31_64` 单独 smoke PASS。
- 全量精度：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.30s`。
- 典型 shape 性能：
  - 输出目录：`tests/pytest/prof_out/fulllocalnew_c64_1616_fp16_p1_half_init_typical_20260507/`。
  - 命令：`cd tests/pytest && TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext python3 run_chunk_gdr_performance_msprof.py --output ./prof_out/fulllocalnew_c64_1616_fp16_p1_half_init_typical_20260507 --ops both --case prefill_b8_s16_d64_h16_hkv4 --case prefill_b8_s64_d64_h16_hkv4 --case prefill_b8_s125_d64_h16_hkv4 --case prefill_b8_s2048_d128_h16_hkv4`

| case | normal task(us) | neumann task(us) | neumann/normal | AIC ratio | AIC MAC time ratio | AIC scalar ratio |
|---|---:|---:|---:|---:|---:|---:|
| `B8,S16,D64,H16,Hkv4` | 627.56 | 834.02 | 1.329x | 1.328x | 1.573x | 1.655x |
| `B8,S64,D64,H16,Hkv4` | 658.30 | 1529.34 | 2.323x | 2.328x | 6.328x | 5.298x |
| `B8,S125,D64,H16,Hkv4` | 836.08 | 2475.82 | 2.961x | 2.969x | 6.295x | 6.279x |
| `B8,S2048,D128,H16,Hkv4` | 5055.34 | 26589.20 | 5.260x | 5.263x | 3.587x | 8.062x |

- 结论：
  - P1 精度闭环通过，可作为 P2 基线。
  - 4/4 典型 shape 仍是 normal 更快。
  - P1 入口 FP16 scratch 消除了原 FP32 GM -> half L1 的一部分绕路，但当前 32x32 materialize 为保证精度拆成四个 16x16，且 inverse/compose/final attn 仍有 GM materialization，性能未改善。
  - 本轮仍是典型 shape smoke 性能对比，未裁剪无关 `TilingData/MatmulTiling` 字段，不能作为最终单方案放大性能结论。

## P2 half identity cache

- 按计划在 P1 精度和性能闭环后进入 P2；本轮只合入 identity cache，不叠加 P3。
- 实现要点：
  - 新增 `preparedHalfIdentityBlockSize_` 状态。
  - `PrepareHalfL1IdentityByMmad(blockSize)` 若 blockSize 已匹配，则直接复用 `aicHalfL1Identity_` 和 `tmpWsGm_` 中的 identity。
  - `InitRawMmadLocalBuffers()` 与 `RestoreStage1MatmulLocalBuffers()` 重置缓存状态，避免跨 pipe reset 或 matmul buffer 生命周期复用。
  - 当前 active 路径 `FP16_L1_RESIDENT_COMPOSE16=true`、`FP16_L1_RESIDENT_COMPOSE32=false`，实际主要缓存 16x16 identity。
- 构建与安装：
  - `bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - 结果：成功。
  - `bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 结果：`SUCCESS`。
- smoke 精度：
  - 同一进程串行跑 `default_b1_t64_d128 / varlen_b2_17_65 / gqa_b2_nk2_nv4 / dims32_b2_t64 / varlen_b4_5_16_31_64`。
  - 结果：全部 PASS。
- 全量精度：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.35s`。
- 典型 shape 性能：
  - 输出目录：`tests/pytest/prof_out/fulllocalnew_c64_1616_fp16_p2_identity_cache_typical_20260507/`。
  - 命令：`cd tests/pytest && TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext python3 run_chunk_gdr_performance_msprof.py --output ./prof_out/fulllocalnew_c64_1616_fp16_p2_identity_cache_typical_20260507 --ops both --case prefill_b8_s16_d64_h16_hkv4 --case prefill_b8_s64_d64_h16_hkv4 --case prefill_b8_s125_d64_h16_hkv4 --case prefill_b8_s2048_d128_h16_hkv4`

| case | normal task(us) | neumann task(us) | P2 ratio | P1 ratio | P2/P1 Neumann | AIC ratio | AIC scalar ratio | AIC MTE1 ratio | AIC MTE2 ratio | Fixpipe ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B8,S16,D64,H16,Hkv4` | 644.80 | 823.72 | 1.277x | 1.329x | 0.988x | 1.278x | 1.650x | 1.165x | 0.680x | 0.610x |
| `B8,S64,D64,H16,Hkv4` | 667.58 | 1156.46 | 1.732x | 2.323x | 0.756x | 1.738x | 2.604x | 3.038x | 1.817x | 1.485x |
| `B8,S125,D64,H16,Hkv4` | 853.12 | 1692.88 | 1.984x | 2.961x | 0.684x | 1.992x | 2.830x | 3.137x | 2.056x | 1.468x |
| `B8,S2048,D128,H16,Hkv4` | 5067.60 | 15762.92 | 3.111x | 5.260x | 0.593x | 3.111x | 3.329x | 2.390x | 2.564x | 1.385x |

- 结论：
  - P2 精度闭环通过，可作为 P3 基线。
  - P2 对 `S>=64` 明显降低 Neumann 绝对耗时，说明重复 identity 准备确实是 P1 后的主要固定开销之一。
  - 4/4 典型 shape 仍是 normal 更快；当前主要剩余问题仍是 16x16/compose 的 GM 中间态、scalar/MTE/Fixpipe 和 final BF16 matmul 前的 `attnWsGm_` materialization。
  - 本轮仍是典型 shape smoke 性能对比，未裁剪无关 `TilingData/MatmulTiling` 字段，不能作为最终单方案放大性能结论。

## P3 inverse 内部 FP16 GM 化

- 按计划在 P2 精度和性能闭环后进入 P3；本轮只合入 inverse 内部 FP16 scratch 数据流，不启动 P4。
- 当前 selector 保持：
  - `op_kernel/selected/stage1_selected.h` include `../fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocalnew.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocalnew"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C64_1616_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`
- 实现要点：
  - `INVInitHalfWsGm_` 从初始 `I-L` scratch 改为 inverse 内部可变 FP16 scratch。
  - 16x16 Neumann block 完成后通过 `CopyHalfL1MatrixToGmHalfByMmad()` 写回 FP16 scratch，不再把该阶段主中间态写到 FP32 `INVWsGm_`。
  - `Compose32From16()` 和 `Compose64From32()` 改为通过 `HalfGmComposeBlock()` 消费/写回 FP16 scratch，当前 `FP16_L1_RESIDENT_COMPOSE32=true`。
  - AIV final `attnWsGm_` 生成改为读取 FP16 scratch，并采用 half -> float -> BF16 cast；final RHS 仍保持 BF16 `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)`。
  - 未修改 Stage2、Stage3、`gBK/vBeta` 生成链路或 final RHS matmul。
- 构建与安装：
  - `bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - 结果：成功。
  - `bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 结果：`SUCCESS`。
- smoke 精度：
  - 覆盖 `S=16/17/33/64/125`，其中 `S=17` 覆盖 16->32 compose 边界，`S=33` 覆盖 32 block 边界。
  - 结果：全部 PASS。
- 全量精度：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.25s`。
- 典型 shape 性能：
  - 输出目录：`tests/pytest/prof_out/fulllocalnew_c64_1616_fp16_p3_half_internal_typical_20260507/`。
  - 命令：`cd tests/pytest && TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext python3 run_chunk_gdr_performance_msprof.py --output ./prof_out/fulllocalnew_c64_1616_fp16_p3_half_internal_typical_20260507 --ops both --case prefill_b8_s16_d64_h16_hkv4 --case prefill_b8_s64_d64_h16_hkv4 --case prefill_b8_s125_d64_h16_hkv4 --case prefill_b8_s2048_d128_h16_hkv4`

| case | normal task(us) | neumann task(us) | P3 ratio | P2 ratio | P3/P2 Neumann | AIC ratio | AIC scalar ratio | AIC MTE1 ratio | AIC MTE2 ratio | Fixpipe ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B8,S16,D64,H16,Hkv4` | 645.76 | 796.98 | 1.234x | 1.277x | 0.968x | 1.235x | 1.572x | 1.207x | 0.783x | 0.643x |
| `B8,S64,D64,H16,Hkv4` | 659.06 | 1076.32 | 1.633x | 1.732x | 0.931x | 1.637x | 3.167x | 2.072x | 1.295x | 1.093x |
| `B8,S125,D64,H16,Hkv4` | 838.76 | 1641.60 | 1.957x | 1.984x | 0.970x | 1.957x | 3.557x | 2.129x | 1.440x | 1.116x |
| `B8,S2048,D128,H16,Hkv4` | 5063.12 | 15473.06 | 3.056x | 3.111x | 0.982x | 3.057x | 4.231x | 1.733x | 1.710x | 1.069x |

- 结论：
  - P3 精度闭环通过，可作为 P4 基线。
  - P3 相对 P2 的 Neumann 绝对耗时小幅下降，`S=16/64/125/2048` 分别为 P2 的 `0.968x/0.931x/0.970x/0.982x`。
  - 4/4 典型 shape 仍是 normal 更快；`S=64+` 主要剩余放大项仍在 AIC scalar、MTE 和当前非 full-local 的 compose/materialization 链路。
  - 本轮仍是典型 shape smoke 性能对比，未裁剪无关 `TilingData/MatmulTiling` 字段，不能作为最终单方案放大性能结论。

## P4 C64 full half L1/TSCM staging 尝试与回退

- 按计划在 P3 精度和典型 shape 性能闭环后进入 P4；本轮只尝试 C64 full half L1/TSCM staging，不进入 P5/P7/P8。
- P4 目标：
  - 将 C64 full matrix 搬入 `aicHalfL1Inv_` / `aicHalfL1Power_`，作为 full half L1/TSCM resident 状态。
  - 4 个 16x16 diagonal inverse 与 16->32、32->64 compose 尽量从 full L1 读写。
  - 最后再将 final inverse 写回 FP16 scratch，后续 AIV 仍从 FP16 scratch cast 到 BF16 `attnWsGm_`；final RHS 继续保持 BF16 `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)`。
- 已尝试的探针/实现：
  - 新增 `SyncMte2ToM()` 作为 full L1 staging 的同步探针。
  - 尝试直接 full-L1 `DataCopy` 路径。
  - 尝试 scalar GM half -> full L1 加载路径。
  - 尝试 blockwise 16x16 compact GM half -> full L1 加载路径。
  - 新增但未启用的实验 helper 包括 `CopyGmHalfMatrixToFullHalfL1Scalar()`、`CopyHalfL1LoadedCompactToFullBlock()`、`LoadGmHalfMatrixToFullHalfL1ByBlocks()`、`Neumann16InverseFullL1Block()`、`Compose32From16FullL1()`、`Compose64From32FullL1()`、`C64InverseBy16x16NeumannFullL1()`。
- P4 最小 smoke 结果：
  - 直接 full-L1 `DataCopy` 路径在 `S=16` smoke 出现约 `1e4` 量级误差。
  - scalar/blockwise full-L1 probe 将误差降到约 `2e-1`，但仍未通过 `S=16` smoke。
  - 由于 `S=16` 不触发 compose，当前失败点应优先定位 full-L1 staging、full-to-compact 读写或写回基础路径，不应先怀疑后续 32/64 compose 数学。
- Gate 处理：
  - P4 未通过最小 `S=16` smoke，按计划要求不能验收为优化项。
  - 未运行 P4 全量 compare。
  - 未运行 P4 典型 shape 性能；未通过精度的优化不得记录性能作为有效结果。
  - 未进入 P5/P7/P8。
- 当前 active path 已恢复为 P3 数据流：
  - `C64InverseBy16x16Neumann()` 当前调用 `Neumann16InverseBlock()`、`Compose32From16()`、`Compose64From32()`。
  - 当前不调用 `C64InverseBy16x16NeumannFullL1()`。
  - P4 实验 helper 暂留在文件中，作为后续 full-L1 copy/writeback probe 的调试代码；默认执行路径不使用它们。
- 恢复后验证：
  - 构建：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`，结果成功。
  - 安装：`bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`，结果成功。
  - smoke：`S=16` PASS，`S=64` PASS。
  - 全量 compare：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.18s`。
- 当前结论：
  - P3 仍是当前可用基线。
  - P4 当前阻塞在 full half L1/TSCM staging 的基础读写闭环，不能作为下一步优化基线。
  - 后续继续 P4 时应先做独立 roundtrip probe：`64x64 half GM -> full half L1 -> 指定 16x16 compact block -> half GM`，与 P3 compact 路径对齐后再重新接入 Neumann。

## 本轮答疑：停下原因与当前计算/搬运链路

- 用户询问“为什么停下来了”和“目前整个计算和搬运链路是什么”。
- 停下原因澄清：
  - 本轮不是因为 P4 完成，而是因为 P4 未通过 `S=16` 最小 smoke 精度门槛。
  - 按 `C64_1616_FAST_INVERSE_OPTIMIZATION_ORDER.md` 的 gate，未通过 smoke 时必须停止叠加后续优化，不能继续进入 P5/P7/P8，也不能跑 P4 性能作为有效结果。
  - 当前可用基线仍是 P3；P4 helper 只作为后续调试代码暂留，默认路径不调用。
- 当前 active 计算/搬运链路澄清：
  - host workspace 为 BF16 主链路 workspace 加 FP16 `L_powHalfWsGm_`、FP16 `INVInitHalfWsGm_`、FP32 `tmpWsGm_`、FP32 `INVWsGm_` 四块 Neumann workspace。
  - AIV 预处理 query/key 到 BF16 连续 workspace；AIC 做 BF16 `key@key` 到 `kkWsGm_`、BF16 `query@key` 到 `outQkGm_`。
  - AIV 从 `kkWsGm_` 读 BF16 到 UB float，乘 beta、gamma/mask 后写 FP16 `L_powHalfWsGm_` 与 FP16 初始 `INVInitHalfWsGm_`。
  - AIC 对 16x16 block 从 FP16 GM 搬到 compact half L1/TSCM，L1->L0A/L0B，MMAD 到 L0C，Fixpipe 回 half L1/TSCM；三轮 Neumann 后写回 FP16 `INVInitHalfWsGm_`。
  - 16->32 与 32->64 compose 当前仍是 FP16 GM block 输入输出：从 `INVInitHalfWsGm_` 搬 block 到 compact half L1/TSCM，MMAD/Fixpipe 后写回 `INVInitHalfWsGm_`。
  - AIV 最终从 FP16 `INVInitHalfWsGm_` 读 inverse，half->float->BF16 cast，写 BF16 `attnWsGm_`。
  - AIV 生成 BF16 `gBKWsGm_` 和 BF16 `vBetaWsGm_`；AIC 后续 final RHS 仍用 BF16 GM `AICProcess(attnWsGm_, gBKWsGm_, ...)` 与 `AICProcess(attnWsGm_, vBetaWsGm_, ...)`。
  - 因此当前不是 full-local resident；它是“16x16/compose 局部 half L1 resident + FP16 GM scratch 串联”的 P3 路径。

## P4 继续执行中的命名纠正与探针结论

- 用户指出“C48 是什么，chunk 不是 64 吗？”这一点成立：后续禁止把 P4 tail 分支简称为 `C48`。
  - host `chunkSize` 一直是 64。
  - 之前记录里的 `C48` 只表示单个 64 chunk 内 `validLen` 覆盖到第三个 16x16 对角块，即 `33..48` tail 范围。
  - 后续文档与汇报统一改称 `validLen<=48`、`tail<=48` 或 `33..48 tail`，避免和 `chunkSize=64` 混淆。
- P4 partial 验证状态补充：
  - `validLen<=48` partial full-L1 diagonal 探针：此前 full compare 已通过，典型 shape Neumann/normal ratio 为 `S16 1.215x`、`S64 1.630x`、`S125 1.891x`、`S2048 3.085x`。这是目前 P4 partial 中相对较好的性能观察。
  - `49..63` 第四对角块 full-L1 + GM compose 探针：full compare `20 passed`，但典型 ratio 退化为 `S16 1.210x`、`S64 1.630x`、`S125 2.233x`、`S2048 3.058x`。
  - `49..63` lower-right `16->32` full-L1 compose 探针：full compare `20 passed`，但典型 ratio 进一步退化为 `S16 1.226x`、`S64 1.659x`、`S125 2.654x`、`S2048 3.122x`。
  - 将完整 `validLen==64` 也切入上述 tail full-L1/lower-right compose 分支后，全量 compare 失败：`18 passed, 2 failed`，失败 case 为 `b1_t129_tail_d64` 与 `b2_t160_h32_h8_d64`；回退 `validLen==64` 后 full compare 恢复 `20 passed in 6.15s`。
  - 回退 `validLen==64`、保留 `49..63` lower-right full-L1 compose 的典型 ratio 为 `S16 1.212x`、`S64 1.692x`、`S125 2.648x`、`S2048 3.085x`，性能仍劣于 P3 与 `validLen<=48` partial。
- 对“full-local compose 怎么可能比 GM 差”的口径修正：
  - 当前这些 helper 只是 full-L1/compact block 探针，不是真正完整的 fast inverse full-local compose。
  - 探针仍包含 full L1 与 compact tile 之间的额外搬运、raw MMAD/Fixpipe 写回、cache flush/sync 和 matmul local buffer restore；部分路径还会继续 materialize 到 GM 后再做后续 compose/final BF16 `attnWsGm_`。
  - 因此当前性能退化说明“这个朴素探针实现固定开销过大”，不能证明真正的 packed/full-local compose 会比 GM 路径差。
  - 真正 fast inverse 主线仍应减少 full/compact 来回搬运，优先走 packed diagonal 16x16 Neumann 与 odd/even packed compose，而不是把每个小 block 通过 full-L1/compact helper 逐个搬来搬去。
- 中断时源码状态：
  - 源码中已插入一个新的 `validLen==64` diagonal-only full-L1 探针：四个 16x16 对角块尝试走 full-L1，lower-right `16->32` 和最终 `32->64` compose 仍走现有 GM half compose。
  - 该版本仅完成 build，尚未 install/smoke/full compare；不得作为可用基线或性能结论。
  - 最近一次已安装且 full compare 通过的版本是“`validLen==64` 回退后”的版本，但其 `49..63` lower-right full-L1 compose 性能已证实退化，不建议作为后续优化基线。

## P4 阶段性能口径修正

- 用户修正性能测试策略：性能应按阶段验收测一次，不是每个 P4 子探针都跑典型 shape。
- 已同步更新 `docs/2026-05-07/C64_1616_FAST_INVERSE_OPTIMIZATION_ORDER.md`：
  - 每个完整优化阶段结束后测一次典型 shape 性能。
  - 阶段内部 probe/substep 只做精度和诊断，不逐个跑性能。
  - P4 完成全量精度、跑完一次阶段性能后必须暂停，等待用户确认后再进入 P5/P7/P8。
- 已执行的 P4 partial 典型 shape profiling 仅作为诊断 smoke 记录，不作为 P4 阶段性能结论，也不作为后续对外汇报口径。
- P4 结束条件重新明确：
  - active path 覆盖完整 `chunkSize=64` 的 C64 full half L1/TSCM staging。
  - `validLen=1..64` 的 16x16 diagonal inverse、16->32 compose、32->64 compose 数据流闭环清楚，且不依赖未验收的 tail-only shortcut。
  - build/install 成功。
  - smoke 至少覆盖 `S=16/17/33/48/49/63/64/65/125/129/160` 中的关键边界。
  - 全量 compare `20 passed`。
  - 只有上述都完成后，才跑一次 P4 阶段典型 shape 性能。
- 当前距离 P4 结束的状态：
  - P4 的基础 full-L1 staging roundtrip 还没形成完整可用闭环。
  - tail-only partial 有精度通过记录，但不能代表完整 P4。
  - 完整 `validLen==64` 接入 lower-right full-L1 compose 曾导致 full compare 失败。
  - 中断时源码里新插入的 `validLen==64` diagonal-only full-L1 探针仅 build 过，尚未完成 install/smoke/full compare，不能计入 P4 完成度。

## P4 full-L1 diagonal staging 验收与暂停

- 按用户要求继续 P4，且 P4 内部 probe 不再跑性能；只在 P4 精度闭合后跑一次典型 shape 性能。
- 当前验收版本口径：
  - active selector 仍为 `fulllocalnew/C64_1616_FP16/chunkSize=64`。
  - `C64InverseBy16x16Neumann()` 调用 `C64InverseBy16x16NeumannFullL1(paraIdx)`。
  - 入口将 FP16 `L_powHalfWsGm_` / `INVInitHalfWsGm_` 搬入 C64 full half L1/TSCM。
  - 四个 16x16 diagonal inverse 走 full-L1 block Neumann。
  - top-left `16->32`、lower-right `16->32` 与最终 `32->64` compose 均回退到已验证的 FP16 GM compose path；full-L1 compose 未作为 P4 验收内容。
  - final RHS 保持 BF16 GM `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)`，未修改 Stage2/Stage3、`gBK/vBeta` 生成或 QK 链路。
- 关键定位结论：
  - 仅最终 `32->64` 回退 GM 后，`B=1,S=129,D=64` PASS，但 `B=2,S=160,Hq=32,Hkv=8,D=64` 仍 out-only FAIL。
  - 将 lower-right `16->32` 回退 GM 后，`B=1,S=129,D=64` PASS；`B=2,S=160,Hq=32,Hkv=8,D=64` 仍 out-only FAIL。
  - 再将 top-left `16->32` 也回退 GM 后，`B=2,S=160,Hq=32,Hkv=8,D=64` PASS。
  - 因此 P4 当前可验收范围是 full-L1 diagonal staging；朴素 full-L1 compose 对完整 64 chunk 仍有精度风险，不能作为后续基线。
- 构建与安装：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - 结果：成功。
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 结果：`SUCCESS`。
- smoke 精度：
  - `S=16/17/33/48/49/63/64/65/125/129/160`，`seed=0`，`B=1,nk=4,nv=4,dk=dv=64`：全部 PASS。
  - 复测旧失败点 `B=1,S=129,seed=33,nk=4,nv=4,dk=dv=64`：PASS。
  - 复测旧失败点 `B=2,S=160,seed=40,nk=8,nv=32,dk=dv=64`：PASS。
- 全量精度：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.23s`。
- P4 阶段典型 shape 性能：
  - 输出目录：`tests/pytest/prof_out/fulllocalnew_c64_1616_fp16_p4_diag_full_l1_gm_compose_typical_20260507/`。
  - 命令：`TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext CGDR_GOLDEN_CHUNK_SIZE=64 python3 tests/pytest/run_chunk_gdr_performance_msprof.py --output tests/pytest/prof_out/fulllocalnew_c64_1616_fp16_p4_diag_full_l1_gm_compose_typical_20260507 --ops both --case prefill_b8_s16_d64_h16_hkv4 --case prefill_b8_s64_d64_h16_hkv4 --case prefill_b8_s125_d64_h16_hkv4 --case prefill_b8_s2048_d128_h16_hkv4`
  - 结果文件：`performance_compare.csv`、`performance_raw.csv`。

| case | normal task(us) | neumann task(us) | P4 ratio | AIC ratio | AIC MAC ratio | AIC scalar ratio | AIC MTE1 ratio | AIC MTE2 ratio | Fixpipe ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B8,S16,D64,H16,Hkv4` | 649.74 | 834.22 | 1.284x | 1.281x | 1.646x | 1.665x | 1.174x | 0.725x | 0.605x |
| `B8,S64,D64,H16,Hkv4` | 667.22 | 1128.18 | 1.691x | 1.688x | 4.803x | 3.250x | 2.347x | 1.911x | 1.969x |
| `B8,S125,D64,H16,Hkv4` | 844.28 | 1822.88 | 2.159x | 2.163x | 4.727x | 3.764x | 2.451x | 2.151x | 1.838x |
| `B8,S2048,D128,H16,Hkv4` | 5081.30 | 17538.76 | 3.452x | 3.452x | 2.852x | 4.629x | 1.942x | 2.513x | 1.575x |

- P4 阶段结论：
  - P4 full-L1 diagonal staging 已完成 build/install/smoke/full compare/一次典型性能闭环。
  - 性能仍是 normal 更快；相对 P3，P4 在 `S=16` 退化明显、`S=64/125/2048` 基本接近或略慢，主要风险仍在 AIC scalar、MTE/Fixpipe 和 full/compact/GM materialization 开销。
  - full-L1 compose 尚未通过验收；后续不能把 P4 解释为“完全 full-local compose”。
  - 按用户规则，P4 阶段性能完成后暂停，不进入 P5/P7/P8，等待用户确认。

## P4 暂停原因与 fulllocal 难点澄清

- 本次暂停不是因为已经做到完整 fulllocal，而是执行用户此前明确的阶段规则：P4 完成精度闭环并跑完一次阶段典型性能后暂停，等待确认后再进入后续阶段。
- 当前 P4 只完成了 diagonal inverse 的 full half L1/TSCM staging：
  - 四个 16x16 diagonal Neumann block 可以从 FP16 GM 搬入 C64 full half L1/TSCM，在片上完成局部求逆。
  - top-left/lower-right `16->32` compose 和最终 `32->64` compose 仍回退到 FP16 GM compose。
  - final RHS 仍通过 BF16 GM `attnWsGm_` 调用 `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)`。
- 距离完整 fulllocal 还差两类闭环：
  - compose resident 闭环：`16->32` 与 `32->64` compose 不能再依赖 FP16 GM 中间态，需要稳定地在 full L1/TSCM 或 packed tile 中完成。
  - final LHS resident 闭环：最终 `INV/attn` 不应先写 BF16 `attnWsGm_` 再被两个 RHS matmul 从 GM 读取；但当前边界暂时要求 final RHS 仍保持新版 BF16 GM 路径。
- 主要难点是 full-L1 compose 的正确性，不是简单把 GM compose 的输入输出换成 L1：
  - 直接 full-L1 `32->64` compose 曾在 `S=33`/完整 64 chunk case 中产生明显漂移。
  - lower-right `16->32` full-L1 compose 修复部分 case 后，`B=2,S=160,nv=32` 仍有 out-only 漂移。
  - top-left `16->32` 也回退 GM 后旧失败点才全部 PASS。
  - 这说明 full/compact offset、tail padding、block 顺序、Fixpipe 写回和后续读取之间仍存在未闭合的布局/同步问题。
- 后续如果继续，不应把 P4 当成完整 fulllocal 基线；应单独开启 full-L1 compose 修复阶段，或直接转向 packed diagonal / odd-even packed compose，并重新按单项 gate 验收。

## P4 执行口径再次修正

- 用户修正规则：P4 必须按整个 fulllocal 阶段完成才停，期间不要测性能，只测精度。
- 新执行口径：
  - P4/full-local 改造期间只跑 build/install、边界 smoke 和 full compare。
  - full-L1 diagonal staging 只能算 P4 子项，不能作为 P4 结束点。
  - 只有完整 fulllocal active path 闭环并通过精度后，才跑一次典型 shape 性能并暂停。
- 因此后续继续推进 full-L1 compose 或 packed/odd-even compose 时，不再对中间 probe 做 msprof 性能测试。

## P4 full-L1 compose 再次尝试与回退

- 按用户要求继续修 P4 正确性；本轮不跑性能，只做 build/install 和精度验证。
- 先确认 active selector 仍为 `fulllocalnew/C64_1616_FP16/chunkSize=64`，且上次失败的 `0xe` 双 flag 同步已无残留；AIC/AIV 仍只使用 `0x7` 作为 AIV 写 FP16 `L/I-L` 后的同步点。
- 新增诊断 fallback：`Compose64From32FullL1ViaGm()` 将 full-L1 中的 top/bottom/lower 32x32 写回 FP16 `INVInitHalfWsGm_`，调用已验证的 `Compose64From32()`，再把 lower-left 32x32 结果加载回 full-L1。
  - build/install 成功。
  - smoke：`S=16/17/33` 均 PASS。
  - 结论：full-L1 diagonal 输入、最终写回和 AIV half->BF16 cast 链路不是当前 `S=33` 失败主因；问题集中在真正 resident 的 `32->64` compose 数学/布局闭环。
- 随后将 `Compose64From32FullL1()` 改成直接 `HalfFullL1ComposeBlock(..., blockSize=32)` 的 resident compose。
  - build/install 成功。
  - `S=33` 失败：`CPU vs ChunkGDRNeumann state FAIL max_abs=9.669919e-02`，`ChunkGDR vs ChunkGDRNeumann state FAIL max_abs=9.655762e-02`。
  - 该误差与此前 direct full-L1 `32->64` compose 失败一致，说明本轮直接 resident 32x32 compose 仍未闭合。
- 按用户指示回退 active 到 P3：
  - `C64InverseBy16x16Neumann()` 不再调用 `C64InverseBy16x16NeumannFullL1()`，恢复为 P3 的 `Neumann16InverseBlock()` + `Compose32From16()` + `Compose64From32()`。
  - 保存 P3 版本到 `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3_baseline.h`，并修正独立 include guard。
- 回退验证：
  - build/install 成功。
  - `S=33` smoke PASS。
  - 全量 compare：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.53s`。
- 本轮没有执行任何性能测试。当前可用 active 基线是 P3；P4 full-local compose 未完成，需要和用户商量下一步是继续修 full/compact NZ 布局，还是转向 packed diagonal / odd-even packed compose。

## 精度定位方法沉淀与探针清理

- 按用户要求，将本轮 P4/full-local compose 精度定位经验沉淀为通用 skill：`SKILL/fulllocal-compose-precision-debugging/SKILL.md`。
  - skill 不再写死为 `C64_1616 P4/full-L1 compose` 专用，而是作为 fulllocal、compose、resident/L1/TSCM、GM fallback、边界 shape 精度问题的通用定位流程。
  - 核心方法：确认 selector/chunk 口径 -> 找最小失败边界 -> 用可信 GM fallback 或已通过路径隔离 -> 判断输入 materialize、resident 计算、写回、cast、后续消费哪个环节出错 -> 先正向修复当前目标；只有明确放弃候选或需要重建干净工作面时，才恢复到干净全量通过基线。
  - 本次 `S=33` resident `32->64` compose 失败作为案例保留：direct resident 失败 state `max_abs=9.669919e-02`，GM fallback `S=16/17/33` PASS，说明问题集中在 resident compose 本体，而不是 diagonal 输入、final 写回或 AIV half->BF16 cast。
- `AGENTS.md` 已接入该 skill：后续遇到 fulllocal/compose/resident/L1/TSCM/GM fallback/边界 shape 精度失败或基线回退问题，先读取该 skill。
- 清理多余探针程序：
  - `op_kernel/fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocalnew.h` 已清理未通过的 P4 full-L1 resident compose/probe helper。
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3_baseline.h` 已同步为干净 P3 版本，仅保留 P3 已通过路径和独立 include guard，不再携带未启用 probe/helper。
  - 静态检查 active/P3 文件中 `Probe|FullL1|Compose64From16Tiles|HalfFullL1|MaterializeInitialInv|aicHalfL1Inv_` 等探针关键词均无残留。
- 验证结果：
  - 构建成功：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
  - 安装成功：`../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`。
  - `S=33` 边界 smoke PASS：`CGDR_GOLDEN_CHUNK_SIZE=64 ... compare_chunk_gdr_three_way.py --seed 0 --batch-size 1 --seqlen 33 --nk 4 --nv 4 --dk 64 --dv 64`。
  - 全量 compare PASS：`CGDR_GOLDEN_CHUNK_SIZE=64 ... bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.57s`。
- 本轮没有执行性能测试，符合“P4/定位期间只测精度”的规则。

## P7/P8 packed fast inverse 方案阅读

- 按用户要求阅读今日进度文档，并对照 `pto-kernels v0.1.2` 的 `kernel_tri_inv_rec_unroll.cpp` 与 `gdn-tri-inverse` fast inverse 文章。
- 当前状态确认：
  - active path 已回退到 P3。
  - P4 direct resident `32->64` compose 仍在 `S=33` 失败，state `max_abs=9.669919e-02`。
  - GM fallback 诊断 `S=16/17/33` PASS，说明问题集中在 resident compose 本体。
- 对 `CopyDiagonalFractalsL1ToL0()` 的理解：
  - 它把 full matrix L1 中的 diagonal 16x16 fractal 抽取到 L0A/L0B 的 block-diagonal 位置。
  - 对 C64 来说，四个 16x16 diagonal block 可被打包成一个 64x64 block-diagonal tile。
  - 后续一次 packed MMAD 可同时完成四个独立 16x16 diagonal block 的 `power@power` 或 `inv@power`，减少逐块 tiny MMAD 的调度、搬运和同步成本。
- 对 `CopyOddOrEvenBlocksL1ToL0()` 的理解：
  - 它不是直接在 full L1/NZ 大矩阵里按子块 offset 做乘法，而是把当前 `X` 的 even/odd diagonal block 结构化搬到 L0A/L0B。
  - L0A 侧放 even blocks，L0B 侧放 odd blocks，目标 tile 预先清零，未参与 block 保持 0。
  - 对应 Unrolled-MBH 的 `LX/RX`，用于 packed compose：`X = (LX @ (-M) + I) @ RX + LX`。
- 对当前算子的映射：
  - P7 替代当前四次 `Neumann16InverseBlock()` 的逐块执行方式。
  - P8 替代当前 `Compose32From16()` 和 `Compose64From32()` 的 GM/局部 half-L1 compose 方式。
  - final RHS 仍保持 BF16 GM `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)`，不触碰 Stage2/Stage3、`gBK/vBeta`、QK 链路。
- 新增方案文档：`docs/2026-05-07/P7_P8_PACKED_FAST_INVERSE_PLAN.md`。
- 下一步建议：
  - 不继续沿朴素 P4 full-L1 compose offset 修补主线推进。
  - 先做 P7 packed `block@I` 探针，验证 diagonal 16x16 抽取、L0 block-diagonal offset、L0C 输出散回。
  - P7 packed diagonal inverse 全量 compare 通过后，再接 P8 odd/even packed MBH compose。
  - P7/P8 完整精度通过前不跑性能。

## P7/P8 方案修正：set2d 与 I-L 初始矩阵

- 用户提醒 `set2d` 可以批量设置 L1 内存初始值为 0，这一点应直接用于 P7/P8 packed tile 初始化。
- 已修正 `P7_P8_PACKED_FAST_INVERSE_PLAN.md`：
  - P7 packed diagonal tile 先用 `set2d` 清零，再填四个 16x16 diagonal fractal。
  - P8 even/odd packed tile 先用 `set2d` 清零，再填本轮 even 或 odd blocks。
  - off-diagonal zero-fill 是 packed MMAD 正确性的前置条件，不能默认用逐元素标量循环作为实现主线。
- 同时复核当前 P3 `InverseCompute()`：
  - Stage1 原本就有 `I-L` 简单构造逻辑。
  - 当前 P3 仍是 AIV 从 `kkWsGm_` 生成 `L`，再 `Muls(..., -1)` 得到 `-L`，对 diagonal `SetValue(..., 1.0f)` 得到 `I-L`。
  - P3 没有把 `I-L` 初始矩阵生成复杂化，主要变化只是输出到 FP16 `L_powHalfWsGm_` / `INVInitHalfWsGm_` scratch，并在写出后做 cache 可见性处理。

## P7/P8 实施后当前状态

- 当前 selector 仍为 `fulllocalnew/C64_1616_FP16/chunkSize=64`。
- 检查发现 `C64InverseBy16x16Neumann()` 曾残留一次 P7 active 尝试：入口调用 `P7PackedDiagonalNeumannGroup()`，且 `P7PackedDiagonalNeumannGroup()` 内部使用 full `HalfL1NeumannCompactBlock(groupSize)`。
- 已将 active 入口恢复为精度通过的保守组合：
  - diagonal inverse 使用 `Neumann16InverseBlock(paraIdx, 0/16/32/48)`。
  - `validLen > 16` 时执行 `Compose32From16(paraIdx, 0)`。
  - `validLen > 48` 时执行 `P8ComposeBoth32From16Scratch64(paraIdx)`。
  - `validLen > 32` 时执行 `P8Compose64From32Scratch64(paraIdx)`。
- 已将 `P7PackedDiagonalNeumannGroup()` 恢复为调用 `P7HalfL1DiagonalNeumannBlock(groupSize)` 的诊断 helper；P7 目前不在 active path。
- P8 当前已进入 active compose：
  - `P8ComposeBoth32From16Scratch64()` 使用 64x64 FP16 L1 scratch 同时完成两个 `16->32` compose。
  - `P8Compose64From32Scratch64()` 使用 64x64 FP16 L1 scratch 完成 `32->64` compose。
  - 相关 scratch 均通过 `ZeroHalfL1()` 批量清零；`ZeroHalfL1()` 内部使用 `InitConstValue`，不是逐元素标量清零。
- 构建安装重新验证：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16` 成功。
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp` 成功。
- 精度验证：
  - smoke `S=17/32/33/49/64/129` 全部 PASS。
  - full compare：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.22s`。
- P7 尝试结论：
  - 原始 packed 32 diagonal active 尝试在 `S=17` 精度失败。
  - diagonal-only 写回诊断在 `S=17` 超时。
  - full 32 block-diagonal MMAD 保守尝试仍在 `S=17` 精度失败。
  - sub-agent 结论中 `CompactMatrixNzOffset(16,16,32)=768` 保持正确；主要风险更可能是 L0/L1 packed diagonal 写回或布局语义，而不是简单 offset 改成 512。
- 当前结论：P8 已实现并通过当前精度 gate；P7 未通过，不进入 active path。当前可交付状态是 `P3 diagonal inverse + P8 scratch64 compose`，final RHS 仍保持 BF16 GM matmul。
