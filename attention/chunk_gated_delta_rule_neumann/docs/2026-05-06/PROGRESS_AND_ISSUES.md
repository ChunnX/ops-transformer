# 2026-05-06 进度与问题

## 当前状态

- 2026-05-06 当日先完成 `fulllocal/C64_3232_FP16` 历史单点闭环，随后用户将当前目标收窄为 FP16 `C16` 与 FP16 `C64_1616` fulllocal；以下当前状态以最新 active selector 为准。
- 当前 selector：`fulllocal/C64_1616_FP16`。
- 当前 include：`op_kernel/selected/stage1_selected.h` 指向 `../fulllocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocal.h`。
- 当前配置：`CGDR_NEUMANN_SELECTED_SUITE="fulllocal"`，`CGDR_NEUMANN_SELECTED_VARIANT="C64_1616_FP16"`，`CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`。
- `baseline/` 与 `halflocal/` 未修改，继续作为稳定参考和回退实现。
- 用户恢复 `TilingData` 结构体后，已重新 build/install/compare 验证 `fulllocal/C16_FP16` 与 `fulllocal/C64_1616_FP16`，二者均通过。

## 最新 FP16 fulllocal 状态

- `fulllocal/C16_FP16`：
  - selector/config 需为 `fulllocal / C16_FP16 / chunkSize=16`。
  - 恢复 `TilingData` 后重新构建安装通过。
  - 精度：`CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.01s`。
  - smoke 性能 CSV：`tests/pytest/prof_out/fulllocal_c16_fp16_smoke_20260506/performance_compare.csv`。
  - smoke 结果：`S=16` Neumann/normal `1.129x`，`S=64` `2.184x`。
- `fulllocal/C64_1616_FP16`：
  - 当前 active selector/config：`fulllocal / C64_1616_FP16 / chunkSize=64`。
  - `FP16_L1_RESIDENT_INVERSE=true`，`FP16_L1_RESIDENT_COMPOSE16=true`，`FP16_L1_RESIDENT_COMPOSE32=false`。
  - 恢复 `TilingData` 后重新构建安装通过。
  - 精度：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.20s`。
  - after-restore smoke 性能 CSV：`tests/pytest/prof_out/fulllocal_c64_1616_fp16_smoke_after_tiling_restore_20260506/performance_compare.csv`。
  - after-restore smoke 结果：`S=16` Neumann/normal `1.234x`，`S=64` `1.722x`。
- 这两组 smoke 均是在恢复后的完整 `TilingData/MatmulTiling` 结构体上跑的简单性能测试，未按单方案放大规则裁剪无关 tiling 字段，因此只作为观察，不作为最终性能结论。
- 静态扫描确认两条 fulllocal 当前均未重新引入 RHS resident/raw final：final RHS 仍是 `AICProcess(INVWsGm_, gBKWsGm_/vBetaWsGm_, ...)`；`gBK/vBeta` 仍是 FP32 GM workspace；QK、Stage2、Stage3 未作为本轮修改范围。

## 已完成

- 已读取并继续执行 `docs/2026-05-05/FULLLOCAL_SINGLE_POINT_BREAKTHROUGH_PLAN.md`。
- 已开 sub-agent 做只读复核，确认缺失点为 `Compose32From16()` / `Compose64From32()` resident 分支中的 scalar full-L1 backfill。
- 已恢复两处 `CopyGmFloatMatrixToHalfL1()` 调用。
- 双算子包构建通过并安装到 `/home/developer/Ascend/cann-9.0.0/opp`。
- `S=16` smoke 通过，覆盖单 16x16 resident path。
- `S=17` smoke 通过，覆盖 16x16 resident + `Compose32From16()` path。
- 全量三方 compare 最终通过：`20 passed in 6.17s`。
- 指定四点性能结果已生成，路径为 `tests/pytest/prof_out/fulllocal_C64_3232_FP16_selected_s_20260506/performance_compare.csv`。

## C64 完成口径

- fulllocal 单点 `C64_3232_FP16` 已完成精度闭环。
- 完整链路是 16x16 resident Neumann + 32/64 compose；final RHS 仍保持 FP32 GM `MatmulImpl` 稳定消费端。
- direct 32x32 resident 不作为当前完成路径；该路径仍因 `S=17` 大误差 / V2 `nan` 阻塞。
- 性能不达突破：四点 Neumann/normal ratio 为 `1.307x / 1.957x / 2.286x / 3.011x`，normal 仍更快。

## 问题与风险

- fulllocal 性能仍被 AIC scalar、MTE/Fixpipe、GM workspace 往返和同步成本拖慢。
- 两处 compose scalar backfill 当前对稳定性仍重要；删除后会导致多 case compare 失败，说明 full L1 state/cache/order 仍未完全理顺。
- 全量 compare 第一轮出现过 `b1_t129_tail_d64` 单 case out 偶发失败；隔离复跑和第二轮全量 compare 均通过。后续若继续调 resident layout，应保留该 case 作为稳定性哨兵。
- direct 32x32 resident 的 NZ->L0A/L0B layout/Fixpipe 闭环尚未关闭，不能切回主线。

## 下一步建议

1. 若继续追性能，优先分析为什么 scalar full-L1 backfill 对正确性有影响，再消除该隐性依赖。
2. 暂不继续 direct 32x32 resident，除非先做独立 32x32 `Fixpipe CFG_NZ -> LoadData A/B -> MMAD` 数值探针。
3. 若目标是超过 normal，需要减少 GM workspace 往返和 tiny MMAD 固定调度，而不是只保留当前 half L1 resident 形态。

## 实现口径补充

- 当前 fulllocal 不是整个 Stage1 的全程 local resident。
- 已 fulllocal 化的范围：16x16 Neumann inverse 的 `power` 和 `INV += INV@P` 在 half L1/TSCM + L0C/CMatrix 中完成。
- 仍依赖 GM 的范围：
  - AIV 初始 `L` / `I-L` 生成。
  - 每个 16x16 inverse 结果写回 `INVWsGm_`。
  - 32/64 compose lower-left block 通过 GM raw MMAD 完成。
  - final RHS `INV @ k_cumdecay` / `INV @ v_beta` 仍为 FP32 GM MatmulImpl。
- 当前主路径不使用 direct `Neumann32InverseBlock()`；实际是 `16x16 resident -> Compose32From16 -> Compose64From32`。

## 性能原因补充

- fulllocal 当前慢于 halflocal 的主因是额外搬运/格式转换/同步，而不是 Neumann 公式或理论 MAC 增多。
- halflocal 在 FP32 GM/raw MMAD 路径上直接做 power、`INV += INV@P` 和 compose；fulllocal 需要反复在 FP32 GM、L0C、half L1/TSCM、L0A/L0B 之间转换。
- fulllocal 的 half L1 resident 只覆盖 16x16 inverse 子流程，结果仍写回 GM；compose 和 final RHS 没有真正留在片上，因此未形成“少一次 GM 往返”的净收益。
- 当前最大差异来自 AIC scalar 时间放大：`S=64/125/2048` 分别从 halflocal 的 `303.115/541.096/8294.395us` 增至 fulllocal 的 `533.904/1009.908/15529.760us`。

## 概念口径：局部 resident

- 局部 resident 指只在一个局部计算子图内保持片上驻留，不代表整个 Stage1 或整个 inverse 流程都不落 GM。
- 当前 fulllocal 的 resident 边界是 16x16 Neumann block 内部；block 完成后仍写回 `INVWsGm_`。
- 真正的全程 resident 目标应至少覆盖：16x16/32x32/64x64 compose、final RHS 输入消费，以及跨 round/跨 block 的下一次读取。

## C16 方向判断

- `C16_FP16` / `C16_FP32` 可以复用当前 16x16 half L1/TSCM resident inverse 思路。
- C16 的优势是没有 32/64 compose，能删除当前 C64 fulllocal 中最明显的 compose GM 往返和 backfill。
- 但当前 C16 主流程仍会在 inverse 后把 `INV` 写回 GM，并用 GM MatmulImpl 做 final RHS，所以不能直接称为完全 resident。
- 后续若实现 C16 fulllocal，建议先验证 “C16 inverse resident + final RHS 仍 GM” 的精度/性能，再单独推进 “final RHS 也 consume L1 resident INV”。

## C16 fulllocal 迁移状态

- 已按用户要求完成 `C16` Stage1 fulllocal 迁移，`final RHS` 暂不处理。
- 当前 active 文件：`op_kernel/fulllocal/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocal.h`。
- 当前 active selector：`fulllocal/C16_FP16/chunkSize=16`。
- 已开 sub-agent 做只读复核，确认 selector、实现边界和性能脚本入口。
- `FP16_L1_RESIDENT_INVERSE=true`，`FP32_SCALAR_RESIDENT_INVERSE=false`。
- `Neumann16InverseBlock()` 使用 half L1/TSCM resident inverse：
  - GM FP32 `L/INV` 通过 raw MMAD 搬到 half L1/TSCM。
  - `HalfL1NeumannCompactBlock()` 在 half L1/TSCM 与 L0A/L0B/L0C 内完成 inverse round。
  - 最终 inverse 仍写回 `INVWsGm_`。
- `final RHS` 当前仍是 GM 路径：
  - `AICProcess(INVWsGm_, gBKWsGm_, outKCumdecayGm_, ...)`
  - `AICProcess(INVWsGm_, vBetaWsGm_, outVInnerGm_, ...)`

## C16 fulllocal 验证结果

- 构建通过：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
- 安装通过：`bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`。
- 精度通过：`CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 5.98s`。
- 选点性能已生成：
  - `tests/pytest/prof_out/stage1_fulllocal_C16_FP16_s_selected_20260506/performance_compare.csv`

## C16 fulllocal 性能结论

- Neumann/normal task duration ratio：
  - `S=16`: `1.084x`
  - `S=64`: `2.140x`
  - `S=125`: `2.503x`
  - `S=2048`: `4.076x`
- 相比 `halflocal/C16_FP16`：
  - `S=16`: `1.000x`
  - `S=64`: `1.007x`
  - `S=125`: `1.011x`
  - `S=2048`: `0.996x`
- 结论：C16 fulllocal 当前精度闭环，但性能只与 C16 halflocal 基本持平，未形成明确收益。
- 原因：当前只去掉了 C64 compose，但 inverse 仍有 GM->half L1 搬运、Fixpipe、identity 构造和写回 GM；`final RHS` 仍从 GM 读取 `INV`，没有把 resident 数据直接消费掉。

## 下一步建议补充

1. 若继续推进 C16，下一步应单独实现 `final RHS` 直接消费 half L1/TSCM 中的 `INV`，或在写回 GM 前完成 `INV @ gBK` 与 `INV @ vBeta`。
2. 在 RHS resident 前，不建议把当前 C16 fulllocal 当作性能突破方案；它更适合作为精度已通过的 resident inverse PoC。
3. 若需要回退稳定性能口径，selector 可切回 `halflocal/C64_3232_FP32` 或对应 C16 halflocal，但必须重新 build/install/compare。

## Fast inverse 参考结论

- 已新增分析文档：`docs/2026-05-06/FAST_INVERSE_REFERENCE_ANALYSIS.md`。
- 参考对象：
  - `gdn-tri-inverse` fast inverse blog。
  - `pto-kernels` 的 `kernel_tri_inv_rec_unroll.cpp`。
- 结论：当前 `fulllocal` 可以参考 fast inverse，但当前实现还只是局部 resident，不能解决全部现有瓶颈。
- fast inverse 对当前最有价值的点：
  - 输入/输出才访问 GM，中间结果留在 L1/L0。
  - 对角块、奇偶块做结构化 fractal 搬运。
  - 使用 L0C 累加和双缓冲重叠搬运/计算。
- 当前缺口：
  - AIV 初始 `L/I-L` 仍写 GM。
  - 16x16 inverse 完成后仍写回 `INVWsGm_`。
  - C64 compose lower-left block 仍依赖 GM raw MMAD 和 scalar backfill。
  - final RHS 仍从 GM 读 `INV`。
- 后续优先级：
  1. C16 final RHS resident PoC，验证 fast inverse dataflow 是否能形成净收益。
  2. C64_3232 compose resident，消除 lower-left GM/backfill 隐性依赖。
  3. GM 往返减少后再做双缓冲和多 tile pipeline。

## C16 final RHS resident 状态

- 已按用户最新要求完成 C16 final RHS resident/raw PoC。
- 当前 active selector 保持：
  - suite: `fulllocal`
  - variant: `C16_FP16`
  - chunk size: `16`
- active include：
  - `op_kernel/selected/stage1_selected.h` -> `../fulllocal/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocal.h`
- 当前实现口径：
  - `FP16_L1_RESIDENT_INVERSE=true`
  - `FP16_L1_RESIDENT_RHS=true`
  - `FP16_L1_WRITEBACK_INV_FOR_DEBUG=false`
  - final RHS 的 `INV` operand 直接使用 `aicHalfL1TileInv_`，不再从 `INVWsGm_` 读回。
  - `gBKWsGm_` / `vBetaWsGm_` 仍是 FP32 GM workspace，进入 RHS raw MMAD 前转成 half L1/TSCM tile。
- 安全边界：
  - `UseResidentRhsForCurrentShape()` 仅在 `dk` 与 `dv` 均为 16 的倍数时启用 resident RHS。
  - 非 16 倍数维度会写回 `INVWsGm_` 并回退原 GM `AICProcess` final RHS。

## C16 final RHS resident 验证结果

- 构建通过：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
- 安装通过：`../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`。
- 全量三方精度通过：`CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 5.96s`。
- 非 16 倍数回退 smoke 通过：`B=1,S=32,nk=4,nv=4,dk=48,dv=80,seed=41`，结果 `PASSED`。

## C16 final RHS resident 性能结论

- RHS resident 性能 CSV：
  - `tests/pytest/prof_out/stage1_fulllocal_C16_FP16_rhs_resident_s_selected_20260506/performance_compare.csv`
- Neumann/normal ratio：
  - `S=16`: `1.109x`
  - `S=64`: `2.423x`
  - `S=125`: `2.671x`
  - `S=2048`: `5.026x`
- 相比 no-RHS-resident C16 fulllocal，Neumann 绝对耗时为：
  - `S=16`: `1.018x`
  - `S=64`: `1.091x`
  - `S=125`: `1.077x`
  - `S=2048`: `1.244x`
- 当前性能判断：
  - RHS resident 已完成数据流目标，但不是性能突破。
  - 瓶颈转移到 RHS FP32 GM -> half L1/TSCM 的转换、raw MMAD 调度、Fixpipe/MTE 同步。
  - 继续扩展前应优先减少 `gBK/vBeta` 的 GM workspace 依赖，或者只在能确认 RHS operand 也片上可消费的路径中启用。

## 当前风险

- RHS resident 仍只是 `INV` operand resident，不是整个 Stage1 all-local。
- `gBK/vBeta` 仍由 AIV 写 GM，AIC 再从 GM 搬入 half L1/TSCM。
- raw MMAD 状态与 `TPipe::Reset()` 生命周期敏感；后续新增分支必须保证两路 RHS 完成后恢复 raw state 与 Stage1 Matmul local buffers。
- 性能数据已显示 RHS resident 更慢，因此该分支当前应作为功能验证线，而不是默认性能最优线。

## RHS 转换口径补充

- `LoadGmFloatTileToHalfL1ByMmad()` 不是理想优化，而是当前为了得到 cube 可消费 half L1/TSCM NZ tile 的绕路。
- 当前路径：`RHS_float_GM @ I_float_GM -> L0C_float -> Fixpipe<half, float, CFG_NZ> -> half L1/TSCM`。
- 普通搬运不能在当前已验证接口里同时完成 FP32->half 数值转换与 L1 NZ layout materialize。
- 后续若要把 RHS resident 做成有效优化，应优先消除这条绕路：让 `gBK/vBeta` 生成时直接进入 half L1/TSCM/NZ，或找到直接 GM->L1 的 dtype+layout 转换接口。

## C64 fast inverse 适配计划状态

- 已新增计划文档：`docs/2026-05-06/C64_FAST_INVERSE_ADAPTATION_PLAN.md`。
- 计划主线：`C64_3232_NeumannCompose`。
- 暂不优先：
  - `C64_1616`，因为 16x16 子块数量和 tiny 调度开销过大。
  - C64 final RHS resident，因为 C16 RHS resident 已证明当前 RHS FP32 GM -> half L1/TSCM 转换会拖慢。
- 优先优化点：
  1. 64x64 full half L1/TSCM block offset 审计。
  2. diagonal 16x16 block packing 探针。
  3. `Compose32From16()` resident 化。
  4. `Compose64From32()` resident 化。
  5. compose resident 通过后，再决定是否做 packed diagonal Neumann round。
- 当前状态：仅完成计划文档，未改 kernel，未切 selector，未运行新验证。

## C64 当前代码复核补充

- 已新增复核文档：`docs/2026-05-06/C64_CURRENT_CODE_OPTIMIZATION_REVIEW.md`。
- 当前 C64 文件不是 active selector；本轮最新复核时 active selector 为 `halflocal/C16_FP16`、`chunkSize=16`。
- 今天 selector 曾多次切换，后续任何 build/install/compare 前必须重新打开 `op_kernel/selected/stage1_selected*.h` 确认真实入口。
- 当前 C64 文件真实状态：
  - `Compose32From16()` / `Compose64From32()` resident 分支已调用 `HalfL1ComposeBlock()`。
  - `HalfL1ComposeBlock()` 内部做 half L1/TSCM 局部 MMAD，但输入输出仍是 GM block。
  - 因此 C64 仍不是 full matrix resident。
- C64 两条线口径：
  - `C64_3232_FP32_halflocal`：`FP16_L1_RESIDENT_INVERSE=false`，稳定 GM/raw MMAD 基线。
  - `C64_3232_FP16_halflocal`：`FP16_L1_RESIDENT_INVERSE=true`，局部 half L1 resident，但仍未做到 full C64 matrix resident。
- 新收益排序：
  1. identity cache，减少重复 `PrepareHalfL1IdentityByMmad()`。
  2. C64 full half L1 staging，用 `aicHalfL1Inv_` / `aicHalfL1Power_` 保存 C64 矩阵状态。
  3. 将 `HalfL1ComposeBlock()` 改成 full L1 block 输入/输出，减少 compose GM 往返。
  4. 真正 L0A/L0B ping-pong overlap 后置。
  5. packed diagonal 16x16 批量 MMAD 后置。
- 结论：所谓“ping-pong 搬运 C64 矩阵”更应先做成 full L1 staging，减少 GM 往返；真正异步 L0 ping-pong 要等 GM/identity 重复成本降下来再做。

## C64 第一阶段口径修正

- 用户明确：其它收益点作为后续优化点，第一步必须按网页 fast inverse 方法优化。
- 已更新 `docs/2026-05-06/C64_FAST_INVERSE_ADAPTATION_PLAN.md`：
  - Phase 1 改为 diagonal fractal/block packing + packed `16x16` MCH/Neumann inverse。
  - Phase 2 改为 odd/even block packing + MBH compose。
  - `identity cache`、L0 ping-pong、RHS resident、减少 `TPipe::Reset()` 等均后置。
- 当前第一步验收口径：
  - 能从 C64 full half L1/TSCM 中抽取四个 `16x16` 对角块，按 block diagonal 形式搬入 L0A/L0B。
  - packed `block @ I`、packed `power @ power` 与逐块 baseline 一致。
  - packed diagonal inverse 写回 full L1 对角位置后，`S=16/17/33/64` smoke 和全量 compare 逐步验证。

## 最新实际状态：回到 halflocal/C16_FP16

- 用户要求收窄修改面后，当前 selector 和已安装包均恢复为 `halflocal/C16_FP16`。
- 当前 include：
  - `op_kernel/selected/stage1_selected.h` -> `../halflocal/chunk_gated_delta_rule_neumann_stage1_C16_FP16_halflocal.h`
- 当前配置：
  - `CGDR_NEUMANN_SELECTED_SUITE="halflocal"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C16_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=16`
- 当前可交付状态：
  - `C16_FP16` inverse 已是真 half L1/TSCM resident inverse。
  - final RHS 仍保持 FP32 GM `AICProcess(INVWsGm_, gBKWsGm_/vBetaWsGm_, ...)`。
  - RHS resident、`gBK/vBeta` half workspace、tail QK、raw final 等扩大修改已清理，不作为当前代码状态。
- 最新可信精度：
  - `CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - `20 passed in 6.03s`

## 当前待解决

- `halflocal/C64_3232_FP16` 构建通过但精度未过：`CGDR_GOLDEN_CHUNK_SIZE=64` 全量 compare 为 `15 failed, 5 passed`。
- 失败集中在完整 C64 compose 路径，短尾/小块 case 有通过，当前怀疑点是 FP16 `HalfL1ComposeBlock()` / `Compose32From16()` / `Compose64From32()` 的 half L1 layout、同步或公式顺序。
- 继续修复时边界必须保持：
  - 不改 final RHS。
  - 不新增 `gBKHalf` / `vBetaHalf`。
  - 不新增 RHS resident/raw final。
  - 不改 QK、Stage2、Stage3。

## 通过/未通过明细

- 通过：
  - `halflocal/C16_FP16`
  - selector：`op_kernel/selected/stage1_selected.h`
  - config：`halflocal / C16_FP16 / chunkSize=16`
  - compare：`CGDR_GOLDEN_CHUNK_SIZE=16 ... test_run.sh compare`
  - 结果：`20 passed in 6.03s`
- 未通过：
  - `halflocal/C64_3232_FP16`
  - compare：`CGDR_GOLDEN_CHUNK_SIZE=64 ... test_run.sh compare`
  - 结果：`15 failed, 5 passed`
  - 当前疑点：`HalfL1ComposeBlock()` / `Compose32From16()` / `Compose64From32()` 的 C64 FP16 compose。
- 只确认构建、未计入精度通过：
  - `baseline/C16_FP16`
  - `baseline/C64_1616_FP16`
  - `halflocal/C64_1616_FP16`
  - `baseline/C64_3232_FP16`
  - `halflocal/C64_3232_FP16`，该版本构建可过，但精度失败，不能作为通过线。
- 当前未改或已恢复：
  - final RHS：保持 `AICProcess(INVWsGm_, gBKWsGm_/vBetaWsGm_, ...)`。
  - `gBK/vBeta`：仍是 FP32 GM workspace。
  - QK、Stage2、Stage3：不作为本次修改范围。

## 性能测试前置规则

- 单方案放大性能测试前必须裁剪无关 `TilingData` / `MatmulTiling` 结构体。
- 只保留当前 selector 实际使用的 tiling 字段，注释或删除其它方案、probe、RHS resident、raw final scratch 等无关结构。
- 裁剪后必须重新检查 host/kernel tiling layout、workspace size 和 kernel constructor 参数是否一致。
- 后续性能 CSV 或文档中必须记录本次测试保留的 TilingData 结构体；否则性能数据不作为最终结论。

## 最新状态：C64_3232_FP16 halflocal half L1 compose 已通过

- Active / installed：
  - `halflocal/C64_3232_FP16`
  - `chunkSize=64`
  - selector：`op_kernel/selected/stage1_selected.h`
  - config：`halflocal / C64_3232_FP16 / chunkSize=64`
- 已通过：
  - `halflocal/C64_3232_FP16`：通过。
  - compare：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.27s`
- 具体实现：
  - `FP16_L1_RESIDENT_INVERSE=true`
  - `FP16_L1_RESIDENT_COMPOSE16=true`
  - `FP16_L1_RESIDENT_COMPOSE32=true`
  - `Neumann16InverseBlock()` / `Neumann32InverseBlock()` 使用 GM FP32 -> raw MMAD -> half L1/TSCM -> L0A/L0B/L0C -> GM FP32 写回。
  - `Compose32From16()` / `Compose64From32()` 已重新启用 `HalfL1ComposeBlock()`，不再默认回退 raw compose。
  - `LoadGmFloatBlockToHalfL1ByMmad()` 读取 GM 前执行 `CleanFloatMatrixCacheLines(...)`，解决 compose 读取刚写回 GM block 的可见性风险。
- 未修改：
  - final RHS：仍为 `AICProcess(INVWsGm_, gBKWsGm_, outKCumdecayGm_, ...)` 和 `AICProcess(INVWsGm_, vBetaWsGm_, outVInnerGm_, ...)`。
  - `gBK/vBeta`：仍为 FP32 GM workspace。
  - QK、Stage2、Stage3：未作为本次修改范围。
- 本次结论：
  - FP16 问题不是数值精度问题。
  - `C64_3232_FP16` 的完整 C64 compose 在当前实现下已能通过精度。
  - 旧的“`halflocal/C64_3232_FP16` 未通过、compose 暂时回退 raw”口径已过期。
- 尚未重新验收：
  - `baseline/C64_3232_FP16` 与 `fulllocal/C64_3232_FP16` 虽有相似代码，但本轮没有切 selector 全量 compare，不计入最新通过线。
  - `C64_1616_FP16` baseline/halflocal 未在本轮重新全量验收。

## 新增计划：fulllocalnew 直接搬运

- 状态：计划已进入实现调试阶段；`fulllocalnew/C16_FP16` 已构建安装，精度尚未全绿。
- 计划文档：`docs/2026-05-06/FULLLOCALNEW_DIRECT_MIGRATION_PLAN.md`。
- 目标：
  - 先把 `attention/chunk_gated_delta_rule` 新代码的 1 个 host tiling 修改和 5 个 kernel 修改迁移到 Neumann。
  - 再在 `op_kernel/fulllocalnew/` 中实现 `C16_FP16` 与 `C64_1616_FP16` 两个新 fulllocal 方案。
- 关键决策：
  - 直接搬运新 BF16 主链路，不在旧 Stage1 上继续增量 patch。
  - FP16 Neumann 求逆只在 Stage1 inverse 局部做转换，算子输入输出和 Stage2/Stage3 保持 BF16 新链路。
  - final RHS 暂不 resident/raw 化。
- 当前风险：
  - `stageOneParaNum=4` 后 Neumann scratch 必须按 parallel chunk 扩容，否则会 workspace 覆盖。
  - 最终 inverse 需要写回 BF16 `attnWsGm_` 供新 BF16 final RHS 消费，raw/Fixpipe writeback helper 需要单独验证。

## fulllocalnew/C16_FP16 当前调试状态

- 当前 selector：
  - `op_kernel/selected/stage1_selected.h` -> `../fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocalnew.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocalnew"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C16_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=16`
- 构建安装已通过。
- 最近一次全量 compare：
  - `CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`
  - 结果：`16 passed, 4 failed`
- 失败 case：
  - `b4_t32_h16_h4_d64`
  - `b2_t96_h8_h2_d128`
  - `b1_t257_tail_d128`
  - `b2_t160_h32_h8_d64`
- 失败特征：
  - `state` 通过，`out` 失败。
  - 当前判断 inverse/state update 主路径基本健康，下一步聚焦 `qkt/qPrime/vInner` 或 Stage3 attention output 链路。
- 下一步：
  - 复跑失败 case。
  - 做 `value=0` 隔离测试。
  - 对比 new normal 与 `fulllocalnew` Stage1 的 `outQkGm_` cache flush、`QPrimeCompute()`、`GBKCompute()`、`VBetaCompute()` 和 `GCumExpCompute()`。
