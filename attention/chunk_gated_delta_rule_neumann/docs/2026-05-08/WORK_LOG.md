# 2026-05-08 工作日志

## P7/P8 当前状态复核与恢复

- 用户询问“现在什么情况”后，先复核 active selector 与 `C64InverseBy16x16Neumann()` 实际调用链。
- 当前 selector：
  - `op_kernel/selected/stage1_selected.h` include `../fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocalnew.h`。
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocalnew"`。
  - `CGDR_NEUMANN_SELECTED_VARIANT="C64_1616_FP16"`。
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`。
- 检查发现 active 入口残留一次失败的 P7 尝试：
  - `C64InverseBy16x16Neumann()` 调用了 `P7PackedDiagonalNeumannGroup()`。
  - `P7PackedDiagonalNeumannGroup()` 内部一度改为 full `HalfL1NeumannCompactBlock(groupSize)`。
- 已恢复到精度通过路径：
  - diagonal inverse 使用 `Neumann16InverseBlock()`。
  - `validLen > 16` 先做 `Compose32From16(paraIdx, 0)`。
  - `validLen > 48` 时做 lower-right 两个 16x16 diagonal inverse，并调用 `P8ComposeBoth32From16Scratch64(paraIdx)`。
  - `validLen > 32` 时调用 `P8Compose64From32Scratch64(paraIdx)`。
- P7 当前只保留为诊断 helper：
  - `P7PackedDiagonalNeumannGroup()` 恢复调用 `P7HalfL1DiagonalNeumannBlock(groupSize)`。
  - P7 不再进入 active path。

## P8 验证结果

- P8 active helper：
  - `P8ComposeBoth32From16Scratch64()` 用 64x64 FP16 L1 scratch 同时完成两个 `16->32` compose。
  - `P8Compose64From32Scratch64()` 用 64x64 FP16 L1 scratch 完成最终 `32->64` compose。
- packed scratch 清零：
  - `ZeroHalfL1()` 内部使用 `InitConstValue` 对 L1 批量清零，等价承接用户提醒的 `set2d`/批量清零方向。
  - P8 helper 在填 block 前均先清零 scratch，避免未参与 off-diagonal block 残留污染。
- 构建：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - 结果：成功。
- 安装：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 结果：成功。
- smoke：
  - `S=17/32/33/49/64/129` 全部 PASS。
- full compare：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.22s`。

## 当前结论

- 当前可用实现不是完整 P7+P8。
- P8 scratch64 compose 已实现、进入 active path，并通过当前精度 gate。
- P7 packed diagonal inverse 仍未通过，不能作为 active path。
- 当前 active 交付口径为 `P3 diagonal inverse + P8 scratch64 compose`。
- final RHS 仍保持 BF16 GM matmul，不修改 Stage2/Stage3、`gBK/vBeta` 或 QK 链路。
- 注意：`S>=33` 的 Neumann out 误差相对 normal 明显增大，但仍在当前 compare 判据内 PASS；如果后续需要更严格精度，需要单独分析 compose 数值误差。

## P3+P8 scratch64 简单性能 smoke

- 按用户要求简单测性能，使用 `msprof --application=... --task-time=on`，每个 op 仍走 warmup、MatMul flush、ReduceMax flush、单次 measurement。
- 输出目录：`tests/pytest/prof_out/fulllocalnew_c64_1616_fp16_p3_p8_scratch64_smoke_20260508/`。
- 本轮只测两个 shape，且未按单方案规则裁剪无关 `TilingData/MatmulTiling` 字段，因此只作为 smoke 观察，不作为最终性能结论。

| case | normal task(us) | neumann task(us) | neumann/normal | normal AIC(us) | neumann AIC(us) |
|---|---:|---:|---:|---:|---:|
| `B8,S16,D64,H16,Hkv4` | 626.56 | 822.86 | 1.313x | 621.924 | 817.845 |
| `B8,S64,D64,H16,Hkv4` | 656.06 | 2294.08 | 3.497x | 652.954 | 2287.061 |

- 观察：
  - P8 scratch64 当前没有性能收益，`S=64` 明显劣化。
  - `S=64` 的 neumann AIC scalar time 为 `1225.263us`，相对 normal `107.335us` 放大 `11.415x`，是当前 smoke 中最突出的异常项。
  - 当前 P8 helper 仍存在 full 64x64 scratch、整块写回 `L_powHalfWsGm_`、再读回子块等中间步骤；它是正确性优先的 scratch64 实现，不是最终 packed fast inverse 性能形态。

## P7/P8 下三角适配计划

- 用户指出参考代码中的 `LX/RX` MBH 公式是上三角求逆方向，当前算子是下三角 compose，不能原样套用；该判断正确。
- 已创建方案文档：`docs/2026-05-08/P7_P8_LOWER_TRIANGULAR_REFERENCE_ADAPTATION_PLAN.md`。
- 关键结论：
  - `CopyDiagonalFractalsL1ToL0()` 的 diagonal fractal 打包思想可参考。
  - `CopyOddOrEvenBlocksL1ToL0()` 的 odd/even 打包思想可参考。
  - 但 compose 数学顺序必须保持当前下三角公式：`out = bottomInv @ lower @ topInv`。
  - 当前 scratch64 P8 数学方向是对的，主要问题是 full scratch、GM 中转、清零和同步开销太重。
- 后续计划分为：
  - P8A：先裁掉 scratch64 的整块 GM 中转，L0C 直接写目标 lower-left 子块。
  - P7A：单独验证 diagonal fractal load/writeback，不接 active。
  - P7B：packed diagonal inverse 替换 P3 diagonal。
  - P8B：实现下三角 odd/even packed compose。
  - P8C：整合 full 64 L1 resident。

## P8A direct compose 实施

- 用户纠正：`32->64` 不应改为四个 16x16 目标子块写回，也不应把“回退保守写法”作为方向；本轮按正向优化实现 P8A direct compose。
- active selector 保持：
  - `op_kernel/selected/stage1_selected.h` include `../fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocalnew.h`。
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocalnew"`。
  - `CGDR_NEUMANN_SELECTED_VARIANT="C64_1616_FP16"`。
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`。
- 实现边界：
  - diagonal inverse 仍使用 P3 `Neumann16InverseBlock()`。
  - `16->32` 改为 `P8Compose32From16SingleDirect(paraIdx, 0)` 和 `P8Compose32From16SingleDirect(paraIdx, C64_BLOCK32)`。
  - `32->64` 改为 `P8Compose64From32Direct32(paraIdx)`。
  - `P8Compose64From32Direct32()` 使用紧凑 32x32 L1/L0C 路径，最终通过 `CopyL0CMatrixToGmHalf(..., C64_BLOCK32, C64_BLOCK32, chunkSize_, ...)` 直接写目标 lower-left 32x32 子块；没有拆成四个 16x16 目标写回。
  - P7 packed diagonal inverse 没有接入 active。
  - final RHS 继续保持 BF16 GM `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)`。
  - Stage2/Stage3、QK、`gBK/vBeta`、tiling/workspace 未修改。
- 构建与安装：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 结果：均成功。
- 精度验证：
  - smoke `S=17/33/49/64` PASS。
  - full compare：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.20s`。
- 性能 smoke：
  - 输出目录：`tests/pytest/prof_out/fulllocalnew_c64_1616_fp16_p8a_direct_smoke_20260508/`。
  - 本轮未裁剪无关 `TilingData/MatmulTiling` 字段，只作为 smoke 观察，不作为最终性能结论。

| case | normal task(us) | neumann task(us) | neumann/normal | normal AIC(us) | neumann AIC(us) |
|---|---:|---:|---:|---:|---:|
| `B8,S16,D64,H16,Hkv4` | 654.82 | 821.14 | 1.254x | 650.432 | 816.541 |
| `B8,S64,D64,H16,Hkv4` | 662.26 | 925.88 | 1.398x | 659.294 | 921.538 |

- 技术结论：
  - P8A 正向 direct path 已通过精度 gate。
  - 相比 P3+P8 scratch64，`S=64` smoke 从 `3.497x` 降到 `1.398x`，说明裁掉 full64 scratch/GM 中转有效。
  - 当前仍不是最终 packed fast inverse；后续继续按计划做 P7A 诊断与 P8B 下三角 odd/even packed compose。

## P8A baseline 保存与后续正向尝试

- 已将 P8A direct 通过版保存到 baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p8a_baseline.h`
  - baseline 内容为 P3 diagonal inverse + P8A direct compose。
  - active `fulllocalnew` 文件恢复后与 `_p8a_baseline` 只有文件名和 include guard 差异。
- 当前 active 仍为：
  - selector：`fulllocalnew/C64_1616_FP16/chunkSize=64`。
  - diagonal inverse：P3 `Neumann16InverseBlock()`。
  - `16->32` compose：`P8Compose32From16SingleDirect()`。
  - `32->64` compose：`P8Compose64From32Direct32()`，直接写目标 lower-left 32x32。
  - final RHS：仍为 BF16 GM `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)`。
- 恢复 P8A 后重新验证：
  - build/install 成功。
  - `S=64` smoke PASS。
  - full compare `20 passed in 6.18s`。

## P7A/P8 后续尝试结果

- 尝试将 top32 diagonal inverse 接入 P7 packed path：
  - P7 active 后 `S=17` 首轮 240s timeout。
  - 将新增 L0C direct diagonal writeback 换回已验证 L1 子块写回后，不再挂起，但 `S=17` 精度失败。
  - 将 P7 组内 MMAD 改为 full 32x32 packed 对照后，`S=17` 仍失败且误差更大。
  - 将 `inv@I + inv@power` 改成 `tmp=inv@power` 后 L1 add，`S=17` 勉强 PASS，但 `S=33` 失败，说明 top32 diagonal 残余误差会在 `32->64` compose 后放大。
  - scalar packed load 对照更差，说明不能把当前 P7 实现作为 active 优化。
- 尝试 P8 完整块合并两个 `16->32` compose：
  - full chunk 路径临时启用 `P8ComposeBoth32From16()`，`32->64` 保持 `P8Compose64From32Direct32()`。
  - `S=64` 240s timeout，说明该合并 helper 当前存在执行风险，不能保留。
- 本轮失败尝试均已从 active 撤回，未保存为 baseline。
- 下一步应先做独立 P7A 诊断，而不是直接把 P7 接入 active：
  - 验证 packed `block@I` / diagonal roundtrip。
  - 分开验证 L0A 与 L0B diagonal fractal load。
  - 分开验证 L0C submatrix Fixpipe 写回和 L1/GM 可见性。

## P7 packed64 base0 override 诊断

- 在当前 active 入口上做了最小诊断：
  - `P7PackedDiagonalNeumann64(paraIdx)` 后只用 `Neumann16InverseBlock(paraIdx, 0)` 覆盖 `(0,0)`。
  - 其余三个 diagonal block 保持 packed64 结果不动。
- build/install：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 均成功。
- smoke：
  - `S=64` 仍 FAIL。
  - 结果与上一轮 packed64 baseline 基本一致：
    - `CPU vs ChunkGDRNeumann out: FAIL max_abs=1.059731e-01`
    - `CPU vs ChunkGDRNeumann state: PASS max_abs=5.440532e-03`
- 结论：
  - `(0,0)` 这块不是当前 out 误差主因。
  - 下一步应继续只替换其它单个 diagonal block，找出真正被 packed64 影响的块。

## P7 packed64 base16/base32 override 诊断

- 将单块覆盖从 `(0,0)` 依次切到：
  - `(16,16)`：`Neumann16InverseBlock(paraIdx, INVERSE_SHAPE)`。
  - `(32,32)`：`Neumann16InverseBlock(paraIdx, C64_BLOCK32)`。
- 每次都重新 build/install，并跑同一个 `S=64` smoke。
- 结果：
  - 两轮均仍然 FAIL。
  - 两轮 `CPU vs ChunkGDRNeumann out max_abs` 都保持 `1.059731e-01`。
  - 两轮 `CPU vs ChunkGDRNeumann state max_abs` 都保持 `5.440532e-03`，state 仍 PASS。
- 结论：
  - 单独恢复 `(16,16)` 或 `(32,32)` 都不能解除主要 out 失败。
  - 当前问题不是某一个独立 diagonal block 的简单错误，更像 packed64 写回后被 compose/attn 消费时形成了跨段空输出，或需要成对恢复 diagonal block 才能解除下三角 compose 链路。

## P7 packed64 base48 override 诊断

- 将单块覆盖切到 `(48,48)`：
  - `P7PackedDiagonalNeumann64(paraIdx)` 后调用 `Neumann16InverseBlock(paraIdx, C64_BLOCK32 + INVERSE_SHAPE)`。
- build/install 成功。
- `S=64` smoke 仍 FAIL：
  - `CPU vs ChunkGDRNeumann out: FAIL max_abs=1.059731e-01`
  - `CPU vs ChunkGDRNeumann state: FAIL max_abs=1.053194e-01`
- 结论：
  - 单独覆盖 `(48,48)` 不但不能修复 out，还会使 state 从 PASS 退化为 FAIL。
  - 下一步做关键对照：`P7PackedDiagonalNeumann64()` 后覆盖四个 P3 diagonal block。若仍 FAIL，说明 packed64 执行本身有副作用；若 PASS，说明 packed64 结果/写回是主因。

## P7 packed64 roundtrip/load-only 诊断

- 当前 active 入口用于诊断，不是目标通过路径：
  - `P7PackedDiagonalRoundTrip64Probe(paraIdx)`。
  - 随后仍调用四个 `Neumann16InverseBlock()` 和已验证的 direct P8：
    - `P8Compose32From16SingleDirect(0)`。
    - `P8Compose32From16SingleDirect(C64_BLOCK32)`。
    - `P8Compose64From32Direct32()`。
- 先验证完整 roundtrip：
  - packed64 从四个 diagonal GM block load 到 packed L1。
  - 再用 scalar probe 从 packed L1 写回四个 diagonal GM block。
  - build/install 成功。
  - `S=64` smoke FAIL：
    - `CPU vs ChunkGDRNeumann out: FAIL max_abs=3.430384e-01`
    - `CPU vs ChunkGDRNeumann state: FAIL max_abs=1.301773e-01`
- 再将 `P7PackedDiagonalRoundTrip64Probe()` 收缩为 load-only：
  - 保留 `ZeroHalfL1()` 和四个 `LoadGmHalfDiagonalBlockToPackedL1()`。
  - 在写回前 `return`，不改写 `INVInitHalfWsGm_`。
  - build/install 成功。
  - `S=64` smoke PASS：
    - `CPU vs ChunkGDRNeumann out: PASS max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state: PASS max_abs=7.139370e-04`
    - `ChunkGDR vs ChunkGDRNeumann out: PASS max_abs=4.882812e-04`
    - `ChunkGDR vs ChunkGDRNeumann state: PASS max_abs=4.882812e-04`
- 结论：
  - packed64 `ZeroHalfL1()` 和 GM -> packed L1 diagonal load 本身不污染后续 P3/P8。
  - 完整 roundtrip 失败点落在 packed L1 -> diagonal GM 写回链路，当前不能把 roundtrip 或 P7 packed64 当作通过路径。
  - 下一步只定位 `StorePackedL1DiagonalBlockToGmHalfScalarProbe()` / packed L1 读出布局 / diagonal GM 写回可见性；不修改 P8，也不进入 P8 odd/even packed compose。
