# 2026-05-08 进度与问题

## 已完成

- 已确认 active selector 为 `fulllocalnew/C64_1616_FP16/chunkSize=64`。
- 已修正 active 入口残留的失败 P7 尝试，恢复为 `P3 diagonal inverse + P8 scratch64 compose`。
- 已完成 P8 scratch64 compose active 验证：
  - `P8ComposeBoth32From16Scratch64()` 负责两个 `16->32` compose。
  - `P8Compose64From32Scratch64()` 负责最终 `32->64` compose。
  - packed scratch 通过 `ZeroHalfL1()`/`InitConstValue` 批量清零。
- 已重新 build/install 并验证精度：
  - build 成功。
  - install 成功。
  - smoke `S=17/32/33/49/64/129` 全部 PASS。
  - full compare `20 passed in 6.22s`。
- 已完成 P3+P8 scratch64 两点性能 smoke：
  - 输出目录：`tests/pytest/prof_out/fulllocalnew_c64_1616_fp16_p3_p8_scratch64_smoke_20260508/`。
  - `B8,S16,D64,H16,Hkv4`：normal `626.56us`，neumann `822.86us`，neumann/normal `1.313x`。
  - `B8,S64,D64,H16,Hkv4`：normal `656.06us`，neumann `2294.08us`，neumann/normal `3.497x`。
- 已完成 P8A direct compose：
  - `16->32` active helper 为 `P8Compose32From16SingleDirect()`。
  - `32->64` active helper 为 `P8Compose64From32Direct32()`。
  - `32->64` 使用紧凑 32x32 L1/L0C 计算后直接写目标 lower-left 32x32，不拆成四个 16x16 子块写回。
  - build/install 成功。
  - smoke `S=17/33/49/64` PASS。
  - full compare `20 passed in 6.20s`。
- 已完成 P8A direct 两点性能 smoke：
  - 输出目录：`tests/pytest/prof_out/fulllocalnew_c64_1616_fp16_p8a_direct_smoke_20260508/`。
  - `B8,S16,D64,H16,Hkv4`：normal `654.82us`，neumann `821.14us`，neumann/normal `1.254x`。
  - `B8,S64,D64,H16,Hkv4`：normal `662.26us`，neumann `925.88us`，neumann/normal `1.398x`。
- 已保存 P8A baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p8a_baseline.h`。
  - active 已恢复到 P8A baseline 内容，仅文件名/include guard 不同。
  - 恢复后 build/install 成功，`S=64` smoke PASS，full compare `20 passed in 6.18s`。

## 进行中

- 当前只关注 `C64_1616` Stage1 inverse/compose 路径。
- 当前 active path：
  - diagonal inverse 仍使用 P3 的 `Neumann16InverseBlock()`。
  - compose 已使用 P8A direct helper：两个 `16->32` direct compose 和一个 `32->64` direct compose。
  - final RHS 保持 BF16 GM `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)`。
- P7 helper 保留为诊断代码，但不进入 active path。
- 已形成新的下三角适配计划：`docs/2026-05-08/P7_P8_LOWER_TRIANGULAR_REFERENCE_ADAPTATION_PLAN.md`。
- 后续不直接搬参考代码的上三角 MBH 公式，而是保留下三角 compose 顺序 `bottomInv @ lower @ topInv`，只参考 diagonal/odd-even block 打包方式。

## 阻塞项

- P7 packed diagonal inverse 未通过：
  - 原始 packed 32 diagonal active 尝试在 `S=17` 精度失败。
  - diagonal-only 写回诊断在 `S=17` 超时。
  - full 32 block-diagonal MMAD 保守尝试在 `S=17` 精度失败。
  - 本轮 top32 P7 active 直接接入后 `S=17` 240s timeout；去掉 direct diagonal writeback 后 `S=17` 精度失败。
  - 避免 L0C accumulate 后 `S=17` 能 PASS，但 `S=33` 失败，说明误差会在 `32->64` compose 后放大。
  - scalar packed load 对照更差，P7 当前不能作为 active path。
- 新增诊断结果：
  - `P7PackedDiagonalNeumann64()` 后只恢复 `(0,0)` 的 `Neumann16InverseBlock()`，`S=64` 仍然 FAIL，且 `max_abs` 基本不变。
  - 后续只恢复 `(16,16)` 或 `(32,32)`，`S=64` 也仍然 FAIL，`out max_abs` 仍为 `1.059731e-01`。
  - 只恢复 `(48,48)` 时，`out max_abs` 仍为 `1.059731e-01`，并且 state 退化到 `max_abs=1.053194e-01` FAIL。
  - 因此当前不是单个 diagonal block 的简单错误，更可能是 packed64 写回/可见性链路，或下三角 compose 需要成对正确的 diagonal block 才能解除空输出。
- 2026-05-08 15:21 新增 roundtrip/load-only 结论：
  - 完整 `P7PackedDiagonalRoundTrip64Probe()` 后接四个 P3 block，`S=64` FAIL：
    - `out max_abs=3.430384e-01`
    - `state max_abs=1.301773e-01`
  - 将该 probe 收缩为 load-only 后接四个 P3 block，`S=64` PASS：
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
  - 因此 packed64 load/zero 不污染后续路径，失败点已收敛到 packed L1 -> diagonal GM 写回链路。
- 当前判断 P7 问题更可能集中在 L0/L1 packed diagonal 写回、Fixpipe 子块散回或布局语义；`CompactMatrixNzOffset(16,16,32)=768` 不应简单改成 512。
- P8 完整块合并两个 `16->32` compose 的尝试未通过：
  - 临时启用 `P8ComposeBoth32From16()` + `P8Compose64From32Direct32()` 后，`S=64` 240s timeout。
  - 该 helper 当前不能进入 active，也不能作为后续 baseline。
- P8 虽然通过当前阈值，但 `S>=33` 的 out 误差较 normal 明显增大，后续若收紧精度阈值需要单独处理。
- P8 scratch64 smoke 性能不理想：
  - `S=16` neumann/normal `1.313x`。
  - `S=64` neumann/normal `3.497x`。
  - `S=64` AIC scalar time 放大到 `11.415x`，说明当前 scratch64 正确性实现仍有大量 scalar/同步/中间写回成本。
  - 若要作为正式性能结论，需要先按单方案规则裁剪无关 `TilingData/MatmulTiling` 字段；当前仅作为快速 smoke。
- P8A direct 虽然显著改善 `S=64` smoke，但仍慢于 normal：
  - `S=16` neumann/normal `1.254x`。
  - `S=64` neumann/normal `1.398x`。
  - 当前 direct compose 仍依赖 FP16 GM scratch 作为阶段边界，尚未形成 P7/P8 packed resident 主线。

## 下一步

- 下一步做 P7A：
  - 继续单独验证 diagonal packed 写回，不改 P8。
  - 优先定位 `StorePackedL1DiagonalBlockToGmHalfScalarProbe()`、packed L1 读出布局和 diagonal GM 写回可见性。
  - P7A 不作为目标完成，直到 `S=64` target packed64 路径稳定。
- P7A 通过后再推进 P7B/P8B：
  - P7B 替换 P3 diagonal inverse。
  - P8B 实现下三角 odd/even packed compose。
