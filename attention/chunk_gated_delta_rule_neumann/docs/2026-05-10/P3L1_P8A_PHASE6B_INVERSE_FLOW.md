# P3L1/P8A Phase 6b 当前求逆流程

本文记录 2026-05-10 清理 probe / 旧 tilingdata 后，active `p3l1_p8a` 路径中 Stage1 inverse 的实际搬运和计算流程。

## Active 入口

- selector:
  - `op_kernel/selected/stage1_selected.h`
  - `op_kernel/selected/stage1_selected_config.h`
- active Stage1:
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- clean baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_phase6b_baseline.h`

## 清理后边界

- active `Stage1` 构造只接收 `StageOneMT &mm`。
- top-level kernel 只保留 `stage1MT_ / stage2MT_ / stage3MT_`，三者都使用 `matmulTilingFp32` 初始化。
- `ChunkGatedDeltaRuleNeumannTilingData` 只保留 `matmulTilingFp32` 一个 cube tiling 字段。
- host `DoMatmulTiling()` 只生成 `matmulTilingFp32`。
- active Stage1 不再保留旧 probe / fallback 名称：
  - scalar packed probe。
  - old `StageOneFp16*` matmul wrappers。
  - old GM direct compose fallback helper。
  - old `INVWsGm_` FP32 final inverse workspace。
- workspace 只按 active 绑定预留：
  - FP16 `L_powHalfWsGm_`。
  - FP16 `INVInitHalfWsGm_`，承载 initial INV 和最终 half inverse scratch。
  - FP32 `tmpWsGm_`，用于 identity materialize。
  - FP16 `gBKHalfWsGm_` / `vBetaHalfWsGm_`，用于 fused RHS。

## 总体同步顺序

Stage1 内 AIC/AIV 分工如下：

1. AIV 预处理 Q/K，写 `queryConGm_` / `keyConGm_`。
2. AIC 等 flag `0x9` 后计算 `kk = key @ key^T` 到 `kkWsGm_`，再设置 flag `0x8`。
3. AIC 同时计算 `qk = query @ key^T` 到 `outQkGm_`。
4. AIV 等 flag `0x8`，读取 `kkWsGm_`，计算 `L_powHalfWsGm_` 和 `INVInitHalfWsGm_`，设置 flag `0x7`。
5. AIC 等 flag `0x7`，对每个 parallel chunk 调用 `C64InverseBy16x16Neumann()`。
6. AIC 写完 BF16 inverse GM 和 resident L1 后设置 flag `0xd`。
7. AIV 等 flag `0xd`，继续生成 half RHS operand：`gBKHalfWsGm_` / `vBetaHalfWsGm_`。
8. AIC 等 flag `0x6` / `0x5` 后用 resident inverse 做 fused RHS，输出 BF16 GM。

## AIV: 生成 FP16 L 和初始 inverse

`KKBetaCompute()`：

1. 从 `kkWsGm_` 按当前 AIV half chunk 搬入 BF16，cast 到 FP32 UB。
2. 将 `betaUbFloat_` broadcast 到 `(halfChunkSize_, chunkSize_)`。
3. 做 `attnUbFloat_ = kk * beta`。

`InverseCompute()`：

1. 若有 `g`，将 `attnUbFloat_` 乘 `gammaUbFloat_`；否则乘 lower-triangular mask。
2. 将结果 cast 为 FP16，写到 `L_powHalfWsGm_`。
3. 做 `inverseRes_ = -attnUbFloat_`。
4. 对当前 half chunk 的对角线写 `1.0`，形成初始 `I - L` 近似。
5. 将 `inverseRes_` cast 为 FP16，写到 `INVInitHalfWsGm_`。
6. 每次 GM 写回后执行 MTE3/MTE2 同步和 cache flush，保证 AIC 后续读取可见。

因此 AIV 给 AIC 提供的是两份 FP16 GM scratch：

- `L_powHalfWsGm_`: 后续 Neumann power 输入。
- `INVInitHalfWsGm_`: diagonal block 初始 inverse，以及 P8 lower-left 初始块。

## AIC: 本地 buffer 布局

`C64InverseBy16x16Neumann()` 第一次进入时会 `InitRawMmadLocalBuffers()`：

- L1/TSCM half tiles:
  - `aicHalfL1Scratch_`
  - `aicHalfL1TileInv_`
  - `aicHalfL1TilePower_`
  - `aicHalfL1Identity_`
  - `aicHalfL1Zero_`
  - `aicHalfL1P3Result_`
  - `aicHalfL1ResidentInv_`
- L0:
  - FP32 `aicL0C_`
  - FP16 ping-pong `aicHalfL0A_ / aicHalfL0B_`
  - FP16 ping-pong alt `aicHalfL0AAlt_ / aicHalfL0BAlt_`

所有 packed inverse 的主承载是 `aicHalfL1P3Result_`，形状按 `64x64` packed NZ 看待。

## P7: diagonal 16x16 求逆

`P7PackedDiagonalNeumannGroupToP3Result(paraIdx, groupBase, 32)` 是 Phase 6b 的 active diagonal 入口。

对于一个 32 group：

1. 清零 `aicHalfL1TileInv_` / `aicHalfL1TilePower_` / `aicHalfL1Scratch_`。
2. 从 GM 读取第一个 diagonal `16x16`：
   - `L_powHalfWsGm_ -> aicHalfL1TilePower_`
   - `INVInitHalfWsGm_ -> aicHalfL1TileInv_`
   - 目的位置是 group packed L1 的 `(0,0)`。
3. 从 GM 读取第二个 diagonal `16x16`：
   - 目的位置是 group packed L1 的 `(16,16)`。
4. 对第一个 `(0,0)` 执行 `P7HalfL1DiagonalNeumannBlock(16)`。
5. 将求解结果复制到 `aicHalfL1P3Result_` 的 `(groupBase, groupBase)`。
6. 将第二个 `(16,16)` 从 group packed 位置复制回 compact `(0,0)`：
   - `aicHalfL1TileInv_ -> aicHalfL1TileInv_`
   - `aicHalfL1TilePower_ -> aicHalfL1TilePower_`
7. 再执行一次 `P7HalfL1DiagonalNeumannBlock(16)`。
8. 将第二个结果复制到 `aicHalfL1P3Result_` 的 `(groupBase + 16, groupBase + 16)`。

`P7HalfL1DiagonalNeumannBlock(16)` 的计算链：

1. `PrepareHalfL1IdentityByMmad(16)` materialize identity。
2. `scratch = power @ power`。
3. `inv = inv + inv @ scratch`。
4. `power = scratch @ scratch`。
5. `inv = inv + inv @ power`。
6. `scratch = power @ power`。
7. `inv = inv + inv @ scratch`。

其中 power 和 inv-add 都走 diagonal-only raw MMAD：

1. 清零 L0A/L0B 的 compact 16x16。
2. 只把 L1 diagonal fractals load 到 L0A/L0B。
3. `Mmad(..., HF32 disabled)` 输出 FP32 L0C。
4. `Fixpipe F322F16` 写回 half L1。

注意：Phase 6b 不是 strict `32x32/64x64` one-shot diagonal solver；它是 group load + two proven `16x16` solves。

## P8: lower-triangular compose

当前 compose 公式保持 lower-triangular 方向：

```text
tmp = lower @ topInv
out = bottomInv @ tmp
```

### 16+16 -> 32

`P8Compose32From16SingleDirect(paraIdx, base)`：

1. `lower` 已由 `LoadP8LowerLeftToPackedL1()` 从 `INVInitHalfWsGm_` 搬到 `aicHalfL1P3Result_` 的 `(base + 16, base)`。
2. 第一次 raw MMAD：
   - A: `aicHalfL1P3Result_[bottom, top]`
   - B: `aicHalfL1P3Result_[top, top]`
   - 输出 L0C。
3. `Fixpipe F322F16` 将 L0C 写到 `aicHalfL1Scratch_`，得到 `tmp`。
4. 第二次 raw MMAD：
   - A: `aicHalfL1P3Result_[bottom, bottom]`
   - B: `aicHalfL1Scratch_`
   - 输出 L0C。
5. `Fixpipe F322F16` 写到 `aicHalfL1TilePower_`，得到 lower-left composed block。
6. 用 identity MMAD 把 `aicHalfL1TilePower_` 复制回 `aicHalfL1P3Result_[bottom, top]`。
7. 同时写回 `INVInitHalfWsGm_[bottom, top]`，供保留的 half scratch 语义使用。

### 32+32 -> 64

`P8Compose64From32Direct32(paraIdx)` 同样执行：

1. `tmp = lower32 @ topInv32`，写 `aicHalfL1Scratch_`。
2. `out = bottomInv32 @ tmp`，写 `aicHalfL1TilePower_`。
3. 复制到 `aicHalfL1P3Result_[32,0]`。
4. 写回 `INVInitHalfWsGm_[32,0]`。

## C64InverseBy16x16Neumann() 主流程

对每个 `paraIdx`：

1. 初始化 / 恢复 raw MMAD 状态。
2. 清零 `aicHalfL1P3Result_`。
3. top 32 group:
   - `P7PackedDiagonalNeumannGroupToP3Result(paraIdx, 0, 32)`。
4. 若 `validLen > 16`:
   - load `(16,0)` lower-left。
   - compose top 32：`P8Compose32From16SingleDirect(paraIdx, 0)`。
5. bottom 32 group:
   - 若 `validLen > 48`，执行 `P7PackedDiagonalNeumannGroupToP3Result(paraIdx, 32, 32)`，load `(48,32)` 并 compose bottom 32。
   - 否则若 `validLen > 32`，只求解单个 `(32,32)` 16x16 diagonal block。
6. 若 `validLen > 32`:
   - load `(32,0)` 32x32 lower-left。
   - compose 64：`P8Compose64From32Direct32(paraIdx)`。
7. `StoreP3DiagonalResultsToGm(paraIdx)`:
   - 将 diagonal 16x16 blocks 从 `aicHalfL1P3Result_` 写回 `INVInitHalfWsGm_`。
8. `StorePackedL1InverseToBf16Attn(paraIdx)`:
   - lower-triangular 16x16 blocks 从 packed L1 写 BF16 `attnWsGm_`。
   - upper blocks 通过 zero tile 写 0。
9. `StorePackedL1InverseToResidentL1(paraIdx)`:
   - lower-triangular 16x16 blocks 复制到 per-para resident L1 slot。

## Fused RHS

AIV 在 inverse BF16 writeback 完成后生成 half RHS：

- `GBKCompute()` 写：
  - BF16 `gBKWsGm_`
  - FP16 `gBKHalfWsGm_`
- `VBetaCompute()` 写：
  - BF16 `vBetaWsGm_`
  - FP16 `vBetaHalfWsGm_`

AIC 端 `FusedRhsAICProcess()`：

1. 取当前 `paraIdx` 的 resident inverse：`aicHalfL1ResidentInv_ + paraIdx * 64 * 64`。
2. 按 `colBase` 分块读取 half RHS GM 到 `aicHalfL1TilePower_` packed L1。
3. 执行 raw MMAD rectangular：
   - A: resident inverse。
   - B: RHS half tile。
   - C: FP32 L0C。
4. `Fixpipe F322BF16` 写到 output GM：
   - `outKCumdecayGm_`
   - `outVInnerGm_`
5. flush BF16 output cache lines。

## 当前验证

本次 probe / tilingdata cleanup 后重新验证：

- build PASS：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install PASS：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- smoke PASS：
  - `S=64`：out max_abs `3.169131e-02`，state max_abs `5.276948e-04`
  - `S=256`：out max_abs `6.722955e-02`，state max_abs `9.178072e-04`
  - `S=1024`：out max_abs `6.722955e-02`，state max_abs `1.006737e-03`
- current compare set PASS：
  - `16 passed in 6.14s`

## 残留风险

- 历史 baseline / 非 active 实验文件里仍能搜到旧 probe 或旧 matmul 字段名；这些不是 selector 当前入口。
- strict `32x32/64x64` group diagonal-only raw-MMAD solver 仍不是通过路径，不得描述为完成。
- 当前 `out` max_abs 仍高于早期 P8 compose 阶段，虽在当前阈值内 PASS，后续优化仍需继续观察。
