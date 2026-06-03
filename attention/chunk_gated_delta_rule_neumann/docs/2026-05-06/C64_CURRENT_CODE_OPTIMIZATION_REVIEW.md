# 2026-05-06 C64 当前代码优化复核

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 当前代码状态

复核文件：

- `op_kernel/fulllocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_fulllocal.h`
- `op_kernel/halflocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_halflocal.h`
- `op_kernel/halflocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_halflocal.h`
- `op_kernel/selected/stage1_selected.h`
- `op_kernel/selected/stage1_selected_config.h`

本轮最新复核时 active selector 为 `halflocal/C16_FP16`，`CGDR_NEUMANN_SELECTED_CHUNK_SIZE=16`，因此 C64 文件不是当前默认编译入口；以下只分析 C64 文件本身。

注意：今天 selector 曾多次被切换，后续任何构建/性能验证前必须重新打开 `op_kernel/selected/stage1_selected*.h` 确认真实入口。

当前 `C64_3232_FP16` 文件与此前计划中的假设已有差异：

- `Compose32From16()` / `Compose64From32()` 当前已经不再直接展开旧的 `RawMmadBlock(... tmpWsGm_ ...)` 两步路径。
- resident 分支现在调用 `HalfL1ComposeBlock()`。
- 但 `HalfL1ComposeBlock()` 仍然是局部 resident，不是 C64 full matrix resident。

当前 `HalfL1ComposeBlock()` 流程：

1. `PrepareHalfL1IdentityByMmad(blockSize)`。
2. `LoadGmFloatBlockToHalfL1ByMmad(lowerGm, aicHalfL1TileInv_, blockSize)`。
3. `LoadGmFloatBlockToHalfL1ByMmad(topDiagGm, aicHalfL1TilePower_, blockSize)`。
4. `HalfL1PowerBlock(lower, topDiag, scratch)`。
5. `LoadGmFloatBlockToHalfL1ByMmad(bottomDiagGm, aicHalfL1TileInv_, blockSize)`。
6. `HalfL1PowerBlock(bottomDiag, scratch, tilePower)`。
7. `CopyHalfL1MatrixToGmFloatByMmad(tilePower, dstGm, blockSize, chunkSize_)`。

也就是说 compose 的两次 MMAD 在 half L1/TSCM 内完成，但每个输入 block 仍从 GM 转入，输出仍写回 GM。

## 关键观察

### 1. C64 full L1 buffer 已有，但主路径基本没用起来

代码分配了：

- `aicHalfL1Inv_`：64x64 full matrix buffer。
- `aicHalfL1Power_`：64x64 full matrix buffer。
- `aicHalfL1Scratch_` / `aicHalfL1TileInv_` / `aicHalfL1TilePower_` / `aicHalfL1Identity_`：32x32 compact tile buffers。

但当前 C64 主路径：

- `Neumann16InverseBlock()` 使用 compact tile，算完写回 `INVWsGm_`。
- `Compose32From16()` / `Compose64From32()` 通过 `HalfL1ComposeBlock()` 从 GM 重新加载所需 block。
- `aicHalfL1Inv_` / `aicHalfL1Power_` 这两个 full 64x64 buffer 没有承担“C64 矩阵片上常驻状态”的角色。

因此当前 C64 fulllocal 实际是“每个 block 局部 resident”，不是“C64 matrix resident”。

### 2. L0 ping-pong buffer 已有，但没有异步重叠

代码已有：

- `aicHalfL0A_` / `aicHalfL0B_`
- `aicHalfL0AAlt_` / `aicHalfL0BAlt_`

它们在 `HalfL1InvAddBlock()` 中用于：

- buffer 0：`INV @ I`
- buffer 1：`INV @ power`

这能避免两次输入覆盖，但 `RawMmadHalfL1AccumulateTile()` 内部仍然是同步流程：

1. `SyncMToMte1()`
2. `CopyHalfL1MatrixToL0A/B()`
3. `SyncMte1ToM()`
4. `Mmad()`
5. `PipeBarrier<PIPE_M>()`

所以当前还不是 fast inverse 那种真正的 double-buffer 异步重叠。要获得 ping-pong 收益，需要拆开 load 与 mmad，用事件让 next tile 的 MTE1 搬运和 current tile 的 MMAD 重叠。

### 3. identity 准备重复且成本偏高

`PrepareHalfL1IdentityByMmad(blockSize)` 会：

1. `WriteIdentityBlockGm(tmpWsGm_, blockSize)`，AIC scalar 写 GM。
2. `LoadGmFloatBlockToHalfL1ByMmad(tmpWsGm_, aicHalfL1Identity_, blockSize)`，再通过 `GM identity @ GM identity -> L0C -> half L1` 生成 half identity。

当前该函数在多个地方重复调用：

- 每个 `Neumann16InverseBlock()`。
- `Neumann32InverseBlock()`。
- 每次 `HalfL1ComposeBlock()`。

对 full C64 chunk，可能会重复多次准备 16/32 identity。这会增加 AIC scalar、MTE2、MTE1、Fixpipe 和同步开销。

### 4. GM -> half L1 转换仍是核心绕路

`LoadGmFloatBlockToHalfL1ByMmad()` 当前用：

```cpp
RawMmadAccumulateTile(gm, tmpWsGm_, aicL0C_, ...)
CopyL0CMatrixToHalfL1(...)
```

即 `GM_float_block @ GM_identity -> L0C_float -> Fixpipe<half, float, CFG_NZ> -> half L1`。

这和 C16 RHS resident 变慢的根因一致：dtype cast + L1 NZ materialize 通过 MMAD/Fixpipe 绕路完成，成本不低。

## 可能收益点排序

### P0：缓存 identity，避免每个 block 重复准备

收益判断：中高，风险低。

当前最容易先做的是把 identity 准备从“每个 Neumann/compose block 调一次”改成“每个 C64 chunk 或每个 AIC task 初始化一次”。

可行方向：

- 使用已有 `InitHalfL1Identity()` 直接在 half L1/TSCM 中构造 32x32 identity。
- GM identity `tmpWsGm_` 也只写一次；16x16 可以复用 32x32 identity 的左上 16x16。
- `CopyHalfL1MatrixToGmFloatByMmad()` 直接用缓存的 `aicHalfL1Identity_`。
- `LoadGmFloatBlockToHalfL1ByMmad()` 如果仍需要 GM identity，则至少避免反复 `WriteIdentityBlockGm()`。

注意：

- 32x32 compact identity 的左上 16x16 offset 与 16x16 compact identity 在 `col < 16` 范围内一致，理论上可复用，但必须 smoke 验证。
- 这不能解决全部 GM 往返，但能减少当前 AIC scalar 和同步开销。

### P1：C64 full-matrix resident staging

收益判断：高，风险中高。

目标是让 `aicHalfL1Inv_` / `aicHalfL1Power_` 真的保存 C64 矩阵状态：

1. AIV 生成的 `L_powWsGm_` / `INVWsGm_` 只作为入口。
2. AIC 把需要的 C64 block 加载到 `aicHalfL1Power_` / `aicHalfL1Inv_`。
3. 16x16 diagonal inverse 结果写回 full L1 的对应 block，而不是马上写 GM。
4. `Compose32From16()` / `Compose64From32()` 从 full L1 取 block 到 compact tile，结果写回 full L1。
5. 完整 C64 inverse 最后一次性写回 GM，供 final RHS `MatmulImpl` 使用。

这样可以删除大量：

- `Neumann16InverseBlock()` 的中间 `CopyHalfL1MatrixToGmFloatByMmad()`。
- compose 前的 `LoadGmFloatBlockToHalfL1ByMmad()`。
- compose 后的 GM writeback/readback。

这比单纯 L0 ping-pong 更可能带来收益，因为它减少的是外层 GM/Fixpipe/MTE2 固定成本。

### P2：先做 local compose，少碰 RHS

收益判断：中高，风险中。

由于 C16 RHS resident 已证明更慢，C64 不应先复制 RHS resident。更合理的是：

- `Compose32From16()` local 化。
- `Compose64From32()` local 化。
- final `INV` 最后写回 GM。
- final RHS 继续走稳定 FP32 GM `MatmulImpl`。

如果这一步不收益，再做 RHS resident 意义更小。

### P3：真正的 L0A/L0B ping-pong 搬运

收益判断：中，风险中高。

当前 L0 ping-pong 只是顺序使用两套 buffer。要形成真正收益，需要改 helper：

- 把 `RawMmadHalfL1AccumulateTile()` 拆成：
  - `PreloadHalfL1ToL0A/B(bufferIdx)`
  - `MmadFromPreloadedL0(bufferIdx)`
  - `StoreL0CToHalfL1/Gm()`
- 对 compose 可尝试：
  - 当前计算 `lower @ topDiag` 时，预取 `bottomDiag`。
  - 当前写回 `tmp` 时，预取下一 block 的 lower/topDiag。
- 对 diagonal Neumann 可尝试：
  - 当前 `power @ power` 时，预取 `INV` 与 identity 或下一 round power。

但依赖链很强，且目前主要慢在 GM 转换和 scalar/sync。建议 P0/P1/P2 完成后再做。

### P4：packed diagonal 16x16 批量 MMAD

收益判断：中高，风险高。

fast inverse 的 diagonal packing 可以用于 C64 的四个 16x16 diagonal block：

- `0:16`
- `16:32`
- `32:48`
- `48:64`

目标是把多个独立 16x16 MMAD 合并成一个 packed 64x64 block-diagonal MMAD，减少 tiny MMAD 调用次数。

难点：

- 需要在 L0A/L0B 中构造 block diagonal，off-diagonal 保持 0。
- 需要 64x64 compact/full NZ 布局与 `LoadData` 参数正确。
- 需要 64x64 identity 或 block diagonal identity。
- 输出 L0C 的 block diagonal 再写回 full L1。

这一步可能有收益，但建议排在 local compose 之后。

### P5：减少每 chunk 的 `TPipe::Reset()` / Matmul re-init

收益判断：可能高，风险高。

当前 C64 inverse 前后：

- `EnsureAicLocalBuffers()` 可能 `pipe_->Reset()` 并初始化 raw local buffers。
- `RestoreStage1MatmulLocalBuffers()` 再 `pipe_->Reset()` 并重建 `MatmulImpl` buffers。

这发生在每个 chunk 的 Stage1 中。长序列时 chunk 数增加，该成本会放大。

可能方向：

- 尝试让 raw MMAD local buffers 与 Stage1 MatmulImpl buffers共存，避免反复 reset。
- 或将 final RHS raw 化，但 C16 RHS resident 已显示 RHS operand 转换会更慢，因此不应直接照搬。
- 或把 raw inverse 之后的 final RHS 保持 MatmulImpl，但优化 buffer 生命周期，避免完整 reset。

该方向需要非常谨慎，因为 TPipe buffer 冲突会导致隐性精度错误或 507015。

## 对“ping-pong 搬运 C64 矩阵”的判断

可以做，但建议分两层理解：

### 1. C64 full L1 matrix ping-pong/staging

这是更值得先做的。

它不是为了异步 overlap，而是为了避免：

```text
GM -> half L1 -> GM -> half L1 -> GM
```

变成：

```text
GM -> full half L1 -> local compose/Neumann -> GM once
```

这对当前问题更直接。

### 2. L0A/L0B true double-buffer overlap

这是后做的。

当前代码虽然有 `aicHalfL0AAlt_` / `aicHalfL0BAlt_`，但没有异步 pipeline。真正实现需要拆 helper 和手写事件依赖。它能优化 MTE1/MMAD overlap，但如果 GM 转换和 identity scalar 仍在，收益会被盖住。

## 建议下一步

1. 先做 P0：identity cache。
   - 改动小，容易 smoke。
   - 直接针对当前 AIC scalar/MTE/Fixpipe 重复成本。

2. 再做 P1/P2：C64 full L1 staging + local compose。
   - 用 `aicHalfL1Inv_` 保存 C64 final inverse 状态。
   - 用 `aicHalfL1Power_` 保存 C64 L/power 或临时 block 状态。
   - compose 只在 full/compact half L1 与 L0C 之间完成。
   - final RHS 先不 resident。

3. 有收益后再做 P3/P4：
   - L0 true ping-pong overlap。
   - packed diagonal 16x16 批量 MMAD。

4. 暂不优先 P5，除非 profiling 明确 `TPipe::Reset()` / Matmul re-init 是主项。

## 本轮新增结论

- 当前默认 selector 不是 C64；C64 方案需要显式切 selector、重建、安装和重新 compare 后才能作为当前性能结论。
- `C64_3232_FP32_halflocal` 的 `FP16_L1_RESIDENT_INVERSE=false`，仍是稳定 GM/raw MMAD 基线。
- `C64_3232_FP16_halflocal` 的 `FP16_L1_RESIDENT_INVERSE=true`，但 resident 边界仍是 block 局部：输入从 GM 进 half L1，输出写回 GM。
- ping-pong 的第一阶段不应先追 L0 异步重叠，而应先把 C64 full matrix staging 做出来：让 `aicHalfL1Inv_` / `aicHalfL1Power_` 跨 16/32/64 compose 保留状态。
- 真正 L0A/L0B ping-pong 只有在 full L1 staging 后才有明显收益判断价值；否则 GM->half L1 转换、identity 准备、Fixpipe/MTE 同步会盖住 overlap 收益。

## 文档修正

此前 `C64_FAST_INVERSE_ADAPTATION_PLAN.md` 中“compose 仍是 raw GM + scalar backfill”的描述已经不完全匹配当前代码。当前更准确的描述是：

- compose 已封装为 `HalfL1ComposeBlock()`。
- 但该 helper 仍以 GM block 为输入输出，内部做局部 half L1 MMAD。
- 所以优化目标应从“替换 raw GM compose”修正为“把 HalfL1ComposeBlock 的输入/输出从 GM block 改成 full half L1 block”。
