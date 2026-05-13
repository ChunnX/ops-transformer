# 2026-05-08 P7/P8 下三角适配计划

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 当前状态

- active selector：`fulllocalnew/C64_1616_FP16/chunkSize=64`。
- 当前可用实现：`P3 diagonal inverse + P8 scratch64 compose`。
- 精度状态：
  - build/install 成功。
  - smoke `S=17/32/33/49/64/129` 全部 PASS。
  - full compare `20 passed in 6.22s`。
- 性能 smoke：
  - `B8,S16,D64,H16,Hkv4`：neumann/normal `1.313x`。
  - `B8,S64,D64,H16,Hkv4`：neumann/normal `3.497x`。
  - 当前 P8 scratch64 是正确性桥接版本，不是最终 fast inverse 形态。

## 关键判断

`ref/ref.cpp` 的 `CopyDiagonalFractalsL1ToL0()` 和 `CopyOddOrEvenBlocksL1ToL0()` 的打包思想可以参考，但 `InvertSingleTile()` 中的 MBH compose 公式不能原样搬到当前算子。

参考代码面向上三角方向：

```text
LX = even_blocks(X)
RX = odd_blocks(X)
Y = LX @ M_neg + I
X = Y @ RX + LX
```

当前算子 inverse/compose 是下三角方向。二分块为：

```text
A = [ A0  0
      L   A1 ]
```

对应 inverse lower-left 应为：

```text
X10 = A1_inv @ (-L10) @ A0_inv
```

这与当前 `Compose32From16()` / `HalfGmComposeBlock()` 的数学顺序一致：

```text
scratch = lower @ topInv
out     = bottomInv @ scratch
```

因此后续 P8 应优化下三角 compose 的 packed 实现形态，而不是直接实现参考代码的上三角 MBH 公式。

## 修改边界

- 只改 Stage1 inverse/compose 相关 helper。
- 不修改 Stage2/Stage3。
- 不修改 QK、`gBK/vBeta` 生成链路。
- final RHS 保持 BF16 GM：
  - `AICProcess(attnWsGm_, gBKWsGm_, outKCumdecayGm_, ...)`
  - `AICProcess(attnWsGm_, vBetaWsGm_, outVInnerGm_, ...)`
- 当前 P3+P8 通过线必须保留为回退点。
- P7/P8 目标路径不能以 P7 diagonal 写回 GM 再由 P8 读 GM 作为完成形态：
  - P7 packed diagonal 结果应作为 resident/cube 内中间态交给 P8。
  - GM 写回只允许作为诊断 probe，或在 P8 完成 full 64 inverse 后为了现有 AIV `CopyInvHalfToBf16Attn()` / final RHS 边界做最终物化。
  - 因此 `diagnostic GM writeback PASS` 不等于 `target resident path PASS`。

## 2026-05-08 阶段性 64 对齐约束

用户已明确本阶段先不处理 tail block：

- 精度 case 暂时只保留 `S` 为 64 整数倍的用例；变量长度 case 也只使用 64 对齐长度。
- 性能 smoke 不跑全量，只测 `S=64/512/2048/10240`。
- P7/P8 实现可以先按完整 64x64 chunk 推进，不为 `S=17/33/49/129` 等 tail 边界补兼容逻辑。
- 后续重新打开 tail 支持时，再恢复 tail 精度 case，并补充 padded identity 语义验证。

## 实施计划

### P8A：修剪当前 scratch64 中间写回

目标：不改变数学公式，只减少当前 P8 的明显冗余。

- 将 `P8ComposeBoth32From16Scratch64()` 中的 `CopyHalfL1MatrixToGmHalfByMmad(... L_powHalfWsGm_) -> LoadGmHalfBlockToHalfL1(...) -> CopyHalfL1MatrixToGmHalfByMmad(...)` 改为从 L0C 直接写两个目标 16x16 lower-left 子块。
- 将 `P8Compose64From32Scratch64()` 中的整块写 `L_powHalfWsGm_` 再读回 32x32 lower-left，改为从 L0C 直接写目标 32x32 lower-left。
- 验证 `CopyL0CSubMatrixToGmHalf()` 的 `srcMatrixSize=64`、`subSize=16/32`、`unitFlag` 和 flush 顺序。
- 预期：精度应保持，性能可能小幅改善，但仍不是最终 fast inverse。

执行状态：

- 已完成 P8A direct compose，不采用四个 16x16 目标子块写回，也不采用保守回退写法。
- 当前 `16->32` active helper 为 `P8Compose32From16SingleDirect()`，分别处理 top 32 group 和 bottom 32 group。
- 当前 `32->64` active helper 为 `P8Compose64From32Direct32()`，使用紧凑 32x32 L1/L0C 计算后，一次 `CopyL0CMatrixToGmHalf()` 直接写目标 lower-left 32x32。
- active path 仍保持 P3 diagonal inverse；P7 packed diagonal inverse 不接入 active。

验收：

- build/install 通过。
- smoke `S=17/33/49/64` PASS。
- full compare `20 passed in 6.20s`。
- 两点性能 smoke `S=16/64` 已完成，只作观察：
  - `B8,S16,D64,H16,Hkv4`：neumann/normal `1.254x`。
  - `B8,S64,D64,H16,Hkv4`：neumann/normal `1.398x`。

### P7A：单独验证 diagonal fractal load

目标：定位 P7 当前 `S=17` 失败根因，不接入 active。

- 保持 active 仍走 P3 diagonal inverse。
- 新增只读/只写 diagnostic helper：
  - 从 `L_powHalfWsGm_` / `INVInitHalfWsGm_` 取四个 16x16 diagonal block。
  - 打包到 L0A/L0B block-diagonal 位置。
  - 做 `block @ I` 或 `I @ block`。
  - 只把 diagonal 子块写回临时/诊断 GM 区，和原 diagonal block 对比。
- 分别验证 L0A 与 L0B，因为参考代码中 left/right tile 的 inner layout 不同。
- 不再尝试简单改 `CompactMatrixNzOffset(16,16,32)=768`，该 offset 当前判断不是首要问题。
- 诊断写回只能用于定位 load/layout/Fixpipe，不作为 P7 目标验收。

验收：

- `S=16/17/33` smoke。
- 如果诊断输出不一致，继续只定位 L0A/L0B load 或 Fixpipe 子块写回，不进入 active。

### P7B：packed diagonal inverse 替换 P3 diagonal

目标：用 packed diagonal 一次处理多个 16x16 Neumann/MCH 子块。

- 先在 resident L1/L0 形态内得到四个 diagonal inverse。
- P7 中间结果不以 GM roundtrip 作为 P8 输入。
- 若为了定位临时写 GM，必须标记为 diagnostic path，不计入 target PASS。
- 每次只替换 diagonal inverse/resident handoff 的一个环节，不同时改 P8 数学公式。

验收：

- 本阶段先验 `S=64` target resident smoke。
- full compare `20 passed`。
- 失败则停在 P7/P8 当前目标上正向定位布局、写回或同步问题；只有明确放弃该候选或需要重建干净工作面时，才恢复到 P3 diagonal。

### P8B：下三角 odd/even packed compose

目标：参考 `CopyOddOrEvenBlocksL1ToL0()` 的打包方式，但使用下三角数学顺序。

- 对 `16->32`：
  - 打包 lower block `-L10`、top inverse `A0_inv`、bottom inverse `A1_inv`。
  - 计算 `tmp = lower @ topInv`。
  - 计算 `out = bottomInv @ tmp`。
  - 目标是同时处理两个 32 group 的 lower-left 16x16。
- 对 `32->64`：
  - 打包 lower block `-L20` 或已 composed lower-left 32x32。
  - 按 `out = bottomInv @ lower @ topInv` 计算最终 32x32 lower-left。
- 尽量让结果停留在 L0C 后直接写目标子块，避免整块 scratch GM 中转。
- P8B 必须消费 P7 resident diagonal 中间态；不能把 P7 diagonal GM 写回当作正式输入边界。

验收：

- 先只开 `16->32` packed compose，`32->64` 继续当前路径。
- 再开 `32->64` packed compose。
- 每步 build/install、smoke、full compare。

### P8C：full 64 L1 resident 整合

目标：减少 GM roundtrip，但不触碰 final RHS。

- 将 C64 当前 chunk 的 `L_powHalfWsGm_` / `INVInitHalfWsGm_` 一次搬入 full 64 L1。
- P7 diagonal inverse 结果、P8 compose 结果都优先留在 full L1。
- 最终只写回 FP16 `INVInitHalfWsGm_`，由现有 AIV half->BF16 `attnWsGm_` 路径继续消费。
- 这不是可选性能阶段，而是 P7/P8 target path 的正确 handoff 边界；P8A/P7A 中的 GM 写回只作为诊断桥。

验收：

- full compare `20 passed`。
- 典型 shape 性能一次，之后暂停。

## 风险

- L0A/L0B layout 不同，不能用同一套 compact offset 假设。
- 参考代码的上三角 MBH 公式不能直接迁移，必须保持下三角 `bottomInv @ lower @ topInv`。
- 当前 `S>=33` out 误差较 normal 明显增大，后续若阈值收紧，需要单独分析数值误差。
- P8A 只能减少冗余，不代表最终性能会显著改善；真正收益要等 P7/P8 packed resident 打通。
