# 2026-05-06 C64 Fast Inverse 适配计划

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 背景

参考 `gdn-tri-inverse` fast inverse 的 Ascend 910B 实现后，C64 方向最值得借鉴的是两类结构化搬运：

1. `CopyDiagonalFractalsL1ToL0()`：对角 16x16 fractal 搬运，用一次大 MMAD 批量处理多个独立对角块。
2. `CopyOddOrEvenBlocksL1ToL0()`：递归 compose 阶段的 odd/even block packing，用结构化 block 搬运把多个 compose 乘法打包到较大的 MMAD。

当前本算子 C64 相关事实：

- `C64_3232` 是 C64 三方案中最值得继续优化的方向。
- `C64_1616` 的 16x16 子块数量过多，固定调度、同步、MTE/Fixpipe 开销已经被性能数据放大，暂不优先。
- `fulllocal/C64_3232_FP16` 已精度通过，但四点性能仍慢于 normal：`1.307x / 1.957x / 2.286x / 3.011x`。
- `fulllocal/C16_FP16` 已验证 final RHS resident/raw PoC，但性能更慢；说明只让 `INV` operand resident 不够，RHS operand 仍从 FP32 GM 转 half L1/TSCM 会引入额外转换和同步成本。

因此 C64 适配目标不是简单复制 C16 RHS resident，而是优先解决 C64 特有的 compose GM/backfill 和 block packing 问题。

## 明确后的第一步

第一步不是 identity cache，也不是泛化的 L0 ping-pong。第一步必须按网页 fast inverse 的 **MXR** 主线做：

1. 对 C64 内的四个 `16x16` 对角块，按网页 `CopyDiagonalFractalsL1ToL0()` 思路做 **diagonal fractal/block packing**。
2. 用 packed diagonal blocks 执行 `16x16` 对角块的 **MCH/Neumann inverse trick**，把多个独立小块乘法合并成更大的 MMAD 调度。
3. 结果回写到 C64 full half L1/TSCM 的对角位置，作为后续展开 MBH compose 的输入。
4. 随后再按网页 `CopyOddOrEvenBlocksL1ToL0()` 思路做 odd/even block packing，完成 `32x32` 与 `64x64` 的 MBH compose。

也就是说，C64 的优化主线应明确为：

```text
GM 初始 L/I-L
  -> C64 full half L1/TSCM
  -> packed diagonal 16x16 MCH/Neumann
  -> odd/even packed MBH compose
  -> final INV 写回 GM
```

`identity cache`、`full L1 staging`、`L0 ping-pong` 都只能作为这条 MXR dataflow 的支撑优化或后续优化点；不能把它们作为第一阶段主目标。

## 总目标

在 `C64_3232_NeumannCompose` 中借鉴 fast inverse 的 diagonal / odd-even block packing，使 C64 inverse 生成尽量满足：

- 16x16 对角 block inverse 在 half L1/TSCM 中完成。
- 32x32 compose 在 half L1/TSCM 内完成，不通过 GM lower-left backfill。
- 64x64 compose 在 half L1/TSCM 内完成，不通过 GM lower-left backfill。
- final `INV` 只在必要边界写回 GM；在 RHS operand 不能片上生成前，不强行启用 RHS resident。

## 非目标

- 不优先做 `C64_1616`。
- 不再尝试 FP32 L1/TSCM resident；DAV2201 上 `float L0C -> float L1/TSCM` 已验证不可编译。
- 不把未通过 compact NZ / LoadData 探针的 resident compose 切到默认 selector。
- 不先优化 Stage2/Stage3。
- 不把 C16 RHS resident 的当前实现直接复制到 C64；该路径已经证明会因为 RHS FP32 GM -> half L1/TSCM 转换变慢。

## 当前阻塞点

### 1. Compose lower-left 仍依赖 GM

当前 `fulllocal/C64_3232_FP16` 的 resident 分支中：

- `Compose32From16()` 仍用 `RawMmadBlock(INVWsGm_[bottom, top], INVWsGm_[top, top], tmpWsGm_[bottom, top])`。
- `Compose64From32()` 仍用 `RawMmadBlock()` 从 GM 读写 lower-left block。
- compose 后通过 `CopyGmFloatMatrixToHalfL1()` 做 scalar full-L1 backfill。

这说明 C64 的 resident 状态没有跨 compose 保持完整，性能和正确性都依赖 GM 中转。

### 2. half L1 compact/full 布局仍脆弱

当前已有：

- `MatrixNzOffset()`：64x64 full half L1/TSCM 布局。
- `CompactMatrixNzOffset()`：16x16/32x32 compact tile 布局。
- `CopyHalfL1FullBlockToCompact()` / `CopyHalfL1CompactToFullBlock()`。

但 direct 32x32 resident 曾出现 `S=17` 大误差和 `LoadData2DParamsV2` `nan`，说明 32x32 compact/full offset、A/B LoadData 语义和 compose block 位置仍需单独探针验证。

### 3. RHS resident 当前不是收益路径

C16 final RHS resident 已经做到 `INV` operand 直接消费 half L1/TSCM，但性能更慢：

- `S=16/64/125/2048` ratio 从 no-RHS-resident 的 `1.084x / 2.140x / 2.503x / 4.076x` 变成 `1.109x / 2.423x / 2.671x / 5.026x`。

根因是 RHS operand `gBK/vBeta` 仍从 FP32 GM workspace 转 half L1/TSCM。C64 若直接复制该路径，额外转换成本会更大。

## 方案分层

### Phase 0：固定回退基线

回退基线固定为：

- 性能基线：`halflocal/C64_3232_FP32`。
- resident 功能基线：已精度通过的 `fulllocal/C64_3232_FP16`。

所有 C64 适配实验必须保留 selector 可回退，且每一步必须先 smoke compare，再全量 compare，再跑性能。

### Phase 1：网页方法第一步，Diagonal fractal packing + 16x16 MCH/Neumann

目标：按网页方法先把 C64 内四个 `16x16` 对角块打包搬入 L0A/L0B，用较大的 MMAD 调度批量完成对角块 MCH/Neumann inverse。

实现内容：

1. 先做必要的 C64 full half L1/TSCM offset 审计，作为 diagonal packing 的正确性前置：
   - top-left: `(0,0)`
   - `A00`
   - `A11`
   - `A22`
   - `A33`
2. 新增 `CopyDiagonal16BlocksHalfL1ToL0A/B()` 或等价 helper。
3. 输入是 C64 full half L1/TSCM。
4. 输出是 L0A/L0B 中 block-diagonal packing 后的 tile，非对角 block 保持 0。
5. 用 packed 对角块批量执行当前 `Neumann16InverseBlock()` 中的三轮 MCH/Neumann：
   - packed `power @ power`
   - packed `INV @ I + INV @ power`
   - packed result 写回 full half L1 的四个对角位置。

用途：

- 批量验证四个 16x16 diagonal block 的 `INV @ I`、`power @ power` 等 MMAD。
- 替换逐个 `Neumann16InverseBlock()` 的 tiny MMAD 调度，减少固定调度、同步和 MTE/Fixpipe 放大。
- 为后续 odd/even MBH compose 准备 C64 full L1 resident 的对角逆。

验收：

- 对 `64x64` full L1 中四个 diagonal 16x16 block 做 `block @ I`，一次 packed MMAD 写回 GM，与逐块 raw MMAD baseline 一致。
- 先只验证 `A` path，再验证 `B` path，最后验证 A/B 同时参与 MMAD。
- `S=16/17/33/64` smoke 通过。
- 全量 compare 通过后，才能进入 Phase 2。

风险：

- 当前 `CopyHalfL1MatrixToL0A/B()` 是 compact tile 语义，不一定能直接搬 full L1 中离散 diagonal block。
- 如果 AscendC `LoadData` 无法表达离散 fractal extract，需要先退化为 scalar full->compact copy，再从 compact L1 LoadData；这会降低性能，但可作为正确性基线。

### Phase 2：网页方法第二步，Odd/even block packing + MBH compose

目标：按网页 `CopyOddOrEvenBlocksL1ToL0()` 思路，把展开 MBH 中的奇偶 block 结构化搬入 L0，完成 `32x32` 与 `64x64` compose。

实现内容：

1. 新增 odd/even block packing helper，先覆盖 C64 的固定层级：
   - `16 -> 32` compose。
   - `32 -> 64` compose。
2. 输入来自 C64 full half L1/TSCM 中的对角逆和 lower-left block。
3. 中间结果保留在 L0C / half L1，不通过 `tmpWsGm_`。
4. 结果写回 full half L1 的 lower-left block。

验收：

- `S=17` smoke 通过，覆盖 `16 -> 32` compose。
- `S=33` smoke 通过，覆盖第二个 32 block。
- `S=64` smoke 通过，覆盖完整 `32 -> 64` compose。
- 全量 compare 通过。

### Phase 3：Compose32From16 resident 对齐实现

目标：消除 `Compose32From16()` 中 GM raw MMAD 和 scalar full-L1 backfill。

数学结构：

对于 32x32 lower triangular block：

```text
A = [A00  0
     A10 A11]

A^-1 lower-left = - A11^-1 * A10 * A00^-1
```

当前实际变量中 lower-left 的符号已经由现有公式/矩阵定义隐含，实施时以 halflocal GM baseline 为准，不重新改数学符号。

实现计划：

1. 对 `base=0` 和 `base=32` 分别处理 32x32。
2. 保持两个 16x16 diagonal inverse 在 full half L1 中：
   - `INV00`
   - `INV11`
3. 将 lower-left 原始 block 或中间 block 搬到 compact half L1。
4. 用 half L1 raw MMAD 执行：
   - `tmp10 = lower10 @ INV00`
   - `new10 = INV11 @ tmp10`
5. 将 `new10` 直接写回 full half L1 的 `(base+16, base)`。
6. 不写 `tmpWsGm_`，不调用 `CopyGmFloatMatrixToHalfL1()` backfill。

建议新增 helper：

- `HalfL1Compose32From16(base)`
- `CopyHalfL1BlockFullToCompact(rowBase, colBase, blockSize)`
- `CopyHalfL1BlockCompactToFull(rowBase, colBase, blockSize)`
- `HalfL1BlockMatmul(lhsCompact, rhsCompact, dstCompact, blockSize)`

验收：

- `S=17` smoke 通过。
- `S=33` smoke 通过。
- `S=64` smoke 通过。
- 全量 compare 通过。
- 与当前 `fulllocal/C64_3232_FP16` GM compose 的中间 lower-left dump 在可接受精度内一致。

### Phase 4：Compose64From32 resident 对齐实现

目标：消除 `Compose64From32()` 中 GM raw MMAD 和 scalar full-L1 backfill。

结构：

```text
A64 = [A00  0
       A10 A11]

lower-left inverse block = A11^-1 * A10_related * A00^-1
```

实施以当前 `Compose64From32()` 的 GM baseline 为准：

当前 GM 路径是：

```cpp
RawMmadBlock(INV[32,0], INV[0,0], tmp[32,0], 32);
RawMmadBlock(INV[32,32], tmp[32,0], INV[32,0], 32);
CopyGmFloatMatrixToHalfL1(INV[32,0], aicHalfL1Inv_, 32, 32, chunkSize_, 32, 0);
```

resident 适配目标：

1. 从 full half L1 取 `INV[32,0]`、`INV[0,0]`、`INV[32,32]` 到 compact 32x32。
2. `tmp = INV[32,0] @ INV[0,0]` 在 compact half L1/L0C 内完成。
3. `new = INV[32,32] @ tmp` 在 compact half L1/L0C 内完成。
4. `new` 写回 full half L1 的 `(32,0)`。
5. 不写 `tmpWsGm_` / `INVWsGm_[32,0]`，不做 scalar backfill。

验收：

- `S=64` smoke 通过。
- `S=125` smoke 通过。
- 全量 compare 通过。
- 性能至少不慢于当前 `fulllocal/C64_3232_FP16`，否则只保留为 PoC。

### Phase 5：C64 final writeback policy

目标：在 compose resident 通过后，再决定 final `INV` 的消费策略。

策略 A：先不做 RHS resident

- 将 full half L1/TSCM 的 final `INV` 一次性写回 `INVWsGm_`。
- final RHS 继续用 FP32 GM `MatmulImpl`。
- 这是最小风险路径，用于验证 compose resident 是否能单独带来收益。

策略 B：条件启用 RHS resident

只有当下列任一条件满足时再启用：

- `gBK/vBeta` 生成端能直接产出 half L1/TSCM/NZ tile。
- 找到 FP32 GM -> half L1 NZ 的直接搬运/转换接口，避免 `RHS @ I` 绕路。
- C64 compose resident 已明显减少总耗时，RHS resident 的额外转换成本可被抵消。

默认选择：

- Phase 5 初始采用策略 A。

### Phase 6：后续优化点

以下不作为第一步，只在网页方法的 diagonal packing + MBH compose 跑通后再考虑：

1. `identity cache`。
2. 真正的 L0A/L0B ping-pong overlap。
3. RHS resident。
4. 减少 `TPipe::Reset()` / Matmul re-init。

这些优化点可以叠加到 MXR dataflow 上，但不能替代网页方法本身。

## 文件与开关建议

建议在 `op_kernel/fulllocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_fulllocal.h` 中新增编译期开关：

```cpp
constexpr bool C64_FASTINV_DIAGONAL_PACKING = false;
constexpr bool C64_FASTINV_ODD_EVEN_PACKING = false;
constexpr bool C64_RESIDENT_COMPOSE32 = false;
constexpr bool C64_RESIDENT_COMPOSE64 = false;
constexpr bool C64_RESIDENT_FINAL_RHS = false;
constexpr bool C64_DEBUG_DUMP_L1_BLOCKS = false;
```

默认全部关闭，逐项打开。

selector 切换规则：

- 未全量 compare 通过前，不允许将新路径作为默认 selector。
- 若当前 active selector 仍是 `fulllocal/C16_FP16`，C64 实验前必须切回 `fulllocal/C64_3232_FP16/chunkSize=64`，并重新 build/install/compare。

## 验证矩阵

### 每个 Phase 的最小 smoke

- `S=16`：短尾，只覆盖 16x16。
- `S=17`：触发 32 compose 的最小越界敏感 case。
- `S=33`：触发第二个 32 block 的最小 case。
- `S=64`：完整 C64。
- `S=125`：多 chunk + tail。

### 全量 compare

```bash
CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare
```

### 性能选点

继续沿用当前四点：

- `B=8,S=16,D=64,Hq=16,Hkv=4`
- `B=8,S=64,D=64,Hq=16,Hkv=4`
- `B=8,S=125,D=64,Hq=16,Hkv=4`
- `B=8,S=2048,D=128,Hq=16,Hkv=4`

对比对象：

- normal
- `halflocal/C64_3232_FP32`
- 当前 `fulllocal/C64_3232_FP16`
- 新 resident compose path

## 成功标准

### 正确性

- 每个 phase 的 smoke 全 PASS。
- 全量 compare `20 passed`。
- 非 16 倍数维度不应进入未验证 resident path；必须 fallback 或显式支持。

### 性能

最低标准：

- resident compose path 不慢于当前 `fulllocal/C64_3232_FP16`。

目标标准：

- `S=64` 和 `S=125` 明显优于当前 `fulllocal/C64_3232_FP16`。
- 至少接近或优于 `halflocal/C64_3232_FP32`。

若只是功能通过但性能更慢：

- 保留为 PoC，不作为默认性能线。
- 继续分析 AIC scalar、MTE1/MTE2、Fixpipe 是否来自 full<->compact scalar copy 或 RHS 转换。

## 风险

1. AscendC 没有 PTO `TEXTRACT` 等价高层接口，结构化 fractal extract 可能需要手写 LoadData offset 或退化为 scalar copy。
2. half L1/TSCM resident 会引入精度变化，必须以 full compare 为准。
3. full L1 64x64 与 compact 16/32 tile 的 NZ offset 容易混淆，尤其是 B operand transpose/load 语义。
4. C64 compose resident 若仍大量使用 scalar full<->compact copy，可能正确但更慢。
5. RHS resident 当前已证明更慢，不能在 C64 中默认打开。

## 预期结论

第一步明确为 Phase 1：diagonal fractal/block packing + 16x16 MCH/Neumann 批处理。

原因：

- 这是网页 fast inverse 的入口优化，对应 `CopyDiagonalFractalsL1ToL0()`。
- 它直接减少 C64 内多个 16x16 diagonal inverse 的 tiny MMAD 调度。
- 它为后续 odd/even MBH compose 提供 full L1 resident 的对角逆状态。

Phase 2 再做 odd/even block packing + MBH compose；identity cache、ping-pong、RHS resident 都是后续叠加优化点。
