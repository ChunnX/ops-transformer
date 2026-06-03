# Strict Packed64 One-shot Flow And Open Issues

本文记录 strict packed64 one-shot diagonal solve 的临时实现流程，以及当前没有采纳为 active 的原因。当前 active 代码仍是 Phase 10 baseline：packed64 容器 + proven `16x16` diagonal solve。

## 结论先行

- strict one-shot 只替换 P7 diagonal solve；P8 compose、BF16 `attnWsGm_` 写回、resident L1 copy 和 fused RHS 仍沿用 Phase 10。
- correctness 已通过，说明 L1->L0 placement、Fixpipe packed offset、identity placement 这三类边界没有明显功能错误。
- performance 没采纳：one-shot ratio 为 `1.319 / 2.091 / 2.961 / 3.034`，Phase 10 为 `1.379 / 2.001 / 2.760 / 2.824`，大 shape 退化。
- 当前最大问题不是 correctness，而是 one-shot 的 full-64 identity 构造、power/update 开销，以及没有把 P8 按 ref 的 odd/even packed 抽取方式真正合并。

## Buffer 命名

| 名称 | 当前 tensor | 语义 | shape |
|---|---|---|---|
| `result64` | `aicHalfL1P3Result_` | diagonal inverse / compose 最新结果 | `64x64 half NZ` |
| `diagPower64` | `aicHalfL1TilePower_` | diagonal power 递推 | `64x64 half NZ` |
| `lowerWork64` | `aicHalfL1Scratch_` | P8 lower-left 输入 | `64x64 half NZ` |
| `identity64` | `aicHalfL1Identity_` | block-diagonal identity | `64x64 half NZ` |
| `compact16` | `aicHalfL1TileInv_` | one-shot identity 构造辅助 / fallback scratch | `16x16 half NZ` 或更大 scratch |
| `cInv` | `aicL0C_` | inverse 累积 L0C | `64x64 fp32` |
| `cPower` | `aicL0C_[64*64]` | power 输出 L0C | `64x64 fp32` |

## 临时 one-shot 调用树

```cpp
C64InverseBy16x16Neumann(paraIdx)
  EnsureAicLocalBuffers()
  PrepareRawMmadState()
  LoadPacked64DiagonalInputsToL1(paraIdx)
  P7Packed64DiagonalNeumannOneShotToResult()
  PreloadP8LowerLeftBlocksToLowerWork64(paraIdx)
  P8Compose32From16SingleDirect(paraIdx, 0)
  P8Compose32From16SingleDirect(paraIdx, 32)
  P8Compose64From32Direct32(paraIdx)
  StorePackedL1InverseToBf16Attn(paraIdx)
  StorePackedL1InverseToResidentL1(paraIdx)
```

临时版本最后恢复到 Phase 10，因此 active 文件中没有 `P7Packed64DiagonalNeumannOneShotToResult()`。

## 1. Diagonal 输入搬入

调用：

```cpp
LoadPacked64DiagonalInputsToL1(paraIdx)
```

动作：

1. `ZeroHalfL1(result64, 64*64)`
2. `ZeroHalfL1(diagPower64, 64*64)`
3. `ZeroHalfL1(lowerWork64, 64*64)`
4. `ZeroHalfL1(compact16, 16*16)`
5. 对 `base = 0, 16, 32, 48` 调：

```cpp
LoadPacked64DiagonalInputBlock(paraIdx, base)
```

每个 block 搬运：

```cpp
L_powHalfWsGm_[paraIdx, base, base]  -> diagPower64[base, base]
INVInitHalfWsGm_[paraIdx, base, base] -> result64[base, base]
```

shape：

- 每次 GM 源是 `16x16 half`，GM stride 为 `64`。
- 目标是 packed `64x64 half NZ` 的对角位置。
- 四次后，`result64` 和 `diagPower64` 都是 block-diagonal packed64，off-diagonal 是 0。

## 2. identity64 构造在做什么，为什么显得多余

临时 one-shot 里有：

```cpp
PreparePacked64IdentityFromPacked16()
```

流程：

1. `WriteIdentityBlockGm(tmpWsGm_, 16)` 生成 `16x16 fp32` identity。
2. `LoadGmFloatBlockToHalfL1ByMmad(tmpWsGm_, compact16, 16)` 把它转成 `16x16 half NZ`。
3. 清零 `identity64`。
4. 循环 `base = 0,16,32,48`：
   - `RawMmadPackedHalfL1AccumulateTile(compact16, compact16, cInv, 16x16)`
   - 得到 `I16 @ I16 = I16`
   - `CopyL0CMatrixToPackedHalfL1(identity64, cInv, base, base, 16, 16, packed=64)`

这一步确实偏多余。它的作用只是生成 block-diagonal `I64`，用于后面：

```cpp
cInv = result64 @ identity64 + result64 @ scratch64
```

更直接的优化方向：

- 复用已有 `aicHalfL1Identity_`，不要每个 para 重建 packed64 identity。
- 在 first accumulate 里避免 `result64 @ I`，直接把 `result64` 作为 C 初值或保留当前 `result64` 并只加 `result64 @ power`。
- 或者构造一次 `identity64` 后跨 para / 跨 chunk 复用，但需要确认 L1 生命周期和 workspace 复用边界。

所以你说“2 很多余”是对的。它是 correctness 定位用的保守写法，不是最终性能实现。

## 3. one-shot diagonal solve 计算流程

临时函数：

```cpp
P7Packed64DiagonalNeumannOneShotToResult()
```

初始：

- `result64 = INV0`，block diagonal `64x64`。
- `diagPower64 = P`，block diagonal `64x64`。
- `identity64 = I`，block diagonal `64x64`。

第 1 轮：

```cpp
scratch64 = diagPower64 @ diagPower64
```

- call: `HalfL1PowerDiagonalOnlyPackedBlockOnC(diagPower64, diagPower64, scratch64, 64, cPower)`
- 底层：`RawMmadHalfL1DiagonalOnlyAccumulateTile(..., blockSize=64)`
- L1->L0：只抽取 4 个 diagonal `16x16` fractal 到 L0A/L0B。
- Mmad: `64x64 x 64x64 -> 64x64`，但 off-diagonal 为 0。
- Fixpipe: `cPower(fp32) -> scratch64(half NZ)`。

然后：

```cpp
cInv = result64 @ identity64 + result64 @ scratch64
result64 = cInv
```

- 第一个 MMAD: `RawMmadHalfL1DiagonalOnlyAccumulateTile(result64, identity64, cInv, 64, INIT, NON_FINAL)`
- 第二个 MMAD: `RawMmadHalfL1DiagonalOnlyAccumulateTile(result64, scratch64, cInv, 64, ACCUM, FINAL)`
- Fixpipe: `CopyL0CMatrixToPackedHalfL1(result64, cInv, 0, 0, 64, 64, 64)`
- 语义：`result64 = result64 * (I + P^2)`。

第 2 轮：

```cpp
diagPower64 = scratch64 @ scratch64   // P^4
result64 += result64 @ diagPower64
```

第 3 轮：

```cpp
scratch64 = diagPower64 @ diagPower64 // P^8
result64 += result64 @ scratch64
```

最终 diagonal result 语义：

```text
result64 = INV0 * (I + P^2) * (I + P^4) * (I + P^8)
```

注：这里是当前 repo 的 Neumann 变量命名语义。它和 ref 里的 `M_neg/I_neg` 符号方向不同，不能直接按 ref 的上三角公式照搬。

## 4. `INVInitHalfWsGm_ -> lowerWork64` 是否应该和 3 并行

应该，这是后续优化方向。

当前 Phase 10 主线里顺序是：

```cpp
LoadPacked64DiagonalInputsToL1(paraIdx);
PreloadP8LowerLeftBlocksToLowerWork64(paraIdx);
P7SolvePacked64DiagonalBlockToResult(...);
P8Compose...
```

one-shot 临时说明里我把 `PreloadP8LowerLeftBlocksToLowerWork64()` 放在 one-shot 后面讲，是为了先讲 diagonal solve 逻辑，不代表最终调度应该这么排。

更合理的调度是：

```text
MTE2/MTE1: preload lowerWork64 lower-left blocks
M:        one-shot diagonal power/inv MMAD
FIX:      previous cTile writeback
```

也就是让 `INVInitHalfWsGm_ -> lowerWork64` 的搬运覆盖 diagonal one-shot 的 MMAD/Fixpipe。当前没有这么做完全，原因是先保 correctness，把 lowerWork64 从 diagonal scratch 生命周期里隔离出来；下一步可以专门做 pipeline overlap。

需要注意两个条件：

- `lowerWork64 = aicHalfL1Scratch_` 在 one-shot diagonal 阶段也被用作 `scratch64`，如果要并行，需要把 diagonal scratch 和 lowerWork64 分开，或重排 scratch 生命周期。
- MTE1 写 lowerWork64 和 Fixpipe 写 scratch64/result64 不能落在同一 L1 区域，否则会踩数据。

所以“应该并行”是对的，但当前资源别名下不能直接并行，需要先解决 `scratch64` 与 `lowerWork64` 的冲突。

## 5. 为什么求逆结果还要搬出

这里有两种“搬出”，含义不同。

### 5.1 L0C -> L1 是必须的

MMAD 结果在 L0C fp32，下一步 compose / RHS 需要 L1 half 输入，所以必须：

```cpp
CopyL0CMatrixToPackedHalfL1(result64, cInv, ...)
```

这个“搬出”是 L0C -> L1，不是 GM fallback。

### 5.2 L1 -> BF16 GM 当前仍保留

当前 active 还会：

```cpp
StorePackedL1InverseToBf16Attn(paraIdx)
```

这会把 `result64` 写到 `attnWsGm_`，包括 lower triangle 和 upper zero block。

理论上最终目标应是只在 L1 消费 inverse：

```cpp
StorePackedL1InverseToResidentL1(paraIdx)
FusedRhsAICProcess(... reads resident L1 ...)
```

但现在 BF16 GM 写回还不能删。已经试过删除/跳过 BF16 写回或 upper zero writeback，`S=1024` 会失败，表现为 `out/state` 接近全 0。说明 `attnWsGm_` 的写回或清零仍然承担了某个跨 chunk 生命周期、workspace side effect 或后续可见性边界。

所以当前状态：

- L1 resident 消费已经有了。
- GM BF16 写回仍作为 correctness 保护保留。
- 下一步如果要删 GM 写回，需要先重新设计 `attnWsGm_` 生命周期/清零边界，而不是直接删除。

## 6. P8 与 ref 的 `16x16*4 -> 32x32` 抽取为什么看起来不一样

ref 的关键函数是：

```cpp
CopyOddOrEvenBlocksL1ToL0(src, dst, block_size)
```

它做的是 MBH 递归层的 packed 抽取：

- `block_size=16` 时，按 even/odd 抽取 `16x16` block，组装成 `32x32` 层的输入。
- `block_size=32` 时，按 even/odd 抽取 `32x32` block，组装成 `64x64` 层的输入。
- 左操作数 L0A 抽 even blocks，右操作数 L0B 抽 odd blocks。
- 目标是把多个小 block 合成一次大 MMAD 的输入，减少小 MMAD 次数并提高矩阵单元利用率。

当前 repo 的 P8 不是按 ref 的 upper-triangular MBH 公式直接写的，而是 lower-triangular compose：

```text
tmp = lower @ topInv
out = bottomInv @ tmp
```

当前代码实现为：

```cpp
P8Compose32From16SingleDirect(base)
  tmp = lowerWork64[bottom, top] @ result64[top, top]        // 16x16
  result64[bottom, top] = result64[bottom,bottom] @ tmp      // 16x16

P8Compose64From32Direct32()
  tmp = lowerWork64[32,0] @ result64[0,0]                    // 32x32
  result64[32,0] = result64[32,32] @ tmp                     // 32x32
```

这和 ref 的差异有两层：

1. 公式方向不同：
   - ref 文档多是 upper/通用 MBH 视角。
   - 本 repo 必须保持 lower-triangular：`out = bottomInv @ lower @ topInv`。

2. 打包方式还没做到 ref 那种一次抽取：
   - 当前 `P8Compose32From16SingleDirect()` 还是对 `(16,0)` 和 `(48,32)` 分别做 `16x16` compose。
   - ref 希望把这两个 independent `16x16` compose 的输入抽到一个 `32x32` packed L0 里，一次 MMAD 覆盖两组。
   - 当前 `P8Compose64From32Direct32()` 已经是 `32x32` compose，但它仍是直接按 lowerWork64/result64 的 packed offset 读，不是完整复刻 `CopyOddOrEvenBlocksL1ToL0()` 的 odd/even 抽取模式。

所以你感觉“不一样”是准确的：当前实现是 correctness-first 的 lower-triangular direct compose；还没有把 ref 的 odd/even block packing 优化真正迁过来。

## 7. 后续更合理的优化顺序

1. 不再重复证明 one-shot placement；它 correctness 已过。
2. 删除或缓存 `identity64` 构造，避免 `result64 @ I` 这种明显冗余路径。
3. 拆开 `scratch64` 和 `lowerWork64`，让 lower-left GM->L1 preload 与 diagonal one-shot MMAD/Fixpipe 重叠。
4. 迁移 ref 的 `CopyOddOrEvenBlocksL1ToL0()` 思路，但按 lower-triangular 方向改写：
   - `16x16*2` independent compose 合成 `32x32` packed MMAD。
   - `32x32*2` compose 合成 `64x64` 层。
5. 重新设计 `attnWsGm_` 生命周期后，再尝试删除 BF16 GM 写回，仅保留 resident L1 RHS 消费。
