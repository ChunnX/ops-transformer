# AIC -> AIV M 轴切分设计

## 1. 目的与边界

本文单独描述 `qkv_rms_norm_rope_cache_with_k_scale` 中 Q/K Cube 结果从 AIC 发送到两个 AIV UB 的切分方式，重点覆盖：

1. AIC 侧 `MTE1 -> MMAD -> FIXP` 的 tile 内数据边界。
2. `dualDstCtl = 1` 时 FIXP 在 M 轴上的双目标切分语义。
3. odd token 场景下 Cube/FIXP 对齐到 2 后，AIV0/AIV1 各自实际消费的行数。
4. Vec 后处理读取 Q/K UB 时，如何区分 AIV-local layout 和全局 tile row offset。

本文只约束 Q/K 的 AIC -> AIV 数据切分。CrossCore flag 的语义、pipe 和 Set/Wait 次数仍以 `06_inter_core_sync_design.md` 为准；核内 pipe event 仍以 `07_intra_core_sync_design.md` 为准。

## 2. 基本公式

一个 AIC 对应两个 AIV：

```text
AIC:
  cubeIndex = GetBlockIdx()

AIV:
  cubeIndex  = GetBlockIdx() / 2
  aivLocalId = GetSubBlockIdx()  // 0: AIV0, 1: AIV1
```

对一个 token tile：

```text
actualToken      = tile.tokenSize
cubeToken        = AlignUp(actualToken, 2)
cubeHalfToken    = cubeToken / 2

aiv0ActualToken  = CeilDiv(actualToken, 2)
aiv1ActualToken  = actualToken - aiv0ActualToken
```

约束：

1. controller 在 `TileParam` 中前置 `aivBlockTokenOffset/aivTokenOffset/aivTokenSize`，仍按 `actualToken` 切分，不能把 padding token 暴露给 Vec 后处理。
2. `cubeToken` 只用于 AIC 侧 Q/K Cube/FIXP 的 M 轴布局。
3. 当 `actualToken` 为奇数时，`cubeToken` 多出的 token 只是 FIXP M 轴二分需要的 padding 行，AIV 不做量化、不写 `q_out`、不写 `k_cache`、不写 `k_scale_cache`。

## 3. AIC 侧 tile 内布局

Q/K 的 `MTE1 -> MMAD -> FIXP` 按 segment 分开处理。当前代码不要求 Q/K aligned rows 同时放入同一个 32K L0A tile，也不再使用旧约束：

```text
AlignUp(cubeToken * qHeadNum, 16) + AlignUp(cubeToken * kvHeadNum, 16) <= 128
```

当前 host 侧需要保证的是 side-by-side L0C 物理行数满足 double buffer 预算：

```text
AlignUp(cubeToken * qHeadNum, 16) + AlignUp(cubeToken * kvHeadNum, 16) <= 256
```

AIC 对 Q segment 和 K segment 分别调用 `ComputeSegmentToL0C`。单个 segment 若 full-D 不满足 L0A 容量，则根据当前 `segmentRows` 和 L0A buffer size 动态计算 D 方向 tile，并在 L0C 累加，不在 M/N 轴拆 Q/K head。

每一轮 segment 内：

```text
MTE1:
  load current Q or K segment chunk to L0A

MMAD:
  current segment @ corresponding rotation -> L0C segment area

FIXP:
  current segment from L0C -> two AIV UB, split by M axis
```

L0C 结果也应保持在同一个 tile 内部：

```text
Q L0C base = 0
K L0C base = AlignUp(cubeToken * qHeadNum, 16) * 128
```

因此当前实现保留 Q/K side-by-side 的 L0C base 关系，但 L0A 输入按 Q 或 K segment 独立加载，必要时只切 D。

## 4. FIXP dual M 轴切分

期望 FIXP 配置：

```text
dualDstCtl = 1
```

语义：

1. 一条 FIXP 指令的 M 轴被分成两半。
2. 前半 M rows 发送到 AIV0 的 UB。
3. 后半 M rows 发送到 AIV1 的 UB。
4. `subBlockId` 不再作为 Q/K 输出分发的核心路由方式。

Q segment 和 K segment 分别执行 FIXP。每个 segment 的 M 轴 row 数为：

```text
Q segment rows = cubeToken * qHeadNum
K segment rows = cubeToken * kvHeadNum
```

由于 `cubeToken` 已经对齐到 2，FIXP 按 M 轴二分后：

```text
Q rows per AIV capacity = cubeHalfToken * qHeadNum
K rows per AIV capacity = cubeHalfToken * kvHeadNum
```

注意这里的 `capacity` 表示 FIXP 输出到每个 AIV UB 的布局容量，不表示当前 AIV 实际需要后处理的有效行数。

## 5. 每个 AIV 的本地 UB layout

FIXP dual M 轴切分后，每个 AIV 看到的是自己的本地 UB，不是完整 tile 的全局 M 轴视图。

AIV0 本地 UB：

```text
qk_after_cube_ub:

row 0
+-----------------------------------------+
| Q rows, capacity cubeHalfToken*qHeadNum|
+-----------------------------------------+
| K rows, capacity cubeHalfToken*kvHeadNum|
+-----------------------------------------+
```

AIV1 本地 UB：

```text
qk_after_cube_ub:

row 0
+-----------------------------------------+
| Q rows, capacity cubeHalfToken*qHeadNum|
+-----------------------------------------+
| K rows, capacity cubeHalfToken*kvHeadNum|
+-----------------------------------------+
```

两个 AIV 的本地 Q base 都是 `0`。两个 AIV 的本地 K base 都按 Q 的本地布局容量计算：

```text
localQBaseRow = 0
localKBaseRow = cubeHalfToken * qHeadNum

localQBaseOffset = localQBaseRow * headDim
localKBaseOffset = localKBaseRow * headDim
```

但是实际有效 token 数必须区分 AIV：

```text
AIV0 validToken = aiv0ActualToken
AIV1 validToken = aiv1ActualToken

Q valid rows = validToken * qHeadNum
K valid rows = validToken * kvHeadNum
```

因此，Vec 后处理不能使用完整 tile 逻辑行空间的全局 row offset 作为 FIXP 后本地 UB 的读取起点。当前 `TileParam::aivTokenOffset/aivTokenSize` 表示本 AIV 的有效 token 范围；FIXP dual 模式已经把 M 轴前半和后半分别写入两个 AIV 的本地 UB，所以后处理读取 Q/K 时应从 AIV-local base 开始，并只消费 `tile.aivTokenSize` 对应的有效行。

`localQBaseRow/localKBaseRow` 只定义 Q/K segment 在本 AIV UB 中的起始位置，不定义 segment 内部 row order。segment 内部 row order 在 RmsNormRopeNz 阶段统一由 `layout_q_out` 决定：`NTD` 为 `[head, localToken]`，`TND` 为 `[localToken, head]`。Cube/Fixpipe 不改变这个 row order，Q/K 后处理必须按 `layout_q_out` 读取。

## 6. Vec 后处理读取规则

`QuantQkToFp8` 和 `ScatterQkOutputs` 应使用同一套 AIV-local Q/K layout。

量化读取：

```text
qSrcOffset = 0
kSrcOffset = cubeHalfToken * qHeadNum * headDim

qRows = tile.aivTokenSize * qHeadNum
kRows = tile.aivTokenSize * kvHeadNum
```

写回遍历。Q/K FP32 source row、Q FP8/scale staging 和普通输出写回均按 `layout_q_out` 解释。`TND->NTD` 时，RmsNormRopeNz 阶段已经按 NTD row order 写 `qRopeNz/kRopeNz`，Fixpipe 输出的 Q/K FP32 也是 NTD row order，因此 Q/K dynamic quant 不需要额外转置：

```text
layout_q_out == NTD:
  q row base = qHead * cubeHalfToken + tokenIdx
  k row base = cubeHalfToken * qHeadNum + kHead * cubeHalfToken + tokenIdx

layout_q_out == TND:
  q row base = tokenIdx * qHeadNum + qHead
  k row base = cubeHalfToken * qHeadNum + tokenIdx * kvHeadNum + kHead

for tokenIdx in [0, tile.aivTokenSize):
  only consume rows belonging to valid tokenIdx
```

`q_scale` staging 和 `k_scale` staging 仍按实际 token 数紧凑排列：

```text
q scale rows = tile.aivTokenSize * qHeadNum
k scale rows = tile.aivTokenSize * kvHeadNum
```

不要用 `cubeHalfToken` 扩大 scale staging 的有效写回范围；padding 行不生成有效 scale。

## 7. 示例

### 7.1 actualToken = 5

```text
actualToken   = 5
cubeToken     = 6
cubeHalfToken = 3

AIV0 validToken = 3
AIV1 validToken = 2
```

FIXP 输出容量：

```text
Each AIV local Q capacity = 3 * qHeadNum rows
Each AIV local K capacity = 3 * kvHeadNum rows
```

实际消费：

```text
AIV0:
  Q valid rows = 3 * qHeadNum
  K valid rows = 3 * kvHeadNum

AIV1:
  Q valid rows = 2 * qHeadNum
  K valid rows = 2 * kvHeadNum
  padding rows = 1 token worth of Q/K capacity, ignored
```

AIV1 的 K 本地 base 仍是：

```text
3 * qHeadNum * headDim
```

不是：

```text
2 * qHeadNum * headDim
```

### 7.2 actualToken = 1

```text
actualToken   = 1
cubeToken     = 2
cubeHalfToken = 1

AIV0 validToken = 1
AIV1 validToken = 0
```

AIV1 没有 Q/K 有效行，不应量化或写回 padding 行。但 AIV1 仍必须参与固定 CrossCore Set/Wait 协议，否则 AIC 或另一个 AIV 会在尾块空工作路径上死锁。

### 7.3 actualToken = 6

```text
actualToken   = 6
cubeToken     = 6
cubeHalfToken = 3

AIV0 validToken = 3
AIV1 validToken = 3
```

无 padding 行，AIV-local capacity 与 valid rows 一致。

## 8. 与当前实现的关系

当前已确认的目标语义是：

```text
FIXP uses dualDstCtl = 1
FIXP splits by M axis
odd token is aligned only inside Cube/FIXP layout
AIV postprocess consumes only actual rows
```

需要注意的实现风险：

1. 当前实现已使用 `dualDstCtl = 1`、`subBlockId = false`，符合本文的 M 轴双目标切分语义。
2. AIV slice 已经前置到 `TileParam::aivBlockTokenOffset/aivTokenOffset/aivTokenSize`，Vec 不再维护私有 `MakeAivSlice`。
3. Q/K 后处理的本地 K base 直接使用 `tile.cubeHalfTokenSize * qHeadNum`，当前代码不再保留 `QkLocalKBaseRow` 这种一行 helper。
4. 有效 rows 使用 `tile.aivTokenSize`，不能扩大到 `cubeHalfToken`；padding 行不量化、不写回。
5. 当前代码的 L0A 策略是按单个 Q/K segment 判断 full-D 是否能放下，放不下时动态计算 D 方向 tile，不恢复旧的 Q/K 同驻 L0A 约束。
