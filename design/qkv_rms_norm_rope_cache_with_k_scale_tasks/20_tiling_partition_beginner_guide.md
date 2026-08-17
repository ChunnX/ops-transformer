# QkvRmsNormRopeCacheWithKScale Tiling 切分

## 1. 总体切分

当前实现只切 Token 轴 `T`：

```text
全局T
  -> 多个AIC
      -> 每个AIC内的多个token tile
          -> 每个tile分给两个AIV
```

`Nq`、`Nk`、`Nv` 和 `D=128` 不切分。一个 tile 包含其 Token 范围内的全部 Q/K/V heads。

主要变量：

| 变量 | 表示的内容 |
|---|---|
| `coreTokenTile` | 一个 AIC 最多负责的 Token 数 |
| `tokenTile` | 一个 AIC 每次循环最多处理的 Token 数 |
| `tokenSize` | 当前 tile 的实际 Token 数 |
| `aivTokenSize` | 单个 AIV 实际处理的 Token 数 |

## 2. 全局 T 分给多个 AIC

Host 计算：

```text
coreTokenTile = ceil(totalTokens / aicNum)
coreGroupNum  = ceil(totalTokens / coreTokenTile)
numBlocks     = coreGroupNum
```

第 `cubeIndex` 个 AIC 负责：

```text
coreBegin = cubeIndex * coreTokenTile
coreEnd   = min(totalTokens, coreBegin + coreTokenTile)

范围 = [coreBegin, coreEnd)
```

AIV 通过下面的计算找到所属 AIC：

```text
cubeIndex = GetBlockIdx() / 2
```

## 3. 确定每个 AIC 内的 `tokenTile`

### 3.1 计算初始值

```text
Nqk  = Nq + Nk
Nqkv = Nq + Nk + Nv

inputRowLimit = 160    # RoPE
inputRowLimit = 80     # M-RoPE

tokenTilePerAiv = min(
    8,
    floor(128 / Nqk),
    floor(inputRowLimit / Nqkv),
    floor(80 / Nv),
    4
)
```

最后一个 `4` 来自当前代码中的：

```text
TOKEN_TILE_PER_AIV_CAP = 4
```

一个 AIC 对应两个 AIV，因此候选值为：

```text
candidateTokenTile = 2 * tokenTilePerAiv
```

如果初始 `tokenTilePerAiv=4`，依次检查：

```text
tokenTile = 8 -> 6 -> 4 -> 2
```

返回第一个通过检查的值；全部未通过时返回 `tokenTile=1`。

### 3.2 UB 检查

对于一个候选 `tokenTile`：

```text
p     = ceil(tokenTile / 2)
qRows = p * Nq
kRows = p * Nk
```

需要满足：

```text
p * (Nq + Nk)      <= 128
p * (Nq + Nk + Nv) <= 160    # RoPE
p * (Nq + Nk + Nv) <= 80     # M-RoPE
p * Nv             <= 80
Q/K预处理NZ字节数   <= 64 KiB
```

Q/K 预处理 NZ 字节数为：

```text
S(rows)     = AlignUp(rows - 1, 16) + 1
Bytes(rows) = (7 * S(rows) + rows) * 32

qkPreprocessBytes = Bytes(qRows) + Bytes(kRows)
```

### 3.3 L0C 检查

```text
rowTile        = tokenTile * (Nq + Nk)
rowTileAligned = AlignUp(rowTile, 16)

qkAlignedRows = AlignUp(tokenTile * Nq, 16)
              + AlignUp(tokenTile * Nk, 16)
```

需要满足：

```text
rowTile        <= 256
rowTileAligned <= 256
qkAlignedRows  <= 256
```

## 4. 每个 AIC 内继续分 tile

```text
tokenOffset = coreBegin

while tokenOffset < coreEnd:
    tokenSize = min(tokenTile, coreEnd - tokenOffset)
    ComputeTile(tokenOffset, tokenSize)
    tokenOffset += tokenSize
```

最后一个 tile 的 `tokenSize` 可以小于 `tokenTile`。

## 5. 一个 tile 分给两个 AIV

先把 `tokenSize` 向 2 对齐：

```text
cubeTokenSize     = AlignUp(tokenSize, 2)
cubeHalfTokenSize = cubeTokenSize / 2
```

两个 AIV 的切分为：

```text
AIV0:
    tokenOffset = tile.tokenOffset
    tokenSize   = cubeHalfTokenSize

AIV1:
    tokenOffset = tile.tokenOffset + cubeHalfTokenSize
    tokenSize   = tile.tokenSize - cubeHalfTokenSize
```

| tile 的 `tokenSize` | AIV0 | AIV1 |
|---:|---:|---:|
| 8 | 4 | 4 |
| 7 | 4 | 3 |
| 5 | 3 | 2 |
| 2 | 1 | 1 |
| 1 | 1 | 0 |

## 6. 示例

### 6.1 选择 `tokenTile`

输入：

```text
Nq=24, Nk=3, Nv=3
```

RoPE：

```text
tokenTilePerAiv = min(8, floor(128/27), floor(160/30), floor(80/3), 4)
                = min(8, 4, 5, 26, 4)
                = 4

tokenTile = 2 * 4 = 8
```

M-RoPE：

```text
tokenTilePerAiv = min(8, floor(128/27), floor(80/30), floor(80/3), 4)
                = min(8, 4, 2, 26, 4)
                = 2

tokenTile = 2 * 2 = 4
```

以上候选值还要通过第 3.2、3.3 节的 UB 和 L0C 检查。

### 6.2 分核和尾块

输入：

```text
T=100, aicNum=8, tokenTile=8
```

分核：

```text
coreTokenTile = ceil(100 / 8) = 13
coreGroupNum  = ceil(100 / 13) = 8

AIC0: [ 0, 13)
AIC1: [13, 26)
AIC2: [26, 39)
AIC3: [39, 52)
AIC4: [52, 65)
AIC5: [65, 78)
AIC6: [78, 91)
AIC7: [91,100)
```

AIC0 内部：

```text
tile0: [0, 8)   -> AIV0=4, AIV1=4
tile1: [8,13)   -> AIV0=3, AIV1=2
```

AIC7 内部：

```text
tile0: [91,99)  -> AIV0=4, AIV1=4
tile1: [99,100) -> AIV0=1, AIV1=0
```

## 7. 源码位置

- `tokenTile` 搜索与资源检查：
  `op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.cpp`
  中的 `SelectTokenTile()` 和 `TrySelectTokenTile()`。
- `coreTokenTile/coreGroupNum`：同一文件中的 `FillTilingData()`。
- 资源常量：
  `op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.h`。
- 分核、核内 tile 循环和双 AIV 切分：
  `op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_controller.h`。

以上路径均相对于：

```text
posembedding/qkv_rms_norm_rope_cache_with_k_scale/
```
