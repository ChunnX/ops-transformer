# QkvRmsNormRopeCacheWithKScale：Cache Stride 与写回地址

## 1. Cache 布局和各维 stride

K/V Cache 的逻辑 shape 是：

```text
k_cache: [BlockNum,Nk,BlockSize,D]
v_cache: [BlockNum,Nv,BlockSize,D]
```

各维含义为：

```text
dim 0：Cache block
dim 1：K/V head
dim 2：block 内的 token slot
dim 3：head 内的 D 维
```

对于连续的 `cache[BlockNum,HeadNum,BlockSize,D]`，从最后一维向前计算：

```text
strideD     = 1
strideToken = D
strideHead  = BlockSize × D
strideBlock = HeadNum × BlockSize × D
```

所以完整 stride 是：

```text
[strideBlock,strideHead,strideToken,strideD]
=
[HeadNum×BlockSize×D, BlockSize×D, D, 1]
```

例如：

```text
v_cache.shape = [4,2,16,128]
```

对应：

```text
v_cache.stride = [4096,2048,128,1]

kvCacheStrideBlock = 4096
kvCacheStrideHead  = 2048
kvCacheStrideToken = 128
```

代码实际直接读取 Cache Tensor 的 stride：

```cpp
kvCacheStrideBlock = k_cache.stride[0];
kvCacheStrideHead  = k_cache.stride[1];
kvCacheStrideToken = k_cache.stride[2];
```

因此连续 Cache 中：

```text
kvCacheStrideHead = BlockSize × D
```

带 padding 的非连续 Cache 则使用 Tensor 自身的实际 stride，不一定等于上述乘法结果。

## 2. `cache_offset[token]` 如何计算

当前输入 token 的逻辑 Cache slot 为：

```text
slot = slot_mapping[token]
```

通过 `BlockSize` 拆成：

```text
block_id     = slot // BlockSize
block_offset = slot % BlockSize
```

目标 Cache 基地址为：

```text
cache_offset[token]
    = block_id     × kvCacheStrideBlock
    + block_offset × kvCacheStrideToken
```

它指向：

```text
&v_cache[block_id,0,block_offset,0]
```

即 `BlockNum` 维和 `BlockSize` 维已经确定，head 维和 D 维留给后续二维搬运。

对应代码：

```cpp
const uint64_t slot = slotMapping.GetValue(tokenIdx);
const uint64_t blockId = slot / blockSize_;
const uint64_t blockOffset = slot % blockSize_;

tile.cacheBaseOffset[tokenIdx] =
    blockId * kvCacheStrideBlock_
    + blockOffset * kvCacheStrideToken_;
```

## 3. V Cache 二维写回中的 stride

V 的 UB 临时输出布局是：

```text
v_out_ub[Nv,tileTokenSize,D]
```

写回伪代码为：

```text
for token in 0 .. tileTokenSize-1:
    DataCopyUbToGm2D(
        dst = v_cache + cache_offset[token],
        src = v_out_ub + token * D,
        blockCount = Nv,
        blockLen = D个FP8,
        srcStride = tileTokenSize * D,
        dstStride = kvCacheStrideHead
    )
```

固定一个 token 后，实际搬运的是：

```text
v_out_ub[:,token,:]
    → v_cache[block_id,:,block_offset,:]

shape：[Nv,D] → [Nv,D]
```

所以：

```text
blockCount = Nv
    二维搬运共有Nv行，每行对应一个head

blockLen = D
    每个head连续搬运D个FP8

srcStride = tileTokenSize × D
    UB中从当前head的token位置跳到下一个head的相同token位置

dstStride = kvCacheStrideHead
    GM中从当前head的Cache slot跳到下一个head的相同Cache slot
```

将二维搬运展开后等价于：

```text
for head in 0 .. Nv-1:
    src =
        v_out_ub
        + token × D
        + head × tileTokenSize × D

    dst =
        v_cache
        + cache_offset[token]
        + head × kvCacheStrideHead

    copy dst[0:D] <- src[0:D]
```

外层 `token` 循环通过 `cache_offset[token]` 固定 `block_id` 和 `block_offset`；单次二维搬运沿 head 轴
搬 `Nv` 行，不是沿 BlockSize 轴搬运。

## 4. 数值例子

假设：

```text
v_cache.shape = [4,2,16,128]
tileTokenSize = 4
token = 1
slot_mapping[1] = 35
```

连续 Cache stride：

```text
kvCacheStrideToken = D = 128
kvCacheStrideHead  = BlockSize × D = 16 × 128 = 2048
kvCacheStrideBlock = Nv × BlockSize × D = 2 × 16 × 128 = 4096
```

先拆分 slot：

```text
block_id     = 35 // 16 = 2
block_offset = 35 % 16  = 3
```

计算目标基地址：

```text
cache_offset[1]
    = 2 × 4096 + 3 × 128
    = 8576
```

计算搬运 stride：

```text
srcStride = tileTokenSize × D = 4 × 128 = 512
dstStride = kvCacheStrideHead = 2048
```

两行搬运分别为：

```text
head 0：
    src = 1 × 128 + 0 × 512 = 128
    dst = 8576 + 0 × 2048   = 8576

    v_out_ub[0,1,0:128]
        → v_cache[2,0,3,0:128]

head 1：
    src = 1 × 128 + 1 × 512 = 640
    dst = 8576 + 1 × 2048   = 10624

    v_out_ub[1,1,0:128]
        → v_cache[2,1,3,0:128]
```

## 5. 总结公式

```text
连续Cache：
    strideToken = D
    strideHead  = BlockSize × D
    strideBlock = HeadNum × BlockSize × D

目标slot：
    block_id     = slot_mapping[token] // BlockSize
    block_offset = slot_mapping[token] % BlockSize

目标基地址：
    cache_offset[token]
        = block_id × strideBlock
        + block_offset × strideToken

V二维写回：
    blockCount = Nv
    blockLen   = D
    srcStride  = tileTokenSize × D
    dstStride  = strideHead
```
