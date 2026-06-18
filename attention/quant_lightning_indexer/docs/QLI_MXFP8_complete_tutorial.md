# QLI 算子 MXFP8 完整教学 —— Q/K/QScale/KScale 的传输与排布

> 本文档面向需要深入理解 QLI 算子 MXFP8 实现的开发者。
> 从数据在 GM（全局内存）中的排布开始，逐层追踪到 L1、L0、Mmad，完整呈现四种数据的全链路传输过程。

---

## 目录

- [一、MXFP8 是什么？为什么需要它？](#一mxfp8-是什么为什么需要它)
- [二、数据类型基础](#二数据类型基础)
- [三、GM 中的排布 —— 数据在显存中长什么样？](#三gm-中的排布--数据在显存中长什么样)
- [四、数据传输全链路总览](#四数据传输全链路总览)
- [五、Q 的传输：GM → L1 → L0 → Mmad](#五q-的传输gm--l1--l0--mmad)
- [六、K 的传输：GM → L1 → L0 → Mmad](#六k-的传输gm--l1--l0--mmad)
- [七、QScale 的传输：GM → L1 → L0（与 Q 同行）](#七qscale-的传输gm--l1--l0与-q-同行)
- [八、KScale 的传输：GM → L1 → L0（与 K 同行）](#八kscale-的传输gm--l1--l0与-k-同行)
- [九、L1 Buffer 布局详解](#九l1-buffer-布局详解)
- [十、L0 加载 —— LoadData Mx 变体](#十l0-加载--loaddata-mx-变体)
- [十一、mmad_mx 矩阵乘 —— 硬件融合 Scale](#十一mmad_mx-矩阵乘--硬件融合-scale)
- [十二、Vector 核后处理 —— 简化流程](#十二vector-核后处理--简化流程)
- [十三、完整数据流时序图](#十三完整数据流时序图)
- [十四、代码定位速查](#十四代码定位速查)

---

## 一、MXFP8 是什么？为什么需要它？

### 1.1 一句话总结

MXFP8（Microscaling FP8）是一种 **per-block 细粒度量化** 方案：每 32 个 fp8_e4m3 数据元素共享一个 fp8_e8m0 scale，通过 `mmad_mx` 硬件指令在矩阵乘法中直接融合 scale，避免了软件层面的 scale 乘除运算。

### 1.2 与普通 FP8 的核心差异

| 对比项 | 普通 FP8（per-token scale） | MXFP8（per-block scale） |
|--------|---------------------------|------------------------|
| Scale 类型 | `float` / `float16` | `fp8_e8m0_t`（8位纯指数） |
| Scale 粒度 | 每个 token 1 个标量 | 每 32 个 D 维度元素 1 个 scale |
| Scale 形状 | `[B, S, N]`（比 data 少一维） | `[B, S, N, D/32]`（与 data 同维数） |
| Scale 处理位置 | **Vector 核**（UB 中逐元素乘） | **Cube 核**（mmad_mx 硬件融合） |
| 矩阵乘指令 | 普通 `Mmad` | `Mmad` with `mx_fp8_e4m3_t` |

### 1.3 为什么 Scale 要从 Vector 核迁移到 Cube 核？

老流程中 qScale 形状为 `[B, S1, N1]`，每个 token 只有 **1 个标量**，可以与 `weights[G]` 做 1:1 的 element-wise 乘法。

MXFP8 中 qScale 形状变为 `[B, S1, N1, D/32]`，每个 token 有 **D/32 = 4 个 scale**（D=128），轴数增多，无法再与 `weights[G]` 做简单的 element-wise 乘法。解决方案是利用 `mmad_mx` 硬件指令，在矩阵乘法阶段直接融合 scale。

---

## 二、数据类型基础

### 2.1 fp8_e4m3fn_t —— 数据

```
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ S │ E │ E │ E │ E │ M │ M │ M │    8 bits
└───┴───┴───┴───┴───┴───┴───┴───┘
 符号  ─── 指数(4位) ───  ─ 尾数(3位)
```

- 动态范围：±448
- 精度：3 位尾数
- 用途：存储 Q 和 K 的量化数据

### 2.2 fp8_e8m0_t —— Scale

```
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ E │ E │ E │ E │ E │ E │ E │    8 bits
└───┴───┴───┴───┴───┴───┴───┴───┘
      ────── 指数(8位) ──────
```

- 无符号、无尾数，纯指数：值 = 2^(E-127)
- 用途：表示每 32 个 fp8_e4m3 元素的缩放因子

### 2.3 打包规则

**关键**：2 个 `fp8_e8m0_t` 打包成 1 个 `bfloat16_t` 进行搬运。

```
bf16 (16 bits):
┌───────────────┬───────────────┐
│ fp8_e8m0 [1]  │ fp8_e8m0 [0]  │
│  (高 8 位)     │  (低 8 位)     │
└───────────────┴───────────────┘
```

代码中通过 `ReinterpretCast<bfloat16_t>()` 实现类型转换，搬运时以 bf16 为单位。

```cpp
// 常量定义
constexpr uint32_t MXFP8_BLOCK_SIZE = 32;  // 每32个元素一个scale
constexpr uint32_t FP8_TWO = 2;            // 2个fp8_e8m0打包成1个bf16
```

---

## 三、GM 中的排布 —— 数据在显存中长什么样？

### 3.1 四种数据的 GM Shape 对比

以 BSND 布局为例，设 B=2, S1=64, S2=512, N1=16, N2=1, D=128, G=N1/N2=16：

| 数据 | Shape（非 MXFP8） | Shape（MXFP8） | 元素类型 | 总元素数（MXFP8） |
|------|-------------------|----------------|---------|-----------------|
| **Q** | `[B, S1, N1, D]` = `[2, 64, 16, 128]` | 同左 | `fp8_e4m3fn_t` | 262144 |
| **K** | `[B, S2, N2, D]` = `[2, 512, 1, 128]` | 同左 | `fp8_e4m3fn_t` | 131072 |
| **QScale** | `[B, S1, N1]` = `[2, 64, 16]` | `[B, S1, N1, D/32]` = `[2, 64, 16, 4]` | `fp8_e8m0_t` | 8192 |
| **KScale** | `[B, S2, N2]` = `[2, 512, 1]` | `[B, S2, N2, D/32]` = `[2, 512, 1, 4]` | `fp8_e8m0_t` | 4096 |

### 3.2 GM 内存排布示意

#### Q 数据 `[B, S1, N1, D]`

```
GM 地址增长方向 →

Batch 0:
  S1=0: Head0: [d0, d1, ..., d127]  Head1: [d0, d1, ..., d127]  ...  Head15: [...]
  S1=1: Head0: [d0, d1, ..., d127]  Head1: [d0, d1, ..., d127]  ...  Head15: [...]
  ...
Batch 1:
  ...
```

每个元素 1 字节（fp8_e4m3），每个 head 的 D 维度连续存储 128 字节。

#### QScale `[B, S1, N1, D/32]`

```
GM 地址增长方向 →

Batch 0:
  S1=0: Head0: [s0, s1, s2, s3]  Head1: [s0, s1, s2, s3]  ...  Head15: [s0, s1, s2, s3]
  S1=1: Head0: [s0, s1, s2, s3]  Head1: [s0, s1, s2, s3]  ...  Head15: [s0, s1, s2, s3]
  ...
```

每个元素 1 字节（fp8_e8m0），每个 head 有 D/32=4 个 scale，连续存储 4 字节。

**Scale 与 Data 的对应关系**：

```
Q Data:   [d0, d1, ..., d31 | d32, d33, ..., d63 | d64, ..., d95 | d96, ..., d127]
             ↕ scale[0]          ↕ scale[1]           ↕ scale[2]      ↕ scale[3]
QScale:   [      s0       ,        s1           ,       s2       ,       s3       ]
```

每 32 个 D 维度元素共享一个 scale。

#### K 和 KScale

K 的排布与 Q 类似，只是 N2=1（单 head），S2 可能远大于 S1。

KScale 的排布与 QScale 类似，shape 为 `[B, S2, N2, D/32]`。

### 3.3 PageAttention 场景下的 K 和 KScale

PA 场景下 K 和 KScale 的 shape 变为 `[block_num, block_size, N2, D]` 和 `[block_num, block_size, N2, D/32]`，物理上按 block 分散存储，通过 `block_table` 映射逻辑位置到物理位置。

```
block_table[B, maxBlockNumPerSeq]:
  Batch 0: [block_id_0, block_id_1, ..., block_id_31]
  Batch 1: [block_id_0, block_id_1, ..., block_id_31]

逻辑 S2 位置 → 物理 block 位置:
  s2_idx → block_id = block_table[bIdx][s2_idx / block_size]
           offset_in_block = s2_idx % block_size
           gm_addr = block_id * keyStride0 + offset_in_block * headDim
```

---

## 四、数据传输全链路总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GM（全局内存 / HBM）                                 │
│                                                                             │
│  Q [S1×G, D=128]     K [S2, D=128]     qScale [S1×G, D/32=4]   kScale [S2, D/32=4]  │
│  fp8_e4m3            fp8_e4m3          fp8_e8m0                fp8_e8m0     │
└───────┬──────────────┬──────────────────┬─────────────────────┬─────────────┘
        │              │                  │                     │
        │ Nd2Nz        │ Nd2Nz            │ Dn2Nz               │ Dn2Nz
        │ (MTE2)       │ (MTE2)           │ (MTE2, 以bf16搬运)   │ (MTE2, 以bf16搬运)
        ▼              ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         L1（片上缓存）                                       │
│                                                                             │
│  bufQL1_ (A1, 64KB)           bufKeyL1_ (B1, 64KB)                          │
│  ┌────────────────────┐       ┌──────────────────────────────┐              │
│  │ Q 数据 (NZ格式)     │       │ K 数据 (NZ格式)               │              │
│  │ [M×D] fp8_e4m3     │       │ [S2×D] fp8_e4m3             │              │
│  ├────────────────────┤       ├──────────────────────────────┤              │
│  │ qScale (NZ格式)     │       │ kScale (NZ格式)              │              │
│  │ [M×D/32] fp8_e8m0  │       │ [S2×D/32] fp8_e8m0          │              │
│  │ (紧跟Q数据之后)      │       │ (所有ping-pong buffer之后)    │              │
│  └────────────────────┘       └──────────────────────────────┘              │
└───────┬──────────────────────────────┬──────────────────────────────────────┘
        │ LoadData Mx                  │ LoadData Mx
        │ (Q数据 + qScale 一起加载)     │ (K数据 + kScale 一起加载)
        │ (MTE1)                       │ (MTE1)
        ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         L0（矩阵乘输入缓存）                                  │
│                                                                             │
│  bufQL0_ (A2)                bufKeyL0_ (B2)                                 │
│  ┌────────────────────┐      ┌────────────────────┐                         │
│  │ mx_fp8_e4m3_t      │      │ mx_fp8_e4m3_t      │                         │
│  │ (数据+scale 打包)    │      │ (数据+scale 打包)    │                         │
│  └────────────────────┘      └────────────────────┘                         │
└───────┬──────────────────────────────┬──────────────────────────────────────┘
        │                              │
        └──────────┬───────────────────┘
                   │ Mmad (mmad_mx 硬件指令)
                   │ Q × K × qScale × kScale → float
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         L0C (CO1) → Fixpipe → UB (VECCALC)                  │
│  QK 结果 [S1G, S2] float（已含 qScale × kScale）                             │
└───────┬─────────────────────────────────────────────────────────────────────┘
        │ CrossCore 同步 (C→V)
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Vector 核 UB                                        │
│  Relu(QK) × weights → ReduceSum → Cast bf16 → SortableKey → score (uint16) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 五、Q 的传输：GM → L1 → L0 → Mmad

### 5.1 GM → L1：QueryNd2Nz

Q 数据从 GM 搬到 L1，进行 ND → NZ 格式转换。

```cpp
// service_cube.h: QueryNd2Nz()
Nd2NzParams nd2nzPara;
nd2nzPara.nValue = s1gL1RealSize;     // 行数 = 当前块的 S1×G 大小（最大 M_BASIC_BLOCK=256）
nd2nzPara.dValue = headDim;           // 列数 = 128
nd2nzPara.srcDValue = headDim;        // 源 stride = 128
nd2nzPara.dstNzC0Stride = CeilAlign(s1gL1RealSize, BLOCK_CUBE);  // 对齐到16

DataCopy(queryL1_[(queryL1Mte2BufIdx_ % 2) * QUERY_BUFFER_OFFSET],
         queryGm_[runInfo.tensorQueryOffset + s1gGmOffset * headDim], nd2nzPara);
```

**NZ 格式**：将 ND 矩阵按 16×16 的小块重排，适合 Cube 单元访问。

```
ND 格式 [M, D=128]:          NZ 格式:
┌──────────────────┐         ┌──────┬──────┬──────┬──────┐
│ row0: d0..d127   │         │16×16 │16×16 │16×16 │16×16 │  ← 8 个 16×16 块
│ row1: d0..d127   │   →     │ blk0 │ blk1 │ blk2 │ blk3 │     (D=128 / 16 = 8)
│ ...              │         ├──────┼──────┼──────┼──────┤
│ rowM: d0..d127   │         │ blk4 │ blk5 │ blk6 │ blk7 │
└──────────────────┘         └──────┴──────┴──────┴──────┘
```

**L1 Buffer 大小**：

```
QUERY_BUFFER_OFFSET = M_BASIC_BLOCK × D_BASIC_BLOCK = 256 × 128 = 32768 元素
                     = 32KB（fp8 为 1 字节/元素）
```

Q 使用 **2 个 ping-pong buffer**（`QUERY_BUF_NUM = 2`），总大小：
- 非 MXFP8：`2 × 32KB = 64KB`
- MXFP8：`2 × (32KB + 128B) = 64KB + 256B`（额外的 128B 用于存储 qScale）

### 5.2 L1 → L0：LoadQueryToL0a

从 L1 加载到 L0，MXFP8 场景使用 `LoadData` 的 Mx 变体，**同时加载数据和 scale**。

```cpp
// MXFP8 分支
LocalTensor<mx_fp8_e4m3_t> dstTensor =
    queryL0_[(l0BufIdx_ % 2) * L0AB_BUFFER_OFFSET]
        .template ReinterpretCast<mx_fp8_e4m3_t>();

// 数据加载参数
LoadData2DParamsV2 loadDataA2DParams;
loadDataA2DParams.mStep = CeilDiv(s1gL0RealSize, 16);    // M 方向分型
loadDataA2DParams.kStep = CeilDiv(128, 32);              // K 方向 = 4（128/32）

// Scale 加载参数
LoadData2DMxParams loadAScaleParam;
loadAScaleParam.xStep = CeilDiv(s1gL0RealSize, 16);     // 与数据 M 方向同步
loadAScaleParam.yStep = CeilDiv(128, 32 * 2);            // = 2（D / (32*FP8_TWO) = 128/64）

LoadData(dstTensor,
         queryL1_[(queryL1Mte1BufIdx_ % 2) * QUERY_BUFFER_OFFSET],  // Q 数据
         scaleL1,                                                     // qScale
         loadDataA2DParams, loadAScaleParam);
```

**L0 中的 mx_fp8_e4m3_t 格式**：

```
mx_fp8_e4m3_t 是硬件层面的打包格式：
- 数据部分：fp8_e4m3 元素
- Scale 部分：fp8_e8m0 scale（每32个元素一个）
两者在 L0 中由硬件统一管理。
```

---

## 六、K 的传输：GM → L1 → L0 → Mmad

### 6.1 GM → L1：KeyNd2Nz / KeyNd2NzForPA

K 数据从 GM 搬到 L1，同样进行 ND → NZ 格式转换。

**非 PA 场景**：

```cpp
// service_cube.h: KeyNd2Nz()
Nd2NzParams nd2nzPara;
nd2nzPara.nValue = s2L1RealSize;      // 行数 = 当前块的 S2 大小（最大 s2BasicBlock_=128）
nd2nzPara.dValue = headDim;           // 列数 = 128
nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, BLOCK_CUBE);

DataCopy(keyL1_[(keyL1BufIdx_ % 3) * keyBufferOffset_],
         keyGm_[runInfo.tensorKeyOffset + s2GmOffset * headDim], nd2nzPara);
```

**PA 场景**（KeyNd2NzForPA）：需要通过 block_table 映射，可能需要多次 DataCopy 拼接跨 block 的数据。

```cpp
// 循环处理跨 block 的情况
while (s2L1Offset < s2L1RealSize) {
    uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / kCacheBlockSize;
    uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % kCacheBlockSize;
    uint64_t keyGmOffset = blkTableGm_.GetValue(bIdx * maxBlockNumPerBatch + s2BlkId)
                            * keyStride0 + s2BlkOffset * headDim;
    // 计算当前 block 内可拷贝的大小
    s2Mte2Size = min(s2L1RealSize - s2L1Offset, kCacheBlockSize - s2BlkOffset);
    DataCopy(keyL1_[...], keyGm_[keyGmOffset], nd2nzPara);
    s2L1Offset += s2Mte2Size;
}
```

**L1 Buffer 大小**：

```
keyBufferOffset_ = s2BasicBlock_ × D_BASIC_BLOCK = 128 × 128 = 16384 元素
                  = 16KB（fp8 为 1 字节/元素）
```

K 使用 **3 个 ping-pong buffer**（`KEY_BUF_NUM = 3`），总大小：
- 非 MXFP8：`3 × 16KB = 48KB`
- MXFP8：`3 × 16KB + 128B = 48KB + 128B`（额外的 128B 在所有 buffer 之后，用于存储 kScale）

### 6.2 L1 → L0：LoadKeyToL0b

与 Q 类似，MXFP8 场景使用 `LoadData` Mx 变体：

```cpp
// MXFP8 分支
LocalTensor<mx_fp8_e4m3_t> dstTensor =
    keyL0_[(l0BufIdx_ % 2) * L0AB_BUFFER_OFFSET]
        .template ReinterpretCast<mx_fp8_e4m3_t>();

// kScale 存储在 keyL1_ buffer 尾部（所有 ping-pong buffer 之后）
uint64_t scaleOffsetInBuf = KEY_BUF_NUM * s2BasicBlock_ * D_BASIC_BLOCK;
                           // = 3 × 128 × 128 = 49152 元素偏移
LocalTensor<fp8_e8m0_t> scaleL1 =
    keyL1_[scaleOffsetInBuf].template ReinterpretCast<fp8_e8m0_t>();

LoadData(dstTensor,
         keyL1_[(keyL1BufIdx_ % 3) * keyBufferOffset_],  // K 数据
         scaleL1,                                          // kScale
         loadData2DV2, loadDataMxParams);
```

---

## 七、QScale 的传输：GM → L1 → L0（与 Q 同行）

### 7.1 GM 中的排布

qScale shape: `[B, S1, N1, D/32]`

在 QLI 算子中，Q 的 N1 维度被展开为 `G × N2`，其中 N2=1。因此 qScale 实际上按 `(S1×G)` 维度组织：

```
qScale 在 GM 中的连续排布（以单个 batch 为例）：

S1=0, Head0: [s0, s1, s2, s3]    ← 4 个 fp8_e8m0 = 4 字节
S1=0, Head1: [s0, s1, s2, s3]
...
S1=0, Head15: [s0, s1, s2, s3]
S1=1, Head0: [s0, s1, s2, s3]
...
```

每个 token（每个 head）有 D/32 = 4 个 scale 值。

### 7.2 GM → L1：LoadQScaleToL1

qScale 通过 `Dn2NzParams` 进行 DN → NZ 格式转换，**以 bf16 为单位搬运**（2个 fp8_e8m0 打包成 1 个 bf16）。

```cpp
// service_cube.h: LoadQScaleToL1()

// 1. 确定 L1 中的存储位置：紧跟在当前 ping-pong buffer 的 Q 数据之后
uint64_t bufStride = M_BASIC_BLOCK * D_BASIC_BLOCK + D_BASIC_BLOCK;
                    // = 256*128 + 128 = 32896 元素/buffer
uint64_t scaleOffsetInBuf = (queryL1Mte2BufIdx_ % 2) * bufStride
                            + s1gL1RealSize * D_BASIC_BLOCK;
                            // Q 数据实际大小之后

// 2. 将 fp8_e8m0 指针重解释为 bf16 进行搬运
LocalTensor<bfloat16_t> scaleL1 = queryL1_[scaleOffsetInBuf].template ReinterpretCast<bfloat16_t>();
GlobalTensor<bfloat16_t> scaleGmCast;
scaleGmCast.SetGlobalBuffer((__gm__ bfloat16_t *)(qScaleGm_.GetPhyAddr()));

// 3. 设置 DN2NZ 参数
uint32_t scalePerToken = headDim / MXFP8_BLOCK_SIZE;  // = 128/32 = 4
Dn2NzParams dn2Nzparam;
dn2Nzparam.nValue = scalePerToken / FP8_TWO;  // = 4/2 = 2（bf16 列数）
dn2Nzparam.dValue = s1gL1RealSize;             // 行数 = token 数
dn2Nzparam.srcDValue = scalePerToken / FP8_TWO; // = 2
dn2Nzparam.dstNzC0Stride = scalePerToken / FP8_TWO; // = 2
dn2Nzparam.dstNzNStride = 1;
dn2Nzparam.dstNzMatrixStride = scalePerToken / FP8_TWO; // = 2

// 4. 计算 GM 偏移并搬运
uint64_t gmOffset = runInfo.tensorQScaleOffset + s1gGmOffset * scalePerToken;
DataCopy(scaleL1, scaleGmCast[gmOffset / FP8_TWO], dn2Nzparam);
```

**DN2NZ 转换示意**：

```
DN 格式（GM 中）:                    NZ 格式（L1 中）:
每个 token 有 2 个 bf16              按 16×16 块重排
（= 4 个 fp8_e8m0）

┌──────────┬──────────┐             ┌──────────────────┐
│ bf16_0   │ bf16_1   │  token 0    │                  │
│ (s0,s1)  │ (s2,s3)  │             │  NZ 16×16 块     │
├──────────┼──────────┤             │  (2列 × N行)     │
│ bf16_0   │ bf16_1   │  token 1    │                  │
│ (s0,s1)  │ (s2,s3)  │             └──────────────────┘
├──────────┼──────────┤
│  ...     │  ...     │
└──────────┴──────────┘
```

### 7.3 L1 中的存储位置

qScale 存储在 **当前 queryL1_ ping-pong buffer 的 Q 数据之后**：

```
queryL1_ buffer (每个 ping-pong):
┌─────────────────────────────────────────┬────────────────┐
│ Q 数据 (NZ 格式)                         │ qScale (NZ)    │
│ s1gL1RealSize × 128 个 fp8_e4m3        │ s1gL1RealSize  │
│ = s1gL1RealSize × 128 字节             │ × 4 字节        │
└─────────────────────────────────────────┴────────────────┘
←──── QUERY_BUFFER_OFFSET (32768) ────→   ←── D_BASIC_BLOCK (128) ──→
```

**为什么 qScale 跟随 Q 的 ping-pong buffer？**

因为 qScale 和 Q 数据在后续的 L0 加载阶段需要同时被 `LoadData` Mx 变体读取。将它们放在同一个 L1 buffer 中，可以确保数据一致性——当 Q 的 ping-pong buffer 切换时，对应的 qScale 也自动切换。

### 7.4 L1 → L0：与 Q 数据一起加载

在 `LoadQueryToL0a` 中，qScale 的 L1 地址通过以下方式计算：

```cpp
uint64_t scaleOffsetInBuf = (queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET
                            + s1gL1RealSize * D_BASIC_BLOCK;
LocalTensor<fp8_e8m0_t> scaleL1 = queryL1_[scaleOffsetInBuf].template ReinterpretCast<fp8_e8m0_t>();
```

注意这里使用 `queryL1Mte1BufIdx_`（MTE1 阶段的 buffer 索引），而不是 MTE2 阶段的索引，因为 L1→L0 是 MTE1 操作。

---

## 八、KScale 的传输：GM → L1 → L0（与 K 同行）

### 8.1 GM 中的排布

kScale shape: `[B, S2, N2, D/32]`

由于 N2=1，kScale 实际上按 S2 维度组织：

```
kScale 在 GM 中的连续排布（以单个 batch 为例）：

S2=0: [s0, s1, s2, s3]    ← 4 个 fp8_e8m0 = 4 字节
S2=1: [s0, s1, s2, s3]
...
S2=511: [s0, s1, s2, s3]
```

### 8.2 GM → L1：LoadKScaleToL1

与 qScale 类似，通过 `Dn2NzParams` 以 bf16 为单位搬运。

```cpp
// service_cube.h: LoadKScaleToL1()

// 1. 确定 L1 中的存储位置：keyL1_ buffer 的尾部（所有 ping-pong buffer 之后）
uint64_t scaleOffsetInBuf = KEY_BUF_NUM * s2BasicBlock_ * D_BASIC_BLOCK;
                           // = 3 × 128 × 128 = 49152 元素偏移

// 2-4. 搬运逻辑与 qScale 相同
uint64_t gmOffset = runInfo.tensorKeyScaleOffset + s2GmOffset * scalePerToken;
DataCopy(scaleL1, scaleGmCast[gmOffset / FP8_TWO], dn2Nzparam);
```

### 8.3 L1 中的存储位置

kScale 存储在 **keyL1_ buffer 的尾部区域**，在所有 3 个 ping-pong buffer 之后：

```
keyL1_ buffer:
┌──────────────────┬──────────────────┬──────────────────┬────────────┐
│ K buf 0 (NZ)     │ K buf 1 (NZ)     │ K buf 2 (NZ)     │ kScale(NZ) │
│ 128×128 fp8_e4m3 │ 128×128 fp8_e4m3 │ 128×128 fp8_e4m3 │ 128×4 fp8  │
│ = 16KB           │ = 16KB           │ = 16KB           │ = 128B     │
└──────────────────┴──────────────────┴──────────────────┴────────────┘
←─ keyBufferOffset ─→
```

**为什么 kScale 放在所有 buffer 之后，而不是跟随某个 ping-pong buffer？**

因为 K 数据使用 3 个 ping-pong buffer 做三级流水，而 kScale 只需要一份——在 S2 循环的每次迭代中，kScale 都会被重新加载覆盖。kScale 不需要 ping-pong，因为：
1. kScale 数据量很小（128 个 token × 4 字节 = 512 字节）
2. 每次 S2 迭代都会重新加载 kScale
3. 加载后立即被 L0 消费，不存在读写冲突

### 8.4 L1 → L0：与 K 数据一起加载

在 `LoadKeyToL0b` 中，kScale 的 L1 地址：

```cpp
uint64_t scaleOffsetInBuf = KEY_BUF_NUM * s2BasicBlock_ * D_BASIC_BLOCK;
                           // 固定偏移，不随 ping-pong 变化
LocalTensor<fp8_e8m0_t> scaleL1 = keyL1_[scaleOffsetInBuf].template ReinterpretCast<fp8_e8m0_t>();
```

---

## 九、L1 Buffer 布局详解

### 9.1 总体布局

```
┌─────────────────────────────────────────────────────────────────────┐
│ bufQL1_ (TPosition::A1)                                            │
│ 总大小 = QUERY_BUF_NUM × (M_BASIC_BLOCK × D_BASIC_BLOCK + extra)   │
│         = 2 × (256 × 128 + 128) = 65792 元素 = 65792 字节 ≈ 64.25KB│
│                                                                     │
│ ┌──────────────────────────────────────┬─────────────┐              │
│ │ Ping-pong 0: Q 数据                  │ qScale 0    │              │
│ │ 256 × 128 = 32768 字节              │ 128 字节     │              │
│ ├──────────────────────────────────────┼─────────────┤              │
│ │ Ping-pong 1: Q 数据                  │ qScale 1    │              │
│ │ 256 × 128 = 32768 字节              │ 128 字节     │              │
│ └──────────────────────────────────────┴─────────────┘              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ bufKeyL1_ (TPosition::B1)                                          │
│ 总大小 = KEY_BUF_NUM × s2BasicBlock × D_BASIC_BLOCK + extra         │
│         = 3 × 128 × 128 + 128 = 49280 元素 = 49280 字节 ≈ 48.1KB  │
│                                                                     │
│ ┌──────────────┬──────────────┬──────────────┬──────────┐          │
│ │ Buf 0: K 数据 │ Buf 1: K 数据 │ Buf 2: K 数据 │ kScale   │          │
│ │ 128×128      │ 128×128      │ 128×128      │ 128 字节  │          │
│ │ = 16384 字节  │ = 16384 字节  │ = 16384 字节  │          │          │
│ └──────────────┴──────────────┴──────────────┴──────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Buffer 初始化代码

```cpp
// service_cube.h: InitBuffers()

// Q L1 buffer（含 qScale 额外空间）
uint64_t qL1ExtraScale = IS_MXFP8 ? D_BASIC_BLOCK : 0;  // 128 或 0
pipe->InitBuffer(bufQL1_,
    QUERY_BUF_NUM * (M_BASIC_BLOCK * D_BASIC_BLOCK + qL1ExtraScale) * sizeof(Q_T));
// = 2 × (256*128 + 128) × 1 = 65792 字节

// K L1 buffer（含 kScale 额外空间）
uint64_t kL1ExtraScale = IS_MXFP8 ? D_BASIC_BLOCK : 0;  // 128 或 0
pipe->InitBuffer(bufKeyL1_,
    KEY_BUF_NUM * s2BasicBlock_ * D_BASIC_BLOCK * sizeof(K_T) + kL1ExtraScale * sizeof(K_T));
// = 3 × 128 × 128 × 1 + 128 × 1 = 49280 字节
```

### 9.3 偏移计算公式汇总

| 数据 | L1 偏移公式 | 说明 |
|------|-----------|------|
| Q 数据 | `(bufIdx % 2) × QUERY_BUFFER_OFFSET` | `QUERY_BUFFER_OFFSET = 256×128 = 32768` |
| qScale | `(bufIdx % 2) × bufStride + s1gL1RealSize × 128` | `bufStride = 32768 + 128 = 32896` |
| K 数据 | `(bufIdx % 3) × keyBufferOffset_` | `keyBufferOffset_ = 128×128 = 16384` |
| kScale | `3 × 128 × 128 = 49152` | 固定偏移，不随 ping-pong 变化 |

---

## 十、L0 加载 —— LoadData Mx 变体

### 10.1 普通 LoadData vs Mx LoadData

| 对比项 | 普通 LoadData | Mx LoadData |
|--------|-------------|-------------|
| 输入 | L1 数据 tensor | L1 数据 tensor + L1 scale tensor |
| 输出 L0 类型 | `fp8_e4m3fn_t` | `mx_fp8_e4m3_t`（数据+scale 打包） |
| 额外参数 | 无 | `LoadData2DMxParams` |
| 硬件行为 | 仅加载数据到 L0 | 同时加载数据和 scale 到 L0 |

### 10.2 LoadData2DMxParams 参数详解

```cpp
LoadData2DMxParams loadAScaleParam;
loadAScaleParam.xStartPosition = 0;        // scale 起始行
loadAScaleParam.yStartPosition = 0;        // scale 起始列
loadAScaleParam.xStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);  // M 方向步进（16 为单位）
loadAScaleParam.yStep = CeilDiv(headDim, FP8_BLOCK_CUBE * FP8_TWO);
                                            // K 方向步进 = 128/(32*2) = 2
loadAScaleParam.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
                                            // L1 中 scale 的 stride
loadAScaleParam.dstStride = loadAScaleParam.yStep;
                                            // L0 中 scale 的 stride
```

**参数含义图解**：

```
Scale 在 L1 中（NZ 格式）:
      yStep=2
      ←────→
     ┌────┬────┐
     │ s0 │ s1 │  ← 每个 bf16 包含 2 个 fp8_e8m0
     │ s2 │ s3 │
xStep│ .. │ .. │  ← 共 xStep 行
     │    │    │
     └────┴────┘
     
每行对应一个 token 的 scale（4 个 fp8_e8m0 = 2 个 bf16）
yStep = D / (32 × 2) = 128 / 64 = 2
```

### 10.3 L0 中的 mx_fp8_e4m3_t 布局

`mx_fp8_e4m3_t` 是硬件层面的特殊格式，将数据和 scale 打包到 L0 中：

```
L0 中的 mx_fp8_e4m3_t:
┌─────────────────────────────────────────────────┐
│ 数据区域：fp8_e4m3 元素                           │
│ [m0_k0, m0_k1, ..., m0_k31]  ← 32 个元素        │
│ [m1_k0, m1_k1, ..., m1_k31]                     │
│ ...                                              │
├─────────────────────────────────────────────────┤
│ Scale 区域：fp8_e8m0 scale                        │
│ [scale_k0]  ← 对应 k[0:31] 的 scale              │
│ [scale_k1]  ← 对应 k[32:63] 的 scale             │
│ [scale_k2]  ← 对应 k[64:95] 的 scale             │
│ [scale_k3]  ← 对应 k[96:127] 的 scale            │
└─────────────────────────────────────────────────┘
```

---

## 十一、mmad_mx 矩阵乘 —— 硬件融合 Scale

### 11.1 触发方式

将 L0 tensor 重解释为 `mx_fp8_e4m3_t` 类型后调用 `Mmad`，硬件自动使用 `mmad_mx` 指令：

```cpp
// service_cube.h: ComputeL0c()
MmadParams mmadParams;
mmadParams.m = CeilAlign(s1gL0RealSize, 16);  // M 对齐到 16
mmadParams.n = s2L0RealSize;                    // N = S2 块大小
mmadParams.k = headDim;                         // K = 128

LocalTensor<mx_fp8_e4m3_t> qL0Mx =
    queryL0_[...].template ReinterpretCast<mx_fp8_e4m3_t>();
LocalTensor<mx_fp8_e4m3_t> kL0Mx =
    keyL0_[...].template ReinterpretCast<mx_fp8_e4m3_t>();

Mmad(cL0_[...], qL0Mx, kL0Mx, mmadParams);
// 硬件执行: C[m,n] = Σ_k (Q[m,k] × qScale[m,k/32]) × (K[k,n] × kScale[k,n/32])
// 输出为 float
```

### 11.2 mmad_mx 的计算语义

```
普通 Mmad:
  C[m,n] = Σ_k Q[m,k] × K[k,n]

mmad_mx:
  C[m,n] = Σ_k (Q[m,k] × qScale[m, k/32]) × (K[k,n] × kScale[k, n/32])
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           硬件自动完成 scale 融合，无需软件干预
```

**注意**：这里的 `qScale[m, k/32]` 和 `kScale[k, n/32]` 是在 L0 加载阶段由 `LoadData` Mx 变体一并加载到 L0 中的，`Mmad` 指令直接读取 L0 中的 scale 信息。

### 11.3 矩阵维度

```
Q: [M, K=128]  ×  K^T: [K=128, N]  →  C: [M, N]

其中：
- M = s1gL0RealSize（最大 M_BASIC_BLOCK=256，按 16 对齐）
- N = s2L0RealSize（最大 S2_BASIC_BLOCK_L0=128）
- K = headDim = 128（固定）
```

---

## 十二、Vector 核后处理 —— 简化流程

### 12.1 MXFP8 带来的简化

老流程中 Vector 核需要：
1. 加载 qScale 到 UB → `weights × qScale`
2. 加载 kScale 到 UB → `accum × kScale`
3. 管理 qScale/kScale 的 Event 同步

MXFP8 流程中，这些步骤全部**移除**，因为 scale 已在 Cube 核由 mmad_mx 融合。

### 12.2 Vector 核计算流程（MXFP8）

```
输入：
  - QK [S1×G, S2] float（来自 Cube 核 Fixpipe，已含 qScale×kScale）
  - weights [G] bf16

计算步骤：
  1. 加载 weights 到寄存器 regW（不乘 qScale）
  2. 初始化累加器 regSum = 0
  3. for i in 0..G:
       加载 QK[i] 到 regQK
       Relu(regQK)
       regwBrc = broadcast(regW[i])
       regSum += regQK × regwBrc    (MulAddDst)
  4. Cast regSum → bf16
  5. FloatToSortableKey → uint16 score
  6. score → scoreGm
```

### 12.3 代码对比

```cpp
// MXFP8 分支：调用 MxFP8 变体
if constexpr (IS_MXFP8) {
    vector1::BatchMulWeightAndReduceSum_MxFP8(
        outBase, outStride,
        qkBase, qkVLstride, qkStride,
        weightBase, weightStride, weightFloatBase,
        gSize_, curAivS1ProcNum);
}
// 老流程分支：传入 qScale 和 kScale
else {
    vector1::BatchMulWeightAndReduceSum(
        outBase, outStride,
        qkBase, qkVLstride, qkStride,
        weightBase, weightStride, weightFloatBase,
        kScaleBase, 0,           // ← kScale
        qScaleBase, qScaleStride, // ← qScale
        gSize_, curAivS1ProcNum);
}
```

### 12.4 Buffer 节省

MXFP8 模式下不分配 kScale/qScale UB buffer：

```cpp
if constexpr (!IS_MXFP8) {
    pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * 16 * sizeof(SCALE_T));
    pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE);
}
```

---

## 十三、完整数据流时序图

### 13.1 单个 S2 块的处理时序

```
时间轴 →

AIC (Cube核):
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 1. WaitFlag(KEY_MTE1_MTE2)                                             │
  │ 2. KeyNd2Nz: K GM → L1 (ND2NZ)                                        │
  │ 3. LoadKScaleToL1: kScale GM → L1 (DN2NZ, bf16)                       │
  │ 4. SetFlag(MTE2_MTE1)                                                  │
  │ 5. ┌─── for s1g block ─────────────────────────────────────────────┐   │
  │    │ 6. QueryNd2Nz: Q GM → L1 (ND2NZ) [仅首次]                     │   │
  │    │ 7. LoadQScaleToL1: qScale GM → L1 (DN2NZ, bf16) [仅首次]     │   │
  │    │ 8. ┌─── for s2L0, s1gL0 block ─────────────────────────────┐  │   │
  │    │    │ 9.  LoadQueryToL0a: Q L1 → L0 (Mx: data+scale)        │  │   │
  │    │    │ 10. LoadKeyToL0b: K L1 → L0 (Mx: data+scale)          │  │   │
  │    │    │ 11. ComputeL0c: Mmad(qL0Mx, kL0Mx) → L0C float        │  │   │
  │    │    │ 12. Fixpipe: L0C → UB (ROW_MAJOR, ND格式)              │  │   │
  │    │    └────────────────────────────────────────────────────────┘  │   │
  │    └────────────────────────────────────────────────────────────────┘   │
  │ 13. CrossCoreSetFlag(CROSS_CV_EVENT) → 通知 V 核                       │
  └──────────────────────────────────────────────────────────────────────────┘

AIV (Vector核):
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 1. 加载 weights: weightsGm → weightUB_ [仅首次 S2]                     │
  │ 2. (MXFP8: 不加载 qScale/kScale)                                       │
  │ 3. CrossCoreWaitFlag(CROSS_CV_EVENT) ← 等 C 核完成                     │
  │ 4. BatchMulWeightAndReduceSum_MxFP8:                                  │
  │    Relu(QK) × weights → ReduceSum → bf16 → SortableKey → score        │
  │ 5. score → scoreGm                                                     │
  │ 6. CrossCoreSetFlag(CROSS_VC_EVENT) → 通知 C 核                        │
  │ 7. (最后一个 S2 块) ProcessTopK: score → TopK → indices → GM           │
  └──────────────────────────────────────────────────────────────────────────┘
```

### 13.2 三层循环嵌套

```
for bN2LoopIdx in [bN2Start, bN2End]:        ← Batch × N2 循环
  for gS1LoopIdx in [gS1Start, gS1End]:      ← S1×G 分块循环
    for s2LoopIdx in [s2Start, s2End]:        ← S2 分块循环
      ProcessBaseBlock():
        AIC: ComputeMm1()                     ← 矩阵乘
        AIV: ProcessVec1()                    ← 后处理
        AIV: ProcessTopK() [仅最后一个S2块]   ← TopK
```

---

## 十四、代码定位速查

### 14.1 按数据流顺序

| 阶段 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **类型定义** | `common.h` | `QLIType` MXFP8 特化 | L56-71 |
| **算子注册** | `def.cpp` | ascend950 位置4 | L84-141 |
| **Shape 校验** | `tiling.cpp` | `CheckScaleShape()` | L740-783 |
| **AIC/AIV 初始化** | `kernel.h` | `Init()` | L426-457 |
| **偏移计算** | `kernel.h` | `CalcRunInfo()` | L568-592 |
| **Q GM→L1** | `service_cube.h` | `QueryNd2Nz()` | L460-476 |
| **K GM→L1** | `service_cube.h` | `KeyNd2Nz()` / `KeyNd2NzForPA()` | L353-399 |
| **qScale GM→L1** | `service_cube.h` | `LoadQScaleToL1()` | L401-429 |
| **kScale GM→L1** | `service_cube.h` | `LoadKScaleToL1()` | L431-456 |
| **Q L1→L0 (Mx)** | `service_cube.h` | `LoadQueryToL0a()` | L478-524 |
| **K L1→L0 (Mx)** | `service_cube.h` | `LoadKeyToL0b()` | L526-571 |
| **mmad_mx** | `service_cube.h` | `ComputeL0c()` | L573-598 |
| **Fixpipe** | `service_cube.h` | `Fixp()` | L600-634 |
| **Vec 后处理** | `service_vector.h` | `ProcessVec1()` | L328-436 |
| **VF MxFP8** | `vector1.h` | `MulWeightAndReduceSum_MxFP8()` | L877-949 |
| **TopK** | `service_vector.h` | `ProcessTopK()` | L438-571 |

### 14.2 按功能分类

| 功能 | 关键标识 |
|------|---------|
| 编译期 MXFP8 判断 | `IS_MXFP8 = QLIT::isMxFp8` |
| L1 Buffer 额外空间 | `qL1ExtraScale`, `kL1ExtraScale` |
| Scale 打包搬运 | `FP8_TWO = 2`, `ReinterpretCast<bfloat16_t>` |
| DN2NZ 格式转换 | `Dn2NzParams`, `nValue = scalePerToken / FP8_TWO` |
| L0 Mx 加载 | `LoadData(dstTensor, dataL1, scaleL1, params2D, paramsMx)` |
| mmad_mx 触发 | `ReinterpretCast<mx_fp8_e4m3_t>()` + `Mmad()` |
| Vector 核简化 | `BatchMulWeightAndReduceSum_MxFP8()` |

---

## 附录 A：关键常量速查

```cpp
// 数据类型
using FP8E4M3 = fp8_e4m3fn_t;     // Q/K 数据
using FP8E8M0 = fp8_e8m0_t;       // Scale

// 量化参数
constexpr uint32_t MXFP8_BLOCK_SIZE = 32;   // 每32个元素一个scale
constexpr uint32_t FP8_TWO = 2;             // 2个fp8_e8m0打包成1个bf16
constexpr uint32_t FP8_BLOCK_CUBE = 32;     // Cube 单元的 FP8 块大小

// Buffer 大小
constexpr uint64_t M_BASIC_BLOCK = 256;     // Q 的 M 方向基本块
constexpr uint64_t D_BASIC_BLOCK = 128;     // D 方向基本块 = headDim
constexpr uint64_t S2_BASIC_BLOCK_L0 = 128; // K 的 S2 方向 L0 基本块

// Buffer 数量
constexpr uint64_t QUERY_BUF_NUM = 2;       // Q L1 ping-pong buffer 数
constexpr uint64_t KEY_BUF_NUM = 3;         // K L1 三级流水 buffer 数
constexpr uint64_t L0_BUF_NUM = 2;          // L0 ping-pong buffer 数

// 偏移
constexpr uint64_t QUERY_BUFFER_OFFSET = 256 * 128;  // = 32768
constexpr uint64_t L0AB_BUFFER_OFFSET = 256 * 128;   // = 32768
constexpr uint64_t L0C_BUFFER_OFFSET = 256 * 128;    // = 32768
```

## 附录 B：GM 偏移计算公式

### qScale 偏移（kernel.h CalcRunInfo）

```cpp
// qScale shape: [B, S1, N1, D/32]
// 每个 batch 的前缀和
uint64_t qScalePrefixSum = actualSeqQPrefixSum * qHeadNum * (headDim / MXFP8_BLOCK_SIZE);
// 当前核的偏移
qScaleCoreOffset_ = qScalePrefixSum + gS1Idx * mBaseSize * (headDim / MXFP8_BLOCK_SIZE);
```

### kScale 偏移（kernel.h CalcRunInfo）

```cpp
// kScale shape: [B, S2, N2, D/32]
keyScaleCoreOffset = (actualSeqKPrefixSum + s2Idx * s2BaseSize)
                     * kHeadNum * (headDim / MXFP8_BLOCK_SIZE);
```

### PA 场景 kScale stride（tiling.cpp）

```cpp
// MXFP8: kScale shape [BlockNum, BlockSize, N2, D/32]
keyDequantScaleStride0 = blockSize_ * (headDim_ / 32);
// 非 MXFP8: keyDequantScaleStride0 = blockSize_
```

---

**文档版本**: 1.0  
**创建日期**: 2026-06-09  
**适用范围**: QLI 算子 MXFP8 量化的数据传输与排布全链路教学
