# Compressed Sparse Attention（CSA）算子技术细节

## 1. 概述

<!-- > 简述 CSA 算子的目标、应用场景及核心思想。

- **算子功能**：
- **输入/输出**：
- **核心优化点**： -->
CSA 是一种融合了**压缩**与**稀疏选择**的高效注意力机制。其核心流程为：

- 将每 $m$ 个 token 的 KV cache **压缩**为一个压缩条目（compressed KV entry），实现 $m\times$ 的序列长度压缩；
- 通过 **Lightning Indexer** 从压缩后的 KV 条目中**稀疏选取 Top-$k$** 个与当前 query 最相关的条目；
- 对选中的稀疏 KV 条目执行 **Shared Key-Value MQA**（Multi-Query Attention），完成最终的注意力计算。

目标是在**极长上下文场景**下大幅降低注意力的计算开销，使百万 token 级别的上下文在工程上可行。

### 1.1. 应用场景

- **超长文本推理**：当上下文长度达到极端规模（如百万 token），注意力成为模型的计算瓶颈时，CSA 作为 DeepSeek-V4 的核心注意力架构之一发挥作用。
- **与 HCA 混合部署**：CSA 与 HCA（Heavily Compressed Attention，以更大的压缩粒度 $m' \gg m$ 进行极简压缩）交织混合使用，在不同层或不同场景下互补，兼顾效率与精度。

### 1.2. 核心思想

| 优化点 | 具体做法 |
|---|---|
| **KV 压缩** | 用两组可学习的压缩权重 $Z^a, Z^b$ 和位置偏置 $B^a, B^b$，经 Softmax 归一化后对每 $m$ 个 token 的 KV 做加权聚合（Hadamard product + 求和），将序列长度压缩 $m$ 倍 |
| **低秩索引器（Lightning Indexer）** | 通过 Down/Up Projection 矩阵将 query 先压缩至低维隐空间，再升维生成 indexer queries；配合 ReLU 激活和加权求和计算每个压缩块与 query 的 index score |
| **Top-$k$ 稀疏选择** | 基于 index score 对压缩块做 Top-$k$ 筛选，每个 query 仅关注最相关的 $k$ 个压缩 KV 条目，避免全量注意力计算 |
| **Shared Key-Value MQA** | 压缩 KV 条目同时作为 attention 的 Key 和 Value，共用一份隐空间表示，减少存储和计算冗余 |
| **Grouped Output Projection** | 将 $n_h$ 个注意力头输出分为 $g$ 组，先在组内投影到较小的中间维度 $d_g$，再拼接投影到最终输出，避免 $c \cdot n_h \to d$ 直接投影带来的大矩阵乘法开销 |

总体而言，CSA = KV 压缩（$m\times$）+ 低秩索引稀疏选择（Top-$k$）+ 共享 KV 的 MQA + 分组输出投影。

**CSA 算子是 DeepSeek V4中最为核心的组件之一，其性能优化对于DeepSeek V4的部署至关重要。本文主要介绍 CSA 算子在950上的极致性能优化。**

---

## 2. Chromosomal Crossover：双 AI Core 共享 KV Cache

### 2.1 动机与问题

950 卡为 CV 不分离架构，Cube 核与 Vector 核之间存在通路。对于这种 CV 不分离架构来说，优势在于融合算子的中间数据变量传输不需要出核（经过 L2 cache 或者 HBM），但是这也会给算子的 tiling 设计带来一些挑战：Cube 核或者 Vector 核产生的数据需要即时地被另一个核消费，L1 cache 及 UB 的空间会更加紧张，同时 Bank 冲突的情况也会加剧。

![](fig/950.png)

对于 DeepSeek V4 的 Attention 机制，K 和 V 为同一份 tensor，单 head，`head_dim = 512`。考虑 K 和 V 实际上是同一份到因此我们需要在 L1 Cache 中为 KV 预留三份 buffer，这样可以保证访存没有 bubble。如下图所示，仅考虑 FlashAttention 主要的计算流程，其中 Sx 表示第 x 次 softmax 计算，Oy 表示第 y 次 rescale 计算。

![](fig/3buffer.png)

L1 cache 中 3buffer KV 占用空间大小为：`128 × 576 × 2 × 3 = 432 KB`，L1 cache 还剩余空间大小为 `80 KB`，因此行方向基本块大小只能切到 64，为 Q 预留空间大小为 `64 × 576 × 2 = 72 KB`。此处如果通过减少 KV 序列方向的基本块大小调大行方向基本块大小，会导致 rescale 计算次数过多，引入大量向量化计算，同时在 UB 上需要常驻更大尺寸的输出 tensor，依然不可行。

对于稀疏的 SparseFlashAttention 而言，性能瓶颈在于对 KV 的离散搬运，也就是 V0 阶段的 mergeKV 操作，这一部分我们会利用 Vector 核进行搬运，通过聚合两个 token 的访存减少指令数，对于量化场景还会涉及到对 KV 的反量化。

当 Group Size `G = 128` 时，单个 token 对应的 attention 需要处理 128 行 Q，而基本块仅能容纳 64 行，因此 **需要两次迭代** 才能完成一个 token 的计算，这意味着 V0（mergeKV）阶段对同一批 KV 需要搬运两次，带来大量冗余的数据搬运开销，性能较差。

**核心矛盾**：L1/UB 容量不足 → 基本块行数受限 → 迭代次数翻倍 → KV 搬运冗余。

### 2.2 Chromosomal Crossover 方案

为解决上述矛盾，我们采用一种**染色体交错（Chromosomal Crossover）**流水方式：**每两个 AICore 配对，协同处理同一个 token 对应的 G=128 的计算**。DeepSeek 的 FlashMLA 仓也采用了这种方式：[A Deep Dive Into The Flash MLA FP8 Decoding Kernel on Hopper](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md#crossover)。

具体流程如下：

1. **V0 阶段（mergeKV）**：两个 AICore 各自负责搬运并聚合当前 token 所需 KV 的一半（即各搬运 G=64 对应的 KV block）。各自对离散的 KV Cache 做反量化后，将聚合结果写出到 GM 上的共享区域。
2. **C1 阶段（Q@Kᵀ 计算）**：两个 AICore 在 C1 阶段**读取同一块共享的 KV**，各自用自己那部分 Q 与共享 KV 做 MatMul，完成 attention 计算。

![Chromosomal Crossover 示意图](fig/crossover.png)

这样，每个 AICore 的 L1 只需容纳 `S_inner = 128` 的 KV（三份 buffer 共 432 KB）+ `S_q = 64` 的 Q（72 KB），恰好满足容量约束；而 G=128 的计算由两个 AICore 并行完成，**无需二次迭代**。

## 3. Cube 优化

### 3.1 旧方案和新方案的区别

> 对比 Cube 优化前后的方案差异。

### 3.2 具体的代码实现

> 关键代码改动与实现细节。

### 3.3 修改后 MAC 性能数据对比

Prefill 场景（N=128, S1=8k, S2=128k）下，Cube 优化把 Cube 总体耗时从 9965µs 降到 6690µs，降幅约三分之一：

| N=128 prefill（S1=8k, S2=128k） | Cube 总体耗时 (µs) |
|---|---:|
| 优化前 | 9965 |
| 优化后 | 6690 |
| **Δ** | **−33%** |

---

## 4. GetKVPhyAddr：VF 向量化实现

### 4.1 动机与功能

PagedAttention 下,KV cache 按 block 物理分页存放,逻辑上连续的 KV 序列在物理上被 `block_table` 重映射到任意物理块。稀疏注意力又只选中部分 block(`sparse_indices`)。因此每个被选中的逻辑索引 `s2Idx` 需要换算成真实物理地址:

![换算真实物理地址](fig/phyaddr.png)

```
blkIdx  = s2Idx / blockSize                       // 落在第几个逻辑块
blkOff  = s2Idx % blockSize                        // 块内偏移
phyBlk  = block_table[blkIdx]                       // 逻辑块 → 物理块号(查表)
phys_addr(元素) = phyBlk * kvStride + blkOff * kvDim
```

- `kvStride`:一个物理块的元素跨度(block_size × kvDim 量级)。
- `kvDim`:单 token 的特征维(每行元素数)。
- **结果是 int64**:`phyBlk * kvStride` 会超过 2^32, 必须用 64 位地址。

`GetKVPhyAddr` 要做的事情是把一整行 query 选中的 `total` 个逻辑索引批量换算成 `total` 个 int64 物理地址,写回 `kvPhyAddrGm`,供后续 `GetRealS2Addr` 取址搬 KV。 `GetKVPhyAddr` 用 VF 实现，把以上换算物理地址换算公式**向量化、并把查表从 GM 标量访问改成 UB 向量 gather**。

### 4.2 标量实现 vs 向量化实现

> 对比标量逐元素计算与 VF 向量化的差异，说明向量化的收益。

<!-- | 维度 | 标量实现 | VF 向量化 |
|------|----------|-----------|
| 每次处理元素数 | 1 | N（向量宽度） |
| 指令条数 | O(N) | O(1) |
| 延迟 | 高 | 低 | -->

向量寄存器以 **256B 为单位**:`RegTensor<uint32_t>` 一cycle 64 个 int32 = 64×4B = **256B**。内核每个 loop 用 **2 个寄存器**(`_1`/`_2`)→ 一次处理 **128 个逻辑索引**; 输出是 int64 (拆成 L/H 两个 int32)。

| 维度 | 标量 `GetkeyOffset`(逐元素) | 向量 `GetKVPhyAddrVFImpl`(VF) |
|---|---|---|
| 处理粒度 | 1 个索引 / 次调用 | **128 个索引 / loop**(2 reg × 64) |
| 除 blockSize | `s2Idx / blockSize` 除法 | `ShiftRights`(blockSize=2^7 → 右移 7 位) |
| 模 blockSize | `s2Idx % blockSize` 取模 | `Muls(blkIdx,blockSize)` + `Sub` 还原余数 |
| 块内 offset | `blkOff * dSize` | `Muls(pa_offset, kvDim)` |
| block_table 查表 | `blockTableGm.GetValue(...)` **GM 标量逐个**,延迟敏感 | 先 `CopyPaTableToUb` 预载 UB,再 `DataCopyGather` **向量 gather** |
| int64 乘 | 标量 64 位乘 | `Mull` → 低 32 位 + overflow 高 32 位 |
| int64 加 | 标量 `+` | `Add`(出 carry) + `AddC`(吃 carry) 高低分算 |
| 写出 | 1 个 int64 | `StoreAlign<DIST_INTLV_B32>` 交织 L/H 成 int64 |
| 访存形态 | GetValue 逐元素访问 | UB 批量 + 全程寄存器内 SIMD |

**向量化收益:**
1. **循环次数 ÷128**:`total` 个索引从 `total` 次标量迭代降到 `ceil(total/128)` 次 VF 迭代。
2. **消除 GM 标量查表延迟**:`block_table` 一次性 `CopyPaTableToUb` 到 UB,之后用
   `DataCopyGather` 一拍 gather 64 个物理块号,替代 `GetValue` 逐个访 GM。
3. **除/模变移位**:`blockSize` 取 2 的幂, 除法和取模降为 `ShiftRights`+`Muls`+`Sub`。
4. **int64 地址 SIMD 化**:用 `Mull`/`Add`/`AddC` + carry mask 在寄存器内完成 64 位运算, 无需标量 64 位 ALU 逐个算。

---

### 4.3 向量化算法细节

<!-- ![GetKVPhyAddr 向量化流程](fig/getkvphyaddr_vf.png) -->
### 4.3.1 常量与排布

```cpp
s2NumPerLoop   = 128   // 每 loop 处理 128 个逻辑索引
s2NumPerReg    = 64    // 每寄存器 64 个 int32 = 256B
outOffsetPerLoop = 256 // 输出 int64 → 翻倍:128 索引 × 2(L/H) = 256 个 uint32
outOffsetPerReg  = 128 // 64 索引 × 2
```
每 loop 用 `_1`(前 64)、`_2`(后 64)两套寄存器并行同一套运算。

### 4.3.2 VF 指令序列(每 loop,以 `_1` 为例)

逻辑公式为 `addr = block_table[idx>>7]*kvStride + (idx - (idx>>7)*blockSize)*kvDim`,

1. 载入 64 个逻辑索引 (256B 对齐)
```cpp
LoadAlign(vreg_sparse_idx_1, sparseIdxUb + i*128);  // _2: + 64
```
2. 右移求逻辑块号, 取代除法
$$
pa\_blk\_idx = idx/128 = idx >> 7,
$$
```cpp
ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_all_b32);
```
3. 求块内偏移
$$
pa\_offset = idx - pa\_blk\_idx*blockSize,
$$
```cpp
Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_all_b32);
Sub (vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_all_b32);
```
4. 计算块内元素物理 offset
$$
offset = pa\_offset * kvDim,
$$
```cpp
Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_all_b32);
```
5. 查 block_table: 逻辑块号 → 物理块号(UB 向量 gather)
$$
phyBlk  = block\_table[blkIdx],
$$
```cpp
DataCopyGather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_all_b32);
```
6. 现在要计算 $$P = phyBlk * kvStride,$$ 需要用 int64 表示，先拆分高低 32 位uint32 进行表示：
$$
   P = P_L + P_H*2^{32},
$$ 此时的物理地址是 addr = P + offset 之和
$$
addr = 
\begin{cases}
P_H*2^{32} + (P_L+offset),\ \ \ \ \ (P_L+offset) < 2^{32}\\
(P_H + 1)*2^{32} + (P_L+offset-2^{32}),\ \ \ \ \ \ \ \  else.
\end{cases}
$$
```cpp
Mull(vreg_blk_id_mul_stride_L_1, vreg_mul_overflow_L_1, vreg_phy_blk_idx_1, vreg_kvStride, preg_all_b32);
Add (add_carry_l_1, vreg_total_offset_L_1, vreg_blk_id_mul_stride_L_1, vreg_phy_offset_1, preg_all_b32);
AddC(add_carry_h_1, vreg_total_offset_H_1, vreg_mul_overflow_L_1, vreg_zero, add_carry_l_1, preg_all_b32);
```
7. 交织写出物理地址的高32位和低32位
```cpp
StoreAlign<DIST_INTLV_B32>(kvPhyAddrUb + i*256, vreg_total_offset_L_1, vreg_total_offset_H_1, preg_all_b32);
```
<!-- ### 4.3.2 VF 指令序列(每 loop,以 `_1` 为例)

逻辑公式 `addr = block_table[idx>>7]*kvStride + (idx - (idx>>7)*blockSize)*kvDim`:

```cpp
// ① 载入 64 个逻辑索引(256B 对齐)
LoadAlign(vreg_sparse_idx_1, sparseIdxUb + i*128);              // _2: + 64
// ② 右移求逻辑块号  pa_blk_idx = idx >> shiftRightNum(=7,即 /128)
ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_all_b32);
// ③ 求块内偏移  pa_offset = idx - pa_blk_idx*blockSize
Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_all_b32);
Sub (vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_all_b32);
// ④ 块内元素 offset = pa_offset * kvDim
Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_all_b32);
// ⑤ 查 block_table:逻辑块号 → 物理块号(UB 向量 gather)
DataCopyGather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_all_b32);
// ⑥ int64 乘:phyBlk * kvStride → 低位 + overflow 高位
Mull(vreg_blk_id_mul_stride_L_1, vreg_mul_overflow_L_1, vreg_phy_blk_idx_1, vreg_kvStride, preg_all_b32);
// ⑦ int64 加:+ phy_offset(低位出 carry)
Add (add_carry_l_1, vreg_total_offset_L_1, vreg_blk_id_mul_stride_L_1, vreg_phy_offset_1, preg_all_b32);
// ⑧ 高位 = overflow + 0 + carry
AddC(add_carry_h_1, vreg_total_offset_H_1, vreg_mul_overflow_L_1, vreg_zero, add_carry_l_1, preg_all_b32);
// ⑨ 交织 L/H 写出成 int64 地址
StoreAlign<DIST_INTLV_B32>(kvPhyAddrUb + i*256, vreg_total_offset_L_1, vreg_total_offset_H_1, preg_all_b32);
```

循环外预置:`Duplicate(vreg_zero, 0)`、`Duplicate(vreg_kvStride, kvStride)`。 -->

### 4.3.3 掩码处理(两类 MaskReg,作用不同)

1. **谓词掩码** `preg_all_b32 = CreateMask<MaskPattern::ALL>()`:每条算术指令的 lane 使能, 当前实现**全开**(所有 64 lane 都算)。
2. **进位掩码** `add_carry_l_1` / `add_carry_h_1`:**不是 lane 谓词,而是 64 位加法的进位**。
   `Add` 把低 32 位相加产生的进位写入 `add_carry_l_1`,`AddC` 再把它吃进高 32 位 (`高 = overflow + 0 + carry`)。这是用 MaskReg 承载 carry 实现寄存器内 int64 加法的关键。

### 4.3.4 边界条件(非对齐长度 `total % 128 ≠ 0`)

首先计算循环参数:
```cpp
s2Loop = (total + 128 - 1) / 128;     // 向上取整
s2Tail = total - (s2Loop - 1) * 128;  // 尾块有效数
```
当前 VFImpl 的处理方式:
- **VF 内核按 `s2Loop` 跑满**,每 loop 都用 `preg_all`(128 个)→ **尾块会多算最多 127 个**地址到 UB 缓冲。
- 输入 `sparseIdxUb` 由 `DataCopyPad` 零填充,尾块读到的是 0,算出的多余地址无害。
- **正确性靠"写出时按 `total` 截断"保证**:`CopyPhyAddrToGm(..., total, ...)`(`blockLen = total*sizeof(int32)*2/32`)只把前 `total` 个有效地址搬回 GM,多算的尾块不写出。

### 4.4 性能数据
当前物理地址向量化的收益明显，Vector部分的收益约35\%，端到端收益约 5\%。

| 场景 | 指标 | 优化前 | 优化后 | 减少量 | 优化幅度 |
|:---|:---|---:|---:|---:|---:|
| **N=128 Decode** | 总耗时 | 140.18 | 133.38 | 6.80 | **↓ 4.85%** |
|            | Vec 耗时 | 130.36 | 84.32 | 46.04 | **↓ 35.3%** |
| **N=128 Prefill** | 总耗时 | 10732.04 | 10242.30 | 489.74 | **↓ 4.56%** |
|             | Vec 耗时 | 10588.48 | 6644.36 | 3944.12 | **↓ 37.3%** |

---

## 5. 核间流水排布

### 5.1 动机与问题

第 4 节把地址计算向量化后，scalar 耗时大幅下降，但端到端总耗时并未随之明显下降——MFU 依然偏低。参考下表，可见仅压低 scalar 不足以兑现端到端收益：

| CSA Prefill | AIV scalar (µs) | 端到端耗时 (µs) | MFU |
|---|---:|---:|---:|
| 向量化前 | 10461 | 10731 | 53.3% |
| 向量化后 | 6500 | 10241 | 55.8% |
| **Δ** | **−38%** | **−4.6%** | **+2.5%** |

scalar 缩减近 40%（−3961µs），但端到端仅降 4.6%（−490µs）、MFU 也几乎持平，scalar 的收益并未兑现到端到端。

本节从**核间流水排布**入手——通过加深流水与改变核间同步的方式，把端到端耗时降下来。

### 5.2 v0preload：加深一级 KV 矩阵预取流水（预取深度 2→3，N=128）

每个 S2 基本块都要用到一份 **bf16 的反量化 KV**（`s2BaseSize=128 × D=512`，合计 **128KB**）。N=128 下它经 GM 在两个 AIC 间共享（G=128 跨 2 个 AIC、各算 64 头）：由 AIV 的 `ProcessVec0` 在 UB 算出后写到 GM 的 `v0ResGm`，AIC 用时再经 `IterateLoadQK` 从 GM `DataCopy` 到 L1 的 KV 矩阵缓冲 `l1RightBuffers`。这份 KV 矩阵从产出一直存活到被 bmm2 消费——既作为 bmm1 的 KV 矩阵（当 K，算 Q·Kᵀ），又作为 bmm2 的 KV 矩阵（当 V，算 P·V）。

回顾第 2 节：正因为一份 KV 矩阵要在 L1 里被 `load→bmm1→bmm2` 连续消费，我们为它在 L1 预留了**三份 buffer**（`l1RightBuffers`，按 `taskIdMod3` 轮转），让相邻基本块的取数与计算彼此错开、消除访存 bubble。

![L1 为 KV 预留的三份 buffer（见 §2）](fig/3buffer.png)

这三份 buffer 能否被持续填满、连续周转，正是本节流水排布要解决的问题。

一份 KV 矩阵每经过一个基本块循环就前进一步，依次走 **vec0 产出（AIV）→ load 取数 GM→L1（AIC）→ bmm1（AIC）→ bmm2（AIC）** 四步。收益的关键，在于 `load` 与 `vec0` 落在哪一轮基本块循环：

**修改前（预取深度 2）——取数与产出同一轮**：`PRELOAD_NUM = 2`、`runInfo[3]`，参与轮转的 KV 矩阵 3 份。`load` 与 `vec0` 落在同一轮、且取的是同一块；AIC 必须等 AIV 的 `ProcessVec0` 产出并跨核同步后才能 `IterateLoadQK`，于是 cube 每轮都要空等一次。

**修改后（预取深度 3）——load 比 vec0 晚一轮**：把预取前瞻 `PRELOAD_NUM` 从 2 提到 3（`runInfo[4]`），让 `load` 与 `vec0` 解耦到相邻两轮基本块循环——`vec0` 在本轮产出并写 GM，`load` 推迟到下一轮才从 GM 取到 L1。AIV 因此有一整轮时间完成"产出 + 跨核交付"，AIC 取数时数据已在 GM 就绪、不再等待，这段延迟被掩盖到前序基本块的 cube 计算之后。

![v0preload 前后核间流水对比](fig/pipeline.svg)

加深预取后，vec0 / load / bmm1 / bmm2 四个阶段各持有一份 KV 矩阵，参与轮转的 KV 矩阵从 3 份增加到 4 份。**但在 N=128 上，这一份的增加并不需要为存储多分配空间**，原因是：N=128 的场景本就把 KV 矩阵经 GM 中转（G=128 跨 2 个 AIC 共享），用来承接这第 4 份的 GM 空间早已存在、无需新开。于是 L1 侧仍只用 `l1RightBuffers` 的 3 块缓冲（按 `taskIdMod3` 轮转）、分配不变——这正是 N=128 成为验证该优化理想场景的原因。

回到 buffer 视角：上图中每一份 KV 矩阵（绿框）从 `vec0` 产出到被 `bmm2` 消费，要在 L1 的 3 块 `l1RightBuffers` 里存活 `load → bmm1 → bmm2` 三轮（图中同色块即同一块 buffer，3 块按 `taskIdMod3` 轮转）。修改前 `load` 与 `vec0` 同轮，cube 每轮都要先等 `vec0` 产出、才能把当前轮到的那块 buffer 填上（上图红框），3 块 buffer 无法连续周转、cube 周期性空等；修改后 `load` 提前一轮、取上一轮就已就绪的 KV，3 块 buffer 依次被 `load / bmm1 / bmm2` 占满、连续周转，cube 不再空等。这正是第 2 节“以三份 buffer 消除 bubble”的设计被真正兑现——cube MFU 从 55.8% 升到 74%。

### 5.3 推广到 N=64：核间同步重构

N=128 的第 4 份产出的 KV 矩阵本就放在既有的 GM 中转区；而 N=64 原先由 AIV 把 KV 矩阵从 UB 直接搬到 L1（`CopyOutKvUb2L1`）、没有走 GM，要再多保留一份产出的 KV 矩阵，就得为它新增存放空间。

但该场景下 **L1 已被占满**——L1 共 512KB，分配为：KV 矩阵区 `l1RightBuffers` `3×128KB = 384KB`、P 缓冲 `l1P` `2×16KB = 32KB`、Q 缓冲 `l1Q` `96KB`，合计 `384 + 32 + 96 = 512KB`，放不下第 4 个 128KB 的 KV 矩阵。

**为了在 N=64 使能该优化，需要把 KV 矩阵也改走 GM 中转**：`ProcessVec0` 算出后由 `CopyOutKvUb2Gm` 写到 GM 上的 `v0ResGm`，AIC 用时再经 `IterateLoadQK` 从 GM `DataCopy` 回 `l1RightBuffers`。这样所有 KV 矩阵都先在 GM 周转、再按需载入 L1；GM 中转区与 L1 一样是 3 缓冲、按 `taskIdMod3` 轮转（每份 KV 矩阵驻留 ≤3 轮：load→bmm1→bmm2），代价是每个 AIC 的 workspace 有限增大 `3×128KB = 384KB`，L1 分配不变。

随之**核间同步从「L1 跨核」改为「GM 跨核」**：barrier 由「KV 矩阵写完 L1」（`CopyOutKvUb2L1` 的 `CROSS_CORE_SYNC_FORWARD`，AIC 在 `MTE1` 等待）前移到「AIC 预取」（`CopyOutKvUb2Gm` 的 `CROSS_CORE_SYNC_BACKWARD`，AIC 在 `IterateLoadQK` 内以 `MTE2` 等待后再 `DataCopy`），使 AIC 能提前发起 GM→L1 预取、与前序 cube 计算 overlap。这条改写同时统一了代码：N=64 与 N=128 不再按 `IS_SPLIT_G` 分作两条路径，双主循环 `ProcessMainLoopN64`/`ProcessMainLoopN128` 也合并为单一 `ProcessMainLoop`。

### 5.4 性能数据

下表为 prefill 场景（N=128）流水重排前后的端到端耗时与 cube MFU：

| CSA Prefill（N=128） | 端到端耗时 (µs) | MFU |
|---|---:|---:|
| 流水重排前 | 10241 | 55.8% |
| 流水重排后 | 7734 | 74.0% |
| **Δ** | **−24.5%** | **+18.2%** |

CSA 算子在 N=64/128 各类 case 上平均取得了约 **20.9%** 的性能收益。

通过 Profiling 可以看出：流水编排修改前后，cube MAC、AIV scalar、AIV vec 等各类 pipe 的耗时均无下降，cube、vec 的 MTE2（搬运）耗时甚至略有增加；但端到端性能有明显收益。这说明，**通过新的流水排布，我们把第 3 节的 Cube（MAC）优化与第 4 节的向量化优化真正转化成了端到端的性能收益**。需要强调，前两项优化是流水编排优化的基础——若没有它们先压低各类 pipe 的耗时，是无法取得端到端性能收益的。

---

## 6. 端到端性能

<!-- > 整合以上优化后的端到端性能数据。

| 配置 | 延迟 (ms) | 带宽利用率 | 内存节省 |
|------|-----------|-----------|----------|
| Baseline | | | |
| + SharedKV | | | |
| + VF GetKVPhyAddr | | | |
| + 核间流水 | | | | -->
当前 QSMLA 算子在 NPU 上的 MFU 可达74%，优于 DeepSeek 的 FlashMLA 仓中 H800 和 B200 的 Peak MFU [FlashMLA: Test & benchmark MLA prefill (Sparse)](https://github.com/deepseek-ai/FlashMLA#test--benchmark-mla-prefill-sparse)。
<!-- |   |Peak MFU (Prefill) |
|------|-----------|
| NPU(QSMLA)-950DT | **74.0%**| 
| GPU(FlashMLA) - H800 | 64.7% |
| GPU(FlashMLA) - B200 | 64.4% | -->

| 场景 | 规格 | 950DT | H800 | B200 |
|:---|:---|---:|---:|---:|
| **flash Decode** | gb=2496 | 43.7% | 25.1% | 19.7% | 
|            | gb=1664 | 37.6% | 22.9% | 20.2% | 
| **pro Decode** | gb=7168 | 71.9% | 35.7% | 25.3% | 
|            | gb=5376 | 66.6% | 35.1% | 24.0% | 
| **flash Prefill** | gb=16 seq=8K | 34.6% | 30.6% | 21.2% | 
|            | gb=5 seq=128K | 47.1% | 28.8% | 22.0% | 
| **pro Prefill** | gb=16 seq=8K | 49.3% | 41.7% | 29.8% | 
|            | gb=1 seq=128K | 73.9% | 41.4% | 29.8% |
| **PeakMFU (Prefill)** | QSMLA/FlashMLA | **74.0%** | 64.7% | 64.4%|

---

## 7. 附录

### 7.1 关键数据结构

> KV Cache 的内存布局结构体定义等。

### 7.2 术语表

| 术语 | 说明 |
|------|------|
| AI Core | 昇腾 NPU 中的计算核心 |
| VF | Vector Functional unit，向量功能单元 |
| GM | Global Memory，全局存储 |
| UB | Unified Buffer，统一缓冲区 |
| KV Cache | Key-Value 缓存 |
| AIC | AI Cube，矩阵计算核（执行 bmm1/bmm2） |
| AIV | AI Vector，向量计算核（产出 KV 矩阵/反量化 KV） |
| L1 | 片上 L1 缓冲（存 KV 矩阵 `l1RightBuffers`、P、Q） |
| bmm1 / bmm2 | 批量矩阵乘：bmm1 = Q·Kᵀ，bmm2 = P·V |
| KV 矩阵 | bmm 的右操作数，即反量化后的 KV（每块 128KB），由 AIV 产出 |
| v0preload | 本章优化：加深软件流水预取 KV 矩阵，掩盖核间交付延迟 |
| MFU | 理论 cube 矩阵乘耗时 / 实测内核耗时，越高越接近计算瓶颈 |