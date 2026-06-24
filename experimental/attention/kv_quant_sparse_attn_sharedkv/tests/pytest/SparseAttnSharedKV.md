# SparseAttnSharedKV 算子技术细节（把名字改为 Compressed Sparse Attention）

## 1. 概述

> 简述 SparseAttnSharedKV 算子的目标、应用场景及核心思想。

- **算子功能**：
- **输入/输出**：
- **核心优化点**：

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

## 3. GetKVPhyAddr：VF 向量化实现

### 3.1 动机与功能

> GetKVPhyAddr 的职责：根据逻辑索引计算 KV Cache 的物理地址。

PagedAttention 下,KV cache 按 block 物理分页存放,逻辑上连续的 KV 序列在物理上被 `block_table` 重映射到任意物理块。稀疏注意力又只选中部分 block(`sparse_indices`)。因此每个被选中的逻辑索引 `s2Idx` 需要换算成真实物理地址:

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

### 3.2 标量实现 vs 向量化实现

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

### 3.3 向量化算法细节

> 具体 VF 指令序列、掩码处理、边界条件（非对齐长度）等。

<!-- ![GetKVPhyAddr 向量化流程](fig/getkvphyaddr_vf.png) -->
### 3.3.1 常量与排布

```cpp
s2NumPerLoop   = 128   // 每 loop 处理 128 个逻辑索引
s2NumPerReg    = 64    // 每寄存器 64 个 int32 = 256B
outOffsetPerLoop = 256 // 输出 int64 → 翻倍:128 索引 × 2(L/H) = 256 个 uint32
outOffsetPerReg  = 128 // 64 索引 × 2
```
每 loop 用 `_1`(前 64)、`_2`(后 64)两套寄存器并行同一套运算。

### 3.3.2 VF 指令序列(每 loop,以 `_1` 为例)

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

循环外预置:`Duplicate(vreg_zero, 0)`、`Duplicate(vreg_kvStride, kvStride)`。

### 3.3.3 掩码处理(两类 MaskReg,作用不同)

1. **谓词掩码** `preg_all_b32 = CreateMask<MaskPattern::ALL>()`:每条算术指令的 lane 使能, 当前实现**全开**(所有 64 lane 都算)。
2. **进位掩码** `add_carry_l_1` / `add_carry_h_1`:**不是 lane 谓词,而是 64 位加法的进位**。
   `Add` 把低 32 位相加产生的进位写入 `add_carry_l_1`,`AddC` 再把它吃进高 32 位 (`高 = overflow + 0 + carry`)。这是用 MaskReg 承载 carry 实现寄存器内 int64 加法的关键。

### 3.3.4 边界条件(非对齐长度 `total % 128 ≠ 0`)

首先计算循环参数:
```cpp
s2Loop = (total + 128 - 1) / 128;     // 向上取整
s2Tail = total - (s2Loop - 1) * 128;  // 尾块有效数
```
当前 VFImpl 的处理方式:
- **VF 内核按 `s2Loop` 跑满**,每 loop 都用 `preg_all`(128 个)→ **尾块会多算最多 127 个**地址到 UB 缓冲。
- 输入 `sparseIdxUb` 由 `DataCopyPad` 零填充,尾块读到的是 0,算出的多余地址无害。
- **正确性靠"写出时按 `total` 截断"保证**:`CopyPhyAddrToGm(..., total, ...)`(`blockLen = total*sizeof(int32)*2/32`)只把前 `total` 个有效地址搬回 GM,多算的尾块不写出。

### 3.4 性能数据

> 向量化前后的 cycles 对比 / 加速比。

---

## 4. 核间流水排布

### 4.1 动机与问题

第 3 节把地址计算向量化后，scalar 耗时大幅下降，但端到端总耗时并未随之明显下降——MFU 依然偏低。参考下表，可见仅压低 scalar 不足以兑现端到端收益：

| N=128 SCFA prefill（S1=1024, S2=8192） | AIV scalar (µs) | 端到端耗时 (µs) | MFU |
|---|---:|---:|---:|
| 向量化前 | 649.5 | 708.5 | 53.8% |
| 向量化后 | 428.1 | 704.2 | 54.1% |
| **Δ** | **−34%** | **−0.6%** | **+0.3%** |

scalar 耗时缩减约三分之一（−221µs），但端到端耗时与 MFU 没有太大变化。

本节从**核间流水排布**入手——通过加深流水与改变核间同步的方式，把端到端耗时降下来。

### 4.2 v0preload：加深流水（N=128）

每个 S2 基本块都要用到一份 **bf16 的反量化 KV**（`s2BaseSize=128 × D=512`，合计 **128KB**），由 AIV 的 `ProcessVec0` 算出后从 UB 搬到 L1 右矩阵缓冲 `l1RightBuffers`。这份右矩阵从产出一直存活到被 bmm2 消费——既作为 bmm1 的右矩阵（当 K，算 Q·Kᵀ），又作为 bmm2 的右矩阵（当 V，算 P·V）。

**修改前（3 级流水）**：`PRELOAD_NUM = 2`、`runInfo[3]`，同一时刻最多 3 份任务在执行。取数（`IterateLoadQK`）与产出（`ProcessVec0`）处于同一个流水级、未被前置预取——AIC 必须等 `ProcessVec0`（AIV 产出右矩阵）完成并跨核同步后才能取数：

```
vec0(t) + load(t)   // 产出右矩阵后同级取数（load 未前置）
bmm1(t-1)           // Q·Kᵀ
bmm2(t-2)           // P·V
```

**修改后（4 级流水）**：v0preload 把 `PRELOAD_NUM` 提到 3、`runInfo[4]`，多预取一拍，同一时刻 4 份右矩阵分处四个阶段：

```
vec0(t)        // AIV 算出第 t 块右矩阵（ProcessVec0）
load(t-1)      // AIC 预取第 t-1 块（IterateLoadQK）
bmm1(t-2)      // Q·Kᵀ
bmm2(t-3)      // P·V
```

这样 AIC 可以提前一个 tile 发起 `IterateLoadQK`。注意：虽然 4 份任务在执行，但任一时刻 L1 上只需保留 3 份右矩阵——处于 load / bmm1 / bmm2 三个阶段的那 3 份需驻留 L1 供 cube 读取，而处于 vec0 阶段（正在产出）的第 4 份此刻还在 UB / GM 的 `v0ResGm`、尚未进 L1，因此 `l1RightBuffers` 的 3 个槽（`taskIdMod3` 轮转）正好够用。

N=128 的右矩阵本就经 GM 在两个 AIC 之间传递（G=128 跨 2 个 AIC，各算 64 头），不涉及 L1 容量问题。

### 4.3 推广到 N=64：核间同步重构

N=128 因右矩阵本就经 GM 中转，第 4 份天然能放进 GM；N=64 原先由 AIV 把右矩阵从 UB 直接搬到 L1（`CopyOutKvUb2L1`），没有走 GM 进行数据中转，加深到 4 级就得给第 4 份找落脚处。

但该场景下 **L1 已被占满**——L1 共 512KB，分配为：右矩阵区 `l1RightBuffers` `3×128KB = 384KB`、P 缓冲 `l1P` `2×16KB = 32KB`、Q 缓冲 `l1Q` `96KB`，合计 `384 + 32 + 96 = 512KB`，放不下第 4 个 128KB 的右矩阵槽。

**为了在 N=64 使能该优化，需要把右矩阵也改走 GM 中转**：`ProcessVec0` 算出后由 `CopyOutKvUb2Gm` 写到 GM 上的 `v0ResGm`，AIC 用时再经 `IterateLoadQK` 从 GM `DataCopy` 回某个 `l1RightBuffers` 槽——用充裕的 GM 承接放不进 L1 的第 4 份。3 个 L1 槽仍按 `taskIdMod3` 复用（每份右矩阵在槽内存活 ≤3 轮：load→bmm1→bmm2），L1 分配不变。

随之**核间同步从「L1 跨核」改为「GM 跨核」**：barrier 由「右矩阵写完 L1」（`CopyOutKvUb2L1` 的 `CROSS_CORE_SYNC_FORWARD`，AIC 在 `MTE1` 等待）前移到「AIC 预取」（`CopyOutKvUb2Gm` 的 `CROSS_CORE_SYNC_BACKWARD`，AIC 在 `IterateLoadQK` 内以 `MTE2` 等待后再 `DataCopy`），使 AIC 能提前发起 GM→L1 预取、与前序 cube 计算 overlap。这条改写同时统一了代码：N=64 与 N=128 不再按 `IS_SPLIT_G` 分作两条路径，双主循环 `ProcessMainLoopN64`/`ProcessMainLoopN128` 也合并为单一 `ProcessMainLoop`。

### 4.4 性能数据

**N=128**：SCFA 平均加速 **25.3%**、CFA **27.7%**（整体 MFU 49.1% → 65.7% / 67.9%）；SWA 性能持平。

| 典型例子（N=128, SCFA） | before(µs) | after(µs) | 加速 | MFU 前→后 |
|---|---:|---:|---:|---:|
| prefill | 10212.13 | 7734.07 | 24.3% | 56.0%→74.0% |
| decode | 132.69 | 101.39 | 23.6% | 50.5%→66.1% |

**N=64**：SCFA 平均加速 **24.4%**、CFA **42.8%**、SWA **19.5%**（整体 MFU 33.5 / 36.6 / 28.8% → 44.3 / 64.1 / 35.8%）。

| 典型例子（N=64, SCFA） | before(µs) | after(µs) | 加速 | MFU 前→后 |
|---|---:|---:|---:|---:|
| prefill | 8422.47 | 6174.28 | 26.7% | 34.0%→46.3% |
| decode | 113.69 | 79.15 | 30.4% | 29.5%→42.3% |

**收益来自核间流水的改善。** 通过 profiling 对比修改前后各流水段的耗时：cube MAC、AIV scalar、AIV vec 等各计算段均无下降，甚至 cube、vec 的 MTE2（搬运）耗时还略有增加；但端到端耗时仍明显下降——说明收益来自流水把各段更好地 overlap 掩盖。

---

## 5. 端到端性能

> 整合以上优化后的端到端性能数据。

| 配置 | 延迟 (ms) | 带宽利用率 | 内存节省 |
|------|-----------|-----------|----------|
| Baseline | | | |
| + SharedKV | | | |
| + VF GetKVPhyAddr | | | |
| + 核间流水 | | | |

---

## 6. 附录

### 6.1 关键数据结构

> KV Cache 的内存布局结构体定义等。

### 6.2 术语表

| 术语 | 说明 |
|------|------|
| AI Core | 昇腾 NPU 中的计算核心 |
| VF | Vector Functional unit，向量功能单元 |
| GM | Global Memory，全局存储 |
| UB | Unified Buffer，统一缓冲区 |
| KV Cache | Key-Value 缓存 |
| AIC | AI Cube，矩阵计算核（执行 bmm1/bmm2） |
| AIV | AI Vector，向量计算核（产出右矩阵/反量化 KV） |
| L1 | 片上 L1 缓冲（存右矩阵 `l1RightBuffers`、P、Q） |
| bmm1 / bmm2 | 批量矩阵乘：bmm1 = Q·Kᵀ，bmm2 = P·V |
| 右矩阵 | bmm 的右操作数，即反量化后的 KV（每块 128KB），由 AIV 产出 |
| v0preload | 本章优化：加深软件流水预取右矩阵，掩盖核间交付延迟 |
| MFU | 理论 cube 矩阵乘耗时 / 实测内核耗时，越高越接近计算瓶颈 |