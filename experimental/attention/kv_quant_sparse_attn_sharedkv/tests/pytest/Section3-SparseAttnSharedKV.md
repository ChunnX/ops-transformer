# SparseAttnSharedKV 算子技术细节

## 1. 概述

> 简述 SparseAttnSharedKV 算子的目标、应用场景及核心思想。

- **算子功能**：
- **输入/输出**：
- **核心优化点**：

---

## 2. Chromosomal Crossover：双 AI Core 共享 KV Cache

### 2.1 动机与问题

> 为什么需要两个 AI Core 访问同一段 KV Cache？解决了什么瓶颈？

### 2.2 地址映射与访存策略

> 两个 AI Core 如何寻址到同一块物理 KV Cache？GM/UB 上的数据排布方式。

![Chromosomal Crossover 示意图](fig/chromosomal_crossover.png)

### 2.3 同步与一致性

> 两个 AI Core 之间的同步机制（barrier / event / flag），如何保证读写的正确性。

### 2.4 性能收益分析

> 相比每个 AI Core 各自持有 KV Cache 副本，共享方案节省的内存带宽 / HBM 容量，以及可能引入的开销。

---

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

### 4.1 多核划分策略

> Sequence Length / Head / Batch 等维度如何划分到多个 AI Core。

### 4.2 流水阶段定义

> 描述各流水阶段（如 LoadKV → Compute → Store）及其依赖关系。

![核间流水排布](fig/inter_core_pipeline.png)

### 4.3 流水排布时序图

> 各 AI Core 在时间轴上的任务分配，展示 overlap 情况。

![流水时序图](fig/pipeline_timeline.png)

### 4.4 负载均衡与调优

> 如何处理尾部不均衡、Bank Conflict、通信隐藏等实际问题。

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