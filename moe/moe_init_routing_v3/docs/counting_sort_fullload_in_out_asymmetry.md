# moe_init_routing_v3 CountingSort FullLoad 模板 —— x 搬入 vs expandedX 搬出不对称分析

## 1. 现象概述

在 `MoeV3CountingSortFullLoadUnquantized`（A5 CountingSort FullLoad 非量子类，模板1）中存在一个明显的"搬入搬出不对称"现象：

- **x 搬入（LoadXBackground）**：一次性、连续、整块地从 GM 搬入 UB，所有 token 行紧凑排布。
- **expandedX 搬出（GatherOneRow）**：按行逐行、多次、离散地从 UB 搬出到 GM，每行一次 `DataCopyPad`。

这两者在搬运次数、连续性、流水效率上差异显著，根本原因在于 **x 在 GM 上是连续的，而 expandedX 在 GM 上每行不连续**。

---

## 2. 搬入侧分析：x 连续整块搬入

### 2.1 代码位置
`op_kernel/arch35/moe_v3_counting_sort_full_load_base.h` → `LoadXBackground()`

### 2.2 关键代码
```cpp
DataCopyExtParams copyParams{
    static_cast<uint16_t>(coreTokenNum_),   // blockCount：本核 token 数
    static_cast<uint32_t>(cols_ * sizeof(T)),// blockLen：每行字节数
    0, 0, 0                                 // srcStride/dstStride = 0
};
DataCopyPad(xLocal, xGm[coreTokenStart_ * cols_], copyParams, padParams);
```

### 2.3 连续性来源
- **源 GM（xGm）**：`xGm[coreTokenStart_ * cols_]` 起始，`coreTokenNum_` 行、每行 `cols_` 个元素，在 GM 上**物理连续**（token × cols 的二维行主序紧凑布局）。
- **目的 UB（xLocal）**：`blockCount = coreTokenNum_`，`blockLen = cols_*sizeof(T)`，`srcStride = 0 / dstStride = 0`。
  - `dstStride` 单位是 32B datablock，`colsAligned_*sizeof(T)` 本就是 `CeilAlign(cols_*sizeof(T), 32)`，每行天然紧邻下一个 32B block，故 `dstStride = 0` 表示"行与行无间隙"。
- **结果**：一次 `DataCopyPad` 把本核所有 token 整块搬入，UB 内 `xLocal[tokenRow * colsAligned_]` 即可按行索引。

### 2.4 性能特征
- **MTE2 指令数 = 1**（整块搬入）。
- **地址全部连续**：GM 侧可命中 burst，UB 侧无 pad gap。
- **后台加载**：`LoadXBackground` 不在原地 wait，与 Phase A 的 Filter/Count 并行，在 `WaitXLoadCommon()` 才同步，隐藏了搬入延迟。

---

## 3. 搬出侧分析：expandedX 逐行离散搬出

### 3.1 代码位置
`op_kernel/arch35/moe_v3_counting_sort_full_load_unquantized.h` → `GatherOneRow()` / `GatherAndWrite()`

### 3.2 关键代码
```cpp
// GatherAndWrite：遍历每个被过滤到的 (token, expert) 对
for (int64_t fIdx = 0; fIdx < this->filteredCount_; fIdx++) {
    ...
    GatherOneRow(newPos, tokenRow, origFlatIdx);
}

// GatherOneRow：每行一次 DataCopyPad
DataCopyExtParams xCopyParams{
    static_cast<uint16_t>(1),                  // blockCount = 1：仅 1 行
    static_cast<uint32_t>(this->cols_ * sizeof(T)),
    0, 0, 0
};
DataCopyPad(expandedXGm[static_cast<int64_t>(newPos) * this->cols_],
            xLocal[tokenRow * this->colsAligned_],
            xCopyParams, padParams);
```

### 3.3 不连续性根因
`expandedXGm` 的目标地址由 `newPos` 决定，而 `newPos` 来自计数排序的映射：

- **dropPad 模式**：`newPos = (expertOffset + expertStart_) * expertCapacity_ + slot`
  - 行地址 = `(expertOffset + expertStart_) * expertCapacity_ + slot`，按"专家分区 + 槽位"组织。
  - 同一个 token 的多次出现会落到不同 expert 分区，**相邻两次写地址跨度 = expertCapacity_ 或跨专家分区**，完全非连续。
- **dropless 模式**：`newPos = slot`（全局累积槽位）
  - 由 prefix sum 决定，不同 token、不同 expert 的写地址按计数顺序递增，但**本核处理的 token 顺序与 newPos 递增顺序无对齐关系**，仍是离散跳跃。

**核心结论**：expandedX 的 GM 布局是"按专家桶排（expert-major）"，而过滤后的 token 遍历顺序是"按原始 token 顺序（token-major）"。两种顺序正交，导致每次写出地址在 GM 上跳跃，**无法合并成一次连续 DataCopy**。

### 3.4 性能特征
- **MTE3 指令数 = filteredCount_**（每个有效 (token, expert) 对一次）。
- **每行单独搬**：blockCount=1，无法利用 burst，MTE3 开销随 filteredCount_ 线性增长。
- **地址离散**：GM 侧写地址跳跃，可能触发多次 cache line 分配 / writeback。

---

## 4. 不对称的根因总结

| 维度 | x 搬入 | expandedX 搬出 |
|------|--------|-----------------|
| GM 布局 | token-major，物理连续 | expert-major（dropPad）或 prefix-sum 顺序（dropless），非连续 |
| 搬运粒度 | 整块（coreTokenNum_ 行） | 单行（1 行） |
| DataCopy 次数 | 1 | filteredCount_（可达 coreTokenNum_ × k_） |
| 源/目的地址连续性 | 源连续 + 目的连续 | 源连续（xLocal 行内）+ 目的离散 |
| 流水并行 | 后台加载，与 Filter 并行 | 必须在 Filter/Count/Offset 之后串行执行 |

**根本原因**：搬入是"读取原始输入"，原始输入天然按 token 顺序连续存储；搬出是"按计数排序结果重排写出"，重排后的目标位置由 (expert, slot) 决定，与源 token 顺序无对应关系，**GM 上每行目标地址不连续**，因此只能逐行搬出。

---

## 5. 为什么不能优化成连续搬出？

### 5.1 目标地址本就不连续
即使 xLocal 在 UB 内连续，expandedXGm 的写地址由 `newPos` 决定，`newPos` 在 token 遍历序列中是跳跃的。除非改变 expandedX 的输出语义（不再按专家桶排），否则写地址无法连续。

### 5.2 token → expert 是多对多映射
一个 token 可被路由到多个专家（k_>1），同一 token 行会被搬到多个不同 newPos；不同 token 也可能搬到同一专家分区相邻槽位。这种多对多映射使搬出天然是 gather/scatter 模式，无法用单次连续 DataCopy 表达。

### 5.3 UB 内可优化空间有限
UB 内 xLocal 已经是连续的，`xLocal[tokenRow * colsAligned_]` 行索引连续。瓶颈不在 UB 侧，而在 GM 写侧的离散性。若要合并写，需要在搬出前在 UB 内按 newPos 重排——但重排本身的向量 gather 开销可能超过节省的 MTE3 次数，且需额外 UB 空间。

### 5.4 现有设计的合理取舍
- 搬入用 1 次大块 DataCopy 最大化 MTE2 效率；
- 搬出用 filteredCount_ 次小块 DataCopyPad 换取正确性（每行独立 newPos）；
- 非连续写不可避免，当前以"行粒度 DataCopyPad + 32B 对齐 pad"为最小可搬运单位，是正确性与效率的折中。

---

## 6. 潜在优化方向（仅记录，不含实现）

| 方向 | 思路 | 代价 / 风险 |
|------|------|-------------|
| 同专家批合并 | 在 GatherAndWrite 中按 expertOffset 排序后遍历，同专家连续行可合并 blockCount>1 | 需额外排序，UB 内重排开销 |
| UB 内重排后连续写 | 先在 UB 按 newPos 重排 expandedX，再一次 DataCopy | UB 空间压力增大，重排 gather 开销 |
| 多核协作写同一专家分区 | 让同一专家的写由固定核负责，减少跨核地址跳跃 | 需重构核间切分策略，影响面大 |
| 放宽 expandedX 布局 | 若上层允许 token-major 输出，则可连续写 | 改变算子输出语义，下游依赖需同步改 |

当前实现选择了"搬入最大化连续、搬出接受离散"的路径，在 dropPad/dropless 两种模式下都保证正确性，优化空间主要在搬出侧的批合并与重排，需结合实际 filteredCount_ 分布评估收益。

---

## 7. 结论

x 搬入连续、expandedX 搬出离散的不对称，**本质是 MOE 路由重排语义在 GM 布局上的投影**：

- 输入 x 按 token 顺序连续存储 → 搬入可整块连续。
- 输出 expandedX 按专家桶排存储，且 token→专家是多对多映射 → 搬出目标地址在 token 遍历序列中跳跃，无法连续。

这不是实现缺陷，而是 CountingSort FullLoad 路径在"保证输出语义正确"约束下的固有特征。任何优化都需在"不改变 expandedX 输出语义"前提下，从搬出侧的批合并或重排入手。
