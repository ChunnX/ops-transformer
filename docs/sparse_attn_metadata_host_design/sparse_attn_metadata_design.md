# SparseAttnSharedkvMetadata 算子 Python+TileLang 方案设计文档

## 1. 项目背景

当前 `SparseAttnSharedkvMetadata` 算子是昇腾NPU平台上 `SparseAttnSharedkv` 算子的前序算子，其主要功能是生成任务分核列表，为后续Attention计算提供每个AI Core的计算任务索引。该算子目前运行在昇腾AICPU上（NPU的辅助CPU单元），通过C++实现。

本项目采用**Python+TileLang融合方案**：
- **Metadata计算**：在Python端完成（数据在CPU上，避免D2H传输）
- **Attn算子**：用TileLang实现，与Metadata计算放在同一个.py文件
- **整体流程**：无需单独的AICPU kernel，融合调用

### 1.1 方案架构

```
┌─────────────────────────────────────────────────────────────────┐
│                   Python+TileLang 融合方案                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  sparse_attn_sharedkv.py (一个文件，融合实现)                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Part 1: Python Metadata计算                                ││
│  │  ─────────────────────────                                  ││
│  │  • compute_metadata_python()                                ││
│  │  • 在CPU端计算切分信息、负载均衡                               ││
│  │  • 数据本来在CPU，无需D2H传输                                 ││
│  │  • 输出metadata[1024]                                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                          │                                      │
│                          │ H2D(metadata) ~10-30μs               │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Part 2: TileLang Attn Kernel                               ││
│  │  ─────────────────────────                                  ││
│  │  • @tilelang.jit() 定义kernel                               ││
│  │  • 读取metadata[1024]，执行Attention计算                     ││
│  │  • 运行在NPU AICore                                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                          │                                      │
│                          ▼                                      │
│                    输出结果 (NPU)                                │
│                                                                 │
│  优势: 无需单独的AICPU kernel，融合调用更简洁                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 三种方案对比

| 方案 | 数据流 | 时延 | 端口占用 |
|------|--------|------|---------|
| **原始AICPU** | CPU → H2D(输入) → NPU → AICPU → NPU | ~50-100μs | ~30-50μs |
| **Host Operator** | CPU → H2D → NPU → D2H → CPU → H2D → NPU | ~110-250μs | ~90-180μs ❌ |
| **Python方案** | CPU → Python计算 → H2D(输出) → NPU | ~40-120μs | ~10-30μs ✅ |

#### 端口占用分析

**PCIe通信端口是共享资源**，被占用时其他任务需排队等待：

```
┌─────────────────────────────────────────────────────────────┐
│                     方案1：原始AICPU                          │
├─────────────────────────────────────────────────────────────┤
│  CPU tensor → H2D(输入) → NPU → AICPU计算 → NPU              │
│  端口占用: ~30-50μs (必需的输入传输)                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  方案2：Host Operator ❌                     │
├─────────────────────────────────────────────────────────────┤
│  CPU → H2D → NPU → D2H → CPU → H2D → NPU                    │
│  端口占用: ~90-180μs (严重阻塞其他任务!)                      │
│  问题: 多了一次D2H和一次H2D，端口长时间被独占                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   方案3：Python方案 ✅                        │
├─────────────────────────────────────────────────────────────┤
│  CPU tensor → Python计算 → H2D(metadata) → NPU              │
│  端口占用: ~10-30μs (最小!)                                  │
│  优势: CPU计算期间端口空闲，其他任务可并发使用PCIe              │
└─────────────────────────────────────────────────────────────┘
```

#### 为什么Metadata计算不用TileLang？

**TileLang-Ascend是NPU Kernel DSL**，设计目标是：
- 矩阵运算、FlashAttention等计算密集型任务
- 运行在AICore/AIVCore（NPU硬件）

**Metadata计算的特点**：
- 整数运算、条件分支、负载均衡算法
- 控制密集型，不适合NPU并行计算
- 适合CPU串行执行

| 特性 | TileLang设计目标 | Metadata计算需求 |
|------|------------------|------------------|
| 计算类型 | 矩阵乘法、并行向量 | 整数运算、串行逻辑 |
| 硬件 | AICore/AIVCore | CPU |
| 并行性 | 高度并行 | 负载均衡是贪心算法（串行） |

**设计决策**：
- Metadata计算用**Python**（CPU端）
- Attn算子用**TileLang**（NPU端）
- 两者融合在同一个.py文件中

### 1.2 Python方案概述

**数据流路径**：

```
CPU tensor → Python计算(无端口占用) → H2D(metadata) → NPU → SparseAttnSharedkv
```

**关键优势**：

| 优势 | 说明 |
|------|------|
| **避免D2H传输** | 数据本来在CPU上创建，直接访问无需传输 |
| **端口占用最小** | 只有H2D输出传输(~10-30μs)，其他任务可并发 |
| **实现简单** | 纯Python+NumPy，无需AICPU开发 |
| **调试方便** | 可直接打印中间结果，便于验证 |

***

## 2. 基本块大小与架构

### 2.1 基本块定义（A5标准）

| 参数 | 值 | 说明 |
|------|-----|------|
| M_BASE_SIZE | 64 | M轴基本块，每次处理64个Q token（一个头组） |
| S2_BASE_SIZE | 128 | S2轴基本块，每次处理128个KV token |

**含义**：一次迭代计算的基本块大小为 `(64, 128)`。

### 2.2 Block数量计算

```
S2轴block数量 = ceil(s2_size / 128)

典型场景 B=4, S1=128, S2=4096:
  每batch: ceil(4096/128) = 32 blocks
  总block数: 4 × 128 × 32 = 16384 blocks
```

***

## 3. 开销计算模型

### 3.1 开销计算函数

```cpp
int64_t WinCalcCost(uint32_t basicM, uint32_t basicS2) {
    winAlignCoefM = 16U;      // M轴对齐系数
    winAlignCoefS2 = 64U;     // S2轴对齐系数
    
    // 位运算实现向上取整除法
    winAlignBasicM = (basicM + 15) >> 4;   // ceil(basicM / 16)
    winAlignBasicS2 = (basicS2 + 63) >> 6; // ceil(basicS2 / 64)
    
    return 6 * winAlignBasicM + 10 * winAlignBasicS2;
}
```

### 3.2 位运算解释

**向上取整的位运算技巧**：

```
ceil(a / b) = (a + b - 1) >> log2(b)

对于 b = 16 = 2^4:
  ceil(a / 16) = (a + 15) >> 4

对于 b = 64 = 2^6:
  ceil(a / 64) = (a + 63) >> 6
```

**位运算 vs 除法**：

| 方式 | 代码 | 性能 |
|------|------|------|
| 普通除法 | `(a + 15) / 16` | 需要除法指令，较慢 |
| 位运算 | `(a + 15) >> 4` | 移位指令，更快 |

### 3.3 系数6和10的含义

**这些是实测得到的性能经验系数**：

| 系数 | 含义 | 来源 |
|------|------|------|
| **6** | M轴每个对齐单元的相对开销 | 实测经验值，反映Q向量加载、部分累加等操作耗时 |
| **10** | S2轴每个对齐单元的相对开销 | 实测经验值，反映KV加载、矩阵乘法、累加等操作耗时 |

**为什么S2轴系数更大（10 > 6）？**

- S2轴迭代涉及更多操作：
  - 加载KV向量
  - 执行矩阵乘法（Q × K^T）
  - 计算注意力分数
  - 累加到输出
- M轴迭代相对简单：
  - 加载Q向量
  - 部分累加操作

### 3.4 开销计算示例

| 场景 | basicM | basicS2 | ceil(M/16) | ceil(S2/64) | Cost |
|------|--------|---------|-----------|------------|------|
| 正常Block(A5) | 64 | 128 | 4 | 2 | 6×4 + 10×2 = 44 |
| M尾Block | 32 | 128 | 2 | 2 | 6×2 + 10×2 = 32 |
| S2尾Block | 64 | 64 | 4 | 1 | 6×4 + 10×1 = 34 |
| 双尾Block | 32 | 64 | 2 | 1 | 6×2 + 10×1 = 22 |

***

## 4. 三级负载均衡分配策略

### 4.1 costLimit的计算

```cpp
costLimit = max(avgCost, maxS1GCost)
```

**为什么取max？**

| 情况 | avgCost | maxS1GCost | 问题 |
|------|---------|-----------|------|
| 正常 | 1000 | 200 | avgCost够用，每行都能塞进去 |
| 异常 | 1000 | **1500** | 有行太大！如果不取max，必须切分 |

**核心原因**：**保证每一行至少能完整分配给一个Core**

如果 `costLimit < maxS1GCost`：
- 最大行必须切分成多个Block
- 跨Core处理，增加同步开销
- 三级策略降级到Block级别

如果 `costLimit >= maxS1GCost`：
- 任何一行都能完整塞入一个Core
- 优先按Row分配
- 减少切分，减少同步开销

### 4.2 三级分配策略

```
┌─────────────────────────────────────────────────────────────┐
│              三级分配策略: Batch → Row → Block                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原则: 能不切分就不切分，尽量按大粒度往里塞                      │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Level 1: AssignByBatch                                 │ │
│  │ ────────────────────────                               │ │
│  │ 策略: 尽可能将完整Batch*N2分配给单个Core                  │ │
│  │ 条件: core_cost + batch_cost <= costLimit * 2           │ │
│  │ 优点: 减少跨batch边界，简化数据访问                        │ │
│  │ 适用: 所有模式                                          │ │
│  └───────────────────────────────────────────────────────┘ │
│                    │ (失败)                                 │
│                    ▼                                        │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Level 2: AssignByRow                                   │ │
│  │ ────────────────────────                               │ │
│  │ 策略: 将完整的M轴行(s1GIdx)分配给单个Core                 │ │
│  │ 条件: core_cost + row_cost <= costLimit * 2             │ │
│  │ 优点: 保持S2循环连续性，减少workspace切换                  │ │
│  │ 适用: 所有模式                                          │ │
│  └───────────────────────────────────────────────────────┘ │
│                    │ (失败)                                 │
│                    ▼                                        │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Level 3: AssignByBlock ⚠️ 仅FlashDecode模式生效          │ │
│  │ ────────────────────────                               │ │
│  │ 策略: 将单个S2 Block分配给Core                           │ │
│  │ 条件: core_cost + blk_cost <= costLimit * 2             │ │
│  │ 优点: 精细控制，支持跨Core归约                            │ │
│  │ ⚠️ 重要: Block切分只在FlashDecode模式下启用!              │ │
│  │          普通模式下不使用Block切分，避免不必要的复杂度      │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 FlashDecode模式说明

**什么时候启用Block切分？**

| 模式 | 是否启用Block切分 | 原因 |
|------|------------------|------|
| **普通Attention** | ❌ 不启用 | 无需跨Core归约，按Row分配足够 |
| **FlashDecode(FD)** | ✅ 启用 | 需要跨Core归约，Block切分支持细粒度分配 |

**FlashDecode是什么？**
- 当M轴行数较多（>某个阈值）时，启用FlashDecode优化
- 多个AIVCore并行计算，需要跨Core归约
- Block切分使得跨Core分配更灵活

**判断条件**：
```python
if num_of_fd_head > 0:  # 需要FlashDecode归约
    # 启用Level 3: AssignByBlock
    split_fd(split_result, aiv_core_num)
else:
    # 普通模式，只使用Level 1和Level 2
    # 不进行Block切分
```

### 4.4 AssignByBlock详细流程（仅FD模式）（仅FD模式）

```
┌─────────────────────────────────────────────────────────────┐
│                AssignByBlock 按块切分分配                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: cur_b, cur_row, cur_blk (当前分配位置)                 │
│        block_costs[(b, row)] = [blk0, blk1, blk2...]        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  while cur_b < batch_size:                             │ │
│  │                                                         │ │
│  │    while cur_row < s1GBaseNum[cur_b]:                  │ │
│  │                                                         │ │
│  │      while cur_blk < len(block_costs[cur_row]):        │ │
│  │        ┌─────────────────────────────────────────────┐ │ │
│  │        │ blk_cost = block_costs[cur_row][cur_blk]    │ │ │
│  │        │                                             │ │ │
│  │        │ if core_cost + blk_cost > costLimit * 2:    │ │ │
│  │        │   break  ← 当前Core满了                     │ │ │
│  │        │                                             │ │ │
│  │        │ assign_block(core, b, row, blk)             │ │ │
│  │        │ core_cost += blk_cost                       │ │ │
│  │        │ cur_blk++                                   │ │ │
│  │        └─────────────────────────────────────────────┘ │ │
│  │                                                         │ │
│  │      cur_row++  ← 当前行所有Block分配完                  │ │
│  │      cur_blk = 0                                        │ │
│  │                                                         │ │
│  │    cur_b++  ← 当前Batch所有Row分配完                     │ │
│  │    cur_row = 0                                          │ │
│  │                                                         │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  输出: 记录Core的 (bN2Start, mStart, s2Start)               │
│        记录Core的 (bN2End, mEnd, s2End)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Metadata字段含义

| 字段 | offset | 含义 | AssignByBlock时的值 |
|------|--------|------|---------------------|
| CORE_ENABLE | 0 | Core是否启用 | 1 |
| BN2_START | 1 | Batch起始索引 | batch索引 |
| M_START | 2 | Row起始索引 | row索引 |
| **S2_START** | 3 | **Block起始索引** | blk索引（关键！） |
| BN2_END | 4 | Batch结束索引 | batch索引 |
| M_END | 5 | Row结束索引 | row索引 |
| **S2_END** | 6 | **Block结束索引** | blk索引（关键！） |

**S2_START/S2_END是AssignByBlock特有的字段**，记录Block级别的切分位置。

***

## 5. 函数原型与参数

### 5.1 函数原型

```
输入参数:
├── 必需输入
│   ├── cu_seqlens_q: Q序列前缀和数组，维度[B+1]，CPU tensor
│   └── seqused_kv: KV实际有效序列长度数组，维度[B]，CPU tensor
├── 属性参数
│   ├── num_heads_q: Q头数，固定值64
│   ├── num_heads_kv: KV头数，固定值1
│   ├── head_dim: 头维度，固定值512
│   ├── batch_size: 批次大小
│   ├── max_seqlen_q: Q最大序列长度
│   ├── max_seqlen_kv: KV最大序列长度
│   ├── cmp_topk: 压缩KV稀疏token数（SWA/CFA: None, SCFA: 512/1024）
│   ├── cmp_ratio: 压缩率（SWA: None, CFA/SCFA: 4/128）
│   ├── ori_mask_mode: 原始KV mask模式，固定值4(BAND)
│   ├── ori_win_left: 滑动窗口左边界，固定值127
│   ├── ori_win_right: 滑动窗口右边界，固定值0
│   ├── layout_q: Q数据排布格式，支持BSND/TND
│   └── layout_kv: KV数据排布格式，固定PA_ND
├── 输出参数
│   └── device: 返回tensor的设备（"npu"或"cpu"）
└── 输出
    └── metadata: 分核结果数组，维度[1024]，类型INT32
```

### 5.2 三种模式

| 模式 | cmp_ratio | cmp_topk | 说明 |
|------|-----------|----------|------|
| **SWA** | None | None | Sliding Window Attention |
| **CFA** | 有值(4/128) | None | Compressed Flash Attention |
| **SCFA** | 有值(4/128) | 有值(512/1024) | Sparse Compressed Flash Attention |

***

## 6. 数据流与时延分析

### 6.1 Python方案数据流

```
┌─────────────────────────────────────────────────────────────┐
│                   Python方案数据流                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   CPU                                                       │
│   ┌─────────────┐   ┌─────────────┐                         │
│   │ cu_seqlens_q│──▶│  Python     │                         │
│   │ seqused_kv  │   │  Compute    │                         │
│   └─────────────┘   │  Metadata   │                         │
│                     │             │                         │
│                     │ ~30-90μs    │                         │
│                     │             │                         │
│                     │ ░░░░░░░░░░░░│ 端口不占用！              │
│                     │ 其他任务可   │                         │
│                     │ 并发使用PCIe│                         │
│                     └─────────────┘                         │
│                           │                                 │
│                           │ H2D (~10-30μs) ████████ 端口占用 │
│                           ▼                                 │
│   NPU                     ┌─────────────┐                   │
│                     ┌─────│  metadata   │                   │
│                     │     └─────────────┘                   │
│   ┌─────────────┐   │                                       │
│   │     q       │───┼───┐                                   │
│   └─────────────┘   │   │                                   │
│                     │   │ SparseAttnSharedkv                 │
│                     │   │ Kernel                              │
│                                                             │
│   总时延: ~40-120μs  端口占用: ~10-30μs                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 时延对比总结

| 方案 | 数据拷贝耗时 | 端口占用耗时 | CPU计算 | **总耗时** |
|------|-------------|-------------|---------|-----------|
| **原始AICPU** | ~30-50μs | ~30-50μs | ~20-50μs | **~50-100μs** |
| **Host Operator** | ~90-130μs | ~90-180μs | ~20-50μs | **~110-250μs** ❌ |
| **Python方案** | ~10-30μs | ~10-30μs | ~30-90μs | **~40-120μs** ✅ |

***

## 7. 核心数据结构

### 7.1 Metadata输出结构

```
Metadata数组布局 (总大小1024 INT32):
┌─────────────────────────────────────────────────────────────┐
│ FA Metadata部分 (36核 × 8字段 = 288 INT32)                   │
│ [0-287]: 每个AI Core的任务索引                                │
│ ├─ faMetadata[coreIdx][0]: FA_CORE_ENABLE (核启用标志)        │
│ ├─ faMetadata[coreIdx][1]: FA_BN2_START (Batch起始索引)       │
│ ├─ faMetadata[coreIdx][2]: FA_M_START (Row起始索引)           │
│ ├─ faMetadata[coreIdx][3]: FA_S2_START (Block起始索引)        │
│ ├─ faMetadata[coreIdx][4]: FA_BN2_END (Batch结束索引)         │
│ ├─ faMetadata[coreIdx][5]: FA_M_END (Row结束索引)             │
│ ├─ faMetadata[coreIdx][6]: FA_S2_END (Block结束索引)          │
│ └─ faMetadata[coreIdx][7]: FA_FIRST_FD_DATA_WORKSPACE_IDX    │
├─────────────────────────────────────────────────────────────┤
│ FD Metadata部分 (72核 × 8字段 = 576 INT32)                   │
│ [288-863]: FlashDecode归约任务索引                            │
├─────────────────────────────────────────────────────────────┤
│ 剩余空间 [864-1023]: 预留扩展                                  │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 切分信息结构

```python
split_info = {
    's1GBaseNum': np.ndarray,  # 每个Batch的M轴行数: ceil(s1*groupSize / 64)
    's2BaseNum': np.ndarray,   # 每个Batch的S2轴block数: ceil(s2 / 128)
    's1GTailSize': np.ndarray, # M轴尾块大小
    's2TailSize': np.ndarray,  # S2轴尾块大小
    'isKvSeqAllZero': bool,    # 是否全空batch
    's2BaseSize': int,         # S2基本块大小: 128
}

cost_info = {
    'bN2CostOfEachBatch': np.ndarray,     # 每个Batch的总开销
    'bN2BlockOfEachBatch': np.ndarray,    # 每个Batch的总block数
    'totalCost': int,                      # 全局总开销
    'totalBlockNum': int,                  # 全局总block数
    'maxS1GCost': int,                     # 最大单行开销 ← 用于costLimit计算
    's1GCostList': np.ndarray,            # 每行的开销列表
    'block_costs': dict,                   # 每行每个block的开销: {(b,row): [(type,blk,cost),...]}
}
```

***

## 8. 核心计算流程

### 8.1 基本块切分

| 计算项 | 公式 | 说明 |
|--------|------|------|
| M轴基本块大小 | `M_BASE_SIZE = 64` | 固定64，与Q头组对应 |
| S2轴基本块大小 | `S2_BASE_SIZE = 128` | A5标准，每次处理128个KV token |
| M轴行数 | `ceil(s1 * groupSize / 64)` | 将Q序列按头组切分 |
| S2轴block数 | `ceil(s2 / 128)` | 将KV序列按128切分 |

### 8.2 开销计算流程

```python
# Step 1: 预分配开销数组
block_costs = {}   # 块开销: {(batch, row): [('win', blk_idx, cost), ('cmp', blk_idx, cost), ...]}
row_costs = {}     # 行开销: {(batch, row): total_cost}
batch_costs = []   # batch开销: [batch_0_cost, batch_1_cost, ...]

# Step 2: 计算每行开销
for b in range(batch_size):
    batch_cost = 0
    for row in range(s1GBaseNum[b]):
        row_cost = 0
        block_costs[(b, row)] = []
        
        # 2.1 计算滑动窗口范围
        s2_first = max(0, row * 64 - win_left)
        s2_last = min(s2_size - 1, row * 64 + win_right)
        
        # 2.2 Win blocks开销
        for blk in range(0, (s2_last - s2_first) // 128 + 1):
            blk_size = min(128, s2_last - s2_first - blk*128 + 1)
            blk_cost = 6 * ceil(64/16) + 10 * ceil(blk_size/64)
            block_costs[(b, row)].append(('win', blk, blk_cost))
            row_cost += blk_cost
        
        # 2.3 Cmp blocks开销 (如果有)
        if has_cmp_kv:
            cmp_s2_size = min((s2_last+1)//cmp_ratio, cmp_topk)
            for blk in range(0, ceil(cmp_s2_size/128)):
                blk_cost = 6 * ceil(64/16) + 10 * ceil(blk_size/64)
                block_costs[(b, row)].append(('cmp', blk, blk_cost))
                row_cost += blk_cost
        
        row_costs[(b, row)] = row_cost
        batch_cost += row_cost
    
    batch_costs.append(batch_cost)
```

### 8.3 负载均衡分配

```python
# Step 1: 计算costLimit
avg_cost = total_cost / 36
cost_limit = max(avg_cost, max_row_cost)  # 保证最大行能完整塞入

# Step 2: 三级分配
cur_b = 0; cur_row = 0; cur_blk = 0

for core in range(36):
    core_cost = 0
    
    # Level 1: AssignByBatch (尽量塞整batch)
    while cur_b < batch_size:
        if core_cost + batch_costs[cur_b] <= cost_limit * 2:
            assign_whole_batch(core, cur_b)
            core_cost += batch_costs[cur_b]
            cur_b += 1
            cur_row = 0
            cur_blk = 0
        else:
            break
    
    # Level 2: AssignByRow (尽量塞整行)
    while cur_b < batch_size and cur_row < s1GBaseNum[cur_b]:
        if core_cost + row_costs[(cur_b, cur_row)] <= cost_limit * 2:
            assign_whole_row(core, cur_b, cur_row)
            core_cost += row_costs[(cur_b, cur_row)]
            cur_row += 1
            cur_blk = 0
        else:
            break
    
    # Level 3: AssignByBlock (按block切分)
    while ...:
        blk_cost = block_costs[(cur_b, cur_row)][cur_blk][2]
        if core_cost + blk_cost <= cost_limit * 2:
            assign_single_block(core, cur_b, cur_row, cur_blk)
            core_cost += blk_cost
            cur_blk += 1
        else:
            break
    
    record_assignment(core)
```

***

## 9. Python+TileLang融合实现架构

### 9.1 整体架构

**设计目标**：将Metadata计算和Attn算子融合在同一个.py文件中，无需单独的AICPU kernel。

```
┌─────────────────────────────────────────────────────────────────┐
│              sparse_attn_sharedkv.py (融合实现)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Part 1: Python Metadata计算                                ││
│  │                                                             ││
│  │  def compute_metadata_python(...):                          ││
│  │      """在CPU端计算切分信息、负载均衡"""                        ││
│  │      # 数据本来在CPU，无需D2H传输                             ││
│  │      metadata = np.zeros(1024, dtype=np.int32)              ││
│  │      # 计算切分信息、开销、负载均衡...                          ││
│  │      return torch.from_numpy(metadata).to("npu")             ││
│  └─────────────────────────────────────────────────────────────┘│
│                          │                                      │
│                          │ metadata传给TileLang kernel          │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Part 2: TileLang Attn Kernel                               ││
│  │                                                             ││
│  │  @tilelang.jit(out_idx=[-1])                                ││
│  │  def sparse_attn_kernel(q, kv, metadata, ...):              ││
│  │      """运行在NPU AICore"""                                  ││
│  │      @T.prim_func                                           ││
│  │      def main(...):                                         ││
│  │          with T.Kernel(36, is_npu=True) as (cid):           ││
│  │              # 读取metadata[1024]                            ││
│  │              # 执行Attention计算                              ││
│  │              ...                                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Part 3: 统一调用接口                                        ││
│  │                                                             ││
│  │  def sparse_attn_sharedkv(q, kv, ...):                      ││
│  │      # Step 1: Python计算metadata                            ││
│  │      metadata = compute_metadata_python(...)                 ││
│  │                                                             ││
│  │      # Step 2: TileLang执行Attn                              ││
│  │      output = sparse_attn_kernel(q, kv, metadata, ...)       ││
│  │                                                             ││
│  │      return output                                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  优势:                                                          ││
│  1. 无需单独的AICPU kernel                                      ││
│  2. Metadata计算和Attn算子融合调用                               ││
│  3. 代码结构清晰，便于维护                                       ││
│  4. 端口占用最小（只有H2D输出）                                   ││
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 文件结构

```
sparse_attn_sharedkv/
├── sparse_attn_sharedkv.py        # 融合实现（Metadata + Attn）
│   ├── compute_metadata_python()  # Python Metadata计算
│   ├── sparse_attn_kernel()       # TileLang Attn kernel
│   └── sparse_attn_sharedkv()     # 统一调用接口
├── fakecode.py                    # 伪代码参考
└── latency_analysis.md            # 时延分析
```

### 9.3 与原方案对比

| 对比项 | 原方案（AICPU） | 新方案（Python+TileLang融合） |
|--------|----------------|------------------------------|
| Metadata算子 | AICPU kernel | Python函数 |
| Attn算子 | AICore kernel | TileLang kernel |
| 文件数量 | 2个独立算子 | 1个融合文件 |
| 端口占用 | ~30-50μs | ~10-30μs |
| 维护复杂度 | 需要AICPU开发 | 纯Python+TileLang |

***

## 10. Python实现参考

详见：
- `sparse_attn_metadata_compute.py` - Metadata计算完整实现
- `fakecode.py` - 简化伪代码
- `latency_analysis.md` - 时延分析
| seqused\_kv    | \[B]   | 4×4 = 16 bytes     | 128×4 = 512 bytes     |
| **总计**         | <br /> | \~36 bytes         | \~1KB                 |

#### 4.1.2 Metadata输出传输

生成的metadata需要通过H2D传输写回NPU：

```cpp
void WriteMetadataOutput(
    aclrtStream stream,
    const int32_t* host_metadata,         // Host侧生成的metadata
    const aclTensor* metadata_tensor,     // NPU侧输出aclTensor
    uint32_t batch_size
) {
    // Step 1: 获取NPU输出tensor的设备指针
    void* device_metadata_ptr = aclGetTensorDataPtr(metadata_tensor);
    
    // Step 2: H2D传输metadata
    size_t metadata_size = SAS_META_SIZE * sizeof(int32_t); // 1024 × 4 = 4KB
    
    aclrtMemcpy(
        device_metadata_ptr,               // NPU destination
        metadata_size,
        host_metadata,                     // Host source
        metadata_size,
        ACL_MEMCPY_HOST_TO_DEVICE,
        stream
    );
    
    // Step 3: 同步确保写入完成
    aclrtSynchronizeStream(stream);
}
```

**数据量**：1024 × 4 = 4KB

#### 4.1.3 内存管理

```cpp
class SparseAttnSharedkvMetadataHost {
public:
    void AllocateHostBuffers(int64_t batch_size) {
        // 分配Host侧输入缓存
        host_cu_seqlens_q_ = new int32_t[batch_size + 1];
        host_seqused_kv_ = new int32_t[batch_size];
        
        // 分配Host侧输出缓存
        host_metadata_ = new int32_t[SAS_META_SIZE]; // 1024
    }
    
    void FreeHostBuffers() {
        delete[] host_cu_seqlens_q_;
        delete[] host_seqused_kv_;
        delete[] host_metadata_;
    }
};
```

**内存开销估算**：< 1KB（输入） + 4KB（输出） + \~1KB（临时结构） = < 6KB

***

### 4.2 TileLang实现框架

```python
# sparse_attn_sharedkv_metadata_host.py

import torch
import numpy as np
from typing import Optional, Tuple, List

class SparseAttnSharedkvMetadataHost:
    """Host端Metadata生成器"""
    
    # 常量定义
    FA_TOLERANCE_RATIO = 2
    AIC_CORE_NUM = 36
    AIV_CORE_NUM = 72
    SAS_META_SIZE = 1024
    
    # Block类型枚举
    WIN_NORMAL_BLOCK = 0
    WIN_TAIL_BLOCK = 1
    CMP_NORMAL_BLOCK = 2
    CMP_TAIL_BLOCK = 3
    
    def __init__(
        self,
        num_heads_q: int = 64,
        num_heads_kv: int = 1,
        head_dim: int = 512,
        aic_core_num: int = 36,
        aiv_core_num: int = 72
    ):
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.group_size = num_heads_q // num_heads_kv
        self.m_base_size = self.group_size
        self.s2_base_size = 512
        self.aic_core_num = aic_core_num
        self.aiv_core_num = aiv_core_num
    
    def compute(
        self,
        cu_seqlens_q: Optional[torch.Tensor],
        seqused_kv: torch.Tensor,
        batch_size: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        cmp_topk: int = 512,
        cmp_ratio: int = 4,
        ori_mask_mode: int = 4,
        cmp_mask_mode: int = 3,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        layout_q: str = "TND",
        layout_kv: str = "PA_ND",
        has_ori_kv: bool = True,
        has_cmp_kv: bool = True
    ) -> torch.Tensor:
        """主计算函数"""
        
        # Step 1: 参数校验
        self._params_check(
            num_heads_q=self.num_heads_q,
            num_heads_kv=self.num_heads_kv,
            cmp_topk=cmp_topk,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            ori_win_left=ori_win_left
        )
        
        # Step 2: 初始化序列长度
        s1_sizes = self._get_s1_sizes(cu_seqlens_q, batch_size, max_seqlen_q, layout_q)
        s2_sizes = self._get_s2_sizes(seqused_kv, batch_size, max_seqlen_kv, layout_kv)
        
        # Step 3: 计算切分信息
        split_context = self._calc_split_info(batch_size, s1_sizes, s2_sizes)
        
        # Step 4: 计算开销信息
        cost_info = self._calc_cost_info(
            split_context, 
            has_cmp_kv=has_cmp_kv,
            cmp_ratio=cmp_ratio,
            cmp_topk=cmp_topk,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right
        )
        
        # Step 5: 负载均衡分配
        split_result = self._balance_schedule(split_context, cost_info)
        
        # Step 6: FlashDecode分配
        if split_result.num_of_fd_head > 0:
            self._split_fd(split_result)
        
        # Step 7: 生成Metadata
        metadata = self._gen_metadata(split_result)
        
        return metadata
    
    def _params_check(self, **kwargs):
        """参数校验"""
        # 实现参数校验逻辑
        pass
    
    def _get_s1_sizes(self, cu_seqlens_q, batch_size, max_seqlen_q, layout_q):
        """获取每个batch的S1大小"""
        if layout_q == "TND":
            return [cu_seqlens_q[i+1] - cu_seqlens_q[i] for i in range(batch_size)]
        else:
            return [max_seqlen_q] * batch_size
    
    def _get_s2_sizes(self, seqused_kv, batch_size, max_seqlen_kv, layout_kv):
        """获取每个batch的S2大小"""
        if seqused_kv is not None:
            return seqused_kv.tolist()
        else:
            return [max_seqlen_kv] * batch_size
    
    def _calc_split_info(self, batch_size, s1_sizes, s2_sizes):
        """计算切分信息"""
        split_info = {
            's1GBaseNum': [],
            's2BaseNum': [],
            's1GTailSize': [],
            's2TailSize': [],
            'isKvSeqAllZero': True
        }
        
        for b in range(batch_size):
            s1_size = s1_sizes[b]
            s2_size = s2_sizes[b]
            
            # M轴切分
            s1_g_base_num = (s1_size * self.group_size + self.m_base_size - 1) // self.m_base_size
            s1_g_tail_size = (s1_size * self.group_size) % self.m_base_size
            
            # S2轴切分
            s2_base_num = (s2_size + self.s2_base_size - 1) // self.s2_base_size
            s2_tail_size = s2_size % self.s2_base_size
            
            split_info['s1GBaseNum'].append(s1_g_base_num)
            split_info['s2BaseNum'].append(s2_base_num)
            split_info['s1GTailSize'].append(s1_g_tail_size if s1_g_tail_size > 0 else self.m_base_size)
            split_info['s2TailSize'].append(s2_tail_size)
            
            if s1_g_base_num > 0 and s2_base_num > 0:
                split_info['isKvSeqAllZero'] = False
        
        return split_info
    
    def _calc_s2_token_range(self, s1_g_idx, batch_cache):
        """计算KV Token有效范围"""
        s1_size = batch_cache['s1Size']
        s2_size = batch_cache['s2Size']
        pre_token_left_up = batch_cache['preTokenLeftUp']
        next_token_left_up = batch_cache['nextTokenLeftUp']
        
        # 计算Q Token范围
        s1_g_first_token = s1_g_idx * self.m_base_size
        s1_g_last_token = min(
            s1_g_first_token + self.m_base_size,
            s1_size * self.group_size
        ) - 1
        
        s1_first_token = s1_g_first_token // self.group_size
        s1_last_token = s1_g_last_token // self.group_size
        
        # 计算KV Token范围
        s2_first_token = s1_first_token - pre_token_left_up
        s2_last_token = s1_last_token + next_token_left_up
        
        # 边界裁剪
        s2_first_token = max(0, min(s2_first_token, s2_size - 1))
        s2_last_token = max(0, min(s2_last_token, s2_size - 1))
        
        return (s2_first_token, s2_last_token)
    
    def _calc_cost(self, basic_m, basic_s2):
        """计算单个Block开销"""
        align_m = (basic_m + 15) // 16
        align_s2 = (basic_s2 + 63) // 64
        return 6 * align_m + 10 * align_s2
    
    def _calc_cost_info(self, split_context, has_cmp_kv, cmp_ratio, cmp_topk, 
                         ori_win_left, ori_win_right):
        """计算开销信息"""
        cost_info = {
            'bN2CostOfEachBatch': [],
            'bN2BlockOfEachBatch': [],
            'bN2LastBlockCostOfEachBatch': [],
            'totalCost': 0,
            'totalBlockNum': 0,
            'maxS1GCost': 0
        }
        
        batch_size = len(split_context['s1GBaseNum'])
        
        for b in range(batch_size):
            batch_cost, batch_block, last_block_cost, max_row_cost = \
                self._calc_batch_cost(
                    b, split_context, has_cmp_kv, cmp_ratio, cmp_topk,
                    ori_win_left, ori_win_right
                )
            
            cost_info['bN2CostOfEachBatch'].append(batch_cost)
            cost_info['bN2BlockOfEachBatch'].append(batch_block)
            cost_info['bN2LastBlockCostOfEachBatch'].append(last_block_cost)
            cost_info['totalCost'] += batch_cost * self.num_heads_kv
            cost_info['totalBlockNum'] += batch_block * self.num_heads_kv
            cost_info['maxS1GCost'] = max(cost_info['maxS1GCost'], max_row_cost)
        
        return cost_info
    
    def _calc_batch_cost(self, b_idx, split_context, has_cmp_kv, cmp_ratio, 
                          cmp_topk, ori_win_left, ori_win_right):
        """计算单个Batch的开销"""
        # 实现batch开销计算
        pass
    
    def _balance_schedule(self, split_context, cost_info):
        """负载均衡分配"""
        # 实现三层分配策略
        pass
    
    def _split_fd(self, split_result):
        """FlashDecode归约分配"""
        # 实现FD分配
        pass
    
    def _gen_metadata(self, split_result):
        """生成Metadata数组"""
        metadata = torch.zeros(self.SAS_META_SIZE, dtype=torch.int32)
        
        # FA Metadata生成
        for i in range(self.aic_core_num):
            if i < split_result['usedCoreNum']:
                metadata[i * 8 + 0] = 1  # FA_CORE_ENABLE
                metadata[i * 8 + 1] = split_result['bN2Start'][i] if i > 0 else 0
                metadata[i * 8 + 2] = split_result['gS1Start'][i] if i > 0 else 0
                metadata[i * 8 + 3] = split_result['s2Start'][i] if i > 0 else 0
                metadata[i * 8 + 4] = split_result['bN2End'][i]
                metadata[i * 8 + 5] = split_result['gS1End'][i]
                metadata[i * 8 + 6] = split_result['s2End'][i]
            else:
                metadata[i * 8 + 0] = 0  # 核禁用
        
        # FD Metadata生成
        fd_start = self.aic_core_num * 8
        for i in range(self.aiv_core_num):
            if i < split_result['fdRes']['fdUsedVecNum']:
                metadata[fd_start + i * 8 + 0] = 1  # FD_CORE_ENABLE
                # ... 其他FD字段
            else:
                metadata[fd_start + i * 8 + 0] = 0
        
        return metadata
```

### 4.3 数据访问偏移计算

```python
def calc_data_offset(self, run_info, const_info):
    """计算数据访问偏移量"""
    
    # TND场景下的偏移计算
    tnd_b_idx_offset_for_q = run_info['actualSeqQPrefixSum'] * const_info['qHeadNum'] * const_info['headDim']
    tnd_b_idx_offset_for_kv = run_info['actualSeqKVPrefixSum'] * const_info['kvHeadNum'] * const_info['headDim']
    
    # Q数据偏移
    tensor_a_offset = tnd_b_idx_offset_for_q + run_info['gS1Idx'] * const_info['headDim']
    
    # KV数据偏移 (PA_ND场景)
    if const_info['kvLayout'] == 'PA_ND':
        tensor_b_offset = tnd_b_idx_offset_for_kv + run_info['n2Idx'] * const_info['headDim']
    
    # 输出偏移
    atten_out_offset = tensor_a_offset
    
    return {
        'tensorAOffset': tensor_a_offset,
        'tensorBOffset': tensor_b_offset,
        'attenOutOffset': atten_out_offset
    }
```

***

## 5. 约束分析

### 5.1 输入约束

| 参数              | 约束条件              | 说明                    |
| --------------- | ----------------- | --------------------- |
| num\_heads\_q   | = 64              | 固定值，对应Q头数             |
| num\_heads\_kv  | = 1               | 固定值，KV共享单头            |
| head\_dim       | = 512             | 固定值，头维度               |
| cmp\_topk       | ∈ {0, 512, 1024}  | 压缩KV稀疏token数上限        |
| cmp\_ratio      | ∈ {4, 128}        | 压缩率，决定了压缩KV大小         |
| ori\_mask\_mode | = 4               | BAND模式（滑动窗口）          |
| cmp\_mask\_mode | = 3               | RIGHT\_DOWN\_CAUSAL模式 |
| ori\_win\_left  | = 127             | 滑动窗口左边界               |
| ori\_win\_right | = 0               | 滑动窗口右边界               |
| layout\_q       | ∈ {"BSND", "TND"} | Q数据排布格式               |
| layout\_kv      | = "PA\_ND"        | KV数据排布格式              |

### 5.2 输出约束

| 约束项            | 条件             | 说明             |
| -------------- | -------------- | -------------- |
| metadata大小     | = 1024 INT32   | 固定大小，与kernel解耦 |
| 数据类型           | INT32          | 与kernel解析一致    |
| FA Metadata字段数 | 8              | 每个核8个索引字段      |
| FD Metadata字段数 | 8              | 每个向量核8个字段      |
| 最大核数           | AIC=36, AIV=72 | Atlas A3规格     |

### 5.3 性能约束

| 约束项      | 建议值                  | 说明                                              |
| -------- | -------------------- | ----------------------------------------------- |
| D2H传输延迟  | \~10-50μs            | cu\_seqlens\_q + seqused\_kv (\~36 bytes @ B=4) |
| H2D传输延迟  | \~10-50μs            | metadata\[1024] (4KB)                           |
| Host计算时间 | O(B×S1×S2/blockSize) | 核心算法计算复杂度                                       |
| 总额外开销    | \~20-100μs           | D2H + Host计算 + H2D                              |
| Host并行处理 | 多线程                  | Batch数据可并行处理                                    |
| 内存开销     | < 6KB                | Host侧输入缓存 + 输出缓存 + 临时结构                         |

**性能影响分析**：

| 场景             | D2H/H2D开销  | 对整体性能影响                         |
| -------------- | ---------- | ------------------------------- |
| 小batch (B=4)   | \~20-100μs | 相对于NPU端Attention计算(通常>1ms)，影响较小 |
| 大batch (B=128) | \~20-100μs | 影响更小，传输开销相对计算开销比例降低             |
| 连续调用           | 可优化        | 通过stream异步执行减少同步开销              |

**关键优化点**：

1. 使用异步stream执行D2H/H2D传输，避免阻塞
2. Host端计算与下一次调用的D2H可并行（pipeline优化）
3. 小batch场景下，可考虑将cu\_seqlens\_q/seqused\_kv提前放入tilingData避免D2H

### 5.4 兼容性约束

1. **输出格式兼容**: metadata数组格式必须与原AICPU版本完全一致，确保后续`SparseAttnSharedkv`算子可正确解析
2. **边界条件兼容**: 空batch、零序列长度等边界情况的处理逻辑需保持一致
3. **数值精度兼容**: 开销计算使用整数运算，避免浮点误差导致分配差异

### 5.5 功能约束

| 场景          | 约束说明                                          |
| ----------- | --------------------------------------------- |
| SWA模式       | 仅使用ori\_kv，不生成cmp相关metadata                   |
| CFA模式       | 使用ori\_kv + cmp\_kv，需计算两部分开销                  |
| SCFA模式      | 使用ori\_kv + cmp\_kv + sparse\_indices，需处理稀疏索引 |
| FlashDecode | 跨核行需记录归约信息，支持多核协作                             |

***

## 6. 测试验证方案

### 6.1 单元测试

```python
def test_metadata_generation():
    """基础功能测试"""
    generator = SparseAttnSharedkvMetadataHost()
    
    # 测试数据
    batch_size = 4
    s1 = 128
    s2 = 4096
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * s1, step=s1, dtype=torch.int32)
    seqused_kv = torch.tensor([s2] * batch_size, dtype=torch.int32)
    
    metadata = generator.compute(
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        cmp_topk=512,
        cmp_ratio=4
    )
    
    # 校验metadata大小
    assert metadata.shape[0] == 1024
    
    # 校验核分配有效性
    used_core_num = count_enabled_cores(metadata)
    assert used_core_num > 0
    assert used_core_num <= generator.aic_core_num
```

### 6.2 对比验证

```python
def test_compare_with_aicpu():
    """与原AICPU版本输出对比"""
    # 生成相同输入
    input_data = generate_test_input()
    
    # Host版本
    host_metadata = host_compute(input_data)
    
    # AICPU版本 (reference)
    aicpu_metadata = aicpu_compute(input_data)
    
    # 对比输出
    assert torch.allclose(host_metadata, aicpu_metadata)
```

***

## 7. 附录

### 7.1 相关文件列表

| 文件路径                                                                          | 功能说明                      |
| ----------------------------------------------------------------------------- | ------------------------- |
| `sparse_attn_sharedkv_metadata/op_kernel_aicpu/*.cpp/.h`                      | AICPU原实现（参考用）             |
| `sparse_attn_sharedkv_metadata/op_host/sparse_attn_sharedkv_metadata_host.py` | **Host端TileLang核心实现（新增）** |
| `sparse_attn_sharedkv_metadata/op_host/op_api/*.cpp`                          | Host端API层，处理D2H/H2D传输     |
| `sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_metadata.h`              | Metadata结构定义              |
| `sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_swa_kernel.h`            | SWA kernel实现              |
| `sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.h`                  | Tiling参数定义                |

### 7.2 Host端新增文件说明

| 新增文件                                                  | 功能                          |
| ----------------------------------------------------- | --------------------------- |
| `sparse_attn_sharedkv_metadata_host.py`               | TileLang实现的Host端Metadata生成器 |
| `op_api/aclnn_sparse_attn_sharedkv_metadata_host.cpp` | Host端aclnn API，包含D2H/H2D逻辑  |

### 7.3 关键API说明

```cpp
// 获取aclTensor的设备数据指针
void* aclGetTensorDataPtr(const aclTensor* tensor);

// D2H传输
aclError aclrtMemcpy(
    void* dst,           // Host destination
    size_t destMax,
    const void* src,     // NPU source
    size_t count,
    aclrtMemcpyKind kind, // ACL_MEMCPY_DEVICE_TO_HOST
    aclrtStream stream
);

// H2D传输
aclError aclrtMemcpy(
    void* dst,           // NPU destination
    size_t destMax,
    const void* src,     // Host source
    size_t count,
    aclrtMemcpyKind kind, // ACL_MEMCPY_HOST_TO_DEVICE
    aclrtStream stream
);

// 同步stream
aclError aclrtSynchronizeStream(aclrtStream stream);
```

### 7.4 关键常量定义

```cpp
// 来自 sparse_attn_sharedkv_metadata.h
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t SAS_META_SIZE = 1024;

constexpr uint32_t FA_METADATA_SIZE = 8;
constexpr uint32_t FD_METADATA_SIZE = 8;

constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
```

---

## 8. Python端直接计算方案（推荐）

### 8.1 方案背景

通过分析实际调用场景发现：
- `cu_seqlens_q` 和 `seqused_kv` 本来在 CPU 上创建
- 用户显式调用 `.npu()` 才移到 NPU
- 直接在 Python 端计算可避免 D2H 传输

### 8.2 使用方式

```python
from sparse_attn_metadata_compute import compute_metadata
import torch
import torch_npu

# Step 1: 数据本来在CPU上创建
cu_seqlens_q = torch.tensor([0, 128, 256, 384, 512], dtype=torch.int32)  # CPU tensor
seqused_kv = torch.tensor([4096, 4096, 4096, 4096], dtype=torch.int32)    # CPU tensor

# Step 2: Python端计算Metadata（无需D2H）
metadata = compute_metadata(
    cu_seqlens_q=cu_seqlens_q,
    seqused_kv=seqused_kv,
    num_heads_q=64,
    num_heads_kv=1,
    head_dim=512,
    batch_size=4,
    max_seqlen_q=128,
    max_seqlen_kv=4096,
    cmp_topk=512,
    cmp_ratio=4,
)

# Step 3: 只需一次H2D传输
metadata_npu = metadata.to("npu")

# Step 4: 调用SparseAttnSharedkv
output = torch_npu.npu_sparse_attn_sharedkv(
    query=query_npu,
    ori_kv=ori_kv_npu,
    metadata=metadata_npu,
    ...
)
```

### 8.3 性能对比

| 方案 | 延迟组成 | 总延迟 |
|------|----------|--------|
| **AICPU原方案** | 算子执行 | ~10-50μs |
| **Host算子方案** | Launch + D2H + 计算 + Sync + H2D + Sync | ~50-250μs |
| **Python端方案** | Python计算 + H2D | ~20-100μs |

Python端方案比Host算子方案减少约 50-150μs。

### 8.4 实现文件

| 文件路径 | 功能 |
|----------|------|
| `sparse_attn_metadata_compute.py` | Python端Metadata计算实现 |

### 8.5 调用示例对比

**原方案（需要显式传输）**：
```python
# 用户需要手动传输数据到NPU
cu_seqlens_q_npu = cu_seqlens_q.npu()
seqused_kv_npu = seqused_kv.npu()

# 调用算子，算子内部再D2H取回数据计算
metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
    cu_seqlens_q=cu_seqlens_q_npu,
    seqused_kv=seqused_kv_npu,
    ...
)
```

**Python端方案（无需传输）**：
```python
# 数据本来在CPU，直接计算
metadata = compute_metadata(
    cu_seqlens_q=cu_seqlens_q,  # CPU tensor，无需传输
    seqused_kv=seqused_kv,      # CPU tensor，无需传输
    ...
)

# 只需传输结果
metadata_npu = metadata.to("npu")
```

