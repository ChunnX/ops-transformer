# SparseAttnSharedkvMetadata 算子 Host 端重设计方案文档

## 1. 项目背景

当前 `SparseAttnSharedkvMetadata` 算子是昇腾NPU平台上 `SparseAttnSharedkv` 算子的前序算子，其主要功能是生成任务分核列表，为后续Attention计算提供每个AI Core的计算任务索引。该算子目前运行在昇腾AICPU上（NPU的辅助CPU单元），通过AscendC语言实现。

本项目要求将该算子迁移到Host端CPU执行，使用TileLang语言重写，实现相同的数据切分优化功能，不依赖AICPU硬件特性。

### 1.1 迁移方案概述

**数据流路径**：

```
输入张量(NPU) → D2H传输 → Host CPU计算 → H2D传输 → metadata(NPU) → SparseAttnSharedkv
```

**关键设计要点**：

| 设计项        | 说明                                                                                                     |
| ---------- | ------------------------------------------------------------------------------------------------------ |
| **张量输入获取** | Host CPU无法直接访问NPU内存，必须通过`aclrtMemcpy(ACL_MEMCPY_DEVICE_TO_HOST)`将`cu_seqlens_q`和`seqused_kv`从NPU传至Host |
| **核心算法执行** | Metadata生成在Host端执行，使用TileLang实现切分、负载均衡、分配算法                                                            |
| **输出传输**   | 生成的`metadata[1024]`通过`aclrtMemcpy(ACL_MEMCPY_HOST_TO_DEVICE)`传回NPU的metadata tensor                     |
| **数据量估算**  | 输入\~36 bytes (B=4)，输出4KB，PCIe传输延迟约10-50μs                                                              |

**Host端 vs AICPU的关键区别**：

| 对比项      | AICPU                        | Host CPU             |
| -------- | ---------------------------- | -------------------- |
| 运行位置     | NPU设备上                       | 主机CPU                |
| Tensor类型 | `aicpu::Tensor`              | `aclTensor`          |
| 数据访问     | `tensor->GetData()`直接读取NPU内存 | **必须D2H传输后才能读取数据内容** |
| 输出写入     | 直接写入NPU内存                    | 需H2D传输写入             |

> **重要说明**：Host端`aclTensor`只包含元信息（shape、dtype、format等），其数据指针指向NPU设备内存。Host CPU无法直接通过该指针读取数据内容，必须使用Ascend API进行D2H传输。

***

## 2. 函数原型与流程

### 2.1 函数原型

```
输入参数:
├── 必需输入
│   ├── cu_seqlens_q (可选): Q序列前缀和数组，维度[B+1]
│   └── seqused_kv: KV实际有效序列长度数组，维度[B]
├── 属性参数
│   ├── num_heads_q: Q头数，固定值64
│   ├── num_heads_kv: KV头数，固定值1
│   ├── head_dim: 头维度，固定值512
│   ├── batch_size: 批次大小
│   ├── max_seqlen_q: Q最大序列长度
│   ├── max_seqlen_kv: KV最大序列长度
│   ├── cmp_topk: 压缩KV稀疏token数，支持512或1024
│   ├── cmp_ratio: 压缩率，支持4或128
│   ├── ori_mask_mode: 原始KV mask模式，固定值4(BAND)
│   ├── cmp_mask_mode: 压缩KV mask模式，固定值3(RIGHT_DOWN_CAUSAL)
│   ├── ori_win_left: 滑动窗口左边界，固定值127
│   ├── ori_win_right: 滑动窗口右边界，固定值0
│   ├── layout_q: Q数据排布格式，支持BSND/TND
│   └── layout_kv: KV数据排布格式，固定PA_ND
└── 输出
    └── metadata: 分核结果数组，维度[1024]，类型INT32
```

### 2.2 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Host端SparseAttnSharedkvMetadata                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Step 1: 参数获取 (D2H传输)                        │    │
│  │                                                                     │    │
│  │  ┌──────────────────┐                                               │    │
│  │  │ NPU设备内存       │   aclrtMemcpy(D2H)    ┌──────────────────┐    │    │
│  │  │ cu_seqlens_q[B+1]│ ──────────────────▶   │ Host内存         │    │    │
│  │  │ seqused_kv[B]    │    ~10-50μs           │ host_cu_seqlens  │    │   │
│  │  │ (aclTensor指针)  │                       │ host_seqused_kv  │     │   │
│  │  └──────────────────┘                       └──────────────────┘    │   │
│  │                                                                     │   │
│  │  属性参数(num_heads_q, batch_size等): 直接传递，无需传输               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Step 2-6: 核心计算 (Host CPU)                     │   │
│  │                                                                     │   │
│  │  ┌──────────┐   ┌──────────────┐   ┌─────────────┐                  │   │
│  │  │ 参数校验  │──▶│ 切分信息计算  │──▶│ 开销估算     │                  │   │
│  │  └──────────┘   └──────────────┘   └─────────────┘                  │   │
│  │       │              │                   │                          │   │
│  │       ▼              ▼                   ▼                          │   │
│  │  ┌──────────────────────────────────────────────────┐               │   │
│  │  │              负载均衡分配(三层策略)                │               │   │
│  │  │  ┌─────────────┐  ┌───────────┐  ┌───────────┐   │               │   │
│  │  │  │AssignByBatch│─▶│AssignByRow│─▶│AssignByBlock│ │               │   │
│  │  │  └─────────────┘  └───────────┘  └───────────┘   │               │   │
│  │  └──────────────────────────────────────────────────┘               │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  ┌─────────────────────┐                                            │   │
│  │  │   Metadata生成      │                                            │   │
│  │  │   ├─ FA Metadata    │                                            │   │
│  │  │   └─ FD Metadata    │                                            │   │
│  │  └─────────────────────┘                                            │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  host_metadata[1024] (Host内存)                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Step 7: 输出传输 (H2D传输)                        │   │
│  │                                                                     │   │
│  │  ┌──────────────────┐                                               │   │
│  │  │ Host内存         │   aclrtMemcpy(H2D)   ┌──────────────────┐      │   │
│  │  │ host_metadata    │ ──────────────────▶  │ NPU设备内存       │     │   │
│  │  │ [1024] INT32     │    ~10-50μs          │ metadata tensor  │     │   │
│  │  │ (4KB)            │                      │ (算子输出)        │     │   │
│  │  └──────────────────┘                      └──────────────────┘     │   │
│  │                                                                     │   │
│  │  aclrtSynchronizeStream(stream) 确保传输完成                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│                    SparseAttnSharedkv (NPU AICore)                         │
│                    ┌─────────────────────────────────────────────────────┐ │
│                    │ 读取metadata[1024]，执行Attention计算                │  │
│                    └─────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 数据传输详细说明

| 传输阶段     | 数据               | 方向       | API                                      | 数据量           | 延迟估算      |
| -------- | ---------------- | -------- | ---------------------------------------- | ------------- | --------- |
| D2H (输入) | `cu_seqlens_q`   | NPU→Host | `aclrtMemcpy(ACL_MEMCPY_DEVICE_TO_HOST)` | (B+1)×4 bytes | \~10-50μs |
| D2H (输入) | `seqused_kv`     | NPU→Host | `aclrtMemcpy(ACL_MEMCPY_DEVICE_TO_HOST)` | B×4 bytes     | \~10-50μs |
| H2D (输出) | `metadata[1024]` | Host→NPU | `aclrtMemcpy(ACL_MEMCPY_HOST_TO_DEVICE)` | 4KB           | \~10-50μs |

> **总传输开销**：约20-100μs（两次传输 + 同步）

***

## 3. 具体设计

### 3.1 核心数据结构定义

#### 3.1.1 Metadata输出结构

```
Metadata数组布局 (总大小1024 INT32):
┌─────────────────────────────────────────────────────────────────┐
│ FA Metadata部分 (36核 × 8字段 = 288 INT32)                       │
│ [0-287]: 每个AI Core的任务索引                                   │
│ ├─ faMetadata[coreIdx][0]: FA_CORE_ENABLE (核启用标志)           │
│ ├─ faMetadata[coreIdx][1]: FA_BN2_START (Batch*N2起始索引)       │
│ ├─ faMetadata[coreIdx][2]: FA_M_START (M轴起始索引)              │
│ ├─ faMetadata[coreIdx][3]: FA_S2_START (S2轴起始索引)            │
│ ├─ faMetadata[coreIdx][4]: FA_BN2_END (Batch*N2结束索引)         │
│ ├─ faMetadata[coreIdx][5]: FA_M_END (M轴结束索引)                │
│ ├─ faMetadata[coreIdx][6]: FA_S2_END (S2轴结束索引)              │
│ └─ faMetadata[coreIdx][7]: FA_FIRST_FD_DATA_WORKSPACE_IDX       │
├─────────────────────────────────────────────────────────────────┤
│ FD Metadata部分 (72核 × 8字段 = 576 INT32)                       │
│ [288-863]: FlashDecode归约任务索引                               │
│ ├─ fdMetadata[vecIdx][0]: FD_CORE_ENABLE                        │
│ ├─ fdMetadata[vecIdx][1]: FD_BN2_IDX                            │
│ ├─ fdMetadata[vecIdx][2]: FD_M_IDX                              │
│ ├─ fdMetadata[vecIdx][3]: FD_WORKSPACE_IDX                      │
│ ├─ fdMetadata[vecIdx][4]: FD_WORKSPACE_NUM                      │
│ ├─ fdMetadata[vecIdx][5]: FD_M_START                            │
│ └─ fdMetadata[vecIdx][6]: FD_M_NUM                              │
├─────────────────────────────────────────────────────────────────┤
│ 剩余空间 [864-1023]: 预留扩展                                     │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 切分信息结构

```cpp
struct SplitInfo {
    vector<uint32_t> s1GBaseNum;      // 每个Batch的M轴基本块数量
    vector<uint32_t> s2BaseNum;       // 每个Batch的S2轴基本块数量
    vector<uint32_t> s1GTailSize;     // M轴尾块大小
    vector<uint32_t> s2TailSize;      // S2轴尾块大小
    bool isKvSeqAllZero;              // 是否全空batch
};

struct CostInfo {
    vector<int64_t> bN2CostOfEachBatch;        // 每个Batch*N2的总开销
    vector<uint32_t> bN2BlockOfEachBatch;      // 每个Batch*N2的总块数
    vector<int64_t> bN2LastBlockCostOfEachBatch; // 最后一块开销
    int64_t totalCost;                         // 全局总开销
    uint32_t totalBlockNum;                    // 全局总块数
    int64_t maxS1GCost;                        // 最大单行开销
};

struct SplitResult {
    uint32_t usedCoreNum;             // 实际使用核数
    vector<uint32_t> bN2End;          // 每个核的BN2结束索引
    vector<uint32_t> gS1End;          // 每个核的GS1结束索引
    vector<uint32_t> s2End;           // 每个核的S2结束索引
    int64_t maxCost;                  // 慢核开销
    FlashDecodeResult fdRes;          // FlashDecode归约信息
};
```

### 3.2 核心计算公式及依据

#### 3.2.1 基本块切分

| 计算项      | 公式                                                        | 依据说明                            |
| -------- | --------------------------------------------------------- | ------------------------------- |
| M轴基本块大小  | `mBaseSize = groupSize = N1/N2 = 64`                      | 与Q头数对应，便于后续kernel按头组切分          |
| S2轴基本块大小 | `s2BaseSize = 512`                                        | 与kernel内部循环的sInnerSize一致，减少尾块开销 |
| M轴块数     | `s1GBaseNum[b] = ceil(s1Size[b] * groupSize / mBaseSize)` | 将Q序列按头组切分                       |
| S2轴块数    | `s2BaseNum[b] = ceil(s2Size[b] / s2BaseSize)`             | 将KV序列按512切分                     |

#### 3.2.2 滑窗注意力负载计算

```
Sliding Window Attention 计算公式:

┌─────────────────────────────────────────────────────────────────┐
│                    KV Token有效范围计算                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: s1GIdx (M轴块索引), batchCache (batch信息)                 │
│                                                                 │
│  Step 1: 计算Q Token范围                                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ s1GFirstToken = s1GIdx * mBaseSize                     │     │
│  │ s1GLastToken = min(s1GFirstToken + mBaseSize,          │     │
│  │                   s1Size * groupSize) - 1              │     │
│  │                                                        │     │
│  │ 若 isS1G_ == true:                                     │     │
│  │     s1FirstToken = s1GFirstToken / groupSize           │     │
│  │     s1LastToken = s1GLastToken / groupSize             │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  Step 2: 计算KV Token范围 (BAND模式)                             │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ preTokenLeftUp = s1Size - s2Size + winLeft (127)       │     │
│  │ nextTokenLeftUp = s2Size - s1Size + winRight (0)       │     │
│  │                                                        │     │
│  │ s2FirstToken = s1FirstToken - preTokenLeftUp           │     │
│  │ s2LastToken = s1LastToken + nextTokenLeftUp            │     │
│  │                                                        │     │
│  │ 边界裁剪:                                               │     │
│  │ s2FirstToken = Clip(s2FirstToken, 0, s2Size-1)         │     │
│  │ s2LastToken = Clip(s2LastToken, 0, s2Size-1)           │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  Step 3: 计算实际Block范围                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ winS2Start = 0                                         │     │
│  │ winS2End = ceil((s2LastToken - s2FirstToken) / s2BaseSize)│  │
│  │ winS2TailSize = (s2LastToken - s2FirstToken + 1) % s2BaseSize│
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

公式依据: 该范围直接决定了后续SparseAttnSharedkv算子在S2轴的循环起止点，
         kernel会根据metadata中的s2Start/s2End确定KV数据加载范围。
```

#### 3.2.3 压缩注意力负载计算（CFA/SCFA模式）

```
Compressed Attention 计算公式:

┌─────────────────────────────────────────────────────────────────┐
│                  Cmp KV Token范围计算                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  cmpS2Start = winS2End (接续在Ori KV之后)                        │
│                                                                 │
│  cmpS2LastTokenSize 计算:                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ CFA模式 (cmpTopK == 0):                                │     │
│  │     cmpS2LastTokenSize = floor((s2LastToken+1) / cmpRatio)│  │
│  │                                                        │     │
│  │ SCFA模式 (cmpTopK > 0):                                │     │
│  │     cmpS2LastTokenSize = min(floor((s2LastToken+1)/cmpRatio),│ │
│  │                              cmpTopK)                  │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  cmpS2End = cmpS2Start + ceil(cmpS2LastTokenSize / s2BaseSize)  │
│  cmpS2TailSize = cmpS2LastTokenSize % s2BaseSize                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

公式依据: cmp_ratio为压缩率，后续kernel会按此比例读取压缩后的KV cache，
         cmpTopK为稀疏选取的token数上限，影响SCFA模式的有效计算范围。
```

#### 3.2.4 开销估算模型

```
开销计算公式 (用于负载均衡):

┌─────────────────────────────────────────────────────────────────┐
│                     Block Cost Model                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  开销计算函数:                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ CalcCost(basicM, basicS2):                             │     │
│  │     alignM = ceil(basicM / 16)      // M轴16对齐        │     │
│  │     alignS2 = ceil(basicS2 / 64)    // S2轴64对齐       │     │
│  │     cost = 6 * alignM + 10 * alignS2                   │     │
│  │     return cost                                        │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  Cost Table 四种Block类型:                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ WIN_NORMAL_BLOCK:  CalcCost(mBaseSize, s2BaseSize)     │     │
│  │ WIN_TAIL_BLOCK:    CalcCost(s1GTailSize, s2BaseSize)   │     │
│  │ WIN_TAIL_S2_BLOCK: CalcCost(mBaseSize, winS2TailSize)  │     │
│  │ WIN_TAIL_BOTH:     CalcCost(s1GTailSize, winS2TailSize)│     │
│  │                                                        │     │
│  │ CMP_NORMAL_BLOCK, CMP_TAIL_BLOCK 等同理                 │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  单行总开销:                                                     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ s1GCost = winS1GCost + cmpS1GCost                      │     │
│  │ 其中:                                                   │     │
│  │ winS1GCost = WIN_NORMAL_COST * normalBlockNum          │     │
│  │            + WIN_TAIL_COST * tailBlockNum              │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

系数依据:
- 系数6和10来源于实际kernel执行时的Matmul操作复杂度
- 16对齐: Cube单元处理的基本单元大小
- 64对齐: S2方向数据搬运的对齐要求

开销模型作用: 确保各核负载均衡，避免慢核影响整体性能
```

### 3.3 负载均衡分配策略

#### 3.3.1 三层分配策略详解

```
负载均衡分配策略 (三层递进):

┌─────────────────────────────────────────────────────────────────┐
│ Level 1: AssignByBatch (按整Batch分配)                           │
├─────────────────────────────────────────────────────────────────┤
│ 策略: 尽可能将完整Batch*N2分配给单个核                             │
│ 条件: costLimit + tolerance >= currentCost + batchCost             │
│ 优点: 减少跨batch边界，简化数据访问                                │
│ 适用: 大batch场景                                                │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: AssignByRow (按M轴行分配)                               │
├─────────────────────────────────────────────────────────────────┤
│ 策略: 将完整的M轴行(s1GIdx)分配给单个核                            │
│ 条件: costLimit + tolerance >= currentCost + rowCost            │
│ 优点: 保持S2循环连续性，减少workspace切换                          │
│ 适用: 中等batch场景                                              │
├─────────────────────────────────────────────────────────────────┤
│ Level 3: AssignByBlock (按Block分配)                             │
├─────────────────────────────────────────────────────────────────┤
│ 策略: 将单个S2 Block分配给核                                      │
│ 条件: costLimit + tolerance >= currentCost + blockCost           │
│ 优点: 精细控制，实现最佳均衡                                       │
│ 适用: 小batch或需要FlashDecode归约的场景                           │
│ 注意: 需记录跨核行的FD归约信息                                     │
└─────────────────────────────────────────────────────────────────┘

容差因子: FA_TOLERANCE_RATIO = 2
作用: 允许一定程度的负载不均衡，避免过度切分
```

#### 3.3.2 分配算法伪代码

```python
def balance_schedule(cost_info, aic_core_num):
    split_result = SplitResult()
    
    # 计算平均负载
    avg_cost = cost_info.totalCost / aic_core_num
    cost_limit = max(avg_cost, cost_info.maxS1GCost)
    
    # 初始化游标
    cur_b_idx = 0
    cur_s1g_idx = 0
    cur_s2_idx = 0
    unassigned_cost = cost_info.totalCost
    
    for core_idx in range(aic_core_num):
        core_cache = CoreCache(costLimit=cost_limit, cost=0, block=0)
        
        # Level 1: 按Batch分配
        assign_by_batch(core_cache, cur_b_idx, cost_info)
        
        # Level 2: 按Row分配
        assign_by_row(core_cache, cur_s1g_idx, cost_info)
        
        # Level 3: Block分配
        assign_by_block(core_cache, cur_s2_idx, cost_info)
        
        # 强制分配至少一个Block
        if core_cache.block == 0:
            force_assign(core_cache, cur_s2_idx)
        
        # 记录分配结果
        split_result.bN2End[core_idx] = cur_bn2_idx
        split_result.gS1End[core_idx] = cur_s1g_idx
        split_result.s2End[core_idx] = cur_s2_idx
        
        unassigned_cost -= core_cache.cost
    
    return split_result
```

***

## 4. 实现逻辑

### 4.1 参数获取与数据传输

由于Host CPU无法直接访问NPU设备内存，张量输入参数必须通过D2H传输获取。

#### 4.1.1 张量输入参数获取

```cpp
// Host端参数获取实现
class SparseAttnSharedkvMetadataHost {
private:
    int32_t* host_cu_seqlens_q_;  // Host侧cu_seqlens_q数据缓存
    int32_t* host_seqused_kv_;    // Host侧seqused_kv数据缓存
    
public:
    // 获取张量输入数据的D2H传输
    void GetTensorInputs(
        aclrtStream stream,
        const aclTensor* cu_seqlens_q,    // NPU侧aclTensor
        const aclTensor* seqused_kv,      // NPU侧aclTensor
        int64_t batch_size
    ) {
        // Step 1: 获取NPU设备指针
        void* device_cu_seqlens_ptr = nullptr;
        void* device_seqused_kv_ptr = nullptr;
        
        if (cu_seqlens_q != nullptr) {
            // 通过aclTensor API获取设备数据指针
            device_cu_seqlens_ptr = aclGetTensorDataPtr(cu_seqlens_q);
        }
        
        if (seqused_kv != nullptr) {
            device_seqused_kv_ptr = aclGetTensorDataPtr(seqused_kv);
        }
        
        // Step 2: D2H传输cu_seqlens_q
        if (device_cu_seqlens_ptr != nullptr) {
            size_t cu_q_size = (batch_size + 1) * sizeof(int32_t);
            aclrtMemcpy(
                host_cu_seqlens_q_,              // Host destination
                cu_q_size,
                device_cu_seqlens_ptr,           // NPU source
                cu_q_size,
                ACL_MEMCPY_DEVICE_TO_HOST,
                stream
            );
        }
        
        // Step 3: D2H传输seqused_kv
        if (device_seqused_kv_ptr != nullptr) {
            size_t seq_kv_size = batch_size * sizeof(int32_t);
            aclrtMemcpy(
                host_seqused_kv_,
                seq_kv_size,
                device_seqused_kv_ptr,
                seq_kv_size,
                ACL_MEMCPY_DEVICE_TO_HOST,
                stream
            );
        }
        
        // Step 4: 同步等待传输完成
        aclrtSynchronizeStream(stream);
    }
};
```

**数据量估算**：

| 参数             | 维度     | 数据量 (B=4)          | 数据量 (B=128)           |
| -------------- | ------ | ------------------ | --------------------- |
| cu\_seqlens\_q | \[B+1] | (4+1)×4 = 20 bytes | (128+1)×4 = 516 bytes |
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

