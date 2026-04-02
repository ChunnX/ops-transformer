# AttentionPioneerMetadata AICPU算子 详细设计文档

## 修订记录

| 日期 | 修订版本 | 修改描述 | 作者 |
| :---: | :---: | :---: | :---: |
| 2026-04-02 | 1.0 | A5平台详细设计初稿 | - |

---

## 1. 概述

### 1.1 背景与目标

在当前ACL Graph入图场景中，AttentionPioneer算子的Tiling刷新功能在Host侧执行。虽然Host Tiling刷新时间通常可以被Kernel执行时间掩盖，但在盘古极低时延场景（TPOT=4ms）下，Host Tiling刷新耗时约6ms，无法被掩盖，形成Host bound瓶颈。

**解决方案**：将Tiling中的LoadBalance（负载均衡分核）部分下沉到AI CPU执行，新增独立的 `AttentionPioneerMetadata` AICPU算子，计算分核信息并输出metaData tensor供主算子使用。

### 1.2 适用平台

- **硬件平台**：A5（Ascend 950DT / 950PR）
- **目标仓库**：ops-transformer/attention/attention_pioneer

### 1.3 参考实现

- **A3参考实现**：omni-ops/inference/ascendc/src/ops-transformer/attention/ai_infra_fused_infer_attention_sink_metadata
- **已有A5代码**：ops-transformer/attention/attention_pioneer（主算子，含split_core.h/cpp）

---

## 2. 算法原理

### 2.1 整体架构

```
┌──────────────────────────────────────────────────────┐
│                     ACL Graph                         │
├──────────────────────┬───────────────────────────────┤
│  Stream1 (AICPU)     │  Stream2 (主计算)              │
│                      │                                │
│  AttentionPioneer    │  aclrtStreamWaitEvent           │
│  Metadata算子        │         │                      │
│  (计算分核信息)      │         ▼                      │
│         │            │  AttentionPioneer (Layer1)      │
│   RecordEvent        │  (读取metaData)                │
│         │            │         │                      │
│         └──Event──→──┤         ▼                      │
│                      │  AttentionPioneer (Layer2)      │
│                      │  (复用metaData)                │
│                      │         │                      │
│                      │        ...                     │
└──────────────────────┴───────────────────────────────┘
```

**核心设计思路**：
1. 将原Host Tiling拆分为两部分：
   - **TilingKey部分**（仍在Host）：生成TilingKey和基础TilingData，不涉及actual_seq_len
   - **LoadBalance部分**（下沉到AICPU）：根据actual_seq_len计算负载均衡分核信息
2. AICPU算子每step只调用1次，多层复用同一个metaData tensor
3. 两流之间通过Event机制同步

### 2.2 为什么不复用已有的分核代码，而要重新实现

A5的 `attention_pioneer` 主算子已有一套分核实现：`op_host/common/split_core.h/cpp`（namespace `ap_optiling`）。一个自然的问题是：为什么不直接把这套代码拿过来给AICPU用，而要在AICPU kernel中重新写一套？

原因有三个层面：

#### 2.2.1 功能缺失——Host侧split_core不支持Sink和FD

Host侧 `BaseInfo` 结构体的字段：
```cpp
// op_host/common/split_core.h — Host侧
struct BaseInfo {
    uint32_t bSize, n2Size, gSize, s1Size, s2Size;
    bool isS1G, isAccumSeqS1, isAccumSeqS2;
    std::vector<int64_t> actualSeqS1Size, actualSeqS2Size;
    uint32_t actualLenQDims, actualLenKvDims;
    bool attenMaskFlag;
    int32_t sparseMode;
    int64_t preToken, nextToken, actualSeqPrefixSize;
    // 注意：没有 sinkNumber / keySinkNumber / supportFD
};
```

AICPU侧 `APBaseInfo` 结构体**额外**需要：
```cpp
// op_kernel_aicpu — AICPU侧
struct APBaseInfo {
    // ... 以上所有字段 ...
    int64_t sinkNumber;      // Sink token数量
    int64_t keySinkNumber;   // K Sink tensor维度
    bool supportFD;          // 是否支持FlashDecode
};
```

这三个字段不是可选的装饰，它们**深入影响分核算法的核心路径**：

| 缺失字段 | 影响的函数 | 具体影响 |
|----------|-----------|---------|
| `sinkNumber` | CalcS1GCacheWithSinkNumber | Sink token在KV cache中占据固定位置，分核时必须计算sink块的独立cost，否则某些核会被分配到被mask掉的空块，造成负载严重不均 |
| `keySinkNumber` | CalcS1GCache、CalcSplitInfo | keySinkNumber影响s2Size的计算（`s2Val = keySinkNumber + GetS2SeqSize()`），如果不加这个偏移，s2维度的block数会算少，分核结果和主算子预期不一致 |
| `supportFD` | CalcSplitPlan中的AssignByBlock/ForceAssign分支 | 只有支持FD时才会做block级细粒度分配和FD归约记录，否则只做batch/row级分配。这直接决定了分核的粒度 |

如果在Host侧split_core中强行添加这些字段和函数，会**破坏已有的Host Tiling流程**——Host侧的Tiling目前工作正常，改动它存在回归风险。

#### 2.2.2 编译环境不兼容

Host侧代码运行在x86/ARM的Host CPU上，编译工具链是标准gcc/g++。AICPU kernel运行在AI CPU上，有自己的编译工具链和头文件依赖（`cpu_kernel.h`、`cpu_context.h`、`cpu_tensor.h`）。

Host侧split_core.h依赖的头文件链与AICPU环境不同。即使功能完全一致，直接include也会导致编译错误。

#### 2.2.3 架构选择——类成员函数 vs 独立函数

Host侧split_core是一组**独立的命名空间函数**，输入输出通过参数传递。

AICPU kernel是一个**CpuKernel派生类**，分核逻辑需要访问类成员变量（`aicCoreNum_`、`cvRatio_`、`mBaseSize_`、`sInnerSize_` 等），作为成员函数实现更自然。如果复用独立函数，要么需要大量参数传递，要么需要结构上的桥接层，增加不必要的复杂度。

**结论**：在AICPU kernel类内完整实现分核逻辑，参考A3的AICPU实现（已验证的代码），是风险最小、功能最完整的方案。

---

### 2.3 Event同步机制——算子间的执行依赖是如何保证的

#### 2.3.1 问题本质

AttentionPioneerMetadata在AICPU流上计算，主算子AttentionPioneer在AI Core流上执行。主算子必须在metadata计算完成**之后**才能读取metadata tensor。如果两者并行执行，主算子可能读到未初始化或正在写入的数据。

#### 2.3.2 Event机制的工作原理

```
时间线 ──────────────────────────────────────────────→

Stream1 (AICPU):
  ┌─────────────────────┐
  │ AttentionPioneer     │
  │ Metadata (计算分核)  │──→ aclrtRecordEvent(event)
  └─────────────────────┘              │
                                       │ event信号
Stream2 (AI Core):                     ▼
  aclrtStreamWaitEvent(stream2, event)
          │ (阻塞等待event完成)
          ▼
  ┌─────────────────────┐
  │ AttentionPioneer     │
  │ Layer1 (读metadata)  │
  └─────────────────────┘
          │
          ▼
  ┌─────────────────────┐
  │ AttentionPioneer     │
  │ Layer2 (读metadata)  │  ← 不需要再等event，因为stream2是顺序执行的
  └─────────────────────┘
          │
         ...
```

**关键API调用**（在ACL Graph构图侧，由框架自动生成或用户手动插入）：
1. `aclrtCreateEvent(&event)` — 创建event对象
2. 在Stream1上：AICPU算子执行完后，`aclrtRecordEvent(stream1, event)` — 在流中插入一个"信号点"
3. 在Stream2上：主算子执行前，`aclrtStreamWaitEvent(stream2, event)` — 在流中插入一个"等待点"

#### 2.3.3 算子内部视角

**重要澄清**：Event同步逻辑**不在算子内部**。AttentionPioneerMetadata算子本身不知道有Event存在，它只是：
1. 从输入tensor读actual_seq_lens
2. 计算分核信息
3. 写入输出metaData tensor
4. 返回

同样，AttentionPioneer主算子也不知道Event——它只是从metaData tensor读数据。

Event的插入是**ACL Graph框架层面**的事情。框架在构图时，根据两个算子之间的数据依赖（metaData tensor是Metadata算子的输出、主算子的输入），自动在两个stream之间插入Event同步。

#### 2.3.4 为什么用双流而不是单流

如果放在同一个stream上顺序执行：
```
AICPU算子 → 主算子Layer1 → 主算子Layer2 → ...
```
AICPU算子必须先完成，主算子才能开始。AICPU的执行时间直接加到关键路径上。

双流方案下：
```
Stream1: AICPU算子 (独立执行)
Stream2: 可以提前做一些准备工作，只在真正需要读metadata时等待
```
AICPU算子可以在前一个step的尾部就开始执行，与当前step的其他计算重叠。在理想情况下，当主算子准备好读metadata时，AICPU已经算完了，等待时间为零。

---

### 2.4 MetaData复用机制——同一step内多层如何共享

#### 2.4.1 复用的前提

在Transformer的decode阶段，同一step内所有layer的Attention计算具有以下**共同特征**：
- **相同的actual_seq_lens**：每个batch样本的序列长度在同一step内不变（Q序列长度固定为1或很小值，KV序列长度是已生成的token数）
- **相同的核数配置**：同一芯片上所有layer用相同的AIC/AIV核
- **相同的head维度结构**：num_heads_q、num_heads_kv、head_dim等对所有layer一致

因此，分核信息（哪个核处理哪些batch/head/token块）对**同一step的所有layer完全相同**。

#### 2.4.2 复用的实现

```
Step N:
  Stream1: AttentionPioneerMetadata(actual_seq_lens_step_N) → metaData tensor
                          │
              RecordEvent ─┤
                           │
  Stream2:    WaitEvent ───┤
              Layer 1: AttentionPioneer(Q1, KV1, metaData)  ← 读同一个metaData
              Layer 2: AttentionPioneer(Q2, KV2, metaData)  ← 读同一个metaData
              ...
              Layer L: AttentionPioneer(QL, KVL, metaData)  ← 读同一个metaData

Step N+1:
  Stream1: AttentionPioneerMetadata(actual_seq_lens_step_N+1) → metaData tensor (重写)
  ...
```

**不是按batch复用，而是按step复用**：
- 同一step内，所有layer的所有batch共享同一份metaData
- 不同step之间，metaData需要重新计算（因为KV序列长度增长了1）
- metaData tensor是一个**全局唯一的tensor**，所有layer通过地址引用同一块内存

#### 2.4.3 为什么每个step只需计算一次

每个step中，KV cache的序列长度增加1（新生成了一个token）。分核的关键输入是 `actual_seq_lens`，它描述每个batch样本的实际KV长度。

- Layer1和LayerL看到的KV长度完全相同（都是step N时的总长度）
- 不同layer之间唯一的差异是Q/K/V的权重和数据值，但这不影响分核策略
- 因此分核信息对所有layer通用

#### 2.4.4 下一个step为什么要重算

Step N+1时，KV cache中多了一个token（刚刚Step N生成的），`actual_seq_lens` 的每个元素都+1。这意味着：
- s2Size（KV序列长度）增大，s2维度的block数可能变化
- cost计算结果不同，负载均衡方案可能完全不同
- 某些batch可能从FlashAttention模式切换到FlashDecode模式（s1很小但s2很大时）

所以每个step都必须重新调用一次AttentionPioneerMetadata。

---

### 2.5 SplitCore分核算法详解

分核是一个**贪心负载均衡**问题：将一个3维工作空间（Batch×N2, S1×G, S2）的计算量尽可能均匀地分配到有限数量的AIC核上。

#### 2.5.1 总体流程

```
SplitCore(coreNum, baseInfo, splitParam)
│
├── Step 1: CalcSplitInfo — 将连续维度离散化为基本块
│     对每个batch:
│       s1GBaseNum[b] = ceil(s1[b] * gSize / mBaseSize)   // M轴块数
│       s2BaseNum[b]  = ceil(s2[b] / s2BaseSize)          // S2轴块数
│       s1GTailSize[b] = (s1[b] * gSize) % mBaseSize      // M轴尾块大小
│       s2TailSize[b]  = s2[b] % s2BaseSize                // S2轴尾块大小
│
├── Step 2: CalcCostInfo — 计算每个BN2维度的总cost
│     对每个batch:
│       CalcBatchCost(b) → bN2CostOfEachBatch[b], bN2BlockOfEachBatch[b]
│     totalCost = Σ bN2CostOfEachBatch[b] * n2Size
│     totalBlockNum = Σ bN2BlockOfEachBatch[b] * n2Size
│
├── Step 3: 搜索最优核数
│     minCore = round(sqrt(totalBlockNum))
│     maxCore = min(coreNum, totalBlockNum)
│     for i in [minCore, maxCore]:
│       CalcSplitPlan(i, ...) → 得到一个分配方案
│       保留 maxCost 最小的方案
│
├── Step 4: CalcSplitPlan — 对给定核数执行贪心分配
│     对每个core:
│       costLimit = remainCost / remainCoreNum  // 平均剩余cost
│       1. AssignByBatch — 按整个BN2粒度分配
│       2. AssignByRow   — 按整个S1G行粒度分配
│       3. AssignByBlock — 按单个S2块粒度分配 (仅FD模式)
│       4. ForceAssign   — 强制分配至少1个块 (仅FD模式)
│       记录该core的结束位置 (bN2End, gS1End, s2End)
│
└── Step 5: SplitFD — FD归约任务的Vector核负载均衡 (仅有FD任务时)
```

#### 2.5.2 Step 1: CalcSplitInfo（离散化）

Attention的计算空间可以抽象为一个2D矩阵：

```
           S2 (KV序列长度)
        ┌──────────────────────┐
        │                      │
  S1×G  │   Attention Score    │
  (Q序列 │   矩阵             │
  ×组数) │                      │
        └──────────────────────┘
```

这个矩阵不能任意切分，必须按**基本块**为单位切分（硬件对齐要求）：
- M轴（S1×G方向）的基本块大小：`mBaseSize`（如256）
- S2轴的基本块大小：`s2BaseSize` = `sInnerSize`（如512）

```
公式：
  s1GBaseNum[b] = ⌈(s1[b] × gSize) / mBaseSize⌉     // M轴方向有多少个完整块
  s2BaseNum[b]  = ⌈s2[b] / s2BaseSize⌉                // S2轴方向有多少个完整块
  s1GTailSize[b] = (s1[b] × gSize) mod mBaseSize       // M轴最后一个块的实际大小(可能小于mBaseSize)
  s2TailSize[b]  = s2[b] mod s2BaseSize                 // S2轴最后一个块的实际大小
```

**为什么需要tail**：序列长度不一定能被基本块整除。尾块的计算量比正常块小，后面计算cost时要区分。

#### 2.5.3 Step 2: CalcCostInfo（开销建模）

每个基本块的计算开销用一个**简化的FLOPS模型**估算：

```cpp
CalcCost(basicM, basicS2) = 6 × ⌈basicM/16⌉ + 10 × ⌈basicS2/64⌉
```

这个公式的含义：
- `6 × ⌈basicM/16⌉`：矩阵乘法部分（Q×K^T），M维度以16对齐（Cube单元的计算宽度）
- `10 × ⌈basicS2/64⌉`：Softmax + V投影部分，S2维度以64对齐（Vector单元的处理宽度）
- 系数6和10是经验权重，反映Cube和Vector计算的相对耗时

对每种块类型组合计算cost（2×2 = 4种）：

```
CostTable[normal_m][normal_s2] = CalcCost(mBaseSize, s2BaseSize)
CostTable[tail_m][normal_s2]   = CalcCost(s1GTailSize, s2BaseSize)
CostTable[normal_m][tail_s2]   = CalcCost(mBaseSize, s2TailSize)
CostTable[tail_m][tail_s2]     = CalcCost(s1GTailSize, s2TailSize)
```

然后对每个batch累加所有S1G行的cost：

```
对每个s1GIdx:
  s2Range = CalcS2Range(s1GIdx)   // 考虑mask后，该行实际需要计算的S2范围
  s1GCost = normalBlocks × CostTable[type_m][normal_s2] + tailBlocks × CostTable[type_m][tail_s2]
bN2Cost[b] = Σ s1GCost(所有s1GIdx)
```

**CalcS2Range的作用**（mask感知）：Causal mask等稀疏模式下，并非所有S2位置都需要计算。例如Left-Up Causal模式下，第i行只需要计算S2≤i+preToken的列。CalcS2Range根据sparseMode计算每行的有效S2范围，避免对全零区域计算cost。

#### 2.5.4 Step 3: 搜索最优核数

不是简单地用所有核，而是搜索一个最优核数：

```
minCore = round(√totalBlockNum)   // 最少核数的下界估计
maxCore = min(aicCoreNum, totalBlockNum)  // 不超过物理核数，也不超过总块数
```

对 `[minCore, maxCore]` 范围内的每个核数，执行一次完整的CalcSplitPlan，保留**最大单核cost最小**的方案。

**为什么不直接用所有核**：如果总计算量很小（比如decode时s1=1），用32个核分配10个块，大部分核空闲，反而因为FD归约开销增大总时间。搜索最优核数可以找到计算量和归约开销的平衡点。

#### 2.5.5 Step 4: CalcSplitPlan（贪心分配）

对给定的核数，按**从粗到细的4级粒度**依次分配：

**为什么是这个顺序（Batch → Row → Block → Force）**：

```
粗粒度 ─────────────────────────────────────── 细粒度
  AssignByBatch  →  AssignByRow  →  AssignByBlock  →  ForceAssign
  (整个BN2)       (整个S1G行)      (单个S2块)        (强制1块)
```

**第1级：AssignByBatch**——按整个BN2分配

```
while (coreCost + bN2Cost ≤ costLimit + tolerance):
    coreCost += bN2Cost
    移到下一个BN2
```

含义：如果加上整个batch的整个head的所有计算量，该核还在允许范围内，就整个分配给它。这是最高效的——不产生跨核切分，无需FD归约。

**第2级：AssignByRow**——按S1G行分配

当一个完整的BN2放不下了，尝试逐行分配：
```
while (coreCost + s1GCost ≤ costLimit + tolerance):
    coreCost += s1GCost
    移到下一个S1G行
```

含义：在同一个BN2内，如果整行的S2块都能放下，就整行分配。仍然不产生S2方向的切分。

**第3级：AssignByBlock**——按S2块分配（仅FD模式）

当整行放不下，逐块分配：
```
while (coreCost + blockCost ≤ costLimit + tolerance):
    coreCost += blockCost
    curS2Idx++
```

这会导致**同一行的S2被切分到不同核上**——产生FD归约需求。这就是FlashDecode的场景：多个核各自计算一行attention的不同S2分段，最后需要归约合并。

**第4级：ForceAssign**——强制分配

如果前3级都没能分配任何块（可能是该核的costLimit太小），强制分配1个块，确保每个活跃核至少有工作。

**tolerance（容忍度）机制**：

分配时不要求严格 `≤ costLimit`，而是允许一定的超额：
```
APIsWithinTolerance(costLimit, lastBlockCost / 2, coreCost + newCost)
// 即: costLimit + lastBlockCost/2 >= coreCost + newCost
```
这个容忍度等于最后一个块cost的一半。目的是避免"差一点就能塞下"导致的严重不均衡——如果一个核差半个块就满了却不分配，这半个块的代价会落到下一个核，造成更大的不均衡。

#### 2.5.6 Step 5: SplitFD（FD归约的Vector核负载均衡）

**为什么最后才做FD负载均衡**：

FD归约是Step 4的**产物**，只有在CalcSplitPlan执行完毕后，才能知道：
1. 哪些(bN2, s1G)被切分到了多个核（产生了FD归约需求）
2. 每个FD任务被切了多少份（s2SplitNum）
3. 总共有多少个FD任务（numOfFdHead）

所以SplitFD必须在CalcSplitPlan之后执行。

**FD归约是什么**：

当一行attention的S2维度被切分到K个核时：
```
Core_0 计算: Attention(Q, KV[s2_0 : s2_1]) → partial_out_0, partial_lse_0
Core_1 计算: Attention(Q, KV[s2_1 : s2_2]) → partial_out_1, partial_lse_1
...
```
这些partial结果需要通过**online softmax归约**合并成最终结果。归约操作由Vector核执行。

**SplitFD做什么**：

将所有FD归约任务均匀分配给Vector核：
```
总FD负载 = Σ (s2SplitNum[i] × gS1SplitNum[i])   // 每个FD任务的归约量
每个Vector的目标负载 = 总FD负载 / maxVectorNum

遍历所有FD任务的所有gS1分片:
  如果当前Vector的负载 + 新分片的归约量 > 目标负载 × 2:
    关闭当前Vector，打开下一个
  将该分片分配给当前Vector
```

结果是每个Vector核知道自己负责哪些FD任务的哪些gS1分片（通过 `gS1IdxEndOfFdHead` 和 `gS1IdxEndOfFdHeadSplit` 两级索引）。

---

### 2.6 MetaData内存布局设计详解

#### 2.6.1 为什么不同类型的核需要不同的参数

NPU的计算架构是**Cube-Vector协同**模式：

```
┌─────────────────────────────────────────┐
│ AIC Core (Cube核) — 负责矩阵乘法        │
│  Q×K^T (Attention Score)                 │
│  Score×V (Attention Output)              │
│  需要知道：处理哪些batch/head/token块    │
│                                          │
│ 配对的 AIV Core (Vector核) — 负责向量运算│
│  Softmax                                 │
│  FlashDecode归约                         │
│  需要知道：FD归约任务分配                │
└─────────────────────────────────────────┘
```

**AIC核** 需要的是"我要算哪一块"——即三维空间中的工作区间：
- `BN2_END`: 我处理到哪个batch×head
- `GS1_END`: 我处理到哪个S1G行
- `S2_END`: 我处理到哪个S2块

**AIV核** 需要的是"我要归约哪些FD任务"：
- `GS1_IDX_END_OF_FD_HEAD`: 我负责到哪个FD任务
- `GS1_IDX_END_OF_FD_HEAD_SPLIT`: 我负责到该任务的哪个gS1分片

两者的信息**不可互换**——AIC核不做归约，AIV核不做矩阵乘，所以放在不同的段中。

#### 2.6.2 每个AIC核的10个参数为什么都是必须的

```
AIC Metadata (每核10个int32):
[0] AIC_CORE_ENABLE       — 必须：该核是否参与计算。usedCoreNum可能<物理核数，未使用的核必须跳过
[1] BN2_END_PTR           — 必须：该核负责的BN2结束位置（exclusive），与前一核的BN2_END配合确定范围
[2] GS1_END_PTR           — 必须：该核负责的S1G结束行，配合BN2确定M轴范围
[3] S2_END_PTR            — 必须：该核负责的S2结束块，确定KV方向的计算范围
[4] BN2_IDX_OF_FD_HEAD    — FD必须：如果该核的某行S2被切分，记录归约任务对应的BN2，用于归约时定位数据
[5] GS1_IDX_OF_FD_HEAD    — FD必须：归约任务对应的S1G行索引
[6] S2_SPLIT_NUM_OF_FD_HEAD — FD必须：该行S2被切成几份，归约时需要知道有多少个partial结果
[7] S2_SPLIT_START_IDX     — FD必须：该核处理的是第几份S2分片，用于确定workspace写入位置
[8] GS1_SPLIT_NUM_OF_FD_HEAD — FD必须：归约任务M轴切分数，用于Vector核的二级调度
[9] GS1_LAST_PART_SIZE     — FD必须：M轴最后一份的实际大小（可能<gS1BaseSizeOfFd），避免越界
```

其中 [0]-[3] 是**所有场景必须**的——它们定义了核的工作区间。[4]-[9] 是**FD场景必须**的——没有S2切分时这些字段为0，不影响计算。

#### 2.6.3 每个AIV核的3个参数为什么都是必须的

```
AIV Metadata (每核3个int32):
[0] AIV_CORE_ENABLE                — 必须：该Vector核是否参与FD归约
[1] GS1_IDX_END_OF_FD_HEAD        — FD必须：一级索引，该Vector负责到第几个FD任务
[2] GS1_IDX_END_OF_FD_HEAD_SPLIT  — FD必须：二级索引，负责到该任务的第几个gS1分片
```

两级索引的必要性：一个FD任务可能包含多个gS1分片（M轴被进一步切分），每个分片的归约量 = s2SplitNum（该行的S2分片数）。单靠一级索引无法精确切分负载边界。

#### 2.6.4 BASE参数的全局性

```
Base Metadata (10个int32，所有核共享):
[0] M_BASE_SIZE        — 主算子需要：M轴基本块大小，kernel按此步长遍历M维度
[1] S_INNER_SIZE       — 主算子需要：S2内切大小，kernel按此步长遍历S2维度
[2] M_FD_BASE_SIZE     — FD归约需要：归约阶段M轴切分粒度
[3] NUM_OF_FD          — 所有核需要：FD归约任务总数，为0则跳过归约
[4] USED_CORE_NUM      — 调度需要：实际使用的AIC核数
[5] USED_VEC_NUM_OF_FD — 调度需要：FD阶段使用的Vector核数
[6] S1_SIZE            — 主算子需要：Q序列长度（所有batch相同的全局值）
[7] S2_SIZE            — 主算子需要：KV序列长度（或最大值）
[8] ACTUAL_LEN_Q_DIM   — 主算子需要：actual_seq_lens_q的维度数，影响序列长度读取方式
[9] ACTUAL_LEN_KV_DIM  — 主算子需要：actual_seq_lens_kv的维度数
```

这些参数之所以放在BASE段而不是每核一份，是因为它们对**所有核完全相同**——如果复制到每个核的段中，会浪费 `32×10=320` 个额外的int32空间。

#### 2.6.5 内存布局的地址计算

整个MetaData tensor是一个连续的int32数组，三段顺序排列：

```
偏移0:                     AIC段 [0 ... AIC_CORE_NUM × 10 - 1]
偏移AIC_CORE_NUM × 10:    AIV段 [0 ... AIV_CORE_NUM × 3 - 1]
偏移AIC×10 + AIV×3:       BASE段 [0 ... 9]

具体 (A5):
AIC段:  偏移 0~319    (32核 × 10 = 320)
AIV段:  偏移 320~511  (64核 × 3 = 192)
BASE段: 偏移 512~521  (10)
总计:   522 个 int32，远小于 1024 的tensor容量
```

主算子kernel侧通过以下函数计算绝对偏移：
```cpp
GetAPAICMetaAbsIndex(coreIdx, metaIdx) = coreIdx × 10 + metaIdx
GetAPAIVMetaAbsIndex(coreIdx, metaIdx) = 32×10 + coreIdx × 3 + metaIdx
GetAPBaseMetaAbsIndex(metaIdx)         = 32×10 + 64×3 + metaIdx
```

**这就是为什么核数错误是致命的**：如果 `AP_AIC_CORE_NUM` 填成48而不是32，AIV段的起始偏移就是480而不是320，主算子读AIV段时会读到AIC段的错误数据。

---

### 2.7 分核计算流程

```
输入参数 → 参数校验 → 判断MLA/GQA模式
                          │
                    ┌─────┴─────┐
                    │           │
                    ▼           ▼
              MLA模式       GQA模式
              CalcInner     CalcInner
              SizeMla       SizeGqa
                    │           │
                    └─────┬─────┘
                          ▼
                   构建分核输入参数
                   (BaseInfo + SplitParam)
                          │
                          ▼
                    SplitCore分核
                    ├─ CalcSplitInfo (划分基本块)
                    ├─ CalcCostInfo (计算开销)
                    ├─ CalcSplitPlan (分配核间负载)
                    │   ├─ AssignByBatch (按batch分配)
                    │   ├─ AssignByRow (按行分配)
                    │   ├─ AssignByBlock (按块分配)
                    │   └─ ForceAssign (强制分配)
                    └─ SplitFD (FD负载均衡)
                          │
                          ▼
                    GenMetaData
                    (生成metadata tensor)
```

### 2.8 关键数据结构

#### 2.8.1 MetaData 内存布局

```
MetaData Tensor (1024 x int32)
├── AIC Metadata [AIC_CORE_NUM][AIC_METADATA_SIZE]
│   ├── [0] AIC_CORE_ENABLE     : 核使能标记
│   ├── [1] BN2_END_PTR         : BN2结束点
│   ├── [2] GS1_END_PTR         : GS1结束点
│   ├── [3] S2_END_PTR          : S2结束点
│   ├── [4] BN2_IDX_OF_FD_HEAD  : FD归约任务BN2索引
│   ├── [5] GS1_IDX_OF_FD_HEAD  : FD归约任务GS1索引
│   ├── [6] S2_SPLIT_NUM_OF_FD_HEAD : S2切分份数
│   ├── [7] S2_SPLIT_START_IDX_OF_CORE : S2切分起始索引
│   ├── [8] GS1_SPLIT_NUM_OF_FD_HEAD : GS1切分份数
│   └── [9] GS1_LAST_PART_SIZE_OF_FD_HEAD : GS1最后一份大小
│
├── AIV Metadata [AIV_CORE_NUM][AIV_METADATA_SIZE]
│   ├── [0] AIV_CORE_ENABLE     : Vector核使能标记
│   ├── [1] GS1_IDX_END_OF_FD_HEAD : FD归约一级索引
│   └── [2] GS1_IDX_END_OF_FD_HEAD_SPLIT : FD归约二级索引
│
└── Base Metadata [BASE_METADATA_SIZE]
    ├── [0] M_BASE_SIZE         : M轴基本块大小
    ├── [1] S_INNER_SIZE        : S2轴内切大小
    ├── [2] M_FD_BASE_SIZE      : FD阶段M轴基本块大小
    ├── [3] NUM_OF_FD           : FD归约任务数
    ├── [4] USED_CORE_NUM       : 使用的核数
    ├── [5] USED_VEC_NUM_OF_FD  : FD使用的Vector核数
    ├── [6] S1_SIZE             : S1维度大小
    ├── [7] S2_SIZE             : S2维度大小
    ├── [8] ACTUAL_LEN_Q_DIM    : Q的actual_seq_len维度
    └── [9] ACTUAL_LEN_KV_DIM   : KV的actual_seq_len维度
```

**A5平台常量适配**：
| 常量 | A3值 | A5值 | 说明 |
|------|------|------|------|
| AIC_CORE_NUM | 36 (数组预留上限，实际24核) | 32 | AIC核数上限 |
| AIV_CORE_NUM | 72 (数组预留上限，实际48核) | 64 | AIV核数上限 |
| FIASINK_META_SIZE / AP_METADATA_SIZE | 1024 | 1024 | MetaData tensor元素数 |

> **说明**：A5 V100平台实际AIC=32核、AIV=64核（AIC:AIV = 1:2）。
> MetaData数组空间计算: AIC=32×10=320; AIV=64×3=192; BASE=10; 总计=522个uint32_t，1024完全够用。
> A3中 AIC_CORE_NUM=36/AIV_CORE_NUM=72 是MetaData数组的预留上限（大于实际的24/48核），A5直接使用实际核数32/64作为数组维度。

---

## 3. 接口设计

### 3.1 算子原型定义

```cpp
REG_OP(AttentionPioneerMetadata)
    .OPTIONAL_INPUT(actual_seq_lengths_q, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OUTPUT(metaData_out, TensorType({DT_INT32}))
    .REQUIRED_ATTR(num_heads_q, Int)
    .REQUIRED_ATTR(num_heads_kv, Int)
    .REQUIRED_ATTR(head_dim_qk, Int)
    .REQUIRED_ATTR(head_dim_v, Int)
    .ATTR(batch_size, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "TND")
    .ATTR(input_layout_kv, String, "TND")
    .ATTR(sink_num, Int, 0)
    .ATTR(k_sink_num, Int, 0)
    .ATTR(rope_head_dim, Int, 0)
    .ATTR(block_size, Int, 0)
    .REQUIRED_ATTR(soc_version, String)
    .REQUIRED_ATTR(aic_core_num, Int)
    .REQUIRED_ATTR(aiv_core_num, Int)
    .OP_END_FACTORY_REG(AttentionPioneerMetadata)
```

### 3.2 输入输出规格

| 序号 | 参数 | 方向 | 类型 | 维度 | 说明 |
|------|------|------|------|------|------|
| 1 | actual_seq_lengths_q | 输入(可选) | INT64 | [B] | Q的前缀和序列长度 |
| 2 | actual_seq_lengths_kv | 输入(可选) | INT64 | [B] | KV的前缀和序列长度 |
| 3 | metaData_out | 输出 | INT32 | [FIASINK_META_SIZE] | 分核信息输出 |

### 3.3 属性规格

| 属性 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| num_heads_q | int64 | 是 | - | Query的head数 |
| num_heads_kv | int64 | 是 | - | Key/Value的head数 |
| head_dim_qk | int64 | 是 | - | Query/Key的head维度 |
| head_dim_v | int64 | 是 | - | Value的head维度 |
| batch_size | int64 | 否 | 0 | Batch大小 |
| sparse_mode | int64 | 否 | 0 | 稀疏模式(0-4) |
| pre_tokens | int64 | 否 | 2147483647 | 前向token数 |
| next_tokens | int64 | 否 | 2147483647 | 后向token数 |
| input_layout | string | 否 | "TND" | Q输入布局 |
| input_layout_kv | string | 否 | "TND" | KV输入布局 |
| sink_num | int64 | 否 | 0 | Sink token数量 |
| k_sink_num | int64 | 否 | 0 | K Sink tensor第0维 |
| rope_head_dim | int64 | 否 | 0 | RoPE head维度 |
| block_size | int64 | 否 | 0 | PageAttention块大小 |
| soc_version | string | 是 | - | SoC版本号 |
| aic_core_num | int32 | 是 | - | AIC核数 |
| aiv_core_num | int32 | 是 | - | AIV核数 |

### 3.4 aclnn接口

```cpp
aclnnStatus aclnnAttentionPioneerMetadataGetWorkspaceSize(
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDimQk,
    int64_t headDimV,
    int64_t batchSizeOptional,
    int64_t sparseModeOptional,
    int64_t preTokensOptional,
    int64_t nextTokensOptional,
    char *inputLayoutOptional,
    char *inputLayoutKvOptional,
    int64_t sinkNumOptional,
    int64_t kSinkNumOptional,
    int64_t ropeHeadDimOptional,
    int64_t blockSizeOptional,
    int64_t aicCoreNumOptional,
    int64_t aivCoreNumOptional,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnAttentionPioneerMetadata(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

---

## 4. 程序架构设计

### 4.1 文件结构

```
attention_pioneer/
├── op_host/
│   ├── attention_pioneer_metadata/           # 新增: AICPU算子的host侧
│   │   ├── attention_pioneer_metadata_def.cpp        # OpDef定义
│   │   ├── attention_pioneer_metadata_infershape.cpp # InferShape
│   │   └── CMakeLists.txt
│   └── op_api/
│       ├── aclnn_attention_pioneer_metadata.h        # aclnn头文件
│       ├── aclnn_attention_pioneer_metadata.cpp       # aclnn实现
│       ├── l0_attention_pioneer_metadata.h            # l0头文件
│       └── l0_attention_pioneer_metadata.cpp           # l0实现
├── op_kernel_aicpu/                          # 新增: AICPU kernel实现
│   ├── attention_pioneer_metadata_aicpu.h            # AICPU kernel头文件
│   ├── attention_pioneer_metadata_aicpu.cpp          # AICPU kernel实现
│   └── CMakeLists.txt
├── op_graph/
│   ├── attention_pioneer_metadata_proto.h            # 新增: 算子原型注册
│   └── CMakeLists.txt
└── common/
    ├── attention_pioneer_metadata.h          # 新增: MetaData常量定义
    └── (已有split_core.h/cpp可复用)
```

### 4.2 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| MetaData常量 | attention_pioneer_metadata.h | 定义A5平台核数常量、MetaData内存布局索引 |
| 算子原型 | attention_pioneer_metadata_proto.h | GE图层算子注册 |
| OpDef | attention_pioneer_metadata_def.cpp | OpDef注册，定义输入输出属性 |
| InferShape | attention_pioneer_metadata_infershape.cpp | 推导输出shape和dtype |
| AICPU Kernel | attention_pioneer_metadata_aicpu.h/cpp | 核心分核计算逻辑 |
| L0 API | l0_attention_pioneer_metadata.h/cpp | L0层AICPU task注册 |
| ACLNN API | aclnn_attention_pioneer_metadata.h/cpp | 对外aclnn接口 |

### 4.3 代码复用策略

A5的 `attention_pioneer` 已有 `split_core.h/cpp`，其分核逻辑与A3的AICPU kernel中的分核逻辑高度相似。**关键差异**：

| 差异点 | A5 split_core (Host) | A3 AICPU kernel |
|--------|---------------------|----------------|
| BaseInfo.sinkNumber | 无 | 有 |
| BaseInfo.keySinkNumber | 无 | 有 |
| BaseInfo.supportFD | 无 | 有 |
| CalcS1GCacheWithSinkNumber | 无 | 有 |
| UpdateCurS2IdxWithSinkNumber | 无 | 有 |
| MLA支持 | 无 | 有(CalcInnerSizeMla等) |

**策略**：AICPU kernel直接在类内实现完整的分核逻辑（参考A3实现），不复用Host侧的split_core。原因：
1. AICPU和Host的编译环境不同
2. AICPU kernel需要支持Sink相关的额外逻辑
3. 分核代码需要作为类的成员函数，访问类成员变量

---

## 5. A3→A5适配要点

### 5.1 核数适配

A5 V100平台核数与A3不同：
- A5 AIC: 32核（A3实际24核，MetaData数组预留36）
- A5 AIV: 64核（A3实际48核，MetaData数组预留72）
- AIC:AIV比例为1:2

MetaData常量适配：
```cpp
constexpr uint32_t AP_AIC_CORE_NUM = 32;  // A5: 32, A3预留: 36, A3实际: 24
constexpr uint32_t AP_AIV_CORE_NUM = 64;  // A5: 64, A3预留: 72, A3实际: 48
constexpr uint32_t AP_METADATA_SIZE = 1024; // 32*10+64*3+10=522, 1024足够
```

### 5.2 算子命名适配

| A3 | A5 |
|----|-----|
| AiInfraFusedInferAttentionSinkMetadata | AttentionPioneerMetadata |
| ai_infra_fused_infer_attention_sink_metadata | attention_pioneer_metadata |

### 5.3 分核逻辑复用

A3的分核核心逻辑可以直接复用：
- 参数校验（CheckSingleParam, ParseAxisInfo等）
- MLA/GQA模式判断（CheckIsMla）
- 内切大小计算（CalcInnerSizeMla, CalcInnerSizeGqa）
- M基本块计算（CalcMBaseSizeMla, CalcMBaseSizeGqa）
- 分核主流程（SplitCore → CalcSplitInfo → CalcCostInfo → CalcSplitPlan → SplitFD）
- MetaData生成（GenMetaData）

### 5.4 Sink特性支持

A5的attention_pioneer已有Sink支持（keySink/keyRopeSink/valueSink），AICPU需要对应支持：
- `sink_num` 属性：传统sink token number
- `k_sink_num` 属性：K Sink tensor第0维
- CalcS1GCacheWithSinkNumber：计算sink块的cost
- UpdateCurS2IdxWithSinkNumber：分配时跳过被mask掉的块

---

## 6. 性能设计

### 6.1 性能目标

| 场景 | 老方案 | 新方案 | 收益 |
|------|--------|--------|------|
| decode(极低时延, TPOT=4ms) | Host tiling 6ms 无法掩盖 | AICPU可完全掩盖 | >30% |
| prefill(不入图) | 不可掩盖 | 可被提前执行 | 0% |
| decode(非极低时延) | 可掩盖 | 无收益 | 0% |

### 6.2 性能优化措施

1. **单step单次计算**：每step只调用一次AttentionPioneerMetadata，step内多层复用
2. **双流并行**：AICPU流和主计算流并行执行
3. **提前执行**：通过Event机制AICPU可提前启动
4. **MetaData复用**：全局metaData tensor，每层地址一致

---

## 7. 约束与限制

1. 仅支持ACL Graph入图，不支持GE入图
2. 只支持TND输入布局
3. AclGraph Tiling下沉场景Profiling中分核信息不准，按最大BlockDim上报
4. metaData Tensor需全局管理，每层地址一致
5. 流依赖只需每step首层设置，后续层不调用AICPU算子

---

## 8. 验收标准

### 8.1 功能验收

| 测试项 | 验收方法 | 验收标准 |
|--------|----------|----------|
| metadata正确性 | 对比新旧方案分核结果 | 完全一致 |
| 多流依赖正确性 | 验证Event同步 | 执行顺序正确 |
| 单step多层复用 | 验证metadata复用 | 各层使用同一metadata |
| 参数校验 | 边界值测试 | 异常参数正确拦截 |

### 8.2 性能验收

| 测试项 | 方法 | 目标 |
|--------|------|------|
| AICPU执行时间 | profiling | < 0.5ms |
| 端到端掩盖 | profiling | AICPU时间被完全掩盖 |
