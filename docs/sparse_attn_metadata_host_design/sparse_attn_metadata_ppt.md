# SparseAttnSharedkvMetadata 算子分析 PPT

---

## 第1页：标题页

### 算子名称
**SparseAttnSharedkvMetadata Host端重设计方案**

### 功能定位
为SparseAttnSharedkv算子提供任务分核列表，实现负载均衡的Attention计算调度

### 目标
将AICPU实现迁移至Host端CPU，使用TileLang语言重写

---

## 第2页：算子整体架构

### 系统架构图
```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户调用层                                     │
│   torch.ops.custom.npu_sparse_attn_sharedkv_metadata(...)           │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Host端 (本项目实现)                           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │   参数解析   │──▶│  负载均衡    │──▶│ Metadata生成 │            │
│  │   参数校验   │   │  任务分配    │   │  输出组装    │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼ metadata[1024]
┌─────────────────────────────────────────────────────────────────────┐
│                        NPU端                                         │
│   SparseAttnSharedkv 算子使用metadata进行分核计算                    │
│   ├─ Cube核: Matmul计算 (Q@K^T, Softmax@V)                         │
│   └─ Vector核: Softmax、累加、FlashDecode归约                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 前序-后续关系
| 算子 | 位置 | 功能 |
|------|------|------|
| **SparseAttnSharedkvMetadata** | Host端 | 生成分核任务列表 |
| **SparseAttnSharedkv** | NPU端 | 执行Attention计算 |

---

## 第3页：输入输出规格

### 输入参数表

| 参数名 | 类型 | 维度 | 约束 | 说明 |
|--------|------|------|------|------|
| cu_seqlens_q | INT32 | [B+1] | TND布局必传 | Q序列前缀和 |
| seqused_kv | INT32 | [B] | PA_ND布局必传 | KV实际有效长度 |

### 属性参数表

| 属性名 | 类型 | 固定值/范围 | 说明 |
|--------|------|-------------|------|
| num_heads_q | INT32 | **64** | Q头数 |
| num_heads_kv | INT32 | **1** | KV头数（SharedKV） |
| head_dim | INT32 | **512** | 头维度 |
| cmp_topk | INT32 | 512/1024 | 稀疏token数 |
| cmp_ratio | INT32 | **4或128** | KV压缩率 |
| ori_mask_mode | INT32 | **4** | BAND模式（滑动窗口） |
| cmp_mask_mode | INT32 | **3** | RIGHT_DOWN_CAUSAL |
| ori_win_left | INT32 | **127** | 滑动窗口左边界 |
| ori_win_right | INT32 | **0** | 滑动窗口右边界 |
| layout_q | STRING | BSND/TND | Q数据排布 |
| layout_kv | STRING | **PA_ND** | KV数据排布 |

### 输出规格

| 参数名 | 类型 | 维度 | 说明 |
|--------|------|------|------|
| metadata | INT32 | **[1024]** | 固定大小，分核结果数组 |

---

## 第4页：核心计算流程（一）——基本块切分

### 基本块定义

```
基本块切分示意图:

          M轴（Q头组方向）                S2轴（KV序列方向）
    ┌───┬───┬───┬───┬───┬───┐        ┌───┬───┬───┬───┬───┐
    │B0 │B1 │B2 │B3 │...│尾块│        │B0 │B1 │B2 │B3 │尾块│
    └───┴───┴───┴───┴───┴───┘        └───┴───┴───┴───┴───┘
     │←── mBaseSize=64 ──→│            │←s2BaseSize=512→│

    每个基本块对应一次kernel循环迭代
```

### 切分公式

| 计算项 | 公式 | 示例（s1=128, s2=4096） |
|--------|------|------------------------|
| M轴块数 | `s1GBaseNum = ceil(s1 × groupSize / 64)` | ceil(128×64/64) = **128块** |
| S2轴块数 | `s2BaseNum = ceil(s2 / 512)` | ceil(4096/512) = **8块** |
| M轴尾块 | `s1GTailSize = (s1 × groupSize) % 64` | (128×64) % 64 = **0** |
| S2轴尾块 | `s2TailSize = s2 % 512` | 4096 % 512 = **0** |

### 公式依据
**与后续kernel循环一致性**: 后续`SparseAttnSharedkv`算子的内层循环按512切分S2轴，metadata切分必须与此一致，否则会产生额外的尾块处理开销。

---

## 第5页：核心计算流程（二）——滑窗注意力负载计算

### Sliding Window Attention（BAND模式）

```
KV Token有效范围示意图:

时间轴 t
    │
    │   Q位置: s1Idx
    │      │
    │      ▼
    │  ┌───┐                    ┌───────────────────────────────────┐
    │  │ Q │ ──────────────────▶│         KV有效窗口                │
    │  └───┘                    │  winLeft=127    winRight=0        │
    │                           │     ◀──────────────────▶          │
    │                           │         有效KV范围                 │
    │                           └───────────────────────────────────┘
    │
    │    KV位置: s2Idx
    │    有效范围: [s2FirstToken, s2LastToken]
    └
    ▼
```

### 计算公式推导

**Step 1**: 计算Q token在batch中的位置
```
s1FirstToken = 当前Q token索引
s1LastToken = s1FirstToken + mBaseSize/groupSize - 1 (当前块内最后一个)
```

**Step 2**: 计算KV token有效范围
```
s2FirstToken = s1FirstToken - (s2Size - s1Size + winLeft)
              = s1FirstToken - preTokenLeftUp

s2LastToken = s1LastToken + (s2Size - s1Size + winRight)
             = s1LastToken + nextTokenLeftUp
```

**Step 3**: 边界裁剪
```
s2FirstToken = max(0, min(s2FirstToken, s2Size-1))
s2LastToken = max(0, min(s2LastToken, s2Size-1))
```

### 公式依据

**Causal Attention语义**: 滑动窗口模式下，每个Q token只能看到以它为中心、左右各一定范围内的KV token。窗口大小=winLeft+winRight+1=128，保证每个Q最多计算128个KV。

**与kernel的一致性**: kernel通过`oriWinLeft`和`oriWinRight`参数控制mask，metadata需计算相同的有效范围，避免无效计算。

---

## 第6页：核心计算流程（三）——压缩注意力负载计算

### Compressed Attention（CFA/SCFA模式）

```
KV分布示意图:

    Ori KV (原始)                    Cmp KV (压缩)
    ┌─────────────────────────┐    ┌───────────────────┐
    │████████████████████████│    │░░░░░░░░░░░░░░░░░░░│
    │   滑动窗口内的KV         │    │  压缩后的稀疏KV   │
    │   (密集计算)             │    │  (稀疏计算)       │
    └─────────────────────────┘    └───────────────────┘
          ↑                              ↑
    winS2End                        cmpS2Start
    
    cmpS2Start = winS2End (压缩KV接续在原始KV之后)
```

### 计算公式

| 模式 | cmpS2LastTokenSize计算 | 依据 |
|------|------------------------|------|
| **CFA** | `floor((s2LastToken+1) / cmp_ratio)` | 全压缩，按比例选取 |
| **SCFA** | `min(floor((s2LastToken+1)/cmp_ratio), cmp_topk)` | 稀疏选取，限制topk |

### Block范围转换
```
cmpS2End = cmpS2Start + ceil(cmpS2LastTokenSize / s2BaseSize)
cmpS2TailSize = cmpS2LastTokenSize % s2BaseSize
```

### 公式依据

**KV Cache压缩机制**: 原始KV Cache通过QLI算法压缩，压缩率=cmp_ratio。后续kernel按压缩后的索引读取KV，metadata需计算正确的压缩范围。

**topk限制**: SCFA模式下，用户指定稀疏选取的token数上限，metadata需截断有效范围。

---

## 第7页：核心计算流程（四）——开销估算模型

### 开销模型设计原理

```
开销估算模型示意图:

    Block类型矩阵:
    ┌────────────────────────────────────────────────────┐
    │                    S2 Normal (512)                  │
    ├────────────────────┬───────────────────────────────┤
    │ WIN_NORMAL_BLOCK   │ WIN_NORMAL_S2_TAIL            │
    │ Cost(NormalM, 512) │ Cost(NormalM, TailS2)         │
    ├────────────────────┼───────────────────────────────┤
    │ M Tail (非64)      │                                │
    ├────────────────────┼───────────────────────────────┤
    │ WIN_M_TAIL         │ WIN_BOTH_TAIL                  │
    │ Cost(TailM, 512)   │ Cost(TailM, TailS2)           │
    └────────────────────┴───────────────────────────────┘
```

### 开销计算公式

```python
def calc_cost(basic_m, basic_s2):
    align_m = ceil(basic_m / 16)      # Cube单元对齐
    align_s2 = ceil(basic_s2 / 64)    # S2数据搬运对齐
    cost = 6 * align_m + 10 * align_s2
    return cost
```

### 系数来源分析

| 系数 | 来源 | 说明 |
|------|------|------|
| **6** | Matmul1计算复杂度 | Q@K^T的单次迭代开销 |
| **10** | Matmul2+数据搬运 | Softmax@V + KV加载开销 |
| **16** | Cube单元限制 | AscendC Cube一次处理16个元素 |
| **64** | 数据对齐要求 | S2方向最小搬运单位 |

### 开销模型用途
**负载均衡**: 根据估算开销分配任务，使各核负载相近，避免"慢核"拖慢整体性能。

---

## 第8页：核心计算流程（五）——负载均衡分配

### 三层分配策略

```
三层分配策略流程图:

┌─────────────────────────────────────────────────────────────────┐
│                    核负载上限 = avgCost                          │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: AssignByBatch                                          │
│ ├─ 条件: currentCost + batchCost ≤ costLimit + tolerance       │
│ ├─ 策略: 尽可能将完整Batch分配给单核                             │
│ └─ 优点: 减少跨batch边界                                         │
└─────────────────────────────────────┬───────────────────────────┘
                                      │ 批次太大无法整分配
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 2: AssignByRow                                            │
│ ├─ 条件: currentCost + rowCost ≤ costLimit + tolerance         │
│ ├─ 策略: 将完整M轴行分配给单核                                   │
│ └─ 优点: 保持S2循环连续性                                        │
└─────────────────────────────────────┬───────────────────────────┘
                                      │ 行太大无法整分配
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 3: AssignByBlock                                          │
│ ├─ 条件: currentCost + blockCost ≤ costLimit + tolerance       │
│ ├─ 策略: 按单个S2 Block分配                                      │
│ ├─ 注意: 记录跨核行信息（用于FlashDecode归约）                   │
│ └─ 优点: 精细控制负载均衡                                        │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 记录切分点:                                                      │
│ ├─ bN2End[coreIdx] = 当前BN2结束索引                            │
│ ├─ gS1End[coreIdx] = 当前GS1结束索引                            │
│ └─ s2End[coreIdx] = 当前S2结束索引                              │
└─────────────────────────────────────────────────────────────────┘
```

### 容差因子
```
FA_TOLERANCE_RATIO = 2
作用: 允许±2倍负载不均衡，避免过度切分导致开销增加
```

---

## 第9页：核心计算流程（六）——FlashDecode归约信息

### FlashDecode场景

```
FlashDecode归约场景示意图:

当一行(Batch, GS1)的S2轴被多个核切分时:

核0                核1                核2
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  S2[0-2]    │   │  S2[3-5]    │   │  S2[6-7]    │
│  部分计算    │   │  部分计算    │   │  部分计算    │
└─────────────┘   └─────────────┘   └─────────────┘
      │                 │                 │
      ▼                 ▼                 ▼
┌─────────────────────────────────────────────────┐
│              需要跨核归约                         │
│   output = softmax_partial0 + softmax_partial1 + ...│
└─────────────────────────────────────────────────┘
                      │
                      ▼ Vector核归约
```

### FD Metadata结构

| 字段 | 说明 |
|------|------|
| fdBN2Idx | 归约任务所属Batch*N2索引 |
| fdMIdx | 归约任务所属GS1索引 |
| fdWorkspaceIdx | 部分结果在workspace中的位置 |
| fdS2SplitNum | S2切分份数（需要归约的部分数） |
| fdMSize | 归约任务的M轴大小 |
| fdMStart/fdMNum | Vector核处理的M轴范围 |

### 公式依据

**FlashDecode原理**: 当KV序列过长时，将S2切分到多个核并行计算，最后由Vector核归约部分结果。metadata需记录归约信息，供后续kernel使用。

---

## 第10页：Metadata输出结构

### FA Metadata布局

```
FA Metadata (每个AI Core 8字段):

Core 0: [Enable=1, BN2_Start=0, M_Start=0, S2_Start=0, BN2_End=2, M_End=128, S2_End=8, FD_Wksp=0]
Core 1: [Enable=1, BN2_Start=2, M_Start=0, S2_Start=0, BN2_End=4, M_End=128, S2_End=8, FD_Wksp=1]
...
Core 35: [Enable=0, ...]  # 无任务的核，Enable=0

索引计算公式:
GetAttrAbsIndex(coreIdx, metaIdx) = coreIdx * 8 + metaIdx
```

### FD Metadata布局

```
FD Metadata (每个Vector Core 8字段):

Vec 0: [Enable=1, BN2_Idx=2, M_Idx=64, Wksp_Idx=0, Wksp_Num=3, M_Start=0, M_Num=32]
Vec 1: [Enable=1, BN2_Idx=2, M_Idx=64, Wksp_Idx=0, Wksp_Num=3, M_Start=32, M_Num=32]
...
Vec 71: [Enable=0, ...]  # 无归约任务

索引计算公式:
GetAttrAbsIndex(vecIdx, metaIdx, isAIV=true) = 36*8 + vecIdx*8 + metaIdx
```

### 总布局
```
metadata[1024] 总布局:
├─ [0-287]:   FA Metadata (36核 × 8字段)
├─ [288-863]: FD Metadata (72核 × 8字段)
└─ [864-1023]: 预留扩展空间
```

---

## 第11页：与后续算子的关联

### SparseAttnSharedkv使用Metadata的方式

```cpp
// Kernel中的Metadata使用代码片段 (sparse_attn_sharedkv_swa_kernel.h)

// 初始化时读取分核范围
void InitCalcParamsEach() {
    if (aiCoreIdx != 0) {
        constInfo.bN2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_BN2_START_INDEX));
        constInfo.gS1Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_M_START_INDEX));
        constInfo.s2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_S2_START_INDEX));
    }
    constInfo.bN2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_BN2_END_INDEX));
    constInfo.gS1End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_M_END_INDEX));
    constInfo.s2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_S2_END_INDEX));
}

// 按metadata切分循环
for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx < constInfo.bN2End; bN2LoopIdx++) {
    for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx < gS1LoopEnd; gS1LoopIdx++) {
        for (uint32_t s2LoopIdx = constInfo.s2Start; s2LoopIdx < s2LoopEnd; s2LoopIdx++) {
            // 执行Q@K^T, Softmax, Softmax@V计算
        }
    }
}
```

### 关联总结表

| Metadata字段 | Kernel使用方式 | 公式依据 |
|--------------|----------------|----------|
| bN2Start/bN2End | 外层Batch循环边界 | Batch切分粒度 |
| gS1Start/gS1End | 中层M轴循环边界 | M轴切分粒度=groupSize |
| s2Start/s2End | 内层S2循环边界 | S2切分粒度=512 |
| FD Metadata | Vector核归约任务分配 | FlashDecode跨核归约 |

---

## 第12页：Host端重设计要点

### 迁移策略对比

| 维度 | AICPU实现 | Host端TileLang实现 |
|------|-----------|-------------------|
| **运行位置** | NPU辅助CPU | Host端CPU |
| **编程语言** | AscendC (C++) | TileLang |
| **并行策略** | 单线程 | 多线程并行 |
| **数据访问** | NPU内存 | Host内存 |
| **输出格式** | 完全一致 | 完全一致 |

### 关键迁移点

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 数据结构迁移                                                  │
│    SplitResult, SplitInfo, CostInfo等结构体定义不变              │
├─────────────────────────────────────────────────────────────────┤
│ 2. 算法逻辑迁移                                                  │
│    CalcSplitInfo → CalcCostInfo → BalanceSchedule → GenMetadata │
│    完全复用原有计算逻辑                                           │
├─────────────────────────────────────────────────────────────────┤
│ 3. 输出格式迁移                                                  │
│    metadata[1024] INT32数组，布局完全一致                        │
├─────────────────────────────────────────────────────────────────┤
│ 4. 性能优化点                                                    │
│    ├─ 多batch并行计算切分信息                                    │
│    ├─ 开销计算使用SIMD加速                                       │
│    └─ 负载均衡使用并行分配算法                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第13页：约束分析汇总

### 输入约束

| 约束项 | 固定值/范围 | 校验逻辑 |
|--------|-------------|----------|
| num_heads_q | **64** | 参数校验 |
| num_heads_kv | **1** | 参数校验（SharedKV特性） |
| head_dim | **512** | 参数校验 |
| cmp_topk | 512或1024 | 参数校验 |
| cmp_ratio | **4或128** | 参数校验 |
| ori_mask_mode | **4** | 参数校验（仅支持BAND） |
| ori_win_left | **127** | 参数校验 |
| ori_win_right | **0** | 参数校验 |

### 输出约束

| 约束项 | 固定值 | 说明 |
|--------|--------|------|
| metadata维度 | **[1024]** | 固定大小，供后续kernel使用 |
| metadata类型 | **INT32** | 索引类型 |
| FA Metadata布局 | 36核×8字段 | 与kernel读取逻辑一致 |
| FD Metadata布局 | 72核×8字段 | 与kernel归约逻辑一致 |

### 功能约束

| 约束项 | 说明 |
|--------|------|
| 无AICPU依赖 | 不使用NPU辅助CPU特性 |
| 输出兼容性 | 必须与原算子输出完全一致 |
| 性能要求 | Host端计算开销应小于AICPU开销 |

---

## 第14页：实现路线图

### Phase 1: 基础实现
```
├─ 数据结构定义
│   └─ SplitInfo, CostInfo, SplitResult, Metadata结构
├─ 基础函数实现
│   ├─ GetS1SeqSize, GetS2SeqSize
│   ├─ CalcSplitInfo
│   └─ CalcS2TokenRange (Mask范围计算)
├─ 单batch测试验证
│   └─ 与AICPU输出对比
```

### Phase 2: 负载均衡
```
├─ 开销模型实现
│   ├─ CalcCost (Block开销)
│   ├─ CalcCostTable
│   └─ CalcCostInfo (全局开销)
├─ 分配算法实现
│   ├─ AssignByBatch
│   ├─ AssignByRow
│   ├─ AssignByBlock
│   └─ BalanceSchedule (总调度)
├─ 多batch测试验证
```

### Phase 3: Metadata生成
```
├─ FA Metadata生成
│   └─ 核启用标志、切分索引写入
├─ FD Metadata生成
│   ├─ 归约信息记录
│   └─ SplitFD (归约负载均衡)
├─ 输出组装
│   └─ metadata数组填充
├─ 全场景测试验证
```

### Phase 4: 性能优化
```
├─ 多线程并行计算
├─ SIMD加速开销计算
├─ 缓存优化
├─ 性能基准测试
```

---

## 第15页：总结

### 关键成果
1. 完整分析了`SparseAttnSharedkvMetadata`算子的计算流程
2. 建立了与后续`SparseAttnSharedkv`算子的关联公式依据
3. 设计了Host端TileLang实现方案

### 核心公式体系
| 公式类别 | 关键公式 |
|----------|----------|
| 基本块切分 | `s1GBaseNum=ceil(s1×G/64)`, `s2BaseNum=ceil(s2/512)` |
| Mask范围 | `s2First=s1-preTokenLeftUp`, `s2Last=s1+nextTokenLeftUp` |
| 开销估算 | `Cost=6×align(M/16)+10×align(S2/64)` |
| 负载均衡 | 三层分配: Batch→Row→Block |

### 后续工作
- TileLang代码实现
- 与原算子输出对比验证
- 性能优化与基准测试

---

## 附录：代码位置索引

| 文件 | 路径 | 核心内容 |
|------|------|----------|
| AICPU Kernel | `sparse_attn_sharedkv_metadata_aicpu.cpp` | 完整实现逻辑 |
| Metadata结构 | `sparse_attn_sharedkv_metadata.h` | 输出结构定义 |
| Kernel使用 | `sparse_attn_sharedkv_swa_kernel.h` | Metadata读取逻辑 |
| Golden实现 | `sparse_attn_sharedkv_golden.py` | Python参考实现 |