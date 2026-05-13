# AlltoAllMatmul 去除转置节点优化方案

## 一、优化背景

### 1.1 当前问题

当前 MX-FP8 tensor_api 分支的流水线：
```
通信(AllToAll) -> 数据转置 -> Scale转置 -> matmul(k=rankK*rankDim)
```

转置节点存在以下开销：
- **数据搬运开销**：完整数据量 commOut -> transOut
- **内存占用**：需要额外分配 transOut 缓存空间
- **流水线开销**：转置节点增加了流水线长度

### 1.2 优化目标

仅针对 mxfp8 场景使用 tensor_api 的分支进行优化：
- 去掉数据转置节点 `transStage_`
- 去掉 Scale 转置节点 `scaleTransStage_`
- 直接使用通信完成的数据进行 matmul 计算
- 利用 GM 自动累加特性，多次调用 matmul 完成跨 rank 的 K 轴累加

---

## 二、核心思路

### 2.1 转置目的分析

转置节点的目的：将分散在各 rank 的 K 轴数据合并为连续的 K 轴

**通信输出布局**（以 rankDim=4 为例）：
```
commOutGM_: [rank0数据][rank1数据][rank2数据][rank3数据]
            每个rank: [tileM/rankDim, rankK]
```

**转置后布局**：
```
transOutGM_: [tileM, rankK*rankDim]（连续K轴）
```

### 2.2 优化核心思想

去掉转置节点后，matmul 直接使用通信输出的数据：

```
通信输出: [rank0][rank1][rank2][rank3]  (非连续K轴)
                ↓
多次调用matmul: 每次处理一个rank的 [tileM, rankK]
                ↓
GM自动累加: 最终结果 = sum(rank0结果 + rank1结果 + rank2结果 + rank3结果)
```

**关键点**：
- matmul 的 K 轴从 `rankK*rankDim` 改为 `rankK`
- 外层添加 rank 循环，每次处理一个 rank 的数据
- 结果在 GM 自动累加（用户已确认硬件支持）

---

## 三、数据流变化

### 3.1 当前流程

```
┌─────────┐   ┌──────┐   ┌─────────────┐   ┌─────────────┐   ┌───────────────────┐
│ 通信    │ → │ Sync │ → │ 数据转置    │ → │ Scale转置   │ → │ matmul(k=rankK*4) │
│(AIV)    │   │      │   │(AIV)        │   │(AIV)        │   │(AIC, BlockMmad)  │
└─────────┘   └──────┘   └─────────────┘   └─────────────┘   └───────────────────┘
     ↓                         ↓                   ↓                    ↓
commOutGM_              transOutGM_        transX1ScaleGM_            y_
commX1ScaleGM_
```

### 3.2 优化后流程

```
┌─────────┐   ┌──────┐   ┌─────────────────────────────────────────────────┐
│ 通信    │ → │ Sync │ → │ matmul(rank循环, k=rankK, GM累加)             │
│(AIV)    │   │      │   │(AIC)                                          │
└─────────┘   └──────┘   └─────────────────────────────────────────────────┘
     ↓                         ↓                                    ↓
commOutGM_              (直接用于matmul)                           y_
commX1ScaleGM_
```

---

## 四、通信数据布局详解（stride 分散布局）

### 4.1 通信与计算的流水掩盖

**背景**：为了实现通信与计算的流水掩盖，数据被切分为多个 tile。

```
fullM = 384      （全局 M 维度，每张卡持有的数据量）
rankDim = 4      （4 张卡）
tileCnt = 8      （切分为 8 个 tile）

★★★ 关键：tiling->tileM 是单个 rank 区的 M 维度 ★★★
tiling->tileM = fullM / rankDim / tileCnt = 384 / 4 / 8 = 12

命名约定：
- tiling->tileM = 12：单个 rank 区在单个 tile 内的 M 维度数据量
- fullM/rankDim = 96：单个 rank 区持有的完整 M 维度数据量（所有 tile）
- tileM * rankDim = 48：一个 tile 内所有 rank 区的 M 维度数据量总和
```

**每次通信的数据量**：
- 发送：`tiling->tileM = 12` 的数据给其他 3 张卡
- 接收：`tiling->tileM * 3 = 36` 的数据来自其他 3 张卡
- 总计：`tiling->tileM * rankDim = 48` 的数据（包含来自所有卡的）

---

### 4.2 ★★★ 关键理解：stride 是全局的 ★★★

通信完成后，数据是 **stride 分散布局**，不是连续布局！

**stride 基于 fullM 计算**，不是基于 tiling->tileM：

```cpp
// 全局 stride（rank 之间的间隔）
globalStride = fullM / rankDim * rankK * sizeof(DTYPE_X1)
             = 96 * 1024 * 1 = 98304 bytes

// tile stride（同一 rank 内，不同 tile 之间的间隔）
tileStride = tiling->tileM * rankK * sizeof(DTYPE_X1)
            = 12 * 1024 * 1 = 12288 bytes

★★★ 注意：tiling->tileM 已经是 fullM/rankDim/tileCnt ★★★
★★★ 所以 tileStride = tiling->tileM * rankK，不需要再除 rankDim ★★★
```

---

### 4.3 通信输出布局图解

```
commOutGM_ 的全局布局（stride 分散）：

┌─────────────────────────────────────────────────────────────────────────────┐
│                              rank0 数据区                                   │
│                    （offset 范围：0 ~ globalStride-1 = 98304-1）            │
│ ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────┐│
│ │ tileIdx=0 │ tileIdx=1 │ tileIdx=2 │ tileIdx=3 │ ...       │tileIdx=7││
│ │ offset:0  │ offset:   │ offset:   │ offset:   │           │ offset: ││
│ │           │ tileStride│ 2*tileStr │ 3*tileStr │           │7*tileStr││
│ │           │ =12288    │ =24576    │ =36864    │           │=86016  ││
│ │[tileM,K] │[tileM,K] │[tileM,K] │[tileM,K] │           │[tileM,K]││
│ │=[12,1024]│=[12,1024]│=[12,1024]│=[12,1024]│           │=[12,1024]││
│ └────────────┴────────────┴────────────┴────────────┴────────────┴────────┘│
│                                                                             │
│                              rank1 数据区                                   │
│           （offset 范围：globalStride ~ 2*globalStride-1 = 98304~196608-1）│
│ ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────┐│
│ │ tileIdx=0 │ tileIdx=1 │ tileIdx=2 │ tileIdx=3 │ ...       │tileIdx=7││
│ │ offset:   │ offset:   │ offset:   │ offset:   │           │ offset: ││
│ │ global    │ global+   │ global+   │ global+   │           │ global+ ││
│ │ Stride    │ tileStride│2*tileStr │3*tileStr │           │7*tileStr││
│ │ =98304    │=110592    │=120960    │=131328    │           │=184320  ││
│ │[tileM,K] │[tileM,K] │[tileM,K] │[tileM,K] │           │[tileM,K] ││
│ │=[12,1024]│=[12,1024]│=[12,1024]│=[12,1024]│           │=[12,1024]││
│ └────────────┴────────────┴────────────┴────────────┴────────────┴────────┘│
│                                                                             │
│                              rank2 数据区                                   │
│         （offset 范围：2*globalStride ~ 3*globalStride-1 = 196608~294912-1）│
│ 类似结构...                                                                  │
│                                                                             │
│                              rank3 数据区                                   │
│         （offset 范围：3*globalStride ~ 4*globalStride-1 = 294912~393216-1）│
│ 类似结构...                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

★★★ 关键理解 ★★★
- globalStride = 98304：rank 数据区之间的间隔（预留空间）
- tileStride = 12288：同一 rank 区内，不同 tile 之间的间隔
- 每个数据块：[tiling->tileM, rankK] = [12, 1024]
- 数据量 = 12 * 1024 = 12288（正好等于 tileStride，无空洞！）
- 注意：tiling->tileM 已经是单个 rank 区的 M 维度，不需要再除 rankDim
```

---

### 4.4 地址计算公式

**数据地址计算**：

```cpp
// 全局 stride（rank 之间的间隔）
globalStride = fullM / rankDim * rankK * sizeof(DTYPE_X1)

// tile stride（同一 rank 内，不同 tile 之间的间隔）
// ★★★ tiling->tileM 已经是 fullM/rankDim/tileCnt ★★★
tileStride = tiling->tileM * rankK * sizeof(DTYPE_X1)

// 地址计算公式
for (rankIdx = 0; rankIdx < rankDim; rankIdx++) {
    aGM_rank = commOutGM_ 
               + rankIdx * globalStride       // 跳到对应 rank 数据区
               + taskIndex * tileStride       // 在 rank 数据区内跳到对应 tile
    
    // ★★★ 每次 matmul 处理 [tiling->tileM, rankK] 的数据 ★★★
    // ★★★ 所有 rank 的结果累加到同一 GM 位置 ★★★
}
```

**示例计算**（tileIdx=1, rankIdx=2）：
```
aGM_rank = commOutGM_ 
           + 2 * 98304       // rank2 数据区起始
           + 1 * 12288       // tileIdx=1 的数据
           = commOutGM_ + 196608 + 12288
           = commOutGM_ + 208896
```

---

### 4.5 Scale 的 stride 计算

MX-FP8 的 Scale 按 32 个元素一组，每组 2 个 fp8_e8m0 值：

```cpp
// 全局 stride（rank 之间的 Scale 间隔）
scaleGlobalStride = fullM / rankDim * (rankK / 32 * 2) * sizeof(fp8_e8m0)
                  = 96 * 64 * 1 = 6144 bytes

// tile stride（同一 rank 内，不同 tile 之间的 Scale 间隔）
// ★★★ tiling->tileM 已经是 fullM/rankDim/tileCnt ★★★
scaleTileStride = tiling->tileM * (rankK / 32 * 2) * sizeof(fp8_e8m0)
               = 12 * 64 * 1 = 768 bytes

// 地址计算公式
for (rankIdx = 0; rankIdx < rankDim; rankIdx++) {
    scaleAGM_rank = commX1ScaleGM1_
                    + rankIdx * scaleGlobalStride    // 跳到对应 rank Scale 区
                    + taskIndex * scaleTileStride    // 在 rank Scale 区内跳到对应 tile
}
```

---

### 4.6 与原代码的对比

| 参数 | 原代码（转置后） | 新方案（直接使用通信输出） |
|------|-----------------|-------------------------|
| **aOffset** | `tileM * rankK * rankDim`（转置后连续 stride） | **不使用** |
| **recvOffset** | `tileM * rankK`（原代码 tile 间隔） | **不使用** |
| **globalStride** | - | `fullM/rankDim * rankK`（rank 数据区间隔） |
| **tileStride** | - | `tileM/rankDim * rankK`（rank 内 tile 间隔） |

---

### 4.7 关键注意事项

1. **stride 是全局的**：globalStride 基于 fullM 计算，不是基于 tiling->tileM
2. **数据分散布局**：通信输出不是连续的，数据之间有空洞
3. **地址计算需考虑两个 stride**：globalStride（跨 rank）+ tileStride（跨 tile）
4. **tiling->tileM 的含义**：单个 rank 区的 M 维度，不需要再除 rankDim

### 4.8 ★★★ tiling 侧校验注意事项 ★★★

**问题**：tiling 参数需要根据新的 K 轴变化进行校验。

**需要关注的 tiling 参数**：

| 参数 | 原值（转置后） | 新值（直接使用通信输出） | 校验要点 |
|------|---------------|------------------------|---------|
| **fullK** | `rankK * rankDim` | `rankK` | L1/L0 缓冲规划需重新计算 |
| **baseK** | 基于 `rankK*rankDim` | 基于 `rankK` | 是否满足 L0 缓冲约束 |
| **kL1** | 基于 `rankK*rankDim` | 基于 `rankK` | 是否满足 L1 缓冲约束 |
| **scaleKL1** | 基于 `rankK*rankDim` | 基于 `rankK` | Scale 复用窗口是否合理 |

**校验要点**：

1. **L1 缓冲校验**：
   - `baseM * kL1 * sizeof(DTYPE_X1)` 是否超过 L1 缓冲容量？
   - `scaleKL1` 的 Scale 数据量是否合理？

2. **L0 缓冲校验**：
   - `baseM * baseK * sizeof(DTYPE_X1)` 是否超过 L0A 缓冲容量？
   - `baseK * baseN * sizeof(DTYPE_X2)` 是否超过 L0B 缓冲容量？

3. **baseK 对齐校验**：
   - baseK 需满足硬件对齐要求（通常是 16 或 32 的倍数）

**建议**：在 `allto_all_mx_quant_matmul_tiling_base.cpp` 的 `FillMxTensorTiling` 函数中添加校验逻辑。

## 五、详细修改方案

### 5.1 修改 `allto_all_mx_quant_matmul_pipeline.h`

#### 5.1.1 去掉转置节点模板参数

```cpp
// 原:
template <typename CommunicationType, typename TransposeType, 
          typename ScaleTransposeType, typename ComputationType, typename ContextType>
class AlltoAllMXQuantMatmulPipeLine {
    CommunicationType *commStage_;
    TransposeType *transStage_;           // 移除
    ScaleTransposeType *scaleTransStage_; // 移除
    ComputationType *computeStage_;
};

// 改:
template <typename CommunicationType, typename ComputationType, typename ContextType>
class AlltoAllMXQuantMatmulPipeLine {
    CommunicationType *commStage_;
    ComputationType *computeStage_;
};
```

#### 5.1.2 修改 Process 函数

**位置**: `allto_all_mx_quant_matmul_pipeline.h:84-102`

```cpp
// 原:
__aicore__ inline void Process(uint32_t taskCnt) {
    commStage_->PrepareAll(taskCnt);
    for (uint32_t index = 0; index < taskCnt; index++) {
        if ASCEND_IS_AIV {
            commStage_->Process(index);
            AscendC::SyncAll<true>();
            transStage_->Process(index);            // 移除
            AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8);
            AscendC::CrossCoreWaitFlag(8);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(9);
            computeStage_->Process(index);
        }
    }
}

// 改:
__aicore__ inline void Process(uint32_t taskCnt) {
    commStage_->PrepareAll(taskCnt);
    for (uint32_t index = 0; index < taskCnt; index++) {
        if ASCEND_IS_AIV {
            commStage_->Process(index);
            AscendC::SyncAll<true>();
            // 移除 transStage_->Process(index);
            AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8);
            AscendC::CrossCoreWaitFlag(8);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(9);
            computeStage_->Process(index);
        }
    }
}
```

#### 5.1.3 修改 ProcessScale 函数

**位置**: `allto_all_mx_quant_matmul_pipeline.h:107-116`

```cpp
// 原:
__aicore__ inline void ProcessScale() {
    if ASCEND_IS_AIV {
        commStage_->PrepareAll(1);
        commStage_->Process(0);
        AscendC::SyncAll<true>();
        scaleTransStage_->Process(0);  // 移除
    }
}

// 改:
__aicore__ inline void ProcessScale() {
    if ASCEND_IS_AIV {
        commStage_->PrepareAll(1);
        commStage_->Process(0);
        AscendC::SyncAll<true>();
        // 移除 scaleTransStage_->Process(0);
    }
}
```

#### 5.1.4 修改 Context 结构体

```cpp
// 原:
template <typename ComputationContextType>
struct AlltoAllMXQmmPipelineContext {
    ComputationContextType *computationContext;
    MC2KernelTemplate::MC2TransposeContext *transposeContext;        // 移除
    MC2KernelTemplate::MC2TransposeContext *scaleTransposeContext;   // 移除
    MC2KernelTemplate::MC2AlltoAllContext *communicationContext;
};

// 改:
template <typename ComputationContextType>
struct AlltoAllMXQmmPipelineContext {
    ComputationContextType *computationContext;
    MC2KernelTemplate::MC2AlltoAllContext *communicationContext;
};
```

---

### 5.2 修改 `allto_all_mx_quant_matmul_arch35.h`

#### 5.2.1 去掉转置输出地址成员

**位置**: `allto_all_mx_quant_matmul_arch35.h:50-53`

```cpp
// 原:
GM_ADDR commX1ScaleGM1_;
GM_ADDR commOutGM_;
GM_ADDR transX1ScaleGM1_;  // 移除
GM_ADDR transOutGM_;       // 移除

// 改:
GM_ADDR commX1ScaleGM1_;   // Scale 通信输出（直接用于matmul）
GM_ADDR commOutGM_;        // 数据通信输出（直接用于matmul）
```

#### 5.2.2 修改 Init 函数

**位置**: `allto_all_mx_quant_matmul_arch35.h:66-104`

```cpp
// 原:
commX1ScaleGM1_ = workspaceGM;
uint64_t x1ScaleLen = CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO *
             static_cast<uint64_t>(mc2Tiling_.rankM) * sizeof(AscendC::fp8_e8m0_t);
x1ScaleLen = CeilDiv(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM;
commOutGM_ = workspaceGM + x1ScaleLen;
transX1ScaleGM1_ = commOutGM_ + mc2Tiling_.commLen;  // 移除
transOutGM_ = all2all_out;                            // 移除
if (all2all_out == nullptr) {
    transOutGM_ = transX1ScaleGM1_ + x1ScaleLen;     // 移除
}

// 改:
commX1ScaleGM1_ = workspaceGM;
uint64_t x1ScaleLen = CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO *
             static_cast<uint64_t>(mc2Tiling_.rankM) * sizeof(AscendC::fp8_e8m0_t);
x1ScaleLen = CeilDiv(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM;
commOutGM_ = workspaceGM + x1ScaleLen;
// 移除 transX1ScaleGM1_ 和 transOutGM_ 的计算
```

#### 5.2.3 修改 ProcessScale 函数

**位置**: `allto_all_mx_quant_matmul_arch35.h:131-159`

```cpp
// 原:
pipeLineContext_.scaleTransposeContext->transposeSrcAddr = commX1ScaleGM1_;   // 移除
pipeLineContext_.scaleTransposeContext->transposeDstAddr = transX1ScaleGM1_; // 移除
// ... 其他 scaleTransposeContext 参数                        // 移除
pipeLine_->ProcessScale();

// 改:
// 移除 scaleTransposeContext 相关参数设置
pipeLine_->ProcessScale();
```

#### 5.2.4 修改 ProcessTile 函数

**位置**: `allto_all_mx_quant_matmul_arch35.h:161-225`

**关键修改**：

```cpp
// 原:
pipeLineContext_.transposeContext->transposeSrcAddr = commOutGM_;           // 移除
pipeLineContext_.transposeContext->transposeDstAddr = transOutGM_;          // 移除
// ... 其他 transposeContext 参数                            // 移除

pipeLineContext_.computationContext->baseData.aGM = transOutGM_;             // 改为 commOutGM_
pipeLineContext_.computationContext->additionalData.x1Scale = transX1ScaleGM1_; // 改为 commX1ScaleGM1_

tilingData_->mxTensorTileTilingData.fullK = mc2Tiling_.rankK * mc2Tiling_.rankDim; // 改为 rankK

// 改:
// 移除 transposeContext 相关参数设置

pipeLineContext_.computationContext->baseData.aGM = commOutGM_;  // 直接使用通信输出
pipeLineContext_.computationContext->additionalData.x1Scale = commX1ScaleGM1_; // 直接使用通信输出

// ★★★ Tiling 参数修改 ★★★
tilingData_->mxTensorTileTilingData.fullK = mc2Tiling_.rankK;  // 改为 rankK（每次处理一个 rank）

// ★★★ 新增：传入 stride 参数（基于第四章的正确理解）★★★
pipeLineContext_.computationContext->rankDim = mc2Tiling_.rankDim;

// globalStride：rank 数据区之间的间隔（基于全局 fullM）
// 注意：mc2Tiling_.rankM 是全局 M 维度，等于 fullM
pipeLineContext_.computationContext->globalStride = 
    static_cast<uint64_t>(mc2Tiling_.rankM) / mc2Tiling_.rankDim 
    * mc2Tiling_.rankK * sizeof(DTYPE_X1);

// ★★★ tileStride：同一 rank 区内，不同 tile 之间的间隔 ★★★
// 注意：mc2Tiling_.tileM 已经是 fullM/rankDim/tileCnt，不需要再除 rankDim
pipeLineContext_.computationContext->tileStride = 
    static_cast<uint64_t>(mc2Tiling_.tileM) 
    * mc2Tiling_.rankK * sizeof(DTYPE_X1);

// Scale 的 globalStride（新增）
pipeLineContext_.computationContext->scaleGlobalStride = 
    static_cast<uint64_t>(mc2Tiling_.rankM) / mc2Tiling_.rankDim 
    * CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO * sizeof(AscendC::fp8_e8m0_t);

// ★★★ Scale 的 tileStride ★★★
// 注意：mc2Tiling_.tileM 已经是 fullM/rankDim/tileCnt，不需要再除 rankDim
pipeLineContext_.computationContext->scaleTileStride = 
    static_cast<uint64_t>(mc2Tiling_.tileM) 
    * CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO * sizeof(AscendC::fp8_e8m0_t);
```

**重要说明**：
- stride 是全局的，基于 fullM（即 mc2Tiling_.rankM）计算
- **mc2Tiling_.tileM 已经是单个 rank 区的 M 维度**，不需要再除 rankDim
- 地址计算公式：`offset = rankIdx * globalStride + taskIndex * tileStride`
- 原 `baseData.aOffset` 不再使用

#### 5.2.5 修改 ProcessTail 函数

类似 ProcessTile 的修改，确保：
- aGM 使用 commOutGM_ 的正确偏移
- x1Scale 使用 commX1ScaleGM1_ 的正确偏移
- fullK 改为 rankK

---

### 5.3 修改 `tensor_mx_matmul_stage.h`

#### 5.3.1 添加 rank 循环与 GM 自动累加

**位置**: `tensor_mx_matmul_stage.h:65-144`

```cpp
template <typename TilingType, bool X2TRANSPOSE>
__aicore__ inline void TensorMXQuantMatmulStage<TilingType, X2TRANSPOSE>::Process(uint32_t taskIndex)
{
    if ASCEND_IS_AIV { return; }
    
    auto& base = MMcontext_.baseData;
    auto& add  = MMcontext_.additionalData;
    
    // ★★★ 获取 stride 参数（基于第四章的正确理解）★★★
    uint32_t rankDim = MMcontext_.rankDim;
    uint64_t globalStride = MMcontext_.globalStride;    // rank 数据区之间的间隔
    uint64_t tileStride = MMcontext_.tileStride;        // 同一 rank 区内，tile 之间的间隔
    uint64_t scaleGlobalStride = MMcontext_.scaleGlobalStride;
    uint64_t scaleTileStride = MMcontext_.scaleTileStride;
    
    tPipePtr_->Reset();
    
    auto* tiling = MMcontext_.tilingDataPtr;
    // ★★★ tiling->tileM 是单个 rank 区的 M 维度 ★★★
    uint32_t m_ = tiling->tileM;              // 单个 rank 区的 M 维度（用于 LayoutA 和 LayoutC）
    uint32_t k_ = tiling->fullK;              // rankK
    uint32_t n_ = tiling->fullN;
    
    // 外层循环：处理每个 rank 的数据
    for (uint32_t rankIdx = 0; rankIdx < rankDim; rankIdx++) {
        // ★★★ rankIdx >= 1 时开启累加 ★★★
        // rankIdx=0 默认不累加，直接写入初始化 GM（清除脏数据）
        if (rankIdx >= 1) {
            SetAtomicAdd<CType>();
        }
        
        // ★★★ 地址计算：globalStride（跨 rank）+ tileStride（跨 tile）★★★
        GM_ADDR aGM_rank = base.aGM 
                           + rankIdx * globalStride       // 跳到对应 rank 数据区
                           + taskIndex * tileStride;      // 在 rank 数据区内跳到对应 tile
        
        // ★★★ C 地址固定，所有 rank 累加到同一位置 ★★★
        GM_ADDR cGM_rank = base.cGM + taskIndex * base.cOffset;
        
        GM_ADDR x1ScaleGM_rank = add.x1Scale 
                                  + rankIdx * scaleGlobalStride
                                  + taskIndex * scaleTileStride;
        
        // BlockMmad 初始化
        BlockOp mxMmadOp_;
        typename BlockOp::L1Params l1Params{tiling->kL1, tiling->scaleKL1};
        typename BlockOp::TupleShape problemShape{
            static_cast<int64_t>(m_),   // ★★★ m_ = tiling->tileM（单个 rank 区的 M 维度）★★★
            static_cast<int64_t>(n_), 
            static_cast<int64_t>(k_)};  // k_ = rankK
        typename BlockOp::BlockShape l0TileShape{
            static_cast<int64_t>(tiling->baseM), 
            static_cast<int64_t>(tiling->baseN), 
            static_cast<int64_t>(tiling->baseK), 0};
        mxMmadOp_.Init(problemShape, l0TileShape, l1Params, tiling->dbL0c > 1);
        
        uint32_t kScaleGrp = (k_ + 32 - 1) / 32;
        
        // ★★★ Layout 构造 ★★★
        // LayoutA 使用 m_（单个 rank 区的 M 维度）
        // LayoutC 使用 m_（所有 rank 累加到同一位置）
        auto layoutA = AscendC::Te::MakeNDLayout<AType>(m_, k_);      // [tiling->tileM, rankK]
        auto layoutB = AscendC::Te::MakeDNLayout<BType>(k_, n_);
        auto layoutC = AscendC::Te::MakeNDLayout<CType>(m_, n_);      // [tiling->tileM, N]
        auto layoutSA = AscendC::Te::MakeScaleANDLayout<ScaleAType>(m_, kScaleGrp);
        auto layoutSB = AscendC::Te::MakeScaleBDNLayout<ScaleBType>(kScaleGrp, n_);
        
        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ AType*>(aGM_rank)), layoutA);
        auto gmB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ BType*>(base.bGM)), layoutB);
        auto gmC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ CType*>(cGM_rank)), layoutC);
        auto gmScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ ScaleAType*>(x1ScaleGM_rank)), layoutSA);
        auto gmScaleB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ ScaleBType*>(add.x2Scale)), layoutSB);
        
        ProblemShape problem{m_, n_, k_, 1L};
        using SchParams = typename SchedulerOp::Params;
        SchParams schParams{
            static_cast<int64_t>(tiling->baseM), 
            static_cast<int64_t>(tiling->baseN),
            static_cast<int64_t>(tiling->mTailTile), 
            static_cast<int64_t>(tiling->nTailTile),
            static_cast<int64_t>(tiling->mBaseTailSplitCnt), 
            static_cast<int64_t>(tiling->nBaseTailSplitCnt),
            static_cast<int64_t>(tiling->mTailMain), 
            static_cast<int64_t>(tiling->nTailMain)};
        SchedulerOp bs(problem, schParams);
        
        using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
        BlockCoord blockIdx;
        constexpr int64_t kPos = 0L;
        
        while (bs.GetTileIdx(blockIdx)) {
            int64_t mPos = AscendC::Std::get<MNK_M>(blockIdx);
            int64_t nPos = AscendC::Std::get<MNK_N>(blockIdx);
            auto singleShape = bs.GetBlockShape(blockIdx);
            
            if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) { break; }
            
            auto gmBlockA = gmA(AscendC::Te::MakeCoord(mPos, kPos), 
                               AscendC::Te::MakeShape(Get<MNK_M>(singleShape), k_));
            auto gmBlockB = gmB(AscendC::Te::MakeCoord(kPos, nPos), 
                               AscendC::Te::MakeShape(k_, Get<MNK_N>(singleShape)));
            auto gmBlockC = gmC(AscendC::Te::MakeCoord(mPos, nPos), 
                               AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
            auto gmBlockScaleA = gmScaleA(AscendC::Te::MakeCoord(mPos, kPos), 
                                           AscendC::Te::MakeShape(Get<MNK_M>(singleShape), kScaleGrp));
            auto gmBlockScaleB = gmScaleB(AscendC::Te::MakeCoord(kPos, nPos), 
                                           AscendC::Te::MakeShape(kScaleGrp, Get<MNK_N>(singleShape)));
            
            // BlockMmad 计算
            // rankIdx=0: 默认不累加，直接写入 GM（初始化）
            // rankIdx>=1: 自动累加到 GM
            mxMmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);
        }
        
        // ★★★ 同步：确保不同 rankIdx 的 matmul 计算完成 ★★★
        AscendC::SyncAll<true>();
    }
    
    // ★★★ 循环结束后关闭累加 ★★★
    SetAtomicNone();
}
```

---

### 5.4 修改 `matmul_base.h` 或 `mc2_mx_quant_matmul.h`

#### 5.4.1 扩展 MC2MMContext 结构体

```cpp
// 原有结构体（位置: matmul_base.h 或 mc2_mx_quant_matmul.h）
template <typename AdditionalDataType, typename TilingType>
struct MC2MMContext {
    BaseMMData baseData;
    AdditionalDataType additionalData;
    TilingType* tilingDataPtr;
    
    // ★★★ 新增字段：用于 stride 计算（基于第四章的正确理解）★★★
    uint32_t rankDim;              // rank 数量（用于外层循环）
    uint64_t globalStride;         // rank 数据区之间的间隔（基于 fullM）
    uint64_t tileStride;           // 同一 rank 区内，tile 之间的间隔
    uint64_t scaleGlobalStride;    // Scale 的 rank 数据区之间的间隔
    uint64_t scaleTileStride;      // Scale 的同一 rank 区内，tile 之间的间隔
};
```

---

### 5.5 修改 `allto_all_matmul_apt.cpp`

#### 5.5.1 修改宏定义 ALLTO_ALL_MX_TENSOR_MATMUL_IMPL

**位置**: `allto_all_matmul_apt.cpp:108-135`

```cpp
// 原:
#define ALLTO_ALL_MX_TENSOR_MATMUL_IMPL(tilingData, pipe, commDataTypeX1)                                              \
    do {                                                                                                               \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(...);                                                                        \
        CommunicationType commImplName(&tilingData);                                                                   \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(commDataTypeX1, TransposeType);            // 移除                    \
        TransposeType transposeImplName(&pipe);                                            // 移除                    \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(AscendC::fp8_e8m0_t, ScaleTransposeType); // 移除                    \
        ScaleTransposeType scaleTransposeImplName(&pipe);                                  // 移除                    \
        using ComputationContextType = ...;                                                                            \
        using ComputationType = Mc2Kernel::TensorMXQuantMatmulStage<...>;                                              \
        ComputationType matmulImplName(&pipe);                                                                         \
        using SchedulerContextType = Mc2Kernel::AlltoAllMXQmmPipelineContext<ComputationContextType>;                  \
        using SchedulerType = Mc2Kernel::AlltoAllMXQuantMatmulPipeLine<                 \
            CommunicationType, TransposeType, ScaleTransposeType, ComputationType, SchedulerContextType>; // 移除 TransposeType\
        SchedulerType SchedulerImpl(&commImplName, &transposeImplName, &scaleTransposeImplName, &matmulImplName);      \
        Mc2Kernel::AlltoAllMxQuantMatmulArch35<...> op(&SchedulerImpl);                                                \
        op.Init(...);                                                                                                  \
        op.Process();                                                                                                  \
    } while (0)

// 改:
#define ALLTO_ALL_MX_TENSOR_MATMUL_IMPL(tilingData, pipe, commDataTypeX1)                                              \
    do {                                                                                                               \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_CCU, MC2AlltoAllContext,             \
                                           AlltoAllQuantMatmulTilingData, MC2AlltoAllPrimitives, 0, 1,                  \
                                           CommunicationType);                                                          \
        CommunicationType commImplName(&tilingData);                                                                   \
        using ComputationContextType = MC2KernelTemplate::MC2MMContext<                                                \
            MC2KernelTemplate::MXQuantMMAdditionalData, TensorMatmulTilingData>;                                       \
        using ComputationType = Mc2Kernel::TensorMXQuantMatmulStage<TensorMatmulTilingData, X2TRANSPOSE>;              \
        ComputationType matmulImplName(&pipe);                                                                         \
        using SchedulerContextType = Mc2Kernel::AlltoAllMXQmmPipelineContext<ComputationContextType>;                  \
        using SchedulerType = Mc2Kernel::AlltoAllMXQuantMatmulPipeLine<CommunicationType, ComputationType, SchedulerContextType>; \
        SchedulerType SchedulerImpl(&commImplName, &matmulImplName);                                                   \
        Mc2Kernel::AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllQuantMatmulTilingData, false>\
            op(&SchedulerImpl);                                                                                        \
        op.Init(x1, x2, bias, y, all2all_out, x1_scale, x2_scale, workspaceGM, &tilingData, &pipe);                    \
        op.Process();                                                                                                  \
    } while (0)
```

---

### 5.6 修改 `allto_all_mx_quant_matmul_tiling_base.cpp`

#### 5.6.1 修改 FillMxTensorTiling 函数

**位置**: `allto_all_mx_quant_matmul_tiling_base.cpp:400`

```cpp
// 原:
void AllToAllMxQuantMatmulTilingBase::FillMxTensorTiling(uint32_t blockM, TensorMatmulTilingData* out) {
    out->tileM = blockM;
    out->fullK = rankK * rankDim;  // 连续 K 轴
    out->fullN = rankN;
    // ... 其他参数
}

// 改:
void AllToAllMxQuantMatmulTilingBase::FillMxTensorTiling(uint32_t blockM, TensorMatmulTilingData* out) {
    out->tileM = blockM;
    out->fullK = rankK;  // 改为 rankK（每次处理一个 rank）
    out->fullN = rankN;
    // ... 其他参数保持不变
}
```

---

## 六、需要修改的文件汇总

| 文件路径 | 修改内容 | 重要程度 |
|---------|---------|:--------:|
| `allto_all_mx_quant_matmul_pipeline.h` | 去掉 TransposeType/ScaleTransposeType 模板参数和相关成员，修改 Process | ⭐⭐⭐ |
| `allto_all_mx_quant_matmul_arch35.h` | 去掉 transOutGM_/transX1ScaleGM1_ 成员，修改地址计算，修改 ProcessTile/ProcessTail | ⭐⭐⭐ |
| `tensor_mx_matmul_stage.h` | 添加 rank 循环，每次处理一个 rank 的数据 | ⭐⭐⭐ |
| `matmul_base.h` 或 `mc2_mx_quant_matmul.h` | 扩展 MC2MMContext 添加 rankDim/rankKStride/rankScaleStride | ⭐⭐⭐ |
| `allto_all_matmul_apt.cpp` | 修改 ALLTO_ALL_MX_TENSOR_MATMUL_IMPL 宏定义 | ⭐⭐ |
| `allto_all_mx_quant_matmul_tiling_base.cpp` | 修改 FillMxTensorTiling，fullK 改为 rankK | ⭐⭐ |
| `allto_all_mx_quant_matmul_tiling_base.h` | 可能需要添加新的 tiling 参数 | ⭐ |

---

## 七、内存布局变化

### 7.1 Workspace 内存节省

**原布局**：
```
workspaceGM:
┌────────────────────────────────────────────────────────────────────────┐
│ commX1Scale | commOut | transX1Scale | transOut（或all2all_out）│
│  (scaleLen) │(commLen)│  (scaleLen)  │    (commLen*rankDim)    │
└────────────────────────────────────────────────────────────────────────┘

总大小 ≈ scaleLen + commLen + scaleLen + commLen*rankDim
```

**新布局**：
```
workspaceGM:
┌─────────────────────────────────────────────────┐
│ commX1Scale | commOut（直接用于matmul）│
│  (scaleLen) │        (commLen)         │
└─────────────────────────────────────────────────┘

总大小 ≈ scaleLen + commLen
节省 ≈ scaleLen + commLen*rankDim
```

**示例计算**（tileM=128, rankK=1024, rankDim=4）：
```
scaleLen = 128 * (1024/32*2) * sizeof(fp8_e8m0) = 128 * 64 = 8192 bytes
commLen = 128 * 1024 * sizeof(fp8_e4m3) = 131072 bytes
节省 = 8192 + 131072 * 4 = 532480 bytes ≈ 520 KB
```

---

## 八、性能评估

### 8.1 收益分析

| 收益项 | 说明 |
|-------|------|
| **减少数据搬运** | 去掉 commOut -> transOut 的完整数据搬运 |
| **减少内存占用** | 节省 transOut 和 transX1Scale 缓存空间 |
| **缩短流水线** | 减少转置节点，降低流水线复杂度 |

### 8.2 成本分析

| 成本项 | 说明 |
|-------|------|
| **增加控制开销** | rank 循环增加少量控制逻辑开销 |
| **增加 GM 累加开销** | 每次 rank 计算需要 GM 读写（但硬件支持高效累加） |

### 8.3 预估结论

**总体收益预期为正**：
- 转置节点的数据搬运开销远大于 rank 循环的控制开销
- GM 累加已确认硬件支持，效率较高
- 内存节省对大规模场景有实际意义

---

## 九、关键注意事项

### 9.1 数据对齐

- 通信输出的 stride 需确认符合硬件对齐要求
- Scale 数据 stride 需考虑 MXFP 的 group size 对齐

### 9.2 GM 自动累加使用方法

Ascend C 提供 API 控制 GM 自动累加：

```cpp
// 开启 GM 自动累加
SetAtomicAdd<typename Intf::ydtype>();  // ydtype 为输出数据类型

// 关闭 GM 自动累加
SetAtomicNone();
```

**★ 重要：第一个 rank 块计算时 GM 是脏数据，不能累加 ★**

**正确的使用流程**：
```cpp
for (uint32_t rankIdx = 0; rankIdx < rankDim; rankIdx++) {
    // ★★★ rankIdx >= 1 时开启累加 ★★★
    // rankIdx=0 默认不累加，直接写入初始化 GM（清除脏数据）
    if (rankIdx >= 1) {
        SetAtomicAdd<DTYPE_Y>();
    }
    
    // matmul(k=rankK) -> 写入/累加到 GM
}

// ★★★ 循环结束后关闭累加 ★★★
SetAtomicNone();
```

**关键注意事项**：
- **rankIdx = 0**：默认不累加（无需显式调用 SetAtomicNone），正常写入初始化 GM
- **rankIdx >= 1**：调用 SetAtomicAdd 开启累加
- **循环结束后**：调用 SetAtomicNone 关闭累加，恢复默认状态
- 输出数据类型 `ydtype` 需与 C 矩阵类型一致

### 9.3 同步点

- rank 循环内部是否需要额外同步需根据实际测试确定
- BlockMmad 内部的流水线同步保持不变

### 9.4 测试验证

- 优化后需要完整精度验证
- 需要对比原方案的性能数据
- 需要测试不同 shape 下的稳定性

---

## 十、附录

### 10.1 相关代码位置索引

| 代码位置 | 文件 | 行号 |
|---------|------|:----:|
| Pipeline Process | `allto_all_mx_quant_matmul_pipeline.h` | 84-102 |
| Pipeline ProcessScale | `allto_all_mx_quant_matmul_pipeline.h` | 107-116 |
| Arch35 Init | `allto_all_mx_quant_matmul_arch35.h` | 66-104 |
| Arch35 ProcessScale | `allto_all_mx_quant_matmul_arch35.h` | 131-159 |
| Arch35 ProcessTile | `allto_all_mx_quant_matmul_arch35.h` | 161-225 |
| Arch35 ProcessTail | `allto_all_mx_quant_matmul_arch35.h` | 228-277 |
| BlockMmad operator() | `mx_block_mmad_swat.h` | 194-367 |
| CopyL0C2GM | `mx_block_mmad_swat.h` | 363 |
| TensorMXQuantMatmulStage Process | `tensor_mx_matmul_stage.h` | 65-144 |
| Macro ALLTO_ALL_MX_TENSOR_MATMUL_IMPL | `allto_all_matmul_apt.cpp` | 108-135 |
| FillMxTensorTiling | `allto_all_mx_quant_matmul_tiling_base.cpp` | 400 |

### 10.2 参考文档

- `tensor_api_replacement.md`: Tensor API 替换方案记录
- `mx_block_mmad_swat.h`: MX BlockMmad 流水线实现
- `matmul_constant.h`: FINAL/NON_FINAL_ACCUMULATION 常量定义

---

**文档版本**: v1.0  
**创建日期**: 2026-05-11  
**状态**: 待实施