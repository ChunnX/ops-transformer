# CausalConv1d Tiling 分析

## 1. 总体框架

```
DoTiling()
  │
  ├─ ParseCausalConv1dInputInfo()     // 解析所有输入 → inputInfo_
  │
  ├─ if (run_mode == 0, forward):
  │     ChooseFnHostPlan()
  │       ├─ dim ≤ 4096 → TOKEN_FIRST + CUTBS
  │       └─ dim > 4096 → TOKEN_DIM_CO_SPLIT + CUTBSD
  │
  └─ if (run_mode == 1, update):
        ChooseCanonicalUpdateBaseDimChoice()
          └─ 候选 baseDim ∈ {4096, 2048, 1024, 512, 384, 192}
```

Tiling 结果写入 `CausalConv1dTilingData`：`baseDim`, `baseDimCnt`, `tokenBlockSize`, `tokenBlockCnt`, `blockDim` 等。

`blockDim` 通过 `context->SetBlockDim()` 传给框架，决定实际调度的 core 数量。

---

## 2. 三种切分场景

| 场景 | 触发条件 | 切分轴 | baseDim 选择 |
|------|----------|--------|-------------|
| **Update** | `run_mode=1` | `batch × baseDimCnt` 二维网格 | 从 {4096,2048,1024,512,384,192} 选，优先整除 dim 且 grid ≥ coreNum |
| **Fn + TOKEN_FIRST** | `run_mode=0 && dim ≤ 4096` | token 块 × 1 (dim 不分片) | `baseDim = dim`，`baseDimCnt = 1` |
| **Fn + TOKEN_DIM_CO_SPLIT** | `run_mode=0 && dim > 4096` | token 块 × baseDimCnt (dim 分片) | UB 容量约束 + coreNum 整除约束 |

---

## 3. Update 模式 Tiling

### 3.1 网格模型
```
grid[i, j] = core 负责 seq[i] 的 dim[j·baseDim : (j+1)·baseDim] 通道块
grid_size = batch × baseDimCnt
```

### 3.2 baseDim 选择算法 (`ChooseCanonicalUpdateBaseDimChoice`)
```
1. 第一轮 (requireExactDiv=true): 遍历候选，只保留整除 dim 的
   → 选 grid ≥ coreNum 且 gap 最小的 (bestOver)
   → 若无≥coreNum，选 grid 最大的 under-utilized (bestUnder)

2. 第二轮 (requireExactDiv=false): 如果第一轮全是 under，放宽整除约束再搜一轮
   → baseDimCnt = CeilDiv(dim, baseDim)，最后一个 tile 不满也允许
```

### 3.3 调度
```cpp
// 每个 core 通过 blockIdx + k×blockNum 遍历整个 batch × baseDimCnt 网格
for task in [blockIdx, blockIdx+blockNum, ...]:
    seq       = task / baseDimCnt
    channelStart = (task % baseDimCnt) × baseDim
```

---

## 4. Forward 模式 Tiling

### 4.1 两级切分
```
第一级 TokenTiling: cuSeqlen 个 token 按 tokenCoreBudget 均分
                    → tokenBlockSize, tokenBlockCnt

第二级 DimTiling:   每个 token tile 再按 baseDimCnt 切 dim 通道
                    → 总网格 = tokenBlockCnt × baseDimCnt
```

### 4.2 CUTBS vs CUTBSD
| Plan | 含义 | tokenCoreBudget | 适用 |
|------|------|-----------------|------|
| CUTBS | Cut Batch+Seq | coreNum (全部 core 分 token) | dim ≤ 4096, baseDimCnt=1 |
| CUTBSD | Cut Batch+Seq+Dim | coreNum / baseDimCnt (core 先分给 dim) | dim > 4096 |

### 4.3 TOKEN_FIRST 路径 (dim ≤ 4096)
```
baseDim = dim (整个 dim 不分片)
baseDimCnt = 1
executionPlan = CUTBS
tokenCoreBudget = coreNum (全部 core 用来分 token)
gridSize = tokenBlockCnt × 1 = tokenBlockCnt
```

### 4.4 TOKEN_DIM_CO_SPLIT 路径 (dim > 4096)
```
// Step 1: UB 约束计算 baseDim 上限
ubLimitedBaseDim = (ubSize - 512) / (5×2 + 2×2 + 8×4)
                 = (ubSize - 512) / 46
                 // 5个 ring slot(fp16) + 2个 out slot(fp16) + 8个 calc slot(fp32)
向下对齐到 16，上限 4096

// Step 2: 调整 baseDimCnt 使其整除 coreNum (负载均衡)
while (baseDimCnt < coreNum && coreNum % baseDimCnt != 0):
    baseDimCnt++
adjustedBaseDim = CeilAlign(CeilDiv(dim, adjustedBaseDimCnt), 16)
// 若调整后 baseDim 超出 UB 限制或 4096，回退

executionPlan = CUTBSD
tokenCoreBudget = coreNum / baseDimCnt
```

---

## 5. Token Tiling 详解 (Fn 模式)

### 5.1 Token 块划分
```
tokenCoreBudget = CUTBS ? coreNum : coreNum / baseDimCnt
idealBlockSize = ceil(cuSeqlen / tokenCoreBudget)
tokenBlockCnt  = ceil(cuSeqlen / idealBlockSize)

gridSize = tokenBlockCnt × baseDimCnt
```

### 5.2 Core 映射 (`BuildFnTokenCoreMappingChoice`)
```
tokenBlocksPerCore = ceil(tokenBlockCnt / tokenCoreBudget)
tokenCoreTailCnt   = tokenBlockCnt - (tokenBlocksPerCore-1) × tokenCoreBudget
blockDim           = tokenCoreBudget × baseDimCnt   // SetBlockDim 的值
```

### 5.3 变长序列 Token→Seq 查表 (`BuildFnTokenSeqRangePlan`)
```
为每个 token tile 预计算 [startSeq, endSeq] 范围，kernel 端直接查表
容量上限: MAX_FN_TOKEN_SEQ_RANGE_COUNT = 128
超过 128 时退化为 kernel 端二分搜索
```

---

## 6. Kernel 侧 UB 内存布局

### 6.1 三个区域
```
┌─────────────────────────────────────────────────────┐
│ calcBuf (VECCALC): (MAX_WIDTH + 4) × baseDim × 4B  │
│   [0:MAX_WIDTH)    → weightF  卷积权重 (W<4 高位补零) │
│   [MAX_WIDTH]      → biasF    偏置 (无则填零)        │
│   [MAX_WIDTH+1]    → accF     累加器                 │
│   [MAX_WIDTH+2]    → tmpF     临时/激活输出           │
│   [MAX_WIDTH+3]    → currF    当前输入 token          │
│   共 8 × 4096 × 4B = 128KB (baseDim=4096 时)        │
├─────────────────────────────────────────────────────┤
│ inBuf (VECIN): 5 × baseDim × 2B                     │
│   Ring Buffer 5 个 slot 循环复用                      │
│   slotCurr(t) = (t+3) % 5                           │
│   布局: [padding][hist0][hist1][hist2][curr]         │
│   padding: 前 MAX_WIDTH-W 个填零，后 W 个为有效窗口    │
│   5 × 4096 × 2B = 40KB (baseDim=4096 时)            │
├─────────────────────────────────────────────────────┤
│ outBuf (VECOUT): 2 × baseDim × 2B                   │
│   Ping-Pong 双缓冲输出                                │
│   2 × 4096 × 2B = 16KB (baseDim=4096 时)            │
└─────────────────────────────────────────────────────┘

总计: 128KB + 40KB + 16KB = 184KB + 512B reserve ≈ 185KB
```

### 6.2 UB 容量约束
```
bytesPerElem = 5×2 + 2×2 + 8×4 = 46 bytes/elem
ubLimitedBaseDim = (ubSize - 512) / 46  // 向下对齐到 16
```

不同 ubSize 下的 baseDim 上限：

| ubSize | ubLimitedBaseDim |
|--------|-----------------|
| 256KB | ~5584 → 上限4096 |
| 224KB | ~4880 → 上限4096 |
| 192KB | ~4096 |

---

## 7. 场景→内存 完整映射表

| 场景 | runMode | dim | baseDim | baseDimCnt | tokenCoreBudget | blockDim | 每core处理量 |
|------|---------|-----|---------|------------|-----------------|----------|------------|
| Update, dim=8192 | 1 | 8192 | 4096 | 2 | - | ~batch×2 | 1 seq × 4096 ch |
| Update, dim=2048 | 1 | 2048 | 2048 | 1 | - | ~batch | 1 seq × 2048 ch |
| Fn TOKEN_FIRST, dim=1024 | 0 | 1024 | 1024 | 1 | 40(coreNum) | 40 | all seq in tile × 1024 ch |
| Fn TOKEN_DIM, dim=8192 | 0 | 8192 | ~4096 | ~2 | ~20(coreNum/2) | 40 | partial token × 4096 ch |

---

## 8. 关键数据结构 (`CausalConv1dTilingData`)

```cpp
struct CausalConv1dTilingData {
    // 基础 shape
    int64_t dim, cuSeqlen, seqLen, batch;
    int64_t kernelWidth, stateLen, numCacheLines;
    int64_t activationMode, padSlotId;

    // dim 切分
    int64_t baseDim, baseDimCnt;

    // token 切分 (仅 fn 模式有效)
    int64_t tokenBlockSize, tokenBlockCnt;

    // 可选输入标志
    int64_t hasBias, hasCacheIndices, hasInitialStateMode;
    int64_t hasNumAcceptedTokens, hasInitStateWorkspace;

    // 变长序列 token→seq 查表
    int64_t hasExplicitTokenSeqRanges;
    int64_t explicitTokenSeqRangeCount;
    int64_t tokenTileStartSeq[128];
    int64_t tokenTileEndSeq[128];
};
```

---

## 9. 模板实例化组合

通过 `(runModeKey, widthKey, fnPlanKey)` 三键组合编译出 7 个 kernel 变体：

| # | runModeKey | widthKey | fnPlanKey | 场景 |
|---|-----------|----------|-----------|------|
| 0 | UPDATE | RUNTIME | INVALID | update, W 运行时确定 |
| 1 | FN | W=2 | CUTBS | fn, W=2, dim≤4096 |
| 2 | FN | W=3 | CUTBS | fn, W=3, dim≤4096 |
| 3 | FN | W=4 | CUTBS | fn, W=4, dim≤4096 |
| 4 | FN | W=2 | CUTBSD | fn, W=2, dim>4096 |
| 5 | FN | W=3 | CUTBSD | fn, W=3, dim>4096 |
| 6 | FN | W=4 | CUTBSD | fn, W=4, dim>4096 |
