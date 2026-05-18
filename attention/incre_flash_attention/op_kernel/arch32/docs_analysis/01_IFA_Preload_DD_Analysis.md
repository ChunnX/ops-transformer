# IFA Preload DD 算子深度分析

> **对应PDF**: 01_IFA_Preload_DD_Analysis.pdf
> **Token**: KNL6bwi7RoGWhtxj16jcJ5Bgn5Y

---

## 一、算子概述

`IncreFlashAttentionPreloadDD` 是昇腾 Flash Attention 项目中针对 Decode 场景优化的增量 Flash Attention 算子，采用 Preload DD（双缓冲预加载）技术实现高性能推理。

---

## 二、核心架构：Vector + Cube 异构流水线

### 2.1 为什么需要异构架构

**代码位置**: `incre_flash_attention_preload_dd.h:3779-3954`

```cpp
if ASCEND_IS_AIV {
    // Vector引擎: QueryPreProcess -> ProcessVec1L -> ProcessVec2L
    QueryPreProcessL(extraInfo[loop % EXTRA_NUM]);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V0_C1_FLAG);
}
if ASCEND_IS_AIC {
    // Cube引擎: ComputeMm1 -> ComputeMm2
    CrossCoreWaitFlag(SYNC_V0_C1_FLAG);
    ComputeMm1(extraInfo[loop % EXTRA_NUM]);
}
```

| 问题类型 | 纯Cube | 纯Vector | Vector+Cube |
|----------|--------|----------|-------------|
| 矩阵乘(Q×K^T, P×V) | 效率高 | 效率极低(~1/16) | Cube处理，高效 |
| Softmax逐元素操作 | 效率极低 | 效率高 | Vector处理，高效 |
| 流水并行 | 无 | 无 | 可实现 |

### 2.2 910B/C硬件参数

| 参数 | 数值 |
|------|------|
| AICore数量 | 24 |
| Cube MAC阵列 | 16×16 (256 MAC/cycle) |
| Vector MAC吞吐 | ~16 MAC/cycle |
| L1容量 | 1MB |
| UB容量 | 128KB×2 |

---

## 三、预加载机制：PRE_LOAD_NUM_DD vs TASK_NUM

### 3.1 【重要修正】参数定义

**原始文档错误**: 混淆了两个参数

```cpp
// 代码位置: incre_flash_attention_preload_dd.h:35-39
#define PRE_LOAD_NUM_DD 4    // 预加载Buffer槽数量
#define EXTRA_NUM 6          // ExtraInfo数组大小
#define TASK_NUM 2           // 任务并行深度
```

| 参数 | 值 | 含义 | 用途 |
|------|-----|------|------|
| PRE_LOAD_NUM_DD | 4 | Workspace预加载槽数 | 中间结果轮转存储深度 |
| TASK_NUM | 2 | 任务并行深度 | 同时派发的任务数 |
| EXTRA_NUM | 6 | 状态数组大小 | 保存中间计算状态 |

### 3.2 为什么这些数值是最优

- TASK_NUM=2: 更深则workspace膨胀，更浅则流水空泡
- PRE_LOAD_NUM_DD=4: 与L1/L2带宽匹配，深度预加载

---

## 四、存储层次规划

### 4.1 L1 Buffer分配

```cpp
// 代码位置: incre_flash_attention_preload_dd.h:262-270
L1Q_BLOCK_SIZE = 64KB   // Q矩阵: 1槽
L1KP_BLOCK_SIZE = 64KB  // K/P矩阵: 3槽 = 192KB
L1V_BLOCK_SIZE = 64KB   // V矩阵: 4槽 = 256KB
// 总计: 512KB < 1MB
```

### 4.2 UB Buffer分配

```cpp
inputBuf1: 32KB
inputBuf2: 16KB
outputBuf1: 16KB
outputBuf2: 8KB
softmaxMaxBuff: 512B × 4
softmaxSumBuff: 512B × 4
```

---

## 五、CrossCore Flag同步机制

### 5.1 四级流水握手链

```cpp
SYNC_V0_C1_FLAG = 6  // Vector完成Q预处理 -> Cube开始MM1
SYNC_C1_V1_FLAG = 7  // Cube完成MM1 -> Vector开始Softmax
SYNC_V1_C2_FLAG = 8  // Vector完成Softmax -> Cube开始MM2
SYNC_C2_V2_FLAG = 9  // Cube完成MM2 -> Vector开始输出
```

### 5.2 SYNC_MODE2双重语义

1. **pipe_flush**: 保证L1/L0/UB写回完成
2. **跨Core通知**: 设置Flag通知对方引擎

---

## 六、Tiling策略

### 6.1 sInnerSize分档

| 分档 | 值 | 场景 |
|------|-----|------|
| 大块 | 8192 | 非GQA+非伪量化 |
| 中块 | 2048 | GQA+伪量化+CUBE_VIEW_MM |
| 小块 | 1024 | GQA+伪量化+DD模式 |

### 6.2 Workspace计算【修正】

**原始文档错误**: 使用dbWorkspaceRatio=6

```cpp
// 实际代码: incre_flash_attention_preload_dd.h:328
const uint32_t dbWorkspaceRatio = PRE_LOAD_NUM_DD;  // = 4
```

---

## 七、关键代码位置索引

| 功能 | 函数 | 行号 |
|------|------|------|
| 宏定义 | 头文件 | 35-54 |
| Buffer初始化 | InitBuffers | 640-692 |
| 主流程 | Process | 3779-3954 |
| BMM1计算 | ComputeMm1 | 3065-3299 |
| BMM2计算 | ComputeMm2 | 3546-3776 |
| Softmax | SoftmaxFlashV2Compute | 2009-2045 |

---

*本文档基于代码实际实现，修正了原始分析中的错误。*