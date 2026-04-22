# moe_gating_top_k_softmax 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeGatingTopKSoftmax`
**算子类**: MoE 门控 Softmax+TopK 融合
**功能描述**: MoE 门控 Softmax+TopK 融合算子。对输入 gating 分数先做 Softmax 归一化，再做 TopK 选择，输出权重、专家索引和源行索引。支持 `finished` 标记（推理阶段 EOS token 处理）。提供 EKFullLoad、KFullLoad、Perf 三种计算策略。

**支持平台**: ascend910b (arch32), ascend910_93 (arch35), ascend950 (regbase)
**支持数据类型**: FP16/BF16/FP32（输入），INT32（索引）

---

## 2. 目录结构与文件职责

```
moe_gating_top_k_softmax/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_gating_top_k_softmax_def.cpp             # 算子定义
│   ├── moe_gating_top_k_softmax_tiling.h             # TilingData结构（3种）
│   ├── moe_gating_top_k_softmax_tiling.cpp           # Tiling策略入口
│   ├── moe_gating_top_k_softmax_tiling_base.h        # Tiling基类
│   ├── moe_gating_top_k_softmax_tiling_base.cpp      # Tiling基类公共逻辑
│   ├── moe_gating_top_k_softmax_tiling_arch35.cpp    # arch35 Tiling策略
│   ├── moe_gating_top_k_softmax_tiling_e_k_full_load.cpp  # EKFullLoad策略
│   ├── moe_gating_top_k_softmax_tiling_k_full_load.cpp    # KFullLoad策略
│   ├── moe_gating_top_k_softmax_tiling_perf.cpp           # Perf高性能策略
│   └── config/ {ascend910b, ascend910_93, ascend950}
├── op_kernel/
│   ├── moe_gating_top_k_softmax.cpp                  # membase Kernel入口（18分支）
│   ├── moe_gating_top_k_softmax_apt.cpp              # regbase Kernel入口
│   ├── moe_gating_top_k_softmax_e_k_fullload.h       # EKFullLoad kernel
│   ├── moe_gating_top_k_softmax_k_fullload.h         # KFullLoad kernel
│   ├── moe_gating_top_k_softmax_perf.h               # Perf高性能kernel
│   └── arch35/                                       # arch35专用Kernel
├── op_graph/
├── framework/
├── examples/
└── tests/ut/ {op_host, op_kernel}
```

注：本算子无独立 ACLNN API 发布接口。

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_gating_top_k_softmax_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | x | REQUIRED | FP16/BF16/FP32 | 2D: [rows, col] 或 3D: [batch, seq, col] |
| 输入 | finished | OPTIONAL | BOOL | 与 x 前 N-1 维相同 |
| 输出 | y | REQUIRED | 同 x | 前 N-1 维同 x, 最后一维=k |
| 输出 | expert_idx | REQUIRED | INT32 | 同 y |
| 输出 | row_idx | REQUIRED | INT32 | 同 y |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| k | Int | 必填 | TopK 选取数量 |

### 3.2 Tiling 策略

**三种 TilingData 结构**:

1. **MoeGatingTopKSoftmaxEKFullLoadTilingData**（EKFullLoad）: 含 `row/col/colAlign/k/kAlignB16/kAlignB32`、核分配 `blockNum/blockFormer/blockTail`、批次循环 `ubLoopOfFormerBlock/ubLoopOfTailBlock`、每批行数 `ubFormer/ubTailOfFormerBlock/ubTailOfTailBlock`、3 组 SoftmaxTiling + 3 组 TopkTiling。

2. **MoeGatingTopKSoftmaxPerfTilingData**（Perf）: 含基础维度 + `topKValuesMask/topKIndicesMask`（GatherMask 位模式）+ `bufferElemSize`。

3. KFullLoad 复用 EKFullLoad TilingData 结构。

**TilingKey 编码**（18 个 Key，范围 0-17）:

| Key | 策略 | 数据类型 | 备注 |
|-----|------|----------|------|
| 0/1 | EKFullLoad | FP32 | 单缓冲/双缓冲 |
| 2/3 | EKFullLoad | FP16 | 单缓冲/双缓冲 |
| 4/5 | EKFullLoad | BF16 | 单缓冲/双缓冲 |
| 6/7/8 | KFullLoad | FP32/FP16/BF16 | - |
| 9/10/11 | Perf | FP32/FP16/BF16 | col<8 |
| 12/13/14 | Perf | FP32/FP16/BF16 | 8<=col<=64 |
| 15/16/17 | Perf | FP32/FP16/BF16 | col>64 |

**Tiling 注册优先级**: Perf=1000（高），EKFullLoad=10000，KFullLoad=20000（低）。

**策略选择**: EKFullLoad 在 col<=8160 时使用；KFullLoad 在 EKFullLoad 失败时使用；Perf 在 col<=1024 且 col%8==0 且 k<=8 或 col<8 时使用。

### 3.3 Shape 推导 (moe_gating_top_k_softmax_infershape.cpp)

```
y          = x前N-1维 + [k]     (dtype 同 x)
expert_idx = x前N-1维 + [k]     (dtype INT32)
row_idx    = x前N-1维 + [k]     (dtype INT32)
```

校验: k > 0 且 <= 1024，k <= col。

### 3.4 ACLNN API

本算子无独立 ACLNN 发布接口，通过 GE 框架直接调用。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**Membase 入口**（moe_gating_top_k_softmax.cpp）: 18 个 TilingKey 分支。

```
moe_gating_top_k_softmax() 入口
  +-- Key 0-5:  EKFullLoad (3 dtype × 2 buffer模式)
  +-- Key 6-8:  KFullLoad  (3 dtype)
  +-- Key 9-17: Perf       (3 dtype × 3 colRange)
```

### 4.2 核心计算流程

**数学公式**:
```
softmax(x[i]) = exp(x[i] - max) / sum(exp(x[j] - max))
topK_indices = TopK(softmax(x), k)
y = softmax(topK_indices)
若 finished[i]=true: expert_idx[i] = col (超出范围标记)
```

**EKFullLoad 路径**:
```
对每批 ubFormer 行:
  CopyIn → SoftmaxFlashV2(高阶API) → TopK(高阶API) → CopyOut
```

**KFullLoad 路径**:
```
对每行:
  逐段 CopyIn (按 ubFormer 切分 col 维度)
  首段: SoftmaxFlashV2(init)
  后段: SoftmaxFlashV2(update) + Muls(expMax)
  ComputeOut: 1/softmaxSum 归一化 + finished 处理
  CopyOut + ComputeRowOut + CopyRowOut
```

**Perf 路径**:
```
Cast(FP32) → SoftmaxFP32Perf(手写 ReduceMax+Exp+ReduceSum+Div)
→ Rearrange + InitIndex
→ Sort32 + MrgSort(手写多路归并)
→ GatherMask(提取前K个) → UpdateIndices(finished处理)
→ Cast回原类型 → CopyOut
```

**SoftmaxFP32Perf 细分**:
- col<8: BlockReduceMax/BlockReduceSum
- 8<=col<=64: WholeReduce
- col>64: 分段 Reduce + BlockReduce

**MergeSortFP32Perf 策略**:
- col<=512: Sort32(32->8) → MrgSort4(8->1)
- col>512: 两段各 Sort32+MrgSort4，最后 MrgSort2 合并

### 4.3 数据结构与内存布局

**EKFullLoad 双缓冲模式 UB**:
```
┌──────────────────────────────────────┐
│ 缓冲区0 (前半UB)                      │
│   gatingQueue0  (colAlign * FP32)    │
│   softmaxMax0   (ubFormer * FP32)    │
│   softmaxSum0   (ubFormer * FP32)    │
│   topkOut0      (kAlign * FP32)      │
│   topkIndices0  (kAlign * INT32)     │
├──────────────────────────────────────┤
│ 缓冲区1 (后半UB)                      │
│   gatingQueue1 / softmaxMax1 / ...   │
└──────────────────────────────────────┘
```

**Perf 模式 UB**:
```
┌──────────────────────────────────────┐
│ xInQueue    (colBytesAlign * FP32)   │  <- 双缓冲
│ xNormBuf    (colAlign * FP32)        │  <- Softmax结果
│ indexBuf    (colAlign * INT32)       │  <- 排序索引
│ sortedBuf   (colAlign * INT64)       │  <- 排序结果
│ topkOutBuf  (kAlign * FP32)          │  <- TopK值
│ topkIdxBuf  (kAlign * INT32)         │  <- TopK索引
│ calcTmpBuf  (colAlign * FP32 * N)    │  <- Reduce临时空间
└──────────────────────────────────────┘
```

### 4.4 多核并行策略

按行均匀分核: `blockFormer = CeilDiv(rows, blockNum)`。各核独立处理自己的行段。

### 4.5 通信模式

无 SyncAll。各核独立按行处理。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节三种 TilingData 结构。CompileInfo 含 `totalCoreNum`, `ubSize`。

### 5.2 Workspace 布局

```
┌──────────────────────────────────────────────────────────────┐
│ System Workspace = 16 MB (固定大小)                            │
└──────────────────────────────────────────────────────────────┘
```

无额外算子级 workspace。

---

## 6. 完整数据流

```
  x [rows, col] (2D/3D)   finished [rows](可选)
       │                       │
       └───────────────────────┘
                   │
         对每行 i (核间并行):
           softmax = Softmax(x[i])   (FlashV2 或手写Perf)
                   │
         topK_indices = TopK(softmax, k)
                   │
         y[i] = softmax[topK_indices]
         若 finished[i]: expert_idx[i] = col
         若 softmaxSum为NaN: expert_idx[i] = 0..k-1
                   │
           ┌───────┴───────┐
           │               │
       y [rows, k]    expert_idx [rows, k]
                    row_idx [rows, k]
```

---

## 7. 性能相关设计

1. **三级策略自动选择**: EKFullLoad（小 col 全量加载，高阶 API）、KFullLoad（大 col 分段累加）、Perf（手写极致优化）。

2. **Perf 手写优化**: 按 col 范围细分 ReduceMax/ReduceSum 策略，Sort32+MrgSort 多路归并替代 TopK 高阶 API，减少 API 调用开销。

3. **预加载流水线（Perf）**: CopyIn(0) 预加载第 0 批，后续每批 Compute(i) 与 CopyIn(i+1) 重叠执行，隐藏内存访问延迟。

4. **SoftmaxFlashV2 分段累加**: KFullLoad 路径将 col 维度切分为多段，首段 init 后续段 update，自动处理数值稳定性。

5. **双缓冲（EKFullLoad）**: UB 空间分为两半交替使用，奇偶 TilingKey 区分单缓冲/双缓冲。

6. **GatherMask 提取**: Perf 路径使用预计算的位模式掩码（`topKValuesMask/topKIndicesMask`）替代逐元素提取。

---

## 8. 依赖关系

```
op_host/moe_gating_top_k_softmax_def.cpp
└── register/op_def_registry.h

op_host/moe_gating_top_k_softmax_tiling.cpp
├── moe_gating_top_k_softmax_tiling.h
│   ├── register/op_impl_registry.h
│   └── tiling_base/tiling_base.h
├── moe_gating_top_k_softmax_tiling_base.h
├── moe_gating_top_k_softmax_tiling_e_k_full_load.cpp
├── moe_gating_top_k_softmax_tiling_k_full_load.cpp
├── moe_gating_top_k_softmax_tiling_perf.cpp
└── platform_ascendc::PlatformAscendC

op_kernel/moe_gating_top_k_softmax.cpp (membase)
├── moe_gating_top_k_softmax_e_k_fullload.h
├── moe_gating_top_k_softmax_k_fullload.h
├── moe_gating_top_k_softmax_perf.h
└── kernel_operator.h

op_kernel/moe_gating_top_k_softmax_apt.cpp (regbase)
├── arch35/
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| k | 必须 > 0 且 <= 1024 |
| k <= col | TopK 数量不超过专家总数 |
| 输入维度 | 2D 或 3D（最后一维为 col） |
| finished | OPTIONAL, BOOL, 标记 EOS token |
| EKFullLoad 条件 | col <= 8160（UB 可容纳） |
| Perf 条件 | (col<=1024 且 col%8==0 且 k<=8) 或 col<8 |
| NaN 检测 | softmaxSum 为 NaN 时 expert_idx 设为 0..k-1 |
| row_idx | 源行映射，支持多 token 场景 |

---

## 10. 与同族变体的核心差异

| 维度 | moe_gating_top_k | moe_gating_top_k_softmax（本算子） | moe_gating_top_k_softmax_v2 |
|------|-------------------|-----------------------------------|------------------------------|
| 归一化策略 | Sigmoid/Softmax 可选 | 仅 Softmax | 仅 Softmax |
| 分组机制 | 有 group 分组 | 无，全局 TopK | 无，全局 TopK |
| finished 支持 | 无 | 有（EOS 标记） | 有（EOS 标记） |
| renorm | 有（L1） | 无 | 有（Softmax 重归一化） |
| softmax 输出 | 有 out | 无 | 有 softmax_result（可选） |
| row_idx 输出 | 无 | 有 | 无 |
| Softmax 实现 | 手写 | FlashV2 + Perf 手写 | FlashV2 + Perf 手写 |
| TopK 实现 | Sort32+MrgSort | 高阶 API + Perf 手写 | 高阶 API + Perf 手写 |
| TilingKey 范围 | 0-2（3个） | 0-17（18个） | 101010-103133（36+个） |
| 典型应用 | 训练 | 推理（基础版） | 推理（增强版） |
