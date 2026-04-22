# moe_gating_top_k_softmax_v2 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeGatingTopKSoftmaxV2`
**算子类**: MoE 门控 Softmax+TopK 融合（第二代）
**功能描述**: moe_gating_top_k_softmax 的增强版本。在 Softmax+TopK 融合基础上新增: (1) renorm 重归一化支持（对 TopK 后的权重再做 Softmax/L1 归一化）; (2) softmax_result 可选输出（完整 Softmax 归一化结果）; (3) 更精细的组合式 TilingKey 编码体系。提供 EKFullLoad、KFullLoad、KRenorm、Perf、PerfRegbase 五种计算策略。

**支持平台**: ascend910b (arch32), ascend910_93 (arch35), ascend950 (regbase)
**支持数据类型**: FP16/BF16/FP32（输入），INT32（索引），FLOAT（softmax_result）

---

## 2. 目录结构与文件职责

```
moe_gating_top_k_softmax_v2/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_gating_top_k_softmax_v2_def.cpp             # 算子定义
│   ├── moe_gating_top_k_softmax_v2_tiling.h             # TilingData结构
│   ├── moe_gating_top_k_softmax_v2_tiling.cpp           # Tiling策略入口
│   ├── moe_gating_top_k_softmax_v2_tiling_base.cpp      # Tiling基类公共逻辑
│   ├── moe_gating_top_k_softmax_v2_tiling_ek_fullload.cpp  # EKFullLoad策略
│   ├── moe_gating_top_k_softmax_v2_tiling_k_fullload.cpp   # KFullLoad策略
│   ├── moe_gating_top_k_softmax_v2_tiling_k_renorm.cpp     # KRenorm策略
│   ├── moe_gating_top_k_softmax_v2_tiling_perf.cpp         # Perf策略
│   ├── moe_gating_top_k_softmax_v2_regbase_tiling_perf.cpp # Regbase Perf策略
│   └── config/ {ascend910b, ascend950}
├── op_kernel/
│   ├── moe_gating_top_k_softmax_v2.cpp                 # membase Kernel入口（36分支）
│   ├── moe_gating_top_k_softmax_v2_apt.cpp             # regbase Kernel入口
│   ├── moe_gating_top_k_softmax_v2_ek_fullload.h       # EKFullLoad kernel
│   ├── moe_gating_top_k_softmax_v2_k_fullload.h        # KFullLoad kernel
│   ├── moe_gating_top_k_softmax_v2_k_renorm.h          # KRenorm kernel
│   ├── moe_gating_top_k_softmax_v2_perf.h              # Perf kernel
│   └── arch35/                                         # arch35专用Kernel
├── op_graph/
├── examples/
└── tests/ut/ {op_host, op_kernel}
```

注：本算子无独立 ACLNN API 发布接口。

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_gating_top_k_softmax_v2_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | x | REQUIRED | FP16/BF16/FP32 | 2D: [rows, col] 或 3D: [batch, seq, col] |
| 输入 | finished | OPTIONAL | BOOL | 与 x 前 N-1 维相同 |
| 输出 | y | REQUIRED | 同 x | 前 N-1 维同 x, 最后一维=k |
| 输出 | expert_idx | REQUIRED | INT32 | 同 y |
| 输出 | softmax_result | OPTIONAL | FLOAT | 同 x 完整 shape |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| k | Int | 必填 | TopK 选取数量 |
| renorm | Int | 0 | 重归一化模式: 0=不做, 1=L1 重归一化 |
| output_softmax_result_flag | Bool | false | 是否输出完整 Softmax 结果 |

**softmax_result 输出条件**: 仅当 `renorm==0` 且 `output_softmax_result_flag==true` 时输出。

### 3.2 Tiling 策略

**TilingKey 组合编码**:
```
TilingKey = 100000 + scene*1000 + renorm*100 + dtype*10 + extra

scene:  1=EKFullLoad, 2=KFullLoad/KRenorm, 3=Perf
renorm: 0=不做, 1=L1重归一化
dtype:  1=FLOAT, 2=FLOAT16, 3=BF16
extra:  EKFullLoad: 0=单缓冲, 1=双缓冲
        Perf: 1=col<8, 2=8-64, 3=>64
```

**Key 示例**:
```
101010 = EKFullLoad, no-renorm, FLOAT, 单缓冲
101131 = EKFullLoad, renorm, BF16, 双缓冲
102021 = KFullLoad, no-renorm, FLOAT16
102111 = KRenorm, renorm, FLOAT16
103112 = Perf, renorm, FLOAT, col 8-64
```

**五种 Tiling 策略**:

| 策略 | 优先级 | 条件 |
|------|--------|------|
| Perf | 1000 | col<=1024 且 col%8==0 且 k<=8，或 col<8；非 regbase |
| EKFullLoad | 10000 | col<=8160（UB 可容纳） |
| KFullLoad | 20000 | renorm==0 |
| KRenorm | 20001 | renorm==1 |
| Regbase | N/A | ascend950 regbase 架构 |

### 3.3 Shape 推导 (moe_gating_top_k_softmax_v2_infershape.cpp)

```
y              = x前N-1维 + [k]        (dtype 同 x)
expert_idx     = x前N-1维 + [k]        (dtype INT32)
softmax_result = 若 softmaxFlag: x 完整 shape (dtype FLOAT)
                否则: [0]
```

### 3.4 ACLNN API

本算子无独立 ACLNN 发布接口，通过 GE 框架直接调用。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**Membase 入口**（moe_gating_top_k_softmax_v2.cpp）: 36 个 TilingKey 分支。

```
moe_gating_top_k_softmax_v2() 入口
  +-- 101xxx: EKFullLoad (6种: 3 dtype × 2 buffer, renorm=0)
  +-- 1011xx: EKFullLoad (6种: 3 dtype × 2 buffer, renorm=1)
  +-- 1020xx: KFullLoad  (3种: 3 dtype, renorm=0)
  +-- 1021xx: KRenorm    (3种: 3 dtype, renorm=1)
  +-- 103xxx: Perf       (18种: 3 dtype × 3 colRange × 2 renorm)
```

### 4.2 核心计算流程

**数学公式**:
```
softmax(x[i]) = exp(x[i] - max) / sum(exp(x[j] - max))
topK_indices = TopK(softmax(x), k)
y = softmax(topK_indices)
若 renorm=1: y = Softmax(y_topK)   (对K个值再做Softmax重归一化)
若 finished[i]=true: expert_idx[i] = col
```

**EKFullLoad renorm 路径**:
```
Softmax(整行, FlashV2) → TopK → 提取K个值
  → 再对K个值做 Softmax(renorm): shape={ubFormer, kAlignB32/4}
```

**KRenorm 路径**:
```
SoftmaxFlashV2(分段) → TopK → 提取K个值
  → 再对K个值做 Softmax(renorm): shape={1, kAlign}
```

**softmax_result 输出**: 当 softmaxFlag=1 时，Kernel 额外将完整 Softmax 结果写出。

### 4.3 数据结构与内存布局

**EKFullLoad 双缓冲模式 UB**:
```
┌──────────────────────────────────────────┐
│ 缓冲区0 (前半UB)                          │
│   gatingQueue0  (colAlign * FP32)        │
│   softmaxMax0   (ubFormer * FP32)        │
│   softmaxSum0   (ubFormer * FP32)        │
│   topkOut0      (kAlign * FP32)          │
│   topkIndices0  (kAlign * INT32)         │
├──────────────────────────────────────────┤
│ 缓冲区1 (后半UB)                          │
│   gatingQueue1 / softmaxMax1 / ...       │
├──────────────────────────────────────────┤
│ [renorm=1 额外空间]                       │
│   softmaxTiling / topkTiling             │
└──────────────────────────────────────────┘
```

**Perf 模式 UB**:
```
┌──────────────────────────────────────────┐
│ xInQueue    (colBytesAlign * FP32)       │  <- 双缓冲
│ xNormBuf    (colAlign * FP32)            │  <- Softmax结果
│ indexBuf    (colAlign * INT32)           │  <- 排序索引
│ sortedBuf   (colAlign * INT64)           │  <- 排序结果
│ topkOutBuf  (kAlign * FP32)              │  <- TopK值
│ topkIdxBuf  (kAlign * INT32)             │  <- TopK索引
│ calcTmpBuf  (colAlign * FP32 * N)        │  <- Reduce临时空间
└──────────────────────────────────────────┘
```

### 4.4 多核并行策略

按行均匀分核: `blockFormer = CeilDiv(rows, blockNum)`。各核独立处理自己的行段。

### 4.5 通信模式

无 SyncAll。各核独立按行处理。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节。复用 V1 的三种 TilingData 结构（EKFullLoad/KFullLoad/Perf），KRenorm 与 KFullLoad 共享结构但额外计算 renorm Softmax Tiling。CompileInfo 含 `totalCoreNum`, `ubSize`。

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
         y = softmax[topK_indices]
                   │
         ┌─────────┴──────────┐
         │ renorm=0:           │ renorm=1:
         │ 直接输出            │ 再对K个值做Softmax重归一化
         │                    │ y = exp(y) / sum(exp(y))
         └─────────┬──────────┘
                   │
         若 finished[i]: expert_idx[i] = col
                   │
           ┌───────┴───────┐
           │               │
       y [rows, k]    expert_idx [rows, k]
                  softmax_result [rows, col] (可选, FLOAT)
```

---

## 7. 性能相关设计

1. **五种策略递减选择**: Perf（手写极致优化）→ EKFullLoad（高阶 API 全量加载）→ KFullLoad/KRenorm（分段加载），按 UB 容量递减自动选择。

2. **renorm 重归一化**: 对 TopK 选出的 K 个值再做一次 Softmax，确保输出权重之和严格为 1，提高下游计算精度。

3. **softmax_result 可选输出**: 支持 Softmax 完整归一化结果输出，避免下游算子重复计算 Softmax。

4. **组合式 TilingKey 编码**: 通过公式 `100000 + scene*1000 + renorm*100 + dtype*10 + extra` 自动组合，新增场景只需调整编码公式，易扩展。

5. **Perf 预加载流水线**: CopyIn(0) 预加载，后续 Compute(i) 与 CopyIn(i+1) 重叠执行，隐藏内存延迟。

6. **EKFullLoad 双缓冲**: UB 分为两半交替使用，提高吞吐量。

---

## 8. 依赖关系

```
op_host/moe_gating_top_k_softmax_v2_def.cpp
└── register/op_def_registry.h

op_host/moe_gating_top_k_softmax_v2_tiling.cpp
├── moe_gating_top_k_softmax_v2_tiling.h
│   ├── register/op_impl_registry.h
│   └── tiling_base/tiling_base.h
├── moe_gating_top_k_softmax_v2_tiling_ek_fullload.cpp
├── moe_gating_top_k_softmax_v2_tiling_k_fullload.cpp
├── moe_gating_top_k_softmax_v2_tiling_k_renorm.cpp
├── moe_gating_top_k_softmax_v2_tiling_perf.cpp
├── moe_gating_top_k_softmax_v2_regbase_tiling_perf.cpp
└── platform_ascendc::PlatformAscendC

op_kernel/moe_gating_top_k_softmax_v2.cpp (membase)
├── moe_gating_top_k_softmax_v2_ek_fullload.h
├── moe_gating_top_k_softmax_v2_k_fullload.h
├── moe_gating_top_k_softmax_v2_k_renorm.h
├── moe_gating_top_k_softmax_v2_perf.h
└── kernel_operator.h

op_kernel/moe_gating_top_k_softmax_v2_apt.cpp (regbase)
├── arch35/
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| renorm | 0=不做重归一化, 1=L1 重归一化（Softmax） |
| output_softmax_result_flag | 仅 renorm==0 时生效 |
| k | 必须 > 0 且 <= 1024 |
| k <= col | TopK 数量不超过专家总数 |
| 输入维度 | 2D 或 3D |
| EKFullLoad 条件 | col <= 8160（UB 可容纳） |
| Perf 条件 | (col<=1024 且 col%8==0 且 k<=8) 或 col<8；非 regbase |
| softmax_result dtype | 固定 FLOAT |
| finished | OPTIONAL, BOOL, 标记 EOS token |

---

## 10. 与同族变体的核心差异

| 维度 | moe_gating_top_k_softmax（V1） | moe_gating_top_k_softmax_v2（本算子） |
|------|-------------------------------|--------------------------------------|
| renorm | 不支持 | 支持（0/1 两种模式） |
| softmax 结果输出 | 不支持 | 可选输出（softmax_result） |
| TilingKey 编码 | 线性枚举 0-17 | 组合编码 101010-103133 |
| Tiling 策略数 | 3 种（EK/K/Perf） | 5 种（EK/K/KRenorm/Perf/PerfRegbase） |
| Kernel 模板数 | 18 | 36+ |
| KFullLoad renorm | 不区分 | 单独 KRenorm 策略处理 |
| softmax 实现 | FlashV2 或手写 Perf | FlashV2 或手写 Perf（继承） |
| renorm 算法 | 无 | Softmax 重归一化（非 L1） |
