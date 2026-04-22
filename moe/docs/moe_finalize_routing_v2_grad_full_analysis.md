# moe_finalize_routing_v2_grad 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeFinalizeRoutingV2Grad`
**算子类**: MoE 路由最终化反向算子
**功能描述**: `MoeFinalizeRoutingV2` 的反向算子。给定上游梯度 `grad_y`，计算对 `expanded_x` 和 `scales` 的梯度。支持 `drop_pad_mode` 和可选 `bias`。

**前向公式回顾**:
```
y[i] = x1[i] + x2[i] + sum_j(scales[i][j] * (expanded_x[row_idx[i][j]] + bias[expert_idx[i][j]]))
```

**反向传播公式**:
```
grad_expanded_x[row_idx[i][j]] += scales[i][j] * grad_y[i]   (scatter-add)
grad_scales[i][j] = sum(grad_y[i] * (expanded_x[row_idx[i][j]] + bias[expert_idx[i][j]]))
```

**四种 Kernel 组合**:
- **WithoutScale + NotCutH (TilingKey=10001)**: 无 scales，hidden 可全载 UB
- **WithoutScale + CutH (TilingKey=10002)**: 无 scales，hidden 需切分
- **WithScale + NotCutH (TilingKey=20001)**: 有 scales，hidden 可全载
- **WithScale + CutH (TilingKey=20002)**: 有 scales，hidden 需切分

**支持平台**: ascend910b, ascend910_93, ascend950（APT/regbase 变体）
**支持数据类型**: FP16/FP32/BF16

---

## 2. 目录结构与文件职责

```
moe_finalize_routing_v2_grad/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_finalize_routing_v2_grad_def.cpp                # 算子定义
│   ├── moe_finalize_routing_v2_grad_tiling.h               # Tiling数据结构定义
│   ├── moe_finalize_routing_v2_grad_tiling.cpp              # 通用 Tiling 基类实现
│   ├── moe_finalize_routing_v2_grad_membase.cpp             # membase Tiling（910b/910_93）
│   ├── moe_finalize_routing_v2_grad_*_tiling_arch35.cpp     # arch35 Tiling 变体（3个文件）
│   ├── moe_finalize_routing_v2_grad_infershape.cpp          # Shape推导
│   ├── op_api/
│   │   ├── aclnn_moe_finalize_routing_v2_grad.h             # ACLNN API头文件
│   │   └── aclnn_moe_finalize_routing_v2_grad.cpp           # ACLNN API实现
│   └── config/ {ascend910b, ascend910_93, ascend950}
├── op_kernel/
│   ├── moe_finalize_routing_v2_grad.cpp                     # 910b/910_93 Kernel入口
│   ├── moe_finalize_routing_v2_grad_apt.cpp                 # 950 APT入口（regbase）
│   ├── moe_finalize_routing_v2_grad_base.h                  # 公共基类模板
│   ├── moe_finalize_routing_v2_grad_with_scale_cut_h.h
│   ├── moe_finalize_routing_v2_grad_with_scale_not_cut_h.h
│   ├── moe_finalize_routing_v2_grad_without_scale_cut_h.h
│   ├── moe_finalize_routing_v2_grad_without_scale_not_cut_h.h
│   └── arch35/ {base.h, regbase.h, regbase_cut_h.h, regbase_not_cut_h.h, ...}
├── docs/ aclnnMoeFinalizeRoutingV2Grad.md
└── tests/ut/ {op_host, op_api}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (op_host/moe_finalize_routing_v2_grad_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | grad_y | REQUIRED | FP16/FP32/BF16 | 2D: [N, H] |
| 输入 | expanded_row_idx | REQUIRED | INT32 | 1D: [N*K] |
| 输入 | expanded_x | OPTIONAL | FP16/FP32/BF16 | 2D/3D |
| 输入 | scales | OPTIONAL | FP16/FP32/BF16 | 2D: [N, K] |
| 输入 | expert_idx | OPTIONAL | INT32 | 2D: [N, K] |
| 输入 | bias | OPTIONAL | FP16/FP32/BF16 | 2D: [E, H] |
| 输出 | grad_expanded_x | REQUIRED | 与 grad_y 同类型 | 同 expanded_x |
| 输出 | grad_scales | REQUIRED | 与 grad_y 同类型 | 2D: [N, K] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| drop_pad_mode | Int | 0 | 0=无丢弃, 1=有丢弃填充 |
| active_num | Int | 0 | mode=0 时 expanded_x 实际有效第一维大小 |
| expert_num | Int | 0 | mode=1 时的专家数量 E |
| expert_capacity | Int | 0 | mode=1 时每专家容量 C |

**平台配置**（def.cpp 第66-149行）: ascend910b/910_93 使用 membase 配置，ascend950 使用 regbase 配置（`_apt` 变体，9 种数据类型组合）。

### 3.2 Tiling 策略

**Tiling 架构**:
```
TilingBaseClass
    └── MoeFinalizeRoutingV2GradTiling (抽象基类, tiling.cpp)
         ├── MoeFinalizeRoutingV2GradMembase (membase, tiling_membase.cpp)
         │   用于: ascend910b, ascend910_93
         └── arch35 子类（3个文件）
              用于: ascend950 (regbase)
```

**基类 Tiling 流程**（tiling.cpp）:

1. `GetShapeAttrsInfo()`（第168-189行）: 获取所有 tensor shape 和属性
2. `CalcBaseInfo()`（第469-487行）: 两阶段核分配
   - 阶段 1 (InitOut): 按 `expandedXDim0` 切分
   - 阶段 2 (Compute): 按 `expandedRowIdxDim0` 切分
   - `BlockDim = max(initOutNeedCoreNum, computeNeedCoreNum)`

**Membase CalcTilingKey**（tiling_membase.cpp 第52-105行）:

无 scales 时:
```
UB剩余 = ubSize - 32 (expandedRowIdx)
hiddenPrePart = UB剩余 / typeSize
hidden <= hiddenPrePart → Key 10001 (NOT_CUT_H)
hidden > hiddenPrePart → Key 10002 (CUT_H)
```

有 scales 时（考虑 FP32 精度提升）:
```
elementTotalSize = 4(gradY FP32) + 4(expandedX FP32) + typeSize(gradExpandedX) [+ 4(bias FP32)]
UB剩余 = ubSize - 4*32 (expandedRowIdx + scale + expertIdx + gradScales)
hiddenPrePart = UB剩余 / elementTotalSize
hidden <= hiddenPrePart → Key 20001 (NOT_CUT_H)
hidden > hiddenPrePart → Key 20002 (CUT_H)
```

**TilingKey 映射**:

| Key | 含义 |
|-----|------|
| 10001 | WITHOUT_SCALE_NOT_CUT_H |
| 10002 | WITHOUT_SCALE_CUT_H |
| 20001 | WITH_SCALE_NOT_CUT_H |
| 20002 | WITH_SCALE_CUT_H |

当 `drop_pad_mode=1` 时设置 `ScheduleMode=1`（batch mode，所有核同时启动）。

### 3.3 Shape 推导 (op_host/moe_finalize_routing_v2_grad_infershape.cpp)

**grad_expanded_x**（第40-107行）:
```
mode=0, active_num>0 且 active_num < N*K:  shape = [active_num, H]
mode=0, 其他:                              shape = [N*K, H]
mode=1:                                   shape = [E, C, H]  (3D)
```

**grad_scales**:
```
scales 存在时: shape = [N, K]
scales 不存在时: shape = [N, 1]
```

**Unknown Rank 处理**（第50-59行）: 两个输出均复制 `gradYShape`。

两个输出的数据类型均与 `grad_y` 相同。

### 3.4 ACLNN API (op_host/op_api/)

采用 CANN 标准两段式接口:

```cpp
aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradY,
    const aclTensor* expandedRowIdx,
    const aclTensor* expandedXOptional,
    const aclTensor* scalesOptional,
    const aclTensor* expertIdxOptional,
    const aclTensor* biasOptional,
    int64_t dropPadMode, int64_t activeNum,
    int64_t expertNum, int64_t expertCapacity,
    const aclTensor* gradExpandedXOut,
    const aclTensor* gradScalesOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)

aclnnStatus aclnnMoeFinalizeRoutingV2Grad(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream)
```

L2 层为薄封装层，直接委托给 `aclnnInner` 实现（aclnn_moe_finalize_routing_v2_grad.cpp），无额外数据转换。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (op_kernel/moe_finalize_routing_v2_grad.cpp)

**入口函数**: 接收 8 个 GM_ADDR 参数 + workspace + tiling。

**四种路径分发**（第36-57行）:

| TilingKey | 实例化的类 |
|-----------|-----------|
| 10001 | `MoeFinalizeRoutingV2GradWithoutScaleNotCutH` |
| 10002 | `MoeFinalizeRoutingV2GradWithoutScaleCutH` |
| 20001 | `MoeFinalizeRoutingV2GradWithScaleNotCutH` |
| 20002 | `MoeFinalizeRoutingV2GradWithScaleCutH` |

Kernel 任务类型: `KERNEL_TYPE_MIX_AIV_1_0`。ascend950 使用 `_apt.cpp`（arch35 子目录）。

### 4.2 核心计算流程

**两阶段执行架构**:

阶段 1 — InitOut（初始化梯度输出为零）:
```
for batchIdx in [startBatchIdx, endBatchIdx):
    InitOutput(gradExpandedXGm[batchIdx * H], H, 0)
    // CutH 时按 hiddenPrePart 切分循环
```

阶段 2 — Compute（梯度计算）:
```
for batchIdx in [computeStart, computeEnd):
    // 读取索引和权重
    expandedRowIdx_ = expandedRowIdxGm.GetValue(batchIdx)
    scale_ = scalesGm.GetValue(batchIdx)

    // 计算 grad_expanded_x (scatter-add)
    Muls(gradExpandedX, gradY, scale_)  // FP32 精度
    CopyOutGradExpandedX → GM[expandedRowIdx_ * H + offset]

    // 计算 grad_scales (reduce-sum)
    加载 expanded_x[row_idx * H + offset] [+ bias]
    Mul → gradY * (expandedX + bias)
    ReduceCalc → gradScalesSum
    CopyOutGradScales → GM
```

### 4.3 数据结构与内存布局

**Base 类成员**（base.h）:

| 类别 | 成员 | 说明 |
|------|------|------|
| GM Tensor | `gradYGm_`, `expandedRowIdxGm_`, `expandedXGm_`, `scalesGm_`, `expertIdxGm_`, `biasGm_`, `gradExpandedXGm_`, `gradScalesGm_` | 8 个 GM 缓冲区 |
| UB Queue | `gradYInQueue_`, `expandedRowIdxInQueue_`, `expandedXInQueue_`, `scalesInQueue_`, `expertIdxInQueue_`, `biasInQueue_`, `gradExpandedXOutQueue_`, `gradScalesOutQueue_` | 8 个队列，深度均为 1 |

**UB 缓冲分配**（WithScale 模式）:

| 缓冲 | 大小 | 说明 |
|------|------|------|
| `gradYInQueue_` | `hiddenPrePart * sizeof(float)` | FP32 精度 |
| `expandedXInQueue_` | `hiddenPrePart * sizeof(float)` | FP32 精度 |
| `gradExpandedXOutQueue_` | `hiddenPrePart * sizeof(T1)` | 原始精度 |
| `biasInQueue_` | `hiddenPrePart * sizeof(float)` | FP32 精度（有 bias 时） |
| `scalesInQueue_` | `BYTE_BLOCK` (32B) | 标量 |
| `expandedRowIdxInQueue_` | `BYTE_BLOCK` (32B) | 标量 |
| `expertIdxInQueue_` | `BYTE_BLOCK` (32B) | 标量（有 bias 时） |
| `gradScalesOutQueue_` | `BYTE_BLOCK` (32B) | 标量 |

FP16/BF16 路径的 UB 布局: `orgF16OBf16UbOffset_ = hiddenPrePart_`，gradY/bias/expandedX 在 FP32 空间中分配，原始精度数据放置在偏移 `hiddenPrePart_` 处，Cast 到 FP32 后使用前半部分。

### 4.4 多核并行策略

两阶段核分配:
- **InitOut**: 按 `expandedXDim0` 行数均匀分配核
- **Compute**: 按 `expandedRowIdxDim0` 行数均匀分配核
- `BlockDim = max(initOutNeedCoreNum, computeNeedCoreNum)`，空闲核在对应阶段直接跳过

### 4.5 通信模式

本算子使用 **Scatter-Add 无原子操作** 策略:
1. 阶段 1 所有核将 `grad_expanded_x` 初始化为零
2. 阶段 2 各核独立计算并使用 `DataCopy` 写回
3. 通过 MTE3→MTE2 事件同步确保写入顺序正确性

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

**MoeFinalizeRoutingV2GradTilingData**（tiling.h 第29-44行）:

```
MoeFinalizeRoutingV2GradTilingData
├── initOutNeedCoreNum      : int64_t   # 初始化阶段所需核数
├── initOutEachCoreBatchNum : int64_t   # 初始化每核处理行数
├── initOutModCoreNum       : int64_t   # 初始化取模核数
├── computeNeedCoreNum      : int64_t   # 计算阶段所需核数
├── computeEachCoreBatchNum : int64_t   # 计算每核处理行数
├── computeModCoreNum       : int64_t   # 计算取模核数
├── dropPadMode             : int64_t   # 丢弃填充模式
├── topK                    : int64_t   # K 的大小
├── hidden                  : int64_t   # hidden 维度 H
├── expandedXDim0           : int64_t   # grad_expanded_x 首维大小
├── hiddenPrePart           : int64_t   # hidden 切分前块大小
├── hiddenInnerLoops        : int64_t   # hidden 切分前块循环次数
├── hiddenLastPart          : int64_t   # hidden 切分尾块大小
└── tilingKey               : int64_t   # 模板标识
```

**辅助结构**:

- `MoeFinalizeRoutingV2GradBaseTilingData`: 基础参数（10 字段，不含 hidden 切分信息）
- `MoeFinalizeRoutingV2GradBinaryAddTilingData`: 二分归约参数（binaryAddQuotient/k/LastNum）
- `MoeFinalizeRoutingV2GradNotSplitHTilingData`: 不切 H 参数（baseParams + hAlign + binAddParams）
- `MoeFinalizeRoutingV2GradSplitHTilingData`: 切 H 参数（含 mainFactor/foldFactor 等）

**CompileInfo**: `totalCoreNum` + `totalUbSize`，用于动态编译静态图。

### 5.2 Workspace 布局

固定 16 MB（tiling.cpp）:
```cpp
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024;
```

Workspace 用于 Scatter-Add 中间缓冲和核间通信。

---

## 6. 完整数据流

### 6.1 WithScale 模式 (TilingKey=20001/20002)

```
  grad_y [N, H]    scales [N, K]    expanded_row_idx [N*K]
       |                |                    |
       v                v                    v
  +----------------------------------------------------------+
  | Kernel (两阶段)                                           |
  |                                                          |
  | 阶段1 InitOut:                                            |
  |   gradExpandedX[0..expandedXDim0] = 0                    |
  |                                                          |
  | 阶段2 Compute:                                            |
  |   per batchIdx:                                           |
  |     rowIdx = expanded_row_idx[batchIdx]                   |
  |     scale = scales[batchIdx]                              |
  |     per hiddenChunk:                                      |
  |       CopyIn(gradY[batchIdx/topK * H + h])                |
  |       Muls(gradExpandedX, gradY_FP32, scale) → FP32       |
  |       Cast → T1 → CopyOut(gradExpandedX[rowIdx * H + h])  |
  |       CopyIn(expandedX[rowIdx * H + h])                   |
  |       [+ CopyIn(bias[expertIdx * H + h])]                 |
  |       Mul(gradScales_tmp, gradY_FP32, expandedX_FP32)     |
  |       BlockReduceSum → gradScalesSum                      |
  |     Cast(gradScalesSum FP32→T1) → CopyOut(gradScales)     |
  +----------------------------------------------------------+
       |                              |
       v                              v
  grad_expanded_x [NK/H, H]    grad_scales [N, K]
  (scatter-add 写回)           (标量写回)
```

### 6.2 WithoutScale 模式 (TilingKey=10001/10002)

```
  grad_y [N, H]    expanded_row_idx [N*K]
       |                    |
       v                    v
  +----------------------------------------------------------+
  | Kernel (两阶段)                                           |
  |                                                          |
  | 阶段1 InitOut:                                            |
  |   gradExpandedX[0..expandedXDim0] = 0                    |
  |                                                          |
  | 阶段2 Compute:                                            |
  |   per batchIdx:                                           |
  |     rowIdx = expanded_row_idx[batchIdx]                   |
  |     per hiddenChunk:                                      |
  |       CopyIn(gradY[batchIdx/topK * H + h])                |
  |       CopyOut(gradExpandedX[rowIdx * H + h])              |
  |       // 纯 scatter，无数学运算                            |
  +----------------------------------------------------------+
       |
       v
  grad_expanded_x [NK/H, H]
  (scatter 写回)
```

---

## 7. 性能相关设计

1. **两阶段核分配**: 初始化和计算阶段使用不同的核分配策略，BlockDim 取两者最大值，空闲核在对应阶段跳过。

2. **FP32 精度提升**: FP16/BF16 路径将 gradY 和 expandedX 升精度到 FP32 进行计算。Muls 和 Add 在 FP32 空间执行，最终 Cast 回原始精度写回。UB 中使用 `orgF16OBf16UbOffset_` 分区原始精度和 FP32 数据。

3. **Scatter-Add 无原子操作**: 通过先 InitOutput 清零再累加的两阶段策略避免硬件原子操作。消除原子操作的性能开销和硬件限制。

4. **BinaryAdd 归约优化**: 使用二分法 BlockReduceSum 优化 H 维度的归约计算，参数通过 `SetBinaryAddParams` 预先计算。

5. **Batch Mode 并行**: 当 `drop_pad_mode=1` 时设置 `ScheduleMode=1`，所有核同时启动，提高并行度。

6. **H 维度切分**: 当 H 无法全载到 UB 时切分 hidden 维度，按 `hiddenPrePart` 循环覆盖，尾块通过 `hiddenLastPart` 处理。

---

## 8. 依赖关系

```
op_host/moe_finalize_routing_v2_grad_tiling.cpp
└── moe_finalize_routing_v2_grad_tiling.h
    ├── register/tilingdata_base.h
    ├── register/op_impl_registry.h
    ├── log/log.h
    ├── tiling_base/tiling_base.h
    ├── tiling_base/tiling_templates_registry.h
    ├── util/math_util.h
    ├── op_common/op_host/util/platform_util.h
    └── tiling_base/tiling_util.h

op_host/moe_finalize_routing_v2_grad_membase.cpp
└── moe_finalize_routing_v2_grad_tiling.h

op_host/op_api/aclnn_moe_finalize_routing_v2_grad.cpp
├── aclnn_moe_finalize_routing_v2_grad.h
│   ├── aclnn/aclnn_base.h
│   └── aclnn_util.h
├── opdev/make_op_executor.h
├── opdev/op_dfx.h, op_executor.h
├── common/op_api_def.h
└── aclnn_kernels/common/op_error_check.h

op_kernel/moe_finalize_routing_v2_grad.cpp
├── moe_finalize_routing_v2_grad_with_scale_cut_h.h
├── moe_finalize_routing_v2_grad_with_scale_not_cut_h.h
├── moe_finalize_routing_v2_grad_without_scale_cut_h.h
├── moe_finalize_routing_v2_grad_without_scale_not_cut_h.h
└── [all above include] moe_finalize_routing_v2_grad_base.h
    └── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| hidden 维度 | 受 UB 容量限制，无法全载时自动切分 |
| expandedXDim0 | mode=0 时为 N*K 或 active_num，mode=1 时为 E*C |
| scales 不存在时 | grad_scales 输出 shape 为 [N, 1]，不计算具体值 |
| bias 梯度 | 对 bias 的梯度通过 grad_scales 间接反映，不单独输出 |
| drop_pad_mode=1 | 需提供 expert_num 和 expert_capacity |
| 两阶段核数差异 | BlockDim 取 max，空闲核跳过 |

---

## 10. 与同族变体的核心差异

### 10.1 vs moe_finalize_routing_grad（V1 Grad）

| 维度 | V1 Grad | V2 Grad |
|------|---------|---------|
| scales 输入 | REQUIRED | OPTIONAL |
| expert_idx | 无 | 有（支持 bias） |
| bias | 不支持 | 支持 |
| drop_pad_mode | 不支持 | 支持（mode=0/1） |
| active_num | 无 | 有（mode=0 时裁剪） |
| grad_expanded_x shape | 固定 [N*K, H] | 按 mode/active_num 变化 |
| Kernel 组合 | 2 种 | 4 种 |
| arch35 变体 | 无 | 有（ascend950 regbase） |

### 10.2 vs moe_finalize_routing_v2（正向）

| 维度 | V2 正向 | V2 Grad |
|------|---------|---------|
| 输入数量 | 6 个（含 x1, x2, expanded_x, scales, bias） | 6 个（grad_y 替代 y，新增索引） |
| Scatter 方向 | 正向 scatter-read | 反向 scatter-add 写回 |
| 两阶段 | 无 | 有（InitOut + Compute） |
| 精度提升 | 可选 | 强制 FP32 |
