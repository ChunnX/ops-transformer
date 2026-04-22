# moe_token_permute_with_routing_map_grad 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenPermuteWithRoutingMapGrad`
**算子类**: MoE Token 重排反向算子，路由矩阵变体
**功能描述**: `moe_token_permute_with_routing_map` 的反向算子。将专家处理后的 token 梯度和 probs 梯度映射回原始 token 顺序，支持 `drop_and_pad` 模式下的 probs 梯度恢复。

**两种计算模式**:
- **非 dropAndPad (TilingKey=0)**: 复用 `KernelMoeTokenUnpermute`，标准 unpermute 逻辑
- **dropAndPad (TilingKey=1000)**: 使用 `KernelMoeTokenPermuteWithRoutingMap`，按 capacity/expertNum 坐标映射恢复 probs 梯度

**支持平台**: ascend910b, ascend910_93, ascend950
**支持数据类型**: BF16/FP16/FP32（含混合精度组合）

---

## 2. 目录结构与文件职责

```
moe_token_permute_with_routing_map_grad/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_token_permute_with_routing_map_grad_def.cpp            # 算子定义
│   ├── moe_token_permute_with_routing_map_grad_infershape.cpp     # Shape推导
│   ├── moe_token_permute_with_routing_map_grad_tiling.h           # Tiling数据结构定义
│   ├── moe_token_permute_with_routing_map_grad_tiling.cpp         # Tiling策略实现
│   ├── moe_token_permute_grad_tiling.h                            # 共享 Unpermute Tiling 头文件
│   ├── op_api/
│   │   ├── aclnn_moe_token_permute_with_routing_map_grad.h        # ACLNN API头文件
│   │   └── aclnn_moe_token_permute_with_routing_map_grad.cpp      # ACLNN API实现
│   └── config/ {ascend910_93, ascend910b}
├── op_kernel/
│   ├── moe_token_permute_with_routing_map_grad.cpp                # Kernel入口（TilingKey分发）
│   ├── moe_token_permute_with_routing_map_grad.h                  # DropPad Kernel实现
│   └── moe_token_unpermute.h                                      # 共享 Unpermute Kernel
├── docs/ aclnnMoeTokenPermuteWithRoutingMapGrad.md
├── examples/ test_aclnn_moe_token_permute_with_routing_map_grad.cpp
└── tests/ut/ {op_host with op_api, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (op_host/moe_token_permute_with_routing_map_grad_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | permuted_token_output_grad | REQUIRED | BF16/FP16/FP32 | 2D: [numOutTokens, H] |
| 输入 | permuted_probs_output_grad_optional | OPTIONAL | BF16/FP16/FP32 | 2D (dropAndPad) |
| 输入 | sorted_indices | REQUIRED | INT32 | 1D |
| 输入 | routing_map_optional | OPTIONAL | INT8/BOOL | 2D |
| 输出 | tokens_grad_out | REQUIRED | 与输入同类型 | 2D: [tokensNum, H] |
| 输出 | probs_grad_out_optional | OPTIONAL | 与输入同类型 | 2D: [tokensNum, topK] |

**属性**:

| 属性名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| num_experts | Int | REQUIRED | 专家数量 |
| tokens_num | Int | REQUIRED | token 数量 |
| padded_mode | Bool | false | 是否 dropAndPad 模式 |

**数据类型组合**（def.cpp 第26-68行）: 共 8 种组合，含混合精度（如 token_grad 为 BF16 但 probs_grad 为 FP32）。

### 3.2 Tiling 策略 (op_host/moe_token_permute_with_routing_map_grad_tiling.cpp)

**两条 Tiling 路径**:

非 dropAndPad 模式（`TilingCompute`）:
1. 复用 `moe_token_permute_grad` 的共享 Tiling 逻辑
2. 计算核数、hidden 维度切分、buffer 数、token 维度切分
3. TilingKey = 0

dropAndPad 模式（`TilingComputeDropPad`）:
1. 读取 `expertsNum` 和 `tokenNum` 属性
2. 计算 `capacity = inputDim0 / expertsNum`
3. 核数: `min(expertsNum * capacity, maxCoreNum)`
4. TilingKey = 1000（`TILINGKEY_DROPPAD`）

### 3.3 Shape 推导 (op_host/moe_token_permute_with_routing_map_grad_infershape.cpp)

**推导逻辑**（第19-38行）:
```
输入:
  permuted_token_output_grad: [numOutTokens, H]
  tokensNum (attr)

推导:
  hidden_size = permuted_token_output_grad.GetDim(1)
  topK_num = permuted_token_output_grad.GetDim(0) / tokensNum
  tokens_grad_out = [tokensNum, hidden_size]
  probs_grad_out_optional = [tokensNum, topK_num]
```

### 3.4 ACLNN API (op_host/op_api/)

采用 CANN 标准两段式接口:

**第一阶段**:
```cpp
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(
    const aclTensor* permutedTokenOutputGrad,
    const aclTensor* permutedProbsOutputGradOptional,
    const aclTensor* sortedIndices,
    const aclTensor* routingMapOptional,
    int64_t numExperts,
    int64_t tokensNum,
    bool dropAndPad,
    aclTensor* tokensGradOut,
    aclTensor* probsGradOutOptional,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

**第二阶段**:
```cpp
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGrad(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream)
```

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (op_kernel/moe_token_permute_with_routing_map_grad.cpp)

**入口函数**（第21-43行）: 接收 8 个 GM_ADDR 参数。

**两种路径分发**:
- **TilingKey=0**: 实例化 `KernelMoeTokenUnpermute<DTYPE, int32_t, DTYPE, false>`（PROBS=false），处理 token 梯度 unpermute
- **TilingKey=1000**: 实例化 `KernelMoeTokenPermuteWithRoutingMap<DTYPE, int32_t, DTYPE>`，处理 dropPad probs 梯度恢复

### 4.2 核心计算流程

**非 dropAndPad 模式**: 复用 `KernelMoeTokenUnpermute`（来自 `moe_token_unpermute.h`），标准 unpermute 逻辑。按 sortedIndices 反向映射梯度，Cast 到 float 累加后 Cast 回原类型。

**dropAndPad 模式**（moe_token_permute_with_routing_map_grad.h）:
```
for i in [startIdx, endIdx):
    permutedProbValue = permutedProbsGM.GetValue(i)
    xIndex = sortedIndicesGM.GetValue(i)         // 原始 token 位置
    yIndex = i / capacity                         // 专家编号
    outOffset = xIndex * expertNum + yIndex       // 输出位置
    probsGradOutGM[outOffset] = permutedProbValue
```

语义: 将已按 `expert * capacity` 排列的 probs 梯度恢复为 `[tokensNum, expertNum]` 的原始布局。

### 4.3 数据结构与内存布局

**DropPad 模式**下 Kernel 使用标量读写（GetValue/SetValue），不需要 UB 缓冲区。每核处理 `singleCoreLen` 或 `lastCoreLen` 个元素。

### 4.4 多核并行策略

- **非 dropAndPad**: 按 tokenNum 均匀分配核（与 `moe_token_permute_grad` 相同）
- **dropAndPad**: 按 `expertsNum * capacity` 分配核，每核独立处理一段连续元素

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

**MoeTokenPermuteWithRoutingMapGradTilingData**（tiling.h）:

```
MoeTokenPermuteWithRoutingMapGradTilingData
├── moeTokenPermuteWithRoutingMapGradUnpermuteTilingData
│   (复用自 moe_token_permute_grad_tiling.h 的共享结构)
│   ├── hidden_size, permuted_probs_grad_length, top_k
│   ├── start, end, num_out_tokens
│   ├── hidden_splited_length/num/remain
│   ├── tokens_core_length/remain
│   ├── tokens_splited_length/num/remain
│   └── buffer_num
└── moeTokenpermuteWithRoutingMapDropPadTilingData
    ├── capacity       : int64_t   # 每个专家的容量
    ├── expertNum      : int64_t   # 专家数
    ├── tokenNum       : int64_t   # token 数
    ├── singleCoreLen  : int64_t   # 每核处理长度
    └── lastCoreLen    : int64_t   # 最后一个核处理长度
```

### 5.2 Workspace 布局

**Workspace 大小**: 固定 16MB（`16 * 1024 * 1024`），由两条 Tiling 路径统一设置。非 dropAndPad 模式下 Workspace 用于排序中间缓冲。

---

## 6. 完整数据流

### 6.1 非 dropAndPad 模式 (TilingKey=0)

```
  permutedTokenOutPutGrad [numOut, H]    sortedIndices [totalLength]
         |                                      |
         +----------+---------------------------+
                    |
          KernelMoeTokenUnpermute (复用基础版)
          per token:
            idx = sortedIndices[token_idx]
            CopyTokenIn(idx * H + h_part)
            Cast → float 累加 (CalFirstToken / CalToken)
            CopyOut → Cast 回原类型
                    |
             tokensGradOut [tokensNum, H]
```

### 6.2 dropAndPad 模式 (TilingKey=1000)

```
  permutedProbsOutPutGrad [E*capacity]    sortedIndices [E*capacity]
         |                                       |
         +----------+----------------------------+
                    |
          KernelMoeTokenPermuteWithRoutingMap
          for i in [startIdx, endIdx):
            probVal = grad[i]
            tokenIdx = sortedIndices[i]
            expertIdx = i / capacity
            out[tokenIdx * expertNum + expertIdx] = probVal
                    |
           probsGradOutOptional [tokensNum, expertNum]
```

---

## 7. 性能相关设计

1. **Kernel 复用**: 非 dropAndPad 模式直接复用 `KernelMoeTokenUnpermute`，避免重复实现，减少维护成本。

2. **DropPad 标量操作**: dropPad 模式使用 `GetValue/SetValue` 标量读写而非 DMA 批量搬运，因为每个 probs 梯度的输出位置是离散的（scatter 写），无法形成连续内存块。

3. **共享 Tiling 代码**: 复用 `moe_token_permute_grad` 的 Tiling 计算逻辑，确保 unpermute 路径的参数与正向 permute_grad 完全一致。

4. **混合精度支持**: 8 种数据类型组合含混合精度场景（token_grad BF16 + probs_grad FP32），由 def.cpp 编译时展开。

---

## 8. 依赖关系

```
op_host/moe_token_permute_with_routing_map_grad_tiling.cpp
└── moe_token_permute_with_routing_map_grad_tiling.h
    └── moe_token_permute_grad_tiling.h                    # 共享 Unpermute Tiling 结构

op_host/op_api/aclnn_moe_token_permute_with_routing_map_grad.cpp
├── aclnn_moe_token_permute_with_routing_map_grad.h
└── L0 算子调用

op_kernel/moe_token_permute_with_routing_map_grad.cpp
├── moe_token_unpermute.h                                  # 共享 Unpermute Kernel
│   └── kernel_operator.h
├── moe_token_permute_with_routing_map_grad.h              # DropPad Kernel
│   ├── kernel_operator.h
│   ├── moe_token_unpermute.h
│   └── kernel_tiling/kernel_tiling.h
└── kernel_operator.h
```

---

## 9. 与同族变体的核心差异

### 9.1 vs moe_token_permute_grad

| 维度 | moe_token_permute_grad | moe_token_permute_with_routing_map_grad |
|------|------------------------|----------------------------------------|
| 输入数量 | 2 个 | 4 个（含可选 probs 梯度和 routing_map） |
| 输出数量 | 1 个 | 2 个（tokens_grad + probs_grad） |
| dropAndPad | 不支持 | 支持（TilingKey=1000） |
| probs 梯度恢复 | 无 | 有（按 capacity/expertNum 重排） |
| Kernel 复用 | 独立实现 | 复用 KernelMoeTokenUnpermute |
| 属性 | num_topk + padded_mode | num_experts + tokens_num + padded_mode |
| 数据类型组合 | 3 种 | 8 种（含混合精度） |

### 9.2 vs moe_token_permute_with_ep_grad

| 维度 | EP Grad | Routing Map Grad |
|------|---------|-----------------|
| range 支持 | [start, end) | 无 range |
| dropAndPad | 不支持 | 支持 |
| probs_grad 计算 | ISTOKEN=false 模式 | KernelMoeTokenPermuteWithRoutingMap |
| TilingData 结构 | 单一结构 | 双子结构（unpermute + dropPad） |
| probs 恢复逻辑 | 按索引 + 范围判断 | 按 capacity/expertNum 坐标映射 |

---

## 10. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| dropAndPad 时 capacity 整除 | capacity = numOutTokens / expertNum，需整除 |
| topK 上限 | topK <= 512 |
| tokens 必须为 2D | shape 维度校验 |
| padded_mode | 非 dropAndPad 时必须为 false |
| 共享 Tiling 一致性 | unpermute 路径的 Tiling 参数须与 moe_token_permute_grad 一致 |
| 混合精度 | probs_grad 可能为 FP32 而 token_grad 为 BF16 |
