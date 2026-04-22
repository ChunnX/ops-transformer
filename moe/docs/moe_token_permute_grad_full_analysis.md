# moe_token_permute_grad 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenPermuteGrad`
**算子类**: MoE Token 重排反向算子，基础版本
**功能描述**: `moe_token_permute` 的反向算子。将专家处理后的梯度（已按 expert 排列的 permuted 梯度）按 sorted_indices 映射回原始 token 顺序，实现 unpermute（逆重排）操作。不处理 probs，每个输出 token 的梯度为 topK 个专家梯度的累加和。

**支持平台**: ascend910b, ascend910_93
**支持数据类型**: BF16/FP16/FP32

---

## 2. 目录结构与文件职责

```
moe_token_permute_grad/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_token_permute_grad_def.cpp            # 算子定义
│   ├── moe_token_permute_grad_infershape.cpp      # Shape推导
│   ├── moe_token_permute_grad_tiling.h            # Tiling数据结构定义
│   ├── moe_token_permute_grad_tiling.cpp          # Tiling策略实现
│   ├── op_api/
│   │   ├── aclnn_moe_token_permute_grad.h         # ACLNN API头文件
│   │   └── aclnn_moe_token_permute_grad.cpp       # ACLNN API实现
│   └── config/ {ascend910_93, ascend910b}
├── op_kernel/
│   ├── moe_token_permute_grad.cpp                 # Kernel入口（TilingKey分发）
│   └── moe_token_permute_grad.h                   # Kernel实现（KernelMoeTokenUnpermute）
├── docs/ aclnnMoeTokenPermuteGrad.md
├── examples/ test_aclnn_moe_token_permute_grad.cpp
└── tests/ut/ {op_host with op_api, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (op_host/moe_token_permute_grad_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | permuted_output_grad | REQUIRED | BF16/FP16/FP32 | 2D: [numOutTokens, H] |
| 输入 | sorted_indices | REQUIRED | INT32 | 1D: [N*topK] |
| 输出 | input_grad | REQUIRED | 与输入同类型 | 2D: [N, H] |

**属性**:

| 属性名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| num_topk | Int | REQUIRED | 每个 token 对应的专家数 (topK) |
| padded_mode | Bool | false | padding 模式（当前仅支持 false） |

### 3.2 Tiling 策略 (op_host/moe_token_permute_grad_tiling.cpp)

**Tiling 流程**（第302-320行 PermuteTilingCompute）:

```
InputParamCheck() → Init() → SetCoreNum() → TilingHiddenSize()
→ SetBufferNum() → ComputeRemainMemerySpace() → TilingToken()
→ SetTilingKey() → SetTilingData()
```

**核数分配**（第140-147行）: `usedCoreNum = min(tokensNum, maxCoreNum)`

**Hidden 维度切分**（第149-161行）:
```
maxHiddenSize = (UB - reserveSpace) / unitHSpace
如果 AlignN(hiddenSize, 512) <= maxHiddenSize: 不切分
否则: splitLength = maxHiddenSize, splitNum = hiddenSize / maxHiddenSize
```

**Buffer 数调整**（SetBufferNum）: 从最大 4 递减 buffer_num 直到 hidden tile 可放入 UB。

**TilingKey 编码**（tiling.h 第33-38行）:
```
Key=0: BF16 + 无probs
Key=2: FP16 + 无probs
Key=4: FP32 + 无probs
```

### 3.3 Shape 推导 (op_host/moe_token_permute_grad_infershape.cpp)

**推导逻辑**（第22-36行）:
```
输入: permuted_output_grad [numOutTokens, H], topK (attr)
推导: tokens_num = numOutTokens / topK
      input_grad = [tokens_num, H]
```

数据类型: `input_grad.dtype = permuted_output_grad.dtype`（第38-42行）

### 3.4 ACLNN API (op_host/op_api/)

采用 CANN 标准两段式接口:

```cpp
aclnnStatus aclnnMoeTokenPermuteGradGetWorkspaceSize(
    const aclTensor* permutedOutputGrad,
    const aclTensor* sortedIndices,
    int64_t numTopk,
    bool paddedMode,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)

aclnnStatus aclnnMoeTokenPermuteGrad(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream)
```

**L2 层双路径分发**（aclnn_moe_token_permute_grad.cpp 第45-85行）:

非 RegBase 平台: 直接委托 `aclnnInnerMoeTokenPermuteGrad*`

RegBase 平台: 委托 `aclnnInnerMoeInitRoutingV2Grad*`，将 `activeNum = permutedOutputGrad.dim(0)`，`dropPadMode` 硬编码为 0，并校验 `paddedMode == false`。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (op_kernel/moe_token_permute_grad.cpp)

**入口函数**（第19-43行）: 接收 3 个 GM_ADDR 参数 + workspace + tiling。

**TilingKey 分发**:
| Key | 实例化 |
|-----|--------|
| 0 | `KernelMoeTokenUnpermute<bfloat16_t, int32_t, bfloat16_t, false>` |
| 2 | `KernelMoeTokenUnpermute<half, int32_t, half, false>` |
| 4 | `KernelMoeTokenUnpermute<float, int32_t, float, false>` |

第 4 个模板参数 `PROBS=false`，表示不处理 probs 乘法。所有实例化传入 probs 为 `nullptr`。

### 4.2 核心计算流程

**KernelMoeTokenUnpermute**（moe_token_permute_grad.h）:

```
Process():
  for 每个token分片 (tokens_splited_num 次 + 尾块):
    CalMultiOutToken(out_offset, out_tokens_number)
      加载 sorted_indices 到 UB
      for 每个输出token:
        CalSingleOutToken(start_token, out_token_idx)
          for 每个hidden分片:
            CalPartOutToken(start_token, h_index, h_length, out_token_index)
              第一个token: CopyTokenIn → CalFirstToken (Cast→float)
              后续topK-1个: CopyTokenIn → CalToken (Cast→float→Add累加)
              CopyOut (Cast回原类型 → DataCopy)
```

**累加语义**:
```
input_grad[token_idx, h_start:h_end] =
    sum_{k=0}^{topK-1} permuted_grad[sorted_indices[token_idx*topK+k], h_start:h_end]
```

当 `sorted_indices[i] >= numOutTokens` 时跳过（不拷贝不累加）。

### 4.3 数据结构与内存布局

**Kernel 类成员**（moe_token_permute_grad.h 第47-62行）:

| 类别 | 成员 | 说明 |
|------|------|------|
| Queue | `tokens_inque` | `TQue<VECIN, 1>`，depth=buffer_num |
| Queue | `indices_inque` | `TQue<VECIN, 1>` |
| Queue | `probs_inque` | `TQue<VECIN, 1>`（PROBS=true 时） |
| Queue | `outque` | `TQue<VECOUT, 1>` |
| Buffer | `temp_buffer0/1` | `TBuf<VECCALC>`（非 float 时用于 Cast） |
| Buffer | `temp_buffer2` | `TBuf<VECCALC>`（PROBS 且非 float 时） |
| GM | `tokensGM`, `outGM` | `GlobalTensor<T1>` |
| GM | `indicesGM` | `GlobalTensor<T2>` |
| Local | `token_tensor0/1`, `probs_tensor` | `LocalTensor<float>` |

**UB 缓冲分配**（Init 第140-158行）:

| 缓冲 | 大小 | 条件 |
|------|------|------|
| `tokens_inque` | `buffer_num * hidden_splited_length_align512 * sizeof(T1)` | 始终 |
| `indices_inque` | `block_splited_length * sizeof(T2)` | 始终 |
| `outque` | `hidden_splited_length_align512 * sizeof(T1)` | 始终 |
| `temp_buffer0` | `hidden_splited_length_align512 * sizeof(float) + 256` | T1 ≠ float |
| `temp_buffer1` | `hidden_splited_length_align512 * sizeof(float)` | T1 ≠ float |
| `probs_inque` | `block_splited_length * sizeof(T3)` | PROBS=true |
| `temp_buffer2` | `block_splited_length * sizeof(float)` | PROBS && T3 ≠ float |

其中 `block_splited_length = tokens_splited_length * top_k`。

### 4.4 多核并行策略

按 tokenNum 均匀分配核（第140-147行 SetCoreNum），`usedCoreNum = min(tokensNum, maxCoreNum)`。尾部核多处理 1 个 token（`tokens_core_remain` 个核各多处理 1 个）。

### 4.5 通信模式

本算子不使用核间同步（SyncAll）。每核独立处理分配到的 token 范围，写入不重叠的 output_grad 行，无需通信。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

**MoeTokenPermuteGradTilingData**（tiling.h 第79-95行）:

```
MoeTokenPermuteGradTilingData
├── hidden_size            : int64_t   # hidden 维度
├── top_k                  : int64_t   # 每 token 的 expert 数
├── num_out_tokens         : int64_t   # 输出 token 总数
├── hidden_splited_length  : int64_t   # hidden 切分块大小
├── hidden_splited_num     : int64_t   # hidden 切分块数
├── hidden_splited_remain  : int64_t   # hidden 切分余数
├── tokens_core_length     : int64_t   # 每核 token 数
├── tokens_core_remain     : int64_t   # token 余数分配
├── tokens_splited_length  : int64_t   # token 切分块大小
├── tokens_splited_num     : int64_t   # token 切分块数
├── tokens_splited_remain  : int64_t   # token 切分余数
└── buffer_num             : int64_t   # UB buffer 数（2-4）
```

共 12 个字段，全部 `int64_t`。

### 5.2 Workspace 布局

仅使用系统 LibApi 工作空间（tiling.cpp 第102-138行）:
```
sysWorkspaceSize = ascendPlatform.GetLibApiWorkSpaceSize()
workspaces[0] = sysWorkspaceSize + 0  // UserWorkspace = 0
```

无额外算子级工作空间。所有计算通过 UB 缓冲完成。

---

## 6. 完整数据流

```
  permuted_output_grad [numOut, H]    sorted_indices [N*topK]
       |                                    |
       |                      +-------------+-------------+
       |                      | 加载到 UB                  |
       |                      | (按 tokens_splited_length) |
       |                      +-------------+-------------+
       |                                    |
       |                     逐 token 遍历:
       |                     for token in tokens_per_core:
       |                       for k in topK:
       |                         idx = sorted_indices[token*topK+k]
       |                         if idx < numOutTokens:
       +---------读取------------+---> permuted_grad[idx, h_part]
       |                                    |
       |                              Cast(BF16/FP16 → float)
       |                              Add 到累加器
       |                                    |
       |                              Cast(float → BF16/FP16)
       +---------写回---------------------+---> input_grad[token, h_part]
                                                    [N, H]
```

---

## 7. 性能相关设计

1. **float 精度累加**: 所有累加在 float 空间进行。BF16/FP16 输入先 Cast 到 float 累加，最后 Cast 回原类型，减少精度损失。

2. **动态 buffer_num**: 从最大 4 递减 buffer_num 直到 hidden tile 可放入 UB（SetBufferNum）。buffer_num 影响 `tokens_inque` 的队列深度 — 值越大双缓冲重叠越好。

3. **hidden 维度切分**: 当 hidden 无法全载到 UB 时自动切分。切分块大小按 512 字节对齐（`hidden_splited_length_align512`），确保 DataCopy 效率。

4. **双缓冲队列**: `tokens_inque` 使用 `buffer_num` 个深度槽位，允许 GM→UB 数据搬运与向量计算重叠。

5. **L2 RegBase 分发**: RegBase 平台上将本算子映射为 `MoeInitRoutingV2Grad`，复用 regbase 实现，避免重复开发。

---

## 8. 依赖关系

```
op_host/moe_token_permute_grad_tiling.cpp
└── moe_token_permute_grad_tiling.h
    ├── register/op_impl_registry.h
    ├── tiling/tiling_api.h
    ├── log/log.h
    └── tiling_base/tiling_base.h

op_host/op_api/aclnn_moe_token_permute_grad.cpp
├── aclnn_moe_token_permute_grad.h
│   ├── aclnn/aclnn_base.h
│   └── aclnn_util.h
├── opdev/make_op_executor.h, op_dfx.h, op_executor.h
├── common/op_api_def.h
├── aclnn_kernels/common/op_error_check.h
└── external/aclnn_kernels/aclnn_platform.h

op_kernel/moe_token_permute_grad.cpp
├── moe_token_permute_grad.h
│   ├── kernel_operator.h
│   └── [条件] moe_token_unpermute_common.h  (来自 moe_token_unpermute)
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| num_topk 必填 | 属性 REQUIRED，调用者必须显式提供 |
| padded_mode | 当前仅支持 false |
| topK 推导 | tokens_num = numOutTokens / topK，需整除 |
| 索引范围 | sorted_indices[i] >= numOutTokens 时跳过 |
| 无 probs | 本算子不处理 probs 乘法，权重始终为 1.0 |

---

## 10. 与同族变体的核心差异

### 10.1 vs moe_token_permute_with_ep_grad

| 维度 | moe_token_permute_grad | moe_token_permute_with_ep_grad |
|------|------------------------|-------------------------------|
| probs 输入 | 无 | 可选 permuted_probs_output_grad |
| probs 输出 | 无 | 必选 probs_grad |
| Kernel 模板 | `PROBS=false` | 两阶段: `ISTOKEN=true` + `ISTOKEN=false` |
| range 属性 | 无 | 支持 range[start, end) |
| TilingKey | 0/2/4 (按数据类型) | 0/1 (按是否有 probs) |
| 硬件事件同步 | PipeBarrier | 显式 SetFlag/WaitFlag |

### 10.2 vs moe_token_permute_with_routing_map_grad

| 维度 | moe_token_permute_grad | moe_token_permute_with_routing_map_grad |
|------|------------------------|----------------------------------------|
| 输入数 | 2 个 | 4 个（含 probs 梯度 + routing_map） |
| 输出数 | 1 个 | 2 个（tokens_grad + probs_grad） |
| dropAndPad | 不支持 | 支持 (TilingKey=1000) |
| 数据类型组合 | 3 种 | 8 种（含混合精度） |
