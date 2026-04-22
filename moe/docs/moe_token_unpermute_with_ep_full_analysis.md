# moe_token_unpermute_with_ep 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenUnpermuteWithEp`
**算子类**: MoE Token 反排列，Expert Parallelism 变体
**功能描述**: 支持 Expert Parallelism 的 MoE Token 反排列算子。在基础版上增加 `num_topk` 和 `range=[start, end]` 属性，用于多卡 EP 场景下指定当前卡负责的 expert token 范围。按 sorted_indices 将 permuted_tokens 映射回原始 token 顺序。

**支持平台**: ascend910b, ascend910_93, kirinx90, kirin9030
**支持数据类型**: BF16/FP16/FP32（probs 独立类型）

---

## 2. 目录结构与文件职责

```
moe_token_unpermute_with_ep/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_token_unpermute_with_ep_def.cpp            # 算子定义
│   ├── moe_token_unpermute_with_ep_tiling.h           # Tiling数据结构
│   ├── moe_token_unpermute_with_ep_tiling.cpp         # Tiling策略
│   ├── moe_token_unpermute_with_ep_infershape.cpp     # Shape推导
│   └── config/ {ascend910_93, ascend910b, kirinx90, kirin9030}
├── op_kernel/
│   ├── moe_token_unpermute_with_ep.cpp                # Kernel入口
│   └── moe_token_unpermute_with_ep.h                  # Kernel模板
├── docs/ aclnnMoeTokenUnpermuteWithEp.md
├── examples/ test_aclnn_moe_token_unpermute_with_ep.cpp
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_token_unpermute_with_ep_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | permuted_tokens | REQUIRED | BF16/FP16/FP32 | 2D: [numOutTokens, H] |
| 输入 | sorted_indices | REQUIRED | INT32 | 1D: [totalLength] |
| 输入 | probs | OPTIONAL | BF16/FP16/FP32 | 2D: [tokensNum, topK] |
| 输出 | unpermuted_tokens | REQUIRED | 与 permuted_tokens 同 | 2D: [tokensNum, H] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| num_topk | Int | 1 | 每个 token 的 expert 数 |
| range | ListInt | [0,0] | EP 范围 [start, end) |
| padded_mode | Bool | false | padding 模式 |
| restore_shape | ListInt | — | 恢复形状 |

### 3.2 Tiling 策略 (moe_token_unpermute_with_ep_tiling.cpp)

**TilingData 结构**（tiling.h 第38-103行）:

```
MoeTokenUnpermuteWithEpTilingData
├── hidden_size              : int64_t
├── top_k                    : int64_t
├── start                    : int64_t   # EP range 起始
├── end                      : int64_t   # EP range 结束
├── num_out_tokens           : int64_t
├── permuted_probs_grad_length : int64_t # probs_grad 长度标记
├── hidden_splited_length    : int64_t
├── hidden_splited_num       : int64_t
├── hidden_splited_remain    : int64_t
├── tokens_core_length       : int64_t
├── tokens_core_remain       : int64_t
├── tokens_splited_length    : int64_t
├── tokens_splited_num       : int64_t
├── tokens_splited_remain    : int64_t
└── buffer_num               : int64_t
```

与基础版 TilingData 对比: 新增 `start`、`end`、`permuted_probs_grad_length`。

**TilingKey 编码**（tiling.cpp 第288-311行）:
```
0 = 无 probs
1 = FP32 probs (isUnpermute=true) 或 有 probs (isUnpermute=false)
2 = FP16 probs
3 = BF16 probs
```

**Init EP 特殊处理**（第155-213行）:
```
读取 num_topk → topK
读取 range → start, end
if isUnpermute:
  topK = probs.dim(1) 或 inputTopK
  tokensNum = probs.dim(0) 或 totalLength / topK
```

### 3.3 Shape 推导 (moe_token_unpermute_with_ep_infershape.cpp)

```
tokensNum = probs 存在 ? probs.dim(0) : indices.dim(0) / inputTopK
输出 Shape: [tokensNum, permuted_tokens.dim(1)]
输出 dtype = permuted_tokens.dtype
```

### 3.4 ACLNN API

无独立的 ACLNN 两段式 API 文档。通过 L0 接口直接调用。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口 (moe_token_unpermute_with_ep.cpp)

**TilingKey 分发**（第32-50行）:

| Key | 实例化 |
|-----|--------|
| 0 | `KernelMoeTokenUnpermuteWithEp<DTYPE, int32_t, DTYPE, false, true>` |
| 1 | `KernelMoeTokenUnpermuteWithEp<DTYPE, int32_t, float, true, true>` |
| 2 | `KernelMoeTokenUnpermuteWithEp<DTYPE, int32_t, half, true, true>` |
| 3 | `KernelMoeTokenUnpermuteWithEp<DTYPE, int32_t, bfloat16_t, true, true>` |

所有路径 `ISTOKEN=true`（此 kernel 仅用于正向）。

### 4.2 核心计算流程

**模板参数**:
```cpp
template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
// ISTOKEN=true  → 正向 unpermute（处理 token 数据）
// ISTOKEN=false → 反向 grad 处理 probs 散射
```

**Range 过滤**（moe_token_unpermute_with_ep.h 第291-294行）:
```
needCopyIn = (acl_token_idx >= start) && (acl_token_idx < end)
// 基础版仅检查: acl_token_idx < num_out_tokens
```

**Token 偏移**（第336行）:
```
offset = (in_token_index - start) * hidden_size + ...
// 基础版: in_token_index * hidden_size + ...
```

计算流程与基础版相同: CopyTokenIn → Cast→float → 可选 Muls(prob) → Add 累加 → Cast 回原类型 → CopyOut。

### 4.3 数据结构与内存布局

与基础版 `KernelMoeTokenUnpermute` 相同的队列/缓冲区结构，额外增加:
- `start`/`end` 成员变量用于 range 过滤
- 8 个 EVENT_ID 用于 MTE2_S/S_MTE2/V_S/S_V 同步（第192-199行）

### 4.4 多核并行策略

按 tokenNum 均匀分配核，与基础版相同。EP range 通过 tiling 中的 start/end 参数影响每核的偏移计算。

### 4.5 通信模式

无 SyncAll。增加 8 个 EVENT_ID 用于硬件事件同步，避免 Scalar 操作阻塞 Vector 流水线。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 结构（15 个 int64_t 字段）。

### 5.2 Workspace 布局

固定 16 MB workspace（tiling.cpp 第390-392行）:
```
WORKSPACE_SIZE_WITH_M * BYTE_SIZE_WITH_K * BYTE_SIZE_WITH_K
```

---

## 6. 完整数据流

```
  permuted_tokens [end-start, H]    sorted_indices [totalLength]
       |                                  |
       |                    range 过滤: start <= idx < end
       |                                  |
       v                                  v
  +-------------------------------------------------------+
  | Kernel (per token_group)                              |
  |                                                       |
  | for each output_token:                                |
  |   for k in topK:                                      |
  |     idx = sorted_indices[token*topK+k]                |
  |     if start <= idx < end:                            |
  |       offset = (idx - start) * H + h_part            |
  |       CopyTokenIn(permuted[offset])                   |
  |       Cast→float → [可选 Muls(prob)] → Add累加        |
  |       Cast回原类型 → CopyOut(unpermuted[token*H+h])   |
  +-------------------------------------------------------+
       |
       v
  unpermuted_tokens [tokensNum, H]
```

---

## 7. 性能相关设计

1. **EP Range 过滤**: 仅处理 `[start, end)` 范围内的 token，跳过其他卡的 token，减少无效计算。

2. **显式硬件事件同步**: 8 个 EVENT_ID 替代隐式 PipeBarrier，减少 Scalar 阻塞 Vector 流水线的时间。

3. **正向/反向共用 Kernel**: 通过 `ISTOKEN` 编译期模板参数区分，避免代码重复。

4. **Token 偏移优化**: `(idx - start) * H` 避免了全局偏移，使 GM 访问局部化。

---

## 8. 依赖关系

```
op_host/moe_token_unpermute_with_ep_tiling.cpp
└── moe_token_unpermute_with_ep_tiling.h
    ├── register/op_impl_registry.h
    ├── tiling/tiling_api.h
    ├── log/log.h
    └── tiling_base/tiling_base.h

op_kernel/moe_token_unpermute_with_ep.cpp
├── moe_token_unpermute_with_ep.h
│   └── kernel_operator.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| topK 上限 | topK <= 512（tiling.cpp 第142-143行） |
| inputTopK | inputTopK >= 1（第110行） |
| probs.dim(1) | 必须等于 num_topk（第133-135行） |
| 310P | 不支持 |
| range | 默认 [0, 0]，仅处理 [start, end) 范围 |

---

## 10. 与同族变体的核心差异

### 10.1 vs moe_token_unpermute（基础版）

| 维度 | 基础版 | WithEp 版 |
|------|--------|-----------|
| range 支持 | 无 | 有 start/end |
| num_topk 属性 | 无 | 有 |
| ISTOKEN 模板 | 无 | 有（正向/反向共用） |
| token 偏移 | `idx * H` | `(idx - start) * H` |
| range 过滤 | `idx < numOutTokens` | `start <= idx < end` |
| Event 同步 | 无显式 | 8 个 EVENT_ID |
| workspace | LibApiWorkSpaceSize | 固定 16 MB |

### 10.2 vs moe_token_unpermute_with_routing_map

| 维度 | WithEp 版 | WithRoutingMap 版 |
|------|----------|------------------|
| routing_map 输入 | 无 | 有 (BOOL/INT8) |
| drop_and_pad | 不支持 | 支持 |
| MaskedSelect 阶段 | 无 | 有 |
| 额外输出 | 无 | out_index, permute_token_id, permute_probs |
| SyncAll | 不需要 | 需要 |
