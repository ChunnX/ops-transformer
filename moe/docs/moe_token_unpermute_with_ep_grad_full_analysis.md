# moe_token_unpermute_with_ep_grad 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenUnpermuteWithEpGrad`
**算子类**: MoE Token 反排列反向算子，Expert Parallelism 变体
**功能描述**: 支持 Expert Parallelism 的 MoE Token 反排列反向算子。将 unpermuted_tokens_grad 按 sorted_indices 映射回 permuted 顺序，同时计算 probs 的梯度。EP range 属性指定当前卡的 token 范围。

**支持平台**: ascend910b, ascend910_93
**支持数据类型**: BF16/FP16/FP32

---

## 2. 目录结构与文件职责

```
moe_token_unpermute_with_ep_grad/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_token_unpermute_with_ep_grad_def.cpp            # 算子定义
│   ├── moe_token_unpermute_with_ep_grad_tiling.h           # Tiling数据结构
│   ├── moe_token_unpermute_with_ep_grad_tiling.cpp         # Tiling策略
│   ├── moe_token_unpermute_with_ep_grad_infershape.cpp     # Shape推导
│   └── config/ {ascend910_93, ascend910b}
├── op_kernel/
│   ├── moe_token_unpermute_with_ep_grad.cpp                # Kernel入口
│   ├── moe_token_unpermute_grad_base.h                    # 基础Kernel（共享）
│   ├── moe_token_unpermute_grad_prob_none.h               # 无probs模式
│   └── moe_token_unpermute_grad_prob_not_none.h           # 有probs模式
├── docs/ aclnnMoeTokenUnpermuteWithEpGrad.md
├── examples/ test_aclnn_moe_token_unpermute_with_ep_grad.cpp
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_token_unpermute_with_ep_grad_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | unpermuted_tokens_grad | REQUIRED | BF16/FP16/FP32 | 2D: [tokensNum, H] |
| 输入 | sorted_indices | REQUIRED | INT32 | 1D: [totalLength] |
| 输入 | permuted_tokens | OPTIONAL | BF16/FP16/FP32 | 2D: [numOutTokens, H] |
| 输入 | probs | OPTIONAL | BF16/FP16/FP32 | 2D: [tokensNum, topK] |
| 输出 | permuted_tokens_grad | REQUIRED | 与 tokens 同 | 2D: [totalLength, H] |
| 输出 | probs_grad | REQUIRED | 与 probs 同 | 2D: [tokensNum, topK] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| padded_mode | Bool | false | padding 模式 |
| restore_shape | ListInt | {1,1} | 恢复形状 |
| range | ListInt | [-1,-1] | EP 范围 |
| topk_num | Int | 1 | 每 token 的 expert 数 |

**关键差异**: 输入顺序与基础版 Grad 不同（unpermuted_tokens_grad 在第0位），permuted_tokens 变为 OPTIONAL。

### 3.2 Tiling 策略 (moe_token_unpermute_with_ep_grad_tiling.cpp)

**TilingData 结构**（tiling.h 第26-53行）:

```
MoeTokenUnpermuteWithEpGradTilingData
├── tokensNum              : int64_t
├── topK                   : int64_t
├── hiddenSize             : int64_t
├── numOutTokens           : int64_t
├── formerCoreNum          : int64_t   # 前组核数
├── tailCoreNum            : int64_t   # 后组核数
├── tokenNumEachCore       : int64_t
├── tokenNumTailCore       : int64_t
├── rowIdMapEachCore       : int64_t
├── rowIdMapTailCore       : int64_t
├── hiddenSizeAlign        : int64_t
├── hiddenSizeLoopTimes    : int64_t
├── hiddenSizeTail         : int64_t
├── inputReserveNum        : int64_t
├── indicesReserveNum      : int64_t
├── indicesReserveNumAlign : int64_t
├── totalUbSize            : int64_t
├── start                  : int64_t   # EP range 起始
└── end                    : int64_t   # EP range 结束
```

**EP Range 处理**（tiling.cpp 第344-382行）:
```
支持负索引（Python风格）:
  end = (end < 0) ? end + totalNum : end
  start = (start < 0) ? start + totalNum : start
边界裁剪后:
  if (start != 0) numOutTokens = end
```

**topK 推导**（第384-432行）:
```
topK = topk_num 属性 或 probs.dim(1)
numOutTokens = permuted_tokens.dim(0) 或 rowMap.dim(0)
```

**核内切分**（第150-198行）: `indicesReserveNum` 需要乘以 topK，与基础版不同。

**TilingKey**:
```
0/10: ProbNone (padded=false/true)
1/11: ProbNotNone (padded=false/true)
```

### 3.3 Shape 推导 (moe_token_unpermute_with_ep_grad_infershape.cpp)

```
permuted_tokens_grad: [rowIdMap.dim(0), hiddenSize]
probs_grad: 同 probs shape，或 nullptr（probs 不存在时）
probs_grad dtype = probs dtype（支持混合精度）
支持 UnknownRank (-2) 输入
```

### 3.4 ACLNN API

无独立的 ACLNN 两段式 API 文档。通过 L0 接口直接调用。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口 (moe_token_unpermute_with_ep_grad.cpp)

**输入参数顺序与基础版 Grad 不同**:
```
基础版: (permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, ...)
WithEp版: (unpermuted_tokens_grad, sorted_indices, permuted_tokens, probs, ...)
```

**TilingKey 分发**:
| Key | 模式 |
|-----|------|
| 0 | ProbNone, padded=false |
| 10 | ProbNone, padded=true |
| 1 | ProbNotNone, padded=false |
| 11 | ProbNotNone, padded=true |

### 4.2 核心计算流程

**复用基础版 Grad kernel 头文件**（第15-16行），但使用不同命名空间 `MoeTokenUnpermuteWithEpGrad`。

EP 版差异通过 tiling 参数体现:
- `start`/`end` 影响 `rowIdMapStartOffset` 和 `numOutTokens` 计算
- 核间切分考虑 EP range 的 `totalNum`

**注意**: 由于复用基础版 kernel，EP 版的 kernel 代码本身并没有显式使用 start/end 做 range 过滤（与 WithEp 正向版不同）。EP 的 range 效果主要通过 tiling 中的 `numOutTokens` 调整和 `sorted_indices` 的预处理来实现。

计算流程与基础版 Grad 相同:
- **ProbNone**: 纯 scatter（unpermuted_grad → permuted_grad）
- **ProbNotNone**: 计算 permuted_tokens_grad（scatter）+ probs_grad（reduce-sum）

### 4.3 数据结构与内存布局

复用基础版 Grad 的 `MoeTokenUnpermuteGradBase` 类。UB 缓冲分配与基础版相同。

### 4.4 多核并行策略

与基础版 Grad 相同: 按 tokensNum 均匀分配核（ProbNotNone）或按 numOutTokens 分配（ProbNone）。

### 4.5 通信模式

与基础版 Grad 相同，无 SyncAll。ProbNotNone 模式使用显式硬件事件同步。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 结构（19 个 int64_t 字段）。

### 5.2 Workspace 布局

仅使用系统 LibApi 工作空间，无额外算子级 workspace。

---

## 6. 完整数据流

### 6.1 ProbNotNone 模式

```
  unpermuted_tokens_grad [T, H]    sorted_indices [totalLength]
  permuted_tokens [numOut, H]      probs [T, topK]
       |                                |
       v                                v
  +----------------------------------------------------------+
  | Kernel (复用基础版Grad)                                    |
  |                                                          |
  | per token:                                               |
  |   for k in topK:                                         |
  |     idx = sorted_indices[token*topK+k]                   |
  |     Muls(grad, unpermutedGrad, prob[k]) → permutedGrad   |
  |     Mul(tmp, permutedTokens, unpermutedGrad)              |
  |     ReduceSum → probGrad[token, k]                       |
  +----------------------------------------------------------+
       |                              |
       v                              v
  permuted_tokens_grad           probs_grad
  [totalLength, H]               [T, topK]
```

### 6.2 ProbNone 模式

```
  unpermuted_tokens_grad [T, H]    sorted_indices [totalLength]
       |                                |
       v                                v
  +----------------------------------------------------------+
  | Kernel (纯 scatter)                                      |
  |                                                          |
  | per outIndex:                                            |
  |   idx = sorted_indices[offset]                           |
  |   CopyIn(unpermutedGrad[tokenIdx, h])                    |
  |   CopyOut(permutedGrad[idx, h])                          |
  +----------------------------------------------------------+
       |
       v
  permuted_tokens_grad [totalLength, H]
```

---

## 7. 性能相关设计

1. **Kernel 复用**: 复用基础版 Grad 的 kernel 头文件，EP 差异通过 tiling 参数体现，避免代码重复。

2. **负索引支持**: Range 属性支持 Python 风格负索引，方便调用者使用。

3. **EP range 预处理**: range 效果通过 tiling 中的 numOutTokens 调整实现，kernel 本身无需 range 过滤逻辑。

4. **FP32 精度累加**: 所有累加在 float 空间进行，减少精度损失。

---

## 8. 依赖关系

```
op_host/moe_token_unpermute_with_ep_grad_tiling.cpp
└── moe_token_unpermute_with_ep_grad_tiling.h
    ├── register/op_impl_registry.h
    ├── tiling/tiling_api.h
    ├── log/log.h
    └── tiling_base/tiling_base.h

op_kernel/moe_token_unpermute_with_ep_grad.cpp
├── moe_token_unpermute_grad_prob_none.h
│   └── moe_token_unpermute_grad_base.h
│       └── kernel_operator.h
├── moe_token_unpermute_grad_prob_not_none.h
│   └── moe_token_unpermute_grad_base.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| topK 上限 | topK <= 512（tiling.cpp 第419-422行） |
| dtype 一致 | permuted_tokens dtype 必须与 unpermuted_tokens_grad 一致（第441-445行） |
| sorted_indices | 必须为 1D（第402-405行） |
| range 顺序 | range[1] 必须 >= range[0]（第427-429行） |
| permuted_tokens | OPTIONAL（基础版是 REQUIRED） |

---

## 10. 与同族变体的核心差异

### 10.1 vs moe_token_unpermute_grad（基础版）

| 维度 | 基础版 Grad | WithEp Grad |
|------|------------|-------------|
| 输入顺序 | permuted_tokens 在前 | unpermuted_tokens_grad 在前 |
| permuted_tokens | REQUIRED | OPTIONAL |
| range 属性 | 无 | 有（含负索引） |
| topk_num 属性 | 无 | 有 |
| tiling start/end | 无 | 有 |
| kernel 代码 | 独立 | 复用基础版头文件 |
| numOutTokens 来源 | permuted_tokens.dim(0) | permuted_tokens 或 rowMap |

### 10.2 vs moe_token_unpermute_with_routing_map_grad

| 维度 | WithEp Grad | WithRoutingMap Grad |
|------|------------|-------------------|
| 输入数量 | 4 个 | 6 个（含 routing_map 等） |
| drop_and_pad | 不支持 | 支持 |
| routing_map | 无 | 有 |
| 子 kernel | 2 个 | 4+ |
| 混合精度 | 不支持 | 支持 |
| capacity/numExpert | 无 | 有 |
