# moe_token_unpermute 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenUnpermute`
**算子类**: MoE Token 反排列
**功能描述**: MoE Token 反排列算子。将经过 expert 排列后的 token（permuted_tokens）根据排序索引（sorted_indices）恢复到原始 token 顺序。支持可选的权重概率（probs）加权聚合——当一个原始 token 被多个 expert 处理时（TopK>1），通过 probs 权重将多个 expert 输出加权合并。支持混合精度（BF16 tokens + FP16/FP32 probs）。

**支持平台**: ascend910b, ascend910_93, ascend310p, kirinx90, kirin9030
**支持数据类型**: FP16/BF16/FP32（token），FP16/BF16/FP32（probs，可混合精度），INT32（索引）

---

## 2. 目录结构与文件职责

```
moe_token_unpermute/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_def.cpp             # 算子定义
│   ├── moe_token_unpermute_tiling.h            # TilingData结构
│   ├── moe_token_unpermute_tiling.cpp          # Tiling策略
│   ├── moe_token_unpermute_infershape.cpp      # Shape推导
│   ├── op_api/
│   │   ├── aclnn_moe_token_unpermute.h         # ACLNN API声明
│   │   └── aclnn_moe_token_unpermute.cpp       # ACLNN API实现
│   └── config/ {ascend310p, ascend910b, ascend910_93, kirinx90, kirin9030}
├── op_kernel/
│   ├── moe_token_unpermute.cpp                 # Kernel入口（19分支）
│   ├── moe_token_unpermute.h                   # Kernel主体模板实现
│   └── moe_token_unpermute_common.h            # 公共Kernel原语
├── examples/
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_token_unpermute_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | permuted_tokens | REQUIRED | FP16/BF16/FP32 | 2D: [numOutTokens, hiddenSize] |
| 输入 | sorted_indices | REQUIRED | INT32 | 1D: [totalLength] |
| 输入 | probs | OPTIONAL | FP16/BF16/FP32 | 2D: [tokensNum, topK] |
| 输出 | unpermuted_tokens | REQUIRED | 同 permuted_tokens | 2D: [tokensNum, hiddenSize] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| padded_mode | Bool | false | padding 模式（属性存在但 kernel 未实现） |
| restore_shape | ListInt | - | 恢复形状 |

**数据类型组合**: 9 种（含混合精度），permuted_tokens 与 probs 可以不同类型。310P/Kirin 仅 6 种（不含 BF16）。

### 3.2 Tiling 策略 (moe_token_unpermute_tiling.cpp)

**TilingData**（tiling.h）: 含 `MoeTokenUnpermuteParam`（InputParam: `tokensNum/topK/hiddenSize/totalLength/numOutTokens/haveProbs`）、3 个 `TilingParam`（`hiddenTiling/tokenTiling/tokenPerCore`: 各含 `length/num/remain`）、`CoreParam`（`maxCoreMemery/usedCoreNum/bufferNum/tilingKey`）。

**TilingKey 编码**（位域组合）:

| TilingKey | Token 类型 | Prob 类型 | 有 Probs |
|-----------|-----------|----------|---------|
| 0 | BF16 | - | 否 |
| 1 | BF16 | BF16 | 是 |
| 17 | BF16 | half | 是（混精） |
| 25 | BF16 | float | 是（混精） |
| 2 | FP16 | - | 否 |
| 3 | FP16 | FP16 | 是 |
| 11 | FP16 | BF16 | 是（混精） |
| 27 | FP16 | float | 是（混精） |
| 4 | FP32 | - | 否 |
| 5 | FP32 | FP32 | 是 |
| 13 | FP32 | BF16 | 是（混精） |
| 21 | FP32 | half | 是（混精） |

**Tiling 流程**: InputParamCheck → Init → SetCoreNum → TilingHiddenSize → SetBufferNum → ComputeRemainMemerySpace → TilingToken → SetTilingKey → SetTilingData。

**核内切分**: Hidden 维度按 512B 对齐切分；Buffer 数量 2~4 个动态调整；Token 维度按 indices+probs 所需空间切分。

### 3.3 Shape 推导 (moe_token_unpermute_infershape.cpp)

```
unpermuted_tokens: [tokensNum, hiddenSize]
  tokensNum = probs存在 ? probs.dim(0) : sorted_indices.dim(0)
  hiddenSize = permuted_tokens.dim(1)
  dtype = permuted_tokens.dtype
```

### 3.4 ACLNN API (aclnn_moe_token_unpermute.h)

**两段式接口**:

```cpp
// 第一段
aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(
    const aclTensor* permutedTokens, const aclTensor* sortedIndices,
    const aclTensor* probsOptional,
    bool paddedMode, const aclIntArray* restoreShapeOptional,
    const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

// 第二段
aclnnStatus aclnnMoeTokenUnpermute(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**Kernel 入口**（moe_token_unpermute.cpp）: 19 个 TilingKey 分支（13 非 310P + 6 310P）。

模板参数: `<typename T1, typename T2, typename T3, bool PROBS>`，T1=token 类型，T2=索引类型（固定 int32_t），T3=probs 类型，PROBS=是否有 probs。

### 4.2 核心计算流程

**数学公式**:
```
unpermuted_tokens[i, h] = sum_k(probs[i,k] * permuted_tokens[sorted_indices[i*topK+k], h])
若 probs=None: unpermuted_tokens[i, h] = permuted_tokens[sorted_indices[i*topK], h]  (scatter)
```

**Process**（三重循环结构）:
```
for tokens_splited_num:        // token 分块
  CalMultiOutToken()
    CopyIn indices + probs
    for out_tokens_number:     // 每个输出 token
      CalSingleOutToken()
        for hidden_splited:    // hidden 维度分块
          CalPartOutToken()
            读取 sorted_indices 中的 topK 个索引
            对每个索引:
              CopyTokenIn: 从 permuted_tokens 搬入数据
              Cast 到 FP32（如需）
              Muls(prob) + Add 累加
            CopyOut: Cast 回原类型并写出
```

### 4.3 数据结构与内存布局

**Kernel 数据流**:
```
permuted_tokens [numOutTokens, H]
  → tokens_inque → Cast(FP32) → Muls(prob) → Add 累加 → Cast 回原类型 → outque → unpermuted_tokens

sorted_indices [totalLength]
  → indices_inque → 读索引

probs [tokensNum, topK] (可选)
  → probs_inque → Cast(FP32) → probs_tensor
```

### 4.4 多核并行策略

按 tokensNum 切分到多核: 每核处理 `tokensNum / usedCoreNum` 个 token。核内按 hidden 维度分块处理。

### 4.5 通信模式

无 SyncAll。各核独立按 token 处理。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 MoeTokenUnpermuteParam。CompileInfo 含 `totalCoreNum`, `ubSize`。

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
  permuted_tokens [numOutTokens, H]   sorted_indices [totalLength]
       │                                    │
       │                              按 sorted_indices 寻址
       │                                    │
       └────────────────────────────────────┘
                   │
         对每行 i (核间并行):
           对每个 k in 0..topK-1:
             idx = sorted_indices[i*topK + k]
             若 probs 存在:
               result[i,h] += probs[i,k] * permuted_tokens[idx, h]
             否则:
               result[i,h] = permuted_tokens[idx, h]  (scatter)
                   │
           unpermuted_tokens [tokensNum, hiddenSize]
```

---

## 7. 性能相关设计

1. **混合精度支持**: token 和 probs 可使用不同数据类型（如 BF16 token + FP32 probs），Cast 到 FP32 累加后 Cast 回原类型。

2. **Hidden 维度分块**: hiddenSize 超过 UB 容量时按 512B 对齐切分，避免 UB 溢出。

3. **动态 Buffer 数量**: 2~4 个 buffer，根据 UB 剩余空间动态调整，最大化 UB 利用率。

4. **310P 适配**: 使用 DataCopyPadCustom/DataCopyCustom 替代标准 API，兼容 310P 硬件限制。

5. **三重循环结构**: token 分块 → 输出 token → hidden 分块，逐层处理减少 UB 占用峰值。

---

## 8. 依赖关系

```
op_host/moe_token_unpermute_def.cpp
└── register/op_def_registry.h

op_host/moe_token_unpermute_tiling.cpp
├── moe_token_unpermute_tiling.h
│   ├── register/op_impl_registry.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/op_api/aclnn_moe_token_unpermute.cpp
├── aclnn_moe_token_unpermute.h
└── aclnn_executor.h

op_kernel/moe_token_unpermute.cpp
├── moe_token_unpermute.h
├── moe_token_unpermute_common.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| permuted_tokens | 必须为 2D |
| probs | 若存在必须为 2D |
| topK | 必须 <= 512 |
| totalLength | 必须 == tokensNum * topK |
| 310P hiddenSize | 必须是 128 的整数倍且 <= 10240 |
| 310P | 不支持 BF16 |
| padded_mode | 属性存在但 kernel 未实现对应逻辑 |
| regbase | ascend950 regbase 版本分发到 MoeFinalizeRoutingV2 |

---

## 10. 与同族变体的核心差异

| 维度 | moe_token_unpermute（本算子） | moe_token_unpermute_with_ep | moe_token_unpermute_with_routing_map |
|------|------------------------------|----------------------------|--------------------------------------|
| EP range | 无 | 有（start/end） | 无 |
| routing_map | 无 | 无 | 有（BOOL/INT8） |
| 额外输入 | 无 | 无 | routing_map, out_index, permute_token_id |
| 额外输出 | 无 | 无 | out_index, permute_token_id, permute_probs |
| drop_and_pad | 不支持 | 不支持 | 支持 |
| SyncAll | 无 | 无 | 有 |
| 310P | 支持 | 不支持 | 不支持 |
| TilingKey 范围 | 0-27（位域组合） | 0-3（简单枚举） | 0/1000+ |
| 典型应用 | 基础推理 | EP 场景推理 | 复杂路由推理 |
