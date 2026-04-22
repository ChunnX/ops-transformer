# moe_token_unpermute_grad 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenUnpermuteGrad`
**算子类**: MoE Token 反排列反向传播
**功能描述**: MoE Token 反排列的反向传播算子。给定前向输出的梯度（unpermuted_tokens_grad），根据 sorted_indices 和可选的 probs，计算 permuted_tokens 的梯度和 probs 的梯度。分 ProbNone（纯 scatter）和 ProbNotNull（scatter + ReduceSum for probs_grad）两种路径。

**支持平台**: ascend910b, ascend910_93
**支持数据类型**: FP16/BF16/FP32（token），FP16/BF16/FP32（probs），INT32（索引）

---

## 2. 目录结构与文件职责

```
moe_token_unpermute_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_grad_def.cpp             # 算子定义
│   ├── moe_token_unpermute_grad_tiling.h            # TilingData结构（16字段）
│   ├── moe_token_unpermute_grad_tiling.cpp          # Tiling策略（4 TilingKey）
│   ├── moe_token_unpermute_grad_infershape.cpp      # Shape推导
│   ├── op_api/
│   │   ├── aclnn_moe_token_unpermute_grad.h         # ACLNN API声明
│   │   └── aclnn_moe_token_unpermute_grad.cpp       # ACLNN API实现
│   └── config/ {ascend910b, ascend910_93}
├── op_kernel/
│   ├── moe_token_unpermute_grad.cpp                 # Kernel入口（4分支）
│   ├── moe_token_unpermute_grad_base.h              # 基类（ReduceSum）
│   ├── moe_token_unpermute_grad_prob_none.h         # 无 probs 路径
│   └── moe_token_unpermute_grad_prob_not_none.h     # 有 probs 路径
├── examples/
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_token_unpermute_grad_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | permuted_tokens | REQUIRED | FP16/BF16/FP32 | 2D: [numOutTokens, hiddenSize] |
| 输入 | unpermuted_tokens_grad | REQUIRED | 同上 | 2D: [tokensNum, hiddenSize] |
| 输入 | sorted_indices | REQUIRED | INT32 | 1D: [totalLength] |
| 输入 | probs | OPTIONAL | FP16/BF16/FP32 | 2D: [tokensNum, topK] |
| 输出 | permuted_tokens_grad | REQUIRED | 同 permuted_tokens | 2D: [totalLength, hiddenSize] |
| 输出 | probs_grad | REQUIRED | 同 probs | 2D: [tokensNum, topK] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| padded_mode | Bool | false | padding 模式 |
| restore_shape | ListInt | {1,1} | 恢复形状 |

### 3.2 Tiling 策略 (moe_token_unpermute_grad_tiling.cpp)

**TilingData**（tiling.h）: 16 个字段，含 `tokensNum/topK/hiddenSize/numOutTokens`（维度）、`formerCoreNum/tailCoreNum/tokenNumEachCore/tokenNumTailCore`（核间切分）、`rowIdMapEachCore/rowIdMapTailCore`（indices 映射）、`hiddenSizeAlign/hiddenSizeLoopTimes/hiddenSizeTail`（hidden 切分）、`inputReserveNum/indicesReserveNum/indicesReserveNumAlign`（UB 缓冲）、`totalUbSize`。

**TilingKey 编码**:

| Key | padded_mode | 有 probs |
|-----|-------------|---------|
| 0 | false | 否 |
| 1 | false | 是 |
| 10 | true | 否 |
| 11 | true | 是 |

**核间切分**: 按 tokensNum 切分到多核，`formerCoreNum = tokensNum % totalCoreNum`（前 formerCoreNum 核多处理 1 个 token），`rowIdMapEachCore = tokenNumEachCore * topK`。

**核内切分**: ProbNone 路径 UB 分配 `indicesReserveNum * (hiddenSizeAlign * inputType + rowIdMapType)`；ProbNotNull 路径 indicesReserveNum 最大 256，需额外 prob fp32 cast 和 ReduceSum 空间。

### 3.3 Shape 推导 (moe_token_unpermute_grad_infershape.cpp)

```
permuted_tokens_grad: [totalLength, hiddenSize]   (同 permuted_tokens 行数)
probs_grad: 同 probs 形状（probs 为 None 时为 nullptr）
```

### 3.4 ACLNN API (aclnn_moe_token_unpermute_grad.h)

**两段式接口**:

```cpp
// 第一段
aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(
    const aclTensor* permutedTokens, const aclTensor* unpermutedTokensGrad,
    const aclTensor* sortedIndices, const aclTensor* probsOptional,
    bool paddedMode, const aclIntArray* restoreShapeOptional,
    const aclTensor* permutedTokensGradOut, const aclTensor* probsGradOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

// 第二段
aclnnStatus aclnnMoeTokenUnpermuteGrad(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**Kernel 入口**（moe_token_unpermute_grad.cpp）: 4 个 TilingKey 分支。

```
moe_token_unpermute_grad() 入口
  +-- Key=0/10: MoeTokenUnpermuteGradProbNone<DTYPE, int32_t>
  +-- Key=1/11: MoeTokenUnpermuteGradProbNotNull<DTYPE, int32_t, DTYPE_PROBS>
```

**类继承体系**:
```
MoeTokenUnpermuteGradBase<OriT, IdxT, ProbT>      ← 基类：公共初始化 + ReduceSum
  +-- MoeTokenUnpermuteGradProbNone<OriT, IdxT>    ← 无 probs 路径
  +-- MoeTokenUnpermuteGradProbNotNull<OriT, IdxT, ProbT>  ← 有 probs 路径
```

### 4.2 核心计算流程

**ProbNone 路径数学公式**:
```
permuted_tokens_grad[sorted_indices[i], h] += unpermuted_tokens_grad[i/topK, h]
（直接 scatter-add）
```

**ProbNotNull 路径数学公式**:
```
permuted_tokens_grad[j, h] += probs[i, k] * unpermuted_tokens_grad[i, h]
    （其中 j = sorted_indices[i*topK+k]）

probs_grad[i, k] = sum_h(permuted_tokens[j, h] * unpermuted_tokens_grad[i, h])
    （其中 j = sorted_indices[i*topK+k]）
```

**ProbNone 流程**:
```
for indicesLoop (按 indicesReserveNum 切分):
  CopyIn sorted_indices
  for hiddenLoop (按 hiddenSizeAlign 切分):
    CopyIn unpermuted_tokens_grad
    for index in indicesLoopNum:
      offset = sorted_indices[index]
      if offset < numOutTokens:
        DataCopyPad(permuted_tokens_grad[offset]) ← unpermuted_tokens_grad[index]
```

**ProbNotNull 流程**:
```
for indicesLoop:
  CopyIn probs → Cast 到 FP32 → 存入 probsArray[]（本地数组，最大 512）
  CopyIn indices → 存入 indicesArray[]
  Duplicate(probGradReduceSum, 0)

  for hiddensizeLoop:
    CopyIn unpermuted_tokens_grad → Cast 到 FP32
    for topKInIndicesLoop / permuteInTopKLoop:
      CopyIn permuted_tokens
      // permuted_tokens_grad: Muls(grad, prob) → Cast 回原类型 → CopyOut
      // probs_grad: Mul(permuted, grad) → ReduceSumFunc

  Add(probGradReduceSum, probGradReduceSum, probGradFp32)
  CopyOut prob_grad
```

### 4.3 数据结构与内存布局

**Ping-Pong Double Buffer**（ProbNotNull 路径）: 手动管理 8 个 event ID 实现 7 组数据的 double buffer 同步:
- indices: EVENT_ID0/1
- probs: EVENT_ID2/3
- unpermutedOutputD: EVENT_ID4/5
- permuted_tokens: EVENT_ID6/7
- permuted_tokens_grad / prob_grad: EVENT_ID0/1（复用）

**本地数组**: `indicesArray[]` 和 `probsArray[]` 最大 512 元素。

### 4.4 多核并行策略

按 tokensNum 均匀分核: `tokenNumEachCore = tokensNum / totalCoreNum + (formerCoreNum > 0 ? 1 : 0)`。核内按 indices（indicesReserveNum）和 hidden（hiddenSizeAlign）双重切分。

### 4.5 通信模式

无 SyncAll。各核独立按 token 处理，scatter 写 GM 不需要同步。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

16 个字段的 MoeTokenUnpermuteGradTilingData（见 3.2 节）。CompileInfo 含 `totalCoreNum`, `ubSize`。

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
  unpermuted_tokens_grad [tokensNum, H]   sorted_indices [totalLength]
       │                                        │
       │                                  按 sorted_indices 寻址
       │                                        │
  ┌────┴────────────────────────────────────────┘
  │
  │ ProbNone (Key=0/10):
  │   对每个 index:
  │     offset = sorted_indices[index]
  │     permuted_tokens_grad[offset, h] += unpermuted_tokens_grad[index/topK, h]
  │
  │ ProbNotNull (Key=1/11):
  │   对每个 token i 和每个 k:
  │     j = sorted_indices[i*topK + k]
  │     permuted_tokens_grad[j, h] += probs[i,k] * unpermuted_tokens_grad[i, h]
  │     probs_grad[i, k] = ReduceSum_h(permuted_tokens[j, h] * unpermuted_tokens_grad[i, h])
  │
  └────┬──────────────────────────────┐
       │                              │
  permuted_tokens_grad           probs_grad
  [totalLength, H]               [tokensNum, topK]
```

---

## 7. 性能相关设计

1. **分层次 ReduceSum**: Base 类的 ReduceSumFunc 根据 hiddenSize 选择策略: >=4096 用 BinaryAdd+3级BlockReduce，>=512 用 2 级，>=64 用 1 级，<64 直接 WholeReduce。

2. **Ping-Pong Double Buffer**: ProbNotNull 路径使用 8 个硬件 event 实现 indices/probs/grad 的双缓冲，计算与搬运重叠。

3. **本地数组缓存**: indicesArray/probsArray 最大 512 元素，避免重复从 GM 读取索引和概率。

4. **scatter-add 直写**: ProbNone 路径直接 DataCopyPad 到目标位置，无需额外累加缓冲。

---

## 8. 依赖关系

```
op_host/moe_token_unpermute_grad_def.cpp
└── register/op_def_registry.h

op_host/moe_token_unpermute_grad_tiling.cpp
├── moe_token_unpermute_grad_tiling.h
│   ├── register/op_impl_registry.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/op_api/aclnn_moe_token_unpermute_grad.cpp
├── aclnn_moe_token_unpermute_grad.h
└── aclnn_executor.h

op_kernel/moe_token_unpermute_grad.cpp
├── moe_token_unpermute_grad_base.h
├── moe_token_unpermute_grad_prob_none.h
├── moe_token_unpermute_grad_prob_not_none.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| topK | 必须 <= 512 |
| 维度 | tokensNum, topK, hiddenSize, numOutTokens 均不能为 0 |
| 平台 | 仅 ascend910b/910_93 |
| probs_grad | 总是存在输出，probs 为 None 时形状可能为空 |
| indicesReserveNum | ProbNotNull 路径最大 256 |
| padded_mode | 属性存在但 kernel 逻辑未区分 Key=0/10 的处理差异 |

---

## 10. 与同族变体的核心差异

| 维度 | moe_token_unpermute_grad（本算子） | moe_token_unpermute_with_ep_grad | moe_token_unpermute_with_routing_map_grad |
|------|-----------------------------------|----------------------------------|-------------------------------------------|
| EP range | 无 | 有（start/end） | 无 |
| routing_map | 无 | 无 | 有（BOOL/INT8） |
| 额外输入 | 无 | 无 | out_index, permute_token_id, routing_map |
| kernel 复用 | 自有 | 复用基础版 ProbNone/ProbNotNull 头文件 | 4 种子 kernel（prob_none/not_none × drop_pad_false/true） |
| 混合精度 | 不支持 | 不支持 | 支持（BF16 tokens + FP32 probs, Key=101/111） |
| 平台 | ascend910b/910_93 | ascend910b/910_93 | 含 ascend950 |
| TilingKey 范围 | 0/1/10/11 | 类似基础版 | 更多（含 drop_pad 组合） |
