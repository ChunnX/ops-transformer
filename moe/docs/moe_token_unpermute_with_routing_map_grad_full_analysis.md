# moe_token_unpermute_with_routing_map_grad 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenUnpermuteWithRoutingMapGrad`
**算子类**: MoE Token 反排列反向算子，路由矩阵变体
**功能描述**: `moe_token_unpermute_with_routing_map` 的反向算子。将 unpermuted_tokens_grad 映射回 permuted 顺序的梯度，同时计算 probs 的梯度。使用正向算子输出的 out_index 和 permute_token_id 作为反向索引，结合 routing_map 计算 probs_grad。

**四种 Kernel 组合**:
- **ProbNone + PadFalse (TilingKey=0)**: 纯 scatter，无 probs 参与
- **ProbNone + PadTrue (TilingKey=10)**: 纯 scatter，Pad 模式索引
- **ProbNotNone + PadFalse (TilingKey=1/101)**: 计算 permuted_tokens_grad 和 probs_grad，支持混合精度
- **ProbNotNone + PadTrue (TilingKey=11/111)**: Pad 模式下的双重梯度计算

**支持平台**: ascend910b, ascend910_93, ascend950（APT 变体）
**支持数据类型**: BF16/FP16/FP32（含 BF16 tokens + FP32 probs 混合精度）

---

## 2. 目录结构与文件职责

```
moe_token_unpermute_with_routing_map_grad/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_token_unpermute_with_routing_map_grad_def.cpp            # 算子定义
│   ├── moe_token_unpermute_with_routing_map_grad_tiling.h           # Tiling数据结构定义
│   ├── moe_token_unpermute_with_routing_map_grad_tiling.cpp         # Tiling策略实现
│   └── config/ {ascend910_93, ascend910b}
├── op_kernel/
│   ├── moe_token_unpermute_with_routing_map_grad.cpp                # 910b/910_93 Kernel入口
│   ├── moe_token_unpermute_with_routing_map_grad_apt.cpp            # 950 APT Kernel入口
│   ├── moe_token_unpermute_with_routing_map_grad_base.h             # 基础Kernel类
│   ├── moe_token_unpermute_with_routing_map_grad_prob_none_drop_pad_false.h
│   ├── moe_token_unpermute_with_routing_map_grad_prob_none_drop_pad_true.h
│   ├── moe_token_unpermute_with_routing_map_grad_prob_not_none_drop_pad_false.h
│   ├── moe_token_unpermute_with_routing_map_grad_prob_not_none_drop_pad_true.h
│   └── arch35/ {base.h, prob_none/not_none_drop_pad_false/true.h}  # ascend950专用
├── docs/ aclnnMoeTokenUnpermuteWithRoutingMapGrad.md
├── examples/ test_aclnn_moe_token_unpermute_with_routing_map_grad.cpp
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (op_host/moe_token_unpermute_with_routing_map_grad_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | unpermuted_tokens_grad | REQUIRED | BF16/FP16/FP32 | 2D: [tokensNum, H] |
| 输入 | out_index | REQUIRED | INT32 | 1D: [numOutTokens] |
| 输入 | permute_token_id | REQUIRED | INT32 | 1D: [numOutTokens] |
| 输入 | routing_map | OPTIONAL | BOOL/INT8 | 2D: [tokensNum, numExperts] |
| 输入 | permuted_tokens | OPTIONAL | 同 tokens 类型 | 2D: [numOutTokens, H] |
| 输入 | probs | OPTIONAL | BF16/FP16/FP32 | 2D: [tokensNum, numExperts] |
| 输出 | permuted_tokens_grad | REQUIRED | 同 tokens 类型 | 2D: [numOutTokens, H] |
| 输出 | probs_grad | OPTIONAL | 同 probs 类型 | 2D: [tokensNum, numExperts] |

**属性**:

| 属性名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| drop_and_pad | Bool | false | 是否开启容量限制模式 |
| restore_shape | ListInt | false | 恢复形状 |

**数据类型组合**（def.cpp 第25-76行）: 共 8 种组合，含混合精度（BF16 tokens + FP32 probs）。routing_map 支持 BOOL/INT8。

**与正向的输入对应关系**:

| 正向输出 | 反向输入 | 用途 |
|---------|---------|------|
| unpermuted_tokens | unpermuted_tokens_grad | 前向输出的梯度 |
| out_index | out_index | Sort 的排序索引 |
| permute_token_id | permute_token_id | token ID 映射 |
| — | routing_map | 从原始输入传递，用于 probs_grad |
| — | permuted_tokens | 从原始输入传递，用于 probs_grad |
| — | probs | 从原始输入传递 |

### 3.2 Tiling 策略 (op_host/moe_token_unpermute_with_routing_map_grad_tiling.cpp)

**三种 Tiling 路径**:

`TilingForProbIsNone()`（第147-188行）:
- 无 probs，纯 scatter 操作
- 按 numOutTokens 均匀分配核
- UB 计算: `hiddenSizeAlign = min(hiddenSize 对齐, totalUbSize / (2 * typeLen))`

`TilingForProbNotNonePadTrue()`（第190-246行）:
- 有 probs + Pad 模式，每行只有一个 prob 值
- 按 numOutTokens 分配核
- 约束: `capacity <= tokensNum`（第202-208行）
- UB 计算: `(totalUbSize - 32 - 512*4) / (2 * (8 + typeLen))` 再乘以 512
- `hiddenSizeLoopTimesAlign = AlignUp(loopTimes, 8)` — 对齐到 FP32 BlockReduceSum 的粒度

`TilingForProbNotNonePadFalse()`（第248-319行）:
- 有 probs + 非 Pad 模式，每行有 topK 个 prob 值
- 按 tokensNum 分配核（不是按 numOutTokens）
- `rowIdMapEachCore = tokenNumEachCore * topK`
- 约束: `topK <= 512`（第257-261行）且 `topK <= numExpert`（第263-269行）
- 复杂 UB 分配: 预留 indices 空间后，剩余空间按 hiddenSize 和 numExpert 分配
- `indicesReserveNum` = min(256, maxPerCore) 向下取 topK 的倍数

**TilingKey 编码**（第465-474行）: `mixKey * 100 + paddedModeKey * 10 + probKey`

| Key | probs | paddedMode | 混合精度 |
|-----|-------|------------|---------|
| 0 | none | false | — |
| 1 | not-none | false | no |
| 10 | none | true | — |
| 11 | not-none | true | no |
| 101 | not-none | false | yes |
| 111 | not-none | true | yes |

### 3.3 Shape 推导

本算子无独立的 infershape 文件。Shape 推导由 L2 层 API 完成:
- `permuted_tokens_grad`: `[numOutTokens, hiddenSize]`，与 `unpermuted_tokens_grad` 的 dim(1) 一致
- `probs_grad`: `[tokensNum, numExperts]`，与 `probs` 同 shape

### 3.4 ACLNN API

本算子无独立的 `op_api/` 目录，ACLNN API 通过构建系统自动生成。

**两段式接口签名**（来自 docs）:

```cpp
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize(
    const aclTensor*   unpermutedTokensGrad,
    const aclTensor*   outIndex,
    const aclTensor*   permuteTokenId,
    const aclTensor*   routingMapOptional,
    const aclTensor*   permutedTokensOptional,
    const aclTensor*   probsOptional,
    bool               dropAndPad,
    const aclIntArray* restoreShapeOptional,
    const aclTensor*   permutedTokensGradOut,
    const aclTensor*   probsGradOutOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)

aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGrad(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream)
```

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (op_kernel/moe_token_unpermute_with_routing_map_grad.cpp)

**入口函数**（第23-26行）: 接收 8 个 GM_ADDR 参数 + workspace + tiling。

**六种路径分发**（第29-70行）:

| TilingKey | 实例化的类 | 模板参数 |
|-----------|-----------|---------|
| 0 | `ProbNoneDropPadFalse` | `<DTYPE, int32_t>` |
| 10 | `ProbNoneDropPadTrue` | `<DTYPE, int32_t>` |
| 1 | `ProbNotNoneDropPadFalse` | `<DTYPE, int32_t, DTYPE>` |
| 11 | `ProbNotNoneDropPadTrue` | `<DTYPE, int32_t, DTYPE>` |
| 101 | `ProbNotNoneDropPadFalse` | `<DTYPE, int32_t, float>` |
| 111 | `ProbNotNoneDropPadTrue` | `<DTYPE, int32_t, float>` |

950 平台使用 `_apt.cpp`（arch35 子目录的相同实现）。

### 4.2 核心计算流程

**ProbNone + PadFalse**（最简单，纯 scatter）:
```
for each outIndex in core's range:
  permuteTokenId = out_index[rowIdMapOffset]
  tokenId = rowIdMapOffset / topK
  for each hiddenChunk:
    DataCopyPad(in, unpermutedGrad[tokenId * H + h_part])
    DataCopyPad(out, permutedGrad[permuteTokenId * H + h_part])
```

**ProbNone + PadTrue**: 与 PadFalse 类似，但 `tokenId` 来自 `permute_token_id`（`sortedTwiceIndicesGm`），而非 topK 除法推导。

**ProbNotNone + PadTrue**（单 prob/行）:
```
for each outIndex in core's range:
  tokenId = sortedTwiceIndicesGm[...]
  permuteTokenId = sortedTwiceIndexGm[...]
  prob = probGm.GetValue(probOffset)  → Cast to float
  for each hiddenChunk:
    Mul(permutedTokensFp32, permutedTokensFp32, unpermutedGradFp32)  → ReduceSum → probGrad 累加
    Muls(permutedTokensFp32, unpermutedGradFp32, prob)               → permutedTokensGrad
  ReduceSum(probGradReduceSumLocal) → 单个 probGrad 值写回 GM
```

**ProbNotNone + PadFalse**（最复杂，4 层循环）:
```
for each indicesBatch:
  读取 indicesReserveNum 个 sorted indices 到 indicesArray[]
  读取对应 probs 到 probsArray[]（通过 routing_map 筛选）
  for each token (topK per token):
    for each hiddenChunk:
      for each topK in indicesBatch:
        Muls(grad, unpermutedGrad, probsArray[k])        → permutedTokensGrad
        Mul(tmp, permutedTokens, unpermutedGrad)          → ReduceSum → probGrad 累加
    Add 跨 hiddenChunk 累加 probGrad
  写回 probs_grad 到 GM（scatter）
```

### 4.3 数据结构与内存布局

**Base 类成员**（base.h 第48-89行）:

| 成员 | 类型 | 用途 |
|------|------|------|
| `unpermutedTokensGradGm` | `GlobalTensor<PermutedTokenT>` | 梯度输入 |
| `sortedTwiceIndexGm` | `GlobalTensor<IdxT>` | out_index |
| `sortedTwiceIndicesGm` | `GlobalTensor<IdxT>` | permute_token_id |
| `routingMapGm` | `GlobalTensor<int8_t>` | routing_map |
| `permutedTokensGm` | `GlobalTensor<PermutedTokenT>` | 正向输入 |
| `probGm` | `GlobalTensor<ProbsT>` | probs |
| `permutedTokensGradGm` | `GlobalTensor<PermutedTokenT>` | 梯度输出 |
| `probGradGm` | `GlobalTensor<ProbsT>` | prob 梯度输出 |

**ProbNotNone PadFalse 的手动双缓冲**（prob_not_none_drop_pad_false.h 第65-72行）:
- `indicesArray[512]`: 缓存的 sorted indices
- `probsArray[512]`: 缓存的 prob 值
- `probsAddrArray[512]`: 缓存的 prob GM 地址
- EVENT_ID 0,1: V<->MTE3（permuted_tokens_grad 写出）
- EVENT_ID 4,5: V<->MTE2（unpermuted_grad 读入）
- EVENT_ID 6,7: V<->MTE2（permuted_tokens 读入）

**ProbNotNone PadTrue 的队列缓冲**（prob_not_none_drop_pad_true.h 第35-39行）:

| 缓冲 | 类型 | 大小 |
|------|------|------|
| `unpermutedGradTQue` | `TQue<VECIN, 2>` | `hiddenSizeAlign * 4` (FP32) |
| `permutedTokensTQue` | `TQue<VECIN, 2>` | `hiddenSizeAlign * 4` (FP32) |
| `permutedTokensGradTQue` | `TQue<VECOUT, 2>` | `hiddenSizeAlign * inputTypeSize` |
| `probGradReduceSumTBuf` | `TBuf<VECCALC>` | `hiddenSizeLoopTimesAlign * 4` |
| `probGradOutTBuf` | `TBuf<VECCALC>` | 32 字节 |

### 4.4 多核并行策略

- **ProbNone 两种模式**: 按 numOutTokens 均匀分配核，前组核多处理 1 个元素
- **ProbNotNone PadTrue**: 按 numOutTokens 分配核
- **ProbNotNone PadFalse**: 按 tokensNum 分配核（每核处理 `tokenNumEachCore * topK` 个 indices）

### 4.5 通信模式

**ProbNotNone PadTrue 初始化同步**: `Init()` 中将所有 probGrad 输出初始化为零后调用 `SyncAll()`（prob_not_none_drop_pad_true.h 第68行），确保所有核的零初始化可见后才开始计算。

**7 种同步辅助函数**（base.h 第123-170行）: `SToMTE2Sync`, `VToMTE3Sync`, `SToMTE3Sync`, `MTE3ToSSync`, `MTE3ToVSync`, `VToSSync`, `MTE2ToVSync` — 用于 ProbNotNone PadFalse 模式下的硬件事件显式同步，控制标量/向量/DMA 引擎间的依赖。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

**MoeTokenUnpermuteWithRoutingMapGradTilingData**（tiling.h 第21-46行）:

```
MoeTokenUnpermuteWithRoutingMapGradTilingData
├── tokensNum                : int64_t   # token 数量
├── topK                     : int64_t   # 每 token 分配的 expert 数
├── capacity                 : int64_t   # 每 expert 容量（Pad 模式）
├── numExpert                : int64_t   # 专家数量
├── hiddenSize               : int64_t   # hidden 维度
├── numOutTokens             : int64_t   # 输出 token 数
├── formerCoreNum            : int64_t   # 前组核数（多处理 1 个）
├── tailCoreNum              : int64_t   # 后组核数
├── tokenNumEachCore         : int64_t   # 前组核每核 token 数
├── tokenNumTailCore         : int64_t   # 后组核每核 token 数
├── rowIdMapEachCore         : int64_t   # 前组核每核 indices 数
├── rowIdMapTailCore         : int64_t   # 后组核每核 indices 数
├── hiddenSizeAlign          : int64_t   # hidden 对齐粒度
├── hiddenSizeLoopTimes      : int64_t   # hidden 切分循环次数
├── hiddenSizeLoopTimesAlign : int64_t   # 对齐循环次数（PadTrue 用）
├── hiddenSizeTail           : int64_t   # hidden 尾块大小
├── inputReserveNum          : int64_t   # 每次 GM-UB 搬运 token 数
├── indicesReserveNum        : int64_t   # 每次 GM-UB 搬运 indices 数
├── indicesReserveNumAlign   : int64_t   # 对齐后的 indices 保留数
├── numExpertAlign           : int64_t   # numExpert 对齐到 32 的值
└── totalUbSize              : int64_t   # 总 UB 容量
```

**CompileInfo**: 空结构，无额外编译信息。

### 5.2 Workspace 布局

仅使用系统 LibApi 工作空间（tiling.cpp 第480-482行）:
```
sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize()
currentWorkspace[0] = sysWorkspaceSize
```

无额外算子级工作空间。所有计算均通过 UB 缓冲区完成。

---

## 6. 完整数据流

### 6.1 ProbNone + PadFalse (TilingKey=0)

```
  unpermuted_tokens_grad [T, H]    out_index [numOut]
         |                              |
         +---> ProbNoneDropPadFalse <---+
               per outIndex:
                 permuteId = out_index[offset]
                 tokenId = offset / topK
                 CopyIn(grad[tokenId, h]) → CopyOut(permutedGrad[permuteId, h])
                  |
          permuted_tokens_grad [numOut, H]
```

### 6.2 ProbNone + PadTrue (TilingKey=10)

```
  unpermuted_tokens_grad [T, H]    out_index [numOut]    permute_token_id [numOut]
         |                              |                        |
         +---> ProbNoneDropPadTrue <----+------------------------+
               per outIndex:
                 permuteId = out_index[offset]
                 tokenId = permute_token_id[offset]
                 CopyIn(grad[tokenId, h]) → CopyOut(permutedGrad[permuteId, h])
                  |
          permuted_tokens_grad [numOut, H]
```

### 6.3 ProbNotNone + PadTrue (TilingKey=11/111)

```
  unpermutedGrad [T, H]   out_index   permute_token_id   probs [T, E]
         |                    |              |                |
         +-----> ProbNotNoneDropPadTrue <----+----------------+
                 per outIndex:
                   tokenId = permute_token_id[...]
                   permuteId = out_index[...]
                   prob = probs[tokenId*E + permuteId/capacity]
                   SyncAll (初始化probGrad为零)
                   per hiddenChunk:
                     Mul(permTokensFp32, permTokensFp32, gradFp32) → ReduceSum → probGrad
                     Muls(permTokensFp32, gradFp32, prob) → permutedTokensGrad
                  |
         +--------+--------+
         |                   |
  permuted_tokens_grad   probs_grad
  [numOut, H]            [T, E]
```

### 6.4 ProbNotNone + PadFalse (TilingKey=1/101)

```
  unpermutedGrad [T,H]  out_index  permute_token_id  routing_map [T,E]
  probs [T,E]           permuted_tokens [numOut,H]
         |
         v
  ProbNotNoneDropPadFalse (4层循环)
         |
  for indicesBatch (indicesReserveNum per batch):
    cache indices[] + probs[] via routing_map
    for each token:
      for each hiddenChunk:
        Muls(grad, unpermutedGrad, prob[k]) → permutedTokensGrad[idx[k]]
        Mul(tmp, permutedTokens, unpermutedGrad) → ReduceSum → probGrad[token, expert]
      scatter probGrad → GM
         |
    +----+----+
    |         |
  permuted   probs_grad
  _tokens    [T, E]
  _grad
  [numOut,H]
```

---

## 7. 性能相关设计

1. **显式硬件事件同步**: ProbNotNone PadFalse 使用 6 个 EVENT_ID 进行手动 ping-pong 双缓冲（prob_not_none_drop_pad_false.h 第65-69行），控制 V/MTE2/MTE3 引擎间的依赖，避免隐式等待。

2. **批量 indices 缓存**: PadFalse 模式将最多 512 个 indices + probs 预取到标量数组 `indicesArray[]/probsArray[]`（第70-72行），减少 GM 读取次数。内循环直接从数组取值，无需反复访问 GM。

3. **FP32 精度累加**: 所有 Mul/ReduceSum 在 float 空间进行，BF16/FP16 先 Cast 到 float 再计算，减少精度损失。

4. **多层次 ReduceSum**: `ReduceSumFunc`（base.h 第188-212行）自适应选择二分归约、BlockReduceSum 或 WholeReduceSum，根据数据量选择最高效的策略。

5. **hiddenSizeLoopTimesAlign 对齐**: PadTrue 模式将 hidden 切分循环次数对齐到 FP32_BLOCK_NUM=8（tiling.cpp 第238行），确保 `BlockReduceSum` 高效执行。

6. **PadTrue 初始化零值 + SyncAll**: 将 probGrad 输出全部初始化为零后 SyncAll，后续计算只需写入非零值，避免读-改-写竞争。

---

## 8. 依赖关系

```
op_host/moe_token_unpermute_with_routing_map_grad_tiling.cpp
└── moe_token_unpermute_with_routing_map_grad_tiling.h
    ├── register/tilingdata_base.h
    ├── register/op_impl_registry.h
    ├── log/log.h
    ├── tiling_base/tiling_base.h
    └── tiling_base/tiling_templates_registry.h

op_kernel/moe_token_unpermute_with_routing_map_grad.cpp
├── moe_token_unpermute_with_routing_map_grad_base.h
│   └── kernel_operator.h
├── moe_token_unpermute_with_routing_map_grad_prob_none_drop_pad_false.h
│   └── ..._base.h
├── moe_token_unpermute_with_routing_map_grad_prob_none_drop_pad_true.h
│   └── ..._base.h
├── moe_token_unpermute_with_routing_map_grad_prob_not_none_drop_pad_false.h
│   └── ..._base.h
├── moe_token_unpermute_with_routing_map_grad_prob_not_none_drop_pad_true.h
│   └── ..._base.h
└── kernel_operator.h

op_kernel/arch35/ (ascend950 APT 变体)
├── base.h
├── prob_none_drop_pad_false.h
├── prob_none_drop_pad_true.h
├── prob_not_none_drop_pad_false.h
└── prob_not_none_drop_pad_true.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| topK 上限 | topK <= 512（tiling.cpp 第257-261行） |
| topK <= numExpert | ProbNotNone PadFalse 模式约束（第263-269行） |
| capacity <= tokensNum | ProbNotNone PadTrue 模式约束（第202-208行） |
| out_index 维度匹配 | out_index.dim(0) == permute_token_id.dim(0)（第359-366行） |
| 梯度必须为 2D | unpermuted_tokens_grad 必须 2D（第327-333行） |
| 索引必须为 1D | out_index 和 permute_token_id 必须 1D（第335-354行） |
| 混合精度 | prob 类型要么与 token 一致，要么 tokens=BF16 + probs=FP32（第420-421行） |
| routing_map 类型 | 仅支持 BOOL 和 INT8 |
| APT 变体 | ascend950 使用 `_apt.cpp` + arch35/ 子目录的 kernel |

---

## 10. 与同族变体的核心差异

### 10.1 vs moe_token_unpermute_grad（基础版）

| 维度 | moe_token_unpermute_grad | moe_token_unpermute_with_routing_map_grad |
|------|-------------------------|------------------------------------------|
| 输入数量 | 4 个 | 6 个（含 routing_map, out_index, permute_token_id） |
| routing_map | 无 | 有 (BOOL/INT8) |
| out_index / permute_token_id | 无 | 有（正向的 Sort 输出） |
| drop_and_pad | 不支持 | 支持（4 种 Kernel 组合） |
| 混合精度 | 不支持 | 支持（BF16 tokens + FP32 probs） |
| 子 Kernel 数 | 2 个 | 4 个（+混精共 6 个 TilingKey） |
| capacity / numExpert | 无 | 有 |
| 显式硬件事件同步 | 无 | 7 种 Sync 辅助函数 |
| workspace | 16 MB | 仅 LibApiWorkSpaceSize |

### 10.2 vs moe_token_unpermute_with_ep_grad

| 维度 | with_ep_grad | with_routing_map_grad |
|------|-------------|----------------------|
| range 支持 | 有 (start/end) | 无 |
| routing_map | 无 | 有 |
| drop_and_pad | 不支持 | 支持 |
| 混合精度 | 不支持 | 支持 |
| 子 Kernel | 2 个（复用基础版） | 4+ 个（独立实现） |
| arch35 优化 | 无 | 有（ascend950 APT 变体） |
