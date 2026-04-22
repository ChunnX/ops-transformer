# moe_token_unpermute_with_routing_map 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenUnpermuteWithRoutingMap`
**算子类**: MoE Token 反排列，路由矩阵变体
**功能描述**: 通过 `routing_map`（标记每个 token 被哪些 expert 处理）和 `probs`，实现基于路由映射的 token 反排列。将专家处理后的 permuted_tokens 按 sorted_indices 映射回原始 token 顺序，支持与 probs 的加权累加。支持 `drop_and_pad` 模式用于 Expert capacity padding 场景。

**两种计算模式**:
- **非 dropAndPad (TilingKey=0/1)**: 先通过 MaskedSelect 从 routing_map+probs 提取 permute_probs，再执行标准 unpermute 逻辑
- **dropAndPad (TilingKey=1000)**: 按 capacity/expertNum 坐标从 probs 矩阵中取值，结合 Sort + InplaceIndexAdd 完成反排列

**支持平台**: ascend910b, ascend910_93, ascend950, kirinx90, kirin9030
**支持数据类型**: tokens BF16/FP16/FP32, routing_map BOOL/INT8, probs BF16/FP16/FP32（含混合精度）

---

## 2. 目录结构与文件职责

```
moe_token_unpermute_with_routing_map/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_token_unpermute_with_routing_map_def.cpp            # 算子定义
│   ├── moe_token_unpermute_with_routing_map_tiling.h           # Tiling数据结构定义
│   ├── moe_token_unpermute_with_routing_map_tiling.cpp         # Tiling策略实现
│   ├── op_api/
│   │   ├── aclnn_moe_token_unpermute_with_routing_map.h        # ACLNN API头文件
│   │   ├── aclnn_moe_token_unpermute_with_routing_map.cpp      # ACLNN API实现
│   │   ├── moe_token_unpermute_with_routing_map.h              # L0内部接口声明
│   │   └── moe_token_unpermute_with_routing_map.cpp            # L0层实现
│   └── config/ {ascend910_93, ascend910b, ascend950, kirinx90, kirin9030}
├── op_kernel/
│   ├── moe_token_unpermute_with_routing_map.cpp                # Kernel入口（TilingKey分发）
│   ├── masked_select.h                                         # 从routing_map+probs提取有效位
│   ├── moe_sort_base_with_routing_map.h                        # 路由图排序基类
│   ├── moe_token_unpermute_with_routing_map_not_pad.h          # 非Pad Kernel实现
│   └── moe_token_unpermute_with_routing_map_pad.h              # Pad Kernel实现
├── docs/ aclnnMoeTokenUnpermuteWithRoutingMap.md
├── examples/ test_aclnn_moe_token_unpermute_with_routing_map.cpp
└── tests/ut/ {op_host with op_api, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (op_host/moe_token_unpermute_with_routing_map_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | permuted_tokens | REQUIRED | BF16/FP16/FP32 | 2D: [numOutTokens, H] |
| 输入 | sorted_indices | REQUIRED | INT32 | 1D: [totalLength] |
| 输入 | routing_map | OPTIONAL | BOOL/INT8 | 2D: [tokensNum, numExperts] |
| 输入 | probs | OPTIONAL | BF16/FP16/FP32 | 2D: [tokensNum, numExperts] |
| 输出 | unpermuted_tokens | REQUIRED | 与permuted_tokens同类型 | 2D: [tokensNum, H] |
| 输出 | out_index | REQUIRED | INT32 | 1D |
| 输出 | permute_token_id | REQUIRED | INT32 | 1D |
| 输出 | permute_probs | REQUIRED | 与probs同类型 | 1D |

**属性**:

| 属性名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| drop_and_pad | Bool | false | 是否开启容量限制模式 |
| restore_shape | ListInt | false | unpermuted_tokens的shape，size=2 |

**数据类型组合**（def.cpp 第23-66行）: 共 8 种组合。含混合精度（permuted_tokens BF16 + probs FP32）。routing_map 支持 BOOL/INT8 两种类型。kirinx90/kirin9030 仅支持 4 种组合（无 BF16 路由）。

### 3.2 Tiling 策略 (op_host/moe_token_unpermute_with_routing_map_tiling.cpp)

**两条 Tiling 路径**:

非 dropAndPad 模式（`TilingCompute`，第528-542行）:
1. `InputParamCheck()`: 校验 2D tokens、1D indices、topK <= 512
2. `SetCoreNum()`: `usedCoreNum = min(tokensNum, maxCoreNum)`（第274-280行）
3. `TilingHiddenSize()`: 按 UB 容量切分 hidden 维度（第282-293行）
4. `SetBufferNum()`: 从最大 4 递减 buffer_num 直到 hidden tile 可放入 UB（第295-302行）
5. `TilingToken()`: 按 token 维度切分（第311-336行）
6. `Tiling4MaskedSelect()`: 按 expert 维度分配核，计算前/后组核参数（第338-409行）
7. `SetTilingKey()`: Key=0（无 probs）或 Key=1（有 probs）（第417-423行）

dropAndPad 模式（`TilingMoeTokenUnpermuteWithRoutingMapPad`，第513-526行）:
1. `InitPad()`: 从 sorted_indices/probs shape 推导 num_experts、capacity（第446-453行）
2. `InitSplitInfoPad()`: 按 UB 容量计算前/后组核切分、循环次数（第147-172行）
3. TilingKey = 1000

**topK 推导方式**（第235-272行）:
```
// probs存在时:
topK = tokensShape->GetDim(0) / probsShape->GetDim(0)
numExperts = probsShape->GetDim(1)

// probs不存在时:
topK = tokensShape->GetDim(0) / restore_shape
```

### 3.3 Shape 推导

本算子无独立的 infershape 文件。Shape 推导由 L2 层 API 完成:
- `unpermuted_tokens`: `[num_tokens, hidden_size]`，其中 `num_tokens` 来自 `restoreShapeOptional` 或 `probsOptional.GetViewShape().GetDim(0)`
- `permute_token_id`/`out_index`: 与 sorted_indices 同 shape
- `permuteProbs`: 与 sortedIndices 同 shape（probs 存在时），否则 `[0]`

### 3.4 ACLNN API (op_host/op_api/)

采用 CANN 标准两段式接口:

**第一阶段 — 获取 Workspace 大小**:
```cpp
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(
    const aclTensor* permutedTokens,
    const aclTensor* sortedIndices,
    const aclTensor* routingMapOptional,
    const aclTensor* probsOptional,
    bool paddedMode,
    const aclIntArray* restoreShapeOptional,
    aclTensor* unpermutedTokens,
    aclTensor* outIndex,
    aclTensor* permuteTokenId,
    aclTensor* permuteProbs,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

**第二阶段 — 执行计算**:
```cpp
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMap(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, const aclrtStream stream)
```

**L2 层处理流程**（aclnn_moe_token_unpermute_with_routing_map.cpp 第252-439行）:

非 dropAndPad 模式（第420-433行）:
```
CheckParams → Contiguous(inputs) → L0算子 → ViewCopy(outputs)
```

dropAndPad 模式（第318-419行）:
```
CheckParams → Contiguous(inputs) → L0算子(计算permute_probs) →
  Reshape(permuteProbs) → Mul(permutedTokens, permuteProbs) →
  Sort(sortedIndices) → [混合精度时 Cast] →
  InplaceIndexAdd(unpermutedTokens, sortIndex, sortValues) →
  [混合精度时 Cast 回 BF16] →
  ViewCopy(outputs)
```

InplaceIndexAdd 的平台分支（第368-392行）: ascend910b/910_93 对 BF16/FP16 使用 `InplaceIndexAddWithSorted`（更快），其他情况使用 `InplaceIndexAddAiCore`。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (op_kernel/moe_token_unpermute_with_routing_map.cpp)

**入口函数**（第51-74行）: 接收 8 个 GM_ADDR 参数 + workspace + tiling。

**三种路径分发**:
- **TilingKey=1000**: 实例化 `KernelMoeTokenUnpermuteWithRoutingMapPad<DTYPE_PERMUTE_PROBS>`，Pad 模式
- **TilingKey=1**: 先执行 `GENERAL_OP_IMPL()`（MaskedSelect 生成 permute_probs），再执行 `MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_IMPL(T1, int32_t, T3, true)`
- **TilingKey=0**: 直接执行 `MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_IMPL(T1, int32_t, T3, false)`

仅 AIV 核执行（第54-56行: `g_coreType == AIC` 时返回）。

### 4.2 核心计算流程

**非 dropAndPad 模式** — 两阶段:

阶段 1 — MaskedSelect（仅 TilingKey=1 时执行）:
```
routing_map [T, E] + probs [T, E] → MaskedSelect → permute_probs [有效位数]
SyncAll() 屏障同步
```

阶段 2 — Unpermute（`KernelMoeTokenUnpermuteWithRoutingMap`，not_pad.h）:
```
for each output_token in [start_token, end_token):
  for each hidden_split:
    idx0 = sortedIndices[first_token_in_group]
    CopyTokenIn(idx0 * H + h_part) → CalFirstToken(prob0, h_part)
    for remaining tokens in topK group:
      CopyTokenIn(idxN * H + h_part) → CalToken(probN, h_part) → 累加
    CopyOut → 写回 unpermuted_tokens[token_idx * H + h_part]
```

**dropAndPad 模式**（`KernelMoeTokenUnpermuteWithRoutingMapPad`，pad.h）:
```
for posIdx in [blockOffset, blockOffset + loopN):
  tokenIdx = sortedIndices[posIdx]
  expertIdx = posIdx / capacity
  probsVal = probs[tokenIdx * num_experts + expertIdx]
  DataCopyPad → 写回 unpermutedTokens
```

语义: 从 probs 矩阵中按 `[sortedIndices[i], i/capacity]` 坐标取值。

### 4.3 数据结构与内存布局

**NotPad Kernel 队列与缓冲区**（not_pad.h 第39-47行）:

| 成员 | 类型 | 用途 |
|------|------|------|
| `tokens_inque` | `TQue<VECIN, 1>` | token 数据输入队列 |
| `indices_inque` | `TQue<VECIN, 1>` | sortedIndices 输入队列 |
| `probs_inque` | `TQue<VECIN, 1>` | probs 输入队列（仅 PROBS=true） |
| `outque` | `TQue<VECOUT, 1>` | 输出队列 |
| `temp_buffer0/1` | `TBuf<VECCALC>` | Cast 缓冲（非 float 类型时） |
| `temp_buffer2` | `TBuf<VECCALC>` | probs Cast 缓冲（PROBS=true 且非 float） |

UB 分配（not_pad.h 第135-153行）:
- `tokens_inque`: `buffer_num * hidden_splited_length_align512 * sizeof(T1)`
- `indices_inque`: `block_splited_length * sizeof(int32_t)`
- `outque`: `hidden_splited_length_align512 * sizeof(T1)`
- `temp_buffer0/1`: 仅 T1 非 float 时分配（用于 Cast 到 float 累加）

**Pad Kernel**（pad.h）: 使用 `TQue<VECOUT, BUFFER_NUM> unpermutedTokensOutQue`，每核按 `num_tokens_each_loop_current_core * sizeof(T)` 分配。使用标量 `GetValue` 读 probs，`DataCopyPad` 写输出。

### 4.4 多核并行策略

**非 dropAndPad 模式**:
- Unpermute: 按 tokenNum 均匀分配核，`usedCoreNum = min(tokensNum, maxCoreNum)`，尾部核多处理 1 个 token（not_pad.h 第94-97行）
- MaskedSelect: 按 expert 维度分配核，`numBlocks = min(numExperts, aivUseNum)`，每核处理 `expertSlice * tokensNum` 个元素

**dropAndPad 模式**:
- 按 `rowVal = num_experts * capacity` 分配核，前组核和后组核处理数量不同（tiling.cpp 第147-172行的 front/tail 切分）

### 4.5 通信模式

非 dropAndPad 模式（TilingKey=1）需要核间同步:
- **MaskedSelect 后 SyncAll()**: 确保 MaskedSelect 写入的 permute_probs 在所有核可见后，才开始 Unpermute 阶段读取

MaskedSelect 内部也使用两阶段同步:
- 阶段 1 计算每个核的 offset 后 `SyncAll()`（masked_select.h 第261行）
- 阶段 2 各核读取所有 offset 计算前缀和后执行数据选择

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

**MoeTokenUnpermuteWithRoutingMapTilingData**（tiling.h 第78-97行）:

```
MoeTokenUnpermuteWithRoutingMapTilingData
├── hidden_size            : int64_t   # hidden维度
├── top_k                  : int64_t   # 每token被分配的expert数
├── num_out_tokens         : int64_t   # 输出token总数
├── hidden_splited_length  : int64_t   # hidden切分块大小
├── hidden_splited_num     : int64_t   # hidden切分块数
├── hidden_splited_remain  : int64_t   # hidden切分余数
├── tokens_core_length     : int64_t   # 每核token数
├── tokens_core_remain     : int64_t   # token余数
├── tokens_splited_length  : int64_t   # token切分块大小
├── tokens_splited_num     : int64_t   # token切分块数
├── tokens_splited_remain  : int64_t   # token切分余数
├── used_core_num          : int64_t   # 实际使用核数
├── buffer_num             : int64_t   # 双缓冲buffer数
├── maskedSelectParamsOp                    # MaskedSelect参数（嵌套）
│   └── MaskedSelectMTUTilingData
│       ├── formerNum/Length, formertileNum/Length/lasttileLength
│       ├── tailNum/Length, tailtileNum/Length/lasttileLength
│       ├── tokenNum           : uint64_t
│       └── needCoreNum        : int64_t
└── moeTokenUnpermuteWithRoutingMapPadTilingData  # Pad模式参数（嵌套）
    ├── core_num_use          : uint64_t   # 使用核数
    ├── num_tokens            : uint64_t   # token数
    ├── num_experts           : uint64_t   # 专家数
    ├── capacity              : uint64_t   # 每专家容量
    ├── front_core            : uint64_t   # 前组核数
    ├── loop_time_each_front_core           : uint64_t
    ├── num_tokens_each_front_core          : uint64_t
    ├── num_tokens_front_core_each_loop     : uint64_t
    ├── num_tokens_front_core_last_loop     : uint64_t
    ├── tail_core             : uint64_t   # 后组核数
    ├── loop_time_each_tail_core            : uint64_t
    ├── num_tokens_each_tail_core           : uint64_t
    ├── num_tokens_tail_core_each_loop      : uint64_t
    └── num_tokens_tail_core_last_loop      : uint64_t
```

**CompileInfo**（tiling.h 第100-106行）: 保存 `aivnum`、`ubSize`、`workSpaceSIZE`，用于动态编译静态图。

### 5.2 Workspace 布局

| 模式 | Workspace 大小 | 说明 |
|------|---------------|------|
| 非 dropAndPad | 32 MB（`16 * 1024 * 1024 * 2`） | 用于 MaskedSelect 中间缓冲 + 核间 offset 通信 |
| dropAndPad | 16 MB（`16 * 1024 * 1024`） | 用于 Pad 模式中间缓冲 |

非 dropAndPad 模式下 Workspace 用于 MaskedSelect 的两阶段处理: 阶段 1 各核将 offset 写入 workspace，阶段 2 读取所有 offset 计算前缀和。

---

## 6. 完整数据流

### 6.1 非 dropAndPad 模式 (TilingKey=1)

```
  routing_map [T, E]    probs [T, E]
        |                    |
        +---> MaskedSelect <--+
        |   (核间SyncAll)    |
        |                    |
        v                    |
  permute_probs [有效位数]    |
        |                    |
        +---------+----------+
                  |
  permuted_tokens [numOut, H]    sortedIndices [totalLength]
        |                              |
        +----> KernelMoeTokenUnpermute <+
               WithRoutingMap (not_pad)
               per token_group:
                 idx = sortedIndices[group_start + k]
                 CopyIn(idx * H + h_part)
                 Cast→float → Muls(prob_k) → Add累加
                 → Cast回原类型 → CopyOut
                  |
          +-------+-------+----------+
          |       |       |          |
  unpermuted   out_index  permute   permute
  _tokens      [total]   _token_id  _probs
  [tokensNum,H]          [total]    [total]
```

### 6.2 非 dropAndPad 模式 — 无 probs (TilingKey=0)

```
  permuted_tokens [numOut, H]    sortedIndices [totalLength]
        |                              |
        +----> KernelMoeTokenUnpermute <+
               WithRoutingMap (not_pad)
               (haveProbs=false)
               per token_group:
                 idx = sortedIndices[group_start + k]
                 CopyIn → Cast→float → Add累加 → Cast回 → CopyOut
                  |
           unpermuted_tokens [tokensNum, H]
           out_index, permute_token_id
```

### 6.3 dropAndPad 模式 (TilingKey=1000)

```
  sortedIndices [E*capacity]    probs [T, E]
        |                            |
        +----> KernelMoeTokenUnpermuteWithRoutingMapPad
              for posIdx in [blockOffset, blockOffset + loopN):
                tokenIdx = sortedIndices[posIdx]
                expertIdx = posIdx / capacity
                val = probs[tokenIdx * E + expertIdx]
                DataCopyPad → unpermutedTokens
                  |
           unpermutedTokens [E*capacity]
```

L2 层补充处理:
```
  permutedTokens * permuteProbs → weightedTokens
  Sort(sortedIndices) → (sortValues, sortIndex)
  InplaceIndexAdd(unpermutedTokens, sortIndex, weightedTokens[sortValues])
```

---

## 7. 性能相关设计

1. **MaskedSelect 两阶段设计**: 阶段 1 各核独立计算 offset 并写入 workspace，SyncAll 后阶段 2 各核读取所有 offset 计算全局前缀和。避免跨核实时通信，用 workspace 换取并行度。

2. **float 精度累加**: NotPad Kernel 中所有累加在 float 空间进行（not_pad.h 第285-327行），即使输入为 BF16/FP16 也先 Cast 到 float 累加，最后 Cast 回原类型，减少精度损失。

3. **UB 空间动态分配**: `SetBufferNum()`（tiling.cpp 第295-302行）从最大 4 递减 buffer_num 直到 hidden tile 可放入 UB。`buffer_num` 影响双缓冲效果 — buffer_num 越大流水线重叠越好，但单个 buffer 越小。

4. **bisheng 并行优化**: NotPad Kernel 中 `CalMultiOutToken` 内部循环使用 `#pragma bisheng auto_sync parallel`（not_pad.h 第199行），允许编译器自动并行化 token 处理循环。

5. **Pad 模式标量读取**: Pad Kernel 使用 `GetValue` 标量读 probs（pad.h 第117行），因为读取位置 `[sortedIndices[i], i/capacity]` 是离散的 scatter 读，无法形成连续 DMA。

6. **混合精度优化**: L2 层检测到 BF16 tokens + FP32 probs 时（aclnn_moe_token_unpermute_with_routing_map.cpp 第273-279行），在累加前 Cast 到 FP32，累加完成后再 Cast 回 BF16。

---

## 8. 依赖关系

```
op_host/moe_token_unpermute_with_routing_map_tiling.cpp
└── moe_token_unpermute_with_routing_map_tiling.h
    ├── register/tilingdata_base.h              # Tiling 数据基类
    ├── register/op_impl_registry.h             # 算子实现注册
    ├── log/log.h                               # 日志宏
    ├── tiling_base/tiling_base.h               # TilingBaseClass 基类
    ├── tiling_base/tiling_templates_registry.h # Tiling 模板注册
    └── util/math_util.h                        # 数学工具

op_host/op_api/aclnn_moe_token_unpermute_with_routing_map.cpp
├── aclnn_moe_token_unpermute_with_routing_map.h
│   ├── aclnn/aclnn_base.h
│   └── aclnn_util.h
├── moe_token_unpermute_with_routing_map.h      # L0 内部接口
├── opdev/common_types.h, data_type_utils.h, format_utils.h
├── opdev/make_op_executor.h, op_dfx.h, op_log.h
├── opdev/shape_utils.h, platform.h
├── aclnn_kernels/contiguous.h, reshape.h, transpose.h, cast.h
├── aclnn_kernels/common/op_error_check.h
├── level0/sort.h, zero_op.h, mul.h
└── [条件编译] inplace_index_add_with_sorted.h, inplace_index_add.h

op_kernel/moe_token_unpermute_with_routing_map.cpp
├── moe_token_unpermute_with_routing_map_pad.h
│   └── kernel_operator.h
├── moe_token_unpermute_with_routing_map_not_pad.h
│   └── kernel_operator.h
├── masked_select.h
│   ├── kernel_tiling/kernel_tiling.h
│   ├── kernel_operator.h
│   └── moe_sort_base_with_routing_map.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| topK 上限 | topK <= 512（tiling.cpp 第214-216行） |
| probs 必须为 2D | shape 校验（tiling.cpp 第218-220行） |
| dropAndPad 时 capacity | capacity = sortedIndices.dim(0) / num_experts，需整除 |
| capacity > tokensNum | Pad 模式下会有 warning（tiling.cpp 第516-518行） |
| Pad 路径 workspace | 需要 userWorkspace 不为空（kernel 第59-61行） |
| 混合精度 | probs 可为 FP32 而 tokens 为 BF16 |
| AIC 核跳过 | 仅 AIV 核执行（kernel 第54-56行） |
| routing_map 类型 | 仅支持 BOOL 和 INT8，INT8 取值需为 0/1 |

---

## 10. 与同族变体的核心差异

### 10.1 vs moe_token_unpermute（基础版）

| 维度 | moe_token_unpermute | moe_token_unpermute_with_routing_map |
|------|---------------------|--------------------------------------|
| 额外输入 | 无 | routing_map (BOOL/INT8) |
| 额外输出 | 无 | out_index, permute_token_id, permute_probs |
| MaskedSelect | 无 | 有（从 routing_map+probs 提取有效位） |
| drop_and_pad | 不支持 | 支持（Pad Kernel + L2 InplaceIndexAdd） |
| 核间同步 | 不需要 | MaskedSelect 后需 SyncAll |
| probs shape | [T, topK] | [T, numExperts] |
| topK 推导 | probs.dim(1) | tokens.dim(0) / probs.dim(0) |
| 工作空间 | LibApiWorkSpaceSize | 32 MB（非 Pad）/ 16 MB（Pad） |
| 平台支持 | 910b/910_93 | 910b/910_93/950/kirinx90/kirin9030 |

### 10.2 vs moe_token_unpermute_with_ep

| 维度 | moe_token_unpermute_with_ep | moe_token_unpermute_with_routing_map |
|------|----------------------------|--------------------------------------|
| range 支持 | 有 (start/end) | 无 |
| routing_map | 无 | 有 (BOOL/INT8) |
| drop_and_pad | 不支持 | 支持 |
| ISTOKEN 模板参数 | 有 | 无 |
| MaskedSelect | 无 | 有 |
| capacity 概念 | 无 | 有 |
| L2 层复杂度 | 直接调用 L0 | Sort + InplaceIndexAdd + 混合精度 Cast |
