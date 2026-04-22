# moe_token_permute_with_routing_map 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenPermuteWithRoutingMap`
**算子类**: MoE Token 重排，路由矩阵变体
**功能描述**: 与基础版本使用 expert ID 索引不同，本算子接受一个二值路由矩阵 `routing_map` (BOOL/INT8)，形状为 [expertNum, tokenNum]，表示每个专家与每个 token 的映射关系。支持 `drop_and_pad` 模式实现专家容量限制。

**支持平台**: ascend910b (Atlas A2), ascend910_93 (Atlas A3), ascend950
**支持数据类型**: tokens BF16/FP16/FP32, routing_map BOOL/INT8, probs BF16/FP16/FP32

**关键特性**:
1. routing_map 在 L2 层 API 中转置为 [tokenNum, expertNum] 后传入 L0
2. drop_and_pad 模式: capacity = numOutTokens / expertNum，每个专家最多处理 capacity 个 token
3. arch35 (ascend910_93) 专用 Gatherv2SimdTwoDim 内部实现

---

## 2. 目录结构与文件职责

```
moe_token_permute_with_routing_map/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_token_permute_with_routing_map_def.cpp            # 算子定义
│   ├── moe_token_permute_with_routing_map_tiling.h           # Tiling数据结构定义
│   ├── moe_token_permute_with_routing_map_tiling.cpp         # Tiling策略实现
│   ├── op_api/
│   │   ├── aclnn_moe_token_permute_with_routing_map.h        # ACLNN API头文件
│   │   ├── aclnn_moe_token_permute_with_routing_map.cpp      # ACLNN API实现
│   │   ├── moe_token_permute_with_routing_map.h              # 内部接口声明
│   │   └── moe_token_permute_with_routing_map.cpp            # L2层实现（Transpose/Contiguous/ViewCopy）
│   └── config/
│       ├── ascend910_93/ {binary.json, simplified_key.ini}
│       └── ascend910b/  {binary.json, simplified_key.ini}
├── op_kernel/
│   ├── moe_token_permute_with_routing_map.cpp                # Kernel入口（TilingKey分发）
│   ├── masked_select_v3.h                                    # 从routing_map提取有效位
│   ├── moe_common.h                                          # 公共常量
│   ├── moe_index_copy.h                                      # 索引拷贝
│   ├── moe_index_copy_split_d.h                              # 按D维切分索引拷贝
│   ├── moe_mrgsort.h / moe_mrgsort_out.h                    # 多路归并排序
│   ├── moe_permute_prob.h                                    # prob重排Kernel
│   ├── moe_sort_base.h                                       # 排序基类
│   ├── moe_sort_multi_core.h                                 # 多核排序
│   ├── moe_sort_one_core.h                                   # 单核排序
│   ├── moe_sort_multi_core_last_dim.h                        # 按最后一维多核排序
│   └── arch35/ gather_v2_simd_two_dim.h                      # arch35专用gather
├── docs/ aclnnMoeTokenPermuteWithRoutingMap.md               # ACLNN API文档
├── examples/ test_aclnn_moe_token_permute_with_routing_map.cpp
└── tests/ut/ {op_host with op_api, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (op_host/moe_token_permute_with_routing_map_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | tokens | REQUIRED | BF16/FP16/FP32 | 2D: [tokenNum, H] |
| 输入 | routing_map | REQUIRED | BOOL/INT8 | 2D: [expertNum, tokenNum] |
| 输入 | probs | OPTIONAL | BF16/FP16/FP32 | 2D: [expertNum, tokenNum] |
| 输出 | permute_tokens | REQUIRED | 与tokens同类型 | 2D: [numOutTokens, H] |
| 输出 | permute_probs | REQUIRED | 与probs同类型 | 1D: [numOutTokens] |
| 输出 | sorted_indices | REQUIRED | INT32 | 1D: [numOutTokens] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| num_out_tokens | Int | REQUIRED | 输出 token 总数 |
| drop_and_pad | Bool | false | 是否开启容量限制模式 |

### 3.2 Tiling 策略 (op_host/moe_token_permute_with_routing_map_tiling.cpp)

**TilingBaseClass 子类**: `MoeTokenPermuteWithRoutingMapTilingBase`

**TilingKey 编码**:

| TilingKey | 排序模式 | IndexCopy模式 | numOutTokens裁剪 | dropAndPad |
|-----------|---------|--------------|-----------------|------------|
| 1 | 单核 | 正常 | 否 | 否 |
| 2 | 多核 | 正常 | 否 | 否 |
| 3 | 单核 | SplitD | 否 | 否 |
| 4 | 多核 | SplitD | 否 | 否 |
| 5 | 单核 | 正常 | 是 | 否 |
| 6 | 多核 | 正常 | 是 | 否 |
| 7 | 单核 | SplitD | 是 | 否 |
| 8 | 多核 | SplitD | 是 | 否 |
| 9 | LastDim | — | — | 是 |

**核心 Tiling 流程差异**（DoOpTiling）:

非 dropAndPad 模式:
1. `Tiling4MaskedSelect()`: 从 routing_map 提取有效索引（按 expert 维度分配核）
2. `Tiling4VBSCompute()`: 排序切分
3. `Tiling4VMSMiddleCompute()`: 归并切分
4. `Tiling4SortOutCompute()`: 排序输出切分
5. `Tiling4IndexCopyCompute()`: 索引拷贝切分

dropAndPad 模式:
1. `Tiling4VBSComputeLastdim()`: 按最后一维（expert 维度）排序
2. `Tiling4SortOutCompute()`: 排序输出切分
3. [可选] `Tiling4GatherCompute()`: regbase 场景的 Gather 切分

**MaskedSelect Tiling**（Tiling4MaskedSelect）: 按 expert 维度分配核，`numBlocks = min(numExperts, aivUseNum)`，每核处理 `expertSlice * tokenNum` 个元素。

**XDtypeImprove**（第522-543行）: 数据类型提升优化，按 lastAxisBytes 对齐到 2/4/8 字节。

### 3.3 Shape 推导

本算子无独立的 infershape 文件。Shape 推导由 L2 层 API 完成，`numOutTokens` 作为必填属性由调用者提供。

### 3.4 ACLNN API (op_host/op_api/)

采用 CANN 标准两段式接口:

**第一阶段 — 获取 Workspace 大小**:
```cpp
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(
    const aclTensor* tokens,
    const aclTensor* routingMap,
    const aclTensor* probsOptional,
    int64_t numOutTokens,
    bool dropAndPad,
    aclTensor* permuteTokensOut,
    aclTensor* permuteProbsOutOptional,
    aclTensor* sortedIndicesOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

**第二阶段 — 执行计算**:
```cpp
aclnnStatus aclnnMoeTokenPermuteWithRoutingMap(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream)
```

**L2 层处理流程**（aclnn_moe_token_permute_with_routing_map.cpp 第217-312行）:
```
CheckParams() → Contiguous(tokens) → Contiguous(routingMap) → Transpose(routingMap [E,N] → [N,E])
→ [可选] Contiguous + Transpose(probs) → L0算子 → ViewCopy(sortedIndices)
→ [可选] ViewCopy(permuteProbs)
→ if regbase: 直接获取permuteTokens
  else if dropAndPad: GatherV2(tokens, sortedIndices) → permuteTokens
  else: 从L0输出获取
→ ViewCopy(permuteTokens)
```

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (op_kernel/moe_token_permute_with_routing_map.cpp)

**入口函数**（第83-125行）: `extern "C" __global__ __aicore__ void moe_token_permute_with_routing_map(...)`

**初始化流程**:
1. 检查 `g_coreType == AIC` 则返回（仅 AIV 核）
2. 通过 `GET_TILING_DATA` 获取 tiling 数据
3. 获取 `userWS = GetUserWorkspace(workspace)`
4. 根据 TilingKey 分发到非 dropAndPad（Key 1-8）或 dropAndPad（Key 9）路径

### 4.2 核心计算流程

**非 dropAndPad 模式**（GENERAL_OP_IMPL 宏，第35-53行）— 三阶段:
1. `KernelMaskedSelectV3`: 从 routing_map 中提取非零位，生成 sortedIndices 和 permuteProbs → `SyncAll()`
2. `MoeSort`: 对 MaskedSelect 输出的索引做稳定排序
3. `MoeindexCopy`: 按 sortedIndices 搬运 token

**dropAndPad 模式**（GENERAL_PAD_OP_IMPL 宏，第55-70行）— 两阶段:
1. `MoeSortMultiLastDimCore`: 按 expert 维度排序 routing_map，核数按 expert 维度分配
2. [可选] `MoePermuteProb`: 根据 sortedIndices 重排 probs（仅 hasProb=true 时）

**arch35 特殊处理**（第72-81行）: 在 `__NPU_ARCH__ == 3510` 上，dropAndPad 模式后额外执行 `Gatherv2SimdTwoDim` 在 kernel 内部做 gather。

### 4.3 数据结构与内存布局

本算子复用 `moe_token_permute` 的 kernel 模块（排序、IndexCopy），额外的 `MaskedSelectV3` 和 `MoePermuteProb` 为本算子独有。

### 4.4 多核并行策略

- **MaskedSelect**: 按 expert 维度分配核，每核处理一组 expert 的所有 token
- **排序**: 与基础版相同，核数为 4 的幂次
- **IndexCopy**: 按 token 数均匀分配核
- **独占全核**: `SetScheduleMode(1)`（PostTiling 第620行），确保 SyncAll 正确

### 4.5 通信模式

本算子使用核间同步（`AscendC::SyncAll()`），用于 MaskedSelect 完成后确保所有核的数据可见性，再进行排序阶段。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

**顶层结构** `MoeTokenPermuteWithRoutingMapTilingData`（tiling.h 第98-112行）:

```
MoeTokenPermuteWithRoutingMapTilingData
├── coreNum                  : int64_t   # AIV核数
├── n                        : int64_t   # token数
├── cols                     : int64_t   # hidden_size H
├── colsAlign                : int64_t   # 对齐后的H
├── topK                     : int64_t   # numOutTokens / tokenNum
├── expertNum                : int64_t   # 专家数
├── capacity                 : int64_t   # drop_and_pad时的专家容量
├── hasProb                  : int64_t   # 是否有prob输入
├── vbsComputeParamsOp       : PermuteVBSComputeRMTilingData      # 排序参数
│   └── (同基础版 + frontcoreTask, tailcoreTask, frontCoreNum, tailCoreNum)
├── vmsMiddleComputeParamsOp : PermuteVMSMiddleComputeRMTilingData # 归并参数
├── sortOutComputeParamsOp   : PermuteSortOutComputeRMTilingData   # 排序输出参数
├── indexCopyComputeParamsOp : IndexCopyComputeRMTilingData        # 索引拷贝参数
└── maskedSelectParamsOp     : MaskedSelectRMTilingData            # MaskedSelect参数
    └── formerNum/Length/tileNum/tileLength/lasttileLength (前组核)
        tailNum/Length/tileNum/tileLength/lasttileLength (后组核)
        tokenNum, needCoreNum
```

**CompileInfo**（tiling.h 第114-118行）: 保存 `aivNum`, `ubSize`, `workSpaceSize`，用于动态编译静态图。

### 5.2 Workspace 布局

**Workspace 计算**（tiling.cpp 第604-613行）:
```
workspaceSize = sortWorkspaceSize + coreSyncWorkspaceSize + 16MB预留

非dropAndPad:
  sortWorkspaceSize = totalLength * sizeof(float) * 4

dropAndPad:
  sortWorkspaceSize = coreNum * totalLength * sizeof(float) * 4

coreSyncWorkspaceSize = coreNum * 32 * 2
```

---

## 6. 完整数据流

### 6.1 非 dropAndPad 模式

```
  tokens [N,H]    routingMap [E,N] (转置后→[N,E])    probs [E,N]
                        |                                  |
                [KernelMaskedSelectV3]                      |
                 |            |                             |
           sortedIndices  permuteProbs                     |
            (有效位)      (对应prob值)                      |
                 |                                         |
            [MoeSort]                                      |
                 |                                         |
            sortedIndices (排序后)                          |
                 |                                         |
            [MoeindexCopy]                                  |
                 |                                         |
         +-------+-------+                                |
         |               |                                |
   permute_tokens   sorted_indices                  permute_probs
   [numOut, H]      [numOut]                       [numOut]
```

### 6.2 dropAndPad 模式

```
  tokens [N,H]    routingMap [E,N]    probs [E,N]
                      |
          [MoeSortMultiLastDimCore]
           (按expert维度排序, 每核处理所有token)
                      |
                sortedIndices [E*capacity]
                      |
            +---------+----------+
            |                    |
    [GatherV2]            [MoePermuteProb]
    (regbase内部/         (仅hasProb时)
     外部L2调用)               |
        |                     |
 permute_tokens          permute_probs
 [E*capacity, H]         [E*capacity]
```

---

## 7. 性能相关设计

1. **MaskedSelect 核间同步**: MaskedSelect 完成后 `SyncAll()`（kernel 第43行），确保所有核的 sortedIndices 写入 GM 可见后才开始排序。

2. **LastDim 排序优化**: dropAndPad 模式使用 `MoeSortMultiLastDimCore` 按 expert 维度排序，核数按 expert 数分配，每个核处理所有 token，避免跨核同步开销。

3. **arch35 内部 Gather**: 在 ascend910_93 上，dropAndPad 模式的 Gather 操作在 kernel 内部完成（`Gatherv2SimdTwoDim`），避免额外的 kernel launch 开销（kernel 第72-81行）。

4. **数据类型提升**: `XDtypeImprove()` 将 routing_map 的数据按字节对齐提升到更大的整数类型，提高排序效率（tiling.cpp 第522-543行）。

5. **独占全核**: `SetScheduleMode(1)` 确保 SyncAll 正确，但意味着该算子运行时独占所有核资源。

---

## 8. 依赖关系

```
op_host/moe_token_permute_with_routing_map_tiling.cpp
└── moe_token_permute_with_routing_map_tiling.h
    ├── register/tilingdata_base.h              # Tiling 数据基类
    ├── register/op_impl_registry.h             # 算子实现注册
    ├── log/log.h                               # 日志宏
    ├── tiling_base/tiling_base.h               # TilingBaseClass 基类
    └── tiling_base/tiling_templates_registry.h # Tiling 模板注册

op_host/op_api/aclnn_moe_token_permute_with_routing_map.cpp
├── aclnn_moe_token_permute_with_routing_map.h
│   └── moe_token_permute_with_routing_map.h    # 内部接口
└── L0算子调用 + Transpose/Contiguous/ViewCopy

op_kernel/moe_token_permute_with_routing_map.cpp
├── moe_mrgsort_out.h → moe_mrgsort.h → moe_common.h
├── moe_sort_multi_core.h / moe_sort_one_core.h
├── moe_sort_multi_core_last_dim.h              # dropAndPad 排序
├── moe_index_copy.h → moe_common.h
├── moe_index_copy_split_d.h → moe_common.h
├── masked_select_v3.h                          # MaskedSelect 独有
├── moe_permute_prob.h                          # Prob 重排独有
└── [arch35] arch35/gather_v2_simd_two_dim.h    # arch35 专用
```

---

## 9. 与同族变体的核心差异

### 9.1 vs moe_token_permute (基础版)

| 维度 | moe_token_permute | moe_token_permute_with_routing_map |
|------|-------------------|-------------------------------------|
| 第二输入 | indices (expert ID) | routing_map (BOOL/INT8 矩阵) |
| 输入语义 | 直接提供专家 ID | 二值映射矩阵，需 MaskedSelect |
| drop_and_pad | 不支持 | 支持 (capacity = numOut/expertNum) |
| MaskedSelect 阶段 | 无 | 有 (KernelMaskedSelectV3) |
| 排序维度 | 按 totalLength | dropAndPad 时按 expert 维度 |
| Gather 阶段 | 无 | dropAndPad+regbase 时有 |
| probs 处理 | 无 | 支持 (permute_probs 输出) |
| L2 层 Transpose | 无 | routingMap 需转置 [E,N] → [N,E] |
| 平台支持 | 910b/910_93 | 910b/910_93/950 |
| arch35 特殊处理 | 无 | Gatherv2SimdTwoDim |
| CompileInfo | 空结构 | 保存 aivNum/ubSize/workSpaceSize |
| numOutTokens | OPTIONAL | REQUIRED |

### 9.2 vs moe_token_permute_with_ep

| 维度 | moe_token_permute_with_ep | moe_token_permute_with_routing_map |
|------|---------------------------|-------------------------------------|
| 第二输入 | indices (ID) | routing_map (矩阵) |
| range 支持 | [start, end) | 无 range |
| drop_and_pad | 不支持 | 支持 |
| MaskedSelect | 无 | 有 |
| L2 层复杂度 | 直接调用 L0 | Transpose + Contiguous + ViewCopy |
| 核间同步 | 仅排序阶段 | MaskedSelect 后需 SyncAll |

---

## 10. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| num_out_tokens 必填 | 属性 REQUIRED，不可省略 |
| dropAndPad 时 | capacity = numOutTokens / expertNum，需整除 |
| routing_map 类型 | 仅支持 BOOL 和 INT8 |
| 独占全核 | schedule_mode=1，不可与其他算子共享核 |
| DynamicCompileStatic | 配置了动态编译静态图，CompileInfo 需保存平台参数 |
| L2 层转置 | routingMap 在 L2 层转置后传入 L0，外部调用者无需预处理 |
