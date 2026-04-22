# moe_token_permute 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenPermute`
**算子类**: MoE Token 排列（基础版）
**功能描述**: 根据 expert 索引矩阵将输入 token 按专家分配顺序重新排列，并输出排序后的索引信息供反向传播使用。核心操作为对 indices 执行稳定排序，再按排序结果搬运 token 数据。

**支持平台**: ascend910b, ascend910_93
**支持数据类型**: BF16/FP16/FP32/INT8 × INT64/INT32（8种组合）

---

## 2. 目录结构与文件职责

```
moe_token_permute/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_token_permute_def.cpp            # 算子定义
│   ├── moe_token_permute_tiling.h           # Tiling数据结构（4个嵌套结构）
│   ├── moe_token_permute_tiling.cpp         # Tiling策略（8档位）
│   ├── moe_token_permute_infershape.cpp     # Shape推导
│   ├── op_api/
│   │   ├── aclnn_moe_token_permute.h        # ACLNN API声明
│   │   └── aclnn_moe_token_permute.cpp      # ACLNN API（RegBase/MemBase双路径）
│   └── config/ {ascend910b, ascend910_93}
├── op_kernel/
│   ├── moe_token_permute.cpp                # Kernel入口（8分支）
│   ├── moe_common.h                         # 公共常量/工具
│   ├── moe_sort_base.h                      # 排序基类
│   ├── moe_sort_one_core.h                  # 单核排序
│   ├── moe_sort_multi_core.h                # 多核排序
│   ├── moe_mrgsort.h                        # 归并排序引擎
│   ├── moe_mrgsort_out.h                    # 归并排序输出（含Extract）
│   ├── moe_index_copy.h                     # 整行搬运模式
│   └── moe_index_copy_spilt_d.h             # 按Hidden维拆分搬运
├── docs/ aclnnMoeTokenPermute.md
├── examples/ test_aclnn_moe_token_permute.cpp
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_token_permute_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | tokens | REQUIRED | BF16/FP16/FP32/INT8 | 2D: [N, H] |
| 输入 | indices | REQUIRED | INT64/INT32 | 1D: [N] 或 2D: [N, topK] |
| 输出 | permute_tokens | REQUIRED | 同 tokens | 2D: [numOutTokens, H] |
| 输出 | sorted_indices | REQUIRED | INT32 | 1D: [N*topK] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| num_out_tokens | Int | 0 | 输出token数；<=0时 = N*topK + num_out_tokens |
| padded_mode | Bool | false | padding模式（当前仅支持false） |

### 3.2 Tiling 策略 (moe_token_permute_tiling.cpp)

**TilingData 层级**（tiling.h 第78-89行）:

```
MoeTokenPermuteTilingData
├── coreNum                              : int64_t   # AIV核数
├── n                                    : int64_t   # token数量
├── cols                                 : int64_t   # hidden_size
├── colsAlign                            : int64_t   # 对齐后hidden_size
├── topK                                 : int64_t   # 每token的expert数
├── vbsComputeParamsOp     : PermuteVBSComputeTilingData        # 排序阶段
├── vmsMiddleComputeParamsOp : PermuteVMSMiddleComputeTilingData # 归并中间
├── sortOutComputeParamsOp : PermuteSortOutComputeTilingData    # 排序输出
└── indexCopyComputeParamsOp : IndexCopyComputeTilingData       # 索引拷贝
```

**IndexCopyComputeTilingData**（tiling.h 第49-75行）: 24个字段，含核分配(`needCoreNum`/`frontCoreNum`/`tailCoreNum`)、数据大小(`oneTokenBtypeSize`/`onceUbTokenNums`)、SplitD参数(`oneTokenMoveTimes`/`oneTokenOnceMove`/`oneTokenlastMove`)、输出配置(`numOutTokens`/`tokenUB`/`indicesUB`)。

**TilingKey 编码**（tiling.cpp 第40-43行）:

```
Key = BASE(1=单核/2=多核) + SPILT_D_MODE(2) + ENABLE_NUMOUTTOKENS(4)
```

| Key | 排序 | 搬运 | 截断 |
|-----|------|------|------|
| 1 | 单核 | 整行 | 全量 |
| 2 | 多核 | 整行 | 全量 |
| 3 | 单核 | SplitD | 全量 |
| 4 | 多核 | SplitD | 全量 |
| 5 | 单核 | 整行 | 截断 |
| 6 | 多核 | 整行 | 截断 |
| 7 | 单核 | SplitD | 截断 |
| 8 | 多核 | SplitD | 截断 |

**关键常量**（tiling.cpp 第18-46行）: `MAX_INDICES_NUM=512`, `SORT_LIMIT_LENGTH=16777215`, `SORT32_ALIGN_ELEMENT=32`。

**Tiling 主流程**:
```
GetPlatformInfo() → GetShapeAttrsInfo() → DoOpTiling()
  ├── Tiling4VBSCompute()         # 排序切分（单核/多核）
  ├── Tiling4VMSMiddleCompute()   # 归并中间切分
  ├── Tiling4SortOutCompute()     # 排序输出切分
  └── Tiling4IndexCopyCompute()   # 索引拷贝切分（含SplitD判断）
→ GetWorkspaceSize() → PostTiling()
```

**Workspace 计算**（第440-448行）: `sizeof(float) * totalLength * 6 + coreNum * 64 + 16MB`。

### 3.3 Shape 推导 (moe_token_permute_infershape.cpp)

```
sorted_indices: [N*topK] (1D indices → [N], 2D → [N*topK])
permute_tokens: [min(max(resolved_numOutTokens, 0), sortedIndicesLen), H]
```

支持动态 Shape (-1) 和 UnknownRank (-2)。

### 3.4 ACLNN API

**两段式接口**（op_api/aclnn_moe_token_permute.h）:

```cpp
// 第一段
aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(
    const aclTensor* tokens, const aclTensor* indices,
    int64_t numOutTokens, bool paddedMode,
    const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

// 第二段
aclnnStatus aclnnMoeTokenPermute(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

**双路径分发**（aclnn_moe_token_permute.cpp 第55-71行）:
- **RegBase 模式**: `IsRegbase()` 为 true 时，转发到 `aclnnInnerMoeInitRoutingV2`
- **MemBase 模式**: 直接调用 `aclnnInnerMoeTokenPermute`

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (moe_token_permute.cpp)

入口函数（第44-82行）: AIC核直接返回，仅 AIV 核执行。使用 `GENERAL_OP_IMPL` 宏串联三阶段流水线:

```
moe_token_permute(tokens, indices, permute_tokens, sorted_indices, workspace, tiling)
    ├── 阶段1: MoeSort<DTYPE_INDICES>     # 按expert ID排序
    ├── 阶段2: MoeSort<int32_t>            # 对排序结果做值排序
    └── 阶段3: MoeindexCopy / MoeindexCopySpiltD  # 按索引搬运token
```

8个 TilingKey 分支按 Key=1-8 分发，组合排序模式(OneCore/MultiCore) × 搬运模式(Normal/SplitD) × 截断模式(false/true)。

### 4.2 核心计算流程

#### 排序阶段

**单核排序**（moe_sort_one_core.h）: 仅 block0 执行。
```
CopyIn(indices) → Cast→FP32 + Negate(-1) → Pad(MIN_FP32)
→ Concat + Sort<float,true> → Extract → CopyOut → SyncAll()
```

**多核排序**（moe_sort_multi_core.h）: 三阶段。
```
VBSProcess(): 每核独立BitonicSort自己的分块 → 本地归并
VMSProcess(): 跨核四路归并，workspace双缓冲ping-pong，每层SyncAll
SortOutProcess(): 最终归并+Extract，仅block0执行
```

**归并排序引擎**（moe_mrgsort.h）: 支持 2/3/4 路归并，`MrgSort<float,true>` 内部指令。`MoeMrgsortOut`（moe_mrgsort_out.h）额外执行 Extract 拆分排序键和源行索引。

#### 索引拷贝阶段

**整行搬运**（moe_index_copy.h）: 单个token行可完整装入UB。
```
for 批次(indices):
  CopyInIndices → for 每个token:
    读取index → CopyIn(token行) → CopyOut(token到目标位置)
```

**SplitD搬运**（moe_index_copy_spilt_d.h）: token行超UB容量时按H维拆分。
```
for 批次(indices):
  CopyInIndices → for 每个token:
    for 每个H维分片:
      CopyIn(分片) → CopyOut(到目标位置+偏移)
```

### 4.3 数据结构与内存布局

**MoeSortBase**（moe_sort_base.h）: 排序基类，含 `sortDataCopyInQueue`/`sortDataCopyOutQueue`(depth=1), `tempBuffer`/`sortedBuffer`(VECCALC), 3 个 GlobalTensor。

**MoeindexCopyOp**（moe_index_copy.h）: `copyInQueue`(VECIN/VECOUT), `indicesBuffer`(VECOUT), 事件同步 `indicesMte2ToS`/`indicesSToMte2`。

**公共常量**（moe_common.h）: `MIN_FP32=-3.4e38f`, `ONE_REPEAT_SORT_NUM=32`, `BLOCK_BYTES=32`, `MERGE_LIST_TWO/THREE/FOUR=2/3/4`。

### 4.4 多核并行策略

- **单核排序**: `totalLength <= sortLoopMaxElement` 时 block0 独立排序
- **多核排序**: 核数取最近的 4 的幂次（4/16/64），VBS 各核独立 → VMS 四路归并树 → SortOut block0
- **索引拷贝**: 按 `coreCalcNum`/`coreCalcTail` 分前组/尾组核，前组尾组任务量可不同

### 4.5 通信模式

- **单核排序**: 完成后一次 `SyncAll()`
- **多核排序**: VBS 各核独立 → VMS 每层归并 `SyncAll()` → SortOut 仅 block0
- **索引拷贝**: 使用显式硬件事件同步（`SetFlag<HardEvent::MTE2_S>`/`WaitFlag`）替代 `PipeBarrier`，更细粒度

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 层级（4 个嵌套 TilingData + 5 个基础 int64_t）。通过 `BEGIN_TILING_DATA_DEF`/`END_TILING_DATA_DEF` 宏定义，Kernel 侧 `GET_TILING_DATA` 解包。

**CompileInfo**（tiling.h 第91-92行）: 空结构。

### 5.2 Workspace 布局

```
┌──────────────────────────────────────────────────────────────┐
│ sortWorkspace = sizeof(float) * totalLength * 2 * 3          │
│ (排序中间缓冲: expert+row float pair，三缓冲轮转)            │
├──────────────────────────────────────────────────────────────┤
│ coreSyncWorkspace = coreNum * 32 * 2                         │
│ (核间同步标记)                                                │
├──────────────────────────────────────────────────────────────┤
│ System Workspace = 16 MB                                      │
│ (AscendC框架内部使用)                                          │
└──────────────────────────────────────────────────────────────┘
```

PostTiling 设置 `SetScheduleMode(1)` 使排序独占全核。

---

## 6. 完整数据流

```
  tokens [N, H]              indices [N, topK]
       │                            │
       │                    ┌───────┴───────┐
       │                    │               │
       │              [MoeSort]        [MoeSort]
       │           (by expert ID)  (value sort)
       │                    │               │
       │                    └───────┬───────┘
       │                            │
       │                     sorted_indices
       │                       [N*topK]
       │                            │
       └──────────┬─────────────────┘
                  │
          [MoeindexCopy]
          (按索引搬运token)
                  │
          ┌───────┴────────┐
          │                │
  permute_tokens     sorted_indices
  [numOut, H]          [N*topK]
```

---

## 7. 性能相关设计

1. **多核并行排序**: 根据 `totalLength` 自动选择单核BitonicSort或多核VBS+VMS+SortOut。核数取最近 4^x，归并层数 = log4(核数)。

2. **Hidden维拆分**: `H` 过大时启用 SplitD 模式，将单个 token 行拆分为 `oneTokenMoveTimes` 次搬运。indices 只加载一次复用于所有分片。

3. **显式硬件事件同步**: 索引拷贝阶段使用 `SetFlag<HardEvent::MTE2_S>`/`WaitFlag` 替代 `PipeBarrier`，更细粒度的流水重叠。

4. **UB空间精细管理**: 前组/尾组核独立计算UB参数，indices 独立 Queue 与 token TQueBind 解耦。所有 DataCopy 参数在 Init 阶段预计算。

5. **排序独占全核调度**: `SetScheduleMode(1)` 排序阶段独占所有 AIV 核，避免资源竞争。

6. **归并排序优化**: `MoeMrgsort` 支持 2/3/4 路归并，`validBitTail` 位掩码动态选择归并路数。workspace 双缓冲 ping-pong 减少写冲突。

7. **ACLNN 双路径分发**: RegBase 平台（ascend950）转发到 `MoeInitRoutingV2` 内部实现，MemBase 平台走原生 kernel。

---

## 8. 依赖关系

```
op_host/moe_token_permute_def.cpp
└── register/op_def_registry.h

op_host/moe_token_permute_tiling.cpp
├── moe_token_permute_tiling.h
│   ├── register/op_impl_registry.h
│   ├── register/tilingdata_base.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/moe_token_permute_infershape.cpp
├── log/log.h
└── register/op_impl_registry.h

op_host/op_api/aclnn_moe_token_permute.cpp
├── aclnn_moe_token_permute.h
├── external/aclnn_kernels/aclnn_platform.h  (IsRegbase判断)
└── aclnnInnerMoeInitRoutingV2 / aclnnInnerMoeTokenPermute

op_kernel/moe_token_permute.cpp
├── moe_sort_one_core.h
│   ├── moe_sort_base.h
│   │   └── kernel_operator.h
│   └── moe_common.h
├── moe_sort_multi_core.h
│   ├── moe_sort_base.h
│   ├── moe_mrgsort.h
│   └── moe_mrgsort_out.h
├── moe_index_copy.h
├── moe_index_copy_spilt_d.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| padded_mode | 当前仅支持 false（第295-298行） |
| indices 维度 | 必须为 1D 或 2D，dim-0 与 tokens 的 dim-0 相同 |
| topK 上限 | <= 512（MAX_INDICES_NUM） |
| totalLength 上限 | <= 16777215（SORT_LIMIT_LENGTH） |
| 排序独占 | SetScheduleMode(1)，排序阶段独占全核 |
| 多核核数 | 必须为 4 的幂次 |
| data type | sorted_indices 固定 INT32 |

---

## 10. 与同族变体的核心差异

### vs moe_token_permute_with_ep

| 维度 | 本算子 | with_ep |
|------|--------|---------|
| probs 输入/输出 | 无 | 有 |
| numOutTokens 语义 | 绝对值或负数偏移 | 支持 range[start,end) |
| 索引拷贝 | MoeindexCopyOp | IndexMixCopyComputeTilingData |
| 工作空间 | totalLength * 6 | totalLength * 4 |
| EP 支持 | 不支持 | 支持 |

### vs moe_token_permute_with_routing_map

| 维度 | 本算子 | with_routing_map |
|------|--------|------------------|
| 第二输入 | indices (expert ID) | routing_map (BOOL/INT8) |
| 排序方式 | 直接排序 indices | MaskedSelectV3 + 排序 |
| drop_and_pad | 不支持 | 支持 |
| probs 处理 | N/A | MoePermuteProb 单独处理 |
| 平台 | 910b/910_93 | 910b/910_93/950 |
