# moe_token_permute_with_ep 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenPermuteWithEp`
**算子类**: MoE (混合专家模型) Token 重排，Expert Parallelism (EP) 增强变体
**功能描述**: 在 EP 场景下，根据 `indices` 将 `tokens` 和 `probs` 按专家索引排序并重排，通过 `range` 参数指定当前设备负责的专家范围 `[start, end)`，仅输出该范围内的 token。相比基础版 `moe_token_permute`，额外支持 probs 的并行重排。

**支持平台**: ascend910b (Atlas A2), ascend910_93 (Atlas A3)
**支持数据类型**: tokens BF16/FP16/FP32, indices INT32/INT64, probs BF16/FP16/FP32

---

## 2. 目录结构与文件职责

```
moe_token_permute_with_ep/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_with_ep_def.cpp            # 算子定义（输入/输出/属性声明）
│   ├── moe_token_permute_with_ep_infershape.cpp     # Shape推导
│   ├── moe_token_permute_with_ep_tiling.h           # Tiling数据结构定义
│   ├── moe_token_permute_with_ep_tiling.cpp         # Tiling策略实现
│   ├── moe_token_permute_tiling_base.h              # 排序/IndexCopy公共Tiling子结构体
│   └── config/
│       ├── ascend910_93/ {binary.json, simplified_key.ini}
│       └── ascend910b/  {binary.json, simplified_key.ini}
├── op_kernel/
│   ├── moe_token_permute_with_ep.cpp                # Kernel入口（TilingKey分发）
│   ├── moe_common.h                                 # 公共常量定义
│   ├── moe_index_copy.h                             # 索引拷贝（含probs搬运）
│   ├── moe_index_copy_spilt_d.h                     # 按D维切分的索引拷贝
│   ├── moe_mrgsort.h / moe_mrgsort_out.h           # 多路归并排序
│   ├── moe_sort_base.h                              # 排序基类
│   ├── moe_sort_multi_core.h                        # 多核排序
│   └── moe_sort_one_core.h                          # 单核排序
├── docs/ aclnnMoeTokenPermuteWithEp.md              # ACLNN API文档
├── examples/ test_aclnn_moe_token_permute_with_ep.cpp
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (op_host/moe_token_permute_with_ep_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | tokens | REQUIRED | BF16/FP16/FP32 | 2D: [N, H] |
| 输入 | indices | REQUIRED | INT64/INT32 | 1D: [N] 或 2D: [N, topK] |
| 输入 | probs | OPTIONAL | BF16/FP16/FP32 | 与indices同形 |
| 输出 | permute_tokens | REQUIRED | 与tokens同类型 | 2D: [numOutTokens, H] |
| 输出 | sorted_indices | REQUIRED | INT32 | 1D: [N*topK] |
| 输出 | permute_probs | REQUIRED | 与probs同类型 | 1D: [numOutTokens] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| range | ListInt | {0, 0} | EP场景的专家范围 [start, end) |
| num_out_tokens | Int | 0 | range为空时的输出token数 |
| padded_mode | Bool | false | padding模式（当前仅支持false） |

**与基础版的关键差异**:
1. 新增 probs 输入/输出：支持同时重排专家路由概率
2. range 属性替代简单的 num_out_tokens，支持区间语义
3. permute_probs 为必选输出（非optional）
4. tokens 不含 INT8 数据类型

### 3.2 Tiling 策略 (op_host/moe_token_permute_with_ep_tiling.cpp)

**TilingBaseClass 子类**: `MoeTokenPermuteWithEpTilingBase`（第149行），继承自 `Ops::Transformer::OpTiling::TilingBaseClass`。

**Tiling 流水线**: `GetPlatformInfo()` → `GetShapeAttrsInfo()` → `IsCapable()` → `DoOpTiling()` → `DoLibApiTiling()` → `GetWorkspaceSize()` → `PostTiling()`

**TilingKey 编码**（8 种策略，第616-618行 + 第838-839行）:

| TilingKey | 排序模式 | IndexCopy模式 | numOutTokens裁剪 |
|-----------|---------|--------------|-----------------|
| 1 | 单核 | 正常 | 否 |
| 2 | 多核 | 正常 | 否 |
| 3 | 单核 | SplitD | 否 |
| 4 | 多核 | SplitD | 否 |
| 5 | 单核 | 正常 | 是 |
| 6 | 多核 | 正常 | 是 |
| 7 | 单核 | SplitD | 是 |
| 8 | 多核 | SplitD | 是 |

编码规则: `base(1=单核/2=多核) + splitD(+2) + numOutTokens(+4)`

**Tiling 四阶段计算**（第599-609行 `DoOpTiling`）:
1. `Tiling4VBSCompute()`: 计算排序阶段的分核参数，核数为 4 的幂次（第661行）
2. `Tiling4VMSMiddleCompute()`: 计算多路归并中间阶段参数
3. `Tiling4SortOutCompute()`: 计算排序输出阶段参数
4. `Tiling4IndexCopyCompute()`: 计算索引拷贝阶段参数，重构为 CoreParams / UBParams / LoopParams 三组

**UB 空间分配差异**（第741-776行 `CalculateUBParams`）:
- 正常模式: UB空间同时容纳 token + indices + probs，`PROBS_SPACE_SCALE=2` 表示 probs 空间按 indices 空间等额预留
- SplitD 模式: 当 cols 过大无法一次装入 UB 时，按 D 维分多次搬运，预留 `MAX_INDICES_NUM * INT32_DTYPE_SIZE` 的 indices 空间

**numOutTokens 计算**（第514-517行）:
```
numOutTokens = end - start
numOutTokens = (numOutTokens <= 0) ? numOutTokens + totalLength : numOutTokens  // 支持环绕语义
numOutTokens = min(numOutTokens, totalLength)
numOutTokens = max(numOutTokens, 0)
```

**校验增强**（第369-461行）:
- `CheckAndGetAttrsInfo()`: 校验 range 大小为 2、paddedMode 为 false
- `CheckInputShape()`: 校验 probs 与 indices 的维度匹配、tokens dim0 与 indices dim0 一致
- `CheckOutputShape()`: 校验 permute_tokens 和 permute_probs 的输出形状

### 3.3 Shape 推导 (op_host/moe_token_permute_with_ep_infershape.cpp)

**推导逻辑**（第34-92行）:
```
sorted_indices 推导:
  同基础版: [N*topK]

permute_tokens 推导:
  若 range != nullptr:
    start = range[0], end = range[1]
  否则:
    start = 0, end = numOutTokens

  若 start >= end:
    outTokens = sortedIndicesLen + (end - start)   // 支持环绕语义
  否则:
    outTokens = end - start

  outTokens = min(outTokens, sortedIndicesLen)
  outTokens = max(outTokens, 0)
  permute_tokens = [outTokens, H]

permute_probs: 不在 infershape 中单独推导（由 tiling 阶段确定）
```

**数据类型推导**（第94-104行）:
```
permute_tokens.dtype = tokens.dtype
sorted_indices.dtype  = DT_INT32
permute_probs.dtype   = probs.dtype
```

### 3.4 ACLNN API (docs/aclnnMoeTokenPermuteWithEp.md)

采用 CANN 标准两段式接口:

**第一阶段 — 获取 Workspace 大小**:
```cpp
aclnnStatus aclnnMoeTokenPermuteWithEpGetWorkspaceSize(
    const aclTensor    *tokens,
    const aclTensor    *indices,
    const aclTensor    *probsOptional,
    const aclIntArray  *rangeOptional,
    int64_t             numOutTokens,
    bool                paddedMode,
    const aclTensor    *permuteTokensOut,
    const aclTensor    *sortedIndicesOut,
    const aclTensor    *permuteProbsOut,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor)
```

**第二阶段 — 执行计算**:
```cpp
aclnnStatus aclnnMoeTokenPermuteWithEp(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

**错误码**: `ACLNN_ERR_PARAM_NULLPTR`(161001) — 空指针; `ACLNN_ERR_PARAM_INVALID`(161002) — 类型/格式不支持; `ACLNN_ERR_INNER_TILING_ERROR`(561002) — shape 维度或 paddedMode 校验失败。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (op_kernel/moe_token_permute_with_ep.cpp)

**入口函数**（第47-86行）: `extern "C" __global__ __aicore__ void moe_token_permute_with_ep(...)`

**初始化流程**:
1. 检查 `g_coreType == AIC` 则直接返回（仅使用 AIV 核）
2. 通过 `GET_TILING_DATA` 获取 tiling 数据
3. 获取 `userWS = GetUserWorkspace(workspace)`
4. 根据 `TILING_KEY_IS` 分发到 8 种路径

**模板参数**:
- `DTYPE_TOKENS`: 默认 `bfloat16_t`
- `DTYPE_INDICES`: 默认 `int64_t`
- `DTYPE_PROBS`: 默认 `bfloat16_t`

### 4.2 核心计算流程

**GENERAL_OP_IMPL 宏**（第32-45行）执行三阶段流水线:

```
阶段1: 排序 indices → sortedIndices (argSort)
    sortClass1<DTYPE_INDICES>::Init(indices, sortedIndices, userWS, t, &sortPipe)
    sortClass1::Process()
    sortPipe.Destroy()

阶段2: 排序 sortedIndices → sortedIndices (value sort)
    sortClass2<int32_t>::Init(sortedIndices, sortedIndices, userWS, t, &sortPipe2)
    sortClass2::Process()
    sortPipe2.Destroy()

阶段3: 索引拷贝（同时搬运 token 和 prob）
    indexCopyClass<DTYPE_TOKENS, DTYPE_PROBS, ifNumOutTokens>::Init(
        tokens, probs, sortedIndices, permuteTokens, permuteProbs, t, &MoeindexCopyPipe)
    indexCopyClass::Process()
```

**阶段3 的 IndexCopy 处理**（moe_index_copy.h）:
- `CopyInIndices`: 从 GM 搬运 sortedIndices 到 UB
- `CopyIn`: 根据 sortedIndices 逐 token 从 GM 搬运到 UB
- `CopyInProbs`: 从 GM 搬运 probs 到 UB（若 probs 存在）
- `CopyOut`: 根据 indicesValue 写回 permuteTokens（ifNumOutTokens=true 时检查 range 裁剪）
- `CopyOutProbs`: 写回 permuteProbs（需 MTE3→S→MTE3 事件同步）

### 4.3 数据结构与内存布局

**MoeindexCopyOp 的 UB 布局**（moe_index_copy.h Init 方法，第286-289行）:
```
UB 空间划分 (由 tiling 中的 tokenUB / indicesUB / probsUB 决定):
+---------------------------------------------------+
| copyInQueue (tokenUB, BUFFER_NUM=2, 双缓冲)       |  ← token 数据
+---------------------------------------------------+
| indicesBuffer (indicesUB, 1份)                     |  ← sorted indices
+---------------------------------------------------+
| copyProbsInQueue (probsUB, 1份)                    |  ← probs 数据
+---------------------------------------------------+
| probsTmpBuffer (32B)                               |  ← 单元素 probs 临时空间
+---------------------------------------------------+
```

**为什么 copyInQueue 用双缓冲而其他队列不用**: token 数据搬运量最大（cols * sizeof(T)），双缓冲允许 CopyIn 和 CopyOut 流水重叠，隐藏延迟。indices 和 probs 搬运量相对较小，单缓冲即可。

### 4.4 多核并行策略

**排序阶段**（VBS Compute）:
- 单核模式（TilingKey=1,3,5,7）: totalLength ≤ sortLoopMaxElement 时使用
- 多核模式（TilingKey=2,4,6,8）: 核数为 4 的幂次（第661行 `pow(4, CeilLog4(...))`），不超过物理 AIV 核数
- 多核排序后通过 VMS 归并（4路归并）合并各核排序结果

**IndexCopy 阶段**:
- 按 token 数均匀分配给前 `frontCoreNum` 核和尾 `tailCoreNum` 核
- 前核处理 `coreCalcNum` 个 token，尾核处理 `coreCalcTail` 个 token（第254-269行）
- 核间通过 `SyncAll()` 同步（coreNum > 1 时调用 `AscendC::SyncAll()`，第212-217行）

**独占全核**: PostTiling 中设置 `SetScheduleMode(1)`（第636行），确保排序阶段的核间同步正确。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

**顶层结构** `MoeTokenPermuteWithEpTilingData`（tiling.h 第59-71行）:

```
MoeTokenPermuteWithEpTilingData
├── coreNum              : int64_t   # AIV核数
├── n                    : int64_t   # token数 N
├── cols                 : int64_t   # hidden_size H
├── colsAlign            : int64_t   # 对齐后的 H（32B对齐）
├── topK                 : int64_t   # 每 token 专家数
├── vbsComputeParamsOp   : PermuteVBSComputeTilingEPData     # 排序分核参数
│   ├── needCoreNum, perCoreElements, perCoreLoops
│   ├── perCorePerLoopElements, perCoreLastLoopElements
│   ├── lastCoreElements, lastCoreLoops
│   ├── lastCorePerLoopElements, lastCoreLastLoopElements
│   ├── oneLoopMaxElements, lastCoreWSindex
├── vmsMiddleComputeParamsOp : PermuteVMSMiddleComputeTilingEPData  # 归并参数
│   └── needCoreNum
├── sortOutComputeParamsOp : PermuteSortOutComputeTilingEPData      # 排序输出参数
│   └── oneLoopMaxElements
└── indexCopyComputeParamsOp : IndexMixCopyComputeTilingData        # 索引拷贝参数
    ├── needCoreNum, frontCoreNum, tailCoreNum, coreCalcNum, coreCalcTail
    ├── oneTokenBtypeSize, oneProbBtypeSize
    ├── onceIndicesTokenMoveTimes, onceUbTokenNums, onceIndicesTokenNums
    ├── onceIndices, oneTokenlastMove, oneTokenOnceMove, oneTokenMoveTimes
    ├── frontCoreLoop, frontCoreLastTokenNums
    ├── tailCoreLoop, tailCoreLastTokenNums
    ├── tailLastonceIndicesTokenMoveTimes, tailLastIndicesLastTokenNums
    ├── frontLastonceIndicesTokenMoveTimes, frontLastIndicesLastTokenNums
    ├── numOutTokens, start, end
    └── tokenUB, indicesUB, probsUB
```

**与基础版 IndexCopyComputeTilingData 的差异**（tiling.h 第26-56行 vs tiling_base.h 第47-73行）:

| 字段 | 基础版 | EP版 |
|------|--------|------|
| oneProbBtypeSize | 无 | 有 |
| start, end | 无 | 有 |
| probsUB | 无 | 有 |
| oneTokenBtypeSize | 有 | 有 |

### 5.2 Workspace 布局

**Workspace 计算**（tiling.cpp 第621-629行）:
```
workspaceSize = sortWorkspaceSize + coreSyncWorkspaceSize + 16MB预留

其中:
  sortWorkspaceSize      = sizeof(float) * totalLength * 2 * 2  (排序临时空间)
  coreSyncWorkspaceSize  = coreNum * 32 * 2                     (多核同步空间)
  预留                   = SIZE_16 * 1024 * 1024 = 16MB
```

**Workspace 使用方式**（kernel 入口第60-63行）:
```
GM_ADDR userWS = GetUserWorkspace(workspace)
```
排序阶段使用 userWS 作为多路归并的临时缓冲区。

---

## 6. 完整数据流

```
输入阶段:
    tokens [N,H]     indices [N,topK]     probs [N,topK]
        |                  |                   |
        +------------------+-------------------+
                           |
                    [Host: Tiling计算]
                    - 计算排序分核参数 (VBS/VMS/SortOut)
                    - 计算IndexCopy分核参数
                    - 计算numOutTokens
                           |
                    [Host: PostTiling]
                    - SetBlockDim(aivNum)
                    - SetScheduleMode(1) 独占全核
                           |

Kernel阶段1 — 排序 (Sort):
    indices [N*topK]
        |
    [MoeSort: argSort by value]  ← 单核或多核(4^x核)
        |
    sortedIndices (临时, 第1次排序结果)
        |
    [MoeSort: value sort]        ← 对第1次结果再做argSort
        |
    sortedIndices [N*topK]       ← 最终排序索引（输出）

Kernel阶段2 — 索引拷贝 (IndexCopy):
    tokens [N,H]   sortedIndices [N*topK]   probs [N,topK]
        |                  |                      |
        +------------------+----------------------+
                           |
              [MoeindexCopyOp / MoeindexCopySpiltDOp]
              每核处理一段token:
                CopyInIndices → CopyIn(token) → CopyInProbs
                → CopyOut(按indicesValue scatter写)
                → CopyOutProbs(按indicesValue scatter写)
                ifNumOutTokens=true时: 仅写 [start, start+numOutTokens) 范围
                           |
              +------------+------------+
              |            |            |
      permute_tokens  sorted_indices  permute_probs
      [numOut, H]     [N*topK]       [numOut]
```

---

## 7. 性能相关设计

1. **双缓冲流水线**: IndexCopy 阶段的 `copyInQueue` 使用 BUFFER_NUM=2 双缓冲（moe_index_copy.h 第286行），token 的 CopyIn 和 CopyOut 可流水重叠。

2. **SplitD 模式**: 当 cols 过大导致单个 token 无法一次装入 UB 时（`ubLeft < 2 * oneTokenBtypeSizeAlign32`），自动切换到 SplitD 模式（TilingKey +2），按 D 维分多次搬运，避免 UB 溢出（tiling.cpp 第764-774行）。

3. **多核排序核数优化**: 排序核数取 4 的幂次（`4^CeilLog4(...)`），与 4 路归并排序的算法特性匹配，避免归并阶段出现不平衡（tiling.cpp 第661行）。

4. **UB 空间预留策略**: UB 预先扣除 `PROBS_SPACE_SCALE * MAX_INDICES_NUM * INT32_DTYPE_SIZE` 用于 probs 和 indices 存储，`PROBS_SPACE_SCALE=2` 确保 indices 和 probs 各有足够空间（tiling.cpp 第744行）。

5. **硬件事件同步**: probs 的 CopyOut 使用 `MTE3→S→MTE3` 事件对（moe_index_copy.h 第189-193行），确保 scalar 操作（SetValue）与 DMA 操作（DataCopyPad）的执行顺序正确。

---

## 8. 依赖关系

```
op_host/moe_token_permute_with_ep_tiling.cpp
├── moe_token_permute_with_ep_tiling.h
│   ├── moe_token_permute_tiling_base.h           # 排序/IndexCopy 公共 Tiling 子结构体
│   │   └── tiling/tiling_api.h                    # CANN Tiling 框架
│   │       └── register/op_impl_registry.h        # 算子注册框架
│   └── log/log.h                                  # 日志宏
└── tiling_base/tiling_base.h                      # TilingBaseClass 基类

op_host/moe_token_permute_with_ep_def.cpp
└── 算子定义框架 (OpDef)

op_host/moe_token_permute_with_ep_infershape.cpp
└── Shape 推导框架

op_kernel/moe_token_permute_with_ep.cpp
├── moe_mrgsort_out.h
│   └── moe_mrgsort.h
│       └── moe_common.h                           # 公共常量
├── moe_sort_multi_core.h                          # 多核排序实现
├── moe_sort_one_core.h                            # 单核排序实现
├── moe_index_copy.h                               # IndexCopy（含 probs）
│   └── moe_common.h
└── moe_index_copy_spilt_d.h                       # SplitD IndexCopy
    └── moe_common.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 | 来源 |
|------|------|------|
| indices 元素上限 | < 16,777,215 (2^24-1) | tiling.cpp 第502行 |
| topK 范围 | (0, 512] | tiling.cpp 第495-496行 |
| paddedMode | 当前仅支持 false | tiling.cpp 第391-394行 |
| range 大小 | 必须为 2 | tiling.cpp 第379-380行 |
| tokens 维度 | 必须 2D | tiling.cpp 校验 |
| indices 维度 | 1D 或 2D | tiling.cpp 第406行 |
| probs 形状 | 与 indices 同形 | tiling.cpp 第440-441行 |
| 确定性计算 | 默认确定性实现 | ACLNN API 文档 |
| 独占全核 | schedule_mode=1，不可与其他算子共享核 | tiling.cpp 第636行 |
| 不支持 INT8 | tokens 仅 BF16/FP16/FP32 | def.cpp |
