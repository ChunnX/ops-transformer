# moe_compute_expert_tokens 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeComputeExpertTokens`
**算子类**: MoE Token 计数，专家分配辅助算子
**功能描述**: 给定已排序的专家编号数组 `sorted_experts`，通过二分查找计算每个专家处理的最后一行（token）在数组中的位置。等价于对每个专家 `i` 执行 `out[i] = BinarySearch(sortedExperts, i)`。典型用于 MoE dispatch/permutation 阶段，计算每个专家负责的 token 累积前缀位置。

**支持平台**: ascend910b, ascend910_93, ascend950, kirinx90, kirin9030
**支持数据类型**: INT32

---

## 2. 目录结构与文件职责

```
moe_compute_expert_tokens/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_compute_expert_tokens_def.cpp            # 算子定义
│   ├── moe_compute_expert_tokens_tiling.h           # Tiling数据结构（38字段）
│   ├── moe_compute_expert_tokens_tiling.cpp         # Tiling策略（三模板选择）
│   ├── moe_compute_expert_tokens_infershape.cpp     # Shape推导
│   └── config/ {ascend910b, ascend910_93, ascend950, kirinx90, kirin9030}
├── op_kernel/
│   ├── moe_compute_expert_tokens.cpp                # Kernel入口（3分支）
│   ├── moe_compute_expert_tokens_int32_ss.h        # SS模板（小×小）
│   ├── moe_compute_expert_tokens_int32_m.h         # M模板（大×小）
│   └── moe_compute_expert_tokens_int32_l.h         # L模板（大×大）
├── op_graph/
│   └── moe_compute_expert_tokens_proto.h            # 图模式原型
├── framework/
│   └── npu_moe_compute_expert_tokens_onnx_plugin.cpp  # ONNX插件
├── docs/ aclnnMoeComputeExpertTokens.md
├── examples/ test_aclnn_moe_compute_expert_tokens.cpp
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_compute_expert_tokens_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | sorted_experts | REQUIRED | INT32 | 1D: [N] |
| 输出 | total_rows_before_expert | REQUIRED | INT32 | 1D: [num_experts] |

**属性**:

| 属性名 | 类型 | 说明 |
|--------|------|------|
| num_experts | Int | 专家总数，范围 (0, 2048] |

**关键配置**（第36-42行）: DynamicCompileStaticFlag、DynamicFormatFlag、DynamicRankSupportFlag、DynamicShapeSupportFlag 均为 true，支持 ACLNN 调用。

### 3.2 Tiling 策略 (moe_compute_expert_tokens_tiling.cpp)

**TilingData 结构**（tiling.h 第29-72行）:

```
MoeComputeExpertTokensTilingData (38个 int64_t 字段, 304字节)
├── 基础平台参数
│   ├── totalCoreNum              : int64_t   # 物理总核数
│   └── ubSize                    : int64_t   # 总UB大小
│
├── SyncAll前 - M模板参数（模板2）
│   ├── usedCoreNumBefore         : int64_t
│   ├── normalCoreHandleNumBefore : int64_t   # 非尾核每核处理数
│   ├── normalCoreLoopNumBefore   : int64_t
│   ├── normalCoreHandleNumPerLoopBefore : int64_t  # 上限320
│   ├── normalCoreHandleNumTailLoopBefore : int64_t
│   ├── tailCoreHandleNumBefore   : int64_t
│   ├── tailCoreLoopNumBefore     : int64_t
│   ├── tailCoreHandleNumPerLoopBefore : int64_t
│   └── tailCoreHandleNumTailLoopBefore : int64_t
│
├── SyncAll前 - L模板参数（模板3）
│   ├── usedCoreNumBefore3        : int64_t
│   ├── handleNumPerCoreBefore    : int64_t
│   ├── handleNumTailCoreBefore   : int64_t
│   ├── loopCountBefore           : int64_t
│   ├── loopCountTailBefore       : int64_t
│   ├── handleNumPerLoopBefore    : int64_t   # =320
│   ├── handleNumTailCorePerLoopBefore : int64_t
│   ├── handleExpertNumLoopCount  : int64_t   # 切E的loop数
│   ├── handleExpertNumMainCorePerLoop : int64_t  # <=256
│   ├── handleExpertNumTailCorePerLoop : int64_t
│   ├── loopCountTailCoreMainLoop : int64_t
│   ├── handleNumTailCoreMainLoop : int64_t
│   ├── loopCountTailCoreTailLoop : int64_t
│   └── handleNumTailCoreTailLoop : int64_t
│
├── SyncAll后参数
│   ├── usedCoreNumAfter          : int64_t
│   ├── normalCoreHandleNumAfter  : int64_t
│   ├── normalCoreLoopNumAfter    : int64_t
│   ├── normalCoreHandleNumPerLoopAfter : int64_t  # <=255
│   ├── normalCoreHandleNumTailLoopAfter : int64_t
│   ├── tailCoreHandleNumAfter    : int64_t
│   ├── tailCoreLoopNumAfter      : int64_t
│   ├── tailCoreHandleNumPerLoopAfter : int64_t
│   └── tailCoreHandleNumTailLoopAfter : int64_t
│
└── 其他
    ├── workLocalNeedSize         : int64_t   # ReduceMax/Sum work空间
    ├── sortedExpertNum           : int64_t   # sorted_expert总长度
    ├── numOfExpert               : int64_t   # 专家总数
    ├── userWorkspaceSize         : int64_t
    └── tilingKey                 : int64_t   # 1001/1002/1003
```

**CompileInfo**（tiling.h 第76-79行）:
```
MoeComputeExpertTokensCompileInfo
├── totalCoreNum : int32_t    # 默认40
└── ubSizePlatForm : uint64_t
```

**场景划分常量**（tiling.cpp 第24-34行）:

| 常量 | 值 | 说明 |
|------|----|------|
| ONCE_HANDLE_COUNT | 320 | 单次处理sorted_expert上限 |
| ONCE_RPG_MAX_COUNT | 255 | 单次Repeat专家数上限 |
| SYS_WORKSPACE | 16 MB | 系统workspace |
| E_BOUND_NUM | 256 | 专家数小/大分界 |
| BSK_BOUND_NUM | 12000 | sorted_expert数小/大分界 |
| E_BOUND_NUM_UP_LIMIT | 2048 | 专家数上限 |
| BSK_BOUND_NUM_UP_LIMIT | 2^24 | sorted_expert数上限 |

**TilingKey 选择**（tiling.cpp 第380-395行）:

| Key | 场景 | 条件 | Kernel类 |
|-----|------|------|----------|
| 1001 | SS（小×小） | sortedExpertNum <= 12000 且 numOfExpert <= 256 | `Int32SS` |
| 1002 | M（大×小） | sortedExpertNum > 12000 且 numOfExpert <= 256 | `Int32M` |
| 1003 | L（大×大） | numOfExpert > 256 | `Int32L` |

特殊网络 case（第391-395行）: `sortedExpertNum == 65536 && numOfExpert == 8` 时强制 key=1001。

**Tiling 主流程**（`Tiling4MoeComputeExpertTokens`，第293-422行）:

```
Tiling4MoeComputeExpertTokens
    ├── CheckParamsShape()            # 参数校验
    ├── 获取平台信息 (totalCoreNum, ubSize)
    ├── 获取输入维度 + num_experts属性
    ├── CalcTemplate3ParamTiling()    # L模板SyncAll前参数
    ├── CalcSortedExpertTiling()      # M模板SyncAll前参数
    ├── CalcNumOfExpertTiling()       # SyncAll后参数
    ├── 确定tilingKey（场景选择）
    ├── workspace = numOfExpert * totalCoreNum * 4 + 16MB
    └── 设置TilingData并返回
```

### 3.3 Shape 推导 (moe_compute_expert_tokens_infershape.cpp)

输出 `total_rows_before_expert` shape 固定为 `[num_experts]`（第46-48行），即 1D INT32，长度等于专家数。

### 3.4 ACLNN API

**第一段**（docs/aclnnMoeComputeExpertTokens.md）:
```cpp
aclnnStatus aclnnMoeComputeExpertTokensGetWorkspaceSize(
    const aclTensor *sortedExperts,    // 输入: INT32, 1D
    int64_t          numExperts,        // 属性: 专家总数
    const aclTensor *out,               // 输出: INT32, 1D
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

**第二段**:
```cpp
aclnnStatus aclnnMoeComputeExpertTokens(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
```

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (moe_compute_expert_tokens.cpp)

入口函数（第21-44行）: `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)` — Vector核模式。

```
moe_compute_expert_tokens(sortExperts, out, workspace, tiling)
    ├── GetUserWorkspace(workspace)
    ├── GET_TILING_DATA(tilingData, tiling)
    ├── if TILING_KEY_IS(1001) → MoeComputeExpertTokensInt32SS<int32_t>
    ├── if TILING_KEY_IS(1002) → MoeComputeExpertTokensInt32M<int32_t>
    └── if TILING_KEY_IS(1003) → MoeComputeExpertTokensInt32L<int32_t>
```

### 4.2 核心计算流程

#### SS模板（`_int32_ss.h`，TilingKey=1001）

适用: `sortedExpertNum <= 12000 && numOfExpert <= 256`。单阶段，无 workspace。

```
Init() (第120-145行)
  ├── ParseTilingData(), 设置GM缓冲
  ├── 分配UB: inputQueue_ + tmpOutQueue_ + maskBuf1_ + totalBuf_(6块) + workQueue_
  ├── if blockIdx==0: InitOutput(gmOutput_, numOfExpert, 0)  # 核0清零
  └── SyncAll()

Process() → CopyIn → Compute → CopyOut (每核独立)

Compute() (第162-233行): 向量Compare + ReduceMax/ReduceSum
  for target = startTarget to endTarget:
    Duplicate(targetBuf, target)
    Compare(mask1, inputCast, targetBuf, EQ)    # 逐元素比较
    Select(resultBuf, mask1, oneBuf, 0)          # 生成01掩码
    ReduceMax(reduceMaxAnsBuf, resultBuf)         # 判断是否存在
    if (!isFind): output[target] = lastVal; continue
    ReduceSum(reduceSumAnsBuf, resultBuf)         # 匹配总数
    targetLocation = maxIndex + sumNum            # 最后匹配的全局位置
    output[target] = startOffset + targetLocation

CopyOut() (第236-249行): SetAtomicAdd + SetAtomicMax 原子写回
```

#### M模板（`_int32_m.h`，TilingKey=1002）

适用: `sortedExpertNum > 12000 && numOfExpert <= 256`。两阶段，使用 workspace。

```
Process()
  ├── ProcessBefore()   # 阶段1: 各核独立处理sorted_expert分块
  │     for n = 0 to loopCountBefore-1:
  │       CopyInBefore(n) → ComputeBefore(n) → CopyOutBefore()
  │     ComputeBefore使用标量二分查找（while low <= high）
  │     写入workspace矩阵 [E × C]
  ├── SyncAll()
  └── ProcessAfter()    # 阶段2: 各核从workspace中ReduceMax
        for n = 0 to loopCount_-1:
          CopyInAfter(n) → Cast(int32→float) → WholeReduceMax(沿行方向)
                         → Cast(float→int32) → CopyOutAfter()
```

#### L模板（`_int32_l.h`，TilingKey=1003）

适用: `sortedExpertNum > 12000 && numOfExpert > 256`。两阶段 + 专家维切块。

```
Process()
  ├── ProcessBefore()
  │     ├── ProcessBeforeForNormalCore()  # 或 ProcessBeforeForTailCore()
  │     │     双层循环:
  │     │       外层: 切E（每批<=256个专家）
  │     │       内层: 切sorted_expert（每批<=320个元素）
  │     │     每批专家通过标量二分查找计算结果
  │     └── CopyOutBefore() → workspace矩阵
  ├── SyncAll()
  └── ProcessAfter()   # 与M模板相同: ReduceMax归约
```

### 4.3 数据结构与内存布局

**SS模板数据成员**（_int32_ss.h 第44-78行）:

| 成员 | 类型 | 用途 |
|------|------|------|
| `pipe_` | `TPipe` | 流水线管理 |
| `inputQueue_` | `TQue<VECIN, 1>` | sorted_expert输入队列 |
| `tmpOutQueue_` | `TQue<VECOUT, 1>` | 临时输出队列（E长度int32） |
| `maskBuf1_` | `TBuf<VECCALC>` | 比较结果掩码 |
| `totalBuf_` | `TBuf<VECCALC>` | 6块256对齐的float缓冲 |
| `workQueue_` | `TQue<VECCALC, 1>` | ReduceSum/Max的work空间 |

**M/L模板额外成员**: `workspaceQueue_`, `outputQueue_`, `inputCastTmpBuf_`, `outputCastTmpBuf_`, `syncGlobal_`。

### 4.4 多核并行策略

**SS模板**: 按 sortedExpertNum 均匀分配核（`normalCoreHandleNumBefore = ceil(N, C)`），每核独立处理，通过原子操作合并结果。

**M/L模板阶段1**: 同样按 sortedExpertNum 分核，每核写入 workspace 矩阵的对应列。

**M/L模板阶段2**: 按 numOfExpert 分核（`normalCoreHandleNumAfter = ceil(E, C')`），每核负责若干专家行的 ReduceMax。

**L模板额外**: 专家维按 `E_BOUND_NUM(256)` 分批，sorted_expert 按 `ONCE_HANDLE_COUNT(320)` 分批，双层循环切分。

### 4.5 通信模式

**SS模板**: Init 中一次 `SyncAll()`（等待核0清零输出），CopyOut 通过 `SetAtomicAdd`/`SetAtomicMax` 原子操作合并多核结果。

**M/L模板**: 两阶段间一次 `SyncAll()`。阶段1各核独立写入 workspace 矩阵各自列，阶段2各核独立从 workspace 读取并 ReduceMax，无需额外同步。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 结构（38 个 int64_t 字段，304 字节）。通过 `BEGIN_TILING_DATA_DEF` / `END_TILING_DATA_DEF` 宏定义，`REGISTER_TILING_DATA_CLASS` 注册。Kernel 侧通过 `GET_TILING_DATA` 宏解包。

### 5.2 Workspace 布局

```
Workspace 总大小 = userSize + 16MB

┌──────────────────────────────────────────────────────────────────┐
│ User Workspace (userSize = numOfExpert * totalCoreNum * 4 bytes)│
│                                                                  │
│   布局: 二维矩阵 [numOfExpert × totalCoreNum]，int32             │
│                                                                  │
│   ┌─────────┬─────────┬───────┬─────────┐                        │
│   │ Core 0  │ Core 1  │  ...  │ Core C  │  ← Expert 0           │
│   ├─────────┼─────────┼───────┼─────────┤                        │
│   │ Core 0  │ Core 1  │  ...  │ Core C  │  ← Expert 1           │
│   ├─────────┼─────────┼───────┼─────────┤                        │
│   │  ...    │  ...    │  ...  │  ...    │                        │
│   ├─────────┼─────────┼───────┼─────────┤                        │
│   │ Core 0  │ Core 1  │  ...  │ Core C  │  ← Expert E-1         │
│   └─────────┴─────────┴───────┴─────────┘                        │
│                                                                  │
│   存储: DataCopyPad stride = (usedCoreNumBefore-1)*sizeof(int32)│
├──────────────────────────────────────────────────────────────────┤
│ System Workspace (16 MB, AscendC框架内部使用)                     │
└──────────────────────────────────────────────────────────────────┘
```

**SS模板**: 不使用 User Workspace，直接原子写输出。
**M/L模板**: workspace 矩阵存储每核对每个专家的局部结果，SyncAll 后 ReduceMax 归约到最终输出。

---

## 6. 完整数据流

### 6.1 SS模板（TilingKey=1001）

```
  sorted_experts [N] INT32 (已排序)
       │
       v
  ┌─────────────────────────────────────────────────────┐
  │ 多核并行 (每核分 N/C 个元素)                          │
  │                                                     │
  │  Core 0         Core 1        ...   Core C-1        │
  │  ┌─────────┐   ┌─────────┐        ┌─────────┐      │
  │  │CopyIn   │   │CopyIn   │        │CopyIn   │      │
  │  │分块N/C  │   │分块N/C  │        │尾块     │      │
  │  └────┬────┘   └────┬────┘        └────┬────┘      │
  │       │              │                  │            │
  │  ┌────▼────┐   ┌────▼────┐        ┌───▼─────┐     │
  │  │向量Compare│   │向量Compare│       │向量     │     │
  │  │+ReduceMax│   │+ReduceMax│       │Compare  │     │
  │  │+ReduceSum│   │+ReduceSum│       │+Reduce  │     │
  │  └────┬────┘   └────┬────┘        └────┬────┘      │
  │       │              │                  │            │
  │  ┌────▼────┐   ┌────▼────┐        ┌───▼─────┐     │
  │  │AtomicAdd│   │AtomicAdd│        │Atomic   │     │
  │  │AtomicMax│   │AtomicMax│        │Add/Max  │     │
  │  └────┬────┘   └────┬────┘        └────┬────┘      │
  └───────┼──────────────┼──────────────────┼──────────┘
          │              │                  │
          v              v                  v
  output [E] INT32  ← 原子操作合并多核结果
  out[i] = 专家i处理的最后一行位置
```

### 6.2 M/L模板（TilingKey=1002/1003）

```
  sorted_experts [N] INT32 (已排序)
       │
       v
  ┌───────────────────────────────────────────────────────────┐
  │ 阶段1: ProcessBefore (各核独立)                             │
  │                                                           │
  │  Core 0         Core 1         ...   Core C-1            │
  │  ┌─────────┐   ┌─────────┐         ┌─────────┐          │
  │  │标量二分   │   │标量二分   │         │标量二分   │          │
  │  │查找      │   │查找      │         │查找      │          │
  │  └────┬────┘   └────┬────┘         └────┬────┘          │
  │       │              │                    │                │
  │  CopyOutBefore → workspace矩阵 [E × C]                   │
  │  (每核写入自己的列)                                         │
  └───────────────────────────────┬───────────────────────────┘
                                  │
                             SyncAll()
                                  │
  ┌───────────────────────────────▼───────────────────────────┐
  │ 阶段2: ProcessAfter (各核独立ReduceMax)                     │
  │                                                           │
  │  Core 0         Core 1         ...   Core A-1            │
  │  ┌─────────┐   ┌─────────┐         ┌─────────┐          │
  │  │读WS矩阵   │   │读WS矩阵   │         │读WS矩阵   │          │
  │  │E/A行块   │   │E/A行块   │         │E/A行块   │          │
  │  └────┬────┘   └────┬────┘         └────┬────┘          │
  │       │              │                    │                │
  │  Cast(int32→float) → WholeReduceMax(沿行方向)              │
  │  → Cast(float→int32) → CopyOut → output                  │
  └───────────────────────────────────────────────────────────┘
       │
       v
  output [E] INT32
  out[i] = 专家i处理的最后一行位置
```

---

## 7. 性能相关设计

1. **三模板自适应选择**: 根据数据规模（sortedExpertNum × numOfExpert）自动选择最优算法。小数据量用向量 Compare+Reduce（SS），大数据量用标量二分查找+workspace 归约（M/L），避免向量指令在小数据上的调度开销。

2. **向量比较 vs 标量二分查找**: SS 模板利用向量单元的 Compare/Select/ReduceMax/ReduceSum 指令，一次处理多个 sorted_expert 元素。当数据量增大时（>12000），标量二分查找的 O(log N) 复杂度优于向量遍历的 O(N) 复杂度，故 M/L 模板切换为标量二分查找。

3. **原子操作合并（SS模板）**: 小数据量场景下，多核通过 `SetAtomicAdd`/`SetAtomicMax` 直接写输出，避免 workspace 分配和 SyncAll 开销。

4. **workspace 矩阵归约（M/L模板）**: 阶段1各核独立写入 workspace 各自列（无冲突），阶段2沿行方向 `WholeReduceMax` 归约。仅需一次 SyncAll，且阶段2按专家数分核可充分并行。

5. **专家维切块（L模板）**: 当 numOfExpert > 256 时，按 `E_BOUND_NUM(256)` 分批处理专家维，配合 sorted_expert 的 `ONCE_HANDLE_COUNT(320)` 分批，形成双层循环切分，控制 UB 内存使用在合理范围。

6. **特殊网络 case 优化**: `sortedExpertNum == 65536 && numOfExpert == 8` 时强制走 SS 模板，因为专家数极少时向量 Compare 的效率远超标量二分查找。

---

## 8. 依赖关系

```
op_host/moe_compute_expert_tokens_tiling.cpp
├── moe_compute_expert_tokens_tiling.h
│   ├── register/op_impl_registry.h
│   ├── register/tilingdata_base.h
│   ├── tiling_base/tiling_base.h
│   └── tiling_base/tiling_templates_registry.h
└── (ge, std)

op_host/moe_compute_expert_tokens_def.cpp
└── register/op_def_registry.h

op_host/moe_compute_expert_tokens_infershape.cpp
├── log/log.h
└── register/op_impl_registry.h

op_kernel/moe_compute_expert_tokens.cpp
├── moe_compute_expert_tokens_int32_ss.h
│   ├── kernel_operator.h
│   └── kernel_tiling/kernel_tiling.h
├── moe_compute_expert_tokens_int32_m.h
│   ├── kernel_operator.h
│   └── kernel_tiling/kernel_tiling.h
└── moe_compute_expert_tokens_int32_l.h
    ├── kernel_operator.h
    └── kernel_tiling/kernel_tiling.h

op_graph/moe_compute_expert_tokens_proto.h
└── graph/operator_reg.h

framework/npu_moe_compute_expert_tokens_onnx_plugin.cpp
├── onnx_common.h
└── nlohmann/json.hpp
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| sorted_experts 元素数 | < 2^24（即 16,777,216） |
| num_experts 范围 | (0, 2048] |
| 数据类型 | 仅支持 INT32 |
| sorted_experts 维度 | 必须为 1D |
| sorted_experts 排序 | 必须已排序，否则结果不正确 |
| axis | 当前仅支持 1D（隐式 axis=0） |

---

## 10. 与同族变体的核心差异

本算子为独立的专家计数辅助算子，无直接同族变体。在 MoE 流水线中位于 `moe_init_routing`（路由计算）之后、`moe_token_permute`（token 重排）之前，用于计算每个专家处理的 token 累积前缀位置，供后续 token 重排使用。
