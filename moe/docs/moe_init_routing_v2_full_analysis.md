# moe_init_routing_v2 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeInitRoutingV2`
**算子类**: MoE 路由初始化（第二代）
**功能描述**: 相比 v1 去除外部 `row_idx` 输入（自行从 expert_idx 推导），新增 Drop/Pad 专家容量控制（3D 输出）、expert_tokens_count/cumsum 输出、expert_tokens_before_capacity 输出。支持 HistRegBase 直方图排序优化和 SIMT 多线程映射。

**支持平台**: ascend910b, ascend910_93, ascend310P（仅 FP16/FP32，不支持 BF16 和 Drop/Pad）, ascend950（regbase，支持 INT64 expert_idx）
**支持数据类型**: FP16/BF16/FP32 × INT32（ascend950 额外支持 INT64 expert_idx）

---

## 2. 目录结构与文件职责

```
moe_init_routing_v2/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_init_routing_v2_def.cpp             # 算子定义
│   ├── moe_init_routing_v2_tiling.h            # Tiling数据结构（9基础+6嵌套）
│   ├── moe_init_routing_v2_tiling.cpp          # Tiling策略（9档位）
│   ├── moe_init_routing_v2_tiling_util.h       # Tiling工具函数
│   ├── moe_init_routing_v2_infershape.cpp      # Shape推导
│   ├── op_api/
│   │   ├── aclnn_moe_init_routing_v2.h         # ACLNN API声明
│   │   └── aclnn_moe_init_routing_v2.cpp       # ACLNN API（转发内部实现）
│   └── config/ {ascend910b, ascend910_93, ascend310p, ascend950}
├── op_kernel/
│   ├── moe_init_routing_v2.cpp                 # Kernel入口（三架构分支×9 TilingKey）
│   ├── moe_v2_common.h                         # 公共常量/工具
│   ├── moe_v2_sort_base.h                      # 排序基类
│   ├── one_core.h                              # 单核排序
│   ├── multi_core.h                            # 多核排序
│   ├── moe_v2_mrgsort.h                        # 多路归并排序引擎
│   ├── moe_v2_mrgsort_out.h                    # 归并排序输出（含Extract）
│   ├── expert_token_out.h                      # 专家Token计数输出
│   ├── moe_v2_src_to_dst_op.h                  # 源到目的映射（Dropless）
│   ├── moe_v2_src_to_dst_op_simt.h             # SIMT源到目的映射
│   ├── moe_v2_src_to_dst_op_with_capacity.h    # DropPad容量映射
│   ├── moe_v2_gather_out.h                     # Gather输出模板
│   ├── moe_v2_init_routing_fullload.h          # FullLoad模式Kernel
│   └── arch35/                                 # ascend950 regbase专用
│       ├── moe_v2_common.h, sort_base.h
│       ├── moe_v2_sort_one_core.h, multi_core.h
│       ├── moe_v2_mrgsort.h, mrgsort_out.h
│       ├── moe_v2_expert_token_out_regbase.h   # DHist/CHist直方图
│       ├── moe_v2_expert_token_out_simt.h      # SIMT Token计数
│       ├── moe_v2_src_to_dst_op_simt.h
│       ├── moe_v2_src_to_dst_with_capacity_simt.h
│       └── moe_v2_gather_out_for_simt.h
├── op_graph/ moe_init_routing_v2_proto.h
├── docs/ aclnnMoeInitRoutingV2.md
├── examples/ test_aclnn_moe_init_routing_v2.cpp
└── tests/ut/ {op_host, op_kernel}, tests/pytest/
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_init_routing_v2_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | x | REQUIRED | FP16/BF16/FP32 | 2D: [N, cols] |
| 输入 | expert_idx | REQUIRED | INT32（950 额外支持 INT64） | 2D: [N, K] |
| 输出 | expanded_x | REQUIRED | 同 x | Dropless: [min(N*K, activeNum), cols]; DropPad: [E, C, cols] |
| 输出 | expanded_row_idx | REQUIRED | INT32 | 1D: [N*K] |
| 输出 | expert_tokens_count_or_cumsum | OPTIONAL | INT32 | 1D: [expertNum] |
| 输出 | expert_tokens_before_capacity | OPTIONAL | INT32 | 1D: [expertNum] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| active_num | Int | 0 | 活跃token行数；0时默认=N*K |
| expert_capacity | Int | 0 | 每个专家最大token容量（Drop/Pad模式） |
| expert_num | Int | 0 | 专家总数 |
| drop_pad_mode | Int | 0 | 0=dropless, 1=drop/pad |
| expert_tokens_count_or_cumsum_flag | Int | 0 | 0=无, 1=cumsum, 2=count |
| expert_tokens_before_capacity_flag | Bool | false | 是否输出capacity前token数 |

**平台差异**: ascend310P 仅 FP16/FP32，不支持 BF16 和 Drop/Pad；ascend950 regbase 支持 INT64 expert_idx。

### 3.2 Tiling 策略 (moe_init_routing_v2_tiling.cpp)

**TilingData 层级**（tiling.h）:

```
MoeInitRoutingV2TilingData
├── coreNum                              : int64_t   # AIV核数
├── n                                    : int64_t   # 输入token行数
├── cols                                 : int64_t   # HiddenSize
├── k                                    : int64_t   # TopK专家数
├── expertCapacity                       : int64_t   # 专家容量
├── expertNum                            : int64_t   # 专家总数
├── dropPadMode                          : int64_t   # Drop/Pad模式
├── expertTokensCountOrCumsumFlag        : int64_t   # Token计数标志
├── expertTokensBeforeCapacityFlag       : int64_t   # Capacity前计数标志
├── vbsComputeParamsOp     : MoeV2VBSComputeTilingData        # 排序切分
├── vmsMiddleComputeParamsOp : MoeV2VMSMiddleComputeTilingData # 多核中间归并
├── sortOutComputeParamsOp : MoeV2SortOutComputeTilingData    # 归并输出
├── srcToDstComputeParamsOp : MoeV2GatherOutComputeTilingData  # 源到目的映射(dropless)
├── srcToDstCapacityComputeParamsOp : MoeV2GatherOutComputeTilingData # 源到目的映射(droppad)
└── gatherOutComputeParamsOp : MoeV2GatherOutComputeTilingData  # 数据搬运切分
```

**MoeV2VBSComputeTilingData**（10 字段）: `needCoreNum`, `perCoreElements`, `perCoreLoops`, `perCorePerLoopElements`, `perCoreLastLoopElements`, 尾核参数（`lastCoreElements/Loops/PerLoopElements/LastLoopElements`）, `oneLoopMaxElements`。

**MoeV2GatherOutComputeTilingData**（13 字段）: 核分配（`needCoreNum/activateRows`），行切分（`perCoreRows/PerLoopRows/LastLoopRows` 等），列切分（`perLoopCols/lastLoopCols/colLoops`）。复用于 srcToDst/dropPad/srcToDstCapacity 三处。

**关键常量**（tiling.cpp 第 19-60 行）:

| 常量 | 值 | 说明 |
|------|----|------|
| HIST_REGBASE_MAX_EXPERT_NUM | 256 | regbase 直方图路径专家数上限 |
| MRG_LIST_NUM | 4 | 四路归并 |
| SORT32_ALIGN_ELEMENT | 32 | 排序对齐粒度 |
| MAX_COLS_ONE_LOOP | 16376 | 内循环最大列数 |
| ONE_CORE_SORT_BUFFER | 6 | FullLoad 排序空间系数（普通） |
| ONE_CORE_SORT_BUFFER_310P | 24 | FullLoad 排序空间系数（310P） |
| SIMT_UB_SIZE_BYTE | 40960 | SIMT UB 预留大小 |

**TilingKey 编码**（tiling.cpp 第 19-27 行）:

| Key | 含义 |
|-----|------|
| 10001 | Dropless + 单核排序 |
| 10002 | Dropless + 多核排序 |
| 11001 | Dropless + HistRegBase + 单核排序 |
| 11002 | Dropless + HistRegBase + 多核排序 |
| 10011 | DropPad + 单核排序 |
| 10012 | DropPad + 多核排序 |
| 11011 | DropPad + HistRegBase + 单核排序 |
| 11012 | DropPad + HistRegBase + 多核排序 |
| 20000 | FullLoad 高性能模式（仅 310P） |

**TilingKey 选择逻辑**（第 388-413 行）:

```
GetTilingKey()
     |
  [isFullLoad?] ──Yes──> 20000
     |
    No
     |
  SetScheduleMode(1)
  histWithRegBase = regBase && expertNum <= 256
     |
  [dropPadMode == 0?]
     ├── Yes: [totalLength <= sortLoopMax?]
     │         ├── Yes → histWithRegBase ? 11001 : 10001
     │         └── No  → histWithRegBase ? 11002 : 10002
     └── No:  [totalLength <= sortLoopMax?]
               ├── Yes → histWithRegBase ? 11011 : 10011
               └── No  → histWithRegBase ? 11012 : 10012
```

**FullLoad 判断**（第 694-715 行）: 四条件全部满足时启用 — `totalLength <= sortLoopMaxElement`、`cols <= 16376`、`dropPadMode == 0`、UB 剩余空间足够。

**Tiling 主流程**:
```
GetPlatformInfo() → GetShapeAttrsInfo() → DoOpTiling()
  ├── Tiling4VBSCompute()         # 排序切分（单核/多核）
  ├── Tiling4VMSMiddleCompute()   # 归并中间切分
  ├── Tiling4SortOutCompute()     # 排序输出切分
  └── Tiling4GatherOutCompute()   # 数据搬运切分（含SrcToDst）
→ GetWorkspaceSize() → PostTiling()
```

### 3.3 Shape 推导 (moe_init_routing_v2_infershape.cpp)

```
expandedRowIdx:                [N*K] (始终 1D)
expanded_x (Dropless):         [min(N*K, activeNum), cols]
expanded_x (DropPad):          [expertNum, expertCapacity, cols]  (3D)
expert_tokens_count_or_cumsum: [expertNum] (仅 Dropless + flag > 0)
expert_tokens_before_capacity: [expertNum] (仅 DropPad + flag == true)
```

**校验约束**（第 87-150 行）: x/expert_idx 必须为 2D，dim-0 一致；DropPad 模式要求 `expertCapacity >= 1` 且 `expertNum >= 1`；`expertCapacity` 不得超过 x 的 dim-0。支持动态 Shape (-1) 和 UnknownRank (-2)。

### 3.4 ACLNN API (op_api/aclnn_moe_init_routing_v2.cpp)

**两段式接口**:

```cpp
// 第一段
aclnnStatus aclnnMoeInitRoutingV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expertIdx,
    int64_t activeNumOptional, int64_t expertCapacityOptional,
    int64_t expertNumOptional, int64_t dropPadModeOptional,
    int64_t expertTokensCountOrCumsumFlagOptional,
    bool expertTokensBeforeCapacityFlagOptional,
    const aclTensor* expandedXOut, const aclTensor* expandedRowIdxOut,
    const aclTensor* expertTokensCountOrCumsumOutOptional,
    const aclTensor* expertTokensBeforeCapacityOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor);

// 第二段
aclnnStatus aclnnMoeInitRoutingV2(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

ACLNN 实现为薄转发层，直接调用 `aclnnInnerMoeInitRoutingV2` 内部实现。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (moe_init_routing_v2.cpp)

入口函数（第 42-46 行）: AIC 核直接返回，仅 AIV 核执行。三架构编译分支:

```
moe_init_routing_v2(x, expertIdx, expandedX, expandedRowIdx, ...)
    ├── __NPU_ARCH__==3510: arch35 SIMT路径（ascend950）
    ├── __CCE_AICORE__==200: 310P路径（FullLoad + ResetIO双遍排序）
    └── else: 标准membase路径（910b/910_93）
```

**标准 membase 管线**（9 TilingKey 分支）:

| Key | 排序 | ExpertTokenOut | SrcToDst | Gather |
|-----|------|---------------|----------|--------|
| 10001/10011 | SortOneCore | 有条件 | Dropless:SrcToDstOp / DropPad:WithCapacity | GatherOut |
| 10002/10012 | SortMultiCore | 有条件 | 同上 | GatherOut |
| 11001/11011 | SortOneCore | 有条件 | 同上 | GatherOut |
| 11002/11012 | SortMultiCore | 有条件 | 同上 | GatherOut |
| 20000 | FullLoad（融合排序+映射+Gather） | 内含 | 内含 | 内含 |

### 4.2 核心计算流程

#### 排序阶段

**单核排序**（moe_v2_sort_one_core.h）: 仅 block0 执行。
```
CopyIn(expertIdx) → Cast→FP32 + Muls(-1) → Pad(MIN_FP32)
→ Concat + Sort<float,true> → Extract(sortedExpert, dstToSrcRow)
→ Neg + Cast→INT32 → CopyOut → SyncAll()
```
310P 路径额外支持 `ResetIO` 双遍排序：第一遍输出 sortedExpertIdx，ResetIO 后第二遍输出 expandedRowIdx。

**多核排序**（moe_v2_sort_multi_core.h）: 三阶段。
```
VBSProcess(): 每核独立排序自己的分块（多loop时本地归并）
VMSProcess(): 跨核四路归并，workspace双缓冲ping-pong，每层SyncAll
SortOutProcess(): 最终归并+Extract（仅block0）
```

#### ExpertTokenOut 阶段（moe_v2_expert_token_out.h）

统计每个专家的 token 数量:
```
读取sortedExpertIds → 追踪当前expert和计数
→ 按flag输出: 1=cumsum, 2=count
→ DropPad模式时初始化expandedRowIdx为-1
```
arch35 regbase 版本使用 DHist/CHist 直方图指令加速；SIMT 版本使用 2048 线程并行计算首索引+计数+cumsum。

#### SrcToDst 阶段

**Dropless**（moe_v2_src_to_dst_op.h）: scatter 逆映射，读 dstToSrc 写 srcToDst。
**DropPad**（moe_v2_src_to_dst_op_with_capacity.h）: 按 expertCapacity 分配槽位，超容量 token 丢弃。
**SIMT 版本**（arch35/）: `Simt::VF_CALL` 2048 线程并行逆映射。

#### Gather 阶段（moe_v2_gather_out.h）

```
读取expandedRowIdx(目的索引) → 按索引从x中Gather行 → 写入expandedX
支持colLoops列切分（大cols场景）
跳过 index==-1 或 index>=activateRows 的无效行
```

### 4.3 数据结构与内存布局

**MoeV2SortBase**（moe_v2_sort_base.h）: 排序基类，含 `sortDataCopyInQueue`(VECIN), `sortDataCopyOutQueue`(VECOUT), `tempBuffer`/`sortedBuffer`(VECCALC), 5 个 GlobalTensor（`expertIdxGm`, `sortedexpertIdxGm`, `expandDstToSrcRowGm`, `expertTokensCountOrCumsumGm`, `expertTokensBeforeCapacityGm`）。310P 额外含 `syncTmpSpaceGm_`。

**公共常量**（moe_v2_common.h）: `MIN_FP32=-3.4e38f`, `ONE_REPEAT_SORT_NUM=16(310P)/32(其他)`, `BLOCK_BYTES=32`, `ASSIST_NUM=256`, `MAX_EXPERT_NUM=5120`, `MERGE_LIST_TWO/THREE/FOUR=2/3/4`。

### 4.4 多核并行策略

- **单核排序**: `totalLength <= sortLoopMaxElement` 时 block0 独立排序
- **多核排序**: 核数取最近的 4 的幂次（4/16/64），VBS 各核独立 → VMS 四路归并树 → SortOut block0
- **Gather**: 按行均匀分核，列方向 colLoops 切分
- **SIMT 映射**: arch35 使用 2048 线程并行完成逆映射和 expert token 计数

### 4.5 通信模式

- **单核排序**: 完成后一次 `SyncAll()`
- **多核排序**: VBS 各核独立 → VMS 每层归并 `SyncAll()` → SortOut 仅 block0
- **ExpertTokenOut**: 使用 `SetAtomicAdd<int32_t>` 原子写入
- **SrcToDst**: Dropless 模式无 SyncAll；DropPad 模式依赖 ExpertTokenOut 已初始化 -1
- **Gather**: 无 SyncAll，按核间行切分独立执行

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 层级（9 个基础 int64_t + 6 个嵌套 TilingData）。通过 `BEGIN_TILING_DATA_DEF`/`END_TILING_DATA_DEF` 宏定义，Kernel 侧 `GET_TILING_DATA` 解包。

**CompileInfo**（tiling.h）: 空结构（平台信息通过 `PlatformAscendC` 运行时获取）。

### 5.2 Workspace 布局

```
非regbase:
┌──────────────────────────────────────────────────────────────┐
│ sortWorkspace = sizeof(float) * totalLength * 2 * 3          │
│ (排序中间缓冲: expert+row float pair，三缓冲轮转)            │
├──────────────────────────────────────────────────────────────┤
│ scatterWorkspace = sizeof(int32_t) * totalLength * 2         │
│ (scatter映射中间结果)                                         │
├──────────────────────────────────────────────────────────────┤
│ expertTokenFlagWorkspace = aivNum * 2 * sizeof(int32_t)      │
│ (expert token计数同步标记)                                    │
├──────────────────────────────────────────────────────────────┤
│ System Workspace = 16 MB                                      │
│ (AscendC框架内部使用)                                          │
└──────────────────────────────────────────────────────────────┘

regbase额外:
├── expertNum * sizeof(int32_t)   (expert token计数缓冲)
├── 16 KB                         (regbase内部空间)

310P额外:
├── sortWorkspace = sizeof(float) * totalLength * 8 * 3 + 8192  (更大排序缓冲)
```

PostTiling 设置 `SetScheduleMode(1)` 使排序独占全核。

---

## 6. 完整数据流

### Dropless 模式

```
  x [N, cols]              expert_idx [N, K]
       │                          │
       │                    ┌─────┴─────┐
       │                    │           │
       │               [Sort]      [Sort]
       │            (One/Multi    (HistRegBase
       │             Core)        可选)
       │                    │           │
       │                    └─────┬─────┘
       │                          │
       │               sorted_expert_idx (N*K,)
       │               dst_to_src_row (workspace)
       │                          │
       │              ┌───────────┼───────────┐
       │              │           │           │
       │         [ExpertToken   [SrcToDst
       │          Out](可选)    Op]
       │              │           │
       │              │     src_to_dst_row
       │              │     expanded_row_idx
       │              │           │
       └──────────────┼───────────┘
                      │
              [GatherOut]
              (按索引搬运x)
                      │
              ┌───────┴──────────┐
              │                  │
      expanded_x           expanded_row_idx
      [activeNum*K, cols]   [N*K]
```

### DropPad 模式

```
  x [N, cols]              expert_idx [N, K]
       │                          │
       │                    [Sort]
       │                          │
       │              sorted_expert_idx, dst_to_src_row
       │                          │
       │         ┌────────────────┼────────────────┐
       │         │                │                │
       │    [ExpertToken    [SrcToDstWith      [ExpertTokens
       │     Out]           Capacity]          BeforeCapacity]
       │         │                │                │
       │         │    expanded_row_idx            │
       │         │    (含-1丢弃标记)              │
       │         │                │                │
       └─────────┼────────────────┘                │
                 │                                  │
         [GatherOut]                                │
         (跳过-1行)                                 │
                 │                                  │
         ┌───────┴───────┐              ┌──────────┘
         │               │              │
 expanded_x        expanded_row_idx  expert_tokens
 [E, C, cols]       [N*K]          _before_capacity [E]
```

---

## 7. 性能相关设计

1. **HistRegBase 直方图优化**: 当 `regBase==true` 且 `expertNum<=256` 时，使用 DHist/CHist 直方图指令替代逐元素比较统计专家 token 数，大幅减少计数阶段耗时。

2. **FullLoad 融合优化**: 数据量小时排序+映射+Gather+ExpertTokenCount 合并为单核一次完成（Key=20000），消除中间写回 GM 开销。仅 310P 启用。

3. **310P ResetIO 双遍排序**: 310P 单核排序使用 `ResetIO` 在同一 Kernel 内执行两遍排序（第一遍输出 sortedExpertIdx，第二遍输出 expandedRowIdx），避免额外 Kernel 启动。

4. **SIMT 多线程映射**: arch35 使用 `Simt::VF_CALL` 启动 2048 线程并行完成逆映射和 expert token 计数，替代向量引擎的标量 `Adds` 操作。

5. **四路归并树**: 多核排序使用 `MRG_LIST_NUM=4`，核数为 4 的幂次，归并层数 = log4(核数)，最少化 SyncAll 次数。workspace 双缓冲 ping-pong 减少写冲突。

6. **DropPad 容量映射**: `MoeV2SrcToDstWithCapacity` 按 expertCapacity 分配固定槽位，超容量 token 标记为 -1，Gather 阶段跳过无效行。

7. **AtomicAdd 原子写入**: ExpertTokenOut 使用 `SetAtomicAdd<int32_t>` 原子累加，避免多核写冲突。

---

## 8. 依赖关系

```
op_host/moe_init_routing_v2_def.cpp
└── register/op_def_registry.h

op_host/moe_init_routing_v2_tiling.cpp
├── moe_init_routing_v2_tiling.h
│   ├── register/op_impl_registry.h
│   ├── register/tilingdata_base.h
│   └── tiling_base/tiling_base.h
├── moe_init_routing_v2_tiling_util.h
└── platform_ascendc::PlatformAscendC

op_host/moe_init_routing_v2_infershape.cpp
├── log/log.h
└── register/op_impl_registry.h

op_host/op_api/aclnn_moe_init_routing_v2.cpp
├── aclnn_moe_init_routing_v2.h
└── aclnnInnerMoeInitRoutingV2 (内部符号)

op_kernel/moe_init_routing_v2.cpp (标准路径)
├── moe_v2_init_routing_fullload.h
│   ├── moe_v2_sort_base.h
│   │   └── kernel_operator.h
│   └── moe_v2_common.h
├── one_core.h / multi_core.h
│   ├── moe_v2_sort_base.h
│   ├── moe_v2_mrgsort.h
│   └── moe_v2_mrgsort_out.h
├── expert_token_out.h
├── moe_v2_src_to_dst_op.h
├── moe_v2_src_to_dst_op_with_capacity.h
├── moe_v2_gather_out.h
└── kernel_operator.h

op_kernel/moe_init_routing_v2.cpp (arch35路径)
├── arch35/moe_v2_common.h
├── arch35/moe_v2_sort_one_core.h / multi_core.h
├── arch35/moe_v2_mrgsort.h / mrgsort_out.h
├── arch35/moe_v2_expert_token_out_regbase.h   # DHist/CHist
├── arch35/moe_v2_expert_token_out_simt.h      # SIMT计数
├── arch35/moe_v2_src_to_dst_op_simt.h
├── arch35/moe_v2_src_to_dst_with_capacity_simt.h
├── arch35/moe_v2_gather_out_for_simt.h
└── kernel_operator.h

op_graph/moe_init_routing_v2_proto.h
└── graph/operator_reg.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| drop_pad_mode | 0 或 1；DropPad 要求 expertCapacity>=1, expertNum>=1 |
| 输入维度 | x 和 expert_idx 必须为 2D，dim-0 相同 |
| expert_capacity | DropPad 模式下不得超过 x 的 dim-0 |
| expert_tokens_count_or_cumsum_flag | 0/1/2；>0 时要求 expertNum>0 |
| FullLoad 限制 | 仅 310P 启用，且 dropPadMode==0 |
| HistRegBase 限制 | regBase 平台且 expertNum<=256 |
| 多核核数 | 必须为 4 的幂次（4, 16, 64...） |
| data type | expert_idx: INT32（ascend950 额外支持 INT64） |
| 310P 限制 | 仅 FP16/FP32，不支持 BF16 和 Drop/Pad |
| expert_tokens_before_capacity | 仅 DropPad 模式下有效 |
| expanded_x 形状 | Dropless 为 2D, DropPad 为 3D [E, C, cols] |

---

## 10. 与同族变体的核心差异

### vs moe_init_routing (v1)

| 维度 | v1 | v2 |
|------|----|-----|
| row_idx 输入 | 外部传入 REQUIRED | 去除，自行推导 |
| Drop/Pad | 不支持 | 支持（3D 输出） |
| Expert Token Count | 不支持 | 支持 Count/Cumsum |
| expert_tokens_before_capacity | 不支持 | 支持 |
| HistRegBase 排序 | 无 | 有（regBase 平台，DHist/CHist） |
| SIMT 映射 | 无 | 有（arch35, 2048线程） |
| FullLoad | 有 | 有（仅 310P，更智能判断） |
| 310P 支持 | 无 | 有（受限） |
| ascend950 INT64 | 不支持 | 支持 expert_idx INT64 |

### vs moe_init_routing_v3

| 维度 | v2 | v3 |
|------|----|-----|
| 量化输出 | 无 | INT8/FP8/HIF8 |
| scale/offset 输入 | 无 | 有 |
| 多种量化模式 | 无 | PerToken/PerChannel/PerExpert |
| expert_tokens_count | Count/Cumsum | 含 KeyValue 变体 |
