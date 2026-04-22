# moe_init_routing 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeInitRouting`
**算子类**: MoE 路由初始化
**功能描述**: 接收输入张量 `x`、行索引 `row_idx` 和专家索引 `expert_idx`，对专家路由表按专家ID排序，并按排序后的行索引从 `x` 中 Gather 数据到 `expanded_x`。这是 MoE 推理中路由分发的第一步——将 token 按专家分组排列。

**支持平台**: ascend910b, ascend910_93 (membase); ascend950 (regbase)
**支持数据类型**: FP16/BF16/FP32

---

## 2. 目录结构与文件职责

```
moe_init_routing/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_init_routing_def.cpp             # 算子定义
│   ├── moe_init_routing_tiling.h            # Tiling数据结构（5个嵌套结构）
│   ├── moe_init_routing_tiling.cpp          # Tiling策略（5档位切分）
│   ├── moe_init_routing_infershape.cpp      # Shape推导
│   └── config/ {ascend910b, ascend910_93, ascend950}
├── op_kernel/
│   ├── moe_init_routing.cpp                 # membase kernel入口（5分支）
│   ├── moe_init_routing_apt.cpp             # regbase kernel入口
│   ├── moe_common.h                         # 公共常量、工具函数
│   ├── moe_gather_out.h                     # 常规Gather输出模板
│   ├── moe_gather_out_small_activate_row.h  # 小激活行Gather优化
│   ├── moe_init_routing_fullload.h          # FullLoad模式Kernel
│   ├── moe_mrgsort.h                        # 多路归并排序
│   ├── moe_mrgsort_out.h                    # 归并排序输出处理
│   ├── moe_sort_base.h                      # 排序基类
│   ├── moe_sort_one_core.h                  # 单核排序
│   ├── moe_sort_multi_core.h                # 多核排序
│   ├── moe_src_to_dst_op.h                  # 源到目的映射
│   └── arch35/                              # ascend950 regbase专用
│       ├── moe_common.h, gather_out.h, sort_base.h
│       ├── moe_sort_one_core.h, moe_sort_multi_core.h
│       ├── moe_init_routing_fullload.h, moe_mrgsort.h
│       ├── moe_mrgsort_out.h, moe_src_to_dst_op.h
│       └── moe_src_to_dst_simt_op.h         # SIMT映射（仅arch35）
├── framework/ npu_moe_init_routing_onnx_plugin.cpp
├── op_graph/ moe_init_routing_proto.h
├── docs/ aclnnMoeInitRouting.md
├── examples/ test_aclnn_moe_init_routing.cpp
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_init_routing_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | x | REQUIRED | FP16/BF16/FP32 | 2D: [N, cols] |
| 输入 | row_idx | REQUIRED | INT32 | 2D: [N, K] |
| 输入 | expert_idx | REQUIRED | INT32 | 2D: [N, K] |
| 输出 | expanded_x | REQUIRED | 同 x | 2D: [min(N,active_num)*K, cols] |
| 输出 | expanded_row_idx | REQUIRED | INT32 | 1D: [N*K] |
| 输出 | expanded_expert_idx | REQUIRED | INT32 | 1D: [N*K] |

**属性**:

| 属性名 | 类型 | 说明 |
|--------|------|------|
| active_num | Int (必填) | 活跃token行数，用于裁剪输出 |

**平台配置**（第58-71行）: ascend910b/910_93 使用 `moe_init_routing` (membase)，ascend950 使用 `moe_init_routing_apt` (regbase)。

### 3.2 Tiling 策略 (moe_init_routing_tiling.cpp)

**TilingData 层级**（tiling.h 第72-82行）:

```
MoeInitRoutingTilingData
├── coreNum                : int64_t   # 可用AIV核数
├── n                      : int64_t   # 输入token行数
├── cols                   : int64_t   # HiddenSize
├── k                      : int64_t   # TopK专家数
├── vbsComputeParamsOp     : VBSComputeTilingData        # 排序阶段切分
├── vmsMiddleComputeParamsOp : VMSMiddleComputeTilingData # 多核中间归并
├── sortOutComputeParamsOp : SortOutComputeTilingData     # 归并排序输出
├── srcToDstComputeParamsOp: GatherOutComputeTilingData   # 源到目的映射
└── gatherOutComputeParamsOp: GatherOutComputeTilingData  # 数据搬运切分
```

**VBSComputeTilingData**（tiling.h 第28-39行）: `needCoreNum`, `perCoreElements`, `perCoreLoops`, `perCorePerLoopElements`, `perCoreLastLoopElements`, 尾核参数, `oneLoopMaxElements`。

**GatherOutComputeTilingData**（tiling.h 第52-70行）: 16个字段，含 `splitFlag`(0=SPLIT_K, 1=SPLIT_N, 2=SPLIT_ACTIVATE_ROW)。

**关键常量**（tiling.cpp 第18-37行）:

| 常量 | 值 | 说明 |
|------|----|------|
| MRG_LIST_NUM | 4 | 四路归并 |
| SORT32_ALIGN_ELEMENT | 32 | 排序对齐粒度 |
| MAX_COLS_ONE_LOOP | 16376 | 内循环最大列数 |
| SPLIT_K_THRESHOLD | 512 | K维切分阈值 |

**TilingKey 编码**（tiling.cpp 第413-686行）:

| Key | 含义 | 条件 |
|-----|------|------|
| 0 | FullLoad全载 | 单核排序 + 数据量小 |
| 1 | 单核排序 + 常规Gather | totalLength <= sortLoopMaxElement |
| 2 | 多核排序 + 常规Gather | 核数 = 4^x，四路归并树 |
| 3 | 单核排序 + SmallActiveRow | N/2 > activeNum |
| 4 | 多核排序 + SmallActiveRow | 多核 + N/2 > activeNum |

**Gather切分策略选择**（第667-686行）:
```
[单核排序?] → IsFullLoad()? → Yes → Key=0
    |                            |
   No                          No
    |                            |
[N/2 > activeNum?] → Yes → SmallActiveRow (Key+2)
    |
   No
    |
[N < coreNum && N < K?] → Yes → SplitK
    |
   No
    |
  SplitN
```

### 3.3 Shape 推导 (moe_init_routing_infershape.cpp)

```
输入: x [N, cols], row_idx [N, K], expert_idx [N, K]
输出:
  expanded_x:          [min(N, active_num) * K, cols]
  expanded_row_idx:    [N * K]
  expanded_expert_idx: [N * K]
```

支持动态 Shape (-1) 和 UnknownRank (-2)。输出 dtype: `expanded_x` 同 x，其余 INT32。

### 3.4 ACLNN API

本算子暂无独立 ACLNN API 发布接口，通过 GE 框架直接调用。如需 ACLNN 两阶段 API，请使用 `MoeInitRoutingV2` 或 `MoeInitRoutingV3`。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**membase 入口**（moe_init_routing.cpp 第27-101行）:

```
moe_init_routing(x, rowIdx, expertIdx, expandedX, expandedRowIdx, expandedExpertIdx, workspace, tiling)
    ├── KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0), AIC核直接返回
    ├── Key 0: MoeFullLoad<DTYPE_X>
    ├── Key 1: MoeSortOneCore → MoeSrcToDstOp → MoeGatherOut
    ├── Key 2: MoeSortMultiCore → MoeSrcToDstOp → MoeGatherOut
    ├── Key 3: MoeSortOneCore → MoeSrcToDstOp → MoeGatherOutSmallActiveRow
    └── Key 4: MoeSortMultiCore → MoeSrcToDstOp → MoeGatherOutSmallActiveRow
```

**regbase 入口**（moe_init_routing_apt.cpp）: 使用 arch35 头文件，`__NPU_ARCH__==3510` 时用 `MoeSrcToDstSimtOp` 替代 `MoeSrcToDstOp`。

### 4.2 核心计算流程

#### Sort Pipeline

**单核排序**（moe_sort_one_core.h）: 仅 block0 执行。
```
CopyIn(expert_idx + row_idx) → Cast→FP32 + Negate + Pad → Sort<float,true>
→ Extract(sorted_expert, dst_to_src) → CopyOut → SyncAll()
```

**多核排序**（moe_sort_multi_core.h）: 三阶段。
```
VBSProcess(): 每核独立排序自己的分块
VMSProcess(): 多阶段四路归并，每阶段 SyncAll()
SortOutProcess(): 最终归并输出（仅block0）
```

#### SrcToDst Pipeline（moe_src_to_dst_op.h）

将排序产生的 dst→src 逆映射转为 src→dst 正映射:
```
CopyIn(dst_to_src) → Compute(Adds生成顺序索引) → CopyOut(scatter写回)
→ SyncAll()
```
arch35 独有 `MoeSrcToDstSimtOp`：使用 `Simt::VF_CALL` 启动 2048 线程并行完成映射。

#### Gather Pipeline

**常规Gather**（moe_gather_out.h）: 按 splitFlag 选择 SPLIT_N/SPLIT_K 策略，三层循环(k/row/col)搬运数据。

**SmallActiveRow Gather**（moe_gather_out_small_activate_row.h）: 当 `activeNum << N` 时使用逆映射直接 Gather，避免 scatter 随机写开销。

**FullLoad**（moe_init_routing_fullload.h）: 数据量小时，排序+映射+Gather合并在一次操作完成。

### 4.3 数据结构与内存布局

**MoeSortBase**（moe_sort_base.h 第23行）: 排序基类，包含 `sortDataCopyInQueue`(VECIN), `sortDataCopyOutQueue`(VECOUT), `tempBuffer`/`sortedBuffer`(VECCALC), 4 个 GlobalTensor。

**MoeMrgsortParam**（moe_mrgsort.h 第24行）: `perListElements`, `lastListElements`, `oneLoopMaxElements`。

**公共常量**（moe_common.h）: `MIN_FP32=-3.4e38f`, `ONE_REPEAT_SORT_NUM=32`, `BLOCK_BYTES=32`, `ASSIST_NUM=256`, `MERGE_LIST_TWO/THREE/FOUR=2/3/4`。

### 4.4 多核并行策略

- **单核排序**: `totalLength <= sortLoopMaxElement` 时，数据一次装入 UB，block0 独立排序
- **多核排序**: 核数为 4 的幂次（4, 16, 64...），各核独立排序后四路归并树合并
- **Gather**: 按 SPLIT_N/SPLIT_K/SPLIT_ACTIVATE_ROW 三种策略分核搬运
- **尾核处理**: `lastCoreElements`/`lastCoreLoops` 单独配置

### 4.5 通信模式

- **单核排序**: 排序完成后一次 `SyncAll()`
- **多核排序**: VBS 阶段各核独立 → VMS 归并阶段每层 `SyncAll()` → SortOut 仅 block0
- **SrcToDst**: 完成后 `SyncAll()`
- **Gather**: 常规模式无 SyncAll；SmallActiveRow 模式多核时开始前 `SyncAll()`

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 层级（5 个嵌套 TilingData 结构 + 4 个基础 int64_t 字段）。通过 `BEGIN_TILING_DATA_DEF`/`END_TILING_DATA_DEF` 宏定义，Kernel 侧 `GET_TILING_DATA` 解包。

**CompileInfo**（tiling.h 第84行）: 空结构（平台信息通过 `PlatformAscendC` 运行时获取）。

### 5.2 Workspace 布局

```
workspaceSize = sortWorkspace + coreSyncWorkspace + scatterWorkspace + 16MB

┌─────────────────────────────────────────────────────────────┐
│ sortWorkspace     = sizeof(float) * totalLength * 2 * 3     │
│ (排序中间缓冲: expert+row 的 float pair，三缓冲轮转)        │
├─────────────────────────────────────────────────────────────┤
│ coreSyncWorkspace = coreNum * 32 * 2                        │
│ (核间同步标记)                                               │
├─────────────────────────────────────────────────────────────┤
│ scatterWorkspace  = sizeof(int32_t) * totalLength           │
│ (scatter映射中间结果)                                        │
├─────────────────────────────────────────────────────────────┤
│ System Workspace = 16 MB                                    │
│ (AscendC框架内部使用)                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 完整数据流

```
  x [N, cols]   row_idx [N, K]   expert_idx [N, K]   attr: active_num
       │               │                  │
       v               v                  v
  ┌─────────────────────────────────────────────────────────┐
  │ Host: Tiling策略选择 → TilingKey(0-4) + TilingData     │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │ 阶段1: Sort (按 expert_idx 排序)                         │
  │                                                         │
  │  expert_idx(N,K) + row_idx(N,K)                        │
  │         │                                               │
  │    [单核/多核排序]  (VBS → VMS → SortOut)               │
  │         │                                               │
  │    expanded_expert_idx (N*K,)                           │
  │    dst_to_src_row (workspace中间结果)                    │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │ 阶段2: SrcToDst (源到目的映射)                           │
  │                                                         │
  │  dst_to_src_row → [映射转换] → expanded_row_idx (N*K,)  │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │ 阶段3: Gather (数据搬运)                                 │
  │                                                         │
  │  x(N,cols) + expanded_row_idx(N*K,)                     │
  │         │                                               │
  │    [按行索引Gather]                                      │
  │         │                                               │
  │    expanded_x (min(N,active_num)*K, cols)                │
  └─────────────────────────────────────────────────────────┘
```

---

## 7. 性能相关设计

1. **FullLoad 融合优化**: 数据量小时排序+映射+Gather 合并为一次操作，消除中间写回 GM 开销。

2. **三段流水线**: Sort → SrcToDst → Gather 各阶段通过 workspace 中间 buffer 传递数据，减少全局内存访问。

3. **双缓冲 Gather**: `MoeGatherOut` 使用 `BUFFER_NUM=2` 双缓冲，一个 buffer 执行 DMAs 搬运时另一个同时计算，实现搬运与计算重叠。

4. **四路归并树**: 多核排序使用 `MRG_LIST_NUM=4` 四路归并，核数为 4 的幂次（4/16/64），归并层数 = log4(核数)，最少化 SyncAll 次数。

5. **SIMT 映射加速**: ascend950 的 `MoeSrcToDstSimtOp` 使用 `Simt::VF_CALL` 启动 2048 线程并行完成映射，替代向量引擎的标量 `Adds` 操作。

6. **SmallActiveRow 优化**: 当 `activeNum << N` 时（大量 token 被丢弃），使用逆映射直接 Gather 避免无效 scatter 写。

7. **SplitK/SplitN 自适应**: Gather 阶段根据 N 和 K 的相对大小选择切分维度，优化 UB 利用率。

---

## 8. 依赖关系

```
op_host/moe_init_routing_def.cpp
└── register/op_def_registry.h

op_host/moe_init_routing_tiling.cpp
├── moe_init_routing_tiling.h
│   ├── register/op_impl_registry.h
│   ├── register/tilingdata_base.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/moe_init_routing_infershape.cpp
├── log/log.h
└── register/op_impl_registry.h

op_kernel/moe_init_routing.cpp (membase)
├── moe_init_routing_fullload.h
│   ├── moe_sort_base.h
│   │   └── kernel_operator.h
│   └── moe_common.h
├── moe_sort_one_core.h / moe_sort_multi_core.h
│   ├── moe_sort_base.h
│   ├── moe_mrgsort.h
│   └── moe_mrgsort_out.h
├── moe_src_to_dst_op.h
├── moe_gather_out.h
├── moe_gather_out_small_activate_row.h
└── kernel_operator.h

op_kernel/moe_init_routing_apt.cpp (regbase)
├── arch35/moe_init_routing_fullload.h
├── arch35/moe_sort_one_core.h / moe_sort_multi_core.h
├── arch35/moe_src_to_dst_op.h
├── arch35/moe_src_to_dst_simt_op.h  (仅 arch35)
├── arch35/moe_gather_out.h
├── arch35/moe_gather_out_small_activate_row.h
└── kernel_operator.h

op_graph/moe_init_routing_proto.h
└── graph/operator_reg.h

framework/npu_moe_init_routing_onnx_plugin.cpp
├── onnx_common.h
└── nlohmann/json.hpp
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| active_num | 必填属性，>= 0 |
| 输入维度 | x/row_idx/expert_idx 必须为 2D（或 UnknownRank -2） |
| dim-0 一致性 | x/row_idx/expert_idx 的 dim-0 必须相同 |
| dim-1 一致性 | row_idx/expert_idx 的 dim-1 必须相同 |
| 多核核数 | 必须为 4 的幂次（4, 16, 64...） |
| data type | expert_idx/row_idx 仅支持 INT32 |
| regbase 排序差异 | arch35 的 NPU_ARCH==3510 跳过 Concat（Sort直接处理） |

---

## 10. 与同族变体的核心差异

| 维度 | v1 (本算子) | v2 | v3 | quant | quant_v2 |
|------|------------|-----|-----|-------|----------|
| 输入 | x+row_idx+expert_idx | x+expert_idx | x+expert_idx+scale+offset | x+row_idx+expert_idx | x+expert_idx+scale+offset |
| 量化输出 | 无 | 无 | INT8/FP8/HIF8 | INT8 | INT8/INT4+动态量化 |
| Drop/Pad | 不支持 | 支持 | 支持 | 不支持 | 支持 |
| Expert Token Count | 不支持 | 支持 | 支持(含KeyValue) | 不支持 | 支持 |
| row_idx_type | 仅Gather | 仅Gather | Gather/Scatter | 仅Gather | 仅Gather |
| FullLoad | 支持 | 支持 | 支持(含多种量化) | 不支持 | 支持 |
| 平台 | 910b/910_93/950 | +310P | 910b/910_93/950 | 910b/910_93 | 910b/910_93/950 |

**v1 核心特点**: 最基础的路由初始化算子，仅实现排序+Gather。需要外部传入 `row_idx`，不支持 Drop/Pad 和 expert_tokens_count 输出。
