# moe_init_routing_v3 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeInitRoutingV3`
**算子类**: MoE 路由初始化（第三代，功能最全面）
**功能描述**: 在 v2 基础上增加量化输出（静态/动态/MX FP8/HIF8）、部分专家路由（`active_expert_range`）、Scatter 模式行索引、专家 token 计数的 KeyValue 模式、GatherFirst 性能优化模板。是 init_routing 系列功能最完整的变体。

**支持平台**: ascend910b, ascend910_93, ascend950（regbase，含 FP8/HIF8 量化）
**支持数据类型**: INT8/FP16/BF16/FP32 × INT32；ascend950 额外支持 FP8_E5M2/FP8_E4M3FN/HIF8 输出

---

## 2. 目录结构与文件职责

```
moe_init_routing_v3/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_init_routing_v3_def.cpp             # 算子定义
│   ├── moe_init_routing_v3_tiling.h            # Tiling数据结构（22基础+8嵌套）
│   ├── moe_init_routing_v3_tiling.cpp          # Tiling策略（910b，六位十进制编码）
│   ├── moe_init_routing_v3_tiling_base.cpp     # Tiling基础工具
│   ├── moe_init_routing_v3_tiling_arch35.cpp   # arch35专用Tiling（7种量化factor）
│   ├── moe_init_routing_v3_infershape.cpp      # Shape推导
│   ├── op_api/
│   │   ├── aclnn_moe_init_routing_v3.h         # ACLNN API声明
│   │   ├── aclnn_moe_init_routing_v3.cpp       # ACLNN API实现
│   │   ├── moe_init_routing_v3.h               # L0 Op辅助头文件
│   │   └── moe_init_routing_v3.cpp             # L0 Op辅助实现
│   └── config/ {ascend910b, ascend910_93, ascend950}
├── op_kernel/
│   ├── moe_init_routing_v3.cpp                 # membase Kernel入口（30+分支）
│   ├── moe_init_routing_v3_apt.cpp             # regbase Kernel入口（24分支）
│   ├── moe_v3_common.h                         # 公共常量/工具
│   ├── moe_v3_sort_base.h / one_core.h / multi_core.h  # 排序
│   ├── moe_v3_mrgsort.h / _out.h / _performance.h      # 归并排序
│   ├── moe_v3_sort_actual_expert.h             # GatherFirst排序
│   ├── moe_v3_gather_sort_multi_core.h         # GatherSort多核
│   ├── expert_tokens_count.h                   # 专家Token计数
│   ├── moe_v3_row_idx_gather.h / _droppad.h / _droppad_dynamic.h  # 行索引映射
│   ├── moe_v3_gather_out.h / _static_quant.h / _dynamic_quant.h    # Gather输出
│   ├── moe_v3_gather_droppad.h / _droppad_static_quant.h           # DropPad Gather
│   ├── moe_v3_full_load.h / _base.h           # FullLoad基类
│   ├── moe_v3_full_load_unquantized.h          # 非量化FullLoad
│   ├── moe_v3_full_load_static_quant.h         # 静态量化FullLoad
│   ├── moe_v3_full_load_dynamic_quant.h        # 动态量化FullLoad
│   └── arch35/                                 # ascend950 regbase专用
│       ├── moe_init_routing_v3_arch35_tiling_def.h  # arch35 TilingData结构
│       ├── moe_v3_common.h, sort_base.h, sort_one_core.h, multi_core.h
│       ├── moe_v3_expert_tokens_count.h
│       ├── moe_v3_row_idx_gather.h             # SIMT行索引映射
│       ├── moe_v3_gather_out.h / _dynamic_quant.h
│       ├── moe_v3_gather_mxfp8_quant.h         # MX FP8量化Gather
│       ├── moe_v3_gather_hif8_quant.h          # HIF8 CAST量化
│       ├── moe_v3_gather_hif8_pertensor_quant.h # HIF8 PerTensor量化
│       └── moe_v3_gather_hif8_pertoken_quant.h  # HIF8 PerToken量化
├── op_graph/ moe_init_routing_v3_proto.h
├── docs/ aclnnMoeInitRoutingV3.md
├── examples/ (2个测试文件)
└── tests/ut/ {op_host, op_kernel}, tests/pytest/
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_init_routing_v3_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | x | REQUIRED | INT8/FP16/BF16/FP32 | 2D: [N, cols] |
| 输入 | expert_idx | REQUIRED | INT32 | 2D: [N, K] |
| 输入 | scale | OPTIONAL | FP32 | 多种（按量化模式） |
| 输入 | offset | OPTIONAL | FP32 | 多种 |
| 输出 | expanded_x | REQUIRED | INT8/FP16/BF16/FP32/FP8/HIF8 | Dropless: [activeNum, cols]; DropPad: [E, C, cols] |
| 输出 | expanded_row_idx | REQUIRED | INT32 | 1D: [N*K] |
| 输出 | expert_tokens_count_or_cumsum | REQUIRED | INT64 | 1D: [expertNum] 或 2D: [expertNum, 2] |
| 输出 | expanded_scale | REQUIRED | FP32/FP8_E8M0 | 按量化模式 |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| active_num | Int | -1 | 活跃 token 行数；-1=N*K |
| expert_capacity | Int | -1 | 专家容量（Drop/Pad） |
| expert_num | Int | -1 | 专家总数 |
| drop_pad_mode | Int | 0 | 0=dropless, 1=drop/pad |
| expert_tokens_num_type | Int | 0 | 0=cumsum, 1=count, 2=key_value |
| expert_tokens_num_flag | Bool | false | 是否输出 token 计数 |
| quant_mode | Int | -1 | -1=非量化, 0=静态, 1=动态, 2/3=MX FP8, 6=HIF8_CAST, 7/8=HIF8 |
| active_expert_range | ListInt | {} | 部分专家路由 [start, end) |
| row_idx_type | Int | 0 | 0=Gather 索引, 1=Scatter 索引 |

**量化模式**:

| quant_mode | 输出类型 | expanded_scale 类型 |
|------------|---------|-------------------|
| -1 (NON_QUANT) | 同 x | FP32 |
| 0 (STATIC_QUANT) | INT8 | FP32 |
| 1 (DYNAMIC_QUANT) | INT8 | FP32 |
| 2 (MX_FP8_E5M2) | FP8_E5M2 | FP8_E8M0 |
| 3 (MX_FP8_E4M3FN) | FP8_E4M3FN | FP8_E8M0 |
| 6 (HIF8_CAST) | HIFLOAT8 | FP32 |
| 7 (HIF8_PERTENSOR) | HIFLOAT8 | FP32 |
| 8 (HIF8_PERTOKEN) | HIFLOAT8 | FP32 |

### 3.2 Tiling 策略

**TilingData 层级**（tiling.h）:

```
MoeInitRoutingV3TilingData
├── coreNum, n, cols, k                        # 基本维度
├── expertStart, expertEnd, actualExpertNum    # 专家范围
├── quantMode, rowIdxType                      # 量化/索引模式
├── isInputScale, isInputOffset                # 可选输入标记
├── expertNum, expertTokensNumType/Flag        # 专家计数配置
├── gatherFirstFullload, ep                    # GatherFirst/EP标记
├── activeNum, dropPadMode, smoothType         # 模式配置
├── expertCountElements, expertCapacity        # 容量配置
├── vbsComputeParamsOp         : MoeV3VBSComputeTilingData        # 排序切分
├── vmsMiddleComputeParamsOp   : MoeV3VMSMiddleComputeTilingData  # 归并中间
├── sortOutComputeParamsOp     : MoeV3SortOutComputeTilingData    # 归并输出
├── expertTokensCountTilingDataOp : MoeV3ExpertTokensCountTilingData  # Token计数
├── gatherOutComputeParamsOp   : MoeV3GatherOutComputeTilingData  # Gather切分
├── srcToDstDropPadParamsOp    : MoeV3SrcToDstCapacityComputeTilingData  # DropPad映射
├── srcToDstDropPadDynamicParamsOp : MoeV3SrcToDstCapacityComputeTilingData  # 动态量化映射
└── srcToDstComputeParamsOp    : MoeV3SrcToDstComputeTilingData   # SrcToDst映射
```

**910b TilingKey 编码**（六位十进制，tiling.cpp 第 83-96 行）:

```
Key = 1000000 + sortMode*100000 + (quantMode+1)*10000 + rowIdxType*1000 + dropPadMode*100
sortMode: 0=单核, 1=多核, 2=GatherSort单核, 3=GatherSort多核
```

**特殊 FullLoad Key**:

| Key | 含义 |
|-----|------|
| 2000000 | 性能打点模板 |
| 2100000 | 非量化 FullLoad |
| 2200000 | 静态量化 FullLoad |
| 2300000+ | 动态量化 FullLoad（含 ep/scaleType 偏移） |

**arch35 TilingKey 编码**:

```
Key = 1000000 + sortMode*100000 + quantModeFactor*10000 + rowIdxType*1000
quantModeFactor: 0=非量化, 1=静态, 2=动态, 3=MXFP8, 7=HIF8_CAST, 8=HIF8_PERTENSOR, 9=HIF8_PERTOKEN
```

**关键常量**（tiling.cpp）:

| 常量 | 值 | 说明 |
|------|----|------|
| TILINGKEY_BASE | 1000000 | 基准偏移 |
| SORT_CORE_TILINGKEY_BASE | 100000 | 排序模式乘子 |
| QUANT_MODE_TILINGKEY_BASE | 10000 | 量化模式乘子 |
| PERFORMANCE_TILINGKEY | 2000000 | 性能打点 |
| MRG_LIST_NUM | 4 | 四路归并 |
| SORT32_ALIGN_ELEMENT | 32 | 排序对齐粒度 |
| MAX_COLS_ONE_LOOP | 16376 | 内循环最大列数 |

**GatherFirst 判断**（第 649-657 行）: 当 `ep==1` 且 `n>=100 && (expertEnd-expertStart)*8 <= expertNum` 时，先 Gather 有效专家再排序，减少排序数据量。

**性能模式判断**（第 731-748 行）: expertNum==256, expertRange<=32, BS∈[384,8192], TopK==8 时启用 GatherSort 性能优化。

### 3.3 Shape 推导 (moe_init_routing_v3_infershape.cpp)

```
expandedRowIdx:                [N*K]
expanded_x (Dropless):         [min(N*K, activeNum), cols]
expanded_x (DropPad):          [expertNum, expertCapacity, cols]  (3D)
expert_tokens_count_or_cumsum:
  key_value: [expertNum, 2]
  其他:      [expertEnd-expertStart]
expanded_scale:
  非量化/动态: [activeNum] 或 [expertNum*expertCapacity]
  MX FP8:     [N*K, ceil(ceil(cols/32)/2)*2]
  HIF8_PERTOKEN: [N*K]
  静态/HIF8_CAST: 不设置
```

### 3.4 ACLNN API (op_api/aclnn_moe_init_routing_v3.cpp)

**两段式接口**:

```cpp
// 第一段（18个参数）
aclnnStatus aclnnMoeInitRoutingV3GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expertIdx,
    const aclTensor* scaleOptional, const aclTensor* offsetOptional,
    int64_t activeNum, int64_t expertCapacity, int64_t expertNum,
    int64_t dropPadMode, int64_t expertTokensNumType,
    bool expertTokensNumFlag, int64_t quantMode,
    const aclIntArray* activeExpertRangeOptional, int64_t rowIdxType,
    const aclTensor* expandedXOut, const aclTensor* expandedRowIdxOut,
    const aclTensor* expertTokensCountOrCumsumOut,
    const aclTensor* expandedScaleOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

// 第二段
aclnnStatus aclnnMoeInitRoutingV3(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**Membase 入口**（moe_init_routing_v3.cpp）: 30+ 个 TilingKey 分支。

```
moe_init_routing_v3()
├── FullLoad路径 (Key=2000000/2100000/2200000/2300xxx)
│   └── MoeV3FullLoad / FullLoadDynamicQuant / FullLoadStaticQuant / FullLoadUnquantized
├── GatherFirst路径 (Key=1200000/1300000)
│   └── MoeSortActualExpert → MoeSortMultiCorePerformance
└── 常规路径 (Key=1000000-1120100)
    ├── Sort (OneCore/MultiCore)
    ├── ExpertTokensCount
    ├── RowIdxGather (if rowIdxType==GATHER)
    └── GatherOut (非量化/静态量化/动态量化/DropPad变体)
```

**Regbase 入口**（moe_init_routing_v3_apt.cpp）: 24 个 TilingKey 分支，支持 7 种量化模式。

### 4.2 核心计算流程

**四阶段流水线**:
```
Sort → ExpertTokensCount → RowIdxGather(可选) → GatherOut(含量化)
```

**排序阶段**: 单核 BitonicSort 或多核 VBS+VMS+SortOut 四路归并树，与 v2 相同。

**ExpertTokensCount**: 支持 CUMSUM/COUNT/KEY_VALUE 三种模式。KEY_VALUE 模式输出 [expertNum, 2]，第一列为 key（expert ID），第二列为 value（count）。

**RowIdxGather**: 当 `rowIdxType==GATHER` 时，将 Scatter 索引转为 Gather 索引（逆映射）。arch35 使用 SIMT `VF_CALL` 2048 线程。

**GatherOut + 量化**:
- **非量化**: 直接 CopyIn/CopyOut
- **静态量化**: x * scale + offset → INT8
- **动态量化**: AbsMax(x) → scale, x/scale → INT8
- **MX FP8**: Block 量化（32 元素一组），FP8 输出 + FP8_E8M0 scale
- **HIF8**: CAST/PERTENSOR/PERTOKEN 三种模式

**FullLoad 融合**: 数据量小时排序+计数+映射+Gather+量化合并在单核一次完成。

### 4.3 数据结构与内存布局

**MoeSortBase**（moe_v3_sort_base.h）: 排序基类，含 `sortDataCopyInQueue`(VECIN), `sortDataCopyOutQueue`(VECOUT), `tempBuffer`/`sortedBuffer`(VECCALC), 5 个 GlobalTensor。

**公共常量**（moe_v3_common.h）: `MIN_FP32=-3.4e38f`, `ONE_REPEAT_SORT_NUM=32`, `BLOCK_BYTES=32`, `MRGSORT_LIST_MAX_ELEMENT=2040`, `GATHER=0/SCATTER=1`, `NO_SCALE=0/SCALE_1H=1/SCALE_EH=2`。

### 4.4 多核并行策略

- **排序**: 核数取最近的 4 的幂次，VBS 各核独立 → VMS 四路归并树 → SortOut block0
- **GatherOut**: 按索引均匀分核，列方向 colLoops 切分
- **ExpertTokensCount**: 按 expertNum 均匀分核
- **arch35 SIMT**: 使用 `Simt::VF_CALL` 2048 线程并行逆映射

### 4.5 通信模式

- **多核排序**: VMS 每层归并 `SyncAll()`
- **其他阶段**: 无 SyncAll，各核独立执行
- **ExpertTokensCount**: 使用 `SetAtomicAdd<int64_t>` 原子写入

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 层级（22 个基础字段 + 8 个嵌套 TilingData）。910b 通过 `BEGIN_TILING_DATA_DEF` 宏定义；arch35 使用普通 C++ struct `MoeInitRoutingV3Arch35TilingData`。

**Tiling 注册优先级**: arch35 优先级 1000（ascend950 优先使用），910b 优先级 10000（fallback）。

**CompileInfo**（910b tiling.h）: `aivNum`, `ubSize`, `socVersion`（非空结构）。

### 5.2 Workspace 布局

```
┌──────────────────────────────────────────────────────────────┐
│ sortWorkspace = sizeof(float) * totalLength * 2 * 3          │
├──────────────────────────────────────────────────────────────┤
│ coreSyncWorkspace = coreNum * 32 * 2                         │
├──────────────────────────────────────────────────────────────┤
│ scatterWorkspace = sizeof(int32_t) * totalLength             │
├──────────────────────────────────────────────────────────────┤
│ expertTokensCountWS = sizeof(int32_t) * (expertEnd-expertStart) │
├──────────────────────────────────────────────────────────────┤
│ expertTokenTotalCountWS = sizeof(int32_t) (对齐)              │
├──────────────────────────────────────────────────────────────┤
│ quantTempWorkspace = aivNum * cols * sizeof(float)  (动态量化) │
├──────────────────────────────────────────────────────────────┤
│ expertIdxValueWS = sizeof(int32_t) * aivNum * 2  (DropPad)   │
├──────────────────────────────────────────────────────────────┤
│ System Workspace = 16 MB                                      │
└──────────────────────────────────────────────────────────────┘
```

PostTiling 设置 `SetScheduleMode(1)` 使排序独占全核。

---

## 6. 完整数据流

```
  x [N, cols]         expert_idx [N, K]   scale/offset(可选)   active_expert_range(可选)
       │                      │                    │                       │
       └──────────────────────┼────────────────────┼───────────────────────┘
                              │                    │
                              ▼                    │
  ┌───────────────────────────────────────────────┐│
  │ 阶段1: Sort (按 expert_idx 排序)              ││
  │                                               ││
  │  expert_idx(N,K) → [OneCore/MultiCore/        ││
  │                      GatherFirst]              ││
  │                       │                        ││
  │                       ├─> sorted_expert_idx    ││
  │                       └─> dst_to_src_row       ││
  └───────────────────────────────────────────────┘│
                              │                    │
                              ▼                    │
  ┌───────────────────────────────────────────────┐│
  │ 阶段2: ExpertTokensCount                      ││
  │                                               ││
  │  支持: CUMSUM / COUNT / KEY_VALUE             ││
  │  输出: INT64 (expertNum,) 或 (expertNum, 2)   ││
  └───────────────────────────────────────────────┘│
                              │                    │
                              ▼                    │
  ┌───────────────────────────────────────────────┐│
  │ 阶段3: RowIdxGather (仅 rowIdxType==0)        ││
  │                                               ││
  │  Scatter索引 → Gather索引 (逆映射)            ││
  └───────────────────────────────────────────────┘│
                              │                    │
                              ▼                    ▼
  ┌─────────────────────────────────────────────────────┐
  │ 阶段4: GatherOut + 量化                             │
  │                                                     │
  │  x(N,cols) + 映射表 + scale/offset                  │
  │       │                                             │
  │  [非量化/静态量化/动态量化/MXFP8/HIF8]                │
  │       │                                             │
  │       ├─> expanded_x (INT8/FP16/FP8/HIF8)           │
  │       └─> expanded_scale (FP32/FP8_E8M0)            │
  └─────────────────────────────────────────────────────┘
```

---

## 7. 性能相关设计

1. **GatherFirst 优化**: 部分专家路由（EP）时，先 Gather 有效专家数据再排序，减少排序数据量。条件: `n>=100 && range*8 <= expertNum`。

2. **性能打点模板**: 固定 shape (x=(1,7168), expert_idx=(1,8)) 的专用 FullLoad 路径，用于性能基准测试。

3. **七种量化模式**: 静态量化（scale+offset）、动态量化（运行时计算 scale）、MX FP8（Block 量化 32 元素/组）、HIF8 CAST/PERTENSOR/PERTOKEN，满足不同精度需求。

4. **KEY_VALUE 计数模式**: 输出 [expertNum, 2] 二维数组，同时记录 expert ID 和对应 count，方便后续查找。

5. **arch35 量化 Gather**: 每种量化模式独立实现 Gather kernel（MXFP8/HIF8_CAST/HIF8_PERTENSOR/HIF8_PERTOKEN），充分利用 MicroAPI 的量化指令。

6. **FullLoad 融合**: 数据量小时排序+计数+映射+Gather+量化合并在单核完成，消除中间 GM 写回。

7. **四路归并树**: 多核排序使用 `MRG_LIST_NUM=4`，核数为 4 的幂次，workspace 双缓冲 ping-pong。

---

## 8. 依赖关系

```
op_host/moe_init_routing_v3_def.cpp
└── register/op_def_registry.h

op_host/moe_init_routing_v3_tiling.cpp (910b)
├── moe_init_routing_v3_tiling.h
│   ├── register/op_impl_registry.h
│   ├── register/tilingdata_base.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/moe_init_routing_v3_tiling_arch35.cpp (arch35)
├── arch35/moe_init_routing_v3_arch35_tiling_def.h
└── tiling_base/tiling_base.h

op_host/moe_init_routing_v3_infershape.cpp
├── log/log.h
└── register/op_impl_registry.h

op_host/op_api/aclnn_moe_init_routing_v3.cpp
├── aclnn_moe_init_routing_v3.h
├── moe_init_routing_v3.h (L0 Op)
└── aclnnInnerMoeInitRoutingV3 (内部符号)

op_kernel/moe_init_routing_v3.cpp (membase)
├── moe_v3_sort_one_core.h / multi_core.h
│   ├── moe_v3_sort_base.h
│   │   └── kernel_operator.h
│   ├── moe_v3_mrgsort.h / mrgsort_out.h
│   └── moe_v3_common.h
├── expert_tokens_count.h
├── moe_v3_row_idx_gather.h / _droppad.h / _droppad_dynamic.h
├── moe_v3_gather_out.h / _static_quant.h / _dynamic_quant.h
├── moe_v3_gather_droppad.h / _droppad_static_quant.h
├── moe_v3_full_load*.h
└── kernel_operator.h

op_kernel/moe_init_routing_v3_apt.cpp (regbase)
├── arch35/moe_v3_sort_one_core.h / multi_core.h
├── arch35/moe_v3_expert_tokens_count.h
├── arch35/moe_v3_row_idx_gather.h
├── arch35/moe_v3_gather_out.h / _dynamic_quant.h
├── arch35/moe_v3_gather_mxfp8_quant.h
├── arch35/moe_v3_gather_hif8_quant.h / _pertensor_quant.h / _pertoken_quant.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| quant_mode | -1/0/1/2/3/6/7/8 |
| drop_pad_mode | 0 或 1；DropPad 要求 expertCapacity>=1, expertNum>=1 |
| 输入维度 | x 和 expert_idx 必须为 2D，dim-0 相同 |
| expert_tokens_count_or_cumsum | 固定 INT64 输出 |
| active_expert_range | 空列表或 [start, end)，start < end |
| row_idx_type | 0=Gather, 1=Scatter |
| GatherFirst 条件 | ep==1 且 n>=100 且 range*8<=expertNum |
| 性能模式条件 | expertNum==256, range<=32, BS∈[384,8192], TopK==8 |
| 多核核数 | 必须为 4 的幂次 |
| regbase 扩展类型 | FP8/HIF8 仅 ascend950 |
| expanded_x 形状 | Dropless: 2D; DropPad: 3D |

---

## 10. 与同族变体的核心差异

### vs moe_init_routing (v1) / v2

| 维度 | v1 | v2 | v3 |
|------|----|----|-----|
| 量化输出 | 无 | 无 | 静态/动态/MX FP8/HIF8 |
| scale/offset 输入 | 无 | 无 | Tensor 输入 |
| 部分专家路由 | 无 | 无 | active_expert_range |
| Scatter 索引 | 无 | 无 | row_idx_type=1 |
| Expert Token 计数类型 | 无 | Count/Cumsum (INT32) | Count/Cumsum/KeyValue (INT64) |
| GatherFirst 优化 | 无 | 无 | 有（先剔除无效专家） |
| 性能打点模板 | 无 | 无 | 有 |
| TilingKey 编码 | 简单数字 | 五位编码 | 六位十进制位编码 |
| FullLoad 量化 | 无 | 无 | 非量化/静态/动态三种 |
| SmoothQuant | 无 | 无 | 有（1H/EH scale 模式） |
| regbase 量化 | 无 | 无 | MX FP8 + HIF8 三种 |
