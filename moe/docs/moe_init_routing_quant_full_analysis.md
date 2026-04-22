# moe_init_routing_quant 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeInitRoutingQuant`
**算子类**: MoE 路由初始化量化版（第一代）
**功能描述**: moe_init_routing 的量化版本。路由排序后 Gather 数据时按固定 scale/offset 量化为 INT8 输出。scale 和 offset 作为标量属性传入，不支持动态量化。

**支持平台**: ascend910b, ascend910_93
**支持数据类型**: FP16/BF16/FP32（输入）→ INT8（输出）× INT32（索引）

---

## 2. 目录结构与文件职责

```
moe_init_routing_quant/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_init_routing_quant_def.cpp             # 算子定义
│   ├── moe_init_routing_quant_tiling.h            # TilingData结构（5个嵌套）
│   ├── moe_init_routing_quant_tiling.cpp          # Tiling策略（4档位）
│   ├── moe_init_routing_quant_infershape.cpp      # Shape推导
│   └── config/ {ascend910b, ascend910_93}
├── op_kernel/
│   ├── moe_init_routing_quant.cpp                 # Kernel入口（4分支）
│   ├── moe_quant_common.h                         # 公共常量/assist数组
│   ├── moe_quant_sort_base.h                      # 排序基类
│   ├── moe_quant_sort_one_core.h                  # 单核排序
│   ├── moe_quant_sort_multi_core.h                # 多核排序
│   ├── moe_quant_mrgsort.h                        # 多路归并排序
│   ├── moe_quant_mrgsort_out.h                    # 归并排序输出（含Extract）
│   ├── moe_quant_src_to_dst_op.h                  # 源到目的映射
│   ├── moe_quant_gather_out.h                     # Gather+量化输出
│   └── moe_quant_gather_out_small_activate_row.h  # 小激活行Gather优化
├── tests/ut/ {op_host, op_kernel}
└── examples/ test_aclnn_moe_init_routing_quant.cpp
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_init_routing_quant_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | x | REQUIRED | FP16/BF16/FP32 | 2D: [N, cols] |
| 输入 | row_idx | REQUIRED | INT32 | 2D: [N, K] |
| 输入 | expert_idx | REQUIRED | INT32 | 2D: [N, K] |
| 输出 | expanded_x | REQUIRED | INT8 | 2D: [min(N, activeNum)*K, cols] |
| 输出 | expanded_row_idx | REQUIRED | INT32 | 1D: [N*K] |
| 输出 | expanded_expert_idx | REQUIRED | INT32 | 1D: [N*K] |

**属性**:

| 属性名 | 类型 | 说明 |
|--------|------|------|
| active_num | Int | 活跃 token 行数 |
| scale | Float | 量化缩放因子 |
| offset | Float | 量化偏移量 |

### 3.2 Tiling 策略 (moe_init_routing_quant_tiling.cpp)

**TilingData 层级**（tiling.h）:

```
MoeInitRoutingQuantTilingData
├── coreNum, n, cols, k          # 基本维度
├── scale                        : float   # 量化缩放
├── offset                       : float   # 量化偏移
├── vbsComputeParamsOp           : QuantVBSComputeTilingData        # 排序切分
├── vmsMiddleComputeParamsOp     : QuantVMSMiddleComputeTilingData  # 归并中间
├── sortOutComputeParamsOp       : QuantSortOutComputeTilingData    # 归并输出
├── srcToDstComputeParamsOp      : QuantGatherOutComputeTilingData  # SrcToDst映射
└── gatherOutComputeParamsOp     : QuantGatherOutComputeTilingData  # Gather+量化切分
```

**关键常量**（tiling.cpp）: `MRG_LIST_NUM=4`, `SORT32_ALIGN_ELEMENT=32`, `MAX_COLS_ONE_LOOP=16376`, `ASSIST_NUM=256`。

**TilingKey 编码**:

| Key | 排序 | Gather 策略 |
|-----|------|------------|
| 1 | 单核 | SplitN/SplitK |
| 2 | 多核 | SplitN/SplitK |
| 3 | 单核 | SmallActiveRow |
| 4 | 多核 | SmallActiveRow |

**Gather 策略选择**: `N/2 > activeNum` 时使用 SmallActiveRow（Key+2），否则根据 N 和 K 选择 SplitN 或 SplitK。

### 3.3 Shape 推导 (moe_init_routing_quant_infershape.cpp)

```
expanded_x:           [min(N, activeNum)*K, cols]
expanded_row_idx:     [N*K]
expanded_expert_idx:  [N*K]
```

### 3.4 ACLNN API

本算子无独立 ACLNN 发布接口，通过 GE 框架直接调用。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口 (moe_init_routing_quant.cpp)

```
moe_init_routing_quant(x, rowIdx, expertIdx, expandedX, expandedRowIdx, expandedExpertIdx, workspace, tiling)
├── 阶段1: Sort
│   Key 1/3: MoeSortOneCore
│   Key 2/4: MoeSortMultiCore
├── 阶段2: SrcToDst → MoeSrcToDstOp
└── 阶段3: Gather+量化
    Key 1/2: MoeGatherOut<DTYPE_X>
    Key 3/4: MoeGatherOutSmallActiveRow<DTYPE_X>
```

### 4.2 核心计算流程

**排序阶段**: 与 moe_init_routing 相同的 Sort→SrcToDst 管线。单核 BitonicSort 或多核 VBS+VMS+SortOut 四路归并。

**量化 Gather**: 三层数据类型处理:
- BF16: `bf16→fp32→fp16→Muls(scale)→Adds(offset)→int32→deq→fp16→int8`
- FP32: `fp32→fp16→Muls(scale)→Adds(offset)→int8`
- FP16: `fp16→Muls(scale)→Adds(offset)→int8`

### 4.3 数据结构与内存布局

**MoeSortBase**（moe_quant_sort_base.h）: 排序基类，含 `sortDataCopyInQueue`(VECIN), `sortDataCopyOutQueue`(VECOUT), `tempBuffer`/`sortedBuffer`(VECCALC), 4 个 GlobalTensor。

**公共常量**（moe_quant_common.h）: `MIN_FP32=-3.4e38f`, `ONE_REPEAT_SORT_NUM=32`, `BLOCK_BYTES=32`, `ASSIST_NUM=256`, `MERGE_LIST_TWO/THREE/FOUR=2/3/4`。全局 `assist[256]` 数组用于 SrcToDst 映射。

### 4.4 多核并行策略

- **排序**: 核数取最近的 4 的幂次，VBS 各核独立 → VMS 四路归并树
- **Gather**: SplitN（按行分核）/ SplitK（按 K 分核）/ SmallActiveRow（按激活行分核）
- **量化**: 三层循环（K/行/列），列方向 maxColsOneLoop 切分

### 4.5 通信模式

- **多核排序**: VMS 每层归并 `SyncAll()`
- **SrcToDst**: 完成后 `SyncAll()`
- **Gather+量化**: 无 SyncAll，各核独立

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 层级（6 个基础字段 + 5 个嵌套 TilingData）。CompileInfo 为空结构。

### 5.2 Workspace 布局

```
┌──────────────────────────────────────────────────────────────┐
│ sortWorkspace = sizeof(float) * totalLength * 2 * 3          │
├──────────────────────────────────────────────────────────────┤
│ coreSyncWorkspace = coreNum * 32 * 2                         │
├──────────────────────────────────────────────────────────────┤
│ scatterWorkspace = sizeof(int32_t) * totalLength             │
├──────────────────────────────────────────────────────────────┤
│ System Workspace = 16 MB                                      │
└──────────────────────────────────────────────────────────────┘
```

PostTiling 设置 `SetScheduleMode(1)` 使排序独占全核。

---

## 6. 完整数据流

```
  x [N, cols]    row_idx [N, K]    expert_idx [N, K]    attrs: scale, offset, active_num
       │               │                  │
       └───────────────┼──────────────────┘
                       │
              [Sort (by expert_idx)]
              (OneCore / MultiCore)
                       │
           sorted_expert_idx [N*K]
           dst_to_src_row (workspace)
                       │
              [SrcToDst 映射]
                       │
           expanded_row_idx [N*K] (src_to_dst)
                       │
              [Gather + 量化]
              x * scale + offset → INT8
                       │
              ┌────────┴──────────┐
              │                   │
      expanded_x [activeNum*K, cols]  expanded_expert_idx [N*K]
      (INT8)
```

---

## 7. 性能相关设计

1. **量化融合**: 量化操作与 Gather 合并在同一 Kernel 阶段，避免额外数据搬运。

2. **SmallActiveRow 优化**: 当 `activeNum << N` 时，逐行读取+顺序写，避免 scatter 随机写开销。

3. **SplitN/SplitK 自适应**: 根据数据维度选择最优切分策略。

4. **四路归并树**: 多核排序使用 `MRG_LIST_NUM=4`，最小化 SyncAll 次数。

---

## 8. 依赖关系

```
op_host/moe_init_routing_quant_def.cpp
└── register/op_def_registry.h

op_host/moe_init_routing_quant_tiling.cpp
├── moe_init_routing_quant_tiling.h
│   ├── register/op_impl_registry.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_kernel/moe_init_routing_quant.cpp
├── moe_quant_sort_one_core.h / moe_quant_sort_multi_core.h
│   ├── moe_quant_sort_base.h
│   │   └── kernel_operator.h
│   ├── moe_quant_mrgsort.h / moe_quant_mrgsort_out.h
│   └── moe_quant_common.h
├── moe_quant_src_to_dst_op.h
├── moe_quant_gather_out.h
├── moe_quant_gather_out_small_activate_row.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| 平台 | 仅 ascend910b/910_93，不支持 ascend950 |
| 量化方式 | 仅静态量化（scale+offset 标量属性） |
| 输出类型 | expanded_x 固定 INT8 |
| row_idx | 必须外部传入（与 v2 不同） |
| 多核核数 | 必须为 4 的幂次 |
| data type | row_idx/expert_idx 仅支持 INT32 |

---

## 10. 与同族变体的核心差异

| 维度 | quant (本算子) | quant_v2 | v2 | v3 |
|------|---------------|----------|-----|-----|
| row_idx 输入 | 外部传入 REQUIRED | 无（自行推导） | 无 | 无 |
| 量化方式 | 静态（标量属性） | 静态+动态（Tensor） | 无 | 静态+动态+MX FP8+HIF8 |
| 量化输出 | INT8 | INT8/INT4 | 无 | INT8/FP8/HIF8 |
| Drop/Pad | 不支持 | 支持 | 支持 | 支持 |
| Expert Token Count | 不支持 | 支持 | 支持 | 支持（含 KeyValue） |
| SmoothQuant | 不支持 | 支持 | 不支持 | 支持 |
| FullLoad | 不支持 | 支持 | 支持 | 支持 |
| regbase | 不支持 | 支持 | 支持 | 支持 |
