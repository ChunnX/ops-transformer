# moe_init_routing_quant_v2 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeInitRoutingQuantV2`
**算子类**: MoE 路由初始化量化版（第二代）
**功能描述**: 在 quant v1 基础上，参考 v2 接口设计（去除 row_idx 输入），新增动态量化（运行时计算 scale）、Drop/Pad 支持、Expert Token Count 输出、SmoothQuant（1H/EH scale 模式）。支持 INT8 和 INT4 两种量化输出。

**支持平台**: ascend910b, ascend910_93, ascend950（regbase，含 SIMT/DHist 优化）
**支持数据类型**: FP16/BF16/FP32（输入）→ INT8/INT4（输出）× INT32（索引）

---

## 2. 目录结构与文件职责

```
moe_init_routing_quant_v2/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_init_routing_quant_v2_def.cpp             # 算子定义
│   ├── moe_init_routing_quant_v2_tiling.h            # TilingData结构（12基础+6嵌套）
│   ├── moe_init_routing_quant_v2_tiling.cpp          # Tiling策略（membase, 10 TilingKey）
│   ├── moe_init_routing_quant_v2_tiling_base.cpp     # Tiling基础（regbase 9 TilingKey）
│   ├── moe_init_routing_quant_v2_infershape.cpp      # Shape推导
│   ├── op_api/ (如有)
│   └── config/ {ascend910b, ascend910_93, ascend950}
├── op_kernel/
│   ├── moe_init_routing_quant_v2.cpp                 # membase Kernel入口（10分支）
│   ├── moe_init_routing_quant_v2_apt.cpp             # regbase Kernel入口（9分支）
│   ├── inner/                                        # 共享Kernel头文件
│   │   ├── moe_v2_common.h                           # 公共常量
│   │   ├── moe_v2_sort_base.h / one_core.h / multi_core.h  # 排序
│   │   ├── moe_v2_mrgsort.h / mrgsort_out.h         # 归并排序
│   │   ├── expert_token_out.h                        # 专家Token计数
│   │   ├── moe_v2_src_to_dst_op.h                    # SrcToDst映射
│   │   ├── moe_v2_src_to_dst_with_capacity.h         # DropPad SrcToDst
│   │   ├── moe_v2_src_to_dst_and_gather.h            # DropPad+动态量化融合
│   │   ├── moe_v2_gather_quant.h                     # 静态量化Gather
│   │   ├── moe_v2_gather_dynamic_quant.h             # 动态量化Gather
│   │   ├── moe_v2_fullload_quant_base.h              # FullLoad量化基类
│   │   ├── moe_v2_fullload_quant.h                   # 静态量化FullLoad
│   │   └── moe_v2_fullload_dynamic_quant.h           # 动态量化FullLoad
│   └── arch35/                                       # ascend950 regbase专用
│       ├── sort/mrgsort变体, SIMT映射
│       └── 量化Gather变体（static/dynamic/droppad）
├── docs/ aclnnMoeInitRoutingQuantV2.md
├── examples/
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_init_routing_quant_v2_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | x | REQUIRED | FP16/BF16/FP32 | 2D: [N, cols] |
| 输入 | expert_idx | REQUIRED | INT32 | 2D: [N, K] |
| 输入 | scale | OPTIONAL | FP32 | 静态: [1]; 动态: [1,cols] 或 [E,cols] |
| 输入 | offset | OPTIONAL | FP32 | [1] |
| 输出 | expanded_x | REQUIRED | INT8/INT4 | Dropless: [activeNum, cols]; DropPad: [E, C, cols] |
| 输出 | expanded_row_idx | REQUIRED | INT32 | 1D: [N*K] |
| 输出 | expert_tokens_count_or_cumsum | OPTIONAL | INT32 | 1D: [expertNum] |
| 输出 | expert_tokens_before_capacity | OPTIONAL | INT32 | 1D: [expertNum] |
| 输出 | dynamic_quant_scale | OPTIONAL | FP32 | 1D: [activeNum] 或 [E*C] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| active_num | Int | 0 | 活跃 token 行数 |
| expert_capacity | Int | 0 | 专家容量（Drop/Pad） |
| expert_num | Int | 0 | 专家总数 |
| drop_pad_mode | Int | 0 | 0=dropless, 1=drop/pad |
| expert_tokens_count_or_cumsum_flag | Int | 0 | 0=无, 1=cumsum, 2=count |
| expert_tokens_before_capacity_flag | Bool | false | 是否输出 capacity 前计数 |
| quant_mode | Int | 0 | 0=静态量化, 1=动态量化 |

**6 种数据类型组合**: fp16→int8, bf16→int8, fp32→int8, fp16→int4, bf16→int4, fp32→int4。

### 3.2 Tiling 策略

**TilingData 层级**（tiling.h）:

```
MoeInitRoutingQuantV2TilingData
├── coreNum, n, cols, k          # 基本维度
├── expertCapacity, expertNum    # 专家配置
├── dropPadMode                  # Drop/Pad模式
├── expertTokensCountOrCumsumFlag
├── expertTokensBeforeCapacityFlag
├── smoothType                   # 0=无, 1=1xH, 2=ExH
├── histWithRegBase              # regbase直方图标记
├── tmpData
├── vbsComputeParamsOp           : InnerMoeV2VBSComputeTilingData
├── vmsMiddleComputeParamsOp     : InnerMoeV2VMSMiddleComputeTilingData
├── sortOutComputeParamsOp       : InnerMoeV2SortOutComputeTilingData
├── srcToDstComputeParamsOp      : InnerMoeV2GatherOutComputeTilingData
├── srcToDstCapacityComputeParamsOp : InnerMoeV2GatherOutComputeTilingData
└── gatherOutComputeParamsOp     : InnerMoeV2GatherOutComputeTilingData
```

**关键常量**:

| 常量 | 值 | 说明 |
|------|----|------|
| MAX_COLS_ONE_LOOP | 8192/16376 | FullLoad/常规列上限 |
| MAX_COLS_DYNAMIC_QUANT | 6144 | 动态量化列上限 |
| FULLLOAD_H_LIMIT | 7168 | FullLoad H 维上限 |
| DYNAMIC_QUANT_INT4_SYM_SCALE | 7.0 | INT4 对称量化缩放因子 |
| HIST_REGBASE_MAX_EXPERT_NUM | 256 | regbase 直方图专家上限 |

**Membase TilingKey 编码**:

```
Key = 10000 + quantMode*1000 + dropPadMode*100 + sortMode*10
quantMode: 0=静态, 1=动态
sortMode: 0=单核, 1=多核
```

| Key | quantMode | dropPadMode | 排序 |
|-----|-----------|-------------|------|
| 10000/10010 | 静态 | Dropless | 单核/多核 |
| 10100/10110 | 静态 | DropPad | 单核/多核 |
| 11000/11010 | 动态 | Dropless | 单核/多核 |
| 11100/11110 | 动态 | DropPad | 单核/多核 |
| 20000 | 静态 FullLoad | - | - |
| 21000 | 动态 FullLoad | - | - |

**Regbase TilingKey**: 复用 v2 的 10001/10002/11001/11002/10011/10012/11011/11012 + 20000（FullLoad），通过 `histWithRegBase` 标记选择 ExpertTokenOut 实现。

### 3.3 Shape 推导 (moe_init_routing_quant_v2_infershape.cpp)

```
expandedRowIdx:                [N*K]
expanded_x (Dropless):         [min(N*K, activeNum), cols]
expanded_x (DropPad):          [expertNum, expertCapacity, cols]
expert_tokens_count_or_cumsum: [expertNum] (仅 Dropless + flag > 0)
expert_tokens_before_capacity: [expertNum] (仅 DropPad + flag == true)
dynamic_quant_scale:           [activeNum] 或 [E*C] (仅动态量化)
```

**Scale/Offset 校验**: 静态量化时 scale/offset 为 REQUIRED（1D, dim=1）；动态量化时 scale OPTIONAL（2D, shape 为 (1,cols) 或 (expertNum,cols)），其第一维决定 smoothType。

### 3.4 ACLNN API

本算子无独立 ACLNN 发布接口，通过 GE 框架直接调用。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**Membase 入口**（moe_init_routing_quant_v2.cpp）: 10 个 TilingKey 分支。

```
moe_init_routing_quant_v2()
├── FullLoad路径 (Key=20000/21000)
│   ├── MoeV2FullLoadQuant: 排序+映射+Gather+静态量化
│   └── MoeV2FullLoadDynamicQuant: 排序+映射+Gather+动态量化
└── 常规路径 (Key=10000-11110)
    ├── Sort (OneCore/MultiCore)
    ├── ExpertTokenOut + SrcToDst
    │   DropPad+动态量化: MoeV2SrcToDstAndGather（融合，直接返回）
    │   其他: MoeV2SrcToDstOp / MoeV2SrcToDstWithCapacity
    └── Gather+量化
        静态: MoeV2GatherQuant
        动态Dropless: MoeV2GatherDynamicQuant
```

**Regbase 入口**（moe_init_routing_quant_v2_apt.cpp）: 使用 arch35 头文件，SIMT 映射 + DHist 直方图优化。

### 4.2 核心计算流程

**静态量化 Gather**（moe_v2_gather_quant.h）:
```
CopyIn(x行) → Cast→FP16 → Muls(scale) → Adds(offset) → Cast→INT8 → CopyOut(scatter写)
```

**动态量化 Gather**（moe_v2_gather_dynamic_quant.h）:
```
CopyIn(x行) → ReduceMax → Muls(1/127.0) → 计算 scale → x/scale → Cast→INT8
INT4: 使用 DYNAMIC_QUANT_INT4_SYM_SCALE=7.0 替代 127.0
Smooth: 支持 smoothType=1(1xH) 和 smoothType=2(ExH)
```

**DropPad+动态量化融合**（moe_v2_src_to_dst_and_gather.h）: SrcToDst 和 Gather 合并在同一阶段，直接输出量化结果和 scale。

**FullLoad 量化**: 排序+计数+映射+Gather+量化合并在单核一次完成。

### 4.3 数据结构与内存布局

**公共常量**（inner/moe_v2_common.h）: `MIN_FP32=-3.4e38f`, `ONE_REPEAT_SORT_NUM=32`, `BLOCK_BYTES=32`, `DYNAMIC_QUANT_INT4_SYM_SCALE=7.0`, `MAX_EXPERT_NUM=5120`。

**排序基类**: 与 v2 相同的 `MoeV2SortBase`，含 VBS/VMS/SortOut 三阶段。

### 4.4 多核并行策略

与 v2 相同: 排序核数取 4 的幂次，Gather 按行/列切分。动态量化额外使用 workspace 缓存 per-row scale（`aivNum * cols * sizeof(float)`）。

### 4.5 通信模式

与 v2 相同: 多核排序每层归并 `SyncAll()`，其他阶段无 SyncAll。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 层级（12 个基础字段 + 6 个嵌套 TilingData）。Membase 通过 `BEGIN_TILING_DATA_DEF` 宏定义。CompileInfo 为空结构。

### 5.2 Workspace 布局

```
┌──────────────────────────────────────────────────────────────┐
│ sortWorkspace = sizeof(float) * totalLength * 2 * 3          │
├──────────────────────────────────────────────────────────────┤
│ scatterWorkspace = sizeof(int32_t) * totalLength * 2         │
├──────────────────────────────────────────────────────────────┤
│ expertTokenFlagWorkspace = aivNum * 2 * sizeof(int32_t)      │
├──────────────────────────────────────────────────────────────┤
│ quantSrcWorkspace = aivNum * cols * sizeof(float)  (动态量化) │
├──────────────────────────────────────────────────────────────┤
│ expertIdxValueWorkspace = sizeof(int32_t) * aivNum * 2 (DropPad) │
├──────────────────────────────────────────────────────────────┤
│ System Workspace = 16 MB                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 完整数据流

### 静态量化模式

```
  x [N, cols]   expert_idx [N, K]   scale [1]   offset [1]
       │               │                │            │
       └───────────────┼────────────────┘            │
                       │                              │
              [Sort + SrcToDst]                       │
                       │                              │
           expanded_row_idx [N*K]                     │
                       │                              │
              [GatherQuant] ←─────────────────────────┘
              x * scale + offset → INT8/INT4
                       │
              expanded_x [activeNum, cols] (INT8/INT4)
```

### 动态量化模式

```
  x [N, cols]   expert_idx [N, K]   scale(可选smooth)
       │               │                │
       └───────────────┼────────────────┘
                       │
              [Sort + SrcToDst]
                       │
           expanded_row_idx [N*K]
                       │
              [GatherDynamicQuant]
              ReduceMax(x) → dynamic_scale
              x / scale → INT8/INT4
                       │
              ┌────────┴──────────┐
              │                   │
      expanded_x           dynamic_quant_scale
      (INT8/INT4)          (FP32)
```

---

## 7. 性能相关设计

1. **动态量化 per-row scale**: 运行时 ReduceMax 计算每行 scale，比静态量化精度更高。INT4 使用 `7.0` 作为对称量化因子。

2. **SmoothQuant**: 动态量化支持 smooth scale（1xH 或 ExH），将量化误差从重要维度转移到不重要维度。

3. **DropPad+动态量化融合**: `MoeV2SrcToDstAndGather` 将 SrcToDst 映射和动态量化 Gather 合并，减少中间 GM 写回。

4. **FullLoad 量化融合**: 数据量小时排序+映射+Gather+量化在单核完成。

5. **INT4 支持**: 6 种数据类型组合含 3 种 INT4 输出，使用对称量化（scale=7.0）。

6. **regbase SIMT+DHist**: ascend950 使用 2048 线程 SIMT 映射和 DHist/CHist 直方图指令加速 ExpertTokenCount。

7. **列分段动态量化**: 大 cols 时使用 workspace 缓存部分列量化结果（`CopyOutPartialXQuant`），避免 UB 溢出。

---

## 8. 依赖关系

```
op_host/moe_init_routing_quant_v2_def.cpp
└── register/op_def_registry.h

op_host/moe_init_routing_quant_v2_tiling.cpp (membase)
├── moe_init_routing_quant_v2_tiling.h
│   ├── register/op_impl_registry.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/moe_init_routing_quant_v2_tiling_base.cpp (regbase)
└── moe_init_routing_quant_v2_tiling.h

op_kernel/moe_init_routing_quant_v2.cpp (membase)
├── inner/moe_v2_sort_one_core.h / multi_core.h
│   ├── inner/moe_v2_sort_base.h
│   ├── inner/moe_v2_mrgsort.h / mrgsort_out.h
│   └── inner/moe_v2_common.h
├── inner/expert_token_out.h
├── inner/moe_v2_src_to_dst_op.h / _with_capacity.h
├── inner/moe_v2_src_to_dst_and_gather.h
├── inner/moe_v2_gather_quant.h
├── inner/moe_v2_gather_dynamic_quant.h
├── inner/moe_v2_fullload_quant.h / _dynamic_quant.h
└── kernel_operator.h

op_kernel/moe_init_routing_quant_v2_apt.cpp (regbase)
├── arch35/ sort/simt/quant 变体
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| quant_mode | 0=静态, 1=动态 |
| drop_pad_mode | 0 或 1；DropPad 要求 expertCapacity>=1, expertNum>=1 |
| 输入维度 | x 和 expert_idx 必须为 2D，dim-0 相同 |
| 静态量化 scale/offset | REQUIRED，1D [1]，FP32 |
| 动态量化 scale | OPTIONAL，2D (1,cols) 或 (expertNum,cols) |
| expanded_x 输出 | INT8 或 INT4（由 def.cpp 类型映射决定） |
| dynamic_quant_scale | 仅动态量化时输出 |
| FullLoad 条件 | totalLength<=sortLoopMax 且 cols<=8192 且 dropless |
| INT4 对称量化因子 | 7.0（`DYNAMIC_QUANT_INT4_SYM_SCALE`） |
| regbase 直方图 | expertNum<=256 时启用 |

---

## 10. 与同族变体的核心差异

| 维度 | quant v1 | quant_v2（本算子） | v2 | v3 |
|------|----------|-------------------|-----|-----|
| row_idx 输入 | REQUIRED | 无（自行推导） | 无 | 无 |
| 量化方式 | 静态（标量属性） | 静态+动态（Tensor） | 无 | 静态+动态+MX FP8+HIF8 |
| 量化输出 | INT8 | INT8/INT4 | 无 | INT8/FP8/HIF8 |
| Drop/Pad | 不支持 | 支持 | 支持 | 支持 |
| Expert Token Count | 不支持 | 支持 | 支持 | 支持（INT64, 含 KeyValue） |
| SmoothQuant | 不支持 | 支持（1H/EH） | 不支持 | 支持 |
| FullLoad | 不支持 | 支持（静态+动态） | 支持 | 支持 |
| regbase | 不支持 | 支持 | 支持 | 支持 |
| 动态量化 scale 输出 | 无 | dynamic_quant_scale | 无 | expanded_scale |
