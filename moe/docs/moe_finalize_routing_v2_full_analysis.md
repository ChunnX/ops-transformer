# moe_finalize_routing_v2 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeFinalizeRoutingV2`
**算子类**: MoE 路由终结（第二代）
**功能描述**: 在 v1 基础上将所有非核心输入改为 OPTIONAL（x1/x2/bias/scales/expert_idx），新增 Drop/Pad 四种模式（DroplessColumn/DropPadColumn/DroplessRow/DropPadRow）、混合精度（BF16 输入+FP32 scale）、ascend950 regbase MicroAPI 路径。

**支持平台**: ascend910b, ascend910_93, ascend310P（FP16/FP32）, ascend950（regbase）, kirinx90/kirin9030（FP16/FP32）
**支持数据类型**: FP16/BF16/FP32 × INT32（索引）

---

## 2. 目录结构与文件职责

```
moe_finalize_routing_v2/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_finalize_routing_v2_def.cpp             # 算子定义
│   ├── moe_finalize_routing_v2_tiling.h            # TilingData结构（24字段+常量）
│   ├── moe_finalize_routing_v2_tiling.cpp          # Tiling基础（公共流程）
│   ├── moe_finalize_routing_v2_tiling_membase.cpp  # membase Tiling策略
│   ├── moe_finalize_routing_v2_tiling_arch35.cpp   # regbase Tiling策略
│   ├── moe_finalize_routing_v2_tiling_arch35.h     # regbase TilingData结构
│   ├── moe_finalize_routing_v2_infershape.cpp      # Shape推导
│   ├── op_api/
│   │   ├── aclnn_moe_finalize_routing_v2.h         # ACLNN API声明
│   │   └── aclnn_moe_finalize_routing_v2.cpp       # ACLNN API实现
│   └── config/ {ascend910b, ascend910_93, ascend950, ascend310p, kirinx90, kirin9030}
├── op_kernel/
│   ├── moe_finalize_routing_v2.cpp                 # membase Kernel入口（~50分支）
│   ├── moe_finalize_routing_v2_apt.cpp             # regbase Kernel入口（48分支）
│   ├── moe_finalize_routing_v2_common.h            # 公共常量
│   ├── moe_finalize_routing_v2_fp_db.h             # 双缓冲FullH
│   ├── moe_finalize_routing_v2_fp_cutk.h           # CutK（K>256）
│   ├── moe_finalize_routing_v2_fp_cuth.h           # CutH通用
│   ├── moe_finalize_routing_v2_fp_cuth_k2.h        # CutH+K2
│   ├── moe_finalize_routing_v2_fp_cuth_k4.h        # CutH+K4
│   ├── moe_finalize_routing_v2_fp_cuth_k2_optimized.h  # H=5120优化
│   ├── moe_finalize_routing_v2_fp_db_all_bias.h    # 全Bias预加载
│   ├── moe_finalize_routing_v2_bf16.h              # BF16 Cast模板
│   ├── moe_finalize_routing_v2_bf16_cutk.h / _cuth.h / _cuth_k2.h / _cuth_k4.h
│   ├── moe_finalize_routing_v2_bf16_all_bias.h
│   └── arch35/                                     # ascend950 regbase专用
│       ├── moe_finalize_routing_v2_base.h          # MicroAPI基类
│       ├── moe_finalize_routing_v2_h_full_load.h   # H全量加载
│       ├── moe_finalize_routing_v2_h_partial_load.h # H分段加载
│       ├── moe_finalize_routing_v2_row_k_h_full_load.h  # 行+K+H全量
│       └── moe_finalize_routing_v2_k_h_full_load.h # K+H全量
├── op_graph/ moe_finalize_routing_v2_proto.h
├── docs/ aclnnMoeFinalizeRoutingV2.md
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_finalize_routing_v2_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | expanded_x | REQUIRED | FP16/BF16/FP32 | Mode 0/2: [N*K, H]; Mode 1/3: [E, C, H] |
| 输入 | expanded_row_idx | REQUIRED | INT32 | 1D: [N*K] |
| 输入 | x1 | OPTIONAL | FP16/BF16/FP32 | 2D: [N, H] |
| 输入 | x2 | OPTIONAL | FP16/BF16/FP32 | 2D: [N, H] |
| 输入 | bias | OPTIONAL | FP16/BF16/FP32 | 2D: [E, H] |
| 输入 | scales | OPTIONAL | FP16/BF16/FP32 | 2D: [N, K] |
| 输入 | expert_idx | OPTIONAL | INT32 | 2D: [N, K] |
| 输出 | y | REQUIRED | 同 expanded_x | 2D: [N, H] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| drop_pad_mode | Int | 0 | 0=DroplessCol, 1=DropPadCol, 2=DroplessRow, 3=DropPadRow |

### 3.2 Tiling 策略

**Membase TilingData**（tiling.h）: 24 个 int64_t 字段，含核分配、H/K 切分、行分配、`skip1IsNull/skip2IsNull/scalesIsNull/dropPadMode` 标记。

**Regbase TilingData**（tiling_arch35.h）: 21 个字段，含 `row/e/c/h/hAligned/k`、核分配、H/K 分段参数、`activeNum`。

**Membase TilingKey**（范围 20000-20049）: 19 个与 v1 相同的基础 Key + 19 个 `_WITHOUT_BIAS` 变体（bias 为空时）+ BF16 混合精度变体（scale dtype=FP32）。

**Regbase TilingKey**: 4 种加载策略 × 4 种 DropPad 模式 × 3 种数据类型 = 48 个 Key:
```
Key = loadStrategy(10000/20000/30000/40000) + dropPadMode(0/10/20/30) + dtypeOffset(0/1/2)
```

**关键常量**: `BOUND_K_V2=256`, `NETWORKSIZE2_V2=5120`, `TRIPLE_BUFFER_V2=3`, `INVALID_ROW_INDEX=-1`。

**Membase Tiling 策略选择**: 优化CutH(H=5120) → 全Bias预加载(K<=256) → CutK(K>256) → 双缓冲FullH → CutH(K=2/K=4/通用)。

**Regbase Tiling 策略选择**: RowKHFullLoad → KHFullLoad → HFullLoad → HPatialLoad（按 UB 容量递减）。

**Tiling 注册优先级**: Regbase=30000（高），Membase=10000（低）。通过 `IsCapable()` 按平台分流。

### 3.3 Shape 推导 (moe_finalize_routing_v2_infershape.cpp)

```
y: [dim0, expanded_x.last_dim]
dim0 = scales.dim(0)（scales 非空时）或 expanded_row_idx.dim(0)
```

**DropPad 模式约束**: Mode 0/2 时 expanded_x 为 2D [N*K, H]；Mode 1/3 时为 3D [E, C, H]。

### 3.4 ACLNN API (op_api/aclnn_moe_finalize_routing_v2.cpp)

**两段式接口**:

```cpp
// 第一段
aclnnStatus aclnnMoeFinalizeRoutingV2GetWorkspaceSize(
    const aclTensor* expandedX, const aclTensor* expandedRowIdx,
    const aclTensor* x1, const aclTensor* x2,
    const aclTensor* bias, const aclTensor* scales,
    const aclTensor* expertIdx,
    int64_t dropPadMode,
    const aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

// 第二段
aclnnStatus aclnnMoeFinalizeRoutingV2(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**Membase 入口**（moe_finalize_routing_v2.cpp）: ~50 个 TilingKey 分支。所有 Kernel 类带 `<typename T, bool ISBIASEXIST>` 模板参数。

**Regbase 入口**（moe_finalize_routing_v2_apt.cpp）: 48 个 TilingKey 分支，使用 MicroAPI 寄存器张量。

### 4.2 核心计算流程

**数学公式**:
```
y[i, h] = x1[i, h] + x2[i, h] + Σ(k=0..K-1) scales[i, k] * (expanded_x[idx(i,k), h] + bias[expert(i,k), h])
```

**DropPad 模式语义**:

| Mode | 名称 | expanded_x 布局 | 索引顺序 |
|------|------|----------------|----------|
| 0 | DroplessCol | 2D [N*K, H] | 列主序: idx = k*N + row |
| 1 | DropPadCol | 3D [E, C, H] | 列主序, -1 标记 padding |
| 2 | DroplessRow | 2D [N*K, H] | 行主序: idx = row*K + k |
| 3 | DropPadRow | 3D [E, C, H] | 行主序, -1 标记 padding |

**BF16 处理**: Cast 到 FP32 累加后 Cast 回 BF16。BF16 模板额外带 `<typename TS>` scale 类型参数，支持混合精度（BF16 输入 + FP32 scale）。

### 4.3 数据结构与内存布局

**公共常量**（moe_finalize_routing_v2_common.h）: `ONE_BLK_SIZE=32`, `PARALLEL_NUM=2`, `INVALID_ROW_INDEX=-1`。

**Membase Kernel 类**: 13 个类（7 个 FP32/FP16 + 6 个 BF16），与 v1 相同但增加 `ISBIASEXIST` 模板参数。

**Regbase Kernel 类**: 4 个 MicroAPI 类:
- `HFullLoad`: H 全量加载，按 K 迭代
- `HPartialLoad`: H 分段加载
- `RowKHFullLoad`: 整行+K+H 全量加载
- `KHFullLoad`: K+H 全量加载，按行迭代

### 4.4 多核并行策略

**Membase**: 按行均匀分核，H/K 按 UB 容量切分。双缓冲乒乓加载。

**Regbase**: 按行分核（`rowPerCore = CeilDiv(row, coreNum)`），H/K 加载策略按 UB 容量递减选择。

### 4.5 通信模式

无 SyncAll。Membase 使用硬件事件标志实现细粒度流水；Regbase 使用 MicroAPI `LocalMemBar` 进行内存排序。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节。Membase 使用 24 字段 TilingData，Regbase 使用 21 字段 struct。CompileInfo 含 `totalCoreNum`, `ubSize`。

### 5.2 Workspace 布局

```
┌──────────────────────────────────────────────────────────────┐
│ System Workspace = 16 MB (固定大小)                            │
└──────────────────────────────────────────────────────────────┘
```

无额外算子级 workspace。

---

## 6. 完整数据流

```
  expanded_x [N*K, H]   expanded_row_idx [N*K]   scales [N, K]
  x1 [N, H](可选)       bias [E, H](可选)        expert_idx [N, K](可选)
       │                        │                         │
       └────────────────────────┼─────────────────────────┘
                                │
                  对每行 i (核间并行):
                    result = x1[i] + x2[i]   (残差, 可选)
                    对每个 k:
                      idx = expanded_row_idx[...]    (按模式寻址)
                      if idx == -1: continue         (DropPad padding)
                      expert = expert_idx[i, k]
                      weighted = scales[i,k] * (expanded_x[idx] + bias[expert])
                      result += weighted
                    y[i] = result
                                │
                        y [N, H]
```

---

## 7. 性能相关设计

1. **DropPad 四种模式**: 支持 Column/Row 主序 + Drop/Pad 组合，适配不同上游算子输出布局。

2. **混合精度 BF16**: BF16 输入 + FP32 scale，Cast 到 FP32 累加后 Cast 回 BF16 输出，平衡精度和性能。

3. **Optional 输入优化**: 所有非核心输入为 OPTIONAL，Kernel 中通过 `ISBIASEXIST` 等模板参数静态消除空输入的分支。

4. **Regbase MicroAPI**: 使用寄存器张量和 `UnalignReg` 进行非对齐数据搬运，`LoadInputDataWithBrc` 广播 scale 加载。

5. **K=2/K=4 展开**: 与 v1 相同，消除循环分支开销。

6. **H=5120 网络优化**: 双缓冲队列合并 skip/scales 为连续缓冲。

---

## 8. 依赖关系

```
op_host/moe_finalize_routing_v2_def.cpp
└── register/op_def_registry.h

op_host/moe_finalize_routing_v2_tiling_membase.cpp
├── moe_finalize_routing_v2_tiling.h
│   ├── register/op_impl_registry.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/moe_finalize_routing_v2_tiling_arch35.cpp
├── moe_finalize_routing_v2_tiling_arch35.h
└── tiling_base/tiling_base.h

op_host/op_api/aclnn_moe_finalize_routing_v2.cpp
├── aclnn_moe_finalize_routing_v2.h
└── aclnnInnerMoeFinalizeRoutingV2 (内部符号)

op_kernel/moe_finalize_routing_v2.cpp (membase)
├── moe_finalize_routing_v2_fp_db.h / _fp_cutk.h / _fp_cuth*.h
├── moe_finalize_routing_v2_bf16*.h
├── moe_finalize_routing_v2_common.h
└── kernel_operator.h

op_kernel/moe_finalize_routing_v2_apt.cpp (regbase)
├── arch35/moe_finalize_routing_v2_base.h
├── arch35/moe_finalize_routing_v2_h_full_load.h
├── arch35/moe_finalize_routing_v2_h_partial_load.h
├── arch35/moe_finalize_routing_v2_row_k_h_full_load.h
├── arch35/moe_finalize_routing_v2_k_h_full_load.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| drop_pad_mode | 0/1/2/3 |
| expanded_x 维度 | Mode 0/2: 2D; Mode 1/3: 3D |
| expanded_row_idx | 必须为 1D |
| dim-1 一致性 | x1, x2, bias, expanded_x 最后一维相同（=H） |
| Optional 输入 | x1/x2/bias/scales/expert_idx 均可为 null |
| DropPad padding | idx==-1 表示无效行，贡献为零 |
| STRIDE_BLOCK_LIMIT | ~1.07 billion rows |
| Regbase 310P | 仅 FP16/FP32 |

---

## 10. 与同族变体的核心差异

| 维度 | v1 | v2（本算子） |
|------|----|----|
| x1 (skip1) | REQUIRED | OPTIONAL |
| bias | REQUIRED | OPTIONAL |
| scales | REQUIRED | OPTIONAL |
| expert_idx | REQUIRED | OPTIONAL |
| Drop/Pad | 不支持 | 4种模式（Col/Row × Drop/Pad） |
| regbase | 不支持 | 支持（ascend950, MicroAPI, 4种加载策略） |
| 混合精度 | 不支持 | 支持（BF16输入+FP32 scale） |
| 310P | 不支持 | 支持 |
| TilingKey 范围 | 19个(20000-20018) | ~50个(20000-20049) + 48个regbase |
| ACLNN API | 有 | 有 |
