# 3rd_moe_inplace_index_add 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeInplaceIndexAdd`
**算子类**: 第三方算子（3rd），原地索引加法
**功能描述**: 在 MoE 场景下实现原地索引加法操作。对输入张量 var 的指定轴 (dim) 上，根据 indices 索引将 updates 中的值累加到 var 对应位置。等价于 PyTorch 的 `Tensor.index_add_(dim, index, source)` 操作，并额外支持 alpha 缩放因子。

**核心语义**: `var[index[i]][j][k] += alpha * updates[i][j][k]`

**五套 Kernel 策略**:
- **SimtNoSort (100000-101112)**: 多线程标量，不排序
- **SimdSort (200000)**: 向量指令 + 排序优化
- **Deterministic (300000-300002)**: 确定性计算（量化累加）
- **SimdNoSort (400000)**: 向量指令，不排序
- **SimtSort (500000-500011)**: 多线程标量 + 排序优化

**支持平台**: ascend950（主要）、ascend910_93
**支持数据类型**: BF16/FP16/FP32/UINT8/INT8/INT16/INT32/INT64/BOOL

---

## 2. 目录结构与文件职责

```
3rd/moe_inplace_index_add/
├── CMakeLists.txt
├── op_host/
│   ├── moe_inplace_index_add_def.cpp                # 算子定义
│   ├── moe_inplace_index_add_tiling.h               # 旧版 Tiling（非 arch35）
│   ├── moe_inplace_index_add_tiling.cpp             # 旧版 Tiling 入口
│   ├── arch35/                                      # arch35 专用 Tiling
│   │   ├── moe_inplace_index_add_tiling.h           # Tiling 数据结构
│   │   ├── moe_inplace_index_add_tiling.cpp         # Tiling 入口
│   │   ├── moe_inplace_index_add_tiling_arch35.h    # Tiling 基类
│   │   ├── moe_inplace_index_add_tiling_arch35.cpp  # Tiling 基类入口
│   │   ├── moe_inplace_index_add_base_tiling.cpp    # 策略选择 + 输入校验
│   │   ├── moe_inplace_index_add_simd_tiling.h/.cpp          # SIMD 无排序
│   │   ├── moe_inplace_index_add_simd_sort_tiling.h/.cpp     # SIMD 排序
│   │   ├── moe_inplace_index_add_simt_tiling.h/.cpp          # SIMT 无排序
│   │   ├── moe_inplace_index_add_simt_sort_tiling.h/.cpp     # SIMT 排序
│   │   └── moe_inplace_index_add_determinstic_tiling.h/.cpp  # 确定性
│   └── op_api/
│       ├── moe_inplace_index_add.h                  # L0 接口声明
│       └── moe_inplace_index_add.cpp               # AiCore 路径选择
└── op_kernel/
    ├── moe_inplace_index_add.cpp                    # Kernel 入口（613行，全分支）
    └── arch35/
        ├── moe_inplace_index_add_common.h           # 公共宏
        ├── moe_inplace_index_add_simt.h             # SIMT kernel
        ├── moe_inplace_index_add_simd.h             # SIMD kernel
        ├── moe_inplace_index_add_simd_sort.h        # SIMD 排序 kernel
        ├── moe_inplace_index_add_simt_sort.h        # SIMT 排序 kernel
        ├── moe_inplace_index_add_determinstic.h     # 确定性 kernel
        ├── moe_inplace_index_add_determinstic_notquant.h  # 确定性非量化
        └── indices_sort_utils.h                     # 索引排序工具
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_inplace_index_add_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 格式 |
|------|------|----------|----------|------|
| 输入 | var | REQUIRED（同时为输出） | BF16/FP16/FP32/UINT8/INT8/INT16/INT32/INT64/BOOL | ND |
| 输入 | indices | REQUIRED | INT32/INT64 | ND |
| 输入 | updates | REQUIRED | 与 var 相同 | ND |
| 输入 | alpha | OPTIONAL | 与 var 相同 | ND（标量） |
| 输出 | var_out | REQUIRED | 与 var 相同 | ND |

**属性**:

| 属性名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| axis | Int | REQUIRED | 操作的目标轴 |

**SoC 配置**: 仅配置 `ascend950`（第69行），支持动态编译静态、动态 Rank、动态 Shape。

### 3.2 Tiling 策略

采用**多策略注册、按优先级尝试**的 TilingBaseClass 体系。基础类 `MoeInplaceIndexAddTiling` 继承自 `TilingBaseClass`，由 5 个子策略类分别注册：

| 优先级 | 策略类 | TilingKey 范围 |
|--------|--------|---------------|
| 0 | MoeInplaceIndexAddSimdSortTiling | 200000 |
| 1 | MoeInplaceIndexAddSimtTiling | 100000-101112 |
| 2 | MoeInplaceIndexAddSimdTiling | 400000 |
| 3 | MoeInplaceIndexAddSimtSortTiling | 500000-500011 |
| 100 | MoeInplaceIndexAddDeterminsticTiling | 300000-300002 |

**策略选择逻辑**（base_tiling.cpp 第46-75行）:
```
GetDeterministic() == 1?
  YES → Deterministic 策略 (仅 FP32/FP16/BF16)
  NO  → updatesInAxis/varInAxis >= 5?
         YES → useSort=true
               afterAxis*typeSize >= 128? → SimdSort : (afterAxis<=1024? SimtSort : SimdSort)
         NO  → useSort=false
               afterAxis >= totalCores*128? → SimdNoSort : SimtNoSort
```

### 3.3 Shape 推导

本算子无独立的 infershape 文件。Shape 推导由 L2 层 API 完成:
- `var_out` shape 与 `var` 相同

### 3.4 ACLNN API (op_host/op_api/)

**L0 接口声明**（moe_inplace_index_add.h）:
```cpp
// AiCore 路径选择
// arch3510: 使用 arch35 的 5 策略 Tiling + 对应 Kernel
// 其他平台: 使用旧版 Tiling + Kernel
```

L2 层由构建系统自动生成 ACLNN 包装，无独立 `aclnn_*.cpp`。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口 (op_kernel/moe_inplace_index_add.cpp)

**入口函数**（613行）:
```cpp
extern "C" __global__ __aicore__ void moe_inplace_index_add(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha,
    GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)
```

**TilingKey 分发**:

| Key 范围 | 模板类 | 说明 |
|----------|--------|------|
| 100000-100012 | `MoeInplaceIndexAddSimt` | UINT32 索引 |
| 100100-100112 | `MoeInplaceIndexAddSimt` | UINT64 索引 |
| 101000-101012 | `MoeInplaceIndexAddSimt` | UINT32 + Alpha |
| 101100-101112 | `MoeInplaceIndexAddSimt` | UINT64 + Alpha |
| 200000 | `MoeInplaceIndexAddSimdSort` | 向量排序 |
| 300000-300002 | `MoeInplaceIndexAddDeterminstic` | 确定性 (FP32/FP16/BF16) |
| 400000 | `MoeInplaceIndexAddSimd` | 向量无排序 |
| 500000-500011 | `MoeInplaceIndexAddSimtSort` | 标量排序 |

### 4.2 核心计算流程

**通用模式**: `var[index[i]][j][k] += alpha * updates[i][j][k]`

各策略的差异:
- **SIMT**: 按标量逐元素处理，适合数据量小的场景
- **SIMD**: 使用向量指令批量处理，适合 afterAxis 较大的场景
- **Sort 变体**: 先对 indices 排序，使访问 var 的地址连续化，减少 cache miss
- **Deterministic**: 使用量化累加（FP32→INT32 量化→累加→反量化），保证浮点结果确定性

### 4.3 数据结构与内存布局

**SimtNoSort TilingData**（simt_tiling.h 第23-32行）:
```
├── preAxis           : axis 之前的元素总数
├── varInAxis         : var 在 axis 维的大小
├── updatesInAxis     : updates 在 axis 维的大小
├── afterAxis         : axis 之后的元素总数
├── ubFactor          : UB 每次处理的元素数
├── colUbFactor       : 列方向 UB 因子
├── indicesStride     : indices 步长（支持 view）
└── indicesUbFactor   : indices 的 UB 因子
```

**SimdNoSort TilingData**: 在 SimtNoSort 基础上增加多核分割参数（usedCoreNum、per-core indices/preAxis/afterAxis 分配、split 标志）。

**Deterministic TilingData**: 在 SimdSort 基础上增加量化参数（ubQuantaIndxFactor、ubVarFactor、eachCoreVarCount、isDeterminstic、isOpti）。

**Kernel 模板参数**（以 SIMT 为例）:
```
MoeInplaceIndexAddSimt<T, DTYPE_INDICES, IndexType, CastType, HasAlpha, ContiguousIndices>
  T: 数据类型
  IndexType: uint32_t/uint64_t
  CastType: 中间类型（INT16/INT8/UINT8→INT32, BOOL→FP16）
  HasAlpha: 是否有 alpha
  ContiguousIndices: indices 是否连续 (stride==1)
```

### 4.4 多核并行策略

- **SIMT 策略**: 单核处理，按 indices 逐个遍历
- **SIMD 策略**: 多核并行，按 indices/preAxis/afterAxis 三维分配核
  - 优先切 afterAxis（afterAxis*typeSize >= 128*coreNum）
  - 其次切 indices（indicesAxis > preAxis）
  - 最后切 preAxis
- **SIMT Sort 策略**: 按 indices 均匀分配核，每核独立排序后处理
- **Deterministic 策略**: 按 indices 和 var 双维度分配核

### 4.5 通信模式

- **Sort 变体**: 使用核内排序（`indices_sort_utils.h`），无需核间通信
- **Deterministic**: 使用 workspace 做核间部分和归约，需要核间同步
- 其他策略: 每核独立，无核间通信

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

**MoeInplaceIndexAddCompileInfo**（tiling.h 第37-49行）:
```
MoeInplaceIndexAddCompileInfo
├── core_num                : int64_t   # 可用核数
├── ub_size                 : int64_t   # UB 大小
├── var_size                : int64_t   # var 数据类型大小
├── var_data_each_block     : int64_t   # 每 block 的 var 数据数
├── indices_size            : int64_t   # indices 数据类型大小
├── indices_data_each_block : int64_t   # 每 block 的 indices 数据数
├── soc_version             : int64_t   # SoC 版本
├── atomic_add              : int64_t   # 是否支持原子加
├── isAscendc               : bool     # AscendC 模式
├── coreNum                 : int64_t   # AscendC 核数
└── ubSize                  : int64_t   # AscendC UB 大小
```

各策略有独立的 TilingData 结构（见 4.3 节），均注册为 TilingData 子类。

### 5.2 Workspace 布局

| 策略 | Workspace |
|------|-----------|
| SimtNoSort | 无额外 workspace |
| SimdNoSort | 无额外 workspace |
| SimdSort | 排序临时空间 |
| SimtSort | 排序临时空间 + 共享缓冲 |
| Deterministic | 多项 workspace（determinstic_tiling.cpp 第303-318行） |

Deterministic 模式 workspace 组成:
- `rCoutWsSize = preAxis * varInAxis * 4`（int32 计数）
- `rValueWsSize = varAxis * 4`（FP32 残差值）
- `sumQuantaIntWsSize = varAxis * 4`（int32 量化累加和）
- `sumWsSize = totalCoreNum * eachCoreIndexCount * afterAxis * 4`（FP32 部分和）
- `sumIdxWsSize = totalCoreNum * eachCoreIndexCount * indicesTypeSize`（部分和索引）

---

## 6. 完整数据流

### 6.1 通用数据流

```
  var [N0, N1, ..., Nk, ..., Nn]       indices [M]        updates [N0, M, ..., Nn]
       |                                    |                       |
       |            axis = k                |                       |
       +--- var[index[i]] <--- index[i] ----+                       |
       |          ^                                                 |
       |          |              alpha (可选)                        |
       |          |              [1]                                 |
       |          |                |                                 |
       |    += alpha * updates[i][...]                              |
       |                         |                                  |
       v                         v                                  |
  var_out [N0, N1, ..., Nk, ..., Nn] (原地更新)                     |
```

### 6.2 Sort 变体数据流

```
  indices [M] → Sort → sorted_indices [M]
                        |
         +--------------+---------------+
         |                              |
   按 sorted_indices 遍历         读取 updates[i]
   连续访问 var[sorted_idx]        |
   减少 cache miss          += alpha * updates[i]
         |                              |
         v                              v
    var_out (原地更新)
```

### 6.3 Deterministic 数据流

```
  per core:
    读取 updates[i] → FP32
    量化到 INT32 → 累加到 workspace 部分和
    |
  核间同步
    |
  汇总 workspace 部分和 → 反量化到 FP32 → Cast 回原类型
    |
  写回 var_out
```

---

## 7. 性能相关设计

1. **五策略自动选择**: 根据数据特征（数据量、数据类型、是否需要排序、确定性要求）自动选择最优策略。SIMD 适合大数据量，SIMT 适合小数据量或低精度类型。

2. **排序优化**: 当 `updatesInAxis/varInAxis >= 5` 时启用排序策略。先对 indices 排序使 var 访问连续化，显著减少 cache miss。

3. **SIMD 多核切分**: 按 afterAxis/indices/preAxis 三维分配核，优先切 afterAxis 以利用连续内存访问的带宽优势。

4. **确定性量化累加**: Deterministic 模式使用 FP32→INT32 量化→累加→反量化流程，保证浮点结果的确定性和可复现性。

5. **类型转换优化**: INT16/INT8/UINT8 先 Cast 到 INT32，BOOL Cast 到 FP16，避免低精度直接计算的开销。

6. **View 支持**: 通过 `InputIsView()` 和 `GetInputStride()` 检测 indices 是否为 view，支持非连续 indices 张量，减少不必要的 Contiguous 操作。

---

## 8. 依赖关系

```
op_host/moe_inplace_index_add_tiling.cpp
└── moe_inplace_index_add_tiling.h
    ├── register/op_impl_registry.h
    └── tiling_base/tiling_base.h

op_host/arch35/moe_inplace_index_add_tiling_arch35.cpp
├── moe_inplace_index_add_tiling_arch35.h
│   └── tiling_base/tiling_base.h
├── moe_inplace_index_add_base_tiling.cpp (策略选择)
├── moe_inplace_index_add_simt_tiling.h/.cpp
├── moe_inplace_index_add_simd_tiling.h/.cpp
├── moe_inplace_index_add_simd_sort_tiling.h/.cpp
├── moe_inplace_index_add_simt_sort_tiling.h/.cpp
└── moe_inplace_index_add_determinstic_tiling.h/.cpp

op_host/op_api/moe_inplace_index_add.cpp
├── moe_inplace_index_add.h
└── external/aclnn_kernels/aclnn_platform.h

op_kernel/moe_inplace_index_add.cpp
├── arch35/moe_inplace_index_add_simt.h
│   └── moe_inplace_index_add_common.h
├── arch35/moe_inplace_index_add_simd.h
├── arch35/moe_inplace_index_add_simd_sort.h
│   └── indices_sort_utils.h
├── arch35/moe_inplace_index_add_simt_sort.h
│   └── indices_sort_utils.h
├── arch35/moe_inplace_index_add_determinstic.h
│   └── moe_inplace_index_add_determinstic_notquant.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| 平台限制 | 仅配置 ascend950，arch35 专用 |
| axis 必填 | 属性 REQUIRED |
| 确定性模式 | 仅支持 FP32/FP16/BF16 |
| indices 范围 | indices[i] 必须在 [0, varInAxis) 范围内 |
| BF16 特殊处理 | 非 arch35 平台需要额外 workspace 做 FP32 中转 |
| workspace | Deterministic 模式需要较大 workspace |
| 多策略选择 | 5 策略按优先级尝试，第一个 IsCapable 返回 true 的被选中 |
