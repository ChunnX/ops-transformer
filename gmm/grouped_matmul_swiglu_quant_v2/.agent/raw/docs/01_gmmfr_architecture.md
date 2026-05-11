# GMMFR MxA8W4 架构总览

> 本文档分析 GroupedMatmulFinalizeRouting (GMMFR) 算子在 MxA8W4（FP8 输入 + FP4 权重 + MX 微缩放）场景下的整体架构，作为 GMMSQ 算子开发的前置参考。

## 目录

- [1. 概述](#1-概述)
- [2. 文件结构](#2-文件结构)
- [3. 数据流总览](#3-数据流总览)
- [4. Kernel 入口分析](#4-kernel-入口分析)
- [5. Tiling Data 分析](#5-tiling-data-分析)
- [6. Tiling Key 分析](#6-tiling-key-分析)
- [7. Common Config 分析](#7-common-config-分析)
- [8. 四层架构详解](#8-四层架构详解)
  - [8.1 Controller 层](#81-controller-层)
  - [8.2 BasicBlock 层](#82-basicblock-层)
  - [8.3 CubeCompute 层（AIC 侧）](#83-cubecompute-层aic-侧)
  - [8.4 VecCompute 层（AIV 侧）](#84-veccompute-层aiv-侧)
- [9. 跨核同步机制](#9-跨核同步机制)
- [10. 关键设计要点](#10-关键设计要点)

---

## 1. 概述

GMMFR 是 GroupedMatmul 和 MoeFinalizeRouting 的融合算子：先对多个 group 分别执行矩阵乘法，再将结果按 rowIndex 做 scatter-add（Routing）动作。

MxA8W4 场景特指：
- **输入 X（左矩阵）**: FP8 (E4M3FN) 格式
- **权重 W（右矩阵）**: FP4 (E2M1) 格式，NZ 排布
- **反量化方式**: MX (Microscaling) —— 每 32 个元素共享一个 E8M0 scale 因子
- **伪量化路径**: 权重在 host 侧已完成伪量化（weight quantization），kernel 内只需做反量化 + Matmul + Routing

该场景通过宏 `V310_GMM_FR_ANTI_QUANT` 激活，与全量化（Full Quantization）路径互斥。

---

## 2. 文件结构

```
op_kernel/arch35/
├── common/
│   ├── basic_block_config.h      # 核心配置结构体（WqmmConfig, VecAntiQuantConfig, BasicBlock*Param）
│   ├── basic_block_vf_mx.h       # MxA8W4 VF 参数结构体（MxA8W4NzParams）和常量
│   └── tool.h                    # 工具函数（CeilAlign, CeilDivide, GetKBUnit, IsMxA8W4）和全局常量
│
└── weight_quant_basic_block/
    ├── gmm_fr_weight_quant_resplit_controller.h  # [L1] Controller: 分组遍历 + 多核 N 轴切分
    ├── gmm_fr_weight_quant_vcv_basic_block.h     # [L2] BasicBlock: AIC/AIV 双核调度 + 跨核同步
    ├── gmm_fr_weight_quant_cube_compute.h        # [L3a] CubeCompute: AIC 侧 Matmul（L1→L0→MMAD→Fixpipe）
    ├── gmm_fr_weight_quant_cube_compute_tools.h  # CubeCompute 工具（LoadA/B L1→L0, MmadCompute, FixL0CToDst）
    ├── gmm_fr_weight_quant_vec_compute.h         # [L3b] VecCompute: AIV 侧反量化 + Routing
    ├── gmm_fr_mx_a8w4_vf.h                       # VecCompute VF 实现（4bit→8bit 解包, Bias 缩放, MulLogits）
    ├── gmm_fr_weight_quant_tiling_data.h         # Tiling 数据结构体
    └── gmm_fr_weight_quant_tiling_key.h          # Tiling Key 模板参数声明
```

---

## 3. 数据流总览

```
                          ┌──────────────────────────────────────────────────────┐
                          │                  Controller (L1)                      │
                          │  · 按 group 遍历，从 groupList 获取 M 轴切分值          │
                          │  · 对每个 group 的 M 轴做 L1 级分块 (mL1Size)           │
                          │  · 对 N 轴做多核轮询切分 (SplitNByMultiCore)            │
                          │  · 跨 group 更新各 GM 地址指针                          │
                          └───────────┬──────────────────────────────┬───────────┘
                                      │                              │
                          ┌───────────▼──────────┐    ┌──────────────▼───────────┐
                          │   BasicBlock (L2)    │    │   BasicBlock (L2)        │
                          │   AIC Core (Cube)    │    │   AIV Core (Vec)         │
                          │                      │    │                          │
                          │  ┌────────────────┐  │    │  ┌────────────────────┐  │
                          │  │ CubeCompute    │  │    │  │ VecCompute (L3b)   │  │
                          │  │ (L3a)          │  │    │  │                    │  │
                          │  │                │  │    │  │ 1. MTE2: GM→UB     │  │
                          │  │ 1. MTE2: GM→L1 │  │    │  │   (weight+scale+   │  │
                          │  │   (A矩阵 ND2NZ │  │    │  │    bias)           │  │
                          │  │    + MX Scale) │  │    │  │                    │  │
                          │  │                │  │    │  │ 2. VF: 4bit→8bit   │  │
                          │  │ 2. L1→L0 + MX │  │◄───┼────│   反量化解包        │  │
                          │  │   Scale 融合   │  │sync│  │   + Bias 缩放      │  │
                          │  │   搬运          │  │    │  │                    │  │
                          │  │                │  │    │  │ 3. MTE3: UB→L1     │  │
                          │  │ 3. MMAD 矩阵乘 │  │    │  │   (高精度结果)       │  │
                          │  │                │  │    │  │                    │  │
                          │  │ 4. Fixpipe:    │──┼───►│  │ 4. VF: MulLogits   │  │
                          │  │   L0C→UB       │  │    │  │   (乘 logits)       │  │
                          │  │                │  │    │  │                    │  │
                          │  └────────────────┘  │    │  │ 5. MTE3: UB→GM     │  │
                          └──────────────────────┘    │  │   (AtomicAdd       │  │
                                                       │  │    Scatter)        │  │
                                                       │  └────────────────────┘  │
                                                       └─────────────────────────┘
```

**核心流水线**:
1. AIC 侧负责 Cube 矩阵乘法（L1→L0→MMAD→Fixpipe→UB）
2. AIV 侧负责 Weight 反量化（4bit→8bit 解包 + Bias 缩放）+ MTE3 写回 L1
3. AIC 和 AIV 通过 CrossCore Sync 实现乒乓流水：AIC 算完一块 → AIV 取结果做 MulLogits + Routing → 写回 GM

---

## 4. Kernel 入口分析

### 4.1 文件: `grouped_matmul_finalize_routing_apt.cpp`

### 4.2 `V310_GMM_FR_ANTI_QUANT` 宏

```cpp
#if (ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && (ORIG_DTYPE_W == DT_FLOAT4_E2M1))
#define V310_GMM_FR_ANTI_QUANT
#endif
```

**含义**: 当输入 X 为 FP8 E4M3FN 且权重 W 为 FP4 E2M1 时，激活伪量化（Weight Quantization）路径。该宏控制两套完全不同的代码路径：
- **定义时**: 走 Weight Quantization 路径（本文分析对象），使用 `GMMFRWeightQuantResplitController`
- **未定义时**: 走 Full Quantization 路径，使用 `grouped_matmul_finalize_routing_pertoken_dequant` 等函数

### 4.3 模板参数

Weight Quantization 路径的模板参数为：

```cpp
template <int ATRANS, int BTRANS, int HASBIAS>
```

| 参数 | 含义 | 可选值 |
|------|------|--------|
| `ATRANS` | 左矩阵 A 是否转置 | 固定为 0（不转置） |
| `BTRANS` | 右矩阵 B 是否转置 | 0（不转置）或 1（转置） |
| `HASBIAS` | 是否有 Bias | 0（无）或 1（有） |

对比 Full Quantization 路径的模板参数 `SCALETYPE, ROWINDEXTYPE`，Weight Quantization 路径更简洁，因为 scale 类型和 rowIndex 类型在 MxA8W4 场景下是固定的。

### 4.4 Controller 实例化

```cpp
GROUPED_MATMUL_FINALIZE_ROUTING::GMMFRWeightQuantResplitController<
    DTYPE_X, DTYPE_W, DTYPE_SCALE, DTYPE_SCALE, DTYPE_PERTOKEN_SCALE, DTYPE_BIAS, DTYPE_Y,
    bfloat16_t, float, int64_t, HASBIAS,
    WeightQuantBatchMatmulV2::Arch35::MXA8W4_NZNK,
    WeightQuantBatchMatmulV2::Arch35::VEC_ANTIQUANT_CONFIG_DYNAMIC>
    controller;
```

模板参数展开：

| 模板参数 | 实际值 | 含义 |
|----------|--------|------|
| `xType` | `DTYPE_X` (fp8_e4m3fn_t) | 输入激活值类型 |
| `wType` | `DTYPE_W` (fp4x2_e2m1_t) | 权重类型 |
| `antiQuantScaleType` | `DTYPE_SCALE` | 反量化 scale 类型 |
| `scaleType` | `DTYPE_SCALE` | 量化 scale 类型 |
| `perTokenScaleType` | `DTYPE_PERTOKEN_SCALE` | per-token scale 类型 |
| `biasType` | `DTYPE_BIAS` (bfloat16_t) | Bias 类型 |
| `yType` | `DTYPE_Y` | 输出类型 |
| `sharedInputDType` | `bfloat16_t` | 共享输入类型 |
| `logitsType` | `float` | Logits 类型 |
| `rowIndexType` | `int64_t` | Row Index 类型 |
| `hasBias` | `HASBIAS` | 是否有 Bias |
| `wqmmConfig` | `MXA8W4_NZNK` | 矩阵乘配置 |
| `vecConfig` | `VEC_ANTIQUANT_CONFIG_DYNAMIC` | 向量反量化配置 |

### 4.5 `MXA8W4_NZNK` 配置

```cpp
static constexpr WqmmConfig MXA8W4_NZNK =
    {false, true, QuantType::MX, false, QuantType::NONE, CubeFormat::NZ};
```

| 字段 | 值 | 含义 |
|------|-----|------|
| `aTrans` | `false` | A 矩阵不转置 |
| `bTrans` | `true` | B 矩阵转置（NZ 格式天然转置） |
| `antiQuantType` | `QuantType::MX` | 使用 MX (Microscaling) 反量化 |
| `hasAntiQuantOffset` | `false` | 无反量化偏移 |
| `quantType` | `QuantType::NONE` | 不做量化（输出直接是反量化后的高精度） |
| `weightFormat` | `CubeFormat::NZ` | 权重排布为 NZ（为 Cube 单元优化） |

### 4.6 `VEC_ANTIQUANT_CONFIG_DYNAMIC` 配置

```cpp
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_DYNAMIC = {4, 0};
```

| 字段 | 值 | 含义 |
|------|-----|------|
| `ubMte2BufferNum` | `4` | MTE2 搬运使用 4 缓冲（四缓冲），提升搬运与计算的重叠度 |
| `ubMte2InnerSize` | `0` | 动态内尺寸（0 表示由 kernel 内部根据实际 shape 动态决定） |

### 4.7 调用流程

```cpp
controller.Init(x, w, scale, bias, group_list, pertoken_scale, logit, row_index, y, share_input, tiling);
controller.Process();
```

`Init` → 保存 tiling 数据、转换 GM 指针、初始化 BasicBlock、根据是否有 Bias 设置动态 K 轴阈值
`Process` → 完整的推理流程（详见 [8.1 Controller 层](#81-controller-层)）

---

## 5. Tiling Data 分析

### 5.1 结构体: `GMMFinalizeRoutingWeightQuantTilingData`

```cpp
#pragma pack(push, 8)
struct GMMFinalizeRoutingWeightQuantTilingData {
    uint8_t  groupListType;       // 分组列表类型
    uint8_t  hasBias;             // 是否有 Bias
    uint16_t reserve1;            // 预留对齐

    uint32_t groupNum;            // Group 数量
    uint32_t coreNum;             // 使用的核数
    uint32_t outputBs;            // 输出 Batch Size
    uint32_t sharedInputOffset;   // 共享输入在输出中的偏移
    uint32_t sharedInputLen;      // 共享输入长度
    float    sharedInputWeight;   // 共享输入权重系数
    uint32_t groupSize;           // 反量化 Group Size

    uint64_t kSize;               // K 轴大小（矩阵乘的 inner dimension）
    uint64_t nSize;               // N 轴大小（权重矩阵的列数）
    uint64_t initSize;            // GM 初始化（清零）大小
};
#pragma pack(pop)
```

### 5.2 字段详解

| 字段 | 类型 | 含义 | 用途 |
|------|------|------|------|
| `groupListType` | `uint8_t` | 分组模式：0 = cumsum（前缀和），1 = count（计数） | Controller 解析 groupList 时使用 |
| `hasBias` | `uint8_t` | 是否有 Bias 项 | 控制 Bias 搬运和计算的开关 |
| `reserve1` | `uint16_t` | 预留对齐字段 | 保证 8 字节对齐 |
| `groupNum` | `uint32_t` | Expert 数量（group 数） | Controller 遍历 group 的循环上限 |
| `coreNum` | `uint32_t` | 参与计算的 AI Core 总数 | 多核 N 轴轮询切分的模数 |
| `outputBs` | `uint32_t` | 输出 Batch Size | 输出矩阵的总行数 |
| `sharedInputOffset` | `uint32_t` | Shared Expert 输出在总输出中的起始位置 | GM 清零时跳过 shared input 区域 |
| `sharedInputLen` | `uint32_t` | Shared Expert 输出的长度 | GM 清零时计算 shared input 区域大小 |
| `sharedInputWeight` | `float` | Shared Expert 与 MoE Expert combine 的权重系数 | VecCompute 做 CastAndMul 时使用 |
| `groupSize` | `uint32_t` | 反量化 group 大小（MX 场景固定为 32） | 计算 scale 数据的地址偏移 |
| `kSize` | `uint64_t` | K 轴维度大小 | 矩阵乘 inner dim，控制 K 轴循环 |
| `nSize` | `uint64_t` | N 轴维度大小 | 权重矩阵列数，控制 N 轴切分 |
| `initSize` | `uint64_t` | GM 输出初始化（清零）的总元素数 | `InitAtomicGm` 中控制清零范围 |

---

## 6. Tiling Key 分析

### 6.1 文件: `gmm_fr_weight_quant_tiling_key.h`

### 6.2 Tiling Key 定义

```cpp
ASCENDC_TPL_ARGS_DECL(GroupedMatmulFinalizeRouting,
    ASCENDC_TPL_UINT_DECL(ATRANS,  ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_DECL(BTRANS,  ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(HASBIAS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1));
```

### 6.3 支持的场景

| 参数 | 位宽 | 可选值 | 说明 |
|------|------|--------|------|
| `ATRANS` | 2 bit | `0` | A 矩阵转置标志，MxA8W4 场景固定为 0 |
| `BTRANS` | 2 bit | `0, 1` | B 矩阵转置标志，0 = RowMajor，1 = ColumnMajor (NZ) |
| `HASBIAS` | 2 bit | `0, 1` | 是否有 Bias |

**组合场景**: `ATRANS x BTRANS x HASBIAS = 1 x 2 x 2 = 4` 种场景

| ATRANS | BTRANS | HASBIAS | 场景描述 |
|--------|--------|---------|----------|
| 0 | 0 | 0 | A 不转置，B 不转置，无 Bias |
| 0 | 0 | 1 | A 不转置，B 不转置，有 Bias |
| 0 | 1 | 0 | A 不转置，B 转置（NZ），无 Bias |
| 0 | 1 | 1 | A 不转置，B 转置（NZ），有 Bias |

Tiling Key 的选择逻辑：
```cpp
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(ATRANS,  ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_SEL(BTRANS,  ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_SEL(HASBIAS, ASCENDC_TPL_UI_LIST, 0, 1)));
```

---

## 7. Common Config 分析

### 7.1 文件: `basic_block_config.h` + `tool.h`

### 7.2 核心结构体

#### `WqmmConfig` — 矩阵乘量化配置

```cpp
struct WqmmConfig {
    bool aTrans;              // A 矩阵是否转置
    bool bTrans;              // B 矩阵是否转置
    QuantType antiQuantType;  // 反量化类型（NONE/PER_TENSOR/PER_CHANNEL/PER_GROUP/MX）
    bool hasAntiQuantOffset;  // 是否有反量化偏移
    QuantType quantType;      // 量化类型
    CubeFormat weightFormat;  // 权重排布格式（NZ/ND）
};
```

#### `VecAntiQuantConfig` — 向量反量化配置

```cpp
struct VecAntiQuantConfig {
    uint64_t ubMte2BufferNum = 2;   // MTE2 搬运 Buffer 数量（默认双缓冲）
    uint64_t ubMte2InnerSize = 512; // MTE2 搬运内尺寸
};
```

#### `BasicBlockControlParam` — BasicBlock 控制参数

```cpp
struct BasicBlockControlParam {
    uint64_t processId;        // 当前处理的 Buffer ID（0 或 1，乒乓切换）
    uint64_t mSize;            // 当前 group 的 M 轴总大小
    uint64_t mL1Size;          // M 轴在 L1 上的分块大小
    uint64_t curBasicBlockId;  // 当前 BasicBlock 的全局 ID
    uint64_t basicBlockLimit;  // BasicBlock ID 的起始偏移
    uint64_t mOffset;          // M 轴当前偏移
    uint64_t nOffset;          // N 轴当前偏移
};
```

#### `BasicBlockOffsetParam` — BasicBlock 偏移参数

```cpp
struct BasicBlockOffsetParam {
    uint64_t mL1Size;   // M 轴 L1 分块实际大小
    uint64_t kaL1Size;  // A 矩阵 K 轴 L1 分块大小
    uint64_t kbL1Size;  // B 矩阵 K 轴 L1 分块大小
    uint64_t nL1Size;   // N 轴 L1 分块实际大小

    uint64_t mOffset;   // M 轴全局偏移
    uint64_t nOffset;   // N 轴全局偏移

    uint64_t mSize;     // M 轴总大小
    uint64_t kSize;     // K 轴总大小
    uint64_t nSize;     // N 轴总大小
    uint64_t kAlign;    // K 轴对齐后大小
    uint64_t nAlign;    // N 轴对齐后大小

    GM_ADDR yGmAddr;    // 输出 GM 地址
};
```

### 7.3 关键常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `QUADRUPLE_BUFFER_NUM` | `4` | 四缓冲数量（用于 weightHighBit 输出） |
| `DOUBLE_BUFFER_NUM` | `2` | 双缓冲数量（用于 weight L1、scale L1、A L1 等） |
| `SINGLE_BUFFER_NUM` | `1` | 单缓冲 |
| `L1_SIZE` | `512` (KB) | L1 Buffer 总大小 |
| `L1_SIZE_BYTE` | `512 x 1024` | L1 Buffer 总字节数 |
| `BIAS_L1_SIZE` | `4` (KB) | Bias L1 Buffer 大小 |
| `A_L1_MAX_SIZE_WITH_BIAS_QUANT` | `240 x 1024` (B) | 有 Bias 时 A 矩阵 L1 最大空间 |
| `MX_BIAS_SINGLE_VECTOR_SIZE` | `128` | MX Bias 单 Vector 大小 |
| `MX_SCALE_K_L1_SIZE` | `4096` | MX Scale K 轴 L1 搬运粒度 |
| `MX_GROUPSIZE` | `32` | MX 量化 Group Size（每 32 元素共享 scale） |
| `BASIC_BLOCK_PROCESS_NUM` | `2` | BasicBlock 乒乓 Buffer 数量 |
| `K_ALIGNMENT64` | `64` | K 轴对齐粒度 |
| `WEIGHT_F16_UB_NZ_STRIDE` | `65` | FP16 权重 UB NZ 排布 stride |

### 7.4 `QuantType` 枚举

```cpp
enum class QuantType {
    NONE = 0,        // 无量化
    PER_TENSOR = 1,  // Per-Tensor 量化
    PER_CHANNEL = 2, // Per-Channel 量化
    PER_GROUP = 3,   // Per-Group 量化
    MX = 4,          // MX (Microscaling) 量化
};
```

### 7.5 同步标志常量

```cpp
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;          // AIV -> AIC 同步标志
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;          // AIC -> AIV 同步标志
static constexpr uint64_t SYNC_AIC_FIX_AIV_VF_FLAG = 3;   // AIC Fixpipe -> AIV VF 同步
static constexpr uint64_t SYNC_AIV_MTE3_AIC_FIX_FLAG = 4; // AIV MTE3 -> AIC Fixpipe 同步
```

---

## 8. 四层架构详解

### 8.1 Controller 层

**文件**: `gmm_fr_weight_quant_resplit_controller.h`
**类**: `GMMFRWeightQuantResplitController`

#### 职责

1. **Group 遍历**: 按 `groupNum` 循环，从 `groupList` 获取每个 group 的 M 轴大小
2. **M 轴 L1 分块**: 将每个 group 的 M 轴按 `M_L1=256` 切分为多个 BasicBlock
3. **N 轴多核切分**: 将 N 轴按 `N_L1=256` 切分，通过 `coreNum` 轮询分配给各核
4. **GM 地址更新**: 每个 group 处理完后，更新 X/W/Scale/Bias/Y/Logits/RowIndex 的 GM 指针
5. **动态 K 轴配置**: MxA8W4 场景根据 M/N 分块大小动态选择 K 轴 L1 大小（256 或 512）

#### 关键流程

```
Process()
├── GetBlockIdx() -> 获取当前 Cube Block ID
├── InitOffsetParam() -> 初始化 offsetParam[0] 和 [1]
│   ├── kbL1Size = 256
│   ├── kaL1Size = kbL1Size
│   ├── kSize / nSize 从 tiling 获取
│   └── kAlign / nAlign = CeilAlign(kSize/nSize, BLOCK_CUBE)
│
├── InitAtomicGm() -> GM 输出清零 + Shared Input MTE3 写出
│
├── for each group:
│   ├── GetSplitValueFromGroupList() -> 获取 M 轴大小
│   ├── mBlkNum = CeilDivide(mSize, M_L1)
│   ├── mL1Size = CeilDivide(mSize, mBlkNum)
│   ├── UpdateGlobalAddr() -> 更新 BasicBlock 的 GM 地址
│   │
│   └── for each M L1 block:
│       └── SplitNByMultiCore()
│           ├── 计算 nOffset, nL1Size
│           ├── 动态选择 kbL1Size (256 or 512)
│           ├── 计算 kaL1Size (A 矩阵 K 轴 L1 分块)
│           └── basicBlock_.ComputeBasicBlock(curParam, lastParam)
│
└── basicBlock_.End() -> 收尾处理
```

#### 动态 K 轴配置逻辑

```cpp
// MxA8W4 场景：根据 M/N 分块大小动态选择 K 轴搬运粒度
offsetParam.kbL1Size =
    (mL1Size <= mxA8W4L1KDynamicConfigMThreshold_ &&
     nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
        MX_A8W4_L1_K_CONFIG_512 :   // 小块 -> 512（减少循环次数）
        MX_A8W4_L1_K_CONFIG_256;    // 大块 -> 256（节省 L1 空间）
```

其中 `mxA8W4L1KDynamicConfigMThreshold_` 在有 Bias 时为 240，无 Bias 时为 256。

#### 乒乓 Buffer 切换

Controller 使用 `processId`（0 或 1）在两个 `offsetParam` 之间乒乓切换：
```cpp
ctrlParam.processId = GetSwitchedProcessId(ctrlParam); // 1 - processId
```

---

### 8.2 BasicBlock 层

**文件**: `gmm_fr_weight_quant_vcv_basic_block.h`
**类**: `GMMFRWeightQuantVcvBasicBlock`

#### 职责

1. **AIC/AIV 双核调度**: 根据 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 分发到不同执行路径
2. **跨核同步**: 通过 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 实现 AIC <-> AIV 流水同步
3. **L1 空间管理**: 分配 Weight L1、Bias L1、Scale L1、A L1 的空间
4. **UB 空间管理**: 分配 weightLowBit、weightHighBit、bias、logits、rowIndex 的 UB Buffer

#### 成员变量

```cpp
GMMFRWeightQuantVecCompute<...> vecCompute_;   // AIV 侧 VecCompute 实例
GMMFRWeightQuantCubeCompute<...> cubeCompute_; // AIC 侧 CubeCompute 实例

LocalTensor<xType> weightL1_;       // Weight L1 Buffer（双缓冲）
LocalTensor<biasType> biasL1_;      // Bias L1 Buffer（双缓冲）
LocalTensor<float> ubOutputF32Buffer_; // UB 输出 Buffer（AIC Fixpipe -> AIV VF 的中间结果）
```

#### AIC 侧执行流程 (`IterateNzNkWithKAic`)

```
for kbL1Offset in [0, kSize) step kbL1Size:
    ├── WaitScaleMTE1ToMTE2() -> 等待 Scale MTE1->MTE2
    ├── CopyMxScaleGmToL1() -> MX Scale A/B GM->L1 (ND2NZ)
    ├── WaitMTE1ToMTE2() -> 等待 A 矩阵 MTE1->MTE2
    ├── CopyAAndBiasGmToL1() -> A 矩阵 GM->L1 (ND2NZ)
    ├── PadL1IfNotAlign() -> K 轴不对齐时补零
    ├── WaitAivToAic() -> 等待 AIV 完成上一轮 Weight 反量化
    ├── LaunchMatmul() -> L1->L0 + MMAD 计算
    ├── SetMTE1ToMTE2() -> 通知 MTE1->MTE2 完成
    └── SetAicToAiv() -> 通知 AIV 可以开始下一轮
```

#### AIV 侧执行流程 (`IterateNzNkWithAiv`)

```
// 第一阶段：预取前 DOUBLE_BUFFER_NUM 块
VecComputeNzNkWithStartLimit(0, Min(kSize, DOUBLE_BUFFER_NUM * kbL1Size))

// 第二阶段：RowIndex + Logits 预取
CopyRowIndexLogitsGmToUb()

// 第三阶段：处理上一轮结果（如果有）
if (cvLoopIdx > 0):
    ├── SetAivToAic() -> 通知 AIC 可以 Fixpipe
    ├── WaitAicToAiv() -> 等待 AIC Fixpipe 完成
    ├── MulLogits() -> 乘 logits
    ├── RoutingYToGm() -> Scatter-Add 写回 GM

// 第四阶段：处理剩余 K 块
VecComputeNzNkWithStartLimit(kMte2Offset, kSize)
```

#### L1 空间布局

**有 Bias 场景**:
```
L1 (0~512KB):
  WeightL1_P0(64KB) | Bias_P0(4KB) | ScaleAL1_P0(32KB) | ScaleBL1_P0(32KB) | AL1_P0(124KB) |
  AL1_P1(124KB)     | ScaleBL1_P1(32KB) | ScaleAL1_P1(32KB) | Bias_P1(4KB) | WeightL1_P1(64KB)
```

**无 Bias 场景**:
```
L1 (0~512KB):
  WeightL1_P0(64KB) | ScaleAL1_P0(32KB) | ScaleBL1_P0(32KB) | AL1_P0(128KB) |
  AL1_P1(128KB)     | ScaleBL1_P1(32KB) | ScaleAL1_P1(32KB) | WeightL1_P1(64KB)
```

---

### 8.3 CubeCompute 层（AIC 侧）

**文件**: `gmm_fr_weight_quant_cube_compute.h` + `gmm_fr_weight_quant_cube_compute_tools.h`
**类**: `GMMFRWeightQuantCubeCompute`

#### 职责

1. **数据搬运**: 将 A 矩阵（ND2NZ）、MX Scale A/B（Dn2Nz）从 GM 搬到 L1
2. **L1->L0 加载**: 将 A/B 矩阵和对应 MX Scale 融合加载到 L0A/L0B
3. **MMAD 计算**: 在 Cube 单元上执行矩阵乘累加
4. **Fixpipe 输出**: 将 L0C 结果通过 Fixpipe 搬到 UB（RowMajor 排布）

#### 关键数据结构

```cpp
// L0 Buffer
LocalTensor<L0DataType> l0a_{TPosition::A2, ...};  // L0A (64KB)
LocalTensor<L0DataType> l0b_{TPosition::B2, ...};  // L0B (64KB)
LocalTensor<float> l0c_{TPosition::CO1, ...};      // L0C (256KB)
LocalTensor<float> biasTable_{TPosition::C2, ...}; // Bias Table (4KB)

// L1 Buffer（双缓冲）
LocalTensor<xType> aL1_;                  // A 矩阵 L1
LocalTensor<fp8_e8m0_t> mxScaleAL1_;      // MX Scale A L1
LocalTensor<fp8_e8m0_t> mxScaleBL1_;      // MX Scale B L1
```

#### 核心流程: `LaunchMatmul`

```
LaunchMatmul(weightL1, kbOffset, kbL1RealSize, cvLoopIdx, param)
│
├── 计算 L0C offset（避免 L0C 溢出）
│
├── for l1KOffset in [0, kbL1RealSize) step kL0Size:
│   ├── WaitFlag<M_MTE1> -> 等待上一轮 MMAD 完成
│   │
│   ├── LoadAAndScaleL1ToL0()
│   │   └── LoadData(l0a, aL1, mxScaleAL1, load2dParams, load2dMxParams)
│   │       - A 矩阵: 2D 分形加载
│   │       - MX Scale: 与 A 矩阵融合加载（硬件自动做 scale 乘法）
│   │
│   ├── LoadBAndScaleL1ToL0()
│   │   └── LoadData(l0b, weightL1, mxScaleBL1, load2dParams, load2dMxParams)
│   │       - B 矩阵: 2D 分形加载
│   │       - MX Scale: 与 B 矩阵融合加载
│   │
│   └── DoMmadCompute()
│       ├── if (hasBias && isFirstKLoop): LoadBiasToBt()
│       ├── SetFlag<MTE1_M> -> 通知 MTE1 可以继续搬运
│       ├── WaitFlag<MTE1_M> -> 等待 MTE1 完成
│       └── MmadCompute(l0c, l0a, l0b, [biasTable])
│           - isFirstKLoop -> cmatrixInitVal=true（初始化累加器）
│           - isLastKLoop -> unitFlag=UPDATE（通知 Fixpipe 可读取）
```

#### MX Scale 融合搬运

MX Scale 的加载与矩阵数据加载**硬件融合**：在 `LoadData` 指令中同时传入数据 Tensor 和 Scale Tensor，硬件自动完成 `data x scale` 的乘法操作，无需额外的 VF 指令。

---

### 8.4 VecCompute 层（AIV 侧）

**文件**: `gmm_fr_weight_quant_vec_compute.h` + `gmm_fr_mx_a8w4_vf.h`
**类**: `GMMFRWeightQuantVecCompute`

#### 职责

1. **Weight 搬运**: MTE2 将 Weight（FP4 NZ 格式）从 GM 搬到 UB
2. **Weight 反量化**: VF 将 4bit 权重解包为 8bit，做 NZ->NK 排布转换
3. **Bias 缩放**: 有 Bias 时，将 Bias 乘以 MX 缩放因子（0.015625）
4. **结果写回 L1**: MTE3 将反量化后的高精度 Weight 和 Bias 写回 L1
5. **MulLogits**: 将 AIC Fixpipe 输出的结果乘以对应 token 的 logits
6. **Routing (Scatter-Add)**: 按 rowIndex 将结果 AtomicAdd 到 GM 输出

#### UB Buffer 布局

```
UB Layout (MxA8W4):
+---------------------+------------------+------+-----------+----------+---------+
| weightLowBit (80KB) | weightHighBit    | Bias | BiasRescale| rowIndex | logits  |
| 4bit 权重输入        | (160KB) 8bit 输出 |(1KB) | (1KB)      | (2KB)    | (1KB)   |
| ubMte2BufferNum=4   | QUADRUPLE=4      |      |            | DOUBLE=2 | DOUBLE=2|
+---------------------+------------------+------+-----------+----------+---------+
```

#### 核心流程: `WeightAntiQuantComputeNzNk`

```
WeightAntiQuantComputeNzNk(kRealSize, biasRealSize, kGmOffset, weightHighBitL1, biasL1, offsetParam)
│
├── WaitFlag<MTE3_V> -> 等待上一轮 MTE3 完成
│
├── AntiQuantProcessNzMxA8W4()
│   ├── 构建 MxA8W4NzParams（含物理地址、循环参数、跳写 stride）
│   └── FrAntiQuantMxA8W4NzNkVf()
│       ├── [if hasBias] FrMxA8W4BiasCompute()
│       │   └── bias x 0.015625（MX Bias 缩放因子）
│       │
│       └── 4bit->8bit 解包循环:
│           for each K block:
│               for each inner block:
│                   ├── LoadAlign<DIST_US_B8> -> 4bit 解包加载
│                   ├── ShiftRight -> 提取低 4bit
│                   ├── ShiftLeft + ShiftRight -> 提取高 4bit
│                   ├── Select -> 交错合并
│                   ├── And -> 掩码保留有效位
│                   └── StoreAlign<DIST_NORM_B8> -> 写回 UB（跳写避免 bank 冲突）
│
├── SetFlag<V_MTE3> + WaitFlag<V_MTE3> -> 同步 MTE3
│
├── CopyWeightHighBitForAligned() -> MTE3: UB->L1（高精度 Weight）
│   └── DataCopy(weightHighBitL1, weightHighBit, params)
│       - 跳读 UB（因为 VF 输出是跳写的）
│       - 连续写 L1
│
└── [if hasBias] DataCopy(biasL1, biasRescale) -> Bias 写回 L1
```

#### 4bit->8bit 解包算法

FP4 E2M1 权重在 NZ 排布下以 4bit 打包存储。解包过程：

```
输入 (4bit packed):  [H L] [H L] [H L] [H L] ...  (每个字节含两个 4bit 元素)
                      |
步骤1: DIST_US_B8 加载 -> 每个 4bit 元素复制为两个
步骤2: ShiftRight 4 -> 提取高 4bit
步骤3: ShiftLeft 4 + ShiftRight 4 -> 提取低 4bit
步骤4: Select -> 交错合并高低位
步骤5: And 0x9C -> 保留 E2M1 有效位
输出 (8bit):  [H] [L] [H] [L] ...
```

#### Routing (Scatter-Add)

```cpp
void RoutingYToGm(mRealSize, ubOutputF32Buffer, offsetParam, rlLoopIdx) {
    SetAtomicAdd<float>();
    for mIdx in [0, mRealSize):
        mGmOffset = rowIndex[mIdx];
        if (mGmOffset < 0) continue;  // 跳过无效 token
        // AtomicAdd: yFp32Global[mGmOffset * nSize + nOffset] += ubOutput[mIdx * 256]
        DataCopyPad2D(yFp32Global[mGmOffset * nSize + nOffset],
                      ubOutputF32Buffer[mIdx * 256], ...);
    SetAtomicNone();
}
```

---

## 9. 跨核同步机制

GMMFR MxA8W4 使用 AIC + AIV 双核协同工作，通过 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 实现流水同步。

### 同步信号

| 信号 | 方向 | 含义 |
|------|------|------|
| `SYNC_AIC_AIV_FLAG` | AIC -> AIV | AIC 通知 AIV：A 矩阵已搬到 L1，可以开始 Weight 反量化 |
| `SYNC_AIV_AIC_FLAG` | AIV -> AIC | AIV 通知 AIC：Weight 反量化完成，可以开始 Matmul |
| `SYNC_AIV_MTE3_AIC_FIX_FLAG` | AIV -> AIC | AIV 通知 AIC：上一轮结果已取走，可以 Fixpipe 新结果 |
| `SYNC_AIC_FIX_AIV_VF_FLAG` | AIC -> AIV | AIC 通知 AIV：Fixpipe 完成，可以取结果做 MulLogits |

### 流水时序

```
时间 ->

AIC:  [CopyA_M1] [CopyA_M2] [CopyA_M3] ...
           |           |           |
AIV:       |  [WQ_M1]  |  [WQ_M2]  |  [WQ_M3] ...
           |     |     |     |     |     |
AIC:       |     |  [MMAD_M1] |  [MMAD_M2] | ...
           |     |     |     |     |     |
AIC:       |     |     |  [Fixp_M1] |  [Fixp_M2] ...
           |     |     |     |     |     |
AIV:       |     |     |     |  [MulLogits+Routing_M1] ...
```

- **AIC CopyA** 与 **AIV WeightQuant** 乒乓：AIC 搬 A 矩阵的同时，AIV 对上一块 Weight 做反量化
- **AIC MMAD** 与 **AIV MTE3** 乒乓：AIC 做 Matmul 的同时，AIV 将反量化结果写回 L1
- **AIC Fixpipe** 与 **AIV MulLogits+Routing** 乒乓：AIC 搬出结果的同时，AIV 处理上一块结果

---

## 10. 关键设计要点

### 10.1 伪量化 vs 全量化分离

通过 `V310_GMM_FR_ANTI_QUANT` 宏在编译期完全分离两条路径，避免运行时分支判断，最大化性能。

### 10.2 MX Scale 硬件融合

MX Scale 的加载与矩阵数据加载在硬件层面融合（`LoadData` 指令同时传入 data 和 scale），消除了额外的 scale 乘法指令开销。

### 10.3 动态 K 轴切分

MxA8W4 场景根据 M/N 分块大小动态选择 K 轴搬运粒度（256 或 512），在 L1 空间利用率和循环效率之间取得平衡。

### 10.4 四缓冲 + 跳写避免 Bank 冲突

Weight 反量化输出使用四缓冲（`QUADRUPLE_BUFFER_NUM=4`），并通过跳写 stride（`innerDstStride = VECTOR_REG_WIDTH x 4`）避免 UB Bank 冲突。

### 10.5 乒乓 Buffer 设计

Controller 层使用 `processId` 在两组 `offsetParam` 之间乒乓切换，BasicBlock 层在 Weight L1、Bias L1、Scale L1、A L1 均使用双缓冲，实现计算与搬运的重叠。

### 10.6 AtomicAdd Scatter

Routing 阶段使用 `SetAtomicAdd<float>()` 实现多核对同一 GM 地址的原子累加，保证 MoE combine 的正确性。

### 10.7 对 GMMSQ 的参考价值

GMMSQ (GroupedMatmulSwiGLUQuantV2) 与 GMMFR 共享相同的 Weight Quantization 架构范式：
- 相同的四层架构（Controller -> BasicBlock -> CubeCompute/VecCompute）
- 相同的 MX Scale 融合搬运机制
- 相同的 4bit->8bit 反量化 VF 实现
- 相同的乒乓 Buffer + 跨核同步模式

主要差异在于 GMMSQ 需要额外处理 SwiGLU 的 gate_proj + up_proj 双权重矩阵，以及 SiLU 激活函数。
