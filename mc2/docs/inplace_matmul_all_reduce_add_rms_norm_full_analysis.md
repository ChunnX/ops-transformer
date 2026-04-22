# inplace_matmul_all_reduce_add_rms_norm 算子全面分析报告

## 1. 算子概述

`inplace_matmul_all_reduce_add_rms_norm` 是 `matmul_all_reduce_add_rms_norm` 的 **原地(In-place) 变体**，融合 **MatMul + AllReduce + Residual Add + RMSNorm** 四阶段操作。关键区别在于 `residual` 输入张量被原地复用为输出，避免额外的内存分配。

```
x1 [M, K] × x2 [K, N]
        ↓ MatMul
   MM 结果 [M, N]
        ↓ AllReduce (跨 TP 域归约)
   归约结果 [M, N]
        ↓ + residual (原地)
   y = residual + AllReduce(MM结果)
        ↓ RMSNorm
   norm_out = RMSNorm(y)
```

### 1.1 三种量化变体

- **基础版**: `aclnnInplaceMatmulAllReduceAddRmsNorm` — FP16/BF16 无量化
- **Per-Token 量化**: `aclnnInplaceQuantMatmulAllReduceAddRmsNorm` — INT8 输入 + dequantScale
- **Weight 量化**: `aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm` — A16W4/A16W8

---

## 2. 目录结构与文件职责

```
inplace_matmul_all_reduce_add_rms_norm/
├── op_kernel/
│   └── inplace_matmul_all_reduce_add_rms_norm.cpp  (6,925字节) 内核入口(薄分发层)
├── op_host/
│   ├── inplace_matmul_all_reduce_add_rms_norm_def.cpp      (8,582字节) 算子定义
│   ├── inplace_matmul_all_reduce_add_rms_norm_infershape.cpp (8,912字节) Shape推导
│   └── op_tiling/
│       ├── inplace_matmul_all_reduce_add_rms_norm_tiling.cpp            (1,895字节) 默认 Tiling
│       ├── inplace_quant_matmul_all_reduce_add_rms_norm_tiling.cpp      (1,439字节) 量化 Tiling
│       └── inplace_weight_quant_matmul_all_reduce_add_rms_norm_tiling.cpp (1,491字节) 权重量化 Tiling
├── op_api/
│   ├── aclnn_inplace_matmul_all_reduce_add_rms_norm.cpp        (2,159字节) 基础 API
│   ├── aclnn_inplace_matmul_all_reduce_add_rms_norm.h          (3,426字节)
│   ├── aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.cpp  (2,242字节) 量化 API
│   ├── aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.h    (3,566字节)
│   ├── aclnn_inplace_weight_quant_matmul_all_reduce_add_rms_norm.cpp (2,303字节) 权重量化 API
│   └── aclnn_inplace_weight_quant_matmul_all_reduce_add_rms_norm.h   (3,914字节)
├── op_graph/
│   ├── inplace_matmul_all_reduce_add_rms_norm_proto.h          (8,212字节) 图模式注册
│   ├── inplace_matmul_all_reduce_add_rms_norm_gen_task.cpp     (2,975字节) 任务生成
│   └── fallback_inplace_matmul_all_reduce_add_rms_norm.cpp    (10,785字节) 回退路径
├── config/ascend910b/
│   ├── inplace_matmul_all_reduce_add_rms_norm_binary.json      (28,129字节) 8种二进制
│   └── inplace_matmul_all_reduce_add_rms_norm_simplified_key.ini (1,256字节)
└── tests/ut/ (省略)
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `inplace_matmul_all_reduce_add_rms_norm_def.cpp`（第18行定义 `InplaceMatmulAllReduceAddRmsNorm` 类，继承 `OpDef`；第21-155行构造函数内声明全部输入/输出/属性；第159行 `OP_ADD` 注册）

**输入张量**（第23-111行声明）:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | x1 | FP16/BF16/INT8 | 矩阵 A [s, M, K] |
| 1 | x2 | FP16/BF16/INT8/INT4 | 矩阵 B [K, N] 或 [N, K] |
| 2 | bias | FP16/BF16/INT32 (可选) | 偏置 |
| 3 | residual | FP16/BF16 | 残差输入 (同时作为输出) |
| 4 | gamma | FP16/BF16 | RMSNorm 缩放因子 |
| 5 | antiquant_scale | FP16/BF16 (可选) | 反量化缩放 (权重量化) |
| 6 | antiquant_offset | FP16/BF16 (可选) | 反量化偏移 (权重量化) |
| 7 | dequant_scale | FP16/BF16/UINT64/INT64 (可选) | 反量化缩放 (Per-Token量化) |

**输出张量**（第113-134行声明）:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | residual | FP16/BF16 | 原地输出: residual + AllReduce(MM(x1, x2)) |
| 1 | norm_out | FP16/BF16 | RMSNorm 输出 |

**关键属性**（第136-142行声明）:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| group | String | - | HCCL 通信域 |
| reduce_op | String | "sum" | 归约操作 |
| is_trans_a | Bool | false | A 是否转置 |
| is_trans_b | Bool | false | B 是否转置 |
| comm_turn | Int | 0 | 通信轮次 |
| antiquant_group_size | Int | 0 | 权重量化分组大小 |
| epsilon | Float | 1e-6 | RMSNorm epsilon |

**平台配置**（第144-155行）：仅 `ascend910b`，启用 `DynamicCompileStaticFlag`/`DynamicFormatFlag`/`DynamicRankSupportFlag`/`DynamicShapeSupportFlag`；第155行 `this->MC2().HcclGroup("group")` 注册 HCCL 通信组。

### 3.2 Shape 推导

**文件**: `inplace_matmul_all_reduce_add_rms_norm_infershape.cpp`（Shape 推导逻辑委托给 `InferShapeForMatmul()` 和 `InferShapeForMC2AddRmsNorm()` 两个通用函数）

```
InferShapeForMatmul():
├── x1: 2D 或 3D [s, M, K]
├── x2: 2D [K, N] (或转置 [N, K])
├── x1 不能转置 (is_trans_a 必须为 false)
├── K 维度必须匹配
└── antiquant_scale shape:
    ├── transB=false: [N]
    ├── transB=true: [K] (PER_TENSOR/PER_CHANNEL)
    └── PER_GROUP: [N, Ceil(K/groupSize)]

InferShapeForMC2AddRmsNorm():
├── residual: 3D [s, M, N]
├── gamma: 1D [N]
├── 输出 residual: [s, M, N] (原地)
├── 输出 norm_out: [s, M, N]
└── residual_idx = bias存在 ? 3 : 2
```

### 3.3 Tiling 策略

**设计模式**: 所有三个 Tiling 文件都是 **类型别名(Type Alias)**，完全委托给 `matmul_all_reduce_add_rms_norm` 的 Tiling 类：

```cpp
// inplace_matmul_all_reduce_add_rms_norm_tiling.cpp（第42行类型别名，第43行注册优先级: 2）
using InplaceMatmulAllReduceAddRmsNormTiling = MatmulAllReduceAddRmsNormTiling;

// inplace_quant_matmul_all_reduce_add_rms_norm_tiling.cpp（第26行类型别名，第27行注册优先级: 0）
using InplaceQuantMatmulAllReduceAddRmsNormTiling = QuantMatmulAllReduceAddRmsNormTiling;

// inplace_weight_quant_matmul_all_reduce_add_rms_norm_tiling.cpp（第25行类型别名，第26行注册优先级: 1）
using InplaceWeightQuantMatmulAllReduceAddRmsNormTiling = WeightQuantMatmulAllReduceAddRmsNormTiling;
```

三个 Tiling 策略以优先级 0/1/2 注册到同一个算子名 `InplaceMatmulAllReduceAddRmsNorm`，Tiling 框架按优先级依次尝试，首个成功的策略胜出。量化变体优先级最高（0），确保量化场景不会被通用 Tiling 截获。

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `inplace_matmul_all_reduce_add_rms_norm.cpp`（第64-67行内核入口签名，接受12个 `GM_ADDR` 参数：`aGM`、`bGM`、`biasGM`、`residualGM`、`gammaGM`、`antiquantScaleGM`、`antiquantOffsetGM`、`dequantGM`、`yGM`、`normOutGM`、`workspaceGM`、`tilingGM`）

这是一个 **纯分发层**，不包含任何计算逻辑，完全委托给 `matmul_all_reduce_add_rms_norm` 的内核实现。

```
内核分发逻辑（第86-144行）:
├── 第86-87行: TPipe tPipe; KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
├── 第88行: REGISTER_TILING_DEFAULT(MatmulAllReduceAddRmsNormTilingData)
├── 第90-94行: 按 TilingKey 注册量化 Tiling 数据:
│   ├── TilingKey == 8 或 4104 → QuantMatmulAllReduceAddRmsNormTilingData
│   └── TilingKey > 10000      → WeightQuantMatmulAllReduceAddRmsNormTilingData
│
├── MC2_QUANT_FP16 路径（第95-102行）:
│   ├── TilingKey 8    → INVOKE_MC2_ARN_QUANT_910_OP_IMPL(BmmDequant)
│   └── TilingKey 4104 → INVOKE_MC2_ARN_QUANT_910_OP_IMPL(BmmDequant, ..., true)
│
├── MC2_QUANT_BF16 路径（第103-110行）:
│   ├── TilingKey 8    → INVOKE_MC2_ARN_QUANT_910_OP_IMPL(BmmDequantBf16)
│   └── TilingKey 4104 → INVOKE_MC2_ARN_QUANT_910_OP_IMPL(BmmDequantBf16, ..., true)
│
├── MC2_WEIGHT_QUANT 路径（第111-136行）:
│   └── 12 个 TilingKey 变体（transB × 量化模式 × isBias 组合）:
│       ├── PER_TENSOR: Keys 2293772/3342348/69402636/70451212
│       ├── PER_CHANNEL: Keys 4390924/5439500/71499788/72548364
│       └── PER_GROUP: Keys 6488076/73596940/7536652/74645516
│
└── 默认路径（第137-144行，无量化）:
    ├── TilingKey 256/260 → INVOKE_MC2_ARN_910_OP_IMPL(Mc2MatmulBaseKernel)
    └── TilingKey 0       → INVOKE_MC2_ARN_910_OP_IMPL(Mc2MatmulBaseUnAlignedKernel)
```

**编译期条件分支**：`MC2_QUANT_FP16`/`MC2_QUANT_BF16`/`MC2_WEIGHT_QUANT` 为编译宏，同一份源码只编译一个分支。这意味着每个二进制只包含一条路径的代码，分发发生在编译期而非运行期。

**包含关系**（第28-57行）：
- 第28-32行: 包含 `matmul_all_reduce` 的通用头文件
- 第33-57行: 根据编译宏选择包含 `matmul_all_reduce_add_rms_norm` 的具体变体头文件（如 `matmul_all_reduce_add_rms_norm_tiling_data.h`）

### 4.2 依赖关系

```
inplace_matmul_all_reduce_add_rms_norm 内核
├── #include "matmul_all_reduce_add_rms_norm/op_kernel/..." (基础版+权重量化)
└── #include "matmul_all_reduce/op_kernel/..."              (Per-Token量化)

所有计算逻辑在依赖的算子中实现，本算子只做 Tiling 数据分发和宏展开。
```

---

## 5. Host-Kernel 接口

由于本算子是 `matmul_all_reduce_add_rms_norm` 的原地变体，所有 Tiling 数据结构和 Workspace 布局均复用父算子的定义。

### 5.1 Tiling 数据结构

Tiling 数据结构定义在 `matmul_all_reduce_add_rms_norm/op_kernel/matmul_all_reduce_add_rms_norm_tiling_data.h`（`namespace Mc2Tiling`，`#pragma pack(push, 8)` 对齐）。本算子通过类型别名直接使用父算子的三种 TilingData：

```
非量化版 MatmulAllReduceAddRmsNormTilingData（第52-57行）:
├── matmulAllReduceTilingData: MatmulAllReduce910TilingData
│   ├── msg: Mc2Msg                          (mc2_tiling_struct.h 第53-103行)
│   │   ├── send/recvOffsets, counts, turn
│   │   ├── commAlg: 通信算法选择
│   │   └── wait/exit 策略, stepSize
│   ├── param: RCSTiling                     (mc2_tiling_struct.h 第105-134行)
│   │   └── rankDims, tileCounts, workspaceOffsets
│   ├── tilematmulTiling: MC2MatmulV3TilingData  (matmul tile 配置)
│   └── tailmatmulTiling: MC2MatmulV3TilingData  (matmul tail 配置)
├── addRMSNormTileTilingData: AddRMSNormTilingData  (第35-43行)
│   ├── numRow, numCol, blockFactor, rowFactor, ubFactor
│   └── epsilon, avgFactor
├── addRMSNormTailTilingData: AddRMSNormTilingData  (tail 复用同结构)
└── addRmsNormTilingeKeyData: AddRMSNormTilingeKeyData  (第45-50行)
    ├── ARNKeyTile, ARNKeyTail
    └── ARNNumBlocksTile, ARNNumBlocksTail
```

量化变体替换首字段：
- `QuantMatmulAllReduceAddRmsNormTilingData`（第59-64行）：首字段为 `QuantMatmulAllReduceTilingData`
- `WeightQuantMatmulAllReduceAddRmsNormTilingData`（第66-71行）：首字段为 `WeightQuantMatmulAllReduceTilingData`

Host 侧 Tiling 计算流程（`matmul_all_reduce_add_rms_norm_tiling.cpp`）：
1. 第57-93行 `DoOpTiling()`：先委托 `MMNTilingTransferHelper` 计算 matmul+allreduce Tiling，再调用 `Tiling4AddRmsNorm` 两次（tile + tail）
2. 第126-142行 `GetWorkspaceSize()`：`myWorkSpace = mc2WorkSpace + (arnWorkSpace - SYS_WORKSPACE_SIZE)`
3. 第144-184行 `PostTiling()`：拷贝 `sizeof(MatmulAllReduceAddRmsNormTilingData)` 到 Tiling buffer，设置 blockDim

### 5.2 Workspace 布局

```
Workspace 总布局:
┌─────────────────────────────────────────────────────────┐
│ mc2WorkSpace                                            │
│ ├── HCCL 通信缓冲区                                      │
│ ├── MatMul 中间结果缓存                                   │
│ └── AllReduce 数据交换区                                  │
├─────────────────────────────────────────────────────────┤
│ arnWorkSpace                                            │
│ ├── 用户数据区 (256 Bytes)                               │
│ └── 系统工作区 (SYS_WORKSPACE_SIZE = 16 MB)              │
│     └── 注: 16 MB 已计入 mc2WorkSpace，实际不重复分配       │
└─────────────────────────────────────────────────────────┘

计算公式 (第126-142行):
  totalWorkspace = mc2WorkSpace + arnWorkSpace - SYS_WORKSPACE_SIZE
```

---

## 6. 完整数据流

```
Host 侧                              Kernel 侧
────────                              ────────
aclnnInplace*GetWorkspaceSize()
  │  (residual 同时作为 y 输出)
  └→ aclnn*GetWorkspaceSize()
       │  计算 Tiling → MatmulAllReduceAddRmsNormTilingData
       │  计算 Workspace 大小
       └→ 返回 workspaceSize, executor

aclnnInplace*(workspace, executor)
  └→ aclnn*(workspace, executor)      inplace_matmul_all_reduce_add_rms_norm.cpp
                                       │  第64行: 内核入口
                                       │  第88行: 解析 TilingData
                                       │  第137-144行: 分发到父算子内核
                                       ↓
                                      Mc2MatmulBaseKernel::Compute()
                                       │
                    ┌──────────────────┤
                    │ AIC (Cube)       │ AIV (Vector)
                    │                  │
                    │ MatMul 分片计算    │
                    │ ↓                │
                    │ AllReduce(HCCL)  │
                    │ ↓                │
                    │                  │ Residual Add: residual += AllReduce结果
                    │                  │ ↓
                    │                  │ RMSNorm(y, gamma)
                    │                  │ ↓
                    │                  │ normOut 输出
                    └──────────────────┘

内存操作:
  x1 [s,M,K] ──→ GM 输入
  x2 [K,N]   ──→ GM 输入
  residual   ──→ GM 输入/输出（原地写回 +AllReduce 结果）
  normOut    ──→ GM 输出（RMSNorm 结果）
  workspace  ──→ HCCL 通信缓冲 + MatMul 中间结果
```

**原地语义的关键路径**：ACLNN 层将 `residual` 张量同时作为非原地版本的 `residual` 输入和 `y` 输出传入，使 MatMul+AllReduce 的结果直接累加到 `residual` 所在内存，省去一次完整 `[s,M,N]` 张量的分配。

---

## 7. ACLNN API

### 7.1 三套 API

```
基础版 (无量化):
  aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(x1, x2, bias, residual, gamma,
      epsilon, group, reduceOp, commTurn, streamMode, normOut, ...)
  → 委托到 aclnnMatmulAllReduceAddRmsNorm (residual 同时作为 y 输出)

Per-Token 量化版:
  aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(x1, x2, bias, dequantScale,
      residual, gamma, epsilon, group, reduceOp, commTurn, streamMode, normOut, ...)
  → 委托到 aclnnQuantMatmulAllReduceAddRmsNorm

Weight 量化版:
  aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(x1, x2, bias,
      antiquantScale, antiquantOffset, residual, gamma, epsilon, group, reduceOp,
      commTurn, streamMode, antiquantGroupSize, normOut, ...)
  → 委托到 aclnnWeightQuantMatmulAllReduceAddRmsNorm
```

**实现细节**（`aclnn_inplace_matmul_all_reduce_add_rms_norm.cpp`）：
- 第38-46行：`aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize` — 直接委托给 `aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize`，将 `residual` 同时作为 `y` 输出传入
- 第48-52行：`aclnnInplaceMatmulAllReduceAddRmsNorm` — 直接委托给 `aclnnMatmulAllReduceAddRmsNorm`

### 7.2 原地语义

```
非原地版本:
  residual_input + AllReduce(MatMul(x1, x2)) → y (新分配)
  RMSNorm(y) → norm_out

原地版本:
  residual_input += AllReduce(MatMul(x1, x2))  ← 直接写回 residual
  RMSNorm(residual_input) → norm_out

ACLNN 实现方式:
  调用非原地版本的 API，但将 residual 张量同时作为 y 输出传入
```

---

## 8. 二进制配置

**文件**: `config/ascend910b/inplace_matmul_all_reduce_add_rms_norm_binary.json`

8 种内核变体:

| 变体 | x1 类型 | x2 类型 | bias 类型 | residual 类型 | 量化模式 |
|------|---------|---------|-----------|--------------|---------|
| ND_ND_FP16_FP16 | FP16 | FP16 | FP16 | FP16 | 无 |
| ND_ND_BF16_BF16 | BF16 | BF16 | BF16 | BF16 | 无 |
| A16W8_ND_ND_FP16_INT8 | FP16 | INT8 | - | FP16 | Weight量化W8 |
| A16W8_ND_ND_BF16_INT8 | BF16 | INT8 | - | BF16 | Weight量化W8 |
| A16W4_ND_ND_FP16_INT4 | FP16 | INT4 | - | FP16 | Weight量化W4 |
| A16W4_ND_ND_BF16_INT4 | BF16 | INT4 | - | BF16 | Weight量化W4 |
| A8W8_ND_ND_INT8_INT8_FP16 | INT8 | INT8 | INT32 | FP16 | Per-Token量化 |
| A8W8_ND_ND_INT8_INT8_BF16 | INT8 | INT8 | INT32 | BF16 | Per-Token量化 |

---

## 9. 与 matmul_all_reduce_add_rms_norm 的核心差异

| 特性 | matmul_all_reduce_add_rms_norm | inplace_matmul_all_reduce_add_rms_norm |
|------|-------------------------------|---------------------------------------|
| 输出 y | 独立分配 | residual 原地复用 |
| 内存占用 | y + norm_out | norm_out (省一个 y) |
| 内核逻辑 | 完整实现 | 纯分发层,委托给前者 |
| Tiling | 完整实现 | 类型别名,委托给前者 |
| ACLNN API | 独立 | 委托 (传 residual 为 y) |
| 平台 | 910b + 910_93 | 仅 910b |

**设计模式**: 典型的 "Thin Wrapper" 模式 — inplace 版本在所有层 (kernel/tiling/api) 都是非 inplace 版本的轻量封装，唯一区别是内存复用语义。

---

## 10. 函数调用关系

```
inplace_matmul_all_reduce_add_rms_norm (内核入口)
├── [默认路径] → INVOKE_MC2_ARN_910_OP_IMPL(Mc2MatmulBaseKernel)
│   └── matmul_all_reduce_add_rms_norm 内核逻辑
│       ├── MatMul (AIC)
│       ├── AllReduce (HCCL)
│       ├── Residual Add (AIV)
│       └── RMSNorm (AIV)
│
├── [Per-Token量化] → INVOKE_MC2_ARN_QUANT_910_OP_IMPL(BmmDequant)
│   └── quant_matmul_all_reduce_add_rms_norm 内核逻辑
│
└── [Weight量化] → INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL
    └── weight_quant_matmul_all_reduce_add_rms_norm 内核逻辑

ACLNN 层:
  aclnnInplace*GetWorkspaceSize → aclnn*GetWorkspaceSize (residual 作为 y)
  aclnnInplace* → aclnn* (residual 作为 y)
```

---

*分析完成时间: 2026-04-17，修订时间: 2026-04-21（补充行号引用、Host-Kernel接口、完整数据流章节）*
