# PR 代码概要

PR: #6142 | 算子: FlashAttention (common memcopy 模块) | 侧别: Kernel

## 变更文件概览

| # | 文件 | 侧别 | 变更类型 | 说明 |
|---|------|------|---------|------|
| 1 | attention/common/op_kernel/memcopy/copy_gm_to_l1.h | Kernel | 未直接变更于 diff | 核心分析文件，GM→L1 数据搬运类模板（Flash Attention 基础设施） |

> **注意**：该文件在 diff 中无直接变更行。PR #6142 为一个大范围 PR（diff 共 44387 行），涉及 CMakeLists（mc62→mc62cm12a 芯片名更新）、文档修改（aclnnBlockSparseAttention 系列）、arch35 内核 bugfix（int64_t→int32_t 类型修正、变量初始化简化）、以及 GMM 侧文件重命名（gmm_copy_gm_to_l1.hpp→copy_gm_to_l1.hpp）等。`copy_gm_to_l1.h` 作为 Flash Attention 通用 memcopy 基础设施，被 `memory_copy.h` 和 `arch35/flash_attention_noquant_block_cube.h` 等文件 include，是理解 PR 中 arch35 变更的关键上下文。

## 文件1: attention/common/op_kernel/memcopy/copy_gm_to_l1.h（Kernel 侧）

### 变更概要
该文件为头文件（header-only），提供模板化的 GM→L1 数据搬运功能，核心是将 GM 上不同 layout（BSNGD/BNGSD/TNGD/NGTD/BNSD 等 24 种 GmFormat）的数据转换为 L1 上的 NZ 格式。包含 Query/KV/Scale 等 6 个搬运类，以及 ND→NZ/DN→NZ 等格式转换 helper 函数。此文件是 Flash Attention 系列算子的数据搬运基础设施，虽然本身在本 PR 中无直接修改，但其被 arch35 内核广泛使用（如 `fia_block_cube_noquant_gqa.h`, `flash_attention_noquant_block_cube.h`），是理解 arch35 相关变更的关键依赖。

### 代码脉络

**入口**: 无统一入口函数。各模板类的 `operator()` 被上层 Flash Attention 内核调用（如 `CopyQueryGmToL1::operator()`, `CopyKvGmToL1::operator()`），触发条件为内核主循环中的 GM→L1 搬运阶段。

**数据流**:
```
GM Tensor (GlobalTensor<T>, 各种 GmFormat layout)
    → OffsetCalculator 计算 offset
    → ND2NZ / DN2NZ / DataCopyParams 搬运
    → L1 Tensor (FaL1Tensor<T, L1Format::NZ>)
    → 后续计算（MatMul, Softmax 等）
```

**计算核心**: 各类 `operator()` 是主调度函数，内部根据 `GmFormat` 编译期分派至 `ProcessS1G()`, `ProcessGS1()`, `ProcessContinuous()`, `ProcessPageAttention()` 等具体处理方法。处理逻辑按"首块→中间块（可能跨 stride 边界）→尾块"三段式设计，每段调用 `CopySingleMatrixNDToNZ()` 或 `CopyMultiMatrixNDToNZ()` 完成矩阵格式转换搬运。

**分支覆盖**:
| 分支条件 | 位置(行) | 触发场景 | 处理逻辑 | 涉及 API |
|---------|---------|---------|---------|----------|
| `GM_FORMAT == GmFormat::BSNGD \|\| TNGD` | 115 | Query 数据为 BSNGD 或 TNGD format | 调用 ProcessS1G：按 G 维度切分，处理跨 G 边界场景 | DataCopy (Nd2NzParams) |
| `GM_FORMAT == GmFormat::BNGSD` | 117 | Query 数据为 BNGSD format | 根据 actualSeqLens 决定 ProcessGS1 或 ProcessContinuous | DataCopy (Nd2NzParams) |
| `GM_FORMAT == GmFormat::NGTD` | 124 | Query 数据为 NGTD format | 调用 ProcessGS1 | DataCopy (Nd2NzParams) |
| `offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0` | 119 | BNGSD 下有变长序列 | ProcessGS1：按实际序列长度动态切分 | DataCopy (Nd2NzParams) |
| `offsetCalculator.actualSeqLensQParser.GetActualLenDims() == 0` | 122 | BNGSD 下定长序列 | ProcessContinuous：连续搬运 | DataCopy (Nd2NzParams) |
| `srcDValue < 65536` (CopySingleMatrixNDToNZ) | 30 | 单次搬运 D 维度较小 | 一次性 DataCopy 整块搬运 | DataCopy (Nd2NzParams) |
| `srcDValue >= 65536` | 46 | 单次搬运 D 维度很大 | 逐行分块搬运（每行 16 或 64 元素 block） | DataCopy (Nd2NzParams) |
| `srcNdMatrixStride > 65536` (CopyMultiMatrixNDToNZ) | 80 | 多矩阵 stride 过大 | 降级为逐矩阵单独调用 CopySingleMatrixNDToNZ | DataCopy (Nd2NzParams) |
| `srcNdMatrixStride <= 65536` | 89 | 正常 stride 范围 | 批量 Multi-Matrix DataCopy | DataCopy (Nd2NzParams) |
| KV `GM_FORMAT == BNSD\|BSND\|NTD\|TND` | 275-276 | 普通 KV 格式 | ProcessContinuousOrTensorlist：单次连续搬运 | DataCopy (Nd2NzParams) |
| KV `GM_FORMAT == PA_*` (PageAttention) | 278-280 | PA 场景（KV Cache 分页） | ProcessPageAttention：按 BlockTable 逐 block 搬运 | DataCopy (Nd2NzParams / DataCopyParams) |
| `IsSameType<T, int4b_t>::value` | 33, 53, 92 | int4b 类型 | D 维度值除以 2，blockElementCnt=64 | DataCopy (Nd2NzParams) |
| `!IsSameType<T, int4b_t>::value` | 37, 57, 96 | 非 int4b 类型 | 使用原始 D 维度值，blockElementCnt=16 | DataCopy (Nd2NzParams) |
| `__CCE_AICORE__ == 310 \|\| __DAV_310R6__` | 444 | Arch310/310R6 平台 | 编译 QueryScale/CopyKeyScale/CopyValueScale 相关代码（MX Scale） | DataCopy (Dn2NzParams/Nd2NzParams/DataCopyParams) |
| Scale DN→NZ vs DN→NZ stride 正常/超限 | 80, 88 | MX Scale 搬运的 stride 路径选择 | 与 Query 类似的三段式分块策略 | DataCopy (Dn2NzParams) |
| `GM_FORMAT == PA_NZ_K_SCALE` | 642 | K Scale 的 PA_NZ format | 特殊 DataCopyParams 参数（blockCount/blockLen 不同计算方式） | DataCopy (DataCopyParams) |
| `ScaleTrans == ND2NZ` | 729, 764 | V Scale ND→NZ 转换 | CopySingleMXScaleNDToNZ | DataCopy (Nd2NzParams) |
| `ScaleTrans == DN2NZ` | 732, 768 | V Scale DN→NZ 转换 | CopySingleMXScaleDNToNZ | DataCopy (Dn2NzParams) |

**关键变量流转**:
| 变量 | 来源 | 用途 | 流转路径 |
|------|------|------|----------|
| `gmCoord` (GmCoord struct) | 上层内核 Compute 函数传入 | 指定搬运的 batch/head/S1/D 索引及处理量 | 传入 operator() → 各 Process 函数计算 offset |
| `gmCoord.gS1Idx` | 上层调度器计算 | 当前处理的 S1 起始位置 | ProcessS1G/ProcessGS1 中切分首/中/尾块 |
| `gmCoord.dDealSize` / `gmCoord.gS1DealSize` | 上层 UBs 分配/Tiling 计算 | 单次 D/S1 维度处理量 | 传入 DataCopy/ND2NZ 函数 |
| `srcTensor.offsetCalculator` | FaGmTensor 成员 | 基于 GmLayout 和 Tiling 信息计算 GM offset | GetOffset() → 计算 GM 起始地址 |
| `srcTensor.gmTensor` | Global Tensor 基地址 | 实际 GM 数据源 | `gmTensor[offset]` 取址 → DataCopy |
| `dstTensor.tensor` | L1 上的 LocalTensor | 目标 L1 缓冲区 | `dstTensor.tensor[l1Offset]` → DataCopy 目标 |
| `dstTensor.rowCount` | FaL1Tensor 成员（上层初始化） | NZ 格式中 C0 维度的 stride | 传入 Nd2NzParams::dstNzC0Stride |
| `blockElementCnt` = `32 / sizeof(T)` | 编译期常量 | 32B 对齐的每行元素数 | L1 offset 累加计算 |
| `srcDValue / dstNzC0Stride` | offsetCalculator stride | 行列 stride 参数 | 传入 Nd2NzParams/Dn2NzParams |
| `s1Size` | 从 actualSeqLensQParser 或 GetDimS1() 获取 | TND/BNGSD 格式下的 S1 维度实际长度 | ProcessGS1 中 gIdx/s1Idx 拆分 |
| `blockTableParser.GetBlockIdx()` | PA 场景从 GM BlockTable 查表获取 | PA 物理 block 索引 | ProcessPageAttention 中 blockIdx → offset 计算 |

**核心 API**: `DataCopy`（含 `Nd2NzParams`, `Dn2NzParams`, `DataCopyParams` 三种参数结构体）、`LocalTensor::GetPhyAddr()`, `GlobalTensor::SetGlobalBuffer()`, `ReinterpretCast<bfloat16_t>()`

**输出**: 写入 `FaL1Tensor<T, L1_FORMAT>::tensor`（L1 LocalTensor），无显式 EnQue/DeQue——同步由上层内核 Pipe 管理。

## 变量溯源

| 变量 | 声明(文件:行) | 初始化(文件:行) | 校验(文件:行) | 来源类型 |
|------|-------------|----------------|-------------|---------|
| `gmCoord` (GmCoord) | gm_coord.h:18-25 (struct 定义) | 上层内核计算 | 上层内核 | 外部输入（上层调度器） |
| `srcTensor.gmTensor` | fa_gm_tensor.h:31 | 上层内核 Pipe.InitBuffer + SetGlobalBuffer | 上层 | 外部输入 |
| `srcTensor.offsetCalculator` | fa_gm_tensor.h:32, offset_calculator_v2.h:1150-1152 | 上层 Init() 调用 | 上层 | Tiling 传递 |
| `dstTensor.tensor` | fa_l1_tensor.h:39 | 上层 Pipe.InitBuffer | 上层 | Pipe 管理 |
| `dstTensor.rowCount` | fa_l1_tensor.h:40 | 上层赋值 | 未在此文件校验 | 外部输入 |
| `SRCDVALUE_LIMIT = 65536` | copy_gm_to_l1.h:28 | constexpr | 编译期 | 编译期常量 |
| `ND_MATRIX_STRIDE_LIMIT = 65536` | copy_gm_to_l1.h:79 | constexpr | 编译期 | 编译期常量 |
| `BLOCK_ELEMENT_CNT` (16 or 64) | copy_gm_to_l1.h:47 | constexpr (根据类型编译期决定) | 编译期 | 编译期常量 |
| `blockElementCnt` = `32 / sizeof(T)` | copy_gm_to_l1.h:165, 238, 303, 394, 641, 744 | 运行时计算（基于 T 类型） | 编译期类型安全 | 编译期常量（T 类型决定） |
| `s1Size` | copy_gm_to_l1.h:204-213, 554-563 | 运行时从 actualSeqLensQParser / GetDimS1 获取 | 上层校验 actualSeqLengths | Tiling 传递 / 硬件配置 |
| `GM_FORMAT` 模板参数 | 类模板参数 | 上层内核实例化时指定 | 编译期 | 编译期常量 |

> 来源类型判定：Tiling 传递 → 已校验；编译期常量（constexpr/const）→ 编译期固定；外部输入 → 需关注

## 代码关联

**上游文件**:
| 文件路径 | 关联方式 | 依据 |
|----------|----------|------|
| gm_layout.h | include | 代码中 `#include "gm_layout.h"` (行18)，定义 GmFormat 枚举和 GmLayout 模板 |
| fa_l1_tensor.h | include | 代码中 `#include "fa_l1_tensor.h"` (行19)，定义 FaL1Tensor 结构体和 L1Format/ScaleTrans 枚举 |
| gm_coord.h | include | 代码中 `#include "gm_coord.h"` (行20)，定义 GmCoord 和 GmKvCoord 结构体 |
| offset_calculator_v2.h | include | 代码中 `#include "offset_calculator_v2.h"` (行21)，定义 OffsetCalculator 模板和 GmLayoutParams/FormatCategory/GmLayout |
| parser.h | 间接 include | 通过 offset_calculator_v2.h → parser.h，定义 ActualSeqLensParser 和 BlockTableParser |
| kernel_operator.h 或 kernel_vec_intf.h + kernel_cube_intf.h | 间接 include | 通过 gm_layout.h 根据 ASC_DEVKIT_MAJOR 版本条件编译 include，提供 LocalTensor/GlobalTensor/DataCopy 等 AscendC API |

**下游文件**:
| 文件路径/API | 关联方式 | 依据 |
|----------|----------|------|
| attention/common/op_kernel/memory_copy.h | include 本文件 | Grep 结果：`#include "memcopy/copy_gm_to_l1.h"` (memory_copy.h:28) |
| attention/common/op_kernel/arch35/flash_attention_noquant_block_cube.h | include 本文件 | Grep 结果：`#include "../memcopy/copy_gm_to_l1.h"` (行22) |

**AscendC API 依赖**:
| API | 用途 | 参数结构体 | 出现位置 |
|-----|------|-----------|---------|
| DataCopy (Nd2NzParams) | ND 格式矩阵→NZ 格式搬运 | Nd2NzParams (ndNum, nValue, dValue, srcDValue, dstNzC0Stride 等) | CopySingleMatrixNDToNZ, CopyMultiMatrixNDToNZ, CopySingleMXScaleNDToNZ |
| DataCopy (Dn2NzParams) | DN 格式矩阵→NZ 格式搬运 (MX Scale 专用) | Dn2NzParams (dnNum, nValue, dValue, srcDnMatrixStride, srcDValue, dstNzC0Stride 等) | CopySingleMXScaleDNToNZ, CopyMultiMXScaleDNToNZ |
| DataCopy (DataCopyParams) | 通用 intri 数据搬运 (PA_NZ 场景) | DataCopyParams (blockCount, blockLen, dstStride, srcStride) | ProcessPageAttention (PA_NZ 分支), ProcessPageAttention (PA_NZ_K_SCALE 分支) |
| ReinterpretCast<bfloat16_t>() | 类型重解释 (MX Scale Dn2Nz 搬运前) | — | CopySingleMXScaleDNToNZ, CopyMultiMXScaleDNToNZ, CopySingleMXScaleNDToNZ |
| GetPhyAddr() / SetGlobalBuffer() | 获取 GM tensor 物理地址 | — | MX Scale 函数中 bfloat16 转换 |

## 高性能设计（仅 Kernel 侧）

**流水线模式**: 纯 Vector（无直接 EnQue/DeQue——同步由上层调度器/Pipe 管理，本文件只做数据格式转换 + 搬运）

**流水线设计**:
| 机制 | 状态 | 设计意图 |
|------|------|----------|
| EnQue/DeQue 同步 | 无（在本文件内） | 本文件是纯数据搬运模板，不管理 Pipe 同步；同步在外层 FA 内核的 Pipe 管理中完成 |
| Buffer 管理模式 | 单缓冲（本文件视角） | 每次 operator() 调用将指定 GM 区域数据搬运到调用者传入的 L1 tensor；Double Buffer 缓冲切换由上层内核管理 |

**切分策略**（需给出具体数值）:
| 维度 | 切分方式 | 每核处理量 | 切分粒度 |
|------|----------|-----------|---------|
| 多核切分 | 不涉及（本文件无多核切分逻辑） | N/A | 由上层 Tiling 侧确定 |
| **S1/G 维度切分 (ProcessS1G)** | 按 G 维度边界切分：首块(headSize为 gIdxEnd-gIdxStart 或 GetDimG()-gIdxStart)，中间块(GetDimG() 行)，尾块(gIdxEnd 行) | 取决于 gmCoord.gS1DealSize | 每行 blockElementCnt 个元素（16 or 64） |
| **GS1 维度切分 (ProcessGS1)** | 按 S1 实际长度边界切分，含变长序列处理 | s1Size 由 actualSeqLensQParser 确定 | 行级 blockElementCnt=32/sizeof(T) |
| **PA 分块搬运 (ProcessPageAttention)** | 按 BlockSize 切分，跨 block 边界时分段搬运 | copyRowCnt = min(BlockSize - curS2Idx%BlockSize, 剩余行数) | blockElementCnt 粒度 |

**Buffer 规划**:
| Buffer | 类型 | 大小(B) | 用途 |
|--------|------|------|------|
| dstTensor.tensor (FaL1Tensor) | LocalTensor (L1) | 由上层 kernel 的 Pipe.InitBuffer 决定（具体值不在本文件内） | 接收从 GM 搬运的数据（NZ 格式） |
| blockElementCnt | 编译期常量 | 32 B（Per row in NZ） | NZ 格式每行对齐 32B |

---

## 文件整体结构摘要

```
copy_gm_to_l1.h (781行)
├── Helper 函数（GM→L1 核心搬运原语）
│   ├── CopySingleMatrixNDToNZ<T>()          (L25-71)   - 单矩阵 ND→NZ，D 维度分大小两种情况
│   ├── CopyMultiMatrixNDToNZ<T>()           (L73-106)  - 多矩阵 ND→NZ，stride 分超限/正常两种情况
│   ├── CopySingleMXScaleDNToNZ<T>()         (L446-464) - 单 MX Scale DN→NZ（Arch310 专用）
│   ├── CopyMultiMXScaleDNToNZ<T>()          (L466-486) - 多 MX Scale DN→NZ（Arch310 专用）
│   └── CopySingleMXScaleNDToNZ<T>()         (L686-704) - 单 MX Scale ND→NZ（Arch310 专用）
│
├── CopyQueryGmToL1<Q_T, GM_FORMAT, L1_FORMAT>  (L108-256)
│   ├── operator()    → 编译期分派到 ProcessS1G / ProcessGS1 / ProcessContinuous
│   ├── ProcessS1G()  → 按 G 维度切分（BSNGD/TNGD）
│   ├── ProcessContinuous() → 连续搬运（BNGSD 定长）
│   └── ProcessGS1()  → 按 S1 实际长度切分（BNGSD 变长/NGTD）
│
├── GmKvCoord 结构体 (L259-266) — KV 搬运坐标
│
├── CopyKvGmToL1<KV_T, GM_FORMAT, L1_FORMAT>  (L268-336)
│   ├── operator()    → 编译期分派到 ProcessContinuousOrTensorlist / ProcessPageAttention
│   ├── ProcessContinuousOrTensorlist() → 单次连续搬运（BNSD/BSND/NTD/TND）
│   └── ProcessPageAttention() → PA 逐 block 搬运（PA_BnBsND/PA_BnNBsD/PA_NZ）
│
├── CopyKKropePAGmToL1<KV_T, GM_FORMAT, L1_FORMAT>  (L338-442)
│   ├── operator()    → 编译期分派到 ProcessPageAttention（双 tensor：K + K-RoPE）
│   ├── GetBlockIdx()  → BlockTable 查表
│   ├── GetOffset()    → PA offset 计算（PA_NZ 分支含 D1/D0 维度）
│   └── ProcessPageAttention() → 逐 block 双搬运
│
├── #if (__CCE_AICORE__ == 310) || (__DAV_310R6__) — Arch310 专属段 (L444-780)
│   ├── CopyQueryScaleGmToL1<Q_SCALE_T, GM_FORMAT, L1_FORMAT>  (L488-605) — Query Scale 搬运
│   │   ├── operator() → ProcessS1G / ProcessS1 / ProcessContinuous / ProcessGS1
│   │   ├── ProcessS1G() → NTGD format 单次搬运
│   │   ├── ProcessS1() → TNGD format 带 N2 维度 stride
│   │   ├── ProcessContinuous() → BNGSD 定长连续搬运
│   │   └── ProcessGS1() → BNGSD 变长/切分搬运
│   ├── CopyKeyScaleGmToL1<K_SCALE_T, GM_FORMAT, L1_FORMAT, SCALE_TRANS>  (L608-683)
│   │   ├── operator() → ProcessContinuousOrTensorlist / ProcessPageAttention
│   │   ├── ProcessContinuousOrTensorlist() → BNSD/BSND/NTD/TND
│   │   └── ProcessPageAttention() → PA_BNBD / PA_NZ_K_SCALE
│   └── CopyValueScaleGmToL1<V_SCALE_T, GM_FORMAT, L1_FORMAT, SCALE_TRANS>  (L706-779)
│       ├── operator() → ProcessContinuousOrTensorlist / ProcessPageAttention
│       ├── ProcessContinuousOrTensorlist() → 含 ND2NZ/DN2NZ 双模式
│       └── ProcessPageAttention() → 含 ND2NZ/DN2NZ 双模式 + PA_NZ DataCopyParams
└── #endif (L780)
```
