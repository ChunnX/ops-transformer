# allto_all_matmul 算子全面分析报告

## 1. 算子概述

`allto_all_matmul` 实现 **AllToAll + MatMul** 融合，是 `matmul_allto_all` 的反向操作：
- `matmul_allto_all`: MatMul → AllToAll (先算后发)
- `allto_all_matmul`: AllToAll → MatMul (先收后算)

支持三种量化模式：无量化 (FP16/BF16)、KC 动态量化 (FP8)、MX 量化。在 arch32 (910B) 和 arch35 (910_93/950) 上有不同实现。

### 1.1 模板参数

**arch32** (910B) — `allto_all_matmul_tiling_key.h`:
```
ASCENDC_TPL_ARGS_DECL(AlltoAllMatmul,
    HAS_BIAS (bool),            // 是否有偏置
    TRANSPOSE_X2 (bool),        // B 矩阵是否转置
    QUANT_TYPE (uint),          // NOQUANT/A16W8/A4W4/A16W4
    QUANT_BIAS_DTYPE (uint));   // FP16/BF16/FP32
```

**arch35** (910_93/950) — `allto_all_matmul_tiling_key.h`:
```
ASCENDC_TPL_ARGS_DECL(AlltoAllMatmul,
    QUANTMODE (uint),           // NON_QUANT/KC_QUANT_FP8E4M3/KC_QUANT_FP8E5M2/MX_QUANT
    X2TRANSPOSE (bool),         // B 矩阵是否转置
    DTYPEBIAS (uint));          // SAME_WITH_X/FP32
```

---

## 2. 目录结构与文件职责

```
allto_all_matmul/
├── op_kernel/
│   ├── allto_all_matmul.cpp                   (83行)   arch32 内核入口
│   ├── allto_all_matmul_apt.cpp               (178行)  arch35 内核入口 (Pipeline)
│   ├── arch32/                                          910B 实现
│   │   ├── allto_all_matmul.h                 (818行)  主类 AlltoAllMatmul
│   │   ├── allto_all_matmul_tiling.h          (60行)   Tiling 数据结构
│   │   ├── allto_all_matmul_tiling_key.h      (58行)   模板参数声明
│   │   ├── allto_all_matmul_util.h            (332行)  CommBase 通信基类
│   │   ├── matmul.hpp                         (279行)  Catlass AlltoAllMatmulKernel
│   │   ├── block_epilogue_dequant.hpp         (393行)  UB 反量化 Epilogue
│   │   └── tile_broadcast_add.hpp             (81行)   行广播加法
│   └── arch35/                                          910_93/950 实现
│       ├── allto_all_matmul_arch35.h          (170行)  FP 模式
│       ├── allto_all_kc_quant_matmul_arch35.h (208行)  KC 量化模式
│       ├── allto_all_mx_quant_matmul_arch35.h (258行)  MX 量化模式
│       ├── allto_all_mx_quant_matmul_pipeline.h (120行) MX 流水线
│       ├── allto_all_matmul_tiling_data.h     (61行)   arch35 Tiling
│       ├── allto_all_matmul_tiling_key.h      (72行)   arch35 模板参数
│       └── template_head.h                    (21行)   Pipeline 引用
├── op_host/
│   ├── allto_all_matmul_def.cpp               (283行)  算子定义
│   └── allto_all_matmul_infershape.cpp        (210行)  Shape 推导
└── op_api/, op_graph/, tests/
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `allto_all_matmul_def.cpp`（第18行定义 `AlltoAllMatmul` 类，继承 `OpDef`；第20-278行构造函数；第281行 `OP_ADD` 注册）

**输入张量**（第22-81行声明）:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | x1 | FP16/BF16/FP8/INT8/INT4 | A 矩阵 [M, K] |
| 1 | x2 | FP16/BF16/FP8 | B 矩阵 [K, N] 或 [N, K] |
| 2 | bias | FP16/BF16/FLOAT | 偏置 (可选) |
| 3 | x1_scale | FLOAT/FP8_E8M0 | A 矩阵量化 scale |
| 4 | x2_scale | FLOAT/FP8_E8M0 | B 矩阵量化 scale |
| 5 | comm_scale | FLOAT | 通信量化 scale |
| 6 | x1_offset | INT32 | A 矩阵量化偏移 |
| 7 | x2_offset | INT32 | B 矩阵量化偏移 |

**输出张量**（第82-97行声明）:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | y | FP16/BF16/FLOAT | MatMul 结果 [ceil(M/rank), N] |
| 1 | all2all_out | 同 x1 | AllToAll 输出 [ceil(M/rank), K*rank] |

**关键属性**（第98-110行声明）:

| 属性名 | 类型 | 说明 |
|--------|------|------|
| group | String | HCCL 通信域标识 |
| world_size | Int | 通信域大小 |
| transpose_x2 | Bool | B 矩阵是否转置 |
| x1_quant_mode | Int | A 矩阵量化模式 (0=无, 7=KC, 6=MX) |
| x2_quant_mode | Int | B 矩阵量化模式 (0=无, 2=KC, 6=MX) |

### 3.2 Shape 推导

**文件**: `allto_all_matmul_infershape.cpp`（第125行 `InferShapeAlltoAllMatmul` 函数入口；第175-207行 `InferDataTypeAlltoAllMatmul`；第209行 `IMPL_OP_INFERSHAPE` 注册）

```
输入:  x1 [M, K], x2 [K, N] (或 [N, K] if transpose)
输出:  y [ceil(M/rankNum), N]
       all2all_out [ceil(M/rankNum), K * rankNum]
```

---

## 4. Kernel 侧分析 — arch32 (910B)

### 4.1 内核入口

**文件**: `allto_all_matmul.cpp`（第36-38行内核入口函数签名；第25-32行 `INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL` 宏；第45-82行按量化模式分发实例化）

```
allto_all_matmul():
├── KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
├── 根据 ORIG_DTYPE_X1/X2 确定量化模式
├── 实例化 AlltoAllMatmul<Bias, TransX2, QuantType, BiasDtype>
├── op.Init(...)
└── op.Process()
```

### 4.2 主类 AlltoAllMatmul

**文件**: `arch32/allto_all_matmul.h`（第59行类定义，继承 `CommBase`；第112-152行 `Init` 方法；第177-268行 `CatlassMatmul`；第271-308行 `QuantPerToken`；第595-717行 `Dequant`；第720-807行 `AlltoAll`）

继承自 `CommBase` (allto_all_matmul_util.h)，复用 AllToAll 通信基础设施。

#### Init() — 初始化

```
Init(x1, x2, bias, y, all2all_out, workspace, tiling):
├── CommBase::SetArgs()       ← 解析 tiling，计算分块参数
├── 设置 GM 地址 (x1, x2, bias, y, all2all_out)
├── 初始化 HCCL: hccl_.Init(contextGM0, ...)
├── 量化缓冲区分配 (workspace 内):
│   ├── quantOutBuf    ← 量化后 X1 数据
│   ├── dequantScale   ← 反量化 scale (per-channel)
│   ├── perTokenScale  ← per-token scale
│   └── accumBuf       ← int32 累加缓冲
└── AIC 初始化 (padding, fixpipe 等)
```

#### Process() — 主流程

**位置**: arch32/allto_all_matmul.h

```
Process():
├── AlltoAll()        ← AIV 侧通信 + 量化
├── CatlassMatmul()   ← AIC 侧矩阵乘法
└── SyncAll()
```

#### AlltoAll() — 通信实现

```
AlltoAll() [仅 AIV]:
├── 初始化 UB 缓冲 (Ping-Pong)
├── Padding X1 (可选)
├── for cal_idx = 0..cal_count+1:
│   ├── WaitEvent(flag_idx)         ← 等待上轮 AIC 完成
│   ├── SetAndWaitAivSync()
│   ├── CrossRankSyncV1()           ← 通知数据就绪
│   ├── MoveWithSplit()             ← 将 X1 分块写入其他卡 Window
│   │   └── Ping-Pong UB 中转: GM→UB→目标卡Window
│   ├── CrossRankSyncV2()           ← 确认数据消费完成
│   ├── SetAicSync(flag_idx)        ← 通知 AIC 开始计算
│   └── [量化] QuantPerToken()      ← per-token 动态量化
└── 写出 all2all_out (可选)
```

通信模式与 `all_gather_matmul_v2` 类似的 Window IPC 直写，但方向为 AllToAll（每卡发送不同数据给不同目标）。

#### CatlassMatmul() — AIC 侧矩阵乘法

```
CatlassMatmul() [仅 AIC]:
├── CrossCoreWaitFlag (等待 AIV 完成通信)
├── 配置 Catlass 模板参数:
│   ├── ElementA = X1Type, ElementB = X2Type
│   ├── ElementC = (quantFlag) ? int32 : YType
│   ├── LayoutA = RowMajor, LayoutB 根据转置/NZ 选择
│   └── DispatchPolicy = MmadAtlasA2Preload
├── 选择分块:
│   ├── m0==128: L1Tile<128, 256, K>, L0Tile<128, 256, K>
│   └── m0==256: L1Tile<256, 128, K>, L0Tile<256, 128, K>
├── AlltoAllMatmulKernel(params):
│   ├── 本地 MatMul: DoLocalMatmul()
│   └── AllToAll 后远程数据 MatMul: Matmul()
│       └── 对每个 cal_idx:
│           ├── WaitEvent(flagIdx)
│           ├── 从 peerMem/gmA 读取 A 矩阵
│           ├── BlockMmad(peerA, gmB, gmC)
│           └── CrossCoreSync
└── 输出到 gmC (非量化) 或 gmAccum + gmC (量化)
```

#### 量化流水线

```
Per-Token 动态量化 (QuantPerToken):
  BF16/FP16 → Cast FP32 → Abs → ReduceMax → Muls(127/max) → Brcb → Div → Cast INT8
  + 写出 perTokenScale

反量化 (Dequant / BlockEpilogue):
  INT32 累加 → Cast FP32 → × perChannelScale → × perTokenScale → [+ Bias] → Cast YType → 写出

量化模式:
┌──────────────┬───────────────────────────────────────────┐
│ NOQUANT      │ 直接 FP16/BF16 计算                       │
│ A16W8        │ A=FP16, W=INT8, per-channel 反量化        │
│ A16W4        │ A=FP16, W=INT4, per-channel 反量化        │
│ A4W4         │ A=INT4, W=INT4, per-channel+per-token     │
└──────────────┴───────────────────────────────────────────┘
```

### 4.3 CommBase — 通信基类

**文件**: `arch32/allto_all_matmul_util.h` (332行)

提供共享的 AllToAll 通信工具:
- `SetArgs()`: 解析 tiling，计算分块参数 (m0, k0, n0, pValue, commNpuSplit 等)
- `CopyGmToUbufAlignB16()` / `CopyUbufToGmAlignB16()`: 对齐数据拷贝
- `CheckBuffFlag()` / `SetBuffFlag()`: IPC Flag 同步 (轮询 Window 状态区)
- `CopyTokensFromGMToGM()`: Ping-Pong GM 到 GM 拷贝 (经 UB 中转)
- `CrossRankSyncV1()`: 跨 Rank 同步 (SetBuffFlagByAdd + CheckBuffFlag)
- 对齐辅助: `Block32B`, `Block256B`, `Block512B`

---

## 5. Kernel 侧分析 — arch35 (910_93/950)

### 5.1 内核入口

**文件**: `allto_all_matmul_apt.cpp`（第129-132行内核入口函数签名；第133-178行按架构和量化模式分发：第38-56行 A3 FP 宏，第58-77行 A5 FP 宏，第79-101行 KC 量化宏，第103-127行 MX 量化宏）

```
allto_all_matmul_apt():
├── KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
├── [910_93 __NPU_ARCH__==2201]:
│   └── MC2KernelPipelineCommTransComputeTemplate<..., AlltoAllMatmulTilingDataA3>
│       Pipeline: Comm → Transpose → MatMul
├── [950 __NPU_ARCH__==3510]:
│   ├── [NON_QUANT]:
│   │   └── MC2KernelPipelineCommTransComputeTemplate<..., AlltoAllMatmulTilingData>
│   ├── [KC_QUANT]:
│   │   └── MC2KernelPipelineCommTransComputeTemplate<..., AlltoAllQuantMatmulTilingData>
│   │       Pipeline: Comm → Transpose+DynamicQuant → MatMul
│   └── [MX_QUANT]:
│       └── AlltoAllMxQuantMatmulPipeLine<...>
│           Pipeline: ScaleComm → ScaleTranspose → DataComm → Transpose → QuantMatMul
```

### 5.2 FP 模式 (AlltoAllMatmulArch35)

**文件**: `arch35/allto_all_matmul_arch35.h`（第24-26行类模板声明；第54-75行 `Init` 方法；第78-93行 `Process` 方法；第96-136行 `ProcessTile` 方法；第139-168行 `ProcessTail` 方法）

```
AlltoAllMatmulArch35:
├── Init(): 设置 GM 地址, Pipeline 上下文
├── Process():
│   ├── ProcessTile()  ← 主块处理 (comm → transpose → compute)
│   ├── ProcessTail()  ← 尾块处理
│   └── pipeLine_->End()
└── 使用 MC2KernelPipelineCommTransComputeTemplate 统一框架
```

### 5.3 KC 量化模式 (AlltoAllKcQuantMatmulArch35)

**文件**: `arch35/allto_all_kc_quant_matmul_arch35.h` (第208行)

```
Pipeline: AllToAll → Transpose + DynamicQuant → MatMul
├── 通信: 同 FP 模式
├── 转置+量化: AllToAll 输出转置为 MatMul 输入格式的同时进行动态量化
│   └── X1(BF16/FP16) → AbsMax → Scale → Cast(FP8)
└── 计算: 量化 MatMul (Mc2QuantBatchMatmulV3)
```

### 5.4 MX 量化模式 (AlltoAllMxQuantMatmulArch35)

**文件**: `arch35/allto_all_mx_quant_matmul_arch35.h` (258行)

```
MX 量化流水线 (最复杂):
├── ProcessScale():
│   ├── Scale 通信: AllToAll(X1_scale)
│   └── Scale 转置: 重排为 MatMul 所需格式
├── 主数据流水线:
│   ├── AllToAll(X1)
│   ├── 转置
│   └── MX 量化 MatMul (MXFP_GROUP_SIZE=64)
└── 特点: Scale 和数据分开通信，因为 Scale 尺寸更小可提前完成
```

### 5.5 Pipeline 框架

arch35 使用 `MC2KernelPipelineCommTransComputeTemplate` 统一框架:

```
Pipeline 阶段:
┌───────────┐     ┌───────────┐     ┌───────────┐
│ CommStage │ ──→ │ TransStage│ ──→ │ CompStage │
│ (AIV)     │     │ (AIV)     │     │ (AIC)     │
│ AllToAll  │     │ 转置/量化  │     │ MatMul    │
└───────────┘     └───────────┘     └───────────┘
     Sync               Sync              Sync
```

---

## 6. Host-Kernel 接口

### 6.1 Tiling 数据结构 (arch32)

**文件**: `arch32/allto_all_matmul_tiling.h`（第22-37行 `AlltoAllMatmulInfo` 结构；第39-51行 `CoCTiling` 结构；第54-59行 `AlltoAllMatmulTilingData` 结构）

```
AlltoAllMatmulTilingData:
├── mc2InitTiling: Mc2InitTiling       ← HCCL 初始化
├── mc2CcTiling: Mc2CcTiling           ← 通信计算参数
├── alltoAllMatmulInfo:                ← 算子参数
│   ├── M, K, N                        ← 矩阵维度
│   ├── rankSize, rankId               ← 通信域参数
│   ├── quantFlag, dequantFlag         ← 量化标志
│   ├── isTransposeX2                  ← 转置标志
│   ├── all2allOutFlag                 ← 是否输出 all2all_out
│   └── quant/dequant 大小参数
└── cocTiling: CoCTiling              ← 通信-计算切分参数
    ├── m0, k0, n0                    ← 分块维度
    ├── pValue                        ← 流水线粒度
    ├── ubMoveNum                     ← UB 搬运量
    ├── commNpuSplit, commDataSplit   ← 通信切分
    └── swizzlCount, swizzlDirect     ← 负载均衡
```

### 6.2 Tiling 数据结构 (arch35)

**文件**: `arch35/allto_all_matmul_tiling_data.h`（第24-41行 `AlltoAllMatmulTilingInfo` 结构；第43-49行 `AlltoAllMatmulTilingData` 结构；第52-58行 `AlltoAllQuantMatmulTilingData` 结构）

**模板参数声明**:
- arch32: `arch32/allto_all_matmul_tiling_key.h`（第30-37行 `ASCENDC_TPL_ARGS_DECL`；第41-57行 `ASCENDC_TPL_SEL`）
- arch35: `arch35/allto_all_matmul_tiling_key.h`（第32-37行 `ASCENDC_TPL_ARGS_DECL`；第41-70行 `ASCENDC_TPL_SEL`）

```
AlltoAllMatmulTilingData (arch35):
├── hcclInitTiling: Mc2InitTiling
├── mc2CcTiling: Mc2CcTiling
├── tilingInfo: AlltoAllMatmulTilingInfo
│   ├── rankDim, rankM, rankN, rankK
│   ├── tileM, tailM, tileCnt, tailCnt
│   ├── aivCoreNum, aicCoreNum
│   └── commLen
├── tileTiling: Mc2MatmulV3TilingData  ← 主块 BMM Tiling
└── tailTiling: Mc2MatmulV3TilingData  ← 尾块 BMM Tiling
```

### 6.3 Workspace 布局

```
Workspace 总布局:
┌──────────────────────────────────────────────────────────┐
│ HCCL AllToAll 通信缓冲区                                 │
│ ├── 发送缓冲: 本卡 A 矩阵分块数据                        │
│ └── 接收缓冲: 其他卡的 A 矩阵数据                        │
├──────────────────────────────────────────────────────────┐
│ 量化缓冲区 (arch32, 量化模式)                            │
│ ├── quantOutBuf: 量化后 X1 数据                          │
│ ├── dequantScale: per-channel 反量化 scale               │
│ ├── perTokenScale: per-token scale                       │
│ └── accumBuf: int32 累加缓冲                             │
├──────────────────────────────────────────────────────────┐
│ Catlass MatMul workspace (arch32)                        │
├──────────────────────────────────────────────────────────┐
│ Pipeline 缓冲区 (arch35)                                 │
│ └── MC2KernelPipelineCommTransComputeTemplate 内部管理    │
├──────────────────────────────────────────────────────────┐
│ 系统工作区 (SetSysWorkspace)                             │
└──────────────────────────────────────────────────────────┘
```

---

## 7. 通信模式

### 7.1 Window IPC AllToAll (arch32)

```
与 all_gather_matmul_v2 类似的 Window IPC 机制:

本卡 AIV 核将 X1 数据分块写入其他卡的 Window:
  X1[row_range] → UB(Ping/Pong) → stateAddrPerRank[dst_rank] + offset

Flag 同步:
  CrossRankSyncV1: 本卡通知 "数据已写入"
  CrossRankSyncV2: 其他卡确认 "数据已消费"

AIC 从 peerMem (本卡 Window 中其他卡写入的数据) 读取 A 矩阵进行 MatMul
```

### 7.2 HCCL API AllToAll (arch35)

```
arch35 使用 HCCL API 进行 AllToAll:
  hccl_.AllToAll(sendBuf, recvBuf, count, dataType)
  hccl_.Commit() / hccl_.Wait()

配合 Pipeline 框架，通信与计算自动重叠
```

---

## 8. 与 matmul_allto_all 的对比

| 特性 | matmul_allto_all | allto_all_matmul |
|------|-----------------|-----------------|
| 方向 | MatMul → AllToAll | AllToAll → MatMul |
| 用途 | Combine: 计算后分发 | Dispatch: 分发后计算 |
| 量化 | 支持 | 支持 (含 KC/MX) |
| arch32 实现 | HCCL API + MatmulImpl | Window IPC + Catlass |
| arch35 实现 | Pipeline 框架 | Pipeline 框架 |
| 输入 | A + B 矩阵 | A + B 矩阵 |
| 额外输出 | all2all_out | all2all_out |

---

## 9. 函数调用关系 (arch32)

```
allto_all_matmul (内核入口)
├── AlltoAllMatmul::Init()
│   ├── CommBase::SetArgs()
│   ├── HCCL Init
│   └── 量化缓冲区分配
└── AlltoAllMatmul::Process()
    ├── AlltoAll() [AIV]
    │   ├── Padding()
    │   ├── for cal_idx:
    │   │   ├── CrossRankSyncV1()
    │   │   ├── MoveWithSplit()
    │   │   │   └── MoveToOtherRankWithSkip()
    │   │   │       ├── CopyGmToUbufAlignB16()
    │   │   │       └── CopyUbufToGmAlignB16() → Window
    │   │   ├── CrossRankSyncV2()
    │   │   ├── SetAicSync()
    │   │   └── [量化] QuantPerToken()
    │   └── [all2all_out] MoveResultToDst()
    ├── CatlassMatmul() [AIC]
    │   └── AlltoAllMatmulKernel::operator()<AIC>()
    │       ├── CrossCoreWaitFlag
    │       ├── DoLocalMatmul() (本地数据)
    │       │   └── BlockMmad()
    │       └── Matmul() (远程数据)
    │           └── BlockMmad() per cal_idx
    └── [量化] Dequant() / BlockEpilogue
```

---

## 10. 依赖关系

```
allto_all_matmul.cpp (arch32)
├── arch32/allto_all_matmul.h
│   ├── arch32/allto_all_matmul_util.h    ← CommBase 通信基类
│   ├── 3rd/template_linear_algebra/...   ← Catlass 模板库
│   ├── arch32/matmul.hpp                 ← AlltoAllMatmulKernel
│   └── arch32/block_epilogue_dequant.hpp ← 反量化 Epilogue
└── arch32/allto_all_matmul_tiling_key.h

allto_all_matmul_apt.cpp (arch35)
├── arch35/allto_all_matmul_arch35.h      ← FP 模式
├── arch35/allto_all_kc_quant_matmul_arch35.h ← KC 量化
├── arch35/allto_all_mx_quant_matmul_arch35.h ← MX 量化
├── arch35/template_head.h
│   └── matmul_allto_all/mc2_templates/scheduler/pipeline_builder.h
└── arch35/allto_all_matmul_tiling_data.h
```

---

*分析完成时间: 2026-04-17，修订时间: 2026-04-21（补充行号引用、§6 Host-Kernel接口含Workspace布局）*
