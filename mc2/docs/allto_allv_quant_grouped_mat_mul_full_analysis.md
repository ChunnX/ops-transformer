# allto_allv_quant_grouped_mat_mul 算子全面分析报告

## 1. 算子概述

`allto_allv_quant_grouped_mat_mul` 是 `allto_allv_grouped_mat_mul` 的量化版本，实现 **AllToAllV + 量化 GroupedMatMul** 融合。引入 mc2_templates 模板化架构，支持 PerToken/PerBlock/MX 量化，并使用通信路径量化减少 AllToAllV 数据传输量。

### 1.1 核心架构: mc2_templates

```
mc2_templates 架构:
├── communication/          ← 通信操作封装 (HcclA2avOp)
├── compute/                ← 计算操作封装 (QuantGroupedMatmul)
├── scheduler/              ← 调度器 (先通信后计算 / 先计算后通信)
├── common/                 ← 共享 Tiling
└── mc2_templates.h         ← 聚合入口
```

### 1.2 模板参数

```cpp
// allto_allv_quant_grouped_mat_mul_tiling_key.h
ASCENDC_TPL_ARGS_DECL(AlltoAllvQuantGroupedMatMul,
    D_T_MM,                    // 0=FP16, 1=BF16, 2=HIF8
    TILINGKEY_MM,              // 是否有共享专家 MM
    TILINGKEY_GMM_WEIGHT_TRANSPOSE,
    TILINGKEY_MM_WEIGHT_TRANSPOSE);
// 6 种有效组合 (D_T_MM 固定为 HIF8=2)
```

---

## 2. 目录结构与文件职责

```
allto_allv_quant_grouped_mat_mul/
├── op_kernel/
│   ├── allto_allv_quant_grouped_mat_mul_apt.cpp     (内核入口)
│   ├── allto_allv_quant_grouped_mat_mul_tiling.h    (Tiling 结构)
│   ├── allto_allv_quant_grouped_mat_mul_tiling_key.h (模板参数)
│   └── mc2_templates/
│       ├── mc2_templates.h                           (聚合入口)
│       ├── common/
│       │   └── a2av_common_tiling.h                  (共享 Tiling)
│       ├── communication/
│       │   └── hccl_a2av_op.h                        (HCCL AllToAllV 封装)
│       ├── compute/
│       │   └── quant_grouped_matmul.h                (量化 GMM)
│       └── scheduler/
│           ├── a2av_gmm_scheduler.h                  (通信→计算调度)
│           └── gmm_a2av_scheduler.h                  (计算→通信调度)
├── op_host/
│   ├── allto_allv_quant_grouped_mat_mul_def.cpp      (算子定义)
│   └── allto_allv_quant_grouped_mat_mul_infershape.cpp (Shape推导)
└── op_api/, tests/
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `allto_allv_quant_grouped_mat_mul_def.cpp`（第18行定义 `AlltoAllvQuantGroupedMatMul` 类，继承 `OpDef`；第20-463行构造函数；第466行 `OP_ADD` 注册）

**输入张量**（第22-179行声明）:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | gmm_x | HIFLOAT8 | GMM 输入 [BSK, H1] |
| 1 | gmm_weight | HIFLOAT8 | GMM 权重 [E, H1, N1] |
| 2 | send_counts_tensor | INT64 | 每卡发送计数 |
| 3 | recv_counts_tensor | INT64 | 每卡接收计数 |
| 4 | mm_x | HIFLOAT8 (可选) | 共享专家输入 |
| 5 | mm_weight | HIFLOAT8 (可选) | 共享专家权重 |
| 6 | gmm_x_scale | FP32 (可选) | GMM 输入量化缩放 |
| 7 | gmm_weight_scale | FP32 (可选) | GMM 权重量化缩放 |
| 8 | mm_x_scale | FP32 (可选) | MM 输入量化缩放 |
| 9 | mm_weight_scale | FP32 (可选) | MM 权重量化缩放 |

**输出张量**（第181-226行声明）:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | gmm_y | BF16/FP16 | GMM 输出 |
| 1 | mm_y | BF16/FP16 (可选) | 共享专家 MM 输出 |
| 2 | permute_out | BF16/FP16 (可选) | Permute 输出 |

**关键属性**（第228-241行声明）: `group`(String), `ep_world_size`(Int), `send_counts`/`recv_counts`(IntArray), `trans_gmm_weight`/`trans_mm_weight`(Bool), `permute_out_flag`(Bool), `gmm_x_quant_mode`/`gmm_weight_quant_mode`/`mm_x_quant_mode`/`mm_weight_quant_mode`(Int), `group_size`(Int), `y_dtype`(Int), `mm_dtype`(Int)

**平台配置**（第244-462行）：针对 `ascend950` 的详细配置，包含独立的输入/输出/属性声明（第245-449行），启用 `DynamicCompileStaticFlag`/`DynamicFormatFlag` 等；第462行注册 HCCL 通信组。

### 3.2 Shape 推导

**文件**: `allto_allv_quant_grouped_mat_mul_infershape.cpp`（257行）

关键函数：
- 第53-81行 `CheckDims()`: 维度校验工具
- 第112-136行 `InferGMMOutputShape()`: GMM 输出 shape 推导
- 第138-161行 `InferMMOutputShape()`: 共享专家 MM 输出 shape 推导
- 第163-181行 `InferPermuteOutputShape()`: Permute 输出 shape 推导
- 第183-226行 `InferShapeAlltoAllvGroupedMatMul()`: 主 shape 推导入口
- 第253-255行 `IMPL_OP_INFERSHAPE` 注册

### 3.3 Tiling 策略

本算子使用 mc2_templates 架构，Tiling 数据结构在 `allto_allv_quant_grouped_mat_mul_tiling.h`（第23-29行）定义。由于采用模板化 Stage 设计，Tiling 计算逻辑由公共框架统一处理，不在此算子目录下单独维护 tiling .cpp 文件。

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `allto_allv_quant_grouped_mat_mul_apt.cpp`（第55-60行内核入口签名；第29-39行定义 `GET_NESTED_TILING_DATA_MEMBER_ADDR` 宏，第41-45行定义 `TILING_TYPE` 宏，第47-53行定义 `INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL` 宏）

```cpp
__global__ __aicore__ void allto_allv_quant_grouped_mat_mul(...)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    
    // 计算操作类型
    using ComputeOpType = QuantGroupedMatmul<
        ..., isLocal=false, isA2avGmm=true>;  // 跨 rank Expert GMM
    using LocalComputeOpType = QuantGroupedMatmul<
        ..., isLocal=true, isA2avGmm=true>;   // 本地共享专家 MM
    
    // 使用 "通信先行" 调度器
    A2avGmmScheduler<HcclA2avOp, ComputeOpType, LocalComputeOpType, ...>
        a2avGmmScheduler;
    
    a2avGmmScheduler.Init(...);
    a2avGmmScheduler.Process();
}
```

### 4.2 HcclA2avOp — AllToAllV 通信封装

**文件**: `mc2_templates/communication/hccl_a2av_op.h`

```cpp
template <typename hcclDataType, bool commBeforeComputeFlag>
class HcclA2avOp;
```

#### 关键方法

```
Init(tiling, sendBuf, recvBuf):
├── hccl_.Init(contextGM, hcclInitTiling)
├── rankId = hccl_.GetRankId()
├── rankDim = hccl_.GetRankDim()
└── 设置 send/recv 全局缓冲区

Launch(expertIdx, expertNum):
├── [commBeforeCompute=true]:
│   └── 以 H1 为通信轴
│       sendCount = expertTokenNum * H1
│       hccl_.AllToAllV<true>(sendBuf, sendCounts, offsets,
│                             recvBuf, recvCounts, offsets, dtype)
└── [commBeforeCompute=false]:
    └── 以 N1 为通信轴 (计算后发送)
        sendCount = expertTokenNum * N1

Wait(handleIdx): hccl_.Wait(handles_[handleIdx])
WaitAll(): 等待所有 handles
End(): hccl_.Finalize()
```

### 4.3 QuantGroupedMatmul — 量化 GMM 计算

**文件**: `mc2_templates/compute/quant_grouped_matmul.h`

```cpp
template <typename TilingDataType, typename GmmTilingDataType,
          typename xType, typename wType, typename scaleType,
          typename yType, typename wFormat,
          bool aTrans, bool bTrans,
          bool isLocal, bool isA2avGmm>
class QuantGroupedMatmul;
```

#### 关键方法

```
Init(xGM, weightGM, scaleAGM, scaleBGM, yGM, sendCnt, recvCnt):
├── 存储 GM 地址
├── 计算 per-expert token 数: expertTokenNum = BSK / E_ep
└── 分配 workspace 用于 groupList 和 ptrTable

Process(expertIdx) [仅 AIC]:
├── UpdateAddr(expertIdx)       ← 更新数据/权重/输出地址偏移
├── BuildPtrTable()             ← 构建指针表 (4 slot: x, weight, scaleB, y)
│   └── 将地址和偏移打包为 dual-indirect 格式供 GetTensorAddr 使用
├── groupList[tokenCount] = expertTokenNum  ← 设置该 Expert 的 Token 数
├── Mc2GmmASWKernel::Init(...)  ← 初始化量化 GMM 引擎
└── Mc2GmmASWKernel::Process()  ← 执行量化 GMM

UpdateAddr(expertIdx):
├── xAddr += expertTokenNum * H1 (或 recvCnt * H1 for non-local)
├── yAddr += expertTokenNum * N1
└── weightAddr += expertIdx * H1 * N1
```

### 4.4 A2avGmmScheduler — 通信先行调度器

**文件**: `mc2_templates/scheduler/a2av_gmm_scheduler.h`

```cpp
template <typename CommOpType, typename ComputeOpType,
          typename LocalComputeOpType, typename TilingDataType,
          typename GmmTilingDataType, typename GmmArrayAddrType,
          bool IsNeedMM>
class A2avGmmScheduler;
```

#### Process() — 调度逻辑

```
Process():
├── if IsNeedMM:
│   ├── localComputeOp_.Init(...)    ← 初始化共享专家
│   ├── localComputeOp_.Process(0)   ← 执行共享专家计算
│   └── SyncAll()
│
├── commOp_.Launch(expertIdx, expertNum)  ← 启动所有 Expert 的 AllToAllV
│
├── for expertIdx = 0..E_ep:
│   ├── commOp_.Wait(expertIdx)     ← 等待该 Expert 通信完成
│   └── computeOp_.Process(expertIdx)  ← 立即执行该 Expert 的 GMM
│
└── commOp_.End()
```

**流水线效果**: Expert N 的 GMM 计算与 Expert N+1 的 AllToAllV 通信重叠。

### 4.5 GmmA2avScheduler — 计算先行调度器

**文件**: `mc2_templates/scheduler/gmm_a2av_scheduler.h`

```
Process():
├── for expertIdx = 0..E_ep:
│   ├── computeOp_.Process(expertIdx)  ← 先计算
│   ├── SyncAll()
│   └── commOp_.Launch(expertIdx, 1)   ← 后通信
│
├── if IsNeedMM:
│   ├── shareComputeOp_.Process(0)
│   └── SyncAll()
│
├── commOp_.WaitAll()              ← 等待所有通信完成
└── commOp_.End()
```

用于 **Combine 阶段** (如 `quant_grouped_mat_mul_allto_allv`)，先计算后发送。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构

Tiling 结构定义在 `allto_allv_quant_grouped_mat_mul_tiling.h`（第23-29行，`#pragma pack(push, 8)` 对齐）：

```
QuantAlltoAllvGroupedMatmulTilingData（第23-29行）:
├── hcclA2avTilingInfo: HcclA2avTilingInfo
│   ├── hcclInitTiling: Mc2InitTiling        ← HCCL 初始化
│   └── mc2CcTiling: Mc2CcTiling             ← 通信参数
├── taskTilingInfo: TaskTilingInfo
│   ├── BSK, BS, H1, H2, A, N1, N2          ← 维度
│   ├── epWorldSize, e (experts per rank)     ← EP 参数
│   ├── mainLoopExpertNum, tailLoopExpertNum  ← 循环调度
│   ├── totalLoopCount                       ← 总循环数
│   ├── sendCnt[256], recvCnt[256]           ← 每卡发送/接收量
│   └── isPermuteOut                         ← permute_out 标志
├── gmmQuantTilingData: GMMQuantTilingData   ← GMM 量化 Tiling
└── mmQuantTilingData: GMMQuantTilingData    ← MM 量化 Tiling
```

**Tiling Key** 定义在 `allto_allv_quant_grouped_mat_mul_tiling_key.h`（第25-35行 `ASCENDC_TPL_ARGS_DECL` 声明4个模板参数，第38-75行 `ASCENDC_TPL_SEL` 枚举6种有效组合）。

### 5.2 Workspace 布局

```
Workspace 总布局:
┌──────────────────────────────────────────────────────────┐
│ HCCL AllToAllV 通信缓冲区                                 │
│ ├── 发送缓冲: 本地 Expert 的量化输入数据 (HIFLOAT8)        │
│ └── 接收缓冲: 来自其他 Rank 的 Expert 输入                 │
├──────────────────────────────────────────────────────────┐
│ GMM 中间结果缓存                                          │
│ ├── groupList 数组: 每 Expert 的 token 数                  │
│ ├── ptrTable: 双间接寻址表 (x, weight, scaleB, y)         │
│ └── GMM 计算临时存储                                      │
├──────────────────────────────────────────────────────────┤
│ 系统工作区 (SetSysWorkspace)                               │
└──────────────────────────────────────────────────────────┘
```

Kernel 通过 `INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL` 宏（第47-53行）将 workspace 地址传递给 `A2avGmmScheduler::Init()`，由各 Stage 组件分别管理自己的缓冲区。

---

## 6. 架构设计: mc2_templates 模式

### 6.1 与传统模式的对比

```
传统模式 (allto_allv_grouped_mat_mul):
  单一大类 AlltoAllvGmmCoarseGrained 包含所有逻辑
  手动管理通信/计算流水线
  紧耦合，难以复用

mc2_templates 模式 (本算子):
  Stage 化设计:
    CommStage = HcclA2avOp (可替换通信策略)
    ComputeStage = QuantGroupedMatmul (可替换计算策略)
  Scheduler 统一调度:
    A2avGmmScheduler (comm → compute)
    GmmA2avScheduler (compute → comm)
  可插拔: 替换任一 Stage 不影响其他
```

### 6.2 可复用性

```
同一组件在不同算子中复用:
├── HcclA2avOp:
│   ├── allto_allv_quant_grouped_mat_mul (Dispatch)
│   └── quant_grouped_mat_mul_allto_allv (Combine)
├── A2avGmmScheduler:
│   └── allto_allv_quant_grouped_mat_mul (先收后算)
├── GmmA2avScheduler:
│   └── quant_grouped_mat_mul_allto_allv (先算后发)
└── QuantGroupedMatmul:
    ├── 作为 ComputeStage 使用
    └── isLocal 参数区分本地/远程 Expert
```

---

## 7. 与 allto_allv_grouped_mat_mul 的核心差异

| 特性 | allto_allv_grouped_mat_mul | allto_allv_quant_grouped_mat_mul |
|------|--------------------------|--------------------------------|
| 架构 | 单一大类 | mc2_templates Stage 化 |
| 量化 | 不支持 | PerToken/PerBlock/MX |
| 通信量化 | 不支持 | INT8 通信减少传输量 |
| 数据类型 | FP16/BF16 | HIFLOAT8/FP8 |
| 调度器 | 硬编码 | 可插拔 (Scheduler 模式) |
| 共享专家 | MatmulImpl | QuantGroupedMatmul (量化) |
| 适用平台 | 910_93/950 | 950 |

---

## 8. 函数调用关系

```
allto_allv_quant_grouped_mat_mul (内核入口)
└── A2avGmmScheduler::Init() + Process()
    ├── [IsNeedMM] LocalComputeOpType::Init() + Process(0)
    │   └── QuantGroupedMatmul<..., isLocal=true>::Process(0)
    │       └── Mc2GmmASWKernel::Init + Process
    ├── HcclA2avOp::Launch(all experts)
    │   └── hccl_.AllToAllV<true>()
    ├── for expertIdx:
    │   ├── HcclA2avOp::Wait(expertIdx)
    │   │   └── hccl_.Wait(handle)
    │   └── QuantGroupedMatmul<..., isLocal=false>::Process(expertIdx)
    │       ├── UpdateAddr(expertIdx)
    │       ├── BuildPtrTable()
    │       └── Mc2GmmASWKernel::Init + Process
    └── HcclA2avOp::End()
        └── hccl_.Finalize()
```

---

*分析完成时间: 2026-04-17，修订时间: 2026-04-21（补充§3 Host侧分析、行号引用、Host-Kernel接口章节）*
