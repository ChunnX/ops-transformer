# allto_allv_grouped_mat_mul 算子全面分析报告

## 1. 算子概述

`allto_allv_grouped_mat_mul` 实现 **AllToAllV + GroupedMatMul** 融合。在 MOE Dispatch 阶段：先通过 AllToAllV 接收来自其他 rank 的 Token 数据，然后执行 GroupedMatMul (多个 Expert 的独立矩阵乘法)。可选地叠加共享专家的额外 MatMul。

```
其他 rank Token ──AllToAllV──→ 本地 Token ──GroupedMatMul──→ Expert 计算结果
                                  (可选) ──MatMul──→ 共享专家结果
```

### 1.1 模板参数

```cpp
// allto_allv_grouped_mat_mul_tiling_key.h (第111行)
ASCENDC_TPL_ARGS_DECL(AlltoAllvGroupedMatMul,
    D_T_MM (uint),                    // 0=FP16, 1=BF16
    TILINGKEY_MM (bool),              // 是否有共享专家 MM
    TILINGKEY_GMM_WEIGHT_TRANSPOSE,   // GMM 权重是否转置
    TILINGKEY_MM_WEIGHT_TRANSPOSE);   // MM 权重是否转置
// 共 12 种有效组合
```

---

## 2. 目录结构与文件职责

```
allto_allv_grouped_mat_mul/
├── op_kernel/
│   ├── allto_allv_grouped_mat_mul.cpp           (66行)   内核入口
│   ├── allto_allv_grouped_mat_mul_coarse_grained.h (278行) 主类
│   ├── allto_allv_gmm.h                         (212行)  GMM 计算引擎
│   ├── allto_allv_gmm_utils.h                   (64行)   工具函数
│   ├── allto_allv_grouped_mat_mul_tiling.h      (65行)   Tiling 结构
│   └── allto_allv_grouped_mat_mul_tiling_key.h  (111行)  模板参数
├── op_host/
│   ├── allto_allv_grouped_mat_mul_def.cpp       (102行)  算子定义
│   ├── allto_allv_grouped_mat_mul_infershape.cpp (253行) Shape 推导
│   └── op_tiling/
│       ├── allto_allv_grouped_mat_mul_tiling.cpp (1193行) Tiling 策略
│       ├── allto_allv_grouped_mat_mul_tiling.h   (31行)
│       └── allto_allv_grouped_mat_mul_tiling_base.h (44行)
└── op_api/, op_graph/, tests/
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `allto_allv_grouped_mat_mul_def.cpp`（第19行定义 `AlltoAllvGroupedMatMul` 类，继承 `OpDef`；第22-98行构造函数；第101行 `OP_ADD` 注册）

**输入张量**（第24-59行声明）:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | gmm_x | FP16/BF16 | GMM 输入 [BSK, H1] |
| 1 | gmm_weight | FP16/BF16 | GMM 权重 [E_ep, H1, N1] |
| 2 | send_counts_tensor | INT64 (可选) | 每卡发送量 |
| 3 | recv_counts_tensor | INT64 (可选) | 每卡接收量 |
| 4 | mm_x | FP16/BF16 (可选) | 共享专家输入 [BS, H2] |
| 5 | mm_weight | FP16/BF16 (可选) | 共享专家权重 [1, H2, N2] |

**输出张量**（第61-75行声明）:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | gmm_y | FP16/BF16 | GMM 输出 [A, N1] |
| 1 | mm_y | FP16/BF16 (可选) | 共享专家输出 [BS, N2] |
| 2 | permute_out | FP16/BF16 (可选) | 通信中间输出 [A, H1] |

**关键属性**（第77-83行声明）:

| 属性名 | 类型 | 说明 |
|--------|------|------|
| group | String | HCCL 通信域 |
| ep_world_size | Int | EP 域大小 (2/4/8/16/32/64/128) |
| send_counts | ListInt | 每卡发送 Token 数 |
| recv_counts | ListInt | 每卡接收 Token 数 |
| trans_gmm_weight | Bool | GMM 权重是否转置 |
| trans_mm_weight | Bool | MM 权重是否转置 |
| permute_out_flag | Bool | 是否输出通信中间结果 |

**平台配置**（第85-97行）：`OpAICoreConfig` 启用 `DynamicCompileStaticFlag`/`DynamicFormatFlag` 等，注册 `ascend910_93` 和 `ascend950` 两个平台；第97行注册 HCCL 通信组。

### 3.2 Shape 推导

**文件**: `allto_allv_grouped_mat_mul_infershape.cpp`（第189行 `InferShapeAlltoAllvGroupedMatMul` 函数入口；第106-134行 `InferGMMOutputShape`；第136-164行 `InferMMOutputShape`；第166-187行 `InferPermuteOutputShape`；第249-251行 `IMPL_OP_INFERSHAPE` 注册）

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `allto_allv_grouped_mat_mul.cpp`（第34-36行模板参数；第37-66行内核入口函数；第25-32行 `INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL` 宏）

```cpp
template<int D_T_MM, bool MM, bool GMM_WT, bool MM_WT>
__global__ __aicore__ void allto_allv_grouped_mat_mul(...)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // D_T_MM==0 → half, D_T_MM==1 → bfloat16_t
    AlltoAllvGmmCoarseGrained<DataType, MM, GMM_WT, MM_WT> op;
    op.Init(...);
    op.Process();
}
```

### 4.2 主类 AlltoAllvGmmCoarseGrained

**文件**: `allto_allv_grouped_mat_mul_coarse_grained.h`（第30-31行类模板定义；第90行类声明结束；第92-129行 `Init` 方法；第131-142行 `Process` 方法；第144-163行 `CalcMatmul`；第165-214行 `HcclAlltoAllvPrepare`；第216-266行 `HcclAlltoAllvExec`；第268-274行 `HcclFinalize`）

#### 模板参数

```cpp
template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
class AlltoAllvGmmCoarseGrained;
```

#### Init() — 初始化 (第62-120行)

```
Init(gmmx, gmmw, sendCounts, recvCounts, mmx, mmw, gmmy, mmy, permuteOut, workspace):
├── 设置 GM 地址
├── hccl_.Init(contextGM, hcclInitTiling)
│   └── 使用 HCCL_SERVER_TYPE_AICPU
├── 读取 Tiling:
│   ├── BSK = 总 Token 数 (Batch × SeqLen × K)
│   ├── BS, H1, H2, A, N1, N2
│   ├── epWorldSize, E_ep (每卡专家数)
│   └── sendCnt[256], recvCnt[256] (每卡发送/接收量)
├── rankDim = epWorldSize
├── rankId = hccl_.GetRankId()
├── expertNumInOneRank_ = E_ep
└── expertNumAll_ = E_ep * rankDim
```

#### Process() — 主流程 (第122-135行)

```
Process():
├── HcclAlltoAllvPrepare()     ← [AIV] 启动 AllToAllV
├── [IsNeedMM] CalcMatmul()    ← [AIC] 共享专家计算
├── HcclAlltoAllvExec()        ← Wait + GMM 计算
└── HcclFinalize()             ← [AIV] 释放 HCCL
```

#### HcclAlltoAllvPrepare() — 通信准备 (第137-170行)

```
HcclAlltoAllvPrepare() [仅 AIV]:
├── 计算 per-expert 发送/接收 counts 和 offsets:
│   ├── 对每个 expert e:
│   │   ├── expertTokenNum = BSK / E_ep (均匀分配)
│   │   ├── sendCnt[rank] = expertTokenNum * H1
│   │   └── recvCnt[rank] = recvCountsTensor[rank] * H1
│   └── 计算各 rank 偏移量
├── for e = 0..E_ep:
│   └── hccl_.AllToAllV<true>(sendBuf, sendCnt, sendOffsets,
│                             recvBuf, recvCnt, recvOffsets, dataType)
│       ← 异步启动 AllToAllV，Handle 存入 handleList_
└── 返回 (不等待通信完成)
```

#### CalcMatmul() — 共享专家 MatMul (第172-183行)

```
CalcMatmul() [仅 AIC]:
├── SyncAll()                     ← 等待 AIV 完成准备工作
├── MatmulImpl<..., IsTranMmW> mm_
├── mm_.SetOrgShape(BS, H2, N2)
├── mm_.SetSingleShape(BS, H2, N2)
├── mm_.SetTensorA(mmxGM_)
├── mm_.SetTensorB(mmwGM_)
├── mm_.IterateAll(mmOutGM)       ← 执行共享专家计算
└── SyncAll()
```

#### HcclAlltoAllvExec() — Wait + GMM (第185-225行)

```
HcclAlltoAllvExec():
├── SyncAll()
├── for expertIdx = 0..E_ep:
│   ├── [AIV] hccl_.Wait(handleList_[expertIdx])
│   │   ← 等待该 Expert 的 AllToAllV 完成
│   ├── SyncAll()
│   └── [AIC] GMMProcess_.Process(expertIdx)
│       └── 对每个 Expert 执行 GMM:
│           ├── 计算该 Expert 的 A/C 地址
│           ├── mm_.SetOrgShape(m, k, n)
│           ├── mm_.SetSingleShape(m, k, n)
│           ├── mm_.SetTensorA(xGM + offset)
│           ├── mm_.SetTensorB(weightGM + offset)
│           └── mm_.IterateAll(yGM + offset)
└── SyncAll()
```

#### HcclFinalize() (第227-235行)

```
HcclFinalize() [仅 AIV]:
└── hccl_.Finalize()
```

### 4.3 GMM 计算引擎

**文件**: `allto_allv_gmm.h` (212行)

#### GMMProcess — 分组调度

```
GMMProcess<ComputeType>:
├── Init(tiling, tPipe)
└── Process(HcclServer&)
    ├── 计算每个 Expert 的 MNConfig:
    │   ├── m = tokenNumPerExpert
    │   ├── k, n 从 tiling 读取
    │   ├── numBlocksM = Ceil(m, baseM)
    │   ├── numBlocksN = Ceil(n, baseN)
    │   └── 分块索引映射
    ├── 对每个 Expert:
    │   ├── 设置 MatmulImpl 参数
    │   └── 调用 MMCompute(expertIdx, mnConfig)
    └── SyncAll
```

#### GMMCompute — 单组 MatMul

```
GMMCompute<mmType, sync>:
├── Init(gm_x, gm_w, gm_y, gm_bias)
└── MMCompute(groupIdx, mnConfig)
    ├── 设置 org/single shape
    ├── SetTensorA/B 地址 (按 groupIdx 偏移)
    └── IterateAll(gm_y + offset)
```

#### MnBlockIdxCompute — 多核负载均衡

```
MnBlockIdxCompute():
  numBlocksM = Ceil(m, baseM)
  numBlocksN = Ceil(n, baseN)
  
  核分配:
    blockIdx = GetBlockIdx()
    mIdx = blockIdx / numBlocksN
    nIdx = blockIdx % numBlocksN
  → 每个 AIC 核负责 M-N 空间的一个分片
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构

**文件**: `allto_allv_grouped_mat_mul_tiling.h`（第24-47行 `AlltoAllvGmmCommonTilingInfo` 结构；第49-52行 `AlltoAllvGmmAicpuTiling` 结构；第54-64行 `AlltoAllvGmmTilingData` 类）

**Tiling Key** 定义在 `allto_allv_grouped_mat_mul_tiling_key.h`（第24-34行 `ASCENDC_TPL_ARGS_DECL` 声明4个模板参数：`D_T_MM`、`TILINGKEY_MM`、`TILINGKEY_GMM_WEIGHT_TRANSPOSE`、`TILINGKEY_MM_WEIGHT_TRANSPOSE`；第37-110行 `ASCENDC_TPL_SEL` 枚举12种有效组合）。

```
AlltoAllvGmmTilingData (第54-64行):
├── hcclInitTiling: Mc2InitTiling              ← 第57行, HCCL 初始化
├── allGatherCcTiling: Mc2CcTiling             ← 第58行, 通信参数
├── alltoAllvCcTiling: Mc2CcTiling             ← 第59行, AllToAllV 通信参数
├── commonTilingInfo: AlltoAllvGmmCommonTilingInfo ← 第60行
│   ├── BSK, BS, K, H1, H2, A, N1, N2         ← 第25-32行, 维度
│   ├── epWorldSize, stepSize, E_ep             ← 第33-35行, EP 参数
│   ├── commOut, aivCoreNum, aicCoreNum, totalUbSize ← 第36-39行
│   └── isGmmWeightTrans, isMmWeightTrans, ...  ← 第40-46行, 标志
├── gmmTilingData: TCubeTiling                 ← 第61行, GMM MatMul Tiling
├── mmTilingData: TCubeTiling                  ← 第62行, MM MatMul Tiling
└── aicpuTiling: AlltoAllvGmmAicpuTiling       ← 第63行
    ├── sendCnt[MAX_EXPERT_SIZE]                ← 第50行, 每卡发送量
    └── recvCnt[MAX_EXPERT_SIZE]                ← 第51行, 每卡接收量
```

### 5.2 Workspace 布局

```
Workspace 总布局:
┌──────────────────────────────────────────────────────────┐
│ HCCL AllToAllV 通信缓冲区                                 │
│ ├── 发送缓冲: 本地 Expert 的输入数据 (FP16/BF16)           │
│ └── 接收缓冲: 来自其他 Rank 的 Expert 输入                 │
├──────────────────────────────────────────────────────────┐
│ GMM 计算中间存储                                          │
│ ├── MatmulImpl 实例所需的 workspace                        │
│ └── 共享专家 MM 的 workspace (如果 IsNeedMM)              │
├──────────────────────────────────────────────────────────┐
│ 系统工作区 (SetSysWorkspace)                               │
└──────────────────────────────────────────────────────────┘
```

Kernel 通过 `INVOKE_ALLTOALLV_GROUPED_MATMUL_OP_IMPL` 宏（第25-32行）将 workspace 地址传递给 `AlltoAllvGmmCoarseGrained::Init()`（第92-129行）。

---

## 6. 完整数据流

```
Host 侧:                              Kernel 侧:
┌──────────────────┐                  ┌──────────────────────────────────────────┐
│ Tiling 计算       │ ──tilingGM──→   │ Kernel 入口 (第37-66行)                   │
│ Workspace 分配    │ ──workspace──→  │                                          │
│ HCCL Context     │ ──contextGM──→  │                                          │
└──────────────────┘                  │                                          │
                                      │ Init() (第92-129行)                      │
                                      │ ├── 解析 Tiling 参数                      │
                                      │ ├── HCCL 初始化                           │
                                      │ └── 设置 GM 地址                          │
                                      │                                          │
                                      │ Process() (第131-142行)                   │
                                      │ ├── [AIV] HcclAlltoAllvPrepare()         │
                                      │ │   └── AllToAllV 发送各 Expert 数据      │
                                      │ ├── [AIC] CalcMatmul() (若 IsNeedMM)     │
                                      │ │   └── 共享专家 MatMul                   │
                                      │ ├── SyncAll()                             │
                                      │ ├── HcclAlltoAllvExec()                   │
                                      │ │   ├── Wait(expertIdx)                   │
                                      │ │   └── GMMProcess(expertIdx)             │
                                      │ │       └── MatmulImpl × E_ep             │
                                      │ └── HcclFinalize()                        │
                                      └──────────────────────────────────────────┘
```

---

## 7. 性能相关设计

1. **通信-计算流水线**: `HcclAlltoAllvPrepare` 一次性异步启动所有 Expert 的 AllToAllV，然后 `HcclAlltoAllvExec` 中逐个 Wait 并立即执行 GMM，Expert N 的 GMM 计算与 Expert N+1 的 AllToAllV 通信重叠
2. **共享专家前置**: `CalcMatmul` 在所有 AllToAllV Wait 之前执行，利用 AllToAllV 异步传输期间的 AIC 空闲时间
3. **多核并行**: AIV 核负责 HCCL 通信，AIC 核负责 MatMul 计算，通过 `SyncAll` 和 flag 同步
4. **变步长 AllToAllV**: 每对 rank 的数据量不同（`sendCnt`/`recvCnt`），与 MOE Token 不均匀分配特性匹配

---

## 8. 通信模式

### 8.1 AllToAllV 变步长通信

```
与标准 AllToAll 不同，AllToAllV 每对 rank 的数据量不同:
  sendCounts[dst_rank] = 每个 Expert 的 Token 数 × H1
  recvCounts[src_rank] = 从 src_rank 接收的 Token 数 × H1

  这与 MOE 中 Token 不均匀分配到各 Expert 的特性匹配

通信流程:
┌──────────────────────────────────────────────────┐
│ Rank 0: [Expert0数据] ──→ AllToAllV ──→ [接收数据]│
│ Rank 1: [Expert1数据] ──→ AllToAllV ──→ [接收数据]│
│ Rank 2: [Expert2数据] ──→ AllToAllV ──→ [接收数据]│
│ ...                                              │
│ 每个 Expert 的数据异步发送                        │
│ Wait(handle[expertIdx]) 后执行该 Expert 的 GMM   │
└──────────────────────────────────────────────────┘
```

### 8.2 执行时序

```
AIV: [AllToAllV Prepare (所有Expert)]
     [Wait(0)] [Wait(1)] [Wait(2)] ...
     
AIC: [共享专家MM]
                 [GMM_0] [GMM_1] [GMM_2] ...
                  ↑       ↑       ↑
                  │       │       │
               Wait(0) Wait(1) Wait(2)
```

---

## 9. 与同族变体的关系

| 算子 | 方向 | 通信 | 计算 |
|------|------|------|------|
| **allto_allv_grouped_mat_mul** | Dispatch | AllToAllV → | → GMM |
| grouped_mat_mul_allto_allv | Combine | → AllToAllV | GMM → |
| allto_allv_quant_grouped_mat_mul | Dispatch | AllToAllV → | → 量化 GMM |

---

## 10. 函数调用关系

```
allto_allv_grouped_mat_mul (内核入口)
├── AlltoAllvGmmCoarseGrained::Init()
│   ├── HCCL Init
│   └── 读取 Tiling 参数
└── AlltoAllvGmmCoarseGrained::Process()
    ├── HcclAlltoAllvPrepare() [AIV]
    │   └── hccl_.AllToAllV<true>() × E_ep
    ├── [IsNeedMM] CalcMatmul() [AIC]
    │   └── MatmulImpl::IterateAll()
    ├── HcclAlltoAllvExec()
    │   └── for expert:
    │       ├── hccl_.Wait(handle)
    │       └── GMMProcess::Process()
    │           └── GMMCompute::MMCompute()
    │               └── MatmulImpl::IterateAll()
    └── HcclFinalize() [AIV]
        └── hccl_.Finalize()
```

---

*分析完成时间: 2026-04-17，修订时间: 2026-04-21（补充§3.2 Shape推导、§5 Host-Kernel接口、§6 完整数据流、§7 性能设计，增补行号引用）*
