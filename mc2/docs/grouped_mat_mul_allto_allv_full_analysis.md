# grouped_mat_mul_allto_allv 算子全面分析报告

## 1. 算子概述

`grouped_mat_mul_allto_allv` 实现 **GroupedMatMul + AlltoAllV** 融合，用于 MOE Combine 阶段。采用 "先计算后通信" 模式：先在本地对各 Expert 做 GroupedMatMul 计算，然后将计算结果通过 AlltoAllV 发送到对应的 EP Rank。可选叠加共享专家的额外 MatMul。

```
本地 Token × Expert 权重 [E_ep, K, N]
        ↓ GroupedMatMul (各Expert独立MM)
   GMM 结果 [各Expert, tokenNum, N]
        ↓ AlltoAllV (EP域)
   发送到对应 Rank 的结果
        (可选) 共享专家 MM → mm_y
```

### 1.1 模板参数

```cpp
// grouped_mat_mul_allto_allv_tiling_key.h
ASCENDC_TPL_ARGS_DECL — 第20-28行       // 声明 3 个 bool 模板参数
ASCENDC_TPL_SEL — 第31-72行             // 枚举全部 8 种组合 (2^3)
```

---

## 2. 目录结构与文件职责

```
grouped_mat_mul_allto_allv/
├── op_kernel/
│   ├── grouped_mat_mul_allto_allv.cpp          (4,694字节)  内核入口
│   ├── grouped_mat_mul_allto_allv.h            (13,831字节) 主类实现
│   ├── grouped_mat_mul_allto_allv_tiling.h     (1,795字节)  Tiling 结构
│   └── grouped_mat_mul_allto_allv_tiling_key.h (3,034字节)  模板参数
├── op_host/
│   ├── grouped_mat_mul_allto_allv_def.cpp      (4,856字节)  算子定义
│   ├── grouped_mat_mul_allto_allv_infershape.cpp (9,278字节) Shape推导
│   └── op_tiling/
│       ├── grouped_mat_mul_allto_allv_tiling.cpp   (49,504字节) Tiling 策略
│       ├── grouped_mat_mul_allto_allv_tiling.h     (1,247字节)
│       └── grouped_mat_mul_allto_allv_tiling_base.h (1,638字节)
├── op_api/
│   ├── aclnn_grouped_mat_mul_allto_allv.cpp    (7,793字节)  ACLNN 实现
│   └── aclnn_grouped_mat_mul_allto_allv.h      (4,878字节)  ACLNN 声明
├── op_graph/
│   ├── grouped_mat_mul_allto_allv_proto.h      (3,586字节)  图模式注册
│   ├── grouped_mat_mul_allto_allv_gen_task_training.cpp (3,622字节)
│   └── fallback_grouped_mat_mul_allto_allv.cpp (7,142字节) 回退路径
└── tests/ut/ (省略)
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `grouped_mat_mul_allto_allv_def.cpp` (4,856字节)
- 类定义 `GroupedMatMulAlltoAllv` — 第19行
- 构造函数 — 第22-92行

**输入张量** (第24-59行):

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | gmm_x | FP16/BF16 | GMM 输入 [BSK, H] |
| 1 | gmm_weight | FP16/BF16 | GMM 权重 [E_ep, H, N1] |
| 2 | send_counts_tensor | INT32/INT64 (可选) | 每卡发送量 |
| 3 | recv_counts_tensor | INT32/INT64 (可选) | 每卡接收量 |
| 4 | mm_x | FP16/BF16 (可选) | 共享专家输入 [BS, H2] |
| 5 | mm_weight | FP16/BF16 (可选) | 共享专家权重 |

**输出张量** (第61-70行):

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | y | FP16/BF16 | GMM 输出 [A, N1] |
| 1 | mm_y | FP16/BF16 (可选) | 共享专家输出 [BS, N2] |

**关键属性** (第72-77行):

| 属性名 | 类型 | 说明 |
|--------|------|------|
| group | String | HCCL 通信域 |
| ep_world_size | Int | EP 域大小 |
| send_counts | ListInt | 每卡发送 Token 数 |
| recv_counts | ListInt | 每卡接收 Token 数 |
| trans_gmm_weight | Bool | GMM 权重转置 |
| trans_mm_weight | Bool | MM 权重转置 |

**平台**: `ascend910_93`, `ascend950`
**注册**: `OP_ADD(GroupedMatMulAlltoAllv)` — 第95行

### 3.2 Tiling 策略

**文件**: `grouped_mat_mul_allto_allv_tiling.cpp` (49,504字节)

#### 核心约束

```
CheckCoreDimensionsAndCommunication():
├── BsK ∈ (0, 52428800]
├── H1/N1 ∈ (0, 65536)
├── E_ep ∈ (0, 32]
└── epWorldSize:
    ├── A5: {2, 4, 8, 16, 32, 64}
    └── A3: {8, 16, 32, 64, 128}
```

#### TilingKey 编码

```
UpdateTilingKey():
  key = GET_TPL_TILING_KEY(isOptionalMatmul, isGmmWeightTrans, isMmWeightTrans)
  → 8 种组合
```

#### BMM Tiling 参数

```
ComputeBaseMNK():
  baseM, baseN, baseK 计算依据:
  ├── L0A 容量 (double buffer): BEST_L1_PARTA = 256KB
  ├── L0B 容量 (double buffer): BEST_L1_PARTB = 128KB
  ├── BEST_BASEN = 256
  └── 根据数据类型确定元素大小
```

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `grouped_mat_mul_allto_allv.cpp` (4,694字节)
- 入口函数签名 — 第57-60行
- `REGISTER_TILING_DEFAULT` — 第70行
- `GET_TILING_DATA` — 第74行

```cpp
// 第57-60行: 入口函数签名
template<bool TILINGKEY_COMPUTE_MATMUL, bool TILINGKEY_GROUPED_MATMUL_TRANS,
         bool TILINGKEY_MATMUL_TRANS>
__global__ __aicore__ void grouped_mat_mul_allto_allv(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, ..., GM_ADDR tilingGM)

// 第70行: 注册 Tiling 数据结构
REGISTER_TILING_DEFAULT(GroupedMatMulAlltoAllvTilingData);

// 第74行: 解析 Tiling
GET_TILING_DATA(tilingData, tilingGM);
```

### 4.2 主类 GroupedMatmulAlltoAllv

**文件**: `grouped_mat_mul_allto_allv.h` (13,831字节)

**Tiling 结构文件**: `grouped_mat_mul_allto_allv_tiling.h` (1,795字节)
- `GmmAlltoAllvAicpuTiling` — 第24-27行
- `GmmAlltoAllvCommonTilingInfo` — 第29-45行
- `GroupedMatMulAlltoAllvTilingData` — 第47-56行

**模板参数文件**: `grouped_mat_mul_allto_allv_tiling_key.h` (3,034字节)
- `ASCENDC_TPL_ARGS_DECL` — 第20-28行
- `ASCENDC_TPL_SEL` — 第31-72行

```cpp
namespace AscendC {
template <typename GMMATAV>  // GMMATAV = GMMATAVType<xType, needMM, gmmWT, mmWT>
class GroupedMatmulAlltoAllv;
}
```

#### 常量定义

```cpp
static constexpr uint32_t MAX_EP_RANK_SIZE = 128;   // 最大 EP Rank 数
static constexpr uint32_t TOTAL_UBSIZE = 95 * 1024; // 95K half-words UB
static constexpr uint32_t MAX_AIV_NUM = 48;          // 最大 AIV 核数
static constexpr uint32_t MAX_HANDLE_ID_NUM = 64;    // 最大 Handle 数
```

#### Init() — 初始化

```
Init(gmmx, gmmw, sendCountsTensor, recvCountsTensor,
     mmx, mmw, gmmy, mmy, workspace, tiling):
├── 存储 GM 地址
├── 读取 Tiling:
│   ├── BsK, H, A, N1, N2, Bs
│   ├── E_ep, epWorldSize
│   ├── sendCnt[512], recvCnt[512]
│   └── isGmmWeightTrans, isMmWeightTrans, isOptionalMatmul
├── HCCL 初始化:
│   ├── A5: hccl_.InitV2(contextGM, hcclInitTiling) with CCU server
│   └── A3: hccl_.InitV2(contextGM, hcclInitTiling) with AICPU server
├── rankId = hccl_.GetRankId()
├── rankDim = hccl_.GetRankDim()
└── 设置 GlobalTensor buffer
```

#### Process() — 主流程

```
Process():
├── HcclAlltoAllvPrepare()           ← [AIV block 0] 预注册所有 AlltoAllV
├── GmmProcessAlltoallv()            ← [AIC+AIV] 逐 Expert 计算并提交通信
│   ├── for expertIdx = 0..E_ep:
│   │   ├── 计算 tokenCount (该Expert的Token总数)
│   │   ├── [AIC] GMMProcess_.Process(expertIdx) ← GMM 计算
│   │   ├── SyncAll<false>()
│   │   └── [AIV block 0] hccl_.Commit(handle)   ← 提交该 Expert 的 AlltoAllV
│   └── ShareMatMulCompute()         ← [AIC] 可选共享专家计算
├── [AIV block 0] WaitAll()          ← 等待所有 AlltoAllV 完成
└── HcclFinalize()                   ← 释放 HCCL
```

#### HcclAlltoAllvPrepare() — 通信准备

```
HcclAlltoAllvPrepare() [仅 AIV block 0]:
├── 预计算每个 Expert 的 per-rank 发送/接收量:
│   ├── for expertIdx = 0..E_ep:
│   │   ├── expertTokenNum = BsK / E_ep (均匀分配)
│   │   ├── sendCnt[rank] = expertTokenNum * N1 (以N1为通信轴)
│   │   ├── recvCnt[rank] = recvCounts[expertIdx * epWorldSize + rank] * N1
│   │   └── 计算偏移量
│   └── hccl_.AllToAllV<false>(sendBuf, sendCnt, sendOffsets,
│                              recvBuf, recvCnt, recvOffsets, dtype)
│       ← 注册 AlltoAllV Handle (不立即发送)
└── 总共注册 E_ep 个 Handle
```

**注意**: 使用 `AllToAllV<false>` 而非 `<true>` — 表示只注册不立即提交，后续通过 `Commit` 显式触发。

#### GmmProcessAlltoallv() — 计算与通信交织

```
GmmProcessAlltoallv():
├── for expertIdx = 0..E_ep:
│   ├── tokenCount = Σ sendCnt[rank] (该 Expert 在各 Rank 的发送量之和)
│   ├── if (tokenCount > 0):
│   │   └── [AIC] GMMProcess_.Process(expertIdx)
│   │       └── 对每个 Expert 独立执行 MM:
│   │           ├── 设置 MatmulImpl 参数
│   │           └── IterateAll(yGM + offset)
│   ├── SyncAll<false>()
│   └── [AIV block 0] hccl_.Commit(handles_[expertIdx])
│       ← 该 Expert 的 GMM 计算完毕后，提交其 AlltoAllV
│
├── SyncAll()
└── ShareMatMulCompute()  ← 共享专家 (如果有)
```

**流水线效果**: Expert N 的 AlltoAllV 通信与 Expert N+1 的 GMM 计算重叠。

#### ShareMatMulCompute() — 共享专家

```
ShareMatMulCompute() [仅 AIC, 需要NEED_MM]:
├── sharedmm_.SetOrgShape(BS, H2, N2)
├── sharedmm_.SetSingleShape(BS, H2, N2)
├── sharedmm_.SetTensorA(mmxGM_)
├── sharedmm_.SetTensorB(mmwGM_)
├── sharedmm_.IterateAll(mmOutGM)
└── SyncAll()
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构

**文件**: `grouped_mat_mul_allto_allv_tiling.h`

```
GmmAlltoAllvAicpuTiling — 第24-27行:
├── sendCnt[512]: uint16_t       ← 每 Expert-每 Rank 发送量
└── recvCnt[512]: uint16_t       ← 每 Expert-每 Rank 接收量

GmmAlltoAllvCommonTilingInfo — 第29-45行:
├── A, H, sharedMatmulH          ← 维度
├── E_ep, N1, N2, Bs, BsK       ← Expert/维度参数
├── epWorldSize                  ← EP 域大小
├── aivCoreNum, aicCoreNum       ← 核数
├── isGmmWeightTrans             ← GMM 权重转置
├── isMmWeightTrans              ← MM 权重转置
├── isOptionalMatmul             ← 共享专家标志
└── isOptionalSendRecvCountTensors ← counts 是否为 tensor

GroupedMatMulAlltoAllvTilingData — 第47-56行:
├── hcclInitTiling: Mc2InitTiling       ← HCCL 初始化
├── alltoAllvCcTiling: Mc2CcTiling      ← AlltoAllV 通信参数
├── commonTilingInfo                     ← 公共 Tiling
├── matmulTiling: TCubeTiling           ← GMM Tiling
├── sharedExpMatmulTiling: TCubeTiling  ← MM Tiling
└── aicpuTilingInfo                      ← send/recv counts
```

### 5.2 Host → Kernel 数据传递

```
Host Tiling (op_tiling/*.cpp)                    Kernel (op_kernel/*.cpp)
━━━━━━━━━━━━━━━━━━━━━━━━━                        ━━━━━━━━━━━━━━━━━━━━━━━
GroupedMatMulAlltoAllvTilingData                  REGISTER_TILING_DEFAULT
├── hcclInitTiling     ─── GM_ADDR tilingGM ──→  第70行: 注册结构
├── alltoAllvCcTiling                                   │
├── commonTilingInfo                                    ↓
├── matmulTiling       ─── GET_TILING_DATA ────→  第74行: 解析到 tilingData
├── sharedExpMatmulTiling                                │
└── aicpuTilingInfo                                      ↓
                                             Kernel Init() 按字段读取:
                                             ├─ hcclInitTiling → HCCL InitV2
                                             ├─ alltoAllvCcTiling → AlltoAllV 参数
                                             ├─ commonTilingInfo → BsK,H,N1,E_ep...
                                             ├─ matmulTiling → GMM MatMul 参数
                                             ├─ sharedExpMatmulTiling → 共享 MM 参数
                                             └─ aicpuTilingInfo → sendCnt/recvCnt
```

### 5.3 Workspace 布局

```
Workspace (GM_ADDR workspaceGM):
├── UserWorkspace (GetUserWorkspace) — 第66行获取
│   ├── GMM 输出中间 buffer [E_ep × maxTokenNum × N1]
│   │   每个 Expert 的 GMM 结果按顺序排列，
│   │   Expert i 的输出偏移 = Σ(j<i) tokenCount[j] × N1
│   ├── AlltoAllV 发送 buffer (复用 GMM 输出)
│   └── AlltoAllV 接收 buffer [A × N1]
│       A = 本 Rank 应接收的总 Token 数 × N1
└── SystemWorkspace (框架管理，HCCL 内部使用)

Kernel 入口检查 (第63-68行):
  if (workspaceGM == nullptr) return;      // Workspace 必须非空
  if (userWorkspace == nullptr) return;    // UserWorkspace 必须非空
```

---

## 6. 完整数据流

### 6.1 Host → Kernel 端到端流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Host 侧                                        │
│                                                                             │
│  1. 用户调用 aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(...)               │
│     ├── Tiling: 计算 TilingKey (8种组合)                                     │
│     ├── Tiling: 填充 GroupedMatMulAlltoAllvTilingData                       │
│     │         ├── hcclInitTiling (HCCL 初始化参数)                           │
│     │         ├── alltoAllvCcTiling (通信参数)                               │
│     │         ├── commonTilingInfo (BsK, H, N1, E_ep, ...)                  │
│     │         ├── matmulTiling (GMM BMM 参数)                               │
│     │         ├── sharedExpMatmulTiling (共享 MM 参数)                       │
│     │         └── aicpuTilingInfo (sendCnt[512], recvCnt[512])              │
│     └── 返回 WorkspaceSize                                                  │
│                                                                             │
│  2. 框架分配 Workspace                                                      │
│                                                                             │
│  3. 用户调用 aclnnGroupedMatMulAlltoAllv(workspace, ..., stream)            │
│     └── 将 TilingData + Workspace 地址传入 Kernel                           │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ GM_ADDR tilingGM
                               │ GM_ADDR workspaceGM
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Kernel 侧                                        │
│                                                                             │
│  4. 入口函数 (第57-60行)                                                     │
│     ├── REGISTER_TILING_DEFAULT (第70行) — 注册 Tiling 结构                  │
│     ├── GET_TILING_DATA (第74行) — 解析到本地 tilingData                     │
│     └── 获取 HCCL Context (第75行)                                           │
│                                                                             │
│  5. GroupedMatmulAlltoAllv::Init()                                          │
│     ├── 读取 commonTilingInfo → BsK, H, N1, E_ep, epWorldSize               │
│     ├── 读取 aicpuTilingInfo → sendCnt[], recvCnt[]                         │
│     ├── HCCL InitV2(hcclInitTiling) → rankId, rankDim                      │
│     └── 绑定 GM buffer (gmmxGM → gmm_x, gmmweightGM → gmm_weight, ...)     │
│                                                                             │
│  6. GroupedMatmulAlltoAllv::Process()                                       │
│     │                                                                       │
│     ├── HcclAlltoAllvPrepare() [AIV block 0]                                │
│     │   └── 为每个 Expert 注册 AlltoAllV Handle (AllToAllV<false>)           │
│     │       总共 E_ep 个 Handle                                              │
│     │                                                                       │
│     ├── GmmProcessAlltoallv() [AIC + AIV]                                   │
│     │   ├── for expertIdx = 0 .. E_ep-1:                                    │
│     │   │   ├── [AIC] GMMProcess_.Process(expertIdx)                        │
│     │   │   │   └── gmm_x[expertOffset..] × gmm_weight[expertIdx] → yGM     │
│     │   │   ├── SyncAll<false>()                                            │
│     │   │   └── [AIV] hccl_.Commit(handles_[expertIdx])                     │
│     │   │       ← 该 Expert 的 GMM 结果开始 AlltoAllV 发送                   │
│     │   └── ShareMatMulCompute() [AIC, 可选]                                │
│     │       └── mm_x × mm_weight → mm_yGM                                  │
│     │                                                                       │
│     ├── [AIV block 0] WaitAll() ← 等待所有 AlltoAllV 完成                    │
│     └── HcclFinalize()                                                      │
│                                                                             │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  输出 yGM → y       │
                    │  输出 mm_yGM → mm_y │
                    │  (AlltoAllV 接收     │
                    │   结果已写入 yGM)    │
                    └─────────────────────┘
```

### 6.2 单 Expert 数据流细节

```
Expert i 的数据流:

gmm_x [expertOffset, expertOffset+tokenCount)
         │
         │ AIC MatMul
         ▼
gmm_weight[expertIdx] (形状 [H, N1] 或 [N1, H] 视转置)
         │
         │ MatmulImpl::IterateAll()
         ▼
yGM [yOffset, yOffset + tokenCount * N1)    ← 即 AlltoAllV 发送 buffer
         │
         │ AIV AlltoAllV Commit(handles_[i])
         ▼
各目标 Rank 的 recvBuffer 中对应位置
         │
         │ WaitAll() 后
         ▼
y (最终输出) — 本 Rank 收到的各 Expert 结果拼接
```

---

## 7. 通信模式

### 7.1 "先计算后通信" 模式

```
与 allto_allv_grouped_mat_mul 的对比:

allto_allv_grouped_mat_mul (Dispatch):
  AlltoAllV(收数据) → GroupedMatMul(计算)
  "先收后算"

grouped_mat_mul_allto_allv (Combine):
  GroupedMatMul(计算) → AlltoAllV(发结果)
  "先算后发"
```

### 7.2 流水线时序

```
AIV block 0: [Prepare AllHandles] ... [Commit(0)] [Commit(1)] [Commit(2)] ... [WaitAll]
AIC:         ... [GMM_0] ... [GMM_1] ... [GMM_2] ... [ShareMM] ...
                              ↑         ↑         ↑
                           Commit(0) Commit(1) Commit(2)
                           (GMM_0完) (GMM_1完) (GMM_2完)

效果: GMM_N 的 AlltoAllV 与 GMM_N+1 的计算并行执行
```

### 7.3 A3 vs A5 HCCL Server

```
A3 (ascend910_93):
  HCCL_SERVER_TYPE_AICPU
  EP 域: {8, 16, 32, 64, 128}

A5 (ascend950):
  HCCL_SERVER_TYPE_CCU
  EP 域: {2, 4, 8, 16, 32, 64}
```

---

## 8. 与同族算子的关系

| 算子 | 方向 | 通信模式 | 架构 |
|------|------|---------|------|
| **allto_allv_grouped_mat_mul** | Dispatch | AlltoAllV → GMM | 单一大类 |
| **grouped_mat_mul_allto_allv** | Combine | GMM → AlltoAllV | 单一类 (本算子) |
| **allto_allv_quant_grouped_mat_mul** | Dispatch | AlltoAllV → 量化GMM | mc2_templates |
| **quant_grouped_mat_mul_allto_allv** | Combine | 量化GMM → AlltoAllV | mc2_templates |

---

## 9. 函数调用关系

```
grouped_mat_mul_allto_allv (内核入口)
├── GroupedMatmulAlltoAllv::Init()
│   ├── HCCL InitV2 (CCU/AICPU)
│   ├── GetRankId/GetRankDim
│   └── 设置 GM buffer
│
└── GroupedMatmulAlltoAllv::Process()
    ├── HcclAlltoAllvPrepare() [AIV block 0]
    │   ├── 计算 per-expert per-rank send/recv counts
    │   └── hccl_.AllToAllV<false>() × E_ep  ← 注册 Handle
    │
    ├── GmmProcessAlltoallv()
    │   ├── for expertIdx:
    │   │   ├── [AIC] GMMProcess_.Process(expertIdx)
    │   │   │   └── GMMCompute → MatmulImpl::IterateAll()
    │   │   ├── SyncAll<false>()
    │   │   └── [AIV block 0] hccl_.Commit(handles_[expertIdx])
    │   └── ShareMatMulCompute() [AIC]
    │       └── sharedmm_.IterateAll()
    │
    ├── [AIV block 0] WaitAll()
    └── HcclFinalize()
```

---

*分析完成时间: 2026-04-21*
