# allto_all_all_gather_batch_mat_mul 算子全面分析报告

## 1. 算子概述

`allto_all_all_gather_batch_mat_mul` 是 mc2 中最复杂的融合算子之一，实现 **AllToAll + AllGather + BatchMatMul + Bias + Activation** 五阶段融合。用于 MOE 的完整 Expert 计算流水线：

```
Token → AllToAll(按 EP 分发) → AllGather(按 TP 收集) → BatchMatMul(Expert 计算) → [Bias] → [Activation]
```

该算子使用 **双 HCCL 通信域**（EP 域 AllToAll + TP 域 AllGather），将通信与计算完全重叠。

### 1.1 模板参数

```cpp
// allto_all_all_gather_batch_mat_mul_tiling_key.h (第21-35行)
ASCENDC_TPL_ARGS_DECL(
    AlltoAllAllGatherBatchMatMul,
    ASCENDC_TPL_UINT_DECL(X_SHARD, {0, 1}),           // 0: H维切分(TP), 1: C维切分(TP)
    ASCENDC_TPL_BOOL_DECL(WEIGHT_TRANSPOSE, {0, 1}),   // 权重矩阵是否转置
    ASCENDC_TPL_BOOL_DECL(IS_BIAS, {0, 1}),            // 是否有偏置
    ASCENDC_TPL_BOOL_DECL(Y2_NEED, {0, 1}),            // 是否输出 AllGather 结果(y2)
    ASCENDC_TPL_BOOL_DECL(Y3_NEED, {0, 1}),            // 是否输出 BMM 中间结果(y3)
);
```

### 1.2 核心类模板

```cpp
// allto_all_all_gather_batch_mat_mul.h (第56-57行)
template <typename DataType1, typename DataType2,
          int64_t ShardType, bool IsTransposeWeight,
          bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
class AlltoAllAllGatherBatchMatMul;
```

| 参数 | 含义 |
|------|------|
| `DataType1` | 数据类型 (BF16/FP16) |
| `DataType2` | Bias 类型 (BF16 时为 float, FP16 时为 half) |
| `ShardType` | 0=H 维切分 (TP), 1=C 维切分 (TP) |
| `IsTransposeWeight` | 权重是否转置 |
| `IsNeedBias` | 是否有偏置 |
| `IsNeedY2` | 是否输出 gather_out |
| `IsNeedY3` | 是否输出 BMM 中间结果 |

### 1.3 激活函数枚举

```cpp
// 第45-54行
enum class AlltoAllAllGatherBatchMatmulActType : uint32_t {
    NONE = 0, GELU = 1, SiLU = 2, ReLU = 3, FastGELU = 4,
    GeGLU = 5, SwiGLU = 6, ReGLU = 7
};
// 注: GeGLU/SwiGLU/ReGLU 在 kernel 代码中未实现分支，仅声明
```

---

## 2. 目录结构与文件职责

```
allto_all_all_gather_batch_mat_mul/
├── op_kernel/
│   ├── allto_all_all_gather_batch_mat_mul.cpp           (82行)   内核入口 (路由 ShardH/C 实现)
│   ├── allto_all_all_gather_batch_mat_mul.h             (1048行) ShardC 主实现类
│   ├── allto_all_all_gather_batch_mat_mul_shard_h.h     (46KB)   ShardH 实现变体
│   ├── allto_all_all_gather_batch_mat_mul_tiling_key.h  (54行)   模板参数声明
│   └── allto_all_all_gather_batch_mat_mul_tiling_struct.h (93行) Tiling 数据结构
├── op_host/
│   ├── allto_all_all_gather_batch_mat_mul_def.cpp       (3.7KB)  算子定义
│   ├── allto_all_all_gather_batch_mat_mul_infershape.cpp (22KB) Shape 推导
│   └── op_tiling/
│       ├── allto_all_all_gather_batch_mat_mul_tiling.cpp (66KB)  Tiling 策略
│       ├── allto_all_all_gather_batch_mat_mul_tiling.h   (6KB)   Tiling 类声明
│       └── allto_all_all_gather_formulaic_tiling.h       (2.4KB) Formulaic Tiling
├── op_api/                                              ACLNN C-API
├── op_graph/                                            图模式 (proto, gen_task, fallback)
└── tests/                                               测试
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `allto_all_all_gather_batch_mat_mul_def.cpp`

**输入张量**:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | x | FP16/BF16 | 输入 Token [E, C, H] |
| 1 | weight | FP16/BF16 | 权重 [E, H, M] 或 [E, M, H] |
| 2 | bias | FP16/FLOAT | 偏置 [E, M] (可选) |

**输出张量**:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | y1 | FP16/BF16 | 最终输出 [E, C, M] |
| 1 | y2 | FP16/BF16 | AllGather 输出 (可选) |
| 2 | y3 | FP16/BF16 | BMM 输出 (可选) |

**关键属性**:

| 属性名 | 类型 | 说明 |
|--------|------|------|
| group_ep | String | EP 通信域标识 |
| group_tp | String | TP 通信域标识 |
| is_transpose_weight | Bool | 权重是否转置 |
| shard_type | Int | 切分方式: 0=H切分, 1=C切分 |
| activation | String | 激活函数: "none"/"gelu"/"silu"/"relu"/"fastgelu" |
| ep_world_size | Int | EP 域大小 |
| tp_world_size | Int | TP 域大小 |

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `allto_all_all_gather_batch_mat_mul.cpp` (第62-82行)

```cpp
template<int XShard, bool WeightTransPose, bool IsBias, bool Y2Need, bool Y3Need>
__global__ __aicore__ void allto_all_all_gather_batch_mat_mul(...)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 混合 AIC+AIV
    if constexpr (XShard == 1) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(...);    // ShardC 实现
    } else {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(...);  // ShardH 实现
    }
}
```

两个实现类:
- `AlltoAllAllGatherBatchMatMul` (allto_all_all_gather_batch_mat_mul.h) — ShardType==1 (C 维切分)
- `AlltoAllAllGatherBatchMatMulShardH` (allto_all_all_gather_batch_mat_mul_shard_h.h) — ShardType==0 (H 维切分)

### 4.2 Init() 方法详解

**位置**: allto_all_all_gather_batch_mat_mul.h 第161-219行

```
Init(xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspaceGM, pipe, tilingData, ...)
├── GetTilingData(tilingData)           ← 解析 Tiling 参数
│   ├── E, C(=COverTp), H, M(=MOverTp) ← 维度 (C和M已按TP切分)
│   ├── expertOverEp = E/ep             ← 每卡 Expert 数
│   ├── epRankSize, tpRankSize          ← 通信域大小
│   ├── localE, localC                  ← 本地 Expert/C 分块信息
│   ├── nonLocalE, nonLocalC            ← 非本地分块信息
│   ├── bmmLocalTiling, bmmNonLocalTileTiling, bmmNonLocalTailTiling
│   └── actType                         ← 激活函数类型
├── shardFactor = (ShardType==0) ? 1 : tpRankSize
├── InitTpipe()                         ← UB 缓冲分配
├── 设置各 GM 地址 (x, weight, bias, y1, y2, y3)
├── 计算 Workspace 布局:
│   ├── alltoallNonLocalOutGM = workspace 基址
│   ├── allgatherLocalOutGM = + E*C*H
│   ├── allgatherNonLocalInGM = + expertOverEp*C*H*tp
│   ├── allgatherNonLocalOutGM = + expertOverEp*C*H*ep
│   ├── transposeOutGM = + E*C*H*tp
│   └── bmmOutGM = + max(local块大小, 非本地块大小)
├── 初始化双 HCCL 实例:
│   ├── hcclAlltoall.Init(contextGM0, ...)   ← EP 域 AllToAll
│   └── hcclAllgather.Init(contextGM1, ...)  ← TP 域 AllGather
└── epRankId = hcclAlltoall.GetRankId()
    tpRankId = hcclAllgather.GetRankId()
```

### 4.3 InitTpipe() — UB 缓冲分配

**位置**: 第223-250行

```
UB 布局 (按 tBuf 分配):
┌────────────────────────────────────────────────────────┐
│ transposeTmp    ← 转置临时缓冲                          │
│ xTmp            ← BMM 输出 (ubCapacityForAct)           │
│ [条件分配]                                              │
│ ├── BF16 或 SiLU:                                      │
│ │   xCastTmp (float)   ← Cast 中间缓冲                 │
│ │   actCastOutTmp (float) ← 激活输出缓冲               │
│ ├── 其他:                                               │
│ │   actOutTmp (half)    ← 激活输出缓冲                 │
│ ├── [IsNeedBias]: biasTmp                               │
│ └── [GELU/FastGELU]: sharedTmp (uint8_t, 用于高级 API)  │
└────────────────────────────────────────────────────────┘
```

### 4.4 Process() — 主流程

**位置**: 第288-294行

```
Process():
├── LocalCommunication()      ← 本地 Expert 数据 AllGather
├── NonLocalCommunication()   ← 非本地 Expert 数据 AllToAll + AllGather
├── LocalCalculation()        ← 本地数据: Transpose + BMM + Bias + Activation
└── NonLocalCalculation()     ← 非本地数据: WaitAllGather + Transpose + BMM + Bias + Act
```

**执行时序图**:

```
时间 →
AIV核: [LocalAG] [NonLocalA2A+AG交错]     [LocalCalc]  [NonLocalCalc]
AIC核:                                              [BMM]       [BMM]
                  ↕ AllGather Wait
                  ↕ AllToAll Wait
                  ↕ SyncAll
```

### 4.5 LocalCommunication() — 本地 Expert AllGather

**位置**: 第297-311行

```
LocalCommunication():
├── allgatherSendBuf = x 基址 + epRankId * expertOverEp * C * H (本卡 Expert)
├── allgatherRecvBuf = allgatherLocalOutGM
├── for e_idx = 0..localE.tileCnt:
│   ├── [AIV] hcclAllgather.AllGather<true>(send, recv, count, dtype, 0)
│   │   ← 异步 AllGather，TP 域内收集所有 TP rank 的本卡 Expert 数据
│   ├── sendBuf += offset  ← 下一轮发送偏移
│   └── recvBuf += offset  ← 下一轮接收偏移
└── Handle 存入 allgatherHandleList[e_idx]
```

注意: 本地 Expert 不需要 AllToAll，因为数据已在本地，只需 AllGather 收集 TP 域数据。

### 4.6 LocalCalculation() — 本地 Expert 计算

**位置**: 第314-389行

```
LocalCalculation():
├── for e_idx = 0..localE.tileCnt:
│   ├── [AIV] hcclAllgather.Wait(handleList[e_idx])  ← 等待 AllGather 完成
│   ├── SyncAll()
│   ├── [AIV] TransposeBeforeBMM(true, false, ...)   ← AllGather 结果转置
│   │   将 [tp, E_local, C, H] → [E_local, C, tp, H]
│   ├── SyncAll()
│   ├── [AIC] Mc2BatchMatMulCommonKernel::Process()  ← BMM 计算
│   │   输入: 转置后的 AllGather 结果 × weight
│   │   输出: bmmOutGM [E_local, C, tp, M]
│   ├── SyncAll()
│   ├── [AIV] TransposeAfterBMM(true, false, y1, y3, bias)
│   │   ← BMM 输出转置回 [tp, E, C, M] + [Bias] + [Activation]
│   └── 更新各张量偏移
```

### 4.7 NonLocalCommunication() — 非本地 Expert 通信

**位置**: 第392-527行

这是最复杂的通信编排，将 AllToAll 和 AllGather 交错执行：

```
NonLocalCommunication():
├── 计算 AllToAll 发送/接收参数 (counts, displs)
├── 计算 AllGather 发送/接收参数和偏移
├── epRankSize==2 特殊处理:
│   └── AllToAll 只需交换一半数据，AllGather 也相应调整
│
├── for e_idx = 0..nonLocalE.tileCnt:              ← 外层: E 维分块
│   ├── for c_idx = 0..nonLocalC.tileCnt:           ← 内层: C 维分块
│   │   ├── [AIV] AllToAllV<true>(sendBuf, ..., recvBuf, ...)
│   │   │   ← 异步 AllToAll，EP 域内分发 Token 到各 EP rank
│   │   │   ← 每个 E 分块内逐子块发送 nonLocalE.tileLen 次
│   │   ├── [AIV] InterHcclGroupSync (epRankSize!=2 时)
│   │   │   ← 等待 AllToAll 完成后再启动 AllGather
│   │   └── [AIV] hcclAllgather.AllGather<true/false>(...)
│   │       ← TP 域内 AllGather 收集
│   │
│   └── [C 尾块处理] (if nonLocalC.tailCnt==1)
│       └── 同上流程，使用 tailLen 替代 tileLen
│
└── [epRankSize==2 特殊路径]
    ├── Wait AllToAll handles
    ├── TransposeAllGatherInput() ← 将 AllToAll 输出转置为 AllGather 输入格式
    └── Commit AllGather
```

**通信交错策略** (epRankSize > 2):

```
AllToAll(E0_C0) → Sync → AllGather(E0_C0) →
AllToAll(E0_C1) → Sync → AllGather(E0_C1) →
...
AllToAll(E1_C0) → Sync → AllGather(E1_C0) →
...
```

每轮 AllToAll 完成后立即启动 AllGather，实现通信流水线。

### 4.8 NonLocalCalculation() — 非本地 Expert 计算

**位置**: 第530-620行

```
NonLocalCalculation():
├── for e_idx = 0..nonLocalE.tileCnt:
│   ├── for c_idx = 0..(nonLocalC.tileCnt + nonLocalC.tailCnt):
│   │   ├── [AIV] hcclAllgather.Wait(handleList[handleIdx+c_idx])
│   │   ├── SyncAll()
│   │   ├── [AIV] TransposeBeforeBMM(false, isCTail, ...)
│   │   │   ← 将 AllGather 结果 [tp, E, ep-1, C_tile, H] 转置为 BMM 输入
│   │   ├── SyncAll()
│   │   ├── [AIC] Mc2BatchMatMulCommonKernel::Process()
│   │   │   ← 使用对应的 tiling (tile 或 tail)
│   │   ├── SyncAll()
│   │   ├── [AIV] TransposeAfterBMM(false, isCTail, y1, y3, bias)
│   │   │   ← 转置 + Bias + Activation → y1
│   │   └── 更新偏移
│   ├── handleIdx += handleOffset
│   └── wGM += weightOffset (切换到下一个 E 块的权重)
└── [AIV] hcclAlltoall.Finalize() + hcclAllgather.Finalize()
```

### 4.9 TransposeBeforeBMMShardC() — BMM 前转置 (ShardC 模式)

**位置**: 第717-819行

将 AllGather 收集的数据从 `[tp, E, ep-1, C_tile, H]` 转置为 `[E, C_tile, tp, ep-1, H]`，满足 BMM 的输入格式要求。

```
核心地址映射:
  输入: [t][k][eIn][C_tile][H]   ← AllGather 输出格式
  输出: [k][eOut][tp][C_tile][H] ← BMM 输入格式

  其中:
    t = TP rank 索引
    k = E 子块内索引 (nonLocalE.tileLen 维度)
    eIn = EP rank 索引 (含本卡, factorIn=epRankSize)
    eOut = EP rank 索引 (不含本卡, factorOut=epRankSize-1)

  本卡数据跳过: eIn = (eOut >= epRankId) ? eOut+1 : eOut
```

多核切分: 按 totalM = tp × E_tile × factorOut × C_tile 的行数均匀分配给各核，每个核按行遍历处理。

### 4.10 TransposeAfterBMMShardC() — BMM 后转置 + Bias + Activation

**位置**: 第822-1035行

将 BMM 输出从 `[E, C_tile, tp, ep-1, M]` 转置回 `[tp, E, ep, C, M]` 格式，同时叠加 Bias 和 Activation。

```
数据流:
  BMM输出 [k][eIn][t][c][M] → 读入 UB →
  ├── [Bias]: Add(xCastTmp, xCastTmp, biasTmp)
  ├── [Y3]: 直接写出 BMM 中间结果
  └── [Activation]:
      ├── GELU:   Gelu(actOut, xTmp, sharedTmp, cnt)
      ├── SiLU:   Silu(actOut, xTmp, cnt) 或 Cast+Silu+Cast (BF16)
      ├── ReLU:   Relu(actOut, xTmp, cnt)
      └── FastGELU: FasterGelu(actOut, xTmp, sharedTmp, cnt)
  → Cast 回 DataType1 → 写出到 y1

精度路径:
┌─────────────────────────────────────────────────────────┐
│ BF16:                                                   │
│   BMM(BF16) → UB(BF16) → Cast FP32 → Add(Bias) →      │
│   [Activation(FP32)] → Cast BF16 → 写出                │
│                                                         │
│ FP16:                                                   │
│   BMM(FP16) → UB(FP16) → Add(Bias, FP16) →             │
│   [Activation: SiLU需Cast FP32→Silu→Cast FP16] → 写出  │
│   [Activation: GELU/ReLU/FastGELU 直接 FP16]           │
└─────────────────────────────────────────────────────────┘
```

### 4.11 TransposeAllGatherInput() — ep==2 时 AllToAll→AllGather 格式转换

**位置**: 第623-694行

当 epRankSize==2 时，AllToAll 的输出需要去除本卡数据并重新排列后才能作为 AllGather 输入:

```
输入 (AllToAll 输出): [E_tile, ep(2), C_tile, H]
  其中 ep 维包含本卡(发送给自己)和其他卡数据

处理: 去除本卡数据 (eIn = eOut >= epRankId ? eOut+1 : eOut)
  将 ep 维从 factorIn=2 转为 factorOut=1

输出 (AllGather 输入): [E_tile, ep-1(1), C_tile, H]
```

---

## 5. 数据结构

### 5.1 AlltoAllAllGatherBatchMatMulTilingData

**文件**: `allto_all_all_gather_batch_mat_mul_tiling_struct.h` (第80-91行)

```
AlltoAllAllGatherBatchMatMulTilingData:
├── hcclInitTiling: Mc2InitTiling              ← HCCL 初始化参数
├── alltoAllCcTiling: Mc2CcTiling              ← AllToAll 通信计算参数
├── allGatherCcTiling: Mc2CcTiling             ← AllGather 通信计算参数
├── commonTiling: Mc2CommonTiling              ← 公共 Tiling 参数
│   ├── epGroupSize, tpGroupSize               ← EP/TP 域大小
│   ├── expert, EOverEp, C, COverTp            ← 维度
│   ├── H, HOverTp, MOverTp                    ← 维度
│   ├── aivCoreNum                             ← AIV 核数
│   ├── isBias, y2Flag, y3Flag, isWeightTrans  ← 功能标志
│   ├── localTileE, domesticTileE              ← E 维分块 (本地/非本地)
│   ├── localTileC, domesticTileC              ← C 维分块 (本地/非本地)
│   ├── localUbTranspose, domesticUbTranspose  ← UB 转置分块
│   ├── localUbAdd, domesticUbAdd              ← UB 加激活分块
│   ├── activateType                           ← 激活函数类型
│   ├── xShardFlag                             ← 切分方式
│   └── totalUbSize                            ← UB 总大小
├── localTiling: BMM_Mc2MatmulTilingData       ← 本地 BMM Tiling
├── domesticTiling: BMM_Mc2MatmulTilingData    ← 非本地 BMM Tiling (Tile)
├── domesticTailTiling: BMM_Mc2MatmulTilingData ← 非本地 BMM Tiling (C 尾块)
└── domesticTailETiling: BMM_Mc2MatmulTilingData ← 非本地 BMM Tiling (E+C 尾块)
```

### 5.2 BMM_TileInfo — 分块信息

```
BMM_TileInfo:
├── tileCnt: uint64  ← 分块数量
├── tileLen: uint64  ← 每块长度
├── tailCnt: uint64  ← 尾块数量 (0 或 1)
└── tailLen: uint64  ← 尾块长度
```

### 5.3 BMM_Mc2MatmulTilingData — BMM Tiling

```
BMM_Mc2MatmulTilingData:
├── rankDim: uint32   ← 通信域大小
├── rankM: uint32     ← M 维对齐
├── rankID: uint32    ← Rank ID
├── enableL2Tile: uint32 ← L2 Cache 优化标志
└── bmmTilingData: Mc2BatchMatmulTilingData ← BatchMatMul V3 Tiling
```

---

## 6. Workspace 布局

```
Workspace 布局 (按偏移递增):
┌──────────────────────────────────────────────────────┐
│ [0, E*C*H)                                           │
│   AllToAll 非本地输出 (alltoallNonLocalOutGM)         │
│   大小: E * C * H * sizeof(DataType1)                │
├──────────────────────────────────────────────────────┤
│ [E*C*H, E*C*H + expertOverEp*C*H*tp)                │
│   AllGather 本地输出 (allgatherLocalOutGM)            │
│   大小: expertOverEp * C * H * tpRankSize            │
├──────────────────────────────────────────────────────┤
│ [..., ... + expertOverEp*C*H*ep)                     │
│   AllGather 非本地输入 (allgatherNonLocalInGM)        │
│   大小: expertOverEp * C * H * epRankSize            │
│   (仅 ep==2 时使用)                                   │
├──────────────────────────────────────────────────────┤
│ [..., ... + E*C*H*tp)                                │
│   AllGather 非本地输出 (allgatherNonLocalOutGM)       │
│   大小: E * C * H * tpRankSize                       │
├──────────────────────────────────────────────────────┤
│ [..., ... + max(本地块, 非本地块))                     │
│   转置临时区 (transposeOutGM)                         │
├──────────────────────────────────────────────────────┤
│ [..., ...)                                            │
│   BMM 输出区 (bmmOutGM)                              │
│   大小: max(localE.tile*localC.tile*H*tp,             │
│            (ep-1)*nonLocalE.tile*maxC*H*tp)           │
└──────────────────────────────────────────────────────┘
```

---

## 7. 通信模式分析

### 7.1 双通信域架构

```
┌─────────────────────────────────────────────────────┐
│ EP 域 (hcclAlltoall):                                │
│   上下文: contextGM0 = GetHcclContext<0>()            │
│   功能: AllToAll/AllToAllV — Token 按 Expert 分发    │
│   epRankId = hcclAlltoall.GetRankId()                │
├─────────────────────────────────────────────────────┤
│ TP 域 (hcclAllgather):                               │
│   上下文: contextGM1 = GetHcclContext<1>()            │
│   功能: AllGather — TP 维度数据收集                   │
│   tpRankId = hcclAllgather.GetRankId()               │
└─────────────────────────────────────────────────────┘
```

### 7.2 数据流路径 (MOE 前向)

```
                    EP 域                        TP 域
Rank 0:           AllToAll                     AllGather
  x[E,C,H] ──→ AllToAllV ──→ [其他卡数据] ──→ AllGather ──→ [完整TP数据]
       ↓            ↓                              ↓
  本地Expert    非本地Expert                    非本地Expert
  LocalAG       收集到本卡                     TP域聚合
       ↓            ↓                              ↓
  LocalCalc:    NonLocalCalc:
  Transpose     Transpose
  BMM           BMM
  Bias+Act      Bias+Act
       ↓                                          ↓
     y1                                         y1
```

### 7.3 通信流水线时序 (epRankSize > 2)

```
AIV 核:
  [LocalAG_E0] [LocalAG_E1] ...
  [A2A_E0_C0] → [Sync] → [AG_E0_C0]
  [A2A_E0_C1] → [Sync] → [AG_E0_C1]
  ...
  [A2A_E1_C0] → [Sync] → [AG_E1_C0]
  ...

AIC 核:
  ............[BMM_Local_E0]..[BMM_Local_E1]
  ...........................................[BMM_NonLocal_E0_C0]..[BMM_NonLocal_E0_C1]

  AllGather.Wait() 确保数据就绪后才启动 BMM
```

### 7.4 同步点汇总

| 同步点 | 类型 | 位置 | 目的 |
|--------|------|------|------|
| SyncAll | 全核 | LocalCalc 内 | AllGather Wait 后同步 |
| SyncAll | 全核 | Transpose 前后 | AIV↔AIC 数据一致性 |
| InterHcclGroupSync | HCCL 域间 | NonLocalComm 内 | AllToAll→AllGather 依赖 |
| hcclAllgather.Wait | HCCL 等待 | LocalCalc/NonLocalCalc | 等待 AllGather 完成 |
| hcclAlltoall.Wait | HCCL 等待 | ep==2 路径 | 等待 AllToAll 完成 |

---

## 8. ShardType 切分模式

### 8.1 ShardType == 0 (H 维切分)

```
输入维度: [E, C, H/tp]
AllToAll: 按 EP 域分发 (C 维度)
AllGather: 按 TP 域收集 (H 维度恢复)
BMM: x[E, C, H] × weight[E, H, M] = y[E, C, M]

特点: TransposeBeforeBMMShardH() 为空 (无需转置)
      TransposeAfterBMMShardH() 为空 (无需转置)
实现: AlltoAllAllGatherBatchMatMulShardH 类 (46KB)
```

### 8.2 ShardType == 1 (C 维切分)

```
输入维度: [E, C/tp, H]
AllToAll: 按 EP 域分发 (C 维度)
AllGather: 按 TP 域收集 (C 维度恢复)
BMM: x[E, C, H] × weight[E, H, M] = y[E, C, M]

特点: 需要在 BMM 前后进行转置
      将 AllGather 的 [tp, E, ep, C, H] 重排为 BMM 所需格式
实现: AlltoAllAllGatherBatchMatMul 类 (1048行)
```

---

## 9. 与同族变体的关系

| 算子 | 通信组合 | 方向 |
|------|---------|------|
| **allto_all_all_gather_batch_mat_mul** | AllToAll + AllGather + BMM | Dispatch + Compute |
| batch_mat_mul_reduce_scatter_allto_all | BMM + ReduceScatter + AllToAll | Compute + Combine |
| allto_allv_grouped_mat_mul | AllToAllV + GroupedMatMul | Dispatch + Compute |
| grouped_mat_mul_allto_allv | GroupedMatMul + AllToAllV | Compute + Combine |

---

## 10. 函数调用关系

```
allto_all_all_gather_batch_mat_mul (内核入口)
├── AlltoAllAllGatherBatchMatMul::Init()
│   ├── GetTilingData()
│   └── InitTpipe()
└── AlltoAllAllGatherBatchMatMul::Process()
    ├── LocalCommunication()
    │   └── hcclAllgather.AllGather<true>() × localE.tileCnt
    ├── NonLocalCommunication()
    │   ├── [for e × c]:
    │   │   ├── hcclAlltoall.AlltoAllV<true>()
    │   │   ├── InterHcclGroupSync()
    │   │   └── hcclAllgather.AllGather<true/false>()
    │   └── [ep==2 路径]:
    │       ├── hcclAlltoall.Wait()
    │       ├── TransposeAllGatherInput()
    │       └── hcclAllgather.Commit()
    ├── LocalCalculation()
    │   └── [for e]:
    │       ├── hcclAllgather.Wait()
    │       ├── TransposeBeforeBMM()
    │       │   └── TransposeBeforeBMMShardC/H()
    │       ├── Mc2BatchMatMulCommonKernel::Process()
    │       └── TransposeAfterBMM()
    │           └── TransposeAfterBMMShardC/H()
    │               ├── [Bias] Add
    │               ├── [Y3] DataCopy
    │               └── [Activation] Gelu/Silu/Relu/FasterGelu
    └── NonLocalCalculation()
        └── [for e × c]:
            ├── hcclAllgather.Wait()
            ├── TransposeBeforeBMM()
            ├── Mc2BatchMatMulCommonKernel::Process()
            └── TransposeAfterBMM()
```

---

## 11. 依赖关系

```
allto_all_all_gather_batch_mat_mul.cpp
├── basic_api/kernel_basic_intf.h          ← AscendC 基础 API
├── allto_all_all_gather_batch_mat_mul.h   ← ShardC 主实现
│   ├── adv_api/hccl/hccl.h               ← HCCL API
│   ├── adv_api/activation/silu.h          ← SiLU 激活
│   ├── adv_api/activation/gelu.h          ← GELU 激活
│   ├── lib/matmul_intf.h                  ← AscendC Matmul 接口
│   ├── 3rd/batch_mat_mul_v3/...           ← BatchMatMul V3 引擎
│   └── allto_all_all_gather_batch_mat_mul_tiling_struct.h
│       └── 3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_tiling_data.h
├── allto_all_all_gather_batch_mat_mul_shard_h.h  ← ShardH 变体实现
└── allto_all_all_gather_batch_mat_mul_tiling_key.h ← 模板参数声明
```

---

*分析完成时间: 2026-04-17*
