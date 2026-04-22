# batch_mat_mul_reduce_scatter_allto_all 算子全面分析报告

## 1. 算子概述

`batch_mat_mul_reduce_scatter_allto_all` 实现 **BatchMatMul + ReduceScatter + AlltoAll** 三重融合算子，专为 MOE 模型的 Expert 并行 (EP) + 张量并行 (TP) 场景设计。在 TP 组内先做 ReduceScatter 归约分片，再通过 AlltoAll 跨 EP 域分发 Expert 计算结果。

```
输入 x [E, M, K] × weight [E, K, N]
        ↓ BatchMatMul
   BMM 结果 [E, M, N]
        ↓ ReduceScatter (TP域)
   [E, M, N/tp] (每TP卡一份)
        ↓ AlltoAll (EP域)
   输出 y (每EP卡不同Expert的结果)
```

### 1.1 模板参数

```cpp
// batch_mat_mul_reduce_scatter_allto_all_tiling_key.h
ASCENDC_TPL_ARGS_DECL(BatchMatMulReduceScatterAlltoAll,
    BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_Y_SHARD,       // uint: 0=H维分片, 1=C维分片
    BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_WEIGHT_TRANSPOSE, // bool
    BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_IS_BIAS,        // bool
    BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_LITE);           // bool: Lite模式 (C/tp<=640)
// 最多 16 种组合 (2×2×2×2)，按条件裁剪
```

---

## 2. 目录结构与文件职责

```
batch_mat_mul_reduce_scatter_allto_all/
├── op_kernel/
│   ├── batch_mat_mul_reduce_scatter_allto_all.cpp              (4,322字节) 内核入口
│   ├── batch_mat_mul_reduce_scatter_allto_all.h                (52,590字节) Shard1 主类
│   ├── batch_mat_mul_reduce_scatter_allto_all_shard_zero.h     (46,029字节) Shard0 主类
│   ├── batch_mat_mul_reduce_scatter_allto_all_tiling_key.h     (2,720字节)  模板参数
│   └── batch_mat_mul_reduce_scatter_allto_all_tiling_struct.h  (2,912字节)  Tiling 结构
├── op_host/
│   ├── batch_mat_mul_reduce_scatter_allto_all_def.cpp          (3,083字节) 算子定义
│   ├── bmm_reduce_scatter_all_to_all_infershape.cpp            (14,888字节) Shape推导
│   └── op_tiling/
│       ├── batch_matmul_reduce_scatter_all_to_all_tiling.cpp   (47,170字节) Tiling 策略
│       ├── batch_matmul_reduce_scatter_all_to_all_tiling.h     (6,208字节)
│       └── batch_matmul_reduce_scatter_all_to_all_formulaic_tiling.h (2,573字节)
├── op_api/
│   ├── aclnn_batch_matmul_reduce_scatter_all_to_all.cpp        (18,044字节) ACLNN 实现
│   └── aclnn_batch_matmul_reduce_scatter_all_to_all.h          (5,238字节)  ACLNN 声明
├── op_graph/
│   ├── batch_mat_mul_reduce_scatter_allto_all_proto.h          (4,570字节)  图模式注册
│   ├── bmm_reduce_scatter_all_to_all_gen_task.cpp              (2,784字节)  任务生成
│   └── fallback_bmm_reduce_scatter_all_to_all.cpp              (3,433字节)  回退路径
└── tests/ut/ (省略)
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `batch_mat_mul_reduce_scatter_allto_all_def.cpp`（第18行定义 `BatchMatMulReduceScatterAlltoAll` 类，继承 `OpDef`；第20-66行构造函数；第68行 `OP_ADD` 注册）

**输入张量**（第21-38行声明）:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | x | FP16/BF16 | 输入 [E, M, K] |
| 1 | weight | FP16/BF16 | 权重 [E, K, N] 或 [E, N, K] |
| 2 | bias | FP16/FP32/BF16 (可选) | 偏置 |

**输出张量**（第40-44行声明）:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | y | FP16/BF16 | 输出 |

**关键属性**（第46-51行声明）:

| 属性名 | 类型 | 说明 |
|--------|------|------|
| group_ep | String | EP 通信域 |
| group_tp | String | TP 通信域 |
| ep_world_size | Int | EP 域大小 |
| tp_world_size | Int | TP 域大小 |
| y_shard_type | Int | 0=H维分片, 1=C维分片 |
| transpose_weight | Bool | 权重是否转置 |

**平台配置**（第53-64行）：仅 `ascend910_93`，第64行注册双 HCCL 通信组（EP + TP）。

### 3.2 Tiling 策略

**文件**: `batch_matmul_reduce_scatter_all_to_all_tiling.cpp`（47,170字节，Tiling 核心逻辑）

#### 核心参数计算

```
维度推导:
  x: [E, M, K]  → E = 专家数, M = token数, K = 隐藏维度
  weight: [E, K, N] 或 [E, N, K]
  
  输出维度:
    yShard=0 (H维分片): y = [E, C, H/tp]  ← ReduceScatter沿H切
    yShard=1 (C维分片): y = [E, C/tp, H]  ← ReduceScatter沿C切
```

#### Lite 模式检测

```
Lite 模式条件:
  C/tp <= 640 && yShard == 1
  
  作用: 对小 C 维度使用 Transpose 围绕 BMM 的策略
  (先转置 x 再做 BMM, 减少计算量)
```

#### TilingKey 编码

```
4 位编码:
  bit0 = yShard (0 或 1)
  bit1 = weightTranspose (0 或 1)
  bit2 = isBias (0 或 1)
  bit3 = isLite (0 或 1)
```

#### HCCL 配置

```
双 HCCL 通信域:
  AlltoAll: group_ep, 算法 "level0:fullmesh;level1:pairwise"
  ReduceScatter: group_tp, 算法 "level0:fullmesh;level1:pairwise"
```

### 3.3 Shape 推导

**文件**: `bmm_reduce_scatter_all_to_all_infershape.cpp` (14,888字节)

```
yShard=0:
  x: [E, M, K], weight: [E, K, C*tp]
  → BMM输出: [E, M, C*tp]
  → ReduceScatter(H维): [E, C*tp, M/tp]
  → AlltoAll(EP): [E, C, M/tp]
  → Transpose: [E, C, H/tp]

yShard=1:
  x: [E, M, K], weight: [E, K, H]
  → BMM输出: [E, M, H]
  → ReduceScatter(C维): [E, C/tp, H]
  → AlltoAll(EP): 最终输出 [E, C/tp, H]
```

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `batch_mat_mul_reduce_scatter_allto_all.cpp`（第52-55行内核入口签名，模板参数 `<YShard, WeightTranspose, IsBias, LiteMode>`；第33-42行定义 `BMMRSATAType` 辅助结构体；第44-50行定义 `INVOKE_BMMRSATA_OP_IMPL` 宏）

```cpp
// 第52-55行: 内核函数签名
template<uint32_t YShard, bool WeightTranspose, bool IsBias, bool LiteMode>
__global__ __aicore__ void batch_mat_mul_reduce_scatter_allto_all(...)
{
    // 第68-81行: FP16 路径
    // 第73-74行: YShard==1 → BatchMatMulReduceScatterAlltoAll<FP16>
    // 第78-79行: YShard==0 → BatchMatMulReduceScatterAlltoAllShard0<FP16>
    // 第83-96行: BF16 路径（对称结构）
}
```

**模板参数声明**在 `batch_mat_mul_reduce_scatter_allto_all_tiling_key.h`（第21-33行 `ASCENDC_TPL_ARGS_DECL`，第36-59行 `ASCENDC_TPL_SEL` 列举全部 16 种组合）。

### 4.2 Shard1 主类 BatchMatMulReduceScatterAlltoAll

**文件**: `batch_mat_mul_reduce_scatter_allto_all.h`（第45行类定义，1059行总计。第48-52行 `Init()` 声明，第53行 `Process()` 声明，第69-85行私有方法声明）

#### 架构概览

```
Shard1 (C维分片) 流水线:

  [AIC] BMM(Local Expert) + [AIV] AlltoAllV(Non-local Expert)
                    ↓
  [AIC] BMM(Non-local Expert) + [AIV] ReduceScatter(Local Expert)
                    ↓
  最终输出 [E, C/tp, H]
```

#### Init() — 初始化（第257-313行实现，第316-353行 `InitTilingData()`，第356-462行 `InitOffset()`）
Init(x, weight, bias, y, workspace, tiling):
├── 读取 Tiling:
│   ├── E_ep, E (专家数相关)
│   ├── C, C/tp, H, H/tp, M/tp
│   ├── epGroupSize, tpGroupSize
│   ├── UB 容量
│   └── TileInfo (local/domestic 的 E/C 轴分块)
├── HCCL 初始化:
│   ├── hcclAlltoAll: AlltoAll 通信域
│   └── hcclReduceScatter: ReduceScatter 通信域
├── rankId, rankDim 获取
├── GM 地址设置 (x, weight, bias, y)
└── 偏移量计算:
    ├── localExpert偏移 (本TP域负责的Expert)
    ├── nonLocalExpert偏移 (需要AlltoAll的Expert)
    └── 各维度的 tile 大小
```

#### Process() — 主流程（第487-496行实现）
Process():
├── NonLocalCommunicationAndCal()  ← 非本地Expert: 通信+计算
│   ├── AlltoAllV: 发送非本地Token到对应EP Rank
│   ├── 等待接收
│   ├── BMM: 对接收到的Token做矩阵乘法
│   └── 结果写回
├── LocalCommunicationAndCal()     ← 本地Expert: 通信+计算
│   ├── BMM: 对本地Token做矩阵乘法
│   ├── ReduceScatter: TP域内归约分片
│   └── 结果写回
└── [LiteMode] 转置处理
```

#### NonLocalCommunicationAndCal() — 非本地 Expert 处理（第499-591行实现）
NonLocalCommunicationAndCal():
├── [AIV] AlltoAllV 准备:
│   ├── 计算每个 EP Rank 的 sendCounts/recvCounts
│   └── hcclAlltoAll_.AllToAllV<true>(handles)
├── for expertIdx in nonLocalExperts:
│   ├── [AIC] BMM:
│   │   ├── 设置 MatmulImpl 参数
│   │   └── IterateAll()
│   ├── SyncAll()
│   └── [AIV] hcclAlltoAll_.Wait(handle[expertIdx])
└── SyncAll()
```

#### LocalCommunicationAndCal() — 本地 Expert 处理（第635-726行实现）
LocalCommunicationAndCal():
├── for expertIdx in localExperts:
│   ├── [AIC] BMM:
│   │   ├── 设置 MatmulImpl 参数
│   │   └── IterateAll()
│   ├── SyncAll()
│   └── [AIV] ReduceScatter:
│       └── hcclReduceScatter_.ReduceScatter<true>(handle)
├── 等待所有 ReduceScatter 完成
└── 结果重排到最终输出位置
```

### 4.3 Shard0 主类 BatchMatMulReduceScatterAlltoAllShard0

**文件**: `batch_mat_mul_reduce_scatter_allto_all_shard_zero.h`（第37行类定义，909行总计。`Init()` 第171-220行，`Process()` 第289-300行）

#### 架构概览

```
Shard0 (H维分片) 流水线:

  Step 1: NonLocalCommunication
    AlltoAllV → 接收非本地 Expert 的输入数据
  
  Step 2: NonLocalCalculation
    BMM(非本地Expert) → Reshape → 结果
  
  Step 3: LocalCommunication
    BMM(本地Expert) → ReshapeAfterBmm → ReduceScatter
  
  Step 4: LocalCalculation
    AddTransposeAfterRS → AlltoAllV → 最终输出
```

#### 与 Shard1 的核心差异

```
Shard1 (C维分片):
  ReduceScatter 沿 C 维切 → 每卡获得 [E, C/tp, H]
  通信和计算紧密交织

Shard0 (H维分片):
  ReduceScatter 沿 H 维切 → 每卡获得 [E, C, H/tp]
  通信和计算分阶段解耦
  需要额外的 ReshapeAfterBmm 和 AddTransposeAfterRS
```

#### ReshapeAfterBmm() — BMM 后重排（第690-757行实现）
ReshapeAfterBmm():
├── BMM 输出: [E, ep, tp, C, H] layout
├── 重排为: [E, C, tp*ep, H/tp]
└── 为 ReduceScatter 准备连续的 H/tp 分片
```

#### AddTransposeAfterRS() — RS 后处理（第760-820行实现）
AddTransposeAfterRS():
├── ReduceScatter 输出: [E, C, ep, H/tp]
├── 加 bias (如果有)
├── 转置重排为 AlltoAllV 所需格式
└── 启动 AlltoAllV 发送到目标 EP Rank
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构

Tiling 结构定义在 `batch_mat_mul_reduce_scatter_allto_all_tiling_struct.h`（`namespace optiling`）：
Mc2RSATATiling（第43-65行，21个字段）:
├── epGroupSize(第44行), tpGroupSize(第45行)       ← EP/TP 组大小
├── expertCount, E, E_ep           ← Expert 数量
├── C, C/tp, H, H/tp, M/tp        ← 维度
├── aicCoreNum, aivCoreNum         ← 核数
├── biasFlag, transposeFlag        ← 功能标志
├── yShardFlag                     ← 分片类型 (0/1)
├── BMM_TileInfo:
│   ├── localETile, localCTile     ← 本地 Expert E/C 轴分块
│   ├── domesticETile, domesticCTile ← 非本地 Expert 分块
│   └── localETail, localCTail, domesticETail, domesticCTail
└── dataSize相关字段

BatchMatMulReduceScatterAlltoAllTilingData（第67-76行，7个字段）:
├── Mc2InitTiling                  ← HCCL 初始化
├── Mc2CcTiling alltoAllCcTiling   ← AlltoAll 通信参数
├── Mc2CcTiling reduceScatterCcTiling ← ReduceScatter 通信参数
├── Mc2RSATATiling commonTilingInfo  ← 公共 Tiling
├── BMM_Mc2MatmulTilingData localBmmTiling    ← 本地 BMM Tiling
├── BMM_Mc2MatmulTilingData domesticBmmTiling ← 非本地 BMM Tiling
├── BMM_Mc2MatmulTilingData localTailBmmTiling
└── BMM_Mc2MatmulTilingData domesticTailBmmTiling
```

**辅助结构**：
- `BMM_TileInfo`（第28-33行）：tileCnt、tileLen、tailCnt、tailLen — E/C 轴分块参数
- `BMM_Mc2MatmulTilingData`（第35-41行）：rankDim、rankM、rankID、enableL2Tile、bmmTilingData — 单次 BMM 的 Matmul Tiling

### 5.2 Workspace 布局

```
Workspace 总布局:
┌──────────────────────────────────────────────────────────┐
│ HCCL AlltoAllV 通信缓冲区                                 │
│ ├── 发送缓冲: 非本地 Expert 的输入数据                      │
│ └── 接收缓冲: 来自其他 EP Rank 的 Expert 输入               │
├──────────────────────────────────────────────────────────┤
│ HCCL ReduceScatter 通信缓冲区                              │
│ ├── 发送缓冲: 本地 Expert 的 BMM 结果                       │
│ └── 接收缓冲: TP 域归约后的分片结果                         │
├──────────────────────────────────────────────────────────┤
│ BMM 中间结果缓存                                          │
│ └── Matmul 分片计算的临时存储                               │
├──────────────────────────────────────────────────────────┤
│ Transpose 缓冲区                                          │
│ └── Shard0 模式下的 ReshapeAfterBmm / AddTranspose 暂存    │
└──────────────────────────────────────────────────────────┘
```

Kernel 通过 `InitBuffers()`（Shard1: 第465-477行，Shard0: 第271-286行）分配各缓冲区的 TBuf 内存。

---

## 6. 完整数据流

```
Host 侧                              Kernel 侧
────────                              ────────
aclnnBatchMatmulReduceScatterAllToAll
  │  GetWorkspaceSize()
  │  计算 Tiling → BatchMatMulReduceScatterAlltoAllTilingData
  │  计算 Workspace 大小
  └→ 返回 workspaceSize, executor

aclnn*Execute(workspace, executor)   batch_mat_mul_reduce_scatter_allto_all.cpp
                                      第52行: 内核入口
                                      按 TilingKey 选择 YShard 分支
                                      ↓
                                     Shard1 (YShard==1, C维分片):
                                     BatchMatMulReduceScatterAlltoAll.h
                                      │
                    ┌─────────────────┤
                    │ 非本地 Expert     │ 本地 Expert
                    │ (第499-591行)    │ (第635-726行)
                    │                 │
                    │ AlltoAllV(AIV)  │ BMM(AIC)
                    │ ↓               │ ↓
                    │ BMM(AIC)        │ ReduceScatter(AIV)
                    │ ↓               │ ↓
                    │ 结果写回         │ 结果重排到输出
                    └─────────────────┤
                                      ↓
                                     输出 y [E, C/tp, H]

                                     Shard0 (YShard==0, H维分片):
                                     BatchMatMulReduceScatterAlltoAllShardZero.h
                                      │
                    ┌─────────────────┤
                    │ Step1(第303行)   │ Step3(第515行)
                    │ AlltoAllV 接收   │ BMM + ReshapeAfterBmm
                    │                 │ + ReduceScatter
                    │ Step2(第398行)   │
                    │ BMM + Reshape   │ Step4(第569行)
                    │                 │ AddTransposeAfterRS
                    │                 │ + AlltoAllV 发送
                    └─────────────────┤
                                      ↓
                                     输出 y [E, C, H/tp]
```

**双通信域协作**：EP 域（AlltoAllV）和 TP 域（ReduceScatter）使用独立的 HCCL 通信组，由 `alltoAllCcTiling` 和 `reduceScatterCcTiling` 分别配置。

---

## 7. 性能相关设计

### 7.1 计算与通信重叠

Shard1 模式下，非本地 Expert 的 AlltoAllV 通信与本地 Expert 的 BMM 计算天然解耦，可实现通信-计算重叠。`NonLocalCommunicationAndCal()` 中（第499-591行），对每个 Expert 先发起 AlltoAllV，再同步等待接收完成后做 BMM。

### 7.2 Lite 模式优化

当 `C/tp <= 640 && yShard == 1` 时启用 Lite 模式（`BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_LITE` 模板参数）。Lite 模式使用 `TransposeAroundBMM()`（Shard1 第977-1056行）策略：先对输入 x 做转置再执行 BMM，减少小 C 维度下的计算量。

### 7.3 UB 分块策略

`Mc2RSATATiling` 中的 `localTileE`/`domesticTileE`/`localTileC`/`domesticTileC`（第60-63行）控制 Expert 数量和 C 维度的 UB 分块，避免单次 BMM 的中间结果超出 UB 容量。`ubCapacityForAdd`（第56行）和 `totalUbSize`（第57行）约束 Transpose 操作的可用空间。

---

## 8. 双通信域架构

### 8.1 EP + TP 双域通信

```
┌──────────────────────────────────────────────────────────┐
│ EP 域 (Expert Parallelism)                               │
│ ┌──────────┐  AlltoAllV  ┌──────────┐                   │
│ │ EP Rank 0│────────────→│ EP Rank 1│                   │
│ │ (TP组0)  │←────────────│ (TP组0)  │                   │
│ └──────────┘             └──────────┘                   │
│                                                          │
│ 每个 EP Rank 内部:                                       │
│ ┌──────────────────────────────────────────────────┐    │
│ │ TP 域 (Tensor Parallelism)                       │    │
│ │ ┌──────┐ ReduceScatter ┌──────┐                 │    │
│ │ │TP卡 0│────────────→  │TP卡 1│                 │    │
│ │ │      │←────────────  │      │                 │    │
│ │ └──────┘               └──────┘                 │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 8.2 数据流转 (Shard1)

```
x [E, M, K] × weight [E, K, C*tp]
           ↓ BatchMatMul
     [E, M, C*tp]
           ↓ ReduceScatter (TP域, 沿C维)
     [E, M, C/tp] (每TP卡一份)
           ↓ AlltoAll (EP域)
     [E, C/tp, H] (每EP卡不同Expert)
           ↓ 输出
```

---

## 9. 与同族算子的关系

| 算子 | 通信 | 分片 | 方向 |
|------|------|------|------|
| **batch_mat_mul_reduce_scatter_allto_all** | ReduceScatter + AlltoAll | C维/H维 | MOE Combine |
| allto_all_all_gather_batch_mat_mul | AllToAll + AllGather | H维/C维 | MOE Dispatch |
| grouped_mat_mul_allto_allv | AlltoAllV | Expert维 | MOE Combine |

---

## 10. 函数调用关系

```
batch_mat_mul_reduce_scatter_allto_all (内核入口)
├── [yShard==1] BatchMatMulReduceScatterAlltoAll
│   └── Process()
│       ├── NonLocalCommunicationAndCal()
│       │   ├── hcclAlltoAll_.AllToAllV<true>() (AIV)
│       │   ├── for expert:
│       │   │   ├── Mc2BatchMatMulCommonKernel (AIC)
│       │   │   └── hcclAlltoAll_.Wait() (AIV)
│       │   └── SyncAll()
│       └── LocalCommunicationAndCal()
│           ├── for expert:
│           │   ├── Mc2BatchMatMulCommonKernel (AIC)
│           │   └── hcclReduceScatter_.ReduceScatter<true>() (AIV)
│           └── WaitAll + 结果重排
│
└── [yShard==0] BatchMatMulReduceScatterAlltoAllShard0
    └── Process()
        ├── NonLocalCommunication()  ← AlltoAllV 接收
        ├── NonLocalCalculation()    ← BMM + Reshape
        ├── LocalCommunication()     ← BMM + ReshapeAfterBmm + RS
        └── LocalCalculation()       ← AddTransposeAfterRS + AlltoAllV
```

---

*分析完成时间: 2026-04-17，修订时间: 2026-04-21（补充行号引用、Host-Kernel接口、完整数据流、性能设计章节）*
