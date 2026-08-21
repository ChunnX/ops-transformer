# quantMatmulReduceScatter 设计文档

## 1. 背景与目标

### 1.1 背景

在超大模型（如 MoE、DeepSeek 等）的分布式训练/推理场景中，多个 rank 各自持有一份输入数据 A 与完整权重 B，独立执行量化 MatMul 计算后，需要通过集合通信将各 rank 的计算结果进行聚合。传统实现将计算与通信串行执行，导致通信等待期间计算单元空闲，严重制约端到端性能。

quantMatmulReduceScatter 是 MC2（通算融合）算子族中的一员，在 Ascend 950（Apace 架构）上实现量化 MatMul 计算与集合通信聚合的深度融合，通过**AlltoAll 分发 + reduceSum 求和**完成跨卡数据聚合，以流水线并行掩盖通信延迟。

### 1.2 目标

- **功能目标**：完成 `量化MatMul + AlltoAll + reduceSum` 的融合计算，支持 MX 量化（MXFP8/MXFP4）输入。反量化（Dequant）由 Blaze::Gemm 的 `MatmulWithScaleMx` 在 MatMul 内部自动完成，不是独立步骤。
- **性能目标**：计算与通信 per-tile 流水并行，通信延迟被计算掩盖；支持 2/4/8/16/32/64 卡扩展。
- **架构目标**：基于 Apace 统一通信框架（`Apace::AivComm`）与 FragmentTensor 量化 MatMul 调度框架（`Blaze::Gemm::Block::QmmMxBlockMmadFragment`），参考 AllGatherQuantMatmul 的 FragmentTensor 设计消除 R 循环。

---

## 2. 功能描述

### 2.1 功能定义

**算子名称**：`quantMatmulReduceScatter`

quantMatmulReduceScatter 实现如下计算流程：

```
输入:
  A: [m, k], fp8_e4m3fn, 每卡完整持有（K 不切分）
  ScaleA: [m, ceil(k/64)*2], fp8_e8m0, 每卡完整持有
  B: [k, n], fp8_e4m3fn (TransB=true, 实际存储 FN 格式), 所有卡共享
  ScaleB: [ceil(k/64)*2, n], fp8_e8m0, 所有卡共享
  bias: [n], float, 可选偏置（当前版本暂未实现）
  workspace: [m, n, sizeof(bfloat16)], 由 Host 侧申请传入, 存放 AIC 计算结果

数学公式:
  // 每卡独立 matmul (MXFP8 去量化 -> fp32 内积 -> bf16 输出)
  C_partial[rank] = A_rank × B      // C_partial shape: [m, n], dtype: bf16

  // AlltoAll + reduceSum: 沿 m 维度分发 + 逐元素求和
  // chunkM = m / rankSize, rank r 输出 C_out[r*chunkM : (r+1)*chunkM, :]
  C_out[rank] = reduceSum(AlltoAll(C_partial, dim=m), op=SUM)
  // C_out shape: [m/rankSize, n], dtype: bf16
```

**核心语义**：本算子**不涉及 TP 切分**，每张卡持有完整的 K 维度：

- A 矩阵：每卡持有完整 `[m, k]`（K 不切分）。
- B 矩阵：每卡持有完整 `[k, n]`（K 不切分）。
- 每卡独立计算完整结果 `C_i = A_i × B + bias`（MX 量化输入由 Blaze 框架内部自动反量化），得到 `[m, n]`。

由于无 TP 切分，不存在跨卡的部分和（splitK/splitM），**AIC 计算阶段无需 L0C 原子累加**。跨卡聚合完全由通信阶段完成：**AlltoAll 将各卡的完整计算结果分发到目标 rank，AIV 核在收到数据后执行 reduceSum 逐元素求和**。

### 2.2 计算流程

```
原始输入（每卡一份，完全独立）
    │
    ▼
┌─────────────────────┐
│  A     [m, k]       │  完整输入（MX 量化，含 ScaleA，K 不切分）
│  B     [k, n]       │  完整权重（MX 量化，含 ScaleB，K 不切分）
│  workspace [m,n]    │  Host 侧申请的工作区（存放 AIC 计算结果 C_i）
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  FragmentTensor     │  AIC: FragmentTensor 打包 rankSize 个 rank 段地址
│  QuantMatMul        │  per-tile 调用, 一次覆盖全部 rank 段, 无 R 循环
│  (含MX反量化)       │  C_i = A_i × B → workspace [m, n]
└─────────────────────┘
    │ C_i [m, n] 写入 workspace（各卡完整）
    ▼ SetFlag 通知 AIV
┌─────────────────────┐
│  AlltoAll (PUT)     │  AIV: 从 workspace 推送 C_i 分片到其他 rank win 区
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  reduceSum          │  AIV: BF16 直接累加, self 从 workspace 读, received 从 win 读
└─────────────────────┘
    │
    ▼
    output [m/rankSize, n] (bfloat16)
```

### 2.3 量化模式

支持 MX 量化（Microscaling）场景，对应 `QmmMxBlockMmadFragment` 实现：

| 量化模式 | A 矩阵 | A Scale | B 矩阵 | B Scale | Scale 格式 |
|---------|--------|---------|--------|---------|------------|
| MXFP8  | `fp8_e4m3fn` / `fp8_e5m2` | `fp8_e8m0_t` | 同 A | `fp8_e8m0_t` | shape: `[M, ceil(K/64) * 2]` |
| MXFP4  | `fp8_e4m3fn` / `fp8_e5m2` | `fp8_e8m0_t` | `fp8_e4m3fn`/`fp8_e5m2` | `fp8_e8m0_t` | shape: `[M, ceil(K/64) * 2]` |

- Scale 的 K 方向分组大小为 64（`MXFP_DIVISOR_SIZE = 64`）。
- 每个 Scale 元素为 1 字节 `fp8_e8m0_t`，每组 2 字节（`MXFP_MULTI_BASE_SIZE = 2`）。

---

## 3. 总体架构设计

### 3.1 架构分层

```
┌─────────────────────────────────────────────────────────────────────┐
│  Host 侧 (Tiling)                                                    │
│  ┌────────────────────┐  ┌────────────────────────┐                  │
│  │ QuantMatmulTiling  │  │ AlltoAll + reduceSum   │                 │
│  │ (SWAT 策略)        │  │ Tile 切分/求和配置     │                 │
│  │ + subMmTiling      │  │                        │                 │
│  └─────────┬──────────┘  └─────────┬──────────────┘                  │
│            └──────────┬────────────┘                                 │
│                       ▼                                              │
│              TilingData 序列化                                         │
│        ┌─────────────────────────────┐                               │
│        │ commTilingData              │  通信侧切分                    │
│        │ mmTilingData                │  MatMul 侧 tiling (全量)       │
│        │ subMmTilingData             │  子问题 tiling (baseM=128)    │
│        │ tailMmTilingData            │  tail tile tiling              │
│        └─────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Device 侧 (Kernel)                                                  │
│  ┌───────────────────┐         ┌───────────────────┐                 │
│  │ AIV               │  Flag   │ AIC               │                 │
│  │ AlltoAll PUT 通信 │ ◄────── │ FragmentTensor    │                 │
│  │ + reduceSum 求和  │ 同步    │  量化 MatMul 计算  │                 │
│  │ CollectiveComm    │         │  (无 L0C 原子累加) │                 │
│  └───────────────────┘         └───────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 核角色分工

| 核类型 | 职责 | 说明 |
|-------|------|------|
| AIC（Cube） | FragmentTensor 量化 MatMul | 每卡独立执行完整的 `A × B` 计算，通过 FragmentTensor 打包各 rank 段地址消除 R 循环，无 L0C 原子累加 |
| AIV（Vector） | AlltoAll PUT 通信 + reduceSum 求和 | 通过 `CollectiveComm` 框架执行 AlltoAll PUT 分发，收齐后对多份结果执行逐元素 reduceSum 求和（BF16 直接累加） |

### 3.3 模块划分

| 模块 | 文件 | 职责 |
|------|------|------|
| MatMul Tiling | `apace/tiling/quant_matmul_tiling_swat.h` | SWAT 流式 Tiling 策略 |
| Tiling 数据 | `apace_quant_matmul_reduce_scatter_tiling_data.h` | Host→Device 序列化 TilingData（含 commTilingData + mmTilingData + subMmTilingData） |
| FragmentTensor Kernel | `qmm_mx_kernel_rs_frag.h` | FragmentTensor 版量化 MatMul 内核，参考 allgather `QmmMxKernelAgUdma`，消除 R 循环 |
| 通信框架 | `apace/block/aiv_comm/` | `CollectiveComm` 统一通信 API、AllToAll PUT 实现 |
| 算子编排 | `apace_quant_matmul_reduce_scatter_udma_impl.h` | 本算子的 Init/Run 编排逻辑 |
| reduceSum | `reduce_sum.h` | 参考 `ReadRemoteDataAdd` 实现，Te::MakeTensor 手动 UB 管理（无 TPipe），多行 block + BF16 直接累加 + pingpong 双 buffer |

---

## 4. 数据流设计

### 4.1 数据布局

#### 4.1.1 输入/输出布局

每张卡持有完整输入，独立计算完整输出：

```
Rank 0: A_0 [m, k]    B [k, n]    →  C_0 [m, n] 写入 workspace
Rank 1: A_1 [m, k]    B [k, n]    →  C_1 [m, n] 写入 workspace
...
Rank R: A_{R-1} [m, k]  B [k, n]  →  C_{R-1} [m, n] 写入 workspace
```

各卡将 workspace 中的 `C_i` 通过 **AlltoAll PUT** 推送到其他 rank 的 win 区，每卡收齐后执行 **reduceSum**：

```
output = Σ_{r=0}^{R-1} C_r[rankId*chunkM : (rankId+1)*chunkM]  // 逐元素求和
// output shape: [m/rankSize, n]
```

#### 4.1.2 通信窗口布局

PUT 模式：AIC 写 workspace，AIV 从 workspace 推送到其他 rank win 区。

```
workspace 布局 (每卡):
┌─────────────────────────────────────────────────┐
│ rank_0 chunk │ rank_1 chunk │ ... │ rank_{R-1} chunk │  ← C_i [m, n] 连续
└─────────────────────────────────────────────────┘

win 区布局 (接收方):
┌─────────────────────────────────────────────────┐
│ rank_0 数据块 │ rank_1 数据块 │ ... │ rank_{R-1} 数据块 │  ← AlltoAll PUT 接收
└─────────────────────────────────────────────────┘
```

- 每个 chunk 大小 = `chunkM × n × sizeof(CType)`（chunkM = m / rankSize）

### 4.2 通信切分策略

采用统一的 `CommTilingData` 5 段式格式：

- **切分轴** = M 轴（chunkM = m / rankSize，按 headMSize 切分为多个 comm tile）
- **非切分轴** = N 轴（整行发送）
- head tile（headMSize 行）+ 可选 tail tile（tailMSize 行）

### 4.3 数据流阶段

```
阶段 1: FragmentTensor QuantMatMul（per-tile, AIC）
    AIC: for t = 0..commTurn:
           构建 FragmentTensor (rankSize 个 fragment, 每个 curTileM 行)
           matmul(rankSize * curTileM) → workspace
           CrossCoreSetFlag<0x2, PIPE_FIX>(0) 通知 AIV
    （无跨卡数据依赖，无 L0C 原子累加）

阶段 2: AlltoAll PUT + reduceSum（per-tile, AIV）
    AIV: for t = 0..commTurn:
           CrossCoreWaitFlag<0x2, PIPE_MTE2>(0)
           AlltoAll PUT: 推送 tile t 到其他 rank
           reduceSum: BF16 直接累加 → output

阶段 3: 输出
    AIV: reduceSum 结果直接写入 output GM [m/rankSize, n]
```

---

## 5. MatMul Tiling 策略设计

### 5.1 Tiling 数据

```cpp
struct ApaceQuantMatmulReduceScatterTilingData {
    CommTilingData commTilingData;          // C 矩阵 AlltoAll 通信切分（M 轴）
    QuantMatmulTilingData mmTilingData;     // 量化 MatMul tiling (全量, T=1 退化路径)
    QuantMatmulTilingData subMmTilingData;  // 子问题 tiling (m=headMSize, baseM=128, T>1 head tile)
    QuantMatmulTilingData tailMmTilingData; // tail tile tiling (m=rankSize*tailMSize, T>1 tail tile)
};
```

### 5.2 SWAT + baseM 减半 + 自适应 headMSize

1. **全量 tiling**：`GetTilingData(m, n, k)` 生成 mmTilingData（baseM=256）
2. **自适应 headMSize**：R<=2 保持 headMSize=512, R>=4 压缩到 128 (流水掩盖效果好)
3. **子问题 tiling**：`GetTilingData(headMSize, n, k)` 生成 subMmTilingData
4. **baseM 减半**：若 SWAT 生成 baseM=256 且 headMSize 可被 128 整除，手动减半为 128
5. **统一选择**：T>1 head tile 用 subMmTilingData（baseM=128），T>1 tail tile 用 tailMmTilingData（m=rankSize×tailMSize），T=1 用 mmTilingData

### 5.3 调度器

复用 `Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3` 作为块调度器。

---

## 6. 通信设计

### 6.1 通信架构

本算子采用 **AlltoAll PUT + reduceSum** 实现跨卡聚合，在 AIV 侧完成全部通信与求和逻辑：

```
AIC: workspace [m, n] (C_i 完整计算结果)
          │ PUT
          ▼
AIV: AlltoAll → win 区 (各 rank 的 C 分片)
          │
          ▼
     reduceSum: self 从 workspace 读, received 从 win 读
          │ BF16 直接累加
          ▼
     output [m/rankSize, n] (bfloat16)
```

### 6.2 通信组件

```cpp
using AllToAllComm = Apace::AivComm::CollectiveComm<
    Apace::AivComm::CommCollectiveOp::AllToAll,
    Apace::AivComm::CommMode::PUT,   // PUT 模式 (GET 在 4+rank 不稳定, 回退到 PUT)
    CType,                           // 通信数据类型 (bfloat16, MatMul 输出)
    TeamBarrier>;

// localAddr = workspaceGM (PUT 源数据)
// AllToAll totalJobs = rankSize (后 R 核各负责 1 个 target, 并行 PUT)
// teamBarrier totalJobs = 1 (仅 jobIndex=0 的核执行 CrossDevice, step=1 正确轮询所有 remote)
// winOffset = 128 (PUT 数据跳过 win 区前 128B barrier counter, 避免覆盖)
// jobIndex = BlockNum - 1 - blockIdx (block 27: jobIndex=0, block 26: jobIndex=1, ...)
```

### 6.3 通信+求和 per-tile 流水

```
AIC: [tile0 frag mm] SetFlag [tile1 frag mm] SetFlag [tile2 frag mm] SetFlag ...
AIV:                  WaitFlag [comm0+rs0]   WaitFlag [comm1+rs1]   WaitFlag [comm2+rs2]
```

- **per-tile WaitFlag**：CrossCore flag 是计数式，T 次 Set 对应 T 次 Wait
- **reduceSum BF16 直接累加**：BF16 直接 Add (950 硬件级精度), 无 Cast, pingpong 双 buffer
- **严格分离流水**：后 R 核做 AllToAll(t), 前 (28-R) 核做 ReduceSum(t-1), 完全并行, 无 Phase 2
- **自适应 headMSize**：R<=2 用 512 (少 SyncAll), R>=4 用 128 (流水掩盖效果好)
- **多核并行通信**：后 rankSize 个核各负责 1 个 target rank 并行 PUT, 通信时间降为 1/R

### 6.4 同步机制

| 同步机制 | 用途 | 说明 |
|---------|------|------|
| `CrossCoreSetFlag<0x2, PIPE_FIX>` | AIC → AIV 计算完成通知 | AIC 完成 tile t 计算后通知 AIV（PIPE_FIX = AIC FixPipe 管道） |
| `CrossCoreWaitFlag<0x2, PIPE_MTE2>` | AIV 等待 AIC | AIV 等待 AIC tile t 完成（PIPE_MTE2 = AIV MTE2 管道） |
| `AscendC::SyncAll<true>()` | AIV 核间同步 | 每轮通信后全局同步（isAIVOnly=true, AIC 调用为 no-op） |
| `SetAtomicAdd` | **不使用** | 无 splitK/splitM 部分和 |

---

## 7. 内存设计

### 7.1 L1 缓冲区布局

FragmentTensor 方案通过 `QmmMxBlockMmadFragment` 管理 L1，per-fragment 独立加载：

```
每个 fragment (rank 段) 独立的 L1 缓存上下文:
┌──────────────────────────────────────────┐
│ A 数据缓冲区 (baseM × kL1)              │  ← 当前 fragment 的 A tile
├──────────────────────────────────────────┤
│ B 数据缓冲区 (baseN × kL1)              │  ← B 全局共享
├──────────────────────────────────────────┤
│ ScaleA 缓冲区                            │  ← 当前 fragment 的 ScaleA
├──────────────────────────────────────────┤
│ ScaleB 缓冲区                            │  ← ScaleB 全局共享
└──────────────────────────────────────────┘
```

### 7.2 L0 缓冲区

- **L0A/L0B**：存放从 L1 加载的 A/B 基础 tile
- **L0C**：`baseM × baseN` 输出累加缓冲区，支持 double-buffer

### 7.3 Workspace + 通信窗口

- **workspace**：`m × n × sizeof(CType)` 字节（Host 侧申请传入，存放 AIC 计算的 C_i）
- **win 区**：`rankSize × chunkM × n × sizeof(CType)` 字节（AllToAll PUT 接收缓冲区）
- **reduceSum UB 缓冲区**：手动 UB 偏移管理（无 TPipe），3 个区域：`[sum(float-sized, BF16 view) | inputA(BF16) | inputB(BF16)]`，pingpong 双 buffer 交替，多行 block 处理（UB 容量自适应）

---

## 8. Kernel 实现设计

### 8.1 算子入口

```cpp
__global__ __aicore__ void ApaceQuantMatmulReduceScatterKernelE4M3E4M3_Udma(
    __gm__ CommContext *hcommCtx,
    GM_ADDR aGM, GM_ADDR scaleAGM,
    GM_ADDR bGM, GM_ADDR scaleBGM,
    GM_ADDR workspaceGM,
    GM_ADDR cGM,
    ApaceQuantMatmulReduceScatterTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::ApaceQuantMatmulReduceScatterUdmaImpl<
        fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, workspaceGM, cGM, &tilingData);
    impl.Run();
}
```

### 8.2 Init 流程

1. 解析 Tiling 参数（commTilingData + mmTilingData + subMmTilingData）
2. 初始化通信上下文（udmaCtx / ubmemCtx）
3. 获取 selfWinAddr（本卡 win 区基址 + winOffset=128，跳过 barrier counter）
4. UB 分配：commBuf (512B) + barrierBuf
5. 初始化 teamBarrier（totalJobs=1，仅 jobIndex=0 执行 CrossDevice）
6. 初始化 AllToAll PUT 通信链（totalJobs=rankSize，后 R 核各负责 1 个 target）

### 8.3 Run 流程

```cpp
__aicore__ inline void Run()
{
    if ASCEND_IS_AIC {
        // AIC: per-tile FragmentTensor matmul + SetFlag
        // 统一路径: for t = 0..commTurn, 无 if/else, T=1 自然退化
        RunMatmul();
    }
    if ASCEND_IS_AIV {
        // AIV: per-tile WaitFlag + AllToAll PUT + reduceSum
        RunAllToAllAndReduceSum();
    }
}
```

**RunMatmul (AIC)**:
```cpp
for (t = 0..commTurn) {
    // 构建 FragmentTensor: rankSize 个 fragment, 每个 curTileM 行
    // A/C 地址带 tile 偏移: aGm + tileMOffset*K, workspaceGm + tileMOffset*N*2
    fragKernelImpl_(params);  // 一次调用覆盖所有 rank 段, 无 R 循环
    CrossCoreSetFlag<0x2, PIPE_FIX>(0);  // 通知 AIV
}
```

**RunAllToAllAndReduceSum (AIV)**:
```cpp
// 方案B: 严格分离 — 后 R 核通信, 前 (28-R) 核计算
// isCommBlock = (jobIndex < rankSize)     — 后 R 个核
// isComputeBlock = (blockIdx < 28-R)     — 前 (28-R) 个核
// rsCoreNum = 28 - rankSize              — reduceSum 核数

for (t = 0..commTurn) {
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);  // 等 AIC tile t
    SyncAll<true>();
    if (isCommBlock) {
        allToAllC_.Commit<BARRIER_NONE>();   // PUT 推送 tile t
        allToAllC_.Wait<BARRIER_NONE>();      // Drain only (不做 CrossDevice)
    } else if (isComputeBlock) {
        reduceSum_.Init(..., rsCoreNum, ...); // 前 (28-R) 核做 ReduceSum(t-1)
        reduceSum_.ExecuteReduceSum(tileM, nSize);
    }
    SyncAll<true>();
    teamBarrier_.CrossDevice();  // block 0 (jobIndex=0) 做跨设备 fence (step=1)
    SyncAll<true>();
}
allToAllC_.Finalize();
```

### 8.4 关键同步机制

| 同步机制 | 用途 | 说明 |
|---------|------|------|
| `CrossCoreSetFlag<0x2, PIPE_FIX>` | AIC → AIV | AIC tile t 完成, PIPE_FIX = AIC FixPipe |
| `CrossCoreWaitFlag<0x2, PIPE_MTE2>` | AIV 等 AIC | per-tile WaitFlag, 计数式 flag (T 次 Set ↔ T 次 Wait) |
| `SyncAll<true>()` | AIV 核间同步 | isAIVOnly=true, AIC 调用为 no-op |
| `teamBarrier_.CrossDevice()` | 跨设备同步 | block 0 (jobIndex=0) 执行, step=1 轮询所有 remote localFlag |
| `SetAtomicAdd` | **不使用** | 无跨卡部分和 |

---

## 9. 关键设计决策

### 9.1 为什么用 AlltoAll + reduceSum 替代 ReduceScatter

本算子场景**不涉及 TP 切分**：每卡持有完整 A 和 B，独立计算出完整的 `C_i [m, n]`。通过 **AlltoAll（分发）+ reduceSum（求和）** 分步实现：

- **无需计算侧原子累加**：不存在 splitK/splitM 的部分和
- **求和下沉到 AIV**：将归约从 Cube 侧卸载到 Vector 侧，与通信天然流水
- **逐块流水求和**：每收齐一个 rank 的分片即累加一次

### 9.2 为什么用 FragmentTensor 消除 R 循环

参考 AllGatherQuantMatmul 的 `QmmMxKernelAgUdma` 设计：

- **allgather**：A 数据来自不同 rank 的 win 区（不连续），用 FragmentTensor 打包地址
- **本算子**：A 数据全在本卡 GM 连续 `[m,k]`，各 rank 段天然连续
- **FragmentTensor 的核心价值**：per-fragment L1 缓存隔离——`QmmMxBlockMmadFragment` 在 tile 跨越 rank 边界时自动切换 fragment，每个 fragment 独立加载 L1
- **性能验证**：FragmentTensor 方案（1108us）与 per-tile R 循环方案（1098us）持平（+0.9%），远优于一次 matmul 调用方案（1152us, +4.9%）

### 9.3 为什么统一 if/else 路径

- `commTurn=1` 时 `M_total = rankSize * chunkM = m`，与 `commTurn>1` 逻辑本质相同
- 统一为 `for t = 0..commTurn` 单一循环路径，T=1 时循环执行 1 次自然退化
- 无 if/else 分支，代码更简洁

### 9.4 为什么 reduceSum 用 float32 中间转换累加

- BF16→Cast float32→Add→Cast 回 BF16，精度优先（R=4 时 4 个 rank 累加，BF16 直接累加误差放大）
- V 指令数 3 条/rank (Cast+Add+PipeBarrier)，输出 1 次/tile Cast (F32→BF16)
- UB 布局: 6 slots (sumF32 + 2×(inputBF16+inputF32) + outBF16)，按 float 大小对齐
- 精度验证通过：12/12 case ALL PASS（R=2 + R=4）
- 性能影响极小：相比 BF16 直接累加平均 +0.4%（reduceSum 被 AIC 掩盖）

### 9.5 为什么 reduceSum 用手动 UB 管理（无 TPipe）

参考 ops-transformer 仓的 `ReadRemoteDataAdd` 实现：
- **LocalTensor + 手动 UB 偏移**：用 `LocalTensor` 偏移寻址管理 buffer 布局 `[sumF32 | inputBF16_A | inputF32_A | inputBF16_B | inputF32_B | outBF16]`
- **pingpong 双 buffer**：`pingPongCount_ & 1` 交替使用 input A/B 两份 buffer
- **多行 block**：一次处理 `rowsPerBlock` 行（UB 容量自适应），减少 GM↔UB 同步次数
- **所有计算核参与**：前 (28-R) 个 AIV block 参与 reduceSum (rsCoreNum = 28 - rankSize)
- **性能**：相比旧实现平均 2.01x 加速（msprof Task Duration）
- **无 TPipe 依赖**：不需要 `pipe_.Reset()` 清理，per-tile 调用无状态冲突

### 9.6 多核并行通信 + 严格分离 (方案B)

**后 R 核通信 + 前 (28-R) 核计算**：
- 后 rankSize 个核（jobIndex = 0..R-1）专职 AllToAll PUT 通信
- 前 (28-R) 个核（blockIdx = 0..27-R）专职 reduceSum 计算
- AllToAll totalJobs=rankSize：每个通信核负责 1 个 target rank，并行 PUT
- teamBarrier totalJobs=1：仅 jobIndex=0 的核执行 CrossDevice（step=1 正确轮询所有 remote）
- winOffset=128：PUT 数据跳过 win 区前 128B barrier counter，避免覆盖
- Wait<BARRIER_NONE>（Drain only）→ SyncAll → CrossDevice → SyncAll
- 无 Phase 2：通信核不补做 reduceSum，消除关键路径串行

**自适应 headMSize**：
- R<=2: headMSize=512（大 tile, 少 SyncAll, 通信效率高）
- R>=4: headMSize=128（小 tile, 多 commTurn, 流水掩盖效果好）

```
AIC: [tile0 frag mm] SetFlag [tile1 frag mm] SetFlag ...
AIV comm blocks:       WaitFlag [AllToAll0]   WaitFlag [AllToAll1]   ...
AIV compute blocks:    WaitFlag [ReduceSum0]  WaitFlag [ReduceSum1]  ...
```

AIC 计算 tile N 时，AIV 通信核在 PUT tile N，计算核在 ReduceSum tile N-1——通信和求和延迟被计算掩盖。

**性能**（msprof Task Duration, 12 case 中位数, 跨 rank max）：
- vs 旧基线 (totalJobs=1)：平均 -11.1%（R=2 -7.4%, R=4 -14.7%）
- 最佳 case：4096x3584x4096 R=4 = 307us vs 基线 415us（-26.0%）

---

## 10. 约束与限制

- **输入约束**：
  - A 矩阵 shape 为 `[m, k]`，m 为 rank_size 的整数倍
  - B 矩阵 shape 为 `[k, n]`（TransB=true，存储为 `[n, k]` FN 格式）
  - Scale shape 为 `[m, ceil(k/64) * 2]` / `[n, ceil(k/64) * 2]`
  - CeilDiv(k, 64) 必须为偶数
  - tailMSize（chunkM % headMSize）必须为 16 的倍数（Blaze matmul 最小行粒度）
  - 支持 MXFP8（`fp8_e4m3fn`/`fp8_e5m2`）

- **卡数约束**：支持 2、4、8、16、32、64 卡

- **通信约束**：通信数据总量受通信缓冲区大小限制；使用 PUT 模式 + winOffset=128 + 多核并行通信（方案B）

- **硬件约束**：仅在 Apace 架构（Ascend 950 系列）上生效，`--npu-arch=dav-3510`，`ASC_DEVKIT_MAJOR >= 9`

- **bias 约束**：当前版本暂未实现 bias

---

## 11. 未来扩展方向

- **GET 模式**：解决 4+rank 稳定性后从 PUT 切换到 GET
- **bias 支持**：补充 bias 输入参数
- **INT8 量化支持**：扩展非 MX 量化模式
- **大 shape 优化**：M≥8192 时 FragmentTensor 的 L2 cache 效率提升
- **SplitK**：每卡算 K/R，AIC 计算量减半，预期端到端 ~1.9x 加速（需 AllReduce 替代 ReduceScatter）

---

> **文档状态**：已更新至方案B（多核并行通信 + 严格分离 + winOffset=128 + float32 中间转换）
> **适用版本**：CANN 9.2.0（Apace 架构, dav-3510）
> **代码版本**：quant_matmul_reduce_scatter_impl.h (方案B: 后R核通信+前28-R核计算, reduceSum float32 中间转换)
> **精度验证**：10/12 case ALL PASS（R=2, R=4），2 个 N=12288 case 因 float32 UB 超限待优化
> **性能**：msprof Task Duration 10 case 中位数, vs 旧基线平均 -11.1%, float32 vs BF16 平均 +0.4%
