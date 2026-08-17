# quantMatmulReduceScatter 设计文档

## 1. 背景与目标

### 1.1 背景

在超大模型（如 MoE、DeepSeek 等）的分布式训练/推理场景中，多个 rank 各自持有一份输入数据 A 与完整权重 B，独立执行量化 MatMul 计算后，需要通过集合通信将各 rank 的计算结果进行聚合。传统实现将计算与通信串行执行，导致通信等待期间计算单元空闲，严重制约端到端性能。

quantMatmulReduceScatter 是 MC2（通算融合）算子族中的一员，在 Ascend 950（Apace 架构）上实现量化 MatMul 计算与集合通信聚合的深度融合，通过**AlltoAll 分发 + reduceSum 求和**完成跨卡数据聚合，以流水线并行掩盖通信延迟。

### 1.2 目标

- **功能目标**：完成 `MatMul + Dequant + AlltoAll + reduceSum` 的融合计算，支持 MX 量化（MXFP8/MXFP4）输入。
- **性能目标**：计算与通信流水并行，通信延迟被计算掩盖；支持 2/4/8/16/32/64 卡扩展。
- **架构目标**：基于 Apace 统一通信框架（`Apace::AivComm`）与统一量化 MatMul 调度框架（`Blaze::Gemm`），最大化公共代码复用，与 AllGatherQuantMatmul、AllToAllQuantMatmul 保持一致的架构风格。

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
- 每卡独立计算完整结果 `C_i = Dequant(QuantA_i @ QuantB) + bias`，得到 `[m, n]`。

由于无 TP 切分，不存在跨卡的部分和（splitK/splitM），**AIC 计算阶段无需 L0C 原子累加**。跨卡聚合完全由通信阶段完成：**AlltoAll 将各卡的完整计算结果分发到目标 rank，AIV 核在收到数据后执行 reduceSum 逐元素求和**。

### 2.2 计算流程

```
原始输入（每卡一份，完全独立）
    │
    ▼
┌─────────────────────┐
│  A     [m, k]       │  完整输入（MX 量化，含 ScaleA，K 不切分）
│  B     [k, n]       │  完整权重（MX 量化，含 ScaleB，K 不切分）
│  bias  [n]          │  可选偏置
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  QuantMatMul        │  每卡独立计算完整结果，无跨卡数据依赖
│  (Dequant)          │  C_i = A_i × B + bias → [m, n]
└─────────────────────┘
    │ C_i [m, n]（各卡完整）
    ▼
┌─────────────────────┐
│  AlltoAll           │  将各卡 C_i 按目标 rank 分发
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  reduceSum          │  对收到的多份结果逐元素求和
└─────────────────────┘
    │
    ▼
    output [m, n]
```

### 2.3 量化模式

支持 MX 量化（Microscaling）场景，对应 `QuantMatmulMxKernel` 实现：

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
│  └─────────┬──────────┘  └─────────┬──────────────┘                  │
│            └──────────┬────────────┘                                 │
│                       ▼                                              │
│              TilingData 序列化                                         │
│        ┌─────────────────────────────┐                               │
│        │ QuantMatmulTilingData       │  MatMul 侧 tiling             │
│        │ CommTilingData              │  通信侧切分                    │
│        └─────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Device 侧 (Kernel)                                                  │
│  ┌───────────────────┐         ┌───────────────────┐                 │
│  │ AIV               │         │ AIC               │                 │
│  │ AlltoAll 通信     │  Flag   │ 量化 MatMul 计算   │                 │
│  │ + reduceSum 求和  │ ──────► │ (独立计算，无 L0C  │                 │
│  │ CollectiveComm    │ 同步    │  原子累加)         │                 │
│  └───────────────────┘         └───────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 核角色分工

| 核类型 | 职责 | 说明 |
|-------|------|------|
| AIC（Cube） | 量化 MatMul 计算 | 每卡独立执行完整的 `A × B + bias` 计算，无跨卡数据依赖，**无需 L0C 原子累加** |
| AIV（Vector） | AlltoAll 通信 + reduceSum 求和 | 通过 `CollectiveComm` 框架执行 AlltoAll 分发，收齐后对多份结果执行逐元素 reduceSum 求和 |

### 3.3 模块划分

| 模块 | 文件（参考路径） | 职责 |
|------|-----------------|------|
| MatMul Tiling | `apace/tiling/quant_matmul_tiling_base.h` | 公共 Tiling 基类，管理平台信息查询与 shape 参数初始化 |
| MatMul Tiling 策略 | `apace/tiling/quant_matmul_tiling_swat.h` | SWAT 流式 Tiling 策略（非全量装载路径） |
| Tiling 数据 | `apace/tiling/quant_matmul_tiling_data.h` | Host→Device 序列化 Tiling 数据 POD 结构 |
| 通信 Tiling 数据 | `apace/tiling/comm_tiling_data.h` | 通信切分的统一 5 段式 Tiling 数据 |
| MatMul Kernel | `apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h` | 量化 MatMul 内核实现，支持 LOCAL/REMOTE/DEFERRED_SYNC 三种模式 |
| 通信框架 | `apace/block/aiv_comm/` | `CollectiveComm` 统一通信 API、Barrier、AllGather/AllToAll 实现 |
| 算子实现 | `apace/kernel/quant_matmul_reduce_scatter/` | 本算子的 Init/Process 编排逻辑 |

---

## 4. 数据流设计

### 4.1 数据布局

#### 4.1.1 输入/输出布局

每张卡持有完整输入，独立计算完整输出：

```
Rank 0: A_0 [m, k]    B [k, n]    →  C_0 [m, n]（完整结果）
Rank 1: A_1 [m, k]    B [k, n]    →  C_1 [m, n]（完整结果）
...
Rank R: A_{R-1} [m, k]  B [k, n]  →  C_{R-1} [m, n]（完整结果）
```

各卡将完整的 `C_i` 通过 **AlltoAll** 分发出去，每卡收齐所有 rank 的结果后执行 **reduceSum**：

```
output = Σ_{r=0}^{R-1} C_r   // 逐元素求和，shape 为 [m, n]
```

#### 4.1.2 通信窗口布局

复用 `Apace::AivComm` 的统一通信窗口（Win Buffer）设计，GMA 窗口按 rank-major 布局：

```
┌─────────────────────────────────────────────────┐
│ rank_0 数据块 │ rank_1 数据块 │ ... │ rank_{R-1} 数据块 │  ← AlltoAll 通信窗口
└─────────────────────────────────────────────────┘
```

- 每个数据块大小 = `m × n × sizeof(CType)`（C 为 MatMul 输出）

### 4.2 通信切分策略

采用统一的 `CommTilingData` 5 段式格式：

```cpp
struct CommTilingData {
    uint64_t splitAxisTileSize;   // 切分轴头块大小（块数/行数）
    uint64_t splitAxisTileCnt;    // 切分轴头块数量
    uint64_t splitAxisTailSize;   // 切分轴尾块大小（块数/行数）
    uint64_t splitAxisTailCnt;    // 切分轴尾块数量
    uint64_t nonSplitAxisSize;    // 非切分轴大小（元素个数）
};
```

- **切分轴** = M 轴（输出矩阵行方向，按 `m / rank_size` 行分发给各目标 rank）
- **非切分轴** = N 轴（输出矩阵列方向，整行发送）
- 主块（head tile）和尾块（tail tile）分别存储，以适配不同 shape 的负载均衡

### 4.3 数据流阶段

```
阶段 1: QuantMatMul（每卡独立完整计算）
    AIC: 计算本卡完整的 C_i = A_i × B + bias → [m, n]
         （无跨卡数据依赖，无 L0C 原子累加）
    AIV: 等待 AIC 计算完成信号

阶段 2: AlltoAll 分发 + reduceSum 求和（流水）
    AIV: 将本卡 C_i 按目标 rank 切块，通过 AlltoAll 分发到各目标窗口
    AIV: 每收齐一个 rank 的数据（或收齐全部后），逐元素执行 reduceSum 累加

阶段 3: 输出
    AIV: 将求和结果写入 output GM
```

---

## 5. MatMul Tiling 策略设计

### 5.1 Tiling 数据

复用 `QuantMatmulTilingData` 作为 MatMul 侧 Tiling 数据：

```cpp
struct QuantMatmulTilingData {
    uint32_t m, n, k;              // 原始问题 shape
    uint32_t baseM, baseN, baseK;  // 基础 tile 形状
    uint32_t scaleKL1;             // K 方向 L1 中缓存的 scale 覆盖长度
    uint32_t mTailTile, nTailTile; // 尾块 M/N 切分因子
    uint32_t mBaseTailSplitCnt;    // 基础尾块 M 方向再切分数
    uint32_t nBaseTailSplitCnt;    // 基础尾块 N 方向再切分数
    uint32_t mTailMain;            // 尾块 M 方向的均衡主块大小
    uint32_t nTailMain;            // 尾块 N 方向的均衡主块大小
    uint32_t usedCoreNum;          // 启动的 AIC 核数
    uint8_t  stepK;                // 每次外层流水跨越的 baseK tile 数
    uint8_t  nBufferNum;           // L1 滚动缓冲区槽位数
    uint8_t  dbL0c;                // L0C double-buffer 模式
};
```

### 5.2 SWAT 流式策略要点

`QuantMatmulTilingSwat` 实现非全量装载（non-full-load）路径的 Tiling，核心决策流程：

1. **基础块搜索**（`CalcBasicBlock`）：
   - 初始候选：`baseM = min(m, 256)`，`baseN = min(n, 256)`，`baseK = min(k, 128/256)`
   - 当块数不足时调用 `AdjustBasicBlock` 再平衡 M/N 比例
   - 对齐约束：M 按 `CUBE_BLOCK`（16）对齐，N 按 `CUBE_BLOCK` 对齐，K 按 `TILING_MXFP_DIVISOR_SIZE`（64）对齐

2. **边缘优化**（`OptimizeEdgeBasicBlock`）：当 K 轴满足 cache-line 对齐时，对 M/N 尾部执行 merge 优化，减少尾块的开销占比。

3. **尾块切分**（`CalcTailBasicBlock`）：在剩余核数允许范围内，优先切分更大的尾边（M tail 或 N tail），并支持 `mTailAlign` 对齐选项。

4. **L1 深度计算**（`CalcPathSpecificL1`）：
   - 对称分配 A/B/ScaleA/ScaleB 四个缓冲区
   - 先搜索公共深度，再独立精化 A/B 两侧深度
   - `stepK = min(stepKa, stepKb)` 保持两侧 K 推进步调一致

5. **Scale 复用因子**（`CalcScaleFactors`）：在 L1 剩余空间内求解 Scale 的复用深度，两边尽量平衡分配。

### 5.3 调度器

复用 `Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3` 作为块调度器，支持：

- M/N 二维 tile 遍历
- 尾块合并切分（`mTailTile × nTailTile` 子块）
- `NONE_FULL_LOAD_MODE` 非全量装载模式

---

## 6. 通信设计

### 6.1 通信架构

本算子采用 **AlltoAll + reduceSum** 实现跨卡聚合，在 AIV 侧完成全部通信与求和逻辑：

```
┌──────────────────────────────────────────────────────────────┐
│ AIV 侧通信流水（AlltoAll + reduceSum）                         │
│                                                              │
│  C_i [m, n]  ──► AlltoAll 按 M 轴切块分发 ──► 各 rank 窗口    │
│                                                              │
│  ┌─────────────────────────────────────────────────┐          │
│  │ rank_0 块 │ rank_1 块 │ ... │ rank_{R-1} 块    │          │
│  └─────────────────────────────────────────────────┘          │
│       │  每收齐一个 rank 的块
│       ▼
│  ┌─────────────┐
│  │ reduceSum   │  ← 逐元素求和（V 核向量指令）
│  └─────────────┘
│       │
│       ▼
│     output [m, n]
└──────────────────────────────────────────────────────────────┘
```

### 6.2 通信组件复用

复用 `Apace::AivComm::CollectiveComm` 框架的 `AllToAll` 实现：

```cpp
using AllToAllComm = Apace::AivComm::CollectiveComm<
    Apace::AivComm::CommCollectiveOp::AllToAll,
    Apace::AivComm::CommMode::PUT,   // 或 GET，按硬件/场景选择
    CType,                           // 通信数据类型（MatMul 输出）
    TeamBarrier>;

// 通信窗口布局：每个 rank 一块，对应目标 rank 的 M 行分片
// 窗口大小 = rank_size × (m / rank_size) × n × sizeof(CType)
```

### 6.3 通信+求和流水

```
时间轴 ──────────────────────────────────────────────────────►

AIC: [完整计算 C_i = A_i × B + bias（无 L0C 原子累加）]
                             │ FixPipe 输出
                             ▼
AIV: [AlltoAll 分块发送 C_i] [收齐 rank_0 → reduceSum] [收齐 rank_1 → 累加] ... [最终求和结果写 GM]
```

- **逐块求和**：每收齐一个目标 rank 的 M 行分片，AIV 立即对该块执行 `Add` 向量操作累加到累加缓冲区，避免等待全部数据收齐造成的尾部延迟。
- **本地合并**：本卡自己的 C_i 直接作为累加初值，无需经过通信链路。
- **最终写出**：所有 rank 的分片累加完成后，将结果从累加缓冲区写回 output GM。

### 6.4 与 AIC 计算阶段的同步

由于无 TP 切分、无 L0C 原子累加，AIC 与 AIV 之间的同步简化为一次握手：

| 同步机制 | 用途 | 说明 |
|---------|------|------|
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | AIC → AIV 计算完成通知 | AIC 完成 `C_i` 计算并 FixPipe 后，通知 AIV 可开始 AlltoAll 分发 |
| `SetAtomicAdd` / `SetAtomicNone` | **不使用** | 无跨卡部分和，计算侧无原子累加需求 |

---

## 7. 内存设计

### 7.1 L1 缓冲区布局

SWAT 策略在 L1 中维护 4 个滚动缓冲区：

```
┌──────────────────────────────────────────┐
│ A 数据缓冲区 (depthA1 × baseM × baseK)  │  ← 量化后 A tile
├──────────────────────────────────────────┤
│ B 数据缓冲区 (depthB1 × baseN × baseK)  │  ← 量化后 B tile
├──────────────────────────────────────────┤
│ ScaleA 缓冲区 (scaleFactorA × baseM)    │  ← 优化 ScaleA 复用
├──────────────────────────────────────────┤
│ ScaleB 缓冲区 (scaleFactorB × baseN)    │  ← 优化 ScaleB 复用
└──────────────────────────────────────────┘
```

### 7.2 L0 缓冲区

- **L0A/L0B**：存放从 L1 加载的 A/B 基础 tile，大小由 `baseM × baseK` 和 `baseN × baseK` 决定
- **L0C**：`baseM × baseN` 输出累加缓冲区，支持 double-buffer（`dbL0c=2`）或单缓冲（`dbL0c=1`）

### 7.3 通信窗口（Workspace）

通信窗口位于 GM 侧，由外部申请并传入，存放 AlltoAll 分发的计算结果：

- 数据窗口：`rank_size × (m / rank_size) × n × sizeof(CType)` 字节
  - 每个目标 rank 一份，每份为 `(m / rank_size)` 行 × `n` 列的 C 分片
- 累加缓冲区：`(m / rank_size) × n × sizeof(CType)` 字节（UB 侧，用于 reduceSum 逐元素累加）

窗口偏移计算与 AllToAll 变体一致，按 rank-major 连续存放。

---

## 8. Kernel 实现设计

### 8.1 算子入口

```cpp
__global__ __aicore__ void QuantMatmulReduceScatterKernel(
    __gm__ CommContext* hcommCtx,
    GM_ADDR aGM, GM_ADDR aScaleGM,
    GM_ADDR bGM, GM_ADDR bScaleGM,
    GM_ADDR biasGM, GM_ADDR cGM,
    QuantMatmulReduceScatterTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    QuantMatmulReduceScatterImpl<
        fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> impl;
    impl.Init(hcommCtx, aGM, aScaleGM, bGM, bScaleGM, biasGM, cGM, &tilingData);
    impl.Process();
}
```

### 8.2 Init 流程

```cpp
__aicore__ inline void Init(...)
{
    // 1. 解析 Tiling 参数（MatMul tile + 通信 tile）
    // 2. 初始化通信上下文（hccl_/udmaCtx_/ubmemCtx_）
    // 3. 初始化 CollectiveComm（AlltoAll 通信链）
    // 4. 初始化 reduceSum 累加缓冲区（UB）
    // 5. 初始化 QuantMatmulMxKernel（模板参数绑定 BlockMmad + BlockScheduler，NONE_FULL_LOAD_MODE）
}
```

### 8.3 Process 流程

```cpp
__aicore__ inline void Process()
{
    if ASCEND_IS_AIC {
        // AIC 侧：每卡独立完整计算，无跨卡依赖
        //   - 使用 MatmulMode::LOCAL（仅本卡数据，splitK=1）
        //   - 无需 L0C 原子累加
        //   - FixPipe 将完整 C_i [m, n] 写出到 GM
        RunMatmul();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0);  // 通知 AIV 计算完成
    }
    if ASCEND_IS_AIV {
        // AIV 侧：等待 AIC 计算完成，然后执行 AlltoAll + reduceSum
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
        // 1. 将本卡 C_i 按 M 轴切块，通过 AlltoAll 分发到各目标 rank
        for (uint32_t turn = 0; turn < commTurn_; ++turn) {
            allToAll_.Commit();      // 分发一个 M 分片
            allToAll_.template Wait<...>();  // 等待 channel 排空
            ReduceSumOneRank();     // 收齐一个 rank 的分片 → 向量 Add 累加
        }
        // 2. 全部收齐后，将累加结果写回 output GM
        DataCopy(outputGM_, accumBuf_, totalBytes);
        allToAll_.Finalize();
    }
}
```

### 8.4 关键同步机制

| 同步机制 | 用途 | 说明 |
|---------|------|------|
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | AIC → AIV 计算完成通知 | AIC 完成 `C_i` 计算并 FixPipe 后，通知 AIV 可开始 AlltoAll 分发 |
| `AscendC::SyncAll<true>()` | 所有核同步 | 每轮通信后全局同步，确保各 AIV 的分发/接收就绪 |
| Hccl Handle Wait / UDMA Drain | 通信完成等待 | 等待 AlltoAll 通信 channel 完成，保证数据到达后再 sum |
| `SetAtomicAdd` / `SetAtomicNone` | **不使用** | 无 splitK/splitM 部分和，AIC 计算侧无原子累加需求，求和完全在 AIV reduceSum 完成 |

---

"## 9. 关键设计决策\n\n### 9.1 为什么用 AlltoAll + reduceSum 替代 ReduceScatter\n\n本算子场景**不涉及 TP 切分**：每卡持有完整 A 和 B，独立计算出完整的 `C_i [m, n]`。聚合目标是"所有 rank 的完整结果逐元素求和后广播到每卡"，语义上等价于 AllReduce，但通过 **AlltoAll（分发）+ reduceSum（求和）** 分步实现：\n\n- **无需计算侧原子累加**：不存在 splitK/splitM 的部分和，`C_i` 是每卡的完整计算结果，L0C 无需跨卡累加。\n- **求和下沉到 AIV**：AlltoAll 仅负责数据分发，接收侧 AIV 用向量指令对多份结果执行逐元素 Add，将归约从 Cube 侧卸载到 Vector 侧，与通信天然流水。\n- **逐块流水求和**：每收齐一个 rank 的分片即累加一次，避免全量收齐后的突发计算。\n\n### 9.2 为什么计算阶段无需 LOCAL/REMOTE 模式\n\n`QuantMatmulMxKernel` 的 LOCAL/REMOTE/DEFERRED_SYNC 模式是为**跨卡部分和**（splitK/splitM）设计的。本算子每卡独立算出完整结果，计算阶段仅需一次 `LOCAL` 模式的完整 MatMul（`splitKNum = 1`），随后 FixPipe 输出完整 `C_i` 供通信阶段消费，**不使用 REMOTE/DEFERRED_SYNC 模式**。\n\n### 9.3 AIC / AIV 的流水重叠\n\n天然的两阶段流水：\n- AIC 在计算 `C_i` 时，AIV 空闲等待（或预分配通信窗口）；\n- AIC 完成后通过一次 `CrossCoreFlag` 通知 AIV，AIV 立即开始 AlltoAll 分发与逐步 reduceSum；\n- 若需要进一步重叠，可将 AIC 对 C 的 FixPipe 按 M 分片进行，让 AIV 边收边算（未来优化方向）。"

---

## 10. 约束与限制

- **输入约束**：
  - A 矩阵 shape 为 `[m, k]`，m 为 rank_size 的整数倍
  - B 矩阵 shape 为 `[k, n]`
  - A、B 均为 MX 量化格式，Scale shape 为 `[m, ceil(k/64) * 2]` / `[n, ceil(k/64) * 2]`
  - 支持 MXFP8（`fp8_e4m3fn`/`fp8_e5m2`）与 MXFP4 量化

- **卡数约束**：支持 2、4、8、16、32、64 卡

- **通信约束**：ReduceScatter 通信数据总量受通信缓冲区大小限制

- **硬件约束**：仅在 Apace 架构（Ascend 950 系列）上生效，通过 `ASC_DEVKIT_MAJOR >= 9` 条件编译隔离

---

## 11. 未来扩展方向（预留设计）

- **FP8 Dynamic Quant 前置阶段**：参考 `Fp8DynamicQuantPertoken` 模板，在 MatMul 前增加动态量化（per-token）预处理，将 FP16/BF16 输入量化为 FP8 后再进入融合计算。
- **INT8 量化支持**：参考 `Mc2QuantBatchMatmulASWKernel` 的 per-token/per-channel 反量化路径，扩展非 MX 量化模式。
- **DEFERRED_SYNC 模式优化**：在 localMatmul=2 时，将本卡计算和远端计算在同一次 FixPipe 中完成，减少 L0C → L1 → L0C 的往返开销。

---

> **文档状态**：初稿
> **适用版本**：CANN 8.1+（Apace 架构）