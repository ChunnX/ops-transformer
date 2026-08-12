# apace_quant_matmul_reduce_scatter_v2 算子设计文档

> **路线**：APACE MC2 通算融合（UDMA PUT 直调）
> **v1 基底**：`apace_quant_matmul_reduce_scatter`（分支 `feat/apace-quant-matmul-reduce-scatter`）
> **参考实现**：ops-nn `quant_matmul_reduce_sum`（流水线 + workspace 轮转模式）

---

## 0. 概述

### 0.0 需求类型判断

**通用**：支持多种 FP8 dtype 组合和 shape 配置，4 卡 ReduceScatter 通信，3 级流水架构。

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | apace_quant_matmul_reduce_scatter_v2 |
| 算子类别 | MC2 通算融合（Quant MatMul + AllToAll PUT 通信 + ReduceSum 流水线） |
| 需求类型 | 通用 |
| 支持数据类型 | A/B = fp8_e4m3fn_t / fp8_e5m2_t（4 种组合），C = bfloat16_t |
| 支持芯片 | Ascend950PR（DAV_3510） |
| NpuArch | DAV_3510（`__NPU_ARCH__` = 3510） |
| `--npu-arch` 编译参数 | `dav-3510` |
| SocVersion | ASCEND950 |
| CANN 版本 | 9.2.0 |
| 通信模式 | UDMA PUT（直调仅支持 UDMA） |
| 多卡拓扑 | 4 卡 ReduceScatter via AllToAll PUT |
| 流水线深度 | PIPELINE_NUM=4（多 buffer workspace 轮转） |
| 特殊约束 | APACE 路线四大红线（见 §2.0） |

### 0.2 用户原始需求

| # | 需求内容 |
|---|---------|
| 1 | 对 v1（apace_quant_matmul_reduce_scatter，AllToAll PUT + AtomicAdd 串行模式）进行架构重构 |
| 2 | 拍 3 级流水 mm→aivComm→reduceSum，使用 pipe，多 buffer 并行，PIPELINE_NUM=4 |
| 3 | reduceSum 参考 ops-nn 仓的 quant_matmul_reduce_sum 实现 |
| 4 | local 前置作为优化项（默认不启用，作为 tiling 可选参数） |
| 5 | 新增 workspace 入参（算子入参，申请额外内存），mm 输出写 workspace，reduceSum 从 workspace 读取 |
| 6 | AIV 核时分复用：每个 AIV 核先做 AllToAll PUT 通信，通信完成后切换到 reduceSum |
| 7 | mm 输出类型 BF16（Blaze MX QuantMatmul 硬件已做 FP8→BF16 反量化），reduceSum 不需要 AscendDequant |
| 8 | 硬件环境：4 × Ascend950PR（dav-3510 架构），CANN 9.2.0，CANN 路径 /home/l00947526/Ascend806/cann |
| 9 | APACE 路线约束：禁止 `__schedmode__(1)` / `[[bisheng::core_ratio(1,1)]]`；Matmul 走 Blaze 模板；禁止修改 `block/` 和 `tiling/`；直调仅 UDMA；禁止 HCCL 高阶 API；必须有 `KERNEL_TYPE_MIX_AIC_1_1` 核配比声明 |

### 0.3 v1 → v2 架构重构概述

v1 采用串行模式：AIV 做 AllToAll PUT 通信，AIC 做 mm 并直接通过 AtomicAdd 写 yGm。v2 重构为 3 级流水线架构，引入 workspace 中间缓冲区，实现 mm→aivComm→reduceSum 的流水并行。

---

## 1. 算子设计

### 1.1 数学公式

```
输入:
  rank i 持有:
    A_i  - shape [M, K_local],  dtype = fp8_e4m3fn_t / fp8_e5m2_t   (K_local = K / rankSize)
    ScaleA_i - shape [M, CeilDiv(K_local,64)*2], dtype = fp8_e8m0_t  (MXFP8 scale)
    B_full - shape [K, N],      dtype = fp8_e4m3fn_t / fp8_e5m2_t   (全量复制)
    ScaleB_full - shape [CeilDiv(K,64)*2, N], dtype = fp8_e8m0_t    (全量复制)
    workspace - GM 地址，大小 = PIPELINE_NUM × coreNum × baseM × baseN × sizeof(bfloat16_t)

输出:
  rank r 获得:
    C_r - shape [M/rankSize, N], dtype = bfloat16_t                 (ReduceScatter 输出)

数学公式:
  C_r = Σ_{i=0}^{rankSize-1} dequant(A_i[r*M/R : (r+1)*M/R, :]) × dequant(B_i[:, :])
  其中 R = rankSize = 4
  跨 rank 累加 Σ_i 在 AIC mm 的 L0C FP32 累加器中完成（for rank 循环 + mmadOp_ remoteRankCnt）

3 级流水:
  Stage 1 (AIC):  mm       — Blaze MX QuantMatmul: A[Win] × B → L0C FP32 跨 rank 累加 → fixpipe → workspace[BF16]
  Stage 2 (AIV):  aivComm  — AllToAll PUT: 发送 A 数据到各 rank 的 Win 区
  Stage 3 (AIV):  reduceSum — DataCopyPad(workspace→UB) + DataCopyPad(UB→yGm)（无 AtomicAdd）

ReduceScatter 语义:
  - 每个 rank 持有完整 M×K_local 的 A，计算全量部分和
  - 输出按 M 轴切分：rank r 获得 C 的第 r 个 M/rankSize 段
  - 跨 rank 部分和累加在 AIC mm 的 L0C 中完成（非 reduceSum 阶段）
  - workspace 中已是最终结果 Σ_i mm(A_i × B_i)，reduceSum 仅做 DataCopyPad 搬运
```

### 1.2 API 映射

| 功能 | API/组件 | 来源 | 验证状态 |
|------|---------|------|---------|
| MX 量化 Matmul 计算 | `Blaze::Gemm::Block::BlockMmad<MatmulWithScaleMx<0,false>, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>` | ops-tensor `blaze/gemm/block/block_mmad_qbmm_mx.h` | ✅ v1 已验证 |
| Matmul tile 调度 | `Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>` | ops-tensor `blaze/gemm/block/block_scheduler_qbmm.h` | ✅ v1 已验证 |
| MX 量化 DispatchPolicy | `Blaze::Gemm::MatmulWithScaleMx<0, false>` | ops-tensor `blaze/gemm/policy/dispatch_policy.h` | ✅ v1 已验证 |
| Apace 组合模式 Kernel | `QuantMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler, CommPolicy>` | v1 kernel（复制+改造） | ✅ v1 已验证 |
| AIV 通信（data） | `CollectiveComm<AllToAll, PUT, AType, TeamBarrier>` | apace `block/aiv_comm/collective_comm_api.h` | ✅ v1 已验证 |
| AIV 通信（scale） | `CollectiveComm<AllToAll, PUT, fp8_e8m0_t, TeamBarrier>` | apace `block/aiv_comm/collective_comm_api.h` | ✅ v1 已验证 |
| 跨卡同步 | `TeamBarrier` | apace `block/aiv_comm/barrier/barrier_ubmem.h` | ✅ v1 已验证 |
| AIC 等 AIV 通信完成 | `CrossCoreWaitFlag<0x2, PIPE_MTE2>(commFlagId)` | AscendC API | ✅ v1 已验证（CrossCoreWaitFlag_ISASI.md，950PR 支持模式 2 + PIPE_MTE2） |
| AIV 通知 AIC 通信完成 | `CrossCoreSetFlag<0x2, PIPE_MTE3>(commFlagId)` | AscendC API | ✅ v1 已验证（CrossCoreSetFlag_ISASI.md，950PR 支持模式 2 + PIPE_MTE3） |
| AIC 通知 AIV mm 完成 | `CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG=5)` | AscendC API | ✅ 验证（CrossCoreSetFlag_ISASI.md，950PR 支持模式 2 + PIPE_FIX） |
| AIV 等 AIC mm 完成 | `CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG=5)` | AscendC API | ✅ 验证（CrossCoreWaitFlag_ISASI.md，950PR 模板参数生效；模式 2 支持 PIPE_MTE2；阻塞 reduceSum 的 DataCopyPad workspace→UB） |
| AIV 通知 AIC workspace 可复用 | `CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG=3)` | AscendC API | ✅ 验证（CrossCoreSetFlag_ISASI.md，950PR 支持模式 2 + PIPE_MTE3） |
| AIC 等 AIV workspace 释放 | `CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(SYNC_AIV_AIC_FLAG=3)` | AscendC API | ✅ 验证（CrossCoreWaitFlag_ISASI.md，950PR 模板参数生效；模式 2 支持 PIPE_M；阻塞 mm 计算） |
| AIV 块间同步 | `SyncAll<true>()` | AscendC API | ✅ v1 已验证（SyncAll.md） |
| reduceSum: workspace→UB | `DataCopyPad(localTensor, globalTensor, dataCopyParams, padParams)` | AscendC API | ✅ 验证（DataCopyPad_GMToUB.md，950PR 支持 bfloat16_t） |
| reduceSum: UB→yGm | `DataCopyPad(globalTensor, localTensor, dataCopyParams)` | AscendC API | ✅ 验证（DataCopyPad_UBToGM.md，950PR 支持 bfloat16_t；普通搬运，无 AtomicAdd） |
| 管线屏障 | `PipeBarrier<PIPE_ALL>()` | AscendC API | ✅ v1 已验证（PipeBarrier_ISASI.md，950PR 支持 PIPE_ALL） |
| AIV 核索引（reduceSum 调度复刻） | `GetBlockIdx()` | AscendC API | ✅ 验证（GetBlockIdx.md：1:1 配比下 AIC/AIV 取值范围均为 [0, numBlocks)） |
| 逻辑核数（reduceSum 调度复刻） | `GetBlockNum()` | AscendC API | ✅ 验证（GetBlockNum.md：返回 numBlocks，AIC/AIV 一致） |
| 核配比（reduceSum 调度复刻） | `GetTaskRatio()` / `GetTaskRation()` | AscendC API | ✅ 验证（GetTaskRatio.md：MIX_AIC_1_1 下 AIC=1、AIV=1） |
| 核配比声明 | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` | kernel_basic_intf.h | ✅ v1 已验证 |
| Host 通信建链 | `CommChannelBuilder::CreateDeviceContext` | apace `utils/comm_channel_builder.h` | ✅ v1 已验证 |
| SWAT tiling 推导 | `QuantMatmulTilingSwat<DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN>` | apace `tiling/quant_matmul_tiling_swat.h` | ✅ v1 已验证 |
| yGm 清零 | 不需要 | ❌ 已移除——reduceSum 使用普通 DataCopyPad（非 AtomicAdd）直接覆盖 yGm 旧值，无需清零 | — |

#### API 语义验证详情

| API | 验证项 | 验证结果 | 文档 |
|-----|--------|---------|------|
| CrossCoreSetFlag | 模式 2 + PIPE_FIX | ✅ 950PR 支持模式 0/1/2/4；模式 0/1/2 支持 PIPE_V/M/MTE1/MTE2/MTE3/FIX | CrossCoreSetFlag_ISASI.md |
| CrossCoreWaitFlag | 950PR 模板参数生效 + pipe 约束 | ✅ 950PR 硬件支持配置模式和流水类型，模板参数生效；模式 0/1/2 不支持 PIPE_S/PIPE_ALL，PIPE_S 仅模式 4 支持；本设计使用 modeId=2 + PIPE_MTE2（AIV 侧）/ PIPE_M（AIC 侧） | CrossCoreWaitFlag_ISASI.md |
| SetAtomicAdd | 不使用 | ❌ 已移除——AIC mm 在 L0C 中跨 rank FP32 累加后单次 fixpipe 写 workspace，workspace 中已是最终结果，reduceSum 只需 DataCopyPad 不需 AtomicAdd | — |
| DataCopyPad GM→UB | bfloat16_t | ✅ 950PR 支持（含 mode 模板参数原型）；支持 bfloat16_t | DataCopyPad_GMToUB.md |
| DataCopyPad UB→GM | bfloat16_t | ✅ 950PR 支持（含 mode 模板参数原型）；支持 bfloat16_t | DataCopyPad_UBToGM.md |
| flagId 计数器 | 范围 0-15，截断低 4bit | ✅ 模式 0/1/2 每核 16 个 flagId（0-15），超出截断；计数器范围 0-15 | CrossCoreSetFlag_ISASI.md |

### 1.3 数据流

```
┌──────────────────────────────────────────────────────────────────────────────┐
│           v2: 3 级流水 ReduceScatter (AllToAll PUT + workspace 轮转)          │
│                                                                              │
│  rank i 持有:                                                                │
│    A_i[M × K_local], ScaleA_i, B_full[K × N], ScaleB_full                   │
│    workspace[PIPELINE_NUM × coreNum × baseM × baseN × BF16]                 │
│                                                                              │
│  ┌──────────── AIV (时分复用: 通信 → reduceSum) ────────────┐                │
│  │                                                            │                │
│  │  Phase 1: AllToAll PUT 通信                                │                │
│  │    for tid in 0..commTurn-1:                               │                │
│  │      Commit(ScaleA_i[tid 段]) → 各 rank Win                │                │
│  │      Commit(A_i[tid 段])       → 各 rank Win                │                │
│  │      Wait<BARRIER_DEVICE>()                                │                │
│  │      SyncAll<true>()                                       │                │
│  │      SetFlag<0x2, PIPE_MTE3>(commFlagId) ───────→ AIC     │                │
│  │                                                            │                │
│  │  Phase 2: reduceSum (mm 输出已在 workspace)                │                │
│  │    WaitFlag<SYNC_MODE2, PIPE_MTE2>(5)  ← AIC mm 完成        │                │
│  │    DataCopyPad(workspace[tid%4] → UB)                      │                │
│  │    DataCopyPad(UB → yGm[my_M_seg])  (无 AtomicAdd)         │                │
│  │    SetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG=3)     │
│  │                                      ───────→ AIC (回压)   │                │
│  │                                                            │                │
│  │  Finalize(ScaleA) + Finalize(A)                            │                │
│  └────────────────────────────────────────────────────────────┘                │
│                              ↕ CrossCore Flags                                │
│  ┌──────────────────── AIC (mm 计算) ────────────────────────┐                │
│  │                                                            │                │
│  │  for each tile (BlockScheduler 调度):                      │                │
│  │    if cubeNum >= PIPELINE_NUM:                             │                │
│  │      WaitFlag<SYNC_MODE2, PIPE_M>(3) ← 回压: workspace 可用│                │
│  │    WaitFlag<0x2, PIPE_MTE2>(commFlagId) ← AIV 通信完成     │                │
│  │    workSpaceOffset = baseM × baseN ×                       │                │
│  │                      (coreIdx + (cubeNum % PIPELINE_NUM)   │                │
│  │                       × coreNum)                           │                │
│  │    mm: Blaze QuantMatmul(A[Win] × B → L0C FP32 累加        │                │
│  │         → fixpipe → workspace[BF16])                       │                │
│  │      gmCWorkspace = MakeTensor(workspace + workSpaceOffset)│                │
│  │    SetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG=5)      │
│  │                                      ───────→ AIV (mm 完成)│                │
│  │    cubeNum++                                               │                │
│  │                                                            │                │
│  │  PostCompute: drain 剩余 Min(cubeNum, PIPELINE_NUM) 个     │                │
│  │              WaitFlag(SYNC_AIV_AIC_FLAG=3)                 │                │
│  └────────────────────────────────────────────────────────────┘                │
│                                                                              │
│  workspace 布局 (PIPELINE_NUM=4 轮转):                                       │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │ slot 0: [coreIdx=0][baseM×baseN×BF16]                   │                  │
│  │ slot 0: [coreIdx=1][baseM×baseN×BF16]                   │                  │
│  │ ...                                                      │                  │
│  │ slot 0: [coreIdx=coreNum-1][baseM×baseN×BF16]           │                  │
│  │ slot 1: [coreIdx=0][baseM×baseN×BF16]                   │                  │
│  │ ...                                                      │                  │
│  │ slot 3: [coreIdx=coreNum-1][baseM×baseN×BF16]           │                  │
│  └────────────────────────────────────────────────────────┘                  │
│  workSpaceOffset = baseM × baseN × (coreIdx + (cubeNum % 4) × coreNum)       │
│                                                                              │
│  输出: rank r 获得 C_r[M/rankSize × N] = DataCopyPad(workspace[my_M_seg])      │
│       其中 workspace = AIC L0C 跨 rank FP32 累加后 fixpipe 写入的最终结果       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 核心计算步骤

**v2 Process() 流水循环伪代码**（参考 ops-nn 的 Process() 模式 + v1 的通信逻辑）：

```cpp
Process():
    // AIC + AIV 均执行
    // AIV: 时分复用 — 先通信后 reduceSum
    // AIC: 连续 mm，写 workspace

    if ASCEND_IS_AIV {
        // Phase 1: AllToAll PUT 通信
        for tid in 0..commTurn-1:
            if GetBlockIdx() < rankSize:
                allToAllScaleA_.Commit()
                allToAllA_.Commit()
                allToAllA_.Wait<BARRIER_DEVICE>()
            SyncAll<true>()
            CrossCoreSetFlag<0x2, PIPE_MTE3>(commFlagId(tid))  // 通知 AIC 通信完成

        allToAllScaleA_.Finalize()
        allToAllA_.Finalize()

        // Phase 2: reduceSum (时分复用切换)
        // ★ AIV 复用 BlockScheduler（方案 b，见 §2.16）：
        //   与 AIC ProcessSingleBatch 完全相同的 problemShape/schParams 构造调度器，
        //   完全相同的调用序列枚举 tile，mPos/nPos/curBaseM/curBaseN 由调度器推导，
        //   循环次数与 AIC SetFlag(5) 次数构造一致（不需要 totalMmTiles）
        ProblemShape problemShape{m, n, k, 1};          // 来自 tileQbmmTilingData（与 AIC 相同）
        BlockSchedulerParams schParams{...};            // 与 AIC 相同的 SetupParams 产物
        BlockScheduler bsAiv(problemShape, schParams);  // blockIdx_/coreNums_ 两侧一致（§2.16 论证）
        // 与 AIC 相同的 UpdateTailTile 判定（restBatch=0, isTailRound=true）
        if ((bsAiv.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= GetBlockNum()) {
            bsAiv.UpdateTailTile(mTailTile, nTailTile);
        }
        cubeNum = 0
        while bsAiv.GetTileIdx(blockIdx):                          // 与 AIC 同一枚举序
            singleShape = bsAiv.GetBlockShape<MX_PERGROUP_MODE, MX_PERGROUP_MODE, weightNz>(blockIdx)
            if singleShape.M <= 0 || singleShape.N <= 0:
                break                                            // 对齐 AIC 的 early-return
            bsAiv.GetTileCoord(blockIdx, mPos, nPos)             // yGm 位置由调度器推导
            curBaseM = singleShape.M; curBaseN = singleShape.N
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG=5)   // 等 AIC mm 完成
            // workSpaceOffset 与 AIC 侧一致（coreIdx = GetBlockIdx()，1:1 下两侧相同）
            uint64_t workSpaceOffset = baseM * baseN * (coreIdx + (cubeNum % PIPELINE_NUM) * coreNum)
            // DataCopyPad workspace → UB（2D 搬运：slot 内行距=curBaseN 紧密排列 srcStride=0，
            // UB 内行距按 32B 对齐，dstStride 公式同 ops-nn DataCopyPad2D）
            DataCopyExtParams cpIn{curBaseM, curBaseN * sizeof(BF16), 0,
                Ceil(curBaseN*2, 64)*2 - Ceil(curBaseN*2, 32), 0}
            DataCopyPadExtParams<BF16> padIn{false, 0, 0, 0}
            DataCopyPad(reduceSumBuf, workspaceGm[workSpaceOffset], cpIn, padIn)
            // DataCopyPad UB → yGm（2D 搬运，无 AtomicAdd）
            // outOffset = mPos * N + nPos（yGm 行主序，行距 N）
            DataCopyExtParams cpOut{
                curBaseM,                                           // blockCount（按行搬运）
                curBaseN * sizeof(BF16),                            // blockLen = 每行字节数
                (AlignUp(curBaseN, 16) - curBaseN) * sizeof(BF16) / 32,  // srcStride（UB 行对齐填充）
                (N - curBaseN) * sizeof(BF16),                      // dstStride（GM 行间距, 950PR int64_t）
                0}
            DataCopyPad(yGm[mPos * N + nPos], reduceSumBuf, cpOut)  // 普通搬运，非 AtomicAdd
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG=3)  // 通知 workspace 可复用
            cubeNum++
    }

    if ASCEND_IS_AIC {
        cubeNum = 0
        for each tile (BlockScheduler 调度):
            if cubeNum >= PIPELINE_NUM:
                CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(SYNC_AIV_AIC_FLAG=3)   // 回压: 等 workspace 槽位释放
            // 等待 AIV 通信完成 (CalcDependTileIdx + waitedMask 去重)
            while readyTileIdx < dependTileIdx:
                readyTileIdx++
                CrossCoreWaitFlag<0x2, PIPE_MTE2>(commFlagId(readyTileIdx))

            // ★ v2 改造点：per-tile 动态创建 workspace Tensor
            // v1 的 gmC 在循环外创建一次 + per-tile Slice；v2 在循环内 per-tile 创建 gmCWorkspace
            workSpaceOffset = baseM * baseN * (coreIdx + (cubeNum % PIPELINE_NUM) * coreNum)
            auto layoutCTile = MakeLayoutC{}(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape))
            auto gmCWorkspace = Te::MakeTensor(
                Te::MakeMemPtr<Te::Location::GM>(
                    workspaceGmAddr_ + workSpaceOffset * sizeof(CType)),
                layoutCTile)  // baseM×baseN 的整块，不需要 Slice

            // mm: A[Win] × B → L0C FP32 跨 rank 累加 → fixpipe → workspace[BF16]
            // mmadOp_ 的 remoteRankCnt 控制 L0C 累加（0=reset, 递增=累加, 计满 splitKNum 触发 fixpipe）
            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias,
                    gmCWorkspace, singleShape, remoteRankCnt)

            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG=5)  // 通知 AIV mm 完成
            cubeNum++

        // PostCompute: drain 剩余回压 flag
        for idx in 0..Min(cubeNum, PIPELINE_NUM)-1:
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(SYNC_AIV_AIC_FLAG=3)
    }
```

**关键设计要点**:

1. **3 级流水**：AIC 做 mm（L0C FP32 跨 rank 累加 → fixpipe → 写 workspace），AIV 先做 AllToAll PUT 通信，后做 reduceSum（从 workspace 读，DataCopyPad 写 yGm）。PIPELINE_NUM=4 多 buffer 轮转，AIC 可领先 AIV reduceSum 最多 4 个 tile。

2. **workspace 轮转**：`workSpaceOffset = baseM × baseN × (coreIdx + (cubeNum % PIPELINE_NUM) × coreNum)`，每个 AIC 核在每个 cubeNum 对应一个 workspace 槽位，4 个槽位循环复用。

3. **CrossCore flag 编排（3 通道，950PR 模板参数生效）**：
   - 通道 1（AIV→AIC，通信就绪）：`SetFlag<0x2, PIPE_MTE3>(commFlagId)` / `WaitFlag<0x2, PIPE_MTE2>(commFlagId)` — per-tile，flagId = tid（避开 3、5、[11-14]）
   - 通道 2（AIC→AIV，mm 完成）：`SetFlag<SYNC_MODE2, PIPE_FIX>(5)` / `WaitFlag<SYNC_MODE2, PIPE_MTE2>(5)` — 计数信号量（AIV 侧阻塞 MTE2 流水）
   - 通道 3（AIV→AIC，回压）：`SetFlag<SYNC_MODE2, PIPE_MTE3>(3)` / `WaitFlag<SYNC_MODE2, PIPE_M>(3)` — 计数信号量（AIC 侧阻塞 M 流水）

4. **AIV 时分复用**：每个 AIV 核先完成所有通信轮次，然后切换到 reduceSum 阶段。通信阶段和 reduceSum 阶段串行执行，但通过 workspace 多 buffer 实现 AIC mm 与 AIV reduceSum 的流水并行。

5. **mm 输出 BF16**：Blaze MX QuantMatmul 硬件已完成 FP8→BF16 反量化，mm 输出直接为 BF16，reduceSum 不需要 AscendDequant。

6. **reduceSum 无 AtomicAdd**：AIC mm 在 L0C 中跨 rank FP32 累加（for rank 循环 + mmadOp_ remoteRankCnt），最后一次 fixpipe 写 workspace。workspace 中已是最终结果 `Σ_i mm(A_i × B_i)`，reduceSum 仅做 DataCopyPad(workspace→UB→yGm) 搬运，**不需要 AtomicAdd**，yGm **不需要清零**。

7. **Blaze mm per-tile 动态输出地址**：v1 的 `gmC` 在循环外创建一次 + per-tile Slice；v2 在 while 循环内 per-tile 创建 `gmCWorkspace` Tensor（基于 `workspaceGmAddr_ + workSpaceOffset`），传入 mmadOp_。workspace slot 是 baseM×baseN 的整块，不需要 Slice。`Te::MakeTensor` 是常规函数，可在循环内任意位置调用。

8. **local 前置（可选优化）**：`localMatmul` 作为 tiling 可选参数，默认 0（不启用）。启用时映射 `MatmulMode::DEFERRED_SYNC`（per-tile 内 local A×B → L0C reset → wait_flag → remote A×B → L0C 累加 → 单次 fixpipe → workspace），reduceSum 仍 = DataCopyPad。**不再支持 v1 的 localMatmul=1（LOCAL+REMOTE+AtomicAdd to yGm）**，因为 v2 的 mm 输出到 workspace 而非 yGm。

### 1.5 内存管理(Buffer 规划)

#### UB 分配（AIV 侧）

| Buffer 名称 | 用途 | 大小 | 来源 |
|------------|------|------|------|
| commBuf (data) | data 通信 workspace | 512B (`COMM_WORKSPACE_SIZE`) | `collective_comm_context.h` |
| commScaleBuf | scale 通信 workspace | 512B (`COMM_WORKSPACE_SIZE`) | `collective_comm_context.h` |
| barrierBuf | 跨卡同步 flag | 32B (`BARRIER_FLAG_SIZE`) | `barrier_ubmem.h` |
| reduceSumBuf | reduceSum 数据暂存 | baseM × baseN × sizeof(bfloat16_t) | TBuf<VECCALC> |
| **合计** | | **1056B + baseM×baseN×2B** | 双通信对象 + barrier + reduceSum |

> UB 容量 248KB（DAV_3510）。reduceSumBuf 大小 = baseM × baseN × 2B，如 baseM=128, baseN=128 → 32KB，加上通信 1056B，远小于 248KB。

#### L0C 分配（AIC 侧）

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| L0C FP32 累加器 | mm 部分和累加 | baseM × baseN × 4B（FP32） |
| 约束 | ≤ 128KB（dbL0C ping-pong 半区）或 ≤ 256KB（单缓冲） | baseM=128, baseN=128 → 64KB ✅ |

#### workspace 分配（GM，算子入参）

| 项目 | 公式 | 说明 |
|------|------|------|
| workspace 总大小 | `PIPELINE_NUM × coreNum × baseM × baseN × sizeof(bfloat16_t)` | 4 × coreNum × tile_size × 2B |
| workSpaceOffset | `baseM × baseN × (coreIdx + (cubeNum % PIPELINE_NUM) × coreNum)` | 每个 AIC 核每步的偏移 |
| 分配方式 | Host 侧 `aclrtMalloc`，作为 kernel 入参传入 | 新增 GM_ADDR workspace 参数 |

**workspace 大小估算**:

| 项目 | 公式 | 示例 (coreNum=28, baseM=128, baseN=128) |
|------|------|----------------------------------------|
| 单 slot 大小 | coreNum × baseM × baseN × 2B | 28 × 128 × 128 × 2 = 917 KB |
| workspace 总大小 | 4 × 单 slot | 4 × 917 KB ≈ 3.58 MB |

#### Win 区（GM 共享区，与 v1 相同）

| 段 | 布局 | 大小 |
|----|------|------|
| data 段 | rankSize × (M/rankSize × K_local × sizeof(AType)) = M × K_local × sizeof(AType) | winOffset=0 |
| scale 段 | rankSize × (M/rankSize × scaleKLen_local × sizeof(fp8_e8m0_t)) | winOffset=M × K_local × sizeof(AType) |

#### GM 输入输出

| Tensor | 分布 | shape (per rank) | 说明 |
|--------|------|-------------------|------|
| A | K-split | [M, K_local] | 本 rank 的 A 段 |
| ScaleA | K-split | [M, CeilDiv(K_local,64)*2] | MXFP8 scale |
| B | 全量复制 | [K, N] (K=rankSize×K_local) | 按 rank 沿 K 拼接 |
| ScaleB | 全量复制 | [CeilDiv(K,64)*2, N] | 全量 scale |
| workspace | per-rank GM | [PIPELINE_NUM × coreNum × baseM × baseN] | **新增**，mm 中间输出 |
| C | M-split (输出) | [M/rankSize, N] | ReduceScatter 输出，需 host 清零 |

---

## 2. 架构设计

### 2.0 APACE 路线约束显式确认

| # | 约束 | 状态 | 验收条件 |
|---|------|------|---------|
| ① | 禁止 `__schedmode__(1)` 和 `[[bisheng::core_ratio(1,1)]]` | ✅ 确认 | 核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证 |
| ② | Matmul 走 Blaze 模板 | ✅ 确认 | 使用 `BlockMmad` + `BlockScheduler` + `MatmulWithScaleMx`，无 `AscendC::Matmul` |
| ③ | 禁止修改 `block/` 和 `tiling/` | ✅ 确认 | 只在 `kernel/apace_quant_matmul_reduce_scatter_v2/` 下创建文件 |
| ④ | 直调仅 UDMA | ✅ 确认 | HCCL windows/CCU 不支持直调；本算子使用 UDMA PUT 模式 |

### 2.1 通信模式选择

**选型**：AllToAll PUT + workspace 流水线（UDMA 引擎）

| 维度 | v1（串行） | v2（流水线） |
|------|-----------|-------------|
| 通信方向 | PUT（AIV→AIC） | PUT（AIV→AIC），相同 |
| 通信引擎 | UDMA | UDMA，相同 |
| 通信原语 | AllToAll PUT | AllToAll PUT，相同 |
| mm 输出目标 | yGm（AtomicAdd） | **workspace**（BF16，L0C 跨 rank 累加后 fixpipe） |
| reduceSum | 无（mm 直写 yGm） | **有**（workspace→UB→yGm，普通 DataCopyPad，无 AtomicAdd） |
| 流水线深度 | 无（串行） | PIPELINE_NUM=4 |
| ReduceScatter 实现 | AllToAll PUT + AtomicAdd（AIC fixpipe 级） | AllToAll PUT + workspace + reduceSum DataCopyPad（AIV 搬运级，无 AtomicAdd） |

### 2.2 Blaze QuantMatmul 组件选择

与 v1 完全相同（Blaze 组件不变，仅 mm 输出地址从 yGm 改为 workspace）：

| 组件 | 选择 | 来源 | 说明 |
|------|------|------|------|
| 路径 | blaze library | `third_party/blaze/gemm/` | MX 量化走 blaze library |
| DispatchPolicy | `MatmulWithScaleMx<0, false>` | `blaze/gemm/policy/dispatch_policy.h` | `0`=NONE_FULL_LOAD_MODE, `false`=非atomic |
| BlockMmad | `BlockMmad<DispatchPolicy, AType, NDExtLayoutPtn, BType, DNExtLayoutPtn, CType, LayoutC, BiasType, LayoutBias>` | `blaze/gemm/block/block_mmad_qbmm_mx.h` | TransA=false, TransB=true（NZ weight） |
| BlockScheduler | `BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, NDExtLayoutPtn, DNExtLayoutPtn, AType>` | `blaze/gemm/block/block_scheduler_qbmm.h` | M/N tile 调度 + 尾块拆分 |
| Kernel | `QuantMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler, CommPolicy>` | v2 kernel（复制自 v1 + 改造） | **改造**：mm 输出写 workspace，新增 flag 编排 |
| CommPolicy | `UdmaCommWaitPolicy` | 算子 impl 中定义 | `WaitTile(tileIdx)` → `CrossCoreWaitFlag<0x2, PIPE_MTE2>(commFlagId)` |
| Tiling 引擎 | `QuantMatmulTilingSwat<DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN>` | apace `tiling/quant_matmul_tiling_swat.h` | SWAT 流式 tiling |

### 2.3 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | M 轴（输出 C 的 M/rankSize 行）× N 轴 |
| ProblemShape | `<M/rankSize, N, ka, 1>` — 与 v1 相同 |
| 单核任务量 | M/rankSize × N 矩阵的 baseM × baseN tile |
| 使用的核数 | `usedCoreNum`（由 `QuantMatmulTilingSwat::GetTilingData` 推导） |
| 核配比 | AIC:AIV = 1:1（`KERNEL_TYPE_MIX_AIC_1_1`） |

### 2.4 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 248KB（DAV_3510） |
| 通信 UB | 1056B（双通信对象 512B×2 + barrier 32B） |
| reduceSum UB | baseM × baseN × sizeof(bfloat16_t)（如 128×128×2=32KB） |
| UB 分配方式 | 通信 UB 局部 `ubOffset` 线性累加；reduceSum UB 通过 `TBuf<VECCALC>` 分配 |

### 2.5 AIC/AIV 分工设计

```
┌─────────────────────── AIV (Vector 核, 时分复用) ──────────────────────────┐
│                                                                             │
│  Init():                                                                    │
│    提取 udmaCtx/ubmemCtx → InitBaseParams → UB 分配                         │
│    → teamBarrier_.Init → allToAllA_.Init<BARRIER_NONE>                      │
│    → allToAllScaleA_.Init(winOffset=rankSize×rankDataBytes)                 │
│    → reduceSumBuf 初始化 (TBuf<VECCALC>, baseM×baseN×sizeof(bfloat16_t))   │
│                                                                             │
│  Run():                                                                     │
│    Phase 1: RunAllToAll()           ← AllToAll PUT 通信                     │
│      for tid in 0..commTurn-1:                                              │
│        if GetBlockIdx() < rankSize:                                         │
│          allToAllScaleA_.Commit()                                           │
│          allToAllA_.Commit()                                                │
│          allToAllA_.Wait<BARRIER_DEVICE>()                                  │
│        SyncAll<true>()                                                      │
│        CrossCoreSetFlag<0x2, PIPE_MTE3>(commFlagId(tid))  → AIC             │
│      allToAllScaleA_.Finalize()                                             │
│      allToAllA_.Finalize()                                                  │
│                                                                             │
  │    Phase 2: RunReduceSum()          ← reduceSum (时分复用切换)              │
  │      // ★ AIV 复用 BlockScheduler（方案 b，见 §2.16）                     │
  │      bsAiv = BlockScheduler(problemShape, schParams)  ← 与 AIC 相同输入   │
  │      if 尾块条件: bsAiv.UpdateTailTile(mTailTile, nTailTile)  ← 与 AIC 相同│
  │      while bsAiv.GetTileIdx(blockIdx):               ← 与 AIC 同一枚举序  │
  │        singleShape = bsAiv.GetBlockShape<MX,MX,weightNz>(blockIdx)        │
  │        if singleShape 为零: break                    ← 对齐 AIC early-return│
  │        bsAiv.GetTileCoord(blockIdx, mPos, nPos)      ← yGm 位置推导       │
  │        CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(5)  ← 等 AIC mm 完成       │
  │        DataCopyPad(workspace[workSpaceOffset] → reduceSumBuf)  (2D 搬运)   │
  │        DataCopyPad(reduceSumBuf → yGm[mPos*N + nPos])  (2D 搬运, 无 AtomicAdd) │
  │        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(3)  → AIC (回压)             │
  │        cubeNum++                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↕ CrossCore Flags (3 通道)
┌─────────────────────── AIC (Cube 核, mm 计算) ──────────────────────────────┐
│                                                                             │
│  Init(params):                                                              │
│    解析 isBias → ResetGmAddr（workspaceGmAddr_ 绑定为 mmadParams.cGmAddr）  │
│    isAtomicAdd_ = false（v2 不使用 AtomicAdd）                               │
│                                                                             │
│  Run():                                                                     │
│    if localMatmul == 1:  (可选优化, 默认 0, 映射 DEFERRED_SYNC)             │
│      RunMatmul()   ← DEFERRED_SYNC: per-tile local→L0C→wait→remote→L0C累加  │
│                     → 单次 fixpipe → workspace                              │
│    else:                                                                    │
│      RunMatmul()   ← REMOTE: mm → L0C 跨 rank 累加 → fixpipe → workspace    │
│                                                                             │
│  RunMatmul() / ProcessSingleBatch():                                        │
│    cubeNum = 0                                                              │
│    while bs.GetTileIdx(blockIdx):                                           │
│      if cubeNum >= PIPELINE_NUM:                                            │
│        CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(3)  ← 回压                     │
│      dependTileIdx = CalcDependTileIdx(mPos+baseM-1, headTileSize)          │
│      while readyTileIdx < dependTileIdx:                                    │
│        readyTileIdx++                                                       │
│        CrossCoreWaitFlag<0x2, PIPE_MTE2>(commFlagId(readyTileIdx))          │
│      // ★ v2 改造点：per-tile 创建 workspace Tensor（v1 是循环外创建+Slice）│
│      workSpaceOffset = baseM*baseN*(coreIdx + (cubeNum%4)*coreNum)          │
│      auto gmCWorkspace = Te::MakeTensor(                                    │
│          Te::MakeMemPtr<Te::Location::GM>(                                  │
│              workspaceGmAddr_ + workSpaceOffset * sizeof(CType)),           │
│          layoutCTile)  // baseM×baseN 整块, 不需要 Slice                    │
│      // mm: A[Win] × B → L0C FP32 跨 rank 累加 → fixpipe → workspace[BF16] │
│      mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB,              │
│              gmBlockBias, gmCWorkspace, singleShape, remoteRankCnt)         │
│      CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(5)  → AIV (mm 完成)             │
│      cubeNum++                                                              │
│    // drain 剩余回压 flag                                                    │
│    for idx in 0..Min(cubeNum, PIPELINE_NUM)-1:                              │
│      CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(3)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**同步点表**：

| 同步点 | 方向 | API | flagId | 含义 |
|--------|------|-----|--------|------|
| AIV→AIC 通信就绪 | AIV→AIC | `SetFlag<0x2, PIPE_MTE3>(commFlagId)` → `WaitFlag<0x2, PIPE_MTE2>(commFlagId)` | tid（避开 3/5/[11-14]） | 通信 tile tid 数据就绪 |
| AIC→AIV mm 完成 | AIC→AIV | `SetFlag<SYNC_MODE2, PIPE_FIX>(5)` → `WaitFlag<SYNC_MODE2, PIPE_MTE2>(5)` | 5（常量） | mm 输出已写入 workspace（AIV 侧阻塞 MTE2 流水） |
| AIV→AIC 回压 | AIV→AIC | `SetFlag<SYNC_MODE2, PIPE_MTE3>(3)` → `WaitFlag<SYNC_MODE2, PIPE_M>(3)` | 3（常量） | workspace 槽位可复用（AIC 侧阻塞 M 流水） |
| AIV 块间同步 | AIV↔AIV | `SyncAll<true>()` | 14 | WriteNbi 对端可见性 |
| 跨卡同步 | AIV↔AIV | `TeamBarrier.CrossDevice()` | GM flag | 每轮通信后跨卡同步 |

### 2.6 通算流水编排

**flag 编排模式（v2 三通道，950PR 模板参数生效）**：

```
AIV (RunAllToAll + RunReduceSum)               AIC (RunMatmul)
─────────────────────────────────               ──────────────────
Phase 1: 通信
round tid:
  Commit(scale A[tid])
  Commit(data A[tid])
  Wait<BARRIER_DEVICE>()
  SyncAll<true>()
  SetFlag<0x2, PIPE_MTE3>(commFlagId) ──→      if cubeNum >= 4: WaitFlag<SYNC_MODE2, PIPE_M>(3) [回压]
                                                 WaitFlag<0x2, PIPE_MTE2>(commFlagId) [通信完成]
                                                 ★ per-tile 创建 gmCWorkspace Tensor
                                                 mmadOp_(A,B → gmCWorkspace) [L0C 跨 rank 累加 → fixpipe → workspace]
                                                 SetFlag<SYNC_MODE2, PIPE_FIX>(5) ──→ [mm 完成]
                                                 cubeNum++

Phase 2: reduceSum (时分复用切换)
  WaitFlag<SYNC_MODE2, PIPE_MTE2>(5) [mm 完成] ←──────────────────────
  DataCopyPad(workspace → UB)  (1D 搬运)
  DataCopyPad(UB → yGm)  (2D 搬运, 无 AtomicAdd)
  SetFlag<SYNC_MODE2, PIPE_MTE3>(3) ──→        [回压: workspace 可复用]

PostCompute (AIC):
  drain Min(cubeNum, 4) 个 WaitFlag<SYNC_MODE2, PIPE_M>(3)
```

**flagId 分配策略**：

| 通道 | flagId | 类型 | 说明 |
|------|--------|------|------|
| 通信就绪 | tid（避开 3、5、[11-14]） | per-tile | `commFlagId = (tid < 3) ? tid : (tid < 5) ? tid+1 : tid+2`，截断低 4bit |
| mm 完成 | 5（常量） | 计数信号量 | AIC 每完成一个 mm tile Set 一次；AIV 每做一次 reduceSum Wait 一次 |
| 回压 | 3（常量） | 计数信号量 | AIV 每完成一次 reduceSum Set 一次；AIC 当 cubeNum ≥ 4 时 Wait 一次 |

> **flagId 冲突规避**：通信就绪 flagId 使用 `commFlagId` 映射函数避开 3、5 和 [11-14]（SyncAll 硬同步内部占用区间）。可用 flagId = {0,1,2,4,6,7,8,9,10,15} 共 10 个。当 tile 总数 > 10 时，`commFlagId` 会命中 [11-14] 或绕回，需在 tiling 推导时限制 `headTileCnt + tailTileCnt <= 10`。
>
> **SyncAll 冲突风险**：apace skill `fusion.md` §3.3 和 `communication.md` §4.2 明确说明 SyncAll 硬同步内部占用 flagId [11-14]，官方不建议同时使用 CrossCoreSetFlag 与 SyncAll 硬同步。v2 每轮通信后调用 `SyncAll<true>()`，同时使用 `CrossCoreSetFlag`，因此必须避开 [11-14]。
>
> **950PR 模板参数生效**：`CrossCoreWaitFlag` 在 950PR 上模板参数 modeId 和 pipe 生效（`api-crosscore-sync.md` §4）。模式 0/1/2 不支持 PIPE_S/PIPE_ALL。v2 使用 modeId=2（SYNC_MODE2）+ PIPE_MTE2（AIV 侧阻塞数据搬入）/ PIPE_M（AIC 侧阻塞 mm 计算），与 SetFlag 的 modeId=2 配对。

### 2.7 tiling_data 结构体设计

```cpp
// apace_quant_matmul_reduce_scatter_v2_tiling_data.h

constexpr uint32_t PIPELINE_NUM_V2 = 4;
constexpr uint64_t SYNC_AIV_AIC_FLAG_V2 = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG_V2 = 5;
constexpr uint64_t SYNC_MODE2_V2 = 2;

#pragma pack(push, 8)
struct alignas(8) ApaceQuantMatmulReduceScatterV2TilingData {
    CommTilingData commTilingData;              // data 通道通信切分
    CommTilingData scaleCommTilingData;         // scale 通道通信切分
    QuantMatmulTilingData tileQbmmTilingData;   // 计算 tiling (m=M/rankSize)
    uint32_t localMatmul;                        // 0:默认不启用, 1:DEFERRED_SYNC (可选优化)
    uint32_t pipelineNum;                        // PIPELINE_NUM=4
    uint32_t coreNum;                            // AIC 核数
    uint32_t ubCalSize;                          // reduceSum UB 计算 size
    uint32_t totalMmTiles;                       // [DEPRECATED §2.16] AIV 不再使用，host 填 0；AIV 循环由 BlockScheduler 枚举隐式终止
    uint64_t workspaceSize;                      // workspace 总大小
};
#pragma pack(pop)

struct CommContext {
    Apace::AivComm::CommUdmaContext udmaCtx;
    Apace::AivComm::CommUbmemContext ubmemCtx;
};
```

**与 v1 tiling 结构体对比**：

| 字段 | v1 | v2 | 说明 |
|------|----|----|------|
| commTilingData | ✅ | ✅ | 相同 |
| scaleCommTilingData | ✅ | ✅ | 相同 |
| tileQbmmTilingData | ✅ | ✅ | 相同 |
| localMatmul | ✅ (默认 1) | ✅ (默认 **0**) | v2 默认不启用 |
| pipelineNum | ❌ | ✅ (=4) | **新增** |
| coreNum | ❌ | ✅ | **新增** |
| ubCalSize | ❌ | ✅ | **新增** (reduceSum UB) |
| totalMmTiles | ❌ | ✅ (DEPRECATED §2.16) | **新增后废弃**（AIV 循环由 BlockScheduler 枚举隐式终止，host 填 0） |
| workspaceSize | ❌ | ✅ | **新增** |

### 2.8 Host 侧 tiling 推导

```
headMSize 推导 (与 v1 相同):
  nTile      = CeilDiv(n, baseN)
  headMSize  = CeilDiv(usedCoreNum, nTile) * baseM
  mLocal     = M / rankSize
  headTileCnt = mLocal / headMSize
  tailMSize   = mLocal % headMSize

workspace 大小推导:
  coreNum = usedCoreNum  (AIC 核数)
  workspaceSize = PIPELINE_NUM * coreNum * baseM * baseN * sizeof(bfloat16_t)

GetTilingData 调用 (与 v1 相同):
  QuantMatmulTilingSwat<DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN> tilingEngine;
  uint32_t ka = K / rankSize;
  tilingEngine.GetTilingData(mLocal, N, ka, tilingData.tileQbmmTilingData);

新增字段填充:
  tilingData.pipelineNum = 4
  tilingData.coreNum = tilingData.tileQbmmTilingData.usedCoreNum
  tilingData.ubCalSize = baseM * baseN  (reduceSum 单 tile 大小)
  tilingData.workspaceSize = 4 * coreNum * baseM * baseN * sizeof(bfloat16_t)
  tilingData.localMatmul = 0  (默认不启用)
  tilingData.totalMmTiles = 0  (DEPRECATED §2.16：AIV reduceSum 循环由 BlockScheduler 枚举隐式终止，
                                不再依赖该字段；保留字段仅为结构体兼容)

Host 侧 workspace 分配:
  aclrtMalloc(&deviceWorkspace, workspaceSize)

yGm 清零: 不需要
  reduceSum 使用普通 DataCopyPad（非 AtomicAdd）直接覆盖 yGm 旧值，无需清零。
```

**输入校验**（与 v1 相同 + 新增）:
- `K % rankSize == 0`
- `M % rankSize == 0`
- `ka % 64 == 0`
- **新增**：`headTileCnt + tailTileCnt <= 10`（flagId 冲突规避：避开 3、5、[11-14]，可用 flagId = {0,1,2,4,6,7,8,9,10,15} 共 10 个）

### 2.9 入口函数设计

**4 个 dtype 变体入口**（与 v1 结构相同，新增 workspace 参数）：

| 入口函数名 | AType | BType | CType |
|------------|-------|-------|-------|
| `ApaceQuantMatmulReduceScatterV2KernelE4M3E4M3_Udma` | fp8_e4m3fn_t | fp8_e4m3fn_t | bfloat16_t |
| `ApaceQuantMatmulReduceScatterV2KernelE5M2E5M2_Udma` | fp8_e5m2_t | fp8_e5m2_t | bfloat16_t |
| `ApaceQuantMatmulReduceScatterV2KernelE4M3E5M2_Udma` | fp8_e4m3fn_t | fp8_e5m2_t | bfloat16_t |
| `ApaceQuantMatmulReduceScatterV2KernelE5M2E4M3_Udma` | fp8_e5m2_t | fp8_e4m3fn_t | bfloat16_t |

**入口签名**（UDMA 模式，8 参数，新增 workspace）：

```cpp
__global__ __aicore__ void ApaceQuantMatmulReduceScatterV2KernelE4M3E4M3_Udma(
    __gm__ CommContext *hcommCtx,        // 1. 通信上下文
    GM_ADDR aGM, GM_ADDR scaleAGM,       // 2-3. A 矩阵与 Scale
    GM_ADDR bGM, GM_ADDR scaleBGM,       // 4-5. B 矩阵与 Scale
    GM_ADDR cGM,                         // 6. C 输出 (M/rankSize × N)
    GM_ADDR workspaceGM,                 // 7. workspace (新增)
    ApaceQuantMatmulReduceScatterV2TilingData tilingData) // 8. tiling
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::ApaceQuantMatmulReduceScatterV2UdmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, workspaceGM, &tilingData);
    impl.Run();
}
```

### 2.10 多卡通信拓扑（4 卡 ReduceScatter）

与 v1 完全相同（通信语义不变，AllToAll PUT 发送 A 的 M/rankSize 段给各 target rank）：

```
4 卡 ReduceScatter 数据流 (与 v1 相同):

rank i: A_i[M×Ka] ──PUT──→ 各 rank r 的 Win 区 (A_i 的第 r 个 M/4 段)

rank r Win 区:
  [A_0[r*M/4:(r+1)*M/4, :]]  ← 来自 rank 0
  [A_1[r*M/4:(r+1)*M/4, :]]  ← 来自 rank 1
  [A_2[r*M/4:(r+1)*M/4, :]]  ← 来自 rank 2
  [A_3[r*M/4:(r+1)*M/4, :]]  ← 来自 rank 3

rank r 计算:
  AIC: mm(A_i[my_seg] × B_i → L0C FP32 跨 rank 累加 → fixpipe → workspace)  for each i
  AIV: reduceSum(workspace → yGm[my_seg])  普通 DataCopyPad 搬运（无 AtomicAdd）
  C_r[M/4 × N] = DataCopyPad(workspace[my_M_seg])
  其中 workspace = Σ_i mm(A_i[my_seg] × B_i)（AIC L0C 累加后 fixpipe 写入的最终结果）
```

### 2.11 AIC/AIV 核分工总结

| 核类型 | 阶段 | 任务 | 说明 |
|--------|------|------|------|
| AIC | 全程 | mm 计算 | Blaze QuantMatmul: A[Win] × B → workspace[BF16] |
| AIV | Phase 1 | AllToAll PUT 通信 | 发送 A 数据到各 rank Win 区 |
| AIV | Phase 2 | reduceSum | workspace→UB→yGm（DataCopyPad，无 AtomicAdd） |
| AIC | PostCompute | drain 回压 flag | 等待剩余 workspace 槽位释放 |

**时分复用设计**：
- 每个 AIV 核先完成所有通信轮次（Phase 1），然后切换到 reduceSum（Phase 2）
- AIC 在 AIV 通信期间持续做 mm，mm 输出写入 workspace（4 槽位轮转）
- AIV reduceSum 从 workspace 读取（落后 AIC 最多 PIPELINE_NUM=4 个 tile）
- 回压 flag 确保 AIC 不覆盖 AIV 尚未消费的 workspace 槽位

**流水线效率分析**：
- AIC 阻塞条件：`cubeNum >= PIPELINE_NUM=4` 时 `WaitFlag<SYNC_MODE2, PIPE_M>(3)` 回压
- AIC 阻塞持续时间 = AIV 完成剩余通信轮次的时间 + AIV 切换到 reduceSum 的时间
- 最坏情况：commTurn 较大时，AIC 在前 4 个 tile 后阻塞，直到 AIV 完成全部 commTurn 轮通信
- 缓解因素：AIV 通信每轮耗时较短（UDMA PUT + SyncAll），且 AIC mm 单 tile 耗时通常 ≥ 通信单轮耗时（Blaze MX QuantMatmul 的 L0C 跨 rank 累加计算量较大）
- 通算并行度：AIC mm 与 AIV 通信在 Phase 1 期间并行（AIC 领先最多 4 tile）；AIC mm 与 AIV reduceSum 在 Phase 2 期间并行（AIC 领先最多 4 tile）
- 回退路径：如果性能不达标，可回退到 v1 串行模式（localMatmul=1 + AtomicAdd 直写 yGm，v1 开发历史数据 143us）

### 2.12 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| dtype: E4M3×E4M3 | 入口 `E4M3E4M3_Udma`，默认 dtype 组合 |
| dtype: E5M2×E5M2 | 入口 `E5M2E5M2_Udma` |
| dtype: E4M3×E5M2 | 入口 `E4M3E5M2_Udma` |
| dtype: E5M2×E4M3 | 入口 `E5M2E4M3_Udma` |
| 大 shape (M×K×N) | 多核切分 + 通信 tile 多轮 + workspace 多 slot |
| 小 shape | 单核或少量核 + 通信 tile 少轮 |
| M % rankSize ≠ 0 | 输入校验拒绝 |
| K % rankSize ≠ 0 | 输入校验拒绝 |
| ka % 64 ≠ 0 | 输入校验拒绝（MXFP8 scale 对齐） |
| 尾块 | `splitAxisTailSize` + `splitAxisTailCnt` 处理 |
| 通信 tile > 10 | 输入校验拒绝（flagId 冲突规避：避开 3、5、[11-14]） |
| localMatmul=0 (默认) | 不做 LOCAL 前置，所有 mm 走 REMOTE 流水线（L0C 跨 rank 累加 → fixpipe → workspace） |
| localMatmul=1 (可选) | 映射 DEFERRED_SYNC 模式：per-tile local→L0C→wait→remote→L0C 累加→单次 fixpipe→workspace（reduceSum 仍 = DataCopyPad） |

### 2.13 精度策略

| 维度 | 策略 |
|------|------|
| 输出 dtype | bfloat16_t |
| mm 输出精度 | BF16（Blaze MX QuantMatmul 硬件 FP8→BF16 反量化） |
| L0C 累加精度 | FP32（mmad 内部 FP32 跨 rank 累加，fixpipe 截断为 BF16 写 workspace） |
| reduceSum 精度 | BF16 普通搬运（workspace BF16 → UB → yGm BF16，无 AtomicAdd） |
| 精度损失来源 | mm fixpipe FP32→BF16 截断（仅此一处，无 AtomicAdd read-modify-write） |
| 预期 max_rel_diff | ≤ 0.79%（rankSize=4，FP8→BF16，与 v1 开发历史数据一致或更优——v2 消除了 AtomicAdd 精度风险） |
| 容差标准 | rtol=atol=1e-2（与 v1 ST 一致） |
| yGm 清零 | 不需要（reduceSum 使用普通 DataCopyPad 直接覆盖 yGm 旧值，非 AtomicAdd） |

### 2.14 与 v1 的关键差异对比表

| 维度 | v1（串行模式） | v2（流水线模式） |
|------|---------------|-----------------|
| 架构模式 | 2 级：AIV 通信 + AIC mm 直写 yGm | 3 级：AIC mm→workspace + AIV 通信 + AIV reduceSum |
| 流水线深度 | 无（串行） | PIPELINE_NUM=4（多 buffer workspace 轮转） |
| mm 输出目标 | yGm（AtomicAdd 直写） | workspace（BF16，无 AtomicAdd） |
| 跨 rank 累加位置 | L0C FP32 累加 → fixpipe AtomicAdd to yGm | L0C FP32 累加 → fixpipe to workspace（reduceSum 仅搬运） |
| reduceSum | 无 | DataCopyPad(workspace→UB) + DataCopyPad(UB→yGm)（无 AtomicAdd） |
| workspace | 无 | 新增算子入参，PIPELINE_NUM×coreNum×baseM×baseN×BF16 |
| AIV 分工 | 仅通信 | 时分复用：先通信后 reduceSum |
| CrossCore flag 通道 | 1 通道（通信就绪） | 3 通道（通信就绪 + mm 完成 + 回压） |
| CrossCoreWaitFlag 模板参数 | 不指定（910b 不生效） | 显式指定 modeId=2 + pipe（950PR 生效） |
| yGm 清零 | 不需要（LOCAL 首次写入） | 不需要（reduceSum 普通搬运覆盖） |
| localMatmul 默认值 | 1（LOCAL+REMOTE+AtomicAdd to yGm） | 0（不启用，纯 REMOTE 流水线） |
| localMatmul=1 语义 | LOCAL+REMOTE+AtomicAdd to yGm | DEFERRED_SYNC（L0C 跨 rank 累加→单次 fixpipe→workspace） |
| flagId 约束 | ≤32（waitedMask uint32_t） | ≤10（避开 3、5、[11-14]） |
| tiling 新增字段 | — | pipelineNum, coreNum, ubCalSize, totalMmTiles, workspaceSize |
| 入口参数 | 7 参数 | 8 参数（新增 workspaceGM） |
| Blaze 组件 | 不变 | 不变（仅 mm 输出地址改 workspace + per-tile 创建 gmCWorkspace） |
| 通信组件 | 不变 | 不变 |
| 精度容差 | rtol=atol=1e-2 | rtol=atol=1e-2（相同，但精度风险更低——无 AtomicAdd） |

### 2.15 文件清单与 REUSE/MODIFY 标记

#### [MODIFY] 算子层文件（3 个 kernel 头文件）

| 文件 | 职责 | 参考来源 |
|------|------|---------|
| `kernel/apace_quant_matmul_reduce_scatter_v2/apace_quant_matmul_reduce_scatter_v2_udma_impl.h` | Impl 类（Init/Run/RunAllToAll/RunReduceSum/RunMatmul/SetupParams）+ UdmaCommWaitPolicy + reduceSum 逻辑 | 复制自 v1 impl + ops-nn 流水线模式 |
| `kernel/apace_quant_matmul_reduce_scatter_v2/quant_matmul_mx_kernel.h` | Blaze matmul 调度（QuantMatmulMxKernel，改造 mm 输出到 workspace + flag 编排） | 复制自 v1 kernel + ops-nn flag 模式 |
| `kernel/apace_quant_matmul_reduce_scatter_v2/apace_quant_matmul_reduce_scatter_v2_tiling_data.h` | tiling 结构体 + CommContext + 常量定义 | 复制自 v1 + 新增字段 |

#### [MODIFY] ST 测试工程

| 文件 | 职责 |
|------|------|
| `st/apace_quant_matmul_reduce_scatter_v2/CMakeLists.txt` | 构建配置 |
| `st/apace_quant_matmul_reduce_scatter_v2/run.sh` | ST 驱动脚本 |
| `st/apace_quant_matmul_reduce_scatter_v2/cases.csv` | 测试用例 |
| `st/apace_quant_matmul_reduce_scatter_v2/src/main.cpp` | Host 入口（tiling 推导 + CommContext 建链 + workspace 分配 + yGm 清零 + launch） |
| `st/apace_quant_matmul_reduce_scatter_v2/src/kernel_launcher.h` | 4 个 `__global__` 入口函数 |
| `st/apace_quant_matmul_reduce_scatter_v2/scripts/gen_data.py` | 数据生成 |
| `st/apace_quant_matmul_reduce_scatter_v2/scripts/verify_result.py` | 精度验证（rtol=atol=1e-2） |
| `st/utils/root_info_exchanger.h` | TCP 交换 HcclRootInfo |

#### [REUSE] 共享层文件（禁止修改）

| 目录 | 来源 |
|------|------|
| `block/aiv_comm/` | `collective_comm_api.h`, `all_to_all/all_to_all_udma_put.h`, `barrier/barrier_ubmem.h` 等 |
| `block/blaze_ext/` | `gemm/block/qmm_mx_block_mmad_fragment.h` 等 |
| `tiling/` | `comm_tiling_data.h`, `quant_matmul_tiling_*.h` |
| `basic/fragment_tensor/` | `fragment_tensor.h` |
| `utils/` | `comm_channel_builder.h`, `constant.h` |

### 2.16 design_issue 解决记录（2026-08-12）：AIV reduceSum 位置推导与 totalMmTiles

> 背景：Developer 实现中发现 2 个 design_issue（PLAN.md §5）——(1) AIV 使用 BlockScheduler 死锁，
> yGm 位置推导方式未定；(2) `totalMmTiles` 公式与 BlockScheduler 实际调度 tile 数不匹配。
> 本节给出正式设计决策，替代 §1.4/§2.5 中旧的线性循环方案。

#### Issue 1 决策：方案 (b) —— AIV 复用 BlockScheduler（与 AIC 构造参数、调用序列完全一致）

**方案评估**：

| 方案 | 评估 | 结论 |
|------|------|------|
| (a) workspace metadata（AIC 写 mPos/nPos 到 workspace，AIV 读取） | **不可行**。AIC 侧无 UB，metadata 只能标量写 GM；而 `CrossCoreSetFlag` 模式 0/1/2 不支持 PIPE_S（CrossCoreSetFlag_ISASI.md §pipe 说明），标量 GM 写无法被任何可 flag 的 pipe fence，AIV 读取可见性无保证。强行加 MTE3 通道需 AIC 侧 UB DataCopyPad，不可行 | ❌ 拒绝 |
| (b) AIV 复用 BlockScheduler（AIV 也跑 BlockScheduler 推导 mPos/nPos，与 AIC 步调对齐） | **可行**（论证见下）。BlockScheduler 是纯标量 int64 运算，AIV 可执行；ops-nn `quant_matmul_reduce_sum` 正是该模式（AIC/AIV 执行同一套 MNConfig 推导循环） | ✅ **采纳** |
| (c) 线性公式推导（cubeNum → mPos/nPos 封闭公式） | **不可行**。BlockScheduler 枚举含 mainWindow 蛇形 swizzle（`if (rowIdx & 1) nIdx = nBlockNums-1-nIdx`）+ 尾块拆分 + 零形状早退，封闭公式复刻风险高；Developer 当前实现的 N-inner 假设（`mIdx = cubeNum / nTileNum`）与调度器实际枚举序（无尾块场景为 M-inner：`mIdx = t % mBlockNums`）相反，必然错位 | ❌ 拒绝 |

**方案 (b) 可行性论证（官方文档级证据）**：

`BlockSchedulerQuantBatchMatmulV3` 内部仅依赖两个硬件查询：
- `blockIdx_ = AscendC::GetBlockIdx() / AscendC::GetTaskRation()`
- `coreNums_ = AscendC::GetBlockNum()`

在 `KERNEL_TYPE_MIX_AIC_1_1`（950PR）下，三个查询在 AIC 与配对 AIV 上**完全相同**：

| 查询 | 文档证据 | 1:1 配比下行为 |
|------|---------|---------------|
| `GetBlockIdx()` | GetBlockIdx.md：「当 AIC 与 AIV 比例为 1:1 时，AIC 上取值范围为 [0, numBlocks)，AIV 上取值范围为 [0, numBlocks)」 | 两侧一致，AIV i 与 AIC i 同属 AI Core i |
| `GetBlockNum()` | GetBlockNum.md：「返回值与 <<<>>> 调用时配置的 numBlocks 一致」 | 两侧均返回 numBlocks |
| `GetTaskRatio()` | GetTaskRatio.md 表 1：KERNEL_TYPE_MIX_AIC_1_1 下 AIV=1、AIC=1 | 两侧均为 1 |

因此，以**相同** `problemShape{m,n,k,1}` 和 `schParams`（同一份 `tileQbmmTilingData` 经同一 `SetupParams` 推导）构造的 BlockScheduler，在 AIC 与配对 AIV 上枚举出**完全一致**的 tile 序列（含 mainWindow 蛇形序、尾块拆分、roundNums 调整），`GetTileCoord` 推导的 (mPos, nPos) 与 `GetBlockShape` 返回的 (curBaseM, curBaseN) 逐 tile 对齐。ops-nn 先例：`QbmmRSProcess::Init` 中 `coreIdx = GetBlockIdx() / GetTaskRation()`（AIV 侧）与 AIC 侧 `coreIdx` 一致，`VectorCompute` 的 workspace 偏移即按此推导（`quant_matmul_reduce_sum.h` / `quant_matmul_reduce_sum_mixcore.h`）。

**死锁根因修正**：PLAN.md §4 中「MIX_AIC_1_1 核配比下 AIV/AIC 的 GetBlockIdx() 或 GetBlockNum() 行为差异」的根因假设**不成立**（上述文档证明 1:1 下三者一致）。死锁真实根因为 AIV 侧复刻循环时**调用序列与 AIC 不一致**（如未调用 `UpdateTailTile`、`GetBlockShape` 模板参数不同、零形状时用 `continue` 而非对齐 AIC 的 `return` 早退），导致 AIV 迭代次数 ≠ AIC SetFlag(5) 次数，flag 配对失衡死锁。

**AIV 侧精确复刻要求清单**（必须与 AIC `ProcessSingleBatch` 逐项一致）：

| # | 复刻项 | AIC 侧行为 | AIV 侧要求 |
|---|--------|-----------|-----------|
| 1 | 调度器构造输入 | `BlockScheduler bs(params.problemShape, params.schParams)` | 同一 `problemShape{m,n,k,1}` + 同一 `schParams{baseM,baseN,mTailTile,nTailTile,mBaseTailSplitCnt,nBaseTailSplitCnt,mTailMain,nTailMain}`（复用同一 SetupParams 产物） |
| 2 | UpdateTailTile 判定 | `if ((bs.GetEndBlockIdx()+1) + restBatch*bs.GetTotalCnt()) * mTailTile * nTailTile <= GetBlockNum()` → `bs.UpdateTailTile(mTailTile, nTailTile)`（restBatch=0, isTailRound=true） | 逐字复刻（GetEndBlockIdx/GetTotalCnt/GetBlockNum 两侧一致，判定结果相同） |
| 3 | 循环内调用顺序 | `GetTileIdx` → `GetBlockShape<MX_PERGROUP_MODE, MX_PERGROUP_MODE, weightNz>` → 零形状检查 → `GetTileCoord` | 逐字复刻（`GetBlockShape` 内部更新 `mSplitAddrOffset_/nSplitAddrOffset_` 状态，顺序不可变） |
| 4 | 零形状早退 | `if (singleShape.M<=0 \|\| singleShape.N<=0) return;`（终止本核后续所有 tile） | 对应 `break` 退出 reduceSum 循环（终止本 AIV 后续所有 WaitFlag(5)） |
| 5 | cubeNum 计数 | 仅在实际处理的 tile 上自增（SetFlag(5) 之后） | 仅在实际处理的 tile 上自增（SetFlag(3) 之后），用于 slot 索引 |
| 6 | coreIdx | `GetBlockIdx()`（AIC，1:1 下 = AI Core 索引） | `GetBlockIdx()`（AIV，1:1 下 = 同一 AI Core 索引） |

**flag 配对平衡证明**（复刻后）：设本核实际处理 tile 数 = T。
- 通道 2（flagId=5）：AIC Set T 次 ⇔ AIV Wait T 次（循环构造一致，含零形状早退截断）✅
- 通道 3（flagId=3）：AIV Set T 次；AIC Wait 次数 = 主循环回压 `max(T-4, 0)` + PostCompute drain `min(T, 4)` = T ✅
- 计数器范围（0-15）：AIC 领先 AIV ≤ PIPELINE_NUM=4 个 tile，flagId=5 未消费计数 ≤ 4，不溢出 ✅

**yGm 位置推导（明确公式）**：

```
每 tile（由 bsAiv 枚举）：
  bsAiv.GetTileCoord(blockIdx, mPos, nPos)     // 元素坐标（含尾块拆分偏移）
  curBaseM = singleShape.M; curBaseN = singleShape.N
  outOffset = mPos * N + nPos                  // yGm 行主序，行距 = N
  DataCopyPad(yGm[outOffset], reduceSumBuf, cpOut)
```

**多核并行**：取消 Developer 临时实现中的 `coreIdx == 0` 限制——每个 AIV 核处理**本核**的 tile 序列（与配对 AIC 一致），不同核的 tile 在 yGm 上区域不相交，无写冲突，恢复多核并行 reduceSum。

**DataCopyPad 参数修正**（Developer 当前实现的 1D 搬入 + 对齐 stride 搬出不匹配，会错位）：

| 搬运 | 参数 | 说明 |
|------|------|------|
| workspace→UB（GM→UB，2D） | `cpIn{blockCount=curBaseM, blockLen=curBaseN*2B, srcStride=0, dstStride=Ceil(curBaseN*2,64)*2 - Ceil(curBaseN*2,32)}` | slot 内数据为 fixpipe 按 `MakeLayoutC(curBaseM, curBaseN)` 写入的紧密排列（行距=curBaseN），故 srcStride=0；UB 内行距按 32B 对齐，dstStride 公式同 ops-nn `DataCopyPad2D`（单位 32B 块） |
| UB→yGm（UB→GM，2D） | `cpOut{blockCount=curBaseM, blockLen=curBaseN*2B, srcStride=(AlignUp(curBaseN,16)-curBaseN)*2/32, dstStride=(N-curBaseN)*2B}` | 与 GM→UB 建立的 UB 行对齐匹配（同 ops-nn `DataCopyOut`） |

> UB 预算不变：`reduceSumBuf` 大小 = baseM × AlignUp(baseN,16) × 2B（baseN 通常 16 对齐，与原 baseM×baseN×2B 相同），远小于 248KB。

#### Issue 2 决策：totalMmTiles 废除（不需要任何计算公式）

**旧公式错误分析**：`totalMmTiles = tileCnt × CeilDiv(N, baseN)`（及更早的 `headTileCnt + tailTileCnt`）存在双重错误：

1. **粒度错误**：`tileCnt`（= headTileCnt + tailTileCnt）是**通信 tile 数**（按 `headMSize` 粒度沿 M 切分），不是 **mm tile 数**（按 `baseM × baseN` 粒度）；两者仅在 `headMSize == baseM` 时偶然相等。
2. **单位错误**：该公式计算的是**全局 tile 数**，而 AIV 需要的是**每核 tile 数**（= 本核 AIC SetFlag(5) 次数）。

**正确语义**：每核 mm tile 数 = 本核 BlockScheduler 实际迭代处理的 tile 数，即 `roundNums_`（经 `UpdateTailTile` 按核调整）再经**零形状早退截断**。该值依赖 `GetBlockNum()` 运行时查询、尾块拆分参数与早退逻辑，host 侧无法可靠预算（host 预算本质上是复刻整个调度器，风险同方案 c）。

**决策**：方案 (b) 下 AIV 循环由 `bsAiv.GetTileIdx()` 返回值隐式终止，与 AIC **构造一致**，不需要 `totalMmTiles`：
- tiling 字段 `totalMmTiles` 标记 **DEPRECATED**（保留字段仅为结构体兼容，host 填 0，kernel 不消费）；
- AIV reduceSum 循环上界 = `bsAiv.GetTileIdx() == false`（或零形状 break），与 AIC 的 SetFlag(5) 次数逐项配对（平衡证明见上表）；
- 如需运行时每核 tile 数用于调试，可在 kernel 内对 cubeNum 终值 printf/写 GM，不由 host 预算。

### 2.17 工程目录结构

```
operators/apace_quant_matmul_reduce_scatter_v2/
├── kernel/apace_quant_matmul_reduce_scatter_v2/   ← [MODIFY] 3 个头文件
│   ├── apace_quant_matmul_reduce_scatter_v2_udma_impl.h
│   ├── quant_matmul_mx_kernel.h
│   └── apace_quant_matmul_reduce_scatter_v2_tiling_data.h
├── block/                                      ← [REUSE] 共享层
├── tiling/                                     ← [REUSE]
├── basic/                                      ← [REUSE]
├── utils/                                      ← [REUSE]
├── st/apace_quant_matmul_reduce_scatter_v2/    ← [MODIFY] ST 工程
│   ├── CMakeLists.txt
│   ├── run.sh
│   ├── cases.csv
│   ├── src/
│   │   ├── main.cpp
│   │   └── kernel_launcher.h
│   └── scripts/
│       ├── gen_data.py
│       └── verify_result.py
├── st/utils/                                   ← [REUSE] root_info_exchanger.h
├── st/CMakeLists.txt                           ← [REUSE]
├── docs/
│   ├── DESIGN.md                               ← 本文件
│   ├── PLAN.md
│   └── environment.md
└── cmake/                                      ← 构建辅助
    └── third_party/ops-tensor.cmake            ← ops-tensor FetchContent
```

---

## 3. 确认清单

- [x] APACE 路线四大约束已显式确认（§2.0）
- [x] 3 级流水架构设计完成：mm→aivComm→reduceSum，PIPELINE_NUM=4（§1.3, §1.4）
- [x] workspace 设计完成：算子入参，PIPELINE_NUM×coreNum×baseM×baseN×BF16（§1.5）
- [x] reduceSum 实现设计完成：DataCopyPad(workspace→UB) + DataCopyPad(UB→yGm)，无 AtomicAdd（§1.4）
- [x] AIV 时分复用设计完成：先通信后 reduceSum + 流水线效率分析（§2.5, §2.11）
- [x] CrossCore flag 三通道编排设计完成（950PR 模板参数生效，modeId=2 + PIPE_MTE2/PIPE_M）（§2.6）
- [x] yGm 清零策略确定：不需要（reduceSum 普通搬运覆盖，非 AtomicAdd）（§2.13）
- [x] Blaze QuantMatmul 组件选择完成（与 v1 相同）（§2.2）
- [x] Blaze mm per-tile 动态输出地址改造方案完成：while 循环内 per-tile 创建 gmCWorkspace Tensor（§1.4, §2.5）
- [x] tiling_data 结构体设计完成（新增 pipelineNum/coreNum/ubCalSize/totalMmTiles/workspaceSize）（§2.7）
- [x] Host 侧 tiling 推导设计完成（§2.8）
- [x] 入口函数设计完成（4 dtype 变体，8 参数含 workspace）（§2.9）
- [x] 多卡通信拓扑设计完成（4 卡 ReduceScatter，与 v1 相同）（§2.10）
- [x] AIC/AIV 分工设计完成（§2.5, §2.11）
- [x] 与 v1 的关键差异对比表完成（§2.14）
- [x] v2 Process() 流水循环伪代码完成（含 WaitFlag 模板参数 + DataCopyPad 参数 + gmCWorkspace 改造）（§1.4）
- [x] local 前置作为可选优化项（默认 0，映射 DEFERRED_SYNC 模式）（§2.12）
- [x] 精度策略确定（rtol=atol=1e-2，无 AtomicAdd 精度风险）（§2.13）
- [x] flagId 冲突规避完成（避开 3、5、[11-14]，约束 tile 总数 ≤ 10）（§2.6, §2.8）
- [x] API 验证完成（所有 API 已通过文档验证）（§1.2）
- [x] 文件清单与 REUSE/MODIFY 标记完成（§2.15）
- [x] 分支场景覆盖完成（§2.12）
- [x] 串讲回应已完成（WALKTHROUGH.md §Architect 回应）
- [x] design_issue 解决（2026-08-12）：AIV reduceSum 复用 BlockScheduler（方案 b，AIC/AIV 步调构造一致）+ totalMmTiles 废除（§2.16）
