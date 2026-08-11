# apace_quant_matmul_reduce_scatter 算子设计文档

> **路线**：APACE MC2 通算融合（UDMA PUT 直调）
> **参考算子**：`all_to_all_quant_matmul`（AllToAll PUT + UDMA）、`all_gather_quant_matmul`（AllGather PUT + UDMA）

---

## 0. 概述

### 0.0 需求类型判断

**通用**：支持多种 FP8 dtype 组合和 shape 配置，4 卡 ReduceScatter 通信。

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | apace_quant_matmul_reduce_scatter |
| 算子类别 | MC2 通算融合（Quant MatMul + ReduceScatter 通信） |
| 需求类型 | 通用 |
| 支持数据类型 | A/B = fp8_e4m3fn_t / fp8_e5m2_t（4 种组合），C = bfloat16_t |
| 支持芯片 | Ascend950PR（DAV_3510） |
| NpuArch | DAV_3510（`__NPU_ARCH__` = 3510） |
| `--npu-arch` 编译参数 | `dav-3510` |
| SocVersion | ASCEND950 |
| CANN 版本 | 9.2.0 |
| 通信模式 | UDMA PUT（直调仅支持 UDMA） |
| 多卡拓扑 | 4 卡 ReduceScatter |
| 特殊约束 | APACE 路线四大红线（见 §2.0） |

### 0.2 用户原始需求

| # | 需求内容 |
|---|---------|
| 1 | 开发 APACE 路线的 MC2 通算融合算子，将量化 MatMul（Quant MatMul）与 ReduceScatter 通信融合在一个 Kernel 中 |
| 2 | 通信方式实现 UDMA 模式（直调仅支持 UDMA） |
| 3 | 硬件环境：4 × Ascend950PR（dav-3510 架构），CANN 9.2.0，CANN 路径 /home/l00947526/Ascend806/cann |
| 4 | CANN 环境变量：执行前必须 `source /home/l00947526/Ascend806/cann/set_env.sh`（设置 ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0） |
| 5 | APACE 路线约束：禁止 `__schedmode__(1)` / `[[bisheng::core_ratio(1,1)]]`；Matmul 走 Blaze 模板；禁止修改 `block/` 和 `tiling/`；直调仅 UDMA；禁止 HCCL 高阶 API；必须有 `KERNEL_TYPE_MIX_AIC_1_1` 核配比声明 |
| 6 | ReduceScatter 可通过 AllToAll PUT + AtomicAdd 模式实现 |
| 7 | 现有参考算子：all_to_all_quant_matmul（AllToAll PUT + UDMA）、all_gather_quant_matmul（AllGather PUT + UDMA） |

---

## 1. 算子设计

### 1.1 数学公式

```
输入:
  rank i 持有:
    A_i  - shape [M, K_local],  dtype = fp8_e4m3fn_t / fp8_e5m2_t   (K_local = K / rankSize)
    ScaleA_i - shape [M, CeilDiv(K_local,64)*2], dtype = fp8_e8m0_t  (MXFP8 scale)
    B_full - shape [K, N],      dtype = fp8_e4m3fn_t / fp8_e5m2_t   (全量复制, K = rankSize × K_local)
    ScaleB_full - shape [CeilDiv(K,64)*2, N], dtype = fp8_e8m0_t    (全量复制)

输出:
  rank r 获得:
    C_r - shape [M/rankSize, N], dtype = bfloat16_t                 (ReduceScatter 输出)

数学公式:
  C_r = Σ_{i=0}^{rankSize-1} dequant(A_i[r*M/R : (r+1)*M/R, :]) × dequant(B_i[:, :])
      = Σ_{i=0}^{rankSize-1} (A_i[local_M_seg] × B_i[K_local × N])  (反量化后矩阵乘)
  其中 R = rankSize = 4, local_M_seg = r*M/R : (r+1)*M/R

ReduceScatter 语义:
  - 每个 rank 持有完整 M×K_local 的 A，计算全量部分和
  - 输出按 M 轴切分：rank r 获得 C 的第 r 个 M/rankSize 段
  - 多 rank 部分和通过 AtomicAdd 累加
```

### 1.2 API 映射

| 功能 | API/组件 | 来源 | 验证状态 |
|------|---------|------|---------|
| MX 量化 Matmul 计算 | `Blaze::Gemm::Block::BlockMmad<MatmulWithScaleMx<0,false>, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>` | ops-tensor `blaze/gemm/block/block_mmad_qbmm_mx.h` | ✅ 官网参考算子验证 |
| Matmul tile 调度 | `Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>` | ops-tensor `blaze/gemm/block/block_scheduler_qbmm.h` | ✅ 官网参考算子验证 |
| MX 量化 DispatchPolicy | `Blaze::Gemm::MatmulWithScaleMx<0, false>`（`0`=NONE_FULL_LOAD_MODE, `false`=非atomic） | ops-tensor `blaze/gemm/policy/dispatch_policy.h` | ✅ 官网参考算子验证 |
| Apace 组合模式 Kernel | `QuantMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler, CommPolicy>` | apace `kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h` | ✅ 官网参考算子验证 |
| AIV 通信（data） | `CollectiveComm<AllToAll, PUT, AType, TeamBarrier>` | apace `block/aiv_comm/collective_comm_api.h` | ✅ 官网参考算子验证 |
| AIV 通信（scale） | `CollectiveComm<AllToAll, PUT, fp8_e8m0_t, TeamBarrier>` | apace `block/aiv_comm/collective_comm_api.h` | ✅ 官网参考算子验证 |
| 跨卡同步 | `TeamBarrier` | apace `block/aiv_comm/barrier/barrier_ubmem.h` | ✅ 官网参考算子验证 |
| AIC 等待通信 | `UdmaCommWaitPolicy::WaitTile(tileIdx)` → `CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx)` | apace kernel | ✅ 官网参考算子验证 |
| AIV 通知 AIC | `CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)` | AscendC API | ✅ asc-devkit 文档验证（CrossCoreSetFlag_ISASI.md，950PR 支持 mode 2） |
| AIV 块间同步 | `SyncAll<true>()` | AscendC API | ✅ asc-devkit 文档验证（SyncAll.md） |
| 原子累加（LOCAL+REMOTE） | `SetAtomicAdd<CType>()` / `SetAtomicNone()` | AscendC API | ✅ asc-devkit 文档验证（asc_set_atomic_add.md，950PR 支持 bfloat16_t） |
| 管线屏障（MTE 修复） | `PipeBarrier<PIPE_ALL>()` | AscendC API | ✅ asc-devkit 文档验证（PipeBarrier_ISASI.md，950PR 支持 PIPE_ALL） |
| 核配比声明 | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` | kernel_basic_intf.h | ✅ 官网参考算子验证 |
| Host 通信建链 | `CommChannelBuilder::CreateDeviceContext` | apace `utils/comm_channel_builder.h` | ✅ 官网参考算子验证 |
| SWAT tiling 推导 | `QuantMatmulTilingSwat<DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN>` | apace `tiling/quant_matmul_tiling_swat.h` | ✅ 官网参考算子验证 |

#### API 语义验证

| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| CrossCoreSetFlag | AIV→AIC 跨核同步 | 通信 tile 就绪通知 | `<0x2, PIPE_MTE3>(flagId)` | mode 2(950PR支持); flagId 0-15(截断低4bit); 避开 flagId 14（SyncAll<true> 占用，见 SyncAll.md 表3） | ✅ | CrossCoreSetFlag_ISASI.md |
| CrossCoreWaitFlag | AIC 等 AIV 通信完成 | 等待通信数据到达 Win | `<0x2, PIPE_MTE2>(flagId)` | 与 SetFlag 配对; 同一 flagId 只 wait 一次(waitedMask 去重) | ✅ | CrossCoreWaitFlag_ISASI.md |
| SetAtomicAdd | L0C→GM fixpipe 原子累加 | REMOTE 部分和累加到 C | `SetAtomicAdd<bfloat16_t>()` | 支持 bfloat16_t(950PR); 使用后必须 SetAtomicNone; 不做 GM 清零; 由 kernel Run() 内部根据 isAtomicAdd_ 自动调用（见 compute.md §4.1） | ✅ | asc_set_atomic_add.md |
| PipeBarrier | 跨流水线屏障 | LOCAL fixpipe 排空后再 REMOTE | `PipeBarrier<PIPE_ALL>()` | 950PR 支持 PIPE_ALL; 性能影响但保证正确性 | ✅ | PipeBarrier_ISASI.md |
| SyncAll | AIV 块间硬同步 | WriteNbi 对端可见性 | `SyncAll<true>()` | isAIVOnly=true 占用 flagId 14（仅 AIV 侧，见 SyncAll.md 表3）；MIX_AIC_1_1 kernel 中仅 AIV 全核同步（AIC 不参与 SyncAll，AIC↔AIV 同步由 CrossCore flag 独立处理，行为与预期一致） | ✅ | SyncAll.md |

### 1.3 数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ReduceScatter via AllToAll PUT + AtomicAdd            │
│                                                                         │
│  rank i 持有:                                                           │
│    A_i[M × K_local], ScaleA_i[M × scaleKLen_local], B_full[K × N]      │
│                                                                         │
│  ┌─────────────── AIV (通信) ──────────────┐                             │
│  │                                           │                             │
│  │  for tid in 0..commTurn-1:                │                             │
│  │    Commit(ScaleA_i[tid 段]) → 各 rank Win │  ← AllToAll PUT:          │
│  │    Commit(A_i[tid 段])       → 各 rank Win │     rank i 发送 A 的       │
│  │    Wait<BARRIER_DEVICE>()                  │     M/rankSize 段给        │
│  │    SyncAll<true>()                         │     target rank r          │
│  │    SetFlag<0x2, PIPE_MTE3>(tid) ────────────────→ AIC (计算)           │
│  │                                           │                             │
│  │  Finalize(ScaleA) + Finalize(A)           │  ┌──────────────────────┐  │
│  └───────────────────────────────────────────┘  │ LOCAL (localMatmul=1)│  │
│                                                 │                      │  │
│  Win[rank r] 布局:                              │ RunLocalMatmul():    │  │
│  ┌────────────────────────────────┐            │  A_r[my_M_seg] ×     │  │
│  │ data: rank0 A段 │ rank1 A段 │…│            │  B_r → C_local       │  │
│  │ scale: rank0 S段 │ rank1 S段 │…│            │  (首次写入,无Atomic) │  │
│  └────────────────────────────────┘            │                      │  │
│                                                 │ PipeBarrier<ALL>() ★│  │
│  每个 A 段 = M/rankSize × K_local              │                      │  │
│                                                 │ RunMatmul():        │  │
│                                                 │  (kernel 内部自动    │  │
│                                                 │   SetAtomicAdd)      │  │
│                                                 │  for rank ≠ r:      │  │
│                                                 │    WaitTile(tid)    │  │
│                                                 │    A_i[my_M_seg] ×  │  │
│                                                 │    B_i → C_local    │  │
│                                                 │    (AtomicAdd 累加) │  │
│                                                 │  (kernel 内部自动    │  │
│                                                 │   SetAtomicNone)     │  │
│                                                 └──────────────────────┘  │
│                                                                         │
│  输出: rank r 获得 C_r[M/rankSize × N] = Σ_i A_i[my_M_seg] × B_i       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 核心计算步骤

**核心计算步骤**:
```
1. AIV 通信循环 (RunAllToAll)
   - 逐 tile Commit ScaleA → Commit A → Wait → SyncAll → SetFlag 通知 AIC
   - AllToAll PUT: rank i 发送 A_i 的 M/rankSize 段给 target rank r

2. AIC LOCAL 计算 (RunLocalMatmul, localMatmul=1)
   - 本 rank A_local[my_M_seg] × B_local → C_local (首次写入, splitKNum=1)
   - 与 AIV PUT 并行执行（通算重叠）

3. PipeBarrier<PIPE_ALL>() ★ 关键修复（新增补丁，官网参考算子无此行）
   - 确保 LOCAL fixpipe (MTE3) 排空后再进入 REMOTE AtomicAdd

4. AIC REMOTE 计算 (RunMatmul)
   - SetupParams(mode=REMOTE, splitKNum=rankSize-1)
   - kernel_(params)：kernel 内部根据 isAtomicAdd_ 自动调 SetAtomicAdd/SetAtomicNone
   - 遍历远端 rank: WaitTile → A_remote[my_M_seg] × B_remote → C_local (AtomicAdd 累加)
   - splitKNum = rankSize - 1 (跳过 self rank)

5. 输出 C_local[M/rankSize × N] 写入 GM
```

**分支差异对比**:

| 操作 | localMatmul=0 (REMOTE-only) | localMatmul=1 (LOCAL+REMOTE) ★推荐 | localMatmul=2 (DEFERRED_SYNC) |
|------|----------------------------|--------------------------------------|-------------------------------|
| AIC 执行顺序 | 直接 REMOTE | LOCAL → PipeBarrier → REMOTE | per-tile: self mmad → wait → remote mmad → fixpipe |
| AtomicAdd | 不需要 | REMOTE 阶段开启 | 不需要 |
| splitKNum | rankSize | rankSize-1 (LOCAL) + rankSize-1 (REMOTE) | rankSize |
| 通算并行 | 无（全部在 wait 之后） | 有（LOCAL 与 AIV PUT 并行） | 弱（per-tile 粒度） |
| 精度 | 标准 | BF16 AtomicAdd 精度损失 ~0.79% | L0C 全 FP32 累加，精度最好 |
| 性能 | 基线 | 最优（通算并行） | 降速 ~27% |

> **选型决策**：localMatmul=1（通算并行是 MC2 核心价值，精度损失在 FP8 量化容差内可忽略）

**关键设计要点**:
1. **PipeBarrier 修复**：localMatmul=1 模式下，RunLocalMatmul 和 RunMatmul 之间**必须**插入 `PipeBarrier<PIPE_ALL>()`，否则 LOCAL 的 fixpipe (MTE3) 未排空时 REMOTE 的 AtomicAdd read-modify-write 会读到未完整写入的 GM 值，触发 MTE 异常 (aclError:507015)
2. **waitedMask ≤ 32 约束**：通信 tile 总数 = `splitAxisTileCnt + splitAxisTailCnt` ≤ 32（`uint32_t` 位掩码限制）
3. **B 全量复制**：PUT + AtomicAdd 模式要求每 rank 持有完整 B（rankSize × K_local × N），内存开销为 rankSize 倍
4. **Win 区空间**：ReduceScatter 的 Win 区 = rankSize × (M/rankSize × K_local × sizeof) = M × K_local × sizeof，与 all_to_all 的 Win 区相同（因每 rank 收到 rankSize 个 M/rankSize 段）

### 1.5 内存管理(Buffer 规划)

#### UB 分配（AIV 侧）

| Buffer 名称 | 用途 | 大小 | 来源 |
|------------|------|------|------|
| commBuf (data) | data 通信 workspace | 512B (`COMM_WORKSPACE_SIZE`) | `collective_comm_context.h` |
| commScaleBuf | scale 通信 workspace | 512B (`COMM_WORKSPACE_SIZE`) | `collective_comm_context.h` |
| barrierBuf | 跨卡同步 flag | 32B (`BARRIER_FLAG_SIZE`) | `barrier_ubmem.h` |
| **合计** | | **1056B** | 双通信对象 + barrier |

#### L0C 分配（AIC 侧）

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| L0C FP32 累加器 | 各 rank 部分和累加 | baseM × baseN × 4B（FP32）|
| 约束 | ≤ 128KB（dbL0C ping-pong 半区）或 ≤ 256KB（单缓冲） | baseM=128, baseN=128 → 64KB ✅ |

#### Win 区（GM 共享区）

| 段 | 布局 | 大小 |
|----|------|------|
| data 段 | rankSize × (M/rankSize × K_local × sizeof(AType)) = M × K_local × sizeof(AType) | winOffset=0 |
| scale 段 | rankSize × (M/rankSize × scaleKLen_local × sizeof(fp8_e8m0_t)) | winOffset=M × K_local × sizeof(AType) |

#### GM 输入输出

| Tensor | 分布 | shape (per rank) | 说明 |
|--------|------|-------------------|------|
| A | K-split | [M, K_local] | 本 rank 的 A 段 |
| ScaleA | K-split | [M, CeilDiv(K_local,64)*2] | MXFP8 scale |
| B | 全量复制 | [K, N] (K=rankSize×K_local) | 按 rank 沿 K 拼接；每 rank 持有完整 B |
| ScaleB | 全量复制 | [CeilDiv(K,64)*2, N] | 全量 scale |
| C | M-split (输出) | [M/rankSize, N] | ReduceScatter 输出 |

**B 内存开销估算**:

| 项目 | 公式 | T3 示例 (K=8192, N=4096, FP8) |
|------|------|-------------------------------|
| B 内存 (per rank) | K × N × sizeof(AType) | 8192 × 4096 × 1 = 32 MB |
| B 总内存 (all ranks) | rankSize × K × N × sizeof(AType) | 4 × 32 = 128 MB |

> 950PR GM 容量通常足够覆盖 T3 的 128MB B 总内存。极大 shape（如 K=16384, N=8192）时 B 可达 128MB/rank，需关注 GM 容量上限。

---

## 2. 架构设计

### 2.0 APACE 路线约束显式确认

| # | 约束 | 状态 | 验收条件 |
|---|------|------|---------|
| ① | 禁止 `__schedmode__(1)` 和 `[[bisheng::core_ratio(1,1)]]` | ✅ 确认 | 核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证，不使用 schedmode |
| ② | Matmul 走 Blaze 模板 | ✅ 确认 | 使用 `BlockMmad` + `BlockScheduler` + `MatmulWithScaleMx`，无 `AscendC::Matmul` |
| ③ | 禁止修改 `block/` 和 `tiling/` | ✅ 确认 | 只在 `kernel/apace_quant_matmul_reduce_scatter/` 下创建文件 |
| ④ | 直调仅 UDMA | ✅ 确认 | HCCL windows/CCU 不支持直调；本算子使用 UDMA PUT 模式 |

### 2.1 通信模式选择

**选型**：AllToAll PUT + AtomicAdd（UDMA 引擎）

**选型理由**：

| 维度 | 说明 |
|------|------|
| 通信方向 | PUT（通信→计算）：AIV 先推 A 到远端 Win，AIC 后算 |
| 通信引擎 | UDMA（`Hcomm::WriteNbi`），直调唯一支持 |
| 通信原语 | AllToAll PUT（复用 `block/aiv_comm/all_to_all/all_to_all_udma_put.h`） |
| ReduceScatter 实现 | AllToAll PUT 发送不同 M 段给不同 rank + AtomicAdd 累加部分和 |
| 切分轴 | M 轴（每 rank 收到 M/rankSize 段，不同 M 段完全独立） |
| 无需新增 block 原语 | ReduceScatter 语义通过 AllToAll PUT + AtomicAdd 模式实现，不新增 `block/aiv_comm/reduce_scatter/` |

**ReduceScatter 实现原理**（基于 fusion.md §6.2 推导）：

```
AllToAll PUT 地址公式:
  srcAddr = localAddr_ + targetRankId * chunkBytes_ + tileOffset
  dstAddr = commBufferAddrs[targetRankId] + winOffset_ + rankId * chunkBytes_ + tileOffset

当 chunkBytes = (M/rankSize) × K_local × sizeof(AType) 时:
  - srcAddr 偏移 targetRankId × (M/rankSize × K_local) → 读取 A 的第 targetRankId 个 M/rankSize 段
  - dstAddr 写入 target rank 的 Win 区 → 每个 target rank 收到不同的 M 段

效果: rank i 将 A_i 按 M 轴分成 rankSize 段，第 r 段发送给 rank r
```

### 2.2 Blaze QuantMatmul 组件选择

| 组件 | 选择 | 来源 | 说明 |
|------|------|------|------|
| 路径 | blaze library | `third_party/blaze/gemm/` | MX 量化走 blaze library |
| DispatchPolicy | `MatmulWithScaleMx<0, false>` | `blaze/gemm/policy/dispatch_policy.h` | `0`=NONE_FULL_LOAD_MODE, `false`=非atomic |
| BlockMmad | `BlockMmad<DispatchPolicy, AType, NDExtLayoutPtn, BType, DNExtLayoutPtn, CType, NDExtLayoutPtn, BiasType, NDExtLayoutPtn>` | `blaze/gemm/block/block_mmad_qbmm_mx.h` | TransA=false, TransB=true（NZ weight） |
| BlockScheduler | `BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, NDExtLayoutPtn, DNExtLayoutPtn, AType>` | `blaze/gemm/block/block_scheduler_qbmm.h` | M/N tile 调度 + 尾块拆分 |
| Kernel | `QuantMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler, CommPolicy>` | apace kernel（复制自参考算子） | 组合模式：Scheduler + Mmad + 通信等待 |
| CommPolicy | `UdmaCommWaitPolicy` | 算子 impl 中定义 | `WaitTile(tileIdx)` → `CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx)` |
| Tiling 引擎 | `QuantMatmulTilingSwat<DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN>` | apace `tiling/quant_matmul_tiling_swat.h` | SWAT 流式 tiling |

### 2.3 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | M 轴（输出 C 的 M/rankSize 行）× N 轴 |
| ProblemShape | `<M/rankSize, N, ka, 1>` — M 为 per-rank 输出行数，ka = K/rankSize 为 per-rank K（与参考算子一致；layoutB = (rankSize × ka, N) = (K, N) 匹配 B 全量数据，rank r 的 B 段起点 r × ka） |
| 单核任务量 | M/rankSize × N 矩阵的 baseM × baseN tile |
| 使用的核数 | `usedCoreNum`（由 `QuantMatmulTilingSwat::GetTilingData` 推导） |
| 负载均衡 | BlockScheduler M/N 双轴 tile 调度 + 尾块拆分 |
| 核配比 | AIC:AIV = 1:1（`KERNEL_TYPE_MIX_AIC_1_1`） |

### 2.4 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 248KB（DAV_3510） |
| UB 分配 | 1056B（双通信对象 512B×2 + barrier 32B），远小于 UB 容量 |
| UB 分配方式 | 局部 `ubOffset` 线性累加（与参考算子一致） |

### 2.5 AIC/AIV 分工设计

```
┌─────────────────────── AIV (Vector 核) ──────────────────────────┐
│                                                                   │
│  Init():                                                          │
│    提取 udmaCtx/ubmemCtx → InitBaseParams → UB 分配               │
│    → teamBarrier_.Init → allToAllA_.Init<BARRIER_NONE>            │
│    → allToAllScaleA_.Init(winOffset=rankSize×rankDataBytes)       │
│                                                                   │
│  RunAllToAll():                                                   │
│    for tid in 0..commTurn-1:                                      │
│      if GetBlockIdx() < rankSize:                                 │
│        allToAllScaleA_.Commit()   ← scale 先                      │
│        allToAllA_.Commit()        ← data 后                      │
│        allToAllA_.Wait<BARRIER_DEVICE>()  ← 同 channel 只 Wait 1次│
│      SyncAll<true>()              ← 全 AIV 执行                   │
│      CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)  ← 通知 AIC            │
│    allToAllScaleA_.Finalize()                                     │
│    allToAllA_.Finalize()                                          │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
                              ↕ CrossCore Flag
┌─────────────────────── AIC (Cube 核) ────────────────────────────┐
│                                                                   │
│  Init(params):                                                    │
│    解析 isBias/isAtomicAdd → ResetGmAddr                          │
│                                                                   │
│  Run():                                                           │
│    if localMatmul == 1:                                           │
│      RunLocalMatmul()   ← A_local × B_local → C (首次写入)       │
│      PipeBarrier<PIPE_ALL>()  // ★ 新增补丁（官网无此行，localMatmul=1 必需）│
│      RunMatmul()        ← REMOTE: AtomicAdd 累加                  │
│    else:                                                          │
│      RunMatmul()                                                  │
│                                                                   │
│  RunLocalMatmul():                                                │
│    SetupParams(mode=LOCAL, splitKNum=1)                           │
│    kernel_(params)  ← 本 rank A[my_M_seg] × B_local → C           │
│                                                                   │
│  RunMatmul():                                                     │
│    SetupParams(mode=REMOTE, splitKNum=rankSize-1)                 │
│    kernel_(params)  ← kernel 内部 isAtomicAdd_ 自动调             │
│                       SetAtomicAdd/SetAtomicNone（见 compute.md   │
│                       §4.1）；遍历远端 rank: WaitTile→A×B→C(Atomic)│
│                                                                   │
│  QuantMatmulMxKernel::ProcessSingleBatch():                       │
│    while bs.GetTileIdx(blockIdx):                                 │
│      mPos, nPos = bs.GetTileCoord()                               │
│      dependTileIdx = CalcDependTileIdx(mPos+baseM-1, headTileSize)│
│      if not waited: commPolicy_.WaitTile(dependTileIdx)           │
│      for rank in 0..rankSize-1:                                   │
│        if REMOTE && rank==rankId && localMatmul==1: continue      │
│        mmadOp_(A[rank], B[rank], scaleA, scaleB, bias, C, shape,  │
│               remoteRankCnt)                                      │
│    drain: 补齐未 wait 的 tile                                     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**同步点表**：

| 同步点 | 方向 | API | flagId | 含义 |
|--------|------|-----|--------|------|
| AIV→AIC 通信就绪 | AIV→AIC | `CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)` → `CrossCoreWaitFlag<0x2, PIPE_MTE2>(tid)` | tid (tile 索引) | 通信 tile tid 数据就绪 |
| AIV 块间同步 | AIV↔AIV | `SyncAll<true>()` | 14（isAIVOnly=true，见 SyncAll.md 表3） | WriteNbi 对端可见性 |
| 跨卡同步 | AIV↔AIV | `TeamBarrier.CrossDevice()` | GM flag | 每轮通信后跨卡同步 |
| LOCAL→REMOTE 管线 | AIC 内部 | `PipeBarrier<PIPE_ALL>()` | — | LOCAL fixpipe 排空 |

### 2.6 通算流水编排

**flag 编排模式**（PUT 模式，基于 fusion.md §3.2）：

```
AIV (RunAllToAll)                              AIC (RunMatmul)
─────────────────                              ──────────────
round tid:
  Commit(scale A[tid])
  Commit(data A[tid])
  Wait<BARRIER_DEVICE>()
  SyncAll<true>()
  SetFlag<0x2, PIPE_MTE3>(tid) ─────→          WaitFlag<0x2, PIPE_MTE2>(tid) [waitedMask 去重]
                                               从 Win 读 A[tid] → Matmul
                                               (localMatmul=1: LOCAL 前置与 AIV PUT 并行)
```

**localMatmul 模式编排**：

```
AIC Run() 编排:
   ┌── localMatmul == 1 ──┐
   │                       │
   │  RunLocalMatmul()     │ ← 与 AIV RunAllToAll() 并行执行
   │  (A_local × B → C)    │   (通算重叠：LOCAL 计算掩盖 PUT 通信)
   │                       │
   │  PipeBarrier<ALL>() ★ │ ← 新增补丁：确保 LOCAL fixpipe 排空
   │                       │
   │  RunMatmul()          │ ← REMOTE: kernel 内部自动 SetAtomicAdd
   │                       │   → 遍历远端 rank AtomicAdd 累加 → SetAtomicNone
   └───────────────────────┘
```

### 2.7 tiling_data 结构体设计

```cpp
// apace_quant_matmul_reduce_scatter_tiling_data.h

#pragma pack(push, 8)
struct alignas(8) ApaceQuantMatmulReduceScatterTilingData {
    CommTilingData commTilingData;          // data 通道通信切分
    CommTilingData scaleCommTilingData;     // scale 通道通信切分 (nonSplitAxisSize = ka/32)
    QuantMatmulTilingData tileQbmmTilingData; // 计算 tiling (m=M/rankSize)
    uint32_t localMatmul;                    // 0:REMOTE-only, 1:LOCAL+REMOTE+AtomicAdd, 2:DEFERRED_SYNC
};
#pragma pack(pop)

// CommContext 聚合体 (UDMA 模式)
struct CommContext {
    CommUdmaContext udmaCtx;    // UDMA 通信通道 (rankId, rankSize, channelHandles[], commBufferAddrs[])
    CommUbmemContext ubmemCtx;  // Barrier 通道 (rankId, rankSize, commBufferAddrs[])
};
```

**CommTilingData 填充（ReduceScatter 特化）**：

| 字段 | data 通道 | scale 通道 |
|------|-----------|-----------|
| `splitAxisTileSize` | headMSize | headMSize |
| `splitAxisTileCnt` | (M/rankSize) / headMSize | 同 data |
| `splitAxisTailSize` | (M/rankSize) % headMSize | 同 data |
| `splitAxisTailCnt` | tailMSize > 0 ? 1 : 0 | 同 data |
| `nonSplitAxisSize` | ka (= K/rankSize) | ka/32 (MXFP 压缩比) |

> **关键差异**（与 all_to_all_quant_matmul 对比）：`splitAxisTileCnt` 基于 `M/rankSize`（而非 `M`），因为 ReduceScatter 每个 target rank 只接收 M/rankSize 行。

**QuantMatmulTilingData 填充**：

| 字段 | 值 | 说明 |
|------|-----|------|
| m | M/rankSize | 输出行数（ReduceScatter 切分） |
| n | N | 完整 N |
| k | ka (= K / rankSize) | per-rank K（与参考算子一致；B 全量数据 [K, N] 由 layoutB = (rankSize × ka, N) 匹配） |
| baseM/baseN/baseK | SWAT 推导 | Blaze tile 形状 |
| usedCoreNum | SWAT 推导 | 启动核数 |
| localMatmul | 1 | LOCAL+REMOTE+AtomicAdd |

### 2.8 Host 侧 tiling 推导

```
headMSize 推导 (内联于 main.cpp):
  nTile      = CeilDiv(n, baseN)
  headMSize  = CeilDiv(usedCoreNum, nTile) * baseM   // 与 matmul tile 粒度对齐
  mLocal     = M / rankSize                            // ReduceScatter per-rank M
  headTileCnt = mLocal / headMSize
  tailMSize   = mLocal % headMSize

GetTilingData 调用:
  QuantMatmulTilingSwat<DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN> tilingEngine;
  uint32_t ka = K / rankSize;                                  // per-rank K
  tilingEngine.GetTilingData(mLocal, N, ka, tilingData);       // m=M/rankSize, k=per-rank K (ka)
```

> **关键约束**：GetTilingData 的 `k` 参数必须为 **per-rank K（ka = K/rankSize）**，而非全量 K。这是 PUT 模式的硬性不变量（参考 `operator-anatomy.md` §3.7：PUT 用 ka，AG 用 totalLogicalM）。若误用全量 K，会导致 layoutA `(rankSize×M, K)` 超出 A 实际列数 ka、layoutB `(rankSize×K, N)` 放大 rankSize 倍、rank r 的 B 段偏移 `r×K` 超出 B 数据范围。

**输入校验**:
- `K % rankSize == 0`（K 必须能被 rankSize 整除）
- `M % rankSize == 0`（M 必须能被 rankSize 整除，保证 M/rankSize 为整数）
- `ka % 64 == 0`（MXFP8 scale 对齐：确保 `ka/32` 等价于 `CeilDiv(ka, 64)×2`，避免 scale 通信数据量截断；详见 §2.7 scale 通道 `nonSplitAxisSize = ka/32`）
- `CeilDiv(ka, 64) % 2 == 0`（MXFP8 scale 对齐，被 `ka % 64 == 0` 蕴含，保留为冗余校验与参考算子一致）

> **cases.csv 的 `head_m_size` 列**：仅作信息性参考，**不被 main.cpp 直接使用**。实际 headMSize 由上述公式内联推导（`CeilDiv(usedCoreNum, nTile) * baseM`），与参考算子 `all_to_all_quant_matmul` main.cpp 一致（`operator-anatomy.md` §3.7）。

### 2.9 入口函数设计

**4 个 dtype 变体入口**（参照 all_to_all_quant_matmul 的 `kernel_launcher.h` 模式）：

| 入口函数名 | AType | BType | CType |
|------------|-------|-------|-------|
| `ApaceQuantMatmulReduceScatterKernelE4M3E4M3_Udma` | fp8_e4m3fn_t | fp8_e4m3fn_t | bfloat16_t |
| `ApaceQuantMatmulReduceScatterKernelE5M2E5M2_Udma` | fp8_e5m2_t | fp8_e5m2_t | bfloat16_t |
| `ApaceQuantMatmulReduceScatterKernelE4M3E5M2_Udma` | fp8_e4m3fn_t | fp8_e5m2_t | bfloat16_t |
| `ApaceQuantMatmulReduceScatterKernelE5M2E4M3_Udma` | fp8_e5m2_t | fp8_e4m3fn_t | bfloat16_t |

**入口签名**（UDMA 模式，7 参数）：

```cpp
__global__ __aicore__ void ApaceQuantMatmulReduceScatterKernelE4M3E4M3_Udma(
    __gm__ CommContext *hcommCtx,        // 1. 通信上下文（GM 指针）
    GM_ADDR aGM, GM_ADDR scaleAGM,       // 2-3. A 矩阵与 Scale
    GM_ADDR bGM, GM_ADDR scaleBGM,       // 4-5. B 矩阵与 Scale
    GM_ADDR cGM,                         // 6. C 输出 (M/rankSize × N)
    ApaceQuantMatmulReduceScatterTilingData tilingData) // 7. tiling（按值）
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::ApaceQuantMatmulReduceScatterUdmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData);
    impl.Run();
}
```

### 2.10 多卡通信拓扑（4 卡 ReduceScatter）

```
4 卡 ReduceScatter 数据流:

rank 0: A_0[M×Ka]  ──PUT──→  rank 0 Win: A_0[0:M/4]      (self, 本地读)
                    ──PUT──→  rank 1 Win: A_0[M/4:2M/4]
                    ──PUT──→  rank 2 Win: A_0[2M/4:3M/4]
                    ──PUT──→  rank 3 Win: A_0[3M/4:M]

rank 1: A_1[M×Ka]  ──PUT──→  rank 0 Win: A_1[0:M/4]
                    ──PUT──→  rank 1 Win: A_1[M/4:2M/4]   (self)
                    ──PUT──→  rank 2 Win: A_1[2M/4:3M/4]
                    ──PUT──→  rank 3 Win: A_1[3M/4:M]

rank 2: A_2[M×Ka]  ──PUT──→  rank 0 Win: A_2[0:M/4]
                    ──PUT──→  rank 1 Win: A_2[M/4:2M/4]
                    ──PUT──→  rank 2 Win: A_2[2M/4:3M/4]  (self)
                    ──PUT──→  rank 3 Win: A_2[3M/4:M]

rank 3: A_3[M×Ka]  ──PUT──→  rank 0 Win: A_3[0:M/4]
                    ──PUT──→  rank 1 Win: A_3[M/4:2M/4]
                    ──PUT──→  rank 2 Win: A_3[2M/4:3M/4]
                    ──PUT──→  rank 3 Win: A_3[3M/4:M]     (self)

rank r Win 区 (r=0..3):
  [A_0[r*M/4:(r+1)*M/4, :]]  ← 来自 rank 0
  [A_1[r*M/4:(r+1)*M/4, :]]  ← 来自 rank 1
  [A_2[r*M/4:(r+1)*M/4, :]]  ← 来自 rank 2
  [A_3[r*M/4:(r+1)*M/4, :]]  ← 来自 rank 3

rank r 计算:
  C_r[M/4 × N] = Σ_{i=0}^{3} A_i[r*M/4:(r+1)*M/4] × B_i[Ka × N]
  LOCAL: A_r[my_seg] × B_r → C_r (首次写入)
  REMOTE: A_0,A_1,A_2,A_3[my_seg] × B_i → C_r (AtomicAdd, 跳过 self)
```

### 2.11 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| dtype: E4M3×E4M3 | 入口 `E4M3E4M3_Udma`，默认 dtype 组合 |
| dtype: E5M2×E5M2 | 入口 `E5M2E5M2_Udma` |
| dtype: E4M3×E5M2 | 入口 `E4M3E5M2_Udma` |
| dtype: E5M2×E4M3 | 入口 `E5M2E4M3_Udma` |
| 大 shape (M×K×N) | 多核切分 + 通信 tile 多轮 |
| 小 shape | 单核或少量核 + 通信 tile 少轮 |
| M % rankSize ≠ 0 | 输入校验拒绝（要求 M % rankSize == 0） |
| K % rankSize ≠ 0 | 输入校验拒绝（要求 K % rankSize == 0） |
| 尾块 (M/rankSize % headMSize ≠ 0) | `splitAxisTailSize` + `splitAxisTailCnt` 处理 |
| 通信 tile > 32 | 输入校验拒绝（waitedMask uint32_t 限制） |
| flagId 14 冲突理论风险 | 当 `splitAxisTileCnt + splitAxisTailCnt > 14` 时，`tid % 16 == 14` 的轮次会与 `SyncAll<true>` 内部占用的 flagId 14 冲突（见 SyncAll.md 表3）；当前测试用例 tile 数 ≤ 2 不会触发；大 shape 场景如需规避，可在 tiling 推导时限制 headMSize 使 tile 总数 ≤ 14（牺牲通算流水深度），或接受此风险（与参考算子一致） |

### 2.12 精度策略

| 维度 | 策略 |
|------|------|
| 输出 dtype | bfloat16_t |
| L0C 累加精度 | FP32（mmad 内部 FP32 累加） |
| AtomicAdd 精度 | BF16 read-modify-write（localMatmul=1 的已知精度损失） |
| 精度损失来源 | LOCAL 结果先截断为 BF16，再被 REMOTE AtomicAdd 累加（多一次 FP32→BF16 截断） |
| 预期 max_rel_diff | ≈ 0.79%（rankSize=4，FP8→BF16，开发历史数据） |
| 容差标准 | rtol=atol=1e-2（与 all_to_all_quant_matmul ST 一致） |
| 不达标回退 | localMatmul=2 (DEFERRED_SYNC)：L0C 全 FP32 累加，单次 fixpipe，精度最好 |

### 2.13 文件清单与 REUSE/MODIFY 标记

#### [MODIFY] 算子层文件（3 个 kernel 头文件）

| 文件 | 职责 | 参考来源 |
|------|------|---------|
| `kernel/apace_quant_matmul_reduce_scatter/apace_quant_matmul_reduce_scatter_udma_impl.h` | Impl 类（Init/Run/RunAllToAll/RunLocalMatmul/RunMatmul/SetupParams）+ UdmaCommWaitPolicy | 复制自 `all_to_all_mx_quant_matmul_udma_impl.h`，重命名+适配 |
| `kernel/apace_quant_matmul_reduce_scatter/quant_matmul_mx_kernel.h` | Blaze matmul 调度（QuantMatmulMxKernel） | 复制自参考算子，基本不变（tiling 参数由 LocalParams 控制） |
| `kernel/apace_quant_matmul_reduce_scatter/apace_quant_matmul_reduce_scatter_tiling_data.h` | tiling 结构体 + CommContext | 复制自 `all_to_all_matmul_tiling_data.h`，重命名+适配 |

#### [MODIFY] ST 测试工程

| 文件 | 职责 |
|------|------|
| `st/apace_quant_matmul_reduce_scatter/CMakeLists.txt` | 构建配置（target 名 + `--npu-arch=dav-3510` + include 路径） |
| `st/apace_quant_matmul_reduce_scatter/run.sh` | ST 驱动脚本（cases.csv 行选择 + `--perf` msprof 采集） |
| `st/apace_quant_matmul_reduce_scatter/cases.csv` | 测试用例（m,k,n,rank_num,head_m_size） |
| `st/apace_quant_matmul_reduce_scatter/src/main.cpp` | Host 入口（tiling 推导 + CommContext 建链 + launch） |
| `st/apace_quant_matmul_reduce_scatter/src/kernel_launcher.h` | 4 个 `__global__` 入口函数 |
| `st/apace_quant_matmul_reduce_scatter/scripts/gen_data.py` | 数据生成（FP8 量化 + MXFP8 scale + golden） |
| `st/apace_quant_matmul_reduce_scatter/scripts/verify_result.py` | 精度验证（rtol=atol=1e-2） |
| `st/utils/root_info_exchanger.h` | TCP 交换 HcclRootInfo（跨 rank barrier） |

#### [REUSE] 共享层文件（禁止修改）

| 目录 | 来源 |
|------|------|
| `block/aiv_comm/` | `collective_comm_api.h`, `collective_comm_base.h`, `all_to_all/all_to_all_udma_put.h`, `barrier/barrier_ubmem.h` 等 |
| `block/blaze_ext/` | `gemm/block/qmm_mx_block_mmad_fragment.h` 等 |
| `tiling/` | `comm_tiling_data.h`, `quant_matmul_tiling_*.h` |
| `basic/fragment_tensor/` | `fragment_tensor.h` |
| `utils/` | `comm_channel_builder.h`, `constant.h` |

### 2.14 与 all_to_all_quant_matmul 的关键差异

| 维度 | all_to_all_quant_matmul | apace_quant_matmul_reduce_scatter |
|------|------------------------|----------------------------------|
| 输出 shape | C[M × N]（每 rank 全量） | C[M/rankSize × N]（每 rank M/rankSize 段） |
| ProblemShape.M | M | M/rankSize |
| originalM | M | M/rankSize |
| CommTilingData chunkSize | M | M/rankSize |
| 通信数据量/rank | M × K_local × sizeof | (M/rankSize) × K_local × sizeof |
| Win 区 per-rank 接收 | rankSize × (M × K_local) | rankSize × (M/rankSize × K_local) = M × K_local |
| 输出 C GM 大小 | M × N × sizeof(bfloat16_t) | (M/rankSize) × N × sizeof(bfloat16_t) |
| localMatmul | 0 (ST 默认) | 1 (LOCAL+REMOTE+AtomicAdd) |
| PipeBarrier | 无（localMatmul=0） | 有（localMatmul=1 必需） |
| 通信语义 | AllToAll（全量交换） | ReduceScatter（部分和+M切分） |

### 2.15 工程目录结构

```
operators/apace_quant_matmul_reduce_scatter/
├── kernel/apace_quant_matmul_reduce_scatter/   ← [MODIFY] 3 个头文件
│   ├── apace_quant_matmul_reduce_scatter_udma_impl.h
│   ├── quant_matmul_mx_kernel.h
│   └── apace_quant_matmul_reduce_scatter_tiling_data.h
├── block/                                      ← [REUSE] 共享层
├── tiling/                                     ← [REUSE]
├── basic/                                      ← [REUSE]
├── utils/                                      ← [REUSE]
├── st/apace_quant_matmul_reduce_scatter/       ← [MODIFY] ST 工程
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
├── st/CMakeLists.txt                           ← [REUSE] 顶层构建
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
- [x] 通信模式选择：UDMA PUT（§2.1）
- [x] ReduceScatter 实现方式：AllToAll PUT + AtomicAdd（§2.1）
- [x] Blaze QuantMatmul 组件选择完成（§2.2）
- [x] 通算流水编排设计完成（flag 编排 + localMatmul=1 + PipeBarrier）（§2.6）
- [x] tiling_data 结构体设计完成（§2.7）
- [x] AIC/AIV 分工设计完成（§2.5）
- [x] 多卡通信拓扑设计完成（4 卡 ReduceScatter）（§2.10）
- [x] 入口函数设计完成（4 dtype 变体）（§2.9）
- [x] 精度策略确定（rtol=atol=1e-2，localMatmul=1）（§2.12）
- [x] API 验证完成（所有 API 已通过 asc-devkit 文档验证）（§1.2）
- [x] 文件清单与 REUSE/MODIFY 标记完成（§2.13）
- [x] Host 侧 tiling 推导设计完成（§2.8）
- [x] 分支场景覆盖完成（§2.11）
