# MegaMoeV2 设计

> 目标：在 Ascend 950PR 上以 **DeepGEMM 风格的 wave-based 大流水** 重写 MegaMoe，
> 把 v1 的"dispatch↔GMM1 一段、combine↔GMM2 一段"两段流水改成
> **dispatch + GMM1 + GMM2 + combine 单 kernel 整体流水**。
> 与 v1 共存，独立算子 `MegaMoeV2`。

---

## 1. 设计原则（与用户原话对齐）

1. **流水照搬 DeepGEMM**：wave-based、dispatch 全局先做、wave 内 GMM1→GMM2 持久化、stage 间靠 per-block arrival mask 重叠。
2. **block 级 MM 复用 v1**：直接 include `Blaze::Gemm::Block::BlockMmad<MatmulWithScaleMx<>>` 与 `BlockEpilogueSwigluMxQuant`，不重写。
3. **核分工 = 32 cube + 32 vec(epilogue) + 32 vec(comm)**：`KERNEL_TYPE_MIX_AIC_1_2` 模式天然给 1 cube : 2 vec；`subBlockIdx_==0` 跑 epilogue（SwiGLU+MX-quant），`subBlockIdx_==1` 跑通信（dispatch pull / combine push）。**两组 vec 真并行**。
4. **wave 流水**：dispatch 是 kernel 入口一次性完成，之后所有 expert × token-block 在 wave 调度器下持久执行 GMM1+epilogue → GMM2+combine push，wave 间靠 arrival mask 衔接。
5. **去掉 BS=256 约束**：workspace 由"线性 `maxOutputSize` buffer"改为 **ring buffer**，BS 上限只受 HCCL win buffer 与 routing 上限决定。
6. **v1/v2 共存**：新算子 `MegaMoeV2`，新 aclnn `aclnnMegaMoeV2`，新 torch_extension wrapper `npu_mega_moe_v2`。

---

## 2. 调研结论摘要（详见两份 brief）

### 2.1 DeepGEMM 关键事实

| 维度 | 做法 |
|---|---|
| Wave 切法 | **expert 维**：`kNumExpertsPerWave` 个本地 expert 一组进流水 |
| Dispatch | 入口一次性 push+NVL barrier+pull，**不切 wave** |
| 核分工 | warp 级固定划分（dispatch warps / load A / load B / MMA / epilogue），**不分组通信 SM/计算 SM** |
| 重叠 | per-pool-block 的 `l1_arrival_count` / `l2_arrival_mask` —— block 粒度握手 |
| Combine | `combine_buf[topk][T_max][hidden]` 唯一槽位单写 + 集中 reduce（无 atomic） |
| 融合 | GMM1 的 epilogue 在同一 warp 内 TMEM→reg→sigmoid·gate·up·topk_w→amax→UE8M0→FP8→STSM→TMA store |

### 2.2 ops-transformer v1 关键事实

| 维度 | 做法 / 文件 |
|---|---|
| Kernel 入口 | `mega_moe.cpp` 38 行薄壳，`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` |
| 核数 | AIC=32, AIV=64, `subBlockIdx_∈{0,1}` 区分 vec0/vec1 |
| 流水 | 2 段：① dispatch↔GMM1（`mega_moe.h:702-720`），② GMM2↔combine（`mega_moe.h:736-746`） |
| 同步 | CrossCoreFlag(SYNC_AIC_AIV_MODE=4)：`AIC_SYNC_AIV_FLAG=4`、`AIV_SYNC_AIC_FLAG=6`，外加 GM atomic flag `ptrFlagDispatchToGmm1`、`ptrFlagSwiGluToGmm2` |
| GMM | `Blaze::Gemm::Block::BlockMmad<MatmulWithScaleMx<>>`，L1_TILE=(128,256,256)，L0_K=128 |
| Epilogue | `BlockEpilogueSwigluMxQuant`（700 行，`mega_moe_impl.h:50-208`）已 vec-tile 化 |
| Dispatch | `MoeInitRoutingV3`（sort+count+gather+mxfp8 quant）→ HCCL win buffer 直读 + magic-flag 同步 + `CopyGMToGMPerTokenV2` 远端 ping-pong 搬运 |
| Combine | `CombineTokens` cube tile 出 UB 直接 `DataCopyPad` 写远端 win → `Unpermute` 在 recv 端做 topk weighted sum |
| BS 上限 | **不是硬编码**，是 `ptrA / ptrA2` 按 `maxOutputSize=m·EP·min(topK,EpR)` 线性分配的 workspace 内存压力 |

---

## 3. V2 高层架构

### 3.1 核心抉择对照表

| 维度 | v1 | DeepGEMM | **V2 抉择** |
|---|---|---|---|
| 流水段数 | 2 段串行 | 4 阶段持久流水 | **4 阶段 wave 流水** |
| Wave 切法 | 无 wave 概念，按 expert 串 | **expert 维** | **expert 维**（每 wave `kExpertsPerWave` 个本地 expert） |
| Dispatch 时机 | 与 GMM1 重叠 | 入口一次性完成 | **入口一次性完成** |
| Stage 衔接 | GM atomic counter 等 totalTile | per-block arrival mask | **per-block arrival mask**（GM uint64 数组） |
| Cube↔Vec 同步 | CrossCoreFlag×2 + GM flag | mbarrier + named bar | **CrossCoreFlag + 扩展 ping-pong 偏移** |
| Combine | tile 出 UB → DataCopyPad 写远端，recv 端 unpermute | 唯一槽位单写 + 集中 reduce | **沿用 v1 的 send + Unpermute**（`combine_send.h`+`Unpermute`），槽位与 v1 一致 |
| Workspace `ptrA/ptrA2` | 线性 `maxOutputSize` | per-pool ring | **ring buffer**（`chunk_depth × chunk_M × hidden`） |
| BS 上限 | ~256 | 无（HBM 上限） | **无内置约束**（仅 HCCL win 决定） |

### 3.2 核分工（950PR：32 AIC + 64 AIV）

```
┌─────────────────────────────────────────────────────┐
│  AIC × 32 (subBlockIdx_=0 视角下与 vec0 同 mix-pair) │
│  - block GMM1 + block GMM2 主体（持久 wave loop）    │
│  - 输出 fixpipe 到 epilogue UB（与 vec0 共享 mix）   │
├─────────────────────────────────────────────────────┤
│  AIV × 32  (vec0, subBlockIdx_=0)                    │
│  - GMM1 epilogue: SwiGLU + MX-quant + 写 ptrA2       │
│  - GMM2 epilogue: BF16 cast + 写 combine UB          │
│  - 与 cube 1:1 配对，沿用 v1 epilogue 类             │
├─────────────────────────────────────────────────────┤
│  AIV × 32  (vec1, subBlockIdx_=1)                    │
│  - dispatch: 远端 win → 本地 ring（CopyGMToGMPerToken）│
│  - combine push: 本地 GMM2 UB → 远端 win              │
│  - cross-rank cnt 同步（magic flag）                  │
│  - 末尾 Unpermute（recv-side topk weighted sum）       │
└─────────────────────────────────────────────────────┘
```

**与 DeepGEMM 不同**：DeepGEMM 是 warp 级在同一 SM 内分工；Ascend 上 AIC/AIV 是物理分核，没有 warp 概念，所以 V2 用 **AIC 专做 cube、AIV 拆 vec0/vec1 双角色** 来等效。这是用户原话"一半的核做通信使用，一半的和配合 matmul 去做 swiglu quant 的计算"的字面落地。

### 3.3 wave-based 整体流水

```
Kernel 入口
  ┌─ AIV: MoeInitRoutingV3 (sort + count + gather + MX-quant) → ptrA0 (peermem)
  │
  ├─ AIV: CrossRankSyncCntPerExpert → 拿到 expert_recv_count
  │
  └─ AIV: 全局 dispatch pull: 把所有 expert 的 token 从远端 win 拉到本地 ring buffer
         （per-chunk 完成时 AtomicAdd dispatchArrival[chunk_id]）
                        │
                        │  「dispatch 完成全局 NVL barrier」
                        ▼
持久 wave 循环 for waveIdx in [0, num_waves):
  本 wave 处理 expert ∈ [wave·E_per_wave, (wave+1)·E_per_wave)
  对每个 expert 的所有 (m_block, n_block) tile：
    
    ┌── AIC: block GMM1（等待 dispatchArrival[chunk] ≥ 阈值）
    │   ├── fixpipe 进 vec0 UB
    │   └── Notify(vec0)
    │
    ├── AIV vec0: SwiGLU + MX-quant（block_epilogue_swiglu_mx_quant.h，复用）
    │   ├── 写 ring 段 ptrA2
    │   └── AtomicAdd l2_arrival_mask[m_block] 通知 GMM2
    │
    ├── AIC: block GMM2（等待 l2_arrival_mask）
    │   ├── fixpipe 进 vec0 UB
    │   └── Notify(vec0) for combine
    │
    ├── AIV vec0: BF16 cast 出到 combine UB
    │   └── Notify(vec1)
    │
    └── AIV vec1: combine push（CombineTokens，复用）
        └── DataCopyPad 写远端 win
        
       （wave i+1 的 GMM1 与 wave i 的 GMM2 通过 ring buffer 重叠）
                        │
                        │  「全局 cross-rank sync」
                        ▼
末尾
  └─ AIV: Unpermute (recv-side topk weighted sum) → 输出 y
```

**关键重叠**：
- **wave 内**：GMM1 → epilogue → GMM2 → combine push 之间靠 cube/vec ping-pong 与 arrival mask block 粒度握手 → 每 4 个 block 阶段重叠
- **wave 间**：wave i 的 GMM2/combine 与 wave i+1 的 GMM1 通过 ring buffer 上不同 chunk 重叠（环形 ptrA2 + 双 ping-pong flag 偏移）
- **dispatch ↔ GMM1**：dispatch 在 kernel 入口一次性做完，但分 chunk 完成 → GMM1 可在第一批 chunk 到位时就启动，不必等全 dispatch（与 DeepGEMM 一致）

### 3.4 同步图（V2）

| 类型 | 位置 | 用途 |
|---|---|---|
| `CrossCoreFlag(SYNC_AIC_AIV_MODE=4)` | cube ↔ vec0 | 每 block tile 的 fixpipe→epilogue ping-pong；`flag id` 沿用 v1 的 `AIC_SYNC_AIV_FLAG=4`、`AIV_SYNC_AIC_FLAG=6`，`+value` 偏移扩到 `wave_pingpong_depth × 2` |
| GM atomic uint32 `dispatchArrival[chunk_id]` | vec1 → AIC | dispatch chunk 完成计数；GMM1 起手 busy-wait（沿用 v1 `ReadGmByPassDCache` 模式，`mega_moe_impl.h:148-157`） |
| GM atomic uint64 `l2_arrival_mask[m_block]` | vec0 → AIC | GMM1 epilogue 完成 → GMM2 启动门票；GMM2 起手等 mask 全 1（DeepGEMM 风格） |
| GM atomic uint32 `gmm2ToCombine[chunk_id]` | AIC → vec1 | GMM2 tile 完成通知 vec1 启动 combine push |
| GM magic flag `tokenPerExpertWin` | 跨 rank | dispatch 前 cross-rank 接收数同步（沿用 v1 `CrossRankSyncCntPerExpert`） |
| GM `crossRankCnt` | 跨 rank | combine 前后全局 sync（沿用 v1 `CrossRankSyncInWorldSize`） |

---

## 4. 文件清单

新建在 `mc2/mega_moe_v2/`，与 v1 同级：

```
mc2/mega_moe_v2/
├── CMakeLists.txt                              [新, 1:1 仿 v1]
├── README.md                                   [新, 后续补]
├── DESIGN.md                                   [本文档]
│
├── op_api/
│   └── aclnn_mega_moe_v2.cpp                   [新, 仿 v1 改 OP 名]
│
├── op_host/
│   ├── CMakeLists.txt                          [新]
│   ├── mega_moe_v2_def.cpp                     [新, OP_DEFINE 注册 MegaMoeV2]
│   ├── mega_moe_v2_infershape.cpp              [新, 与 v1 同 shape 推导]
│   └── op_tiling/
│       ├── mega_moe_v2_tiling.cpp              [新, ~1500 行, 含 wave/ring 参数]
│       └── mega_moe_v2.h                       [新, tilingData struct]
│
├── op_graph/
│   └── CMakeLists.txt                          [新]
│
└── op_kernel/
    ├── mega_moe_v2.cpp                         [新, 38 行薄壳]
    ├── mega_moe_v2.h                           [新, V2 主类 + Process]
    ├── mega_moe_v2_impl.h                      [新, GMM1/GMM2 wave-loop 版]
    ├── mega_moe_v2_base.h                      [新, sync 包装扩展]
    ├── mega_moe_v2_wave_scheduler.h            [新, wave 调度器 + arrival mask]
    ├── mega_moe_v2_tiling.h                    [新]
    ├── mega_moe_v2_tiling_key.h                [新]
    └── mega_moe_v2_workspace_info.h            [新, ring buffer 版]
```

**从 v1 复制（V2 独立维护，不跨目录 include、不抽公共目录）**：

V2 是临时实现，若实测优于 v1 后续会统一替换 v1，所以现阶段直接复制保持独立性最干净。复制映射：

| v1 路径 | V2 路径 |
|---|---|
| `mega_moe/op_kernel/block_epilogue_swiglu_mx_quant.h` | `mega_moe_v2/op_kernel/mega_moe_v2_block_epilogue_swiglu_mx_quant.h` |
| `mega_moe/op_kernel/mega_moe_combine_send.h` | `mega_moe_v2/op_kernel/mega_moe_v2_combine_send.h` |
| `mega_moe/op_kernel/mega_moe_base.h` | `mega_moe_v2/op_kernel/mega_moe_v2_base.h`（V2 在此基础上**扩展 sync wrappers**：CrossCoreFlag 的 `+value` 偏移要扩到 `wave_pingpong_depth × 2`） |
| `mega_moe/op_kernel/moe_init_routing_v3/` 整目录 | `mega_moe_v2/op_kernel/moe_init_routing_v3/` |

**新提取**（v1 在 `mega_moe.h` 内联，V2 抽到独立 helper 头）：
- `CopyGMToGMPerTokenV2`（`mega_moe.h:419-456`）
- `CrossRankSyncCntPerExpert`（`mega_moe.h:348-416`）
- `Unpermute`（`mega_moe.h:606-657`）
- `DispatchTokens` / token pull loop（`mega_moe.h:459-504`）

→ 落地到 `mega_moe_v2/op_kernel/mega_moe_v2_helpers.h`

---

## 5. tiling 设计（`mega_moe_v2_tiling.cpp`）

### 5.1 新增字段（在 v1 字段基础上）

| 字段 | 来源 | 公式 |
|---|---|---|
| `kExpertsPerWave` | tuning，先 hardcode = `min(2, expertPerRank)` | wave 内一组 expert 数 |
| `kNumWaves` | 推 | `ceil(expertPerRank / kExpertsPerWave)` |
| `chunkDepth` | hardcode = 4 | ring buffer 深度（每 wave 占用 1 槽，相邻 4 个 wave 同时存活） |
| `chunkM` | hardcode = 128 | ring 每槽行数（与 epilogue `singleM_=128` 对齐） |
| `numChunksPerWave` | 推 | `ceil(maxTokensPerExpert × kExpertsPerWave / chunkM)` |
| `dispatchArrivalLen` | 推 | `chunkDepth × kNumWaves × numChunksPerWave` |
| `l2ArrivalMaskLen` | 推 | 同上（uint64 数组） |
| `gmm2ToCombineLen` | 推 | 同上 |
| `commVecCount` | hardcode = 32 | 通信 vec 数（subBlockIdx_=1） |
| `epiVecCount` | hardcode = 32 | epilogue vec 数（subBlockIdx_=0） |

### 5.2 v1 字段保留 / 调整

- 保留：`m, k, n, topK, expertPerRank, epWorldSize, MoeInitRoutingV3 子 tiling`
- 调整：`maxOutputSize` 不再是 ptrA/ptrA2 的尺寸基准，仅作 dispatch ring buffer 容量上限；改为 `dispatchRingTokens = chunkDepth × chunkM × kExpertsPerWave`
- 移除：`blockNumPerEP`（V2 通信用 32 vec 而非 cube，公式语义变）；改为 `dispatchVecPerRank = commVecCount / epWorldSize`
- 移除：v1 单 `dispatchRows=2` 硬编码（V2 让 vec1 内部按 chunk 自调）

### 5.3 tilingKey

模板特化与 v1 同（`DispatchQuantMode × DispatchQuantOutType`），**新增 1 个维度**：`kExpertsPerWave`（先固定 1 个值，预留 key 字段）。

---

## 6. 风险与回退点（设计阶段就标记）

| # | 风险 | 触发条件 | 回退 |
|---|---|---|---|
| ~~R1~~ | ~~跨目录 include~~ | **已废**（用户决策：复制 v1 文件，不跨目录 include 不抽公共目录） | — |
| R2 | `MoeInitRoutingV3` 子 tiling 在 V2 不能直接复用（内部依赖 `MegaMoeTilingData` 字段） | tiling 编译失败 | 给 V2 单独 wrap 一份 `MegaMoeV2InitRoutingTiling`，转发到子模块 |
| R3 | wave-by-expert 调度下 ring buffer chunk 数太少 → cube 等 vec1 dispatch 反而比 v1 慢 | profile 发现 GMM1 起步 stall | host tiling 调大 `chunkDepth`（4→8）、或先做 dispatch 全局再启动 wave（DeepGEMM 原版做法） |
| R4 | vec1（comm）仅 32 核做 dispatch+combine 比 v1 用全 64 vec 反而带宽不够 | dispatch latency 升 | 临时调成 vec1 = 48 / vec0 = 16，或 wave 起步前/末尾让 vec0 也参与 dispatch pull |
| ~~R5~~ | ~~block_epilogue 跨目录 include 类型不匹配~~ | **已废**（用户决策：复制 v1 文件后可自由改 V2 副本以接 V2 的 tiling 类型） | — |
| R6 | 950PR 的 GM atomic uint64 mask 性能不如预期 | profile 显示 mask wait 高 | 改用 32-bit counter + 多 flag 数组 |

---

## 7. 实现拆分（任务 #4 的子任务）

设计确认后，并行启动 **5 个实现 agent**：

| Agent | 子任务 | 输入 | 输出文件 |
|---|---|---|---|
| I1 | wave 调度器 + Process 主循环 | 本设计 §3 + v1 `mega_moe.h` `Process()` | `mega_moe_v2.h`、`mega_moe_v2_wave_scheduler.h` |
| I2 | GMM1/GMM2 wave-loop impl | 本设计 §3 + v1 `mega_moe_impl.h` | `mega_moe_v2_impl.h`、`mega_moe_v2_base.h` |
| I3 | host tiling | 本设计 §5 + v1 `mega_moe_tiling.cpp` | `op_host/op_tiling/mega_moe_v2_tiling.cpp` + `mega_moe_v2.h` |
| I4 | op_def + op_api + infershape + tilingKey | 本设计 §3.1 + v1 同名文件 | `op_host/mega_moe_v2_def.cpp`、`op_host/mega_moe_v2_infershape.cpp`、`op_api/aclnn_mega_moe_v2.cpp`、`op_kernel/mega_moe_v2_tiling_key.h`、`op_kernel/mega_moe_v2.cpp` |
| I5 | CMake 接入 + workspace_info + 跨目录 include 验证 | 本设计 §4、§6 R1 | `CMakeLists.txt`、`op_kernel/mega_moe_v2_workspace_info.h`、`op_host/CMakeLists.txt`、`op_graph/CMakeLists.txt` |

I1 / I2 强耦合（共享 wave scheduler 头文件），但 API 在本设计里已固定，可并行写实现，最后 merge 时只解决 include。
I3 / I4 / I5 互相独立。

---

## 8. 验证计划（任务 #5）

1. **本机 commit**：在 `master_for_ai` 分支上 add `mc2/mega_moe_v2/` 全量，commit message：`add MegaMoeV2 — wave-based pipeline`
2. **push 到 fork**：`git push origin master_for_ai`（参考 `project_fork_ops_transformer.md`，PAT 已配置）
3. **服务器侧 pull + build**：`cd ~/gitcode/Allan_Yu/ops-transformer && git pull && bash build.sh`
4. **harness 扩展**：`~/test_mega_moe/test_mega_moe_4p.py` 增加 v2 调用分支（torch_extension 同时注册 v1 / v2，参数完全一致）
5. **对比测试**：BS ∈ {32, 128, 256, 512, 1024, 2048}，对比 v1（mega_moe）vs v2（mega_moe_v2）的 host wallclock 与 NPU profile
6. **数值正确性**：相同输入下 v2 输出 vs v1 输出的 max abs / rel diff（同条件 dummy weight 下应近 0；用真 MX-FP8 weight 时按 MX 量化误差范围比较）

---

## 9. 用户已确认的最终决策（2026-05-05）

1. ✅ **核分工**：32 cube + 32 vec(epilogue, vec0) + 32 vec(comm, vec1)，沿用 `KERNEL_TYPE_MIX_AIC_1_2`
2. ✅ **Dispatch 形态 = DeepGEMM 原版**：kernel 入口启动完整 dispatch lifecycle（push 路由 → 单次跨卡同步 → pull token chunk 滑窗），跨卡 barrier 全程仅 1 次（用 v1 `CrossRankSyncCntPerExpert` 路径），dispatch chunk 完成时 `AtomicAdd dispatchArrival[chunk_id]` 让 GMM1 在 block 粒度自旋等就绪即起步
3. ✅ **Wave 切 expert 维**（DeepGEMM 原版），每 wave `kExpertsPerWave` 个本地 expert，wave 内持久流水跑 GMM1+epilogue → GMM2+combine
4. ✅ **复制 v1 相关文件到 V2 目录**（不跨目录 include、不抽公共目录）；V2 是临时实现，若实测胜出后续将统一替换 v1

设计冻结，进入实现阶段（任务 #4，并行 agent 启动）。
