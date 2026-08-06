# 参考仓 pull 流程详解（ops-transformer_megaMoeOpt @ ababd4a92）

> 面向没接触过 `mega_moe_wave.h` 的读者。术语：**pull** = dispatch 从 peermem 接收区取数、搬进本卡 workspace。
> 所有行号基于 `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h`（impl.h 另行标注）。

---

## 0. 这个文件在算什么

MegaMoE（MoE 专家并行）：每张卡负责若干专家。每个 token 被路由到 topK 个专家（可能在其他卡上）。
一张卡上要做两件事：

1. **收**：把"其他卡路由到本卡专家"的 token 行从通信接收区（peermem）搬到本卡工作区（workspace）——这就是 **pull/dispatch**；
2. **算**：对这些行做两轮矩阵乘（GMM1 + SwiGLU + GMM2），再把结果按路由合并回原 token 行（Combine）→ 最终 Unpermute 恢复顺序。

`mega_moe_wave.h` 里的 `MegaMoeWave` 模板类就是"收 + 算 + 合"的流水调度器。整个算子按 **wave（批次）** 滚动执行，`MetaInfoCalAndDispatch` 是"收"的核心，`ProcessGmm1Wave` 是"算"的消费端。

---

## 1. 调用入口链

```
Host/Tiling（分配 workspace、tiling 参数）
  └─ 每个物理核执行 Kernel 入口 → MegaMoeWave::Process()        @2603
       ├─ 阶段1 输入准备：QuantProcessInRank（量化本卡 token）
       │              → SendMaskCal（把本卡 token 的路由 mask 推到各目标卡 peermem）  ← 真正跨卡"发"
       │              → ResetFlagList（清零 GM 里所有 flag 槽）
       │              → SyncAll<false>（AIC 等 AIV 准备完）
       ├─ 共享专家 GMM1（可选）
       ├─ CrossRankSyncInWorldSize()   ← 全卡 barrier：所有 rank 输入就绪后才能读远端数据
       ├─ 阶段2 路由专家滚动流水：ProcessRoutedExpertWaves()    @2461
       │       ├─ AIV1 分支（subBlockIdx_==1）：DispatchWaveAndGetEndCursor（规划 wave）
       │       │        → DispatchExpert（每专家：SendCntCal → UpdateGroupParams → UpdateGlobalBuffer
       │       │          → MetaInfoCalAndDispatch 全量 pull）
       │       ├─ AIC/AIV0 分支：ProcessGmm1Wave（消费 wave，内部 WaitForUpstreamReady 等 flag）
       │       ├─ ProcessGmm2Wave（按 wave 终点重放 GMM2）
       │       └─ ProcessCombineExperts（AIV1 按完整专家合并结果）
       ├─ SyncAll<true>
       ├─ 共享专家 GMM2（可选）
       └─ 阶段3：CrossRankSyncInWorldSize + Unpermute
```

硬件拓扑：一个 block = 1 个 C（AIC 矩阵计算核，真算 GMM1/GMM2）+ 2 个 V（AIV 向量核），共 28 个 block（84 核）。**subBlockIdx_** 是 block 内 V 的编号：0 = V0（AIV0，与 C 配对跑 SwiGLU 激活 + 虚拟推进 GMM2 分核）、1 = V1（AIV1，专职 pull 和 Combine）。所有核执行同一份 `Process`，靠 `g_coreType` / `subBlockIdx_` / `blockIdx_` 分流（`blockIdx_` = block 编号 0..27）。

---

## 2. 名词对照表（代码里的"黑话" ↔ 实际语义）

### A. 硬件与核角色

| 名词 | 代码位置 | 实际语义 |
| --- | --- | --- |
| AIC / AIV | `g_coreType` | 矩阵计算核(C) / 向量核(V)；一个 block = 1 C + 2 V |
| AIV0 / AIV1 | `subBlockIdx_` | block 内 V 编号（subBlockIdx_）：0=V0 配合 C 跑 SwiGLU；1=V1 专职 dispatch(pull)+combine |
| `blockIdx_` / `blockNum_` | 类成员 | block 编号 / block 总数（28）。dispatch 分核的坐标（每个 block 的 V1 参与） |
| `blockNumPerRank_` | `tilingData->blockNumPerEP` | 每个 source rank 的行段切成几片（=shard 数，默认 2） |
| `aivCoreIdx_` | 类成员 | AIV 物理核号，用于"只有 1 号核输出专家 token 数" |
| UB | — | 核上统一缓冲（Unified Buffer），mask/临时数据都放这里，容量有限需滚动复用 |
| pingpong | `gmm1PingPongIdx_` | UB 双缓冲：GMM1 算一块、SwiGLU 读另一块，交替使用 |

### B. 数据与内存布局

| 名词 | 代码位置 | 实际语义 |
| --- | --- | --- |
| rank / `worldSize_` | 类成员 | 参与通信的卡数（如 4/16）。"source rank"= 给当前专家发 token 的那张卡 |
| peermem | `params_.peermemInfo.maskRecvPtr` | 跨卡通信接收区（每卡一块）。SendMaskCal 写入，dispatch 读取；含 mask、count、token 三部分 |
| route mask | `maskSlotSize_` | 位图：每个 (token, topk 槽) 是否路由到 (本卡, 某专家)；每 (expert, source rank) 一个槽 |
| count | mask 槽内 `maskAlignSize_` 之后 | 某 source rank 发给当前专家的 token 行数 |
| `cumsumInfoTensor_` | UB 张量 | 前缀和 C[expert·worldSize+rank] = 累计行数；把"行"映射到 workspace 行号的依据 |
| topkIndex / tokenIndex | `validTopkIndexTensor_` | route 数组的全局索引 / 原始 token 行号（= topkIndex ÷ topK_） |
| match ordinal | 局部变量 | 某 rank 段内"命中项的第几个"，与段内行号一一对应 |
| batch | `recvRouteBatchCount_` | route 数组分批扫描，每批 `recvRouteItemsPerBatch_` 个元素，mask 按批搬（省 UB） |
| workspace | `dispatchRevDataPtr` | GM 中排布好的 token 行区（pull 的输出、GMM1 的输入） |
| `maxOutputSize_` | 类成员 | workspace 可写行数上限，超出截断（防止越界） |
| `moeExpertPerRank_` | 类成员 | 本卡负责的专家个数 |

### C. 调度概念（wave 模板的核心）

| 名词 | 代码位置 | 实际语义 |
| --- | --- | --- |
| expert / `expertIdx` | 游标字段 | 本卡专家编号 0..moeExpertPerRank_-1；每个专家一份独立权重 |
| `ExpertRowCursor` | `{expertIdx, rowOffsetInExpert}` | 半开游标：指向"下一个还没处理的行"；专家处理完规范化为 (nextExpert, 0) |
| `ExpertLoopState` | `{problemShape, baseOffset, ...}` | 一个专家的完整计算状态：M/N/K、各矩阵地址偏移、前序专家累计行数 |
| `GMMAddrInfo` | 地址集合 | 一次 GMM 调用的入参：A/B/C 矩阵地址 + flag 指针等 |
| **M Group** | `GMM1_TILE_M = 256` | 256 行一组。GMM1 一个计算 tile 处理一组；**flag 按组（wave 槽）计数** |
| **wave** | `mGroupsPerWave_` | 处理批次。GMM1 每轮消费 `mGroupsPerWave_` 个 M Group；wave 终点可落在热点专家内部 |
| `mGroupsPerWave_` | `GetMGroupsPerWave()` | 每轮 wave 的组数预算（由 GMM1/GMM2 铺核需求算，不是 256 行本身） |
| tile / `mLoc` | `WaitForUpstreamReady` | AIC 上的计算分块；`mLoc` = 当前 tile 的 M 向起点，`waveIdx = mLoc / tileM` |
| **flag** | `dispatchToGmm1Flag` | GM 里的计数信号：dispatch 每 pull 一行就 AtomicAdd 1；GMM1 轮询等满才开算 |
| flag 槽 | `waveIdx * INT_CACHELINE` | 每个 M Group 一个 int32 槽，槽间距 64B（防多核伪共享） |
| `expertMGroupOffset` | `UpdateGlobalBuffer` | 专家内"已完成的分组数"；分段调用下标从 0 起，加它才能定位专家内全局 flag 槽 |
| `dispatchFlagSlotsPerExpert_` | 类成员 | 每个专家预留的 flag 槽数 |
| 预取 | `preparedWaveEndCursor` / `TopkWeightsPrefetch` | AIV1 提前规划下一 wave 终点 / 权重随 token 提前搬 |
| shard | `dispatchShardIdx` | 把某 source rank 的行段按行切成 `blockNumPerRank_` 片，一片 = 一个分核任务 |

### D. 流程与同步

| 名词 | 位置 | 实际语义 |
| --- | --- | --- |
| `SendMaskCal` | 阶段1 | 源卡把本卡 token 的 mask 推到各目标卡 peermem（真正跨卡"发"） |
| `SendCntCal` | DispatchExpert ① | 读 peermem 里各 source rank 的 count → 算前缀和 → 写 GM 并 AtomicAdd sendCntFlag 通知 AIC |
| `DataCopy` | 指令 | GM ↔ UB 搬运 |
| `GatherMask` | MetaInfoCalAndDispatch | 按 mask 压缩 route 索引：输入全量 route 数组，输出命中项的索引数组 + 命中数 |
| `CopyGMToGMPerToken` | MetaInfoCalAndDispatch | 从 peermem 的 token 区把命中行 **pull** 到 workspace（MTE3 搬数） |
| `WaitForUpstreamReady` | impl.h L165 | 计算前轮询 flag（`ReadGmByPassDCache` + 100 cycle 空转），等满 target 才继续 |
| `AtomicAdd` | 指令 | GM 原子加；多核分片累加同一 flag 槽的安全手段 |
| 事件同步 | `SyncFuncStatic<...>` | 流水线事件等待（如 MTE2_V：GM→UB 搬完；V_S：向量算完；MTE3_S：搬数落 GM） |
| `CrossRankSyncInWorldSize` | @2135 | 全卡 barrier（读远端数据前 / Combine 后） |
| SwiGLU | epilogue | GMM1 输出的激活 + 量化，结果写给 GMM2 的输入区，并累加 swigluToGmm2Flag |
| Combine / Unpermute | 阶段2/3 | 按路由把专家结果合并回 token 行 / 恢复原始 token 顺序 |

---

## 3. 参考仓 pull 流程（基于上面名词）

### 3.1 前置：数据是怎么到 peermem 的（不是 dispatch 干的）

1. `SendMaskCal`：每张卡把自己 token 的路由 mask、count、token 数据写进**各目标卡的 peermem**（跨卡通信）。
2. `ResetFlagList`：清零所有 flag 槽。
3. `CrossRankSyncInWorldSize`：全卡 barrier，保证 peermem 数据完整，dispatch 才开始读。

### 3.2 AIV1 的 pull 主链路（每专家一次，DispatchExpert @1028）

对每个专家（localExpertId）：

```
① SendCntCal：从 peermem 读该专家 worldSize_ 个 count
   → 算前缀和写 cumsumInfoTensor_（UB，所有核共享读）
   → 写 expertRevNums 到 GM（AIC/AIV0 读 M 用）
   → AtomicAdd(sendCntFlag) 通知 AIC：count 就绪
② UpdateGroupParams<GMM1>：推进专家状态
   → M = 该专家总行数；baseOffset 里各矩阵偏移前移；IDX_FLAG_OFFSET 定位专家 flag 基址
③ UpdateGlobalBuffer<GMM1>：把地址打包进 GMMAddrInfo
   → dispatchToGmm1Flag = flagDispatchToGmm1Ptr + 专家基址 + expertMGroupOffset*64B
④ MetaInfoCalAndDispatch（单专家版 @1282）：全量 pull 该专家全部行（见 3.3）
```

### 3.3 MetaInfoCalAndDispatch：全量 pull 一个专家（rank-shard 分核）

1. **分核（shard）**：任务 = (source rank, rank 内 shard) 展平，共 `worldSize_ × blockNumPerRank_` 个；
   28 个 block 的 V1 按 `dispatchShardIdx += blockNum_` 领取（一个 V1 可能领多片）。每个 shard 是某 rank 段内
   `ceil(段行数 / blockNumPerRank_)` 行的一个连续区间。
2. **定位**：cumsumInfo 得段起点行号；`maxOutputSize_` 截断得可写行数。
3. **逐 batch 扫描**（省 UB）：每批搬 mask 切片 → `GatherMask` 压缩出命中项全局索引 + 命中数
   → 与"本核负责的 ordinal 区间"求交集。
4. **pull**：交集内的每行，`CopyGMToGMPerToken` 从 peermem token 区搬到 workspace 目标行。
5. **喂 flag**：每 rank 段搬完，先 `MTE3_S` 保证数据落 GM，再把本核区间按 256 行拆到
   专家内 wave 槽，`AtomicAdd(槽, 重叠行数)`。

### 3.4 wave 规划层（DispatchWaveAndGetEndCursor @1051）

- 只推游标，**不 pull**（注释原文："这里不再触发任何数据发送"）。每轮预算 `mGroupsPerWave_` 个 M Group：
  `segmentRowCount = min(专家剩余行, 剩余组数 × 256)`。
- 专家首次进入 wave（`rowOffsetInExpert == 0`）时才触发 3.2 的全量 pull；热点专家跨 wave 只推进游标。
- **通信终点（专家）≠ 计算终点（wave）**：数据一次 pull 完，计算按 wave 分批。

### 3.5 GMM1 消费与 flag 握手（AIC/AIV0，ProcessGmm1Wave @2296）

1. 每轮消费 `mGroupsPerWave_` 个 M Group（与 3.4 的规划同口径）。
2. 每个行段：`UpdateGlobalBuffer<GMM1>`（flag 基址带 `expertMGroupOffset`）→ `GroupMatmulWithSwigluQuant`。
3. 每个 tile 算之前 `WaitForUpstreamReady`（impl.h L165）：
   - `waveIdx = mLoc / tileM`（专家内第几个 256 行组）
   - `target = min(tileM, m - mLoc)`（尾段不足 256 按实际行数）
   - 轮询 `dispatchToGmm1Flag + waveIdx*64B == target` → 才允许读 workspace 开算。
4. SwiGLU 完成 → 累加 `swigluToGmm2Flag`（同样按 wave 槽）→ 喂 GMM2；GMM2 完成后专家粒度发布 ready → AIV1 Combine。

**flag 槽为什么天然对齐**：dispatch 按"专家内行号 ÷ 256"累加槽；GMM1 的 `waveIdx = mLoc / tileM`
是专家内全局分组号（`UpdateGlobalBuffer` 已加 `expertMGroupOffset`）——两边算的是同一个数。

### 3.6 与 GMM1 的交互总表（flag 协议）

| flag | 写入方 | 读取方 | 粒度 | 语义 |
| --- | --- | --- | --- | --- |
| sendCntFlag | AIV1 SendCntCal（AtomicAdd 1） | AIC/AIV0 UpdateGroupParams | 每 AIC 核 64B | 专家 count 已写入 GM |
| dispatchToGmm1Flag | AIV1 MetaInfoCalAndDispatch（AtomicAdd 行数） | AIC GMM1 WaitForUpstreamReady | 专家内 wave 槽 64B | 该 wave 已 pull 完成的行数 |
| swigluToGmm2Flag | AIV0 SwiGLU（AtomicAdd tile 数） | GMM2 WaitForUpstreamReady | 专家内 wave 槽 64B | 该 wave SwiGLU 完成 |
| gmm2 ready | GMM2 专家最后一段完成 | AIV1 Combine WaitGmm2Ready | 专家 slot | 专家可整体 Combine |

---

## 4. 参考仓 pull 的特点（一句话）

**按专家全量 pull、按 wave 分批计算**：每个专家首次进入时一次拉完（rank-shard 分核，负载随
rank 段分布）；后续专家要等 AIV1 的轮次（每轮"当前 wave + 预取下一 wave"）才拉，
GMM1 可能因 flag 未满而空转等待。
