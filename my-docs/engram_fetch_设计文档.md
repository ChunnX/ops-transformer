# engram_fetch 算子设计文档（含全量本 NPU 模式 / fetch_mode）

> 覆盖算子：`engram_fetch`（推理 / 训练前向）、`engram_fetch_wait`（RDMA 排空）、`engram_fetch_grad`（训练反向）。
> 本文前半部分为算子背景与 LLM 使用场景（含既有 **切片跨 NPU 通信** 模式的完整设计），后半部分为本次新增
> **fetch_mode 全量本 NPU（replicated / local）模式** 的需求背景、方案设计与接口/实现细节。
> **第 4～10 章为本仓算子的 tiling 策略、逐字节内存布局、AI Core 数据分配与 bank 冲突规避详设**，
> 所有布局公式与常量均以 kernel / tiling 源码为单一事实来源（引用处标注文件:行号）。

---

## 目录

1. [LLM 上下文背景：什么是 engram，为什么需要 engram_fetch](#1-llm-上下文背景什么是-engram为什么需要-engram_fetch)
2. [整体架构与数据面](#2-整体架构与数据面)
3. [通信底座与上下文（EngramCommContext）](#3-通信底座与上下文engramcommcontext)
4. [tiling 总体策略（host 侧）](#4-tiling-总体策略host-侧)
5. [fetch_mode=0 推理内核详设（EngramFetchArch35）](#5-fetch_mode0-推理内核详设engramfetcharch35)
6. [fetch_mode=0 训练内核详设（EngramFetchTrainArch35）](#6-fetch_mode0-训练内核详设engramfetchtrainarch35)
7. [engram_fetch_wait / engram_fetch_grad 详设](#7-engram_fetch_wait--engram_fetch_grad-详设)
8. [新需求：arch35 全量本 NPU 模式（fetch_mode=1）](#8-新需求arch35-全量本-npu-模式fetch_mode1)
9. [local 模式方案设计（含逐字节布局）](#9-local-模式方案设计含逐字节布局)
10. [多核数据分配与 bank 冲突规避](#10-多核数据分配与-bank-冲突规避)
11. [接口变更（aclnn / PTA）](#11-接口变更aclnn--pta)
12. [PTA（torch_extension）适配](#12-ptatorch_extension适配)
13. [边界与约束](#13-边界与约束)
14. [验证计划与结果](#14-验证计划与结果)

---

## 1. LLM 上下文背景：什么是 engram，为什么需要 engram_fetch

### 1.1 大模型推理/训练中的"权重不常驻"问题

LLM（大语言模型）的参数量已达数百 GB（如 DeepSeek-V3 约 671B 参数，BF16 下 1.3+ TB），远超单卡 HBM
容量（950 单卡数十 GB 量级）。业界主流解法之一是 **权重流式加载（weight streaming）**：

- 权重不全部常驻 NPU HBM，而是存放在 **host 侧 pinned memory**（或更大的主机内存层级）；
- 每步计算前，仅把 **本步实际需要的权重行** 按索引拉到 NPU 上参与计算；
- 计算与加载异步重叠，用"时间换空间"跑超大模型。

在 Ascend 的 mc2 算子族中，一条权重行（宽 `hiddenSize`、128 对齐）称为一个 **engram（记忆条目）**。
engram 全局按行编号（global entry index），构成全局索引空间。

### 1.2 engram_fetch 三件套在 LLM 数据面的位置

以 PD 分离 / 专家并行推理为典型场景：

```
 indices(本步需要的权重行号, int32)
      │
      ▼
 engram_fetch(with_grad=0)   ── 推理：RDMA ReadNbi 异步单边读，不阻塞后续算子下发
      │ (fetched 异步填充中)
      ▼
 <其它算子继续下发>           ── 通算/搬数重叠
      │
      ▼
 engram_fetch_wait           ── Drain 排空 RDMA 读通道，fetched 数据保证可见
      │
      ▼
 计算（GEMM 等）消费 fetched
```

训练场景（with_grad=1）则记录反向上下文，反向阶段由 `engram_fetch_grad` 把梯度沿原路回传并聚合：

```
 前向: engram_fetch(with_grad=1) → fetched + (perm, sendCounts, recvCounts, recvLocalEntry, numRecv)
 反向: engram_fetch_grad(gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv)
       → (gradUnique, uniqueLocalEntry, numUnique)   # 去重 + FP32 原子加聚合，供优化器更新权重表
```

### 1.3 既有部署形态：全局表按 rank 切片

- 全局 engram 表（numEntriesGlobal 行）按 `ownerRank = index / numEntriesPerRank` 均匀切片，
  每个 rank 本地只持有 `numEntriesPerRank × hiddenSize` 的分片（存于 host pinned + 设备映射内存）；
- 任一 rank 的 indices 可能落在任意 owner rank 上 → **取数必须跨 NPU 通信**：
  - 推理：本 rank 对远端 rank 的注册内存做 RDMA 单边读（Urma `ReadNbi`，攒批 `BatchCommit`）；
  - 训练：通信窗口 a2a（先交换索引请求，再回传权重行），反向再做一次逆向 a2a。

该形态内存最省（每卡仅 1/N 权重），但 **每次取数都有跨卡带宽与时延开销**，且依赖卡间 RDMA 链路。

---

## 2. 整体架构与数据面

```
┌─────────────────────────── 每 rank ───────────────────────────┐
│  host pinned mem (2MB 对齐, num_cpu_bytes)                    │
│   └─ engram 分片表 (fetch_mode=0) / 全量表 (fetch_mode=1)      │
│        │ aclrtHostRegisterV2(ACL_HOST_REG_MAPPED) + HcclCommMemReg
│        ▼                                                      │
│  device 映射地址 (commBuffer[rankId] / localStorageAddr)       │
│                                                               │
│  comm_context (int32 tensor, 由 PTA 生成下刷)                  │
│   └─ EngramCommContext{rankId, rankSize,                       │
│        commBuffer[1024]   // 各 rank 表基址(设备映射地址)        │
│        hcommHandle[2048]  // 各 rank RDMA 通道句柄              │
│        channelsPerRank}                                        │
└───────────────────────────────────────────────────────────────┘
```

- kernel 侧通过 `comm_context` 拿到 rankId/rankSize/各 rank 表基址/通道句柄；
- 推理 local 命中读 `commBuffer[rankId]`；训练 local 读 `localStorageAddr`（int64 标量张量，同为
  host pinned 的设备映射地址）。

---

## 3. 通信底座与上下文（EngramCommContext）

### 3.1 结构体逐字节布局（`engram_fetch_utils.h:48-54`，以 int32 视角下刷至 GM）

| 偏移 | 大小 | 字段 | 类型 | 说明 |
|---|---|---|---|---|
| 0x0000 | 4B | `rankId` | uint32 | 本 rank 编号 |
| 0x0004 | 4B | `rankSize` | uint32 | 通信域 rank 数 W（2..1024） |
| 0x0008 | 8192B | `commBuffer[1024]` | uint64×1024 | 各 rank **表基址**（host pinned 注册内存的设备映射地址）；训练模式下 `commBuffer[rankId]` 同时是本 rank 的 **a2a 通信窗口基址**（见 §6.3） |
| 0x2008 | 16384B | `hcommHandle[2048]` | uint64×2048 | 各 rank RDMA 通道句柄，按 `rank * channelsPerRank + ch` 索引（每 rank 最多 2 条：发送/接收通道） |
| 0x6008 | 4B | `channelsPerRank` | uint32 | 每 rank 通道数（≤ `MAX_CHANNELS_PER_RANK=3`） |
| **0x600C** | **24588B** | 结构体总大小 | | 即 6147 个 int32（PTA 下刷张量按此规模向上取整分配） |

> local（fetch_mode=1）模式下 csrc 仅填 `commBuffer[rankId]`（本 rank 表基址），其余表项为 0，
> `hcommHandle` 全 0 —— kernel local 路径不读任何通道句柄。

### 3.2 通信原语与通道获取

- 通信原语：`Hcomm<COMM_PROTOCOL_UBC_CTP>`（Urma 统一 RDMA 接口）；
  推理 `ReadNbi/BatchCommit`（批量 WQE）；训练 `WriteNbi/WriteWithNotifyNbi/Drain`；
- 通道获取（PTA csrc）：`HcclRankGraphGetLinksFunc` 找 UBC_CTP/UBC_TP/UBG 链路 →
  `HcclChannelAcquireFunc`（AIV engine）→ `HcclChannelGetRemoteMemsFunc` 换远端表基址；
- **训练 a2a 窗口**：窗口本体在 `commBuffer[rankId]`（本 rank 注册内存的设备映射地址）上，
  窗口大小 = `comm_buffer_size` attr（PTA from `HcclGetHcclBufferFunc` 查询值），见 §6.3 逐字节分区。

---

## 4. tiling 总体策略（host 侧）

### 4.1 tiling 流水（`engram_fetch_tiling.cpp:655-720` / `engram_fetch_grad_tiling.cpp:822-875`）

```
EngramFetchTilingFunc / EngramFetchGradTilingFunc
  ├─ 0. 解析 attrs（with_grad / fetch_mode 越界校验 ∈ [0,1]，否则 GRAPH_FAILED）
  ├─ 1. CheckInterfaceTensor：tensor ptr/dtype/shape/format 校验
  │     （fetch：hidden 128 对齐、fetched (T,H)、perm (T,)、numRecv (1,)…；
  │       grad：sendCounts.dim0 % 8 == 0、rankSize = dim0/8 ≤ 1024、
  │             numEntriesPerRank×rankSize ≤ INT32_MAX（local 模式仅校验自身，§9.5））
  ├─ 2. SetPlatformInfo：aivNum ← GetCoreNumAiv；blockDim ← CalcTschBlockDim(aivNum, 0, aivNum)
  │     （fetch/wait/grad 一致：纯 AIV 启动，AIC=0）
  ├─ 3. SetTilingData：填充下表字段（含 ubSize ← GetCoreMemSize(UB)）
  ├─ 4. SetTilingKey：按 (with_grad, fetch_mode) 派生 tiling key（§4.3）
  └─ 5. SetWorkSpace：按模式计算 workspace（§4.5）
```

### 4.2 tiling data 字段全表

**EngramFetchTilingData（`engram_fetch_tiling_data.h:21-34`，kernel/host 共享）**

| 字段 | 类型 | 来源 | 用途（kernel 侧消费者） |
|---|---|---|---|
| `numTokens` | int64 | indices.dim0 | token 总数 T；分批/分核范围 |
| `numEntriesPerRank` | int32 | attr `num_entries_per_rank` | mode=0：本 rank 分片行数（ownerRank = idx / numEntriesPerRank）；mode=1：**全表行数**（entry 直接索引本表） |
| `hiddenDim` | int64 | attr `hidden_size` | 行元素数（128 对齐校验） |
| `hiddenBytes` | int64 | hiddenDim × dtypeSize | GM 行字节（LocalCopySlice 每行长度、token 槽容量分母） |
| `aivNum` | uint32 | 平台 GetCoreNumAiv | workspace 中 per-core scratch 区的维度 |
| `ubSize` | uint64 | 平台 GetCoreMemSize(UB) | kernel 侧 UB 预算（推导 indicesBatchSize_） |
| `rankSize` | uint32 | 训练：sendCountsOut.dim0（见 §4.6 已知偏差） | 通信域 W（kernel 实际读 ctx->rankSize） |
| `numMaxTokensPerRank` | int64 | attr | 训练：slotSize_ 推导（§6.4 区 4/5） |
| `totalRecv` | int64 | 训练：recvLocalEntryOut.dim0 (= maxR) | 接收上界（窗口 token 区容量校验） |
| `commBufferSize` | int64 | attr | 训练：**通信窗口总大小** winSize_（§6.3） |
| `numSfPacks` | int64 | fetchedSf.dim1 | 推理 sf gather：sfBytes = numSfPacks × sfElemSize |
| `sfElemSize` | int64 | sf dtype size | 同上 |

**EngramFetchGradTilingData（`engram_fetch_grad_tiling_data.h:42-55`）**

| 字段 | 类型 | 来源 | 用途 |
|---|---|---|---|
| `numTokens` | int64 | gradFetched.dim0 | UnsortGrad 的行数（local 模式闲置） |
| `numEntriesPerRank` | int32 | attr | 排序 maxValue（mode=0 需 ×W ≤ INT32_MAX；mode=1 即全表行数） |
| `hiddenDim` / `hiddenBytes` | int64 | gradFetched.dim1 × dtypeSize | 行宽；gradBuf/cast/accum 尺寸推导 |
| `aivNum` | uint32 | 平台 | per-core scratch 维度 |
| `ubSize` | uint64 | 平台 | sortPool UB 预算 |
| `rankSize` | uint32 | sendCounts.dim0 / 8 | 通信域 W（sdispls/rdispls 区大小） |
| `totalRecv` | int64 | recvLocalEntry.dim0 (= maxR) | 排序容量上界 maxSortCount；recvGrad 区大小 |
| `commBufferSize` | int64 | attr | a2a 收发缓冲对齐校验 |
| `inputDtype` / `outputDtype` | int32 | gradFetched / gradUniqueOut dtype | 决定 castFp32Buf 是否分配（§7.5） |
| `gradSubBatch` | uint32 | tiling 常量派生 | unique 阶段每批 grad 行数（≤ GRAD_SUB_BATCH=8） |

### 4.3 tiling key 派生（模板参数实例化）

fetch（`engram_fetch_tiling_key.h`，`ASCENDC_TPL_UINT_DECL(EngramFetchMode, 8bit, list)`）：

| with_grad | fetch_mode | tiling key | 内核类 |
|---|---|---|---|
| 0 | 0 | 0 `ENGRAM_FETCH_DEFAULT_MODE` | EngramFetchArch35 |
| 1 | 0 | 1 `ENGRAM_FETCH_TRAIN_MODE` | EngramFetchTrainArch35 |
| 0 | 1 | 2 `ENGRAM_FETCH_LOCAL_MODE` | **EngramFetchLocalArch35（新）** |
| 1 | 1 | 3 `ENGRAM_FETCH_TRAIN_LOCAL_MODE` | **EngramFetchTrainLocalArch35（新）** |

grad（`engram_fetch_grad_tiling_key.h`）：0 = DEFAULT（不变）/ 1 = **LOCAL（新）**。

kernel 入口按 `if constexpr (EngramFetchMode == …)` 编译期裁剪，同一 `.o` 内含全部 4（或 2）个
实例（二进制 `kernelList` 中的 `_0.._3` / `_0.._1`，见 §14.2 编译核验）。

### 4.4 blockDim / 核分配策略

| 算子 | blockDim | 说明 |
|---|---|---|
| engram_fetch（4 模式） | `CalcTschBlockDim(aivNum, 0, aivNum)` | 纯 AIV；核内按角色/区间再划分（§5.3 / §6.2 / §9.2 / §9.3） |
| engram_fetch_wait | 同上 | 与 fetch 推理一致的核分配，仅 Drain |
| engram_fetch_grad（2 模式） | `CalcTschNumBlocks(aivNum, 0, aivNum)` | 纯 AIV；核角色见 §7.3 |

### 4.5 workspace 尺寸计算（tiling 侧公式，与 kernel 侧布局一一对应）

**fetch：**

| 模式 | workSpaces[0] |
|---|---|
| 推理 DEFAULT | `SYSTEM_NEED_WORKSPACE`（16MB，框架保留） |
| 推理 LOCAL / 训练 LOCAL | `SYSTEM_NEED_WORKSPACE`（无 a2a，无自管区） |
| 训练 DEFAULT | `Align2MB( Σ 区1..10 ) + 16MB`，区定义见 §6.4 |

**grad（DEFAULT 与 LOCAL 同公式，local 复用布局、区 1/2/3 闲置）：**

```
wsGradSorted = numTokens × hiddenBytes                    // 区1 UnsortGrad 输出
wsRecvGrad   = totalRecv × hiddenBytes                    // 区2 a2a 接收梯度
wsSdispls/wsRdispls = W × 32B                             // 区3/4 逆向 a2a 偏移表
wsCounterScratch = aivNum × 32B                           // 区5 per-core 流控 scratch
wsFlagScratch = 32B                                       // 区6 barrier 标志
wsSegCount/wsCoreStart = Align32(aivNum × 4B)             // 区7/8 unique 阶段分段表
wsSortTotal  = 3×Align32(totalRecv×4B)                    // 区9 排序 tempVal/tempIdx/companion
             + ceil(totalRecv/4096)×256×4B                //    直方图 hist（tile 数×桶）
             + 256×4B + aivNum×256×4B                     //    prefix + coreSums
             + ceil(totalRecv/4096)×256×4B                //    桶偏移 offsets
             + aivNum×2×Align32(8192×4 + 256×28 + 32)     //    SIMT gather staging ×2
wsTotal = Align2MB(Σ) + 16MB
```

### 4.6 已知偏差（记录，非本分支引入）

fetch tiling `engram_fetch_tiling.cpp:545` 以 `sendCountsOut.dim0`（= W×8，32B 步长契约）直接赋
`tilingData.rankSize`，未除以 8；kernel 侧 rankSize 一律读 `ctx->rankSize`（GM），仅 tiling 侧
workspace 计算使用该值（偏大约 8 倍，功能无害）。grad tiling 已正确 `dim0/8`（`engram_fetch_grad_tiling.cpp:277`）。

---

## 5. fetch_mode=0 推理内核详设（EngramFetchArch35）

### 5.1 数据流与 indices 分批

`ownerRank = index / numEntriesPerRank`，AIV 按 rank 分组（§5.3）：

- **本地命中**（ownerRank == rankId）：`LocalCopySlice`（relayQue 双缓冲 32KB，GM→UB→GM）；
- **远端命中**：`hcomm_.ReadNbi(channelHandle, dst=fetched行, src=远端表行, hiddenBytes)` 异步单边读；
- indices 超过 UB 批容量时分批：`indicesBatchSize_ = min(T, availableUb / bytesPerIndice)`，
  `bytesPerIndice = 4+4+4×3+1 = 21B`（indices + tokenIdxInRank + rankIDs/positions/divisor + mask 近似）。

### 5.2 UB 逐字节布局（`engram_fetch_arch35.h:169-213`，分配顺序即地址顺序）

| # | buffer | 大小公式 | 用途 |
|---|---|---|---|
| 1 | `hcommBuf_` | 512B（`HCOMM_INIT_SIZE`） | Hcomm 上下文 |
| 2 | `relayQue_` | 2 × 32KB（`RELAY_BUFFER_NUM×TILE_BYTES`，TQueBind VECIN/VECOUT） | 本地拷贝中继双缓冲 |
| 3 | `hcommBatchBuf_` | 8KB（128 WQE × 64B，`ENGRAM_BATCH_BUFFER_BYTES`） | `MakeBatchHandle/BatchCommit` 批量 WQE 暂存 |
| 4 | `rankCountsBuf_` | Align32(W×4B) | 每 owner rank 命中计数 |
| 5 | `rankOffsetsBuf_` | Align32(W×4B) | 每 rank 分组起始偏移 |
| 6 | `commBufferBuf_` | Align32(W×8B) | ctx commBuffer 的 UB 镜像 |
| 7 | `hcommHandleBuf_` | Align32(W×channelsPerRank×8B) | 通道句柄 UB 镜像 |
| 8 | `indicesBuf_` | Align32(batch×4B) | 当前批 indices |
| 9 | `tokenIdxInRankBuf_` | Align32(batch×4B) | index % numEntriesPerRank |
| 10 | `rankIDsBuf_` / `positionsBuf_` / `divisorBuf_` / `gatherTmpBuf_` | 各 `compareCntMax×4B`，`compareCntMax = Align256(batch×4B)/4` | 向量化除法/比较/前缀扫 |
| 11 | `maskBuf_` | Align32(ceil(compareCntMax/8)) | Compare 掩码 |

UB 预算：`availableUb = ubSize − (区1..7) − 8KB(UB_RESERVED_SIZE)`；sf 路径共用 relayQue（sfBytes =
numSfPacks×sfElemSize ≪ 32KB）。**所有 buffer 以 32B 对齐分配**（TBuf 分配器保证），向量比较段按
256B 对齐补齐（`ALIGNED_LEN_256`）——见 §10.3。

### 5.3 分核策略（`engram_fetch_arch35.h:291-296`，`GetCoreAssignment`）

- `totalBlocks ≥ W`：**连续分组**——`GetCoreAssignment`（`engram_fetch_utils.h:62-79`）把
  totalBlocks 个 AIV 按 `base = totalBlocks/W, remainder = totalBlocks%W` 切成 W 组（前 remainder
  组多 1 核），每核只服务 `assignedRank` 一张远端表 → 同一远端表的 RDMA 请求集中在少数核，
  通道句柄局部性好；
- `totalBlocks < W`：**跨步分组**——`for (ownerRank = aivId; ownerRank < W; ownerRank += totalBlocks)`
  均摊远端表；
- 核内 token 维度不再切分（按批流水：每核扫全量 indices 批，仅处理命中自己 ownerRank 的元素）。

### 5.4 RDMA 攒批（`engram_fetch_arch35.h:358-375`）

ReadNbi 写入 `hcommBatchBuf_` 的 WQE 累积至 `ENGRAM_BATCH_CAPACITY=128` 条即
`MakeBatchHandle + BatchCommit` 一次下发（批量 doorbell，减少寄存器写）；批未满时在批切换/收尾处
冲刷。挂起句柄上限 `MAX_PENDING_HANDLES=8`（grad 侧 utils）。

---

## 6. fetch_mode=0 训练内核详设（EngramFetchTrainArch35）

### 6.1 五阶段流水（核内 `SyncAll<true>` 全核栅栏分隔）

1. `CountAndScatterPhase`：统计每 owner rank 的 token 数 → sendCounts（32B 步长布局，见 §10.3-2）；
2. `ComputeSdispls`：send 侧偏移表（workspace 区 1）；
3. `GatherAndSortPhase`：按 rank 分组重排 indices → sortedIndices + perm（分组位置→原始位置）；
4. `SendCountPhase` + `ExchangeIndices`：a2a 交换索引请求（4 槽位窗口 + 读写计数器背压，
   `WriteWithNotifyNbi` 带计数通知，5s 超时保护）→ recvLocalEntry / recvCounts；
5. `LocalReadTable` + `ExchangeToken`：本 rank 查本地表得权重行，再 a2a 回传 → recvData；
   `ReorderAndSaveCtx`：按 perm 还原到 fetched 原始 token 序，写 numRecv。

### 6.2 核角色划分（`engram_fetch_train_arch35.h:395-478`）

```
numSendCores = max(1, aivNum/2)              // 若 ≥ W 则向下取 W 的倍数
numRecvCores = totalBlocks − numSendCores
groupSize    = clamp(numSendCores/W, 1, channelsPerRank)   // 每 (rank, 通道) 的发送组数 = totalSlots
rankCores    = min(totalBlocks, W)
tokenGroups  = ceil(totalBlocks / rankCores)
myOwnerRank  = aivId % rankCores             // 交错：偶数轮次覆盖全部 rank
myTokenGroup = aivId / rankCores
isSender   = aivId < numSendCores
isReceiver = aivId ≥ numSendCores
isFlagCore = aivId == totalBlocks−1 且 totalBlocks > 1   // 专职窗口状态维护
```

- **sender**（前半核）：按 `myOwnerRank` 把分组后的索引/权重行写入各远端窗口槽；
- **receiver**（后半核）：按 `myOwnerRank × tokenGroups + myTokenGroup` 交错认领 token 槽，
  轮询读计数器搬运回本地 fetched；
- **flag core**：维护跨 rank 窗口状态位。

### 6.3 GM 通信窗口逐字节布局（`InitWinLayout`，`engram_fetch_train_arch35.h:414-456`）

窗口基址 = `ctx->commBuffer[rankId]`（本 rank 注册内存设备映射地址），总大小 `winSize_ =
commBufferSize`。**7 段顺序布局**（N=W，G=groupSize，SLOT 步长 32B）：

| 区 | 偏移起点 | 大小 | 内容 |
|---|---|---|---|
| ① countFlag | 0 | N×32B | 每 (rank) 一个 32B 状态槽 |
| ② indicesW | N×32 | N×32B | indices 写计数器（每 rank 一槽） |
| ③ indicesR | 2N×32 | N×32B | indices 读计数器 |
| ④ tokenW | 3N×32 | N×32B×G | token 写计数器（每 (rank, group) 一槽） |
| ⑤ tokenR | 3N×32+N×32×G | N×32B×G | token 读计数器 |
| ⑥ sendCount | … | N×32B | 每 rank 的 sendCount（32B 步长契约） |
| ⑦ indices 数据区 | fixedSize 后 | `indicesArea` | 按 (rank, slot) 交错分槽 |
| ⑧ token 数据区 | ⑦ 之后 | `tokenArea` | 按 (rank, group) 交错分槽 |

```
fixedSize    = 4×(N×32B) + 2×(N×32B×G)                       // 区①②③⑥ + ④⑤
remaining    = winSize − fixedSize
indicesArea  = Ceil(remaining/50) 向上取整到 N×4×32B 粒度      // INDICES_RATIO=50 → indices:token = 1:49
tokenArea    = remaining − indicesArea 向下取整到 N×G×hiddenBytes 粒度
indicesSlotSize = indicesArea / N / 4                        // 每槽字节（NUM_SLOTS=4 槽/rank）
tokenSlotSize   = tokenArea / N / G
maxIndicesPerSlot = indicesSlotSize / 4                      // 槽容量（int32 个数）
maxTokensPerSlot  = tokenSlotSize / hiddenBytes               // 槽容量（行数）
```

槽寻址：indices 槽 `(rank, slot)` 交错排列 → 相邻槽属于**不同 rank**，多 sender 并发写天然错开
32B 行（§10.3-3）；token 槽同理按 (rank, group) 交错。

### 6.4 本地 workspace 逐字节布局（`InitWorkspaceLayout`，`engram_fetch_train_arch35.h:496-541`）

基址 = kernel workspace（tiling §4.5 公式同源）。**10 区顺序布局**：

| 区 | 基址偏移 | 大小 | 内容 |
|---|---|---|---|
| 1 | 0 | Align32(W×8B) | `sdisplsGM_` send 偏移表 |
| 2 | 区1 后 | Align32(W×8B) | `rdisplsGM_` recv 偏移表 |
| 3 | 区2 后 | Align32(T×4B) | `sortedIndicesGM_` 分组序 indices |
| 4 | 区3 后 | aivNum×perCoreTempSize | `sortedIndicesTempGM_` 每核分组暂存 |
| 5 | 区4 后 | aivNum×perCoreTempSize | `permOutTempGM_` 每核 perm 暂存 |
| 6 | 区5 后 | aivNum×32B | `creditScratchGM_` 每核信用 scratch |
| 7 | 区6 后 | numSendCores×streamsPerCore×16×32B | `notifyScratchGM_` 通知槽（ENGRAM_NOTIFY_SLOT_CNT=16） |
| 8 | 区7 后 | aivNum×W×4B | `partialCountsGM_` 每核部分计数 |
| 9 | 区8 后 | aivNum×32B | `flagScratchGM_` 标志 scratch |
| 10 | 区9 后 | W×4B | `indicesReadyFlagGM_` 就绪标志 |

```
slotSize          = Align32(numMaxTokensPerRank × 4B)
numOwnerRanksMax  = ceil(W / rankCores)
perCoreTempSize   = numOwnerRanksMax × slotSize              // 区4/5 的每核份额
streamsPerCore    = ceil(W / numSendCores)
```

### 6.5 UB 逐字节布局（`InitUbBuffers`，`engram_fetch_train_arch35.h:543-583`）

| # | buffer | 大小公式 |
|---|---|---|
| 1 | `hcommBuf_` | 512B |
| 2 | `hcommBatchBuf_` | HCOMM_BATCH_BUFFER_BYTES（批量 WQE） |
| 3 | `relayQue_` | 2×32KB |
| 4 | `rankCountsBuf_` | Align32(W×4B) |
| 5 | `statusBuf_` | Align32(W×32B)（窗口状态 UB 镜像） |
| 6 | `tempBuf_` | Align32(W×8B) |
| 7 | `partialCountsBuf_` | Align32(totalBlocks×W×4B) |
| 8 | `creditBuf_` | 32B |
| 9 | `indicesBuf_` / `tokenIdxInRankBuf_` | Align32(batch×4B) 各一 |
| 10 | `rankIDs/positions/divisor/gatherTmp` | compareCntMax×4B 各一（256B 对齐段） |
| 11 | `maskBuf_` | Align32(ceil(compareCntMax/8)) |

分批：`bytesPerIndice = 4+4+12+1+4 = 25B`（比推理多 1 个 int32 的 perm 位置）；
`availableUb = ubSize − fixedUb(区1..8) − relayUb(2×32KB) − 8KB`。

### 6.6 流控与背压

- **4 槽窗口**（NUM_SLOTS=4）：发送方写 `slot = writeCnt % 4`，读计数器 ≥ 写计数器 − 3 才继续
  （至少保留 1 个空槽），窗口满即自旋等待；
- **计数器通知**：`WriteWithNotifyNbi` 携带计数，`ENGRAM_NOTIFY_INTERVALS=4` →
  `notifyGran = maxTokensPerSlot/4`（每 1/4 槽发一次通知，减少通知开销）；
- **超时保护**：接收自旋 5s 未推进即 assert 失败（防对端异常死等）。

---

## 7. engram_fetch_wait / engram_fetch_grad 详设

### 7.1 wait

对每个远端 rank 读通道执行 `Drain`，排空未完成的 RDMA 读，保证 fetched 可见后才能消费
（纯同步，不碰数据本体；核分配与 fetch 推理一致）。

### 7.2 grad 数据流（DEFAULT）

1. `UnsortGrad`：按 perm 把 gradFetched 还原成分组序 → gradSorted（workspace 区 1）；
2. `ExchangeGrad`：逆向 a2a（sendCounts↔recvCounts 语义互换）把梯度行送回 owner rank → recvGrad（区 2）；
3. `UniqueScatterAdd`：LSD 基数排序（256 桶，排序键 = recvLocalEntry − rankId×numEntriesPerRank，
   maxValue = numEntriesPerRank）+ 相邻去重 + FP32 原子加聚合 → (gradUnique(K,H), uniqueLocalEntry(K,), numUnique)。

### 7.3 grad 核角色（`engram_fetch_grad_arch35.h:508-534`）

```
sendersPerRank = clamp(halfBlocks/W, 1, min(NUM_SLOTS=4, channelsPerRank))
numSendCores   = min(sendersPerRank×W, halfBlocks)          // halfBlocks = totalBlocks/2
numRecvCores   = totalBlocks − numSendCores − 1             // 预留 1 个 flag core
isSender     = aivId < numSendCores
isReceiver   = numSendCores ≤ aivId < totalBlocks−1
isFlagCore   = aivId == totalBlocks−1
```

与 fetch 训练不同：grad 的 sender 数按 `sendersPerRank×W` 对齐（每 rank 等量发送流，与 §7.6 排序
阶段 per-core staging 布局对齐）。

### 7.4 grad workspace 逐字节布局（kernel `Init`，`engram_fetch_grad_arch35.h:571-596`；tiling §4.5 同源）

| 区 | 大小 | 内容 | local 模式状态 |
|---|---|---|---|
| 1 | T×hiddenBytes | `gradSortedGM_` | **闲置**（recvGradGM_ := gradFetchedGM_） |
| 2 | totalRecv×hiddenBytes | `recvGradGM_` | **闲置** |
| 3/4 | W×32B 各 | `sdisplsGM_` / `rdisplsGM_` | 闲置 |
| 5 | aivNum×32B | `counterScratchGM_`（per-core 流控） | 闲置 |
| 6 | 32B | `flagScratchGM_` barrier 标志 | 闲置 |
| 7/8 | Align32(aivNum×4B) 各 | `segCountGM_` / `coreStartGM_` unique 分段表 | **使用** |
| 9 | Align32(totalRecv×4B) | `sortCompanionGM_` 排序伴随数组 | **使用** |
| 10 | §4.5 wsSortTotal | `sortWorkspaceGm_` 排序内部区（下表） | **使用** |

排序子区（区 10 内部，`engram_fetch_grad_sort.h:381-401`）：

| 子区 | 大小 | 内容 |
|---|---|---|
| tempVal / tempIdx / companion | 各 Align32(totalRecv×4B) | 基数排序双缓冲 + 伴随 |
| hist | sortTileCount×256×4B | 每 tile 直方图（sortTileCount = ceil(totalRecv/4096)，≥ aivNum） |
| prefix | 256×4B | 桶前缀和 |
| coreSums | aivNum×256×4B | 每 core 桶计数 |
| offsets | sortTileCount×256×4B | 每 tile 桶起始偏移 |
| simtStaging | aivNum×2×Align32(8192×4+256×28+32) | SIMT gather 每核暂存 ×2（值+索引；256 桶 × 28B 组垫 + 32B 保护位） |

### 7.5 sortPool UB 两阶段复用（`engram_fetch_grad_arch35.h:627-644, 1287-1294`）

排序阶段与 unique 阶段共用一个 TPipe 池，**二次 InitBuffer 重排**（同一块 UB 重新分区，零额外开销）：

```
排序阶段:  entryBuf 64KB | gradBuf 64KB                    // ENTRY_BUF_BYTES/GRAD_BUF_BYTES
unique阶段: entryBuf uniqueEntryBytes | gradBuf 64KB | castFp32Buf? | accumBuf
```

```
alignRowStride    = Align32(hiddenBytes)                    // GM 行→UB 行步长（32B 对齐）
maxByPong         = min(GRAD_PING_BYTES/alignRowStride, gradSubBatch)   // ping 半缓冲行数, ≤8
castFp32Buf       = Align32(Align32(hiddenDim×4) × maxByPong × 2)       // fp32 转换双缓冲（仅非 fp32 输入）
accumBuf          = Align32(hiddenDim×4) × 2                // fp32 累加行双缓冲（ACCUM_BUF_COPIES=2）
uniqueEntryBytes  = FLUSH_CAST_HEAD_BYTES(20KB)             // 5 批 × 1024 int32 头
                  + (非 fp32 输出) 2×Align32(hiddenDim×2)   // cast 写回半缓冲 ×2
```

entry 阶段按 `ENTRY_BATCH_CAP=1024` 个 int32 分批流式处理（`engram_fetch_grad_arch35.h:1113`）。

### 7.6 LSD 基数排序（256 桶）

排序键 `entry − entryOffset`（DEFAULT：entryOffset = rankId×numEntriesPerRank；LOCAL：0），
值域 `[0, numEntriesPerRank)` → pass 数 = ceil(log256(maxValue))。多 tile 并行统计直方图 →
coreSums 归约 → 桶偏移 → 分桶散布（tempVal/tempIdx 双缓冲）→ SIMT gather 回收（simtStaging
每核独立槽，槽间 32B 对齐 + 28B 组垫防越界）。排序伴随数组（companion）保持 token 与 entry 对齐，
unique 阶段以 companion 对齐梯度行。

---

## 8. 新需求：arch35 全量本 NPU 模式（fetch_mode=1）

### 8.1 需求描述

arch35（ascend950）上支持**将全局数据整表放到每个 NPU 上**：

- 之前：全局数据切片分发到不同 NPU，取任意数据需走跨 NPU 通信（RDMA / a2a）；
- 现在：**每个 NPU 上都有全局全量表**，任意 index 均在本卡本地，**取数完全不需要跨 NPU 通信**。

### 8.2 动机与适用场景

| 维度 | fetch_mode=0（切片，默认） | fetch_mode=1（全量本 NPU） |
|---|---|---|
| host 内存占用/卡 | 全表 1/N | 全表（N 倍） |
| 取数路径 | 本地命中走 GM 拷贝；远端命中走 RDMA 读（推理）/ a2a（训练） | **纯本地 GM 拷贝** |
| 是否需要卡间 RDMA 链路 | 是（context 构建即获取通道） | **否**（不获取任何通道） |
| 时延 | 受跨卡带宽/背压/Drain 影响 | 本地带宽，时延最低 |
| 适用 | 权重表大到单卡 host 放不下 | 表可整表冗余（如 KV/LoRA/小表），追求低时延 |

### 8.3 关键约束

- **默认行为不变**：不传新参数时仍走跨 NPU 通信（fetch_mode=0），完全后向兼容；
- `engram_write` 语义随模式变化：mode=1 时写入的是**全量表**（storage.size(0) = 全局 entry 数，
  此时 `num_entries_per_rank` attr 承载"本 rank 表行数 = 全局行数"）；
- 训练反向：grad 聚合的 `localEntry = entry − rankId×numEntriesPerRank` 在 mode=1 下必须改为
  `localEntry = entry`，排序 maxValue 相应收缩，否则索引换算错误。

---

## 9. local 模式方案设计（含逐字节布局）

### 9.1 总体思路

新增算子属性 `fetch_mode`（int，0=切片跨 NPU（默认）/ 1=全量本 NPU），经 tiling key 派生
4 个 fetch 内核实例 + 2 个 grad 内核实例（§4.3）。

### 9.2 推理 local 内核（EngramFetchLocalArch35）

```
for batch in indices (UB 容量分批):
    copy batch → UB
    for i in batch (核内串行):
        entry = indices[i]                     // 直接就是本表行号（无 rank 偏移）
        if 0 ≤ entry < numEntries(=attr num_entries_per_rank):
            LocalCopySlice(fetched + t×hiddenBytes, commBuffer[rankId] + entry×hiddenBytes, hiddenBytes)
            (+ GatherSf: fetchedSf 行 ← sfTable 行, 当 numSfPacks>0)
```

- 表基址 = `commBuffer[rankId]`（host pinned 注册内存的设备映射地址，注册路径不变）；
- 不 Init Hcomm、不读通道、不调用 wait（kernel 完成即数据可见）；
- 越界 entry 跳过（沿用训练内核的防御性风格）。

**分核策略（token 连续区间分片）**：

```
totalPerCore = ceil(T / totalBlocks)
start = aivId × totalPerCore; end = min(start + totalPerCore, T)   // 每核独占一段连续 token
```

**UB 逐字节布局（`engram_fetch_local_arch35.h:124-136`）**：

| # | buffer | 大小公式 | 用途 |
|---|---|---|---|
| 1 | `relayQue_` | 2×32KB（TQueBind） | LocalCopySlice + GatherSf 共用中继（串行复用，无竞争） |
| 2 | `indicesBuf_` | Align32(batch×4B) | 当前批 indices |

```
availableUb     = ubSize − 64KB − 8KB
indicesBatchSize = min(T, availableUb / 4)
```

### 9.3 训练 local 内核（EngramFetchTrainLocalArch35）

产出与 TRAIN_MODE **完全相同的 6 个输出**（PTA/反向契约不变），但全部本地生成、零通信：

| 输出 | local 模式取值 | 说明 |
|---|---|---|
| fetched | `table[indices[t]]` | 纯本地 gather（表基址 localStorageAddr） |
| perm | 恒等置换 `perm[t]=t` | 分组序 == 原始 token 序（ArithProgression 向量生成） |
| sendCounts | `[me]=T`，其余 0（32B 步长布局） | 我的 token 全部"发给我自己" |
| recvCounts | `[me]=T`，其余 0 | 我只服务我自己的请求 |
| recvLocalEntry | `= indices`（token 序，批量 DataCopy 直写） | 收到的请求即本卡 indices |
| numRecv | `= T` | |

**分核策略（token 连续区间分片 + 0 号核独写计数器）**：fetched/perm/recvLocalEntry 按 §9.2 区间
分片并行；sendCounts/recvCounts/numRecv 由 **aivId==0 独占写**（写者唯一，无需跨核同步）。
无窗口布局、无状态位、无计数器、无 SyncAll 依赖。

**UB 逐字节布局（`engram_fetch_local_arch35.h:327-349`）**：

| # | buffer | 大小公式 | 用途 |
|---|---|---|---|
| 1 | `pingBuf_` | 32KB | 拷贝乒乓左半 |
| 2 | `pongBuf_` | 32KB | 拷贝乒乓右半（MTE2/MTE3 事件对隔离） |
| 3 | `tempBuf_` | Align32(W×8×4B) | sendCounts(W×8)/recvCounts(W)/numRecv(1) 写出暂存（零填+置位） |
| 4 | `indicesBuf_` | Align32(batch×4B) | 当前批 indices |
| 5 | `permBuf_` | Align32(batch×4B) | ArithProgression 恒等 perm |

```
availableUb      = ubSize − 64KB − tempBufSize − 8KB
indicesBatchSize = min(T, availableUb / 8)                  // indices+perm 各 4B
```

**乒乓拷贝事件链（`LocalCopySlice`，`engram_fetch_local_arch35.h:469-509`）**：

```
tile k (bufIdx = k%2):
  [k≥2] WaitFlag(MTE3_MTE2, bufIdx)         // 等该 buffer 上一次 MTE3 完成
  DataCopyPad(UB[bufIdx] ← GM src)          // MTE2
  SetFlag/WaitFlag(MTE2_MTE3, bufIdx)       // 拷入完成 → 可发出
  DataCopyPad(GM dst ← UB[bufIdx])          // MTE3
  SetFlag(MTE3_MTE2, bufIdx)                // 发出完成 → 下一轮可复用
尾: 等最后两个 tile 的 MTE3_MTE2
```

### 9.4 反向 grad local 分支

grad 收到上述"退化上下文"后：

1. **跳过** InitFlagsAndDispls / ClearWinCounters / CrossRankBarrier / UnsortGrad / ExchangeGrad
   （零窗口访问、零跨核 rank 交互；workspace 区 1-6 闲置，§7.4）；
2. `recvGradGM_ := gradFetchedGM_`（梯度已在"接收序"= token 序，免拷贝）；
3. `UniqueScatterAdd` 复用：entries = recvLocalEntry(=indices)，companion 对齐 token 序；
   - 排序 maxValue：`numEntriesPerRank`（mode=1 下即全表行数），不再乘 rankSize；
   - **`localEntry = entry`**（`SetEntryOffset(0)`，去掉 `rankId×numEntriesPerRank` 偏移）；
4. 保留 `SyncAll<true>` ×2（InitCompanion 后 + CountUniquesParallel 后），保证 coreStart/segCount
   GM 可见性（替代正常路径 FinishExchangeGrad 的收尾同步）；numRecv 读取方式与 DEFAULT 一致
   （DCCI + GetValue）。

> 正确性：mode=1 下每卡持全量表副本，本卡梯度只聚合本卡 token 覆盖的 entry，产出稀疏
> (gradUnique, uniqueLocalEntry)；跨卡一致性由上层优化器按副本归约（与 DP 梯度 all-reduce 同语义），
> 不在本算子范围内。

### 9.5 tiling / workspace 差异

- fetch local（推理/训练）：workspace 仅 `SYSTEM_NEED_WORKSPACE(16MB)`（无 a2a 区域）；
- 其余校验（dtype/shape/format/hidden 128 对齐等）与既有路径一致；
- `fetch_mode` 越界（∉{0,1}）报参数错；
- grad local：workspace 保留原计算（区 1-6 闲置但无害，避免 tiling 双路径）；
- grad tiling 的 `rankSize×numEntriesPerRank ≤ INT32_MAX` 上限校验在 local 模式下放宽为
  `numEntriesPerRank ≤ INT32_MAX`（entry 即全表行号，无 rank 偏移组合，`engram_fetch_grad_tiling.cpp:588-596`）。

---

## 10. 多核数据分配与 bank 冲突规避

### 10.1 各内核 per-core 工作划分汇总

| 内核 | 划分维度 | 公式 | 写者唯一性 |
|---|---|---|---|
| 推理 DEFAULT | **owner rank**（连续分组/跨步） | `GetCoreAssignment` / `ownerRank ≡ aivId (mod totalBlocks)` | fetched 行：本地核写本地命中、RDMA 引擎直写远端命中（每行单写者） |
| 推理 LOCAL | **token 连续区间** | `[aivId×ceil(T/B), min(…))` | fetched/sf 行每行单写者 |
| 训练 DEFAULT | **rank×tokenGroup 交错** | `myOwnerRank = aivId%rankCores; myTokenGroup = aivId/rankCores`；sender/receiver/flag 三角色 | 窗口槽按 (rank, slot/group) 交错独占；fetched 由 receiver 按槽写 |
| 训练 LOCAL | **token 连续区间 + 0 号核写计数** | 同推理 LOCAL；counters 仅 aivId==0 | 6 输出全部单写者（§10.2） |
| grad DEFAULT | **sender(rank 流)/receiver/flag** | `sendersPerRank×W` 发送流；unique 阶段按 coreStart/segCount 分段 | gradUnique 原子加（FP32 AtomicAdd 同行多写者聚合）；uniqueLocalEntry 去重后单写 |
| grad LOCAL | 同 grad DEFAULT 的 unique 分段 | coreStart/segCount 分段并行 | 同上 |

### 10.2 训练 local 写者唯一性矩阵（无锁无栅栏的依据）

| 输出 | 写者 | 依据 |
|---|---|---|
| fetched[t] | 持有 token 区间含 t 的核 | 区间不相交 |
| perm[t] | 同上 | 同上 |
| recvLocalEntry[t] | 同上 | 同上 |
| sendCounts[r] (r∈[0,W)) | aivId==0 | 独占 |
| recvCounts[r] | aivId==0 | 独占 |
| numRecv[0] | aivId==0 | 独占 |

→ 训练 local 全程无需 SyncAll / 原子操作 / 锁；唯一跨核依赖是 grad 阶段读取（跨算子，stream 顺序保证）。

### 10.3 bank 冲突 / 伪共享规避机制清单

1. **UB 32B 对齐基线**（`UB_ALIGN=32`，所有 `InitBuffer` 尺寸经 `Ceil(…, 32B)` 向上取整）：
   UB 访存通路以 32B 为粒度交错，全部 buffer 起址 32B 对齐消除跨 bank 截断；MTE `DataCopy`
   的 blockLen 亦以 32B 对齐（hiddenBytes 128 对齐保证行分片对齐）。
2. **GM 状态槽 32B 独占**（`STATE_OFFSET=32`）：sendCounts 每 rank 8×int32=32B 步长、窗口读/写
   计数器每 (rank, senderIdx) 一槽 32B —— 不同 rank / 不同核的 4B 计数器**不共享同一 32B 行**，
   规避跨核 GM 写伪共享（同一 cacheline 争用导致的性能塌陷与计数丢失风险）。
3. **槽交错布局**（§6.3 区⑦⑧）：indices 槽按 (rank, slot)、token 槽按 (rank, group) 交错，多核并发
   写相邻槽天然落在不同 32B 行；`indicesSlotSize/tokenSlotSize` 以 32B 为单位整除，槽内无跨行残片。
4. **向量段 256B 对齐补齐**（`ALIGNED_LEN_256`，`compareCntMax`）：Div/Compare/前缀扫等向量指令
   按 256B 对齐段处理（尾部 mask 过滤 dummy 元素），保证向量通路满宽访问。
5. **双缓冲/乒乓**：`relayQue`（RELAY_BUFFER_NUM=2）、grad `castFp32Buf`/`accumBuf`
   （ACCUM_BUF_COPIES=2）、train-local `ping/pong`（独立 MTE2_MTE3 / MTE3_MTE2 事件对）——
   MTE2（读入）与 MTE3（写出）始终错开不同 buffer，既隐藏搬运时延，又避免同 buffer 读写
   的 bank 争用与数据冒险。
6. **GM 行 128B 对齐**（attr 校验 hiddenSize % 128 == 0）：权重行/fetched 行起址跨 bank 均匀分布，
   多核按行并行拷贝时访问相位自然错开。
7. **workspace 区间隔离**：per-core scratch（counterScratch/creditScratch/notifyScratch 等）按
   `aivId × 32B` 步长切分独占槽，核间调试/流控写互不干扰（同 2 的伪共享规避）。

---

## 11. 接口变更（aclnn / PTA）

### 11.1 aclnn（attr 顺序追加在末尾，保持既有参数不动）

```
aclnnEngramFetchGetWorkspaceSize(..., int64_t withGrad, int64_t fetchMode, uint64_t*, aclOpExecutor**)
aclnnEngramFetchGradGetWorkspaceSize(..., int64_t commBufferSize, int64_t fetchMode, uint64_t*, aclOpExecutor**)
```

### 11.2 属性定义

```
EngramFetch      .Attr("fetch_mode").AttrType(OPTIONAL).Int()   // 0 sliced(默认) / 1 local
EngramFetchGrad  .Attr("fetch_mode").AttrType(OPTIONAL).Int()
```

attr 索引：fetch 侧 `fetch_mode` 为第 6 个 attr（index 5），grad 侧第 3 个（index 2），
与 tiling 侧 `ATTR_FETCH_MODE_INDEX` 常量对齐（由 def 文件自动生成 aclnnInner 头）。

---

## 12. PTA（torch_extension）适配

`ElasticBuffer` 新增 `fetch_mode` 参数，**默认 0 = 跨 NPU 通信**：

```python
ElasticBuffer.FETCH_MODE_SLICED = 0   # 切片 + 跨 NPU 通信（默认，行为与历史版本完全一致）
ElasticBuffer.FETCH_MODE_LOCAL  = 1   # 全量表冗余到每个 NPU，取数纯本地

buf = ElasticBuffer(group, num_cpu_bytes=get_engram_storage_size_hint(num_entries_full, hidden),
                    fetch_mode=ElasticBuffer.FETCH_MODE_LOCAL)
buf.engram_write(full_table)          # mode=1 写入全量表
w = buf.engram_fetch(indices)()       # 纯本地 gather；mode=1 下 wait 被短路（无 RDMA 在途）
# 训练： with_grad=True 时反向 engram_fetch_grad 自动走 local 分支
```

- op schema：`engram_fetch(..., int dtype, int fetch_mode)`、
  `engram_fetch_train(..., int rank_size, int fetch_mode)`（PrivateUse1 + Meta 同步适配）；
- csrc：构造函数追加 `fetchMode`（默认 0，pybind 关键字默认值保持向后兼容）；
  - mode=1 时 context 构建**跳过通道获取与远端地址交换**（不依赖任何卡间 RDMA 链路），
    仅保留 host pinned 注册 + 本地基址；with_grad 仍查询 HCCL buffer size（attr 校验需要）；
  - `EngramWrite` 的 INT32 上限校验按模式区分（mode=1：行数 ≤ INT32_MAX）；
  - 推理 mode=1 的 wait 调用在 Python 层短路，不发 aclnnEngramFetchWait。

---

## 13. 边界与约束

| 项 | 约束 |
|---|---|
| hidden_size | 128 对齐（两模式一致；亦是 §10.3-6 的行对齐依据） |
| indices | int32 1D，NPU 张量；mode=1 取值域 [0, 全局 entry 数) |
| 训练 num_max_tokens_per_rank | ≥ numTokens（local 模式沿用） |
| sendCounts 32B 步长 | fetch↔grad 接口契约，local 模式保持同一布局 |
| 平台 | 仅 ascend950（arch35）；内核需 ASC_DEVKIT>9.0 且 HCOMM>9.0 |
| rankSize | [2, 1024]（local 模式同样要求组内 ≥2 卡，走 HCCL 组构建） |
| 兼容性 | 不传 fetch_mode ⇔ fetch_mode=0，aclnn/PTA 行为与历史完全一致（eager；序列化 graph 因 schema 追加参数不兼容旧图） |

---

## 14. 验证计划与结果

### 14.1 本机环境结论

- 本机为 A3（ascend910_93）服务器：engram_fetch 为 950 专属算子，**无法本机运行 kernel 级功能验证**
  （执行器无法匹配 950 kernel 二进制；host 侧 tiling 链路经 UT 的 950 平台 mock 全覆盖，见 14.2）；
- 编译验证环境：**CANN 9.2.0 原生环境**（ASC_DEVKIT/HCOMM 9.2）。
  > 早期验证曾需要 CANN 9.1.0 + asc-comm 开发补丁（MakeBatchHandle/BatchCommit 批量 Hcomm API）；
  > 该 API 已进入 9.2.0 主线，当前无需补丁。

### 14.2 已完成的编译/UT 验证（2026-09-16 复验）

| 项 | 命令 | 结果 |
|---|---|---|
| arch35 全量出包 | `bash build.sh --pkg --soc=ascend950 --module=mc2 --ops=engram_fetch,engram_fetch_grad,engram_fetch_wait -j16` | ✅ exit=0，`build_out/cann-ops-transformer-custom_linux-aarch64.run` 生成（含 19 个 kernel .o） |
| kernel 实例覆盖 | 二进制 `kernelList` 解析 | ✅ EngramFetch 每 dtype 含 **4 实例**（`_0.._3` = DEFAULT/TRAIN/LOCAL/TRAIN_LOCAL，5 dtype）；EngramFetchGrad 含 **2 实例**（`_0/_1` = DEFAULT/LOCAL，bf16/fp16/fp32）；EngramFetchWait 5 dtype |
| op_api UT 编译+运行 | `bash build.sh --opapi_test --noexec --soc=ascend950 --module=mc2 --ops=engram_fetch,engram_fetch_grad -j16` 后 `transformer_op_api_ut --gtest_filter='*Engram*'` | ✅ **15/15 通过**（nullptr 校验 / local success / **invalid fetch_mode 拒绝**；host 侧 aclnn→tiling→tiling key→workspace 全链路在 950 平台 mock 下验证） |
| PTA schema | python 导入 + `torch.library.define` | ✅ 4 个 schema（含新增 `int fetch_mode`）全部解析注册成功；`FETCH_MODE_SLICED/LOCAL` 常量可用 |

### 14.3 用户在 arch35（950）环境的功能验证建议

1. mode=1 推理：W 卡各写同一全量表 → fetch 随机 indices → 与 CPU golden 逐行比对（含 sf 路径）；
2. mode=1 训练：fetched 前向比对 + gradFetched 回传 → gradUnique/uniqueLocalEntry 与参考
   scatter-add（按 indices 去重求和）比对；perm 应为恒等、sendCounts/recvCounts 仅 [me]=T、numRecv=T；
3. 回归：mode=0（不传 fetch_mode）行为与基线一致；
4. 边界：空 indices、单 entry 表、indices 全部越界（应全跳过）、大表（行数接近 INT32_MAX）。

### 14.4 关键环境备注（950 编译复现方法）

```bash
# 1) 依赖（首次构建网络受限时手动补齐子模块）
cd third_party/ops-tensor && git submodule update --init --recursive --depth=1
# 2) 编译（CANN ≥ 9.2.0 原生环境，无需额外补丁）
source ~/Ascend/cann-9.2.0/bin/setenv.bash
bash build.sh --pkg --soc=ascend950 --module=mc2 --ops=engram_fetch,engram_fetch_grad,engram_fetch_wait -j16
```
