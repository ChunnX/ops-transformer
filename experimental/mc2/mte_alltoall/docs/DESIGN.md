# MTE AllToAll 算子设计文档

---

## 0. 概述

### 0.0 需求类型判断

**特定用例**：4 卡场景，INT8 数据类型，AllToAll 通信。

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | mte_alltoall |
| 算子类别 | MC2（多卡 MTE 通信） |
| 需求类型 | 特定用例（4 卡，INT8） |
| 支持数据类型 | int8_t |
| 支持芯片 | Ascend950PR/Ascend950DT（A5 架构） |
| 特殊约束 | 需要 4 卡 HCCL 环境；MTE 通信引擎；KERNEL_TYPE_AIV_ONLY |

### 0.2 算子语义

AllToAll 是分布式通信原语，语义为：第 i 张卡将输入数据按 rank 切分成 rankDim 块，其中第 j 块发送给第 j 张卡；同时从所有卡接收属于本卡的数据块。最终每张卡输出数据的第 i 块来自第 i 张卡的输入。

```
输入（每张卡 myRank）:  x[rankDim × sendCount]，按 rank 切分为 rankDim 块
  x[0], x[1], ..., x[rankDim-1]，每块 sendCount 个 int8_t 元素

输出（每张卡 myRank）:  y[rankDim × sendCount]，按 rank 排列
  y[remoteRank * sendCount ... (remoteRank+1) * sendCount - 1] 
    = 来自 rank(remoteRank) 的输入中第 myRank 块的数据
```

### 0.3 AllToAll 与 AllGather 的核心区别

| 维度 | AllGather | AllToAll |
|------|-----------|----------|
| 输出大小 | rankDim × 输入大小 | 与输入大小相同 |
| 数据来源 | 每张卡的完整输入 | 每张卡输入中属于本卡的块 |
| Execute 读取偏移 | 远程 Win 的同偏移位置 | 远程 Win 的 myRankId * sendCount 偏移 |
| 输出拼接 | output[remoteRank] = 完整输入 | output[remoteRank] = 输入的第 myRankId 块 |

---

## 1. 算子设计

### 1.1 数学公式

```
定义:
  rankDim    = 卡数（4）
  sendCount  = 每块数据量（元素数）
  myRankId   = 当前卡的 rank ID
  totalElements = rankDim × sendCount

输入:
  x ∈ Z^{totalElements}, dtype=int8_t
  卡 i 的输入: x_i[0 .. totalElements-1]
    其中 x_i[j * sendCount .. (j+1) * sendCount - 1] 是要发给卡 j 的数据块

AllToAll 通信:
  卡 myRankId 从卡 remoteRank 接收:
    recv_block[remoteRank] = x_{remoteRank}[myRankId * sendCount .. (myRankId+1) * sendCount - 1]

输出:
  y[remoteRank * sendCount + k] = x_{remoteRank}[myRankId * sendCount + k]
    对所有 remoteRank ∈ [0, rankDim), k ∈ [0, sendCount)
```

### 1.2 API 映射

| 数学操作 | 对应 API | 关键参数 | 数据布局 | 官方文档 |
|---------|---------|---------|---------|---------|
| GM→UB 数据搬运 | `DataCopy` | count=copyBlockNum | 连续，32B 对齐 | [基础数据搬运](../../asc-devkit/docs/api/context/基础数据搬运.md) |
| UB→GM 数据搬运 | `DataCopy` | count=copyBlockNum | 连续，32B 对齐 | 同上 |
| UB→Win 数据写入 | `DataCopy` | 通过 mc2_comm.h CopyDataToWin | 连续 | 同上 |
| Win→UB 数据读取 | `DataCopy` | 通过 Execute 直接调用 | 连续 | 同上 |
| 状态初始化 | `Duplicate<float>` | scalarValue=0.0 | 标量填充 | [Duplicate](../../asc-devkit/docs/api/context/Duplicate.md) |
| 状态归约（全卡就绪判定）| `Sum` | SumParams{1, statusCnt, statusCnt} | 一维向量 | [Sum](../../asc-devkit/docs/api/context/Sum.md) |
| 流水线屏障 | `PipeBarrier<PIPE_ALL>` | 无 | — | mc2_comm.h 内部 |
| 事件同步 | `SyncFunc<HardEvent::*>` | MTE2_S / S_MTE3 等 | — | mc2_comm.h 内部 |

#### 1.2.1 API 语义验证

| API | 数据布局 | 功能需求 | API 选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| DataCopy (GM→UB) | 连续，按元素搬运，32B 对齐 | 将 Win/输入数据搬入 UB | `DataCopy(Local, Global, count)` | count*sizeof(T) 需 32B 对齐，否则向下取整 | ✅ | [链接](../../asc-devkit/docs/api/context/基础数据搬运.md) |
| DataCopy (UB→GM) | 连续，按元素搬运，32B 对齐 | 将 UB 数据搬出到 Win/输出 | `DataCopy(Global, Local, count)` | 同上 | ✅ | 同上 |
| Duplicate | UB LocalTensor, float 类型 | 将状态 buffer 初始化为 0.0 | `Duplicate<float>(dst, 0.0, count)` | int8_t 不支持 Duplicate 标量模式，但 float 类型支持 | ✅（用于 float 状态 buffer） | [链接](../../asc-devkit/docs/api/context/Duplicate.md) |
| Sum | 一维 float 向量 | 对 rankDim 个状态值求和判定全卡就绪 | `Sum(dst, src, SumParams{1, cnt, cnt})` | inner*sizeof(float) 需 32B 对齐；float 类型支持 | ✅ | [链接](../../asc-devkit/docs/api/context/Sum.md) |

**验证说明**：
- DataCopy 用于 int8_t 的 GM↔UB 搬运：Ascend950PR 支持 b8 类型（即 int8_t），连续搬运通过 count 参数指定元素数 ✅
- **b8 与 int8_t 的映射关系**：Ascend950PR/DT 的 GM↔UB DataCopy 文档列出支持类型为 `b8, b16, b32, b64`，而 mc2_kernel 官方模板直接使用 `int8_t` 作为 DataCopy 模板参数（`GlobalTensor<int8_t>`、`DataCopy(xTmpTensor, remoteWinXTensor_[srcOffset], count)`）。编译器在模板实例化时将 `int8_t` 隐式映射为 `b8`，无需手动类型转换。若遇到编译错误，可尝试将 `GlobalTensor<int8_t>` 替换为 `GlobalTensor<b8>` ✅
- Sum/Duplicate 仅用于 float 类型的状态同步 buffer，不涉及业务数据类型 ✅
- 所有 API 均已在 mc2_kernel 模板代码中使用并验证，本算子复用相同的 API 调用方式 ✅

### 1.3 数据流

```
┌─────────────── 每个 AIV 核的执行流程 ───────────────┐
│                                                       │
│  Phase 1: CopyDataToWin（写入本地 Win）               │
│  ┌─────────┐      DataCopy      ┌──────────────┐     │
│  │ Input GM │ ──────────────────→ │ 本地 Win 窗口 │     │
│  │ (x)      │  多核并行分块搬运   │ (全量输入)    │     │
│  └─────────┘                     └──────────────┘     │
│                                                       │
│  Phase 2: WriteStatusToWin（通知所有卡数据就绪）      │
│  ┌──────────────┐    DataCopy    ┌──────────────┐     │
│  │ 本核状态(1.0)│ ──────────────→ │ 所有卡的状态区│     │
│  └──────────────┘                 └──────────────┘     │
│                                                       │
│  Phase 3: ReadStatus（自旋等待所有卡就绪）            │
│  ┌──────────────┐   Sum 归约    ┌──────────────┐      │
│  │ 本地状态区    │ ─────────────→ flag ≥ rankDim │      │
│  └──────────────┘               └──────────────┘      │
│                                                       │
│  Phase 4: Execute（AllToAll 业务读取）                │
│  对每个 remoteRank ∈ [0, rankDim):                    │
│  ┌────────────────────┐  DataCopy  ┌─────┐  DataCopy  ┌────────────┐
│  │ 远程 rank 的 Win    │ ─────────→ │  UB │ ─────────→ │ Output GM  │
│  │ offset=myRank*send │            └─────┘            │ (y)        │
│  │     +localOffset   │                                └────────────┘
│  └────────────────────┘                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.4 核心计算步骤

**Phase 1-3（通信框架，复用 mc2_comm.h）**：
1. `CopyDataToWin()` — 将本卡全部输入数据（rankDim × sendCount）写入本地 Win 窗口，多核并行
2. `WriteStatusToWin()` — 向所有远程卡的状态区写入就绪标志
3. `ReadStatus()` — 自旋等待所有卡就绪（Sum 归约判定）

**Phase 4（AllToAll 业务逻辑，替换模板 Execute）**：
1. 遍历本核 Execute 阶段分配的数据块
2. 对每个数据块，遍历所有远程 rank
3. 从远程 rank 的 Win 中读取 **myRankId 块**（偏移 = myRankId × sendCount + localOffset）
4. 写入本卡输出的 **remoteRankId 位置**（偏移 = remoteRankId × sendCount + localOffset）

**关键设计要点**：
1. **双阶段核分配**：CopyDataToWin 按 totalElements 分配多核；Execute 按 sendCount 分配多核（因为每次只处理 sendCount 大小的块）
2. **偏移计算**：AllToAll 的核心区别在于 Execute 中从远程 Win 读取的偏移不是 `curXOffset`（AllGather），而是 `myRankId * sendCount + execOffset`
3. **输出拼接**：`output[remoteRankId * sendCount + execOffset]` 而非 `output[remoteRankId * totalElements + curXOffset]`

### 1.5 内存管理（Buffer 规划）

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| xQueue（mc2_comm） | CopyDataToWin 数据搬运管道 | 2 × 1024B = 2KB | VECIN↔VECOUT |
| winFlagsBuf | Win 双缓冲标志位 | 32B | VECIN |
| writeStateBuf | 写状态通知 buffer | 32B | VECIN |
| readStateBuf | 读状态等待 buffer | rankDim × 32B = 128B | VECIN |
| stateResetBuf | 状态重置 buffer | rankDim × 32B = 128B | VECIN |
| xInQueue（kernel） | Execute 数据搬运管道 | 2 × 1024B = 2KB | VECIN↔VECOUT |

**总 UB 使用量**: ~4.3 KB（远小于 192KB/248KB 限制，无压力）

**Buffer 时间互斥说明**：xQueue（mc2_comm）和 xInQueue（kernel）各分配 2×1024B = 4KB，占总 UB 使用量的 93%。两者在时间上互斥：xQueue 仅在 CopyDataToWin 阶段使用，xInQueue 仅在 Execute 阶段使用。受限于 mc2_comm.h 不可修改的约束，无法共享空间。当前 ~4.3KB 远小于 248KB 限制，无优化必要。若未来需支持更大 block size 或更多 buffer，可考虑统一管理 Buffer 分配。

---

## 2. 架构设计

### 2.1 多核切分策略

本算子采用**双阶段多核切分**：CopyDataToWin 阶段和 Execute 阶段使用不同的核分配策略。

#### CopyDataToWin 阶段

| 项目 | 说明 |
|-----|------|
| 切分对象 | totalElements = rankDim × sendCount（每卡全部输入） |
| 切分方式 | 按 1024 字节块切分，均匀分配到 aivNum 个 AIV 核 |
| 单核任务量 | round 或 round+1 个 1024B 块 |
| 使用的核数 | aivNum（由 Tiling 传入，默认 64） |
| 负载均衡 | 尾块分配给前 tailBlockNums 个核 |

#### Execute 阶段

| 项目 | 说明 |
|-----|------|
| 切分对象 | sendCount（每卡从每个远程 rank 读取的数据量） |
| 切分方式 | 按 1024 字节块切分，均匀分配到 aivNum 个 AIV 核 |
| 单核任务量 | execRound 或 execRound+1 个 1024B 块 |
| 使用的核数 | aivNum（与 CopyDataToWin 相同） |
| 负载均衡 | 尾块分配给前 execTailBlockNums 个核 |

**核心公式**：

```cpp
// CopyDataToWin 阶段
totalElements = rankDim * sendCount;
totalBytes = totalElements * sizeof(int8_t);  // = totalElements (INT8)
copyTotalBlocks = CeilDiv(totalBytes, X_BLOCK_BYTES);  // X_BLOCK_BYTES = 1024
copyRound = copyTotalBlocks / aivNum;
copyTailBlockNums = copyTotalBlocks % aivNum;

// Execute 阶段
sendBytes = sendCount * sizeof(int8_t);  // = sendCount (INT8)
execTotalBlocks = CeilDiv(sendBytes, X_BLOCK_BYTES);
execRound = execTotalBlocks / aivNum;
execTailBlockNums = execTotalBlocks % aivNum;
```

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 248KB (A5/Ascend950) |
| 单次处理数据量 | X_PRE_BLOCK_NUM = 1024 个 int8_t = 1024 字节 |
| 是否需要分 chunk | 否（UB 使用量极小，仅 ~4.3KB） |
| chunk 大小 | X_BLOCK_BYTES = 1024B |

### 2.3 分支场景覆盖

本算子为特定用例（4 卡 INT8），无 dtype 分支。主要变量为 sendCount 大小：

| 场景 | 条件 | 处理策略 |
|------|------|---------|
| 标准场景 | sendCount × rankDim ≥ 1024B | 多核并行，每个核分配多个块 |
| 小数据量 | sendCount × rankDim < 1024B | copyTotalBlocks = 1，仅 1 个核工作 |
| 尾块处理 | execTotalBlocks % aivNum ≠ 0 | 前 execTailBlockNums 个核多处理一个块 |
| 全对齐 | sendCount 是 1024 的倍数 | 尾块元素数 = X_PRE_BLOCK_NUM，无需特殊处理 |
| 非对齐 | sendCount 非 1024 倍数 | 最后一个核的最后一块处理剩余元素 |

**对齐约束**：sendCount 必须是 32 的倍数（int8_t 场景下 DataCopy 的 `count × sizeof(T)` 需 32 字节对齐）。当 sendCount 不满足 32 字节对齐时，DataCopy 尾块搬运量将向下取整，导致数据丢失。由于 rankDim = 4，sendCount 为 32 的倍数时 totalElements = 4 × sendCount 也满足 32 字节对齐，CopyDataToWin 和 Execute 两阶段均安全。

### 2.4 类别特有设计（MC2 通信算子）

#### 2.4.1 通信拓扑选择

| 项目 | 配置 |
|-----|------|
| 拓扑字符串 | `"AlltoAll=level0:fullmesh;level1:pairwise"` |
| Level 0（同节点） | fullmesh — 4 卡同节点，全互联效率最高 |
| Level 1（跨节点） | pairwise — 预留扩展性 |
| 选择理由 | 模板已针对 AllToAll 配置；fullmesh 在小规模同节点通信中延迟最低 |

#### 2.4.2 Win 窗口管理策略

**双缓冲 ping-pong 机制**（复用 mc2_comm.h）：

```
Win 窗口布局（每个 rank）:
  Offset 0                        → State Region A（状态区 A）
  Offset SINGLE_STATE_REGION_SIZE → State Region B（状态区 B）
  Offset STATE_WIN_SIZE           → Data Region A（数据区 A）
  Offset 0 (windowsOut)           → Data Region B（数据区 B）

通过 winBufferFlags (0/1) 在 A/B 之间做 ping-pong 切换。
```

- 每次 Kernel 调用时，读取当前 flag 并切换为下一个（1 - flag）
- CopyDataToWin 写入 Data Region，WriteStatusToWin 写入 State Region
- Execute 从远程 rank 的 Data Region 读取

**Win 数据区大小说明**：Win 数据区大小由 HCCL 通过 `HcclAllocComResourceByTiling` 自动分配，取决于 Mc2CcTilingConfig 参数和数据量。开发者无需手动计算 Win 空间，但需注意：
- 大数据量场景（如 sendCount > 100K）时，需确认 HcclAllocComResourceByTiling 返回成功
- Win 空间需容纳 totalElements = rankDim × sendCount 字节的数据区（加上状态区）
- 首次运行大数据量测试用例时，需检查 Win 分配是否成功，以及运行时是否出现 Win 空间溢出错误

#### 2.4.3 mc2Context 初始化流程

```cpp
// Host 端（test_mte_alltoall.cpp 中的 CreatTilingDataAndContext）

// 1. 构造 Tiling 数据
MteAllToAllTilingData tilingData;
tilingData.mteAllToAllTilingInfo.sendCount = sendCount;
tilingData.mteAllToAllTilingInfo.rankDim = rankDim;
tilingData.mteAllToAllTilingInfo.aivNum = BLOCK_NUM;

// 2. 配置 MC2 通信 Tiling
Mc2CcTilingConfig mc2CcTilingConfig(hcomName, 8, "AlltoAll=level0:fullmesh;level1:pairwise");
mc2CcTilingConfig.SetCommEngine(3);  // 3 = MTE 通信引擎
mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling);

// 3. 通过 HCCL 分配通信资源
HcclAllocComResourceByTiling(commHandle, stream, &tilingData, &mc2Context);
```

**Mc2CcTilingConfig 参数说明**：

| 参数 | 值 | 含义 |
|------|-----|------|
| hcomName | HCCL 通信域名称 | 标识通信组 |
| transportUnitSize | 8 | 传输单元大小（字节），HCCL 传输协议参数，与数据类型大小无直接对应关系 |
| topology | `"AlltoAll=level0:fullmesh;level1:pairwise"` | AllToAll 拓扑策略 |
| commEngine | 3 | MTE 通信引擎 |

#### 2.4.4 Tiling 参数设计（三段式）

```cpp
// === 第一段：mc2InitTiling（MC2 初始化参数）===
// 由 Mc2CcTilingConfig.GetTiling() 自动填充
Mc2InitTiling mc2InitTiling;

// === 第二段：mc2CcTiling（MC2 通道参数）===
// 由 Mc2CcTilingConfig.GetTiling() 自动填充
Mc2CcTiling mc2CcTiling;

// === 第三段：业务 Tiling（手动填充）===
struct MteAllToAllTilingInfo {
    uint64_t sendCount;   // 每个 rank 的块大小（元素数）
    uint64_t rankDim;     // rank 总数（卡数）
    uint64_t aivNum;      // 使用的 AIV 核数
};

// 完整 Tiling 结构
struct MteAllToAllTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MteAllToAllTilingInfo mteAllToAllTilingInfo;
};
```

**业务 Tiling 字段说明**：

| 字段 | 类型 | 取值范围 | 含义 |
|------|------|---------|------|
| sendCount | uint64_t | ≥ 1 | 每个 rank 发送/接收的数据块大小（int8_t 元素数） |
| rankDim | uint64_t | = 4（本需求）| 通信域内的卡数 |
| aivNum | uint64_t | 1 ~ 64（Ascend950） | 使用的 AIV 核数 |

#### 2.4.5 多核分配策略

**CopyDataToWin 阶段核分配**（复用 mc2_comm.h 框架）：

```cpp
// 已由 mteComm_ 管理，计算方式：
mteComm_.round_ = copyTotalBlocks / aivNum;
mteComm_.tailBlockNums_ = copyTotalBlocks % aivNum;
// InitParams() 计算:
//   assignedBlockNums_ = (aivId < tailBlockNums) ? round+1 : round
//   xOffset_ = aivId * round + min(aivId, tailBlockNums) 个块 * X_PRE_BLOCK_NUM
```

**Execute 阶段核分配**（新增计算，存储在 kernel 类成员变量中）：

```cpp
// Execute 阶段独立的核分配
uint32_t aivId = mteComm_.aivId_;  // 来自 mteComm_ 的核 ID

execRound_ = execTotalBlocks / aivNum;
execTailBlockNums_ = execTotalBlocks % aivNum;
execAssignedBlockNums_ = (aivId < execTailBlockNums_) ? execRound_ + 1 : execRound_;
uint64_t execBlockIdx = aivId * execRound_ + (aivId < execTailBlockNums_ ? aivId : execTailBlockNums_);
execXOffset_ = execBlockIdx * X_PRE_BLOCK_NUM;
execTailXNums_ = BlockAlignMod(sendCount, X_PRE_BLOCK_NUM);

// Execute 阶段尾块核 ID（与 mc2_comm.h ComputeTailAivId 逻辑一致）
if (execRound_ == 0) {
    execLastAivId_ = execTailBlockNums_ - 1;
} else {
    execLastAivId_ = aivNum - 1;
}
```

#### 2.4.6 状态同步机制

复用 mc2_comm.h 的三步同步：

| 步骤 | API | 时机 | 说明 |
|------|-----|------|------|
| 1. 写入数据 | `CopyDataToWin()` | 数据写入 Win 后 | 多核并行写入本地 Win |
| 2. 通知就绪 | `WriteStatusToWin()` | CopyDataToWin 完成后 | 每核向所有远程卡的 State Region 写入 1.0 |
| 3. 等待就绪 | `ReadStatus()` | WriteStatusToWin 完成后 | 自旋等待 Sum(flag) ≥ rankDim - 0.5 |

**Sum 归约判定条件**：
- 期望值：所有 rank 都写入 1.0 → Sum = rankDim
- 判定范围：`flag ∈ [rankDim - 0.5, rankDim + 0.5]`
- 防止浮点误差导致的误判

**跨核同步假设**：

mc2_comm.h 采用按 core index 对应的同步机制：core i 只等待所有远程 rank 的 core i 就绪，不等待同 rank 其他 core。这意味着当 core 0 在 ReadStatus 后开始 Execute 读取远程 Win 数据时，假设远程 rank 的所有 core 的 CopyDataToWin 写入已全局可见。

此假设成立的依据：
1. `CopyDataToWin` 末尾有 `PipeBarrier<PIPE_ALL>()` 确保本核 MTE3 写入完成
2. `WriteStatusToWin` 在 `CopyDataToWin` 之后执行，其 `DataCopy` 写远程状态区隐含硬件级全局可见性保证
3. AllGather 模板使用相同机制并已验证通过

> ⚠️ **验证要点**：首次运行时需重点验证 ReadStatus 后 Execute 读取远程 Win 数据的完整性。如出现数据不完整，需在 CopyDataToWin 和 WriteStatusToWin 之间增加全局屏障。

**Sum in-place 说明**：

mc2_comm.h 的 ReadStatus 使用 `Sum(statusTensor, statusTensor, sumParams)` 的 in-place 模式，而 Sum 官方文档标注"不支持源操作数与目的操作数地址重叠"。此为模板已验证用法，保持一致。首次运行时需重点验证 ReadStatus 同步正确性，如出现异常，考虑分配独立的 dst tensor。

#### 2.4.7 Execute 核心流程伪代码

**适用场景**：所有数据量场景（通过 execRound/execTailBlockNums 自适应）

**变量来源映射**：

| 伪代码变量 | 来源 | 初始化位置 |
|-----------|------|-----------|
| `aivId_` | `mteComm_.aivId_` | `mteComm_.InitParams()` 中通过 `GetBlockIdx()` 初始化 |
| `myRankId_` | `mteComm_.hcclContext_->rankId` | `mteComm_.InitHcclContextByAddr()` |
| `rankDim_` | `mteComm_.hcclContext_->rankDim` | `mteComm_.InitHcclContextByAddr()` |
| `sendCount_` | `tilingDataOut->mteAllToAllTilingInfo.sendCount` | Init 中从 Tiling 读取 |
| `winBufferFlags_` | `mteComm_.winBufferFlags_` | `mteComm_.InitGMTensor()` 中从 Win 标志位读取 |
| `execAssignedBlockNums_` | Execute 核分配计算 | Init 中计算（§2.4.5） |
| `execXOffset_` | Execute 核分配计算 | Init 中计算（§2.4.5） |
| `execLastAivId_` | Execute 核分配计算 | Init 中计算（§2.4.5） |
| `execTailXNums_` | `BlockAlignMod(sendCount, X_PRE_BLOCK_NUM)` | Init 中计算（§2.4.5） |

```cpp
__aicore__ inline void Execute()
{
    for (uint64_t curBlock = 0; curBlock < execAssignedBlockNums_; ++curBlock) {
        // 当前块在 sendCount 范围内的偏移
        uint64_t execOffset = execXOffset_ + curBlock * X_PRE_BLOCK_NUM;
        
        // 计算本次拷贝的元素数（最后一个核的最后一块可能是尾块）
        uint32_t copyBlockNum = X_PRE_BLOCK_NUM;
        if ((mteComm_.aivId_ == execLastAivId_) && (curBlock == execAssignedBlockNums_ - 1)) {
            copyBlockNum = execTailXNums_;
        }
        
        // 遍历所有远程 rank
        for (uint32_t remoteRankId = 0; remoteRankId < mteComm_.hcclContext_->rankDim; ++remoteRankId) {
            // 获取远程 rank 的 Win 数据区地址
            GM_ADDR remoteDataGm = mteComm_.GetWinDataAddrGm(remoteRankId, mteComm_.winBufferFlags_);
            remoteWinXTensor_.SetGlobalBuffer((__gm__ int8_t*)remoteDataGm);
            
            // AllToAll 偏移计算（核心区别）：
            // - 源：远程 rank 的 Win 中第 myRankId 块 + 本核的 Execute 偏移
            uint64_t srcOffset = mteComm_.hcclContext_->rankId * sendCount_ + execOffset;
            // - 目的：输出的第 remoteRankId 块 + 本核的 Execute 偏移
            uint64_t dstOffset = remoteRankId * sendCount_ + execOffset;
            
            // Win → UB → Output
            LocalTensor<int8_t> xTmpTensor = xInQueue_.AllocTensor<int8_t>();
            DataCopy(xTmpTensor, remoteWinXTensor_[srcOffset], copyBlockNum);
            xInQueue_.EnQue(xTmpTensor);
            xTmpTensor = xInQueue_.DeQue<int8_t>();
            DataCopy(mteComm_.outputTensor_[dstOffset], xTmpTensor, copyBlockNum);
            xInQueue_.FreeTensor(xTmpTensor);
        }
    }
}
```

---

## 3. 确认清单

- [x] 多核切分策略已确定（双阶段：CopyDataToWin 按 totalElements，Execute 按 sendCount）
- [x] UB 切分策略已确定（X_PRE_BLOCK_NUM = 1024，UB 使用 ~4.3KB）
- [x] Buffer 规划已完成（6 个 buffer，总 ~4.3KB）
- [x] 分支场景已覆盖（标准/小数据量/尾块/对齐/非对齐，sendCount 需 32 字节对齐）
- [x] 通信拓扑已选择（AlltoAll fullmesh/pairwise）
- [x] Win 窗口管理已设计（复用 mc2_comm.h 双缓冲机制，Win 大小由 HCCL 自动管理）
- [x] Tiling 参数已设计（三段式：mc2InitTiling + mc2CcTiling + 业务 tiling）
- [x] 状态同步机制已明确（CopyDataToWin → WriteStatusToWin → ReadStatus → Execute，含跨核同步假设说明）
- [x] 所有 API 已通过文档验证（DataCopy, Sum, Duplicate 均支持 Ascend950 + int8_t/float，b8 与 int8_t 映射已说明）
- [x] 与 AllGather 的核心差异已标注（Execute 偏移计算不同）
