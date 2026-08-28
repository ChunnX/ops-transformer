# MoeEpCombine 基于 metadata 直接发送设计

## 1. 设计目标

本设计用于将 `MoeEpCombine` 从“扫描 metadata 并重建逐 token 地址表”改为“直接根据 5 列 metadata 定位 `x` 和 `topk_weights` 并发送”。

目标如下：

- 不构建逐 token 地址表，不产生地址表的 GM 写入、回读和计数前缀和。
- 使用 metadata 第 5 列 `recv_x_idx` 定位 token 和 weight 的输入行。
- token 与 weight 保持独立输入和独立发送，不拼接 payload。
- 本卡使用全部 AIV 并行执行 DataCopy，完成后再进入远端发送阶段。
- 远端保持“一 rank 一 channel、一个 channel 仅由一个核访问”的约束。
- `MoeEpCombine` 主 kernel 只保留一次全核 `SyncAll`。
- 保持 Combine 通信窗口、完成 flag 和 `MoeEpCombineEpilogue` 协议不变。

当前设计面向“一 rank 一 channel”。未来若单 Jetty/channel 对应多个 rank，分核所有权需要从 rank 提升为 channel，但本文的数据路径和同步原则保持不变。

## 2. 输入与 metadata 契约

### 2.1 有效数据范围

```text
actualA = sum(num_recv_tokens_per_expert)
```

Combine 只处理 metadata、`x` 和 `topk_weights` 的有效前缀。metadata 的实际分配容量不能代替 `actualA`。

### 2.2 metadata 布局

`recv_src_metadata` 为紧凑的 `[A_alloc, 5] int32`：

```text
[src_rank, src_token_idx, src_topk_idx, src_slot_idx, recv_x_idx]
```

字段用途：

| 字段 | Combine 用途 |
|---|---|
| `src_rank` | 选择本卡或远端目标 rank；有效前缀按该字段连续排列 |
| `src_token_idx` | 计算目标 token slot |
| `src_topk_idx` | 计算目标 top-k slot |
| `src_slot_idx` | 当前 Combine 不使用 |
| `recv_x_idx` | 定位 `x` 和 `topk_weights` 的输入行 |

源和目标映射为：

```text
token source  = x[recv_x_idx]
weight source = topk_weights[recv_x_idx]

dst_slot      = src_token_idx * topK + src_topk_idx
token dst     = combine_data_window + dst_slot * perSlotBytes
weight dst    = combine_state_window + dst_slot * WIN_ADDR_ALIGN
```

metadata 行号不能用于索引 `x` 或 `topk_weights`。metadata 是 rank-order，而 `x` 和 `topk_weights` 保持 expert-order，两者通过 `recv_x_idx` 关联。

### 2.3 rank 区间定位

metadata 的有效前缀按 `src_rank` 排序，因此通过二分查找得到任意 rank 的连续区间：

```text
rank_begin = lower_bound(metadata.src_rank, rank)
rank_end   = lower_bound(metadata.src_rank, rank + 1)
```

该过程不需要构建逐 token 地址表，也不要求同一 rank 内的 `recv_x_idx` 有序。

## 3. 总体执行流程

```text
所有 AIV 启动
    │
    ├─ 1. 初始化
    │     ├─ 读取 tiling、rank、通信窗口和 Hcomm context
    │     ├─ 每核独立计算 actualA
    │     ├─ 初始化 metadata UB、token queue、weight buffer/queue
    │     └─ 初始化每核 Hcomm batch 和 flag source
    │
    ├─ 2. 定位本卡 metadata 区间
    │     ├─ lower_bound(rankId)
    │     └─ lower_bound(rankId + 1)
    │
    ├─ 3. 全核并行搬运本卡数据
    │     ├─ 按本卡 metadata 记录数均分
    │     ├─ x[recv_x_idx] -> UB -> 本卡 data window
    │     └─ weight[recv_x_idx] -> 独立 UB -> 本卡 state window
    │
    ├─ 4. 各核等待本核最后一次 MTE3 完成
    │
    ├──────────── SyncAll #1（Combine 主 kernel 唯一一次）────────────
    │
    ├─ 5. AIV0 发布本卡完成 flag
    │
    ├─ 6. 全核按远端 rank 重新分配任务
    │     ├─ 排除本卡 rank
    │     ├─ 每个远端 rank 仅有一个 owner 核
    │     └─ rank 数大于核数时，一个核顺序处理多个 rank/channel
    │
    ├─ 7. owner 核直接发送远端数据
    │     ├─ x[recv_x_idx] -> token WriteNbi
    │     ├─ weight[recv_x_idx] -> weight WriteNbi
    │     ├─ 同核同 channel 发送 so/odr=6 flag
    │     └─ BatchCommit
    │
    └─ 8. Kernel 结束，无第二次 SyncAll
```

`MoeEpCombineEpilogue` 是独立 kernel。其 AIV0 等齐并清除所有 sender flag 后，仍需保留自身的 `SyncAll`，再允许全部 AIV 开始归约；该同步不属于本文所述的 Combine 发送阶段。

## 4. 分核设计

设：

```text
C = 本次实际启用的 AIV 数
R = epWorldSize
r = 本卡 rankId
```

硬件最大 AIV 数可能为 56～72，但 `C` 必须以本次 tiling 下发的实际控核数为准，不能写死。

### 4.1 本卡阶段：按 metadata 记录数分核

本卡区间为：

```text
[localBegin, localEnd)
localCount = localEnd - localBegin
```

所有 `C` 个核使用 quotient/remainder 均分：

```text
base      = localCount / C
remainder = localCount % C

coreCount = base + (aivId < remainder ? 1 : 0)
coreBegin = localBegin + aivId * base + min(aivId, remainder)
coreEnd   = coreBegin + coreCount
```

分核流程：

```text
本卡 metadata 连续区间 [localBegin, localEnd)
                         │
                         ├─ core 0: 连续分片 0
                         ├─ core 1: 连续分片 1
                         ├─ ...
                         └─ core C-1: 连续分片 C-1

每个非空分片：metadata -> recv_x_idx -> token/weight DataCopy
每个空分片：不搬数据，但必须继续到达 SyncAll
```

实际参与搬运的核数最多为 `min(C, localCount)`，但 kernel 仍启动和同步全部 `C` 个核。

### 4.2 远端阶段：按 rank/channel 分核

当前一 rank 一 channel，先将远端 rank 映射为不包含本卡的 ordinal：

```text
remoteRankCount = R - 1
remoteOrdinal in [0, remoteRankCount)

dstRank = remoteOrdinal < r ? remoteOrdinal : remoteOrdinal + 1
```

再将 `remoteRankCount` 均分给全部 `C` 个核：

```text
base      = remoteRankCount / C
remainder = remoteRankCount % C

rankCount = base + (aivId < remainder ? 1 : 0)
rankBegin = aivId * base + min(aivId, remainder)
rankEnd   = rankBegin + rankCount
```

分核流程：

```text
远端 rank ordinal [0, R-1)
          │
          ├─ owner core 0: 完整 rank 区间 0
          ├─ owner core 1: 完整 rank 区间 1
          ├─ ...
          └─ owner core C-1: 完整 rank 区间 C-1

单个 rank 的 metadata 区间不能跨 owner 核拆分
一个 owner 核可以顺序处理多个 rank/channel
```

当 `R - 1 > C` 时，每核处理约 `ceil((R - 1) / C)` 个 rank。例如 `R=1024, C=56` 时，15 个核各处理 19 个远端 rank，其余 41 个核各处理 18 个远端 rank。

当前路由期望各 rank 数据量近似均匀，因此第一版按 rank 数静态均分。若后续出现明显倾斜，再考虑按各 rank 的 metadata 数加权分配，但仍禁止拆分单个 channel。

### 4.3 两阶段复用同一批核

本卡阶段和远端阶段不重新 launch kernel，也不改变 blockDim：

```text
                 本卡阶段                         远端阶段
core 0     local records slice 0       -> remote ranks owned by core 0
core 1     local records slice 1       -> remote ranks owned by core 1
...                    SyncAll
core C-1   local records slice C-1     -> remote ranks owned by core C-1
```

`SyncAll` 只是阶段边界，之后每个核根据同一个 `aivId` 使用另一套切分公式计算远端任务。

## 5. 流水与同步设计

### 5.1 全核同步数量

`MoeEpCombine` 主 kernel 只有一次全核同步：

```text
本卡多核 DataCopy 完成 -> SyncAll #1 -> 本卡 flag + 远端发送
```

不需要以下同步：

- 不再为地址表计数和前缀和执行 `SyncAll`。
- 远端数据和 flag 之间不执行 `SyncAll`。
- 所有远端 owner 完成后不执行最终 `SyncAll`。

### 5.2 核内流水同步

全核 `SyncAll` 不能替代核内流水事件。实现必须满足：

```text
metadata GM -> UB 完成
    -> 才能由标量侧读取 metadata 字段

x/weight GM -> UB 完成
    -> 才能复用该 UB 并写本卡窗口

本核最后一次 UB -> 本卡窗口完成
    -> 才能参加全核 SyncAll
```

建议：

- metadata chunk 使用 `DataCopyPad`，随后显式等待 MTE2 完成。
- token 使用 TQue 的 `AllocTensor -> DataCopyPad -> EnQue -> DeQue -> DataCopyPad -> FreeTensor` 生命周期。
- weight 使用独立的小 buffer/queue，不与 token 拼接。
- 有非空本卡任务的核，在本卡循环结束后执行一次 `MTE3_S`。

### 5.3 flag 顺序

本卡 flag：

- 仅由 AIV0 发布。
- 必须在全核 `SyncAll` 后发布。
- 不依赖 URMA `so/odr`。

远端 flag：

- 每个 rank/channel 仅由其 owner 核发布。
- token、weight 和 flag 使用同一个 channel。
- flag 位于该 rank 所有 token/weight WQE 之后。
- flag 使用 `so/odr=6` 和 fence 配置，保证接收端看到 flag 时该 rank 数据已经可见。
- 即使某个远端 rank 的 metadata 区间为空，也必须发布 flag。

## 6. 建议的 Process 伪代码

```cpp
__aicore__ inline void Process()
{
    // 每核独立准备，避免为 actualA/rank boundary 增加同步。
    PrepareRuntimeInfo();
    InitPerCoreBuffersAndFlagSource();

    uint64_t localBegin = LowerBoundSrcRank(rankId_);
    uint64_t localEnd = LowerBoundSrcRank(rankId_ + 1U);

    uint64_t coreBegin;
    uint64_t coreEnd;
    SplitRange(localBegin, localEnd, aivNum_, aivId_, coreBegin, coreEnd);

    bool didLocalWrite = ProcessLocalMetadataRange(coreBegin, coreEnd);
    if (didLocalWrite) {
        SyncFunc<HardEvent::MTE3_S>();
    }

    // 所有 AIV 必须无条件到达。
    SyncAll<true>();

    if (aivId_ == 0U) {
        SendChannelFlag(rankId_);
    }

    uint32_t remoteOrdinalBegin;
    uint32_t remoteOrdinalEnd;
    SplitRange(0U, epWorldSize_ - 1U, aivNum_, aivId_,
               remoteOrdinalBegin, remoteOrdinalEnd);

    for (uint32_t ordinal = remoteOrdinalBegin;
         ordinal < remoteOrdinalEnd;
         ++ordinal) {
        uint32_t dstRank = ordinal < rankId_ ? ordinal : ordinal + 1U;
        uint64_t rankBegin = LowerBoundSrcRank(dstRank);
        uint64_t rankEnd = LowerBoundSrcRank(dstRank + 1U);

        BeginPreparedWrites(dstRank);
        ProcessRemoteMetadataRange(dstRank, rankBegin, rankEnd);
        SendChannelFlag(dstRank);
        FlushPreparedWrites();
    }
}
```

## 7. 设计约束

### 7.1 正确性约束

- `RECV_META_FIELDS` 和 Host 的 metadata dim1 校验必须同时从 4 改为 5。
- `actualA` 不得超过 `x.shape[0]`、metadata 容量和可选 weight 容量。
- Combine Kernel 热路径不逐行检查 metadata 字段合法性，信任 Dispatch Epilogue 或 cached handle 的 metadata 契约。字段范围和排序必须由 metadata 生产侧保证。
- metadata 有效前缀必须按 `src_rank` 非降序，否则不能使用二分区间定位。
- metadata 行号不能作为 token/weight 源索引。
- token 和 weight 保持独立输入、独立目的区域，不修改通信窗口布局。
- 每个 rank/channel 在一次 kernel 执行中只能有一个 owner 核。
- 一个核可顺序访问多个不同 channel；切换 channel 前必须提交前一个 batch。
- 所有核必须无条件到达唯一的 `SyncAll`，不得因 `actualA==0`、空分片或非法行而提前返回。

### 7.2 规模约束

设计需覆盖：

```text
AIV 数：运行时控核，硬件上限约 56～72
epWorldSize：测试 8/16/32/64/128，上限 1024
numTokens / numMaxTokensPerRank：典型 4096/5120/7168，上限 32768
topK：典型 6/8，上限 32
H：典型 4096/7168/8192，不能按固定上限设计索引
```

所有长度和窗口 offset 计算必须使用足够宽的无符号整数，避免 `numMaxTokensPerRank * topK * perSlotBytes` 溢出。

### 7.3 空数据约束

- `actualA == 0` 时不读取 metadata，但仍需完成一次 `SyncAll`。
- 本卡空区间时，AIV0 在 barrier 后仍发布本卡 flag。
- 远端空 rank 仍由 owner 发布 flag。
- 当 `epWorldSize == 1` 时远端 rank 数为 0，仅执行本卡阶段和本卡 flag。

### 7.4 性能约束

- 本卡按 metadata 记录数切分，不按 rank 数切分。
- 远端按完整 rank/channel 切分，不能为利用更多核而跨核拆同一 channel。
- 第一版按 rank 数均分，依赖实际业务中各 rank 数据量近似均匀的假设。
- `R < C` 时远端阶段最多使用 `R-1` 个核，这是单 channel 约束下的并行上限。
- `R > C` 时一个核顺序处理多个 rank，需在 Hcomm batch 容量边界正确提交。
- lower-bound 的重复计算为 `O(log actualA)`，不应重新引入 `O(actualA)` 的全量 rank 扫描或逐 token workspace。

## 8. 代码实现映射

### 8.1 Combine Kernel

- `RECV_META_FIELDS` 从 4 改为 5，并增加五个字段 offset。
- `Process()` 已切换到 `SendPhaseDirectFromMetadata()`。
- 原地址表入口 `SendPhaseExpertToToken()` 仅在调用点注释，并标记 `TODO(delete)`；相关函数和成员暂时保留，待直接路径完成板上验证后统一删除。
- `IsMetadataRowValid()` 定义暂时保留，本卡和远端热路径调用点均已注释，不产生逐行校验开销。
- 增加：
  - `LowerBoundSrcRank`；
  - `SplitRange`；
  - `ProcessLocalMetadataRange`；
  - `ProcessRemoteMetadataRange`。
- `SendLocalSlot` 和 `SendRemoteSlot` 的源参数改为 `recvXIdx`。
- 本卡 token 和 weight 使用独立搬运路径。
- 有效直接路径按第 6 节流程执行，只保留一次 `SyncAll`。旧地址表函数定义中仍有历史 `SyncAll`，但该入口不再被调用。

### 8.2 Combine Host Tiling

- metadata dim1 校验从 4 改为 5。
- 保留通信窗口大小和 offset 的计算。
- workspace 生效值改为系统预留空间加每 AIV 512B flag source；旧地址表 workspace 公式仅注释并标记 `TODO(delete)`。
- `aivNum` 继续使用本次平台和控核结果，不写死 56 或 72。
- 通过 tiling 新增 `recvCapacity=x.shape[0]`，设备侧将 `actualA` 限制在该容量内；Host 同时保证 metadata 和可选 weight 的 dim0 与 `x` 一致。

### 8.3 不需要修改的模块

- Dispatch 主 kernel：不消费 Combine metadata。
- Dispatch Epilogue：已负责生产 5 列 metadata 和 `recv_x_idx`。
- Combine Epilogue：不读取 metadata；只要窗口布局和每 sender rank 一个完成 flag 的协议不变，就不需要因本设计修改。

## 9. 实现检查项

- [x] 5 列 metadata stride 在 Combine Host 和 Kernel 中一致；现有 Torch extension 常量也是 5。
- [ ] 使用非恒等、非单调的 `recv_x_idx` 验证 token 和 weight 均取对输入行。
- [x] token 和 weight 未拼接，远端保持两次独立 WriteNbi。
- [x] Combine 有效直接路径仅有一次 `SyncAll`。
- [x] 静态检查确认 `actualA==0` 和空分片仍能到达该 `SyncAll`。
- [x] 本卡有写入的核在 barrier 前等待 MTE3 完成。
- [x] 每个远端 rank 的 token、weight、flag 由同一 owner 核访问同一 channel。
- [x] 空 rank 也发送完成 flag。
- [x] metadata 逐行合法性校验已从有效热路径注释掉，Kernel 信任生产侧契约。
- [x] 分核模型检查覆盖 `R<C`、`R=C`、`R>C` 和 `R=1024`，确认 rank 归属无重叠、无遗漏。
- [ ] 覆盖 `actualA=0`、本卡空区间和 metadata chunk 边界。
- [ ] 覆盖 Hcomm batch 容量边界及一个核连续切换多个 channel。
- [ ] 单步执行和正常流水执行均通过，不能只依赖 single-commit 模式验证。

当前环境没有可用的 Ascend C 编译工具链和 NPU，未勾选项需要在 CANN 构建/板上测试环境完成。

## 10. 修订记录

| 版本 | 日期 | 内容 |
|---|---|---|
| 0.1 | 2026-08-28 | 记录 5 列 metadata 直接发送、全核本卡优先、一次 SyncAll 和 rank/channel owner 分核方案 |
| 0.2 | 2026-08-28 | 记录直接路径代码映射、旧地址表保留策略和当前静态验证状态 |
| 0.3 | 2026-08-28 | 按性能约束关闭 Kernel metadata 逐行合法性校验 |
