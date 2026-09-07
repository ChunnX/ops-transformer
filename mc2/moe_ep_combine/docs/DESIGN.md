# MoeEpCombine 基于 metadata 直接发送设计

## 1. 设计目标

本设计用于将 `MoeEpCombine` 从“扫描 metadata 并重建逐 token 地址表”改为“直接根据 5 列 metadata 定位 `x` 和 `topk_weights` 并发送”。

目标如下：

- 不构建逐 token 地址表，不产生地址表的 GM 写入、回读和计数前缀和。
- 使用 metadata 第 5 列 `recv_x_idx` 定位 token 和 weight 的输入行。
- token 与 weight 保持独立输入和独立发送，不拼接 payload。
- 本卡使用全部 AIV 并行执行 DataCopy，完成后再进入远端发送阶段。
- 远端沿用 upstream 动态多 channel；每个 rank/channel 仅由一个核访问。
- `MoeEpCombine` 主 kernel 只保留一次全核 `SyncAll`。
- 保持接收完成 flag 协议不变；Combine state 尾部增加 512B 常量发送源，后续数据区偏移统一后移。

已创建 channel 数、Combine 名义 channel 数、实际 owner 数是不同概念；实际 owner 还受 AIV 数和目标 rank 影响。发送端与接收端使用第 4.2 节的同一规则。

## 2. 输入与 metadata 契约

### 2.1 有效数据范围

```text
actualA = recv_rank_offsets[ep_world_size]
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

Dispatch Epilogue 输出 `recv_rank_offsets[ep_world_size + 1]`。它是每个源 rank metadata
行数的 exclusive prefix，因此任意 rank 的连续区间可直接读取：

```text
rank_begin = recv_rank_offsets[rank]
rank_end   = recv_rank_offsets[rank + 1]
```

其中 `recv_rank_offsets[0] = 0`，`recv_rank_offsets[ep_world_size] = actualA`。该过程不需要
扫描 metadata、执行二分查找或构建逐 token 地址表，也不要求同一 rank 内的
`recv_x_idx` 有序。

## 3. 总体执行流程

```text
所有 AIV 启动
    │
    ├─ 1. 初始化
    │     ├─ 读取 tiling、rank、通信窗口和 Hcomm context
    │     ├─ 连续读取 recv_rank_offsets
    │     ├─ 从最后一个 offset 得到 actualA
    │     ├─ 初始化 metadata UB、token queue、weight buffer/queue
    │     └─ 初始化每核 Hcomm batch 和 UB flag；AIV0 初始化 win 中共享的 flag source
    │
    ├─ 2. 定位本卡 metadata 区间
    │     └─ 直接读取 offsets[rankId:rankId + 2]
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
    ├─ 5. 沿用 upstream 三分支分配 rank/channel owner
    │     ├─ 核数充足：远端组在前，本卡组在后
    │     ├─ 核数中等：按 rank 均分 AIV，组内索引即 channel
    │     └─ 核数小于 EP：每核按 rank = aivId + n*AIV 轮转，仅使用 channel 0
    │
    ├─ 6. 每个 owner 完成本核远端 payload 阶段
    │     ├─ 本卡 payload 已完成，不重复搬运
    │     ├─ x[recv_x_idx] -> token WriteNbi
    │     └─ weight[recv_x_idx] -> weight WriteNbi
    │
    ├─ 7. 每核进入自己的 flag 阶段并最终 Flush
    │     ├─ 本卡 flag 由本卡组 owner 发布，不再由 AIV0 集中发布
    │     ├─ 远端 flag 与 payload 使用同核同 channel
    │     ├─ 低核数分支再次轮转本核全部 rank，发布 channel 0 flag
    │     └─ BatchCommit；只有 SQ 高水位检查点执行 Drain
    │
    └─ 8. 保留全 data cache DCCI 和原有诊断记录，结束，无第二次 SyncAll
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

### 4.2 通信阶段：恢复 upstream 三分支 owner 分核

区分三个量：

```text
P = max(context.channelsPerRank, 1)  // 已创建的每 rank channel 数
Q = min(max(ceil(numMaxTokensPerRank * topK / 32368), 1), P, 7)
                                    // Combine 名义 channel 数
D(dstRank) = 本次向目标 rank 实际发布 flag 的 channel 数
```

当前 context 在无 UBG 且 `R<64` 时创建 `P=64/R`（整数除法），否则创建 1。
Combine 从 context 读取 P；Q 决定 flag 的 rank stride，D 决定实际启用的 owner 和等待数量。

`GetCoreAssignment` 恢复为有效的分核 helper，不恢复旧地址表构建、计数、workspace 或发送路径：

| 条件 | rank/channel owner 归属 | D(dstRank) |
|---|---|---|
| `C>R*Q` | 前 `(R-1)*Q` 核按远端 rank（跳过本卡）分组，每组 Q 核；剩余核归本卡组，仅其前 Q 核发本卡 flag | Q |
| `R<=C<=R*Q` | 按 rank quotient/remainder 均分 AIV；组内索引就是 channel，远端 metadata 按该组大小均分 | `C/R + (dstRank<C%R ? 1 : 0)` |
| `C<R` | core i 轮转 rank `i+n*C`，每个 rank 只使用 channel 0 | 1 |

第一分支远端核的 `ordinal=aivId/Q`、`channel=aivId%Q`，目标 rank 跳过本卡；
本卡组内索引为 `aivId-(R-1)*Q`。第二分支每组起点为
`dstRank*(C/R)+min(dstRank,C%R)`，metadata 必须按 groupSize 而不是名义 Q 切分。
“本卡组”只描述此阶段 flag 的归属，本卡 payload 仍由前一阶段全部 C 核搬运，不能重复搬运。

第三分支每核先遍历自己的全部 rank，发送所有远端 payload，再第二次遍历同一列表发布
本卡或远端的 channel 0 flag，最后 `FlushPreparedWrites()`。这是每核自己的先 payload、
后 flags 顺序，不增加跨核 barrier，也不是每处理完一个 rank 就立即发其 flag。
batch 满或切换 channel 时仍提交已有 WQE；空 metadata 分片也必须发 flag。

所有分支的 flag 地址均为 `senderRank*Q+channel`。epilogue 读取并清零 `R*Q` 个槽，
但期望收到的有效 flag 数为 `R*D(本卡rank)`；D<Q 时未使用 channel 槽保持 0。
发送与 epilogue 必须使用一致的 C、R、P 和 Q。

例如 `EP=8、nmt=4096、topK=8、P=8`，Q=2：

| AIV 数 C | 实际分核 | 每张目标卡期望 flag 数 |
|---|---|---|
| 64 | 前 14 核负责远端；后 50 核是本卡组，仅 core 14、15 发本卡 flag | 16 |
| 12 | rank 0～3 各 2 核，rank 4～7 各 1 核 | 目标 rank 0～3 等 16 个，rank 4～7 等 8 个 |
| 4 | core i 负责 rank i、i+4，均用 channel 0 | 8 |

三个例子都读取/清零 16 个 flag 槽，本卡 payload 分别由 64、12、4 核均分。
R=1 时没有远端 payload，本卡组按相同规则发布 D(0) 个 flag。

### 4.3 两阶段复用同一批核

本卡阶段和远端阶段不重新 launch kernel，也不改变 blockDim：

```text
                 本卡 payload 阶段                 通信/flag 阶段
core 0     local records slice 0       -> 本核的 rank/channel owner 任务
core 1     local records slice 1       -> 本核的 rank/channel owner 任务
...                    SyncAll
core C-1   local records slice C-1     -> 本核的 rank/channel owner 任务
```

`SyncAll` 只是阶段边界，之后每核按第 4.2 节计算 owner 任务。本卡 flag owner 不必是 AIV0；
AIV0 仍负责初始化所有远端 flag 共用的持久发送源。

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
- Combine epilogue 清 flag 前保留 `S_V`，清零使用 `V_MTE3`，清零搬出后保留 `MTE3_S`，不以 `SyncAll` 替代核内依赖。

### 5.3 flag 顺序

本卡 flag：

- 由第 4.2 节的本卡组 owner 发布，共 D(r) 个；低核数分支由轮转列表包含本卡 rank 的核发布 channel 0。
- 必须在全核 `SyncAll` 后发布。
- 不依赖 URMA `so/odr`。

远端 flag：

- 每个 rank/channel 仅由其 owner 核发布。
- token、weight 和 flag 使用同一个 channel。
- flag 位于该 rank/channel 分片所有 token/weight WQE 之后；低核数分支先完成本核全部 rank 的 payload 阶段，再进入全部 flags 阶段。
- flag 使用 `so/odr=6` 和 fence 配置，保证接收端看到 flag 时对应 channel 的数据已经可见；收齐全部 channel 后归约。
- 即使某个远端 rank 的 metadata 区间为空，也必须发布 flag。

远端 flag 的非 inline 发送源位于本卡持久通信窗口，不再位于临时算子 workspace：

```text
Combine state = [nmt * topK 个 512B token/weight 槽]
              + [epWorldSize * 7 个 512B 接收完成 flag 预留槽]
              + [1 个 512B 常量发送源槽]
```

`combineFlagSourceWinOffset` 位于完整的 7-channel 预留区之后，由公共布局计算并通过 Combine tiling 传入。
接收 flag 按名义 Q 排列为 `senderRank * Q + channel`，固定预留上限 7 不是实际 stride。
只有 `channel<D(目标rank)` 的槽实际发布；epilogue 读取/清零 `epWorldSize*Q` 个槽，
期望的有效通知数量为 `epWorldSize*D(本卡rank)`。
每次 Combine 的 AIV0 将源槽完整写为 64 个 `uint64_t(1)`，经 `MTE3_S` 后，
复用本卡搬运之后已有的全核 `SyncAll` 向远端发送核保证初始化完成。
源槽不参与接收 flag 清零，后续调用只重写相同常量；单跑 Combine 无需依赖 Dispatch 初始化。
末尾只提交 batch，不执行无条件 Drain；SQ 水位保护仍保留。
四个算子必须使用更新后的公共布局，Python 容量估算同步增加每卡 512B，并重新创建通信窗口。

## 6. Process 伪代码

```cpp
PrepareRuntimeInfo();
InitFlagSource();
SplitRange(offsets[rankId_], offsets[rankId_ + 1], aivNum_, aivId_, localBegin, localEnd);
if (ProcessLocalMetadataRange(localBegin, localEnd)) {
    SyncFunc<HardEvent::MTE3_S>();
}
SyncAll<true>();

bool splitRankTokens = aivNum_ >= epWorldSize_;
if (splitRankTokens) {
    GetCoreAssignment(aivNum_, targetRank, coreIndexInGroup, groupSize);
}
if (actualA_ != 0U) {
    if (splitRankTokens) {
        if (targetRank != rankId_) {
            SplitRange(offsets[targetRank], offsets[targetRank + 1],
                       groupSize, coreIndexInGroup, begin, end);
            ProcessRemoteMetadataRange(targetRank, begin, end, coreIndexInGroup);
        }
    } else {
        // 第一遍：先发送本核所有远端 rank 的 payload。
        for (uint32_t rank = aivId_; rank < epWorldSize_; rank += aivNum_) {
            if (rank != rankId_) {
                ProcessRemoteMetadataRange(rank, offsets[rank], offsets[rank + 1], 0);
            }
        }
    }
}
// 空数据也必须发布 flags，本卡 payload 不在此处重做。
if (splitRankTokens) {
    if (coreIndexInGroup < combineChannelCount_) {
        SendChannelFlag(targetRank, coreIndexInGroup);
    }
} else {
    // 第二遍：再发本核所有 rank 的 channel 0 flag，包含本卡。
    for (uint32_t rank = aivId_; rank < epWorldSize_; rank += aivNum_) {
        SendChannelFlag(rank, 0);
    }
}
FlushPreparedWrites();
// 保留整个 data cache 的 DCCI 和原有发送完成位置诊断记录。
```

## 7. 设计约束

### 7.1 正确性约束

- `RECV_META_FIELDS` 和 Host 的 metadata dim1 校验必须同时从 4 改为 5。
- `recv_rank_offsets` 必须是 `[ep_world_size + 1] int32`，满足首项为 0、单调非降且末项为
  `actualA`；源 rank `r` 对应区间为 `[offsets[r], offsets[r + 1])`。
- `actualA` 不得超过 `x.shape[0]`、metadata 容量和可选 weight 容量。
- Combine Kernel 热路径不逐行检查 metadata 字段合法性，信任 Dispatch Epilogue 或 cached handle 的 metadata 契约。字段范围和排序必须由 metadata 生产侧保证。
- metadata 有效前缀必须按 `src_rank` 非降序，并与 `recv_rank_offsets` 描述的 rank 区间一致。
- metadata 行号不能作为 token/weight 源索引。
- token 和 weight 保持独立输入、独立目的区域；公共通信窗口布局额外预留第 5.3 节所述的常量 flag 源槽。
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
- 本卡空区间时，本卡组 owner 在 barrier 后仍发布各自的本卡 flag。
- 远端空 rank 仍由 owner 发布 flag。
- 当 `epWorldSize == 1` 时远端 rank 数为 0，仅执行本卡阶段和本卡 flag。

### 7.4 性能约束

- 本卡按 metadata 记录数切分，不按 rank 数切分。
- 远端按完整 rank/channel 切分，不能为利用更多核而跨核拆同一 channel。
- C>R*Q 时远端使用 `(R-1)*Q` 个 owner 核；R<=C<=R*Q 时使用 `C-groupSize(r)` 个远端 owner 核，rank 内 metadata 按该 rank 的 groupSize 均分。
- C<R 时一个核轮转多个 rank，各 rank 只使用 channel 0，最多 C 核参与远端阶段；不把全部 Q 个 channel 任务平摊给少量核。
- 本卡 flag 由 upstream 对应的 owner 发布；C>=R 时使用本卡组，不占用远端组做集中式通知；C<R 时轮转核可同时负责本卡 flag 和其他 rank 的远端通信。
- 切换 channel 和 batch 满时均正确提交；仅 SQ 高水位 CQE 检查点 Drain，末尾不做无条件 Drain。
- rank 边界必须直接来自 Epilogue 输出，不应重新引入 lower-bound、`O(actualA)` 的全量 rank 扫描
  或逐 token workspace。

## 8. 代码实现映射

### 8.1 Combine Kernel

- `RECV_META_FIELDS` 从 4 改为 5，并增加五个字段 offset。
- 增加 `recv_rank_offsets` 输入，每核一次性连续搬入 UB；`actualA` 取最后一个 offset。
- `Process()` 已切换到 `SendPhaseDirectFromMetadata()`。
- 已删除原地址表入口及其构表、计数、发送函数，以及专属 workspace 指针、UB buffer 和常量。
- `GetCoreAssignment` 已恢复为直接 metadata 路径使用的有效分核 helper，保持 upstream 三分支 owner 归属；不恢复旧地址表数据路径。
- 已删除不再调用的逐行合法性校验函数，热路径继续信任生产侧 metadata 契约。
- 增加：
  - `SplitRange`；
  - `ProcessLocalMetadataRange`；
  - `ProcessRemoteMetadataRange`。
- 已删除 rank lower-bound 查询和旧 actualA 归约代码，直接使用 rank offsets。
- `SendLocalMetadataSlot` 和 `SendRemoteMetadataSlot` 使用 `recvXIdx` 定位输入；旧的两套 slot 发送函数已删除。
- 本卡 token 和 weight 使用独立搬运路径。
- 有效直接路径按第 6 节流程执行，只保留一次 `SyncAll`；原有诊断记录、DCCI、SQ 高水位 CQE/Drain 均保留。

### 8.2 Combine Host Tiling

- metadata dim1 校验从 4 改为 5。
- 增加 `recv_rank_offsets` 的一维 shape、长度和 int32 dtype 校验。
- 公共布局在 Combine 接收 flag 后预留 512B，统一更新 direct/hybrid 的后续偏移及容量检查。
- 删除已无消费者的每 AIV 512B workspace 预留，仅保留 `SYSTEM_NEED_WORKSPACE`。
- 删除旧地址表专用 tiling 字段；Host 和 Kernel 必须成套重新编译，不能混用清理前后的 tiling 二进制布局。
- `aivNum` 继续使用本次平台和控核结果，不写死 56 或 72。
- 通过 tiling 新增 `recvCapacity=x.shape[0]`，设备侧将 `actualA` 限制在该容量内；Host 同时保证 metadata 和可选 weight 的 dim0 与 `x` 一致。

### 8.3 相邻模块配合

- Dispatch 主 kernel：不消费 Combine metadata。
- Dispatch Epilogue：负责生产 5 列 metadata、`recv_x_idx` 和 `recv_rank_offsets`。
- Combine Epilogue：不读取 metadata；按同一规则计算 Q 和 D(本卡rank)，读取/清零 `epWorldSize*Q` 个槽，等待 `epWorldSize*D(本卡rank)` 个有效通知，通过公共 Host 布局获取更新后的数据区偏移。

## 9. 实现检查项

- [x] 5 列 metadata stride 在 Combine Host 和 Kernel 中一致；现有 Torch extension 常量也是 5。
- [ ] 使用非恒等、非单调的 `recv_x_idx` 验证 token 和 weight 均取对输入行。
- [x] token 和 weight 未拼接，远端保持两次独立 WriteNbi。
- [x] Combine 有效直接路径仅有一次 `SyncAll`。
- [x] 静态检查确认 `actualA==0` 和空分片仍能到达该 `SyncAll`。
- [x] 本卡有写入的核在 barrier 前等待 MTE3 完成。
- [x] 每个远端 rank/channel 分片的 token、weight、flag 由同一 owner 核访问同一 channel。
- [x] 空 rank 也发送完成 flag。
- [x] metadata 逐行合法性校验旧代码已删除，Kernel 信任生产侧契约。
- [x] 分核模型检查覆盖 `C>R*Q`、`R<=C<=R*Q`、`C<R` 及边界，确认 upstream owner 归属、payload 分片与 D 计数一致；此项为纯算法检查，不替代上板验证。
- [x] 纯算法覆盖 `actualA=0`、本卡空区间及 channel 分片边界，空数据仍发布完整 flags。
- [ ] 板上覆盖 `actualA=0`、本卡空区间和 metadata chunk 边界。
- [ ] 覆盖 Hcomm batch 容量边界及一个核连续切换多个 channel。
- [ ] 单步执行和正常流水执行均通过，不能只依赖 single-commit 模式验证。

当前环境没有可用的 Ascend C 编译工具链和 NPU，未勾选项需要在 CANN 构建/板上测试环境完成。

## 10. 修订记录

| 版本 | 日期 | 内容 |
|---|---|---|
| 0.1 | 2026-08-28 | 记录 5 列 metadata 直接发送、全核本卡优先、一次 SyncAll 和 rank/channel owner 分核方案 |
| 0.2 | 2026-08-28 | 记录直接路径代码映射、旧地址表保留策略和当前静态验证状态 |
| 0.3 | 2026-08-28 | 按性能约束关闭 Kernel metadata 逐行合法性校验 |
| 0.4 | 2026-09-04 | 增加 Epilogue rank offsets 输出，停用 Combine 的 actualA 归约和 rank lower-bound 调用 |
| 0.5 | 2026-09-07 | 在已跑通版本上删除废弃地址表路径、旧 workspace 及 tiling 字段，保留有效同步与发送逻辑 |
| 0.6 | 2026-09-07 | Rebase upstream 32d64c27f：动态多 channel 分片收发，完整预留区之后放常量 flag 源 |
| 0.7 | 2026-09-07 | Codex：恢复 upstream GetCoreAssignment 三分支 owner；区分 P/Q/D、本卡组 flag、低核数先 payload 后 flags 及接收端实际等待数，保留 metadata 直接发送和持久 flag 源 |
