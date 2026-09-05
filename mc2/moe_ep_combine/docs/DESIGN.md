# MoeEpCombine 基于 metadata 直接发送设计

## 1. 设计目标

本设计用于将 `MoeEpCombine` 从“扫描 metadata 并重建逐 token 地址表”改为“直接根据 5 列 metadata 定位 `x` 和 `topk_weights` 并发送”。

目标如下：

- 不构建逐 token 地址表，不产生地址表的 GM 写入、回读和计数前缀和。
- 使用 metadata 第 5 列 `recv_x_idx` 定位 token 和 weight 的输入行。
- token 与 weight 保持独立输入和独立发送，不拼接 payload。
- 本卡使用全部 AIV 并行执行 DataCopy，完成后再进入远端发送阶段。
- 复用主线的 rank/channel 分核逻辑，单个 channel 在一次 kernel 中只由一个核访问。
- `MoeEpCombine` 主 kernel 只保留一次全核 `SyncAll`。
- 保持接收完成 flag 协议不变；Combine state 尾部增加 512B 常量发送源，后续数据区偏移统一后移。

每个 rank 的实际通道数沿用主线算法，由 token slot 数、可用 Hcomm handle 数和 7 通道上限共同决定。

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
    │     └─ 初始化每核 Hcomm batch；AIV0 初始化通信窗口中的共享 flag source
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
    ├─ 5. 本卡 owner 核按主线协议发布各通道完成 flag
    │
    ├─ 6. 复用主线 GetCoreAssignment 分配 rank/channel
    │     ├─ 核数充足时，同一 rank 可由多个 channel owner 核分片
    │     └─ rank 数大于核数时，一个核顺序处理多个 rank
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

### 4.2 发送阶段：复用主线 rank/channel 分核

通道数沿用主线计算：

```text
Q = min(ceil(numMaxTokensPerRank * topK / 32368),
        channelsPerRank,
        7)
```

当 `C >= R` 时，`GetCoreAssignment` 将 AIV 按 rank 分组；同一 rank 的 metadata 连续区间再按组内核数均分，
`coreIndexInGroup` 同时作为 channel index。若 `C > R * Q`，远端 rank 各使用 `Q` 个 channel，
其余 AIV 分配给本卡拷贝和完成标记。

当 `C < R` 时，每个 AIV 按 `rank += C` 顺序处理多个 rank，并使用 channel 0。以上分配保持主线
Combine 与 Combine Epilogue 的完成 flag 数量协议不变。

排序后的 metadata 和 `recv_rank_offsets` 组成隐式地址表：

```text
entryCount(rank) = offsets[rank + 1] - offsets[rank]
metadataRow      = offsets[rank] + relativeEntry
```

因此原有 `GetAddressTableCount`、`SendAddressTableRange` 和 `SendPhaseExpertToToken` 可以直接改为读取
该隐式地址表，无需新增一套并行的 direct-path 函数，也无需 GM 地址表 workspace。

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

- 由 `GetCoreAssignment` 分配到本卡 rank 的 owner 核发布，通道数与 Combine Epilogue 的
  `GetCompletionChannelCount` 一致。
- 必须在全核 `SyncAll` 后发布。
- 本卡直接写 state window，不依赖 URMA `so/odr`。

远端 flag：

- 每个 rank/channel 仅由其 owner 核发布。
- token、weight 和 flag 使用同一个 channel。
- flag 位于该 channel 所有 token/weight WQE 之后。
- flag 使用 `so/odr=6` 和 fence 配置，保证接收端看到 flag 时该 channel 数据已经可见。
- 即使某个远端 rank 的 metadata 区间为空，也必须发布约定数量的 flag。

远端 flag 的非 inline 发送源位于本卡持久通信窗口，不使用临时算子 workspace：

```text
Combine state = [nmt * topK 个 512B token/weight 槽]
              + [epWorldSize * combineChannelCount 个 512B 接收完成 flag 槽]
              + [1 个 512B 常量发送源槽]
```

`combineFlagSourceWinOffset` 由公共窗口布局计算并通过 Combine tiling 传入。每次 Combine 的 AIV0 将源槽
写为 64 个 `uint64_t(1)`；`BuildAddressTable()` 原有的全核 `SyncAll` 同时保证本卡数据搬运和常量源初始化
完成，之后各 owner 核才能提交远端 flag。该源槽不在 Combine Epilogue 的接收 flag 清零范围内。

## 6. 建议的 Process 伪代码

```cpp
__aicore__ inline void SendPhaseExpertToToken()
{
    ValidateRankOffsets();

    // 保留主线函数名和阶段边界；函数内部直接处理本卡的隐式 metadata 地址表。
    BuildAddressTable();
    ResetBatchState();

    bool splitRankTokens = aivNum_ >= epWorldSize_;
    if (splitRankTokens) {
        GetCoreAssignment(aivNum_, targetRank, coreIndexInGroup, groupSize);
        uint32_t count = GetAddressTableCount(targetRank);
        auto [entryStart, entryEnd] = Split(count, groupSize, coreIndexInGroup);
        SendAddressTableRange(targetRank, entryStart, entryEnd, coreIndexInGroup);
    } else {
        for (uint32_t rank = aivId_; rank < epWorldSize_; rank += aivNum_) {
            SendAddressTableRange(rank, 0U, GetAddressTableCount(rank), 0U);
        }
    }

    PublishAssignedChannelFlags();
    FlushPreparedWrites();
}
```

这里的“地址表”是排序 metadata 与 `recv_rank_offsets` 组成的隐式表；上述三个原有函数均不再分配或访问
逐 token GM 地址表 workspace。

## 7. 设计约束

### 7.1 正确性约束

- `RECV_META_FIELDS` 和 Host 的 metadata dim1 校验必须同时从 4 改为 5。
- `recv_rank_offsets` 必须是 `[ep_world_size + 1] int32`，满足首项为 0、单调非降且末项为
  `actualA`；源 rank `r` 对应区间为 `[offsets[r], offsets[r + 1])`。
- `actualA` 不得超过 `x.shape[0]`、metadata 容量和可选 weight 容量。
- Combine Kernel 热路径不逐行检查 metadata 字段合法性，信任 Dispatch Epilogue 或 cached handle 的 metadata 契约。字段范围和排序必须由 metadata 生产侧保证。
- metadata 有效前缀必须按 `src_rank` 非降序，并与 `recv_rank_offsets` 描述的 rank 区间一致。
- metadata 行号不能作为 token/weight 源索引。
- token 和 weight 保持独立输入、独立目的区域；通信窗口仅在 Combine state 尾部增加共享 flag source。
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
- 本卡空区间时，分配到本卡 rank 的 owner 核在 barrier 后仍发布约定数量的 flag。
- 远端空 rank 仍由 owner 发布 flag。
- 当 `epWorldSize == 1` 时远端 rank 数为 0，仅执行本卡阶段和本卡 flag。

### 7.4 性能约束

- 本卡按 metadata 记录数切分，不按 rank 数切分。
- 远端沿用主线 rank/channel 分组；同一 channel 不跨核，同一 rank 可按多个 channel 分片。
- 每个 rank 内按 metadata 记录数均分给组内 owner 核。
- `R < C` 时可使用多 channel 提升单 rank 并行度，通道数仍受 handle 数和 7 通道上限约束。
- `R > C` 时一个核顺序处理多个 rank，需在 Hcomm batch 容量边界正确提交。
- rank 边界必须直接来自 Epilogue 输出，不应重新引入 lower-bound、`O(actualA)` 的全量 rank 扫描
  或逐 token workspace。

## 8. 代码实现映射

### 8.1 Combine Kernel

- `RECV_META_FIELDS` 从 4 改为 5，并增加 Combine 使用的字段 offset。
- 增加 `recv_rank_offsets` 输入，每核一次性连续搬入 UB；`actualA` 取最后一个 offset。
- `Process()` 继续调用主线 `SendPhaseExpertToToken()`，不新增并行入口。
- 原 `BuildAddressTable()` 改为按 rank offsets 并行完成本卡拷贝和阶段同步，不再生成 GM 地址表。
- 原 `GetAddressTableCount()` 直接计算相邻 rank offset 的差值。
- 原 `SendAddressTableRange()` 直接遍历排序 metadata 的目标区间，保留主线多通道 batch 发送逻辑。
- 移除地址表计数、前缀和、逐 token workspace 及相关 UB 成员。
- `SendLocalSlot` 和 `SendRemoteSlot` 的源参数改为 `recvXIdx`。
- 本卡 token 和 weight 使用独立搬运路径。
- 有效路径按第 6 节流程执行，只保留一次 `SyncAll`。

### 8.2 Combine Host Tiling

- metadata dim1 校验从 4 改为 5。
- 增加 `recv_rank_offsets` 的一维 shape、长度和 int32 dtype 校验。
- 公共窗口布局在 Combine 接收 flag 后增加 512B 持久 flag source，并统一更新后续 offset 和容量检查。
- Combine 私有 workspace 不再保存 flag source，仅保留框架要求的系统预留空间。
- `aivNum` 继续使用本次平台和控核结果，不写死 56 或 72。
- 通过 tiling 新增 `recvCapacity=x.shape[0]`，设备侧将 `actualA` 限制在该容量内；Host 同时保证 metadata 和可选 weight 的 dim0 与 `x` 一致。

### 8.3 不需要修改的模块

- Dispatch 主 kernel：不消费 Combine metadata。
- Dispatch Epilogue：负责生产 5 列 metadata、`recv_x_idx` 和 `recv_rank_offsets`。
- Combine Epilogue：不读取 metadata；继续按主线动态通道数等待并清除每个 sender rank 的完成 flag，
  通过公共 Host 布局获取后移后的数据区 offset。

## 9. 实现检查项

- [x] 5 列 metadata stride 在 Combine Host 和 Kernel 中一致；现有 Torch extension 常量也是 5。
- [ ] 使用非恒等、非单调的 `recv_x_idx` 验证 token 和 weight 均取对输入行。
- [x] token 和 weight 未拼接，远端保持两次独立 WriteNbi。
- [x] Combine 有效直接路径仅有一次 `SyncAll`。
- [x] 静态检查确认 `actualA==0` 和空分片仍能到达该 `SyncAll`。
- [x] 本卡有写入的核在 barrier 前等待 MTE3 完成。
- [x] 每个远端 rank/channel 的 token、weight、flag 由同一 owner 核访问。
- [x] 空 rank 也发送完成 flag。
- [x] metadata 热路径不做逐行合法性校验，Kernel 信任生产侧契约。
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
| 0.4 | 2026-09-04 | 增加 Epilogue rank offsets 输出，停用 Combine 的 actualA 归约和 rank lower-bound 调用 |
