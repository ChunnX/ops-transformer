# MoeEpCombine 地址表改造前后逻辑

## 1. 文档范围

本文说明 `mc2/moe_ep_combine/op_kernel/moe_ep_combine.h` 从原始重复扫描逻辑、初版提交
`8a1e5ef1b Build combine send address table`，到当前按 dispatch PR 10309 重构后的发送流程，重点解释：

- 为什么单 channel 下远端通信核仍有较长的 Scalar 扫描时间；
- 地址表如何由所有 AIV 共同生成；
- 远端通信核和本地拷贝核如何消费地址表；
- workspace、同步、通信 channel 和完成 flag 是否发生变化；
- 改造带来的收益和新增开销。

该改造参考了 dispatch PR 10309 的计数、per-core prefix 和批量写表思路。Combine 的输入 metadata
已经包含源 rank 和目标 slot，因此不需要 dispatch 中的 payload stash、rank 去重和目标 slot 推导；
这里只计算地址表写入位置的 prefix。

## 2. Combine metadata 与发送目标

`recvSrcMetadata` 中每个 token 有 4 个 `int32_t` 字段：

```text
metadata[tokenIndex][0] = srcRank
metadata[tokenIndex][1] = srcTokenIdx
metadata[tokenIndex][2] = srcTopKIdx
metadata[tokenIndex][3] = 保留字段
```

其中：

- `tokenIndex`：当前卡 expert 输出中的 token 下标，用于定位本卡 `x`；
- `srcRank`：combine 结果需要返回的目标 rank；
- `srcTokenIdx`：目标 rank 上的原始 token 下标；
- `srcTopKIdx`：该 token 的 topK 位置。

目标 payload slot 的字节偏移为：

```cpp
sendTokenOffset =
    (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
```

目标 weight/state slot 的字节偏移为：

```cpp
recvStateOffset =
    (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;
```

## 3. 修改前：每个通信核扫描原始 metadata

### 3.1 核分工

修改前通过 `GetCoreAssignment()` 分配核。

以常见配置为例：

```text
epWorldSize     = 8
channelsPerRank = 1
aivNum          = 48
```

此时：

```text
maxChannelAivNum = epWorldSize * channelsPerRank = 8
remoteAivNum     = (epWorldSize - 1) * channelsPerRank = 7
```

核分工如下：

| AIV | 任务 | groupSize |
| --- | --- | ---: |
| AIV 0～6 | 分别负责 7 个远端 rank | 1 |
| AIV 7～47 | 负责本 rank 的本地拷贝 | 41 |

远端 rank 会跳过本 rank。例如本 rank 为 3 时，前 7 个核依次负责 rank
`0、1、2、4、5、6、7`。

### 3.2 扫描逻辑

修改前每个发送核直接扫描 `recvSrcMetadata`：

```cpp
uint64_t scanStart = 0;
uint64_t scanEnd = actualA_;

if (splitRankTokens && groupSize > 1) {
    // 只有同一 rank 分到多个核时才切分区间
    Split(actualA_, groupSize, coreIndexInGroup, scanStart, scanEnd);
}

for (uint64_t tokenIndex = scanStart; tokenIndex < scanEnd; ++tokenIndex) {
    int32_t srcRank = metadata[tokenIndex][0];
    if (srcRank != targetRank) {
        continue;
    }
    SendSlot(...);
}
```

单 channel 时，远端核的 `groupSize == 1`，所以每个远端核的范围都是：

```text
[0, actualA_)
```

每个远端核都要完整读取和遍历全部 metadata，只保留属于自己目标 rank 的条目。

本地 41 个核的 `groupSize == 41`，会把 `[0, actualA_)` 切成 41 段，因此本地核合计扫描一次
`actualA_`。

### 3.3 修改前的总扫描量

8 rank、单 channel 场景下：

```text
7 个远端核：7 * actualA_
本地核合计：1 * actualA_
--------------------------------
总扫描量：   8 * actualA_
```

这也是前 7 个远端核即使暂时关闭远端发送，仍可能出现约 600 us Scalar 时间的原因：
它们仍在完整搬运和扫描 metadata。

### 3.4 修改前的完整时序

```text
每个远端通信核
    |
    +-- 搬运完整 metadata 到 UB
    +-- Scalar 遍历 actualA_
    +-- 过滤 srcRank == targetRank
    +-- 对匹配项调用 SendSlot
    |       +-- 准备 payload WriteNbi
    |       +-- 可选：准备 weight WriteNbi
    +-- 发送 channel completion flag
    +-- FlushPreparedWrites

本地核组
    |
    +-- 分片搬运 metadata 到 UB
    +-- Scalar 遍历各自区间
    +-- 过滤 srcRank == rankId_
    +-- GM -> UB -> 本地 receive window
```

## 4. 修改后：所有核建表，通信核按 rank 消费

修改后把流程拆成两个阶段，其中建表阶段内部采用与 PR 10309 相同的“计数 → 前缀 → 批量写表”结构：

```text
阶段 1：所有 AIV 分片计数并计算 prefix；本地条目直接拷贝，远端条目批量生成地址表
                                  |
                             SyncAll<true>()
                                  |
阶段 2：通信核只读取远端 rank 的地址表，并独立发送 payload 与 completion flag
```

原来的 `SendSlot()`、batch WriteNbi、channel 分配和 completion flag 协议保持不变。

## 5. 阶段一：BuildAddressTable

### 5.1 初始化 UB 局部计数

每个 rank 有一个独立的 512B counter slot：

```text
rankCountWorkspace[rank][0] = 当前 rank 的地址表条目数
```

每个 AIV 只清零自己的 UB per-rank 计数行：

```cpp
Duplicate(rankCounts, 0);
```

每个 rank 的计数占一个 32B block，实际值放在 block 的第一个 `int32_t`。这里不清零 GM 总计数，
也不调用 `WriteGmByPassDCache()`；最终总数会由最后一个 AIV 根据 prefix 直接覆盖写入。

### 5.2 所有核切分 actualA

所有 `aivNum_` 个核共同切分 `[0, actualA_)`：

```cpp
tokensPerCore = actualA_ / aivNum_;
remainder = actualA_ % aivNum_;

scanStart = aivId_ * tokensPerCore + min(aivId_, remainder);
scanEnd = scanStart + tokensPerCore + (aivId_ < remainder ? 1 : 0);
```

48 个 AIV 时，每个核只扫描大约：

```text
actualA_ / 48
```

计数阶段所有核合计扫描一次原始 metadata；后面的组表阶段会再扫描一次。

### 5.3 保存 per-core 计数行

每个核第一次扫描自己的区间，只在 UB 中增加对应远端 rank 的局部计数；本 rank 不进入地址表计数：

```cpp
rankCounts[srcRank * 8]++;
```

扫描完成后，每个核执行一次普通 MTE3 `DataCopy`，把自己的完整 per-rank 计数行写入：

```text
perCoreRankCountWorkspace[aivId][rank]
```

所有核写完后执行 `SyncAll<true>()`。整个过程没有 `AtomicAdd`、`SetAtomicAdd` 或 `SetAtomicNone`。

### 5.4 exclusive prefix 分配无冲突区间

完成一次 `SyncAll<true>()` 后，AIV `c` 分组搬入编号小于 `c` 的 per-core 计数行。对每个远端 rank `r`：

```text
rankPrefix[c][r] = sum(perCoreRankCount[0 .. c-1][r])
```

每组搬入后使用与 PR 10309 相同的RA归约和Vector累加：

```cpp
ReduceSum<int32_t, Pattern::Reduce::RA, true>(prefixSums, prefixRows, shape, false);
Add(rankPrefix, rankPrefix, prefixSums, rankCountElements);
```

这就是 AIV `c` 在 `addressTable[r]` 中的起始位置。一次 MTE2 可以搬入多行，Vector沿前序核维度
同时计算所有rank，避免原实现逐核 `DataCopy + MTE2_S` 和Scalar双层累加。不同核得到的区间首尾
相接且不重叠，所以写表时不需要atomic。

最后一个 AIV 的 prefix 已经包含所有前序核计数，因此最终总条目数为：

```text
rankCount[r] = rankPrefix[lastAiv][r] + perCoreRankCount[lastAiv][r]
```

最后一个 AIV 通过普通 strided MTE3 `DataCopyPad` 覆盖写入所有 rank 的总计数，仍不需要 atomic。

### 5.5 UB 分组和 MTE3 批量写表

逻辑布局为：

```text
addressTable[srcRank][entryIndex]
```

每条记录在 GM 中占一个 512B slot，但当前只使用前 16B：

| 字节偏移 | 类型 | 字段 | 用途 |
| ---: | --- | --- | --- |
| 0 | `uint32_t` | `tokenIndex` | 定位本卡 expert 输出 `x` |
| 4 | `int32_t` | `srcTokenIdx` | 计算目标 token slot |
| 8 | `int32_t` | `srcTopKIdx` | 计算目标 topK slot |
| 12 | `uint32_t` | `weight` | 可选 topK weight |
| 16～511 | 未使用 | padding | 隔离 cache line，并复用现有 workspace 布局 |

每个核第二次扫描自己的 metadata 区间：

- `srcRank == rankId`：像 PR 10309 的 `SetAddrInfoTable()` 本地分支一样，立即执行本地拷贝，不写地址表；
- 远端 rank：把记录按 `srcRank` 放入 40KB UB buffer，再批量写地址表。

每组 token 数不超过单个 rank 的 UB 分区容量，因此即使一组远端记录全部属于同一 rank 也不会越界。

每个 rank 的写入起点为 `rankPrefix[rank]`，写入地址为：

```cpp
entryAddr = GetLocalSendDataWorkspaceAddr(srcRank)
          + static_cast<uint64_t>(rankPrefix[srcRank]) * 512;
```

通过一次 strided MTE3 `DataCopy` 把同一 rank 的多条 32B UB 记录写到间隔 512B 的 GM 表项中，随后
推进该 rank 的局部 prefix。这里不调用 `WriteGmByPassDCache()`。

普通 `DataCopy` 的 `DataCopyParams` 使用 32B block 为单位，因此参数是 `blockLen=1`、
`dstStride=(512-32)/32=15`，与 PR 10309 的 `addrInfoCopyOutParams` 一致。

每个 rank 最多允许：

```text
numMaxTokensPerRank_ * topK_
```

条记录，代码中通过 `ascendc_assert` 检查越界。

### 5.6 建表完成同步

所有表项写完后：

```cpp
SyncFunc<HardEvent::MTE3_S>();
SyncAll<true>();
```

同步完成后，通信核才能读取计数器和地址表。

### 5.7 与 dispatch PR 10309 的对应关系

| PR 10309 dispatch | 当前 combine | 说明 |
| --- | --- | --- |
| PR 已有的 per-core count | 第一次 metadata 扫描 | 为新增地址表逻辑准备每核计数 |
| `scaleupCounter[core][rank]` | `perCoreRankCountWorkspace[core][rank]` | 保存每核计数行，普通 MTE3 写入 |
| `GetSlotStartNum()` | 分组搬运 + RA `ReduceSum` + Vector `Add` | 同时计算所有rank的exclusive prefix |
| `SetAddrInfoTable()` 本地分支 | 建表时直接 `SendSlot()` 本地拷贝 | 本地数据不进入地址表 |
| `SetAddrInfoTable()` 远端分支 | UB 中按 rank 组装 Combine 记录 | 两者记录字段不同 |
| `WriteAddrInfoTable()` | strided MTE3 `DataCopy` | 32B 有效块写入 512B stride 的 GM |

PR 10309 新增的地址表 diff 没有 atomic。其完整文件中的 `SetAtomicAdd` 属于 PR 之前已经存在的
`CalSendCnt()`，不算该 PR 的地址表实现。当前 Combine 地址表流程同样不使用任何 atomic；10309 中
出现的 `DataCacheCleanAndInvalid` 用于 notify/topkIdx 等其他数据的可见性处理，不属于地址表构建。

## 6. 阶段二：SendAddressTableRange

### 6.1 核和 channel 分工保持不变

地址表改造没有把多个远端核强行绑定到同一个单 channel。

仍以 8 rank、48 AIV、单 channel 为例：

| AIV | 修改后任务 |
| --- | --- |
| AIV 0～6 | 每个核读取一个远端 rank 的地址表，并使用该 rank 的 channel 0 |
| AIV 7～47 | 本地拷贝已在建表阶段完成；这里不再消费本 rank 地址表 |

所以不会产生多个远端 AIV 并发操作同一个 Hcomm channel 的问题。

### 6.2 远端 rank

远端核先读取目标 rank 的实际条目数：

```cpp
DataCopy(statusTensor, rankCountWorkspace[targetRank], 32B);
entryCount = statusTensor.GetValue(0);
```

计数读取走 MTE2，也不调用 `ReadGmByPassDCache()`。

单 channel 时 `groupSize == 1`，该核直接处理：

```text
[0, entryCount)
```

它不再扫描 `actualA_`，也不再为其他 7 个 rank 做无效的 `srcRank` 判断。

### 6.3 本地 rank

本地条目在 `BuildAddressTable()` 第二次分片扫描时由所有 AIV 直接处理，因此仍然是多核并行，且省掉：

```text
本地地址表 GM 写入
本地地址表 GM 读回
发送阶段为额外本地 AIV 设置的第二次 SyncAll
```

建表末尾的 `SyncAll<true>()` 已保证所有本地 MTE3 写入完成。之后每个远端 channel 可以独立发送
completion flag，不再被最慢的其他 channel 通过全核 barrier 拖住。

### 6.4 地址表批量搬入 UB

GM 表项间隔 512B，`SendAddressTableRange()` 使用跨步 `DataCopyPad`，把每条记录前 32B 紧凑搬到 UB：

```text
GM:
entry 0: [32B data][480B gap]
entry 1: [32B data][480B gap]
entry 2: [32B data][480B gap]

UB:
[entry 0 32B][entry 1 32B][entry 2 32B]
```

对应参数为：

```cpp
blockCount = curEntries;
blockLen = 32;
srcStride = 512 - 32;
dstStride = 0;
```

随后 Scalar 只遍历有效表项并调用原来的：

```cpp
SendSlot(tokenIndex, targetRank, srcTokenIdx, srcTopKIdx, channelIndex, weight);
```

## 7. SendSlot 与通信协议

### 7.1 远端路径

远端条目仍执行：

```text
x[tokenIndex]
    -> PrepareWrite(payload, ODR=5)

topkWeights[tokenIndex]（可选）
    -> PrepareWrite(weight/state, ODR=5)
```

batch 满 127 个 WQE 时提交，但不销毁当前 channel 的 `BatchHandle`；后续 payload、weight 和最终
ODR=6 flag 继续复用同一个 handle。只有切换 rank/channel 或发送阶段结束时才释放，和 PR 10309
`SendTokenByChannel()` 在多次 `BatchCommit()` 间复用 handle 的方式一致。

### 7.2 本地路径

本地条目在建表阶段直接执行：

```text
x GM -> UB -> local combine receive window
```

存在 topK weight 时，weight 仍写到本地 state window。

### 7.3 完成 flag

地址表消费和 payload 写入完成后，原有 completion flag 逻辑不变：

- payload 使用 ODR=5；
- channel completion flag 使用 ODR=6；
- 本地写入在建表末尾的全核同步前完成；
- 原有 channel 核发布约定数量的完成 flag；
- 远端发送结束后不再增加第二次全核同步；
- 最后调用 `FlushPreparedWrites()`。

地址表只是改变了“如何找到需要发送的 token”，没有改变接收窗口地址、channel handle 或完成协议。

## 8. Workspace 变化

### 8.1 地址表复用现有 per-rank send workspace

原有 workspace 已为每个 rank 预留：

```text
sendDataWorkspaceSizePerRank
    = numMaxTokensPerRank * topK * perSlotBytes
```

地址表每条使用 512B，而 `perSlotBytes` 按通信窗口要求至少按 512B 对齐，因此地址表可以安全复用该区域：

```text
rank 0 address table
rank 1 address table
...
rank N-1 address table
```

不需要新增大块地址表 workspace。

### 8.2 新增计数 workspace

新增两块空间：

```text
rankCountWorkspace:
    epWorldSize * 512B

perCoreRankCountWorkspace:
    aivNum * AlignUp(epWorldSize * 32B, 512B)
```

第一块保存每个 rank 的最终条目总数；第二块保存每个 AIV 的 per-rank 局部计数，供后续 AIV 计算
exclusive prefix。第二块中的每个 rank 占 32B，每个 AIV 的整行再按 512B 对齐。

完整 workspace 逻辑布局变为：

```text
| per-rank send/address-table workspace |
| per-AIV completion flag workspace     |
| per-rank address-table counters       |
| per-AIV/per-rank count rows           |
```

host tiling 中相应增加：

```cpp
epWorldSize * COMM_ALIGN
+ aivNum * AlignUp(epWorldSize * UB_ALIGN, COMM_ALIGN)
```

## 9. 修改前后扫描量对比

以 8 rank、48 AIV、单 channel，且各 rank 数据大致均衡为例。

### 修改前

```text
远端核关键路径：扫描 actualA_ + 发送约 actualA_/8
本地核组：      合计扫描 actualA_ + 本地拷贝约 actualA_/8

全卡 metadata 扫描量约 8 * actualA_
```

### 修改后

```text
建表阶段：计数和组表各扫描一次，全卡合计 2 * actualA_
消费阶段：只遍历远端 rank 的 entry，均衡 8 卡时全卡合计约 7/8 * actualA_

全卡 Scalar 条目处理量约 2.875 * actualA_
```

单个远端通信核的 metadata 相关关键路径从：

```text
actualA_
```

变为大致：

```text
2 * actualA_/48 + entryCount[targetRank]
```

如果 rank 分布均匀，则约为：

```text
2 * actualA_/48 + actualA_/8
```

## 10. 修改前后对照表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 原始 metadata 扫描 | 每个远端核完整扫描 | 所有 AIV 分片，合计扫描一次 |
| rank 过滤 | 每个通信核重复判断 | 建表阶段统一完成 |
| 远端核输入 | 原始 metadata | 目标 rank 的紧凑地址表 |
| 本地核输入 | 原始 metadata 分片 | 建表第二遍直接本地拷贝，不落地址表 |
| 单 channel 核分工 | 每个远端 rank 一个核 | 不变 |
| `groupSize` | 远端等于 channel 数，本地等于剩余核数 | 不变 |
| `SendSlot()` | 直接由 metadata 扫描调用 | 本地由建表调用，远端由地址表消费调用 |
| Hcomm channel | 原有 handle | 不变 |
| batch WriteNbi | 127 WQE 一批 | 不变 |
| completion flag | 原有 ODR=6 协议 | 不变 |
| 大块 workspace | 已存在 per-rank send workspace | 复用为地址表 |
| 新增 workspace | 无 | rank 总计数 + per-AIV/per-rank 计数行 |

## 11. 新增开销与注意事项

地址表消除了重复扫描，但不是零成本，新增了：

- metadata 两次分片扫描（第一次计数，第二次组表）；
- 每个 AIV 一次 per-rank 计数行普通 MTE3 写入；
- 分组读取前序 AIV 计数行并计算 exclusive prefix；
- 最后一个 AIV 写入各 rank 最终条目数；
- 地址表 GM 写入和再次读取；
- 计数发布和建表结束处的全核同步；
- 地址表每条按 512B stride 存放，空间利用率较低。

因此性能收益需要以实测为准，建议分别记录：

1. `BuildAddressTable()` 时间；
2. 前 7 个远端核的 `SendAddressTableRange()` 时间；
3. 每个 rank 的 `entryCount`；
4. `PrepareWrite` 和 `BatchCommit` 时间；
5. 修改前后的端到端 kernel 时间。

当前实现已经采用 PR 10309 的per-core count、分组搬运、RA `ReduceSum`、Vector `Add`、本地直接
拷贝和远端批量MTE3写表结构。

## 12. 正确性检查建议

建议至少检查以下条件：

```text
sum(rankCount[0..epWorldSize-1]) == 有效远端 metadata 条目数
rankCount[rankId] == 0
rankCount[rank] <= numMaxTokensPerRank * topK
```

并逐条校验：

```text
addressTable[srcRank][entry].tokenIndex
addressTable[srcRank][entry].srcTokenIdx
addressTable[srcRank][entry].srcTopKIdx
addressTable[srcRank][entry].weight
```

最终应验证：

- 每个目标 `(srcRank, srcTokenIdx, srcTopKIdx)` slot 都只写到正确位置；
- 本地和远端 payload 内容一致；
- `HasTopkWeight=0/1` 都正确；
- 空 rank 仍能发布 completion flag；
- 尾 batch 被 `FlushPreparedWrites()` 提交；
- 接收端看到 completion flag 时，对应 payload 已经可见。

## 13. 关于性能实验中的编译器优化

如果为了测纯扫描耗时，把 `SendSlot()` 的远端和本地逻辑全部注释，编译器可能内联
`SendSlot()` 后发现循环没有可观察副作用，进而删除 metadata 读取、条件判断甚至整个循环。

因此“空 SendSlot 约 10 us”不能代表真实前处理时间。测纯扫描时应保留可观察结果，例如统计
`matchedCount` 并写入 GM workspace，防止死代码删除。

地址表正式实现天然具有以下副作用，不会被整体删除：

```text
MTE3 DataCopy 发布 per-core count 和最终 rank count
MTE3 batch DataCopy addressTable
PrepareWrite / SendLocalToken
completion flag
```

## 14. 8 卡、topK=8 均衡示例

### 14.1 场景假设

假设：

```text
epWorldSize     = 8
topK            = 8
channelsPerRank = 1
aivNum          = 48
当前计算卡      = rank 3
每张源卡 token 数 = 2
topK weight     = 1/8 = 0.125
```

为了构造最均衡的例子，每个源 token 的 8 个 topK 项恰好分别落到 rank 0～7：

```text
topKIdx 0 -> rank 0 上的 expert
topKIdx 1 -> rank 1 上的 expert
topKIdx 2 -> rank 2 上的 expert
topKIdx 3 -> rank 3 上的 expert
topKIdx 4 -> rank 4 上的 expert
topKIdx 5 -> rank 5 上的 expert
topKIdx 6 -> rank 6 上的 expert
topKIdx 7 -> rank 7 上的 expert
```

因此每张源卡的每个 token 都会在 rank 3 上产生一份 expert 输出，而且该输出对应
`srcTopKIdx = 3`。

rank 3 一共接收：

```text
8 张源卡 * 每卡 2 个 token = 16 个 expert 输出
```

所以在本例中：

```text
actualA_ = 16
```

### 14.2 rank 3 上的原始 metadata

为方便展示，假设 rank 3 上的 expert 输出按照“先放各卡 token 0，再放各卡 token 1”的顺序排列：

| 本地 `tokenIndex` | `srcRank` | `srcTokenIdx` | `srcTopKIdx` | weight |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 3 | 0.125 |
| 1 | 1 | 0 | 3 | 0.125 |
| 2 | 2 | 0 | 3 | 0.125 |
| 3 | 3 | 0 | 3 | 0.125 |
| 4 | 4 | 0 | 3 | 0.125 |
| 5 | 5 | 0 | 3 | 0.125 |
| 6 | 6 | 0 | 3 | 0.125 |
| 7 | 7 | 0 | 3 | 0.125 |
| 8 | 0 | 1 | 3 | 0.125 |
| 9 | 1 | 1 | 3 | 0.125 |
| 10 | 2 | 1 | 3 | 0.125 |
| 11 | 3 | 1 | 3 | 0.125 |
| 12 | 4 | 1 | 3 | 0.125 |
| 13 | 5 | 1 | 3 | 0.125 |
| 14 | 6 | 1 | 3 | 0.125 |
| 15 | 7 | 1 | 3 | 0.125 |

逻辑上的原始 metadata 是交错的：rank 0 的两条记录分别位于 `tokenIndex=0` 和
`tokenIndex=8`，rank 1 的两条记录分别位于 1 和 9，以此类推。

修改前，每个远端核都要扫描这 16 条记录，才能找到其中属于自己目标 rank 的 2 条。

### 14.3 per-core 计数和 prefix

建表开始时：

```text
rankCount = [0, 0, 0, 0, 0, 0, 0, 0]
```

本例 `actualA=16`、`aivNum=48`，所以 AIV 0～15 各处理一条，AIV 16～47 的局部计数全为 0。
例如：

```text
AIV 0: perCoreRankCount[0] = [1,0,0,0,0,0,0,0]
AIV 1: perCoreRankCount[1] = [0,1,0,0,0,0,0,0]
...
AIV 8: perCoreRankCount[8] = [1,0,0,0,0,0,0,0]
```

当前卡是 rank 3，所以 AIV 3 和 AIV 11 对应的本地条目不会增加 `perCoreRankCount[][3]`。

最后一个 AIV 根据自己的 prefix 加自身计数得到：

```text
rankCount = [2, 2, 2, 0, 2, 2, 2, 2]
```

对 rank 0，AIV 0 和 AIV 8 的 exclusive prefix 分别为：

```text
AIV 0: rankPrefix[0] = 0
AIV 8: rankPrefix[0] = sum(perCoreRankCount[0..7][0]) = 1
```

所以第二次扫描和批量写表时：

```text
AIV 0 -> addressTable[0][0] = {0, 0, 3, 0.125}
AIV 8 -> addressTable[0][1] = {8, 1, 3, 0.125}
```

其他远端 rank 同理。本地 rank 3 的两条记录在第二次扫描时直接拷贝，不生成表项。

### 14.4 最终逻辑地址表

由于 prefix 按 AIV 编号计算，最终地址表为：

| `srcRank` | `entryIndex` | `tokenIndex` | `srcTokenIdx` | `srcTopKIdx` | weight |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 3 | 0.125 |
| 0 | 1 | 8 | 1 | 3 | 0.125 |
| 1 | 0 | 1 | 0 | 3 | 0.125 |
| 1 | 1 | 9 | 1 | 3 | 0.125 |
| 2 | 0 | 2 | 0 | 3 | 0.125 |
| 2 | 1 | 10 | 1 | 3 | 0.125 |
| 4 | 0 | 4 | 0 | 3 | 0.125 |
| 4 | 1 | 12 | 1 | 3 | 0.125 |
| 5 | 0 | 5 | 0 | 3 | 0.125 |
| 5 | 1 | 13 | 1 | 3 | 0.125 |
| 6 | 0 | 6 | 0 | 3 | 0.125 |
| 6 | 1 | 14 | 1 | 3 | 0.125 |
| 7 | 0 | 7 | 0 | 3 | 0.125 |
| 7 | 1 | 15 | 1 | 3 | 0.125 |

也可以按 rank 展开为：

```text
addressTable[0] = [
    {tokenIndex=0,  srcTokenIdx=0, srcTopKIdx=3, weight=0.125},
    {tokenIndex=8,  srcTokenIdx=1, srcTopKIdx=3, weight=0.125},
]

addressTable[1] = [
    {tokenIndex=1,  srcTokenIdx=0, srcTopKIdx=3, weight=0.125},
    {tokenIndex=9,  srcTokenIdx=1, srcTopKIdx=3, weight=0.125},
]

addressTable[3] = []  // 当前卡，本地条目直接拷贝

...

addressTable[7] = [
    {tokenIndex=7,  srcTokenIdx=0, srcTopKIdx=3, weight=0.125},
    {tokenIndex=15, srcTokenIdx=1, srcTopKIdx=3, weight=0.125},
]
```

真实运行时多个 AIV 可以并发批量写表，但各核区间已经由 prefix 隔离，不会覆盖；同一 rank 内按
AIV 编号及各 AIV 本地扫描顺序排列。

### 14.5 地址表在 GM 中的物理样子

每个 rank 有独立的地址表 base：

```text
rankTableBase(rank)
    = combineSendDataWorkspaceAddr
    + rank * sendDataWorkspaceSizePerRank
```

以 rank 0 表为例：

```text
rankTableBase(0) + 0 * 512:
    +0   uint32 tokenIndex   = 0
    +4   int32  srcTokenIdx  = 0
    +8   int32  srcTopKIdx   = 3
    +12  uint32 weightBits   = 0x3e000000  // float 0.125
    +16 ... +511             = 未使用

rankTableBase(0) + 1 * 512:
    +0   uint32 tokenIndex   = 8
    +4   int32  srcTokenIdx  = 1
    +8   int32  srcTopKIdx   = 3
    +12  uint32 weightBits   = 0x3e000000
    +16 ... +511             = 未使用
```

8 张 rank 表整体可表示为，其中本地 rank 3 为空：

```text
combineSendDataWorkspaceAddr
|
+-- rank 0 table base
|      +-- entry 0: offset 0
|      +-- entry 1: offset 512
|
+-- rank 1 table base = base + 1 * sendDataWorkspaceSizePerRank
|      +-- entry 0: offset 0
|      +-- entry 1: offset 512
|
...
|
+-- rank 7 table base = base + 7 * sendDataWorkspaceSizePerRank
       +-- entry 0: offset 0
       +-- entry 1: offset 512
```

### 14.6 消费地址表时的核分工

当前卡是 rank 3，因此前 7 个远端通信核负责：

| AIV | `targetRank` | 读取的表 | channel |
| ---: | ---: | --- | ---: |
| 0 | 0 | `addressTable[0][0..2)` | 0 |
| 1 | 1 | `addressTable[1][0..2)` | 0 |
| 2 | 2 | `addressTable[2][0..2)` | 0 |
| 3 | 4 | `addressTable[4][0..2)` | 0 |
| 4 | 5 | `addressTable[5][0..2)` | 0 |
| 5 | 6 | `addressTable[6][0..2)` | 0 |
| 6 | 7 | `addressTable[7][0..2)` | 0 |

AIV 7～47 在发送阶段不再读取本地表：

```text
addressTable[3] = empty
```

本地两条记录已经分别由处理 `tokenIndex=3` 和 `tokenIndex=11` 的建表 AIV 直接拷贝完成。

### 14.7 一个表项最终写到哪里

以 `addressTable[0][1]` 为例：

```text
tokenIndex  = 8
srcRank     = 0
srcTokenIdx = 1
srcTopKIdx  = 3
topK        = 8
```

payload 源地址来自当前 rank 3：

```cpp
srcPayload = xGm_[8 * axisH_];
```

目标 rank 0 上的逻辑 slot 为：

```text
slot = srcTokenIdx * topK + srcTopKIdx
     = 1 * 8 + 3
     = 11
```

因此 payload 目标地址为：

```cpp
dstPayload = rank0CombineDataWindow + 11 * perSlotBytes_;
```

weight/state 目标地址为：

```cpp
dstState = rank0CombineStateWindow + 11 * 512;
```

再看建表阶段遇到的本地记录：

```text
tokenIndex  = 3
srcRank     = 3，也就是当前卡
srcTokenIdx = 0
srcTopKIdx  = 3
slot        = 0 * 8 + 3 = 3
```

它不进入地址表，也不走远端 Hcomm，而是立即执行：

```text
x[3] -> UB -> rank 3 本地 combine data window 的 slot 3
```

### 14.8 修改前后在该例中的工作量

修改前：

```text
7 个远端核：每核扫描 16 条，共扫描 112 条
本地核组：  合计扫描 16 条
总计：      128 次 metadata 条目检查
```

修改后：

```text
建表阶段：计数和组表各处理 16 条，共 32 条
消费阶段：7 张远端表各 2 条，共处理 14 条
总计：    46 次条目处理
```

该小例子中条目处理量从 128 降到 46。实际 `actualA_` 更大时，远端核不再重复扫描其他 rank
metadata 的收益会更加明显，但还需要扣除计数汇总、prefix、地址表 GM 写入/读取和全核同步的新增成本。
