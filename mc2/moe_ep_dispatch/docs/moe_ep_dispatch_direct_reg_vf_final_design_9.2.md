# moe_ep_dispatch Direct 原生 VF 性能优化最终设计（CANN 9.2.0）

## 1. 文档信息与结论

| 项目 | 内容 |
|---|---|
| 目标算子 | `moe_ep_dispatch` |
| 优化范围 | 非 Cache 的 Direct 模板 |
| 暂不修改 | Hybrid、Cache、通信协议、GM 输出格式、Host Tiling 结构 |
| 目标平台 | Ascend 950，CANN 9.2.0 |
| 输入约束 | `1 <= topK <= 32`，`2 <= epWorldSize <= 1024`，`2 <= numExperts <= 2048`，且 `numExperts % epWorldSize == 0` |
| 核心方案 | 原生 VF 指令实现 `CalSendCnt`、rank token 压缩和 slot 前缀累加 |
| 参考实现 | `dispatch_combine_train/dispatch_src/src/include/dispatch.hpp` 中的 `deduplicate` 与 squeeze 流程 |
| 当前状态 | 源码已实现；算法模型和静态边界已检查；尚缺 CANN 9.2.0 编译及 950 真机验证 |

最终选择原生 VF，不按 K 生成多个模板，也不引入 SIMT 分支。K 是运行时参数，按 `128 / K` 或 `64 / K` 计算每个寄存器批次可容纳的完整 token 行数。Direct 的 K 上限只有 32，单 token 去重最多比较 32 个位置，VF 的规则行布局、寄存器重排和直方图能力与本算子高度匹配；SIMT 会增加线程映射、共享统计及同步成本，没有足够依据证明在当前小 K 规格下更优。

本方案不是用一个 VF 调用跨越整个算子的 GM 搬运和通信阶段，而是：

- 每个 `CalSendCnt` UB 分组调用一次 `DeduplicateVf`，覆盖该组的转换、校验、expert 统计、rank 计算、去重、rank 统计和持久计数累加。
- `SetSrcTokenIdx` 每个目标 rank 调用一次 `SqueezeTokenIdxVf`，保留现有 rank 顺序、GM 偏移和 MTE3 调度。
- `GetSlotStartNum` 每个装入 UB 的 counter row 分组调用一次 `PrefixSlotVf`。

## 2. 优化目标与基线瓶颈

优化前 Direct 路径将同一批路由数据拆成多个 AscendC 高层 Vec API 阶段，包括 `Cast`、`Div`、`Compare`、`Select`、`Gather`、`GatherMask`、`GetExpertFreq` 和 `Add`。阶段之间需要把中间值和位掩码反复写回 UB，再从 UB 读入，主要成本为：

1. 高频 Vec API 调用与相邻阶段启动开销。
2. expert id、rank id、gather index 和多个 mask 的 UB 往返。
3. expert/rank 统计分别扫描数据并做多次类型转换。
4. `GatherMask` 压缩后通过标量读取返回计数，流水衔接较碎。
5. mask UB 随分组数据量扩大，但本质上这些谓词可留在 VF predicate 寄存器中。

优化目标保持数学语义不变：

```text
expert = valid(topkId) ? topkId : -1
rawRank = expert >= 0 ? expert / expertsPerRank : -1
dedupRank[token, k] =
    rawRank[token, k], 当且仅当它在同 token 更早的 topK 位置未出现
    -1,                其他情况

sendCntPerExpert[e] += count(expert == e)
sendCntPerRank[r]   += count(dedupRank == r)
```

其中 expert 计数保留同一 token 内的重复 expert/rank 路由；rank 计数只统计去重后的目标 rank。这一点与原实现和参考 `dispatch.hpp` 一致。

## 3. 最终代码组织

### 3.1 文件边界

- `op_kernel/moe_ep_dispatch_vf.h`
  - 仅放 Direct 使用的原生 VF 实现。
  - 定义 `DeduplicateVf`、`SqueezeTokenIdxVf`、`PrefixSlotVf`。
- `op_kernel/moe_ep_dispatch.h`
  - 保留 GM/UB 搬运、事件同步、通信和 Direct 主流程。
  - 包含 VF 头并传入 UB 物理地址。
  - 已删除不再使用的旧 Vec-API 去重/直方图私有函数，避免与缩小后的 `maskBuf_` 形成潜在越界入口。
- Hybrid 相关文件和分支不修改。

不把实现放入通用 `base.h`。这些 VF 函数依赖 Direct 特有的数据布局、计数语义和 scratch 生命周期，放入独立头文件更能约束可见性，也避免让 Hybrid 或其他算子误用。

### 3.2 三个 VF 入口

```cpp
DeduplicateVf(
    topkS32, expertScratch, dstRankS16,
    expertHistU32, rankHistU32,
    expertCountS32, rankCountS32,
    calCnt, tokenCnt, topK,
    expertNum, rankNum, expertsPerRank);

SqueezeTokenIdxVf(
    tokenIdxOut, countOut, rankEntity,
    dstRankId, tokenOffset, tokenNum, topK);

PrefixSlotVf(
    counterRows, slotStart,
    rowCount, rowStride, validColumns);
```

`DeduplicateVf` 是一个 VF 调用，内部按数据依赖组织为五个逻辑区段：

1. `PrepareExpert`
2. `SetExpertHistogram`
3. `DedupAndSendDirect`
4. `SetRankHistogram`
5. 结果落盘与两个直方图的持久计数累加

区段不是五次 `asc_vf_call`，也不是五个高层 Vec API；它们位于同一 `__VEC_SCOPE__` 中。必要的 UB 可见性由 `mem_bar(VST_VLD)` 显式保证。

## 4. K 的运行时切分设计

不区分 K 模板。Tiling 已拒绝 K=0，实际规格是 `1..32`。

### 4.1 去重阶段

B16 向量有 128 个 lane。为了不让一个 token 行跨寄存器批次：

```text
tokenPerReg = floor(128 / K)
mainSize    = tokenPerReg * K       // 1..128
loopCount   = ceil(tokenCnt / tokenPerReg)
```

例如：

| K | tokenPerReg | mainSize |
|---:|---:|---:|
| 1 | 128 | 128 |
| 3 | 42 | 126 |
| 8 | 16 | 128 |
| 16 | 8 | 128 |
| 32 | 4 | 128 |

最后一轮按 `mainSize` 做流式读写，可能覆盖逻辑尾部之外的 padding，但后续直方图和 GM 写回都只消费 `calCnt` 个有效元素。UB 为这一行为预留了 256B guard，不能删除。

### 4.2 token index 压缩阶段

rank 从 B16 解包到 B32 后参与比较，B32 向量有 64 个 lane：

```text
tokenPerReg = floor(64 / K)
mainSize    = tokenPerReg * K       // 1..64
fullLoops   = floor(tokenCnt / tokenPerReg)
tailSize    = (tokenCnt - fullLoops * tokenPerReg) * K
```

主循环使用 `mainMask`，尾轮使用 `tailMask`，因此不会把 padding lane 压入输出。

## 5. `DeduplicateVf` 详细设计

### 5.1 PrepareExpert：校验、int32→int16、rank 计算

输入 `topk_idx` 通过 MTE2 直接落到 UB 的 int32 缓冲。VF 内每轮读取两组 64-lane B32：

```text
vldus two vector_s32
vcmps_ge(id, 0)
vcmps_le(id, expertNum - 1)
vsel(invalid -> -1)
vdintlvv2(..., LOWER)        // 两个 B32 向量窄化/交织为一个 B16 向量
vdiv(expert, expertsPerRank)
vsel(invalid rank -> -1)
vstus expertScratch
vstus dstRankS16
mem_bar(VST_VLD)
```

int32→int16 必须在 VF 内完成。MTE2 只负责按输入类型搬运，不执行类型转换；若先调用高层 `Cast` 落另一块 UB，再进入 VF，会恢复一次额外的 API 启动和 UB 往返。实现沿用参考 `dispatch.hpp::deduplicate` 的 `vdintlvv2(..., LOWER)` 布局，但在窄化前增加上、下界校验，保证大于 `numExperts - 1` 或负数统一成为 `-1`，避免截断后伪装成合法 expert。

这里仍把 `expertScratch` 和原始 `dstRankS16` 写入 UB，而不是强行跨后续所有区段常驻寄存器，原因是：

- expert 直方图可能按最多 8 个 256-bin 分组反复扫描 expert 数据；
- rank 去重按 token 行布局读取 rank；
- 同时长期保留两份全组数据不可能放进有限向量寄存器；
- 明确的 UB 阶段边界比隐式寄存器 spill 更可控。

### 5.2 SetExpertHistogram

沿用参考实现的 256-bin 组织：

1. 每次加载 256 个 B16 expert（低/高各 128 lane）。
2. 减去当前 `binStart`。
3. 用 `vcmps_ge`、`vcmps_le` 形成 `[0, 255]` 谓词。
4. `pdintlvv2_b8` 合并谓词，`vdintlvv2` 将 B16 expert 收窄到 U8 bin 编号。
5. `dhistv2(..., Bin_N0/Bin_N1)` 更新两个 U16 计数向量。
6. `vst(..., INTLV_B16)` 与零交织，直接生成 U32 直方图布局。

每个 Direct 分组最大 `calCnt = PER_GROUP_SIZE / sizeof(int16_t) = 16384`，所以局部 U16 bin 计数最大 16384，不会溢出。持久累计使用 int32。

### 5.3 DedupAndSendDirect

该区段负责同 token 内目标 rank 去重，并保留第一次出现的位置：

```text
vci(linear)
tokenIndex = linear / K
gatherBase = tokenIndex * K
laneIndex  = linear - gatherBase

for k in [0, K):
    gatherIndex = gatherBase + k
    previousRank = vselr(rankData, gatherIndex)
    earlier = laneIndex > k
    duplicate |= earlier && (rankData == previousRank)

dedupData = duplicate ? -1 : rankData
```

使用的核心原生 VF 接口为 `vci`、`vdup`、`vdiv`、`vmuls`、`vsub`、`vadds`、`vselr`、`vcmps_gt`、`vcmp_eq`、`por`、`vsel`、`vldus` 和 `vstus`。无效 rank 已在 `PrepareExpert` 中变为 `-1`；即使多个无效位置互相命中，输出仍为 `-1`，不会进入 rank 直方图。

去重结果暂存到已不再需要的 `topkS32` UB。写完后执行 `mem_bar(VST_VLD)`，再由 `SetRankHistogram` 读取。

### 5.4 SetRankHistogram

算法与 expert 直方图相同，输入换成去重后的 rank，bin 数为 `ceil(rankNum / 256)`。rank 直方图使用 `dstExpBuf_` 中独立于 `expertScratch` 的区域，避免同一地址先作为 VLD 源、随后立即作为 VST 目的造成复用风险。

### 5.5 结果物化与持久累计

- 去重 rank 从 `topkS32` scratch 流式写回 `dstRankS16`，保持原 GM 输出地址和 int16 格式不变。
- expert/rank 局部 U32 直方图分别与 `sendCntPerExpertTensor_`、`sendCntPerRankTensor_` 做 B32 `vadd`。
- 尾块用 `plt_b32` 形成有效谓词；流式 store 仍按 64 lane 写入，目标计数 buffer 已按 256B 对齐。
- VF 末尾执行 `mem_bar(VST_VLD)`，外层再按现有事件链做 V/MTE3 同步和 GM 写回。

## 6. `SqueezeTokenIdxVf` 详细设计

`SetSrcTokenIdx` 的外层 rank 遍历顺序保持不变：先远端 rank，最后本端 rank。每个 rank 的 VF 内执行：

```text
vci -> vdiv(K) -> vadds(tokenOffset)    // 构造每个路由位置对应的 token id
vldus rank B16
vunpack B16 -> B32
vcmps_eq(rank, dstRankId)
vsqz(tokenIndex, predicate, MODE_STORED)
vstur compressed indices
vstar flush
sprsts(SPR_AR, countOut)
```

`vsqz` 是按谓词压缩有效 lane 的 VF 指令；`vstur` 将压缩结果按 `SPR_AR` 管理的连续位置写出；`sprsts` 把最终压缩数量写入 UB。这样替代原来的 `CompareScalar + GatherMask + 标量 rsvdCnt` 数据路径，但外层仍从 32B `maskBuf_` 的首个 U32 读取数量，以保持控制流接口简单。

正确性约束：

- `tokenIndex` 由路由位置除 K 得到，同一个 token 的 K 项映射到同一 token id。
- 去重后的 `rankEntity` 中每个 token 对同一 rank 最多一个有效位置，因此压缩结果不重复。
- 主循环和尾轮均有明确谓词，输出顺序与原线性扫描顺序一致。

## 7. `PrefixSlotVf` 详细设计

`GetSlotStartNum` 需要按列累加前序 AIV 的 `sendCntPerRank` 行：

```text
for each 64-column block:
    sum = 0
    for each copied counter row:
        sum += row[block]
    slotStart[block] += sum
```

VF 使用 `vldus`、`vadd`、`vstus`，`validColumns = epWorldSize`，`rowStride = counterCnt_`。`GetSlotStartNum` 按 `topkIdsBuf_` 容量分批装入 counter rows，因此调用次数是该 AIV 前序行数的 UB 分组数，而不是固定一次。最后一个 64-column block 使用 `plt_b32`；`slotIdxPerRankTensor_` 位于 256B 对齐的 `sendCntExpertBuf_` 中，满足尾块流式写入边界。

## 8. UB 布局与边界可行性

### 8.1 `CalSendCnt` 组内布局

设：

```text
tmpOffset       = AlignUp(calCnt * sizeof(int16), 256)
expertHistSize  = ceil(numExperts / 256) * 256 counters
rankHistSize    = ceil(epWorldSize / 256) * 256 counters
histogramOffset = tmpOffset + 256
rankHistOffset  = histogramOffset + expertHistSize * sizeof(uint32)
```

`dstExpBuf_` 大小固定为 `2 * perGroupSizeAlign_ = 65536B`。最坏规格：

```text
tmpOffset                         = 32768B
guard                             =   256B
expert histogram (2048 * 4)       =  8192B
rank histogram (1024 * 4)         =  4096B
end                               = 45312B < 65536B
```

因此 dstRank、guard、expert histogram 和 rank histogram 可同时容纳。

### 8.2 其他 buffer 变化

| Buffer | 最终设计 | 说明 |
|---|---:|---|
| `topkIdsBuf_` | `2 * perGroupSizeAlign_ + 256B` | 为去重/物化/Prefix 尾部流式访问提供 guard |
| `tempBuf_` | 不变 | expert B16 scratch |
| `dstExpBuf_` | 不变 | dstRank + guard + 两个局部 histogram |
| `maskBuf_` | 32B | 只物化 squeeze count；谓词保留在寄存器 |
| expert count | 按 256B 对齐 | 支持 64-lane B32 尾块 store |
| rank count | 按 256B 对齐 | 已有对齐布局 |

与优化前相比，新增的 topk guard 为 256B，expert count 最多增加 224B 对齐空间；而旧 mask 在最大分组下需要三份约 2048B mask，缩为 32B 后至少节省约 6KB。VF 改造不会增加 Direct 总体 UB 压力，净 UB 使用反而下降。完整 UB 是否通过最终编译器静态分配仍必须由 CANN 9.2.0 构建确认。

### 8.3 关键不变量

- `topK >= 1`，所以所有除法和 `tokenPerReg` 均有效。
- `topK <= 32`，因此 `128 / K >= 4`、`64 / K >= 2`。
- `numExperts % epWorldSize == 0`，所以 `expertsPerRank >= 1`。
- `numExperts >= epWorldSize`，故将 `slotIdxPerRankTensor_` 放在 expert count buffer 内不会小于 rank 列数。
- `calCnt <= 16384`，U16 局部直方图不会溢出。

## 9. 同步与内存可见性设计

原生 VF 不自动替代跨流水同步。最终依赖如下：

| 生产者 | 消费者 | 保证方式 |
|---|---|---|
| MTE2 写 `topkS32` | VF `PrepareExpert` 读 | 外层 `MTE2_V` |
| VF 写 expert/rank scratch | 后续 histogram/dedup 读 | VF 内 `mem_bar(VST_VLD)` |
| VF 写 dedup scratch | `SetRankHistogram` 读 | VF 内 `mem_bar(VST_VLD)` |
| VF 写两个 histogram | 持久 count 累加读取 | VF 内 `mem_bar(VST_VLD)` |
| VF 写 `dstRankS16` | MTE3 写 GM | 外层 `V_MTE3` |
| MTE3 读上一组 `dstExpBuf_` | 下一组 VF 覆写 | 外层 `MTE3_V` |
| MTE2 写 Prefix counter rows | `PrefixSlotVf` 读 | 外层 `MTE2_V` |
| `PrefixSlotVf` 写 slot | 下一批 MTE2 复用 UB | VF 内 barrier + 外层 `V_MTE2` |
| squeeze 写 token index/count | 标量和 MTE3 消费 | `vstar`、`sprsts` + 外层 `V_S`/`V_MTE3` |

不允许为了减少指令而删除这些 barrier/event；必须以 9.2.0 编译结果和流水文档逐项复核。

## 10. 与参考 `dispatch.hpp` 的一致与差异

### 10.1 直接复用的核心模式

- 两个 S32 向量通过 `vdintlvv2(..., LOWER)` 生成一个 S16 向量。
- 256-bin histogram：范围归一化、predicate 组合、U8 bin、`dhistv2`、U16/zero 交织为 U32。
- 规则 token 行内用 `vci` 构造 lane，`vselr` 做寄存器 gather，再用 predicate 去重。
- token index 压缩使用 `SPR_AR + vsqz + vstur + sprsts`。

### 10.2 本算子必须保留的差异

| 项目 | 参考代码 | 当前 Direct 实现 |
|---|---|---|
| K/Expert/Rank | 参考中部分为模板常量 | 当前由 Tiling 运行时传入 |
| 输入合法性 | 参考流程偏向既定输入 | 当前在窄化前同时检查 `<0` 和 `>=numExperts` |
| expert 统计 | 局部结果服务参考 dispatch 流程 | 需累加到 Direct 的 persistent per-expert count 并最终 atomic 写 GM |
| rank 统计 | 参考上下文独立 | 当前必须统计去重后 rank，并写 per-AIV workspace/per-rank GM |
| dstRank | 参考有自己的阶段布局 | 当前必须保持既有 GM offset、int16 格式及通信消费关系 |
| 分组 | `UBId5/UBId6/UBId7` 按 32KB 固定分区；`dedup_rank_entity` 先按 `CORENUM_FOR_DEDUP` 生成每核 `topk_tile`，再在对应 UB 分区内调用一次 `deduplicate` | 保持原 Direct 的 `PER_GROUP_SIZE = 32KB` 分组；先按 AIV 分核，单核数据按 32KB B16 工作组循环，每组调用一次 `DeduplicateVf` |
| slot | 参考文件未直接替代本算子 slot 流程 | 当前单独使用 `PrefixSlotVf` 按前序 AIV 行做列累加 |

因此不能直接拷贝整个参考函数；可复用的是经过验证的原生 VF 指令组织，外围 UB 布局、累计方式、同步和 GM 写回必须适配当前 Direct 实现。

当前的“UB 分组”不是为了 VF 新增的算法步骤，而是优化前 Direct `CalSendCnt` 已有的容量切分，最终实现明确保持 `PER_GROUP_SIZE = 32 * 1024` 不变。这里的 32KB 是每组 B16 expert/rank 工作集容量，因此 `perGroupTokenNum = 32KB / sizeof(int16_t) / K`；对应的 int32 `topk_idx` 输入占用最多 64KB。参考实现也围绕 32KB UB 分区组织 tile，但通过固定核数预先限制每核 `topk_tile`；当前算子没有同样的每核上限保证，所以继续显式循环多个 32KB 组。当单核数据不超过一组时，`groupCnt=1`。

## 11. 性能预期与取舍

预期收益：

- `CalSendCnt` 每组由多次高层 Vec API 调用收敛为一次 VF 调用。
- 去重谓词不再反复写 mask UB。
- S32→S16、rank 推导和去重使用流式寄存器路径。
- histogram 直接生成 U32 布局，省去原先的多级 `GatherMask/Cast/GetExpertFreq/Cast` 调度。
- squeeze 通过 `vsqz` 直接压缩，不再单独生成 UB compare mask。
- `maskBuf_` 从数据量相关缩到 32B。

有意保留的 UB 阶段边界：

- expert/rank 原始向量仍落 UB，供跨循环扫描和不同布局阶段读取。
- expert/rank histogram 仍落 UB，再与持久计数相加。
- 这两类落盘是数据规模和寄存器容量决定的，不应以“全程不落 UB”为目标强行消除，否则更可能引入寄存器 spill 或重复计算。

主要性能风险：

1. K 是运行时参数。VF 支持 `for` 和条件控制，当前循环变量使用 `uint16_t`、从 0 开始、步长为 1、边界在循环内不修改，满足硬件循环优化的基本编码形态；但 `topK` 不是编译期常量，编译器是否将内层去重循环生成为硬件循环必须检查反汇编。若退化为 software loop，K=32 时的 32 次 `vselr + compare` 可能成为热点。
2. expertNum=2048 时 expert histogram 需要 8 个 bin 分组并多次扫描该组数据，这是 256-bin `dhistv2` 的固有限制。
3. 单个 `DeduplicateVf` 较大，可能产生寄存器 spill；必须检查 9.2.0 编译日志/反汇编与真机 profiling。若确认 spill 显著，首选在 `SetExpertHistogram` 与 `DedupAndSendDirect` 的 UB barrier 处拆成两个 VF 调用，而不是退回整套 Vec API。

### 11.1 K 模板化分析

参考 `dispatch.hpp` 将 `kNumTopk` 作为模板参数。与当前运行时 K 相比，模板 K 的潜在收益主要集中在 `DedupAndSendDirect` 和 `SqueezeTokenIdxVf`：

- `tokenPerReg = 128 / K`、`mainSize = tokenPerReg * K` 可在编译期计算。
- squeeze 中的 `tokenPerReg = 64 / K`、主循环长度和尾块布局可进一步简化。
- 去重内层 `for (k = 0; k < K; ++k)` 的边界成为常量，编译器可以选择硬件循环或适度展开。
- `vdiv/vmuls/vadds` 使用的 K 成为常量，可减少部分运行时标量准备和控制指令。

但直接实例化 K=1..32 的 32 份完整 `DeduplicateVf<K>` 并不一定更快。`DeduplicateVf` 还包含两个较大的 histogram 和持久计数区段，全量复制可能造成：

- kernel 二进制显著膨胀和指令缓存压力；
- 编译时间、跳转距离和维护成本增加；
- 大 K 完全展开后指令数和寄存器活跃区间增加，可能加重 register spill；
- 大量低频 K 特化没有实际收益。

如果 9.2.0 反汇编或 950 profiling 证明动态 K 的去重循环退化为 software loop并成为热点，推荐采用“热点 K 模板特化 + 动态 K 回退”：

```cpp
switch (axisK_) {
    case 1:
        DeduplicateVf<1>(...);
        break;
    case 2:
        DeduplicateVf<2>(...);
        break;
    case 4:
        DeduplicateVf<4>(...);
        break;
    case 8:
        DeduplicateVf<8>(...);
        break;
    default:
        DeduplicateVfDynamic(..., axisK_);
        break;
}
```

具体特化集合必须根据生产模型真实 K 分布选择，不能默认固定为示例中的 1/2/4/8。模板化时还应同时评估 `SqueezeTokenIdxVf<K>`；`PrefixSlotVf` 与 K 无关，不需要模板化。

最终决策为：

1. 当前版本保留运行时 K=1..32，作为完整规格和功能正确性的统一基线。
2. 本次不生成 32 个 K 模板，避免在没有反汇编和真机数据时引入代码膨胀。
3. 将热点 K 特化作为二阶段性能优化；只有当动态循环确认为瓶颈，且生产 K 集合稳定时实施。
4. 特化版本必须与动态版本逐项比较精度、二进制大小、寄存器 spill、指令缓存和端到端耗时，不能仅以局部循环指令数判断。

## 12. 正确性与性能验证方案

### 12.1 已完成的静态/模型检查

- Host Tiling 约束确认：K 实际为 `1..32`，不是 `0..32`。
- 随机算法模型检查：K=1..32 的 rank 去重和 token squeeze 与标量 golden 一致。
- 极值布局检查：expert=2048、rank=1024、`calCnt=16384` 时 `dstExpBuf_` 不越界。
- 多组/尾组布局检查：不同 K、expert、rank 组合下 guard 和 histogram 区域不重叠。
- `git diff --check` 通过。
- 确认 Hybrid 无代码改动。

这些检查证明算法映射和源码级内存规划自洽，不等价于编译器和硬件验证通过。

### 12.2 必须补齐的 CANN 9.2.0 编译验证

当前环境未发现 BiSheng/CANN 编译器，因此以下项目仍是合入门槛：

1. 所有原生 VF 类型和 intrinsic 在 9.2.0 下的签名、类型转换和 `POST_UPDATE` 用法通过编译。
2. `vdintlvv2`、`pdintlvv2_b8`、`dhistv2`、`vselr`、`vsqz`、`vstur`、`vstar`、`sprsts` 的 lane/layout 与参考一致。
3. 查看编译日志，确认没有明显 vector register spill、UB 超限和非法流水依赖。
4. 检查生成指令中不再出现原旧路径的大量 Vec API 展开序列。

### 12.3 功能用例

至少覆盖：

- K：1、2、3、7、8、16、31、32。
- token：1、小于一个 VF 批次、等于批次、批次±1、多组、最后一组不足。
- expert/rank：跨 256 边界的 255/256/257，最大 2048/1024。
- 路由：全合法、负数、等于/大于 numExperts、同 token 重复 expert、不同 expert 同 rank、全发同 rank、均匀分布。
- 多 AIV：空核、单组、多组、slot 前序行跨 UB 分组。
- 输出逐项比较：`dstRank`、`sendCntPerExpert`、`sendCntPerRank`、workspace counter、squeeze token index、slot 起始值及最终通信结果。

### 12.4 真机性能验收

用相同输入、相同 Tiling、相同运行次数比较优化前后：

- kernel 总耗时与稳定分位数；
- `CalSendCnt`、`SetSrcTokenIdx`、`GetSlotStartNum` 分阶段耗时；
- vector 指令与 UB 访问占比；
- 不同 K、token 数、expert/rank 规模下的收益曲线；
- 编译器寄存器使用和 spill；
- 通信阶段耗时不应因计算改造回退。

若性能不升，按以下顺序定位：VF spill → histogram 多扫描 → K=32 去重循环 → MTE/V event 等待 → 尾块 guard 访问。只有数据证明 SIMT 在目标规格持续占优时，才考虑新增 SIMT 专用路径。

## 13. 实现可行性结论

从当前代码、Tiling 规格和参考实现看，该方案在算法与 UB 布局上可行：K 的动态切分始终保证完整 token 行，局部 histogram 不溢出，最大 expert/rank 规格可放入既有 `dstExpBuf_`，Direct 的 GM/通信契约不变。原生 VF 所需的核心指令组织均能在参考 `dispatch.hpp` 找到对应用法。

但不能在缺少 CANN 9.2.0 编译器和 950 真机的条件下宣称“最终功能已完全确认”。当前结论是：**实现已完成且源码级设计自洽，具备进入 9.2.0 构建、精度回归和真机性能验证的条件；编译、精度和性能三项结果全部通过后方可合入。**

## 14. 官方资料依据与适用边界

- [CANN 9.2.0-beta.2 文档入口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/index/index.html)
- [CANN 9.2.0 SIMD/SIMT 核函数与 VF 函数约束](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/programug/Ascendcopdevg/docs/zh/guide/%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97/%E9%AB%98%E7%BA%A7%E7%BC%96%E7%A8%8B/%E9%AB%98%E7%BA%A7AI-Core%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B/SIMD%E4%B8%8ESIMT%E6%B7%B7%E5%90%88%E7%BC%96%E7%A8%8B/%E6%A0%B8%E5%87%BD%E6%95%B0%E4%B8%8EVF%E5%87%BD%E6%95%B0.md)
- [CANN 9.2.0 Ascend C API 分类](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/compiler/BishengCompiler/nottoctopics/zh-cn_topic_0000002723184999.html)

当前使用的 `vldus/vstus/vselr/dhistv2/vsqz` 等属于底层、硬件相关的原生 VF 能力，不能假定跨架构兼容。本设计只承诺面向 Ascend 950 与配套 CANN 9.2.0 工具链；后续若扩展到其他硬件，必须重新核对 intrinsic 可用性、lane 布局和性能。

## 15. 修订记录

| 版本 | 日期 | 变更 |
|---|---|---|
| Final v2.2 | 2026-09-18 | 按实际原生 VF 实现复核：修正动态 K 批次、尾块、UB guard、同步链、参考差异和验证状态；删除旧 Vec-API 死代码 |
| Final v2.1 | 2026-09-18 | 确定 Direct 原生 VF 方向，Hybrid 暂不修改 |
