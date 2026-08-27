# Combine 阶段收发并行优化方案

## 核心思路

把 combine 阶段从"所有核先发完 → 全核同步 → 所有核再收"的串行模式，改成"一部分核专门发、一部分核专门收"的并行模式，让发送和接收同时进行。

## 四个要点

### 1. 收发分核

初版先按固定比例分配，比如发:收 = 3:1，一部分核专门发、一部分核专门收，两组同时工作、互不等待。

### 2. 发送按 token 切分

发送组内部按 token 数量均分，每个核负责一部分 token 的发送。

### 3. 发送按 tokenId 从小到大有序发出

目的是让接收侧 window 中 tokenId 小的区域先被填满。实现方式可以排序，也可以不排序——用分批循环近似（把 token 划分成若干批，逐批发送，每批内的 tokenId 范围从小到大递增）。具体用哪种方式，需要评估排序耗时和分批循环耗时哪个更少，取耗时少的方案。

### 4. 接收按 tokenId 分配核

接收组内部也按 tokenId 有序分配，每个核负责一段交错的 token 范围（如核0负责 token 0, N, 2N...，核1负责 token 1, N+1, 2N+1...），配合发送侧的有序性，让每个接收核都能尽早拿到自己负责的 token 开始处理。

## 整体目标

通过收发并行 + 有序发送 + 有序接收，把原来的"串行总时间 = 发送 + 接收"压缩到"并行总时间 ≈ max(发送, 接收)"，同时缩短接收侧的等待时间。

## 涉及文件

| 文件 | 说明 |
|------|------|
| `mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h` | kernel 主体，Process/发送/接收逻辑的主要改动 |
| `mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h` | tiling 结构体定义，新增分核比例等参数 |
| `mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp` | tiling 计算，发送/接收核数、token 切分 |

## 当前打点结论：全 token 轮询不是优化方向

### 现象

最新 `core_summary` 中，接收核的 `wait_recv_cycles` 与 `wait_recv_polls` 关系比较稳定：

| 接收核 | wait_recv_cycles | wait_recv_polls | 每轮轮询开销 |
|---|---:|---:|---:|
| core 48 | 16588 | 26 | 约 638 cycles |
| core 52 | 30456 | 44 | 约 692 cycles |
| core 68 | 30216 | 50 | 约 604 cycles |

因此当前一轮 `WaitDispatch` 本身已经有约 0.6～0.7us 的开销（1000 cycles = 1us）。如果同时轮询多个 token，这个开销会按未完成 token 数量放大，不能解决接收启动晚的问题，反而会增加接收核的空转时间。

当前应保持“当前 token 完成后再检查下一个 token”的顺序，不恢复全 token 轮询。严格顺序的主要问题是队头阻塞，但在降低单 token 轮询成本之前扩大轮询范围，收益会被轮询开销抵消。

### 当前 shape 的轮询成本估算

以 `bs=96、h=7168、k=6、expandx=fp8_e4m3fn、quant_mode=4` 为例：

1. 一个 token 的通信数据需要约 16 个 packed block。
2. 一个 token 的 6 个槽位需要检查约 96 个 flag block。
3. 每个 flag block 为 32B，一轮检查约搬运和比较 3KB flag 数据。
4. 代码还会对 compare 计数做 64 对齐，并执行 UB 清零、MTE2/V 同步和 compare mask 扫描。

所以目前瓶颈不是简单的“轮询次数多”，而是“每次轮询检查范围太大”。

## 优先优化方案

### 方案一：哨兵 flag 预检查，再执行完整检查

保留现有完整检查作为最终正确性判断，增加一个很轻量的预检查：

1. 对当前 token 的每个槽位只检查一个哨兵 flag，建议使用该槽位最后一个 packed block 的 flag。
2. 任意一个哨兵 flag 未到，立即结束本轮轮询。
3. 所有哨兵 flag 都到后，再调用现有完整 `CheckPackedFlagRangeArrive`。
4. 只有完整检查通过后，才认为 token 到齐并进入后续处理。

哨兵只能用于快速判定“肯定未完成”，不能直接用于判定“已经完成”。这样不会改变通信完成语义，也不会依赖发送端 flag 写入顺序。

当前 shape 的普通轮询可以从检查约 96 个 flag block 降为检查 6 个哨兵 flag。理论数据检查量约减少 16 倍，实际收益还要扣除同步和控制指令的固定开销，但这是目前最有希望降低 `wait_recv_cycles` 的方向。

### 方案二：避免每轮重复清零 compare buffer

`CheckPackedFlagRangeArrive` 当前每轮都会对对齐后的 compare buffer 执行全量 `Duplicate` 清零。有效 flag 区域随后会被 `DataCopyPad` 覆盖，真正需要清零的只有对齐产生的尾部。

可以在进入轮询前完成一次初始化，后续每轮只更新有效 flag 区域，避免重复的向量清零和相关 barrier。需要保证不同路径（普通、expert mask、特殊/共享专家）使用的 compare 长度和尾部范围不会互相覆盖。

### 方案三：暂不增加多 token lookahead

有限 lookahead（例如同时观察 2 个 token）可以缓解严格顺序带来的队头阻塞，但在当前每轮约 600～700 cycles 的情况下，额外检查会直接增加轮询成本。

建议先完成哨兵预检查和 compare buffer 初始化优化，再根据新的 `wait_recv_cycles / wait_recv_polls` 结果决定是否尝试 2-token lookahead。不要直接恢复全 token 轮询。

### 方案四：后续再考虑已到达槽位缓存

一个 token 的多个槽位可能分批到达。可以记录已经确认到达的槽位，后续轮询跳过这些槽位，只检查未完成槽位。这个方案可以减少重复访问，但需要增加状态维护和分支，复杂度高于哨兵预检查，暂列为后续方案。

## 推荐验证顺序

每个优化单独验证，避免无法判断收益来源：

1. 先实现哨兵预检查，保持原完整检查不变。
2. 对比 `wait_recv_cycles`、`wait_recv_polls`、`wait_recv_cycles / wait_recv_polls`、`format_end_cycle` 和总耗时。
3. 再单独去掉循环内的全量 compare buffer 清零。
4. 如果轮询开销明显下降但仍有队头阻塞，再尝试 2-token lookahead。
5. 最后根据不同 `bs/h/k/moeExpertNum/卡数/量化模式/专家 mask` 复测，不能只根据当前 `bs=96、h=7168、k=6` 的结果决定方案。

需要区分两个概念：哨兵预检查只能减少接收核发现完成状态的成本，不能让网络中的完整 token 提前到达。如果优化后 `wait_recv_cycles / wait_recv_polls` 降低，但接收开始时间仍然很晚，应继续分析发送端 MTE3 的实际完成时间和最后一个槽位的到达时间。

## 发送核完成后加入接收

### 固定轮数不能通用

需要平衡的是两类核的最终结束时间，而不是简单保证两类核的 token 数不同：

```text
纯接收核完成时间 = 接收开始时间 + 接收任务耗时
发送后接收核完成时间 = 本核发送耗时 + 加入接收后的任务耗时
```

如果发送核比纯接收核晚 `deltaTime` 开始接收，单 token 平均接收成本为 `tokenCost`，静态分配理论上需要满足：

```text
纯接收核比发送后接收核多处理约 deltaTime / tokenCost 个 token
```

但 `deltaTime` 和 `tokenCost` 会随 `bs/h/k/卡数/量化模式/mask/共享和特殊专家/实际通信拥塞` 改变，因此固定多一轮或者固定多几个 token 都不能在所有场景达到最优。

### 最通用方案：单 token 动态取任务

使用一个本 rank 内共享的单调递增 `nextToken`：

1. 纯接收核启动时先静态处理第一批 token，避免启动阶段争抢计数器，并保证最小 tokenId 最先开始接收。
2. `nextToken` 初始值为第一批已经分配的 token 数。
3. 纯接收核完成当前 token 后，通过返回旧值的 `AtomicAdd` 领取下一个 token。
4. 每个发送核完成本核发送并执行 MTE3 drain 后，初始化接收资源，然后加入同一个取任务循环。
5. 每次只领取一个 token，领取值超过 `tokenNum` 后退出。

```text
纯接收核：接收 -> 领取 -> 接收 -> 领取 -> ...
发送核：  发送 -----------------> 领取 -> 接收 -> ...
```

这种方式不需要预先估算任务比例：启动早、处理快的核自然领取更多 token；发送结束晚或者遇到重 token 的核自然领取更少 token。每个核同时只等待一个已经领取的 token，不会退化为一个核反复扫描全部 token。

工程实现需要注意：

- `nextToken` 必须按当前 `dataState` 或迭代代数隔离，避免连续执行读到上一轮计数。
- 计数器初始化不能引入全核 `SyncAll`。可以由第一个纯接收核提前初始化并发布 ready 状态，其他核处理完首 token、第一次动态领取时再检查 ready。
- 动态任务的编号是逻辑 token 顺序；expert mask 场景继续通过已有有效 token 映射得到实际 `tokenIndex`。
- 一 token 一次原子操作可以得到最细粒度的均衡，但需要单独测试标量原子竞争开销。当前 token 上限不超过 512，首批静态分配还能减少 `recvAivNum` 次原子操作。

## 按分核比例计算的静态任务方案

如果暂时不引入共享计数器，可以先使用按分核比例计算的静态加权方案。定义：

```text
A = aivNum
R = recvAivNum                 // 从开始就纯接收
S = sendAivNum = A - R         // 发送完成后加入接收
N = 本次需要接收的逻辑 token 数
```

假设纯接收核的接收能力权重为 `1`。发送核需要先完成发送，其可用于接收的有效时间较短；只使用分核比例进行估算时，把每个发送后接收核的权重设为：

```text
lateWeight = S / A
```

这个权重满足合理的单调关系：发送核越多，每个发送核平均发送任务越少、越可能较早加入接收，因此权重越接近纯接收核；发送核越少，每个发送核平均发送任务越多，加入越晚，因此接收权重越低。

两组的总权重为：

```text
totalWeight = R + S * S / A
```

静态分给纯接收组和发送后接收组的 token 数为：

```text
earlyTokenNum = N * R / totalWeight
lateTokenNum  = N - earlyTokenNum
```

等价的整数计算形式为：

```text
earlyTokenNum = N * A * R / (A * R + S * S)
```

为了让同组核任务均衡，并保持发送波次与接收分配一致，可以把 `earlyTokenNum` 取到最接近的 `R` 的整数倍，再限制在 `[0, N]`。当 `N <= R` 时直接令 `earlyTokenNum = N`，不再给晚加入核分配任务；其他场景至少给纯接收组保留第一波 `R` 个 token。低 tokenId 的 `[0, earlyTokenNum)` 分给纯接收核，剩余 `[earlyTokenNum, N)` 分给发送后接收核，两组内部都使用交错分配。

以 `N=96、A=72` 为例：

| R | S | 公式计算的 earlyTokenNum | 按 R 对齐 | 纯接收核平均任务 | 后加入核平均任务 |
|---:|---:|---:|---:|---:|---:|
| 24 | 48 | 约 41 | 48 | 2 | 1 |
| 32 | 40 | 约 57 | 64 | 2 | 0.8 |
| 36 | 36 | 64 | 72 | 2 | 约 0.67 |

例如 `R=24`：

```text
纯接收核 i：token i、token i + 24，共处理 token 0～47
发送后接收核 j：token 48 + j，共处理 token 48～95
```

发送侧继续按 token 范围逐波发送时，token 0～47 会优先供给纯接收核；发送核完成后加入接收时，token 48～95 已经处于后续波次或接近完整到达，能够减少无效轮询。

这个比例权重没有引入额外的经验轮数，适合作为静态方案的第一版验证，但它仍然基于“发送核有效接收时间约为 `S/A`”这一近似。最终应通过以下指标判断是否平衡：

- 两组核的最大结束时间是否接近。
- 发送核完成后是否仍存在较长的接收尾部。
- 每组的每核实际完成 token 数和平均 token 处理成本。
- 不同 rank 的发送结束分布，而不只是全局最大值。

如果两组结束时间仍明显不一致，说明仅靠分核比例不能描述当前 shape 的发送和接收成本，应切换到动态取任务方案，而不是继续增加固定轮数或经验常量。
