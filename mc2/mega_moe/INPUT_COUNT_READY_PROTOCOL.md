# MTE 输入 count 就绪协议

MTE 路径删除共享专家前的入口跨卡握手，以量化硬件通知和 count epoch 传递输入依赖。
共享入口的 `SyncAll<false>()` 拆成输入就绪和 UB 释放两条硬件依赖；
保留 AIV 本核 `PipeBarrier<PIPE_ALL>()` 及出口跨卡同步，不增加 GM ready 状态。
公共入口位于 `mega_moe_arch35.h`，各 MTE Wave 派生类共用。
Dispatch 通过整表 MTE2 搬运和 VF 批量 epoch 校验等待完整 count 表；失败后退避 500 cycle 重试。
远端 24-bit count + 8-bit epoch 布局和 VF cumsum 不变。

## 两类入口同步分别如何替代

这里的“全卡同步”指参与通信的 rank 之间的入口跨卡握手；“全核同步”指本卡
AIC/AIV 在共享专家入口执行的 `SyncAll<false>()`。两者传递的依赖不同。

### 全卡同步：由 count 自带的 epoch ready 标记传递远端依赖

每个 count 是一个 int32：低 24 位保存计数，高 8 位保存本轮 epoch，兼任 ready Flag。
两部分随同一个 4B 表项发布，不增加独立的跨卡 ready 消息。这里的 24/8 是位数，
与发送 UB 中每项预留的 32B 空间无关。

```text
源卡全部 MoE token/scale 量化写回完成 + 本发送核对应的 index 已写完
  → 发布 count | (本轮 epoch << 24)
  → 接收端 VF 确认全部有效 count 的 epoch 匹配本轮
  → 去掉 epoch，使用 count 做 cumsum，随后 Dispatch 读取源卡 token/scale
```

关键是先满足数据依赖，再允许 count 发布。量化按 token 分核、路由按专家分核，
因此每个 count 发送核都通过 mode 0 MTE3 事件等待全体本卡量化生产者，并在 index
完成后发布 count。接收端匹配到本轮 epoch，才将该项视为远端输入已就绪；
不能仅检查 count 非零，零 count 同样必须携带本轮 epoch。

这样省去 `CrossRankSyncInWorldSize` 入口握手及其末尾 `SyncAll<true>()`，将远端依赖
交给真正消费这些数据的接收 AIV。当前接收实现仍等待整张表就绪，尚未改成某一项
count 到达就启动对应专家的 Dispatch；共享计算则不再受入口跨卡握手阻塞。
出口跨卡同步继续保留，以维持跨轮次窗口复用约束。

### 全核同步：拆成按数据用途和流水划分的硬件依赖

原 `SyncAll<false>()` 把量化、路由/count 发送、清零及共享输入准备一起汇合。
修改后，清零与共享输入准备提前，AIV 发出输入通知后继续生成路由，AIC 只等待
自己开始访问数据所必需的输入条件。具体分成以下几条依赖：

| 依赖 | 通知与等待 | 允许继续的工作 |
|---|---|---|
| 本卡 MoE 量化完成 | 全体 AIV mode 0 flag 8；发送核在 MTE3 等待 | index 后发布带 epoch 的 count |
| 分段清零、共享输入写回完成 | AIV mode 4 通知配对 AIC；AIC 再经 mode 0 汇合本卡输入 | AIC 继续下发共享输入加载和计算 |
| 全体输入 epoch 已记账 | AIC 汇合后用 mode 4 回通知各 AIV | 接收 AIV 可安全读取本轮 epoch |
| 路由/count 源 UB 已用完 | AIV 本核 PIPE_ALL 后用 mode 4 通知；配对 AIC 仅在 FIX 等待 | AIC 可以向配对 AIV 的 UB 写结果 |

因此，移除的是共享入口统一等待整个准备阶段的 `SyncAll<false>()`，改为更细粒度的
硬件流水同步；本卡输入仍需汇合，必要的等待并未消失。AIC 的输入等待仍在 PIPE_S，
但不再把所有 AIV 的路由完成纳入该入口条件；UB 复用等待放在 PIPE_FIX，使输入加载
和 MMAD 可以按各自依赖提前推进。每个 AIV 的本核 PIPE_ALL、MoE 末尾 AIV 同步和
出口同步继续保留。

跨卡的 epoch Flag 是 count 的高 8 位数据；本卡的 mode 0/mode 4 Flag 是硬件事件
计数器。两者不是同一类状态，也没有为本卡同步新增 GM ready 轮询槽。

## 修改前后对比

修改前，每个 AIV 完成自身量化后生成并发送 index/count，接着清零 workspace、
整理共享输入。所有核经过本卡 `SyncAll<false>()` 后，AIV 还要完成入口跨卡握手及
其末尾的 `SyncAll<true>()`，才能进入共享激活或 Dispatch。AIC 不执行跨卡握手，
但其共享 GMM1 与 AIV 激活存在 UB/ping-pong 交接，仍会受到 AIV 跨卡等待的影响。

修改后，分段清零提前到量化之前。同一份量化 UB 直接写通信记录及共享 data/scale，
省去共享输入从通信 GM 读回的 MTE2 搬运，同时保留两套独立的输出布局。
MoE 量化就绪通知控制 count 发布；共享输入及状态就绪通知控制 AIC 入口。
AIC 汇合本卡输入后可以继续下发共享计算，FIX 单独等待配对 AIV 完成路由、释放 UB。
接收端通过 count epoch 确认远端输入可读，入口跨卡握手被删除；count 发送异步返回。

| 项目 | 修改前 | 修改后 |
|---|---|---|
| 量化依赖 | 入口本卡/跨卡同步收口 | 量化后发 mode 0 MTE3 事件，index 后、count 前在 MTE3 等待 |
| 量化发布端 | 无独立通知 | 硬件在量化写回完成后上报，无需标量完成等待 |
| count 发布端 | 非对齐段重排并等待，函数末尾等待 MTE3 | 每项 count 使用独立 32B UB 槽，Normal 二维搬运异步返回 |
| 入口跨卡握手 | AIV 参与，末尾另有 AIV 全核同步 | 删除 |
| 共享输入准备 | 量化写通信 GM，再由 GM→UB→共享 GM 拆分 | 量化 UB 直接写通信和共享 GM，保留独立 data/scale 地址 |
| 共享计算入口 | 等待本卡全部路由、清零、共享输入；配对 AIV 还可能阻塞在跨卡握手 | 输入/清零硬件汇合后下发，FIX 等待配对 AIV 释放 UB |
| 接收 count | 整表搬入 UB 后逐槽标量验证 epoch，缺项标量重读 | 整表 MTE2 搬运、VF 校验；失败后退避 500 cycle 重搬，整表就绪后 VF cumsum |
| 入口本卡同步 | `SyncAll<false>()` 汇合整个输入阶段 | mode 4 配对通知 + mode 0 AIC 汇合；AIV 保留本核 PIPE_ALL |
| 出口跨卡同步 | 存在 | 保留 |

量化就绪由硬件事件传递。原逐 AIV 计数槽用于跨轮次 epoch 记账，入口和出口各推进
一次，兼容原 count 和出口协议。

## 发送与消费顺序

```text
每个 AIV（包括空 token / 无专家任务）：
清零本核负责的 workspace 状态分段
  → 量化自己的 token，每个 token 从 UB 写通信记录及共享 data/scale
      → 最后一条通信记录下发后：推进本核 epoch，Set<0, MTE3>(8)
      → 随后下发最后一个 token 的共享 data/scale
  → Set<4, MTE3>(8)：本核清零及共享输入就绪，不等待其他核
  → 生成、发送自己负责的 compact index
  → 收口 route ring 的完成事件
  → Wait<0, MTE3>(8)：等待本卡全体 MoE 量化生产者
  → PIPE_MTE3 barrier
  → 给 count 添加 epoch，S_MTE3 保护标量写入的 UB 数据
  → 异步发布 count
  → 本核 PIPE_ALL：确认 route/count 源 UB 可复用
  → Set<4, MTE3>(10)：释放本核输入阶段 UB
  → Wait<4, S>(9)：接收本卡输入已汇合的回通知
  → 共享激活及 MoE 阶段

每个 AIC（与上述 AIV 并行）：
Wait<4, S>(8)、Wait<4, S>(24)：等待配对两个 AIV 的输入/清零
  → Set<0, FIX>(9)、Wait<0, S>(9)：汇合所有 AIC，覆盖本卡所有输入
  → Set<4, S>(9)、Set<4, S>(25)：回通知两个 AIV
  → Wait<4, FIX>(10)、Wait<4, FIX>(26)：只在 FIX 上等待输入 UB 释放
  → 下发共享 GMM1：MTE2/MTE1/MMAD 可按自身依赖推进，FIX 输出受上述等待保护

接收 AIV1：
本核 PIPE_ALL 及输入硬件回通知完成
  → 读取本轮 epoch
  → 立即 MTE2 搬入整表，MTE2_V 后由 VF 校验有效槽的 epoch 并去 tag
  → V_S 后标量只读取一个整表未就绪标志
      → 未就绪：退避 500 cycle，V_MTE2 保护 UB 复用后重搬整表
      → 已就绪：直接使用本次快照计算 prefix
  → Dispatch 拉取源卡 token/scale
```

量化按 token 分工，路由按专家分工。某专家可以引用其他 AIV 量化的 token，故 count
发布必须等待全体量化生产者。硬件 mode 0 汇合本卡所有 AIV 的 MTE3 完成通知。
无专家任务的核在返回路由函数前也必须消费一次事件；AIC 不发、不等该事件。

在 Ascend 950 上，Wait 的 `PIPE_MTE3` 参数生效：等待进入 MTE3 队列，不要求标量
同步等完再下发 count。Set 位于 index 之前，因此慢核仍可生成 index，快核只在发布
count 的 MTE3 前等待。清零与共享输入写回均放在该等待之前；后续同一 MTE3 队列
仍受等待影响。MoE 事件位于最后一个共享 token 写回之前，但较早 token 的共享写回
也在同一 MTE3 队列中，因此两条依赖拆开并不代表获得两条独立的 MTE3 流水。

发送端已有 route ring 的 `MTE3_V` 完成等待，并在 count 前保留 `PIPE_MTE3` barrier。
协议依赖上述操作在设备上保证 index/量化写回先于 count 可见；CPU 模型不能证明
NPU 互连可见性，仍需多卡设备验证。

## 共享输入与 AIC 入口

`QuantizeLocalTokens` 使用原有双缓冲：BF16 输入只加载、量化一次，输出直接写通信 GM
和共享 GM。共享 data 行长为 `hiddenDim / packedElementsPerByte`，scale 行长为
`ceil(hiddenDim / 64) * 2`。`hiddenDim % 64 == 32` 的补齐 scale 保持为零，通信记录的
padding 和可选 topk weights 不进入共享 data/scale。每个双缓冲的 `MTE3_MTE2` 复用事件
放在两个目标都下发之后，保证共享写回结束前不会覆盖量化 UB。

当前共享和 MoE 使用相同量化配置，因此可复用量化结果。输出仍是两套独立地址，
没有让 BlockMmad 读取通信记录布局。未来共享与 MoE 激活量化类型不同时，需要从原始
BF16 UB 分别量化并按各自完成边界通知；本修改不新增混合量化配置或宣称已经支持该组合。

公共基类在量化前下发分段清零，保留原分核和清零范围。输入通知排在同核 MTE3 清零、
共享写回之后；AIC 汇合所有配对通知，才可读取整张共享输入及同步状态区。
这样无需新增 GM 软同步槽，也不需要一个 AIV 串行清零整个 workspace。

共享 GMM 首次 Fixpipe 可直接写配对 AIV 的 UB，而此时 AIV 可能仍在同一地址生成路由。
因此输入就绪不能等同于 UB 可写：保留每个 AIV 的 PIPE_ALL，再发布独立的 UB 释放事件。
AIC 在 FIX 上等待这两个事件，不能把该等待前移到 AIC 全局汇合之前，否则可能阻塞
汇合事件并形成依赖环。无共享专家、空 token、无专家任务或无计算 tile 的核也必须
完整发出并消费这些事件，不能让分支遗留计数器。

预期可重叠的是 AIC 的输入加载与计算。真实推进程度还受原 GMM 内部依赖约束，例如
A8W4 权重 prologue 需要配对 AIV，不能据此保证所有路径的 MMAD 都同样提前。
清零前移和共享直写也会改变输入阶段 MTE3 排队，最终净收益需要设备测量。

## 接收端整表轮询

每个物理 block 的 AIV1 准备自己的完整 count 表。表项数为
`worldSize * moeExpertPerRank = moeExpertNum`，当前 arch35 最多 2048 项，即 8 KiB；
固定总专家数时，表大小不随卡数增长。每次重试都重新搬入全部有效项。

`CheckMteCountEpochsVF` 逐向量比较高 8 位 epoch，将所有有效 lane 的不匹配结果累积并
归约成一个标量，同时在该 UB 快照中保留低 24 位 count。尾部无效 lane 不参与比较和
写回；即使 count 为零，也必须匹配本轮 epoch。只有整表匹配后才允许计算 cumsum。

整表未就绪标志临时复用 `expertTokenNumsOutTensor` 的首项，与 count 区不重叠，不增加
UB 预算。`MTE2_V` 保护 VF 读取，`V_S` 保护标量读取归约结果，`V_MTE2` 保护下一轮
搬运覆盖 count UB。成功后沿用原 `S_V`，再由 cumsum VF 覆盖专家计数输出区。

首次读取不退避；失败后按现有 `GetSystemCycle()` 计数口径等待 500 cycle，再下发下一次
整表搬运。该常量仅用于 MTE 输入 count，其他 GM 状态轮询仍沿用原间隔。重复搬运以
减少标量逐项处理为目的，实际带宽占用和检测延迟需要设备 profiling 验证。

## count 的 UB 布局和生命周期

```text
发送端 UB： [count0:4B][padding:28B][count1:4B][padding:28B] ...
远端 GM：  [localExpert][sourceRank]，每项仍为 4B tagged count
```

`MTE_COUNT_UB_STRIDE=8` 同时用于 host 预算、kernel 分配和索引。Normal 模式下每个
block 的 UB 步长为 32B。实际搬运参数是 `blockLen=4`、`srcStride=0`，所以每个
block 仅向目标写入 4B，旁边的 28B padding 不写入远端；GM 的额外步长为
`(worldSize-1)*4B`，相邻目标表项地址相差 `worldSize*4B`。这是接口层面的有效写入长度，
不代表物理链路每个事务恰好为 4B；底层事务粒度及协议开销需要设备测量。

一个 sourceRank 写入 `[localExpert][sourceRank]` 表中的一列，各源卡填充后整表连续，
每项只有 4B，没有发送 UB 的 padding。接收 AIV 直接整表搬入 UB 供 VF 校验及 cumsum，
最多 2048 项、8 KiB。因此每项预留 32B 增加的是发送端 UB 占用，没有把接收表扩大八倍。

任意专家分段的 UB 起点都对齐，不再复用 `topkIndexTensor` 重排 count，段间和函数末尾
均无需 `MTE3_S`。每个目的 rank 仍合并为一次二维搬运，不新增跨卡状态消息。

发送端 count 固定占用从 `align32(maxOwnedExperts*4)` 改为 `maxOwnedExperts*32` 字节。
Host 自适应 route buffer 预算同步扣除这部分占用，因此部分 shape 的 route batch 或
ring 深度可能变化，需要一起评估性能。

Count 源 UB 是输入阶段尾部的独立区域，不与 workspace 清零源或共享输入双缓冲重叠。
所有 tagged count 在搬运下发前写好，返回后直到入口 `PIPE_ALL` 都不再改写。该同步
完成之后 AIV 才能发布 mode 4 UB 释放事件；AIC 的 FIX 等待该事件后才可覆盖配对 UB。

## 硬件事件与 epoch 生命周期

量化硬件事件使用 mode 0、flag ID 8。避开本地 SDK `SyncAll` 使用的 11--14；
GMM 的 mode 4 事件使用独立计数器空间。所有 AIV 每轮恰好 Set/Wait 各一次，
入口 `PIPE_ALL` 保证等待已执行，不能让无专家任务的核积累未消费事件。
不依赖不同 flag ID 的 Set 执行先后关系。

新增输入硬件事件如下；mode 0 与 mode 4 使用独立计数器空间：

| 模式 / flag | 生产者 → 消费者 | 边界 |
|---|---|---|
| mode 0 / 8 | 全体 AIV → 每个 AIV 的 MTE3 | 本卡 MoE 量化写回 |
| mode 4 / 8（AIC 收到 8、24） | 每个 AIV 的 MTE3 → 配对 AIC 的 S | 分段清零、共享输入写回 |
| mode 0 / 9 | 全体 AIC 的 FIX → 每个 AIC 的 S | 本卡输入汇合 |
| mode 4 / 9、25 | AIC 的 S → 配对两个 AIV 的 S | 本卡输入及 epoch 记账完成 |
| mode 4 / 10（AIC 收到 10、26） | 每个 AIV 的 MTE3 → 配对 AIC 的 FIX | 输入阶段 UB 可覆盖 |

mode 4 避开 GMM/prologue 使用的 0/1、4--7，以及 SyncAll 保留的 11--14。
AIV1 的配对偏移为 16，与本地 950 SDK `SyncAllImpl<false>` 的 mixed 1:2 实现一致。
AIC mode 0 发布使用 FIX，与 SDK 一致；950 编译器不允许该发布指令使用 PIPE_S。

原逐 AIV 计数槽保持：

```text
rankSyncBase + 48 KiB + aivIndex * 64B
```

仅所属 AIV 标量读写，不清零、不使用 DMA 写入。若上次出口计数为 `s`：

| 时刻 | 槽值 | 含义 |
|---|---|---|
| 本轮入口记账之前 | s | 上轮出口序列 |
| 量化写回下发后 | s + 1 | 本轮 epoch 已分配；不代表量化已经完成 |
| 硬件量化等待完成后 | s + 1 | 此时才允许 MTE3 发布 count |
| 出口同步 | s + 2 | 本轮出口序列 |

发送函数直接使用发布函数返回的完整序列；count epoch 为 `(sequence & 0x7f) | 0x80`。
递增使用无符号加法，保持符号边界及 uint32 回绕行为。

输入硬件汇合及回通知确保所有本卡 epoch 记账完成。接收端在其后读取 slot 0；MoE 阶段末尾
AIV 同步保证所有接收者读完后，才可能在出口更新 slot 0。现有整表等待和出口握手继续
约束各 rank 的跨轮次进度。首次窗口初始化、rank 调用顺序和活动核集合约定沿用原协议。

## 验证

本地验证（2026-09-08）：CPU 模型检查已通过，覆盖 count 发送、整表 VF 轮询、
输入硬件事件配对及共享输入布局；验证脚本不随本 PR 入库。
这些检查不替代真实 NPU 内存顺序、精度或性能验证。Host tiling 翻译单元编译通过。
CANN 9.2 beta2、Ascend950PR_9599 下 A8W8/W4 两组共 36 个 MTE tiling key 编译通过，
包含 combine 量化和 topk weights prefetch 变体。内核编译使用验证专用 Hcomm shim
屏蔽当前 SDK 缺失的无关 URMA API；URMA 和真实多卡运行不在本次验证范围内。
提交检查通过。

设备验收仍需覆盖共享专家开关、A8W8/A8W4/A4W4、空专家、小 bs、连续调用及多卡精度。
性能对比需同时查看算子耗时、共享首个 MMAD/FIX 的启动时间、MTE3 排队、
量化后的标量停顿和 route buffer 配置变化。

## 接口依据（CANN 9.2 beta2）

- [CrossCoreSetFlag](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md)
- [CrossCoreWaitFlag](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.md)
- [SyncAll](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md)
