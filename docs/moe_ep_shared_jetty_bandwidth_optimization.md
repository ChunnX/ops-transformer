# MoE EP Shared-Jetty Dispatch/Combine 带宽优化方案

> 2026-09-19 实现状态：当前工作区采用 **32 Jetty/SQ4096/CQ64**：32 sender 独立均分 BS，剩余 `aivNum - 32` 个 prepare AIV 独立均分 BS，以区间交集建立 ready 依赖。Dispatch payload 跨 peer 复用 owner batch，首批32 WQEBB、后续512 WQEBB；Combine 按每 peer metadata 区间做32路切分。卡数不参与核/Jetty绑定。下文 `(P-1)*C_ctx=J`、56 Jetty和peer-exclusive lane保留为历史基线；当前设计、实现边界与后续实验以 [共享 Jetty 分组与带宽对照方案](moe_ep_shared_jetty_grouping_comparison.md) 为准。尚未执行 NPU 编译、运行或性能测试。

> 阅读约定：从第 1 节起出现的“当前代码”“本次 SQ=2048”“固定 8P/56 Jetty”等表述，均指 2026-09-18 的旧方案快照，不描述当前工作区，也不能作为本版验收条件。当前工作区只以本页首段和上述对照方案为准。

## 1. 目标、范围与结论

本文针对当前 `sharedjettyD_C` 分支的 MoE EP continue 路径。BS=4096、K=6、H=7168、256 专家、8 卡只是本文的典型性能算例，不是实现限制；协议、tiling、window 和 kernel 分核必须支持运行时 `epWorldSize`。目标是：

1. 提高 Shared-Jetty 通信活跃区间内的有效带宽，尽量接近单卡单向理论带宽 400 GB/s；
2. 降低 `MoeEpDispatchTotal` 和 `MoeEpCombineTotal`，而不是只缩短发送 kernel 的请求下发时间；
3. 在 SQ 深度 2048 WQEBB、CQ 深度 64 的约束下，不破坏 data-before-ready，也不在消费者完成前复用 data/state window；
4. 保持主线输出排布、接口和数值语义，优先做可独立验证的最小改造。

核心结论如下。

- 典型 8P 均匀负载不是“单次请求把 SQ 填满”的问题。估算每条 lane 的 Dispatch 仅约 191 WQEBB，Combine 约 385 WQEBB（无权重）或 769 WQEBB（带权重），均显著低于 2048；其他卡数必须用运行时参数重新计算。
- Dispatch 当前每条 lane 约 48 个数据请求，小于 `HCOMM_BATCH_CAPACITY=128`，所以通常一直到末尾状态请求加入后才第一次 `BatchCommit`。这会延迟首包，是优先级最高的低风险优化点。
- 当前 payload 和最终状态均为 `cqe=0`；CQ=64 不在热路径上承担逐请求完成记录，因此它不是当前典型负载的直接带宽瓶颈。不要为了“看见完成”给每个 payload 或每个 batch 打 CQE。
- profiler 把大部分时间归入 epilogue；其中网络等待、全局 barrier 和本地处理的比例尚未分离。全量 rank/channel ready barrier 是最重要的候选瓶颈，需要在 P0 证实后再改成按 channel/token chunk 增量消费。
- `BatchCommit` 只是提交，不是远端完成。发送 kernel 的 55~76 us 不能用作网络传输时长；真实完成边界应以接收端最终有序状态可见为准。
- 当前代码把 world size 写死为 8，并由 `56/(8-1)` 写死每 peer 8 条 lane；这不应被性能优化继续固化。必须先把 `epWorldSize`、`channelsPerPeer`、有效 handle/flag 数参数化，再按实际拓扑和 AIV 数分核。
- 不建议直接固定“一半核发送、一半核计算”。应保留全部运行时 active Shared-Jetty lane 的并发能力，再按实测的准备/下发比例扫描分核点；发送核少于 active lane 数时，一个发送核可以轮转管理多个互斥 lane。

建议按 P0 → P1 → P2 → P3 依次实施：先建立真实完成边界，再做首批早提交，随后改造接收侧增量消费，最后才考虑持久化融合和跨轮双缓冲。

## 2. 基线与带宽账本

### 2.1 典型配置

| 参数 | 值 |
| --- | ---: |
| 每卡 BS | 4096 |
| TopK | 6 |
| hidden size | 7168 |
| 数据类型 | BF16/FP16，2 B |
| 全局专家数 | 256 |
| 卡数 | 8 |
| 每卡专家数 | 32 |
| Shared Jetty | 56，共 7 个远端 rank × 每 rank 8 条 lane |
| SQ 深度 | 2048 WQEBB/lane |
| CQ 深度 | 64 CQE/lane |
| 理论带宽 | 400 GB/s，本文按单卡单向聚合带宽计算 |

全文使用以下运行时符号：

| 符号 | 含义 |
| --- | --- |
| `P` | `epWorldSize`，不固定为 8 |
| `E` | 全局专家数 |
| `J_max` | 当前平台/context 能容纳的 Shared-Jetty handle 上限；当前实现为 56 |
| `J` | 当前 context 实际创建/取得的 handle 数，首版等于 `(P-1)*C_ctx` |
| `C_ctx` | context 为每个远端 peer 分配的物理 lane 数 |
| `C_active` | 当前调用内全 communicator 一致的每 peer active lane 数；`P>1` 时 `1 <= C_active <= C_ctx`，`P=1` 时为 0 |
| `A` | 当前可用 AIV 数 |
| `Q` | 仅在 P2.3 token-channel 协议启用时有效的逻辑 token chunk 数；首版 `Q=C_active` |

在当前“每个远端 peer/channel 独占一个物理 WQ owner”的映射下，`P>1` 时应满足：

```text
(P - 1) * C_ctx = J <= J_max

if A >= P:
    C_active <= min(C_ctx, floor(A / P))
else:
    C_active = 1，AIV 以 rank-stride 方式覆盖多个 peer
```

`P=1` 走纯本地路径，不创建远端 Jetty，也不进入下面包含 `(P-1)` 分母的网络公式。`P>1` 时，`C_ctx` 应由 Host 根据 `P`、创建前已知的 `J_max` 和可选调优参数选择，再创建 `J=(P-1)*C_ctx` 个 handle，而不是通过编译期 `WORLD_SIZE=8` 得到。若资源 API 不能满足请求数量，所有 rank 在发布 context 前一致失败，不能用部分成功的 `J` 反推不同的 `C_ctx`。8P 典型值是 `P=8, J_max=J=56, C_ctx=C_active=8`。

### 2.2 证据快照与输入假设

| 项目 | 当前证据 |
| --- | --- |
| ops-transformer | branch=`sharedjettyD_C`，HEAD=`ce1e81d7ac91f27f9dc81c4127a2e338bfdf4371` |
| asc-comm | HEAD=`610bb2b9486d877c14aba62ac4ed55b2c3e21f9b` |
| cannbot-skills | HEAD=`80bd6c22c85f830774369fc52ab447424921e8f3` |
| SQ/CQ | 本次需求最新输入为 SQ=2048 WQEBB、CQ=64；源码只在运行时读取深度，P0 必须 dump 实际 channel context |
| 理论带宽 | 本次需求输入为 400 GB/s；尚无本机硬件 counter 或隔离通信微基准证明 |
| 性能日志 | 本次对话提供，统计样本 `count=400`；完整命令、原始日志路径、warmup、rank 聚合方式未知 |
| 未知配置 | `HasTopkWeight`、cached/non-cached、具体 dtype、路由分布未随日志给出 |

仓内旧文档 `mc2/D&C性能优化.md:157,173` 记录过 SQ=8192；它与本次最新输入的 2048 不一致。本文按 2048 做容量设计，但任何实现和性能结论都必须以 P0 的运行时 dump 为准，不能把 2048 或 8192 当成固定 ABI 常量。

### 2.3 实测基线

| 阶段 | mean |
| --- | ---: |
| MoeEpDispatch | 75.751 us |
| MoeEpDispatchEpilogue | 1280.826 us |
| MoeEpDispatchTotal | 1356.577 us |
| MoeEpCombine | 55.117 us |
| MoeEpCombineEpilogue | 1931.154 us |
| MoeEpCombineTotal | 1986.270 us |

主 kernel 覆盖发送侧计数、metadata/token table 准备和 `BatchCommit` 下发，但不覆盖远端完成。当前配置均使用 `cqe=0`，且发送端没有完成等待；因此不能把 75.751 us 或 55.117 us 当作网络完成时间，也不能据此计算网络带宽。

### 2.4 端到端等效带宽估算

以下估算适用于 `P>1`，并假设专家在 `P` 张卡上均匀分布、TopK 在 `E` 个专家间近似均匀且不放回采样，排除本卡流量，只计算真正跨卡的数据。一般形式为：

```text
Dispatch expected remote ranks/token
  = (P - 1) * (1 - C(E - E/P, K) / C(E, K))

Dispatch average slots/lane
  = BS * expected_remote_ranks_per_token / ((P - 1) * C_active)

Combine average remote routes/card
  = BS * K * (P - 1) / P

Combine average routes/lane
  = BS * K / (P * C_active)
```

若专家不能被 `P` 整除，或 expert placement 不是均匀连续分布，应按真实 `expert->rank` 映射求和，不能套用上式。实际验收必须改用运行时 `sendCnt` 统计。

在典型 `P=8、E=256、C_active=8` 下，Dispatch 中一个 token 命中不同远端 rank 的期望值为：

```text
E[remote ranks/token]
  = 7 * (1 - C(224, 6) / C(256, 6))
  = 3.8850
```

每个远端 slot 包含 14336 B hidden 和 512 B metadata：

```text
Dispatch remote bytes
  = 4096 * 3.8850 * (14336 + 512)
  = 236.28 MB

Dispatch network-only ideal
  = 236.28 MB / 400 GB/s
  = 590.69 us

Dispatch end-to-end effective bandwidth
  = 236.28 MB / 1.356577 ms
  = 174.17 GB/s = 43.54%
```

Combine 每个 route 只有 7/8 的概率跨卡：

```text
Combine remote routes
  = 4096 * 6 * 7 / 8
  = 21504

Combine remote hidden bytes
  = 21504 * 14336
  = 308.28 MB

Combine network-only ideal
  = 308.28 MB / 400 GB/s
  = 770.70 us

Combine end-to-end effective bandwidth
  = 308.28 MB / 1.986270 ms
  = 155.21 GB/s = 38.80%
```

带权重时还需增加 `21504 * 4 B = 0.086 MB`，总 remote payload 为 308.367 MB；对带宽分子影响约 0.03%，但小粒度 weight 搬运的指令开销不能因此忽略。基线日志未说明是否带权重，所以两种模式必须分别复测。本文采用 SI 单位：`MB=10^6 B`、`GB/s=10^9 B/s`。

这两个“端到端等效带宽”包含 epilogue 的本地扫描、类型转换、向量累加和输出回写，不能直接解释为链路利用率。后续应同时报告：

```text
payload_goodput_window
  = actual_remote_payload_bytes / (last_remote_ready - first_batch_commit)

issue_rate
  = submitted_payload_bytes / (last_commit - first_commit)

completion_tail
  = last_remote_ready - last_commit

BW_e2e
  = actual_remote_payload_bytes / total_D_or_C_time
```

`payload_goodput_window` 包含 sender 组包、metadata 扫描、commit 间隙和 lane 长尾，只能表示活跃窗口 goodput，不能单独证明物理链路达到 400 GB/s。真正的链路利用率要用设备/网络 profiler counter 或隔离通信微基准。若 commit 与 ready 来自不同 rank，只有在多 rank Timeline 已证明统一时基，或完成时钟校准后才能相减；裸 `GetSystemCycle` 不能跨卡直接做差。

## 3. 当前请求深度与 SQ/CQ 判断

### 3.1 Dispatch

Dispatch 已使用多 SGE：每个 slot 为 hidden + metadata 两个 SGE，每个 WQE 最多 12 个 SGE，即一次写 6 个 slot。一个满 WQE 的大小为：

```text
48 B SQE header + 12 * 16 B SGE = 240 B
round_up(240, 64) = 4 WQEBB
```

典型均匀负载下：

```text
remote slots/lane = 15913 / 56 = 284.2
data requests/lane = ceil(284.2 / 6) ≈ 48
payload-phase SQ usage/lane ≈ 191 WQEBB（含末尾状态请求，不含更早的 count exchange）
SQ occupancy ≈ 9.3%
```

当前按 128 个通信请求才中途提交，而典型 lane 只有约 48 个数据请求，因此通常没有中途提交，首个 doorbell 到末尾才出现。这里的主要问题不是 SQ 太浅，而是首批提交太晚。

### 3.2 Combine

Combine 无权重时每个 route 是 1 个 WQEBB；带权重时现有 multi-SGE 把 hidden 与 4 B weight 合到同一个远端连续 slot，单请求为 2 个 WQEBB。

```text
remote routes/lane = 21504 / 56 = 384

无权重：384 + 1 flag = 385 WQEBB，SQ occupancy = 18.8%
带权重：384 * 2 + 1 flag = 769 WQEBB，SQ occupancy = 37.5%
```

当前 Combine 的 batch 上限是 256 WQEBB。无权重通常产生两次提交，带权重通常产生四次提交；但第一次提交分别要等到约 2/3 或 1/3 的 lane 工作已经组织完成，仍有提前首包的空间。

均匀平均值不能替代最坏路由检查。全部命中同一个远端 expert rank 时：

```text
Dispatch worst slots/lane = ceil(BS / C_active)       // per-rank dedup
Combine current route-entry split worst routes/lane = ceil(BS * K / C_active)
```

典型 `BS=4096、K=6、C_active=8` 时，Dispatch 每 lane 最多约 512 个 slot，即约 343 WQEBB，仍低于 SQ2K；Combine 则可能达到：

```text
routes/lane = 4096 * 6 / 8 = 3072
无权重     = 3072 + 1 flag = 3073 WQEBB
带权重     = 3072 * 2 + 1 flag = 6145 WQEBB
```

因此“单个 batch `<2048`”并不能保证安全；还必须满足：

```text
submitted_wqebb - known_completed_wqebb < runtime_sq_depth
```

典型均匀快路径继续使用全 `cqe=0`、无 `Drain`。Combine 在精确预计算的 lane WQEBB 将越过安全水位时进入条件慢路径：把 lane 切为多个 credit window，每个 window 的最后一个 payload WQE 使用 strong order + `cqe=1`，`BatchCommit` 后 `Drain` 确认该 window 完成，再提交下一 window；业务 final flag 只在最后一个 window 后发布。这样每 lane 同时最多保留一个 checkpoint CQE，适配 CQ64。若不接受这个条件慢路径，就必须把相应极端倾斜明确列为不支持，不能宣称 SQ2K 下全路由安全。

### 3.3 CQ=64 的含义

当前 payload 与最终状态均为 `cqe=0`，所以这些请求不会逐条占用 CQE。接收端通过最终强序状态判断远端数据可消费，CQ=64 不是热路径带宽瓶颈。

如果为了诊断或上述倾斜慢路径增加发送端完成点，应只让“能代表一个 credit window 完成的尾请求”设置 `cqe=1`，并在提交下一 window 前回收。不能给每个 payload、每次早提交或每个 token 产生 CQE，否则 CQ=64 很容易成为新的限制。若选择跨 epoch 累积尾 CQE，必须额外保证 `outstanding_cqe < runtime_cq_depth`；本文首选逐 credit window Drain，不跨 epoch 累积。

## 4. 当前关键路径为什么没有重叠

### 4.1 Dispatch

当前发送侧顺序为：

```text
CalSendCnt
  -> SyncAll
  -> Communication(count exchange)
  -> [last AIV] GetRecvCount(wait) -> SendPhase(its token slice)
     [other AIVs] -----------------> SendPhase(their token slices)
  -> SetStatus
  -> SyncAll
  -> WriteToRemoteWindow(all payload + final flags)
```

`GetRecvCount()` 只有最后一个 AIV 真正轮询，其余 AIV 已经进入 `SendPhase()`，所以 count wait 与大部分 table 构建已有部分重叠。当前缺失的是“table 的 chunk 准备完成后立即下发 payload”；所有 AIV 仍在第二个 `SyncAll` 汇合后才进入 `WriteToRemoteWindow()`。

当前接收侧顺序为：

```text
Wait all rank/channel flags
  -> SyncAll
  -> CountHits(scan all received metadata)
  -> SyncAll
  -> BuildRankRowStarts
  -> CopyFromWindowByExpert(scan/copy again)
```

结果是：payload 请求下发、网络传输、接收统计和输出回搬仍按大阶段串行。现有每 rank/channel 状态槽已经提供了比“全局 ready”更细的完成信息，但 `WaitDispatch()` 把它们求和成一个总条件，丢失了增量消费机会。

### 4.2 Combine

发送侧先遍历分配给本 lane 的 metadata 并发送所有 route，之后发布 channel flag。接收侧流程为：

```text
BuildLocalRecvIndex
  -> AIV0 waits all completion-channel flags
  -> clear all flags
  -> SyncAll
  -> every AIV processes its token range
```

`ProcessTopKToken()` 对每个 token 顺序处理 K=6 路 hidden：搬入、BF16→FP32、累加，再 FP32→BF16 回写。4096×6×14336 B 约为 352 MB 的本地读取量，另有约 58.7 MB 输出，并包含约 1.76 亿个元素的转换/累加。因此 1.93 ms epilogue 不只是网络等待；即使网络达到 400 GB/s，也必须通过通信与本地 reduction 重叠、以及 reduction 自身优化，才能继续下降。

另外，当前 Combine 在所有 lane 完成 payload 后执行一次 `SyncAll<true>()`，再统一发 channel flag；因此最快 lane 的 ready 也被最慢发送核拖住。这是无需等待 P2 布局改造就可验证的直接长尾来源。

### 4.3 当前 flag 与拟新增状态的职责

“flag 清零”不是发送端可观察的 ACK。当前两个 epilogue 都是在全量 ready 后先清本地 flag，再执行 CountHits/copy 或 reduction；之所以没有覆盖数据，是因为 continue 调用链在 epilogue 完成前不会启动下一轮对同一 window 的写入，而不是因为 sender 收到了消费确认。

| 状态 | 存放位置 | 唯一写者 | 读者 | 完成含义与复用条件 |
| --- | --- | --- | --- | --- |
| 当前 remote channel-ready | 注册通信 window | 对应物理 Jetty/WQ 的 sender owner | receiver epilogue | 同一 WQ 的前序 payload 已远端完成；本地清零不是 ACK |
| P1 prepare-ready | 本地 workspace/ring | 唯一 prepare chunk owner | 唯一 send-lane owner | chunk 数据已写完且 MTE3 完成 |
| P1 prepare-consumed | 本地 workspace/ring | send-lane owner | prepare owner | 已通过完成checkpoint或显式接收完成反馈确认NIC不再读取该源GM slot；仅提交WQE不允许复用 |
| P2 channelReadyEpoch | receiver 本地 workspace | 每 remote channel 的唯一 poll owner | 相交的 virtual consumer AIV | remote ready 的本地快照，避免多个消费者竞争清 flag |
| P2 virtual-slice counts | 扩展后的注册 count window | source-rank prepare owner | dispatch epilogue | 与旧 `SplitToCore` 等价的 prefix 输入；影响 window ABI |
| P2 combine bucket index | sender 本地 workspace | metadata bucket builder | 对应 send-lane owner | 单次扫描得到的 route 索引，不跨轮持久化 |
| SQ checkpoint CQE | Hcomm CQ | 对应 lane 的尾 WQE | 同一 send-lane owner | 一个 credit window 已完成，可回收 outstanding credit |
| P3 remote consumed-ACK | 新增注册 window | receiver | 下一轮 sender | receiver 已完成消费；只有跨轮重叠时需要 |

P1/P2 首版明确不做跨轮 window 重叠，依赖同一 stream 上 epilogue 完成后再启动下一轮。若要跨轮重叠，必须增加 remote consumed-ACK/epoch 或 data/state 双缓冲；不能把现有 flag clear 改名为 ACK。

## 5. 分阶段优化方案

### P-1：先消除 8P/8-lane 编译期硬编码

这不是性能调参，而是所有通用优化的前置正确性改造。当前以下位置把 8P 固化进了 ABI、校验、window 和 handle 索引：

- `moe_shared_jetty_context.h`：`MOE_SHARED_JETTY_WORLD_SIZE=8`、`MOE_SHARED_JETTY_LANES=56/(8-1)`；
- `moe_shared_jetty_context.cpp`：`rankSize_ == 8` 检查，并固定创建 56 个 handle；
- C++ `elastic_buffer.cpp` 与 Python `elastic_buffer.py`：world size 检查、lane/flag 数和 buffer size 公式；
- `moe_ep_window_layout.h/.cpp`：dispatch notify、combine flag 和 state window 容量；
- Dispatch/Combine kernel：`peerIndex * 8 + channelIndex` 的 handle 索引和固定 channel count。

最简参数化方案是保留 `multiChannelHandles[56]` 作为容量上限，删除 world-size 常量，在 context/tiling 中增加或使用：

```text
epWorldSize
sharedJettyCapacity  // J_max，当前为 56
sharedJettyNum       // 实际创建/取得的 J
channelsPerPeer      // C_ctx
activeChannels       // 当前算子的 C_active
activeJettyNum       // 可选派生值：(P - 1) * C_active
```

Host 按 `peerIndex * C_ctx + channelIndex` 建立唯一映射，只创建 `sharedJettyNum=J` 个 handle；`activeJettyNum=(P-1)*C_active` 只是本轮遍历所需的派生值，不代表重新创建 handle。kernel、state flag、dump metadata 和 Python size 公式全部读取相同的 `P/C_ctx/C_active`。Host 必须校验 `P>=1`；`P=1` 走纯本地分支，`P>1` 时再校验 `(P-1)*C_ctx=J<=J_max` 以及 handle/flag/window 容量不溢出。若 `P-1>J_max`，当前 peer-exclusive WQ 设计无法做到每 peer 至少一条 lane，应明确不支持或另行设计跨 peer 时分复用，不能静默取 0。

handle 与 flag 使用不同的 rank 索引域，不能复用同一个公式：

```text
peerIndex   = rank < localRank ? rank : rank - 1       // [0, P-2]，排除本卡
handleIndex = peerIndex * C_ctx + channel              // [0, J-1]

flagIndex   = sourceRank * C_ctx + channel             // [0, P*C_ctx-1]，包含本卡
```

本轮只等待 `sourceRank * C_active + activeChannel` 对应的逻辑集合；实现可继续使用以 `C_ctx` 为 stride 的物理地址，不能因 `C_active<C_ctx` 把后续 rank 的 flag 错压紧。`A<P` 时一个 AIV 可以 rank-stride 拥有多个 peer 的 channel-0 handle，但每个 handle 仍只有一个 owner，flag 仍按完整 `sourceRank` 编址，不能按 owner 编号。

为避免同一 context 因不同 tiling 反复改变注册窗口，context/window 的不可变容量和所有物理 offset/stride 均按 `P/C_ctx` 计算：按 `P*C_ctx` 分配 flag/state 容量，flag 地址固定使用 `sourceRank*C_ctx+channel`；kernel 每轮只遍历、等待和清理 `channel<C_active` 的 `P*C_active` 个逻辑 flag。handle 容量为 `(P-1)*C_ctx`，本地 rank 只需要 state/consumer 分片，不占远端 Jetty handle。

context cache/tag 与严格复用校验至少包含 `layoutVersion/P/C_ctx`，Python 和 C++ 的 window 公式只能按 `C_ctx` 计算容量，不能因某次调用的 `C_active` 较小而缩小物理布局。基础协议 dump 记录 `C_ctx/C_active`；只有启用 P2.3 token-channel 的 protocol version 才记录并校验 `Q`。后续调用只要 `C_active<=C_ctx` 即可安全复用，增大 `C_ctx` 则必须创建新 context/window。

兼容性要求：context/layout 版本升级，新旧 context tag 不混用；C++ 与 Python 的 size/offset 公式用相同测试向量覆盖至少 1P、2P、4P、8P、16P，以及 `A<P` 的 rank-stride 分支。后续所有 P0~P3 公式和状态数均使用这些运行时字段。

### P0：先建立真实完成边界和负载账本

这是所有分核、batch 和流水参数选择的前置条件。不要在轮询循环内逐次打点，只记录阶段入口、出口和必要的 channel ready 分位点。

建议记录：

| 路径 | 必要打点/计数 |
| --- | --- |
| Dispatch sender | `CalSendCnt` 完成；count exchange 提交完成；每 lane 第一次 `BatchCommit`；末尾 flag `BatchCommit`；实际 slot/WQEBB/commit 数 |
| Dispatch receiver | 等待开始；最早/中位/最晚 channel ready；`CountHits` 完成；prefix 完成；最后输出完成 |
| Combine sender | 每 lane 第一个 payload commit；末尾 flag commit；route/WQEBB/commit 数 |
| Combine receiver | local index 完成；最早/中位/最晚 completion-channel ready；reduction 开始/结束；输出完成；P2.3 启用后再记录 token-channel ready |

同时把以下项目设为进入分核或 reduction 流水改造前的硬门槛：

1. dump 实际 TilingData，包括 `P/J_max/J/C_ctx/C_active/aivNum`、每阶段有效核数、每 rank/channel 切片和 UB 分配；
2. 在等价输入和 `tileCnt=1` 下分别做通信隔离与本地计算隔离，取得 `T_comm`、`T_compute`；
3. 计算 `R=T_comm/T_compute` 并判定 communication-bound、compute-bound 或 balanced；
4. dump 每条 lane 的 runtime SQ/CQ depth，并核对 batch buffer capacity；
5. 只有完成以上数据后，才能确定 prepare/send 核数和 K=6 reduction tile，本文列出的扫描值只是实验候选。

完成边界要求：

- `BatchCommit` 时间是“已下发”，不是“远端已完成”；
- 生产端结束时间需要对应真实异步流水完成点；
- 网络完成优先用接收端最终有序 flag 可见时间；
- 显式轮询等待和有效计算分别统计；
- 多 rank MC2 用完整 profiling/Timeline 采集，不能用只支持单算子语义的简化采集方式替代。
- 发送 rank 的 commit 与接收 rank 的 ready 只有在统一 Timeline 已校准时才能做差；否则分别报告 sender issue interval、receiver ready spread，并用隔离微基准或硬件 counter 判断链路利用率。

验收输出必须按 rank、目标 rank、channel 给出 actual bytes、WQEBB、commit 数和 ready 时间分布。否则平均值会掩盖专家倾斜或单 lane 长尾。

### P1：低风险，先让网络更早启动

#### P1.1 两级自适应 BatchCommit

把“首次提交阈值”和“稳定阶段提交阈值”拆开：

```text
first_commit_wqebb  = 32 或 64
steady_commit_wqebb = 256、512 或 1024 中实测选择
```

必须同时满足：

```text
first/steady_commit_wqebb <= batch_buffer_bytes / 64
batch_buffer_bytes / 64 < runtime_sq_depth
```

当前 Dispatch batch UB buffer 为 `128 requests * 4 WQEBB = 512 WQEBB`，Combine 为 256 WQEBB。在不改 UB 配置时，扫描上限分别只能是 512 和 256 WQEBB；如果扫描 512/1024，必须在同一个改动里扩大对应 batch buffer，重新核算 kernel 的全部 UB 占用和 tiling 上限，否则会先触发 batch buffer capacity 错误。

Dispatch 现有阈值按请求数计数，建议改为实际 WQEBB 计数，避免满 12-SGE WQE 占 4 WQEBB、data 尾 WQE 占 2~4 WQEBB时估算不准。建议扫描：

| 项目 | Dispatch | Combine |
| --- | --- | --- |
| 首批阈值 | 16/32 个 data WQE，或统一为 64/128 WQEBB | 32/64/128 WQEBB |
| 稳态阈值 | 当前 buffer 下最多 128 个满 data WQE；扩 UB 后再试 256 | 当前 buffer 下最多 256 WQEBB；扩 UB 后再试 512/1024 |
| 末尾 | payload 后追加最终有序 flag，再提交 | payload 后追加最终有序 flag，再提交 |

原理：小首批减少“组包完成后网络才启动”的首包延迟；较大的稳态批次降低 doorbell 开销。目标不是把每批做成 2048，而是在首包时延和 doorbell 数之间找到拐点。

预期：主要减少首包空窗和接收侧最早 ready 时间。具体微秒收益没有隔离数据支撑；以 `first_commit`、`first_ready` 和 `last_ready` 的变化为准。如果 commit 数增加而 `last_ready` 不降，应回退该阈值。

#### P1.2 Dispatch 的准备核/发送核并行

当前 count wait 已与多数 AIV 的 `SendPhase()` 部分重叠。P1.2 的新增目标不是重复这段重叠，而是让已准备 chunk 的 payload issue 不再等待完整 table 和第二个全局 `SyncAll`。`CalSendCnt()` 后仍需要一次必要的跨核发布，使 count 数据完整；之后可以把核分为：

- prepare cores：按 BS chunk 扫描 TopK，生成 per-rank `srcTokenIdx` 与 metadata stash；
- send cores：轮询 chunk-ready，使用 `(P-1)*C_active` 条远端 Jetty lane 发送已准备 chunk；
- count exchange：保持每 peer 的 channel-0 所有权；可与 token table 构建和 channel `1..C_active-1` 的 payload issue 并行。

推荐协议：

```text
prepare(chunk N)
  -> MTE3 completion
  -> publish ready[buffer][dstRank, epoch]

send waits ready
  -> submit payload batches for chunk N
  -> wait completion checkpoint or explicit receiver completion feedback
     covering every SQ that reads this source GM slot
  -> publish consumed[buffer][dstRank, epoch]

prepare waits consumed before reusing the ring slot
```

每个 `[dstRank][channel]` 物理 Jetty/WQ 必须有唯一 send owner，count、payload、checkpoint 和最终 flag 都由该 owner 串行提交，禁止两个核各自 `MakeBatchHandle` 并发驱动同一 WQ。当前 count exchange 占用每 peer 的 channel 0；首版让同一个 channel-0 owner 先 `BatchCommit` count、结束该 BatchHandle 的使用，再在同一 WQ 上顺序提交 payload。其他 active channel 可更早发送，最后一个 AIV 的 `GetRecvCount()` 继续与发送重叠。也可以未来为 count 分配独立 Jetty，但不能无所有权交接地复用 channel 0。

每个 prepare ring slot 同样需要唯一 producer，ready 必须包含 `producer/chunkSlot/dstRank/epoch/dstSlotStart/count`；send owner 只能在该 chunk 的连续远端 slot 起点已确定后下发。最终 channel flag 仍必须等该 WQ 的全部 payload chunk 已提交，并保持强序完成关系。ring slot 只有收到全部相关 send owner 在确认NIC完成读取后发布的本地 consumed 才能复用，BatchCommit返回不等于源GM生命周期结束。首版优先使用持久source stash直至本轮完成，有限ring及其完成反馈另行验证；原ready清零不构成反馈。

分核不固定为 32/32。对任意 `A`，先扫描 prepare cores 约为 `A/8、A/4、3A/8、A/2`（取整并保证至少一个 send owner）；典型 `A=64` 才对应 8/16/24/32。发送核少于 active lane 数时，一个核轮转驱动 2 条或更多互不共享的 Jetty handle，不能把 lane 数同步缩减为发送核数。参考 A5 full-mesh 实现时，应复用其“按阶段工作量决定核数”的原则，而不是照搬固定一半核。

风险：当前 dispatch 主 kernel 只有约 75.8 us，分核和 ready 协议本身可能吞掉收益。只有 P0 证明 token-table 准备与首包下发之间存在明显空窗时才实施。

#### P1.3 Combine 去掉 payload 后的全核 flag barrier

把 `InitFlagSource()` 移到 payload 前，由 AIV0 初始化后只做一次必要的 `SyncAll<true>()`；随后每个物理 lane 完成自己的 payload，就立即由该 lane 的唯一 owner 发布 final flag，不再在 payload 结束后等待最慢 AIV。data-before-flag 仍由同一 WQ 的 strong-order flag 保证。

这一改动不改变 remote slot、flag 数量或 epilogue 输出排布，是比 token-channel 重排更低风险的 Combine 优化。必须验证空 lane 仍发布 flag、AIV0 的 flag source 在所有 lane 使用前可见，并记录最早/最晚 ready spread 是否收窄。

#### P1.4 负载均衡而非盲目增加 lane

对典型 BS=4096，应保留 tiling 选出的全部 `C_active` lane。`C_active` 是所有 rank 共同的调用级标量，只能由 Host/tiling 根据全局一致的 `P/A/BS/config` 决定，不能由某个 source rank 的局部 `sendCnt` 单独缩减。对每个 rank 的 `sendCnt` 做均匀切片，并统计每 lane 实际 slot/WQEBB。如果最大 lane 时间显著高于中位数，可采用：

- 按实际 token 数分段，保证每 lane 相差不超过一个 chunk；
- 发送核轮转多个 lane，让准备慢的 rank 不阻塞其他 rank；
- 全局小 BS 时可以由一致 tiling 减少 `C_active`，避免空 lane flag/doorbell；即使某个 source-rank/channel 为空，也必须按统一 `C_active` 发布空 flag。若未来按 per-peer `sendCnt` 非对称缩减，必须新增 channel-count 协商和 per-source flag 协议，不能沿用 `P*C_active`。

### P2：主要收益，改成 ready 子集立即消费

#### P2.1 Dispatch 第一步：channel ready 后立即 CountHits

首版仅适用于 non-cached 路径，并保持现有 virtual consumer-AIV 排布：每个 AIV 仍使用当前 `SplitToCore(slotCnt, aivNum)` 的 rank slot 区间，只把“等待全部 channel”收窄为“等待覆盖本 AIV 区间的 channel”。不能把 sender channel 直接等同于 consumer core。

```text
each virtual consumer AIV owns the original rank/slot slices
  -> wait all sender-channel flags intersecting its slices
  -> CountHits for the original slices immediately
  -> publish consumer-slice-count-ready

all slice counts ready
  -> global prefix
  -> CopyFromWindowByExpert
```

每个 remote channel flag 仍只有一个 poll owner；owner 观察到 ready 后先发布本地 `channelReadyEpoch`，所有相交 consumer AIV 只读这个本地快照。remote flag 在本轮全部消费者完成后统一清理，避免一个 consumer 提前清零导致另一个 consumer 永久等待。这样能把 metadata 扫描与其他 consumer slice 的网络传输重叠，同时保留当前 `[consumerAiv][sourceRank][expert]` 计数维度和 `recv_x` 排布。channel 与旧 slice 的交集必须由 tiling 明确定义，等待、CountHits、prefix 和 copy 使用同一映射。

当前 cached 路径依赖已有 rank offsets 和 cached metadata，没有 channel→cached-row 映射；首版保持原全量等待，不启用 P2.1。只有缓存 ABI 增加 channel/slot 区间且能恢复同一 `recvXIdx` 后，才单独开启 cached 快路径。

#### P2.2 Dispatch 第二步：按 channel 预先分配输出区并立即回搬

要让 copy 也重叠，`rank × channel × localExpert` 计数还不够恢复主线顺序。当前 prefix 包含 virtual consumer-AIV 维度；因此必须在发送侧或接收侧预计算与旧 `SplitToCore` 等价的 `[virtualConsumerAiv][sourceRank][localExpert]` 计数，并记录 sender channel 与这些 slot 区间的交集，才能得到完全相同的输出 offset。随后：

```text
all channel flags intersecting a virtual-consumer slice are ready
  -> use the original-layout predetermined offsets
  -> scan that original slice once
  -> copy hidden/meta/weight directly to final region
  -> publish local slice consumed
```

这可以进一步融合当前 `CountHits()` 和 `CopyFromWindowByExpert()` 的两次 metadata 扫描。必须满足：

- 每个输出区间唯一所有者，不重不漏；
- expert 内 `(consumerAiv, sourceRank, slot)` 顺序与主线完全一致；
- non-cached 先落地，cached 未扩展 ABI 前保持原路径；
- 本地 flag clear 不能称为远端 ACK；首版禁止跨轮复用，跨轮版本必须增加 ACK/epoch 或双缓冲。

如果产品契约未来允许同 expert 内重排，可以退化为更简单的 channel×expert prefix；在“排布与主线一致”的当前要求下不采用该方案。

这一级会修改 count/state 协议和布局，必须同步更新：C++ `moe_ep_window_layout`、Python `elastic_buffer.py` 的 size/offset 镜像、tiling 字段和校验、注册窗口 `requiredBufferSize`、缓存元数据、异常 dump regions/version、离线解析与兼容测试。新增字段要明确初始值、唯一 writer/reader、data-before-ready、清理点、epoch 回绕和提交失败时不得发布 ready。

这是 Dispatch 最主要的端到端优化候选。0.591 ms 只是网络部分在 400 GB/s 下的理想下界；`1.356-0.591=0.766 ms` 混合了本地工作、真实通信差额和已有重叠，不能作为本地时间或理论下界。P0 取得 `T_comm`、`T_local` 和 issue/tail 后，再用 `T_pipeline >= issue/tail + max(T_comm, T_local)` 建模。0.90~1.10 ms 仅作为探索目标，不是性能承诺。

#### P2.3 Combine：把 channel 语义改为目标 token 范围

当前 channel 只是发送 metadata 片段，不能直接表示一段输出 token 已齐。对 `P>1`，首版令逻辑 token chunk 数 `Q=C_active>=1`，并对任意 `BS/Q` 使用平衡区间：

```text
token_begin(c) = floor(c * BS / Q)
token_end(c)   = floor((c + 1) * BS / Q)
channel(token) = ceil((token + 1) * Q / BS) - 1  // BS>0，匹配上面的 floor 边界
```

典型 `BS=4096、C_active=8` 时自然得到 8 个 512-token chunk，但代码中不出现常量 8 或 512。

`P=1` 直接执行纯本地 combine/reduction，不创建 bucket，不发布或等待 remote flag，也不进入上述含 `Q` 的公式。

发送端一次扫描 metadata，把 route 放入 `[dstRank][tokenChannel]` bucket；每个 source rank 完成该 bucket 的 payload 后发布对应 flag。即使 bucket 为空，也必须发布 flag。

接收端对 token channel `c`：

```text
wait flags from all source ranks for channel c
  -> reduce tokens [token_begin(c), token_end(c))
  -> write output
  -> mark local token-channel consumed
```

每个 token channel 只有一个 poll owner；它收齐所有 source-rank flag 后发布本地 `tokenChannelReadyEpoch[c]`，对应 reduction cores 再开始消费。这样能让早到的 token 子区间与其余通信并行，且 token-range 分核天然无输出竞争。remote flag 的清零仍不是 sender ACK；首版在本轮全部 token channel 消费完成后统一清理，并禁止在 epilogue 完成前复用 window。

首版保持 `Q=C_active`，所以每个逻辑 chunk 恰好有一个物理 lane 和一个 ready flag。若希望逻辑 chunk 数独立于物理 lane 数，则必须扩展状态槽，或设计多波复用 + ACK/epoch，不能复用一个未确认消费的 flag。`A<P` 时 `C_active=Q=1`，每个 send owner 以 rank-stride 覆盖多个 peer；协议仍正确，只是没有 token-chunk 重叠收益，不需要写死 8P fallback。

本轮实际等待的 flag 数由 `P*C_active` 决定，注册 window 容量按 `P*C_ctx` 预留。即使典型 8P 首版仍等待 64 个 slot、物理大小不变，flag 从“metadata 分片完成”变成“token 范围完成”，也应升级 dump/protocol version，并让 host tiling 与离线解析显式记录 `P/C_ctx/C_active/Q`，避免新旧 kernel/context 混用。

不要为了分桶把完整 metadata 重复扫描 `Q` 次。优先单次扫描生成 channel index bucket；如果 bucket 构建成本过高，再比较“小 metadata 多次扫描”和“workspace 分桶”两种实现。空 bucket 仍必须发布 flag；`WriteNbi`/`BatchCommit` 失败时必须 Trap，不能发布 ready。

P2.3 改成 token-range 绑定后，最坏单 token-channel route 数要按 `K * ceil(BS/Q)` 计算，而不是沿用当前 route-entry 均分的 `ceil(BS*K/C_active)`。SQ credit 慢路径必须使用前者预判；两者只在 BS 恰好整除 Q 时相同。

Combine 的 0.771 ms 只是网络部分在 400 GB/s 下的理想下界；`1.986-0.771=1.216 ms` 不是测得的本地时间，不能作为理论下界。P0 隔离 `T_comm/T_local` 后再判断 reduction 是否成为主瓶颈。1.30~1.55 ms 仅作为探索目标，不是性能承诺。

#### P2.4 Combine 本地 reduction 优化

通信开始重叠后，`ProcessTopKToken()` 很可能成为主瓶颈。建议：

1. 对固定 K=6 做专用展开，减少标量循环、地址计算和分支；
2. 一次 tile 处理 2~4 个 token，双缓冲 hidden 的 MTE2 搬入与 Vector cast/add；
3. 将同一 token 的 K 个 topk weight 暂存在 UB，最终一次连续写出，避免每 route 一个很小的 MTE3；
4. 保持 FP32 累加及主线规定的累加次序，数值验收必须覆盖误差；
5. 用单核流水数据确认 MTE2、Vector、MTE3 哪一条真正占主导，再决定 tile 大小。

这一项只在 P0 完成实际 TilingData、`T_comm/T_compute/R` bound 判定，且通信重叠后实施，避免优化一个原本被等待完全遮住的局部阶段。

### P3：高风险长期方案

#### P3.1 发送与 epilogue 持久化融合

把 dispatch+epilogue 或 combine+epilogue 合并为一个持久化 kernel，可省去 kernel 边界，并直接维护 per-channel 状态机，实现 `prepare -> issue -> wait-ready -> consume`。代价是核角色、UB 生命周期、异常退出和窗口复用协议明显复杂化，只有 P1/P2 仍达不到目标时再做。

#### P3.2 跨轮双缓冲

如果要让下一轮 Dispatch 与上一轮 epilogue 重叠，必须给 data、state、ready/ack 和 epoch 成对双缓冲。当前单窗口不能仅靠去掉 `SyncAll` 就安全复用；否则下一轮可能覆盖上一轮尚未消费的数据或状态。

## 6. 明确不建议的优化

### 6.1 不在典型均匀快路径末尾直接加 Drain

这样会把远端完成等待重新塞回发送 kernel，使耗时从 epilogue 移回 Dispatch/Combine，并破坏通信与接收处理的重叠。当前设计由接收端最终有序 flag 判定可消费，方向是合理的。

这条规则不覆盖 3.2 定义的 Combine 倾斜慢路径：当预测累计 outstanding 将越过 SQ2K 时，checkpoint 必须出现在越界前，而不能只在整轮末尾补一个 CQE。若确认跨轮 SQ/句柄复用还需要发送端完成回收，应另行设计尾 CQE 生命周期；该结论需要 Hcomm 资源复用实测，不能用 `BatchCommit` 返回值代替。

### 6.2 不把 batch 阈值直接扩大到 2048

Hcomm 实现要求单批 `preSqCnt < runtime_sq_depth` 且不超过 batch buffer capacity，所以在本次 SQ=2048 假设下，2048 本身非法；而典型 lane 总工作也只有约 191/385/769 WQEBB。过大的 batch 只会延迟首个 doorbell。阈值要同时受 runtime SQ、累计 outstanding 和 UB batch buffer 约束，并以 payload goodput、issue rate 和尾延迟选择最终值。

### 6.3 不给每个 batch 或 payload 开 CQE

CQ 只有 64 深度。逐 batch CQE 会增加 CQ 写回、轮询和溢出风险，也不提供比最终 channel flag 更有用的业务完成语义。

### 6.4 不移除最终强序 flag

当前 payload 使用 `odr=5` 的 relax order，最终 flag 使用 `odr=6` 的 strong order；强序 flag 必须等待前面的弱序/强序事务完成，正好承担 data-before-state 的完成协议。强序有吞吐成本，但只用在每 channel 的末尾依赖点是合理的。

`fence=1` 主要约束前序读/原子操作，不应在没有微基准证据时把它当作写吞吐瓶颈。优先优化提交时机和全局等待，不先改 order/fence。

### 6.5 不把 Combine 的多个稀疏 route 强行打成一个多 SGE

多 SGE 写只有一个连续远端目标区。Combine 的目标 slot 由 `srcTokenIdx * topK + srcTopKIdx` 决定，同一 source expert/rank 上的 route 在目标 window 中通常稀疏，不能无布局改造地合成一个连续远端写。现有“hidden + weight 写同一个连续 slot”的 2-SGE 用法是合适的。

### 6.6 不固定 32 核发送、32 核准备

参考实现本身也是按 expert/count 工作量选择 cum-sum 核数，并把剩余核用于 AllToAll。固定一半可能让运行时 active Jetty lane 降并发，也可能让 prepare 侧过度配置。必须通过 P0 数据扫描分核点。

## 7. 建议实验矩阵与停止条件

### 7.1 第一轮：不改协议，只调提交

| 实验 | 变量 | 其他保持不变 | 通过条件 |
| --- | --- | --- | --- |
| D1 | Dispatch first commit=16/32 requests | 现有稳态128 requests | `first_ready` 提前且 `last_ready`/Total 下降 |
| D2 | Dispatch 改为实际 WQEBB 计数 | 相同数据与排序 | 正确性一致，无 batch overflow |
| C1 | Combine first=32/64/128 WQEBB | steady=256 WQEBB | Total 下降，commit/doorbell 增量可接受 |
| C2 | Combine steady=256；扩 UB 后再试512/1024 | 最优 first threshold | payload goodput 提升且 p99 不恶化 |

如果更多 commit 只增加 sender 时间而不提前 `last_ready`，停止继续降低阈值。

### 7.2 第二轮：Dispatch 生产/发送流水

P0 门槛通过后，按 `A/8、A/4、3A/8、A/2` 扫描 prepare cores，并按 `BS/C_active` 的比例选择 chunk 候选；典型 `A=64、BS=4096、C_active=8` 才对应 8/16/24/32 核和约 512-token chunk。同时记录每个发送核管理的 lane 数和物理 Jetty/WQ 唯一 owner。只有当 table 准备与 payload issue 确实重叠、`(P-1)*C_active` 条远端 lane 并发未降低时接受。

### 7.3 第三轮：接收侧增量消费

1. Dispatch 先做 channel-ready CountHits；
2. 验证收益后再增加 channel×expert count 和直接回搬；
3. Combine 首版只做 `Q=C_active` 的平衡 token channel；`Q!=C_active` 等状态槽/epoch 协议就绪后另立实验；
4. 最后优化 K=6 reduction。

每一步都要求以下用例通过后才能叠加下一步：

- count/index/offset/flag 及 non-reduction 输出必须与当前主线 bitwise 一致；
- P1/P2 不改变 K=6 累加顺序，BF16/FP16 输出要求 bitwise 一致；只有 P2.4 主动改变向量流水/累加次序时，才按现有项目精度基线单独批准 `rtol/atol`；
- 零 token、某 rank/channel 零发送、空 bucket 仍发 flag；
- 有/无 topk weight、cached/non-cached、低 AIV 动态 `C_active`/rank-stride；
- 1P local-only、2P/4P/8P/16P 与 `A<P` rank-stride，核对动态 handle/flag/window 计数；
- 均匀、单 rank 倾斜和最坏 Combine lane 超 SQ2K 的条件慢路径；
- 足够长的连续 epoch，验证 flag/window/CQ/SQ 复用；
- 任一 `WriteNbi`/`BatchCommit` 失败时 Trap 且不发布 ready/flag。

### 7.4 建议验收指标

下表中的绝对时间和 320~360 GB/s 目标只对应本文给定的 8P 典型算例；其他 `P/C_active` 必须按实际 remote bytes、隔离带宽和 bound 结果重新建模，不沿用同一微秒门槛。

| 指标 | 第一阶段目标 | 说明 |
| --- | ---: | --- |
| 隔离通信链路带宽 | 320~360 GB/s | 探索目标；由硬件 counter 或隔离微基准验证，不用跨 rank 裸 cycle 推导 |
| DispatchTotal | 0.90~1.10 ms | 探索目标，不是承诺 |
| CombineTotal | 1.30~1.55 ms | 探索目标，不是承诺 |
| lane 长尾 | max ready - median ready 持续下降 | 判断倾斜/单 lane 拖尾 |
| CQE outstanding | 均匀快路径为0；倾斜慢路径 ≤1/lane/credit window | 逐 window Drain，不跨 epoch 累积 |
| single batch WQEBB | `<runtime_sq_depth` 且 `<=batch_buffer_bytes/64` | 另校验累计 outstanding 不越过 SQ2K |

目标带宽必须同时报告 p50、p95、p99 和各 rank 分布；只看 400 次 mean 可能掩盖某个 rank/channel 的尾部问题。

## 8. 推荐落地顺序

```text
P0 完成边界/实际负载打点
  -> P1.1 首批早提交
  -> P1.3 Combine 每 lane 独立发 flag
  -> P2.1 Dispatch channel-ready CountHits
  -> P2.3 Combine token-channel ready/reduce
  -> P2.2 Dispatch 预分配 offset + 单次扫描回搬
  -> P2.4 Combine K=6 reduction
  -> 按证据决定 P1.2 分核流水或 P3 持久化融合
```

优先级的原因是：早提交和移除 Combine 的 payload 后全核 flag barrier 改动较小；接收侧增量消费用于验证全量等待候选瓶颈；固定分核和持久化融合改动更大，应在知道各阶段真实时长后再做。

## 9. 依据与代码来源

### 9.1 当前仓代码

- `mc2/common/op_kernel/moe_shared_jetty_context.h:17-27`：当前把 56 个 Shared Jetty、world size=8、每远端 rank 8 条 lane 写成编译期常量；P-1 要移除后两者。
- `mc2/common/torch_extension/csrc/moe_shared_jetty_context.cpp:380-396`：每个 Jetty 建立多对端 MultiChannel handle。
- `mc2/common/torch_extension/csrc/moe_shared_jetty_context.cpp:205-214`、`mc2/common/torch_extension/elastic_buffer.py:32-36,420-435`、`mc2/common/torch_extension/csrc/elastic_buffer.cpp:105-110`：当前 Host/Python 的 8P 检查和派生 lane/flag 数，均需参数化。
- `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.h:79-85`：2 SGE/slot、12 SGE/WQE、batch 阈值和 order/CQE 配置。
- `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.h:1108-1172`：6 slot multi-SGE、按 128 个请求提交、末尾有序 flag。
- `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.h:1176-1188`：当前 Dispatch 控制流；只有最后一个 AIV 在 `GetRecvCount` 轮询。
- `mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.h:412-437`：全量等待后 CountHits/prefix/copy。
- `mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.h:452-485`：单核对全部通知求和等待并统一清零。
- `mc2/moe_ep_combine/op_kernel/moe_ep_combine.h:66-74`：256 WQEBB batch、payload relax order、最终 strong-order flag、均为 `cqe=0`。
- `mc2/moe_ep_combine/op_kernel/moe_ep_combine.h:275-336`：按实际 WQEBB 计数的 batch 提交。
- `mc2/moe_ep_combine/op_kernel/moe_ep_combine.h:521-562`：先发 payload、同步后发 channel flag、发送端不等待。
- `mc2/moe_ep_combine_epilogue/op_kernel/moe_ep_combine_epilogue.h:330-350`：等待全部完成 channel。
- `mc2/moe_ep_combine_epilogue/op_kernel/moe_ep_combine_epilogue.h:411-509`：本地 index 与逐 token TopK reduction。
- `mc2/moe_ep_combine_epilogue/op_kernel/moe_ep_combine_epilogue.h:532-568`：全局 ready 后统一 reduction。
- `mc2/common/op_host/op_tiling/moe_ep_window_layout.cpp:41-115` 与 `mc2/common/torch_extension/elastic_buffer.py:330-410`：注册 window 的 C++/Python 双份 size/offset 公式及 dump region。
- `mc2/common/op_kernel/moe_ep_exception_dump_defs.h:17,87`：当前 dump layout/version，协议布局变化时必须同步升级。
- `mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h:385-398,1817-1827`：按工作量拆分 AllToAll 与 CumSum 核，而不是固定一半核。

### 9.2 Hcomm/URMA 依据

以下路径对应 asc-comm commit `610bb2b9486d877c14aba62ac4ed55b2c3e21f9b`；接口/顺序语义以 Hcomm 正式文档和实现为准。

- `D:/code/asc-comm/docs/zh/programming_model/execution_model.md:144-186`：URMA relax/strong order 与 fence 语义；强序存在吞吐代价，应只约束有依赖任务。
- `D:/code/asc-comm/docs/zh/api/aicore/hcomm/BatchCommit.md`：BatchCommit 只提交批次，可以多次提交后统一处理完成。
- `D:/code/asc-comm/docs/zh/api/aicore/hcomm/Drain.md:62-66`：全 `cqe=0` 时 Drain 直接返回，不能据此证明请求完成；确认整批完成需要代表整批的 `cqe=1` 请求。
- `D:/code/asc-comm/src/aicore/hcomm/impl/platform_v310/hcomm_aiv_urma.h:937-983`：`preSqCnt` 必须非零且严格小于 SQ depth；BatchCommit 搬运 WQEBB、推进 SQ head、敲 doorbell 并清当前批计数。

### 9.3 优化方法来源

以下 cannbot 内容对应 commit `80bd6c22c85f830774369fc52ab447424921e8f3`，只作为分析方法和检查清单，不替代 Hcomm 接口或硬件规格依据。本机绝对路径用于保留当前审计证据；文档若发布到其他仓库，应换成该 commit 的稳定仓库链接。

- `D:/code/cannbot-skills/ops/ascendc-perf-optimize/SKILL.md:8-18,57-79`：先建模，再分别分析卡间、核间、单核流水；MC2 分核前需要实际 TilingData 和通信/计算隔离数据。
- `D:/code/cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/dispatch-reading-guide.md:63-70`：检查增量状态发布、ready 子集消费、全量轮询和 prefix 约束。
- `D:/code/cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md:86-102`：等待与聚合不必完全串行，应检查 ready 子集和回写串行点。
- `D:/code/cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/split-core-design.md:7-15,17-43`：按阶段独立分核；等待与回搬共用边界；保证不重不漏、无竞争、顺序可验证。
- `D:/code/cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/api-rules/sync-and-visibility.md:16-18,49-70`：数据先于状态、消费后清状态；`SyncAll` 只解决同卡跨核阶段对齐。
- `D:/code/cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/double-buffer-protocol.md:1-9,47-51`：跨轮重叠必须防止下一轮覆盖上一轮尚未消费的 buffer/state。
- `D:/Codex/home/plugins/cache/ascendc-agent-toolkit/ascendc-critical-path/0.1.0/skills/instrument-ascendc-critical-path/references/completion-boundary-audit.md:1-20,71-84`：异步函数返回不是完成边界，关键路径打点必须证明真实流水完成。

## 10. 验证边界

本文结论来自固定 SHA 的当前分支源码、Hcomm 实现/文档、本次对话给定的性能数据和静态通信量模型。尚未在 Ascend NPU 上采集 per-channel ready 时间、真实 `sendCnt`、实际 SQ/CQ depth/counter、跨 rank 对齐时钟或隔离测试数据，因此：

- 320~360 GB/s、Dispatch 0.90~1.10 ms、Combine 1.30~1.55 ms 都是实验目标，不是已验证结果；
- 分核比例、first/steady batch 阈值和 token chunk 必须通过 P0 数据选择；
- 典型均匀路径不需要热路径 Drain；SQ2K 下 Combine 极端倾斜需要本文定义的条件 checkpoint 慢路径或明确判为不支持，该机制仍需设备验证；
- 当前 flag clear 不是 ACK，P1/P2 依赖跨轮串行；跨轮重叠必须另加 ACK/epoch 或双缓冲；
- 当前没有修改 kernel、tiling 或 Host 接口，也没有执行 CANN 编译和多 rank NPU 测试。
