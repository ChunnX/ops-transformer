# MoE Combine 并行优化阶段性报告

## 1. 报告目的

当前优化的目标，是在 `MoeDistributeCombineV2` 的 `fullmesh_v2` 路径中，让发送和接收尽可能重叠，并缩短最后一个 token 完成的尾部时间。

目前最值得关注的现象是：**某个前序 token 的某一个 slot 长时间没有被接收端观察到，接收端无法按 slot 顺序继续累加；同时后续 token 已经被激活或等待，导致后续 token 的完成时间被拖长。**

这份报告汇总当前数据、已经尝试的方案、已确认和未确认的问题，便于进一步寻求调度和硬件流水方面的意见。

## 2. 当前测试场景

主要分析场景为：

| 项目 | 值 |
|---|---|
| batch size | 96 |
| hidden size | 7168 |
| top-k | 6 |
| 输入类型 | bf16 |
| 通信类型 | fp8_e4m3fn |
| 通信模式 | fullmesh_v2 |
| rank 数 | 8 |
| AIV 数 | 64（该组；其他配置测试过 72） |
| 接收核数 | 21（典型测试），也测试过 16/24/32 等配置 |
| 发送核数 | 43（接收核为 21 时） |
| 时间单位 | 1000 cycles = 1 us |

历史上原始实现的一个场景约为 29.7 us；当前并行方案在不同基线和调试开关下测得约 43～52 us。两组数字来自不同代码状态和测试配置，不能直接作为同一组 A/B 结论。后续比较必须固定 commit、shape、接收核数、debug 开关和性能模式。

本报告对应的代码实验链为：`3247a1855` 基线，之后依次尝试发送顺序/扫描、索引分桶、交错发送分配等方案；debug 记录使用 `eaab8f993` 的内容重放。当前工作分支曾以 `d7582546f` 作为交错发送实验，随后用 `221ba7562` 做单活跃 token 实验；这两个实验结果应分别比较，不应把不同实验叠加后归因于单一改动。

## 3. 当前实现的接收流程

接收核通过如下方式取得本核负责的 token：

```text
tokenIndex = newAivId + tokenLocalIdx * recvAivNum
```

因此每个接收核负责一个固定的 token 序列。对一个 token，接收端必须按照 slot 顺序处理：

```text
slot 0 -> slot 1 -> ... -> slot K-1
```

如果下一个 slot 的 flag 尚未观察到，`ProcessExpert` 会暂停该 token，不能跳过该 slot 继续累加。原实现同时维护两个 token buffer，在循环中交替处理两个 active token。一个 token 还没有完成时，后面的 token 不能进入已经占用的 buffer。

这里有两个容易混淆的时间概念：

1. `RECV_ROUND_BEGIN` 是接收核把 token 放入 active buffer 的时间，不是第一个 slot 的真实网络到达时间。
2. slot arrival 脚本记录的是接收核执行 flag 检查并观察到 slot 的时间，不一定是发送端真正完成写入的时间。

所以当前 debug 数据可以证明“接收端观察到 slot 很晚”，但不能单独证明“网络写入本身很晚”。要严格区分二者，需要把发送端 slot 完成时间和接收端首次观察时间配对。

## 4. 关键观测数据

### 4.1 发送任务均衡已明显改善，但端到端收益不明显

发送任务交错分配后，发送组统计从此前的：

```text
active=12~13/43, max=5
```

改善为：

```text
active=28~29/43, max=2
```

对于一个组内 48 个任务和 43 个发送核，理论最小最大负载是：

```text
ceil(48 / 43) = 2
```

因此 `max=2` 已经达到离散任务分配的理论下限。分析脚本仍显示 `UNBALANCED`，主要是因为它使用了连续负载阈值；该标记不能直接说明当前任务分配仍然存在严重问题。

已有无 debug 的对比数据如下：

| 方案 | 时间（us） |
|---|---:|
| 基线 | 50.8006 |
| 发送顺序优化 | 49.3869 |
| 接收专家拷贝流水优化 | 49.6368 |
| 发送索引分桶 | 51.6146 |
| 首 token flag 提前检查 | 51.7678 |
| 双 token flag 检查流水 | 51.2085 |

另一次固定场景测试中，交错发送分配为 48.16 us，原版本为 47.95 us。差值约 0.21 us，可能属于运行抖动，不能据此判定该方案有确定劣化。比较可靠的结论是：**交错分配达成了发送任务均衡目标，但目前尚未证明能够稳定改善端到端时间。**

### 4.2 接收循环启动并不晚，尾部和等待占主导

以 21 个接收核的 core summary 为例：

- 接收 format 主循环大约在 1.7～2.1 us 开始；
- 发送核大多在约 32～33 us 结束发送；
- 接收 format 结束约在 44～45 us；
- 最慢接收核的整体结束约在 47～48 us；
- 接收主循环持续约 35～44 us。

因此“接收核完全没有及时启动”不是完整解释。接收核实际上很早进入循环，但在循环中等待、检查和处理 slot 的时间很长，最后形成约 12 us 以上的尾部。

### 4.3 token 之间存在明显的长驻和空泡

典型 token 时间如下：

```text
core 47:
token  4: start  1.932 us, end  9.388 us, duration  7.456 us
token 25: start  2.037 us, end 27.197 us, duration 25.160 us
token 46: start 10.745 us, end 35.880 us, duration 25.135 us
token 67: start 28.203 us, end 40.748 us, duration 12.545 us
token 88: start 39.083 us, end 44.912 us, duration  5.829 us

core 58:
token 15: start  1.827 us, end 11.097 us, duration  9.270 us
token 36: start  1.920 us, end 38.697 us, duration 36.777 us
token 57: start 14.217 us, end 41.441 us, duration 27.224 us
token 78: start 39.983 us, end 45.225 us, duration  5.242 us

core 59:
token 16: start  2.243 us, end 22.283 us, duration 20.040 us
token 37: start  2.332 us, end 25.055 us, duration 22.723 us
token 58: start 23.491 us, end 34.207 us, duration 10.716 us
token 79: start 26.204 us, end 38.476 us, duration 12.272 us
```

slot 观察脚本还报告了以下大间隔：

```text
core 61, token 60, slot 1-4, gap 13.189 us
core 47, token 25, slot 0-4, gap 11.756 us
core 58, token 57, slot 0-4, gap 11.289 us
core 55, token 33, slot 0-5, gap 11.229 us
core 46, token 66, slot 1,   gap 10.821 us
```

这些结果与“某一个 slot 阻塞 token，后续 token 被拖住”的假设一致。但由于时间点是在接收端检查时记录的，仍需要结合发送端时间确认是通信延迟还是接收端检查不及时。

### 4.4 wave 统计显示尾部逐步堆积

21 个接收核的 wave 统计中，p50 观察时间大致从：

```text
wave 0: 约 7～10 us
wave 1: 约 15～22 us
wave 2: 约 25～36 us
wave 3: 约 32～39 us
wave 4: 约 38 us，max 约 42.3 us
```

这说明后续 wave 并非在前一 wave 完成后才开始，而是存在重叠；问题在于重叠过程中有些 token 的 slot 前缀推进不连续，产生较大的空泡和长尾。

## 5. 已尝试方案及结论

### 5.1 发送顺序和任务分配

尝试过按 expert slot、分桶、交错分核和不同扫描顺序发送。

- 交错分核可以把每组最大任务数降到 2，任务均衡指标明显改善。
- 分桶和额外的索引整理引入了 UB 搬运、步长访问或扫描开销，在部分场景反而变慢。
- `permStride` 或大范围打乱虽然可能降低局部热点，但会破坏 token/slot 的到达前沿，可能形成长尾，不能默认启用。

当前结论：发送均衡是必要条件，但不是当前唯一瓶颈；仅继续调整发送核负载，预计很难填补数 us 的差距。

### 5.2 双 token 与单 token 接收

原实现同时维护两个 active token。单 token 实验把 active buffer 数量限制为 1，但保留了原有 UB 分配，结果约 48.15 us，与原版本约 48.16 us 基本一致。

这说明：

- 双 token 过早激活确实会造成 token 长时间驻留和反复检查；
- 但单纯关闭第二个 token，并没有带来可观的端到端收益；
- 长 token duration 中相当一部分可能是等待远端 slot 到达，而不是纯粹的调度开销。

因此不能简单把“双 token”认定为主因。更合理的方向是：第二个 token 只有在有可处理的 slot 前缀时才激活，或者让两个 token 的检查顺序随就绪状态变化。

### 5.3 接收端专家拷贝和 flag 检查

尝试过专家拷贝流水、首 token flag 提前检查和双 token flag 检查流水。部分版本有约 1 us 以内的变化，但没有稳定达到目标。

当前 flag 检查路径大致包含：

```text
Duplicate flag buffer
DataCopyPad flag
CompareScalar
PipeBarrier / SyncFunc
向量结果转标量扫描
```

如果 slot 尚未就绪，每次轮询仍可能产生一组非零成本操作。两个远期 token 同时 active 时，这部分成本会被放大。

## 6. 当前问题的抽象

把一个通信任务表示为三元组：

```text
(source_rank, token_id, slot_id)
```

每个三元组有一个发送完成时间 `A(r,t,s)`。接收核 `p(t)` 负责 token `t`，每个 slot 的处理时间记为 `P(t,s)`。由于 token 内必须保持 slot 顺序，接收完成时间满足：

```text
C(t,0) >= A(t,0)
C(t,s) >= max(A(t,s), C(t,s-1) + P(t,s))
```

同一个接收核上还存在 token 间资源约束。若 token `t` 尚未完成，后续 token 可能不能占用其 buffer：

```text
Start(t_next) >= Finish(t_current)
```

因此整体目标不是简单地最小化发送时间，而是：

```text
最小化 max_p Finish(p)
```

同时满足：

1. 每个 token 内 slot 顺序不变；
2. batch 一致性和累加顺序不变；
3. 不引入全核同步；
4. 不超过 UB 和通信窗口容量；
5. 发送端不能集中访问同一目的 rank 形成热点。

这本质上是一个带有前缀依赖、有限 buffer 和通信资源约束的流水调度问题。

## 7. 当前最可能的根因

目前最符合数据的解释是：

1. 发送端整体 span 约 34～35 us，并不是特别长；
2. 接收端约 2 us 已进入主循环，启动延迟不是主要问题；
3. 某些 token 的一个 slot 在接收端很晚才被观察到；
4. token 内 slot 顺序不能跳过，导致该 token 长时间阻塞；
5. active buffer 被阻塞 token 占据，后续 token 即使部分数据已到，也不能及时完成；
6. 最终时间由最后一批被阻塞 token 的 slot、累加和搬出共同决定。

需要强调：目前还不能确认第 3 点是“发送完成晚”还是“接收端轮询晚”。现有 slot arrival 数据属于 observed arrival，不能直接作为物理到达时间。

## 8. 建议下一步验证

### 8.1 配对发送和接收时间

对每个三元组 `(source_rank, token_id, slot_id)` 记录：

```text
send_begin
send_end
receiver_first_observed
receiver_process_begin
receiver_process_end
```

然后计算：

```text
receiver_first_observed - send_end
receiver_process_begin - receiver_first_observed
```

前者大说明通信/flag 发布延迟，后者大说明接收端轮询或调度不及时。必须优先区分这两个问题。

### 8.2 做一个不改通信协议的就绪调度实验

保持现有 token 分配和 slot 累加顺序，只改变 active token 的选择：

- 当前 token 的下一个 slot 未就绪时，不要无条件激活一个更远的 token；
- 第二个 buffer 只在其第一个有效通信 slot 已经可观察时激活；
- 不做全 token 扫描，不加全核同步，不增加 atomic；
- mask、共享专家和特殊专家场景必须找到“第一个有效通信 slot”，不能硬编码 slot0。

这是验证“远期 token 过早占用 buffer”假设的最小改动。

### 8.3 发送顺序只围绕 slot 前沿优化

如果发送端确实存在到达抖动，优先评估：

```text
(token_group, slot_id, token_id)
```

即组内 slot-major，组间 token group 递增。不要使用无约束的 `permStride` 打乱全局 token 顺序，除非数据证明它能缩短最后一个 token 的最后 slot 到达时间。

### 8.4 评估指标

每个版本至少记录：

- 无 debug 的多轮性能均值、p50、p90、p99；
- 发送 global span；
- 每个接收核的 format duration；
- 每个 token 的最后 slot observed 时间；
- 最大 slot gap；
- `send_end -> receiver_first_observed` 的分布；
- 最后一个 token 的累加和搬出时间。

单轮结果和带 debug 结果只能用于定位，不能用于判断数百 ns 级优化是否成立。

## 9. 当前阶段结论

当前没有证据表明发送核数量不均衡是唯一主因。发送交错分配已经把组内最大任务负载降到理论下限，但端到端性能变化仍在抖动范围内。

更需要解决的是接收端的前缀依赖：一个 token 的迟到 slot 会阻塞该 token，两个有限 buffer 又会把这种阻塞传播到后续 token，形成明显的空泡和尾部。

下一阶段应优先取得真实的“发送完成到接收首次观察”配对数据，再决定是优化发送发布顺序、接收轮询方式，还是改变 token buffer 的就绪调度。仅凭当前 observed slot 时间，不能把问题简单归因于发送慢或接收启动慢。
