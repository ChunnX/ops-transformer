# MegaMoe 256 行动态 Wave 调度设计

## 1. 状态与范围

- 设计日期：2026-08-03
- 主线基线：`upstream/master@27455755ec4fb4aac40cb6ff3d8ee962139ad5d3`
- 设计分支：`codex/mega-moe-wave-dynamic-scheduler-20260803`
- 独立 worktree：`/home/mace/workspace/ops-transformer_megaMoeOpt_wave_sched_20260803`
- 目标路径：Arch35、MTE、`MegaMoeWave` 路由专家路径
- 当前状态：V1 Kernel 已实现；Ascend 950 的 8 个 MegaMoe Kernel 变体编译通过；尚未做 NPU 实测

当前 V1 已改为真正的 256-row segment 调度：热点专家允许跨 Wave，Host 预测的固定专家数
不再参与 Wave 收口。本文后半部分仍保留 stride、owner table 和尾 Wave DP 等进一步探索；
当前实现的精确开发说明以 `mega_moe_dynamic_wave_v1_development_design.md` 为准。

## 2. 当前 V1 实现

### 2.1 固定 chunk 公式

定义：

```text
C = GetBlockNum()                                      // 真实物理 AIC 数
P1 = ceil(schedulerN / 256)                            // 每个 chunk 的 GMM1 owner 数
schedulerN = interleaved ? hiddenDim : hiddenDim / 2
P2 = ceil(h / 256)                                     // 每个 chunk 的 GMM2 owner 数
Q1 = ceil(2 * C / P1)                                  // 每核至少两个 GMM1 owner
Q2 = ceil(C / P2)                                      // 每核至少一个 GMM2 owner
Q = max(Q1, Q2)                                        // 正常 Wave 固定 chunk 数
```

每个专家逻辑拆成 expert-local `M<=256` chunk，正常 Wave 固定消费 `Q` 个 chunk。Wave 终点
使用 `(expertIdx, rowOffsetInExpert)` 表示，因此可以位于热点专家内部。同一专家、同一 Wave 的连续
chunk 会合并成一个 segment GMM，降低 problem 初始化开销。

Host 仍为 ABI 兼容下发 `expertsPerBatch`，但当前 Wave Kernel 不再使用它决定边界。最后剩余
chunk 少于 `Q` 时直接执行 partial Wave。

### 2.2 三角色执行

- AIV1 在 `SendCntCal` 得到真实 count 后按 `Q` 推进 segment cursor，并完整预取下一 Wave；
- AIC/AIV0 在 `UpdateGroupParams<GMM1>` 读到同一 count 后流式执行同一 segment 范围；
- GMM2 重放相同的 `(beginCursor, endCursor)`；
- 三者共用 `GetWaveSegmentRowCount`，不增加边界 workspace 或批次 barrier；
- Combine 仍以完整专家为粒度，只消费已经完成最后一个 GMM2 segment 的专家。

### 2.3 明确不变项

- 热点专家允许切分，但不同专家不能拼成同一个 GMM；
- segment 同时平移激活、scale、meta、输出和 per-256 ready flag，权重仍只按专家选择；
- TopkWeightsPrefetch status 使用 expert-local 全局 tile offset，避免跨 segment 命中旧状态；
- 不改现有 `startBlockIdx_` 的朴素滚动分核和核角色重复修正；
- 不改 GMM1 -> GMM2 -> Combine 的阶段顺序；
- 不增加 workspace、tiling 字段或 Host/Kernel ABI。

实现位置：

- `mega_moe_wave.h::DispatchWaveAndGetEndCursor`
- `mega_moe_wave.h::BuildExpertSegmentState`
- `mega_moe_wave.h::ProcessGmm1Wave`
- `mega_moe_wave.h::ProcessGmm2Wave`
- `mega_moe_wave.h::ProcessRoutedExpertWaves`
- `mega_moe_utils.h::GetMGroupsPerWave/GetWaveSegmentRowCount`
- `mega_moe_impl.h` 的 GMM1 tile-status segment offset

与 V1 完全一致的机器可执行模型见 `mega_moe_dynamic_wave_v1_reference.py`。下文保留更完整的
加权分核探索；其中 stride 搜索、owner table、物理 GMM1/GMM2 独立 cursor 和尾部 DP 仍未实现。

## 3. 不能完全删除的专家边界

“弱化专家粒度”不等于把不同专家的行拼成一次 GEMM。

例如两个专家各有 128 token：

```text
expert 0: 128 rows, weight W0
expert 1: 128 rows, weight W1
```

虽然总数是 256，但它们不能成为一次 `256 x K @ K x N`，因为权重不同。实际仍是两个
128-row problem，只是它们的任务可以在同一 Wave、不同核上并行执行。

因此必须区分：

```text
X = Wave 内有效 token 行数之和
Q = Wave 内 expert-local chunk 数
```

仅当所有 chunk 都恰好为 256 行时：

```text
X = 256 * Q
```

冷专家很多时通常 `Q > ceil(X / 256)`。最优 planner 必须看真实 chunk/task，不能只数
`X/256`。

## 4. 基线代码为何空转

基线 `ProcessRoutedExpertWaves` 使用 Host 预测的固定 `expertsPerBatch`：

```text
GMM1(expert batch) -> GMM2(expert batch) -> next batch
```

每个 expert problem 内的 owner 通过：

```text
loopIdx = localStart; loopIdx < tileNum; loopIdx += blockNum
startBlockIdx = (startBlockIdx + tileNum) % blockNum
```

滚动到物理 AIC。这个 modulo 机制本身能跨 expert 延续核起点；问题是 Wave 仍由预测的
完整专家数截断。当多个冷专家产生的累计 GMM1 owner 太少时：

- 少数核执行 GMM1；
- 其余核很快进入 GMM2；
- GMM2 owner 在 SwiGLU-ready flag 上等待；
- 明明后面还有 GMM1 工作，却因为 Wave 截断没有提前铺开。

现有 Host 公式还只按平均 token 估计，无法看到本卡真实专家尾 M。

## 5. 与当前 kernel 一致的任务模型

定义：

```text
C  = blockNum_                         // AIC 数
h  = k_                                // GMM1 K、GMM2 N
2n = hiddenDim_                        // GMM1 gate+up 总 N
n  = hiddenDim_ / 2                    // SwiGLU 输出、GMM2 K
TM = TN = 256
m_i = 第 i 个 expert-local chunk 的有效行数，1..256
```

### 5.1 non-interleaved GMM1

当前默认路径：

```text
schedulerN = n
P1_non = ceil(n / 256)
```

第 `j` 个 owner 的 N 宽度为 `w1[j] <= 256`。一个 owner 固定在同一 AIC 上依次执行 gate
和 up 两次 `BlockMMad`：

```text
Task1_non(i,j) = (chunk i, N tile j, gate+up pair)
cost1_non(i,j) ~= 2 * m_i * w1[j] * h
```

gate/up 两次调用不是两个可独立分给不同核的 owner。若强行拆核，会破坏当前直出 UB 与
配对 AIV0 的 SwiGLU 路径，需要另一套跨核中间结果协议，本方案不采用。

### 5.2 interleaved GMM1

interleaved 路径：

```text
schedulerN = 2n
P1_int = ceil(2n / 256)
```

每个 owner 一次 MMAD：

```text
Task1_int(i,j) = (chunk i, interleaved N tile j)
cost1_int(i,j) ~= m_i * w1[j] * h
```

两种布局的总理论 GMM1 工作相同，但 interleaved 把工作拆成约两倍 owner，粒度更细，通常
需要更少 chunk 就能铺满全部 AIC。

### 5.3 GMM2

GMM2：

```text
P2 = ceil(h / 256)
Task2(i,j) = (chunk i, output-H tile j)
cost2(i,j) ~= m_i * w2[j] * n
```

### 5.4 h/n 的真正影响

在纯 MAC 模型中，无论 `h/n` 比例以及 interleaved 与否：

```text
W1(chunk) = 2 * m * h * n
W2(chunk) =     m * h * n
W1 : W2 = 2 : 1
```

所以 `h/n` 不改变两级总计算量约 2:1 的事实，但会改变：

- `P1/P2`，即单个 chunk 有多少独立 owner；
- N 尾 tile 的宽度；
- 一个 Wave 至少需要多少 chunk 才能铺满核；
- interleaved 是否能显著缩短 Wave；
- 每个 BlockMmad 的 K 循环长度和实际效率。

这正是不能只按 GMM1/GMM2 tile 总数或固定核数拍方案的原因。

## 6. 优化目标与可证明下界

对某一阶段 `s`，给每个 Cube task 一个预测成本 `w_t`。任务分配后，核 `c` 的负载：

```text
L_s[c] = sum(w_t | owner(t) = c)
T_s = max_c L_s[c]
```

当前设计是两阶段 Wave，因此预测 Cube 利用率为：

```text
U = (sum_t W1_t + sum_t W2_t) / (C * (T1 + T2))
```

每个阶段不可突破的下界：

```text
LB_s = max(
    ceil(sum_t w_t / C),
    max_t w_t
)
```

若 `T_s = LB_s`，该阶段已达到任务不可再拆条件下的最优 makespan。满 256、所有 N tile
等宽且总 owner 数可被 C 整除时，cyclic 映射可以让每核任务数完全一致，因此可证明达到
下界。

任意真实尾 M 的加权多机分配本质上是 `P||Cmax`，一般情形没有简单闭式精确解。设备侧
实用方案采用 LPT/最轻核分配，并由 Scalar 参考模型报告相对上述下界的 gap；不能把启发式
结果宣称成数学全局最优。

## 7. 每个 256-row 组如何分核

### 7.1 规则 chunk 的无表快路径

GMM1/GMM2 各自维护独立 cursor 和 chunk stride：

```text
base_s(i)  = (cursor_s + i * stride_s) % C
owner_s(i,j) = (base_s(i) + j) % C
```

其中：

- `i` 是当前 Wave 内 chunk 编号；
- `j` 是该 chunk 内阶段 s 的 N tile 编号；
- `stride_1` 与 `stride_2` 独立；
- Kernel 初始化时枚举 `0..C-1`，选择预测 `T_s` 最小的 stride；
- 完全相同时优先保留当前的 `stride=P_s`，避免无收益改动。

这比固定“每个 chunk 使用某几个核”更灵活：当 `P_s<C` 时，第一个 chunk 只占用部分核，
但第二个 chunk 的 base 会旋转到其他核。chunk 之间没有 barrier，所以多个 chunk 会自然
交错并行。

### 7.2 冷专家尾 M/N 尾 tile 的加权路径

若 cyclic 最佳候选仍达不到利用率目标，Scalar 为每级生成 owner table：

1. 建立 `(chunk, nTile, predictedCost)` 任务；
2. 按成本从大到小排序；
3. 每次把任务分给当前累计成本最小的核；
4. 平局按上一 Wave cursor 轮转，避免长期固定低编号核；
5. 分配完成后，每核执行队列重新按 chunk 编号排序。

第 5 步不改变每核总负载，但保证：

- GMM1 的前部 256 行尽早完成；
- GMM2 每核先消费更老、通常更早 ready 的 chunk；
- 不会因为 LPT 把某核的第一项排成后部未 ready chunk 而空等。

owner table 只保存逻辑任务的 owner 或每核 task range，不改变数据 ABI。它是冷专家尾部
负载真正不等时的慢速规划路径；全满 chunk 仍走简单 cyclic。

### 7.3 为什么不固定每组 4 核

若 GMM2 每 chunk 有 16 个 tile，固定 4 核意味着每核串行做 4 个。只有当另外 20 核能
持续执行等服务率的 GMM1 时才可能合理。纯 MAC 模型中两级工作量为 2:1，24 核的稳态
空间流水应约为：

```text
GMM1: 16 cores
GMM2:  8 cores
```

而不是 20:4。固定 4 核会让 GMM2 成为瓶颈，最终 producer 反过来等待 consumer。

本方案第一版不做固定空间核池。GMM2 使用全部 AIC，并让不同 chunk 的 tile 在全局任务流
中滚动；它能在不预测跨阶段相对耗时的情况下，让每一级分别达到最优或近最优负载。

## 8. Wave 大小怎么求

### 8.1 满 256、等成本的闭式候选

令：

```text
P1 = non-interleaved ? ceil(n/256) : ceil(2n/256)
P2 = ceil(h/256)
D1 = 2                    // 每核目标 GMM1 owner 深度
```

最小深度：

```text
Qmin = max(
    ceil(D1 * C / P1),
    ceil(C / P2)
)
```

两级 owner 数同时整除 C 的周期：

```text
Q1align = C / gcd(C, P1)
Q2align = C / gcd(C, P2)
Qalign  = lcm(Q1align, Q2align)
Qexact  = ceil(Qmin / Qalign) * Qalign
```

当 N tile 等宽时，`Qexact` 是满足 producer 深度后、用当前连续 owner 模型让两级都没有
modulo 尾核的最小闭式候选。

### 8.2 一般真实 count 的最优实用搜索

真实 planner 不直接使用 `Qexact`，而是从当前 chunk 起逐个扩展候选 Wave：

```text
for q in [Qmin, ..., Qcap]:
    rows = next q actual expert-local chunks
    evaluate best cyclic stride for GMM1/GMM2
    if needed, evaluate weighted owner-table mapping
    calculate T1, T2, U and lower-bound gap
```

选择采用词典序目标：

1. 满足 GMM1 producer 深度；
2. 达到指定 Cube 利用率目标；
3. 在满足前两项的候选中选择最小 X/最小 Q，保留流水启动速度；
4. 若剩余总工作不可能达标，选择预测 makespan 最小者。

如果目标是绝对最大 Cube 利用率，利用率阈值设为 100%；工程上可把 98% 作为低延迟
Pareto 点，再用 trace 在 98% 与 100% 之间选择。没有延迟/最大 Wave 约束时，把所有 token
放进一个 Wave 会成为无意义的“最优”，因此必须采用“达到目标后的最小 Wave”定义。

对满 chunk，搜索到 C 个 chunk 已足以让任一固定 tile 位置轮转遍历所有核；因此常规
`Qcap=C` 即可覆盖 100% cyclic 周期。实际总 chunk 少于 Q 时，只能接受总工作不足的下界。

### 8.3 整个 case 的尾 Wave

不能简单重复 `Q*` 后把余数单独做成一个很小的最后 Wave。已知全部专家 count 后，对 chunk
序列做有界动态规划：

```text
dp[end] = min over q:
    dp[end-q] + predictedMakespan(chunks[end-q:end])
```

约束：

- 正常 Wave 至少为 `Qmin`；
- 最大 Wave 建议不超过 `2*Qselected-1`；
- 最后一小段允许并入前一 Wave；
- 主目标最小总预测 makespan，平局选择 Wave 更少者。

这样可避免例如 `12 + 12 + 1` 的极小尾 Wave，改成更合理的相邻大小组合。

## 9. h/n 与 interleaved 的例子

以下均为 `C=24`、满 256 chunk、`D1=2`、纯 MAC/等 tile 模型：

| h | n | GMM1 模式 | P1 | P2 | Qmin | 最小 100% Q | Wave token X |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4096 | 1536 | non-interleaved | 6 | 16 | 8 | 12 | 3072 |
| 4096 | 1536 | interleaved | 12 | 16 | 4 | 6 | 1536 |
| 6144 | 1536 | non-interleaved | 6 | 24 | 8 | 8 | 2048 |
| 6144 | 1536 | interleaved | 12 | 24 | 4 | 4 | 1024 |
| 4096 | 4096 | non-interleaved | 16 | 16 | 3 | 3 | 768 |
| 4096 | 4096 | interleaved | 32 | 16 | 2 | 3 | 768 |
| 7168 | 2048 | non-interleaved | 8 | 28 | 6 | 6 | 1536 |
| 7168 | 2048 | interleaved | 16 | 28 | 3 | 6 | 1536 |

结论：

- interleaved 经常因 GMM1 owner 更细而减小 Q；
- 当 GMM2 的整除周期成为主导时，interleaved 不一定减小最终 Q；
- `h/n` 改变的是并行颗粒和 Q，而不是纯 MAC 下的 2:1 总工作量。

### 9.1 16 个 GMM2 tile 的具体解释

`C=24、P1_non=6、P2=16`：

```text
Q=8:
  GMM1 = 48 owners -> 每核 2
  GMM2 = 128 tiles -> 每核 5 或 6
  理论总利用率约 96%

Q=12:
  GMM1 = 72 owners -> 每核 3
  GMM2 = 192 tiles -> 每核 8
  两级 100% 计数均衡
```

第一个 chunk 的 16 个 GMM2 tile 可以先落到 16 个核；另外 8 个核直接取得第二 chunk
任务。三个 chunk 已有 48 个 GMM2 tile，24 核各做两个。正确做法是扩大扁平 ready 窗口，
不是把 16 个 tile 压到 4 核。

### 9.2 h/n 非 256 对齐

`C=24、h=4128、n=1568` 时，N 尾 tile 使单纯 owner 计数整除不再等于成本整除。参考模型
得到的 Pareto 点：

```text
non-interleaved:
  Q=7   -> 90.85%
  Q=8   -> 93.81%
  Q=12  -> 98.41%
  Q=24  -> 100%

interleaved:
  Q=4   -> 90.35%
  Q=6   -> 95.37%
  Q=12  -> 98.41%
  Q=18  -> 99.46%
  Q=24  -> 100%
```

这说明 100% 目标可能把 Wave 拉到 6144 token；若更重视启动延迟，Q=12 是明显更好的
Pareto 点。最终阈值需要 NPU trace 校准，而不是由静态模型武断决定。

## 10. GMM1/GMM2 的执行顺序

一个 Wave 的 Cube 侧顺序：

```text
GMM1 weighted task queues for Wave k
    -> AIV0 SwiGLU per chunk
    -> GMM2 weighted task queues for Wave k
    -> next Wave
```

但不存在每个 chunk 的全核 barrier：

- 各核按自己的 GMM1 owner 队列执行；
- 前部 chunk 的任务在每核队列中优先；
- GMM2 仍用现有每 256 行 ready flag 做正确性等待；
- 各核 GMM2 队列按 chunk 顺序，先消费最老 ready chunk；
- `D1=2`/负载阈值负责提供 ready 距离，不替代 flag。

这实现的是 Wave 级 producer-depth，而不是“每个专家 GMM1 完成就立刻切 GMM2”。

### 10.1 是否做空间流水核池

理论上可采用弹性三段式：

```text
fill:   全核 GMM1
steady: 约 2/3 核 GMM1 + 1/3 核 GMM2
drain:  全核 GMM2
```

它与满载的全核两阶段 Wave 有相同理想吞吐下界，但可更早启动首个 GMM2/Combine。代价是：

- 通用 scheduler 必须支持 `logicalCoreCount/localCoreIdx`；
- AIC/AIV0 配对角色、两级 cursor 和 ready 到达数都要重构；
- 某一级因 Dispatch/AIV0 抖动停顿时，固定 pool 不能自动借核；
- 频繁换角色会引入事件排空和 Scalar 控制成本。

因此本轮以“最大化 Cube 利用率”为目标时，全核两阶段加权 Wave 已能达到相同计算下界，
第一版不引入空间核池。若后续 trace 显示首结果延迟而非 Cube 利用率成为主矛盾，再设计弹性
pool，且只能按工作量枚举，不能固定 4 核。

## 11. Runtime planner 的输入与输出

### 11.1 输入

```text
C, h, n, interleaved
sendCount[e] for every local expert
GMM tile constants and calibrated task-cost table
producer-depth / utilization target / Q cap
```

`bs` 只能给出容量和总路由行数上界，不能描述专家分布；真正的 Wave 规划必须使用
`sendCount[e]`。

### 11.2 chunk/segment

```text
Chunk {
    expertId;
    expertRowBegin;
    globalRowBegin;
    validM;              // 1..256
}

Segment {
    expertId;
    firstChunk;
    chunkCount;
    expertRowBegin;
    globalRowBegin;
    validRows;
    isExpertFinal;
}
```

同一专家、同一 Wave 内相邻 chunk 应合并为 segment，减少 problem 初始化；owner 规划仍按
逻辑 chunk/task 计算。

### 11.3 输出

```text
WaveDesc {
    firstChunk;
    chunkCount;
    validTokenCount X;
    gmm1Policy / stride / optional owner table;
    gmm2Policy / stride / optional owner table;
    predictedT1 / predictedT2 / utilization / lowerBoundGap;
}
```

这些是内部调度描述，不改变对外输出 ABI。

## 12. 地址与同步不变量

### 12.0 源码范围约束

当前通用 GMM 实现直接设置：

```text
config.blockNum = GetBlockNum()
config.blockIdx = GetBlockIdx() / GetTaskRation()
loopIdx += config.blockNum
startBlockIdx = (startBlockIdx + tileNum) % config.blockNum
```

因此只修改 `wave.h` 可以做到：

- 根据真实 count 改 Wave/segment 边界；
- 继续使用当前 `stride=P_s` 的全核 cyclic；
- 通过选择 Q 让规则满 chunk case 达到精确均衡。

但只在 `wave.h` 屏蔽部分核或声明 owner table 会漏算原本属于其他物理 `blockIdx` 的 tile，
不能正确实现任意 `stride_s` 或加权 owner table。

若以后要求完整落地本文最优路径，需要给 `mega_moe_impl.h` 的内部 scheduler 增加 Wave 专用的
逻辑 owner 输入，例如：

```text
logicalCoreCount / logicalCoreIdx
或 taskOwner/taskList
```

该扩展只改变 Wave 内部任务归属，不要求改变算子外部 ABI，也不应影响 Layered/Shared 等其他
调用方。若源码范围被严格限制为 `wave.h` 单文件，则只能落地本文 cyclic 快路径；加权尾部
方案必须明确标记为未实现，不能宣称已经达到真实尾 M 的最优分核。

### 12.1 Dispatch

- 物理 buffer 保持 expert-major，不需要重排 token；
- Dispatch 仍按专家扫描 source rank/mask，避免重复扫描；
- per-256 Dispatch-ready flag 允许 GMM1 提前消费专家前部 chunk；
- Wave planner 只改变逻辑消费窗口。

### 12.2 segment 地址

每个 segment 必须同时平移：

- dispatch data/scale/meta；
- GMM1 输出与 SwiGLU data/scale；
- GMM2 输出；
- Dispatch/SwiGLU flag 的 expert-local chunk offset；
- `expertBeforeCnt/globalRowBegin`。

Weight1/Weight2 只按 `expertId` 选择，不随 segment 行偏移。

### 12.3 GMM2 -> Combine

第一版 Combine 保持完整专家粒度：

- 只有某专家最后一个 GMM2 segment 完成后才发布 expert ready；
- AIV1 仍等待整个专家，再按原始 meta 路由；
- 不改变 y、scale、meta 或远端输出顺序。

这会放弃大专家前部 segment 的早 Combine，但不影响本轮 Cube 利用率目标。

### 12.4 TopkWeightsPrefetch

当前 status 使用：

```text
slot = expertIdx * maxTilesPerExpert + localLoopIdx
tag  = expertIdx + 1
```

同一专家跨 segment 后 `localLoopIdx` 会重新从 0 开始，可能命中陈旧状态。实现时必须改成
expert-local 全局 tile offset，或让 tag 包含 segment generation；不能依赖 kernel 起始清零。

## 13. 后续物理 GMM1/GMM2 cursor 解耦（V1 未实现）

当前单一 `startBlockIdx_` 同时累计两级 tile，并需要 fixed-role resonance 修正。新设计使用：

```text
gmm1Cursor
gmm2Cursor
```

- AIC 与配对 AIV0 共同使用 GMM1 owner/queue；
- GMM2 只有 AIC 使用自己的 owner/queue；
- AIV0 不再为了下一轮 GMM1 而虚拟推进 GMM2 cursor；
- 两级分别验证长期核负载，不让 `h/n` 的余数互相污染下一 Wave 起点。

## 14. Scalar 参考模型

当前 V1 的精确边界模型：

```text
mc2/mega_moe/docs/mega_moe_dynamic_wave_v1_reference.py
```

它逐项复刻 expert-local chunk、固定 `Q`、热点专家 segment cursor、GMM1/GMM2 N-tile 数和
GMM2 范围重放，并验证 AIV1 与 AIC/AIV0 的确定性边界。完整加权分核方案的探索模型为：

```text
mc2/mega_moe/docs/mega_moe_wave_scheduler_reference.py
```

能力：

- non-interleaved/interleaved 精确 owner 公式；
- h/n 的 256 尾 tile；
- 真实 expert token count 与 expert-local 尾 M；
- 每级全部 cyclic stride 穷举；
- chunk-order 最轻核与 LPT owner table；
- 阶段下界、gap、预测利用率；
- Q Pareto frontier；
- 固定核池工作量对比；
- 有界动态规划选择整个 case 的 Wave 划分。

示例：

```bash
python3 mc2/mega_moe/docs/mega_moe_wave_scheduler_reference.py \
  --cores 24 --h 4096 --n 1536 \
  --mode non_interleaved --chunks 24

python3 mc2/mega_moe/docs/mega_moe_wave_scheduler_reference.py \
  --cores 24 --h 4096 --n 1536 \
  --mode interleaved \
  --expert-tokens 32,511,1,256,700,0,255,257
```

已完成的设计级验证：

- 闭式 `Qexact` 对 262,144 组 `C/P1/P2` 组合检查整除、深度和每核计数；
- 最终参考模型随机检查 200 组不同 C、h、n、M 尾和两种布局；
- 验证 interleaved/non-interleaved 总理论 GMM1 工作一致；
- 验证搜索策略不劣于当前 `stride=P`；
- 验证任务总工作无重无漏。

这些仅是 Scalar 数学/模型验证，不是 Ascend 编译、NPU 正确性或性能成功。

## 15. 验证状态与性能判据

当前分支已完成：

- `bash build.sh --opkernel --ops=mega_moe --soc=ascend950 -j8`，8 个 Kernel 变体编译成功；
- V1 模型 56,160 组穷举配置和 20,000 组随机配置通过；
- 覆盖 0/255/256/257 token、热点专家跨 Wave、不同核数、两级 N-tile 数和多种 depth；
- `git diff --check` 通过；
- 本环境没有 `npu-smi`，因此未执行 NPU 正确性、trace 或性能回归，不能把上述静态/编译检查
  当作设备运行成功。

服务器验证时，trace 必须证明：

1. NPU 输出与基线在全部支持的量化/layout 模式下一致；
2. AIV1、AIC、AIV0 实际采用的 `(expertIdx, rowOffsetInExpert)` 终点与 Scalar 模型一致；
3. 非尾 Wave 的 GMM1 owner 数不少于 AIC 数的两倍，GMM2 owner 数不少于 AIC 数；
4. GMM2 等待 SwiGLU-ready 的时间显著下降；
5. interleaved/non-interleaved 分别使用正确的 GMM1 owner 数；
6. 冷专家、均衡专家、单热点专家、冷热混合和全零尾部均无死锁或越界；
7. 热点/均衡 case 的 segment 初始化开销和 L2 顺序变化没有造成不可接受回归；
8. 最后 partial Wave 未达目标时只包含不可避免的剩余工作；
9. 多 rank 运行的输出 ABI、count 输出、Combine 和 Unpermute 行为保持不变。

应使用 trace 反标定：

```text
cost(M,N,K,mode,layout)
```

再替换纯 MAC 权重。静态公式不能宣称已经解决设备上的所有等待。

## 16. 当前决策与后续增强

建议敲定：

1. **已实现**：Wave 边界不再是专家边界；基本描述是 expert-local `M<=256` chunk。
2. 不把不同专家行错误合并成一个 GEMM；X 与 Q 分开记录。
3. 分核依据是加权 Cube task，不是专家数、token 总数或 tile 数中的单一指标。
4. 规则 case 用搜索 stride 的 cyclic 快路径；真实尾部必要时用 owner table。
5. 后续：GMM1/GMM2 独立物理映射、独立 `startBlockIdx`，均跨 chunk 全局滚动。
6. Wave Q/X 由真实 count 的负载搜索得到；闭式 Q 只作满 chunk 快路径候选。
7. 绝对利用率目标选最小 100% Q；兼顾启动延迟时选最小达到 98% 的 Pareto Q。
8. GMM2 不固定 4 核；第一版不做永久空间核池。
9. 每核执行队列按 chunk 顺序，保证前 256 行和最老 ready chunk 优先。
10. 整个 case 用尾部重分配/DP，避免极小最终 Wave。

这是在现有硬件任务不可再拆、同理论计算量同耗时假设下，既能证明规则 Wave 最优、又能
适配真实冷专家尾部的方案。设备上的最终 Q 和成本权重仍必须由 NPU trace 校准。
