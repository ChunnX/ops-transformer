# arch35 ChunkGatedDeltaRule 融合优化分析

本文对比 `76bcecade`（`add fast cgdr`）与其父提交 `1bdece856`。优化集中在 Ascend 950 的 arch35 实现，arch22 仍使用原有分阶段路径。下文从代码解释性能收益来源；用户提供的 3～5 倍加速结果未在本次注释任务中重新测量，各优化点的收益比例仍需通过设备 profiling 和消融实验确认。

## 1. 从三阶段调度改为按 head 持续递推

旧版 arch35 在每个 batch 的每个 chunk group 中依次执行 Stage1、Stage2、Stage3，每个阶段结束后执行 `SyncAll<false>()`，并在阶段间重置 `TPipe`。Stage1 构造分数、逆矩阵和变换后的 Q/K/V，Stage2 沿 chunk 递推状态、计算跨 chunk 的输出贡献，Stage3 计算 chunk 内输出贡献。阶段中间量、掩码和递推状态需要通过 GM 传递。

新版调用链为：

```text
chunk_gated_delta_rule_apt.cpp
  -> CGDR::Process：校验输入
  -> FusedStage123Exact::Process
       -> ProcessAic / ProcessAiv
            -> workId = batchId * Nv + valueHeadId
            -> Cube 融合路径，或 Streaming 批搬运/逐 token 递推路径
```

每个工作项由一个 AIC 和两个 AIV 共同拥有，工作项之间按 `aiCoreNum` 轮转分配。同一个 head 的全部 chunk 留在同一核组内，状态只在序列开始时读入、结束时写回；相邻 chunk 间通过片上状态完成递推。两个 AIV 在预处理阶段各负责 32 个 token，在状态阶段各负责最多 64 个 Dv 行。

这移除了 chunk group 内的三次全核屏障，保留内核入口和结束时的汇合，以及必要的核组内生产/消费同步。`GetWorkspaceSize()` 对 ASCEND950 仅申请 16 MiB 系统工作区，不再申请 `interWorkspaceSz` 和 `stageWorkspaceSz`。这不代表输入输出不访问 GM，也不代表所有输入都只读取一次：AIC 和 AIV 仍分别加载各自需要的 Q/K。

## 2. 数学公式与代码变量

以一个 chunk 为例，令 Q/K 为 `[C,Dk]`，V 为 `[C,Dv]`，进入 chunk 时的状态 S 为 `[Dv,Dk]`，其中 C=64。以下公式忽略中间 BF16 舍入；`G_i=sum(g_0,...,g_i)` 是 chunk 内累积门控，无 g 时 G 全零。

```text
L_ij = beta_i * (K_i · K_j) * exp(G_i-G_j), j<i；其余为 0
A = (I+L)^(-1)

rawKcd_i = -beta_i * exp(G_i) * K_i
vBeta_i  = beta_i * V_i
qPrime_i = scale * exp(G_i) * Q_i
kg_i     = exp(G_last-G_i) * K_i
kcd      = A * rawKcd

Vnew     = A * vBeta + kcd * S^T
S_next   = exp(G_last) * S + Vnew^T * kg
P_ij     = scale * (Q_i · K_j) * exp(G_i-G_j), j<=i；其余为 0
O        = qPrime * S^T + P * Vnew
```

`rawKcd` 预先吸收负号，因此 `RunStatePair()` 可以将残差项直接累加到保存 `A*vBeta` 的 L0C，不需要额外的减法中间张量。这里的 S 布局与部分 Python golden 内部的 `[Dk,Dv]` 布局相反，阅读矩阵乘时应同时核对转置方向。

`BuildAttnMaskedQktExactSliceVF()` 对 L 使用严格下三角，对 P 使用含对角线的下三角。L 排除当前 token 的自依赖，P 则包含当前 token 的输出贡献。同一次 `exp(G_i-G_j)` 服务两者，省去了独立 Gamma 矩阵的搬运；直接对差值取 exp 也避免了两个很小的指数结果相除。

## 3. Cube、Vector 与搬运流水的配合

| 优化位置 | 实现方式 | 减少的开销 |
| --- | --- | --- |
| `RunQkPrefixMm` | 将 `[K前32行,Q前32行,K后32行,Q后32行]` 拼入 L0A，一次 MMAD 同时产生 KKT/QKT | 重复加载 L0B 中的 K、分开矩阵乘的调度开销 |
| `PrepareStage1*SliceVF` | 在 FP32 寄存器中完成四个变换，随 BF16 转换直接写 compact zN | 中间 UB 张量和单独的布局转换 |
| `BuildInverseSlice` | 两个 AIV 各求一个 32×32 对角逆块，AIC 用两次矩阵乘补齐左下块 | 将块间组合交给 Cube，保留 AIV 对小三角块的处理 |
| `RunStatePair` | `kcd*S^T` 与 `qPrime*S^T` 复用同一份 L0B 状态 | 再次加载整块状态 |
| `PublishDirect` / `PublishL1` | L0C 结果通过 Fixpipe 直接发往 UB/L1，并进行所需类型和布局转换 | L0C→GM→UB/L1 的往返 |
| `ProcessRecurrenceAic` | 输出两项在 L0C FP32 累加后一次写出 | 两项输出分别落 GM、读回后相加 |

Ascend 950 的 L0C→UB 和 L0C→L1 搬运能力可参见 CANN 官方的 [Ascend 950 特性导航](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/asc_950_feature_guide.md)。上述具体调用与缓冲区安排以当前仓库代码为准。

当前求逆入口 `InverseAIVPanel8VF()` 实际调用 `InverseAIVPanel8DirectBrcVFImpl()`：一次维护八行的累加器，每个已求得的逆矩阵行加载一次后供八行复用，组内新求得的行保留在寄存器中。系数通过 B32 从 UB 直接广播，省去八个系数行寄存器和对应 Gather；各行仍按 `j=0..i-1` 的顺序累加。文件中的 Gather、Pair、AoSoA 等其他版本不参与当前入口的计算。

对角块求逆后，AIV 发布的矩阵暂为 `[[A00,0],[-L10,A11]]`。AIC 依次计算 `tmp=(-L10)*A00` 和 `A10=A11*tmp`，再用 A10 覆盖左下块，得到完整 A。把临时左下块误当成最终逆矩阵会导致错误的依赖分析。

## 4. 双缓冲与同步边界

L1 使用两个 160 KiB 的 chunk 槽，按 `chunk&1` 交替；后方放置 32 KiB 的 BF16 状态，总计 352 KiB。每槽 KCD 区的前 64 行是 kcd，后 64 行是 qPrime。L0A/L0B 也分成两个 stage，用于交替加载和计算；L0C 为临时结果、Vnew、输出分别留出区域。

AIC 在处理 chunk n 的递推前，先预取并计算 chunk n+1 的 QK 前缀。AIV 也在等待 chunk n 的状态增量前准备 chunk n+1 的前缀。下一 chunk 的预处理不依赖状态，因此可以与当前递推重叠。AIV 每次把自己的 32 行输入拆成 16+16 行：前 16 行被 VF 消费后，立即在这部分 UB 中加载下一 chunk 的数据，同时 V 流水继续消费当前后 16 行。

| 信号或锁 | 生产者与消费者 | 必须保留的边界 |
| --- | --- | --- |
| `Mutex` | 同一核内 MTE2/MTE1/M/V/FIX/MTE3/S 等流水 | 确认本地物理缓冲的上次使用已结束 |
| `PREFIX_DIRECT*_READY/FREE` | AIC 发布 KKT/QKT，AIV 消费并归还 | 下一次 Fixpipe 写入前，旧的 UB 分数已被读完 |
| `PREFIX*_CRITICAL_READY` | AIV 发布逆块，AIC 开始拼接完整逆矩阵 | 之前排在同一 MTE3 上的 rawKcd/vBeta/qPrime/kg 也已就绪 |
| `PREFIX*_READY` | AIV 发布 maskedQkt，AIC 计算输出第二项 | 只在使用 maskedQkt 时等待完整前缀 |
| `STATE_READY/FREE` | AIV 发布 L1 状态，AIC 搬入 L0B 后归还 | L1 状态可提前复用，L0B 则需保护到两次状态矩阵乘结束 |
| `RECURRENCE_DIRECT0_READY/FREE` | AIC 发布状态增量，AIV 更新主状态后归还 | 增量 UB 不可在 AIV 消费前覆盖 |
| `SLOT*_FREE` | AIC 释放 L1 槽，AIV 准备后续 chunk | chunk n+2 仅能在 chunk n 的操作数已读入 L0 后覆盖同槽 |

FP32 状态增量复用当前槽的 Stage1 发布 UB，依靠既有发布/消费顺序保证安全。`gCum` 也需要双缓冲，因为下一 chunk 预处理后，当前递推仍要读取自己的 `G_last`。每个 head 结束时要收回末次 DIRECT 的归还信号，避免计数信号影响同一核组接下来的工作项。固定偏移的容量断言只能检查空间大小，不能替代这些生命周期约束。

## 5. Streaming、边界与精度

当以下静态代价比较成立时，使用纯 AIV 逐 token 递推；长度为 0 时仅复制初态：

```text
L * Dk * Dv * 7 * 256 < ceil(L/64) * 2 * 5832704
```

左边根据有效维度估计 Vector 工作量，右边按固定物理 tile 估计 Cube 工作量。`5832704` 可由每 chunk 的矩阵乘 MAC 数相加得到：前缀 `128*64*128`，逆块组合 `2*32^3`，kcd 和 vBeta `2*64*128*64`，状态相关两次矩阵乘 `2*64*128*128`，状态增量 `128*128*64`，输出第二项 `64*128*64`。常数 256 是保守的静态权重，不能视为实际 Vector/Cube 耗时比；短序列也不一定选择 Streaming。

Streaming 每批最多搬入 64 个 token，在一次 VF 内顺序递推并按批写回输出；当前为单缓冲，尚未在批间重叠 MTE2/V/MTE3。Dk≤8 时将最多8行状态打包进64个FP32 lane，用分段归约计算；通用版本逐行计算，但同样在一批 token 内将状态保留在寄存器。Streaming 直接计算 `delta=beta*(v-alpha*S*k)`、`S=alpha*S+delta*k^T`、`out=S*(scale*q)`，避免构造和填充完整的分数/逆矩阵。两个 AIV 沿 Dv 独立更新状态，AIC 对相同工作项跳过计算和 chunk 握手。

| 路径 | 主状态 | 矩阵乘/状态增量 | 状态舍入位置 |
| --- | --- | --- | --- |
| Cube + FP32 状态 | UB 行主序 FP32 | Cube 读 BF16 状态副本，状态增量以 FP32 发往 UB | 主状态更新保持 FP32，供 Cube 的副本转换成 BF16 |
| Cube + BF16 状态 | UB compact zN BF16 | Cube 增量先转换成 BF16，再升至 FP32 做衰减与相加 | 每 chunk 更新结束写回 BF16 |
| Streaming | 两种接口类型均为 FP32 | Vector 按有效 Dk/Dv 逐 token 计算 | BF16 finalState 在序列结束时转换 |

因此 `Exact` 不表示全流程 FP32，也不保证与旧实现或两条新路径之间逐位一致。输出都转换为 BF16，数值精度需按算子误差标准判断。

尾 chunk 的 Q/K/V、beta 和无效特征补零，g 补零后再做前缀和，使 `G[63]` 等于真实末 token 的累积值。最终仅写回有效 token 和 Dk/Dv，空 batch 返回原始状态；`actualSeqLengths` 是各序列长度，必须非负且总和等于 T。初态按 batch/head stride 读取，终态按连续布局写回。入口遇到不满足融合路径约束的输入直接返回，没有旧 arch35 实现可回退。

## 6. 后续性能验证重点

从代码结构看，消除 GM 中间量和全核屏障是最直接的收益来源，片上复用、寄存器融合和流水重叠进一步缩短执行时间。但仅凭源码无法量化它们各自对 3～5 倍加速的贡献。

后续实测应关注以下仍存在的限制：

- 当 `B*Nv` 小于核组数时，按 head 持续处理会限制可用并行度；不同 batch 长度差异大也会造成静态分配不均。
- Cube 始终计算 64×128 的物理 tile，较小 Dk/Dv 或尾 chunk 存在无效计算；可结合设备数据评估维度特化与 Streaming 分界。
- 多个 value-head 共享同一 Q/K head，但当前工作项仍分别加载和处理 Q/K；是否跨 head 复用需要同时考虑片上容量与同步成本。
- 求逆的 Vector 广播/加载、Fixpipe 搬运、关键 READY 等待谁占主导，应由 profiling 决定，不能仅凭减少某类指令推断总体收益。

本次修改仅添加或更新中文注释及本文，不调整布局、同步、运算或分支。6 个源码文件去除注释后的代码标记与原提交一致，`git diff --check` 通过。忽略量化后，chunk 公式与逐 token 递推的 88 组 CPU FP64 数值核对通过，最大绝对误差为 `1.11e-15`，覆盖空序列、尾块、多 chunk、非对齐维度及无 g 场景。这些检查不替代 NPU 精度回归和性能测试，本次未重跑设备测试。

本次 rebase 保留 `121832027` 的批量 Streaming/PackedRows 实现，仅将原中文注释适配到新版逻辑；去除注释后的源码与该提交一致。新版设备验证见 [RDV性能分析报告](rdv_regression_analysis_20260908.md)。
