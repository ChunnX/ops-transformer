# MegaMoe Kernel 逐阶段流程解读

本文以 `ops-transformer_5/mc2/mega_moe/op_kernel` 当前代码为准，只解释 kernel。源码快照、执行核及行为差异复核日期：2026-10-08。用户提供的旧流程仅作为叙述风格参考，本文不沿用其中的调度结论。

阅读顺序建议：先看第 1、2 节和第 8 节，再展开输入、同步及特殊分支。每个编号小节均可折叠。本文放在 `arch92`，链接相对该目录；`#L` 后为源码行号，可在支持源码行号定位的阅读器中跳转。不支持行号片段的阅读器仍可打开文件，随后搜索链接中的函数名。

各工作小节就地标明执行核。“全部 AIV”在 Arch35 指 AIV0/AIV1 共 2P 个，在 Arch92 指 P 个唯一 AIV；“某核负责”不等于该角色的每个核都有数据任务。共享阶段按共享量化模式分工，MoE 阶段按 MoE 量化模式分工，二者不可互相代替。

**比较口径：** 两边都是全 AIV 执行同一算法、按物理编号分任务并遵守同一同步协议时，本文归为“行为一致”，不把 2P/P 的资源数量差异当成职责迁移。只有任务归属、阶段重叠、执行顺序、数据交接或实现策略真的变化，才单列差异。性能评估只说明代码可支持的开销变化与可能的瓶颈，**不是实测结论**；不同芯片的核数、吞吐、带宽和实际 Wave 参数也不能假设相同。文中的固定 P 对比仅用于隔离软件行为。

<details>
<summary><strong>1. 阅读范围、芯片角色和当前支持状态</strong></summary>

本文主线是 MTE 通信的 Wave kernel：Arch35 的 A8W8、A8W4、A4W4，以及 Arch92 的统一 Wave。Arch35 的 URMA/Layered 是另一套入口，单独在后文折叠说明；Arch22 不在本次比较范围。Tensor 仓的 MMAD 内部实现不属于本算子 kernel，本文只解释本仓如何切片、调用和同步，不把外部实现当成本仓逻辑。

假设启动了 `P` 个逻辑 block：

| 角色 | Arch35 | Arch92 |
|---|---|---|
| AIC | P 个；负责矩阵乘 | P 个；负责矩阵乘 |
| AIV | 每个逻辑 block 两个，共 2P 个 | 每个逻辑 block 一个，共 P 个 |
| AIV0 | 计算向量角色；根据模式做 Activation 或 W4 权重展开 | 没有可独立并行的第二个子角色 |
| AIV1 | Dispatch 向量角色；根据模式兼做 Activation、Combine | 唯一 AIV 同时承担计算向量和 Dispatch 角色 |
| 逻辑 block 编号 | AIV 物理编号 / `GetTaskRation()` | 直接使用物理编号 |

`RunsComputeVectorWork()` / `RunsDispatchVectorWork()` 是角色选择，不是完整的核类型判断；调用者还必须保证运行在相应 AIV 分支。Arch92 两个谓词都为真，不代表有两个向量核。

代码：[Arch35/Arch92 硬件角色策略](../arch35_92/common/mega_moe_arch_policy.h#L29)。

当前 Arch35 的三种量化模式有各自 Wave 类。Arch92 共用 `MegaMoeArch92Wave`，支持 A8W8、A4W4；A8W4 的模板形态保留，但当前有 `static_assert` 阻止实例化，等待 Tensor 直接 FP8×FP4 接口。不能把保留模板理解为已能运行 A8W4。共享专家的量化模式与 MoE 专家模式分别传入，不能仅看 MoE 模式就确定共享专家的分工。

代码：[Arch35 入口选择](../arch35/mega_moe_apt.cpp#L61)、[Arch92 Wave 类型与 A8W4 编译期限制](mega_moe_wave.h#L84)。

当前 Arch92 入口还明确要求 `ORIG_DTYPE_X == DT_BF16`、输出 BF16，并且只实例化 MTE 路径。下文预量化输入、共享专家直接读原 X 的分支，是公共模板以及 Arch35 的实际分支；**当前 Arch92 入口没有开放它们**，Arch92 实际输入准备走 BF16 量化。入口门控与公共模板包含某段代码是两回事。

代码：[Arch92 实际入口门控](mega_moe_apt.cpp#L80)。

Arch92 采用 GM 交接 GMM1 结果、保存 Dispatch cumsum、A4W4 Activation 输出 FP4 的执行策略。这是当前软件流水的选择，不是“1C1V 硬件不能写 UB”。Arch35 的默认 Generic GMM1 使用 UB 交接，但 A8W4、TopK 权重预取等分支另有 GM 路径。

代码：[两种执行策略](../arch35_92/common/mega_moe_execution_policy.h#L24)。

</details>

<details>
<summary><strong>2. 从进入算子到输出：先建立完整顺序</strong></summary>

进入后，先绑定本卡/对端通信窗口、workspace、权重和 tiling 参数。随后每个核进入公共 `ProcessWave`：

1. **输入准备：** Arch35 全部 AIV0/AIV1、Arch92 全部唯一 AIV，均分清状态和本地 token 量化/打包。Arch35 无共享时两类 AIV 都发路由，有共享时仅 AIV1 发；Arch92 始终由唯一 AIV 发。通知配对 AIC 的是 Arch35 AIV0 / Arch92 唯一 AIV；AIC 等可读条件，不量化、不发路由。
2. **共享 GMM1/Activation：** AIC 做矩阵乘。Arch35 共享 A8W8/A4W4 由 AIV0 做 Activation，共享 A8W4 则由 AIV0 展开权重、AIV1 做 Activation；Arch92 当前由唯一 AIV 做 Activation。共享直接吃本地 token，不经过 Dispatch。
3. **MoE Wave：** Arch35 AIV1 / Arch92 唯一 AIV 接收 count、做 Dispatch 和稳态 Combine；AIC 做两段 GMM。Arch35 MoE A8W8/A4W4 的 Activation 在 AIV0，A8W4 的在 AIV1，W4 展开按阶段由 AIV0 做；Arch92 Activation 在唯一 AIV。所有相关角色各自推进调度游标。
4. **导出/清 count：** Arch35 AIV1 block0 导出、AIV0 分担清零；Arch92 唯一 AIV block0 导出、全部唯一 AIV 分担清零。全部物理 AIV 参加尾部同步，AIC 不参加。
5. **共享 GMM2：** AIC 做矩阵乘并写本卡 GM；Arch35 共享 A8W4/A4W4 还由 AIV0 展开 W2。其他 AIV 不做矩阵乘或 MoE Combine，继续后续工作。
6. **出口握手/Unpermute：** Arch35 全部 AIV0/AIV1、Arch92 全部唯一 AIV，分担跨卡握手和本地 token 聚合。它们读 MoE 回传及共享结果，写 Y；AIC 不参与。
7. **收尾：** 所有核走到收尾调用，实际剩余 GM-handoff ACK 由 AIC 消费；各核恢复自己的控制状态后结束。

这是**每个核看到的函数调用次序，不是每一步都有全核屏障**。某个 AIV 已经进入下一步时，另一个 AIC 可以还在上一阶段；是否允许继续取决于后面具体的数据 ready 或 buffer 复用通知。

例如 Arch35 共享 A8W8：AIV1 不做共享 Activation，因此它可以先接收 MoE count、做 Dispatch；与此同时 AIC/AIV0 仍在做共享 GMM1/Activation。Arch92 唯一 AIV 必须先完成自己的共享 Activation，才能用同一核和 UB 做后续 Dispatch。

代码：[公共主流程 ProcessWave](../arch35_92/mega_moe.h#L1078)。

数据主线是：

```text
源卡本地 X
  -> 量化/打包的源卡通信记录
  -> 专家所在卡 Dispatch 到紧凑 GM + 路由 metadata
  -> GMM1 Gate/Up (Arch35 部分路径为 AIV0 UB；其他为 GM)
  -> Activation/SwiGLU + 量化 -> GM 数据和 E8M0 scales
  -> GMM2 BF16 GM
  -> Combine 写回源卡 [token, topk槽]
  -> Unpermute 加权聚合 -> Y

共享专家：本卡输入 -> 共享 GMM1 -> Activation -> 共享 GMM2 GM
                                                     -> 同一 Unpermute 加到 Y
```

</details>

<details>
<summary><strong>3. 分核和几个容易混淆的单位</strong></summary>

本节是单位说明，不是独立执行阶段：AIC 用 problem/MN tile 划分矩阵乘；相应 Activation、权重展开及普通 Combine AIV 重放 tile 分配；Arch35 AIV1 / Arch92 唯一 AIV 的 Dispatch 按专家行范围搬运；全部物理 AIV 的最终 Unpermute 则按原 token 分工。不能将这几种工作单位互换。

下文用 `H` 表示原 token 宽度，用 `N1` 表示 GMM1 的完整 Gate+Up 输出宽度，用 `F=N1/2` 表示 Activation/GMM2 输入宽度。

| 名称 | 实际含义 |
|---|---|
| GMM1 problem | 一个专家或该专家在一个 Wave 内的 slice：M 行，N=N1，K=H |
| GMM2 problem | 对应 slice：M 行，N=H，K=F |
| M-group | 最多 256 个专家行；末组可以更少 |
| GMM tile | scheduler 分配的 MN 区块，通常 M256×N256；K 由 MMAD 内部分块累加 |
| Activation 子块 | 权重预取分支可将 M256 拆成 M128 两段，不是两个独立 Wave |
| Wave | 有限数量 M-group 的调度区间，可覆盖多个专家，也可截断一个专家 |
| Dispatch 行 | 专家输入行，不等于源卡原 token ID；一个源 token 可产生多个专家行 |
| Combine 行 | 一个专家输出的 H 个数，或其量化通信记录 |
| TopK flattened index | `tokenId * topK + topkSlot`；路由发送的是该下标 |

`mGroupsPerWave_` 在 kernel 的 `Init` 中从 tiling 读取。kernel 用它推进专家/token 游标，不是在每个 V 核上重新根据 H 算 Wave 容量。GMM1 的 N 方向分块看的是 N1，而不是 H；H 是 GMM1 的 K。

代码：[读取 Wave 参数](../arch35_92/mega_moe.h#L300)、[构造 GMM1/GMM2 problem](../arch35_92/common/mega_moe_gmm_common.h#L309)。

同一专家尾部不足 256 行也独占一个 M-group；不同专家使用不同权重，不能把两个专家的尾部塞进同一个 MMAD problem。所谓“每核两个 tile”只能理解为预算设计目标，不是每核必然调用两次 GMM、两次 Combine。

代码：[GetMGroupCountForRows](../arch35_92/common/mega_moe_utils.h#L156)、[AdvanceExpertTokenPositionInWave](../arch35_92/common/mega_moe_utils.h#L349)。

</details>

<details>
<summary><strong>4. 初始化：各核先拿到什么，哪些状态留到后面</strong></summary>

**执行核：** AIC、Arch35 AIV0/AIV1、Arch92 唯一 AIV 都执行 `Init`，但各自绑定的是本核视图，不是由一个核替其他核统一初始化。Dispatch UB 仅由 Arch35 AIV1 / Arch92 唯一 AIV 建立；MMAD context 的实际 Init/更新/矩阵乘在 AIC 分支。

`Init` 先确定本卡 rank、逻辑 block 编号、物理 AIV 编号，绑定通信窗口和 workspace。通信窗口最前面保留异常信息区域；实际 peer-memory 布局在其后。构造不同工作上下文时，输入量化/最终 Unpermute 使用物理 AIV 分工，GMM/普通 Wave Combine 使用逻辑 block 分工，不能混用两个编号空间。

Dispatch UB 只有 Dispatch 角色初始化：Arch35 AIV1 或 Arch92 唯一 AIV。AIC 只绑定 GM count 视图，不建立 Dispatch 的搬运环。Arch35 AIV0 后续如果需要在末轮参与量化/去重 Combine，会在那时才绑定对应 UB。

代码：[InitGmmConfigs](../arch35_92/mega_moe.h#L241)、[Init](../arch35_92/mega_moe.h#L300)、[DispatchBuffInit](../arch35_92/mega_moe.h#L358)。

Arch92 和 Arch35 W4 流水会把 Dispatch cumsum 保存在 GM。原因是同一 Dispatch AIV 后面还要做 Activation，UB 中的 cumsum 区可能被覆盖；下一轮 Dispatch 前需要重新装入必要范围。保存它不代表每个 token 都重新读一遍完整前缀表。

Arch92 还保存带类型的 MMAD context。配置签名不变时复用已初始化对象；每个专家仍更新动态 M/problem 参数。切换 GMM1/GMM2、layout 或 L1 配置可能触发重新 Init，因此不能说成“全算子绝对只 Init 一次”。

代码：[Dispatch cumsum 保存策略](../arch35_92/mega_moe.h#L379)、[BlockMmadContext 的配置比较和参数更新](../arch35_92/common/mega_moe_gmm_common.h#L393)。

**行为差异与潜在性能影响：** 对照 Arch35 A8W8，Arch92 的 Dispatch AIV 还做 Activation，会覆盖同一 UB，因此增加 cumsum 的 GM 保存/恢复；Arch35 A8W4 本来就有这项行为，不能称为所有 Arch35 路径都新增的开销。Arch92 还传入可跨调用复用的 MMAD context，Arch35 Generic 默认局部创建。这项软件策略可减少配置不变时的重复 Init，但 GMM1/GMM2 或 layout/L1 配置切换仍会重新 Init；专家很多、单个 problem 很小时更值得观察，不能直接推导矩阵乘吞吐提升。具体交接和初始化差异在 7.1、9.1 展开。

</details>

<details>
<summary><strong>5. 输入准备：所有 AIV 先清状态，再准备本卡 token</strong></summary>

<details>
<summary><strong>5.1 每个 AIV 的实际执行顺序</strong></summary>

Arch35 的 2P 个 AIV、Arch92 的 P 个 AIV，都参与这一步，不区分 AIV0/AIV1：

1. 按物理 AIV 工作上下文均分本卡 token，得到自己的 `[start,count)`。
2. 初始化量化、路由发送和清零使用的 UB。
3. 清理分给自己的 workspace flag 区；预取模式还清独立的 GMM1 tile 状态区。
4. 在 MTE3 上发布 `INPUT_RESET_DONE_FLAG`。
5. 若自己的 token 非空，BF16 输入做 MX 量化；已经是 FP8/FP4 的输入只打包，不重新量化。
6. 推进本物理 AIV 的调用轮次，再在 MTE3 上发布 `TOKEN_DATA_READY_FLAG`。

空 token 核也必须发两个通知、推进轮次；否则后面全 AIV 事件的参与者会不齐。这里清的是算法状态区，不是马上清对端发来的 count；通信 count 接收区在流程尾部清。

代码：[PrepareLocalInput](../arch35_92/mega_moe.h#L941)、[ResetSyncStatus](../arch35_92/stage/mega_moe_workspace_reset.h#L24)。

**行为一致：** 两边都是全部物理 AIV 分担清零和输入量化/打包，并发布相同的 reset/token-ready 协议。Arch92 没有在这里额外接手原来 AIC 或专职 AIV 的工作。物理 AIV 数量会影响每核工作量，但这是资源并行度，不是新增算法或不同流水；不能仅凭 2P 变 P 就宣称时间翻倍。

</details>

<details>
<summary><strong>5.2 量化/打包到底产出什么，内部为什么等 buffer</strong></summary>

**执行核：** Arch35 全部 AIV0/AIV1、Arch92 全部唯一 AIV，分别处理按物理 AIV 编号均分的本地 token。下面的读 X、量化、打包、写通信记录及双 buffer 等待，都发生在这些 AIV 本核；AIC 不执行。当前 Arch92 只实际进入 BF16 量化分支。

每个 token 通信记录包括对齐后的量化 data、E8M0 scales。MoE 记录在开启 TopK 权重预取**或 Combine 去重**时还携带 FP32 TopK 权重：去重合并端即使不开预取，也需要这些权重做成员乘加。独立共享输入记录不参与路由合并，不因 Combine 去重追加权重段。数据字节数按 FP8/FP4 元素打包比例计算；scale 对应每 32 个数的 MX 分组，搬运视图按 64-K scale 对组织。

BF16 路径在 UB 里先读一行/一组输入，求量化参数，再生成 FP8 或 FP4 数据及 scales，写源卡通信窗口。有共享专家且量化类型不同，会另写共享输入区；模式相同则尽量复用。

代码：[CreateQuantProcessConfig](../arch35_92/stage/mega_moe_token_quant.h#L39)、[QuantizeTokenInUb](../arch35_92/stage/mega_moe_token_quant.h#L122)。

双 buffer 的次序是：复用前等上次 MTE3 读完 UB；MTE2 搬入后通知 V；V 量化后通知 MTE3；最后排空两个 buffer。`MTE3_MTE2` 防止搬入覆盖还在发送的 UB，`MTE2_V` 防止 V 读未搬完数据，`V_MTE3` 防止发送未算完结果。这些都是本核流水依赖，不是跨卡同步。

已经量化的输入走 Pack：将原始 data/scales 搬成窗口规定的记录格式。共享专家直接读原始预量化输入与 MoE 通信区已经打包完成，是两个不同条件。

这里的 Pack/共享直接输入用于说明公共代码分支；当前 Arch92 的 BF16-only 入口不会进入它们。预量化 Pack 的权重搬运按其编译期预取分支执行，不能仅凭 BF16 量化路径带有 Combine 去重权重段，就推断所有预量化组合也已支持。

代码：[QuantizeLocalTokens](../arch35_92/stage/mega_moe_token_quant.h#L169)、[PackPreQuantizedLocalTokens](../arch35_92/stage/mega_moe_token_quant.h#L286)。

**行为一致的范围：** 同为 BF16、采用相同 MoE/共享量化类型及权重选项时，两边复用同一量化、记录格式和本核 buffer 协议；共享独立量化也不是 Arch92 新增任务。Arch35 的预量化 Pack 是额外支持分支，当前 Arch92 未开放，不能拿它和 Arch92 BF16 量化比较后归因为 CV 配比。

</details>

<details>
<summary><strong>5.3 发下标 -> 等准备条件 -> 通知 AIC -> 发 count</strong></summary>

**先明确谁发送、谁通知。** 所有物理 AIV 已各自执行 `PrepareLocalInput`；后面的发送不是始终由所有 AIV 做：

| 场景 | 发下标、发 count 的核及分工 | 通知配对 AIC 的核 | 其他核此时做什么 |
|---|---|---|---|
| Arch35，无共享专家 | 全部 2P 个 AIV0/AIV1，按物理 AIV 编号均分全局专家 | 各 AIV0；AIV1 不发该通知 | AIC 等输入完成条件 |
| Arch35，有共享专家 | P 个 AIV1，按逻辑 block 编号均分全局专家 | 各 AIV0，在独立分支中通知；AIV1 不通知 | AIV0 等准备条件后进入共享工作；AIC 承接其通知，直接输入分支可提前读原 X |
| Arch92，无/有共享专家 | P 个唯一 AIV，均分全局专家 | 同一个唯一 AIV，在发送流程中通知 | AIC 等输入完成条件；没有另一个 AIV 分担发送或共享 Activation |

代码：[调用点的 AIV/AIC 分流](../arch35_92/mega_moe.h#L1087)、[发送任务及通知者选择](../arch35_92/mega_moe.h#L996)。

**下面第 1～6 步只由上表的下标/count 发送 AIV 执行。** 每个发送 AIV 已准备完自己那份输入，然后为自己分到的全局专家扫描本卡 TopK。筛出命中该专家的 flattened index，写到专家所在卡的 `[本地专家][源rank]` 下标区域，同时累计该专家的 count。专家任务按发送工人均分，不是量化 token 范围原样沿用；每个发送 AIV 要扫描本卡 TopK 中该专家的命中。

例如 token 7 的 TopK 第 2 槽命中了远端专家，topK=4，那么发送的是 30。Dispatch 收到 30 后算 `tokenId=30/4=7`、`slot=30%4=2`；它不是“发送 token 内容”，也不是发送 expertId。

代码：[SendTopkIdsIndexForExperts](../arch35_92/stage/mega_moe_send_mask.h#L241)、[SendSelectedTopkIdsIndex](../arch35_92/stage/mega_moe_send_mask.h#L113)。

下标发送可以先于全卡 token 数据就绪，因为下标只依赖 TopK 输入。count 则是接收端开始取 token 的承诺：发布 count 前，本卡所有输入记录必须写完，本核对应下标搬运也必须完成。

发送 AIV 的实际顺序为：

1. **发送 AIV 的 MTE3：** 写自己负责的专家下标；下标只依赖 TopK，可以先于其他 AIV 完成 token 记录。
2. **同一发送 AIV 的 MTE3：** 等全部物理 AIV 发布的 `INPUT_RESET_DONE_FLAG`；Arch35 生产者包括 AIV0/AIV1，Arch92 是全部唯一 AIV。随后在本核记录 `MTE3_S` 事件。
3. **同一发送 AIV 的 MTE3：** 等这些物理 AIV 的 `TOKEN_DATA_READY_FLAG`，确保整卡量化/打包记录已写完，不只是自己的 token 已写完。
4. **通知者的 MTE3：** Arch35 无共享时的 AIV0，以及 Arch92 无/有共享时的唯一 AIV，向配对 AIC 发 mode4 `INPUT_PREPARE_DONE_FLAG`。Arch35 的 AIV1 在此永远不发；Arch35 有共享时 AIV0 的通知走下面的独立分支。
5. **发送 AIV 的 MTE3/Scalar：** 排空 MTE3 前序操作，读取本物理 AIV 已推进的轮次，向对应专家卡发送自己的 count+epoch。MTE3 排空也保证第 1 步下标已经写完，不能将 count 提前到下标提交后、写完前。
6. **同一 AIV 的 Scalar：** 在公共尾部消费第 2 步记录的 `MTE3_S`，承接 reset 完成依赖，之后才离开输入准备阶段。

**Arch35 有共享时的 AIV0 另走一条线：** 它不执行上述下标/count 发送，而是 MTE3 等全 AIV reset、记录本核 `MTE3_S`；若共享输入来自量化/打包记录，则再等全 AIV token-ready；然后通知配对 AIC。共享直接读原预量化 X 时，这个 AIV0 将 token-ready 等待推迟到 MoE 前。当前 Arch92 BF16-only 入口不走共享直接输入，也不存在这条独立 AIV0 分支。

**AIC 是接收通知的一方：** 通常在 MTE2 等配对 AIV 的输入完成通知；无共享时再用本核 `MTE2_S` 将依赖传到 Scalar。Arch35 共享直接输入时先读取原 X，在 MoE 前才消费输入通知。AIC 不扫描 TopK、不发下标/count，也不替 AIV 等 mode0 事件。

reset 在每个生产者上先于 token-ready，所以 token-ready 已包含此前 reset 的完成顺序；代码仍独立消费 reset 事件并把依赖传给 Scalar，维持 reset 协议及后续状态读取的顺序。不要理解成 count 有两个互不相关的输入数据条件。

代码：[sendIndexAndCount lambda](../arch35_92/mega_moe.h#L973)、[共享时 AIV0 独立通知](../arch35_92/mega_moe.h#L1007)、[AIC 的等待](../arch35_92/mega_moe.h#L1029)、[共享直接输入的延后等待](../arch35_92/mega_moe.h#L659)。

count 的低 24 位放数量，高 8 位放调用 epoch。零 token 的专家也要发布当前 epoch；接收端不仅看 count 数值，还看是不是本轮数据。轮次计数不是 GMM1 ready flag。

这里区分两种“空”：**发送 AIV 没分到专家**时，不搬下标/count，但仍执行约定的事件等待和本角色的 AIC 通知；**分到了专家、该专家 count=0**时，发送 AIV 仍为该专家发布本轮 epoch。接收 count 的是专家卡 Arch35 AIV1 / Arch92 唯一 AIV；Arch35 满足去重协作条件的 AIV0 还会独立读表。

代码：[SendTopkIdsCountForExperts](../arch35_92/stage/mega_moe_send_mask.h#L188)、[GetSyncRoundTag](../arch35_92/common/mega_moe_utils.h#L48)。

**行为差异与潜在性能影响：** 无共享时，两边都是全 AIV 均分同一套专家扫描/发送工作，通知者只是按硬件配对选择，不构成额外任务。**有共享时则真正不同：** Arch35 AIV0 不扫描/发送路由，完成准备条件即可通知 AIC，AIV1 同时发路由；Arch92 唯一 AIV 必须先扫描并提交自己负责的下标，等准备条件后才通知 AIC，再发 count。因此其共享 GMM1 启动可能被本核 TopK 扫描和下标提交延后，专家数/TopK 扫描量大时更需关注。不能说 AIC 一定等到 count 全部发送完：通知位于 count 之前，AIC 仍可与后续 count 搬运重叠；AIV 自己复用 UB 则要等前序 MTE3。

</details>

<details>
<summary><strong>5.4 Arch35/Arch92 和有无共享专家的分工差别</strong></summary>

**没有共享专家：** Arch35 全部 2P 个 AIV 均分专家下标/count 发送；AIV0 在发布 count 之前通知对应 AIC，AIV1 不负责该通知。AIV0 不只是通知核，它也量化/打包、清零、发送自己负责的专家。Arch92 的唯一 AIV 做同样的准备/发送，同时通知 AIC。

**有共享专家：** Arch35 改为只有 P 个 AIV1 发送全部路由；AIV0 等准备条件后单独通知 AIC，以便 AIC/AIV0 开始共享计算，AIV1 仍可继续路由发送。Arch92 唯一 AIV 发送路由并通知 AIC，随后自己做共享 Activation；不存在另一个 AIV0 与发送核并行。

代码：[NoSharedExpert 与共享分支](../arch35_92/mega_moe.h#L996)。

**共享直接输入：** X 已预量化且 H 是 64 的倍数时，共享 GMM1 可直接读原 X/scales，AIC 不必先等 MoE 通信打包完成；Arch35 AIV0 将 TOKEN_DATA_READY 的等待推迟到进入 MoE 前。有 BF16 输入或不满足直接读取条件时，共享 GMM1 要等量化/打包输入可读。

当前 Arch92 的入口只收 BF16 X，因此有共享专家时实际选择 `QuantOrPackDone`，不会选择 `InputDirectly`。上述直接输入差别不能作为当前 Arch92 可以提前启动共享 GMM1 的依据。

Arch35 UB-handoff 还多一个条件：AIV0 发 `INPUT_UB_FREE_FLAG`，AIC 的 FIX 等它再往 AIV0 UB 写结果，防止覆盖输入准备仍使用的 UB。Arch92 GM-handoff 没有这组 UB 输出通知，但仍有自己的输入和 UB 复用依赖。

代码：[GetSharedGmmStartCondition](../arch35_92/mega_moe.h#L1048)、[AIC 等待和 UB 释放](../arch35_92/mega_moe.h#L1029)、[进入 MoE 前补齐依赖](../arch35_92/mega_moe.h#L659)。

**差异归因：** 有共享时发送/通知落到同一个 AIV，是当前 1C1V 调度的职责安排；UB-handoff 改 GM-handoff 是软件交接策略，不是硬件禁止写 UB。Arch35 共享直接输入可提前启动、Arch92 当前 BF16 必须先量化，则是入口支持范围差异，不应混为“少一个 AIV 导致必须量化”。三者的潜在耗时分别来自发送关键路径、GM 中间交接、输入来源，不能重复计算成同一种成本。

</details>

</details>

<details>
<summary><strong>6. 共享 GMM1/Activation：哪些核在分开干活</strong></summary>

<details>
<summary><strong>6.1 共享专家的输入和遍历方式</strong></summary>

**执行核：** 所有 AIC、Arch35 AIV0/AIV1、Arch92 唯一 AIV 都走共享地址配置和专家循环，但实际数据工作不同。AIC 做 GMM1；Arch35 共享 A8W8/A4W4 的 AIV0、共享 A8W4 的 AIV1，以及 Arch92 唯一 AIV 做 Activation；Arch35 共享 A8W4 的 AIV0 做权重展开。无实际工作角色的调用返回或仅推进游标，不表示它也计算了共享输出。

共享专家没有路由筛选，每个共享专家处理本卡全部 token。先绑定共享输入地址，再依次遍历共享专家，换该专家 W1/scales 和 Activation 输出基址，调用 GMM1/Activation。

输入为可直接读取的预量化 X 时使用原 X、原 scales；否则使用量化/打包通信记录，类型不同的共享专家使用独立共享输入区。GMM1 输出是 Gate+Up，Activation 输出是 F 宽的量化数据/scales，存入共享专属 GM，不能与 MoE 专家输出混淆。

代码：[ConfigureSharedGmm1Input](../arch35_92/mega_moe.h#L563)、[ProcessSharedExpertGmm1Loop](../arch35_92/mega_moe.h#L617)。

**行为一致的部分：** 共享专家遍历、直接处理本卡全部 token、切换对应权重和独立输出地址的算法一致，1C1V 没有新增共享 token 搬运次数。真正变化的是后面 6.2/6.3 的参与角色、交接位置及能否与 MoE 通信重叠。

</details>

<details>
<summary><strong>6.2 Arch35 按共享专家量化模式选择角色</strong></summary>

**共享 A8W8/A4W4：** AIC 用 Generic MMAD，FIX 将 GMM1 结果交给 AIV0 UB；AIV0 等 tile 就绪，做 Activation 并写量化 GM。AIV1 进入同一公共调用，但角色判断使它不执行 Activation，仅按 tile 总数推进必要的分核游标，随后可以进入 MoE count/Dispatch。

**共享 A8W4：** AIV0 从 GM 读 FP4 权重，在 UB 展开为 FP8，再写 AIC 的 L1；AIC 等对应权重块，计算 GMM1 并写 GM；AIV1 等 GMM1 GM 结果，读回 UB 后做 Activation。此时 AIV1 不能越过共享 Activation 直接做 MoE Dispatch，因为共享 Activation 就是它的工作。

这里按的是 `SharedQuantConfig::AXW_MODE`，不是 MoE 的模式。比如 MoE A8W8 + 共享 A8W4，仍然是上述第二种共享分工。

代码：[RunSharedExpertGmm1Activation 的两种调用](../arch35_92/mega_moe.h#L587)、[Generic 的角色裁剪](../arch35_92/stage/mega_moe_gmm1_activation.h#L647)。

**对照 Arch92 的重叠差异：** 共享 A8W8/A4W4 时，Arch35 AIV1 跳过共享 Activation 后可以读 MoE count/建表/做首轮 Dispatch，而 AIC/AIV0 还在共享阶段；Arch92 唯一 AIV 必须先完成自己负责的共享 Activation，才进入这些 MoE 通信任务。这会减少共享计算与 MoE 通信的重叠，可能增加首个 MoE GMM1 等 Dispatch 的时间。Arch35 共享 A8W4 的 AIV1 本来就负责共享 Activation，不能将这项“新增串行化”同样套给它；当前 Arch92 也未开放共享 A8W4，不能声称已有同模式性能对照。

</details>

<details>
<summary><strong>6.3 Arch92 的顺序及共享输出同步</strong></summary>

Arch92 没有独立权重展开 AIV0；当前支持的共享 A8W8/A4W4 都用 Generic。AIC 写共享 GMM1 GM，唯一 AIV 等配对 tile-ready，从 GM 搬入 UB，执行 Activation/量化，再写共享 Activation GM。

共享 GM-handoff 在 `ConfigureSharedGmm1Input` 明确挂上 `sharedGmm1ActivationSync`，所以该路径是硬件通知，不是后文 Arch92 非预取 MoE GMM1 的 GM 序号通知。

AIC 用 FIX 发 ready，AIV 在 MTE2 等 ready，防止提前搬未写完的 GMM1；AIV 在 MTE3 发 ACK，AIC 的 Scalar 在提交额度用满时等 ACK，防止未消费硬件事件累积超限。独立共享事件链不与 MoE GMM1 混用，剩余 ACK 在公共流程末尾排空。

代码：[挂共享同步对象](../arch35_92/mega_moe.h#L581)、[Gmm1ActivationSync](../arch35_92/common/mega_moe_gmm_epilogue_sync.h#L135)。

开始共享工作前的 UB 交接也按角色区分：Arch92 唯一 AIV 等本核 `MTE3_MTE2` 后才能搬入 GM 结果；Arch35 共享 A8W4 的 AIV0 搬权重、AIV1 搬 GM 结果，均等本核 `MTE3_MTE2`；Arch35 共享 A8W8/A4W4 的 AIV0 直接做 Activation，等本核 `MTE3_V`，AIV1 不在这个分支追加该等待。AIC 不执行这些 AIV UB 等待。它们保护各自 UB 复用，不是所有核一起等共享 GMM1 开始的屏障。

代码：[SyncBeforeSharedExpertGmm1](../arch35_92/mega_moe.h#L633)。

**交接差异与潜在性能影响：** 对照 Arch35 共享 A8W8/A4W4，Arch92 从 AIC→AIV0 UB 改为 AIC→GM→唯一 AIV UB，增加共享 GMM1 BF16 中间结果的 GM 写入/读回以及 AIV MTE2 装载。另一方面不再按两个 UB 槽等待复用 ACK，而是独立硬件事件链的 15-credit 限制，允许不同 GM tile 在额度内先写出；这可能缓解 AIC 等输入槽，但并未取消 ready/ACK。GM 带宽成本与放宽生产领先的收益谁占主导需实测，不能把“写 GM”直接等同于更慢；A4W4 的后续量化类型差异另见 10.3。

</details>

<details>
<summary><strong>6.4 进入 MoE 前：各核怎样交接 UB 和 reset 依赖</strong></summary>

**执行核：** 全部 AIC、Arch35 AIV0/AIV1、Arch92 唯一 AIV 都调用公共 `SyncBeforeMoeExpertStages`，但不是全核屏障；每类核承接自己的前序流水依赖：

| 执行核/场景 | 在这里执行的等待 | 为什么此核需要等 |
|---|---|---|
| Arch92 唯一 AIV | 本核 `MTE3_MTE2` | 输入/count 发送或共享 Activation 写出仍可能读 UB；后面本核 MTE2 要装 count 表及 Dispatch 数据 |
| Arch35 AIV1 | 本核 `MTE3_MTE2` | 后面本核要读 count；前面可能是路由发送或共享 A8W4 Activation 写出 |
| Arch35 AIV0，共享直接输入 | 先消费全 AIV `TOKEN_DATA_READY_FLAG` | 共享 GMM1 不依赖 MoE 打包，所以该等待先前被推迟；现在不能在 MoE 尚未准备好时继续 |
| Arch35 AIV0，去重开且 N1<=2048 | 本核 `MTE3_MTE2` | 后面要读自己的 count/cumsum，参加 rowDesc 建表 |
| Arch35 AIV0，不参加上述建表，MoE A8W4 | 派生 Wave 中本核 `MTE3_MTE2` | 后面 MTE2 搬入 W4 权重覆盖 UB，须等前序读 UB 的搬出 |
| Arch35 AIV0，不参加上述建表，MoE A8W8/A4W4 | 预取开等本核 `MTE3_MTE2`；关等本核 `MTE3_V` | 预取从 GM 搬 GMM1，非预取由 V 直接处理 UB 结果；保护对应的 UB 写入者 |
| AIC，无共享专家 | 这里不重复等输入通知 | 入口已在 MTE2 等通知并用 MTE2_S 传给 Scalar |
| AIC，共享量化/打包输入 | 本核 `MTE2_S` | 通知已在共享 GMM1 前消费；现在承接该输入/reset 依赖及此前 MTE2 搬入，才可轮询 MoE GM 状态 |
| AIC，共享直接输入 | Scalar 消费配对 AIV0 的 `INPUT_PREPARE_DONE_FLAG` | 共享阶段可提前读原 X，未消费这次通知；进入 MoE 时必须承接 reset 完成条件 |

Arch35 AIV0 的“共享直接输入”等待可与后面的建表/派生 Wave 等待同时发生，不是互斥行。N1 指 GMM1 完整 Gate+Up 宽度，H 才是原 token 宽度。当前 Arch92 只开放 BF16 输入，不进入共享直接输入分支。

代码：[公共分核等待](../arch35_92/mega_moe.h#L659)、[A8W8 AIV0 后继等待](../arch35/mega_moe_wave_a8w8.h#L613)、[A8W4 AIV0 后继等待](../arch35/mega_moe_wave_a8w4.h#L261)、[A4W4 AIV0 后继等待](../arch35/mega_moe_wave_a4w4.h#L280)、[N1/H 参数绑定](../arch35_92/mega_moe.h#L259)。

**行为差异与潜在性能影响：** Arch35 分别保护 AIV0 的计算 UB 和 AIV1 的通信 UB；Arch92 是同一 AIV 在发送、共享 Activation、count/Dispatch 间交接 UB，因此前序 MTE3 尚未结束时，本核后继 MTE2 不能先覆盖。这是职责合用后的真实等待链，但没有新增全 C/V 屏障。潜在成本是本来能在另一 AIV 上推进的工作，现在可能等本核 UB 释放；若前序搬运已结束，这个事件只承接既有顺序，不能把每个 Wait 都当成一段独立延时相加。

</details>

</details>

<details>
<summary><strong>7. 接收 count -> 得到专家行数 -> Dispatch 把数据拉到本卡</strong></summary>

<details>
<summary><strong>7.1 count 表和前缀：此时才知道每个专家实际多少行</strong></summary>

**执行核：** 每个 Arch35 AIV1 / Arch92 唯一 AIV 都独立读取本卡整张 count 表、校验 epoch、计算 cumsum，并写自己的 workspace 副本；不是一个核算完再广播。AIC 和通常的 Arch35 AIV0 不算表，只等自己对应的 workspace ready 后读专家行数。Arch35 去重开且 `N1=tiling.hiddenDim<=2048` 时，AIV0 还独立读表用于建表。

本卡接收表按 `[本卡专家][源rank]` 排列。每个值是该源卡发给本专家的 count+epoch；只有全部条目的 epoch 属于本轮才能继续。数量来自输入 TopK，不是 tiling 预先算出的实际路由结果。

Dispatch AIV 搬整表到 UB，反复检查 epoch；不齐就退避后重读。齐了剥离 tag，计算专家优先、源 rank 次序的全局 cumsum，并向 workspace 发布专家 token 数及其 ready。其他角色用自己的 workspace 视图读取专家行数，随后构造 GMM problem。

当前 MTE 不是每来一个专家count就逐专家发布ready，而是**本核整张count表计算/写回后发布一次ready**。每核的完整专家行数表位于自己的workspace副本，ready用预留标志区第0个专家平面的本核槽；后续读取任意专家都只需先等这次整表ready。

举例本卡两专家、三 rank，count 为 `[2,0,3, 1,4,0]`，cumsum 为 `[2,2,5, 6,10,10]`。专家 0 占紧凑行 `[0,5)`，专家 1 占 `[5,10)`。其中专家 0 的 rank2 片段是 `[2,5)`。cumsum 是跨条目的累计行边界，不是“本核已经发送多少 token”。

代码：[LoadAndComputeExpertCountTable](../arch35_92/stage/mega_moe_token_dispatch.h#L446)、[WaitForCountTable](../arch35_92/stage/mega_moe_expert_count.h#L56)、[PrepareMoeExpertTokenCountTable](../arch35_92/stage/mega_moe_token_dispatch.h#L475)。

Arch35 正常只有 AIV1 读表/发布；AIV0 报告 count-read-done，并在需要专家参数时等 workspace ready。满足上述 N1 阈值的去重建表时，AIV0 另读一份带 epoch 校验的表，后文单独说明。Arch92 唯一 AIV 同时读表、发布并保留 GM cumsum 备份。AIC 不做表计算，但要等数量 ready 后读取。

代码：[PrepareGmmExpertState](../arch35_92/mega_moe.h#L879)、[ReloadDispatchCumsumRange](../arch35_92/stage/mega_moe_token_dispatch.h#L516)。

**算法一致，保存策略有差异：** 整表 epoch 校验、cumsum 和 ready 的算法一致；固定 P 时，正常读表工人都是 P 个（Arch35 AIV1 / Arch92 唯一 AIV），不是 Arch92 相对该阶段砍掉一半工人。对照 Arch35 A8W8，Arch92 在首次算完表后每核多保存一份 cumsum，后续 Dispatch 恢复覆盖范围；会产生前缀表的 MTE3/MTE2 读写，但不是每 token 重算整表。对照 Arch35 A8W4/A4W4，保存/恢复本来已有，不能重复列成迁移新增成本。AIV0 协助去重的额外读表在 12.2 单独比较。

</details>

<details>
<summary><strong>7.2 当前 Dispatch 怎样分核，和旧版按 rank 分核不同</strong></summary>

**执行核：** P 个 Arch35 AIV1 / Arch92 唯一 AIV 各自计算本核负责的行区间并执行 Dispatch。Arch35 AIV0 和 AIC 不拉 token、不写 Dispatch metadata；AIC 只在后续等这些 AIV 写出的输入 ready。

先规划要处理的连续 Wave/专家 slice 行区间，裁到 `maxOutputSize` 容量；再把**整个连续行区间**在 P 个 Dispatch 角色之间均分，余数分配带起点旋转。每核拿到自己 `[globalBegin,globalEnd)` 后，再和专家边界、源 rank 前缀区间求交。

因此当前不是“先给核心分卡，再在每张卡里平均分 token”。一个核心的连续行区间可以跨专家或跨源 rank，也可以完全没有行；只搬它与各段的交集。Arch35 使用 P 个 AIV1，Arch92 使用 P 个唯一 AIV，基本算法一致。

例如待搬范围 `[0,10)`、P=4、旋转起点为 0，可以分为 `[0,3)`、`[3,6)`、`[6,8)`、`[8,10)`。上一小节专家边界在 5，则第 1 核搬专家 0 的两行及专家 1 的一行。旋转起点变化时，长段归属变化，不能把该示例作为固定物理编号分配。

代码：[CalcDispatchCoreRowRange](../arch35_92/stage/mega_moe_token_dispatch.h#L415)、[GetRotatedBalancedWorkRange](../arch35_92/common/mega_moe_utils.h#L207)、[DispatchTokenRange](../arch35_92/stage/mega_moe_token_dispatch_dedup.h#L484)。

**行为一致：** 两边复用相同连续行分核，固定 P 时都是 P 个 Dispatch 工人，没有因 Arch92 全部唯一 AIV 都做 Dispatch 就增加或减少该阶段的工人数。是否被 Activation 占用、何时开始下一段 Dispatch 是 Wave 重叠差异，见 8.2，不应误算为 Dispatch 分核算法不同。

</details>

<details>
<summary><strong>7.3 每核搬一段数据的顺序，以及什么时候通知 GMM1</strong></summary>

**执行核：** 搬数据和发布 Dispatch ready 的是 Arch35 AIV1 / Arch92 唯一 AIV 中有行任务的核；读取 ready 并启动 GMM1 的是分到相应 tile 的 AIC。Arch35 AIV0 不执行下面的 Dispatch 搬运。

每个这样的 Dispatch AIV 先通过 cumsum 找到自己第一行属于哪个源 rank。读本卡已接收的 flattened indices，再用 `index/topK` 定位源卡 token 记录，从对应 peer window 拉量化数据、scales、可选权重。写到本卡紧凑 Dispatch GM，同时写 metadata：源rank、原 token 行、TopK 槽、可选权重位模式。

metadata 是以后 Combine 找回源卡目的地址的依据，不能只搬 token 而丢失路由。输入 fetch/store 用环形 UB：槽复用前等旧 MTE3，输入到齐后才允许输出；metadata 的 Scalar 写也通过事件排在 MTE3 读取之前。

代码：[FetchDispatchTokenAndMetaInfo](../arch35_92/stage/mega_moe_token_dispatch.h#L97)、[DispatchOwnedExpertRows](../arch35_92/stage/mega_moe_token_dispatch.h#L364)。

写完一段后，同一个 Dispatch AIV 先等本核 `MTE3_S`，确保实际 GM 数据已经写完；再按专家 M-group 将本段真实写入行数 AtomicAdd 到 `dispatchToGmm1Flag`。多个 Arch35 AIV1 / Arch92 唯一 AIV 的行段可以落在同一个 M256 组，它们分别加贡献，最终凑到该组实际行数。

AIC 对自己要读的组轮询等值：完整组等 256，末组等剩余行数，例如 25。它不是等“256 个核心”或“16 个 N tile”，而是等这一组的输入行全部搬到本卡。相同 M-group 的多个 N tile 复用同一次等待。

代码：[PublishGmm1TileReady](../arch35_92/stage/mega_moe_token_dispatch.h#L267)、[WaitForGmm1InputReady](../arch35_92/stage/mega_moe_gmm1_activation.h#L34)。

**行为一致：** token/scales/metadata 的拉取、写紧凑 GM、按真实行数发布 ready，以及 AIC 的组等待协议一致。Arch92 不因为 AIV 合用职责就多拉一次 token；可能不同的是同一 AIV 何时有空执行这段工作，而非本段每行新增了操作。

</details>

</details>

<details>
<summary><strong>8. 正式 Wave：分别按当前各核顺序展开</strong></summary>

<details>
<summary><strong>8.1 先规划边界：一个 Wave 可以停在专家内部</strong></summary>

**执行核：** 各核在自己的 Wave 调度状态中规划边界，不是一个 AIV 算完再通知全核。Dispatch 边界由 Arch35 AIV1 / Arch92 唯一 AIV 的 Dispatch 路径使用；GMM1/GMM2 边界由 AIC 及相应 Activation、权重展开、Combine 角色各自遍历。相同调度范围使用相同 count 和 M-group 预算，但各角色可以在墙钟时间上前后错开。

游标记录三个量：当前专家编号、该专家内部的行偏移、所有专家紧凑数据的全局行偏移。跳过零行专家，从剩余 M-group 预算取一段非空专家行；专家结束就转下一个，预算满则返回 Wave 终点。终点落在专家内部时保留行偏移，下个 Wave 从这个行接着做。

假设预算 4 个 M-group，专家行数 100、600、900：W0 取专家0的100行(1组)，再取专家1的600行(3组)，恰好满；W1 从专家2取最多1024行，这里900行(4组)就结束。若专家1是800行，W0只取它的768行，W1从该专家第768行继续。保存的是专家紧凑行偏移，不是原始源卡 token ID。

代码：[PlanNextExpertTokenRangeInWave](../arch35_92/common/mega_moe_utils.h#L406)、[AdvanceExpertTokenPositionInWave](../arch35_92/common/mega_moe_utils.h#L349)。

</details>

<details>
<summary><strong>8.2 Arch92：唯一 AIV 的 Dispatch/Activation/Combine 顺序</strong></summary>

进入 MoE 后，唯一 AIV 先读 count 和保存 cumsum；去重开时先做自己分到的 rowDesc 建表。然后初次 `PrepareDispatchWave` 搬 W0，再有后续范围就搬 W1。Dispatch 过程中已发布的 W0 M-group 可以被 AIC 提前消费，不必等 AIV 完成 W1。

接下来 AIV 调 `ProcessGmm1Wave(W0)`，这里它不算矩阵乘，而是按与配对 AIC 一致的 scheduler 等待 GMM1 结果，再做 Activation/量化。做完后按 lag 分支进入 GMM2/Combine，或先留下当前 GMM2 待处理边界。

稳态进入 W1 时，W1 已 Dispatch，AIV 先准备 W2，再做 W1 Activation；之后逐轮如此。Activation 覆盖过 cumsum UB 后，下次 Dispatch 先从 GM 恢复必要前缀。此版本是“完整下一 Wave Dispatch -> 当前 Wave Activation”的调用组织，不是原示例按每个专家交替。

无 lag 的单核顺序：

```text
AIV: Dispatch W0/W1 -> Activation W0 -> Combine W0
                   -> Dispatch W2    -> Activation W1 -> Combine W1
                   -> ... -> Activation 最后W -> Combine 最后W
AIC: GMM1 W0 -> GMM2 W0 -> GMM1 W1 -> GMM2 W1 -> ...
```

普通非量化 Combine 在 GMM2 调用内部按 tile 做；量化/去重是独立专家粒度 Combine，专家未完整结束时没有相应独立发送。图中的 `Combine W` 表示消费该调度范围允许消费的结果，不保证每个 Wave 都产生一次完整专家发送。

代码：[PrepareDispatchWave](mega_moe_wave.h#L491)、[当前 MoE 主循环](mega_moe_wave.h#L526)、[ProcessGmm1Wave](mega_moe_wave.h#L253)。

**行为差异与潜在性能影响：** 对照 Arch35 A8W8，下一 Wave Dispatch 在 AIV1、当前 Activation 在 AIV0，二者可同时推进；Arch92 同一 AIV 先做完整下一 Wave Dispatch，再做当前 Activation，后续又由该核 Combine。lookahead 仍可让 AIC 当前 GMM1 与 AIV 的下一 Wave Dispatch 重叠，但**同一 AIV 上的 Dispatch 与 Activation 不能并行**。若下一 Wave 搬运久，当前 Activation ready 会晚，AIC 可能在转 GMM2 时多等；若当前 Activation/Combine 久，后续 Dispatch 也会晚。这个关键路径变化才是差异，不能只说“少一半 AIV”。对照 Arch35 A8W4，通信/Activation 本来已在 AIV1，额外变化是下一节的整 Wave 与逐专家 slice 排序。

</details>

<details>
<summary><strong>8.3 Arch35 A8W8：AIV0 计算，AIV1 提前 Dispatch</strong></summary>

AIC 和 AIV0 遍历当前 Wave：AIC 做 GMM1；AIV0 等输出并做 Activation。AIV1 在 `ProcessGmm1Wave` 的角色分支直接返回，再进入 `PrepareDispatchWave`，首次准备 W0/W1，稳态准备下一 Wave。

所以从各核看：

```text
AIC : 等 Dispatch ready -> GMM1 当前W -> 按lag安排 GMM2
AIV0: 等 GMM1 ready    -> Activation 当前W -> 下一个调度阶段
AIV1: Dispatch 预备W   -> 在 GMM2 调用里 Combine，或做独立专家 Combine
```

AIV1 虽然不做 GMM1，普通逐 tile Combine 需要与 AIC 的滚动分核游标一致，因此额外按已跳过的 GMM1 tile 数推进 `startBlockIdx_`。Arch92 的 AIV实际做 Activation，其 GMM1 调用自然推进游标，不能照抄 AIV1 的跳过补偿，否则推进两次。

TopK 权重预取时 AIC 写 GMM1 GM，AIV0 仍然是 Activation 核，只是从 GM 读而不是直接消费 UB；不能因为“结果写 GM”就把它说成 AIV1 Activation。

代码：[ProcessGmm1Wave](../arch35/mega_moe_wave_a8w8.h#L228)、[AdvanceStartBlockIdxForSkippedGmm1](../arch35/mega_moe_wave_a8w8.h#L200)、[PrepareDispatchWave](../arch35/mega_moe_wave_a8w8.h#L494)、[主循环](../arch35/mega_moe_wave_a8w8.h#L684)。

**对照说明：** 这里独立 AIV0/AIV1 的向量重叠在 Arch92 不再存在，性能风险见 8.2；但 AIC 的输入依赖和 lookahead 设计未被取消。TopK 预取两边都走 GMM1 GM，不能在该场景再把 Arch92 的 GM 结果读回算成相对 Arch35 新增的成本。

</details>

<details>
<summary><strong>8.4 Arch35 A8W4：AIV0 展开权重，AIV1 交替 Dispatch/Activation</strong></summary>

首次 AIV1 先 Dispatch W0。正式循环中，AIV1 不再一口气只做当前 Activation：它先取下一 Wave 一个专家 slice 做 Dispatch，再取当前 Wave 一个专家 slice 等 GMM1、做 Activation；两条游标分别推进，哪边到终点就只做另一边。

AIC 同期只按当前 Wave slice 做 GMM1；AIV0 按相同 tile 分配展开当前 GMM1 权重并送 L1。AIC 的 GMM1 输出写 GM，AIV1 从 GM 读回后做 Activation。下一 Wave Dispatch 与当前 GMM1/Activation 交叠，因此当前 AIC 不会等“下一 Wave 全部 Dispatch 完”才开始。

当前 Wave GMM1/Activation 走完，按 lag 决定当前/上一 Wave GMM2：AIV0 再展开 W2，AIC 做 GMM2，AIV1 Combine。三个角色采用一致的 lag 调度次序，不能只有 AIC 延后。

代码：[DispatchFirstWave](../arch35/mega_moe_wave_a8w4.h#L125)、[ProcessNextDispatchAndCurrentGmm1](../arch35/mega_moe_wave_a8w4.h#L179)、[RunMoeExpertLoop](../arch35/mega_moe_wave_a8w4.h#L329)。

**行为差异与潜在性能影响：** Arch35 A8W4 的 AIV1 已兼做 Dispatch、Activation、Combine；Arch92 不是把该 AIV1 与权重展开 AIV0 的向量任务简单合并，当前支持类型的 MMAD 由 Tensor 直接承担，唯一 AIV 不做旧 prologue。调度上，Arch35 A8W4 启动仅 Dispatch W0，稳态按下一 Wave 的一个专家 slice / 当前 Wave 的一个专家 slice 交替；Arch92 启动准备 W0/W1，稳态先做完整下一 Wave Dispatch 再做当前 Activation。后者可更早备齐下一 Wave，但可能延后当前 Activation，尤其下一范围包含很多专家或搬运耗时长时；旧版逐 slice 交替也可能因等待当前 Activation 中断后续 Dispatch。谁更有利要结合输入 ready、Activation ready 及 AIC 等待测量，不能从模板来源断言顺序完全相同。

</details>

<details>
<summary><strong>8.5 Arch35 A4W4 与 Arch92 A4W4 不能按名字当成同一分工</strong></summary>

Arch35 A4W4 的第一段是直接 FP4×FP4 Generic：AIC GMM1，AIV0 Activation，AIV1 Dispatch。Activation 当前量化成 FP8，所以第二段实际是 FP8×FP4，复用本仓 A8W4：AIV0 展开 W2，AIC GMM2，AIV1 Combine。

该 Wave 文件沿用了“下一 Wave Dispatch / 当前 Wave GMM1”的组织，但不能仅凭注释断定 AIV1 做 Activation：`RunGmm1Generic` 会让非计算向量角色返回，真实 Activation 核是 AIV0。当前 A4W4 主循环每轮紧跟本轮 GMM2，没有 A8W8/A8W4 的 token 阈值 lag 分支。

代码：[A4W4 GMM1 调用](../arch35/mega_moe_wave_a4w4.h#L117)、[Generic 角色判断](../arch35_92/stage/mega_moe_gmm1_activation.h#L647)、[A4W4 主循环](../arch35/mega_moe_wave_a4w4.h#L344)。

Arch92 A4W4 目前 Activation 输出 FP4，两段都是 Tensor Generic FP4×FP4，唯一 AIV做 Dispatch、Activation、Combine，调度与 Arch92 A8W8 共用。将来 A8W4 Tensor 接口支持后，若改为与 Arch35 一样让 Activation 输出 FP8，则第二段类型及对应策略也要切换；不能仅改变 GMM1 输入类型。

代码：[FP4_SWIGLU_OUTPUT 策略](../arch35_92/common/mega_moe_execution_policy.h#L24)、[Arch92 GMM2 类型](mega_moe_wave.h#L334)。

**行为差异与潜在性能影响：** 当前 A4W4 同时改变了 Activation 工作者、Activation 输出精度、GMM2 输入类型和权重准备实现，不是只把 AIV0 改名为唯一 AIV。Arch35 第一段可让 AIV1 Dispatch 与 AIV0 Activation 重叠，第二段还有 AIV0 W2 展开；Arch92 第一段向量工作串行，第二段无旧展开链。减少展开/中间数据量与减少向量重叠可能方向相反，且 Tensor 两种矩阵输入类型吞吐未知，不能将总耗时变化全部归因于 CV 配比。量化字节数和 lag 差异分别在 10.3、8.6 展开。

</details>

<details>
<summary><strong>8.6 GMM2 lag：为什么隔一轮，以及怎样收尾</strong></summary>

**执行核：** lag 选择和待处理边界保存在各核本地，由同一路径的 AIC 与向量角色共同遵循。Arch35 A8W8 中 AIC 做 GMM、AIV0 做 Activation、AIV1 做 Dispatch/Combine；Arch35 A8W4 中 AIV0 改做权重展开、AIV1 兼做 Activation；Arch92 由 AIC 与唯一 AIV 执行。下面的 G1/G2 箭头是阶段次序，实际矩阵乘只有 AIC 执行。

Arch35 A8W8/A8W4、Arch92 在本卡原 token 数达到 `GMM2_LAG_MIN_TOKEN_NUM=4096` 时启用 lag。判据是输入 tokenNum，不是某专家实际接收行数，也不是 Wave 数。

初次当前 GMM1 W0 后，不立刻消费 GMM2 W0，而是缓存 W0 终点。再做 GMM1 W1 后做 GMM2/Combine W0；再做 GMM1 W2 后消费 W1。循环退出后必须补跑最后缓存的 GMM2/Combine，否则最后一个 Wave 丢输出。

```text
无lag: G1 W0 -> G2 W0 -> G1 W1 -> G2 W1 -> G1 W2 -> G2 W2
有lag: G1 W0 -> G1 W1 -> G2 W0 -> G1 W2 -> G2 W1 -> G2 W2(收尾)
```

目的是让下一 Wave 的 GMM1覆盖上一 Wave 尾部 Activation 的时间，降低 AIC 紧跟 GMM2 时等 Activation 的自旋。实际 ready 依赖仍然存在：GMM2 不能读未量化完成的数据。各角色按同一调用序延后，不要求他们墙钟时刻完全相同。

Arch92 在此只沿用调度策略；源码明确保留“收益需在 A6 单独测量”的限定，不能从代码推导实际性能一定更好。

代码：[Arch92 lag 和收尾](mega_moe_wave.h#L573)、[Arch35 A8W8 lag](../arch35/mega_moe_wave_a8w8.h#L672)、[Arch35 A8W4 lag](../arch35/mega_moe_wave_a8w4.h#L329)。

**相同策略与真正差异：** 对照 Arch35 A8W8/A8W4，阈值和整体 lag 调用序一致，不是 Arch92 新增了一套 GMM2 延迟协议；改变的是唯一 AIV 的任务负载，原 Arch35 的收益不能直接外推。对照当前 Arch35 A4W4，则存在实际差异：Arch35 该模式无阈值 lag，Arch92 A4W4 共用统一 Wave 并在达到阈值时开启。它可能减少 AIC 等尾部 Activation，但也增加待消费范围及收尾延迟；应分别看稳态与 drain，不能仅看循环内 GMM2 开始得更晚就判为退化。

</details>

<details>
<summary><strong>8.7 GMM 分核滚动游标：不是所有核都做所有 tile</strong></summary>

**执行核：** AIC 根据逻辑编号选本核矩阵乘 tile；对应 AIV 用相同 scheduler 选本核 Activation、权重展开或普通 Combine tile。Arch35 Generic GMM1 配 AIV0，自定义 A8W4 GMM1 配 AIV0 展开/AIV1 Activation，GMM2 普通 Combine 配 AIV1；Arch92 两段都配唯一 AIV。跳过某阶段的角色也按该路径规则推进自己的游标，不能把“没有数据工作”理解成“没有调度状态”。

每个 problem 先算 `tileCount=ceil(M/256)*ceil(N/256)`。若该 problem 从逻辑核 s 开始，逻辑核 b 的第一个 tile 序号为 `(b-s+P)%P`；随后取该序号加 P、再加 P，只要小于 tileCount 就做。problem 结束时每个角色都将 s 更新为 `(s+tileCount)%P`，即使本核没有 tile 也要推进。

举例 P=64、N1=2048、H=4096、s=0：专家0 GMM1 M=256，共8 tile，核0..7各做1个，s变8；专家1 GMM1 M=300，共16 tile，核8..23各做1个，s变24。若随后调度专家0 GMM2，N=H=4096，共16 tile，核24..39做，s变40。这里 GMM2 不是沿用 GMM1 的8个 N tile，因为两段 N 不同。

再来一个 problem M=10000、N=2048，共40*8=320 tile。若 s=40，每核首个序号是 `(b-40+64)%64`，每核取5个序号。这说明大 problem 可以让同一核多次循环；若受 Wave 预算截断，必须用本次 slice 的 M 重新算 tileCount，不能直接拿整个10000行来分。

tile 序号不是简单 row-major 坐标，`BlockSchedulerSwizzle` 再把它变成 M/N 坐标，按 swizzle 分组及方向决定顺序。C/AIV 重放同一 scheduler 是配对 tile 通知正确的前提。

代码：[HandleWaveProblemWithoutWork](../arch35_92/common/mega_moe_utils.h#L165)、[Generic 起始循环和游标推进](../arch35_92/stage/mega_moe_gmm1_activation.h#L655)、[GetBlockCoord](../arch35_92/blaze/gemm/block/block_scheduler_swizzle.h#L70)。

Arch35 A8W8 和 Arch92 还检查一个特殊情况：整个 GMM1+GMM2 调度后游标恰好绕回 Wave 起点，但 GMM1结束点不是起点，则下个 Wave改从 GMM1结束点开始，避免固定核持续担当同类任务。这个调整在相关角色一致的主循环位置执行，不是对某个核单独偏移。

代码：[Arch35 游标调整](../arch35/mega_moe_wave_a8w8.h#L715)、[Arch92 游标调整](mega_moe_wave.h#L613)。

**行为一致的算法与角色差异：** tile 归属、滚动分核及无 tile 时推进游标的算法一致。Arch35 A8W8 AIV1 跳过 GMM1，需补游标；Arch92 唯一 AIV 实际做 Activation，正常调用自然推进，不能再补一次。少掉这段补偿是职责变化的结果，不是重新发明分核或保证性能更好；其 Scalar 开销相对 Dispatch/Activation 重叠变化通常不宜单独作为性能结论。

</details>

</details>

<details>
<summary><strong>9. GMM1：AIC 做矩阵乘，结果交给哪个 AIV</strong></summary>

<details>
<summary><strong>9.1 Generic 调用、Gate/Up 排布与 MMAD context</strong></summary>

**执行核：** Generic 的矩阵乘、MMAD context 初始化/参数更新只由 AIC 做；Arch35 MoE/共享 A8W8/A4W4 的 Activation 由 AIV0 做，Arch92 当前支持模式的 Activation 由唯一 AIV 做。Arch35 AIV1 不执行 Generic Activation，只返回或做必要的游标补偿；自定义 A8W4 另见 9.4，不在这里。

执行 Generic 的 AIC 和 Activation AIV 先各自用专家 slice 构造 scheduler，求自己的 tile 序号。每个 tile 切出相应张量视图；其中 A、B、ScaleA、ScaleB 的矩阵输入由 AIC 读取，Activation AIV 读取的是 GMM1 输出（预取时另读 metadata）。视图构造只计算地址，AIC 真正读输入前由前述 Dispatch ready 约束；Activation AIV 真正读输出前由 GMM1 ready 约束。

GMM1 用 QGMM `GroupedMatmulWithScaleMx`，Gate/Up 按 concat/interweave 输出到一个逻辑 N256 tile；底层 GMM1接口以单半边 N128描述，配合成对 Gate/Up 权重视图，最终交接给 Activation 的一行是 `[gate片段,up片段]`。Activation 将完整 N256 变成 F 方向的 N128，不是把整个专家全宽 Gate、全宽 Up 原样当作两个无关 problem。

不能把 scheduler 的256列、传入 MMAD的单半边128列、最终 Activation的128列混为一谈。ND/NZ 决定 B/scales视图和搬运布局，不改变专家输出数学宽度。

代码：[Generic 配置及 QGMM 类型](../arch35_92/common/mega_moe_gmm_common.h#L94)、[Gmm1AicMmadTileToGmGeneric](../arch35_92/stage/mega_moe_gmm1_activation.h#L75)、[NZ Gate/Up concat 搬运](../arch35_92/blaze/gemm/tile/copy_gmm1_concat.h#L107)、[结果有效范围与UB/GM目的选择](../arch35_92/blaze/gemm/tile/copy_gmm1_concat.h#L122)。

传入可复用 context 时，先按 K、tile 和 L1 参数比较配置，必要时 Init，并更新当前动态 problem。没传 context 则本调用建立局部对象。Arch92 从 Wave成员取类型匹配对象，Arch35 Generic 默认使用局部对象；这改变的是初始化生命周期，不是把第一次专家 M 固定到所有后续专家。

代码：[BlockMmadContext](../arch35_92/common/mega_moe_gmm_common.h#L393)、[Gmm1ExecGeneric 的局部/外部 context 分支](../arch35_92/stage/mega_moe_gmm1_activation.h#L476)。

**软件策略差异与潜在性能影响：** Arch92 的 AIC 传成员 context，Arch35 Generic 默认不传而建立局部 context。相邻专家 problem 的配置签名相同时，Arch92 可跳过 `blockMmad.Init`，但仍执行动态参数更新；GMM1 单半边 N128 与 GMM2 N256 等配置切换可导致同一个复用对象重新 Init。因此可能减少的是重复初始化及相关 Scalar 开销，不是 GMM tile 数或输入字节数，也不是全流程只 Init 一次。这项差异并非 1C1V 的必然要求，应与 CV 职责重排分开评估。

代码：[Arch92 取得实际 context 的条件](mega_moe_wave.h#L99)、[Arch35 GMM1 显式传空 context](../arch35/mega_moe_wave_a8w8.h#L295)。

</details>

<details>
<summary><strong>9.2 Arch35 UB 交接：双 buffer 为什么必须回通知</strong></summary>

默认非预取 Generic GMM1：AIC 第一个 tile写 ping、第二个写 pong；FIX 发 ready，AIV0 的 V 等 ready 后计算 Activation。AIV0 用完对应输入槽后通过 V 发 ACK，AIC 的 FIX 在第三次及以后的槽复用前等 ACK。

ready 解决“不能读还没写完的 tile”；ACK 解决“不能覆盖 AIV0 还在读的同一 UB 槽”。两个方向保护的是不同风险。前两个 tile 可直接填空槽，后面不能无条件一直写。阶段末 AIC 要收最后一到两个 ACK，即使没有再复用槽，也要确认最后两份已经消费、事件不留到下一阶段。

此处 ready 使用 mode4 的4/5，ACK用6/7，两个id对应 ping/pong。生产者为 AIC/FIX、消费者为 AIV0/V；反向生产者为 AIV0/V、消费者为 AIC/FIX。与输入阶段的全 AIV通知不是同一作用域。

代码：[Gmm1AicMmadGeneric](../arch35_92/stage/mega_moe_gmm1_activation.h#L180)、[Gmm1UbActivationSync](../arch35_92/common/mega_moe_gmm_epilogue_sync.h#L77)。

**与 Arch92 的交接差异：** 当前 Arch92 不走此 UB ping/pong，而是 GM-handoff，协议和性能代价见 9.3。并不是“Arch92 不需要同步”或“删除 AIV0 的 ACK 就更快”：它用另一地址生命周期和 ready 协议保证结果可读。只有 Arch35 原来走 UB 的非预取 Generic/共享分支，才产生改 GM 的新增结果流量；已走 GM 的分支不在此列。

</details>

<details>
<summary><strong>9.3 GM 交接：当前哪些用硬件事件，哪些用 GM 序号</strong></summary>

**执行核：** GM 结果始终由分到 tile 的 AIC 写出，实际读回并做 Activation 的 AIV 如下表。没有分到 tile 的配对核不凭空发/等一次 tile 事件；非 Activation 角色也不消费这条 ACK 链。

同一个 `Gmm1GmActivationSync` 根据调用者是否挂了硬件对象决定协议，不能看函数名就说“所有 GM-handoff 都一样”。当前 MTE 调用点如下：

| 场景 | 消费者 | 实际 GMM1->Activation ready |
|---|---|---|
| Arch35 MoE A8W8/A4W4，TopK预取开 | AIV0 | 挂 `gmm1ActivationSync`，mode4硬件ready/ACK |
| Arch35 MoE A8W4 | AIV1 | 挂对象，mode4硬件ready/ACK |
| Arch35 共享 A8W4 | AIV1 | 独立共享mode4事件链 |
| Arch92 MoE，TopK预取开 | 唯一AIV | 挂对象，mode4硬件ready/ACK |
| Arch92 MoE，TopK预取关 | 唯一AIV | **没挂对象，使用每逻辑block的GM递增序号** |
| Arch92 共享 A8W8/A4W4 | 唯一AIV | 显式挂共享对象，独立共享mode4链 |

Arch92 非预取 MoE：AIC 每写好一个 GMM1 tile，先 `FIX_S` 确认写完，再把自己的 GM序号加1；AIV为下一个 tile求 `expectedSequence=本地已消费序号+1`，轮询GM值 `>=expectedSequence` 后搬入。允许大于，是因为AIC可先写完后续不同GM tile；GM结果不复用同一个ping/pong UB槽，不需要按UB槽ACK限速。AIV消费后更新本地序号，此软同步分支不产生硬件ACK。

代码：[Arch92 仅预取时挂MoE对象](mega_moe_wave.h#L554)、[配置每block GM序号地址](../arch35_92/stage/mega_moe_gmm1_activation.h#L822)、[Gmm1GmActivationSync 的硬件/GM分支](../arch35_92/common/mega_moe_gmm_epilogue_sync.h#L197)。

挂硬件对象时：AIC FIX发ready，AIV MTE2等ready；AIV完成整个逻辑tile的所有Activation子块及GM输出后MTE3发ACK。AIC最多保留15个未收ACK的tile，提交第16个前Scalar消费一个ACK；该限制保护硬件事件积压，不是全核屏障，也不是每次提交都锁步等待。

mode4 GMM1 MoE事件为8/9，共享为12/13；Arch35 AIC侧加16选择AIV1时，消费者仍使用本地id。Arch92选唯一AIV不加16。未挂对象的预取后备分支还存在 per-tile GM tag，但不应把它当成上述已挂对象的 MTE 主路径。

代码：[Gmm1ActivationSync 的credit与EndSync](../arch35_92/common/mega_moe_gmm_epilogue_sync.h#L135)、[当前Arch35 A8W8挂对象](../arch35/mega_moe_wave_a8w8.h#L650)。

**行为差异与潜在性能影响分三类：**

1. **非预取 MoE Generic：** Arch35 AIC→AIV0 UB 的两槽 ready/ACK，变成 Arch92 AIC→GM→唯一 AIV，并用 GM 递增序号。对实际 M×N1 个 BF16 结果，增加一次 GM 写和一次读回，理想有效数据量为 `2 * sizeof(BF16) * M * N1`，实际搬运还受对齐/尾块影响。AIV 增加结果 MTE2 搬入，AIC 每 tile 经 FIX_S 写序号，AIV 轮询序号；若唯一 AIV 正忙于 Dispatch，轮询和消费开始时间都可能变化。代价是 GM 带宽、装载延迟及轮询；可能的收益是 AIC 不再为了复写两槽 UB 等 ACK，可以先写不同 GM tile。无 ACK 不等于没有额外成本。
2. **TopK 预取 MoE：** Arch35 Generic 已写 GM 并使用硬件 ready/ACK，Arch92 仍然如此。没有上述额外一次结果 GM 写/读，也没有把硬件通知改成软轮询；真正变化是消费核由独立 AIV0 变为还负责 Dispatch/Combine 的唯一 AIV。若它消费不及时，15-credit 用满后 AIC 仍会等 ACK。
3. **共享 Generic：** Arch35 共享 A8W8/A4W4 原为 UB，Arch92 改 GM，但显式使用共享硬件事件链，不走非预取 MoE 的 GM 序号轮询。因此只可列新增结果搬运和两槽约束变化，不能再附加“共享也反复读 GM ready”这个不存在的成本。

以上均需区分 AIC 的矩阵计算时间、FIX/MTE 搬运及等待时间。特别在模拟环境，GM 轮询重试的指令成本可能放大，不能把模拟耗时比例直接当成芯片带宽比例。

</details>

<details>
<summary><strong>9.4 Arch35 A8W4 内部：AIV0 权重展开与 AIC 的 L1 握手</strong></summary>

AIV0按本核scheduler取B的FP4块：MTE2搬到UB，V做FP4到FP8表示展开，MTE3写到AIC使用的B L1槽。这个转换本身不把ScaleB乘进去；矩阵乘仍要读ScaleA和ScaleB，FP4展开为可表示同值的FP8不能理解成取消MX scales。

AIC在对应K块的MTE1读B L1之前等AIV0的权重ready。读取消费后，AIC在MTE1发槽释放，AIV0的MTE3等该释放才复写L1。主循环为一个K_L1块分两半组织GM加载/向量展开，UB使用四槽，L1使用配置中的B槽数。

跨核mode4权重ready id1：AIV0/MTE3发，AIC/MTE1等；槽释放id0：AIC/MTE1发，AIV0/MTE3等。构造/析构还预置并排空buffer事件，不能只保留ready而去掉反向释放。

代码：[ComputeBasicBlockAivNdKnNzNk](../arch35_92/blaze/prologue/block_prologue_mx_fp8fp4.h#L259)、[权重ready与槽释放](../arch35_92/blaze/prologue/block_prologue_mx_fp8fp4.h#L329)、[AIC对应等待/通知](../arch35_92/blaze/gemm/block/block_mmad_mx_fp8fp4.h#L599)。

AIC侧还使用本核MTE2→MTE1、MTE1→M、M→MTE1等事件保护A/scales的L1和L0A/L0B轮转，完成K方向累加后FIX写结果。AIV0只负责B展开，不做SwiGLU；AIV1等GM输出后做SwiGLU。Arch92当前直接Generic路径没有这条独立AIV0权重链，不能迁入其两个子角色等待。

代码：[AIC L1/K循环](../arch35_92/blaze/gemm/block/block_mmad_mx_fp8fp4.h#L528)、[A8W4三角色分流](../arch35_92/stage/mega_moe_gmm1_activation.h#L565)。

**与 Arch92 当前实现的差异：** Arch92 当前支持的 A8W8/A4W4 Generic 不执行本仓 AIV0 权重展开、展开后 B 的 L1 写入以及 AIC/AIV0 的跨核权重槽握手。这些软件准备成本减少，但 Tensor 内部仍有自己的权重装载/矩阵流水，不是“权重无需搬运”。且 Arch35 A8W8 本来就没有此展开链；当前 Arch92 A8W4 被编译门控，不能拿未运行的直接 FP8×FP4 路径声称替代旧 A8W4 后性能已提升。能实际对比的 A4W4 第二段，还同时改变了 A 的 FP8/FP4 类型，见 10.3。

</details>

</details>

<details>
<summary><strong>10. Activation/SwiGLU：读什么、算什么、何时允许 GMM2 开始</strong></summary>

<details>
<summary><strong>10.1 一块 Activation 的完整顺序</strong></summary>

**执行核：** MoE 中 Arch35 A8W8/A4W4 是 AIV0、Arch35 A8W4 是 AIV1，Arch92 当前 A8W8/A4W4 是唯一 AIV；共享阶段按共享量化模式采用同样的角色对应。下面的等待、搬结果、激活、量化、写 GM 和发布 Activation ready 全由这些有 tile 的 AIV 做。AIC 只生产 GMM1、等待 GMM2 输入，不执行 SwiGLU。

这些 Activation AIV 先等配对 AIC 的GMM1 tile-ready。UB-handoff直接消费输入槽；GM-handoff先MTE2搬结果，等MTE2_V后才计算。每行从concat结果取Gate和Up，执行选定gated activation；输出宽度是输入Gate+Up的一半。

然后在UB将Activation中间结果按MX分组求max exponent、产生E8M0 scales和逆scale，生成FP8或FP4打包数据。V_MTE3后写Activation data/scales到GM，再等MTE3_V和MTE3_S；此时才累计 `activationToGmm2Flag`，并按交接协议通知GMM1消费完成。

不能说“收到GMM1 flag后启动GMM1”：AIV此时启动的是Activation，不是矩阵乘。不能把Activation GM写出前的V计算完成当成GMM2输入ready。

代码：[GM结果装载](../arch35_92/stage/mega_moe_gmm1_activation.h#L350)、[Activation主调用](../arch35_92/blaze/epilogue/block_epilogue_activation_mx_quant.h#L413)、[NotifyGmm2InputReady](../arch35_92/stage/mega_moe_gmm1_activation.h#L61)。

**数学和发布顺序一致，工作归属不同：** 读 Gate/Up→激活→量化→写 data/scales→发布 GMM2 ready 的顺序相同。Arch35 A8W8/A4W4 的独立 AIV0 工作转由 Arch92 通信兼用的 AIV 做，导致向量重叠变化；输入从 UB 变 GM 时还多结果装载。二者的性能影响分别见 8.2 和 9.3，不应再在本节当成另一套新增激活算法重复计费。

</details>

<details>
<summary><strong>10.2 SwiGLU等激活模式与TopK权重在哪乘</strong></summary>

**执行核：** 激活数学及量化前乘 TopK 权重，由 Arch35 MoE A8W8/A4W4 的 AIV0、A8W4 的 AIV1，或 Arch92 唯一 AIV 执行；不是 Dispatch 发送核或 AIC 在这里乘。共享 Activation 仍按共享模式分核，但不乘 MoE TopK 权重。

默认SwiGLU核心是 `gate/(1+exp(-gate))*up`，前面有源码规定的clamp：普通SwiGLU对Gate做上界clamp，Up做双向clamp；STEP分支改变clamp作用位置。OAI分支使用alpha、beta；SITU及其LINEAR子模式用独立的SiTU运算。它们只替换V核的激活数学，不改变Dispatch、GMM ready或Combine的参与角色。

所有激活模式都在同一个epilogue中选择，之后复用量化和写出。具体数值表达、常量及尾部mask应看对应VF，不把所有模式简写为普通SwiGLU公式。

代码：[RunGatedActivationTile](../arch35_92/blaze/epilogue/block_epilogue_activation_mx_quant.h#L305)、[SwiGLU VF](../arch35_92/blaze/epilogue/activation/swiglu_activation.h#L46)、[OAI VF](../arch35_92/blaze/epilogue/activation/swigluoai_activation.h#L50)、[SiTU VF](../arch35_92/blaze/epilogue/activation/situglu_activation.h#L85)。

TopK预取开时，Dispatch metadata携带本路由的weight。Activation核取对应weight，在激活结果量化前乘进去；GMM2是线性变换，因此最终Unpermute不再重复乘。共享专家固定关闭TopK预取，它的输出最后直接相加。

TopK预取还将M256结果拆成最多两个M128子块处理，分别贡献到同一个M256 Activation flag，整个逻辑tile处理完才归还一次GMM1硬件ACK。不开预取则按整块贡献一次。

代码：[Gmm1Aiv0PrefetchEpilogueTileGeneric](../arch35_92/stage/mega_moe_gmm1_activation.h#L262)、[SwiGLU读取并乘metadata权重](../arch35_92/blaze/epilogue/activation/swiglu_activation.h#L109)。

**行为一致：** 相同 actMode、预取开关、量化类型下，两边复用同一激活数学、clamp 和乘权位置；M128 子块/完成贡献也是同一协议。不能将换了执行 AIV 说成多乘一次 TopK，或把 STEP/OAI/SiTU 当成 Arch92 迁移额外加入的计算。当前 A4W4 的量化类型不同另见 10.3。

</details>

<details>
<summary><strong>10.3 FP4/FP8输出与GMM2等待目标</strong></summary>

**执行核/同步双方：** Arch35 A8W8/A4W4 的 AIV0、A8W4 的 AIV1，以及 Arch92 唯一 AIV，完成对应 Activation data/scales 写出后累计块数；GMM2 的 AIC 在读本 M-group 前等待这个计数。共享阶段按共享模式选择 Activation AIV，写的是独立共享计数。FP4/FP8 量化在这些 AIV 上，AIC 不做量化。

epilogue已有FP8和FP4两个编译期量化分支；“SwiGLU支持FP4”准确地说是本仓这套Activation+MX量化epilogue支持FP4输出，并非SwiGLU数学本身有另一套FP4公式。Arch35 A4W4当前输出FP8，Arch92 A4W4当前输出FP4；两者GMM2输入类型跟随这里。

代码：[RunMxQuantTile](../arch35_92/blaze/epilogue/block_epilogue_activation_mx_quant.h#L337)。

Wave模式GMM2对每个M256组等待：

```text
targetLoops = ceil(该组实际行数 / activationTileM) * ceil((GMM2.K*2) / 256)
            = Activation子行块数 * GMM1完整Gate+Up的N tile数
```

比如N1=512、不预取，M=25尾组，目标1*2=2；预取时activationTileM=128，该尾组仍是1*2=2；完整M256组在预取模式则是2*2=4。只有整个F输入列和对应scale写完，GMM2才可沿K方向读全输入。GMM1一个N tile完成，不能代表GMM2的完整K输入已完成。

非Wave-grained调用则等待专家范围的全部Activation tile总数；当前MTE MoE和共享调用使用M-group粒度。wait是等值，重复发布或用错子块数会永远等不到，不能把异常大于目标当成正常完成。

代码：[WaitForGmm2InputReady](../arch35_92/stage/mega_moe_gmm2_combine.h#L495)。

**A4W4 的真实类型差异及潜在性能影响：** Arch35 的 Activation 输出 FP8，GMM2 实际 FP8×FP4且由 AIV0 展开 W2；Arch92 输出 packed FP4，GMM2 直接 FP4×FP4。同为 M×F 个有效输出时，FP4 data 理想存储字节数是 FP8 的一半，可减少 Activation GM 写及 GMM2 的 A 读取；**MX scale 按逻辑分组数，不随 data 一起减半**。FP4 量化/打包、Tensor 两种 MMAD 输入类型的效率又不同，所以字节变少不保证总耗时变短。这项软件选择也改变量化误差，不能归为单纯换核；后续切回 FP8 Activation 时需重新评估。

**等待协议一致：** 两边 GMM2 都等完整逻辑 K 输入及 scales，就绪目标由 M/N tile 和预取子块决定，不因 FP4 每字节两个值而把 targetLoops 除 2。若 AIC 等待更久，需检查生产 AIV 的排队/量化/写回，不能仅根据 bit 数减少就预测等待减少。

</details>

</details>

<details>
<summary><strong>11. GMM2/Combine：矩阵乘和回源卡发送怎样衔接</strong></summary>

<details>
<summary><strong>11.1 AIC做GMM2的次序和分工</strong></summary>

沿已完成GMM1的Wave边界取对应专家slice。AIC按滚动scheduler取M/N tile，切Activation A/ScaleA、W2/ScaleB和BF16输出GM；第一次读该M-group前等上一节Activation计数。输出N=H、输入K=F，不能沿用GMM1的N宽度。

Arch35 A8W8、Arch92 A8W8/A4W4用Generic。Arch35 A8W4以及当前Arch35 A4W4第二段用自定义A8W4：AIV0展开W2到L1，AIC等权重；AIV1消费结果。普通GMM2 ready和W4权重ready是两条不同链，AIV0不参与GMM2->Combine配对ACK。

代码：[Gmm2AicMmadGeneric](../arch35_92/stage/mega_moe_gmm2_combine.h#L552)、[Gmm2AicMmadA8W4](../arch35_92/stage/mega_moe_gmm2_combine.h#L664)、[W4基类GMM2调度](../arch35_92/mega_moe.h#L801)。

**行为一致与差异边界：** A8W8 两边都是 AIC 等 Activation 后做 Generic GMM2并写 BF16 GM，矩阵乘没有迁给唯一 AIV；潜在差异来自输入 ready 的到达时机及 context 复用，见 8.2、9.1。A8W8 的权重 L2 提示策略相同，见 17.4。A4W4 则确有第二段输入类型及 prologue 的变化，见 10.3，不能与 A8W8 归成同一个比较。

</details>

<details>
<summary><strong>11.2 普通非量化、非去重Combine：配对核逐tile通知</strong></summary>

AIC提交GMM2 tile前预留一个硬件credit，写完GM通过FIX发mode4 ready；Arch35对应AIV1、Arch92唯一AIV在MTE2等ready后，从GM搬该tile及metadata。

AIV按每行metadata找源卡和原token/topK槽，将当前N列片段写回源卡：目的行=`tokenIdx*topK+slot`，目的列=nLoc。不同N tile可以并行写同一行的不重叠片段。不是每处理一个tile就必须把整行H列全部搬到UB。

本tile发送结束后，AIV等MTE3_MTE2保护下一次复用UB，然后MTE3发ACK。AIC最多保留15个未收ACK的tile，第16个前Scalar等ACK；ready/ACK使用10/11，Arch35 AIC侧加16指向AIV1，Arch92不加。末尾回收剩余ACK。

代码：[Gmm2CombineSync](../arch35_92/common/mega_moe_gmm_epilogue_sync.h#L292)、[CombineTokenRange](../arch35_92/stage/mega_moe_gmm2_combine.h#L623)、[CombineTokens的远端地址](../arch35_92/stage/mega_moe_gmm2_combine.h#L106)。

AIV重放与AIC相同的scheduler、同一startLoopIdx和步长，收到的第几个事件就对应自己的第几个tile。因此普通路径不需要再重建一个按行平均分核的独立Combine阶段。这是同步正确性要求，不仅是性能优化。

**协议和工人数一致，前序负载有差异：** 固定 P 时，Arch35 P 个 AIV1与Arch92 P 个唯一 AIV 都逐 tile 搬回传，使用相同 15-credit ready/ACK；AIC侧选子角色的 +16 地址规则不同不是多一次数据操作。对照 Arch35 A8W8，Arch92 的消费 AIV 先前还做 Activation，可能更晚进入 Combine，从而在额度用满时使 AIC 等 ACK；对照 Arch35 A8W4，AIV1 本来就兼做 Activation，不能将这项负载增加再次算进去。

</details>

<details>
<summary><strong>11.3 量化/去重Combine：先等完整专家，再按行发送</strong></summary>

**执行核：** 专家 ready 由全部 P 个 AIC 发布。稳态等待 ready、读整行、量化/去重合并、回传的是 Arch35 AIV1 / Arch92 唯一 AIV 中分到行任务的核；Arch35 最后一个非空专家还允许 AIV0 加入。AIV0 的加入条件及分核在 11.4 就地展开，不能理解成每个专家都由全部 AIV 发送。

量化需要拿到完整H行才能按MX分组量化；去重需要同卡多个专家成员的结果一起合并。因此这两类路径关闭逐tile硬件握手，单独做专家粒度Combine。

当某专家最后一个Wave slice处理完，每个AIC先FIX_S确认自己此前GMM2写完，再写该专家自己独占cache line的ready=1。**没有分到该专家tile的AIC也发布**：它已经按调度序越过该专家，消费者等的是全部P个AIC确认，不是“实际做过矩阵乘的几个核”。

上述有行任务的 Arch35 AIV1（末专家还包括 AIV0）/ Arch92 唯一 AIV，搬该专家的P个ready槽做向量ReduceSum，和达到P才继续；空行核不必等待。大专家跨Wave时，前几段不会发布完整专家ready，必须到最后slice才按行发送。

代码：[NotifyWaveGmm2Ready](../arch35_92/stage/mega_moe_gmm2_combine.h#L1096)、[WaitWaveGmm2Ready](../arch35_92/stage/mega_moe_gmm2_combine.h#L1109)、[Arch92专家结束发布](mega_moe_wave.h#L377)、[Arch35发布位置](../arch35/mega_moe_wave_a8w8.h#L353)。

之后按逻辑Combine核均分专家行，先分批装载metadata，再通过行环：搬一整行BF16到UB；若量化则V产生FP8数据和MX scale；MTE3按metadata写回源卡；槽复用前等旧MTE3。这组UB槽计数必须在排空后归零，避免下一Wave把已消费事件当成未回收事件再等。

代码：[RunWaveCombineStage](../arch35_92/stage/mega_moe_gmm2_combine.h#L1143)、[SendWaveCombineToken](../arch35_92/stage/mega_moe_gmm2_combine.h#L420)、[DrainCombineRowBuffers](../arch35_92/stage/mega_moe_gmm2_combine.h#L388)。

**稳态行为一致：** 完整专家 ready、P 个 AIC 各写独占槽、按行等待/发送的协议一致；固定 P 时稳态 Combine 都只有 P 个通信角色。不能把 Arch92 全唯一 AIV 做 Combine 误解为相对 Arch35 稳态少一半消费者。专家最后 slice 才可发送造成的等待两边都有；真正不同的是末专家扩工和唯一 AIV 的前序负载，分别见 11.4、8.2。

</details>

<details>
<summary><strong>11.4 Arch35末专家全AIV参与；Arch92只扩大buffer</strong></summary>

Arch35稳态独立Combine仍由AIV1做，AIV0还要做下一轮Activation或W4权重展开。最后一个非空专家完成后没有后续MoE计算，AIV0可以加入末专家Combine，两类AIV按合并的2P工人编号重新分行。

因此不是“每个专家每次都动用全部AIV”。普通逐tile路径始终配对AIV1；量化/去重专家粒度路径通常只有AIV1，特定末专家才全部AIV。W4 AIV0加入前需要等prologue MTE3读完UB，再建立Combine视图。

Arch92没有额外AIV0可加入：量化末专家只是先按旧槽数排空，再扩大行环buffer；分核数不乘2。去重使用其专用固定布局，不能套用普通量化扩容动作。

代码：[GetWaveCombineOwnedRange](../arch35_92/stage/mega_moe_gmm2_combine.h#L366)、[Arch35末专家分支](../arch35/mega_moe_wave_a8w8.h#L422)、[W4末专家UB复用](../arch35_92/mega_moe.h#L858)、[Arch92末专家扩容](mega_moe_wave.h#L452)。

**末专家的数据行为一致，阶段切换有区别：** 末专家两边都是全物理 AIV 按行处理同一种 Combine，不能把 2P/P 本身称为新增职责或合并算法变化。需要单独记录的是进入该阶段的方法：Arch35 从稳态 AIV1 子集切到全 AIV，原 AIV0 要结束 Activation/展开并交接 UB；Arch92 的通信角色本来已覆盖全部 AIV，没有第二角色接管动作，普通量化仅排空并扩本核行环。前者的跨角色 UB 等待与后者的 buffer 管理是实际调度区别。

**潜在性能影响：** 若固定 P，末专家 Arch35 的物理工人数是 2P、Arch92 是 P，可能影响大尾专家的行搬运/量化/合并时间，但这属于全核阶段的资源并行度，不是 Arch92 每行多做了任务，不能推导尾部必然翻倍。Arch92不必等待另一子角色的 prologue UB交接；扩 buffer 可增加单核在途能力，仍不等价于增加工人。去重布局不走普通量化扩容，不能套用该补偿。最终需同时看实际工人数、链路是否饱和和阶段切换是否位于关键路径。

</details>

</details>

<details>
<summary><strong>12. 去重分支：建表、少拉输入、合并回传</strong></summary>

<details>
<summary><strong>12.1 哪些去重开关及rowDesc到底是什么</strong></summary>

**执行核：** 建 rowDesc 的是全部 Arch35 AIV1 / Arch92 唯一 AIV；Arch35 去重开且 `N1=tiling.hiddenDim<=2048` 时全部 AIV0 也作为工人参加，各工人只扫描分给自己的工作项。AIC 不扫描 TopK、不建立 desc。Dispatch 消费 desc 的是 AIV1 / 唯一 AIV；Combine 消费者为稳态 AIV1 / 唯一 AIV，Arch35 末专家可加 AIV0。

`dedupMode` 的有效值：0都关，1仅Combine去重，2仅Dispatch去重，3都开。两种去重独立，但任一种开启都需要prepass建立rowDesc，不能因为“只开Combine”就省掉建表。

prepass按 `(源rank,源token分段)` 分工作项，扫描本卡各专家的route indices，把同一源token在本卡多个专家的出现聚成组。每行desc包括GROUP/FIRST/LAST/成员数、该token对应本卡的最小TopK槽，以及成员三元组：紧凑GM行、flattened index、Dispatch ready槽偏移。

组仍保留每个专家自己的输入/输出行，并不是把多个专家合成一个矩阵乘；减少的是跨卡输入拉取与输出回传次数。plain表示只命中本卡一个专家；group的FIRST用于输入拉取，LAST用于输出合并。

代码：[IsDispatchDedupOn/IsCombineDedupOn](../arch35_92/common/mega_moe_peermem.h#L126)、[DedupPrepassScanSlot](../arch35_92/stage/mega_moe_token_dedup.h#L184)、[DedupStoreGroupRowDesc](../arch35_92/stage/mega_moe_token_dedup.h#L347)。

maxOutputSize裁掉部分行时，建表同时保留全量命中数/最小TopK槽和实际幸存成员数。即使只剩一个可计算成员，如果原本同卡多命中，仍按GROUP写desc，确保返回槽与源卡canonical槽一致。

代码：[DedupStoreRowDescChunk](../arch35_92/stage/mega_moe_token_dedup.h#L382)。

**描述符算法一致：** 两边使用相同 group/FIRST/LAST/canonical 规则，去重并没有把多个专家的矩阵乘合成一次。真正的核分工差异是下一节的 AIV0 协作启用/取消；Dispatch 扇出及 Combine 合并的数据语义本身一致。

</details>

<details>
<summary><strong>12.2 Arch35可让AIV0帮建表；为什么先阻止GMM1写UB</strong></summary>

**执行核：** Arch35 所有 AIV1 都是 prepass 工人；`N1=tiling.hiddenDim<=2048` 且去重开时，所有 AIV0 也加入，按工作编号均分 item，空 item 核仍报告 worker-done。Arch92 只有 P 个唯一 AIV 建表，不存在第二类协作工人。这里的阈值是 **GMM1 完整 Gate+Up 宽度 N1，不是原 token 宽度 H**。

Arch35 协作时，工作编号把AIV1放低P、AIV0放高P。AIV0建立自己的count/cumsum UB视图，必须像AIV1一样按count epoch确认本轮数据，不能拿可能残留的workspace非零ready当成新轮到达。

每个工作项desc写出排空后，写该项ready=1；本核做完全部工作项后，done低16位由AIV1每核加1，高16位由AIV0每核加1。函数尾不互等，允许其他AIV1建表与先完成AIV1的Dispatch交叠。

代码：[GetDedupPrepassWorkerSlot](../arch35_92/stage/mega_moe_token_dedup.h#L404)、[PrepareDedupPrepassAiv0Local](../arch35_92/stage/mega_moe_token_dispatch_dedup.h#L550)、[工作项ready和worker done发布](../arch35_92/stage/mega_moe_token_dedup.h#L434)。

但是在第一次Dispatch ready发布前，所有AIV1必须等高16位AIV0 done到P。原因是GMM1 ready一旦出现，AIC就可能往AIV0 UB写结果，而AIV0还可能用同一UB建表。这里等的是“所有会被GMM1覆盖的AIV0 UB已经空出来”，不是等所有Dispatch核互相完成，也不是跨卡屏障。

Arch35 的 N1 超过2048时只让AIV1建表；Arch92只有唯一AIV建表，没有高半word完成计数、没有保护另一个AIV0 UB的gate。自己的建表完成后才能在同核继续后续工作。

代码：[WaitDedupPrepassPublishGate](../arch35_92/stage/mega_moe_token_dispatch_dedup.h#L444)。

**全核建表算法一致，协作依赖有差异：** `N1<=2048` 时，双方都是全物理 AIV 分担同一 item 扫描/desc 写出；Arch35 的 2P / Arch92 的 P 是资源数量，不应把它称为 Arch92 新增一次建表。实际差别是 Arch35 还让原计算 AIV0 独立读 count/cumsum，并在首次 Dispatch ready 前通过高半 word gate 确认所有 AIV0 建表 UB 已释放；Arch92 同核建表后才继续 Dispatch/Activation，没有另一个 AIV0 UB 被 AIC 提前覆盖的风险，因此没有这条额外协作链。

**潜在性能影响：** Arch92可减少协作 AIV0 的重复读表和 gate 等待；实际全核工人数也可能影响大量 item 的完成时间，但须作为资源因素另计，不能凭两者都全核工作的阶段认定新流水退化。`N1>2048` 时双方都只用 P 个 Dispatch角色建表，既无 AIV0 协作也无其 gate，不能仍说“建表工人减半”。少量/零 item 时更不能用启动核数估耗时，应看真正有任务的工人及数据就绪关键路径。

</details>

<details>
<summary><strong>12.3 Dispatch去重：拉一次，扇出到多个本卡专家行</strong></summary>

**执行核/同步双方：** Arch35 AIV1 / Arch92 唯一 AIV 执行下面的去重 Dispatch；不由协作建表的 Arch35 AIV0 做搬运。它们等建表工人发布的 item-ready（Arch35 工人可能含 AIV0），扇出写 GM 后向对应 AIC 发布 Dispatch ready；AIC 等真实输入行数。

这些 Dispatch AIV 按自己的行段加载route+desc，先等待这些条目对应prepass工作项ready，而不是整表所有worker全部完成。只保留plain和GROUP FIRST为实际fetch任务；同组其他行跳过跨卡拉取。

FIRST条目从源卡拉一次token/scales，向本卡各成员专家的紧凑GM行扇出，同时给各成员写自己的TopK槽metadata。扇出写完后按每个实际写入行/组贡献Dispatch ready，不能只为FIRST行加1，否则其他专家的AIC等不齐。数据相同，metadata中的专家路由槽并不相同。

代码：[WaitDedupPrepassForBatch](../arch35_92/stage/mega_moe_token_dispatch_dedup.h#L187)、[CompactDedupActiveEntries](../arch35_92/stage/mega_moe_token_dispatch_dedup.h#L243)、[FanoutDedupDataAndMeta](../arch35_92/stage/mega_moe_token_dispatch_dedup.h#L67)、[PublishDedupTileReadyRuns](../arch35_92/stage/mega_moe_token_dispatch_dedup.h#L305)。

只开Combine去重时仍走普通Dispatch fetch/store，但保留Combine所需desc和权重记录；Dispatch和Combine开关不应混用。

**数据行为一致：** 两边都是 FIRST 拉一次并扇出各成员行，ready 按真实成员贡献；同一组的远端数据读次数没有因为 1C1V 增加。性能差异可能来自前置 item-ready 到达、唯一 AIV 的任务排队及 12.2 的建表 gate，而非改了扇出算法。

</details>

<details>
<summary><strong>12.4 Combine去重：LAST行合并，源卡只读canonical槽</strong></summary>

**执行核/同步双方：** 专家卡稳态由 Arch35 AIV1 / Arch92 唯一 AIV 做成员读回、FP32 加权合并和发送；Arch35 最后一个非空专家可加入 AIV0。它们等待全部 AIC 的专家 ready 及全部建表工人的 done。源卡最终只读 canonical 槽的是全部物理 AIV 按本地 token 分工的 Unpermute，Arch35 此时包括 AIV0/AIV1；AIC 不合并、不做 Unpermute。

该路径不用GMM2逐tile硬件credit，等完整专家ready，并额外等prepass全部worker done。然后读取当前行的desc：plain正常发送；group非LAST不发送；LAST驱动读本卡所有成员专家的GMM2输出，做FP32乘权/累加，转BF16并只写源卡该token最小TopK槽。

专家按顺序推进，LAST所在的最后成员专家已经ready时，同组前面专家也已经由这些AIC处理完成，所以不另外为每个成员创建跨专家ready协议。不能在专家处理次序改乱后仍无条件沿用这个推论。

代码：[RunWaveCombineStageDedup](../arch35_92/stage/mega_moe_gmm2_combine_dedup.h#L332)、[PLAIN/SKIP/LAST分流](../arch35_92/stage/mega_moe_gmm2_combine_dedup.h#L296)、[SendWaveCombineTokenDedupMerge](../arch35_92/stage/mega_moe_gmm2_combine_dedup.h#L213)。

TopK预取关时，合并端必须按成员权重乘后再加；预取开时GMM2结果已含权重，只累加。权重通过Dispatch在组LAST行保存整token权重记录供合并读取。FP32累加分4096元素段，成员输入双buffer和合并发送buffer都有本核事件保护。

源卡Unpermute根据原TopK专家所在rank，只读每个rank第一次出现的槽；同rank后续槽根本不会到达，不能再读取它们。未预取时，同rank多命中group已经乘权，不能重复乘；单命中plain仍需要源卡乘权。

代码：[StoreDedupGroupWeights](../arch35_92/stage/mega_moe_token_dispatch_dedup.h#L106)、[合并成员权重](../arch35_92/stage/mega_moe_gmm2_combine_dedup.h#L141)、[JudgeDedupCanonicalSlot](../arch35_92/stage/mega_moe_unpermute.h#L109)。

去重使用的额外事件首轮先预挂一个可复用Set，使用时Wait、结束后Set，阶段尾Drain全部消费。这样首轮不空等、下轮不继承悬挂事件。这些不是另加跨卡同步。

代码：[ArmWaveCombineDedupEvents](../arch35_92/stage/mega_moe_gmm2_combine_dedup.h#L64)、[DrainCombineRowBuffersDedup](../arch35_92/stage/mega_moe_gmm2_combine_dedup.h#L257)。

**合并语义一致，工作机会有差异：** LAST 驱动成员加权/累加、canonical 回源及源卡跳过重复槽的算法一致，固定 P 时稳态都用 P 个通信角色。Arch35 末专家可加入原 AIV0，Arch92不能，相关尾部风险见 11.4；Arch35 协作 prepass 的 done 包含高半 word，而 Arch92只等唯一角色 done，前置建表风险见 12.2。不能把这些 worker/ready 差异说成 Arch92 多读/多乘了一遍成员结果。

</details>

</details>

<details>
<summary><strong>13. MoE 之后：清 count、共享 GMM2、跨卡握手和最终输出</strong></summary>

<details>
<summary><strong>13.1 导出专家 token 数，再清通信 count 接收区</strong></summary>

**执行核：** Arch35 AIV1 的逻辑 block0 导出、全部 AIV0 均分清零；Arch92 唯一 AIV 的 block0 导出、全部唯一 AIV 均分清零。下面第 2、4 步由全部物理 AIV 执行，包括没分到清零元素的核；AIC 不进入本函数。

MoE Wave 结束后，所有 AIV 进入 `ExportAndResetExpertCounts`，AIC 不进入。这一步并不是清 GMM2 输出，也不是清下一阶段将读的共享结果：

1. Dispatch 角色的逻辑 block0 导出本卡各专家的接收 token 数。
2. 所有 AIV 在 MTE3 消费 `COUNT_TABLE_READ_DONE_FLAG`。清零不能抢在仍使用原 count 表的核前面。
3. 计算向量角色均分固定 8KB 的通信 count 接收区，等旧 MTE3 读完复用 UB，V 填零，V_MTE3 后写回 GM。
4. 全部 AIV 排空自己的流水，然后 `SyncAll<true>()`，确保本卡 MoE 回传和 count 清理阶段都结束。

Arch35 由 AIV1 block0 导出，AIV0 清零；Arch92 唯一 AIV 同时具有两个角色，block0 导出，全部唯一 AIV 按逻辑编号清零。没有分到清零元素的核也必须参加事件消费和尾部 AIV 屏障。

代码：[ExportAndResetExpertCounts](../arch35_92/mega_moe.h#L915)。

count-read-done 的生产者分支也要配齐：每个正常读表的 Arch35 AIV1 / Arch92 唯一 AIV 在读完后发；不读表的 Arch35 AIV0 也按协议发；去重开且 `N1=tiling.hiddenDim<=2048` 时，AIV0 真正读取 count，由它的读表分支发，不能重复发两次。该通知表明“本轮原 count 表已用完”，与入口的 workspace reset-done 不同。

代码：[正常读表发布完成](../arch35_92/stage/mega_moe_token_dispatch.h#L446)、[不读表角色的对应通知](../arch35_92/stage/mega_moe_token_dispatch.h#L477)、[去重 AIV0 的独立读表](../arch35_92/stage/mega_moe_token_dispatch_dedup.h#L550)。

这次 `SyncAll<true>()` 在有无共享专家时都执行，只等本卡 AIV；不能把它说成共享 GMM2 完成的全 C/V 屏障。跨卡握手在后面，AIC 的共享计算由另一套数据 ready 约束。

**职责差异与潜在性能影响：** 固定 P 时，Arch35 清零的是 P 个 AIV0，Arch92 清零的是 P 个唯一 AIV，固定 8KB 清理量及同步协议不变；不同点是 Arch35 AIV1 block0 导出 count，可与 AIV0 的清零在不同核上推进，Arch92 block0 同时负责导出和自己的清零，按本核顺序完成。这个小的尾部职责合用可能影响最后到达 AIV 屏障的核，但没有多清一遍 count 或新增屏障。是否显著取决于导出/清零是否位于关键路径，不能只根据核身份更换判断性能。

</details>

<details>
<summary><strong>13.2 共享 GMM2：结果写本卡 GM，不走 MoE Combine</strong></summary>

**执行核：** GMM2 矩阵乘/共享完成计数由 AIC 做；Arch35 共享 A8W4/A4W4 另由 AIV0 展开 W2，共享 A8W8 没有这项 AIV0 工作。Arch35 AIV1、Generic 共享 GMM2 下的 AIV0、Arch92 唯一 AIV 不做这里的矩阵乘或回传。最终读共享结果的不是早期 Activation 专用角色，而是全部物理 AIV 中分到本地 token 的 Unpermute 核。

各核遍历共享专家，输入是此前共享 Activation GM，M 是本卡原 token 数，K=F，N=H。AIC 按滚动 scheduler 做 W2 矩阵乘；第一次读一个 M-group 前，等它的共享 Activation flag 到齐。

Arch35 共享 A8W8 用 Generic；共享 A8W4，以及当前 Activation 输出 FP8 的共享 A4W4，第二段用自定义 A8W4，由 AIV0 展开 W2。Arch92 当前共享 A8W8/A4W4 两段都用 Generic，不需要专职 prologue AIV。

这里不需要向源卡回传：共享专家处理本卡全部输入，输出放本卡 `[共享专家][本地token][H]` GM。AIV 不执行 MoE 的逐 tile Combine，相关角色返回或只推进分核状态，然后可以去做出口握手/Unpermute。

代码：[RunSharedExpertGmm2](../arch35_92/mega_moe.h#L743)、[ProcessSharedExpertGmm2Loop](../arch35_92/mega_moe.h#L764)、[共享 GMM2 地址与 flag 绑定](../arch35_92/stage/mega_moe_gmm2_combine.h#L1038)。

每完成一个共享 GMM2 N256 tile，AIC 先 FIX_S 确认 GM 写完，再给该共享专家、该 M256-group 的 counter 原子加1。对 H=4096，一个 M-group 需要16个 N tile，因此 counter 目标是16；不是256个 token 各加一次，也不是等16个特定 AIC。

最终处理 token t 的 Arch35 AIV0/AIV1 或 Arch92 唯一 AIV，用 `group=t/256` 找 counter，等 `ceil(H/256)` 后，才读该共享专家的完整 H 行。多个核对同一组取不同 token，可以读同一完成计数。计数不按读者消费减回去，下一 launch 入口清零。

代码：[NotifySharedExpertTileCompletion](../arch35_92/stage/mega_moe_gmm2_combine.h#L538)、[AccumulateSharedExpertForToken](../arch35_92/stage/mega_moe_unpermute.h#L199)。

所以当前 MTE Arch35/Arch92 **都不在共享 GMM2 后加 `SyncAll<false>()`**。AIV 到得早，就在真正读取某组共享结果时等那组 counter；无关组和 MoE 回传无需等整个共享矩阵都完成。后文 Layered 是另一套实现，那里确实有全 C/V 屏障，不能混用结论。

**行为一致与模式差异：** 对照共享 A8W8，两边都是 AIC 写本卡 GM、逐 N tile 发组 counter，全部 AIV 后续按自己的本地 token 消费，共享 GMM2 不需要对端 Combine，也没有 Arch92 专属全核等待。它们仍允许 AIC 共享 GMM2 与 AIV 出口握手重叠。共享 A4W4 的第二段则有 FP8×FP4+展开与 FP4×FP4 的实现差异，潜在影响见 10.3；这不是改变共享结果的读者或新增一次回传。

</details>

<details>
<summary><strong>13.3 出口跨卡握手：保证源卡能读所有 MoE 回传</strong></summary>

Unpermute 读取的是其他卡 Combine 写回的结果，本卡不能仅凭“自己已经发完”就开始读。所有物理 AIV 在 `ProcessTokenUnpermuteStage` 中先调用出口跨卡同步，Arch35 是2P个参与者，Arch92 是P个；AIC 不参与。

每个 AIV 取本物理核的 sequence，推进到下一值，按 `rank=jobIndex; rank<worldSize; rank+=totalJobs` 分担 peer：向该 peer 的“本rank槽”写 sequence，再等本卡对应 peer 槽等于本轮值。最后更新自己的 sequence；没分到 peer 的 AIV 也要推进，不能停留旧轮次。

各 rank 的 peer 任务已被均分，所以一个 AIV 不需要遍历全部 peer。随后本卡 `PIPE_ALL` 和 `SyncAll<true>()` 汇合各 AIV 的握手结果。某个 Unpermute 核负责的 token 可能依赖另一握手核负责的 rank，这就是尾部 AIV 屏障的目的。

代码：[ProcessTokenUnpermuteStage](../arch35_92/mega_moe.h#L1063)、[CrossRankHandshakeInWorldSize](../arch35_92/common/mega_moe_utils.h#L432)、[CrossRankSyncInWorldSize](../arch35_92/common/mega_moe_utils.h#L447)。

当前函数在跨卡握手**前没有另加 `SyncAll<true>()`**；不要把旧实现的调用顺序套过来。前面的 count 清理阶段已经在 MoE 回传排空后做了全 AIV 汇合；共享 GMM2 的结果只在本卡使用，另由共享 counter 约束，不必为了它延后所有跨卡握手。

入口每 AIV sequence 已在输入量化/打包后推进一次，出口再推进一次。count 的 epoch 和这里的完整 sequence 都利用同一物理核轮次来源，但二者存储格式、消费位置、用途不同；不能把 count 的8位 tag 当作共享 GMM2 或 GMM1 完成数。

**行为一致：** 双方都是全物理 AIV 分 peer 做同一握手，尾部全 AIV 汇合；Arch92 没把 AIC 的工作搬到握手核，也没增加一轮握手。2P/P 可影响每核负责多少 peer、参与本卡汇合的资源数量，但不构成新的同步协议或职责串行化。握手测得更久还可能是之前的 Combine 更晚、对端更晚，不能自动归因于本函数变慢。

</details>

<details>
<summary><strong>13.4 Unpermute：按原 token 聚合 TopK 和共享专家</strong></summary>

**执行核：** Arch35 全部 2P 个 AIV0/AIV1、Arch92 全部 P 个唯一 AIV，均分本地原 token。下面读回传、反量化、乘权累加、等共享 counter、Cast 和写 Y，全由拿到这些 token 的物理 AIV 完成；AIC 不做最终聚合。这里不沿用 Wave 中只让 AIV1 做稳态 Combine 的分工。

跨卡握手后，每个物理 AIV 均分本卡原始 token，不沿用专家 GMM 的 tile 分工。某核拿到 `[start,count)`，再按自己的 UB 容量分 batch：

1. 预取关时，装本 batch 的 TopK 权重；FP32直接读，BF16先转FP32。
2. 对一个原 token，按 TopK 槽从 `[token,slot,H]` 通信区读 MoE 结果。普通 Combine 的 BF16 转FP32；量化 Combine 的 FP8 data/scales 先反量化。
3. 未预取时每个专家先乘自己的 TopK 权重再加；预取开时权重已进入 Activation，不重复乘。
4. 有 Combine 去重时，只读每个目标rank的 canonical槽；同rank多成员的合并值已在发送端乘权，只累加。单成员 plain 的非预取值仍在此乘权。
5. 对每个共享专家，先等该 token 所属共享 M-group counter，再搬共享 BF16 行、转FP32、直接相加，不乘 TopK 权重。
6. FP32累加结果用 `CAST_RINT` 转BF16，V_MTE3 后写 Y 的原 token 行。

代码：[UnpermuteTokens](../arch35_92/stage/mega_moe_unpermute.h#L335)、[ProcessTokenUnpermuteBatch](../arch35_92/stage/mega_moe_unpermute.h#L240)、[AccumulateMoeExpertsForToken](../arch35_92/stage/mega_moe_unpermute.h#L158)、[加载与反量化](../arch35_92/stage/mega_moe_unpermute.h#L71)。

输入环的 V_MTE2 防止覆盖 V 还在用的槽，MTE2_V 防止读取未到达输入；输出的 MTE3_V 防止下一个 token 的Cast覆盖还在发送的结果槽。末 batch 也排空输入事件和最后一次输出 MTE3。

Arch35 两类 AIV 此时不再分“计算/通信”，都做上述同一件事；Arch92 是相同数学流程，少一半物理 AIV 工人。空 token 核在完成跨卡同步后可以直接退出 Unpermute，不需要再做 token 数据操作。

代码：[输出槽保护与写回](../arch35_92/stage/mega_moe_unpermute.h#L261)、[Unpermute末尾排空](../arch35_92/stage/mega_moe_unpermute.h#L362)。

**行为一致：** Arch35 此处已经让 AIV0/AIV1 做相同的 token 聚合，Arch92全部唯一AIV也做同一算法、同一乘权位置和共享组等待。没有发生“把 AIV0 的另一项工作串进 AIV1”的任务合并；只存在物理工人数及芯片能力差异。按固定 P，Arch92每核可能分到更多本地 token，但总读取/累加次数不因此增加，不能将核数比等同于耗时比。

</details>

<details>
<summary><strong>13.5 EndSync 与返回：收回的是事件，不是再算一遍结果</strong></summary>

**执行核：** AIC、Arch35 AIV0/AIV1、Arch92 唯一 AIV 都走公共尾部调用，但 `EndSync` 内实际 Wait ACK 仅在 AIC 执行；AIV 的 ACK 已在消费 tile 时发布，不在这里替 AIC 回收。控制位恢复和完成阶段记录则是各核自己的收尾工作。

公共 `ProcessWave` 末尾依次 `sharedGmm1ActivationSync.EndSync()`、`gmm1ActivationSync.EndSync()`、`gmm2CombineSync.EndSync()`，AIC 消费本事件链剩余 ACK。它们使用独立事件对，故共享 GMM1 的收尾可以留到 MoE 之后，不会占住 MoE GMM1 的ready/ACK。

没有实际提交这组硬件 tile 的场景，pending数为0，EndSync不凭空等待；Arch92 非预取 MoE GMM1 使用 GM序号，也不产生这条硬件 ACK。UB-handoff 的 ping/pong ACK 在它自己的调用尾部排空，不是都留到这三个EndSync。

EndSync保护事件生命周期以及生产/消费收尾，不代替 Unpermute 的共享 counter 或出口跨卡同步。最后恢复入口 overflow 控制位，更新 exception stage 为完成，kernel返回。

代码：[主流程尾部三个EndSync](../arch35_92/mega_moe.h#L1122)、[GMM1 EndSync](../arch35_92/common/mega_moe_gmm_epilogue_sync.h#L175)、[GMM2事件链](../arch35_92/common/mega_moe_gmm_epilogue_sync.h#L292)。

**收尾规则一致，事件来源不同：** GM 硬件链两边都由 AIC 收剩余 ACK，各核恢复本核控制状态；Arch92非预取MoE GMM1不产生这条 ACK，Arch35 UB ACK则在自己的调用尾排空。这是 9.2/9.3 的交接差异延续，不是 Arch92 新增或删除全核完成屏障。收尾 Wait 耗时取决于对应 AIV 最后消费是否完成，不能只按源码 Wait 数量累计性能成本。

</details>

</details>

<details>
<summary><strong>14. 同步速查：生产者、消费者与计数对象不要混淆</strong></summary>

<details>
<summary><strong>14.1 MTE 主线的数据依赖表</strong></summary>

| 同步 | 生产者 -> 消费者 | 何时发布/等待 | 防止什么 |
|---|---|---|---|
| INPUT_RESET_DONE，mode0 | 全物理AIV/MTE3 -> 全物理AIV/MTE3 | workspace清零后发，count发布前等 | reset还在执行时下游开始生产/消费状态 |
| TOKEN_DATA_READY，mode0 | 全物理AIV/MTE3 -> 下标/count发送AIV，以及Arch35有共享时的AIV0/MTE3 | 发送者：35无共享为AIV0/AIV1、有共享为AIV1，92为唯一AIV；35共享直接输入的AIV0推迟到MoE前等 | 接收端见count却读到未准备输入 |
| INPUT_PREPARE_DONE，mode4 | Arch35 AIV0或Arch92唯一AIV/MTE3 -> 配对AIC/MTE2或Scalar | 输入条件完成后发，AIC开始相应输入读取前等 | AIC过早读量化输入/状态 |
| INPUT_UB_FREE，mode4 | Arch35 AIV0/MTE3 -> 配对AIC/FIX | 输入UB释放后发 | FIX覆盖输入准备仍使用的UB |
| count+epoch，通信GM | 源卡35无共享AIV0/AIV1、有共享AIV1，或92唯一AIV -> 专家卡35 AIV1/92唯一AIV；35 N1<=2048去重时AIV0也读 | token和route index写好后发布，整表epoch齐后消费 | 读上次launch count或未完整路由 |
| 整张专家行数表ready，workspaceGM | 35各AIV1/92各唯一AIV -> 配对AIC，以及需要该表的35 AIV0 | 每个发布AIV写自己的完整表，在第0专家平面的本逻辑block槽发布一次 | 根据旧行数构造problem |
| Dispatch ready，GM原子计数 | 35 AIV1/92唯一AIV -> 分到相应GMM1 tile的AIC | 实际紧凑行写完后累计，GMM1读M-group前等 | GMM1读未到达A/scales |
| GMM1 UB ready/ACK，mode4 | AIC/FIX <-> Arch35 AIV0/V | 每ping/pong tile交接 | 读未写完结果/覆盖未消费UB |
| GMM1 GM ready/ACK，mode4 | AIC/FIX <-> 35 Generic预取AIV0、35 A8W4 AIV1、92预取或共享唯一AIV；AIV在MTE2等ready、MTE3发ACK | 对应MoE/共享模式以9.3表为准；AIC Scalar收ACK，15-credit限流 | 过早搬结果/硬件事件积压 |
| GMM1 GM递增序号 | Arch92非预取MoE AIC -> 唯一AIV | FIX_S后递增，AIV等下一个序号 | 过早搬GMM1结果 |
| Activation ready，GM原子计数 | 35 A8W8/A4W4 AIV0、35 A8W4 AIV1、92唯一AIV -> GMM2 AIC | MoE按MoE模式、共享按共享模式；data/scales落GM后累计，GMM2读输入组前等 | 只完成部分N块就沿完整K读 |
| GMM2普通Combine ready/ACK，mode4 | AIC/FIX <-> 35 AIV1/92唯一AIV，AIV在MTE2等ready、MTE3发ACK | 普通非量化/非去重每tile交接，AIC Scalar收ACK | 过早取结果/事件超额 |
| GMM2专家ready，GM独占槽 | 全P个AIC -> 35稳态AIV1、末专家可加AIV0，或92唯一AIV | 专家最后slice结束后每AIC写1；有行任务的量化/去重AIV ReduceSum等P | 按整行读部分完成专家 |
| COUNT_TABLE_READ_DONE，mode0 | 全物理AIV按读/不读分支 -> 全AIV/MTE3 | count使用结束发，出口清表前等 | 清掉仍在读的原count |
| 出口SyncAll(true) | 本卡全AIV | MoE回传排空/清count后 | 本卡部分发送尚未结束即握手 |
| 共享GMM2 counter，GM原子计数 | 分到共享N tile的AIC -> 35 AIV0/AIV1或92唯一AIV中分到相应本地token的Unpermute核 | 每N tile写好加1，读共享整行前等ceil(H/256) | 共享矩阵还没写齐就聚合 |
| 出口rank sequence+SyncAll(true) | 35全部AIV0/AIV1或92全部唯一AIV，按物理编号分peer -> 本卡同一组全部AIV | peer互相确认后本卡汇合，AIC不参与 | 提前读其他卡Combine回传 |

上述对应源码已在各阶段紧随说明给出。内部的 MTE2_V、V_MTE3、MTE3_MTE2 等 HardEvent 主要保护本核 UB/搬运流水；它们不是所有核的屏障。mode0、mode4相同数字id也不是同一作用域的同一通知。

**同步差异的性能归因：** 输入/reset/count-read-done及出口全AIV协议本身一致；明确变更的是 GMM1 UB→GM 的 ready/ACK链、非预取MoE序号轮询、去重协作gate，以及职责合用后的本核UB依赖。配对事件换目标 AIV 不自动新增一次同步；一个 Wait 未阻塞与反复轮询很久也不是同一成本。阶段重叠变少可能让原有等待更久，并不要求源码新增 Wait；应以实际生产者何时完成、消费者何时到达来评估。

</details>

<details>
<summary><strong>14.2 什么时候一定要全部核参加，什么时候允许空任务返回</strong></summary>

全AIV的 `SyncAll<true>()`，以及输入/reset/count-read-done的全AIV事件协议，参与者不能按“本核有没有token/专家”删减。空核仍须发布/消费约定次数；否则有任务核可能永远等不齐。

配对GMM tile ready/ACK只在真正分到tile时产生。分到0个tile的核不应该为了“全核一致”凭空等一次ready；但是仍要推进scheduler游标，保证后续自己与配对核顺序一致。

GMM2专家ready是第三种：即使本AIC没分到tile，也要在该专家最后slice越过时写ready=1，因为消费者等全部P个AIC。不能拿配对tile的“空核不发”规则套在这里。

去重prepass的item-ready、worker-done又是两层：Arch35 AIV1 / Arch92 唯一 AIV 的 Dispatch 只等自己需要的item，Combine读成员表的AIV要等worker-done全齐；Arch35 去重开且 `N1=tiling.hiddenDim<=2048` 时，AIV1 额外等全部 AIV0 worker-done保护UB。这里判断的是 GMM1 Gate+Up 宽度 N1，不是原 token 宽度 H。没有理由把所有局部ready都替换成一个全核屏障。

代码：[被跳过problem仍推进](../arch35_92/common/mega_moe_utils.h#L165)、[专家ready的发布/等待](../arch35_92/stage/mega_moe_gmm2_combine.h#L1096)、[输入空核也通知](../arch35_92/mega_moe.h#L941)。

</details>

<details>
<summary><strong>14.3 workspace中的槽分别按什么维度分配</strong></summary>

**执行核：** 各 AIC/AIV 的 `Init` 绑定本核使用的 workspace 地址；实际清零由全部物理 AIV 均分。槽的发布/读取角色按下面索引和 14.1 的具体参与者对应，不能由“槽按 AIC 分配”推断它一定是 AIC 写入，例如整表 ready 实际由配对 Dispatch AIV 写。

本节只解读kernel使用的布局，不展开host tiling实现。`INT_CACHELINE`用于让标志槽隔开缓存行，不等于“一个slot里有这么多个AIV”。

| 状态区 | 逻辑索引/用途 |
|---|---|
| flagDispatchToGmm1 | `[专家][最多M-group]`，值为到达紧凑行数 |
| flagActivationToGmm2 | `[专家][最多M-group]`，值为完成Activation块数 |
| flagSendCntCalToUpdParams | 分配为`[专家][AIC]`；当前MTE整表一次发布，只用第0专家平面的本核槽。Layered另按专家发布 |
| flagGmmToEpilogue | `[逻辑block]`，非预取GM-handoff的递增序号 |
| gmm2Ready | `[专家][AIC]`，专家粒度Combine的每AIC完成标记 |
| sharedActivationToGmm2 | `[共享专家][本地token M-group]`，共享第二段输入ready |
| sharedExpertGmm2TileCounter | `[共享专家][本地token M-group]`，完成N tile计数 |
| dedupPrepassReady | done总槽 + `[源rank,tokenChunk]` item-ready槽 |
| cumsumInfo | 每Dispatch逻辑核的count前缀备份，不是同步完成计数 |

入口清零长度来自 `flagResetElementCount`，按实际分配的连续flag区算；预取tile-status等独立区按其专门逻辑清。不能按“比Arch35少一半AIV”就把所有flag数量统一除2：有的按专家行组分，有的按P个AIC分，有的才按物理AIV分。

代码：[InitializeMteSyncFlags](../arch35_92/common/mega_moe_workspace.h#L236)、[InitializeRoutingFlags](../arch35_92/common/mega_moe_workspace.h#L308)、[ResetSyncStatus](../arch35_92/stage/mega_moe_workspace_reset.h#L24)。

**当前 MTE 普通 Combine 不再用旧的“每M-group给若干consumer槽加N tile计数”主协议**：普通非量化用硬件配对事件；量化/去重用完整专家的 `[专家][AIC]` ready。共享输出用每M-group单counter。历史CV槽位数量错配不能直接照搬为当前三种协议的解释。

按 `max(M-group数,逻辑consumer数)` 展开GroupSyncSlotLayout、给一个M-group多个复制counter的逻辑，当前主要在Layered Combine使用，生产者和消费者都应使用相同layout。即便复制多个槽，它们计数的是同一组的N tile完成，绝不是每个AIV各贡献一个token。

代码：[CalcGroupSyncSlotLayout/GetGroupSyncSlotRange](../arch35_92/common/mega_moe_utils.h#L130)、[Layered消费者读组槽](../arch35/stage/mega_moe_layered_combine.h#L108)、[Layered生产者复制通知](../arch35_92/stage/mega_moe_gmm2_combine.h#L521)。

**布局差异与潜在性能影响：** Arch92 的 `GM_HANDOFF_EXPERT_WAVE` 对 MoE 强制预留 BF16 GMM1 GM 结果，并保存 cumsum；Arch35 非预取 A8W8 通常不需要这两个区域，Arch35 W4/预取则已有相应 GM 区域。Arch92 A4W4 的 Activation data 按 FP4缩小；部分 GMM1→Activation 标志区也按 GM-handoff 的需要分配。它们改变 workspace占用、实际搬运及少量reset工作，但**分配了区域不代表每轮清零整个结果矩阵**，入口清的是相应连续 flag 区。GMM2 BF16 GM、Dispatch数据、metadata等共同区域不能都算成Arch92新增流量。

代码：[MTE按策略绑定布局](../arch35_92/common/mega_moe_workspace.h#L124)、[结果与cumsum预留条件](../arch35_92/common/mega_moe_workspace.h#L322)、[Activation data打包比例](../arch35_92/common/mega_moe_workspace.h#L210)。

</details>

</details>

<details>
<summary><strong>15. Arch35 URMA/Layered 支线：不要套用 MTE Wave 的顺序</strong></summary>

<details>
<summary><strong>15.1 入口和完整顺序：这是另一套kernel类</strong></summary>

Arch35 的 URMA 入口在 Devkit/Hcomm 版本宏满足要求、X是BF16、Dispatch为MXFP时实例化 `MegaMoeLayered`。当前 Arch92入口只有MTE，没有对应Layered实例。Layered共享路径也不是MTE公共类里独立 `SharedQuantConfig` 的那套模式组合，不能直接套用混合共享量化模式的分工表。

**比较边界：** 本章没有已运行的 Arch92 Layered 对应物；这里的两级 relay、Hcomm通道归属和全C/V屏障是 Arch35另一通信实现的行为，不能直接列成“1C1V缺失某项任务”或据此估Arch92迁移退化。评估CV迁移应先对齐双方MTE拓扑；跨拓扑的通信差异需要另做实验。

代码：[Arch35版本宏](../arch35/mega_moe_apt.cpp#L25)、[URMA入口](../arch35/mega_moe_apt.cpp#L125)。

该类 `Process` 的实际顺序为：

1. 全部物理 AIV0/AIV1 按目标 rank 分路由任务，生成并发送该 rank 各专家的 bitmask/count；它们还均分清 workspace/relay 状态，有共享时均分本地 token 准备共享输入。AIC 不执行这些数据准备。
2. AIV排空，**全C+V `SyncAll<false>()`**，AIC等状态清零和输入准备完成。
3. 可选共享 GMM1：AIC 矩阵乘，A8W8/A4W4 的 AIV0 做 Activation，A8W4 的 AIV0 展开权重、AIV1 做 Activation；随后全部 AIC/AIV0/AIV1 再次 `SyncAll<false>()`，共享使用的 UB 和结果交接已结束。
4. 全物理AIV均分原token，量化到本Server relay区域，批量写本地token-ready，构造跨Server一级发送队列。
5. 全部物理 AIV0/AIV1 分 peer 做第一次跨rank握手，沿用发送mask已经取得的Hcomm通道锁，完成后Drain/Unlock，并全AIV汇合。AIC 不做握手。
6. 固定通道 owner 的 AIV1 提交一级 relay 队列、预取首个宏 Wave，并做后续二级 Dispatch/Combine；AIC 做两段 GMM，AIV0/AIV1 按模式做 Activation/权重展开。
7. Wave末由 AIV1 导出专家 token 数、Drain/Unlock 固定归属通道，全部 AIV0/AIV1 汇合；AIC 收共享GMM1剩余ACK。
8. 可选共享 GMM2：AIC 矩阵乘，A8W4/A4W4 的 AIV0 展开 W2，其他向量角色不做共享回传；结束时全部 AIC/AIV0/AIV1 确实参加 `SyncAll<false>()`。
9. 全部 AIV0/AIV1 建立自己的 Unpermute UB，按物理编号分 peer 做第二次跨rank握手，重新获取通道锁、确认Combine回传；再均分本地 token 做 Unpermute 写 Y，AIC 不做聚合。

代码：[Layered Process](../arch35/mega_moe_layered.h#L1198)。

这里出现的 `SyncAll<false>()` 要由同一分支中的全部C/V参加；不是“只有做共享计算的几个核随意调用”。例如共享GMM2后的全C/V屏障，AIV1虽然不做共享矩阵乘仍调用，才能让后续直接读取共享结果安全。MTE没有该屏障是因为用了共享M-group计数，不是可以把Layered的屏障删掉。

</details>

<details>
<summary><strong>15.2 路由和两级Dispatch：先送Server中继，再由专家卡拉取</strong></summary>

**执行核：** 入口路由、relay 量化及一级队列建表由全部物理 AIV0/AIV1 分担；进入 Wave 后一级队列提交、二级 Dispatch 则只由固定通道 owner 的 AIV1 执行。AIC 不生成 mask、不建队列、不提交 URMA 数据搬运；它在后面等二级 Dispatch 的行 ready。

路由发送按 `curRankId=aivCoreIdx_; curRankId<worldSize; curRankId+=blockAivNum_` 分目标 rank。拿到目标 rank 的 AIV0 或 AIV1，扫描该 rank 所有本地专家的 flattened TopK位置，生成各专家压缩bitmask和count；没有 rank 任务的 AIV 不做发送，但仍参加后续屏障。一个命中位置占1bit，故对应字节数有 `/8`；它与MTE发送紧凑int16/int32索引列表不同。读取mask后才能确定各源卡哪些token/槽命中本专家。

代码：[BuildExpertMask](../arch35/stage/mega_moe_layered_dispatch.h#L382)、[SendMaskCal](../arch35/stage/mega_moe_layered_dispatch.h#L442)。

全部物理 AIV0/AIV1 先量化自己均分的原token，写本卡relay记录，再给这些token批量写ready。同一个 AIV 扫描自己负责 token 的TopK目标Server，将同一个目标Server去重后加入一级队列：同一token在远端同Server多个专家需要时，不重复跨Server发完整输入。这里只建队列，还不是固定 owner AIV1 的 URMA 提交。

代码：[QuantizeLocalTokensToRelay](../arch35/stage/mega_moe_layered_dispatch.h#L617)、[SetLocalRelayReadyFlags](../arch35/stage/mega_moe_layered_dispatch.h#L662)、[AppendTokenToDispatchRelayQueues](../arch35/stage/mega_moe_layered_dispatch.h#L688)、[BuildDispatchRelayQueues](../arch35/stage/mega_moe_layered_dispatch.h#L730)。

进入Wave后，AIV1先发布首宏Wave专家count，然后 `SendDispatchRelayQueues` 一次提交跨全部专家的一级队列，再发布剩余专家count。**一级队列不按计算Wave截断**。一级输入到目标Server中继后，专家所在rank的AIV1按自己的专家范围进行二级拉取，生成紧凑GM和metadata。

二级远端读取先分窗口读relay token-ready，确认当前batch需要的记录到了，才批量提交数据读取；本地relay读取也等对应token-ready。完成写入紧凑GM后，按实际行数给专家M256-group的Dispatch flag原子累计，AIC仍按行到齐启动GMM1。

代码：[PrepareDispatch](../arch35/stage/mega_moe_layered_dispatch.h#L1432)、[SendDispatchRelayQueues](../arch35/stage/mega_moe_layered_dispatch.h#L799)、[ReceiveRemoteDispatchBatch](../arch35/stage/mega_moe_layered_dispatch.h#L1039)、[PublishDispatchRows](../arch35/stage/mega_moe_layered_dispatch.h#L994)。

一级ready保证“中继中的原token已到”，Dispatch ready保证“本专家紧凑输入已到”。两者分别保护不同地址、不同阶段，不能只看到已有relay ready就省掉GMM1的Dispatch等待。

</details>

<details>
<summary><strong>15.3 宏Wave：按专家区间分组，而不是M-group截断游标</strong></summary>

**执行核：** 全部 AIC/AIV0/AIV1 在各自调度对象中计算宏 Wave 专家边界；AIC 算 GMM，AIV0 按模式做 Activation 或 W1/W2 展开，AIV1 按 owner rank 做二级 Dispatch/Combine，A8W4 时还做 Activation。计算宏 Wave 边界不是一个通信核替全部核集中调度。

Layered根据本地token×TopK估计行量、专家数量和常量阈值，选择目标宏Wave数，计算首Wave专家数和稳态专家数。这是组织专家区间的启发式，不是实际路由count，也不是MTE的 `mGroupsPerWave_`。

例如当前宏Wave专家区间是 `[begin,end)`：AIV1先二级接收下一宏Wave专家区间；AIC/AIV0同时处理当前区间。随后各角色按当前区间依次调用所有专家GMM1/Activation，再依次调用所有专家GMM2；AIV1再按目标rank发送当前区间Combine。

```text
AIC : 当前宏Wave的全部GMM1 -> 当前宏Wave的全部GMM2 -> 下一宏Wave
AIV0: 当前宏Wave Activation 或W1展开 -> GMM2时按模式展开W2
AIV1: 下一宏Wave二级Dispatch -> 当前宏Wave各阶段角色工作 -> 当前宏Wave Combine
```

第一轮由AIV1在循环前接收首宏Wave，使AIC可以逐组等待输入后启动；后续预取范围与当前计算范围分别推进。当前宏Wave按专家区间组织，不以256-token预算把一个大专家拆进多个MTE Wave。

代码：[目标宏Wave与首Wave数量](../arch35/mega_moe_layered.h#L405)、[ProcessMoeExpertWave](../arch35/mega_moe_layered.h#L1132)、[ProcessMoeExpertWaveLoop](../arch35/mega_moe_layered.h#L1092)。

模式分工仍需展开看：A8W8/A4W4第一段Generic由AIV0 Activation，AIV1跳过对应计算；A8W4第一段AIV0展开权重，AIV1负责GM结果Activation。A4W4的第二段Activation是FP8，AIV0要做W2展开；A8W8第二段没有该展开。

代码：[RunGmm1WithEpilogue](../arch35/mega_moe_layered.h#L885)、[GroupMatmulWithCombine](../arch35/mega_moe_layered.h#L941)。

该分支不采用MTE的GMM2 token阈值lag。非W4模式仍有滚动分核的固定角色共振调整；不要把“宏Wave中先G1后G2”解释成这些调用之间新增全核屏障，tile/group ready才是数据约束。

</details>

<details>
<summary><strong>15.4 通道固定归属、批量Combine与组完成计数</strong></summary>

**执行核/同步双方：** AIC 写 GMM2 GM 并发布组计数；各目标 rank 的固定 owner AIV1 等组计数、搬整行、可选量化并提交 Combine。AIV0 不提交 Wave 通道通信，W4 场景只协作权重展开。Wave 尾部汇合则要全部 AIV0/AIV1 参加，而不只参加过通信的 owner 核。

Wave通信按 `ownerBlock=rank%P` 固定分配通道，只有该block的AIV1访问此rank的通道。一级Dispatch、二级拉取、Combine和最终Drain采用同一映射；这保证同一通道的锁、batch和提交counter不会由两个不同核并发推进。入口/出口跨rank握手另按物理AIV分peer，故锁交接必须在阶段边界完成。

代码：[GetChannelOwnerBlock/IsChannelOwner](../arch35/mega_moe_layered.h#L388)、[Wave取得固定归属锁](../arch35/mega_moe_layered.h#L1137)。

Layered 的 AIC 写好一个N tile后，先在本核 FIX_S 确认写出，再给该M-group的GroupSyncSlotLayout槽原子加1。AIV1按自己拥有的目标rank，从cumsum找到该rank在各专家中的行区间；处理到某M-group时读其完成槽，等值不小于 `ceil(H/256)` 才搬整行及metadata。

这里没有MTE普通Combine的配对tile ready/ACK：通信核按目标rank拥有行，而AIC按MN tile拥有计算，两种分工不一一对应。因此使用“该M-group完整H已写齐”的GM计数，把计算和通信分工解耦。

代码：[AIC组完成通知](../arch35_92/stage/mega_moe_gmm2_combine.h#L521)、[ProcessCombineRank](../arch35/stage/mega_moe_layered_combine.h#L92)。

对一个目标rank，Combine在当前宏Wave内跨专家积累WQE，满256token提交，宏Wave尾不足256也立即flush，不能为了凑满包阻塞尾部。通信数据可以是BF16或FP8+scale，量化发生在整行已可读之后。

Wave尾只有AIV1负责导出count、排空并释放其归属通道；之后所有AIV `PIPE_ALL + SyncAll<true>()`，确保没有未完成回传或UB发送。共享GMM2另有全C/V屏障，然后才是最终跨rank握手。

代码：[ProcessCombineExpertRange](../arch35/stage/mega_moe_layered_combine.h#L148)、[Wave尾Drain与全AIV同步](../arch35/mega_moe_layered.h#L1186)。

</details>

<details>
<summary><strong>15.5 两次跨rank握手和最终Unpermute为何这样放</strong></summary>

**执行核：** 两次握手均由全部 2P 个物理 AIV0/AIV1 按 peer 分工；最终 Unpermute 也由这两类 AIV 按本地 token 均分。它们不是 Wave 中固定通道 owner AIV1 的同一组通信分工。AIC 不做握手或聚合，只在共享 GMM2 的全 C/V 屏障中确保共享输出写完。

Layered跨rank握手的实现先向本AIV负责的全部peer异步写sequence，再进入逐peer等待；排除本rank。等peer的GM值到齐后对通道Drain、Unlock，最后全AIV汇合。它不是MTE那种逐peer直接GM写/等。

第一次在全量本地量化/relay队列构建后：沿用 `SendMaskCal` 取得的通道锁，确认跨卡路由阶段，释放这些锁，让后面的固定owner AIV1接管Wave通道。relay记录还具有各token-ready协议，不能把第一次握手单独当成二级Dispatch全部完成。

第二次在全部Wave回传及共享GMM2后：Wave固定owner已Drain/Unlock；握手工人可能是另一个物理AIV，必须重新Lock它负责的peer，再确认结果回传完成。尾部全AIV同步让任何Unpermute工人都能消费来自所有peer的结果。

代码：[Layered CrossRankSyncInWorldSize](../arch35/mega_moe_layered.h#L838)、[两次握手的调用位置](../arch35/mega_moe_layered.h#L1236)。

最终Unpermute仍按原token累加MoE TopK结果，非预取乘权、预取只累加，量化通信先反量化，最后加共享结果并写Y。但Layered共享读函数没有MTE的逐组counter等待，因为 `ProcessSharedExpertGmm2` 已经全C/V汇合，全部共享GM结果已可读。

代码：[Layered共享GMM2后的屏障](../arch35/mega_moe_layered.h#L1047)、[UnpermuteSharedExpert](../arch35/stage/mega_moe_layered_combine.h#L273)、[Layered Unpermute](../arch35/stage/mega_moe_layered_combine.h#L395)。

该类当前没有MTE的DedupPrepass/rowDesc/canonical Combine状态机，不应把MTE去重节的具体ready和成员合并协议补写成Layered也在执行。

</details>

</details>

<details>
<summary><strong>16. 按单核再串一次：只看实际工作，不看函数名</strong></summary>

<details>
<summary><strong>16.1 Arch92 当前MTE：唯一AIV和AIC</strong></summary>

**唯一AIV：** 先均分输入/清flag，发自己负责的专家route index，等全AIV准备条件，通知AIC并发count；有共享专家就先等本AIC共享GMM1结果、做共享Activation。然后读count/cumsum、可选建去重表，预取Dispatch首两个Wave。每轮准备后继Dispatch，再按本核tile顺序等MoE GMM1、做Activation；按lag安排消费GMM2/Combine。量化/去重在专家完全结束后按行发送。末尾导出/清count并全AIV汇合；共享GMM2不在此做矩阵乘，继续出口跨卡握手和Unpermute，读到共享组时再等其counter，写Y并排空输出。

**AIC：** 等输入准备；有共享专家就先做共享GMM1。MoE每轮按scheduler取tile，等Dispatch行数到齐，做GMM1；按lag进入GMM2时等Activation完整K输入到齐。普通Combine每tile通知唯一AIV并管理credit；量化/去重在专家终点发布本核专家ready。MoE之后继续做共享GMM2，按N tile发布共享counter；末尾收硬件ACK，恢复控制状态返回。

AIC与AIV不是同一墙钟时间进入下一Wave；对端卡更不要求按专家编号锁步。两核依靠具体tile/group/专家状态决定何时能读，源码主循环次序约束的是各自工作序。

代码：[Arch92 ProcessMoeExpertStages](mega_moe_wave.h#L526)、[公共 ProcessWave](../arch35_92/mega_moe.h#L1078)。

</details>

<details>
<summary><strong>16.2 Arch35 MTE A8W8：两类AIV何时确实不同</strong></summary>

**AIV0：** 输入量化/清零与AIV1一样；无共享时也分担路由发送，有共享时不负责路由而尽早通知AIC、做共享模式规定的计算。MoE A8W8每轮等GMM1结果、做Activation，不负责稳态Dispatch/普通Combine；去重开且 `N1=tiling.hiddenDim<=2048` 时还可先帮建表；量化/去重末专家可加入独立Combine。尾部负责清count，然后与AIV1一样出口握手、Unpermute。

**AIV1：** 输入量化/清零；有共享时负责全部路由，有共享A8W8/A4W4时跳过共享Activation而先推进MoE通信，有共享A8W4时先完成共享Activation。MoE稳态准备Dispatch下一Wave，跳过GMM1计算但补齐分核游标；后面普通tile Combine或专家级量化/去重Combine。尾部block0导出count，所有AIV同步后做出口握手和Unpermute。

**AIC：** 数据依赖与Arch92相同，但非预取Generic GMM1往AIV0 UB写，等的是AIV0 ready/ACK；GMM2普通Combine通知的是AIV1。这两个消费者不同，允许AIV1 Dispatch与AIV0 Activation并行，不能通过删除AIV0后仍原样保留两条角色等待。

代码：[Arch35 A8W8 Wave](../arch35/mega_moe_wave_a8w8.h#L684)、[共享模式分派](../arch35_92/mega_moe.h#L587)。

</details>

<details>
<summary><strong>16.3 Arch35 MTE W4：不能笼统说“AIV0只有prologue”</strong></summary>

**A8W4：** AIV0在两段GMM都展开FP4权重送AIC L1，AIV1交替下一Wave Dispatch/当前GMM1 GM Activation，再做Combine。AIC每个K块等权重、每个M-group等数据，与AIV0/AIV1各有独立同步链。

**A4W4：** 第一段直接FP4×FP4，AIV0做Activation；Activation输出FP8后第二段才由AIV0展开W2。因此“Arch35 AIV0是可直接删掉的权重展开核”只适用于A8W4具体阶段，不能扩展到A4W4全部阶段。

当前Arch92统一Wave相当于把通信和Activation放同一AIV，同时由Tensor直接完成支持的矩阵输入类型；它没有复制Arch35所有W4主循环的逐专家交替和第二子角色。两者同样可以共享stage数学实现，但调度顺序并不完全相同。

**性能判断必须按基线模式：** 对照Arch35 A8W8是独立Activation/通信角色变为同核，主要看重叠损失；对照Arch35 A8W4，AIV1本来就兼做这些工作，主要看调度粒度及权重展开替代，而且Arch92当前没有开放同模式；对照Arch35 A4W4，还改变第二段数据精度和实现。不能用同一句“把两个AIV工作合并”解释这三种情况。阶段差异的统一核对表在第18节。

代码：[Arch35 A8W4 Dispatch/Activation交替](../arch35/mega_moe_wave_a8w4.h#L179)、[Arch35 A4W4实际Generic调用](../arch35/mega_moe_wave_a4w4.h#L117)、[Arch92统一Wave](mega_moe_wave.h#L526)。

</details>

</details>

<details>
<summary><strong>17. 其他分支和阅读边界：哪些改变工作量，哪些不改变流水</strong></summary>

<details>
<summary><strong>17.1 容量裁剪、零专家、零任务以及尾块</strong></summary>

**执行核：** 专家行数裁剪在各 Arch35 AIV1 / Arch92 唯一 AIV 的 count 表计算中完成；发送侧下标容量限制由 35无共享 AIV0/AIV1、有共享 AIV1，或92唯一AIV执行。AIC 和相应 Activation/权重展开角色按裁剪后的行数规划 GMM tile；Dispatch/Combine AIV 只搬本核实际分到的有效行，不能将这些动作统称成“所有核裁剪数据”。

count表的cumsum记录接收到的原始路由累计量；供GMM使用的专家行数还要按全局 `maxOutputSize` 裁剪。设专家原始区间为 `[previousEnd,end)`，实际行数是：

```text
min(end,maxOutputSize) - min(previousEnd,maxOutputSize)
```

例如原始专家行数5、5、capacity=7：原cumsum仍能定位各源rank片段，实际GMM专家行数是5、2；Dispatch只搬全局 `[0,7)` 内的交集，不为被裁掉的行发布ready。容量是本卡紧凑输出总容量，不是“每个专家都最多7行”。

代码：[ComputeExpertCountTablesVF](../arch35_92/stage/mega_moe_expert_count.h#L95)、[Dispatch范围裁剪](../arch35_92/stage/mega_moe_token_dispatch.h#L415)。

路由发送端还按本专家记录的剩余容量限制复制下标，累计count对应实际复制量。该限制在 `SendSelectedTopkIdsIndex`；不要拿TopK匹配总次数直接替代它最终发布的count。

代码：[SendSelectedTopkIdsIndex的容量限制](../arch35_92/stage/mega_moe_send_mask.h#L119)。

实际行数0的专家由Wave游标跳过，不执行MMAD/Activation；没有本核tile的problem按总tile数推进分核游标；没有本核Dispatch/Combine行时不做行搬运。尾M-group等待真实行数而不是固定256；Activation预取尾子块也按实际M计算完成数。空任务不取消入口/出口全AIV协议，第14.2节分别解释了哪些通知不能省。

代码：[专家范围规划和零专家跳过](../arch35_92/common/mega_moe_utils.h#L406)、[无tile处理](../arch35_92/common/mega_moe_utils.h#L165)、[尾组GMM1输入等待](../arch35_92/stage/mega_moe_gmm1_activation.h#L34)。

这些说明仅描述kernel的裁剪/调度。超容量丢弃的输出语义、哪些组合允许进入kernel，还受host/API合同约束；本文不把“kernel里有保护”当作整个算子输入合法性的证明。

**行为一致：** 相同capacity/形状/量化配置时，零专家跳过、尾块、无tile推进及实际行裁剪算法一致。若不同芯片拿到不同Wave预算或启动核数，problem划分/每核工作量可能不同，但不能在未对齐参数时将其归为新增的kernel算法差异。

</details>

<details>
<summary><strong>17.2 ND/NZ、FP8/FP4、scale以及interweave</strong></summary>

**执行核：** 布局选择和视图构造在调用该 stage 的 AIC/相应 AIV 各自发生；A/B及ScaleA/ScaleB的矩阵读取、MMAD由 AIC 执行。Arch35 自定义 A8W4 的 B 展开由 AIV0 执行，Activation由AIV1；Generic GMM1的 Activation由Arch35 AIV0 / Arch92唯一AIV执行。类型/layout变化不把矩阵乘转移到AIV，也不把发送工作转移到AIC。

ND/NZ主要选择B的物理layout、C0和GM搬运方式。Generic A/B分别按自己的元素类型取C0；FP4打包元素需要按两个逻辑值一存储单元计算跨度。自定义A8W4另固定B的C0=32，FP4先经prologue展开后才进入AIC。它们都仍把ScaleA、ScaleB传给矩阵乘，不能因为FP4展开为FP8同值就把ScaleB删掉。

代码：[Config中的C0和layout](../arch35_92/common/mega_moe_gmm_common.h#L82)、[自定义W4展开](../arch35_92/blaze/prologue/block_prologue_mx_fp8fp4.h#L259)。

Generic GMM1调用QGMM/concat接口，logical N描述Gate+Up合计宽度，调用的singleShape和B slice用半宽；本仓针对NZ的view保留concat搬运标记，针对输出的view保留有效M/N和UB/GM目的。普通tensor视图沿用Tensor仓对应实现。共享与MoE调用同一组stage，区别主要是地址、角色、输入ready和handoff策略，不是另写一份SwiGLU数学。

代码：[GMM1半宽参数及成对输出](../arch35_92/stage/mega_moe_gmm1_activation.h#L75)、[本仓concat视图与copy重载](../arch35_92/blaze/gemm/tile/copy_gmm1_concat.h#L22)。

上述layout、element和scale分支会改变字节数、切片和MMAD内部策略；不会把MTE出口跨卡握手变成C核工作，也不会改变“GMM2必须等整个Activation K输入到齐”的数据依赖。编译门控仍以第1节和实际entry为准，保留类型模板不等于当前A8W4已经开放。

**共同接口与实际差异：** Generic GMM1两边都用QGMM/concat表达Gate/Up，ND/NZ选择本身不是Arch92新增interweave，不能因此推导多一次GMM。真正的差异是是否传复用context、GMM1 UB/GM目的和当前A4W4第二段输入类型，分别对应初始化、结果流量和量化/矩阵实现。权重 L2 提示还需按量化模式区分，下一节单独核对，不能笼统称为 Arch92 独有策略。

</details>

<details>
<summary><strong>17.3 不要把接口占位、历史同步名字或异常记录当成新阶段</strong></summary>

当前MTE入口带有bias、sharedBias、maskBuffer等参数，但实际 `op.Init` 未传入这些bias/外部maskBuffer；`xActiveMask`虽然传进公共Init，该实现没有存储或读取它。因此不能根据参数名编造“每个AIV先按active mask过滤token”或“GMM后加bias”这样的当前阶段。

代码：[Arch35入口实际透传](../arch35/mega_moe_apt.cpp#L96)、[Arch92入口实际透传](mega_moe_apt.cpp#L65)、[公共Init的参数绑定](../arch35_92/mega_moe.h#L300)。

本版MTE入口完成条件已经放在count+epoch协议中；当前主流程没有旧的 `SyncInputAcrossRanks` 或 `SyncTopkValidIndexSendCores` 调用。workspace仍保留某些兼容字段也不代表对应旧协议在运行。出口的 `CrossRankSyncInWorldSize` 则确实执行，位置在MoE回传之后。

代码：[输入发布count+epoch](../arch35_92/mega_moe.h#L969)、[当前ProcessWave完整调用序](../arch35_92/mega_moe.h#L1078)。

exceptionDump登记通信窗口前部、阶段枚举和GMM循环计数，帮助判断卡在哪一阶段；它不是Dispatch/Activation数据就绪协议，更新stage不能让其他核开始计算。本解读没有添加任何debug打印或set_cond，也没有修改同步和kernel逻辑。

代码：[登记异常区](../arch35_92/mega_moe.h#L316)、[阶段记录的调用位置](../arch35_92/mega_moe.h#L1084)、[Wave循环计数](mega_moe_wave.h#L593)。

</details>

<details>
<summary><strong>17.4 权重 L2 提示：A8W8 共同策略与 A4W4 调用差别</strong></summary>

**执行核：** AIC 在 Wave Generic GMM1/GMM2 建立 B/ScaleB 视图时设置 L2 提示。**Arch35 A8W8 与 Arch92 A8W8 的两段调用都显式传入 `isWholeExpert`，使用相同的公共判定函数。** 不能根据 stage 的默认参数 `allowWeightL2Bypass=false` 推断 Arch35 实际调用没有启用，也不能将它列为 A8W8 迁移新增行为。AIV 不执行矩阵权重缓存策略。

两边都不是对所有权重关闭 L2：必须一次 problem 从第 0 行覆盖完整专家、M 不超过一个 tile，且满足对应物理行的对齐条件，才让 B 或 ScaleB 绕过 L2 填充；B 与 ScaleB 分别判定。跨 Wave 的热点专家即使尾 slice 很小，也保留正常提示。相同专家在不同 Wave 划分下可能触发不同结果，但判定策略本身相同。

代码：[Arch35 A8W8 GMM1 显式透传](../arch35/mega_moe_wave_a8w8.h#L295)、[Arch35 A8W8 GMM2 显式透传](../arch35/mega_moe_wave_a8w8.h#L339)、[Arch92 GMM1 条件及透传](mega_moe_wave.h#L310)、[Arch92 GMM2 条件及透传](mega_moe_wave.h#L356)、[B/ScaleB 各自的 cache 判定](../arch35_92/common/mega_moe_gmm_common.h#L435)。

**A4W4 的实际调用差别：** Arch35 A4W4 的 Generic GMM1 未传这项参数，使用默认 false；Arch92 A4W4 共用统一 Wave，仍传 `isWholeExpert`，满足公共条件时可使用 bypass。Arch35 A4W4 的第二段走自定义 A8W4，不能把 Generic 的默认参数套给它。Arch35 A8W4 也走自定义实现；共享专家调用不是这里的 MoE whole-expert 策略启用点。

代码：[Arch35 A4W4 GMM1 的实际参数](../arch35/mega_moe_wave_a4w4.h#L117)、[Generic GMM1 的默认值](../arch35_92/stage/mega_moe_gmm1_activation.h#L635)、[Arch35 W4 第二段选择](../arch35_92/mega_moe.h#L801)。

**潜在性能影响：** A8W8 没有因迁移新增的 cache 策略开销。A4W4 第一段满足条件时，Arch92 可能减少一次性权重对 L2 的填充，为其他复用数据留空间；实际效果受芯片缓存、并发读取及权重布局影响，必需的 GM 读取仍然存在。它是该模式的软件调用差别，不来自 AIV 职责迁移；评估 A4W4 时应记录是否触发条件，再与 FP4 输出、MMAD 实现和阶段重叠的变化分别归因。

</details>

</details>

<details>
<summary><strong>18. 差异与性能核对表：哪些真正变化，哪些只是全核资源不同</strong></summary>

<details>
<summary><strong>18.1 真正的职责、重叠、交接及实现差异</strong></summary>

以下固定逻辑核数P只为辨认软件差异；实际Arch35/Arch92芯片资源可能不同。涉及A8W4的行是调度来源/旧实现对照，不表示当前Arch92该模式已开放。各行的具体分工和同步链已在对应小节就地说明。

| 对比场景 | Arch35 -> Arch92真正改变了什么 | 可能的不利影响 | 可能减掉/缓解的成本 | 详述小节 |
|---|---|---|---|---|
| 有共享时输入发布 | AIV1发路由、AIV0独立通知 -> 唯一AIV先发下标再通知 | TopK扫描/下标提交可能进入共享GMM1启动关键路径 | 不能假设固有收益；AIC仍可与通知后的count搬运重叠 | 5.3、5.4 |
| 共享A8W8/A4W4之后启动MoE | AIV1可先通信、AIV0仍Activation -> 唯一AIV先共享Activation后MoE通信 | 共享计算和MoE count/建表/首轮Dispatch重叠减少 | 无新增全C/V屏障；仅等本核必要数据和UB | 6.2、6.3 |
| MoE A8W8 Dispatch/Activation | AIV1/AIV0并行 -> 同一AIV按阶段执行 | 当前Activation或后继Dispatch可能排队，AIC等输入更久 | lookahead仍保留，可与AIC矩阵计算重叠，但不是恢复两个AIV并行 | 8.2、8.3 |
| 以Arch35 A8W4为调度基线 | 单专家slice交替、启动W0 -> 整下一Wave先Dispatch、启动W0/W1 | 当前Activation可能等待更多下一范围搬运 | 下一Wave输入可能更早备齐；直接Generic无旧专职展开链 | 8.4、9.4 |
| 原UB-handoff的Generic GMM1 | AIC直接写AIV0两槽 -> GM结果及唯一AIV装载 | 增加BF16结果GM写/读；非预取MoE增加序号发布/轮询 | 不再由两UB槽复用ACK限制生产领先；共享仍受15-credit限制 | 9.2、9.3 |
| 已GM-handoff的TopK预取 | 数据交接仍GM、硬件协议仍ready/ACK，消费角色合用 | 唯一AIV忙时可晚消费、额度用满后AIC等待 | 没有额外一次相对Arch35的结果GM读写，不能重复计费 | 8.3、9.3、10.2 |
| cumsum生命周期，基线A8W8 | 留在Dispatch AIV1 UB -> GM保存、按需恢复 | 表读写和UB交接成本 | 支持同一AIV复用UB；相对Arch35 W4不是新成本 | 4、7.1 |
| Generic MMAD context | 局部对象 -> Arch92条件复用成员对象 | 配置切换仍Init，收益受专家/Wave排列影响 | 相同签名的相邻problem可省重复Init；动态M更新仍在 | 4、9.1 |
| 当前A4W4 Activation/GMM2 | FP8输出+W2展开、FP8×FP4 -> FP4输出、直接FP4×FP4 | FP4量化/打包和Tensor实现效率需评估；向量重叠减少 | data字节减少、无旧W2展开/跨核权重握手；scales不减半 | 8.5、9.4、10.3、11.1 |
| A4W4达到lag阈值 | Arch35该模式不lag -> Arch92统一Wave启用lag | 待消费范围和drain延迟增加 | 可能用下一GMM1覆盖AIC等尾部Activation | 8.6 |
| 去重prepass，N1<=2048 | 建表都用全AIV；Arch92取消协作AIV0独立读表及高半word UB保护gate | 实际工人数影响属于资源因素，不能算新增建表算法 | 减少重复读表/跨角色gate；N1>2048时两边无此协作差异 | 12.2 |
| 量化/去重末专家的进入方式 | 末专家都用全AIV；Arch35原AIV0转入Combine，Arch92无第二角色接管、量化仅扩行环 | 物理工人数及尾部带宽需单独对齐，不是每行新增任务 | 不需另一角色接管UB；buffer在途能力需评估但不等价于增核 | 11.4、12.4 |
| 导出/清count | AIV1 block0导出、AIV0清 -> 唯一AIV兼做 | block0导出/清零可少一项跨核重叠 | 清理量和全AIV协议相同，没有多清一次 | 13.1 |
| A4W4 第一段小专家 cache hint | Arch35 GMM1 默认 false -> Arch92 传完整专家条件 | 实际缓存收益不确定，跨芯片不可直接外推 | 满足公共条件时可能减少一次性权重填充；A8W8 两边原本相同 | 17.4 |

当前更应关注**哪个生产者被同核前序工作延后，导致哪个消费者多等**，而不是仅数“删了/加了几条同步语句”。这些因素会互相影响，不能将每行潜在成本直接相加成总性能差。

</details>

<details>
<summary><strong>18.2 行为一致的阶段：不人为包装成迁移差异</strong></summary>

| 工作 | 两边相同的任务/协议 | 不应作出的结论 |
|---|---|---|
| 入口reset、BF16量化/打包 | 全物理AIV分任务，相同数据格式和本核buffer保护 | 2P/P不等于新增量化步骤或必然两倍耗时 |
| 无共享时路由发布 | 全物理AIV均分专家，发送下标及count+epoch | 配对通知核标识不同不等于多发一轮路由 |
| count/cumsum计算与Dispatch | 同一epoch校验、连续行分核、metadata和按行ready；正常固定P都用P个Dispatch角色 | “92全AIV参与”不等于该阶段消费者变少；cumsum保存差异另算 |
| GMM tile分配和输入等待 | 同一scheduler/滚动游标、真实行数和完整Activation K就绪目标 | 角色改变不等于另一套分核或FP4使ready目标减半 |
| A8W8 权重 L2 提示 | 两段 GMM 都传完整专家条件，公共函数分别判断 B/ScaleB | stage 默认 false 不等于 Arch35 实际调用 false，不能列为 Arch92 独有优化 |
| 普通/专家级Combine稳态 | 同一逐tile硬件链或全部P个AIC专家ready，固定P为P个通信角色 | 末专家扩工差异不能泛化到所有专家 |
| 末专家Combine、N1<=2048的去重建表 | 两边都是全物理AIV执行相同的阶段算法 | 2P/P本身不构成职责迁移；仅另列35协作角色的UB交接/gate、92的buffer管理 |
| 去重扇出、LAST合并和canonical | 数据读/扇出/成员加权语义一致 | 不要将prepass工人改变说成多做一次数学合并 |
| 共享A8W8 GMM2 | AIC写本卡GM并发组counter，后续全AIV按token读，无MoE回传 | 没有Arch92新增的共享后全C/V屏障 |
| 出口握手和Unpermute | 全物理AIV分peer/本地token，相同完成与聚合协议 | 少一类子角色并非在此新增同核异类任务；核数量只属资源因素 |
| 尾块、零专家、容量裁剪 | 相同有效行与游标规则 | 未对齐Wave预算/启动参数的差异不能自动算kernel迁移算法变化 |

“行为一致”不等于实际耗时相同：同样全AIV算法可能因物理工人数、核吞吐、带宽或对端到达时间而耗时不同。但那是资源/配置/前序依赖，不是本阶段新增职责，应与18.1的流水差异分开归因。

</details>

<details>
<summary><strong>18.3 如何验证潜在性能影响，而不把推测写成结论</strong></summary>

**观察对象仍按核：** AIC看GMM矩阵计算、FIX写回、等Dispatch/Activation/ACK的时间；Arch35分别看AIV0的Activation/展开与AIV1的通信，Arch92看唯一AIV的Dispatch、Activation、Combine串行区间。全AIV同算法阶段则比较实际工人数、每核工作量和最后完成核，避免假设同核数等吞吐。

1. 对齐MTE拓扑、实际输入/路由、MoE及共享类型、actMode、TopK预取、去重、Combine量化和capacity；同时记录启动P、mGroupsPerWave和L1/tile配置。不同芯片不能强行认为这些参数必相同；未对齐的部分单独注明。
2. 分开看启动、稳态、末专家和drain。输入发布/共享重叠主要影响启动，Dispatch/Activation合用影响稳态，末专家扩工和lag收尾主要影响尾部，不能只用一个总时间解释所有机制。
3. 用已有流水/指令日志检查消费者为何等待：其自身晚进入，还是生产者数据晚完成，或credit确实用满。不要把每个Wait的数量当作阻塞时长，也不要用跨芯片总耗时直接证明某段公共函数退化。
4. 对GM交接区分新增有效数据字节、实际传输、轮询重试和排队；对FP4区分data减少、scales不变与量化/MMAD类型变化；对context统计哪些problem真正省了Init，哪些配置切换仍Init；对L2记录哪些专家满足bypass条件。
5. 有无共享、预取开关、N1阈值两侧、单个重专家/多个小专家、尾专家大小、lag阈值两侧分别覆盖。只覆盖一个分支不能外推所有Arch35/Arch92场景。

这些是评估依据，不是要求现在改kernel或新增打点。**本次没有性能实测，也没有添加GM哈希扫描、Scalar逐元素统计、debug打印或任何新同步。** 模拟器指令耗时与芯片真实带宽不是同一量纲；文档只给条件化的潜在影响，最终方向和幅度需要对应环境验证。

</details>

</details>
