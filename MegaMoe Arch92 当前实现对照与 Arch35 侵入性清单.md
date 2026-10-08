# MegaMoe Arch92 当前实现对照与 Arch35 侵入性清单

## 1. 对照范围与结论

本文对照三份代码：

1. `_910b2`：`D:\gitcode\ops-transformer_910b2` 的 `a9938e985`，独立维护的 Arch92 A8W8 实现。
2. `_5` Arch92：`D:\gitcode\ops-transformer_5` 的 `ab438a48a` **加当前未提交工作树**，正在迁入的实现。
3. `_5` Arch35：同一工作树中的 Arch35 入口、Wave 与 `arch35_92` 公共层；统计“原代码被改动多少”时，以 `ab438a48a^` 的 Arch35 为迁入前基线。

本文记录 **A8W8 首次迁入阶段的静态快照**；随后 Arch92 已接入 A4W4 源码路径、统一 Wave 命名，A8W4 仍被门禁拒绝。最新改动及未验收项见 [A4W4/A8W4 实施记录](/D:/gitcode/ops-transformer_5/MegaMoe%20Arch92%20A4W4与A8W4统一实施记录.md)。下文的文件统计和“仅 A8W8”描述均按该历史快照理解，不代表当前支持矩阵。`_5` 的实现不是 `_910b2` 的逐文件复制：它使用 `_5` Arch35 已有的 QGMM interweave 和较新的 Expert Wave/Host tiling，把 Arch92 专属调度留在 `arch92`，把共用计算放在 `arch35_92`。[组合判定](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:635)。

这是一份**静态代码对照**。`_910b2` 曾有运行结果，不代表 `_5` 的 QGMM、调度和 Workspace 已通过同等编译、精度或性能验证。

## 2. 与 `_910b2` Arch92 的差别

### 2.1 文件所有权与入口签名

`_910b2` 的 tiling、基类、stage、Workspace 和量化 epilogue 都在独立的 `arch92` 树下，Kernel 从算子级 `mega_moe_apt.cpp` 进入。`_5` 的 Arch92 只保留专属入口和 A8W8 Wave，通用类型、stage、Blaze、Workspace 与 Host tiling 由 `arch35_92` 提供。两边都指定 `KERNEL_TYPE_MIX_AIC_1_1`。对照：[旧入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:33)、[新入口](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_apt.cpp:63)、[公共基类](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:53)。

`_910b2` 的 A8W8 Wave 模板主要传一种权重类型和一个 NZ 布尔值，MoE/shared 使用同一套推导。`_5` 遵循本仓接口，分别传 MoE/shared 权重类型、各自的 GMM1/GMM2 format、量化模式、topk index 类型；ND/NZ 和共享专家的类型由这些参数确定。这里是**接口与实例化能力的差别**，不是宣称新的 Arch92 量化模式已经放行。[旧 Wave 模板](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:40)、[新入口模板](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_apt.cpp:51)、[Host 能力门禁](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2603)。

### 2.2 BlockMmad 与 Gate/Up 组织：最重要的计算差别

`_910b2` 的普通 A8W8 使用 `MatmulWithScaleMx`（QBMM）；GMM1 对 Gate 和 Up 的两段权重循环调用 BlockMmad，分别写到输出的两段地址。`_5` 使用 Arch35 现有的 `GroupedMatmulWithScaleMx`（QGMM）：GMM1 传入 Gate 起点和单半 N，QGMM 内部读取 Gate/Up 两半并形成完整输出，调用方只提交一次 BlockMmad。GMM2 也使用同一个 QGMM 类型，但走普通完整 N 路径。因此不能把 `_910b2` 的 BlockMmad Init、L1 参数或 epilogue 双缓冲写法机械搬到 `_5`。[旧 policy](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:71)、[旧 GMM1 两次调用](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:116)、[新 policy](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_gmm_common.h:85)、[新 GMM1 一次调用](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:72)。

`_5` 没有复制 `_910b2/arch92/blaze/epilogue`；使用 Arch35 已有的 Gate/Up 拼接搬运视图和公共 activation/量化 epilogue。QGMM 的 GMM1 Init 把物理计算 tileN 设为完整 tile 的一半，GMM2 使用完整 tileN；每个新 problem 调用 `UpdateParamsForNextProblem`。这改变的是实现接口和缓冲组织，不能仅凭 GM 输入相同推断两版精度或性能相同。[拼接视图](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/blaze/gemm/tile/copy_gmm1_concat.h:36)、[QGMM Init](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_gmm_common.h:409)、[公共 epilogue](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/blaze/epilogue/block_epilogue_activation_mx_quant.h:1)。

### 2.3 Wave 粒度与阶段顺序

`_910b2` 在 Wave 中按 `ARCH92_GMM1_TILES_PER_JOB_TARGET=2` 现场算每 Wave 的 M group 数；`_5` 用本仓 Host tiling 的 `mGroupsPerWave`，由 GMM1/GMM2 tile 数、AIC 数和形状共同决定。相同输入不保证两版具有相同 Wave 边界或相同的 GM 同步次数。[旧 Wave 目标](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:18)、[新 Host 计算](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:142)、[新 Wave 使用](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:234)。

`_910b2` 在专属 `Process()` 内准备输入、共享专家和跨卡同步，随后使用一个相对直接的 Dispatch/GMM1/GMM2 专家循环。`_5` 通过公共 `ProcessWave()` 负责输入准备、共享专家、同步和输出收尾，Arch92 专属 `ProcessMoeExpertStages()` 负责逐 Wave 调度：唯一 AIV 按顺序先准备下一 Dispatch Wave，再处理当前 GMM1/activation；大 token 形状可让 AIC 与 AIV 同步滞后一拍执行 GMM2/Combine。这是控制流差别，不只是函数被移动。[旧主循环](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:247)、[新公共主流程](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:782)、[新 Arch92 主循环](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:555)。

### 2.4 保留的 Arch92 语义与实现位置变化

两版都必须按 **1C1V** 计算逻辑 block、AIV job 和 Combine consumer；`_910b2` 使用独立的 Arch92 context/Host tiling，`_5` 分别通过公共硬件策略与 Host DAV 策略选择 1C1V。这里主要是实现位置变化，不能把 Arch35 的 `GetTaskRation()` 或 AIV 数乘 2 套到 Arch92。[旧核映射](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:19)、[新硬件策略](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_arch_policy.h:69)、[新 Host 比例校验](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2574)。

两版都为普通 Arch92 A8W8 维护跨 Wave 的 Dispatch cumsum，并让非 prefetch GMM1 结果经 GM 交给唯一 AIV。`_5` 把原本散在 `_910b2` 专属文件里的三个选择显式放进 Execution Policy：`PERSIST_DISPATCH_CUMSUM=true`、`USE_GM_HANDOFF=true`、`WORKSPACE_LAYOUT=GM_HANDOFF_EXPERT_WAVE`；普通 GMM1 使用 `UseGmHandoff`，共享专家仍沿用公共基类的既有处理。差别在复用边界和具体同步实现，**不是** `_5` 取消了 GM handoff。[旧 Arch92 策略](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:19)、[新执行策略](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_execution_policy.h:18)、[新 GMM1 GM 交接](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:139)、[新 Dispatch 恢复](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:234)。

`_910b2` 把一个 BlockMmad context 传给 routed/shared GMM1/GMM2，并用类型断言要求四段一致。`_5` 仍在 GMM1/GMM2 之间复用同类型 context；若 routed 与 shared 类型不同，则各持有一份。`Init` 会按 K、tile 与 L1 配置变化刷新，不等于构造后绝不重新 Init。[旧单 context](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:81)、[新条件复用](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:79)、[Init 刷新](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_gmm_common.h:409)。

### 2.5 Host tiling 与 Workspace

`_910b2` 使用独立的 Arch92 tiling，写死 Arch92 的 1C1V 和 GM handoff Workspace 布局。`_5` 在同一 Host tiling 文件中按 DAV 校验 Arch35 1C2V/Arch92 1C1V，并在 Host 和 Kernel 两端选择相同的 Workspace policy；`_5` 还继承本仓较新的 Dispatch/SendMask/Unpermute 自适应 UB 预算、Wave 切分和 Combine 计数规则。两份 tiling 数据字段和资源分配算法不能只按文件名视为等价。[旧 Host 入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2031)、[新平台选择](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2574)、[新自适应预算](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:1468)、[新 Workspace 提交](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2626)。

## 3. 与 `_5` 当前 Arch35 的差别

| 检视项 | `_5` Arch35 | `_5` Arch92 |
| --- | --- | --- |
| 入口与能力 | `ascend950`，1C2V；保留 A8W8/A8W4/A4W4 与 URMA/Layered | `ascend910_96`，1C1V；当前只放行 A8W8/MTE |
| AIV 职责 | AIV0 做 GMM1 epilogue，AIV1 做 Dispatch/部分 Combine，可并行 | 唯一 AIV 分时承担两种职责；先安排 Dispatch，再处理 GMM1 epilogue |
| A8W8 BlockMmad | QGMM，Gate/Up interweave | 同一公共 QGMM，Gate/Up interweave；区别在调用的 Wave/交接协议 |
| 普通非 prefetch GMM1 | AIC 到 AIV 的 UB ping-pong | AIC 写本地 GM，AIV 等序号并从 GM 搬入后做 activation |
| Dispatch cumsum | A8W8 默认不强制跨 Wave GM 备份；W4 按原逻辑使用备份 | A8W8 持久备份，UB 被后续阶段复用后按下一 Wave 范围恢复 |
| Workspace | 默认条件布局 | 额外保留 cumsum、GMM1 BF16 GM 输出和每 AIC 顺序 flag；其余共用布局公式 |
| GMM2/Combine | 公共计算数据面，1C2V 角色/consumer 口径 | 公共计算数据面，1C1V 角色/consumer 口径；Wave 控制流独立 |
| 共享专家 | 使用公共共享 GMM1/activation/GMM2 | 使用同一公共实现，由 1C1V 核映射适配；本次没有发明另一套共享专家算法 |

Arch35 入口仍指定 1C2V，[代码](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp:104)；Arch92 入口指定 1C1V，[代码](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_apt.cpp:74)。两条 A8W8 Wave 的先后安排见 [Arch35](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:506) 与 [Arch92](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:555)。普通 GMM1 的分支与 GM flag 在 [公共 stage](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:435)，cumsum/Workspace 选择见 [Execution Policy](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_execution_policy.h:18)。

特别说明：Arch35 **A8W4 也支持 Gate/Up interweave**，通过其专用 BlockMmad 的 `Init(..., true)`、`concatHalfN_` 与 W4 Prologue 完成；Arch35 A8W8 和本次 Arch92 A8W8 使用 QGMM interweave。不能把“Arch92 用 QGMM”写成“Arch35 不支持 interweave”。[A8W4 Init](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:238)、[A8W4 concat 参数](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/blaze/gemm/block/block_mmad_mx_fp8fp4.h:470)。

## 4. 对原 Arch35 代码的侵入性修改

### 4.1 统计口径

以 `_5` 的 `ab438a48a^` 为迁入前 Arch35 基线，把其 `arch35/` 原文件按相对路径映射到当前 `arch35_92/`；`mega_moe_arch35.h` 对应新的 `mega_moe.h`。按 Git clean-filter 后的内容哈希对比，**32 个有原文件对应的 Kernel 公共文件中，24 个内容未变，8 个内容改变**。另有 2 个 Kernel 公共策略文件（硬件、执行）是新增文件；Host 侧 1 个原 tiling 文件有内容修改，1 个 Host 架构策略文件是新增文件。这个数字统计的是**从原 Arch35 到当前工作树的累计变化**，不是只看本轮未提交 diff。

当前 `_5` HEAD `ab438a48a` 已经包含第一次“移动到 `arch35_92`”的示例；相对该 HEAD 的本轮未提交改动只命中 **7 个既有 Kernel 公共文件 + 1 个既有 Host tiling 文件**，另新增 `mega_moe_execution_policy.h`。`arch35_92/common/mega_moe_gmm_common.h` 和 `stage/mega_moe_gmm2_combine.h` 虽可能被 Git 报 `M`（行尾变化），当前 `git diff` 对 HEAD 无内容差异；其 QGMM/角色处理分别来自迁入前 Arch35 和第一次示例，不计作本轮新增逻辑。旧设计评估中的文件数、增删行数是旧快照，不能直接当作当前统计。

### 4.2 八个被改动的 Kernel 公共文件

| 文件 | 相对原 Arch35 的实际改动 | 对 Arch35 的影响/侵入程度 |
| --- | --- | --- |
| `mega_moe.h` | 基类加默认 Execution Policy；Workspace/cumsum 按策略选择；共享 GMM1/GMM2 与 `ProcessWave` 传入可复用 BlockMmad context。 | **高**：修改公共模板和阶段调用签名；Arch35 走默认策略、仍可传空 context。[代码](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:53) |
| `stage/mega_moe_gmm1_activation.h` | 通用 GMM1 增加 `UseGmHandoff`，将原 GM 写出路径用于非 prefetch A8W8，新增顺序 flag 等待/通知；GM AIV 消费函数由原 W4 场景泛化。 | **高**：改了计算阶段的分支和同步签名；默认 `false` 保留 Arch35 UB 路径。[代码](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:435) |
| `common/mega_moe_workspace.h` | `WorkspaceLayoutPolicy` 显式决定 MTE A8W8 是否分配顺序 flag、cumsum 备份和 GMM1 输出；Host/Kernel 共用同一布局类。 | **中高**：改地址和容量计算，必须与 Host 同步；默认策略维持 Arch35 原条件。[代码](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_workspace.h:92) |
| `stage/mega_moe_gmm2_combine.h` | 角色门禁、AIV 数与逻辑 job 映射改为硬件策略；不另造 Arch92 Combine 算法。 | **中**：改同步参与者选择，1C2V/1C1V 都经过此文件；Arch35 策略仍返回两 AIV 角色。[代码](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm2_combine.h:346) |
| `stage/mega_moe_token_dispatch.h` | Dispatch 职责门禁从硬编码子块改为 `RunsDispatchVectorWork()`。 | **低中**：函数主体不变，Arch35 策略对应原 AIV1。[代码](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_token_dispatch.h:365) |
| `common/mega_moe_utils.h` | GM 清零的 AIV job 编号/总数从策略取得；设备代码条件宏也改成公共宏。 | **低中**：影响清零分工与预处理门禁，Arch35 映射保持原口径。[代码](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_utils.h:263) |
| `common/mega_moe_peermem.h` | 扩展 device 编译宏识别到 Arch92，使用统一 `MEGA_MOE_IS_KERNEL_COMPILE` 门禁。 | **低**：主要是编译条件，非 peermem 地址公式修改。[代码](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_peermem.h:20) |
| `stage/mega_moe_workspace_reset.h` | 原 Arch35 设备宏改为统一 device 宏，使用公共 Workspace 布局记账的清零范围。 | **低**：门禁变化，未另写第二套清零公式。[代码](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_workspace_reset.h:18) |

新增策略文件本身不算改动原 Arch35 文件，但会被公共代码调用：硬件角色映射见 [Arch Policy](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_arch_policy.h:21)，默认/Arch92 执行策略见 [Execution Policy](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_execution_policy.h:18)。**BlockMmad QGMM 特化、Gate/Up 拼接搬运视图、公共 activation epilogue 与 A8W4 BlockMmad 没有因本轮 Arch92 整理再被改写**；它们是 `_5` 原 Arch35 已有能力。[QGMM 特化](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_gmm_common.h:409)、[拼接视图](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/blaze/gemm/tile/copy_gmm1_concat.h:36)。

### 4.3 Host 与 Arch35 专属文件

原 Arch35 的 Host tiling 现在位于 `arch35_92`：增加按 DAV 的 1C2V/1C1V 校验、Arch92 A8W8/MTE 能力门禁、GM handoff 相关 flag 的 UB 固定预算，以及按策略提交 Workspace；既有自适应 Dispatch/SendMask/Unpermute tiling 仍共用。这是**1 个被侵入的 Host 算法文件**，新增 Host 硬件策略文件单独计数。Arch35 走 `DEFAULT` Workspace 和 1C2V 校验。检视入口：[Host Policy](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_host_arch_policy.h:20)、[预算](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:1333)、[平台校验](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2574)、[Workspace 提交](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2626)。

Arch35 专属的 3 个 Wave、Layered 头和入口在首次移动公共目录时仅调整了 include 路径；当前未把 Arch92 控制流塞进这些文件。构建按 SoC 选择入口：`ascend950` 保持 Arch35 的 auto-sync 选项，`ascend910_96` 使用 Arch92 入口与 1C1V。见 [Arch35 A8W8 Wave](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:17)、[构建分支](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/CMakeLists.txt:26)。

## 5. 检视和验证边界

优先检视四处：`Host/Kernel WorkspaceLayoutPolicy` 是否同值；普通 GMM1 的 GM 顺序 flag 与共享专家 UB 同步是否分开；QGMM Gate/Up 视图与输出 N 口径；Combine consumer 的 1C1V 槽位和 `startBlockIdx` 推进。相关代码：[Host Workspace](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2630)、[Kernel Workspace](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:305)、[GMM1 同步](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:139)、[Combine 角色](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm2_combine.h:346)。

本次仅完成代码/历史对照与 `git diff --check` 静态检查；**没有**在 A6 CANN 环境编译、跑精度或性能，也没有对 Arch35 做回归。特别是从 `_910b2` QBMM 改为 `_5` QGMM 后，不能引用旧仓的运行结果作为新仓验收证据。
