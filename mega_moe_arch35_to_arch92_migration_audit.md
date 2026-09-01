# MegaMoe 从 arch35 到 arch92 的迁移审计

更新日期：2026-09-14

适用范围：MegaMoe 在 A6 `ascend910_96` 上的 arch92 A8W8、MXFP、MTE、1C1V 路径。本文以当前 arch35 实现为对照基线，只记录 arch92 为适配 A6 而新增、替换或改变的内容。

## 1. 文档范围

本文把 arch35 到 arch92 的迁移视为一次完整修改，说明 Host tiling、Kernel、Workspace、构建和 Torch extension 如何共同形成 A6 执行链。

以下内容不作为迁移修改展开：

- 两个架构中算法和输入输出契约都没有变化的通用流程。
- 仅有排版、命名或格式差异的代码。
- 开发过程中的临时验证逻辑。
- 共享专家等已经按 arch35 既有语义正常接通、没有形成 arch92 特有协议的功能。

当前正式入口只放行 A8W8。A8W4 和 A4W4 是后续计划支持的场景，相关模板、类型和辅助实现被有意保留；它们不是待删除代码，但在入口、Host 能力门禁、tiling key、1C1V 协议和测试全部接通前，不能视为已经支持。

## 2. 总体修改结论

arch92 相对 arch35 的实质变化集中在六个方面：

1. Kernel task type 从 1C2V 改为 1C1V，同一物理 block 的 AIC 与 AIV 使用相同编号。可对照 [arch35 1C2V 入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp:72) 和 [arch92 1C1V 入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)。
2. arch92 独立维护 Host tiling，增加 A6 当前能力门禁，并用相等的 AIC/AIV 数建立启动契约。对应 [arch92 Host 主入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2010)。
3. arch92 复用 arch35 A8W4 已有的 routed expert Wave，只保留原 AIV1 承担的 Dispatch、Activation 和 Combine 路径，删除原 AIV0 的 W4 到 W8 Prologue 角色；不是把两个 AIV 的工作合并到一个 AIV。对应 [arch35 AIV1 Wave 分工](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w4.h:221) 和 [arch92 A8W8 Wave](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:243)。
4. `ArchitecturePolicy` 把原来由 `ENABLE_A8W4` 和 tiling mode 隐式选择的 cumsum、GM handoff 和 Workspace 条件显式化，使 arch92 的 A8W8 实例能够选择与 arch35 A8W4 相同的交接行为。对应 [ArchitecturePolicy](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18)。
5. arch92 A8W8 使用 arch35 已有的通用 QBMM MX BlockMmad，不迁移 A8W4 的独立 Prologue；GMM1/GMM2 进一步复用一个 `mmadContext_`。对应 [GMM Config](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47) 和 [context 成员](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:123)。
6. OpDef、Kernel 根入口、Host 编译依赖、发布规则和 Torch backend 需要共同补齐 A6 路由。对应 [A6 OpDef](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/mega_moe_def.cpp:195) 和 [Torch backend](/D:/gitcode/ops-transformer_910b2/torch_extension/cann_ops_transformer/ops/mega_moe.py:265)。

当前能力边界如下：

| 维度 | arch35 | arch92 当前入口 |
| --- | --- | --- |
| C/V 配比 | 1C2V | 1C1V |
| 已放行主路径 | A8W8、A8W4、A4W4 等 | A8W8 |
| Dispatch 量化 | 多组合 | MXFP |
| A8W8 类型 | E5M2、E4M3FN | E5M2、E4M3FN，activation 与 weight 类型匹配 |
| 权重布局 | 多种布局 | General/DN 与 A8W8_NZ |
| Combine | 多组合 | NoQuant、E5M2、E4M3FN |
| 通信 | MTE、URMA/Layered 等 | MTE |


## 3. 文件归属变化

### 3.1 arch92 独立维护的文件

只有直接依赖 1C1V、A6 能力、UB/Workspace 或 BlockMmad 接口的文件才独立维护：

| 类别 | arch92 文件 | 相对 arch35 的职责变化 |
| --- | --- | --- |
| Host tiling | [`op_host/op_tiling/arch92/mega_moe_tiling.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2010) | 1C1V 核数、A6 能力门禁、arch92 Workspace policy |
| Kernel 入口 | [`op_kernel/mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44) | A6 根入口、1C1V task type、A8W8 ND/NZ 实例 |
| Kernel 基类 | [`op_kernel/arch92/mega_moe.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:55) | 架构策略、单 AIV UB 视图和 Workspace 初始化 |
| A8W8 Wave | [`mega_moe_a8w8_wave.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:48) | 复用 A8W4 Wave 调度，移除 AIV1 sub-block 守卫并接入 1C1V job |
| Tiling key | [`mega_moe_tiling_key.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_tiling_key.h:27) | 只枚举 A6 当前可达组合 |
| 架构策略 | [`mega_moe_core_context.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18) | 物理核编号和跨阶段 policy |
| GMM 公共层 | [`mega_moe_gmm_common.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47) | QBMM MX、layout、L1 参数和 context |
| Workspace | [`mega_moe_workspace.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_workspace.h:146) | 显式选择 A8W4 式 GM handoff 条件区，并按 1C1V 计算 Combine stride |
| Dispatch stage | [`mega_moe_token_dispatch.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:638) | 去除 AIV1 sub-block 前提，并接收 arch92 cumsum policy |
| GMM1 stage | [`mega_moe_gmm1_activation.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:632) | 增加 `UseGmHandoff` 选择，使 A8W8 复用 A8W4 已有 GM 交接路径 |
| GMM2 stage | [`mega_moe_gmm2_combine.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:759) | 显式接收 1C1V logical Combine job 数；通知协议不变 |
| Reset stage | [`mega_moe_workspace_reset.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_workspace_reset.h:73) | 接收 arch92 Workspace policy；分块清零算法不变 |

arch92 Host tiling 只包含 Host 声明和 arch92 契约头，不再通过 Kernel 主头文件间接取得定义：

~~~cpp
#include "../mega_moe_tiling_host.h"
#include "../../../op_kernel/arch92/mega_moe_tiling.h"
#include "../../../op_kernel/arch92/mega_moe_tiling_key.h"
#include "../../../op_kernel/arch92/common/mega_moe_workspace.h"
~~~

这项边界调整可从 [arch92 tiling include 区](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:24) 检视。

### 3.2 继续复用 arch35 的文件

Quant、SendMask、共享输入准备和 Unpermute 的算法没有因 1C1V 改变。arch92 只改变它们的调用时序、job 配置和 UB 视图，因此继续复用：

- [`QuantizeLocalTokens`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_quant.h:76)
- [`GatherAndSendExpertMasks`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_send_mask.h:117)
- [`PrepareSharedExpertInput`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_shared_expert_input.h:34)
- [`UnpermuteTokens`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_unpermute.h:240)

复用关系在 [arch92 基类 include 区](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:35) 中直接体现。它们不是 arch92 新增功能，因此后文不再展开其内部算法。

### 3.3 独立与复用的判断标准

文件是否独立维护取决于是否携带架构协议，而不是目录名称：

1. 读取 `GetSubBlockIdx()` 或区分 AIV0/AIV1 职责的代码不能直接复用。
2. 假设 `blockAivNum == 2 * aicNum` 的 job、slot 或 flag 公式必须在 arch92 重写。
3. 自行推导 arch35 Workspace 地址、flag 数量或 UB 固定基址的代码必须独立维护。
4. tensor layout、dtype、对齐和 BlockMmad 接口变化时，适配层必须独立维护。
5. 只消费调用方传入 job/config、且没有跨阶段架构状态的纯算法可以复用。

arch35 的核编号和 sub-block 分工可分别从 [核编号成员](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_arch35.h:133) 与 [sub-block 分工示例](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_arch35.h:265) 检视；arch92 的替代边界集中在 [`MegaMoeArch92ArchitecturePolicy`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18)。

### 3.4 Tiling schema 的共享边界

`MegaMoeTilingData` 仍复用 arch35 定义，以保持 Host/Kernel ABI 一致；arch92 独立的是字段计算、能力检查和 key 生成，而不是再复制一份 schema。

共享定义见 [`MegaMoeTilingData`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_tiling.h:161)，arch92 的包含入口见 [`arch92/mega_moe_tiling.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_tiling.h:14)。

这意味着 schema 仍是跨架构契约。新增、删除或改变字段语义时，必须同时检查 arch35 Host、arch92 Host 和两边 Kernel；仅在 arch92 Host 或 Kernel 单侧增加字段是不完整的修改。

## 4. 1C2V 到 1C1V

### 4.1 task type 与核编号

arch35 使用：

~~~cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
~~~

arch92 改为：

~~~cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
~~~

对应入口分别是 [arch35 task type](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp:72) 和 [arch92 task type](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)。

arch35 的逻辑 AIC job 通常由 `GetBlockIdx() / GetTaskRation()` 得到，两个 AIV 再依靠物理 AIV 编号或 sub-block 分工。arch92 中 AIC 与 AIV 都直接使用当前物理 block：

~~~cpp
__aicore__ static inline uint32_t GetPhysicalBlockIdx()
{
    return static_cast<uint32_t>(GetBlockIdx());
}
~~~

该实现位于 [`GetPhysicalBlockIdx`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18)。因此同一个 `j` 同时标识 `Cj` 与 `Vj`，arch92 stage 不应再派生 `2 * blockIdx + subBlockIdx`。

### 4.2 Host 启动契约

arch92 Host 新增 AIC/AIV 数量相等的约束，并按 1C1V 设置 block dim：

~~~cpp
OP_TILING_CHECK(aivNum != aicNum, ..., return ge::GRAPH_FAILED);
tilingData->aicNum = aicNum;
tilingData->blockAivNum = aivNum;
context->SetBlockDim(
    ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
context->SetScheduleMode(1);
~~~

这段变化位于 [`MegaMoeTilingFuncImplPublic`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2010)。`blockAivNum` 在 arch92 表示真实逻辑 job 数，后续 rank 分摊、Combine slot 和 reset 分块都不能沿用 arch35 的“一半 AIV”公式。

### 4.3 AIV 角色裁剪

迁移基线是 arch35 A8W4 Wave。在该路径中，AIV1 已经负责 routed expert 的 Dispatch、GMM1 后处理、GMM2 后处理和 Combine，AIV0 承担 W4 到 W8 的 Prologue。arch92 当前支持 A8W8，权重不需要这段 W4 Prologue，因此 1C1V 迁移只保留原 AIV1 的职责并映射到唯一 AIV，原 AIV0 的职责直接删除。

arch35 对 routed Wave 的 AIV1 限制见 [`GetSubBlockIdx() == 1U`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w4.h:221)；arch92 对应循环不再判断 sub-block，见 [`ProcessMoeExpertStages`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:243)。Wave 切片、下一 Wave Dispatch 与当前 GMM1 交错、当前 Wave GMM2/Combine 的算法均沿用 arch35 A8W4，不属于 arch92 新增调度。

## 5. Host Tiling 的 arch92 差异

### 5.1 独立入口与 Workspace policy

arch92 Host 不直接复用 arch35 的注册实现，因为它需要生成 1C1V 专用启动参数和 A6 能力组合。主入口仍沿用通用的输入、属性和 Workspace 建契约顺序，但这条通用调用链本身不是 arch92 新增内容。

arch92 特有入口见 [`MegaMoeTilingFuncImplPublic`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2010)，Host 固定选择的布局是 [`WorkspaceLayoutPolicy::GM_HANDOFF_EXPERT_WAVE`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:41)。

### 5.2 A6 当前能力门禁

arch92 新增 `CheckArchitectureCapability`，当前只允许：

- MTE topology。
- `GROUPED_MATMUL_MODE_GENERAL` 或 `GROUPED_MATMUL_MODE_A8W8_NZ`。
- MXFP Dispatch。
- E5M2 activation 对 E5M2 weight，或 E4M3FN activation 对 E4M3FN weight。
- NoQuant、E5M2、E4M3FN Combine。

门禁实现见 [`CheckArchitectureCapability`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:1970)，可生成组合见 [arch92 tiling key](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_tiling_key.h:27)。两者必须同步，否则会出现 Host 接受但没有 Kernel 实例，或存在实例却永远不可达。

### 5.3 A6 核数与 UB 容量输入

这里真正需要改变的是 `blockAivNum` 的语义，不是 UB 容量的获取方式。

arch35 的 Kernel task type 是 1C2V。在 A5 平台上，Host 读取到的物理核数通常满足 `aivNum = 2 * aicNum`，并将 `blockAivNum=aivNum` 下发。因此：

- 使用全部 AIV 的 Dispatch、SendMask、Unpermute 等分工，可以直接把 `blockAivNum` 当作 job 数。
- 只允许每对 AIV 中 AIV1 参与的 A8W4 Combine，需要把物理 AIV 数折算为 `blockAivNum / 2` 个逻辑 consumer。

arch92 是 1C1V，Host 明确校验 `aivNum == aicNum`，此时 `blockAivNum` 已经是逻辑 job/consumer 数。SendMask、Unpermute 和 reset 仍可直接按它分块，但 A8W4 的“成对 AIV 再除以 2”规则必须删除；否则 28 个有效 consumer 会被错误计算成 14 个。arch35 的核数获取见 [arch35 Host 平台信息](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp:2001)，arch92 的 1C1V 校验见 [arch92 Host 平台信息](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2023)。Combine 中由此产生的具体差异见 5.4。

“固定容量假设”不适用于当前代码。arch35 和 arch92 都通过 `GetCoreMemSize(CoreMemType::UB, ubSize)` 获取目标平台的真实 UB 容量，再把 `ubSize` 传给 `SetAdaptiveBufferConfigs`。A6 返回约 384 KiB 后，同一类公式可以据此增加 ring slot 或 batch 大小；这里没有从 arch35 复制一个写死的 UB 容量。两边对应调用分别见 [arch35 UB 容量传递](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp:2016) 和 [arch92 UB 容量读取](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2044)。

arch92 的自适应配置入口见 [`SetAdaptiveBufferConfigs`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:975)，使用真实容量的计算包括 [`CalcDispatchBufferConfig`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:704)、[`CalcSendMaskBufferConfig`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:782) 和 [`CalcUnpermuteBufferConfig`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:843)。

### 5.4 Combine 的 C/V GM 软同步协议

这里先明确责任：当时卡死的直接问题是实际生效的 Host Tiling 仍使用 1C2V 的折半规则；arch92 Kernel 的 C 侧 producer 和 V 侧 consumer 已经都按 1C1V 的完整逻辑核数运行。协议不一致发生在“Kernel C/V 使用的 slot 范围”与“Host 为每个 expert 划定的 GM counter stride”之间，不是 C 侧仍然乘 2。

这套 GM 软同步由三部分组成：

1. C 侧根据逻辑 Combine job 数生成 token group 到 counter slot 的映射。每完成一个 GMM2 N tile，就对该 group 对应的全部 consumer slot 执行一次 `AtomicAdd(1)`。arch92 Wave 显式传入 `gmm2Config_.blockJob.totalJobs`，C 侧通过 `logicalCombineJobCount` 使用该值，见 [arch92 GMM2 调用](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:231) 和 [C 侧 slot 布局](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:550)。
2. V 侧以 `jobIndex=blockIdx_`、`totalJobs=blockNum_`、`participates=true` 计算自己负责的 token group 和等待的 slot，见 [arch92 `QuantCombineConfig`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:282) 和 [V 侧等待](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:1306)。
3. Host 下发 `combineSyncSlotCountPerExpert`。该值不是 C 或 V 的等待目标，而是相邻 expert GM counter 区域之间的 stride，expert 基地址按 `expertIdx * combineSyncSlotCountPerExpert` 计算。arch92 的正确公式位于 [`CalcCombineSyncSlotCountPerExpert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:922)。

1C2V 和 1C1V 的差别在于 `blockAivNum` 所代表的物理 AIV 集合不同：

~~~text
A5 1C2V:
aicNum = 28
blockAivNum = 56
只有每对中的 AIV1 参加该 Combine 路径
逻辑 consumer 数 = 56 / 2 = 28

A6 1C1V:
aicNum = 28
blockAivNum = 28
28 个 AIV 全部参加 Combine
逻辑 consumer 数 = 28，不能再次除以 2
~~~

因此 A5 的 `/2` 不是通用 Combine 公式，而是把“56 个物理 AIV”折算成“28 个实际参加的 AIV1”。A6 的 `blockAivNum=28` 已经等于实际参加的逻辑 consumer 数；继续 `/2` 会错误地得到 14。

当时已经确认的实际错误链如下：

1. A6 Kernel 配置为 `GetBlockNum=28`、`aicNum=28`、`blockAivNum=28`。
2. arch92 C 侧按 28 个逻辑 consumer 为当前 expert 生成相对 slot `0..27`；V 侧也按 `totalJobs=28` 选择自己的 slot。
3. 实际生效的 Host Tiling 错误执行 A5 AIV1-only 的 `/2`，下发 `combineSyncSlotCountPerExpert=14`。
4. 因此 expert 0 的 counter 基址为 `B + 0 * 14`，expert 1 的基址为 `B + 1 * 14`。expert 0 使用相对 slot `14..27` 时，已经落入 expert 1 的相对 slot `0..13`。
5. 两个 expert 的 C 侧通知随后累加到同一批 GM counter。对于 `N=4096`，每个 expert 正常只应发布 `4096 / 256 = 16` 次 N-tile 完成通知；地址重叠后，同一 counter 会继续增长到 26、30、32 等超调值。
6. V 侧使用严格相等条件等待 `counter == 16`。如果第一次读到时已经是 26，后续增长到 32，counter 不会回退，V 就永久卡在轮询中。

Reset 记录已经证明这些 slot 初始为 0；GMM2 记录也证明单个 expert 只发布了 16 个唯一 N tile，不存在重复 GMM2 调度。超调从 slot 14 的边界开始，与错误的 per-expert stride 完全一致。因此根因是：**1C2V 改为 1C1V 后，Host 仍按 AIV1-only 规则把已经是逻辑 consumer 数的 `blockAivNum` 再除以 2，导致 Host GM 布局与 arch92 Kernel C/V 的 slot 范围不一致。**

### 5.5 Host/Kernel Workspace 一致性

Host 与 Kernel 都显式选择 `GM_HANDOFF_EXPERT_WAVE`。Host 用该 policy 计算大小，Kernel 用同一 policy 解释地址：

~~~cpp
WorkspaceInfo workspaceInfo(
    (uint8_t *)0, tilingData, 1U, HOST_WORKSPACE_LAYOUT);
~~~

Host 的最终设置见 [`SetWorkspace`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:1087)，Kernel 的构造见 [arch92 `WorkspaceInfo` 使用点](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:55)。只改一侧会使数据区、flag 区和 counter 区整体错位。

### 5.6 与当前 arch35 的能力差距

当前 arch92 Host 仍有两处不是 A6 硬件要求、而是尚未同步 arch35 新能力的差异：

- arch92 对 `hiddenDim` 仍采用 1024 粒度和较窄范围；arch35 的 MTE 路径已支持更细粒度。
- arch92 的部分通信窗口和 buffer 公式仍直接使用 `bs`，没有完整继承 arch35 的 `numMaxTokensPerRank` 上界语义。

相关检查位于 [`CheckAttrParams`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:435)，相关 buffer 公式位于 [`CalcDispatchBufferConfig`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:704)。这两项应作为后续能力同步处理，不能简单复制单个条件而不重算通信窗口、Workspace 和 UB。

## 6. Kernel 入口与模板系统

### 6.1 A6 根入口

A6 根入口只在当前 Host 可生成的 A8W8 条件下实例化 `MegaMoeArch92A8W8Wave`，并用最后一个模板参数区分 DN 与 NZ：

~~~cpp
MegaMoeArch92A8W8Wave<
    DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS,
    DTYPE_WEIGHT1, DispatchQuantOutType,
    CombineQuantOutType, TopkWeightsPrefetch,
    IsWeightNz> op;
~~~

实例化分支位于 [`op_kernel/mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)。这与 arch35 自己的根入口并存，构建时必须按 SoC 二选一。

### 6.2 ArchitecturePolicy 模板参数

arch92 增加 `ArchitecturePolicy` 模板参数，是为了拆开 arch35 中“计算类型”和“执行协议”之间的隐式依赖，而不是增加三套新算法。

arch35 把以下行为绑定在同一组条件上：

| arch35 隐式条件 | 随条件一起选择的行为 |
| --- | --- |
| `ENABLE_A8W4=true` | Dispatch cumsum 写入 GM；GMM1/GMM2 进入 A8W4 专用函数；AIC 写 GM 后通过 sequence 通知 AIV1 |
| `groupedMatmulMode=A8W4` | Workspace 分配 cumsum、GMM1/GMM2 结果区和 `gmmToEpilogueFlag` |

这种绑定在 arch35 成立，是因为 FP4 weight、A8W4 BlockMmad、AIV0 Prologue、AIV1 epilogue 和 GM handoff 总是成套出现。因此代码可以从 dtype 推导 `ENABLE_A8W4`，再从 grouped-matmul mode 推导 Workspace，不需要额外描述执行协议。对应关系可从 [arch35 `ENABLE_A8W4`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_arch35.h:143)、[A8W4 的 AIV0/AIV1 分支](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:838) 和 [Workspace 条件分配](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/common/mega_moe_workspace.h:204) 检视。

arch92 中这条隐式推导不再成立：

1. 当前计算类型是 A8W8，weight 为 FP8，所以 `ENABLE_A8W4=false`，grouped-matmul mode 也是 General 或 A8W8_NZ。
2. 执行调度却以 arch35 A8W4 Wave 的 AIV1 路径为基线：AIV0 的 W4 到 W8 Prologue 被删除，但当前实现仍保留 Dispatch cumsum 持久化和 AIC/AIV 的 GM handoff。
3. 因而“不是 A8W4”已经不能推出“不需要 cumsum GM 备份、GM handoff 和对应 Workspace”。若继续使用旧条件，这些分支和地址不会生成；若直接伪造 `ENABLE_A8W4=true`，又会错误实例化 FP4 Prologue、A8W4 BlockMmad 和 1C2V sub-block 分工。

所以 arch92 将计算类型与执行协议分开，由编译期 policy 独立表达：

~~~cpp
static constexpr bool PERSIST_DISPATCH_CUMSUM = true;
static constexpr bool USE_GM_HANDOFF = true;
static constexpr WorkspaceLayoutPolicy WORKSPACE_LAYOUT =
    WorkspaceLayoutPolicy::GM_HANDOFF_EXPERT_WAVE;
~~~

`PERSIST_DISPATCH_CUMSUM` 将 cumsum 生命周期从 `ENABLE_A8W4` 中解耦。当前设为 `true`，是因为 arch92 现有 UB 布局没有为 cumsum 保留跨 Wave 常驻区；它在行为上仍与 arch35 A8W4 相同。policy 的使用点见 [Dispatch Wave 调用](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:153)，arch35 的原条件见 [cumsum Workspace 分配](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/common/mega_moe_workspace.h:241)。

`USE_GM_HANDOFF` 将输出交接方式从 A8W4 专用函数中解耦。arch35 没有这个变量，因为 `RunGmm1A8W4` 和 `RunGmm2A8W4` 内部已经固定使用 GM handoff；arch92 新增 `UseGmHandoff` 模板参数，使通用 A8W8 GMM1/GMM2 也能选择同一条路径。参数在 [GMM1 调用](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:195) 和 [GMM2 调用](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:230) 传入；GM 写结果、发布 sequence 和 AIV 等待的协议本身没有改变。

`WORKSPACE_LAYOUT` 将 Workspace 条件区从 grouped-matmul mode 中解耦。arch92 A8W8 的 mode 不会进入 arch35 A8W4 分配条件，但 Kernel 又会访问 handoff 和 cumsum 区域，因此 Host 与 Kernel 必须显式选择 `GM_HANDOFF_EXPERT_WAVE`。新增的是布局选择接口，不是 cumsum、GMM 结果区或 sequence 本身。可对照 [arch92 policy 条件](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_workspace.h:188) 和 [Host policy](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:43)。

这些开关采用编译期模板 policy，是因为它们会改变 `if constexpr` 分支、GMM 输出位置和 Workspace 地址契约，而当前 A6 Kernel 实例的协议在编译时已经确定。这样既不会引入运行时判断，也不会为了复用 handoff 而实例化错误的 A8W4 数据类型和 Prologue。

### 6.3 IsWeightNz 模板参数

DNExt/ZN 两种 B layout 及其 BlockMmad 配置在 arch35 中已经存在。arch92 的改动只是把选择上提到 `MegaMoeArch92A8W8Wave` 的模板参数，并在根入口按 tiling mode 实例化 DN 或 NZ：

~~~cpp
using LayoutB = Std::conditional_t<
    IsWeightNz,
    Te::ZNLayoutPtn,
    Te::DNExtLayoutPtn>;
~~~

arch35 的既有 layout 选择见 [`GmmKernel::Config`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/common/mega_moe_gmm_common.h:60)；arch92 的外层模板和实例化分支分别见 [`MegaMoeArch92A8W8Wave`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:43) 与 [A6 根入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:59)。这样一个 Kernel 实例只有一种 B layout，也为单一 `mmadContext_` 提供了类型前提。

## 7. BlockMmad 与 context

### 7.1 移除 A8W4 Prologue

arch35 A8W4 使用 `MatmulMxFp8Fp4DynamicKL1TailResplit`，需要 AIV0 完成 W4 到 W8 Prologue。arch92 当前是 A8W8，直接使用 arch35 通用 A8W8 已有的 `MatmulWithScaleMx`：

~~~cpp
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
~~~

因此这里的差异是删除 A8W4 专用 Prologue 及其 AIV 角色，不是 arch92 新发明了一套 BlockMmad。arch35 两种 policy 的选择见 [arch35 `DispatchPolicy`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/common/mega_moe_gmm_common.h:86)，arch92 当前选择见 [arch92 `DispatchPolicy`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:71)。

### 7.2 单一持久化 context

arch92 Wave 在 AIC 上只保留一个 `mmadContext_`，由 GMM1 和 GMM2 顺序复用：

~~~text
GMM1 -> InitBlockMmad(mmadContext_, gmm1 config) -> compute
GMM2 -> InitBlockMmad(mmadContext_, gmm2 config) -> compute
~~~

该复用是 arch92 的结构调整。成立条件是 `IsWeightNz` 已在外层固定，且 `static_assert` 验证 GMM1/GMM2 得到相同的 BlockMmad 类型；实际 M/N/K 和 L1 参数仍在每次调用前更新。类型约束和 AIC-only 存储见 [`MmadContextStorage`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:78)，唯一成员见 [`mmadContext_`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:123)。`static_assert` 只在编译期生效，不增加运行时开销。

## 8. arch35 A8W4 Wave 的 1C1V 适配

arch92 的 expert 位置推进、热点 expert 跨 Wave、下一 Wave Dispatch 与当前 GMM1 交错，以及当前 Wave 完成后执行 GMM2/Combine，均来自 arch35 A8W4 Wave，不作为新增能力重复说明。

实际修改只有核角色映射：arch35 通过 `GetSubBlockIdx() == 1U` 让 AIV1 执行 routed Wave；arch92 删除 sub-block 判断，让 1C1V 中唯一的 AIV 执行原 AIV1 路径。两边对应总循环见 [arch35 `ProcessMoeExpertStages`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w4.h:201) 和 [arch92 `ProcessMoeExpertStages`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:243)。

## 9. Workspace 与 UB 的实际差异

### 9.1 显式 Workspace policy

arch92 `WorkspaceInfo` 中的 Dispatch/activation 数据区、GM handoff 结果区、sequence、Combine counter、cumsum 备份和连续 flag 清零机制在 arch35 都已经存在，不能列为 arch92 新增区域。

实际变化是 `WorkspaceInfo` 增加 `WorkspaceLayoutPolicy` 参数。arch35 根据 `GROUPED_MATMUL_MODE_A8W4` 等 tiling mode 隐式进入 handoff 条件分支；arch92 当前 mode 是 A8W8，因此通过 `GM_HANDOFF_EXPERT_WAVE` 显式让 `flagGmmToEpiloguePtr`、`cumsumInfoPtr`、`gmm1MmadResPtr` 和 `gmm2MmadResPtr` 进入与 arch35 A8W4 等价的分配路径。可对照 [arch35 条件区](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/common/mega_moe_workspace.h:204) 和 [arch92 policy 条件区](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_workspace.h:188)。

另一个真实差异是量化 Combine counter 的 per-expert stride 按全部 1C1V consumer 计算，不能使用 A5 AIV1-only 的折半结果；该问题已在 5.4 展开。Host 与 Kernel 必须使用同一 policy，否则条件区之后的地址会错位。Host 选择见 [`HOST_WORKSPACE_LAYOUT`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:43)，Kernel 选择见 [arch92 `WorkspaceInfo` 构造](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:338)。

`ResetSyncStatus` 的分块清零算法与 arch35 相同，只是清零元素总数和参与 job 数跟随上述 layout 与 1C1V 配置变化，因此不作为独立迁移点。

### 9.2 保留 AIV1 UB，删除 Prologue AIV

arch92 没有把 arch35 两个 AIV 的 UB 或职责合并到一个 AIV。arch35 A8W4 的 routed Wave 原本就由 AIV1 执行；arch92 保留这条 AIV1 路径使用的 Dispatch、Activation、Combine 和 Unpermute UB 视图，并去掉 sub-block 判断。AIV0 用于 W4 到 W8 Prologue 的职责及其局部资源没有迁入 arch92 A8W8。

arch35 的 AIV1 入口见 [A8W4 Wave 的 sub-block 判断](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w4.h:223)，arch92 保留路径的 UB 视图见 [`DispatchBuffInit`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:374)、[`InitCombineBuffers`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:450) 和 [`InitTokenUnpermuteBuffers`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:551)。因此 UB 审计关注的是保留下来的 AIV1 视图在 A6 384 KiB UB 中是否正确排布，不是把 A5 两份 AIV UB 容量相加。

### 9.3 Dispatch cumsum 的现状与后续方向

当前 arch92 与 arch35 A8W4 一样，都会把 Dispatch cumsum 写入 Workspace GM，并在后续需要时恢复。两者只有条件表达不同：arch35 由 `ENABLE_A8W4` 控制，arch92 由 `PERSIST_DISPATCH_CUMSUM=true` 控制，因此这不是当前迁移形成的行为差异。

arch35 的调用见 [`RunMoeExpertDispatchStage<..., ENABLE_A8W4>`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_arch35.h:696)，arch92 的调用见 [`ArchitecturePolicy::PERSIST_DISPATCH_CUMSUM`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:153)。

A6 有 384 KiB UB，后续可以评估为 cumsum 保留常驻 UB 区，从而取消这部分 GM 往返。只有实施该优化后，这里才会形成 arch92 相对 arch35 的实际行为差异；修改前需要核算保留 AIV 路径各阶段的同时存活区和最高 UB 地址。

## 10. A8W4/A4W4 预留

arch92 保留了 A8W4/A4W4 的模板分支、FP4 traits、prologue 类型、条件 Workspace 和 stage 接口。这是为后续支持保留的代码，不是当前正式能力，也不应作为无用代码删除。

保留的主要 stage 入口包括 [`RunGmm1A8W4`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:906) 和 [`RunGmm2A8W4`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:951)。

当前不可达由四层共同保证：

1. `GmmKernel::Config` 的编译期约束，见 [A8W8 `static_assert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47)。
2. Host 能力门禁，见 [`CheckArchitectureCapability`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:1970)。
3. arch92 tiling key 未枚举 E2M1 等组合，见 [`mega_moe_tiling_key.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_tiling_key.h:27)。
4. A6 根入口只实例化 A8W8，见 [`mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)。

正式放开时必须成套修改入口、Host、tiling key、BlockMmad/Prologue、Workspace、同步和测试，不能只去掉 `static_assert`。

## 11. 构建、注册与封装

### 11.1 OpDef 注册

`MegaMoe` 的 OpDef 增加 `ascend910_96`，使框架允许在 A6 上选择该算子：

~~~cpp
this->AICore().AddConfig(
    "ascend910_96", aicore_config);
~~~

注册位置见 [`mega_moe_def.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/mega_moe_def.cpp:195)。这只完成 SoC 声明，不能替代 Host tiling 和 Kernel 源选择。

### 11.2 Kernel 与 Host 源选择

A6 Kernel 应选择根 [`op_kernel/mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)，arch35 仍选择自己的 [`arch35/mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp:72)。

Host 同时存在 arch35 和 arch92 的 tiling 注册。构建系统必须确保一个 SoC 目标只链接一套 `IMPL_OP_OPTILING(MegaMoe)`；两套注册位置分别见 [arch35 Host 注册](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp:2064) 和 [arch92 Host 注册](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2094)。

通用递归收集和按架构过滤分别位于 [`obj_func.cmake` 通用收集](/D:/gitcode/ops-transformer_910b2/cmake/obj_func.cmake:108) 与 [按架构收集](/D:/gitcode/ops-transformer_910b2/cmake/obj_func.cmake:291)。

### 11.3 Host 编译依赖

arch92 Host 与 Kernel 增加独立头文件后，`op_host/CMakeLists.txt` 必须为 A6 compute unit 提供 Tensor API、Blaze、通信和生成头文件的 include 闭包。相关配置见 [MegaMoe Host CMake](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/CMakeLists.txt:25)。



### 11.4 Torch backend

Torch extension 为 A6 平台映射 `channel` backend。修改位置见 [`CommContextManager` backend 映射](/D:/gitcode/ops-transformer_910b2/torch_extension/cann_ops_transformer/ops/mega_moe.py:265)。

需要继续核对运行时实际平台字符串是 `Ascend910_9699`、`Ascend910_96` 还是其他别名；映射名不一致会使上层未进入 A6 通信后端。

### 11.5 发布分类

当前发布规则仍需确认 arch92 Host、Kernel 目录和 A6 根入口被打包。检查入口见 [`classify_rule.yaml` 的 MegaMoe 条目](/D:/gitcode/ops-transformer_910b2/classify_rule.yaml:740)。

这一步与编译成功相互独立：本地构建能找到源码，不代表安装包一定包含新增目录。

## 12. 修改点总表与未闭环事项

### 12.1 修改点总表

| 修改面 | 关键函数/代码段 | arch92 相对 arch35 的变化 |
| --- | --- | --- |
| C/V 配比 | [`KERNEL_TYPE_MIX_AIC_1_1`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44) | 1C2V 改为同号 1C1V |
| 物理编号 | [`GetPhysicalBlockIdx`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18) | AIC/AIV 均直接使用 `GetBlockIdx()` |
| Host 核数 | [`MegaMoeTilingFuncImplPublic`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2010) | 校验 `aivNum == aicNum` |
| Host 能力 | [`CheckArchitectureCapability`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:1970) | 当前只放行 A8W8、MXFP、MTE |
| Wave 角色 | [`ProcessMoeExpertStages`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:243) | 保留 arch35 A8W4 的 AIV1 Wave，删除 AIV0 Prologue 角色 |
| GM handoff 选择 | [`USE_GM_HANDOFF`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:20) | 通过模板参数让 A8W8 选择 arch35 A8W4 已有的 GM 协议 |
| 布局选择 | [A6 根入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:59) | 将 DN/NZ 选择上提为 Wave 编译期实例 |
| BlockMmad | [arch92 `DispatchPolicy`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:71) | 删除 A8W4 Prologue policy，复用 arch35 通用 A8W8 QBMM MX |
| Context | [`mmadContext_`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:123) | GMM1/GMM2 复用一个 AIC context；DN/NZ 编译期互斥 |
| Combine slot | [`CalcCombineSyncSlotCountPerExpert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:922) | consumer 数按全部 1C1V job 计算 |
| Workspace | [arch92 `WorkspaceInfo`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_workspace.h:146) | 新增显式 policy 选择既有条件区，并按 1C1V 设置 Combine stride |
| UB | [`DispatchBuffInit`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:374) | 保留原 AIV1 的 UB 视图，不合并已删除的 Prologue AIV 资源 |
| 注册 | [`AddConfig("ascend910_96")`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/mega_moe_def.cpp:195) | OpDef 接受 A6 |
| 封装 | [`CommContextManager`](/D:/gitcode/ops-transformer_910b2/torch_extension/cann_ops_transformer/ops/mega_moe.py:265) | A6 选择 channel backend |

### 12.2 未闭环的代码整改

1. **构建映射**：确认 `ascend910_96` 只选择 [arch92 Host tiling](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2094) 和 [A6 Kernel 根入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)，不会同时链接 arch35 注册。
2. **发布分类**：更新 [`classify_rule.yaml`](/D:/gitcode/ops-transformer_910b2/classify_rule.yaml:740)，把 arch92 Host、Kernel 和根入口纳入交付清单。
3. **平台名称**：核对 [Torch backend 映射](/D:/gitcode/ops-transformer_910b2/torch_extension/cann_ops_transformer/ops/mega_moe.py:265) 是否覆盖运行时所有 A6 名称。
4. **Host 能力同步**：评估 [`CheckAttrParams`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:435) 中 `hiddenDim` 和 `numMaxTokensPerRank` 是否应同步当前 arch35 语义，并成套重算 buffer/Workspace。
5. **Context 复用回归**：覆盖多 expert、多 Wave、DN/NZ、E4M3FN/E5M2、prefetch on/off 和三种 Combine mode。
6. **384 KiB cumsum 常驻**：评估取消 [Dispatch cumsum GM 恢复](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:495) 的收益；实施前完成全 UB 生命周期和最高地址核算。
7. **routed handoff 优化**：评估 [`USE_GM_HANDOFF`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18) 的 GM、UB 单槽、双 bank 或混合策略，补齐 credit/ACK 后再改变介质。
8. **A8W4/A4W4**：保留现有 stage 骨架，按入口、Host、BlockMmad、Workspace、同步和测试的完整链条逐步放开。
9. **A6 回归矩阵**：覆盖编译、单多卡、零/小/热点 expert、Wave 边界、共享专家、三种 Combine、最大 shape 和性能基线。

### 12.3 最终判断

当前代码已经形成独立的 arch92 A8W8 1C1V 主链：Host 能力门禁和启动参数、Kernel 根入口、同号 C/V、Wave、QBMM MX、GM handoff、Workspace policy 与 channel backend 的方向一致。

正式闭环仍取决于三点：

1. A6 构建和发布稳定选择 arch92 Host/Kernel。
2. 1C1V 下的 Workspace、flag、UB 和多 Wave 协议完成全组合回归。
3. A8W4/A4W4 保持“骨架保留、入口不可达”，直到配套能力全部完成。
