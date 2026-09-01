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
3. arch35 两个 AIV 的阶段分工改为一个 AIV 的 Wave 调度，Dispatch、GMM1 后处理和 Combine 由同号 AIV 串联并与 AIC 协作。对应 [arch92 A8W8 Wave](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:243)。
4. `ArchitecturePolicy` 集中描述物理核编号、Dispatch cumsum 生命周期、GM handoff 和 Workspace layout，避免 arch35 的 1C2V 假设散落到各 stage。对应 [ArchitecturePolicy](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18)。
5. GMM 公共层切换到 QBMM MX BlockMmad 接口，ND/NZ 由模板参数选择，GMM1/GMM2 复用一个 `mmadContext_`。对应 [GMM Config](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47) 和 [context 成员](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:123)。
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
| A8W8 Wave | [`mega_moe_a8w8_wave.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:48) | 单 AIV 下的 Dispatch/GMM1/GMM2/Combine 调度 |
| Tiling key | [`mega_moe_tiling_key.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_tiling_key.h:27) | 只枚举 A6 当前可达组合 |
| 架构策略 | [`mega_moe_core_context.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18) | 物理核编号和跨阶段 policy |
| GMM 公共层 | [`mega_moe_gmm_common.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47) | QBMM MX、layout、L1 参数和 context |
| Workspace | [`mega_moe_workspace.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_workspace.h:146) | 1C1V flag、GM handoff 和 Combine counter 布局 |
| Dispatch stage | [`mega_moe_token_dispatch.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:638) | 单 AIV Dispatch 与 GMM1 ready 协议 |
| GMM1 stage | [`mega_moe_gmm1_activation.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:632) | GMM1、GM handoff、SwiGLU 和 GMM2 ready |
| GMM2 stage | [`mega_moe_gmm2_combine.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:759) | GMM2、GM handoff 和 1C1V Combine |
| Reset stage | [`mega_moe_workspace_reset.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_workspace_reset.h:73) | arch92 flag/counter 连续区清零 |

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

### 4.3 唯一 AIV 的职责重排

arch35 可以把不同阶段分给两个 AIV；arch92 的同号 AIV 必须依次复用自身 UB，承担 Dispatch、GMM1 后处理、Combine 和 Unpermute。为保留流水并行，arch92 新增 Wave 调度，让 AIV 准备下一 Wave 的 Dispatch 时，AIC 计算当前 Wave 的 GMM1。

总调度位于 [`ProcessMoeExpertStages`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:243)，各阶段入口位于 [`DispatchMoeExpert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:148)、[`RunGmm1ActivationForExpert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:176) 和 [`RunGmm2CombineForExpert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:204)。

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

arch92 从平台读取相等的 AIC/AIV 数和实际 UB 容量。已有的 buffer 计算框架不是 arch92 新算法，迁移变化在于它接收 A6 的 1C1V 核数和 384 KiB UB，而不能继续使用 arch35 的 AIV 配比或固定容量假设。

统一入口见 [`SetAdaptiveBufferConfigs`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:975)，受影响的地址计算包括 [`CalcDispatchBufferConfig`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:704)、[`CalcSendMaskBufferConfig`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:782) 和 [`CalcUnpermuteBufferConfig`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:843)。

### 5.4 Combine 同步槽

arch35 某些 1C2V 路径只让一个 AIV 子角色参与 Combine，因此可以按一半 AIV 计算 consumer 数。arch92 每个 AIV 都是独立逻辑 job，`logicalCoreCount` 必须直接使用 `blockAivNum`：

~~~cpp
logicalCoreCount = tilingData->blockAivNum;
return max(maxTokenGroupCountForOneExpert,
           logicalCoreCount);
~~~

该差异位于 [`CalcCombineSyncSlotCountPerExpert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:922)。它必须和 Kernel 的 group slot 布局、producer 通知范围及 consumer 分工同时一致。

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

arch92 给 `MegaMoe` 基类增加架构策略模板参数，将以下差异集中为编译期 policy：

~~~cpp
static constexpr bool PERSIST_DISPATCH_CUMSUM = true;
static constexpr bool USE_GM_HANDOFF = true;
static constexpr WorkspaceLayoutPolicy WORKSPACE_LAYOUT =
    WorkspaceLayoutPolicy::GM_HANDOFF_EXPERT_WAVE;
~~~

基类模板定义见 [`MegaMoe`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:55)，policy 定义见 [`MegaMoeArch92ArchitecturePolicy`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18)。

这三个值分别控制 Dispatch cumsum 保存、routed GMM 的 GM 交接和 Workspace 条件区。它们是 arch92 的跨模块契约，不能只在某个 stage 局部修改。

### 6.3 IsWeightNz 模板参数

arch92 Wave 增加 `IsWeightNz`，使 B layout 在编译期选择：

~~~cpp
using LayoutB = Std::conditional_t<
    IsWeightNz,
    Te::ZNLayoutPtn,
    Te::DNExtLayoutPtn>;
~~~

Wave 模板见 [`MegaMoeArch92A8W8Wave`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:48)，layout 配置见 [`GmmKernel::Config`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47)。同一实例只存在 DN 或 NZ 一种类型，不需要同时保留两套 context。

### 6.4 功能模板参数传播

arch92 为 1C1V 同步补充并向 GMM1/GMM2 传播：

| 模板参数 | arch92 中的作用 |
| --- | --- |
| `IsWaveFlagGrained` | 热点 expert 分 Wave 后，每个 256-row 区间独立同步 |
| `UseGmHandoff` | routed AIC 输出通过 GM sequence 交给同号 AIV |
| `IsWeightNz` | 选择 DNExt 或 ZN 的 B layout |
| `IsShared` | 保留 routed/shared 的地址和完成协议差异 |

GMM1 传播终点见 [`RunGmm1ActivationByMode`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:1088) 和 [`Gmm1ExecGeneric`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:632)；GMM2 传播终点见 [`RunGmm2ByMode`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:1211) 和 [`Gmm2ExecGeneric`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:759)。

### 6.5 GMM1 Gate/Up 布局

arch92 的 GMM1 scheduler 按 `hiddenDim / 2` 遍历 logical N tile，一个 tile 分别读取 Gate 与 Up：

~~~text
Gate B offset = nLoc
Up B offset   = nLoc + outputN
outputN       = hiddenDim / 2
~~~

两次 MMAD 完成后才发布同一份 SwiGLU 输入。实现位于 [`Gmm1ExecGeneric`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:632)。

## 7. BlockMmad 与 context

### 7.1 QBMM MX 接口

arch92 GMM 公共层改为直接使用与 A5 一致的通用 QBMM MX 接口：

~~~cpp
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
~~~

接入点位于 [`mega_moe_gmm_common.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47)。arch92 不再单独维护一套改变 Tensor API 语义的 BlockMmad 复制实现；架构差异留在外层 config、layout 和同步协议。

### 7.2 layout 与实际 shape

当前 A8W8 配置统一使用：

~~~cpp
LayoutA      = NDExt
LayoutB      = DNExt 或 ZN
LayoutScaleA = ScaleAND
LayoutScaleB = ScaleBDN
LayoutC      = NDExt
~~~

这些类型在 [`GmmKernel::Config`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47) 中定义。GMM1 与 GMM2 的 M/N/K 不同，但 BlockMmad 模板类型可以相同；每次调用的 tensor view 和 `actualShape` 决定真实问题大小。

### 7.3 小 M 与 L1 配置

外层 scheduler 仍以 256x256 tile 分工。arch92 的适配点是：小 M 时根据有效行数生成对齐后的 `blockM`，再在 A/B 数据、ScaleA/ScaleB、半片 L1 容量和单次 scale 搬运上限之间选择 `kL1`、`scaleKL1`。

小 M 选择位于 [`SelectBlockMmadTilingConfig`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:202)，L1 参数计算位于 [`CalcAdaptiveL1Params`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:131)。

当前初始化显式使用 `enableL0CPingPong=false`。这是 arch92 当前配置，不应在没有覆盖多 K chunk、K tail、DN/NZ 和 Wave 切换前单独改回双 buffer；初始化代码见 [`InitBlockMmad`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:337)。

### 7.4 四个概念 context 合并

由于 `IsWeightNz` 是编译期常量，且 GMM1/GMM2 的 `BlockMmad` 类型由 `static_assert` 保证一致，arch92 将以下四个概念成员：

~~~text
gmm1NdMmadContext_
gmm1NzMmadContext_
gmm2NdMmadContext_
gmm2NzMmadContext_
~~~

合并为一个 `mmadContext_`。类型约束和 AIC/AIV 存储选择位于 [`MmadContextStorage`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:82)，唯一成员位于 [`mmadContext_`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:123)。

`static_assert` 是编译期类型检查，不生成运行时代码，也不会带来 Kernel 性能开销。

### 7.5 Init 缓存条件

`BlockMmadContext` 只缓存 `Init` 真正消费的配置：`problemK`、`tileM`、`tileN`、`kL1` 和 `scaleKL1`。这些值变化时才重新初始化；problem 的实际 M/N 仍由本次 view 传入。

缓存逻辑位于 [`InitBlockMmad`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:337)。这个复用减少重复对象和重复初始化，但要求覆盖 GMM1/GMM2 切换、多 expert、多 Wave、DN/NZ 和两种 FP8 类型。

## 8. arch92 Wave 主流程

### 8.1 Process 的架构变化

`Process()` 中 Quant、SendMask、共享专家和 Unpermute 的算法仍复用 arch35；arch92 新增的是单 AIV 上的调用顺序、Workspace reset 位置以及 routed expert Wave：

~~~text
AIV 前处理
Reset + 本卡同步
共享专家 GMM1
跨 rank 同步
routed expert Wave
共享专家 GMM2
跨 rank 同步
Unpermute
~~~

入口见 [`MegaMoeArch92A8W8Wave::Process`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:358)。共享专家本身不是 arch92 新增能力，因此这里只把它作为时序边界，不展开既有算法。

### 8.2 routed expert Wave

arch92 新增三个位置状态：下一段 Dispatch 位置、当前 GMM1 位置，以及本 Wave 的起止位置。推进函数按 expert 剩余 token 和 Wave 的 M-group 容量截取切片，因此热点 expert 可以跨 Wave，一个 Wave 也可以覆盖多个小 expert。

位置推进见 [`AdvanceArch92ExpertTokenPositionInWave`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:23)，总循环见 [`ProcessMoeExpertStages`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:243)。

### 8.3 Dispatch 与当前 GMM1 交错

第一个 Wave 先 Dispatch。随后 AIV 准备下一 Wave 的 Dispatch，AIC/AIV 同时处理当前 Wave 的 GMM1 与 epilogue，从而在 1C1V 下保留搬运与 Cube 计算重叠。

相关调用位于 [`DispatchMoeExpert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:148)、[`PrepareGmmExpertState`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:162) 和 [`RunGmm1ActivationForExpert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:176)。

### 8.4 当前 Wave 的 GMM2/Combine

当前 Wave 的 GMM1 全部完成后，代码遍历该 Wave 覆盖的 expert 切片执行 GMM2 和 Combine。`startBlockIdx` 在 GMM1/GMM2 间连续传递，使 tile owner 仍按确定性轮转分配，而不是由空闲核心抢占。

GMM2 调度位于 [`RunGmm2CombineForExpert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:204)，量化 Combine 调度位于 [`ScheduleQuantizedMoeExpertCombine`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:1306)。

## 9. 1C1V 同步协议

### 9.1 初始化清零

arch92 把自身 Workspace flag/counter 组织成连续区，由 AIV 按逻辑 job 分块清零；AIC 在首次消费前等待本卡 reset 完成。实现位于 [`ResetSyncStatus`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_workspace_reset.h:73)，Host 端元素总数由 [`CalcHostFlagElementCount`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:938) 计算。

### 9.2 Dispatch 到 GMM1

arch92 的 Dispatch producer 按当前 source-rank segment 与 256-row GMM1 tile 的交集累加 ready 行数，GMM1 只等待自己将消费的 M-group。这样热点 expert 被拆成多个 Wave 时，各切片不会共用一个 expert 全局完成值。

生产端见 [`PublishGmm1TileReady`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:263)，Dispatch job 见 [`RunMoeExpertDispatchJob`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:638)，消费端见 [`WaitForGmm1InputReady`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:66)。

### 9.3 GMM1 到同号 AIV

routed Wave 启用 `UseGmHandoff` 后，AIC 将 Gate/Up 写入 GM，完成可见性保证后发布递增 sequence；同号 AIV 等待该 sequence，再从 GM 读取并执行 SwiGLU 和 MX quant。一个 logical tile 的 Gate 与 Up 都完成后才发布。

GMM1 handoff 位于 [`Gmm1ExecGeneric` 的 epilogue](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:760)，policy 来源见 [`USE_GM_HANDOFF`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18)。

### 9.4 activation 到 GMM2

AIV 完成 activation 量化后按 Wave 粒度通知 GMM2。生产端和消费端必须使用同一 `IsWaveFlagGrained` 语义，并根据 prefetch 路径的 activation tile 粒度共同推导通知次数与等待目标。

生产端见 [`NotifyGmm2InputReady`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:98)，消费端见 [`WaitForGmm2InputReady`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:479)。

### 9.5 GMM2 到 Combine

非量化 Combine 复用 GM handoff 模式：AIC 写 BF16 tile 并发布 sequence，同号 AIV 搬回 UB 后 Combine。量化 Combine 改为 token-group counter，producer 对对应 consumer slot 通知，consumer 等完整 N tile 数到齐后量化并发送。

producer 逻辑位于 [`NotifyCombineConsumersOfTileCompletion`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:512)，consumer 调度位于 [`ScheduleQuantizedMoeExpertCombine`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:1306)。

### 9.6 跨 rank 分工

1C1V 下 AIV 使用自身 `blockIdx_` 为起点、以 `blockNum_` 为步长分摊远端 rank，最后再进行本卡同步。这里的变化是逻辑 job 数和物理编号不再包含第二个 AIV，而不是改变跨 rank 握手算法。

实现见 [`CrossRankSyncInWorldSize`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:606)。

## 10. Workspace 与 UB

### 10.1 arch92 WorkspaceInfo

arch92 独立 `WorkspaceInfo` 的原因是 1C1V 下 flag 数量、GM handoff 条件区、Combine counter 和 Dispatch cumsum 区域都不同。不能让 arch92 stage 使用 arch35 地址公式。

可对照 [arch35 `WorkspaceInfo`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/common/mega_moe_workspace.h:137) 与 [arch92 `WorkspaceInfo`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_workspace.h:146)。

arch92 新增或重新解释的关键区域包括：

- Wave 粒度 Dispatch-to-GMM1 与 Activation-to-GMM2 状态。
- routed GMM1/GMM2 GM handoff 数据和 sequence。
- 量化 Combine group counters。
- Dispatch cumsum 跨阶段备份。
- 与单 AIV reset 分块匹配的连续 flag 区。

### 10.2 单 AIV UB 生命周期

arch92 的多个 `LocalTensor` 成员是同一物理 UB 上的地址视图。迁移需要重新安排唯一 AIV 在 Dispatch、GMM1 epilogue、Combine 和 Unpermute 间的复用关系，而不能把 arch35 两个 AIV 的常驻区简单相加。

关键视图分别由 [`DispatchBuffInit`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:374)、[`InitCombineBuffers`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:450)、[`SendAndQuantBuffInit`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:464) 和 [`InitTokenUnpermuteBuffers`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:551) 建立。

共享输入准备复用 Quant 输出槽，不额外叠加一块常驻 UB；别名关系见 [共享输入 buffer 绑定](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:522)。

### 10.3 Dispatch cumsum 持久化

`PERSIST_DISPATCH_CUMSUM=true` 时，唯一 AIV 在跨 expert/阶段复用 UB 前把 Dispatch 前缀状态写入 GM，需要继续 Dispatch 时再恢复。实现位于 [Dispatch cumsum 分支](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:495)。

A6 有 384 KiB UB，后续可以评估给 cumsum 预留常驻区以减少 GM 往返。但不能只删除该分支：必须先核算 Dispatch、activation、Combine、Unpermute 各阶段的同时存活区和最高地址，并同步修改 Host buffer 配置。

### 10.4 GM handoff 的后续优化

当前 GM handoff 是 arch92 1C1V 流水的明确 policy，不是共享专家功能正确性的前提。若后续改成 UB 直通、双 bank 或混合策略，必须同时建立 bank ownership、producer ready、consumer ACK、Wave 回绕规则和 Dispatch 占用约束。

当前选择位于 [`MegaMoeArch92ArchitecturePolicy`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18)，Workspace 条件区位于 [arch92 `WorkspaceInfo`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_workspace.h:146)。

## 11. A8W4/A4W4 预留

arch92 保留了 A8W4/A4W4 的模板分支、FP4 traits、prologue 类型、条件 Workspace 和 stage 接口。这是为后续支持保留的代码，不是当前正式能力，也不应作为无用代码删除。

保留的主要 stage 入口包括 [`RunGmm1A8W4`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:906) 和 [`RunGmm2A8W4`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:951)。

当前不可达由四层共同保证：

1. `GmmKernel::Config` 的编译期约束，见 [A8W8 `static_assert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47)。
2. Host 能力门禁，见 [`CheckArchitectureCapability`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:1970)。
3. arch92 tiling key 未枚举 E2M1 等组合，见 [`mega_moe_tiling_key.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_tiling_key.h:27)。
4. A6 根入口只实例化 A8W8，见 [`mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)。

正式放开时必须成套修改入口、Host、tiling key、BlockMmad/Prologue、Workspace、同步和测试，不能只去掉 `static_assert`。

## 12. 构建、注册与封装

### 12.1 OpDef 注册

`MegaMoe` 的 OpDef 增加 `ascend910_96`，使框架允许在 A6 上选择该算子：

~~~cpp
this->AICore().AddConfig(
    "ascend910_96", aicore_config);
~~~

注册位置见 [`mega_moe_def.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/mega_moe_def.cpp:195)。这只完成 SoC 声明，不能替代 Host tiling 和 Kernel 源选择。

### 12.2 Kernel 与 Host 源选择

A6 Kernel 应选择根 [`op_kernel/mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)，arch35 仍选择自己的 [`arch35/mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp:72)。

Host 同时存在 arch35 和 arch92 的 tiling 注册。构建系统必须确保一个 SoC 目标只链接一套 `IMPL_OP_OPTILING(MegaMoe)`；两套注册位置分别见 [arch35 Host 注册](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp:2064) 和 [arch92 Host 注册](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2094)。

通用递归收集和按架构过滤分别位于 [`obj_func.cmake` 通用收集](/D:/gitcode/ops-transformer_910b2/cmake/obj_func.cmake:108) 与 [按架构收集](/D:/gitcode/ops-transformer_910b2/cmake/obj_func.cmake:291)。

### 12.3 Host 编译依赖

arch92 Host 与 Kernel 增加独立头文件后，`op_host/CMakeLists.txt` 必须为 A6 compute unit 提供 Tensor API、Blaze、通信和生成头文件的 include 闭包。相关配置见 [MegaMoe Host CMake](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/CMakeLists.txt:25)。



### 12.4 Torch backend

Torch extension 为 A6 平台映射 `channel` backend。修改位置见 [`CommContextManager` backend 映射](/D:/gitcode/ops-transformer_910b2/torch_extension/cann_ops_transformer/ops/mega_moe.py:265)。

需要继续核对运行时实际平台字符串是 `Ascend910_9699`、`Ascend910_96` 还是其他别名；映射名不一致会使上层未进入 A6 通信后端。

### 12.5 发布分类

当前发布规则仍需确认 arch92 Host、Kernel 目录和 A6 根入口被打包。检查入口见 [`classify_rule.yaml` 的 MegaMoe 条目](/D:/gitcode/ops-transformer_910b2/classify_rule.yaml:740)。

这一步与编译成功相互独立：本地构建能找到源码，不代表安装包一定包含新增目录。

## 13. 修改点总表与未闭环事项

### 13.1 修改点总表

| 修改面 | 关键函数/代码段 | arch92 相对 arch35 的变化 |
| --- | --- | --- |
| C/V 配比 | [`KERNEL_TYPE_MIX_AIC_1_1`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44) | 1C2V 改为同号 1C1V |
| 物理编号 | [`GetPhysicalBlockIdx`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18) | AIC/AIV 均直接使用 `GetBlockIdx()` |
| Host 核数 | [`MegaMoeTilingFuncImplPublic`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2010) | 校验 `aivNum == aicNum` |
| Host 能力 | [`CheckArchitectureCapability`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:1970) | 当前只放行 A8W8、MXFP、MTE |
| Wave | [`ProcessMoeExpertStages`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:243) | 两 AIV 分工改为单 AIV 分阶段交错 |
| Dispatch ready | [`PublishGmm1TileReady`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:263) | 按 Wave/M-group 发布有效行数 |
| GMM1 | [`Gmm1ExecGeneric`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h:632) | Gate/Up 成对计算并与同号 AIV handoff |
| GMM2 | [`Gmm2ExecGeneric`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:759) | 适配 Wave ready 与 1C1V Combine |
| 布局 | [`GmmKernel::Config`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47) | `IsWeightNz` 编译期选择 DNExt/ZN |
| BlockMmad | [`InitBlockMmad`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:337) | 接入 QBMM MX 并缓存有效 Init 配置 |
| Context | [`mmadContext_`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:123) | GMM1/GMM2、ND/NZ 四个概念对象合并为一个 |
| Combine slot | [`CalcCombineSyncSlotCountPerExpert`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:922) | consumer 数按全部 1C1V job 计算 |
| Workspace | [arch92 `WorkspaceInfo`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_workspace.h:146) | 新增 Wave flag、GM handoff、cumsum 和 group counter |
| UB | [`DispatchBuffInit`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe.h:374) | 唯一 AIV 分阶段复用 384 KiB UB |
| 注册 | [`AddConfig("ascend910_96")`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/mega_moe_def.cpp:195) | OpDef 接受 A6 |
| 封装 | [`CommContextManager`](/D:/gitcode/ops-transformer_910b2/torch_extension/cann_ops_transformer/ops/mega_moe.py:265) | A6 选择 channel backend |

### 13.2 未闭环的代码整改

1. **构建映射**：确认 `ascend910_96` 只选择 [arch92 Host tiling](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2094) 和 [A6 Kernel 根入口](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)，不会同时链接 arch35 注册。
2. **发布分类**：更新 [`classify_rule.yaml`](/D:/gitcode/ops-transformer_910b2/classify_rule.yaml:740)，把 arch92 Host、Kernel 和根入口纳入交付清单。
3. **平台名称**：核对 [Torch backend 映射](/D:/gitcode/ops-transformer_910b2/torch_extension/cann_ops_transformer/ops/mega_moe.py:265) 是否覆盖运行时所有 A6 名称。
4. **Host 能力同步**：评估 [`CheckAttrParams`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:435) 中 `hiddenDim` 和 `numMaxTokensPerRank` 是否应同步当前 arch35 语义，并成套重算 buffer/Workspace。
5. **Context 复用回归**：覆盖多 expert、多 Wave、DN/NZ、E4M3FN/E5M2、prefetch on/off 和三种 Combine mode。
6. **384 KiB cumsum 常驻**：评估取消 [Dispatch cumsum GM 恢复](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:495) 的收益；实施前完成全 UB 生命周期和最高地址核算。
7. **routed handoff 优化**：评估 [`USE_GM_HANDOFF`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18) 的 GM、UB 单槽、双 bank 或混合策略，补齐 credit/ACK 后再改变介质。
8. **BlockMmad 配置**：评估 [`enableL0CPingPong=false`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:337) 的性能影响；任何调整都要覆盖 K tail、layout 和 Wave 切换。
9. **A8W4/A4W4**：保留现有 stage 骨架，按入口、Host、BlockMmad、Workspace、同步和测试的完整链条逐步放开。
10. **A6 回归矩阵**：覆盖编译、单多卡、零/小/热点 expert、Wave 边界、共享专家、三种 Combine、最大 shape 和性能基线。

### 13.3 最终判断

当前代码已经形成独立的 arch92 A8W8 1C1V 主链：Host 能力门禁和启动参数、Kernel 根入口、同号 C/V、Wave、QBMM MX、GM handoff、Workspace policy 与 channel backend 的方向一致。

正式闭环仍取决于三点：

1. A6 构建和发布稳定选择 arch92 Host/Kernel。
2. 1C1V 下的 Workspace、flag、UB 和多 Wave 协议完成全组合回归。
3. A8W4/A4W4 保持“骨架保留、入口不可达”，直到配套能力全部完成。

