# MegaMoe Arch92 合入 Arch35 迁移说明

## 1. 文档定位

本文记录 `D:\gitcode\ops-transformer_5\mc2\mega_moe` 中，以最新 arch35 为基础完整接入 arch92 A8W8/MTE/1C1V 的示例实现，并回答两个代码检视问题：

1. arch92 的哪些行为必须独立维护，哪些实现可以与 arch35 共用；
2. 为支持 arch92，原 arch35 公共文件实际被侵入式修改了多少。

“完整接入”是指 Host tiling、Workspace、Kernel 入口、1C1V Wave、Dispatch cumsum、GMM1 GM handoff、BlockMmad context、GMM2/Combine 和构建注册已形成静态闭环。按本次约定，没有进行编译、运行、精度或性能验证，因此本文不把示例代码描述为已验收能力。

本文的能力范围和侵入性统计是 **A8W8 首次迁入阶段的快照**：当时 arch92 只放行 A8W8 + MTE。之后 A4W4 已接入源码路径，A8W4 仍被门禁拒绝；最新状态及未验收项见 [A4W4/A8W4 实施记录](/D:/gitcode/ops-transformer_5/MegaMoe%20Arch92%20A4W4与A8W4统一实施记录.md)。

## 2. 最终目录

```text
mc2/mega_moe/op_kernel/
|-- arch35/       # arch35 专属入口、A8W8/A8W4/A4W4 Wave、Layered
|-- arch35_92/    # 两个平台共用的基类、stage、Blaze、Workspace 和策略接口
`-- arch92/       # arch92 专属入口和 A8W8 Expert Wave

mc2/mega_moe/op_host/op_tiling/
|-- arch22/
`-- arch35_92/    # arch35/arch92 共用 Host tiling，运行时按 DAV 选择策略
```

`arch35_92` 不是新架构名，而是代码所有权边界。依赖方向固定为：

```text
arch35  ----+
            +----> arch35_92 ----> mc2/common、Tensor API、系统头
arch92  ----+
```

arch92 不再反向包含 arch35 文件；控制流差异大的入口和 Wave 留在架构目录，数据面和通用 GMM 实现放在公共目录。

公共基类见 [`mega_moe.h`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:53)，arch92 Wave 见 [`mega_moe_wave.h`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:43)。

## 3. 迁入内容

### 3.1 Kernel 入口与能力边界

arch35 入口继续使用 `KERNEL_TYPE_MIX_AIC_1_2`，并保留 A8W8、A8W4、A4W4 和 URMA/Layered 实例。arch92 使用独立入口和 `KERNEL_TYPE_MIX_AIC_1_1`，只实例化 FP8 A8W8 的 MTE 路径。

arch92 的 task type 和实例门禁见 [`mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_apt.cpp:73)。Host 侧再次按 DAV、拓扑和 GMM mode 做能力校验，避免只靠 Kernel 预处理条件静默跳过不支持组合，见 [`CheckArchitectureCapability`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2595)。

### 3.2 1C2V 与 1C1V 硬件策略

arch35 的一个逻辑 block 对应 1 个 AIC 和 2 个 AIV；arch92 对应 1 个 AIC 和 1 个 AIV。公共代码不能继续散落使用 `GetSubBlockIdx()`、`GetTaskRation()`、`GetBlockNum() * 2` 等固定 1C2V 假设，因此将以下信息集中到 Hardware Policy：

- 逻辑 block 编号；
- AIV job 编号和 job 总数；
- compute AIV 与 dispatch/combine AIV 的角色门禁；
- 同一 AIC 对应的第一个物理 AIV 编号。

arch35 与 arch92 的映射分别见 [`MegaMoeArch35HardwarePolicy`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_arch_policy.h:21) 和 [`MegaMoeArch92HardwarePolicy`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_arch_policy.h:61)。

Hardware Policy 只回答“有几个 C/V、当前是谁”，不决定 cumsum、Workspace 或 GM/UB 交接方式。

### 3.3 Execution Policy 与隐式依赖解除

arch35 原代码中有三组成立于既有实例、但迁入 arch92 后不再成立的隐式绑定：

1. cumsum 是否持久化原本由 W4 mode 间接决定，但 arch92 是 A8W8，也需要跨 Wave 保存 cumsum；
2. GMM1 写 GM 还是 UB 原本由 prefetch/W4 路径间接决定，但 arch92 非 prefetch A8W8 也需要 GM handoff；
3. Workspace 条件区原本由 GMM mode 决定，但 arch92 的 mode 仍是 A8W8，无法据此推导 cumsum、GMM1 输出和顺序 flag。

因此公共基类增加 `ExecutionPolicy` 模板参数，默认策略保持 arch35 行为；arch92 A8W8 显式选择：

```cpp
PERSIST_DISPATCH_CUMSUM = true
USE_GM_HANDOFF = true
WORKSPACE_LAYOUT = GM_HANDOFF_EXPERT_WAVE
```

策略定义见 [`mega_moe_execution_policy.h`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_execution_policy.h:18)，arch92 Wave 的策略绑定见 [`MegaMoeArch92Wave`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:43)。

这三个值由具体 Kernel variant 选择，而不是简单由“arch92”或“A8W8”二选一。这样后续增加其他 arch92 流水时，不会被迫继承本实例的 GM handoff。

### 3.4 Arch92 Expert Wave

arch92 保留独立 Wave 文件，因为唯一 AIV 必须在不同时段串行承担 arch35 AIV1 的 Dispatch、AIV0 的 GMM1 epilogue，以及 GMM2/Combine 工作。这里不是把两个 AIV 的并行工作机械合并，而是复用 A8W4 风格的 Expert Wave 生命周期，同时删除 A8W4 的 W4 到 W8 Prologue。

主循环先准备当前/下一 Wave 的 Dispatch，再执行当前 Wave GMM1，随后按门槛选择即时或滞后一拍执行 GMM2/Combine。调用顺序见 [`ProcessMoeExpertStages`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:516)。

arch35 的 A8W8/A8W4/A4W4 Wave 没有被合并进这个函数，因为双方在角色并行、UB 生命周期和同步时机上的差异已经超过少量 policy 能清晰表达的范围。

### 3.5 Dispatch cumsum 跨 Wave 持久化

arch92 唯一 AIV 的 Activation 会复用 Dispatch cumsum 所在的 UB 区。第一次生成完整 count/prefix 表时，`PrepareMoeExpertTokenCountTable<true>` 同步将 cumsum 备份到按物理 block 划分的 GM 区。启动阶段连续 Dispatch W0/W1 时 UB prefix 仍有效；后续每次预取下一 Wave 前，只恢复该 Wave 涉及的专家前缀范围。

备份入口见 [`PrepareMoeExpertTokenCountTable`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_token_dispatch.h:421)，范围恢复见 [`ReloadDispatchCumsumRange`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_token_dispatch.h:486)，arch92 调用点见 [`DispatchNextWave`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:187)。

该设计先保证与既有 A8W4 语义一致。A6 有 384 KiB UB，后续可以单独评估 cumsum 全程常驻 UB，但不能在没有完整 UB 最高地址和生命周期核算时直接删除 GM 备份。

### 3.6 GMM1 GM handoff

arch92 非 prefetch A8W8 不再让 AIC 把 GMM1 结果写入只对某个 AIV 可见的 UB。AIC 将完整 tile 写入 `gmm1MmadResPtr`，随后按同一物理 block 的单调序号更新 `gmmToEpilogueFlag`；唯一 AIV 等待目标序号，从 GM 搬入 tile，执行 SwiGLU/量化并发布 GMM2-ready。

公共 GMM1 入口新增 `UseGmHandoff` 模板参数。默认值为 `false`，因此 arch35 普通 A8W8 仍走原 UB ping-pong；arch92 显式选择 GM 路径。AIC 写 GM 和通知逻辑见 [`Gmm1AicMmadToGmGeneric`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:139)，AIV 取回和 epilogue 逻辑见 [`Gmm1ExecGeneric`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:432)。

arch92 地址配置显式要求 GMM1 GM 输出和顺序 flag，见 [`UpdateMoeExpertGmm1GlobalBuffer`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:752) 与 arch92 [`ProcessGmm1Wave`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:243)。

GM handoff 模式不会执行 UB ping-pong 的 `EndSync`。否则 GM 序号会被误当作 UB 同步计数，可能产生错误等待。该条件位于 arch92 Wave 的阶段收尾处，见 [`ProcessMoeExpertStages`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:615)。

### 3.7 持久 BlockMmad context

公共 GMM1/GMM2 接口原本已经支持调用方传入 `BlockMmadContext`；传空指针时创建函数内局部 context。Arch92 Wave 持有 AIC 专用 GMM1 context；GMM2 类型相同时复用它，类型不同时传空指针。共享专家两段类型不同时也使用各阶段局部 context。

context 类型由 GMM1/GMM2 的实际 Generic Config 推导，按 `std::is_same_v` 选择是否复用，而不再要求所有模式的 BlockMmad 类型一致。定义见 [`Gmm1PersistentConfig`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:64)，GMM1/GMM2 传递点分别见 [`ProcessGmm1Wave`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:294) 和 [`ProcessGmm2WaveProblem`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:375)。

`BlockMmadContext::Init` 仍会在 K、tile 或 L1 配置变化时重新初始化；“持久”表示对象生命周期跨函数/Wave，而不是无条件只调用一次 Init。

### 3.8 Workspace Host/Kernel 同源

公共 `WorkspaceLayout` 增加 `WorkspaceLayoutPolicy`。`GM_HANDOFF_EXPERT_WAVE` 在 MTE A8W8 的默认布局上额外分配：

- 每个 AIV block 的 cumsum 备份；
- GMM1 BF16 输出区；
- 每个 AIC 独占 cache line 的 GMM-to-epilogue 顺序 flag。

布局枚举和构造入口见 [`mega_moe_workspace.h`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_workspace.h:31)，MTE 条件分配见 [`InitializeMte`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_workspace.h:122)。

Kernel 从 `ExecutionPolicy::WORKSPACE_LAYOUT` 构造地址，见 [`MegaMoe::Init`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:302)。Host tiling 根据 `GetCurNpuArch()` 选择同一枚举后计算 workspace 总量，见 [`CommitTilingResult`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2619)。这避免了 Host 预算按默认 A8W8、Kernel 却按 GM handoff 推进地址的错位。

reset 不复制第二套公式；它继续使用 `WorkspaceLayout` 在分配时记录的连续 flag 元素数，因此新增顺序 flag 会自动纳入清零范围。

### 3.9 GMM2/Combine 与 1C1V 同步

GMM2 的通用数据面继续共用。arch92 对非量化 Combine 通过 `NotifyCombineTileReady` 让唯一 AIV 消费 GM 输出；量化 Combine 使用现有 Wave-ready/计数协议。consumer 数、逻辑 job 数和同步槽 stride 均从 Hardware Policy 和 Host 的真实 AIV 数推导，不再套用 arch35 的 1C2V 除二规则。

公共 GMM2 的角色门禁见 [`Gmm2ExecGeneric`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm2_combine.h:735)，Host 的 1C1V 校验见 [`CheckAndSetPlatformParams`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2567)。

### 3.10 共享专家

共享专家继续复用公共基类现有 GMM1、Activation、GMM2 和输出聚合实现。本次没有新增“arch92 特殊共享专家算法”；变化仅来自 1C1V Hardware Policy 和 arch92 Workspace variant。arch35 已有且两边行为相同的共享专家功能不作为 arch92 新增点重复描述。

公共共享专家调用位于 [`ProcessWave`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:771)。

### 3.11 构建与 OpDef

`ascend950` 保持 arch35 编译选项；`ascend910_96` 选择 arch92 入口并保留自动同步。此前临时加入的 `Quant -> quant` 宏没有实际接口依据，正式迁入不再包含该兼容头，公共代码直接使用当前量化头提供的 `Quant` 命名空间。

构建分支见 [`CMakeLists.txt`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/CMakeLists.txt:24)，双平台 OpDef 注册见 [`mega_moe_def.cpp`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/mega_moe_def.cpp:293)。

## 4. 公共文件侵入量

统计基线为 `ops-transformer_5` 的原 arch35 实现（迁移提交的父版本），只统计迁入 `arch35_92` 的 Kernel 文件和公共 Host tiling，不把 arch92 专属入口/Wave 算作公共侵入。

### 4.1 文件级结果

| 指标 | 数量 |
| --- | ---: |
| `arch35_92` 公共文件总数 | 36 |
| 可对应原 arch35 文件 | 33 |
| 与原 arch35 字节/行语义一致 | 26 |
| 对原 arch35 有内容修改 | 7 |
| 新增薄 policy 文件 | 3 |

也就是说，33 个原 arch35 文件中 26 个保持不变，约 79% 的可迁移文件只是移动所有权；真正修改的原文件占约 21%。

### 4.2 修改文件规模

| 公共文件 | 相对原 arch35 增删 | 修改原因 |
| --- | ---: | --- |
| `arch35_92/mega_moe.h` | `+26/-19` | 增加 Execution Policy 模板参数、Workspace policy 透传和 cumsum GM 视图绑定 |
| `common/mega_moe_utils.h` | `+3/-2` | 1C1V/1C2V 逻辑核数量改由 Hardware Policy 提供 |
| `common/mega_moe_workspace.h` | `+25/-15` | 增加 GM handoff Expert Wave 的条件布局 |
| `stage/mega_moe_gmm1_activation.h` | `+61/-46` | Generic A8W8 增加显式 GM handoff，复用已有 GM 同步和 epilogue 数据面 |
| `stage/mega_moe_gmm2_combine.h` | `+12/-12` | consumer/job 数和角色门禁改由 Hardware Policy 提供 |
| `stage/mega_moe_token_dispatch.h` | `+4/-4` | Dispatch 角色和 job 数改由 Hardware Policy 提供 |
| Host `mega_moe_tiling.cpp` | `+45/-9` | DAV 识别、1C1V 校验、能力门禁和 Workspace policy 选择 |

7 个原文件合计约 `+176/-107`。新增的 3 个 policy 文件是：

- `common/mega_moe_arch_policy.h`：108 行；
- `common/mega_moe_execution_policy.h`：33 行；
- Host `mega_moe_host_arch_policy.h`：37 行。

### 4.3 侵入性判断

公共修改量属于“文件面较小、关键路径中等侵入”：

- 文件面较小：26/33 个原文件完全不需要改函数逻辑；Blaze、量化、SendMask、Unpermute、reset 等主体直接共用。
- 关键路径中等侵入：Workspace、GMM1 handoff、Dispatch cumsum 和 Combine 核角色会影响地址与同步，虽然改动行数不大，但必须成套检视。
- 没有把 arch92 Wave 控制流塞进 arch35 文件，也没有在公共函数中大面积铺设备宏。

如果只追求更少 diff，可以复制整套 arch92 stage；那会降低眼前的公共修改量，但会形成两套长期漂移的数据面。当前方案把真正不同的 Wave 留在 arch92，只在公共层增加无法回避的策略入口，维护成本更低。

## 5. Arch35 行为保持方式

arch35 不显式传 Execution Policy，使用 `MegaMoeDefaultExecutionPolicy`：不强制持久化 A8W8 cumsum、不启用 A8W8 GM handoff、Workspace 使用默认布局。A8W4/A4W4 仍由原 GMM mode 条件分配 cumsum/GMM 中间区并走原路径。

因此新增模板参数不会改变 arch35 现有 Wave 的调用表达；arch35 的专属 A8W8/A8W4/A4W4/Layered 文件只改公共 include 路径和原有 Hardware Policy 接口，不承载 arch92 控制流。

默认策略见 [`MegaMoeDefaultExecutionPolicy`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_execution_policy.h:18)。

## 6. 建议检视顺序

1. 检视 arch92 根入口的 1C1V 和能力实例化。
2. 检视 Hardware Policy 的逻辑核/角色映射。
3. 检视 Execution Policy 三个值及 Host/Kernel Workspace 一致性。
4. 沿 `ProcessMoeExpertStages -> ProcessGmm1Wave -> RunGmm1Generic` 检视 GM handoff。
5. 沿 `PrepareMoeExpertTokenCountTable -> DispatchNextWave -> ReloadDispatchCumsumRange` 检视 cumsum 生命周期。
6. 检视持久 BlockMmad context 的实际类型和 GMM1/GMM2 传递点。
7. 最后检视 GMM2/Combine consumer 数、Host CV 比例和构建注册。

## 7. 验证边界

本次只做静态迁移和一致性检查，没有声称以下事项已通过：

- ascend950 与 ascend910_96 双平台编译；
- 单卡/多卡运行、精度和性能；
- A8W8 DN/NZ、E4M3FN/E5M2、prefetch 开关和三种 Combine mode；
- 共享专家、零 token expert、热点 expert、跨 Wave expert；
- arch35 A8W4/A4W4/URMA 回归。

正式合入前应至少完成上述矩阵。特别需要验证 GM 顺序 flag 单调性、Host/Kernel Workspace 总量一致性，以及 GMM1/GMM2 交替时 `BlockMmadContext::Init` 的配置刷新行为。
