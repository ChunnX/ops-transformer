# Arch92 单 AIV 迁移协议问题修复记录

## 1. 范围与进度

- 日期：2026-09-27；更新至 2026-09-28 的共享 GMM1 GM 交接方案。
- 修改仓库：`D:\gitcode\ops-transformer_5`。
- 修改前基线：`df26153c0`（`mega合并示例`），开始修改时工作区干净。
- 状态：四类问题的代码修复、源码约束检查和主机协议模型验证已完成；CANN 编译、模拟器运行、精度与性能回归待验证。
- 累计修改范围：一个 Arch92 Wave 文件、六个 `arch35_92` 公共头文件；其中架构策略头文件只修正注释。本次共享 GM 交接只修改公共 `mega_moe.h` 和 `stage/mega_moe_gmm1_activation.h`。
- 没有修改 Arch35/Arch22 专用文件、Host tiling、Tensor 仓、golden、解析脚本或构建配置；没有增加 kernel 打点。

本次按 `_910b2` 的迁移原理核对新代码，不把所有 Wave 顺序变化都当成 1C1V 不兼容。参考说明见 [原迁移审计](</D:/gitcode/ops-transformer_910b2/mega文档/MegaMoe 从 Arch35 到 Arch92 迁移审计.md>) 和 [UB 生命周期分析](</D:/gitcode/ops-transformer_910b2/mega文档/MegaMoe A6 A8W8 384KiB UB 优化方案.md>)。

## 2. 必须保持的迁移约束

Arch92 routed Wave 的来源是 Arch35 A8W4：保留原 AIV1 的 Dispatch、Activation、Combine 职责，删除原 AIV0 的 W4 到 W8 prologue，使用直接计算的通用 BlockMmad。不是把 Arch35 A8W8 的两个 AIV 分支直接拼到一起。

1. AIC 生产 tile 与唯一 AIV 消费 tile 必须使用相同的 scheduler 和分核游标；执行了真实调度，就不能再补一次“跳过调度”的推进。
2. routed GMM 结果通过 GM 交接，避免 AIC 提前覆盖唯一 AIV 正在使用的 Dispatch/Activation UB；cumsum 仍需按原策略备份和恢复。
3. GM 数据地址与同步协议是两件事。数据继续写 GM，不代表 Ready/ACK 仍然是原来的 GM sequence。
4. 硬件事件目标、物理 AIV 编号、GM 计数槽和实际参与者必须同时按 1C1V 映射。
5. Arch92 共享阶段沿用 Arch35 A8W4 的发送后 Activation 分工：唯一 AIV 先发送路由，再从 GM 读共享 GMM1 结果；AIC 使用直接计算接口写 GM，避免覆盖路由 UB。

本次保留 `PERSIST_DISPATCH_CUMSUM`、`USE_GM_HANDOFF` 和 `GM_HANDOFF_EXPERT_WAVE` 的取值，并将已有 `USE_GM_HANDOFF` 传入共享 Generic GMM1。Arch35 默认值仍为 false。见 [执行策略](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_execution_policy.h:18>)。

## 3. 问题一：非量化路径重复推进 GMM1 游标

### 3.1 原因与错误链

Arch35 A8W8 中，AIV1 执行 Dispatch，不进入 GMM1/Activation scheduler，因此要用 `AdvanceStartBlockIdxForSkippedGmm1` 补齐游标，再与配对 AIC 一起进入非量化 GMM2/Combine。

Arch92 却在 `PrepareDispatchWave` 补推进后，仍调用 `ProcessGmm1Wave` 执行 Activation。`GmmRuntimeState::startBlockIdx` 是引用，公共 `RunGmm1Generic` 又会推进同一份游标。

```text
修改前：
AIC：GMM1 scheduler 按原游标生产 -> 推进一次
AIV：Dispatch 时补推进一次 -> 按错误游标消费 GMM1 -> 再推进一次

28 对 C/V、初始游标 0、当前 Wave 只有 4 个 tile：
AIC 生产者 = 0、1、2、3
AIV 消费者 = 4、5、6、7
```

这会使 per-core 完成通知与实际消费 tile 不匹配，可能读错结果或等待不存在的通知。满 Wave 的 tile 总数恰为核数整数倍时，取模会掩盖问题，尾 Wave、小专家不能依赖这一巧合。

该补偿逻辑来自 [!10956](https://gitcode.com/cann/ops-transformer/pull/10956/) 的 Arch35 A8W8 分支；问题在于迁入 Arch92 后，原先“不执行 Activation”的前提不再成立。

### 3.2 修改

- 删除 Arch92 中该补偿函数的声明、定义和调用。
- `PrepareDispatchWave` 只负责准备 Dispatch，不再接受无用的 `waveBeginPosition`，也不再返回随后会被 GMM1 结果覆盖的位置。
- 保留实际 GMM1/GMM2 调度的游标推进、GMM2 lag 和原有轮转修正。
- Arch35 A8W8 的补偿函数保持不变，因为其 AIV1 仍然跳过 GMM1/Activation。

对应代码：[Dispatch 准备](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:466>)、[唯一 AIV 的实际调用顺序](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:554>)、[公共 GMM1 游标更新](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:667>)。

## 4. 问题二：GMM2 Ready/ACK 固定指向 AIV1

### 4.1 原因与错误链

[!11640](https://gitcode.com/cann/ops-transformer/pull/11640/) 除 interweave 外还引入 GM 输出的有界 Mode 4 硬通知。非量化 GMM2 的 `Gmm2CombineSync` 固定在 AIC 侧加 16，选择 AIV1；AIV 侧使用本地事件编号。

Arch92 只有 sub-block 0，复制该类却不改变事件目标，就不能与唯一 AIV 的本地 Ready/ACK 匹配。只把 `RunsDispatchVectorWork()` 改为 true 不会改变硬件事件地址。

### 4.2 修改

`Gmm2CombineSync` 用架构配比生成编译期偏移，并同步修改 `WaitForCombine`、`NotifyCombine` 和 `EndSync` 三个 AIC 端点：

```cpp
static constexpr uint16_t COMBINE_AIV_FLAG_OFFSET =
    (MegaMoeHardwarePolicy::VECTOR_CORE_COUNT_PER_CUBE - 1U) * GMM_AIV_FLAG_OFFSET;
```

| 架构 | 配比 | AIC Ready | AIC 等待 ACK | AIV 本地 Ready/ACK |
| --- | --- | --- | --- | --- |
| Arch35 | 1C2V | 26 | 27 | 10 / 11 |
| Arch92 | 1C1V | 10 | 11 | 10 / 11 |

不改变 Mode 4、FIX/MTE2/MTE3/Scalar 流水线选择、15 个未确认 tile 的额度管理或结束排空。偏移是编译期常量，没有新增运行时分支或同步对象。

对应代码：[Gmm2CombineSync](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_gmm_epilogue_sync.h:294>)。这是源码中的事件映射修复，仍需在 A6 验证实际硬件行为。

## 5. 问题三：共享输入同步参与者和 UB 所有权不一致

### 5.1 原因与错误链

[!11620](https://gitcode.com/cann/ops-transformer/pull/11620/) 让 Arch35 的共享计算与路由发送重叠：AIV1 发送路由并进行输入同步，AIC/AIV0 提前执行共享计算。

迁入后，`SyncInputAcrossRanks` 的共享分支只检查 `RunsDispatchVectorWork()`。Arch92 该判断恒为 true，AIC 也进入发送核屏障，和对应 AIV 以同一个逻辑编号写 arrival/epoch 槽。AIC 没发送路由，不能代表路由已完成；这种重复参与可能造成提前放行和计数竞争。

仅让 AIC 跳过输入同步仍不足以修复：它可能提前进入共享 GMM1 并写 UB，而唯一 AIV 还在读取和发送路由。Arch35 依靠 AIV0/AIV1 两份独立 UB 避免冲突，1C1V 不具备这个条件。

上述冲突针对原先共享 GMM1 写 UB 的实现。首轮曾保留 UB 输出并补全核屏障；2026-09-28 按用户确认的迁移方案改为 GM 交接，已移除这次额外屏障。路由输入 UB 基址见 [SendAndQuantBuffInit](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:510>)，原 Generic GMM1 输出 UB 基址见 [Gmm1AicMmadGeneric](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:193>)。

### 5.2 修改

1. `SyncInputAcrossRanks` 开头显式排除 AIC，仅由合适的 AIV 执行路由子组屏障和跨卡握手。
2. Arch92 共享 GMM1 调用已有 Generic GM 输出分支；唯一 AIV 完成发送和输入同步后，按 Ready/ACK 读取 GM tile 并执行 Activation。
3. 输入准备完成后的第一次全核 `SyncAll<false>()` 保留；共享 GMM1 前不再添加第二次全核屏障。结果空间、事件目标、ACK 排空与 GM 路径一起配置，详见第 10 节。

修复后的共享输入阶段顺序：

```text
全部核：输入量化与 flag 清零完成 -> 原有全核屏障
AIC：跳过路由和跨卡握手 -> 直接计算共享 GMM1 -> 写 GM / 发布 Ready
AIV：发送路由 -> 子组屏障 -> 跨卡握手 -> 子组屏障 -> 等 Ready / 读 GM / Activation / ACK
```

路由发送函数已有完成等待，跨卡子组同步末尾保留 `PipeBarrier<PIPE_ALL>()`。共享 GM 输出不会覆盖路由 UB，因此 AIC 可在路由期间计算；最多允许 15 个未确认 tile，额度耗尽后等待 AIV 的 ACK。

Arch35 原共享 GMM1 选择、输出位置和发送重叠顺序保持不变；Arch92 不启用共享专家时不执行共享 GMM1。

对应代码：[参与者判断](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:769>)、[共享 GM 交接调用](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:822>)。

## 6. 问题四：1C1V 错写下一核的 epoch 槽

### 6.1 原因与错误链

共享输入同步由每个逻辑 block 的路由 AIV 代为握手。Arch35 需要同时更新这对 AIV 的物理计数槽，使后续全部 AIV 参与输出同步时，起始 epoch 一致。

```text
Arch35：block b -> 物理槽 2b、2b+1
Arch92：block b -> 物理槽 b
```

原实现不区分架构，握手写当前槽后，还写 `syncCount + 64B`。在 Arch92 上，这已是下一 block 的槽，而非当前 block 的另一个 AIV。

例如所有槽初始为 5：block0 握手后将槽0、槽1都写成6；block1随后读取槽1，计算自己的新 epoch 为7。不同核由此使用不同 epoch，影响本次握手、count 到达校验或后续输出同步。这个问题独立于 AIC 错误参与，即使排除 AIC 仍需修复。

### 6.2 修改

保留原握手和子组屏障，只在 `VECTOR_CORE_COUNT_PER_CUBE == 2U` 时更新相邻槽：Arch35 继续更新一对槽，Arch92 只更新自己。

对应代码：[架构条件和相邻槽更新](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_utils.h:530>)。

第 7 节描述当前行为；第 8、9 节保留前两轮验证记录，最新共享 GM 交接与验证见第 10 节。

## 7. 保留的行为与性能边界

- 没有撤销 Dispatch lookahead、slice 搬运、GMM2 整体滞后一轮及最终 flush。
- 没有撤销 QGMM interweave、持久 BlockMmad context 或动态 problem 参数更新。
- Arch92 共享 GMM1 改为 GM 交接；Arch35 A8W8/A4W4 的共享 UB 路径和 A8W4 的共享 GM 路径保留。
- 没有改动当前 MTE 量化 Combine 的 per-expert AIC ready 表，也没有套回 `_910b2` 旧 token-group counter 协议。
- Arch35 的 AIV1 游标补偿、26/27 事件编号、双槽 epoch 更新和共享并行保持不变；仍需设备回归确认公共头文件变更无编译回归。
- 首轮新增的共享 GMM1 前全核屏障已经移除。当前 GM 方案增加结果的 GM 写入/读取，允许共享计算与路由发送重叠；具体收益需要测量。
- 不新增 set_cond、哈希、逐元素打印或扫描整个输入的诊断操作。

## 8. 首轮验证记录（历史）

### 8.1 已完成

| 检查 | 结果与边界 |
| --- | --- |
| `git diff --check` | 通过，无补丁空白错误 |
| 主机轻量检查 | 12 项通过：6 项源码约束、6 项协议模型；不是编译执行设备代码 |
| 修改前源码反查 | 对 HEAD 跑同一组源码约束，5 项预期失败；Arch35 原补偿保留检查通过，说明检查能识别本次修复点 |
| tile 游标模型 | 2,077 次 problem 归属比较，覆盖 1/2/4/28/32 个逻辑核、零 tile、满块、尾块、初始偏移及 lag 开/关 |
| epoch 槽模型 | 穷举三核共 1,770 种读写交错，分别验证 1C1V 单槽及 1C2V 双槽互不覆盖；旧单 AIV 双写模型存在反例 |
| 共享交接模型 | 检查三对 C/V 共 720 种到达顺序，验证混合屏障前 C 不能获得共享 UB 写权限；不模拟设备流水线 |
| 修改范围 | Arch35/Arch22 专用文件、Host、Tensor 仓及构建配置未修改 |

验证脚本保留在本机临时目录，不放入算子目录，不参与产品构建。本机没有 CANN 编译器和设备运行环境，因此上述结果不能替代编译、精度、死锁压力或性能验证。

### 8.2 待设备回归

1. Arch92 A8W8/A4W4，ND/NZ、TopkWeightsPrefetch 开/关、量化/非量化 Combine 的可生成组合。
2. 非量化 Combine 优先覆盖小专家、空专家、尾 Wave，尤其 GMM1 tile 总数不是逻辑核数整数倍的情况。
3. GMM2 lag 关闭和开启的规模，确保跨 Wave 的游标一致且末尾 GMM2/Combine 正常排空。
4. 共享专家关闭和开启；开启时检查输入握手、路由数据、共享 GMM1 输出以及最终结果。
5. 多卡连续多轮运行，覆盖 epoch 累加及回绕，检查输入 count 到达校验与输出同步是否卡住。
6. Arch35 原有 A8W8/A8W4/A4W4 的编译和代表性共享/非共享运行回归。

A8W4 的 Arch92 直接 FP8/FP4 接口门禁保持原样，本次不将其标为新增可运行能力。若以后增加新的 vector 角色分工，必须同时复核事件目标、物理计数槽与 UB 交接，不能只改变角色判断函数。

## 9. 2026-09-28 清理与补充修复

### 9.1 量化 Combine 扩容前缺少旧事件排空

Arch92 稳态量化 Combine 使用一个行槽。原代码处理最后一个非空专家时调用 `PrepareFinalWaveCombineBuffers`，因唯一 AIV 同时满足 `RunsComputeVectorWork()`，会重新初始化为最多六槽。这个判断在 Arch35 表示给原来不参与 Combine 的 AIV0 建立缓冲，在 Arch92 却作用于已经发出 Combine 搬运的同一个 AIV。

如果同一次 `ProcessCombineExperts` 调用先处理了其他专家，`rowSequence` 已非零，直接换槽数就破坏事件协议。例如 hiddenDim=4096，原单槽已经发送七行：仅 slot0 有完成事件；扩成六槽后下一行按 `7 % 6` 复用 slot1，并等待从未产生的 slot1 事件，可能永久卡住。若末专家开始时序号为零，则不触发这个具体反例。

修复方式：到达最后非空专家时，先按旧 `rowBufferCount` 调用 `DrainCombineRowBuffers`，消费全部在途事件并将行序号归零，再调用 `InitWaveCombineBuffers` 扩容。保留末专家最多六槽的优化和函数结尾的排空，不新增逐行屏障。

代码：[扩容前排空与统一 Combine 调用](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:415>)、[排空和序号清零](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm2_combine.h:388>)。

Arch92 不再传 `IncludeAiv0=true`，所有专家走相同的唯一 AIV 分工。1C1V 下展开公式是 `jobIndex * 1 + 0`、`totalJobs * 1`，分工不变；Arch35 的双 AIV 最终专家分支保留。

代码：[公共分工公式](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm2_combine.h:365>)。

### 9.2 删除的无效代码

- 删除 Arch92 `allCoreCombineExpertState` 及其初始化、赋值和贯穿参数，没有实际消费者。最后非空专家索引仍用于决定扩容，改名为 `finalCombineExpertIndex`，查找函数改名为 `FindFinalCombineExpert`。
- 删除 routed Wave 收尾中恒假的 `!TopkWeightsPrefetch && !ExecutionPolicy::USE_GM_HANDOFF` UB 同步分支。Arch92 routed 策略固定 GM handoff。共享 GMM1 的 UB 同步当时保留，现已按第 10 节只在原 Arch35 策略下执行。局部序号改名 `gmm1TileSequence`，没有删除它承担的 GM tile 序号功能。
- 删除 `BlockMmadContext` 未被当前调用实例化的旧主模板实现，保留主模板声明及实际使用的 QGMM 特化。Generic GMM1/GMM2 和 Arch92 context 均选择 `Config<false>` 的 `GroupedMatmulWithScaleMx`；Arch35 旧 W4 路径直接使用自己的 BlockMmad。持久 context 和 `UpdateParamsForNextProblem` 均未移除。将来增加其他 DispatchPolicy 必须提供对应实现，不能依赖旧通用实现忽略动态参数。
- 修正 Arch92 注释中“双 AIV”“cumsum 全程常驻”和未经 A6 验证的性能数字，不改变 lag 门控或 UB 地址布局。

代码：[最终专家状态](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:154>)、[实际 GMM1 序号](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:533>)、[保留的 QGMM context](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_gmm_common.h:384>)。

### 9.3 遗留问题与不能直接删除的内容

| 项目 | 当前处理与后续要求 |
| --- | --- |
| 公共头文件仍包含旧 W4 prologue/MMAD | Arch35 仍使用它们，不是全仓死代码。本次不删除；后续若要求 Arch92 编译依赖也完全隔离，需要一起梳理 include、selector 和声明，并验证 Arch35 Wave/Layered 编译组合。 |
| 共享 GMM1 按 A8W4 类型选择旧函数 | 当前 Arch92 A8W4 门禁阻止进入。直接 FP8/FP4 接口就绪后，应按执行策略选择 Generic 路径，不能仅解除门禁。 |
| 共享 GMM2 的旧 W4 选择 | A8W4，或 A4W4 且 Activation 输出 FP8 时，仍可选择旧 `RunGmm2A8W4`。当前 Arch92 A4W4 输出 FP4，不走该分支；未来要同步修改选择条件。 |
| A4W4 SwiGLU 后续恢复 FP8 | 需要同时调整 shared/routed GMM2 类型、直接 A8W4 接口选择及 workspace 字节数，不能只改一个输出类型开关。 |
| 平台 L1 容量实机核对 | Host 已改为通过 `GetCoreMemSize(CoreMemType::L1)` 获取实际容量，并用 `OP_LOGD` 输出 `platform L1 size is <bytes> bytes`。后续须分别在 Arch35、Arch92 目标环境确认日志值符合芯片规格，并核对生成布局的最终地址不超过该容量。代码见 [L1 查询与日志](../mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp#L2929)。 |
| 编译、实际事件和精度验证 | 本机无 CANN/设备。需验证 ND/NZ、共享开关、量化 Combine、多专家同 Wave 尾部、lag 开关及 Arch35 回归，不能将主机模型通过视为设备已跑通。 |

相关代码：[旧 W4 头文件依赖](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_gmm_common.h:22>)、[共享 GMM1 选择](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:545>)、[共享 GMM2 选择](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:617>)。

### 9.4 清理完成时的验证与范围（历史）

- 主机检查共 19 项通过：10 项源码约束、9 项协议模型；包括更新后的函数声明/定义/调用参数数量和实际 context 特化检查。
- Combine 模型覆盖 5,166 种情况：旧槽数为 1，新槽数为 1～6，旧段 0～40 行、末段 0～20 行；检查无缺失事件等待、无重复 set，并确认最终排空。旧七行扩六槽的实现可复现缺失事件反例。
- 前轮 2,077 次 tile 归属比较、1,770 种 epoch 交错及 720 种共享交接到达顺序检查仍通过。
- 累计修改六个 kernel 头文件（前轮五个，加本轮 `mega_moe_gmm_common.h`）及本文档。未修改 Arch35/Arch22 专用文件、Host、Tensor 仓或构建配置，未增加打点。
- 以上是源码检查和协议抽象模型，不是 C++ 编译或设备执行；性能开销尚未测量。

## 10. 共享 GMM1 统一采用 Arch92 GM 交接

### 10.1 目标和实际修改

按 Arch35 A8W4 的角色顺序保留路由 AIV：Arch92 唯一 AIV 先发送路由及跨卡同步，再执行共享 Activation；AIC 使用现有 Tensor Generic BlockMmad，提前将共享 GMM1 结果写 GM。无需引入 W4 到 W8 prologue，也无需另写矩阵计算实现。

`RunSharedExpertGmm1Activation` 将 `ExecutionPolicy::USE_GM_HANDOFF` 显式传给 `RunGmm1Generic`。Arch92 为 true，选择 `Gmm1AicMmadToGmGeneric` 和 GM 读取后处理；Arch35 为 false，等价于原默认实参。`IsShared=true`、共享 `TopkWeightsPrefetch=false`、布局参数及 BlockMmad context 均保留。

代码：[共享调用](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:545>)、[已有 Generic GM 执行分支](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:479>)。

`ProcessSharedExpertGmm1` 在旧 A8W4 或执行策略要求 GM 时，绑定共享 Ready/ACK 对象并建立 `gmm1OutGlobal`。输出仍为每个专家独立的 `[bs, gmm1OutputDim]` BF16 区域。地址准备函数的布尔参数由 `UseA8W4Gmm1` 改名为 `UseGmOutput`，其地址计算不变。工作空间原先已经为所有共享模式预留 `sharedExpertGmm1Out`，因此本次无需扩大 Host workspace。

代码：[地址与同步对象绑定](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:596>)、[GM 输出地址](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm1_activation.h:848>)、[现有共享输出空间](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_workspace.h:361>)。

### 10.2 同步与 UB 使用顺序

共享硬件事件对象的目标取 `VECTOR_CORE_COUNT_PER_CUBE - 1`：Arch35 仍为 AIV1，AIC 使用 28/29；Arch92 为唯一 AIV，AIC 使用 12/13。AIV 使用本地 12/13。保持共享事件与 MoE GMM1/GMM2 独立，并保留 15 tile 的额度限制。唯一 AIV 的路由/跨卡阶段不等待共享 AIC，因此 AIC 在额度耗尽时等待 ACK 不会阻止 AIV 完成路由后开始消费。

代码：[事件目标](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:825>)、[Ready/ACK 与额度](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_gmm_epilogue_sync.h:138>)。

Arch92 不再执行 `Gmm1UbActivationSync::EndSync`：GM 路径使用的序号非零，但不会发送 UB ping-pong ACK，不能据此等待 UB 事件。共享 GM ACK 继续由原主流程末尾的 `sharedGmm1ActivationSync.EndSync()` 排空。Activation 之后保护 UB 复用的 `MTE3_MTE2` 等待仍保留。

代码：[按执行策略选择收尾](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:614>)、[共享 ACK 最终排空](</D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:871>)。

### 10.3 Arch35 保持的执行结果

| 场景 | 当前公共代码在 Arch35 的选择 |
| --- | --- |
| 共享 A8W8/A4W4 | `USE_GM_HANDOFF=false`，仍由 AIC 写 AIV0 UB、AIV0 做 Activation；AIV1 发送路由 |
| 共享 A8W4 | 仍调用旧 `RunGmm1A8W4`，AIV0 展开权重，AIC 写 GM，AIV1 发送后做 Activation；通知目标仍为 AIV1 |
| 无共享专家 | 原 `sharedExpertNum_` 条件跳过共享阶段；输入准备、发送、跨卡同步和进入 MoE Wave 的顺序不变 |

源代码回归检查将本次增加的策略实参按 Arch35 默认 false 展开，比较共享调用、共享执行函数和 `ProcessWave` 与 HEAD 基线，除注释和等价常量外一致；地址准备头文件除布尔参数改名外一致。该检查不代替 CANN 编译。

### 10.4 验证和待运行项

- 共 25 项主机检查通过。替换此前两项依赖共享 UB 屏障的历史测试，保留其余 17 项检查，新增 8 项共享 GM 检查。
- 异步模型覆盖 81,908 个可达状态，将 AIC 提交、FIX 写完并发布 Ready、AIV 消费、MTE3 ACK 和最终 ACK 排空分开推进，覆盖 0/1/2/14/15/16/31 tile、路由延迟及额度边界。错误 AIV 目标和错误 UB 收尾可由反例检测。
- 原 Combine、分核游标和 epoch 检查继续通过；`git diff --check` 通过。
- 脚本位于本机临时目录 `verify_megamoe_shared_gm_handoff_20260928.py`，未放入算子目录。
- 尚无 CANN 编译或设备结果。设备回归需覆盖共享数 0/1/多专家、两种量化及 ND/NZ、每核无 tile/尾 tile/超过 15 tile、重复 launch、跨卡路由慢于共享 GMM1 的场景，并回归 Arch35 原路径。
- Arch92 A8W4 直接混合接口门禁保留；本次未修改共享 GMM2。未来开放 A8W4 和恢复 A4W4 的 FP8 Activation 仍按第 9.3 节处理。
