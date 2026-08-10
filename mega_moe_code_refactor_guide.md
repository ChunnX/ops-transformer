# MegaMoe 代码整改说明

这份文档面向已经熟悉 MoE、Dispatch、GMM、设备侧并行和通信同步，但第一次阅读 MegaMoe 实现的开发者。文档不重复介绍 MoE 的基础算法，而是聚焦本算子的代码组织、阶段边界、核心分工、模板差异和函数调用关系。

## 1. 阅读前提与代码对象

阅读本文前，可以把下面的流程视为已知背景；这里列出它只是为了标记本次整改涉及的代码边界：

```text
输入 token
  -> 量化并准备路由信息
  -> 把 token 发送给对应的专家
  -> 专家 GMM1 + SwiGLU
  -> 专家 GMM2 + Combine
  -> 把不同专家的结果重新合并回原 token 顺序
```

本次整改重点不在改变上述算法，而在于让每个阶段在代码中有明确的入口和责任边界。

当前实现中需要特别关注两个代码事实：

- **MoE 专家（路由专家）**：由 `topKIds` 决定 token 去哪个专家，是主流程中必定处理的一类专家。某个专家没有 token 时只跳过无效工作，不代表 MoE 专家阶段不存在。
- **共享专家**：共享专家是可选能力，由 `sharedExpertNum` 决定。共享专家的权重地址、输入准备和 GMM 阶段都必须在数量大于零时使用。

代码中的设备侧角色可以这样对应：

- **AIC**：主要执行 Cube/矩阵乘法，也就是 GMM 的 MMAD。
- **AIV**：主要执行向量计算、数据搬运、通信和部分后处理。
- **AIV0/AIV1**：表示两个逻辑向量角色，例如一侧负责 Prologue 或 Dispatch，另一侧负责 Epilogue 或 Combine。它们不是固定的物理核心编号，具体由调度映射决定。

## 2. 这次整改解决了什么问题

这次修改主要有两条主线。

### 2.1 把主流程拆成按设备角色执行的阶段函数

以前主流程函数中同时混合了很多内容：

- 判断当前是 AIC 还是 AIV；
- 计算当前核心应该处理的任务范围；
- 申请和切分 UB；
- 计算当前专家的地址；
- 执行通信、GMM、SwiGLU 和 Combine；
- 处理不同核心之间的同步。

这样阅读一个函数时，很难知道某一段代码是在做“阶段调度”，还是在做“实际计算”。

整改后，主流程中的阶段函数会明确知道当前任务由哪一类核心执行。常见形式包括：

```cpp
if constexpr (g_coreType == AIC) {
    // AIC 路径：执行矩阵乘法
} else {
    // AIV 路径：执行通信、向量后处理或数据搬运
}
```

阶段函数还会结合 `GetBlockIdx()`、`GetBlockNum()`、`GetSubBlockIdx()` 和各阶段的 job 配置，决定当前逻辑任务处理哪一段 token、哪一个专家或哪一批 tile。

这里的“单核函数”更准确地说是“逻辑核函数”：函数描述一个可以被调度的逻辑核任务，但不绑定某个固定的物理核心。物理核心到逻辑任务的映射由调度配置和 job 范围决定，因而可以灵活调整。例如：

- 物理 AIV 编号 0、3、4、8 可以被映射到逻辑任务 A；
- 物理 AIV 编号 1、2、5、6、7 可以被映射到逻辑任务 B；
- A 和 B 分别执行不同的阶段函数或不同的阶段角色。

当前实现采用全核运行：参与 kernel 的物理核心都会进入执行流程，再根据自身的 block/job 信息映射到对应的逻辑任务。代码关注的是“当前逻辑任务处理什么”，而不是把逻辑写死在某几个物理核心编号上。

因此，AIC/AIV 表示设备侧执行角色，`GetBlockIdx()` 等信息表示当前逻辑任务的位置，二者不应理解成固定的物理核编号绑定。

这样做的好处是：

1. 核心分工从主流程和阶段入口中就能看出来。
2. AIC 的矩阵计算和 AIV 的通信、Prologue、Epilogue 不再混在一个大函数里。
3. 普通模板和 Wave 模板可以复用相同的阶段函数，模板差异保留在各自的外层调度逻辑中。
4. 共享专家可以复用 MoE 专家的 GMM1/GMM2 实现，而不需要再维护一套完全重复的矩阵计算代码。

### 2.2 按处理阶段重新组织目录

整改前，代码主要按历史实现来源分布在 `aiv_comm`、`aiv_compute`、`impl`、`group_matmul` 等多个目录或文件中。文件名体现的是“实现从哪里来的”，不一定能直接说明“算子当前处于哪个处理阶段”。

整改后，目录更接近算子的执行顺序：

```text
arch35/
├── mega_moe.h                 普通模板主流程
├── mega_moe_wave.h            Wave 模板主流程
├── mega_moe_layered.h         Layered 模板主流程，保持独立
├── stage/                     算子的阶段函数
├── common/                    公共类型、常量、工具和 workspace 定义
└── blaze/                     更底层的矩阵、Prologue、Epilogue 和调度实现
```

这棵目录树的阅读重点是：kernel 模板的核心代码仍然位于 `arch35` 主目录下，主要入口是 `mega_moe.h`、`mega_moe_wave.h` 和 `mega_moe_layered.h`；这些模板使用的阶段模块、公共定义和底层计算实现，再分别放入 `stage/`、`common/` 和 `blaze/` 等子目录。也就是说，主目录负责“把算子跑起来”，子目录负责“提供主模板调用的具体能力”。

因此，第一次阅读这个算子时，应先从主目录下的模板 `Process()` 入口建立全局视图，再进入对应子目录追踪某一个阶段的实现，而不是把 `stage` 误认为独立的 kernel 入口。

## 3. 当前目录中各层的职责

### 3.1 主流程层：`mega_moe.h` 和 `mega_moe_wave.h`

这两个文件负责把所有阶段串起来，但不应该承载每个阶段的具体细节。

普通模板的主入口是：

```text
MegaMoe::Process()
```

Wave 模板的主入口是：

```text
MegaMoeWave::Process()
```

它们主要负责：

- 初始化各阶段需要的配置和 UB 视图；
- 按正确顺序调用各阶段；
- 进行跨 rank、AIC/AIV 和 wave 之间的必要同步；
- 在共享专家存在时调用共享专家流程；
- 维护外层调度所需要的游标和任务状态。

主流程中的 `InitInputPrepareConfigs`、`InitTokenDispatchConfig`、`InitGmm1Config`、`InitGmm2CombineConfigs` 和 `InitTokenUnpermuteConfig`，负责建立阶段配置；`DispatchBuffInit`、`SendAndQuantBuffInit` 和 `InitCombineBuffers` 负责建立 UB/GM 使用的视图。

这些初始化函数解决的是“资源在哪里、当前核心负责什么”的问题，不是阶段本身的计算逻辑。

### 3.2 阶段层：`stage/`

`stage` 中的函数是主流程直接调用的阶段函数。它们负责一个完整阶段的调度、分核和阶段内同步，并调用更底层的基础实现完成实际工作。

| 文件 | 阶段 | 代码作用和阅读重点 |
| --- | --- | --- |
| `stage/mega_moe_token_quant.h` | Token Quant | 把本卡输入 token 转成通信和专家计算需要的量化布局，必要时把 top-k 权重一起准备好。入口是 `QuantizeLocalTokens`。 |
| `stage/mega_moe_send_mask.h` | Send Mask | 根据路由关系生成每个专家对应的 mask，并把 mask 发送到目标 rank。入口是 `GatherAndSendExpertMasks`。 |
| `stage/mega_moe_workspace_reset.h` | Workspace Reset | 清理 Dispatch、GMM1 等阶段会复用的 flag/counter workspace，避免上一轮状态影响当前轮。入口是 `ResetDispatchWorkspace`。 |
| `stage/mega_moe_shared_expert_input.h` | Shared Expert Input | 当存在共享专家时，从公共量化输入中整理共享专家使用的输入和 scale。入口是 `PrepareSharedExpertInput`。 |
| `stage/mega_moe_token_dispatch.h` | Token Dispatch | 统计当前专家收到多少 token，并根据远端 mask 把 token 和 metadata 搬到专家计算所需的位置。入口是 `RunMoeExpertDispatchStage`，内部会调用 `ComputeExpertTokenCountAndNotify` 和 `DispatchExpertTokens`。 |
| `stage/mega_moe_gmm1_swiglu.h` | GMM1 + SwiGLU | 执行第一段 Group Matmul，并在 AIV 上执行 SwiGLU、量化或结果后处理。入口包括 `RunMoeExpertGmm1SwigluStage` 和 `RunSharedExpertGmm1SwigluStage`。 |
| `stage/mega_moe_gmm2_combine.h` | GMM2 + Combine | 执行第二段 Group Matmul，并把不同 rank/专家产生的结果按当前 Combine 模式汇合。入口包括 `RunMoeExpertGmm2Stage`、`RunSharedExpertGmm2Stage` 和 `ScheduleQuantizedMoeExpertCombine`。 |
| `stage/mega_moe_unpermute.h` | Token Unpermute | 将专家计算后的结果按原始 token 顺序写回输出。入口是 `UnpermuteTokens`。 |

这里把 GMM1 和 SwiGLU 放在同一个阶段，是因为 GMM1 输出马上要被 SwiGLU 或后处理消费；把 GMM2 和 Combine 放在同一个阶段，是因为 GMM2 输出需要按照相应的同步计数和通信结果立即完成 Combine。

这并不表示 GMM1 和 GMM2 是同一个动作。二者仍然是两个独立的矩阵计算阶段，只是各自与紧邻、强耦合的后处理放在同一个阶段文件中。

### 3.3 公共基础层：`common/`

`common` 不表示某一个处理阶段，而是多个阶段都会使用的基础内容：

- `mega_moe_constants.h`：多个文件共享的常量。只被单个文件使用的常量仍放在使用位置附近。
- `mega_moe_types.h`：公共结构体和类型，例如 `Params`、`GMMAddrInfo`、`MoeStageCommonConfig`、`BlockJobContext` 和专家权重地址列表。
- `mega_moe_utils.h`：公共小函数，例如任务范围切分、workspace reset、rank/地址相关的通用辅助逻辑。
- `mega_moe_workspace.h`：workspace 的布局和地址计算，负责把 workspace GM 划分成不同功能区域。
- `mega_moe_gmm_common.h`：GMM 阶段共享的 scheduler、layout、配置和通用 GMM 辅助逻辑。
- `mega_moe_mxfp8_utils.h`：MXFP8 量化和反量化相关的通用函数。

例如，阶段函数不需要自己重新定义 rank、token 数和 hidden size，而是从 `MoeStageCommonConfig` 中读取公共信息；每个阶段再通过自己的 `*Config` 补充阶段专属参数。

### 3.4 基础计算层：`blaze/`

`blaze` 放的是更底层的计算原语和通用组件：

- `blaze/gemm/block/`：Block MMAD、Block Scheduler 等矩阵计算基础实现。
- `blaze/prologue/`：矩阵计算前的权重或输入准备。
- `blaze/epilogue/`：矩阵计算后的 SwiGLU、量化和输出写回。
- `blaze/gemm/utils/`：GEMM 相关的基础工具。

可以用下面的关系理解 `stage` 和 `blaze`：

```text
stage 决定：什么时候做、哪个核心做、处理哪个专家和哪个 tile
blaze 决定：一个 MMAD、一个 Prologue 或一个 Epilogue 具体怎样做
```

因此，`stage/mega_moe_gmm1_swiglu.h` 中的 `Gmm1AicMmadGeneric` 负责把 GMM1 的 AIC 工作组织起来，而更底层的 `BlockMmad` 负责执行具体的矩阵乘法。

## 4. 一次普通模板执行的调用链

下面是 `MegaMoe::Process()` 的简化调用链。省略了模板参数和部分同步细节，但保留了理解流程所需的层级。

```text
MegaMoe::Process
├── SendAndQuantBuffInit                         建立输入准备阶段的 buffer 视图
├── QuantizeLocalTokens                          本卡 token 量化
├── GatherAndSendExpertMasks                    生成并发送专家 mask
├── ResetDispatchWorkspace                      清理 flag/counter workspace
├── [sharedExpertNum > 0]
│   └── PrepareSharedExpertInput                 准备共享专家输入
├── SyncAll / CrossRankSyncInWorldSize           等待输入准备和跨卡通信状态
├── [sharedExpertNum > 0]
│   └── ProcessSharedExpertGmm1
│       └── RunSharedExpertGmm1SwigluStage       复用 GMM1 + SwiGLU 阶段
├── ProcessMoeExpertStages
│   ├── DispatchBuffInit
│   ├── RunMoeExpertDispatchStage                处理当前 MoE 专家 Dispatch
│   │   ├── ComputeExpertTokenCountAndNotify     统计 token 数并通知后续阶段
│   │   └── DispatchExpertTokens                 搬运 token 和 metadata
│   ├── RunMoeExpertGmm1SwigluStage              GMM1 + SwiGLU
│   │   ├── WaitAndUpdateGmm1GroupParams         准备当前专家的 GMM1 参数
│   │   ├── UpdateGmm1GlobalBuffer                建立当前专家地址
│   │   └── RunGmm1SwigluByMode                  按数据类型选择 GMM1 实现
│   ├── FinishMoeGmm1SwigluStage                 完成 GMM1 阶段同步
│   ├── PrepareMoeExpertGmm2Stage                准备当前专家的 GMM2 参数和地址
│   ├── RunMoeExpertGmm2Stage                   GMM2
│   │   └── RunGmm2ByMode                       按数据类型选择 GMM2 实现
│   └── ScheduleQuantizedMoeExpertCombine        量化 Combine 路径的调度
├── [sharedExpertNum > 0]
│   └── ProcessSharedExpertGmm2                  复用 GMM2，但共享专家不做 routed Combine
└── UnpermuteTokens                              写回原 token 顺序
```

普通模板一般按照本卡专家顺序处理。每处理一个专家，会先根据 token 数更新 GMM 参数和地址，再调用 GMM1/GMM2 阶段函数。某个专家如果没有 token，阶段函数会返回或跳过当前专家，这是“没有工作可做”，不是“没有 MoE 专家”。

## 5. 普通模板和 Wave 模板的函数复用

本节只说明代码复用关系，不展开描述两个模板各自的执行策略。

两个模板分别保留自己的主流程和外层调度循环；外层循环负责组织当前模板的调用顺序、参数和任务范围。阶段函数则抽到 `stage/` 中，由两个模板共同调用。

例如，两个模板都会调用下面这组 stage 入口：

- `RunMoeExpertDispatchStage`
- `RunMoeExpertGmm1SwigluStage`
- `PrepareMoeExpertGmm2Stage`
- `RunMoeExpertGmm2Stage`

因此，同一个 Dispatch、GMM1、GMM2 阶段不需要在普通模板和 Wave 模板中各维护一份实现。两个模板只是在外层循环中传入各自的 job、专家索引、地址和模板参数，stage 函数本身负责完成当前调用对应的逻辑核任务。

共享专家也复用同一组 GMM stage 文件：主流程调用 `RunSharedExpertGmm1SwigluStage` 和 `RunSharedExpertGmm2Stage`，通过共享权重地址、共享配置和 `IsShared` 等模板参数选择共享专家路径。

### 5.1 common 函数如何复用

`common` 中的内容不负责区分普通模板或 Wave 模板，而是提供双方共同使用的类型、配置和基础函数。例如：

1. `MoeStageCommonConfig` 在两个模板中分别按相同结构构造，并嵌入 Dispatch、GMM1、GMM2 和 Unpermute 各自的 `*Config`。这些 stage 可以统一读取 `rankId`、`worldSize`、`tokenNum`、`topK` 和 `tokenHiddenDim`。
2. `ResetWorkspaceRegion` 位于 `common/mega_moe_utils.h`，两个模板的 `ResetDispatchWorkspace` 都通过它按 job 范围清理 workspace。上层只提供目标区域、元素数量和 job 配置，公共函数复用具体的分段和批量搬运逻辑。

这种复用方式把模板差异留在外层，把阶段动作和公共基础规则集中到可共享的位置。

## 6. 共享专家如何复用 GMM 阶段

共享专家没有 routed Combine，但它仍然需要 GMM1 和 GMM2：

```text
共享专家输入
  -> RunSharedExpertGmm1SwigluStage
  -> 共享专家 GMM1 + SwiGLU
  -> RunSharedExpertGmm2Stage
  -> 共享专家 GMM2
```

共享专家调用的阶段函数与 MoE 专家使用同一套底层 GMM 实现，通过以下信息区分路径：

- 传入共享专家权重地址列表或 MoE 专家权重地址列表；
- 使用 `IsShared` 等模板参数选择共享专家地址和输出布局；
- 共享专家的 `CombineMode` 为 `COMBINE_NO_QUANT`，不走 routed MoE 的 Combine 调度；
- 主流程和资源初始化处都用 `sharedExpertNum_ > 0` 判断共享专家是否存在。

这样既避免了共享专家重复实现一套 GMM，也避免在共享专家不存在时读取空的共享权重地址。

## 7. 重要公共结构体怎么理解

### `Params`

`Params` 是一次算子执行的公共参数集合，保存输入、权重、输出、workspace、远端通信内存和 tiling 数据指针。阶段函数通过 `const Params &` 读取它们，不需要把每个 GM 地址逐个传遍调用链。

### `MoeStageCommonConfig`

这是多个阶段都会高频使用的一小组公共值，包括：

- 当前 rank 和 world size；
- 每个 rank 的 MoE 专家数；
- 共享专家数；
- token 数、top-k、hidden size；
- GMM1 输出维度。

它不是一个“所有参数的大杂烩”，而是为了减少各阶段重复传递同一批公共信息。

### 各阶段 `*Config`

例如 `TokenDispatchConfig`、`Gmm1SwigluConfig`、`Gmm2Config` 和 `TokenUnpermuteConfig`。它们在公共配置基础上补充某一个阶段需要的 job 范围、flag 槽位、tile 数、buffer 配置等调度信息。

### `Scratch`、`WorkSet` 和 `GMMAddrInfo`

- `Scratch`：当前阶段在 UB 中使用的 tensor 视图和临时空间。
- `WorkSet`：GMM 执行时已经建立好的 scheduler、GM tensor、scale、bias、输出 tensor 等工作集合。
- `GMMAddrInfo`：当前专家的实际 GM 地址和同步 flag 地址。切换专家时，先更新它，再执行对应 GMM 阶段。

这三个对象的职责不同：`Config` 说明怎么调度，`Scratch` 说明临时空间在哪里，`WorkSet` 说明当前一次 GMM 计算使用哪些 tensor，`GMMAddrInfo` 说明当前专家的数据和同步地址在哪里。

## 8. 首次接触本算子时的推荐阅读顺序

对于第一次接触 MegaMoe 的开发者，建议按下面的顺序阅读，不要一开始就从几百行的 GMM 模板实现读起：

1. 先读 `mega_moe.h` 的 `MegaMoe::Process()`，标出普通模板调用的 stage 入口。
2. 再读 `mega_moe_wave.h` 的 `MegaMoeWave::Process()`，对照两个模板如何调用同一组 stage 函数。
3. 按主流程顺序阅读 `stage/mega_moe_token_quant.h`、`mega_moe_send_mask.h`、`mega_moe_workspace_reset.h` 和 `mega_moe_token_dispatch.h`。
4. 再读 `stage/mega_moe_gmm1_swiglu.h` 和 `stage/mega_moe_gmm2_combine.h`，重点看 AIC/AIV 入口和共享专家复用入口。
5. 最后读 `stage/mega_moe_unpermute.h`，了解结果如何回到原 token 顺序。
6. 需要深入底层时，再进入 `common/` 查看数据结构和地址布局，进入 `blaze/` 查看 Block MMAD、Prologue 和 Epilogue 的具体实现。

## 9. 这次整改后的几个不变量

阅读或继续修改代码时，建议保持以下约束：

- MoE 专家主流程始终存在，不要增加“是否运行 MoE 专家”的判断；没有 token 的专家只需要跳过无效工作。
- 共享专家是可选的，所有共享权重、共享输入和共享阶段都必须在 `sharedExpertNum > 0` 时使用。
- GMM1 和 GMM2 是两个独立的计算阶段；当前分别与 SwiGLU、Combine 组合成两个阶段文件。
- 普通模板和 Wave 模板应复用相同的阶段函数；模板各自的外层逻辑只负责组织调用所需的参数和任务范围。
- `stage` 负责阶段级调度和核心角色分工，`common` 负责公共定义，`blaze` 负责底层计算原语。
- Layered 模板保持自己的入口和兼容逻辑，不要因为普通/Wave 模板的目录整改而改变其既有行为。

如果后续继续拆分代码，可以先问自己三个问题：

1. 这段代码是在决定“谁、何时、处理什么”吗？如果是，优先放在对应的 `stage`。
2. 这段代码是不是多个阶段都会用到的类型、常量、地址计算或小工具？如果是，考虑放在 `common`。
3. 这段代码是不是只实现一次 Block MMAD、Prologue 或 Epilogue 的底层动作？如果是，优先放在 `blaze`。

## 10. 函数复用总结

普通模板和 Wave 模板的复用关系可以概括为：

```text
MegaMoe::Process()       ─┐
                          ├─> stage/ 中的公共阶段入口
MegaMoeWave::Process()   ─┘
```

模板主流程保留在 `arch35` 根目录，stage 函数只实现一份；两个模板通过各自的外层循环、配置对象和模板参数调用它。`common/` 中的公共类型和工具函数再被这些 stage 以及两个模板共同使用。
