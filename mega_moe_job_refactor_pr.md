# MegaMoe 逻辑 Job 模块化重构 PR 说明

## PR 概述

本 PR 对 Ascend 950 架构下 MegaMoe Kernel 的 Dispatch 前处理、Token Dispatch、
GMM1 和 SwiGLU 流程进行模块化重构。重构后的函数不依赖 `MegaMoe` 类实例，
调用方通过显式的 Context、Args、Scratch 和 State 提供所需信息。

这里的“单核模块”表示每个物理执行核只处理调用方映射给它的一个逻辑 AIV job
或逻辑 MIX block job。MegaMoe 算子本身没有改成单核算子，仍由
`MegaMoe::Process` 组织完整流程。

## 目标

- 抽出可被其他模板或算子复用的计算、通信和单 block 调度模块。
- 由调用方决定参与模块的物理 AIV 或物理 MIX block 数量。
- 由调用方自主完成物理核心到逻辑 job 的映射。
- 逻辑 `jobIndex` 从 0 开始连续，但参与的物理核心可以不连续，也可以不从 0 开始。
- 模块内部不读取 `MegaMoe` 成员变量，只使用显式传入的数据契约。
- 保持现有 A8W8、A8W4、A4W4/NZ、TopK weight prefetch、共享专家、
  大 BS Combine 和 Layered 路径的计算与同步关系。

当前 MegaMoe 仍然全核运行，并采用物理编号到逻辑编号的一一映射。这只是
MegaMoe 当前的调用策略，不是模块接口的限制。

## 主流程

`MegaMoe::Process` 的阶段顺序保持不变：

```text
SendAndQuantBuffInit
  -> QuantizeLocalTokens
  -> GatherAndSendExpertMasks
  -> ResetDispatchWorkspace
  -> PrepareSharedExpertInput（可选）
  -> ControlSharedExpertGmm1SwigluPipeline（可选）
  -> CrossRankSyncInWorldSize
  -> ControlDispatchGmm1SwigluPipeline
  -> ExpertTokenNumCopyOut
  -> 原有 GMM2 / Combine
  -> 原有共享专家 GMM2 / Unpermute
```

本 PR 没有把 GMM2、Combine、Cross-rank sync 或 Unpermute 改成新的模块。

## 逻辑 Job

### AIV Job

纯 AIV 阶段使用：

```cpp
struct AivJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};
```

`jobIndex` 表示当前物理 AIV 被调用方映射成的逻辑编号，`totalJobs` 表示本阶段
参与计算的逻辑 AIV 数。`TilingByJobContext` 根据这两个值完成一维任务切分。

例如只选择物理 AIV `{2, 5, 9, 14}` 时，调用方可以映射为：

```text
physical AIV 2  -> jobIndex 0, totalJobs 4
physical AIV 5  -> jobIndex 1, totalJobs 4
physical AIV 9  -> jobIndex 2, totalJobs 4
physical AIV 14 -> jobIndex 3, totalJobs 4
```

模块只观察逻辑 job，不在内部推导物理 AIV ID。

### MIX Block Job

Dispatch/GMM1/SwiGLU 使用：

```cpp
struct BlockJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};
```

一个 Ascend 950 MIX block 由 1 个 AIC 和 2 个 AIV 组成。调用
`ControlDispatchGmm1SwigluPipeline` 时，同一物理 MIX block 内的 AIC、AIV0 和
AIV1 必须使用相同的 `BlockJobContext`。否则三者处理的 tile 会错位，数据链和
同步链都会断开。

## BlockWorkspaceContext

```cpp
struct BlockWorkspaceContext {
    uint32_t blockIdx;
    uint32_t blockNum;
};
```

`BlockJobContext` 用于计算分工，`BlockWorkspaceContext` 用于 GM workspace 寻址，
两者不能互相替代。

- `blockIdx` 是当前物理 MIX block 对应的固定 workspace 槽位。
- `blockNum` 是 workspace 在物理 block 维度预留的总槽位数，也是 expert 之间的 stride。
- `expertRevTokenNumsPtr` 和 `sendCntFlagPtr` 等按
  `[expert][physicalBlock]` 保存的数据使用
  `expertIdx * blockNum * stride + blockIdx * stride` 寻址。

例如只启用物理 block `{3, 7, 11}`，并将它们映射成逻辑 job `{0, 1, 2}`：

```cpp
BlockJobContext job{1, 3};
BlockWorkspaceContext workspace{7, physicalBlockCapacity};
```

物理 block 7 负责逻辑 job 1，但仍读写物理 workspace 槽位 7。这样后续阶段即使
采用不同的逻辑 job 数，也不会因为逻辑编号压缩而使 GM 数据错位。

当前 MegaMoe 的 GMM1 和 GMM2 都全 block 运行，因此两种编号一一对应。拆开结构体
是为了让复用接口能够表达部分核和非连续物理核场景，而不是改变当前 MegaMoe 的寻址。

## 参数分层

模块参数按用途和生命周期分为四类：

| 类型 | 共同特征 | 典型内容 |
| --- | --- | --- |
| `Context` | 一次模块调用期间稳定、只读，描述任务是什么以及如何切分 | 逻辑 job、rank/world size、shape、模式、tile 和 buffer 配置 |
| `Args` | 调用方拥有的外部资源，不属于模块临时状态 | GM 地址、rank window、weight、workspace 和 flag 地址 |
| `Scratch` | 模块执行期间使用的临时视图或工作空间 | `GlobalTensor`、`LocalTensor`、UB ring 基址和累计临时值 |
| `State` | 随专家或 tile 流水推进而变化，并需要跨调用保存 | `startBlockIdx`、同步状态和 `gmTileSequence` |

Context 不保存 GM 地址或 UB Tensor；Args 不负责描述任务切分；Scratch 不拥有外部
内存生命周期；State 不放稳定配置。MegaMoe 在 `Init` 或 buffer 初始化阶段构造这些
对象，`Process` 只按引用传给模块，不复制 Tensor 数据。

## 文件组织

目录归类依据是执行单元和是否涉及跨卡通信。

### `aiv_compute`

该目录放不依赖 AIC 协作的 AIV 计算或本地数据处理。它可以写本阶段自己的完成标志，
但不负责跨卡 window 通信。

| 文件 | 主要函数 | 作用 |
| --- | --- | --- |
| `aiv_compute/mega_moe_quant_process.h` | `QuantizeLocalTokens` | 按 AIV job 量化本卡 Token，并可附带 TopK weight prefetch |
| `aiv_compute/mega_moe_shared_expert_prepare.h` | `PrepareSharedExpertInput` | 将交错保存的共享专家 data/scale 拆成连续输入 |
| `aiv_compute/mega_moe_expert_token_count.h` | `ComputeExpertTokenCountAndNotify` | 汇总各 Rank 的当前专家 Token 数，写 cumsum/count，并发布 SendCnt 完成标志 |

### `aiv_comm`

该目录放不依赖 AIC 协作、但涉及跨卡 window 或 Dispatch 通信数据搬运的 AIV 模块。
函数内部可以包含必要的 Vector 计算。

| 文件 | 主要函数 | 作用 |
| --- | --- | --- |
| `aiv_comm/mega_moe_send_mask.h` | `GatherAndSendExpertMasks` | 生成各专家 bitmap/count，并写入目标 Rank window |
| `aiv_comm/mega_moe_token_dispatch.h` | `DispatchExpertTokens` | 扫描远端 mask，搬运 Token/Scale/MetaInfo，并按 GMM tile 发布 Dispatch wave 计数 |

### `arch35` 根目录

MegaMoe 专属的公共契约、GMM 适配、跨 AIC/AIV 的流水控制器放在 `arch35` 根目录。
这些文件不是 Blaze 底层通用实现，不再放入 `blaze` 目录。

| 文件 | 主要内容 |
| --- | --- |
| `mega_moe_job_context.h` | AIV job、拓扑、Token shape 和 `TilingByJobContext` |
| `mega_moe_workspace_reset.h` | `ResetDispatchWorkspace` 及 GMM2/共享专家 counter reset |
| `mega_moe_gmm1_swiglu.h` | GMM1 shape、地址更新和模式分派 |
| `mega_moe_dispatch_gmm1_swiglu.h` | SendCnt、Dispatch、GMM1、SwiGLU 的模板级组合流水 |
| `mega_moe_shared_expert_gmm1_swiglu.h` | 共享专家 GMM1/SwiGLU 组合流水 |
| `mega_moe_group_matmul_compute.h` | Generic/A8W4 的单 tile AIC MMAD |
| `mega_moe_group_matmul_prologue.h` | A8W4 权重展开调度 |
| `mega_moe_group_matmul_epilogue.h` | Generic/A8W4 的 SwiGLU tile 包装及原有 GMM2 AIV 后处理 |
| `mega_moe_group_matmul_pipeline.h` | GMM1/SwiGLU 的 AIC/AIV tile 流水和 GMM1/GMM2 分派 |
| `block_epilogue_swiglu_mx_quant.h` | 既有 SwiGLU + MX Quant 算子本体 |

原兼容聚合头 `mega_moe_single_core.h` 已删除。`mega_moe.h` 直接包含所需的功能头，
避免一个与实际职责不对应的统一聚合层。

## 主要模块

### `QuantizeLocalTokens`

原型函数为 `MegaMoe::QuantProcessInRank`。该函数按 `AivJobContext` 切分本卡 Token，
使用双 buffer 完成输入搬运、MX 量化和可选的 TopK weight 搬运。

非 prefetch 路径的 `MTE2_V` 同步放在 `LoadTopkWeightsToUb` 调用侧，紧跟输入
`DataCopyPad`，并位于第一次 Vector 消费之前。这样同步所保护的数据和调用位置一致，
同时不改变原有执行顺序。

### `GatherAndSendExpertMasks`

原型函数为 `MegaMoe::SendMaskCal`。函数按 AIV job 分配全局专家，逐 route batch：

1. 搬入 TopK expert ID。
2. 为当前专家生成 bitmap。
3. 累计该专家的 Token count。
4. 将 `[mask | count]` 写入目标 Rank window。

`GatherAndSendExpertMaskBatch` 只抽取单个 batch 的内部实现，ring event 的
Set/Wait 顺序保持不变。

### `ResetDispatchWorkspace`

原型函数为 `MegaMoe::ResetFlagList`。它按 AIV job 分批清零：

- SwiGLU 到 GMM2 flag；
- 大 BS Combine sync counter；
- 共享专家 GMM2 tile counter；
- TopK weight prefetch 使用的 GMM1 tile status。

### `PrepareSharedExpertInput`

原型函数为 `MegaMoe::SharedExpertCopyInput`。它读取本卡量化后交错保存的
Token data 和 scale，使用双 buffer 拆分后写入共享专家连续输入区。该过程不访问
rank window，也不依赖 AIC flag。

### `ComputeExpertTokenCountAndNotify`

原型函数为 `MegaMoe::SendCntCal`。它读取每个源 Rank 在 `[mask | count]` 槽位中
保存的 count，计算：

- 当前专家收到的 Token 总数；
- 按 expert/rank 排列的累计接收数；
- 当前物理 workspace block 的 `expertRevTokenNums`。

数据写入完成后，函数发布本阶段自己的 SendCnt ready flag。

### `DispatchExpertTokens`

原型函数为 `MegaMoe::TripleInfoCalAndDispatch`。它按
`worldSize * blockNumPerRank` 切分 Dispatch shard，扫描远端 mask，并调用：

- `CalculateDispatchShardRange` 计算当前逻辑 block job 的匹配区间；
- `CopyTokensAndMetaForDispatch` 使用 UB ring 搬运 Token、Scale 和 MetaInfo；
- `FetchDispatchTokenAndMetaInfo` / `StoreDispatchTokenAndMetaInfo` 完成单 ring 槽位流水。

Dispatch 每完成一段数据，就按 GMM1 使用的 `PipelineTileM` 更新对应 wave 的 ready
计数。该通知属于 Dispatch 自身的数据完整性契约，因此保留在 Dispatch 模块内。

### `ControlDispatchGmm1SwigluPipeline`

这是当前 MegaMoe 模板的组合控制器，不是独立数学算子。它维持原有专家级重叠：

1. AIV1 预先执行 expert 0 的 SendCnt 和 Dispatch。
2. 处理 expert `e` 时，AIV1 预取并 Dispatch expert `e + 1`。
3. AIC/AIV0 等待并读取 expert `e` 的 Token count。
4. 更新 GMM1 输入、Weight、Scale、输出和 flag 地址。
5. 按 A8W8、A8W4、A4W4/NZ 模式执行 GMM1/SwiGLU。
6. 所有专家完成后执行原有收尾同步。

组合控制器使用拆开的 `TokenDispatchContext/Args/Scratch` 和
`Gmm1SwigluContext/Args/Scratch`。细分模块可以单独复用；当前模板的数据依赖和
同步依赖由组合层组织。

## GMM1 与 SwiGLU 流水

### Generic 路径

- AIC 在每个 Dispatch wave 开始前等待对应 ready 计数。
- AIC 执行 `ComputeGmm1TileGeneric`。
- 非 prefetch 路径通过 Cube/Vector flag 将 tile 交给 AIV。
- prefetch 路径通过 `gmm1TileStatus` 发布 tile ready。
- AIV 执行 `ComputeSwigluTileGeneric`。

### A8W4 路径

- AIV0 使用 `AivPrologueA8W4` 展开 FP4 Weight。
- AIC 使用 `ComputeGmm1TileA8W4` 执行 MMAD，并将中间结果写入 GM。
- AIV1 等待 tile ready，搬入 GMM1 结果并执行 `ComputeSwigluTileA8W4`。

### 同步边界

`BlockEpilogueSwigluMxQuant::operator()` 只执行一个 SwiGLU/MX Quant 子 tile，
不再隐式通知 GMM2。调用侧在每次 `operator()` 返回后立即调用
`NotifyCompletedTile()`，保持修改前逐子 tile 发布的时序；不会把两个 128 行
子 tile 合并完成后才一次性通知。

GMM1/SwiGLU 的 `WaitForCube`、`NotifyCube`、tile status 和 A8W4 GM sequence
均由 `mega_moe_group_matmul_pipeline.h` 控制。纯 tile 计算函数不绑定某一种上层
模板的专家循环。

## 共享专家

共享专家在路由专家之前执行：

- `PrepareSharedExpertInput` 负责纯 AIV 输入重排。
- `ControlSharedExpertGmm1SwigluPipeline` 负责共享专家的 GMM1/SwiGLU 组合调度。
- 共享专家没有 Token Dispatch，但 GMM1/SwiGLU 仍要求同一 MIX block 内的
  AIC 和 AIV 协作。

共享专家 GMM2、输出合并和 tile counter 逻辑保持原实现。

## GMM1/GMM2 兼容性

GMM1 和 GMM2 仍共享 `Config`、scheduler、layout、MMAD 和 AIV 后处理骨架，
通过 Policy 选择具体阶段。GMM1 新入口可以显式接收 `BlockJobContext`；
GMM2 和 Layered 仍保留现有物理 block 调用入口。

本 PR 只移除了 `MegaMoe::UpdateGroupParams` 中已无调用方的 GMM1 分支，并将保留的
GMM2 分支命名为 `UpdateGmm2GroupParams`。GMM2、Combine 以及 Layered 的主流程
没有改成新的调度上下文。

## 兼容性

- Kernel 输入、输出、Tiling 和 Host 调用方式不变。
- A8W8、A8W4、A4W4/NZ 路径保留。
- TopK weight prefetch 的 tile status 流水保留。
- Combine quant/non-quant 和大 BS 动态 counter 布局保留。
- 共享专家和 Layered 路径保留。
- Dispatch wave、SendCnt、Cube/Vector、A8W4 GM sequence 的通知时序保留。

## 验证重点

- 编译覆盖 A8W8、A8W4、A4W4、NZ 和 TopK weight prefetch。
- 单卡、多卡及不同 EP world size 的精度。
- `expert_token_nums` 与修改前一致。
- 普通 BS 和大 BS Combine 无越界、超时或死锁。
- 共享专家开启和关闭场景。
- Layered、MTE、URMA 等现有路径。
- 1 AIC + 2 AIV 的 Dispatch/GMM1/SwiGLU 同步链。
- 小、中、大 shape 的 Kernel 性能对比。
