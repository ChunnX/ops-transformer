# MegaMoe Wave 模板与普通模板差异说明

## 1. 文档范围

本文对比 Ascend 950 `arch35` 下的两条 MTE 执行路径：

- 普通模板：`MegaMoe`，主实现位于 `mc2/mega_moe/op_kernel/arch35/mega_moe.h`。
- Wave 模板：`MegaMoeWave`，主实现位于 `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h`。

`MegaMoeLayered` 是 URMA 拓扑使用的 Layered 模板，不是本文所说的“原有模板”。本文只在需要说明入口选择时提及它。

本文描述的是两套模板的算法组织、核间分工和流水差异，不讨论单核函数重构带来的文件组织变化。

## 2. 核心结论

普通模板按“大阶段”推进路由专家：

1. 在 `Dispatch -> GMM1 -> SwiGLU` 阶段处理完全部路由专家。
2. 再进入 `GMM2 -> Combine` 阶段，逐专家完成第二层计算和回发。

Wave 模板按若干连续专家组成的 wave 推进：

1. Host 根据预计 token 数和 GMM tile 数计算 `expertsPerBatch`。
2. Kernel 将连续的多个专家组成一个外层 expert wave。
3. 对当前 wave 执行 `GMM1/SwiGLU -> GMM2 -> Combine`，同时让 AIV1 预取后续 wave 的 Dispatch。
4. 不在每个 wave 之间执行全局 barrier，而是依靠阶段级 flag 推进。

因此，Wave 的本质是“跨专家的调度融合和滚动流水”，并不是把不同专家的 token 混合成一个不区分专家的大矩阵。

## 3. 模板入口选择

入口选择位于 `mega_moe_apt.cpp`。

| 条件 | 选择的模板 |
| --- | --- |
| `CommModeType == TILINGKEY_TPL_MTE`，MXFP，Dispatch 输出和 Weight1 均为 E5M2 或均为 E4M3FN | `MegaMoeWave` |
| `CommModeType == TILINGKEY_TPL_MTE`，但不满足上述 A8W8 dtype 配对条件 | `MegaMoe` |
| `CommModeType == TILINGKEY_TPL_URMA` | `MegaMoeLayered` |

当前 Wave 模板是 MTE A8W8 专用路径，支持 ND/NZ Weight，并保留 `TopkWeightsPrefetch`、Combine 通信量化和共享专家能力。A8W4、A4W4 等路径仍由普通模板或 Layered 模板承担。

Wave 还额外接收 `IsGmm1Interleaved` 编译期参数，用于选择 GMM1 交织布局及相应的 ping-pong 调度。普通模板没有这个模板参数。

## 4. 两种 wave 概念

代码中存在两个容易混淆的 wave：

1. **外层 expert wave**

   由 `expertsPerBatch` 控制，一次包含多个连续专家。`ProcessRoutedExpertWaves` 使用的是这个含义。

2. **专家内部的 row wave**

   以 `GMM1_TILE_M = 256` 行为粒度。Dispatch-ready 和 Wave 模板中的 SwiGLU-ready flag 按这个粒度发布。

本文将前者称为“expert wave”，将后者称为“256-row wave”。

## 5. 多专家 token 融合的准确含义

### 5.1 融合了什么

Wave 模板把多个专家放入同一个调度窗口：

- 共同参与 AIC tile 填充，减少单专家 tile 太少造成的空闲核。
- AIV1 以 expert wave 为单位预取 Dispatch。
- AIC/AIV0 连续推进当前 wave 内所有专家的 GMM1/SwiGLU 和 GMM2 调度游标。
- Combine 的 UB 行环形缓冲区和 `rowSequence` 在一个 expert wave 内跨专家复用。
- 当前 wave 与下一 wave 形成滚动流水。

### 5.2 没有融合什么

每个专家仍然保持独立的：

- token 数 `M`；
- Weight1、Weight2 和对应 scale；
- Dispatch 输出区间；
- SwiGLU 输出区间；
- GMM2 输出区间；
- metaInfo 和同步 flag。

`ProcessGmm1Wave`、`ProcessGmm2Wave` 和 `ProcessCombineExperts` 内部仍逐专家循环。各专家 token 只是按 `expertBeforeCnt` 或 `rowOffset` 连续排布，没有跨专家进行数学意义上的混算。

### 5.3 `expertsPerBatch` 的计算目的

Host 侧 `CalcExpertsPerBatch` 先估计单专家平均 token 数：

```text
expectedTokens = ceil(bs * topK / moeExpertPerRank)
```

再估算单专家的 GMM1、GMM2 tile 数，以二者较小值作为填核能力，计算填满 AIC 所需的专家数：

```text
expertsToFillAic = ceil(aicNum / limitingTilesPerExpert)
```

当 GMM1 的 gate/up 输出宽度大于 GMM2 输出宽度时，还会按 `ceil(hiddenDim / h)` 加深 wave。最终值被限制在 `[1, moeExpertPerRank]`。

所以 `expertsPerBatch` 的目标是提升并行度和流水深度，不是改变专家语义。

## 6. 公共流程

两套模板在路由专家主流水之外，大体保持相同的执行阶段：

1. 本卡 token 量化。
2. 计算并发送专家 mask。
3. 清零 Dispatch/GMM 同步 workspace。
4. 可选的共享专家输入准备。
5. AIC/AIV block 内同步。
6. 可选的共享专家 GMM1 + SwiGLU。
7. 跨 rank 同步，确保远端 Dispatch 输入可读。
8. 路由专家主流水。
9. block 内同步。
10. 可选的共享专家 GMM2。
11. 跨 rank 同步，确保 Combine 发送完成。
12. Unpermute。

主要差异集中在第 8 步路由专家主流水，以及为该流水服务的 workspace 和同步协议。

## 7. 普通模板流水

普通模板的路由专家流程可以概括为：

```text
Dispatch(E0)
    |
    +-- GMM1/SwiGLU(E0)，同时 AIV1 预取 Dispatch(E1)
    +-- GMM1/SwiGLU(E1)，同时 AIV1 预取 Dispatch(E2)
    +-- ...
    +-- GMM1/SwiGLU(En)

完成全部专家的 GMM1/SwiGLU 阶段
    |
    +-- GMM2/Combine(E0)
    +-- GMM2/Combine(E1)
    +-- ...
    +-- GMM2/Combine(En)
```

主要特点：

- Dispatch 对下一个专家做一专家深度的预取。
- `ControlDispatchGmm1SwigluPipeline` 统一处理全部专家的 Dispatch、GMM1 和 SwiGLU。
- 非 prefetch 路径通过 `EndSync`，prefetch 路径通过 all-done 状态，收束整个 GMM1/SwiGLU 阶段。
- GMM2 在全部路由专家的 GMM1/SwiGLU 阶段结束后启动。
- `GroupMatmulWithCombine` 将每个专家的 GMM2 与对应 Combine 组织在一起。
- GMM2 与 Combine 的具体耦合方式随 A8W8/A8W4/A4W4、是否 Combine 量化而变化。

## 8. Wave 模板流水

设当前 expert wave 为 `B`，下一 expert wave 为 `B+1`。稳态流水为：

```text
AIC : GMM1(B) ----------------> GMM2(B) ----------------> GMM1(B+1)
          |                         |
          | GMM1 -> SwiGLU          | GMM2-ready
          v                         v
AIV0: SwiGLU(B) ---------------------------------------> SwiGLU(B+1)

AIV1: Dispatch(B+1) -------------> Combine(B) --------> Dispatch(B+2) -> Combine(B+1)
```

第一轮需要先完成当前 wave 和下一 wave 的 Dispatch 预热。进入稳态后，当前 wave 的 Dispatch 已由上一轮完成，因此 AIV1 每轮只补充新出现的后续 wave。

单次 `ProcessRoutedExpertWaves` 循环的代码顺序是：

1. `DispatchExpertsUntil`：AIV1 将 Dispatch 推进到下一 wave 末尾。
2. `ProcessGmm1Wave`：AIC 执行 GMM1，AIV0 执行 SwiGLU。
3. `ProcessGmm2Wave`：AIC 执行 GMM2，AIV0 只推进对应的虚拟调度游标。
4. `ProcessCombineExperts`：AIV1 等待当前专家 GMM2-ready 后执行 Combine。

这些函数在不同核上独立前进。代码调用顺序不等价于全核串行执行，中间没有 expert-wave 级 `SyncAll`。

## 9. 一个 block 内的核角色变化

950 上一个完整 block 包含一个 AIC 和两个 AIV。以下角色描述限定在路由专家主流水内。

| 核角色 | 普通模板 A8W8 主路径 | Wave 模板 |
| --- | --- | --- |
| AIC | GMM1、GMM2 | 当前 wave 的 GMM1、GMM2，并连续进入下一 wave |
| AIV0，`subBlockIdx == 0` | GMM1 后的 SwiGLU；其他职责随路径变化 | 专职当前 wave 的 SwiGLU，并为 GMM2推进虚拟 tile 游标 |
| AIV1，`subBlockIdx == 1` | SendCount、Dispatch；Combine 参与方式随具体路径变化 | 专职 SendCount、Dispatch 和 Combine，不参与 GMM1/GMM2 tile 调度 |

Wave 对 AIV1 的职责划分更固定。这样可以让 AIV1 在 AIC/AIV0 计算当前 wave 时准备后续输入，并在 GMM2 完成后回发当前 wave 结果。

## 10. 各阶段差异

### 10.1 SendCount 和 Dispatch

普通模板：

- AIV1 先处理专家 0。
- 在处理专家 `E` 的 GMM1/SwiGLU 时，提前计算并 Dispatch 专家 `E+1`。
- 预取深度主要是一位专家。

Wave 模板：

- AIV1 持有独立的 `nextDispatchExpert` 游标。
- 第一轮 Dispatch 当前 wave 和下一 wave。
- 稳态下在 Combine 当前 wave 之前完成后续 wave 的 Dispatch。
- Dispatch 与 Combine 复用 UB 和事件资源，因此同一 AIV1 上仍按明确顺序串行复用资源。

两套模板的 Dispatch 都保持专家边界，并按 256-row wave 更新 GMM1 ready 计数。Wave 改变的是预取范围和上层调度顺序，不是 Dispatch 的路由语义。

### 10.2 GMM1 和 SwiGLU

普通模板：

- 对全部专家执行完 GMM1/SwiGLU 后再结束该大阶段。
- 阶段末有整体收束同步。
- GMM2 不会以 expert wave 为单位穿插进来。

Wave 模板：

- 只处理当前 expert wave 内的专家。
- AIC 和 AIV0 不执行 wave 级全局 barrier。
- GMM1 到 SwiGLU 仍使用原有 Cube/Vector 事件或 prefetch GM tile-status 协议。
- SwiGLU-ready 按专家内 256 行粒度发布，使 GMM2 能按相同 row wave 等待上游结果。
- 可通过 `IsGmm1Interleaved` 使用 GMM1 ping-pong/交织调度。

### 10.3 GMM2

普通模板：

- 在全部路由专家 GMM1/SwiGLU 完成后逐专家执行。
- GMM2 与 Combine 使用路径相关的联合实现。
- Combine 量化路径通常通过 token-group counter 通知消费核。

Wave 模板：

- 每完成一个 expert wave 的 GMM1/SwiGLU 调度，就进入该 wave 的 GMM2。
- GMM2 固定生成 BF16 结果并写入本地 GM workspace。
- 每个路由 token 行在整个 kernel 生命周期内拥有唯一的 `[H]` 输出区域，不使用 batch ring 覆盖。
- AIC 写完结果后发布独立的 GMM2-ready 到达标记。
- AIC 可以继续推进后续计算，无需等待 Combine 返回 buffer ACK。

### 10.4 Combine

普通模板：

- Combine 位于逐专家 `GroupMatmulWithCombine` 流程中。
- 不同量化和 GMM 路径可能使用不同 AIV 分工及同步方式。
- Combine 量化路径按 token group 等待 GMM2 tile counter。

Wave 模板：

- Combine 从 GMM2 计算中明确拆出，由 AIV1统一执行。
- AIV1 先等待当前专家所有 AIC 的 GMM2-ready 到齐。
- 每个 AIV1 负责该专家的一段连续 token 区间。
- metaInfo 最多按 1536 个 token 分批搬入 UB。
- GMM2 行通过 UB row ring 搬入；无量化时直接发送，量化时先执行 BF16 到 FP8 的转换再发送。
- `rowSequence` 在一个 expert wave 内跨专家连续使用，减少每个专家重新建立流水的开销。

## 11. Wave 的同步链

Wave 路由专家阶段使用四类主要依赖：

```text
Dispatch
  |  每个专家、每 256 行的 Dispatch-ready 计数
  v
GMM1 on AIC
  |  Cube/Vector event，或 TopkWeightsPrefetch 的 GM tile status
  v
SwiGLU on AIV0
  |  每个专家、每 256 行的 SwiGLU-ready flag
  v
GMM2 on AIC
  |  每个专家、每个 AIC 独占 64B cache line 的 GMM2-ready
  v
Combine on AIV1
```

GMM2-ready 的具体协议为：

1. 每个 AIC 在此前 GMM2 GM 写回完成后执行 `FIX_S` 同步。
2. 每个 AIC 向自己的 cache line 写入到达标记。
3. 每个 AIV1 将当前专家的全部 AIC ready 区搬入 UB。
4. AIV1 使用 `ReduceSum` 统计到达数，直到达到物理 block 数。
5. 到齐后读取 GMM2 BF16 结果并执行 Combine。

每个通知槽独占 cache line，用于避免不同核轮询时的伪共享。

## 12. 调度游标与相位修正

普通模板按完整大阶段推进，GMM1 结束后的 `startBlockIdx` 可以直接作为 GMM2 的起始调度状态。

Wave 模板中 GMM1 和 GMM2 在每个 expert wave 内连续推进同一个核归属相位。若两级 GMM 的 tile 数对 block 数取模后刚好互相抵消，下一 wave 可能持续固定在同一组物理核上，形成“固定角色共振”。

`ProcessRoutedExpertWaves` 会比较：

- wave 入口游标；
- GMM1 结束游标；
- GMM2 结束游标。

检测到固定角色共振时，将下一 wave 的入口恢复到 GMM1 结束游标，以改变下一 wave 的核归属相位，同时不破坏当前 wave 内 GMM1 到 GMM2 的映射关系。

AIV1 不参与两级 GMM 的真实或虚拟 tile 调度，因此不会触发该修正。

## 13. Workspace 差异

Wave 为滚动流水增加或改变了以下 workspace：

- SwiGLU-ready 从每专家一个槽扩展为每专家、每 256-row wave 一个槽。
- 增加每专家、每 AIC 一个 cache-line 隔离的 GMM2-ready 区域。
- 为全部路由结果分配完整的 BF16 GMM2 输出区，大小约为 `maxOutputSize * h * sizeof(bfloat16_t)`。
- Wave 的 Combine 不使用普通 Combine 量化路径的 group-sync counter。
- AIV1 在 UB 中维护 Combine row ring、量化临时区、metaInfo 分批缓冲区和 GMM2-ready ReduceSum 缓冲区。

代价是 Wave workspace 更大，静态 UB 布局和事件编号管理也更严格。

## 14. 共享专家处理

共享专家没有并入路由专家的 expert wave：

- 共享专家输入准备仍在路由通信前完成。
- 共享专家 GMM1 + SwiGLU 仍提前执行。
- 路由专家的全部 expert wave 完成后，再执行共享专家 GMM2。
- 最终由 Unpermute 将路由专家和共享专家结果纳入同一输出累加流程。

Wave 的共享专家 GMM1 会复用 Wave 版 GMM1/SwiGLU 实现，并在结束后重置 `startBlockIdx`，避免共享专家调度相位影响路由专家 wave。

## 15. 性能收益来源

Wave 主要针对单专家 token 较少、单专家 tile 无法填满所有 AIC，以及多阶段间存在空洞的场景：

- 多专家共同填充一个调度窗口，提高 AIC 利用率。
- Dispatch 从一专家预取扩大为一整个后续 wave。
- 移除 expert-wave 之间的全局 barrier。
- AIC 完成当前 GMM2 后可以继续下一 wave 的 GMM1。
- AIV1 将下一 wave Dispatch 与当前 wave Combine 交错执行。
- GMM2 输出使用唯一 GM 区域，避免生产者等待环形 buffer 消费确认。
- Combine row ring 跨专家复用，降低短专家频繁启动和排空流水的成本。

## 16. 代价与适用边界

Wave 的主要代价是：

- 仅服务特定的 MTE A8W8 dtype 路径，通用性低于普通模板。
- 增加 BF16 GMM2 中间结果和 GMM2-ready workspace。
- 增加 GM flag 写入、轮询和 ReduceSum。
- AIV1 同时承担 Dispatch 和 Combine，二者必须严格管理 UB 与事件资源的复用顺序。
- `expertsPerBatch`、tile 数、block 数和核归属相位会共同影响性能，调度更复杂。
- 小 case 中额外的 GM 同步成本可能抵消流水收益；大 case 或专家 tile 已充分填核时，收益也可能受限。

因此 Wave 不是普通模板的无条件替代，而是针对 MTE A8W8 多专家场景的专用性能模板。

## 17. 输出语义

两套模板的数学语义和最终输出保持一致：

- Dispatch 仍按 top-k expert 路由。
- 每个专家仍独立执行 GMM1、SwiGLU 和 GMM2。
- Combine 仍将专家结果发送回 token 来源 rank。
- Unpermute 仍按 top-k weight 聚合路由专家和共享专家结果。

Wave 改变的是专家和阶段的执行顺序、并行覆盖关系及同步协议，不应改变 token 到专家的映射、专家计算结果或最终输出布局。

## 18. 关键代码位置

| 内容 | 文件或函数 |
| --- | --- |
| 模板入口选择 | `mega_moe_apt.cpp::mega_moe` |
| expert wave 大小计算 | `mega_moe_tiling.cpp::CalcExpertsPerBatch` |
| 普通模板主流程 | `mega_moe.h::MegaMoe::Process` |
| 普通 Dispatch/GMM1/SwiGLU 调度 | `mega_moe_dispatch_gmm1_swiglu.h::ControlDispatchGmm1SwigluPipeline` |
| Wave 主流程 | `mega_moe_wave.h::MegaMoeWave::Process` |
| Wave 路由专家滚动调度 | `mega_moe_wave.h::ProcessRoutedExpertWaves` |
| Wave GMM1/SwiGLU | `mega_moe_wave.h::ProcessGmm1Wave` |
| Wave GMM2 | `mega_moe_wave.h::ProcessGmm2Wave` |
| Wave Combine | `mega_moe_wave.h::ProcessCombineExperts` |
| GMM2-ready 协议 | `mega_moe_wave.h::PublishGmm2Ready`、`WaitGmm2Ready` |
| Wave workspace 布局 | `mega_moe_workspace_info.h::WorkspaceInfo` |
