# MegaMoe A6（arch92）A8W8 迁移详细方案

## 1. 文档目的

本文档用于指导后续将 A5 `arch35` 的 MegaMoe A8W8 能力迁移到 A6 `arch92`。方案以当前仓库代码为基线，目标不是简单复制 A5 Kernel，而是在保持 A5 已有功能和性能路径不变的前提下，建立适配 A6 `AIC:AIV = 1:1` 拓扑的 A8W8 Wave 流水。

本文档覆盖：

- Kernel 入口、核心映射、Wave 调度和同步协议。
- Dispatch 前处理、共享专家、MoE 专家流水和 Unpermute 后处理。
- Workspace、Win 区和 UB 使用约束。
- A5/A6 源码隔离、可复用头文件边界和 arch92 本地副本策略。
- ACLNN、OpDef、InferShape、Tiling、CMake、SoC 注册和打包依赖。
- 支持能力边界、实施顺序、验证矩阵和主要风险。

本阶段不新增或迁移测试代码，但正式实现后仍需使用已有编译、精度和性能用例进行验证。

## 2. 已确认约束

1. A6 使用独立 `mc2/mega_moe/op_kernel/arch92` 目录。
2. A6 默认物理拓扑是一个 AIC 配一个 AIV，Kernel task 类型使用 `KERNEL_TYPE_MIX_AIC_1_1`。
3. 函数名、变量名和类型名不使用 `OneAiv`、`SingleAiv`、`Aiv1Pipeline` 等名称。1C1V 是默认架构事实，不是特殊业务模式。
4. 当前只支持 A8W8，不支持 A8W4、A4W4 和 A4W4_NZ。
5. A6 不支持 URMA，只支持 MTE。
6. 除 URMA 和非 A8W8 数据类型路径外，A5 当前 A8W8 已支持的功能均需要保留。
7. A5 `arch35` 保持零修改。A6 需要改变的 Workspace、Dispatch、GMM1、GMM2、reset 和主类实现复制到 `arch92` 后独立修改。
8. arch92 不复制整套 arch35 目录；仅复制 A6 必须改变的文件，其余类型、工具、Blaze 和独立 stage 仍引用 arch35 稳定头文件。
9. Dispatch 前处理和 Combine 后的 Unpermute 仍由全部 AIV 参与，按实际 AIV 核数分核。
10. MoE 专家主体采用 A8W4 Wave 的专家调度骨架，但计算使用 A8W8 Generic MMAD，不复用 A8W4 FP4 Prologue。

## 3. 基线代码与关键事实

主要参考文件：

- `mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp`
- `mc2/mega_moe/op_kernel/arch35/mega_moe.h`
- `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h`
- `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w4.h`
- `mc2/mega_moe/op_kernel/arch35/common/mega_moe_workspace.h`
- `mc2/mega_moe/op_kernel/arch35/common/mega_moe_types.h`
- `mc2/mega_moe/op_kernel/arch35/common/mega_moe_gmm_common.h`
- `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h`
- `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h`
- `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h`
- `mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp`

### 3.1 A5 入口隐含的是 1C2V

当前 `mega_moe_apt.cpp` 使用：

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
```

当前 `MegaMoe` 主类还直接使用：

```cpp
blockNum_ = GetBlockNum();
blockAivNum_ = GetBlockNum() * 2;
blockIdx_ = GetBlockIdx() / GetTaskRation();
aivCoreIdx_ = GetBlockIdx();
```

因此 arch92 不能直接继承现有 `MegaMoe` 主类并只删除 AIV0 分支。核心数量、AIV 编号、Combine 参与核、Scratch 初始化和若干同步槽都隐含了 1C2V。

### 3.2 A8W4 Wave 的“2”只约束 GMM1 调度目标

当前 A8W4 Wave 使用：

```cpp
A8W4_GMM1_TILES_PER_JOB_TARGET = 2U;
waveMGroupTarget = ceil(
    2U * gmm1Config_.blockJob.totalJobs /
    gmm1TilesPerMGroup);
```

准确含义是：常规 Wave 以每个逻辑 AIC 约两个 GMM1 scheduler tile 为目标负载，而不是每个 Wave 固定执行两次 GMM1 和两次 GMM2。

- 尾 Wave 可以少于目标。
- Dispatch仍按完整专家执行，但GMM1/GMM2/Combine允许在256-token M-group边界切分专家。
- 单个大专家可以跨多个Wave连续处理，不再独自撑大一个Wave。
- GMM2没有独立的两轮目标，只处理当前Wave覆盖的token切片。

A6固定使用非交织Weight1，GMM1和GMM2的tile数分别近似为：

```text
GMM1: sum(ceil(expertTokens / 256) * ceil((hiddenDim / 2) / 256))
GMM2: sum(ceil(expertTokens / 256) * ceil(h / 256))
```

因此A6 Generic GMM1 scheduler固定使用`hiddenDim / 2`作为N方向调度宽度。arch35已有的Weight1 Interleave能力不在本次修改范围内。

### 3.3 不能直接复用 A8W4 MMAD 函数

`MegaMoeA8W4Wave` 静态限制 FP8 Activation + FP4 Weight；`RunGmm1A8W4` 和 `RunGmm2A8W4` 还依赖 AIV0 的 FP4 Prologue。A6 A8W8 必须使用 Generic A8W8 BlockMmad，只复用 A8W4 的 token-grained Wave 划分和 AIV 调度顺序。

### 3.4 不能直接使用 A5 A8W8 Wave 主流程

A5 A8W8 Wave 是围绕 1C2V 设计的：一个 AIV 长时间负责 SwiGLU，另一个 AIV 负责 Dispatch/Combine。A6 只有一个 AIV，三个阶段共享 UB，无法保留这种角色并行。

采用 A8W4 Wave 骨架的原因是：它已经在同一个 AIV 角色上组织 Dispatch、Activation 和 Combine，并通过 GM 承接 GMM 输出。A6 只需要删除 FP4 Prologue 角色，并将 GMM 替换为 A8W8 Generic 实现。

## 4. A6 总体执行流程

### 4.1 Phase 1：Dispatch 前处理

全部 AIV 按 `jobIndex = physicalAivIdx`、`totalJobs = aivNum` 分核执行：

1. `QuantizeLocalTokens`
2. `GatherAndSendExpertMasks`
3. `ResetSyncStatus`
4. 存在共享专家时执行 `PrepareSharedExpertInput`
5. AIV 完成必要的 `PipeBarrier<PIPE_ALL>()`
6. AIC/AIV 共同进入 `SyncAll<false>()`

要求：

- `blockAivNum` 必须等于平台真实 AIV 数，不能使用 `GetBlockNum() * 2`。
- Quant、SendMask、Reset、Shared input 和 Unpermute 已接收 JobContext，原则上直接复用。
- Host 侧 SendMask/Reset/Unpermute UB 配置必须按 A6 的真实 AIV 数重新计算。
- `SyncAll` 的调用次数和顺序必须保证 AIC/AIV 两侧一致，不能因某个功能关闭而只让一侧跳过全核同步。

### 4.2 Phase 2：共享专家 GMM1

共享专家仍位于 routed expert 流水之前，保持 A5 现有语义：

1. 使用 `sharedExpertInputDataPtr/sharedExpertInputScalePtr` 作为 GMM1 输入。
2. 使用 A8W8 Generic GMM1。
3. AIV执行激活和量化，写入 `sharedExpertActivationDataPtr/sharedExpertActivationScalePtr`。
4. 不参与 Dispatch，也不进入 routed expert 的 Wave 划分。

共享专家阶段没有并发 Dispatch，可以优先复用 1C1V 下的 AIC/AIV CV 握手；如果 arch92 的 CV/UB 接口与 A5 Generic 路径不兼容，则再为共享专家切到 GM handoff。不能复用 A8W4 Prologue。

### 4.3 Phase 3：跨卡同步

调用 `CrossRankSyncInWorldSize`：

- 各 AIV 按实际 `aivNum` 分摊 rank。
- 每个 AIV 的本地 sequence 位仍使用 `48KB + physicalAivIdx * 64B`。
- rank 同步区仍位于 Win 区头部 60KB 保留区中。
- 完成远端握手后执行本卡 AIV 同步。

### 4.4 Phase 4：MoE专家Wave流水

整体沿用 A8W4 Wave 的专家循环形态：

```text
启动：AIV开始Dispatch第一个Wave；AIC不等整个Wave完成，按tile-ready flag启动GMM1。

稳态：
  AIV：Dispatch(next wave covered experts) -> Activation(current wave token slice)
  AIC：GMM1(current wave token slice)

当前Wave的GMM1/Activation完成后：
  AIC：GMM2(current wave token slices)
  AIV：Combine(current wave token slices)

随后进入下一Wave。
```

关键点：

- 启动阶段没有“Dispatch整个Wave完成”的全局 barrier。AIC只等待当前 GMM1 tile 所需的 Dispatch 行数到齐。
- AIV 在内层循环中先推进下一 Wave 的 Dispatch 游标，再消费当前 Wave 的一个专家切片的 GMM1 输出；Dispatch 本身仍按完整专家执行。
- AIC 跳过 Dispatch 代码，直接推进当前 Wave GMM1；AIV和AIC允许在时间上不同步，但必须从相同 token count 计算出一致的二维 Wave 边界。
- 当前 Wave 的 GMM2/Combine 完成后，`gmm1Position` 才从 `waveEndPosition` 继续推进。
- Dispatch、SwiGLU 和 Combine 在同一 AIV 上共享 UB，因此只能串行调用，不能尝试在同一 AIV 内做并发 UB 生命周期重叠。

建议的上层控制伪代码：

```cpp
ExpertTokenPosition dispatchPosition{};
ExpertTokenPosition gmm1Position{};
if constexpr (g_coreType == AIV) {
    DispatchFirstWave(dispatchPosition);
}

while (gmm1Position.expertIdx < expertCount) {
    ExpertTokenPosition waveBeginPosition = gmm1Position;
    ExpertTokenPosition waveEndPosition = gmm1Position;
    while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
        if constexpr (g_coreType == AIV) {
            if (nextWaveNeedsDispatch) {
                DispatchCurrentExpertIfNeeded(dispatchPosition);
                AdvancePositionByMGroup(dispatchPosition);
            }
        }

        if (currentWaveNeedsGmm1) {
            LoadExpertTokenCountWhenEnteringExpert(gmm1Position.expertIdx);
            auto slice = AdvancePositionByMGroup(gmm1Position);
            if constexpr (g_coreType == AIC) {
                WaitRequiredDispatchRowsAndRunGmm1SliceToGm(slice);
                NotifyGmm1ResultReady();
            } else {
                WaitGmm1ResultReadyAndRunActivationSliceFromGm(slice);
                NotifyGmm2InputReady();
            }
            waveEndPosition = gmm1Position;
        }
    }

    for (auto slice : TokenSlicesBetween(waveBeginPosition, waveEndPosition)) {
        LoadExpertTokenCountWhenEnteringExpert(slice.expertIdx);
        if constexpr (g_coreType == AIC) {
            WaitActivationReadyAndRunGmm2SliceToGm(slice);
            NotifyCombineReady();
        } else {
            WaitOwnedGmm2ResultsAndRunCombineSliceFromGm(slice);
        }
    }
}
```

Generic A8W8 BlockMmad应使用持久上下文跨专家和Wave复用，避免在每个专家边界重复构造/析构。ND和NZ分别实例化对应的 `PersistentBlockMmadContext`，由arch92主类在整个专家流水期间持有。

### 4.5 Phase 5：导出专家Token数

- 每个物理 block 对应的 AIV都将本 block 的专家 token 数写入 `expertRevTokenNumsPtr`。
- 只选一个明确的 AIV Job 将本卡各专家 token 数导出到 `expertTokenNumsOut`。
- 参与者判断应由上层主流程根据 `jobIndex == 0` 控制，不在可复用底层函数中硬编码 `GetSubBlockIdx() == 1`。
- A6继续使用 `BlockWorkspaceContext{physicalBlockIdx, aicNum}`，保证 count producer 和 AIC consumer 使用相同 workspace 槽。

### 4.6 Phase 6：共享专家GMM2

保持现有顺序：routed expert 阶段结束并完成本卡同步后，再执行共享专家 GMM2。

- 使用 A8W8 Generic GMM2。
- 输出到 `sharedExpertResultPtr`。
- AIC按 token group 更新 `sharedExpertGmm2TileCounterPtr`。
- Unpermute 在累加共享专家输出前等待自己所需 token group 就绪。

### 4.7 Phase 7：Unpermute后处理

1. 全部AIV再次执行跨卡同步，确保Combine发送完成。
2. 全部AIV按 `aivNum` 对本卡 `bs` 分核。
3. 从Combine接收区读取MoE专家结果。
4. 按需读取topk weights并累加。
5. 存在共享专家时累加共享专家结果。
6. 写出 `y`。

## 5. 核心编号与上下文设计

### 5.1 初始化原则

arch92 主类在 `Init` 中根据 TilingData 显式初始化核心上下文，不使用 `GetBlockNum() * 2`：

```cpp
physicalBlockIdx = static_cast<uint32_t>(GetBlockIdx() / GetTaskRation());
physicalBlockCount = tilingData->aicNum;
physicalAivIdx = static_cast<uint32_t>(GetBlockIdx());
physicalAivCount = tilingData->blockAivNum;
```

Host Tiling必须保证：

```text
aicNum > 0
aivNum > 0
aicNum == aivNum
blockAivNum = aivNum
```

`GetTaskRation()` 在 1C1V Kernel task 下预期为1，但仍应在目标编译器/模拟器上确认。业务代码不应依靠 `GetSubBlockIdx()` 区分“哪个AIV参加流水”。

### 5.2 Context职责

- `BlockJobContext`：描述某次GMM scheduler的逻辑Job编号和Job总数。
- `BlockWorkspaceContext`：描述物理block对应的count/flag workspace槽。A6中与物理1C1V block一一对应。
- `AivJobContext`：描述AIV任务池中的分核编号，用于Quant、SendMask、Reset、Combine量化和Unpermute。
- `MoeStageCommonConfig`：保存所有stage共享且在一次Kernel执行期间不变的shape、rank和专家配置。

不要因为A6当前1C1V就删除 `BlockWorkspaceContext`。Job调度编号和物理workspace编号仍属于不同语义，后续部分核调度时二者可能再次不同。

## 6. Wave划分方案

### 6.1 按M-group控制每个Wave的GMM1负载

arch92同步A5最新A8W4 token-grained Wave策略，每个逻辑AIC仍以两个GMM1 scheduler tile为目标：

```cpp
GMM1_TILES_PER_JOB_TARGET = 2;
gmm1TilesPerMGroup = ceil(schedulerN / L1_TILE_N);
waveMGroupTarget = ceil(
    GMM1_TILES_PER_JOB_TARGET * gmm1BlockJob.totalJobs /
    gmm1TilesPerMGroup);
```

其中A6固定使用：

```cpp
schedulerN = hiddenDim / 2;
```

Wave以256-token M-group为最小切分单位。一个专家超过当前Wave剩余容量时，只把当前容量覆盖的M-group放入本Wave，剩余token由下一Wave继续处理。

### 6.2 A6不保留Weight1 Interleave

A6首版只支持非交织Weight1，布局契约为：

```text
[全部Gate列][全部Up列]
```

该决策基于A6实际流水，而不是仅把编译开关固定为0：

- AIC执行GMM1后将Gate/Up结果写入GM，AIV等待GM sequence后执行GM2UB和SwiGLU。
- SwiGLU不直接消费AIC共享UB，也不会通过占用同一块UB同步阻塞下一次GMM1。
- 原Interleave路径的主要收益是把Gate/Up双区改为Ping/Pong区，以支持AIC/AIV直接UB交接时的双缓冲；该机制在A6 `USE_GM_HANDOFF=true`路径中不可达。
- Wave按M-group提供足够的GMM1调度并行度，没有必要为了更细的N轴切片继续维护特殊Weight1布局、额外scheduler粒度、Activation分支、flag计数和模板实例。

因此arch92删除`MEGA_MOE_WEIGHT1_INTERLEAVED`入口开关、ArchitecturePolicy能力位以及GMM1/Epilogue/GMM2中的Interleave模板分支。arch35现有Interleave代码和能力保持不变。

### 6.3 AIC与AIV如何得到一致Wave边界

- AIV在 `ComputeExpertTokenCountAndNotify` 中得到 `sendCnt`。
- 每个物理block将相同专家token数写入 `expertRevTokenNumsPtr[expert][physicalBlock]`。
- AIC和AIV都从自己的物理block槽读取专家token数，再使用相同公式推进`(expertIdx, tokenIndexInExpert)`二维游标。
- Wave边界不能依赖只有AIV本地可见的临时变量。

### 6.4 GMM2不重新划Wave

GMM2直接处理当前GMM1 Wave覆盖的二维token区间`[waveBeginPosition, waveEndPosition)`，不根据GMM2 tile数重新分Wave。同一个大专家可在多个Wave中分别执行GMM1、GMM2和Combine。

### 6.5 专家内切片约束

该策略已经进入A6实现，约束如下：

1. Dispatch仍按完整专家执行；后续Wave从同一专家中间继续时只读取已发布token count，不重复Dispatch。
2. GMM1/GMM2的数据地址使用`expertGlobalTokenStart + tokenIndexInExpert`。
3. Dispatch和Activation flag使用`tokenIndexInExpert / 256`对应的M-group槽。
4. GMM2量化Combine counter仍以完整专家为坐标，生产端通知`tokenIndexInExpert + mLoc`对应group。
5. Combine只消费当前切片覆盖的group，但保持原有全专家分核等差序列。

Wave目标仍为：

```text
targetTiles = 2 * gmm1JobCount
waveMGroupTarget = ceil(targetTiles / gmm1TilesPerMGroup)
```

尾Wave在剩余总工作量不足目标时允许欠载。该策略只控制GMM1负载，不能单独保证绝无流水等待，验证时仍需观察：

- Dispatch是否能在当前Wave GMM1时间内维持下一Wave输入领先。
- 单AIV优先补Dispatch时是否推迟Activation并导致GMM2等待。
- GMM2的tile数由`h`决定，不能沿用GMM1的两轮负载结论。
- 部分专家切片的Combine是否存在额外等待或重复启动。

## 7. 分阶段同步协议

### 7.1 SendCount发布

Producer：每个AIV block。

Data：

- `expertRevTokenNumsPtr[expert][physicalBlock]`
- A8W4式持久cumsum（A6同样需要）

Notification：

- `flagSendCntCalToUpdParamsPtr[expert][physicalBlock]`

Consumer：对应物理block的AIC和AIV。

SendCount发布属于Dispatch stage自身职责，保持“写count后发布flag”的现有原子语义，不需要由上层拆开。

### 7.2 Dispatch到GMM1

Producer：所有AIV Job共同Dispatch当前专家。

Data：

- `dispatchRevDataPtr`
- `dispatchRevScalePtr`
- 可选topk weight/meta数据

Notification：

- `flagDispatchToGmm1Ptr[expert][mGroup]`
- 每个完成的Dispatch shard按其与256行m-group的重叠行数执行AtomicAdd。

Consumer：负责对应GMM1 tile的AIC。

启动条件：当前flag值等于该m-group真实需要的行数。只等待当前tile，不等待整个专家或整个Wave Dispatch结束。

该通知属于Dispatch完成定义的一部分，继续保留在Dispatch底层函数内部。

### 7.3 GMM1到Activation

Data：AIC将BF16 GMM1结果写入 `gmm1MmadResPtr`。

非topk预取路径：

- 每个AIC使用独立的 `flagGmmToEpiloguePtr[physicalBlock]` 单调sequence。
- AIC完成FIX到GM的可见性保证后递增sequence。
- 配对AIV等待期望sequence，再从GM取回对应tile。

topk预取路径：

- 可继续使用 `gmm1TileStatusPtr[expert][schedulerTile]` 和 `expertIdx + 1` round tag。
- 状态位写入必须发生在GMM1结果对GM可见之后。
- AIV等待当前expert/tile对应tag，不读取上一轮残留状态。

上层Wave控制器负责“等待/通知”；纯MMAD tile函数和纯Activation tile函数不嵌入对另一stage的调度假设。

### 7.4 Activation到GMM2

Data：

- `activationQuantDataPtr`
- `activationQuantScalePtr`

Notification：

- `flagActivationToGmm2Ptr[expert][mGroup]`
- 每个Activation scheduler tile完成后向所属M-group槽AtomicAdd。

GMM2切片只等待当前切片所需M-group的Activation tile完成，不再等待整个大专家完成。

### 7.5 GMM2到非量化Combine

Data：AIC将每个GMM2 BF16 tile写入 `gmm2MmadResPtr`。

Notification：复用每物理block的单调GM sequence。AIC保证FIX结果可见后发布，配对AIV按相同scheduler顺序等待并消费。

特点：

- 一个AIC tile只由配对AIV消费。
- 不等待当前专家全部AIC完成。
- 不使用A5 A8W8 Wave的 `gmm2ReadyPtr[expert][aic]` 全专家barrier。

### 7.6 GMM2到量化Combine

量化Combine按专家token group分配给全部AIV Job，不要求AIV与产生该token的AIC物理配对。

- group大小保持256 token。
- AIC每完成一个 `(mGroup, nTile)`，向该group对应的consumer slot计数。
- 某group需要等待的目标值为 `ceil(h / 256)`。
- AIV只等待自己负责的group，然后一次处理该group中分配给自己的token范围。
- `logicalCoreCount` 必须显式传入实际Combine Job数，A6为 `aivNum`，不能继续通过 `config.blockNum * 2` 推导。

### 7.7 全核和跨卡同步

必须保留的全局顺序点：

1. Reset完成后，AIC才能进入后续flag轮询。
2. 共享专家GMM1后、routed expert通信前的跨卡同步。
3. routed expert结束后，导出count和共享专家GMM2前的本卡同步。
4. Unpermute前确保Combine远端发送完成的跨卡同步。

所有分支必须保证同一种core type执行相同次数的 `SyncAll`，防止部分核心跳过造成超时。

## 8. AIV阶段顺序与UB生命周期

### 8.1 AIV串行顺序

同一个AIV在稳态内执行：

```text
Dispatch(next wave expert)
Activation(current wave expert)
Combine(current wave expert，在GMM2阶段)
```

这不是三段并行流水。SwiGLU/量化会占用大部分或全部UB，Dispatch和Combine必须在函数边界上与其串行。

### 8.2 首版cumsum落GM

Dispatch的 `cumsumInfoTensor` 位于UB，而首版Activation布局会覆盖256KB范围内的Dispatch UB。首版A6像A8W4一样：

- 每次需要时从 `cumsumInfoPtr` 恢复前序cumsum。
- 当前专家SendCount计算完成后写回GM。
- arch92 本地副本将 `EnableA8W4` 中与数据类型无关的持久化语义抽象为 `PersistDispatchCumsum`；A5 原文件不修改。

这只是首版在尚未划分arch92常驻UB区时的保守实现，不是A6最终必须保留的GM协议。A6额外128KB UB为cumsum常驻提供了条件，后续优化见9.8节。

### 8.3 Scratch对象只保存视图和标量状态

Scratch中LocalTensor只是UB地址视图，不能被误认为跨stage保存了数据。跨Dispatch/Activation/Combine仍需保留的数据必须位于GM或明确的Scalar状态中。

### 8.4 Event ID审计

虽然三个stage串行，仍需检查每个函数退出前是否已回收其MTE/V event：

- Dispatch ring的MTE3_MTE2/MTE3_S事件必须drain。
- Activation GM2UB和V_MTE2事件必须闭合。
- Combine发送ring必须在函数返回前等待所有已发slot。
- 不能把未完成event跨stage复用，否则下一个stage可能错误等待同一event ID。

## 9. 384KB UB优化

### 9.1 优化边界

A6每个Vector核的UB从A5的256KB增加到384KB。UB容量增加不会直接提高Vector指令吞吐，主要用于：

- 扩大批处理粒度，减少Scalar循环、DataCopy指令和Event同步次数。
- 增加ping-pong/ring buffer深度，使MTE2、Vector和MTE3流水获得更多重叠空间。
- 让严格串行的阶段复用同一片UB，避免为不同阶段同时保留互不重叠的buffer。

A6的Dispatch、Activation和Combine虽然由同一个AIV执行，但它们的Vector计算仍然只能串行。384KB UB不能让三段Vector代码同时执行，只能通过独立buffer和Event管理，让某一阶段的MTE搬运与另一阶段的Vector计算适度重叠。

第一版必须继续保持以下边界：

- 采用A5已验证的A8W4 token-grained Wave骨架和目标负载；384KB优化不再改变M-group边界或二维游标推进规则。
- 不修改Dispatch使用的256行分组语义。
- 不修改GMM1/GMM2的scheduler tile和AIC分核方式。
- 不修改Activation到GMM2、GMM2到Combine的同步粒度。
- 不因为UB增大而把GMM `TileM`改成384；GMM tile同时受L1、L0、MMAD和同步协议约束。
- 不尝试把整个专家常驻UB；专家token数动态且数据需要跨物理核心消费。

### 9.2 第一版启用：按平台UB容量生成自适应buffer配置

第一版只启用不改变Kernel控制流和同步语义的自适应UB优化。

Host Tiling必须通过平台接口读取arch92实际可用UB容量，并将其作为 `availableUbBytes` 传给公共buffer配置计算。禁止在Host或Kernel中直接写死 `384 * 1024`；实际可用上限以平台返回值、编译器保留空间和对齐要求为准。

沿用A5现有自适应策略后，384KB可自动用于：

1. Dispatch优先增加copy ring深度，最大仍受Event ID数量约束为6；达到ring上限后，用剩余UB扩大 `routeItemsPerBatch`，减少route batch循环。
2. SendMask优先增加mask ring深度；达到ring上限后扩大route batch，减少Gather、mask生成和发送批次。
3. Unpermute优先增加输入ring深度；达到ring上限后扩大 `tokensPerBatch` 和topk weight batch，减少后处理批次。
4. Reset继续使用固定上限的分批清零策略，不因UB增加而扩大Workspace或改变flag布局。

arch92 Host复用公共Tiling实现时，必须确认以下字段使用A6的1C1V物理数量重新计算，而不是继承A5的 `blockAivNum = 2 * aicNum`：

- 每个AIV负责的token数和专家数。
- Dispatch/SendMask的实际批次数。
- Unpermute的full/tail token chunk。
- buffer数达到Event上限后可用于扩大batch的剩余UB预算。

该优化不会改变Workspace、Win区、GM数据布局、Wave边界和阶段间flag，可以直接进入第一版。第一版的GMM1/Activation和GMM2/Combine主流水仍可继续使用A5的256KB安全UB布局，暂不消费额外128KB。

### 9.3 后续TODO：GMM1输出到Activation的128行双缓冲

A6采用A8W8 Generic GMM1写GM、AIV从GM读取并执行SwiGLU/量化。一个非交织的256行Activation输入包含两块BF16数据：

```text
gate = 256 * 256 * sizeof(bfloat16_t) = 128KB
up   = 256 * 256 * sizeof(bfloat16_t) = 128KB
total                                     256KB
```

在256KB UB上，完整输入已占满主要空间，只能按“等待GMM1、GM2UB、Activation/量化、UB2GM、下一tile”串行推进。后续可保持AIC GMM1 tile为256行，仅把AIV消费粒度改为128行：

```text
one 128-row input bank = gate 64KB + up 64KB = 128KB
two input banks                                        = 256KB
remaining UB                                           = 128KB
```

目标流水为：

```text
MTE2搬入half 0
Vector计算/量化half 0  || MTE2搬入half 1
MTE3写回half 0         || Vector计算/量化half 1 || MTE2预取下一tile
```

实现约束：

- 只改变AIV从GM消费GMM1结果的子tile粒度，不改变GMM1 scheduler、Dispatch分组和Wave边界。
- 每个128行bank内部可继续利用生命周期覆盖：Activation结果覆盖已消费的gate区，量化结果覆盖已消费的up区。
- topk weights预取开关开启时，weight UB和metadata不能与任一活跃bank重叠。
- 两个half以及当前256行token group所需的全部N方向tile处理完成后，才能按既有协议发布GMM2-ready状态。
- 若改为按128行累计ready count，Host预留的flag数量、AIC等待目标和Reset范围必须同步更新；优先保持外部256行同步语义不变。
- 需要分别验证SwiGLU、SwiGLU-Step、SwiGLU-OAI、SituGLU和topk weights预取开关。

这是主流水中优先级最高的384KB专项优化，因为A6的GMM1到Activation必须经过GM中转，双缓冲有机会隐藏一部分GM2UB时间。

### 9.4 后续TODO：GMM2输出到非量化Combine的tile双缓冲

一个 `256 * 256` BF16 GMM2输出tile约为128KB。384KB可规划为：

```text
[0KB,   128KB)  GMM2 tile bank 0
[128KB, 256KB)  GMM2 tile bank 1
[256KB, 384KB)  metadata、ready scan和临时区
```

目标是在非量化Combine路径中重叠：

```text
MTE3发送当前GMM2 tile || MTE2搬入下一GMM2 tile
```

实现约束：

- 每个bank复用前必须等待该bank关联的MTE3操作完成。
- AIV仍按AIC发布的sequence消费tile，不能因预取而越过未完成的GMM2 tile。
- 当前 `META_INFO_TENSOR_ADDR = 200KB` 会与第二个128KB bank重叠，实施时必须通过arch UB layout policy把metadata移到256KB以上。
- 非量化Combine仍按当前GMM2 tile粒度启动，不改变其与AIC的对应关系。
- 量化Combine按完整token行读取GMM2结果，不直接套用该tile双缓冲设计。

### 9.5 后续TODO：前后处理阶段独占并复用UB

当前 `SendAndQuantBuffInit` 同时为Reset、Quant和SendMask建立互不重叠的UB视图，但三个阶段在主流程中严格串行。后续可拆成阶段级UB布局：

```text
Quant阶段     独占绝大部分UB
SendMask阶段  复用同一片UB
Reset阶段     再次复用同一片UB
```

这样可以避免Host在计算SendMask batch时长期扣除Quant和Reset的buffer预算，并支持以下优化：

1. Quant从“单token双缓冲”扩展为“多token batch双缓冲”，一次搬入连续token，减少逐token DataCopy和Event开销。
2. SendMask使用更大的route batch，减少Gather和mask发送循环。
3. Unpermute继续独占后处理阶段UB，根据384KB容量扩大输入ring或weight batch。

A6前处理可用AIV数是A5的一半，该优化在大BS场景可能获得更直接的端到端收益。实施前必须保证上一阶段所有异步MTE操作和Event均已回收，再重新解释同一片UB。

### 9.6 后续TODO：量化Combine ring深度与跨阶段MTE重叠

量化Combine当前使用双row buffer。后续应先通过profiling观察 `MTE3_MTE2` 等待占比，再决定是否扩大到3或4个slot：

- 只有MTE3回收等待明显时，增加slot才可能有收益。
- slot上限同时受MTE2_V、V_MTE3、MTE3_MTE2 Event ID数量约束。
- 扩大ring不能改变“所需GMM2 token全部ready后才启动量化Combine”的外部同步语义。

更激进的优化是把额外128KB作为Dispatch/Combine的MTE旁路buffer，使其搬运与SwiGLU的Vector计算重叠。但该方案需要：

- Dispatch、Activation和Combine使用互不重叠的UB区间。
- 为跨函数存活的异步操作分配独立Event集合。
- 将函数末尾的Event drain提升到上层Wave控制逻辑统一管理。
- 重新验证Dispatch领先水位、Activation对GMM2的阻塞和Combine完成时机。

该方案会改变阶段内部流水和Event生命周期，不进入第一版。

### 9.7 UB布局公共化要求

后续启用384KB专项优化前，应将以下固定地址或容量判断收敛为架构UB layout policy或模板参数：

- `block_epilogue_activation_mx_quant.h` 中的 `256U * 1024U` 容量判断。
- `mega_moe_gmm2_combine.h` 中的 `META_INFO_TENSOR_ADDR = 200KB`。
- Wave Combine的 `WAVE_COMBINE_UB_BASE = 64KB`。
- GMM2 ready scan的 `WAVE_GMM2_READY_SCAN_UB_ADDR = 184KB`。

A5 arch35 继续使用未修改的 256KB 布局，arch92 本地 stage 实例化 384KB 布局。后续对稳定 Blaze 头文件做 UB layout 抽象时，应使用独立兼容性 PR，不在 A6 迁移中修改 A5 热路径。

### 9.8 后续TODO：Dispatch cumsum常驻UB并删除无效策略分支

`cumsumInfoTensor`的实际容量为：

```text
Align32(moeExpertPerRank * epWorldSize * sizeof(int32_t))
= Align32(moeExpertNum * 4B)
```

当前Host约束`moeExpertNum <= 2048`，因此每个AIV所需的cumsum常驻区最多只有8KB。与A6新增的128KB UB相比，该容量很小，优先考虑从高地址UB划出独立持久区，使它在Dispatch、Activation、Combine串行切换期间不被覆盖。

目标布局和行为：

1. 由arch92 UB layout policy从平台报告的可用UB顶端预留`Align32(moeExpertNum * 4B)`，不要在Kernel中直接写死384KB地址。
2. Dispatch、Activation、Combine及后续384KB专项布局均不得占用该区；UB复用只发生在持久区以下。
3. `cumsumInfoTensor`在本轮算子执行期间始终指向该常驻区。专家`e`完成后，专家`e + 1`直接读取UB中的历史前缀，不再执行GM到UB恢复和UB到GM备份。
4. 最终`ExportExpertTokenCounts`直接读取常驻UB，不再以`ReloadCumsumFromGm=true`重新加载。
5. 确认所有路径都不再依赖GM副本后，从arch92 Workspace布局中删除`cumsumInfoPtr`及对应的`aicNum * Align32(moeExpertNum * 4B)`空间；Host和Kernel公式必须在同一次修改中更新。
6. 需要验证普通A8W8 Wave、topk weights预取、Combine 0/3/4、空专家、专家截断和最大`moeExpertNum=2048`场景，并检查所有stage的最大UB地址没有越过持久区边界。

这项优化不仅减少最多8KB的一次搬运。当前实现对专家`e`会恢复前`e`个专家的prefix并回写前`e + 1`个专家的prefix，累计搬运量随本卡专家数近似二次增长，而且每个Dispatch AIV都会重复执行。常驻UB可以同时消除这些MTE2/MTE3操作及其相关等待。

无论是否立即实施常驻UB，arch92都应收敛掉`PersistDispatchCumsum`模板维度：

- `MegaMoeArch92ArchitecturePolicy::PERSIST_DISPATCH_CUMSUM`当前恒为`true`，arch92也只保留A8W8路径，不存在第二种有效实例化。
- 现有`if constexpr (PersistDispatchCumsum)`不会产生运行时分支，因此单纯删除判断没有直接性能收益；收益是减少无效模板参数和错误配置空间。
- 若暂时继续落GM，arch92本地`ComputeExpertTokenCountAndNotify`直接无条件恢复/写回cumsum。
- 若启用常驻UB，则直接删除两次GM搬运和`cumsumInfoGlobalTensor`，而不是把策略常量改成`false`后继续保留双路径代码。

### 9.9 优化实施优先级

建议顺序如下：

1. 第一版：平台UB容量驱动的Dispatch、SendMask和Unpermute自适应配置。
2. 后续低风险优先项：预留最多8KB高地址持久区，使Dispatch cumsum常驻UB并删除GM副本。
3. 后续第一优先级：GMM1输出到Activation的128行双缓冲。
4. 后续第二优先级：非量化GMM2输出到Combine的tile双缓冲。
5. 后续第三优先级：前后处理阶段独占UB和Quant多token batch。
6. Profiling驱动：量化Combine扩大ring。
7. 最后评估：Dispatch/Activation/Combine跨阶段MTE重叠。

每项优化均应独立提交和测试，并同时记录AIC等待、AIV Vector占用、MTE2/MTE3等待及端到端耗时，避免将UB容量优化与Wave调度变化混在同一次验证中。

## 10. Workspace布局

### 10.1 设计原则

在 `arch92/common/mega_moe_workspace.h` 本地副本中增加行为命名的布局策略，例如：

```cpp
enum class WorkspaceLayoutPolicy {
    DEFAULT,
    GM_HANDOFF_EXPERT_WAVE
};
```

A5 Workspace 不修改；A6 Host Tiling 和 Kernel Init 都显式传入 `GM_HANDOFF_EXPERT_WAVE`。不要通过 `blockAivNum == aicNum` 猜测布局。

### 10.2 A6必需区域

| 区域 | 大小/布局 | 用途 |
|---|---|---|
| `dispatchRevDataPtr` | `Align512(maxOutputSize * h)` bytes | A8W8 Dispatch接收数据 |
| `dispatchRevScalePtr` | `Align512(maxOutputSize * ceil(h/32) * MXFP_MULTI_BASE_SIZE)` bytes | Dispatch接收scale |
| `dispatchRevWeightsPtr` | topk预取开启时按现有布局 | 可选topk weight记录 |
| `activationQuantDataPtr` | `Align512(maxOutputSize * hiddenDim/2)` bytes | SwiGLU量化输出 |
| `activationQuantScalePtr` | 按现有MXFP scale公式 | GMM2输入scale |
| `expertRevTokenNumsPtr` | `Align512(moeExpertPerRank * 32B * aicNum)` | 每专家、每物理block token数 |
| `metaInfoPtr` | `Align512(maxOutputSize * 32B)` | token来源rank/row/topk元数据 |
| `flagActivationToGmm2Ptr` | `moeExpertPerRank * 64B` | 每专家Activation完成计数 |
| `flagDispatchToGmm1Ptr` | `moeExpertPerRank * ceil(maxOutputSize/256) * 64B` | Dispatch m-group就绪行数 |
| `flagSendCntCalToUpdParamsPtr` | `moeExpertPerRank * aicNum * 64B` | 专家token数发布 |
| `flagGmmToEpiloguePtr` | `aicNum * 64B` | GMM1/GMM2到配对AIV的单调sequence |
| `gmm2CombineSyncCounterPtr` | 仅combine量化时分配 | GMM2 token-group计数 |
| `cumsumInfoPtr` | `aicNum * Align32(moeExpertPerRank * worldSize * 4B)` | Dispatch cumsum持久化 |
| `gmm1MmadResPtr` | `maxOutputSize * hiddenDim * sizeof(BF16)` | GMM1 GM handoff |
| `gmm2MmadResPtr` | `maxOutputSize * h * sizeof(BF16)` | GMM2 GM handoff |
| `gmm1TileStatusPtr` | 仅topk预取路径按现有上限分配 | GMM1 tile round tag |
| 共享专家区域 | 按A5现有公式 | 共享专家输入、GMM1/Activation/GMM2结果 |

### 10.3 A6不需要的区域

- A5 A8W8 Wave专属的 `gmm2ReadyPtr[expert][aic]` 全专家完成数组。
- 所有URMA专属的 `maskSlotPtr`、`dispatchL1CommPtr`、`dispatchCursorPtr`、`dispatchDonePtr` 和 `dispatchL2CommPtr`。

### 10.4 Reset区域必须同步更新

`WorkspaceInfo`中连续flag区、Host的 `CalcHostFlagElementCount` 和Kernel的 `CalcResetFlagElementCount` 必须使用同一布局策略。新增、删除或改变任何flag槽时，三处必须一起修改，否则可能出现未清零旧flag导致越过等待，或Reset越界覆盖后续workspace。

### 10.5 对齐要求

- 所有独立flag/sequence/counter槽继续独占64B cache line。
- 大型GMM输出至少保持512B区域对齐。
- Host和Kernel必须调用同一个Workspace布局实现，禁止分别手写偏移。
- `workspaceSize`仍叠加系统workspace和当前保留的50MB `RESERVED_WORKSPACE_SIZE`，除非平台规范明确要求调整。

## 11. Win区布局

A6只使用MTE布局，保持：

```text
base
├─ [0, 60KB) rank同步及本地sequence保留区
├─ maskRecv
│  └─ Align512(moeExpertPerRank * epWorldSize * (maskAlignSize + 32B))
├─ quantTokenScale
│  └─ bs * tokenScaleBytes
└─ combineSend
   └─ bs * topK * combineRecordBytes
```

其中：

```text
maskAlignSize = Align32(Align256(bs * topK * sizeof(int32)) / sizeof(int32) / 8)
tokenScaleBytes = Align32(Align256(h) + Align32(ceil(h/32)))
topk预取开启时，tokenScaleBytes额外包含Align32(topK * sizeof(float))
```

非量化Combine记录：

```text
combineRecordBytes = h * sizeof(OutputType)
```

量化Combine记录：

```text
tokenBytes = Align256(h)
scaleBytes = Align(ceil(h/32), MXFP_MULTI_BASE_SIZE)
combineRecordBytes = Align32(tokenBytes + scaleBytes)
```

注意事项：

- Win布局本身不依赖1C2V，因此可复用。
- SendMask、跨rank同步的任务分配依赖实际AIV数，相关stride必须改为 `aivNum`。
- ACLNN传入的 `cclBufferSize` 校验必须继续使用与Kernel完全一致的公式。
- A6 Tiling必须拒绝URMA，不能只在tiling key中默默落到MTE而保留URMA布局校验。

## 12. A5/A6 源码隔离方案

### 12.1 总原则

1. `arch35` Kernel 和 Host tiling 必须与 A6 迁移前 HEAD 保持一致。
2. 只要 A6 需要修改文件内的同步、分核或 Workspace 语义，就将该文件复制到 `arch92` 后修改。
3. 不需要修改的类型、工具、Blaze MMAD/Epilogue、Quant、SendMask、Shared input 和 Unpermute 仍直接引用 arch35 头文件。
4. arch92 不包含 A5 `mega_moe.h` 或 `mega_moe_wave_a8w4.h`，避免把 1C2V 主类和 A8W4 模板带入 A6。
5. 同步等待/通知与纯计算函数保持分层；Dispatch 自身完成定义中的 Wave 计数发布除外。

### 12.2 arch92 本地副本

以下文件从 arch35 复制后在 arch92 内独立演进：

- `mega_moe.h`
- `common/mega_moe_workspace.h`
- `stage/mega_moe_workspace_reset.h`
- `stage/mega_moe_token_dispatch.h`
- `stage/mega_moe_gmm1_activation.h`
- `stage/mega_moe_gmm2_combine.h`
- Host `op_tiling/arch92/mega_moe_tiling.cpp`

Workspace 本地副本增加：

- 架构无关的 `WorkspaceLayoutPolicy`。
- A6 GM handoff Wave布局构造入口。
- A6需要的 `flagGmmToEpilogue/cumsum/gmm1MmadRes/gmm2MmadRes/group-counter` 条件。
- Host 与 Kernel 都显式使用 `GM_HANDOFF_EXPERT_WAVE`。
- A5 Workspace 构造函数不修改。

### 12.3 仍直接复用的稳定头文件

- `common/mega_moe_constants.h`
- `common/mega_moe_types.h`
- `common/mega_moe_utils.h`
- `common/mega_moe_gmm_common.h`
- `common/mega_moe_mxfp8_utils.h`
- `stage/mega_moe_token_quant.h`
- `stage/mega_moe_send_mask.h`
- `stage/mega_moe_shared_expert_input.h`
- `stage/mega_moe_unpermute.h`
- Generic `block_scheduler_swizzle.h`
- Generic A8W8 BlockMmad，前提是arch92编译接口兼容

`arch92` 必须先包含本地 `common/mega_moe_workspace.h`，再包含 arch35 `mega_moe_types.h`。两个 Workspace 头文件保留相同 include guard，藉此使稳定公共头文件在 A6 翻译单元中解析到 arch92 Workspace 定义。

### 12.4 隔离验证标准

- `git diff -- mc2/mega_moe/op_kernel/arch35 mc2/mega_moe/op_host/op_tiling/arch35` 必须为空。
- arch92 不得包含 `../arch35/mega_moe.h` 或 `mega_moe_wave_a8w4.h`。
- arch92 活跃路径不得使用 `GetBlockIdx() / GetTaskRation()` 映射物理 block。
- Host tiling 使用 arch92 本地 Workspace 和 tiling key，不通过宏复用 arch35 `.cpp`。

## 13. arch92 Kernel目录与文件职责

当前第一版结构：

```text
mc2/mega_moe/op_kernel/
├─ mega_moe_apt.cpp
└─ arch92/
   ├─ mega_moe.h
   ├─ mega_moe_a8w8_wave.h
   ├─ mega_moe_tiling.h
   ├─ mega_moe_tiling_key.h
   ├─ common/
   │  ├─ mega_moe_core_context.h
   │  └─ mega_moe_workspace.h
   └─ stage/
      ├─ mega_moe_workspace_reset.h
      ├─ mega_moe_token_dispatch.h
      ├─ mega_moe_gmm1_activation.h
      └─ mega_moe_gmm2_combine.h
```

### 13.1 `op_kernel/mega_moe_apt.cpp`

- 入口文件必须直接位于 `op_kernel`，即 `arch92` 的上级目录，以匹配 A6 编译仓扫描 Kernel 源文件的目录约定；放在 `arch92` 内不会进入 A6 Kernel 编译。
- 通过 `arch92/` 显式包含 `mega_moe_a8w8_wave.h`、`mega_moe_tiling.h` 和 `mega_moe_tiling_key.h`，不依赖编译命令额外注入 `arch92` 目录。
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)`。
- 不提供Weight1 Interleave编译开关；A6固定实例化非交织Generic GMM1。
- 注册与arch92 Host一致的TilingData。
- 只实例化MTE。
- 只接受以下A8W8组合：
  - Dispatch E5M2 + Weight E5M2。
  - Dispatch E4M3FN + Weight E4M3FN。
- Combine支持NoQuant、E5M2、E4M3FN。
- 不包含Layered/URMA入口，不包含A8W4/A4W4 fallback。

### 13.2 `mega_moe.h`

负责：

- 参数和TensorList地址绑定。
- 1C1V核心上下文初始化。
- arch92 本地 stage 的 Config/Scratch 绑定。
- Phase 1、共享专家、跨卡同步、导出count、共享专家GMM2和Unpermute。
- 保留从 A5 复制而来的公共初始化框架，但核心数量、Workspace 布局和 stage 调用均按 1C1V 改造。

不建议直接继承arch35 `MegaMoe`，因为其中包含大量1C2V成员初始化和subBlock参与判断。

> 实施更新：arch92 不继承、不包含 arch35 `MegaMoe`，而是使用本地副本作为基类。上述“不建议直接继承”已落实为强制的源码隔离约束。

### 13.3 `mega_moe_a8w8_wave.h`

负责：

- Wave边界计算。
- `gmm1Position/waveBeginPosition/waveEndPosition/dispatchPosition`二维游标。
- Dispatch、GMM1、Activation、GMM2和Combine调用顺序。
- 所有跨stage等待与通知。
- AIC/AIV scheduler cursor和GM sequence状态。

业务函数命名使用：

```text
MegaMoeA8W8Wave
RunGmm1ToGm
RunActivationFromGm
RunGmm2ToGm
RunCombineFromGm
PipelineJobContext
```

### 13.4 `common/stage/blaze`

- `common/stage` 只复制 A6 需要改变行为的文件。
- 若只是转发 A5 公共类型，不创建无内容的重复文件。
- 当前 Generic A8W8 Blaze 头文件直接复用 arch35；只有 arch92 指令、L1 策略或 BlockMmad 接口不兼容时才新增 `blaze` 本地实现。
- 新文件名称继续以 `mega_moe_` 开头。

## 14. arch35头文件跨arch92复用的编译问题

### 14.1 芯片宏

当前公共头文件使用：

```cpp
__DAV_C310__
__DAV_C310_CUBE__
__DAV_C310_VEC__
```

例如：

- `common/mega_moe_workspace.h` 只在C310宏下定义Kernel侧 `PeermemInfo`。
- `common/mega_moe_utils.h` 和 `stage/mega_moe_workspace_reset.h` 使用C310宏控制Kernel实现。
- `blaze/epilogue/block_epilogue_activation_mx_quant.h` 整体受 `__DAV_C310__` 控制。

arch92 不能假设这些宏仍会定义，也不能为了兼容 A6 而原地修改 A5 头文件。实际 A6 工具链验证时：

- 若稳定 Blaze/工具头在 arch92 上可直接编译，继续跨目录复用。
- 若某头文件确实被 C310 宏限制，将该文件复制到 arch92 并使用行为能力条件，不修改 arch35。
- 不得伪造 `__DAV_C310__`，否则可能让底层库错误选择 A5 指令实现。

### 14.2 Include和打包

构建系统默认只复制当前SoC映射到的arch目录。arch92直接include `../arch35/...` 时，源码树编译可能成功，但生成binary目录或安装包中可能没有arch35头文件。

需要：

- CMake 同时安装 MegaMoe `arch92` 目录和已有 `arch35` 目录，保证稳定头文件可见。
- 只复制 A6 需要修改的文件，不把 arch35 入口 `.cpp` 带入 arch92。
- 检查生成的 `build/binary/<soc>/src/mega_moe_apt` 目录中 arch92 本地副本和 arch35 稳定依赖是否完整。

## 15. Host Tiling方案

### 15.1 第一版复制 arch35 Tiling 后隔离修改

为保证 A5 零侵入，第一版将 arch35 Host Tiling 机械复制为 `op_host/op_tiling/arch92/mega_moe_tiling.cpp`，然后只在 arch92 副本中增加 A6 能力校验、1C1V 核数检查、Workspace layout 和精简 tiling key。

这会带来 Host 校验公式双份维护的成本，但不会改变 A5 源码或产物。后续若 A5/A6 公共 Tiling 演进频繁，可再以独立 PR 抽取 `op_tiling/common`，不在首版迁移中实施。

### 15.2 平台策略建议

```cpp
struct MegaMoePlatformPolicy {
    bool requireEqualCoreCounts;
    bool supportUrma;
    bool supportA8W4;
    bool supportA4W4;
    WorkspaceLayoutPolicy workspaceLayout;
    PipelineMode pipelineMode;
};
```

A6策略：

```text
requireEqualCoreCounts = true
supportUrma = false
supportA8W4 = false
supportA4W4 = false
workspaceLayout = GM_HANDOFF_EXPERT_WAVE
pipelineMode = EXPERT_WAVE
```

命名描述行为，不描述“OneAiv”。

### 15.3 A6平台信息

Tiling读取：

```cpp
aivNum = ascendcPlatform.GetCoreNumAiv();
aicNum = ascendcPlatform.GetCoreNumAic();
```

检查二者相等后：

```cpp
tilingData->aicNum = aicNum;
tilingData->blockAivNum = aivNum;
context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
context->SetScheduleMode(1);
```

### 15.4 A6能力校验

必须接受：

- `x/y = BF16`，保持A5当前A8W8约束。
- Dispatch Quant Mode = MXFP。
- Dispatch dtype = E5M2或E4M3FN。
- Weight1/Weight2 dtype与Dispatch dtype一致。
- General ND和A5当前支持的A8W8_NZ组合。
- Combine NoQuant/E5M2/E4M3FN。
- `swiglu`与`situglu`及其现有activationParams语义。
- topk weight关闭/开启。
- 共享专家0到4。
- 每专家TensorList和堆叠Tensor布局。
- 非交织Weight1布局：`[全部Gate列][全部Up列]`。
- xActiveMask、maxRecvTokenNum、numMaxTokensPerRank等A5现有A8W8参数。

必须拒绝：

- `topoType != TOPO_TYPE_MTE`。
- FP4 weight或FP4 Dispatch输出。
- A8W4/A4W4/A4W4_NZ groupedMatmulMode。
- A5本身未支持的dtype、format、activation和通信算法。

### 15.5 Tiling key

arch92使用独立的精简 `mega_moe_tiling_key.h`：

- Dispatch mode只含MXFP。
- Dispatch输出只含E5M2/E4M3FN。
- Combine含0/3/4。
- 通信模式只含MTE，或直接去掉通信维度。
- 保留topk weights bool。

由于arch92二进制本身已隔离1C1V，不需要把核心比例加入tiling key。

### 15.6 A6专属计算

- `combineSyncSlotCountPerExpert` 的逻辑Job数使用 `aivNum`，不能 `/2`。
- combine非量化不分配group counter。
- combine量化按 `max(maxTokenGroupCountForOneExpert, aivNum)` 预留slot。
- Dispatch flag每专家按 `ceil(maxOutputSize/256)` 个cache line分配。
- Activation flag每专家一个cache line。
- 分配 `flagGmmToEpilogue`、cumsum、GMM1/GMM2 GM结果。
- 不分配A5 A8W8 Wave `gmm2Ready`。
- `mGroupsPerWave` 对A6专家Wave不参与边界计算，可保留字段兼容但不能误用。

## 16. ACLNN、OpDef与InferShape

### 16.1 ACLNN

`aclnnMegaMoeGetWorkspaceSize/aclnnMegaMoe` 接口不新增arch92版本，继续调用同一个InnerOp。

需要审计：

- 当前只有 `DAV_2201` 使用 `ACL_UINT64` weight scale，A6应使用 `ACL_FLOAT8_E8M0`。
- 建议将判断封装为“是否使用UINT64 scale”的显式能力函数，避免将“非arch22”隐式等同于A5。
- Int32到Int4 weight view转换仍只用于arch22，A6不能进入该分支。
- 可选共享专家dummy TensorList继续继承MoE weight dtype和format。
- ACLNN保留 `topoType` 参数，但A6的MTE限制由Tiling返回明确错误。

### 16.2 OpDef

A6 的正式 SoC 字符串已确认为 `ascend910_96`。首版已在 `mc2/mega_moe/op_host/mega_moe_def.cpp` 中增加：

```cpp
this->AICore().AddConfig("ascend910_96", aicore_config);
```

当前 `ascend910_96` 复用 `ascend950` 的 `aicore_config`，因此：

- `opFile.value` 仍为 `mega_moe_apt`。
- 保留 ACLNN、dynamic shape、multi-kernel 等现有配置。
- 共享专家输入、weight scale 和输出定义与 A5 A8W8 保持一致。
- 该修改只负责 OpDef 能力注册，不负责将 `ascend910_96` 映射到 `arch92`。

当前全局输入定义还包含 A5 的 FP4/A8W4 格式，A6 Host Tiling 会拒绝这些不支持组合。正式版若要使 SoC 能力元数据也严格收窄到 A8W8，应再新建 `aicore_config_arch92`，只声明 E5M2/E4M3FN 及对应 ND/NZ 组合。

### 16.3 InferShape/InferDataType

无需新增arch92实现：

- `y.shape = [bs, h]`
- `expert_token_nums.shape = [moeExpertNum / epWorldSize]`
- `y.dtype = x.dtype`
- `expert_token_nums.dtype = int32`

## 17. CMake、SoC映射和打包

正式接入 `ascend910_96` 时至少检查：

1. 根目录 `CMakeLists.txt`
   - `ASCEND_ALL_COMPUTE_UNIT`
   - `SOC_VERSION_LIST`
   - `ARCH_DIRECTORY_LIST`，映射到 `arch92`
2. `build.sh`
   - 支持SoC短名列表
   - 产品名映射和特殊分支
3. `scripts/util/soc_validator.sh`
4. `cmake/scripts/get_soc_version.py`
5. `cmake/scripts/util/opdesc_parser.py`
6. `cmake/scripts/util/const_var.py`
7. `scripts/ci/gen_ops_soc.py`
8. `scripts/ci/ops_st_test.sh` 和相关SoC到arch映射；本阶段不新增测试代码，但构建路由仍要可识别arch92
9. `mc2/mega_moe/op_host/CMakeLists.txt`
   - A6同样使用 `--cce-auto-sync=off`
   - 添加公共头文件include和copy依赖
10. `mega_moe_def.cpp` 的SoC注册

构建系统当前会根据arch映射过滤整个 `arch*` 目录，因此必须专门解决arch92引用arch35公共头文件时的打包问题，不能只在本地源码include路径上验证。

## 18. Blaze与指令兼容性

### 18.1 优先复用

优先复用：

- Generic A8W8 BlockMmad。
- BlockScheduler和swizzle。
- A8W8 ND/NZ DispatchPolicy。
- Activation MX Quant epilogue。

### 18.2 必须确认的接口

- GM到L1、L1到L0和FIX到GM接口在arch92是否保持兼容。
- `BlockMmad`写GM的Tensor layout是否一致。
- `SetFlag/WaitFlag` event和`FIX_S`可见性语义是否一致。
- `ReadGmByPassDCache/WriteGmByPassDCache`是否可用。
- `AtomicAdd`到GM cache-line flag是否保持一致。
- `SyncAll`和`KERNEL_TYPE_MIX_AIC_1_1`组合的参与核语义。
- `InitSocState`和Tensor API版本宏。
- `block_epilogue_activation_mx_quant.h` 的C310条件需要能力宏适配。

只有这些接口不兼容时，才在 `arch92/blaze` 新增对应文件。

## 19. 功能支持矩阵

| 功能 | A6初版 | 说明 |
|---|---|---|
| A8W8 E5M2 | 支持 | weight dtype一致 |
| A8W8 E4M3FN | 支持 | weight dtype一致 |
| A8W8_NZ | 支持 | 以A5现有E4M3FN NZ约束为准 |
| A8W4/A4W4 | 不支持 | Tiling和OpDef共同拒绝 |
| MTE | 支持 | 唯一通信拓扑 |
| URMA/Layered | 不支持 | 不实例化、不分配workspace |
| Combine非量化 | 支持 | 每物理block的GM sequence |
| Combine E5M2/E4M3FN量化 | 支持 | token-group counter |
| topk weights预取 | 支持 | 保留meta和tile status路径 |
| SwiGLU | 支持 | 参数语义与A5一致 |
| SwiGLU-Step | 支持 | 参数为空或`[clamp]`；先计算SiLU再执行上限裁剪 |
| SwiGLU-OAI | 支持 | 参数为空或`[clamp, alpha, beta]`；默认`alpha=1.702`、`beta=1.0` |
| SituGLU | 支持 | beta/linear_beta校验不变 |
| 共享专家0到4 | 支持 | Generic GMM路径 |
| xActiveMask | 支持 | 沿用A5前处理语义 |
| 每专家TensorList | 支持 | 地址解析不变 |
| 堆叠weight Tensor | 支持 | 地址偏移解析不变 |
| Weight1交织编译路径 | 不支持 | GM handoff下无UB双缓冲收益，arch92已删除该模板链路 |

## 20. 实施顺序

### 阶段A：建立 arch92 本地副本

1. 复制 A6 必须修改的 Workspace、Dispatch、GMM1、GMM2、reset 和主类文件。
2. 在 arch92 本地副本中加入 Workspace layout policy。
3. 在 arch92 本地 stage 中增加无 subBlock 的 Dispatch 入口、GM handoff GMM/Activation 入口和显式 Combine Job 数。
4. 将 arch35 本次迁移产生的所有差异恢复到 HEAD。

完成标准：arch35 Kernel/Host tiling 差异为空，arch92 不包含 A5 主类或 A8W4 Wave 主文件。

### 阶段B：Host Tiling 隔离

1. 机械复制 arch35 Host Tiling 到 `op_tiling/arch92`。
2. 在 arch92 副本中增加 1C1V、MTE-only、A8W8-only 和 dtype pair 校验。
3. 使用 arch92 本地 Workspace 布局和精简 tiling key。
4. 将 Host common 抽取保留为后续独立重构，不进入首版。

### 阶段C：arch92入口和前后处理

1. 新增arch92 tiling key和apt入口。
2. 设置1C1V Kernel task。
3. 完成 Init、核心上下文和 arch92 本地 stage 配置绑定。
4. 先打通Quant/SendMask/Reset/CrossRank/Unpermute空流水。

### 阶段D：专家Wave主体

1. 接入SendCount和Dispatch。
2. 接入A8W8 Generic GMM1到GM。
3. 接入Activation从GM和Activation ready计数。
4. 接入GMM2到GM。
5. 分别接入非量化Combine和量化Combine。
6. 接入expert token count导出。

### 阶段E：特性补齐

1. E5M2/E4M3FN。
2. General/NZ。
3. topk weights预取开关。
4. SwiGLU/SwiGLU-Step/SwiGLU-OAI/SituGLU。
5. 共享专家。
6. per-expert/stacked weight布局。

### 阶段F：构建和注册

1. SoC到arch92映射。
2. OpDef能力注册。
3. ACLNN平台dtype分支审计。
4. 公共头文件打包。
5. 自同步编译选项。

## 21. 验证计划

本阶段不新增test代码，但正式实现时至少执行以下验证。

### 21.1 编译检查

- arch35 A5全量模板编译。
- arch92 E5M2/E4M3FN、General/NZ、Combine 0/3/4、topk预取0/1组合编译。
- 确认没有C++11窄化、dependent template、宏重复和未声明芯片宏问题。
- 检查生成binary目录中arch35公共依赖齐全。

### 21.2 Tiling检查

- `aicNum == aivNum`成功；不相等明确失败。
- URMA明确失败。
- FP4明确失败。
- Host和Kernel Workspace总大小、每段偏移一致。
- CCL buffer最小值与PeermemInfo公式一致。
- Combine量化slot数与Kernel `combineJobCount`一致。

### 21.3 精度场景

- 小BS/大BS。
- token极度不均衡、空专家和单个超大专家。
- 尾Wave不足目标。
- 专家数小于、等于、大于核心数。
- Combine非量化、E5M2、E4M3FN。
- topk预取开/关。
- sharedExpertNum 0、1、4。
- SwiGLU/SwiGLU-Step/SwiGLU-OAI/SituGLU。
- General/NZ和两种weight布局。
- 非交织Weight1下确认Gate/Up地址、Wave边界和Activation完成计数一致。

### 21.4 同步稳定性

- 所有核心无timeout/trap。
- 无MTE DDR越界。
- Reset后所有flag为0。
- 多轮连续执行无上一轮sequence污染。
- 空专家不会导致AIC/AIV游标或sequence不一致。
- AIV因Dispatch变慢时，AIC只在正确flag上等待。

### 21.5 性能观察

- 分别记录Dispatch、GMM1、Activation、GMM2、Combine时间。
- 观察唯一AIV是否成为bound。
- 观察GMM2是否因Activation延迟长期自旋。
- 对比A8W4 Wave骨架目标值2和其他目标值，但初版不修改已验证值。
- 对比GM handoff额外带宽与避免UB/CV阻塞的收益。

## 22. 主要风险与应对

### 22.1 唯一AIV成为瓶颈

Dispatch、Activation和Combine全部串行在AIV上。该风险无法通过UB内并行解决，只能依靠Wave预取和GMM时间掩盖。初版优先保证正确流水，不调整Wave目标。

### 22.2 GMM2等待Activation

如果Dispatch或上一阶段Combine拖慢Activation，AIC可能在GMM2输入flag上等待。需要性能数据确认是否AIV已成为整体bound；不能为了减少自旋破坏Dispatch预取。

### 22.3 GM handoff带宽

GMM1和GMM2都写BF16 GM，增加带宽和workspace。它是单AIV复用UB、保持Dispatch/GMM重叠的必要代价。后续只有在arch92提供更合适的AIC/AIV直接通道时才重新评估。

### 22.4 sequence错位

AIC和AIV必须以相同scheduler顺序和相同专家边界推进 `startBlockIdx/gmTileSequence`。空专家、尾tile和topk预取分支都不能只让一侧增加sequence。

### 22.5 A5回归

A6 迁移不修改 arch35 Kernel 或 Host tiling，因此不应改变 A5 控制流、模板实例、Workspace 布局或寄存器占用。每次提交都要将 arch35 零差异作为强制检查。

### 22.6 Host/Kernel布局分叉

Workspace、Win区、flag数量和Combine Job数必须由共享策略计算。禁止arch92 Host和Kernel分别维护一份公式。

### 22.7 跨arch头文件未被打包

本地include成功不代表安装包成功。必须检查生成源码和最终包内容，显式声明arch35公共头文件依赖。

## 23. 最终评审结论

方案可行，推荐的实现形态是：

> A8W4 Wave的token-grained调度骨架 + A8W8 Generic MMAD + GMM结果GM中转 + 单调sequence/group-counter同步 + arch92独立Host/Kernel入口。

它能够在不保留 AIV0 Prologue 的情况下适配 A6 1C1V。A6 需要改变的 stage/common/Host tiling 采用 arch92 本地副本；仅对无架构行为差异的类型、工具、Blaze 和独立 stage 继续复用 arch35。A5 主流程和公共头文件保持零修改。

正式抽取前最先需要验证的三个技术前提是：

1. arch92 Generic A8W8 BlockMmad和Activation epilogue指令兼容性。
2. `KERNEL_TYPE_MIX_AIC_1_1` 下 `GetBlockIdx/GetTaskRation/SyncAll` 的实际语义。
3. 跨arch35公共头文件的构建、生成源码和安装包依赖是否完整。

## 24. ascend910_96黄区编译适配复盘

### 24.1 已确认的编译环境事实

黄区实际编译链路为：

```text
ascend910_96
    -> Ascend910_9691
    -> dav-920r1
    -> __NPU_ARCH__ = 9201
```

参考编译命令：

```bash
source /home/y00879306/0817/ascend/set_env.sh
bash build.sh --pkg --soc=ascend910_96 --ops=mega_moe
```

因此不能继续以 `__DAV_C310_CUBE__` 或 `__DAV_C310_VEC__` 是否定义作为arch92 Kernel代码可见性的唯一判断。arch92设备代码应以 `defined(__NPU_ARCH__)` 或由它统一派生的本地平台宏识别Kernel编译。

### 24.2 编译修改评审结论

以下修改均已经在A6黄区编译过程中验证能够解决对应问题。本节不再判断它们是否“能工作”，只评审复盘文本是否一致、当前写法是否便于长期维护，以及是否存在等价但更集中的组织方式。

| 分类 | 结论 | 正式处理方式 |
|---|---|---|
| A1 Host tiling头路径 | 已验证正确，黄区写法也是最直接方案 | 最终代码只在黄区维护时直接使用 `tiling_base/tiling_base.h` 最简洁；只有同一份Host源码还要回灌蓝区时才值得增加 `__has_include`。 |
| A2 ACLNN matmul util路径 | 已验证正确，黄区写法也是最直接方案 | 最终黄区源码直接使用 `common/op_host/op_api/matmul_util.h`；跨仓共用单份源码时再增加双路径兼容。 |
| B1 Kernel `-I` 路径 | 已验证正确 | 三条路径共同构成黄区实际编译闭包。即使v3没有出现在MegaMoe直接include中，也可能来自生成源码或传递依赖，不能仅凭源码文本删除。长期可优化为CMake target依赖集中传播include目录。 |
| C1 `quantize_functions.h` fallback | 已验证正确，但复盘描述自相矛盾 | fallback本身是最稳妥的源码树/生成binary双形态兼容。需要明确它究竟修改了arch35公共头，还是arch92本地副本；前者不能再写“arch35所有文件零修改”。 |
| D1-D4 Kernel宏扩展 | 已验证正确，可做维护性合并 | `defined(__NPU_ARCH__)` 是A6真正生效条件，额外 `__DAV_*` 分支保留其他环境兼容。最优组织是统一定义 `MEGA_MOE_IS_KERNEL_COMPILE`，但不应为了整理宏改变已验证行为。 |
| E1 Hcomm模板适配 | 已验证正确，当前是低侵入兼容方案 | 使用A6双模板参数Hcomm让共享 `Params` 结构继续可解析，且不改变任何可达MTE路径。删除死字段更纯粹，但会扩大types和URMA条件代码改动，不是首版必须优化项。 |
| F1-F11 arch92本地include | 已验证正确且必要 | 本地副本形成同一套arch92 types/utils/GMM common/epilogue闭包，避免混用arch35定义。这也是当前最清晰的依赖方向。 |
| G1 `Quant`命名空间别名 | 已验证正确，需校正文档中的可见性描述 | 别名必须在第一个解析 `Quant::` 的头文件之前可见。若实际代码已经编译通过，说明实际include顺序满足该条件；复盘中“约第49行、stage include之后”的描述需要按真实位置修正。 |
| H1 arch92 GMM common裁剪A8W4 | 已验证正确且最优 | arch92只支持A8W8，直接删除无法解析且永不实例化的A8W4类型，比继续制造占位声明更整洁。 |

总体结论：所有列出的修改都接受其已验证正确性。需要处理的不是功能回退，而是C1的“arch35零修改”表述冲突、G1的别名可见性描述，以及D类宏和构建include未来是否集中管理。

#### 24.2.1 复盘文本中的一致性问题

1. C1明确修改 `arch35/common/mega_moe_mxfp8_utils.h`，但“未修改的文件”又写“arch35所有文件零修改”。两者必须二选一。若实际修改发生在arch35，应将结论改为“除一处跨构建形态的公共include兼容修复外，arch35计算逻辑零修改”。
2. G1写明别名位于 `mega_moe.h` 约第49行，而当前include结构中stage头通常在此之前被解析。命名空间别名只有在首次出现 `Quant::` 之前声明才有效。既然修改已经编译验证，应以黄区真实include顺序重写该条，不能只记录近似行号。
3. B1的原因把 `mc2_kernel_utils.h` 与三条新增路径并列，但该头通常由已有 `mc2/common/op_kernel` include路径提供，不是新增三条路径之一。应区分“原有include闭包”和“本次新增include闭包”。
4. GMM/Blaze部分写“修改已逐项验证”，同时又写“整体尚未完成”，二者并不冲突，但需要表述为：每一步已验证消除对应编译错误，完整A8W8 BlockMmad链路仍在继续补齐。

### 24.3 Host和构建侧正式方案

黄区Host源码采用黄区现有公共头名称：

```cpp
#include "tiling_base/tiling_base.h"
#include "common/op_host/op_api/matmul_util.h"
```

当前最终编译仓是黄区，因此直接include是最优方案。只有同一份源码需要同时回灌蓝区时，才改为：

```cpp
#if __has_include("tiling_base/tiling_base.h")
#include "tiling_base/tiling_base.h"
#else
#include "op_host/tiling_base.h"
#endif
```

`matmul_util.h` 使用相同的双路径策略。

Kernel include目录保留已经验证的完整编译闭包：

```cmake
OPTIONS -I${CMAKE_SOURCE_DIR}/mc2/moe_distribute_dispatch_v2/op_kernel
OPTIONS -I${CMAKE_SOURCE_DIR}/mc2/moe_distribute_dispatch_v3/op_kernel
OPTIONS -I$ENV{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common
```

后续可通过CMake target依赖传播替代手工 `-I`，但优化前必须确认生成源码和传递依赖仍然得到完全相同的路径。include路径只解决头文件可见性，不能代替 `mega_moe_apt_depends` 对安装包和生成binary依赖的声明。

### 24.4 arch92本地头文件闭包

arch92本地复制并修改以下类别后，引用关系必须在arch92内部闭合：

```text
arch92/mega_moe.h
    -> arch92/common/mega_moe_types.h
    -> arch92/common/mega_moe_utils.h
    -> arch92/common/mega_moe_workspace.h
    -> arch92/common/mega_moe_gmm_common.h
    -> arch92/blaze/epilogue/block_epilogue_activation_mx_quant.h
```

GMM1、GMM2、workspace reset和token dispatch的arch92副本应优先include上述本地文件。仍然引用arch35的文件只允许是经过确认不包含架构宏、Hcomm签名、Workspace类型或A8W4符号差异的稳定公共实现。

`mega_moe_mxfp8_utils.h` 的 `<quantize_functions.h>` fallback已经验证有效。本实现选择arch92本地副本，并通过 `mega_moe_quant_compat.h` 统一处理相对路径和 `<quantize_functions.h>` fallback，因此arch35保持零修改。该兼容头在所有stage之前引入，也保证A6的 `namespace Quant = quant;` 早于第一个 `Quant::` 使用点可见。

### 24.5 Hcomm兼容处理

A6首版明确不支持URMA和Layered，但共享 `Params` 类型仍包含Hcomm指针。已经验证的低侵入方案是保留字段并适配A6模板签名：

```cpp
Hcomm<CommEngine::AIV, CommProtocol::ROCE> *hcomm;
```

该字段只负责让类型声明成立；A6所有可达MTE路径都不初始化、不解引用它，也不能据此宣称A6已经支持ROCE/URMA Combine。

后续完成arch92 URMA死代码整体裁剪时，可以再将结构收敛为：

```cpp
struct CombineCommParams {
    uint32_t rankId;
    __gm__ Mc2MoeContext *mc2Context;
};
```

但这属于代码整洁度优化，不应混入已经验证的首版编译适配。

### 24.6 arch92 GMM common裁剪原则

arch92 `mega_moe_gmm_common.h` 只保留A8W8类型系统：

```cpp
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
using BlockPrologue = void;
```

同时删除：

- `block_mmad_mx_fp8fp4.h` include；
- `block_prologue_mx_fp8fp4.h` include；
- `MatmulMxFp8Fp4DynamicKL1TailResplit`；
- A8W4 `BlockMmadSelector`；
- A8W4 `DefaultL1Params` 和动态tiling分支。

这不是功能裁减风险，因为arch92 Host tiling和Kernel入口已经共同拒绝A8W4/A4W4，相关模板不应在A6编译单元中存在。

### 24.7 A8W8 BlockMmad处理结论

以下修改均是有效的递进式编译修复：

1. 从A5 CANN整份复制 `block_mmad_qbmm_mx.h`、`block_mmad.h` 和arch35 tile工具；
2. 直接把 `__NPU_ARCH__ == 3510` 放宽为同时接受9201；
3. 持续补齐 `PadMxKAL1/PadMxKBL1` 等A5内部依赖。

它们逐步证明了原始失败不仅是一个架构宏问题，还包含A5 CANN内部tile工具在A6环境不可见的问题。复盘中“仍在处理中”表示完整依赖链尚未闭合，不代表前面各步修改无效。

从最终维护成本看，持续复制完整A5 CANN实现会带入较大的内部依赖闭包。更集中的方案是在仓内提供MegaMoe专用、接口最小化的arch92实现：

```text
arch92/blaze/gemm/block/mega_moe_block_mmad_mx_fp8.h
```

该文件只特化MegaMoe当前使用的：

```cpp
BlockMmad<MatmulWithScaleMx<>, ...>
```

并自行完成：

- FP8 A/B的GM到L1搬运；
- E8M0 ScaleA/ScaleB的独立双缓冲和 `scaleKL1` 复用；
- A/B/Scale的L1到L0搬运；
- `MmadTraitMX` Cube计算；
- L0C到GM或UB输出。

它继续依赖A6公开的底层AscendC能力，但不再依赖A5 CANN的完整 `BlockMmad<MatmulWithScaleMx>` 特化及其私有tile工具。若A6编译继续失败，错误应收敛到 `L0ScaleA/L0ScaleB`、`CopyL12L0ScaleA/B` 或 `MmadTraitMX` 等具体底层能力。

### 24.8 编译适配验证清单

完成上述调整后，至少检查：

1. 生成binary中的arch92文件不再include A8W4 prologue/MMAD头。
2. arch92本地文件不再意外回退到同名arch35 types/utils/common。
3. `__NPU_ARCH__=9201` 下Kernel实现可见，Host编译不解析设备专属代码。
4. `Quant::` 最终只映射到一个有效命名空间，不出现别名冲突。
5. A6 MTE路径可以解析Hcomm占位类型，但不初始化、不解引用、不调用Hcomm对象。
6. `BlockMmad<MatmulWithScaleMx<>>` 命中arch92本地特化，不再落入主模板 `static_assert`。
7. arch35工作区保持零差异；若接受公共include fallback例外，需要单独记录并执行A5回归编译。

# MegaMoE A6 编译修复总结

## 环境

- **编译命令**: `source /home/y00879306/0817/ascend/set_env.sh && bash build.sh --pkg --soc=ascend910_96 --ops=mega_moe`
- **CANN 版本**: 9.1.0（黄区）
- **代码路径**: `/home/y00879306/0817/ops-transformer/mc2/mega_moe/op_kernel/arch92/`

## 修复一：FP8 数据类型不在 AscendC 命名空间中

### 问题

编译报错 `no member named 'fp8_e4m3fn_t' in namespace 'AscendC'`、`no member named 'fp8_e5m2_t' in namespace 'AscendC'`。

### 原因

`fp8_e4m3fn_t`、`fp8_e5m2_t`、`fp8_e8m0_t` 是 CANN SDK 在 `c_api/` 头文件中定义的全局命名空间类型，不在 `AscendC` 命名空间中。但 `mega_moe_block_mmad_mx_fp8.h` 中使用了 `AscendC::fp8_e4m3fn_t`、`AscendC::fp8_e5m2_t`、`AscendC::fp8_e8m0_t` 带命名空间前缀的形式，导致编译失败。

### 影响文件

`op_kernel/arch92/blaze/gemm/block/mega_moe_block_mmad_mx_fp8.h`

### 修改

1. `static_assert` 中的 `AscendC::fp8_e4m3fn_t` / `AscendC::fp8_e5m2_t` 去掉 `AscendC::` 前缀
2. `sizeof(AscendC::fp8_e8m0_t)` 去掉 `AscendC::` 前缀
3. `MakeMemPtr<..., AscendC::fp8_e8m0_t>(...)` 去掉 `AscendC::` 前缀

共涉及约 10 处引用。

## 修复二：共享专家路径 CopyL0C2UB 参数不匹配（临时绕过）

### 问题

编译报错 `no matching function for call to 'asc_copy_l0c2ub'`，candidate 需要 21 个参数但调用传了 22 个。

### 调用链

```yaml
mega_moe_a8w8_wave.h:335  ProcessSharedExpertGmm1()
  → mega_moe.h:673        RunSharedExpertGmm1ActivationStage()
    → gmm1_activation.h:1361  RunGmm1Generic<..., true, ...>(共享专家, NZ权重)
      → BlockMmad<MatmulWithScaleMx<>> (CANN官方BlockMmad)
        → BlockEpilogue → CopyL0C2UB
          → CANN impl层 instruction.h:47  调用 asc_copy_l0c2ub(22参数)
          → CANN接口层 cube_datamove.h    声明 asc_copy_l0c2ub(21参数)
```

### 原因

CANN 9.1.0 内部实现层 (`impl/tensor_api/arch/cube/l0c_to_ub/copy_impl/instruction.h`) 与接口层 (`include/c_api/cube_datamove/cube_datamove.h`) 签名不一致：

- **instruction.h（impl层）**: 调用 `asc_copy_l0c2ub()` 传 22 个参数（第8位是 `bool sub_blockid`，第9位是 `uint8_t clip_relu_pre`）
- **cube_datamove.h（接口层）**: 部分类型重载只有 21 个参数（缺少 `sub_blockid`，第7位是 `bool quant_pre_rnd`）

该问题仅在共享专家的 `RunGmm1Generic<QuantPre=QF322BF16_PRE>` 路径触发，moe 专家路径未触发。

### 影响文件

`op_kernel/arch92/mega_moe_a8w8_wave.h`

### 修改（临时绕过）

```cpp
// 第 334-336 行，注释掉共享专家 GMM1 调用
if (sharedExpertNum_ > 0U) {
    // FIXME: CANN 9.1.0 CopyL0C2UB parameter mismatch, skip for now
    // ProcessSharedExpertGmm1(gmTileSequence);
}
```

### 注意

- 这是 CANN 9.1.0 内部 SDK 版本兼容性问题，需升级 CANN 后恢复
- moe 专家路径（`ProcessMoeExpertStages`）未受影响，编译通过

## 修复三：同步A5新增门控激活

arch92 Host Tiling和Kernel现已同步支持：

- `swiglu`：参数为空或`[clamp]`。
- `swiglustep`：参数为空或`[clamp]`，先计算SiLU再执行上限裁剪。
- `swigluoai`：参数为空或`[clamp, alpha, beta]`，默认`alpha=1.702`、`beta=1.0`。
- `situglu`：保留原有一个或两个非零有限参数的校验语义。

空参数列表允许`GetData()==nullptr`。Kernel激活实现位于arch92本地
`blaze/epilogue/gated_activation.h`，不引用A5私有CANN实现；Epilogue继续保留A6的
`MEGA_MOE_IS_KERNEL_COMPILE`保护、Gate/Up双区tail-N固定`TileN`行距及现有UB/GM交接逻辑。

## 本次工作区静态审查

已确认：

1. arch35 Kernel和Host Tiling保持零差异。
2. `__NPU_ARCH__=9201`可通过`MEGA_MOE_IS_KERNEL_COMPILE`看到设备实现，Host不会解析MicroAPI代码。
3. arch92入口与Host能力校验仅允许MTE A8W8 E5M2/E4M3FN匹配组合，不存在可达的A8W4/A4W4或Layered回退。
4. FP8类型使用全局命名空间，本地`BlockMmad<MatmulWithScaleMx<>>`特化在选择器实例化前可见。
5. `Quant`兼容别名早于arch92 MXFP8辅助函数使用，Hcomm字段在A6 MTE可达路径中不被解引用。
6. Host、Workspace构造和Reset对group-grained activation flag使用相同容量公式。
7. `git diff --check`、冲突标记和arch35差异检查通过。

本机没有A6 CANN黄区编译环境，因此以上是源码静态审查结论；仍需在黄区执行完整模板编译。
共享专家GMM1当前仍受前述CopyL0C2UB临时绕过限制，不能把“编译通过”解释为共享专家功能已恢复。
