# 重构报告：GMM 模块化拆分与 PostOp 回调机制引入

## 总体目标

将 `mega_moe_impl.h` 中 `Detail` 命名空间内耦合的 GMM1/GMM2 计算与 AIV 后处理逻辑（SwiGLU、Combine）拆分为**纯计算模块**，通过 PostOp functor 回调模式实现 AIV 后处理的可配置注入。消除模块内部对 `GetBlockIdx()/GetBlockNum()/GetSubBlockIdx()` 的硬依赖，改为显式参数传递（`tileIdx/totalCoreNum/subTileIdx`），使模块可独立测试与复用。

## 文件清单

| 原始文件 | 重构后文件 | 说明 |
|---|---|---|
| `mega_moe_impl.h` (Detail 命名空间) | `gmm_modules/gmm_common.h` | 参数化共享 GMM 基础设施 |
| `mega_moe_impl.h` → `GroupMatmulSwigluQuant` / `GroupMatmulSwigluQuantA8W4` | `gmm_modules/gmm1_module.h` | 纯 GMM1 模块 |
| `mega_moe_impl.h` → `GroupMatmul2` / `GroupMatmul2CombineA8W4` | `gmm_modules/gmm2_module.h` | 纯 GMM2 模块 |
| `mega_moe.h` (原始编排) | `gmm_modules/mega_moe.h` | 编排层 + PostOp functor |
| `mega_moe_layered.h` (原始编排) | `gmm_modules/mega_moe_layered.h` | 同上 + IsLayered 支持 |

---

## 各文件变更详情

### 1. gmm_common.h

**替代了 `mega_moe_impl.h` 的 `MegaMoeImpl::Detail` 命名空间核心基础设施**

#### 删除项

| 旧名称 | 旧位置 (mega_moe_impl.h) | 说明 |
|---|---|---|
| `AivGmm1PostGeneric` | Detail 命名空间, 行410-433 | 硬编码 SwiGLU+Quant 后处理，被 `AivPostLoop` + PostOp functor 替代 |
| `AivGmm2PostGeneric` | Detail 命名空间, 行435-474 | 硬编码 CombineTokens 后处理（含 IsLayered 分支），被 `AivPostLoopPingPong` + PostOp functor 替代 |
| `AivGmm1PostA8W4` | Detail 命名空间, 行722-766 | A8W4 GMM1 SwiGLU 后处理，被 `mega_moe.h` 中的 `Gmm1SwigluPostOpA8W4` functor 替代 |
| `AivGmm2PostA8W4` | Detail 命名空间, 行768-800 | A8W4 GMM2 Combine 后处理，被 `mega_moe.h` 中的 `Gmm2CombinePostOpA8W4` functor 替代 |
| `Detail::Gmm1ArgsGeneric<SwigluQuantOp>` | 行477-480 | 以 `SwigluQuantOp& swigluQuantOp` 为成员，现改为 `GmmParam::Gmm1ArgsGeneric<PostOp>` 以 `PostOp& postOp` 为成员 |
| `Detail::Gmm2ArgsGeneric` (含 `groupCnt`) | 行482-487 | `groupCnt` 成员移至 PostOp 内部保存，`Gmm2ArgsGeneric<PostOp>` 仅保留 `PostOp& postOp` + 同步字段 |
| `GroupMatmulSwigluQuant` (自由函数) | 行593-605 | 外部入口函数，由 `gmm1_module.h` 替代 |
| `GroupMatmul2` (自由函数) | 行610-622 | 外部入口函数，由 `gmm2_module.h` 替代 |
| `GroupMatmulSwigluQuantA8W4` (自由函数) | 行906-916 | 外部入口函数，由 `gmm1_module.h` 替代 |
| `GroupMatmul2CombineA8W4` (自由函数) | 行919-930 | 外部入口函数，由 `gmm2_module.h` 替代 |

#### 新增项

| 新名称 | 新位置 (gmm_common.h) | 说明 |
|---|---|---|
| `GmmParam` 命名空间 | 行39 | 替代 `MegaMoeImpl::Detail`，独立命名空间 |
| `AivPostLoop<PostOp>` | 行432-442 | 泛化 AIV 后处理循环，每 tile 调用 `postOp(workSet, loopIdx)`，替代 `AivGmm1PostGeneric` |
| `AivPostLoopPingPong<ElementC, PostOp>` | 行444-464 | 泛化 AIV pingpong 后处理循环，每 tile 调用 `postOp(workSet, loopIdx, l0cOutUb)`，替代 `AivGmm2PostGeneric` |
| `Gmm1ArgsGeneric<PostOp>` | 行284-288 | 以 `PostOp& postOp` 替代 `SwigluQuantOp&` |
| `Gmm2ArgsGeneric<PostOp>` | 行290-296 | 以 `PostOp& postOp` 替代硬编码 groupCnt |
| `Gmm1ArgsA8W4<PostOp>` | 行298-302 | A8W4 路径，以 `PostOp& postOp` 替代 `SwigluQuantOp&` |
| `Gmm2ArgsA8W4<PostOp>` | 行304-308 | A8W4 路径，以 `PostOp& postOp` 替代硬编码 groupCnt |
| `NotifyGmm2TileGroupComplete` | 行108-125 | 重命名自 `NotifyCombineTileComplete`，语义更清晰 |
| `GroupMatmulImplGenericParam` | 行492-547 | 新签名：显式 `tileIdx, totalCoreNum, subTileIdx` 参数 |
| `GroupMatmulImplA8W4Param` | 行657-710 | 新签名：显式 `tileIdx, totalCoreNum, subTileIdx` 参数 |
| `GroupMatmulExecA8W4Param` | 行632-655 | 新签名：显式 `subTileIdx` 参数 |
| `Config::BuildProblemConfigParam` | 行233-257 | 新签名：`problemShape, tileIdx, totalCoreNum`，不再内部调用 `GetBlockNum()/GetBlockIdx()` |

#### 变更项

| 项目 | 旧版 | 新版 | 说明 |
|---|---|---|---|
| 命名空间 | `MegaMoeImpl::Detail` | `GmmParam` | 独立命名空间，与编排层解耦 |
| `ComputeCoreGrouping` | `MegaMoeImpl::ComputeCoreGrouping` | `GmmParam::ComputeCoreGrouping` | 命名空间迁移（Bug 修复：原版编排层调用 `MegaMoeImpl::ComputeCoreGrouping` 现改为 `GmmParam::ComputeCoreGrouping`） |
| `NotifyCombineTileComplete` | `MegaMoeImpl::NotifyCombineTileComplete` | `GmmParam::NotifyGmm2TileGroupComplete` | 重命名，语义更准确 |
| `GroupMatmulExecGeneric` AIV 分支 | 调用 `AivGmm1PostGeneric` 或 `AivGmm2PostGeneric` | 调用 `AivPostLoop(args.postOp)` 或 `AivPostLoopPingPong(args.postOp)` | PostOp 回调替代硬编码后处理 |
| `GroupMatmulExecA8W4Param` AIV 分支 | `GetSubBlockIdx()` 判断 sub0/sub1 并调用 `AivPrologueA8W4` 或 `AivGmm1PostA8W4/AivGmm2PostA8W4` | 显式 `subTileIdx` 参数；sub0 调 Prologue，sub1 调 `args.postOp(workSet, config, params, ...)` | 消除 `GetSubBlockIdx()` 硬依赖 |
| Config 常量引用 | `L1_TILE_K` 等（同命名空间内直接使用） | `MegaMoeImpl::L1_TILE_K` 等（带命名空间前缀） | 因命名空间迁移，需显式引用 `MegaMoeImpl::` 下的常量 |
| `GroupMatmulImplGeneric` → `GroupMatmulImplGenericParam` | 参数：`(params, problemShape, gmmAddrInfo, startBlockIdx, args)` | 参数：`(tileIdx, totalCoreNum, subTileIdx, params, problemShape, gmmAddrInfo, startBlockIdx, args)` | 消除 `GetBlockIdx()/GetBlockNum()/GetSubBlockIdx()` |
| `ProblemConfig::blockNum/blockIdx` | 从 `GetBlockNum()/GetBlockIdx()/GetTaskRation()` 获取 | 从显式参数 `totalCoreNum/tileIdx` 获取 | 解耦硬件 API |
| subBlockIdx 判断 | `GetSubBlockIdx() != 0` | `subTileIdx != 0` | 显式参数替代硬件 API |

#### 迁移项

| 代码 | 从 | 到 | 说明 |
|---|---|---|---|
| `Gmm1Policy`, `Gmm2Policy` | `MegaMoeImpl::Detail` | `GmmParam` | 策略结构体，逻辑不变 |
| `BlockMmadSelector` | `MegaMoeImpl::Detail` | `GmmParam` | BlockMmad 选择器，逻辑不变 |
| `Config` (全量结构体) | `MegaMoeImpl::Detail` | `GmmParam` | 统一配置结构体，逻辑基本不变（`BuildProblemConfig` → `BuildProblemConfigParam` 签名变更） |
| `WorkSetGeneric` | `MegaMoeImpl::Detail` | `GmmParam` | WorkSet 携带结构体，逻辑不变 |
| `WorkSetA8W4` | `MegaMoeImpl::Detail` | `GmmParam` | A8W4 WorkSet，逻辑不变 |
| `AicComputeGeneric` | `MegaMoeImpl::Detail` | `GmmParam` | AIC 计算循环，逻辑不变（常量前缀变更） |
| `AicComputeA8W4` | `MegaMoeImpl::Detail` | `GmmParam` | AIC A8W4 计算循环，逻辑不变 |
| `AivPrologueA8W4` | `MegaMoeImpl::Detail` | `GmmParam` | AIV Prologue，逻辑不变 |
| `WaitForUpstreamReady` | `MegaMoeImpl` (自由函数) | `GmmParam` | 上游等待函数，逻辑不变（参数类型 `GMMAddrInfo` 前缀改为 `MegaMoeImpl::GMMAddrInfo`） |
| `ComputeCoreGrouping` | `MegaMoeImpl` (自由函数) | `GmmParam` | 核分组函数，逻辑不变 |
| `ComputeGroupRange` | `MegaMoeImpl` (自由函数) | `GmmParam` | 核范围计算，逻辑不变 |
| `GroupMatmulExecGeneric` | `MegaMoeImpl::Detail` | `GmmParam` | 执行层，AIV 分支由硬编码改为 PostOp 回调 |

---

### 2. gmm1_module.h

**纯 GMM1 矩阵乘法模块，替代 `GroupMatmulSwigluQuant` / `GroupMatmulSwigluQuantA8W4`**

#### 删除项

| 旧名称 | 说明 |
|---|---|
| `MegaMoeImpl::GroupMatmulSwigluQuant` | 自由函数入口，内部包揽 SwiGLU+Quant 后处理，由模块化入口替代 |
| `MegaMoeImpl::GroupMatmulSwigluQuantA8W4` | A8W4 自由函数入口，内部包揽 SwiGLU 后处理，由模块化入口替代 |
| 内部 AIV 硬编码 SwiGLU+Quant 逻辑 | 不再存在于模块内部 |

#### 新增项

| 新名称 | 说明 |
|---|---|
| `Gmm1Module` 命名空间 | 独立命名空间 |
| `Gmm1Module::GroupMatmul1<PostOp>` | 纯 GMM1 generic 入口，PostOp functor 注入 AIV 后处理，显式 `tileIdx/totalCoreNum/subTileIdx` |
| `Gmm1Module::GroupMatmul1A8W4<PostOp>` | 纯 GMM1 A8W4 入口，PostOp functor 注入 AIV sub1 后处理 |

#### 变更项

| 项目 | 旧版 | 新版 |
|---|---|---|
| 入口参数 | `(epilogueOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom)` | `(tileIdx, totalCoreNum, subTileIdx, postOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom)` |
| AIV 后处理 | 内部硬编码 `AivGmm1PostGeneric` / `AivGmm1PostA8W4` | 由 `postOp` functor 在 `AivPostLoop` / `GroupMatmulExecA8W4Param` 中回调 |
| SwiGLU 调用位置 | `Detail::AivGmm1PostGeneric` 内部直接调用 `args.swigluQuantOp(epilogueShape, epilogueOffset)` | `MegaMoe::Gmm1SwigluPostOp::operator()` 内部调用 `epilogueOp_(epilogueShape, epilogueOffset)` |

---

### 3. gmm2_module.h

**纯 GMM2 矩阵乘法模块，替代 `GroupMatmul2` / `GroupMatmul2CombineA8W4`**

#### 删除项

| 旧名称 | 说明 |
|---|---|
| `MegaMoeImpl::GroupMatmul2` | 自由函数入口，内部包含 CombineTokens 后处理逻辑 |
| `MegaMoeImpl::GroupMatmul2CombineA8W4` | A8W4 自由函数入口，内部包含 Combine 后处理逻辑 |
| `ProcessCombine` 在模块内的隐式委托 | 原版 `GroupMatmulWithCombine` 在编排层末尾调用 `ProcessCombine`，新版模块内不包含 ProcessCombine |

#### 新增项

| 新名称 | 说明 |
|---|---|
| `Gmm2Module` 命名空间 | 独立命名空间 |
| `Gmm2Module::GroupMatmul2<CombineQuantMode, PostOp>` | 纯 GMM2 generic 入口，PostOp functor 注入 AIV 后处理 |
| `Gmm2Module::GroupMatmul2A8W4<CombineQuantMode, PostOp>` | 纯 GMM2 A8W4 入口，PostOp functor 注入 AIV sub1 后处理 |

#### 变更项

| 项目 | 旧版 | 新版 |
|---|---|---|
| 入口参数 | `(params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom2, groupCnt, pingpongIdx, groupIdx)` | `(tileIdx, totalCoreNum, subTileIdx, postOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom2, pingpongIdx, groupIdx)` |
| AIV 后处理 | 内部硬编码 `AivGmm2PostGeneric` / `AivGmm2PostA8W4` | 由 `postOp` functor 在 `AivPostLoopPingPong` / `GroupMatmulExecA8W4Param` 中回调 |
| Combine 调用位置 | `Detail::AivGmm2PostGeneric` 内部直接调用 `MegaMoeCombineImpl::CombineTokens` | `MegaMoe::Gmm2CombinePostOp::operator()` 内部调用 `CombineTokens` |
| Combine-quant 模式 AIV 路径 | 原版 AIV 不走此路径（注释标注"由 ProcessCombine 独立处理"） | 新版 AIV 接收 NoOp functor，空操作；ProcessCombine 仍由编排层自行调用 |

---

### 4. mega_moe.h (gmm_modules 版)

**编排层重构：PostOp functor 定义 + 模块调用 + 自包含 ProcessCombine**

#### 删除项

| 旧名称 | 说明 |
|---|---|
| `#include "mega_moe_impl.h"` | 原编排层直接依赖完整 impl 头文件 |
| `MegaMoeImpl::GroupMatmulSwigluQuant(...)` 调用 | 编排层直接调用硬耦合函数 |
| `MegaMoeImpl::GroupMatmul2(...)` 调用 | 编排层直接调用硬耦合函数 |
| `MegaMoeImpl::GroupMatmulSwigluQuantA8W4(...)` 调用 | A8W4 路径硬耦合调用 |
| `MegaMoeImpl::GroupMatmul2CombineA8W4(...)` 调用 | A8W4 GMM2 硬耦合调用 |
| `MegaMoeImpl::ComputeCoreGrouping` 调用（在 ProcessCombine 内） | 使用旧命名空间 |

#### 新增项

| 新名称 | 说明 |
|---|---|
| `#include "gmm1_module.h"` | 替代 `mega_moe_impl.h` 的 GMM1 部分 |
| `#include "gmm2_module.h"` | 替代 `mega_moe_impl.h` 的 GMM2 部分 |
| `#include "../mega_moe_impl_base.h"` / `"../mega_moe_base.h"` 等 | 引用路径调整为 `../` 前缀（因 gmm_modules 子目录） |
| `Gmm1SwigluPostOp` (嵌套 struct) | PostOp functor：封装原 `AivGmm1PostGeneric` 的 SwiGLU+Quant 逻辑，签名 `(WorkSet, loopIdx)` |
| `Gmm1SwigluPostOpA8W4` (嵌套 struct) | PostOp functor：封装原 `AivGmm1PostA8W4` 的 SwiGLU 逻辑，签名 `(WorkSetT, ConfigT, Params, startLoopIdx, tileNum)` |
| `Gmm2CombinePostOp` (嵌套 struct) | PostOp functor：封装原 `AivGmm2PostGeneric` 的 CombineTokens 逻辑，签名 `(WorkSet, loopIdx, l0cOutUb)` |
| `Gmm2CombinePostOpA8W4` (嵌套 struct) | PostOp functor：封装原 `AivGmm2PostA8W4` 的 Combine 逻辑，签名 `(WorkSetT, ConfigT, Params, startLoopIdx, tileNum)` |
| `NoOp` / `NoOpA8W4` (局部 struct) | combine-quant 模式下 AIV 后处理的空操作 functor |

#### 变更项

| 项目 | 旧版 | 新版 |
|---|---|---|
| `GroupMatmulWithSwigluQuant` 函数体 | 直接调用 `MegaMoeImpl::GroupMatmulSwigluQuant` / `GroupMatmulSwigluQuantA8W4` | 构造 PostOp functor → 调用 `Gmm1Module::GroupMatmul1` / `GroupMatmul1A8W4`，显式传递 `blockIdx_/blockNum_/subBlockIdx_` |
| `GroupMatmulWithCombine` 函数体 | 直接调用 `MegaMoeImpl::GroupMatmul2` / `GroupMatmul2CombineA8W4`，末尾调用 `ProcessCombine` | 构造 PostOp functor → 调用 `Gmm2Module::GroupMatmul2` / `GroupMatmul2A8W4`，显式传递 `blockIdx_/blockNum_/subBlockIdx_`；combine-quant 模式用 NoOp functor，末尾仍调用 `ProcessCombine` |
| `ProcessCombine` 中 `ComputeCoreGrouping` | `MegaMoeImpl::ComputeCoreGrouping` | `GmmParam::ComputeCoreGrouping` |
| include 路径 | `#include "mega_moe_impl.h"` | `#include "gmm1_module.h"` + `#include "gmm2_module.h"`；其余头文件路径加 `../` 前缀 |
| `epilogueOp_.UpdateNextProblem` 调用位置 | 原版在 `GroupMatmulImplGeneric` 内部（AIV1/SwiGLU 进分支前调用） | 新版在编排层 `GroupMatmulWithSwigluQuant` 中，构造 PostOp 前，编排层显式调用 `epilogueOp_.UpdateNextProblem({mVal, outputN, kVal, 0})` |

#### 逻辑不变项

编排层其余代码（Init, Process, DispatchBuffInit, SendMaskCal, SendCntCal, TripleInfoCalAndDispatch, CopyGMToGMPerToken, UpdateGroupParams, UpdateGlobalBuffer, ResetFlagList, ResetGmm2CombineSyncCounters, InitCombineBuffers, ProcessCombine, Unpermute, CrossRankSyncInWorldSize, ExpertTokenNumCopyOut, QuantProcessInRank）逻辑完全不变，仅 include 路径和 ProcessCombine 中的命名空间引用变更。

---

### 5. mega_moe_layered.h (gmm_modules 版)

**与 mega_moe.h 相同的重构，额外增加 IsLayered 支持**

#### 删除项

同 mega_moe.h 的删除项，加上：
| 旧名称 | 说明 |
|---|---|
| `AivGmm2PostGeneric` 中的 `IsLayered` 模板参数及 `CombineTokensLayered` 分支 | 原版通过 `IsLayered` 模板参数在 `AivGmm2PostGeneric` 内做 if-else 分支 |

#### 新增项

同 mega_moe.h 的新增项，加上：
| 新名称 | 说明 |
|---|---|
| `Gmm2CombinePostOp::isLayered_` 成员 | PostOp functor 内保存 `isLayered_` bool，在 `operator()` 内按此分支调用 `CombineTokensLayered` 或 `CombineTokens` |
| `Gmm2CombinePostOp` 构造函数增加 `bool layered = false` 参数 | 用于控制 Combine 分支 |

#### 变更项

| 项目 | 旧版 | 新版 |
|---|---|---|
| `GroupMatmul2<IsLayered=true>` 模板参数 | `IsLayered` 作为模板参数传入 `GroupMatmulImplGeneric` → `AivGmm2PostGeneric` | `IsLayered` 不再作为 GMM 模板参数，而是通过 `Gmm2CombinePostOp::isLayered_` 在 functor 内部判断 |
| `ProcessCombine` 中 `ComputeCoreGrouping` | `MegaMoeImpl::ComputeCoreGrouping` | `GmmParam::ComputeCoreGrouping` |
| include 路径 | `#include "mega_moe_impl.h"` | `#include "gmm1_module.h"` + `#include "gmm2_module.h"`；其余加 `../` 前缀 |
| `GroupMatmulWithSwigluQuant` | 直接调用 `MegaMoeImpl::GroupMatmulSwigluQuant` / `GroupMatmulSwigluQuantA8W4` | 构造 PostOp → 调用 `Gmm1Module::GroupMatmul1` / `GroupMatmul1A8W4` |
| `GroupMatmulWithCombine` | 直接调用 `MegaMoeImpl::GroupMatmul2<..., true>` (IsLayered=true) / `GroupMatmul2CombineA8W4` | 构造 PostOp（`Gmm2CombinePostOp(params_, state.expertBeforeCnt, true)`） → 调用 `Gmm2Module::GroupMatmul2` / `GroupMatmul2A8W4`；IsLayered 从模板参数变为 PostOp 内部状态 |

---

## 关键设计变更

### 1. PostOp functor 回调模式

- **旧设计**：AIV 后处理逻辑（SwiGLU+Quant / CombineTokens）硬编码在 `AivGmm1PostGeneric` / `AivGmm2PostGeneric` / `AivGmm1PostA8W4` / `AivGmm2PostA8W4` 中，与 GMM 计算循环强耦合。
- **新设计**：`AivPostLoop` 和 `AivPostLoopPingPong` 为泛化 AIV 循环骨架，每 tile 调用 `postOp(workSet, loopIdx)` 或 `postOp(workSet, loopIdx, l0cOutUb)`。具体后处理由编排层定义的 functor（`Gmm1SwigluPostOp`、`Gmm2CombinePostOp` 等）注入。A8W4 路径同样：`GroupMatmulExecA8W4Param` 的 AIV sub1 分支调用 `args.postOp(workSet, config, params, startLoopIdx, tileNum)`。

### 2. 模块独立性

- **旧设计**：`GroupMatmulImplGeneric` / `GroupMatmulImplA8W4` 内部调用 `GetBlockIdx()/GetBlockNum()/GetSubBlockIdx()`，与硬件调度强绑定。
- **新设计**：所有多核身份参数（`tileIdx, totalCoreNum, subTileIdx`）由编排层显式传入，模块内部不调用任何硬件 API。编排层负责 `blockIdx_ = GetBlockIdx() / GetTaskRation()` 等硬件查询，再以参数形式传递给模块。

### 3. ProcessCombine 自包含

- **旧设计**：`ProcessCombine` 在编排层 `GroupMatmulWithCombine` 末尾被调用，逻辑上独立于 GMM2 matmul 循环（仅 combine-quant 模式使用）。
- **新设计**：ProcessCombine 逻辑未变，仍由编排层在 `GroupMatmulWithCombine` 末尾自行调用。GMM2 模块在 combine-quant 模式下接收 NoOp functor（AIV 空操作），不做任何后处理。这确保模块与 ProcessCombine 完全解耦。

### 4. 命名空间变更

- `MegaMoeImpl::Detail` → `GmmParam`：基础设施独立命名空间
- `MegaMoeImpl::ComputeCoreGrouping` → `GmmParam::ComputeCoreGrouping`：编排层 ProcessCombine 中的调用同步变更
- `MegaMoeImpl::NotifyCombineTileComplete` → `GmmParam::NotifyGmm2TileGroupComplete`：语义更清晰
- 常量引用（`L1_TILE_M_256` 等）需带 `MegaMoeImpl::` 前缀

### 5. IsLayered 参数化方式变更

- **旧版**：`IsLayered` 为 GMM 模板参数，透传至 `AivGmm2PostGeneric` 内做 `if constexpr (IsLayered)` 分支
- **新版**：`IsLayered` 不再是 GMM 模板参数，而是 `Gmm2CombinePostOp` functor 的 `isLayered_` 运行时 bool 成员，在 `operator()` 内 `if (isLayered_)` 分支

---

## Bug 修复

| Bug | 旧版 | 新版 | 说明 |
|---|---|---|---|
| `ComputeCoreGrouping` 命名空间 | `MegaMoeImpl::ComputeCoreGrouping`（定义在 `MegaMoeImpl` 自由命名空间） | `GmmParam::ComputeCoreGrouping` | 原版定义在 `MegaMoeImpl` 自由命名空间，`Detail` 内部 `Config::BuildProblemConfig` 曾错误引用 `MegaMoeImpl::` 版而非 `Detail::` 版；新版统一为 `GmmParam::ComputeCoreGrouping`，编排层调用同步修正 |

---

## 待办事项

1. **`mega_moe_impl.h` 最终废弃**：取决于编译入口切换。当前 `gmm_modules` 下的 `mega_moe.h` / `mega_moe_layered.h` 已不再 include `mega_moe_impl.h`，但原版文件仍存在。需确认 `mega_moe_apt.cpp` 等编译入口修改 include 路径指向 `gmm_modules/` 版本后，方可废弃 `mega_moe_impl.h`。

2. **`mega_moe_apt.cpp` include 路径修改**：需将 `#include "mega_moe.h"` 改为 `#include "gmm_modules/mega_moe.h"` 或调整编译系统 include path。

3. **回归测试**：因 PostOp functor 模式改变了 AIV 后处理的调用时机和参数传递路径，需确认：
   - generic A8W8/A4W4 路径 SwiGLU+Quant 输出一致
   - A8W4 路径 SwiGLU 输出一致
   - COMBINE_NO_QUANT 路径 CombineTokens 输出一致
   - combine-quant 路径 ProcessCombine 输出一致
   - IsLayered 路径 CombineTokensLayered 输出一致
