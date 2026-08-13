# apace_quant_matmul_reduce_scatter_v3 算子设计文档

> **路线**：APACE MC2 通算融合（UDMA PUT 直调）
> **v2 基底**：`operators/apace_quant_matmul_reduce_scatter_v2/`（已通过审查 PASS 98/100，12/12 精度 PASS，57.6us vs v1 143us ≈ 2.48×）
> **v3 定位**：在 v2 基础上做**外科手术式优化**——取消 workspace 轮转回压，workspace 按「每核全量 tile 数」开辟，每 tile 独占槽位（无取模）

---

## 0. 概述

### 0.0 需求类型判断

**通用**：支持多种 FP8 dtype 组合和 shape 配置，4 卡 ReduceScatter 通信，3 级流水架构（继承 v2）。

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | apace_quant_matmul_reduce_scatter_v3 |
| 算子类别 | MC2 通算融合（Quant MatMul + AllToAll PUT 通信 + ReduceSum 流水线） |
| 需求类型 | 通用 |
| 支持数据类型 | A/B = fp8_e4m3fn_t / fp8_e5m2_t（4 种组合），C = bfloat16_t |
| 支持芯片 | Ascend950PR（DAV_3510） |
| NpuArch | DAV_3510（`__NPU_ARCH__` = 3510）（经 `/npu-arch` skill 确认） |
| `--npu-arch` 编译参数 | `dav-3510` |
| SocVersion | ASCEND950 |
| CANN 版本 | 9.2.0（`/home/l00947526/Ascend806/cann-9.2.0`，编译/验证前必须 `source set_env.sh`；`/home/l00947526/Ascend806/cann` 为指向 `cann-9.2.0` 的软链接，已核实，二者等价） |
| 通信模式 | UDMA PUT（直调仅支持 UDMA） |
| 多卡拓扑 | 4 卡 ReduceScatter via AllToAll PUT |
| 流水线 | 3 级流水 mm→comm→reduceSum（继承 v2），**workspace 取消轮转，每 tile 独占槽位** |
| 特殊约束 | APACE 路线四大红线（与 v2 相同，见 v2 DESIGN §2.0） |

### 0.2 用户原始需求

| # | 需求内容 |
|---|---------|
| 1 | 在已完成的 v2（3 级流水 mm→comm→reduceSum，PIPELINE_NUM=4 workspace 轮转）基础上做外科手术式优化：**取消 workspace 轮转回压** |
| 2 | workspace 按「每核全量 tile 数」开辟，每 tile 独占槽位（无取模），删除通道 3 回压 flag（AIC 永不等待槽位释放） |
| 3 | 不考虑 local 后置和 fragmentTensor（后续版本） |
| 4 | 不考虑 buffer 复用（Win 区/workspace 足够大，大 shape workspace 增大到 ~18MB 可接受） |
| 5 | 其余全部保留：通道 1（通信就绪 per-tile）、通道 2（mm 完成 flagId=5）、reduceSum UB buffer + MTE2/MTE3 event、Blaze 组件、AllToAll PUT 通信、localMatmul 默认 0 |
| 6 | CommFlagIdV2 映射可简化（3 不再需要避开；只需避开 5 和 [11-14]）——可选优化 |
| 7 | 必须核实 BlockSchedulerQuantBatchMatmulV3 的 UpdateTailTile 尾块拆分行为，给出可证明安全的 maxTilesPerCore 上界公式 |

### 0.3 v2 → v3 变更概述

v2 中 workspace 采用 PIPELINE_NUM=4 槽位轮转，需要通道 3 回压 flag（AIV reduceSum 消费完通知 AIC 槽位可复用）防止 AIC 覆盖未消费数据。v3 将 workspace 扩大为「每核全量 tile 数 × 核数」个独占槽位，AIC 每完成一个 tile 写入新的槽位，**无需等待任何槽位释放**，删除通道 3 全部逻辑。数学语义、通信拓扑、Blaze 组件、reduceSum 搬运逻辑完全不变。

---

## 1. v3 变更点（相对 v2 的 diff 级描述）

> 本节为 v3 全部变更的完整清单。**未列出的部分与 v2 完全一致**（见 v2 DESIGN.md）。

### 1.1 变更点清单

| # | 位置（v2 文件:行） | v2 行为 | v3 行为 | 类型 |
|---|---------------------|---------|---------|------|
| C1 | `quant_matmul_mx_kernel.h` L403-405 | AIC 在 `cubeNum_ >= PIPELINE_NUM(4)` 时 `CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(SYNC_AIV_AIC_FLAG=3)` 等 AIV 释放槽位 | **删除整段**（AIC 永不等待槽位释放） | 删除 |
| C2 | `quant_matmul_mx_kernel.h` L418-419 | `workSpaceOffset = baseM×baseN×(coreIdx + (cubeNum_ % pipelineNum_)×coreNum_)` | `workSpaceOffset = baseM×baseN×(coreIdx + cubeNum_×coreNum_)`（**去取模**，每 tile 独占槽位） | 修改 |
| C3 | `quant_matmul_mx_kernel.h` L216-217 + L507-519 | `Run()` 末尾调用 `PostCompute()` drain 剩余 Min(cubeNum_, 4) 个回压 flag | **删除 `PostCompute()` 调用与方法本体** | 删除 |
| C4 | `udma_impl.h` RunReduceSum L362-363 | `workSpaceOffset = baseM×baseN×(coreIdx + (cubeNum % pipelineNum)×coreNum)` | `workSpaceOffset = baseM×baseN×(coreIdx + cubeNum×coreNum)`（与 AIC 侧同步去取模） | 修改 |
| C5 | `udma_impl.h` RunReduceSum L399-400 | AIV 每 tile 末尾 `CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG=3)` 通知槽位可复用 | **删除**（cubeNum 自增保留，仍用于槽位寻址） | 删除 |
| C6 | `tiling_data.h` 及 `pipelineNum` 全链 | `uint32_t pipelineNum{4}`（轮转深度） | 替换为 `uint32_t maxTilesPerCore{0}`（每核全量 tile 数上界，host 推导，仅用于 workspace 分配与校验；kernel 槽位寻址不消费该字段）。**plumbing 全链删除**（C2/C4 去取模后无消费者，避免死字段误导）：`tiling_data.h` 字段 → `main.cpp` 赋值（v2 L201）→ `InitBaseParams` 传参（udma_impl.h L421）→ `baseParams_.pipelineNum`（L197）→ `SetupParams` → `LocalParams.pipelineNum`（kernel L136）→ kernel 成员 `pipelineNum_`（L193）及两处赋值代码，一并删除/替换 | 修改 |
| C7 | Host `main.cpp` workspace 分配 | `workspaceSize = PIPELINE_NUM × coreNum × baseM × baseN × 2B` | `workspaceSize = maxTilesPerCore × coreNum × baseM × baseN × sizeof(BF16)`，公式见 §2 | 修改 |
| C8 | Host `main.cpp` 输入校验 | 无 | **新增**：`maxTilesPerCore ≤ 15`（flagId=5 计数器上界保护，见 §4.3）。**执行顺序**：在 SWAT tiling 完成之后、workspace `aclrtMalloc` **之前**执行（避免大 shape 先分配后拒绝）；**取值来源**：`maxTilesPerCore` 的四个输入（m、n、baseM、baseN、usedCoreNum）全部取自 SWAT tiling 最终输出 `tilingData.tileQbmmTilingData`（v2 main.cpp L196-198 已有取数先例），不使用 raw `M/4` 或 baseM 候选默认值 | 新增 |
| C9 | `udma_impl.h` CommFlagIdV2 L59-66 | 避开 3、5、[11-14]，可用集 10 个 | **可选**：3 不再需要避开，可用集 11 个（见 §4.4）；默认实现可保持不变（原映射天然满足新约束） | 可选 |
| C10 | 常量定义 | `SYNC_AIV_AIC_FLAG_V2=3`（回压通道）、`PIPELINE_NUM_V2=4` | 删除回压常量；`SYNC_AIC_AIV_FLAG=5`、`SYNC_MODE2=2` 保留（更名 _V3） | 修改 |

### 1.2 明确保留不变的部分

| 项 | 说明 |
|----|------|
| 通道 1（通信就绪） | AIV `SetFlag<0x2, PIPE_MTE3>(commFlagId)` / AIC `WaitFlag<0x2, PIPE_MTE2>(commFlagId)`，per-tile，含 `CalcDependTileIdx` 去重与循环末尾 `readyTileIdx` 兜底 drain（kernel L501-504），**完全不变** |
| 通道 2（mm 完成） | AIC `SetFlag<SYNC_MODE2, PIPE_FIX>(5)` / AIV `WaitFlag<SYNC_MODE2, PIPE_MTE2>(5)`，每 tile 配对一次，**完全不变** |
| reduceSum 实现 | UB buffer（baseM×AlignUp(baseN,32)×2B）、DataCopyPad 2D 搬运参数（cpIn/cpOut）、MTE2↔MTE3 event 配对（RAW/WAR）、`outOffset = mPos×N + nPos`，**完全不变** |
| AIV 复用 BlockScheduler | v2 DESIGN §2.16 方案 (b) 的 6 项复刻要求**全部保留**（cubeNum 序列构造一致是 v3 槽位寻址两侧对齐的前提） |
| Blaze 组件 | BlockMmad / BlockSchedulerQuantBatchMatmulV3 / MatmulWithScaleMx / QuantMatmulTilingSwat，**完全不变** |
| 通信 | AllToAll PUT（data + scale 双通道）、TeamBarrier、SyncAll、Win 区布局，**完全不变** |
| localMatmul | 默认 0（REMOTE-only）；DEFERRED_SYNC 分支代码保留（mm 输出地址同样去取模） | 
| 精度策略 | rtol=atol=1e-2，BF16 输出，L0C FP32 跨 rank 累加，**完全不变** |
| APACE 红线 | 禁止 `__schedmode__`/`core_ratio`；Blaze 模板；禁止改 block/tiling 共享层；UDMA 直调；无 Hccl::；`KERNEL_TYPE_MIX_AIC_1_1`，**完全不变** |

---

## 2. maxTilesPerCore 安全上界公式（Blaze 源码依据）

### 2.1 结论

```
blockDimM        = CeilDiv(mLocal, baseM)          // mLocal = M / rankSize
blockDimN        = CeilDiv(N, baseN)
blockNums        = blockDimM × blockDimN
maxTilesPerCore  = CeilDiv(blockNums, coreNum)     // 严格上界，无需 +1（§2.2 证明）

workspaceSize    = maxTilesPerCore × coreNum × baseM × baseN × sizeof(bfloat16_t)
```

### 2.2 Blaze 源码依据

来源：`operators/apace_quant_matmul_reduce_scatter_v2/third_party/ops-tensor/include/blaze/gemm/block/block_scheduler_qbmm.h`（`BlockSchedulerQuantBatchMatmulV3`）。

**证据 1：每核基础 tile 数 = `roundNums_` ≤ CeilDiv(blockNums_, coreNums_)**（构造函数 L51-66）：

```cpp
mBlockNums_ = CeilDiv(m, baseM_);            // L51
nBlockNums_ = CeilDiv(n, baseN_);            // L52
blockNums_  = mBlockNums_ * nBlockNums_;     // L53
endBlockIdx_ = (blockNums_ - 1) % coreNums_; // L62
roundNums_  = CeilDiv(blockNums_, coreNums_);// L63
if (blockIdx_ > endBlockIdx_) roundNums_ -= 1; // L64-66：只会减少，不会增加
```

**证据 2：`GetTileIdx()` 的迭代次数严格等于 `roundNums_`**（L190-235）：`if (roundIdx_ >= roundNums_) return false;`，每次返回 true 时 `roundIdx_++`。即**每核处理的 tile 数 = roundNums_（UpdateTailTile 调整后的值）**，不多不少。

**证据 3：`UpdateTailTile` 对每核 tile 数的影响至多为 +1**（L85-101）：

```cpp
totalTailTile_ = mTailTile * nTailTile;                          // L89
tailOriginalBlockNums = min(blockNums_, endBlockIdx_ + 1);       // L90
newEndBlockIdx = endBlockIdx_ + tailOriginalBlockNums * (totalTailTile_ - 1); // L91
if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
    roundNums_ += 1;                                             // L92-94：唯一的增量，且只 +1
}
if (blockIdx_ > newEndBlockIdx) { mTailTile_ = 1; ... }          // L95-99：超出范围的核不拆分
```

关键点 1：`+1` 的核与 `roundNums_ = CeilDiv` 的核互斥——`roundNums_ = CeilDiv` 恰是 `blockIdx_ ≤ endBlockIdx_` 的核（L64-66 未减 1），永远不满足 L92 的 `blockIdx_ > endBlockIdx_` 条件；满足 +1 条件的核（`blockIdx_ > endBlockIdx_`）从 `CeilDiv - 1` 起跳，+1 后**至多回到 CeilDiv**。因此 **max(roundNums_) = CeilDiv，与是否尾块拆分无关，严格上界不需要 +1**。

关键点 2：尾块拆分**不是**把一个 tail tile 在同一核上变成 `totalTailTile_` 个子 tile，而是把子 tile **摊到空闲核上**（这正是 kernel 侧调用条件 `(endBlockIdx_+1) × mTailTile × nTailTile ≤ GetBlockNum()` 的语义——拆分后的子 tile 总数能被空闲核容纳）。落到单个核上，只是多处理**一个**子 tile（最后一轮 `roundIdx_ == roundNums_-1` 时处理 `blockIdx_ / totalTailTile_` 对应的子块，见 L199）。且 `UpdateTailTile` 在 kernel 中由 `needUpdateTail_` 保护**至多调用一次**（v2 kernel L375-380，单 batch，`ProcessSingleBatch` 仅调用一次），`UpdateNextBatchBlockRoundParams`（L176-188，多 batch 用）本算子不触及。

**证据 4：实际处理 tile 数只会 ≤ roundNums_**：尾块拆分轮中 `GetBlockShape` 可能返回 `{0,0,0,0}`（L168-170，子块偏移越界），AIC 侧 `return` 早退、AIV 侧 `break`，`cubeNum` 不再自增。因此实际 `cubeNum` 终值 T ≤ roundNums_。

**综上（可证明安全，且为严格上界）**：

```
T_actual ≤ roundNums_ ≤ CeilDiv(blockNums_, coreNums_) = maxTilesPerCore
```

最大槽位偏移：`baseM×baseN×((coreNum-1) + (maxTilesPerCore-1)×coreNum) + baseM×baseN = maxTilesPerCore×coreNum×baseM×baseN`，恰好等于 workspace 分配大小，**无越界写**。

> **可选加固**（防御性编程，非必需）：host 可按 `maxTilesPerCore + 1` 分配，代价仅 1 行槽位（128KB×coreNum≈4MB），为未来 Blaze 调度器行为变化留余量。v3 默认按上述严格上界（CeilDiv，无 +1）分配。

### 2.3 与 v2 废弃公式 `totalMmTiles` 的关系

v2 DESIGN §2.16 Issue 2 已论证「每核 tile 数依赖运行时 `GetBlockNum()`、尾块拆分参数与早退逻辑，host 无法**精确**预算」，因此 AIV 循环由 `bsAiv.GetTileIdx()` 隐式终止。v3 的 `maxTilesPerCore` 不是精确预算，而是**可证明的上界**（证据 1-4 给出严格不等式），用途仅限于 host 侧 workspace 分配与校验，kernel 运行时不消费——与 v2 的决策不冲突。

---

## 3. workspace 大小新公式与各 shape 估算表

### 3.1 公式

```
workspaceSize = maxTilesPerCore × coreNum × baseM × baseN × sizeof(bfloat16_t)
其中 maxTilesPerCore = CeilDiv(CeilDiv(mLocal,baseM) × CeilDiv(N,baseN), coreNum)   // 严格上界，无 +1
     coreNum = usedCoreNum（QuantMatmulTilingSwat 推导，≤ 32 @950PR Server / ≤ 28 @950PR PCIE）
```

> baseM/baseN/usedCoreNum 由 SWAT tiling 引擎运行时推导（初始候选 256，按 shape 细化），下表为**代表性配置估算**，精确值以运行时 tiling 日志为准。

### 3.2 估算表（sizeof(BF16)=2B；coreNum = min(32, blockNums) 为**假设值**）

| shape (M,K,N) | mLocal | baseM×baseN | blockDimM×blockDimN | blockNums | coreNum | maxTiles=T | v3 workspace | v2 workspace (4 槽轮转) | 备注 |
|---------------|--------|-------------|---------------------|-----------|---------|------------|--------------|--------------------------|------|
| 512,1024,512 | 128 | 128×256 | 1×2 | 2 | 2 | 1 | 0.13 MB | 0.52 MB | v2 ST 用例 |
| 1120,2048,1024 | 280 | 256×256 | 2×4 | 8 | 8 | 1 | 1.0 MB | 4.2 MB | v2 ST 用例 |
| 2048,4096,2048 | 512 | 256×256 | 2×8 | 16 | 16 | 1 | 2.1 MB | 8.4 MB | v2 ST 主用例 |
| 16384,4096,2048 | 4096 | 256×256 | 16×8 | 128 | 32 | 4 | **16.8 MB** | 16.8 MB | 大 shape，T=4（v2 回压恰在边界触发） |
| 16384,4096,2048 | 4096 | 128×256 | 32×8 | 256 | 32 | 8 | **16.8 MB ≈ ~18MB** | 8.4 MB | 大 shape 另一 tiling 配置，T=8>4，回压消除收益场景 |
| 16384,4096,4096 | 4096 | 256×256 | 16×16 | 256 | 32 | 8 | **32.0 MB** | 16.8 MB | tiles/core=8，重点验证回压消除收益 |

> ⚠️ **coreNum=32 为假设值，以运行时 SWAT tiling 日志为准**（见 §8 T11/T12 前置验证步骤）。若实际 usedCoreNum < 32，T 与 workspace 需按实际值重算。

结论：
- 小 shape（T ≤ 4）：v3 workspace **不大于** v2，内存反而下降；
- 大 shape（T > 4）：v3 workspace 随 T 线性增长；M=16384,N=2048 两种 tiling 配置均为 16.8MB（符合用户「~18MB 可接受」口径）；**M=16384,N=4096 最大配置为 32.0MB，超出用户 ~18MB 口径约 1.8 倍——已提交用户确认（见 §8 备注与 WALKTHROUGH.md 串讲记录）**；
- T ≤ 15 的 shape 均被覆盖（见 §4.3 计数器约束），T=15 对应 blockNums≈480（如 mLocal=4096、N=7680 @256×256），覆盖目标 shape 范围。

---

## 4. 删除通道 3 后的 flag 编排正确性论证

### 4.1 v3 通道编排（2 通道 + SyncAll）

| 通道 | 方向 | API 配对 | flagId | 配对关系 |
|------|------|----------|--------|----------|
| 1 通信就绪 | AIV→AIC | `SetFlag<0x2, PIPE_MTE3>(fid)` / `WaitFlag<0x2, PIPE_MTE2>(fid)` | CommFlagIdV2(tid)，避开 5、[11-14] | 每个 fid Set 1 次 / Wait ≤1 次（去重 + 兜底 drain 配平） |
| 2 mm 完成 | AIC→AIV | `SetFlag<2, PIPE_FIX>(5)` / `WaitFlag<2, PIPE_MTE2>(5)` | 5（常量） | AIC Set T 次 ⇔ AIV Wait T 次（同一 BlockScheduler 枚举序，v2 §2.16 已证） |
| — 块间同步 | AIV↔AIV | `SyncAll<true>()` | 3510 架构下 `isAIVOnly=true` 实际仅占 **AIV flagId 14**（`SyncAll.md` 表 3 已核实）；本设计**保守避开 [11-14]** | 不变 |

### 4.2 通道 1/2 配对不受通道 3 删除影响

- **flagId 空间独立**：通道 1 用 fid∈{0..4,6..10,15}，通道 2 用 5，SyncAll 实际占用 14（本设计保守避开 [11-14]），原通道 3 的 3 被释放后**无任何通道使用 3**，无新增冲突。
- **配对计数独立**：每个 flagId 有独立计数器（key_features.md：Set 完成 → 计数器 +1，Wait 解除阻塞 → 计数器 -1）。删除通道 3 的 Set/Wait 不改变通道 1/2 的 Set/Wait 次数与顺序。
- **AIV 侧顺序不变**：RunReduceSum 循环结构、BlockScheduler 复刻序列、cubeNum 自增位置全部保留，仅删末尾 Set(3) 一行；AIV 的 Wait(5) 次数仍 = AIC Set(5) 次数 = T。
- **AIC 侧顺序不变**：除删除回压 Wait 外，通信等待（通道 1）→ mm → Set(5) → cubeNum++ 的 per-tile 序列不变；循环末尾 readyTileIdx 兜底 drain（通道 1）保留。

### 4.3 ⚠️ 新风险与约束：flagId=5 计数器上界（0-15）——必须 host 校验

**风险**：官方文档（`CrossCoreWaitFlag_ISASI.md` flagId 约束，已核实原文）：「flagId 对应的计数器**计数范围为 0-15。如果计数器的值超出该范围，则会异常报错，中断流程**。」v2 中回压机制将 AIC 领先 AIV 的 tile 数钳制在 ≤4，故 flagId=5 的未消费计数 ≤4，安全。**v3 删除回压后，AIC 不再被 AIV reduceSum 进度钳制**：当 AIV 尚处于通信阶段（Phase 1）时，AIC 可连续完成全部 T 个 tile 并 Set(5) T 次，计数器峰值 = T。若 T > 15（计数器需达到 16），将触发硬件异常报错。

**结论**：v3 的正确性依赖 **T ≤ maxTilesPerCore ≤ 15**，必须作为 host 侧输入校验（变更点 C8）：

```
校验：maxTilesPerCore ≤ 15   // maxTilesPerCore = CeilDiv(blockNums, coreNum)（严格上界，§2.2）
                              // 等价于 flagId=5 计数器峰值 ≤ 15，恰好用满硬件上限、不越界
不满足 → 拒绝 launch（报错提示 shape 超出 v3 支持范围，可回退 v2）
```

注：采用 §2.2 的严格上界后，校验门控的是 `CeilDiv ≤ 15`，T=15 的边界 shape 可合法放行（计数器峰值 15 ∈ [0,15]）；不存在保守 1 档误拒。

覆盖性评估：T ≤ 15 对应 blockNums ≤ 480（coreNum=32），即 mLocal×N ≤ 480×65536 元素（256×256 tile 下，如 mLocal=4096 时 N ≤ 7680），覆盖全部目标 shape（M=16384 大 shape T=5~9）。**该约束不是回压的变相恢复**——它是静态 shape 校验，不在 kernel 热路径上引入任何等待。

> 可选加固（本期不实施）：若未来需要 T > 15，可利用释放的 flagId=3 做 mm 完成双 flag 轮转（tile 奇偶分别 Set 5/3，AIV 同序 Wait），将上界提升至 30。需要时再启用。

### 4.4 CommFlagIdV2 简化（可选优化，变更点 C9）

通道 3 删除后，通信就绪 flag 只需避开 5 和 SyncAll 占用。经核实 `SyncAll.md` 表 3：3510 架构下 `SyncAll<true>`（isAIVOnly=true）实际**仅占 AIV flagId 14**；占 AIC 11/12/13 + AIV 12/13 的是 `isAIVOnly=false + __mix__(1,1)` 模式。v2/v3 实际调用为 `SyncAll<true>()`（v2 udma_impl.h L299，AIV-only 路径），故真实占用只有 14。本设计**保守避开 [11-14]**（覆盖未来改用 `SyncAll<false>` 的情形，此时需重新评估 AIC 侧 11/12/13 冲突）。保守口径下可用集 = {0,1,2,3,4,6,7,8,9,10,15} 共 **11** 个（原 10 个）：

```cpp
// v3 可选简化映射（tid ≤ 10，即通信 tile 上限 10→11）
fid = (tid < 5) ? tid : (tid + 1);   // tid 0..4→0..4；tid 5..9→6..10；tid 10→11 命中 [11-14]
if (fid >= 11 && fid <= 14) fid = 15;
```

**决策**：默认**保持 v2 映射不变**（原映射避开 3 是更强约束，天然满足新约束，零风险）；仅当未来需要第 11 个通信 tile 时才切换。host 校验 `headTileCnt + tailTileCnt ≤ 10` 保持不变。

### 4.5 无死锁论证

- **AIC 侧**：唯一等待源为通道 1（通信就绪）。AIV Phase 1 无条件完成全部 commTurn 轮通信并 Set 全部 commFlagId（不依赖 AIC 任何状态），故通道 1 的 Wait 必然解除；AIC 再无其他 Wait（回压已删，PostCompute drain 已删）→ **AIC 必然运行至结束**。
- **AIV 侧**：Phase 1 完成后进入 Phase 2，唯一等待源为通道 2 Wait(5)，次数 = 本核 BlockScheduler 枚举 tile 数 T；AIC 侧同一枚举序 Set(5) T 次（v2 §2.16 方案 b 复刻论证不变）→ 配对平衡，**AIV 必然运行至结束**。
- **跨核/跨卡**：SyncAll、TeamBarrier、AllToAll Wait<BARRIER_DEVICE> 逻辑完全未动，无新依赖环。
- **计数器**：通道 1 每 fid 计数 ≤1；通道 2 计数峰值 = AIC 领先数 ≤ T ≤ 15（§4.3 校验保证）→ 无溢出。
- **早退分支残留计数（已知行为，继承 v2，非 v3 回归）**：尾块拆分轮 `GetBlockShape` 返回零形状时 AIC 直接 `return`（v2 kernel L397-398），**跳过循环末尾的通道 1 兜底 drain**（L501-504）；此时 AIV Phase 1 已无条件 Set 全部 commFlagId，未被 Wait 的 fid 计数器在 kernel 退出时残留为 1。残留计数 ≤ commTurn ≤ 10，不超 0-15 上限，**无溢出风险**；残留计数是否跨 launch 持久以硬件行为为准——v2 12/12 精度 PASS（每用例独立进程/launch）实证未暴露问题，大概率硬件在每次 kernel launch 时初始化调度模块计数器。v3 保持该行为完全不变，记录为已知行为，无需代码改动。

### 4.6 无越界论证

- AIC 写：`workspaceGmAddr_ + baseM×baseN×(coreIdx + cubeNum×coreNum)`，cubeNum ≤ T-1 ≤ maxTilesPerCore-1 → 最大写地址 < workspaceSize（§2.2 证明）。
- AIV 读：同一公式、同一 cubeNum 序列（BlockScheduler 复刻保证枚举序一致）→ 读地址集合 = 写地址集合，且每个槽位**先写后读**由通道 2（Wait(5) 在 DataCopyPad 之前）保证。
- 槽位独占：每 (coreIdx, cubeNum) 对应唯一槽位，无复用 → **无需回压的根本依据**；不同核槽位区间不相交（coreIdx 项错位 coreNum 步长），无写冲突。

---

## 5. L2 cache 影响分析

| 维度 | v2（4 槽轮转） | v3（独占槽位） | 影响评估 |
|------|----------------|----------------|----------|
| workspace 热工作集 | 4×coreNum×slot（4~17MB），槽位循环复写，大概率高频驻留 L2 | maxTiles×coreNum×slot（0.1~32MB），每槽位写 1 次读 1 次，**无复用** | v3 失去「槽位复用驻留」红利 |
| AIV reduceSum 读 workspace | 读最近 4 个槽位，L2 命中率高 | 读 tile k 时距 AIC 写 tile k 的时间差 = AIC 领先度 × 单 tile mm 耗时；领先度小则仍在 L2 | **流水近距离消费时影响小** |
| 最坏情形 | 不存在（回压钳制领先 ≤4） | AIV 通信阶段 AIC 可无阻塞跑完全部 T 个 tile，最早写入的槽位可能被后续写流 + B/ScaleB/A Win 数据流挤出 L2 | 950PR Server L2=128MB（PCIE 112MB），workspace ≤32MB 占比 <1/3，且 mm 本身是计算瓶颈、fixpipe 写流速率有限，整体挤占有限 |
| reduceSum 读 miss 代价 | — | GM 读走 MTE2，单 tile 128KB，流水掩盖下增量有限 | 可接受 |
| 净结论 | — | — | **预期性能净收益为正**：消除的是 AIC 侧 M 流水硬阻塞（v2 在 T>4 时每 tile 都可能等 AIV 搬运）；L2 命中率下降只影响 AIV MTE2 读延迟，且被流水掩盖。以 M=16384（T=5~9）用例实测对比 v2 57.6us 基线验证 |

> 缓解项（如实测回退再考虑，本期不实施）：对 workspace Tensor 可设置 `SetL2CacheHint(CACHE_MODE_DISABLE)` 让 fixpipe 写绕开 L2、避免污染 B/ScaleB 的 L2 驻留——但这会同时让 AIV 读必 miss，需实测权衡。

---

## 6. 与 v2 的差异对比表

| 维度 | v2 | v3 |
|------|----|----|
| workspace 槽位分配 | PIPELINE_NUM=4 槽轮转，`cubeNum % 4` | 每 tile 独占槽位，无取模 |
| workspace 大小 | 4×coreNum×baseM×baseN×2B（固定） | maxTilesPerCore×coreNum×baseM×baseN×2B（随 shape 伸缩；小 shape 更小，M=16384 大 shape ~17-32MB） |
| 回压通道（flagId=3） | AIV Set(3) / AIC Wait(3) + PostCompute drain | **全部删除**（3 处：kernel L403-405、L507-519、udma_impl L399-400） |
| CrossCore 通道数 | 3（通信就绪 + mm 完成 + 回压） | 2（通信就绪 + mm 完成） |
| AIC 阻塞条件 | 通道 1 通信等待 + cubeNum≥4 回压 | 仅通道 1 通信等待 |
| flagId=5 计数器峰值 | ≤4（回压钳制） | ≤T（≤15，host 校验保证，§4.3） |
| 新增 host 校验 | — | maxTilesPerCore ≤ 15 |
| tiling 字段 | `pipelineNum=4` | `maxTilesPerCore`（替换，kernel 不消费，仅 host 分配/校验用） |
| CommFlagIdV2 | 避开 3/5/[11-14]，上限 10 tile | 默认不变；可选简化为避开 5/[11-14]，上限 11 tile |
| 通信就绪 flag 校验 | headTileCnt+tailTileCnt ≤ 10 | 不变（≤10） |
| 数学语义 / 精度 | BF16 输出，L0C FP32 跨 rank 累加，rtol=atol=1e-2 | **完全相同**（v3 仅改同步与寻址，不改任何数值路径） |
| Blaze 组件 / 通信组件 | — | **完全不变** |
| reduceSum 搬运 | DataCopyPad 2D + MTE2/MTE3 event | **完全不变**（仅槽位寻址去取模 + 删 Set(3)） |

---

## 7. 文件清单（v3，复制自 v2 + 修改）

#### [MODIFY] 算子层文件（3 个 kernel 头文件，从 v2 复制后按 §1.1 变更点修改）

| 文件 | 变更 |
|------|------|
| `kernel/apace_quant_matmul_reduce_scatter_v3/quant_matmul_mx_kernel.h` | C1（删回压 Wait）、C2（寻址去取模）、C3（删 PostCompute）、C10（常量更名 _V3，删回压常量） |
| `kernel/apace_quant_matmul_reduce_scatter_v3/apace_quant_matmul_reduce_scatter_v3_udma_impl.h` | C4（AIV 寻址去取模）、C5（删 Set(3)）、C9（可选 CommFlagIdV2 简化）、类名/常量更名 _V3 |
| `kernel/apace_quant_matmul_reduce_scatter_v3/apace_quant_matmul_reduce_scatter_v3_tiling_data.h` | C6（pipelineNum→maxTilesPerCore，含 LocalParams/baseParams_/kernel 成员全链清理，见 §1.1 C6）、结构体/常量更名 _V3 |

#### [MODIFY] ST 工程（从 v2 复制后修改）

| 文件 | 变更 |
|------|------|
| `st/apace_quant_matmul_reduce_scatter_v3/src/main.cpp` | C7（workspace 新公式）、C8（maxTilesPerCore ≤ 15 校验）、日志更名 |
| `st/apace_quant_matmul_reduce_scatter_v3/src/kernel_launcher.h` | 4 个入口函数更名 _V3 |
| `st/apace_quant_matmul_reduce_scatter_v3/cases.csv` | 保留 v2 10 用例 + 新增大 shape 用例（§8） |
| 其余（CMakeLists/run.sh/scripts） | 复制更名 |

#### [REUSE] 共享层（禁止修改，同 v2）

`block/`、`tiling/`、`basic/`、`utils/`、`third_party/ops-tensor`（Blaze）。

---

## 8. 测试计划

精度标准（继承 v2）：rtol=atol=1e-2，4 dtype 组合覆盖。

| 编号 | 用例 | shape (M,K,N) | 目的 |
|------|------|---------------|------|
| T0（前置） | tiling 假设核实 | 复用 v2 工程对 T11/T12 shape 打印 SWAT tiling 日志 | 框架搭建阶段确认实际 `usedCoreNum`（§3.2 表中 coreNum=32 为假设值）。若 T12 实际 usedCoreNum ≤ 17（`CeilDiv(256, usedCoreNum) > 15`），host 校验将拒绝 launch → 预设备选方案：调整 shape 或 baseM/baseN 候选使 `CeilDiv ≤ 15` 成立后再进入 T12 |
| T1-T10 | v2 全部 10 用例回归 | 2048/512/1120 系列 × 4 dtype | 回归：小 shape（T≤4）功能等价 v2 |
| T11 | 大 shape 边界 | M=16384, K=4096, N=2048（mLocal=4096，T=4~8 视 tiling） | 大 shape 正确性 + workspace ~17MB 分配 |
| T12 | **tiles/core > 4 重点用例** | M=16384, K=4096, N=4096（T=8，workspace 32.0MB） | **验证回压消除收益**（v2 在此类 shape 上 AIC 每 tile 等回压）；精度 + 性能双验证 |
| T13 | maxTilesPerCore 上界校验 | 构造 `CeilDiv(blockNums, coreNum) > 15` 的 shape（如 M=32768, N=8192）；**增补边界用例**：构造 `CeilDiv = 15` 恰好放行的 shape | 验证 host 校验正确拒绝（不 launch、报错信息清晰）+ 边界恰好放行 |
| T14 | 尾块拆分路径 | M 非 baseM 整数倍且满足 UpdateTailTile 条件（如 M=1120 已覆盖；可补 M=2252 类 shape） | 验证严格上界（CeilDiv，无 +1）在尾块拆分路径下的正确性（精度 PASS 即证明槽位寻址无错位）；同时覆盖早退跳过通道 1 兜底 drain 分支（§4.5 已知行为） |

**性能验收**：以 v2 同 shape 数据为基线（v2 主用例 57.6us）；T12 为重点对比用例，预期 v3 < v2（回压消除）；T1-T10（T≤4）预期与 v2 持平（回压本不触发，仅省 PostCompute drain）。

> ⚠️ **用户确认事项**：T12 对应的最大 workspace 为 **32.0MB**（收紧公式后），超出用户「~18MB 可接受」口径约 1.8 倍；M=16384,N=2048 的两种 tiling 配置均为 16.8MB（符合口径）。32.0MB 是否可接受需用户确认（见 WALKTHROUGH.md 串讲记录）。

---

## 9. 确认清单

- [x] v3 变更点 diff 级清单完成（§1.1，C1-C10）
- [x] 未变更部分明确标注（§1.2）
- [x] maxTilesPerCore 安全上界公式 + Blaze 源码 4 项证据（§2，block_scheduler_qbmm.h L51-66/L85-101/L190-235/L168-170）
- [x] workspace 新公式 + 各 shape 估算表（§3，收紧上界后 M=16384 ~17-32MB；32MB 项已提交用户确认）
- [x] 删除通道 3 后 flag 编排正确性论证（§4：通道 1/2 配对不受影响、无死锁、无越界）
- [x] **新风险识别**：flagId=5 计数器 0-15 上界 → maxTilesPerCore ≤ 15 host 校验（§4.3，官方文档 CrossCoreWaitFlag_ISASI.md 已核实）
- [x] L2 cache 影响分析（§5）
- [x] 与 v2 差异对比表（§6）
- [x] 测试计划（§8，重点 T12 tiles/core>4 回压消除收益验证）
- [x] NpuArch 确认：DAV_3510 / `__NPU_ARCH__`=3510 / `--npu-arch dav-3510`（`/npu-arch` skill）
- [x] APACE 红线确认（继承 v2 §2.0，本期变更不触碰任何红线项）
