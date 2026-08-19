# arch35 MegaMoe Kernel 实现分析

> 分析对象：`mc2/mega_moe/op_kernel/arch35/`
> Kernel 入口：[mega_moe_apt.cpp](mega_moe_apt.cpp)

## 1. 总览：共 4 种实现

| # | 实现类 | 文件 | 一句话定位 |
|---|--------|------|-----------|
| 1 | `MegaMoeImpl::MegaMoe` | [mega_moe.h](mega_moe.h) | 通用基础实现（按 expert 串行流水），兜底路径，同时覆盖 A4W4 |
| 2 | `MegaMoeImpl::MegaMoeWave` | [mega_moe_wave.h](mega_moe_wave.h) | A8W8 Wave 级流水实现（fp8 act × fp8 weight） |
| 3 | `MegaMoeImpl::MegaMoeA8W4Wave` | [mega_moe_wave_a8w4.h](mega_moe_wave_a8w4.h) | A8W4 动态 Wave 调度实现（fp8 act × fp4 weight），继承 `MegaMoe` |
| 4 | `MegaMoeImpl::MegaMoeLayered` | [mega_moe_layered.h](mega_moe_layered.h) | URMA 跨服务器分层实现（通信方式与 MTE 完全不同） |

公共结构：`stage/`（各阶段子模块）、`common/`（workspace/types/utils/constants）、`blaze/`（GMM tile 化的 prologue/epilogue/policy）。

---

## 2. 路由条件：什么情况下走什么实现

### 2.1 入口处的统一门禁（mega_moe_apt.cpp L73-79）

所有实现共用一个 `mega_moe` kernel 入口，先经过**编译期宏 + 数据类型**门禁：

1. **版本门禁**：`ENABLE_TENSOR_API`（devkit > 9.0）；`ENABLE_MEGA_MOE_LAYERED_KERNEL`（devkit > 9.0 **且** hcomm > 9.0，仅 Layered 需要）。
2. **数据类型门禁**：
   - `x` / `y` 均为 **BF16**；
   - `weight1` / `weight2` 为**同一类型**，且 ∈ {FP8-E5M2, FP8-E4M3FN, FP4-E2M1}；
   - `DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP`（即 dispatchQuantMode attr = 4，MXFP 量化场景）。

### 2.2 第一层分发：通信模式（tiling key `TILINGKEY_COMM_MODE`）

由 tiling key 的 `CommModeType` 决定（[mega_moe_apt.cpp](mega_moe_apt.cpp) L80 / L117）：

- `TILINGKEY_TPL_MTE`（0）→ MTE 单机/单超拓扑路径（实现 1/2/3）
- `TILINGKEY_TPL_URMA`（1）→ URMA 跨超拓扑路径（实现 4 `MegaMoeLayered`）

> ⚠️ **现状**：arch35 host tiling 中 `CalTilingKey` 目前**硬编码 `topoType = TILINGKEY_TPL_MTE`**
> （[mega_moe_tiling.cpp](../../op_host/op_tiling/arch35/mega_moe_tiling.cpp) L308），
> README 也注明"当前暂不支持 URMA 通信方式"。即 URMA/Layered 路径代码已就绪但当前不会被选中。

### 2.3 第二层分发（仅 MTE 路径）：量化输出 dtype × 权重 dtype（mega_moe_apt.cpp L82-116）

| dispatchQuantOutDtype（attr） | weight1/weight2 dtype | host 侧 groupedMatmulMode | 走到的实现 | 场景名 |
|---|---|---|---|---|
| E5M2 | FP8-E5M2（ND） | `GENERAL` | **MegaMoeWave** | A8W8-FP (E5M2) |
| E4M3FN | FP8-E4M3FN（ND） | `GENERAL` | **MegaMoeWave** | A8W8-FP (E4M3FN) |
| E4M3FN | FP8-E4M3FN（FRACTAL_NZ） | `A8W8_NZ` | **MegaMoeWave** | A8W8-FP NZ |
| E4M3FN | FP4-E2M1（NZ_C0_32） | `A8W4` | **MegaMoeA8W4Wave** | A8W4-FP |
| E2M1 | FP4-E2M1（ND 或 NZ） | `A4W4` / `A4W4_NZ` | **MegaMoe**（generic） | A4W4-FP |
| E5M2 | FP8-E4M3FN 等混合 | `GENERAL` | **MegaMoe**（generic） | 混合 dtype 兜底 |

判定逻辑是 kernel 侧的 `isA8W8DtypePair` / `isA8W4DtypePair` 两个 `constexpr`（注释明确说明：**Dispatch 输出类型和权重类型可区分 A8W8/A8W4 Wave，无需增加 tiling key 维度**）：

- `isA8W8DtypePair`：量化输出与权重同为 E5M2，或同为 E4M3FN → `MegaMoeWave`
- `isA8W4DtypePair`：量化输出 E4M3FN + 权重 FP4 → `MegaMoeA8W4Wave`
- 其余（A4W4、混合 dtype 等）→ `MegaMoe`（generic）

host 侧对应判定在 [mega_moe_tiling.cpp](../../op_host/op_tiling/arch35/mega_moe_tiling.cpp) L628-659（写进 `tilingData->groupedMatmulMode`，供各 stage 内部选择 GMM 具体实现，不改变 kernel 类选择）。

### 2.4 tiling key 完整维度

见 [mega_moe_tiling_key.h](mega_moe_tiling_key.h)：dispatchQuantMode（固定 MXFP）、dispatchQuantOutDtype（E5M2/E4M3FN/E2M1）、combineQuantOutType（无量化/E5M2/E4M3FN）、commMode（MTE/URMA）、topkWeightsType（前移开关）。同一 tiling 下 kernel 类由上述 dtype pair 决定，**combineQuantOutType / topkWeightsPrefetch 只影响类内行为，不影响实现类选择**。

---

## 3. 各实现详解与差异

### 3.1 `MegaMoe`（generic，[mega_moe.h](mega_moe.h)）

**执行流程**（`Process()`，L779）：

```
Phase 1 (AIV): 量化本卡 token → 广播路由 mask → 清 flag → (可选)共享专家输入拆分
    ↓ SyncAll
共享专家 GMM1 (AIC) → 跨卡同步 CrossRankSyncInWorldSize
    ↓
MoE 专家逐 expert 串行流水:
    for e in experts:
        dispatch(e+1) 预取 (AIV1)  ‖  GMM1+SwiGLU+ActivationQuant(e) (AIC)
    for e in experts:
        GMM2(e) → 紧接着调度 Combine(e)
    ↓
共享专家 GMM2 → 跨卡同步 → Unpermute (AIV)
```

**关键特征**：
- **expert 粒度串行**：GMM1 一轮循环做完所有 expert，再 GMM2 一轮循环；dispatch 与 GMM1 之间仅有一级"下一 expert 预取"。
- 类内用 `ENABLE_A8W4` / `ENABLE_A4W4` constexpr 分支选择 GMM 具体实现：
  - A4W4：GMM1 走 generic A4W4，GMM2 走 A8W4（activation 提升为 fp8_e4m3fn，避免两段都 A4W4 精度损失过大，L168-172）。
- Combine 参与 core：A8W4/A4W4 配对路径只让 `subBlockIdx==1` 参与，否则全部 AIV（L402-409）。
- **同时是 `MegaMoeA8W4Wave` 的基类**（复用 Init/Phase1/共享专家/Unpermute 等大部分逻辑）。

### 3.2 `MegaMoeWave`（A8W8 Wave，[mega_moe_wave.h](mega_moe_wave.h)）

**执行流程**（`Process()`，L1132）：

```
Phase 1 (AIV): 同 generic
    ↓
AIV1: DispatchAllExpertTokens —— 一次性把所有 expert 全部 dispatch 完
    ↓
Wave 循环 (一个 Wave ≈ mGroupsPerWave 个 256 行 M 分组):
    AIC/AIV0: GMM1 Wave → GMM2 Wave（跟随 wave 游标）
    AIV1:     对上一个 Wave 已完成 GMM2 的 experts 执行 Combine（与下一 Wave 的 GMM1 重叠）
    ↓
共享专家 GMM2 → 跨卡同步 → Unpermute (AIV)
```

**关键特征**：
- **Wave 划分**：由 host `CalcMGroupsPerWave`（[mega_moe_tiling.cpp](../../op_host/op_tiling/arch35/mega_moe_tiling.cpp) L71-90）按"每个 AIC 至少 4 个逻辑 N tile"目标计算，经 tiling 下发。
- **dispatch 与计算解耦**：所有 expert 的 dispatch 先行完成，GMM 按行游标（`ExpertRowCursor`）以 Wave 为单位推进；Wave 边界内 AIV1 的 Combine 与下一 Wave 的 GMM1 天然重叠。
- **角色分工**：AIC+AIV0 做 GMM1/GMM2，AIV1 做 dispatch + combine（与 generic 的"AIC 做 GMM、全体 AIV 做 combine"不同）。
- **GMM2 恒定 `COMBINE_NO_QUANT`**（`ProcessGmm2Wave`，L982）：GMM2 直接输出 BF16，量化推迟到 Combine 阶段（`RunWaveExpertCombineStage<CombineQuantMode>`，行环形 buffer + `DrainWaveCombineRowRing` 回收事件）。
- **GMM2→Combine 软同步**：每个 AIC 在 GMM2 写入可见后发布独占 cache line 完成标记（`NotifyWaveGmm2Ready`），Combine 侧等待。
- **L2 cache 优化**：只有"整 expert 覆盖"的 problem 才允许绕过 L2（`isWholeExpert` 判断）。
- 额外模板参数 `IsGmm1Interleaved`（编译宏 `MEGA_MOE_WEIGHT1_INTERLEAVED`，默认 0），支持交织布局的 gate/up weight1。

### 3.3 `MegaMoeA8W4Wave`（A8W4 动态 Wave，[mega_moe_wave_a8w4.h](mega_moe_wave_a8w4.h)）

继承 `MegaMoe`，只重写 `ProcessMoeExpertStages`，其余（Init、Phase1、共享专家、Unpermute）全部复用基类。

**执行流程**（`Process()`，L258）：

```
启动: AIV1 先把第一个完整 Wave 的 experts 全部 dispatch 完（waveTileTarget = 2 tiles/AIC job）
    ↓
Wave 循环:
    内层 while（当前 Wave GMM1 未完 或 下一 Wave dispatch 未完）:
        AIV1:    dispatch 下一 Wave 的一个 expert（与当前 Wave 计算交替，削峰）
        AIC/AIV0: GMM1+Activation 当前 Wave 的一个 expert
        （AIC/AIV0 通过 GM ready flag 等待该 expert 的 sendCnt 发布）
    当前 Wave 完成后立即:
        for e in 当前 Wave: GMM2(e) + 调度 Combine(e)  ← 与下一 Wave 的 dispatch/GMM1 重叠
```

**关键特征**：
- **动态 Wave**：Wave 大小按 **GMM1 tile 负载**（`GetA8W4Gmm1TileCount`）动态累积到 `A8W4_GMM1_TILES_PER_JOB_TARGET(2) × AIC job 数`；末尾仅剩 1 个 expert 时并入当前 Wave（L45-53）。
- **dispatch 与 GMM1 独立推进**：AIV1 每发布一个 expert 的 token 数（GM flag `WaitUntilGmFlagIsNonZero`），AIC/AIV0 即可消费，不像 Wave 实现那样要求全部 dispatch 完成后才开始。
- **A8W4 GMM**：fp8 激活 × fp4 权重，GMM prologue 中做 W4→W8 提升 + MMAD（`RunGmm1A8W4`/`RunGmm2A8W4`）。
- **V1 UB 被覆盖问题**：A8W4 GMM1 会覆盖 V1 UB，跨 expert 状态改用 GM 持久备份（`cumsumInfoGlobalTensor`），这是基类 `MegaMoe` 中专门为此加的注释（L138-142）。
- GMM1/GMM2/Combine 通过 `gmTileSequence`（GM tile ready 序列）做 AIC↔AIV1 软同步。

### 3.4 `MegaMoeLayered`（URMA，[mega_moe_layered.h](mega_moe_layered.h)）

**与前三者本质区别在通信方式**：不使用 MTE peermem 窗口，而是通过 `Hcomm<COMM_PROTOCOL_UBC_CTP>` 的 **URMA 通道**，支持**跨服务器（超节点间）**通信（成员含 `serverNum_` / `serverId_` / `rankIdInServer_` / relay 偏移等）。

**执行流程**（`Process()`，L2575）：

```
Phase 1 (AIV): SendMaskCal（按全局专家算 mask 推送到目标卡）→ ResetFlagList → ResetDispatchState
    （URMA 模式下 quantTokenScalePtr 未填充，量化直接从原始 BF16 输入做，L2330 附近）
    ↓
共享专家 GMM1 → 跨卡同步
    ↓
逐 expert:
    AIV1: DispatchTokenToRmtServer(e+1) 预取（跨服务器 relay 链路）
          + SendCntCal + MetaInfoCalAndDispatch
    AIC:  GMM1+ActivationQuant(e)（读 relay buffer，含本地/远端 relay 两类拷贝）
    ↓ SyncAll
逐 expert: GMM2(e) + Combine(e)（BuildCombineRankInfo 从本地 URMA mask slot 读各源 rank 信息，
           ProcessCombineGroups/DrainCombineChannels 冲刷 URMA 通道）
    ↓
共享专家 GMM2 → 跨卡同步 → Unpermute
```

**关键特征**：
- **跨服务器 relay dispatch**：token 从源卡 → 目标服务器内的 relay rank → 目标 rank（`DispatchSendKernel`/`LoadTokenFromLocalRelay`/`CopyTokensFromRemoteRelay`），带 round 轮次与 `DispatchSyncWithCompute` 计算发送同步。
- **独立的 buffer/同步体系**：`DispatchBuffInit`/`UnpermuteBuffInit`/`EndSync` 均为自有实现，不复用 MTE 路径的 `stage/` 模块；Unpermute 用 `UNPERMUTE_LIST_NUM=3` 的列表结构。
- **对齐要求不同**：URMA/Layered 路径 hiddenDim 保持 1K 对齐（MTE W4 路径按 64、其余按 32，[mega_moe_tiling.cpp](../../op_host/op_tiling/arch35/mega_moe_tiling.cpp) L1870-1877）。
- workspace/flag 布局按 URMA 需求扩展（[mega_moe_workspace.h](common/mega_moe_workspace.h) 中多处 `topoType == TOPO_TYPE_URMA` 分支）。
- 编译需要 `ENABLE_MEGA_MOE_LAYERED_KERNEL`（devkit 与 hcomm 均 > 9.0）。

---

## 4. 差异对比总表

| 维度 | MegaMoe (generic) | MegaMoeWave | MegaMoeA8W4Wave | MegaMoeLayered |
|---|---|---|---|---|
| 通信方式 | MTE peermem | MTE peermem | MTE peermem | **URMA (Hcomm UBC_CTP)，跨服务器 relay** |
| 适用量化场景 | A4W4、混合 dtype 兜底 | A8W8（E5M2/E4M3FN，ND/NZ） | A8W4（E4M3FN act × FP4 w） | MXFP（URMA 拓扑） |
| Dispatch 时机 | 逐 expert 预取下一 expert | **全部 expert 先 dispatch 完** | 动态 Wave：先 dispatch 首 Wave，之后与 GMM1 逐 expert 交替 | 逐 expert 预取下一 expert（跨服务器 relay） |
| GMM 推进粒度 | 整 expert | **Wave（≈mGroupsPerWave×256 行）** | **动态 Wave（按 GMM1 tile 负载累积）** | 整 expert |
| GMM2 输出 | 按 CombineQuantMode 量化 | **恒 BF16**，量化推迟到 Combine | 按 CombineQuantMode 量化 | 按 CombineQuantMode 量化 |
| Combine 执行者 | 全体 AIV（A8W4/A4W4 时仅 AIV1） | **仅 AIV1**，且与下一 Wave GMM1 重叠 | 仅 AIV1（配对路径） | AIV（按 rank 分组冲刷 URMA 通道） |
| GMM↔Combine 同步 | dispatch/activation flag | GMM2 ready cache line 标记 + 行环形 buffer | GM tile ready 序列 (gmTileSequence) | flag + URMA 通道 drain |
| 类继承关系 | 基类 | 独立类（与 MegaMoe 平行） | **继承 MegaMoe**，仅重写专家流水 | 独立类（自成一体的 stage 函数） |
| 代码规模 | 中 | 大 | 小（复用基类） | 最大（自带全部 stage） |
| 当前是否会被 host 选中 | 是 | 是 | 是 | **否**（tiling 硬编码 MTE，预留待启用） |

## 5. 小结

- **实现选择的两个正交维度**：
  1. **通信拓扑**（tiling key `COMM_MODE`）：MTE → 前三种；URMA → Layered（当前关闭）。
  2. **量化 dtype pair**（dispatchQuantOutDtype × weight dtype，编译期 constexpr）：同为 fp8 → Wave；fp8 act × fp4 w → A8W4Wave；其余 → generic。
- **演进方向**：generic（expert 串行）→ Wave（静态行分组流水）→ A8W4Wave（按 tile 负载动态 Wave，dispatch 与计算深度重叠），流水线并发度逐级提高；Layered 则是面向跨服务器拓扑的独立分支。
- `topkWeightsPrefetch`（tiling key 第 5 维）与 `combineQuantOutType` 是**所有实现共用的行为开关**，不参与实现类选择。
