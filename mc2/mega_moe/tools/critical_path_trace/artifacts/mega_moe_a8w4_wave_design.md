# MegaMoe A8W4 wave 流水改造设计

> 基于 A8W4 关键路径 trace 数据的 wave 流水改造方案设计。
> 配套打点设计见 [`mega_moe_a8w4_critical_path_trace_design.md`](mega_moe_a8w4_critical_path_trace_design.md)。

## 摘要

A8W4 当前走 `MegaMoe` 逐专家/分阶段主流程：AIC 先跑全部专家的 GMM1（Phase1）、再跑全部专家的 GMM2（Phase2），两段的空泡与富余方向相反却无法互相吸收。本设计把 A8W4 迁移到 **wave 滚动流水**（对齐 A8W8 已用的 `MegaMoeWave`），核心结论：

- **目标粒度：2 专家/wave**。AIC 按 `G1(e0)→G1(e1)→G2(e0)→G2(e1)` 连续运行，SwiGLU 的 12us 被同 wave 的另一个 GMM1 掩盖，AIC 无空泡、连续 111us/专家。
- **模拟收益：算子总时长 ≈ 508us**（当前分阶段 ≈ 528us，约 -20us / 4%），且已贴近单核代表视图的模型下限。
- **两个反直觉结论**：① 1 专家/wave 反而更慢（≈570us，每专家多 12us "等 S" 空泡）；② wave 不是越大越好（4 专家/wave 回到 528us），**2 是甜点**。
- **后续真正提速点在 combine**（48us/专家、AIV1-bound），wave 结构救不了它；combine 跨核并行可再降到 ~498us。

> ⚠️ 说明：当前收益量化基于单一 case（4 专家、固定 token 分布）的模拟，真实时长随 token 分布浮动，落地后需用 A8W4 trace 实测复核。

---

## 1. 背景与目标

### 1.1 现状

- A8W8 已使用 `MegaMoeWave`（`mega_moe_wave.h`）：把多个专家组成 wave，使 Dispatch / GMM1 / SwiGLU / GMM2 / Combine 交叠执行。
- **A8W4 仍使用 `MegaMoe` 逐专家主流程**（`mega_moe.h` + `mega_moe_impl.h` 的 `AicComputeA8W4` / `AivGmm2PostA8W4`）。两段之间的衔接是**逐 wave / 逐 tile 的 flag**（`dispatchToGmm1Flag`、`swigluToGmm2Flag`、`gmmToEpilogueFlag`）；A8W4 没有任务型 EndSync、也没有 SyncAll 全局屏障（`SyncAll` 属 topkWeightsPrefetch 路径）。

### 1.2 当前流水的三个结构性浪费

用 A8W4 关键路径 trace 数据把当前流水画出来（单核代表视图，4 专家）：

| 角色 | 职责 | 单专家时长 | 关键路径判断 |
| --- | --- | --- | --- |
| AIC | GMM1 / GMM2 矩阵计算 | 73us / 38us | **Phase1 bound**（73us/专家） |
| AIV0 | GMM1/GMM2 权重 prologue（W4→W8） | 与对应 GMM 重叠、领先 2~5us 结束 | 不在关键路径 |
| AIV1 | Dispatch（跨卡）/ Swiglu / Combine | 30us / 12us / 48us | **Phase2 bound**（48us/专家） |

![当前 A8W4 分阶段流水（算子总时长 ≈ 528us）](../tools/critical_path_trace/artifacts/a8w4_pipeline_overview.svg)

三个结构性浪费：

1. **Phase1 的 AIV1 空泡 ~31us/专家**（等 GMM1；稳态下限 = 73 − 30 − 12）。
2. **Phase2 的 AIC 富余 ~10us/专家**（combine 48us > G2 38us，AIC 空转）。
3. **两段互补却无法互相吸收**：AIV1 的空泡与 AIC 的富余方向相反，在"全部 G1 → 全部 G2"的分阶段结构下各自独立存在。

### 1.3 目标

把 A8W4 迁移到 wave 滚动流水，使：

- AIC 的 GMM1/GMM2 **连续工作、无空泡**；
- AIV1 的 Dispatch / Swiglu 与 Combine **跨专家重叠**；
- AIV0 的 prologue **跨 wave 预取**（本身就不在关键路径，无代价）；
- 用**最小改动**先验证 wave 循环结构、量化收益，而不是只改代码结构。

---

## 2. 优化点

基于 trace 数据与代码实现，wave 化改造的核心优化点：

| # | 优化点 | 现状（分阶段） | wave 化后 |
| --- | --- | --- | --- |
| 1 | AIC 连续性 | Phase1 G1（73us/专家）+ Phase2 G2（38us/专家）分两段 | `G1(B)→G2(B)→G1(B+1)` 单循环连续 |
| 2 | 消除"等 S"空泡 | 分阶段靠 S 藏在 G1(e+1) 后 | wave 内 S(e) 藏在同 wave 的 G1(e+1)/G2(e) 后 |
| 3 | G2 / combine 前移 | G2(0) 要等全部 G1 完成（322us 才起） | 2 专家/wave 下 G2(0) 176us 就起 |
| 4 | combine 与下一 wave 重叠 | C(e) 只与相邻专家部分重叠 | 结构上保证 C(e) 与下一 wave 的 G1 重叠 |
| 5 | dispatch 提前预取 | D(e) 逐专家 | D(e+1) 在 C(e) 之前完成 |
| 6 | prologue 跨 wave 预取 | P/Q 与对应 GMM 重叠、非关键路径 | 天然可提前，无代价 |

---

## 3. 收益来源与量化

### 3.1 三种形态的模拟对比（4 专家 case，实测单专家时长）

| 形态 | 算子总时长 | 相对现状 | 说明 |
| --- | --- | --- | --- |
| 当前分阶段 | ≈ 528us | — | S 藏在 G1(e+1) 后、AIC 连续，但 G2 全晚 |
| wave-1（1 专家/wave） | ≈ 570us | **+42us（变慢 ~8%）** | 负面案例：G2(e) 等 S(e) 的 12us 空泡无处藏 |
| **wave-2（2 专家/wave）** | **≈ 508us** | **−20us（提升 ~4%）** | **最优**：S 藏好 + G2/combine 前移 |

### 3.2 wave-1：为什么 1 专家/wave 不划算

![wave-1 模拟（算子总时长 ≈ 570us）](../tools/critical_path_trace/artifacts/a8w4_wave1_pipeline_overview.svg)

1 专家/wave 时 AIC 按 `G1(e)→G2(e)` 逐专家跑，而 **G2(e) 依赖 S(e)**（swiglu 12us）；wave 内没有第二个专家可藏，AIC 每个专家插入 **12us "等 S" 空泡**，周期 123us/专家，4 专家 ≈ 570us，比现状还慢。

> **结论：1 专家/wave 只适合用来验证 wave 循环结构本身，不是收益形态。**

### 3.3 wave-2：收益形态

![wave-2 模拟（算子总时长 ≈ 508us）](../tools/critical_path_trace/artifacts/a8w4_wave2_pipeline_overview.svg)

2 专家/wave 时 AIC 按 `G1(0)→G1(1)→G2(0)→G2(1)` 跑：**S(0) 做 12us 期间，AIC 正在跑 G1(1)（73us）**，等 S 空泡被吸收，AIC 连续 111us/专家、无空泡。同时第一波 G2(0) 提前到 176us（vs 分阶段 322us），combine 跟着前移，整体 ≈ 508us。

### 3.4 收益来源分解

1. **消除"等 S"空泡**：相对 wave-1，2 专家/wave 把 4 × 12us = 48us 的空泡全部藏掉。
2. **G2 / combine 前移**：相对现状，第一波 G2(0) 提前 146us（322→176），AIV1 的 combine 流提前清空，使最后一波 C(2)/C(3) 能贴着 G2 tile 就绪点（412/450us）启动，尾段缩短约 20us。
3. **AIC / AIV0 满载**：AIC 的 444us GMM 工作连续执行；AIV0 的 P+Q 与 AIC 持平，仍不在关键路径。

### 3.5 为什么 wave-2 已是模型下限（508us）

两个约束同时卡住，wall 无法更低（单核代表视图 + combine 串行模型）：

- **AIC 连续**：G2(3) 最早只能 `30 + 4×73 + 3×38 = 436us` 起；
- **combine 串行**：`C(3) ≥ C(2) 结束 ≥ G2(2) 首 tile(412) + 48 = 460`；
- 因此 **wall ≥ 460 + 48 = 508us**，wave-2 恰好每个约束都顶满。

> **注意：wave 不是越大越好。** N=4（= 当前分阶段结构）因为 G2(0) 被推到 322、第一波 combine 全部延后、AIV1 在 412us 时还忙着 C(1)，C(2)/C(3) 被拖到 432/480us，反而回到 528us。**2 专家/wave 是甜点。**

---

## 4. 改造方案

> 本节的"怎么改"落到**函数与数据结构层面**，与现状代码（`MegaMoe::Process()`）和 A8W8 wave 参考（`MegaMoeWave`）一一对应，可直接指导实现。

### 4.1 现状代码结构：两段式 for 循环就是"分阶段"的根源

A8W4 主流程 `MegaMoe::Process()`（`mega_moe.h`）当前是**两个独立的逐专家全量循环**：

```cpp
// —— GMM1 阶段（Phase 1）——
for (localExpertId = 0; localExpertId < moeExpertPerRank_; localExpertId++) {
    AIV1: prefetch dispatch(localExpertId + 1);   // SendCntCal + MetaInfoCalAndDispatch
    GroupMatmulWithSwigluQuant(...);              // GMM1 + SwiGLU
}
EndSync(vecSetSyncCom);   // 或 topkWeightsPrefetch 的 allDone 轮询 + SyncAll  ← 分阶段串行点

// —— GMM2 阶段（Phase 2）——
for (expertIdx = 0; expertIdx < moeExpertPerRank_; expertIdx++) {
    GroupMatmulWithCombine(...);                  // GMM2 + Combine
}
EndGMM2Sync(vecSetSyncCom, gmm2PingPongIdx_);
```

两个循环之间被 `EndSync` / allDone / `SyncAll` **强制切断**：必须等全部专家的 GMM1 + SwiGLU 完成，才进入 GMM2。这就是分析中"Phase1 与 Phase2 的互补浪费无法互相吸收"的**代码根源**。dispatch 预取目前只发生在 GMM1 循环内（e+1 与 e 并发），没有延伸到 GMM2 / combine。

### 4.2 目标结构：单 wave 滚动循环

把两个循环合并为**一个 wave 循环**，每 wave 含 `expertsPerBatch`（目标 2）个专家，顺序参考 A8W8 `MegaMoeWave::ProcessRoutedExpertWaves`：

```cpp
for (waveBegin = 0; waveBegin < moeExpertPerRank_; waveBegin += waveExpertNum) {
    waveExpertNum = GetWaveExpertCount(waveBegin, moeExpertPerRank_, expertsPerBatch); // 尾 wave 可能不足

    AIV1: DispatchExpertsUntil(下一 wave 结束);    // 预取 B+1（复用 SendCntCal / MetaInfoCalAndDispatch）
    AIC : ProcessGmm1Wave(waveBegin, waveEnd);    // GMM1(B) 逐 expert；AIV0 并发 P/Q prologue(B)
    AIC : ProcessGmm2Wave(waveBegin, waveEnd);    // GMM2(B) 直写 GM；逐 expert 后 PublishGmm2Ready(e)
    AIV1: ProcessCombineExperts(waveBegin, waveEnd); // WaitGmm2Ready(e) + Combine(e)
}
// 收尾：仅剩 AIC→AIV0 的 UB 消费确认回收（参考 wave 路径 EndSync<GMM1_UB_PINGPONG>）
```

- AIC 顺序即 `G1(e0)G1(e1)G2(e0)G2(e1)`，**S(e) 被同 wave 的 G1(e1)/G2(e0) 掩盖**，AIC 连续 111us/专家。
- **删除两阶段之间的 `EndSync` / allDone / SyncAll**：wave 之间不再有批次级全局 barrier，只靠 wave 内 flag 依赖 + 专家粒度 GMM2-ready 推进。

### 4.3 关键改造点（函数/数据结构级）

| # | 改造点 | 现状（A8W4） | 目标（参考 A8W8 wave） |
| --- | --- | --- | --- |
| 1 | 主循环 | `Process()` 两个 for | 合并为 wave 循环，拆出 `ProcessGmm1Wave` / `ProcessGmm2Wave` / `ProcessCombineExperts(begin, end)` |
| 2 | GMM1 批量 | 循环内直接调 `GroupMatmulWithSwigluQuant` | 包进 `ProcessGmm1Wave`；AIV0 预取 P/Q prologue 到 B+1 |
| 3 | GMM2 批量 | 循环内直接调 `GroupMatmulWithCombine` | 包进 `ProcessGmm2Wave`；每 expert 结束后 `PublishGmm2Ready(e)` |
| 4 | combine 触发 | 逐 tile 轮询 `gmmToEpilogueFlag` | `ProcessCombineExperts`：`WaitGmm2Ready(e)`（专家粒度）触发 + tile 级 combine |
| 5 | dispatch 预取 | GMM1 循环内 e+1 | `DispatchExpertsUntil(B+1)`，跨到 combine 之前 |
| 6 | 去串行点 | GMM1 循环后 `EndSync`/allDone/SyncAll | 删除；改 wave 内 flag 推进 |
| 7 | 模板路由 | `apt.cpp` 由 dtype 对决定（`isA8W8DtypePair`→`MegaMoeWave`，否则→`MegaMoe`） | 让 A8W4（fp8×fp4）也进入 wave 实例（见 4.5） |
| 8 | 空专家守卫 | 已有 `UpdateGroupParams` 跳过空专家 | 保留，并 guard `BlockMmad` / `BlockPrologue` 构造（`startLoopIdx < tileNum`），防空专家 flag 溢出 |

### 4.4 依赖、同步与缓冲

- **沿用 wave 粒度 flag**：`dispatchToGmm1Flag[waveIdx]`、`swigluToGmm2Flag[waveIdx]`（`IS_WAVE_FLAG_GRAINED`）；combine 保留 `gmmToEpilogueFlag` 做 tile 级细化。
- **补专家粒度 GMM2-ready 扇入**（复用 A8W8 模式）：workspace 按 `[slot][AIC][64B cache line]` 排布；每个 AIC 在 `FIX_S`（保证 GMM2 直写 GM 完成）后 `PublishGmm2Ready(slot)` 写 1；AIV1 `WaitGmm2Ready(slot)` 搬入并归约所有 AIC 的到达数。作为 combine 的主触发，tile 级标志只提供首 tile 提前量（G2 启动后 14us）。
- **wave 双缓冲**：dispatch 槽、GMM2-ready slot 按 wave / 专家索引环形复用；AIV1 的 D(B+1) 与 C(B) 复用同一片 UB / 事件（参考 `DrainCombineRowRing` 与事件槽复用）。
- **无批次级全局 barrier**：不用 EndSync / SyncAll 做 wave 同步。

### 4.5 两条实施路线（决策点）

- **路线 A（推荐，长期）——把 A8W4 接入 wave 框架**：给 `MegaMoeWave` 增加 `IsA8W4` 模板参数，把 A8W4 特有的 `BlockPrologue`（W4→W8，AIV0）与 fp8×fp4 `BlockMmad` 接进 wave 主循环；`apt.cpp` 的 wave 路由从"仅 A8W8 dtype 对"扩为"含 A8W4"。复用整套 wave 结构（GMM2-ready 扇入、combine 分核、`startBlockIdx_` 游标），改动集中在模板差异，长期维护最干净。
- **路线 B（过渡，风险小）——先在 `MegaMoe::Process()` 内做 wave 循环**：只合并 4.1 的两个 for 为 wave 循环、删除中间 barrier、补 GMM2-ready，不改类层次。改动面小、可快速验证收益；验证后再决定是否抽到 `MegaMoeWave`。
- **建议：先 B 后 A**。B 用最小改动确认"wave 结构 + 2 专家/wave"能拿到模拟的收益（AIC 连续、无等 S、总时长下降），再按 A 把成熟形态沉淀进 wave 框架，避免一上来就动模板矩阵。

### 4.6 分步实施与验收

1. **结构先行**：合并两个 for 为 wave 循环（可先用 `expertsPerBatch = 1` 验证循环结构——按 3.2 这是负面形态，只验证结构、不做目标）。
2. **切 2 专家/wave**：`expertsPerBatch = 2`，确认 AIC 连续（G2(e) 不再等 S(e)）。
3. **去串行点**：删除 GMM1 循环后的 `EndSync` / allDone / SyncAll，改 wave 内 flag 推进。
4. **补 GMM2-ready 扇入**：实现 `PublishGmm2Ready` / `WaitGmm2Ready`（`[slot][AIC][64B]`），combine 改按专家触发。
5. **跨 wave 预取**：AIV1 `DispatchExpertsUntil(B+1)`、AIV0 P/Q prologue(B+1)。
6. **边界收口**：尾 wave 不足（`GetWaveExpertCount`）、空专家守卫、共享专家保持前后置、combine-quant 路径（`gmm2MmadResPtr` 直写）与 no-quant 路径（GMM2 直写 GM + `FIX_S` 发布）分别验证。
7. **trace 验收**：复用 A8W4 关键路径 trace（GMM1 wave 采样、combine tile 采样已就位），量化：AIC 空泡率、G1→S、S→G2、G2→C 三处 handoff 等待、算子总时长，与模拟 508us 对比；**确认"真正缩短了哪些等待，而不是只改了代码结构"**。

### 4.7 风险与边界

- **AIV0 潜在共瓶颈**：P+Q ≈ 111us/专家 ≈ AIC；2 专家/wave 下不构成瓶颈，wave > 2 需复查 AIV0 占用。
- **combine 仍是真瓶颈**：48us/专家（AIV1-bound），wave 只消除 G1→G2 空泡、不动 combine；后续靠 combine 跨核并行（→ ~498us）/ 减时长。
- **case 局限**：模拟基于单一 case（4 专家、固定 token 分布）；真实时长随 token 分布浮动，落地后用真实 trace 复核，再决定是否微调 `expertsPerBatch`。
- **同步语义**：wave 化后仍无 EndSync / SyncAll 全局屏障（那是 topkWeightsPrefetch 路径的机制），保持 flag 依赖语义不变，避免引入新的全局同步。

---

## 5. 结论

- A8W4 当前分阶段流水（≈528us）结构上无法让 Phase1 的 AIV1 空泡与 Phase2 的 AIC 富余互相吸收。
- **wave-2（2 专家/wave）是收益形态**：消除"等 S"空泡 + G2/combine 前移，模拟总时长 ≈ 508us（相对现状约 −20us / 4%）。
- 1 专家/wave（570us）与 4 专家/wave（528us）都不如 2 专家/wave，**2 是甜点粒度**。
- 后续真正的提速点在 **combine**（跨核并行 / 减时长），而非 wave 大小。

---

## 6. 参考资料

- `mega_moe_a8w4_critical_path_trace_design.md` —— 关键路径打点设计（本设计的量化前置）
- `mega_moe_wave.h` —— A8W8 wave 流水参考实现
- `mega_moe_impl.h` / `mega_moe.h` —— A8W4 现状主流程
- 模拟图：`tools/critical_path_trace/artifacts/a8w4_pipeline_overview.svg`（现状）、`a8w4_wave1_pipeline_overview.svg`（wave-1）、`a8w4_wave2_pipeline_overview.svg`（wave-2）
