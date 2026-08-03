# A5 500000系 vs A3 1300000系 CutOrigin 对比分析

> **生成日期**：2026-08-18
> **目的**：在决定 A3 1300000/1301000 模板向 950（arch35）迁移方向前，先精确对比 A5 现有 500000 系与 A3 1300000 系 CutOrigin 的架构与语义差异。
> **结论先行**：A5 500000 系已是 A3 1300000 系裁剪/增强后的超集（剪掉角色、增强流水），**直接复用 A5 模板、放弃 A3 模板迁移**是更优方向。全案见 §10。

---

## 1. 定位对照

| 维度 | A5 500000 系 | A3 1300000 系 |
|------|-------------|---------------|
| 架构代号 | arch35 / 950（RegBase） | base / 非950 |
| host 入口 | `op_host/moe_init_routing_v3_tiling_arch35.cpp`（`MoeInitRoutingV3TilingArch35`） | `op_host/moe_init_routing_v3_tiling.cpp`（`MoeInitRoutingV3TilingBase`） |
| kernel 入口 | `op_kernel/moe_init_routing_v3_apt.cpp` | `op_kernel/moe_init_routing_v3.cpp` |
| CS CutOrigin 模板 | `op_kernel/arch35/moe_v3_counting_sort_cut_origin_{base,unquantized,static_quant,dynamic_quant}.h` | `op_kernel/moe_v3_cut_origin_t.h`（单模板含全部量化分支） |
| tilingKey | **500000**（unquant）/ **510000**（static）/ **520000**（dynamic）+ `rowIdxType_*1000 + dropPadMode_*100` | **1300000**（GATHER）/ **1301000**（SCATTER），**只编码 rowIdxType** |
| TilingData | `MoeInitRoutingV3Arch35TilingData`（含 `countingSortParamsOp` 子结构） | `MoeInitRoutingV3TilingData`（含 `gatherOutComputeParamsOp`） |
| 分派变量 | `countingSortMode_`（0=off / 1=FullLoad / 2=CutOrigin） | `countingSortFullLoad_`（−1=builtin / 0=CutOrigin / 1=FullLoad）|

> tilingKey 数值差 10 倍、编码位不同：base `TILINGKEY_BASE=1M, SORT_CORE=100K`；arch35 `TILINGKEY_BASE=10M, SORT_CORE=1M`。两套口径不可混用。

---

## 2. 触发条件差异（host 端）

### 2.1 A3 `ComputeCountingSortCutOriginTiling()`（tiling.cpp L902-994）

```
IsFullLoad()==false
  ├─ L1 前置：n∈[256,1M] && 非SCATTER-PerformanceMode排除 && ShouldEnableCountingSortByShape()
  │     （ShouldEnable：expertNum≤1024；dropless: 2*LE≤E；quant!=DYNAMIC；排除小T静态量化；dropPad需 n≥512）
  ├─ L2 FullLoad八条 AND 任一不满足 → countingSortFullLoad_=0 → L3
  └─ L3 CutOrigin 判定（L902-994）：
        csColsLoops>1 && quantMode_!=UN_QUANT → 回退 builtin(countingSortFullLoad_=-1)
        否则 → CutOrigin(1300000/1301000)
```

- **列切分**：A3 支持 `csColsLoops` 多列循环（大列 `cols×dtype > maxPerLoopColsBytes` 时切分）。**多列 + 非量化除外会回退 builtin**（L960）。
- **量化**：CutOrigin 全量化通用（unquant/static/dynamic 都经同一 `MoeV3CutOriginT`，运行时 `quantMode_` 分支）。

### 2.2 A5 `IsCountingSortApplicable()` + `ComputeArch35CountingSortCutOriginTiling()`（arch35.cpp L1951-1997 / L2089-2178）

```
countingSortMode_ = 2（FullLoad UB 预算不够时落到 CutOrigin）
```

| A5 判定 | 与 A3 差异 |
|---------|-----------|
| `quantMode_` 白名单：**仅 UNQUANT / STATIC**，dynamic 硬关 CS | **A3 支持 dynamic**（L3 全量化通用）|
| `actualExpertNum <= CS_MAX_ACTUAL_EXPERT_NUM = 256` | **A3 上限 128**（L2 条件7）|
| dropless：`actualExpertNum==expertNum_ || actualExpertNum*2<=expertNum_` | A3 无全专家范围例外（L1 `2*LE≤E` 严格）|
| `IsAnyDynamicQuantCase()` 直接 false | A3 不排除 dynamic |
| 列切分：**colsLoops 恒 1**，`csColsLoops`**字段复用为双缓冲使能信号**（1=双缓冲可容纳 / 0=降级串行） | A3 的 `csColsLoops` 是真列循环次数 |

### 2.3 触发条件差异汇总

| 项 | A5 500000 系 | A3 1300000 系 |
|----|-------------|---------------|
| 列切分 | ❌ 恒 1（大列降级串行，不切） | ✅ 可 `csColsLoops>1`（大列切 loop）|
| dynamic 量化 | ❌ 硬关 CS | ✅ 支持（经同一模板）|
| static 量化 | ✅ 独立子模板 | ✅ 单模板内分支 |
| LE 上限 | 256 | 128 |
| dropless 全专家范围（LE==E）| ✅ 例外放行 | ❌ 拦截（回到 builtin）|
| dropPad | ✅ 支持（n≥512）| ✅ 支持 |
| 分派粒度 | countingSortMode_（FullLoad↔CutOrigin 按 UB 预算）| countingSortFullLoad_ 三层判定 |

---

## 3. 模板组织结构（kernel 端）

### 3.1 A5：基类 + 量化子类（编译期去分支）

```
MoeV3CountingSortCutOriginBase<T>          // arch35/moe_v3_counting_sort_cut_origin_base.h
  ├─ InitCommon / PrefillAndSync / ZeroOutExpandedDropPad
  ├─ ComputeCommonUbLayout（公共 + Phase C 双缓冲统一布局）
  ├─ LoadExpertIdx / FilterAndCount（默认纯标量扫描）
  ├─ FilterAndCountVector（宏 MOE_CS_CUTORIGIN_VECTOR_FILTER，默认关）
  ├─ FilterAndCountHistogram（宏 MOE_CS_USE_HISTOGRAM，默认关）
  ├─ WriteExpertCountToWorkspace / ComputeGlobalOffset / WriteExpertTokens
  └─ GatherAndWrite / GatherOneRow / GatherAndWriteDoubleBuffer（由子类实现）

子类：
  MoeV3CountingSortCutOriginUnquantized<T> // unquantized.h：x 直接搬运 + isInputScale 透传 + DB ping-pong
  MoeV3CountingSortCutOriginStaticQuant<T> // static_quant.h
  MoeV3CountingSortCutOriginDynamicQuant<T>// dynamic_quant.h（A5 下不激活，因 IsCountingSortApplicable 关闭）
```

- 量化差异集中在子类 `GatherOneRow` / `Quant*OneRow`，**基类 Phase A/B 完全共享**，编译期分支，无运行时开销。

### 3.2 A3：单模板全量化分支（运行时）

```
MoeV3CutOriginT<T>                          // moe_v3_cut_origin_t.h（无基类/子类分层）
  FilterAndCountChunked（ep_==0 免过滤 / ep_==1 向量过滤，chunk 分批）
  WriteExpertCountToWorkspace / ComputeGlobalOffset / WriteExpertTokens
  GatherAndWriteChunked（colsLoops==1 unquant DB / static DB / colsLoops>1 串行）
  GatherAndWriteDroppad
  PadZerosForDropPad
```

- 单模板内用 `if (quantMode_ == ...)` / `if constexpr` 分支；`shortPath_`（coreEntries≤2048 && ep_==1）走 UB 驻留免 workspace。
- **ep_==0 特判**：全专家范围免过滤，直接全载计数。

---

## 4. Phase A（过滤 + 计数）

| 维度 | A5 | A3 |
|------|----|----|
| 默认过滤 | 纯标量单扫描（`FilterAndCount`，逐 entry `GetValue`）| `ep_==1`：向量过滤（Cast+CompareScalar+And+GatherMask）按 chunk 缩小后标量累加；`ep_==0`：免过滤全载 |
| 向量过滤宏 | `MOE_CS_CUTORIGIN_VECTOR_FILTER`（默认**关**，该宏引入收益有限：active_range 全通过时压缩无淘汰）| 非宏，`ep_==1` 恒走向量过滤 |
| chunk 划分 | `filterChunkSize_` 仅决定 `csChunkSize` 字段，标量扫描**整个 coreEntries_** 一次 | `maxChunks_` 分批，每 chunk 单独 DataCopy + 过滤 |
| 免过滤路径 | ❌ 无（A5 无 ep_==0 特判）| ✅ `ep_==0` 全载免过滤 |

> 实测背景（记忆 `project_moe_v3_histogram_ab.md`）：向量过滤/直方图在两轮 A/B 均负收益，故 A5 默认关闭。A3 的 ep_==0 免过滤路径在 dropless 全专家范围（LE==E）场景有优势——但此场景 A5 用例外放行走 FullLoad，不走 CutOrigin。

---

## 5. Phase B（核间归约 + 全局偏移）

| 维度 | A5 | A3 |
|------|----|----|
| 归约方式 | 一次性 `filterNeedCoreNum_*expertCountStride_` DMA 全载+向量 Add 累加 | 分批 batchBuf（`batchCores` 按 UB 预算算）逐批 DMA+Add |
| expertTokens | 全部模式（COUNT/CUMSUM/KEY_VALUE）由 core0 统一结构 | 同样 core0，但 COUNT 用 `Cast`、CUMSUM 用标量累加 |
| dropPad seed | `expertCountLocal[e]=prefixForExpert`（跨核 per-expert 前缀）| `(expertStart_+e)*expertCapacity_ + clampedPrefix`（含基址偏移）|

> 语义等价：都是"总 count + 前缀偏移"决定写入起点。A5 因 arch35 全载 UB 充裕，归约不分批。

---

## 6. Phase C（搬运）——核心差异

### 6.1 双缓冲演化

| 维度 | A5（500000 系当前态）| A3（1300000 系当前态）|
|------|-------|-------|
| 默认路径 | `GatherAndWriteDoubleBuffer`：x 双 slot ping-pong | `GatherAndWriteChunked`：x 双 slot ping-pong |
| 使能信号 | `csColsLoops` 复用 = `csDbEnable_`（tiling 判定，1=DB / 0=串行）| `colsLoops_==1` 走 DB；`colsLoops_>1` 走串行 |
| 降级路径 | 串行 `GatherAndWrite`（**保留**）| 串行（colsLoops>1 分支，**保留**）|
| rowIdx 写出优化 | 连续批 staging 一次 DMA（`idxStageBuf`）+ 非连续逐行双 slot | 逐行 V 写 idxBuf（`Duplicate/Adds`）+ DMA |

> 记忆注记：`project_moe_v3_a3_vs_a5_rootcause.md` 明确指出 A3→A5 迁移劣化的 **Phase C 双缓冲丢失**是根因之一，且 A5 已经通过把 `GatherAndWriteDoubleBuffer` 固化为默认实现修复。因此 A5 当前 Phase C 已追平（甚至超）A3 的 `colsLoops==1` DB 路径。

### 6.2 dropPad 搬运

- **A3** 单独 `GatherAndWriteDroppad`：按 srcRow 分组双缓冲，跨 colsLoop 搬运，跳槽位预计算 newPos。
- **A5** 在统一 `GatherAndWrite` / `GatherAndWriteDoubleBuffer` 内嵌 `dropPadMode_==DROP_PAD` 分支（`slot>=expertCapacity_` 丢弃写 −1），零填充由 `ZeroOutExpandedDropPad` + scale 透传零清处理。

### 6.3 UB 布局纪律

| 维度 | A5 | A3 |
|------|----|----|
| 布局函数 | `ComputeCommonUbLayout()`（kernel 内，tiling 精确镜像）| 分散在 Init/各函数局部算 offset |
| 镜像纪律 | 强制：host `ComputeArch35CountingSortCutOriginTiling` 的 serialBytes/dbExtraBytes 必须 ≥ kernel `InitBuffer` 实占，**注释明示不可叠加保守项** | host 在 `ComputeCountingSortCutOriginTiling` 按 Phase A/B/C 估算 |
| 向量过滤缓冲 | 宏控制恒算或仅启用时算 | `ep_==1` 恒入局 |

---

## 7. 量化支持对照

| 量化 | A5 500000 系 | A3 1300000 系 |
|------|-------------|---------------|
| UNQUANT | `_unquantized.h` 子模板 | 单模板内 `quantMode_==-1` 分支 |
| STATIC | `_static_quant.h` 子模板（独立 DB：MTE2↔V 重叠）| 单模板内 `quantMode_==0` DB / 串行 |
| DYNAMIC | `_dynamic_quant.h` 存在但 **IsCountingSortApplicable 关闭，不激活** | 单模板内 `quantMode_==1` 集成（经 smooth 行）|
| MXFP8/FP8 等 | 不走 CS（白名单外）| 不走 CS |

> A3 在非950路径对 dynamic 的 CS 支持，A5 侧被 arch35 判定主动放弃（A5 dynamic 仍走旧 FullLoad/gather 模板）。

---

## 8. 已知 A3→A5 迁移回归点（与本文档的对应）

| 记忆中的迁移根因 | 对应差异 | A5 现状 |
|-----------------|---------|---------|
| Phase C 双缓冲丢失（主因）| §6.1：A3 有 DB，早期 A5 无 → 劣化 | **已修复**：`GatherAndWriteDoubleBuffer` 固化默认 |
| Phase A 向量过滤丢失（次因）| §4：A3 ep_==1 向量过滤，早期 A5 纯标量 | 已实现但**默认关**（宏 MOE_CS_CUTORIGIN_VECTOR_FILTER）|
| 动态量化关 CS（用户决策）| §7：A5 `IsCountingSortApplicable` 硬关 dynamic | 维持关闭 |
| dropless 全专家范围例外放行（回归点 fl_k8 +65.9%）| §2：A5 `actualExpertNum==expertNum_` 例外存在、A3 无 | **维持**（但该场景走 FullLoad 而非 CutOrigin 时劣化，见 `cs_fullload_k8_degradation_analysis.md`）|

---

## 9. tilingKey 分派对照（arch35）

```
A5 GetTilingKey() CS 分支（L182-195）:
  CS_CUTORIGIN_UNQUANT_TILINGKEY_BASE(500000) + rowIdxType_*1000 + dropPadMode_*100
  CS_CUTORIGIN_STATIC_TILINGKEY_BASE(510000) + ...
  CS_CUTORIGIN_DYNAMIC_TILINGKEY_BASE(520000) + ...
kernel moe_init_routing_v3_apt.cpp L361-453 按 500000/510000/520000 前缀 + _S/_D 后缀分派类型守卫模板。
```

| A5 key | 含义 | A3 等价 key |
|--------|------|------------|
| 500000（GATHER, dropless）/ 500100 | 非量化 CutOrigin | 1300000 |
| 501000（SCATTER）/ 501100 | SCATTER 变体 | 1301000 |
| 510xxx | 静态量化 | —（A3 无独立 key，同一 1300000）|
| 520xxx | 动态量化 | —（A3 无独立 key）|

> A3 的 1300000/1301000 只切 GATHER/SCATTER，量化与 dropPad 在 kernel 运行时由 `quantMode_`/`dropPadMode_` 字段分流；A5 把量化/dropPad 编进 key，切到独立子模板（编译期）。

---

## 10. 结论：迁移方向建议

1. **A5 500000 系是 A3 1300000 系的"裁剪 + 增强"超集**：
   - 剪掉：列切分（恒 1）、full dynamic 支持、ep_==0 免过滤特判。
   - 增强：基类/子类分层的编译期去分支、Phase C 双缓冲固化默认 + 连续批 staging、tiling 精确镜像 UB、量化编码进 key 切独立子模板。

2. **A3 1300000 系相对 A5 的独占能力**仅剩：
   - `csColsLoops > 1` 真列切分（但非量化外会回退，且 A5 用降级串行替代，净效果更简单）；
   - `ep_==0` 免过滤（全专家范围，A5 走 FullLoad 例外放行覆盖）。

3. **直接复用 A5 500000 系比迁移 A3 更优**：
   - 迁移意味着把 A3 的运行时分支 + 列切分 + ep_==0 特判搬进 arch35，而 arch35 的 `MoeInitRoutingV3Arch35TilingData`、UB 镜像纪律、`countingSortParamsOp` 都是为 A5 模板定制的——硬套 A3 结构会破坏镜像一致性与已落地的双缓冲修复。
   - A5 已补齐 A3 的 Phase C 双缓冲（迁移劣化主因已解决），剩余独占能力收益有限（列切分仅在超大列+非量化有用；免过滤被 FullLoad 覆盖）。

4. **若确要真正复用 A3 模板**（用户此前选"迁移 A3 1300000/1301000"），唯一自洽路径是：
   把 A3 的伪装的非950文件整体作为另一套 CS CutOrigin 实现平移到 `op_kernel/arch35/`，新增独立 tilingKey 与分派，并改造 `MoeV3CutOriginT` 读 `MoeInitRoutingV3Arch35TilingData` + 按 A5 的 `ComputeCommonUbLayout` 镜像纪律重校 UB。**成本 ≈ 重写一次**，收益仅是列切分单一能力。

> 建议在正式规划迁移前，与用户对齐：是否可放弃 A3 独占的列切分/ep_==0 能力，从而启用 A5 模板。

---

**文档版本**：v1.0（2026-08-18）
**关联文档**：
- `cs_a3_fullload_entry_conditions.md`（A3 FullLoad/CutOrigin 全部启用条件）
- `cs_fullload_k8_degradation_analysis.md`（fl_k8 A5 FullLoad 劣化根因）
- `project_moe_v3_a3_vs_a5_rootcause.md`（迁移根因总览）
