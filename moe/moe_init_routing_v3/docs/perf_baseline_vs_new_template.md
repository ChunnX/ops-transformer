# MoeInitRoutingV3 新 CountingSort 模板性能验证记录

> 目标：验证新增 CountingSort 模板（FullLoad/CutOrigin）性能**不劣于原代码**。
> 芯片：Ascend950DT (arch35)，device 2。DYN_PERF = kernel 耗时（**越小越好**）。判定：新 ≤ 原（±1% 视为持平）即通过。

---

## 1. 背景与验证机制

- **新代码**：算子仓 `ops-transformer0716`，含 8 个 `moe_v3_counting_sort_*` 模板，命中时 TilingKey = 4xxxxx(FullLoad)/5xxxxx(CutOrigin)。custom_transformer vendor 包。
- **原代码**：备份 `/home1/l00939308/moeInitrouteV3/moe_init_routing_v3`，经确认与 CANN **builtin** 版本 bit 一致（旧全载 `moe_v3_full_load_base.h` diff 一致），**不含** counting_sort 新模板。同批用例走**旧路径**（TilingKey 200000 等）。
- **切换方式**：装 custom 包 → 新代码；卸载 custom 包 → 自动回退 builtin(=原代码)。
- **测量**：TTK 自带 profiling，日志出 `DYN_PERF: <数值>` + `PRECISION_STATUS: PASS/FAIL`。
- **对比集**：8 条核心代表用例，A/B 各测 3 次。

## 2. 核心代表用例

| 组 | 用例名 | 新TilingKey | 关键参数 |
|----|--------|-------------|----------|
| FullLoad 非量化 | moe_v3_cs_fullload_256 | 400000 | n=10000,cols=128 |
| FullLoad 静态量化 | moe_v3_cs_fullload_static_quant | 410000 | n=10000,cols=128 |
| FullLoad 动态量化 | moe_v3_cs_fullload_dynamic_quant | 420000 | n=10000,cols=128 |
| CutOrigin 非量化 | moe_v3_cs_cutorigin_256 | 500000 | n=9000,cols=7168 |
| CutOrigin 静态量化 | moe_v3_cs_cutorigin_static_quant | 510000 | n=9000,cols=7168 |
| CutOrigin 动态量化 | moe_v3_cs_cutorigin_dynamic_quant | 520000 | n=9000,cols=7168 |
| CutOrigin 大列 | moe_v3_cs_cutorigin_cols32768 | 500000 | cols=32768 |
| Switch 中列 | moe_v3_cs_switch_cols1024 | 500000 | cols=1024 |

---

## 3. S1 新代码基线（custom 包已装，走新 CS 模板）

DYN_PERF（越小越好），每用例 3 次；device2 偶发瞬时 NONE 已剔除。

| 用例 | TilingKey | run1 | run2 | run3 | **min** | PRECISION |
|------|-----------|------|------|------|---------|-----------|
| ti=0 fullload_256 | 400000 | 71.581 | 70.461 | 70.489 | **70.461** | PASS |
| ti=17 fullload_static | 410000 | 76.239 | 77.086 | 75.941 | **75.941** | PASS |
| ti=18 fullload_dynamic | 420000 | 113.862 | 114.244 | 114.06 | **113.862** | PASS |
| ti=1 cutorigin_256 | 500000 | 191.853 | 191.01 | 189.612 | **189.612** | PASS |
| ti=19 cutorigin_static | 510000 | 187.387 | 185.98 | 185.748 | **185.748** | PASS |
| ti=20 cutorigin_dynamic | 520000 | 761.446 | 761.314 | 760.325 | **760.325** | PASS |
| ti=27 cutorigin_cols32768 | 500000 | 223.768 | (NONE) | 224.516 | **223.768** | PASS |
| ti=29 switch_cols1024 | 500000 | 119.613 | 119.448 | 119.89 | **119.448** | PASS |

## 4. S3 原代码基线（卸载 custom → builtin）

> S2a 连通性自检：**通过**。卸载 custom 包后 ti=0 TilingKey=11000000（旧 sort-core 路径，非 4xx/5xx）→ 证实走 builtin 原代码；正常出 DYN_PERF=58.262 且 PRECISION PASS。

原代码全部走**旧路径**（TilingKey 10xxxxxx / 11xxxxxx），非 CS 模板。device2 偶发瞬时 NONE 已剔除。

| 用例 | TilingKey | run1 | run2 | run3 | **min** | PRECISION |
|------|-----------|------|------|------|---------|-----------|
| ti=0 fullload_256 | 11000000 | 57.192 | (NONE) | 56.99 | **56.99** | PASS |
| ti=17 fullload_static | 11010000 | 63.412 | 63.146 | 64.221 | **63.146** | PASS |
| ti=18 fullload_dynamic | 11020000 | 94.7 | 95.34 | 94.712 | **94.7** | PASS |
| ti=1 cutorigin_256 | 11000000 | 106.584 | 109.158 | 106.482 | **106.482** | PASS |
| ti=19 cutorigin_static | 11010000 | 98.166 | 101.916 | 100.554 | **98.166** | PASS |
| ti=20 cutorigin_dynamic | 11020000 | 178.513 | (NONE) | 179.302 | **178.513** | PASS |
| ti=27 cutorigin_cols32768 | 10000000 | 207.528 | 208.686 | 207.12 | **207.12** | PASS |
| ti=29 switch_cols1024 | 11000000 | 49.502 | (NONE) | 49.428 | **49.428** | PASS |

> S4 环境还原自检：重装 custom 包后 ti=0 TilingKey 回到 **400000**（新 CS 模板），DYN_PERF=70.39（与 S1 的 70.461 一致）→ 新代码环境已恢复。

---

## 5. S5 对比与判定

> Δ% = (新 - 原)/原 ×100%，越负越好。全部 ≤ +1% → 通过。DYN_PERF = 耗时（越小越好）。

| 用例 | 新(min) | 原(min) | Δ% | 结论 |
|------|---------|---------|-----|------|
| ti=0 fullload_256 | 70.461 | 56.99 | **+23.6%** | 劣化 |
| ti=17 fullload_static | 75.941 | 63.146 | **+20.3%** | 劣化 |
| ti=18 fullload_dynamic | 113.862 | 94.7 | **+20.2%** | 劣化 |
| ti=1 cutorigin_256 | 189.612 | 106.482 | **+78.1%** | 劣化 |
| ti=19 cutorigin_static | 185.748 | 98.166 | **+89.2%** | 劣化 |
| ti=20 cutorigin_dynamic | 760.325 | 178.513 | **+325.9%** | 劣化 |
| ti=27 cutorigin_cols32768 | 223.768 | 207.12 | **+8.0%** | 劣化 |
| ti=29 switch_cols1024 | 119.448 | 49.428 | **+141.6%** | 劣化 |

**总体结论**：**8/8 用例全部劣化，均远超 +1% 判定阈值，验证不通过。**

- **FullLoad 三用例**：劣化 +20~24%（相对温和）。原代码走旧全载路径（11000000/11010000/11020000），新代码走 CS FullLoad（400000/410000/420000）。
- **CutOrigin 三用例 + Switch**：劣化最严重（+78%~+326%）。CutOrigin 动态量化 ti=20 高达 +326%（760 vs 178），是最大劣化点。原代码走旧 sort-core，新 CS CutOrigin 计数段的**纯标量 filter+count+pairs 单扫描**（见 histogram plan §2.2）在 active_expert_range=[0,256] 全通过场景下无任何过滤收益，成为主瓶颈。
- **cols32768 (ti=27)**：劣化最轻 +8%，因大列场景主耗时在 MTE 搬运，计数段占比低。

**根因方向**（S6 待深入）：
1. CutOrigin 计数段纯标量扫描整个 coreEntries_（filter+count+pairs 一体），是 CutOrigin 系用例的主瓶颈。
2. 新 CS 模板相比旧 sort-core 引入了额外的跨核归约/gather 写回开销，且分核/分块策略可能不如旧路径。
3. Switch 用例（cols=1024）本可走更优路径却降级到 CS CutOrigin，模板选择边界可能不当。

---

## 6. S6 劣化根因分析：A3 → A5 迁移差异（2026-08-05）

> 用户澄清：这 2 个 CS 模板（FullLoad / CutOrigin）是从 **A3 迁移**到 A5 的，**A3 上有收益**才迁过来。
> 故本节不直接改代码，而是**逐段对比 A3 与 A5 的 kernel + tiling**，定位迁移过程中丢失了哪些优化，解释 A5 劣化根因。

### 6.1 对比对象（文件定位）

| 层 | A3（有收益，原始） | A5（劣化，当前算子仓） |
|----|--------------------|------------------------|
| kernel 入口 | `moeInitrouteV3/.../op_kernel/moe_init_routing_v3.cpp` | `arch35/moe_v3_counting_sort_dispatch`（各模板 .h） |
| FullLoad CS | `moe_v3_full_load_cut_origin_t.h`（`MoeV3FullLoadCutOriginT`，单文件含量化分支） | `moe_v3_counting_sort_full_load_{base,unquantized,static_quant,dynamic_quant}.h`（base+3子类） |
| CutOrigin CS | `moe_v3_cut_origin_t.h`（`MoeV3CutOriginT`，1535 行单文件） | `moe_v3_counting_sort_cut_origin_{base,unquantized,static_quant,dynamic_quant}.h` |
| tiling | `op_host/moe_init_routing_v3_tiling.cpp`（`ComputeCountingSortCutOriginTiling`） | `op_host/moe_init_routing_v3_tiling_arch35.cpp`（`ComputeArch35CountingSortCutOriginTiling`） |

三相结构两代一致：**Phase A**（核内 filter+count）→ SyncAll → **Phase B**（核间归约+全局偏移）→ **Phase C**（gather 搬运+写 expandedX/rowIdx）。差异集中在 **Phase A 的过滤方式** 与 **Phase C 的搬运流水**。

### 6.2 结论速览：FullLoad 基本等价，CutOrigin 丢了两大优化

| 维度 | FullLoad | CutOrigin |
|------|----------|-----------|
| A5 劣化幅度 | +20~24%（温和） | +78%~+326%（严重） |
| Phase A（过滤计数） | A3/A5 **均为向量过滤**（Cast+CompareScalar+And+GatherMask），基本等价 | A3=**chunk 向量过滤**；A5=**纯标量单扫描** ❌ 丢失 |
| Phase C（搬运） | A3/A5 **均 x 一次性全载 UB**，Phase C 只做 UB→GM 写，基本等价 | A3=**双缓冲 ping-pong 流水**（MTE2/MTE3 跨 pair 重叠）；A5=**单行串行流式** ❌ 丢失 |
| pairs 存储 | 均在 UB（filteredPairsLocal） | A3=**workspace + shortPath UB 优化**；A5=UB filteredPairsLocal |

**核心结论**：A5 **FullLoad** 与 A3 算法几乎一致（+20% 属次要差异，见 §6.5）；A5 **CutOrigin** 在迁移时**丢弃了 A3 的两大关键优化**——(1) Phase A chunk 向量过滤，(2) Phase C 双缓冲 ping-pong 流水，退化为纯标量过滤 + 单行串行搬运，这是 CutOrigin 系（尤其 ti=20 动态量化 +326%）严重劣化的直接根因。

### 6.3 CutOrigin Phase A 差异（过滤+计数）

**A3 `FilterAndCountChunked()`（moe_v3_cut_origin_t.h:372-566）**——按 `filterChunkSize` 分 chunk，`ep_==1` 走**向量过滤**：
```
每 chunk：DataCopyPad 载入 → Cast int32→fp32 → CompareScalar(GE expertStart)+CompareScalar(LT expertEnd)
         → And → GatherMask 压缩出 gatheredExpertLocal + gatheredIdxLocal（合法专家）
         → Adds(-expertStart) 向量化算 expertOffset → 仅对压缩后的 filteredInChunk 个做标量 count 累加
         → pairs 写 workspace（或 shortPath 留 UB）
```
标量循环长度 = `filteredInChunk`（过滤后），且过滤本身向量化。

**A5 `FilterAndCount()`（cut_origin_base.h:430，MOE_CS_USE_HISTOGRAM 默认关）**——**纯标量单扫描整个 coreEntries_**：
```cpp
for (j = 0; j < coreEntries_; j++) {               // 标量扫全部 entry
    expertId = expertIdxLocal.GetValue(j);
    if (expertId < expertStart_ || expertId >= expertEnd_) continue;  // 标量过滤
    ... 标量 count 累加 + 标量填 filteredPairsLocal ...
}
```
**差异**：A5 连过滤都是标量、扫描长度是**未过滤的 coreEntries_**（本测试用例 active_expert_range=[0,256] 全通过，coreEntries_=9000×1）。A3 的向量过滤+GatherMask 压缩在 A5 完全丢失。
> 注：A5 base 里其实有个 `FilterAndCountHistogram()`（向量过滤+Histograms），但 `MOE_CS_USE_HISTOGRAM` 默认关（此前 A/B 负收益回退，见 histogram_counting_optim_plan.md §8）。**A5 遗漏的不是 Histograms，而是 A3 本就有的 chunk 向量过滤（Cast+Compare+GatherMask 那套，非 Histograms）**——A5 默认路径退回了纯标量。

### 6.4 CutOrigin Phase C 差异（搬运，劣化主因）

**A3 `GatherAndWriteChunked()`（moe_v3_cut_origin_t.h:762-979，非量化 colsLoops_==1 路径）**——**双缓冲 ping-pong 流水**：
```
预取 pair0→xBuf0、pair1→xBuf1（SetFlag MTE2_MTE3）
主循环 i：WaitFlag(evMte2[bc]) → 写 expandedX/rowIdx（MTE3）→ SetFlag(evMte3[bc])
         → 若 i+2<batch：WaitFlag(evMte3[bc]) → 预取 pair i+2 到 curXBuf（MTE2）→ SetFlag(evMte2[bc])
尾声：drain 剩余 MTE3 flag
```
关键：**pair i 的 MTE3 写出与 pair i+2 的 MTE2 载入重叠**，两条流水交替占用双 buffer，隐藏搬运延迟。newPos/srcRow/origFlatIdx 提前批量算好（newPosArr/srcRowArr[2048]），标量与 DMA 解耦。

**A5 `GatherAndWrite()`→`GatherOneRow()`（cut_origin_unquantized.h:132/98，静/动量化子类同构）**——**单行串行流式**：
```cpp
for (fIdx = 0; fIdx < filteredCount_; fIdx++) {   // 逐 pair
    ... 算 slot/newPos ...
    GatherOneRow():  写 rowIdx(MTE3) →
        SetWaitFlag<MTE3_MTE2> → DataCopyPad(xLocal ← xGm)  // 载入本行
        SetWaitFlag<MTE2_MTE3>（量化子类再 MTE2_V→量化→V_MTE3）
        DataCopyPad(expandedXGm ← xLocal)                    // 写出本行
}
```
关键：**每行 载入(MTE2)→写出(MTE3) 严格串行**，SetWaitFlag 链把相邻行强制串起来，**无 ping-pong 重叠**。每 pair 都是一次 GM→UB→GM 往返，延迟无法隐藏。

**为何 CutOrigin 劣化远大于 FullLoad**：CutOrigin 用例 cols=7168（大列），Phase C 每 pair 要流式搬 7168 元素整行；A3 双缓冲把这些整行搬运重叠起来，A5 串行则每行延迟全额累加。动态量化 ti=20 还在单行内插入 per-token absmax 两趟 VF（QuantDynamicOneRow），串行链更长 → +326% 最严重。FullLoad 因 x 在 Phase A 已一次性全载 UB（A3/A5 都是 `LoadXBackground`），Phase C 只有 UB→GM 单向写、无 GM→UB 往返，故两代都快、差异小。

### 6.5 FullLoad +20% 的次要差异（非算法级）

A5 FullLoad（full_load_base.h + unquantized.h）与 A3（full_load_cut_origin_t.h）**算法完全同构**：同样 `LoadXBackground` 全载、同样 `FilterAndCountVector` 向量过滤、同样 Phase C UB→GM 写。+20% 差异非来自主算法，候选：
1. **拆分子类的调用开销**：A5 拆 base+子类（虚调用/内联边界、Init 两段式 InitCommon+子类），A3 单类单文件全内联。
2. **UB 布局/对齐**：A5 `ComputeCommonUbLayout` 与 A3 `Init` 的 offset 计算细节（AlignBytes vs OpsBaseCeilAlign）可能导致 buffer 尺寸/DataCopy 效率微差。
3. **Phase C 每行独立 SetWaitFlag**：A5 unquant `GatherOneRow` 每行 `S_MTE3`/`MTE3_S`，A3 `GatherAndWrite` 同为逐行但同一函数体内，同步 flag 发射次数可能不同。
> FullLoad +20% 属次要，优先级低于 CutOrigin。

### 6.6 tiling 差异

| 项 | A3 `ComputeCountingSortCutOriginTiling` | A5 `ComputeArch35CountingSortCutOriginTiling` |
|----|------------------------------------------|------------------------------------------------|
| Phase A 预算 | 按 `chunkAligned`(=CeilAlign(CHUNK_SIZE)) 算 chunk 缓冲 | 按 `coreEntries` **全载** idx + `coreEntries*2` pairs（非 chunk） |
| Phase C 列切分 | 算 `csPerLoopCols/csColsLoops/csLastLoopCols`（支持大列分段+双缓冲预算） | `csColsLoops` **恒=1**、`csPerLoopCols=cols_`（不切列） |
| 双缓冲预算 | `phaseCFixedOverhead` 含 **2 xBuf slots**（double buffer）+ newPosArr/srcRowArr/origFlatIdxArr[2048] | 无双缓冲 slot 预算；仅 idx+pairs+chunk 辅助 |
| pairs 存储 | 分配 **pairsWorkspace(GM)**（pairsPerCore×coreNum）+ expertCountWorkspace | **仅 expertCount workspace**，pairs 留 UB |
| shortPath | `shortPath_=(coreEntries<=2048 && ep_==1)` UB 免 workspace 往返 | 无 shortPath 概念 |

tiling 与 kernel **双向一致**：A3 tiling 给双缓冲/chunk/pairsWorkspace 留预算 → 支撑 A3 kernel 的 ping-pong+chunk；A5 tiling 只留全载 idx+UB pairs → 对应 A5 kernel 的单扫描+单行串行。**迁移时 tiling 和 kernel 一起被简化了**，不是单边缺失。

### 6.7 适用性判定差异（可能致 Switch ti=29 劣化）

| 判定 | A3 | A5 `IsCountingSortApplicable` |
|------|----|-------------------------------|
| actualExpertNum 上限 | **≤128**（FullLoad 拦截条件） | **≤256**（`CS_MAX_ACTUAL_EXPERT_NUM=256`） |
| n 范围 | shape 判定 | `[CS_MIN_N=256, CS_MAX_N=1M]` |
| dropless 收益门槛 | actualExpertNum*2≤expertNum | 同（`actualExpertNum!=expertNum && *2>expertNum` 排除） |

A5 放宽到 ≤256，使更多场景（含 Switch cols=1024 ti=29）落入 CutOrigin CS。而 A5 CutOrigin Phase C 是串行流式 → Switch 用例本可走更优内置路径却降级到慢 CS，+142% 劣化与此边界放宽相关。

### 6.8 根因总结与后续方向

**劣化根因（按影响排序）**：
1. **【主因】CutOrigin Phase C 丢失双缓冲 ping-pong**：A5 单行串行流式，大列(7168)整行搬运延迟无法隐藏 → ti=1/19/20 +78%~+326%。
2. **【次因】CutOrigin Phase A 丢失 chunk 向量过滤**：A5 纯标量单扫描 coreEntries_ → 计数段变慢（active_range 全通过时更吃亏）。
3. **【边界】适用性放宽 ≤256**：Switch(ti=29) 等场景降级到慢 CS，+142%。
4. **【轻微】FullLoad +20%**：非算法级，拆子类/UB布局/同步 flag 差异。

**后续优化方向（若要追平 A3）**：
- 优先在 A5 CutOrigin 子类 Phase C 重建 A3 的**双缓冲 ping-pong 搬运**（xBuf 双 slot + evMte2/evMte3 交替 + newPos/srcRow 批量预算），tiling 侧补 `csColsLoops/csPerLoopCols` 与双缓冲 UB 预算。
- Phase A 恢复 A3 **chunk 向量过滤**（Cast+Compare+GatherMask），替换纯标量单扫描。
- 复核 A5 `IsCountingSortApplicable` 的 ≤256 放宽是否让 Switch 类场景不该走 CS——必要时收窄。
- 每项改动 A/B 复测 + 全 40 条精度回归（PRECISION PASS 不退化）。

---

## 7. S7 阶段一（Phase C 双缓冲）+ 动态量化关闭：改造后复测（2026-08-06）

> 按 §6.8 方向实施阶段一：A5 CutOrigin 三模板重建 A3 双缓冲 ping-pong（编译宏 `MOE_CS_CUTORIGIN_DOUBLE_BUFFER`，**默认开启**，旧串行路径逐字保留在 `#else`），tiling 侧补双缓冲 UB 预算（`csDbEnable_`，0=超 UB 走串行 fallback）。
> 另经用户决策，**动态量化完全关闭 CountingSort**（见 §7.4）。芯片：Ascend950DT (arch35)，device 3。

### 7.1 阶段一 A/B：双缓冲 vs 串行 vs 原代码（各 3 次，DYN_PERF 越小越好）

| 用例 | 串行基线(S1) | **双缓冲后** | 原代码(S3) | Δ% vs 原 | 结论 |
|------|-------------|-------------|-----------|---------|------|
| ti=1 cutorigin_256（非量化 7168） | 189.612 | 114.246/114.704/114.144 | 106.482 | **+7%** | 大幅收敛（+78%→+7%），残余待 Phase A |
| ti=19 cutorigin_static | 185.748 | 102.282/100.253/100.458 | 98.166 | **+2.5%** | 收敛（+89%→+2.5%） |
| ti=20 cutorigin_dynamic | 760.325 | 700.706/701.012/700.594（仍 CS） | 178.513 | **+292%** | 双缓冲对动态量化无效 → 关闭 CS（§7.2） |
| ti=27 cutorigin_cols32768 | 223.768 | 226.395/225.511/225.543 | 207.12 | **+8.9%** | 中性（大列 MTE 主导，双缓冲影响小） |
| ti=29 switch_cols1024 | 119.448 | **60.443** | 49.428 | +22% | 显著收敛（+142%→+22%），边界残留 |

**关键结论**：
- **双缓冲对非量化/静态量化收益显著**：ti=1 从 +78% → +7%，ti=19 从 +89% → +2.5%，ti=29 从 +142% → +22%。Phase C 串行链被 ping-pong 隐藏，验证 §6.4 主因判断。
- **动态量化（ti=20）双缓冲无效**（700µs，仍 +292%）：per-token absmax 两趟 VF（smooth 预乘写回 + absmax、Div+Cast f32→f16→int8）是**纯计算串行链**，DMA 重叠只是副项 → 双缓冲无法追平。此场景 A3 本就不走 CS（`ShouldEnableCountingSortByShape` 的 `quantValid = quantMode_ != DYNAMIC_QUANT`）。

### 7.2 动态量化关闭 CountingSort（tiling 改动，用户决策）

`op_host/moe_init_routing_v3_tiling_arch35.cpp` `IsCountingSortApplicable()` L2274-2282：
```cpp
// 动态量化场景不支持 CountingSort（含 int8/int4）：
// 1) 与 A3 ShouldEnableCountingSortByShape 对齐（quantValid = quantMode_ != DYNAMIC_QUANT，动态量化走
//    sort + per-token absmax 分段路径）；
// 2) A5 实测 CutOrigin 动态量化（ti=20 9000×7168，smooth 输入）即使重建双缓冲仍 700µs vs 原路径 178µs
//    (+292%)——瓶颈在 per-token absmax 两趟 VF 的串行链，DMA 重叠只是副项，双缓冲无法追平。
//    关闭后 ti=20/37 回落到原代码路径（~178µs）。
if (IsAnyDynamicQuantCase()) {
    return false;
}
```

### 7.3 全量 39 条精度回归（双缓冲开 + 动态量化关）

device 3 逐条跑（瞬时 device retain/sqcq/HDC 故障已重试剔除），**39/39 PRECISION PASS**。动态量化用例全部回落原路径：

| 用例类别 | TilingKey | 结论 |
|---------|-----------|------|
| fullload_dynamic_quant / cutorigin_dynamic_quant | **11020000**（原路径） | 动态 CS 关闭 ✓ |
| fullload_bf16_dynamic_quant / cutorigin_bf16_dynamic_quant | **11020000** | 动态 CS 关闭 ✓ |
| fullload_dynamic_quant_k2 / fullload_dynamic_quant_le64 | **11020000** | 动态 CS 关闭 ✓ |
| cutorigin_static_quant / cutorigin_bf16_static_quant | 510000（CutOrigin CS） | 静态保留 CS ✓ |
| cutorigin_256 / cutorigin_bf16 | 500000 | 非量化 CS ✓ |
| fullload_static_quant / _k2 / _scatter | 410000/411000 | FullLoad CS ✓ |
| 其余 FullLoad / switch / droppad | 400000/400100 等 | CS ✓ |

关键用例 DYN_PERF（关闭动态后）：ti=20 cutorigin_dynamic **175.648**（原 178.5，-0.3% 追平）、cutorigin_256 114.754、cutorigin_static 171.7→100.886、cols32768 224.002、switch_cols1024 60.989、switch_cols2048 68.348、bf16_dynamic 系列 90-105（原路径）。

### 7.4 阶段一总结与取舍

| 维度 | 结论 |
|------|------|
| 双缓冲（Phase C 主因修复） | ✓ 有效：CutOrigin 非量化/静态 +78%/+89% → +7%/+2.5%；switch +142% → +22% |
| 动态量化关闭 CS | ✓ 用户决策：动态量化 8 用例全部回落原路径，ti=20 从 700µs → 175.6µs，追平原代码（-0.3%） |
| 精度 | ✓ 39/39 PASS，无退化 |
| 残余 | 非量化 CutOrigin +7%（计数段标量扫描，待阶段二 Phase A 向量过滤）；cols32768 +8.9%（大列 MTE 主导，中性）；switch +22%（适用性边界，A5 ≤256 vs A3 ≤128） |
| 可回退 | 双缓冲 `#ifdef MOE_CS_CUTORIGIN_DOUBLE_BUFFER`（默认开），旧串行路径 `#else` 逐字保留；动态量化关闭为 tiling 硬编码（保留注释说明） |

**下一步（阶段二）**：Phase A chunk 向量过滤（`MOE_CS_CUTORIGIN_VECTOR_FILTER` 宏，见 §6.3），目标消除 CutOrigin 非量化残余 +7%。

---

## 8. S8 阶段二（Phase A 向量过滤）：改造后 A/B（2026-08-06）

> 按 §6.8 次因方向实施阶段二：A5 CutOrigin `FilterAndCount()` 增加向量过滤路径 `FilterAndCountVector()`
> （编译宏 `MOE_CS_CUTORIGIN_VECTOR_FILTER`），复用 FullLoad 已验证的整批向量过滤实现
> （Cast int32→fp32 → CompareScalar GE/LT → And → GatherMask 压缩 → 仅对压缩后做标量 count），
> 复用已存在的 `MOE_CS_USE_HISTOGRAM` UB 布局（`#if defined(...) || defined(...)` 放宽）。宏关保留纯标量单扫描。
> 顺带修 `entriesAligned_` 潜在 bug：原按 `filterChunkSize_` 对齐，向量过滤扫全量 `coreEntries_`，改为 `Ceil(coreEntries_,64)*64`（与 FullLoad 一致）。tiling 侧补对应向量过滤缓冲预算（fp32+3mask+3×int32，恒计）。芯片：Ascend950DT，device 4。

### 8.1 阶段二 A/B：向量过滤 ON vs OFF（阶段一双缓冲保持开，DYN_PERF 越小越好）

| 用例 | VF OFF | VF ON | Δ (ON vs OFF) | 结论 |
|------|--------|-------|---------------|------|
| ti=1 cutorigin_256（非量化 7168） | 112.846 | 113.4 | **+0.5%** | 中性（VF 略慢，噪声内） |
| ti=19 cutorigin_static | 101.428 | 100.1 | **-1.3%** | 中性（VF 略快） |
| ti=27 cutorigin_cols32768 | 224.411 | 224.2 | **~0%** | 中性 |
| ti=29 switch_cols1024 | 60.585 | 59.9 | **-1.1%** | 中性（VF 略快） |

两侧均 PRECISION PASS。ON 侧全量 39/39 PASS。

**关键结论**：向量过滤在**本测试集净收益近似为零**（±1% 全在 run-to-run 噪声内，无一致方向）。根因与 §6.3 一致——本测试集 `active_expert_range=[0,256]` **全通过**（`coreEntries_` 全合法），GatherMask 压缩**淘汰不掉任何 entry**（filteredInBatch == coreEntries_），向量过滤的 Cast+Compare+GatherMask 额外开销与其向量化收益相互抵消，标量 count 循环长度不变。A3 向量过滤的收益场景是**窄 active_expert_range**（大量淘汰、标量循环显著缩短），本对比集不具备。

### 8.2 阶段二取舍与决策

| 维度 | 结论 |
|------|------|
| 净收益 | ✗ 本测试集近似零（±1% 噪声）；窄 range 场景理论上正收益，但缺对比用例 |
| 精度 | ✓ 双侧 PASS，ON 侧 39/39 PASS |
| UB 成本 | 向量过滤缓冲（fp32+3mask+3×int32×coreEntries）占额外 UB，全通过场景无回报 |
| 决策 | **保留宏 `MOE_CS_CUTORIGIN_VECTOR_FILTER`，默认关**（与 plan §2.2「若为负则保留宏默认关，只在窄 range 场景开」一致；亦与 Histograms 结论 §6.3 注一致）。安装包为 VF-off 状态。 |
| 附带修复 | `entriesAligned_` 从 `filterChunkSize_` 改为 `coreEntries_` 对齐（向量过滤扫全量的正确前提；宏关时不影响纯标量路径） |
| 可回退 | 向量过滤 `#elif defined(MOE_CS_CUTORIGIN_VECTOR_FILTER)`，纯标量 `#else` 逐字保留 |

**阶段二总结**：Phase A 向量过滤在当前全通过对比集**无净收益**，按预案保留宏默认关。CutOrigin 非量化残余 +7%（vs 原 106.5）主要来自 A5 tiling/框架结构性差异（拆子类、UB 布局、每 pair GM↔UB 往返 vs A3 shortPath/workspace 优化），非计数段标量扫描——向量过滤无法消除该残差。至此 CutOrigin 优化两阶段收敛：**主因（双缓冲）已修复并大幅收敛（+78%/+89%/+142% → +7%/+2.5%/+22%），次因（向量过滤）在本场景无收益故默认关但保留可用**。
