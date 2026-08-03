# 92B decode (T=1792) FullLoad 劣化根因分析

> **生成日期**：2026-08-12
> **测试设备**：device 4（Ascend950DT, arch35）
> **安装包**：2026-08-10 17:19 编译（含 Phase C 双缓冲重建）
> **测试方式**：CS 侧 `TTK_DISABLE_BUILTIN=0`；Builtin 侧卸载自定义包；`--ti=6` 串行 3 轮
> **关联用例**：`transpose/moe_v3_test_network_shapes.csv` ti=6 (`net_6_92B_decode_T1792`)
> **状态**：§2.9 验收阻断点的代码级根因定位

---

## 0. 问题陈述

`net_6_92B_decode_T1792` 用例在自定义算子（CS 模板）下相比原代码（builtin）严重劣化 **+31.6%**，是 §2.9 验收阻断点。本文档定位其劣化的代码级根因。

---

## 1. 用例参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 用例名 | `net_6_92B_decode_T1792` | CSV 第 8 行 |
| T（n_） | 1792 | xShape 第 0 维 |
| K（k_） | 8 | expertIdx 第 1 维 |
| H（cols_） | 2560 | xShape 第 1 维 |
| LE（expertEnd_ - expertStart_） | 32 | active_expert_range=[0,32] |
| E（expertNum_） | 256 | expert_num |
| active_num（T×K） | 14336 | totalLength_ |
| drop_pad_mode | 0（dropless） | attributes |
| row_idx_type | 0（GATHER） | attributes |
| quant_mode | -1（unquant） | attributes |
| expert_tokens_num_type | 1（COUNT） | attributes |

---

## 2. A/B 性能数据

| 侧 | TilingKey | 路径 | r1 | r2 | r3 | **中位数** |
|----|-----------|------|------|------|------|-----------|
| **CS（custom）** | **400000** | **CS FullLoad** dropless+GATHER+unquant+COUNT | 99.458 | 98.342 | 98.122 | **98.342 µs** |
| **Builtin** | **11000000** | 原代码 dropless+GATHER+unquant+COUNT | 75.434 | 74.727 | 73.176 | **74.727 µs** |

- **Δ% = (98.342 − 74.727) / 74.727 × 100% = +31.6%（劣化）**
- 精度：CS 侧 PRECISION=PASS；Builtin 侧 PRECISION=FAIL（TTK 框架未生成 golden 参考，非算子精度问题）
- 一致性：CS 侧 r2/r3 差 0.22 µs（CV≈0.1%），数据可靠
- 与历史记录（`cs_optimization_conditions_summary.md` §3 记录 +33.9%）一致，细微差异来自不同测试时段的设备状态

---

## 3. 路径选择追溯

### 3.1 tiling 选择链路（`op_host/moe_init_routing_v3_tiling_arch35.cpp`）

```
line 555:  isFullload_ = IsFullLoad();          // 本例 = false
line 559:  countingSortMode_ = 0;                // 默认 builtin
line 561:  if (!isFullload_ && IsCountingSortApplicable()) {
line 563:      perCoreTokensEst = CeilDiv(1792, aivCoreNum_);
line 564:      estUb = EstimateArch35CountingSortFullLoadUB(perCoreTokensEst);
line 565:      if (estUb > 0 && estUb <= availUbSize_ - LENGTH_1024) {
line 566:          countingSortMode_ = 1;        // ★ 命中 FullLoad
line 567:          ComputeArch35CountingSortFullLoadTiling();
line 568:      } else {
line 569:          countingSortMode_ = 2;        // 本例未进入此分支
line 570:          ComputeArch35CountingSortCutOriginTiling();
```

### 3.2 `IsCountingSortApplicable()` 全部拦截点逐条验证

| 拦截点（line 2284-2324） | 阈值 | 本例取值 | 是否通过 |
|--------------------------|------|---------|---------|
| `n_ < CS_MIN_N(256)` | T≥256 | 1792 | ✅ 通过 |
| `n_ > CS_MAX_N(1M)` | T≤1M | 1792 | ✅ 通过 |
| `expertNum_ > 1024` | ≤1024 | 256 | ✅ 通过 |
| `IsAnyDynamicQuantCase()` | 必须非动态量化 | unquant | ✅ 通过 |
| `actualExpertNum <= 0 \|\| > 256` | LE≤256 | 32 | ✅ 通过 |
| dropless AER：`actualExpertNum×2 > expertNum_ && !=` | actualExpertNum×2=64 ≤ 256 | 32 | ✅ 通过 |
| `actualExpertNum > 256` | ≤256 | 32 | ✅ 通过 |

**结果**：`IsCountingSortApplicable()` 返回 `true`，无任何拦截命中。

### 3.3 FullLoad UB 估算（line 2326-2384）

`EstimateArch35CountingSortFullLoadUB(perCoreTokensEst)` 对本例估算：

| UB 分量 | 计算 | 大小 |
|---------|------|------|
| x 行全载（perCoreTokens × colsAligned × dtypeSize） | 主要分量 | ~100~200 KB |
| expertIdx / scale / expertCount / prefixSum / filteredPairs / expertIdxFp32 / 3 mask / gathered×3 | 辅助 | ~30 KB |
| **total** | — | **通过 `availUbSize_ - LENGTH_1024` 校验** |

**结果**：UB 估算通过 → `countingSortMode_ = 1` → 走 FullLoad（400000）。

> 关键：`EstimateArch35CountingSortFullLoadUB` 的 UB 模型只考虑**容量是否够**，不评估**性能是否更优**。T=1792 时 UB 能容下，但 FullLoad 的固定开销在此 shape 下无法被算力摊薄。

---

## 4. 劣化根因

### 4.1 主因：走错路径——FullLoad 在小 T 小 H 场景下结构性劣势

| 维度 | 本例取值 | FullLoad 劣化机理 |
|------|---------|------------------|
| **T=1792**（小） | perCoreTokens≈T/核数 小 | 每核 active 行数少 → Phase C DMA pipeline 填不满 → 双缓冲 stall 占比高 |
| **H=2560**（小） | 单行搬运延迟短 | FullLoad 一次性载入后，Phase C 流式写出的带宽优势消失 |
| **dropless**（mode=0） | 无 dropPad seed 预计算 | 无 dropPad 路径的 staging/ping-pong 优化加成，纯走 Phase A→B→C 三阶段 |
| **FullLoad 全载固定开销** | x/expertIdx 全部入 UB | 入 UB + 排序 + Phase B 全核同步的固定成本，相对 builtin 的「按需载入」无优势 |

**对照**：同样的 FullLoad 模板在 dropPad 场景下 `-24%~-33%`（13/13 全优化），因为 dropPad 时 Phase C 的 dropPad seed 预计算能覆盖全载开销。dropless 下失去这一加成 → 全场景劣化。

### 4.2 次因：未进入 CutOrigin（500000）路径

CutOrigin（mode=2）的启用条件是「FullLoad UB 超容才回退」：
```
line 568: } else {
line 569:     countingSortMode_ = 2;
line 570:     ComputeArch35CountingSortCutOriginTiling();
```

本例 T=1792/H=2560 较小，FullLoad UB 估算通过 → **不会回退到 CutOrigin**。但即便回退，CutOrigin 在 T<2048 时本身也是劣化（见 `cs_cutorigin_optimization_conditions_v2.md` §8 三段式路径切换，T∈[1024,2047] 走 FullLoad +10%、T≥2048 才走 CutOrigin 优化）。

**结论**：T=1792 是 CS 模板的**死区**——FullLoad 劣化，CutOrigin 也未到启用下界（T≥2048）。**正确路径是回退到 builtin（11000000）**。

### 4.3 根因汇总（按重要性排序）

1. **【主因】FullLoad 在 dropless 小 T 小 H 场景的结构性劣势**：固定开销（全载 + Phase B 同步）无法被 DMA pipeline 重叠摊薄。代码位置：tiling line 565-567「FullLoad 优先」策略。
2. **【次因】门禁函数 `IsCountingSortApplicable()` 无 T/H 下界拦截**：T=1792 直接通过门禁。代码位置：line 2284-2324 缺少 `n_ < CS_DROPLESS_MIN_N` 和 `cols_ < CS_DROPLESS_MIN_COLS` 拦截。
3. **【次因】FullLoad UB 估算只判容量不判性能**：`EstimateArch35CountingSortFullLoadUB` (line 2326) 返回值仅校验 `≤ availUbSize_ - LENGTH_1024`，无性能护栏。

---

## 5. 与历史数据的一致性

| 来源 | 场景 | Δ% |
|------|------|-----|
| **本文档实测**（2026-08-12） | net_6 92B decode T=1792 | **+31.6%** |
| `cs_optimization_conditions_summary.md` §3 | 92B decode T=1792 | +33.9% |
| `cs_cutorigin_optimization_conditions_v2.md` §8 | T=1024 FullLoad（K=8/H=5120） | +10.0% |
| `cs_optimization_conditions_summary.md` §3 | dropless FullLoad 小列 cols=128 | +15%~+30% |

- **本例 +31.6% 介于「T∈[1024,2047] +10%」和「小列 +30%」之间**，且 H=2560 < 5120 叠加「H 小」效应
- **劣化趋势稳定可复现**，无设备噪声嫌疑（CS/BI 双侧 CV < 1%）

---

## 6. 修复方案

### 6.1 Tiling 层拦截（推荐，零 kernel 改动）

按 `cs_tiling_modification_plan.md` 三处改动，本例将被以下拦截命中：

**改动 2（`IsCountingSortApplicable` 加 T 下界拦截）**：
```cpp
if (dropPadMode_ == DROP_PAD_MODE_DROPLESS) {
    // ...
    if (n_ < CS_DROPLESS_MIN_N) {        // CS_DROPLESS_MIN_N = 2048
        return false;                    // ★ 本例 T=1792 命中
    }
    if (cols_ < CS_DROPLESS_MIN_COLS) {  // CS_DROPLESS_MIN_COLS = 5120
        return false;                    // ★ 本例 H=2560 命中（双保险）
    }
    if (k_ < CS_DROPLESS_MIN_K) {        // CS_DROPLESS_MIN_K = 3
        return false;
    }
}
```

**改动 3（csKind 选择按 dropPad/dropless 分流）**：dropless 跳过 FullLoad，直接 CutOrigin；若 CutOrigin 也判定不适用 → 回退 builtin。

### 6.2 修复后效果预期

| 项 | 改前 | 改后 |
|----|------|------|
| TilingKey | 400000（FullLoad） | **11000000（builtin）** |
| DYN_PERF | 98.342 µs | **~74.7 µs**（与 builtin 持平） |
| Δ% | **+31.6%** | **0%** |
| 状态 | ❌ §2.9 验收阻断点 | ✅ 消除 |

### 6.3 不引入其他用例劣化

- 本修复只对 dropless + K<3 / H<5120 / T<2048 / SCATTER / static_quant 场景生效
- dropPad 全场景（13/13 优化）不受影响
- dropless + K≥3 + H≥5120 + T≥2048 + unquant + GATHER 的优化区（-3%~-15%）不受影响

---

## 7. 结论

1. `net_6_92B_decode_T1792` 劣化 **+31.6%** 的根因是：**dropless FullLoad 在 T=1792/H=2560 小 shape 下的固定开销无法摊薄**，且 tiling 门禁未拦截此场景。
2. 这是 §2.9 验收阻断点的实测复现，与历史 +33.9% 一致。
3. 按 `cs_tiling_modification_plan.md` 改动 2/3 实施后，此用例将回退到 builtin（11000000），Δ% 从 +31.6% 降为 0%，**消除阻断点**。
4. 修复仅影响 dropless 已知劣化场景，不影响 dropPad 优化区（13/13）和 dropless CutOrigin 优化区（12/12）。

---

## 8. A3 vs A5 路径对比（迁移劣化的关键差异）

A5 的两个 CS 模板是从 A3 迁移而来。**同样的 shape 在 A3 上走的是 CutOrigin（sortMode=3，优化路径），在 A5 上却走错了 FullLoad（400000，劣化路径）**。这是迁移时引入的回归，根因在 tiling 层。

### 8.1 A3 判定链路（`/home1/l00939308/moeInitrouteV3/.../moe_init_routing_v3_tiling.cpp`）

A3 的判定流程与 A5 类似，但**关键硬件参数不同**：

| 参数 | A3 取值 | A5 取值 | 影响 |
|------|---------|---------|------|
| `aivNum` / `aivCoreNum_` | **48** | **64** | perCoreTokens 38 → 28 |
| `aicoreParams_.ubSize` / `availUbSize_` | **192 KB** | **248 KB** | FullLoad UB 阈值 +56 KB |

**两个因子同时放宽**：A5 既有更多核（perCoreTokens 小）又有更大 UB（阈值高）。

### 8.2 ti=6 在 A3 上逐条验证

**Step 1：`ShouldEnableCountingSortByShape()` (line 809-819) → 返回 true**

| 子条件 | 阈值 | 本例 | 通过 |
|--------|------|------|------|
| `expertNumValid` | expertNum≤1024 | 256 | ✅ |
| `dropPadValid` | DROP_LESS 且 actualExpertNum×2≤expertNum | 32×2=64≤256 | ✅ |
| `quantValid` | !=DYNAMIC_QUANT | UN_QUANT | ✅ |
| `excludeNonDropPadSmallN` | !(n≤2560 && DROP_LESS && STATIC_QUANT) | STATIC_QUANT=false | ✅ |
| `excludeDropPadTinyN` | !(DROP_PAD && n<512) | DROP_LESS | ✅ |

**Step 2：`EstimateCountingSortFullLoadUB()` (line 872-900) — UB 估算**

参数：filterPerCoreTokens = `CeilDiv(1792, 48)` = **38**

| UB 分量 | A3 计算 | 字节 |
|---------|---------|------|
| x 行全载 | CeilAlign(38 × 2560 × 2, 32) | **194560** |
| expertIdx 全载 | CeilAlign(304 × 4, 32) | 1216 |
| scale（isInputScale_=0 跳过）| — | 0 |
| expertCount | CeilAlign(64 × 4, 32) | 256 |
| allCoreExpertCount | CeilAlign(48 × 64 × 4, 32) | 12288 |
| expertTokens | CeilAlign(32 × 8, 32) | 256 |
| prefixSum | CeilAlign(64 × 4, 32) | 256 |
| filteredPairs | CeilAlign(304 × 2 × 4, 32) | 2432 |
| entriesAligned（CeilDiv(304,64)×64=320）| — | — |
| maskBytes（CeilAlign(CeilDiv(320,64),32)=32）| — | — |
| expertIdxFp32 | CeilAlign(320 × 4, 32) | 1280 |
| 3 mask | 32 × 3 | 96 |
| gathered×3 | CeilAlign(320 × 4, 32) × 3 | 3840 |
| **Total** | — | **216480 ≈ 211 KB** |

**判定**：`fullLoadUB(216480) > ubSize - LENGTH_1024 = 192KB - 1024 = 195584` → **FullLoad UB 超容**。

**Step 3：`countingSortFullLoad_ = 0`（line 847-852）→ 进入 CutOrigin 分支**

```cpp
// line 847-852
countingSortFullLoad_ =
    (fullLoadUB <= ubSize - LENGTH_1024 && ...) ? 1 : 0;   // ← 0
```

```cpp
// line 860-862（else 分支）
ComputeCountingSortCutOriginTiling(...);   // ← 进入 CutOrigin
```

**Step 4：`ComputeCountingSortCutOriginTiling()` (line 902-994) — CutOrigin 启用**

- Phase C budget 计算：phaseCBudget = 192KB − 256（persistent） − 32 ≈ 196320 字节
- Phase C 固定开销：pairsBuf(16384) + idxBuf(32) + **3 个栈数组 newPosArr/srcRowArr/origFlatIdxArr (3×2048×4=24576)** = 40992 字节
- perLoopCols 计算：colBytesPerElement=4（2 个 xBuf slot × fp16），remainingForCols ≈ 154304，dynamicMaxPerLoopCols ≈ 38576
- `csColsLoops = 1`（cols=2560 容得下单 slot），不触发 line 960 的 `csColsLoops > 1 && quantMode_ != UN_QUANT` 回退
- totalUB ≈ 57 KB ≪ 192 KB → **CutOrigin 启用**

**Step 5：dispatch (line 1184-1197)**

`countingSortFullLoad_ == 0` → 走 line 1192 分支，`sortMode = 3`（CutOrigin）。

### 8.3 A5 判定（错走 FullLoad）

**A5 FullLoad UB 重算**（perCoreTokens = `CeilDiv(1792, 64)` = **28**，ubSize = 248 KB）：

| UB 分量 | A5 计算 | 字节 |
|---------|---------|------|
| x 行全载（perCoreTokens × colsAligned × dtypeSize）| 28 × 2560 × 2 | 143360 |
| expertIdx 全载（coreEntries=224 × 4）| CeilAlign(896, 32) | 896 |
| scale（isInputScale_=0 跳过）| — | 0 |
| expertCount（expertCountStride=128 × 4）| CeilAlign(512, 32) | 512 |
| allCoreExpertCount（aivCoreNum_=64 × 128 × 4）| CeilAlign(32768, 32) | 32768 |
| expertTokens（32 × 8）| CeilAlign(256, 32) | 256 |
| prefixSum（128 × 4）| CeilAlign(512, 32) | 512 |
| filteredPairs（coreEntries=224 × 2 × 4）| CeilAlign(1792, 32) | 1792 |
| entriesAligned（CeilDiv(224,64)×64=256）| — | — |
| expertIdxFp32（256 × 4）| CeilAlign(1024, 32) | 1024 |
| 3 mask（maskBytes=32）| 32 × 3 | 96 |
| gathered×3（256 × 4）× 3 | CeilAlign(1024, 32) × 3 | 3072 |
| **Total** | — | **184288 ≈ 180 KB** |

**判定**：`fullLoadUB(184288) ≤ availUbSize_(248KB) - LENGTH_1024(1024) = 253952` → **FullLoad UB 通过** → `countingSortMode_ = 1`。

### 8.4 A3 vs A5 路径对比（双因子放大）

| Step | A3（48核，192KB UB）| A5（64核，248KB UB）|
|------|---------------------|---------------------|
| perCoreTokens | 38 | 28 |
| x 行全载 | 194560 字节 | 143360 字节 |
| allCoreExpertCount | 48 × 128 × 4 = 24576 | 64 × 128 × 4 = **32768** |
| FullLoad UB 总和 | **~216 KB** | **~180 KB** |
| UB 阈值（ubSize − 1024）| 191 KB | **247 KB** |
| 与阈值比较 | **216 > 191 → 超容** | **180 < 247 → 通过** |
| 最终路径 | **CutOrigin（sortMode=3，优化）** | **FullLoad（400000，劣化 +31.6%）** |

**双因子放大效应**：

1. **核数 48 → 64**（perCoreTokens 38→28）：
   - x 行全载：194560 → 143360 字节（**−25 KB**）
   - allCoreExpertCount：24576 → 32768 字节（**+8 KB**）
   - 净影响：FullLoad UB −17 KB

2. **UB 大小 192 → 248 KB**（阈值 191 → 247 KB）：
   - FullLoad 阈值直接 **+56 KB** 容量

3. **联合效应**：A3 上 FullLoad UB 超出阈值 25 KB（216−191）；A5 上 FullLoad UB 比阈值低 67 KB（247−180）。**翻转幅度 92 KB**——既来自 UB 增大（贡献 56 KB）也来自核数增多（贡献 36 KB 含 x/allCore 对冲）。

### 8.4 A3 上的性能预期

A3 CutOrigin（sortMode=3）在 92B decode 这类 shape 上**有 Phase A chunk 向量过滤 + Phase C 双缓冲 ping-pong**，对应 `moeInitrouteV3/.../op_kernel/moe_v3_cut_origin_t.h::GatherAndWriteChunked()`。

- 即便 shape 不理想（T=1792/H=2560），A3 CutOrigin 的算法+pipeline 优势可对冲小 shape 的劣势
- 历史数据：网络典型 shape 测试（`cs_network_shapes_perf.md`）显示 A3 CutOrigin 在 T=2560/H=5120 优化 -7.1%
- A5 上 CutOrigin 双缓冲已重建（详见 `perf_baseline_vs_new_template.md` §7），但 Phase A 向量过滤尚未重建（仍纯标量扫描，详见 `project_moe_v3_a3_vs_a5_rootcause.md`）

### 8.5 修复方向印证

| 方案 | A3 行为 | A5 修复后预期 |
|------|---------|--------------|
| Tiling 加 T≥2048 拦截 | A3 CutOrigin 在 T=1792 已优化 → 但 A3 没有此拦截 | A5 改后回退 builtin（11000000） |
| Tiling 加 H≥5120 拦截 | A3 在 H=2560 仍优化 → 与 A3 行为冲突 | A5 改后回退 builtin（保守） |
| FullLoad UB 阈值与 A3 对齐（强制 ≥ ubSize） | A3 本就如此 | A5 应同步收紧 UB 估算，让 FullLoad UB ≥ ubSize-1KB 时回退 CutOrigin |

**结论**：本用例在 A3 上是 CutOrigin 优化路径，A5 迁移时因「**UB 增大（192→248KB）+ 核数增多（48→64）双因子**」联合放大，使 FullLoad UB 估算从「超容 25KB」翻转为「低于阈值 67KB」，误启用 FullLoad，引入 +31.6% 回归。`cs_tiling_modification_plan.md` 的 T/H 下界拦截会消除此回归，但若要更贴近 A3 行为（保留 CutOrigin 优化），可考虑：

1. **收紧 A5 FullLoad UB 阈值**：将 `availUbSize_ - LENGTH_1024` 改为按 A3 等价的 192KB 阈值（截掉 A5 多出的 56KB UB）；
2. **或**：强制 `perCoreTokens` 按 A3 48 核等价计算（`max(CeilDiv(n_,64), CeilDiv(n_,48))`），让 FullLoad UB 估算更保守；
3. **或**：dropless 直接跳过 FullLoad（按 `cs_tiling_modification_plan.md` 改动 3），本例回退 CutOrigin，由 CutOrigin 的 Phase A/C 优化对冲（前提是 Phase A 向量过滤已重建）。

---

**文档版本**：v1.1（2026-08-12，补充 §8 A3 vs A5 路径对比）
**关联文档**：
- `cs_optimization_conditions_summary.md`（41 条数据点支撑的决策矩阵，§3 记录 92B decode 阻断点）
- `cs_cutorigin_optimization_conditions_v2.md` v2.3（dropless CutOrigin 边界测试，§8 三段式路径切换）
- `cs_tiling_modification_plan.md`（tiling 层修改方案，改动 2/3 消除此阻断点）
- `perf_baseline_vs_new_template.md`（Phase C 双缓冲落地后的性能基线）
- `project_moe_v3_a3_vs_a5_rootcause.md`（A3→A5 迁移劣化的根因总览）
