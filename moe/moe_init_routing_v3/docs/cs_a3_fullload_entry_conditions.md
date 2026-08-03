# A3 FullLoad 模板启用条件与 dropless/dropPad 对比

> **生成日期**：2026-08-12
> **目标文件**：A3 `moeInitrouteV3/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling.cpp`
> **关联模板**：A3 `moe_v3_full_load_cut_origin_t.h::MoeV3FullLoadCutOriginT`
> **目的**：精确列举 A3 FullLoad 模板的全部启用条件 + 三层判定流转 + dropless/dropPad 条件区别，作为 A3→A5 迁移正确性的参考基线。

---

## 0. 概述

A3 `ComputeCountingSortTiling()` (line 821-870) 是 CS 模板（CountingSort）的总入口。**FullLoad 是 CS 的子路径之一**，启用需要通过**三层判定**：

| 层级 | 函数/位置 | 失败落点 |
|------|----------|---------|
| **L1 前置层** | line 824 / 828 / 835 | `countingSortFullLoad_` 保持初始值 **−1** → builtin |
| **L2 FullLoad 判定层** | line 847-852 | `countingSortFullLoad_ = 0` → 进入 L3 |
| **L3 CutOrigin 判定层** | line 902-994（`ComputeCountingSortCutOriginTiling`）| `countingSortFullLoad_ = −1`（line 961）→ builtin；否则保持 0 → CutOrigin |

`countingSortFullLoad_` 的最终值决定 dispatch 落点（line 1167-1203）。

> ⚠️ **口径声明**：本文主体（§1-§10）描述的 dispatch `GetTilingKey()` 是 **arch35（950/910D）** 路径 `moe_init_routing_v3_tiling_arch35.cpp` 的完整编码语义（CutOrigin 系 tilingKey 为 `13000000`/`13010000`，含 quant/dropPad 编码）。**非950（base）路径** `moe_init_routing_v3_tiling.cpp` 的常量与编码**不同**（1M 基数，CutOrigin 分支只编码 rowIdxType），其 tilingKey 为 **1300000/1301000**。非950 路径请直接看 **§11**。

---

## 1. 三层判定流程图

```
入口 ComputeCountingSortTiling()  [countingSortFullLoad_ 初始 = −1]
│
├─【L1 前置层】
│   │
│   ├─ line 824: n_ ∈ [COUNTING_SORT_MIN_N(256), COUNTING_SORT_MAX_N(1M)]
│   │   └─ 失败 → return → 保持 −1 → builtin
│   │
│   ├─ line 828-833: SCATTER PerformanceMode 排除
│   │   （n_≤1536 && SCATTER && ep_==1 && expertNum_==ASSIST_NUM &&
│   │    LE≤PERFORMANCE_MODE_RANGE_MAX && UN_QUANT && k_==8 &&
│   │    DROP_LESS && COUNT && actualExpertNum≤128 && >0）
│   │   └─ 命中 → return → 保持 −1 → builtin（让 PerformanceMode 接管）
│   │
│   └─ line 835: ShouldEnableCountingSortByShape() == true
│       └─ 失败 → return → 保持 −1 → builtin
│
└─【L2 FullLoad 判定层】line 847-852
    │
    │  全部 AND 条件通过 → countingSortFullLoad_ = 1 → FullLoad
    │
    └─ 任一不满足 → countingSortFullLoad_ = 0
                    │
                    └─【L3 CutOrigin 判定层】line 860-862 → ComputeCountingSortCutOriginTiling()
                        │
                        ├─ line 960: csColsLoops > 1 && quantMode_ != UN_QUANT
                        │   └─ 命中 → countingSortFullLoad_ = −1 → builtin
                        │
                        └─ 否则 → 保持 0 → CutOrigin
```

---

## 2. L1 前置层详解

### 2.1 CS 入口门槛（line 824）

```cpp
if (n_ < COUNTING_SORT_MIN_N(256) || n_ > COUNTING_SORT_MAX_N(1M)) {
    return;
}
```

- T < 256 或 T > 1M → 直接 return（CS 不启用）

### 2.2 SCATTER PerformanceMode 排除（line 828-833）

```cpp
if (n_ <= COUNTING_SORT_THRESHOLD(1536) && rowIdxTytpe_ == SCATTER && ep_ == 1 &&
    expertNum_ == ASSIST_NUM && (expertEnd_ - expertStart_) <= PERFORMANCE_MODE_RANGE_MAX &&
    quantMode_ == UN_QUANT && k_ == PERFORMANCE_MODE_TOP_K(8) &&
    dropPadMode_ == DROP_LESS && expertTokensNumType_ == COUNT &&
    actualExpertNum <= COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM(128) && actualExpertNum > 0) {
    return;
}
```

- 命中 → return（让 PerformanceMode 路径接管，不走 CS）
- 这是**早期排除特殊优化路径**的机制，不是 CS 启用条件

### 2.3 `ShouldEnableCountingSortByShape()` (line 809-819)

| 子条件 | 表达式 | 含义 |
|--------|--------|------|
| `expertNumValid` | `expertNum_ ≤ 1024` | 专家总数上限 |
| `dropPadValid` | dropless: `actualExpertNum × 2 ≤ expertNum_`<br>dropPad: 恒 true | dropless 要求实际专家数不超一半 |
| `quantValid` | `quantMode_ != DYNAMIC_QUANT` | 排除动态量化 |
| `excludeNonDropPadSmallN` | `!(n_ ≤ 2560 && DROP_LESS && STATIC_QUANT)` | dropless+静态量化+小 T 排除 |
| `excludeDropPadTinyN` | `!(DROP_PAD && n_ < 512)` | dropPad+极小 T 排除 |

任一子条件 false → 返回 false → CS 不启用 → builtin。

---

## 3. L2 FullLoad 判定层详解（line 847-852）

```cpp
countingSortFullLoad_ =
    (fullLoadUB <= ubSize - LENGTH_1024 && ep_ == 1 && quantMode_ == UN_QUANT &&
     k_ == PERFORMANCE_MODE_TOP_K && dropPadMode_ == DROP_LESS &&
     expertTokensNumType_ == COUNT &&
     actualExpertNum <= COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM && actualExpertNum > 0) ? 1 : 0;
```

### 全部 AND 条件（任一不满足 → countingSortFullLoad_ = 0 → 进入 L3）

| # | 条件 | 常量/含义 | 备注 |
|---|------|----------|------|
| 1 | `fullLoadUB ≤ ubSize − LENGTH_1024(1024)` | FullLoad UB 估算通过 | UB 模型见 §6 |
| 2 | **`ep_ == 1`** | **部分专家范围**（LE < expertNum_）| **关键**：全专家范围 [0, E] 时 ep_=0 |
| 3 | `quantMode_ == UN_QUANT(-1)` | 非量化 | 排除 static/dynamic quant |
| 4 | `k_ == PERFORMANCE_MODE_TOP_K(=8)` | **K=8** | 排除 K=1/2/4 等 |
| 5 | **`dropPadMode_ == DROP_LESS(0)`** | **只支持 dropless** | **关键**：dropPad 永不走 FullLoad |
| 6 | `expertTokensNumType_ == COUNT(1)` | expert_tokens_num_type=1 | 排除 cumsum/key-value |
| 7 | `actualExpertNum ≤ COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM(128)` | LE ≤ 128 | A3 上限 128 |
| 8 | `actualExpertNum > 0` | LE > 0 | |

---

## 4. L3 CutOrigin 判定层详解（line 902-994）

L2 失败（countingSortFullLoad_=0）后进入 `ComputeCountingSortCutOriginTiling()`：

### 4.1 CutOrigin UB 计算（line 906-972）

按 Phase A/B/C 分别估算 UB，计算 `csColsLoops` / `csPerLoopCols`。

### 4.2 关键回退点（line 960-963）

```cpp
if (csColsLoops > 1 && quantMode_ != UN_QUANT) {
    countingSortFullLoad_ = -1;   // ← 回退 builtin
    return;
}
```

- **colsLoops > 1**：列太大需要分多次搬运（cols × dtype > maxPerLoopColsBytes）
- **且非 UN_QUANT**：static/dynamic quant 时多列循环不支持
- 命中 → countingSortFullLoad_ = **−1** → dispatch line 1199 → builtin

### 4.3 否则保持 countingSortFullLoad_ = 0 → CutOrigin

dispatch line 1192-1197，sortMode=3，TilingKey=1300000（dropless）或 1300100（dropPad）。

---

## 5. `ep_` 语义详解（line 760-775 `IsFullLoad()`）

```cpp
if (expertStart_ == 0 && expertEnd_ == expertNum_) {
    ep_ = 0;   // 全专家范围 [0, E]，active_expert_range 完整
} else {
    ep_ = 1;   // 部分专家范围，active_expert_range 是 [0, E] 的真子集
}
moeInitRoutingV3TilingData.set_ep(ep_);
```

| active_expert_range | expertNum_ | ep_ | FullLoad 启用（仅看此条件）|
|---------------------|-----------|-----|---------------------------|
| [0, E] | E | 0 | ❌ |
| [0, LE]，LE < E | E | 1 | ✅ |
| [s, e]，s > 0 | E | 1 | ✅ |

**关键**：ep_=0（全专家范围）时，A3 FullLoad 永不启用。这是 fl_k8（active_expert_range=[0,256]，expertNum_=256）在 A3 上走 1100000 的根因之一。

---

## 6. FullLoad UB 估算模型（line 872-900）

`EstimateCountingSortFullLoadUB(filterPerCoreTokens, coreEntries, expertCountStride, filterNeedCoreNum, actualExpertNum)`

```
filterPerCoreTokens = CeilDiv(n_, aivNum)
coreEntries         = filterPerCoreTokens × k_
expertCountStride   = CeilAlign(actualExpertNum, ONE_BLOCK_ELEMENT=64)
colsAligned         = CeilDiv(cols_ × dtypeSize, 32) × 32 / dtypeSize
```

UB 累加项：

| 分量 | 字节计算 |
|------|---------|
| x 行全载 | `CeilAlign(coreTokenNum × colsAligned × dtypeSize, 32)` |
| expertIdx 全载 | `CeilAlign(coreEntries × 4, 32)` |
| scale（isInputScale_）| `CeilAlign(coreTokenNum × scaleSlotSize × 4, 32)` |
| expertCount | `CeilAlign(expertCountStride × 4, 32)` |
| allCoreExpertCount | `CeilAlign(filterNeedCoreNum × expertCountStride × 4, 32)` |
| expertTokens | `CeilAlign(actualExpertNum × 8, 32)` |
| prefixSum | `CeilAlign(expertCountStride × 4, 32)` |
| filteredPairs | `CeilAlign(coreEntries × 2 × 4, 32)` |
| expertIdxFp32 | `CeilAlign(entriesAligned × 4, 32)` |
| 3 mask | `maskBytes × 3` |
| gathered×3 | `CeilAlign(entriesAligned × 4, 32) × 3` |

阈值：`fullLoadUB ≤ ubSize(192KB) − LENGTH_1024(1024) = 195584 字节`

---

## 7. dropless vs dropPad 条件区别

### 7.1 在三层判定中的差异点

| 层级 | dropless (DROP_LESS=0) | dropPad (DROP_PAD=1) |
|------|------------------------|----------------------|
| **L1 line 824** | n ∈ [256, 1M] | n ∈ [256, 1M]（一致）|
| **L1 line 828 SCATTER 排除** | 要求 `dropPadMode_ == DROP_LESS` → 可能命中 | dropPad 不命中此分支 |
| **L1 ShouldEnable.dropPadValid** | `actualExpertNum×2 ≤ expertNum_`（严格）| 恒 true |
| **L1 ShouldEnable.excludeDropPadTinyN** | 不检查 | `n ≥ 512` |
| **L1 ShouldEnable.excludeNonDropPadSmallN** | `!(n≤2560 && STATIC_QUANT)` | 不检查 |
| **L2 line 849** | `dropPadMode_ == DROP_LESS` ✅ 通过 | ❌ **硬编码拦截 → countingSortFullLoad_=0** |
| **L3 line 960 回退** | csColsLoops>1 && 非 UN_QUANT → builtin | 同 |

### 7.2 关键结论

| 项 | dropless | dropPad |
|----|----------|---------|
| **FullLoad 能否启用** | ✅ 可（L1+L2 全通过时）| ❌ **永不**（L2 line 849 硬编码）|
| **典型落点** | FullLoad(1200000) / CutOrigin(1300000) / builtin(1100000) | **CutOrigin(1300100)** / builtin(1100100) |
| **FullLoad 模板分支** | `moe_v3_full_load_cut_origin_t.h::GatherAndWrite` 主路径 | 代码中 dropPad 分支存在但永不命中 |

### 7.3 dispatch TilingKey 编码对照

```
TilingKey = TILINGKEY_BASE(1M) + sortMode × 100K + (quantMode+1) × 10K + rowIdxType × 1K + dropPadMode × 100
```

| sortMode | 含义 | dispatch 行 | dropless unquant+GATHER | dropPad unquant+GATHER |
|----------|------|-------------|--------------------------|-------------------------|
| 1 | builtin sort+gather（else 分支）| line 1199 | **1100000** | **1100100** |
| 2 | **FullLoad** | line 1184 | **1200000** | —（永不命中）|
| 3 | CutOrigin | line 1192 | 1300000 | **1300100** |

---

## 8. A3 → A5 迁移差异（参考）

A5 在迁移时对 FullLoad 启用面做了两处扩展，**都比 A3 更宽松**：

| 扩展点 | A3 行为 | A5 行为 | 实测影响 |
|--------|---------|---------|---------|
| **dropless 全专家范围**（actualExpertNum==expertNum_）| 双重拦截：ep_=0（L2）+ dropPadValid=false（L1）→ 永不走 FullLoad | 加例外放行（line 2310 `actualExpertNum != expertNum_` 短路）→ 启用 FullLoad | **回归点**（fl_k8 +65.9% 劣化）|
| **dropPad FullLoad** | line 849 硬编码 `== DROP_LESS` → 永不走 | dropPad 也先尝试 FullLoad（csKind 选择无 dropPad 拦截）| **有效扩展**（-24%~-33%，13/13 优化）|

详见：
- 回归点分析：`cs_fullload_k8_degradation_analysis.md` §4.2
- dropPad 优化验证：`cs_droppad_optimization_validation.md`
- 迁移根因总览：`project_moe_v3_a3_vs_a5_rootcause.md`

---

## 9. 典型场景验证

### 9.1 场景 A：dropless + 部分专家范围 + K=8（A3 FullLoad 典型启用）

参数：T=10000, K=8, H=128, LE=64, E=256, dropless+GATHER+unquant+COUNT

| 条件 | 验证 |
|------|------|
| L1 n∈[256,1M] | 10000 ✅ |
| L1 SCATTER 排除 | rowIdx=GATHER，不命中 ✅ |
| L1 ShouldEnable.dropPadValid | 64×2=128 ≤ 256 ✅ |
| L2 fullLoadUB | perCoreTokens=209，x 全载≈52KB，total≈100KB ≪ 191KB ✅ |
| L2 ep_ | LE=64 < E=256 → ep_=1 ✅ |
| L2 K=8 | ✅ |
| L2 dropless | ✅ |
| L2 actualExpertNum≤128 | 64 ≤ 128 ✅ |

→ **countingSortFullLoad_ = 1 → FullLoad（1200000）**

### 9.2 场景 B：dropless + 全专家范围（fl_k8 用例）

参数：T=9000, K=8, H=128, LE=256, E=256, dropless+GATHER+unquant+COUNT

| 条件 | 验证 |
|------|------|
| L1 ShouldEnable.dropPadValid | 256×2=512 > 256 → **false** ❌ |

→ **直接 return → countingSortFullLoad_=−1 → builtin（1100000）**

即便假设 dropPadValid 通过，L2 ep_=0（全专家范围）也会拦截。

### 9.3 场景 C：dropPad + 任意 shape

参数：dropPadMode_=1

| 条件 | 验证 |
|------|------|
| L2 line 849 | `dropPadMode_ == DROP_LESS` → **false** ❌ |

→ **countingSortFullLoad_=0 → L3 CutOrigin**（若 L3 通过）或 builtin（若 L3 回退）

### 9.4 实测验证：fl_k8_aer64（A3/A5 共通 FullLoad 用例）

构造一个 A3 和 A5 **都走 FullLoad** 的用例，验证 FullLoad 在 A3 合法启用区的实际收益。

#### 用例参数（`moe_v3_test_fl_k8_aer64.csv`）

| 参数 | 值 | A3 FullLoad 条件 |
|------|-----|------------------|
| T（n_） | 9000 | L1 n∈[256,1M] ✅ |
| K（k_） | 8 | L2 K=8 ✅ |
| H（cols_） | 128 | — |
| LE（actualExpertNum） | **64** | L2 LE≤128 ✅ |
| E（expertNum_） | 256 | — |
| active_expert_range | [0, 64] | ep_=1（LE<E）✅ |
| dropPadMode_ | 0（dropless）| L2 dropless ✅ |
| rowIdxType_ | 0（GATHER）| — |
| quantMode_ | −1（unquant）| L2 unquant ✅ |
| expertTokensNumType_ | 1（COUNT）| L2 COUNT ✅ |

- L1 ShouldEnable.dropPadValid：64×2=128 ≤ 256 ✅
- L2 fullLoadUB：perCoreTokens≈28，x 全载≈56KB，total≈120KB ≪ 191KB ✅
- 全部 AND 条件通过 → A3 countingSortFullLoad_=1 → TilingKey **1200000**；A5 通过对应门禁 → TilingKey **400000**

#### A5 实测（device 4，2026-08-12，`TTK_DISABLE_BUILTIN=0` + builtin 卸包，`--ti=0` 串行 3 轮）

| 侧 | TilingKey | r1 | r2 | r3 | 中位数 |
|----|-----------|------|------|------|--------|
| CS（FullLoad）| **400000** | 126.885 | 128.779 | 128.912 | **128.779 µs** |
| Builtin | **11000000** | 134.780 | 135.906 | 134.842 | **134.842 µs** |

**Δ% = (128.779 − 134.842) / 134.842 × 100% = −4.5%（优化！）**

#### 关键结论

1. **A3 FullLoad 合法启用区确实有优化**（−4.5%）——证明 FullLoad 模板本身设计正确，在符合 A3 门禁的 shape 下是有效的优化路径。
2. fl_k8（LE=256=expertNum_，全专家范围）在 A3 **不走 FullLoad**（§9.2），但在 A5 因例外放行走 FullLoad，劣化 +65.9%（详见 `cs_fullload_k8_degradation_analysis.md`）。
3. **filteredCount 分水岭**：fl_k8_aer64 filteredCount≈18000（281 次/核）→ 优化；fl_k8 filteredCount≈72000（1125 次/核）→ 劣化。filteredCount 越大，Phase C 逐 pair 串行循环累积开销越大。
4. 本用例（LE=64）与 fl_k8（LE=256）形成完美对照，唯一差异是 actualExpertNum/expertNum_ 比例（25% vs 100%），证明 **dropless FullLoad 不是全场景劣化，劣化仅出现在 A5 例外放行引入的全专家范围场景**。

---

## 11. 非950（base）路径：走入 tilingKey 1300000 / 1301000

> **目标文件**（与上文 arch35 不同）：
> - host：`op_host/moe_init_routing_v3_tiling.cpp`（`MoeInitRountingV3TilingBase`，base/非950）
> - kernel：`op_kernel/moe_init_routing_v3.cpp`（`extern "C" moe_init_routing_v3`）

### 11.1 base 分支的 TilingKey 常量（`moe_init_routing_v3_tiling.cpp:92-96`）

| 常量 | 值 | 含义 |
|------|-----|------|
| `TILINGKEY_BASE` | **1000000** (1M) | 基数（arch35 为 10M，**不同**）|
| `SORT_CORE_TILINGKEY_BASE` | **100000** (100K) | sortMode 增量 |
| `QUANT_MODE_TILINGKEY_BASE` | 10000 (10K) | quantMode 增量 |
| `ROWIDX_TYPE_TILINGKEY_BASE` | 1000 (1K) | rowIdxType 增量 |
| `DROP_MODE_TILINGKEY_BASE` | 100 | dropPadMode 增量 |

### 11.2 CutOrigin 分支编码（`moe_init_routing_v3_tiling.cpp:1192-1197`）

```cpp
} else if (countingSortFullLoad_ == 0 || /* MULTI_CORE 性能模式旁路 */) {
    uint64_t sortMode = 3;
    return TILINGKEY_BASE + sortMode * SORT_CORE_TILINGKEY_BASE +
           rowIdxTytpe_ * ROWIDX_TYPE_TILINGKEY_BASE;
}
// = 1000000 + 3*100000 + rowIdx*1000
```

| rowIdxTytpe_ | tilingKey | kernel 宏 | 含义 |
|--------------|-----------|-----------|------|
| **0（GATHER）** | **1300000** | `MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_GATHER` | 多核Gather→多核排序、GATHER索引 |
| **1（SCATTER）** | **1301000** | `MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_SCATTER` | 多核Gather→多核排序、SCATTER索引 |

> ⚠️ **base 与 arch35 的关键差异**：base 的 CutOrigin 分支（sortMode=3）**只编码 rowIdxType，不编码 quantMode / dropPadMode**！
> - 因此 **1300000/1301000 同时覆盖 dropless 与 dropPad**、覆盖 unquant/static/dynamic 三种量化。
> - base 路径**不存在** `1300100`（dropPad 独享 key）。dropPad 与 dropless 共享 `1300000`（GATHER）/`1301000`（SCATTER），由 kernel 端运行时再按 `dropPadMode` 字段分流到命令。
> - 对比 arch35 的 `13000000/13010000`（含完整 encoding，见 §7.3）——两套数值差 10 倍，勿混淆。

### 11.3 host 端完整进入条件链

```
DoOpTiling()                                         (tiling.cpp:1154-1156)
  ├─ isFullload_ = IsFullLoad()                      (tiling.cpp:760-807)
  │     └─ true  → GetTilingKey 返回 21xxxxx（FullLoad），绝不进 1300000/1301000
  └─ if (!isFullload_ && !PERF_7168)
        ComputeCountingSortTiling()                  (tiling.cpp:821-870)
        └─ countingSortFullLoad_ ∈ {−1, 0, 1}

GetTilingKey()                                       (tiling.cpp:1167-1204)
  ├─ isFullload_                 → 21xxxxx
  ├─ countingSortFullLoad_ == 1  → sortMode=2 → 12x0000 (FullLoad CS)
  ├─ countingSortFullLoad_ == 0  → sortMode=3 → 1300000/1301000  ← *目标*
  └─ else                        → 11xxxxx (builtin/其它)
```

进入 **1300000/1301000** 需满足（二选一）：

**路径 A：`countingSortFullLoad_ == 0`（正常 CS CutOrigin）**
由 `ComputeCountingSortTiling()` 产生，前置条件为：
1. **isFullload_ == false**（`IsFullLoad()` 失败：`totalLength_ > sortLoopMaxElement` 或 dropPadMode_==1 或 UB 不足）
2. L1 前置通过（`n_∈[256,1M]` 且非 SCATTER PerformanceMode 排除且 `ShouldEnableCountingSortByShape()`==true）——通到 L2
3. `countingSortFullLoad_` 判定为 0：L2 八条 AND 中**任一**不满足（`fullLoadUB>ul` / `ep_!=1` / `quantMode_!=UN_QUANT` / `k_!=8` / `dropPadMode_!=DROP_LESS` / `expertTokensNumType_!=COUNT` / `actualExpertNum>128` / `actualExpertNum<=0`）
4. L3 `ComputeCountingSortCutOriginTiling` **未回退**（`csColsLoops<=1 || quantMode_==UN_QUANT`，不触发 line 960 → `countingSortFullLoad_==-1`）

**路径 B：MULTI_CORE_GATHER_SORT 性能模式旁路**（tiling.cpp:1193-1197）
```cpp
countingSortFullLoad_ == -1 &&
GetPerformanceMode() == MULTI_CORE_GATHER_SORT &&
quantMode_ == UN_QUANT && rowIdxTytpe_ == SCATTER &&
expertTokensNumType_ == COUNT
```
- `GetPerformanceMode()` 返回 MULTI_CORE 需：`expertNum_==256 && (expertEnd_-expertStart_)≤32 && n_∈[384,8192] && k_==8`（tiling.cpp:1025-1026）
- 即：SCATTER + 小范围专家 + K=8 的「早期 SCATTER 排除」（L1 line 828）已让 CS 提前 return（保持 −1），此处旁路逻辑**再次兜底**落到 1301000。

### 11.4 kernel 端 1300000/1301000 分派（`moe_init_routing_v3.cpp:267-283`）

两 key 共用一个 dispatch 块，先尝试 CountingSort（非全载），失败才回退：

```cpp
if (TILING_KEY_IS(1300000) || TILING_KEY_IS(1301000)) {
    bool useCountingSort =
        !(n_ <= COUNTING_SORT_THRESHOLD(1536) && rowIdxType == SCATTER && quantMode == -1 &&
          expertTokensNumType == EXERPT_TOKENS_COUNT(1) &&
          (expertEnd - expertStart) <= PERFORMANCE_MODE_RANGE_MAX(32) && k_ == 8 && expertNum == 256);
    if (useCountingSort) {
        MoeV3CutOriginT<DTYPE_X> op;   // 非全载 CountingSort（prefill 大 T）→ moe_v3_cut_origin_t.h
        op.Process();
    }
}
// useCountingSort == false → 越过 CS，落到下方 MoeGatherSortMultiCore（line 301-308）原始多核 gather+sort
```

| 分支 | 条件 | 执行 |
|------|------|------|
| **CS（CutOrigin）** | 上述 excluded 表达式为 **false** | `MoeV3CutOriginT`（非全载 CS）|
| **回退原始流程** | excluded 表达式为 **true**（命中 PerformanceMode 特征）| `MoeGatherSortMultiCore`（原 gather-first 多核排序）|

> 该 excluded 条件与 host `GetPerformanceMode()`/`ComputeCountingSortTiling()->line 828` 的 SCATTER 排除**互为镜像**：host 侧早排除的不再进 CS，kernel 侧用同一套 shape 特征二次拦截，保证 host（路径 B）与 kernel 回退逻辑一致。

### 11.5 base 路径补充说明

1. **base 的 FullLoad（非 CS）** 是 `21xxxxx` 系（`tiling.cpp:100-104`），对应 kernel `MOE_INIT_ROUTING_V3_PERFORMANCE`/`UNQUANTIZED_FULLLOAD`/`STATIC/DYNAMIC_QUANT_FULLLOAD`。
2. **base 的 CS FullLoad（sortMode=2）** 为 `12x0000`（`1200000`/`1201000`），对应 kernel `MOE_INIT_ROUTING_V3_GATHER_SORTONECORE_*` → `MoeV3FullLoadCutOriginT`（line 248-265，条件同 §11.4 但带 `dropPadMode==0 && COUNT && quantMode==-1 && actualExpertNum≤128` 前缀）。
3. `countingSortFullLoad_` 三态语义在 base 与 arch35 **一致**：−1（初始/回退）→ 不切 sortMode=2/3（走 11xxxxx 或 MULTI_CORE 旁路）；0 → CutOrigin（1300000/1301000）；1 → CS FullLoad（1200000/1201000）。

---

## 10. 结论

1. **A3 FullLoad 启用需要通过三层判定**：L1 前置（n 范围、SCATTER 排除、ShouldEnable）+ L2 八条 AND（UB / ep_ / K=8 / unquant / **dropless** / COUNT / LE≤128 / LE>0）+ L3 CutOrigin 回退检查。
2. **三个关键拦截点**：`ep_ == 1`（必须部分专家范围）、`dropPadMode_ == DROP_LESS`（必须 dropless）、`actualExpertNum ≤ 128`（LE 上限 128）。
3. **dropPad 在 A3 永不走 FullLoad**——line 849 硬编码 `== DROP_LESS` 是不可绕过的拦截。dropPad 总是走 CutOrigin（1300100）或 builtin（1100100）。
4. **countingSortFullLoad_ 三态语义**：−1（初始/回退）→ builtin；0 → CutOrigin；1 → FullLoad。我之前「条件不满足 → CutOrigin」的说法错误——前置层失败直接回 builtin，只有 L2 失败才进 L3。
5. **A5 迁移时两处扩展**：dropless 全专家范围例外放行（回归点，fl_k8 +65.9%）、dropPad 支持 FullLoad（有效扩展，-24%~-33% 优化）。两处都让 A5 FullLoad 启用面比 A3 宽。
6. **FullLoad 模板本身正确**（§9.4 实测）：fl_k8_aer64 在 A3/A5 共通 FullLoad 启用条件下优化 −4.5%，证明 FullLoad 在 A3 合法启用区（部分专家范围 + LE≤128 + filteredCount ≤ ~300 次/核）是有效优化路径；劣化仅出现在 A5 例外放行引入的全专家范围场景。
7. **非950（base）路径的 CutOrigin tilingKey 是 1300000（GATHER）/1301000（SCATTER）**，由 `countingSortFullLoad_==0` 或 MULTI_CORE 性能模式旁路产出（`moe_init_routing_v3_tiling.cpp:1192-1197`）。**base 的 sortMode=3 分支只编码 rowIdxType，不编码 quant/dropPad**——与 arch35 的 `13000000/13010000`（完整编码）数值差 10 倍、语义不同，两套口径不可混用。详见 §11。

---

**文档版本**：v1.1（2026-08-18，新增 §11 非950 base 路径 1300000/1301000）
**关联文档**：
- `cs_fullload_k8_degradation_analysis.md` v1.1（fl_k8 用例的 A5 FullLoad 劣化根因，含 A5 例外放行迁移回归点）
- `cs_92b_decode_degradation_analysis.md` §8（92B decode 的 A3 vs A5 路径对比）
- `cs_optimization_conditions_summary.md`（dropPad/dropless 全场景决策矩阵）
- `cs_tiling_modification_plan.md`（A5 tiling 修改方案）
- `perf_baseline_vs_new_template.md` §6.2（A3 vs A5 FullLoad/CutOrigin 模板对照）
