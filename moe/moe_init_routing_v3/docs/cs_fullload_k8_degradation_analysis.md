# A5 FullLoad 在 K=8 + dropless 场景下的劣化根因分析

> **生成日期**：2026-08-12
> **测试设备**：device 4（Ascend950DT, arch35）
> **安装包**：2026-08-10 17:19 编译（含 Phase C 双缓冲重建）
> **测试方式**：CS 侧 `TTK_DISABLE_BUILTIN=0`；Builtin 侧卸载自定义包；`--ti=0` 串行 3 轮
> **关联用例**：`transpose/moe_v3_test_fl_k8.csv` ti=0 (`fl_k8_dropless_unquant`)
> **状态**：A5 FullLoad 在 K=8 大 active_num 场景的代码级根因定位

---

## 0. 问题陈述

A5 FullLoad 模板（TilingKey 400000）在 K=1 已知劣化 +23.6%（详见 `perf_baseline_vs_new_template.md` §6.5）。**K=8 时劣化是否更严重？** 通过构造 K=8 + H=128 + dropless + 全专家范围（actualExpertNum=256=expertNum_）用例实测。

> **修订记录**：初版（v1.0）误判 A3/A5 路径同构，A3 实测验证后发现走 TilingKey 1100000（builtin sort+gather），与 A5 走的 400000（CS FullLoad）**路径不同构**。v1.1 重新定位迁移回归点为 A5 `IsCountingSortApplicable()` line 2309-2312 的例外放行。

---

## 1. 用例构造与路径分歧（A3 vs A5）

### 1.1 用例参数（`fl_k8_dropless_unquant`）

| 参数 | 值 |
|------|-----|
| T（n_） | 9000 |
| K（k_） | **8** |
| H（cols_） | 128 |
| LE（actualExpertNum=expertEnd_−expertStart_）| **256**（active_expert_range=[0,256]）|
| E（expertNum_） | 256 |
| active_num（T×K）| **72000** |
| drop_pad_mode | 0（dropless）|
| row_idx_type | 0（GATHER）|
| quant_mode | -1（unquant）|
| expert_tokens_num_type | 1（COUNT）|
| dtype | fp16 |

> **关键**：`actualExpertNum(256) == expertNum_(256)`——全专家范围场景。这一条件在 A3/A5 上的判定差异决定了路径分歧（详见 §1.2）。

### 1.2 A3 vs A5 路径分歧（实测验证）

**A3 实测 TilingKey = 1100000**（builtin sort+gather 旧路径，sortMode_=1，dispatch line 1199 else 分支）。
**A5 实测 TilingKey = 400000**（CS FullLoad dropless unquant+GATHER+COUNT）。

**两边路径不同构**——A3 上 CS 模板根本未启用，A5 上启用了 FullLoad。

### 1.3 分歧根因：A5 迁移时引入的「例外放行」

A3 和 A5 的 CS 门禁函数对 `actualExpertNum == expertNum_` 的处理不同：

**A3 `ShouldEnableCountingSortByShape()` (line 813-814)**——严格不等式：
```cpp
bool dropPadValid =
    (dropPadMode_ == DROP_PAD) ||
    (dropPadMode_ == DROP_LESS && actualExpertNum * NUM_TWO <= expertNum_);
//                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                                  fl_k8: 256 × 2 = 512 > 256 → false
```

**A5 `IsCountingSortApplicable()` (line 2309-2312)**——加了 `actualExpertNum != expertNum_` 例外：
```cpp
if (dropPadMode_ == DROP_PAD_MODE_DROPLESS) {
    if (actualExpertNum != expertNum_ && actualExpertNum * NUM_TWO > expertNum_) {
    //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //  fl_k8: 256 == 256 → 短路为 false → 整个 if 不执行 → 不拦截
        return false;
    }
}
```

### 1.4 代入 fl_k8 验证

| 平台 | actualExpertNum | expertNum_ | 判定表达式 | 结果 | CS 启用 |
|------|----------------|------------|-----------|------|---------|
| **A3** | 256 | 256 | `256×2 ≤ 256` → 512 ≤ 256 | **false** | ❌ |
| **A5** | 256 | 256 | `256 != 256 && ...` → false && ... | 不拦截 | ✅ → FullLoad 400000 |

### 1.5 后续 dispatch 链路

**A3**（CS 未启用）：
```
ShouldEnableCountingSortByShape() = false
  → ComputeCountingSortTiling() line 835-837 直接 return
  → countingSortFullLoad_ 保持初始值 -1
  → GetTilingKey() dispatch line 1199 else 分支
  → sortMode_ = 1（旧 sort+gather）
  → TilingKey = 1000000 + 1×100000 + 0×10000 + 0×1000 + 0×100 = 1100000
```

**A5**（CS 启用）：
```
IsCountingSortApplicable() = true（例外放行）
  → EstimateArch35CountingSortFullLoadUB 通过
  → countingSortMode_ = 1
  → TilingKey = 400000（CS FullLoad）
```

### 1.6 模板函数对照（仅 A5 FullLoad 内部分析，不作 A3/A5 等价对比）

由于 A3 上 fl_k8 不走 FullLoad，§4 的代码级根因只针对 A5 FullLoad 内部结构分析。A5 FullLoad（`moe_v3_counting_sort_full_load_*.h`）内部各阶段仍可对照 A3 FullLoad **模板文件本身**（不是该用例的实际路径）做算法同构性确认：

| 阶段 | A3 模板（`moe_v3_full_load_cut_origin_t.h`）| A5 模板（`moe_v3_counting_sort_full_load_*.h`）| 同构 |
|------|---------------------------------------------|-----------------------------------------------|------|
| Phase A 全载 x | `LoadXBackground` line 300 | `LoadXBackground` line 332（base）| ✅ |
| Phase A 向量过滤 | `FilterAndCountVector` line 327 | `FilterAndCountVector` line 363（base）| ✅ |
| Phase C 逐 pair 写 | `GatherAndWrite` line 488 | `GatherAndWrite` line 132 + `GatherOneRow` line 98 | ✅ |

**但这是模板文件层面的同构，不代表 fl_k8 在 A3 上走了这条路径**——A3 上 fl_k8 实际走 sortMode_=1（旧 sort+gather），与 A5 FullLoad 无对比意义。

---

## 2. A/B 性能数据

| 侧 | TilingKey | r1 | r2 | r3 | **中位数** |
|----|-----------|------|------|------|-----------|
| **CS（FullLoad）** | **400000** | 冷启动 | 377.006 | 376.474 | **376.474 µs** |
| **Builtin** | **11000000** | 227.040 | 231.865 | 226.639 | **227.040 µs** |

- **Δ% = (376.474 − 227.040) / 227.040 × 100% = +65.9%（劣化）**
- 精度：CS 侧 PRECISION=PASS；Builtin 侧 PRECISION=FAIL（TTK 框架无 golden 参考，非算子精度问题）
- 数据可靠性：CS 侧 r2/r3 差 0.5 µs（CV≈0.07%），双侧数据可靠

---

## 3. K 放大效应对比

| 用例 | K | T | H | active_num | 每核循环 | A5 Δ% |
|------|---|---|---|-----------|---------|-------|
| `fullload_256`（历史）| 1 | 10000 | 128 | 10000 | ~157/核 | **+23.6%** |
| `fl_k8`（本次）| **8** | 9000 | 128 | **72000** | **~1125/核** | **+65.9%** |

**关键观察**：
- K 翻 8 倍 → active_num 翻 7.2 倍 → 每核循环次数翻 ~7.2 倍
- Δ% 从 +23.6% → +65.9%，**劣化幅度近似线性放大**
- **K 是 FullLoad Phase C 循环次数的主驱动**，也是劣化幅度的主驱动

---

## 4. 劣化根因

### 4.1 主因：Phase C 逐 pair 串行循环 + 极小 DataCopyPad 的累积开销

A5 `GatherAndWrite`（line 132-171）+ `GatherOneRow`（line 98-129）单 pair 处理：

```cpp
// 每个 pair 执行：
GetValue×2 (filteredPairs) +
GetValue + SetValue (expertCount) +
SetValue (idxBuf) +
SetWaitFlag<S_MTE3> + DataCopyPad(4 字节 idx → rowIdxGm) +   // ★ 极小搬运
SetWaitFlag<MTE3_S> +
DataCopyPad(cols×dtype = 256 字节 x → expandedXGm)
```

**问题**：
1. **K=8 + active_num=72000** → 每核循环 ~1125 次（A5 64 核）
2. 每次循环有 **2 次 SetWaitFlag + 2 次 DataCopyPad**
3. 其中 **idx DataCopyPad 只搬 4 字节**——纯 DMA setup 开销，无有效数据搬运
4. 1125 次 × 极小 DataCopyPad 固定开销 → 累积成主成本

### 4.2 迁移回归点：A5「例外放行」让 CS 启用门槛低于 A3

**关键发现**：fl_k8 在 A3 上根本不走 CS（走 1100000 builtin sort+gather），A5 上却启用了 FullLoad。根因是 A5 门禁函数加了一个例外放行：

| 平台 | dropless AER 条件 | fl_k8(256, 256) 判定 |
|------|------------------|----------------------|
| A3 line 814 | `actualExpertNum × 2 ≤ expertNum_` | 512 > 256 → **拦截** |
| A5 line 2309-2312 | `actualExpertNum != expertNum_ && actualExpertNum × 2 > expertNum_` | 256==256 短路 → **例外放行** |

A5 多出的 `actualExpertNum != expertNum_` 短路条件让「全专家范围」（actualExpertNum==expertNum_）场景绕过了 A3 的严格 dropless AER 上限检查。**这是迁移时引入的回归点**，使 A5 在 A3 本不走 CS 的场景下错误启用了 FullLoad。

注释（A5 line 2308）写道：
> 例外：actualExpertNum == expertNum（全专家范围，对齐文档 §3.1 le == expertNum_）时也走计数排序

这个例外的初衷可能是想覆盖某些边界用例，但**没有同步 A3 的实际行为**，导致 A3 上走 builtin 的场景在 A5 上误启用 FullLoad。

### 4.3 与 builtin 11000000 的性能对比解释

A5 上 fl_k8：
- CS FullLoad（400000）= 376.474 µs（逐 pair 串行，1125 次/核）
- Builtin 11000000 = 227.040 µs（A5 builtin sort+gather）

A5 builtin 比 CS FullLoad 快 149 µs（+65.9%）。可能原因：

| 假设 | 评估 |
|------|------|
| A5 builtin 用批量 DataCopy（一次提交多行）| **最可能**——避免了 1125 次 SetWaitFlag + 极小 DataCopyPad 的累积开销 |
| A5 builtin 用 sort+gather 算法（非计数排序）| 可能——sort 后批量 gather，避免逐 pair 串行 |
| CS FullLoad Phase A 的纯标量过滤（FilterAndCount）也是瓶颈 | 次要——但 Phase A 时间相对 Phase C 较小 |

A3 上 fl_k8 走 1100000（与 A5 builtin 11000000 算法同源——都是旧 sort+gather），所以 A3 没有引入劣化。

### 4.4 根因汇总

1. **【主因】Phase C 逐 pair 串行循环无法摊薄 DMA setup 开销**：K=8 + active_num=72000 下，每核 1125 次「SetWaitFlag + 极小 DataCopyPad（4 字节 idx）」累积成主成本。代码位置：`moe_v3_counting_sort_full_load_unquantized.h::GatherAndWrite` line 132-171 + `GatherOneRow` line 98-129。
2. **【迁移回归点】A5 line 2309-2312 的例外放行**：`actualExpertNum != expertNum_` 让全专家范围场景绕过 A3 的严格 AER 检查，使 A5 在 A3 本不走 CS 的场景下错误启用 FullLoad。
3. **【参照系】A5 builtin（11000000）在大 active_num 下更高效**：A5 builtin 可能用批量 sort+gather，比 CS FullLoad 的逐 pair 串行快 65.9%。
4. **【放大因子】K 越大劣化越严重**：K 是 active_num 的线性因子，K=8 时 active_num 是 K=1 的 7.2 倍，劣化也线性放大。

---

## 5. 修复方向

### 5.1 最直接：对齐 A3 的 dropless AER 条件（移除 A5 例外放行）

**移除 A5 `IsCountingSortApplicable()` line 2309-2312 的 `actualExpertNum != expertNum_` 例外**，回归 A3 严格语义：
```cpp
// 修改前（A5 当前）
if (actualExpertNum != expertNum_ && actualExpertNum * NUM_TWO > expertNum_) {
    return false;
}
// 修改后（对齐 A3 line 814）
if (actualExpertNum * NUM_TWO > expertNum_) {
    return false;
}
```

**效果**：fl_k8（actualExpertNum=256=expertNum_）的 dropPadValid 变为 false，CS 不启用 → TilingKey 400000 → **11000000（builtin）**，与 A3 行为完全对齐。

| 项 | 改前 | 改后 |
|----|------|------|
| TilingKey | 400000（CS FullLoad）| **11000000（builtin）**，对齐 A3 的 1100000 |
| DYN_PERF | 376.474 µs | **~227 µs** |
| Δ% | **+65.9%** | **0%** |

**风险**：需验证 dropPad + 全专家范围场景是否依赖此例外（dropPad 走另一分支 line 2313-2317，不受影响）。

### 5.2 短期：dropless 跳过 FullLoad（与 `cs_tiling_modification_plan.md` 改动 3 一致）

dropless 路径跳过 FullLoad 估算，直接尝试 CutOrigin；CutOrigin 若不适用则回退 builtin。比 5.1 更激进——不限于全专家范围场景。

### 5.3 中期：Phase C 批量化重构（保留 FullLoad 路径）

参考 CutOrigin Phase C 双缓冲 ping-pong（`moe_v3_counting_sort_cut_origin_unquantized.h::GatherAndWriteDoubleBuffer` line 167-331）：
1. 批量预计算 newPos/srcRow/origFlatIdx 数组（栈上 int32_t[2048]）
2. 主循环按 batch（如 64 pair/批）ping-pong：
   - batch i：构造 idxBuf 数组（Duplicate + Adds newPos/origFlatIdx）→ 单次 DataCopyPad 写 batch 行 rowIdx
   - batch i+1：预取
   - 主循环每行 x DataCopyPad 与下一 batch 的 idx 构造重叠
3. 1125 次/核循环降为 ~18 批/核（按 64 pair/批），SetWaitFlag + DataCopyPad 次数大幅降低

### 5.4 长期：评估是否保留 dropless FullLoad 模板

`perf_baseline_vs_new_template.md` §6.5 记录 FullLoad K=1 劣化 +20%（非算法级），本次实测 K=8 劣化 +65.9%。结合 `cs_optimization_conditions_summary.md` §3（dropless FullLoad 全场景劣化）：
- **dropless FullLoad 无优化信号**——建议按 5.1/5.2 短期拦截，长期考虑废弃该模板（dropless 路径全部走 CutOrigin 或 builtin）
- **dropPad FullLoad 保持优化**（-24%~-33%，13/13 复现）——dropPad 路径 FullLoad 必须保留

---

## 6. 结论

1. K=8 + H=128 + dropless + GATHER + unquant + COUNT 用例（`fl_k8`，全专家范围 actualExpertNum=256=expertNum_）在 A5 FullLoad 下劣化 **+65.9%**（376.5 vs 227.0 µs）。
2. **该用例在 A3 上不走 FullLoad**——A3 实测 TilingKey=1100000（builtin sort+gather 旧路径），与 A5 走的 400000（CS FullLoad）**路径不同构**。
3. **迁移回归点定位**：A5 `IsCountingSortApplicable()` line 2309-2312 加了 `actualExpertNum != expertNum_` 例外放行，让全专家范围场景绕过 A3 line 814 的严格 AER 检查（`actualExpertNum×2 ≤ expertNum_`），使 A5 在 A3 本不走 CS 的场景下错误启用 FullLoad。
4. 劣化主因：**Phase C 逐 pair 串行循环**中每核 ~1125 次「SetWaitFlag + 极小 DataCopyPad（4 字节 idx）」的累积固定开销，而 A5 builtin（11000000）用批量 sort+gather 在大 active_num 下更高效。
5. K 是放大因子：K=1 时 +23.6%，K=8 时 +65.9%，劣化随 K 近似线性放大。
6. 修复方向：
   - **最直接**：移除 A5 例外放行（§5.1），对齐 A3 行为，本例回退 builtin Δ%=0%；
   - **短期**：按 `cs_tiling_modification_plan.md` 改动 3 dropless 跳过 FullLoad；
   - **中期**：Phase C 批量化重构（参考 CutOrigin 双缓冲）。

---

**文档版本**：v1.1（2026-08-12，A3 实测验证后重大修订：路径不同构，定位 A5 例外放行迁移回归点）
**关联文档**：
- `perf_baseline_vs_new_template.md` §6.5（FullLoad K=1 +20% 劣化的次要差异分析，本文档补充 K=8 的放大效应）
- `cs_optimization_conditions_summary.md` §3（dropless FullLoad 全场景劣化的决策矩阵）
- `cs_92b_decode_degradation_analysis.md`（92B decode T=1792 FullLoad 劣化 +31.6% 的根因）
- `cs_tiling_modification_plan.md`（改动 3：dropless 跳过 FullLoad 的实施方案）
- `project_moe_v3_a3_vs_a5_rootcause.md`（A3→A5 迁移劣化的整体根因总览）
