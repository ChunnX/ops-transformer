# A5 CountingSort 优化规律分析（42 条用例交叉总结）

> **生成日期**：2026-08-10
> **数据来源**：3 份报告共 42 条用例交叉分析
> - `cs_vs_builtin_perf.md`（19 条混合场景：FullLoad 10 + CutOrigin 8 + 动态量化回落 1）
> - `cs_droppad_optimization_validation.md`（13 条 dropPad 同类验证）
> - `cs_network_shapes_perf.md`（10 条网络典型 shape：RL 35B + 92B + 505B decode/prefill）
> **指标**：DYN_PERF（µs，越小越好），Δ% = (CS - Builtin) / Builtin × 100%
> **判定**：|Δ%| > 1% 为劣化/优化，否则持平

---

## 1. 强相关因子排序

按对 Δ% 的区分度（能否稳定区分优化/劣化），影响因子从强到弱：

| 排名 | 因子 | 区分度 | 说明 |
|------|------|--------|------|
| **1** | **drop_pad_mode** | 极强 | dropPad=1 几乎必优化，drop=0 多数劣化 |
| **2** | **row_idx_type (GATHER/SCATTER)** | 强 | SCATTER 必劣化，GATHER 视模板而定 |
| **3** | **模板 × cols** | 强 | FullLoad+小列劣化，CutOrigin+大列接近持平 |
| **4** | **active_num (T×K) 规模** | 强 | active_num≥20000 优化，≤9000 劣化（LE/AER 无影响，见 `cs_le_boundary_test.md`）|
| **5** | **dtype / quant_mode** | 弱 | 趋势一致，幅度差异 <5% |
| **6** | **LE / AER 宽窄** | 弱 | LE≤256 范围内不影响优化方向，仅影响幅度 <10%（见 `cs_le_boundary_test.md`）|
| **7** | **expertCapacity / N** | 弱 | 测试区间内影响 <5% |

---

## 2. 四个明确区间

### 2.1 ✅ 优化区（CS 应启用）

#### 规律 1：dropPad 是 CS 唯一稳定优化信号

| 模板 | cols | Δ% 区间 | 复现率 |
|------|------|---------|--------|
| FullLoad + dropPad | 128/512 | **-24% ~ -33%** | 7/7 |
| CutOrigin + dropPad | 4096/7168 | **-5% ~ -15%** | 6/6 |

- **13/13 全复现**，跨 dtype（fp16/bf16/fp32）/ N（4096~9000）/ cap（32~128）稳定；**quant 仅验证 unquant=-1**，static/dynamic quant + dropPad 组合未测
- **根因**：dropPad 场景旧路径需逐 pair 判定 slot 溢出 + 丢弃行写 -1；CS 的**批量 staging + ping-pong + dropPad seed 预计算**有结构性优势，不依赖列大小或 T
- **幅度规律**：FullLoad(-30%) > CutOrigin(-10%)，因为 FullLoad 无 Phase C 流式载入开销，dropPad 收益净占比更高

#### 规律 2：CutOrigin + dropless + GATHER 优化由 active_num 驱动（非 LE/AER）

> **v1.1 修正（2026-08-11）**：v1.0 原归因为「窄 AER 优化潜力」，经 `cs_le_boundary_test.md` 补测推翻。LE=6/AER=1.6% 在 active_num=9000 时劣化 +10%，在 active_num=20480 时优化 -7.1%，**优化方向由 active_num 决定，与 LE/AER 无关**。

| 用例 | active_num | LE | AER | Δ% | 判定 |
|------|-----------|-----|------|------|------|
| 补测 ti=0 | **9000** | 6 | 1.6% | **+10.0%** | ❌ |
| 补测 ti=4 | **9000** | 256 | 100% | +1.9% | ❌ |
| 网络 ti=7 | **20480** | 6 | 1.6% | **-7.1%** | ✅ |
| 网络 ti=5 | **163840** | 24 | 12.5% | **-3.7%** | ✅ |

- **优化边界**：active_num ≈ **15000~20000**（9000 劣化，20480 优化，中间无数据点）
- **根因**：CS 复杂度 O(active_num+LE) vs 旧路径 O(active_num×log(active_num))；active_num 大时 CS 算法优势体现，固定开销被摊薄
- **LE/AER 非因子**：LE≤256 范围内 O(LE) 远小于 O(active_num)，对 Δ% 贡献 <10%，无法独立改变方向

---

### 2.2 ❌ 劣化区（CS 应禁用或优化）

#### 规律 3：SCATTER（row_idx_type=1）必劣化

| 用例 | 模板 | Δ% |
|------|------|-----|
| ti=3 FL unquant scatter | FullLoad | **+42.7%** |
| ti=7 FL static scatter | FullLoad | **+30.1%** |
| ti=13 CO unquant scatter | CutOrigin | **+26.4%** |
| ti=14 CO static scatter | CutOrigin | **+22.9%** |

- **4/4 劣化，幅度 +22%~+43%**，是所有因子中最一致的劣化信号
- **根因**：SCATTER 的 `expandedRowIdx[newPos]=origFlatIdx` newPos 离散，**逐行写小 DMA(4B)**，S↔MTE3 双向同步成最长串行链；GATHER 连续批可批量 staging，SCATTER 无法
- **规律**：劣化幅度 FullLoad > CutOrigin，unquant ≈ static

#### 规律 4：FullLoad + 小列（cols=128）+ dropless 普遍劣化

| 子集 | Δ% 区间 | 平均 |
|------|---------|------|
| unquant GATHER（ti=0/1/2/4/5）| +15.0% ~ +22.5% | +19.0% |
| static GATHER（ti=6/8）| +16.4% ~ +20.1% | +18.2% |
| 任意 SCATTER（ti=3/7）| +30.1% ~ +42.7% | +36.4% |

- **根因**：FullLoad 的 Phase A/B/C 三相流水线 + SyncAll + workspace 读写是**固定开销**；cols=128 数据量小，旧路径常数开销本就不是瓶颈，CS 固定成本无法摊薄
- **唯一例外**：dropPad（规律 1），因 dropPad 收益足以覆盖固定开销

#### 规律 5：FullLoad + 中等 T decode（网络 shape）劣化

- ti=6 92B decode（T=1792, cols=14336）FullLoad **+33.9%** — 验收阻断点（§2.9 首要验收 shape）
- 与规律 4 同源：FullLoad 固定开销在 decode 中等 T 下仍无法摊薄
- **边界**：T 再大（T≥65536 prefill）CS UB 无法全载 → 回落 11000000，CS 不适用

---

### 2.3 ⚪ 持平区（CS 边际收益不明显）

#### 规律 6：CutOrigin + 大列 + GATHER + dropless 接近持平

| 用例 | Δ% | 判定 |
|------|-----|------|
| ti=12 CO fp32 GATHER COUNT | **-4.0%** | ✅轻微优化 |
| ti=15 CO static GATHER CUMSUM | -0.8% | ⚪持平 |
| ti=11 CO bf16 GATHER COUNT | +5.9% | ❌轻微劣化 |
| ti=40 CO CUMSUM | +5.4% | ❌轻微劣化 |
| ti=41 CO KEY_VALUE | +3.8% | ❌轻微劣化 |

- 双缓冲重建后，CutOrigin 大列 GATHER 已能**接近旧路径**，但仍未完全追平（多数在 0~+6%）
- **规律**：fp32 / CUMSUM 偏优化侧，bf16 / 偏劣化侧，幅度都小

#### 规律 7：Histograms 在全通过 AER 下纯负收益

| 模板 | 平均 Δ% | 判定 |
|------|---------|------|
| CutOrigin（MOE_CS_USE_HISTOGRAM）| +0.3% | 持平偏劣化 |
| FullLoad（MOE_CS_FULLLOAD_USE_HISTOGRAM）| +2.3% | 一致劣化 |

- **根因**：测试集 AER 全通过（expertIdx 值域 `[0,LE)`），向量过滤压缩不出淘汰元素，Histograms 整条计数链（Adds + 两步 Cast + VF + 回写）是纯开销
- **推论**：窄 AER 真实过滤场景下可能有正收益，但当前测试集无法验证，两宏默认关闭

---

### 2.4 🚫 不适用区（CS 回落）

| 场景 | 回落路径 | 原因 |
|------|---------|------|
| 小 T + 窄 AER（RL decode T≤256）| builtin 200000 | builtin 有专属优化路径，CS 完全不介入 |
| 超大 T prefill（T≥65536）| builtin 11000000 | CS UB 无法全载 expertIdx |
| 动态量化（quant_mode=1）| builtin 11020000 | CS 主动关闭 |

---

## 3. 决策矩阵（建议）

基于上述规律，给出 CS 启用决策矩阵：

| 条件 | 当前行为 | 建议 | 依据 | 预期 Δ% |
|------|---------|------|------|---------|
| **dropPad=1** | 启用 | ✅ **保留启用** | 规律1 | -5% ~ -33% |
| **SCATTER (row_idx=1)** | 启用 | ❌ **建议禁用** | 规律3 | +22% ~ +43% |
| **FullLoad + dropless + 小列** | 启用 | ❌ **建议禁用** | 规律4 | +15% ~ +30% |
| **FullLoad + 中等 T decode** | 启用 | ❌ **建议禁用**（92B 验收 shape）| 规律5 | +33.9% |
| **CutOrigin + dropless + GATHER（active_num≥20000）** | 启用 | ✅ **保留启用** | 规律2（v1.1修正）| -3.7% ~ -7.1% |
| **CutOrigin + dropless + GATHER（active_num≤9000）** | 启用 | ❌ **建议禁用** | 规律2（v1.1修正）| +1.9% ~ +10% |
| **CutOrigin + 大列 + GATHER + dropless（active_num 9000~20000）** | 启用 | ⚪ **边界未知** | 规律2/6 | 边界区间无数据 |
| **CutOrigin + 大列 + GATHER + dropless（AER=100% 全专家）** | 启用 | ⚪ **可选保留** | 规律6 | -4% ~ +6% |
| **动态量化** | 回落 | ✅ 保持回落 | — | 同路径 |
| **超大 T (≥65536)** | 回落 | ✅ 保持回落 | UB 约束 | 同路径 |
| **Histograms 宏** | 关闭 | ✅ 保持关闭 | 规律7 | +0.3% ~ +2.3% |

---

## 4. 核心结论

1. **dropPad 是 CS 唯一的结构性优化场景**（13/13 复现），收益来自批量 staging + seed 预计算，与列大小/T 无关
2. **SCATTER 是 CS 的结构性劣势场景**（4/4 劣化），逐行小 DMA 无法批量，是算法级瓶颈，短期难优化
3. **FullLoad 固定开销**是劣化主因（decode 小/中列场景），三相流水线 + SyncAll 在小数据上无法摊薄
4. **CutOrigin 大列 GATHER** 经双缓冲重建后已接近追平旧路径，但多数仍轻微劣化（0~+6%），未完全追平
5. **active_num 是 CutOrigin + dropless + GATHER 的真正驱动因子**（v1.1 修正）：active_num≥20000 优化（-3.7%~-7.1%），≤9000 劣化（+1.9%~+10%）。LE/AER 在 LE≤256 范围内不影响优化方向（详见 `cs_le_boundary_test.md`）
6. ~~窄 AER 的优化潜力~~（v1.0 推测）已被补测推翻：网络 shape ti=5/7 的优化来自 active_num 大，不是 AER 窄

---

## 5. 对 92B decode 验收阻断点的建议

§2.9 首要验收 shape（T=1792, H=2560, K=8, E=256, LE=32）当前 CS FullLoad 劣化 **+33.9%**，是性能验收的阻断点。

**建议方案**（按优先级）：

1. **调整 `IsCountingSortApplicable` 启用门槛**：对「FullLoad + dropless + 中等 T（decode 类，T≤4096）」禁用 CS，走 builtin 11000000 路径（当前 perf=73.782µs 已较优）
2. **保留 dropPad 场景启用**：dropPad 收益稳定，不应波及
3. **保留 active_num≥20000 的 CutOrigin + dropless + GATHER 启用**（v1.1 修正）：大 active_num 场景有稳定优化收益

**预期效果**：消除 92B decode +33.9% 劣化，同时保留 dropPad（-5~-33%）和大 active_num CutOrigin（-3.7%~-7.1%）的优化收益。

---

**文档版本**：v1.1（2026-08-11 修正规律2：active_num 替代 LE/AER 作为驱动因子）
**生成日期**：2026-08-10（v1.1：2026-08-11）
**关联文档**：
- `cs_vs_builtin_perf.md`（19 条混合场景对比）
- `cs_droppad_optimization_validation.md`（13 条 dropPad 同类验证）
- `cs_network_shapes_perf.md`（10 条网络典型 shape）
- `cs_le_boundary_test.md`（**LE 边界探索，推翻 LE/AER 驱动假设**）
- `fullload_histogram_plan.md`（FullLoad Histograms 改造方案与 A/B 结果）
