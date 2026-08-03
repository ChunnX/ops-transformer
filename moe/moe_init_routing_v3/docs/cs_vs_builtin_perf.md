# A5 CountingSort 模板 vs 源代码（builtin）性能对比报告

> **测试日期**：2026-08-07（v1.1：2026-08-08 ti=12/ti=15 重测勘误）
> **对比方式**：卸载 vendor 下 custom_transformer 包 → 跑 builtin（源代码）基线 → 重装 custom 包
> **指标**：DYN_PERF（µs，越小越好）
> **判定阈值**：|Δ%| > 1% 为劣化/优化，否则持平
> **测试设备**：device 4 / 6（Ascend950DT）
>
> ⚠️ **v1.1 勘误**：原 v1.0 测试用 ttk 默认并发跑，log 行错位导致部分用例 perf 误配。ti=12/ti=15 已于 2026-08-08 用 `--pc=1` 单条串行 3 次重测（标 ²），其余用例仍为 v1.0 数据待后续复测。

---

## 1. 完整对比表（19 条用例）

| ti | 用例名 | 模板 | dtype | quant | row_idx | etnt | dropPad | CS 模板 (µs) | 源代码 (µs) | Δ% | 判定 | CS TilingKey | 源代码 TilingKey | csv位置 | 宏关 baseline (µs) | Histograms (µs) | Hist Δ% |
|----|--------|------|-------|-------|---------|------|---------|-------------|------------|-----|------|-------------|-----------------|---------|--------------------|-----------------|---------|
| 0 | FL_unquant_fp16_gather_count_drop | FullLoad | fp16 | -1 | GATHER | COUNT | 0 | 65.9 | 54.1 | **+21.9%** | ❌劣化 | 400000 | 11000000 | cs_coverage.csv#ti=0 | 65.417 | 67.353 | +3.0% ❌ |
| 1 | FL_unquant_bf16_gather_count_drop | FullLoad | bf16 | -1 | GATHER | COUNT | 0 | 66.2 | 54.1 | **+22.5%** | ❌劣化 | 400000 | 11000000 | cs_coverage.csv#ti=1 | 66.764 | 67.260 | +0.7% ⚪ |
| 2 | FL_unquant_fp32_gather_count_drop | FullLoad | fp32 | -1 | GATHER | COUNT | 0 | 66.1 | 57.5 | **+15.0%** | ❌劣化 | 400000 | 11000000 | cs_coverage.csv#ti=2 | 65.963 | 67.696 | +2.6% ❌ |
| 3 | FL_unquant_fp16_scatter_count_drop | FullLoad | fp16 | -1 | SCATTER | COUNT | 0 | 71.5 | 50.1 | **+42.7%** | ❌劣化 | 401000 | 11001000 | cs_coverage.csv#ti=3 | 71.460 | 72.733 | +1.8% ❌ |
| 4 | FL_unquant_fp16_gather_cumsum_drop | FullLoad | fp16 | -1 | GATHER | CUMSUM | 0 | 66.0 | 55.5 | **+19.1%** | ❌劣化 | 400000 | 11000000 | cs_coverage.csv#ti=4 | 66.005 | 67.120 | +1.7% ❌ |
| 5 | FL_unquant_fp16_gather_kv_drop | FullLoad | fp16 | -1 | GATHER | KEY_VALUE | 0 | 65.7 | 56.4 | **+16.5%** | ❌劣化 | 400000 | 11000000 | cs_coverage.csv#ti=5 | 65.067 | 66.364 | +2.0% ❌ |
| 6 | FL_static_fp16_gather_count_drop | FullLoad | fp16 | 0 | GATHER | COUNT | 0 | 71.0 | 59.1 | **+20.1%** | ❌劣化 | 410000 | 11010000 | cs_coverage.csv#ti=6 | 69.828 | 72.410 | +3.7% ❌ |
| 7 | FL_static_fp16_scatter_count_drop | FullLoad | fp16 | 0 | SCATTER | COUNT | 0 | 75.4 | 58.0 | **+30.1%** | ❌劣化 | 411000 | 11011000 | cs_coverage.csv#ti=7 | 75.191 | 76.845 | +2.2% ❌ |
| 8 | FL_static_fp16_gather_cumsum_drop | FullLoad | fp16 | 0 | GATHER | CUMSUM | 0 | 69.9 | 60.1 | **+16.4%** | ❌劣化 | 410000 | 11010000 | cs_coverage.csv#ti=8 | 70.147 | 71.004 | +1.2% ❌ |
| 9 | FL_unquant_fp16_gather_count_pad | FullLoad | fp16 | -1 | GATHER | COUNT | 1 | 62.5 | 90.8 | **-31.2%** | ✅优化 | 400100 | 11000100 | cs_coverage.csv#ti=9 | 63.918 | 66.622 | +4.2% ❌ |
| 10 | FL_dynamic_fp16_gather_count_drop | FullLoad | fp16 | 1 | GATHER | COUNT | 0 | 回落 | 87.8 | N/A | 持平 | 11020000 | 11020000 | cs_coverage.csv#ti=10 | N/A（dynamic 回落） | N/A | — |
| 11 | CO_unquant_bf16_gather_count_drop | CutOrigin | bf16 | -1 | GATHER | COUNT | 0 | 112.7 | 106.4 | **+5.9%** | ❌劣化 | 500000 | 11000000 | cs_coverage.csv#ti=11 | 114.174 | 115.034 | +0.8% ⚪ |
| 12 | CO_unquant_fp32_gather_count_drop | CutOrigin | fp32 | -1 | GATHER | COUNT | 0 | 174.996² | 182.343² | **-4.0%** | ✅优化 | 500000 | 11000000 | cs_coverage.csv#ti=12 | 175.587 | 176.278 | +0.4% ⚪ |
| 13 | CO_unquant_fp16_scatter_count_drop | CutOrigin | fp16 | -1 | SCATTER | COUNT | 0 | 135.2 | 106.9 | **+26.4%** | ❌劣化 | 501000 | 11001000 | cs_coverage.csv#ti=13 | 135.385 | 136.204 | +0.6% ⚪ |
| 14 | CO_static_fp16_scatter_count_drop | CutOrigin | fp16 | 0 | SCATTER | COUNT | 0 | 120.8 | 98.3 | **+22.9%** | ❌劣化 | 511000 | 11011000 | cs_coverage.csv#ti=14 | 121.935 | 123.178 | +1.0% ⚪ |
| 15 | CO_static_fp16_gather_cumsum_drop | CutOrigin | fp16 | 0 | GATHER | CUMSUM | 0 | 100.328² | 101.081² | **-0.8%** | ⚪持平 | 510000 | 11010000 | cs_coverage.csv#ti=15 | 100.962 | 101.942 | +1.0% ⚪ |
| 40 | A5_cutorigin_cumsum | CutOrigin | fp16 | -1 | GATHER | CUMSUM | 0 | 114.3 | 108.4 | **+5.4%** | ❌劣化 | 500000 | 11000000 | moe_v3_test.csv#ti=40 | 114.822 | 113.365 | -1.3% ✅ |
| 41 | A5_cutorigin_key_value | CutOrigin | fp16 | -1 | GATHER | KEY_VALUE | 0 | 114.8 | 110.6 | **+3.8%** | ❌劣化 | 500000 | 11000000 | moe_v3_test.csv#ti=41 | 114.494 | 114.129 | -0.3% ⚪ |
| 42 | A5_cutorigin_droppad | CutOrigin | fp16 | -1 | GATHER | COUNT | 1 | 178.0 | 192.4 | **-7.5%** | ✅优化 | 500100 | 11000100 | moe_v3_test.csv#ti=42 | 176.960 | 176.473 | -0.3% ⚪ |

### 汇总

| 判定 | 数量 | 占比 |
|------|------|------|
| ❌ 劣化 | 14 | 73.7% |
| ✅ 优化 | 3 | 15.8% |
| ⚪ 持平 | 2 | 10.5% |

> **ti=12 / ti=15 重测勘误（2026-08-08，device 4 单条串行 3 次）**：
> - **ti=12**（CO unquant fp32 GATHER COUNT）：原 CS=176.3 / builtin=181.4 / Δ%=-2.8% → 重测 CS=**174.996** / builtin=**182.343**（181.477/182.141/183.41 均值）/ Δ%=**-4.0%**，仍判定 ✅优化，幅度略增。
> - **ti=15**（CO static fp16 GATHER CUMSUM）：原 CS=100.4 / builtin=102.9 / Δ%=-2.4% → 重测 CS=**100.328**（100.586/100.07 均值）/ builtin=**101.081**（102.002/102.815/98.425 均值）/ Δ%=**-0.8%**，|Δ%|<1% 改判 ⚪持平。
> - 汇总相应调整：✅优化 4→3，⚪持平 1→2。
> - 这两条 dropless+GATHER+大列场景的优化幅度（< 5%）远不及 dropPad（-5%~-33%），不属于 CS 模板的稳定优化区间。

> **Histograms A/B v2（2026-08-08，Phase C 双缓冲落地后复测）**：
> - 在 CutOrigin 计数段引入 `MOE_CS_USE_HISTOGRAM` 宏（向量过滤+int32→int16→uint8 窄化+`Histograms` API 计数），FullLoad 未改造。
> - 新增"Histograms (µs)"列 = 宏开 perf，"Hist Δ%"列 = (宏开 - 宏关) / 宏关 × 100%（基于本次重测 baseline，非表中"CS 模板 (µs)"列的历史数据）。
> - 测试条件：device 1/4/6 单条串行 `--pc=1` + `TTK_DISABLE_BUILTIN=0`（builtin 作 golden），CutOrigin 8 条（cs_coverage.csv ti=11-15 + moe_v3_test.csv ti=40-42）。
> - 结果：8/8 PRECISION PASS，平均 Δ% ≈ **+0.3%（基本持平，轻微劣化趋势）**。✅优化 1（ti=40 -1.3%）/ ⚪持平 7（|Δ%| ≤ 1.0%）/ ❌劣化 0。
> - **结论与 v1（2026-08-05）一致：Histograms 在当前测试集（AER=[0,256] 全通过）为非正收益**。根因：active_expert_range 全通过时 Phase A 向量过滤压缩不出淘汰元素，Histograms 整条计数链是纯开销。宏默认关闭，代码保留供未来窄 range / 极端倾斜场景复验。
> - 数据可靠性：ti=13 首次在 device 1 跑出 380.976µs（+181%），疑设备不稳；device 6 重测 136.204µs（+0.6%），确认 +181% 为伪数据。

> **FullLoad Histograms A/B（2026-08-08，device 6，独立宏 `MOE_CS_FULLLOAD_USE_HISTOGRAM`）**：
> - 在 FullLoad 模板 Phase A `FilterAndCountVector` 的标量 count 循环引入 Histograms（与 CutOrigin 的 `MOE_CS_USE_HISTOGRAM` 独立，两个模板按 TilingKey 分流互不污染）。
> - 测试条件：device 6 单条串行 `--pc=1` + `TTK_DISABLE_BUILTIN=0`，FullLoad 10 条（cs_coverage.csv ti=0-9；ti=10 dynamic 回落 N/A）。
> - 结果：10/10 PRECISION PASS，平均 Δ% ≈ **+2.3%（一致劣化）**。✅优化 0 / ⚪持平 1（ti=1 +0.7%）/ ❌劣化 9（+1.2% ~ +4.2%）。
> - **结论：FullLoad Histograms 为纯负收益，劣化幅度（+2.3%）比 CutOrigin（+0.3%）更明显**。根因：FullLoad cols=128 小列、coreEntries 小、baseline 耗时本就短（~65-76µs），Histograms 固定开销（Adds + 两步 Cast + Histograms VF + uint16→int32 回写）的 ~1-3µs 增量在小基数上占比更高。ti=9 dropPad 劣化最严重（+4.2%），说明 dropPad 优化由批量 staging 承担、非计数段，Histograms 无加成。
> - 宏 `MOE_CS_FULLLOAD_USE_HISTOGRAM` 默认关闭，代码保留。两个模板 Histograms 均为负收益，方向一致。方案见 `fullload_histogram_plan.md`。

---

## 2. 按模板分组分析

### 2.1 FullLoad（cols=128，10 条有效对比）

| 维度 | 劣化区间 | 平均劣化 | 说明 |
|------|---------|---------|------|
| unquant GATHER（ti=0/1/2/4/5）| +15.0% ~ +22.5% | **+19.0%** | dtype/etnt 变化不影响趋势 |
| unquant SCATTER（ti=3）| +42.7% | **+42.7%** | SCATTER 劣化最严重 |
| static（ti=6/7/8）| +16.4% ~ +30.1% | **+22.2%** | static quant 劣化显著 |
| dropPad（ti=9）| **-31.2%** | **优化** | FullLoad + dropPad 唯一优化点 |

**FullLoad 结论**：cols=128 小列场景，CS 模板整体劣化（平均 +20%）。原因：
- 小列场景数据量小，旧 sort 路径的常数开销已不是瓶颈
- CS 模板的 Phase A/B/C 三相流水线引入额外同步开销（SyncAll、workspace 读写）
- FullLoad 全载模型在小列下 UB 利用率低，无法摊薄 Phase A 向量过滤的固定成本

**唯一优化点**：FullLoad + dropPad（ti=9，-31.2%）。dropPad 场景旧路径需逐 pair 判定 slot 溢出，CS 模板的批量 staging + ping-pong 在此场景有正收益。

### 2.2 CutOrigin（cols=7168，8 条有效对比）

| 维度 | 劣化/优化 | 说明 |
|------|----------|------|
| unquant GATHER COUNT（ti=11/12）| -4.0% ~ +5.9%² | bf16 轻微劣化，fp32 轻微优化（ti=12 重测 -4.0%） |
| unquant SCATTER（ti=13）| **+26.4%** | SCATTER 劣化严重 |
| static SCATTER（ti=14）| **+22.9%** | static + scatter 劣化严重 |
| static GATHER CUMSUM（ti=15）| -0.8%² | 重测确认接近持平（|Δ%|<1%） |
| CUMSUM/KEY_VALUE（ti=40/41）| +3.8% ~ +5.4% | 轻微劣化 |
| dropPad（ti=42）| **-7.5%** | dropPad 优化 |

**CutOrigin 结论**：
- **GATHER + 大列** 接近持平或轻微劣化（±5%），CS 双缓冲已重建但仍未完全追平旧路径
- **SCATTER** 劣化严重（+22%~26%），SCATTER 场景 rowIdx 写出策略在 CS 模板下效率更低
- **dropPad** 优化（-7.5%），与 FullLoad dropPad 趋势一致

---

## 3. 关键模式总结

### 3.1 SCATTER 场景普遍劣化严重

| 用例 | 模板 | Δ% |
|------|------|-----|
| ti=3 FL unquant scatter | FullLoad | **+42.7%** |
| ti=7 FL static scatter | FullLoad | **+30.1%** |
| ti=13 CO unquant scatter | CutOrigin | **+26.4%** |
| ti=14 CO static scatter | CutOrigin | **+22.9%** |

**根因**：SCATTER 场景 rowIdx 写出策略（`expandedRowIdx[newPos] = origFlatIdx`）在 CS 模板下逐行写小 DMA（4B），S↔MTE3 双向同步往返成为最长串行链。GATHER 连续批可批量 staging，但 SCATTER 的 newPos 离散，无法批量。

### 3.2 dropPad 场景一致优化

| 用例 | 模板 | Δ% |
|------|------|-----|
| ti=9 FL dropPad | FullLoad | **-31.2%** |
| ti=42 CO dropPad | CutOrigin | **-7.5%** |

**根因**：dropPad 场景旧路径需逐 pair 判定 slot 溢出 + 丢弃行写 -1，CS 模板的批量 staging + ping-pong + dropPad seed 预计算在此场景有正收益。

### 3.3 动态量化持平（CS 关闭回落）

ti=10 两边都走 TilingKey=11020000（旧路径），性能一致（CS 模板对动态量化已主动关闭）。

---

## 4. 建议

### 4.1 短期（当前状态）

CS 模板在 **大多数小列/中列场景劣化**，仅在 **dropPad 场景优化**。当前 `IsCountingSortApplicable` 的启用门槛可能过于宽松，导致 CS 在不该启用的场景也被启用。

**建议调整 tiling 启用条件**（待讨论）：
- FullLoad cols=128 小列场景：考虑禁用 CS（劣化 15-42%）
- SCATTER 场景：考虑禁用 CS 或优化 rowIdx 写出策略
- dropPad 场景：保留 CS（稳定优化 7-31%）

### 4.2 中期（性能优化方向）

1. **Phase A 向量过滤**（plan `precious-mixing-micali.md` 阶段二）：当前 Phase A 仍为纯标量扫描，小列场景下 Phase A 开销占比高，向量过滤可压缩
2. **SCATTER rowIdx 批量写出**：参考 GATHER 的连续批 staging 策略，对 SCATTER 的 newPos 排序后批量写出
3. **FullLoad 小列常数开销优化**：Phase A/B 的 SyncAll + workspace 读写在小列场景占比高，考虑合并

---

## 5. 附录：TilingKey 编码对照

| 路径 | TilingKey 编码 | 示例 |
|------|---------------|------|
| **CS FullLoad** | 400000 + quant×10000 + rowIdx×1000 + dropPad×100 | 400000/401000/410000/411000/400100 |
| **CS CutOrigin** | 500000 + quant×10000 + rowIdx×1000 + dropPad×100 | 500000/501000/510000/511000/500100 |
| **源代码（旧分阶段）** | 11000000 + quant×10000 + rowIdx×1000 + dropPad×100 | 11000000/11001000/11010000/11011000/11000100 |
| **动态量化回落** | 11020000（固定） | 11020000 |

---

**文档版本**：v1.1（ti=12/ti=15 重测勘误）
**生成日期**：2026-08-07（v1.1：2026-08-08）
**关联文档**：`cs_coverage_test_report.md`（CS 模板覆盖验证）、`cs_a3_a5_support_comparison.md`（支持矩阵）
