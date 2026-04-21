# Split-KN 收益再评估 (950, 三档 decode)

**对照基线**: split-N baseline (`n_groups=1, k_groups=1`, 即关闭 split-KN)
**模型约定**: H6 下 `sync_overhead_us = 0`；所有时间为 chip-total pipeline 关键路径 (µs)
**搜索空间**: MM1 配置 32 个 + MM2 配置 14 个 = 495 种 (MM1, MM2) 联合

---

## 1. 基线 vs Split-KN vs Split-KN-Fused

| Testcase | Baseline (µs) | Split-KN best (µs) | 绝对节省 | 相对节省 | Split-KN-Fused best (µs) | Fused 额外收益 |
|---|---|---|---|---|---|---|
| `perf_decode_bs1` (BF16, T=1) | 91.26 | **71.75** | −19.51 | −21.4% | 71.83 | **−0.09 µs (劣化)** |
| `perf_decode_int8` (FULL_QUANT+KV_PT) | 56.38 | **39.89** | −16.49 | −29.2% | 40.00 | **−0.11 µs (劣化)** |
| `perf_decode_bs8` (BF16, T=8) | 91.27 | **71.76** | −19.51 | −21.4% | 71.98 | **−0.22 µs (劣化)** |

**结论 1 — Split-KN 收益稳定且显著**：三档用例均 **20–30% 端到端节省**，int8 收益最大（baseline MM3 时间减半后，MM1/MM2 相对占比更高）。

**结论 2 — Split-KN-Fused 在当前建模约定下 "负收益"**：与 CLAUDE.md §8 limitation #10 完全吻合。
- 当 `sync_overhead_us = 0` (H6) 时，融合省掉的"独立 barrier" 成本是 0。
- 融合路径仍需多走一次 UB 搬运 + accum_load/store 开销，反而比 split-kn 基线慢 0.1–0.3%。
- **不建议** 在当前硬件约定下实现 split-kn-fused；仅作为未来若重新纳入 sync 成本时的开关保留。

---

## 2. 最优切分配置

| Testcase | 最优 (MM1, MM2) | MM1 n×k | MM2 n×k | MM1 cube (µs) | MM2 cube (µs) | MM3 cube (µs) | Cube util |
|---|---|---|---|---|---|---|---|
| decode_bs1 BF16 | `MM1:N4xK7 + MM2:N1xK28` | 4 × 7 = 28c | 1 × 28 = 28c | 15.62 | 6.42 | **49.04** | 100% |
| decode_int8 | `MM1:N2xK14 + MM2:N1xK28` | 2 × 14 = 28c | 1 × 28 = 28c | 8.47 | 3.44 | **27.27** | 99% |
| decode_bs8 BF16 | `MM1:N4xK7 + MM2:N1xK28` | 4 × 7 = 28c | 1 × 28 = 28c | 15.62 | 6.42 | **49.04** | 100% |

**观察**：
- **MM2 稳定收敛到 N1×K28**（把 N 放在 1 组、K 切成 28 片）。原因：`He=7168` 作为 K 轴远大于 `Hckv+Dr=576` 的 N 轴，K 方向切得越狠越能平分 HBM 压力，而 N=576 本身已经 ≤ `singleN` 常用值，再切 N 没收益。
- **MM1 有三种等效最优**（`N4×K7`、`N2×K14`、`N1×K28`），端到端时间完全相同（见第 3 节）——说明 MM1 已被 MM3 shadow，任何 28 核配置下 MM1 cube 都能藏在 MM3 关键路径里。
- **AIV 负载只占 1.1–1.7 µs**（vs pipeline ~70 µs）——AIV 侧累加成本几乎被完全 pipeline 重叠掩盖。

---

## 3. 联合搜索 vs 独立搜索：是否漏解？

默认实现 (`perf/tiling/split_kn_search.py`) 的策略是 **独立取 best(MM1) + best(MM2)**，理论上可能漏掉 AIC/AIV 耦合下的 pareto-optimal 联合解。本次做了完整 495 种 (MM1 × MM2) 联合搜索：

```
decode_bs1 BF16 Top-3 (全部并列最优 71.75 µs):
  MM1:N4xK7  + MM2:N1xK28   →  71.75  (AIC=71.75, AIV=1.13)  ← 独立搜索也会选到
  MM1:N2xK14 + MM2:N1xK28   →  71.75  (AIC=71.75, AIV=1.21)
  MM1:N1xK28 + MM2:N1xK28   →  71.75  (AIC=71.75, AIV=1.37)
```

**结论 3 — 独立搜索策略在 3 档用例下均找到真·全局最优**。没有任何联合配置能超过 best(MM1)+best(MM2) 的独立组合。
- 根本原因：split-KN 后 cube_util = 100%，pipeline = AIC 关键路径；AIC 关键路径是 stages 之和（在 DAG 里 MM1→MM2→MM3→MM4 串行），每段独立优化 ⇒ 全局最优。
- AIV 累加时间远低于 AIC，不会成为新的瓶颈，因此没有"MM1 让步 AIV 换 MM2 更激进 K-split" 的 trade-off 空间。

⇒ **不需要** 扩展 `search_split_kn` 做联合搜索。

---

## 4. 瓶颈已迁移：新的首要目标是 MM3

Split-KN 落地后，所有三档用例的 **MM3 占比高达 68–69%**：

| Testcase | Total | MM1 | MM2 | MM3 | MM4 | Accum | MM3 占比 |
|---|---|---|---|---|---|---|---|
| decode_bs1 BF16 | 71.75 | 15.62 | 6.42 | **49.04** | 0.66 | 0.18 | **68.4%** |
| decode_int8 | 39.89 | 8.47 | 3.44 | **27.27** | 0.33 | 0.26 | **68.4%** |
| decode_bs8 BF16 | 71.76 | 15.62 | 6.42 | **49.04** | 0.67 | 0.54 | **68.4%** |

**即使把 MM1+MM2 cube 时间缩到 0**（假设法宝），端到端也只能再省 ~22 µs (decode_bs1)，**上限 ~30% 额外加速，不如直接做 MM3 优化**。

**当前建模未覆盖的后续方向**：
1. **MM3 split-KN**（CLAUDE.md §8 limitation #6 明确未建模）：MM3 M=T=1, K=Hcq=1536, N=768/core，N 方向已饱满但 K 方向尚有 1536/128=12 片可切；K-split 可进一步下放 HBM 压力。
2. **DequantOpt 对 int8 链路的进一步收益**（`--mode dequant-opt` 已建模）：与 split-KN 正交、已落地节省 3.3 µs。
3. **Grouped-compute / msplit** (已建模)：对 T>1 或多 head 场景另辟路径。

---

## 5. 行动建议

1. ✅ **继续落地 split-KN**，三档 decode 用例收益稳定在 20–30%，配置固定为：
   - `MM1: n_groups=4, k_groups=7`（BF16）或 `n_groups=2, k_groups=14`（FULL_QUANT）
   - `MM2: n_groups=1, k_groups=28`（通用）
2. ❌ **暂缓 split-KN-fused 实现**：当前建模约定下无收益甚至轻微劣化。若未来改回 `sync_overhead_us > 0`，需重跑本分析。
3. ⏭ **下一目标优先级：MM3 优化**（建模扩展 + split-KN/grouped-compute 评估），潜在收益 > 剩余 MM1+MM2 空间。
4. ⚙ **默认搜索策略足够**：`search_split_kn` 的独立 best+best 与 495 种联合搜索一致，无需扩展为 Cartesian 搜索。

---

## 附：复现命令

```bash
cd attention/mla_prolog_v3/tests/perf

# 基线 + 最优搜索
python perf_analyzer.py --chip 950 --from-testcase perf_decode_bs1   --mode split-kn
python perf_analyzer.py --chip 950 --from-testcase perf_decode_int8  --mode split-kn
python perf_analyzer.py --chip 950 --from-testcase perf_decode_bs8   --mode split-kn

# Fused 对比（确认当前约定下 ≈ 0 收益）
python perf_analyzer.py --chip 950 --from-testcase perf_decode_bs1   --mode split-kn-fused

# 联合 MM1 × MM2 搜索（一次性验证独立搜索无遗漏）
python joint_split_kn_search.py
```
