# MlaPrologV3 Ascend 950 综合优化设计

> 版本：v1.1  日期：2026-04
> 适用范围：Ascend 950 chip，decode 与 small prefill 场景
> 状态：Phase 1（perf model）已完成，Phase 2（kernel 实现）待启动
> 关联 commit：`66b9f779f` (perf 重构) · `4c46bda97` (dequant-opt model) · `10e54bd18` (split-kn-fused model) · `45d7341ad` (split-kn chip-total 修正)
> 关联文档：`MlaPrologV3_DequantOpt设计.md`、`MlaPrologV3_SplitKN_FusedAccum设计.md`、`tests/perf/report_split_kn.md`
> **v1.1 变更**：vector 阶段从 chip-total ops/sec 粗粒度模型升级为指令级建模（256 B/cycle, IPC=1.3，按量化场景区分指令数）。FP8/INT8 baseline 偏移 ~0.25 µs，cube 时间及 split-kn 节省不变。

---

## 0. Executive Summary

本文汇总三项相互正交、可组合落地的 950 优化方案，覆盖 MlaPrologV3 整条流水：

| # | 优化 | 作用阶段 | 主要收益来源 | 实现成本 | 已落地 |
|---|---|---|---|---|---|
| 1 | **Split-KN** (修正版) | MM1 / MM2 cube | K 维核间切分压缩 cube 时间 | 中 (host tiling + cube 累加路径) | perf model ✓ |
| 2 | **Split-KN-Fused** | MM1 / MM2 vector 段 | K-accumulation 与 RmsNorm 融合，避免新 barrier + GM 往返 | 低 (vector 接口 + 一段 inner loop) | perf model ✓ |
| 3 | **DequantOpt** | MM3 → MM4 段 | L0C → UB → L1 intra-core 直通，避免 3 次 cross-core sync + 3 次 GM 往返 | 中 (intra-core handover + l0c_to_ub iterator 接通) | perf model ✓ |

**典型场景综合收益**（来自 `tests/perf/report_split_kn.md` 第 5 节及 perf model 实测）：

| 场景 | dtype | Baseline (us) | + Split-KN | + Split-KN-Fused | + DequantOpt | 综合 vs Baseline |
|---|---|---:|---:|---:|---:|---:|
| decode bs1 hd128 | FP8  | 48.97 | 39.45 (−19.4%) | ~37.92 (−22.6%) | **~34.39** | **−14.58 us / −29.8%** |
| decode bs1 hd128 | INT8 | 51.35 | 39.45 (−23.2%) | ~37.92 (−26.1%) | **~34.40** | **−16.95 us / −33.0%** |
| decode bs1 hd128 | BF16 | 87.63 | 71.53 (−18.4%) | ~70.01 (−20.1%) | n/a | −17.62 us / −20.1% |
| decode bs1 hd8   | BF16 | 43.33 | 27.23 (−37.2%) | ~25.72 (−40.7%) | n/a | −17.61 us / −40.7% |

> Split-KN 数据为 perf model `--mode split-kn` 实测；DequantOpt 数据为 `--mode dequant-opt` 实测；Split-KN-Fused 增量从 `--mode split-kn-fused` 取。FP8/INT8 baseline 的 vector 阶段采用指令级建模（256 B/cycle, IPC=1.3），与旧版 chip-total 模型相比 baseline 偏移 ~0.25 µs。

**核心思想**：
- Split-KN 解决 cube 阶段的 HBM 带宽利用率
- Split-KN-Fused 解决 cube → vector 衔接的浪费（新 barrier + 中转 GM）
- DequantOpt 解决 vector → cube 衔接的浪费（cross-core sync + 多次 GM 往返）
- 三者作用区域不重叠 → 完全可叠加

---

## 1. 现状与目标流水

### 1.1 Baseline 数据流（解码 bs1 / FP8 hd128 ≈ 49.0 µs，hd8 BF16 ≈ 43.3 µs）

```
AIC pipeline (cube)                         AIV pipeline (vector)
═══════════════════════════════════         ═══════════════════════════════════
MM1_Cq  ──FixPipe──→ GM(mmCqResGm)
       │ CrossCoreSetFlag PIPE_FIX (0.8 µs)
       ↓
       wait FINISH_VEC_RMSNORM_CQ           CrossCoreWaitFlag(FINISH_MM_CQ)
                                            WaitAllCore<ALL_VEC>             ★ existing barrier (kept)
                                            RmsNormCq:
                                              MTE2 GM → UB → PIPE_V → MTE3 → GM
                                            WaitAllCore<ALL_VEC>             ★ existing barrier (kept)
                                            CrossCoreSetFlag PIPE_MTE3 (0.8 µs)
                                            ────────────────────────────
MM2_CkvKr ──FixPipe──→ GM(mmCkvKrResGm)     CrossCoreWaitFlag(FINISH_MM_CKVKR)
       │ CrossCoreSetFlag (0.8 µs)          WaitAllCore<ALL_VEC>             ★ existing barrier (kept)
                                            RmsNormCkvKr + RopeKr + Scatter
       ↓
       wait FINISH_RMS_CKVKR
       ↓
MM3_QcQr ──FixPipe──→ GM(mmQcQrResGm)
       │ CrossCoreSetFlag PIPE_FIX (0.8 µs)
       ↓                                    CrossCoreWaitFlag(FINISH_MM_QCQR)
                                            DequantQc:
                                              MTE2 GM → UB → PIPE_V → MTE3 → GM
                                            CrossCoreSetFlag (0.8 µs)
                                            RopeQr:
                                              MTE2 GM → UB → PIPE_V → MTE3 → GM
                                            CrossCoreSetFlag (0.8 µs)
       wait FINISH_VEC_DEQUANT_QC ←────────┘
       ↓
MM4_Qn ──→ GM(mmQnResGm)
```

baseline 关键问题：
- **MM1/MM2 cube 时间被 HBM 带宽限制**：每核读取 weight (He × N_per_core × dtype) + token (M × He × dtype)。decode bs1 hd128 BF16 下 ~17.89 µs，是流水主体之一。
- **MM3 → MM4 段 vector 串行**：3 次 cross-core sync × 0.8 µs + 3 次 GM 往返 = ≥ 3 µs 浪费。FP8/INT8 量化下尤其突出。
- **现有全核 barrier (`WaitAllCore<ALL_VEC>` at lines 854/859/867)** 是 cube N-split / vector M-split 错配的副作用，本设计**保留不动**，与三项优化无关。

### 1.2 目标流水（三项优化全开）

```
AIC pipeline (cube)                         AIV pipeline (vector)
═══════════════════════════════════         ═══════════════════════════════════
[Split-KN]
MM1_Cq (BF16: n_groups=4 × k_groups=7 = 28 cores; FP8/INT8: n_groups=8 × k_groups=4 = 32 cores)
   BF16 每核 K=He/7=1024, N=Hcq/4=384; FP8/INT8 每核 K=He/4=1792, N=Hcq/8=192
   FixPipe → GM(mmCqPartialGm[kg])         ← 32 cores 写 4×8=32 个 partial 子区
   CrossCoreSetFlag PIPE_FIX (0.8 µs)
                                            CrossCoreWaitFlag(FINISH_MM_CQ)
                                            WaitAllCore<ALL_VEC>             ★ kept
                                            [Split-KN-Fused] AccumCq + RmsNormCq:
                                              for kg in 0..k_groups-1:
                                                MTE2 GM partial[kg] → UB partialBuf
                                                PIPE_V accumUbFp32 += partialBuf
                                              RmsNorm(rmsOut, accumUbFp32)
                                              MTE3 UB → GM(rmsNormCqResGm)
                                            ★ no new barrier vs naive accum
                                            WaitAllCore<ALL_VEC>             ★ kept
                                            CrossCoreSetFlag PIPE_MTE3
                                            ────────────────────────────
[Split-KN]
MM2_CkvKr (n_groups=1 × k_groups=28 cores)
   每核 K=He/28=256, N=576
   FixPipe → GM(mmCkvKrPartialGm[kg])
   CrossCoreSetFlag (0.8 µs)
                                            CrossCoreWaitFlag(FINISH_MM_CKVKR)
                                            WaitAllCore<ALL_VEC>             ★ kept
                                            [Split-KN-Fused] AccumCkvKr + RmsNormCkvKr + RopeKr + Scatter
                                              fused k_groups loads + reduce in UB
                                              → RmsNorm in UB
                                              → RoPE Kr in UB
                                              → Scatter to KV cache GM
       wait FINISH_RMS_CKVKR
       ↓
MM3_QcQr (split-N, baseline cube 配置)
   ──FixPipe──→ L0C  (NOT GM, intra-core handover)
   ───[DequantOpt]─────────────────────────────────────────────
   SetFlag<HardEvent::FIX_V> → intra-core to AIV
   ↓                                       l0c_to_ub<int32, half>(qcqrUb, l0cBuf, ...)
                                             with VDEQ16 inline dequant
                                           PIPE_V: Cast → BF16, RoPE on Qr region
                                           SetFlag<HardEvent::V_MTE3> → ub_to_l1
                                           ub_to_l1<half>(qcL1Buf, qcUb, ...)
                                           SetFlag<HardEvent::MTE3_M> → cube
   ↓
MM4_Qn ←──L1── (qcL1Buf intra-core)
   ──→ GM(mmQnResGm)
```

**节省汇总**：
- **MM1/MM2 cube** (Split-KN)：14–16 µs（每 MM ~6 µs cube 时间下降）
- **MM1/MM2 vector** (Split-KN-Fused)：~1.5 µs（避免 2 个新 barrier + 2 次 GM round-trip）
- **MM3→MM4** (DequantOpt)：~3.5 µs（消除 3 个 cross-core sync + 3 次 GM 往返；仅量化场景）

---

## 2. 三项优化的理论性能建模

### 2.1 优化 1：Split-KN（修正版）

**Theory**：在 cube 侧把 MM1 / MM2 的核分成 `n_groups × k_groups` 网格。N 维 split 增加并行核数，K 维 split 减少每核 HBM 加载量。**两者一起做** 能同时提升核利用率与每核带宽预算。

**修正前的 bug**（commit pending）：`estimate_matmul_split_kn` 把 `N = N_per_core × n_groups, active_cores = total_cores` 传给 `estimate_matmul_detailed`，导致 `per_core_N = N / active_cores = N_per_core / k_groups`。每核 cube 计算量、L0B 加载、FixPipe 字节都被低估了 `k_groups×`，让 K-split 看起来"白嫖"算力（hd8 best 假象 9.32 µs / 78.5%）。**修正**：采用 chip-total / contention 约定，传入 `N = N_per_core × total_cores, active_cores = total_cores`，per-core 带宽 = `chip_bw / total_cores`，反映多核竞争 HBM 带宽。这使得 32 核配置的 per-core 带宽比 28 核低 ~10%，导致 BF16 场景最优 MM1 从 N8×K4 切换到 N4×K7。

**FixPipe 压力建模**：当 `k_groups > 1`，cube 输出 dtype 为 FP32 (4 字节，不是 BF16 的 2 字节)；per-core FixPipe 字节 = `M × N_per_core × 4`。修正后 per_core_N 正确，per-core FixPipe 时间 k_groups× 大于错误版本，**正确反映 K-split 引入的搬出压力**。

**Vector accumulation 建模**：所有 vec_cores 个 V 核分摊 chip-total 累加工作（chip-total / contention 约定），总 load = `k_groups × M × N_full × 4`，compute = `M × N_full` 个元素 × `(k_groups − 1)` 条 Add 指令（每条 256 B/cycle，IPC=1.3），store = `M × N_full × out_size` 字节。

**Pipeline 掩盖**：`build_pipeline_dag` 把 `AccumCq` 作为 AIV 节点，`depends_on=["aiv_wait_CQ"]`（与 `MM2_CkvKr` 的 AIC 节点同时启动）。DAG 前向求解自然让 AccumCq 在 MM2 cube 内被掩盖。**只要 `accum_us < parallel AIC window`**（实测 9 个场景全满足），accum 不上关键路径。

**最优配置**（`tests/perf/report_split_kn.md` 第 4 节）：

| MM | n_groups | k_groups | 总核数 | per-core N | per-core K | 备注 |
|---|---:|---:|---:|---:|---:|---|
| MM1 (Cq) BF16 | 4 | 7  | 28 | Hcq/4 = 384  | He/7 = 1024 | 28 核避免 32 核 HBM 竞争 |
| MM1 (Cq) FP8/INT8 | 8 | 4 | 32 | Hcq/8 = 192 | He/4 = 1792 | FP8 weight 减半，32 核竞争不显著 |
| MM2 (CkvKr) | 1 | 28 | 28 | Hckv+Dr = 576 | He/28 = 256 | N=576 偏小，全部投到 K |

**理论 vs 实测**：

| 场景 | 修正前 best | 修正后 best |
|---|---|---|
| decode bs1 hd8 BF16  | MM1:N1×K28 + MM2:N3×K8 → 9.32 µs (78.5%, 假象)  | MM1:N4×K7 + MM2:N1×K28 → 27.23 µs (37.2%, 真实) |
| decode bs1 hd128 BF16 | MM1:N1×K28 + MM2:N3×K8 → 53.62 µs (38.8%) | MM1:N4×K7 + MM2:N1×K28 → 71.53 µs (18.4%) |

修正后的 best：BF16 场景为 `MM1:N4×K7 + MM2:N1×K28`（28 核，避免 32 核 HBM contention），FP8/INT8 场景仍用 `MM1:N8×K4 + MM2:N1×K28`（32 核，FP8 weight 减半使 HBM 竞争较不显著）。下一节给出 He 维度敏感度分析，验证这一最优在不同模型规模下的稳定性。

#### 2.1.1 He 维度敏感度分析（1k–8k 与 7680）

不同 He（hidden_size 输入维度）下的最优切分与收益。MM3/MM4 不依赖 He，所以**收益完全来自 MM1/MM2 cube 时间的压缩**。

**decode bs1 hd128 BF16 — 主流推理场景**

| He | Baseline (us) | Best Config | Opt pipeline (us) | 节省 (us) | 节省 % |
|---:|---:|---|---:|---:|---:|
| 1024 | 56.62 | MM1:N8×K4 + MM2:N3×K8  | 54.71 | −1.90 | −3.4% |
| 2048 | 61.39 | MM1:N8×K4 + MM2:N3×K8  | 57.57 | −3.82 | −6.2% |
| 4096 | 72.12 | MM1:N8×K4 + MM2:N1×K32 | 62.66 | −9.46 | −13.1% |
| **7168** | 87.63 | **MM1:N4×K7 + MM2:N1×K28** | **71.53** | **−16.10** | **−18.4%** |
| 7680 | 87.63 | MM1:N6×K5 + MM2:N3×K10 | 72.12 | −15.51 | −17.7% |
| 8192 | 97.17 | MM1:N8×K4 + MM2:N1×K32 | 73.47 | −23.70 | −24.4% |

**decode bs1 hd128 FP8 — 量化推理场景**

| He | Baseline (us) | Best Config | Opt pipeline (us) | 节省 (us) | 节省 % |
|---:|---:|---|---:|---:|---:|
| 1024 | 32.27 | MM1:N8×K4 + MM2:N9×K2  | 31.78 | −0.49 | −1.5% |
| 2048 | 34.65 | MM1:N8×K4 + MM2:N3×K8  | 32.76 | −1.89 | −5.5% |
| 4096 | 39.43 | MM1:N8×K4 + MM2:N3×K8  | 35.61 | −3.82 | −9.7% |
| **7168** | 48.97 | **MM1:N8×K4 + MM2:N1×K28** | **39.45** | **−9.52** | **−19.4%** |
| 7680 | 49.56 | MM1:N6×K5 + MM2:N3×K10 | 40.02 | −9.54 | −19.3% |
| 8192 | 50.16 | MM1:N8×K4 + MM2:N1×K32 | 40.70 | −9.46 | −18.9% |

**decode bs1 hd8 BF16 — 小规模 / 调试场景**（MM3 极小，split-kn 收益最显著）

| He | Baseline (us) | Best Config | Opt pipeline (us) | 节省 (us) | 节省 % |
|---:|---:|---|---:|---:|---:|
| 1024 | 12.32 | MM1:N8×K4 + MM2:N3×K8  | 10.42 | −1.90 | −15.5% |
| 2048 | 17.09 | MM1:N8×K4 + MM2:N3×K8  | 13.27 | −3.82 | −22.4% |
| 4096 | 27.83 | MM1:N8×K4 + MM2:N1×K32 | 18.36 | −9.46 | −34.0% |
| **7168** | 43.33 | **MM1:N4×K7 + MM2:N1×K28** | 27.23 | −16.10 | −37.2% |
| 7680 | 43.33 | MM1:N6×K5 + MM2:N3×K10 | 27.83 | −15.51 | −35.8% |
| 8192 | 52.87 | MM1:N8×K4 + MM2:N1×K32 | 29.18 | −23.70 | −44.8% |

#### 2.1.2 关键发现

1. **MM1 最优在 BF16 场景切换为 `n_groups=4, k_groups=7`（28 核），避免 32 核的 HBM contention；FP8/INT8 场景仍用 `n_groups=8, k_groups=4`（32 核，FP8 weight 减半使 HBM 竞争较不显著）**。

2. **MM2 最优 k_groups 随 He 的因数结构变化**：

   | He | 可用 k_groups (single_k = He / kg ≥ 128 且整除) | 选中的 k_groups |
   |---:|---|---:|
   | 1024 | 1, 2, 4, 8                 | 8 (single_k=128) |
   | 2048 | 1, 2, 4, 8, 16             | 8 (single_k=256) |
   | 4096 | 1, 2, 4, 8, 16, 32         | 32 (single_k=128) |
   | 7168 | 1, 2, 4, 7, 8, 14, 16, 28, 32, 56 | **28** (single_k=256) |
   | 7680 | 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60 | **10** (single_k=768)*  |
   | 8192 | 1, 2, 4, 8, 16, 32, 64     | **32** (single_k=256) |

   *He=7680 的 K10 选择不是因为 K10 性能更好，而是因为 7680 不能被 28 整除。perf model 在 (n_groups=3, k_groups=10) 与 (n_groups=1, k_groups=30) 两个配置上得到**相同的 pipeline 时间** (71.53 µs)，因为两者都打到了同一个 HBM-bound 地板。

3. **优化后 MM1+MM2 cube 时间的"地板"**：He ≥ 4096 后，最优 config 的 (MM1+MM2) cube 时间稳定在 **~19.7 µs** (= 14.31 + 5.37) — 这是单 AI core 在 1.6 TB/s chip-total HBM 与 32 cores 竞争下的 per-core 极限。**继续增大 He 不会让优化版本更慢**，但 baseline 会线性变长 → 收益随 He 单调增长。

4. **He=7168 与 He=7680 收益基本相同**：BF16 都是 16.10 µs / 18.4%；FP8 略有差异（9.52 vs 9.54 µs）且 < 1%。**模型从 7168 升级到 7680 不需要重新调优**。

5. **He=8192 收益跳升**：BF16 hd128 节省从 16.10 → 23.70 µs，因为 baseline 在 He=8192 时突破了一个 per-core 带宽上限点（baseline MM1 cube 从 17.89 → 21.47 µs，MM2 cube 从 17.89 → 23.86 µs），而最优版本的地板 ~19.7 µs 不变（slightly higher due to contention）。

6. **小 He 场景收益微弱**：He=1024 的 baseline 本身就只有 56 µs（hd128），MM1+MM2 加起来才 4.78 µs，留给 split-kn 的优化空间几乎没有。**He < 2048 不建议启用 split-kn**。

7. **量化场景的相对收益小于 BF16**：FP8/INT8 已经把 baseline 的 MM1/MM2 cube 时间砍到 BF16 的一半（每核 weight 加载减半），留给 split-kn 的余地也减半。FP8 hd128 He=7168 收益 −19.4% vs BF16 −18.4%（FP8 baseline 更小，相对 % 反而略大）；但**绝对节省**FP8 是 9.52 µs，BF16 是 16.10 µs。

#### 2.1.3 推荐切分（按 He 分档）

| He 范围 | 推荐 Best Config (BF16) | 推荐 Best Config (FP8/INT8) |
|---|---|---|
| He < 2048 | 不启用 split-kn | 不启用 split-kn |
| He = 2048 | MM1:N8×K4 + MM2:N3×K8 | MM1:N8×K4 + MM2:N3×K8 |
| He = 4096 | MM1:N8×K4 + MM2:N1×K32 | MM1:N8×K4 + MM2:N3×K8 |
| He = 7168 | **MM1:N4×K7 + MM2:N1×K28** | **MM1:N8×K4 + MM2:N1×K28** |
| He = 7680 | MM1:N6×K5 + MM2:N3×K10 | MM1:N6×K5 + MM2:N3×K10 |
| He = 8192 | MM1:N8×K4 + MM2:N1×K32 | MM1:N8×K4 + MM2:N1×K32 |

> **BF16 场景 He >= 7168 最优 MM1 从 N8×K4（32 核）切换为 N4×K7（28 核）**，因为 chip-total / contention 建模让 32 核配置的 per-core 带宽下降 ~10%。FP8/INT8 不受影响（weight 减半使 BW 需求远低于 cap）。MM2 选择逻辑不变。
> **MM2 切换**：根据 He 的可整除 k_groups 集合自动选择，host tiling 应实现一个查表函数 `select_mm2_split(He) → (n_groups, k_groups)`。

### 2.2 优化 2：Split-KN-Fused（K-accum + RmsNorm 融合）

**Theory**：split-kn 引入 K 累加后，naive 实现要求一个独立的 vector accum step：先把 k_groups 个 partial 累加成完整的 N 切片，写回 GM，再让 RmsNorm 从 GM 读这个完整的 N 切片。这引入：

1. **新的全核 barrier**：所有 k_groups cube cores 必须完成才能让 vector 开始累加。
2. **多余 GM 往返**：accum 输出 → GM → RmsNorm 输入 → UB（一次 UB → GM → UB 浪费）。

**Fused approach**：每个 V 核在它现有的 M 行范围内顺序读 k_groups 个 partial，UB 内 FP32 累加，立即在同一 UB 内跑 RmsNorm，最后写 GM 一次。**没有新 barrier**，**没有 GM 中转**。

**理论分析**（每 MM）：

| 项目 | Naive separate | Fused | 节省 |
|---|---|---|---|
| 新全核 barrier | 0.8 µs | 0 | 0.8 µs |
| accum store → GM → load 中转 | M × N_per_core × out_size × 2 / HBM | 0 | ~0 ~ 几 µs |
| RmsNorm input source | GM read | UB (free) | 几 µs |

实测每 MM ~0.7 µs 节省（`tests/perf/test_golden/decode_bs1_n8_split_kn_fused.txt`）。MM1 + MM2 合计 ~1.4 µs。

**Pipeline overlap**：fused step 整体放在 AIV 流水的 RmsNormCq 节点位置，仍由 `build_pipeline_dag` 处理与 AIC MM2/MM3 的并行掩盖——和 split-kn naive accum 共享同一个 DAG 处理路径。

### 2.3 优化 3：DequantOpt（L0C → UB → L1 intra-core 直通）

**Theory**：MM3 → MM4 段当前走 cube → GM → vector → GM → cube 流水：
- mmQcQr cube 输出 → FixPipe → GM (mmQcQrResGm)
- AIV 从 GM MTE2 加载 → DequantQc → MTE3 写 GM
- AIV 从 GM 加载 → RopeQr → MTE3 写 GM
- mmQn cube 从 GM MTE2 加载 Qc/Qr

3 个 cross-core sync × 0.8 µs = **2.4 µs**；3 次 GM 往返浪费带宽。

**Optimized**：让同一 AI core 内的 cube 与 vector 互为生产/消费者：
1. mmQcQr cube 写 L0C（不写 GM）
2. `l0c_to_ub<int32_t, half>` 把 L0C 直接搬到 UB，附带 `DeqScale::VDEQ16` inline dequant（INT32 → FP16，dequant 折叠进 MTE）
3. PIPE_V 内做 cast + RoPE
4. `ub_to_l1<half>` 把 UB 直接搬到 L1（mmQn 的 A 输入位置）
5. mmQn cube 直接从 L1 读，不经 GM

**hardware paths**（`mla_preprocess` 已用）：
- `attention/mla_preprocess/op_kernel/l0c_to_ub_iterator.h:45-67` — `l0c_to_ub<int32_t, half>` 含 VDEQ16
- `attention/mla_preprocess/op_kernel/l1_to_ub_iterator.h:30-41` — `ub_to_l1<half>`
- `hardware.py:167-168` — `l0c_to_out_per_cycle = 256` (422 GB/s), `ub_to_l1_per_cycle = 245` (404 GB/s)

**节省**：
- 3 cross-core sync (2.4 µs) → 3 intra-core HardEvent (~0)
- 3 次 GM round-trip → 0 次

实测 hd128 FP8 decode 节省 ~3.5 µs（`tests/perf/test_golden/decode_bs1_fp8_dequant_opt.txt`）。

---

## 3. 流水设计 — 改动详图

### 3.1 Split-KN cube 切分（MM1 / MM2）

```
Baseline (split-N only):
  AIC layout: 24 cores split N=Hcq into 24 × 64 columns
  Each core does full K=7168, partial N=64
  Output: BF16 → GM[24 × M × 64]

Split-KN (n_groups × k_groups):
  AIC layout: n_groups columns × k_groups rows = total_cores
  Each (kg, ng) core does K=He/k_groups, N=Hcq/n_groups
  Output: FP32 partial → GM[k_groups × M × N_per_core] per n-group

  For MM1 (BF16): 4 × 7 = 28 cores, partial size = M × 384 × 4 bytes per core
  For MM1 (FP8/INT8): 8 × 4 = 32 cores, partial size = M × 192 × 4 bytes per core
  For MM2: 1 × 28 = 28 cores, partial size = M × 576 × 4 bytes per core
```

**改动**：
1. Host tiling 增加 `mm1_n_groups`, `mm1_k_groups`, `mm2_n_groups`, `mm2_k_groups` 字段（默认 1×n_groups 即原 split-N，BF16 设为最优值 4×7 / 1×28、FP8/INT8 设为 8×4 / 1×28 启用 split-KN）
2. Host workspace 增加 partial 子区：
   - `mmCqPartialGm`: `k_groups × M × N_per_core × 4 bytes` per n-group
   - `mmCkvKrPartialGm`: `k_groups × M × N_per_core × 4 bytes` per n-group
3. Cube kernel 加新分发函数 `MatmulSplitKN` 替代 `MatmulSplitN`，按 (kg, ng) 计算 K/N range，输出 FP32 至 partial 子区

### 3.2 Split-KN-Fused vector 累加（替代 naive accum + rmsnorm）

```
Baseline (naive Accum + RmsNorm):
  AIV: WaitAllCore<ALL_VEC>(NEW_BARRIER)        ← would-be new barrier
       for kg in 0..k_groups-1:
         MTE2 GM partial[kg] → UB partialBuf[kg]
       PIPE_V: reduce 8 buffers → accumUb
       MTE3: accumUb → GM(reducedRes)            ← UB → GM 浪费
       WaitAllCore<ALL_VEC>(SECOND_BARRIER)
       MTE2: GM(reducedRes) → UB rmsInBuf       ← GM → UB 浪费
       PIPE_V: RmsNorm(rmsOutBuf, rmsInBuf)
       MTE3: rmsOutBuf → GM(rmsNormCqResGm)

Fused (Split-KN-Fused):
  AIV: (no new barrier — fused into existing M-split RmsNorm step)
       for c in 0..num_chunks-1:
         Duplicate(accumUbFp32, 0.0f, ...)
         for kg in 0..k_groups-1:
           MTE2: GM partial[kg] for my M rows → UB partialBuf[ping]
           PIPE_V: accumUbFp32 += partialBuf[ping]   ← UB-resident accum
         RmsNormLocal(rmsOutBuf, accumUbFp32, ...)   ← reads UB directly
         MTE3: rmsOutBuf → GM(rmsNormCqResGm)
```

**改动**：
1. `service_rms_norm.h`：加 `RmsNormLocal` 重载，输入从 `GlobalTensor` 改为 `LocalTensor` (UB)
2. AIV 主循环：当 `enableSplitKnFusedAccum && k_groups > 1`，把 `RmsNormCq` 调用替换成新的 `AivProcess_RmsNormCq_FusedAccum`
3. UB layout 增加 FP32 accumulator 区（详见 §5）

### 3.3 DequantOpt MM3 → MM4 段（intra-core L0C→UB→L1）

```
Baseline:
  AIC: mmQcQr.IterateAll → FixPipe → mmQcQrResGm (GM)
       CrossCoreSetFlag PIPE_FIX(QCQR)         ← 0.8 µs
  AIV: CrossCoreWaitFlag(QCQR)
       MTE2 GM → UB → DequantQc → MTE3 GM
       CrossCoreSetFlag(DEQUANT_QC)             ← 0.8 µs
       MTE2 GM → UB → RopeQr → MTE3 GM
       CrossCoreSetFlag(...)                   ← 0.8 µs
  AIC: CrossCoreWaitFlag(DEQUANT_QC)
       mmQn from GM

DequantOpt:
  AIC: mmQcQr.IterateAll → L0C (NOT GM)
       SetFlag<HardEvent::FIX_V>               ← intra-core, ~0
  AIV: WaitFlag<HardEvent::FIX_V>
       l0c_to_ub<int32, half>(qcqrUb, l0cBuf, VDEQ16)
       PIPE_V: Cast/RoPE
       SetFlag<HardEvent::V_MTE3>              ← intra-core, ~0
       ub_to_l1<half>(qcL1Buf, qcUb)
       SetFlag<HardEvent::MTE3_M>              ← intra-core, ~0
  AIC: WaitFlag<HardEvent::MTE3_M>
       mmQn.SetTensorA(qcL1Buf)
       mmQn.IterateAll → L0C → FixPipe → mmQnResGm
```

**改动**：
1. `mla_prolog_v3.cpp`：新增 `EnableDequantOpt` template 参数（已存在但未连线）
2. `kernel_mla_prolog_split_n.h`：新增 `MmQcQrDequantRopeFused` 函数替换 `MatmulAndSyncQcQr` + `DequantAndRopeSyncMMQcQr` 调用
3. `service_dequant.h`：保留 baseline `DequantPerTokenQc`，新增 UB 直消费版本 `DequantQcInUb`
4. UB layout 增加 qcqrUb / qcUb / qrUb 区（详见 §5）
5. L1 layout 增加 qcL1Buf / qrL1Buf 区（详见 §4）

---

## 4. GM / Workspace 内存规划

### 4.1 当前 workspace 布局

引用 `mla_prolog_tiling.cpp:517-556`，按顺序：

```
workspaceSize_ = libapiSize_ + …
+ stepBatchSize × Hcq × NUM_BYTES_BF16 × 2          (Cq 中间缓冲)
+ stepBatchSize × (Hckv+Dr) × NUM_BYTES_BF16        (CkvKr 中间缓冲)
+ stepBatchSize × (headSizeQc+headSizeQr) × NUM_BYTES_INT32  (mmQcQr buffer)
+ stepBatchSize × N × D × NUM_BYTES_BF16            (mmQn 输入暂存)
[FP8/INT8 加：dequant scale buffers]
```

### 4.2 优化后 workspace 布局

```
workspaceSize_v2 = libapiSize_
  ─── Split-KN partial buffers (新增) ─────────────────────────────────
  + mm1_partial_size = M × Hcq × 4 × mm1_k_groups   ← FP32 partials for MM1
  + mm2_partial_size = M × (Hckv+Dr) × 4 × mm2_k_groups ← FP32 partials for MM2
  ─── 现有中间缓冲（部分可省，详见下表） ───────────────────────────────
  + Cq 中间缓冲                                     ← 仍保留 (RmsNormCq 输出去这里给 MM3 读)
  + CkvKr 中间缓冲                                  ← 仍保留 (final scatter destination 之一)
  + mmQcQr buffer                                   ★ 可省 (DequantOpt 走 L0C→UB→L1)
  + mmQn 输入暂存                                   ★ 可省 (DequantOpt 走 L1)
  ─── DequantOpt 不需要新增 GM (走 UB/L1) ─────────────────────────────
  [FP8/INT8 dequant scale buffers 不变]
```

**对比**：

| 区段 | Baseline 大小 (decode bs1 stepBS=128) | Optimized 大小 | 备注 |
|---|---:|---:|---|
| Cq mm1 中间缓冲 | 128 × 1536 × 2 × 2 = 768 KB | 768 KB | 不变 (RmsNormCq 输出仍写 GM) |
| MM1 K-partials (新增) | — | 128 × 1536 × 4 × 7 = **5.25 MB** (BF16, k=7) / 128 × 1536 × 4 × 4 = **3 MB** (FP8/INT8, k=4) | mm1_k_groups 视 dtype |
| CkvKr mm2 中间缓冲 | 128 × 576 × 2 = 144 KB | 144 KB | 不变 |
| MM2 K-partials (新增) | — | 128 × 576 × 4 × 28 = **8 MB** | mm2_k_groups=28，FP32 |
| mmQcQrRes buffer | 128 × 4096 × 4 = 2 MB | **0** | DequantOpt 省去 |
| mmQnInput buffer | 128 × 128 × 128 × 2 = 4 MB | **0** | DequantOpt 省去 |
| **合计变化** | ~6.9 MB | ~12 MB | **+5 MB workspace** (主要 K-partials) |

**关键点**：
- **K-partial 区是新增的最大开销**：MM2 partial 占 8 MB（k_groups=28，每 partial 较小但份数多）。Decode bs1 stepBS=128 可接受，**prefill stepBS 大时需评估**。
- **DequantOpt 省去 mmQcQr / mmQn 中间 GM 区**：节省 ~6 MB（~MB 量级，对总 workspace 净影响小）
- 新 partial 区位于 L2 容量内（Ascend 950 L2 ~192 MB），**不会引发 HBM 颠簸**

### 4.3 Partial 子区索引

```cpp
// MM1 K-partials, layout [k_group_idx][M, N_per_core]
__gm__ float* mmCqPartialGm[k_groups];

inline int64_t Mm1PartialOffset(int kg, int row, int col) {
  return kg * M_full * N_per_core
       + row * N_per_core
       + col;
}

// 类似的 MM2 K-partials
__gm__ float* mmCkvKrPartialGm[k_groups];
```

V 核读取自己 M 行的 partial：
```cpp
DataCopyPad(
    partialUb_,
    mmCqPartialGm[kg][my_m_offset * N_per_core],   // 自己的 M 切片
    /*nBurst=*/per_v_M,
    /*lenBurst=*/N_per_core * sizeof(float),
    /*srcGap=*/0, /*dstGap=*/0);
```

---

## 5. UB 内存规划（每 V 核 256 KB / 可用 ~220 KB）

### 5.1 静态 buffer（始终占用）— ~34 KB

引用 `kernel_mla_prolog_split_n.h:510-578`：

```
dequantScaleWDqBuffer_       : 1536 × 4 = 6 KB
rmsnormGammaCqBuffer_        : 1536 × 2 = 3 KB
rmsnormGammaCkvBuffer_       :  512 × 2 = 1 KB
dequantScaleWDkvKrBuffer_    :  576 × 4 = 2.3 KB
sincosBuffer_                : 2 × 64 × stepBS × 4 = 16 KB (stepBS=32)
smoothScaleCqBuffer_         : 1536 × 4 = 6 KB (dynamic quant only)
quantScale*Buffer_           : ~2.3 KB
stepActualSeqBuffer_         : stepBS × 8 = 256 B
─────────────────────────────────────────
合计静态                     : ~34 KB
```

### 5.2 Split-KN-Fused 段 UB 布局（per V 核，每个 chunk）

复用 share buffer 区，按 chunk 切分：

```
chunk_m × N_full × 20 bytes per chunk:
  partialUb_[2]            : 2 × chunk_m × N_full × 4   (FP32 partial input, double-buffered)
  accumUbFp32_             :     chunk_m × N_full × 4   (FP32 accumulator)
  rmsNormWorkUb_           :     chunk_m × N_full × 4   (FP32 RmsNorm working)
  rmsOutputUb_[2]          : 2 × chunk_m × N_full × 2   (BF16 output, double-buffered)
```

**实测 UB 占用**（来自 `tests/perf/perf/model/split_kn_fused.py` UB budget）：

| 配置 | per_v_M | chunks | per_v_UB | / 220 KB |
|---|---:|---:|---:|---:|
| decode bs1 hd128 | 1 | 1 | ~30 KB | 13.5% |
| decode bs8 hd128 | 1 | 1 | ~30 KB | 13.5% |
| prefill T=128 hd128 | 2 | 1 | ~60 KB | 27.3% |
| 假设 stepBS=512 | 8 | 2 | ~210 KB | 95% (chunked) |

### 5.3 DequantOpt 段 UB 布局（per AI core）

```
qcqrUb_[2]               : 2 × M × per_core_n × in_size  (L0C→UB 落点，含 VDEQ16 后已是 FP16)
dequant_scale_buffer     : per_core_n × 4                 (per-N quant scale)
fp32WorkUb_              : M × per_core_n × 4             (FP32 中间)
bf16CastUb_              : M × per_core_n × 2             (BF16 cast 输出)
sin/cos                  : 2 × Dr × 2                     (RoPE 系数)
ropeTempUb_              : Dr × 2                         (RoPE 临时)
qcUb_                    : M × per_core_qc × 2            (qcUb，UB→L1 source)
qrUb_                    : M × per_core_qr × 2            (qrUb，UB→L1 source)
```

**实测**（`tests/perf/test_golden/decode_bs1_fp8_dequant_opt.txt`）：

| 配置 | UB used | / 220 KB |
|---|---:|---:|
| FP8 decode bs1 hd128 | 17.3 KB | 7.9% |
| INT8 decode bs1 hd128 | 17.3 KB | 7.9% |
| INT8 decode bs8 hd128 | 114 KB | 51.8% |

### 5.4 三段共存 UB 总占用

**关键洞察**：split-kn-fused 与 dequant-opt **不会同时活跃**——它们在不同时间段使用 UB：
- Split-kn-fused 段：MM1/MM2 → RmsNorm 阶段
- DequantOpt 段：MM3 → MM4 阶段

两段可以**复用同一片 UB share buffer**（依次清零并重用），无需同时分配。最大 UB 占用 = max(split-kn-fused, dequant-opt) + 静态 ~34 KB。

| 场景 | 静态 | Split-KN-Fused 段 | DequantOpt 段 | 复用后总占用 |
|---|---:|---:|---:|---:|
| decode bs1 hd128 BF16 | 34 KB | 30 KB | n/a | **64 KB** (24%) |
| decode bs1 hd128 FP8 | 34 KB | 30 KB | 17 KB | **64 KB** (24%) |
| prefill T=128 hd128 | 34 KB | 60 KB | (BF16 不适用) | **94 KB** (36%) |

UB 充裕，无溢出风险。

---

## 6. L1 内存规划（每 AI core 512 KB）

baseline L1 主要用于 cube 的 weight buffer（B 矩阵双缓冲）。DequantOpt 新增的 L1 区域：

```
qcL1Buf_[2]              : 2 × M × N_qc × 2     (BF16 Qc，UB→L1 destination, 双缓冲)
qrL1Buf_[2]              : 2 × M × N_qr × 2     (BF16 Qr，UB→L1 destination)
```

每核 L1 占用：M × (N_qc + N_qr) × 4 bytes（双缓冲）。FP8 decode bs1 hd128 ≈ 6 KB，远低于 512 KB 的 weight buffer 需求，无冲突。

引用 `tests/perf/test_golden/decode_bs1_fp8_dequant_opt.txt`：

| 配置 | L1 used (DequantOpt) | / 512 KB |
|---|---:|---:|
| FP8 decode bs1 hd128 | 3 KB | 0.6% |
| INT8 decode bs8 hd128 | 24.6 KB | 4.7% |

---

## 7. Tiling 数据结构改动

`mla_prolog_tiling_data.h` 新增字段：

```cpp
struct MlaPrologBaseParams {
    // 现有字段保持不变
    uint32_t stepBatchSize;
    uint32_t headSizeCq;
    uint32_t headSizeCkv;
    uint32_t headSizeKr;
    uint32_t mm1BlockNum;        // 现有 split-N 兼容字段
    uint32_t mm1SingleCoreN;
    uint32_t mm2BlockNum;
    uint32_t mm2SingleCoreN;
    uint32_t mm3BlockNum;
    uint32_t mm4BlockNum;
    bool enableGroupComputeOpt;
    bool enableDequantOpt;       // 已存在的 template 参数

    // === 新增：Split-KN 字段 ===
    uint32_t mm1NGroups;         // MM1 N-axis split count (default = mm1BlockNum, KN: e.g. 8)
    uint32_t mm1KGroups;         // MM1 K-axis split count (default = 1, KN: e.g. 4)
    uint32_t mm1SingleCoreK;     // = He / mm1KGroups
    uint32_t mm2NGroups;         // MM2 N-axis split count (default = mm2BlockNum, KN: e.g. 1)
    uint32_t mm2KGroups;         // MM2 K-axis split count (default = 1, KN: e.g. 28)
    uint32_t mm2SingleCoreK;     // = He / mm2KGroups

    // === 新增：Workspace partial offsets (Split-KN) ===
    uint64_t mmCqPartialOffset;     // FP32 K-partials for MM1
    uint64_t mm1PartialSize;        // = mm1KGroups × M × mm1SingleCoreN × 4
    uint64_t mmCkvKrPartialOffset;  // FP32 K-partials for MM2
    uint64_t mm2PartialSize;        // = mm2KGroups × M × mm2SingleCoreN × 4

    // === 新增：feature flags (template params 通过 tilingKey 选路) ===
    bool enableSplitKn;             // 是否启用 Split-KN cube + Split-KN-Fused vector
    bool enableSplitKnFusedAccum;   // = enableSplitKn (绑定开启)
    // enableDequantOpt 已存在
};
```

`mla_prolog_tiling.cpp` 的 `Tiling()` 函数新增：

```cpp
// === Split-KN tile selection ===
if (enableSplitKn) {
    // Recommended config from perf model report
    // BF16: N4×K7 (28 cores) avoids 32-core HBM contention
    // FP8/INT8: N8×K4 (32 cores), weight halved makes contention less impactful
    if (isBF16) {
        cfg.mm1NGroups = 4;
        cfg.mm1KGroups = 7;
        cfg.mm1SingleCoreN = ceil_div(headSizeCq, cfg.mm1NGroups);  // 384
        cfg.mm1SingleCoreK = ceil_div(headSizeX, cfg.mm1KGroups);   // 1024
        cfg.mm1BlockNum = cfg.mm1NGroups * cfg.mm1KGroups;          // 28 cores
    } else {
        cfg.mm1NGroups = 8;
        cfg.mm1KGroups = 4;
        cfg.mm1SingleCoreN = ceil_div(headSizeCq, cfg.mm1NGroups);  // 192
        cfg.mm1SingleCoreK = ceil_div(headSizeX, cfg.mm1KGroups);   // 1792
        cfg.mm1BlockNum = cfg.mm1NGroups * cfg.mm1KGroups;          // 32 cores
    }

    cfg.mm2NGroups = 1;
    cfg.mm2KGroups = 28;
    cfg.mm2SingleCoreN = headSizeCkv + headSizeKr;              // 576
    cfg.mm2SingleCoreK = ceil_div(headSizeX, cfg.mm2KGroups);   // 256
    cfg.mm2BlockNum = cfg.mm2NGroups * cfg.mm2KGroups;          // 28 cores

    // Workspace partials
    cfg.mm1PartialSize = (uint64_t)cfg.mm1KGroups * stepBatchSize *
                         cfg.mm1SingleCoreN * sizeof(float);
    cfg.mm2PartialSize = (uint64_t)cfg.mm2KGroups * stepBatchSize *
                         cfg.mm2SingleCoreN * sizeof(float);

    workspaceSize += cfg.mm1PartialSize + cfg.mm2PartialSize;
}
```

---

## 8. Kernel 改动 — 伪代码

### 8.1 顶层 dispatch（`mla_prolog_v3.cpp`）

```cpp
template <
    typename T, typename DequantScaleType,
    bool EnableSplitKn,                     // ← 新增
    bool EnableSplitKnFusedAccum,            // ← 新增 (绑定 EnableSplitKn)
    bool EnableDequantOpt,                   // 已存在但未连线
    bool EnableGroupComputeOpt,
    uint8_t EmptyTensorMode,
    /* 其他 template params */
>
class MlaPrologV3Kernel { ... };

// dispatch 表加 4 条新路径（笛卡尔积裁剪 — 只对量化场景启用 dequant_opt）：
// (Split=N, Fused=N, Deq=N) — 原 baseline
// (Split=Y, Fused=Y, Deq=N) — Split-KN + Fused, BF16
// (Split=Y, Fused=Y, Deq=Y) — 三优化全开, FP8/INT8/MXFP8
// (Split=N, Fused=N, Deq=Y) — 仅 DequantOpt, FP8/INT8 (split-kn 收益较小时备选)
```

### 8.2 MM1 / MM2 cube — Split-KN（`kernel_mla_prolog_split_n.h::AicProcess`）

```cpp
template<typename MLAPT>
template<bool needQnDynamicQuant>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::AicProcess(
    AicOffset &aicOffset, int64_t batchOffset, int64_t mmQnLoops)
{
    int64_t tokenXOffset = batchOffset * baseParams_->headSizeX;

    // === MM1 (Cq) ===
    if constexpr (MLAPT::EnableSplitKn) {
        // 当前 cube 核的 (n_group_idx, k_group_idx)
        const uint32_t kg = cubeBlockIdx_ / baseParams_->mm1NGroups;
        const uint32_t ng = cubeBlockIdx_ % baseParams_->mm1NGroups;
        const int64_t k_offset = kg * baseParams_->mm1SingleCoreK;
        const int64_t n_offset = ng * baseParams_->mm1SingleCoreN;

        // tokenX[M, K_per_core], weight[K_per_core, N_per_core]
        // 输出 FP32 partial 至 mmCqPartialGm[kg][M, N_per_core]
        MatmulSplitKN<mmInputType, /*OutputType=*/float>(
            mmCqPartialGm_[kg * mm1PartialStride + n_offset * stepBS],
            tokenXGm_[tokenXOffset + k_offset],         // K 切片
            weightDqGm_[k_offset * Hcq + n_offset],     // (K_per_core × N_per_core)
            mmCqParam_,
            UsedBlockParams{kg * mm1NGroups + ng, baseParams_->mm1BlockNum});
    } else {
        // 原 split-N path
        MatmulSplitN<mmInputType, mmCqOutputType, dequantScaleType>(
            mmCqResGm_[aicOffset.cqResOffset], ...);
    }
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CQ);

    // === MM2 (CkvKr) ===
    if constexpr (MLAPT::EnableSplitKn) {
        const uint32_t kg = cubeBlockIdx_ / baseParams_->mm2NGroups;
        const uint32_t ng = cubeBlockIdx_ % baseParams_->mm2NGroups;
        const int64_t k_offset = kg * baseParams_->mm2SingleCoreK;
        const int64_t n_offset = ng * baseParams_->mm2SingleCoreN;

        if (cubeBlockIdx_ < baseParams_->mm2BlockNum) {  // 28 cores active
            MatmulSplitKN<mmInputType, /*OutputType=*/float>(
                mmCkvKrPartialGm_[kg * mm2PartialStride + n_offset * stepBS],
                tokenXGm_[tokenXOffset + k_offset],
                weightDkvKrGm_[k_offset * (Hckv+Dr) + n_offset],
                mmCkvKrParam_,
                UsedBlockParams{kg * mm2NGroups + ng, baseParams_->mm2BlockNum});
        }
    } else {
        MatmulSplitN<mmInputType, mmCkvKrOutputType, dequantScaleType, true>(...);
    }
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CKVKR);

    CrossCoreWaitFlag(FINISH_VEC_RMSNORM_CQ);

    // === MM3 (QcQr) — DequantOpt 替换 MatmulAndSyncQcQr ===
    if constexpr (MLAPT::EnableDequantOpt) {
        MmQcQrDequantRopeFused(aicOffset, batchOffset, mmQnLoops);
        // ↑ 这个函数封装了 cube + intra-core handover + vector + L1 → cube + MM4
    } else {
        if constexpr (std::is_same<mmInputType, FP8E4M3>::value) {
            MatmulQcQr(aicOffset);
        } else {
            MatmulAndSyncQcQr(aicOffset);
        }
        // ... 原 path 走完 MM4 ...
        PreloadQnAndSync(aicOffset, mmQnLoops);
        MatmulQnSyncDynamicQuantAndMulQr<needQnDynamicQuant>(...);
    }
}
```

### 8.3 MM1/MM2 vector — Split-KN-Fused（`kernel_mla_prolog_split_n.h::AivProcess`）

```cpp
template<typename MLAPT>
template<bool needQnDynamicQuant>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::AivProcess(
    AivOffset &aivOffset, int64_t batchOffset, int64_t curStepBatchSize, ...)
{
    int64_t tokenIndex = batchOffset + aivOffset.curBlockTokenOffset;

    if (batchOffset == 0) { CopyGlobalParams(); }
    CopyInSinCos(tokenIndex, aivOffset.curVecToken, batchOffset, curStepBatchSize);

    CrossCoreWaitFlag(FINISH_MM_CQ);
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);   // 现有 barrier，保留

    // === Fused K-accum + RmsNormCq (替换原 RmsNormCq 调用) ===
    if constexpr (MLAPT::EnableSplitKnFusedAccum) {
        AivProcess_RmsNormCq_FusedAccum(
            batchOffset, curStepBatchSize, baseParams_->mm1KGroups);
    } else {
        // 原 path
        RmsNormCq(tokenIndex, aivOffset.rmsNormCqOffset, ...);
    }

    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);   // 现有 barrier，保留
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_RMSNORM_CQ);

    CrossCoreWaitFlag(FINISH_MM_CKVKR);
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);   // 现有 barrier，保留

    // === Fused K-accum + RmsNormCkvKr+RopeKr+Scatter ===
    if constexpr (MLAPT::EnableSplitKnFusedAccum) {
        AivProcess_RmsNormCkvKrRopeScatter_FusedAccum(
            batchOffset, curStepBatchSize, baseParams_->mm2KGroups);
    } else {
        RmsNormRopeScatterCkvKr(tokenIndex, ...);
    }

    // === MM3 → MM4 vector segment ===
    // 当 EnableDequantOpt = true 时，AIC 端走 MmQcQrDequantRopeFused，
    // AIV 端不需要做 DequantQc/RopeQr（这些工作转到 AIC 同核的 vector 内）。
    // 否则继续走原 vector path。
    if constexpr (!MLAPT::EnableDequantOpt) {
        if constexpr (MLAPT::enableGroupComputeOpt) {
            // ... existing dequant + rope groups path ...
        } else if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
            CrossCoreWaitFlag(FINISH_MM_QCQR);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
            DequantQc(...);
            // ... etc ...
        } else {
            // ... etc ...
        }
    }
    // (DequantOpt 启用时这一段全部由 AIC 端的 MmQcQrDequantRopeFused 处理)
}
```

### 8.4 Split-KN-Fused 内层函数（新增）

```cpp
// 引用 Phase 2 of MlaPrologV3_SplitKN_FusedAccum设计.md
template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::AivProcess_RmsNormCq_FusedAccum(
    int64_t batchOffset, int64_t curStepBatchSize, uint32_t k_groups)
{
    const int64_t my_m_offset = aivOffset.curBlockTokenOffset;   // 现有 M-split offset
    const uint32_t my_m_full = aivOffset.curVecToken;            // 现有 per_v_M

    // UB chunking when M is too big to fit
    const uint32_t chunk_m = ComputeChunkM_Mm1(my_m_full);       // tiling 给定阈值
    const uint32_t num_chunks = ceil_div(my_m_full, chunk_m);

    for (uint32_t c = 0; c < num_chunks; ++c) {
        const uint32_t cur_m = min(chunk_m, my_m_full - c * chunk_m);
        const int64_t row_offset = batchOffset + my_m_offset + c * chunk_m;

        // 1. FP32 累加器清零
        Duplicate(accumUbFp32_, 0.0f, cur_m * baseParams_->headSizeCq);

        // 2. 顺序读 k_groups 个 partial，UB 内累加
        for (uint32_t kg = 0; kg < k_groups; ++kg) {
            const uint32_t ping = kg & 1;

            // MTE2: 自己 M 行的 K-partial[kg] → UB
            DataCopyPad(
                partialUb_[ping],
                mmCqPartialGm_[kg * mm1PartialStride + row_offset * baseParams_->headSizeCq],
                /*nBurst=*/cur_m,
                /*lenBurst=*/baseParams_->headSizeCq * sizeof(float),
                /*srcGap=*/0, /*dstGap=*/0);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + ping);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0 + ping);

            // PIPE_V: accumulator += partial
            Add(accumUbFp32_, accumUbFp32_, partialUb_[ping],
                cur_m * baseParams_->headSizeCq);
            PipeBarrier<PIPE_V>();
        }

        // 3. RmsNorm 直接读 UB 内的累加结果（不经 GM）
        RmsNormLocal(
            rmsOutputUb_,
            accumUbFp32_,
            rmsnormGammaCqUb_,
            rmsNormParams_,
            cur_m,
            baseParams_->headSizeCq);
        PipeBarrier<PIPE_V>();

        // 4. UB → GM (final destination, MM3 will read this)
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopyPad(
            rmsNormCqResGm_[row_offset * baseParams_->headSizeCq],
            rmsOutputUb_,
            cur_m,
            baseParams_->headSizeCq * sizeof(rmsNormCqOutputType));
    }
    // 注意：没有插入新的 WaitAllCore；
    // 现有的 line 854/859 barriers 仍在外层调用处。
}
```

`AivProcess_RmsNormCkvKrRopeScatter_FusedAccum` 模式相同，只是 N_full = Hckv+Dr，并在 RmsNorm 之后接 RoPE Kr + Scatter。

### 8.5 DequantOpt 内层函数（新增）

```cpp
// 引用 §7 of MlaPrologV3_DequantOpt设计.md
template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmQcQrDequantRopeFused(
    AicOffset &aicOffset, int64_t batchOffset, int64_t mmQnLoops)
{
    constexpr uint32_t UB_HALF = TOTAL_UB_PER_PINGPONG;
    constexpr uint32_t L1_HALF = TOTAL_L1_PER_PINGPONG;

    const uint32_t my_m = mmQcQrParam_.m;
    const uint32_t per_core_n = mm3SingleCoreN;
    const uint32_t qcLen = my_m * (per_core_n - Dr_per_head);  // Qc 区
    const uint32_t qrLen = my_m * Dr_per_head;                  // Qr 区

    for (uint32_t step = 0; step < stepNum_; ++step) {
        const uint32_t ubPing = step & 1;
        const uint32_t l1Ping = step & 1;

        // === 1. Cube: mmQcQr 写 L0C（不写 GM） ===
        mmQcQr_.IterateAll<false>(/*L0C dst*/, /*Atomic=*/false);
        mmQcQr_.End();

        // === 2. intra-core: cube → vector handover ===
        SetFlag<HardEvent::FIX_V>(EVENT_ID0 + ubPing);
        WaitFlag<HardEvent::FIX_V>(EVENT_ID0 + ubPing);

        // === 3. L0C → UB direct，VDEQ16 inline dequant ===
        // 引用 mla_preprocess/op_kernel/l0c_to_ub_iterator.h:45
        l0c_to_ub<ArchType::ASCEND_950, int32_t, half>(
            qcqrUb_[ubPing * UB_HALF],
            l0cBuf_,
            /*nBurst=*/my_m,
            /*lenBurst=*/per_core_n * sizeof(int32_t) / 32,
            /*srcStride=*/0, /*dstStride=*/0);

        // === 4. UB 上 cast + RoPE ===
        Cast(qcUb_, qcqrUb_[ubPing * UB_HALF],
             RoundMode::CAST_RINT, qcLen);
        RopeApply(qrUb_,
                  qcqrUb_[ubPing * UB_HALF + qcLen],
                  sinUb_, cosUb_,
                  /*headDim=*/Dr_per_head);
        PipeBarrier<PIPE_V>();

        // === 5. intra-core: vector → MTE3 handover ===
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2 + ubPing);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2 + ubPing);

        // === 6. UB → L1 direct（无 GM 中转） ===
        // 引用 l1_to_ub_iterator.h:30
        ub_to_l1<ArchType::ASCEND_950, half>(
            qcL1Buf_[l1Ping * L1_HALF],
            qcUb_,
            /*nBurst=*/my_m,
            /*lenBurst=*/(qcLen + qrLen) * sizeof(half) / 32,
            /*srcStride=*/0, /*dstStride=*/0);

        // === 7. intra-core: MTE3 → cube handover ===
        SetFlag<HardEvent::MTE3_M>(EVENT_ID4 + l1Ping);
        WaitFlag<HardEvent::MTE3_M>(EVENT_ID4 + l1Ping);

        // === 8. Cube: mmQn 直接从 L1 读 Qc/Qr ===
        mmQn_.SetTensorA(qcL1Buf_[l1Ping * L1_HALF]);
        mmQn_.IterateAll<false>(mmQnOutGm_[aicOffset.qnResOffset], /*Atomic=*/false);
        mmQn_.End();

        // 推进偏移
        aicOffset.qnResOffset += stepBatchSize_ * Hckv * numHeadSize;
    }
}
```

---

## 9. 改动文件清单

### Host 侧

| 文件 | 改动 |
|---|---|
| `op_host/mla_prolog_v3_tiling.cpp` | 新增 `enableSplitKn` 决策逻辑、最优 (n_groups, k_groups) 选择、partial workspace 大小计算 |
| `op_host/mla_prolog_v3_tiling.h` | 新增 partial offset / size 字段，新增 `mm1NGroups` / `mm1KGroups` / `mm2NGroups` / `mm2KGroups` |
| `op_kernel/mla_prolog_tiling_data.h` | 上述 BaseParams 字段同步 |

### Kernel 侧

| 文件 | 改动 |
|---|---|
| `op_kernel/mla_prolog_v3.cpp` | 新增 `EnableSplitKn` / `EnableSplitKnFusedAccum` template 参数；连线 `EnableDequantOpt`；扩 dispatch 表 |
| `op_kernel/kernel_mla_prolog_split_n.h` | 改 `AicProcess` 加 split-kn 路径；改 `AivProcess` 加 fused-accum 路径；加 `MmQcQrDequantRopeFused` 函数 |
| `op_kernel/service_matmul.h` | 新增 `MatmulSplitKN` 模板（输出 FP32 partial 到 GM 子区） |
| `op_kernel/service_rms_norm.h` | 新增 `RmsNormLocal` 重载，输入接受 `LocalTensor` (UB) |
| `op_kernel/service_dequant.h` | 保留 baseline `DequantPerTokenQc`；新增 UB 直消费版本 `DequantQcInUb`（VDEQ16 不适用时回退） |
| `op_kernel/l0c_to_ub_iterator.h` | **不动**，引用 `attention/mla_preprocess/op_kernel/l0c_to_ub_iterator.h` |
| `op_kernel/l1_to_ub_iterator.h` | **不动**，引用 `attention/mla_preprocess/op_kernel/l1_to_ub_iterator.h` |

### 测试与文档

| 文件 | 改动 |
|---|---|
| `tests/perf/perf/model/matmul_split_kn.py` | 已修复 (Bug A/B/C)，pending commit |
| `tests/perf/perf/model/dequant_opt.py` | 已落地 (commit 4c46bda97) |
| `tests/perf/perf/model/split_kn_fused.py` | 已落地 (commit 10e54bd18) |
| `tests/perf/test_golden/*.txt` | 已更新 24/24 (split-kn 修正后) |
| `tests/pytest/test_*.py` | Phase 2：新增 enableSplitKn / enableDequantOpt 组合用例 |
| `docs/MlaPrologV3_950_综合优化设计.md` | **本文档** |

---

## 10. 实施步骤建议

按风险与依赖排序，分三轮 land：

### Round 1：DequantOpt（最低风险，独立可验）
- 修改 `mla_prolog_v3.cpp` 连线 `EnableDequantOpt` template 参数
- 新增 `MmQcQrDequantRopeFused` 函数
- 引入 `l0c_to_ub` / `ub_to_l1` iterators
- pytest：新增 FP8/INT8 hd128 decode 用例对比 baseline 与 dequantOpt
- profile：实测 ~3 µs 节省，对比 perf model
- **预期**：FP8 decode bs1 hd128 从 48.97 → ~45.44 µs

### Round 2：Split-KN（中风险，影响 host 与 cube）
- Host tiling 加 `enableSplitKn` flag、`mm1/mm2NGroups/KGroups`、partial workspace
- Kernel `MatmulSplitKN` cube 路径，输出 FP32 partial 到 GM 子区
- AIV 端**先用 baseline accum + RmsNorm**（不开 fused），让 split-kn 独立验证
- pytest：新增 hd8 BF16 decode 用例（split-kn 收益最大）
- profile：实测 ~16 µs cube 时间下降（hd8/hd128/INT8/FP8 sweep）
- **预期**：FP8 decode bs1 hd128 从 ~45.44 → ~39.45 µs（加上 dequantOpt）

### Round 3：Split-KN-Fused（在 Round 2 基础上）
- 扩 `service_rms_norm.h` 加 LocalTensor 重载
- 新增 `AivProcess_RmsNormCq_FusedAccum` / `AivProcess_RmsNormCkvKrRopeScatter_FusedAccum`
- 删除 fused 路径下的（本应新增的）barrier
- pytest：对比 fused vs naive 的输出 bit-identical（同时验证累加顺序确定性）
- profile：实测 ~1.5 µs 节省
- **预期**：FP8 decode bs1 hd128 从 ~39.45 → ~37.92 µs

最终预期：FP8 decode bs1 hd128 从 48.97 → ~34.39 µs（−29.8%，**1.42× 加速**）

---

## 11. 风险与缓解

| 风险 | 等级 | 缓解 |
|---|:---:|---|
| Split-KN partial workspace 占用 ~11 MB（k_groups=4 + 28），prefill 大 stepBS 下需要 K-partial 区在 L2 cache 之外，HBM 带宽变高 | 中 | perf model 已假设 partial 走 L2 命中。Phase 3 profiling 校准；如果 L2 miss 显著，缩小 mm2KGroups（K 切到 14 或 8） |
| Bug A 修正后 hd128 split-kn 收益从 38.8% 降到 18.4%，与早期 split-kn 报告矛盾 | 低 | 早期数字源自 buggy model；修正后才是真值。已在 `report_split_kn.md` 第 1 节解释 |
| `l0c_to_ub` 的 VDEQ16 模式不覆盖 V3 全部量化组合（per-token kvQuantMode=3 需要 multi-burst scale） | 中 | per-token 路径回退到 §8.5 第 4 步显式 Cast + Mul；perf model 已注释此限制 |
| `service_rms_norm.h` 当前签名只接受 GlobalTensor 输入，需要扩展 | 中 | Phase 2 kernel 工作；接口变更不影响外部，只在 fused 路径下走新重载 |
| Decode bs1 stepBS 小，per-V-core M=1 时 chunks=1，M-split 假设依然成立，但 V 核利用率低（只有 stepBS 个 V 核工作） | 低 | 与 baseline 等效，不会更差。perf model 已正确反映 |
| Intra-core HardEvent (`FIX_V`, `V_MTE3`, `MTE3_M`) 实际硬件延迟未知，模型按 0 估算 | 低 | per H6 假设，profiling 校准。即便有 1–2 cycle stalls，不影响相对收益 |
| 三 template 参数组合的 dispatch 表 4 条新路径，编译时间膨胀 | 低 | EnableDequantOpt 仅在量化路径开启，Split-KN 与 Fused 绑定。实际新增 ~3 个 template 实例 |

---

## 12. 验证

### 12.1 Perf model 验证（已完成）

```bash
cd attention/mla_prolog_v3/tests/perf
python test_regression.py        # 24/24 passes (corrected baseline)
python test_perf_unit.py         # 35/35 passes

# 复现报告中的数字
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 8   --mode split-kn
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode split-kn
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --weight-quant-mode 3 --mode dequant-opt
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 8   --mode split-kn-fused
```

### 12.2 Kernel 实测验证（Phase 2 完成后）

```bash
# pytest with new feature flags
pytest attention/mla_prolog_v3/tests/pytest/test.py \
       -k "test_split_kn_decode_bs1_hd128 or test_dequant_opt_fp8" -v

# profile (实机)
bash build.sh --pkg --soc=ascend910b --ops=mla_prolog_v3 -j16
python tests/profile/profile_decode_fp8.py --enable-split-kn --enable-dequant-opt
# 期望: ~33.8 µs ± 15%
```

### 12.3 与 perf model 对齐

每完成一个 Round，运行实机 profiling 并对比 perf model 预测：

| Round | 配置 | Perf model 预测 | 实机目标 (±15%) |
|---|---|---:|---:|
| 1 (DequantOpt) | FP8 decode bs1 hd128 | 45.43 µs | 38.6–52.2 µs |
| 2 (+ Split-KN) | FP8 decode bs1 hd128 | 38.58 µs | 32.8–44.4 µs |
| 3 (+ Fused) | FP8 decode bs1 hd128 | 37.12 µs | 31.6–42.7 µs |

---

## 13. 参考

- `MlaPrologV3_DequantOpt设计.md` — Round 1 的详细设计
- `MlaPrologV3_SplitKN_FusedAccum设计.md` — Round 3 的详细设计
- `tests/perf/report_split_kn.md` — Round 2 的 perf model sweep 与最优配置选择
- `tests/perf/methodology.md` — 性能建模假设 (H1–H9)
- `tests/perf/CLAUDE.md` — perf 模块导览，§3 hypotheses 含 H9 (per-AI-core split-kn scope)
- `attention/mla_preprocess/op_kernel/l0c_to_ub_iterator.h` — DequantOpt 引用的 L0C→UB iterator
- `attention/mla_preprocess/op_kernel/l1_to_ub_iterator.h` — DequantOpt 引用的 UB→L1 iterator
- `attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/` — UB pingpong 工程范式
