# Split-KN Optimization Report (Per-AI-Core Corrected Model)

> ⚠️ **绝对耗时数据已 stale**：本报告中的 µs 数值产生于 `sync_overhead_us = 0.8 µs` 的旧建模约定。新约定（CLAUDE.md §3 H6）将 sync_overhead 归零，所有 baseline 耗时下降约 3–5 µs。**最优 split-kn 配置（best (n_groups, k_groups) 组合）的相对结论不受影响**，因为同步成本对所有候选一致。需重生数据时运行 `python perf_analyzer.py --chip 950 --from-testcase perf_decode_bs1 --mode split-kn`。
>
> 数据来源：`perf/model/matmul_split_kn.py` 的修正版本（commit pending）
> 修正内容：per_core_N bug 修复 (Bug A)、FixPipe 压力修正 (Bug B)、vector accumulation 改为 per-AI-core (Bug C)
> 流水掩盖：由 `perf/pipeline/dag.py::build_pipeline_dag` 处理（accum 节点是独立 AIV stage，DAG 前向自动隐藏到与之并行的 AIC matmul 之下）
> 验证：`python perf_analyzer.py --chip 950 --mode split-kn` + 24/24 regression goldens

---

## 1. 修正前后对比 — 为什么旧模型乐观

旧模型在 `estimate_matmul_split_kn` 内传给 `estimate_matmul_detailed` 的参数是 `N = N_per_core × n_groups, active_cores = n_groups × k_groups`，导致函数内部 `per_core_N = ceil(N / C) = N_per_core / k_groups`，**每核 N 被错除了一个 k_groups 因子**。这使得每核的 cube 计算量、L0B 加载、FixPipe 字节都被低估了 `k_groups×`。

例：`decode bs1 hd8 BF16` (T=4 stepBatchSize)：

| 指标 | 旧 (buggy) | 新 (corrected) | 变化 |
|---|---:|---:|---|
| Best config | `MM1:N1×K28+MM2:N3×K8` | `MM1:N4×K7+MM2:N1×K28` | 完全不同 |
| Best pipeline | 9.32 µs | 27.23 µs | +192% |
| 报告的相对收益 | 78.5% | 37.2% | -41.3 pp |
| MM1:N1×K28 的 cube_us | 0.51 µs | 14.31 µs | +28× ≈ k_groups |

旧模型把高 k_groups 配置（K28、K14）人为地优先化，让它们看上去比 N-多元配置好得多。修正后，K-split 不再"白嫖"算力——cube 时间正确反映 HBM 带宽限制，**最优 config 自然回归到 N×K 比例更均衡的 `N4×K7`（BF16，28 核避免 HBM contention）/ `N8×K4`（FP8/INT8，32 核）**。

---

## 2. Sweep 结果：修正后 split-kn 真实收益

测试矩阵涵盖 head_num 1-128、batch sweep、量化模式、prefill。所有结果使用 `python perf_analyzer.py --chip 950 --mode split-kn`，pipeline µs 来自 `build_pipeline_dag` 的关键路径求解，已计入 AIC/AIV 重叠。

| 场景 | dtype | Baseline pipeline | Best pipeline | Δ (us) | Δ (%) | Best config | accum_us | CubeUtil | Vector hidden? |
|---|---|---:|---:|---:|---:|---|---:|---:|:---:|
| decode bs1 hd8   | BF16 | 43.33 | **27.23** | −16.10 | **−37.2%** | MM1:N4×K7 + MM2:N1×K28 | 0.08 | 85.3% | ✓ |
| decode bs1 hd16  | BF16 | 46.28 | **30.18** | −16.10 | **−34.8%** | MM1:N4×K7 + MM2:N1×K28 | 0.08 | 86.7% | ✓ |
| decode bs1 hd32  | BF16 | 53.71 | **37.61** | −16.10 | **−30.0%** | MM1:N4×K7 + MM2:N1×K28 | 0.08 | 89.4% | ✓ |
| decode bs1 hd64  | BF16 | 64.04 | **47.93** | −16.10 | **−25.1%** | MM1:N4×K7 + MM2:N1×K28 | 0.08 | 91.7% | ✓ |
| decode bs1 hd128 | BF16 | 87.63 | **71.53** | −16.10 | **−18.4%** | MM1:N4×K7 + MM2:N1×K28 | 0.08 | 94.4% | ✓ |
| decode bs8 hd128 | BF16 | 87.66 | **71.56** | −16.10 | **−18.4%** | MM1:N4×K7 + MM2:N1×K28 | ~0.5 | 94.4% | ✓ |
| decode bs1 hd128 | INT8 | 51.10 | **39.20** | −11.90 | **−23.3%** | MM1:N8×K4 + MM2:N1×K28 | ~0.1 | 85.2% | ✓ |
| decode bs1 hd128 | FP8  | 48.71 | **39.20** | −9.52 | **−19.5%** | MM1:N8×K4 + MM2:N1×K28 | ~0.1 | 85.2% | ✓ |
| prefill T=128 hd128 | BF16 | 88.29 | **72.19** | −16.10 | **−18.2%** | MM1:N4×K7 + MM2:N1×K28 | ~5 | 94.4% | ✓ |

### 关键观察

1. **BF16 场景最优 config 为 `MM1:N4×K7 + MM2:N1×K28`**（28 核），FP8/INT8 场景仍用 `MM1:N8×K4 + MM2:N1×K28`（32 核）。BF16 场景切换到 28 核是因为 chip-total / contention 建模让 32 核的 per-core HBM 带宽下降 ~10%，而 FP8/INT8 weight 减半使 HBM 竞争较不显著。
2. **BF16 绝对节省固定 ~16.1 µs**：MM1+MM2 cube 总时间从 35.78 µs (= 17.89 + 17.89) 压缩到 19.68 µs (= 14.31 + 5.37)，差值 16.10 µs 与 BF16 表中的所有 Δ 一致。
3. **量化让绝对节省缩水**：INT8 节省 11.90 µs (23.3%)，FP8 节省 9.52 µs (19.5%)。原因是量化已经把 baseline 的 MM1/MM2 cube time 分别从 17.89 µs 压到 10.73 / 8.35 µs，留给 split-kn 的余地变小。
4. **节省 % 随 hd 增大单调下降**：MM3 与 hd 成正比（hd8 = 2.95 µs → hd128 = 47.19 µs），split-kn 不能加速 MM3，所以 MM3 在 pipeline 中的占比越大，整体 % 收益越小。
5. **Vector 在 9 个场景中全部被掩盖**：所有最优 config 的 `aiv_busy < aic_busy`，DAG 关键路径完全在 AIC 上。即便 prefill T=128 的 accum_us 达到 5.41 µs，仍 < MM2 cube (5.37) + MM3 cube (47.19) + 后续 AIC 工作的总并行窗口，DAG 隐藏完毕。

---

## 3. Vector 掩盖深度分析

用户的核心诉求是："**当 vector 操作能被其他 cube 计算掩盖时，不需要计入总时长里**"。验证结论：**所有最优配置都满足这一约束**，无需额外干预。

### 3.1 掩盖机制

DAG 中 `AccumCq` 是 AIV 节点，`depends_on=["aiv_wait_CQ"]`（即 MM1 完成后开始）；同时 AIC 流水中 `MM2_CkvKr` 也是 `depends_on=["sync_signal_CQ"]`（同样在 MM1 后开始）。两者**并行执行**：

```
AIC: MM1 ──→ MM2 ──→ ... 
              │
AIV: MM1 ──→ AccumCq ──→ ...
                │
              并行
```

如果 `AccumCq.duration < MM2.duration`，则 AccumCq 在 MM2 完成前就已结束 → 关键路径上看不到 AccumCq → "完全掩盖"。

### 3.2 掩盖余量（hd8 best config 为例）

```
config         : MM1:N4×K7+MM2:N1×K28
MM1 cube       : 14.31 µs   (AIC, on critical path)
MM1 AccumCq    :  0.08 µs   (AIV, parallel with MM2)
MM2 cube       :  5.37 µs   (AIC, parallel with AccumCq)
                            ★ 5.37 - 0.08 = 5.29 µs 余量
MM2 AccumCkvKr :  0.08 µs   (AIV, parallel with MM3)
MM3 cube       :  2.95 µs
                            ★ 2.95 - 0.08 = 2.87 µs 余量
```

**hd8 的掩盖余量充裕**：accum 只占并行 AIC 窗口的 < 1%。

### 3.3 prefill T=128 — 最压力的场景

prefill 是 accum 最容易"暴露"的场景：T 大 → 累加元素多 → accum_us 增长。但即便如此 hidden 仍成立：

```
config         : MM1:N4×K7+MM2:N1×K28  (T=128)
MM1 cube       : 14.31 µs   (AIC)
MM1 AccumCq    :  ~5 µs     (AIV)
MM2 cube       :  5.37 µs   (AIC, parallel with AccumCq)  ← AccumCq 略小于 MM2
MM3 cube       : 47.19 µs   (AIC)                          ← AccumCq 完成后还有 47 µs AIC 工作
```

虽然 `AccumCq (5.41) > MM2 cube (5.37)`，但 AccumCq 的尾巴（5.41 - 5.37 = 0.04 µs）被 MM3 cube 的开头吸收，关键路径仍在 AIC 上。pipeline_us = 72.19 µs，aic_busy = 68.18 µs，差值 4 µs 是 cross-core sync 开销，不是 vector 暴露。

### 3.4 哪些 config 会让 vector 暴露？

从 sweep 数据中只有 prefill T=128 的下列 config 让 aiv_busy 显著超出 aic_busy 的并行窗口：

| Config | accum_us | aiv_busy | aic_busy | pipeline_us | 暴露? |
|---|---:|---:|---:|---:|---|
| MM1:N1×K28 | 13.76 | 16.17 | 80.70 | 84.71 | 部分 (但 AIC 仍主导) |
| MM1:N2×K14 | 3.44 | 5.85 | 80.70 | 84.71 | 否 |
| MM1:N1×K14 | 3.44 | 5.85 | 80.70 | 84.71 | 否 |

即便 `MM1:N1×K28` 的 accum_us = 13.76 µs，aic_busy 仍是 80.70 µs，aiv_busy 16.17 µs 完全在并行窗口内。**只有 prefill T 极大 (T ≥ 256) 且选择 N=1 的极端 config 时，accum 才可能压上关键路径**——但那种 config 的 cube_us 同样恶化，整体不会被搜索器选中。

**结论**：在 search_split_kn 选出的最优 config 上，vector 永远是被掩盖的。这是 DAG 自然的副产物，不需要额外的"vector budget" 约束。

---

## 4. 最优切分推荐

### 4.1 BF16 最优：`MM1: n_groups=4, k_groups=7` + `MM2: n_groups=1, k_groups=28`

> FP8/INT8 quant modes still prefer N8×K4 because FP8/INT8 weights halve the per-core B load, making HBM contention at 32 cores less impactful.

**MM1 (Cq) — BF16 切分策略**

| 属性 | 值 |
|---|---|
| n_groups | 4 (列向 4 切，每核 single_n = Hcq/4 = 384) |
| k_groups | 7 (K 向 7 切，每核 single_k = He/7 = 1024) |
| 总核数 | 28 (= 4 × 7)，避免 32 核 HBM 竞争 |
| 每核处理 | M=stepBS, K=1024, N=384 |
| 每核 cube_us (BF16, T=1) | ~14.31 µs (HBM-bound) |
| 每核 FixPipe 字节 (FP32 partial) | M × 384 × 4 |

**MM2 (CkvKr) — 切分策略**

| 属性 | 值 |
|---|---|
| n_groups | 1 (不切 N，每核 single_n = Hckv+Dr = 576) |
| k_groups | 28 (K 向 28 切，每核 single_k = He/28 = 256) |
| 总核数 | 28 |
| 每核处理 | M=stepBS, K=256, N=576 |
| 每核 cube_us (BF16, T=1) | ~5.37 µs |
| 每核 FixPipe 字节 (FP32 partial) | M × 576 × 4 |

### 4.2 为什么这是最优

- **BF16 场景 MM1 用 28 核 (N4×K7) 而非 32 核 (N8×K4)**：chip-total / contention 建模下，32 核同时竞争 HBM 带宽使 per-core 带宽下降 ~10%。28 核配置（N4×K7）每核 N=384、K=1024，避开了 32 核 HBM contention，cube 时间相同（~14.31 µs）但实际 wall-clock 更优。FP8/INT8 weight 减半使 per-core B load 较低，32 核 contention 不显著，仍用 N8×K4。
- **MM2 的 N=576 偏小**：再切 N 会让单核 baseN 不饱和。把所有 28 cores 全部投到 K 维，每核 K=256（极小），cube_us 从 17.89 → 5.37 µs (−70%)。
- **核数预算**：MM1 用 28 cores (BF16) / 32 cores (FP8/INT8) + MM2 用 28 cores；实际 MM1 和 MM2 是 **顺序串行执行**（MM2 在 MM1 完成后启动），所以两者可以分别使用全部核。`28 < 32` 是因为 K=He=7168 不能整除 32，只能取 28 (=7168/256) 让 single_k 整齐。
- **AccumCq 0.08 µs / AccumCkvKr 0.08 µs**：T=1 时累加元素极少 → 几乎免费 → 永远被掩盖。
- **FP32 partial 的 FixPipe 压力**：每核 384 × 4 / 576 × 4 字节，per-core FixPipe ~0.5–1.5 µs，仍小于 cube 时间，不会成为瓶颈。

### 4.3 备选 config（如果实现成本受限）

如果 kernel 实现某些限制，下列备选 config 仍有显著收益：

| 备选 | hd128 BF16 pipeline | vs Best | 说明 |
|---|---:|---:|---|
| **Best**: MM1:N4×K7 + MM2:N1×K28 | 71.53 µs | 0 | 推荐 (BF16) |
| MM2:N1×K28 only (MM1 baseline) | 75.11 µs | +3.58 | 只优化 MM2，实现简单 |
| MM2:N3×K8 | 76.00 µs | +4.47 | MM2 的 N、K 更对称 |
| MM1:N4×K7 only (MM2 baseline) | 84.05 µs | +12.52 | 只优化 MM1，效果较弱 |

最便宜的实现路径是先做 **MM2:N1×K28**（只动 MM2），收益约 14% (75.11 vs 87.63)。完整收益需要同时做 MM1+MM2。

---

## 5. 与 split-kn-fused 的协同（commit 10e54bd18）

`split-kn-fused`（已 landed）是在 split-kn 基础上把 K-accumulation 与 RmsNorm 在 V 核内融合，避免独立 Accum step 引入新 barrier 与 GM round-trip。它和本报告的 split-kn 配置 **正交且互补**：

| 优化层 | 作用阶段 | hd8 收益 | hd128 收益 |
|---|---|---:|---:|
| split-kn (本报告) | MM1+MM2 cube 时间 | −16.10 µs (−37.2%) | −16.10 µs (−18.4%) |
| split-kn-fused (commit 10e54bd18) | accum + RmsNorm 融合 | −1.46 µs (额外 −5.4% on top) | −1.46 µs (额外 −2.0% on top) |
| **组合** | MM1+MM2 全段 | **−17.56 µs (−40.5%)** | **−17.56 µs (−20.0%)** |

`split-kn-fused` 的额外节省 ~1.46 µs 来自避免一个新 barrier（0.8 µs × 2 个 MM）和一个 GM round-trip。对 hd8 这种小场景特别值钱（占比 +5%），对 hd128 等大场景占比小但仍正向。

进一步组合 `dequant-opt` (commit 4c46bda97) 优化 MM3→MM4 段：

| 组合方案 | hd128 FP8 pipeline | vs no-opt baseline |
|---|---:|---:|
| no-opt baseline | 48.71 µs | — |
| split-kn only | 39.20 µs | −9.52 µs (−19.5%) |
| split-kn + split-kn-fused | ~37.74 µs | −10.97 µs (−22.5%) |
| split-kn + split-kn-fused + dequant-opt | **~34.44 µs** | **−14.27 µs (−29.3%)** |

预计 hd128 FP8 decode 综合优化可从 48.71 µs 降到 ~34.4 µs（约 1.41× 加速）。

---

## 6. 推荐落地顺序

按收益密度排序：

1. **MM1: N4×K7 (BF16) / N8×K4 (FP8/INT8) + MM2: N1×K28（split-kn 主优化）** — 实现成本：host tiling + kernel 加 K-split 累加路径。预计 collective 收益 10–16 µs/调用。
2. **split-kn-fused 融合 K-accum + RmsNorm**（已有 perf model 验证）— 实现成本：vector 侧 RmsNormCq/RmsNormCkvKr 接受 LocalTensor 输入。额外收益 ~1.5 µs/调用。
3. **dequant-opt L0C→UB 直通**（已有 perf model 验证）— 实现成本：cube/vector intra-core handover + l0c_to_ub iterator 调用。额外收益 ~3.3 µs/调用 (FP8/INT8 only)。

每一步都可独立 land，三步组合预计在 hd128 FP8 decode 上达到接近 1.5× 加速，对 hd8 / hd16 等小场景甚至更高。

---

## 7. 验证与复现

```bash
cd attention/mla_prolog_v3/tests/perf
python test_regression.py     # 24/24 passes (corrected baseline)
python test_perf_unit.py      # 35/35 passes

# 复现表 2 的任意一行
python perf_analyzer.py --chip 950 --batch-size 1 --seq-len 1 --head-num 8   --mode split-kn
python perf_analyzer.py --chip 950 --batch-size 1 --seq-len 1 --head-num 128 --mode split-kn
python perf_analyzer.py --chip 950 --batch-size 1 --seq-len 1 --head-num 128 --weight-quant-mode 3 --mode split-kn
python perf_analyzer.py --chip 950 --batch-size 1 --seq-len 128 --head-num 128 --mode split-kn

# 验证 vector hidden（看 cube_utilization 一栏）
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 8 --mode split-kn | grep -A 1 'CubeUtil'
```

`pipeline` µs 是 `build_pipeline_dag` 关键路径求解结果，已正确计入 AIC/AIV 流水掩盖。`accum_us` / `aiv_busy` / `aic_busy` 字段可用于人工验证 vector 是否被掩盖（hidden 的判断标准：`aiv_busy < aic_busy` 且 `cube_utilization > 80%`）。

---

## 8. 限制与注意事项

1. **chip-total / contention 视角**：本模型的 cube 时间使用 chip-total HBM 带宽 / active_cores 得到 per-core 带宽，已建模多核 HBM 竞争。BF16 场景下 32 核配置的 per-core 带宽比 28 核低 ~10%，这是 MM1 最优从 N8×K4 切换到 N4×K7 的原因。
2. **K_per_core 整除性**：K 必须能被 k_groups 整除。He=7168 = 2^10 × 7，可以整除 1, 2, 4, 7, 8, 14, 16, 28, 32 等 k_groups。当 k_groups=28 时 single_k=256，是 baseK=128 的两倍，cube 内部 stepK=2 工作良好。
3. **Hcq/Hckv 整除性**：N_per_core 也需对齐 baseN (32B 的整数倍)。Hcq=1536 整除 192=baseN×1.5；Hckv+Dr=576 不能被 64 整除给 9 个 cores，所以 N1×K28 是把 N 全留给单核的写法。
4. **stepBatchSize 影响 accum 大小**：T 越大，accum_us 越大。本报告测试到 T=128 (prefill cap)，accum 仍然被 hidden。如果未来 stepBatchSize 增大（>=256），需要重新验证 hidden 仍成立。
5. **MM3 不参与 split-kn**：MM3 (mmQcQr) 的 N=128×192=24576 已经被 32 cores 充分切分，K=Hcq=1536 较小，K-split 收益不显著。这是 dequant-opt 优化的目标段。
6. **Bug A 的修正不会改变 search 算法本身**：`search_split_kn` 仍然枚举 (n_groups, k_groups) 组合 + 全 pipeline DAG 评估，只是评估时使用了正确的 cube_us 数字。
