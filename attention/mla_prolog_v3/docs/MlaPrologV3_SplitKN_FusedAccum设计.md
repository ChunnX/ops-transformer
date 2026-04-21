# MlaPrologV3 Split-KN 融合 K-Accumulation + RmsNorm 设计

> ⚠️ **Status: superseded under no-sync convention** — 本设计的核心收益是节省"独立 Accum 步骤新增 barrier"的 ~0.8 µs/MM 成本。新建模约定（CLAUDE.md §3 H6）将 `sync_overhead_us = 0`，使该收益归零，split-kn-fused 在数值上与 split-kn 等价。文档保留作历史与未来恢复 sync 建模的参考。
>
> **状态**：设计阶段（perf model 已落地，kernel 实现待定）
> **适用范围**：Ascend 950 chip，仅当 split-kn 启用且 `k_groups > 1` 时生效
> **定位**：本方案是 **`--mode split-kn` 的补充**，不是 split-kn 的替代

---

## 1. 设计动机

`--mode split-kn` 把 MM1 / MM2 cube 核分成 `n_groups × k_groups` 网格。当 `k_groups > 1` 时，K 维被切分到多个 cube 核上，每核产生一个 K-partial。这些 partial 必须沿 K 维相加才能给下游 RmsNorm 用。

如果按"独立 Accum 步骤"实现 K 累加：

1. 需要插入一个**新的全核同步**——所有 k_groups cube 核必须先把 partial 写到 GM workspace，vector 才能开始读取并累加；
2. Accum 输出 → GM → RmsNorm 输入 → UB 多一次 GM 往返。

本方案把 K 累加**融合**到现有的 M-split RmsNorm V 核内：每个 V 核在自己的 M 行范围内顺序读取 k_groups 个 partial，UB 内累加到一个 FP32 buffer，立即在 UB 内跑 RmsNorm，最后把 BF16 结果直接写回 GM。**无需新的 barrier，无需 GM 中转**。

> **关键边界**：本方案**不动**现有 kernel barrier (`kernel_mla_prolog_split_n.h:854,859,867` 三处 `WaitAllCore<SYNC_MODE_ALL_VEC>`)，它们与 split-kn 无关，根因是 cube N-split / vector M-split 的策略错配。本方案只避免 split-kn 引入的**新增** barrier。

---

## 2. 当前状态

`perf/model/matmul_split_kn.py:84-98` 给出了 baseline AccumCq / AccumCkvKr 的建模：

```python
if spec.k_groups > 1:
    per_group_load   = k_groups × M × N_per_core × 4    # FP32 partials
    per_group_compute = M × N_per_core × (k_groups - 1)
    per_group_store  = M × N_per_core × out_size         # ★ 写回 GM
    accum_timing = estimate_vector_op(...)
```

独立 Accum + RmsNorm 的流程（想象中的 split-kn kernel 实现）：

```
AIC: cube_K[0..k_groups-1] → GM partials region [0..k_groups-1]
     ★ would need NEW barrier — all k_groups cubes done
AIV: read k_groups partials → reduce in UB → write reduced result to GM
     ★ separate barrier (vector finishes accum)
     read reduced result from GM → RmsNorm in UB → write to GM
     ★ existing barrier (line 859) — cube core split mismatch with MM3
```

`perf/model/matmul_split_kn.py` 当前**没有**显式建模 "新 barrier" 成本，这是 baseline 与 fused 版本对比时需要修正的地方（详见 §9）。

---

## 3. 现有 kernel barrier 不变

| 行号 | barrier | 根因 | 本方案 |
|---|---|---|---|
| `kernel_mla_prolog_split_n.h:854` | `WaitAllCore<SYNC_MODE_ALL_VEC>(FINISH_VEC_ALL)` after `CrossCoreWaitFlag(FINISH_MM_CQ)` | cube N-split → vector M-split 错配 | **保留** |
| `:859` | 同上 after RmsNormCq | 同上 | **保留** |
| `:867` | 同上 after `CrossCoreWaitFlag(FINISH_MM_CKVKR)` | 同上 | **保留** |

`:857-858` 注释明确说明：
> 由于 RmsNormCq 和 MatmulQcQr 的分核策略不一样，需要等所有 vector 上的 RmsNormCq 执行完成后才能启动 MatmulQcQr

这与 split-kn 完全无关——即使 `k_groups == 1`，这些 barrier 也存在。本方案**不触碰**它们。

---

## 4. 新方案数据流

### 4.1 数据流图

```
AIC (split-kn): cube_K1 → L0C → FixPipe → GM partials region 1
                cube_K2 → L0C → FixPipe → GM partials region 2
                ...
                cube_K{k_groups} → L0C → FixPipe → GM partials region k_groups
                CrossCoreSetFlag PIPE_FIX (FINISH_MM_CQ)         ── 与 baseline 相同

AIV: CrossCoreWaitFlag(FINISH_MM_CQ)                              ── 不变
     WaitAllCore<ALL_VEC>(FINISH_VEC_ALL)                         ── 现有 barrier，保留

     [对每个 V 核] my_M = aivOffset.curBlockTokenOffset + per_v_M
     for chunk c in 0..num_chunks-1:                              ── UB chunking 内层循环
         FP32 accumulator in UB ← 0
         for kg in 0..k_groups-1:
             MTE2: GM partials_region[kg][my_M_rows × N_full] → UB partial buf (db)
             accumulator += partial buf      (PIPE_V Add)
         ─ 累加完成；结果在 UB 内，不写 GM ─
         RmsNorm(rmsOutputUB, accumulator, gamma, ...) (PIPE_V)
         MTE3: UB rmsOutputUB → GM rmsNormCqResGm[my_M_rows × N_full]

     WaitAllCore<ALL_VEC>(FINISH_VEC_ALL)                         ── 现有 barrier #2，保留
     CrossCoreSetFlag PIPE_MTE3 (FINISH_VEC_RMSNORM_CQ)           ── 不变
```

### 4.2 同步点对照

| 同步点 | 独立 Accum + RmsNorm（baseline） | Fused（本方案） |
|---|---|---|
| cube → V handover (FINISH_MM_CQ) | CrossCoreSetFlag 0.8 µs | 不变 |
| 现有 WaitAllCore #1 (line 854) | 0.8 µs | 不变（保留） |
| **K-accum 步骤前的新 barrier** | **+0.8 µs (新增)** | **0**（融合到 RmsNorm 步骤内） |
| K-accum 计算 | k_groups GM reads + reduce + GM store | k_groups GM reads + reduce **(no GM store)** |
| RmsNorm 输入 → 输出 | GM read → vector → GM store | UB → vector → GM store |
| 现有 WaitAllCore #2 (line 859) | 0.8 µs | 不变（保留） |
| V → cube handover (FINISH_VEC_RMSNORM_CQ) | CrossCoreSetFlag 0.8 µs | 不变 |
| **本方案避免的开销（每 MM）** | — | **新 barrier (0.8 µs) + 1 次 GM round-trip (~0)** |

每 MM 节省约 0.65–0.78 µs；MM1 + MM2 合计节省 ~1.4 µs（详见 §8 实测数据）。

### 4.3 关键不变量

- 每 V 核只处理自己的 M 行；M 行间无依赖
- K 累加是 V 核**核内顺序**操作，不跨 V 核
- **没有新的全核 barrier**——这是核心收益
- 累加结果**留在 UB**，直接 RmsNorm，省一次 UB → GM → UB 往返

---

## 5. 950 硬件能力清单

| 能力 | 数值 | 来源 |
|---|---|---|
| UB 容量（每核） | 256 KB | `hardware.py:157` |
| AIV 静态 buffer 占用 | ~34 KB（gamma + dequant scale + sin/cos + share buffer） | `kernel_mla_prolog_split_n.h:510-578` |
| 可用 UB 工作区 | ~220 KB/core | `cache.ub_size - STATIC_AIV_UB_BYTES` |
| HBM/L2 单核带宽 | 50 / 162 GB/s contended | `hardware.py:163,164` |
| Vector ops/sec | 20.0e12 (chip total) | `hardware.py:149` |
| Cross-core sync overhead | 0.8 µs / `CrossCoreSetFlag` | `hardware.py:152` |
| **关键约束** | UB 是 per-core，跨 V 核共享必须经 GM/L2 | — |

---

## 6. UB 资源规划与切分调整

V 核工作区（fused 步骤）：

```
per_v_UB ≈ static (~34 KB)
         + 2 × chunk_m × N_full × 4    (FP32 partial input, double-buffered)
         + chunk_m × N_full × 4         (FP32 accumulator)
         + chunk_m × N_full × 4         (FP32 RmsNorm working buffer)
         + 2 × chunk_m × N_full × 2     (BF16 output, double-buffered)
         = 34 KB + chunk_m × N_full × 20
```

For MM1 (`N_full = Hcq = 1536`)，每行约 30 KB；available 220 KB 允许 `chunk_m ≤ 7` 行：

| 配置 | step_batch_size | per_v_M | chunk_m | chunks | per_v_UB | 决策 |
|---|---:|---:|---:|---:|---:|---|
| decode bs1 seq1 hd128 | 1 | 1 | 1 | 1 | ~30 KB | ✓ |
| decode bs8 seq1 hd128 | 8 | 1 | 1 | 1 | ~30 KB | ✓ |
| prefill T=128 hd128 | 128 | 2 | 2 | 1 | ~60 KB | ✓ |
| 假设 stepBS=512 | 512 | 8 | 7 | 2 | ~210 KB / ~210 KB | ✓ chunks=2 |
| 假设 stepBS=1024 | 1024 | 16 | 7 | 3 | 同上 | ✓ chunks=3 |

For MM2 (`N_full = Hckv + Dr = 576`)，每行 ~11 KB，per_v_M 上限更宽松（chunk_m ≤ 19）。

**决策规则**：
```
per_v_UB_max = 0.7 × 220 KB ≈ 154 KB
chunk_m_max  = floor((per_v_UB_max - 34 KB) / (N_full × 20))
chunks       = ceil(per_v_M / chunk_m_max)
```

**结论**：实际 stepBatchSize 通常 ≤ 128（`tiling/config.py:117`），所有 decode + 中等 prefill 场景 `chunks=1`。chunking 是大 stepBatchSize 的安全网。

---

## 7. 伪代码

```cpp
// 引用：mla_prolog/op_kernel/service_rms_norm.h
//       perf/model/matmul_split_kn.py:84-98 (baseline accum 模型)
//
// Phase 2 kernel 会引入 EnableSplitKnFusedAccum template 参数，并扩展
// service_rms_norm 接口接收 LocalTensor (UB) 输入。

template <bool EnableSplitKnFusedAccum>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::AivProcess_RmsNormCq_FusedAccum(
    int64_t batchOffset, int64_t curStepBatchSize, uint32_t k_groups)
{
    if constexpr (!EnableSplitKnFusedAccum) {
        // 走原 path：先 Accum 后 RmsNorm，需要新 barrier
        SeparateAccumThenRmsNormCq(batchOffset, curStepBatchSize, k_groups);
        return;
    }

    // === 融合路径 ===
    const int64_t my_m_offset = aivOffset.curBlockTokenOffset;
    const uint32_t my_m_full = aivOffset.curVecToken;  // 现有 M-split per_v_M

    // UB chunking when M is too big to fit
    const uint32_t chunk_m = ComputeChunkM(my_m_full, baseParams_->headSizeCq);
    const uint32_t num_chunks = ceil_div(my_m_full, chunk_m);

    for (uint32_t c = 0; c < num_chunks; ++c) {
        const uint32_t cur_m = min(chunk_m, my_m_full - c * chunk_m);
        const int64_t row_offset = batchOffset + my_m_offset + c * chunk_m;

        // 1. 累加器清零
        Duplicate(accumUbFp32_, 0.0f, cur_m * baseParams_->headSizeCq);

        // 2. 顺序读 k_groups 个 partial，UB 内累加
        for (uint32_t kg = 0; kg < k_groups; ++kg) {
            const uint32_t ping = kg & 1;
            // MTE2: GM partial[kg] for my M rows × full N → UB
            DataCopyPad(
                partialUb_[ping],
                mmCqPartialGm_[kg][row_offset * baseParams_->headSizeCq],
                /*nBurst=*/cur_m,
                /*lenBurst=*/baseParams_->headSizeCq * sizeof(float));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + ping);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0 + ping);

            // PIPE_V: accumulator += partial
            Add(accumUbFp32_, accumUbFp32_, partialUb_[ping],
                cur_m * baseParams_->headSizeCq);
            PipeBarrier<PIPE_V>();
        }
        // 累加完成；累加结果已经在 UB 里，**没有 GM 写回，没有新 barrier**

        // 3. PIPE_V: RmsNorm 直接读 UB 内的累加结果
        //    复用 service_rms_norm.h 的 row-wise 实现，
        //    输入从 GlobalTensor 改为 LocalTensor (Phase 2 kernel 工作)
        RmsNormLocal(
            rmsOutputUb_,                  // dest BF16
            accumUbFp32_,                  // src FP32 (UB)
            rmsnormGammaCqUb_,
            rmsNormParams_,
            cur_m,                         // 本 chunk 行数
            baseParams_->headSizeCq);
        PipeBarrier<PIPE_V>();

        // 4. PIPE_V → MTE3: UB → GM (final destination)
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopyPad(
            rmsNormCqResGm_[row_offset * baseParams_->headSizeCq],
            rmsOutputUb_,
            cur_m,
            baseParams_->headSizeCq * sizeof(rmsNormCqOutputType));
    }
    // 注意：**没有插入新的 WaitAllCore**；
    // 现有的 line 854/859 barriers 仍在外层调用处，由调用方管理，不变。
}
```

### 7.1 关键设计点

- **k_groups 顺序循环 in V core**：每 V 核串行读 k_groups 个 partial。单 partial 大小 = `chunk_m × N_full × 4 bytes`，远小于 UB
- **累加器 FP32**：保持精度，避免 BF16 累加误差。FP32 累加器 + 双缓冲 partial input 是 UB 的主要消费者
- **RmsNorm 直接消费 UB 累加结果**：不需要 GM 中转。`service_rms_norm.h:47` 当前签名只接受 GlobalTensor，需要在 Phase 2 添加 LocalTensor 重载
- **没有新的全核 barrier**：每 V 核独立完成累加 + RmsNorm + 写回；外层的现有 barrier (line 854/859) 由调用方管理，不变
- **chunking**：大 stepBatchSize 时按 chunk_m 切分内层循环；外层 cube/V 接口不变
- **MM2 对称**：相同模式，N_full = Hckv+Dr，per-row footprint 更小

---

## 8. Perf 模型实测数据

由 `perf/cli/modes/split_kn_fused.py` 给出。运行命令：

```bash
python perf_analyzer.py --chip 950 --batch-size 1 --seq-len 1 --head-num 8 --mode split-kn-fused
```

四个代表配置的最佳 fused 结果（best fused config 对每行的对比）：

| 配置 | Baseline split-kn | Fused | Δ (saved) | 占比 | 最佳 config | MM1 UB | MM2 UB |
|---|---:|---:|---:|---:|---|---:|---:|
| **decode bs1 seq1 hd8 BF16** | 9.01 µs | **7.55 µs** | **−1.46 µs** | **−16.2%** | MM1:N1×K28 + MM2:N3×K8 | 30 KB (14%) | 11 KB (5%) |
| decode bs1 seq1 hd128 BF16 | 53.31 µs | 51.84 µs | −1.46 µs | −2.7% | MM1:N1×K28 + MM2:N3×K8 | 30 KB (14%) | 11 KB (5%) |
| decode bs8 seq1 hd128 BF16 | 54.07 µs | 52.70 µs | −1.37 µs | −2.5% | MM1:N1×K28 + MM2:N3×K8 | 30 KB (14%) | 11 KB (5%) |
| decode bs1 seq1 hd128 INT8 | 30.88 µs | 29.41 µs | −1.46 µs | −4.7% | MM1:N1×K28 + MM2:N3×K8 | 30 KB (14%) | 11 KB (5%) |

**hd8 case 占比最大**：因为 hd8 整体 pipeline 短（9 µs），1.46 µs 占了 16% 的整体收益。这正是 split-kn 最有意义的场景（小 N 时单核 N 不够用，K-split 提升核利用率），fused 让每次 split-kn 调用都更便宜。

### 8.1 节省构成分解（每 MM）

```
MM1 (Cq) k_groups=28:
  Fused total       : 0.262 µs  (accum_load 0.108 + accum_compute 0.138 + rmsnorm 0.015 + store 0.002)
  Baseline total    : 0.947 µs  (accum 0.133 + new_barrier 0.800 + rmsnorm 0.015)
  Δ saved           : +0.685 µs   ← 主要来自避免的新 barrier (0.8 µs)

MM2 (CkvKr) k_groups=8:
  Fused total       : 0.033 µs  (accum_load 0.012 + accum_compute 0.015 + rmsnorm 0.006 + store 0.001)
  Baseline total    : 0.810 µs  (accum 0.004 + new_barrier 0.800 + rmsnorm 0.006)
  Δ saved           : +0.777 µs   ← 主要来自避免的新 barrier
```

每 MM 节省 ≈ 0.7 µs，几乎全部来自避免的 1 个 barrier；MM1 + MM2 合计 ~1.4 µs。

---

## 9. 与现有 perf 模型 split-kn 的关系

`--mode split-kn` 输出每个 (n_groups, k_groups) 配置的 baseline 总耗时（其中包含 AccumCq + RmsNormCq 的 vector op 时间，但**不包含**新 barrier 成本）。`--mode split-kn-fused` 在每个 `k_groups > 1` 配置上**额外**计算 fused 变体：

```
                              ┌─ Modeling correction in --mode split-kn-fused only ─┐
Baseline (split-kn) total  :  AccumCq + RmsNormCq + new_barrier (0.8 µs)
Fused total                :  fused step (load + accum + rmsnorm + store, no GM round-trip)
Δ saved                    :  baseline_total - fused_total
```

**重要**：`new_barrier` 这一项**只在 split-kn-fused mode 内部对比时加入**，不污染 `--mode pipeline` / `--mode split-kn` 的输出 DAG。这样：
- `--mode split-kn` 的 4 个现有 regression goldens **保持 byte-identical**
- `--mode split-kn-fused` 显示一个**统一的 corrected baseline**，正确反映独立 Accum 步骤在真实 kernel 中需要的同步成本

这一设计沿用了 dequant-opt（commit `4c46bda97`）的"corrected baseline"模式。

---

## 10. 与其他优化的协同

| 优化 | 阶段 | 节省（FP8 decode bs1 hd128） | 状态 |
|---|---|---|---|
| `split-kn`（perf 模型探索） | MM1 / MM2 cube | ~17.9 µs (−36.7%) | perf model only |
| `split-kn-fused`（本方案） | 在 split-kn 之上避免 K-accum barrier | **~1.4 µs (−2.7%)** | perf model 已落地，本设计 |
| `dequant-opt`（commit `4c46bda97`） | MM3 → MM4 vector | ~3.3 µs (−6.7%) | 已落地 |

三者**正交**：
- `split-kn` 改 cube 切分
- `split-kn-fused` 改 vector 累加路径（仅当 split-kn 启用时）
- `dequant-opt` 改 MM3→MM4 段的 cube/vector handover

预计组合收益（FP8 decode bs1 hd128）：48.71 → ~25 µs（≈ **1.95× 加速**）。

---

## 11. 风险与限制

| 风险 | 等级 | 缓解 |
|---|---|---|
| `k_groups` GM partial 区域布局 stride 模型偏差 | 中 | perf model 用保守 strided MTE2 估算；Phase 2 kernel profiling 校准 |
| `chunk_m × N_full × 20` 在大 stepBatchSize 下溢 UB | 中 | chunking 通过内层循环；perf model 显式输出 `chunks` 字段；> 70% 阈值警告 |
| `service_rms_norm.h` 当前签名只接受 GlobalTensor 输入；融合需要 LocalTensor 输入重载 | 中 | Phase 2 kernel 工作；perf model 不依赖此细节 |
| FP32 累加器是 UB 主要消费者；BF16 累加可降低，但损失精度 | 低 | 保守用 FP32；split-kn 的 K 累加非确定性问题已在 H7 假设中处理 |
| Baseline `--mode split-kn` 当前没建模"新 barrier"成本 | 中 | corrected baseline 仅在 split-kn-fused 内部使用，不动 baseline goldens |
| `k_groups == 1` 时方案不适用 | low risk | perf model 在 k_groups = 1 情况下返回 `applicable=False` |
| 每 MM 节省仅 ~0.7 µs（barrier + 微量 GM round-trip） | low risk → upside | 正面收益但不夸大；hd8 场景占整体 16%，hd128 占 ~3% |
| 现有 kernel barriers (line 854, 859, 867) 与本方案无关，文档需明确说明 | 已记录 | §3 表格 + §10 表格 + 风险表都强调"existing barriers stay" |

---

## 12. 实施步骤

| 阶段 | 工作 | 状态 |
|---|---|---|
| Phase 1（本设计） | 写文档 + perf 模型扩展 + 4 个 golden 快照 + 5 个单元测试 | **已完成** |
| Phase 2（后续） | Kernel 实现：在 mla_prolog_v3.cpp 增加 `EnableSplitKnFusedAccum` template 参数 + 修改 vector AivProcess 的 RmsNormCq/RmsNormCkvKr 调用，扩展 `service_rms_norm.h` 接口 | 待启动 |
| Phase 3（后续） | Profiling 校准：实测 partial GM 读 strided 模式开销、FP32 累加器 UB 占用、融合后的实际 µs | 待启动 |

---

## 13. 参考文件

**性能模型：**
- `attention/mla_prolog_v3/tests/perf/perf/model/split_kn_fused.py` — 时序与 budget 模型
- `attention/mla_prolog_v3/tests/perf/perf/cli/modes/split_kn_fused.py` — `--mode split-kn-fused`
- `attention/mla_prolog_v3/tests/perf/test_golden/decode_*_split_kn_fused.txt` — 4 个 golden 快照
- `attention/mla_prolog_v3/tests/perf/test_perf_unit.py::TestSplitKnFused` — 5 个单元测试

**已有 perf 模型依赖（不变）：**
- `attention/mla_prolog_v3/tests/perf/perf/model/matmul_split_kn.py:36-102` — 现有 split-kn 时序模型 + AccumCq baseline
- `attention/mla_prolog_v3/tests/perf/perf/tiling/split_kn_search.py:64-139` — 现有 split-kn 配置搜索
- `attention/mla_prolog_v3/tests/perf/perf/pipeline/dag.py:120-150` — 现有 baseline DAG（不变）

**Kernel 待修改（Phase 2，本设计不含）：**
- `attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp:26,79-175` — 增加 `EnableSplitKnFusedAccum` template 参数
- `attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h:854-865` — RmsNormCq 调用站点（融合分支）
- `attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h:867-868` — RmsNormRopeScatterCkvKr 调用站点
- `attention/mla_prolog/op_kernel/service_rms_norm.h:47-90` — 添加 LocalTensor 输入重载

**不动（参考前文 §3）：**
- `kernel_mla_prolog_split_n.h:854,859,867` — 三个现有 `WaitAllCore<SYNC_MODE_ALL_VEC>` barrier，与本方案无关
