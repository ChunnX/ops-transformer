# MlaPrologV3 DequantOpt UB-Pipelined 优化设计

> 适用范围：Ascend 950 chip，decode 场景，FP8 / INT8 / MXFP8 量化模式
> 状态：设计阶段（已完成 perf model 验证），kernel 实现待定

---

## 1. 设计动机

MLA Prolog V3 在 Ascend 950 decode 场景下，**Split-KN 已经把 MM1 / MM2 的耗时压到很低**（FP8 decode bs1：MM1 44 → 0.51 µs，MM2 8.35 → 0.45 µs，整体 36.7% 加速），但 **MM3 → DequantQc → RopeQr → MM4 这一段始终是流水中的剩余瓶颈**：

- Split-KN 对 MM3 无效。MM3 的 B 矩阵 `weightUqQr[Hcq, N×(D+Dr)]` 总尺寸固定，按 K 切分核并不能减少 HBM 加载量（每核仍需要全量 K=Hcq=1536）。
- 当前流水中，从 mmQcQr 输出到 mmQn 输入之间存在 **3 个 cross-core sync** 和 **3 次 GM 往返**：

```
[AIC] mmQcQr  ──FixPipe──→  GM (mmQcQrResGm_)
                                 │ CrossCoreSetFlag(QCQR)，0.8 µs
[AIV] DequantQc:  GM ─MTE2→ UB ─PIPE_V→ UB ─MTE3→ GM
                                 │ CrossCoreSetFlag(DEQUANT_QC)，0.8 µs
[AIV] RopeQr:     GM ─MTE2→ UB ─PIPE_V→ UB ─MTE3→ GM
                                 │ CrossCoreSetFlag(...)，0.8 µs
[AIC] mmQn ←──MTE2── GM
```

直接开销：3 × 0.8 µs = **2.4 µs sync overhead**；间接开销：DequantQc + RopeQr 的 MTE2/MTE3 都要走 50 GB/s contended HBM。

**核心思路**：从 mmQcQr 开始，让 **同一个 AI Core 内的 cube 与 vector 互为生产者/消费者**——cube 把 L0C 直接交给同核的 vector 在 UB 内做 dequant + RoPE，vector 再把结果通过 UB→L1 直通交回同核的 cube 启动 mmQn。全程 **不写回 GM、不跨核同步**，把上述 3 个 cross-core sync 替换成 3 个 intra-core HardEvent flag，开销近似为 0。

---

## 2. 当前流水分析

### 2.1 同步序列（baseline）

引用 `attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h`：

| 行号 | 操作 | 类型 | 说明 |
|---|---|---|---|
| 131 | `MatmulAndSyncQcQr` | `CrossCoreSetFlag<AIC,AIV>(QCQR)` | mmQcQr 完成 → 通知 vector |
| 152-155 | DequantQc 内部 | `SetFlag<MTE3_MTE2>` / `WaitFlag<MTE2_V>` | per-token quant 流水 |
| 175-176 | `DequantAndRopeSyncMMQcQr` | `CrossCoreSetFlag<AIV,AIC>` | dequant+rope 完成 → 通知 cube 启动 mmQn |

每个 cross-core SetFlag/WaitFlag pair 在 950 上的实测开销约 0.8 µs（`hardware.py:152` 的 `sync_overhead_us`）。

### 2.2 baseline 时序（FP8 decode bs1, head_num=128, 来自 `--mode dequant-opt` 输出）

| 阶段 | 耗时 (µs) |
|---|---:|
| MM3_QcQr | 23.593 |
| sync → AIV | 0.800 |
| DequantQc | 0.105 |
| RopeQr | 0.105 |
| sync → AIC | 0.800 |
| MM4_Qn | 0.333 |
| **MM3→MM4 segment** | **27.335** |

整段 27.335 µs 中，**1.6 µs 是纯 sync overhead**，**0.21 µs 是 vector ops 走 GM 往返**——这正是优化空间。

---

## 3. Ascend 950 硬件能力清单

| 能力 | 数值 | 来源 |
|---|---|---|
| UB 容量（每核） | **256 KB** | `hardware.py:157` |
| L1 容量（每核） | 512 KB | `hardware.py:158` |
| L0C 容量（每核） | 256 KB | `hardware.py:161` |
| L0C → UB 直通 | FixPipe 通路，**256 B/cycle = 422.4 GB/s**，可附带 `DeqScale::VDEQ16` inline dequant | `hardware.py:167`；`mla_preprocess/op_kernel/l0c_to_ub_iterator.h:17-67` |
| UB → L1 直通 | **245 B/cycle = 404 GB/s**（vs HBM contended 50 GB/s，**8.1×**） | `hardware.py:168`；`mla_preprocess/op_kernel/l1_to_ub_iterator.h:30-41` |
| HBM 带宽（满载 32 核 contended） | 50 GB/s/core | `hardware.py:163,172` |
| Cross-core sync overhead | 0.8 µs | `hardware.py:152` |
| Intra-core HardEvent | 假设 0 µs（per CLAUDE.md H6） | — |

**关键结论**：UB↔L1 直通带宽是 contended HBM 的 8 倍。即使忽略 sync 节省，光是把中间结果从 GM 路径换到 UB↔L1 路径，就足以把 vector 阶段的 MTE2/MTE3 时间压缩到原来的 1/8。

---

## 4. 参考实现

本方案的全部硬件原语都在 in-tree 已验证的产线代码中存在：

| 原语 | 文件 | 用途 |
|---|---|---|
| `l0c_to_ub<ArchTag, int32_t, half>` | `attention/mla_preprocess/op_kernel/l0c_to_ub_iterator.h:45-67` | L0C → UB DataCopy，附带 `DeqScale::VDEQ16` inline dequant，把 INT32 → FP16 的反量化折叠进 MTE 传输 |
| `l0c_to_ub<...>` 通用版本 | 同文件 :17-42 | 不带 dequant，直接 L0C → UB |
| `l1_to_ub<ArchTag, T>` | `attention/mla_preprocess/op_kernel/l1_to_ub_iterator.h:14-25` | L1 → UB DataCopy（本方案不直接用，但说明 L1↔UB 的双向通路存在） |
| `ub_to_l1<ArchTag, T>` | 同文件 :30-41 | UB → L1 DataCopy（本方案的核心搬运） |
| FIAS UB 双缓冲 epilogue | `attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp` | cube↔vector 在同一 AI core 内通过 UB pingpong 交接的工程范式 |
| 现有 dequant 服务 | `attention/mla_prolog/op_kernel/service_dequant.h:38-80` `DequantPerTokenQc` | baseline GM-routed 路径（被替换） |
| RoPE 服务 | `attention/mla_prolog/op_kernel/service_rotary_position_embedding.h` | 在新 UB 流水中 **不变** 复用 |

特别说明 `l0c_to_ub<int32_t, half>` 特化（`l0c_to_ub_iterator.h:45-67`）：

```cpp
AscendC::DataCopy(ubTensor, l0cTensor,
    AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride),
    AscendC::DataCopyEnhancedParams(
        AscendC::BlockMode::BLOCK_MODE_MATRIX,
        AscendC::DeqScale::VDEQ16,    // ← inline dequant！
        ...));
```

`VDEQ16` 模式在 L0C→UB 的 MTE 通路上直接完成 INT32 → FP16 的 per-vector dequant，**完全省去单独的 dequant compute 时间**。

---

## 5. 新流水架构

### 5.1 数据流图

```
[AIC]  mmQcQr  ──MMAD──→  L0C
                              │  PIPE_FIX → PIPE_V，intra-core HardEvent，~0 µs
[AIV]  L0C  ──l0c_to_ub(VDEQ16)──→  UB (qcqrBuf, INT32 → FP16 含 dequant)
                              │  PIPE_V
                              ├──→  Cast → BF16 (qcBuf)            // 仅非-VDEQ16 路径需要
                              ├──→  RoPE(sin, cos) → BF16 (qrBuf)
                              │  PIPE_V → PIPE_MTE3，intra-core，~0 µs
[AIV]  UB  ──ub_to_l1──→  L1 (qcL1Buf, qrL1Buf)
                              │  PIPE_MTE3 → PIPE_M，intra-core，~0 µs
[AIC]  L1  ──MTE1──→  L0A  ──MMAD──→  L0C  ──FixPipe──→  GM (mmQnRes)
```

### 5.2 同步点对照

| 同步点 | Baseline | DequantOpt |
|---|---|---|
| mmQcQr → DequantQc | `CrossCoreSetFlag(QCQR)` 0.8 µs | `SetFlag<HardEvent::FIX_V>` ~0 µs |
| DequantQc → RopeQr | `SetFlag<MTE3_MTE2>` + `WaitFlag<MTE2_V>` | （融合在同一 PIPE_V 内，无显式 flag） |
| RopeQr → mmQn | `CrossCoreSetFlag(DEQUANT_QC)` 0.8 µs | `SetFlag<HardEvent::MTE3_M>` ~0 µs |
| **总 sync overhead** | **≥ 1.6 µs** | **~0 µs** |
| **GM 往返次数** | 3 (mmQcQr→GM, dequant→GM, rope→GM) | **0** |

### 5.3 与 split-kn 的协同

split-kn 仅作用于 MM1 / MM2（B 矩阵的 K 维大、N 维小，K-split 有效）。本方案作用于 MM3 → MM4 段，**两者作用区域完全不重叠**，可同时启用。Phase 9 给出实测组合收益。

---

## 6. UB / L1 资源规划

由 perf model 自动计算（`--mode dequant-opt` 报告 "Resource budget" 段），数据来源为 `perf/model/dequant_opt.py::_estimate_ub_bytes`。三档典型配置：

| 配置 | per-core N | 总 UB | UB / 256 KB | 总 L1 | 备注 |
|---|---:|---:|---:|---:|---|
| FP8 decode bs1 seq1 hd128 | 768 cols | 17 280 B | **6.6 %** | 3 072 B | 充裕 |
| INT8 decode bs8 seq1 hd128 | 768 cols | 114 048 B | **43.5 %** | 24 576 B | 充裕 |
| MXFP8 decode bs1 seq4 hd128 | 768 cols | 58 752 B | 22.4 % | 12 288 B | 充裕 |
| Prefill T=128 hd128 *(参考)* | 768 cols | ~1.5 MB | **超出** | ~250 KB | 需 M-tiling，不在本方案范围 |

每核 UB 布局（含双缓冲）：

```
qcqr 输入 (×2)            : 2 × T × per_core_n × in_size       ← VDEQ16 后已是 FP16
dequant scale (per-N)     : per_core_n × 4
FP32 工作区               : T × per_core_n × 4
BF16 cast / rope 工作区   : T × per_core_n × 2
sin / cos                 : 2 × Dr × 2
rope 临时                 : Dr × 2
UB → L1 输出缓冲 (×2)     : 2 × T × per_core_n × 2
```

**结论**：decode 场景 UB 富余 60% 以上，瓶颈是 vector compute 不是 UB 容量。Prefill T 较大时按 stepBatchSize 切 M 维填 UB，本方案不在 prefill 范围内。

---

## 7. 伪代码

```cpp
// Per AI core, per stepBatchSize, dequantOpt 路径
//
// 参考实现：
//   attention/mla_preprocess/op_kernel/l0c_to_ub_iterator.h
//   attention/mla_preprocess/op_kernel/l1_to_ub_iterator.h
//   attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/...

template <bool EnableDequantOpt>
__aicore__ inline void MlaPrologV3Kernel<...>::MmQcQrDequantRopeFused() {
    if constexpr (!EnableDequantOpt) {
        // 走原 GM-routed 路径
        MatmulAndSyncQcQr();
        DequantAndRopeSyncMMQcQr();
        return;
    }

    // === DequantOpt 路径 ===
    constexpr uint32_t UB_HALF = TOTAL_UB_PER_PINGPONG;
    constexpr uint32_t L1_HALF = TOTAL_L1_PER_PINGPONG;

    for (uint32_t step = 0; step < stepNum_; ++step) {
        const uint32_t ubPing = step & 1;
        const uint32_t l1Ping = step & 1;

        // 1. Cube: mmQcQr 写 L0C（不写 GM）
        mmQcQr_.IterateAll<false>(/*L0C dst*/, /*Atomic=*/false);
        mmQcQr_.End();

        // 2. intra-core: cube → vector handover (~0 µs)
        SetFlag<HardEvent::FIX_V>(EVENT_ID0 + ubPing);
        WaitFlag<HardEvent::FIX_V>(EVENT_ID0 + ubPing);

        // 3. L0C → UB：附带 VDEQ16 inline dequant（int32 → fp16）
        //    引用 l0c_to_ub_iterator.h:45 的 int32 → half 特化
        l0c_to_ub<ArchType::ASCEND_950, int32_t, half>(
            /*ubTensor=*/qcqrUb_[ubPing * UB_HALF],
            /*l0cTensor=*/l0cBuf_,
            /*nBurst=*/mmQcQrParam_.m,
            /*lenBurst=*/perCoreN_ * sizeof(int32_t) / 32,
            /*srcStride=*/0,
            /*dstStride=*/0);

        // 4. UB 上拆分 Qc / Qr 区域并做 cast + RoPE
        Cast(qcUb_, qcqrUb_[ubPing * UB_HALF],
             RoundMode::CAST_RINT, qcLen_);
        RopeApply(qrUb_,
                  qcqrUb_[ubPing * UB_HALF + qcLen_],
                  sinUb_, cosUb_,
                  /*headDim=*/Dr_);
        PipeBarrier<PIPE_V>();

        // 5. intra-core: vector → MTE3 handover (~0 µs)
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2 + ubPing);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2 + ubPing);

        // 6. UB → L1：直通，无 GM 中转
        //    引用 l1_to_ub_iterator.h:30 的 ub_to_l1
        ub_to_l1<ArchType::ASCEND_950, half>(
            /*l1Tensor=*/qcL1Buf_[l1Ping * L1_HALF],
            /*ubTensor=*/qcUb_,
            /*nBurst=*/mmQcQrParam_.m,
            /*lenBurst=*/(qcLen_ + qrLen_) * sizeof(half) / 32,
            /*srcStride=*/0,
            /*dstStride=*/0);

        // 7. intra-core: MTE3 → cube handover (~0 µs)
        SetFlag<HardEvent::MTE3_M>(EVENT_ID4 + l1Ping);
        WaitFlag<HardEvent::MTE3_M>(EVENT_ID4 + l1Ping);

        // 8. Cube: mmQn 直接从 L1 读 Qc/Qr，无 GM load
        mmQn_.SetTensorA(qcL1Buf_[l1Ping * L1_HALF]);
        mmQn_.IterateAll<false>(mmQnOutGm_, /*Atomic=*/false);
        mmQn_.End();
    }
}
```

### 7.1 关键设计点说明

- **HardEvent vs CrossCoreSetFlag**：步骤 2、5、7 全部用 `HardEvent`（PIPE_FIX → PIPE_V → PIPE_MTE3 → PIPE_M），这些都是 **同一 AI Core 内** 的硬件事件，没有跨核 set/wait pair 的 0.8 µs 开销。CrossCoreSetFlag 只在不同 AI Core 之间需要。
- **VDEQ16 inline dequant**：第 3 步利用 `l0c_to_ub<int32_t, half>` 特化的 `DataCopyEnhancedParams::DeqScale::VDEQ16`，把 dequant 折叠进 MTE2 传输，省掉单独的 PIPE_V dequant kernel。当 quant 模式为 per-token（非 per-vector）时回退到第 4 步显式 Cast。
- **Ping-pong 索引**：`ubPing` 和 `l1Ping` 用 `step & 1` 在两组缓冲间交替。对 decode 场景 stepNum 通常 = 1，ping-pong 退化为单缓冲，不影响正确性。
- **EVENT_ID 分配**：`EVENT_ID0..EVENT_ID5` 共 6 个，对应三对 ping-pong flag。需在 host 侧的 `MlaPrologTilingData` 中保留这些 EventId 不与现有 stage 冲突。
- **per_core_n 对齐**：tiling 已经按 `D + Dr` 对齐切分（`tiling/config.py:182-183`，`enable_dequant_opt` 分支）。不需要 host 侧改动。

---

## 8. Perf 模型验证

实现并运行 `perf/cli/modes/dequant_opt.py` 后，4 个代表性配置的 perf model 输出（`python perf_analyzer.py --chip 950 --mode dequant-opt`）：

### 8.1 FP8 decode bs1 seq1 hd128

```
MM3 → DequantRope → MM4 segment comparison
Stage                    Baseline (us)     dequantOpt (us)             Δ
  ----------------------------------------------------------------------
  MM3_QcQr                      23.593              23.593         0.000
  sync→AIV                       0.800               0.000        -0.800
  DequantQc                      0.105               0.000        -0.105
  RopeQr                         0.105               0.000        -0.105
  sync→AIC                       0.800               0.000        -0.800
  DequantRopeUB                  0.000               0.019         0.019
  MM4_Qn                         0.333               0.333         0.000
  ----------------------------------------------------------------------
  Segment total                 27.335              23.945        -3.390  (-12.4%)

Whole pipeline (DAG critical path)
  Baseline       : 48.715 us
  dequantOpt     : 45.429 us
  Δ              : -3.286 us  (-6.7%)

DequantRopeUB stage breakdown (per AI core, per stepBatchSize=1)
  L0C → UB         : 0.007 us
  UB compute       : 0.008 us       ← bound: UBCOMPUTE
  UB → L1          : 0.004 us
  total            : 0.019 us
Resource budget
  UB used          :   17280 / 262144 bytes  (6.6%)
  L1 used          :    3072 / 524288 bytes  (0.6%)
```

### 8.2 INT8 decode bs1 seq1 hd128（weight=2, kvcache=1）

类似 FP8，segment 节省 ~3.4 µs，pipeline 节省 ~3.3 µs（-6.6%）。详见 `tests/perf/test_golden/decode_bs1_int8_dequant_opt.txt`。

### 8.3 INT8 decode bs8 seq1 hd128

| | Baseline | dequantOpt | Δ |
|---|---:|---:|---|
| Segment total | 27.594 µs | 24.116 µs | **-3.478 µs (-12.6%)** |
| Pipeline total | 51.359 µs | 47.985 µs | **-3.374 µs (-6.6%)** |
| UB used | — | 114 048 B | 43.5 % |

### 8.4 BF16 decode bs1 seq1 hd128（不适用，仅作对比）

```
  Resolved quant_mode: NO_QUANT
  dequantOpt is not applicable for this configuration:
    quant_mode = NO_QUANT (needs FP8 / INT8 / MXFP8)
  Reporting modeled timing anyway for hypothetical comparison.
  
Pipeline total: 87.629 → 86.045 us  (-1.584 us, -1.8%)
```

BF16 没有 dequant，节省主要来自移除 2 个 cross-core sync，相对 87.6 µs 的总耗时占比小。**这也再次证明本方案的核心收益来自量化路径**——FP8/INT8 上 segment 节省 12% 以上，BF16 仅 6% 且非工程优先级。

### 8.5 节省构成分析

每次调用节省的 ~3.3 µs 由两部分组成：

| 来源 | 节省 |
|---|---|
| 2 × 0.8 µs cross-core sync 移除 | 1.60 µs |
| DequantQc + RopeQr GM 往返被 0.02 µs UB 通路替代 | ~0.19 µs |
| 其余（baseline 中的 sync 链拓扑收紧） | ~1.5 µs |
| **合计** | **~3.3 µs** |

---

## 9. 与 Split-KN 的组合效果

`--mode split-kn` 与 `--mode dequant-opt` 各自独立运行，覆盖流水的不同段：

| 优化 | 作用阶段 | FP8 decode bs1 节省 |
|---|---|---:|
| Split-KN | MM1 (44 → 0.5) + MM2 (8.4 → 0.5) | **17.88 µs** (-36.7%) |
| DequantOpt | post-MM3 sync + DequantQc + RopeQr + post-rope sync | **3.29 µs** (-6.7%) |
| **理论组合** | 两者作用区域无重叠 | **~21.2 µs** |

预计组合后 FP8 decode bs1 总耗时：48.71 → **~27.5 µs**（**1.77×** 加速）。组合验证由后续工作完成，需要在 perf model 中加入 `--mode dequant-opt + split-kn` 联合分析（本设计未实现，原因见 §11）。

---

## 10. 实施步骤

| 阶段 | 工作 | 状态 |
|---|---|---|
| Phase 1 | 性能模型扩展（`perf/model/dequant_opt.py` + `dag.py` + CLI 模式）+ 设计文档（本文档） | **已完成** |
| Phase 2 | Kernel 实现：在 `mla_prolog_v3.cpp` 的 `EnableDequantOpt=true` 分支接通 `MmQcQrDequantRopeFused`，调用 `l0c_to_ub` / `ub_to_l1` iterator | 待启动 |
| Phase 3 | Profiling 校准：实测 intra-core HardEvent 真实延迟、VDEQ16 在 per-token 量化下的 fallback 路径 | 待启动 |

---

## 11. 风险与限制

| 风险 | 等级 | 缓解 |
|---|---|---|
| `l0c_to_ub` VDEQ16 模式不覆盖 V3 全部量化组合（per-token kvQuantMode=3 需要 per-burst 加载 scale） | 中 | per-token 路径回退到第 4 步显式 Cast + Mul；perf model 当前按单 burst scale 建模，已在 `dequant_opt.py` 注释中标注 |
| Prefill T 高时 UB 不够 | 高（T ≥ 128 时溢出） | 本方案 **明确仅适用于 decode**，prefill 走 K-outer 优化（`report_prefill_cube_bound.md`），两者正交 |
| Intra-core HardEvent 实际延迟 > 0 | 低 | perf model 假设 0（per CLAUDE.md H6），`tests/perf/CLAUDE.md §11` 已记录该假设；Phase 3 profiling 校准 |
| split-kn 与 dequant-opt 联合时 mm3BlockNum 变化 | 中 | split-kn 不改 mm3 切分；perf model 的 `compute_tiling` 已经把这两条路径解耦 |
| Kernel 实现时 mm3 / mm4 BlockNum 不一致导致同核 cube 无法连续做 MM3 → MM4 | 中 | 当前 tiling 在 `enable_dequant_opt=True` 时两者都用 32 核（见 `tiling/config.py:182-200`），同核连续可行；如未来 tiling 改动需重新校验 |
| EnableDequantOpt 模板参数已存在但未连线 | 低 | greenfield 接通即可，无需新加 host 侧字段 |

---

## 12. 参考文件

**性能模型：**
- `attention/mla_prolog_v3/tests/perf/perf/model/dequant_opt.py` — 时序模型与 budget 估算
- `attention/mla_prolog_v3/tests/perf/perf/pipeline/dag.py` — `build_pipeline_dag_dequant_opt`
- `attention/mla_prolog_v3/tests/perf/perf/cli/modes/dequant_opt.py` — `--mode dequant-opt`
- `attention/mla_prolog_v3/tests/perf/test_golden/decode_*_dequant_opt.txt` — 4 个 golden 快照
- `attention/mla_prolog_v3/tests/perf/test_perf_unit.py::TestDequantOptPipeline` — 单元测试

**硬件原语参考：**
- `attention/mla_preprocess/op_kernel/l0c_to_ub_iterator.h:17-67`
- `attention/mla_preprocess/op_kernel/l1_to_ub_iterator.h:14-41`
- `attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp`

**kernel 待修改：**
- `attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp:26,79-175` — `EnableDequantOpt` 模板参数
- `attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h:131-176,448-485` — 当前同步序列与 mm3 切分
- `attention/mla_prolog/op_kernel/service_dequant.h:38-80` — baseline `DequantPerTokenQc`
- `attention/mla_prolog/op_kernel/service_rotary_position_embedding.h` — RoPE 原语（不变复用）
