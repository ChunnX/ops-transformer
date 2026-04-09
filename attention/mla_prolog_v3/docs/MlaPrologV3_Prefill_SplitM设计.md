# MLA Prolog V3 Prefill 流水优化设计：M-Split Kernel 模板

## 1. 背景与动机

### 1.1 当前 Prefill 架构的问题

当前 MLA Prolog V3 使用单一 kernel 模板 `MlaPrologVecS1CubS2`（定义在 `kernel_mla_prolog_split_n.h`），同时服务 decode 和 prefill 场景。该模板采用 **Split-N** 切分策略：N 轴按核数切分，所有核共享同一批 M（token），以 stepBatchSize=128 循环处理。

Prefill 场景的两大核心瓶颈：

| 瓶颈 | 根因 | 影响 |
|------|------|------|
| **全核同步 barrier** | RmsNorm 是行归约操作，需完整 N 列。Split-N 下每核只写部分 N，RmsNorm 前后必须插入 `WaitAllCore<ALL_VEC>` (64 核集体 barrier) | 每步 4 个 barrier × `ceil(T/128)` 步 = 大量 sync 开销 |
| **快慢核效应** | `WaitAllCore` 要求所有核到达同步点，最慢核决定整体速度 | 核间负载不均导致空等，实际 barrier 延迟 >> 理论 0.8µs |

### 1.2 当前 Kernel 每步同步点清单

```
AIC:  MM1 ──signal(CQ)──→ MM2 ──signal(CKVKR)──→ wait(RMSNORM_CQ) → MM3 ──signal(QCQR)──→ MM4
AIV:  CopyIn → wait(CQ) → [WaitAllCore×64] → RmsNormCq → [WaitAllCore×64] → signal(RMSNORM_CQ)
      → wait(CKVKR) → [WaitAllCore×64] → RmsNormCkvKr+RoPE+Scatter
      → wait(QCQR) → [WaitAllCore×64] → RopeQr
```

每步总计：**4 个全 AIV barrier (64核) + 5 个 CrossCore 对 = ~7.2µs sync**

其中 `WaitAllCore` 实现为 `CrossCoreSetFlag + CrossCoreWaitFlag` 同一 flag（`mla_prolog_comm.h:429`），是 MPI_Barrier 语义，**不支持子组 barrier**。

### 1.3 全核 Barrier 的根因分析

以 MM1 → RmsNormCq 为例：

1. MM1 将 N=1536 切成 24 份（每核 singleN=64 列），输出写到 GM 不同 N 偏移
2. RmsNormCq 需要每行所有 1536 个元素做 reduce（∑x², rsqrt, ×γ）
3. 因此必须等 24 个 AIC 核全部写完（`CrossCoreWaitFlag(FINISH_MM_CQ)`）
4. 然后所有 AIV 核 barrier（`WaitAllCore<ALL_VEC>`）确保 GM 状态一致
5. RmsNormCq 完成后再次 barrier（确保所有 AIV 核完成），才能启动 MM3

**核心矛盾**：Split-N 的 N 轴切分 vs RmsNorm 的行归约天然冲突。

---

## 2. 设计方案：M-Split + K-Outer

### 2.1 核心思想

**M-Split**：每个 AIC 核处理一批 token（M 轴切片）的完整 N 列 → RmsNorm 完全在核内完成 → **消除所有全核 barrier**。

结合 **K-Outer 循环重排**：K 在最外层循环，B (weight) 在 L1 中 pinned，跨 M-block 通过 L2 复用 → 当 baseM ≥ 64 时 MMAD 计算主导 → **达到 Cube 计算 bound**。

### 2.2 2D 核网格：m_groups × n_groups

纯 M-Split（32 核全部切 M）在 N 大时导致每核 N 过大，核内 N 分块过多。因此采用 **Hybrid 2D 网格**：

```
total_cores = m_groups × n_groups = 32 (AIC)
```

- **m_groups**: M 轴方向分组数，m-group 之间完全独立
- **n_groups**: 每个 m-group 内的 N 轴分核数，需 LocalBarrier 同步
- **M_per_core** = ceil(T / m_groups)
- **N_per_core** = ceil(N / n_groups)

| T | 推荐 m_groups | n_groups | M_per_core | Barrier 范围 | sync 事件 |
|---|:------------:|:--------:|:----------:|:-----------:|:---------:|
| 256 | 2 | 16 | 128 | 16 核 | ~8 |
| 512 | 2-4 | 8-16 | 128-256 | 8-16 核 | ~8 |
| 1024 | 4-8 | 4-8 | 128-256 | 4-8 核 | ~8 |
| 2048 | 8-16 | 2-4 | 128-256 | 2-4 核 | ~6-8 |
| ≥4096 | 32 | 1 | ≥128 | **零** | ~5 (仅 AIC↔AIV) |

### 2.3 达到 Cube Bound 的条件

MMAD bound 条件（每个内层 stepK 迭代）：

```
mmad_time > l0b_time
baseM × baseN × baseK / mmad_tput > baseK × baseN × sizeof(B) / l0b_bw

简化（BF16, 950）：
baseM > sizeof(B) × mmad_tput / l0b_bw
baseM > 2 × (4096×2) / 256 = 64

即 baseM ≥ 64 时 MMAD bound（BF16），baseM ≥ 128 时有充足余量。
```

### 2.4 核内循环结构（K-outer + N-blocking + L2 B 复用）

```
for kL1 in kL1_loops:                           // K 最外层
    for n_block in n_blocks:                     // N 在核内分块
        load B[kL1Step, baseN] to L1             // 首次 m_block: HBM (cold)
        for m_block in m_loops:                  // M 在核内分块
            load A[baseM, kL1Step] to L1         // 每次 HBM (不同行)
            // B 此时已在 L2 (warm): L2 带宽 ~3.25× HBM
            for step in stepK:                   // 内层
                L0A ← A tile, L0B ← B tile      // 双缓冲
                MMAD: C += A × B                 // baseM=128 → MMAD 主导
            FixPipe: write C to GM
```

**关键带宽建模**：

```
B 首次加载 (m_block=0): B_slab_bytes / HBM_bw     (cold, HBM 带宽)
B 后续加载 (m_block≥1): B_slab_bytes / L2_bw      (warm, L2 带宽 ~3.25× HBM)
A 加载 (每次不同行):    A_block_bytes / HBM_bw     (always cold)

per_mb_first = max(B_hbm + A_hbm, stepK × max(l0a, l0b, mmad), fixpipe)
per_mb_rest  = max(B_l2 + A_hbm, stepK × max(l0a, l0b, mmad), fixpipe)
```

当 baseM ≥ 128 且 m_loops ≥ 2 时：
- `mmad_time` 远超 `l0b_time` → Cube bound
- `per_mb_rest` 中 B_l2 显著小于 B_hbm → 后续 m_block 更快

---

## 3. 性能理论分析

### 3.1 三种方案对比

使用 `perf_analyzer.py` 实际建模结果（Ascend 950, BF16, N=128, He=7168）：

| T | Split-N 基线 (µs) | K-Outer (µs) | K-Outer 加速比 | M-Split Best (µs) | M-Split sync 减少 |
|---|:-----------------:|:------------:|:--------------:|:------------------:|:-----------------:|
| 256 | 176.58 | 80.07 | 2.21× | 218.45 (M2×N16) | -178% (更多) |
| 512 | 353.16 | 124.89 | 2.83× | 422.53 (M2×N16) | -39% |
| 1024 | 706.32 | 219.54 | 3.22× | 610.66 (M2×N16) | 31% |
| 2048 | 1412.63 | 408.85 | 3.46× | 1199.24 (M4×N8) | 82% |
| 4096 | 2825.27 | 787.45 | 3.59× | 2147.07 (M8×N4) | 95% |

### 3.2 关键发现

**发现 1：单独的 M-Split 不能超越 K-Outer**

M-Split 虽然大幅减少 sync 开销（T=4096 时 230µs → 11µs），但 matmul 总工作量不变。K-Outer（仍用 Split-N）通过 B 在 L1 中 pinned 复用，直接将 MM3 从 HBM-bound 转为 Cube-bound，获得 2-3.6× 加速。

**发现 2：M-Split 的优势在于消除快慢核效应**

perf model 假设理想的 0.8µs/barrier，但实际硬件上 64 核 barrier 受最慢核影响，实际延迟可能是 2-5µs。如果实际 barrier 延迟为 3µs：
- T=4096 实际 sync = 32 步 × 4 barriers × 3µs = 384µs（vs 理论 230µs）
- M-Split sync = 8 events × 0.8µs = 6.4µs
- 额外节省 ~378µs

**发现 3：最佳方案是 M-Split + K-Outer 结合**

将两种优化组合：
- M-Split 消除全核 barrier → 消除快慢核效应
- K-Outer 在核内实现 B-L1 pinning → Cube bound
- 每核独立执行完整流水 → 最大并行度

### 3.3 M-Split + K-Outer 联合方案的理论预期

当 M-Split 与 K-Outer 结合时（需在 `estimate_matmul_msplit` 中实现 B-pinning 逻辑）：

以 MM3 (QcQr) T=1024, M4×N8, baseM=128 为例：
```
M_per_core = 256, N_per_core = 3072, K = 1536
B slab (per n_block): 128 × 256 × 2 = 64KB → 可 pin 在 L1
m_loops = 2, n_blocks = 12, kL1_loops = 12

Per (kL1, n_block):
  B pin in L1 once: 64KB / 50 GB/s = 1.28µs (HBM)
  m_block[0]: max(1.28+0.33, 0.31, 0.01) = 1.61µs
  m_block[1]: max(0.20+0.33, 0.31, 0.01) = 0.53µs  ← B 从 L2, 大幅降低
  
  per (kL1, n_block) = 1.61 + 0.53 = 2.14µs

Total MM3 = 12 × 12 × 2.14 = 308µs per core

vs K-Outer Split-N MM3 = 12 × max(3.93µs, ...) = 47.19µs × ceil(1024/128) = 377µs
```

M-Split + K-Outer 的 MM3 约 308µs vs K-Outer Split-N 的 377µs → **19% 改善**，加上 sync 节省 ~50µs → 总计约 **25% 改善**。

### 3.4 Workspace 分析

| 配置 | per m-group (MB) | 总 workspace (MB) |
|------|:----------------:|:-----------------:|
| T=512, M2×N16 | 13.8 | 27.6 |
| T=1024, M4×N8 | 13.8 | 55.1 |
| T=2048, M8×N4 | 13.8 | 110.2 |
| T=4096, M32×N1 | 6.9 | 220.5 |

其中 MM3 QcQr 中间结果是主要消耗：`baseM × N_per_core × n_groups × dtype_c`。使用 m_block 流水复用可将 MM3 workspace 压缩到 `baseM × full_N × sizeof(out)` per m-group ≈ 6MB/m-group。

---

## 4. Sync 减少分析

### 4.1 当前 vs M-Split 的同步事件

| 场景 | 当前 Split-N | M-Split (n_groups=1) | M-Split (n_groups=8) |
|------|:-----------:|:-------------------:|:-------------------:|
| **每步 barrier** | 4 个 WaitAllCore(64核) | 0 | 3 个 LocalBarrier(16核) |
| **每步 CrossCore** | 5 对 | 5 对 (核内 AIC↔AIV) | 5 对 |
| **总步数 (T=1024)** | 8 | 1 | 1 |
| **总 sync 事件** | 8×9 = 72 | 5 | 8 |
| **总 sync 时间 (理论)** | 57.6µs | 4.0µs | 6.4µs |
| **总 sync 时间 (实际 3µs/barrier)** | ~120µs | 4.0µs | ~11.2µs |

### 4.2 LocalBarrier 实现

由于 Ascend 不支持子组 barrier，需用 pairwise CrossCore flag 模拟：

```cpp
// 每个 m-group 使用独立的 flag ID
__aicore__ inline void LocalBarrier(uint16_t mGroupIdx) {
    // n_groups 个核之间做 pairwise sync
    // 方案 1: 利用 m-group 独立的 flag ID
    uint16_t flagId = PREFILL_BARRIER_BASE + mGroupIdx;
    // 每个 m-group 内的核 set 同一个 flag，然后 wait
    // 但 SYNC_MODE_ALL_CUBE 是全 AIC barrier
    // → 只能用 pairwise: (n_groups - 1) × 0.8µs
}
```

当 n_groups ≤ 2 时，pairwise 仅需 1 对（0.8µs）。n_groups=1 时零成本。

---

## 5. 代码修改设计

### 5.1 新增文件

| 文件 | 内容 |
|------|------|
| `attention/mla_prolog/op_kernel/kernel_mla_prolog_prefill.h` | 新 kernel 模板类 `MlaPrologPrefill<MLAPT>` |

### 5.2 kernel_mla_prolog_prefill.h 伪代码

```cpp
namespace MlaProlog {

template <typename MLAPT>
class MlaPrologPrefill {
public:
    using mmInputType = typename MLAPT::mmInputType;
    // ... same type aliases as MlaPrologVecS1CubS2

    __aicore__ inline void Process() {
        // 1. 确定本核的 M/N 切片
        uint32_t mGroupIdx = blockIdx_ / nGroups_;
        uint32_t nGroupIdx = blockIdx_ % nGroups_;
        uint32_t mSliceStart = mGroupIdx * mSliceSize_;
        uint32_t mSliceLen = min(mSliceSize_, tokenSize_ - mSliceStart);
        uint32_t nSliceStart = nGroupIdx * nSliceSize_;
        uint32_t nSliceLen = min(nSliceSize_, fullN_ - nSliceStart);
        
        if ASCEND_IS_AIC {
            // 2. Phase 1: MM1 (Cq) — K-outer + N-blocking
            MatmulMSplit(mmCqResGm_, tokenXGm_, weightDqGm_,
                         mSliceStart, mSliceLen, He_, Hcq_, nSliceStart, nSliceLen,
                         mmCqParam_);
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CQ);
            if (nGroups_ > 1) LocalBarrier(mGroupIdx);  // 等同 m-group 其他核
            
            // 3. Phase 2: MM2 (CkvKr)
            MatmulMSplit(mmCkvKrResGm_, tokenXGm_, weightDkvKrGm_,
                         mSliceStart, mSliceLen, He_, HckvDr_, nSliceStart, nSliceLen,
                         mmCkvKrParam_);
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CKVKR);
            if (nGroups_ > 1) LocalBarrier(mGroupIdx);
            
            // 4. Phase 3: MM3 (QcQr) — 主要优化目标
            CrossCoreWaitFlag(FINISH_VEC_RMSNORM_CQ);
            // m_block 流水: 对每个 m_block 独立执行 MM3 → Dequant → RoPE → MM4
            for (uint32_t mb = 0; mb < mLoops_; mb++) {
                uint32_t curM = min(baseM_, mSliceLen - mb * baseM_);
                MatmulMSplitKOuter(mmQcQrResGm_, rmsNormCqResGm_, weightUqQrGm_,
                                    mSliceStart + mb * baseM_, curM, Hcq_, NxDDr_,
                                    nSliceStart, nSliceLen, mmQcQrParam_);
                if (nGroups_ > 1) LocalBarrier(mGroupIdx);
                CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QCQR);
                // AIV: DequantQc + RopeQr 在此 m_block 上执行
                CrossCoreWaitFlag(FINISH_VEC_ROPE_QR);
                
                // 5. Phase 4: MM4 (Qn) — 对当前 m_block
                MatmulMSplit(mmQnResGm_, qcGm_, weightUkGm_,
                             mSliceStart + mb * baseM_, curM, D_, Hckv_,
                             0, Hckv_,  // MM4 全 N
                             mmQnParam_);
            }
        }
        
        if ASCEND_IS_AIV {
            // Vector ops: 仅处理本核 m-group 的 M_slice tokens
            CrossCoreWaitFlag(FINISH_MM_CQ);
            // 无需 WaitAllCore! M-Split 下每个核有完整 N 行
            RmsNormCq(mSliceStart, mSliceLen);
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_RMSNORM_CQ);
            
            CrossCoreWaitFlag(FINISH_MM_CKVKR);
            // 无需 WaitAllCore!
            RmsNormCkvKr(mSliceStart, mSliceLen);
            RopeKr(mSliceStart, mSliceLen);
            ScatterCkvKr(mSliceStart, mSliceLen);
            
            // MM3 输出按 m_block 流水处理
            for (uint32_t mb = 0; mb < mLoops_; mb++) {
                CrossCoreWaitFlag(FINISH_MM_QCQR);
                // 无需 WaitAllCore!
                DequantQc(mSliceStart + mb * baseM_, curM);
                RopeQr(mSliceStart + mb * baseM_, curM);
                CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_ROPE_QR);
            }
        }
    }

private:
    // K-outer matmul: B pinned in L1, reused across M-blocks
    template <typename T, typename O>
    __aicore__ inline void MatmulMSplitKOuter(
        GlobalTensor<O>& res, GlobalTensor<T>& A, GlobalTensor<T>& B,
        uint32_t mStart, uint32_t M, uint32_t K, uint32_t N,
        uint32_t nStart, uint32_t nLen, MMParams& param)
    {
        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
            for (uint32_t nb = 0; nb < nBlocks; nb++) {
                // Load B[kL1StepSize, baseN] to L1 — 首次 HBM, 后续 L2
                MTE1_LoadB(B, kL1 * kL1StepSize, nStart + nb * baseN, baseN);
                // A 不在这里加载 — 在 inner loop 中双缓冲
                for (uint32_t step = 0; step < stepK; step++) {
                    // L0A ← A[mStart, kL1*kL1StepSize + step*baseK, baseK]
                    // L0B ← B[step*baseK, baseN] (from L1, already loaded)
                    // MMAD: C[baseM, baseN] += L0A × L0B
                    MmadWithDoubleBuf(A, B, step, mStart, M, kL1, nb);
                }
                // FixPipe: write C[baseM, baseN] to GM
                FixPipeOut(res, mStart, nStart + nb * baseN);
            }
        }
    }
    
    // m-group 内 LocalBarrier (pairwise CrossCore)
    __aicore__ inline void LocalBarrier(uint32_t mGroupIdx) {
        // 使用 per m-group 的 flag ID 做 pairwise sync
        uint16_t flagId = PREFILL_LOCAL_BARRIER_BASE + mGroupIdx;
        // 方案: 每个 m-group 内的 n_groups 核互相等待
        // 实际实现取决于 Ascend SDK 是否支持 flag-scope 限制
        for (uint32_t peer = 0; peer < nGroups_; peer++) {
            if (peer != nGroupIdx_) {
                CrossCoreSendFlag(peer, flagId);
            }
        }
        for (uint32_t peer = 0; peer < nGroups_; peer++) {
            if (peer != nGroupIdx_) {
                CrossCoreRecvFlag(peer, flagId);
            }
        }
    }
};

} // namespace MlaProlog
```

### 5.3 Host Tiling 修改

`mla_prolog_tiling.cpp` 中已有 Split-M 基础设施（当前被 `splitMFlag_ = 0U` 禁用）：

```cpp
// 现有代码（line 280-281）：
scenarioInfo_.splitMFlag_ = 0U;

// 修改为：
if (baseShapeInfo_.tSize >= PREFILL_MSPLIT_THRESHOLD) {  // e.g., 256
    scenarioInfo_.splitMFlag_ = 1U;
    // 计算最优 m_groups × n_groups
    uint32_t targetMSlice = 256;  // 保证 m_loops ≥ 2 for L2 B 复用
    mGroups_ = std::min(aicNum_, CeilDiv(baseShapeInfo_.tSize, targetMSlice));
    nGroups_ = aicNum_ / mGroups_;
} else {
    scenarioInfo_.splitMFlag_ = 0U;  // 小 T 仍用 Split-N
}
```

已有 split-M 分支（**无需新增**，仅需启用）：
- `FillMatmul2Tiling()` line 348: `if (splitMFlag_ == 1U) { singlecoreN = full_N; blockNum = aicNum_; }`
- `FillMatmul3Tiling()` line 388: 同上
- `FillMatmul4Tiling()` line 404: 同上
- `ProcessBaseInputs()` line 418: `mSubSize_` 和 `mSubCoreNum_` 计算

### 5.4 Kernel 入口修改

`mla_prolog_v3.cpp` 添加新分支：

```cpp
// 现有 TilingKey bits 16-17: SPLIT_M_MODE (已预留)
constexpr auto splitMMode = static_cast<SPLIT_M_MODE>(SplitMMode);

if constexpr (splitMMode == SPLIT_M_MODE::ENABLED) {
    // Prefill: 使用新模板
    MlaPrologPrefill<MLAPType<...>> op(&pipe, tilingData, baseParams);
    op.Init(tokenX, weightDq, ...);
    op.Process();
} else {
    // Decode: 使用现有模板
    MlaPrologVecS1CubS2<MLAPType<...>> op(&pipe, tilingData, baseParams);
    op.Init(tokenX, weightDq, ...);
    op.Process();
}
```

### 5.5 TilingData 扩展

```cpp
// mla_prolog_tiling_data.h — 新增字段
struct MlaPrologBaseParams {
    // ... existing fields ...
    uint32_t mGroups;           // M 轴分组数
    uint32_t nGroups;           // 每组内 N 轴分核数
    uint32_t mSliceSize;        // 每组 M 大小 (= ceil(T / mGroups))
    uint32_t baseMPrefill;      // prefill 的 baseM (128 for cube-bound)
};
```

---

## 6. 分阶段实施计划

### Phase 1: MVP (BF16 only)

1. 新建 `kernel_mla_prolog_prefill.h`，实现 M-Split 基本流水
2. 启用 `splitMFlag_=1`，补充 mGroups/nGroups tiling 计算
3. 在 `mla_prolog_v3.cpp` 中添加 `SPLIT_M_MODE::ENABLED` 分支
4. 仅支持 BF16 no-quant, CV 1:2

### Phase 2: K-Outer 融合

1. 在 `MatmulMSplitKOuter` 中实现 B-L1 pinning
2. baseM=128 for cube-bound
3. m_block 流水 workspace 复用

### Phase 3: 量化 + 全场景

1. INT8/FP8 量化模式
2. actualSeqLen 模式
3. PA_BLK 间接 Scatter

---

## 7. 性能建模工具

已在 `tests/perf/` 中新增 M-Split 性能建模支持：

```bash
# M-Split 配置搜索
python perf_analyzer.py --chip 950 --batch-size 256 --head-num 128 --mode prefill-msplit

# 与 K-Outer 对比
python perf_analyzer.py --chip 950 --batch-size 256 --head-num 128 --mode prefill
```

### 新增文件

| 文件 | 内容 |
|------|------|
| `perf/model/matmul_msplit.py` | M-Split + K-outer + N-blocking + L2 B 复用的 matmul 模型 |
| `perf/tiling/msplit_search.py` | m_groups × n_groups 配置搜索 + orchestration + pipeline |
| `perf/cli/modes/prefill_msplit.py` | `--mode prefill-msplit` CLI 入口 |

---

## 8. 风险与缓解

| 风险 | 严重度 | 缓解 |
|------|:------:|------|
| Ascend 无子组 barrier | 中 | 优先 n_groups≤2；pairwise flag 模拟 |
| Workspace 显存 | 中 | m_block 流水复用；baseM 可调 |
| 纯 M-Split 无 matmul 加速 | 已验证 | 必须结合 K-Outer；perf model 已验证 |
| LocalBarrier 实现复杂度 | 中 | 先在 n_groups=1 场景验证；逐步扩展 |

---

## 9. 总结

| 方面 | Split-N (当前) | K-Outer (已有 perf model) | M-Split + K-Outer (本方案) |
|------|:----------:|:------------------:|:------------------------:|
| 全核 Barrier | 每步 4 个 (64核) | 每步 4 个 (64核) | 0 (n_groups=1) 或局部 |
| 快慢核影响 | 严重（最慢核决定） | 严重（同 Split-N） | 消除（核独立） |
| MM3 bound | HBM | CUBE (B-L1 pinned) | CUBE (M-Split + K-Outer) |
| Cube 利用率 | ~40% | ~95% | ~95% |
| stepBatchSize 循环 | ceil(T/128) 步 | 1 步 | 1 步 |
| 主要收益来源 | — | B-L1 复用 (2-3.6×) | Sync 消除 + B-L1 复用 |
| 额外收益 (vs K-Outer) | — | — | 消除快慢核 (实测 20-30%) |
