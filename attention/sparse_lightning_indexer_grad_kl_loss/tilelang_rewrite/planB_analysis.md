# Plan B 架构分析与瓶颈诊断

## 1. 总体架构

```
输入 (13个tensor)
  │
  ├─ Python Preprocess ──────────────────────────────────────┐
  │   • S2 pad to 256-align                                   │
  │   • Gather Key/KeyIndex/KeyRope by sparseIndices          │
  │   • RoPE dim pad to BLOCK_K=128 align                      │
  │   • gather_key_index_t = transpose(gather_key_index)       │
  │                                                           │
  ├─ Kernel1 (C→V, K-parallel) ──────────────────────────────┤
  │   C1: MM1(Q||Qrope @ GK||GKrope^T) + MM2(QI @ GKI^T)     │
  │   V1: P_reduced(chunk) + ReduceSumY(chunk)                 │
  │   Output: bmm1Res, bmm2Res, PReduced, ReduceSumY          │
  │                                                           │
  ├─ Kernel2 (V1-only, K-parallel) ──────────────────────────┤
  │   V1: SoftmaxI(global) + diff + dWeight(partial)          │
  │        + reluGrad + Loss(partial)                          │
  │   Output: dWeightPartial, reluGradRes, LossPartial         │
  │                                                           │
  ├─ Kernel3 (C→V, K-parallel) ──────────────────────────────┤
  │   C: MM6 = reluGrad @ GKIT, atomic_add to dQueryIndex     │
  │   V: barrier (dummy)                                      │
  │   Output: dQueryIndex                                     │
  │                                                           │
  └─ Python Postprocess ─────────────────────────────────────┘
      • dWeight = sum(dWeightPartial, dim=2), cast
      • Loss = sum(LossPartial), cast
      • dQueryIndex = dQueryIndex_ws.to(fp16)
      • dKeyIndex = reluGrad^T @ QI → scatter_add_
```

## 2. Tiling 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `BLOCK_M` | 16 | S1 维度 tile（每 core 处理的序列行数） |
| `BLOCK_N` | 128 | K 维度 tile（每 core 处理的 key 位置数） |
| `BLOCK_K` | 128 | D 维度 tile（matmul 内维分块） |
| `BLOCK_N1` | 32 | N1/head 维度 tile |
| `BLOCK_NIDX1` | 16 | Nidx1/index_head 维度 tile |

**K-parallel 分解**：`block_num = B × ceil(S1/BLOCK_M) × (K/BLOCK_N)`

默认配置下 K=2048, BLOCK_N=128 → `n_num = 16` 个 core 并行处理不同的 K-block。

## 3. Kernel1 详细分析

### 3.1 C1 Scope（Cube 单元）

```
内存分配：
  Q_L1:     (N1, BLOCK_K) = (128, 128) = 32KB FP16     ← Query/QueryRope
  GK_L1:    (BLOCK_N, BLOCK_K) = (128, 128) = 32KB FP16 ← GatherKey/GatherKeyRope
  C_L0C:    (N1, BLOCK_N) = (128, 128) = 64KB FP32      ← MM1 结果
  QI_L1:    (Nidx1, BLOCK_K) = (64, 128) = 16KB FP16    ← QueryIndex
  GKI_L1:   (BLOCK_N, BLOCK_K) = (128, 128) = 32KB FP16 ← GatherKeyIndex
  C2_L0C:   (Nidx1, BLOCK_N) = (64, 128) = 32KB FP32    ← MM2 结果
  总 L1:    32+32+16+32 = 112KB
  总 L0C:   64+32 = 96KB（复用，同时只用一块）
```

**MM1 计算**（`transpose_B=True`, `T.Pipelined(num_stages=3)`）：
```
Q(N1, k_block) @ GK^T(k_block, BLOCK_N) → (N1, BLOCK_N)
内维 = DQuery + d_rope = 512 + 128 = 640
分块: d_num_total = (512+128)/128 = 5 个 D-tile
每次迭代: (128,128) @ (128,128) → (128,128), L0C 累加
5 次迭代后 copy 到 bmm1Res[:, :, 0:N1, k_start:k_end]
```

**MM2 计算**（`transpose_B=True`）：
```
QI(Nidx1, k_block) @ GKI^T(k_block, BLOCK_N) → (Nidx1, BLOCK_N)
内维 = DQueryIndex = 128, 分块数 = 1
单次 gemm: (64,128) @ (128,128) → (64,128)
copy 时 enable_relu=True: ReLU(BMM2) 直接写入 bmm2Res
```

### 3.2 V1 Scope（Vector 单元）— 串行按 s1_row

```
UB 分配（主要）：
  bmm1_ub:   (BLOCK_N1, n_block) = (32,128) × 4B = 16KB  FP32
  bmm2_ub:   (BLOCK_NIDX1, n_block) = (16,128) × 4B = 8KB
  weight_ub: (Nidx1,) = 64 × 4B = 0.25KB
  p_chunk_ub, rsy_chunk_ub, sm_max/sum_ub, eps_ub 等: ~2KB
  估算总 UB: ~30-40KB
```

**Pass 1 — P_reduced**：对每个 N1 tile (4 个)，读 bmm1Res → softmax → reduce_sum
```
for n1_c in 0..3:
    bmm1_ub = bmm1Res[:, :, n1_s:n1_s+32, k_s:k_e]
    bmm1_ub *= scale_value
    bmm1_ub[g_i] -= sm_max[n1_s+g_i]  ← per-row serial
    exp(bmm1_ub)
    bmm1_ub[g_i] /= sm_sum[n1_s+g_i]  ← per-row serial
    reduce_sum(bmm1_ub, dim=0) → 累加到 p_chunk_ub
p_chunk_ub *= 1/N1
copy → PReduced[:, :, k_start:k_end]
```

**Pass 2 — ReduceSumY**：对每个 Nidx1 tile (4 个)，读 bmm2Res → mul weights → reduce_sum
```
for nidx_c in 0..3:
    bmm2_ub = bmm2Res[:, :, nidx_s:nidx_s+16, k_s:k_e]
    bmm2_ub[g_i] *= weight[g_i]  ← per-row serial
    reduce_sum(bmm2_ub, dim=0) → 累加到 rsy_chunk_ub
copy → ReduceSumY[:, :, k_start:k_end]
```

### 3.3 Kernel1 瓶颈

- **per-row serial 循环**：`for g_i in T.serial(BLOCK_N1)` 将 N1=128 拆成 128 次标量操作，无法利用 Vector 单元并行
- **T.barrier_all()**：在 tiling 循环间存在冗余同步
- **C→V 同步**：flag 同步引入硬件流水线断流

## 4. Kernel2 详细分析

### 4.1 V1 Scope（Vector 单元）

```
UB 分配（关键）：
  si_all_ub:  (K,) = 2048 × 4B = 8KB       ← 全局 SoftmaxI workspace
  bmm2_ub:    (16,128) × 4B = 8KB
  dw_total_ub:(64,) × 4B = 0.25KB
  rg_chunk_ub:(16,128) × 4B = 8KB
  rg_fp16_ub: (16,128) × 2B = 4KB
  si 临时:    ~2KB
  估算总 UB: ~35-40KB
```

**计算流程（per s1_id）**：

**Step 1 — 全局 SoftmaxI**（冗余计算，每个 core 都算全量）：
```
si_all = ReduceSumY[:, s1_id, :]          ← copy 完整 K=2048
si_max = reduce_max(si_all)
si_all = exp(si_all - si_max)
si_sum = reduce_sum(si_all), clamp(1e-8)
si_all /= si_sum                          ← global SoftmaxI
```

**Step 2 — diff**：
```
p_chunk = PReduced[:, s1_id, k_s:k_e]    ← 本 K-block
si_chunk = si_all[k_s:k_e]               ← 对应 SoftmaxI chunk
diff = si_chunk - p_chunk                 ← (BLOCK_N,) = (128,)
```

**Step 3 — dWeight + reluGrad**（`T.Pipelined(num_stages=3)`）：
```
for nidx_c in Pipelined(0..3):
    bmm2_ub = bmm2Res[:, s1_id, nidx_s:nidx_s+16, k_s:k_e]
    
    # dWeight: diff ⊙ bmm2 → reduce_sum over K-block
    rg_chunk[g_i] = diff * bmm2[g_i]          ← per-row serial (16次)
    dw_chunk = reduce_sum(rg_chunk, dim=-1)    ← (16,)
    dw[16:32] += dw_chunk
    
    # reluGrad: weight ⊙ diff ⊙ bmm2 > 0 ? : 0
    rg_chunk[g_i] = diff * weight[g_abs] * bmm2[g_i]  ← per-row (16次)
    mask = bmm2 > 0
    rg_chunk = select(mask, rg_chunk, 0)
    cast FP32→FP16(CAST_ROUND) → reluGradRes[:, s1_id, nidx_s:nidx_s+16, k_s:k_e]
```

**Step 4 — Loss**：
```
kl_local = max(p_chunk, 1e-8)
log_p = ln(kl_local)
log_si = ln(max(si_chunk, 1e-8))
loss = p_chunk * (log_p - log_si)
loss_partial = reduce_sum(loss)
copy → LossPartial[:, s1_id, n_idx]
```

### 4.2 Kernel2 瓶颈

- **K=2048 的 `si_all_ub`**：8KB UB 占用，且每个 core 都要 copy 完整的 ReduceSumY(K=2048)
- **全局 SoftmaxI 冗余计算**：16 个 core 各自独立计算相同的 softmax，浪费 15/16 的计算
- **per-row serial**：`T.serial(BLOCK_NIDX1)` × `T.serial(BLOCK_N1)` 等标量循环
- **BMM2 的 4 次 Pipelined 循环**：每次 copy 2D tile (16×128)，总共 4×16×128×4B = 32KB DMA

## 5. Kernel3 详细分析（当前精度异常）

### 5.1 设计意图

将 Python 中的 `torch.bmm(reluGrad, GKI)` 移入 C→V kernel，用 `T.tile.atomic_add(L0C→Global)` 做跨 K-block 累加。

### 5.2 C Scope

```
内存：
  RG_L1:   (BLOCK_NIDX1, BLOCK_N) = (16,128) × 2B = 4KB   FP16
  GKIT_L1: (BLOCK_K, BLOCK_N) = (128,128) × 2B = 32KB     FP16
  DQ_L0C:  (BLOCK_NIDX1, BLOCK_K) = (16,128) × 4B = 8KB   FP32

计算（per s1_row, per nidx_c, per dq_c）：
  RG(16,128) @ GKIT^T(128,128) → (16,128) in L0C
  atomic_add → dQueryIndex[:, s1, nidx_s:nidx_s+16, dq_s:dq_s+128]
```

### 5.3 当前问题

```
[DEBUG] Kernel3 vs Python bmm: max_diff=<待测试>
[DEBUG] Kernel3: inf=False/True nan=False min=-7.6285e+13/?
```

**现象**：
- 首次调用：值在 ±10^13 量级（正常应为 ±10^3）
- 后续调用：累积溢出到 ±inf
- relu_grad 和 gather_key_index_t 均为正常值（无 inf/nan）

**疑似根因**：
1. `d_query_index_ws.zero_()` 未生效或 `func3` 返回了不同的 tensor 对象
2. `T.gemm_v0` 计算结果数量级错误（transpose_B 语义差异？）
3. `T.tile.atomic_add` 写入地址错位

**V→C 尝试失败**：最初尝试在 Kernel2 内加 C1 scope（V1→C1 方向），触发 "Illegal instruction, unaligned UUB addresses"，说明 Ascend 硬件不支持 V→C 方向的 cross_flag 同步（或 TileLang 对此支持不完善）。

## 6. Python 部分分析

### 6.1 预处理（每调用一次）

```python
# S2 pad
key = F.pad(key, (0,0, 0,0, 0,S2_pad-S2, 0,0))       # if S2_pad > S2

# Gather: fused cat+gather (减少 gather 次数)
key_all = cat([key, key_index, key_rope], dim=-1)       # (B,S2,1,DQ+DQI+d_rope)
idx_all = sparse_idx.expand(B, S1*K, total_dim)
gather_all = gather(key_all, idx_all).reshape(B,S1,K,total_dim)
gather_all *= valid_mask
gather_key, gather_key_index, gather_key_rope = split(gather_all)

# RoPE pad
query_rope = F.pad(query_rope, (0, pad_len))
gather_key_rope = F.pad(gather_key_rope, (0, pad_len))
```

### 6.2 后处理（每调用一次）

```python
# dWeight: sum partials
d_weight = d_weight_partial.sum(dim=2)    # (B,S1,n_num,Nidx1)→(B,S1,Nidx1)
d_weight = d_weight.to(fp16)

# Loss: sum partials
kl_loss = loss_partial.sum().reshape(1).to(fp32)

# dQueryIndex: cast (Kernel3 已计算 → FP32 workspace)
d_query_index = d_query_index_ws.to(fp16)

# dKeyIndex: 2次 bmm + scatter_add
relu_grad_t = relu_grad.permute(0,1,3,2)  # (B,S1,K,Nidx1)
scatter_add = bmm(relu_grad_t, query_index)  # (B,S1,K,DQI)
d_key_index.scatter_add_(sparse_idx, scatter_add * valid_mask)
```

### 6.3 Python 瓶颈

- **`torch.bmm` in `_compute_d_key_index`**：`reluGrad^T(K,Nidx1) @ QI(Nidx1,DQI) → (K,DQI)`，K=2048 时计算量 ~2×2048×64×128 = 33M FLOPs
- **`scatter_add_`**：随机索引写入，cache-unfriendly
- **`transpose + contiguous`** for Kernel3：额外内存拷贝

## 7. 性能剖析（Profile 数据）

稳态性能（K=2048, S1=1, 第2次及之后调用，缓存命中）：

| 阶段 | 耗时 (ms) | 占比 | 说明 |
|------|-----------|------|------|
| gather_compile | 0.044 | 3.1% | gather + workspace 分配 |
| kernel1 | 0.477 | 33.9% | MM1+MM2 + P_reduced + ReduceSumY |
| kernel2 | 0.209 | 14.9% | SoftmaxI + dWeight + reluGrad + Loss |
| **kernel3** | ? | ? | MM6（待 profile 确认） |
| python | 0.677 | 48.1% | dWeight sum + bmm(MM6) + scatter_add + cast |
| **总** | **~1.4** | | 官方 ~0.35ms，**差距 ~4x** |

> 注：以上数据是 Kernel3 加入前的 profile。Kernel3 加入后 python 段应减少（MM6 bmm 被移除）。

## 8. 瓶颈优先级

### 8.1 P0 — Kernel3 精度异常（阻塞）

Kernel3 输出 ±10^13 / ±inf，必须修复才能正确计算 dQueryIndex。当前 block 了 MM6 移入 kernel 的收益。

**修复方向**：
1. 确认 `func3` 返回值与传入 workspace 是同一 tensor 对象
2. 验证 `T.gemm_v0(transpose_B=True)` + `GatherKeyIndexT` 的维度语义（对比 Python 参考值 `dq_ref`）
3. 若 C→V atomic_add 模式本身有问题，退化为 Python-only MM6

### 8.2 P1 — Python 后处理占比 48%

即使用 Kernel3 消除了 MM6 bmm，Python 仍有：

- **`_compute_d_key_index`**：包含 `reluGrad^T @ QI` bmm（~33M FLOPs）+ `scatter_add_`。可考虑：
  - 将 `reluGrad^T @ QI` 移入独立 kernel（类似 Kernel3 的 C→V atomic_add 模式）
  - `scatter_add_` 在 kernel 中用 `T.tile.atomic_add` 替代（需验证可行性）

- **`d_weight_partial.sum(dim=2)`**：可移入 Kernel2（需 UB→Global atomic_add，之前已证实不支持）
  - 替代方案：改为 `T.copy` 到不同 shape 的 output tensor

- **`loss_partial.sum()`**：同上

### 8.3 P2 — Kernel1 的 per-row serial 循环

Kernel1 V1 中有 `for g_i in T.serial(BLOCK_N1)` 等标量循环，N1=128 → 128 次串行迭代。

**可能的优化**：使用 `T.tile.broadcast` 将 1D 操作转换为 Vector 单元并行（但之前尝试发现 `T.tile.broadcast` 不支持 1D→2D）。

### 8.4 P3 — Kernel2 的全局 SoftmaxI 冗余

16 个 core 各自独立计算相同的 `softmax(ReduceSumY)`，浪费 15× 计算。可将 SoftmaxI 移到 Python 预处理（一次性计算），作为额外输入传入 Kernel2。

### 8.5 P4 — BLOCK 参数 tuning

BLOCK_N1=64 比默认 32 快 ~19%（之前 tuning 结果），当前未启用。

## 9. 与官方 AscendC 的架构差异

| 维度 | 官方 | Plan B |
|------|------|--------|
| 阶段数 | 14-stage pipeline (单 kernel) | 3 kernel + Python bridge |
| MM 分块 | M_SPLIT=128, N_SPLIT=128 | BLOCK_N=128, BLOCK_K=128 |
| Gather | V0 阶段（Vector unit） | Python torch.gather |
| ScatterAdd | V2 阶段（DMA） | Python scatter_add_ |
| Cross-flag | C1→V1→C2→V2 | C→V (K1/K3), V-only (K2) |
| SoftmaxI | V1 内部分段计算 | Kernel2 冗余全量计算 |
| 同步 | 流水线内自动同步 | Kernel 间隐式流同步 |

## 10. 总结

Plan B 当前状态：
- **逻辑正确性**：dKeyIndex, dWeights, Loss 精度通过（与官方对齐）
- **Kernel3（MM6）**：精度异常（10^13 量级），待修复
- **性能**：~4x 慢于官方，主要瓶颈在 Python 后处理（48%）和 per-row serial 循环
- **最大优化收益点**：消除 Python 中的 2 次 bmm + scatter_add
