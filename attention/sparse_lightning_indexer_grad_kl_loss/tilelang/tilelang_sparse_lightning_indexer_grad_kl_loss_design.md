# SparseLightningIndexerGradKLLoss — TileLang 算子说明书

## 1. 算子概述

**SparseLightningIndexerGradKLLoss** 是 LightningIndexer 的反向算子，额外融合了 KL Loss 计算。前向公式：

```
I_{t,:} = W_{t,:} @ ReLU(q_{t,:} @ (K_{:t,:})^T)
```

Loss 为 KL 散度：
```
L(I) = Σ_t D_KL(p_{t,:} || Softmax(I_{t,:}))
     = Σ_k P_reduced[k] * (ln(P_reduced[k]) - ln(softmaxI[k]))
```

TileLang 版本采用**单 kernel CV 分离架构**，参考 BSND3.py 模式，将所有 NPU 计算放在一个 kernel 中，通过 C scope 和 V scope 交替执行。Python 层仅负责 Gather 预处理和 ScatterAdd（dKeyIndex）。

---

## 2. 全局常量 / 超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `BLOCK_M` | 128 | Cube 矩阵乘法的 M 轴分块大小（S1 序列维度） |
| `BLOCK_N` | 128 | Cube 矩阵乘法的 N 轴分块大小（K 轴 = topk 维度） |
| `BLOCK_K` | 128 | Cube 矩阵乘法的 K 轴分块大小（DQuery: 512→4块; d_rope: 64→1块; DQueryIndex: 128→1块） |
| `VECTOR_BASEG` | 16 | V scope 中一次循环处理的 QueryIndex 头数量 |
| `VECTOR_BASEN` | 256 | S2 轴对齐的基本单位，用于 padding |
| `DQUERY` | 512 | Query/Key 的隐藏层维度 |
| `DQUERY_INDEX` | 128 | QueryIndex/KeyIndex 的隐藏层维度 |

---

## 3. TilingData 数据结构

```python
@dataclass
class SparseLightningIndexerGradKLLossTilingData:
    B: int         # Batch 大小
    S1: int        # Query 序列长度
    S2: int        # Key 序列长度
    N1: int        # Query 头数
    N2: int        # Key 头数（MQA 模式固定为 1）
    G: int         # 每组的 Query 头数 = N1 // N2
    Nidx1: int     # QueryIndex 头数（通常 = Query 头数的子集, 如 64）
    Nidx2: int     # KeyIndex 头数（通常 = N2 = 1）
    DQuery: int    # Q/K 的 head dim（固定 512）
    DQueryIndex: int  # QueryIndex/KeyIndex 的 head dim（固定 128）
    K: int         # topk 数量（1024~8192，必须是 1024 的倍数）
    sparse_mode: int  # 默认 3
```

### Computed Properties

| 属性 | 公式 | 含义 |
|------|------|------|
| `g_size` | `G` | 每组 Query 头数 |
| `n2_size` | `N2` | Key 头数 |
| `block_num` | `B * ceil(S1 / BLOCK_M)` | 总共的核数 / Tile 数 |
| `n_num` | `K // BLOCK_N` | topk 的 N 轴分块数 |
| `d_num_query` | `DQuery // BLOCK_K = 4` | Q/K 的 K 轴分块数 |
| `d_num_rope` | `ceil(d_rope / BLOCK_K)` | Rope 的 K 轴分块数 |
| `d_num_total` | `d_num_query + d_num_rope` | Matmul1 的总 K 轴分块数（含 rope） |
| `d_num_query_index` | `DQueryIndex // BLOCK_K = 1` | Qidx/Kidx 的 K 轴分块数 |
| `vector_baseg` | `min(Nidx1, 16)` | V scope 中 g_group 的批大小 |
| `g_num` | `Nidx1 // vector_baseg` | g_group 的分组数 |

---

## 4. Tensor 布局与形状

### 输入 Tensor

| Tensor | 形状 (BSND) | 含义 |
|--------|-------------|------|
| `Query` | `(B, S1, N1, DQuery)` | Attention 的 Q 输入 |
| `QueryRope` | `(B, S1, N1, d_rope)` | Query 的 RoPE 部分 |
| `GatherKey` | `(B, S1, K, DQuery)` | 按 sparse_indices Gather 后的 Key |
| `GatherKeyRope` | `(B, S1, K, d_rope)` | 按 sparse_indices Gather 后的 Key RoPE 部分 |
| `QueryIndex` | `(B, S1, Nidx1, DQueryIndex)` | LightningIndexer 的 QueryIndex |
| `GatherKeyIndex` | `(B, S1, K, DQueryIndex)` | 按 sparse_indices Gather 后的 KeyIndex |
| `GatherKeyIndexT` | `(B, S1, DQueryIndex, K)` | GatherKeyIndex 的转置，供 Matmul4 使用 |
| `Weights` | `(B, S1, Nidx1)` | 权重 |
| `SoftmaxMax` | `(B, N2, S1, G)` | 前向 softmax 的 max 值 |
| `SoftmaxSum` | `(B, N2, S1, G)` | 前向 softmax 的 sum 值 |

### Workspace Tensor（中间结果）

| Tensor | 形状 | 由谁写 | 谁读 | 说明 |
|--------|------|--------|------|------|
| `bmm1Res` | `(B, S1, N1, K)` | C scope 1 | V scope 1 | `[Query||QueryRope] @ [GatherKey||GatherKeyRope]^T` 结果 |
| `bmm2Res` | `(B, S1, Nidx1, K)` | C scope 1 | V scope 1 | ReLU(`QueryIndex @ GatherKeyIndex^T`) 结果 |
| `PReduced` | `(B, S1, K)` | V scope 1 | V scope 1 | P_reduced = (1/G)*sum_g P[g,k]，GM 中转 |
| `ReduceSumY` | `(B, S1, K)` | V scope 1 | V scope 1 | reduceSumY = sum_g(reluRes*weight)，GM 中转 |
| `SoftmaxI` | `(B, S1, K)` | V scope 1 | V scope 1 | softmaxI = softmax(reduceSumY)，GM 中转 |
| `dQueryIndex_ws` | `(B, S1, Nidx1, DQueryIndex)` | C scope 2 | V scope 2 | `reluGrad @ GatherKeyIndex` FP32 中间结果 |

### 输出 Tensor

| Tensor | 形状 | 说明 |
|--------|------|------|
| `dQueryIndex` | `(B, S1, Nidx1, DQueryIndex)` | QueryIndex 的梯度 |
| `dWeight` | `(B, S1, Nidx1)` | Weights 的梯度 |
| `reluGradRes` | `(B, S1, Nidx1, K)` | ReLU 梯度（C scope 2 读取用于 Matmul4） |
| `LossOut` | `(B, S1, 16)` | KL Loss 结果（V scope 1 Pass 3 计算） |

### Python 层输出

| Tensor | 形状 | 说明 |
|--------|------|------|
| `dKeyIndex` | `(B, S2, Nidx2, DQueryIndex)` | KeyIndex 的梯度（Python ScatterAdd） |

---

## 5. 核分配（Kernel Launch）

```python
s1_block = BLOCK_M = 128
s1_tile_num = ceil_div(S1, s1_block)
block_num = B * s1_tile_num
# block_num 个 Tile，每个 Tile 处理固定范围的 S1 行
```

核块 ID 解码：
```python
bz = cid // s1_tile_num       # Batch 索引
m = cid % s1_tile_num         # S1 tile 索引
s1_start = m * s1_block       # 当前 Tile 的起始 S1 位置
s1_end = min(s1_start + s1_block, S1)  # 结束位置
```

---

## 6. 计算流程总览

```
┌──────────────────────────────────────────────────────────────────┐
│  Single Kernel: CV-Separated Architecture                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  C scope 1:  Matmul1: [Query||QueryRope]                        │
│                     @ [GatherKey||GatherKeyRope]^T → bmm1Res     │
│              Matmul2: QueryIndex @ GatherKeyIndex^T → bmm2Res    │
│              (bmm2Res with ReLU)                                 │
│              │  set_cross_flag("FIX", 1)                         │
│                                                                  │
│  V scope 1:  wait_cross_flag(1)                                 │
│              Pass 1: P_reduced[K], reduceSumY[K] 累积            │
│              Global softmax: reduceSumY → softmaxI[K]            │
│              Pass 2: dWeight[Nidx1], reluGrad[Nidx1, K]          │
│              Pass 3: KL Loss = Σ P_reduced*(ln(P_reduced)       │
│                                    - ln(softmaxI))               │
│              │  set_cross_flag("V", 2)                           │
│                                                                  │
│  C scope 2:  wait_cross_flag(2)                                 │
│              Matmul4: reluGrad @ GatherKeyIndexT^T → dQueryIndex │
│              │  set_cross_flag("FIX", 3)                         │
│                                                                  │
│  V scope 2:  wait_cross_flag(3)                                 │
│              Cast: dQueryIndex_ws FP32 → FP16                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Python: ScatterAdd (dKeyIndex)
```

---

## 7. 各阶段详细解析

### 7.1 C scope 1: Matmul1 (with RoPE) + Matmul2

**同步**：`set_cross_flag("FIX", 1)` 通知 V scope 1

#### 7.1.1 Matmul1: [Query||QueryRope] @ [GatherKey||GatherKeyRope]^T → bmm1Res

```
形状: (N1, DQuery+d_rope) @ (K, DQuery+d_rope)^T → (N1, K)
gemm: A(N1, k_block) @ B(n_block, k_block)^T → C(N1, n_block)
K 轴分块: d_num_total = d_num_query + d_num_rope
  前 d_num_query 块: 用 Query / GatherKey
  后 d_num_rope 块: 用 QueryRope / GatherKeyRope
```

L1/L0C buffer：
- `Q_L1`: `(N1, k_block)` = `(128, 128)` input_dtype
- `GK_L1`: `(n_block, k_block)` = `(128, 128)` input_dtype
- `C_L0C`: `(N1, n_block)` = `(128, 128)` calc_dtype

循环：`s1_row → n_idx → d_idx`，`init=(d_idx==0)`

#### 7.1.2 Matmul2: QueryIndex @ GatherKeyIndex^T → bmm2Res

```
形状: (Nidx1, DQueryIndex) @ (K, DQueryIndex)^T → (Nidx1, K)
gemm: A(Nidx1, k_block) @ B(n_block, k_block)^T → C(Nidx1, n_block)
输出带 enable_relu=True
```

L1/L0C buffer：
- `QI_L1`: `(Nidx1, k_block)` = `(64, 128)` input_dtype
- `GKI_L1`: `(n_block, k_block)` = `(128, 128)` input_dtype
- `C2_L0C`: `(Nidx1, n_block)` = `(64, 128)` calc_dtype

---

### 7.2 V scope 1: softmaxI, P_reduced, dWeight, reluGrad, Loss

**同步**：`wait_cross_flag(1)`，`set_cross_flag("V", 2)`

#### 数学公式（arch22 reference）

```
P[g, k] = exp(bmm1Res[g, k] * scale - softmaxMax[g]) / softmaxSum[g]
P_reduced[k] = (1/G) * Σ_g P[g, k]

reduceSumY[k] = Σ_g (ReLU(bmm2Res[g, k]) * weight[g])
softmaxI[k] = softmax(reduceSumY[k])  // 标准 softmax

diff[k] = softmaxI[k] - P_reduced[k]
dWeight[g] = Σ_k (diff[k] * reluRes[g, k])
reluGrad[g, k] = weight[g] * diff[k] * I(reluRes[g, k] > 0)
  // I(reluRes > 0) 用 1 - exp(-reluRes * 10000) 近似

Loss = Σ_k P_reduced[k] * (ln(P_reduced[k]) - ln(softmaxI[k]))
  // T.tile.ln 即自然对数 ln，不需要额外乘 ln(2)
```

#### UB buffer 分配

| 变量 | 大小 | 类型 | 含义 |
|------|------|------|------|
| `bmm1_ub` | `(N1, n_block)` = `(128, 128)` | FP32 | bmm1Res 的 K 分块 |
| `bmm2_ub` | `(Nidx1, n_block)` = `(64, 128)` | FP32 | bmm2Res 的 K 分块 |
| `weight_ub` | `vector_baseg=16` | FP32 | Weights FP32 |
| `weight_half_ub` | `vector_baseg=16` | FP16 | Weights 原始 dtype |
| `p_reduced_tmp_ub` | `n_block=128` | FP32 | P_reduced 的 K 分块累加 |
| `rsy_tmp_ub` | `(16, 128)` | FP32 | reduceSumY 的分块中间结果 |
| `rsy_reduce_ub` | `128` | FP32 | reduceSumY 的分块累加 |
| `diff_ub` | `128` | FP32 | softmaxI - P_reduced 的分块 |
| `dw_accum_ub` | `16` | FP32 | dWeight 累加器 |
| `rg_ub` | `(16, 128)` | FP32 | reluGrad 分块 |
| `rg_half_ub` | `(16, 128)` | FP16 | reluGrad cast 后 |
| `sm_max_ub` | `G=128` | FP32 | softmaxMax |
| `sm_sum_ub` | `G=128` | FP32 | softmaxSum |
| `rsy_2d_ub` | `(16, 128)` | FP32 | 2D 包装用于 reduce_max/reduce_sum |
| `rsy_max_ub` | `16` | FP32 | reduceSumY 的 max |
| `rsy_sum_ub` | `16` | FP32 | reduceSumY 的 sum |
| `dw_chunk_ub` | `(16, 128)` | FP32 | dWeight 分块 = diff * reluRes |
| `dw_reduce_ub` | `16` | FP32 | dw_chunk 的 reduce_sum 结果 |
| `dw_half_ub` | `16` | FP16 | dWeight cast 后 |
| `p_reduced_chunk_ub` | `128` | FP32 | P_reduced 的 K 分块 |
| `softmax_i_chunk_ub` | `128` | FP32 | softmaxI 的 K 分块 |
| `mask_ub` | `128` | FP32 | I(reluRes>0) 近似 |
| `ones_ub` | `128` | FP32 | 常量 1.0 |
| `loss_accum_ub` | `16` | FP32 | Loss 累加器 |
| `loss_chunk_ub` | `128` | FP32 | Loss 的 K 分块 |
| `loss_log_p_ub` | `128` | FP32 | ln(P_reduced) |
| `loss_log_si_ub` | `128` | FP32 | ln(softmaxI) |
| `loss_val_ub` | `128` | FP32 | P_reduced * (ln(P_reduced) - ln(softmaxI)) |
| `loss_2d_ub` | `(16, 128)` | FP32 | 2D 包装用于 reduce_sum |
| `loss_sum_ub` | `16` | FP32 | loss_val 的 reduce_sum 结果 |
| `eps_ub` | `128` | FP32 | 数值稳定 eps=1e-8 |

#### 计算流程

```
for s1_id in T.serial(s1_start, s1_end):
    初始化 ones_ub = 1.0
    加载 softmaxMax[G], softmaxSum[G]

    # Pass 1: 累积 P_reduced 和 reduceSumY
    for n_idx in T.serial(n_num):
        加载 bmm1Res[N1, n_block] → bmm1_ub
        for g_id in range(G):
            P[g, :] = exp(bmm1[g,:] * scale - sm_max[g]) / sm_sum[g]
            p_reduced_tmp += P[g, :]
        p_reduced_tmp *= (1/G)
        写 p_reduced_tmp → PReduced GM

        加载 bmm2Res[Nidx1, n_block] → bmm2_ub
        for g_id in T.serial(g_num):
            加载 Weights[vector_baseg] → weight_ub
            for i in range(vector_baseg):
                rsy_tmp[i, :] = bmm2_ub[g_row, :] * weight[i]
            rsy_reduce += Σ rsy_tmp[i, :]
        写 rsy_reduce → ReduceSumY GM

    # Global softmax: reduceSumY → softmaxI
    rsy_max = max(reduceSumY)
    softmaxI_unnorm = exp(reduceSumY - rsy_max)
    softmaxI = softmaxI_unnorm / sum(softmaxI_unnorm)

    # Pass 2: 计算 dWeight 和 reluGrad
    for g_id in T.serial(g_num):
        加载 Weights[vector_baseg] → weight_ub
        dw_accum = 0
        for n_idx in T.serial(n_num):
            加载 bmm2Res, softmaxI_chunk, P_reduced_chunk
            diff = softmaxI_chunk - p_reduced_chunk
            for i in range(vector_baseg):
                dw_chunk[i, :] = bmm2_ub[g_row, :] * diff
                mask = 1 - exp(-bmm2_ub[g_row, :] * 10000)
                rg_ub[i, :] = diff * weight[i] * mask
            dw_accum += reduce_sum(dw_chunk, axis=1)
            写 rg_ub → reluGradRes GM (cast to FP16)
        写 dw_accum → dWeight GM (cast to FP16)

    # Pass 3: 计算 KL Loss
    loss_accum = 0
    for n_idx in T.serial(n_num):
        log_p = ln(P_reduced_chunk + 1e-8)
        log_si = ln(softmaxI_chunk + 1e-8)
        loss_val = P_reduced_chunk * (log_p - log_si)
        loss_accum += reduce_sum(loss_val)
    写 loss_accum → LossOut GM
```

---

### 7.3 C scope 2: Matmul4

**同步**：`wait_cross_flag(2)`，`set_cross_flag("FIX", 3)`

#### Matmul4: reluGrad @ GatherKeyIndexT^T → dQueryIndex_ws

```
dQueryIndex = reluGrad @ GatherKeyIndex
            = reluGrad @ (GatherKeyIndexT)^T  (use transpose_B=True)
形状: (Nidx1, K) @ (K, DQueryIndex) → (Nidx1, DQueryIndex)
gemm: A(Nidx1, n_block) @ B(k_block, n_block)^T → C(Nidx1, k_block)
K_axis 按 n_block 分块，N_axis 按 k_block 分块
```

L1/L0C buffer：
- `RG_L1`: `(Nidx1, n_block)` = `(64, 128)` input_dtype
- `GKIT_L1`: `(k_block, n_block)` = `(128, 128)` input_dtype
- `DQ_L0C`: `(Nidx1, k_block)` = `(64, 128)` calc_dtype

**注意**：GatherKeyIndexT 是 GatherKeyIndex 的预转置 `(B,S1,DQueryIndex,K)`，在 Python 层通过 `permute(0,1,3,2).contiguous()` 生成，避免在 kernel 中使用 `transpose_B=False`（可能导致 Cube 指令非法）。

---

### 7.4 V scope 2: Cast dQueryIndex

**同步**：`wait_cross_flag(3)`

```
dQueryIndex_ws FP32 → Cast(ROUND) → FP16 → dQueryIndex (输出)
```

---

## 8. Python 层计算

### 8.1 Gather 预处理

```python
def _preprocess_gather(key, key_index, sparse_indices, ...):
    # Key: (B, S2, N2, DQuery) → GatherKey: (B, S1, K, DQuery)
    # KeyIndex: (B, S2, Nidx2, DQueryIndex) → GatherKeyIndex: (B, S1, K, DQueryIndex)
    gather_key = key[batch_idx, sparse_idx_3d, 0, :]
    gather_key_index = key_index[batch_idx, sparse_idx_3d, 0, :]

# RoPE Gather
gather_key_rope = key_rope[batch_idx, sparse_idx_3d, 0, :]

# 预转置 GatherKeyIndex 供 Matmul4 使用
gather_key_index_t = gather_key_index.permute(0, 1, 3, 2).contiguous()
```

### 8.2 ScatterAdd（dKeyIndex 计算）

```python
scatterAdd = reluGrad^T @ QueryIndex  # (K, Nidx1) @ (Nidx1, DQueryIndex) → (K, DQueryIndex)
dKeyIndex = scatter via sparse_indices
```

---

## 9. AscendC vs TileLang 对比

| 特性 | AscendC | TileLang |
|------|---------|----------|
| **架构** | 多 kernel (Cube+Vec 交叉) | 单 kernel CV 分离 |
| **Rope 支持** | ✅ QueryRope/KeyRope 拼接 | ✅ QueryRope/GatherKeyRope 拼接 |
| **矩阵乘法** | Mm1-Mm6（6 个 matmul） | Matmul1+2 (C1) + Matmul4 (C2) |
| **Gather 实现** | V0 在 AIVCore 上向量 Gather | Python 层预处理 |
| **ScatterAdd** | V2 在 AIVCore 上原子操作 | Python 层 |
| **Vec 计算** | softmaxI/P/dW/reluGrad 在 V1 | V scope 1 内完成 |
| **Loss 计算** | 在 V1 内计算并 atomic_add | V scope 1 Pass 3 内完成 |
| **CV 同步** | CrossCoreSetFlag/WaitFlag | T.set_cross_flag / T.wait_cross_flag |

---

## 10. 数据流图

```
                    SparseIndices (B,S1,Nidx2,K)
                          │
                          ▼  [Python Gather]
    Key ──────► GatherKey (B,S1,K,DQuery)
    KeyRope ──► GatherKeyRope (B,S1,K,d_rope)
    KeyIndex ─► GatherKeyIndex (B,S1,K,DQueryIndex)
                 GatherKeyIndexT (B,S1,DQueryIndex,K)  [预转置]

    ┌────────────────────────────────────────────────────────────┐
    │  C scope 1                                                │
    │  Matmul1: [Query||QueryRope]                              │
    │           @ [GatherKey||GatherKeyRope]^T → bmm1Res        │
    │  Matmul2: QueryIndex @ GatherKeyIndex^T → bmm2Res (ReLU)  │
    └────────────────────────────────┬───────────────────────────┘
                                     │  set_cross_flag("FIX", 1)
    ┌────────────────────────────────▼──────────────────────────┐
    │  V scope 1                                                │
    │  Pass 1: P_reduced[K], reduceSumY[K] 累积                 │
    │  Global softmax: reduceSumY → softmaxI[K]                 │
    │  Pass 2: dWeight[Nidx1], reluGrad[Nidx1, K]               │
    │  Pass 3: Loss = Σ P_reduced*(ln(P_reduced) - ln(softmaxI))│
    └────────────────────────────────┬──────────────────────────┘
                                     │  set_cross_flag("V", 2)
    ┌────────────────────────────────▼──────────────────────────┐
    │  C scope 2                                                │
    │  Matmul4: reluGrad @ GatherKeyIndexT^T → dQueryIndex      │
    └────────────────────────────────┬───────────────────────────┘
                                     │  set_cross_flag("FIX", 3)
    ┌────────────────────────────────▼──────────────────────────┐
    │  V scope 2                                                │
    │  Cast: dQueryIndex_ws FP32 → FP16 (输出)                  │
    └───────────────────────────────────────────────────────────┘

    [Python 层]
    ScatterAdd(reluGrad, QueryIndex, SparseIndices) → dKeyIndex
```

---

## 11. 精度验证结果

对比官方 Ascend C 算子（with rope, BSND layout）：

| 输出 | max_diff | 精度等级 |
|------|----------|----------|
| dQueryIndex | 0.000320 | 3.2e-4 |
| dKeyIndex | 0.001953 | 2.0e-3 |
| dWeights | 0.000000 | 0 |
| Loss | 0.000011 | 1.1e-5 |

测试配置：B=1, S1=1, S2=2048, N1=128, N2=1, Nidx1=64, K=2048, DQuery=512, DQueryIndex=128, d_rope=64

---

## 12. 关键约束条件

| 条件 | 要求 | 原因 |
|------|------|------|
| `N2 == 1` | 必须为 1 | MQA 模式，G = N1/N2 |
| `DQuery == 512` | 固定 | Cube 分块需要 512/128=4 等分 |
| `DQueryIndex == 128` | 固定 | 单 BLOCK_K=128 即可覆盖 |
| `d_rope` | 64（可配置） | Matmul1 K 轴扩展为 DQuery+d_rope |
| `K % 1024 == 0` | 1024~8192 | N 轴分块需要能被 128 整除 |
| `N1 in (32, 64, 128)` | 32/64/128 | Query 头数限制 |
| `Nidx1 in (8,16,32,64)` | 指定范围 | QueryIndex 头数限制 |
| `sparse_mode == 3` | 固定 | 当前仅支持 RightDown 模式 |
| gemm M >= 16 | 硬件约束 | Ascend Cube M 维度最小 16 |
| UB 总量 <= 192KB | 硬件约束 | K=8192 时可能接近上限 |

---

## 13. TileLang API 注意事项

| API | 注意 |
|-----|------|
| `T.tile.mul/add/sub/exp/div/fill` | **必须在 FP32 buffer 上操作**，不支持 half 精度 |
| `T.tile.cast` | 用于 FP16↔FP32 转换，half buffer 只能出现在 cast 的源或目标 |
| `T.tile.ln` | **自然对数 ln**（不是 log2），不需要额外乘 ln(2) |
| `T.tile.fill` | 代替 `T.tile.dup`（TileLang 无 dup） |
| `T.reduce_sum/reduce_max` | 要求 2D→1D（沿 dim=0 或 dim=1 reduce），不支持 1D→scalar |
| `T.tile.atomic_add(dst, src)` | dst 和 src size 必须完全相同，且 dst 必须是 GM tensor |
| `T.copy` | 源和目标维度数必须匹配；**不能以 UB 行切片作为目标**（MTE 越界） |
| **`range()`** | V scope 中**可以**用于 UB 行索引（编译期常量展开），但**不能**用于 GM 索引 |
| `T.serial()` | V scope 中循环变量是运行时变量，**可以**索引 GM，但**不能**索引 UB 行 |
| `T.serial(s1_start, s1_end)` | 用于 S1 tiling，避免 S1 < block_m 时越界 |
| `T.tile.div(dst, src, scalar)` | int 除数可能不支持，建议用 `T.tile.mul(dst, src, 1.0/G)` |
| P_reduced/reduceSumY/softmaxI | 存储在 GM workspace 中，V scope 按 K 分块 copy 到 UB 处理 |
| `transpose_B=True` | Matmul4 中通过预转置 GatherKeyIndexT 实现，避免 `transpose_B=False` 导致设备崩溃 |
