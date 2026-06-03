# SparseLightningIndexerGradKLLoss — TileLang 算子说明书

## 1. 算子概述

**SparseLightningIndexerGradKLLoss** 是 LightningIndexer 的反向算子，额外融合了 KL Loss 计算。前向公式：

```
I_{t,:} = W_{t,:} @ ReLU(q_{t,:} @ (K_{:t,:})^T)
```

Loss 为 KL 散度：
```
L(I) = Σ_t D_KL(p_{t,:} || Softmax(I_{t,:}))
```

TileLang 版本实现了 BSND 布局的完整计算流程（TND 通过 reshape 为 BSND 复用），包含 4 个硬件阶段：**Cube1 → Vec1 → Cube2 → Vec2**，并在 Python 层完成 ScatterAdd。

---

## 2. 全局常量 / 超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `BLOCK_M` | 128 | Cube 矩阵乘法的 M 轴分块大小（S1 序列维度） |
| `BLOCK_N` | 128 | Cube 矩阵乘法的 N 轴分块大小（K 轴 = topk 维度） |
| `BLOCK_K` | 128 | Cube 矩阵乘法的 K 轴分块大小（DQuery: 512 → 512/128=4 块; DQueryIndex: 128 → 128/128=1 块） |
| `VECTOR_BASEG` | 16 | 向量处理中一次循环处理的 Query Index 头数量（即 vec1 中 g_group 的批大小） |
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
| `block_num` | `B * N2 * ceil(S1 / BLOCK_M)` | 总共的核数 / Tile 数 |
| `n_num` | `K // BLOCK_N` | topk 的 N 轴分块数 |
| `d_num_query` | `DQuery // BLOCK_K = 4` | Q/K 的 K 轴分块数 |
| `d_num_query_index` | `DQueryIndex // BLOCK_K = 1` | Qidx/Kidx 的 K 轴分块数 |

---

## 4. Tensor 布局与形状

### 输入 Tensor

| Tensor | AscendC 名称 | 形状 (BSND) | 含义 |
|--------|-------------|-------------|------|
| `Query` | query | `(B, S1, N1, DQuery)` | Attention 的 Q 输入 |
| `GatherKey` | gatherKey | `(B, N2, S1, G, K, DQuery)` | 按 sparse_indices Gather 后的 Key（MQA 展开） |
| `QueryIndex` | queryIndex | `(B, S1, Nidx1, DQueryIndex)` | LightningIndexer 的 QueryIndex |
| `GatherKeyIndex` | gatherKeyIndex | `(B, N2, S1, Nidx1, K, DQueryIndex)` | 按 sparse_indices Gather 后的 KeyIndex |
| `Weights` | weight | `(B, S1, Nidx1)` | 权重 |
| `SoftmaxMax` | softmaxMax | `(B, N2, S1, G)` | 前向 softmax 的 max 值（用于反推 softmax） |
| `SoftmaxSum` | softmaxSum | `(B, N2, S1, G)` | 前向 softmax 的 sum 值 |
| `ScaleValue` | scale | `(1,)` | 缩放系数 = 1/sqrt(DQuery) |
| `SparseIndices` | sparseIndices | `(B, S1, Nidx2, K)` | top-k 索引 |

### Workspace Tensor（中间结果）

| Tensor | 形状 | 由谁写 | 谁读 | 说明 |
|--------|------|--------|------|------|
| `bmm1Res_ws` | `(B, N2, S1, G, K)` | Cube1-MM1 | Vec1 | `Query @ GatherKey^T` 结果 |
| `bmm2Res_ws` | `(B, N2, S1, Nidx1, K)` | Cube1-MM2 | Vec1 | `QueryIndex @ GatherKeyIndex^T` 结果 |
| `reluGrad_ws` | `(B, N2, S1, Nidx1, K)` | Vec1 | Cube2 | ReLU 梯度掩码加权后的值 |
| `scatterAdd_ws` | `(B, N2, S1, K, DQueryIndex)` | Cube2-MM3 | Python | `reluGrad^T @ QueryIndex` 用于 ScatterAdd |
| `dQueryIndex_ws` | `(B, S1, Nidx1, DQueryIndex)` | Cube2-MM4 | Vec2 | `reluGrad @ GatherKeyIndex` = 对 Qidx 的梯度 |
| `dWeights_ws` | `(B, S1, Nidx1, K)` | Vec1 | Vec2 | KL Loss 对 weights 的梯度（带 K 轴） |

### 输出 Tensor

| Tensor | 形状 | 说明 |
|--------|------|------|
| `dQueryIndex` | `(B, S1, Nidx1, DQueryIndex)` | QueryIndex 的梯度 |
| `dKeyIndex` | `(B, S2, Nidx2, DQueryIndex)` | KeyIndex 的梯度（通过 Python 层 ScatterAdd） |
| `dWeights` | `(B, S1, Nidx1)` | Weights 的梯度（K 轴 sum 后） |
| `Loss` | `(1,)` | KL Loss 标量值 |

---

## 5. 核分配（Kernel Launch）

```python
block_m = BLOCK_M = 128
s1_tile_num = ceil_div(S1, block_m)
block_num = B * N2 * s1_tile_num
# block_num 个 Tile，每个 Tile 处理固定范围的 S1 行
```

核块 ID 解码：
```python
bz = cid // (N2 * s1_tile_num)       # Batch 索引
n2_id = cid // s1_tile_num % N2      # Key 头索引（通常为 0）
m = cid % s1_tile_num                 # S1 tile 索引
s1_start = m * block_m                # 当前 Tile 的起始 S1 位置
s1_end = min(s1_start + block_m, S1)  # 结束位置
```

---

## 6. 计算流程总览

```
┌──────────────────────────────────────────────────────────────────┐
│  TileLang Pipeline                                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Cube1:  Query @ GatherKey^T → bmm1Res         (Q头组 G 维度)   │
│          QueryIndex @ GatherKeyIndex^T → bmm2Res (Qidx 头组)    │
│              │                                                   │
│              ▼  set_cross_flag("FIX", 1)                         │
│  Vec1:   在线 Softmax 恢复 + KL Loss + dW + reluGrad            │
│              │                                                   │
│              ▼  set_cross_flag("V", 2)                           │
│  Cube2:  reluGrad^T @ QueryIndex → scatterAdd   (dK 计算)       │
│          reluGrad @ GatherKeyIndex → dQueryIndex (dQidx 计算)   │
│              │                                                   │
│              ▼  set_cross_flag("FIX", 3)                         │
│  Vec2:   FP32→FP16 类型转换（dQueryIndex, dWeights）             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Python: ScatterAdd (dynamic indexing) + Loss 累加
```

---

## 7. 各阶段详细解析

### 7.1 Cube1（矩阵乘法阶段）

**硬件单元**：AICore（Cube 单元）  
**同步**：等待 Vec1 完成（首次无依赖），通过 `set_cross_flag("FIX", 1)` 通知 Vec1

#### 7.1.1 Matmul1: Query @ GatherKey^T → bmm1Res_ws

```
形状: (G, 512) @ (K, 512)^T → (G, K)
                                      head_dim
    query_group[g_idx, :]  ←─── query[b, s1, g_idx, :]     (s1固定, G个head)
    gatherKey[...]         ←─── GatherKey[b, N2, s1, g_idx, :, :]  (K, 512)
    gemm_v0(Q_L1, GK_L1, C_L0C, transpose_B=True)
```

**变量**：

| 变量 | 含义 |
|------|------|
| `Q_L1` | L1 缓存中 Query 的分块 `(128, 128)` = `(BLOCK_M, BLOCK_K)`，shape = `(block_m, block_k)` |
| `GK_L1` | L1 缓存中 GatherKey 的分块 `(128, 128)` = `(BLOCK_N, BLOCK_K)` |
| `C_L0C` | L0C 累加器中的结果 `(128, 128)` = `(BLOCK_M, BLOCK_N)` |
| `s1_row` | 当前处理的 S1 位置（0 ~ block_m-1） |
| `g_idx` | 当前处理的 Query 头组索引（0 ~ G-1） |
| `n_idx` | topk 的 N 轴分块（0 ~ n_num-1） |
| `k_idx` | head_dim 的 K 轴分块（0 ~ d_num_query-1，即 0~3） |
| `init=(k_idx==0)` | 第一次分块时初始化 L0C，后续累加 |

**伪代码**：
```
for s1_row in range(block_m):
    for g_idx in range(G):
        n1_idx = n2_id * G + g_idx
        for n_idx in range(n_num):
            C_L0C = 0  # 隐式初始化
            for k_idx in range(d_num_query):  # 4次
                Q_L1  = Query[b, s1_start+s1_row, n1_idx, k_idx*128:(k_idx+1)*128]
                GK_L1 = GatherKey[b, n2_id, s1_start+s1_row, g_idx, n_idx*128:(n_idx+1)*128, :]
                C_L0C += gemm_v0(Q_L1, GK_L1, transpose_B=True)
            bmm1Res_ws[s1_row, g_idx, n_idx] = C_L0C
```

#### 7.1.2 Matmul2: QueryIndex @ GatherKeyIndex^T → bmm2Res_ws

```
形状: (Nidx1, 128) @ (K, 128)^T → (Nidx1, K)
```

**变量**：

| 变量 | 含义 |
|------|------|
| `QI_L1` | L1 中 QueryIndex 的分块 `(128, 128)` |
| `GKI_L1` | L1 中 GatherKeyIndex 的分块 `(128, 128)` |
| `C2_L0C` | L0C 中的结果 `(128, 128)` |
| `gidx_idx` | QueryIndex 头索引（0 ~ Nidx1-1） |

**伪代码**：
```
for s1_row in range(block_m):
    for gidx_idx in range(Nidx1):
        for n_idx in range(n_num):
            for k_idx in range(d_num_query_index):  # d_num_query_index=1
                QI_L1  = QueryIndex[b, s1_start+s1_row, gidx_idx, :]  # 全部DQueryIndex
                GKI_L1 = GatherKeyIndex[b, n2_id, s1_start+s1_row, gidx_idx, n_idx*128:(n_idx+1)*128, :]
                C2_L0C += gemm_v0(QI_L1, GKI_L1, transpose_B=True)
            bmm2Res_ws[s1_row, gidx_idx, n_idx] = C2_L0C
```

---

### 7.2 Vec1（向量处理阶段）

**硬件单元**：AIVCore（向量核心）  
**同步**：等待 Cube1 完成（`wait_cross_flag(1)`），`set_cross_flag("V", 2)` 通知 Cube2

**UB（Unified Buffer）变量**：

| 变量 | 大小 | 类型 | 含义 |
|------|------|------|------|
| `weight_half_ub` | `VECTOR_BASEG=16` | `input_dtype` | 16 个 Weights 值的 UB 缓冲区（原始 dtype） |
| `weight_fp32_ub` | `VECTOR_BASEG=16` | `calc_dtype=FP32` | 16 个 Weights 值的 FP32 缓冲区 |
| `bmm1_row_ub` | `BLOCK_N=128` | `calc_dtype` | bmm1Res 的一行（G 个 head 共享） |
| `bmm2_row_ub` | `BLOCK_N=128` | `calc_dtype` | bmm2Res 的一行（当前 gidx_idx 对应） |
| `sm_max_ub` | 1 | `calc_dtype` | 当前 g_idx 的 softmax_max |
| `sm_sum_ub` | 1 | `calc_dtype` | 当前 g_idx 的 softmax_sum |
| `scale_ub` | 1 | `calc_dtype` | 缩放系数 |
| `p_row_ub` | `BLOCK_N=128` | `calc_dtype` | 恢复后的 Softmax(p) 值 |
| `softmax_i_row_ub` | `BLOCK_N=128` | `calc_dtype` | 缩放的 I（bmm2Res * scale） |
| `di_row_ub` | `BLOCK_N=128` | `calc_dtype` | KL Loss 对 I 的梯度 = Softmax(I) - p |
| `relu_grad_row_ub` | `BLOCK_N=128` | `calc_dtype` | ReLU 梯度 |
| `relu_grad_half_ub` | `BLOCK_N=128` | `input_dtype` | 转成 FP16/BF16 的 ReLU 梯度 |
| `mask_ub` | `BLOCK_N=128` | `calc_dtype` | RelU 梯度掩码（bmm2Res > 0 的位置） |
| `g_group` | — | — | 外层循环：按 `VECTOR_BASEG=16` 分组处理 QueryIndex 头 |
| `g_local` | — | — | 内层循环：组内每个头（0~15） |

**计算步骤**：

```
1. 复制 ScaleValue 到 scale_ub

2. for g_group in range(Nidx1 / 16):
       for s1_row in range(block_m):
           # 2.1 加载 16 个 weights
           Weights[b, s1_start+s1_row, g_group*16 : (g_group+1)*16] → weight_half_ub
           weight_half_ub → cast(FP32) → weight_fp32_ub

           for g_local in range(16):  # 逐个 head 处理
               gidx_idx = g_group * 16 + g_local
               g_idx = gidx_idx  # 注意: gidx_idx 与 g_idx 数值上相等

               # 2.2 加载 softmax 统计量
               SoftmaxMax[b, n2_id, s1_start+s1_row, g_idx] → sm_max_ub
               SoftmaxSum[b, n2_id, s1_start+s1_row, g_idx] → sm_sum_ub

               for n_idx in range(n_num):
                   # 2.3 在线 Softmax 恢复
                   bmm1_row_ub = bmm1Res_ws[b, n2_id, s1_start+s1_row, g_idx, n_idx*128:(n_idx+1)*128]
                   p_row_ub = exp(bmm1_row_ub - sm_max_ub) / sm_sum_ub    # Softmax 反推

                   # 2.4 加载加权得分
                   bmm2_row_ub = bmm2Res_ws[b, n2_id, s1_start+s1_row, gidx_idx, n_idx*128:(n_idx+1)*128]
                   softmax_i_row_ub = bmm2_row_ub * scale_ub              # 缩放 I

                   # 2.5 KL Loss 梯度: dI = Softmax(I) - p
                   di_row_ub = softmax_i_row_ub - p_row_ub                # KL 散度的梯度

                   # 2.6 dWeights 计算: dW = dI * Softmax(I)
                   di_row_ub = di_row_ub * softmax_i_row_ub
                   dWeights_ws[b, s1_start+s1_row, gidx_idx, n_idx*128:(n_idx+1)*128] = di_row_ub

                   # 2.7 ReLU 梯度: relu_grad = dI * mask * W
                   mask_ub = (bmm2_row_ub > 0)                             # ReLU 掩码
                   di_row_ub = softmax_i_row_ub - p_row_ub                 # 重新计算 dI
                   relu_grad_row_ub = select(mask_ub, di_row_ub, 0.0)      # 应用 ReLU 掩码
                   relu_grad_row_ub = relu_grad_row_ub * weight_fp32_ub[g_local]  # 乘以权重
                   relu_grad_row_ub → cast(FP16, round) → relu_grad_half_ub
                   reluGrad_ws[b, n2_id, s1_start+s1_row, gidx_idx, n_idx*128:(n_idx+1)*128] = relu_grad_half_ub
```

---

### 7.3 Cube2（矩阵乘法阶段）

**硬件单元**：AICore（Cube 单元）  
**同步**：等待 Vec1 完成（`wait_cross_flag(2)`），`set_cross_flag("FIX", 3)` 通知 Vec2

#### 7.3.1 Matmul3: reluGrad^T @ QueryIndex → scatterAdd_ws

```
形状: (K, Nidx1) @ (Nidx1, 128) → (K, 128)
```

**变量**：

| 变量 | 含义 |
|------|------|
| `RG_L1` | L1 中 reluGrad 的分块 `(128, 128)` |
| `QI2_L1` | L1 中 QueryIndex 的分块 `(128, 128)` |
| `SA_L0C` | L0C 中的 scatterAdd 结果 |

**用途**：`scatterAdd_ws` → Python 层 ScatterAdd → `dKeyIndex`

---

#### 7.3.2 Matmul4: reluGrad @ GatherKeyIndex → dQueryIndex_ws

```
形状: (Nidx1, K) @ (K, 128) → (Nidx1, 128)
```

**变量**：

| 变量 | 含义 |
|------|------|
| `GKI2_L1` | L1 中 GatherKeyIndex 的分块 `(128, 128)` |
| `DQ_L0C` | L0C 中的 dQueryIndex 结果 |

**用途**：对 QueryIndex 的梯度

---

### 7.4 Vec2（向量后处理阶段）

**硬件单元**：AIVCore（向量核心）  
**同步**：等待 Cube2 完成（`wait_cross_flag(3)`）

**变量**：

| 变量 | 大小 | 含义 |
|------|------|------|
| `dq_fp32_ub` | `BLOCK_K=128` | dQueryIndex_ws 的一行（FP32） |
| `dq_fp16_ub` | `BLOCK_K=128` | 转成 FP16/BF16 后写入 |
| `dw_fp32_ub` | `BLOCK_N=128` | dWeights_ws 的一行（FP32） |
| `dw_fp16_ub` | `BLOCK_N=128` | 转成 FP16/BF16 后写入 |

**步骤**：

```
1. dQueryIndex_ws FP32 → Cast(ROUND) → FP16 → dQueryIndex (输出)

2. dWeights_ws FP32 → Cast(ROUND) → FP16 → dWeights (输出)
```

**注意**：dWeights 输出是 `(B, S1, Nidx1, K)` 形状，Python 层会做 `sum(dim=-1)` 得到 `(B, S1, Nidx1)`。Loss 的计算在 TileLang 内核中暂未实现，由 Python 层在 ScatterAdd 时一并计算。

---

## 8. Python 层计算

### 8.1 Gather 预处理

```python
def _preprocess_gather(key, key_index, sparse_indices, ...):
    # Key: (B, S2, N2, DQuery) → GatherKey: (B, N2, S1, G, K, DQuery)
    # 去除 -1 padding，按 sparse_indices 做 gather，再按 G 展开
    gather_key = key[batch_idx, sparse_idx_3d, :, :]  # (B, S1, K, N2, DQuery)
    gather_key = gather_key.permute(0, 3, 1, 2, 4)    # (B, N2, S1, K, DQuery)
    gather_key = gather_key.unsqueeze(3).expand(B, N2, S1, G, K, DQuery)

    # KeyIndex 类似处理
    # KeyIndex: (B, S2, Nidx2, DQueryIndex) → GatherKeyIndex: (B, N2, S1, Nidx1, K, DQueryIndex)
```

### 8.2 ScatterAdd（dKeyIndex 计算）

`scatterAdd_ws` 形状 `(B, N2, S1, K, DQueryIndex)`，稀疏索引 `indices[s1, k]` 将 `scatterAdd[s1][k]` 累加到 `dKeyIndex[indices[s1][k]]`。

```
dKeyIndex[b][indices[s1][k]][n2_idx][:] += scatterAdd[b][n2_id][s1][k][:]
```

### 8.3 dWeights 降维

```
dWeights (B, S1, Nidx1, K) → sum(-1) → (B, S1, Nidx1)
```

### 8.4 Loss 计算

从 `dWeights_ws`（= `dI * Softmax(I)`）和 `bmm2Res_ws` 等计算 KL Loss。当前 TileLang 版本中 Loss 计算在 Python 层实现。

---

## 9. AscendC vs TileLang 对比

| 特性 | AscendC | TileLang |
|------|---------|----------|
| **Rope 支持** | ✅ 有 QueryRope/KeyRope | ❌ 不支持 |
| **矩阵乘法** | Mm1-Mm6（6 个 matmul） | Cube1 (MM1+MM2) + Cube2 (MM3+MM4) |
| **Gather 实现** | V0 在 AIVCore 上向量 Gather | Python 层预处理（`_preprocess_gather`） |
| **ScatterAdd** | V2 在 AIVCore 上原子操作 | Python 层 |
| **确定性模式** | ✅ 支持 Deterministic | ❌ 不支持 |
| **Loss 计算** | 在 V1 内计算并累加 | Python 层 |
| **流水线深度** | 3 级（双缓冲 + prefetch） | 2 级（Cube→Vec→Cube→Vec） |
| **缓存同步** | 显式 Event 管理 | TileLang 框架自动管理 |

---

## 10. 数据流图

```
                    SparseIndices (B,S1,Nidx2,K)
                          │
                          ▼  [Python Gather]
    Key ──────► GatherKey (B,N2,S1,G,K,DQuery)
    KeyIndex ─► GatherKeyIndex (B,N2,S1,Nidx1,K,DQueryIndex)

    ┌────────────────────────────────────────────────────────┐
    │  Cube1                                                │
    │  Query @ GatherKey^T      → bmm1Res (B,N2,S1,G,K)    │
    │  QueryIndex @ GatherKeyIndex^T → bmm2Res (B,N2,S1,Nidx1,K) │
    └────────────────────────────────┬───────────────────────┘
                                     │
    ┌────────────────────────────────▼──────────────────────┐
    │  Vec1                                                 │
    │  bmm1Res + SoftmaxMax/Sum → Softmax(p)               │
    │  bmm2Res * Scale → Softmax(I)                        │
    │  dI = Softmax(I) - p                                  │
    │  dWeights = dI * Softmax(I)                           │
    │  reluGrad = dI * ReLU_mask * Weights                  │
    └────────────────────────────────┬──────────────────────┘
                                     │
    ┌────────────────────────────────▼──────────────────────┐
    │  Cube2                                                │
    │  reluGrad^T @ QueryIndex      → scatterAdd (K,DQueryIndex)│
    │  reluGrad @ GatherKeyIndex    → dQueryIndex (Nidx1,128)│
    └────────────────────────────────┬──────────────────────┘
                                     │
    ┌────────────────────────────────▼──────────────────────┐
    │  Vec2                                                 │
    │  dQueryIndex: FP32 → FP16 (输出)                     │
    │  dWeights_ws: FP32 → FP16 (输出, 含K轴)              │
    └───────────────────────────────────────────────────────┘

    [Python 层]
    1. ScatterAdd(scatterAdd_ws, SparseIndices) → dKeyIndex
    2. dWeights = dWeights.sum(dim=-1)  # 消去 K 轴
    3. Loss 计算
```

---

## 11. 关键约束条件

| 条件 | 要求 | 原因 |
|------|------|------|
| `N2 == 1` | 必须为 1 | MQA 模式，G = N1/N2 |
| `DQuery == 512` | 固定 | Cube 分块需要 512/128=4 等分 |
| `DQueryIndex == 128` | 固定 | 单 BLOCK_K=128 即可覆盖 |
| `K % 1024 == 0` | 1024~8192 | N 轴分块需要能被 128 整除 |
| `N1 in (32, 64, 128)` | 32/64/128 | Query 头数限制 |
| `Nidx1 in (8,16,32,64)` | 指定范围 | QueryIndex 头数限制 |
| `sparse_mode == 3` | 固定 | 当前仅支持 RightDown 模式 |