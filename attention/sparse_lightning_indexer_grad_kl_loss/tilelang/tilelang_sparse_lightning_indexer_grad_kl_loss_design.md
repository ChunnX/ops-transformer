# SparseLightningIndexerGradKLLoss — TileLang 算子说明书 (0622.py)

## 1. 算子概述

**SparseLightningIndexerGradKLLoss** 是 LightningIndexer 的反向算子，额外融合 KL Loss 计算。

前向公式：
```
I_{t,:} = W_{t,:} @ ReLU(q_{t,:} @ (K_{:t,:})^T)
```

KL Loss：
```
L(I) = Σ_t D_KL(p_{t,:} || Softmax(I_{t,:}))
     = Σ_k P_reduced[k] * (ln(P_reduced[k]) - ln(softmaxI[k]))
```

TileLang 版本采用**单 kernel CV 分离 + K 块并行架构**，每个 AI Core 处理一个 K 块（BLOCK_N=128），通过跨 scope cross_flag 同步。Python 层负责 Gather 预处理和 ScatterAdd（dKeyIndex）。

---

## 2. 全局常量 / 超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `BLOCK_M` | 128 | S1 序列维度分块大小 |
| `BLOCK_N` | 128 | K 维度分块大小（= n_num 个核各处理一块） |
| `BLOCK_K` | 128 | DQuery/d_rope/DQueryIndex 维度分块大小 |
| `VECTOR_BASEG` | 16 | V scope 中一次循环处理的 Index 头数 |
| `VECTOR_BASEN` | 256 | S2 轴对齐的基本单位 |
| `DQUERY` | 512 | Query/Key 的隐藏层维度（固定） |
| `DQUERY_INDEX` | 128 | QueryIndex/KeyIndex 的隐藏层维度（固定） |

---

## 3. TilingData 数据结构

```python
@dataclass
class SparseLightningIndexerGradKLLossTilingData:
    B: int         # Batch
    S1: int        # Query 序列长度
    S2: int        # Key 序列长度
    N1: int        # Query 头数（32/64/128，官方仅支持 ≥64）
    N2: int        # Key 头数（MQA 固定为 1）
    G: int         # = N1 // N2
    Nidx1: int     # QueryIndex 头数（8/16/32/64）
    Nidx2: int     # KeyIndex 头数（= N2 = 1）
    DQuery: int    # 固定 512
    DQueryIndex: int  # 固定 128
    K: int         # topk，官方仅支持 2048
    sparse_mode: int  # 固定 3
```

### Computed Properties

| 属性 | 公式 | 含义 |
|------|------|------|
| `block_num` | `B * ceil(S1/BLOCK_M) * (K/BLOCK_N)` | 总核数，K 维度并行 |
| `n_num` | `K // BLOCK_N` | K 维度分块数 = 每核处理的 K 块数（Matmul）+ 冗余块数（ReduceSumY） |
| `d_num_query` | `DQuery // BLOCK_K = 4` | Q/K 的 K 轴分块数 |
| `d_num_total` | `d_num_query + ceil(d_rope/BLOCK_K)` | Matmul1 总 K 轴分块（含 rope） |
| `d_num_query_index` | `DQueryIndex // BLOCK_K = 1` | Qidx/Kidx 的 K 轴分块数 |
| `vector_baseg` | `min(Nidx1, 16)` | V scope g_group 批大小 |
| `g_num` | `Nidx1 // vector_baseg` | g_group 分组数 |

---

## 4. Tensor 布局

### 输入 Tensor（Host 预处理后传入内核）

| Tensor | 形状 (BSND) | 含义 |
|--------|-------------|------|
| `Query` | `(B, S1, N1, DQuery)` | Q 输入 |
| `QueryRope` | `(B, S1, N1, d_rope)` | Q 的 RoPE 部分 |
| `GatherKey` | `(B, S1, K, DQuery)` | Host torch.gather 预处理：Key 按 sparse_indices 重排 |
| `GatherKeyRope` | `(B, S1, K, d_rope)` | Host torch.gather：KeyRope 按 sparse_indices 重排 |
| `QueryIndex` | `(B, S1, Nidx1, DQueryIndex)` | LightningIndexer 的 QueryIndex |
| `GatherKeyIndex` | `(B, S1, K, DQueryIndex)` | Host torch.gather：KeyIndex 按 sparse_indices 重排 |
| `GatherKeyIndexT` | `(B, S1, DQueryIndex, K)` | GatherKeyIndex 预转置，供 Matmul4 使用 |
| `Weights` | `(B, S1, Nidx1)` | 权重 |
| `SoftmaxMax` | `(B, N2, S1, G)` | 前向 softmax max 值 |
| `SoftmaxSum` | `(B, N2, S1, G)` | 前向 softmax sum 值 |

### Workspace Tensor

| Tensor | 形状 | 由谁写 | 谁读 | 说明 |
|--------|------|--------|------|------|
| `bmm1Res` | `(B, S1, N1, K)` | C Scope 1（每核写自己的 K 块） | V Scope 1 | Q@GK^T 结果 |
| `bmm2Res` | `(B, S1, Nidx1, K)` | C Scope 1（每核写自己的 K 块） | V Scope 1 | ReLU(QI@GKI^T) 结果 |
| `PReduced` | `(B, S1, K)` | V Scope 1（每核写自己的 K 块） | V Scope 1 | P 的 G 维度均值 |
| `ReduceSumY` | `(B, S1, K)` | V Scope 1（**每核冗余写全部 K 块**） | V Scope 1 | 加权 bmm2Res 和 |
| `SoftmaxI` | `(B, S1, K)` | V Scope 1（**每核冗余写全部 K 块**） | V Scope 1 | ReduceSumY 的 softmax |
| `dQueryIndex_ws` | `(B, S1, Nidx1, DQueryIndex)` | C Scope 2（atomic_add 累加） | V Scope 2 | Matmul4 FP32 中间结果 |

### 输出 Tensor

| Tensor | 形状 | 说明 |
|--------|------|------|
| `dQueryIndex` | `(B, S1, Nidx1, DQueryIndex)` | QueryIndex 的梯度（V Scope 2 Cast 输出） |
| `dWeight_partial` | `(B, S1, n_num, Nidx1)` | 每核部分 dWeight（Python 端 sum 汇总） |
| `reluGradRes` | `(B, S1, Nidx1, K)` | ReLU 梯度（C Scope 2 读取用于 Matmul4） |
| `LossOut` | `(B, S1, n_num * 16)` | 每核部分 KL Loss（Python 端 sum 汇总） |
| `dKeyIndex` | `(B, S2, Nidx2, DQueryIndex)` | KeyIndex 梯度（Python Host ScatterAdd） |

---

## 5. 核分配（Kernel Launch）

```python
block_num = B * s1_tile_num * n_num   # K 维度并行
# 每个 Tile = (batch, s1_tile, k_block) 三元组
```

核 ID 解码：
```python
bz      = cid // (s1_tile_num * n_num)
m       = (cid // n_num) % s1_tile_num    # S1 tile
n_idx_k = cid % n_num                      # K block（每个核只负责一个）
s1_start = m * BLOCK_M
s1_end   = min(s1_start + BLOCK_M, S1)
```

### K 块并行策略

| 计算 | 每核处理范围 | 原因 |
|------|-------------|------|
| Matmul1, Matmul2 | 只算自己的 K 块 | Cube 算力分布 |
| P_reduced | 只算自己的 K 块 | 无跨核依赖 |
| **ReduceSumY** | **冗余遍历全部 K 块** | Global SoftmaxI 需要全部 K 块，TileLang 同 scope 内无可用跨核 barrier |
| Global SoftmaxI | 冗余遍历全部 K 块 | 本来就需遍历 |
| dWeight, reluGrad | 只算自己的 K 块 | 无跨核依赖 |
| Matmul4 (C Scope 2) | 只算自己的 K 块 | **atomic_add 累加**，消除 n_num× 冗余 Cube 计算 |
| Cast (V Scope 2) | 冗余执行全部 | V 核极轻量，可忽略 |

---

## 6. 计算流程总览

```
┌──────────────────────────────────────────────────────────────────┐
│  Single Kernel: 3-Stage CV-Separated + K-Block Parallel           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  C Scope 1:  Matmul1: [Q||QR] @ [GK||GKR]^T → bmm1Res           │
│              Matmul2: QI @ GKI^T → bmm2Res (ReLU)                │
│              每核算 1 个 K 块                                      │
│              │  set_cross_flag("FIX", 1)                         │
│                                                                  │
│  V Scope 1:  wait_cross_flag(1)                                 │
│              Pass 1a: P_reduced[K块]（每核算自己，无冗余）          │
│              Pass 1b: ReduceSumY[全部K块]（★ 每核冗余 16×）       │
│              Pass 1c: Global SoftmaxI (全部K块，本来就冗余)         │
│              Pass 2:  dWeight[Nidx1, K块] + reluGrad[Nidx1, K块]  │
│              Pass 3:  KL Loss (每核算自己，Python sum)              │
│              │  set_cross_flag("V", 12)                          │
│                                                                  │
│  C Scope 2:  wait_cross_flag(12)                                 │
│              Matmul4: rg @ GKIT^T → dQueryIndex_ws               │
│              每核算 1 个 K 块，★ T.tile.atomic_add 累加           │
│              │  set_cross_flag("FIX", 3)                         │
│                                                                  │
│  V Scope 2:  wait_cross_flag(3)                                 │
│              Cast: dQueryIndex_ws FP32 → FP16 (dQueryIndex)       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Python Host:
  1. _preprocess_gather: Key/KeyIndex → GatherKey/GatherKeyIndex
  2. gather_key_index.permute → GatherKeyIndexT
  3. kernel call
  4. dWeight = dWeight_partial.sum(dim=2)
  5. _compute_d_key_index: ScatterAdd → dKeyIndex
```

---

## 7. 各阶段详细解析

### 7.1 C Scope 1: Matmul1 + Matmul2

#### Matmul1: [Query||QueryRope] @ [GatherKey||GatherKeyRope]^T

```
形状: (N1, DQuery+d_rope) @ (K, DQuery+d_rope)^T → (N1, K)
gemm: A(N1, k_block) @ B(n_block, k_block)^T → C(N1, n_block)
d_idx 循环: T.Pipelined(num_stages=2) 重叠 MTE 搬运与 Cube 计算
```

L1/L0C buffer：
- `Q_L1`: `(N1, k_block)` = `(N1, 128)` input_dtype
- `GK_L1`: `(n_block, k_block)` = `(128, 128)` input_dtype
- `C_L0C`: `(N1, n_block)` = `(N1, 128)` calc_dtype

#### Matmul2: QueryIndex @ GatherKeyIndex^T

```
形状: (Nidx1, DQueryIndex) @ (K, DQueryIndex)^T → (Nidx1, K)
gemm: A(Nidx1, k_block) @ B(n_block, k_block)^T → C(Nidx1, n_block)
输出带 enable_relu=True
```

---

### 7.2 V Scope 1: 核心向量计算

#### 数学公式

```
P[g, k] = softmax(bmm1Res[g, k] * scale)  // 使用前向 SoftmaxMax/Sum
P_reduced[k] = (1/G) * Σ_g P[g, k]

reduceSumY[k] = Σ_g (ReLU(bmm2Res[g, k]) * weight[g])
softmaxI[k] = softmax(reduceSumY[k])

diff[k] = softmaxI[k] - P_reduced[k]
dWeight[g] = Σ_k (diff[k] * bmm2Res[g, k])
reluGrad[g, k] = weight[g] * diff[k] * sigmoid'(bmm2Res[g, k])
  // sigmoid' 用 1 - exp(-bmm2Res * 10000) 近似

KL Loss = Σ_k P_reduced[k] * (ln(P_reduced[k]) - ln(softmaxI[k]))
```

#### 关键设计决策：ReduceSumY 冗余

**为什么 ReduceSumY 要冗余遍历全部 K 块？**

Global SoftmaxI 需要读取全部 K 块的 ReduceSumY。在 K 块并行架构下，每个核只写了 1/16 的 ReduceSumY。核间需要同步才能安全读取。

- `T.set_cross_flag + T.wait_cross_flag`：同 scope 内死锁
- `T.sync_all()`：同 scope 内卡死

**唯一可用的方案**：每核遍历全部 n_num 个 K 块，冗余计算全部 ReduceSumY。代价是 V 核工作膨胀 ~16×（约 0.3ms），但消除了跨核数据依赖。

#### SoftmaxI 也用 16 行归约的原因

`T.reduce_max/reduce_sum` 需要 2D buffer 且归约维度有最小元素数要求。对 1D [BLOCK_N=128] 做 reduce，通过复制 16 行变成 [16, 128] 后沿 dim=1 reduce → [16]，取 [0] 即为结果。这是一种 API 适配的 workaround。

---

### 7.3 C Scope 2: Matmul4（atomic_add 优化）

```
dQueryIndex = Σ_k reluGrad[:, k] @ GatherKeyIndexT[:, k]^T
            = Σ_k RG[Nidx1, n_block] @ GKIT[k_block, n_block]^T

原始: 每核遍历全部 n_num 个 K 块，16 核做 256 次 gemm
0622: 每核只算自己的 K 块 (init=True)，T.tile.atomic_add 累加到 dQueryIndex_ws
     16 核共做 16 次 gemm，消除 94% Matmul4 Cube 工作量
```

**Python 端配套**：`d_query_index_ws.zero_()` 在每次内核调用前清零，确保 atomic_add 从 0 累加。

**注意**：由于 V Scope 1 的 ReduceSumY 冗余是当前瓶颈，atomic_add 的收益被遮住。V Scope 1 瓶颈消除后此优化才会显著体现。

---

### 7.4 V Scope 2: Cast

```
dQueryIndex_ws FP32 → Cast(CAST_ROUND) → dQueryIndex FP16
```

所有核冗余执行，V 核算力充裕可忽略。

---

## 8. Python 层计算

### 8.1 Gather 预处理 (`_preprocess_gather`)

```
Key [B,S2,N2,DQ] → squeeze N2 → [B,S2,DQ]
sparse_indices [B,S1,Nidx2,K] → squeeze Nidx2 → [B,S1,K]
torch.gather(dim=1) → reshape → GatherKey [B,S1,K,DQ]
同法处理 KeyIndex 和 KeyRope
```

**为什么在 Host 做**：TileLang `T.copy` 不支持运行时动态索引（`T.copy(Src[idx_ub[0]], dst)` 编译报错 `Can't deduce copy extents`），无法在内核内实现 gather。

### 8.2 ScatterAdd (`_compute_d_key_index`)

```
dKeyIndex = scatter(reluGrad^T @ QueryIndex, sparse_indices)
1. bmm: reluGrad^T [B*S1,K,Nidx1] @ QI [B*S1,Nidx1,DQI] → [B*S1,K,DQI]
2. scatter_add_: 按 sparse_indices 累加到 dKeyIndex [B,S2,Nidx2,DQI]
```

**为什么在 Host 做**：TileLang 无 scatter 操作，同样受限于动态索引。

---

## 9. 与 AscendC 官方实现的差异

| 维度 | TileLang 0622 | 官方 AscendC | 差距原因 |
|------|-------------|-------------|---------|
| **Gather** | Host torch.gather (0.34ms) | 内核内 AIVCore gather | TileLang 不支持动态索引 |
| **Scatter** | Host bmm+scatter_add (0.17ms) | 内核内原子 scatter | 同上 |
| **ReduceSumY** | 每核冗余 16× (0.35ms) | AscendC 原生跨核 barrier | TileLang 同 scope 无可用 barrier |
| **Python 开销** | permute+pad 等 ~0.15ms | 无 | 全在单个内核内 |
| **内核结构** | 3-CV-stage 固定 | 可灵活调度 | TileLang 编译约束 |
| **atomic_add** | C Scope 2 L0C→GM | 原生支持 | 已成功使用 |
| **精度** | rel_err < 5e-4 (BSND) | 基线 | 等价 |
| **总耗时** | ~2.5ms | ~0.32ms | 7.8× 差距 |

---

## 10. 精度与性能测试结果

### 测试通过范围

| 参数 | 支持值 | 不支持 | 原因 |
|------|--------|--------|------|
| K | **2048 唯一** | ≠ 2048 | 官方算子硬限制 |
| N1 | 64, 128 | 32 | 官方 blockDim=0 |
| Nidx1 | 8, 16, 32, 64 | — | 全部通过 |
| S2 | 2048, 4096, 8192 | — | 全部通过 |
| S1 (BSND) | 1~16 | — | 全部通过 |
| TND | — | **不可用** | BSND 内核逐 row 展开，无 S1 并行 |

### 基线精度 (S2=2048, K=2048, N1=128, Nidx1=64)

| 输出 | max_diff | rel_err | 状态 |
|------|----------|---------|------|
| dQueryIndex | 9.77e-4 | 7.6e-5 | ✓ |
| dKeyIndex | 1.46e-4 | 1.8e-4 | ✓ |
| dWeights | 1.56e-2 | 3.3e-4 | ✓ |
| Loss | 0 | 0 | ✓ |

### S2 扩展测试 (K=2048 固定)

| S2 | dQI rel_err | dW rel_err | TL 耗时 | 加速比 |
|-----|------------|-----------|---------|--------|
| 2048 | 7.6e-5 | 3.3e-4 | 1.18ms | 0.27x |
| 4096 | 6.5e-5 | 5.0e-4 | 1.21ms | 0.26x |
| 8192 | 2.4e-3 | 3.1e-3 | 1.20ms | 0.30x |

**TL 耗时不受 S2 影响**，因为内核只处理固定 K=2048，S2 仅影响 Host gather/scatter。

### S1 扩展测试 (BSND, 无官方对比)

| S1 | TL 耗时 | 无 NaN/Inf | 趋势 |
|----|---------|-----------|------|
| 1 | 1.21ms | ✓ | 基线 |
| 2 | 1.61ms | ✓ | sub-linear |
| 4 | 2.44ms | ✓ | |
| 8 | 4.13ms | ✓ | |
| 16 | 7.63ms | ✓ | 16×S1 → 6.3×延迟 |

### 性能瓶颈分解（基线，profiled）

```
阶段      耗时      占比    说明
─────────────────────────────────────
gather   0.34ms    29%    Host torch.gather
prep     0.07ms     6%    permute+pad+cache
kernel   0.63ms    53%    ┬ V Scope 1: 0.35ms (ReduceSumY冗余)
│                         ├ C Scope 1: 0.18ms
│                         ├ C Scope 2: 0.02ms (atomic_add已优化)
│                         └ sync:      0.08ms
reduce   0.02ms     2%    dW.sum + loss.sum
scatter  0.17ms    14%    Host bmm+scatter_add
─────────────────────────────────────
profiled 1.20ms   100%    (不含Python调度等overhead)
```

---

## 11. 优化历程回顾

| 版本 | 关键改动 | 效果 |
|------|---------|------|
| 0611→0612 | K 块并行化 (block_num 扩展) | 并行度 ×n_num，dWeight/Loss 改 partial |
| 0612→0615 | V Scope Cast buffer 缩减 | 减少 UB 占用 |
| 0615→0616 | Rope gather 合并 | 代码简化，微性能提升 |
| 0616→0617 | ReduceSumY 冗余（修复跨核竞争） | **精度从间歇 NaN → 稳定** |
| 0617→0622 | C Scope 2 atomic_add + T.Pipelined(Matmul1) | 消除 Matmul4 冗余 Cube 计算（被 V Scope 1 遮住） |

### 已尝试但失败的优化

| 方向 | 阻塞原因 |
|------|---------|
| V Scope 1 同 scope 跨核 barrier | `cross_flag` 同 scope 死锁，`sync_all` 卡死 |
| T.Parallel 向量化 | 循环大小与向量宽度不匹配 |
| T.Pipelined (ReduceSumY) | 嵌套循环 + clear=False，编译报错/精度异常 |
| Gather 移入内核 | `T.copy` 不支持动态索引 |
| Matmul4 移出内核 (Python bmm) | 缺少 DMA 同步屏障，输出未就绪 |
| 5D partial workspace | TileLang 5D 张量 T.copy 地址计算疑似 bug |
| bool mask (NPU) | NPU 类型转换行为与 CUDA 不同 |

---

## 12. 后续改进方向

| 优先级 | 方向 | 预估收益 | 依赖条件 |
|--------|------|---------|---------|
| P0 | ReduceSumY 消除冗余 | kernel -0.3ms | TileLang 支持同 scope 跨核同步 |
| P1 | 内核内 gather | gather -0.34ms | TileLang 支持动态索引 T.copy |
| P2 | TND 原生支持 | TND 可用 | 重写 TND 专用内核，S1 维度并行 |
| P3 | 进一步微优化 | < 0.1ms | P0/P1 完成后 C Scope 1 成为瓶颈时 |

---

## 13. 测试脚本参数说明

测试脚本位于 `tilelang/precision_test_general/`，各参数含义：

| 脚本 | 参数 | 含义 |
|------|------|------|
| `test_s2_k.py` | S2=2048/4096/8192 | Key 序列长度，影响 Host gather 计算量 |
| | K=2048（固定） | topk 数，官方唯一支持值 |
| `test_n1_nidx1.py` | N1=64/128 | Query 头数，影响 gemm 规模 |
| | Nidx1=8/16/32/64 | QueryIndex 头数，影响 Matmul2/4 和 V 核算量 |
| `test_multi_s1.py` | S1=1/2/4/8/16 | Query 序列长度，影响内核迭代次数 |
| `test_tnd.py` | S1=64/128/512/2048 | TND 格式测试，验证 BSND 内核的 TND 兼容性 |

所有脚本均使用 `B=1, N2=1, Nidx2=1, DQuery=512, DQueryIndex=128, d_rope=64, sparse_mode=3` 固定参数。

---

## 14. TileLang API 约束速查

| API | 约束 |
|-----|------|
| `T.copy` | 索引必须是编译期可确定的常量或循环变量，**不支持运行时动态值** |
| `T.tile.mul/add/sub/exp/div` | 必须在 FP32 buffer 上操作 |
| `T.tile.cast` | FP16↔FP32 转换，half buffer 只能出现在 cast 的源或目标 |
| `T.reduce_sum/reduce_max` | 要求 2D→1D，归约维度需满足最小元素数 |
| `T.tile.atomic_add` | L0C→GM 路径已验证可用（0622），dst 必须是 GM tensor，调用前需 zero_() |
| `T.sync_all` / `T.set_cross_flag+wait` | 同 scope 内不可用（死锁/卡死），仅跨 scope 使用 |
| `T.Parallel` | 循环大小需匹配向量宽度 |
| `T.Pipelined` | 对 Cube-bound 循环无收益，嵌套循环可能编译出错 |
| `range()` vs `T.serial()` | `range` 用于 UB 索引（编译期展开）；`T.serial` 用于 GM 索引（运行时变量） |
