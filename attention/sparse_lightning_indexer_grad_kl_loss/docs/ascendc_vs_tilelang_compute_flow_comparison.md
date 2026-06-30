# AscendC vs TileLang-Ascend (0629) 计算流图详细对比

> 算子: `sparse_lightning_indexer_grad_kl_loss`  
> 对比对象: 官方 AscendC (arch22, Ascend 910B) vs TileLang-Ascend 0629.py 单核改写

---

## 目录

1. [数学公式回顾](#一数学公式回顾)
2. [官方 AscendC 计算流图](#二官方-ascendc-计算流图)
3. [TileLang 0629 计算流图](#三tilelang-0629-计算流图)
4. [逐阶段详细对比](#四逐阶段详细对比)
5. [核心架构差异总结](#五核心架构差异总结)
6. [性能差距根因分析](#六性能差距根因分析)

---

## 一、数学公式回顾

算子计算 LightningIndexer 的 KL loss 反向传播梯度：

**Forward（参考）:**

$$I_{t,:} = W_{t,:} \cdot \text{ReLU}(q_{t,:} @ K_{:t,:}^T)$$

**KL Loss:**

$$L(I) = \sum_t D_{KL}(p_{t,:} \parallel \text{Softmax}(I_{t,:}))$$

**Gradient:**

$$\begin{aligned}
dI_{t,:} &= \text{Softmax}(I_{t,:}) - p_{t,:} \\[4pt]
dW_{t,:} &= dI_{t,:} @ \text{ReLU}(q_{t,:} @ K_{:t,:}^T)^T \\[4pt]
d\text{QueryIndex}_{t,:} &= dS_{t,:} @ K_{:t,:} \\[4pt]
d\text{KeyIndex}_{:t,:} &= \text{scatter\_add}((dS_{t,:})^T @ q_{t,:})
\end{aligned}$$

其中:
- $q$ = query (B,S1,N1,DQuery), $K$ = key (B,S2,N2,DQuery)
- $W$ = weights (B,S1,Nidx1)
- $p$ = 从 attention forward 传入的 softmax_max/softmax_sum 恢复的 target distribution
- $\text{topK}$ = sparse_indices 指定的 top-K 索引
- $dS$ = 经过 ReLU 反向传播的中间梯度 (relu_grad)

---

## 二、官方 AscendC 计算流图

### 2.1 硬件架构

Ascend 910B 使用 **AIC + AIV 双核架构**：
- **AIC (AI Core)**: 负责 Cube 矩阵乘法 + Vector 标量计算（通过 pipe 切换）
- **AIV (AI Vector)**: 专门负责 Vector 标量计算 + DMA

在 arch22 实现中，每个 "task" 由 **1个 AIC + 2个 AIV** 组成（`aicIdx = aivIdx / 2`），共享同一个 workspace。

### 2.2 整体流水线 (MainProcess, 3-stage software pipeline)

```
         s1_idx=0     s1_idx=1     s1_idx=2     s1_idx=3    ...
AIV-V0:  [Gather 0]   [Gather 1]   [Gather 2]   [Gather 3]
AIC-C1:               [MM1+2_0]    [MM1+2_1]    [MM1+2_2]
AIV-V1:               [Softmax 0]  [Softmax 1]  [Softmax 2]
AIC-C2:               [MM5+6_0]    [MM5+6_1]    [MM5+6_2]
AIV-V2:                             [Scatter 0]  [Scatter 1]
```

**关键**: AIC 和 AIV 同时运行在不同 task 上，通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 同步。

### 2.3 逐步详解

#### Step 0: V0 — Gather (AIV 执行)

**功能**: 根据 `sparseIndices` 从 Key/KeyIndex/KeyRope 中 gather 数据到 workspace。

```
输入:
  - Key:       (B, S2, 1, DQuery)        FP16
  - KeyRope:   (B, S2, 1, d_rope)        FP16
  - KeyIndex:  (B, S2, 1, DQueryIndex)   FP16
  - SparseIndices: (B, S1, 1, K)         int32

处理 (以单个 s1_idx 为例):
  1. 读取 top-K 个 sparse_indices[s1_idx] → 得到 K 个 key 位置索引
  2. 对每个有效索引 k_idx (0 ≤ idx < S2):
     - DMA copy Key[idx, :]     → gatherPRes[..., k_idx, :DQuery]
     - DMA copy KeyRope[idx, :] → gatherPRes[..., k_idx, DQuery:]  (拼接在 Key 后面)
     - DMA copy KeyIndex[idx, :]→ gatherSYRes[..., k_idx, :]
  3. 对无效索引 (idx < 0): 跳过（workspace 已初始化为 0）

输出:
  - gatherPRes:  (topK, DQuery + d_rope)  FP16  (Key 和 KeyRope 拼接)
  - gatherSYRes: (topK, DQueryIndex)      FP16  (KeyIndex)
```

**优化**: 
- 双缓冲 (double-buffering): V0 使用 ping-pong buffer，一边 gather 一边被 C1 消费
- Key + KeyRope 在 gather 时直接拼接成一个连续行: `[DQuery | d_rope]`

**同步**: 
- 完成后设置 `CrossCoreSetFlag(SYNC_V0_TO_C1_P_FLAG)` 通知 C1
- 完成后设置 `CrossCoreSetFlag(SYNC_V0_TO_C1_SY_FLAG)` 通知 C1

---

#### Step 1: C1 — Cube MatMul (AIC 执行)

**功能**: 两个矩阵乘法：MM1 (query @ gatherKey^T) 和 MM2 (queryIndex @ gatherKeyIndex^T)。

##### MM1: BMM1 = Query||QueryRope @ GatherP^T

```
输入:
  - Query:     (N1, DQuery)        FP16  (reshape from B,S1,N1,DQuery)
  - QueryRope: (N1, d_rope)        FP16
  - GatherP:   (topK, DQuery+d_rope) FP16  (来自 V0 workspace)

计算:
  1. L1 中预加载 Query 全量 (N1=128, DQuery=512):
     [Query | QueryRope] → L1 buffer shape (128, 576)
     Query 在前，QueryRope 拼接在后 (与 V0 gather 的 Key+Rope 布局对应)
  2. K 维度分块 (outer loop: s2LoopTimes, inner loop: kInnerLoopTimes):
     每次处理 N_SPLIT_SIZE=128 个 K 行
  3. D 维度分块 (dLoopTimes: DQuery/128=4, d_rope/128=1 → 共5次 gemm):
     每次处理 K_SPLIT_SIZE=128 列
     - Query[:, d_start:d_end] @ GatherP[k_start:k_end, d_start:d_end]^T
     - transpose_A=False, transpose_B=True (GatherP 转置)
     - init=(d_idx==0): 第一次 gemm 初始化 L0C，后续累加
  4. 5 次 gemm 累加后 → BMM1 (N1, 128) FP32
  5. Fixpipe L0C→GM (workspace)

输出:
  - bmm1Res: (N1, topK)  FP32  (每 128 行 K 一个分块)
```

**关键参数**:
- singleM = G = N1/N2 = 128 (MQA下N2=1)
- singleN = 128 (K 维度)
- singleK = 128 (D 维度)
- 双缓冲: L1 key buffer ping-pong (V0 写一个，C1 读另一个)

##### MM2: BMM2 = QueryIndex @ GatherKeyIndex^T (with ReLU)

```
输入:
  - QueryIndex:  (Nidx1, DQueryIndex)  FP16
  - GatherSYRes: (topK, DQueryIndex)   FP16  (来自 V0 workspace)

计算:
  1. L1 中预加载 QueryIndex 全量 (Nidx1=64, DQueryIndex=128)
  2. K 维度分块: 每次处理 N_SPLIT_SIZE=128 行
  3. 单次 gemm (DQueryIndex 正好 = 128, 无需分块):
     QueryIndex @ GatherSYRes^T
     - transpose_A=False, transpose_B=True
  4. Fixpipe L0C→GM 时 enable_relu=True (硬件 ReLU 激活)

输出:
  - bmm2Res: (Nidx1, topK)  FP32  (已过 ReLU)
```

**同步**: 
- 等待 `CrossCoreWaitFlag(SYNC_V0_TO_C1_P_FLAG)` 和 `SYNC_V0_TO_C1_SY_FLAG`
- 完成后设置 `CrossCoreSetFlag(SYNC_C1_TO_V1_P_FLAG)` 通知 V1

---

#### Step 2: V1 — Vector Softmax + dWeight + Loss + reluGrad (AIV 执行)

**功能**: 最复杂的 Vector 阶段，在一个 V scope 内完成所有标量计算。

```
输入:
  - bmm1Res:     (N1, topK)    FP32
  - softmaxMax:  (G,)          FP32  (来自 forward)
  - softmaxSum:  (G,)          FP32  (来自 forward)
  - bmm2Res:     (Nidx1, topK) FP32  (已过 ReLU)
  - weights:     (Nidx1,)      FP16

处理步骤 (对每个 s1_idx 循环):

  Step 2a: 计算 P = Softmax(BMM1)
    1. P = BMM1 * scale_value
    2. 对每个 head g ∈ [0, G):
       P[g, :] = exp(P[g, :] - softmaxMax[g]) / softmaxSum[g]
    3. P_reduced[k] = mean(P[:, k]) over G heads  → (topK,) FP32

  Step 2b: 计算 ReduceSumY (对所有 K blocks 遍历)
    对每个 K-block n_rsy:
      1. Y = bmm2Res[:, n_rsy*128:(n_rsy+1)*128]  → (Nidx1, 128)
      2. 按 group 分块 (vector_baseg=16):
         weighted_Y[g, :] = sum_i(weights[i] * Y[g*16+i, :])
      3. ReduceSumY[:, n_rsy*128:(n_rsy+1)*128] = weighted_Y

  Step 2c: 计算 Global SoftmaxI
    SoftmaxI = Softmax(ReduceSumY)  over all K positions
    1. 全局 max reduce (遍历所有 K blocks)
    2. exp(ReduceSumY - max), 全局 sum reduce
    3. SoftmaxI = exp_vals / total_sum

  Step 2d: 计算 dWeight
    diff = SoftmaxI - P_reduced
    dWeight[g, i] = sum_k(diff[k] * bmm2Res[g*16+i, k])  (对当前 K-block)
    输出到 dWeight_partial (每个 K-block 一份，host 侧 sum)

  Step 2e: 计算 reluGrad
    mask = sigmoid_approx(bmm2Res)  (用 exp 近似)
    reluGrad[g*16+i, :] = weights[i] * diff * mask

  Step 2f: 计算 KL Loss (partial)
    loss = P_reduced * (log(P_reduced) - log(SoftmaxI))
    对当前 K-block 求和 → loss_out partial

输出:
  - P_reduced:     (topK,)       FP32
  - ReduceSumY:    (topK,)       FP32
  - SoftmaxI:      (topK,)       FP32
  - dWeight_partial: (n_num, Nidx1) FP16  (每个 K-block 一份)
  - reluGradRes:   (Nidx1, topK) FP16
  - LossOut:       (n_num*16,)   FP32  (partial per K-block)
```

**同步**: 
- 等待 `CrossCoreWaitFlag(SYNC_C1_TO_V1_P_FLAG)` 和 `SYNC_C1_TO_V1_SY_FLAG`
- 完成后设置 `CrossCoreSetFlag(SYNC_V1_TO_C2_DW_FLAG)` 通知 C2

---

#### Step 3: C2 — Cube MatMul (AIC 执行)

**功能**: MM5 (reluGrad^T @ queryIndex → bmm3Res) 和 MM6 (reluGrad @ gatherKeyIndex → dQueryIndex)。

##### MM5: BMM3 = reluGrad^T @ QueryIndex

```
输入:
  - reluGradRes:  (Nidx1, topK)     FP16
  - QueryIndex:   (Nidx1, DQueryIndex) FP16

计算 (等价于 dKeyIndex 的 bmm):
  1. reluGrad 分块 (kLoopTimes): 每次取 RELU_GRAD_SPLIT_SIZE=2048 行 K
  2. 逐 K_SPLIT_SIZE=128 做 gemm:
     reluGrad[k:k+128, :]^T  @  QueryIndex
     - transpose_A=True (reluGrad 转置!):  (Nidx1, 128) → (128, Nidx1)
     - transpose_B=True (QueryIndex 转置!): (Nidx1, 128) → (128, Nidx1)
     - 结果: (128, DQueryIndex)  FP32
  3. ScatterAdd 到 bmm3Res (每个 core 有自己独立的 workspace 区域)
  4. Fixpipe 带 F322F16 量化到 output dtype

输出:
  - bmm3Res: (topK, DQueryIndex)  FP32  (workspace, 各 core 独立区域)
```

**注意**: MM5 的 `transpose_A=True, transpose_B=True` 是官方代码真实使用的！这说明 AscendC 的 `transpose_A=True` 是可用的，与我们在 TileLang 中遇到的 `transpose_A` 疑似 bug 形成对比。

##### MM6: dQueryIndex = reluGrad @ GatherKeyIndex

```
输入:
  - reluGradRes:     (Nidx1, topK)        FP16
  - GatherKeyIndex:  (topK, DQueryIndex)   FP16  (来自 V0 workspace)

计算:
  1. reluGrad 分块: 每次取 K_SPLIT_SIZE=128 行
  2. 逐次 gemm:
     reluGrad[:, k:k+128] @ GatherKeyIndex[k:k+128, :]
     - transpose_A=False, transpose_B=True
     - init=(first_iter): 第一次初始化，后续 L0C 累加
  3. Fixpipe L0C→dQueryIndexGm (直接输出到 GM, 带 F322F16 量化)

输出:
  - dQueryIndex: (Nidx1, DQueryIndex)  FP16/BF16  (最终输出)
```

**同步**: 
- 等待 `CrossCoreWaitFlag(SYNC_V1_TO_C2_DW_FLAG)`
- MM5 完成后设置 `CrossCoreSetFlag(SYNC_C2_TO_V2_SA_FLAG)` 通知 V2

---

#### Step 4: V2 — ScatterAdd + Loss Reduce (AIV 执行)

**功能**: 
1. 将 bmm3Res (gathered dKeyIndex) scatter-add 回原始 key 位置
2. 多 core 间的 partial loss 汇总

```
输入:
  - scatterAddRes: (S2 * DQueryIndex)  FP32  (workspace, 各 core 分段初始化)
  - topKIndex:     (topK,)             int32
  - lossRes:       (n_cores,)          FP32  (每个 core 一个 partial loss)
  - bmm3Res:       (topK, DQueryIndex) FP32

处理:
  对每个有效 top-K 位置 k:
    1. idx = topKIndex[k]  (原始 key 位置)
    2. if idx >= 0:
       scatterAddRes[idx, 0:DQueryIndex] += bmm3Res[k, 0:DQueryIndex]
  
  注意: 每个 AIV core 负责 S2 的一段区间 (t2Start:t2End)，
        只有落在自己区间内的 scatter 才执行，避免多核写入冲突。

  Loss:
    遍历所有 core 的 lossRes → reduce sum → lossGm[0]

输出:
  - dKeyIndex: (S2, 1, DQueryIndex)  FP16/BF16  (最终输出，从 scatterAddRes copy)
  - loss:      scalar                 FP32       (最终输出)
```

**注意**: 官方 AscendC 的 scatter 在 **设备侧 V2 scope 内** 完成，使用硬件 DMA scatter 指令，不是 host 侧的 `scatter_add_`。

---

### 2.4 AscendC 数据流总结图

```
GM Inputs                    Workspace (per-core pairs)           GM Outputs
───────                      ─────────────────────────           ──────────

Key ─────────┐
KeyRope ─────┤
KeyIndex ────┤               ┌─────────────────────┐
             ├──[V0:Gather]──┤ gatherPRes           │
SparseIndices┘               │ (topK, DQuery+DRope) │
                             │ gatherSYRes          │
                             │ (topK, DQueryIndex)  │
                             └──────┬──────────────┘
                                    │
Query ─────────┐                     ├──[C1:MM1]──→ bmm1Res (N1,topK)
QueryRope ─────┤                     │
               ├──[C1]───────────────┤
QueryIndex ────┤                     ├──[C1:MM2]──→ bmm2Res (Nidx1,topK) [ReLU]
               └─────────────────────┘                    │
                                                         │
softmaxMax ──────────────────────────────────────────────┤
softmaxSum ──────────────────────────────────────────────┤
weights ─────────────────────────────────────────────────┤
                    ┌────────────────────────────────────┘
                    │
                    └──[V1:Softmax+P_reduced+dWeight+Loss+reluGrad]──→ P_reduced, SoftmaxI
                         │                                              dWeight_partial
                         │                                              reluGradRes, LossOut
                         │
          ┌──────────────┴──────────────────────┐
          │                                     │
[C2:MM5] reluGrad^T @ QueryIndex      [C2:MM6] reluGrad @ GatherKeyIndex
     │                                        │
     └──→ bmm3Res (topK, DQI)                └──→ dQueryIndex ───────────→ Output
              │
              │
SparseIndices─┴──[V2:ScatterAdd]──→ dKeyIndex ────────────────────────────→ Output
lossRes ────────[V2:LossReduce]──→ loss ──────────────────────────────────→ Output
dWeight_partial (host-side sum) ──→ dWeight ──────────────────────────────→ Output
```

**同步标志流**:
```
V0 --(SYNC_V0_TO_C1)--> C1(MM1+MM2) --(SYNC_C1_TO_V1)--> V1 --(SYNC_V1_TO_C2)--> C2(MM5+MM6) --(SYNC_C2_TO_V2)--> V2
```

---

## 三、TileLang 0629 计算流图

### 3.1 硬件架构

TileLang-Ascend 0629 使用 **单核架构**（每个 core 包含 Cube + Vector 单元，通过 scope 切换）：
- `T.Scope("C")`: Cube 单元，执行矩阵乘法
- `T.Scope("V")`: Vector 单元，执行标量计算和 DMA

**核分解策略**: `block_num = B × ceil(S1/BLOCK_M) × (K // BLOCK_N)`
- 沿 K 维度并行: 每个 core 处理 BLOCK_N=128 个 K 行
- 当 K=2048 时，有 16 个 core 并行

### 3.2 逐步详解

#### Step 0: Host-side Gather (Python)

**这部分在 AscendC 中是 V0 scope，但在 0629 中被移到 host 侧。**

```
输入:
  - Key:       (B, S2, 1, DQuery)        FP16
  - KeyIndex:  (B, S2, 1, DQueryIndex)   FP16
  - KeyRope:   (B, S2, 1, d_rope)        FP16
  - SparseIndices: (B, S1, 1, K)         int32

处理 (_preprocess_gather):
  1. key_all = cat([Key, KeyIndex, KeyRope], dim=-1)  → (B, S2, 1, total_dim)
  2. idx_all = sparse_indices.clamp(min=0).expand(...)
  3. gather_all = torch.gather(key_all, dim=1, idx_all)
  4. gather_all *= valid_mask (将无效索引位置清零)
  5. split:
     - gather_key       = gather_all[..., :DQuery]
     - gather_key_index = gather_all[..., DQuery:DQuery+DQueryIndex]
     - gather_key_rope  = gather_all[..., DQuery+DQueryIndex:]

输出 (传入 kernel):
  - GatherKey:       (B, S1, K, DQuery)        FP16
  - GatherKeyRope:   (B, S1, K, d_rope)        FP16
  - GatherKeyIndex:  (B, S1, K, DQueryIndex)   FP16
  - GatherKeyIndexT: (B, S1, DQueryIndex, K)   FP16  (预转置，供 C2 使用)
```

**性能代价**: ~0.04ms (cat + gather + split + permute)

---

#### Step 1: C1 Scope — MatMul1 + MatMul2 (Cube)

**与 AscendC C1 对应，但每个 core 只处理自己的 K-block。**

##### MM1: BMM1 = Query||QueryRope @ GatherKey^T

```
输入:
  - Query:          (N1, 128)        FP16  (D 维度分块加载)
  - QueryRope:      (N1, 128)        FP16  (padded d_rope)
  - GatherKey_ws:   (128, 128)       FP16  (K-block 切片)
  - GatherKeyRope_ws:(128, 128)      FP16  (K-block 切片)

计算:
  for s1_row in T.serial(s1_start, s1_end):        # 通常 1 个 row
    for d_idx in T.Pipelined(5, num_stages=3):      # 5 个 D 分块, 3-stage 流水
      if d_idx < 4:
        Q = Query[:, d_idx*128:(d_idx+1)*128]       # DQuery 维度
        K = GatherKey[k_start:k_end, d_idx*128:(d_idx+1)*128]
      else:
        Q = QueryRope[:, 0:128]                     # rope 维度 (padded)
        K = GatherKeyRope[k_start:k_end, 0:128]
      gemm_v0(Q, K, C, transpose_B=True, init=(d_idx==0))
    T.copy(C_L0C → bmm1Res)                          # 输出到 GM workspace

输出:
  - bmm1Res: (N1, 128)  FP32  (每个 core 写自己的 K-block 切片)
```

##### MM2: BMM2 = QueryIndex @ GatherKeyIndex^T (with ReLU)

```
输入:
  - QueryIndex:         (Nidx1, 128)  FP16
  - GatherKeyIndex_ws:  (128, 128)    FP16

计算:
  for s1_row:
    for k_idx in T.serial(1):                       # DQueryIndex=128, 正好 1 块
      QI = QueryIndex[:, 0:128]
      GKI = GatherKeyIndex[k_start:k_end, 0:128]
      gemm_v0(QI, GKI, C, transpose_B=True, init=True)
    T.copy(C_L0C → bmm2Res, enable_relu=True)        # 带 ReLU

输出:
  - bmm2Res: (Nidx1, 128)  FP32
```

**同步**: `T.set_cross_flag("FIX", 1)` 通知 V1

---

#### Step 2: V1 Scope — Vector 计算 (全部标量操作)

**与 AscendC V1 对应，但在单 core 的 Vector 单元上串行执行。**

```
输入 (与前一步通过 workspace 传递):
  - bmm1Res:    (N1, 128)       FP32
  - bmm2Res:    (Nidx1, 128)    FP32
  - softmaxMax: (G,)            FP32
  - softmaxSum: (G,)            FP32
  - weights:    (Nidx1,)        FP16

处理 (对 s1_id 循环, 通常是 1 次):
  
  Step 2a: P = Softmax(BMM1)                       [仅当前 K-block]
    1. P = bmm1Res * scale_value (T.tile.mul)
    2. 对每个 head:
       P[g,:] = exp(P[g,:] - softmaxMax[g]) / softmaxSum[g]
    3. P_reduced = mean(P, axis=G)
    4. 输出到 PReduced workspace

  Step 2b: ReduceSumY                            [遍历所有 K blocks!]
    for n_rsy in T.serial(n_num):                  # 16 次遍历 (K=2048)
      1. 读取 bmm2Res[:, n_rsy*128:(n_rsy+1)*128]
      2. 按 group 分块, weight * bmm2, reduce_sum
      3. 输出到 ReduceSumY workspace

  Step 2c: Global SoftmaxI                       [遍历所有 K blocks!]
    1. 全局 max reduce: 遍历 n_num=16 次
       - 每次读取 ReduceSumY chunk → 16 路并行 max
    2. 全局 sum reduce: 遍历 n_num=16 次
       - exp(chunk - max), 16 路并行 sum
    3. SoftmaxI = exp_vals / total_sum
    4. 输出到 SoftmaxI workspace

  Step 2d: dWeight + reluGrad + Loss             [仅当前 K-block]
    1. diff = SoftmaxI[my_k] - P_reduced[my_k]
    2. 对每个 group:
       - dWeight[g] += diff * bmm2Res[g, :] (reduce_sum)
       - reluGrad[g,:] = weights * diff * sigmoid_mask(bmm2Res)
    3. Loss = P_reduced * (log(P_reduced) - log(SoftmaxI))
    4. 输出到 dWeight_partial, reluGradRes, LossOut workspace

输出:
  - PReduced:         (128,)       FP32  (仅当前 K-block)
  - ReduceSumY:       (K,)         FP32  (全量, 所有 core 冗余计算)
  - SoftmaxI:         (K,)         FP32  (全量, 所有 core 冗余计算)
  - dWeight_partial:  (n_num, Nidx1) FP16
  - reluGradRes:      (Nidx1, 128)  FP16
  - LossOut:          (n_num*16,)  FP32
```

**同步**: `T.set_cross_flag("V", 12)` 通知 C2

**⚠️ 关键差异**: 0629 中每个 core **冗余计算** 全量 ReduceSumY 和 SoftmaxI（遍历所有 16 个 K-block），而 AscendC 中这是由 AIV 独立处理的。

---

#### Step 3: C2 Scope — MatMul4 (Cube)

**与 AscendC C2 对应。只做 MM4 (dQueryIndex)，不做 MM5 (dKeyIndex)。**

##### MM4: dQueryIndex_ws += reluGrad @ GatherKeyIndex

```
输入:
  - reluGradRes:      (Nidx1, 128)            FP16
  - GatherKeyIndexT:  (DQueryIndex, K)        FP16  (预转置!)

计算:
  for s1_row:
    for d_idx in T.serial(1):                         # DQueryIndex=128, 1 块
      RG = reluGradRes[:, :]                          # (Nidx1, 128)
      GKI_T = GatherKeyIndexT[d_idx*128:, k_start:k_end]  # (128, 128) from transposed
      gemm_v0(RG, GKI_T, DQ, transpose_B=True, init=True)
      T.tile.atomic_add(dQueryIndex_ws[:, d_idx*128:], DQ)  # 跨 core 累加!

输出:
  - dQueryIndex_ws: (Nidx1, DQueryIndex)  FP32  (通过 atomic_add 跨 core 累加)
```

**⚠️ 注意**: 0629 的 MM4 等价于 AscendC 的 MM6，但：
- AscendC: 使用 GatherKeyIndex (非转置) + transpose_B=True
- 0629: 使用预转置的 GatherKeyIndexT + transpose_B=True（数学等价）
- 差异原因: 0629 的 GatherKeyIndexT 在 host 侧预计算（一次 permute），省去 kernel 内转置

**同步**: `T.set_cross_flag("FIX", 3)` 通知 V3

---

#### Step 4: V3 Scope — Cast FP32→FP16 (Vector)

**这是 0629 独有的 scope（AscendC MM6 自带量化）。**

```
输入:
  - dQueryIndex_ws: (Nidx1, DQueryIndex)  FP32

处理:
  for s1_row:
    for g_id in range(Nidx1/4):
      for d_idx in range(DQueryIndex/128):
        T.copy(ws[...] → ub)                     # FP32 chunk
        T.tile.cast(ub → ub_half, "CAST_ROUND")  # FP32→FP16
        T.copy(ub_half → dQueryIndex)            # 输出

输出:
  - dQueryIndex: (Nidx1, DQueryIndex)  FP16  (最终输出)
```

---

#### Step 5: Host-side Postprocessing

**与 AscendC V2 对应，但在 host 侧完成。**

```
dWeight: d_weight = dWeight_partial.sum(dim=2)       # 跨 K-block 求和
Loss:    kl_loss = loss_values.sum()                  # 跨 K-block 求和

dKeyIndex (scatter_add):
  1. reluGrad^T @ QueryIndex → scatter_add (FP16 torch.bmm)
     relu_grad_t: (K, Nidx1), query_index: (Nidx1, DQI)
     → (K, DQI) per S1
  2. d_key_index.scatter_add_(dim, sparse_indices, scatter_add)
  3. dtype cast

性能代价: ~0.15ms (bmm) + ~0.05ms (scatter) = ~0.20ms
```

---

### 3.3 0629 数据流总结图

```
                      Host Side                    │           Kernel Side (16 cores)
                                                   │
Key ───────┐                                       │
KeyRope ───┤                                       │
KeyIndex ──┤                                       │
           ├─[torch.cat]→[torch.gather]→[split]───┐│
SparseIndices┘                     │              ││
                                   │              ││
                  ┌────────────────┘              ││
                  │                               ││
          GatherKey (B,S1,K,DQuery) ──────────────┼│→ [C1:MM1] Query @ GK^T → bmm1Res
          GatherKeyRope (B,S1,K,d_rope) ──────────┼│→            (pipelined, num_stages=3)
          GatherKeyIndex (B,S1,K,DQueryIndex) ────┼│→ [C1:MM2] QI @ GKI^T → bmm2Res [ReLU]
          GatherKeyIndexT (B,S1,DQueryIndex,K) ────┼│→           (host预转置)
                                                   ││
          Query, QueryRope ────────────────────────┼│→ [C1]
          QueryIndex ──────────────────────────────┼│→ [C1]
                                                   ││
          weights ─────────────────────────────────┼│→ [V1:Softmax+dW+Loss+reluGrad]
          softmaxMax, softmaxSum ──────────────────┼│→
                                                   ││      ↓ cross_flag
                                                   ││
          reluGradRes ─────────────────────────────┼│→ [C2:MM4] RG @ GKI^T → [atomic_add] → dQI_ws
                                                   ││
                                                   ││      ↓ cross_flag
                                                   ││
                                                   ││→ [V3:Cast] FP32→FP16 → dQueryIndex ──→ Output
                                                   ││
                                                   ││  reluGradRes ──────────────────────────→ (host)
                                                   ││      │
dKeyIndex ← [scatter_add_] ← [torch.bmm] ←────────┼│──────┘
dWeight   ← [sum over K blocks] ←─────────────────┼│── dWeight_partial
Loss      ← [sum over K blocks] ←─────────────────┼│── LossOut
```

---

## 四、逐阶段详细对比

### 4.1 Gather 阶段

| 维度 | AscendC (V0) | TileLang 0629 |
|------|:------------:|:-------------:|
| **执行位置** | 设备侧 AIV core | **Host 侧 Python** |
| **实现方式** | HW DMA gather with double-buffering | `torch.cat` + `torch.gather` + split |
| **Key+Rope 布局** | 在 gather 时直接拼接成一行 | 先 cat 后 gather 再 split |
| **并行度** | 每个 AIV 处理独立 s1, P/SY 两个 ping-pong | 串行 (host 单线程) |
| **与后续流水** | 与 C1 3-stage pipeline 重叠 | 完全串行, kernel 等待 gather 完成 |
| **耗时** | 被紧随的 C1 掩盖 | ~0.04ms 暴露在关键路径上 |

### 4.2 MM1/MM2 (Cube MatMul) 阶段

| 维度 | AscendC (C1) | TileLang 0629 (C1) |
|------|:------------:|:-------------------:|
| **并行策略** | 沿 S1 维度分 core | **沿 K 维度分 core** |
| **单 core 工作量** | 全量 K (2048) | BLOCK_N=128 (1/16 K) |
| **D 维度流水** | 5 次 gemm, 双缓冲 L1 | 5 次 gemm, `T.Pipelined(num_stages=3)` |
| **L1 缓冲** | 双缓冲 (ping-pong) | 单缓冲 (Pipelined 内部管理) |
| **输出位置** | 双缓冲 workspace | 直接写 bmm1Res/bmm2Res workspace |
| **transpose_B** | ✅ `isRightTranspose=true` | ✅ `transpose_B=True` |
| **ReLU** | Fixpipe 硬件 ReLU | `T.copy(enable_relu=True)` |

### 4.3 Softmax / dWeight / Loss / reluGrad (Vector) 阶段

| 维度 | AscendC (V1) | TileLang 0629 (V1) |
|------|:------------:|:-------------------:|
| **执行单元** | 专用 AIV core (独立硬件) | 共享 Vector 单元 (与 Cube 分时) |
| **ReduceSumY** | AIV 遍历所有 K blocks | **每个 core 冗余遍历所有 16 个 K blocks** |
| **SoftmaxI** | AIV 冗余计算 (单 core 可见全量) | **每个 core 冗余计算 (16×128 遍历)** |
| **标量循环** | C++ 编译优化, 部分向量化 | `T.serial` + `for i in range(16)` **标量循环** |
| **数据类型** | FP32 (中间), FP16 I/O | FP32 (calc_dtype), FP16 I/O |
| **同步** | CrossCoreSetFlag↔WaitFlag | `T.set_cross_flag` + `T.wait_cross_flag` 同 core 内 |

**⚠️ 性能关键**: 0629 的 V1 中 SoftmaxI 计算需要遍历 16 个 K-block，每个 block 的 reduce max/sum 又展开为 16 路并行。由于 `T.tile.broadcast` 不支持 1D→2D 扩展，所有操作都是标量形式，**这是 0629 性能的最大瓶颈**。

### 4.4 dQueryIndex MatMul (MM6) 阶段

| 维度 | AscendC (C2:MM6) | TileLang 0629 (C2:MM4) |
|------|:----------------:|:----------------------:|
| **输入 B 矩阵** | GatherKeyIndex (非转置) | GatherKeyIndexT (host 预转置) |
| **gemm 参数** | `transpose_A=False, transpose_B=True` | `transpose_B=True` |
| **GatherKeyIndexT** | N/A (直接使用 V0 输出) | host 侧 `permute(0,1,3,2).contiguous()` |
| **输出累加** | Fixpipe 直接写 dQueryIndexGm | `T.tile.atomic_add` 跨 core 累加到 dQueryIndex_ws |
| **量化** | Fixpipe F322F16 (硬件) | V3 scope 单独 `T.tile.cast` (软件) |

### 4.5 dKeyIndex (MM5 + Scatter) 阶段

| 维度 | AscendC (C2:MM5 + V2) | TileLang 0629 |
|------|:---------------------:|:-------------:|
| **MM5 执行位置** | 设备侧 AIC (Cube) | **Host 侧** `torch.bmm` |
| **MM5 transpose_A** | ✅ `isLeftTranspose=true` | N/A (host bmm 不需要) |
| **Scatter 执行位置** | 设备侧 AIV (V2 scope) | **Host 侧** `scatter_add_` |
| **Scatter 实现** | HW DMA scatter + 区间分段并行 | PyTorch `scatter_add_` (CANN 优化) |
| **Loss reduce** | AIV 遍历所有 core 的 lossRes, cross-core 通信 | Host 侧 `loss_values.sum()` |
| **总耗时** | 被流水掩盖 | ~0.20ms 暴露在关键路径上 |

**⚠️**: 官方 AscendC 的 MM5 使用了 `transpose_A=True` 和 `transpose_B=True`，这证明在原生 AscendC 中 transpose_A 是正常工作的。TileLang 中 `T.gemm_v0(transpose_A=True)` 导致 ~100% 误差是一个 **TileLang 特有的 bug**。

---

## 五、核心架构差异总结

### 5.1 双核 vs 单核

```
AscendC (910B):
┌──────────────────────────────────────────────────────┐
│  AIC Core 0         AIV Core 0      AIV Core 1       │
│  ┌─────────┐       ┌─────────┐     ┌─────────┐       │
│  │ C1 MM1  │──┐    │ V0 Gath │     │ V0 Gath │       │
│  │ C1 MM2  │  │    │ V1 Soft │     │ V1 Soft │       │
│  │ C2 MM5  │  │    │ V2 Scat │     │ V2 Scat │       │
│  │ C2 MM6  │  │    └─────────┘     └─────────┘       │
│  └─────────┘  │                                       │
│       │       │    CrossCoreSetFlag/WaitFlag          │
│       └───────┴───────────── 同步 ──────────────────  │
└──────────────────────────────────────────────────────┘

TileLang 0629:
┌──────────────────────────────────────┐
│           Single Core                 │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐     │
│  │ C1 │→│ V1 │→│ C2 │→│ V3 │        │
│  └────┘  └────┘  └────┘  └────┘     │
│  严格串行, 无重叠                      │
└──────────────────────────────────────┘
```

### 5.2 流水线深度对比

| | AscendC | TileLang 0629 |
|------|:------:|:------------:|
| **总 Stage 数** | 5 (V0→C1→V1→C2→V2) | 4 (C1→V1→C2→V3) |
| **流水重叠** | 3-stage software pipeline (3 个 s1 同时进行) | 无 (严格串行) |
| **D 维度流水** | 5-pass, 双缓冲 MTE1↔MTE2 | 5-pass, `T.Pipelined(num_stages=3)` |
| **K 维度并行** | 无 (单 core 处理全量 K) | 16 路 (K//128 core 并行) |

### 5.3 在核 vs 离核

| 操作 | AscendC | TileLang 0629 |
|------|:------:|:------------:|
| **Gather** | ✅ 在核 (V0, 被掩盖) | ❌ 离核 (host, ~0.04ms) |
| **dKeyIndex bmm (MM5)** | ✅ 在核 (C2, 被掩盖) | ❌ 离核 (host, ~0.15ms) |
| **dKeyIndex scatter** | ✅ 在核 (V2, 被掩盖) | ❌ 离核 (host, ~0.05ms) |
| **dQueryIndex MM** | ✅ 在核 (C2:MM6) | ✅ 在核 (C2:MM4) |
| **Loss reduce** | ✅ 在核 (V2) | ❌ 离核 (host sum) |

### 5.4 冗余计算对比

| 计算 | AscendC | TileLang 0629 |
|------|:------:|:------------:|
| **ReduceSumY** | 每个 AIV 独立计算各自 s1 的全量 | **16 个 core 各算一遍全量** |
| **SoftmaxI** | 每个 AIV 独立计算 | **16 个 core 各算一遍全量** |
| **P_reduced** | 每个 AIV 只算自己的 s1 | 每个 core 只算自己的 K-block ✅ |

---

## 六、性能差距根因分析

以 Case 0 (B=1,S1=1,S2=2048,N1=128,Nidx1=64,K=2048) 为例：

| 项目 | 官方 AscendC | TileLang 0629 | 差距 |
|------|:-----------:|:------------:|:---:|
| **总耗时** | ~0.31ms | ~0.90ms | **0.34x** |
| Gather | 被掩盖 | ~0.04ms | host 侧 |
| C1 (MM1+MM2) | 被掩盖 | ~0.25ms | K-block 并行有效 |
| V1 (Softmax等) | 部分掩盖 | **~0.35ms** | 标量循环 + 16× 冗余 |
| C2 (MM6→dQI) | 被掩盖 | ~0.05ms | atomic_add |
| MM5 (dKI bmm) | 被掩盖 | ~0.15ms | host 侧 bmm |
| Scatter (dKI) | 被掩盖 | ~0.05ms | host 侧 scatter_add_ |
| Loss reduce | 被掩盖 | ~0.01ms | host 侧 sum |

### 根因排序 (影响从大到小)

1. **V1 标量循环** (~0.15ms 损失): SoftmaxI 遍历 + dWeight/reluGrad 循环无法向量化（`T.tile.broadcast` 不支持 1D→2D）

2. **V1 冗余计算** (~0.10ms 损失): 16 个 core 各算一遍全量 ReduceSumY 和 SoftmaxI

3. **dKeyIndex 离核** (~0.20ms 损失): MM5 (bmm) + scatter 在 host 侧串行执行

4. **无 3-stage 流水** (~0.10ms 损失): TileLang 的 Scope 模型不支持 C/V 交叉流水

5. **Gather 离核** (~0.04ms 损失): host 侧 torch.gather

---

## 附录: 符号对照表

| 符号 | AscendC 变量名 | TileLang 0629 变量名 | 含义 |
|------|:------------:|:------------------:|------|
| B | bSize | B | Batch size |
| S1 | s1Size | S1 | Query 序列长度 |
| S2 | s2Size | S2 | Key 序列长度 |
| N1 | gSizeQuery | N1 | Query head 数 |
| N2 | n2Size | N2 | Key head 数 (MQA=1) |
| G | gSizeQuery (same as N1) | G = N1//N2 | Head 组数 |
| Nidx1 | gSizeQueryIndex | Nidx1 | QueryIndex head 数 |
| Nidx2 | N/A (=1) | Nidx2 | KeyIndex head 数 |
| DQuery | dSizeQuery (512) | DQuery (512) | Query 维度 |
| DQueryIndex | dSizeQueryIndex (128) | DQueryIndex (128) | Index 维度 |
| d_rope | dSizeQueryRope (64) | d_rope (64) | RoPE 维度 |
| K | topKSize | K | Top-K 稀疏数 |
| topKSize | topKSize | K | 同上 |
| scale_value | scaleValue | scale_value | Attention scale |
| gatherPRes | gatherPRes | GatherKey + GatherKeyRope | Gather 的 P 部分 |
| gatherSYRes | gatherSYRes | GatherKeyIndex | Gather 的 SY 部分 |
| bmm1Res | bmm1Res | bmm1Res (workspace) | MM1 结果 |
| bmm2Res | bmm2Res | bmm2Res (workspace) | MM2 结果 (带ReLU) |
| bmm3Res | bmm5Res (MM5结果) | N/A (host侧) | dKeyIndex 中间结果 |
| reluGradRes | reluGradRes | reluGradRes | ReLU 梯度 |
| psySyncGm | P_reduced + SoftmaxI | PReduced + SoftmaxI | Vector→Cube 同步数据 |
