# AscendC vs TileLang-Ascend (0630) 计算流图详细对比

> 算子: `sparse_lightning_indexer_grad_kl_loss`  
> 对比对象: 官方 AscendC (arch22, Ascend 910B) vs TileLang-Ascend 0630.py 单核改写（Gather 在核版）

---

## 目录

1. [数学公式回顾](#一数学公式回顾)
2. [官方 AscendC 计算流图](#二官方-ascendc-计算流图)
3. [TileLang 0630 计算流图](#三tilelang-0630-计算流图)
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

## 三、TileLang 0630 计算流图

### 3.1 硬件架构

TileLang-Ascend 0630 使用 **单核架构**（每个 core 包含 Cube + Vector 单元，通过 scope 切换）：
- `T.Scope("V")`: Vector 单元，执行 DMA gather + 标量计算 + cast
- `T.Scope("C")`: Cube 单元，执行矩阵乘法

**核分解策略**: `block_num = B × ceil(S1/BLOCK_M) × (K // BLOCK_N)`
- 沿 K 维度并行: 每个 core 处理 BLOCK_N=128 个 K 行
- 当 K=2048, S1=1 时，有 `1 × 1 × 16 = 16` 个 core 并行

**5-Scope 串行架构**:
```
V0 (gather) → C1 (MM1+MM2) → V1 (softmax/dW/Loss/reluGrad) → C2 (MM4) → V3 (cast)
```

### 3.2 逐步详解

#### Step 0: V0 Scope — In-Kernel Gather (Vector)

**0630 将 gather 移入 kernel**，每个 core 只 gather 自己负责的 K-block 行（k_start:k_end）。

```
输入:
  - Key:          (B, S2, 1, DQuery)          FP16
  - KeyRope:      (B, S2, 1, d_rope)          FP16
  - KeyIndex:     (B, S2, 1, DQueryIndex)     FP16
  - SparseIndices:(B, S1, 1, K)               int32

处理 (每个 core):
  for s1_id in T.serial(s1_start, s1_end):
    for k_id in T.serial(k_start, k_end):           # 仅 gather 本 core 的 K-block (128 行)
      idx = SparseIndices[bz, s1_id, 0, k_id]
      if idx >= 0:
        DMA copy Key[idx, :]       → GatherKey_ws[k_id, :]           # 1×512 FP16
        DMA copy KeyIndex[idx, :]  → GatherKeyIndex_ws[k_id, :]      # 1×128 FP16
        if d_rope > 0:
          DMA copy KeyRope[idx, :] → GatherKeyRope_ws[k_id, :]       # 1×64  FP16

输出 (写入 workspace):
  - GatherKey_ws:       (B, S1, K, DQuery)        FP16  (各 core 写自己的 k_start:k_end 切片)
  - GatherKeyRope_ws:   (B, S1, K, d_rope)        FP16
  - GatherKeyIndex_ws:  (B, S1, K, DQueryIndex)   FP16
```

**与 0629 的区别**:
- 0629: Host 侧 `torch.cat` + `torch.gather` + split → 3 个 gather workspace 传入 kernel (~0.04ms)
- 0630: Kernel 内 V0 scope DMA gather → 直接写 workspace（被后续 C1 掩盖，不暴露在关键路径）

**同步**: `T.set_cross_flag("V", 10)` 通知 C1

---

#### Step 1: C1 Scope — MatMul1 + MatMul2 (Cube)

**与 AscendC C1 对应，从 V0 写入的 gather workspace 中读取。**

##### MM1: BMM1 = Query||QueryRope @ GatherKey^T

```
输入:
  - Query:                (N1, 128)           FP16  (D 维度分块加载)
  - QueryRope:            (N1, 128)           FP16  (padded d_rope)
  - GatherKey_ws:         (128, 128)          FP16  (V0 写入的 K-block 切片)
  - GatherKeyRope_ws:     (128, 128)          FP16

计算:
  for s1_row in T.serial(s1_start, s1_end):          # 通常 1 个 row
    for d_idx in T.Pipelined(d_num_total, num_stages=3):  # 4+1=5 个 D 分块, 3-stage
      if d_idx < d_num_query:                        # d_num_query = DQuery/128 = 4
        Q = Query[:, d_idx*128:(d_idx+1)*128]         # DQuery 维度
        K = GatherKey_ws[k_start:k_end, d_idx*128:(d_idx+1)*128]
      else:                                           # rope 维度
        Q = QueryRope[:, 0:128]                       # padded d_rope
        K = GatherKeyRope_ws[k_start:k_end, 0:128]
      gemm_v0(Q, K, C_L0C, transpose_B=True, init=(d_idx==0))
    T.copy(C_L0C → bmm1Res[...])                      # 输出到 GM workspace

输出:
  - bmm1Res: (N1, 128)  FP32  (每个 core 写自己的 K-block 切片)
```

##### MM2: BMM2 = QueryIndex @ GatherKeyIndex^T (with ReLU)

```
输入:
  - QueryIndex:           (Nidx1, 128)  FP16
  - GatherKeyIndex_ws:    (128, 128)    FP16  (V0 写入)

计算:
  for s1_row:
    for k_idx in T.serial(d_num_query_index):         # DQueryIndex/128 = 1
      QI = QueryIndex[:, k_idx*128:(k_idx+1)*128]
      GKI = GatherKeyIndex_ws[k_start:k_end, k_idx*128:(k_idx+1)*128]
      gemm_v0(QI, GKI, C2_L0C, transpose_B=True, init=(k_idx==0))
    T.copy(C2_L0C → bmm2Res, enable_relu=True)        # 硬件 ReLU

输出:
  - bmm2Res: (Nidx1, 128)  FP32 (已过 ReLU)
```

**同步**: `T.set_cross_flag("FIX", 1)` 通知 V1

---

#### Step 2: V1 Scope — Vector 计算 (全部标量操作, barrier 已合并)

**与 AscendC V1 对应，在一个 V scope 内完成所有标量计算。0630 已移除 11 个不必要的 VEC→VEC barrier。**

```
输入 (通过 workspace 传递):
  - bmm1Res:    (N1, 128)       FP32
  - bmm2Res:    (Nidx1, 128)    FP32 (已过 ReLU)
  - softmaxMax: (G,)            FP32
  - softmaxSum: (G,)            FP32
  - weights:    (Nidx1,)        FP16

处理 (对 s1_id 循环):

  Step 2a: P_reduced = Softmax(BMM1) · mean over G     [仅当前 K-block]
    1. P = bmm1Res * scale_value (T.tile.mul, 1 条 VEC)
    2. for g_id in range(G=128):                        # 标量循环, tracer 展开
       P[g,:] = exp(P[g,:] - softmaxMax[g]) / softmaxSum[g]
       (sub → exp → div, 3×128=384 条 VEC 指令)
    3. P_reduced = mean(P, axis=0) over G heads
       (add 累加 + mul 1/G, 129 条 VEC 指令)
    4. 输出到 PReduced workspace

  Step 2b: ReduceSumY = Σ_i(weights[i] * bmm2Res[i,:])  [遍历所有 K blocks!]
    for n_rsy in T.serial(n_num):                      # 16 次迭代 (K=2048)
      1. 读取 bmm2Res[:, n_rsy*128:(n_rsy+1)*128]       # 全量 K 维度!
      2. for g_id in range(g_num=4):                    # 按 vector_baseg=16 分 4 组
         weight[0:16] * bmm2Res[g*16:(g+1)*16, :]       # 标量循环 16×mul
         reduce_sum → rsy_reduce_ub
      3. 输出到 ReduceSumY workspace

  Step 2c: Global SoftmaxI = Softmax(ReduceSumY)       [遍历所有 K blocks!]
    1. 全局 max reduce: 遍历 n_num=16 次
       - 每次 16 路并行 max (16×T.tile.mul → T.reduce_max)
    2. exp(ReduceSumY - max), 输出到 SoftmaxI workspace
    3. 全局 sum reduce: 遍历 n_num=16 次
       - 每次 16 路并行 sum (broadcast → T.reduce_sum)
    4. SoftmaxI = exp_vals / total_sum

  Step 2d: dWeight + reluGrad + Loss                  [仅当前 K-block]
    1. diff = SoftmaxI[my_k] - P_reduced[my_k]        (T.tile.sub, 1 条)
    2. for g_id in range(g_num=4):                     # 按 group 分块
       - dWeight[g] += diff * bmm2Res[g*16:(g+1)*16,:] (标量循环, reduce_sum)
       - reluGrad = weights * diff * sigmoid_mask(bmm2Res)  (标量循环×7 ops)
    3. Loss = P_reduced * (ln(P_reduced) - ln(SoftmaxI))  (T.tile.ln + T.tile.sub + T.tile.mul)
    4. Loss: 16 路 broadcast → reduce_sum → LossOut

输出:
  - PReduced:         (128,)         FP32  (仅当前 K-block)
  - ReduceSumY:       (K,)           FP32  (全量, 所有 core 冗余计算)
  - SoftmaxI:         (K,)           FP32  (全量, 所有 core 冗余计算)
  - dWeight_partial:  (n_num, Nidx1) FP16  (每个 K-block 一份, host 侧 sum)
  - reluGradRes:      (Nidx1, 128)   FP16
  - LossOut:          (n_num*16,)    FP32  (每个 K-block 一份)
```

**同步**: `T.set_cross_flag("V", 12)` 通知 C2

**⚠️ 0630 的 barrier 优化**: 原 0629 有 ~26 个 `T.barrier_all()`，0630 移除了 VEC→VEC 方向连续操作间的 11 个 barrier（只保留 VEC→MTE/DMA 的 barrier），V1 scope 内 barrier 从 26 降至 15 个。

---

#### Step 3: C2 Scope — MatMul4 (Cube)

**与 AscendC C2:MM6 对应。直接读取 GatherKeyIndex_ws（不再需要预转置）。**

##### MM4: dQueryIndexAccum += reluGrad @ GatherKeyIndex

```
输入:
  - reluGradRes:          (Nidx1, 128)            FP16
  - GatherKeyIndex_ws:    (128, 128)              FP16  (V0 写入, 非转置)

计算:
  for s1_row:
    for d_idx in T.serial(d_num_query_index):           # DQueryIndex/128 = 1
      RG = reluGradRes[:, :]                            # (Nidx1, 128)
      GKI = GatherKeyIndex_ws[k_start:k_end, d_idx*128:(d_idx+1)*128]  # (128, 128)
      gemm_v0(RG, GKI, DQ_L0C, transpose_B=False, init=True)
      T.tile.atomic_add(dQueryIndexAccum[:, d_idx*128:(d_idx+1)*128], DQ_L0C)  # 跨 core FP32 累加

输出:
  - dQueryIndexAccum: (Nidx1, DQueryIndex)  FP32  (跨 core atomic_add 累加)
```

**与 0629 的区别**:
- 0629: 使用 host 预转置的 `GatherKeyIndexT (DQueryIndex, K)` + `transpose_B=True` → 等价于 `reluGrad @ GKI^T`
- 0630: 直接使用 `GatherKeyIndex_ws (K, DQueryIndex)` + `transpose_B=False` → 数学等价于 `reluGrad @ GKI`（不是转置），省去 host 侧 permute

**同步**: `T.set_cross_flag("FIX", 3)` 通知 V3

---

#### Step 4: V3 Scope — Cast FP32→FP16 (Vector)

```
输入:
  - dQueryIndexAccum: (Nidx1, DQueryIndex)  FP32

处理:
  for s1_row:
    for g_id in T.serial(Nidx1/4):                    # Nidx1 分 4 个一组
      for d_idx in T.serial(DQueryIndex/128):          # D 维度分块
        T.copy(Accum[g_id*4:(g_id+1)*4, d_idx*128:...] → ub)   # FP32 chunk (4×128)
        T.tile.cast(ub_half, ub_fp32, "CAST_ROUND", 4*128)      # FP32→FP16
        T.copy(ub_half → dQueryIndex[...])             # 输出

输出:
  - dQueryIndex: (Nidx1, DQueryIndex)  FP16  (最终输出)
```

---

#### Step 5: Host-side Postprocessing

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

**⚠️ dKeyIndex 仍在 host 侧**: 0630 尚未将 dKeyIndex 的 MM5 (transpose_A=True) 移入 kernel，原因是 TileLang `T.gemm_v0(transpose_A=True)` 存在 bug（~100% 误差）。

---

### 3.3 0630 数据流总结图

```
                      Host Side                    │           Kernel Side (16 cores)
                                                   │
Key ───────────────────────────────────────────────┼│→ [V0:Gather] per-core DMA gather
KeyRope ───────────────────────────────────────────┼│→   → GatherKey_ws, GatherKeyRope_ws, GatherKeyIndex_ws
KeyIndex ──────────────────────────────────────────┼│→   (各 core 只写自己的 k_start:k_end 切片)
SparseIndices ─────────────────────────────────────┼│→
                                                   ││      ↓ cross_flag(10)
                                                   ││
Query, QueryRope ──────────────────────────────────┼│→ [C1:MM1] Q @ GK^T → bmm1Res (pipelined, num_stages=3)
QueryIndex ────────────────────────────────────────┼│→ [C1:MM2] QI @ GKI^T → bmm2Res [ReLU]
                                                   ││      ↓ cross_flag(1)
                                                   ││
weights ───────────────────────────────────────────┼│→ [V1:Softmax+dW+Loss+reluGrad]
softmaxMax, softmaxSum ────────────────────────────┼│→   (15 barrier, VEC→VEC 已合并)
                                                   ││      ↓ cross_flag(12)
                                                   ││
reluGradRes ───────────────────────────────────────┼│→ [C2:MM4] RG @ GKI → [atomic_add] → dQueryIndexAccum (FP32)
                                                   ││      ↓ cross_flag(3)
                                                   ││
dQueryIndex ← [V3:Cast] FP32→FP16 ←───────────────┼│── dQueryIndexAccum
                                                   ││
                                                   ││  reluGradRes ────────────────────────────→ (host)
                                                   ││      │
dKeyIndex ← [scatter_add_] ← [torch.bmm] ←────────┼│──────┘
dWeight   ← [sum over K blocks] ←─────────────────┼│── dWeight_partial
Loss      ← [sum over K blocks] ←─────────────────┼│── LossOut
```

---

## 四、逐阶段详细对比

### 4.1 Gather 阶段

| 维度 | AscendC (V0) | TileLang 0630 (V0) |
|------|:------------:|:-------------------:|
| **执行位置** | 设备侧 AIV core | **设备侧 V scope** |
| **实现方式** | HW DMA gather with double-buffering | `T.serial` + per-element DMA copy |
| **Key+Rope 布局** | 在 gather 时直接拼接成一行 | 分别 gather 到独立的 workspace |
| **并行度** | 每个 AIV 处理独立 s1, P/SY 两个 ping-pong | **16 core 并行**, 每 core gather 128 行 |
| **与后续流水** | 与 C1 3-stage pipeline 重叠 | 串行 (V0→C1), 但同一 core 内 DMA→Cube 切换 |
| **耗时** | 被紧随的 C1 掩盖 | 在 kernel 内 (~0.02ms), 小于 host 侧 gather |

### 4.2 MM1/MM2 (Cube MatMul) 阶段

| 维度 | AscendC (C1) | TileLang 0630 (C1) |
|------|:------------:|:-------------------:|
| **并行策略** | 沿 S1 维度分 core | **沿 K 维度分 core** |
| **单 core 工作量** | 全量 K (2048) | BLOCK_N=128 (1/16 K) |
| **D 维度流水** | 5 次 gemm, 双缓冲 L1 | 5 次 gemm, `T.Pipelined(num_stages=3)` |
| **L1 缓冲** | 双缓冲 (ping-pong) | 单缓冲 (Pipelined 内部管理) |
| **输入数据来源** | V0 双缓冲 workspace | **V0 gather workspace** (同 core, 无 ping-pong) |
| **transpose_B** | ✅ `isRightTranspose=true` | ✅ `transpose_B=True` |
| **ReLU** | Fixpipe 硬件 ReLU | `T.copy(enable_relu=True)` |

### 4.3 Softmax / dWeight / Loss / reluGrad (Vector) 阶段

| 维度 | AscendC (V1) | TileLang 0630 (V1) |
|------|:------------:|:-------------------:|
| **执行单元** | 专用 AIV core (独立硬件) | 共享 Vector 单元 (与 Cube 分时) |
| **ReduceSumY** | AIV 遍历所有 K blocks | **每个 core 冗余遍历所有 16 个 K blocks** |
| **SoftmaxI** | AIV 冗余计算 (单 core 可见全量) | **每个 core 冗余计算 (16×128 遍历)** |
| **标量循环** | C++ 编译优化, 部分向量化 | `T.serial` + `for g_id in range(G)` **标量循环展开** |
| **数据类型** | FP32 (中间), FP16 I/O | FP32 (calc_dtype), FP16 I/O |
| **Barrier 数量** | HW 自动管理 | 15 个 (0630 已合并 VEC→VEC, 原 0629 有 ~26 个) |
| **同步** | CrossCoreSetFlag↔WaitFlag (跨核) | `T.set_cross_flag`/`T.wait_cross_flag` (同核 scope 间) |

**⚠️ 性能关键**: 0630 的 V1 中 SoftmaxI 仍需要遍历 16 个 K-block，且所有操作都是标量形式（`T.tile.broadcast` 不支持 1D→2D 扩展）。这仍是最大瓶颈，但 barrier 合并已减少 ~10% V1 耗时。

### 4.4 dQueryIndex MatMul (MM6/MM4) 阶段

| 维度 | AscendC (C2:MM6) | TileLang 0630 (C2:MM4) |
|------|:----------------:|:----------------------:|
| **输入 B 矩阵** | GatherKeyIndex (非转置) | **GatherKeyIndex_ws (非转置, V0 写入)** |
| **gemm 参数** | `transpose_A=False, transpose_B=True` | `transpose_B=False` |
| **与 0629 的区别** | — | 0629 用预转置 GKI^T + transpose_B=True; 0630 直接读 GKI + transpose_B=False |
| **输出累加** | Fixpipe 直接写 dQueryIndexGm | `T.tile.atomic_add` 跨 core FP32 累加 |
| **量化** | Fixpipe F322F16 (硬件) | V3 scope 单独 `T.tile.cast` (软件) |

### 4.5 dKeyIndex (MM5 + Scatter) 阶段

| 维度 | AscendC (C2:MM5 + V2) | TileLang 0630 |
|------|:---------------------:|:-------------:|
| **MM5 执行位置** | 设备侧 AIC (Cube) | **Host 侧** `torch.bmm` |
| **MM5 transpose_A** | ✅ `isLeftTranspose=true` | N/A (host bmm 不需要) |
| **Scatter 执行位置** | 设备侧 AIV (V2 scope) | **Host 侧** `scatter_add_` |
| **Scatter 实现** | HW DMA scatter + 区间分段并行 | PyTorch `scatter_add_` (CANN 优化) |
| **Loss reduce** | AIV 遍历所有 core 的 lossRes, cross-core 通信 | Host 侧 `loss_values.sum()` |
| **总耗时** | 被流水掩盖 | ~0.11ms (host scatter, 比 0629 的 ~0.20ms 优化) |

**⚠️**: 官方 AscendC 的 MM5 使用了 `transpose_A=True` 和 `transpose_B=True`，这证明在原生 AscendC 中 transpose_A 是正常工作的。TileLang 中 `T.gemm_v0(transpose_A=True)` 导致 ~100% 误差是一个 **TileLang 特有的 bug**。这阻止了 dKeyIndex 计算移入 kernel。

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

TileLang 0630:
┌─────────────────────────────────────────────────┐
│                  Single Core                      │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐        │
│  │ V0 │→│ C1 │→│ V1 │→│ C2 │→│ V3 │           │
│  │Gath│  │Cube│  │Vec │  │Cube│  │Cast│           │
│  └────┘  └────┘  └────┘  └────┘  └────┘        │
│  严格串行, 5 scopes, 无 CV 重叠                   │
└─────────────────────────────────────────────────┘
```

### 5.2 流水线深度对比

| | AscendC | TileLang 0630 |
|------|:------:|:------------:|
| **总 Stage 数** | 5 (V0→C1→V1→C2→V2) | 5 (V0→C1→V1→C2→V3) |
| **流水重叠** | 3-stage software pipeline (3 个 s1 同时进行) | 无 (严格串行) |
| **D 维度流水** | 5-pass, 双缓冲 MTE1↔MTE2 | 5-pass, `T.Pipelined(num_stages=3)` |
| **K 维度并行** | 无 (单 core 处理全量 K) | 16 路 (K//128 core 并行) |
| **Gather 位置** | 在核 V0 (AIV) | **在核 V0 (Vector scope)** ✅ |

### 5.3 在核 vs 离核

| 操作 | AscendC | TileLang 0630 |
|------|:------:|:------------:|
| **Gather** | ✅ 在核 (V0, AIV, 被掩盖) | ✅ **在核 (V0 scope)** |
| **dKeyIndex bmm (MM5)** | ✅ 在核 (C2, 被掩盖) | ❌ 离核 (host, ~0.15ms) |
| **dKeyIndex scatter** | ✅ 在核 (V2, 被掩盖) | ❌ 离核 (host, ~0.05ms) |
| **dQueryIndex MM** | ✅ 在核 (C2:MM6) | ✅ 在核 (C2:MM4) |
| **Loss reduce** | ✅ 在核 (V2) | ❌ 离核 (host sum) |

### 5.4 冗余计算对比

| 计算 | AscendC | TileLang 0630 |
|------|:------:|:------------:|
| **ReduceSumY** | 每个 AIV 独立计算各自 s1 的全量 | **16 个 core 各算一遍全量** (冗余) |
| **SoftmaxI** | 每个 AIV 独立计算 | **16 个 core 各算一遍全量** (冗余) |
| **P_reduced** | 每个 AIV 只算自己的 s1 | 每个 core 只算自己的 K-block ✅ |
| **Gather** | 每个 AIV gather 全量 K 行 | **16 core 分摊 gather**, 每 core 128 行 ✅ |

---

## 六、性能差距根因分析

以 Case 0 (B=1,S1=1,S2=2048,N1=128,Nidx1=64,K=2048) 为例：

| 项目 | 官方 AscendC | TileLang 0630 | 差距 |
|------|:-----------:|:------------:|:---:|
| **总耗时** | ~0.31ms | ~0.88ms | **0.35x** |
| Gather | 被掩盖 | ✅ 在核 (~0.02ms kernel内) | 已解决 |
| C1 (MM1+MM2) | 被掩盖 | ~0.25ms | K-block 并行有效 |
| V1 (Softmax等) | 部分掩盖 | **~0.33ms** | 标量循环 + 16× 冗余 (barrier 已优化 ~10%) |
| C2 (MM4→dQI) | 被掩盖 | ~0.05ms | atomic_add |
| V3 (cast) | 被掩盖 | ~0.02ms | 软件 cast |
| MM5 (dKI bmm) | 被掩盖 | ~0.08ms | host 侧 bmm (已优化) |
| Scatter (dKI) | 被掩盖 | ~0.03ms | host 侧 scatter_add_ |
| Loss reduce + dWeight sum | 被掩盖 | ~0.02ms | host 侧 sum |

### 0630 vs 0629 改进

| 改进项 | 0629 | 0630 | 收益 |
|--------|:----:|:----:|:----:|
| Gather 位置 | host (~0.04ms) | 在核 V0 | -0.04ms 从关键路径移除 |
| Barrier 数量 (V1) | ~26 个 | 15 个 | V1 耗时 -10% |
| dKeyIndex host 侧 | ~0.20ms | ~0.11ms | bmm+scatter 优化 |
| **总耗时** | ~0.90ms | **~0.88ms** | -0.02ms |

### 剩余瓶颈 (影响从大到小)

1. **V1 标量循环** (~0.15ms 损失): SoftmaxI + dWeight/reluGrad 循环无法向量化（`T.tile.broadcast` 不支持 1D→2D）

2. **V1 冗余计算** (~0.10ms 损失): 16 个 core 各算一遍全量 ReduceSumY 和 SoftmaxI（无跨核同步机制）

3. **dKeyIndex 离核** (~0.11ms 损失): MM5 (bmm) + scatter 在 host 侧串行执行，等待 TileLang 修复 `transpose_A` bug

4. **无 CV 流水** (~0.08ms 损失): TileLang Scope 模型不支持 C/V 交叉流水，5 个 scope 严格串行

5. **UB Buffer 压力** (间接): 30+ UB buffer 分配，Memory Planning 有一定开销

---

## 附录: 符号对照表

| 符号 | AscendC 变量名 | TileLang 0630 变量名 | 含义 |
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
| scale_value | scaleValue | scale_value | Attention scale |
| gatherPRes | gatherPRes | GatherKey_ws + GatherKeyRope_ws | Gather 的 Key+Rope (V0 写入) |
| gatherSYRes | gatherSYRes | GatherKeyIndex_ws | Gather 的 KeyIndex (V0 写入) |
| bmm1Res | bmm1Res | bmm1Res (workspace) | MM1 结果 (Q@K^T) |
| bmm2Res | bmm2Res | bmm2Res (workspace) | MM2 结果 (QI@KI^T, 带ReLU) |
| bmm3Res | bmm5Res (MM5结果) | N/A (host侧) | dKeyIndex 中间结果 |
| reluGradRes | reluGradRes | reluGradRes | ReLU 梯度 |
| dQueryIndexAccum | N/A (Fixpipe 直接输出) | dQueryIndexAccum | dQueryIndex FP32 累加器 |
