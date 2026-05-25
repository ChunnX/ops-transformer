# Lightning Indexer Grad 算子详细分析

## 目录
- [1. 算子概述](#1-算子概述)
- [2. 计算公式详解](#2-计算公式详解)
- [3. 实现架构](#3-实现架构)
- [4. 数据流分析](#4-数据流分析)
- [5. 核心算子实现](#5-核心算子实现)
- [6. 参数说明](#6-参数说明)
- [7. 约束条件](#7-约束条件)

---

## 1. 算子概述

### 1.1 功能定位
`lightning_indexer_grad` 是 `lightning_indexer` 正向算子的反向算子，用于训练场景下的梯度计算。该算子基于正向计算输出的 `sparseIndices`（TopK索引），高效计算 `query`、`key`、`weights` 的梯度。

### 1.2 核心特点
- **稀疏计算优化**：利用正向的 TopK 索引，只计算关键位置的梯度，大幅降低计算量
- **双核流水线架构**：采用 AIV（Vector核）+ AIC（Cube核）协同计算
- **多级 Pipeline**：通过 Ping-Pong 缓冲实现高效流水并行
- **支持确定性计算**：可选确定性模式保证计算结果可复现

---

## 2. 计算公式详解

### 2.1 正向公式回顾

对于某个 token 对应的 Index Query $Q_{index}\in\R^{g\times d}$，给定上下文 Index Key $K_{index}\in\R^{S_{k}\times d}$ 和权重 $W\in\R^{g\times 1}$，其中：
- $g$ 为 GQA 对应的 group size
- $d$ 为每一个头的维度（通常为 128）
- $S_{k}$ 是上下文 Key 的长度

正向计算公式：

$$
Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
$$

**公式步骤解析：**

1. **计算注意力分数**：$Q_{index}@K_{index}^T \rightarrow (g, S_k)$
   - 每个 group 的 Query 与所有 Key 计算内积
   
2. **ReLU 激活**：$\text{ReLU}(Q_{index}@K_{index}^T) \rightarrow (g, S_k)$
   - 激活后的注意力分数，记为 $S$

3. **权重广播**：$W@[1]_{1\times S_k} \rightarrow (g, S_k)$
   - 将权重 $W$ 广播到每个 Key token 位置

4. **加权注意力**：$(W@[1])\odot S \rightarrow (g, S_k)$
   - 每个 group 的加权注意力分数，记为 $Y$

5. **Group 求和**：$[1]_{1\times g}@Y \rightarrow (1, S_k)$
   - 对 group 维度求和，得到每个 Key token 的总重要性分数

6. **TopK 选择**：选出重要性最高的 $k$ 个 Key token 索引

**简化表示**：
- $S = \text{ReLU}(Q_{index}@K_{index}^T)$ - 注意力分数矩阵
- $Y = (W@[1])\odot S$ - 加权注意力分数
- 最终输出：TopK 个 Key token 的索引

### 2.2 反向梯度计算公式

反向梯度计算基于正向公式：

$$
Y = (W@[1]_{1\times S_k})\odot\text{ReLU}(Q_{index}@K_{index}^T)
$$

反向计算步骤：

```
# Step 1: ReLU 反向梯度
# 根据 TopK 索引，从 Key 中 Gather 对应数据
KeyGather = Gather(Key, sparseIndices)  # (topK, D)

# 计算 ReLU 前的中间结果
reluIn = Matmul(Query, KeyGather^T)      # (g, topK)

# ReLU 反向：只传递 reluIn > 0 的位置
reluGrad = Dy ⊙ (reluIn > 0)             # Dy 为上层传来的梯度

# Step 2: weights 梯度
# dW = Reduce(reluIn ⊙ Dy) 对每个 group 求和
dW = ReduceSum(reluIn ⊙ Dy, axis=topK)   # (g,)

# Step 3: query 梯度
# dQ = reluGrad @ KeyGather
dQ = Matmul(reluGrad, KeyGather)         # (g, D)

# Step 4: key 梯度
# dK = ScatterAdd(reluGrad^T @ Query, sparseIndices)
scatterAdd = Matmul(reluGrad^T, Query)   # (topK, D)
dK = ScatterAdd(scatterAdd, sparseIndices)  # 累加到 Key 原位置
```

**公式解析：**
```
┌─────────────────────────────────────────────────────────────────┐
│                    反向传播数学流程                               │
└─────────────────────────────────────────────────────────────────┘

输入:
├── Q ∈ R^{g×d}          (Query)
├── K ∈ R^{S_k×d}        (Key)
├── W ∈ R^{g×1}          (Weights)
├── Dy ∈ R^{g×topK}      (上层梯度)
├── Indices ∈ Z^{topK}   (TopK索引)

Step 1: GatherTopk
│   K_gather = Gather(K, Indices) ∈ R^{topK×d}
│
Step 2: Cube1
│   reluIn = Q @ K_gather^T ∈ R^{g×topK}
│
Step 3: Cube2
│   reluGrad_in = W ⊙ Dy ∈ R^{g×topK}
│
Step 4: Vec2
│   ├── mask = (reluIn > 0)
│   ├── reluGrad_out = reluGrad_in ⊙ mask ∈ R^{g×topK}
│   └── dW = ReduceSum(reluIn ⊙ Dy, axis=topK) ∈ R^{g}
│
Step 5: Cube3
│   dQ = reluGrad_out @ K_gather ∈ R^{g×d}
│
Step 6: Cube4
│   scatterAdd = reluGrad_out^T @ Q ∈ R^{topK×d}
│
Step 7: Vec3
│   dK[Indices[j]] += scatterAdd[j]  (AtomicAdd)
│
输出:
├── dQ ∈ R^{g×d}
├── dK ∈ R^{S_k×d}
├── dW ∈ R^{g}
```

| 梯度 | 计算含义 | 形状变化 |
|------|---------|---------|
| `dW` | 对每个 group，累加 reluIn×Dy 在 TopK 维度的值 | (B, S1, N1) |
| `dQ` | ReLU反向梯度与聚合后的 Key 做矩阵乘 | (B, S1, N1, D) |
| `dK` | 将梯度 scatter 到 Key 原位置并原子累加 | (B, S2, N2, D) |

**反向公式推导逻辑：**

1. **ReLU 反向**：正向有 ReLU 激活，反向时只有 reluIn > 0 的位置才传递梯度
2. **Weights 梯度**：W 在正向中做广播乘法，反向需要对广播维度求和
3. **Query 梯度**：Q 参与 Q@K^T 的 MatMul，反向需要 K 参与 MatMul
4. **Key 梯度**：K 参与 Q@K^T 的 MatMul，反向需要 Q 参与 MatMul，并用 ScatterAdd 将梯度放回原位置

### 2.4 正向与反向的关系说明

**重要理解**：

正向公式输出的是 **TopK 索引**（哪些 Key token 重要），反向公式需要计算 **每个参数的梯度**（参数该怎么调整）。

反向计算利用正向输出的 `sparseIndices`（TopK 索引）来：
1. **降低计算量**：只计算 TopK 个 Key 的梯度，而不是全部 S_k 个
2. **Gather 操作**：根据索引提取对应的 Key 数据
3. **ScatterAdd 操作**：将计算出的梯度累加回 Key 的原位置

**为什么反向公式看起来与正向公式不完全一致？**

正向公式包含 `Top-k` 操作，这是一个**不可逆的选择操作**：
- 正向：选出 TopK 索引，丢弃其他位置的信息
- 反向：只能为 TopK 位置计算梯度，其他位置梯度为 0

因此反向公式中的 `Gather(Key, sparseIndices)` 就是正向公式中隐含的"选择 TopK 个 Key"的逆操作。

---

## 3. 实现架构

### 3.1 硬件资源分配

| 资源类型 | 数量 | 用途 |
|---------|------|------|
| AIC (Cube核) | 24个 | 矩阵乘法计算 (MatMul) |
| AIV (Vector核) | 48个 | 向量运算 (Gather/Scatter/ReluGrad) |
| UB (Unified Buffer) | ~189KB | 向量计算的中间缓冲 |
| L1/L0 Buffer | 多级缓冲 | Cube 计算的数据缓冲 |

### 3.2 软件流水线设计

算子采用 **三段式流水** 设计：

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Stage 1                          │
│  Vec1: GatherTopk                                                │
│  ├── 从 sparseIndices 加载索引                                   │
│  └── 根据索引从 Key Gather TopK 数据                             │
│  Cube1: Query @ KeyGather^T + ReLU                              │
│  └── 计算 reluIn (G, K)                                         │
│  Cube2: Weights @ Dy                                            │
│  └── 计算 reluGrad (G, K)                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Stage 2                          │
│  Vec2: ReluGrad + ReduceSum                                     │
│  ├── 计算 reluGradOut                                           │
│  └── 计算 dWeights (Reduce)                                     │
│  Cube3: reluGrad @ KeyGather                                    │
│  └── 计算 dQuery (G, D)                                         │
│  Cube4: reluGrad^T @ Query                                      │
│  └── 计算 scatterAdd (K, D)                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Stage 3                          │
│  Vec3: ScatterAdd                                               │
│  └── 将 scatterAdd 累加到 dKey                                  │
│  └── 支持 deterministic 模式的归约                              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 核间同步机制

使用 **CrossCore Flag** 实现核间同步：

| Flag ID | 名称 | 用途 |
|---------|------|------|
| 7 | SYNC_V1_C1_FLAG | Vec1 完成 → Cube1 开始 |
| 8 | SYNC_C1_V2_FLAG | Cube1/2 完成 → Vec2 开始 |
| 9 | SYNC_V2_C2_FLAG | Vec2 完成 → Cube3/4 开始 |
| 10 | SYNC_C2_V3_FLAG | Cube 完成 → Vec3 开始 |
| 11 | SYNC_C2_V1_FLAG | Cube2 完成 → Vec1 下一轮 |

---

## 4. 数据流分析

### 4.1 输入数据流向

```
                         输入 Tensor
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
    Query              Key                  Dy
        │                     │                     │
        │                     ▼                     │
        │         ┌─────────────────────┐          │
        │         │   Vec1: GatherTopk  │          │
        │         │   根据 Indices       │          │
        │         │   从 Key 提取数据    │          │
        │         └─────────────────────┘          │
        │                     │                     │
        │              KeyGather (TopK×D)           │
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────┐
│              Cube1: MatMul + ReLU                        │
│   Query(G, D) @ KeyGather^T(D, K) → (G, K)              │
│   + ReLU 激活                                            │
│   输出: reluIn (G, K)                                    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
    reluIn (float)
```

### 4.2 梯度计算流向

```
Weights + Dy
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              Cube2: 广播乘法                             │
│   Weights(G, 1) @ Dy(1, K) → reluGrad(G, K)             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              Vec2: ReluGrad + Reduce                     │
│   1. ReluGrad: reluGrad * (reluIn > 0)                  │
│   2. ReduceSum: dW = reduce(S * Dy)                     │
│   输出: reluGradOut, dWeights                            │
└─────────────────────────────────────────────────────────┘
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  │
┌───────────────┐ ┌───────────────┐      │
│   Cube3       │ │   Cube4       │      │
│ reluGrad @    │ │ reluGrad^T @  │      │
│ KeyGather     │ │ Query         │      │
│               │ │               │      │
│ 输出: dQuery  │ │ 输出: scatter │      │
└───────────────┘ └───────────────┘      │
                        │                 │
                        ▼                 │
                ┌───────────────┐         │
                │   Vec3        │         │
                │ ScatterAdd    │         │
                │ 累加到 dKey   │         │
                └───────────────┘         │
                        │                 │
                        ▼                 ▼
                    dKey              dWeights
```

### 4.3 Workspace 数据布局

```
Workspace 内存布局:
┌─────────────────────────────────────────────────────────────────┐
│  dkWorkSpace        │  dkCoreWorkspace  │  keyGatherWorkspace  │
│  (float累加缓冲)    │  (确定性模式用)   │  (TopK数据缓冲)      │
│                     │                   │  Ping/Pong双缓冲     │
├─────────────────────────────────────────────────────────────────┤
│  reluInWorkspace    │  reluGradWorkspace│  scatterAddWorkspace │
│  (Cube1输出)        │  (Cube2输出)      │  (Cube4输出)         │
│  Ping/Pong双缓冲    │  Ping/Pong双缓冲  │  Ping/Pong双缓冲     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 核心算子实现

### 5.1 GatherTopk (Vec1)

**功能**：根据 TopK 索引从 Key 中提取数据

```cpp
// 实现逻辑
void GatherTopk(sparseIndices, key, gatherK) {
    // 1. 加载索引到 UB
    DataCopy(indiceUb, sparseIndices, topK);
    
    // 2. 遍历每个索引，Gather 对应 Key 数据
    for (i in topK) {
        int32_t idx = indiceUb[i];
        DataCopy(gatherUb, key[idx * D], D);  // D=128
    }
    
    // 3. 写出到 workspace (Ping/Pong)
    DataCopy(gatherKTensor, gatherUb, topK * D);
}
```

**优化策略**：
- 双核并行：每个核处理 topK/2 个索引
- Ping-Pong 缓冲：交替使用两组 UB 缓冲
- 批量写入：每 8 个索引合并一次写入

### 5.2 Cube MatMul (Cube1-4)

**Cube1**: Query @ KeyGather^T + ReLU
```
(G, 128) @ (2048, 128)^T → (G, 2048)
baseM = G, baseN = 128, baseK = 128
```

**Cube2**: Weights @ Dy (广播乘法)
```
(G, 1) @ (1, K) → (G, K)
广播 weights 到每个 TopK 位置
```

**Cube3**: reluGrad @ KeyGather → dQuery
```
(G, K) @ (K, 128) → (G, 128)
累加 TopK 维度的梯度到 dQuery
```

**Cube4**: reluGrad^T @ Query → scatterAdd
```
(K, G) @ (G, 128) → (K, 128)
准备 scatter 累加的中间结果
```

### 5.3 ReluGrad + Reduce (Vec2)

```cpp
void ReluGrad(reluIn, reluGrad, dy, dweights) {
    // 1. 计算 ReluGrad
    Compare(mask, reluIn, 0, CMPMODE::GT);  // reluIn > 0 ?
    Select(reluGradOut, mask, reluGrad, 0); // 选择性传递梯度
    
    // 2. 计算 dWeights
    Mul(s_mul_dy, reluIn, dyFloat);         // S * Dy
    ReduceSum(dweights, s_mul_dy);          // 累加到 dW
}
```

### 5.4 ScatterAdd (Vec3)

```cpp
void ScatterAdd(sparseIndices, scatterAdd, dk) {
    // 1. 加载索引
    DataCopy(indiceUb, sparseIndices, topK);
    
    // 2. 遍历索引，AtomicAdd 到 dKey
    for (i in topK) {
        int32_t idx = indiceUb[i];
        SetAtomicAdd<float>();
        DataCopy(dk[idx * D], scatterAdd[i * D], D);
        SetAtomicNone();
    }
}
```

**确定性模式**：
- 使用 dkCoreWorkspace 作为核间分离的缓冲
- 最后通过 DeterministicMerge 合并结果
- 避免 AtomicAdd 的非确定性

---

## 6. 参数说明

### 6.1 输入参数

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `query` | (B, S1, N1, D) 或 (T1, N1, D) | FLOAT16/BF16 | 查询向量 |
| `key` | (B, S2, N2, D) 或 (T2, N2, D) | FLOAT16/BF16 | 键向量 |
| `dy` | 与 query 相同 | FLOAT16/BF16 | 上层梯度 |
| `sparseIndices` | (B, S1, K) 或 (T1, K) | INT32 | TopK 索引 |
| `weights` | (B, S1, N1) 或 (T1, N1) | FLOAT16/BF16 | 权重向量 |
| `actualSeqLengthsQuery` | (B,) | INT32 | TND模式下 query 的前缀和长度 |
| `actualSeqLengthsKey` | (B,) | INT32 | TND模式下 key 的前缀和长度 |

### 6.2 属性参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `layout` | STRING | "BSND" | 数据排布格式 (BSND/TND) |
| `headNum` | INT64 | - | Head 数量 (N1) |
| `sparseMode` | INT32 | 0 | 稀疏模式 (0: default, 3: causal) |
| `preTokens` | INT64 | 2^63-1 | 窗口左边界 (默认值) |
| `nextTokens` | INT64 | 2^63-1 | 窗口右边界 (默认值) |
| `deterministic` | BOOL | false | 是否启用确定性计算 |

### 6.3 输出参数

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `dQuery` | 与 query 相同 | FLOAT16/BF16 | query 梯度 |
| `dKey` | 与 key 相同 | FLOAT16/BF16 | key 梯度 |
| `dWeights` | 与 weights 相同 | FLOAT16/BF16/FLOAT | weights 梯度 |

---

## 7. 约束条件

### 7.1 Shape 约束 (BSND 模式)

| 维度 | 符号 | 取值范围 |
|------|------|----------|
| Batch | B | 1 ~ 1024 |
| Query SeqLen | S1 | 1 ~ 128K |
| Key SeqLen | S2 | topK ~ 128K |
| Head Num Q | N1 | 1 ~ 64 |
| Head Num K | N2 | 1 (通常) |
| Head Dim | D | 128 (固定) |
| TopK | K | 2048 |

### 7.2 数据类型约束

- Query/Key/Dy/Weights: FLOAT16 或 BFLOAT16
- SparseIndices: INT32
- DWeights 输出可额外支持 FLOAT

### 7.3 Layout 约束

- BSND: 标准 Batch×Seq×Head×Dim 排布
- TND: Token×Head×Dim 压缩排布（需要 actualSeqLengths）

### 7.4 Sparse Mode 约束

| sparseMode | 含义 | 适用场景 |
|------------|------|----------|
| 0 | Default Mask | 全注意力 |
| 3 | RightDownCausal | 因果注意力（下三角） |

### 7.5 确定性计算约束

- 默认为非确定性实现（AtomicAdd 并行累加）
- 确定性模式通过核间分离缓冲 + 后续归约实现
- 确定性模式会增加额外同步开销

---

## 8. 性能优化要点

### 8.1 内存访问优化
- L1/L0 多级缓冲减少 GM 访问
- Ping-Pong 双缓冲消除等待
- 数据预取隐藏内存延迟

### 8.2 计算优化
- Cube 核高效矩阵乘法
- Vector 核并行向量运算
- 核间流水线最大化吞吐

### 8.3 通信优化
- CrossCore Flag 低开销同步
- Workspace 共享减少数据搬运
- 核内数据复用减少冗余计算