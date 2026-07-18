# 稀疏 FlashAttention 算法演进与芯片设计影响分析

> 本文聚焦 FlashAttention 算子的**核心计算环路**（\( QK^T \)、Softmax、\( PV \)），分析稀疏化对计算/访存负载的影响，并基于 Roofline 模型探讨对 AI 芯片设计的启示。

---

## 第一章：Dense FlashAttention 计算流程与 Roofline 基线

### 1.1 FA 算子的核心计算环路

FlashAttention 的核心计算通过三层嵌套循环实现 Tiling + Recompute，避免 \( O(N^2) \) 中间矩阵完整写入 HBM：

```
for s2_block in [0, S2 / B_s2):           // 外层：遍历 KV chunk
  load K_block [B_s2, D], V_block [B_s2, D] from HBM → SRAM
  for s1_block in [0, S1 / B_s1):         // 中层：遍历 Q chunk
    load Q_block [B_s1, D] from HBM → SRAM
    S = Q_block × K_block^T               // BMM1: [B_s1, D] × [D, B_s2] → [B_s1, B_s2]
    P = softmax(S)                         // online softmax，维护 running max/sum
    O += P × V_block                      // BMM2: [B_s1, B_s2] × [B_s2, D] → [B_s1, D]
  write O to HBM
```

**关键参数**：`B_s1`（Q tile 行数）、`B_s2`（KV tile 行数）、`D`（head_dim）。以典型配置 \( B_{s1} = B_{s2} = D = 128 \) 为例，单次 tile 执行两次 [128, 128] × [128, 128] 矩阵乘。

### 1.2 计算量与访存量

对于单头（single head），**完整**一次 FA 前向（不做 tiling 的情况下）的理论量：

**总计算量（FLOPs）**：

| 步骤 | 规模 | FLOPs |
|------|------|-------|
| \( S_1 = QK^T \) | \( S_1 \times S_2 \times D \) | \( 2 S_1 S_2 D \) |
| Softmax | \( S_1 \times S_2 \) | \( \approx 5 S_1 S_2 \) |
| \( O = PV \) | \( S_1 \times S_2 \times D \) | \( 2 S_1 S_2 D \) |
| **合计** | | \( \approx 4 S_1 S_2 D \) |

> Softmax FLOPs 远小于 GEMM，常被忽略。总计算量近似为 \( 4 S_1 S_2 D \) FLOPs。

**总访存量（Bytes）**：

在 FA Tiling 下，Q 被复用、K/V 按块遍历，实际 HBM→SRAM 总读取量为：

| 张量 | 访存 | 说明 |
|------|------|------|
| Q | \( S_1 D \times \text{dtype} \) | 每个 S1 行的 Q 仅加载一次（S2 循环内复用） |
| K | \( S_2 D \times \text{dtype} \) | K 的每个 S2 块加载一次 |
| V | \( S_2 D \times \text{dtype} \) | V 的每个 S2 块加载一次 |
| O（写出） | \( S_1 D \times \text{dtype} \) | 最终输出写回 HBM |

**总访存 ≈ \( (S_1 D + 2 S_2 D + S_1 D) \times 2 \) bytes（BF16） ≈ \( 4 S D \times 2 \) bytes（当 \( S_1 = S_2 = S \) 时）**。
更精确地：`Bytes = 2 × D × (S_1 + 2S_2 + S_1) × dtype_size`。

### 1.3 Roofline 建模

算术强度（Arithmetic Intensity, AI）：

\[
AI = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{4 S_1 S_2 D}{2 \times (2S_1 D + 2S_2 D) \times 2}
\]

当 \( S_1 = S_2 = S \) 时：

\[
AI \approx \frac{4 S^2 D}{8 S D \times 2} = \frac{S}{4}
\]

**关键结论**：Dense FA 的 AI 正比于序列长度 \( S \)，与 head_dim \( D \) **无关**。当 \( S = 8192 \) 时 \( AI \approx 2048 \) FLOPs/Byte，约 2K。在典型 AI 加速器上（HBM 带宽 ~TB/s，算力 ~数百 TFLOPs），ridge point（带宽-算力转折点）约为 `TFLOPs / TB/s = 100~300` FLOPs/Byte。

> 因此 Dense FA 在 **短序列时 Memory Bound，中等及长序列时 Compute Bound**。对于 BF16/FP16 典型算力，S≥1024 时基本进入 Compute Bound 区域。

**在 tile 级别**，每个 tile 的 AI 为：

\[
AI_{tile} = \frac{4 B_{s1} B_{s2} D}{2 \times (B_{s1} D + 2 B_{s2} D + B_{s1} D) \times 2} \approx \frac{B_{s2}}{4}
\]

tile 级 AI 与 seq_len 无关，仅与 S2 tile 大小成正比，tile 级往往是 Memory Bound（\( B_{s2}=128 \)，AI≈32）。

---

## 第二章：稀疏 FA 概述

### 2.1 稀疏 FA 的本质

在 FA 的核心循环中，稀疏 FA 通过**减少实际参与 \( QK^T \) 或 \( PV \) 计算的 KV token 数或数据维度**来降低总运算量。主流两路线：

| 路线 | 优化目标 | 在 FA 循环中的改动点 |
|------|----------|---------------------|
| **MLA 维度压缩** | 降低 K/V 的 head_dim \( D \) | K/V block 加载量从 \( B_{s2} \cdot D \) 降为 \( B_{s2} \cdot d_c \)；BMM1 规模缩为 \([B_{s1}, d_c] \times [d_c, B_{s2}]\) |
| **GQA Token 稀疏** | 降低参与计算的 KV token 数 \( S_2' \) | 每次 S2 循环仅加载 Top-K 个 token 的 K/V，跳过其余 token |

### 2.2 对 Roofline 的通用影响

**MLA 路线**：访存和计算同步下降，但访存下降更多（K/V 变得更"窄"），AI 提升。

**GQA Token 稀疏路线**：计算量随稀疏度线性下降，访存也下降但受限于稀疏索引的随机访问开销。理想 AI 可达 Dense 的数倍，但实际有效 AI 取决于稀疏索引效率和 SpMM 实现质量。

---

## 第三章：MLA 稀疏 FA —— DeepSeek V3.2

### 3.1 FA 计算流程变化

DSV3.2 的 MLA 核心创新：将 K 和 V 压缩到低秩潜在空间。在 FA 算子内部，**K 和 V 共享同一压缩表示**（\( key = value \)），head_dim 从 \( D \) 降为 \( d_c \)（如 \( D=512, d_c=128 \)）。

FA 主循环的变化（对比第一章 Dense FA）：

```
// Dense FA: 加载 K [B_s2, D], V [B_s2, D]
// MLA FA:   加载 K [B_s2, d_c]，K 即 V（共享），仅加载一次
```

| 对比项 | Dense FA | MLA FA（DSV3.2） |
|--------|----------|-------------------|
| K 加载量（per tile） | \( B_{s2} \cdot D \cdot 2 \) bytes | \( B_{s2} \cdot d_c \cdot 2 \) bytes |
| V 加载量（per tile） | \( B_{s2} \cdot D \cdot 2 \) bytes | **0**（复用 K） |
| BMM1 规模 | \([B_{s1}, D] \times [D, B_{s2}]\) | \([B_{s1}, d_c] \times [d_c, B_{s2}]\) |
| BMM2 规模 | \([B_{s1}, B_{s2}] \times [B_{s2}, D]\) | \([B_{s1}, B_{s2}] \times [B_{s2}, d_c]\) |
| 压缩比 | — | \( \rho = d_c / D \)（典型 0.25） |

> 注意：以上仅描述 FA 算子内部的计算。KV 压缩过程（\( c_{t}^{KV} \) 的低秩投影）在 FA 算子之外完成，不在本文讨论范围内。

### 3.2 Roofline 变化

以典型参数 \( D=512, d_c=128, B_{s1}=B_{s2}=128 \) 分析：

**计算量（per tile）**：

| | Dense | MLA |
|--|-------|-----|
| BMM1 | \( 2 B_{s1} B_{s2} D = 2 \times 128^2 \times 512 = 16.78M \) | \( 2 \times 128^2 \times 128 = 4.19M \) |
| BMM2 | \( 2 B_{s1} B_{s2} D = 16.78M \) | \( 2 \times 128^2 \times 128 = 4.19M \) |
| **总计算** | **33.55M FLOPs** | **8.39M FLOPs** |

**访存量（per tile）**：

| | Dense | MLA |
|--|-------|-----|
| Q 加载 | \( B_{s1} D \cdot 2 = 128 \times 512 \times 2 = 128KB \) | \( 128 \times 128 \times 2 = 32KB \) |
| K 加载 | \( B_{s2} D \cdot 2 = 128KB \) | \( B_{s2} d_c \cdot 2 = 32KB \) |
| V 加载 | \( B_{s2} D \cdot 2 = 128KB \) | 0（共享）|
| **总访存** | **384KB** | **64KB** |

**Arithmetic Intensity（per tile）**：

\[
AI_{Dense} = \frac{33.55M}{384KB} \approx \frac{33.55 \times 10^6}{384 \times 10^3} \approx 87 \ \text{FLOPs/Byte}
\]

\[
AI_{MLA} = \frac{8.39M}{64KB} \approx \frac{8.39 \times 10^6}{64 \times 10^3} \approx 131 \ \text{FLOPs/Byte}
\]

**AI 提升幅度**：\( 131 / 87 \approx 1.5\times \)。

在全序列级别，MLA 的 AI 提升更为显著：

\[
AI_{Dense} \approx \frac{S}{4}, \quad AI_{MLA} \approx \frac{2 S d_c}{S D + 2 S d_c + S d_c} \times \frac{D}{2} \approx \frac{2 d_c}{D + 3 d_c} \times \frac{S D}{2}
\]

简化后，MLA 访存约为 Dense 的 \( (D + 3d_c) / (4D) \)，计算约为 Dense 的 \( d_c / D \)。AI 变化比：

\[
\frac{AI_{MLA}}{AI_{Dense}} \approx \frac{4 d_c}{D + 3 d_c}
\]

当 \( D=512, d_c=128 \) 时，比值 ≈ \( 512 / 896 \approx 0.57 \)。注意在全序列级，因为 Q 复用的边际效应，MLA 的 AI 可能 **略低于** Dense 的 AI（因为 V 共享节省的访存被 Q 维度压缩带来的计算损失部分抵消），但总负载显著下降，**实际吞吐（token/s）大幅提升**。

**关键结论**：
1. MLA 使单次 FA 的总 GBytes 和总 TFLOPs **同步大幅下降**（约 4×）
2. 由于 KV 共享（key=value），HBM→SRAM 的 K/V 加载 `2D` 变为 `1·d_c`，访存减少超 8×
3. 对芯片而言，MLA 使得 **同等 HBM 带宽下可支撑更大 batch** 或 **更小芯片仍可达吞吐目标**

---

## 第四章：MLA 稀疏 FA —— DeepSeek V4

### 4.1 FA 计算流程变化

DSV4 在 V3.2 基础上进一步优化 FA 算子内的计算：

1. **更激进的 KV 压缩比**：\( d_c \) 进一步缩小（例如从 128 → 64），压缩比 \( \rho \) 可达 8×
2. **Q 侧压缩**：DSV4 将 Q 也投影到低秩空间（\( d_q \ll D \)），BMM1 的规模从 \([B_{s1}, d_c] \times [d_c, B_{s2}]\) 进一步缩到 \([B_{s1}, d_q] \times [d_q, B_{s2}]\)
3. **减少 online softmax 中间数据量**：S 矩阵维度 \( B_{s1} \times B_{s2} \) 不变，但 due to 维度压缩，softmax 输入/输出的精度和存储要求有所变化

FA 主循环对比：

```
// v3.2: S = Q_dc × K_dc^T      [B_s1, d_c] × [d_c, B_s2]
// v4:   S = Q_dq × K_dc^T      [B_s1, d_q] × [d_c, B_s2]   (dq ≤ dc)
```

| 对比项 | V3.2 MLA | V4 MLA |
|--------|----------|--------|
| \( d_c \) | ~128 | ~64 |
| Q 维度 | \( d_c \) | \( d_q \)（≤ \( d_c \)） |
| BMM1 规模 | \([B_{s1}, d_c] \times [d_c, B_{s2}]\) | \([B_{s1}, d_q] \times [d_c, B_{s2}]\) |
| Q 加载量（per tile） | \( B_{s1} \cdot d_c \cdot 2 \) | \( B_{s1} \cdot d_q \cdot 2 \) |

### 4.2 Roofline 变化

以 \( d_c=64, d_q=64, B_{s1}=B_{s2}=128 \) 为例：

| | V3.2 (\( d_c=128 \)) | V4 (\( d_c=d_q=64 \)) |
|--|---------------------|----------------------|
| BMM1 FLOPs | \( 2 \times 128^2 \times 128 = 4.19M \) | \( 2 \times 128^2 \times 64 = 2.10M \) |
| BMM2 FLOPs | \( 2 \times 128^2 \times 128 = 4.19M \) | \( 2 \times 128^2 \times 64 = 2.10M \) |
| Q 加载 | 32KB | 16KB |
| K 加载 | 32KB | 16KB |
| V 加载 | 0 | 0 |
| 总访存 | 64KB | 32KB |
| AI | 131 | 131 |

在 tile 级 AI **保持不变**（因为 FLOPs 和 Bytes 等比缩放），但全序列级 AI 变化：

\[
\frac{AI_{V4}}{AI_{V3.2}} \approx \frac{d_c^{(V4)}}{d_c^{(V3.2)}} \cdot \frac{D + 3d_c^{(V3.2)}}{D + 3d_c^{(V4)}}
\]

当 \( D=512, d_c^{(V3.2)}=128, d_c^{(V4)}=64, d_q=64 \) 时：
\[
\frac{AI_{V4}}{AI_{V3.2}} \approx \frac{64}{128} \cdot \frac{512 + 384}{512 + 192} = 0.5 \times \frac{896}{704} \approx 0.64
\]

**关键解读**：
- V4 相比 V3.2 的 AI 并非单调增加，而是随压缩比变化呈现**非单调特征**
- 由于 Q 也被压缩，BMM1 的计算量二次下降（\( d_q \times d_c \) 缩小），AI 可能反而下降
- **但绝对负载大幅降低**：全序列 FA 总 FLOPs 降至 V3.2 的 \( d_c^{(V4)} / d_c^{(V3.2)} = 0.5 \) 倍，总访存也同比降低
- **Roofline 上的实际意义**：工作点向**左下**移动（更低算力、更低带宽），使得更多芯片平台可落在平台能力范围内

---

## 第五章：GQA 稀疏 FA —— 字节与腾讯

### 5.1 GQA 基础

GQA（Grouped Query Attention）不改变 FA 核心计算的算法逻辑，但改变了 **tile 循环的组织方式**。设 `G = N1 / N2` 为 Query Head 数与 KV Head 数之比（典型 32 或 64），FA 循环中：

- 每个 KV head 的 K/V block 被 `G` 个 Q head 共享
- **外层循环**新增 G 维度的迭代，Q 的 `S1` token 被扩展到 `S1 × G` 个 `gS1` token
- 对于大 G（如 64），KV → HBM 的访存可被 `G` 个 Q head **分摊**

Dense GQA FA 的 Roofline 与第一章基本一致（G 维度的引入不改变 per-tile AI，只是增加了 tile 数），本文重点讨论在此基础上的 **Token 稀疏化**。

### 5.2 字节跳动方案：Token-Sparse FA（在线 Top-K 选择）

**核心思想**：在 FA 的 S2 循环中，不是计算所有 KV block，而是通过**在线估计**选择 Top-K 个 KV token。

FA 循环流程变化：

```
for s2_block in [0, S2 / B_s2):
  load K_block [B_s2, D_m], V_block [B_s2, D_m]    // 只加载压缩的 K/V 元信息
  S_approx = Q_block × K_block^T                     // 近似 QK^T（低精度/小维度）

TopK_indices = select_topk(S_approx, k)             // 跨所有 S2 block 选 Top-K

for idx in TopK_indices:                             // 仅对选中的 token 精确计算
  load K[idx], V[idx]
  S_exact = Q × K[idx]^T
  P = softmax(S_exact)
  O += P × V[idx]
```

**计算量变化**：
- 近似 Score 阶段：\( 2 S_1 S_2 D_m \) FLOPs，其中 \( D_m \ll D \)（如 \( D_m=64, D=512 \)）
- 精确 Score + PV 阶段：\( 4 S_1 K D \) FLOPs，其中 \( K \ll S_2 \)（如 \( K=2048, S_2=65536 \)）

**访存量变化**：
- 近似阶段：K/V 元信息只加载 \( D_m \) 维，访存大幅降低
- 精确阶段：仅加载 Top-K 个 KV token，理论访存降至 \( K/S_2 \) 倍
- **额外开销**：Top-K 选择引入不规则索引（Gather/Scatter），non-contiguous 访存

**Top-K 选择代价**：需要维护跨 S2 block 的全局 score min-heap，或使用分块 Top-K + 合并策略。这是计算开销——约 \( O(S_2 \log K) \) 次比较操作，通常由 Vector 单元执行。

**Roofline 变化**：

\[
AI_{TokenSparse} = \frac{4 S_1 K D + 2 S_1 S_2 D_m}{2 \times (S_1 D + S_2 D_m + 2K D) \times 2}
\]

当 \( S_1 = S_2 = S \)、\( K \ll S \)、\( D_m \ll D \) 时：

\[
AI \approx \frac{2 K D}{D_m S / 2 + K D} \times S
\]

**关键特征**：
- 理论 AI 随稀疏度 \( K/S \) 可达到 Dense FA 的 **数倍至数十倍**
- 实际受限于 Top-K 选择开销 + 不规则访存，有效 AI 大打折扣
- 工作点呈现 **"高 AI、低算力利用"** 特征——可以落在 ridge point 右侧很远，但距离硬件峰值 FLOPs 仍有距离

### 5.3 腾讯方案：Chunk 级稀疏 + 局部窗口混合

**核心思想**：以 **chunk 为粒度** 做稀疏判决（而非 token 级），保留局部窗口 + 稀疏全局采样。

FA 循环流程：

```
window_size = 1024  // 局部窗口
chunk_size = 128    // 稀疏判决粒度

for s2_block in [0, S2 / B_s2):
  if is_in_local_window(s2_block):                  // 局部窗口内 → 全算
    compute_dense_fa(Q_block, K_block, V_block)
  elif is_selected_global_chunk(s2_block):           // 全局稀疏采样 → 算
    compute_sparse_fa(Q_block, K_block, V_block)
  else:
    skip                                            // 跳过
```

**计算量与访存量**：
- 局部窗口内：全量计算，与 Dense FA 一致
- 全局稀疏 chunk：仅少量 chunk 参与计算（如每 128 chunk 选 1）
- 总体 sparsity 由两部分混合决定：窗口内 dense + 窗口外 sparse

**Total FLOPs** ≈ \( 4 S_1 D \times (W + C) \)，其中 \( W \) 为窗口大小，\( C \) 为选中的全局 chunk 总和。

**与字节方案的关键差异**：
| | 字节 Token-Sparse | 腾讯 Chunk-Sparse |
|--|-------------------|-------------------|
| 稀疏粒度 | Token 级（逐 token） | Chunk 级（128 tokens/块） |
| 选择方式 | 在线 Top-K（K 固定） | 局部窗口 + 全局采样 |
| 访存模式 | 不规则 Gather/Scatter | 规则 chunk 级跳过 |
| Top-K 开销 | 需要跨 tile 排序 | 仅需 chunk 级判断，开销低 |
| 与硬件适配 | 需要高效 SpMM / 不规则 GEMM | 与 Dense GEMM 适配性更好（chunk 内仍是稠密块） |

**Chunk 级稀疏对 Roofline 的影响**：

由于 chunk 内仍为 dense 计算，tile 级 AI 不变（≈ 87 FLOPs/Byte）。全序列级 AI 提升来自减少了有效 \( S_2 \)：

\[
AI_{chunk} \approx AI_{Dense} \times \frac{S_2}{W + C}
\]

当 \( S_2 = 65536, W = 1024, C = 512 \) 时，AI 提升约 \( 65536 / 1536 \approx 42.7\times \)。但因为跳过了大量 chunk，总负载大幅降低，**更容易达到 Compute Bound 上限**。

---

## 第六章：Roofline 全景对比

### 6.1 统一 Roofline 图表

以典型 AI 加速器（HBM BW = 2TB/s，BF16 Peak = 312 TFLOPs，ridge point ≈ 156 FLOPs/Byte）为参考：

```
Performance (TFLOPs)
^
|                                    Ridge Point
|                                    x  (156 FLOPs/B)
| 312 ┤·········································· Peak Compute
|     │                             ↗ TokenSparse(ideal)
|     │                        ↗ MLA-V3.2
|     │                   ↗ Dense
|     │              ↗ MLA-V4
|     │          ↗ ChunkSparse
|     │         ↗ TokenSparse(actual)
|     │
|   0 ┼──────┬──────┬──────┬──────┬──────┬──────┬──────→ AI (FLOPs/Byte)
|       0    50    100   150   200   250   300   350

  Dense FA (S=8K):      AI ≈ 2048 → Compute Bound  (右侧)
  MLA-V3.2 (S=8K):      AI ≈ 1800 → Compute Bound
  MLA-V4 (S=8K):        AI ≈ 1150 → Compute Bound (更左，但总负载更低)
  TokenSparse (ideal):  AI ≈ 4000+ → Deep Compute Bound
  TokenSparse (actual): AI ≈ 3000+, Util < 30% peak
  ChunkSparse:          AI ≈ 50000+ (有效S2极短) → 极右, 但易达峰值
```

### 6.2 关键参数敏感性分析

| 参数 | Dense AI | MLA AI | TokenSparse AI | ChunkSparse AI |
|------|----------|--------|----------------|----------------|
| ↑ seq_len S | ↑ 线性 | ↑ 线性 | ↑ 近线性 | ↑ 线性 |
| ↑ head_dim D | **不变** | ↓ 略微 | ↓ 分母 | **不变** |
| ↑ 稀疏度 | N/A | ↑ ρ 增大 | ↑ K ↓ | ↑ W↓/C↓ |
| ↑ batch B | **不变** | **不变** | ↑ (分摊索引) | **不变** |

**核心发现**：
1. **Dense FA 的 AI 完全由 S 决定**，与 D 无关——这是 FlashAttention Tiling 的核心特性
2. **MLA 的 AI 同时受 S 和压缩比 ρ 影响**——更小的 ρ 可能导致 AI 下降（因为访存节省被 Q 侧压缩的 BMM1 计算下降抵消）
3. **TokenSparse 的有效 AI 依赖于 SpMM 硬件支持**——没有原生稀疏 GEMM 时，大量计算时间消耗在 scatter/gather 和 index 操作上
4. **ChunkSparse 是硬件最友好的方案**——chunk 内仍是 dense GEMM，仅 chunk 级跳过

### 6.3 不同算法在 Roofline 上的"工作点"迁移总结

```
                    访存负载    计算负载    AI       Bound 类型
                    ────────   ────────   ──────   ─────────
Dense FA            基准        基准        ~S/4    S<512: Mem
                                                     S>512: Compute
MLA-V3.2            ↓↓ 2~4×    ↓↓ 4×      ↑ 1.5×   更偏 Compute
MLA-V4              ↓↓ 4~8×    ↓↓ 8×      →/↓     工作点左移,负载大降
TokenSparse(理想)   ↓↓ K/S     ↓↓ K/S     ↑↑ 10×+  Deep Compute
TokenSparse(实际)   ↓ K/S      ↓ K/S+TopK ↑ 5×     Compute(BW利用差)
ChunkSparse         ↓ (W+C)/S  ↓ (W+C)/S  ↑ 10~40× 极易Peak
```

---

## 第七章：对 AI 芯片设计的需求分析

### 7.1 HBM 带宽需求的分化

| 算法 | 带宽瓶颈程度 | 芯片需求 |
|------|-------------|---------|
| Dense FA | **核心瓶颈**，长序列时需 GHz 级 HBM BW | 最大内存带宽（HBM3e 4+ TB/s） |
| MLA-V3.2 | **显著缓解**，压缩比 4× → 带宽需求 1/4 | 中等带宽 + 更高算力利用率 |
| MLA-V4 | **大幅缓解**，压缩比 8× → 带宽需求 1/8 | 可为边缘/低功耗芯片所接受 |
| TokenSparse | 理论缓解大，实际受不规则访存拖累 | 需要高带宽 + 低延迟随机访问 |
| ChunkSparse | **几乎不由带宽限制** | 带宽需求最低 |

**芯片启示**：对于纯 Dense FA 场景，HBM 带宽是**绝对硬需求**；而稀疏 FA 的普及意味着芯片可在带宽-算力之间做更大范围的 **trade-off**。

### 7.2 低精度 GEMM 算力需求

MLA 系列将 AI 从 Dense 的 ~S/4 提升或改变后，对计算单元的压力发生结构性变化：

- **MLA-V3.2**：BMM1/BMM2 规模从 128×128 缩小到 128×128（D→d_c），Tensor Core **利用率下降**（K 维度短 → 更少的并行乘累加）
- **MLA-V4**：K 维度进一步缩小到 64，MM 形状变为 `[128, 64] × [64, 128]`，**短 K 维度对 systolic array 的利用率挑战更大**
- 芯片需要**对小 K 维度 GEMM 做专门优化**，例如增加 systolic array 对窄 K 的 lane 级并行能力

对于 FP8/INT8 量化：
- MLA 的低秩空间天然适合量化——d_c 小，量化表项少，解量化开销低
- TokenSparse 需要支持 **mixed-precision**：近似阶段 INT8/FP8，精确阶段 BF16

### 7.3 稀疏计算原语

**TokenSparse 的需求**：

| 操作 | 需求 | 芯片支持要求 |
|------|------|-------------|
| Top-K 选择 | \( O(S_2 \log K) \) 跨 tile 比较 | Vector 单元支持高效的 reduce + heap 操作 |
| Gather K/V | 按稀疏索引收集 token | 高效的 `gather` 向量指令，高吞吐的随机 L2/L1 读取 |
| 稀疏 GEMM | `[M, K] × [K_sparse, N]` | **2:4 结构化稀疏 GEMM**（如 NVIDIA Sparse Tensor Core）支持 2× 加速 |
| Scatter O gradient | 梯度反向写入 | 高效 `scatter_add` 原子操作 |

**ChunkSparse 的需求**：
- 相对简单：chunk 级选择性跳过，利用 **conditional execution** 即可
- 需要 **快速 chunk 级掩码判断**——条件分支对 GPU/NPU warp/block 的统一执行模型是挑战

**MLA 的稀疏需求**：
- MLA 的"稀疏"本质是**维度压缩**而非 token 选择
- 芯片需要优化 **窄矩阵乘（small-K GEMM）** 的效率——包括更小的 tile 尺寸、更细粒度的 loop tiling

### 7.4 Local Memory（SRAM）设计

| 算法 | SRAM 需求特征 | 设计启示 |
|------|-------------|---------|
| Dense FA | Q/K/V 各 32KB tile（总 ~288KB） | 512KB L1 可满足 |
| MLA-V3.2 | Q tile → 16KB，K tile → 16KB（总 ~96KB） | SRAM 需求降低，但 **tile 数量增加** → 更多小 tile 的加载开销 |
| MLA-V4 | 进一步缩小 | 小 tile 场景对 DMA 引擎的小粒度搬运效率要求高 |
| TokenSparse | 需要额外 buffer 存 TopK indices + 中间 score | SRAM 需要容纳 **索引表** 和 **分块 score buffer** |
| ChunkSparse | 与 Dense 一致 | 不变 |

### 7.5 算子融合能力

稀疏 FA 在 tile 内融合了更多步骤：

```
Dense FA tile:     QK^T → Softmax → PV
MLA FA tile:       QK^T(低维) → Softmax → PV(低维)
TokenSparse tile:  近似 QK^T → TopK → 精确 QK^T → Softmax → PV
ChunkSparse tile:  [判断] → QK^T → Softmax → PV (跳过不判断)
```

芯片需要：
1. **灵活的指令调度**：在 GEMM（Cube/Tensor Core）和 Vector（softmax）之间高效切换
2. **跨 tile 的数据共享**：TopK 结果的跨 tile 传递（需要跨 core 或跨 block 的低延迟通信）
3. **条件执行**：ChunkSparse 的 chunk 级跳过，在 SIMT/SIMD 架构上需要带掩码的执行或 warp-level divergent 处理

---

## 第八章：总结与展望

### 8.1 稀疏 FA 演进的两条主线

```
              ┌─────────────────────────────────────────────┐
              │          稀疏 FlashAttention 演进             │
              └─────────────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
    维度压缩 (MLA)                     Token 选择 (GQA Sparse)
           │                               │
    ┌──────┴──────┐              ┌─────────┼─────────┐
    ▼             ▼              ▼                   ▼
  V3.2          V4          字节TokenSparse    腾讯ChunkSparse
  KV低秩        Q+KV深度      在线TopK选择      窗口+全局采样
  d_c~128       d_c~64        K个token          块粒度判决
```

### 8.2 对芯片的核心影响总结

| 需求维度 | Dense FA 时代 | 稀疏 FA 时代 |
|---------|-------------|------------|
| **HBM 带宽** | 绝对瓶颈，越大越好 | 需求降低但有分化，TokenSparse 仍需带宽 |
| **算力（GEMM）** | 大 GEMM（K=128/D=512）利用率高 | 小 GEMM（K=64/128）利用率面临挑战 |
| **稀疏加速器** | 不需要 | TokenSparse 需要 2:4 稀疏 GEMM，否则效率极低 |
| **SRAM 设计** | 32KB× tile buffer | 需额外缓冲索引、分块 score |
| **指令灵活性** | 标准 GEMM+softmax | 需融合 TopK、条件执行、跨 core 通信 |

### 8.3 展望：混合范式与下一代芯片设计建议

1. **MLA + Token Sparse 混合**：用 MLA 的维度压缩降低每个 token 的 GEMM 开销，同时用 Token 级稀疏进一步减少处理的 KV 数量。这对芯片提出"窄矩阵 + 稀疏"的双重需求。

2. **结构化稀疏（2:4 N:M）与 MLA 结合**：将 MLA 的压缩维度与硬件 2:4 稀疏结合，K/V 加载时每 4 个元素仅保留 2 个非零，在 GEMM 层面获得 2× 加速——要求 Tensor Core 原生支持 N:M 稀疏模式。

3. **异步稀疏选择引擎**：将 TopK 选择从 Vector 核心中分离为专用轻量单元（Sparse Indexer），在 Cubic GEMM 执行时并行进行下一 tile 的 KV 选择，实现 **选择-计算 的流水线重叠**。

4. **自适应 Roofline 调度**：芯片运行时根据 `d_c`、`S`、稀疏度在 Dense/MLA/Sparse 模式间切换，动态选择最优 tiling 策略——对驱动和编译器做全链路协同优化。

---

*本文档仅介绍 FlashAttention 算子核心计算环路内的稀疏化演进，不涉及 FA 算子之外的 KV 压缩、RoPE 应用、吸收解码等前/后处理算子。*
