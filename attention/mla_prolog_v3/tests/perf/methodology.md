# MLA Prolog V3 性能建模与优化方法论

本文档面向 MLA Prolog 内核开发者，介绍如何分析和优化算子在 Ascend 950 上的理论性能。每一节均包含基于实际参数的计算示例（BF16 Decode BS=1, head_num=128）。

> **快速开始**: 运行 `python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode full` 查看完整分析。

---

## 目录

1. [关键结论](#1-关键结论)
2. [硬件存储层次](#2-硬件存储层次)
3. [Roofline 模型](#3-roofline-模型)
4. [Matmul 性能建模](#4-matmul-性能建模)
5. [Vector 操作建模](#5-vector-操作建模)
6. [缓冲策略](#6-缓冲策略)
7. [Pipeline 流水分析](#7-pipeline-流水分析)
8. [优化决策指南](#8-优化决策指南)

---

## 1. 关键结论

在 Ascend 950 上，MLA Prolog V3 的 Decode 场景（BS=1~32）**全面受 HBM 带宽瓶颈限制**：

| 瓶颈 | 原因 | 主要优化手段 |
|------|------|------------|
| MM3 占 54% 时间 | B 矩阵 72MB，必须全量从 HBM 加载 | INT8/FP8 量化 (2×) |
| MM1/MM2 核利用率低 | N 轴小，split-N 只用 9~24 核 | Split-KN 2D 切分 (最高 4.6×) |
| AIV 几乎空闲 | Vector 操作远小于 Cube | 无需优化（非瓶颈） |

**优化后 Pipeline 时间**: 87.63 us → 53.31 us (Split-KN) → 可进一步通过量化降至 ~27 us。

---

## 2. 硬件存储层次

### 2.1 数据流架构

Ascend AI Core 包含两条计算流水线，共享 HBM 访问：

```mermaid
graph TB
    HBM["<b>HBM (片外存储)</b><br/>芯片总带宽: 1.6 TB/s"]
    L2["<b>L2 Cache (片上共享)</b><br/>5.2 TB/s"]

    HBM -->|"33.3 B/cycle<br/>(MTE1/MTE2)"| L2

    subgraph AIC ["AIC Cube 流水线"]
        direction TB
        L1["<b>L1 Buffer</b><br/>512 KB"]
        L0A["<b>L0A</b><br/>64 KB"]
        L0B["<b>L0B</b><br/>256 KB"]
        MMAD["<b>MMAD</b><br/>BF16: 4096 MACs/cycle<br/>FP8: 8192 MACs/cycle"]
        L0C["<b>L0C</b><br/>256 KB"]
        L1 -->|"256 B/cycle<br/>(MTE1)"| L0A
        L1 -->|"256 B/cycle<br/>(MTE1)"| L0B
        L0A --> MMAD
        L0B --> MMAD
        MMAD --> L0C
        L0C -->|"256 B/cycle<br/>(FixPipe)"| FPOut["HBM/UB 写回"]
    end

    subgraph AIV ["AIV Vector 流水线"]
        direction TB
        UB["<b>UB (Unified Buffer)</b><br/>256 KB"]
        VEC["<b>向量计算引擎</b><br/>256 B/cycle, IPC=1.3"]
        UB -->|"PIPE_V"| VEC
        VEC --> UB
        UB -->|"MTE3"| VOut["HBM 写回"]
    end

    L2 -->|"MTE1/MTE2"| L1
    L2 -->|"MTE2"| UB

    style HBM fill:#4a90d9,color:#fff,stroke:#2c5f9e
    style L2 fill:#67b7dc,color:#fff,stroke:#4a90a4
    style L1 fill:#f5a623,color:#fff,stroke:#c48418
    style L0A fill:#ff7675,color:#fff,stroke:#d63031
    style L0B fill:#ff7675,color:#fff,stroke:#d63031
    style L0C fill:#ff7675,color:#fff,stroke:#d63031
    style MMAD fill:#e17055,color:#fff,stroke:#b34a35
    style UB fill:#a29bfe,color:#fff,stroke:#6c5ce7
    style VEC fill:#6c5ce7,color:#fff,stroke:#5241b5
```

> [在线编辑此图](https://l.mermaid.ai/tERtCD)

### 2.2 数据搬运引擎

| 引擎 | 路径 | 用途 | 带宽 (950) |
|------|------|------|-----------|
| **MTE1/MTE2** | HBM/L2 → L1 | Cube: 加载 A/B 矩阵到 L1 | 33.3 B/cycle |
| **MTE1** | L1 → L0A | Cube: 搬运 A tile 到计算缓冲 | 256 B/cycle |
| **MTE1** | L1 → L0B | Cube: 搬运 B tile 到计算缓冲 (转置) | 256 B/cycle |
| **MMAD** | L0A × L0B → L0C | Cube: 矩阵乘累加 | BF16: 4096 MACs/cycle, FP8: 8192 MACs/cycle |
| **FixPipe** | L0C → HBM/UB | Cube: 输出写回 | 256 B/cycle |
| **MTE2** | HBM/L2 → UB | Vector: 加载输入数据 | 33.3 B/cycle |
| **PIPE_V** | UB → UB | Vector: 计算 (RmsNorm, RoPE 等) | — |
| **MTE3** | UB → HBM | Vector: 输出写回 | 33.3 B/cycle |

### 2.3 带宽争抢

多核并发时，单核实际带宽受芯片总带宽限制：

```
单核实际带宽 = min(单核理论带宽, 芯片总带宽 / 活跃核数)
```

**示例 (32 核)**:
- HBM: min(33.3 × 1.65GHz, 1.6TB/s ÷ 32) = min(54.9, 50.0) = **50.0 GB/s**
- L2: min(108 × 1.65GHz, 5.2TB/s ÷ 32) = min(178.2, 162.5) = **162.5 GB/s**
- L1→L0A: 256 × 1.65GHz = **422.4 GB/s** (核内独享，无争抢)

> **工具**: `python perf_analyzer.py --chip 950 --mode bound` 查看各阶段带宽瓶颈

### 2.4 L2 Cache 复用

- **Split-N 复用**: 所有核共享同一 A 矩阵 → 首核从 HBM 加载，其余核命中 L2
- **跨 Matmul 复用**: MM2 复用 MM1 已加载的 tokenX（A 矩阵）→ L2 热数据

---

## 3. Roofline 模型

### 3.1 算术强度

```
AI = 2 × M × N × K / (M × K × sizeof(A) + K × N × sizeof(B) + M × N × sizeof(C))
```

单位: FLOPs/Byte。AI 越高，计算与访存越平衡。

### 3.2 Ridge Point (拐点)

| 数据类型 | 单核峰值算力 | 单核 HBM 带宽 | Ridge Point |
|---------|------------|-------------|-------------|
| BF16 | 4096 × 2 × 1.65e9 = 13,517 GFLOPS | 50 GB/s | **270.3 FLOPs/B** |
| FP8 | 8192 × 2 × 1.65e9 = 27,034 GFLOPS | 50 GB/s | **540.7 FLOPs/B** |

- AI < Ridge → **HBM 带宽瓶颈** (典型 Decode 场景)
- AI > Ridge → **Cube 计算瓶颈** (大批量 Prefill 场景)

### 3.3 示例: MM3 (QcQr), BF16, BS=1

`Cq[1, 1536] × W_UqQr[1536, 24576]`, 32 核, N_per_core = 768

```
AI = 2 × 1 × 768 × 1536 / (1×1536×2 + 1536×768×2 + 1×768×2) = 1.0 FLOPs/B

Ridge = 270.3 → AI (1.0) << Ridge → HBM 瓶颈
效率 = 1.0 / 270.3 = 0.4%
```

> **工具**: `python perf_analyzer.py --chip 950 --mode roofline` 查看各 Matmul 的 AI 和效率

---

## 4. Matmul 性能建模

### 4.1 内层循环结构与时间模型

```mermaid
graph LR
    subgraph OUTER ["外层循环: kL1 次迭代 (HBM/L2 → L1)"]
        direction TB
        MTE1["<b>Stage 1: MTE1</b><br/>HBM/L2 → L1<br/>加载 A[baseM, kL1Step]<br/>加载 B[kL1Step, N_per_core]<br/>time = max(A_load, B_load)"]
        INNER["<b>Stage 2: Inner Loop</b><br/>stepK 次迭代<br/>L1 → L0A + L0B + MMAD<br/>time = stepK × max(l0a, l0b, mmad)"]
        FIXPIPE["<b>Stage 3: FixPipe</b><br/>L0C → HBM/UB 写回<br/>time = M × N × sizeof(C) / bw"]

        MTE1 --> INNER --> FIXPIPE
    end

    subgraph OVERLAP ["双缓冲流水重叠"]
        direction TB
        FORMULA["<b>per_kl1 = max(mte1, inner, fixpipe)</b><br/><br/>总时间 = kL1Loops × per_kl1<br/><br/>瓶颈判定:<br/>mte1 最大 → HBM bound<br/>inner 最大 → L0B/MMAD bound<br/>fixpipe 最大 → 写回 bound"]
    end

    FIXPIPE --> FORMULA

    style MTE1 fill:#4a90d9,color:#fff
    style INNER fill:#e17055,color:#fff
    style FIXPIPE fill:#00b894,color:#fff
    style FORMULA fill:#2d3436,color:#fff
```

> [在线编辑此图](https://l.mermaid.ai/PiNdeZ)

三个 Stage 在 kL1 迭代之间通过双缓冲实现流水重叠，总时间由瓶颈 Stage 决定。

### 4.3 模式 A: Full-Load (MM4)

K 足够小，A 矩阵一次性加载到 L1，只迭代 N 分片。

**示例: MM4** — `Qc[1, 128] × Uk[128, 512]`, 32 核, N_per_core=16

```
A = 1 × 128 × 2B = 256B → 远小于 L1 (512KB)，单缓冲全载
加载 A: 256 / 50e9 = 0.005 us (一次性)

Per N split (nSplitSize=128):
  加载 B[128, 16] = 4096B: 4096 / 50e9 = 0.082 us  ← 瓶颈
  L0A/L0B/MMAD/FixPipe 均 < 0.01 us

总时间 ≈ 0.66 us, HBM 瓶颈
```

### 4.4 模式 B: Split-N (MM3)

N 轴按核切分，每核加载完整 K。

**示例: MM3** — `Cq[1, 1536] × W[1536, 24576]`, BF16, 32 核

```
N_per_core = 768, Block: baseK=64, stepK=2
kL1StepSize = 128, kL1Loops = ceil(1536/128) = 12

每次 kL1:
  MTE1: A=256B (0.005us), B=196,608B (3.932us) → max = 3.932 us ← B 加载主导
  内层: 2 × max(l0a, l0b, mmad) = 2 × 0.233 = 0.466 us
  FixPipe: 0.004 us

  per_kl1 = max(3.932, 0.466, 0.004) = 3.932 us

总时间 = 12 × 3.932 = 47.19 us
→ HBM 瓶颈 (B 矩阵每核 2304KB >> L1 512KB)
```

**关键瓶颈**: B 矩阵总量 72MB 固定，无论如何切分核都必须全量加载，Split-KN 无法加速 MM3。

> **工具**: `python perf_analyzer.py --chip 950 --mode block-search` 搜索最优 baseM/baseN/baseK

### 4.5 模式 C: Split-KN (MM1/MM2)

同时切 N 和 K 轴，组成 `n_groups × k_groups` 核网格。K 切分的核产生 float32 部分积，由 Vector 侧累加。

**示例: MM1** — `X[1, 7168] × W[7168, 1536]`, 配置 N8×K4 (32 核)

```
K_per_core = 7168 / 4 = 1792
N_per_core = 1536 / 8 = 192
输出 dtype = float32 (4B, 部分积)

Cube (每核):
  kL1Loops = 4, per_kl1 ≈ 0.98 us
  Cube 总时间 = 3.93 us (vs split-N 的 17.89 us, 4.6× 加速)

Vector 累加 (8 组并行):
  每组加载 4 个部分积 × 1 × 192 × 4B = 3,072B
  累加 = 0.002 us (在 Pipeline 中被下一个 AIC matmul 完全遮盖)

MM1 总时间 = 3.93 us
```

> **工具**: `python perf_analyzer.py --chip 950 --mode split-kn` 搜索最优 2D 切分并评估完整 Pipeline

### 4.6 模式 D: Split-M (Prefill)

M 轴按核切分，每核处理 `M_per_core = ceil(T / m_groups)` 个 token，N 轴不切分。适用于 Prefill 场景（大 T）。

**核心差异**:
- 每核处理完整 N → 无需核间同步
- B（权重）矩阵所有核共享 → L2 复用
- N 需在核内分块迭代（3 级循环）

**循环结构**: `m_loops × kL1_loops × n_blocks`

```
per M-block (m_loops = ceil(M_per_core / baseM)):
  per kL1-slab (kL1_loops = ceil(K / kL1StepSize)):
    加载 A[baseM, kL1StepSize] from HBM (每核独立行)
    per N-block (n_blocks = ceil(N / baseN)):
      加载 B[kL1StepSize, baseN] from L2 (共享权重，L2 复用)
      stepK × max(L0A, L0B, MMAD)   ← 内层流水
      FixPipe: 写回 C[baseM, baseN]
```

**时间模型**:

```
per_nb = max(load_B_l2, stepK × max(l0a, l0b, mmad), fixpipe)
per_kl1 = max(load_A_hbm, n_blocks × per_nb)
总时间 = m_loops × kL1_loops × per_kl1
```

**示例: MM3 (N=128 heads)** — T=512, BF16, 16 核, M_per_core=32

```
baseM=32, baseN=256, baseK=64, stepK=2
m_loops=1, kL1_loops=12, n_blocks=96

per N-block:
  load_B(L2) = 128×256×2 / 155e9 = 0.0011 us
  l0b = 64×256×2 / 422.4e9 = 0.078 us
  mmad = 32×256×64 / 6758.4e9 = 0.077 us
  total_inner = 2 × 0.078 = 0.156 us
  per_nb = max(0.0011, 0.156, 0.039) = 0.156 us  ← L0B/MMAD 边界

per_kl1 = max(0.164, 96×0.156) = 14.98 us
总时间 = 1 × 12 × 14.98 = 179.7 us → L0B/MMAD 边界
```

**与 Split-N 对比**: Split-N 对 512 tokens 需 4 步 × 88.29 us = 353 us，Split-M 需 783 us（L2 加载 B 全量开销大）。

**关键洞察**: 要达到 Cube bound，需要 `baseM >= dtype_b × mmad_tput / b_bw ≈ 91`（BF16 MM3）。但当 m_groups 较大时 M_per_core 太小，无法使用大 baseM。Split-M 仅在 **N 较小**（如 N=8）时优于 Split-N。

> **工具**: `python perf_analyzer.py --chip 950 --mode split-m` 搜索最优 M 轴切分并与 Split-N 对比

---

## 5. Vector 操作建模

Vector 操作在 AIV 流水线执行: **MTE2** (HBM→UB) → **PIPE_V** (UB 内计算) → **MTE3** (UB→HBM)。

```
总时间 = max(mte2_time, vec_time, mte3_time)    ← 三阶段流水重叠
```

### 5.1 各操作建模

| 操作 | 输入 | 计算 | 输出 | 示例时间 (BS=1) |
|------|------|------|------|----------------|
| **RmsNormCq** | T×Hcq×4B (INT32) | ×3 ops/elem | T×Hcq×2B | 0.015 us |
| **RmsNormCkvKr** | T×(Hckv+Dr)×4B | ×3 | T×Hckv×2B | 0.005 us |
| **RopeKr** | T×Dr×2B + sin/cos | ×4 | T×Dr×2B | 0.001 us |
| **ScatterCkvKr** | T×(Hckv+Dr)×2B | 地址计算 | 散列写入 (×0.4效率) | 0.002 us |
| **DequantQc** | T×N×D×4B (INT32) | ×2 (乘+加) | T×N×D×2B | ~0.1 us |
| **RopeQr** | T×N×Dr×2B + sin/cos | ×4 | T×N×Dr×2B | 0.105 us |
| **DynamicQuantQn** | T×N×Hckv×2B | ×3 (规约+缩放+量化) | T×N×Hckv×1B | ~0.2 us |

### 5.2 示例: RopeQr (Decode BS=1 最大 Vector 操作)

```
加载: 1 × 128 × 64 × 2B + 1 × 64 × 2B = 16,512 B
计算: 1 × 128 × 64 × 4 = 32,768 ops
写回: 1 × 128 × 64 × 2B = 16,384 B

vec = 32,768 / (20e12 / 64) = 0.105 us  ← 计算主导
总时间 = 0.105 us, VECTOR 瓶颈
```

> 所有 Vector 操作在 Decode BS=1 下总计仅 ~0.63 us，远小于 AIC 的 83.63 us，非性能瓶颈。

---

## 6. 缓冲策略

### 6.1 默认: 双缓冲 (2×)

双缓冲通过乒乓切换实现搬运与计算重叠，每个缓冲区需要 2 倍 tile 大小：

```
L0A: baseM × baseK × sizeof(A) × 2 ≤ 64 KB
L0B: baseK × baseN × sizeof(B) × 2 ≤ 256 KB
L0C: baseM × baseN × sizeof(C) × 2 ≤ 256 KB
L1:  (A_per_step + B_per_step) × 2 ≤ 512 KB
```

### 6.2 全载优化 (单缓冲)

若**完整矩阵**可装入缓冲区，无需乒乓 → 单缓冲 (1×)，仅在性能有收益时启用。

### 6.3 stepK 推导

```python
per_step = baseM × baseK × sizeof(A) + baseK × N_per_core × sizeof(B)

# 尝试单缓冲全载:
single_stepK = L1_size // per_step
if single_stepK >= ceil(K / baseK):
    stepK = ceil(K / baseK)    # 全载成功, kL1=1

# 否则双缓冲:
stepK = L1_size // 2 // per_step
```

**示例: MM3** — baseK=64, N_per_core=768

```
per_step = 32×64×2 + 64×768×2 = 100 KB
单缓冲: 512 / 100 = 5 < 24 (需 24 步) → 不够
双缓冲: 256 / 100 = 2 → stepK=2, kL1Loops=12
```

**示例: MM4** — M=1, K=128, N_per_core=16

```
全量 A+B = 256 + 4096 = 4,352 B << 512 KB → 单缓冲, kL1=1
```

---

## 7. Pipeline 流水分析

### 7.1 AIC/AIV 并行结构

```mermaid
sequenceDiagram
    participant AIC as AIC (Cube)
    participant AIV as AIV (Vector)

    Note over AIV: CopyIn + GetSinCos

    rect rgb(225, 112, 85, 0.3)
    AIC->>AIC: ① MM1 MatmulCq (17.89 us)
    end
    AIC-->>AIV: SYNC_MMCQ ✉️

    rect rgb(108, 92, 231, 0.3)
    Note over AIV: 全核同步 (V→V)
    AIV->>AIV: ② RmsNormCq (0.015 us)
    end

    par AIC 做 MM2 & AIV 做 RmsNormCq
        rect rgb(225, 112, 85, 0.3)
        AIC->>AIC: ③ MM2 MatmulCkvKr (17.89 us)
        end
    end

    AIV-->>AIC: SYNC_RMSNORM_CQ ✉️
    AIC-->>AIV: SYNC_MMCKVKR ✉️

    rect rgb(108, 92, 231, 0.3)
    AIV->>AIV: ④ RmsNormCkv (0.005 us)
    AIV->>AIV: ⑤ RopeKr (0.001 us)
    AIV->>AIV: ⑥ ScatterCkvKr (0.002 us)
    end

    rect rgb(225, 112, 85, 0.3)
    AIC->>AIC: ⑦ MM3 MatmulQcQr (47.19 us)
    end
    AIC-->>AIV: SYNC_MMQCQR ✉️

    Note over AIC,AIV: SYNC_ALL_CUBE (全核同步)

    par AIC 做 MM4 & AIV 做 RopeQr
        rect rgb(225, 112, 85, 0.3)
        AIC->>AIC: ⑨ MM4 MatmulQn (0.66 us)
        end
        rect rgb(108, 92, 231, 0.3)
        AIV->>AIV: ⑧ RopeQr (0.105 us)
        end
    end
```

> [在线编辑此图](https://l.mermaid.ai/y7XF2T)

`[括号]` 表示可选阶段（取决于量化模式或 Split-KN 配置）。
AccumCq/AccumCkvKr 在 AIV 上执行，与下一个 AIC matmul **并行**，可被完全遮盖。

### 7.2 示例: BF16 Decode BS=1 流水线

```
AIC 阶段:
  MM1_Cq:     [ 0.00 — 17.89]  17.89 us  ★ 关键路径
  MM2_CkvKr:  [18.69 — 36.58]  17.89 us  ★ 关键路径
  MM3_QcQr:   [38.18 — 85.37]  47.19 us  ★ 关键路径 (54%)
  MM4_Qn:     [86.97 — 87.63]   0.66 us  ★ 关键路径

AIV 阶段:
  RmsNormCq:     0.015 us  (被 MM2 遮盖)
  RmsNormCkvKr:  0.005 us  (被 MM3 遮盖)
  RopeQr:        0.105 us  (被 MM4 遮盖)

Pipeline 总计: 87.63 us
AIC 利用率: 83.63 / 87.63 = 95.4%
```

> **工具**: `python perf_analyzer.py --chip 950 --mode pipeline` 查看 ASCII 时间线

### 7.3 Split-KN 优化后

```
MM1:N8×K4 + MM2:N3×K8:
  MM1:  3.93 us (↓ 78%)   AccumCq:   0.002 us (被 MM2 遮盖)
  MM2:  0.78 us (↓ 96%)   AccumCkvKr: 0.001 us (被 MM3 遮盖)
  MM3: 47.19 us (不变)
  MM4:  0.66 us

Pipeline 总计: 53.31 us (↓ 39.2%)
```

---

## 8. 优化决策指南

```mermaid
graph TD
    START["性能优化决策"] --> SCENE{"场景类型?"}

    SCENE -->|"Decode<br/>(T=1~32)"| DECODE_BOUND{"瓶颈类型?<br/>(--mode roofline)"}
    SCENE -->|"Prefill<br/>(T≥128)"| PREFILL_BOUND{"N 轴大小?"}

    DECODE_BOUND -->|"AI << Ridge<br/>HBM 瓶颈"| HBM_OPT{"N 轴核利用率?<br/>(--mode bound)"}
    DECODE_BOUND -->|"AI >> Ridge<br/>Cube 瓶颈"| CUBE_OPT["增大 baseM×baseK<br/>FP8 (2× MMAD)<br/>(--mode block-search)"]

    HBM_OPT -->|"N 小, 核数不足<br/>(MM1: 24核, MM2: 9核)"| SPLIT_KN["Split-KN 2D 切分<br/>n_groups × k_groups = 32<br/>(--mode split-kn)<br/>收益: 最高 4.6×"]
    HBM_OPT -->|"N 大, 核数充足<br/>(MM3: 32核)"| QUANT["INT8/FP8 量化<br/>B 矩阵减半<br/>(--mode report)<br/>收益: ~2×"]

    PREFILL_BOUND -->|"N 较小<br/>(N≤16)"| SPLIT_M["Split-M (M轴切分)<br/>每核处理 T/m_groups tokens<br/>B 矩阵 L2 复用<br/>(--mode split-m)<br/>收益: 最高 4.8×"]
    PREFILL_BOUND -->|"N 较大<br/>(N=128)"| K_OUTER{"baseM 是否够大?<br/>(baseM ≥ 88 for BF16)"}

    K_OUTER -->|"是: Cube bound"| K_OUTER_OPT["K-Outer Loop 优化<br/>B 在 L1 内跨 N-block 复用<br/>(--mode prefill)<br/>收益: 3.3×"]
    K_OUTER -->|"否: HBM bound"| BLOCK_SEARCH["搜索最优 block 尺寸<br/>(--mode block-search)<br/>+ 量化"]

    style START fill:#2d3436,color:#fff
    style SPLIT_KN fill:#00b894,color:#fff
    style QUANT fill:#00b894,color:#fff
    style SPLIT_M fill:#00b894,color:#fff
    style K_OUTER_OPT fill:#00b894,color:#fff
    style CUBE_OPT fill:#00b894,color:#fff
    style BLOCK_SEARCH fill:#00b894,color:#fff
    style SCENE fill:#fdcb6e,color:#2d3436
    style DECODE_BOUND fill:#fdcb6e,color:#2d3436
    style PREFILL_BOUND fill:#fdcb6e,color:#2d3436
    style HBM_OPT fill:#fdcb6e,color:#2d3436
    style K_OUTER fill:#fdcb6e,color:#2d3436
```

> [在线编辑此图](https://l.mermaid.ai/7cVZj0)

### 8.1 HBM 带宽瓶颈 (AI << Ridge, 典型 Decode)

| 优化手段 | 适用场景 | 预期收益 |
|---------|---------|---------|
| INT8/FP8 量化 | 所有 Matmul | ~2× (B 矩阵减半) |
| Split-KN 2D 切分 | N 轴小的 MM (MM1, MM2) | 最高 4.6× (受益于更多核) |
| L2 复用 | Split-N 共享 A 矩阵 | 已内置 |
| 增大批量 (Prefill) | T > 512 | 转为计算瓶颈 |
| Split-M (M 轴切分) | N 较小 + 大 T | 避免核间同步，L2 复用 B |

### 8.2 计算瓶颈 (AI >> Ridge, 大批量 Prefill)

| 优化手段 | 适用场景 | 预期收益 |
|---------|---------|---------|
| 增大 baseM × baseK 填满 L0A | Prefill T ≥ 512 | Cube 效率 ~90%+ |
| 增大 stepK 减少 kL1 循环 | L1 允许时 | 减少 MTE1 开销 |
| FP8 (2× MMAD 吞吐) | 精度允许时 | 2× 计算 |

### 8.3 关键洞察

**B 矩阵总量固定**: 对于给定 Matmul，`K × N × sizeof(B)` 总量不变。Split-KN 不减少总 HBM 访存量，只在 per-core 负载不均时（小 N）通过增加并行度获益。当 B 矩阵占主导时（如 MM3），**量化是唯一有效手段**。

---

## 工具命令速查

| 分析目标 | 命令 |
|---------|------|
| 各阶段瓶颈分析 | `--mode bound` |
| AIC/AIV 流水时间线 | `--mode pipeline` |
| 内核总时间估算 | `--mode estimate` |
| 优化建议 | `--mode advice` |
| stepBatchSize 搜索 | `--mode search` |
| 全测试用例报告 | `--mode report` |
| Roofline 分析 | `--mode roofline` |
| 最优 baseM/N/K 搜索 | `--mode block-search` |
| 2D Split-KN 搜索 | `--mode split-kn` |
| M 轴切分搜索 (Prefill) | `--mode split-m` |
| 全部分析 | `--mode full` |

所有命令均需加 `--chip 950` 以使用 per-cycle 模型。
