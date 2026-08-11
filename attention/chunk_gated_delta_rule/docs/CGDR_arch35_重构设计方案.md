# CGDR arch35 算子重构设计方案

## 1 背景与目标

ChunkGatedDeltaRule（CGDR）算子当前在 arch35（Ascend 950PR / 950DT / Atlas A3）平台上采用三阶段（Stage1 / Stage2 / Stage3）流水线实现，各 Stage 之间通过 `SyncAll<false>()` 全局同步。Stage1 内部以 `paraNum_`（当前为 4）个 chunk 为一批并行处理，Stage2 按序列递推逐 chunk 执行，Stage3 全 chunk 并行。

本方案围绕以下 6 项重构需求展开，目标是**减少 Stage1 的向量计算量、削减 workspace 内存、消除冗余 DataCopy**，从而提升 arch35 平台整体性能：

| 编号 | 需求摘要 |
|:-----|:---------|
| R1 | 分析各 Stage 改为"单 chunk 从头算到尾"的性能收益 |
| R2 | 分析 Stage1 中 QKPreprocess 能否省略，直接对原始 GM 做 KKT / QKT matmul |
| R3 | 将 K 衰减系数（`g_cum_exp`）移至 Stage2 计算，Stage2 直接与 K 做 matmul |
| R4 | Stage1 中 beta 直接与求逆结果做按元素乘，再与 V 做 matmul |
| R5 | Stage1 中 beta 与 A 相乘后再与 `-gamma` 做元素乘，然后直接与 K 做 matmul，作为 `-W` 矩阵 |
| R6 | Stage2 中根据 `cumsumg` 重新计算（不再从 Stage1 传递 g_cum） |

---

## 2 当前实现概述

### 2.1 整体流程

```
Process()
│
└── for batch in [0, B):
      for pos in [seqStart, seqEnd, maxGroupLength):   # ChunkGroup 循环
            │
            ├── RunStage1(cg)        → 产出 gCum, kCumDecay, vInner, qPrime, kg, qkt
            ├── SyncAll<false>()
            ├── RunStage2(cg, state) → 递推 state，产出 attnInter，更新 vInner→v_new
            ├── SyncAll<false>()
            ├── RunStage3(cg)        → 产出最终 attnOut
            └── SyncAll<false>()
```

### 2.2 Stage1 当前计算流程

Stage1 以 `paraNum_` 个 chunk 为一批，AIC / AIV 协同：

**AIV 侧：**

| 步骤 | 函数 | 说明 |
|:-----|:-----|:-----|
| 1 | `QKPreProcess` | 将 Q/K 从 strided GM `(T, Nk, Dk)` 搬入连续 workspace `(C, Dk)`，同时输出 `kg`（无 gamma 时） |
| 2 | `GCumExpCompute` | `g_cum = g.cumsum()`, `g_cum_exp = g_cum.exp()` |
| 3 | `GammaCompute` | `decay_mask = exp(g_cum[:,None] - g_cum[None,:]) * tril_mask`（gamma 矩阵） |
| 4 | `BetaCopyInWithStride` | 搬入 beta |
| 5 | `KKBetaCompute` | `attn = (k@k^T) * beta[:,None]`（beta 乘到 kkt 行上） |
| 6 | `InverseCompute` | `attn *= gamma; attn = -attn; 前代法求逆 → A = (I + kkt*beta*gamma)^{-1}` |
| 7 | `GBKCompute` | `k_cumdecay = A @ (-k * beta * g_cum_exp)`; `kg = k * exp(g_cum[-1] - g_cum)` |
| 8 | `VBetaCompute` | `v_beta = v * beta`; 输出到 workspace |
| 9 | `QPrimeCompute` | `q_prime = q * scale * g_cum_exp` |

**AIC 侧：**

| 步骤 | matmul | 说明 |
|:-----|:-------|:-----|
| 1 | `keyCon @ keyCon^T → kkWs` | KKT = K @ K^T |
| 2 | `queryCon @ keyCon^T → qkt` | QKT = Q @ K^T |
| 3 | `AttnInverseMMCompute` | 求逆辅助 matmul（左下角块） |
| 4 | `attn @ gBK → kCumDecay` | k_cumdecay = A @ gBK |
| 5 | `attn @ vBeta → vInner` | v_inner = A @ v_beta |

**Stage1 输出：** `gCum`(FP32), `kCumDecay`(BF16), `vInner`(BF16/FP32), `qPrime`(BF16), `kg`(BF16), `qkt`(BF16)

### 2.3 Stage2 当前计算流程

Stage2 按 chunk 序列递推（state 依赖），每个 chunk 内 AIV/AIC 协同：

**AIV 侧：**
- 搬入 state，Cast 到 BF16（`curStateBf16`）
- 读取 `gCum` 最后一个值 `g_cum[-1]`，计算 `exp(g_cum[-1])`，对 state 做行缩放：`state *= exp(g_cum[-1])`

**AIC 侧：**

| 步骤 | matmul | 说明 |
|:-----|:-------|:-----|
| 1 | `kCumdecay @ state^T → vInner` (accum=1) | `v_prime = k_cumdecay @ state^T`; `v_new = v_inner + v_prime` |
| 2 | `qPrime @ state^T → out` | `attn_inter = q_prime @ state^T` |
| 3 | `vInner^T @ kg → stateOut` (accum=1) | `state_new = state * exp(g_cum[-1]) + v_new^T @ kg` |

### 2.4 Stage3 当前计算流程

全 chunk 并行，AIV 计算 `masked_qkt = qkt * scale * decay_mask * mask`，AIC 执行 `attn_out = masked_qkt @ v_new` (accum=1，叠加 attn_inter)。

---

## 3 需求分析与设计方案

### 3.1 R1 — 单 chunk 从头算到尾的性能分析

#### 3.1.1 当前模式

当前各 Stage 独立执行，Stage1 内部以 `paraNum_=4` 个 chunk 为批，三个 Stage 之间通过 `SyncAll<false>()` 严格串行：

```
Stage1(chunk0..3) → SyncAll → Stage2(chunk0..3) → SyncAll → Stage3(chunk0..3) → SyncAll
```

#### 3.1.2 提议模式

改为"单 chunk 从头算到尾"：对每个 chunk 依次执行 Stage1 → Stage2 → Stage3，无需 Stage 间全局同步：

```
for chunk in [0, numChunk):
    Stage1(chunk) → Stage2(chunk) → Stage3(chunk)
```

#### 3.1.3 性能分析

| 维度 | 当前（多 chunk 批处理） | 提议（单 chunk 贯穿） | 结论 |
|:-----|:----------------------|:---------------------|:-----|
| **Cube 利用率** | Stage1 可将 4 个 chunk 的 matmul 排队，Cube 连续执行 | 单 chunk matmul 数量少，Cube 空闲风险增大 | 当前更优 |
| **AIC/AIV 流水** | Stage1 内 AIC 做 matmul 时 AIV 可做下一个 chunk 的预处理 | 单 chunk 贯穿时，同一 chunk 的 Stage1 AIV 完成后即可启动 Stage2 AIV，Stage1 AIC 仍可与 Stage2 AIV 并行 | **各有优劣** |
| **SyncAll 开销** | 3 次 SyncAll per chunk group | 可减至 0~1 次（仅 ChunkGroup 间同步） | **提议更优** |
| **Workspace 内存** | 需 `paraNum_` 份临时 buffer | 仅需 1 份 | **提议更优** |
| **Stage2 递推** | Stage2 已是逐 chunk 递推，不受影响 | 无变化 | 持平 |
| **尾 chunk 延迟** | 最后一批需等全部 chunk 的 Stage1 完成 | 最后一个 chunk 完成即可输出 | **提议更优** |

**关键结论：**

- **不建议完全改为单 chunk 贯穿**。Stage1 的 matmul（KKT、QKT、求逆辅助、k_cumdecay、v_inner 共 5 次）在单 chunk 模式下 Cube 利用率会显著下降，尤其是 `chunkSize=64, dk/dv<=128` 时单次 matmul 计算量很小。
- **建议采用折中方案**：Stage1 仍保留 `paraNum_` 批处理以维持 Cube 利用率；但去掉 Stage 间 `SyncAll`，改为**Stage 间流水**——当 Stage1 完成 chunk i 后，Stage2 可以立即开始处理 chunk i，同时 Stage1 继续处理 chunk i+1。这需要将 Stage2/Stage3 改为按单 chunk 触发，通过 CrossCore flag 而非 SyncAll 同步。
- **Workspace 削减**：即使保留 `paraNum_`，R3~R6 的变更本身会大幅减少 Stage1 输出量，间接降低 workspace 需求。

#### 3.1.4 推荐方案

```
Stage1: 仍以 paraNum_ 批处理，但输出逐 chunk 就绪
Stage2: 逐 chunk 递推（保持不变），通过 CrossCore flag 与 Stage1 同步
Stage3: 全 chunk 并行（保持不变）
去除 Stage 间 SyncAll，仅保留 ChunkGroup 间 SyncAll
```

### 3.2 R2 — 省略 QKPreprocess 的分析

#### 3.2.1 当前 QKPreprocess 的作用

`QKPreProcess` 将 Q/K 从原始 GM `(T, Nk, Dk)` 按 stride `nk*dk` 逐行搬入连续 workspace `(C, Dk)`：

```cpp
// queryGm: (T, Nk, Dk), 行间距 = nk * dk
// queryConGm: (C, Dk), 连续
DataCopyInBf16WithStride(subValidRows, dk_, srcGm, nk_ * dk_);  // strided → continuous
```

之后所有 matmul（KKT、QKT）均从 `queryConGm_` / `keyConGm_` 读取。

#### 3.2.2 直接使用原始 GM 做 matmul 的可行性

MatmulImpl 支持通过 `SetOrgShape(M, N, K, Ka, Kb)` 指定 A/B 矩阵的行 stride：

- A 矩阵（Q 或 K）：`K = dk`，`Ka = nk * dk`（行间距）
- B 矩阵（K，transB=true）：`K = dk`，`Kb = nk * dk`

```cpp
// 直接从原始 GM 读取，无需 QKPreprocess
mmBf16_.SetOrgShape(chunkSize_, chunkSize_, dk_, nk_ * dk_, nk_ * dk_);
mmBf16_.SetTensorA(queryGm_[chunkStart * nk * dk + head * dk]);  // strided GM
mmBf16_.SetTensorB(keyGm_[chunkStart * nk * dk + head * dk], true);
```

#### 3.2.3 影响分析

| 项目 | 当前（QKPreprocess） | 提议（直接 strided GM） |
|:-----|:---------------------|:----------------------|
| MTE3 带宽 | 需搬 Q+C×Dk + K+C×Dk 到 workspace | 无搬入，Cube 直接从 GM 读取 |
| Workspace | 需 `queryConGm_` + `keyConGm_`（各 `paraNum_ × C × Dk × coreNum`） | 节省这部分 workspace |
| L1/L0A 带宽 | Cube 从连续 workspace 读，L1 命中率高 | strided GM 读取可能导致 L1 miss 增加 |
| 尾块处理 | QKPreprocess 内做 padding 补零 | 需 matmul 支持非整块 M 维度（已有 `validLenBatch_` 支持） |
| `kg` 输出 | QKPreprocess 同时输出 kg（无 gamma 时） | 需另外处理 kg 的输出路径 |

**结论：可行，但需注意 L1 命中率。**

arch35 的 L1 cache 对 strided GM 访问的命中率取决于 stride 大小。当 `nk * dk` 较大时（如 `nk=16, dk=128 → stride=2048B`），每行 256B 远小于 cache line 效率窗口，L1 仍能有效预取。建议：

1. **KKT 和 QKT 的 matmul 直接使用原始 GM**，通过 `SetOrgShape` 的 Ka/Kb 参数指定 stride。
2. **`GBKCompute` 中的 K 读取仍需连续拷入 UB**（因为需要在 AIV 侧做按元素乘），这部分保持不变。
3. **`kg` 的输出路径**：当 `gOptional=false` 时，当前 QKPreprocess 同时输出 kg（即 key 的连续拷贝）。省略后，kg 可在 `GBKCompute` 中直接从原始 GM 搬入并输出，或在 Stage2 中按需计算。
4. 删除 `queryConGm_`、`keyConGm_` workspace 分配。

### 3.3 R3 — K 衰减系数移至 Stage2

#### 3.3.1 当前计算

Stage1:
```
k_cumdecay_raw = -k * beta * g_cum_exp          # 逐行缩放, shape (C, Dk)
k_cumdecay = A @ k_cumdecay_raw                  # matmul, shape (C, Dk)
```

Stage2:
```
v_prime = k_cumdecay @ state^T                   # matmul, shape (C, Dv)
v_new = v_inner + v_prime
```

#### 3.3.2 变更后计算

Stage1 不再乘 `g_cum_exp`，仅计算：
```
k_beta_inv = A @ (-k * beta)                     # 不含 g_cum_exp, shape (C, Dk)
```

Stage2 在 matmul 前对 `k_beta_inv` 逐行乘 `g_cum_exp`，或等价地在 matmul 后对结果逐行乘：
```
# 方式 A：matmul 前缩放左矩阵
v_prime = (k_beta_inv * g_cum_exp[:,None]) @ state^T

# 方式 B：matmul 后缩放结果（数学等价，因 g_cum_exp 是行缩放）
tmp = k_beta_inv @ state^T
v_prime = tmp * g_cum_exp[:,None]
```

**数学等价性证明：**

由于 `g_cum_exp` 是对 (C, Dk) 矩阵的**行缩放**（每行乘同一个标量），而 matmul `(C,Dk) @ (Dk,Dv) → (C,Dv)` 的输出行与左矩阵行一一对应：

$$
\text{v\_prime}[i,:] = g_{\text{cum\_exp}}[i] \cdot \sum_j (\text{k\_beta\_inv}[i,j] \cdot S[j,:])
= g_{\text{cum\_exp}}[i] \cdot (\text{k\_beta\_inv} @ S^T)[i,:]
$$

因此方式 A 与方式 B 完全等价。**推荐方式 B**（matmul 后缩放），因为：
- matmul 可直接使用 BF16 输入（无需先将 `g_cum_exp` 乘入 k_beta_inv 导致精度变化）
- 缩放操作在 AIV 侧用一条 `Muls` + `Mul` 即可完成，开销极小

#### 3.3.3 实现变更

| 组件 | 变更内容 |
|:-----|:---------|
| Stage1 `GBKCompute` | 删除 `g_cum_exp` 相关计算：`gBUbFloat_ = -beta`（不再乘 `g_cum_exp`）；`gBK = gBUb * k`；`k_beta_inv = A @ gBK` |
| Stage1 输出 | `kCumDecay` → `kBetaInv`（不含 g 衰减），workspace 大小不变 |
| Stage2 `CalVPrime` | matmul 后增加 `Muls(v_prime, g_cum_exp)` 行缩放 |
| Stage2 输入 | 需要获取 `g_cum_exp`（见 R6） |

### 3.4 R4 — beta 与求逆结果做元素乘后再与 V 做 matmul

#### 3.4.1 当前计算

```
v_beta = v * beta[:,None]        # V 逐行乘 beta
v_inner = A @ v_beta             # matmul
```

即 `v_inner[i,:] = Σ_j A[i,j] · v[j,:] · beta[j]`（beta 作用在归约维 j 上）。

#### 3.4.2 变更后计算

```
A_beta = beta[:,None] * A        # A 逐行乘 beta
v_inner = A_beta @ v             # matmul
```

即 `v_inner[i,:] = beta[i] · Σ_j A[i,j] · v[j,:]`（beta 作用在输出维 i 上）。

#### 3.4.3 数学等价性分析

两种方式**在当前 KKT 包含 beta 的前提下不等价**：

$$
\underbrace{\sum_j A[i,j] \cdot v[j,:] \cdot \beta[j]}_{\text{当前}} \neq \underbrace{\beta[i] \cdot \sum_j A[i,j] \cdot v[j,:]}_{\text{变更后}}
$$

**关键前提变更**：本重构同时变更 KKT 计算——KKT 不再乘 beta（见 R5），即求逆的矩阵变为 `A = (I + (k@k^T) * gamma)^{-1}`（不含 beta）。在此前提下，beta 需要从 KKT 中移出，改为在求逆结果 A 上以行缩放方式施加。

变更后的完整等式：

$$
v_{\text{inner}} = \text{diag}(\beta) \cdot A \cdot V
$$

这对应 delta rule 中 `beta_t` 作为每个时间步的更新强度因子，在 chunked 公式中以行缩放形式作用于输出——与原始递推公式 `S_t += beta_t · (v_t - S_{t-1} k_t) k_t^T` 中 beta 的时间步粒度一致。

#### 3.4.4 实现变更

| 组件 | 变更内容 |
|:-----|:---------|
| Stage1 `KKBetaCompute` | **移除 beta 乘法**：`attn = kk`（不再 `* beta`） |
| Stage1 `InverseCompute` | 求逆输入不含 beta，得到 `A = (I + kk * gamma)^{-1}` |
| Stage1 `InverseCompute` 后 | 新增：`A = beta[:,None] * A`（行缩放，一条 `Mul` + `Broadcast`） |
| Stage1 `VBetaCompute` | **移除 beta 乘法**：直接输出 v（不再 `v * beta`） |
| Stage1 AIC matmul 5 | `attn @ v → vInner`（直接用 v，不再用 v_beta） |
| Stage1 workspace | 删除 `vBetaWsGm_` |

### 3.5 R5 — beta·A·(-gamma)·K 作为 -W 矩阵

#### 3.5.1 当前计算

```
k_cumdecay_raw = -k * beta * g_cum_exp
k_cumdecay = A @ k_cumdecay_raw
```

其中 `A = (I + kk * beta * gamma)^{-1}`（当前 KKT 含 beta）。

#### 3.5.2 变更后计算

在 R4 的前提下（KKT 不含 beta，`A = (I + kk * gamma)^{-1}`），且 R3 将 `g_cum_exp` 移至 Stage2：

```
A_beta = beta[:,None] * A              # R4 已计算
A_beta_neg_gamma = A_beta * (-gamma)   # 逐元素乘 -gamma（decay_mask）
-W = A_beta_neg_gamma @ K              # matmul with K
→ W = (beta * A * gamma) @ K
```

即 Stage1 输出的 `-W` 矩阵为：

$$
W = \text{diag}(\beta) \cdot A \cdot \Gamma \cdot K
$$

其中 $\Gamma$ 为 decay_mask 矩阵（`gamma`），$A = (I + K K^T \Gamma)^{-1}$。

> **注**：这里 `-gamma` 的负号来自原始公式中 `k_cumdecay = A @ (-k * beta * g_cum_exp)` 的负号。变更后负号通过 `-gamma` 施加在 A 上，等价于 `W = -A @ (-gamma) @ K = A @ gamma @ K` 乘以 beta 行缩放后取负。

#### 3.5.3 实现变更

| 组件 | 变更内容 |
|:-----|:---------|
| Stage1 `GBKCompute` | **完全重写**：不再计算 `-k * beta * g_cum_exp`；改为 `A_beta *= (-gamma)`，然后 AIC 执行 `A_beta_neg_gamma @ K → kBetaInv` |
| Stage1 AIC matmul 4 | 左矩阵从 `gBKWsGm_` 变为 `A_beta_neg_gamma`（从 attn workspace 读取），右矩阵从 `gBKWsGm_` 变为原始 K GM 或连续 K |
| Stage1 `kg` 计算 | `kg = k * exp(g_cum[-1] - g_cum)` 仍需保留（Stage2 state 更新用），可独立计算 |

#### 3.5.4 A_beta_neg_gamma 的计算路径

`A_beta`（beta 行缩放后的求逆结果）在 AIV 侧 UB 中完成。接下来需要乘 `-gamma`（decay_mask，shape `(C, C)`），这也是在 AIV 侧完成的。然后结果需要写到 GM 供 AIC matmul 使用。

```
AIV:  A(beta缩放) → Mul(A, -gamma) → Cast BF16 → DataCopyPad → GM(attnWs)
AIC:  attnWs @ K → kBetaInv (输出到 GM)
```

### 3.6 R6 — Stage2 根据 cumsumg 重新计算

#### 3.6.1 动机

R3 将 `g_cum_exp` 移至 Stage2 后，Stage2 需要每个 chunk 的 `g_cum_exp`。当前 `g_cum` 由 Stage1 计算并写入 GM workspace。为减少 Stage1→Stage2 的数据传递和 workspace 占用，Stage2 直接从原始 `g` 输入重新计算 `cumsumg`。

#### 3.6.2 实现方案

Stage2 AIV 侧，在每个 chunk 处理时：

```cpp
// 从原始 g GM 搬入当前 chunk 的 g (长度 curChunkSize)
DataCopyInFp32WithStride(curChunkSize, 1, gGm_[nvId * Sp + chunkPos * Nv], Nv);
// CumSum
CumSum<float>(gCumUb, gTmpUb, gTmpUb, {1, chunkSize});
// Exp
Exp<float, 0, true>(gCumExpUb, gCumUb, chunkSize);
```

计算出的 `g_cum_exp` 用于：
1. **R3**：`v_prime = (k_beta_inv @ state^T) * g_cum_exp[:,None]` — 行缩放 v_prime
2. **state 衰减**：`state *= exp(g_cum[-1])` — 已有逻辑，改为从重新计算的 g_cum 取末值

#### 3.6.3 Stage1 输出削减

| 输出 | 当前 | 变更后 | 说明 |
|:-----|:-----|:-------|:-----|
| `gCum` (FP32, Nv×Sp) | Stage1 计算 → GM | **删除**，Stage2 重算 | R6 |
| `kCumDecay` (BF16, Nv×Sp×Dk) | `A@(-k*beta*g_exp)` | `kBetaInv` = `(beta*A*(-gamma))@K`，不含 g_exp | R3+R5 |
| `vInner` (BF16/FP32) | `A@(v*beta)` | `(beta*A)@v` | R4 |
| `qPrime` (BF16) | `q*scale*g_exp` | 保持不变 | — |
| `kg` (BF16) | `k*exp(g_cum[-1]-g_cum)` | 保持不变 | — |
| `qkt` (BF16) | `q@k^T` | 保持不变 | — |

**Workspace 节省**：`gCum` 的 `sizeof(float) * nv * maxGroupLength` 字节被完全删除。

#### 3.6.4 Stage3 的影响

Stage3 当前从 GM 读取 `gCum` 计算 `decay_mask`。R6 删除 Stage1 的 `gCum` 输出后，Stage3 也需要重新计算 `cumsumg`：

```cpp
// Stage3 AIV: 从原始 g 重新计算
DataCopyInFp32WithStride(curChunkSize, 1, gGm_[nvId*Sp + chunkPos*Nv], Nv);
CumSum → Exp → g_cum
// 然后计算 decay_mask = exp(g_cum[:,None] - g_cum[None,:]) * tril
```

---

## 4 计算过程变更总览

### 4.1 Stage1 变更对比

| 计算项 | 变更前 | 变更后 | 需求 |
|:-------|:-------|:-------|:-----|
| QKPreprocess | strided GM → 连续 WS | **删除**，matmul 直接读 strided GM | R2 |
| KKT | `(k@k^T) * beta` | `k@k^T`（不含 beta） | R4/R5 |
| 求逆矩阵 M | `I + kk * beta * gamma` | `I + kk * gamma` | R4/R5 |
| 求逆结果 A | `(I + kk*beta*gamma)^{-1}` | `(I + kk*gamma)^{-1}` | R4/R5 |
| beta 施加位置 | KKT 行缩放 + V/K 逐行缩放 | A 行缩放 | R4/R5 |
| v_inner | `A @ (v * beta)` | `(beta * A) @ v` | R4 |
| k_beta_inv | `A @ (-k * beta * g_cum_exp)` | `(beta * A * (-gamma)) @ K` | R3/R5 |
| g_cum_exp 计算 | Stage1 GCumExpCompute | **删除**，移至 Stage2/Stage3 | R3/R6 |
| gCum 输出 | 写入 GM | **删除** | R6 |
| vBeta workspace | `v * beta` → GM | **删除**，直接用 V | R4 |
| queryCon/keyCon WS | QKPreprocess 输出 | **删除** | R2 |

### 4.2 Stage2 变更对比

| 计算项 | 变更前 | 变更后 | 需求 |
|:-------|:-------|:-------|:-----|
| g_cum_exp 获取 | 从 GM 读 Stage1 输出的 gCum | 从原始 g 重新 cumsum+exp | R6 |
| v_prime | `k_cumdecay @ state^T` (accum=1) | `(kBetaInv @ state^T) * g_cum_exp[:,None]` (accum=1, 然后行缩放) | R3 |
| state 衰减 | `state *= exp(gCum[-1])` | `state *= exp(g_cum[-1])`（从重算的 g_cum 取末值） | R6 |
| attn_inter | `qPrime @ state^T` | 不变 | — |
| state_new | `v_new^T @ kg` (accum=1) | 不变 | — |

### 4.3 Stage3 变更对比

| 计算项 | 变更前 | 变更后 | 需求 |
|:-------|:-------|:-------|:-----|
| decay_mask 计算 | 从 GM 读 gCum | 从原始 g 重新 cumsum+exp | R6 |
| masked_qkt | `qkt * scale * decay * mask` | 不变 | — |
| attn_out | `masked_qkt @ v_new` (accum=1) | 不变 | — |

### 4.4 数据流图（变更后）

```
原始输入: Q, K, V, beta, g, state, scale
                    │
    ┌───────────────┼───────────────────────────┐
    │            Stage1 (per chunk)              │
    │                                           │
    │  AIV:                                     │
    │    beta ← GM (stride)                     │
    │    gamma = exp(g_cum[:,None]-g_cum[None,:])│  ← g_cum 在 AIV 内部计算
    │    kk = K @ K^T (AIC matmul, strided GM)  │  ← R2: 直接读 GM
    │    attn = kk * gamma                      │  ← R4: 不含 beta
    │    A = (I + attn)^{-1}  (前代法求逆)       │
    │    A_beta = beta[:,None] * A              │  ← R4: beta 行缩放
    │    A_beta_neg_gamma = A_beta * (-gamma)   │  ← R5: 乘 -gamma
    │    → DataCopyPad → GM (attnWs)            │
    │    q_prime = q * scale * g_cum_exp        │
    │    kg = k * exp(g_cum[-1] - g_cum)        │
    │                                           │
    │  AIC:                                     │
    │    KKT = K @ K^T  (strided GM)            │  ← R2
    │    QKT = Q @ K^T  (strided GM)            │  ← R2
    │    求逆辅助 matmul                         │
    │    kBetaInv = A_beta_neg_gamma @ K → GM   │  ← R3/R5
    │    vInner = A_beta @ V → GM               │  ← R4
    │                                           │
    │  输出: kBetaInv, vInner, qPrime, kg, qkt  │
    │  (不输出 gCum)                             │  ← R6
    └───────────────┬───────────────────────────┘
                    │
    ┌───────────────┼───────────────────────────┐
    │            Stage2 (per chunk, 递推)        │
    │                                           │
    │  AIV:                                     │
    │    g ← GM (stride) → cumsum → exp         │  ← R6: 重算 g_cum_exp
    │    state *= exp(g_cum[-1])                │
    │                                           │
    │  AIC:                                     │
    │    tmp = kBetaInv @ state^T (accum=1)     │  ← vInner += tmp
    │    v_prime = tmp * g_cum_exp[:,None]      │  ← R3: 行缩放
    │    (vInner 已含 v_inner, v_new = vInner)  │
    │    attn_inter = qPrime @ state^T → out    │
    │    state_new = vInner^T @ kg (accum=1)    │
    └───────────────┬───────────────────────────┘
                    │
    ┌───────────────┼───────────────────────────┐
    │            Stage3 (per chunk, 并行)        │
    │                                           │
    │  AIV:                                     │
    │    g ← GM → cumsum → exp → decay_mask     │  ← R6: 重算
    │    masked_qkt = qkt * scale * decay * mask│
    │                                           │
    │  AIC:                                     │
    │    attn_out = masked_qkt @ v_new (accum=1)│
    └───────────────────────────────────────────┘
```

---

## 5 Workspace 变更

### 5.1 删除的 workspace

| 项目 | 大小 | 说明 |
|:-----|:-----|:-----|
| `queryConGm_` | `BF16 × C × Dk × paraNum × coreNum` | R2: QKPreprocess 删除 |
| `keyConGm_`` | `BF16 × C × Dk × paraNum × coreNum` | R2: QKPreprocess 删除 |
| `vBetaWsGm_` | `BF16 × C × Dv × paraNum × coreNum` | R4: v_beta 删除 |
| `gBKWsGm_` | `BF16 × C × Dk × paraNum × coreNum` | R5: gBK 删除，改为 A_beta_neg_gamma @ K |
| `gCum_` (interWorkspace) | `FP32 × Nv × maxGroupLength` | R6: gCum 不再输出 |

### 5.2 新增的 workspace

| 项目 | 大小 | 说明 |
|:-----|:-----|:-----|
| Stage2 g_cum UB | `FP32 × C`（UB 内） | R6: Stage2 重算 g_cum（无需额外 GM workspace） |
| Stage3 g_cum UB | `FP32 × C`（UB 内） | R6: Stage3 重算 g_cum（无需额外 GM workspace） |

### 5.3 净效果

GM workspace 显著减少（删除 4 个 `paraNum × coreNum` 级别的 buffer + 1 个 `Nv × maxGroupLength` 级别的 buffer），UB 开销增加极小（仅 2 个 `C` 长度的 FP32 buffer）。

---

## 6 实现计划

### 6.1 实施步骤

| 步骤 | 内容 | 涉及文件 | 验证方式 |
|:-----|:-----|:---------|:---------|
| 1 | Stage1: 移除 KKBetaCompute 中的 beta 乘法 | `stage1.h` | UT 比对 |
| 2 | Stage1: InverseCompute 后新增 beta 行缩放 | `stage1.h` | UT 比对 |
| 3 | Stage1: VBetaCompute 改为直接输出 V（删除 beta 乘法） | `stage1.h` | UT 比对 |
| 4 | Stage1: GBKCompute 重写为 A_beta*(-gamma) → GM，AIC matmul 用 K | `stage1.h` | UT 比对 |
| 5 | Stage1: 删除 GCumExpCompute 的 gCum GM 输出（仅保留 AIV 内部 g_cum 计算） | `stage1.h` | UT 比对 |
| 6 | Stage1: 删除 QKPreprocess，matmul 改用 strided GM + SetOrgShape | `stage1.h`, `tiling.cpp` | UT + matmul tiling 验证 |
| 7 | Stage2: 新增 g cumsum+exp 重算 | `stage2.h` | UT 比对 |
| 8 | Stage2: CalVPrime 后新增 g_cum_exp 行缩放 | `stage2.h` | UT 比对 |
| 9 | Stage3: 新增 g cumsum+exp 重算（替代从 GM 读 gCum） | `stage3.h` | UT 比对 |
| 10 | Host: 更新 tiling workspace 大小计算 | `tiling.cpp` | 编译验证 |
| 11 | Host: 更新 matmul tiling（strided GM 的 Ka/Kb 参数） | `tiling.cpp` | 编译验证 |
| 12 | 全量 pytest 回归 | `tests/pytest/` | 精度+性能 |

### 6.2 风险与对策

| 风险 | 影响 | 对策 |
|:-----|:-----|:-----|
| KKT 不含 beta 导致求逆结果变化 | 数值精度变化 | 需更新 golden 参考实现，验证精度在阈值内 |
| strided GM matmul 的 L1 命中率下降 | 性能回退 | 先在典型 shape（nk=4/8, dk=64/128）上 benchmark，若回退则保留 QKPreprocess |
| Stage2/Stage3 重算 g_cum 的 AIV 开销 | AIV 负载增加 | cumsum+exp 仅 `C` 个元素，开销可忽略 |
| A_beta_neg_gamma 需从 UB 搬到 GM 供 AIC matmul | MTE3 带威增加 | 与当前 attnWs 搬出路径相同，无额外开销 |
| Stage1 AIV/AIC 同步关系变化 | 死锁或数据竞争 | 重新设计 CrossCore flag 序列，逐 chunk 验证 |

### 6.3 兼容性

- **TilingKey 不变**：BF16/FP32 state × 有/无 gamma 的 4 种组合保持不变
- **接口不变**：`aclnnChunkGatedDeltaRule` 入参与返回值不变
- **arch22 不受影响**：本次重构仅涉及 `op_kernel/arch35/` 目录

---

## 7 总结

本重构方案通过 6 项变更，实现以下目标：

1. **R1**：建议保留 Stage1 批处理 + 去除 Stage 间 SyncAll 的折中方案，兼顾 Cube 利用率与流水效率
2. **R2**：省略 QKPreprocess，matmul 直接读 strided GM，削减 workspace 和 MTE3 带宽
3. **R3**：g_cum_exp 衰减移至 Stage2，以 matmul 后行缩放实现，数学等价
4. **R4**：beta 从 V/K 逐行缩放改为 A 行缩放，配合 R5 中 KKT 去 beta
5. **R5**：k_beta_inv = (beta·A·(-gamma))@K，将 beta 和 gamma 从操作数移到求逆结果上
6. **R6**：Stage2/Stage3 从原始 g 重算 cumsumg，删除 gCum 的 GM workspace

**预期收益**：Stage1 AIV 计算量减少（删除 v_beta、gBK 中的按元素乘），workspace 显著削减（5 个 GM buffer 删除），Stage 间同步开销降低。
