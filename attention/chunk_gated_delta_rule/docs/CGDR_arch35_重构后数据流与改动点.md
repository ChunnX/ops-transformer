# CGDR arch35 重构后数据流与改动点

> 本文档描述重构后 arch35 平台 CGDR 算子的完整计算流程，与重构前逐项对比。
> 重构仅保留 **R2**（strided GM 直读），数学公式与重构前完全一致。

---

## 0 重构范围说明

初始重构方案曾包含 R2-R7 六项改动，经与参考实现（`chunk_gated_delta_rule_golden.py`）
逐一比对后，发现 R3/R4/R5/R7 存在数学公式错误，R6 存在 strided 读取 bug。
最终仅保留 **R2**（纯数据流优化，不改变任何数学公式）。

| 编号 | 需求 | 状态 | 原因 |
|:-----|:-----|:-----|:-----|
| R2 | 省略 QKPreProcess，matmul 直接读 strided GM | **保留** | 纯数据流优化，数学不变 |
| R3 | K 衰减系数移至 Stage2 | **回退** | 对 A@V_beta 项错误施加 g_cum_exp 缩放 |
| R4 | beta 改为求逆后 A 行缩放 | **回退** | `(I-D·M)⁻¹ ≠ D·(I-M)⁻¹`，求逆不分配行缩放 |
| R5 | kBetaInv 用 A_beta*(-gamma)@K | **回退** | 逐元素乘 γ ≠ 矩阵乘对角 g_exp |
| R6 | Stage2/3 从原始 g 重算 cumsumg | **回退** | g 布局 (T,Nv) 非连续，DataCopyPad 读到跨 head 垃圾数据 |
| R7 | 删除 kg_，Stage2 直接用 K | **回退** | `g_exp[-1]/g_exp[j] ≠ g_exp[-1]`，数学不等价 |

### 数学错误详解

以下用参考实现 `chunk_gated_delta_rule_golden.py:102-124` 的正确公式对照说明：

**正确公式（参考实现）：**
```python
k_beta = key * beta.unsqueeze(-1)          # K 乘 beta
v_beta = value * beta.unsqueeze(-1)        # V 乘 beta
attn = -((k_beta @ key.T) * decay_mask)    # KKT 含 beta，乘 gamma
# 逐行前代求逆 (伍德伯里恒等式)
attn = attn + I                            # A = (I - k_beta@K^T*gamma)^{-1}
value = attn @ v_beta                      # v_inner = A @ (V*beta)，beta 在 V 中
k_cumdecay = attn @ (k_beta * g.exp())     # k_cumdecay = A @ (K*beta*g_exp)
```

**R4 错误：** 将 beta 从 KKT/V/K 中移出，改为求逆后 `A_beta = beta·A`。
- 正确：`A = (I - diag(β)·kk·γ)⁻¹`
- R4：`A' = (I - kk·γ)⁻¹`，`A_β = diag(β)·A'`
- 反例（标量）：`b/(1-ab) ≠ b·1/(1-a)`，即 `(I-D·M)⁻¹ ≠ D·(I-M)⁻¹`

**R3 错误：** 将 g_cum_exp 从 k_cumdecay 移到 vInner 整体行缩放。
- 正确：`v_inner = A@V_beta`（无 g_exp），`k_cumdecay = A@(K·β·g_exp)`（有 g_exp）
- R3：`v_inner = (A@V_beta + A@(-K·β)@state) · g_exp`（A@V_beta 项多了 g_exp）

**R5 错误：** 用 `A_beta·(-γ)` 逐元素乘替代 `A@diag(-K·β·g_exp)` 矩阵乘。
- 正确：`k_cumdecay = A @ (-K·β·g_exp)` = `-A @ diag(β·g_exp) @ K`（矩阵乘对角阵）
- R5：`k_cumdecay = (β·A'·(-γ)) @ K`（逐元素乘 γ 后矩阵乘 I）
- `A·diag(d) ≠ diag(d')·A`（除非 A 是对角阵）

**R7 错误：** 假设 `v_new·exp(g[-1])` 等价于 `v_new·(g_exp[-1]/g_exp)`。
- 正确：`state_new += v_new^T @ (K · exp(g[-1]-g_cum)[...,None])`（每行不同缩放）
- R7：`state_new += (v_new·exp(g[-1]))^T @ K`（标量缩放，每行相同）
- `exp(g[-1]-g_cum[j])` 是逐行向量，`exp(g[-1])` 是标量，二者不等价

**R6 bug：** g 在 GM 中布局为 `(T, Nv)`，head `nvId` 的时间序列不连续。
- 正确（Stage1 `GCopyInWithStride`）：`DataCopyInFp32WithStride(validLen, 1, src, nv_)`，srcStride=(nv-1)×4 字节
- R6（Stage2/3）：`DataCopyExtParams{1, curChunkSize×4, 0, 0, 0}`，blockCount=1，srcStride=0 → 连续读取 = 跨 head 垃圾数据

---

## 1 R2 改动点详解

### 1.1 动机

重构前 AIV 通过 `QKPreProcess` 将 Q/K 从 strided GM 搬入连续 workspace
（`queryConGm_`/`keyConGm_`），AIC 再从 workspace 读取做 matmul。
这额外消耗 `2 × coreNum × P × C × Dk × sizeof(bf16_t)` 的 workspace（典型 ~3MB）。

### 1.2 方案

AIC matmul 直接从 strided GM 读取 Q/K，通过 `SetOrgShape(M, N, K, Ka, Kb)` 指定行 stride。
AIV 侧删除 `QKPreProcess` 函数及 `queryConGm_`/`keyConGm_` workspace 分配。

### 1.3 SetOrgShape 用法

```cpp
// strided GM matmul: 用 5 参数 SetOrgShape, orgKc=N (匹配 3 参数行为)
mmRef.SetOrgShape(M, N, Ka, Kb, N);

// 连续 workspace matmul: 用 3 参数 SetOrgShape (Ka=Kb=K, orgKc=N)
mmRef.SetOrgShape(M, N, K);
```

`AICProcess` 根据 `MatmulShapeParams.ka` 是否 >0 自动选择：
- `ka > 0`：strided GM，用 5 参数 `SetOrgShape(m, n, ka, kb, n)`
- `ka == 0`：连续矩阵，用 3 参数 `SetOrgShape(m, n, k)`（保持原始行为）

### 1.4 !gOptional 路径 kg_ 写入

重构前 `QKPreProcess` 在 `!gOptional` 时将 K 拷贝到 `kg_` GM（此时 `kg = K`）。
R2 删除 `QKPreProcess` 后，此逻辑移至 `GBKCompute` 的 `else` 分支：
从 `kUbFloatCon_`（已从 strided GM 读入 UB 的 K）Cast 到 BF16 写出 `kg_`。

---

## 2 GM Workspace 布局对比

### 2.1 重构前

```
offset 0:  gCum_         FP32  [Nv][Sp]
offset 1:  kCumDecay_    BF16  [Nv][Sp][Dk]
offset 2:  vInner_       FP32/BF16 [Nv][Sp][Dv]
offset 3:  vInnerBf16_   BF16  [Nv][Sp][Dv]            (FP32 path)
offset 4:  qPrime_       BF16  [Nv][Sp][Dk]
offset 5:  attnInter_    BF16  [Nv][Sp][Dv]
offset 6:  kg_           BF16  [Nv][Sp][Dk]
offset 7:  qkt_          BF16  [Nv][Sp][C]
offset 8:  curStateBf16_ BF16  [Nv][Dv][Dk]            (FP32 path)
offset 9:  stageOneMask_ FP32  [coreNum*2][C][C]
offset 10: stageThreeMask_ FP32 [coreNum*2][C][C]
offset 11: stageWsAddr_  → gBKWs/queryCon/keyCon/kkWs/attnWs/vBetaWs (6 个 WS buffer)
```

### 2.2 重构后

```
offset 0:  gCum_         FP32  [Nv][Sp]                ← 不变
offset 1:  kCumDecay_    BF16  [Nv][Sp][Dk]            ← 不变
offset 2:  vInner_       FP32/BF16 [Nv][Sp][Dv]        ← 不变
offset 3:  vInnerBf16_   BF16  [Nv][Sp][Dv]            (FP32 path, 不变)
offset 4:  qPrime_       BF16  [Nv][Sp][Dk]            ← 不变
offset 5:  attnInter_    BF16  [Nv][Sp][Dv]            ← 不变
offset 6:  kg_           BF16  [Nv][Sp][Dk]            ← 不变
offset 7:  qkt_          BF16  [Nv][Sp][C]             ← 不变
offset 8:  curStateBf16_ BF16  [Nv][Dv][Dk]            (FP32 path, 不变)
offset 9:  stageOneMask_ FP32  [coreNum*2][C][C]       ← 不变
offset 10: stageThreeMask_ FP32 [coreNum*2][C][C]      ← 不变
offset 11: stageWsAddr_  → gBKWs/kkWs/attnWs/vBetaWs (4 个 WS buffer)
```

**GM workspace 削减（仅 Stage1 WS）：**
- 删除 `queryConGm_` (BF16 × coreNum × P × C × Dk) — R2
- 删除 `keyConGm_` (BF16 × coreNum × P × C × Dk) — R2

**stageWorkspaceSz 变化：**
- 重构前：`sizeLow * C * (2*C + 3*Dk + Dv) * P * coreNum`（6 个 WS buffer）
- 重构后：`sizeLow * C * (2*C + 1*Dk + Dv) * P * coreNum`（4 个 WS buffer）
- `STAGE_ONE_THREE`: 3 → 1

---

## 3 Stage1 重构后数据流伪代码

### 3.1 AIV 侧 (ParaChunkAIV)

```
// ============ 步骤 1: g_cum + gamma (不变) ============
if gOptional:
    for i in [0, curParaNum):
        // g (T,Nv) strided → Gather → UB(C) → CumSum → gCumSumUB(C)
        // Exp → gCumExpUB(C)
        // DataCopyPad → GM gCum_ (供 Stage2/3 读)
        GCumExpCompute(gGm_[bgOffset[i]], gCumGm_[chunkRowBase[i]],
                       gCumSumUB[i*C], gCumExpUB[i*C], validLen[i])
        // gamma = exp(g_cum[:,None] - g_cum[None,:]) * tril_mask → gammaUB(C,C)
        GammaCompute(gBroadUB, gammaUB, gCumSumUB[i*C])

// R2: 不再调用 QKPreProcess, AIC 直接从 strided GM 读取 Q/K
// R2: 不再 SetFlag(0x9) (sync0 删除, AIC 无需等待 Q/K 连续化)

// ============ 步骤 2: beta 搬入 (不变) ============
for i in [0, curParaNum):
    BetaCopyInWithStride(betaGm_[bgOffset[i]], betaUB[i*H], subValidLen[i])

// ============ 步骤 3: KKBetaCompute + 求逆 (不变) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x8)                                  // ← AIC: KKT 就绪
    // attn = kk * beta (含 beta)
    KKBetaCompute(kkWs_[i*C*C], betaUB[i*H])
    // 求逆: A = (I + kk*beta*gamma)^{-1}  → 写入 attnWs (GM)
    InverseCompute(attnWs_[i*C*C], gammaUB)
    CrossCoreSetFlag<PIPE_MTE3>(0x7)                        // → AIC: 求逆完成

// ============ 步骤 4: GBKCompute (R2: K 从 strided GM 读) ============
for i in [0, curParaNum):
    // kg = key * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
    // k_cumdecay = -1.0 * k * beta * g_cum_exp
    outKgGm_ = kgGm_[chunkRowBase[i]*Dk]
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    // R2: K 从 strided GM 读取 (不再从 keyConGm_)
    GBKCompute(gBKWs_[i*C*Dk], outKgGm_, betaUB[i*H],
               gCumSumUB[i*C], gCumExpUB[i*C],
               keyGm_[qk_base + subOffset*qkStride], qkStride)
    CrossCoreSetFlag<PIPE_MTE3>(0x6)                        // → AIC: gBKWs 就绪

// ============ 步骤 5: VBetaCompute (不变) ============
for i in [0, curParaNum):
    vOffset = chunkStartRow[i] * (Nv*Dv) + nId[i]*Dv
    VBetaCompute(valueGm_[vOffset], vBetaWs_[i*C*Dv], betaUB[...], ...)
    CrossCoreSetFlag<PIPE_MTE3>(0x5)                        // → AIC: vBetaWs 就绪

// ============ 步骤 6: q_prime (R2: Q 从 strided GM 读) ============
for i in [0, curParaNum):
    // q_prime = query * scale * g_cum_exp[:,None]
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    // R2: Q 从 strided GM 读取 (不再从 queryConGm_)
    QPrimeCompute(qPrimeGm, gCumSumUB[i*C], gCumExpUB[i*C],
                  queryGm_[qk_base + subOffset*qkStride], subValidLen[i], qkStride)

CrossCoreWaitFlag(0xA)                                      // ← AIC: QKT 完成
```

### 3.2 AIC 侧 (ParaChunkAIC)

```
// ============ matmul 1: KKT = K @ K^T (R2: 直接读 strided GM) ============
for i in [0, curParaNum):
    // R2: 不再 WaitFlag(0x9) (sync0 删除)
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    // A=keyGm(C,Dk) stride=Nk*Dk, B=keyGm(C,Dk) transB stride=Nk*Dk
    AICProcess(mmBf16_, keyGm_[qk_base], keyGm_[qk_base], kkWs[i*C*C],
               {M=C, N=C, K=Dk, sM=C, sN=C, sK=Dk, Ka=Nk*Dk, Kb=Nk*Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0x8)                         // → AIV: KKT 完成

// ============ matmul 2: QKT = Q @ K^T (R2: 直接读 strided GM) ============
for i in [0, curParaNum):
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    AICProcess(mmBf16_, queryGm_[qk_base], keyGm_[qk_base], qktGm[...],
               {M=validLen[i], N=validLen[i], K=Dk,
                sM=validLen[i], sN=validLen[i], sK=Dk,
                Ka=Nk*Dk, Kb=Nk*Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0xA)                         // → AIV: QKT 完成

// ============ matmul 3: 求逆辅助 (不变, 读 attnWs) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x7)                                  // ← AIV: 求逆完成
    AttnInverseMMCompute(i * C*C)

// ============ matmul 4: k_cumdecay = A @ gBKWs (不变, 读连续 WS) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x6)                                  // ← AIV: GBKCompute 完成
    AICProcess(mmBf16_, attnWs_[i*C*C], gBKWs_[i*C*Dk], kCumDecayGm[...],
               {M=C, N=Dk, K=C, sM=C, sN=Dk, sK=C})        // Ka=0, Kb=0 → 3 参数 SetOrgShape

// ============ matmul 5: vInner = A @ vBeta (不变, 读连续 WS) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x5)                                  // ← AIV: VBetaCompute 完成
    AICProcess(mmVInner_, attnWs_[i*C*C], vBetaWs_[i*C*Dv], vInnerGm[...],
               {M=C, N=Dv, K=C, sM=C, sN=Dv, sK=C})        // Ka=0, Kb=0 → 3 参数 SetOrgShape
```

### 3.3 Stage1 CrossCore 同步关系图（重构后）

```
AIV                                       AIC
────                                       ────
[R2: 不再 QKPreProcess, 不再 SetFlag(0x9)]
GCumExpCompute + GammaCompute (→ gCum_ GM)
beta 搬入
                                            KKT matmul (strided GM → kkWs)  ← R2
                                            SetFlag(0x8) ──────────→ WaitFlag(0x8)
KKBetaCompute (读 kkWs, 含 beta)
InverseCompute (求逆 A → attnWs)
SetFlag(0x7) ─────────────────────────→   WaitFlag(0x7)
                                            AttnInverseMMCompute
GBKCompute (K 从 strided GM 读 → gBKWs)    ← R2
SetFlag(0x6) ─────────────────────────→   WaitFlag(0x6)
                                            k_cumdecay = A @ gBKWs
VBetaCompute (→ vBetaWs)
SetFlag(0x5) ─────────────────────────→   WaitFlag(0x5)
                                            vInner = A @ vBetaWs
                                            QKT matmul (strided GM → qkt_)  ← R2
                                            SetFlag(0xA) ──────────→ WaitFlag(0xA)
QPrimeCompute (Q 从 strided GM 读)         ← R2
```

**与重构前同步关系变化：**
- 删除 sync0 (`0x9`)：AIC 不再等待 AIV 的 QKPreProcess，直接从 strided GM 读 Q/K

### 3.4 Stage1 输出（与重构前完全一致）

| 输出 | 公式 | 变化 |
|:-----|:-----|:-----|
| gCum_ | `g.cumsum()` FP32 Nv×Sp | 不变 |
| kCumDecay_ | `A@(-k*beta*g_exp)` BF16 Nv×Sp×Dk | 不变 |
| vInner_ | `A@(v*beta)` FP32/BF16 Nv×Sp×Dv | 不变 |
| qPrime_ | `q*scale*g_exp` BF16 Nv×Sp×Dk | 不变 (R2: Q 来源从 queryConGm 改为 strided GM) |
| kg_ | `k*exp(g[-1]-g_cum)` BF16 Nv×Sp×Dk | 不变 |
| qkt_ | `Q@K^T` BF16 Nv×Sp×C | 不变 (R2: Q/K 来源从 queryConGm/keyConGm 改为 strided GM) |

---

## 4 Stage2 数据流（与重构前完全一致）

R3-R7 全部回退，Stage2 无任何改动。详见重构前文档 §4。

### 4.1 核心计算公式（正确，与参考实现一致）

```
v_prime = k_cumdecay @ state^T              // k_cumdecay 含 g_exp
v_new = v_inner - v_prime                    // v_inner = A@(V*beta)
attn_inter = q_prime @ state^T               // q_prime 含 g_exp
core_attn_out = attn_inter + attn @ v_new

state_new = state * exp(g[-1]) + v_new^T @ kg
           // kg = K * exp(g[-1]-g_cum)[...,None]
```

### 4.2 同步关系（与重构前一致）

```
AIV                                            AIC
────                                            ────
state CopyIn + Cast
CalGCumExpFp32/Bf16 (从 GM gCum_ 读)
SetFlag(0x3) [FP32 only]
                                                WaitFlag(0x3) [FP32 only]
                                                CalVPrime (vInner += k_cumdecay@state^T)
                                                CalAttnInter (qPrime@state^T → out)
                                                SetFlag(0x2) ─────────→ WaitFlag(0x2)
CopyOutState
[FP32: CastGmToGm (vInner FP32→BF16)]
SetFlag(0x5)                                   WaitFlag(0x5)
                                                CalStateNew (state += v_new^T @ kg)
WaitFlag(0x4)                                  SetFlag(0x4) ─────────→ WaitFlag(0x4)
[下一 chunk]
```

---

## 5 Stage3 数据流（与重构前完全一致）

R6 回退，Stage3 无任何改动。decay_mask 从 GM `gCum_` 读取（Stage1 已写入）。

### 5.1 核心计算公式（正确）

```
decay_mask = exp(g_cum[:,None] - g_cum[None,:]) * tril    // 从 GM gCum_ 读
masked_qkt = qkt * scale * decay_mask * mask
attn_out = masked_qkt @ v_new (accum=1)
```

---

## 6 各级存储数据流总表（重构后）

### 6.1 GM 层

| 变量 | dtype | shape | 写入方 | 读取方 | 与重构前变化 |
|:-----|:------|:------|:-------|:-------|:-------------|
| query_/key_/value_/beta_ | BF16 | T×Nk×Dk / T×Nv×Dv | 用户输入 | Stage1 | R2: Stage1 直接读 strided GM |
| g_ | FP32 | T×Nv | 用户输入 | Stage1 | 不变 |
| gCum_ | FP32 | Nv×Sp | Stage1 | Stage2/3 | 不变 |
| kCumDecay_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | 不变 |
| vInner_ | FP32/BF16 | Nv×Sp×Dv | Stage1→Stage2 | Stage2→Stage3 | 不变 |
| vInnerBf16_ | BF16 | Nv×Sp×Dv | Stage2 | Stage2/3 | 不变 |
| qPrime_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | 不变 |
| kg_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | 不变 |
| qkt_ | BF16 | Nv×Sp×C | Stage1 | Stage3 | 不变 |
| stateOut_ | FP32/BF16 | Nv×Dv×Dk | Stage2 | Stage2 (递推) | 不变 |
| out_ | BF16 | T×Nv×Dv | Stage2→Stage3 | 用户输出 | 不变 |

### 6.2 Stage1 WS 层（重构后）

| 变量 | dtype | shape | 写入方 | 读取方 | 与重构前变化 |
|:-----|:------|:------|:-------|:-------|:-------------|
| ~~queryConGm_~~ | ~~BF16~~ | ~~P×C×Dk~~ | ~~AIV~~ | ~~AIC~~ | **R2 删除** |
| ~~keyConGm_~~ | ~~BF16~~ | ~~P×C×Dk~~ | ~~AIV~~ | ~~AIC~~ | **R2 删除** |
| gBKWsGm_ | BF16 | P×C×Dk | AIV GBKCompute | AIC matmul 4 | 不变 |
| kkWsGm_ | BF16 | P×C×C | AIC matmul 1 | AIV KKBetaCompute | 不变 |
| attnWsGm_ | BF16 | P×C×C | AIV InverseCompute | AIC matmul 3/5 | 不变 |
| vBetaWsGm_ | BF16 | P×C×Dv | AIV VBetaCompute | AIC matmul 5 | 不变 |

### 6.3 UB 层

无新增/变更。所有 UB buffer 与重构前一致。

---

## 7 重构前后计算公式对比

**结论：所有计算公式完全一致，R2 不改变任何数学。**

### 7.1 Stage1 核心计算

| 计算项 | 公式 | 重构前→后 |
|:-------|:-----|:----------|
| KKT | `(k@k^T) * beta * gamma` | 不变 |
| 求逆矩阵 M | `I - kk*beta*gamma` | 不变 |
| 求逆结果 A | `(I-kk*beta*gamma)^{-1}` | 不变 |
| v_inner | `A @ (v * beta)` | 不变 |
| k_cumdecay | `A @ (-k * beta * g_cum_exp)` | 不变 |
| kg | `k * exp(g[-1]-g_cum)[...,None]` | 不变 |
| q_prime | `q * scale * g_cum_exp[:,None]` | 不变 |
| QKT | `Q @ K^T` | 不变 |
| gCum | `g.cumsum()` → GM | 不变 |
| Q/K 读取 | 从 queryConGm_/keyConGm_ (WS) | **R2: 从 strided GM (SetOrgShape Ka/Kb)** |

### 7.2 Stage2 核心计算

| 计算项 | 公式 | 重构前→后 |
|:-------|:-----|:----------|
| g_cum_exp 来源 | 从 GM gCum_ 读 | 不变 |
| v_prime | `k_cumdecay @ state^T` | 不变 |
| v_new | `v_inner - v_prime` | 不变 |
| state 衰减 | `state *= exp(gCum[-1])` | 不变 |
| attn_inter | `q_prime @ state^T` | 不变 |
| state_new | `state*exp(g[-1]) + v_new^T @ kg` | 不变 |

### 7.3 Stage3 核心计算

| 计算项 | 公式 | 重构前→后 |
|:-------|:-----|:----------|
| decay_mask | 从 GM gCumExp 读 → broadcast | 不变 |
| masked_qkt | `qkt * scale * decay * mask` | 不变 |
| attn_out | `masked_qkt @ v_new` (accum=1) | 不变 |

---

## 8 数据流图（重构后）

```
原始输入: Q, K, V, beta, g, state, scale
                    │
    ┌───────────────┼───────────────────────────────┐
    │            Stage1 (per chunk, P 并行)           │
    │                                               │
    │  AIV:                                         │
    │    g → cumsum → g_cum_exp → gCum_ (GM)        │
    │    gamma = decay_mask (UB)                     │
    │    beta → UB                                   │
    │    [R2: 不再 QKPreProcess]                     │
    │    WaitFlag(0x8) ← AIC KKT                     │
    │    attn = kk * beta                            │
    │    A = (I - kk*beta*gamma)^{-1} → attnWs       │
    │    SetFlag(0x7) → AIC                          │
    │    GBKCompute: K 从 strided GM 读 → gBKWs      │  ← R2
    │    SetFlag(0x6) → AIC                          │
    │    VBetaCompute: V*beta → vBetaWs              │
    │    SetFlag(0x5) → AIC                          │
    │    q_prime = q*scale*g_exp (Q 从 strided GM)   │  ← R2
    │    kg = K*exp(g[-1]-g_cum) → kg_ (GM)          │
    │                                               │
    │  AIC:                                         │
    │    KKT = K@K^T (strided GM, Ka=Nk*Dk)         │  ← R2
    │    QKT = Q@K^T (strided GM)                    │  ← R2
    │    求逆辅助 matmul (读 attnWs)                  │
    │    k_cumdecay = A @ gBKWs                      │
    │    vInner = A @ vBetaWs                        │
    │                                               │
    │  输出: gCum_, kCumDecay_, vInner_, qPrime_,    │
    │        kg_, qkt_ (公式不变)                    │
    └───────────────┬───────────────────────────────┘
                    │
    ┌───────────────┼───────────────────────────────┐
    │            Stage2 (per chunk, 递推)             │
    │                                               │
    │  AIV:                                         │
    │    g_cum_exp (从 GM gCum_ 读)                  │
    │    state *= exp(g_cum[-1])                     │
    │    SetFlag(0x3) → AIC                          │
    │    WaitFlag(0x2) ← AIC                         │
    │    [FP32: CastGmToGm (vInner FP32→BF16)]      │
    │    SetFlag(0x5) → AIC                          │
    │                                               │
    │  AIC:                                         │
    │    v_prime = k_cumdecay @ state^T (accum=1)   │
    │    attn_inter = qPrime @ state^T               │
    │    state_new = state*exp(g[-1]) + v_new^T@kg  │
    └───────────────┬───────────────────────────────┘
                    │
    ┌───────────────┼───────────────────────────────┐
    │            Stage3 (per chunk, 并行)             │
    │                                               │
    │  AIV:                                         │
    │    decay_mask (从 GM gCum_ 读 → broadcast)     │
    │    masked_qkt = qkt * scale * decay * mask     │
    │                                               │
    │  AIC:                                         │
    │    attn_out = masked_qkt @ v_new (accum=1)     │
    └───────────────────────────────────────────────┘
```
