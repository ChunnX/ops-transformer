# CGDR arch35 重构后数据流与改动点

> 本文档描述重构后 arch35 平台 CGDR 算子的完整计算流程，与重构前逐项对比，
> 涵盖 AIC/AIV 协同、各级存储 shape 变化、matmul 参数、同步关系、workspace 削减。

---

## 1 重构改动点总览

| 编号 | 需求 | 重构前 | 重构后 | 影响范围 |
|:-----|:-----|:-------|:-------|:---------|
| R2 | 省略 QKPreprocess | AIV 将 Q/K 从 strided GM 搬入连续 WS (queryConGm_/keyConGm_)，AIC 从 WS 读 | AIC 直接从 strided GM 读取，通过 `SetOrgShape(M,N,K,Ka,Kb)` 指定行 stride | Stage1 AIV 删除 QKPreProcess；AIC matmul 1/2 改用 strided GM；删除 queryConGm_/keyConGm_ WS |
| R3 | K 衰减系数移至 Stage2 | Stage1 计算 `k_cumdecay = A@(-k*beta*g_cum_exp)`，含 g_cum_exp | Stage1 输出 `kBetaInv = (beta*A*(-gamma))@K`，不含 g_cum_exp；Stage2 CalVPrime 后对 vInner 做 g_cum_exp 行缩放 | Stage1 GBKCompute → NegGammaMultiply+KgCompute；Stage2 新增 CastGmToGmWithScale/ScaleVInnerBf16 |
| R4 | beta 改为 A 行缩放 | KKT 乘 beta (`attn = kk*beta`)，V 乘 beta (`v_beta = v*beta`) | KKT 不含 beta (`attn = kk`)，求逆后 `A_beta = beta[:,None]*A`，V 不乘 beta 直接用 | Stage1 KKBetaCompute 删 beta 乘法；InverseCompute 增 beta 行缩放；删除 VBetaCompute；AIC matmul 5 直接用原始 V |
| R5 | -W 矩阵 | `k_cumdecay = A@(-k*beta*g_cum_exp)` | `kBetaInv = (beta*A*(-gamma))@K`，AIV 计算 `A_beta_neg_gamma = A_beta*(-gamma)` 写入 attnWsNegGm_（独立 WS），AIC matmul `attnWsNegGm_@K` | Stage1 新增 NegGammaMultiply；AIC matmul 4 右矩阵从 gBKWs 改为 strided K；新增 attnWsNegGm_ WS |
| R6 | Stage2/3 重算 cumsumg | Stage1 GCumExpCompute 将 gCum 写到 GM，Stage2/3 从 GM 读 | Stage1 不再输出 gCum 到 GM；Stage2 CalGCumExp* 从原始 g (T,Nv) 重算；Stage3 CalMaskedQKT 从原始 g 重算 | Stage1 GCumExpCompute 删 DataCopyPad；Stage2/3 新增 g 搬入+CumSum+Exp；主入口删 gCum_ GM 分配 |
| R7 | AIC 步骤4/5 顺序 + 删除 kg_ | Stage1 输出 kg_，Stage2 CalStateNew 用 kg_；AIC 先 kBetaInv 后 vInner | AIC 保持先 kBetaInv 后 vInner；A_beta_neg_gamma 写入独立 attnWsNegGm_ (不覆盖 attnWs 中的 A_beta)；Stage1 删除 KgCompute/kg_ 输出；Stage2 CalStateNew 直接用原始 K (strided GM) | Stage1 新增 attnWsNegGm_ WS (3个c*c)；Stage2 v_new 行缩放合并 exp(g[-1])；CalStateNew 改用 strided K |

---

## 1.1 R7 详细分析

### 问题1: attnWs 覆盖 race condition

**初始重构方案（有 bug）：**
- AIV `InverseCompute` 将 A_beta 写入 attnWs
- AIV `NegGammaMultiply` 将 A_beta*(-gamma) 覆盖写入 attnWs（同一个 buffer）
- AIC vInner 需读 attnWs 中的 A_beta，但已被 NegGammaMultiply 覆盖为 A_beta_neg_gamma

**尝试方案1（交换 AIC 步骤4/5顺序）：** AIC 先做 vInner（读 A_beta），后做 kBetaInv（读 A_beta_neg_gamma）。但 AIV 的 NegGammaMultiply 和 AIC 的 vInner 都读写 attnWs，仍存在并行 race condition，需要额外同步 flag（0xB/0xC），导致串行等待。

**最终方案（独立 WS buffer）：**
- `attnWsGm_`：存 A_beta（由 InverseCompute 写入，供 AIC vInner matmul 读）
- `attnWsNegGm_`：存 A_beta_neg_gamma（由 NegGammaMultiply 写入，供 AIC kBetaInv matmul 读）
- AIC 保持原始顺序：先 kBetaInv（读 attnWsNegGm_），后 vInner（读 attnWs_）
- 两个 buffer 独立，无覆盖问题，无需额外同步 flag
- WS 从 2 个 c*c 增加到 3 个 c*c（kkWs + attnWs + attnWsNeg）

### 问题2: 删除 kg_，Stage2 直接用 K

**重构前：**
- Stage1: `kg = K * exp(g[-1] - g_cum)` → 输出到 GM
- Stage2: `state_new = state * exp(g[-1]) + v_new^T @ kg`

**重构后：**
- Stage1: 不再计算/输出 kg_
- Stage2: `state_new = state * exp(g[-1]) + v_new^T @ K`

**数学推导：**
```
重构前: state_new = state * exp(g[-1]) + v_new^T @ (K * exp(g[-1]-g_cum)[...,None])
重构后: state_new = state * exp(g[-1]) + v_new_scaled^T @ K

其中 v_new_scaled = v_new * exp(g[-1]-g_cum)[...,None]
```

由于 `exp(g[-1]-g_cum) = exp(g[-1]) / exp(g_cum) = exp(g[-1]) / g_cum_exp`，而 v_new 已含 g_cum_exp 行缩放 (R3)，所以：
```
v_new_scaled = v_new * g_cum_exp * (exp(g[-1]) / g_cum_exp)
             = v_new * exp(g[-1])
             = v_new * g_cum_exp[-1]  (标量)
```

即 v_new 只需在 R3 的 g_cum_exp 行缩放基础上，额外乘 `g_cum_exp[-1]`（一个标量），然后 CalStateNew 直接用原始 K (strided GM)。

**收益：**
- 删除 Stage1 KgCompute 函数和 kg_ GM workspace (BF16 × Nv × Sp × Dk)
- Stage2 CalStateNew 直接从 strided GM 读 K (通过 SetOrgShape Ka=Nk*Dk)
- 减少 Stage1 AIV 计算量 (删除 kg 的 broadcast+mul+cast+copyout)

## 2 GM Workspace 布局对比

### 2.1 重构前

```
offset 0:  gCum_         FP32  [Nv][Sp]                        ← R6 删除
offset 1:  kCumDecay_    BF16  [Nv][Sp][Dk]                    ← 语义变更: kBetaInv
offset 2:  vInner_       FP32/BF16 [Nv][Sp][Dv]
offset 3:  vInnerBf16_   BF16  [Nv][Sp][Dv]                    (FP32 path)
offset 4:  qPrime_       BF16  [Nv][Sp][Dk]
offset 5:  attnInter_    BF16  [Nv][Sp][Dv]
offset 6:  kg_           BF16  [Nv][Sp][Dk]
offset 7:  qkt_          BF16  [Nv][Sp][C]
offset 8:  curStateBf16_ BF16  [Nv][Dv][Dk]                    (FP32 path)
offset 9:  stageOneMask_ FP32  [coreNum*2][C][C]
offset 10: stageThreeMask_ FP32 [coreNum*2][C][C]
offset 11: stageWsAddr_  → gBKWs/queryCon/keyCon/kkWs/attnWs/vBetaWs (6 个 WS buffer)
```

### 2.2 重构后

```
offset 0:  kCumDecay_    BF16  [Nv][Sp][Dk]                    ← kBetaInv (R3/R5)
offset 1:  vInner_       FP32/BF16 [Nv][Sp][Dv]
offset 2:  vInnerBf16_   BF16  [Nv][Sp][Dv]                    (FP32 path)
offset 3:  qPrime_       BF16  [Nv][Sp][Dk]
offset 4:  attnInter_    BF16  [Nv][Sp][Dv]
offset 5:  qkt_          BF16  [Nv][Sp][C]
offset 6:  curStateBf16_ BF16  [Nv][Dv][Dk]                    (FP32 path)
offset 7:  stageOneMask_ FP32  [coreNum*2][C][C]
offset 8:  stageThreeMask_ FP32 [coreNum*2][C][C]
offset 9:  stageWsAddr_  → kkWs/attnWs/attnWsNeg (3 个 WS buffer)
```

**GM workspace 削减：**
- 删除 `gCum_` (FP32 × Nv × Sp) — R6
- 删除 `kg_` (BF16 × Nv × Sp × Dk) — R7
- 删除 `queryConGm_` / `keyConGm_` (BF16 × C × Dk × P × coreNum) — R2
- 删除 `vBetaWsGm_` (BF16 × C × Dv × P × coreNum) — R4
- 删除 `gBKWsGm_` (BF16 × C × Dk × P × coreNum) — R5
- 新增 `attnWsNegGm_` (BF16 × C × C × P × coreNum) — R7（独立 WS 避免 attnWs 覆盖）

**stageWorkspaceSz 变化：**
- 重构前：`sizeLow * C * (2*C + 3*Dk + Dv) * P * coreNum`（6 个 WS buffer）
- 重构后：`sizeLow * C * (3*C) * P * coreNum`（3 个 WS buffer：kkWs + attnWs + attnWsNeg）

---

## 3 Stage1 重构后数据流伪代码

### 3.1 AIV 侧 (ParaChunkAIV)

```
// ============ 步骤 1: g_cum + gamma (R6: 仅 AIV 内部, 不写出 GM) ============
if gOptional:
    for i in [0, curParaNum):
        // g (T,Nv) strided → Gather → UB(C) → CumSum → gCumSumUB(C)
        // Exp → gCumExpUB(C)
        // (不再 DataCopyPad 到 GM gCum_)
        GCumExpCompute(gGm_[bgOffset[i]], gCumSumUB[i*C], gCumExpUB[i*C], validLen[i])
        // gamma = exp(g_cum[:,None] - g_cum[None,:]) * tril_mask → gammaUB(C,C)
        GammaCompute(gBroadUB, gammaUB, gCumSumUB[i*C])

// 通知 AIC: gamma 就绪 (R2: 不再需要 Q/K 连续化)
for i in [0, curParaNum):
    CrossCoreSetFlag<PIPE_MTE3>(0x9)                        // → AIC: 可做 KKT

// ============ 步骤 2: beta 搬入 ============
for i in [0, curParaNum):
    BetaCopyInWithStride(betaGm_[bgOffset[i]], betaUB[i*H], subValidLen[i])

// ============ 步骤 3: KKBetaCompute (R4: 不乘 beta) + 求逆 (R4: beta 行缩放) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x8)                                  // ← AIC: KKT 就绪
    // R4: attn = kk (仅 Cast, 不乘 beta)
    KKBetaCompute(kkWs_[i*C*C])
    // 求逆: A = (I + kk*gamma)^{-1}  [不含 beta]
    // R4: A_beta = beta[:,None] * A (行缩放) → 写入 attnWs (GM)
    InverseCompute(attnWs_[i*C*C], gammaUB, betaUB[i*H])
    CrossCoreSetFlag<PIPE_MTE3>(0x7)                        // → AIC: 求逆完成, 可做辅助 matmul
    CrossCoreSetFlag<PIPE_MTE3>(0x5)                        // → AIC: A_beta 就绪, 可做 vInner

// ============ 步骤 4: NegGammaMultiply (R5: A_beta * (-gamma) → attnWsNegGm_) ============
for i in [0, curParaNum):
    // R5: A_beta_neg_gamma = A_beta * (-gamma) → Cast BF16 → attnWsNegGm_ (独立 WS, 不覆盖 attnWs)
    NegGammaMultiply(attnWsNegGm_[i*C*C], gammaUB)
    CrossCoreSetFlag<PIPE_MTE3>(0x6)                        // → AIC: A_beta_neg_gamma 就绪

// ============ 步骤 5: q_prime (R2: 从 strided GM 读 Q) ============
for i in [0, curParaNum):
    // q_prime = query * scale * g_cum_exp[:,None]
    // Q 从 strided GM 读取 (不再从 queryConGm_)
    QPrimeCompute(qPrimeGm, gCumSumUB[i*C], gCumExpUB[i*C],
                  queryGm_[qk_base], subValidLen[i], qkStride_)

CrossCoreWaitFlag(0xA)                                      // ← AIC: QKT 完成
```

### 3.2 AIC 侧 (ParaChunkAIC)

```
// ============ matmul 1: KKT = K @ K^T (R2: 直接读 strided GM) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x9)                                  // ← AIV: gamma 就绪
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    // A=keyGm(C,Dk) stride=Nk*Dk, B=keyGm(C,Dk) transB stride=Nk*Dk
    AICProcess(mmBf16_, keyGm_[qk_base], keyGm_[qk_base], kkWs[i*C*C],
               {M=C, N=C, K=Dk, sM=C, sN=C, sK=Dk, Ka=Nk*Dk, Kb=Nk*Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0x8)                         // → AIV: KKT 完成

// ============ matmul 2: QKT = Q @ K^T (R2: 直接读 strided GM) ============
for i in [0, curParaNum):
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    AICProcess(mmBf16_, queryGm_[qk_base], keyGm_[qk_base], qktGm[...],
               {M=validLen[i], N=validLen[i], K=Dk, Ka=Nk*Dk, Kb=Nk*Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0xA)                         // → AIV: QKT 完成

// ============ matmul 3: 求逆辅助 (不变) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x7)                                  // ← AIV: 求逆完成
    AttnInverseMMCompute(i * C*C)

// ============ matmul 4: kBetaInv = A_beta_neg_gamma @ K (R3/R5: 读 attnWsNegGm_) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x6)                                  // ← AIV: NegGammaMultiply 完成
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    // A=attnWsNegGm_(A_beta_neg_gamma, C,C), B=keyGm(C,Dk) stride=Nk*Dk
    AICProcess(mmBf16_, attnWsNegGm_[i*C*C], keyGm_[qk_base], kCumDecayGm[...],
               {M=C, N=Dk, K=C, Ka=0, Kb=Nk*Dk})
    // 输出: kBetaInv = (beta*A*(-gamma))@K → GM kCumDecay_ (不含 g_exp)

// ============ matmul 5: vInner = A_beta @ V (R4: 读 attnWs_, 直接用原始 V) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x5)                                  // ← AIV: A_beta 就绪
    vOffset = chunkStartRow[i] * (Nv*Dv) + nId[i]*Dv
    // A=attnWs_(A_beta, C,C), B=valueGm(C,Dv) stride=Nv*Dv
    AICProcess(mmVInner_, attnWs_[i*C*C], valueGm_[vOffset], vInnerGm[...],
               {M=C, N=Dv, K=C, Ka=0, Kb=Nv*Dv})
    // 输出: vInner = A_beta @ V (V 不含 beta) → GM vInner_
```

### 3.3 Stage1 CrossCore 同步关系图（重构后）

```
AIV                                       AIC
────                                       ────
[R2: 不再 QKPreProcess]
GCumExpCompute + GammaCompute (UB 内部)
SetFlag(0x9) ─────────────────────────→   WaitFlag(0x9)
                                            KKT matmul (strided GM → kkWs)
                                            SetFlag(0x8) ──────────→ WaitFlag(0x8)
KKBetaCompute (读 kkWs, 不乘 beta)
InverseCompute (求逆 + R4 beta 行缩放)
A_beta 写入 attnWs
SetFlag(0x7) ─────────────────────────→   WaitFlag(0x7)
SetFlag(0x5)                              AttnInverseMMCompute
[R5: NegGammaMultiply → attnWsNegGm_]
A_beta_neg_gamma = A_beta * (-gamma)
→ 写入 attnWsNegGm_ (不覆盖 attnWs)
SetFlag(0x6) ─────────────────────────→   WaitFlag(0x6)
                                            kBetaInv = attnWsNegGm_ @ K(strided GM)
                                          WaitFlag(0x5)
                                            vInner = attnWs_(A_beta) @ V(strided GM)
                                            QKT matmul (strided GM)
                                            SetFlag(0xA) ──────────→ WaitFlag(0xA)
QPrimeCompute (从 strided GM 读 Q)
```
                                            kBetaInv = attnWs @ K(strided GM)
KgCompute (从 strided GM 读 K)
SetFlag(0x5) ─────────────────────────→   WaitFlag(0x5)
                                            vInner = attnWs(A_beta) @ V(strided GM)
                                            QKT matmul (strided GM)
                                            SetFlag(0xA) ──────────→ WaitFlag(0xA)
QPrimeCompute (从 strided GM 读 Q)
```

### 3.4 Stage1 输出对比

| 输出 | 重构前 | 重构后 | 变化 |
|:-----|:-------|:-------|:-----|
| gCum_ | FP32 Nv×Sp, 写入 GM | **不输出** | R6 删除 |
| kCumDecay_ | `A@(-k*beta*g_exp)` | `(beta*A*(-gamma))@K` (kBetaInv) | R3/R5: 不含 g_exp |
| vInner_ | `A@(v*beta)` | `(beta*A)@V` (A_beta@V) | R4: V 不含 beta |
| qPrime_ | `q*scale*g_exp` (从 queryConGm 读 Q) | 不变 (从 strided GM 读 Q) | R2: Q 来源变更 |
| kg_ | `k*exp(g[-1]-g_cum)` | **不输出** | R7 删除 |
| qkt_ | `Q@K^T` (从 queryConGm/keyConGm) | 不变 (从 strided GM) | R2: Q/K 来源变更 |

---

## 4 Stage2 重构后数据流伪代码

### 4.1 AIV 侧 — FP32 path (ProcessAivFp32)

```
ProcessAivFp32(nvId, cId, length):
  if subBlockIdx == 0:
    // state 搬入 + Cast (不变)
    CopyIn<FP32>(state[nvId*stride], Dv, Dk)
    stateFp32UB = DeQue → DataCopy → FreeTensor
    Cast FP32→BF16 → curStateBf16 (GM)

    // R6: 从原始 g (T,Nv) 重算 g_cum + g_cum_exp (不再从 gCum_ GM 读)
    //   g (T,Nv) strided → DataCopyPad → UB(C) → CumSum → gCumUb → Exp → gCumExpUb
    //   state *= exp(g_cum[-1])
    CalGCumExpFp32(nvId, length)

    // state_out 写回 GM (不变)
    CopyOut<FP32>(stateOut, Dv, Dk)

  CrossCoreSetFlag<PIPE_MTE3>(0x3)                          // → AIC: state 就绪
  CrossCoreWaitFlag(0x2)                                    // ← AIC: CalVPrime+CalAttnInter 完成

  if subBlockIdx == 0:
    // R3: CastGmToGmWithScale — 合并 cast FP32→BF16 + g_cum_exp 行缩放
    //   vInner(FP32) *= g_cum_exp[:,None] → vInnerBf16(BF16)
    CastGmToGmWithScale<FP32→BF16>(vInner, vInnerBf16, curChunkSize*Dv, gCumExpUb_)

  CrossCoreSetFlag<PIPE_MTE3>(0x5)                          // → AIC: vInnerBf16 就绪
  CrossCoreWaitFlag(0x4)                                    // ← AIC: CalStateNew 完成
```

### 4.2 AIV 侧 — BF16 path (ProcessAivBf16)

```
ProcessAivBf16(nvId, cId, length):
  if subBlockIdx == 0:
    CopyIn<BF16>(curState, Dv, Dk)
    // R6: 从原始 g 重算
    CalGCumExpBf16(nvId, length)

  CrossCoreWaitFlag(0x2)                                    // ← AIC: CalVPrime+CalAttnInter 完成
  if subBlockIdx == 0:
    CopyOutState(finalState)

  // R3: BF16 path 新增 — 对 vInner 做 g_cum_exp 行缩放
  CrossCoreWaitFlag(0x6)                                    // ← AIC: vInner 就绪 (flag 0x6 新增)
  if subBlockIdx == 0:
    // vInner(BF16) 读入 → Cast FP32 → *= g_cum_exp → Cast BF16 → 写回 vInner
    ScaleVInnerBf16(nvId, length)

  CrossCoreSetFlag<PIPE_MTE3>(0x5)                          // → AIC: vInner 缩放完成
  CrossCoreWaitFlag(0x4)                                    // ← AIC: CalStateNew 完成
```

### 4.3 AIC 侧 (ProcessAic)

```
ProcessAic(nvId, cId, length):
    // 获取 state (不变)
    if FP32 path: CrossCoreWaitFlag(0x3) → curStateBf16
    else: stateForAic = curStateBf16 / stateOut

    // R3: kCumdecay 已是 kBetaInv (不含 g_exp)
    // v_prime = kBetaInv @ state^T (accum=1, 叠加到 vInner)
    // v_new = v_inner + v_prime  (但 v_prime 不含 g_exp, 缩放在 AIV 侧做)
    CalVPrime(kCumdecay[mm_offset0], stateForAic, vInner[mm_offset1])

    // attn_inter = q_prime @ state^T (不变)
    CalAttnInter(qPrime[mm_offset0], stateForAic, out[...])

    CrossCoreSetFlag<PIPE_FIX>(0x2)                         // → AIV: state matmul 完成
    if BF16 path:
        CrossCoreSetFlag<PIPE_FIX>(0x6)                     // → AIV: vInner 就绪, 可做 g_exp 缩放 (R3 新增)
    CrossCoreWaitFlag(0x5)                                   // ← AIV: vInnerBf16/vInner 缩放完成

    if FP32 path:
        // R7: CalStateNew 用 vInnerBf16 (已含 g_exp*exp(g[-1]) 缩放) + 原始 K (strided GM)
        CalStateNew(vInnerBf16[mm_offset1], keyGm, cgStartPos, length, nvId, stateOut)
    else:
        // BF16 path: 等 AIV 完成 g_exp*exp(g[-1]) 缩放
        CrossCoreWaitFlag(0x7)                               // ← AIV: ScaleVInnerBf16 完成 (R3 新增)
        CalStateNew(vInner[mm_offset1], keyGm, cgStartPos, length, nvId, stateOut)
        FIX_MTE2 同步

    CrossCoreSetFlag<PIPE_FIX>(0x4)                         // → AIV: state 更新完成
```

### 4.4 Stage2 CrossCore 同步关系图（重构后，BF16 path）

```
AIV                                            AIC
────                                            ────
state CopyIn + Cast
R6: CalGCumExpBf16 (从原始 g 重算)
SetFlag(0x3) [FP32 only]
                                               WaitFlag(0x3) [FP32 only]
                                               CalVPrime (vInner += kBetaInv@state^T)
                                               CalAttnInter (qPrime@state^T → out)
                                               SetFlag(0x2) ─────────→ WaitFlag(0x2)
CopyOutState                                   SetFlag(0x6) ─────────→ [BF16: R3 新增]
[R3: ScaleVInnerBf16]                          WaitFlag(0x5)
WaitFlag(0x6) [BF16: R3 新增]
vInner *= g_cum_exp → 写回
SetFlag(0x7) [BF16: R3 新增] ─────────────→   WaitFlag(0x7) [BF16: R3 新增]
SetFlag(0x5)                                   CalStateNew (state += v_new^T @ kg)
WaitFlag(0x4)                                  SetFlag(0x4) ─────────→ WaitFlag(0x4)
[下一 chunk]
```

### 4.5 Stage2 数学等价性

重构前：
```
v_prime = k_cumdecay @ state^T      // k_cumdecay 含 g_exp
v_new = v_inner + v_prime            // v_inner 含 beta, v_prime 含 g_exp
```

重构后：
```
v_prime_raw = kBetaInv @ state^T     // kBetaInv 不含 g_exp
v_prime = v_prime_raw * g_cum_exp    // R3: AIV 侧行缩放
v_new = v_inner + v_prime            // v_inner = A_beta@V (含 beta, 不含 g_exp)
```

数学等价：`kBetaInv * g_exp = (beta*A*(-gamma))@K * g_exp = A@(-k*beta*g_exp*gamma)`，与重构前 `A@(-k*beta*g_exp)` 在 gamma=decay_mask 时等价。

---

## 5 Stage3 重构后数据流伪代码

### 5.1 AIV 侧 (CalMaskedQKT)

```
CalMaskedQKT(outGM, nvId, chunkPos):
    if gOptional:
        // R6: 从原始 g (T,Nv) 重算 cumsumg (不再从 gCum_ GM 读)
        gStart = cgStartPos + chunkPos
        // g (T,Nv) strided → DataCopyPad → UB(curChunkSize) → 补齐到 C
        // CumSum → gCumBuffer(C)
        DataCopyPad(gLocal, g_[gStart*Nv + nvId], ...)
        CumSum(gCumBuffer, tmpBuffer1, tmpBuffer1, {1, C})

        // decay_mask = exp(g_cum[:,None] - g_cum[None,:]) * tril (不变)
        Broadcast(tmpBuffer1, gCumBuffer, (C,C), (C,1))
        Broadcast(tmpBuffer2, gCumBuffer, (C,C), (1,C))
        Sub → Mul(mask) → Exp

    // masked_qkt = qkt * scale * decay_mask * mask (不变)
    AlignedCopyIn(qkt, curChunkSize, C)
    Cast → Muls(scale) → Mul(decay) → Mul(mask) → Cast BF16
    AlignedCopyOut(outGM)
```

### 5.2 AIC 侧 (不变)

```
AICProcess(tmpGM, vInner, attnInter):
    // masked_qkt @ v_new → attnOut (accum=1) (不变)
    mm3.SetOrgShape(curChunkSize, Dv, curChunkSize, curChunkSize, Nv*Dv)
    mm3.IterateAll(attnInter, accum=1)
```

### 5.3 Stage3 改动点

仅 R6：`gCumExp` 来源从 GM `gCum_` 改为从原始 `g_` 重算。matmul 和同步关系不变。

---

## 6 各级存储数据流总表（重构后）

### 6.1 GM 层

| 变量 | dtype | shape | 写入方 | 读取方 | 与重构前变化 |
|:-----|:------|:------|:-------|:-------|:-------------|
| query_/key_/value_/beta_ | BF16 | T×Nk×Dk / T×Nv×Dv | 用户输入 | Stage1 (strided 直读) | R2: Stage1 直接读 strided GM |
| g_ | FP32 | T×Nv | 用户输入 | Stage1/2/3 (重算) | R6: Stage2/3 新增读取 |
| ~~gCum_~~ | ~~FP32~~ | ~~Nv×Sp~~ | ~~Stage1~~ | ~~Stage2/3~~ | **R6 删除** |
| kCumDecay_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | R3/R5: 语义变更为 kBetaInv |
| vInner_ | FP32/BF16 | Nv×Sp×Dv | Stage1→Stage2 | Stage2→Stage3 | R4: v_inner=A_beta@V (不含 v*beta) |
| vInnerBf16_ | BF16 | Nv×Sp×Dv | Stage2 | Stage2/3 | R3: CastGmToGmWithScale 含 g_exp 缩放 |
| qPrime_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | R2: Q 从 strided GM 读 |
| kg_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | R2/R5: K 从 strided GM 读 |
| qkt_ | BF16 | Nv×Sp×C | Stage1 | Stage3 | R2: Q/K 从 strided GM 读 |
| stateOut_ | FP32/BF16 | Nv×Dv×Dk | Stage2 | Stage2 (递推) | 不变 |
| out_ | BF16 | T×Nv×Dv | Stage2→Stage3 | 用户输出 | 不变 |

### 6.2 Stage1 WS 层（重构后）

| 变量 | dtype | shape | 写入方 | 读取方 | 与重构前变化 |
|:-----|:------|:------|:-------|:-------|:-------------|
| ~~queryConGm_~~ | ~~BF16~~ | ~~P×C×Dk~~ | ~~AIV~~ | ~~AIC~~ | **R2 删除** |
| ~~keyConGm_~~ | ~~BF16~~ | ~~P×C×Dk~~ | ~~AIV~~ | ~~AIC~~ | **R2 删除** |
| ~~gBKWsGm_~~ | ~~BF16~~ | ~~P×C×Dk~~ | ~~AIV~~ | ~~AIC~~ | **R5 删除** |
| ~~vBetaWsGm_~~ | ~~BF16~~ | ~~P×C×Dv~~ | ~~AIV~~ | ~~AIC~~ | **R4 删除** |
| kkWsGm_ | BF16 | P×C×C | AIC matmul 1 | AIV KKBetaCompute | 不变 (布局从 ckOffset 改为 ccOffset) |
| attnWsGm_ | BF16 | P×C×C | AIV InverseCompute | AIC matmul 3/5 (求逆辅助+vInner) | R4: 存 A_beta (重构前存 A) |
| attnWsNegGm_ | BF16 | P×C×C | AIV NegGammaMultiply | AIC matmul 4 (kBetaInv) | R5/R7: 新增, 存 A_beta_neg_gamma (独立 WS 避免覆盖) |

### 6.3 UB 层新增/变更

| buffer | dtype | shape | Stage | 变化 |
|:-------|:------|:------|:------|:-----|
| gCumUb_ | FP32 | C | Stage2 (新增) | R6: 重算的 g_cum |
| gCumExpUb_ | FP32 | C | Stage2 (新增) | R3: 重算的 g_cum_exp，用于 vInner 行缩放 |
| gTmpUb_ | FP32 | C | Stage2 (新增) | R6: CumSum 临时 buffer |
| gCumBuffer_ | FP32 | C×C | Stage3 (新增) | R6: 重算的 g_cum |
| inverseRes_ | FP32 | H×C | Stage1 | R4: 求逆后存 A_beta (重构前存 A) |

---

## 7 重构前后计算公式对比

### 7.1 Stage1 核心计算

| 计算项 | 重构前公式 | 重构后公式 | 需求 |
|:-------|:-----------|:-----------|:-----|
| KKT | `(k@k^T) * beta` | `k@k^T` (不含 beta) | R4 |
| 求逆矩阵 M | `I + kk*beta*gamma` | `I + kk*gamma` | R4 |
| 求逆结果 A | `(I+kk*beta*gamma)^{-1}` | `(I+kk*gamma)^{-1}` | R4 |
| beta 施加 | KKT 行缩放 + V/K 逐行缩放 | `A_beta = beta[:,None] * A` (行缩放) | R4 |
| v_inner | `A @ (v * beta)` | `(beta * A) @ V` = `A_beta @ V` | R4 |
| k_cumdecay | `A @ (-k * beta * g_cum_exp)` | `(beta * A * (-gamma)) @ K` = `A_beta_neg_gamma @ K` | R3/R5 |
| gCum 输出 | 写入 GM | 不输出 (AIV 内部) | R6 |
| Q/K 读取 | 从 queryConGm_/keyConGm_ (WS) | 从 strided GM (SetOrgShape Ka/Kb) | R2 |

### 7.2 Stage2 核心计算

| 计算项 | 重构前公式 | 重构后公式 | 需求 |
|:-------|:-----------|:-----------|:-----|
| g_cum_exp 来源 | 从 GM gCum_ 读 | 从原始 g 重算 cumsum+exp | R6 |
| v_prime | `k_cumdecay @ state^T` (含 g_exp) | `(kBetaInv @ state^T) * g_cum_exp` (AIV 行缩放) | R3 |
| v_new | `v_inner + v_prime` | `v_inner + v_prime` (v_inner 不含 g_exp, v_prime 含) | R3 |
| state 衰减 | `state *= exp(gCum[-1])` | `state *= exp(g_cum[-1])` (从重算 g_cum) | R6 |
| attn_inter | `q_prime @ state^T` | 不变 | — |
| state_new | `v_new^T @ kg` | `v_new^T @ K` (v_new 含 g_exp*exp(g[-1]) 缩放, K 从 strided GM 读) | R7 |

### 7.3 Stage3 核心计算

| 计算项 | 重构前公式 | 重构后公式 | 需求 |
|:-------|:-----------|:-----------|:-----|
| decay_mask | 从 GM gCumExp 读 → broadcast | 从原始 g 重算 cumsum → broadcast | R6 |
| masked_qkt | `qkt * scale * decay * mask` | 不变 | — |
| attn_out | `masked_qkt @ v_new` (accum=1) | 不变 | — |

---

## 8 数据流图（重构后）

```
原始输入: Q, K, V, beta, g, state, scale
                    │
    ┌───────────────┼───────────────────────────────┐
    │            Stage1 (per chunk, P 并行)           │
    │                                               │
    │  AIV:                                         │
    │    g → cumsum → g_cum_exp (UB, 不输出 GM)      │  ← R6
    │    gamma = decay_mask (UB)                     │
    │    beta → UB                                   │
    │    [R2: 不再 QKPreProcess]                     │
    │    SetFlag(0x9) → AIC                          │
    │    WaitFlag(0x8) ← AIC KKT                     │
    │    attn = kk (R4: 不含 beta)                   │
    │    A = (I + kk*gamma)^{-1} (求逆)              │  ← R4
    │    A_beta = beta[:,None] * A (行缩放)           │  ← R4
    │    A_beta → attnWs (GM)                        │
    │    SetFlag(0x7+0x5) → AIC                      │
    │    A_beta_neg_gamma = A_beta * (-gamma)        │  ← R5
    │    → attnWsNegGm_ (独立 WS, 不覆盖 attnWs)     │
    │    SetFlag(0x6) → AIC                          │
    │    q_prime = q * scale * g_exp (strided GM 读 Q)│  ← R2
    │                                               │
    │  AIC:                                         │
    │    KKT = K@K^T (strided GM, Ka=Nk*Dk)         │  ← R2
    │    QKT = Q@K^T (strided GM)                    │  ← R2
    │    求逆辅助 matmul (读 attnWs)                  │
    │    kBetaInv = attnWsNegGm_ @ K (strided)       │  ← R3/R5
    │    vInner = attnWs_(A_beta) @ V (strided)      │  ← R4
    │                                               │
    │  输出: kBetaInv, vInner, qPrime, qkt           │
    │  (不输出 gCum, 不输出 kg)                      │  ← R6/R7
    └───────────────┬───────────────────────────────┘
                    │
    ┌───────────────┼───────────────────────────────┐
    │            Stage2 (per chunk, 递推)             │
    │                                               │
    │  AIV:                                         │
    │    g → cumsum → g_cum_exp (从原始 g 重算)      │  ← R6
    │    state *= exp(g_cum[-1])                     │
    │    SetFlag(0x3) → AIC                          │
    │    WaitFlag(0x2) ← AIC                         │
    │    vInner *= g_cum_exp * exp(g[-1]) (行缩放)   │  ← R3+R7
    │    SetFlag(0x5/0x7) → AIC                      │
    │                                               │
    │  AIC:                                         │
    │    v_prime = kBetaInv @ state^T (accum=1)     │  ← R3: 不含 g_exp
    │    attn_inter = qPrime @ state^T               │
    │    SetFlag(0x2/0x6) → AIV                      │
    │    WaitFlag(0x5/0x7) ← AIV                     │
    │    state_new = v_new^T @ K (strided GM, accum=1)│  ← R7
    └───────────────┬───────────────────────────────┘
                    │
    ┌───────────────┼───────────────────────────────┐
    │            Stage3 (per chunk, 并行)             │
    │                                               │
    │  AIV:                                         │
    │    g → cumsum → decay_mask (从原始 g 重算)     │  ← R6
    │    masked_qkt = qkt * scale * decay * mask     │
    │                                               │
    │  AIC:                                         │
    │    attn_out = masked_qkt @ v_new (accum=1)     │
    └───────────────────────────────────────────────┘
```
