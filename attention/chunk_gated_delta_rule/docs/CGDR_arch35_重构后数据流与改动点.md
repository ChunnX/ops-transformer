# CGDR arch35 重构后数据流与改动点

> 本文档描述 arch35 CGDR 算子重构方案。R2 已实现，R4/R5/R7 待实现。
> 数学等价性已用 Python 验证（`verify_r4_r5_r7.py`，误差 < 1e-12）。

---

## 0 优化点总览

| 编号 | 需求 | 状态 | 方案 |
|:-----|:-----|:-----|:-----|
| R2 | matmul 直接读 strided GM | **已实现** | 删除 QKPreProcess/queryConGm_/keyConGm_，SetOrgShape Ka/Kb |
| R4 | beta 移到 A 列缩放 | **待实现** | `A_beta = A @ diag(beta)`，删除 VBetaCompute/vBetaWsGm_ |
| R5 | g_exp 移到 A 列缩放 | **待实现** | `A_beta_gexp = A_beta @ diag(g_exp)`，删除 GBKCompute/gBKWsGm_ |
| R6 | gCum_ 输出 + 后续重算 | **已满足** | 原始代码已是此做法 |
| R7 | kg 系数移到 v_new | **待实现** | `v_new *= d[:,None]`，删除 KgCompute/kg_ |

### 正确公式（参考实现 `chunk_gated_delta_rule_golden.py:102-149`）

矩阵形式（`D=diag(beta)`, `G=diag(g_exp)`, `d=exp(g[-1]-g_cum)`）：
```
Stage1:
  KKT:        attn = -(D @ K @ K^T) * gamma      // beta 在 KKT 内部
  求逆:       A = (I + attn)^{-1}                 // beta 在求逆内部
  v_inner:    A @ D @ V
  k_cumdecay: A @ D @ G @ (-K)                    // kernel 用负号 + accum=1

Stage2:
  v_new = v_inner + k_cumdecay @ state            // accum=1 融合
  state_new = state * g_exp[-1] + v_new^T @ diag(d) @ K
```

### R4+R5 数学推导（两步列缩放）

利用结合律将对角阵从 K/V 左边移到 A 右边：
```
R4: A_beta = A @ D = A * beta[None,:]            // 第 j 列乘 beta[j]
    v_inner = A_beta @ V                          // V 不乘 beta

R5: A_beta_gexp = A_beta @ G = A_beta * g_exp[None,:]  // 第 j 列再乘 g_exp[j]
    k_cumdecay = -A_beta_gexp @ K                 // K 不乘 beta/g_exp

等价性: A @ D @ G = A @ diag(β·g_exp) = A_beta_gexp  (对角阵可合并)
```

### R7 数学推导（d 移到 v_new 行缩放）

```
原始: state_new += v_new^T @ diag(d) @ K
R7:   state_new += (v_new^T @ diag(d)) @ K = (v_new * d[:,None])^T @ K
```

`v^T @ diag(d) = (diag(d) @ v)^T = (v * d[:,None])^T`（v 的行缩放）。

---

## 1 Stage1 重构后计算流程（R2+R4+R5）

### 1.0 任务调度（不变）

```
Stage1::Process():
    totalChunk = Nv * Nc
    [start, end) = 核间均分 (former 核多 1 个 chunk)
    for taskId in [start, end, P):
        curParaNum = min(P, end - taskId)
        for i in [0, curParaNum): SetChunkOffset(i, ...)
        ProcessParaChunk(curParaNum)   // AIC/AIV 并行
```

### 1.1 AIV 侧

```
// ===== 步骤 1: g_cum + gamma (不变) =====
if gOptional:
    for i in [0, curParaNum):
        // g(T,Nv) strided → Gather → UB(C) → CumSum → gCumSumUB
        // Exp → gCumExpUB
        // DataCopyPad → GM gCum_ (供 Stage2/3 读)
        GCumExpCompute(gGm_[bgOffset[i]], gCumGm_[chunkRowBase[i]],
                       gCumSumUB[i*C], gCumExpUB[i*C], validLen[i])
        // gamma = exp(g_cum[:,None]-g_cum[None,:]) * tril → gammaUB(C,C)
        GammaCompute(gBroadUB, gammaUB, gCumSumUB[i*C])

// R2: 不再 QKPreProcess (AIC 直接读 strided GM), 不再 SetFlag(0x9)

// ===== 步骤 2: beta 搬入 (改为读全量 C 个, 供列缩放) =====
for i in [0, curParaNum):
    BetaCopyInWithStride(betaGm_[bgOffset[i]], betaFullUB[C], subValidLen[i])

// ===== 步骤 3: KKBetaCompute + 求逆 (不变) =====
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x8)                         // ← AIC: KKT 就绪
    // attn = kk * beta * gamma (含 beta, gamma)
    KKBetaCompute(kkWs_[i*C*C], betaUB[i*H])
    // A = (I - kk*beta*gamma)^{-1} → 写 attnWs (GM)
    InverseCompute(attnWs_[i*C*C], gammaUB)
    CrossCoreSetFlag<PIPE_MTE3>(0x7)               // → AIC: 求逆完成

// ===== 步骤 4: R4 列缩放 β → A_beta (供 matmul5 v_inner) =====
for i in [0, curParaNum):
    CrossCoreWaitFlag(0xB)                         // ← AIC: AttnInverseMMCompute 完成
    // 读 A from attnWs (GM, H×C BF16 → Cast FP32)
    DataCopyIn(attnWs_[i*C*C], H*C) → A_local(H,C)
    // R4: A_beta = A * beta[None,:] (列缩放, 第 j 列乘 beta[j])
    Broadcast(betaFullUB, (1,C)) → Mul(A_local, A_local, betaBroad, H*C)
    // 写 A_beta → attnWs (给 matmul5)
    Cast BF16 → DataCopyPad → attnWs_[i*C*C]
    CrossCoreSetFlag<PIPE_MTE3>(0x5)               // → AIC: A_beta 就绪

// ===== 步骤 5: R5 列缩放 g_exp → A_beta_gexp (供 matmul4 k_cumdecay) =====
for i in [0, curParaNum):
    // A_beta 仍在 A_local (UB, 未释放)
    // R5: A_beta_gexp = A_beta * g_exp[None,:] (列缩放, 第 j 列乘 g_exp[j])
    Broadcast(gCumExpUB, (1,C)) → Mul(A_beta_local, A_beta_local, gExpBroad, H*C)
    // 写 A_beta_gexp → attnWsNegGm_ (给 matmul4)
    Cast BF16 → DataCopyPad → attnWsNegGm_[i*C*C]
    CrossCoreSetFlag<PIPE_MTE3>(0x6)               // → AIC: A_beta_gexp 就绪

// ===== 步骤 6: q_prime (R2: Q 从 strided GM) =====
for i in [0, curParaNum):
    qk_base = chunkStartRow[i]*Nk*Dk + nId[i]*(Nk/Nv)*Dk
    QPrimeCompute(qPrimeGm, gCumSumUB[i*C], gCumExpUB[i*C],
                  queryGm_[qk_base + subOffset*qkStride], subValidLen[i], qkStride)

// ===== 步骤 7: KgCompute (保留, R7 待删除) =====
for i in [0, curParaNum):
    // kg = K * exp(g[-1]-g_cum)[...,None] → kg_ GM
    KgCompute(kgGm_[chunkRowBase[i]*Dk], gCumSumUB[i*C], keyGm_[qk_base], qkStride)

CrossCoreWaitFlag(0xA)                             // ← AIC: QKT 完成
```

### 1.2 AIC 侧

```
// ===== matmul 1: KKT (R2: strided GM) =====
for i in [0, curParaNum):
    qk_base = chunkStartRow[i]*Nk*Dk + nId[i]*(Nk/Nv)*Dk
    // A=keyGm(C,Dk) stride=Nk*Dk, B=keyGm(C,Dk) transB stride=Nk*Dk
    AICProcess(mmBf16_, keyGm_[qk_base], keyGm_[qk_base], kkWs[i*C*C],
               {M=C, N=C, K=Dk, sM=C, sN=C, sK=Dk, Ka=Nk*Dk, Kb=Nk*Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0x8)                // → AIV: KKT 完成

// ===== matmul 2: QKT (R2: strided GM) =====
for i in [0, curParaNum):
    qk_base = chunkStartRow[i]*Nk*Dk + nId[i]*(Nk/Nv)*Dk
    AICProcess(mmBf16_, queryGm_[qk_base], keyGm_[qk_base], qktGm[...],
               {M=validLen[i], N=validLen[i], K=Dk, sM=validLen[i], sN=validLen[i], sK=Dk,
                Ka=Nk*Dk, Kb=Nk*Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0xA)                // → AIV: QKT 完成

// ===== matmul 3: 求逆辅助 (不变) =====
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x7)                         // ← AIV: 求逆完成
    AttnInverseMMCompute(i*C*C)                    // 读/写 attnWs 的原始 A
    CrossCoreSetFlag<PIPE_FIX>(0xB)                // → AIV: 辅助 matmul 完成 (新增)

// ===== matmul 4: k_cumdecay (R4+R5: 读 A_beta_gexp, K 从 strided GM) =====
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x6)                         // ← AIV: A_beta_gexp 就绪
    qk_base = chunkStartRow[i]*Nk*Dk + nId[i]*(Nk/Nv)*Dk
    // A=attnWsNegGm(A_beta_gexp, C,C), B=keyGm(C,Dk) stride=Nk*Dk
    AICProcess(mmBf16_, attnWsNegGm_[i*C*C], keyGm_[qk_base], kCumDecayGm[...],
               {M=C, N=Dk, K=C, sM=C, sN=Dk, sK=C, Ka=0, Kb=Nk*Dk})
    // k_cumdecay = -A_beta_gexp @ K = A @ (-K·β·g_exp) ✓

// ===== matmul 5: vInner (R4: 读 A_beta, V 从 strided GM) =====
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x5)                         // ← AIV: A_beta 就绪
    vOffset = chunkStartRow[i]*(Nv*Dv) + nId[i]*Dv
    // A=attnWs(A_beta, C,C), B=valueGm(C,Dv) stride=Nv*Dv
    AICProcess(mmVInner_, attnWs_[i*C*C], valueGm_[vOffset], vInnerGm[...],
               {M=C, N=Dv, K=C, sM=C, sN=Dv, sK=C, Ka=0, Kb=Nv*Dv})
    // v_inner = A_beta @ V = A @ (V·β) ✓
```

### 1.3 Stage1 同步关系图

```
AIV                                       AIC
────                                       ────
[R2: 不再 QKPreProcess, 不再 SetFlag(0x9)]
GCumExpCompute + GammaCompute (→ gCum_ GM)
beta 搬入 (全量 C)
                                            KKT matmul (strided GM → kkWs)  ← R2
                                            SetFlag(0x8) ──────────→ WaitFlag(0x8)
KKBetaCompute (读 kkWs, 含 beta)
InverseCompute (求逆 A → attnWs)
SetFlag(0x7) ─────────────────────────→   WaitFlag(0x7)
                                            AttnInverseMMCompute (读/写 attnWs 的 A)
                                            SetFlag(0xB) ──────────→ WaitFlag(0xB)  // 新增
[R4: 列缩放 β → A_beta]
读 A → 列缩放 β → A_beta → 写 attnWs
SetFlag(0x5) ─────────────────────────→   WaitFlag(0x5) → matmul5: A_beta @ V (strided)
[R5: 列缩放 g_exp → A_beta_gexp]
A_beta (UB) → 列缩放 g_exp → A_beta_gexp → 写 attnWsNegGm_
SetFlag(0x6) ─────────────────────────→   WaitFlag(0x6) → matmul4: -A_beta_gexp @ K (strided)
KgCompute (→ kg_ GM)                      QKT matmul (strided GM → qkt_)  ← R2
                                          SetFlag(0xA) ──────────→ WaitFlag(0xA)
QPrimeCompute (Q 从 strided GM)           ← R2
```

### 1.4 Stage1 输出

| 输出 | 公式 | dtype/shape | 变化 |
|:-----|:-----|:------------|:-----|
| gCum_ | `g.cumsum()` | FP32 Nv×Sp | 不变 |
| kCumDecay_ | `-A_beta_gexp @ K` = `A@(-K·β·g_exp)` | BF16 Nv×Sp×Dk | R4+R5: K 从 strided GM |
| vInner_ | `A_beta @ V` = `A@(V·β)` | FP32/BF16 Nv×Sp×Dv | R4: V 从 strided GM |
| qPrime_ | `q*scale*g_exp` | BF16 Nv×Sp×Dk | R2: Q 从 strided GM |
| kg_ | `K*exp(g[-1]-g_cum)` | BF16 Nv×Sp×Dk | 不变 (R7 待删除) |
| qkt_ | `Q@K^T` | BF16 Nv×Sp×C | R2: Q/K 从 strided GM |

### 1.5 Stage1 WS 布局

```
kkWsGm_      BF16 [P×C×C]     // KKT 结果 (不变)
attnWsGm_    BF16 [P×C×C]     // A_beta (R4 后; 重构前存 A, 重构后存 A_beta)
attnWsNegGm_ BF16 [P×C×C]     // A_beta_gexp (R5 新增, 供 matmul4)
// 删除: queryConGm_/keyConGm_ (R2), vBetaWsGm_ (R4), gBKWsGm_ (R5)
```

---

## 2 Stage2 重构后计算流程（R7）

### 2.0 任务调度（不变）

```
Stage2::Process():
    nvPerCore = Ceil(Nv, coreNum)
    for nvId in [nvStart, nvEnd):
        for cId in [0, Nc):
            length = cId * C
            curChunkSize = (cId == Nc-1) ? seqLen%C : C
            AIV: ProcessAivFp32 / ProcessAivBf16
            AIC: ProcessAic
```

### 2.1 AIV 侧 — FP32 path

```
ProcessAivFp32(nvId, cId, length):
  if subBlockIdx == 0:
    // state 搬入 + Cast (不变)
    CopyIn<FP32>(state[nvId*stride], Dv, Dk)
    stateFp32UB = DeQue → DataCopy → FreeTensor
    Cast FP32→BF16 → curStateBf16 (GM)

    // state *= exp(g_cum[-1]) (不变, 从 gCum_ 读)
    CalGCumExpFp32(gCum_[nvId*Sp + length])

    // state_out 写回 GM (不变)
    CopyOut<FP32>(stateOut, Dv, Dk)

  CrossCoreSetFlag<PIPE_MTE3>(0x3)                 // → AIC: state 就绪
  CrossCoreWaitFlag(0x2)                           // ← AIC: CalVPrime+CalAttnInter

  if subBlockIdx == 0:
    // R7: vInner FP32→BF16 + d 行缩放
    //   d = exp(gCum[-1] - gCum[0:curChunkSize])    // 从 gCum_ 读
    //   v_new_scaled = vInner * d[:,None]
    //   CastGmToGmWithScale: 合并 cast + 行缩放
    mm_offset1 = nvId*Sp*Dv + length*Dv
    CastGmToGmWithScale<FP32→BF16>(vInner[mm_offset1], vInnerBf16[mm_offset1],
                                    curChunkSize*Dv, d)
    // vInnerBf16 此时存 v_new_scaled (含 d 行缩放)

  CrossCoreSetFlag<PIPE_MTE3>(0x5)                 // → AIC: v_new_scaled 就绪
  CrossCoreWaitFlag(0x4)                           // ← AIC: CalStateNew 完成
```

### 2.2 AIV 侧 — BF16 path

```
ProcessAivBf16(nvId, cId, length):
  curState = (cId==0) ? curStateBf16[stride] : stateOut[nvId*Dv*Dk]
  finalState = stateOut[nvId*Dv*Dk]

  if subBlockIdx == 0:
    CopyIn<BF16>(curState, Dv, Dk)
    CalGCumExpBf16(gCum_[nvId*Sp + length])         // state *= exp(g[-1])

  CrossCoreWaitFlag(0x2)                           // ← AIC: CalVPrime+CalAttnInter

  if subBlockIdx == 0:
    CopyOutState(finalState)

    // R7: vInner 行缩放 (in-place)
    //   d = exp(gCum[-1] - gCum[0:curChunkSize])
    //   vInner *= d[:,None]
    ScaleVInnerBf16(nvId, length, d)

  CrossCoreSetFlag<PIPE_MTE3>(0x5)                 // → AIC: v_new_scaled 就绪
  CrossCoreWaitFlag(0x4)                           // ← AIC: CalStateNew
```

### 2.3 AIC 侧

```
ProcessAic(nvId, cId, length):
    mm_offset0 = nvId*Sp*Dk + length*Dk
    mm_offset1 = nvId*Sp*Dv + length*Dv
    stateOut = stateOut[nvId*Dv*Dk]

    // 获取 state (不变)
    if FP32 path:
        CrossCoreWaitFlag(0x3)                     // ← AIV: curStateBf16 就绪
        stateForAic = curStateBf16[nvId*Dv*Dk]
    else:
        stateForAic = (cId==0) ? curStateBf16[stride] : stateOut

    // matmul 1: v_prime (accum=1, 不变)
    //   vInner += k_cumdecay @ state^T → vInner = v_new
    CalVPrime(kCumdecay[mm_offset0], stateForAic, vInner[mm_offset1])

    // matmul 2: attn_inter (不变)
    CalAttnInter(qPrime[mm_offset0], stateForAic, out[...])

    CrossCoreSetFlag<PIPE_FIX>(0x2)                // → AIV: state matmul 完成
    CrossCoreWaitFlag(0x5)                         // ← AIV: v_new_scaled 就绪

    // matmul 3: state_new (R7: 用 strided K, 不用 kg_)
    //   state += v_new_scaled^T @ K (accum=1)
    //   K 从原始 GM (T,Nk,Dk) strided 读, SetOrgShape Ka=Nk*Dk
    kStart = cg.startPos + length
    kOffset = kStart * Nk*Dk + nvId * (Nk/Nv)*Dk
    CalStateNew(vInnerBf16[mm_offset1], keyGm_[kOffset], stateOut, kStride=Nk*Dk)
    // 等价: state_new = state*exp(g[-1]) + (v_new*d)^T @ K ✓

    CrossCoreSetFlag<PIPE_FIX>(0x4)                // → AIV: state 更新完成
```

### 2.4 Stage2 同步关系图

```
AIV                                            AIC
────                                            ────
state CopyIn + Cast → curStateBf16
state *= exp(g[-1]) (从 gCum_ 读)
state_out 写回 GM
SetFlag(0x3) →                                WaitFlag(0x3)
                                                CalVPrime (vInner += kcd@state^T)
                                                CalAttnInter (qPrime@state^T → out)
                                                SetFlag(0x2) ─────────→ WaitFlag(0x2)
[R7: v_new *= d[:,None] 行缩放]
vInner FP32→BF16 + d 缩放 / vInner in-place 缩放
SetFlag(0x5) →                                WaitFlag(0x5)
                                                CalStateNew (state += v_new_scaled^T @ K_strided)
                                                SetFlag(0x4) ─────────→ WaitFlag(0x4)
[下一 chunk]
```

### 2.5 Stage2 输出

| 输出 | 公式 | 变化 |
|:-----|:-----|:-----|
| vInner_ | `v_inner + k_cumdecay@state` = `v_new` | 不变 (accum=1) |
| vInnerBf16_ | `v_new * d[:,None]` (含 R7 行缩放) | R7: 新增 d 行缩放 |
| stateOut_ | `state*exp(g[-1]) + v_new_scaled^T@K` | R7: K 从 strided GM (不用 kg_) |
| out_ | `q_prime @ state^T` | 不变 |

---

## 3 Stage3 计算流程（不变）

R6 已满足，Stage3 无改动。decay_mask 从 GM `gCum_` 读取（Stage1 已写入）。

### 3.1 AIV 侧

```
CalMaskedQKT(outGM, nvId, chunkPos):
    if gOptional:
        // 从 GM gCum_ 读 → broadcast → decay_mask
        AlignedCopyIn(gCum_[nvId*Sp + chunkPos], 1, curChunkSize)
        g_cum = DeQue
        Broadcast(tmpBuffer1, g_cum, (C,C), (C,1))
        Broadcast(tmpBuffer2, g_cum, (C,C), (1,C))
        Sub → Mul(mask) → Exp
    else:
        Duplicate(tmpBuffer1, 1.0, curChunkSize*C)

    // masked_qkt = qkt * scale * decay * mask
    AlignedCopyIn(qkt, curChunkSize, C)
    Cast → Muls(scale) → Mul(decay) → Mul(mask) → Cast BF16
    AlignedCopyOut(outGM)
```

### 3.2 AIC 侧

```
AICProcess(tmpGM, vInner, attnInter):
    // attn_out = masked_qkt @ v_new (accum=1)
    mm3.SetOrgShape(curChunkSize, Dv, curChunkSize, curChunkSize, Nv*Dv)
    mm3.SetSingleShape(curChunkSize, Dv, curChunkSize)
    mm3.SetTensorA(tmpGM)           // masked_qkt
    mm3.SetTensorB(vInner)          // v_new
    mm3.IterateAll(attnInter, 1)    // attn_out = attn_inter + masked_qkt @ v_new
    mm3.End()
```

---

## 4 Python 验证

```python
# verify_r4_r5_r7.py — C=8, Dk=16, Dv=16, float64
# R4: A_beta = A * beta[None,:]            (列缩放)
# R5: A_beta_gexp = A_beta * g_exp[None,:]  (第二步列缩放)
# R7: state_new += (v_new * d[:,None])^T @ K

# 结果:
#   v_inner diff:    6.94e-18
#   k_cumdecay diff: 1.39e-17
#   v_new diff:      1.39e-17
#   state_new diff:  1.39e-17
#   PASS (tol=1e-12)
```

---

## 5 GM Workspace 布局对比

### 重构前

```
gCum_ kCumDecay_ vInner_ vInnerBf16_ qPrime_ attnInter_ kg_ qkt_ curStateBf16_ masks stageWs
stageWs: gBKWs/queryCon/keyCon/kkWs/attnWs/vBetaWs (6 buffer)
```

### 重构后（R2+R4+R5+R7）

```
gCum_ kCumDecay_ vInner_ vInnerBf16_ qPrime_ attnInter_ qkt_ curStateBf16_ masks stageWs
stageWs: kkWs/attnWs/attnWsNeg (3 buffer)
// 删除: kg_ (R7), queryCon/keyCon (R2), vBetaWs (R4), gBKWs (R5)
// 新增: attnWsNeg (R5, C×C)
```

### WS 削减汇总

| 删除项 | 大小 | 来源 |
|:-------|:-----|:-----|
| queryConGm_ + keyConGm_ | ~3MB | R2 |
| vBetaWsGm_ | ~2.6MB | R4 |
| gBKWsGm_ | ~2.6MB | R5 |
| kg_ | ~8MB | R7 |
| **新增** attnWsNegGm_ | ~0.5MB | R5 |
| **净削减** | **~15.7MB** | |
