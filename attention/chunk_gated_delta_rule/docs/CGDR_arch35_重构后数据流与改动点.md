# CGDR arch35 重构后数据流与改动点

> 本文档描述 arch35 CGDR 算子重构后的最终实现状态。
> 包含 R2/R4/R5/R7 优化 + CV 数据通路优化 + 死 workspace 清理。
> 数学等价性已用 Python 验证（误差 < 1e-12），RDV 288 用例全部 PASS。

---

## 0 优化点总览

| 编号 | 需求 | 状态 | 方案 |
|:-----|:-----|:-----|:-----|
| R2 | matmul 直接读 strided GM | **已实现** | 删除 QKPreProcess/queryConGm_/keyConGm_，SetOrgShape Ka/Kb |
| R3 | g_exp 移到 Stage2 | **不可行** | g_exp 在 A 右边（内维），A 非对角不可移到结果行 |
| R4 | beta 移到 A 列缩放 | **已实现** | `A_beta = A @ diag(beta)`，删除 VBetaCompute/vBetaWsGm_ |
| R5 | g_exp 移到 A 列缩放 | **已实现** | `A_beta_gexp = A_beta @ diag(g_exp)`，删除 GBKCompute/gBKWsGm_ |
| R6 | gCum_ 输出 + 后续重算 | **已满足** | 原始代码已是此做法 |
| R7 | kg 系数移到 v_new | **已实现** | `v_new *= d[:,None]`，删除 KgCompute，Stage2 用 strided K |
| CV | Stage2 CV 数据通路优化 | **已实现** | 合并 gCum 读 + CastGmToGmDual 融合 + VPrime/AttnInter 融合 |
| WS | 死 workspace 清理 | **已实现** | 删除 attnInter_（从未被任何 Stage 读取） |

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
        GCumExpCompute(gGm_[bgOffset[i]], gCumGm_[chunkRowBase[i]],
                       gCumSumUB[i*C], gCumExpUB[i*C], validLen[i])
        GammaCompute(gBroadUB, gammaUB, gCumSumUB[i*C])

// R2: 不再 QKPreProcess (AIC 直接读 strided GM), 不再 SetFlag(0x9)

// ===== 步骤 2: beta 搬入 (供列缩放) =====
for i in [0, curParaNum):
    BetaCopyInWithStride(betaGm_[bgOffset[i]], betaFullUB[C], subValidLen[i])

// ===== 步骤 3: KKBetaCompute + 求逆 (不变) =====
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x8)                         // ← AIC: KKT 就绪
    KKBetaCompute(kkWs_[i*C*C], betaUB[i*H])
    InverseCompute(attnWs_[i*C*C], gammaUB)
    CrossCoreSetFlag<PIPE_MTE3>(0x7)               // → AIC: 求逆完成

// ===== 步骤 4: R4 列缩放 β → A_beta (独立循环, AIC matmul5 连续执行) =====
for i in [0, curParaNum):
    CrossCoreWaitFlag(0xB)                         // ← AIC: AttnInverseMMCompute 完成
    ColumnScaleBeta(attnWs_[i*C*C], betaUB[i*H])   // R4: A_beta = A * beta[None,:]
    CrossCoreSetFlag<PIPE_MTE3>(0x5)               // → AIC: A_beta 就绪

// ===== 步骤 5: R5 列缩放 g_exp → A_beta_gexp (独立循环, 从 GM 重读 A_beta) =====
for i in [0, curParaNum):
    ColumnScaleGExp(attnWsNegGm_[i*C*C], attnWs_[i*C*C], gCumExpUB[i*C])
    //   从 GM attnWs_ 重读 A_beta (不依赖 UB inverseRes_)
    //   R5: -A_beta_gexp = -(A_beta * g_exp[None,:])
    CrossCoreSetFlag<PIPE_MTE3>(0x6)               // → AIC: A_beta_gexp 就绪

// ===== 步骤 6: q_prime (R2: Q 从 strided GM) =====
for i in [0, curParaNum):
    QPrimeCompute(qPrimeGm, gCumSumUB[i*C], gCumExpUB[i*C],
                  queryGm_[qk_base + subOffset*qkStride], subValidLen[i], qkStride)

CrossCoreWaitFlag(0xA)                             // ← AIC: QKT 完成
```

### 1.2 AIC 侧

```
// ===== matmul 1: KKT (R2: strided GM) =====
for i in [0, curParaNum):
    AICProcess(mmBf16_, keyGm_[qk_base], keyGm_[qk_base], kkWs[i*C*C],
               {M=C, N=C, K=Dk, sM=C, sN=C, sK=Dk, Ka=Nk*Dk, Kb=Nk*Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0x8)                // → AIV: KKT 完成

// ===== matmul 2: QKT (R2: strided GM) =====
for i in [0, curParaNum):
    AICProcess(mmBf16_, queryGm_[qk_base], keyGm_[qk_base], qktGm[...],
               {M=validLen[i], N=validLen[i], K=Dk, Ka=Nk*Dk, Kb=Nk*Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0xA)                // → AIV: QKT 完成

// ===== matmul 3: 求逆辅助 (不变) =====
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x7)                         // ← AIV: 求逆完成
    AttnInverseMMCompute(i*C*C)
    CrossCoreSetFlag<PIPE_FIX>(0xB)                // → AIV: 可做列缩放 (新增)

// ===== matmul 5: vInner (先做: AIV R4 完成即可启动, 与 AIV R5 并行) =====
//   v_inner = (A @ diag(β)) @ V
//   A = attnWs_(C,C) [A_beta], B = valueGm_(C,Dv) strided, result → vInner_ (C,Dv)
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x5)                         // ← AIV: A_beta 就绪
    AICProcess(mmVInner_, attnWs_[i*C*C], valueGm_[vOffset], vInnerGm[...],
               {M=C, N=Dv, K=C, Ka=0, Kb=Nv*Dv})
    // v_inner = A_beta @ V = A@(V·β) ✓

// ===== matmul 4: k_cumdecay (后做: 等 AIV R5 完成) =====
//   k_cumdecay = -(A @ diag(β) @ diag(g_exp)) @ K
//   A = attnWsNegGm_(C,C) [A_beta_gexp, 含负号], B = keyGm_(C,Dk) strided, result → kCumDecay_ (C,Dk)
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x6)                         // ← AIV: A_beta_gexp 就绪
    AICProcess(mmBf16_, attnWsNegGm_[i*C*C], keyGm_[qk_base], kCumDecayGm[...],
               {M=C, N=Dk, K=C, Ka=0, Kb=Nk*Dk})
    // k_cumdecay = -A_beta_gexp @ K = A@(-K·β·g_exp) ✓
```

### 1.3 Stage1 同步关系图

```
AIV                                       AIC
────                                       ────
GCumExpCompute + GammaCompute (→ gCum_ GM)
beta 搬入 (全量 C)
                                            KKT matmul (strided GM → kkWs)
                                            SetFlag(0x8) ──────────→ WaitFlag(0x8)
KKBetaCompute (读 kkWs, 含 beta)
InverseCompute (求逆 A → attnWs)
SetFlag(0x7) ─────────────────────────→   WaitFlag(0x7)
                                            AttnInverseMMCompute (读/写 attnWs 的 A)
                                            SetFlag(0xB) ──────────→ WaitFlag(0xB)
[R4 循环: 全部 chunk 连续做列缩放 β]
读 A → 列缩放 β → A_beta → 写 attnWs
SetFlag(0x5) ─────────────────────────→   WaitFlag(0x5) → matmul5: A_beta @ V (连续执行)
                                            [matmul5 batch ×paraNum]
[R5 循环: 全部 chunk 连续做列缩放 g_exp]   ← matmul5 与 AIV R5 并行
重读 A_beta from GM → 列缩放 g_exp → negate → 写 attnWsNegGm_
SetFlag(0x6) ─────────────────────────→   WaitFlag(0x6) → matmul4: -A_beta_gexp @ K
                                            [matmul4 batch ×paraNum]
                                            QKT matmul (strided GM → qkt_)
                                            SetFlag(0xA) ──────────→ WaitFlag(0xA)
QPrimeCompute (Q 从 strided GM)
```

### 1.4 Stage1 输出

| 输出 | 公式 | dtype/shape | 变化 |
|:-----|:-----|:------------|:-----|
| gCum_ | `g.cumsum()` | FP32 Nv×Sp | 不变 |
| kCumDecay_ | `-A_beta_gexp @ K` = `A@(-K·β·g_exp)` | BF16 Nv×Sp×Dk | R4+R5: K 从 strided GM |
| vInner_ | `A_beta @ V` = `A@(V·β)` | FP32/BF16 Nv×Sp×Dv | R4: V 从 strided GM |
| qPrime_ | `q*scale*g_exp` | BF16 Nv×Sp×Dk | R2: Q 从 strided GM |
| qkt_ | `Q@K^T` | BF16 Nv×Sp×C | R2: Q/K 从 strided GM |

### 1.5 Stage1 WS 布局

```
kkWsGm_      BF16 [P×C×C]     // KKT 结果 (不变)
attnWsGm_    BF16 [P×C×C]     // A_beta (R4 后; 重构前存 A, 重构后存 A_beta)
attnWsNegGm_ BF16 [P×C×C]     // A_beta_gexp (R5 新增, 供 matmul4)
// 删除: queryConGm_/keyConGm_ (R2), vBetaWsGm_ (R4), gBKWsGm_ (R5)
```

---

## 2 Stage2 重构后计算流程（R7 + CV 数据通路优化）

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

### 2.1 CV 数据通路 3-Phase 流水线

Stage2 采用 CV (Cube-Vector) 3-phase 流水线，AIC 的 VPrime+AttnInter 融合执行（无中间同步），StateNew 与下一 chunk 的 AIV state_prep 天然重叠。

```
Phase layout:
  AIV: state_prep + CalGCumExpAndD → setFlag(0x3) → waitFlag(0x2) → v_new_scale → setFlag(0x5) → waitFlag(0x4)
  AIC:                              waitFlag(0x3) → VPrime+AttnInter(fused) → setFlag(0x2) → waitFlag(0x5) → StateNew → setFlag(0x4)
```

### 2.2 AIV 侧 — FP32 path

```
ProcessAivFp32(nvId, cId, length):
  if subBlockIdx == 0:
    // state 搬入 + Cast (不变)
    CopyIn<FP32>(state[nvId*stride], Dv, Dk)
    stateFp32UB = DeQue → DataCopy → FreeTensor
    Cast FP32→BF16 → curStateBf16 (GM)

    // CV: 合并 gCum 读 — state decay + d 向量在一次 GM 读取中完成
    //   原流程: CalGCumExpFp32(读gCum_[-1]) + CalD(再读gCum_[-1]+g_cum[0:C]) = 2次读取
    //   新流程: CalGCumExpAndD(一次读gCum_[-1]+g_cum[0:C]) = 1次读取
    CalGCumExpAndD(gCum_[nvId*Sp + length])

    // state_out 写回 GM (不变)
    CopyOut<FP32>(stateOut, Dv, Dk)

  CrossCoreSetFlag<PIPE_MTE3>(0x3)                 // → AIC: state 就绪
  CrossCoreWaitFlag(0x2)                           // ← AIC: CalVPrime+CalAttnInter 完成

  if subBlockIdx == 0:
    // CV 融合: CastGmToGmDual — 单次 GM→UB 读取, 产出两份输出
    //   原: CastGmToGm(读vInner→写vInnerBf16) + CastGmToGmWithScale(再读vInner→写vNewScaledGm) = 2次GM→UB
    //   新: CastGmToGmDual(读vInner 1次 → 写vInnerBf16 unscaled + 写vNewScaledGm scaled) = 1次GM→UB
    mm_offset1 = nvId*Sp*Dv + length*Dv
    CastGmToGmDual<FP32→BF16>(vInner[mm_offset1], vInnerBf16[mm_offset1],
                               vNewScaledGm_, curChunkSize*Dv)
    // vInnerBf16_ = v_new (unscaled, 供 Stage3)
    // vNewScaledGm_ = v_new * d[:,None] (scaled, 供 CalStateNew)

  CrossCoreSetFlag<PIPE_MTE3>(0x5)                 // → AIC: v_new_scaled 就绪
  CrossCoreWaitFlag(0x4)                           // ← AIC: CalStateNew 完成
```

### 2.3 AIV 侧 — BF16 path

```
ProcessAivBf16(nvId, cId, length):
  curState = (cId==0) ? curStateBf16[stride] : stateOut[nvId*Dv*Dk]
  finalState = stateOut[nvId*Dv*Dk]

  if subBlockIdx == 0:
    CopyIn<BF16>(curState, Dv, Dk)
    // CV: 合并 gCum 读 — CalGCumExpBf16AndD
    //   state decay (BF16→FP32→Muls(expGEnd)→BF16) + d 向量计算 在一次 gCum 读取中完成
    CalGCumExpBf16AndD(gCum_[nvId*Sp + length])

  CrossCoreSetFlag<PIPE_MTE3>(0x3)                 // → AIC: state 就绪
  CrossCoreWaitFlag(0x2)                           // ← AIC: CalVPrime+CalAttnInter 完成

  if subBlockIdx == 0:
    CopyOutState(finalState)
    // CV: 写 vNewScaledGm_ only (vInner_ 保持不变供 Stage3 直接读)
    //   CastGmToGmWithScale<BF16→BF16>: BF16→FP32→Scale(d)→BF16
    mm_offset1 = nvId*Sp*Dv + length*Dv
    CastGmToGmWithScale<BF16→BF16>(vInner[mm_offset1], vNewScaledGm_, curChunkSize*Dv)

  CrossCoreSetFlag<PIPE_MTE3>(0x5)                 // → AIC: v_new_scaled 就绪
  CrossCoreWaitFlag(0x4)                           // ← AIC: CalStateNew
```

### 2.4 AIC 侧

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

    // CV: VPrime + AttnInter 融合 — 两个 matmul 连续执行, 无中间 CrossCore 同步
    //   vInner += k_cumdecay @ state^T → vInner = v_new  (accum=1)
    CalVPrime(kCumdecay[mm_offset0], stateForAic, vInner[mm_offset1])
    //   attn_inter = qPrime @ state^T → out_  (直接写最终输出)
    CalAttnInter(qPrime[mm_offset0], stateForAic, out[...])

    CrossCoreSetFlag<PIPE_FIX>(0x2)                // → AIV: state matmul 完成
    CrossCoreWaitFlag(0x5)                         // ← AIV: v_new_scaled 就绪

    // R7: CalStateNew 用 strided K (不用 kg_)
    //   state += v_new_scaled^T @ K (accum=1)
    //   K 从原始 GM (T,Nk,Dk) strided 读, SetOrgShape Ka=Nk*Dk
    //   v_new_scaled 从 vNewScaledGm_ 读 (per-core, 已含 d 行缩放)
    kStart = cg.startPos + length
    kOffset = kStart * Nk*Dk + nvId * (Nk/Nv)*Dk
    CalStateNew(vNewScaledGm_, keyGm_[kOffset], stateOut, kStride=Nk*Dk)
    // 等价: state_new = state*exp(g[-1]) + (v_new*d)^T @ K ✓

    CrossCoreSetFlag<PIPE_FIX>(0x4)                // → AIV: state 更新完成
```

### 2.5 Stage2 同步关系图

```
AIV                                            AIC
────                                            ────
state CopyIn + Cast → curStateBf16
CV: CalGCumExpAndD (合并 state decay + d 计算)
state_out 写回 GM
SetFlag(0x3) →                                WaitFlag(0x3)
                                                CV: CalVPrime (vInner += kcd@state^T)     ┐ 融合
                                                CV: CalAttnInter (qPrime@state^T → out)   ┘ 无中间同步
                                                SetFlag(0x2) ─────────→ WaitFlag(0x2)
CV: CastGmToGmDual / CastGmToGmWithScale
    (vInner→vInnerBf16 + vNewScaledGm_)
SetFlag(0x5) →                                WaitFlag(0x5)
                                                R7: CalStateNew (state += v_new_scaled^T @ K_strided)
                                                SetFlag(0x4) ─────────→ WaitFlag(0x4)
[下一 chunk]                                    ← StateNew 与下一 chunk state_prep 重叠
```

### 2.6 Stage2 新增/修改函数详解

#### CalGCumExpAndD (FP32 path) — 合并 gCum 读

**原流程**：`CalGCumExpFp32` + `CalD` 两个独立函数，各读一次 `gCum_[-1]`（scalar）和 `g_cum[0:C]`（vector）。

**新流程**：合并为单次 GM 读取，同时计算：
- `exp(g[-1])` — state decay 标量，用于 `state *= exp(g[-1])`
- `d[j] = exp(g[-1] - g_cum[j])` — v_new 行缩放向量，用于 `v_new *= d[:,None]`

```
CalGCumExpAndD(gCumGm):
    gEnd = gCumGm.GetValue(curChunkSize-1)     // 读 gCum_[-1] (scalar)
    expGEnd = exp(gEnd)
    gLocal = DataCopyPad(gCumGm, curChunkSize)  // 读 g_cum[0:C] (vector, 单次)
    dUb = Duplicate(gEnd, C) → Sub(dUb, gLocal) → Exp(dUb)  // d = exp(gEnd - g_cum)
    stateFp32Ub *= expGEnd                       // state decay
```

**收益**：消除 1 次 `DataCacheCleanAndInvalid` + 1 次 `GetValue` 往返。

#### CalGCumExpBf16AndD (BF16 path) — 合并 gCum 读 + state cast

与 FP32 path 对应，额外处理 BF16 state 的 Cast：
```
CalGCumExpBf16AndD(gCum):
    expGEnd = exp(gCum[-1])
    stateIn(BF16) → Cast(FP32) → Muls(expGEnd) → Cast(BF16) → stateOut  // state decay
    dUb = exp(gEnd - g_cum[0:C])                                           // d 向量
```

#### CastGmToGmDual (FP32 path) — 单次 GM→UB 读取产出双输出

**原流程**：`CastGmToGm`（读 vInner FP32 → 写 vInnerBf16 BF16 unscaled）+ `CastGmToGmWithScale`（再读 vInner FP32 → 写 vNewScaledGm BF16 scaled）= 2 次 GM→UB 往返。

**新流程**：单次 GM→UB 读取，同一 `srcLocal` 产出两份输出：
```
CastGmToGmDual(srcGm=vInner[FP32], dstUnscaled=vInnerBf16, dstScaled=vNewScaledGm):
    srcLocal = DataCopyPad(srcGm)              // 单次 GM→UB
    // 写 unscaled (for Stage3)
    dstUnscaled = Cast(BF16, srcLocal) → DataCopyPad(dstUnscaledGm)
    // 写 scaled (for CalStateNew) — 复用同一 srcLocal
    srcLocal *= d[:,None] (Broadcast + Mul)
    dstScaled = Cast(BF16, srcLocal) → DataCopyPad(dstScaledGm)
```

**收益**：节省 1 次 GM→UB `DataCopyPad` 往返（~16KB/chunk）。

#### CastGmToGmWithScale — 模板特化支持 BF16→BF16

原 `CastGmToGmWithScale` 仅支持 FP32→BF16。BF16 path 需要在不改变 dtype 的情况下施加 d 缩放。

新增 `if constexpr` 分支：
- **FP32→BF16**：`Broadcast(d) → Mul(srcLocal) → Cast(BF16)`
- **BF16→BF16**：`Cast(FP32) → Broadcast(d) → Mul → Cast(BF16)`（需中间 FP32 转换）

#### vNewScaledGm_ — per-core scaled v_new buffer

**问题**：R7 将 `d = exp(g[-1]-g_cum)` 行缩放施加到 v_new 上，但 `vInner_` 同时被 Stage3 读取做 `masked_qkt @ v_new`。如果直接覆写 `vInner_`，Stage3 会读到 `v_new * d` 而非 `v_new`（数学错误）。

**解决**：新增 per-core GM buffer `vNewScaledGm_`（复用 `stageWsAddr_`，每核 `C×Dv` BF16）：
- AIV 将 `v_new * d` 写入 `vNewScaledGm_`（供 CalStateNew）
- `vInner_` / `vInnerBf16_` 保持未缩放（供 Stage3）

### 2.7 Stage2 输出

| 输出 | 公式 | 变化 |
|:-----|:-----|:-----|
| vInner_ | `v_inner + k_cumdecay@state` = `v_new` | 不变 (accum=1) |
| vInnerBf16_ | `v_new` (unscaled, 供 Stage3) | CV: CastGmToGmDual 单次读 |
| vNewScaledGm_ | `v_new * d[:,None]` (scaled, 供 CalStateNew) | 新增 per-core buffer |
| stateOut_ | `state*exp(g[-1]) + v_new_scaled^T@K` | R7: K 从 strided GM |
| out_ | `q_prime @ state^T` | 不变 (直接写最终输出) |

---

## 3 Stage3 计算流程（不变）

Stage3 无改动。decay_mask 从 GM `gCum_` 读取（Stage1 已写入），vInner 从 `vInnerBf16_`（FP32 path）或 `vInner_`（BF16 path）读取。

### 3.1 AIV 侧

```
CalMaskedQKT(outGM, nvId, chunkPos):
    if gOptional:
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
AICProcess(tmpGM, vInner, attnOut):
    // attn_out = masked_qkt @ v_new (accum=1)
    // attnOut 直接绑定 out_, 与 Stage2 CalAttnInter 写入的地址相同
    // accum=1 在 out_ 上叠加: attn_out = attn_inter + masked_qkt @ v_new
    mm3.SetOrgShape(curChunkSize, Dv, curChunkSize, curChunkSize, Nv*Dv)
    mm3.SetSingleShape(curChunkSize, Dv, curChunkSize)
    mm3.SetTensorA(tmpGM)           // masked_qkt
    mm3.SetTensorB(vInner)          // v_new (unscaled)
    mm3.IterateAll(attnOut, 1)      // out_ += masked_qkt @ v_new
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

### 重构后（R2+R4+R5+R7+CV+WS清理）

```
gCum_ kCumDecay_ vInner_ vInnerBf16_ qPrime_ kg_ qkt_ curStateBf16_ masks stageWs
stageWs: kkWs/attnWs/attnWsNeg (3 buffer) + vNewScaledGm_ (per-core C×Dv, 复用 stageWs)
// 删除: attnInter_ (死代码), queryCon/keyCon (R2), vBetaWs (R4), gBKWs (R5)
// 新增: attnWsNeg (R5, C×C), vNewScaledGm_ (per-core C×Dv)
// 注: kg_ 仍分配(arch22需要), arch35 Stage2 不再使用
```

### WS 削减汇总

| 删除项 | 大小 | 来源 |
|:-------|:-----|:-----|
| attnInter_ | ~48MB | 死代码清理 (从未被任何 Stage 读取) |
| queryConGm_ + keyConGm_ | ~3MB | R2 |
| vBetaWsGm_ | ~2.6MB | R4 |
| gBKWsGm_ | ~2.6MB | R5 |
| **新增** attnWsNegGm_ | ~0.5MB | R5 |
| **新增** vNewScaledGm_ | ~0.5MB | R7 (per-core, 复用 stageWs) |
| **净削减** | **~55.2MB** | |

---

## 6 Bug 修复记录

### Bug 1: R4/R5 inverseRes_ 串扰

**问题**：R4 和 R5 为两个独立循环时，R4 循环遍历所有 chunk 覆写 `inverseRes_`，R5 循环从 chunk 0 重新开始但 `inverseRes_` 已被最后一个 chunk 的 A_beta 覆盖。合并为单循环后修复了此问题，但牺牲了 AIC 连续执行 matmul 的能力。

**最终修复**：将 R4/R5 重新拆分为两个独立循环，`ColumnScaleGExp` 从 GM `attnWsGm_` 重读 A_beta（不依赖 UB `inverseRes_` 持久性）。代价是多 1 次 GM→UB 读取（~4KB/chunk），收益是 AIC 可连续做 paraNum 个 matmul5 再连续做 paraNum 个 matmul4。

### Bug 2: Stage3 读到 d 缩放后的 v_new

**问题**：R7 将 `d` 行缩放直接施加到 `vInner_`/`vInnerBf16_` GM buffer，Stage3 从同一 buffer 读取得到 `v_new*d` 而非 `v_new`。

**修复**：新增 per-core `vNewScaledGm_` buffer，AIV 将缩放后的 v_new 写入此 buffer 供 CalStateNew 使用，`vInner_`/`vInnerBf16_` 保持未缩放供 Stage3 读取。
