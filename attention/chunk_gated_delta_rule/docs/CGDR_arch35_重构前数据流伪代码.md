# CGDR arch35 重构前数据流伪代码

> 本文档描述重构前（当前 upstream master）arch35 平台 CGDR 算子的完整计算流程，
> 包含 AIC/AIV 协同、各级存储（GM/UB）中的 shape 变化、matmul 参数、同步关系。
> 作为重构方案的前置参考。

---

## 0 全局符号约定

```
C       = chunkSize = 64              // chunk 长度
H       = C / TASK_RATIO = 32         // sub-block 处理行数 (TASK_RATIO=2, 1:2 模式)
Dk      = key 维度 (≤128)
Dv      = value 维度 (≤128)
DkAlign = Ceil(Dk, 16) * 16           // BF16 对齐 (16 elements = 32B)
DvAlign = Ceil(Dv, 16) * 16
Nv      = value 头数 (≤64)
Nk      = key 头数 (≤64), Nv % Nk == 0
Sp      = Ceil(cgLen, C) * C           // padding 后的 chunk group 长度
Nc      = Sp / C                       // chunk group 内 chunk 数
P       = stageOneParaNum = 4          // Stage1 并行 chunk 数
coreNum = aiCoreNum
FLAG_ID_MAX = 16                       // CrossCore flag 偏移 (1:2 模式下 intra-block 同步)
```

## 1 GM Workspace 布局 (user workspace)

```
offset 0:    gCum_         FP32  [Nv][Sp]                         // g_cumsum, Stage1→Stage2/3
+ Nv*Sp*4
offset 1:    kCumDecay_    BF16  [Nv][Sp][Dk]                     // k_cumdecay = A@(-k*beta*g_exp)
+ Nv*Sp*Dk*2
offset 2:    vInner_       FP32/BF16 [Nv][Sp][Dv]                 // v_inner = A@(v*beta), FP32 when stateIsFp32
+ Nv*Sp*Dv*sizeof(vInnerType)
offset 3:    vInnerBf16_   BF16  [Nv][Sp][Dv]                     // (FP32 path only) vInner 的 BF16 副本
+ Nv*Sp*Dv*2   (FP32 path only)
offset 4:    qPrime_       BF16  [Nv][Sp][Dk]                     // q*scale*g_exp
+ Nv*Sp*Dk*2
offset 5:    attnInter_    BF16  [Nv][Sp][Dv]                     // (arch22 compat, arch35 不直接用)
+ Nv*Sp*Dv*2
offset 6:    kg_           BF16  [Nv][Sp][Dk]                     // k*exp(g_cum[-1]-g_cum)
+ Nv*Sp*Dk*2
offset 7:    qkt_          BF16  [Nv][Sp][C]                      // Q@K^T
+ Nv*Sp*C*2
offset 8:    curStateBf16_ BF16  [Nv][Dv][Dk]                     // (FP32 path only) state 的 BF16 副本
+ Nv*Dv*Dk*2   (FP32 path only)
offset 9:    stageOneMask_ FP32  [coreNum*TASK_RATIO][C][C]       // 下三角 mask (Stage1)
+ coreNum*2*C*C*4
offset 10:   stageThreeMask_ FP32 [coreNum*TASK_RATIO][C][C]      // 上三角 mask (Stage3)
+ coreNum*2*C*C*4
offset 11:   stageWsAddr_                                       // Stage1 临时 workspace (见 §2)
```

## 2 Stage1 临时 Workspace 布局 (stageWsAddr_)

```
每个 buffer 大小 = coreNum * P * <chunk_size>
分配顺序:
  ws[0]: gBKWsGm_     BF16 [coreNum*P][C][Dk]     // -k*beta*g_exp, 供 AIC matmul
  ws[1]: queryConGm_  BF16 [coreNum*P][C][Dk]     // Q 连续化 (QKPreProcess 输出)
  ws[2]: keyConGm_    BF16 [coreNum*P][C][Dk]     // K 连续化 (QKPreProcess 输出)
  ws[3]: kkWsGm_      BF16 [coreNum*P][C][C]      // K@K^T 结果
  ws[4]: attnWsGm_    BF16 [coreNum*P][C][C]      // 求逆中间结果 A
  ws[5]: vBetaWsGm_   BF16 [coreNum*P][C][Dv]     // v*beta, 供 AIC matmul
总大小 = sizeLow * C * (2*C + 3*Dk + Dv) * P * coreNum
```

---

## 3 整体流程 (CGDR::Process)

```
SyncAll()
for bid in [0, B):
    seqStart, seqEnd = 累积序列范围
    for pos in [seqStart, seqEnd, maxGroupLength):
        cg = {startPos=pos, length=min(maxGroupLen, seqEnd-pos), chunkSize=C}
        useInitState = (pos == seqStart)
        curState = useInitState ? initState[bid] : finalState[bid]

        RunStage1(cg)                      // 全 chunk 并行
        SyncAll()                          // ← Stage 间全局同步 #1

        RunStage2(cg, curState, finalState[bid], useInitState)  // 逐 chunk 递推
        SyncAll()                          // ← Stage 间全局同步 #2

        RunStage3(cg)                      // 全 chunk 并行
        SyncAll()                          // ← Stage 间全局同步 #3
```

---

## 4 Stage1 数据流伪代码

### 4.1 任务调度

```
Stage1::Process():
    totalChunk = Nv * Nc
    // 核间均分: former 核多处理 1 个 chunk
    formerCoreNum = totalChunk % coreNum
    tailChunkNum  = totalChunk / coreNum
    formerChunkNum = tailChunkNum + 1
    [start, end) = 本核负责的 taskId 范围

    for taskId in [start, end, P):        // 每批 P 个 chunk
        curParaNum = min(P, end - taskId)
        for i in [0, curParaNum):
            curTaskId = taskId + i
            curNId  = curTaskId % Nv       // head 编号
            curCgId = curTaskId / Nv       // chunk 编号
            SetChunkOffset(i, curNId, curCgId)
        ProcessParaChunk(curParaNum)       // AIC/AIV 并行执行
```

### 4.2 SetChunkOffset (每 chunk 偏移计算)

```
SetChunkOffset(id, curNId, curCgId):
    validLen = C  (尾 chunk: cgLen % C)
    subValidLen = subBlockIdx==0 ? min(validLen, H) : validLen - H
    cgLenPad = Ceil(cgLen, C) * C
    chunkRowBase[id]    = curNId * cgLenPad + curCgId * C      // GM 中行基地址
    chunkStartRow[id]   = cg.startPos + curCgId * C            // T 维度的绝对行号
    nIdBatch[id]        = curNId
    bgOffsetBatch[id]   = chunkStartRow[id] * Nv + curNId      // beta/g 的 GM 偏移
```

### 4.3 AIV 侧 (ParaChunkAIV)

```
// ============ 步骤 1: QKPreProcess ============
for i in [0, curParaNum):
    subRow    = chunkStartRow[i] + subOffset                    // subOffset = subBlockIdx * H
    qk_base   = subRow * Nk * Dk + nId[i] * (Nk/Nv) * Dk       // 原始 GM (T,Nk,Dk) 偏移
    wsOffset  = i * (C*Dk) + subOffset * Dk

    // Q: GM(T,Nk,Dk) strided → WS(C,Dk) continuous
    QKPreProcess(queryGm_[qk_base], queryConGm_[wsOffset], outKgGm, subValidLen[i])
    // K: GM(T,Nk,Dk) strided → WS(C,Dk) continuous
    //     !gOptional 时同时输出 kg = key (连续化)
    QKPreProcess(keyGm_[qk_base],   keyConGm_[wsOffset],   outKgGm, subValidLen[i], kgFlag=true)

    CrossCoreSetFlag<PIPE_MTE3>(0x9)    // → AIC: Q/K 连续化完成, 可做 KKT

    // QKPreProcess 内部:
    //   GM(T,Nk,Dk) stride=Nk*Dk → DataCopyPad → UB(C, DkAlign) BF16
    //   补零到 H 行 → DataCopyPad → WS(C, Dk) BF16 (queryConGm_/keyConGm_)

// ============ 步骤 2: g_cum + gamma (gOptional only) ============
if gOptional:
    for i in [0, curParaNum):
        // g_cum = g.cumsum(dim=-1)
        //   GM(T,Nv) stride=Nv → Gather → UB(C) FP32 → CumSum → gCumSumUB(C)
        //   Exp → gCumExpUB(C)
        //   DataCopyPad → GM gCum_[chunkRowBase[i]] (FP32, Nv×Sp)
        GCumExpCompute(gGm_[bgOffset[i]], gCum_[chunkRowBase[i]],
                       gCumSumUB[i*C], gCumExpUB[i*C], validLen[i])

        // gamma = exp(g_cum[:,None] - g_cum[None,:]) * tril_mask
        //   Broadcast g_cum → (C,C), Sub, Mul(mask), Exp, Mul(mask) → gammaUB(C,C)
        GammaCompute(gBroadUB[i*C*maxLen], gammaUB[i*C*maxLen], gCumSumUB[i*C])

// ============ 步骤 3: beta 搬入 ============
for i in [0, curParaNum):
    // GM(T,Nv) stride=Nv → Gather → UB(H) BF16 → Cast → betaUB(H) FP32
    BetaCopyInWithStride(betaGm_[bgOffset[i]], betaUB[i*H], subValidLen[i])

// ============ 步骤 4: KKT*beta → 求逆 → 写出 A ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x8)                // ← AIC: KKT matmul 完成 (kkWs 就绪)

    // KKBetaCompute: attn = KKT * beta[:,None]
    //   WS kkWs(C,C) BF16 → DataCopyPad → UB(H,C) → Cast FP32 → kkLocal(H,C)
    //   Broadcast beta(H,1) → (H,C) → Mul → attnUB(H,C) FP32
    KKBetaCompute(kkWs_[i*C*C], betaUB[i*H])

    // InverseCompute: A = (I + attn * gamma)^{-1}  [含 beta]
    //   if gOptional: attn *= gamma[subOffset*C : subOffset*C + H*C]  (行缩放)
    //   else:         attn *= mask (下三角)
    //   inverseRes = -attn
    //   前代法求逆 (InverseAIVVF, 32×32 对角块)
    //   A 结果在 inverseRes(H,C) → Cast BF16 → DataCopyPad → WS attnWs(H,C)
    InverseCompute(attnWs_[i*C*C], gammaUB[i*C*maxLen])
    //   注: KKBetaCompute 和 InverseCompute 共用 attnUbFloat_
    //   attnWs 中存的是 A = (I + KKT*beta*gamma)^{-1} 的左半部分 (subBlock 0: 上 H 行, subBlock 1: 下 H 行)

    CrossCoreSetFlag<PIPE_MTE3>(0x7)      // → AIC: 求逆完成, 可做辅助 matmul

// ============ 步骤 5: gBK + kg ============
for i in [0, curParaNum):
    outKgGm = outKgBaseGm_[chunkRowBase[i]*Dk]
    wsOffset = i*(C*Dk) + subOffset*Dk

    // GBKCompute:
    //   gOptional:  gB = -beta * g_cum_exp            UB(H) FP32
    //   !gOptional: gB = -beta                        UB(H) FP32
    //   Broadcast gB(H,1) → (H,DkAlign) → Mul kUbFloatCon(H,DkAlign)
    //     kUbFloatCon: 从 keyConGm_(WS) DataCopyPad → Cast FP32
    //   gBK = gB * k → Cast BF16 → DataCopyPad → WS gBKWs(H,Dk)
    //   if gOptional:
    //     gEnd = exp(g_cum[-1] - g_cum[subOffset])    UB(H) FP32
    //     Broadcast gEnd(H,1) → (H,DkAlign) → Mul k → Cast BF16 → DataCopyPad → GM kg(H,Dk)
    GBKCompute(gBKWs_[i*C*Dk], outKgGm, betaUB[i*H],
               gCumSumUB[i*C], gCumExpUB[i*C], keyConGm_[wsOffset])

    CrossCoreSetFlag<PIPE_MTE3>(0x6)      // → AIC: gBK 就绪, 可做 k_cumdecay matmul

// ============ 步骤 6: v_beta ============
for i in [0, curParaNum):
    vOffset = chunkStartRow[i] * (Nv*Dv) + nId[i]*Dv

    // VBetaCompute: v_beta = value * beta[:,None]
    //   GM(T,Nv,Dv) stride=Nv*Dv → DataCopyPad → UB(H,DvAlign) BF16 → Cast FP32
    //   Broadcast beta(H,1) → (H,DvAlign) → Mul → vBetaUB(H,DvAlign) → Cast BF16
    //   DataCopyPad → WS vBetaWs(H,Dv)
    VBetaCompute(valueGm_[vOffset], vBetaWs_[i*C*Dv], betaUB[i*H],
                 valueUbFloat[i*C*maxLen], subValidLen[i])

    CrossCoreSetFlag<PIPE_MTE3>(0x5)      // → AIC: vBeta 就绪, 可做 v_inner matmul

// ============ 步骤 7: q_prime ============
for i in [0, curParaNum):
    wsOffset = i*(C*Dk) + subOffset*Dk

    // QPrimeCompute: q_prime = query * scale * g_cum_exp[:,None]
    //   queryConGm(WS) → DataCopyPad → UB(H,DkAlign) → Cast FP32
    //   gOptional: Muls(scale) → Mul(g_cum_exp broadcast) → qPrimeUB(H,DkAlign)
    //   !gOptional: Muls(scale) → qPrimeUB(H,DkAlign)
    //   Cast BF16 → DataCopyPad → GM qPrime(H,Dk)
    QPrimeCompute(qPrimeGm_[chunkRowBase[i]*Dk], gCumSumUB[i*C],
                 gCumExpUB[i*C], queryConGm_[wsOffset])

CrossCoreWaitFlag(0xA)                    // ← AIC: QKT matmul 完成
```

### 4.4 AIC 侧 (ParaChunkAIC)

```
// ============ matmul 1: KKT = K @ K^T ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x9)                // ← AIV: Q/K 连续化完成
    // A=keyConGm(C,Dk), B=keyConGm(C,Dk) transB, C=kkWsGm(C,C)
    // SetOrgShape(M=C, N=C, K=Dk), SetSingleShape(C,C,Dk)
    AICProcess(mmBf16_, keyCon[i*C*Dk], keyCon[i*C*Dk], kkWs[i*C*C],
               {M=C, N=C, K=Dk, sM=C, sN=C, sK=Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0x8)       // → AIV: KKT 完成

// ============ matmul 2: QKT = Q @ K^T ============
for i in [0, curParaNum):
    // A=queryConGm(validLen,Dk), B=keyConGm(validLen,Dk) transB, C=qktGm(validLen,validLen)
    AICProcess(mmBf16_, queryCon[i*C*Dk], keyCon[i*C*Dk],
               qktGm[chunkRowBase[i]*C],
               {M=validLen[i], N=validLen[i], K=Dk, sM=validLen[i], sN=validLen[i], sK=Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0xA)       // → AIV: QKT 完成

// ============ matmul 3: 求逆辅助 (AttnInverseMMCompute) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x7)                // ← AIV: 求逆完成, attnWs 已写
    // 对 attnWs 的左下角 INVERSE_SHAPE×C 块做两次 matmul:
    //   1. attnWs[leftDown(C-32, 0..C)] @ attnWs[0..32, 0..C] → attnWs[leftDown]  (左下 @ 左上 → 左下)
    //   2. attnWs[rightDown(C-32, C-32..C)] @ attnWs[leftDown] → attnWs[leftDown] (右下 @ 左下 → 左下)
    AttnInverseMMCompute(i * C*C)

// ============ matmul 4: k_cumdecay = A @ gBK ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x6)                // ← AIV: gBK 就绪
    // A=attnWs(C,C), B=gBKWs(C,Dk), C=kCumDecayGm(C,Dk)
    AICProcess(mmBf16_, attnWs[i*C*C], gBKWs[i*C*Dk],
               kCumDecayGm[chunkRowBase[i]*Dk],
               {M=C, N=Dk, K=C, sM=C, sN=Dk, sK=C})
    // 输出: k_cumdecay = A @ (-k*beta*g_exp) → GM kCumDecay_ (Nv, Sp, Dk)

// ============ matmul 5: v_inner = A @ v_beta ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x5)                // ← AIV: vBeta 就绪
    // A=attnWs(C,C), B=vBetaWs(C,Dv), C=vInnerGm(C,Dv)
    // FP32 state: mmVInner_ 输出 FP32; BF16 state: mmVInner_ 输出 BF16
    AICProcess(mmVInner_, attnWs[i*C*C], vBetaWs[i*C*Dv],
               vInnerGm[chunkRowBase[i]*Dv],
               {M=C, N=Dv, K=C, sM=C, sN=Dv, sK=C})
    // 输出: v_inner = A @ (v*beta) → GM vInner_ (Nv, Sp, Dv)
```

### 4.5 Stage1 CrossCore 同步关系图

```
AIV                                    AIC
────                                   ────
QKPreProcess → WS queryCon/keyCon
SetFlag(0x9) ──────────────────────→  WaitFlag(0x9)
                                       KKT matmul → kkWs
                                       SetFlag(0x8) ──────────→ WaitFlag(0x8)
KKBetaCompute (读 kkWs)
InverseCompute → attnWs
SetFlag(0x7) ──────────────────────→  WaitFlag(0x7)
                                       AttnInverseMMCompute
GBKCompute → gBKWs
SetFlag(0x6) ──────────────────────→  WaitFlag(0x6)
                                       k_cumdecay matmul
VBetaCompute → vBetaWs
SetFlag(0x5) ──────────────────────→  WaitFlag(0x5)
                                       v_inner matmul
                                       QKT matmul
                                       SetFlag(0xA) ──────────→ WaitFlag(0xA)
QPrimeCompute
```

### 4.6 Stage1 输出汇总

```
GM 输出:
  gCum_       (FP32, Nv×Sp)        ← GCumExpCompute: g.cumsum().exp()
  kCumDecay_  (BF16, Nv×Sp×Dk)    ← matmul 4: A@(-k*beta*g_exp)
  vInner_     (FP32/BF16, Nv×Sp×Dv) ← matmul 5: A@(v*beta)
  qPrime_     (BF16, Nv×Sp×Dk)    ← QPrimeCompute: q*scale*g_exp
  kg_         (BF16, Nv×Sp×Dk)    ← GBKCompute: k*exp(g_cum[-1]-g_cum)
  qkt_        (BF16, Nv×Sp×C)     ← matmul 2: Q@K^T
```

---

## 5 Stage2 数据流伪代码

### 5.1 任务调度

```
Stage2::Process():
    // 按 Nv 分配核, 逐 chunk 递推 (chunk 间 state 依赖)
    nvPerCore = Ceil(Nv, coreNum)
    for nvId in [nvStart, nvEnd):
        curChunkSize = C
        for cId in [0, Nc):
            length = cId * C
            if cId == Nc-1: curChunkSize = seqLen % C
            AIV: ProcessAivFp32 / ProcessAivBf16 (nvId, cId, length)
            AIC: ProcessAic(nvId, cId, length)
```

### 5.2 AIV 侧 — FP32 path (ProcessAivFp32)

```
ProcessAivFp32(nvId, cId, length):
  if subBlockIdx == 0:
    // ---- state 搬入 + Cast ----
    //   cId==0 && isFirstGroup: 从 stateIn (GM, FP32, Nv×Dv×Dk) 搬入
    //   else:                    从 stateOut (GM, FP32) 搬入
    CopyIn<FP32>(state[nvId*stride], Dv, Dk)            // GM → UB(Dv, curDk) FP32
    stateIn = DeQue<FP32>()
    DataCopy(stateFp32UB, stateIn, Dv*curDk)            // UB FP32
    FreeTensor(stateIn)

    // state Cast FP32→BF16 → curStateBf16 (GM)
    bf16State = AllocTensor<BF16>()
    Cast(bf16State, stateFp32UB, Dv*curDk)              // FP32→BF16
    EnQue(bf16State) → CopyOut<BF16>(curStateBf16[nvId*Dv*Dk], Dv, Dk)

    // ---- state *= exp(g_cum[-1]) ----
    //   从 GM gCum_ 读取本 chunk 最后一个值
    //   gVal = gCum[nvId*Sp + length + curChunkSize - 1]
    //   lastGCum = exp(gVal)
    //   stateFp32UB *= lastGCum  (Muls, Dv*curDk)
    CalGCumExpFp32(gCum_[nvId*Sp + length])             // 读 GM gCum_

    // ---- state_out = state (写回 GM) ----
    fp32Local = AllocTensor<FP32>()
    DataCopy(fp32Local, stateFp32UB, Dv*curDk)
    EnQue → CopyOut<FP32>(stateOut[nvId*Dv*Dk], Dv, Dk)

  // ---- 通知 AIC: state (BF16) 就绪 ----
  CrossCoreSetFlag<PIPE_MTE3>(0x3)                      // → AIC
  CrossCoreWaitFlag(0x2)                                // ← AIC: CalVPrime+CalAttnInter 完成

  if subBlockIdx == 0:
    // ---- vInner FP32 → vInnerBf16 (CastGmToGm) ----
    //   GM vInner(FP32, C×Dv) → UB → Cast → GM vInnerBf16(BF16, C×Dv)
    mm_offset1 = nvId*Sp*Dv + length*Dv
    CastGmToGm<FP32→BF16>(vInner[mm_offset1], vInnerBf16[mm_offset1], curChunkSize*Dv)

  // ---- 通知 AIC: vInnerBf16 就绪 ----
  CrossCoreSetFlag<PIPE_MTE3>(0x5)                      // → AIC: CalStateNew 可启动
  CrossCoreWaitFlag(0x4)                                // ← AIC: CalStateNew 完成
```

### 5.3 AIV 侧 — BF16 path (ProcessAivBf16)

```
ProcessAivBf16(nvId, cId, length):
  curState = (cId==0) ? curStateBf16[nvId*stride] : stateOut[nvId*Dv*Dk]
  finalState = stateOut[nvId*Dv*Dk]

  if subBlockIdx == 0:
    CopyIn<BF16>(curState, Dv, Dk)                      // GM → UB BF16
    // ---- state *= exp(g_cum[-1]) ----
    //   gVal = gCum[nvId*Sp + length + curChunkSize - 1]
    //   lastGCum = gOptional ? exp(gVal) : 1.0
    //   Cast state BF16→FP32 → Muls(lastGCum) → Cast BF16 → stateOut UB
    CalGCumExpBf16(gCum_[nvId*Sp + length])

  CrossCoreWaitFlag(0x2)                                // ← AIC: CalVPrime+CalAttnInter 完成

  if subBlockIdx == 0:
    CopyOutState(finalState)                             // UB BF16 → GM stateOut

  CrossCoreSetFlag<PIPE_MTE3>(0x5)                      // → AIC: CalStateNew 可启动
  CrossCoreWaitFlag(0x4)                                // ← AIC: CalStateNew 完成
```

### 5.4 AIC 侧 (ProcessAic)

```
ProcessAic(nvId, cId, length):
    mm_offset0 = nvId*Sp*Dk + length*Dk                 // k_cumdecay / qPrime / kg 的 GM 偏移
    mm_offset1 = nvId*Sp*Dv + length*Dv                 // vInner 的 GM 偏移

    // ---- 获取 state (BF16) ----
    if FP32 path:
        CrossCoreWaitFlag(0x3)                          // ← AIV: curStateBf16 就绪
        stateForAic = curStateBf16[nvId*Dv*Dk]          // GM BF16
    else:
        stateForAic = (cId==0) ? curStateBf16[stride] : stateOut[nvId*Dv*Dk]

    // ---- matmul 1: v_prime = k_cumdecay @ state^T (accum=1, 叠加到 vInner) ----
    //   A=kCumdecay(C,Dk) transA=false, B=state(Dv,Dk) transB=true
    //   C=vInner(C,Dv), accum=1 → vInner += k_cumdecay @ state^T
    //   即 v_new = v_inner + v_prime
    CalVPrime(kCumdecay[mm_offset0], stateForAic, vInner[mm_offset1])
    //   FP32 path: RunMatmul(mm1Fp32, ..., accum=1)
    //   BF16 path: RunMatmul(mm1Bf16, ..., accum=1)

    // ---- matmul 2: attn_inter = q_prime @ state^T ----
    //   A=qPrime(C,Dk) transA=false, B=state(Dv,Dk) transB=true
    //   C=out(C,Dv), accum=0 (覆盖)
    //   SetOrgShape(M=C, N=Dv, K=Dk, Ka=Dk, Kb=Nv*Dv)
    CalAttnInter(qPrime[mm_offset0], stateForAic,
                 out[nvId*Dv + cId*C*Nv*Dv])

    CrossCoreSetFlag<PIPE_FIX>(0x2)                     // → AIV: state 相关 matmul 完成
    CrossCoreWaitFlag(0x5)                               // ← AIV: vInnerBf16 就绪

    // ---- matmul 3: state_new = v_new^T @ kg (accum=1, 叠加到 state) ----
    //   A=vInnerBf16(C,Dv) transA=true, B=kg(C,Dk) transB=false
    //   C=state(Dv,Dk), accum=1 → state += v_new^T @ kg
    //   即 state_new = state*exp(g[-1]) + v_new^T @ kg
    if FP32 path:
        CalStateNew(vInnerBf16[mm_offset1], kg[mm_offset0], stateOut)
    else:
        CalStateNew(vInner[mm_offset1], kg[mm_offset0], stateOut)
        FIX_MTE2 同步 (state 立即可读供下一 chunk)

    CrossCoreSetFlag<PIPE_FIX>(0x4)                     // → AIV: state 更新完成
```

### 5.5 Stage2 CrossCore 同步关系图

```
AIV                                         AIC
────                                         ────
state CopyIn + Cast → curStateBf16 (GM)
state *= exp(g[-1])
state_out 写回 GM
SetFlag(0x3) ───────────────────────────→   WaitFlag(0x3)
                                             CalVPrime (vInner += kcd@state^T)
                                             CalAttnInter (qPrime@state^T → out)
                                             SetFlag(0x2) ──────────→ WaitFlag(0x2)
CastGmToGm (vInner FP32 → vInnerBf16)        [BF16 path: CopyOutState]
SetFlag(0x5) ───────────────────────────→   WaitFlag(0x5)
                                             CalStateNew (state += v_new^T @ kg)
                                             SetFlag(0x4) ──────────→ WaitFlag(0x4)
[下一 chunk]
```

### 5.6 Stage2 输出汇总

```
GM 更新:
  vInner_     (FP32/BF16, Nv×Sp×Dv)  ← accum: v_inner + k_cumdecay @ state^T = v_new
  vInnerBf16_ (BF16, Nv×Sp×Dv)      ← CastGmToGm: v_new 的 BF16 副本 (FP32 path)
  stateOut_   (FP32/BF16, Nv×Dv×Dk) ← state*exp(g[-1]) + v_new^T @ kg = state_new
  out_        (BF16, T×Nv×Dv)       ← q_prime @ state^T = attn_inter (部分输出)
```

---

## 6 Stage3 数据流伪代码

### 6.1 任务调度

```
Stage3::Process():
    totalChunks = Nv * Nc                              // Nv×Nc 融合调度
    chunksPerCore = Ceil(totalChunks, coreNum)
    for idx in [startChunk, endChunk):
        nvId    = idx / Nc
        chunkId = idx % Nc
        chunkPos = chunkId * C
        curChunkSize = (chunkId == Nc-1) ? lastChunkSize : C

        AIV: CalMaskedQKT → tmpGM
        AIC: masked_qkt @ v_new → attnOut (accum=1)
```

### 6.2 AIV 侧 (CalMaskedQKT)

```
CalMaskedQKT(outGM, nvId, chunkPos):
    if gOptional:
        // ---- 从 GM gCum_ 读取本 chunk 的 g_cum ----
        //   AlignedCopyIn: GM(Nv,Sp) → UB(C) FP32 (自动补齐到 C)
        AlignedCopyIn(gCumExp[nvId*Sp + chunkPos], 1, curChunkSize)
        g_cum = DeQue<FP32>()

        // ---- decay_mask = exp(g_cum[:,None] - g_cum[None,:]) * tril ----
        //   Broadcast g_cum → (C,C) 行方向 (tmpBuffer1)
        //   Broadcast g_cum → (C,C) 列方向 (tmpBuffer2)
        //   Sub: tmpBuffer1 = g_cum[:,None] - g_cum[None,:]
        //   Mul(maskBuffer): * tril
        //   Exp: exp(diff * mask)
        Broadcast(tmpBuffer1, g_cum, (C,C), (C,1))     // 行广播
        Broadcast(tmpBuffer2, g_cum, (C,C), (1,C))     // 列广播
        Sub(tmpBuffer1, tmpBuffer1, tmpBuffer2, curChunkSize*C)
        Mul(tmpBuffer1, tmpBuffer1, maskBuffer, ...)   // * 下三角 mask
        Exp(tmpBuffer1, tmpBuffer1, ...)
        FreeTensor(g_cum)
    else:
        Duplicate(tmpBuffer1, 1.0, curChunkSize*C)     // decay_mask = 1

    // ---- 读取 qkt ----
    //   AlignedCopyIn: GM qkt(Nv,Sp,C) → UB(curChunkSize, C) BF16
    AlignedCopyIn(qkt[nvId*Sp*C + chunkPos*C], curChunkSize, C)
    qkt = DeQue<BF16>()

    // ---- masked_qkt = qkt * scale * decay_mask * mask ----
    //   Cast qkt BF16→FP32 (tmpBuffer2)
    //   Muls(scale)
    //   Mul(decay_mask = tmpBuffer1)
    //   Mul(mask = maskBuffer, 上三角)
    //   Cast BF16 → outQueue
    Cast(tmpBuffer2, qkt, CAST_NONE, curChunkSize*C)
    Muls(tmpBuffer2, tmpBuffer2, scale, ...)
    Mul(tmpBuffer2, tmpBuffer2, tmpBuffer1, ...)       // * decay
    Mul(tmpBuffer2, tmpBuffer2, maskBuffer, ...)        // * 上三角 mask
    Cast(scale_qkt, tmpBuffer2, CAST_RINT, ...)
    EnQue(scale_qkt) → AlignedCopyOut(outGM, curChunkSize, C)
    FreeTensor(qkt)

    // 输出: masked_qkt → tmpGM (GM, coreId×C×C 区域)
```

### 6.3 AIC 侧 (AICProcess)

```
AICProcess(tmpGM, vInner, attnInter):
    // masked_qkt @ v_new → attnOut (accum=1, 叠加 attn_inter)
    //   A=tmpGM(curChunkSize, C) transA=true
    //   B=vInner(curChunkSize, Dv) transB=true → 实际 B^T = v_new (Dv, curChunkSize)
    //   等价: attnOut = masked_qkt @ v_new
    //   C=attnInter(curChunkSize, Dv), accum=1 → attnOut += masked_qkt @ v_new
    //   SetOrgShape(M=curChunkSize, N=Dv, K=curChunkSize, Ka=curChunkSize, Kb=Nv*Dv)
    mm3.SetOrgShape(curChunkSize, Dv, curChunkSize, curChunkSize, Nv*Dv)
    mm3.SetSingleShape(curChunkSize, Dv, curChunkSize)
    mm3.SetTensorA(tmpGM)                    // masked_qkt
    mm3.SetTensorB(vInner)                   // v_new (来自 Stage2 的 vInner/vInnerBf16)
    mm3.IterateAll(attnInter, accum=1)       // attn_out = attn_inter + masked_qkt @ v_new
    mm3.End()
```

### 6.4 Stage3 CrossCore 同步关系图

```
AIV                                          AIC
────                                          ────
CalMaskedQKT → tmpGM
SetFlag(0x4) ────────────────────────────→   WaitFlag(0x4)
                                              AICProcess: masked_qkt @ v_new → attnOut
                                              SetFlag(0x3) ──────────→ WaitFlag(0x3)
[下一 chunk]
```

### 6.5 Stage3 输出汇总

```
GM 输出:
  out_ (BF16, T×Nv×Dv) ← attn_inter + masked_qkt @ v_new = 最终注意力输出 o_t
```

---

## 7 各级存储数据流总表

### 7.1 GM 层 (Global Memory)

| 变量 | dtype | shape | 写入方 | 读取方 | 说明 |
|:-----|:------|:------|:-------|:-------|:-----|
| query_/key_/value_/beta_ | BF16 | T×Nk×Dk / T×Nv×Dv / T×Nv×Dv / T×Nv | 用户输入 | Stage1 | 原始输入 |
| g_ | FP32 | T×Nv | 用户输入 | Stage1 | 衰减系数指数 |
| gCum_ | FP32 | Nv×Sp | Stage1 | Stage2, Stage3 | g.cumsum() |
| kCumDecay_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | A@(-k*beta*g_exp) |
| vInner_ | FP32/BF16 | Nv×Sp×Dv | Stage1→Stage2 | Stage2→Stage3 | v_inner→v_new |
| vInnerBf16_ | BF16 | Nv×Sp×Dv | Stage2 (FP32 path) | Stage2, Stage3 | v_new 的 BF16 |
| qPrime_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | q*scale*g_exp |
| kg_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | k*exp(g[-1]-g_cum) |
| qkt_ | BF16 | Nv×Sp×C | Stage1 | Stage3 | Q@K^T |
| stateOut_ | FP32/BF16 | Nv×Dv×Dk | Stage2 | Stage2 (递推) | state_new |
| out_ | BF16 | T×Nv×Dv | Stage2→Stage3 | 用户输出 | attn_inter→最终输出 |

### 7.2 Stage1 WS 层 (stageWsAddr_)

| 变量 | dtype | shape | 写入方 | 读取方 | 说明 |
|:-----|:------|:------|:-------|:-------|:-----|
| queryConGm_ | BF16 | P×C×Dk | AIV QKPreProcess | AIC matmul 1,2 | Q 连续化 |
| keyConGm_ | BF16 | P×C×Dk | AIV QKPreProcess | AIC matmul 1,2 | K 连续化 |
| gBKWsGm_ | BF16 | P×C×Dk | AIV GBKCompute | AIC matmul 4 | -k*beta*g_exp |
| kkWsGm_ | BF16 | P×C×C | AIC matmul 1 | AIV KKBetaCompute | K@K^T |
| attnWsGm_ | BF16 | P×C×C | AIV InverseCompute | AIC matmul 3,4,5 | 求逆结果 A |
| vBetaWsGm_ | BF16 | P×C×Dv | AIV VBetaCompute | AIC matmul 5 | v*beta |

### 7.3 UB 层 (Unified Buffer, 关键 buffer)

| buffer | dtype | shape | 所在 Stage | 用途 |
|:-------|:------|:------|:-----------|:-----|
| betaUbFloat_ | FP32 | H×P | Stage1 | beta (Cast 后) |
| gCumSumUbFloat_ | FP32 | C×P | Stage1 | g.cumsum() |
| gCumExpUbFloat_ | FP32 | C×P | Stage1 | exp(g.cumsum()) |
| gammaUbFloat_ | FP32 | C×C | Stage1 | decay_mask |
| kkLocal_ | FP32 | H×C | Stage1 | KKT (Cast 后) |
| attnUbFloat_ | FP32 | H×C | Stage1 | KKT*beta*gamma → 求逆输入 |
| inverseRes_ | FP32 | H×C | Stage1 | 求逆结果 A |
| gBKUbFloat_ | FP32 | H×DkAlign | Stage1 | -k*beta*g_exp |
| kgUbFloat_ | FP32 | H×DkAlign | Stage1 | k*exp(g[-1]-g_cum) |
| vBetaUbFloat_ | FP32 | H×DvAlign | Stage1 | v*beta |
| qPrimeUbFloat_ | FP32 | H×DkAlign | Stage1 | q*scale*g_exp |
| stateFp32Ub_ | FP32 | Dv×DkAlign | Stage2 | state (FP32 path) |
| tmpBuffer1_ | FP32 | Dv×DkAlign | Stage2 | state Cast (BF16 path) |
| lastGCum_ | FP32 | 1 | Stage2 | exp(g_cum[-1]) |
| tmpBuffer1_/2_ | FP32 | C×C | Stage3 | decay_mask 计算 |
| maskBuffer_ | FP32 | C×C | Stage3 | 上三角 mask |
