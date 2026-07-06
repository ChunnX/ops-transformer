# mega_moe.h 代码调用接口逻辑图

> 基于文件 `mega_moe.h` (1376行) 的完整函数调用关系与数据流向分析

---

## 一、顶层调用流程总图 (Process 主流程)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Process() — 主执行入口                                │
│                                                                             │
│  ┌─ 阶段0: 源卡数据准备 ──────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  SetCtrlSpr(overflow=0)                                                 │ │
│  │  │                                                                      │ │
│  │  ├─► SendAndQuantBuffInit()   ── 初始化UB buffer                       │ │
│  │  │                                                                      │ │
│  │  ├─► SendMaskCal()            ── 源卡计算mask+count,推送至目标卡win     │ │
│  │  │      │                                                               │ │
│  │  │      ├ CompareScalar()     ── 按全局expertId生成bit mask             │ │
│  │  │      ├ GatherMask()        ── 统计匹配token数(sendCnt)               │ │
│  │  │      ├ SetValue(count)     ── count写入槽位末尾                      │ │
│  │  │      └ DataCopyPad→远端win ── [mask|count]跨卡推送(ping-pong)       │ │
│  │  │                                                                      │ │
│  │  ├─► ResetFlagList()          ── 清零workspace Flag区域                 │ │
│  │  │      │                                                               │ │
│  │  │      ├ Duplicate(resetTensor_, 0)                                    │ │
│  │  │      ├ DataCopyPad→GM Flag区                                         │ │
│  │  │      └─► ResetGmm2CombineSyncCounters()  (仅CombineQuant≠NO_QUANT)  │ │
│  │  │                                                                      │ │
│  │  ├─► QuantProcessInRank()     ── 本卡token MX量化(fp8/fp4)             │ │
│  │  │      │                                                               │ │
│  │  │      ├ DataCopyPad(GM→UB)  ── 搬入bf16 token(双buffer)              │ │
│  │  │      ├ quant::ComputeMaxExp()                                        │ │
│  │  │      ├ quant::ComputeScale()                                         │ │
│  │  │      ├ quant::ComputeFp4Data() / ComputeFp8Data()                    │ │
│  │  │      └ DataCopyPad(UB→GM peermem) ── 搬出量化结果(双buffer)         │ │
│  │  │                                                                      │ │
│  │  SyncAll<false>()             ── AIC等Flag reset完成                    │ │
│  │  CrossRankSyncInWorldSize()   ── 全卡同步(mask+量化就绪)               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ 阶段1: Dispatch + GMM1 + SwiGLU + SwigluQuant ──────────────────────┐  │
│  │                                                                        │  │
│  │  DispatchBuffInit()           ── 初始化dispatch UB buffer             │  │
│  │                                                                        │  │
│  │  ┌── 预调度expert 0 (仅V1/subBlockIdx==1) ─────────────────────────┐ │  │
│  │  │  SendCntCal(0, nextSendCnt)                                     │ │  │
│  │  │      │                                                          │ │  │
│  │  │      ├ DataCopy(win→UB gatherMask)  ── 读[mask|count]槽位      │ │  │
│  │  │      ├ GetValue(count) × worldSize ── 逐卡累加sendCnt          │ │  │
│  │  │      ├ SetValue(cumsum)                                      │ │  │
│  │  │      ├ DataCopy(UB→GM expertRevNums)                         │ │  │
│  │  │      └ AtomicAdd(sendCntFlag)  ── 通知AIC/V0                  │ │  │
│  │  │  │                                                              │ │  │
│  │  │  UpdateGroupParams<kGmm1>(dispatchState, 0, nextSendCnt)       │ │  │
│  │  │  UpdateGlobalBuffer<kGmm1>(dispatchAddrInfo)                   │ │  │
│  │  │  TripleInfoCalAndDispatch(dispatchAddrInfo, 0)                 │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                        │  │
│  │  ┌── 逐expert循环 (localExpertId = 0 .. expertPerRank_) ───────────┐ │  │
│  │  │                                                                  │ │  │
│  │  │  curSendCnt = nextSendCnt  ── dispatch预取计数转发              │ │  │
│  │  │                                                                  │ │  │
│  │  │  ┌─ V1预调度expert e+1 (与GMM1(e)并发) ─────────────────────┐  │ │  │
│  │  │  │  SendCntCal(e+1, nextSendCnt)                             │  │ │  │
│  │  │  │  UpdateGroupParams<kGmm1>(dispatchState, e+1)            │  │ │  │
│  │  │  │  UpdateGlobalBuffer<kGmm1>(dispatchAddrInfo)             │  │ │  │
│  │  │  │  TripleInfoCalAndDispatch(dispatchAddrInfo, e+1)          │  │ │  │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │  │
│  │  │                                                                  │ │  │
│  │  │  ┌─ GMM1 consumer (AIC/V0/V1) 消费expert e ─────────────────┐  │ │  │
│  │  │  │  UpdateGroupParams<kGmm1>(gmm1State, e, curSendCnt)      │  │ │  │
│  │  │  │      │                                                    │  │ │  │
│  │  │  │      ├ if(V0): 自旋等SendCntCal Flag                     │  │ │  │
│  │  │  │      ├ DataCacheCleanAndInvalid()                         │  │ │  │
│  │  │  │      ├ GetValue(expertRevNums) → problemShape.M           │  │ │  │
│  │  │  │      ├ if(V1): 直接使用sendCnt                            │  │ │  │
│  │  │  │      └ 偏移baseOffset(14个字段)                           │  │ │  │
│  │  │  │                                                          │  │ │  │
│  │  │  │  UpdateGlobalBuffer<kGmm1>(gmm1AddrInfo)                 │  │ │  │
│  │  │  │      │                                                    │  │ │  │
│  │  │  │      ├ aGlobal = dispatchRevData + offset               │  │ │  │
│  │  │  │      ├ aScaleGlobal = dispatchRevScale + offset         │  │ │  │
│  │  │  │      ├ bGlobal = weight1 + offset                       │  │ │  │
│  │  │  │      ├ bScaleGlobal = weightScales1 + offset            │  │ │  │
│  │  │  │      ├ gmm1OutGlobal (仅A8W4)                           │  │ │  │
│  │  │  │      └ epilogueOp_.UpdateGlobalAddr()                   │  │ │  │
│  │  │  │                                                          │  │ │  │
│  │  │  │  GroupMatmulWithSwigluQuant(gmm1AddrInfo, gmm1State)    │  │ │  │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │  │
│  │  └───────────────────────────────────────────────────────────────┘ │  │
│  │                                                                        │  │
│  │  EndSync()                    ── GMM1同步结束                         │  │
│  │  ExpertTokenNumCopyOut()      ── V1输出各专家token数(V1 only)        │  │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ 阶段2: GMM2 + Combine ────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  InitCombineBuffers()           ── 初始化Combine UB buffer          │   │
│  │                                                                      │   │
│  │  ┌── 逐expert循环 (expertIdx = 0 .. expertPerRank_) ──────────────┐ │   │
│  │  │                                                                 │ │   │
│  │  │  UpdateGroupParams<kGmm2>(gmm2State, expertIdx)                │ │   │
│  │  │      │                                                          │ │   │
│  │  │      ├ 偏移baseOffset(14个字段)                                 │ │   │
│  │  │      ├ DataCacheCleanAndInvalid()                               │ │   │
│  │  │      └ GetValue(expertRevNums) → problemShape.M                 │ │   │
│  │  │                                                                 │ │   │
│  │  │  UpdateGlobalBuffer<kGmm2>(gmm2AddrInfo)                        │ │   │
│  │  │      │                                                          │ │   │
│  │  │      ├ aGlobal = swigluQuantData + offset                     │ │   │
│  │  │      ├ aScaleGlobal = swigluQuantScale + offset               │ │   │
│  │  │      ├ bGlobal = weight2 + offset                              │ │   │
│  │  │      ├ bScaleGlobal = weightScales2 + offset                  │ │   │
│  │  │      └ gmm2OutGlobal (仅A8W4/A4W4/CombineQuant)               │ │   │
│  │  │                                                                 │ │   │
│  │  │  GroupMatmulWithCombine(gmm2AddrInfo, gmm2State, expertIdx)    │ │   │
│  │  │      │                                                          │ │   │
│  │  │      ├─► GMM2路径分发 (见下方详细图)                            │ │   │
│  │  │      └─► ProcessCombine() (仅CombineQuant≠NO_QUANT && AIV)    │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  EndGMM2Sync()     (仅COMBINE_NO_QUANT)                             │   │
│  │  PipeBarrier + SyncAll<true>()                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ 阶段3: Unpermute ──────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  UnpermuteBuffInit()            ── 初始化Unpermute UB buffer        │   │
│  │  CrossRankSyncInWorldSize()     ── 全卡同步(Combine send完成)       │   │
│  │  Unpermute()                    ── 反排列+加权累加                   │   │
│  │      │                                                              │   │
│  │      ├ DataCopy(GM→UB) ── 搬入专家还回的token                     │   │
│  │      ├ Cast(bf16→fp32)                                             │   │
│  │      ├ Muls(token × topkWeight)                                    │   │
│  │      ├ Add(累加)                                                    │   │
│  │      └ Cast(fp32→bf16) + DataCopy(UB→GM) ── 搬出最终结果         │   │
│  │                                                                      │   │
│  │  SetCtrlSpr(恢复overflow模式)                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、TripleInfoCalAndDispatch 详细调用图

```
TripleInfoCalAndDispatch(gmmAddrInfo, localExpertId)
│
│  priorExpertCumsum = cumsum[localExpertId*worldSize - 1]
│
│  ┌── if ENABLE_A8W4 && localExpertId≠0 ──────────────────────────┐
│  │  CreateVecIndex(topkIndexTensor_)   ── 重建连续索引             │
│  │  PipeBarrier<PIPE_V>                                              │
│  └──────────────────────────────────────────────────────────────────┘
│
│  ┌── 逐rank分组循环 (idx = blockIdx .. worldSize*blockNumPerRank) ─┐
│  │                                                                   │
│  │  dstRankIdx = idx / blockNumPerRank                               │
│  │                                                                   │
│  │  GatherMask(validTopkIndex, topkIndex, gatherMask[dstRankIdx槽位])│
│  │      → sendToCurExpTokenCnt (该rank发给本专家的token数)          │
│  │                                                                   │
│  │  ┌── 数据搬运 (rowsNumInCore > 0) ────────────────────────────┐ │
│  │  │                                                             │ │
│  │  │  分配 tripleTensor_ (UB)                                    │ │
│  │  │                                                             │ │
│  │  │  ─► CopyGMToGMPerToken(rowDstOffset, dstRankIdx,            │ │
│  │  │                        rowSrcIdx, rowsNumInCore)             │ │
│  │  │      │                                                      │ │
│  │  │      ├ Phase1: 三元组组装                                    │ │
│  │  │      │   validTopkIndex.GetValue → topkIndex                │ │
│  │  │      │   tokenIndex = topkIndex / topK                      │ │
│  │  │      │   triple.SetValue(RANK_ID/TOKEN_ID/TOPK_INDEX)      │ │
│  │  │      │                                                      │ │
│  │  │      ├ Phase2 prime: 连发6个MTE2跨卡读                      │ │
│  │  │      │   DataCopy(远端win量化数据 → copyTmpTensors_[0..5]) │ │
│  │  │      │                                                      │ │
│  │  │      ├ Phase2 steady: 6-buffer软流水                        │ │
│  │  │      │   Wait<MTE2_MTE3> → 分离token/scale                  │ │
│  │  │      │   DataCopyPad(token → GM dispatchRevData)            │ │
│  │  │      │   DataCopyPad(scale → GM dispatchRevScale)           │ │
│  │  │      │   Wait<MTE3_MTE2> → DataCopy(下一个token从远端win) │ │
│  │  │      │                                                      │ │
│  │  │      ├ Phase2 drain: 等6个MTE3完成                          │ │
│  │  │      │                                                      │ │
│  │  │      └ Phase3: DataCopy(triple → GM tripleInfo)             │ │
│  │  └─────────────────────────────────────────────────────────────┘ │
│  │                                                                   │
│  │  ┌── Phase2: Flag AtomicAdd (rowsThisCore > 0) ────────────────┐ │
│  │  │                                                             │ │
│  │  │  rowStartLocal = rowDstOffset - priorExpertCumsum            │ │
│  │  │  waveLo..waveHi = 行范围 / L1_TILE_M                       │ │
│  │  │                                                             │ │
│  │  │  for(w = waveLo .. waveHi):                                 │ │
│  │  │      lo/hi = 交集范围                                        │ │
│  │  │      AtomicAdd(flagBase + w, hi - lo)                       │ │
│  │  │        → 通知AIC该wave有多少行dispatch数据就绪              │ │
│  │  └─────────────────────────────────────────────────────────────┘ │
│  └──────────────────────────────────────────────────────────────────┘
```

---

## 三、GroupMatmulWithSwigluQuant 路径分发图

```
GroupMatmulWithSwigluQuant(gmmAddrInfo, state)
│
├── if ENABLE_A8W4 (fp8 act + fp4 w1)
│   └─► MegaMoeImpl::GroupMatmulSwigluQuantA8W4
│        <QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>
│        (epilogueOp_, params_, problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_)
│        │
│        ├ AIC: W4→W8 prologue + MMAD (A8W4路径)
│        └ AIV: BlockEpilogueSwigluMxQuant (SwiGLU + MX量化)
│
├── else if groupedMatmulMode == A8W8_NZ / A4W4_NZ (NZ weight layout)
│   └─► MegaMoeImpl::GroupMatmulSwigluQuant
│        <QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t,
│         QuantScaleOutType, QuantScaleOutType, true=isWeightNZ>
│        │
│        ├ AIC: NZ format GMM (ZN layout)
│        └ AIV: SwiGLU + MX量化 (SwigluQuantOutType自动类型提升)
│
└── else (ND format: fp8/fp4 × fp8/fp4, includes A4W4 ND)
    └─► MegaMoeImpl::GroupMatmulSwigluQuant
         <QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t,
          QuantScaleOutType, QuantScaleOutType, false=isWeightNZ>
         │
         ├ AIC: ND format GMM
         └ AIV: SwiGLU + MX量化
```

---

## 四、GroupMatmulWithCombine 路径分发图

```
GroupMatmulWithCombine(gmmAddrInfo, state, expertIdx)
│
├── if ENABLE_A8W4 || ENABLE_A4W4
│   └─► MegaMoeImpl::GroupMatmul2CombineA8W4
│        <CombineQuantMode, SwigluQuantOutType, Weight1Type, bfloat16_t,
│         QuantScaleOutType, QuantScaleOutType>
│        (params_, problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_,
│         expertBeforeCnt, gmm2PingPongIdx_, expertIdx)
│        │
│        ├ AIC: A8W4 prologue (W4→W8 + MMAD) + Combine量化
│        └ AIV: Combine后处理
│
├── else if groupedMatmulMode == A8W8_NZ
│   └─► MegaMoeImpl::GroupMatmul2
│        <CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
│         QuantScaleOutType, QuantScaleOutType, true=isWeightNZ>
│        │
│        ├ AIC: NZ format GMM2 (ZN layout)
│        └ AIV: Combine后处理
│
└── else (ND format: generic A8W8 / A4W4 ND)
    └─► MegaMoeImpl::GroupMatmul2
         <CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
          QuantScaleOutType, QuantScaleOutType, false=isWeightNZ>
         │
         ├ AIC: ND format GMM2
         └ AIV: Combine后处理
│
└── if CombineQuantMode ≠ COMBINE_NO_QUANT && AIV
    └─► ProcessCombine(gmmAddrInfo, state, expertIdx)
         │
         ├ ComputeCoreGrouping → myGroup/myIdxInGrp/myGrpSize
         ├ 自旋等 gmm2CombineSyncCounter == nTilesPerGroup
         ├ DataCopy(triple from GM → UB)
         ├ PipeBarrier<PIPE_MTE2>
         └─► MegaMoeCombineImpl::CombineTokenGroup
              <CombineQuantMode, bfloat16_t>
              (tokenStart+offset, count, k, expertIdx, rankId,
               gmm2OutGlobal, params_, tripleTensor, ...)
```

---

## 五、AIC / AIV 协作分工图

```
┌──────────────────────────────────────────────────────────────────┐
│                    AIC / AIV 协作模型                             │
│                                                                  │
│  每个 block = 1 AIC(Cube) + 2 AIV(Vector)                       │
│  AIV[0] = subBlockIdx_ == 0  (V0)                               │
│  AIV[1] = subBlockIdx_ == 1  (V1)                               │
│                                                                  │
│  ┌── 阶段0 ───────────────────────────────────────────────────┐ │
│  │                                                              │ │
│  │  AIC:  不参与SendMaskCal/QuantProcessInRank(直接return)     │ │
│  │  AIV0+V1:  共同执行 SendMaskCal + QuantProcessInRank       │ │
│  │  AIV0+V1:  共同执行 ResetFlagList                           │ │
│  │  全核:  SyncAll + CrossRankSyncInWorldSize                  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌── 阶段1 (Dispatch + GMM1) ────────────────────────────────┐ │
│  │                                                              │ │
│  │  ┌─────────┬──────────────────────────────────────────┐    │ │
│  │  │ 角色    │ 职责                                       │    │ │
│  │  ├─────────┼──────────────────────────────────────────┤    │ │
│  │  │ AIV1    │ SendCntCal (count计算+AtomicAdd通知)      │    │ │
│  │  │         │ UpdateGroupParams<kGmm1>(dispatch预取)    │    │ │
│  │  │         │ UpdateGlobalBuffer<kGmm1>(dispatch预取)   │    │ │
│  │  │         │ TripleInfoCalAndDispatch (dispatch搬运)   │    │ │
│  │  │         │ ExpertTokenNumCopyOut (token数输出)       │    │ │
│  │  ├─────────┼──────────────────────────────────────────┤    │ │
│  │  │ AIV0    │ UpdateGroupParams<kGmm1> (等V1的Flag)    │    │ │
│  │  │         │ UpdateGlobalBuffer<kGmm1>                │    │ │
│  │  │         │ (协助GMM1 Epilogue等)                    │    │ │
│  │  ├─────────┼──────────────────────────────────────────┤    │ │
│  │  │ AIC     │ GroupMatmulWithSwigluQuant (矩阵乘+SwiGLU│    │ │
│  │  │         │ + MX量化 Epilogue)                       │    │ │
│  │  └─────────┴──────────────────────────────────────────┘    │ │
│  │                                                              │ │
│  │  同步机制:                                                   │ │
│  │  V1: AtomicAdd(sendCntFlag) → AIC/V0: 自旋等Flag           │ │
│  │  V1: AtomicAdd(dispatchToGmm1Flag[wave]) → AIC: 自旋等Flag │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌── 阶段2 (GMM2 + Combine) ────────────────────────────────┐ │
│  │                                                              │ │
│  │  ┌─────────┬──────────────────────────────────────────┐    │ │
│  │  │ 角色    │ 职责                                       │    │ │
│  │  ├─────────┼──────────────────────────────────────────┤    │ │
│  │  │ AIC     │ GroupMatmul2 (矩阵乘)                     │    │ │
│  │  │         │ A8W4 prologue / generic GMM2              │    │ │
│  │  ├─────────┼──────────────────────────────────────────┤    │ │
│  │  │ AIV0+V1 │ UpdateGroupParams<kGmm2>                 │    │ │
│  │  │         │ UpdateGlobalBuffer<kGmm2>                │    │ │
│  │  │         │ ProcessCombine (仅CombineQuant≠NO_QUANT) │    │ │
│  │  │         │ A8W4下仅V1参与Combine                     │    │ │
│  │  └─────────┴──────────────────────────────────────────┘    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌── 阶段3 (Unpermute) ─────────────────────────────────────┐ │
│  │                                                              │ │
│  │  AIC:  不参与(整个阶段skip)                                  │ │
│  │  AIV0+V1:  共同执行 Unpermute                               │ │
│  │             CrossRankSyncInWorldSize → Unpermute             │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 六、跨卡数据流图 (分布式通信视角)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    跨卡数据流向 (worldSize = N)                          │
│                                                                          │
│  ┌──── Rank 0 ────┐    ┌──── Rank 1 ────┐    ┌──── Rank 2 ────┐       │
│  │                 │    │                 │    │                 │       │
│  │  ┌─ 本卡token ─┐│    │               │    │               │       │
│  │  │  x (bf16)   ││    │               │    │               │       │
│  │  │    │         ││    │               │    │               │       │
│  │  │    ├─►QuantProcessInRank ─► peermem.quantTokenScale     │       │
│  │  │    │  (MX量化fp8/fp4)    (本卡win的量化数据区)          │       │
│  │  │    │                                                     │       │
│  │  │    ├─►SendMaskCal ─► 远端win.maskRecv [mask|count]     │       │
│  │  │    │  (按全局expertId    (推送到每张目标卡的mask接收区) │       │
│  │  │    │   计算mask+count)                                   │       │
│  │  └──────────────┘│    │               │    │               │       │
│  │                   │    │               │    │               │       │
│  │  ─── 全卡同步: CrossRankSyncInWorldSize ──────────────────►│       │
│  │                   │    │               │    │               │       │
│  │  ┌─ 专家接收 ───┐│    │               │    │               │       │
│  │  │               ││    │               │    │               │       │
│  │  │  SendCntCal:  ││    │               │    │               │       │
│  │  │  读本卡win的  ││    │               │    │               │       │
│  │  │  maskRecv区   ││    │               │    │               │       │
│  │  │  → token总数  ││    │               │    │               │       │
│  │  │               ││    │               │    │               │       │
│  │  │  TripleInfoCal││    │               │    │               │       │
│  │  │  AndDispatch: ││    │               │    │               │       │
│  │  │  CopyGMToGM   ││    │               │    │               │       │
│  │  │  PerToken:    ││    │               │    │               │       │
│  │  │  从远端win的  │◄────┼─── peermem    │◄───┼─── peermem   │       │
│  │  │  quantToken   ││    │  .quantToken  ││    │  .quantToken │       │
│  │  │  ScalePtr读取 ││    │  ScalePtr     ││    │  ScalePtr    │       │
│  │  │  量化token    ││    │  (Rank0读取   ││    │  (Rank0读取  │       │
│  │  │               ││    │   Rank1的数据) ││    │   Rank2的数据│       │
│  │  │               ││    │               ││    │               │       │
│  │  │  → workspace  ││    │               ││    │               │       │
│  │  │  .dispatchRev ││    │               ││    │               │       │
│  │  │  Data/Scale   ││    │               ││    │               │       │
│  │  │  + tripleInfo ││    │               ││    │               │       │
│  │  └───────────────┘│    │               │    │               │       │
│  │                   │    │               │    │               │       │
│  │  GMM1+SwiGLU+Quant│    │               │    │               │       │
│  │    │              │    │               │    │               │       │
│  │    ├► swigluQuant ││    │               │    │               │       │
│  │    │  Data/Scale  ││    │               │    │               │       │
│  │    │  (GMM1量化输出│    │               │    │               │       │
│  │    │   → GMM2的A) ││    │               │    │               │       │
│  │    │              │    │               │    │               │       │
│  │  GMM2+Combine     │    │               │    │               │       │
│  │    │              │    │               │    │               │       │
│  │    ├► combineSend │────►│  peermem      │────►│  peermem     │       │
│  │    │  (各专家还回  │    │  .combineSend │    │  .combineSend│       │
│  │    │   的token写  │    │  (Rank0还回给 │    │  (Rank0还回给│       │
│  │    │   到远端win) │    │   Rank1的token│    │   Rank2的token│       │
│  │    │              │    │               │    │               │       │
│  │  ─── 全卡同步 ──────────────────────────────────────────────►│       │
│  │                   │    │               │    │               │       │
│  │  Unpermute:       │    │               │    │               │       │
│  │  读本卡peermem    │◄────┼─── combineSend│◄───┼─── combineSend│       │
│  │  .combineSendPtr  ││    │  (Rank1还回给││    │  (Rank2还回给│       │
│  │  → topk权重累加   ││    │   Rank0的token││    │   Rank0的token│       │
│  │  → 最终输出yOut   ││    │               ││    │               │       │
│  └───────────────────┘    └───────────────┘    └───────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 七、数据生命周期图 (各阶段数据在 GM/UB/win 中的流转)

```
┌────────────────────────────────────────────────────────────────────┐
│  数据生命周期: 从输入到输出                                          │
│                                                                    │
│  GM输入:  x(bf16) ──► QuantProcessInRank ─► peermem.win量化区     │
│           topkIds ──► SendMaskCal ─► peermem.win mask区            │
│           topkWeights ──► UnpermuteBuffInit ─► UB topKWeights     │
│           weight1/2, scales ──► UpdateGlobalBuffer ─► GMM地址视图 │
│                                                                    │
│  ┌── 阶段0 → 阶段1 过渡 ───────────────────────────────────────┐ │
│  │                                                                │ │
│  │  peermem.win量化区 ──► CopyGMToGMPerToken ─► workspace        │ │
│  │  peermem.win mask区 ──► SendCntCal ─► expertRevNums(GM)       │ │
│  │  peermem.win mask区 ──► GatherMask ─► validTopkIndex(UB)      │ │
│  │                                                                │ │
│  │  workspace.dispatchRevData/Scale ──► GMM1的A/A_scale地址       │ │
│  │  workspace.tripleInfo ──► ProcessCombine读取                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌── 阶段1 → 阶段2 过渡 ───────────────────────────────────────┐ │
│  │                                                                │ │
│  │  GMM1+SwiGLU+Quant输出:                                        │ │
│  │    workspace.swigluQuantData/Scale ──► GMM2的A/A_scale地址     │ │
│  │    workspace.gmm1MmadResPtr (仅A8W4)                            │ │
│  │                                                                │ │
│  │  GMM2+Combine输出:                                              │ │
│  │    workspace.gmm2MmadResPtr (仅量化Combine)                     │ │
│  │    peermem.combineSendPtr ──► 写入远端win                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌── 阶段2 → 阶段3 过渡 ───────────────────────────────────────┐ │
│  │                                                                │ │
│  │  peermem.combineSendPtr ──► Unpermute读取                      │ │
│  │    → 读各专家还回的token(含量化token或bf16 token)              │ │
│  │    → DeQuantMxFp8(量化路径) 或 Cast(bf16→fp32)                │ │
│  │    → Muls × topkWeight + Add 累加                              │ │
│  │    → Cast(fp32→bf16) ─► GM yOut                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  GM输出:  yOut(bf16)         ── 最终MoE输出                       │
│           expertTokenNumsOut ── 各专家收到的token数统计             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 八、同步机制汇总图

```
┌────────────────────────────────────────────────────────────────────┐
│                    同步机制类型与使用位置                             │
│                                                                    │
│  ┌── 类型1: AtomicAdd Flag (V1 → AIC/V0) ──────────────────────┐ │
│  │                                                                │ │
│  │  SendCntCal:    AtomicAdd(sendCntFlag) ──► V0自旋等Flag       │ │
│  │  TripleInfo:    AtomicAdd(dispatchToGmm1Flag[wave])           │ │
│  │                              ──► AIC自旋等wave数据就绪         │ │
│  │  使用: UpdateGroupParams<kGmm1> (V0读Flag/等V1)              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌── 类型2: PipeBarrier / SyncFuncStatic (同核内管线同步) ──────┐ │
│  │                                                                │ │
│  │  PipeBarrier<PIPE_ALL>  ── 全管线屏障(所有stage间)            │ │
│  │  PipeBarrier<PIPE_V>    ── Vector计算屏障                     │ │
│  │  PipeBarrier<PIPE_MTE2> ── MTE2搬入屏障                      │ │
│  │  SyncFuncStatic<V_MTE3> ── Vector→MTE3同步                   │ │
│  │  SyncFuncStatic<MTE2_V> ── MTE2→Vector同步                   │ │
│  │  SyncFuncStatic<V_S>    ── Vector→Scalar同步                  │ │
│  │  SyncFuncStatic<S_MTE3> ── Scalar→MTE3同步                   │ │
│  │  SyncFuncStatic<MTE2_S> ── MTE2→Scalar同步                   │ │
│  │  使用: 函数内部各计算步骤间确保数据就绪                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌── 类型3: SetFlag/WaitFlag (buffer级软流水同步) ─────────────┐ │
│  │                                                                │ │
│  │  MTE3_MTE2 ── 搬出完成→搬入就绪 (Unpermute/Quant等双buffer) │ │
│  │  MTE2_V    ── 搬入完成→计算就绪                              │ │
│  │  V_MTE3    ── 计算完成→搬出就绪                              │ │
│  │  MTE3_V    ── 搬出完成→buffer可复用 (SendMaskCal ping-pong) │ │
│  │  MTE2_MTE3 ── 搬入就绪→可搬出 (CopyGMToGM 6-buffer)        │ │
│  │  S_V       ── Scalar→Vector (Unpermute读权重)               │ │
│  │  使用: 所有软流水函数的buffer级事件同步                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌── 类型4: SyncAll (全核同步) ────────────────────────────────┐ │
│  │                                                                │ │
│  │  SyncAll<false> ── 阶段0→1 (AIC等Flag reset)                 │ │
│  │  SyncAll<true>  ── 阶段2→3 (GMM2+Combine完成)               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌── 类型5: CrossRankSyncInWorldSize (跨卡软同步) ─────────────┐ │
│  │                                                                │ │
│  │  阶段0→1: mask+量化数据就绪                                   │ │
│  │  阶段2→3: Combine send完成                                    │ │
│  │  机制: 写远端win同步区 + GmSignalWaitBarrier等所有卡写入      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌── 类型6: EndSync / EndGMM2Sync (GMM计算结束同步) ───────────┐ │
│  │                                                                │ │
│  │  EndSync(vecSetSyncCom_)    ── GMM1全部expert计算结束         │ │
│  │  EndGMM2Sync(...)           ── GMM2全部expert计算结束         │ │
│  │  (仅COMBINE_NO_QUANT路径)                                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌── 类型7: 自旋等待 (轻量级忙等) ────────────────────────────┐ │
│  │                                                                │ │
│  │  UpdateGroupParams: while(ReadGmByPassDCache(flag)==0) {}     │ │
│  │    → V0/AIC等V1的AtomicAdd                                    │ │
│  │  ProcessCombine: while(ReadGmByPassDCache(counter)!=nTiles)   │ │
│  │    → AIV等GMM2的wave计算完成                                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## 九、完整函数调用关系矩阵

```
调用者                    被调用函数
──────────────────────────────────────────────────────────────────────
Process()                │ SendAndQuantBuffInit()
                         │ SendMaskCal()
                         │ ResetFlagList()
                         │ QuantProcessInRank()
                         │ SyncAll<false>()
                         │ CrossRankSyncInWorldSize()
                         │ DispatchBuffInit()
                         │ SendCntCal()
                         │ UpdateGroupParams<kGmm1>()
                         │ UpdateGlobalBuffer<kGmm1>()
                         │ TripleInfoCalAndDispatch()
                         │ GroupMatmulWithSwigluQuant()
                         │ EndSync()
                         │ ExpertTokenNumCopyOut()
                         │ InitCombineBuffers()
                         │ UpdateGroupParams<kGmm2>()
                         │ UpdateGlobalBuffer<kGmm2>()
                         │ GroupMatmulWithCombine()
                         │ EndGMM2Sync()
                         │ SyncAll<true>()
                         │ UnpermuteBuffInit()
                         │ CrossRankSyncInWorldSize() (第二次)
                         │ Unpermute()
──────────────────────────────────────────────────────────────────────
ResetFlagList()          │ ResetGmm2CombineSyncCounters()
──────────────────────────────────────────────────────────────────────
TripleInfoCalAndDispatch │ CopyGMToGMPerToken()
                         │ GatherMask()
                         │ AtomicAdd()
──────────────────────────────────────────────────────────────────────
SendMaskCal()            │ CompareScalar()
                         │ GatherMask()
                         │ DataCopyPad()
──────────────────────────────────────────────────────────────────────
QuantProcessInRank()     │ quant::ComputeMaxExp()
                         │ quant::ComputeScale()
                         │ quant::ComputeFp4Data() / ComputeFp8Data()
                         │ DataCopyPad()
──────────────────────────────────────────────────────────────────────
SendCntCal()             │ DataCopy()
                         │ DataCopyPad()
                         │ AtomicAdd()
──────────────────────────────────────────────────────────────────────
CopyGMToGMPerToken()     │ DataCopy()
                         │ DataCopyPad()
──────────────────────────────────────────────────────────────────────
GroupMatmulWithSwigluQuant│ GroupMatmulSwigluQuantA8W4 (外部)
                          │ GroupMatmulSwigluQuant (外部)
──────────────────────────────────────────────────────────────────────
GroupMatmulWithCombine() │ GroupMatmul2CombineA8W4 (外部)
                         │ GroupMatmul2 (外部)
                         │ ProcessCombine()
──────────────────────────────────────────────────────────────────────
ProcessCombine()         │ ComputeCoreGrouping (外部)
                         │ CombineTokenGroup (外部)
                         │ DataCopy()
──────────────────────────────────────────────────────────────────────
Unpermute()              │ DataCopy()
                         │ Cast()
                         │ Muls()
                         │ Add()
                         │ DeQuantMxFp8 (外部, 仅CombineQuant≠NO_QUANT)
──────────────────────────────────────────────────────────────────────
CrossRankSyncInWorldSize│ WriteGmByPassDCache()
                         │ ReadGmByPassDCache()
                         │ GmSignalWaitBarrier()
──────────────────────────────────────────────────────────────────────
ExpertTokenNumCopyOut() │ DataCopyPad()
                         │ SetValue()/GetValue()
──────────────────────────────────────────────────────────────────────
Init (外部调用)          │ WorkspaceInfo构造
                         │ PeermemInfo构造
                         │ epilogueOp_.Init()
                         │ SetGlobalBuffer (4个GM Tensor)
──────────────────────────────────────────────────────────────────────
```
