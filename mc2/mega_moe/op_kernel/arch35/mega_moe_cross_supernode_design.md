# MegaMoE 跨超节点两级通信详细设计书

## 1. 背景与问题

### 1.1 现有架构

当前 MegaMoE（arch35）假设所有 Rank 位于同一超节点内，使用 UB_MEM（peermem）进行跨 Rank 直接内存访问：

- **SendMaskCal**：源卡按全局专家计算 bit-packed mask，通过 peermem 推送到目标专家卡
- **QuantProcessInRank**：本地 Token 量化后写入 peermem
- **SendInfoCal + TripleInfoCalAndDispatch**：从远端 peermem 读取量化 Token 数据
- **CombineTokens**：GMM2 计算结果通过 peermem 写回源卡

### 1.2 跨超节点通信约束

| 约束项 | 说明 |
|--------|------|
| **URMA 单核限制** | 同一时间只有一个核可以给同一个对端 Rank 发送数据 |
| **同层级配对通信** | 跨超节点通信只能发生在相同 local_rank_id 的卡之间。例如超节点0的卡0只能与超节点1的卡0通信 |
| **通信方式差异** | 超节点内使用 UB_MEM（peermem），跨超节点使用 URMA（需要显式 send） |
| **缓冲区限制** | URMA 通信需要 hcomm 通道初始化，每次发送需要独立的 buffer 管理 |

### 1.3 示例拓扑

```
超节点0 (Supernode 0): Rank 0~127
超节点1 (Supernode 1): Rank 128~255

跨超节点配对关系:
  Rank 0  ↔ Rank 128  (local_rank_id=0)
  Rank 1  ↔ Rank 129  (local_rank_id=1)
  ...
  Rank 127 ↔ Rank 255  (local_rank_id=127)
```

## 2. 设计目标

1. **两级通信**：Dispatch 阶段先跨超节点，再超节点内；Combine 阶段先超节点内，再跨超节点
2. **通信效率**：充分利用 URMA 单核限制下的并行能力
3. **兼容性**：尽量复用现有 arch35 的 GMM 计算逻辑
4. **扩展性**：支持任意超节点数量（2~N）

## 3. 整体流程设计

### 3.1 Dispatch 阶段（数据分发到专家）

```
原始流程（单超节点）:
  SendMaskCal → CrossRankSync → SendInfoCal + Dispatch → GMM1+SwiGLU

新流程（跨超节点）:
  Phase 1: 跨超节点通信（URMA）
    ├─ CrossSupernodeSendMask    → 跨超节点推送 mask
    ├─ CrossSupernodeSendQuant   → 跨超节点推送量化 Token 数据
    ├─ CrossSupernodeSync        → 跨超节点同步完成确认
    │
  Phase 2: 超节点内通信（UB_MEM）
    ├─ SendMaskCalIntra          → 超节点内推送 mask（复用现有逻辑）
    ├─ QuantProcessInRank        → 本卡量化写入本地 win（复用现有逻辑）
    ├─ CrossRankSyncInWorldSize  → 超节点内同步（改为仅超节点内范围）
    │
  Phase 3: 数据汇聚 & 计算
    ├─ SendInfoCal               → 汇聚跨超节点 + 超节点内 mask/count
    ├─ TripleInfoCalAndDispatch  → 从本地 win + URMA recv buffer 读取数据
    ├─ GMM1 + SwiGLU + Quant    → 复用现有计算逻辑
```

### 3.2 Combine 阶段（GMM2 结果回传源卡）

```
原始流程（单超节点）:
  GMM2 → CombineTokens（直接写回源卡 peermem）

新流程（跨超节点）:
  Phase 1: 超节点内回传（UB_MEM）
    ├─ CombineTokensIntra        → 超节点内 GMM2 结果写回源卡 win
    │
  Phase 2: 跨超节点回传（URMA）
    ├─ CombineCrossSupernode     → 跨超节点 GMM2 结果发送回源卡
    ├─ CrossSupernodeCombineSync → 跨超节点同步完成确认
    │
  Phase 3: 结果合并
    ├─ Unpermute                 → 合并超节点内 + 跨超节点回传结果
```

### 3.3 时序图

```
时间轴 →

Dispatch Phase 1 (跨超节点 URMA):
  ┌────────────────────────────────────┐
  │ 核0: URMA Send mask+quant → SN1卡0 │
  │ 核1: URMA Send mask+quant → SN1卡1 │
  │ ...                                │
  │ 核K: URMA Send mask+quant → SN1卡K │
  └────────────────────────────────────┘
                     ↓
Dispatch Phase 2 (超节点内 UB_MEM):
  ┌────────────────────────────────────┐
  │ SendMaskCal (超节点内 mask 推送)    │
  │ QuantProcessInRank (本卡量化)      │
  │ CrossRankSyncIntra (超节点内同步)   │
  └────────────────────────────────────┘
                     ↓
Dispatch Phase 3 (数据汇聚+GMM1):
  ┌────────────────────────────────────┐
  │ SendInfoCal (读取所有 mask+count)  │
  │ TripleInfoCalAndDispatch           │
  │ GMM1 + SwiGLU + Quant             │
  └────────────────────────────────────┘
                     ↓
Combine Phase 1 (超节点内回传):
  ┌────────────────────────────────────┐
  │ GMM2 计算                          │
  │ CombineTokensIntra (UB_MEM 回传)   │
  └────────────────────────────────────┘
                     ↓
Combine Phase 2 (跨超节点回传):
  ┌────────────────────────────────────┐
  │ CombineCrossSupernode (URMA 回传)  │
  └────────────────────────────────────┘
                     ↓
Unpermute (结果合并)
```

## 4. 核心数据结构设计

### 4.1 Mc2MoeContext 扩展

```cpp
struct Mc2MoeContext {
    uint32_t epRankId;           // 全局 Rank ID
    uint32_t epRankSize;         // 全局 Rank 总数
    uint32_t supernodeId;        // 超节点 ID (新增)
    uint32_t supernodeSize;      // 每个超节点内的 Rank 数 (新增)
    uint32_t localRankId;        // 超节点内的 local Rank ID (新增)
    uint32_t supernodeNum;       // 超节点总数 (新增)
    uint64_t winSize;            // peermem 窗口大小 (超节点内)
    uint64_t urmaWinSize;        // URMA 窗口大小 (跨超节点, 新增)
    uint64_t epHcclBuffer[1024]; // 超节点内 peermem 地址
    uint64_t urmaBuffer[1024];   // 跨超节点 URMA 地址 (新增)
    uint64_t hcommHandle[1024];  // URMA hcomm 通道句柄 (新增)
};
```

### 4.2 PeermemInfo 扩展

```cpp
struct PeermemInfo {
    // === 超节点内 UB_MEM (peermem) ===
    GM_ADDR rankSyncInWorldPtr;      // 超节点内同步基址
    GM_ADDR maskRecvPtr;            // mask 接收区
    GM_ADDR quantTokenScalePtr;     // 量化 Token 数据区
    GM_ADDR combineSendPtr;         // Combine 回传区

    // === 超节点间 URMA ===
    GM_ADDR urmaRecvBasePtr;        // URMA 接收缓冲区基址 (新增)
    GM_ADDR urmaMaskRecvPtr;        // URMA mask 接收区 (新增)
    GM_ADDR urmaQuantRecvPtr;       // URMA 量化 Token 接收区 (新增)
    GM_ADDR urmaCombineRecvPtr;     // URMA Combine 接收区 (新增)
    GM_ADDR urmaSendBasePtr;        // URMA 发送缓冲区基址 (新增)
};
```

### 4.3 URMA 窗口布局

每个超节点配对的 URMA 通信窗口布局：

```
URMA 窗口 (对端超节点中同 local_rank_id 的卡):

┌─────────────────────────────────────────────┐
│ STATE_SIZE (同步状态区)                       │
├─────────────────────────────────────────────┤
│ mask 区域:                                   │
│   [localExpert_0][srcSN0_localRank0] slot    │
│   [localExpert_1][srcSN0_localRank0] slot    │
│   ...                                        │
│   共 expertPerRank × supernodeNum 个 slot     │
├─────────────────────────────────────────────┤
│ quant Token 区域:                            │
│   每个 Token: [FP8 data | MXFP scale]        │
│   共 maxTokensPerExpert × supernodeNum       │
├─────────────────────────────────────────────┤
│ Combine 回传区域:                             │
│   每个 Token: [GMM2 output | topK index]     │
│   共 maxOutputSize 个 Token                   │
└─────────────────────────────────────────────┘
```

## 5. 核心算法设计

### 5.1 Dispatch - 跨超节点 Mask 推送

#### 问题：URMA 单核发送限制

同一时间只有一个核可以给同一个对端发送。由于跨超节点只能 local_rank_id 配对通信，每个核天然只对应一个对端 Rank，**无需额外仲裁**。

#### 算法：CrossSupernodeSendMask

```
输入: topkIdsTensor_ (本卡所有 Token 的 topk 专家 ID)
输出: 推送 mask 到对端超节点中同 local_rank_id 的所有 Rank

逻辑:
  1. 对每个对端超节点的专家，计算本卡 Token 中属于该专家的 bit-packed mask
  2. 只需要计算对端超节点中 local_rank_id==本卡local_rank_id 的专家
     即: 全局专家 ID = dstSupernodeId * expertPerRank + localExpertId
     其中 dstSupernodeId ∈ {0,1,...,supernodeNum-1} 且 != 本卡超节点
  3. 使用 URMA (hcomm) 推送 mask+count 到对端 Rank
```

```cpp
__aicore__ inline void CrossSupernodeSendMask()
{
    if constexpr(g_coreType == AIC) { return; }

    // 每个 AIV 核对应一个 local_rank_id，天然只有一个 URMA 对端
    // 只有 localRankId_ == aivCoreIdx_ 的核才执行跨超节点发送
    // （因为 URMA 只允许同层级配对通信）

    int32_t myPairRankId = supernodeId_ == 0 ?
        localRankId_ + supernodeSize_ : localRankId_;
    // 例: SN0卡0 → SN1卡128, SN0卡1 → SN1卡129

    // 加载 topk IDs（复用现有逻辑）
    Duplicate<int32_t>(topkIdsTensor_, 0, compareCount_);
    DataCopyPad(topkIdsTensor_, srcGlobalTensor, loadParams, loadPad);

    // 对端超节点的所有专家计算 mask
    int32_t dstSupernodeId = (supernodeId_ == 0) ? 1 : 0;
    int32_t expertStart = dstSupernodeId * expertPerRank_;
    int32_t expertEnd = expertStart + expertPerRank_;

    for (int32_t g = expertStart; g < expertEnd; g++) {
        CompareScalar(maskBuf, topkIdsTensor_, g, CMPMODE::EQ, compareCount_);
        GatherMask(sendGatherOutTensor_, topkIdsTensor_, maskBufU32, true,
            sendTotalNum_, {1, 1, 0, 0}, sendCnt);
        maskBufI32.SetValue(countWordIdx, static_cast<int32_t>(sendCnt));

        // URMA 发送 mask 到对端 Rank
        uint64_t dstOffset = urmaMaskOffset_ +
            (g % expertPerRank_) * maskSlotSize_;
        GlobalTensor<uint8_t> dstGlobalTensor;
        dstGlobalTensor.SetGlobalBuffer(
            (__gm__ uint8_t*)GetUrmaAddrByRankId(myPairRankId) + dstOffset);
        DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
    }
}
```

### 5.2 Dispatch - 跨超节点量化数据推送

#### 算法：CrossSupernodeSendQuant

```
逻辑:
  1. 对本卡量化后的 Token 数据，按 topk 中属于对端超节点专家的 Token
     索引，将量化数据通过 URMA 推送到对端 Rank
  2. 使用 GatherMask 从 topkIdsTensor_ 中筛选属于对端超节点专家的 Token
  3. 逐 Token 通过 URMA 写入对端 Rank 的 quant 接收区
```

```cpp
__aicore__ inline void CrossSupernodeSendQuant()
{
    if constexpr(g_coreType == AIC) { return; }

    int32_t myPairRankId = GetPairRankId();
    int32_t dstSupernodeId = (supernodeId_ == 0) ? 1 : 0;
    int32_t expertStart = dstSupernodeId * expertPerRank_;
    int32_t expertEnd = expertStart + expertPerRank_;

    for (int32_t g = expertStart; g < expertEnd; g++) {
        // 用 GatherMask 找出属于专家 g 的 Token 索引
        CompareScalar(maskBuf, topkIdsTensor_, g, CMPMODE::EQ, compareCount_);
        uint64_t sendCnt = 0;
        GatherMask(gatherOutTensor_, topkIdsTensor_, maskBufU32, true,
            sendTotalNum_, {1, 1, 0, 0}, sendCnt);

        // 逐 Token 从本卡 quant 数据区读取并发送到对端
        for (int32_t i = 0; i < sendCnt; i++) {
            int32_t tokenIdx = gatherOutTensor_.GetValue(i); // Token 在 topk 中的位置
            int32_t srcTokenRow = tokenIdx / topK_;          // 原始 Token 行号

            // 从本卡 peermem quant 区读取
            GlobalTensor<QuantOutType> srcQuant;
            srcQuant.SetGlobalBuffer((__gm__ QuantOutType*)
                params_.peermemInfo.quantTokenScalePtr + srcTokenRow * quantTokenBytes_);

            // 计算对端接收偏移
            uint64_t dstOffset = urmaQuantOffset_ + (g % expertPerRank_) *
                maxTokensPerExpert_ * quantTokenBytes_ + i * quantTokenBytes_;

            GlobalTensor<QuantOutType> dstQuant;
            dstQuant.SetGlobalBuffer((__gm__ QuantOutType*)
                GetUrmaAddrByRankId(myPairRankId) + dstOffset);

            // URMA 拷贝
            DataCopyPad(dstQuant, srcQuant, quantCopyParams);
        }
    }
}
```

### 5.3 Dispatch - 跨超节点同步

#### 算法：CrossSupernodeSync

```
逻辑:
  1. 本卡向对端 Rank 写入状态标记（float 1.0），表示 mask+quant 数据已全部推送
  2. 等待对端 Rank 的状态标记，表示对端已完成向本卡的数据推送
  3. 使用 URMA 状态区进行同步
```

```cpp
__aicore__ inline void CrossSupernodeSync()
{
    if constexpr(g_coreType == AIC) { return; }

    int32_t myPairRankId = GetPairRankId();

    // 1. 向对端发送完成信号
    GlobalTensor<float> dstStatus;
    dstStatus.SetGlobalBuffer((__gm__ float*)
        GetUrmaStateAddrByRankId(myPairRankId));
    Duplicate<float>(statusFp32Tensor_, 1.0f, 1);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(dstStatus, statusFp32Tensor_, statusCopyParams);

    // 2. 等待对端发送完成信号
    GlobalTensor<float> srcStatus;
    srcStatus.SetGlobalBuffer((__gm__ float*)
        GetUrmaStateAddrByRankId(myPairRankId));  // 对端写到本卡的 urma 状态区
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    DataCopyPad(statusFp32Tensor_, srcStatus, statusCopyParams);
    WaitFlag<AscendC::HardEvent::MTE2_S>();
    // 检查状态值 == 1.0f
    float statusVal = statusFp32Tensor_.GetValue(0);
    // 如果 statusVal != 1.0f, 需要轮询等待
}
```

### 5.4 Dispatch - 超节点内通信（复用现有逻辑）

SendMaskCal 和 QuantProcessInRank 的逻辑基本不变，但 **worldSize 缩小为 supernodeSize**：

```
原来: mask 推送给 worldSize 个 Rank
现在: mask 只推送给 supernodeSize 个 Rank（超节点内）

原来: CrossRankSyncInWorldSize 同步 worldSize 个 Rank
现在: CrossRankSyncIntra 同步 supernodeSize 个 Rank（超节点内）
```

### 5.5 Dispatch - 数据汇聚（SendInfoCal 扩展）

#### 算法：SendInfoCal 扩展

```
逻辑:
  1. 读取超节点内 peermem 的 mask+count（复用现有逻辑）
  2. 读取跨超节点 URMA 接收区的 mask+count（新增）
  3. 合并两部分 count，计算 cumsum
  4. 确定每个专家的全局 Token 总数
```

```cpp
__aicore__ inline void SendInfoCal(int32_t localExpertId)
{
    totalSendCntInExp_ = 0;

    // ----- 1. 读取超节点内 peermem mask+count (复用现有) -----
    uint32_t slotWordStride = maskSlotSize_ / sizeof(uint32_t);
    uint32_t countWordIdx = maskAlignSize_ / sizeof(uint32_t);

    GlobalTensor<uint8_t> intraMaskSrcGlobal;
    intraMaskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.maskRecvPtr +
        localExpertId * supernodeSize_ * maskSlotSize_));
    DataCopy(intraMaskTensor_, intraMaskSrcGlobal, supernodeSize_ * maskSlotSize_);

    // ----- 2. 读取跨超节点 URMA mask+count (新增) -----
    GlobalTensor<uint8_t> urmaMaskSrcGlobal;
    urmaMaskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.urmaMaskRecvPtr +
        localExpertId * (supernodeNum_ - 1) * maskSlotSize_));
    DataCopy(urmaMaskTensor_, urmaMaskSrcGlobal, (supernodeNum_ - 1) * maskSlotSize_);

    // ----- 3. 逐源卡读取 count + cumsum -----
    // 超节点内
    for (int32_t rankId = 0; rankId < supernodeSize_; rankId++) {
        int32_t sendCnt = intraMaskInt32Tensor_.GetValue(rankId * slotWordStride + countWordIdx);
        totalSendCntInExp_ += sendCnt;
        cumsumRevCntInRank_ += sendCnt;
        cumsumInfoTensor_.SetValue(localExpertId * supernodeSize_ + rankId,
            static_cast<int32_t>(cumsumRevCntInRank_));
    }
    // 跨超节点
    for (int32_t snIdx = 0; snIdx < supernodeNum_ - 1; snIdx++) {
        int32_t sendCnt = urmaMaskInt32Tensor_.GetValue(snIdx * slotWordStride + countWordIdx);
        totalSendCntInExp_ += sendCnt;
        cumsumRevCntInRank_ += sendCnt;
        // 全局 cumsum 索引: localExpertId * supernodeSize_ + supernodeSize_ + snIdx
        cumsumInfoTensor_.SetValue(
            localExpertId * worldSize_ + supernodeSize_ + snIdx * supernodeSize_ + localRankId_,
            static_cast<int32_t>(cumsumRevCntInRank_));
    }

    // 写入专家接收总数并通知 AIC
    expertTokenCntTensor_.SetValue(0, totalSendCntInExp_);
    DataCopy(expertRevNumsGloablTensor_[...], expertTokenCntTensor_, ...);
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(2);
}
```

### 5.6 Dispatch - 数据搬运（TripleInfoCalAndDispatch 扩展）

#### 算法：TripleInfoCalAndDispatch 扩展

```
逻辑:
  1. 超节点内的 Token: 从 peermem 读取（复用 CopyGMToGMPerToken）
  2. 跨超节点的 Token: 从 URMA 接收区读取（新增 CopyUrmaToWorkspace）
  3. 两部分数据合并写入 workspace 的 dispatch 区
```

```cpp
__aicore__ inline void CopyUrmaToWorkspace(
    int32_t rowDstOffset, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum)
{
    // 从 URMA 接收区读取量化 Token+Scale 数据到 workspace
    GlobalTensor<QuantOutType> urmaSrc;
    urmaSrc.SetGlobalBuffer((__gm__ QuantOutType*)
        params_.peermemInfo.urmaQuantRecvPtr +
        remoteRankIdx * maxTokensPerExpert_ * quantTokenBytes_ +
        copyStartIdx * quantTokenBytes_);

    GlobalTensor<QuantOutType> dstToken;
    dstToken.SetGlobalBuffer((__gm__ QuantOutType*)
        params_.workspaceInfo.dispatchRevDataPtr + rowDstOffset * widthA);

    // URMA 接收区已经在本地内存，使用普通 DataCopy 读取
    DataCopyPad(dstToken, urmaSrc, copyParams);
}
```

### 5.7 Combine - 超节点内回传（复用现有逻辑）

CombineTokensIntra 与现有 CombineTokens 逻辑相同，但 **只写回超节点内的源卡**：

```cpp
// 只写回超节点内的 Rank（通过 peermem）
for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
    uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
    if (IsInSameSupernode(toRankId)) {
        // 超节点内: 直接 peermem 写
        gmRemoteD.SetGlobalBuffer((__gm__ ElementMMadOut2*)
            GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset));
        DataCopyPad(gmRemoteD[gmDstOffset], l0cOutUbGMM2[...], ub2GmParams);
    }
}
```

### 5.8 Combine - 跨超节点回传

#### 算法：CombineCrossSupernode

```
逻辑:
  1. 从 tripleTensor 中筛选目标 Rank 属于对端超节点的 Token
  2. 将这些 Token 的 GMM2 结果通过 URMA 发送回对端 Rank
  3. 由于 URMA 单核限制，每个核只向一个对端 Rank 发送
```

```cpp
__aicore__ inline void CombineCrossSupernode()
{
    if constexpr(g_coreType == AIC) { return; }

    int32_t myPairRankId = GetPairRankId();

    // 遍历 tripleTensor，筛选目标为对端超节点的 Token
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
        if (!IsInSameSupernode(toRankId) &&
            GetLocalRankId(toRankId) == localRankId_) {
            // 跨超节点且配对的 Rank: 通过 URMA 发送

            uint64_t dstOffset = urmaCombineOffset_ +
                tripleTensor.GetValue(tileIdx * 8 + 1) * topK_ * n_ +
                tripleTensor.GetValue(tileIdx * 8 + 2) * n_ + nLoc;

            GlobalTensor<ElementMMadOut2> dstGlobal;
            dstGlobal.SetGlobalBuffer((__gm__ ElementMMadOut2*)
                GetUrmaAddrByRankId(myPairRankId) + dstOffset);
            DataCopyPad(dstGlobal, l0cOutUbGMM2[...], ub2GmParams);
        }
    }
}
```

### 5.9 Combine - 跨超节点同步 & Unpermute

```
逻辑:
  1. 跨超节点 Combine 完成后，发送状态信号通知对端
  2. 等待对端 Combine 完成的状态信号
  3. 从 URMA Combine 接收区读取数据，合并到最终输出
```

## 6. Process 函数整体流程

```cpp
__aicore__ inline void Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();

    // ========== Dispatch 阶段 ==========

    BuffInit();

    // Phase 1: 跨超节点通信 (URMA)
    CrossSupernodeSendMask();      // 推送 mask 到对端超节点
    CrossSupernodeSendQuant();     // 推送量化 Token 到对端超节点
    CrossSupernodeSync();          // 跨超节点同步确认

    // Phase 2: 超节点内通信 (UB_MEM)
    SendMaskCal();                 // 超节点内 mask 推送 (复用，worldSize→supernodeSize)
    ResetFlagList();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    QuantProcessInRank();          // 本卡量化写入 peermem (复用)
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    CrossRankSyncIntra();          // 超节点内同步 (复用，范围缩小)

    // Phase 3: 数据汇聚 & GMM1+SwiGLU
    FirstBuffInit();
    GMMAddrInfo gmmAddrInfo;
    for (int localExpertId = 0; localExpertId < expertPerRank_; localExpertId++) {
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                SendInfoCal(localExpertId);  // 合并超节点内+跨超节点 mask/count
            }
        }
        if (!UpdateGroupParams(localExpertId, 0)) { continue; }
        UpdateGlobalBuffer(gmmAddrInfo, 0, localExpertId);
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                TripleInfoCalAndDispatch(gmmAddrInfo, localExpertId);
                // 内部: 超节点内从 peermem 读取，跨超节点从 URMA recv 读取
            }
        }
        if (subBlockIdx_ == 0) {
            GroupMatmulSwigluQuant(...);  // 计算逻辑不变
        }
    }
    EndSync(vecSetSyncCom_);

    // ========== Combine 阶段 ==========

    preOffset_ = 0;
    vecSetSyncCom_ = 0;
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx, 1)) { continue; }
        UpdateGlobalBuffer(gmmAddrInfo, 1, expertIdx);
        if (subBlockIdx_ == 0) {
            GroupMatmul2Combine(...);  // GMM2 计算逻辑不变
            // CombineTokensIntra 内部只写超节点内的 Rank
            // CombineCrossSupernode 内部通过 URMA 写对端 Rank
        }
    }
    EndGMM2Sync(vecSetSyncCom_, gmm2PingPongIdx_);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // 跨超节点 Combine 同步
    CrossSupernodeCombineSync();

    // 结果合并 & Unpermute
    if constexpr(g_coreType == AIV) {
        Unpermute();  // 合并超节点内+跨超节点回传结果
    }

    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
```

## 7. URMA 通信核分配策略

### 7.1 单核发送限制下的核分配

由于 URMA 同一时间只有一个核可以给同一对端发送，且跨超节点只能配对通信（local_rank_id 相同），核分配策略如下：

```
假设: supernodeSize=128, aivNum=16

每个超节点内 128 张卡，每张卡有 16 个 AIV 核

跨超节点通信核分配:
  - 每个 AIV 核天然对应一个 local_rank_id
  - aivCoreIdx_ 0~15 分别负责与对端超节点的卡 0~15 通信
  - 核 0 → 对端卡 0 (local_rank_id=0)
  - 核 1 → 对端卡 1 (local_rank_id=1)
  - ...
  - 核 15 → 对端卡 15 (local_rank_id=15)

对于 local_rank_id > 15 的卡（16~127）:
  - 这些卡的跨超节点通信由其本卡的 AIV 核负责
  - 每张卡的 AIV 核只负责自己配对的超节点间通信
```

### 7.2 核分工

```
AIV 核分工:
  subBlock 0: GMM 计算（不变）
  subBlock 1:
    ├─ 跨超节点 URMA 发送/接收 (aivCoreIdx_ < supernodeSize 时)
    ├─ 超节点内 peermem 数据搬运 (复用)
    └─ Combine 回传
```

## 8. 数据布局详细设计

### 8.1 超节点内 Peermem 窗口（不变）

```
布局与现有 arch35 一致，只是 worldSize 替换为 supernodeSize:

┌─────────────────────────────────────────────┐
│ 同步状态区 (STATE_SIZE)                      │
├─────────────────────────────────────────────┤
│ mask 区域: [localExpert][srcRank_in_SN]      │
│   expertPerRank × supernodeSize 个 slot       │
├─────────────────────────────────────────────┤
│ quant Token 区域:                            │
│   每个 Token: [FP8 data | MXFP scale]        │
├─────────────────────────────────────────────┤
│ Combine 回传区域:                             │
│   [srcRank_in_SN][tokenIdx][topKIdx]         │
└─────────────────────────────────────────────┘
```

### 8.2 跨超节点 URMA 窗口（新增）

```
本卡的 URMA 窗口（对端超节点中同 local_rank_id 的卡写入）:

┌─────────────────────────────────────────────┐
│ URMA 状态区 (用于跨超节点同步)               │
│   每个 supernodeNum-1 个对端一个状态位        │
├─────────────────────────────────────────────┤
│ URMA mask 区域:                              │
│   [localExpert][srcSupernodeId] slot          │
│   expertPerRank × (supernodeNum-1) 个 slot    │
├─────────────────────────────────────────────┤
│ URMA quant Token 区域:                       │
│   每个 Token: [FP8 data | MXFP scale]        │
│   maxTokensPerExpert × (supernodeNum-1)       │
├─────────────────────────────────────────────┤
│ URMA Combine 回传区域:                       │
│   [tokenIdx][topKIdx] × maxOutputSize         │
└─────────────────────────────────────────────┘
```

### 8.3 Workspace 布局

```
Dispatch workspace:
┌─────────────────────────────────────────────┐
│ dispatchRevDataPtr:                          │
│   [超节点内 Token data | 跨超节点 Token data] │
│   超节点内排在前面，跨超节点排在后面            │
├─────────────────────────────────────────────┤
│ dispatchRevScalePtr:                         │
│   [超节点内 Token scale | 跨超节点 Token scale]│
├─────────────────────────────────────────────┤
│ cumsumInfo:                                  │
│   超节点内 cumsum + 跨超节点 cumsum            │
│   [localExpert × (supernodeSize + SN-1×1)]   │
├─────────────────────────────────────────────┤
│ groupList / expertRevNums:                   │
│   全局专家 Token 数 = 超节点内 + 跨超节点       │
└─────────────────────────────────────────────┘
```

## 9. 同步机制设计

### 9.1 同步时序

```
完整同步流程:

1. 跨超节点 URMA 同步 (CrossSupernodeSync):
   ├─ 本卡写状态 → 对端超节点卡
   ├─ 等待对端超节点卡写状态 → 本卡
   └─ 确认跨超节点 mask+quant 数据已全部到达

2. 超节点内 UB_MEM 同步 (CrossRankSyncIntra):
   ├─ 超节点内 mask 推送完成
   ├─ 本卡量化数据写入 peermem 完成
   └─ 超节点内所有 Rank 同步

3. AIC-AIV 同步 (CrossCoreSetFlag):
   ├─ AIV 通知 AIC: 数据就绪，可以开始 GMM1
   └─ AIC 通知 AIV: GMM2 完成，可以 Combine

4. 跨超节点 Combine 同步 (CrossSupernodeCombineSync):
   ├─ 本卡写状态 → 对端超节点卡
   ├─ 等待对端超节点卡写状态 → 本卡
   └─ 确认跨超节点 Combine 数据已全部到达
```

### 9.2 同步容错

- **超时机制**：URMA 同步需要超时保护，防止对端超节点故障导致死等
- **状态检查**：每次同步前检查 URMA 状态区的有效性
- **双缓冲**：URMA 状态区使用双缓冲，避免读写冲突

## 10. 性能优化设计

### 10.1 通信与计算重叠

```
优化策略: 跨超节点通信与超节点内计算并行

时间轴:
  t0: 跨超节点 mask+quant 发送 (URMA)
  t1: 超节点内 mask 推送 + 本卡量化 (UB_MEM) ← 与 t0 并行
  t2: 超节点内同步 + 数据汇聚
  t3: GMM1+SwiGLU 计算
  t4: GMM2 计算 + 超节点内 Combine (UB_MEM)
  t5: 跨超节点 Combine 回传 (URMA)
```

### 10.2 URMA 批量发送

由于 URMA 单核限制，每个核只向一个对端发送，可以**批量发送**减少通信次数：

```
优化: 将同一专家的所有 Token mask+quant 数据合并为一次 URMA 发送

原始: 逐 Token 发送 → N 次 URMA 操作
优化: 按专家批量发送 → expertPerRank 次 URMA 操作
```

### 10.3 数据压缩

- **Mask 压缩**：bit-packed mask 已经是高效格式
- **Quant 数据紧凑排列**：FP8 data + MXFP scale 拼接存储
- **URMA 发送合并**：同一专家的 mask 和 quant 数据一起发送

## 11. 多超节点扩展（>2 个超节点）

### 11.1 配对关系

```
supernodeNum = N (N ≥ 2)

每个超节点的卡 i 需要与所有其他超节点的卡 i 通信:
  SN0 卡0 → SN1 卡0, SN2 卡0, ..., SN(N-1) 卡0
  SN0 卡1 → SN1 卡1, SN2 卡1, ..., SN(N-1) 卡1
  ...

URMA 单核限制: 同一时间只能向一个对端发送
解决: 串行发送，按超节点 ID 顺序:
  先发 SN1, 再发 SN2, ..., 最后发 SN(N-1)
```

### 11.2 流程调整

```
CrossSupernodeSendMask/Quant:
  for (int32_t snIdx = 0; snIdx < supernodeNum_; snIdx++) {
      if (snIdx == supernodeId_) continue;
      int32_t pairRankId = snIdx * supernodeSize_ + localRankId_;
      // 向 pairRankId 发送 mask+quant
  }

CrossSupernodeSync:
  等待所有其他超节点的配对卡写入状态

SendInfoCal:
  合并超节点内 + 所有其他超节点的 mask+count
```

## 12. Tiling 参数扩展

### 12.1 新增 Tiling 参数

```cpp
struct MegaMoeTilingData {
    // 现有参数 ...
    uint32_t expertPerRank;
    uint32_t bs;
    uint32_t h;
    uint32_t hiddenDim;
    uint32_t epWorldSize;
    uint32_t topK;

    // 新增参数
    uint32_t supernodeId;         // 当前超节点 ID
    uint32_t supernodeSize;       // 超节点内 Rank 数
    uint32_t supernodeNum;        // 超节点总数
    uint32_t localRankId;         // 超节点内 local Rank ID
    uint32_t urmaWinSize;         // URMA 窗口大小
    uint32_t maxTokensPerExpertCrossSN; // 跨超节点每个专家最大 Token 数
    uint32_t crossSNBufferNum;    // 跨超节点通信缓冲区数量
};
```

## 13. 错误处理与边界情况

| 场景 | 处理方式 |
|------|---------|
| 超节点数为 1 | 退化为现有单超节点逻辑，不执行 URMA 通信 |
| 跨超节点 Token 数为 0 | 跳过 URMA 发送，只做超节点内通信 |
| URMA 通信失败 | 超时检测 + 状态标记错误上报 |
| 某个超节点故障 | 跳过该超节点的通信，其他超节点继续 |
| Token 数超过 URMA 缓冲区 | 分批发送，每批大小由 tiling 参数控制 |

## 14. 总结

| 项目 | 方案 |
|------|------|
| **通信架构** | 两级通信：跨超节点 URMA + 超节点内 UB_MEM |
| **Dispatch 流程** | 跨超节点→超节点内→数据汇聚→GMM1+SwiGLU |
| **Combine 流程** | GMM2→超节点内回传→跨超节点回传→Unpermute |
| **URMA 限制应对** | 配对通信 + 串行发送 + 批量合并 |
| **核分配** | 每个 AIV 核负责一个 local_rank_id 的跨超节点通信 |
| **数据布局** | peermem 超节点内 + URMA 跨超节点，两部分独立布局 |
| **同步机制** | 四级同步：跨超节点→超节点内→AIC-AIV→跨超节点Combine |
| **扩展性** | 支持 2~N 个超节点，串行发送避免单核冲突 |