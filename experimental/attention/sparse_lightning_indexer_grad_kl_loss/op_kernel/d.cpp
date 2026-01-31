template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessDeterVector2(SLIGradKLLossRunInfo &runInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_C2_TO_V2_SA_FLAG[runInfo.taskIdMod2]);
    CrossCoreSetFlag<0, PIPE_MTE2>(SYNC_C2_TO_V2_DETER_SA_FLAG_MOD0[runInfo.taskIdMod2]);
    CrossCoreWaitFlag<0, PIPE_MTE2>(SYNC_C2_TO_V2_DETER_SA_FLAG_MOD0[runInfo.taskIdMod2]);

    CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);

    int32_t coreNum = GetBlockNum();
    for (int32_t idx = 0; idx < coreNum; idx++) {

        //重新获取b和S1的值
        int64_t bS1StartIndex = tilingData->multiCoreParams.bS1Index[idx];
        int64_t bS1EndIndex = idx + 1 < optiling::MAX_CORE_NUM ?
            tilingData->multiCoreParams.bS1Index[idx + 1] : tilingData->multiCoreParams.totalSize;
        int64_t bS1Index = bS1StartIndex + runInfo.taskId;
        if (bS1Index >= bS1EndIndex) {
            continue;
        }
        int64_t bIdx, s1Idx;
        int64_t actualSum = 0;
        if constexpr (LAYOUT_T == SLILayout::TND) {
            for (int index = 0; index < constInfo.bSize; index++) {
                int64_t actualLen = this->actualSeqLengthsQueryGm.GetValue(index);
                if (bS1Index < actualLen) {
                    bIdx = index;
                    break;
                }
                actualSum = actualLen;
            }
            s1Idx = bS1Index - actualSum;
        } else {
            bIdx = bS1Index / constInfo.s1Size;
            s1Idx = bS1Index - bIdx * constInfo.s1Size;
        }

        GetRunInfo(runInfo.taskId, bIdx, s1Idx, runInfo);
        
        int32_t v0RealKSize, v1RealKSize, vRealKSize;
        int32_t perCoreKSize, tailCoreKSize, curCoreKSize; 
        int64_t coreKOffset, vCoreKOffset;

        perCoreKSize = CeilDiv(runInfo.kRealSize, coreNum);
        perCoreKSize = SLIGAlign(perCoreKSize, 4);
        int32_t usedCoreNum = CeilDiv(runInfo.kRealSize, perCoreKSize);
        if (constInfo.aicIdx >= usedCoreNum) {
            CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
            CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
            continue;
        }
        tailCoreKSize = runInfo.kRealSize - (usedCoreNum - 1) * perCoreKSize;
        curCoreKSize = constInfo.aicIdx == usedCoreNum - 1 ? tailCoreKSize : perCoreKSize;
        v0RealKSize = CeilDiv(curCoreKSize, 2);
        v0RealKSize = Min(SLIGAlign(v0RealKSize, 2), curCoreKSize);
        v1RealKSize = curCoreKSize - v0RealKSize;

        coreKOffset = constInfo.aicIdx * perCoreKSize;
        if (constInfo.subBlockIdx == 0) {
            vRealKSize = v0RealKSize;
            vCoreKOffset = coreKOffset;
        } else {
            vRealKSize = v1RealKSize;
            vCoreKOffset = coreKOffset + v0RealKSize;
        }
        if (vRealKSize <= 0) {
            CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
            CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
            continue;
        }

        int64_t srcOffset = idx * topKSize * constInfo.dSizeQueryIndex * 2;
        GlobalTensor<MM5_OUT_T> srcGm = bmm5ResGm[srcOffset + (runInfo.taskIdMod2 * topKSize + vCoreKOffset) * constInfo.dSizeQueryIndex];
        LocalTensor<MM5_OUT_T> scatterAddTmpUb;
        int32_t kSplitSize = ubAllocPolicy.scatterAddUbSize / (2 * sizeof(T) * constInfo.dSizeQueryIndex);
        int32_t tailkSize = vRealKSize % kSplitSize;
        int32_t kTailSize = (!tailkSize) ? kSplitSize : tailkSize;
        int32_t kProcessSize = kSplitSize;
        int32_t kLoopTimes = CeilDiv(vRealKSize, kSplitSize);
        int64_t realS2Idx1, realS2Idx2, s2GmOffset;
        event_t eventIdArr[2] = {eventIdScatterAdd, eventIdScatterAddPong};
        runInfo.s2Idx = 0;
        int64_t s2IdxOffset = runInfo.s2Idx * constInfo.s2BaseSize;

        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
        SetAtomicAdd<T>();

        for (int32_t kLoopIdx = 0; kLoopIdx < kLoopTimes; ++kLoopIdx) {
            if (kLoopIdx >= kLoopTimes - 1) {
                kProcessSize = kTailSize;
            }
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
            scatterAddTmpUb = scatterAddUb[scatterAddPingpong * kSplitSize * constInfo.dSizeQueryIndex].template ReinterpretCast<MM5_OUT_T>();
            DataCopy(scatterAddTmpUb, srcGm[kLoopIdx * kSplitSize * constInfo.dSizeQueryIndex], kProcessSize * constInfo.dSizeQueryIndex);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
            constexpr int32_t topKSplitSize = 2;
            int32_t TailSize = kProcessSize % topKSplitSize;
            int32_t topKTailSize = ( !TailSize ) ? topKSplitSize : TailSize;
            int32_t topKProcessSize = topKSplitSize;
            int32_t topKLoopTimes = CeilDiv(kProcessSize, topKSplitSize);
            for (int32_t topKIdx = 0; topKIdx < topKLoopTimes; ++topKIdx) {
                if (topKIdx >= topKLoopTimes - 1) {
                    topKProcessSize = topKTailSize;
                }
                s2GmOffset = vCoreKOffset + kLoopIdx * kSplitSize + topKIdx * topKSplitSize;
                GetRealS2Idx(s2GmOffset, s2IdxOffset, realS2Idx1, runInfo);
                GetRealS2Idx(s2GmOffset + 1, s2IdxOffset, realS2Idx2, runInfo);

                int64_t s2IdLimit = runInfo.s2SparseLen;
                int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
                int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
                bool keyOffset1Pass = (keyOffset1 >= 0);
                bool keyOffset2Pass = (keyOffset2 >= 0);
                if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
                    SetAtomicNone();
                    break;
                }

                int64_t keySrcStride = 0;
                keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
                                (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize) * constInfo.dSizeQueryIndex * sizeof(T);

                bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0);
                bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                    realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);
                bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);

                int64_t ub1Offset = topKIdx * topKSplitSize * constInfo.dSizeQueryIndex;
                int64_t ub2Offset = ub1Offset + constInfo.dSizeQueryIndex;

                if (strideInvalid || copyOutOfRange || key1LessThankey2) {
                    // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
                    ScatterAddCopyOutSingle(scatterAddTmpUb[ub1Offset], keyOffset1);
                    ScatterAddCopyOutSingle(scatterAddTmpUb[ub2Offset], keyOffset2);
                } else {
                    DataCopyExtParams dataCopyParams(
                        keyOffset1Pass + keyOffset1Pass, constInfo.dSizeQueryIndex * sizeof(T), 0, keySrcStride, 0
                    );
                    int64_t keyStartOffset = (keyOffset1Pass) ? keyOffset1 : keyOffset2;
                    int64_t ubStartOffset = (keyOffset1Pass) ? ub1Offset : ub2Offset;
                    LocalTensor<T> srcTmpUb = scatterAddTmpUb.template ReinterpretCast<T>();
                    DataCopyPad(scatterAddResGm[keyStartOffset * constInfo.dSizeQueryIndex], srcTmpUb[ubStartOffset], dataCopyParams);
                }
            }
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
            scatterAddPingpong = 1 - scatterAddPingpong;
        }
        SetAtomicNone();
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
    }
    CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
}