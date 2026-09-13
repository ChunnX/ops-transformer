#pragma once
#include "kernel_operator.h"
namespace SFAG_KEY_OWNED_FULL_REDUCE {
using namespace AscendC;
constexpr static uint32_t KeyOwnedAivPerAic = 2;
constexpr static uint32_t KeyOwnedMaxAivCoreCount = 48;
static_assert(16704 + KeyOwnedMaxAivCoreCount * 8 * sizeof(int32_t) <= 20480);

template<class T> __aicore__ inline uint32_t KeyOwnedVectorCoreCount(const T *t)
{
    return t->opInfo.castUsedCoreNum;
}

template<class T> __aicore__ inline uint64_t ScatterBase(const T *t) {
    return (t->opInfo.selectedKWorkspaceLen + 4 * t->opInfo.mm12WorkspaceLen) * t->opInfo.usedCoreNum
        + t->opInfo.dqWorkspaceLen + t->opInfo.dkWorkspaceLen + t->opInfo.dvWorkspaceLen;
}
template<class T> __aicore__ inline bool MarkerMatches(const T *t)
{
    uint32_t splitK = 0;
    for (uint32_t i = 0; i < 5; ++i) splitK |= static_cast<uint32_t>(t->opInfo.rsv[i+1]) << i;
    const auto &o = t->opInfo;
    const bool supportedCoreCount = (o.usedCoreNum == 20 || o.usedCoreNum == 24) &&
        KeyOwnedVectorCoreCount(t) == o.usedCoreNum * KeyOwnedAivPerAic;
    return o.rsv[0] && splitK == 21 && o.B == 1 && o.S1 == 1024 && o.S2 == 131072 &&
        o.N2 == 1 && o.G == 64 && o.D == 512 && o.D2 == 512 && o.ropeD == 64 &&
        o.layout == 3 && o.selectedBlockCount == 2048 && o.selectedBlockSize == 1 &&
        o.deterministic && o.kvMerge == 1 && !o.enableOptimizedScatter && o.scaleValue == 1.0f &&
        supportedCoreCount;
}


template<class T> __aicore__ inline bool SafeIndices(GM_ADDR indices, GM_ADDR workspace, const T *t)
{
    GlobalTensor<int32_t> indexGm, statusGm, aggregateGm;
    indexGm.SetGlobalBuffer((__gm__ int32_t *)indices);
    const uint64_t base = ScatterBase(t);
    const uint32_t vectorCoreCount = KeyOwnedVectorCoreCount(t);
    statusGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + base));
    aggregateGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + base +
                                static_cast<uint64_t>(vectorCoreCount) * 8 * sizeof(int32_t)));
    TPipe prePipe;
    TBuf<TPosition::VECCALC> preBuf;
    LocalTensor<int32_t> ix, status;
    bool unsafe = false;
    if ASCEND_IS_AIV {
        prePipe.InitBuffer(preBuf, 20480);
        ix = preBuf.GetWithOffset<int32_t>(2048, 0);
        auto fp = preBuf.GetWithOffset<float>(2048, 8192);
        auto tmp = preBuf.GetWithOffset<float>(64, 16384);
        auto lo = preBuf.GetWithOffset<float>(8, 16640);
        auto hi = preBuf.GetWithOffset<float>(8, 16672);
        status = preBuf.GetWithOffset<int32_t>(KeyOwnedMaxAivCoreCount * 8, 16704);
        for (uint32_t q = GetBlockIdx(); q < 1024; q += vectorCoreCount) {
            DataCopy(ix, indexGm[q * 2048], 2048);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast<float, int32_t>(fp, ix, RoundMode::CAST_NONE, 2048);
            PipeBarrier<PIPE_V>();
            ReduceMin<float>(lo, fp, tmp, 2048, false);
            PipeBarrier<PIPE_V>();
            ReduceMax<float>(hi, fp, tmp, 2048, false);
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
            unsafe |= lo.GetValue(0) < 0.0f || hi.GetValue(0) >= 131072.0f;
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
        for (uint32_t j = 0; j < 8; ++j) status.SetValue(j, j == 0 && unsafe ? 1 : 0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0); WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(statusGm[GetBlockIdx() * 8], status, 8);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    SyncAll<false>(); // Publishes metadata and every active AIV status slot; no output workspace touched.
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            DataCopy(status, statusGm, vectorCoreCount * 8);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            int32_t aggregate = 0;
            for (uint32_t v = 0; v < vectorCoreCount; ++v) aggregate |= status.GetValue(v * 8);
            for (uint32_t j = 0; j < 16; ++j) ix.SetValue(j, j == 0 ? aggregate : 0);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0); WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(aggregateGm, ix, 16);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        DataCopy(ix, aggregateGm, 16);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        unsafe = ix.GetValue(0) != 0;
    }
    if ASCEND_IS_AIC {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(aggregateGm);
        unsafe = aggregateGm.GetValue(0) != 0;
    }
    SyncAll<false>(); // Release prepass only after every coherent control read completes.
    prePipe.Destroy();
    return !unsafe;
}


} // namespace SFAG_KEY_OWNED_FULL_REDUCE
