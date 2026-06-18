/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify
 * it under the terms of the CANN Open Software License Agreement Version 2.0.
 */

#ifndef MOE_EP_DISPATCH_EPILOGUE_H
#define MOE_EP_DISPATCH_EPILOGUE_H

#include "kernel_operator.h"
#include "kernel_operator_data_copy_intf.h"
#include "moe_ep_dispatch_epilogue_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "../../common/op_kernel/mc2_moe_context.h"
#include "moe_ep_dispatch_epilogue_tiling.h"

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) (((x) + (align)-1) & ~((align)-1))
#endif

namespace MoeEpDispatchEpilogueImpl {

using namespace AscendC;
using Mc2Aclnn::Mc2MoeContext;
using Mc2Aclnn::HCCL_MAX_RANK_SIZE;

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t RECV_META_FIELDS = 3;

template <typename XType, uint32_t IsCached>
class KernelDispatchEpilogue {
public:
    __aicore__ inline KernelDispatchEpilogue() {};
    __aicore__ inline void Init(
        GM_ADDR context,
        GM_ADDR dstBufferSlotIdx,
        GM_ADDR recvSrcMetadataInternal,
        GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
        GM_ADDR recvSrcMetadata,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline GM_ADDR GetBaseWinAddrByRankId(uint32_t rankId)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + Mc2Kernel::A5_MTE_STATE_WIN_SIZE);
    }

    __aicore__ inline void ComputePrefixSums();
    __aicore__ inline void CountHits();
    __aicore__ inline void CopyFromWindowByExpert();
    __aicore__ inline void CopyInputsToOutputs();

    __aicore__ inline void SplitToCore(uint32_t totalCnt, uint32_t &startIdx, uint32_t &endIdx);

    TPipe* tpipe_{nullptr};
    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    uint32_t rankId_{0};
    uint32_t aivId_{0};
    uint32_t recvCounterOffset_{0};
    GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];

    GlobalTensor<int32_t> numRecvPerRankHpGm_;
    GlobalTensor<int64_t> numRecvPerExpertHpGm_;

    GlobalTensor<XType> recvXGm_;
    GlobalTensor<int32_t> recvTopkIdxGm_;
    GlobalTensor<float> recvTopkWeightsGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> cachedRecvSrcMetadataGm_;   // cached 路径专用：来自上一轮 dispatch 的 recv_src_metadata

    GlobalTensor<int32_t> hitCountGm_;

    LocalTensor<XType> ubToken_;
    LocalTensor<int32_t> ubHitCount_;
    LocalTensor<int32_t> ubLocalSeq_;
    LocalTensor<int64_t> ubRowStart_;
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<int32_t> ubTopkIds_;
    LocalTensor<int32_t> ubTargetExpertId_;
    LocalTensor<int32_t> ubSubExpertId_;
    LocalTensor<float> ubReduceWork_;
    LocalTensor<int32_t> ubRecvCnt_;
    LocalTensor<int64_t> ubExpertPfx_;
    LocalTensor<int32_t> ubHitCountFull_;

    TBuf<QuePosition::VECIN> ubTokenBuf_;
    TBuf<QuePosition::VECIN> ubHitCountBuf_;
    TBuf<QuePosition::VECIN> ubLocalSeqBuf_;
    TBuf<QuePosition::VECIN> ubRowStartBuf_;
    TBuf<QuePosition::VECIN> ubMetaBuf_;
    TBuf<QuePosition::VECIN> ubTopkIdsBuf_;
    TBuf<QuePosition::VECIN> ubTargetExpertIdBuf_;
    TBuf<QuePosition::VECIN> ubSubExpertIdBuf_;
    TBuf<QuePosition::VECIN> ubReduceWorkBuf_;
    TBuf<QuePosition::VECIN> ubRecvCntBuf_;
    TBuf<QuePosition::VECIN> ubExpertPfxBuf_;
    TBuf<QuePosition::VECIN> ubHitCountFullBuf_;

    uint32_t expertSum_{0};
    GM_ADDR workspace_{nullptr};
    GM_ADDR localWinBase_{nullptr};
    uint32_t metaOffset_{0};
    uint32_t hitCountOffset_{0};
    uint32_t metaBytes_{0};
    uint32_t paddedMetaElems_{0};
    uint32_t paddedTopkElems_{0};
    uint32_t ubMetaCapElems_{0};   // ubMeta_ 容纳的 int32 元素数（用于 cached 路径分块拷贝）
    uint32_t numLocalExperts_{0};
    uint32_t aivNum_{0};
    uint32_t topK_{0};
    uint32_t hidden_{0};
    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t perSlotBytes_{0};
};

template <typename XType, uint32_t IsCached>
__aicore__ inline void KernelDispatchEpilogue<XType, IsCached>::SplitToCore(
    uint32_t totalCnt, uint32_t &startIdx, uint32_t &endIdx)
{
    uint32_t perAiv = totalCnt / aivNum_;
    uint32_t remainder = totalCnt % aivNum_;
    startIdx = perAiv * aivId_;
    if (aivId_ < remainder) {
        perAiv += 1;
        startIdx += aivId_;
    } else {
        startIdx += remainder;
    }
    endIdx = startIdx + perAiv;
}

template <typename XType, uint32_t IsCached>
__aicore__ inline void KernelDispatchEpilogue<XType, IsCached>::Init(
    GM_ADDR context,
    GM_ADDR dstBufferSlotIdx,
    GM_ADDR recvSrcMetadataInternal,
    GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
    GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
    GM_ADDR recvSrcMetadata,
    GM_ADDR workspace, GM_ADDR tilingGM,
    TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspace_ = workspace;
    numLocalExperts_ = tilingData->cfg.numLocalExperts;
    aivNum_ = tilingData->aivNum;
    topK_ = tilingData->cfg.topK;
    hidden_ = tilingData->cfg.hidden;
    epWorldSize_ = tilingData->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData->cfg.numMaxTokensPerRank;
    perSlotBytes_ = tilingData->cfg.perSlotBytes;

    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    rankId_ = mc2Context_->epRankId;
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
    }

    localWinBase_ = GetBaseWinAddrByRankId(rankId_);
    recvCounterOffset_ = ALIGN_UP(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN);
    metaOffset_ = ALIGN_UP(hidden_ * sizeof(XType), WIN_ADDR_ALIGN);

    numRecvPerRankHpGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRankHp);
    numRecvPerExpertHpGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpertHp);

    recvXGm_.SetGlobalBuffer((__gm__ XType*)recvX);
    recvTopkIdxGm_.SetGlobalBuffer((__gm__ int32_t*)recvTopkIdx);
    recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float*)recvTopkWeights);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadata);

    if constexpr (IsCached) {
        cachedRecvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadataInternal);
    }

    hitCountOffset_ = ALIGN_UP(numLocalExperts_ * sizeof(int64_t), 32);
    uint32_t hitCountElems = aivNum_ * numLocalExperts_;

    hitCountGm_.SetGlobalBuffer((__gm__ int32_t*)(workspace + hitCountOffset_), hitCountElems);

    uint32_t ubTokenBytes     = ALIGN_UP(hidden_ * sizeof(XType), 32);
    uint32_t ubHitCountBytes  = ALIGN_UP(numLocalExperts_ * sizeof(int32_t), 32);
    uint32_t ubLocalSeqBytes  = ALIGN_UP(numLocalExperts_ * sizeof(int32_t), 32);
    uint32_t ubRowStartBytes  = ALIGN_UP(numLocalExperts_ * sizeof(int64_t), 32);

    metaBytes_ = (2 * topK_ + 2) * sizeof(int32_t);
    paddedMetaElems_ = ((2 * topK_ + 2) + 7) / 8 * 8;
    paddedTopkElems_ = (topK_ + 7) / 8 * 8;
    uint32_t maxSlotsPerAiv = (numMaxTokensPerRank_ + aivNum_ - 1) / aivNum_;
    uint32_t ubMetaBytes = ALIGN_UP(maxSlotsPerAiv * paddedMetaElems_ * sizeof(int32_t), 32);
    uint32_t ubTopkIdsBytes = ALIGN_UP(maxSlotsPerAiv * paddedTopkElems_ * sizeof(int32_t), 32);
    ubMetaCapElems_ = ubMetaBytes / sizeof(int32_t);
    uint32_t ubRecvCntBytes = ALIGN_UP(epWorldSize_ * sizeof(int32_t), 32);
    uint32_t ubExpertPfxBytes = ALIGN_UP(numLocalExperts_ * sizeof(int64_t), 32);
    uint32_t ubHitCountFullBytes = ALIGN_UP(aivNum_ * numLocalExperts_ * sizeof(int32_t), 32);

    tpipe_->InitBuffer(ubTokenBuf_, ubTokenBytes);
    tpipe_->InitBuffer(ubHitCountBuf_, ubHitCountBytes);
    tpipe_->InitBuffer(ubLocalSeqBuf_, ubLocalSeqBytes);
    tpipe_->InitBuffer(ubRowStartBuf_, ubRowStartBytes);
    tpipe_->InitBuffer(ubMetaBuf_, ubMetaBytes);
    tpipe_->InitBuffer(ubTopkIdsBuf_, ubTopkIdsBytes);
    tpipe_->InitBuffer(ubTargetExpertIdBuf_, ubTopkIdsBytes);
    tpipe_->InitBuffer(ubSubExpertIdBuf_, ubTopkIdsBytes);
    tpipe_->InitBuffer(ubReduceWorkBuf_, ubTopkIdsBytes);
    tpipe_->InitBuffer(ubRecvCntBuf_, ubRecvCntBytes);
    tpipe_->InitBuffer(ubExpertPfxBuf_, ubExpertPfxBytes);
    tpipe_->InitBuffer(ubHitCountFullBuf_, ubHitCountFullBytes);

    ubToken_     = ubTokenBuf_.Get<XType>();
    ubHitCount_  = ubHitCountBuf_.Get<int32_t>();
    ubLocalSeq_  = ubLocalSeqBuf_.Get<int32_t>();
    ubRowStart_  = ubRowStartBuf_.Get<int64_t>();
    ubMeta_      = ubMetaBuf_.Get<int32_t>();
    ubTopkIds_   = ubTopkIdsBuf_.Get<int32_t>();
    ubTargetExpertId_ = ubTargetExpertIdBuf_.Get<int32_t>();
    ubSubExpertId_ = ubSubExpertIdBuf_.Get<int32_t>();
    ubReduceWork_ = ubReduceWorkBuf_.Get<float>();
    ubRecvCnt_   = ubRecvCntBuf_.Get<int32_t>();
    ubExpertPfx_ = ubExpertPfxBuf_.Get<int64_t>();
    ubHitCountFull_ = ubHitCountFullBuf_.Get<int32_t>();

    DataCopy(ubExpertPfx_, numRecvPerExpertHpGm_, numLocalExperts_);
    PipeBarrier<PIPE_MTE2>();
    uint32_t expertSum = 0;
    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        expertSum += static_cast<uint32_t>(ubExpertPfx_.GetValue(localExpertIdx));
    }
    expertSum_ = expertSum;

    DataCopy(ubRecvCnt_, numRecvPerRankHpGm_, epWorldSize_);
    PipeBarrier<PIPE_MTE2>();
}

template <typename XType, uint32_t IsCached>
__aicore__ inline void KernelDispatchEpilogue<XType, IsCached>::Process()
{
    ComputePrefixSums();

    CountHits();
    SyncAll<true>();

    uint32_t myTotalHits = 0;
    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        myTotalHits += static_cast<uint32_t>(ubHitCount_.GetValue(localExpertIdx));
    }

    if (myTotalHits != 0) {
        CopyFromWindowByExpert();
    }

    if (aivId_ == 0) {
        CopyInputsToOutputs();
    }
}

template <typename XType, uint32_t IsCached>
__aicore__ inline void KernelDispatchEpilogue<XType, IsCached>::ComputePrefixSums()
{
    int64_t cumulativeRowOffset = 0;
    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        int64_t expertTokenCnt = ubExpertPfx_.GetValue(localExpertIdx);
        ubExpertPfx_.SetValue(localExpertIdx, cumulativeRowOffset);
        cumulativeRowOffset += expertTokenCnt;
    }
}

template <typename XType, uint32_t IsCached>
__aicore__ inline void KernelDispatchEpilogue<XType, IsCached>::CountHits()
{
    Duplicate(ubHitCount_, (int32_t)0, numLocalExperts_);

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t slotCnt = ubRecvCnt_.GetValue(rankId);
        if (slotCnt == 0) { continue; }

        uint32_t slotStart, slotEnd;
        SplitToCore(static_cast<uint32_t>(slotCnt), slotStart, slotEnd);
        if (slotStart >= slotEnd) { continue; }

        GM_ADDR srcRankBase = localWinBase_ + recvCounterOffset_
                              + rankId * numMaxTokensPerRank_ * perSlotBytes_;
        uint32_t slotCntPerAiv = slotEnd - slotStart;
        GM_ADDR firstSlotAddrThisAiv = srcRankBase + slotStart * perSlotBytes_;

        GlobalTensor<int32_t> srcTopkIdsGm;
        srcTopkIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(firstSlotAddrThisAiv + metaOffset_));
        uint32_t topkBytes = topK_ * sizeof(int32_t);
        DataCopyExtParams topkCopyParams{static_cast<uint16_t>(slotCntPerAiv),
            static_cast<uint32_t>(topkBytes), static_cast<int64_t>(perSlotBytes_ - topkBytes), 0, 0};
        DataCopyPadExtParams<int32_t> topkPadParams{true, 0,
            static_cast<uint8_t>(paddedTopkElems_ - topK_), -1};
        DataCopyPad(ubTopkIds_, srcTopkIdsGm, topkCopyParams, topkPadParams);
        PipeBarrier<PIPE_MTE2>();

        int32_t calCnt = static_cast<int32_t>(slotCntPerAiv * paddedTopkElems_);
        for (uint32_t localExpertId = 0; localExpertId < numLocalExperts_; ++localExpertId) {
            int32_t targetExpertId =
                static_cast<int32_t>(rankId_ * numLocalExperts_ + localExpertId);

            SyncFunc<AscendC::HardEvent::S_V>();
            Duplicate<int32_t>(ubTargetExpertId_, targetExpertId, calCnt);
            
            Sub(ubSubExpertId_, ubTopkIds_, ubTargetExpertId_, calCnt);
            
            LocalTensor<float> tmpFp32 = ubSubExpertId_.ReinterpretCast<float>();
            LocalTensor<float> tmpOutFp32 = ubTargetExpertId_.ReinterpretCast<float>();
            Abs(tmpOutFp32, tmpFp32, calCnt);
            
            Mins(ubSubExpertId_, ubTargetExpertId_, 1, calCnt);
            
            ReduceSum<float>(tmpOutFp32, tmpFp32, ubReduceWork_, calCnt);
            SyncFunc<AscendC::HardEvent::V_S>();

            int32_t curOtherExpertCnt = ubTargetExpertId_(0);
            int32_t curExpertCnt = (calCnt >= curOtherExpertCnt) ? (calCnt - curOtherExpertCnt) : 0;
            int32_t currentHits = ubHitCount_.GetValue(localExpertId);
            ubHitCount_.SetValue(localExpertId, currentHits + curExpertCnt);
        }
    }

    DataCopy(hitCountGm_[aivId_ * numLocalExperts_], ubHitCount_, numLocalExperts_);
    PipeBarrier<PIPE_MTE3>();
}

template <typename XType, uint32_t IsCached>
__aicore__ inline void KernelDispatchEpilogue<XType, IsCached>::CopyFromWindowByExpert()
{
    Duplicate(ubLocalSeq_, (int32_t)0, numLocalExperts_);

    DataCopy(ubHitCountFull_, hitCountGm_, aivNum_ * numLocalExperts_);
    PipeBarrier<PIPE_MTE2>();

    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        int64_t expertGlobalRowStart = ubExpertPfx_.GetValue(localExpertIdx);
        for (uint32_t aiv = 0; aiv < aivId_; ++aiv) {
            expertGlobalRowStart += ubHitCountFull_.GetValue(aiv * numLocalExperts_ + localExpertIdx);
        }
        ubRowStart_.SetValue(localExpertIdx, expertGlobalRowStart);
    }

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t slotCnt = ubRecvCnt_.GetValue(rankId);
        if (slotCnt == 0) { continue; }

        uint32_t slotStart, slotEnd;
        SplitToCore(static_cast<uint32_t>(slotCnt), slotStart, slotEnd);
        if (slotStart >= slotEnd) { continue; }

        GM_ADDR srcRankBase = localWinBase_ + recvCounterOffset_
                              + rankId * numMaxTokensPerRank_ * perSlotBytes_;
        uint32_t slotCntPerAiv = slotEnd - slotStart;
        GM_ADDR firstSlotAddrThisAiv = srcRankBase + slotStart * perSlotBytes_;

        GlobalTensor<int32_t> srcMetaGm;
        srcMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(firstSlotAddrThisAiv + metaOffset_));
        DataCopyExtParams metaCopyParams{static_cast<uint16_t>(slotCntPerAiv), metaBytes_,
            perSlotBytes_ - metaBytes_, 0, 0};
        DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
        DataCopyPad(ubMeta_, srcMetaGm, metaCopyParams, metaPadParams);
        PipeBarrier<PIPE_MTE2>();

        bool hasPendingWrite = false;

        for (uint32_t localSlot = 0; localSlot < slotCntPerAiv; ++localSlot) {
            uint32_t metaBase = localSlot * paddedMetaElems_;
            GM_ADDR slotAddr = srcRankBase + (slotStart + localSlot) * perSlotBytes_;

            int32_t src_rank_id   = ubMeta_.GetValue(metaBase + 2 * topK_);
            int32_t src_token_idx = ubMeta_.GetValue(metaBase + 2 * topK_ + 1);

            GlobalTensor<XType> srcTokenTensor;
            srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(slotAddr),
                                             hidden_);
            DataCopy(ubToken_, srcTokenTensor, hidden_);
            PipeBarrier<PIPE_MTE2>();

            for (uint32_t topkIdx = 0; topkIdx < topK_; ++topkIdx) {
                int32_t expertId = ubMeta_.GetValue(metaBase + topkIdx);
                if (expertId < 0) { continue; }
                uint32_t dstRank = static_cast<uint32_t>(expertId) / numLocalExperts_;
                if (dstRank != rankId_) { continue; }
                uint32_t localExpertId = static_cast<uint32_t>(expertId) % numLocalExperts_;

                int64_t expertRowStart = ubRowStart_.GetValue(localExpertId);
                int32_t localWrittenRow = ubLocalSeq_.GetValue(localExpertId);
                uint32_t globalRow = static_cast<uint32_t>(expertRowStart + localWrittenRow);

                if (globalRow >= expertSum_) { continue; }

                ubLocalSeq_.SetValue(localExpertId, localWrittenRow + 1);

                DataCopy(recvXGm_[globalRow * hidden_], ubToken_, hidden_);
                DataCopy(recvTopkIdxGm_[globalRow * topK_], ubMeta_[metaBase], topK_);
                hasPendingWrite = true;

                int32_t rawBits = ubMeta_.GetValue(metaBase + topK_ + topkIdx);
                recvTopkWeightsGm_.SetValue(globalRow, *reinterpret_cast<float*>(&rawBits));
                // SetValue 走 scalar cache，需 DCCI 把 cache line 刷回 GM，保证 host / 其他 core 可见
                AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(recvTopkWeightsGm_[globalRow]);

                if constexpr (!IsCached) {
                    recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 0, src_token_idx);
                    recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 1, src_rank_id);
                    recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 2, static_cast<int32_t>(topkIdx));
                    // 同上：3 个 int32（12B）落在同一 cache line 内，单条 DCCI 即可
                    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                        AscendC::DcciDst::CACHELINE_OUT>(recvSrcMetadataGm_[globalRow * RECV_META_FIELDS]);
                }
            }

            if (hasPendingWrite) {
                PipeBarrier<PIPE_MTE3>();
                hasPendingWrite = false;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CopyInputsToOutputs
//   non-cached: recvSrcMetadata 已在 CopyFromWindowByExpert 中由各 aiv 写好，本函数无操作
//   cached:     把 cachedRecvSrcMetadataGm_ 整段拷贝到 recvSrcMetadataGm_
//               总元素 = expertSum_ * RECV_META_FIELDS（int32），用 DataCopyPad 兜底尾部
//               调用方已通过 if (aivId_ == 0) 守卫，本函数体不再判定
// ---------------------------------------------------------------------------
template <typename XType, uint32_t IsCached>
__aicore__ inline void KernelDispatchEpilogue<XType, IsCached>::CopyInputsToOutputs()
{
    PipeBarrier<PIPE_MTE3>();   // 等待 CopyFromWindowByExpert 写完 recvX/recvTopkIdx/recvTopkWeights

    if constexpr (IsCached) {
        const uint32_t totalElems = expertSum_ * RECV_META_FIELDS;
        if (totalElems == 0) {
            return;
        }
        // 复用 ubMeta_（int32），其元素容量 ubMetaCapElems_ 已在 Init 中算好
        // 取容量做分块，向下对齐到 8（int32 → 32B/4）以便对齐路径
        const uint32_t ubElemsCap = ubMetaCapElems_;
        uint32_t chunkElems = (ubElemsCap / 8U) * 8U;
        if (chunkElems == 0) {
            chunkElems = 8U;
        }

        for (uint32_t off = 0; off < totalElems; off += chunkElems) {
            uint32_t copyElems = (off + chunkElems > totalElems) ? (totalElems - off) : chunkElems;
            // 整段（含尾部不对齐）统一用 DataCopyPad，兼容 < 8 elem 余数（实现简单优先）
            DataCopyExtParams copyParams{1U,
                                         static_cast<uint32_t>(copyElems * sizeof(int32_t)),
                                         0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(ubMeta_, cachedRecvSrcMetadataGm_[off], copyParams, padParams);
            PipeBarrier<PIPE_MTE2>();
            DataCopyPad(recvSrcMetadataGm_[off], ubMeta_, copyParams);
            PipeBarrier<PIPE_MTE3>();
        }
    }
}

}  // namespace MoeEpDispatchEpilogueImpl

#endif  // MOE_EP_DISPATCH_EPILOGUE_H
