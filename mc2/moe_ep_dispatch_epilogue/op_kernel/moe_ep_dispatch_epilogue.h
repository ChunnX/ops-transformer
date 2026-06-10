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

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];

__aicore__ inline GM_ADDR GetBaseWinAddrByRankId(uint32_t rankId)
{
    return (GM_ADDR)(winRankAddr_[rankId] + Mc2Kernel::A5_MTE_STATE_WIN_SIZE);
}

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t RECV_META_FIELDS = 3;

class KernelDispatchEpilogue {
public:
    __aicore__ inline KernelDispatchEpilogue() {};
    __aicore__ inline void Init(
        GM_ADDR context, GM_ADDR dstBufferSlotIdx,
        GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
        GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData);
    __aicore__ inline void Process();

private:
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

    GlobalTensor<int32_t> numRecvPerRankHpGm_;
    GlobalTensor<int64_t> numRecvPerExpertHpGm_;

    GlobalTensor<bfloat16_t> recvXGm_;
    GlobalTensor<int32_t> recvTopkIdxGm_;
    GlobalTensor<float> recvTopkWeightsGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;

    GlobalTensor<int32_t> hitCountGm_;

    LocalTensor<bfloat16_t> ubToken_;
    LocalTensor<int32_t> ubHitCount_;
    LocalTensor<int32_t> ubLocalSeq_;
    LocalTensor<int64_t> ubRowStart_;
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<int32_t> ubRecvCnt_;
    LocalTensor<int64_t> ubExpertPfx_;
    LocalTensor<int32_t> ubHitCountFull_;

    TBuf<QuePosition::VECIN> ubTokenBuf_;
    TBuf<QuePosition::VECIN> ubHitCountBuf_;
    TBuf<QuePosition::VECIN> ubLocalSeqBuf_;
    TBuf<QuePosition::VECIN> ubRowStartBuf_;
    TBuf<QuePosition::VECIN> ubMetaBuf_;
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
    uint32_t numLocalExperts_{0};
    uint32_t aivNum_{0};
    uint32_t topK_{0};
    uint32_t hidden_{0};
    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t perSlotBytes_{0};
};

__aicore__ inline void KernelDispatchEpilogue::SplitToCore(
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

__aicore__ inline void KernelDispatchEpilogue::Init(
    GM_ADDR context, GM_ADDR dstBufferSlotIdx,
    GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
    GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
    GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
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
    metaOffset_ = ALIGN_UP(hidden_ * sizeof(bfloat16_t), WIN_ADDR_ALIGN);

    numRecvPerRankHpGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRankHp);
    numRecvPerExpertHpGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpertHp);

    recvXGm_.SetGlobalBuffer((__gm__ bfloat16_t*)recvX);
    recvTopkIdxGm_.SetGlobalBuffer((__gm__ int32_t*)recvTopkIdx);
    recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float*)recvTopkWeights);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadata);
    numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpert);

    hitCountOffset_ = ALIGN_UP(numLocalExperts_ * sizeof(int64_t), 32);

    hitCountGm_.SetGlobalBuffer((__gm__ int32_t*)(workspace + hitCountOffset_));

    uint32_t ubTokenBytes     = ALIGN_UP(hidden_ * sizeof(bfloat16_t), 32);
    uint32_t ubHitCountBytes  = ALIGN_UP(numLocalExperts_ * sizeof(int32_t), 32);
    uint32_t ubLocalSeqBytes  = ALIGN_UP(numLocalExperts_ * sizeof(int32_t), 32);
    uint32_t ubRowStartBytes  = ALIGN_UP(numLocalExperts_ * sizeof(int64_t), 32);

    metaBytes_ = (2 * topK_ + 2) * sizeof(int32_t);
    paddedMetaElems_ = ((2 * topK_ + 2) + 7) / 8 * 8;
    uint32_t maxSlotsPerAiv = (numMaxTokensPerRank_ + aivNum_ - 1) / aivNum_;
    uint32_t ubMetaBytes = ALIGN_UP(maxSlotsPerAiv * paddedMetaElems_ * sizeof(int32_t), 32);
    uint32_t ubRecvCntBytes = ALIGN_UP(epWorldSize_ * sizeof(int32_t), 32);
    uint32_t ubExpertPfxBytes = ALIGN_UP(numLocalExperts_ * sizeof(int64_t), 32);
    uint32_t ubHitCountFullBytes = ALIGN_UP(aivNum_ * numLocalExperts_ * sizeof(int32_t), 32);

    tpipe_->InitBuffer(ubTokenBuf_, ubTokenBytes);
    tpipe_->InitBuffer(ubHitCountBuf_, ubHitCountBytes);
    tpipe_->InitBuffer(ubLocalSeqBuf_, ubLocalSeqBytes);
    tpipe_->InitBuffer(ubRowStartBuf_, ubRowStartBytes);
    tpipe_->InitBuffer(ubMetaBuf_, ubMetaBytes);
    tpipe_->InitBuffer(ubRecvCntBuf_, ubRecvCntBytes);
    tpipe_->InitBuffer(ubExpertPfxBuf_, ubExpertPfxBytes);
    tpipe_->InitBuffer(ubHitCountFullBuf_, ubHitCountFullBytes);

    ubToken_     = ubTokenBuf_.Get<bfloat16_t>();
    ubHitCount_  = ubHitCountBuf_.Get<int32_t>();
    ubLocalSeq_  = ubLocalSeqBuf_.Get<int32_t>();
    ubRowStart_  = ubRowStartBuf_.Get<int64_t>();
    ubMeta_      = ubMetaBuf_.Get<int32_t>();
    ubRecvCnt_   = ubRecvCntBuf_.Get<int32_t>();
    ubExpertPfx_ = ubExpertPfxBuf_.Get<int64_t>();
    ubHitCountFull_ = ubHitCountFullBuf_.Get<int32_t>();

    DataCopy(ubExpertPfx_, numRecvPerExpertHpGm_, numLocalExperts_);
    PipeBarrier<PIPE_MTE2>();
    uint32_t expertSum = 0;
    for (uint32_t e = 0; e < numLocalExperts_; ++e) {
        expertSum += static_cast<uint32_t>(ubExpertPfx_.GetValue(e));
    }
    expertSum_ = expertSum;

    DataCopy(ubRecvCnt_, numRecvPerRankHpGm_, epWorldSize_);
    PipeBarrier<PIPE_MTE2>();
}

__aicore__ inline void KernelDispatchEpilogue::Process()
{
    ComputePrefixSums();

    CountHits();
    SyncAll<true>();

    uint32_t myTotalHits = 0;
    for (uint32_t e = 0; e < numLocalExperts_; ++e) {
        myTotalHits += static_cast<uint32_t>(ubHitCount_.GetValue(e));
    }

    if (myTotalHits != 0) {
        CopyFromWindowByExpert();
    }

    if (aivId_ == 0) {
        CopyInputsToOutputs();
    }
}

__aicore__ inline void KernelDispatchEpilogue::ComputePrefixSums()
{
    int64_t acc = 0;
    for (uint32_t e = 0; e < numLocalExperts_; ++e) {
        int64_t val = ubExpertPfx_.GetValue(e);
        ubExpertPfx_.SetValue(e, acc);
        acc += val;
    }
}

__aicore__ inline void KernelDispatchEpilogue::CountHits()
{
    Duplicate(ubHitCount_, (int32_t)0, numLocalExperts_);

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t cnt = ubRecvCnt_.GetValue(rankId);
        if (cnt == 0) { continue; }

        uint32_t slotStart, slotEnd;
        SplitToCore(static_cast<uint32_t>(cnt), slotStart, slotEnd);
        if (slotStart >= slotEnd) { continue; }

        GM_ADDR srcRankBase = localWinBase_ + recvCounterOffset_
                              + rankId * numMaxTokensPerRank_ * perSlotBytes_;
        uint32_t numSlots = slotEnd - slotStart;
        GM_ADDR firstSlotAddr = srcRankBase + slotStart * perSlotBytes_;

        GlobalTensor<int32_t> srcMetaGm;
        srcMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(firstSlotAddr + metaOffset_));
        DataCopyExtParams metaCopyParams{static_cast<uint16_t>(numSlots), metaBytes_,
            perSlotBytes_ - metaBytes_, 0, 0};
        DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
        DataCopyPad(ubMeta_, srcMetaGm, metaCopyParams, metaPadParams);
        PipeBarrier<PIPE_MTE2>();

        for (uint32_t localSlot = 0; localSlot < numSlots; ++localSlot) {
            uint32_t metaBase = localSlot * paddedMetaElems_;

            for (uint32_t kk = 0; kk < topK_; ++kk) {
                int32_t expert = ubMeta_.GetValue(metaBase + kk);
                if (expert < 0) { continue; }
                uint32_t dst_rank = static_cast<uint32_t>(expert) / numLocalExperts_;
                if (dst_rank != rankId_) { continue; }
                uint32_t local_expert = static_cast<uint32_t>(expert) % numLocalExperts_;

                int32_t cur = ubHitCount_.GetValue(local_expert);
                ubHitCount_.SetValue(local_expert, cur + 1);
            }
        }
    }

    DataCopy(hitCountGm_[aivId_ * numLocalExperts_], ubHitCount_, numLocalExperts_);
    PipeBarrier<PIPE_MTE3>();
}

__aicore__ inline void KernelDispatchEpilogue::CopyFromWindowByExpert()
{
    Duplicate(ubLocalSeq_, (int32_t)0, numLocalExperts_);

    DataCopy(ubHitCountFull_, hitCountGm_, aivNum_ * numLocalExperts_);
    PipeBarrier<PIPE_MTE2>();

    for (uint32_t e = 0; e < numLocalExperts_; ++e) {
        int64_t offset = ubExpertPfx_.GetValue(e);
        for (uint32_t aiv = 0; aiv < aivId_; ++aiv) {
            offset += ubHitCountFull_.GetValue(aiv * numLocalExperts_ + e);
        }
        ubRowStart_.SetValue(e, offset);
    }

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t cnt = ubRecvCnt_.GetValue(rankId);
        if (cnt == 0) { continue; }

        uint32_t slotStart, slotEnd;
        SplitToCore(static_cast<uint32_t>(cnt), slotStart, slotEnd);
        if (slotStart >= slotEnd) { continue; }

        GM_ADDR srcRankBase = localWinBase_ + recvCounterOffset_
                              + rankId * numMaxTokensPerRank_ * perSlotBytes_;
        uint32_t numSlots = slotEnd - slotStart;
        GM_ADDR firstSlotAddr = srcRankBase + slotStart * perSlotBytes_;

        GlobalTensor<int32_t> srcMetaGm;
        srcMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(firstSlotAddr + metaOffset_));
        DataCopyExtParams metaCopyParams{static_cast<uint16_t>(numSlots), metaBytes_,
            perSlotBytes_ - metaBytes_, 0, 0};
        DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
        DataCopyPad(ubMeta_, srcMetaGm, metaCopyParams, metaPadParams);
        PipeBarrier<PIPE_MTE2>();

        bool hasPendingWrite = false;

        for (uint32_t localSlot = 0; localSlot < numSlots; ++localSlot) {
            uint32_t metaBase = localSlot * paddedMetaElems_;
            GM_ADDR slotAddr = srcRankBase + (slotStart + localSlot) * perSlotBytes_;

            int32_t src_rank_id   = ubMeta_.GetValue(metaBase + 2 * topK_);
            int32_t src_token_idx = ubMeta_.GetValue(metaBase + 2 * topK_ + 1);

            GlobalTensor<bfloat16_t> srcTokenTensor;
            srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(slotAddr),
                                             hidden_);
            DataCopy(ubToken_, srcTokenTensor, hidden_);
            PipeBarrier<PIPE_MTE2>();

            for (uint32_t kk = 0; kk < topK_; ++kk) {
                int32_t expert = ubMeta_.GetValue(metaBase + kk);
                if (expert < 0) { continue; }
                uint32_t dst_rank = static_cast<uint32_t>(expert) / numLocalExperts_;
                if (dst_rank != rankId_) { continue; }
                uint32_t local_expert = static_cast<uint32_t>(expert) % numLocalExperts_;

                int64_t expertRowStart = ubRowStart_.GetValue(local_expert);
                int32_t localSeq = ubLocalSeq_.GetValue(local_expert);
                uint32_t globalRow = static_cast<uint32_t>(expertRowStart + localSeq);

                if (globalRow >= expertSum_) { continue; }

                ubLocalSeq_.SetValue(local_expert, localSeq + 1);

                DataCopy(recvXGm_[globalRow * hidden_], ubToken_, hidden_);
                DataCopy(recvTopkIdxGm_[globalRow * topK_], ubMeta_[metaBase], topK_);
                hasPendingWrite = true;

                int32_t rawBits = ubMeta_.GetValue(metaBase + topK_ + kk);
                recvTopkWeightsGm_.SetValue(globalRow, *reinterpret_cast<float*>(&rawBits));
                recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 0, src_token_idx);
                recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 1, src_rank_id);
                recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 2, static_cast<int32_t>(kk));
            }

            if (hasPendingWrite) {
                PipeBarrier<PIPE_MTE3>();
                hasPendingWrite = false;
            }
        }
    }
}

__aicore__ inline void KernelDispatchEpilogue::CopyInputsToOutputs()
{
    DataCopy(numRecvPerRankGm_, ubRecvCnt_, epWorldSize_);

    DataCopy(ubExpertPfx_, numRecvPerExpertHpGm_, numLocalExperts_);
    PipeBarrier<PIPE_MTE2>();
    DataCopy(numRecvPerExpertGm_, ubExpertPfx_, numLocalExperts_);

    PipeBarrier<PIPE_MTE3>();
}

}  // namespace MoeEpDispatchEpilogueImpl

#endif  // MOE_EP_DISPATCH_EPILOGUE_H