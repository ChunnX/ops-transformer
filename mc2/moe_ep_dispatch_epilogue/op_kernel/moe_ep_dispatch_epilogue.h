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
static constexpr uint32_t MAX_TOPK = 8;
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
    __aicore__ inline void ComputeAivOffsets();
    __aicore__ inline void CopyFromWindowByExpert();
    __aicore__ inline void CopyInputsToOutputs();

    __aicore__ inline void SplitToCore(uint32_t totalCnt, uint32_t &startIdx, uint32_t &endIdx);

    TPipe* tpipe_{nullptr};
    const MoeEpDispatchEpilogueInfo* td_{nullptr};
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

    LocalTensor<bfloat16_t> ubToken_;
    LocalTensor<int32_t> ubHitCount_;
    LocalTensor<int32_t> ubLocalSeq_;

    TBuf<QuePosition::VECIN> ubTokenBuf_;
    TBuf<QuePosition::VECIN> ubHitCountBuf_;
    TBuf<QuePosition::VECIN> ubLocalSeqBuf_;

    uint32_t expertSum_{0};
    GM_ADDR workspace_{nullptr};
    GM_ADDR localWinBase_{nullptr};
    uint32_t metaOffset_{0};
    uint32_t hitCountOffset_{0};
    uint32_t rowStartOffset_{0};
};

__aicore__ inline void KernelDispatchEpilogue::SplitToCore(
    uint32_t totalCnt, uint32_t &startIdx, uint32_t &endIdx)
{
    uint32_t perAiv = totalCnt / td_->aivNum;
    uint32_t remainder = totalCnt % td_->aivNum;
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
    td_ = tilingData;
    aivId_ = GetBlockIdx();
    workspace_ = workspace;

    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    rankId_ = mc2Context_->epRankId;
    for (uint32_t i = 0; i < td_->cfg.epWorldSize; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
    }

    localWinBase_ = GetBaseWinAddrByRankId(rankId_);
    recvCounterOffset_ = ALIGN_UP(td_->cfg.epWorldSize * sizeof(int32_t), WIN_ADDR_ALIGN);
    metaOffset_ = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), WIN_ADDR_ALIGN);

    numRecvPerRankHpGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRankHp);
    numRecvPerExpertHpGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpertHp);

    uint32_t expertSum = 0;
    for (uint32_t e = 0; e < td_->cfg.numLocalExperts; ++e) {
        expertSum += static_cast<uint32_t>(numRecvPerExpertHpGm_.GetValue(e));
    }
    expertSum_ = expertSum;

    recvXGm_.SetGlobalBuffer((__gm__ bfloat16_t*)recvX);
    recvTopkIdxGm_.SetGlobalBuffer((__gm__ int32_t*)recvTopkIdx);
    recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float*)recvTopkWeights);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadata);
    numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpert);

    hitCountOffset_ = ALIGN_UP(td_->cfg.numLocalExperts * sizeof(int64_t), 32);
    rowStartOffset_ = ALIGN_UP(
        hitCountOffset_ + td_->aivNum * td_->cfg.numLocalExperts * sizeof(int32_t), 32);

    uint32_t ubTokenBytes     = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), 32);
    uint32_t ubHitCountBytes  = ALIGN_UP(td_->cfg.numLocalExperts * sizeof(int32_t), 32);
    uint32_t ubLocalSeqBytes  = ALIGN_UP(td_->cfg.numLocalExperts * sizeof(int32_t), 32);

    tpipe_->InitBuffer(ubTokenBuf_, ubTokenBytes);
    tpipe_->InitBuffer(ubHitCountBuf_, ubHitCountBytes);
    tpipe_->InitBuffer(ubLocalSeqBuf_, ubLocalSeqBytes);

    ubToken_     = ubTokenBuf_.Get<bfloat16_t>();
    ubHitCount_  = ubHitCountBuf_.Get<int32_t>();
    ubLocalSeq_  = ubLocalSeqBuf_.Get<int32_t>();
}

__aicore__ inline void KernelDispatchEpilogue::Process()
{
    if (aivId_ == 0) {
        ComputePrefixSums();
    }
    SyncAll<true>();

    CountHits();
    SyncAll<true>();

    if (aivId_ == 0) {
        ComputeAivOffsets();
    }
    SyncAll<true>();

    CopyFromWindowByExpert();
    SyncAll<true>();

    if (aivId_ == 0) {
        CopyInputsToOutputs();
    }
}

__aicore__ inline void KernelDispatchEpilogue::ComputePrefixSums()
{
    GM_ADDR expertPfxBase = workspace_;

    int64_t expertAcc = 0;
    for (uint32_t e = 0; e < td_->cfg.numLocalExperts; ++e) {
        *reinterpret_cast<__gm__ int64_t*>(expertPfxBase + e * sizeof(int64_t)) = expertAcc;
        expertAcc += numRecvPerExpertHpGm_.GetValue(e);
    }
}

__aicore__ inline void KernelDispatchEpilogue::CountHits()
{
    for (uint32_t e = 0; e < td_->cfg.numLocalExperts; ++e) {
        ubHitCount_.SetValue(e, 0);
    }

    for (uint32_t sr = 0; sr < td_->cfg.epWorldSize; ++sr) {
        int32_t cnt = numRecvPerRankHpGm_.GetValue(sr);
        if (cnt == 0) continue;

        uint32_t slotStart, slotEnd;
        SplitToCore(static_cast<uint32_t>(cnt), slotStart, slotEnd);
        if (slotStart >= slotEnd) continue;

        GM_ADDR srcRankBase = localWinBase_ + recvCounterOffset_
                              + sr * td_->cfg.numMaxTokensPerRank * td_->cfg.perSlotBytes;

        for (uint32_t slotIdx = slotStart; slotIdx < slotEnd; ++slotIdx) {
            GM_ADDR slotAddr = srcRankBase + slotIdx * td_->cfg.perSlotBytes;
            __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(slotAddr + metaOffset_);

            int32_t seen_expert_local[MAX_TOPK];
            uint32_t n_expert_hit = 0;

            for (uint32_t kk = 0; kk < td_->cfg.topK; ++kk) {
                int32_t expert = metaPtr[kk];
                if (expert < 0) continue;
                uint32_t dst_rank = static_cast<uint32_t>(expert) / td_->cfg.numLocalExperts;
                if (dst_rank != rankId_) continue;
                uint32_t local_expert = static_cast<uint32_t>(expert) % td_->cfg.numLocalExperts;

                int32_t hit = -1;
                for (uint32_t j = 0; j < n_expert_hit; ++j) {
                    if (seen_expert_local[j] == (int32_t)local_expert) { hit = j; break; }
                }
                if (hit >= 0) continue;

                seen_expert_local[n_expert_hit] = local_expert;
                n_expert_hit++;

                int32_t cur = ubHitCount_.GetValue(local_expert);
                ubHitCount_.SetValue(local_expert, cur + 1);
            }
        }
    }

    GM_ADDR hitCountBase = workspace_ + hitCountOffset_;
    for (uint32_t e = 0; e < td_->cfg.numLocalExperts; ++e) {
        *reinterpret_cast<__gm__ int32_t*>(
            hitCountBase + (aivId_ * td_->cfg.numLocalExperts + e) * sizeof(int32_t))
            = ubHitCount_.GetValue(e);
    }
}

__aicore__ inline void KernelDispatchEpilogue::ComputeAivOffsets()
{
    GM_ADDR expertPfxBase = workspace_;
    GM_ADDR hitCountBase = workspace_ + hitCountOffset_;
    GM_ADDR rowStartBase = workspace_ + rowStartOffset_;

    for (uint32_t e = 0; e < td_->cfg.numLocalExperts; ++e) {
        int64_t offset = *reinterpret_cast<__gm__ int64_t*>(expertPfxBase + e * sizeof(int64_t));
        for (uint32_t aiv = 0; aiv < td_->aivNum; ++aiv) {
            *reinterpret_cast<__gm__ int64_t*>(
                rowStartBase + (aiv * td_->cfg.numLocalExperts + e) * sizeof(int64_t)) = offset;
            offset += *reinterpret_cast<__gm__ int32_t*>(
                hitCountBase + (aiv * td_->cfg.numLocalExperts + e) * sizeof(int32_t));
        }
    }
}

__aicore__ inline void KernelDispatchEpilogue::CopyFromWindowByExpert()
{
    for (uint32_t e = 0; e < td_->cfg.numLocalExperts; ++e) {
        ubLocalSeq_.SetValue(e, 0);
    }

    GM_ADDR rowStartBase = workspace_ + rowStartOffset_;

    for (uint32_t sr = 0; sr < td_->cfg.epWorldSize; ++sr) {
        int32_t cnt = numRecvPerRankHpGm_.GetValue(sr);
        if (cnt == 0) continue;

        uint32_t slotStart, slotEnd;
        SplitToCore(static_cast<uint32_t>(cnt), slotStart, slotEnd);
        if (slotStart >= slotEnd) continue;

        GM_ADDR srcRankBase = localWinBase_ + recvCounterOffset_
                              + sr * td_->cfg.numMaxTokensPerRank * td_->cfg.perSlotBytes;

        for (uint32_t slotIdx = slotStart; slotIdx < slotEnd; ++slotIdx) {
            GM_ADDR slotAddr = srcRankBase + slotIdx * td_->cfg.perSlotBytes;
            __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(slotAddr + metaOffset_);

            int32_t src_rank_id   = metaPtr[td_->cfg.topK * 2];
            int32_t src_token_idx = metaPtr[td_->cfg.topK * 2 + 1];

            bool tokenLoaded = false;
            int32_t seen_expert_local[MAX_TOPK];
            uint32_t n_expert_hit = 0;
            bool hasPendingWrite = false;

            for (uint32_t kk = 0; kk < td_->cfg.topK; ++kk) {
                int32_t expert = metaPtr[kk];
                if (expert < 0) continue;
                uint32_t dst_rank = static_cast<uint32_t>(expert) / td_->cfg.numLocalExperts;
                if (dst_rank != rankId_) continue;
                uint32_t local_expert = static_cast<uint32_t>(expert) % td_->cfg.numLocalExperts;

                int32_t hit = -1;
                for (uint32_t j = 0; j < n_expert_hit; ++j) {
                    if (seen_expert_local[j] == (int32_t)local_expert) { hit = j; break; }
                }
                if (hit >= 0) continue;

                seen_expert_local[n_expert_hit] = local_expert;
                n_expert_hit++;

                if (!tokenLoaded) {
                    GlobalTensor<bfloat16_t> srcTokenTensor;
                    srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(slotAddr),
                                                     td_->cfg.hidden);
                    DataCopy(ubToken_, srcTokenTensor, td_->cfg.hidden);
                    PipeBarrier<PIPE_MTE2>();
                    tokenLoaded = true;
                }

                int64_t expertRowStart = *reinterpret_cast<__gm__ int64_t*>(
                    rowStartBase + (aivId_ * td_->cfg.numLocalExperts + local_expert) * sizeof(int64_t));
                int32_t localSeq = ubLocalSeq_.GetValue(local_expert);
                ubLocalSeq_.SetValue(local_expert, localSeq + 1);
                uint32_t globalRow = static_cast<uint32_t>(expertRowStart + localSeq);

                if (globalRow >= expertSum_) continue;

                DataCopy(recvXGm_[globalRow * td_->cfg.hidden], ubToken_,
                         td_->cfg.hidden);
                hasPendingWrite = true;

                for (uint32_t k2 = 0; k2 < td_->cfg.topK; ++k2) {
                    recvTopkIdxGm_.SetValue(globalRow * td_->cfg.topK + k2, metaPtr[k2]);
                }
                int32_t rawBits = metaPtr[td_->cfg.topK + kk];
                recvTopkWeightsGm_.SetValue(globalRow, *reinterpret_cast<float*>(&rawBits));
                recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 0, src_token_idx);
                recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 1, src_rank_id);
                recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 2, static_cast<int32_t>(kk));
            }

            if (hasPendingWrite) {
                PipeBarrier<PIPE_MTE3>();
            }
        }
    }
}

__aicore__ inline void KernelDispatchEpilogue::CopyInputsToOutputs()
{
    for (uint32_t i = 0; i < td_->cfg.epWorldSize; ++i) {
        numRecvPerRankGm_.SetValue(i, numRecvPerRankHpGm_.GetValue(i));
    }
    for (uint32_t i = 0; i < td_->cfg.numLocalExperts; ++i) {
        numRecvPerExpertGm_.SetValue(i, numRecvPerExpertHpGm_.GetValue(i));
    }
}

}  // namespace MoeEpDispatchEpilogueImpl

#endif  // MOE_EP_DISPATCH_EPILOGUE_H