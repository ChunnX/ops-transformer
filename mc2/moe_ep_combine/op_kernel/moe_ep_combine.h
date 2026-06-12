/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_ep_combine.h
 * \brief MoE Expert-Parallel Combine kernel implementation
 */
#ifndef MOE_EP_COMBINE_H
#define MOE_EP_COMBINE_H

#include "kernel_operator.h"
#include "kernel_operator_data_copy_intf.h"
#include "moe_ep_combine_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "../../common/op_kernel/mc2_moe_context.h"
#include "moe_ep_combine_tiling.h"

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#endif

namespace MoeEpCombineImpl {

using namespace AscendC;
using Mc2Aclnn::HCCL_MAX_RANK_SIZE;
using Mc2Aclnn::Mc2MoeContext;

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];

static constexpr uint32_t MAX_TOPK = 8;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;

__aicore__ inline GM_ADDR GetBaseWinAddrByRankId(uint32_t rankId)
{
    return (GM_ADDR)(winRankAddr_[rankId] + Mc2Kernel::A5_MTE_STATE_WIN_SIZE);
}

template <uint32_t HAS_TOPK_WEIGHTS>
class KernelMoeEpCombine {
public:
    __aicore__ inline KernelMoeEpCombine(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadata,
                                GM_ADDR numRecvPerRank, GM_ADDR topkWeights, GM_ADDR bias0, GM_ADDR bias1,
                                GM_ADDR combinedX, GM_ADDR combinedTopkWeightsOptional, GM_ADDR workspace,
                                GM_ADDR tilingGM, TPipe *pipe, const MoeEpCombineInfo *tilingData);

    __aicore__ inline void Process();

private:
__aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                    uint32_t &endTokenId);
    __aicore__ inline void SendPhaseExpertToToken();
    __aicore__ inline void RecvPhaseStashing();
    __aicore__ inline void RecvPhaseReduce();

    TPipe *tpipe_{nullptr};
    const MoeEpCombineInfo *tilingData_{nullptr};
    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    uint32_t rankId_{0};

    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t numTokens_{0};
    uint32_t topK_{0};
    uint32_t hidden_{0};
    uint32_t perSlotBytes_{0};
    uint32_t actuallA{0};
    uint32_t aivNum_{0};
    uint32_t tStart_{0};
    uint32_t tEnd_{0};
    uint32_t ubXBytes_{0};
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<int32_t> dstSlotIdxGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<float> topkWeightsGm_;

    GlobalTensor<bfloat16_t> combinedXGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<bfloat16_t> ubX_;
    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;
    LocalTensor<float> ubWeighted_;

    TBuf<QuePosition::VECIN> ubXBuf_;
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<QuePosition::VECIN> ubWeightedBuf_;

    GM_ADDR workspace_{nullptr};
    uint32_t aivId_{0};
};

template <uint32_t HAS_TOPK_WEIGHTS>
__aicore__ inline void KernelMoeEpCombine<HAS_TOPK_WEIGHTS>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank,
    GM_ADDR topkWeights, GM_ADDR bias0, GM_ADDR bias1, GM_ADDR combinedX, GM_ADDR combinedTopkWeightsOptional,
    GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe, const MoeEpCombineInfo *tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    workspace_ = workspace;
    
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    numTokens_ = tilingData_->cfg.numTokens;
    topK_ = tilingData_->cfg.topK;
    hidden_ = tilingData_->cfg.hidden;
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    aivNum_ = tilingData_->aivNum;

    SplitToCore(numTokens_, aivNum_, tStart_, tEnd_);

    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
    }

    actuallA = epWorldSize_ * numMaxTokensPerRank_;

    xGm_.SetGlobalBuffer((__gm__ bfloat16_t *)x);
    dstSlotIdxGm_.SetGlobalBuffer((__gm__ int32_t *)dstBufferSlotIdx);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    combinedXGm_.SetGlobalBuffer((__gm__ bfloat16_t *)combinedX);

    if constexpr (HAS_TOPK_WEIGHTS == 1) {
        topkWeightsGm_.SetGlobalBuffer((__gm__ float *)topkWeights);
        combinedTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)combinedTopkWeightsOptional);
    }

    ubXBytes_ = ALIGN_UP(2 * hidden_ * sizeof(bfloat16_t), 32);
    uint32_t ubAccBytes = hidden_ * sizeof(float);
    uint32_t ubTmpBytes = hidden_ * sizeof(float);
    uint32_t ubWeightBytes = hidden_ * sizeof(float);

    tpipe_->InitBuffer(ubXBuf_, ubXBytes_);
    tpipe_->InitBuffer(ubAccFp32Buf_, ubAccBytes);
    tpipe_->InitBuffer(ubTmpFp32Buf_, ubTmpBytes);
    if constexpr (HAS_TOPK_WEIGHTS == 1) {
        tpipe_->InitBuffer(ubWeightedBuf_, ubWeightBytes);
    }

    ubX_ = ubXBuf_.Get<bfloat16_t>();
    ubAccFp32_ = ubAccFp32Buf_.Get<float>();
    ubTmpFp32_ = ubTmpFp32Buf_.Get<float>();
    if constexpr (HAS_TOPK_WEIGHTS == 1) {
        ubWeighted_ = ubWeightedBuf_.Get<float>();
    }
}

template <uint32_t HAS_TOPK_WEIGHTS>
__aicore__ inline void KernelMoeEpCombine<HAS_TOPK_WEIGHTS>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
    uint32_t &endTokenId)
{
    uint32_t sendTokenNum = curSendCnt / curUseAivNum;                // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t newAivId = aivId_;

    startTokenId = sendTokenNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <uint32_t HAS_TOPK_WEIGHTS>
__aicore__ inline void KernelMoeEpCombine<HAS_TOPK_WEIGHTS>::SendPhaseExpertToToken()
{   
    // uint32_t aStart = aivId_ * actuallA / aivNum_;
    // uint32_t aEnd = (aivId_ + 1) * actuallA / aivNum_;
    uint32_t aStart = 0;
    uint32_t aEnd = 0;
    SplitToCore(actuallA, aivNum_, aStart, aEnd);
    uint16_t copyLen = static_cast<uint16_t>(hidden_);
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(copyLen), 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
    for (uint32_t a = aStart; a < aEnd; ++a) {
        int32_t src_token_global = recvSrcMetadataGm_.GetValue(a * 3 + 0);
        int32_t src_rank = recvSrcMetadataGm_.GetValue(a * 3 + 1);
        int32_t src_slot = recvSrcMetadataGm_.GetValue(a * 3 + 2);

        if (src_rank < 0)
            continue;

        GM_ADDR dstWinBase = GetBaseWinAddrByRankId(src_rank);
        GM_ADDR slotAddr = dstWinBase + (rankId_ * numMaxTokensPerRank_ + src_slot) * perSlotBytes_;
        //to do   疑问: 1.perSlotBytes_的值在tiling侧的赋值，该是多少，跟后面的weight_extra有关系么
        //     info.cfg.perSlotBytes = static_cast<uint32_t>(
        //     ((static_cast<uint64_t>(info.cfg.hidden) * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN 
        //      + 80UL + 16UL);

        DataCopyPad(ubX_, xGm_[a * hidden_], copyParams, copyPadParams);
        // 疑问：2.这里是不是要考虑udma通信的方式由localRankGM →  remoteRankGM   
        // 由 ubX_ 直接到 dstTokenTensor ，也就是 xGm_ 到 slotAddr
        GlobalTensor<bfloat16_t> dstTokenTensor;
        dstTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(slotAddr));
        DataCopyPad(dstTokenTensor, ubX_, copyParams);
        PipeBarrier<PIPE_ALL>();
        if constexpr (HAS_TOPK_WEIGHTS == 1) {
            //to do   Window[src_rank][src_slot].weight_extra = topk_weights[?]
            //疑问：3.topk_weights的哪个值？topk_weights[src_token_global*top_k + top_k_idx]
            //      4.Window[src_rank][src_slot].weight_extra 是哪个地址？slotAddr的后面要留4B么？
        }
        cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(slotAddr), hidden_ * sizeof(bfloat16_t));
        //没隔64B做一次DCCI
    }
}

template <uint32_t HAS_TOPK_WEIGHTS>
__aicore__ inline void KernelMoeEpCombine<HAS_TOPK_WEIGHTS>::RecvPhaseStashing()
{
    GM_ADDR localWinBase = GetBaseWinAddrByRankId(rankId_);
    GM_ADDR stashBase = workspace_;
    uint16_t copyLen = static_cast<uint16_t>(hidden_);
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(copyLen), 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
    for (uint32_t t = tStart_; t < tEnd_; ++t) {
        for (uint32_t k = 0; k < topK_; ++k) {
            int32_t slot = dstSlotIdxGm_.GetValue(t * topK_ + k);
            // dstSlotIdxGm_ 是 dispatch 的输出
            // dispatch时 sender 端为每个 (token, k) 预留的 receiver slot 索引
            if (slot < 0)
                continue;

            uint32_t numValidRanks = 0;
            GM_ADDR slotAddr = nullptr;

            for (uint32_t src_r = 0; src_r < epWorldSize_; ++src_r) {
                int32_t cnt = numRecvPerRankGm_.GetValue(src_r);
                // numRecvPerRankGm_ 是dispatch的输出
                // 疑问：1.如果 numRecvPerRankGm_ 存储的是每个 rank 的计数，那么应该使用累积计数？？cnt是累加的？
                // dispatch时 本 rank 从各 EP rank 接收 token 数
                // 疑问：2.slot值是一个slot的索引？他的意义是什么？ 为什么要跟numRecvPerRankGm_作比较
                if (slot < cnt) {
                    slotAddr = localWinBase + src_r * numMaxTokensPerRank_ * perSlotBytes_ + slot * perSlotBytes_;
                    // 疑问：3.怀疑此偏移的正确性 slot是rank内的offsetInRank？？
                    break;
                }
            }

            if (slotAddr == nullptr)
                continue;

            GlobalTensor<bfloat16_t> LocalRecvTokenTensor;
            LocalRecvTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(slotAddr));
            uint16_t Tokneindex = (k % 2 == 0)? 0 : ubXBytes_/sizeof(bfloat16_t);
            DataCopyPad(ubX_[Tokneindex], LocalRecvTokenTensor, copyParams, copyPadParams);
            // 疑问：4.所谓ubX_= ALIGN_UP(hidden × 2, 32)，是基于top_k遍历的时候交替拷贝么？
            //         4.2双 slot pipeline：ubSlot × 2 双缓冲（RecvPhase / Combine 阶段 A）。
            
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            GM_ADDR stashAddr = stashBase + ((t - tStart_) * topK_ + k) * hidden_ * sizeof(bfloat16_t);
            GlobalTensor<bfloat16_t> dstTokenTensor;
            dstTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(stashAddr));
            DataCopyPad(dstTokenTensor, ubX_[Tokneindex], copyParams);
        }
    }
}

template <uint32_t HAS_TOPK_WEIGHTS>
__aicore__ inline void KernelMoeEpCombine<HAS_TOPK_WEIGHTS>::RecvPhaseReduce()
{

    GM_ADDR stashBase = workspace_;
    uint16_t copyLen = static_cast<uint16_t>(hidden_);
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(copyLen), 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
    for (uint32_t t = tStart_; t < tEnd_; ++t) {
        Duplicate<float>(ubAccFp32_, (float)0, hidden_);

        for (uint32_t k = 0; k < topK_; ++k) {
            int32_t slot = dstSlotIdxGm_.GetValue(t * topK_ + k);
            if (slot < 0)
                continue;

            GM_ADDR stashAddr = stashBase + ((t - tStart_) * topK_ + k) * hidden_ * sizeof(bfloat16_t);
            GlobalTensor<bfloat16_t> srcTokenTensor;
            srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(stashAddr));
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            DataCopyPad(ubX_, srcTokenTensor, copyParams, copyPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(ubTmpFp32_, ubX_, AscendC::RoundMode::CAST_NONE, hidden_);
            PipeBarrier<PIPE_V>();
            if constexpr (HAS_TOPK_WEIGHTS == 0) {
                Add(ubAccFp32_, ubAccFp32_, ubTmpFp32_, hidden_);
            } else {
                float w = topkWeightsGm_.GetValue(t * topK_ + k);
                AscendC::Muls(ubWeighted_, ubTmpFp32_, w, hidden_);
                PipeBarrier<PIPE_V>();
                Add(ubAccFp32_, ubAccFp32_, ubWeighted_, hidden_);
                combinedTopkWeightsGm_.SetValue(t * topK_ + k, w);
                // combinedTopkWeightsGm_（dispatch 反向时承载 grad_topk_weights 的 reduce 结果）
                // 疑问：1.combine的reduce   这个reduce指的是 
                //        AtomicAdd(combined_topk_weights[t, k], weighted_w_reduce) ？
            }
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<bfloat16_t> ubResultBf16 = ubX_;
        Cast(ubResultBf16, ubAccFp32_, RoundMode::CAST_RINT, hidden_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(combinedXGm_[t * hidden_], ubResultBf16, copyParams);
    }
}

template <uint32_t HAS_TOPK_WEIGHTS>
__aicore__ inline void KernelMoeEpCombine<HAS_TOPK_WEIGHTS>::Process()
{
    SendPhaseExpertToToken();
    SyncAll<true>();
    // HcclBarrier(); 
    // 疑问：1.HcclBarrier中check状态设计？
    RecvPhaseStashing();

    RecvPhaseReduce();
    printf("moe_ep_combine End\r\n");
}

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H