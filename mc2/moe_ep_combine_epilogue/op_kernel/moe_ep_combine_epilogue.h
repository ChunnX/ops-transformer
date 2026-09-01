/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_ep_combine_epilogue.h
 * \brief MoE Expert-Parallel Combine Epilogue kernel — recv + reduce phase.
 *        Reads expert outputs from HCCL Window, accumulates per-token topK results,
 *        and writes combined_x / combined_topk_weights.
 */
#ifndef MOE_EP_COMBINE_EPILOGUE_H
#define MOE_EP_COMBINE_EPILOGUE_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_COMBINE_EPILOGUE_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#include "../common/mc2_moe_context.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#endif

#include "moe_ep_combine_epilogue_tiling_key.h"
#include "moe_ep_combine_epilogue_tiling.h"

namespace MoeEpCombineEpilogueImpl {

#if defined(ENABLE_MOE_EP_COMBINE_EPILOGUE_KERNEL)

using namespace AscendC;

#define TemplateMoeEpCombineEpilogueTypeClass typename XType, uint32_t HasTopkWeight
#define TemplateMoeEpCombineEpilogueTypeFunc XType, HasTopkWeight

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2U;

template <TemplateMoeEpCombineEpilogueTypeClass>
class MoeEpCombineEpilogue {
public:
    __aicore__ inline MoeEpCombineEpilogue(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR topkIdx, GM_ADDR combinedX, GM_ADDR combinedTopkWeights,
                                GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpCombineEpilogueInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &tokenPerAivNum);
    __aicore__ inline void BuffInit();
    __aicore__ inline bool WaitDispatch(uint32_t completionChannelCount);
    __aicore__ inline void ProcessTopKToken(uint32_t tokenIndex);
    __aicore__ inline void RecvPhaseReduce();

    __aicore__ inline GM_ADDR GetUrmaWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        PipeBarrier<PIPE_ALL>();
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }

    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        PipeBarrier<PIPE_ALL>();
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }

    TPipe *tpipe_{nullptr};
    const MoeEpCombineEpilogueInfo *tilingData_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t numTokens_{0};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0};
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    uint32_t aivNum_{0};
    uint32_t aivId_{0};

    uint32_t tStart_{0};
    uint32_t tEnd_{0};
    uint32_t tPerCore_{0};

    GlobalTensor<XType> combinedXGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;

    TQue<QuePosition::VECIN, 1> xInQue_;
    TQue<QuePosition::VECOUT, 1> xOutQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> weightQue_;
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<> stateBuf_;
    TBuf<> stateSumBuf_;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
};

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::Init(
    GM_ADDR context, GM_ADDR topkIdx, GM_ADDR combinedX, GM_ADDR combinedTopkWeights, GM_ADDR workspace,
    GM_ADDR tilingGM, TPipe *pipe, const MoeEpCombineEpilogueInfo *tilingData)
{
    PipeBarrier<PIPE_ALL>();
    tpipe_ = pipe;
    PipeBarrier<PIPE_ALL>();
    tilingData_ = tilingData;
    PipeBarrier<PIPE_ALL>();
    aivId_ = GetBlockIdx();
    PipeBarrier<PIPE_ALL>();
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    PipeBarrier<PIPE_ALL>();
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    PipeBarrier<PIPE_ALL>();
    numTokens_ = tilingData_->cfg.numTokens;
    PipeBarrier<PIPE_ALL>();
    topK_ = tilingData_->cfg.topK;
    PipeBarrier<PIPE_ALL>();
    axisH_ = tilingData_->cfg.hidden;
    PipeBarrier<PIPE_ALL>();
    aivNum_ = tilingData->aivNum;
    PipeBarrier<PIPE_ALL>();
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;

    PipeBarrier<PIPE_ALL>();
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    PipeBarrier<PIPE_ALL>();
    rankId_ = mc2Context_->epRankId;
    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        PipeBarrier<PIPE_ALL>();
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }

    PipeBarrier<PIPE_ALL>();
    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    PipeBarrier<PIPE_ALL>();
    combineDataWinOffset_ = tilingData->combineDataWinOffset;

    PipeBarrier<PIPE_ALL>();
    combinedXGm_.SetGlobalBuffer((__gm__ XType *)combinedX);

    PipeBarrier<PIPE_ALL>();
    if constexpr (HasTopkWeight == 1) {
        PipeBarrier<PIPE_ALL>();
        combinedTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)combinedTopkWeights);
    }
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    PipeBarrier<PIPE_ALL>();
    sendTokenNum = curSendCnt / curUseAivNum;
    PipeBarrier<PIPE_ALL>();
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;
    PipeBarrier<PIPE_ALL>();
    uint32_t newAivId = aivId_;

    PipeBarrier<PIPE_ALL>();
    startTokenId = sendTokenNum * newAivId;
    PipeBarrier<PIPE_ALL>();
    if (newAivId < remainderTokenNum) {
        PipeBarrier<PIPE_ALL>();
        sendTokenNum += 1;
        PipeBarrier<PIPE_ALL>();
        startTokenId += newAivId;
    } else {
        PipeBarrier<PIPE_ALL>();
        startTokenId += remainderTokenNum;
    }
    PipeBarrier<PIPE_ALL>();
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::BuffInit()
{
    PipeBarrier<PIPE_ALL>();
    SplitToCore(numTokens_, aivNum_, tStart_, tEnd_, tPerCore_);
    PipeBarrier<PIPE_ALL>();
    if (tStart_ >= numTokens_) {
        PipeBarrier<PIPE_ALL>();
        return;
    }

    PipeBarrier<PIPE_ALL>();
    if constexpr (HasTopkWeight == 1) {
        PipeBarrier<PIPE_ALL>();
        tpipe_->InitBuffer(weightQue_, DOUBLE_BUFFER_NUM, UB_ALIGN);
    }
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(xInQue_, DOUBLE_BUFFER_NUM, hAlignSize_);
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(xOutQue_, DOUBLE_BUFFER_NUM, hAlignSize_);
    PipeBarrier<PIPE_ALL>();
    uint32_t ubFp32Bytes = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(ubAccFp32Buf_, ubFp32Bytes);
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(ubTmpFp32Buf_, ubFp32Bytes);

    PipeBarrier<PIPE_ALL>();
    ubAccFp32_ = ubAccFp32Buf_.Get<float>();
    PipeBarrier<PIPE_ALL>();
    ubTmpFp32_ = ubTmpFp32Buf_.Get<float>();
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline bool MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::WaitDispatch(
    uint32_t completionChannelCount)
{
    PipeBarrier<PIPE_ALL>();
    uint32_t completionFlagCount = epWorldSize_ * completionChannelCount;
    PipeBarrier<PIPE_ALL>();
    uint64_t flagOffset = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>();
    GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + flagOffset;
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<uint32_t> stateGMTensor;
    PipeBarrier<PIPE_ALL>();
    stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);

    PipeBarrier<PIPE_ALL>();
    LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams params = {static_cast<uint16_t>(completionFlagCount), sizeof(uint32_t),
                                WIN_ADDR_ALIGN - sizeof(uint32_t), 0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyPadExtParams<uint32_t> padParams = {true, 0, 0, 0};

    PipeBarrier<PIPE_ALL>();
    DataCopyPad<uint32_t>(stateTensor, stateGMTensor, params, padParams);
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    PipeBarrier<PIPE_ALL>();
    LocalTensor<uint32_t> stateSumTensor = stateSumBuf_.Get<uint32_t>();
    PipeBarrier<PIPE_ALL>();
    uint32_t shape[] = {completionFlagCount, UB_ALIGN / sizeof(uint32_t)};
    PipeBarrier<PIPE_ALL>();
    ReduceSum<uint32_t, AscendC::Pattern::Reduce::RA, false>(stateSumTensor, stateTensor, shape, true);
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::V_S>();

    PipeBarrier<PIPE_ALL>();
    return stateSumTensor(0) == completionFlagCount;
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::ProcessTopKToken(uint32_t tokenIndex)
{
    PipeBarrier<PIPE_ALL>();
    Duplicate<float>(ubAccFp32_, (float)0, axisH_);
    PipeBarrier<PIPE_ALL>();
    DataCopyPadParams padParams = {false, 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};
    PipeBarrier<PIPE_ALL>();
    DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};
    PipeBarrier<PIPE_ALL>();
    for (uint32_t topkId = 0U; topkId < topK_; topkId++) {
        PipeBarrier<PIPE_ALL>();
        uint64_t slotOffset = (static_cast<uint64_t>(tokenIndex) * topK_ + topkId) * tilingData_->cfg.perSlotBytes;
        PipeBarrier<PIPE_ALL>();
        GM_ADDR wAddr = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_) + slotOffset;
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<XType> srcTokenTensor;
        PipeBarrier<PIPE_ALL>();
        srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(wAddr));
        PipeBarrier<PIPE_ALL>();
        LocalTensor<XType> xLocal = xInQue_.AllocTensor<XType>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(xLocal, srcTokenTensor, xCopyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        xInQue_.EnQue(xLocal);
        PipeBarrier<PIPE_ALL>();
        LocalTensor<XType> xIn = xInQue_.DeQue<XType>();
        PipeBarrier<PIPE_ALL>();
        Cast(ubTmpFp32_, xIn, AscendC::RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_ALL>();
        Add(ubAccFp32_, ubAccFp32_, ubTmpFp32_, axisH_);
        PipeBarrier<PIPE_ALL>();
        xInQue_.FreeTensor(xIn);
        PipeBarrier<PIPE_ALL>();
        if constexpr (HasTopkWeight == 1) {
            PipeBarrier<PIPE_ALL>();
            GM_ADDR weightAddr = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) +
                                 (tokenIndex * topK_ + topkId) * WIN_ADDR_ALIGN;
            PipeBarrier<PIPE_ALL>();
            GlobalTensor<float> srcWeightTensor;
            PipeBarrier<PIPE_ALL>();
            srcWeightTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightAddr));
            PipeBarrier<PIPE_ALL>();
            LocalTensor<float> weightLocal = weightQue_.AllocTensor<float>();
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(weightLocal, srcWeightTensor, weightCopyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            weightQue_.EnQue(weightLocal);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<float> weightOut = weightQue_.DeQue<float>();
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(combinedTopkWeightsGm_[tokenIndex * topK_ + topkId], weightOut, weightCopyParams);
            PipeBarrier<PIPE_ALL>();
            weightQue_.FreeTensor(weightOut);
        }
    }
    PipeBarrier<PIPE_ALL>();
    LocalTensor<XType> ubResultBf16 = xOutQue_.AllocTensor<XType>();
    PipeBarrier<PIPE_ALL>();
    Cast(ubResultBf16, ubAccFp32_, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_ALL>();
    xOutQue_.EnQue(ubResultBf16);
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::RecvPhaseReduce()
{
    PipeBarrier<PIPE_ALL>();
    constexpr uint32_t completionChannelCount = 1U;
    PipeBarrier<PIPE_ALL>();
    uint32_t completionFlagCount = epWorldSize_ * completionChannelCount;
    PipeBarrier<PIPE_ALL>();
    uint32_t flagBufferBytes = completionFlagCount * STATE_OFFSET;
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(stateBuf_, flagBufferBytes);
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(stateSumBuf_, UB_ALIGN);

    PipeBarrier<PIPE_ALL>();
    if (aivId_ == 0U) {
        PipeBarrier<PIPE_ALL>();
        while (!WaitDispatch(completionChannelCount)) {
        }
        PipeBarrier<PIPE_ALL>();
        LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_V>();
        PipeBarrier<PIPE_ALL>();
        Duplicate<uint32_t>(stateTensor, static_cast<uint32_t>(0), flagBufferBytes / sizeof(uint32_t));
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        PipeBarrier<PIPE_ALL>();
        uint64_t flagOffset = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN;
        PipeBarrier<PIPE_ALL>();
        GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + flagOffset;
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<uint32_t> stateGMTensor;
        PipeBarrier<PIPE_ALL>();
        stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams clearParams = {static_cast<uint16_t>(completionFlagCount), STATE_OFFSET, 0,
                                         WIN_ADDR_ALIGN - STATE_OFFSET, 0};
        PipeBarrier<PIPE_ALL>();
        DataCopyPad<uint32_t>(stateGMTensor, stateTensor, clearParams);
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    PipeBarrier<PIPE_ALL>();
    if (tPerCore_ == 0) {
        PipeBarrier<PIPE_ALL>();
        return;
    }

    PipeBarrier<PIPE_ALL>();
    DataCopyPadParams padParams = {false, 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    PipeBarrier<PIPE_ALL>();
    for (uint32_t tokenIdx = tStart_; tokenIdx < tEnd_; ++tokenIdx) {
        PipeBarrier<PIPE_ALL>();
        ProcessTopKToken(tokenIdx);
        PipeBarrier<PIPE_ALL>();
        LocalTensor<XType> ubResult = xOutQue_.DeQue<XType>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(combinedXGm_[tokenIdx * axisH_], ubResult, xCopyParams);
        PipeBarrier<PIPE_ALL>();
        xOutQue_.FreeTensor(ubResult);
    }
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::Process()
{
    PipeBarrier<PIPE_ALL>();
    BuffInit();
    PipeBarrier<PIPE_ALL>();
    RecvPhaseReduce();
}

#endif

} // namespace MoeEpCombineEpilogueImpl

#endif // MOE_EP_COMBINE_EPILOGUE_H
