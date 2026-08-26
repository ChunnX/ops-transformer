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
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
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
    __aicore__ inline void MaskAlign();
    __aicore__ inline void MaskCheck();
    __aicore__ inline bool WaitDispatch(uint32_t tokenIndex, uint32_t beginIndex);
    __aicore__ inline void ProcessTopKToken(uint32_t tokenIndex);
    __aicore__ inline void RecvPhaseReduce();

    __aicore__ inline GM_ADDR GetUrmaWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }

    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }

    TPipe *tpipe_{nullptr};
    const MoeEpCombineEpilogueInfo *tilingData_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t numTokens_{0};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0};
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    uint32_t aivNum_{0};
    uint32_t aivId_{0};
    uint64_t sendDataWorkspaceSizePerRank_{0};

    uint32_t tStart_{0};
    uint32_t tEnd_{0};
    uint32_t tPerCore_{0};
    uint64_t mask_calc_workspace_offset_{0};
    uint32_t mask_tokenNum_{0};
    uint32_t bsKCastCnt_{0};
    uint32_t activeMaskAlignSize_{0};

    GlobalTensor<XType> combinedXGm_;
    GlobalTensor<int32_t> topkIdxGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;
    LocalTensor<bool> maskGenerateTensor_;
    LocalTensor<bool> maskStrideTensor_;
    LocalTensor<half> tokenTargetTensor_;

    TQue<QuePosition::VECIN, 1> xInQue_;
    TQue<QuePosition::VECOUT, 1> xOutQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> weightQue_;
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<> tokenStatusBuf_;
    TBuf<> stateBuf_;
    TBuf<> stateSumBuf_;
    TBuf<> stateResetBuf_;

    TBuf<> compareBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> tokenTargetTBuf_;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    GM_ADDR maskCalcWorkspaceGM_{nullptr};
};

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::Init(
    GM_ADDR context, GM_ADDR topkIdx, GM_ADDR combinedX, GM_ADDR combinedTopkWeights, GM_ADDR workspace,
    GM_ADDR tilingGM, TPipe *pipe, const MoeEpCombineEpilogueInfo *tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numTokens_ = tilingData_->cfg.numTokens;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    aivNum_ = tilingData_->aivNum;
    sendDataWorkspaceSizePerRank_ = tilingData->sendDataWorkspaceSizePerRank;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;

    mask_calc_workspace_offset_ = epWorldSize_ * sendDataWorkspaceSizePerRank_;
    maskCalcWorkspaceGM_ = (GM_ADDR)workspace + mask_calc_workspace_offset_;
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    rankId_ = mc2Context_->epRankId;
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }

    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    combineDataWinOffset_ = tilingData->combineDataWinOffset;

    combinedXGm_.SetGlobalBuffer((__gm__ XType *)combinedX);
    topkIdxGm_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);

    if constexpr (HasTopkWeight == 1) {
        combinedTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)combinedTopkWeights);
    }
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;
    uint32_t newAivId = aivId_;

    startTokenId = sendTokenNum * newAivId;
    if (newAivId < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::BuffInit()
{
    mask_tokenNum_ = tPerCore_ * topK_;
    uint32_t bsKInt32Align = Ceil(mask_tokenNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t bsKFloatAlign = Ceil(mask_tokenNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t bsKHalfAlign = Ceil(mask_tokenNum_ * sizeof(half), UB_ALIGN) * UB_ALIGN;
    uint32_t bsHalfAlign = Ceil(tPerCore_ * sizeof(half), UB_ALIGN) * UB_ALIGN;

    bsKCastCnt_ = Ceil(mask_tokenNum_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    activeMaskAlignSize_ = tPerCore_ * Ceil(topK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN * sizeof(half);
    bsKInt32Align = (bsKInt32Align > bsKCastCnt_ ? bsKInt32Align : bsKCastCnt_);
    bsKHalfAlign = (bsKHalfAlign > activeMaskAlignSize_ ? bsKHalfAlign : activeMaskAlignSize_);
    bsKFloatAlign = (bsKFloatAlign > bsKCastCnt_ ? bsKFloatAlign : bsKCastCnt_);
    bsKFloatAlign = (bsKFloatAlign > activeMaskAlignSize_ ? bsKFloatAlign : activeMaskAlignSize_);

    tpipe_->InitBuffer(tokenTargetTBuf_, bsHalfAlign);
    tpipe_->InitBuffer(compareBuf_, bsKInt32Align);
    tpipe_->InitBuffer(rowTmpFloatBuf_, bsKFloatAlign);
    tpipe_->InitBuffer(tokenBuf_, bsKHalfAlign);

    if constexpr (HasTopkWeight == 1) {
        tpipe_->InitBuffer(weightQue_, DOUBLE_BUFFER_NUM, UB_ALIGN);
    }
    tpipe_->InitBuffer(xInQue_, DOUBLE_BUFFER_NUM, hAlignSize_);
    tpipe_->InitBuffer(xOutQue_, DOUBLE_BUFFER_NUM, hAlignSize_);
    uint32_t ubFp32Bytes = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(ubAccFp32Buf_, ubFp32Bytes);
    tpipe_->InitBuffer(ubTmpFp32Buf_, ubFp32Bytes);

    ubAccFp32_ = ubAccFp32Buf_.Get<float>();
    ubTmpFp32_ = ubTmpFp32Buf_.Get<float>();
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline bool MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::WaitDispatch(uint32_t tokenIndex,
                                                                                                uint32_t beginIndex)
{
    // 计算地址偏移
    GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + tokenIndex * topK_ * WIN_ADDR_ALIGN;
    GlobalTensor<uint32_t> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);

    LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopyExtParams params = {static_cast<uint16_t>(topK_), UB_ALIGN, WIN_ADDR_ALIGN - UB_ALIGN, 0, 0};
    DataCopyPadExtParams<uint32_t> padParams = {false, 0, 0, 0};

    DataCopyPad<uint32_t>(stateTensor, stateGMTensor, params, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<uint32_t> stateSumTensor = stateSumBuf_.Get<uint32_t>();
    uint32_t shape[] = {topK_, UB_ALIGN / sizeof(uint32_t)};
    ReduceSum<uint32_t, AscendC::Pattern::Reduce::RA, false>(stateSumTensor, stateTensor, shape, true);
    SyncFunc<AscendC::HardEvent::V_S>();

    uint32_t localState = stateSumTensor(1);
    int32_t target = static_cast<int32_t>(tokenTargetTensor_.GetValue(tokenIndex - beginIndex));
    if (localState == target) {
        return true;
    }
    return false;
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::MaskAlign()
{
    // 扩展后的二维mask通过GM对齐内轴元素个数
    uint32_t calcCntSize =
        Ceil((numTokens_ + aivNum_ - 1UL) / aivNum_ * topK_ * sizeof(bool), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    GlobalTensor<bool> MaskGMTensor;
    maskCalcWorkspaceGM_ = maskCalcWorkspaceGM_ + aivId_ * calcCntSize;
    MaskGMTensor.SetGlobalBuffer((__gm__ bool *)maskCalcWorkspaceGM_);
    DataCopyExtParams maskCalcParams = {1U, static_cast<uint32_t>(calcCntSize), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(MaskGMTensor, maskGenerateTensor_, maskCalcParams);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopyExtParams xActiveMaskCopyParams{static_cast<uint16_t>(tPerCore_),
                                            static_cast<uint32_t>(topK_ * sizeof(bool)), 0U, 0U, 0U};
    uint8_t rightPad = static_cast<uint8_t>(UB_ALIGN - topK_);
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{true, 0U, rightPad, 0U};
    DataCopyPad(maskStrideTensor_, MaskGMTensor, xActiveMaskCopyParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::MaskCheck()
{
    LocalTensor<half> maskCalcTensor = tokenBuf_.Get<half>();
    LocalTensor<int32_t> topkIdsTensor = compareBuf_.Get<int32_t>();
    LocalTensor<float> topkIdsFloatTensor = rowTmpFloatBuf_.Get<float>();
    LocalTensor<uint8_t> maskTensor = compareBuf_.Get<uint8_t>();
    LocalTensor<half> maskCalcSelectedTensor = rowTmpFloatBuf_.Get<half>();
    maskGenerateTensor_ = compareBuf_.Get<bool>();
    LocalTensor<half> tempTensor = rowTmpFloatBuf_.Get<half>();
    maskStrideTensor_ = tokenBuf_.Get<bool>();
    tokenTargetTensor_ = tokenTargetTBuf_.Get<half>();

    // 构造二维全true的mask
    Duplicate<half>(maskCalcTensor, static_cast<half>(1),
                    Ceil(mask_tokenNum_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half));
    // 拷入topkIds
    DataCopyExtParams topkIdsCntParams = {1U, static_cast<uint32_t>(mask_tokenNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(topkIdsTensor, topkIdxGm_[tStart_ * topK_], topkIdsCntParams, topkIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // 根据topkIds大于0，得到考虑(-1专家) 后的mask
    uint32_t calcCnt = bsKCastCnt_ / sizeof(int32_t);
    Cast(topkIdsFloatTensor, topkIdsTensor, RoundMode::CAST_NONE, calcCnt);
    CompareScalar(maskTensor, topkIdsFloatTensor, static_cast<float>(0), AscendC::CMPMODE::GE, calcCnt);
    Select(maskCalcSelectedTensor, maskTensor, maskCalcTensor, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           calcCnt);
    Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcSelectedTensor, RoundMode::CAST_NONE, calcCnt);
    // 通过GM对齐内轴元素个数
    MaskAlign();
    // 更新考虑特殊专家后的
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor_.ReinterpretCast<int8_t>();
    activeMaskAlignSize_ = tPerCore_ * Ceil(topK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize_);
    uint32_t innerAlign = UB_ALIGN;
    SumParams axisKSumParams{tPerCore_, innerAlign, topK_};
    Sum(tokenTargetTensor_, tempTensor, axisKSumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::ProcessTopKToken(uint32_t tokenIndex)
{
    Duplicate<float>(ubAccFp32_, (float)0, axisH_);
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};
    DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};
    for (uint32_t topkId = 0U; topkId < topK_; topkId++) {
        uint64_t slotOffset = (static_cast<uint64_t>(tokenIndex) * topK_ + topkId) * tilingData_->cfg.perSlotBytes;
        GM_ADDR wAddr = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_) + slotOffset;
        GlobalTensor<XType> srcTokenTensor;
        srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(wAddr));
        bool maskExpertFlag = maskGenerateTensor_.GetValue((tokenIndex - tStart_) * topK_ + topkId);
        if (!maskExpertFlag) {
            continue;
        }
        LocalTensor<XType> xLocal = xInQue_.AllocTensor<XType>();
        DataCopyPad(xLocal, srcTokenTensor, xCopyParams, padParams);
        xInQue_.EnQue(xLocal);
        LocalTensor<XType> xIn = xInQue_.DeQue<XType>();
        Cast(ubTmpFp32_, xIn, AscendC::RoundMode::CAST_NONE, axisH_);
        Add(ubAccFp32_, ubAccFp32_, ubTmpFp32_, axisH_);
        xInQue_.FreeTensor(xIn);

        if constexpr (HasTopkWeight == 1) {
            GM_ADDR weightAddr = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) +
                                 (tokenIndex * topK_ + topkId) * WIN_ADDR_ALIGN;
            GlobalTensor<float> srcWeightTensor;
            srcWeightTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightAddr));
            LocalTensor<float> weightLocal = weightQue_.AllocTensor<float>();
            DataCopyPad(weightLocal, srcWeightTensor, weightCopyParams, padParams);
            weightQue_.EnQue(weightLocal);
            LocalTensor<float> weightOut = weightQue_.DeQue<float>();
            DataCopyPad(combinedTopkWeightsGm_[tokenIndex * topK_ + topkId], weightOut, weightCopyParams);
            weightQue_.FreeTensor(weightOut);
        }
    }
    LocalTensor<XType> ubResultBf16 = xOutQue_.AllocTensor<XType>();
    Cast(ubResultBf16, ubAccFp32_, RoundMode::CAST_RINT, axisH_);
    xOutQue_.EnQue(ubResultBf16);
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::RecvPhaseReduce()
{
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    tpipe_->InitBuffer(tokenStatusBuf_, Ceil(tPerCore_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(stateBuf_, topK_ * STATE_OFFSET);
    tpipe_->InitBuffer(stateSumBuf_, UB_ALIGN);
    tpipe_->InitBuffer(stateResetBuf_, topK_ * STATE_OFFSET);
    LocalTensor<uint32_t> stateResetTensor = stateResetBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(stateResetTensor, (uint32_t)0.0, static_cast<uint32_t>(topK_ * FLOAT_PER_UB_ALIGN));
    LocalTensor<int32_t> tokenStatusTensor = tokenStatusBuf_.Get<int32_t>();
    Duplicate<int32_t>(tokenStatusTensor, static_cast<int32_t>(0), tPerCore_);
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t CompletedtokenNum = static_cast<uint32_t>(0);
    uint32_t copyCount = topK_;
    while (CompletedtokenNum != tPerCore_) {
        for (uint32_t tokenIdx = tStart_; tokenIdx < tEnd_; ++tokenIdx) {
            if (tokenStatusTensor(tokenIdx - tStart_) == 1) {
                continue;
            }
            if (!WaitDispatch(tokenIdx, tStart_)) {
                continue;
            }
            CompletedtokenNum++;
            tokenStatusTensor.SetValue(tokenIdx - tStart_, 1);
            ProcessTopKToken(tokenIdx);
            LocalTensor<XType> ubResultBf16 = xOutQue_.DeQue<XType>();
            DataCopyPad(combinedXGm_[tokenIdx * axisH_], ubResultBf16, xCopyParams);
            xOutQue_.FreeTensor(ubResultBf16);

            GM_ADDR stateGM =
                GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + tokenIdx * topK_ * WIN_ADDR_ALIGN;
            GlobalTensor<uint32_t> stateGMTensor;
            stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);
            DataCopyExtParams resetParams = {static_cast<uint16_t>(topK_), UB_ALIGN, 0, WIN_ADDR_ALIGN - UB_ALIGN, 0};
            DataCopyPad<uint32_t>(stateGMTensor, stateResetTensor, resetParams);
        }
    }
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::Process()
{
    SplitToCore(numTokens_, aivNum_, tStart_, tEnd_, tPerCore_);
    if (tStart_ >= numTokens_) {
        return;
    }
    BuffInit();
    MaskCheck();
    RecvPhaseReduce();
}

#endif

} // namespace MoeEpCombineEpilogueImpl

#endif // MOE_EP_COMBINE_EPILOGUE_H
