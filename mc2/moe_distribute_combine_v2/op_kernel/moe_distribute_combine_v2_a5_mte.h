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
 * \file moe_distribute_combine_v2_a5_mte.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_A5_MTE_H
#define MOE_DISTRIBUTE_COMBINE_V2_A5_MTE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_combine_v2_quant.h"
#include "../common/op_kernel/mc2_moe_context.h"
#if __has_include("../moe_distribute_dispatch_v2/check_winsize.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_constant.h"
#include "../common/op_kernel/moe_distribute_base.h"
#include "../moe_distribute_dispatch_v2/check_winsize.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_elastic.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/check_winsize.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_elastic.h"
#endif

#define A5_MTE_FLOAT_OVERFLOW_MODE_CTRL 60
namespace MoeDistributeCombineV2A5MteImpl {
using namespace MoeDistributeV2Base;
using namespace Mc2Kernel;
using namespace Mc2Aclnn;
#define A5MteCombineTypeClass typename ExpandXType, typename XType, typename ExpandIdxType, \
    uint8_t QuantMode, bool HasAddRmsNorm
#define A5MteCombineTypeFunc ExpandXType, XType, ExpandIdxType, QuantMode, HasAddRmsNorm

constexpr uint32_t SPLIT_BLOCK_SIZE = 512U;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_FLAG_SIZE = 32U;
constexpr uint32_t SPLIT_BLOCK_FLAG_COUNT = SPLIT_BLOCK_FLAG_SIZE / sizeof(float);

using namespace AscendC;
template <A5MteCombineTypeClass>
class MoeDistributeCombineV2A5Mte {
public:
    __aicore__ inline MoeDistributeCombineV2A5Mte() {};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
        GM_ADDR epSendCount, GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expertScales,
        GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
        GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR yOut, GM_ADDR rstdOut,
        GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitInputAndOutput(GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expandX, GM_ADDR expertIds,
        GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
        GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
        GM_ADDR performanceInfo, GM_ADDR yOut, GM_ADDR rstdOut, GM_ADDR XOut);
    __aicore__ inline void InitCommContext(GM_ADDR mc2Context, const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void InitAttrs(GM_ADDR mc2Context, const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void InitTilingAttrs(const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void AlltoAllBuffInitAndMaskCal();
    __aicore__ inline void AlltoAllCommBuffInit();
    __aicore__ inline void TokenMaskCalCnt();
    __aicore__ inline void ExpertMaskCalCnt();
    __aicore__ inline void GenerateActiveMask(half val);
    __aicore__ inline void MaskSpecialExpert();
    __aicore__ inline void MaskAlign();
    __aicore__ inline void SetWaitTpStatusAndDisPatch();
    __aicore__ inline void ExpertAlltoAllDispatchInnerCopyAdd(uint32_t toRankId, uint32_t tokenId, uint32_t topkId,
        uint32_t tkIndex);
    __aicore__ inline void ExpertAlltoAllDispatchCopyAdd();
    __aicore__ inline void ProcessConstantExpert(uint32_t tokenIndex, uint32_t const_expert_idx, float scaleVal);
    __aicore__ inline void ProcessCopyExpert(uint32_t tokenIndex, float scaleVal);
    __aicore__ inline void ProcessMoeExpert(uint32_t tokenIndexOffset, uint32_t topkId, float scaleVal);
    __aicore__ inline void ProcessExpert(uint32_t tokenIndex, uint32_t processLen);
    __aicore__ inline void ProcessMoeExpertsLoop(uint32_t tokenIndex, uint32_t tokenIndexOffset, uint32_t& index);
    __aicore__ inline void ProcessSharedExpertsLoop(uint32_t tokenIndex, uint32_t tokenIndexOffset,
        uint32_t processLen);
    __aicore__ inline void AddSharedExpertX(uint32_t tokenIndex, uint32_t processLen);
    __aicore__ inline void ExpertScaleCopy(const uint32_t beginIndex, const uint32_t endIndex,
        const uint32_t tokenPerAivNum);
    __aicore__ inline void CalConstExpertAlpha(GlobalTensor<ExpandXType> constExpertAlphaGM, uint32_t const_expert_idx,
        float &alphaFloat);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline bool WaitDispatch(uint32_t tokenIndex, uint64_t performanceTimeStart, uint32_t beginIndex);
    __aicore__ inline void PerformanceInfoPerToken(
        uint32_t tokenIndex, uint32_t tokenLocalIdx, uint64_t performanceTimeStart, uint32_t beginIndex);
    __aicore__ inline bool CheckPackedTokenArriveBatch(uint32_t tokenIndex);
    __aicore__ inline bool NeedWaitTokenSlot(uint32_t tokenIndex, uint32_t slotIdx) const;
    __aicore__ inline void ClearPackedTokenFlags(uint32_t tokenIndex, LocalTensor<float> flagTensor);
    __aicore__ inline uint32_t CalcDispatchBufferNum(uint32_t perDispatchBufBytes);
    __aicore__ inline bool CheckPackedFlagRangeArrive(
        GM_ADDR flagBaseAddr, uint16_t blockCount, uint32_t flagFloatNum, uint32_t srcStrideBytes);
    __aicore__ inline bool CheckPackedTokenArrive(GM_ADDR rankGM);
    __aicore__ inline void AddRmsNormAddCompute(uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol,
                                                LocalTensor<float>& x1TmpFloatLocal,
                                                LocalTensor<float>& x2TmpFloatLocal,
                                                LocalTensor<float>& addOutTmpFloatLocal,
                                                const DataCopyExtParams& copyExtParams,
                                                const DataCopyPadExtParams<XType>& copyPadExtParams);
    __aicore__ inline void AddRmsNormRmsNormCompute(uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol,
                                                    LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
                                                    LocalTensor<ExpandXType>& gammaLocal,
                                                    const DataCopyExtParams& copyExtParams);
    __aicore__ GM_ADDR GetWinAddrByRankId(const int32_t rankId, const uint8_t domain)
    {
        if (isMc2Context_) {
            return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffsetEp_;
        }
        return Mc2Kernel::GetBaseWindAddrByRankId(epWinContext_, rankId, epRankIdOriginal_) + winDataSizeOffsetEp_;
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(const int32_t rankId, const uint8_t domain)
    {
        if (isMc2Context_) {
            return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + winStatusOffset_;
        }
        return Mc2Kernel::GetBaseWindStateAddrByRankId(epWinContext_, rankId, epRankIdOriginal_) + winStatusOffset_;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<bool> xActiveMaskGM_;
    GlobalTensor<int32_t> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<ExpandIdxType> epSendCountGM_;
    GlobalTensor<ExpandIdxType> elasticInfoGM_;
    GlobalTensor<int32_t> performanceInfoGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<XType> sharedExpertXGM_;
    GlobalTensor<XType> residualXGM_;
    GlobalTensor<XType> gammaGM_;
    GlobalTensor<XType> yOutGlobal_;
    GlobalTensor<float> rstdOutGlobal_;
    GlobalTensor<XType> expandOutGlobal_;
    GlobalTensor<XType> rankWindow_;                 // 用于存对端window的变量
    GlobalTensor<XType> tpRankWindow_;
    GlobalTensor<XType> rowTmpGlobal_;
    GlobalTensor<ExpandXType> oriXGM_;
    GlobalTensor<ExpandXType> constExpertAlpha1GM_;
    GlobalTensor<ExpandXType> constExpertAlpha2GM_;
    GlobalTensor<ExpandXType> constExpertVGM_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    GM_ADDR epWindowGM_;
    GM_ADDR tpWindowGM_;
    GM_ADDR stateGM_;
    GM_ADDR maskCalcWorkspaceGM_;
    GM_ADDR statusDataSpaceGm_;

    __gm__ Mc2MoeContext* mc2Context_{nullptr};

    LocalTensor<XType> winTpSendCountTensor_;
    LocalTensor<ExpandXType> gmTpSendCountTensor_;
    LocalTensor<XType> outTensor_;
    LocalTensor<float> winTpSendCountFloatTensor_;
    LocalTensor<float> gmTpSendCountFloatTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<int32_t> performanceInfoTensor_;
    LocalTensor<int32_t> performanceInfoTmpTensor_;
    LocalTensor<int32_t> firstRecordTensor_;;
    LocalTensor<bool> maskStrideTensor_;
    LocalTensor<bool> maskGenerateTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    uint32_t epRankId_{0};
    uint32_t epRankIdOriginal_{0};
    uint32_t coreIdx_{0};  // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};
    uint32_t rankNumPerShareExpert_{0};
    uint32_t moeExpertPerRankNum_{0};     // 每张卡部署的moe专家数
    uint32_t moeSendNum_{0};              // moeExpertPerRankNum_ * epWorldSize_
    uint32_t bufferNum_{0};
    uint32_t zeroExpertNum_{0};
    uint32_t copyExpertNum_{0};
    uint32_t constExpertNum_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertOriginalNum_{0};
    uint32_t globalBS_{0};
    __gm__ Mc2Kernel::HcclOpParam *epWinContext_{nullptr};
    uint32_t bsKNum_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendCntNum_{0};
    uint32_t ubSize_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winStatusOffset_{0};
    uint64_t totalWinSizeEp_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint32_t selfSendCnt_{0};
    uint32_t activeMaskAlignSize_{0};
    uint32_t hExpandXTypeSize_{0};
    uint32_t hFloatAlign32Size_{0};
    uint32_t hFloatAlign256Size_{0};
    uint32_t hExpandXAlign32Size_{0};
    uint32_t hExpandXAlignSize_{0};
    uint32_t hAlignRawWinSize_{0};
    uint32_t hAlignRawWinCnt_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t tokenScaleCnt_{0};
    uint32_t commDataBytes_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t scaleNumAlignSize_{0};
    uint32_t flagRcvCount_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t expertScaleBeginIdx_{0};
    uint32_t performanceInfoSizeAlign_{0};
    uint32_t tokenNumPerCoreAlign_{0};
    float armAvgFactor_{0.0};
    float epsilon_{0.0};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> moeQueue_;
    TQue<QuePosition::VECIN, 1> moeSumQueue_;
    TQue<QuePosition::VECIN, 1> gmTpSendCountQueue_;
    TQue<QuePosition::VECIN, 1> gmTpSendCountInQueue_;
    TQue<QuePosition::VECIN, 1> winTpSendCountInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutPackageQueue_;
    TBuf<> quantResultBuf_;
    TBuf<> reduceResultBuf_;
    TQue<QuePosition::VECIN, 1> moeMainSumQueue_;
    TQue<QuePosition::VECIN, 1> moeCopySumQueue_;
    TBuf<> readStateBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> winTpSendCountFloatBuf_;
    TBuf<> gmTpSendCountFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> gammaBuf_;
    TBuf<TPosition::VECCALC> reduceFp32Buf_;
    TBuf<> xActMaskTBuf_;
    TBuf<> xActMaskCastTBuf_;
    TBuf<> tokenTargetTBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> xActMaskSumTBuf_;
    TBuf<> expertMaskBuf_;
    TBuf<> performanceInfoBuf_;
    TBuf<> performanceInfoTmpBuf_;
    TBuf<> firstRecordBuf_;
    TBuf<> packedCheckFlagBuf_;
    TBuf<> packedCheckCompareBuf_;
    TBuf<> packedClearFlagBuf_;
    TBuf<> calBeginBuf_;
    TBuf<> calEndBuf_;
    bool isInputTokenMaskFlag_ = false;
    bool isInputExpertMaskFlag_ = false;
    bool hasSharedExpertX_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isPerformanceFlag_ = false;
    bool isScalingDownFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool enableSpecialExpert_ = false;
    bool isMc2Context_ = false;

    // int8量化
    TBuf<> xAbsBuf_;
    TBuf<> xMaxBuf_;
    TBuf<> xScaleMulBuf_;

    LocalTensor<half> fp16CastTensor_;
    LocalTensor<float> absFloatTensor_;
    LocalTensor<float> reduceMaxFloatTensor_;
    LocalTensor<float> scaleDivFloatTensor_;
    LocalTensor<float> scaleDupLocalTensor_;
    LocalTensor<XType> sendLocalTensor_;
    LocalTensor<half> tokenTargetTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<bool> expertMaskTensor_;
    LocalTensor<float> expertScalesLocal_;
    uint32_t dispatchBufferNum_{1};
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatBufLocal_;

    uint32_t scaleNum_{0};
    MoeDistributeCombineQuant<A5MteCombineTypeFunc> quantInst_;
    MoeDistributeElastic elasticInst_;
};

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::TokenMaskCalCnt()
{
    // 一维mask, 计算得到有效bs数量
    LocalTensor<bool> xActiveMaskTensor = xActMaskTBuf_.Get<bool>();
    LocalTensor<half> tempTensor = xActMaskCastTBuf_.Get<half>();
    LocalTensor<half> sumOutTensor = xActMaskSumTBuf_.Get<half>();
    DataCopyExtParams xActiveMaskParams{1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGM_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ExpertMaskCalCnt()
{
    // 二维mask, 挑选有效token
    uint64_t rsvdCnt = 0;
    uint32_t mask = axisBS_;
    LocalTensor<bool> maskStrideTensor = tokenBuf_.Get<bool>();
    LocalTensor<half> tempTensor = rowTmpFloatBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = sumFloatBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = tokenBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = mulBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = tokenBuf_.Get<uint32_t>();
    DataCopyExtParams xActiveMaskParams{
        static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)),  0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    SumParams axisBsSumParams{
        1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN) * ALIGNED_LEN / sizeof(half);

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    DataCopyPad(maskStrideTensor, xActiveMaskGM_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    Sum(tokenTargetTensor_, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor_, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(validBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::InitInputAndOutput(
    GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
    GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR yOut,
    GM_ADDR rstdOut, GM_ADDR XOut)
{
    if constexpr (HasAddRmsNorm) {
        residualXGM_.SetGlobalBuffer((__gm__ XType*)residualX);
        gammaGM_.SetGlobalBuffer((__gm__ XType*)gamma);
        yOutGlobal_.SetGlobalBuffer((__gm__ XType*)yOut);
        rstdOutGlobal_.SetGlobalBuffer((__gm__ float*)rstdOut);
    }
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType*)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t*)epSendCount);
    expertScalesGM_.SetGlobalBuffer((__gm__ float*)expertScales);
    xActiveMaskGM_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    sharedExpertXGM_.SetGlobalBuffer((__gm__ XType*)sharedExpertX);
    elasticInfoGM_.SetGlobalBuffer((__gm__ int32_t*)elasticInfo);
    oriXGM_.SetGlobalBuffer((__gm__ ExpandXType*)oriX);
    constExpertAlpha1GM_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertAlpha1);
    constExpertAlpha2GM_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertAlpha2);
    constExpertVGM_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertV);
    performanceInfoGM_.SetGlobalBuffer((__gm__ int32_t*)performanceInfo);

    expandOutGlobal_.SetGlobalBuffer((__gm__ XType*)XOut);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::InitTilingAttrs(
    const MoeDistributeCombineV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeCombineV2Info.bs;
    axisH_ = tilingData->moeDistributeCombineV2Info.h;
    axisK_ = tilingData->moeDistributeCombineV2Info.k;
    aivNum_ = tilingData->moeDistributeCombineV2Info.aivNum;
    ubSize_ = tilingData->moeDistributeCombineV2Info.totalUbSize;
    globalBS_ = tilingData->moeDistributeCombineV2Info.globalBs;
    hasElasticInfoFlag_ = tilingData->moeDistributeCombineV2Info.hasElasticInfo;
    isPerformanceFlag_ = tilingData->moeDistributeCombineV2Info.isPerformance;
    epWorldSizeOriginal_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    epRankId_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epRankIdOriginal_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epWorldSize_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    moeExpertPerRankNum_ = tilingData->moeDistributeCombineV2Info.moeExpertPerRankNum;
    totalWinSizeEp_ = tilingData->moeDistributeCombineV2Info.totalWinSizeEp;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineV2Info.isTokenMask;
    isInputExpertMaskFlag_ = tilingData->moeDistributeCombineV2Info.isExpertMask;
    hasSharedExpertX_ = tilingData->moeDistributeCombineV2Info.hasSharedExpertX;
    bufferNum_ = tilingData->moeDistributeCombineV2Info.bufferNum;
    zeroExpertNum_ = tilingData->moeDistributeCombineV2Info.zeroExpertNum;
    copyExpertNum_ = tilingData->moeDistributeCombineV2Info.copyExpertNum;
    constExpertNum_ = tilingData->moeDistributeCombineV2Info.constExpertNum;
    moeExpertNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
    moeExpertOriginalNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
    sharedExpertRankNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertRankNum;
    enableSpecialExpert_ = (constExpertNum_ + zeroExpertNum_ + copyExpertNum_ > 0U);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::InitCommContext(
    GM_ADDR mc2Context, const MoeDistributeCombineV2TilingData *tilingData)
{
    uint32_t epRankIdHccl{0};
    uint32_t epWorldSizeHccl{0};
    if (isMc2Context_) {
        // Using Mc2Context instead of hccl context
        mc2Context_ = (__gm__ Mc2MoeContext*)mc2Context;
        epRankIdHccl = mc2Context_->epRankId;
        epWorldSizeHccl = tilingData->moeDistributeCombineV2Info.epWorldSize;
        statusDataSpaceGm_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankIdHccl]);
    } else {
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        epWinContext_ = (__gm__ Mc2Kernel::HcclOpParam*)contextGM0;
        statusDataSpaceGm_ = Mc2Kernel::GetStatusDataSpaceGm(epWinContext_);
        epRankIdHccl = Mc2Kernel::GetRankId(epWinContext_);
        epWorldSizeHccl = Mc2Kernel::GetRankDim(epWinContext_);
    }
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempEpRankId = 0; tempEpRankId < epWorldSize_; tempEpRankId++) {
        OOMCheckAddrRange<XType>((__gm__ XType*)(GetWinAddrByRankId(tempEpRankId, EP_DOMAIN)), totalWinSizeEp_);
        OOMCheckAddrRange<float>((__gm__ float*)(GetWinStateAddrByRankId(tempEpRankId, EP_DOMAIN)), STATE_SIZE);
    }
#endif
    selfDataStatusGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)(statusDataSpaceGm_ + COMBINE_STATE_WIN_OFFSET +
                                                                coreIdx_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_,
                                moeExpertNum_, epWorldSizeOriginal_, globalBS_, dataStateBuf);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::InitAttrs(GM_ADDR mc2Context,
    const MoeDistributeCombineV2TilingData *tilingData)
{
    InitTilingAttrs(tilingData);
    InitCommContext(mc2Context, tilingData);
    if (hasElasticInfoFlag_) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(elasticInfoGM_);
        isScalingDownFlag_ = elasticInfoGM_.GetValue(0);
        elasticInst_.SetElasticInitParams(tpipe_, elasticInfoGM_);
        elasticInst_.InitElasticInfo(isScalingDownFlag_, epWorldSize_, sharedExpertRankNum_, moeExpertNum_, epRankId_,
            moeExpertPerRankNum_);
    }
    sharedExpertNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;
    if (epRankId_ < sharedExpertRankNum_) {
        isShareExpertRankFlag_ = true;
    }

    rankNumPerShareExpert_ = sharedExpertRankNum_ / sharedExpertNum_;

    stateOffset_ = STATE_OFFSET;
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    hFloatAlign32Size_ = Ceil(hFloatSize, UB_ALIGN) * UB_ALIGN;
    hFloatAlign256Size_ = Ceil(hFloatSize, ALIGNED_LEN) * ALIGNED_LEN;
    if constexpr ((QuantMode == INT8_COMM_QUANT) || (QuantMode == MXFP8_E5M2_COMM_QUANT) ||
        (QuantMode == MXFP8_E4M3_COMM_QUANT)) {
        constexpr uint32_t kA5FusedQuantAlign = ALIGNED_LEN * INT8_DIVIVE;
        uint32_t hFloatA5FusedQuantSize = Ceil(hFloatSize, kA5FusedQuantAlign) * kA5FusedQuantAlign;
        hFloatAlign32Size_ = hFloatA5FusedQuantSize;
        hFloatAlign256Size_ = hFloatA5FusedQuantSize;
    }
    hExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    hExpandXAlign32Size_ = Ceil(hExpandXTypeSize_, UB_ALIGN) * UB_ALIGN;
    hAlignRawWinSize_ = Ceil(hExpandXTypeSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hAlignRawWinCnt_ = hAlignRawWinSize_ / sizeof(ExpandXType);
    commDataBytes_ = hExpandXTypeSize_;
    if constexpr (QuantMode > UNQUANT) {
        quantInst_.QuantInit(scaleNum_, hExpandXAlign32Size_, hExpandXAlignSize_, scaleNumAlignSize_,
                             hFloatAlign256Size_, tokenScaleCnt_, axisH_);
        commDataBytes_ = tokenScaleCnt_ * sizeof(ExpandXType);
    }
    blockCntPerToken_ = Ceil(commDataBytes_, SPLIT_BLOCK_DATA_SIZE);
    hAlignWinSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXType);
    bsKNum_ = axisBS_ * axisK_;
    if constexpr (HasAddRmsNorm) {
        armAvgFactor_ = tilingData->moeDistributeCombineV2Info.armAvgFactor;
        epsilon_ = tilingData->moeDistributeCombineV2Info.epsilon;
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
    GM_ADDR epSendCount, GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expertScales,
    GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo,
    GM_ADDR yOut, GM_ADDR rstdOut, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeCombineV2TilingData *tilingData)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    maskCalcWorkspaceGM_ = workspaceGM + coreIdx_ * MASK_CALC_NEED_WORKSPACE;
    InitInputAndOutput(
        residualX, gamma, expandX, expertIds, expandIdx, epSendCount, expertScales, xActiveMask, sharedExpertX,
        elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, yOut, rstdOut, XOut);
    if (tilingData->moeDistributeCombineV2Info.isMc2Context) {
        isMc2Context_ = true;
        mc2Context_ = (__gm__ Mc2MoeContext*)mc2Context;
    } else {
        auto realWinSize = Mc2Kernel::GetWinSize(epWinContext_);
        CheckWindowSize(totalWinSizeEp_, realWinSize, tpipe_, XOut);
    }
    InitAttrs(mc2Context, tilingData);

    PipeBarrier<PIPE_ALL>();
    // 当前win区划分为前后两半区，连续两次dispatch，切换半区
    winDataSizeOffsetEp_ = static_cast<uint64_t>(dataState_)
        * (tilingData->moeDistributeCombineV2Info.totalWinSizeEp / 2UL);
    winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET; // 前面的预留给dispatch使用
    epWindowGM_ = GetWinAddrByRankId(epRankIdOriginal_, EP_DOMAIN);
    if (isShareExpertRankFlag_) {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(epSendCountGM_[epWorldSize_ - 1]);
        selfSendCnt_ = epSendCountGM_(epWorldSize_ - 1);
    } else {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(epSendCountGM_[moeSendNum_ - 1]);
        selfSendCnt_ = epSendCountGM_(moeSendNum_ - 1);
    }
    SplitCoreCal();
    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, hExpandXAlign32Size_);
    flagRcvCount_ = axisK_ + sharedExpertNum_;
}

template <A5MteCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::CalcDispatchBufferNum(
    uint32_t perDispatchBufBytes)
{
    tpipe_->InitBuffer(calEndBuf_, UB_ALIGN);
    uint64_t beginUbAddr = calBeginBuf_.Get<uint8_t>().GetPhyAddr();
    uint64_t endUbAddr = calEndBuf_.Get<uint8_t>().GetPhyAddr();
    uint64_t usedUbSize = endUbAddr - beginUbAddr + UB_ALIGN;
    uint64_t remainUbSize = (ubSize_ > usedUbSize) ? (ubSize_ - usedUbSize) : 0U;
    uint32_t bufNum = static_cast<uint32_t>(remainUbSize / perDispatchBufBytes);
    if (bufNum == 0U) {
        bufNum = 1U;
    }
    if (bufNum > 8U) {
        bufNum = 8U;
    }
    return bufNum;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::BuffInit()
{
    tpipe_->Reset(); // reset后 ctrl寄存器会复位为默认值
    // 单指令饱和模式
    AscendC::SetCtrlSpr<A5_MTE_FLOAT_OVERFLOW_MODE_CTRL, A5_MTE_FLOAT_OVERFLOW_MODE_CTRL>(0);
    tpipe_->InitBuffer(calBeginBuf_, UB_ALIGN);
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN);  // 32
    tpipe_->InitBuffer(indexCountsBuf_, sendCntNum_ * EXPAND_IDX_INFO * sizeof(int32_t));
    uint32_t tokenScaleAlign32Size = Ceil(tokenScaleCnt_ * sizeof(ExpandXType), UB_ALIGN) * UB_ALIGN;
    if constexpr (QuantMode > UNQUANT) {
        tpipe_->InitBuffer(xAbsBuf_, hFloatAlign256Size_);
        uint32_t hFloatAlign256Cnt = hFloatAlign256Size_ / sizeof(float);
        tpipe_->InitBuffer(xMaxBuf_, (hFloatAlign256Cnt / REDUCE_NUM) * sizeof(float));
        tpipe_->InitBuffer(xScaleMulBuf_, hFloatAlign256Size_);
        tpipe_->InitBuffer(winTpSendCountFloatBuf_, hFloatAlign32Size_);
        winTpSendCountFloatTensor_ = winTpSendCountFloatBuf_.Get<float>();
        absFloatTensor_ = xAbsBuf_.Get<float>();
        reduceMaxFloatTensor_ = xMaxBuf_.Get<float>();
        scaleDupLocalTensor_ = xScaleMulBuf_.Get<float>();
        fp16CastTensor_ = xAbsBuf_.Get<half>();
        Duplicate(absFloatTensor_, float(0), hFloatAlign256Cnt);
        quantInst_.SetQuantInitParams(winTpSendCountFloatTensor_, fp16CastTensor_, absFloatTensor_,
            reduceMaxFloatTensor_, scaleDupLocalTensor_);
        tpipe_->InitBuffer(quantResultBuf_, tokenScaleAlign32Size);
    }
    if (isScalingDownFlag_) {
        elasticInst_.InitElasticInfoTensor(epWorldSizeOriginal_, elasticInfoTensor_);
    }
    uint32_t perDispatchBufBytes = 0U;
    if constexpr (QuantMode > UNQUANT) {
        perDispatchBufBytes = hExpandXAlignSize_ + hAlignWinSize_;
    } else {
        perDispatchBufBytes = hExpandXAlign32Size_ + hAlignWinSize_;
    }
    dispatchBufferNum_ = CalcDispatchBufferNum(perDispatchBufBytes);
    if constexpr (QuantMode > UNQUANT) {
        tpipe_->InitBuffer(gmTpSendCountQueue_, dispatchBufferNum_, hExpandXAlignSize_);
        tpipe_->InitBuffer(xOutPackageQueue_, dispatchBufferNum_, hAlignWinSize_);
    } else {
        tpipe_->InitBuffer(gmTpSendCountQueue_, dispatchBufferNum_, hExpandXAlign32Size_);
        tpipe_->InitBuffer(xOutPackageQueue_, dispatchBufferNum_, hAlignWinSize_);
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::MaskAlign()
{
    // 扩展后的二维mask通过GM对齐内轴元素个数
    uint32_t calcCnt = Ceil(axisBS_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    GlobalTensor<bool> MaskGMTensor;
    MaskGMTensor.SetGlobalBuffer((__gm__ bool*)maskCalcWorkspaceGM_);
    DataCopyExtParams maskCalcParams = {1U, static_cast<uint32_t>(calcCnt * sizeof(bool)), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(MaskGMTensor, maskGenerateTensor_, maskCalcParams);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopyExtParams xActiveMaskParams{
        static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{true, 0U, static_cast<uint8_t>(UB_ALIGN - axisK_), 0U};
    DataCopyPad(maskStrideTensor_, MaskGMTensor, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::GenerateActiveMask(half val)
{
    maskStrideTensor_ = tokenBuf_.Get<bool>();
    LocalTensor<half> maskCalcTensor = tokenBuf_.Get<half>();
    
    if (isInputTokenMaskFlag_) {
        // 根据一维场景下的activeMaskBsCnt_，构造出二维mask
        uint32_t calcCnt = Ceil(axisBS_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
        Duplicate<half>(maskCalcTensor, static_cast<half>(0), calcCnt);
        PipeBarrier<PIPE_V>();
        uint32_t activeCalcCnt = Ceil(activeMaskBsCnt_ * axisK_ * sizeof(half), ALIGNED_LEN_256)
            * ALIGNED_LEN_256 / sizeof(half);
        Duplicate<half>(maskCalcTensor, static_cast<half>(1), activeCalcCnt);
        PipeBarrier<PIPE_V>();
        Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcTensor, RoundMode::CAST_NONE, calcCnt);
    } else {
        // 构造二维全true的mask
        uint32_t calcCnt = Ceil(axisBS_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
        Duplicate<half>(maskCalcTensor, val, calcCnt);
        PipeBarrier<PIPE_V>();
        Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcTensor, RoundMode::CAST_NONE, calcCnt);
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::MaskSpecialExpert()
{
    LocalTensor<int32_t> expertIdsTensor_ = mulBuf_.Get<int32_t>();
    LocalTensor<float> expertIdsFloat = rowTmpFloatBuf_.Get<float>();
    LocalTensor<uint8_t> maskTensor = mulBuf_.Get<uint8_t>();
    LocalTensor<half> maskCalcTensor = tokenBuf_.Get<half>();
    LocalTensor<half> maskCalcSelectedTensor = rowTmpFloatBuf_.Get<half>();
    maskStrideTensor_ = tokenBuf_.Get<bool>();
    LocalTensor<half> tempTensor = rowTmpFloatBuf_.Get<half>();

    // 拷入expertIds
    uint32_t mask = axisBS_ * axisK_;
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(mask * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGM_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 根据expertId小于moeExpertNum，得到考虑特殊专家后的mask
    uint32_t calcCnt = Ceil(mask * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    Cast(expertIdsFloat, expertIdsTensor_, RoundMode::CAST_NONE, calcCnt);
    PipeBarrier<PIPE_V>();
    int32_t moeExpertNumInt32 = static_cast<int32_t>(moeExpertOriginalNum_);
    CompareScalar(maskTensor, expertIdsFloat, static_cast<float>(moeExpertNumInt32), AscendC::CMPMODE::LT, calcCnt);
    PipeBarrier<PIPE_V>();
    if (isInputExpertMaskFlag_) {
        Cast(maskCalcTensor, expertMaskTensor_.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, calcCnt);
    } else {
        Cast(maskCalcTensor, maskGenerateTensor_.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, calcCnt);
    }
    PipeBarrier<PIPE_V>();
    Select(
        maskCalcSelectedTensor, maskTensor, maskCalcTensor, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
        calcCnt);
    PipeBarrier<PIPE_V>();
    Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcSelectedTensor, RoundMode::CAST_NONE, calcCnt);
    
    // 通过GM对齐内轴元素个数
    MaskAlign();

    // 更新考虑特殊专家后的
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN) * ALIGNED_LEN / sizeof(half);
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor_.ReinterpretCast<int8_t>();
    activeMaskAlignSize_ = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    Sum(tokenTargetTensor_, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AlltoAllCommBuffInit()
{
    activeMaskBsCnt_ = axisBS_;
    activeMaskAlignSize_ = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t maxSizeTokenBuf = hExpandXAlign32Size_;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size_;
    uint32_t bsKFloatAlign = Ceil(bsKNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size_ > bsKFloatAlign ? hFloatAlign256Size_ : bsKFloatAlign;
    if (isInputExpertMaskFlag_ || enableSpecialExpert_) {
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize_ * sizeof(half);
        maxSizeTokenBuf = (activeMaskAlignSize_ > hExpandXAlign32Size_ ? activeMaskAlignSize_ : hExpandXAlign32Size_);
        maxSizeRowTmpFloatBuf = (activeMaskAlignHalfSize > hFloatAlign32Size_
            ? activeMaskAlignHalfSize : hFloatAlign32Size_);
    }
    // InitBuffer需要在tiling中计算ub总量
    tpipe_->InitBuffer(tokenBuf_, maxSizeTokenBuf);                          // 16K 用于搬入输入token
    tpipe_->InitBuffer(rowTmpFloatBuf_, maxSizeRowTmpFloatBuf);              // 32K 用于存储cast之后的fp32 token数据
    tpipe_->InitBuffer(mulBuf_, mulBufSize);                   // 32K buffer复用， 最大用于存储Brcb之后的token，需要256对齐
    tpipe_->InitBuffer(sumFloatBuf_, hFloatAlign32Size_);                    // 32K add
    uint32_t packedDataBytes = blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE;
    uint32_t rawPackedDataBytes = Ceil(hExpandXTypeSize_, SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_DATA_SIZE;
    uint32_t moeQueueBytes = Ceil(packedDataBytes, UB_ALIGN) * UB_ALIGN;
    moeQueueBytes = (moeQueueBytes > rawPackedDataBytes) ? moeQueueBytes : rawPackedDataBytes;
    moeQueueBytes = (moeQueueBytes > hExpandXAlign32Size_) ? moeQueueBytes : hExpandXAlign32Size_;
    tpipe_->InitBuffer(moeSumQueue_, 2, moeQueueBytes);
    tpipe_->InitBuffer(moeMainSumQueue_, bufferNum_, moeQueueBytes);
    tpipe_->InitBuffer(moeCopySumQueue_, 2, moeQueueBytes);
    if constexpr (HasAddRmsNorm) {
        tpipe_->InitBuffer(gammaBuf_, hExpandXAlign32Size_);
        tpipe_->InitBuffer(reduceFp32Buf_, NUM_PER_REP_FP32 * sizeof(float) * 2);
        // H取最大值时，根据ReduceSum接口公式计算所需空间至少为64 * 2 = 128个元素
    }
    uint32_t checkFlagCount = flagRcvCount_ * blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT;
    uint32_t checkCompareCount = Ceil(checkFlagCount, 64U) * 64U;
    uint32_t checkFlagBufSize = Ceil(checkCompareCount * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t checkCompareBufSize = Ceil(checkCompareCount * sizeof(uint8_t), ALIGNED_LEN) * ALIGNED_LEN;
    uint32_t clearFlagCount = flagRcvCount_ * blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT;
    uint32_t clearFlagBufSize = Ceil(clearFlagCount * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(packedCheckFlagBuf_, checkFlagBufSize);
    tpipe_->InitBuffer(packedCheckCompareBuf_, checkCompareBufSize);
    tpipe_->InitBuffer(packedClearFlagBuf_, clearFlagBufSize);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AlltoAllBuffInitAndMaskCal()
{
    tpipe_->Reset();
    AlltoAllCommBuffInit();
    if constexpr (QuantMode > UNQUANT) {
        tpipe_->InitBuffer(xAbsBuf_, scaleNumAlignSize_);
        fp16CastTensor_ = mulBuf_.Get<half>();
        absFloatTensor_ = rowTmpFloatBuf_.Get<float>();
        scaleDupLocalTensor_ = mulBuf_.Get<float>();
        scaleDivFloatTensor_ = xAbsBuf_.Get<float>();
        quantInst_.SetDeQuantInitParams(fp16CastTensor_, absFloatTensor_, scaleDupLocalTensor_, scaleDivFloatTensor_);
    }
    if (isInputTokenMaskFlag_) {
        axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xActMaskTBuf_, axisBsAlignSize_);
        tpipe_->InitBuffer(xActMaskCastTBuf_, axisBsAlignSize_ * sizeof(half));
        tpipe_->InitBuffer(xActMaskSumTBuf_, axisBsAlignSize_ * sizeof(half));
        TokenMaskCalCnt(); // 计算一维mask
    }
    if (isInputExpertMaskFlag_) {
        tpipe_->InitBuffer(tokenTargetTBuf_, Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN);
        tpipe_->InitBuffer(validBsIndexTBuf_, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
        tpipe_->InitBuffer(expertMaskBuf_, Ceil(axisBS_ * axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
        tokenTargetTensor_ = tokenTargetTBuf_.Get<half>();
        validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
        ExpertMaskCalCnt(); // 计算二维mask
        expertMaskTensor_ = expertMaskBuf_.Get<bool>();
        DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyExtParams maskParams{1U, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(bool)), 0U, 0U, 0U};
        DataCopyPad(expertMaskTensor_, xActiveMaskGM_, maskParams, maskCopyPadParams);
        SyncFunc<AscendC::HardEvent::V_S>();
    }
    if (enableSpecialExpert_) {
        maskGenerateTensor_ = sumFloatBuf_.Get<bool>();
        if (!isInputExpertMaskFlag_) {
            tpipe_->InitBuffer(tokenTargetTBuf_, Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN);
            tokenTargetTensor_ = tokenTargetTBuf_.Get<half>();
            GenerateActiveMask(static_cast<half>(1));
        }
        MaskSpecialExpert();
    }
    if (isPerformanceFlag_) {
        uint32_t performanceInfoSize = JUMP_WRITE * epWorldSizeOriginal_ * sizeof(int32_t);
        performanceInfoSizeAlign_ = Ceil(performanceInfoSize, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(performanceInfoBuf_, performanceInfoSizeAlign_);
        performanceInfoTensor_ = performanceInfoBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceInfoTensor_, 0, JUMP_WRITE * epWorldSizeOriginal_);
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::SplitCoreCal()
{
    // 对需要发送的token数平均分核，得到每个核上处理的卡的数量
    sendCntNum_ = selfSendCnt_ / aivNum_;
    uint32_t remainderRankNum = selfSendCnt_ % aivNum_;
 
    startTokenId_ = sendCntNum_ * coreIdx_;
 
    if (coreIdx_ < remainderRankNum) {
        sendCntNum_++;
        startTokenId_ += coreIdx_;
    } else {
        startTokenId_ += remainderRankNum;
    }
    endTokenId_ = startTokenId_ + sendCntNum_;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::SetWaitTpStatusAndDisPatch()
{
    PipeBarrier<PIPE_ALL>();
    if (coreIdx_ >= selfSendCnt_) {
        return;
    }
    ExpertAlltoAllDispatchCopyAdd();
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ExpertAlltoAllDispatchCopyAdd()
{
    if (sendCntNum_ == 0U) { // 空闲核，直接返回
        return;
    }

    LocalTensor<ExpandIdxType> expandIdxLocal = indexCountsBuf_.Get<ExpandIdxType>();
    const DataCopyExtParams bskParams{
        1U, static_cast<uint32_t>(sendCntNum_ * EXPAND_IDX_INFO * sizeof(uint32_t)),
        0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expandIdxLocal, expandIdxGM_[startTokenId_ * EXPAND_IDX_INFO], bskParams, copyPadParams);

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t loop = 0; loop < sendCntNum_; loop++) {
        uint32_t tkIndex = startTokenId_ + ((loop + epRankId_) % sendCntNum_);
        uint32_t baseOffset = (tkIndex - startTokenId_) * EXPAND_IDX_INFO;
        uint32_t rankIdExpandIdx = static_cast<uint32_t>(expandIdxLocal(baseOffset));
        uint32_t toRankId = rankIdExpandIdx;
        uint32_t tokenId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 1));
        uint32_t topkId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 2));
        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET +
                epWorldSizeOriginal_ + rankIdExpandIdx);
        }
        ExpertAlltoAllDispatchInnerCopyAdd(toRankId, tokenId, topkId, tkIndex);
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ExpertAlltoAllDispatchInnerCopyAdd(
    uint32_t toRankId, uint32_t tokenId, uint32_t topkId, uint32_t tkIndex)
{
    uint32_t epOffset = tokenId * (axisK_ + sharedExpertNum_) + topkId;
    uint32_t tokenGMOffset = tkIndex * axisH_;
    uint32_t tokenWinOffset = tkIndex * hAlignRawWinCnt_;
    GM_ADDR rankGM = GetWinAddrByRankId(toRankId, EP_DOMAIN) + epOffset * hAlignWinSize_;
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{true, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyExtParams xScaleCopyParams{1U, static_cast<uint32_t>(tokenScaleCnt_ * sizeof(ExpandXType)), 0U, 0U, 0U};
    GlobalTensor<float> dstPackedGlobal;
    dstPackedGlobal.SetGlobalBuffer((__gm__ float*)(rankGM));
    if constexpr (QuantMode > UNQUANT) {
        gmTpSendCountTensor_ = gmTpSendCountQueue_.AllocTensor<ExpandXType>();
        LocalTensor<uint8_t> singleByteTok = gmTpSendCountTensor_.template ReinterpretCast<uint8_t>();
        if constexpr ((QuantMode == MXFP8_E5M2_COMM_QUANT) || (QuantMode == MXFP8_E4M3_COMM_QUANT)) {
            Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(ExpandXType));
        }
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
        gmTpSendCountQueue_.EnQue(gmTpSendCountTensor_);
        gmTpSendCountTensor_ = gmTpSendCountQueue_.DeQue<ExpandXType>();
        LocalTensor<XType> quantResultLT_ = quantResultBuf_.Get<XType>();
        quantInst_.QuantProcess(quantResultLT_, gmTpSendCountTensor_);
        gmTpSendCountQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
        PipeBarrier<PIPE_V>();
        sendLocalTensor_ = xOutPackageQueue_.AllocTensor<ExpandXType>();
        LocalTensor<float> srcDataTensor = quantResultLT_.template ReinterpretCast<float>();
        LocalTensor<float> padFlagFloatTensor = sendLocalTensor_.template ReinterpretCast<float>();
        Duplicate(padFlagFloatTensor, float(1.0), hAlignWinSize_ / sizeof(float));
        PipeBarrier<PIPE_V>();
        Copy(padFlagFloatTensor, srcDataTensor, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
        Copy(padFlagFloatTensor[64], srcDataTensor[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
        xOutPackageQueue_.EnQue(sendLocalTensor_);
        sendLocalTensor_ = xOutPackageQueue_.DeQue<ExpandXType>();
        DataCopy(dstPackedGlobal, padFlagFloatTensor, blockCntPerToken_ * SPLIT_BLOCK_SIZE / sizeof(float));
        xOutPackageQueue_.FreeTensor<ExpandXType>(sendLocalTensor_);
    } else {
        gmTpSendCountTensor_ = gmTpSendCountQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
        gmTpSendCountQueue_.EnQue(gmTpSendCountTensor_);
        gmTpSendCountTensor_ = gmTpSendCountQueue_.DeQue<ExpandXType>();
        outTensor_ = xOutPackageQueue_.AllocTensor<XType>();
        LocalTensor<float> srcfloat = gmTpSendCountTensor_.template ReinterpretCast<float>();
        LocalTensor<float> packedfloat = outTensor_.template ReinterpretCast<float>();
        Duplicate(packedfloat, float(1.0), hAlignWinSize_ / sizeof(float));
        PipeBarrier<PIPE_V>();
        Copy(packedfloat, srcfloat, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
        Copy(packedfloat[64], srcfloat[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
        xOutPackageQueue_.EnQue(outTensor_);
        outTensor_ = xOutPackageQueue_.DeQue<ExpandXType>();
        DataCopy(dstPackedGlobal, packedfloat, blockCntPerToken_ * SPLIT_BLOCK_SIZE / sizeof(float));
        xOutPackageQueue_.FreeTensor<XType>(outTensor_);
        gmTpSendCountQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
    }
}



template <A5MteCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::CheckPackedFlagRangeArrive(
    GM_ADDR flagBaseAddr, uint16_t blockCount, uint32_t flagFloatNum, uint32_t srcStrideBytes)
{
    if (flagFloatNum == 0U) {
        return true;
    }
    const uint32_t compareCount = Ceil(flagFloatNum, 64U) * 64U;
    const uint32_t compResultU64Num = Ceil(flagFloatNum, 64U);
    LocalTensor<float> flagTensor = packedCheckFlagBuf_.Get<float>();
    LocalTensor<uint8_t> compResultTensor = packedCheckCompareBuf_.Get<uint8_t>();
    LocalTensor<uint64_t> compResultU64Tensor = packedCheckCompareBuf_.Get<uint64_t>();
    Duplicate<float>(flagTensor, float(0), compareCount);
    PipeBarrier<PIPE_V>();
    DataCopyExtParams expFlagCopyParams{blockCount, SPLIT_BLOCK_FLAG_SIZE, srcStrideBytes, 0U, 0U};
    DataCopyPadExtParams<float> expFlagPadParams{false, 0U, 0U, 0U};
    GlobalTensor<float> dataFlagGlobal;
    dataFlagGlobal.SetGlobalBuffer((__gm__ float*)(flagBaseAddr));
    DataCopyPad(flagTensor, dataFlagGlobal, expFlagCopyParams, expFlagPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    CompareScalar(compResultTensor, flagTensor, float(1), AscendC::CMPMODE::EQ, compareCount);
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t arriveFlagNum = 0U;
    for (uint32_t i = 0U; i < compResultU64Num; ++i) {
        uint64_t flagCompMask = compResultU64Tensor.GetValue(i);
        int64_t firstInvalidIdx = ScalarGetSFFValue<0>(flagCompMask);
        if (firstInvalidIdx == -1) {
            arriveFlagNum += 64U;
        } else {
            arriveFlagNum += static_cast<uint32_t>(firstInvalidIdx);
            break;
        }
    }
    if (arriveFlagNum > flagFloatNum) {
        arriveFlagNum = flagFloatNum;
    }
    return arriveFlagNum == flagFloatNum;
}

template <A5MteCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::CheckPackedTokenArrive(GM_ADDR rankGM)
{
    if (blockCntPerToken_ == 0U) {
        return true;
    }
    const uint32_t flagFloatNum = blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT;
    return CheckPackedFlagRangeArrive(
        rankGM + SPLIT_BLOCK_DATA_SIZE, static_cast<uint16_t>(blockCntPerToken_), flagFloatNum, SPLIT_BLOCK_DATA_SIZE);
}

template <A5MteCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::CheckPackedTokenArriveBatch(
    uint32_t tokenIndex)
{
    if (blockCntPerToken_ == 0U) {
        return true;
    }

    const uint32_t totalSlotCount = flagRcvCount_;
    if (!isInputExpertMaskFlag_ && (zeroExpertNum_ + copyExpertNum_ + constExpertNum_) == 0U) {
        const uint32_t flagFloatNum = totalSlotCount * blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT;
        GM_ADDR tokenBaseAddr = (__gm__ uint8_t*)(epWindowGM_) + tokenIndex * totalSlotCount * hAlignWinSize_;
        return CheckPackedFlagRangeArrive(
            tokenBaseAddr + SPLIT_BLOCK_DATA_SIZE,
            static_cast<uint16_t>(totalSlotCount * blockCntPerToken_),
            flagFloatNum,
            SPLIT_BLOCK_DATA_SIZE);
    }

    uint32_t needWaitCount = 0U;
    uint32_t arriveCount = 0U;
    for (uint32_t slotIdx = 0U; slotIdx < totalSlotCount; ++slotIdx) {
        if (!NeedWaitTokenSlot(tokenIndex, slotIdx)) {
            continue;
        }
        needWaitCount++;
        GM_ADDR wAddr = (__gm__ uint8_t*)(epWindowGM_) + (tokenIndex * totalSlotCount + slotIdx) * hAlignWinSize_;
        if (CheckPackedTokenArrive(wAddr)) {
            arriveCount++;
        }
    }
    return arriveCount == needWaitCount;
}

template <A5MteCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::NeedWaitTokenSlot(
    uint32_t tokenIndex, uint32_t slotIdx) const
{
    if (slotIdx >= axisK_) {
        return true;
    }
    if (isInputExpertMaskFlag_) {
        bool maskExpertFlag = expertMaskTensor_.GetValue(tokenIndex * axisK_ + slotIdx);
        if (!maskExpertFlag) {
            return false;
        }
    }
    uint32_t expertId = expertIdsGM_.GetValue(tokenIndex * axisK_ + slotIdx);
    return expertId < moeExpertOriginalNum_;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ClearPackedTokenFlags(
    uint32_t tokenIndex, LocalTensor<float> flagTensor)
{
    if (blockCntPerToken_ == 0U) {
        return;
    }

    uint32_t clearTokenIndex = tokenIndex;
    if (isInputExpertMaskFlag_) {
        clearTokenIndex = validBsIndexTensor_.GetValue(tokenIndex);
    }

    const DataCopyExtParams clearFlagParams{static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_FLAG_SIZE, 0U,
        SPLIT_BLOCK_DATA_SIZE, 0U};
    for (uint32_t copyId = 0U; copyId < flagRcvCount_; ++copyId) {
        GM_ADDR wAddr = (__gm__ uint8_t*)(epWindowGM_) + (clearTokenIndex * flagRcvCount_ + copyId) * hAlignWinSize_;
        GlobalTensor<float> dstFlagGlobal;
        dstFlagGlobal.SetGlobalBuffer((__gm__ float*)(wAddr + SPLIT_BLOCK_DATA_SIZE));
        DataCopyPad(dstFlagGlobal, flagTensor, clearFlagParams);
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::PerformanceInfoPerToken(
    uint32_t tokenIndex, uint32_t tokenLocalIdx, uint64_t performanceTimeStart, uint32_t beginIndex)
{
    for (uint32_t i = 0; i < flagRcvCount_; i ++) {
        if (!NeedWaitTokenSlot(tokenIndex, i)) {
            continue;
        }
        uint32_t fromRankId;
        GM_ADDR wAddr = (__gm__ uint8_t*)(epWindowGM_) + (tokenIndex * flagRcvCount_ + i) * hAlignWinSize_;
        if (!CheckPackedTokenArrive(wAddr)) {
            continue;
        }
        if (i < axisK_) {
            uint32_t moeExpertId = expertIdsGM_.GetValue(tokenIndex * axisK_ + i);
            fromRankId = moeExpertId / moeExpertPerRankNum_ + sharedExpertRankNum_;
        } else {
            fromRankId = (i - axisK_) * rankNumPerShareExpert_ + epRankId_ % rankNumPerShareExpert_;
        }
        if (isScalingDownFlag_) {
            fromRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + fromRankId);
        }
        if (firstRecordTensor_.GetValue(tokenLocalIdx * flagRcvCount_ + i) == 0) {
            uint64_t performanceTimeCheck = static_cast<uint64_t>(GetSystemCycle());
            int32_t performanceTimeWait = static_cast<int32_t>((performanceTimeCheck - performanceTimeStart)
                / CYCLES_PER_US);
            uint32_t fromRankIdTime = performanceInfoTensor_.GetValue(JUMP_WRITE * fromRankId);
            uint32_t maxTimeValue = (fromRankIdTime < performanceTimeWait) ? performanceTimeWait : fromRankIdTime;
            performanceInfoTensor_.SetValue(JUMP_WRITE * fromRankId, maxTimeValue);
            firstRecordTensor_.SetValue(tokenLocalIdx * flagRcvCount_ + i, 1);
        }
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::WaitDispatch(
    uint32_t tokenIndex, uint64_t performanceTimeStart, uint32_t beginIndex)
{
    bool tokenArrived = CheckPackedTokenArriveBatch(tokenIndex);
    if (!tokenArrived) {
        return false;
    }
    if (isPerformanceFlag_) {
        PerformanceInfoPerToken(tokenIndex, tokenIndex - beginIndex, performanceTimeStart, beginIndex);
    }

    return true;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AddRmsNormAddCompute(
    uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol, LocalTensor<float>& x1TmpFloatLocal,
    LocalTensor<float>& x2TmpFloatLocal, LocalTensor<float>& addOutTmpFloatLocal,
    const DataCopyExtParams& copyExtParams, const DataCopyPadExtParams<XType>& copyPadExtParams)
{
    // 计算x + residual_x
    LocalTensor<XType> x2 = tokenBuf_.Get<XType>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(x2, residualXGM_[tokenIndex * axisH_ + tokenOffset], copyExtParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(x2TmpFloatLocal, x2, AscendC::RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    AscendC::Add(addOutTmpFloatLocal, x1TmpFloatLocal, x2TmpFloatLocal, numCol);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AddRmsNormRmsNormCompute(
    uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol, LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
    LocalTensor<ExpandXType>& gammaLocal, const DataCopyExtParams& copyExtParams)
{
    // 计算rstd
    LocalTensor<float> reduceBufLocal = reduceFp32Buf_.Get<float>();
    Mul(sqx, xFp32, xFp32, numCol);
    PipeBarrier<PIPE_V>();
    Muls(sqx, sqx, armAvgFactor_, numCol);
    PipeBarrier<PIPE_V>();
    ReduceSum(sqx, sqx, reduceBufLocal, numCol);
    PipeBarrier<PIPE_V>();
    Adds(sqx, sqx, epsilon_, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(sqx, sqx, 1);
    Duplicate(reduceBufLocal, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(reduceBufLocal, reduceBufLocal, sqx, 1);

    // rstd结果搬出
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyParams copyParams;
    copyParams.blockLen = sizeof(uint32_t);
    copyParams.blockCount = 1;
    DataCopyPad(rstdOutGlobal_[tokenIndex * 1 + tokenOffset], reduceBufLocal, copyParams);

    // 计算y
    SyncFunc<AscendC::HardEvent::V_S>();
    float rstdValue = reduceBufLocal.GetValue(0);
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(xFp32, xFp32, rstdValue, numCol);
    PipeBarrier<PIPE_V>();
    LocalTensor<XType> yLocal = rowTmpFloatBuf_.Get<XType>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);
    PipeBarrier<PIPE_V>();
    Cast(xFp32, yLocal, RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol);  // gamma_fp32 reuse sqx
    PipeBarrier<PIPE_V>();
    Mul(xFp32, xFp32, sqx, numCol);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);

    // y结果搬出
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(yOutGlobal_[tokenIndex * axisH_ + tokenOffset], yLocal, copyExtParams);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::CalConstExpertAlpha(
    GlobalTensor<ExpandXType> constExpertAlphaGM, uint32_t const_expert_idx, float &alphaFloat)
{
    LocalTensor<ExpandXType> weightLocal = moeSumQueue_.AllocTensor<ExpandXType>();
    LocalTensor<float> weightFloatLocal = mulBuf_.Get<float>();
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};

    // 使用moeSumQueue_分配缓冲区来存储alpha1对应的权重矩阵Wc
    DataCopyPad(weightLocal, constExpertAlphaGM[const_expert_idx * axisH_], expandXCopyParams, copyPadExtParams);
    moeSumQueue_.EnQue(weightLocal);
    weightLocal = moeSumQueue_.DeQue<ExpandXType>();
    Cast(weightFloatLocal, weightLocal, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();

    // 计算Wc * x
    Mul(weightFloatLocal, weightFloatLocal, rowTmpFloatLocal_, axisH_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams params{1, innerAlign, axisH_};
    Sum(weightFloatLocal, weightFloatLocal, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    alphaFloat = weightFloatLocal.GetValue(0);
    moeSumQueue_.FreeTensor<ExpandXType>(weightLocal);
}

// 处理常量专家
template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessConstantExpert(
    uint32_t tokenIndex, uint32_t const_expert_idx, float scaleVal)
{
    PipeBarrier<PIPE_ALL>();
    LocalTensor<ExpandXType> rowTmpLocal = tokenBuf_.Get<ExpandXType>();
    LocalTensor<float> alphaFloatLocal = tokenBuf_.Get<float>();
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    float alpha1Float = static_cast<float>(0.0);
    float alpha2Float = static_cast<float>(0.0);

    // 读取输入token
    DataCopyPad(rowTmpLocal, oriXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(rowTmpFloatLocal_, rowTmpLocal, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();

    // 计算Wc * x
    CalConstExpertAlpha(constExpertAlpha1GM_, const_expert_idx, alpha1Float);
    CalConstExpertAlpha(constExpertAlpha2GM_, const_expert_idx, alpha2Float);

    // 计算softmax(Wc * x)
    float maxAlphaFloat = (alpha1Float > alpha2Float) ? alpha1Float : alpha2Float;
    alphaFloatLocal.SetValue(0, alpha1Float - maxAlphaFloat);
    alphaFloatLocal.SetValue(1, alpha2Float - maxAlphaFloat);
    SyncFunc<AscendC::HardEvent::S_V>();
    Exp(alphaFloatLocal, alphaFloatLocal, 2);
    SyncFunc<AscendC::HardEvent::V_S>();
    float alphaSumFloat = alphaFloatLocal.GetValue(0) + alphaFloatLocal.GetValue(1);
    alpha1Float = alphaFloatLocal.GetValue(0) / alphaSumFloat;
    alpha2Float = alphaFloatLocal.GetValue(1) / alphaSumFloat;

    // 使用moeSumQueue_分配缓冲区来存储常量专家向量v
    LocalTensor<float> constVFloatLocal = mulBuf_.Get<float>();
    LocalTensor<ExpandXType> const_v_ub = moeSumQueue_.AllocTensor<ExpandXType>();
    DataCopyPad(const_v_ub, constExpertVGM_[const_expert_idx * axisH_], expandXCopyParams, copyPadExtParams);
    moeSumQueue_.EnQue(const_v_ub);
    const_v_ub = moeSumQueue_.DeQue<ExpandXType>();

    Cast(constVFloatLocal, const_v_ub, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    moeSumQueue_.FreeTensor<ExpandXType>(const_v_ub);

    // 计算 alpha1 * x + alpha2 * v
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(rowTmpFloatLocal_, rowTmpFloatLocal_, alpha1Float, axisH_);
    Muls(constVFloatLocal, constVFloatLocal, alpha2Float, axisH_);
    PipeBarrier<PIPE_V>();
    Add(rowTmpFloatLocal_, rowTmpFloatLocal_, constVFloatLocal, axisH_);
    PipeBarrier<PIPE_V>();

    // 乘以专家权重
    Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, axisH_);
    PipeBarrier<PIPE_V>();
    Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, axisH_);
    PipeBarrier<PIPE_V>();
}

// 处理拷贝专家
template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessCopyExpert(
    uint32_t tokenIndex, float scaleVal)
{
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    LocalTensor<ExpandXType> tmpUb = moeCopySumQueue_.AllocTensor<ExpandXType>();
    DataCopyPad(tmpUb, oriXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
    moeCopySumQueue_.EnQue(tmpUb);
    tmpUb = moeCopySumQueue_.DeQue<ExpandXType>();

    Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    moeCopySumQueue_.FreeTensor<ExpandXType>(tmpUb);

    Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, axisH_);
    PipeBarrier<PIPE_V>();
    Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, axisH_);
    PipeBarrier<PIPE_V>();
}

// 处理Moe专家
template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessMoeExpert(
    uint32_t tokenIndexOffset, uint32_t topkId, float scaleVal)
{
    uint32_t processLen = axisH_;
    const DataCopyExtParams xScaleCopyParams{
        static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_DATA_SIZE, SPLIT_BLOCK_FLAG_SIZE, 0U, 0U};
    const DataCopyExtParams expandXCopyParams{
        static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_DATA_SIZE, SPLIT_BLOCK_FLAG_SIZE, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};

    GM_ADDR wAddr = (__gm__ uint8_t*)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
    rowTmpGlobal_.SetGlobalBuffer((__gm__ XType*)wAddr);
    LocalTensor<XType> tmpUb = moeMainSumQueue_.AllocTensor<XType>();
    if constexpr (QuantMode > UNQUANT) {
        DataCopyPad(tmpUb, rowTmpGlobal_, xScaleCopyParams, copyPadExtParams);
    } else {
        DataCopyPad(tmpUb, rowTmpGlobal_, expandXCopyParams, copyPadExtParams);
    }
    moeMainSumQueue_.EnQue(tmpUb);
    tmpUb = moeMainSumQueue_.DeQue<XType>();
    LocalTensor<XType> outLocalTensor = fp16CastTensor_.template ReinterpretCast<XType>();
    if constexpr (QuantMode > UNQUANT) {
        quantInst_.DeQuantProcess(tmpUb, outLocalTensor, rowTmpFloatLocal_, sumFloatBufLocal_, scaleVal);
    } else {
        Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
        PipeBarrier<PIPE_V>();
        AscendC::Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, processLen);
        PipeBarrier<PIPE_V>();
        AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, processLen);
    }
    moeMainSumQueue_.FreeTensor<XType>(tmpUb);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ExpertScaleCopy(
    const uint32_t beginIndex, const uint32_t endIndex, const uint32_t tokenPerAivNum)
{
    expertScaleBeginIdx_ = beginIndex;
    uint32_t expertScaleEndIdx = endIndex;
    uint32_t expertScaleCntPerCore = tokenPerAivNum * axisK_;
    if (isInputExpertMaskFlag_) {
        expertScaleBeginIdx_ = validBsIndexTensor_.GetValue(beginIndex);
        expertScaleEndIdx = validBsIndexTensor_.GetValue(endIndex - 1);
        expertScaleCntPerCore = (expertScaleEndIdx - expertScaleBeginIdx_ + 1) * axisK_;
    }
    tpipe_->InitBuffer(expertScalesBuf_, Ceil(expertScaleCntPerCore * sizeof(float), UB_ALIGN) * UB_ALIGN);
    expertScalesLocal_ = expertScalesBuf_.Get<float>();
    const DataCopyExtParams tokenScaleParams{1U, static_cast<uint32_t>(expertScaleCntPerCore
        * sizeof(float)), 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    DataCopyPad(expertScalesLocal_, expertScalesGM_[expertScaleBeginIdx_ * axisK_],
        tokenScaleParams, copyPadFloatParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessMoeExpertsLoop(
    uint32_t tokenIndex, uint32_t tokenIndexOffset, uint32_t& index)
{
    float scaleVal = 0.0;
    if ((zeroExpertNum_ + copyExpertNum_ + constExpertNum_) == 0U) {
        for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
            if (isInputExpertMaskFlag_) {
                bool maskExpertFlag = expertMaskTensor_.GetValue(tokenIndex * axisK_ + topkId);
                if (!maskExpertFlag) {
                    index++;
                    continue;
                }
            }
            scaleVal = expertScalesLocal_.GetValue(index);
            ProcessMoeExpert(tokenIndexOffset, topkId, scaleVal);
            index++;
        }
    } else {
        for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
            // 读取expert_id
            DataCacheCleanAndInvalid<int32_t,
                CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertIdsGM_[tokenIndex * axisK_ + topkId]);
            uint32_t expert_id = expertIdsGM_.GetValue(tokenIndex * axisK_ + topkId);
            if (isInputExpertMaskFlag_) {
                bool maskExpertFlag = expertMaskTensor_.GetValue(tokenIndex * axisK_ + topkId);
                if (!maskExpertFlag) {
                    index++;
                    continue;
                }
            }
            scaleVal = expertScalesLocal_.GetValue(index);

            if (expert_id < moeExpertOriginalNum_) {
                ProcessMoeExpert(tokenIndexOffset, topkId, scaleVal);
                index++;
            } else if (expert_id < moeExpertOriginalNum_ + zeroExpertNum_) {
                // 零专家不需要任何操作
                index++;
            } else if (expert_id < moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_) {
                ProcessCopyExpert(tokenIndex, scaleVal);
                index++;
            } else if (expert_id < moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_ + constExpertNum_) {
                uint32_t const_expert_idx = expert_id - (moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_);
                ProcessConstantExpert(tokenIndex, const_expert_idx, scaleVal);
                index++;
            }
        }
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessSharedExpertsLoop(
    uint32_t tokenIndex, uint32_t tokenIndexOffset, uint32_t processLen)
{
    GM_ADDR wAddr;
    const DataCopyExtParams xScaleCopyParams{
        static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_DATA_SIZE, SPLIT_BLOCK_FLAG_SIZE, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    const DataCopyExtParams expandXCopyParams{
        static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_DATA_SIZE, SPLIT_BLOCK_FLAG_SIZE, 0U, 0U};
    LocalTensor<XType> tmpUb;

    for (uint32_t topkId = axisK_; topkId < (axisK_ + sharedExpertNum_); topkId++) {
        wAddr = (__gm__ uint8_t*)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
        rowTmpGlobal_.SetGlobalBuffer((__gm__ XType*)wAddr);
    
        tmpUb = moeMainSumQueue_.AllocTensor<XType>();

        if constexpr (QuantMode > UNQUANT) {
            DataCopyPad(tmpUb, rowTmpGlobal_, xScaleCopyParams, copyPadExtParams);
        } else {
            DataCopyPad(tmpUb, rowTmpGlobal_, expandXCopyParams, copyPadExtParams);
        }
        moeMainSumQueue_.EnQue(tmpUb);
        tmpUb = moeMainSumQueue_.DeQue<XType>();

        LocalTensor<XType> outLocalTensor = fp16CastTensor_.template ReinterpretCast<XType>();
        if constexpr (QuantMode > UNQUANT) {
            quantInst_.DeQuantProcess(tmpUb, outLocalTensor, rowTmpFloatLocal_, sumFloatBufLocal_, 1.0f);
        } else {
            Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
            PipeBarrier<PIPE_V>();
            AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, rowTmpFloatLocal_, processLen);
            PipeBarrier<PIPE_V>();
        }
        moeMainSumQueue_.FreeTensor<XType>(tmpUb);
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AddSharedExpertX(
    uint32_t tokenIndex, uint32_t processLen)
{
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    LocalTensor<XType> rowTmpLocal = tokenBuf_.Get<XType>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(rowTmpLocal, sharedExpertXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(rowTmpFloatLocal_, rowTmpLocal, AscendC::RoundMode::CAST_NONE, processLen);
    PipeBarrier<PIPE_V>();
    AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, rowTmpFloatLocal_, processLen);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessExpert(uint32_t tokenIndex,
    uint32_t processLen)
{
    uint32_t index = (tokenIndex - expertScaleBeginIdx_) * axisK_;
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 与结果搬出datacopy同tensor
    Duplicate(sumFloatBufLocal_, static_cast<float>(0), axisH_);
    uint32_t tokenIndexOffset = tokenIndex * (axisK_ + sharedExpertNum_);

    ProcessMoeExpertsLoop(tokenIndex, tokenIndexOffset, index);
    ProcessSharedExpertsLoop(tokenIndex, tokenIndexOffset, processLen);

    if (hasSharedExpertX_) {
        AddSharedExpertX(tokenIndex, processLen);
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::LocalWindowCopy()
{
    if (activeMaskBsCnt_ == 0U) {
        return;
    }
    uint32_t beginIndex = 0U;
    uint32_t endIndex = 0U;
    uint32_t processLen = 0U;
    uint32_t tokenOffset = 0U;
    uint32_t statePos = 1U;
    uint32_t tokenPerAivNum = activeMaskBsCnt_ / aivNum_;
    uint32_t remainderToken = activeMaskBsCnt_ % aivNum_;

    beginIndex = tokenPerAivNum * coreIdx_;
    if (coreIdx_ < remainderToken) {
        tokenPerAivNum++;
        beginIndex += coreIdx_;
    } else {
        beginIndex += remainderToken;
    }
    endIndex = beginIndex + tokenPerAivNum;
    if (tokenPerAivNum == 0U) {
        return;
    }
    processLen = axisH_;
    TBuf<> opPosDfxBuf;
    tpipe_->InitBuffer(opPosDfxBuf, UB_ALIGN);
    dataStateLocalTensor_ = opPosDfxBuf.Get<uint32_t>();
    rowTmpFloatLocal_ = rowTmpFloatBuf_.Get<float>();
    mulBufLocal_ = mulBuf_.Get<float>();
    sumFloatBufLocal_ = sumFloatBuf_.Get<float>();
    const DataCopyPadExtParams<XType> copyPadXTypeParams{false, 0U, 0U, 0U};
    DataCopyParams dataStateParams{1U, sizeof(uint32_t), 0U, 0U};
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    LocalTensor<XType> gammaLocal;
    if constexpr (HasAddRmsNorm) {
        gammaLocal = gammaBuf_.Get<XType>();
        DataCopyPad(gammaLocal, gammaGM_, expandXCopyParams, copyPadXTypeParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }
    ExpertScaleCopy(beginIndex, endIndex, tokenPerAivNum);
    TBuf<> tokenStatusBuf;
    tpipe_->InitBuffer(tokenStatusBuf, Ceil(tokenPerAivNum * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    LocalTensor tokenStatusTensor = tokenStatusBuf.Get<int32_t>();
    Duplicate<int32_t>(tokenStatusTensor, static_cast<int32_t>(0), tokenPerAivNum);
    uint32_t tokenNumCompleted = static_cast<uint32_t>(0);
    if (isScalingDownFlag_) {
        elasticInst_.InitElasticInfoTensor(epWorldSizeOriginal_, elasticInfoTensor_);
    }
    if (isPerformanceFlag_) {
        uint32_t tokenNumPerCore = tokenPerAivNum * flagRcvCount_ * sizeof(int32_t);
        tokenNumPerCoreAlign_ = Ceil(tokenNumPerCore, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(firstRecordBuf_, tokenNumPerCoreAlign_);
        firstRecordTensor_ = firstRecordBuf_.Get<int32_t>();
        Duplicate<int32_t>(firstRecordTensor_, static_cast<int32_t>(0), tokenPerAivNum * flagRcvCount_);
    }
    LocalTensor<float> flagTensor = packedClearFlagBuf_.Get<float>();
    Duplicate<float>(flagTensor, float(0), blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT);
    SyncFunc<AscendC::HardEvent::V_S>();
    uint64_t performanceTimeStart = static_cast<uint64_t>(GetSystemCycle());
    while (tokenNumCompleted != tokenPerAivNum) {
        for (uint32_t curIdx = beginIndex; curIdx < endIndex; curIdx++) {
            if (tokenStatusTensor(curIdx - beginIndex) == 1) {
                continue;
            }
            uint32_t tokenIndex = curIdx;
            if (isInputExpertMaskFlag_) {
                tokenIndex = validBsIndexTensor_.GetValue(curIdx);
            }
            if (!WaitDispatch(tokenIndex, performanceTimeStart, beginIndex)) {
                continue;
            }
            tokenNumCompleted++;
            tokenStatusTensor.SetValue(curIdx - beginIndex, 1);

            statePos++;
            dataStateLocalTensor_.SetValue(0, statePos);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams);
            ProcessExpert(tokenIndex, processLen);
            ClearPackedTokenFlags(curIdx, flagTensor);

            if constexpr (HasAddRmsNorm) {
                AddRmsNormAddCompute(tokenIndex, tokenOffset, processLen, sumFloatBufLocal_,
                    rowTmpFloatLocal_, sumFloatBufLocal_, expandXCopyParams, copyPadXTypeParams);
            }
            // 结果搬出
            PipeBarrier<PIPE_V>();
            LocalTensor<XType> sumBufLocal = tokenBuf_.Get<XType>();
            Cast(sumBufLocal, sumFloatBufLocal_, AscendC::RoundMode::CAST_RINT, processLen);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopyPad(expandOutGlobal_[tokenIndex * axisH_ + tokenOffset], sumBufLocal, expandXCopyParams);
            if constexpr (HasAddRmsNorm) {
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                AddRmsNormRmsNormCompute(tokenIndex, tokenOffset, processLen, sumFloatBufLocal_, mulBufLocal_,
                    gammaLocal, expandXCopyParams);
            }
        }
    }
    if (isPerformanceFlag_) {
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        SetAtomicMax<int32_t>();
        DataCopyExtParams performanceInfoCopyParams{1U, static_cast<uint32_t>(JUMP_WRITE * epWorldSizeOriginal_
            * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(performanceInfoGM_, performanceInfoTensor_, performanceInfoCopyParams);
        SetAtomicNone();
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        BuffInit();
        SetWaitTpStatusAndDisPatch();
        PipeBarrier<PIPE_ALL>(); // AlltoAllBuffInitAndMaskCal中包含reset操作，需确保前面操作完成
        AlltoAllBuffInitAndMaskCal();
        LocalWindowCopy();
    }
}

} // MoeDistributeCombineV2A5MteImpl
#endif // MOE_DISTRIBUTE_COMBINE_V2_A5_MTE_H
