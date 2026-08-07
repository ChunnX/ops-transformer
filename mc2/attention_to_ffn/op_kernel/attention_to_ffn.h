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
 * \file attention_to_ffn.h
 * \brief
 */

#ifndef ATTENTION_TO_Ffn_H
#define ATTENTION_TO_Ffn_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "adv_api/reduce/reduce.h"
#include "kernel_tiling/kernel_tiling.h"
#include "attention_to_ffn_tiling.h"
#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/attention_ffn_context.h"
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/attention_ffn_context.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

namespace AttentionToFfnImpl {
constexpr uint8_t BUFFER_NUM = 2; // 多Buf
constexpr uint32_t UB_ALIGN = 32; // UB按32字节对齐
constexpr uint8_t WIN_OFFSET_CNT = 2;
constexpr uint32_t SCALE_PARAM_PAD_SIZE = 128;   // 预留128B给量化参数
constexpr uint32_t WIN_ALIGN = 512;              // win offset 512字节对齐
constexpr uint32_t REP_STRIDE = 8;               // 相邻迭代间的地址步长
constexpr uint32_t STATUS_REP_STRIDE = 8;        // 32B / sizeof(int32) = 8
constexpr uint32_t EXPERT_TABLE_REP_STRIDE = 16; // 64B / sizeof(int32) = 16
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 128;
constexpr uint32_t DYNAMIC_QUANT = 2;                   // 动态量化
constexpr uint32_t RANK_OFFSET_STRIDE = 2;              // RankTable第一个为rankCnt，后面两两组合
constexpr uint32_t TOKEN_INFO_TABLE_RS = 2;             // tokenInfoTable前2位为flag和layer_id
constexpr uint32_t TOKEN_INFO_TABLE_COPY_BLOCK_CNT = 2; // tokenInfoTable在DataCopy时每次搬运2个int32
constexpr float INT8_MAX_VALUE = 127.0f;


enum class AttentionToFFNContextType : uint8_t {
    HCCL_CONTEXT = 0,
    MC2_CONTEXT = 1,
};

#define TemplateAttentionToFfnTypeClass                                                                             \
    typename XType, bool isQuant, bool isSync, bool isActiveMask, AttentionToFFNContextType contextType
#define TemplateAttentionToFfnTypeFunc XType, isQuant, isSync, isActiveMask, contextType

using namespace AscendC;
template <TemplateAttentionToFfnTypeClass>
class AttentionToFfn {
public:
    __aicore__ inline AttentionToFfn(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId, GM_ADDR layerId,
                                GM_ADDR expertIds, GM_ADDR expertRankTable, GM_ADDR scales, GM_ADDR active_mask,
                                GM_ADDR workspaceGM, TPipe *pipe, const AttentionToFfnTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count);
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void InitByTinglingData(const AttentionToFfnTilingData *tilingData);
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum);
    __aicore__ inline void SendTokenToFfn();
    __aicore__ inline void SendTokenToFfnByTokenIdx(uint32_t tokenIdx);
    __aicore__ inline void SetFlagToFfn();
    __aicore__ inline void ActiveMaskCalCnt();
    __aicore__ inline void SetFlagInAttn();
    __aicore__ inline void FindExpertRank(int32_t expertId);
    __aicore__ inline void SetExpertAndRank(uint32_t tokenIdx, uint32_t tokenId, uint32_t topkId);
    __aicore__ inline GM_ADDR GetWindowAddr(uint32_t rankId);
    __aicore__ inline GM_ADDR GetLocalWindowAddr();
    __aicore__ inline void CheckFlagAndSetTableGM(int32_t toRankId, GM_ADDR &toRankAddr,
                                                  GlobalTensor<int32_t> &tokenInfoTableGMTensor);
    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> sessionIdGMTensor_;
    GlobalTensor<int32_t> microBatchIdGMTensor_;
    GlobalTensor<int32_t> layerIdGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<int32_t> expertRankTableGMTensor_;
    GlobalTensor<int32_t> syncStatusGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<bool> activeMaskGMTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> scalesFp32Tensor_;
    LocalTensor<int8_t> xOutTensor_;
    LocalTensor<int32_t> ffnStatusTensor_;
    LocalTensor<int32_t> syncStatusWorkspaceTensor_;
    LocalTensor<int32_t> ffnFlagTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> expertIdsTensor_;

    TBuf<> smoothScalesBuf_; // 量化使用，搬入smoothScales
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> sumOutBuf_;
    TBuf<> activeMaskBuf_;
    TBuf<> castTempBuf_;
    TBuf<> ffnStatusBuf_;
    TBuf<> syncStatusWorkspaceBuf_;
    TBuf<> ffnFlagBuf_;
    TBuf<> attnStatusBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 非量化使用
    TQue<QuePosition::VECIN, 1> xInQueue_;                        // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;                      // 量化使用，量化后的输出
    GM_ADDR syncStatusWorkspaceGM_;                               // 异步场景使用workSpace记录发送状态信息

    int32_t dstExpertId_{0};
    int32_t toRankId_{0};
    int32_t localExpId_{0};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t axisX_{0};
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisL_{0};
    uint32_t axisK_{0};
    bool isScales_{false};
    uint32_t expertNum_{0};
    uint32_t expRankTableM_{0};
    uint32_t attentionWorkerNum_{0};
    uint32_t infoTableLastDimNum_{0};
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint64_t hSize_{0};
    uint32_t sessionId_{0};
    uint32_t microBatchId_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t layerId_{0};
    uint32_t totalSendNum_{0};
    uint32_t sendNum_{0};
    uint32_t quantMode_{0};
    uint32_t scaleParamPad_{0};
    uint32_t ffnNum_{0};
    uint32_t ffnStartRankId_{0};
    uint64_t microBatchNum_{0};
    uint64_t axisHS_{0};
    uint64_t hCommuSize_{0};
    uint64_t ffnNumAlignSize_{0};
    uint64_t layIdsExpRankTableOffset_{0};
    uint64_t preExpRankTableOffset_{0};
    uint64_t winTokenDataOffset_{0};
    uint64_t winInfoTableOffset_{0};
    uint64_t winOffset_[WIN_OFFSET_CNT]{0, 0};
    uint32_t curBsCnt_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t moeExpertNum_{0};
    uint32_t expertRankTableCnt_{0};
    uint32_t ffnNumAlignCnt_{0};
    uint32_t aivWorkspaceOffset_{0};
    uint8_t windowType_{0};
    uint8_t sharedExpertNum_{1};
    __gm__ HcclOpResParam *hcclContext_{nullptr};
    __gm__ Mc2Aclnn::AttentionFFNContext *mc2Context_{nullptr};
};

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::QuantInit(GM_ADDR scales)
{
    scaleParamPad_ = SCALE_PARAM_PAD_SIZE; // 预留128B给量化参数
    hCommuSize_ = axisH_ * sizeof(int8_t) + scaleParamPad_;
    tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hSize_);       // 14K *2
    tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_); // 7K *2 + 256B
    if (isScales_) {                                         // 输入smoothScale参数
        scalesGMTensor_.SetGlobalBuffer((__gm__ float *)(scales));
    }
    uint32_t hFp32Size = axisH_ * sizeof(float);
    tpipe_->InitBuffer(receiveDataCastFloatBuf_, hFp32Size); // H * 4
    tpipe_->InitBuffer(smoothScalesBuf_, hFp32Size);
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void
AttentionToFfn<TemplateAttentionToFfnTypeFunc>::InitByTinglingData(const AttentionToFfnTilingData *tilingData)
{
    aivId_ = GetBlockIdx();
    if constexpr (contextType == AttentionToFFNContextType::HCCL_CONTEXT) {
        rankId_ = hcclContext_->localUsrRankId;
    } else {
        rankId_ = mc2Context_->epRankId;
    }
    axisX_ = tilingData->attentionToFfnInfo.X;
    axisBS_ = tilingData->attentionToFfnInfo.BS;
    axisH_ = tilingData->attentionToFfnInfo.H;
    axisL_ = tilingData->attentionToFfnInfo.L;
    axisK_ = tilingData->attentionToFfnInfo.K;
    expertNum_ = tilingData->attentionToFfnInfo.expertNum; // 所有专家数：共享专家+所有的moe专家
    moeExpertNum_ = tilingData->attentionToFfnInfo.moeExpertNum;
    expRankTableM_ = tilingData->attentionToFfnInfo.expRankTableM;
    microBatchNum_ = tilingData->attentionToFfnInfo.microBatchNum;
    attentionWorkerNum_ = tilingData->attentionToFfnInfo.attentionWorkerNum;
    infoTableLastDimNum_ = tilingData->attentionToFfnInfo.infoTableLastDimNum;
    axisHS_ = tilingData->attentionToFfnInfo.HS;
    aivNum_ = tilingData->attentionToFfnInfo.aivNum;
    worldSize_ = tilingData->attentionToFfnInfo.worldSize;
    quantMode_ = tilingData->attentionToFfnInfo.quantMode;
    isScales_ = tilingData->attentionToFfnInfo.isScales;
    ffnStartRankId_ = tilingData->attentionToFfnInfo.ffnStartRankId;
    windowType_ = tilingData->attentionToFfnInfo.windowType;
    sharedExpertNum_ = tilingData->attentionToFfnInfo.sharedExpertNum;
    ffnNum_ = worldSize_ - attentionWorkerNum_;
    curBsCnt_ = axisBS_;
    hSize_ = axisH_ * sizeof(XType);
    expertIdsCnt_ = axisX_ * axisBS_ * axisK_;
    expertRankTableCnt_ = expertNum_ * expRankTableM_;
    ffnNumAlignSize_ = Ceil(ffnNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    aivWorkspaceOffset_ = Ceil(ffnNum_ * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET) * WORKSPACE_ELEMENT_OFFSET;
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId, GM_ADDR layerId, GM_ADDR expertIds,
    GM_ADDR expertRankTable, GM_ADDR scales, GM_ADDR activeMask, GM_ADDR workspaceGM, TPipe *pipe,
    const AttentionToFfnTilingData *tilingData)
{
    tpipe_ = pipe;
    if constexpr (contextType == AttentionToFFNContextType::HCCL_CONTEXT) {
        hcclContext_ = reinterpret_cast<__gm__ HcclOpResParam *>(context);
    } else {
        mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::AttentionFFNContext *>(context);
    }
    InitByTinglingData(tilingData);

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    sessionIdGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sessionId);
    microBatchIdGMTensor_.SetGlobalBuffer((__gm__ int32_t *)microBatchId);
    layerIdGMTensor_.SetGlobalBuffer((__gm__ int32_t *)layerId);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expertRankTableGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertRankTable);
    activeMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)activeMask);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sessionIdGMTensor_);
    sessionId_ = sessionIdGMTensor_.GetValue(0); // 当前x=1，session_ids直接从gm上读取第一个值
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(microBatchIdGMTensor_);
    microBatchId_ = microBatchIdGMTensor_.GetValue(0); // 当前x=1，microBatchId_直接从gm上读取第一个值
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(layerIdGMTensor_);
    layerId_ = layerIdGMTensor_.GetValue(0); // 当前x=1，layerId_直接从gm上读取第一个值
    layIdsExpRankTableOffset_ = layerId_ * expertRankTableCnt_;

    uint32_t expertIdsAlign = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // 约束32对齐
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsAlign);                                    // 对齐32B
    tpipe_->InitBuffer(ffnFlagBuf_, UB_ALIGN);                                            // 对齐32B
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    ffnFlagTensor_ = ffnFlagBuf_.Get<int32_t>();
    if constexpr (isQuant) {
        QuantInit(scales);
        castTempBuf_ = receiveDataCastFloatBuf_;
        sumOutBuf_ = smoothScalesBuf_;
    } else {
        tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hSize_); // H * 2
    }
    if constexpr (isSync) {
        tpipe_->InitBuffer(ffnStatusBuf_, ffnNumAlignSize_); // ffnNum_
        ffnStatusTensor_ = ffnStatusBuf_.Get<int32_t>();
        syncStatusWorkspaceGM_ = workspaceGM;
    }
    if constexpr (isActiveMask) {
        tpipe_->InitBuffer(activeMaskBuf_, axisBsAlignSize_); // BS
        if constexpr (!isQuant) {
            uint32_t bsAlignHalf = Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN; // 约束32对齐
            tpipe_->InitBuffer(castTempBuf_, bsAlignHalf);                            // BS * 2
            tpipe_->InitBuffer(sumOutBuf_, bsAlignHalf);                              // BS * 2
        }
        attnStatusBuf_ = expertIdsBuf_;
    }

    winOffset_[0] = 0;
    winOffset_[1] = Ceil(attentionWorkerNum_ * microBatchNum_ * infoTableLastDimNum_ * sizeof(int32_t), WIN_ALIGN) *
                    WIN_ALIGN; // token_info_table大小 偏移向上取整
    winInfoTableOffset_ = (sessionId_ * microBatchNum_ * infoTableLastDimNum_ + microBatchId_ * infoTableLastDimNum_) *
                          sizeof(int32_t); // tokenInfoTable上当前attnWorkId以及microBatchId偏移
    winTokenDataOffset_ =
        (sessionId_ * microBatchNum_ * axisBS_ * (axisK_ + sharedExpertNum_) * axisHS_) +
        (microBatchId_ * axisBS_ * (axisK_ + sharedExpertNum_) * axisHS_); // tokenData上attnWorkId以及microBatchId偏移
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::ActiveMaskCalCnt()
{ // 搬运x_active_mask, 当前仅用于计算有效token总数
    LocalTensor<bool> activeMaskTensor = activeMaskBuf_.Get<bool>();
    LocalTensor<half> tempTensor = castTempBuf_.Get<half>();
    LocalTensor<half> sumOutTensor = sumOutBuf_.Get<half>();
    DataCopyExtParams activeMaskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> activeMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(activeMaskTensor, activeMaskGMTensor_, activeMaskParams, activeMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> activeMaskInt8Tensor = activeMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, activeMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    curBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void
AttentionToFfn<TemplateAttentionToFfnTypeFunc>::ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count)
{
    uint64_t repsFp32 = count >> 6;       // 6 is count / elemPerRefFp32
    uint64_t offsetsFp32 = repsFp32 << 6; // 6 is repsFp32 * elemPerRefFp32
    uint64_t remsFp32 = count & 0x3f;     // 0x3f 63, count % elemPerRefFp32
    const uint64_t elemPerRefFp32 = 64UL; // 256 bit / sizeof(float)
    if (likely(repsFp32 > 1)) {           // 8 is rep stride
        Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, REP_STRIDE, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, REP_STRIDE, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
    // 8 is rep stride
    WholeReduceMax(srcLocal, srcLocal, mask, 1, REP_STRIDE, 1, REP_STRIDE);
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::QuantProcess(uint32_t expertIndex)
{
    float dynamicScale = 0.0;
    uint32_t hOutSizeAlign = Ceil(axisH_ * sizeof(int8_t), UB_ALIGN) * UB_ALIGN;
    LocalTensor<float> floatLocalTemp;
    floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();
    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    xInQueue_.FreeTensor<XType>(xInTensor_);

    if (isScales_) {
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
        DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(axisH_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesCopyInParams, copyPadExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor_, axisH_);
        PipeBarrier<PIPE_V>();
    }

    if (quantMode_ == DYNAMIC_QUANT) {
        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(floatLocalAbsTemp, axisH_);

        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(INT8_MAX_VALUE) / floatLocalAbsTemp.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

    PipeBarrier<PIPE_V>();
    Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    floatLocalTemp.SetValue(hOutSizeAlign / sizeof(float), float(1.0) / dynamicScale); // int8->float32
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::FindExpertRank(int32_t expertId)
{
    uint64_t expRankTableOffset = expertId * expRankTableM_ + layIdsExpRankTableOffset_;
    // 此前已刷新expertRankTableGM的整个Cache，可直接读取
    uint32_t rankCnt = expertRankTableGMTensor_.GetValue(expRankTableOffset); // 该专家部署在多少卡上
    uint32_t rankOffset = (sessionId_ % rankCnt) * RANK_OFFSET_STRIDE + 1;    // 第1位为rankCnt, 后面两两组合
    toRankId_ = expertRankTableGMTensor_.GetValue(expRankTableOffset + rankOffset); // 拿到当前要发送的卡号
    localExpId_ = expertRankTableGMTensor_.GetValue(expRankTableOffset + rankOffset + 1); // 拿到当前要发送的local专家号
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    startTokenId = sendTokenNum * aivId_;                   // 每个aiv发送时的起始rankid
    if (aivId_ < remainderTokenNum) {                       // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void
AttentionToFfn<TemplateAttentionToFfnTypeFunc>::SetExpertAndRank(uint32_t tokenIdx, uint32_t tokenId, uint32_t topkId)
{
    if (tokenIdx == 0) {
        SyncFunc<AscendC::HardEvent::MTE2_S>(); // 等待前面的expertIdsTensor_的搬入
    }
    if (topkId < axisK_) { // moe专家
        dstExpertId_ = expertIdsTensor_.GetValue(tokenId * axisK_ + topkId);
    } else {
        dstExpertId_ = moeExpertNum_ + (topkId - axisK_); // 前moeExpertNum_个为路由专家，共享专家排在后面
    }
    FindExpertRank(dstExpertId_); // 查表获取当前要发送的rankId以及localExpId
    statusTensor_.SetValue(tokenIdx * STATUS_REP_STRIDE, localExpId_);
    if constexpr (isSync) { // 异步场景记录Ffn节点发送状态
        if (tokenIdx == 0) {
            SyncFunc<AscendC::HardEvent::V_S>(); // 等待前面ffnStatusTensor_初始化
        }
        ffnStatusTensor_.SetValue(toRankId_ - ffnStartRankId_, 1);
    }
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline GM_ADDR AttentionToFfn<TemplateAttentionToFfnTypeFunc>::GetWindowAddr(uint32_t rankId)
{
    if constexpr (contextType == AttentionToFFNContextType::MC2_CONTEXT) {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId];
    } else {
        if (hcclContext_->userMemType == 0) {
            return (GM_ADDR)(((HcclRankRelationResV2 *)(hcclContext_->remoteRes[rankId].nextDevicePtr))->windowsIn);
        }
        return (GM_ADDR)(__gm__ uint8_t *)(hcclContext_->userMemRes[rankId].addr);
    }
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline GM_ADDR AttentionToFfn<TemplateAttentionToFfnTypeFunc>::GetLocalWindowAddr()
{
    if constexpr (contextType == AttentionToFFNContextType::MC2_CONTEXT) {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId_];
    } else {
        return (GM_ADDR)hcclContext_->localWindowsIn;
    }
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void
AttentionToFfn<TemplateAttentionToFfnTypeFunc>::CheckFlagAndSetTableGM(int32_t toRankId, GM_ADDR &toRankAddr,
                                                                       GlobalTensor<int32_t> &tokenInfoTableGMTensor)
{
    toRankAddr = GetWindowAddr(toRankId);
    GM_ADDR tokenInfoTable = (__gm__ uint8_t *)(toRankAddr + winInfoTableOffset_);
    tokenInfoTableGMTensor.SetGlobalBuffer((__gm__ int32_t *)tokenInfoTable);

    int32_t ffnFlage = 1;
    while (ffnFlage == 1) {
        DataCopy(ffnFlagTensor_, tokenInfoTableGMTensor, STATUS_REP_STRIDE);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        ffnFlage = ffnFlagTensor_.GetValue(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>(); // 等待flag
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::SendTokenToFfnByTokenIdx(uint32_t tokenIdx)
{
    GM_ADDR toRankAddr;
    GlobalTensor<int32_t> tokenInfoTableGMTensor;
    DataCopyParams statusCopyParams = {1U, static_cast<uint16_t>(1 * sizeof(int32_t)), 0U, 0U};
    DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U, 0U};
    DataCopyExtParams hCommuCopyOutParams = {1U, static_cast<uint32_t>(hCommuSize_), 0U, 0U, 0U};
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};

    uint32_t tokenOffset = aivId_ + (tokenIdx * aivNum_);         // 当前token在BS*(K+1)的偏移
    uint32_t tokenId = tokenOffset / (axisK_ + sharedExpertNum_); // 当前token在BS中的Id
    uint32_t topkId = tokenOffset % (axisK_ + sharedExpertNum_);  // 当前token的第几个topk
    if constexpr (isQuant) {
        GlobalTensor<int8_t> tokenDataGMTensor;
        xInTensor_ = xInQueue_.AllocTensor<XType>();
        DataCopyPad(xInTensor_, xGMTensor_[tokenId * axisH_], xCopyParams, copyPadExtParams);
        xInQueue_.EnQue(xInTensor_);
        xInTensor_ = xInQueue_.DeQue<XType>();
        xOutTensor_ = xOutQueue_.AllocTensor<int8_t>();
        SetExpertAndRank(tokenIdx, tokenId, topkId);
        QuantProcess(dstExpertId_);
        CheckFlagAndSetTableGM(toRankId_, toRankAddr, tokenInfoTableGMTensor);
        GM_ADDR tokenData =
            (__gm__ uint8_t *)(toRankAddr + winOffset_[1] +
                               (winTokenDataOffset_ + (tokenId * (axisK_ + sharedExpertNum_) * axisHS_) +
                                (topkId * axisHS_)) *
                                   sizeof(int8_t));
        tokenDataGMTensor.SetGlobalBuffer((__gm__ int8_t *)tokenData);
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<int8_t>();
        DataCopyPad(tokenDataGMTensor, xOutTensor_, hCommuCopyOutParams);
        xOutQueue_.FreeTensor<int8_t>(xOutTensor_);
    } else {
        GlobalTensor<XType> tokenDataGMTensor;
        xInTensor_ = xQueue_.AllocTensor<XType>();
        DataCopyPad(xInTensor_, xGMTensor_[tokenId * axisH_], xCopyParams, copyPadExtParams);
        SetExpertAndRank(tokenIdx, tokenId, topkId);
        CheckFlagAndSetTableGM(toRankId_, toRankAddr, tokenInfoTableGMTensor);
        GM_ADDR tokenData =
            (__gm__ uint8_t *)(toRankAddr + winOffset_[1] +
                               (winTokenDataOffset_ + (tokenId * (axisK_ + sharedExpertNum_) * axisHS_) +
                                (topkId * axisHS_)) *
                                   sizeof(XType));
        tokenDataGMTensor.SetGlobalBuffer((__gm__ XType *)tokenData);
        xQueue_.EnQue(xInTensor_);
        xInTensor_ = xQueue_.DeQue<XType>();
        DataCopyPad(tokenDataGMTensor, xInTensor_, xCopyParams);
        xQueue_.FreeTensor<XType>(xInTensor_);
    }

    // 当前Ffn节点的token Flag位
    DataCopyPad(tokenInfoTableGMTensor[TOKEN_INFO_TABLE_RS + tokenOffset], statusTensor_[tokenIdx * STATUS_REP_STRIDE],
                statusCopyParams);
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::SendTokenToFfn()
{
    uint32_t startId = 0;
    uint32_t endId = 0;
    totalSendNum_ = axisX_ * curBsCnt_ * (axisK_ + sharedExpertNum_); // 总发送数：axisX_ * curBsCnt_ * (axisK_ + 1)
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, copyPadParams);
    uint32_t aivWorkspaceStride = aivWorkspaceOffset_ / sizeof(int32_t);
    if constexpr (isSync) {
        ffnNumAlignCnt_ = ffnNumAlignSize_ / sizeof(int32_t);
        syncStatusGMTensor_.SetGlobalBuffer((__gm__ int32_t *)syncStatusWorkspaceGM_);
        Duplicate<int32_t>(ffnStatusTensor_, static_cast<int32_t>(0), ffnNumAlignCnt_); // 初始化workSpace
    }

    SplitToCore(totalSendNum_, aivNum_, startId, endId, sendNum_);
    uint32_t statusCnt = sendNum_ > 0 ? sendNum_ : 1;     // buf复用
    tpipe_->InitBuffer(statusBuf_, statusCnt * UB_ALIGN); // 对齐32B
    statusTensor_ = statusBuf_.Get<int32_t>();

    if (startId >= totalSendNum_) {
        if constexpr (isSync) {
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(syncStatusGMTensor_[aivId_ * aivWorkspaceStride], ffnStatusTensor_, ffnNumAlignCnt_);
        }
        return;
    }

    // 一次性刷新expertRankTableGM对应的Cache Line
    for (uint32_t idx = 0; idx < Ceil(expertRankTableCnt_, EXPERT_TABLE_REP_STRIDE); ++idx) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            expertRankTableGMTensor_[layIdsExpRankTableOffset_ + idx * EXPERT_TABLE_REP_STRIDE]);
    }

    for (uint32_t tokenIdx = 0; tokenIdx < sendNum_; ++tokenIdx) {
        SendTokenToFfnByTokenIdx(tokenIdx);
    }
    if constexpr (isSync) { // 异步场景存储token发送状态到workspace
        DataCopy(syncStatusGMTensor_[aivId_ * aivWorkspaceStride], ffnStatusTensor_, ffnNumAlignCnt_);
    }
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::SetFlagToFfn()
{
    // 同步所有Ffn节点全发 异步只有选中的节点发送
    uint32_t startFfnId = 0;
    uint32_t endFfnId = 0;
    uint32_t sentFfnNum = 0;
    SplitToCore(ffnNum_, aivNum_, startFfnId, endFfnId, sentFfnNum);
    statusTensor_.SetValue(0, 1);
    statusTensor_.SetValue(1, layerId_);
    GlobalTensor<int32_t> tableFlagGMTensor;
    DataCopyExtParams statusParams = {1U, static_cast<uint32_t>(sizeof(int32_t) * TOKEN_INFO_TABLE_COPY_BLOCK_CNT), 0U,
                                      0U, 0U};
    if (startFfnId >= ffnNum_) {
        return;
    }

    startFfnId += ffnStartRankId_;
    endFfnId += ffnStartRankId_;
    GM_ADDR toRankAddr;
    if constexpr (isSync) {
        uint32_t sentFfnNumAlignSize = Ceil(sentFfnNum * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
        DataCopyExtParams syncStatusParams = {
            static_cast<uint16_t>(aivNum_), static_cast<uint32_t>(sentFfnNum * sizeof(int32_t)),
            static_cast<uint32_t>(aivWorkspaceOffset_ - sentFfnNum * sizeof(int32_t)), 0U, 0U};
        DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
        tpipe_->InitBuffer(syncStatusWorkspaceBuf_, aivNum_ * sentFfnNumAlignSize);
        syncStatusWorkspaceTensor_ = syncStatusWorkspaceBuf_.Get<int32_t>();
        DataCopyPad(syncStatusWorkspaceTensor_, syncStatusGMTensor_[startFfnId - ffnStartRankId_], syncStatusParams,
                    copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        LocalTensor<float> syncStatusWorkspaceTensorFloat = syncStatusWorkspaceTensor_.ReinterpretCast<float>();
        LocalTensor<float> ffnStatusTensorFloat = ffnStatusTensor_.ReinterpretCast<float>();
        const uint32_t shape[] = {aivNum_, static_cast<uint32_t>(sentFfnNumAlignSize / sizeof(float))};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(ffnStatusTensorFloat,
                                                                      syncStatusWorkspaceTensorFloat, shape, true);
        SyncFunc<AscendC::HardEvent::V_S>();
    }

    for (uint32_t ffnIdx = startFfnId; ffnIdx < endFfnId; ++ffnIdx) {
        if constexpr (isSync) {
            if (ffnStatusTensor_.GetValue(ffnIdx - startFfnId) == 0) {
                continue;
            }
        }
        toRankAddr = GetWindowAddr(ffnIdx);
        GM_ADDR tableFlagGM = (__gm__ uint8_t *)(toRankAddr + winInfoTableOffset_);
        tableFlagGMTensor.SetGlobalBuffer((__gm__ int32_t *)tableFlagGM);
        if (ffnIdx == startFfnId) {
            SyncFunc<AscendC::HardEvent::S_MTE3>();
        }
        DataCopyPad(tableFlagGMTensor, statusTensor_, statusParams);
    }
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::SetFlagInAttn()
{
    // 未mask数据总量：axisX_ * (axisBS_ - curBsCnt_) * (axisK_ + sharedExpertNum_)
    uint32_t startId = 0;
    uint32_t endId = 0;
    uint32_t sendNum = 0;
    uint32_t totalNum = axisX_ * (axisBS_ - curBsCnt_) * (axisK_ + sharedExpertNum_);
    if (totalNum == 0) {
        return;
    }
    SplitToCore(totalNum, aivNum_, startId, endId, sendNum);
    if (startId >= totalNum) {
        return;
    }

    uint64_t sendMaskTokenCnt = curBsCnt_ * (axisK_ + sharedExpertNum_);
    uint64_t attnTokenInfoTableOffset =
        (microBatchId_ * axisBS_ * (axisK_ + sharedExpertNum_) + sendMaskTokenCnt) * sizeof(int32_t);
    GM_ADDR selfRankAddr = GetLocalWindowAddr();
    GM_ADDR attnTokenInfoTableGM = (__gm__ uint8_t *)(selfRankAddr + attnTokenInfoTableOffset);
    GlobalTensor<int32_t> attnTableGMTensor;
    attnTableGMTensor.SetGlobalBuffer((__gm__ int32_t *)attnTokenInfoTableGM);
    DataCopyExtParams dataCopyParams = {1U, static_cast<uint32_t>(sendNum * sizeof(int32_t)), 0U, 0U, 0U};
    LocalTensor<int32_t> tempTensor = attnStatusBuf_.Get<int32_t>();
    Duplicate<int32_t>(tempTensor, (uint32_t)1, sendNum);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(attnTableGMTensor[startId], tempTensor, dataCopyParams);
}

template <TemplateAttentionToFfnTypeClass>
__aicore__ inline void AttentionToFfn<TemplateAttentionToFfnTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if constexpr (isActiveMask) {
            ActiveMaskCalCnt();
        }
        SendTokenToFfn();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        SetFlagToFfn();
        if constexpr (isActiveMask) { // activeMask attn将die状态区上的未mask的token flag位先写上
            SetFlagInAttn();
        }
    }
}
} // namespace AttentionToFfnImpl
#endif
