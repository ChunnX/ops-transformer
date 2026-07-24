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
 * \file moe_distribute_dispatch_v2_a5_atomic.h
 * \brief A5 原子定序实现（仅 A5 && BS<=4 && MTE && 非 fullmesh/layered 路由到本文件；
 *        每专家原子计数器定序 + workspace 暂存密排重排 + 按专家分核发送）
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_A5_ATOMIC_H
#define MOE_DISTRIBUTE_DISPATCH_V2_A5_ATOMIC_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_dispatch_v2_tiling.h"
#include "../moe_distribute_dispatch_v2_quant.h"
#include "../moe_distribute_elastic.h"

#include "../check_winsize.h"
#include "../../../common/op_kernel/mc2_moe_context.h"
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "../hccl_context_holder.h"
#include "../mc2_moe_context_holder.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60
namespace Mc2Kernel {
#define TemplateDispatchV2AtomicTypeClass \
    typename ContextHolder, typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist
#define TemplateDispatchV2AtomicTypeFunc ContextHolder, XType, ExpandXOutType, QuantMode, IsSmoothScaleExist

using namespace AscendC;
using namespace MoeDistributeV2Base;
using namespace Mc2Aclnn;

// 本文件固定走按专家分核发送（bitmask）：SendToMoeExpertByExpert 按专家分核发 token，
// SetStatusByExpert 同分核写 flag（同核 MTE3 保序，纯 MOE 免 SyncAll），不含 token 打平基线路径
constexpr uint32_t SFFVALUE_BITS = 64U; // ScalarGetSFFValue 按 64 bit 一次

// 原子定序：每专家独立原子计数器区（状态区尾段，PINGPONG 两块，基址 statusDataSpaceGm_）
constexpr uint64_t EXPERT_COUNTERS_OFFSET = 880UL * 1024UL;            // dataState_ == 0
constexpr uint64_t EXPERT_COUNTERS_PINGPONG_OFFSET = 890UL * 1024UL;   // dataState_ == 1
constexpr uint32_t COUNTER_STRIDE = WIN_ADDR_ALIGN / sizeof(uint32_t); // GM 内每计数器 512B（AtomicAdd 对齐）

template <TemplateDispatchV2AtomicTypeClass>
class MoeDistributeDispatchV2Atomic {
public:
    using XInType = typename std::conditional<
        (Std::IsSame<XType, fp4x2_e2m1_t>::value) || (Std::IsSame<XType, fp4x2_e1m2_t>::value), uint8_t, XType>::type;
    using XOutType = typename std::conditional<(Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value) ||
                                                   (Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value),
                                               uint8_t, ExpandXOutType>::type;
    __aicore__ inline MoeDistributeDispatchV2Atomic(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                GM_ADDR expertScales, GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expandScalesOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessToken(GlobalTensor<XOutType> &outTokenGT, uint32_t tokenIndex, uint32_t topKIndex,
                                        DataCopyPadParams &padParams, DataCopyParams &scaleInParams,
                                        uint32_t expertIndex, bool writeExpertScale);
    __aicore__ inline void SendToSharedExpert();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void ExpIdsCopyAndMaskCal();
    __aicore__ inline void SplitExpertNumToCore();
    __aicore__ inline void CalExpertSendNum();
    __aicore__ inline void SendToMoeExpertByExpert();
    __aicore__ inline void SetStatusByExpert();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void SetExpertTokenNums();
    __aicore__ inline void ResetNextRoundCounters();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void CalValidBSCnt(LocalTensor<bool> maskStrideTensor);
    __aicore__ inline void CalValidExpIdx(LocalTensor<bool> maskInputTensor);
    __aicore__ inline void GenerateGatherMaskTensor(uint32_t maskCnt);
    __aicore__ inline void MaskZeroComputeExpert(uint32_t maskCnt);
    __aicore__ inline void ZeroComputeExpertMaskCal();
    __aicore__ inline void InitPerformanceBuffers();
    __aicore__ inline void BufferInit();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void FillTriple(LocalTensor<XOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k,
                                      bool writeExpertScale);
    __aicore__ inline void WaitStatusA5(LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline void PerformanceInfoPerRank(uint64_t performanceTimeStart, bool isCompactStatus);
    __aicore__ inline void PerformanceInfoCopyOut();
    __aicore__ inline void RunPosRecord();
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindAddrByRankId(rankId, epRankIdOriginal_) + winDataSizeOffsetEp_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindStateAddrByRankId(rankId, epRankIdOriginal_) + dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    __aicore__ inline int32_t ReduceSumWorkNeedSize(int32_t calCnt)
    {
        int typeSize = static_cast<int>(sizeof(int32_t));
        int32_t elementsPerBlock = 32 / typeSize;
        int32_t iter1OutputCount = calCnt;
        int32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        return iter1AlignEnd;
    }

    __aicore__ inline uint64_t ExpertCountersRegionOffset(uint32_t dataState)
    {
        return (dataState == 0U) ? EXPERT_COUNTERS_OFFSET : EXPERT_COUNTERS_PINGPONG_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XInType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<float> expertScalesGMTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<int32_t> performanceInfoGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;
    GlobalTensor<uint32_t> selfhcclDataStatusTensor_;
    GlobalTensor<int32_t> sendCountsGlobal_;
    GlobalTensor<uint32_t>
        expertCountersGMTensor_; // 原子定序：每专家独立计数器（statusDataSpaceGm_ + EXPERT_COUNTERS_OFFSET）

    LocalTensor<XOutType> xTmpTensor_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<XInType> xInTensor_;
    LocalTensor<XOutType> xOutTensor_;
    LocalTensor<float> xOutFp32Tensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> expertScalesTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> validExpertIndexTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<int32_t> performanceTmpTensor_;
    LocalTensor<int32_t> firstRecordTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    // SetExpertTokenNums 使用（本文件仅 A5 编译单元包含，无需 arch 守卫）
    LocalTensor<int64_t> expertTokenNumsLocalTensor_;

    TBuf<> expertIdsBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> statusBuf_;
    TBuf<> scalarBuf_; // 辅助gather tensor定义
    TBuf<> rowMaxBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> maskBuf_;
    TBuf<> validExpertIndexBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> performanceTmpBuf_;
    TBuf<> firstRecordBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> expertMaskBuf_;       // per-expert bitmask（按专家分核发送用）
    TBuf<> tokenNumToExpertBuf_; // per-expert token 计数（CalExpertSendNum 写、SendToMoeExpertByExpert 读）
    LocalTensor<int32_t> tokenNumToExpertTensor_;
    uint32_t maskSizePerExpert_{0};
    // 步进分核发送专用（不复用 startExpertId_/sendExpertNum_，后者在接收侧 WaitDispatch/LocalWindowCopy 复用）
    uint32_t startId_{0};                                         // 本核首个 MOE 专家全局 id = aivId_
    uint32_t sendNum_{0};                                         // 本核步进分到的 MOE 专家数
    uint32_t remainderExpertNum_{0};                              // moeExpertNum_ % moeUsedAivNum_
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, 1> xInQueue_;                        // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;                      // 量化使用，量化后的输出
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tripleQueue_;
    uint32_t tripleDataSize_{0};

    GM_ADDR expandXOutGM_;
    GM_ADDR expandIdxOutGM_;
    GM_ADDR expandScalesOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    GM_ADDR statusDataSpaceGm_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    uint32_t epRankId_{0};
    int32_t epRankIdOriginal_{0};
    uint32_t aivId_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};    // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0}; // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t globalBS_{0};
    uint32_t moeExpertRankNum_{0}; // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t totalExpertNum_{0};
    uint32_t dealRankPerCore_{0};
    uint32_t hOutSize_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t stagingTokenBytes_{0}; // 原子定序：单个完整 Token 暂存字节 = hOutAlignUbSize_
    uint32_t counterRegionSize_{0}; // 原子定序：一块计数器区大小（reset 用，按 UB 对齐）
    uint32_t hAlignSize_{0};
    uint32_t hScaleIdxSize_{0};
    uint32_t expertScaleAlign_{0};
    uint32_t scaleInBytes_{0};
    uint32_t scaleOutBytes_{0};
    uint32_t scalesCount_{0};
    uint32_t startExpertId_;
    uint32_t endExpertId_;
    uint32_t sendExpertNum_;
    uint32_t dataState_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t totalUsedUB_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t recvWinBlockNum_; // 接收Win区块数
    uint64_t sendToMoeExpTokenCnt_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isPerformanceFlag_ = false;
    uint32_t cumsumTmpMinSize_ = 0;
    bool isScalingDownFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool hasExpertScalesFlag_ = false;
    float sumTarget_;
    uint64_t totalWinSizeEp_{0};
    uint32_t totalUbSize_{0};
    uint32_t expertTokenNumsType_{1};
    uint32_t stateOffset_{0};
    uint32_t recStatusNumPerCore_{0};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    int32_t zeroComputeExpertNum_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t remainderRankNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t sendToSharedExpTokenCnt_{0};
    uint32_t maxSize_{0};
    uint32_t bufferNum_{0};
    uint32_t copyInAxisH_{0};
    uint32_t copyOutAxisH_{0};

    // 统一上下文持有器：v2 走 HcclOpParam，v3 走 Mc2MoeContext
    ContextHolder ctx_;

    DataCopyParams expandXCopyParams_;
    DataCopyParams xCopyParams_;
    DataCopyParams hCommuCopyOutParams_;
    DataCopyExtParams scaleOutParams_;

    MoeDistributeDispatchV2Quant<XInType, ExpandXOutType, XOutType, QuantMode, IsSmoothScaleExist> quantInst_;
    MoeDistributeElastic elasticInst_;
};

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
    GM_ADDR expandScalesOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    // A3不支持MX量化，无需使能饱和模式
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    epRankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    epRankIdOriginal_ = tilingData->moeDistributeDispatchV2Info.epRankId;

    // 检查hcclwinsize是否越界
    totalWinSizeEp_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    ctx_.InitAndCheck(mc2Context, tilingData->moeDistributeDispatchV2Info.epWorldSize, totalWinSizeEp_, tpipe_,
                      expandXOut);

    totalUbSize_ = static_cast<uint32_t>(tilingData->moeDistributeDispatchV2Info.totalUbSize);

    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    epWorldSizeOriginal_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    hasElasticInfoFlag_ = tilingData->moeDistributeDispatchV2Info.hasElasticInfo;
    isPerformanceFlag_ = tilingData->moeDistributeDispatchV2Info.isPerformance;
    cumsumTmpMinSize_ = tilingData->moeDistributeDispatchV2Info.cumsumTmpMinSize;
    hasExpertScalesFlag_ = tilingData->moeDistributeDispatchV2Info.hasExpertScales;
    epWorldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    globalBS_ = tilingData->moeDistributeDispatchV2Info.globalBs;
    scaleInBytes_ =
        tilingData->moeDistributeDispatchV2Info.scalesCol * tilingData->moeDistributeDispatchV2Info.scalesTypeSize;
    scalesCount_ = tilingData->moeDistributeDispatchV2Info.scalesCount;

    copyInAxisH_ = axisH_;
    copyOutAxisH_ = axisH_;
    if constexpr (Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value ||
                  Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value) {
        copyOutAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
    }
    if constexpr (Std::IsSame<XType, fp4x2_e2m1_t>::value || Std::IsSame<XType, fp4x2_e1m2_t>::value) {
        copyInAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
    }

    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    statusDataSpaceGm_ = ctx_.GetStatusDataSpaceGm();
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempEpRankId = 0; tempEpRankId < epWorldSize_; tempEpRankId++) {
        OOMCheckAddrRange<XOutType>((__gm__ XOutType *)(GetWindAddrByRankId(tempEpRankId)), totalWinSizeEp_);
        OOMCheckAddrRange<float>((__gm__ float *)(GetWindStateAddrByRankId(tempEpRankId)), STATE_SIZE);
    }
#endif
    uint32_t epRankIdHccl = ctx_.GetEpRankId();
    uint32_t epWorldSizeHccl = ctx_.GetEpWorldSize();
    selfDataStatusGMTensor_.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusDataSpaceGm_ + DISPATCH_STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_, moeExpertNum_,
                              epWorldSizeOriginal_, globalBS_, dataStateBuf);
    elasticInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(elasticInfo));
    performanceInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(performanceInfo));
    elasticInst_.SetElasticInitParams(tpipe_, elasticInfoGMTensor_);
    if (hasElasticInfoFlag_) {
        totalUsedUB_ += elasticInst_.GetElasticInfoSizeAlign(epWorldSizeOriginal_);
        elasticInst_.InitElasticInfoTensor(epWorldSizeOriginal_, elasticInfoTensor_);
        isScalingDownFlag_ = elasticInfoTensor_.GetValue(0);
        elasticInst_.InitElasticInfo(isScalingDownFlag_, epWorldSize_, sharedExpertRankNum_, moeExpertNum_, epRankId_,
                                     moeExpertNumPerRank_);
    }

    if (epRankId_ < sharedExpertRankNum_) {
        isShareExpertRankFlag_ = true;
    }

    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    zeroComputeExpertNum_ = tilingData->moeDistributeDispatchV2Info.zeroComputeExpertNum;
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;

    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isExpertMask;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    xGMTensor_.SetGlobalBuffer((__gm__ XInType *)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ uint8_t *)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOut));
    sendCountsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOut));
    if constexpr (IsSmoothScaleExist) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    }
    expandXOutGM_ = expandXOut;
    recvCntWorkspaceGM_ = workspaceGM;

    hOutSize_ = copyOutAxisH_ * sizeof(XOutType);
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, tokenQuantAlign_, hScaleIdxSize_, scaleOutBytes_,
                         axisH_);
    if (hasExpertScalesFlag_) {
        expertScaleAlign_ = tokenQuantAlign_ + EXPAND_IDX_INFO;
        hScaleIdxSize_ += sizeof(float);
    }
    hAlignWinSize_ = Ceil(hScaleIdxSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // win区token起始地址对齐512
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(XOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hAlignWinSize_;
    hOutAlignUbSize_ = Ceil(hScaleIdxSize_, UB_ALIGN) * UB_ALIGN;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivNum_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    expertIdsCnt_ = axisBS_ * axisK_;
    recvWinBlockNum_ = epWorldSize_ * moeExpertNumPerRank_;
    moeUsedAivNum_ = aivNum_ - sharedUsedAivNum_;
    dealRankPerCore_ = (recvWinBlockNum_ + aivNum_ - 1) / aivNum_;
    stateOffset_ = STATE_OFFSET;
    PipeBarrier<PIPE_ALL>();
    if (isShareExpertRankFlag_) {
        rscvStatusNum_ = epWorldSize_;
    } else {
        rscvStatusNum_ = recvWinBlockNum_;
    }
    recStatusNumPerCore_ = rscvStatusNum_ / aivNum_; // 每个aiv需要处理的专家数
    remainderRankNum_ = rscvStatusNum_ % aivNum_;
    startStatusIndex_ = recStatusNumPerCore_ * aivId_; // + sharedExpertRankNum_, 每个aiv发送的
    if (aivId_ < remainderRankNum_) {                  // 前remainderRankNum个aiv需要多发1个卡的数据
        recStatusNumPerCore_ += 1;
        startStatusIndex_ += aivId_;
    } else {
        startStatusIndex_ += remainderRankNum_;
    }

    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    // SetStatusByExpert 与 token 发送同分核（front 步进 / rear 连续），statusTensor_ 按核内相对索引
    // [0, sendNum_) / [0, sharedSendNum) 使用（UB 每核私有，front/rear 互不冲突）。容量取
    // ceil(moeExpertNum_/moeUsedAivNum_) 与 ceil(sharedExpertRankNum_/sharedUsedAivNum_) 较大者
    // （moeUsedAivNum_ 可能 < aivNum_，步进分核下单核专家数大于 token-flat 的 totalExpertNum_/aivNum_）。
    uint32_t moeSendNumMax = Ceil(moeExpertNum_, moeUsedAivNum_);
    uint32_t sharedSendNumMax = (sharedExpertRankNum_ == 0) ? 0 : Ceil(sharedExpertRankNum_, sharedUsedAivNum_);
    uint32_t statusBlockMax = (moeSendNumMax > sharedSendNumMax) ? moeSendNumMax : sharedSendNumMax;
    uint32_t statusBufCntAlign = Ceil(statusBlockMax, 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    uint32_t statusBufSize = statusBufCntAlign * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    totalUsedUB_ += statusBufSize;
    statusTensor_ = statusBuf_.Get<int32_t>(); // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, statusBufCntAlign * 8); // 8 = UB_ALIGN / sizeof(int32_t)
    statusSpaceGm_ = GetWindStateAddrByRankId(epRankIdOriginal_);
    sumTarget_ = static_cast<float>(1.0);
    uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign / 8, 1, 8); // 0x3F800000是float的1

    // 当前tpWin区划分为前后两半区，连续两次dispatch，切换半区, combine 数据区使用前面，
    // 即axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine, dispatch使用后面
    // DataAsFlag场景每512B有效数据为480B
    uint64_t hSizeAlignCombine = Ceil(axisH_ * COMBINE_IN_DATA_SIZE, 480) * WIN_ADDR_ALIGN;
    winDataSizeOffsetEp_ =
        dataState_ * (totalWinSizeEp_ / 2) + axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;

    windowGM_ = GetWindAddrByRankId(epRankIdOriginal_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    GlobalTensor<XOutType> winDouble;
    winDouble.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    winDouble.SetGlobalBuffer((__gm__ XOutType *)(windowGM_));
    OOMCheckAddrRange<XOutType>((__gm__ XOutType *)(winDouble.GetPhyAddr()), totalWinSizeEp_);
#endif
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));

    // 原子定序：每专家独立原子计数器，绑定到 statusDataSpaceGm_ + PINGPONG 偏移
    stagingTokenBytes_ = hOutAlignUbSize_; // 单个完整 Token 暂存字节
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1U : moeExpertNumPerRank_;
    counterRegionSize_ = Ceil(localExpertNum * COUNTER_STRIDE * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    expertCountersGMTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    expertCountersGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>((__gm__ uint8_t *)(statusDataSpaceGm_) +
                                                                                ExpertCountersRegionOffset(dataState_)),
                                            localExpertNum * COUNTER_STRIDE);

    expertIdsCnt_ = axisBS_ * axisK_;
    uint32_t hFp32Size = axisH_ * sizeof(float);
    if constexpr ((QuantMode == PERGROUP_DYNAMIC_QUANT) && IsSmoothScaleExist) {
        hFp32Size = Align128(axisH_) * sizeof(float);
    }
    uint32_t expertIdsSize = expertIdsCnt_ * sizeof(int32_t);
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    uint32_t bsKAlign256 = Ceil(expertIdsCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    // CompareScalar 按 256B(ALIGNED_LEN_256) 粒度处理，compareCount=ceil(mask/64)*64；
    // expertIdsBuf_ 须按 256B 对齐申请，否则 BS*K%64!=0 时 compareCount 越界读（对标 full_mesh L1103）
    uint32_t expertIdsBufSize = Ceil(expertIdsSize, ALIGNED_LEN_256) * ALIGNED_LEN_256;
    expertIdsSize = Ceil(expertIdsSize, UB_ALIGN) * UB_ALIGN;
    maxSize_ = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSize_ = maxSize_ > xActivateMaskSize ? maxSize_ : xActivateMaskSize;
    maxSize_ = maxSize_ > bsKAlign256 ? maxSize_ : bsKAlign256;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize); // BS * K * 4 = 32K
    totalUsedUB_ += expertIdsSize;
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    // bitmask buffer：maskSizePerExpert_ = Ceil(BS*K bits → bytes, UB_ALIGN)；总 = per-expert × 核分到的 expert 数
    maskSizePerExpert_ = Ceil(Ceil(expertIdsCnt_, 8U), UB_ALIGN) * UB_ALIGN;
    uint32_t maxExpertPerCore = Ceil(moeExpertNum_, moeUsedAivNum_); // 步进分核下单核最多拥有的 MOE 专家数
    uint32_t expertMaskBufSize = maskSizePerExpert_ * maxExpertPerCore;
    tpipe_->InitBuffer(expertMaskBuf_, expertMaskBufSize);
    totalUsedUB_ += expertMaskBufSize;
    // tokenNumToExpertBuf_ 需容纳 sendNum_(≤maxExpertPerCore) 个 int32，按 UB_ALIGN 对齐申请
    uint32_t tokenNumBufSize = Ceil(maxExpertPerCore * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tokenNumToExpertBuf_, tokenNumBufSize);
    totalUsedUB_ += tokenNumBufSize;
    tokenNumToExpertTensor_ = tokenNumToExpertBuf_.Get<int32_t>();
    LocalTensor<int32_t> expertMaskInitTensor = expertMaskBuf_.Get<int32_t>();
    Duplicate<int32_t>(expertMaskInitTensor, 0, expertMaskBufSize / sizeof(int32_t));
    tpipe_->InitBuffer(gatherMaskTBuf_, maxSize_);
    totalUsedUB_ += maxSize_;
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();
    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        uint32_t axisBSAlign = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(validBsIndexTBuf_, axisBSAlign);
        totalUsedUB_ += axisBSAlign;
        uint32_t validBufferSize = expertIdsSize > xActivateMaskSize ? expertIdsSize : xActivateMaskSize;
        tpipe_->InitBuffer(validExpertIndexBuf_, validBufferSize);
        totalUsedUB_ += expertIdsSize;
        validExpertIndexTensor_ = validExpertIndexBuf_.Get<int32_t>();
        validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
    }

    if (hasExpertScalesFlag_) {
        expandScalesOutGM_ = expandScalesOut;
        expertScalesGMTensor_.SetGlobalBuffer((__gm__ float *)expertScales);
        tpipe_->InitBuffer(expertScalesBuf_, expertIdsBufSize);
        totalUsedUB_ += expertIdsBufSize;
    }

    if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
        tpipe_->InitBuffer(receiveDataCastFloatBuf_, maxSize_); // max{28K, BS * K * 4B}
        totalUsedUB_ += maxSize_;
        floatLocalTemp_ = receiveDataCastFloatBuf_.Get<float>();
        tpipe_->InitBuffer(smoothScalesBuf_, maxSize_); // max{28K, hAlign * 4B}
        totalUsedUB_ += maxSize_;
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
        if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN); // 32B
            rowMaxTensor_ = rowMaxBuf_.Get<float>();
        }
        uint32_t tmpTotalUB = totalUsedUB_ + BUFFER_NUM * hAlignSize_ + hOutAlignUbSize_ * BUFFER_NUM;
        bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
        tpipe_->InitBuffer(xInQueue_, bufferNum_, hAlignSize_);       // 14K * 2
        tpipe_->InitBuffer(xOutQueue_, bufferNum_, hOutAlignUbSize_); // 7K * 2 + 32 + 6
        dstExpBuf_ = receiveDataCastFloatBuf_;                        // 内存复用
        subExpBuf_ = smoothScalesBuf_;                                // 内存复用
    } else {
        tpipe_->InitBuffer(dstExpBuf_, maxSize_); // BS * K * 4 = 32K
        totalUsedUB_ += maxSize_;
        tpipe_->InitBuffer(subExpBuf_, maxSize_); // BS * K * 4 = 32K
        totalUsedUB_ += maxSize_;
        uint32_t tmpTotalUB = totalUsedUB_ + hOutAlignUbSize_ * BUFFER_NUM;
        bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
        tpipe_->InitBuffer(xQueue_, bufferNum_, hOutAlignUbSize_); // 7k*2 + 32 + 12
    }

    uint32_t axisHCommu = hScaleIdxSize_ / sizeof(XOutType); // 有效搬运长度
    xCopyParams_ = {1U, static_cast<uint16_t>(copyInAxisH_ * sizeof(XInType)), 0U, 0U};
    hCommuCopyOutParams_ = {1U, static_cast<uint16_t>(axisHCommu * sizeof(XOutType)), 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint16_t>(hOutSize_), 0U, 0U};
    scaleOutParams_ = {1U, static_cast<uint16_t>(scaleOutBytes_), 0U, 0U, 0U};

    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf_, dynamicScalesOutGMTensor_);
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum,
    bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else {
        newAivId = aivId_ - moeUsedAivNum_; // 由于是后面的核作为发送的共享专家，因此需要换算
    }
    startTokenId = sendTokenNum * newAivId; // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {     // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::FillTriple(
    LocalTensor<XOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k, bool writeExpertScale)
{
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = epRankId_;
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex;
    xOutTint32(tokenQuantAlign_ + 2) = k;
    if (writeExpertScale) {
        LocalTensor<float> xOutTfloat = xOutTensor.template ReinterpretCast<float>();
        xOutTfloat(expertScaleAlign_) = expertScalesTensor_.GetValue(tokenIndex * axisK_ + k);
    }
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::ProcessToken(
    GlobalTensor<XOutType> &outTokenGT, uint32_t tokenIndex, uint32_t topKIndex, DataCopyPadParams &padParams,
    DataCopyParams &scaleInParams, uint32_t expertIndex, bool writeExpertScale)
{
    if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
        xInTensor_ = xInQueue_.AllocTensor<XInType>();
        LocalTensor<uint8_t> singleByteTok = xInTensor_.template ReinterpretCast<uint8_t>();
        if constexpr ((QuantMode == MX_QUANT) || (QuantMode == MX_QUANT_CLIP) ||
                      (QuantMode == PERGROUP_DYNAMIC_QUANT)) { // 提前填0，避免脏数据
            Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XInType));
        }
        SyncFunc<HardEvent::V_MTE2>();
        DataCopyPad(xInTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, padParams);
        xInQueue_.EnQue(xInTensor_);
        xInTensor_ = xInQueue_.DeQue<XInType>();
        xOutTensor_ = xOutQueue_.AllocTensor<XOutType>();
        if constexpr (QuantMode > UNQUANT) {
            quantInst_.QuantProcess(xOutTensor_, xInTensor_, expertIndex, scalesCount_, scalesGMTensor_); // 量化
        } else { // 非量化直转hifp8
            Cast(floatLocalTemp_, xInTensor_, RoundMode::CAST_NONE, axisH_);
            Cast(xOutTensor_, floatLocalTemp_, RoundMode::CAST_ROUND, axisH_);
        }
        SyncFunc<AscendC::HardEvent::V_S>();
        FillTriple(xOutTensor_, tokenIndex, topKIndex, writeExpertScale);
        xOutQueue_.EnQue(xOutTensor_);
        xInQueue_.FreeTensor<XInType>(xInTensor_);
        xOutTensor_ = xOutQueue_.DeQue<XOutType>();
        DataCopyPad(outTokenGT, xOutTensor_, hCommuCopyOutParams_);
        xOutQueue_.FreeTensor<XOutType>(xOutTensor_);
    } else {
        xTmpTensor_ = xQueue_.AllocTensor<XOutType>();
        if constexpr (!IsSmoothScaleExist) {
            DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, padParams);
        } else {
            auto tmp = scalesGMTensor_.ReinterpretCast<uint8_t>();
            DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * copyInAxisH_], xCopyParams_, padParams);
            DataCopyPad(xTmpTensor_[Align32(copyInAxisH_)].template ReinterpretCast<uint8_t>(),
                        tmp[tokenIndex * scaleInBytes_], scaleInParams, padParams);
        }
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        FillTriple(xTmpTensor_, tokenIndex, topKIndex, writeExpertScale);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XOutType>();
        DataCopyPad(outTokenGT, xTmpTensor_, hCommuCopyOutParams_);
        xQueue_.FreeTensor<XOutType>(xTmpTensor_);
    }
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::SendToSharedExpert()
{
    uint32_t curSendCnt = activeMaskBsCnt_ * sharedExpertNum_;
    uint32_t startTokenId, endTokenId, sendTokenNum; // 每个aiv发送时的起始rankid
    SplitToCore(curSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= curSendCnt) {
        return;
    }

    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_; // 计算目的共享专家卡在其所在共享专家组的id
    GlobalTensor<XOutType> dstWinGMTensor;
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};
    for (uint32_t virtualTokenIndex = startTokenId; virtualTokenIndex < endTokenId; ++virtualTokenIndex) {
        uint32_t tokenIndex = virtualTokenIndex % activeMaskBsCnt_;
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        int32_t toRankId = idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_;
        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
        }
        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ XOutType *)(GetWindAddrByRankId(toRankId) + expertPerSizeOnWin_ * epRankId_));
        uint32_t srcTokenIndex = tokenIndex;
        if (isExpertMaskFlag_) {
            srcTokenIndex = validBsIndexTensor_.GetValue(tokenIndex);
        }
        GlobalTensor<XOutType> tempTensor = dstWinGMTensor[tokenIndex * hAlignWinCnt_];
        ProcessToken(tempTensor, srcTokenIndex, axisK_ + toSharedExpertIndex, padParams, scaleInParams,
                     toSharedExpertIndex, false);
    }
}

// 步进分核：core i 处理全局 MOE 专家 i, i+stride, i+2*stride, ...（stride=moeUsedAivNum_）。
// 跨区间打散热专家以均衡负载；并使"同一核发送 token 与写 flag"对齐（SetStatusByExpert 同分核），
// MTE3 同队列保序，纯 MOE 场景免 SyncAll。对标参考实现 atomic_expert.h SplitExpertNumToCore。
template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::SplitExpertNumToCore()
{
    if (moeUsedAivNum_ == 0) {
        return; // 成员默认初始化为 0，sendNum_==0 使调用方早退（正常路径不会到此）
    }
    uint32_t stride = moeUsedAivNum_;
    remainderExpertNum_ = moeExpertNum_ % stride;
    sendNum_ = moeExpertNum_ / stride;
    startId_ = aivId_;
    if (aivId_ < remainderExpertNum_) {
        sendNum_ += 1;
    }
}

// 预建 per-expert bitmask + token count（startId_ 已是 moe 专家空间，无需 shared 偏移）
template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::CalExpertSendNum()
{
    // expertIdsTensor_ 为"原位选择 + (-1) 哨兵"全量数组（见 AlltoAllDispatch），哨兵不等于任何 expert id，
    // 全量比较计数安全；1D token-mask 时仅前缀 activeMaskBsCnt_*K 为实数据，mask 取前缀长度
    uint32_t mask = isTokenMaskFlag_ ? (static_cast<uint32_t>(activeMaskBsCnt_) * axisK_) : expertIdsCnt_;
    uint32_t compareCount = Ceil(mask * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    LocalTensor<uint8_t> expertMaskTensorU8 = expertMaskBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> expertMaskTensorU32 = expertMaskBuf_.Get<uint32_t>();
    LocalTensor<int32_t> gatherTempTensor = dstExpBuf_.Get<int32_t>();
    uint64_t maskCnt = 0;
    for (uint32_t expertIndex = 0; expertIndex < sendNum_; expertIndex++) {
        uint32_t moeExpertId = startId_ + expertIndex * moeUsedAivNum_;
        CompareScalar(expertMaskTensorU8[maskSizePerExpert_ * expertIndex], expertIdsTensor_, moeExpertId, CMPMODE::EQ,
                      compareCount);
        PipeBarrier<PIPE_V>();
        GatherMask(gatherTempTensor, expertIdsTensor_,
                   expertMaskTensorU32[maskSizePerExpert_ * expertIndex / sizeof(uint32_t)], true, mask, {1, 1, 0, 0},
                   maskCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        tokenNumToExpertTensor_.SetValue(expertIndex, static_cast<uint32_t>(maskCnt));
    }
}

// 按专家分核发送（bitmask + ScalarGetSFFValue 精准遍历，免去逐 token 的前缀计数扫描）
template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::SendToMoeExpertByExpert()
{
    // 仅分 moe 专家给 moe 发送核（前段核），后段核走 SendToSharedExpert 不会到达此处
    SplitExpertNumToCore();
    if (sendNum_ == 0) {
        return;
    }
    CalExpertSendNum();

    // expertIdsTensor_ 为"原位选择 + (-1) 哨兵"全量数组：bitmask 仅在真实匹配处置位（哨兵永不命中），
    // 全长度遍历安全，且索引即原始 BS*K 空间索引
    uint32_t maskN64Num = Ceil(expertIdsCnt_, SFFVALUE_BITS);
    GlobalTensor<XOutType> dstWinGMTensor;
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};
    LocalTensor<uint64_t> expertMaskTensorU64 = expertMaskBuf_.Get<uint64_t>();

    for (uint32_t index = 0; index < sendNum_; index++) {
        // 旋转起始：叠加 (epRankId_ + aivId_) 作为偏移，使不同源 rank 与同 rank 不同核均错峰遍历自有专家，
        // 降低对同一目标 rank 窗口的写冲突。偏移仅改变遍历顺序（仍为 [0,sendNum_)
        // 的排列，每专家访问一次），不影响正确性。
        uint32_t expertIndex = (index + epRankId_ + aivId_) % sendNum_;
        if (tokenNumToExpertTensor_(expertIndex) == 0) {
            continue;
        }
        uint32_t moeExpertId = startId_ + expertIndex * moeUsedAivNum_;
        uint32_t maskExpertU64Cnt = maskSizePerExpert_ * expertIndex / sizeof(uint64_t);
        int32_t dstTokenPreCnt = 0;

        for (uint32_t maskIndex = 0; maskIndex < maskN64Num; maskIndex++) {
            uint64_t dstExpInfoMask = expertMaskTensorU64(maskIndex + maskExpertU64Cnt);
            int64_t curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            while (curValidIdx >= 0) {
                int32_t calExpertIdsIdx = static_cast<int32_t>(curValidIdx) + SFFVALUE_BITS * maskIndex;
                if (calExpertIdsIdx >= static_cast<int32_t>(expertIdsCnt_)) {
                    break;
                }
                int32_t topKIndex = calExpertIdsIdx % axisK_;
                int32_t srcTokenIndex = calExpertIdsIdx / axisK_;
                int32_t toRankId = static_cast<int32_t>(moeExpertId / moeExpertNumPerRank_) + sharedExpertRankNum_;
                if (isScalingDownFlag_) {
                    toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
                }
                GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(toRankId) +
                                                    (expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ +
                                                                            moeExpertId % moeExpertNumPerRank_)) +
                                                    hAlignWinSize_ * dstTokenPreCnt);
                dstWinGMTensor.SetGlobalBuffer((__gm__ XOutType *)rankGM);
                // 此处 topKIndex 恒 < axisK_，专家scale 写入条件等价于旧 FillTriple 内 (k < axisK_) &&
                // hasExpertScalesFlag_
                ProcessToken(dstWinGMTensor, srcTokenIndex, topKIndex, padParams, scaleInParams,
                             moeExpertId + sharedExpertNum_, hasExpertScalesFlag_);
                dstTokenPreCnt++;
                dstExpInfoMask &= ~(uint64_t(1) << curValidIdx);
                curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            }
        }
    }
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::TokenActiveMaskCal()
{
    // 搬运x_active_mask, 当前仅用于计算有效token总数
    LocalTensor<half> maskTmpTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    maskInputTensor = dstExpBuf_.Get<bool>();
    maskTmpTensor = subExpBuf_.Get<half>();
    sumOutTensor = gatherMaskTBuf_.Get<half>();
    DataCopyExtParams maskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(maskTmpTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, maskTmpTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::CalValidExpIdx(
    LocalTensor<bool> maskInputTensor)
{
    uint32_t mask = expertIdsCnt_;
    uint32_t curMaskCnt = axisBS_ * axisK_;
    uint32_t calCnt = Ceil(curMaskCnt * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    LocalTensor<half> tempTensor = subExpBuf_.Get<half>();
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> expertsIndexTensor = expertIdsBuf_.Get<int32_t>();

    Duplicate<half>(tempTensor, (half)0, calCnt);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, curMaskCnt);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0,
                        Ceil(expertIdsCnt_, ALIGNED_LEN_256) * ALIGNED_LEN_256 / BITS_PER_BYTE / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, curMaskCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor_, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::CalValidBSCnt(
    LocalTensor<bool> maskStrideTensor)
{
    uint64_t rsvdCnt = 0;
    uint32_t mask = axisBS_;
    uint32_t activeMaskAlignSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    LocalTensor<half> tempTensor = validExpertIndexBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = expertIdsBuf_.Get<half>();
    LocalTensor<half> tokenTargetTensor = validBsIndexTBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = subExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = gatherMaskTBuf_.Get<uint32_t>();
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    SumParams axisBsSumParams{
        1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize);
    PipeBarrier<PIPE_V>();
    Sum(tokenTargetTensor, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(validBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::ExpertActiveMaskCal()
{
    // 计算当前有效bs数量, stride搬入xActiveMask进行sum计算, 用于moe专家发送
    LocalTensor<bool> maskStrideTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskStrideCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskStrideParams{static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)), 0U,
                                       0U, 0U};
    DataCopyPad(maskStrideTensor, xActiveMaskGMTensor_, maskStrideParams, maskStrideCopyPadParams);
    CalValidBSCnt(maskStrideTensor);
    // 计算validExpIndexTensor, 连续搬入xActiveMask进行GatherMask计算, 用于moe专家的发送
    LocalTensor<bool> maskInputTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    CalValidExpIdx(maskInputTensor);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::MaskZeroComputeExpert(
    uint32_t maskCnt)
{
    LocalTensor<int32_t> expertsIndexTensor = dstExpBuf_.Get<int32_t>();
    int32_t maskTensorInt16Cnt = Ceil(expertIdsCnt_, UB_ALIGN / 2);
    LocalTensor<uint8_t> maskTensorInt8 = validExpertIndexBuf_.Get<uint8_t>();    // bs*k*1
    LocalTensor<uint32_t> maskTensorInt32 = validExpertIndexBuf_.Get<uint32_t>(); // bs*k*1
    LocalTensor<half> expertIdsTensorCast = subExpBuf_.Get<half>();               // bs*k*4
    int32_t moeExpertNumInt32 = static_cast<int32_t>(moeExpertNum_);

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    // half类型最大可精准表示的整数是2048，当前dispatch最大可支持专家数为1024，大于该值的情况均为特殊专家，dispatch均不做发送即可，combine再分专家类型进行对应处理；
    Cast(expertIdsTensorCast, expertIdsTensor_, RoundMode::CAST_NONE, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(maskTensorInt32, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    uint32_t calcCnt = Ceil(expertIdsCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    // 逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。筛掉零计算量专家
    CompareScalar(maskTensorInt8, expertIdsTensorCast, static_cast<half>(moeExpertNumInt32), AscendC::CMPMODE::LT,
                  calcCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint16_t> maskTensorInt16 = validExpertIndexBuf_.Get<uint16_t>();  // 空间bs*k*1
    LocalTensor<uint16_t> gatherMaskTensorint16 = gatherMaskTBuf_.Get<uint16_t>(); // 空间bs*k*4
    /* 特殊专家的maskTensorInt16和之前的gatherMaskTensor_结果按位相与，AND 支持uint16，
     * gatherMaskTensor_和gatherMaskTensorint16是同一个地址 */
    And(gatherMaskTensorint16, gatherMaskTensorint16, maskTensorInt16, maskTensorInt16Cnt);
    PipeBarrier<PIPE_V>();
    // 再筛一次
    CreateVecIndex(expertsIndexTensor, 0, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor_, expertsIndexTensor, gatherMaskTensor_, true, maskCnt, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::GenerateGatherMaskTensor(
    uint32_t maskCnt)
{
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0xFFFFFFFF, Ceil(maskCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::ZeroComputeExpertMaskCal()
{
    uint32_t maskCnt = expertIdsCnt_;
    if (isTokenMaskFlag_) { // 一维
        maskCnt = activeMaskBsCnt_ * axisK_;
    }

    if (!isExpertMaskFlag_) { // 非二维要生成gatherMaskTensor_
        GenerateGatherMaskTensor(maskCnt);
    }

    // 零计算量专家剪枝
    MaskZeroComputeExpert(maskCnt);
}

// expertIds 加载与 mask 剪枝（对标 a5_full_mesh ExpIdsCopyAndMaskCal）：
// 原位选择 + (-1) 哨兵——expertIdsTensor_ 保持全量 BS*K 长度不压缩，无效/被剪枝/对齐填充位置一律填 -1
// （Select 假分支写 -1.0f 位模式 / DataCopyPad 右填充 -1），expert id 恒 >= 0 哨兵永不命中，
// 后续 CompareScalar/bitmask 全长度遍历安全，索引即原始 BS*K 空间索引（无需 validExpertIndex 映射）。
// 二维 expert-mask（或 zeroCompute 剪枝，二者走同一 mask 通道 gatherMaskTensor_）按位原位剔除；
// 无二维 mask 时 1D token-mask 为前缀语义，仅搬 [0, activeMaskBsCnt_*K) 前缀。
// mask 无脏位保障：CalValidExpIdx 先零初始化 gatherMaskTensor_ 再 CompareScalar（[0, calCnt) 位全部
// 显式写 0/1），TokenActiveMaskCal 复用 gatherMaskTBuf_ 的残留不会进入 Select；Select 块粒度取整
// 读到 mask=0 的尾部位置写 -1，哨兵语义保持。
template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::ExpIdsCopyAndMaskCal()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }
    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }
    if (activeMaskBsCnt_ == 0) { // DataCopyPad 不能拷贝 0 个
        return;
    }
    if (zeroComputeExpertNum_ != 0) {
        ZeroComputeExpertMaskCal();
    }

    uint32_t expertIdsCntU = static_cast<uint32_t>(expertIdsCnt_);
    uint32_t expertIdsBufCnt =
        Ceil(expertIdsCntU * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    Duplicate<int32_t>(expertIdsTensor_, -1, expertIdsBufCnt); // 全量 -1 预填（容量与 Init 按 256B 对齐一致）
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) { // 按 gatherMaskTensor_ 原位保留/剔除
        LocalTensor<int32_t> tmpExpertIdsTensor = subExpBuf_.Get<int32_t>();
        LocalTensor<float> tmpExpertIdsTensorFloat = subExpBuf_.Get<float>();
        LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTensor_.ReinterpretCast<uint8_t>();
        DataCopyPad(tmpExpertIdsTensor, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_V>();
        LocalTensor<float> expertIdsTensorFloat = expertIdsTensor_.ReinterpretCast<float>();
        Select(expertIdsTensorFloat, gatherMaskTensorInt8, tmpExpertIdsTensorFloat, static_cast<float>(-1),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, expertIdsCnt_);
        SyncFunc<AscendC::HardEvent::V_S>();
    } else {
        // 无二维 mask：右侧 -1 填充至 32B 对齐
        uint32_t expertIdsMask = static_cast<uint32_t>(activeMaskBsCnt_) * axisK_;
        uint32_t expertIdsAlignCnt = Ceil(expertIdsMask, BITS_PER_BYTE) * BITS_PER_BYTE;
        uint32_t rightPadding = expertIdsAlignCnt - expertIdsMask; // 不超过 7 个 int32（32B 内）
        DataCopyPadExtParams<int32_t> expertIdsPadParams{true, 0U, static_cast<uint8_t>(rightPadding), -1};
        DataCopyExtParams expertIdsPrefixParams = {1U, expertIdsMask * sizeof(int32_t), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::V_MTE2>(); // 先等 Duplicate(-1) 的 V 写完成，再让 MTE2 覆写前缀
        DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsPrefixParams, expertIdsPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        SyncFunc<AscendC::HardEvent::MTE2_V>(); // CalExpertSendNum 的 CompareScalar/GatherMask 为 V 读，需 MTE2→V 同步
    }
    if (hasExpertScalesFlag_) {
        expertScalesTensor_ = expertScalesBuf_.Get<float>();
        DataCopyExtParams expertScalesParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> expertScalesPadParams{false, 0U, 0U, 0U};
        DataCopyPad(expertScalesTensor_, expertScalesGMTensor_, expertScalesParams, expertScalesPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
}

/*
共享专家卡：所有核用于给moe专家发送数据
moe专家卡：部分核用于给共享专家发送数据，部分核用于给moe专家发送数据
*/
template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::AlltoAllDispatch()
{
    ExpIdsCopyAndMaskCal(); // expertIds 原位(-1)加载：所有核（含共享发送核）均加载，
                            // SetStatusByExpert rear 分支与 SendToSharedExpert 的 validBsIndexTensor_ 依赖在此就绪
    if (activeMaskBsCnt_ == 0) {
        return;
    }
    // 后面的核向共享专家发数据
    bool isSendShared = (aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0);
    if (isSendShared) {
        SendToSharedExpert();
        return;
    }
    SendToMoeExpertByExpert();
}

// 按专家分核的状态写入（对标参考 atomic_expert.h SetStatus），与 token 发送同分核、同核 MTE3 保序：
// - front 核（aivId_ < moeUsedAivNum_）写 MOE 专家 flag：专家集合与 SendToMoeExpertByExpert 的步进分核
//   完全一致，计数直接读 CalExpertSendNum 写入的 tokenNumToExpertTensor_（同核已算好）。token 与 flag
//   由同一核同一条 MTE3 队列先后发出，PipeBarrier 排空流水后写 flag，MTE3 按序执行保证"flag 到 ⇒ 数据到"，
//   纯 MOE 场景免 SyncAll；elastic 场景 dstRankId 经 elasticInfoTensor_ 逐专家重映射（不改变分核）。
// - rear 核写共享专家 flag（连续分核，与 SendToSharedExpert 同 rear 池）：共享专家 token 按 token 维度
//   切分跨多个 rear 核，写 flag 核无法确信其余核已发完，须 PipeBarrier 后 SyncAll 再写 flag。
// - activeMaskBsCnt_==0（AlltoAllDispatch 早退、token 未发）时计数兜底 0、flag 照发，
//   保证接收侧 WaitStatus 的 flag 求和收敛（与基线 SetStatus 行为一致）。
// statusTensor_（statusBuf_）为核私有 UB，front 用 [0, sendNum_)、rear 用 [0, sharedSendNum_)，互不冲突。
template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::SetStatusByExpert()
{
    // 按核池分工：front 核（aivId_ < moeUsedAivNum_）写 MOE 专家 status；rear 核写共享专家 status。
    bool isFrontCore = (aivId_ < moeUsedAivNum_);
    uint32_t sharedStartId = 0;
    uint32_t sharedEndId = 0;
    uint32_t sharedSendNum = 0;
    if (!isFrontCore) {
        // rear 核在 sharedUsedAivNum_ 池内连续分核（isFront=false ⇒ newAivId = aivId_ - moeUsedAivNum_）
        SplitToCore(sharedExpertRankNum_, sharedUsedAivNum_, sharedStartId, sharedEndId, sharedSendNum, false);
    }

    // Phase 1：写计数（statusTensor_ 每块 8 个 int32，首元素为 flag 位，第二个为发送 token 数）
    if (isFrontCore) {
        // MOE 专家：步进映射（幂等重设，覆盖 activeMaskBsCnt_==0 早退时 sendNum_ 未初始化的场景）
        SplitExpertNumToCore();
        for (uint32_t expertIndex = 0; expertIndex < sendNum_; expertIndex++) {
            int32_t curExpertCnt = (activeMaskBsCnt_ > 0) ? tokenNumToExpertTensor_.GetValue(expertIndex) : 0;
            statusTensor_(expertIndex * 8 + 1) = curExpertCnt;
        }
    } else {
        // 共享专家：计数为 activeMaskBsCnt_（仅本卡所属共享组写非零）
        for (uint32_t i = 0; i < sharedSendNum; i++) {
            uint32_t curExpertId = sharedStartId + i;
            int32_t curExpertCnt = 0;
            if ((activeMaskBsCnt_ > 0) &&
                (curExpertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_)) {
                curExpertCnt = activeMaskBsCnt_;
            }
            statusTensor_(i * 8 + 1) = curExpertCnt;
        }
    }

    // Phase 1.5：排空本核流水（MTE3 token 写全部落盘），并建立 Phase1 标量写的 S→MTE3 依赖
    PipeBarrier<PIPE_ALL>();
    if (sharedExpertRankNum_ != 0) {
        // 存在共享专家：SendToSharedExpert 按 token 切分，一个共享专家的 token 跨多个 rear 核，
        // 写 flag 核无法确信其余核已发完，须 SyncAll 等所有核 token 落盘后再写 flag（与基线一致）
        SyncAll<true>();
    }

    // Phase 2：逐专家 DataCopy status 到远端（按 32B 对齐拷贝，8 是 32 字节包含的元素个数）
    GlobalTensor<int32_t> rankGMTensor;
    if (isFrontCore) {
        // MOE 专家
        for (uint32_t expertIndex = 0; expertIndex < sendNum_; expertIndex++) {
            uint32_t dstExpertId = startId_ + expertIndex * moeUsedAivNum_;
            uint32_t dstRankId = (dstExpertId / moeExpertNumPerRank_) + sharedExpertRankNum_;
            uint32_t offset =
                stateOffset_ * epRankId_ + (dstExpertId % moeExpertNumPerRank_) * epWorldSize_ * stateOffset_;
            if (isScalingDownFlag_) {
                dstRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId);
            }
            GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(dstRankId) + offset);
            rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
            DataCopy<int32_t>(rankGMTensor, statusTensor_[expertIndex * 8], 8UL);
        }
    } else {
        // 共享专家
        for (uint32_t i = 0; i < sharedSendNum; i++) {
            uint32_t curExpertId = sharedStartId + i;
            uint32_t dstRankId = curExpertId;
            uint32_t offset = stateOffset_ * epRankId_;
            if (isScalingDownFlag_) {
                dstRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId);
            }
            GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(dstRankId) + offset);
            rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
            DataCopy<int32_t>(rankGMTensor, statusTensor_[i * 8], 8UL);
        }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::InitPerformanceBuffers()
{
    if (isPerformanceFlag_) {
        uint32_t performanceTmpSize = JUMP_WRITE * epWorldSizeOriginal_ * sizeof(int32_t);
        uint32_t performanceTmpAlign = Ceil(performanceTmpSize, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(performanceTmpBuf_, performanceTmpAlign);
        totalUsedUB_ += performanceTmpAlign;
        performanceTmpTensor_ = performanceTmpBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceTmpTensor_, 0, performanceTmpAlign / sizeof(int32_t));
        uint32_t firstRecordSize = recStatusNumPerCore_ * sizeof(int32_t);
        uint32_t firstRecordSizeAlign = Ceil(firstRecordSize, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(firstRecordBuf_, firstRecordSizeAlign);
        totalUsedUB_ += firstRecordSizeAlign;
        firstRecordTensor_ = firstRecordBuf_.Get<int32_t>();
        Duplicate<int32_t>(firstRecordTensor_, 0, firstRecordSizeAlign / sizeof(int32_t));
    }
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::BufferInit()
{
    tpipe_->Reset();
    totalUsedUB_ = 0U;
    uint32_t waitStatusBufSize = (((recStatusNumPerCore_ * UB_ALIGN) > 256) ? (recStatusNumPerCore_ * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize); // 1024/24 * 32B = 43 * 32B
    totalUsedUB_ += waitStatusBufSize;
    // 原子定序后 SyncCntOnCore/GetCumSum 已删除，sumCoreBuf_/sumLocalBuf_/sumContinueBuf_/gatherMaskOutBuf_ 不再申请
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3); // 96B
    totalUsedUB_ += UB_ALIGN * 3;
    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hOutAlignUbSize_); // 7k*2 + 32 + 12
    totalUsedUB_ += BUFFER_NUM * hOutAlignUbSize_;
    InitPerformanceBuffers();
    tpipe_->InitBuffer(tokenNumBuf_, Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN);
    totalUsedUB_ += Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;

    // statusBuf_ 在接收路径仅用于 ResetNextRoundCounters 的 counterRegionSize_ 清零
    tpipe_->InitBuffer(statusBuf_, counterRegionSize_);
    totalUsedUB_ += counterRegionSize_;

    // tripleQueue_ 占剩余 UB（不计入 totalUsedUB_），LocalWindowCopy 与 SetExpertTokenNums 分阶段复用
    uint32_t leftUBSize = ((totalUbSize_ - totalUsedUB_) / UB_ALIGN) * UB_ALIGN;
    tripleDataSize_ = (leftUBSize > UB_ALIGN) ? ((leftUBSize - UB_ALIGN) / BUFFER_NUM) : 0U;
    tpipe_->InitBuffer(tripleQueue_, BUFFER_NUM, tripleDataSize_);
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::WaitDispatchClearStatus()
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::PerformanceInfoPerRank(
    uint64_t performanceTimeStart, bool isCompactStatus)
{
    uint64_t performanceTimecheck = static_cast<uint64_t>(GetSystemCycle());
    int32_t performanceTimeWait = static_cast<int32_t>((performanceTimecheck - performanceTimeStart) / CYCLES_PER_US);
    SyncFunc<AscendC::HardEvent::V_S>();
    for (uint32_t i = 0; i < recStatusNumPerCore_; i++) {
        uint32_t flagIndex = isCompactStatus ? i : i * FLAG_OFFSET;
        if (statusFp32Tensor_.GetValue(flagIndex) > float(0.5) && firstRecordTensor_.GetValue(i) == 0) {
            uint32_t fromRankId;
            uint32_t index = startExpertId_ + i;
            if (isScalingDownFlag_) {
                fromRankId =
                    elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + index % epWorldSize_);
            } else {
                fromRankId = index % epWorldSize_;
            }
            if (performanceTimeWait > performanceTmpTensor_.GetValue(JUMP_WRITE * fromRankId)) {
                performanceTmpTensor_.SetValue(JUMP_WRITE * fromRankId, performanceTimeWait);
                firstRecordTensor_.SetValue(i, 1);
            }
        }
    }
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::PerformanceInfoCopyOut()
{
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SetAtomicMax<int32_t>();
    DataCopyExtParams performanceInfoCopyParams{
        1U, JUMP_WRITE * epWorldSizeOriginal_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(performanceInfoGMTensor_, performanceTmpTensor_, performanceInfoCopyParams);
    SetAtomicNone();
}

// A5 用 Compact 搬运只轮询 flag 字段
template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::WaitStatusA5(
    LocalTensor<float> &statusSumOutTensor)
{
    float compareTarget = sumTarget_ * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyExtParams flagCopyParams{static_cast<uint16_t>(recStatusNumPerCore_), sizeof(float),
                                     STATE_OFFSET - sizeof(float), 0U, 0U};
    DataCopyPadExtParams<float> flagCopyPadParams{false, 0U, 0U, 0.0f};
    uint32_t flagSumInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams flagSumParams{1, flagSumInner, recStatusNumPerCore_};
    uint64_t performanceTimeStart = static_cast<uint64_t>(GetSystemCycle());
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopyPad<float, PaddingMode::Compact>(statusFp32Tensor_,
                                                 windowInstatusFp32Tensor_[startStatusIndex_ * FLAG_OFFSET],
                                                 flagCopyParams, flagCopyPadParams);
        if (isPerformanceFlag_) {
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            PerformanceInfoPerRank(performanceTimeStart, true);
        }
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, flagSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    if (isPerformanceFlag_) {
        PerformanceInfoCopyOut();
    }
    // 清状态仍按完整32B block回写，需要先补搬完整status以保留count字段。
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * FLAG_OFFSET], intriParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::WaitDispatch()
{
    BufferInit();
    startExpertId_ = startStatusIndex_; // 后面LocalWinCopy分核与此处保持一致
    endExpertId_ = startExpertId_ + recStatusNumPerCore_;
    sendExpertNum_ = recStatusNumPerCore_;
    if (unlikely(startStatusIndex_ >= rscvStatusNum_)) {
        return;
    }
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    if (isScalingDownFlag_) {
        elasticInst_.InitElasticInfoTensor(epWorldSizeOriginal_, elasticInfoTensor_);
    }
    WaitStatusA5(statusSumOutTensor);
    // 清状态
    WaitDispatchClearStatus();
    // 原子定序：不再需要 SyncCntOnCore + SyncAll 的跨核前缀和同步，
    // 核间定序由 LocalWindowCopy 中的 AtomicAdd 完成，Process 在 LocalWindowCopy 后统一 SyncAll。
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::RunPosRecord()
{
    return;
    TBuf<> runPosBuf;
    tpipe_->InitBuffer(runPosBuf, UB_ALIGN);
    DataCopyParams dataStateParams{1U, sizeof(uint32_t), 0U, 0U};
    dataStateLocalTensor_ = runPosBuf.Get<uint32_t>();
    dataStateLocalTensor_.SetValue(0, FLAG_AFTER_WAIT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams);
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::LocalWindowCopy()
{
    RunPosRecord(); // 维测打点
    if (startExpertId_ >= rscvStatusNum_) {
        return;
    } // 分核已与前面的waitDispatch里保持一致return;

    statusTensor_ = waitStatusBuf_.Get<int32_t>();

    // tripleQueue_ 已在 BufferInit 按剩余 UB 申请；此处取每块可容纳的 token 数
    uint32_t countPerDataSize = tripleDataSize_ / hOutAlignUbSize_;
    for (uint32_t index = startExpertId_; index < endExpertId_; index++) {
        uint32_t i = (index - startExpertId_), winOffset = index;
        uint32_t count = statusTensor_.GetValue(i * 8 + 1);
        if (count == 0) {
            continue; // 本窗口无 token，跳过原子累加与搬运
        }
        if ((!isShareExpertRankFlag_) &&
            (moeExpertNumPerRank_ > 1)) { // moe专家卡且一卡多专家场景 转换成数据区的排布偏移
            winOffset = index % epWorldSize_ * moeExpertNumPerRank_ + index / epWorldSize_;
        }

        // 原子定序：按目标专家原子累加 count，返回值 expertOffset 即本窗口块在该专家区内的起始槽位。
        // baseSlot = targetExpertId*globalBS_ + expertOffset：专家0(targetExpertId==0，含单专家)时退化为
        // expertOffset（== 最终输出槽，直写 expandXOut 头部）；专家>0 时为分段暂存槽（前面有未填满
        // globalBS_ 的空洞，需 SetExpertTokenNums 密排重排）。
        uint32_t targetExpertId = 0;
        if (!isShareExpertRankFlag_ && moeExpertNumPerRank_ > 1) {
            targetExpertId = index / epWorldSize_;
        }
        uint32_t expertOffset = AtomicAdd<uint32_t>(
            reinterpret_cast<__gm__ uint32_t *>(expertCountersGMTensor_.GetPhyAddr(targetExpertId * COUNTER_STRIDE)),
            count);
        bool directWrite = (targetExpertId == 0);
        uint32_t baseSlot = targetExpertId * globalBS_ + expertOffset;

        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + winOffset * expertPerSizeOnWin_;
        GlobalTensor<XOutType> tokGlobal;
        tokGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        uint32_t loopTimes = Ceil(count, countPerDataSize);
        uint32_t lastTimeCount = count - countPerDataSize * (loopTimes - 1);
        for (uint32_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
            uint32_t tokenCopiedNumThisExpert = countPerDataSize * loopIdx;
            tokGlobal.SetGlobalBuffer((__gm__ XOutType *)(wAddr + tokenCopiedNumThisExpert * hAlignWinSize_));

            uint16_t blockCount = static_cast<uint16_t>(((loopIdx < loopTimes - 1) ? countPerDataSize : lastTimeCount));
            uint16_t copyBlockLen = static_cast<uint16_t>(hOutAlignUbSize_ / UB_ALIGN);
            uint16_t srcStrideTokGlobal = static_cast<uint16_t>((hAlignWinSize_ - hOutAlignUbSize_) / UB_ALIGN);
            DataCopyParams dataCopyParamsTokGlobal{blockCount, copyBlockLen, srcStrideTokGlobal, 0U};
            LocalTensor<XOutType> tripleLocal = tripleQueue_.AllocTensor<XOutType>();
            DataCopy(tripleLocal, tokGlobal, dataCopyParamsTokGlobal);
            tripleQueue_.EnQue(tripleLocal);
            tripleLocal = tripleQueue_.DeQue<XOutType>();

            if (directWrite) {
                // 专家0 直写最终输出（恢复移植前基础 fan-out，baseSlot 取代原 beginIdx）
                GlobalTensor<XOutType> expandXOutGlobal;
                GlobalTensor<float> expertScaleInGlobal, expandScalesOutGlobal;
                LocalTensor<float> xTmpTensorFloat;
                expandXOutGlobal.SetGlobalBuffer(
                    (__gm__ XOutType *)(expandXOutGM_ + (uint64_t)(baseSlot + tokenCopiedNumThisExpert) * hOutSize_));
                uint32_t srcStrideExpandX = (hOutAlignUbSize_ - hOutSize_) / UB_ALIGN;
                DataCopyExtParams dataCopyParamsExpandX{blockCount, hOutSize_, srcStrideExpandX, 0U, 0U};
                DataCopyPad(expandXOutGlobal, tripleLocal, dataCopyParamsExpandX);

                uint32_t srcStrideScales = (hOutAlignUbSize_ - scaleOutBytes_) / UB_ALIGN;
                DataCopyExtParams dataCopyParamsScales{blockCount, scaleOutBytes_, srcStrideScales, 0U, 0U};
                quantInst_.CopyScalesToOut(baseSlot + tokenCopiedNumThisExpert, scaleOutBytes_, tripleLocal,
                                           dataCopyParamsScales);

                LocalTensor<int32_t> tripleLocalInt32 = tripleLocal.template ReinterpretCast<int32_t>();
                uint32_t expandIdxSize = EXPAND_IDX_INFO * static_cast<uint32_t>(sizeof(int32_t));
                uint32_t srcStrideExpandIdx = (hOutAlignUbSize_ - expandIdxSize) / UB_ALIGN;
                DataCopyExtParams dataCopyExpandIdx{blockCount, expandIdxSize, srcStrideExpandIdx, 0U, 0U};
                DataCopyPad(expandIdxGMTensor_[(baseSlot + tokenCopiedNumThisExpert) * EXPAND_IDX_INFO],
                            tripleLocalInt32[tokenQuantAlign_], dataCopyExpandIdx);

                if (hasExpertScalesFlag_) {
                    xTmpTensorFloat = tripleLocal.template ReinterpretCast<float>();
                    expertScaleInGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                        wAddr + tokenCopiedNumThisExpert * hAlignWinSize_ + expertScaleAlign_ * sizeof(float)));
                    expandScalesOutGlobal.SetGlobalBuffer((__gm__ float *)(expandScalesOutGM_) + baseSlot +
                                                          tokenCopiedNumThisExpert);
                    DataCopyExtParams expertScaleCopyInParams{blockCount, static_cast<uint32_t>(sizeof(float)),
                                                              static_cast<uint32_t>(hAlignWinSize_ - sizeof(float)), 0U,
                                                              0U};
                    DataCopyPadExtParams<float> expertScalePadParams{false, 0U, 0U, 0.0f};
                    DataCopyExtParams expertScaleCopyOutParams{blockCount, static_cast<uint32_t>(sizeof(float)), 0U, 0U,
                                                               0U};
                    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
                    DataCopyPad(xTmpTensorFloat, expertScaleInGlobal, expertScaleCopyInParams, expertScalePadParams);
                    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                    DataCopyPad(expandScalesOutGlobal, xTmpTensorFloat, expertScaleCopyOutParams);
                }
            } else {
                // 专家>0：完整 Token（data + 量化scales + expandIdx + 专家scale）整体暂存到 workspace，
                // 密排与分项输出交由 SetExpertTokenNums 重排阶段完成。
                GlobalTensor<uint8_t> stagingGM;
                stagingGM.SetGlobalBuffer((__gm__ uint8_t *)(recvCntWorkspaceGM_) +
                                          (uint64_t)(baseSlot + tokenCopiedNumThisExpert) * stagingTokenBytes_);
                DataCopyExtParams stagingParams{blockCount, stagingTokenBytes_, 0U, 0U, 0U};
                DataCopyPad(stagingGM, tripleLocal.template ReinterpretCast<uint8_t>(), stagingParams);
            }

            tripleQueue_.FreeTensor(tripleLocal);
        }
    }
}

// 原子定序：更新 expertTokenNumsOut + Token 密排重排。计数来源为每专家原子计数器，多核并行（按专家切分）。
template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::SetExpertTokenNums()
{
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1U : moeExpertNumPerRank_;
    // 多核并行：每核处理 [startExpertIdx, endExpertIdx)
    uint32_t expertNumPerCore = (localExpertNum + aivNum_ - 1) / aivNum_;
    uint32_t startExpertIdx = aivId_ * expertNumPerCore;
    uint32_t endExpertIdx = MIN(startExpertIdx + expertNumPerCore, localExpertNum);
    if (startExpertIdx >= localExpertNum) {
        return;
    }

    // 计数器由 AtomicAdd 写 GM（L2 关闭）；DataCacheCleanAndInvalid 确保 cache 一致后，
    // 直接从 expertCountersGMTensor_ 标量读取（不经 UB，避免 gather DataCopy 卡死）。
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertCountersGMTensor_);

    // 计算本核起始专家之前的 token 总数（前缀和）
    uint32_t runningSum = 0;
    for (uint32_t e = 0; e < startExpertIdx; e++) {
        runningSum += expertCountersGMTensor_.GetValue(e * COUNTER_STRIDE);
    }

    expertTokenNumsLocalTensor_ = tokenNumBuf_.Get<int64_t>();
    DataCopyExtParams expertTokenNumsCopyParams{1U, sizeof(int64_t), 0U, 0U, 0U};
    uint32_t countPerDataSize = tripleDataSize_ / hOutAlignUbSize_;

    GlobalTensor<XOutType> expandXOutGlobal;
    GlobalTensor<float> expandScalesOutGlobal;

    for (uint32_t myExpertIdx = startExpertIdx; myExpertIdx < endExpertIdx; myExpertIdx++) {
        uint32_t globalOffset = runningSum;
        uint32_t expertTokenCount = expertCountersGMTensor_.GetValue(myExpertIdx * COUNTER_STRIDE);
        runningSum += expertTokenCount;

        // (a) expertTokenNumsOut：保持现有语义（0=cumsum，1=count），各核按专家写各自表项
        int64_t expertTokenNumsVal = (expertTokenNumsType_ == 0) ?
                                         static_cast<int64_t>(globalOffset + expertTokenCount) :
                                         static_cast<int64_t>(expertTokenCount);
        expertTokenNumsLocalTensor_.SetValue(0, expertTokenNumsVal);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(expertTokenNumsOutGMTensor_[myExpertIdx], expertTokenNumsLocalTensor_, expertTokenNumsCopyParams);

        // (b) 最后一个专家：sendCountsGlobal_ 最后一元素记录本 rank Token 总数（combine 仅读该元素）
        if (myExpertIdx == localExpertNum - 1) {
            LocalTensor<int32_t> sendCntLocal = scalarBuf_.Get<int32_t>();
            sendCntLocal.SetValue(0, static_cast<int32_t>(runningSum));
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyExtParams sendCntCopyParams{1U, sizeof(int32_t), 0U, 0U, 0U};
            DataCopyPad(sendCountsGlobal_[rscvStatusNum_ - 1], sendCntLocal, sendCntCopyParams);
        }

        if (expertTokenCount == 0) {
            continue;
        }

        // 专家0 已在 LocalWindowCopy 直写就位（输出头部天然密排、无空泡），无需重排
        if (myExpertIdx == 0) {
            continue;
        }

        // (c) Token 密排重排：暂存区 → 最终输出（blockCount 批量搬运，复用 directWrite 模板）
        uint32_t srcPosition = myExpertIdx * globalBS_;
        uint32_t loopTimes = Ceil(expertTokenCount, countPerDataSize);
        uint32_t lastTimeCount = expertTokenCount - countPerDataSize * (loopTimes - 1);
        DataCopyPadExtParams<uint8_t> relayReadPadParams{false, 0U, 0U, 0U};
        for (uint32_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
            uint32_t tokenCopiedNum = countPerDataSize * loopIdx;
            uint16_t blockCount = static_cast<uint16_t>((loopIdx < loopTimes - 1) ? countPerDataSize : lastTimeCount);

            GlobalTensor<uint8_t> stagingGM;
            stagingGM.SetGlobalBuffer((__gm__ uint8_t *)(recvCntWorkspaceGM_) +
                                      (uint64_t)(srcPosition + tokenCopiedNum) * stagingTokenBytes_);
            DataCopyExtParams stagingReadParams{blockCount, stagingTokenBytes_, 0U, 0U, 0U};
            LocalTensor<XOutType> relayTokX = tripleQueue_.AllocTensor<XOutType>();
            LocalTensor<uint8_t> relayTokU8 = relayTokX.template ReinterpretCast<uint8_t>();
            DataCopyPad(relayTokU8, stagingGM, stagingReadParams, relayReadPadParams); // GM→UB（MTE2）
            tripleQueue_.EnQue(relayTokX);
            relayTokX = tripleQueue_.DeQue<XOutType>();
            LocalTensor<int32_t> relayTokI32 = relayTokX.template ReinterpretCast<int32_t>();
            LocalTensor<float> relayTokF = relayTokX.template ReinterpretCast<float>();

            uint32_t dstTokenIdx = globalOffset + tokenCopiedNum;
            // data → expandXOut
            expandXOutGlobal.SetGlobalBuffer((__gm__ XOutType *)(expandXOutGM_ + (uint64_t)dstTokenIdx * hOutSize_));
            DataCopyExtParams paramsExpandX{blockCount, hOutSize_, (hOutAlignUbSize_ - hOutSize_) / UB_ALIGN, 0U, 0U};
            DataCopyPad(expandXOutGlobal, relayTokX, paramsExpandX);
            // 量化 scales → dynamicScalesOut
            DataCopyExtParams paramsScales{blockCount, scaleOutBytes_, (hOutAlignUbSize_ - scaleOutBytes_) / UB_ALIGN,
                                           0U, 0U};
            quantInst_.CopyScalesToOut(dstTokenIdx, scaleOutBytes_, relayTokX, paramsScales);
            // expandIdx 三元组
            uint32_t expandIdxSize = EXPAND_IDX_INFO * static_cast<uint32_t>(sizeof(int32_t));
            DataCopyExtParams paramsExpandIdx{blockCount, expandIdxSize, (hOutAlignUbSize_ - expandIdxSize) / UB_ALIGN,
                                              0U, 0U};
            DataCopyPad(expandIdxGMTensor_[dstTokenIdx * EXPAND_IDX_INFO], relayTokI32[tokenQuantAlign_],
                        paramsExpandIdx);
            // 专家 scale → expandScalesOut
            if (hasExpertScalesFlag_) {
                expandScalesOutGlobal.SetGlobalBuffer((__gm__ float *)(expandScalesOutGM_) + dstTokenIdx);
                DataCopyExtParams paramsExpertScale{
                    blockCount, static_cast<uint32_t>(sizeof(float)),
                    (hOutAlignUbSize_ - static_cast<uint32_t>(sizeof(float))) / UB_ALIGN, 0U, 0U};
                DataCopyPad(expandScalesOutGlobal, relayTokF[expertScaleAlign_], paramsExpertScale);
            }
            tripleQueue_.FreeTensor(relayTokX);
        }
    }
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::ResetNextRoundCounters()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }
    LocalTensor<uint32_t> zeroLocal = statusBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(zeroLocal, 0, counterRegionSize_ / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    GlobalTensor<uint32_t> nextCountersGM;
    nextCountersGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    nextCountersGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>((__gm__ uint8_t *)(statusDataSpaceGm_) +
                                                                       ExpertCountersRegionOffset(1U - dataState_)));
    DataCopyExtParams zeroCopyParams{1U, counterRegionSize_, 0U, 0U, 0U};
    DataCopyPad(nextCountersGM, zeroLocal, zeroCopyParams);
}

template <TemplateDispatchV2AtomicTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Atomic<TemplateDispatchV2AtomicTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        AlltoAllDispatch();
        SetStatusByExpert();
        WaitDispatch();
        LocalWindowCopy();
        ResetNextRoundCounters(); // PINGPONG：末核把下一轮计数器区 reset 为 0
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>(); // 确保所有核完成 AtomicAdd 与暂存写入，全局可见后再重排
        SetExpertTokenNums();
    }
}
} // namespace Mc2Kernel
#endif // MOE_DISTRIBUTE_DISPATCH_V2_A5_ATOMIC_H
