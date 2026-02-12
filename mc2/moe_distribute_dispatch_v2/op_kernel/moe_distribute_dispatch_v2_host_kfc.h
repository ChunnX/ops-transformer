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
 * \file moe_distribute_dispatch_v2_host_kfc.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_HOST_KFC_H
#define MOE_DISTRIBUTE_DISPATCH_V2_HOST_KFC_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_v2_base.h"
#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#else
#include "../../common/inc/kernel/moe_distribute_base.h"
#endif
#include "check_winsize.h"

namespace Mc2Kernel {
constexpr uint32_t STATE_SIZE = 2048 * 1024; // 2M
constexpr uint64_t TIMEOUT_OFFSET = 1024UL * 1024UL;
constexpr uint8_t BUFFER_NUM = 2; // 多buf
constexpr uint8_t BUFFER_SINGLE = 1;
constexpr uint32_t STATE_OFFSET = 32U; // 状态空间偏移地址
constexpr uint8_t COMM_NUM = 2;        // 通信域大小
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint8_t COMM_TP_IDX = 1;
constexpr uint64_t WIN_STATE_OFFSET = 500UL * 1024UL;
constexpr uint64_t STATE_WIN_OFFSET = 950UL * 1024UL;
constexpr uint64_t TIMEOUT_DETECTION_THRESHOLD = 50000UL;
constexpr uint64_t CYCLES_PER_US = 50UL;
constexpr uint64_t TIMEOUT_DETECTION_TX_UNITS = 8UL;
constexpr uint32_t TP_STATE_SIZE = 100U * 1024U;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512U;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t RANK_LIST_NUM = 2U;
constexpr uint32_t EXPAND_IDX_INFO = 3U; // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t FLAG_AFTER_WAIT = 2U;
constexpr uint8_t EP_WORLD_SIZE_IDX = 1;
constexpr uint8_t SHARE_RANK_NUM_IDX = 2;
constexpr uint8_t MOE_NUM_IDX = 3;
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t MAX_UB_SIZE = 170U * 1024U;
constexpr uint32_t BW_ITEM_SIZE = 32; // batchWriteItemSize
constexpr uint32_t B64_PER_BLOCK = 4;
constexpr uint64_t SERVER_STATE_ALIGN = 512UL;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t COMPARE_COUNT_PER_BLOCK = 256U;

#define TemplateDispatchKFCTypeClass                                                                                           \
    typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist, bool IsNeedAllgather
#define TemplateDispatchKFCTypeFunc XType, ExpandXOutType, QuantMode, IsSmoothScaleExist, IsNeedAllgather

using namespace AscendC;
using namespace MoeDistributeV2Base;
template <TemplateDispatchKFCTypeClass>
class MoeDistributeDispatchV2HostKfc {
public:
    __aicore__ inline MoeDistributeDispatchV2HostKfc(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo,
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DispatchBetweenServer();
    __aicore__ inline void CommunicateBetweenServer(uint32_t beginServerId, uint32_t serverNum, uint32_t aivNum);
    __aicore__ inline void WaitWindow(uint32_t aivNumForWait);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void TimeOutDetection();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void CalValidBSCnt(LocalTensor<bool> maskStrideTensor);
    __aicore__ inline void CalValidExpIdx();
    __aicore__ inline void FillQuadruple(LocalTensor<ExpandXOutType> &xOutTensor, uint32_t tokenIndex);
    __aicore__ inline void CopyTokenToWinOut(LocalTensor<ExpandXOutType> &xoutTensor, uint32_t dstServerId,
                                             uint32_t cnt);
    __aicore__ inline void SingleTokenProcess(uint32_t tokenIndex, uint32_t dstServerId, uint32_t cnt);
    __aicore__ inline void DispatchAndCountTokens(uint32_t startIndex, uint32_t endIndex, bool process);
    __aicore__ inline void ConstructDataAndFlagBatchWriteInfo(uint32_t beginServerId, uint32_t serverNum,
                                                              uint32_t aivNum);
    __aicore__ inline void WaitStatusFlag(uint32_t serverIdx, uint32_t &tokenCnt);
    __aicore__ inline void WaitToken(uint32_t tokenCnt, uint32_t serverIdx, uint32_t startTokenIdx, TBuf<> &tBuf);
    __aicore__ inline void CheckDataArriveWithFlag(uint32_t beginIdx, uint32_t serverIdx, uint32_t &arriveCount);
    __aicore__ inline void CopyInAndOut(LocalTensor<float> xOutFp32Tensor, LocalTensor<int32_t> xOutInt32Tensor,
                                        GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount);
    __aicore__ inline void ExpertOffsetCal();
    __aicore__ inline void SendToServer();
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count);
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void SendToExpert();
    __aicore__ inline void resetMaxCnt(int32_t cntPosIndex, int32_t curExpertCnt);
    __aicore__ inline void SetStatus();
    __aicore__ inline void BufferInit();
    __aicore__ inline void InitElasticInfo(bool isWaitDispatch = false);
    __aicore__ inline void InitComputeInfo();
    __aicore__ inline void InitCommBetweenServerInfo();
    __aicore__ inline void InitDispatchBetweenServerInfo();
    __aicore__ inline void InitExtraInfo();
    __aicore__ inline void InitMaskInfo();
    __aicore__ inline void InitSetWindows(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void InitRecieveTilingContext(GM_ADDR expandXOut, GM_ADDR workspaceGM, TPipe *pipe,
                                                    const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void WaitDispatch();
    __aicore__ inline bool IsInSameServer(uint32_t targetRankId);
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t totalCount);
    __aicore__ inline void DoWindowCopy(LocalTensor<int32_t> &outLocal);
    __aicore__ inline void UpdateTokenNumsOut();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void FillTriple(LocalTensor<ExpandXOutType> &xOutTensor, uint32_t srcRankIndex,
                                      uint32_t tokenIndex, uint32_t k);
    __aicore__ inline void SyncCntOnCore(LocalTensor<float> &gatherMaskOutTensor,
                                         LocalTensor<uint32_t> &gatherTmpTensor,
                                         LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline GM_ADDR GetBaseWindOutAddrByServer(__gm__ HcclOpParam *addr, const int32_t serverId,
                                                         const int32_t curRankId);
    __aicore__ inline GM_ADDR GetBaseWindInAddrByServer(__gm__ HcclOpParam *addr, const int32_t serverId,
                                                        const int32_t curRankId);
    __aicore__ inline GM_ADDR GetBaseWindAddrByRankId(__gm__ HcclOpParam *addr, const int32_t serverId,
                                                      const int32_t curRankId);
    __aicore__ inline GM_ADDR GetBaseWindStateAddrByRankId(__gm__ HcclOpParam *addr, const int32_t serverId,
                                                           const int32_t curRankId);

    __aicore__ inline GM_ADDR GetSendAddrBetweenServer(uint8_t ctxIdx, const int32_t serverId)
    {
        uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankIdOriginal_ : tpRankId_);
        return GetBaseWindOutAddrByServer(winContext_[ctxIdx], serverId, curRankId);
    }

    __aicore__ inline GM_ADDR GetReceiveAddrBetweenServer(uint8_t ctxIdx, const int32_t serverId)
    {
        uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankIdOriginal_ : tpRankId_);
        return GetBaseWindInAddrByServer(winContext_[ctxIdx], serverId, curRankId);
    }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankIdOriginal_ : tpRankId_);
        uint64_t winDataSizeOffset = (ctxIdx == COMM_EP_IDX) ? winDataSizeOffsetEp_ : winDataSizeOffsetTp_;
        return GetBaseWindAddrByRankId(winContext_[ctxIdx], rankId, curRankId) + winDataSizeOffset;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankIdOriginal_ : tpRankId_);
        return GetBaseWindStateAddrByRankId(winContext_[ctxIdx], rankId, curRankId) + dataState_ * WIN_STATE_OFFSET;
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

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<ExpandXOutType> winTpGatherOutGMTensor_;
    GlobalTensor<int32_t> sendCountGMTensor_;
    GlobalTensor<float> fpWinTpGatherOutGMTensor_;
    GlobalTensor<int32_t> winTpEpCntGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;
    GlobalTensor<uint32_t> selfhcclDataStatusTensor_;
    GlobalTensor<uint64_t> dataBatchWriteInfoTensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;

    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<int32_t> tpTmpTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<ExpandXOutType> xOutTensor_;
    LocalTensor<float> xOutFp32Tensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> countTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> validExpertIndexTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    LocalTensor<uint32_t> serverCountTensor_;
    LocalTensor<uint32_t> tokenSendMap_;
    LocalTensor<uint32_t> flagGatherOutTensor_;
    LocalTensor<uint32_t> flagRecvTensor_;
    LocalTensor<uint8_t> flagCompResultU8_;
    LocalTensor<uint64_t> flagCompResultLtU64_;
    LocalTensor<uint32_t> flagRecvGatherMask_;
    LocalTensor<uint64_t> batchWriteU64Tensor_;
    LocalTensor<uint32_t> batchWriteU32Tensor_;
    LocalTensor<uint32_t> finishNumTensor_;
    LocalTensor<uint32_t> expertOffsetCntTensor_;
    LocalTensor<bool> expertMaskInputTensor_;

    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_; // gather mask输出buf
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_; // 辅助gather tensor定义
    TBuf<> rowMaxBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> expertMaskInputBuf_;
    TBuf<> subExpBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> maskBuf_;
    TBuf<> validExpertIndexBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> elasticInfoBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> serverCountBuf_;
    TBuf<> serverMapBuf_;
    TBuf<> batchWriteInfoBuf_;
    TBuf<> expertOffsetCntBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, 1> xInQueue_;                        // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;                      // 量化使用，量化后的输出

    GM_ADDR expandXOutGM_;
    GM_ADDR expandIdxOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR sendTpCountOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;
    GM_ADDR tpWindowGM_;
    GM_ADDR tpStatusWindowGM_;
    GM_ADDR tpLocalWindowGM_;
    GM_ADDR tpLocalStatusWindowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    GM_ADDR statusDataSpaceGm_;
    GM_ADDR dataBatchWriteInfo_;

    GM_ADDR gmTemp;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t bufferId_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    uint32_t tpWorldSize_{0};
    int32_t epRankId_{0};
    int32_t serverId_{0};         // 当前卡所在server编号
    int32_t serverRankSize_{0};   // 一个server内的rank数 epWorldSize_ -> serverRankSize_
    int32_t epRankIdInServer_{0}; // server内局部编号 epRankId_ -> epRankIdInServer_
    int32_t epRankIdOriginal_{0};
    uint32_t tpGatherRankId_{0}; // gather 对端ID
    uint32_t tpRankId_{0};       // 本卡 ID
    uint32_t aivId_{0};          // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};    // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0}; // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t globalBS_{0};
    uint32_t moeExpertRankNum_{0}; // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t startMoeExpertId_{0};     // 当前server的起始moe专家id 全局编号
    uint32_t startRankId_{0};          // 当前server的起始卡号 全局编号
    uint32_t shareRankNumInServer_{0}; // 当前server中共享专家卡数
    uint32_t expertNumInServer_{0};    // 当前server中共享专家副本数 + moe专家数 totalExpertNum_ -> expertNumInServer_
    uint32_t dealRankPerCore_{0};
    uint32_t hOutSize_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t moeListAlign_{0};
    uint32_t moeCntAlign_{0};
    uint32_t hScaleSizeAlign_{0};
    uint32_t shareListAlign_{0};
    uint32_t shareCntAlign_{0};
    uint32_t statusBufCntAlign_{0};
    uint32_t startExpertId_;
    uint32_t endExpertId_;
    uint32_t sendExpertNum_;
    uint32_t totalCnt_;
    uint32_t lastCore_{0};
    uint32_t dataState_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t totalUsedUB_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t winDataSizeOffsetTp_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t recvWinBlockNum_; // 接收Win区块数
    uint64_t sendToMoeExpTokenCnt_{0};
    uint64_t flagPadOffset_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    float sumTarget_;
    uint64_t totalWinSizeTp_{0};
    uint64_t totalWinSizeEp_{0};
    uint32_t gatherCount_{0};
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
    uint32_t blockCntPerToken_{0};
    uint32_t sendTokenLengthAlign_{0};
    uint32_t sendTokenLength_{0};
    uint32_t serverNum_{0};
    uint32_t hScaleIdxSize_{0};

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpParam *winContext_[COMM_NUM]{nullptr, nullptr};

    DataCopyExtParams floatDataCopyParams_;
    DataCopyExtParams expandXCopyParams_;
    DataCopyExtParams xCopyParams_;
    DataCopyExtParams hCommuCopyOutParams_;
};


template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitElasticInfo(bool isWaitDispatch)
{
    uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t);
    uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
    if (!isWaitDispatch) {
        totalUsedUB_ += elasticInfoSizeAlign;
    }
    elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
    DataCopyExtParams elasticInfoParams = {
        1U, static_cast<uint32_t>((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t)), 0U,
        0U, 0U};
    DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor_, elasticInfoParams, elasticInfoCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitComputeInfo()
{
    serverId_ = epRankId_ / serverRankSize_;
    epRankIdInServer_ = epRankId_ % serverRankSize_;
    startRankId_ = serverId_ * serverRankSize_;
    if (startRankId_ + serverRankSize_ <= sharedExpertRankNum_) { // 只有共享专家
        shareRankNumInServer_ = serverRankSize_;
        startMoeExpertId_ = 0;
    } else if (startRankId_ < sharedExpertRankNum_) { // 有共享专家也有moe专家
        shareRankNumInServer_ = sharedExpertRankNum_ - startRankId_;
        startMoeExpertId_ = 0;
    } else { // 只有moe专家
        shareRankNumInServer_ = 0;
        startMoeExpertId_ = (startRankId_ - sharedExpertRankNum_) * moeExpertNumPerRank_;
    }
    expertNumInServer_ = shareRankNumInServer_ + (serverRankSize_ - shareRankNumInServer_) * moeExpertNumPerRank_;

    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;

    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivNum_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    expertIdsCnt_ = axisBS_ * axisK_;
    recvWinBlockNum_ = serverRankSize_ * moeExpertNumPerRank_;
    moeUsedAivNum_ = aivNum_ - sharedUsedAivNum_;
    dealRankPerCore_ = (recvWinBlockNum_ + aivNum_ - 1) / aivNum_;
    stateOffset_ = STATE_OFFSET;
    PipeBarrier<PIPE_ALL>();
    if (isShareExpertRankFlag_) { // 当前卡是共享专家卡
        rscvStatusNum_ = serverRankSize_;
    } else { // 当前卡是moe专家卡
        rscvStatusNum_ = recvWinBlockNum_;
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitExtraInfo()
{ // 三元组、moelist信息
    hOutSize_ = axisH_ * sizeof(ExpandXOutType);
    hOutSizeAlign_ = Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN; // scale起始放置偏移
    hScaleSizeAlign_ = hOutSizeAlign_ + UB_ALIGN;          // 填充三元组起始偏移
    tokenQuantAlign_ = hScaleSizeAlign_ / sizeof(int32_t);
    // 实际搬运大小，搬运token_align32B + 32B(float) + 3*4B(三元组)

    uint32_t moeListSizeAlign = hScaleSizeAlign_ + UB_ALIGN; // moeExpertIdList起始偏移
    moeListAlign_ = moeListSizeAlign / sizeof(int32_t);

    uint32_t moeCntSizeAlign = moeListSizeAlign + Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // moeCntList偏移
    moeCntAlign_ = moeCntSizeAlign / sizeof(int32_t);

    sendTokenLength_ = moeCntSizeAlign + Ceil(axisK_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    blockCntPerToken_ = Ceil(sendTokenLength_, SPLIT_BLOCK_DATA_SIZE);
    sendTokenLengthAlign_ = blockCntPerToken_ * SERVER_STATE_ALIGN;

    hScaleIdxSize_ = hScaleSizeAlign_ + EXPAND_IDX_INFO * sizeof(int32_t);
    hAlignWinSize_ = Ceil(hScaleIdxSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // win区token起始地址对齐512
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hAlignWinSize_;
    hOutAlignUbSize_ = Ceil(hScaleIdxSize_, UB_ALIGN) * UB_ALIGN;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitMaskInfo()
{
    expertIdsCnt_ = axisBS_ * axisK_;
    uint32_t hFp32Size = axisH_ * sizeof(float);
    uint32_t expertIdsSize = expertIdsCnt_ * sizeof(int32_t);
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    uint32_t bsAlign256 = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t bsKAlign256 = Ceil(expertIdsCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t expertIdsBufSize = expertIdsSize > bsAlign256 ? expertIdsSize : bsAlign256;
    expertIdsSize = Ceil(expertIdsSize, UB_ALIGN) * UB_ALIGN;
    maxSize_ = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSize_ = maxSize_ > xActivateMaskSize ? maxSize_ : xActivateMaskSize;
    maxSize_ = maxSize_ > bsKAlign256 ? maxSize_ : bsKAlign256;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize); // BS * K * 4 = 32K
    totalUsedUB_ += expertIdsSize;
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    tpipe_->InitBuffer(gatherMaskTBuf_, maxSize_);
    totalUsedUB_ += maxSize_;
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
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
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitDispatchBetweenServerInfo()
{
    tpipe_->InitBuffer(dstExpBuf_, maxSize_); // BS * K * 4 = 32K
    tpipe_->InitBuffer(expertMaskInputBuf_, expertIdsCnt_ * sizeof(bool));
    totalUsedUB_ += maxSize_;
    tpipe_->InitBuffer(subExpBuf_, maxSize_); // BS * K * 4 = 32K
    totalUsedUB_ += maxSize_;
    uint32_t tmpTotalUB = totalUsedUB_ + hOutAlignUbSize_ * BUFFER_NUM;
    bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
    tpipe_->InitBuffer(xQueue_, bufferNum_, hOutAlignUbSize_); // 7k*2 + 32 + 12

    expertMaskInputTensor_ = expertMaskInputBuf_.Get<bool>();
    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();

    uint32_t axisHCommu = hScaleIdxSize_ / sizeof(ExpandXOutType); // 有效搬运长度
    floatDataCopyParams_ = {1U, sizeof(float), 0U, 0U, 0U};
    xCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U, 0U};
    hCommuCopyOutParams_ = {1U, static_cast<uint32_t>(axisHCommu * sizeof(ExpandXOutType)), 0U, 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(ExpandXOutType)), 0U, 0U, 0U};

    uint32_t serverBuferLength = serverNum_ * sizeof(uint32_t);
    uint32_t serverMapLength = serverNum_ * axisMaxBS_ * sizeof(uint32_t);
    tpipe_->InitBuffer(serverCountBuf_, serverBuferLength);
    tpipe_->InitBuffer(serverMapBuf_, serverMapLength);
    serverCountTensor_ = serverCountBuf_.Get<uint32_t>();
    tokenSendMap_ = serverMapBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(serverCountTensor_, uint32_t(0), serverBuferLength);
    Duplicate<uint32_t>(tokenSendMap_, uint32_t(0), serverMapLength);

    tpipe_->InitBuffer(expertOffsetCntBuf_, expertIdsCnt_ * sizeof(uint32_t));
    expertOffsetCntTensor_ = expertOffsetCntBuf_.Get<uint32_t>();
    tpipe_->InitBuffer(xInQueue_, bufferNum_, hOutSizeAlign_);    // 14K * 2
    tpipe_->InitBuffer(xOutQueue_, bufferNum_, sendTokenLength_); // 7K * 2 + 32 + 6
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitCommBetweenServerInfo()
{
    recStatusNumPerCore_ = rscvStatusNum_ / aivNum_; // 每个aiv需要处理的专家数
    remainderRankNum_ = rscvStatusNum_ % aivNum_;
    startStatusIndex_ = recStatusNumPerCore_ * aivId_; // + sharedExpertRankNum_, 每个aiv发送的
    if (aivId_ < remainderRankNum_) {                  // 前remainderRankNum个aiv需要多发1个卡的数据
        recStatusNumPerCore_ += 1;
        startStatusIndex_ += aivId_;
    } else {
        startStatusIndex_ += remainderRankNum_;
    }
    statusBufCntAlign_ = Ceil(Ceil(expertNumInServer_, aivNum_), 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    uint32_t statusBufSize = statusBufCntAlign_ * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    totalUsedUB_ += statusBufSize;
    statusTensor_ = statusBuf_.Get<int32_t>();                  // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, recvWinBlockNum_ * 8); // 8 = UB_ALIGN / sizeof(int32_t)
    statusSpaceGm_ = GetWindStateAddrByRankId(COMM_EP_IDX, epRankIdOriginal_);

    batchWriteU64Tensor_ = batchWriteInfoBuf_.Get<uint64_t>();
    batchWriteU32Tensor_ = batchWriteU64Tensor_.template ReinterpretCast<uint32_t>();

    dataBatchWriteInfo_ = recvCntWorkspaceGM_ + WORKSPACE_ELEMENT_OFFSET * aivNum_ * aivNum_;
    dataBatchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint64_t *)(dataBatchWriteInfo_), serverNum_ * B64_PER_BLOCK);
    tpipe_->InitBuffer(batchWriteInfoBuf_, BW_ITEM_SIZE);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitRecieveTilingContext(
    GM_ADDR expandXOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
    gmTemp = workspaceGM;

    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    GM_ADDR hcclContext = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_[COMM_EP_IDX] = (__gm__ HcclOpParam *)hcclContext;
    winContext_[COMM_TP_IDX] = (__gm__ HcclOpParam *)AscendC::GetHcclContext<1>(); // 没有相关公共宏

    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(workspaceGM));
    bufferId_ = bufferChosenGlobal_(0);

    // 检查hcclwinsize是否越界
    totalWinSizeEp_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    totalWinSizeTp_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeTp);
    CheckWindowSize(totalWinSizeEp_, GetWinSize(winContext_[COMM_EP_IDX]), tpipe_, expandXOut);

    serverNum_ = 1;
    serverRankSize_ = 8;
    epRankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    epRankIdOriginal_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    epWorldSizeOriginal_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    hasElasticInfoFlag_ = tilingData->moeDistributeDispatchV2Info.hasElasticInfo;
    epWorldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    globalBS_ = tilingData->moeDistributeDispatchV2Info.globalBs;
    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isExpertMask;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    zeroComputeExpertNum_ = tilingData->moeDistributeDispatchV2Info.zeroComputeExpertNum;
    tpRankId_ = tilingData->moeDistributeDispatchV2Info.tpRankId;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    tpWorldSize_ = tilingData->moeDistributeDispatchV2Info.tpWorldSize;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitSetWindows(const MoeDistributeDispatchV2TilingData *tilingData)
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempepRankId = 0; tempepRankId < epWorldSize_; tempepRankId++) {
        OOMCheckAddrRange<ExpandXOutType>((__gm__ ExpandXOutType *)(GetWindAddrByRankId(COMM_EP_IDX, tempepRankId)),
                                          totalWinSizeEp_);
        OOMCheckAddrRange<float>((__gm__ float *)(GetWindStateAddrByRankId(COMM_EP_IDX, tempepRankId)), STATE_SIZE);
    }
#endif
    sumTarget_ = static_cast<float>(1.0);
    uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign_ / 8, 1, 8); // 0x3F800000是float的1

    // 当前tpWin区划分为前后两半区，连续两次dispatch，切换半区, combine 数据区使用前面，
    // 即axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine, dispatch使用后面
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffsetEp_ =
        dataState_ * (totalWinSizeEp_ / 2) +
        axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine; // 就是分成两块，去掉combine前面的

    winDataSizeOffsetTp_ =
        dataState_ * (totalWinSizeTp_ / 2) + tilingData->moeDistributeDispatchV2Info.a * hSizeAlignCombine;

    windowGM_ = GetWindAddrByRankId(COMM_EP_IDX, epRankIdOriginal_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    GlobalTensor<ExpandXOutType> winDouble;
    winDouble.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    winDouble.SetGlobalBuffer((__gm__ ExpandXOutType *)(windowGM_));
    OOMCheckAddrRange<ExpandXOutType>((__gm__ ExpandXOutType *)(winDouble.GetPhyAddr()), totalWinSizeEp_);
#endif
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
    GM_ADDR tpSendCountsOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
    InitRecieveTilingContext(expandXOut, workspaceGM, pipe, tilingData);
    statusDataSpaceGm_ = GetStatusDataSpaceGm(winContext_[COMM_EP_IDX]);
    selfDataStatusGMTensor_.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusDataSpaceGm_ + STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, winContext_[COMM_EP_IDX], epRankIdOriginal_, moeExpertNum_,
                              epWorldSizeOriginal_, globalBS_, dataStateBuf);
    elasticInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(elasticInfo));
    if (hasElasticInfoFlag_) {
        InitElasticInfo(false);
    }
    if (epRankId_ < sharedExpertRankNum_) {
        isShareExpertRankFlag_ = true;
    }
    InitComputeInfo();

    tpGatherRankId_ = ((tpRankId_ == 0) ? 1 : 0);
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOut));
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut; // 无GlobalTensor
    sendTpCountOutGM_ = tpSendCountsOut;

    sendCountGMTensor_.SetGlobalBuffer((__gm__ int32_t *)workspaceGM);
    int32_t sendCntSize = epWorldSize_ * expertNumInServer_ * sizeof(int32_t);
    TBuf<> cleanSendBuf;
    tpipe_->InitBuffer(cleanSendBuf, sendCntSize);
    LocalTensor<int32_t> tempTensor;
    tempTensor = cleanSendBuf.Get<int32_t>();
    Duplicate<int32_t>(tempTensor, 0, expertNumInServer_);
    DataCopy(sendCountGMTensor_, tempTensor, expertNumInServer_); // 清零
    recvCntWorkspaceGM_ = workspaceGM + Ceil(sendCntSize * UB_ALIGN, UB_ALIGN) * UB_ALIGN;

    InitCommBetweenServerInfo();
    InitSetWindows(tilingData);
    InitExtraInfo();
    InitMaskInfo();
    InitDispatchBetweenServerInfo();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum,
                                                                 uint32_t &startTokenId, uint32_t &endTokenId,
                                                                 uint32_t &sendTokenNum, bool isFront)
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

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::TokenActiveMaskCal()
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


template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalValidExpIdx()
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
    LocalTensor<int8_t> maskInputInt8Tensor = expertMaskInputTensor_.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, curMaskCnt);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0,
                        Ceil(expertIdsCnt_, ALIGNED_LEN_256) * ALIGNED_LEN_256 / BITS_PER_BYTE / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, curMaskCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor_, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_); // 有效的专家的索引以及需要发送的有效专家总数
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalValidBSCnt(LocalTensor<bool> maskStrideTensor)
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

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ExpertActiveMaskCal()
{
    // 计算当前有效bs数量, stride搬入xActiveMask进行sum计算, 用于moe专家发送
    LocalTensor<bool> maskStrideTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskStrideCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskStrideParams{static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)), 0U,
                                       0U, 0U};
    DataCopyPad(maskStrideTensor, xActiveMaskGMTensor_, maskStrideParams, maskStrideCopyPadParams);
    CalValidBSCnt(maskStrideTensor);

    // 计算validExpIndexTensor, 连续搬入xActiveMask进行GatherMask计算, 用于moe专家的发送
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(expertMaskInputTensor_, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    CalValidExpIdx();
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::FillQuadruple(LocalTensor<ExpandXOutType> &xOutTensor,
                                                                   uint32_t tokenIndex)
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = epRankId_;      // 源rankId
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex; // TokenID ,

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(axisK_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xOutTint32[moeListAlign_], expertIdsGMTensor_[tokenIndex * axisK_], expertIdsCntParams,
                expertIdsCntCopyPadParams);

    for (uint32_t index = 0; index < axisK_; index++) {
        uint32_t cntPos = moeCntAlign_ + index;
        uint32_t expertPos = moeListAlign_ + index;
        uint32_t offset = tokenIndex * axisK_ + index;
        if (isExpertMaskFlag_ && !expertMaskInputTensor_(offset)) {
            xOutTint32(expertPos) = moeExpertNum_;
        }
        xOutTint32(cntPos) = expertOffsetCntTensor_(offset);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CopyTokenToWinOut(LocalTensor<ExpandXOutType> &xOutTensor,
                                                                       uint32_t dstServerId, uint32_t cnt)
{
    TBuf<> outBuf;
    tpipe_->InitBuffer(outBuf, sendTokenLengthAlign_);
    GlobalTensor<ExpandXOutType> dataDstWinGMTensor;
    GlobalTensor<uint32_t> flagDstWinGMTensor;
    dataDstWinGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType *)(GetSendAddrBetweenServer(COMM_EP_IDX, dstServerId) +
                                                                 SERVER_STATE_ALIGN + cnt * sendTokenLengthAlign_));
    flagDstWinGMTensor.SetGlobalBuffer((__gm__ uint32_t *)(GetSendAddrBetweenServer(COMM_EP_IDX, dstServerId) +
                                                           SERVER_STATE_ALIGN + cnt * sendTokenLengthAlign_));

    LocalTensor<uint32_t> flagTensor = outBuf.Get<uint32_t>();
    Duplicate<uint32_t>(flagTensor, uint32_t(1), {0x0101010101010101}, Ceil(blockCntPerToken_ * UB_ALIGN, 256),
                        uint16_t(1), uint8_t(8));

    DataCopyExtParams dataCopyOutParams = {static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_DATA_SIZE, 0U,
                                           UB_ALIGN, 0U};
    DataCopyPad(dataDstWinGMTensor, xOutTensor, dataCopyOutParams);

    DataCopyExtParams flagCopyOutParams = {static_cast<uint16_t>(blockCntPerToken_), UB_ALIGN, 0U,
                                           SPLIT_BLOCK_DATA_SIZE, 0U};
    DataCopyPad(flagDstWinGMTensor[SPLIT_BLOCK_DATA_SIZE / UB_ALIGN], flagTensor, flagCopyOutParams);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SingleTokenProcess(uint32_t tokenIndex,
                                                                                               uint32_t dstServerId,
                                                                                               uint32_t cnt)
{
    xInTensor_ = xInQueue_.AllocTensor<XType>();
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(xInTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
    xInQueue_.EnQue(xInTensor_);
    xInTensor_ = xInQueue_.DeQue<XType>();
    xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
    DataCopy(xOutTensor_, xInTensor_, hOutSizeAlign_);
    xInQueue_.FreeTensor<XType>(xInTensor_);
    FillQuadruple(xOutTensor_, tokenIndex);
    CopyTokenToWinOut(xOutTensor_, dstServerId, cnt);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::DispatchAndCountTokens(uint32_t startIndex,
                                                                                                   uint32_t endIndex,
                                                                                                   bool process)
{
    uint32_t rankIDSharedGroup = epRankId_ % rankNumPerSharedExpert_; // 计算目的共享专家卡在其所在共享专家组的id
    for (uint32_t index = startIndex; index < endIndex; index++) {
        uint32_t tokenIndex = isExpertMaskFlag_ ? validBsIndexTensor_(index) : index;
        uint32_t sumExpert =
            sharedExpertNum_ + axisK_; // 每个Token需要发送 sharedExpertNum_ 个共享专家 和axisK_ 个moe专家
        for (uint32_t expertIndex = 0; expertIndex < sumExpert; expertIndex++) {
            uint32_t expertId = expertIndex;
            uint32_t dstRankId = expertId * rankNumPerSharedExpert_ + rankIDSharedGroup;
            if (expertIndex >= sharedExpertNum_) { // moe专家重新计算目标
                uint32_t curIndex = expertIndex - sharedExpertNum_;
                expertId = expertIdsTensor_(tokenIndex * axisK_ + curIndex);
                dstRankId = expertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
                if (zeroComputeExpertNum_ != 0 && expertId >= moeExpertNum_) { // 处理零专家
                    continue;
                }
            }
            uint32_t dstServerId = dstRankId / serverRankSize_;
            uint32_t pos = dstServerId * activeMaskBsCnt_ + tokenIndex; // 计算当前token是否发给这个server
            if (tokenSendMap_.GetValue(pos) == 1) {                     // 必须清0，
                continue;
            }
            tokenSendMap_.SetValue(pos, 1);
            uint32_t CntValue = serverCountTensor_.GetValue(dstServerId);
            serverCountTensor_.SetValue(dstServerId, CntValue + 1); // 当前server已经收到多少Token
            if (process) {
                SingleTokenProcess(tokenIndex, dstServerId, CntValue); // 将数据写入对应的发送区
            }
        }
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendToServer()
{
    uint32_t totalSendCnt = activeMaskBsCnt_;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(totalSendCnt, aivNum_, startTokenId, endTokenId, sendTokenNum, true);
    if (startTokenId >= totalSendCnt || sendTokenNum == 0) {
        return;
    }

    DispatchAndCountTokens(0, startTokenId, false);
    DispatchAndCountTokens(startTokenId, endTokenId, true);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

    if (endTokenId == totalSendCnt) {
        GlobalTensor<uint32_t> dstStateGMTensor;
        uint32_t flagOffset = SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t); // 前面的状态区的最后32B的第一个元素是flag位为1
        for (uint32_t index = 0; index < serverNum_; index++) {
            dstStateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)GetSendAddrBetweenServer(COMM_EP_IDX, index));
            dstStateGMTensor(0) = serverCountTensor_(index);
            dstStateGMTensor(flagOffset) = 1;
        }
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ExpertOffsetCal()
{
    TBuf<> expertOffTempBuf;
    LocalTensor<int32_t> expertOffsetTemp;

    tpipe_->InitBuffer(expertOffTempBuf, moeExpertNum_ * sizeof(int32_t));
    expertOffsetTemp = expertOffTempBuf.Get<int32_t>();

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams); // copy expertid
    Duplicate<int32_t>(expertOffsetTemp, (int32_t)0, moeExpertNum_);

    for (uint32_t index = 0; index < expertIdsCnt_; index++) { // 计算当前token是发给此moe专家的第几个token
        if (isExpertMaskFlag_ && !expertMaskInputTensor_(index)) {
            continue;
        }
        uint32_t expertId = expertIdsTensor_.GetValue(index);
        uint32_t exprtOffset = expertOffsetTemp(expertId);
        expertOffsetCntTensor_(index) = exprtOffset;
        expertOffsetTemp(expertId) = exprtOffset + 1;
    }
}


template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::DispatchBetweenServer()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    if (isTokenMaskFlag_) { // 1维
        TokenActiveMaskCal();
    }

    if (isExpertMaskFlag_) { // 2维
        ExpertActiveMaskCal();
    }

    if (activeMaskBsCnt_ == 0) {
        return;
    }

    ExpertOffsetCal();
    SendToServer();
    SyncAll<true>();
}

// 构建发往其他server的所有data报文
template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ConstructDataAndFlagBatchWriteInfo(
    uint32_t beginServerId, uint32_t serverNum, uint32_t aivNum)
{
    // 计算当前core要处理的server
    uint32_t startServerId, endServerId, batchWriteItemNum;

    SplitToCore(serverNum, aivNum, startServerId, endServerId, batchWriteItemNum, true);

    if (startServerId >= serverNum || batchWriteItemNum == 0) {
        return;
    }
    startServerId += beginServerId;
    endServerId += beginServerId;
    // 当前aiv负责 [startServerId,endServerId) 个 server
    for (uint32_t dstServerInd = startServerId; dstServerInd < endServerId; ++dstServerInd) {
        uint32_t dstRankId = epRankId_ % serverRankSize_ + dstServerInd * serverRankSize_; // 目标Server

        uint64_t dstDataAddr = (uint64_t)(GetReceiveAddrBetweenServer(COMM_EP_IDX, dstRankId));
        // src卡GetWindowsInAddr地址, 要发给serverIndex
        uint64_t srcDataAddr = (uint64_t)(GetSendAddrBetweenServer(COMM_EP_IDX, serverId_));
        // 去往该Server的传输的数据量
        uint32_t validTokenCount = serverCountTensor_(dstServerInd);
        uint32_t validDataLength = SERVER_STATE_ALIGN + validTokenCount * sendTokenLengthAlign_;

        batchWriteU64Tensor_(0) = srcDataAddr;     // 源地址
        batchWriteU64Tensor_(1) = dstDataAddr;     // 目的地址
        batchWriteU64Tensor_(2) = validDataLength; // 数据长度
        batchWriteU32Tensor_(6) = HcclDataType::HCCL_DATA_TYPE_INT8;
        batchWriteU32Tensor_(7) = dstRankId; // dst卡

        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t dstServerOffset = dstServerInd;
        uint32_t sendInfoCount = B64_PER_BLOCK; // 只发送一个结构
        DataCopy(dataBatchWriteInfoTensor_[dstServerOffset * sendInfoCount], batchWriteU64Tensor_, sendInfoCount);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitWindow(uint32_t aivNum)
{
    // 确定当前核处理哪个Server
    // aivNum个核处理serverNum_个Server数据
    uint32_t startServerId, endServerId, serverCnt;

    SplitToCore(serverNum_, aivNum, startServerId, endServerId, serverCnt, true);
    if (startServerId >= serverNum_ || serverCnt == 0) {
        return;
    }
    for (uint32_t dstServerInd = startServerId; dstServerInd < endServerId; ++dstServerInd) {
        uint32_t tokenCnt;
        WaitStatusFlag(dstServerInd, tokenCnt);
        // buf分配
        tpipe_->Reset();
        uint32_t flagMaxRecvNum = (blockCntPerToken_ * UB_ALIGN) / sizeof(uint32_t);
        uint32_t gatherOutSize =
            Ceil(blockCntPerToken_ * sizeof(uint32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256; // 256对齐
        uint32_t tBufRealSize_ = MAX_UB_SIZE - (BUFFER_NUM * UB_ALIGN * 3);

        TBuf<> flagMaskBuf, statusCleanBuf, finishNumBuf, tBuf;
        tpipe_->InitBuffer(tBuf, tBufRealSize_); // 其余buffer空间统一申请
        tpipe_->InitBuffer(flagMaskBuf, BUFFER_NUM * UB_ALIGN);
        tpipe_->InitBuffer(statusCleanBuf, BUFFER_NUM * UB_ALIGN);
        tpipe_->InitBuffer(finishNumBuf, BUFFER_NUM * UB_ALIGN);

        flagGatherOutTensor_ = tBuf.GetWithOffset<uint32_t>(gatherOutSize / sizeof(uint32_t), 0); // buf复用
        flagRecvTensor_ = tBuf.GetWithOffset<uint32_t>(flagMaxRecvNum, gatherOutSize);            // buf复用

        flagCompResultU8_ = flagMaskBuf.Get<uint8_t>();
        flagCompResultLtU64_ = flagMaskBuf.Get<uint64_t>();
        flagRecvGatherMask_ = statusCleanBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
        finishNumTensor_ = finishNumBuf.Get<uint32_t>();

        WaitToken(tokenCnt, dstServerInd, 0, tBuf);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitStatusFlag(uint32_t serverIdx,
                                                                                           uint32_t &tokenCnt)
{
    tpipe_->Reset();
    TBuf<> tBuf;
    LocalTensor<uint32_t> statusFlagLocal, statusCntLocal;
    GlobalTensor<uint32_t> statusCntGlobal;
    GM_ADDR wAddr = GetReceiveAddrBetweenServer(COMM_EP_IDX, serverIdx); // 对应server接收区地址
    statusCntGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(wAddr));
    tpipe_->InitBuffer(tBuf, UB_ALIGN * BUFFER_NUM);
    statusFlagLocal = tBuf.Get<uint32_t>();
    statusCntLocal = tBuf.Get<uint32_t>();

    while (true) {
        // 512B的最后32B
        DataCopy(statusFlagLocal, statusCntGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)],
                 SERVER_STATE_ALIGN - SPLIT_BLOCK_DATA_SIZE);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        LocalTensor<uint32_t> flagVal = statusFlagLocal.ReinterpretCast<uint32_t>();
        if (flagVal.GetValue(0) == 1) {
            break;
        }
    }
    DataCopy(statusCntLocal, statusCntGlobal, SPLIT_BLOCK_DATA_SIZE);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    LocalTensor<uint32_t> cntUint32 = statusCntLocal.ReinterpretCast<uint32_t>();
    tokenCnt = cntUint32.GetValue(0);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitToken(uint32_t tokenCnt, uint32_t serverIdx,
                                                               uint32_t startTokenIdx, TBuf<> &tBuf)
{
    LocalTensor<float> xOutFp32Tensor = xTmpTensor_.template ReinterpretCast<float>();
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor_.template ReinterpretCast<int32_t>();
    GlobalTensor<int32_t> cleanGlobal;
    uint32_t index = 0;
    uint32_t finishNum = 0;

    for (uint32_t idx = 0; idx < tokenCnt; idx++) {
        finishNumTensor_(idx) = 0;
    }

    while (true) {
        if (finishNumTensor_(index) == 1) {
            index = (index + 1) % tokenCnt; // 轮询查询每个有效的index
            continue;
        }

        uint32_t arriveCount;
        CheckDataArriveWithFlag(index + startTokenIdx, serverId_, arriveCount);
        if (arriveCount == 1) {
            uint32_t dstPosition = index;
            GM_ADDR wAddr = GetReceiveAddrBetweenServer(COMM_EP_IDX, serverIdx) + SERVER_STATE_ALIGN;
            CopyInAndOut(xOutFp32Tensor, xOutInt32Tensor, wAddr, index, dstPosition, arriveCount);

            SyncFunc<AscendC::HardEvent::MTE2_V>();
            SendToExpert();

            // finish更新并clean
            finishNumTensor_(index) = 1;
            uint32_t cleanUpNum = blockCntPerToken_;
            DataCopyExtParams cleanUpParams = {uint16_t(cleanUpNum), sizeof(int32_t), 0U,
                                               +SERVER_STATE_ALIGN - sizeof(int32_t), 0U}; // 清理512B的最后4字节
            LocalTensor<int32_t> cleanBuf =
                tBuf.GetWithOffset<int32_t>(UB_ALIGN / sizeof(int32_t), 0); // 在0偏移位置存放比较结果
            cleanGlobal.SetGlobalBuffer((__gm__ int32_t *)(wAddr));
            SyncFunc<AscendC::HardEvent::MTE3_V>();
            Duplicate<int32_t>(cleanBuf, 0, cleanUpNum * 8); // 8 = UB_ALIGN / 4
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopyPad(cleanGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(int32_t)], cleanBuf, cleanUpParams);
            finishNum++;

            PipeBarrier<PIPE_ALL>();
        } else {
            index = (index + 1) % tokenCnt;
        }
        if (tokenCnt == finishNum) {
            break;
        }
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CheckDataArriveWithFlag(uint32_t beginIdx, uint32_t serverIdx,
                                                                             uint32_t &arriveCount)
{
    uint64_t rsvdCnt = 0;
    uint32_t arriveFlagNum = 0;
    uint32_t flagNum = blockCntPerToken_; // flag数量
    uint32_t compareCount =
        Ceil(flagNum, COMPARE_COUNT_PER_BLOCK) * COMPARE_COUNT_PER_BLOCK; // flagNum 向上取整到256倍数
    uint32_t compResultU64Num = Ceil(flagNum, 64);                        // 64：按照64bit位进行划分

    DataCopyExtParams expFlagCopyParams{static_cast<uint16_t>(flagNum), static_cast<uint32_t>(sizeof(float)),
                                        static_cast<uint32_t>(SERVER_STATE_ALIGN - sizeof(uint32_t)), 0, 0};
    DataCopyPadExtParams<uint32_t> expFlagPadParams{false, 0U, 0U, 0U}; // 不特殊填充
    GlobalTensor<uint32_t> dataFlagGlobal;

    // token的起始地址
    GM_ADDR wAddr =
        GetReceiveAddrBetweenServer(COMM_EP_IDX, serverIdx) + SERVER_STATE_ALIGN + beginIdx * sendTokenLengthAlign_;
    dataFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(wAddr));

    // flag放到flagRecvTensor_
    DataCopyPad(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams, expFlagPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    GatherMask(flagGatherOutTensor_, flagRecvTensor_, flagRecvGatherMask_, true, uint32_t(1),
               {1, (uint16_t)(flagNum), 1, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    CompareScalar(flagCompResultU8_, flagGatherOutTensor_, uint32_t(1), AscendC::CMPMODE::EQ, compareCount);
    SyncFunc<AscendC::HardEvent::V_S>();

    for (uint32_t i = 0; i < compResultU64Num; i++) {
        uint64_t flagCompMask = flagCompResultLtU64_(i);
        int64_t firstValidIdx = ScalarGetSFFValue<0>(flagCompMask); // 找到0则表示数据没到
        if (firstValidIdx == -1) {                                  // 本次数据全到
            arriveFlagNum += 64U;                                   // 64：ScalarGetSFFValue操作单位为64bit位
        } else {
            arriveFlagNum += uint32_t(firstValidIdx);
            break;
        }
    }
    if (arriveFlagNum > flagNum) {
        arriveFlagNum = flagNum;
    }
    arriveCount = uint32_t(arriveFlagNum / blockCntPerToken_); // 返回token总数
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CopyInAndOut(
    LocalTensor<float> xOutFp32Tensor, LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index,
    uint32_t dstPosition, uint32_t arriveCount)
{
    GlobalTensor<ExpandXOutType> dataFlagGlobal, expandXOutGlobal;
    dataFlagGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr));
    expandXOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOutGM_) + (dstPosition)*axisH_);
    // 连续512的数据块搬运
    DataCopyExtParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerToken_ * arriveCount),
                                         static_cast<uint32_t>(SERVER_STATE_ALIGN), static_cast<uint32_t>(UB_ALIGN), 0,
                                         0};
    // 量化缩放因子
    DataCopyExtParams scalesCopyParams{uint16_t(arriveCount), static_cast<uint32_t>(sizeof(float)),
                                       static_cast<uint32_t>((blockCntPerToken_ * SERVER_STATE_ALIGN) / UB_ALIGN - 1),
                                       0U, 0U};
    // 重组分发
    DataCopyExtParams tokenCopyParams{
        uint16_t(arriveCount), hOutSize_,
        static_cast<uint32_t>((blockCntPerToken_ * SERVER_STATE_ALIGN - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams expandIdxCopyParams{
        uint16_t(arriveCount), EXPAND_IDX_INFO * sizeof(int32_t),
        static_cast<uint32_t>((blockCntPerToken_ * SERVER_STATE_ALIGN) / UB_ALIGN - 1), 0U, 0U};
    DataCopyPadExtParams<ExpandXOutType> srcTokenPadParams{false, 0U, 0U, 0U};

    DataCopyPad(xTmpTensor_, dataFlagGlobal[finishNumTensor_(index) * sendTokenLength_ / sizeof(ExpandXOutType)],
                srcTokenCopyParams, srcTokenPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
}

// 0-31 server  32-63 recieve

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CommunicateBetweenServer(uint32_t beginServerId,
                                                                              uint32_t serverNum, uint32_t aivNum)
{
    ConstructDataAndFlagBatchWriteInfo(beginServerId, serverNum, aivNum);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    if ASCEND_IS_AIV {
        if (aivId_ == 0) {
            // 调用BatchWrite API
            bufferChosenGlobal_(0) = bufferId_ ^ 1;
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                bufferChosenGlobal_);
        }
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::resetMaxCnt(int32_t cntPosIndex,
                                                                                        int32_t curExpertCnt)
{
    AscendC::SetAtomicMax<int32_t>();
    TBuf<> tbuf;
    tpipe_->InitBuffer(tbuf, UB_ALIGN);
    LocalTensor<int32_t> localSet = tbuf.Get<int32_t>();
    localSet(0) = curExpertCnt;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(sendCountGMTensor_[cntPosIndex], localSet, UB_ALIGN);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::FillTriple(LocalTensor<ExpandXOutType> &xOutTensor,
                                                                uint32_t srcRankIndex, uint32_t tokenIndex, uint32_t k)
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = srcRankIndex;
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex;
    xOutTint32(tokenQuantAlign_ + 2) = k;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::IsInSameServer(uint32_t targetRankId)
{
    return targetRankId / serverRankSize_ == serverId_;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendToExpert()
{
    // xTmpTensor_是否是 int32_t LocalTensor<ExpandXOutType> xTmpTensor_;
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    TBuf<> tbuf;
    tpipe_->InitBuffer(tbuf, hScaleSizeAlign_);
    LocalTensor<ExpandXOutType> tokenData = tbuf.Get<ExpandXOutType>();
    DataCopy(tokenData, xTmpTensor_, hScaleSizeAlign_);

    LocalTensor<uint32_t> xInTensor = xTmpTensor_.template ReinterpretCast<uint32_t>();
    uint32_t srcRankIndex = xInTensor(tokenQuantAlign_);
    uint32_t tokenIndex = xInTensor(tokenQuantAlign_ + 1);

    for (int32_t topKIndex = 0; topKIndex < axisK_; topKIndex++) {                      // 发moe专家
        uint32_t dstExpertId = xInTensor(moeListAlign_ + topKIndex);                    // 取出专家id
        uint32_t curExpertCnt = xInTensor(moeCntAlign_ + topKIndex);                    // 取出cnt
        uint32_t dstRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_; // 发往的卡号

        if (dstExpertId >= moeExpertNum_)
            continue; // 无效数据
        if (!IsInSameServer(dstRankId))
            continue; // 不同server

        resetMaxCnt((shareRankNumInServer_ + dstExpertId - startMoeExpertId_) * epWorldSize_ + srcRankIndex,
                    curExpertCnt);
        FillTriple(tokenData, srcRankIndex, tokenIndex, topKIndex);

        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                                            (expertPerSizeOnWin_ * (srcRankIndex * moeExpertNumPerRank_ +
                                                                    dstExpertId % moeExpertNumPerRank_)) +
                                            hAlignWinSize_ * curExpertCnt); // 计算地址偏移
        dstWinGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType *)rankGM);

        DataCopyPad(dstWinGMTensor, tokenData, hCommuCopyOutParams_);
    }


    for (int32_t shareIndex = 0; shareIndex < sharedExpertNum_; shareIndex++) { // 发共享专家
        // 计算需要发送的所有的共享专家的卡号
        uint32_t rankIDSharedGroup = epRankId_ % rankNumPerSharedExpert_;

        uint32_t dstRankId = shareIndex * rankNumPerSharedExpert_ + rankIDSharedGroup;
        uint32_t curExpertCnt = tokenIndex; // cnt即tokenid

        if (!IsInSameServer(dstRankId))
            continue; // 不同server

        resetMaxCnt((dstRankId - startRankId_) * epWorldSize_ + srcRankIndex, curExpertCnt);
        FillTriple(tokenData, srcRankIndex, tokenIndex, axisK_ + shareIndex);

        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                                            expertPerSizeOnWin_ * srcRankIndex + hAlignWinSize_ * curExpertCnt);

        dstWinGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType *)rankGM);

        DataCopyPad(dstWinGMTensor, tokenData, hCommuCopyOutParams_);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetStatus()
{
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    // 专家编号均为server内编号, 卡号均为全局卡号
    SplitToCore(expertNumInServer_, aivNum_, startExpertId_, endExpertId_, sendExpertNum_);

    if (startExpertId_ >= expertNumInServer_) {
        return;
    }

    for (uint32_t curExpertId = startExpertId_; curExpertId < endExpertId_; ++curExpertId) {
        for (uint32_t srcRankId = 0; srcRankId < epWorldSize_; ++srcRankId) {
            int32_t localPosIndex = ((curExpertId - startExpertId_) * epWorldSize_ + srcRankId) * 8 + 1;
            int32_t globalPosIndex = curExpertId * epWorldSize_ + srcRankId;
            statusTensor_(localPosIndex) = sendCountGMTensor_(globalPosIndex);
        }
    }
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t expertIndex = startExpertId_; expertIndex < endExpertId_; ++expertIndex) {
        for (uint32_t srcRankId = 0; srcRankId < epWorldSize_; ++srcRankId) {
            uint32_t dstRankId = expertIndex + startRankId_; // 共享专家
            uint32_t offset = stateOffset_ * srcRankId;
            if (expertIndex >= shareRankNumInServer_) {
                dstRankId =
                    (expertIndex - shareRankNumInServer_) / moeExpertNumPerRank_ + shareRankNumInServer_ + startRankId_;
                offset += ((expertIndex - shareRankNumInServer_) % moeExpertNumPerRank_ * epWorldSize_ * stateOffset_);
            }
            GM_ADDR rankGM =
                (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_EP_IDX, dstRankId) + offset); // 计算地址偏移
            rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
            // 按32对齐拷贝，8是32字节包含的元素个数, 本卡数据需要去掉起始index偏移
            DataCopy<int32_t>(rankGMTensor, statusTensor_[expertIndex * 8], 8UL);
        }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ReduceMaxInplace(const LocalTensor<float> &srcLocal,
                                                                      uint32_t count)
{
    uint64_t repsFp32 = count >> 6;       // 6 is count / elemPerRefFp32
    uint64_t offsetsFp32 = repsFp32 << 6; // 6 is repsFp32 * elemPerRefFp32
    uint64_t remsFp32 = count & 0x3f;     // 0x3f 63, count % elemPerRefFp32
    const uint64_t elemPerRefFp32 = 64UL; // 256 bit / sizeof(float)
    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
    // 8 is rep stride
    WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SyncCntOnCore(LocalTensor<float> &gatherMaskOutTensor,
                                                                   LocalTensor<uint32_t> &gatherTmpTensor,
                                                                   LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, 2); // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;              // 源操作数每个datablock只需要处理两个元素
    SyncFunc<AscendC::HardEvent::S_V>();

    // 将当前核对应的专家recv cnt收集到gatherMaskOutTensor
    uint64_t rsvdCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
               {1, (uint16_t)recStatusNumPerCore_, 1, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // 对当前核对应的专家recv cnt求和
    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) // 对inner要求32对齐
                                        * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);

    // 把当前核的所有专家的recv cnt之和写到workspace
    uint32_t coreOffset = WORKSPACE_ELEMENT_OFFSET * aivNum_;
    GM_ADDR wAddr = (__gm__ uint8_t *)(recvCntWorkspaceGM_) + coreOffset * aivId_; // 写workspace需要按照512字节对齐
    GlobalTensor<int32_t> sumTensor;
    sumTensor.SetGlobalBuffer((__gm__ int32_t *)wAddr);
    uint16_t workCoreNum = MIN(recvWinBlockNum_, aivNum_);
    // 每个核把sumOfRecvCnt重复写workCoreNum份
    LocalTensor<int32_t> sumCoreTensor = sumCoreBuf_.Get<int32_t>();
    // 仅处理每个datablock的首元素（对应maskArray[0]的bit0）。操作数为32bit情况下，maskArray只有第0个元素有效
    // 每个元素占4字节，每个32字节处理8份，mask中每8个bit的填充第1位
    uint64_t maskArray[2] = {0x0101010101010101, 0};
    // 每个核一个datablock，总共需要处理workCoreNum个核。每个repeat总共256字节，可以处理8个datablock
    uint8_t repeatTimes = (workCoreNum + 7) / 8;
    // 1代表单个repeat内不同的datablock连续，没有跳过
    // 8代表不同repeat的首元素间隔8个datablock
    Duplicate<int32_t>(sumCoreTensor, sumOfRecvCnt, maskArray, repeatTimes, 1, 8); // [cnt,...,cnt...,cnt]aivnum_;
    DataCopyParams sumIntriParams{static_cast<uint16_t>(workCoreNum), 1, 0, 15};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(sumTensor, sumCoreTensor, sumIntriParams);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::BufferInit()
{
    tpipe_->Reset();
    uint32_t waitStatusBufSize = (((recStatusNumPerCore_ * UB_ALIGN) > 256) ? (recStatusNumPerCore_ * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize); // 1024/24 * 32B = 43 * 32B
                                                           // 内存复用，取大
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = recvWinBlockNum_ * sizeof(float);
    uint64_t gatherMaskOutSize =
        (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);     // recStatusNumPerCore_32对齐后大小  * 32B
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);          // 48 * 32B
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN);         // 48 * 32B
    tpipe_->InitBuffer(sumContinueBuf_, aivNum_ * sizeof(float)); // 48 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);                 // 96B
    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hOutAlignUbSize_);    // 7k*2 + 32 + 12
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::TimeOutDetection()
{
    uint32_t toRankId;
    uint64_t stateCheckOffset = (dataState_ == 0) ? TIMEOUT_OFFSET : (TIMEOUT_OFFSET - WIN_STATE_OFFSET);
    GlobalTensor<float> timeoutCheckGMTensor;
    for (uint32_t index = startStatusIndex_; index < startStatusIndex_ + recStatusNumPerCore_; index++) {
        toRankId = index % epWorldSize_;
        GM_ADDR timeoutCheckGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_EP_IDX, toRankId) + stateCheckOffset);
        timeoutCheckGMTensor.SetGlobalBuffer((__gm__ float *)(timeoutCheckGM));
        DataCopy<float>(timeoutCheckGMTensor, statusFp32Tensor_, TIMEOUT_DETECTION_TX_UNITS);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitDispatchClearStatus()
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101,
                                 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0,这里理状态区
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitDispatch()
{
    BufferInit();
    startExpertId_ = startStatusIndex_; // 后面LocalWinCopy分核与此处保持一致
    endExpertId_ = startExpertId_ + recStatusNumPerCore_;
    sendExpertNum_ = recStatusNumPerCore_;
    if (unlikely(startStatusIndex_ >= rscvStatusNum_)) {
        SyncAll<true>();
        return;
    }
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    gatherTmpTensor.SetValue(0, 1);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = 1;                                       // gatherMask + sum 相关参数
    float compareTarget = sumTarget_ * recStatusNumPerCore_; // 1.0 *recStatusNumPerCore_
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t timeoutCheckStart = static_cast<uint64_t>(GetSystemCycle());
    uint64_t timeoutCheckEnd, timeoutCheckDuration;
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
        timeoutCheckEnd = static_cast<uint64_t>(GetSystemCycle());
        timeoutCheckDuration = (timeoutCheckEnd - timeoutCheckStart) / CYCLES_PER_US;
        if (timeoutCheckDuration > TIMEOUT_DETECTION_THRESHOLD) {
            TimeOutDetection();
        }
    }
    // 清状态
    WaitDispatchClearStatus();

    // 核间同步token cnt ,为了防止同时读，每个核读不同的偏移量
    SyncCntOnCore(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);

    SyncAll<true>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::GetCumSum(LocalTensor<int32_t> &outLocal,
                                                                                      uint32_t totalCount)
{
    outLocal = gatherMaskOutBuf_.Get<int32_t>(); // recStatusNumPerCore_32对齐后大小  * 32B
    // 获取workspace中每个核的recvcnt
    GM_ADDR wAddr = (__gm__ uint8_t *)(recvCntWorkspaceGM_) + WORKSPACE_ELEMENT_OFFSET * aivId_;
    GlobalTensor<int32_t> sumTensor;
    sumTensor.SetGlobalBuffer((__gm__ int32_t *)wAddr);

    // 只需要拷贝totalCount个核的recv cnt
    uint16_t copySumNum = totalCount;
    uint16_t copyStride = 16 * aivNum_ - 1;
    DataCopyParams sumIntriParams{static_cast<uint16_t>(copySumNum), 1, copyStride, 0}; // 每隔512B取一个数
    LocalTensor<int32_t> sumLocalTensor = sumLocalBuf_.Get<int32_t>();                  // aivNum_*32B
    DataCopy(sumLocalTensor, sumTensor, sumIntriParams);

    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0); // 8
    gatherSumPattern.SetValue(0, 1);
    uint32_t mask = 1;
    uint64_t rsvdCnt = 0;
    LocalTensor<int32_t> sumContinueTensor = sumContinueBuf_.Get<int32_t>(); // aivnum *float
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::S_V>();
    // 把之前核收到的cnt取出来
    GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, copySumNum, 1, 0}, rsvdCnt);
    // height, width(按照32字节对齐padding后总元素个数), nNum，结果矩阵第一列为对应行的求和结果
    uint32_t innerSumParams = (copySumNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN / sizeof(float);
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = sumContinueTensor.ReinterpretCast<float>();
    // 0核前面所有核recv cnt总和是0
    if (totalCount == 0) {
        outLocal.SetValue(0, 0);
        return;
    }
    SumParams sumParams{1, innerSumParams, totalCount};
    Sum(recvCntSumOutTensor, tmpFp32, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    // 最终输出outLocal第0个元素是当前核前面所有核recv cnt总和
    outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0)); // 求一个累加和
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::DoWindowCopy(LocalTensor<int32_t> &outCountLocal)
{
    uint32_t index = 0;
    uint32_t beginIdx = outCountLocal.GetValue(0);
    statusTensor_ = waitStatusBuf_.Get<int32_t>();
    DataCopyPadExtParams<ExpandXOutType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams dataCopyExpandIdxParams{1U, sizeof(int32_t) * EXPAND_IDX_INFO, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    for (uint32_t index = startExpertId_; index < endExpertId_; index++) {
        uint32_t i = index - startExpertId_;
        uint32_t count = statusTensor_.GetValue(i * 8 + 1); // 收到的专家的cnt
        outCountLocal.SetValue(i, beginIdx + count);

        uint32_t winOffset = index;
        if (!isShareExpertRankFlag_) {
            if (moeExpertNumPerRank_ > 1) { // moe专家卡且一卡多专家场景 转换成数据区的排布偏移
                winOffset = index % epWorldSize_ * moeExpertNumPerRank_ + index / epWorldSize_;
            }
        }
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + winOffset * expertPerSizeOnWin_;
        GlobalTensor<ExpandXOutType> tokGlobal;
        GlobalTensor<ExpandXOutType> expandXOutGlobal;
        tokGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        LocalTensor<int32_t> xTmpTensorInt;
        for (uint32_t j = 0; j < count; j++) {
            tokGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr + j * hAlignWinSize_));
            // 将数据从Window拷贝到UB
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopyPad(xTmpTensor_, tokGlobal, hCommuCopyOutParams_, copyPadExtParams);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            xTmpTensorInt = xTmpTensor_.template ReinterpretCast<int32_t>();
            DataCopyPad(expandIdxGMTensor_[(beginIdx + j) * EXPAND_IDX_INFO], xTmpTensorInt[tokenQuantAlign_],
                        dataCopyExpandIdxParams); // 拷贝三元组
            // 拷贝数据
            expandXOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOutGM_) + (beginIdx + j) * axisH_,
                                             axisH_);
            DataCopyPad(expandXOutGlobal, xTmpTensor_, expandXCopyParams_);
            xQueue_.FreeTensor(xTmpTensor_);
        }
        beginIdx += count;
    }
    totalCnt_ = beginIdx;
    lastCore_ = MIN(rscvStatusNum_, aivNum_) - 1;

    GlobalTensor<int32_t> sendCountsGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    DataCopyPad(sendCountsGlobal[startExpertId_], outCountLocal, dataCopyOutParams);
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::LocalWindowCopy()
{
    DataCopyParams dataStateParams{1U, sizeof(uint32_t), 0U, 0U};
    dataStateLocalTensor_ = gatherMaskOutBuf_.Get<uint32_t>();
    dataStateLocalTensor_.SetValue(0, FLAG_AFTER_WAIT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams);
    LocalTensor<int32_t> outCountLocal;
    if (startExpertId_ >= rscvStatusNum_) { // 分核已与前面的waitDispatch里保持一致
        return;
    }
    GetCumSum(outCountLocal, aivId_);
    DoWindowCopy(outCountLocal);
}

// 更新tokenNumsOut tensor
template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::UpdateTokenNumsOut()
{
    // 最后一个核做更新，Moe专家只有最后一个核有计算出所有 sendCountsGlobal
    if (!isShareExpertRankFlag_) {
        if (moeExpertNumPerRank_ > 1) {
            SyncAll<true>();
        }
    }

    if (aivId_ == lastCore_) {
        // Moe专家token总数在Cumsum内计算得出
        uint32_t tokenNum = totalCnt_;
        expertTokenNumsOutGMTensor_.SetValue(0, tokenNum);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            expertTokenNumsOutGMTensor_);
        // moe一卡多专家场景下更新moe专家卡对应expertTokenNums数据
        if (moeExpertNumPerRank_ != 1) {
            if (!isShareExpertRankFlag_) {
                uint32_t tokenSums = 0;
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                GlobalTensor<int32_t> sendCountsGlobal;
                sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    sendCountsGlobal[epWorldSize_ - 1]);

                uint32_t firstMoeCnt = sendCountsGlobal.GetValue(epWorldSize_ - 1);
                tokenSums = firstMoeCnt;
                expertTokenNumsOutGMTensor_.SetValue(0, tokenSums);
                DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    expertTokenNumsOutGMTensor_[0]);
                for (uint32_t localMoeIndex = 1; localMoeIndex < moeExpertNumPerRank_; ++localMoeIndex) {
                    uint32_t preOffset = epWorldSize_ * (localMoeIndex - 1) + epWorldSize_ - 1;
                    uint32_t curOffset = epWorldSize_ * localMoeIndex + epWorldSize_ - 1;
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        sendCountsGlobal[preOffset]);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        sendCountsGlobal[curOffset]);
                    uint32_t preMoeIndexCnt = sendCountsGlobal.GetValue(preOffset);
                    uint32_t curMoeIndexCnt = sendCountsGlobal.GetValue(curOffset);
                    tokenSums = ((expertTokenNumsType_ == 0) ? tokenSums : 0) + (curMoeIndexCnt - preMoeIndexCnt);
                    expertTokenNumsOutGMTensor_.SetValue(localMoeIndex, tokenSums);
                    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        expertTokenNumsOutGMTensor_[localMoeIndex]);
                }
            }
        }
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        DispatchBetweenServer();
        CommunicateBetweenServer(0, serverNum_, aivNum_);
        WaitWindow(aivNum_);
        PipeBarrier<PIPE_ALL>();
        SetStatus();
        WaitDispatch();
        LocalWindowCopy();
        UpdateTokenNumsOut();
    }
}

} // namespace Mc2Kernel
#endif // MOE_DISTRIBUTE_DISPATCH_V2_H
