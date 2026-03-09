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
#include "adv_api/reduce/reduce.h"
#include "adv_api/math/xor.h"
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_v2_constant.h"
#include "moe_distribute_dispatch_v2_quant.h"
#include "moe_distribute_elastic.h"

#include "moe_distribute_v2_base.h"
#include "check_winsize.h"
#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#else 
#include "../../common/inc/kernel/moe_distribute_base.h"
#endif
#include "log.h"
#include "newlog.h"

namespace MoeDistributeDispatchV2HostKfcImpl {
constexpr uint32_t STATE_SIZE = 2048 * 1024; // 2M
constexpr uint64_t TIMEOUT_OFFSET = 1024UL * 1024UL;
constexpr uint8_t BUFFER_NUM = 2; // 多buf
constexpr uint8_t BUFFER_SINGLE = 1;
constexpr uint32_t STATE_OFFSET = 32U; // 状态空间偏移地址
constexpr uint8_t COMM_NUM = 2;        // 通信域大小
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint8_t COMM_TP_IDX = 1;
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL;
constexpr uint64_t DISPATCH_STATE_WIN_OFFSET = 768UL * 1024UL;
constexpr uint64_t CUMSUM_CAL_OFFSET = 868UL * 1024UL; // 768 + 100
constexpr uint64_t CUMSUM_FLAG_OFFSET = 876UL * 1024UL; // 868 + 8
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
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint8_t UB_ALIGN_DATA_COUNT = 8U; // 8 = UB_ALIGN / sizeof(float) = UB_ALIGN / sizeof(int32_t)

#define TemplateDispatchKFCTypeClass                                                                                           \
    typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist, bool IsNeedAllgather
#define TemplateDispatchKFCTypeFunc XType, ExpandXOutType, QuantMode, IsSmoothScaleExist, IsNeedAllgather

using namespace AscendC;
using namespace MoeDistributeV2Base;
using namespace MoeDispatchLog;

using namespace MoeDispatchLog_NEW;
#define LOG_INFO(...) this->logger.LogInfo(__LINE__,__VA_ARGS__)
#define LOG_INIT(...) this->logger.Init(__VA_ARGS__)

template <TemplateDispatchKFCTypeClass>
class MoeDistributeDispatchV2HostKfc {
public:
    __aicore__ inline MoeDistributeDispatchV2HostKfc(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, 
                                GM_ADDR expertScales, GM_ADDR elasticInfo,GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                GM_ADDR expandIdxOut,GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
                                GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:

    __aicore__ inline void InitTemp();
    __aicore__ inline void CalExpertTotalSendCnt();
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
    __aicore__ inline void FillQuadruple(LocalTensor<XType> &xOutTensor, uint32_t tokenIndex);
    template <typename T> __aicore__ inline void CopyTokenToWinOut(LocalTensor<T> &xOutTensor, GM_ADDR rankGM, uint32_t blockCnt);
    __aicore__ inline void SingleTokenProcess(uint32_t tokenIndex, uint32_t dstServerId, uint32_t cnt);
    __aicore__ inline void DispatchAndCountTokens(uint32_t startIndex, uint32_t endIndex, uint32_t serverId, bool process);
    __aicore__ inline void ConstructDataAndFlagBatchWriteInfo(uint32_t beginServerId, uint32_t serverNum,
                                                              uint32_t aivNum);
    __aicore__ inline void WaitStatusFlag(uint32_t serverIdx, uint32_t &tokenCnt);
    __aicore__ inline void WaitToken(uint32_t tokenCnt, uint32_t serverIdx, uint32_t startTokenIdx, TBuf<> &tBuf);
    __aicore__ inline void CheckDataArriveWithFlag(GlobalTensor<uint32_t> &dataFlagGlobal, uint32_t &arriveCount, uint32_t flagNum,
                                                   LocalTensor<uint32_t> &flagRecvTensor, LocalTensor<uint32_t> &flagGatherOutTensor);
    __aicore__ inline void CopyInAndOut(LocalTensor<float> xOutFp32Tensor, LocalTensor<int32_t> xOutInt32Tensor,
                                        GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount);
    __aicore__ inline void SetServerFlag();
    __aicore__ inline void ExpertOffsetCal();
    __aicore__ inline void SendToServer();
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void ProcessToken(
            GM_ADDR rankGM, uint32_t tokenIndex, uint32_t topKIndex,
            DataCopyPadParams& padParams, DataCopyParams& scaleInParams, uint32_t expertIndex,
            LocalTensor<XType>& tokenData, uint32_t srcRankIndex, float expertScale);
    __aicore__ inline void SendToExpert(uint32_t rcvCnt);
    __aicore__ inline void resetMaxCnt(int32_t cntPosIndex, int32_t curExpertCnt);
    __aicore__ inline void SetStatus();
    __aicore__ inline void InitStatusTensor();
    __aicore__ inline void BufferInit();
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
                                      uint32_t tokenIndex, uint32_t k, float expertScale);
    __aicore__ inline void SyncCntOnCore(LocalTensor<float> &gatherMaskOutTensor,
                                         LocalTensor<uint32_t> &gatherTmpTensor,
                                         LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline GM_ADDR GetBaseWindOutAddrByServer(__gm__ HcclOpParam *addr, const int32_t serverId,
                                                         const int32_t curRankId);
    __aicore__ inline GM_ADDR GetBaseWindInAddrByServer(__gm__ HcclOpParam *addr, const int32_t serverId,
                                                        const int32_t curRankId);
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
                                            LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void SetExpertTokenNums();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum);
    __aicore__ inline uint32_t CheckDataArriveWithFlagInServer(uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt);
    __aicore__ inline void CopyOut(LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount);
    __aicore__ inline void WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum);
    // __aicore__ inline GM_ADDR GetBaseWindAddrByRankId(__gm__ HcclOpParam *addr, const int32_t serverId,
    //                                                   const int32_t curRankId);
    // __aicore__ inline GM_ADDR GetBaseWindStateAddrByRankId(__gm__ HcclOpParam *addr, const int32_t serverId,
    //                                                        const int32_t curRankId);

    __aicore__ inline GM_ADDR GetSendAddrBetweenServer(uint8_t ctxIdx, const int32_t serverId)
    {
        uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankIdOriginal_ : tpRankId_);
        //return GetBaseWindOutAddrByServer(winContext_[ctxIdx], serverId, curRankId);
        return gmTemp + 2 * 1024 * 1024 + serverId * 4UL * 1024 * 1024;
    }

    __aicore__ inline GM_ADDR GetReceiveAddrBetweenServer(uint8_t ctxIdx, const int32_t serverId)
    {
        uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankIdOriginal_ : tpRankId_);
        //return GetBaseWindInAddrByServer(winContext_[ctxIdx], serverId, curRankId);
        return gmTemp + 2 * 1024 * 1024 + serverId * 4UL * 1024 * 1024;
    }

    __aicore__ inline GM_ADDR GetSendStatusAddrBetweenServer(uint8_t ctxIdx, const int32_t serverId)
    {
        return gmTemp;
    }

    __aicore__ inline GM_ADDR GetReceiveStatusAddrBetweenServer(uint8_t ctxIdx, const int32_t serverId)
    {
        return gmTemp;
    }

    // __aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    // {
    //     uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankIdOriginal_ : tpRankId_);
    //     uint64_t winDataSizeOffset = (ctxIdx == COMM_EP_IDX) ? winDataSizeOffsetEp_ : winDataSizeOffsetTp_;
    //    // return GetBaseWindAddrByRankId(winContext_[ctxIdx], rankId, curRankId) + winDataSizeOffset;
    //     return gmTemp + 1UL * 1024 * 1024 *1024 + rankId * 4UL * 1024 * 1024;
    // }

    // __aicore__ inline GM_ADDR GetWindStateAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    // {
    //     uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankIdOriginal_ : tpRankId_);
    //     //return GetBaseWindStateAddrByRankId(winContext_[ctxIdx], rankId, curRankId) + dataState_ * WIN_STATE_OFFSET;
    //     return gmTemp + 1UL * 1024 * 1024 *1024 + 512UL * 1024 * 1024 + rankId * 1024;
    // }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankIdOriginal_ : tpRankId_);
        uint64_t winDataSizeOffset = (ctxIdx == COMM_EP_IDX)? winDataSizeOffsetEp_ : winDataSizeOffsetTp_;
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
    GlobalTensor<float> expertScalesGMTensor_;
    GlobalTensor<float> expandScalesOutGMTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<int32_t> sendCountGMTensor_;
    GlobalTensor<float> fpWinTpGatherOutGMTensor_;
    GlobalTensor<int32_t> winTpEpCntGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;
    GlobalTensor<uint32_t> selfhcclDataStatusTensor_;
    GlobalTensor<uint64_t> dataBatchWriteInfoTensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<float> selfRankWinInGMTensor_;

    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<XType> recvTmpTensor_;
    LocalTensor<int32_t> tpTmpTensor_;
    LocalTensor<float> floatLocalTemp_;
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
    LocalTensor<uint32_t> flagTensor_;
    LocalTensor<float> statusCleanFp32Tensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<int32_t> tokenNumToExpertTensor_;


    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> tokenNumBuf_;
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
    TBuf<> xSendBuf_;
    TBuf<> flagBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, 1> xInQueue_;                        // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;                      // 量化使用，量化后的输出

    LocalTensor<XType> tokenData_;
    TQue<QuePosition::VECOUT, 1> sendtoexpertQueue_;

    TBuf<> localSetTbuf_;
    LocalTensor<int32_t> localSet_;

    GM_ADDR expandXOutGM_;
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
    uint32_t hAlignSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t hCommuSize_{0};
    uint32_t axisHCommu_{0};
    uint32_t moeListAlign_{0};
    uint32_t moeCntAlign_{0};
    uint32_t moeExpertScalesAlign_{0};
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
    uint64_t totalWinSizeTp_{0};
    uint64_t totalWinSizeEp_{0};
    uint32_t gatherCount_{0};
    uint32_t expertTokenNumsType_{1};
    uint32_t stateOffset_{0};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    uint32_t sendXTypeElemAlign_{0};
    int32_t zeroComputeExpertNum_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t remainderRankNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t sendToSharedExpTokenCnt_{0};
    uint32_t maxSize_{0};
    uint32_t bufferNum_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t blockCntPerCnt_{0};
    uint32_t blockCntPerTokenToExpert_{0};
    uint32_t maxBlockCnt_{0};
    uint32_t sendTokenLengthAlign_{0};
    uint32_t sendTokenLength_{0};
    uint32_t serverNum_{0};
    uint32_t hScaleIdxSize_{0};
    uint32_t expertScaleAlign_{0};
    uint32_t scaleInBytes_{0};
    uint32_t scaleOutBytes_{0};
    uint32_t scalesCount_{0};
    uint32_t aivUsedCumSum_{0}; //todo：初始化
    uint32_t aivUsedDispatch_{0};
    uint32_t startId_{0};
    uint32_t endId_{0};
    uint32_t sendNum_{0};
    uint32_t statusCntAlign_{0};
    uint32_t tBufRealSize_{0};

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpParam *winContext_[COMM_NUM]{nullptr, nullptr};

    DataCopyExtParams floatDataCopyParams_;
    DataCopyExtParams expandXCopyParams_;
    DataCopyParams xCopyParams_;
    DataCopyExtParams hCommuCopyOutParams_;
    DataCopyExtParams scaleOutParams_;
    DataCopyParams doWindowCopyOutParams_;

    MoeDistributeDispatchV2Quant<TemplateDispatchKFCTypeFunc> quantInst_;

    MoeDispatchLog_NEW::Log logger;
};

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalExpertTotalSendCnt() 
{
    uint64_t maskCnt = 0;
    uint32_t compareCount = 0;
    uint32_t maskSizePerExpert = 0;
    uint32_t calCnt = moeExpertNum_;
    //uint32_t startExpertId, endExpertId, sendExpertNum;
    uint32_t mask = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    LocalTensor<uint8_t> expertMaskTensorU8 = subExpBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> expertMaskTensorU32 = subExpBuf_.Get<uint32_t>();
    LocalTensor<int32_t> gatherTempTensor = gatherMaskTBuf_.Get<int32_t>();;
    //SplitToCore(calCnt, aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum);
    LOG_INFO("expertIdsTensor_",expertIdsTensor_,{axisBS_,axisK_});
    // if(startExpertId >= endExpertId || sendExpertNum == 0){
    //     return;
    // }
    compareCount = Ceil(mask * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    maskSizePerExpert = Ceil(compareCount / BITS_PER_BYTE, UB_ALIGN) * UB_ALIGN; // 为了匹配CompareScalar接口1bit一共8位
    
    // if(isExpertMaskFlag_) { //TODO:需要去放在某个地方
    //     uint64_t rsvdCnt = 0;
    //     uint32_t mask = expertIdsCnt_;
    //     LocalTensor<int32_t> tmpExpertIdsTensor = subExpBuf_.Get<int32_t>();
    //     DataCopyPad(tmpExpertIdsTensor, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    //     SyncFunc<AscendC::HardEvent::MTE2_V>();
    //     GatherMask(expertIdsTensor_, tmpExpertIdsTensor, gatherMaskTensor_, true, mask, {1, 1, 1, 0}, rsvdCnt);
    // }

    for (int32_t expertIndex = 0; expertIndex < moeExpertNum_; expertIndex++) {
        int32_t dstExpertId = expertIndex;// + startExpertId;
        CompareScalar(expertMaskTensorU8[maskSizePerExpert * expertIndex], expertIdsTensor_, dstExpertId, CMPMODE::EQ, compareCount);
        PipeBarrier<PIPE_V>();
        GatherMask(gatherTempTensor, expertIdsTensor_, expertMaskTensorU32[maskSizePerExpert * expertIndex / sizeof(uint32_t)],
            true, mask, {1, 1, 0, 0}, maskCnt); // 是否可以简化计算
        SyncFunc<AscendC::HardEvent::V_S>();
        tokenNumToExpertTensor_.SetValue(expertIndex, static_cast<uint32_t>(maskCnt)); //todo :性能优化，大量使用SetValue
    }
    // LOG_INFO("tokenNumToExpertTensor_ ",tokenNumToExpertTensor_, {1,calCnt});

    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    GM_ADDR rankGM = (GetSendStatusAddrBetweenServer(COMM_EP_IDX, serverId_));
    CopyTokenToWinOut(tokenNumToExpertTensor_, rankGM, blockCntPerCnt_);

    // GlobalTensor<int32_t> sendCntTensor;
    // sendCntTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(GetSendStatusAddrBetweenServer(COMM_EP_IDX, serverId_)));
    // DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(sendExpertNum * sizeof(int32_t)), 0U, 0U, 0U};
    // DataCopyPad(sendCntTensor[startExpertId], tokenNumToExpertTensor_, dataCopyOutParams);
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitRecieveTilingContext(
    GM_ADDR expandXOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
    gmTemp = workspaceGM + 1UL * 1024 * 1024 * 1024;
    GM_ADDR bufferChoseAddr = workspaceGM + 2UL  * 1024 * 1024 * 1024 + 1000UL * 1024 * 1024;

    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    winContext_[COMM_EP_IDX] = (__gm__ HcclOpParam*)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    //LogInfo(__LINE__, "START InitRecieveTilingContext");
    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(bufferChoseAddr));
    bufferId_ = bufferChosenGlobal_(0);

    //LogInfo(__LINE__, "START bufferChosenGlobal_");

    // 检查hcclwinsize是否越界 
    totalWinSizeEp_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    totalWinSizeTp_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeTp);
    //CheckWindowSize(totalWinSizeEp_, GetWinSize(winContext_[COMM_EP_IDX]), tpipe_, expandXOut);
    //LogInfo(__LINE__,"totalWinSizeEp_: ",totalWinSizeEp_);
    //LogInfo(__LINE__, "START CheckWindowSize");

    serverNum_ = 1; //ToDo
    serverRankSize_ = 2;
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
    scaleInBytes_ = tilingData->moeDistributeDispatchV2Info.scalesCol * tilingData->moeDistributeDispatchV2Info.scalesTypeSize; // 量化使用
    scalesCount_ = tilingData->moeDistributeDispatchV2Info.scalesCount; // 量化使用
    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isExpertMask;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    zeroComputeExpertNum_ = tilingData->moeDistributeDispatchV2Info.zeroComputeExpertNum;
    tpRankId_ = tilingData->moeDistributeDispatchV2Info.tpRankId;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    tpWorldSize_ = tilingData->moeDistributeDispatchV2Info.tpWorldSize;
    LOG_INIT(tpipe_,epRankId_);
    //LogInfo(__LINE__, "epRankId_: ",epRankId_);
    //LogInfo(__LINE__, "axisBS_: ",axisBS_);
    //LogInfo(__LINE__, "axisH_: ",axisH_);
    //LogInfo(__LINE__, "epWorldSizeOriginal_: ",epWorldSizeOriginal_);
    //LogInfo(__LINE__, "expertTokenNumsType_: ",expertTokenNumsType_);
    //LogInfo(__LINE__, "zeroComputeExpertNum_: ",zeroComputeExpertNum_);
    //LogInfo(__LINE__, "axisK_: ",axisK_);
    //LogInfo(__LINE__, "aivNum_: ",aivNum_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitComputeInfo()
{
    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;

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
    //LogInfo(__LINE__, "shareRankNumInServer_: ",shareRankNumInServer_);
    //LogInfo(__LINE__, "serverRankSize_: ",serverRankSize_);
    //LogInfo(__LINE__, "moeExpertNumPerRank_: ",moeExpertNumPerRank_);
    //LogInfo(__LINE__, "expertNumInServer_: ",expertNumInServer_);
    
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
    if (isShareExpertRankFlag_) { // 当前卡是共享专家卡
        rscvStatusNum_ = epWorldSize_;
    } else { // 当前卡是moe专家卡
        rscvStatusNum_ = recvWinBlockNum_;
    }
    //LogInfo(__LINE__, "serverId_: ",serverId_);
    //LogInfo(__LINE__, "epRankIdInServer_: ",epRankIdInServer_);
    //LogInfo(__LINE__, "startRankId_: ",startRankId_);
    //LogInfo(__LINE__, "expertNumInServer_: ",expertNumInServer_);
    //LogInfo(__LINE__, "moeExpertRankNum_: ",moeExpertRankNum_);
    //LogInfo(__LINE__, "moeExpertNumPerRank_: ",moeExpertNumPerRank_);
    //LogInfo(__LINE__, "expertIdsCnt_: ",expertIdsCnt_);
    //LogInfo(__LINE__, "dealRankPerCore_: ",dealRankPerCore_);
}


template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitCommBetweenServerInfo()
{
    statusBufCntAlign_ = Ceil(Ceil(expertNumInServer_ * serverNum_, aivNum_), 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    //LogInfo(__LINE__,"statusBufCntAlign_",statusBufCntAlign_);
    uint32_t statusBufSize = statusBufCntAlign_ * UB_ALIGN;
    totalUsedUB_ += statusBufSize;
    statusSpaceGm_ = GetWindStateAddrByRankId(COMM_EP_IDX, epRankIdOriginal_);
    //printf("[rankID:%d]",epRankId_);
    //LogInfo(__LINE__,"epRankIdOriginal_: ",epRankIdOriginal_);
    //LogInfo(__LINE__,"ADDR statusSpaceGm_",statusSpaceGm_);

    tpipe_->InitBuffer(batchWriteInfoBuf_, BW_ITEM_SIZE);
    batchWriteU64Tensor_ = batchWriteInfoBuf_.Get<uint64_t>();
    batchWriteU32Tensor_ = batchWriteU64Tensor_.template ReinterpretCast<uint32_t>();


    dataBatchWriteInfo_ = recvCntWorkspaceGM_ + WORKSPACE_ELEMENT_OFFSET * aivNum_ * aivNum_;
    dataBatchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint64_t *)(dataBatchWriteInfo_), serverNum_ * B64_PER_BLOCK);

}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitSetWindows(const MoeDistributeDispatchV2TilingData *tilingData)
{
    //LogInfo(__LINE__, "start InitSetWindows");
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempepRankId = 0; tempepRankId < epWorldSize_; tempepRankId++) {
        OOMCheckAddrRange<ExpandXOutType>((__gm__ ExpandXOutType *)(GetWindAddrByRankId(COMM_EP_IDX, tempepRankId)),
                                          totalWinSizeEp_);
        OOMCheckAddrRange<float>((__gm__ float *)(GetWindStateAddrByRankId(COMM_EP_IDX, tempepRankId)), STATE_SIZE);
    }
#endif

    // 当前tpWin区划分为前后两半区，连续两次dispatch，切换半区, combine 数据区使用前面，
    // 即axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine, dispatch使用后面
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffsetEp_ =
        dataState_ * (totalWinSizeEp_ / 2) +
        axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine; // 就是分成两块，去掉combine前面的
    //LogInfo(__LINE__,"winDataSizeOffsetEp_",winDataSizeOffsetEp_);

    winDataSizeOffsetTp_ =
        dataState_ * (totalWinSizeTp_ / 2) + tilingData->moeDistributeDispatchV2Info.a * hSizeAlignCombine;

    //LogInfo(__LINE__,"winDataSizeOffsetTp_",winDataSizeOffsetTp_);

    windowGM_ = GetWindAddrByRankId(COMM_EP_IDX, epRankIdOriginal_);
    ////LogInfo(__LINE__,"windowGM_",windowGM_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    GlobalTensor<ExpandXOutType> winDouble;
    winDouble.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    winDouble.SetGlobalBuffer((__gm__ ExpandXOutType *)(windowGM_));
    OOMCheckAddrRange<ExpandXOutType>((__gm__ ExpandXOutType *)(winDouble.GetPhyAddr()), totalWinSizeEp_);
#endif
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
    //LogInfo(__LINE__, "end InitSetWindows");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitExtraInfo()
{ 
    //LogInfo(__LINE__, "start InitExtraInfo: ");

    //量化使用
    hOutSize_ = axisH_ * sizeof(ExpandXOutType);
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, tokenQuantAlign_, hScaleIdxSize_, scaleOutBytes_, axisH_);

    hScaleIdxSize_ = Ceil(hScaleIdxSize_, UB_ALIGN) * UB_ALIGN; 
    expertScaleAlign_ = hScaleIdxSize_ / sizeof(uint32_t);
    hScaleIdxSize_ = hScaleIdxSize_ + sizeof(float); //用于存放expertscale
    //框间使用
    sendXTypeElemAlign_ = hAlignSize_ / sizeof(uint32_t);
    uint32_t moeListSizeAlign = hAlignSize_ + UB_ALIGN; // moeExpertIdList起始偏移
    moeListAlign_ = moeListSizeAlign / sizeof(int32_t);

    uint32_t moeCntSizeAlign = moeListSizeAlign + Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // moeCntList偏移
    moeCntAlign_ = moeCntSizeAlign / sizeof(int32_t);

    uint32_t moeExpertScaleSizeAlign =  moeCntSizeAlign + Ceil(axisK_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN; // moe expertscales
    moeExpertScalesAlign_ = moeExpertScaleSizeAlign / sizeof(uint32_t);


    sendTokenLength_ = moeExpertScaleSizeAlign + Ceil(axisK_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    blockCntPerToken_ = Ceil(sendTokenLength_, SPLIT_BLOCK_DATA_SIZE);
    blockCntPerCnt_ = Ceil(moeExpertNum_ * sizeof(int32_t), SPLIT_BLOCK_DATA_SIZE);
    blockCntPerTokenToExpert_ = Ceil(hScaleIdxSize_ / sizeof(ExpandXOutType), SPLIT_BLOCK_DATA_SIZE); //框内块数
    sendTokenLengthAlign_ = blockCntPerToken_ * SERVER_STATE_ALIGN;

    hAlignWinSize_ = Ceil(hScaleIdxSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // win区token起始地址对齐512
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hAlignWinSize_;
    hOutAlignUbSize_ = Ceil(hScaleIdxSize_, UB_ALIGN) * UB_ALIGN;

    //LogInfo(__LINE__, "hOutSize_; ",hOutSize_);
    //LogInfo(__LINE__, "hAlignSize_; ",hAlignSize_);
    //LogInfo(__LINE__, "scaleInBytes_; ",scaleInBytes_);
    //LogInfo(__LINE__, "tokenQuantAlign_; ",tokenQuantAlign_);
    //LogInfo(__LINE__, "hScaleIdxSize_; ",hScaleIdxSize_);
    //LogInfo(__LINE__, "axisH_; ",axisH_);
    //LogInfo(__LINE__, "hAlignWinSize_; ",hAlignWinSize_);
    //LogInfo(__LINE__, "expertPerSizeOnWin_; ",expertPerSizeOnWin_);
    LOG_INFO("hOutAlignUbSize_",hOutAlignUbSize_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitMaskInfo()
{
    //LogInfo(__LINE__, "start InitMaskInfo: ");
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
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize); // BS * K * 4 = 32K
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
    //LogInfo(__LINE__, "end InitMaskInfo: ");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitTemp()
{
    if constexpr (QuantMode > UNQUANT) {
        LOG_INFO("[InitTemp]start init");
        // maxSize_ = Ceil(axisH_,UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(receiveDataCastFloatBuf_, maxSize_); // max{28K, BS * K * 4B}
        floatLocalTemp_ = receiveDataCastFloatBuf_.Get<float>();
        tpipe_->InitBuffer(smoothScalesBuf_, maxSize_); // max{28K, BS * K * 4B}
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
        if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN); // 32B
            rowMaxTensor_ = rowMaxBuf_.Get<float>();
        }
    } 

    bufferNum_ = 1;
    tpipe_->InitBuffer(xInQueue_, bufferNum_, hAlignSize_); // 14K * 2
    tpipe_->InitBuffer(xOutQueue_, bufferNum_, hOutAlignUbSize_); // 7K * 2 + 32 + 6
    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf_, dynamicScalesOutGMTensor_);
    LOG_INFO("[InitTemp] END int Temp");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitDispatchBetweenServerInfo()
{
    //LogInfo(__LINE__, "InitDispatchBetweenServerInfo start");
    tpipe_->InitBuffer(expertMaskInputBuf_, expertIdsCnt_ * sizeof(bool));
    totalUsedUB_ += expertIdsCnt_ * sizeof(bool);
    expertMaskInputTensor_ = expertMaskInputBuf_.Get<bool>();

    if constexpr (QuantMode > UNQUANT) {
        tpipe_->InitBuffer(receiveDataCastFloatBuf_, maxSize_); // max{28K, BS * K * 4B}
        totalUsedUB_ += maxSize_;
        floatLocalTemp_ = receiveDataCastFloatBuf_.Get<float>();
        tpipe_->InitBuffer(smoothScalesBuf_, maxSize_); // max{28K, BS * K * 4B}
        totalUsedUB_ += maxSize_;
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
        if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN); // 32B
            rowMaxTensor_ = rowMaxBuf_.Get<float>();
        }
        
        dstExpBuf_ = receiveDataCastFloatBuf_;  // 内存复用
        subExpBuf_ = smoothScalesBuf_;          // 内存复用
    } else {
        tpipe_->InitBuffer(dstExpBuf_, maxSize_);             // BS * K * 4 = 32K
        totalUsedUB_ += maxSize_;
        tpipe_->InitBuffer(subExpBuf_, maxSize_);             // BS * K * 4 = 32K
        totalUsedUB_ += maxSize_;
        // uint32_t tmpTotalUB = totalUsedUB_ + hOutAlignUbSize_ * BUFFER_NUM;
        // bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
        //tpipe_->InitBuffer(xQueue_, bufferNum_, hOutAlignUbSize_); // 7k*2 + 32 + 12
    }
    uint32_t tmpTotalUB = totalUsedUB_ + BUFFER_NUM * hAlignSize_ + hOutAlignUbSize_ * BUFFER_NUM;
    bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
    // tpipe_->InitBuffer(xInQueue_, bufferNum_, hAlignSize_); // 14K * 2
    // tpipe_->InitBuffer(xOutQueue_, bufferNum_, hOutAlignUbSize_); // 7K * 2 + 32 + 6
    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();

    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf_, dynamicScalesOutGMTensor_);

    // uint32_t axisHCommu = hScaleIdxSize_ / sizeof(ExpandXOutType); // 有效搬运长度
    hCommuSize_ = blockCntPerTokenToExpert_ * SERVER_STATE_ALIGN; //拼接flag后的字节数
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);

    floatDataCopyParams_ = {1U, sizeof(float), 0U, 0U, 0U};
    xCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U};
    // hCommuCopyOutParams_ = {1U, static_cast<uint32_t>(axisHCommu * sizeof(ExpandXOutType)), 0U, 0U, 0U};
    // doWindowCopyOutParams_ = {1U, static_cast<uint32_t>(axisHCommu * sizeof(ExpandXOutType)), 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(ExpandXOutType)), 0U, 0U, 0U};
    scaleOutParams_ = {1U, static_cast<uint16_t>(scaleOutBytes_), 0U, 0U, 0U};

    uint32_t serverBuferLength = serverNum_;
    uint32_t serverMapLength = serverNum_ * axisMaxBS_;
    tpipe_->InitBuffer(serverCountBuf_, serverBuferLength * sizeof(uint32_t));
    tpipe_->InitBuffer(serverMapBuf_, serverMapLength * sizeof(uint32_t));
    serverCountTensor_ = serverCountBuf_.Get<uint32_t>();
    tokenSendMap_ = serverMapBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(serverCountTensor_, uint32_t(0), serverBuferLength);
    Duplicate<uint32_t>(tokenSendMap_, uint32_t(0), serverMapLength);

    tpipe_->InitBuffer(expertOffsetCntBuf_, expertIdsCnt_ * sizeof(uint32_t));
    expertOffsetCntTensor_ = expertOffsetCntBuf_.Get<uint32_t>();
    tpipe_->InitBuffer(xSendBuf_,sendTokenLength_);

    maxBlockCnt_ = max(max(blockCntPerToken_, blockCntPerCnt_), blockCntPerTokenToExpert_);
    LOG_INFO("[blockCntPerToken_]", blockCntPerToken_);
    LOG_INFO("[blockCntPerCnt_]", blockCntPerCnt_);
    LOG_INFO("[blockCntPerTokenToExpert_]", blockCntPerTokenToExpert_);
    LOG_INFO("[maxBlockCnt_]", maxBlockCnt_);
    tpipe_->InitBuffer(flagBuf_, maxBlockCnt_ * UB_ALIGN);
    uint64_t mask[1] = {0x0101010101010101};
    uint8_t repeatTime = static_cast<uint8_t>(Ceil(maxBlockCnt_ * UB_ALIGN, 256));
    flagTensor_ = flagBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(flagTensor_, uint32_t(1), mask, repeatTime,uint16_t(1), uint8_t(8));

    //LogInfo(__LINE__, "axisHCommu",axisHCommu);
    //LogInfo(__LINE__, "serverBuferLength",serverBuferLength);
    //LogInfo(__LINE__, "serverMapLength",serverMapLength);

    //LogInfo(__LINE__, "InitDispatchBetweenServerInfo end");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    //LogInfo(__LINE__, "Init start");
    InitRecieveTilingContext(expandXOut, workspaceGM, pipe, tilingData);
    statusDataSpaceGm_ = GetStatusDataSpaceGm(winContext_[COMM_EP_IDX]);
    selfRankWinInGMTensor_.SetGlobalBuffer((__gm__ float*)(statusDataSpaceGm_));
    //LogInfo(__LINE__, "statusDataSpaceGm_ ");
    selfDataStatusGMTensor_.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusDataSpaceGm_ + DISPATCH_STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    //LogInfo(__LINE__, "selfDataStatusGMTensor_ over");
    TBuf<> dataStateBuf;
    //LogInfo(__LINE__, "dataStateBuf over");
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    //LogInfo(__LINE__, "InitBuffer over");
    //dataState_ = 0;
    dataState_ = InitWinState(selfDataStatusGMTensor_, winContext_[COMM_EP_IDX], epRankIdOriginal_, moeExpertNum_,
                              epWorldSizeOriginal_, globalBS_, dataStateBuf);
    //LogInfo(__LINE__,"dataState_",dataState_);
    elasticInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(elasticInfo));
    //LogInfo(__LINE__, "elasticInfoGMTensor_ ");
    if (epRankId_ < sharedExpertRankNum_) {
        isShareExpertRankFlag_ = true;
    }
    //LogInfo(__LINE__, "isShareExpertRankFlag_ ");
    InitComputeInfo();

    tpGatherRankId_ = ((tpRankId_ == 0) ? 1 : 0);
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    expertScalesGMTensor_.SetGlobalBuffer((__gm__ float *)expertScales);
    // LOG_INFO("[expertScalesGMTensor_]",expertScalesGMTensor_,{axisBS_,axisK_});
    expandScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)expandScalesOut);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ uint8_t *)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOut));
    if constexpr (IsSmoothScaleExist) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ float*)scales);
    }
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut; // 无GlobalTensor
    sendTpCountOutGM_ = tpSendCountsOut;

    sendCountGMTensor_.SetGlobalBuffer((__gm__ int32_t *)workspaceGM);
    recvCntWorkspaceGM_ = workspaceGM + epWorldSize_ * expertNumInServer_ * UB_ALIGN;

    //LogInfo(__LINE__, "epWorldSize_ * expertNumInServer_ * UB_ALIGN", epWorldSize_ * expertNumInServer_ * UB_ALIGN);

    InitCommBetweenServerInfo();
    InitSetWindows(tilingData);
    InitExtraInfo();
    InitMaskInfo();
    InitDispatchBetweenServerInfo();
    //LogInfo(__LINE__, "Init end: ");
    aivUsedCumSum_ = 2; //TODO:
    aivUsedDispatch_ = (aivNum_ -aivUsedCumSum_) / 2;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
    uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;                // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t newAivId;  
    if (aivId_ < aivUsedCumSum_) {
        newAivId = aivId_;
    } else if (aivId_ < aivUsedCumSum_ + aivUsedDispatch_) {
        newAivId = aivId_ - aivUsedCumSum_;
    } else {
        newAivId = aivId_ - (aivUsedCumSum_ + aivUsedDispatch_);
    }
    startTokenId = sendTokenNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
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
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::FillQuadruple(LocalTensor<XType> &xOutTensor, uint32_t tokenIndex)
{
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    LocalTensor<float> xoutTfloat32 = xOutTensor.template ReinterpretCast<float>();
    xOutTint32(sendXTypeElemAlign_) = epRankId_;      // 源rankId
    xOutTint32(sendXTypeElemAlign_ + 1) = tokenIndex; // TokenID ,

    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(axisK_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> CopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xOutTint32[moeListAlign_], expertIdsGMTensor_[tokenIndex * axisK_], copyParams, CopyPadParams);

    for (uint32_t index = 0; index < axisK_; index++) {
        uint32_t cntPos = moeCntAlign_ + index;
        uint32_t expertPos = moeListAlign_ + index;
        uint32_t offset = tokenIndex * axisK_ + index;
        if (isExpertMaskFlag_ && !expertMaskInputTensor_(offset)) {
            xOutTint32(expertPos) = moeExpertNum_;
        }
        xOutTint32(cntPos) = expertOffsetCntTensor_(offset);
    }
    DataCopyPadExtParams<float> expertScalePadParams{false, 0U, 0U, 0U};
    DataCopyPad(xoutTfloat32[moeExpertScalesAlign_], expertScalesGMTensor_[tokenIndex * axisK_], copyParams, expertScalePadParams);
}

template <TemplateDispatchKFCTypeClass>
template <typename T>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CopyTokenToWinOut(LocalTensor<T> &xOutTensor,
                                                                       GM_ADDR rankGM, uint32_t blockCnt)
{
    GlobalTensor<T> dataDstWinGMTensor;
    GlobalTensor<uint32_t> flagDstWinGMTensor;
    dataDstWinGMTensor.SetGlobalBuffer((__gm__ T *)rankGM);
    flagDstWinGMTensor.SetGlobalBuffer((__gm__ uint32_t *)rankGM);
    DataCopyExtParams dataCopyOutParams = {static_cast<uint16_t>(blockCnt), SPLIT_BLOCK_DATA_SIZE, 0U,
                                           UB_ALIGN, 0U};
    DataCopyPad(dataDstWinGMTensor, xOutTensor, dataCopyOutParams);

    DataCopyExtParams flagCopyOutParams = {static_cast<uint16_t>(blockCnt), UB_ALIGN, 0U,
                                           SPLIT_BLOCK_DATA_SIZE, 0U};
    DataCopyPad(flagDstWinGMTensor[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)], flagTensor_, flagCopyOutParams);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SingleTokenProcess(uint32_t tokenIndex,
                                                                                               uint32_t dstServerId,
                                                                                               uint32_t cnt)
{
    LocalTensor<XType> sendTokenTensor = xSendBuf_.Get<XType>();
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopyPadParams copyPadExtParams{true, 0U, 0U, 0U};
    DataCopyPad(sendTokenTensor, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
    FillQuadruple(sendTokenTensor, tokenIndex);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    GM_ADDR rankGM = (GetSendAddrBetweenServer(COMM_EP_IDX, dstServerId) +
                                                                 SERVER_STATE_ALIGN + cnt * sendTokenLengthAlign_);
    CopyTokenToWinOut(sendTokenTensor, rankGM, blockCntPerToken_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::DispatchAndCountTokens(uint32_t startIndex,
                                                                                                   uint32_t endIndex, uint32_t serverId,
                                                                                                   bool process)
{
    uint32_t rankIDSharedGroup = epRankId_ % rankNumPerSharedExpert_; // 计算目的共享专家卡在其所在共享专家组的id
    for (uint32_t index = startIndex; index < endIndex; index++) {
        uint32_t tokenIndex = isExpertMaskFlag_ ? validBsIndexTensor_(index) : index;
        uint32_t sumExpert = sharedExpertNum_ + axisK_; // 每个Token需要发送 sharedExpertNum_ 个共享专家 和axisK_ 个moe专家
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
            if(dstServerId != serverId) {
                continue;
            }
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
    LOG_INFO("start SendToServer");
    uint32_t totalSendCnt = activeMaskBsCnt_;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(totalSendCnt, aivUsedDispatch_, startTokenId, endTokenId, sendTokenNum, true);
    LOG_INFO("startTokenId",startTokenId);
    LOG_INFO("endTokenId",endTokenId);
    LOG_INFO("sendTokenNum",sendTokenNum);
    if (startTokenId >= endTokenId || sendTokenNum == 0) {
        return ;
    }
    for (uint32_t serverIndex = 0; serverIndex < serverNum_; serverIndex++) {
        DispatchAndCountTokens(0, startTokenId, serverIndex, false);
        DispatchAndCountTokens(startTokenId, endTokenId, serverIndex, true);
    }
    // SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    // uint32_t totalSendCnt = activeMaskBsCnt_;
    // uint32_t startTokenId, endTokenId, sendTokenNum;
    // SplitToCore(totalSendCnt, aivUsedDispatch_, startTokenId, endTokenId, sendTokenNum, true);
    // if (startTokenId >= totalSendCnt || sendTokenNum == 0) {
    //     return;
    // }
    // //LogInfo(__LINE__,"totalSendCnt ",totalSendCnt);
    // //LogInfo(__LINE__,"startTokenId ",startTokenId);
    // //LogInfo(__LINE__,"endTokenId ",endTokenId);
    // //LogInfo(__LINE__,"sendTokenNum ",sendTokenNum);

    // DispatchAndCountTokens(0, startTokenId, false);
    // DispatchAndCountTokens(startTokenId, endTokenId, true);
    // SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

    if (endTokenId == totalSendCnt) {
        LOG_INFO("start endTokenId");
        GlobalTensor<uint32_t> dstStateGMTensor;
        TBuf<> tempbuf;
        tpipe_->InitBuffer(tempbuf, SERVER_STATE_ALIGN);
        LocalTensor<uint32_t> outTensor = tempbuf.Get<uint32_t>();
        uint32_t flagOffset = SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t); // 前面的状态区的最后32B的第一个元素是flag位为1
        outTensor(flagOffset) = 1;
        for (uint32_t index = 0; index < serverNum_; index++) {
            dstStateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)GetSendAddrBetweenServer(COMM_EP_IDX, index));
            outTensor(0) = serverCountTensor_(index);
            DataCopy(dstStateGMTensor,outTensor,SERVER_STATE_ALIGN/sizeof(uint32_t));
        }
        //LogInfo(__LINE__,serverCountTensor_ ,tpipe_,2);
        for (uint32_t index = 0; index < serverNum_; index++) {
            dstStateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)GetSendAddrBetweenServer(COMM_EP_IDX, index));
            LOG_INFO("server cnt",dstStateGMTensor(0));
            LOG_INFO("server flag",dstStateGMTensor(flagOffset));
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ExpertOffsetCal()
{
    //LogInfo(__LINE__,"ExpertOffsetCal start");
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
        if (aivId_ < aivUsedCumSum_) {
            tokenNumToExpertTensor_(expertId) = exprtOffset + 1;
        }
    }
    //LogInfo(__LINE__,"ExpertOffsetCal end");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetServerFlag()
{
    uint32_t totalSeverCnt = serverNum_;
    uint32_t startServerNum, endServerNum, sendServerNum;
    SplitToCore(totalSeverCnt, aivUsedDispatch_, startServerNum, endServerNum, sendServerNum, true);
    if(startServerNum >= totalSeverCnt || sendServerNum == 0){
        return;
    }
    GlobalTensor<uint32_t> dstStateGMTensor;
    TBuf<> tempbuf;
    tpipe_->InitBuffer(tempbuf, UB_ALIGN);
    LocalTensor<uint32_t> outTensor = tempbuf.Get<uint32_t>();
    outTensor(0) = 1;
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    uint32_t flagOffset = SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t); // 前面的状态区的最后32B的第一个元素是flag位为1
    for (uint32_t index = startServerNum; index < endServerNum; index++) {
        dstStateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)GetSendAddrBetweenServer(COMM_EP_IDX, index));
        DataCopy(dstStateGMTensor[flagOffset],outTensor,UB_ALIGN/sizeof(uint32_t));
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::DispatchBetweenServer()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;

    SetServerFlag();
    
    if (isTokenMaskFlag_) { // 1维
        //LogInfo(__LINE__, "isTokenMaskFlag_ is true");
        TokenActiveMaskCal();
    }

    if (isExpertMaskFlag_) { // 2维
        //LogInfo(__LINE__, "isExpertMaskFlag_ is true");
        ExpertActiveMaskCal();
    }

    if (activeMaskBsCnt_ == 0) {
        //LogInfo(__LINE__,"activeMaskBsCnt_ == 0");
        return;
    }

    ExpertOffsetCal();
    SendToServer();
    // SyncAll<true>();
}

// 构建发往其他server的所有data报文
template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ConstructDataAndFlagBatchWriteInfo(
    uint32_t beginServerId, uint32_t serverNum, uint32_t aivNum)
{
    // 计算当前core要处理的server
    uint32_t startServerId, endServerId, batchWriteItemNum;

    SplitToCore(serverNum, aivNum, startServerId, endServerId, batchWriteItemNum, true);
    ////LogInfo(__LINE__, "[BATCHWRITE][ConstructDataAndFlagBatchWriteInfo] batchWriteItemNum:", batchWriteItemNum);

    if (startServerId >= serverNum || batchWriteItemNum == 0) {
        return;
    }
    startServerId += beginServerId;
    endServerId += beginServerId;
    // //LogInfo(__LINE__, "[BATCHWRITE][ConstructDataAndFlagBatchWriteInfo] startServerID:", startServerId);
    // //LogInfo(__LINE__, "[BATCHWRITE][ConstructDataAndFlagBatchWriteInfo] endServerID:", endServerId);
    // 当前aiv负责 [startServerId,endServerId) 个 server
    for (uint32_t dstServerInd = startServerId; dstServerInd < endServerId; ++dstServerInd) {
        uint32_t dstServerId = epRankId_ % serverRankSize_ + dstServerInd * serverRankSize_; // 目标Server

        uint64_t dstDataAddr = (uint64_t)(GetReceiveAddrBetweenServer(COMM_EP_IDX, dstServerId));
        // src卡GetWindowsInAddr地址, 要发给serverIndex
        uint64_t srcDataAddr = (uint64_t)(GetSendAddrBetweenServer(COMM_EP_IDX, serverId_));
        // 去往该Server的传输的数据量
        uint32_t validTokenCount = serverCountTensor_(dstServerInd);
        uint32_t validDataLength = SERVER_STATE_ALIGN + validTokenCount * sendTokenLengthAlign_;

        batchWriteU64Tensor_(0) = srcDataAddr;     // 源地址
        batchWriteU64Tensor_(1) = dstDataAddr;     // 目的地址
        batchWriteU64Tensor_(2) = validDataLength; // 数据长度
        batchWriteU32Tensor_(6) = HcclDataType::HCCL_DATA_TYPE_INT8;
        batchWriteU32Tensor_(7) = dstServerId; // dst卡
        // //LogInfo(__LINE__,"srcDataAddr",srcDataAddr);
        // //LogInfo(__LINE__,"dstDataAddr",dstDataAddr);
        // //LogInfo(__LINE__,"validDataLength",validDataLength);
        // //LogInfo(__LINE__,"dstServerId",dstServerId);

        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t dstServerOffset = dstServerInd;
        uint32_t sendInfoCount = B64_PER_BLOCK; // 只发送一个结构
        ////LogInfo(__LINE__, "[BATCHWRITE][ConstructDataAndFlagBatchWriteInfo] batchWriteU64Tensor_:");
        ////LogInfo(__LINE__,  batchWriteU64Tensor_, tpipe_, 9);
        DataCopy(dataBatchWriteInfoTensor_[dstServerOffset * sendInfoCount], batchWriteU64Tensor_, sendInfoCount);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitWindow(uint32_t aivNum)
{
    LogInfo(__LINE__, "[BATCHWRITE][WaitWindow] enter WaitWindow.");

    uint32_t startServerId, endServerId, serverCnt;
    SplitToCore(serverNum_, aivNum, startServerId, endServerId, serverCnt, true);
    if (startServerId >= serverNum_ || serverCnt == 0) {
        return;
    }
    LogInfo(__LINE__, "[BATCHWRITE][WaitWindow] startServerId:", startServerId);
    LogInfo(__LINE__, "[BATCHWRITE][WaitWindow] endServerId:", endServerId);
    LogInfo(__LINE__, "[BATCHWRITE][WaitWindow] serverCnt:", serverCnt);

    
    for (uint32_t dstServerInd = startServerId; dstServerInd < endServerId; ++dstServerInd) {
        uint32_t tokenCnt;
        ////LogInfo(__LINE__, "[BATCHWRITE][WaitWindow] curServerId:", dstServerInd);
        WaitStatusFlag(dstServerInd, tokenCnt);
        //LogInfo(__LINE__, "[BATCHWRITE][WaitWindow] tokenCnt:", tokenCnt);
        // buf分配
        //tpipe_->Reset(); //TODO 

        tpipe_->Reset();
        InitTemp();

        uint32_t flagMaxRecvNum = (blockCntPerToken_ * UB_ALIGN) / sizeof(uint32_t);
        uint32_t gatherOutSize =
        Ceil(blockCntPerToken_ * sizeof(uint32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256; // 256对齐
        uint32_t tBufRealSize_ = UB_ALIGN >(gatherOutSize + blockCntPerToken_ * UB_ALIGN) ?UB_ALIGN: (gatherOutSize + blockCntPerToken_ * UB_ALIGN);
        TBuf<> flagMaskBuf, statusCleanBuf, finishNumBuf, tBuf, tmpBuf;
        tpipe_->InitBuffer(tBuf, tBufRealSize_); // 其余buffer空间统一申请
        tpipe_->InitBuffer(flagMaskBuf, BUFFER_NUM * UB_ALIGN);
        tpipe_->InitBuffer(statusCleanBuf, BUFFER_NUM * UB_ALIGN);
        tpipe_->InitBuffer(finishNumBuf, Ceil(512 * sizeof(uint32_t),UB_ALIGN)* UB_ALIGN);
        tpipe_->InitBuffer(tmpBuf, sendTokenLengthAlign_);
        tokenData_ = xInQueue_.AllocTensor<XType>();

        flagGatherOutTensor_ = tBuf.GetWithOffset<uint32_t>(gatherOutSize / sizeof(uint32_t), 0); // buf复用
        flagRecvTensor_ = tBuf.GetWithOffset<uint32_t>(flagMaxRecvNum, gatherOutSize);            // buf复用

        flagCompResultU8_ = flagMaskBuf.Get<uint8_t>();
        flagCompResultLtU64_ = flagMaskBuf.Get<uint64_t>();
        flagRecvGatherMask_ = statusCleanBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
        finishNumTensor_ = finishNumBuf.Get<uint32_t>();

        recvTmpTensor_ = tmpBuf.Get<XType>();
        
        #if defined(__DAV_C310__)
            LocalTensor<uint8_t> singleByteTok = tokenData_.template ReinterpretCast<uint8_t>();
            if constexpr (QuantMode > UNQUANT && ((QuantMode == MX_QUANT) || (QuantMode == PERGROUP_DYNAMIC_QUANT))) {
                // 由于MX以及PERGROUP量化在计算scales时每次搬入256字节数据，所以在token搬入前需要对空间填0，避免引入脏数据
                Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XType)); 
            }
        #endif

        tpipe_->InitBuffer(localSetTbuf_, UB_ALIGN);
        localSet_ = localSetTbuf_.Get<int32_t>();

        LocalTensor<uint32_t> flagCompResultLtU32 = flagMaskBuf.Get<uint32_t>();
        Duplicate<uint32_t>(flagCompResultLtU32, 0, BUFFER_NUM * UB_ALIGN / sizeof(uint32_t));
        Duplicate<uint32_t>(flagRecvGatherMask_, 0, UB_ALIGN / sizeof(uint32_t));
        SyncFunc<AscendC::HardEvent::V_S>();
        flagRecvGatherMask_.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();

        WaitToken(tokenCnt, dstServerInd, 0, tBuf);
        xInQueue_.FreeTensor<XType>(tokenData_);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitStatusFlag(uint32_t serverIdx,
                                                                                           uint32_t &tokenCnt)
{
    LOG_INFO("[BATCHWRITE][WaitStatusFlag] enter WaitStatusFlag.");
    tpipe_->Reset();
    TBuf<> tBuf;
    LocalTensor<uint32_t> statusFlagLocal, statusCntLocal;
    GlobalTensor<uint32_t> statusCntGlobal;
    GM_ADDR wAddr = GetReceiveAddrBetweenServer(COMM_EP_IDX, serverIdx); // 对应server接收区地址
    statusCntGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(wAddr));
    uint32_t tBufRealSize_ = MAX_UB_SIZE - (BUFFER_NUM * UB_ALIGN * 3);
    tpipe_->InitBuffer(tBuf, 512*2); // todo: 只需要512*2
    statusFlagLocal = tBuf.Get<uint32_t>();
    statusCntLocal = tBuf.Get<uint32_t>();
    // int32_t count = 0;

    while (true) {

        DataCopy(statusFlagLocal, statusCntGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)],
                 (SERVER_STATE_ALIGN - SPLIT_BLOCK_DATA_SIZE) / sizeof(uint32_t));
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        LocalTensor<uint32_t> flagVal = statusFlagLocal.ReinterpretCast<uint32_t>();
        if (flagVal.GetValue(0) == 1) {
            LogInfo(__LINE__, "[BATCHWRITE][WaitStatusFlag] flag Get.");
            break;
        }
    }
    DataCopy(statusCntLocal, statusCntGlobal, SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    LocalTensor<uint32_t> cntUint32 = statusCntLocal.ReinterpretCast<uint32_t>();
    tokenCnt = cntUint32.GetValue(0);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitToken(uint32_t tokenCnt, uint32_t serverIdx,
                                                               uint32_t startTokenIdx, TBuf<> &tBuf)
{
    //LogInfo(__LINE__, "[BATCHWRITE][WaitToken] enter.");
    LocalTensor<float> xOutFp32Tensor = recvTmpTensor_.template ReinterpretCast<float>();
    LocalTensor<int32_t> xOutInt32Tensor = recvTmpTensor_.template ReinterpretCast<int32_t>();
    GlobalTensor<int32_t> cleanGlobal;
    uint32_t index = 0;
    uint32_t finishNum = 0;

    for (uint32_t idx = 0; idx < tokenCnt; idx++) {
        finishNumTensor_(idx) = 0;
    }

    uint32_t count = 0;
    while (true) {
        count++;
        if (count == 100000) {
            LogInfo(__LINE__, "[BATCHWRITE][WaitToken] error count.");
            break;
        }
        if (finishNumTensor_(index) == 1) {
            index = (index + 1) % tokenCnt; // 轮询查询每个有效的index
            continue;
        }
        LOG_INFO("[BATCHWRITE][WaitToken][true] index:", index);
        uint32_t arriveCount = 0;

        GlobalTensor<uint32_t> dataFlagGlobal;
        // token的起始地址
        GM_ADDR wAddr =
            GetReceiveAddrBetweenServer(COMM_EP_IDX, serverIdx) + SERVER_STATE_ALIGN + (index + startTokenIdx) * sendTokenLengthAlign_;
        dataFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(wAddr));
        CheckDataArriveWithFlag(dataFlagGlobal, arriveCount, blockCntPerToken_, flagRecvTensor_, flagGatherOutTensor_);

        if (arriveCount == 1) {
            uint32_t dstPosition = index;
            GM_ADDR wAddr = GetReceiveAddrBetweenServer(COMM_EP_IDX, serverIdx) + SERVER_STATE_ALIGN;
            CopyInAndOut(xOutFp32Tensor, xOutInt32Tensor, wAddr, index, dstPosition, arriveCount);
            LOG_INFO("after CopyInAndOut");

            SyncFunc<AscendC::HardEvent::MTE2_V>();
            SendToExpert(index);

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
        } else {
            index = (index + 1) % tokenCnt;
        }
        if (tokenCnt == finishNum) {
            //LogInfo(__LINE__, "[BATCHWRITE][WaitToken] [finishNum]index:", index);
            //LogInfo(__LINE__, "[BATCHWRITE][WaitToken] finishNum:", finishNum);
            break;
        }
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CheckDataArriveWithFlag(GlobalTensor<uint32_t> &dataFlagGlobal,
                                                                            uint32_t &arriveCount, uint32_t flagNum,
                                                                            LocalTensor<uint32_t> &flagRecvTensor,
                                                                            LocalTensor<uint32_t> &flagGatherOutTensor)
{
    uint64_t rsvdCnt = 0;
    uint32_t arriveFlagNum = 0;
    uint32_t compareCount =
        Ceil(flagNum, COMPARE_COUNT_PER_BLOCK) * COMPARE_COUNT_PER_BLOCK; // flagNum 向上取整到256倍数
    uint32_t compResultU64Num = Ceil(flagNum, 64);                        // 64：按照64bit位进行划分

    DataCopyExtParams expFlagCopyParams{static_cast<uint16_t>(flagNum), static_cast<uint32_t>(sizeof(uint32_t)),
                                        static_cast<uint32_t>(SERVER_STATE_ALIGN - sizeof(uint32_t)), 0, 0};
    DataCopyPadExtParams<uint32_t> expFlagPadParams{false, 0U, 0U, 0U}; // 不特殊填充

    // flag放到flagRecvTensor
    DataCopyPad(flagRecvTensor, dataFlagGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)], expFlagCopyParams, expFlagPadParams);
    LOG_INFO("dataFlagGlobal", dataFlagGlobal, {16, 8});
    LOG_INFO("flagRecvTensor", flagRecvTensor, {1, flagNum});

    SyncFunc<AscendC::HardEvent::MTE2_V>();
    GatherMask(flagGatherOutTensor, flagRecvTensor, flagRecvGatherMask_, true, uint32_t(1),
               {1, (uint16_t)(flagNum), 1, 0}, rsvdCnt);
    LOG_INFO("flagGatherOutTensor", flagGatherOutTensor, {1, 1});
    PipeBarrier<PIPE_V>();
    CompareScalar(flagCompResultU8_, flagGatherOutTensor, uint32_t(1), AscendC::CMPMODE::EQ, compareCount);
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
    LOG_INFO("arriveFlagNum", arriveFlagNum);
    if (arriveFlagNum > flagNum) {
        arriveFlagNum = flagNum;
    }
    arriveCount = uint32_t(arriveFlagNum / flagNum); // 返回token总数
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CopyInAndOut(
    LocalTensor<float> xOutFp32Tensor, LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index,
    uint32_t dstPosition, uint32_t arriveCount)
{
    ////LogInfo(__LINE__,"[CopyInAndOut] START");
    GlobalTensor<XType> dataFlagGlobal, expandXOutGlobal;
    dataFlagGlobal.SetGlobalBuffer((__gm__ XType *)(wAddr));
    expandXOutGlobal.SetGlobalBuffer((__gm__ XType *)(expandXOutGM_) + (dstPosition)*axisH_);
    // 连续512的数据块搬运
    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerToken_ * arriveCount),
                                         static_cast<uint32_t>(SPLIT_BLOCK_DATA_SIZE), static_cast<uint32_t>(SERVER_STATE_ALIGN-SPLIT_BLOCK_DATA_SIZE), 0};
    // 量化缩放因子
    DataCopyExtParams scalesCopyParams{uint16_t(arriveCount), static_cast<uint32_t>(sizeof(float)),
                                       static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1),
                                       0U, 0U};
    // 重组分发
    DataCopyExtParams tokenCopyParams{
        uint16_t(arriveCount), hOutSize_,
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams expandIdxCopyParams{
        uint16_t(arriveCount), EXPAND_IDX_INFO * sizeof(int32_t),
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};

    DataCopyPad(recvTmpTensor_, dataFlagGlobal[(index * sendTokenLengthAlign_) / sizeof(XType)],
                srcTokenCopyParams, srcTokenPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    ////LogInfo(__LINE__,"[CopyInAndOut] END");
}

// 0-31 server  32-63 recieve

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CommunicateBetweenServer(uint32_t beginServerId,
                                                                              uint32_t serverNum, uint32_t aivNum)
{
    //LogInfo(__LINE__, "[BATCHWRITE][CommunicateBetweenServer] start");
    // ConstructDataAndFlagBatchWriteInfo(beginServerId, serverNum, aivNum);
    // PipeBarrier<PIPE_ALL>();
    // if ASCEND_IS_AIV {
    //     if (aivId_ == aivUsedCumSum_) {
    //         bufferChosenGlobal_(0) = bufferId_ ^ 1;
    //         DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
    //             bufferChosenGlobal_);
    //     }
    // }
    //LogInfo(__LINE__, "[BATCHWRITE][CommunicateBetweenServer] end");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::resetMaxCnt(int32_t cntPosIndex,
                                                                                        int32_t curExpertCnt)
{
    AscendC::SetAtomicMax<int32_t>();
    localSet_(0) = curExpertCnt;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(sendCountGMTensor_[cntPosIndex * UB_ALIGN / sizeof(int32_t)], localSet_, UB_ALIGN / sizeof(int32_t));
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::FillTriple(LocalTensor<ExpandXOutType> &xOutTensor,
                                                                uint32_t srcRankIndex, uint32_t tokenIndex,
                                                                uint32_t k, float expertScale)
{
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    LocalTensor<float> xOutTfloat = xOutTensor.template ReinterpretCast<float>();
    xOutTint32(tokenQuantAlign_) = srcRankIndex;
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex;
    xOutTint32(tokenQuantAlign_ + 2) = k;
    xOutTfloat(expertScaleAlign_) = expertScale;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::IsInSameServer(uint32_t targetRankId)
{
    return targetRankId / serverRankSize_ == serverId_;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ProcessToken(
            GM_ADDR rankGM, uint32_t tokenIndex, uint32_t topKIndex,
            DataCopyPadParams& padParams, DataCopyParams& scaleInParams, uint32_t expertIndex,
            LocalTensor<XType>& tokenData, uint32_t srcRankIndex, float expertScale)
{
    if constexpr (QuantMode > UNQUANT) {
        xInQueue_.EnQue(tokenData);
        tokenData = xInQueue_.DeQue<XType>();
        xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
        quantInst_.QuantProcess(xOutTensor_, tokenData, expertIndex, scalesCount_, scalesGMTensor_); // 量化
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
        FillTriple(xOutTensor_, srcRankIndex, tokenIndex, topKIndex, expertScale);
        //DataCopyPad(outTokenGT, xOutTensor_, hCommuCopyOutParams_);
        CopyTokenToWinOut(xOutTensor_, rankGM, blockCntPerTokenToExpert_);
        xOutQueue_.FreeTensor<ExpandXOutType>(xOutTensor_);
    } else {
        xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
        DataCopy(xOutTensor_, tokenData, hAlignSize_ / sizeof(ExpandXOutType));
        #if defined(__DAV_C310__)
        if constexpr (IsSmoothScaleExist) {
            auto tmp = scalesGMTensor_.ReinterpretCast<uint8_t>();
            DataCopyPad(xOutTensor_[axisH_].template ReinterpretCast<uint8_t>(), tmp[tokenIndex * scaleInBytes_],
                scaleInParams, padParams);
        }
        #endif
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
        FillTriple(xOutTensor_, srcRankIndex, tokenIndex, topKIndex, expertScale);
        //DataCopyPad(outTokenGT, xOutTensor_, hCommuCopyOutParams_);
        CopyTokenToWinOut(xOutTensor_, rankGM, blockCntPerTokenToExpert_);
        xOutQueue_.FreeTensor<ExpandXOutType>(xOutTensor_);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendToExpert(uint32_t rcvCnt)
{
    LogInfo(__LINE__, "[SendToExpert] start");
    //LogInfo(__LINE__, "[SendToExpert] index: ",index, epRankId_);
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};

    DataCopy(tokenData_, recvTmpTensor_, hAlignSize_ / sizeof(XType));
    // PipeBarrier<PIPE_ALL>();
    
    LocalTensor<uint32_t> xInTensor = recvTmpTensor_.template ReinterpretCast<uint32_t>();
    LocalTensor<float> xInToFloatTensor = recvTmpTensor_.template ReinterpretCast<float>();
    uint32_t srcRankIndex = xInTensor(sendXTypeElemAlign_);
    uint32_t tokenIndex = xInTensor(sendXTypeElemAlign_ + 1);
    float expertScale = float(1.0);
    LogInfo(__LINE__, "[SendToExpert] srcRankIndex",srcRankIndex);
    LogInfo(__LINE__, "[SendToExpert] tokenIndex",tokenIndex);

    LogInfo(__LINE__, "[SendToExpert] do moe");

    for (int32_t topKIndex = 0; topKIndex < axisK_; topKIndex++) {                      // 发moe专家
        uint32_t dstExpertId = xInTensor(moeListAlign_ + topKIndex);                    // 取出专家id
        uint32_t dstRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_; // 发往的卡号
        uint32_t curExpertCnt = xInTensor(moeCntAlign_ + topKIndex);                    // 取出cnt
        expertScale = xInToFloatTensor(moeExpertScalesAlign_ + topKIndex); 
        LogInfo(__LINE__, "[SendToExpert][moe] topKIndex",topKIndex);
        LogInfo(__LINE__, "[SendToExpert][moe] dstExpertId",dstExpertId);
        LogInfo(__LINE__, "[SendToExpert][moe] dstRankId",dstRankId);
        LogInfo(__LINE__, "[SendToExpert][moe] curExpertCnt",curExpertCnt);
        // LOG_INFO("[SendToExpert][moe] expertScale",expertScale);
        // if(dstRankId !=0 && dstRankId != 1){
        //     //LogInfo(__LINE__,"[SendToExpert][error]",dstRankId);
        // }

        if (dstExpertId >= moeExpertNum_)
            continue; // 无效数据
        if (!IsInSameServer(dstRankId))
            continue; // 不同server
        
        ////LogInfo(__LINE__, "[SendToExpert][moe] is same server",dstRankId);
        // int32_t cntPosIndex = (shareRankNumInServer_ + dstExpertId - startMoeExpertId_) * serverNum_ + srcRankIndex % serverNum_;
        // resetMaxCnt(cntPosIndex, curExpertCnt + 1);
        //LogInfo(__LINE__,"[cntPosIndex]",cntPosIndex);
        //LogInfo(__LINE__,"[curExpertCnt]",curExpertCnt);

        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                                            (expertPerSizeOnWin_ * (srcRankIndex * moeExpertNumPerRank_ +
                                            dstExpertId % moeExpertNumPerRank_)) + hCommuSize_ * curExpertCnt); // 计算地址偏移
        
        ProcessToken(rankGM, tokenIndex, topKIndex,
                    padParams, scaleInParams, dstExpertId + sharedExpertNum_, tokenData_, srcRankIndex, expertScale);
        // PipeBarrier<PIPE_ALL>();
    }

    // //LogInfo(__LINE__, "[SendToExpert] do share");

    for (int32_t shareIndex = 0; shareIndex < sharedExpertNum_; shareIndex++) { // 发共享专家
        // 计算需要发送的所有的共享专家的卡号
        
        uint32_t rankIDSharedGroup = epRankId_ % rankNumPerSharedExpert_;
        uint32_t dstRankId = shareIndex * rankNumPerSharedExpert_ + rankIDSharedGroup;
        uint32_t curExpertCnt = rcvCnt; // cnt即tokenid
        //LogInfo(__LINE__, "[SendToExpert][share] shareIndex",shareIndex);
        //LogInfo(__LINE__, "[SendToExpert][share] rankIDSharedGroup",rankIDSharedGroup);
        //LogInfo(__LINE__, "[SendToExpert][share] dstRankId",dstRankId);
        //LogInfo(__LINE__, "[SendToExpert][share] curExpertCnt",curExpertCnt);

        if (!IsInSameServer(dstRankId))
            continue; // 不同server
        // //LogInfo(__LINE__, "[SendToExpert][share] is same server",dstRankId);

        resetMaxCnt((dstRankId - startRankId_) * serverNum_ + srcRankIndex % serverNum_, curExpertCnt + 1);

        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                                            expertPerSizeOnWin_ * srcRankIndex + hCommuSize_ * curExpertCnt);
        // //LogInfo(__LINE__, "[SendToExpert][share] rankGM offset1", srcRankIndex);
        // //LogInfo(__LINE__, "[SendToExpert][share] rankGM offset2", hAlignWinSize_ * curExpertCnt);
        // //LogInfo(__LINE__, "[SendToExpert][share] rankGM offset3", expertPerSizeOnWin_ * srcRankIndex + hAlignWinSize_ * curExpertCnt);

        ProcessToken(rankGM, tokenIndex, axisK_ + shareIndex,
                    padParams, scaleInParams, shareIndex, tokenData_, srcRankIndex, expertScale);
        //PipeBarrier<PIPE_ALL>();
    }
    //LogInfo(__LINE__, "[SendToExpert] end");
    // PipeBarrier<PIPE_ALL>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitStatusTensor()
{
    
    tpipe_->InitBuffer(statusBuf_, statusBufCntAlign_ * UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>();                  // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, statusBufCntAlign_ * 8);
    uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign_ / 8, 1, 8); // 0x3F800000是float的1
    PipeBarrier<PIPE_V>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetStatus() //框间状态区接收 + 框内状态区发送
{
    InitStatusTensor();

    GlobalTensor<uint32_t> statusWinGMTensor;
    statusWinGMTensor.SetGlobalBuffer((__gm__ uint32_t *)GetReceiveStatusAddrBetweenServer(COMM_EP_IDX, epRankId_)); //todo: 接收区地址
    uint32_t flagMaxRecvNum = (blockCntPerCnt_ * UB_ALIGN) / sizeof(uint32_t);
    uint32_t gatherOutSize =
        Ceil(blockCntPerCnt_ * sizeof(uint32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256; // 256对齐
    
    TBuf flagRecvBuf, flagGatherOutBuf, statusCleanBuf;
    tpipe_->InitBuffer(flagRecvBuf, flagMaxRecvNum);
    tpipe_->InitBuffer(flagGatherOutBuf, gatherOutSize);
    tpipe_->InitBuffer(statusCleanBuf, BUFFER_NUM * UB_ALIGN);
    flagRecvTensor_ = flagRecvBuf.Get<uint32_t>();
    flagGatherOutTensor_ = flagGatherOutBuf.Get<uint32_t>();
    flagRecvGatherMask_ = statusCleanBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    Duplicate<uint32_t>(flagRecvGatherMask_, 0, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::V_S>();
    flagRecvGatherMask_.SetValue(0, 1);

    int32_t count = 0;
    while(true) { //确保等到
        count++;
        if (count == 100000) {
            LOG_INFO("SetStatus error count");
            break;
        }
        DataCopyExtParams expFlagCopyParams{static_cast<uint16_t>(blockCntPerCnt_), static_cast<uint32_t>(sizeof(uint32_t)),
                                        static_cast<uint32_t>(SERVER_STATE_ALIGN - sizeof(uint32_t)), 0, 0};
        DataCopyPadExtParams<uint32_t> expFlagPadParams{false, 0U, 0U, 0U}; // 不特殊填充

        // flag放到flagRecvTensor_
        DataCopyPad(flagRecvTensor_, statusWinGMTensor[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)], expFlagCopyParams, expFlagPadParams);
        LOG_INFO("1111111");
        LOG_INFO("dataFlagGlobal", statusWinGMTensor, {16, 8});
        LOG_INFO("flagRecvTensor_", flagRecvTensor_, {1, blockCntPerCnt_});

        SyncFunc<AscendC::HardEvent::MTE2_V>();
        LOG_INFO("2222222");
        uint64_t recvCnt = 0;
        GatherMask(flagGatherOutTensor_, flagRecvTensor_, flagRecvGatherMask_, true, uint32_t(1),
                {1, (uint16_t)(blockCntPerCnt_), 1, 0}, recvCnt);
        LOG_INFO("flagGatherOutTensor_", flagGatherOutTensor_, {1, 1});

        if (flagGatherOutTensor_(0) == blockCntPerCnt_) {
            LOG_INFO("flagGatherOutTensor_(0)", flagGatherOutTensor_(0));
            break;
        }
    }  //todo :后续改

    

    uint32_t startPosId;
    uint32_t endPosId;
    uint32_t sendPosNum;
    SplitToCore(expertNumInServer_ * serverNum_, aivUsedCumSum_, startPosId, endPosId, sendPosNum);
    if (startPosId >= expertNumInServer_ * serverNum_) {
        return;
    }

    for (uint32_t posIndex = startPosId; posIndex < endPosId; posIndex ++) {
        uint32_t expertIndex = posIndex % expertNumInServer_;
        uint32_t srcServerIndex = posIndex / expertNumInServer_;
        uint32_t srcPos = posIndex / 120 * 128 + posIndex % 120; //todo:间隔取偏移量
        // uint32_t srcPos = posIndex;
        statusTensor_((posIndex - startPosId) * 8 + 1) = statusWinGMTensor(srcPos);
        LOG_INFO("srcPos",srcPos);
        LOG_INFO("statusWinGMTensor(srcPos)",statusWinGMTensor(srcPos));
    }
    LOG_INFO("[statusTensor_]",statusTensor_,{8,8});

    SyncFunc<AscendC::HardEvent::S_MTE3>(); 
    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t posIndex = startPosId; posIndex < endPosId; posIndex ++) {
        uint32_t expertIndex = posIndex % expertNumInServer_;
        uint32_t srcServerIndex = posIndex / expertNumInServer_;
        uint32_t srcRankId = srcServerIndex * serverRankSize_ + epRankId_ % serverRankSize_;
        uint32_t dstRankId = expertIndex; // 共享专家
        uint32_t offset = stateOffset_ * srcRankId;
        if (expertIndex >= shareRankNumInServer_) { // moe专家
            dstRankId =
                (expertIndex - shareRankNumInServer_) / moeExpertNumPerRank_ + shareRankNumInServer_;
            offset += ((expertIndex - shareRankNumInServer_) % moeExpertNumPerRank_ * epWorldSize_ * stateOffset_);
        }
        GM_ADDR rankGM =
            (__gm__ uint8_t *)GetWindStateAddrByRankId(COMM_EP_IDX, dstRankId) + offset; // 计算地址偏移
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(posIndex - startPosId) * 8], 8UL);
        LOG_INFO("[rankGM]", rankGM);
        LOG_INFO("[srcRankId]",srcRankId);
        LOG_INFO("[dstRankId]",dstRankId);
        LOG_INFO("[statusTensor_]",statusTensor_,{8,8});
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::BufferInit()
{
    uint32_t waitStatusBufSize = Ceil((recStatusNumPerCore_ * UB_ALIGN), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = epWorldSize_ * moeExpertNumPerRank_ * sizeof(float);
    uint64_t gatherMaskOutSize = (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    uint64_t sumContinueAlignSize = Ceil((aivNum_ * sizeof(float)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);           // recStatusNumPerCore_32对齐后大小  * 32B
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);                // 48 * 32B
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN);               // 48 * 32B
    tpipe_->InitBuffer(sumContinueBuf_, sumContinueAlignSize);          // 48 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);                       // 96 B
    
    uint32_t statusBufSize = rscvStatusNum_ * UB_ALIGN;
    uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t workLocalBufSize = Ceil(epWorldSize_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
    tpipe_->InitBuffer(workLocalBuf_, workLocalBufSize);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitDispatchClearStatus()
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0}; // 一次操作256字节，每8个数将首个设置为0
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8); // 8 = 256 / 32
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::GatherSumRecvCnt(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
    LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, 2);  // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;               // 源操作数每个datablock只需要处理两个元素
    SyncFunc<AscendC::HardEvent::S_V>();

    // 将当前核对应的专家 recvCnt 收集到gatherMaskOutTensor
    uint64_t recvCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
        {1, (uint16_t)recStatusNumPerCore_, 1, 0}, recvCnt);
    PipeBarrier<PIPE_V>();

    // 对当前核对应的专家recv cnt求和
    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) // 对inner要求32对齐
        * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);

    // 把当前核的所有专家recv cnt之和写到状态区
    // 每个核把sumOfRecvCnt重复写 aivUsedCumSum_ 份
    LocalTensor<float> sumCoreFP32Tensor = sumCoreBuf_.Get<float>();
    uint64_t maskArrayCount[2] = {0x0101010101010101, 0};
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8); // 8 = 256 / 32
    // 每次处理256字节，8个datablock，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFP32Tensor, sumOfRecvCnt, maskArrayCount, repeatTimes, 1, 8);
    uint64_t maskArrayFlag[2] = {0x0202020202020202, 0};
    Duplicate<float>(sumCoreFP32Tensor, static_cast<float>(1.0), maskArrayFlag, repeatTimes, 1, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)], sumCoreFP32Tensor, sumIntriParams);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId)
{
    outLocal = gatherMaskOutBuf_.Get<int32_t>();
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, static_cast<uint16_t>(aivUsedCumSum_ - 1), 0};
    LocalTensor<float> sumLocalTensor = sumLocalBuf_.Get<float>();
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> sumContinueTensor = sumContinueBuf_.Get<float>();
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);

    uint32_t mask = 2;
    uint64_t recvCnt = 0;
    uint32_t innerSumParams = Ceil(aivUsedCumSum_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, innerSumParams, aivUsedCumSum_};
    int32_t cumSumFlag = 0;
    gatherSumPattern.SetValue(0, 2);
    SyncFunc<AscendC::HardEvent::S_V>();

    // 获取状态区中每个核的recvCnt
    while (true) {
        DataCopy(sumLocalTensor, selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumIntriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(aivUsedCumSum_), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        Sum(recvCntSumOutTensor, sumContinueTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = static_cast<int32_t>(recvCntSumOutTensor.GetValue(0));
        if (cumSumFlag == aivUsedCumSum_) {
            break;
        }
    }

    // 0核前面所有核recv cnt总和是0
    if (newAivId == 0) {
        outLocal.SetValue(0, 0);
    } else {
        mask = 1;
        recvCnt = 0;
        gatherSumPattern.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(newAivId), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }
    // 清除 flag 用于下次aivUsedCumSum_软同步
    LocalTensor<float> sumCoreFp32Tensor = sumLocalBuf_.Get<float>();
    // 一次处理256字节，8个datablock
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8);
    // 64 = 256 / sizeof(float) 一次操作字节数，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFp32Tensor, static_cast<float>(0), 64, repeatTimes, 1, 8);
    DataCopyParams cleanParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumCoreFp32Tensor, cleanParams);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitDispatch()
{
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = 1;
    gatherTmpTensor.SetValue(0, 1);
    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    int count = 0;
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)], intriParams);
        LOG_INFO("get statusFp32Tensor_", statusFp32Tensor_, {8, 8});
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
        LOG_INFO("sumOfFlag:", sumOfFlag);
        count ++;
        if(count == 10000) {
            LOG_INFO("waitDispatch error count");
            break;
        }
    }
    // 清状态
    WaitDispatchClearStatus();
    GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalRecvAndSetFlag()
{
    // check flag 用于 aivUsedCumSum_ 软同步并计算 aivUsedCumSum_ 个核各自的recvCount
    LocalTensor<int32_t> outCountLocal;
    GetCumSum(outCountLocal, aivId_);
    // 计算epRecvCnt
    uint32_t preSum = outCountLocal.GetValue(0);
    uint32_t curCnt = preSum;
    statusTensor_ = waitStatusBuf_.Get<int32_t>();
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        uint32_t i = index - startStatusIndex_;
        uint32_t count = statusTensor_.GetValue(i * UB_ALIGN_DATA_COUNT + 1);
        curCnt += count;
        outCountLocal.SetValue(i, curCnt);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    GM_ADDR wAddr = (__gm__ uint8_t*)(recvCntWorkspaceGM_);
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer((__gm__ int32_t*)wAddr);
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(recStatusNumPerCore_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(sendCountsGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);
    // 复制aivNum_份
    for (uint32_t index = 0; index < aivNum_; index++) {
        DataCopyPad(workspaceGlobal[index * rscvStatusNum_ + startStatusIndex_], outCountLocal, dataCopyOutParams);
    }
    uint8_t repeatTimes = Ceil(aivNum_, 8);  // 一次处理256字节，8个datablock
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivNum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    LocalTensor<int32_t> syncOnCoreTensor = sumCoreBuf_.Get<int32_t>();
    LocalTensor<float> syncOnCoreFP32Tensor = sumCoreBuf_.Get<float>();
    // 每次处理256字节，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<int32_t>(syncOnCoreTensor, static_cast<int32_t>(1), SIZE_ALIGN_256 / sizeof(int32_t), repeatTimes, 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_FLAG_OFFSET + aivId_ * UB_ALIGN) / sizeof(float)], syncOnCoreFP32Tensor, sumIntriParams);  // 软同步
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetExpertTokenNums()
{
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    DataCopyParams totalStatusCopyParams{static_cast<uint16_t>(localExpertNum * epWorldSize_), 1, 0, 0};
    LocalTensor<float> totalStatusTensorFp32 = statusBuf_.Get<float>();
    DataCopy(totalStatusTensorFp32, windowInstatusFp32Tensor_, totalStatusCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    int64_t expertTokenNumCumsum = 0;
    LocalTensor<int64_t> expertTokenNumsLocalTensor = tokenNumBuf_.Get<int64_t>();
    LocalTensor<float> expertTokenNumTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), 0);
    LocalTensor<float> workLocalTensor = workLocalBuf_.Get<float>();

    for (uint32_t localExpertIdx = 0; localExpertIdx < localExpertNum; ++localExpertIdx) {
        LocalTensor<float> expertStatusTensor = statusBuf_.GetWithOffset<float>(
            epWorldSize_ * UB_ALIGN / static_cast<uint32_t>(sizeof(float)), localExpertIdx * epWorldSize_ * UB_ALIGN);
        uint32_t mask = 2;
        SyncFunc<AscendC::HardEvent::S_V>();
        ReduceSum(expertTokenNumTensor, expertStatusTensor, workLocalTensor, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        int64_t expertTokenNum = static_cast<int64_t>(expertTokenNumTensor.ReinterpretCast<int32_t>().GetValue(0));
        expertTokenNumCumsum += expertTokenNum;
        if (expertTokenNumsType_ == 0) {
            expertTokenNumsLocalTensor.SetValue(localExpertIdx, expertTokenNumCumsum);
        } else {
            expertTokenNumsLocalTensor.SetValue(localExpertIdx, expertTokenNum);
        }
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams expertTokenNumsCopyParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)),
                                                0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor, expertTokenNumsCopyParams);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalCumSum()
{
    //todo：cnt计算
    //todo：cnt框间发送（调用hccl接口

    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams); // copy expertid

    SetServerFlag();
    
    if (isTokenMaskFlag_) { // 1维
        //LogInfo(__LINE__, "isTokenMaskFlag_ is true");
        TokenActiveMaskCal();
    }

    if (isExpertMaskFlag_) { // 2维
        //LogInfo(__LINE__, "isExpertMaskFlag_ is true");
        ExpertActiveMaskCal();
    }

    if (activeMaskBsCnt_ == 0) {
        //LogInfo(__LINE__,"activeMaskBsCnt_ == 0");
        return;
    }

    tokenNumToExpertTensor_ = dstExpBuf_.GetWithOffset<int32_t>(moeExpertNum_, 0);
    Duplicate(tokenNumToExpertTensor_, 0, moeExpertNum_);
    PipeBarrier<PIPE_V>();
    if (aivId_ == 0) {
        CalExpertTotalSendCnt();
    }
    // ExpertOffsetCal();

    //cnt框间接收 + 框内发送
    SetStatus();
    
    SplitToCore(rscvStatusNum_, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_, false);
    LOG_INFO("startStatusIndex_", startStatusIndex_);
    LOG_INFO("endStatusIndex_", endStatusIndex_);
    LOG_INFO("recStatusNumPerCore_", recStatusNumPerCore_);
    tpipe_->Reset();
    BufferInit();
    WaitDispatch();
    CalRecvAndSetFlag();
    // 使用newAivId为0的核进行计算
    if (aivId_ == 0) {
        SetExpertTokenNums();
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitCumSumFlag()
{
    // Check cumsum is finished
    int32_t cumSumFlag = 0;
    int32_t targetFlag = aivUsedCumSum_ * UB_ALIGN_DATA_COUNT;
    uint32_t cumSumFlagOffset = (CUMSUM_FLAG_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float);
    uint32_t innerSumParams = aivUsedCumSum_ * UB_ALIGN / sizeof(float);
    SumParams sumFlagParams{1, innerSumParams, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT};
    LocalTensor<float> statusSumOutTensor = scalarBuf_.Get<float>();

    int count = 0;
    while (true) {
        count++;
        if(count == 100000) {
            LOG_INFO("WaitCumSumFlag error count");
            break;
        }
        DataCopy(statusFp32Tensor_, selfRankWinInGMTensor_[cumSumFlagOffset], aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);
        if (cumSumFlag == targetFlag) {
            break;
        }
    }
    // Clean flag for next round
    Duplicate<float>(statusCleanFp32Tensor_, static_cast<float>(0), aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[cumSumFlagOffset], statusCleanFp32Tensor_, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum)
{
    // 获取cumSum
    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(recvCntWorkspaceGM_));
    DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    PipeBarrier<PIPE_ALL>();

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    for (uint32_t index = startId_; index < endId_; index++) { // 从sendCnt中挑选当前有发送过来的卡的token数量
        expertMapTensor_(validNum) = index;
        if (index == 0) {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index);
        } else {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index) - sendCntTensor_(index - 1);
        }
        if (expertLeftNumTensor_(validNum) != 0) {
            validNum += 1;
        } 
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CheckDataArriveWithFlagInServer(uint32_t srcExpDataIdx,
    int32_t beginIdx, int32_t copyCnt)
{
    uint64_t rsvdCnt = 0;
    uint32_t arriveFlagNum = 0;
    uint32_t flagNum = blockCntPerTokenToExpert_ * uint32_t(copyCnt);
    uint32_t compareCount = Ceil(flagNum, COMPARE_COUNT_PER_BLOCK) * COMPARE_COUNT_PER_BLOCK;
    uint32_t compResultU64Num = Ceil(flagNum, 64); // 64：按照64bit位进行划分
    DataCopyExtParams expFlagCopyParams{static_cast<uint16_t>(flagNum), static_cast<uint32_t>(sizeof(float)),
        static_cast<uint32_t>(SERVER_STATE_ALIGN - sizeof(uint32_t)), 0, 0};
    DataCopyPadExtParams<uint32_t> expFlagPadParams{false, 0U, 0U, 0U};
    GlobalTensor<uint32_t> dataFlagGlobal;
    GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcExpDataIdx * expertPerSizeOnWin_ + // 拿到第一个起始位置
        beginIdx * hCommuSize_ + SPLIT_BLOCK_DATA_SIZE;
    dataFlagGlobal.SetGlobalBuffer((__gm__ uint32_t *)(wAddr));
    DataCopyPad(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams, expFlagPadParams);
    LOG_INFO("flagRecvTensor_:", flagRecvTensor_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    GatherMask(flagGatherOutTensor_, flagRecvTensor_, flagRecvGatherMask_, true, uint32_t(1),
         {1, (uint16_t)(flagNum), 1, 0}, rsvdCnt); 
    PipeBarrier<PIPE_V>();
    CompareScalar(flagCompResultU8_, flagGatherOutTensor_, uint32_t(1), AscendC::CMPMODE::EQ, compareCount);
    SyncFunc<AscendC::HardEvent::V_S>();

    for (uint32_t i = 0; i < compResultU64Num; i++) { 
        uint64_t flagCompMask = flagCompResultLtU64_(i);
        int64_t firstValidIdx = ScalarGetSFFValue<0>(flagCompMask); // 找到0则表示数据没到
        if (firstValidIdx == -1) { // 本次数据全到
            arriveFlagNum += 64U; // 64：ScalarGetSFFValue操作单位为64bit位
        } else {
            arriveFlagNum += uint32_t(firstValidIdx);
            break;
        }
    }
    if (arriveFlagNum > flagNum) {
        arriveFlagNum = flagNum;
    }
    return uint32_t(arriveFlagNum / blockCntPerTokenToExpert_); // 返回token总数
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CopyOut(
    LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount)
{
    GlobalTensor<ExpandXOutType> dataFlagGlobal, expandXOutGlobal;
    dataFlagGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr));
    expandXOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOutGM_) + (dstPosition) * axisH_);
    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerTokenToExpert_ * arriveCount), 
        static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE), static_cast<uint16_t>(UB_ALIGN), 0};
    DataCopyExtParams scalesCopyParams{uint16_t(arriveCount), static_cast<uint32_t>(scaleOutBytes_), 
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE - scaleOutBytes_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams tokenCopyParams{uint16_t(arriveCount), hOutSize_, 
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams expandIdxCopyParams{uint16_t(arriveCount), EXPAND_IDX_INFO * sizeof(int32_t),
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyExtParams dataCopyexpandScaleParams{uint16_t(arriveCount), sizeof(float),
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};

    DataCopyPad(xTmpTensor_, dataFlagGlobal[expertFinishNumTensor_(index) * hCommuSize_ / sizeof(ExpandXOutType)],
                srcTokenCopyParams, srcTokenPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    quantInst_.CopyScalesToOut(dstPosition, scaleOutBytes_, xTmpTensor_, scalesCopyParams);
    DataCopyPad(expandXOutGlobal, xTmpTensor_, tokenCopyParams);
    DataCopyPad(expandIdxGMTensor_[dstPosition * EXPAND_IDX_INFO], xOutInt32Tensor[tokenQuantAlign_],
                expandIdxCopyParams);
    if(!isShareExpertRankFlag_) {
        LocalTensor<float> xOutFloatTensor;
        xOutFloatTensor = xOutInt32Tensor.template ReinterpretCast<float>();
        DataCopyPad(expandScalesOutGMTensor_[dstPosition], xOutFloatTensor[expertScaleAlign_],
                dataCopyexpandScaleParams); //拷贝expert专家系数
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum)
{
    LOG_INFO("in WaitAndFormatOutput");
    uint32_t index = 0;
    uint32_t finishNum = 0;
    uint32_t maxCopyTokenCnt = tBufRealSize_ / hCommuSize_;
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    uint32_t srcExpRankId, dstPosition, arriveCount, copyCnt, srcDataBlockIdx;
    uint32_t flagMaxRecvNum = (blockCntPerTokenToExpert_ * maxCopyTokenCnt * UB_ALIGN) / sizeof(uint32_t);
    uint32_t gatherOutSize = Ceil(blockCntPerTokenToExpert_ * maxCopyTokenCnt * sizeof(uint32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    GlobalTensor<int32_t> cleanGlobal;
    flagGatherOutTensor_ = tBuf.GetWithOffset<uint32_t>(gatherOutSize / sizeof(uint32_t), 0); // buf复用
    flagRecvTensor_ = tBuf.GetWithOffset<uint32_t>(flagMaxRecvNum, gatherOutSize);  // buf复用
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor_.template ReinterpretCast<int32_t>();
    LOG_INFO("WaitAndFormatOutput finishNum:", finishNum);
    int count = 0;
    while (true) {
        count ++;
        if(count > 10000) {
            LogInfo("WaitAndFormatOutput error count");
            break;
        }
        if (expertLeftNumTensor_(index) == 0) { // 当前核负责的不需要收集
            index = (index + 1) % validNum; // 轮询查询每个有效的index
            continue;
        }
        LOG_INFO("WaitAndFormatOutput index:", index);
        srcExpRankId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ? maxCopyTokenCnt : expertLeftNumTensor_(index); // 按照ub大小一次搬入多个token
        srcDataBlockIdx = srcExpRankId % epWorldSize_ * localExpertNum + srcExpRankId / epWorldSize_; // 转换成数据区的排布偏移
        arriveCount = CheckDataArriveWithFlagInServer(srcDataBlockIdx, expertFinishNumTensor_(index), copyCnt);
        LOG_INFO("arriveCount:", arriveCount);
        if (arriveCount == copyCnt) {
            dstPosition = srcExpRankId != 0 ? sendCntTensor_(srcExpRankId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);
            GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_;
            CopyOut(xOutInt32Tensor, wAddr, index, dstPosition, arriveCount);
            // finish更新并clean
            expertFinishNumTensor_(index) += arriveCount;
            expertLeftNumTensor_(index) -= arriveCount;
            if (expertLeftNumTensor_(index) == 0) {
                uint32_t cleanUpNum = expertFinishNumTensor_(index) * blockCntPerTokenToExpert_;
                DataCopyExtParams cleanUoParams = {uint16_t(cleanUpNum), sizeof(int32_t), 0U, SERVER_STATE_ALIGN - sizeof(int32_t), 0U};
                LocalTensor<int32_t> cleanTensor = tBuf.GetWithOffset<int32_t>(UB_ALIGN / sizeof(int32_t), 0); // 在0偏移位置存放比较结果
                cleanGlobal.SetGlobalBuffer((__gm__ int32_t *)(wAddr));
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                Duplicate<int32_t>(cleanTensor, 0, UB_ALIGN_DATA_COUNT);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                DataCopyPad(cleanGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(int32_t)], cleanTensor, cleanUoParams);
                finishNum++;
            }
            PipeBarrier<PIPE_ALL>();
        } else {
            index = (index + 1) % validNum;
        }
        if (validNum == finishNum) {
            break;
        }
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::LocalWindowCopy()
{
    LOG_INFO("in LocalWindowCopy");
    // 分核负责源专家数量
    tpipe_->Reset();
    TBuf<> cumSumBuf, statusWaitBuf, statusCleanBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusWaitBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(statusCleanBuf, aivUsedCumSum_ * UB_ALIGN);
    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    statusCleanFp32Tensor_ = statusCleanBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();
    SplitToCore(rscvStatusNum_, aivNum_, startId_, endId_, sendNum_, true);
    LOG_INFO("startId_", startId_);
    LOG_INFO("endId_", endId_);
    LOG_INFO("sendNum_", sendNum_);
    // 软同步
    WaitCumSumFlag();
    LOG_INFO("after WaitCumSumFlag");
    if (sendNum_ == 0) {
        LOG_INFO("sendNum_ == 0");
        return;
    }
    // 连续化
    TBuf<> expertMapBuf, expertFinishBuf, expertLeftBuf, flagMaskBuf, tBuf;
    uint32_t validNum = 0;
    uint32_t expInfoSize = Ceil(sendNum_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);
    tpipe_->InitBuffer(flagMaskBuf, BUFFER_NUM * UB_ALIGN);  // max CompareScalar
    tBufRealSize_ = MAX_UB_SIZE - (UB_ALIGN + rscvNumAlign + 2 * aivUsedCumSum_ * UB_ALIGN) -
        (expInfoSize * 3) - BUFFER_NUM * UB_ALIGN; // 3为expInfoSize大小buffer申请个数
    tpipe_->InitBuffer(tBuf, tBufRealSize_); // 其余buffer空间统一申请
    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();
    SetValidExpertInfo(expInfoSize, validNum);
    if (validNum == 0) { // 本核负责的Expert对应rank收到数据
        return;
    }
    flagCompResultU8_ = flagMaskBuf.Get<uint8_t>();
    flagCompResultLtU64_ = flagMaskBuf.Get<uint64_t>();
    flagRecvGatherMask_ = statusCleanBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    xTmpTensor_ = tBuf.Get<ExpandXOutType>();
    LocalTensor<uint32_t> flagCompResultLtU32 = flagMaskBuf.Get<uint32_t>();
    Duplicate<uint32_t>(flagCompResultLtU32, 0, BUFFER_NUM * UB_ALIGN / sizeof(uint32_t));
    Duplicate<uint32_t>(flagRecvGatherMask_, 0, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::V_S>();
    flagRecvGatherMask_.SetValue(0, 1);
    SyncFunc<AscendC::HardEvent::S_V>();
    WaitAndFormatOutput(tBuf, validNum);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::Process()
{   
    if ASCEND_IS_AIV {
        if (aivId_ < aivUsedCumSum_) { //cnt计算 + cumsum
            CalCumSum();
        } else if(aivId_ < aivUsedCumSum_ + aivUsedDispatch_) { //框间dispatch + 发送
            DispatchBetweenServer();
            // CommunicateBetweenServer(0, serverNum_, aivUsedDispatch_);
        } else { //框内接收 + 框内发送
            WaitWindow(aivNum_ - aivUsedDispatch_ - aivUsedCumSum_);
        }
        // localWindowCopy中包含reset操作，需确保前面操作完成
        PipeBarrier<PIPE_ALL>();
        LocalWindowCopy();      // 本卡上专家数据连续化，输出expandX/scales/expandIdx
    }
}

} // namespace MoeDistributeDispatchV2HostKfcImpl
#endif // MOE_DISTRIBUTE_DISPATCH_V2_H
