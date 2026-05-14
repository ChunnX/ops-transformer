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
#include "adv_api/hcomm/hcomm.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_dispatch_v2_tiling.h"
#include "../moe_distribute_v2_constant.h"
#include "../moe_distribute_dispatch_v2_quant.h"
#include "../moe_distribute_elastic.h"
#include "../moe_distribute_v2_base.h"
#include "../check_winsize.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#if __has_include("../../common/op_kernel/moe_distribute_base.h")
#include "../../common/op_kernel/moe_distribute_base.h"
#else 
#include "../../../common/op_kernel/moe_distribute_base.h"
#endif
#include "../newlog.h"

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
constexpr uint32_t MAX_UB_SIZE = 240U * 1024U;
constexpr uint32_t BW_ITEM_SIZE = 32; // batchWriteItemSize
constexpr uint32_t B64_PER_BLOCK = 4;
constexpr uint64_t SERVER_STATE_ALIGN = 512UL;
constexpr uint64_t SERVER_CNT_SIZE = 512UL;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t COMPARE_COUNT_PER_BLOCK = 256U;
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint32_t TOKEN_RECV_CNT_MAX = 512U;
constexpr uint32_t SERVER_CNT_OFFSET = 0U;
constexpr uint32_t INSERT_DATA_STRIDE = 1U;
constexpr uint32_t COMM_COPY_BLOCK = 15U;  // 480 / 32 =15
constexpr uint32_t FLAG_COPY_BLOCK = 1U;  // 32 / 32 =1
constexpr uint32_t INSERT_FLAG_STRIDE = 15U; // 480 / 32 =15
constexpr uint32_t COMM_FLAG_COPY_STRIDE = 508; // WIN_ADDR_ALIGN - sizeof(float32_t)copy flag的间隔
constexpr uint32_t EXPERT_CNT_FLAG_SPACE = 0; // 专家结果flag全部顺序存放
constexpr uint8_t UB_ALIGN_DATA_COUNT = 8U; // 8 = UB_ALIGN / sizeof(float) = UB_ALIGN / sizeof(int32_t)
constexpr uint32_t SERVER_ADDR_OFFSET_ONE = 1U;
constexpr uint32_t SERVER_ADDR_OFFSET_ZERO = 0U;
constexpr uint32_t HCCL_MAX_SERVER_SIZE = 512;
constexpr uint32_t ONE = 1U;
constexpr uint32_t TWO = 2U;

enum class CommMode
{
    INTRA_NODE = 0, // server内
    INTER_NODE = 1 // server间
};

enum class ExecutMode
{
    CUMSUM = 0, // 做cumsum发送
    COMM = 1   // 做通信发送
};

#define TemplateDispatchKFCTypeClass                                                                                           \
    typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist, bool IsNeedAllgather
#define TemplateDispatchKFCTypeFunc XType, ExpandXOutType, QuantMode, IsSmoothScaleExist, IsNeedAllgather

using namespace AscendC;
using namespace MoeDistributeV2Base;
using namespace Mc2Aclnn;

using namespace MoeDispatchLog_NEW;
#define LOG_INFO(...) //this->logger.LogInfo(__LINE__,__VA_ARGS__)
#define LOG_INIT(...) //this->logger.Init(__VA_ARGS__)

template <TemplateDispatchKFCTypeClass>
class MoeDistributeDispatchV2HostKfc {
public:
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    using XInType = typename std::conditional<
            (Std::IsSame<XType, fp4x2_e2m1_t>::value) || (Std::IsSame<XType, fp4x2_e1m2_t>::value),
            uint8_t,
            XType
        >::type;
    using XOutType = typename std::conditional<
            (Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value) || (Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value),
            uint8_t,
            ExpandXOutType
        >::type;
#else
    using XInType = XType;
    using XOutType = ExpandXOutType;
#endif
#define TemplateDispatchQuant XInType, ExpandXOutType, XOutType,  QuantMode, IsSmoothScaleExist, IsNeedAllgather
    __aicore__ inline MoeDistributeDispatchV2HostKfc(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, 
                                GM_ADDR expertScales, GM_ADDR elasticInfo,GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                GM_ADDR expandIdxOut,GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
                                GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void QuantCleanTensor();
    template <CommMode mode> __aicore__ inline void InitQuantBuffer();
    __aicore__ inline void InitScatterBuffer(uint32_t serverId);
    __aicore__ inline void SendTokenToLocalServer(uint32_t firstAivId, uint32_t usedAivNum);
    __aicore__ inline void InitLocalCopyBuffer(uint32_t processNum);
    __aicore__ inline void CleanTokenFlag(uint32_t serverId, uint32_t startTokenId, uint32_t cleanTokenNum);
    __aicore__ inline void CalServerSplitCore(uint32_t serverNum, uint32_t startAivId, uint32_t useAivNum,
                                              uint32_t &startServerId, uint32_t &endServerId, uint32_t &useAivPerServer, uint32_t &firstAivId);
    __aicore__ inline void InitCommInfo();
    __aicore__ inline void CalExpertOffset(uint32_t startTokenId, uint32_t endTokenId);
    __aicore__ inline void HcclCommSend(uint32_t dstServerId);
    template <bool isContinueWait> __aicore__ inline bool WaitFlag(GM_ADDR flagAddr, LocalTensor<float32_t> flagInLTTensor, uint32_t stride, uint32_t waitNum);
    template <bool isStoreOffset> __aicore__ inline void CalServerSendCnt();
    __aicore__ inline void WriteExpertOffset();
    __aicore__ inline void CalTokenSendExpertOffset();
    template <typename T> __aicore__ inline void ReadDataFromWindows(GM_ADDR dataAddr, uint32_t blockCnt, LocalTensor<T> &xInTensor);
    __aicore__ inline void SendExpertCnt();
    __aicore__ inline void InitGatherTokenBuffer();
    __aicore__ inline void CalExpertTotalSendCnt();
    __aicore__ inline void ScatterTokenToRemote();
    __aicore__ inline void GatherTokenFrRemote();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void InitMaskReSource();
    __aicore__ inline void TimeOutDetection();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void FillQuadruple(LocalTensor<XType> &xOutTensor, uint32_t startTokenIndex, uint32_t tokenIndex);
    template <typename T> __aicore__ inline void WriteDataToWinOut(LocalTensor<T> &xOutTensor, GM_ADDR rankGM, uint32_t blockCnt);
    template <typename T> __aicore__ inline void WriteDataToWinOut(LocalTensor<T> &xOutTensor,  LocalTensor<T> &xTempTensor,
                                                                       GM_ADDR rankGM, uint32_t blockCnt);
    __aicore__ inline void SingleTokenProcess(uint32_t startTokenIndex, uint32_t tokenIndex, uint32_t dstServerId, uint32_t cnt);
    __aicore__ inline void ScatterToken(uint32_t startIndex, uint32_t endIndex, uint32_t dstServerId);
    __aicore__ inline void WaitStatusFlag(uint32_t dstServerId, TBuf<> &serverFlagBuf, uint32_t &rcvTokenCnt);
    __aicore__ inline void WaitToken(uint32_t tokenCnt, uint32_t serverIdx, uint32_t startTokenIdx);
    __aicore__ inline void CopyInAndOut(uint32_t serverIdx, uint32_t dstPosition, uint32_t processTokenNum);
    __aicore__ inline void SetServerFlag(uint32_t dstServerId);
    __aicore__ inline void SendTokenToRemoteServer(uint32_t serverIndex, uint32_t firstAivId, uint32_t usedAivNum);
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void ProcessToken(GM_ADDR rankGM, uint32_t tokenIndex,uint32_t topKIndex,
                                        uint32_t expertIndex, uint32_t srcRankIndex, float expertScale);
    __aicore__ inline void SendToExpert(uint32_t rcvCnt);
    __aicore__ inline void resetMaxCnt(int32_t cntPosIndex, int32_t curExpertCnt);
    __aicore__ inline void SetStatus();
    __aicore__ inline void CalTokenMask();
    __aicore__ inline void InitStatusTensor();
    __aicore__ inline void InitCalCumSumBuffer();
    __aicore__ inline void InitComputeInfo();
    __aicore__ inline void InitDispatchBetweenServerInfo();
    __aicore__ inline void InitExtraInfo();
    __aicore__ inline void InitMaskInfo();
    __aicore__ inline void InitSetWindows(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void InitRecieveTilingContext(TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void WaitDispatch();
    __aicore__ inline bool IsInSameServer(uint32_t targetRankId);
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t totalCount);
    __aicore__ inline void DoWindowCopy(LocalTensor<int32_t> &outLocal);
    __aicore__ inline void UpdateTokenNumsOut();

    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t startAivId, uint32_t curUseAivNum, uint32_t &startTokenId,
                                        uint32_t &endTokenId, uint32_t &sendTokenNum);
    __aicore__ inline void FillTriple(LocalTensor<ExpandXOutType> &xOutTensor, uint32_t srcRankIndex,
                                      uint32_t tokenIndex, uint32_t k, float expertScale);
    __aicore__ inline void SyncCntOnCore(LocalTensor<float> &gatherMaskOutTensor,
                                         LocalTensor<uint32_t> &gatherTmpTensor,
                                         LocalTensor<float> &statusSumOutTensor);
    // __aicore__ inline GM_ADDR GetBaseWindOutAddrByServer(__gm__ HcclOpParam *addr, const int32_t serverId,
    //                                                      const int32_t curRankId);
    // __aicore__ inline GM_ADDR GetBaseWindInAddrByServer(__gm__ HcclOpParam *addr, const int32_t serverId,
    //                                                     const int32_t curRankId);
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
                                            LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void SetExpertTokenNums();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void SetValidExpertInfo(uint32_t startProcessId, uint32_t endProcessId,
                                              uint32_t processNum, uint32_t &validNum);
    __aicore__ inline uint32_t CheckDataArriveWithFlagInServer(uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt);
    __aicore__ inline void CopyOut(LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount);
    __aicore__ inline void WaitAndFormatOutput(uint32_t validNum);
    
    __aicore__ inline uint32_t GetServerCommHandleIndex(const int32_t dstServerId)
    {
        return dstServerId > localServerId_? dstServerId -1 : dstServerId;
    }

    __aicore__ inline GM_ADDR GetServerRemoteBaseAddr(const int32_t dstServerId)
    {
        uint32_t curServerIndex = GetServerCommHandleIndex(dstServerId);
        // __gm__ ChannelEntity* serverEntity = reinterpret_cast<__gm__ ChannelEntity*>(dpuCommHandle_[curServerIndex]);
        return reinterpret_cast<GM_ADDR>(dpuCommHandle_[curServerIndex]->remoteBufferAddr->addr);
    }

    __aicore__ inline GM_ADDR GetServerLocalBaseAddr(const int32_t dstServerId)
    {
        uint32_t curServerIndex = GetServerCommHandleIndex(dstServerId);
        // __gm__ ChannelEntity* serverEntity = reinterpret_cast<__gm__ ChannelEntity*>(dpuCommHandle_[curServerIndex]);
        return reinterpret_cast<GM_ADDR>(dpuCommHandle_[curServerIndex]->localBufferAddr->addr);
    }

    template <ExecutMode mode>
    __aicore__ inline GM_ADDR GetReadAddrFromRemoteServer(const int32_t dstServerId)
    {
        GM_ADDR baseAddr = GetServerRemoteBaseAddr(dstServerId);
        if constexpr (mode == ExecutMode::CUMSUM) {
            return baseAddr + serverWinDataSizeOffset_ + CumSumWinSizePerServer_ * SERVER_ADDR_OFFSET_ZERO;
        } else {
            return baseAddr + serverWinDataSizeOffset_ + CumSumWinSizePerServer_ * TWO + 
                   dataWinsizePerServer_ * SERVER_ADDR_OFFSET_ZERO;
        }
    }

    template <ExecutMode mode>
    __aicore__ inline GM_ADDR GetReadAddrFromLocalServer(const int32_t dstServerId)
    {
        GM_ADDR baseAddr = GetServerLocalBaseAddr(dstServerId);
        if constexpr (mode == ExecutMode::CUMSUM) {
            return baseAddr + serverWinDataSizeOffset_ + CumSumWinSizePerServer_ * SERVER_ADDR_OFFSET_ONE;
        } else {
            return baseAddr + serverWinDataSizeOffset_ + CumSumWinSizePerServer_ * TWO + 
                dataWinsizePerServer_ * SERVER_ADDR_OFFSET_ONE;
        }
    }

    template <ExecutMode mode>
    __aicore__ inline GM_ADDR GetDataWriteAddrFromRemoteServer(const int32_t dstServerId)
    {
        GM_ADDR baseAddr = GetServerRemoteBaseAddr(dstServerId);
        if constexpr (mode == ExecutMode::CUMSUM){
            return baseAddr + serverWinDataSizeOffset_ + CumSumWinSizePerServer_ * SERVER_ADDR_OFFSET_ONE;
        } else {
            return baseAddr + serverWinDataSizeOffset_ + CumSumWinSizePerServer_ * TWO + 
                   dataWinsizePerServer_ * SERVER_ADDR_OFFSET_ONE;
        }
    }

    template <ExecutMode mode>
    __aicore__ inline GM_ADDR GetDataWriteAddrFromLocalServer(const int32_t dstServerId)
    {
        GM_ADDR baseAddr = GetServerLocalBaseAddr(dstServerId);
        if constexpr (mode == ExecutMode::CUMSUM){
            return baseAddr + serverWinDataSizeOffset_ + CumSumWinSizePerServer_ * SERVER_ADDR_OFFSET_ZERO;
        } else {
            return baseAddr + serverWinDataSizeOffset_ + CumSumWinSizePerServer_ * TWO + 
                dataWinsizePerServer_ * SERVER_ADDR_OFFSET_ZERO;
        }
    }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffsetEp_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
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
    GlobalTensor<float> fpWinTpGatherOutGMTensor_;
    GlobalTensor<int32_t> winTpEpCntGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;
    GlobalTensor<uint32_t> selfhcclDataStatusTensor_;
    GlobalTensor<float> selfRankWinInGMTensor_;

    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<XType> recvTmpTensor_;
    LocalTensor<int32_t> tpTmpTensor_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<ExpandXOutType> xOutTensor_;
    LocalTensor<ExpandXOutType> insertFlagTensor_;
    LocalTensor<XType> tokenData_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> countTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    LocalTensor<uint32_t> serverCountTensor_;
    LocalTensor<int32_t> tokenSendMap_;
    LocalTensor<float32_t> flagRecvTensor_;
    LocalTensor<uint8_t> flagCompResultU8_;
    LocalTensor<uint64_t> flagCompResultLtU64_;
    LocalTensor<uint32_t> flagRecvGatherMask_;
    LocalTensor<uint32_t> tokenRcvStatus_;
    LocalTensor<int16_t> expertOffsetCntTensor_;
    LocalTensor<bool> expertMaskInputTensor_;
    LocalTensor<uint32_t> commFlagTensor_;
    LocalTensor<float> statusCleanFp32Tensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<int32_t> tokenNumToExpertTensor_;
    LocalTensor<float> expertScalesTensor_;
    LocalTensor<uint32_t> dpuInitBufferTensor_;
    LocalTensor<uint32_t> waitWriteTensor_;
    LocalTensor<uint8_t> dpuCommInitBufTensor_;

    TBuf<> expertScalesBuf_;
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
    TBuf<> elasticInfoBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> serverCountBuf_;
    TBuf<> serverMapBuf_;
    TBuf<> expertOffsetCntBuf_;
    TBuf<> xSendBuf_;
    TBuf<> flagBuf_;
    TBuf<> waitWriteBuf_;
    TBuf<> dpuCommInitBufBuf_;

    TQue<QuePosition::VECIN, 1> tokenInQueue_; // token输入队列
    TQue<QuePosition::VECIN, 1> xInQueue_;     // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;   // 量化使用，量化后的输出
    TQue<QuePosition::VECOUT, 1> xTempQueue_;   // 用于填充flag


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

    GM_ADDR gmTemp;
    GM_ADDR gmMoeCntFlg;
    GM_ADDR gmExpertOffsetCnt;
    GM_ADDR gmSendTokenFlg;
    GM_ADDR workspaceGM_;

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
    uint32_t tpWorldSize_{0};
    int32_t epRankId_{0};
    int32_t localServerId_{0};         // 当前卡所在server编号
    int32_t rankSizePerServer_{0};   // 一个server内的rank数 epWorldSize_ -> rankSizePerServer_
    int32_t epRankIdInServer_{0}; // server内局部编号 epRankId_ -> epRankIdInServer_
    int32_t epRankIdOriginal_{0};
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
    uint32_t totalCnt_;
    uint32_t lastCore_{0};
    uint32_t dataState_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t totalUsedUB_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t serverWinDataSizeOffset_{0};
    uint64_t CumSumWinSizePerServer_{0};
    uint32_t dataWinsizePerServer_{0};
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
    uint32_t blockCntExpertPerServer_{0};
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
    uint32_t aivUsedScatter_{0}; 
    uint32_t aivUsedRemotComm_{0}; // 用于专门框间通信的aiv数
    uint32_t aivUsedGather_{0};
    uint32_t statusCntAlign_{0};
    uint32_t tBufRealSize_{0};
    uint32_t perServerMapLength_{0};
    uint32_t perServerMapCnt_{0};
    uint32_t moeCntRowAlign_{0};
    uint32_t hcommBlockCnt_{0};


    __gm__ HcclOpParam *winContext_[COMM_NUM]{nullptr, nullptr};
    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    ChannelEntity* dpuCommHandle_[HCCL_MAX_SERVER_SIZE];

    DataCopyParams xCopyParams_;
    DataCopyExtParams hCommuCopyOutParams_;
    DataCopyParams doWindowCopyOutParams_;
    MoeDistributeDispatchV2Quant<TemplateDispatchQuant> quantInst_;
    AscendC::Hcomm<CommEngine::AIV, CommProtocol::ROCE> hcommHandle_;

    MoeDispatchLog_NEW::Log logger;
};

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitRecieveTilingContext(
    TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
    gmMoeCntFlg = workspaceGM_ + 1UL * 1024 * 1024 * 1024;
    gmExpertOffsetCnt = gmMoeCntFlg + 32;
    gmSendTokenFlg = gmExpertOffsetCnt + 20 * 1024 * 1024;
    gmTemp = gmSendTokenFlg + 10 * 1024 * 1024;

    tpipe_ = pipe;
    aivId_ = GetBlockIdx();

    totalWinSizeEp_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    totalWinSizeTp_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeTp);
    rankSizePerServer_ = mc2Context_->rankSizePerServer;
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
    serverNum_ = epWorldSize_ / rankSizePerServer_;
    aivUsedCumSum_ = 2; //TODO:这个需要根据实际情况来确定
    aivUsedRemotComm_ = serverNum_; // 每个核负责一个核发送
    aivUsedScatter_ = (aivNum_ -aivUsedCumSum_ - aivUsedRemotComm_) / 2;
    aivUsedGather_ = aivNum_ - aivUsedScatter_ - aivUsedRemotComm_ - aivUsedCumSum_;
    LOG_INIT(tpipe_,epRankId_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitComputeInfo()
{
    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    if (epRankId_ < sharedExpertRankNum_) {
        isShareExpertRankFlag_ = true;
    }
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;

    localServerId_ = epRankId_ / rankSizePerServer_;
    epRankIdInServer_ = epRankId_ % rankSizePerServer_;
    startRankId_ = localServerId_ * rankSizePerServer_;
    if (startRankId_ + rankSizePerServer_ <= sharedExpertRankNum_) { // 只有共享专家
        shareRankNumInServer_ = rankSizePerServer_;
        startMoeExpertId_ = 0;
    } else if (startRankId_ < sharedExpertRankNum_) { // 有共享专家也有moe专家
        shareRankNumInServer_ = sharedExpertRankNum_ - startRankId_;
        startMoeExpertId_ = 0;
    } else { // 只有moe专家
        shareRankNumInServer_ = 0;
        startMoeExpertId_ = (startRankId_ - sharedExpertRankNum_) * moeExpertNumPerRank_; // 本server的第一个moe专家的索引
    }
    expertNumInServer_ = shareRankNumInServer_ + (rankSizePerServer_ - shareRankNumInServer_) * moeExpertNumPerRank_;
    recvCntWorkspaceGM_ = workspaceGM_ + epWorldSize_ * expertNumInServer_ * UB_ALIGN;
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
}


template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitSetWindows(const MoeDistributeDispatchV2TilingData *tilingData)
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempepRankId = 0; tempepRankId < epWorldSize_; tempepRankId++) {
        OOMCheckAddrRange<ExpandXOutType>(reinterpret_cast<__gm__ ExpandXOutType *>(GetWindAddrByRankId(COMM_EP_IDX, tempepRankId)),
                                          totalWinSizeEp_);
        OOMCheckAddrRange<float>(reinterpret_cast<__gm__ float *>(GetWindStateAddrByRankId(COMM_EP_IDX, tempepRankId)), STATE_SIZE);
    }
#endif

    // 当前tpWin区划分为前后两半区，连续两次dispatch，切换半区, combine 数据区使用前面，
    // 即axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine, dispatch使用后面
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffsetEp_ =
        dataState_ * (totalWinSizeEp_ / 2) +
        axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine; // 就是分成两块，去掉combine前面的
    windowGM_ = GetWindAddrByRankId(COMM_EP_IDX, epRankIdOriginal_);
    serverWinDataSizeOffset_ = dataState_ * (totalWinSizeEp_ / 2) + (totalWinSizeEp_ / 4); // server间win区大小与server内一致
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    GlobalTensor<ExpandXOutType> winDouble;
    winDouble.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    winDouble.SetGlobalBuffer(reinterpret_cast<__gm__ ExpandXOutType *>(windowGM_));
    OOMCheckAddrRange<ExpandXOutType>(reinterpret_cast<__gm__ ExpandXOutType *>(winDouble.GetPhyAddr()), totalWinSizeEp_);
#endif
    windowInstatusFp32Tensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(statusSpaceGm_));
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitExtraInfo()
{
    //量化使用
    hOutSize_ = axisH_ * sizeof(ExpandXOutType);
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, tokenQuantAlign_, hScaleIdxSize_, scaleOutBytes_, axisH_);

    hScaleIdxSize_ = Ceil(hScaleIdxSize_, UB_ALIGN) * UB_ALIGN; 
    expertScaleAlign_ = hScaleIdxSize_ / sizeof(uint32_t);
    hScaleIdxSize_ = hScaleIdxSize_ + UB_ALIGN; //用于存放expertscale 16.06KB

    //框间使用
    sendXTypeElemAlign_ = hAlignSize_ / sizeof(uint32_t);
    uint32_t moeListSizeAlign = hAlignSize_ + UB_ALIGN; // moeExpertIdList起始偏移
    moeListAlign_ = moeListSizeAlign / sizeof(int32_t);

    uint32_t moeCntSizeAlign = moeListSizeAlign + Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // moeCntList偏移
    moeCntAlign_ = moeCntSizeAlign / sizeof(int16_t); // 注意这里的offsetCnt是int16_t

    uint32_t moeExpertScaleSizeAlign =  moeCntSizeAlign + Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN; // moe expertscales
    moeExpertScalesAlign_ = moeExpertScaleSizeAlign / sizeof(uint32_t);

    uint32_t moeNumPerServer = moeExpertNumPerRank_ * rankSizePerServer_;
    sendTokenLength_ = moeExpertScaleSizeAlign + Ceil(axisK_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    blockCntPerToken_ = Ceil(sendTokenLength_, SPLIT_BLOCK_DATA_SIZE);
    blockCntExpertPerServer_ = Ceil(moeNumPerServer * sizeof(int32_t), SPLIT_BLOCK_DATA_SIZE);
    blockCntPerTokenToExpert_ = Ceil(hScaleIdxSize_, SPLIT_BLOCK_DATA_SIZE); //框内块数
    sendTokenLengthAlign_ = blockCntPerToken_ * SERVER_STATE_ALIGN; // 17K

    hCommuSize_ = blockCntPerTokenToExpert_ * SERVER_STATE_ALIGN; //拼接flag后的字节数
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);
    hAlignWinSize_ = Ceil(hCommuSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // win区token起始地址对齐512
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hAlignWinSize_;
    hOutAlignUbSize_ = Ceil(hScaleIdxSize_, UB_ALIGN) * UB_ALIGN;
    maxBlockCnt_ = max(max(blockCntPerToken_, blockCntExpertPerServer_), blockCntPerTokenToExpert_);
    CumSumWinSizePerServer_ = blockCntExpertPerServer_ * SPLIT_BLOCK_DATA_SIZE;
    dataWinsizePerServer_ =  (totalWinSizeEp_ / 4 - CumSumWinSizePerServer_ * TWO) / TWO; //dispatch和combine共用，
    //LOG_INFO("hOutAlignUbSize_",hOutAlignUbSize_);
    //LOG_INFO("hOutAlignUbSize_",hOutAlignUbSize_);
    //LOG_INFO("hScaleIdxSize_",hScaleIdxSize_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitMaskInfo()
{
    activeMaskBsCnt_ = axisBS_;
    expertIdsCnt_ = axisBS_ * axisK_;
    
    uint32_t hFp32Size = axisH_ * sizeof(float);
    uint32_t expertIdsSize = expertIdsCnt_ * sizeof(int32_t);
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    uint32_t expertCntAlign = Ceil(moeExpertNum_ * sizeof(int32_t), SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_DATA_SIZE;
    uint32_t bsAlign256 = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t bsKAlign256 = Ceil(expertIdsCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t expertIdsBufSize = expertIdsSize > bsAlign256 ? expertIdsSize : bsAlign256;
    uint32_t expertOffsetCntSize = axisBS_ * Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN;
    uint32_t exprtflagCntSize =  Ceil((aivNum_ - aivUsedCumSum_) * sizeof(float32_t), UB_ALIGN) * UB_ALIGN;
    moeCntRowAlign_ = Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN / sizeof(int16_t);
    expertIdsSize = Ceil(expertIdsSize, UB_ALIGN) * UB_ALIGN;
    maxSize_ = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSize_ = maxSize_ > xActivateMaskSize ? maxSize_ : xActivateMaskSize;
    maxSize_ = maxSize_ > bsKAlign256 ? maxSize_ : bsKAlign256;
    maxSize_ = maxSize_ > expertCntAlign ? maxSize_ : expertCntAlign;
    maxSize_ = maxSize_ > expertOffsetCntSize ? maxSize_ : expertOffsetCntSize;
    maxSize_ = maxSize_ > exprtflagCntSize ? maxSize_ : exprtflagCntSize; // 因为计算flag时复用了buffer因此需要保证32B对齐

    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize); // BS * K * 4 = 32K
    tpipe_->InitBuffer(gatherMaskTBuf_, maxSize_); // max:32K
    tpipe_->InitBuffer(dstExpBuf_, maxSize_); // max:32K
    tpipe_->InitBuffer(subExpBuf_, maxSize_); // max:32K

    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();

    totalUsedUB_ += maxSize_;
    totalUsedUB_ += maxSize_;
    totalUsedUB_ += expertIdsSize;
    totalUsedUB_ += maxSize_;

    if (isExpertMaskFlag_) { //zeroComputeExpertNum_ TODO:零专家视乎没有加上
        tpipe_->InitBuffer(expertMaskInputBuf_, expertIdsCnt_ * sizeof(bool)); //16k
        totalUsedUB_ += expertIdsCnt_ * sizeof(bool);
        expertMaskInputTensor_ = expertMaskInputBuf_.Get<bool>();
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitDispatchBetweenServerInfo()
{
    xCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U};

    perServerMapLength_ = Ceil(axisMaxBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    perServerMapCnt_ = perServerMapLength_ / sizeof(int32_t);
    uint32_t serverBuferLength = serverNum_;
    uint32_t serverMapLength = serverNum_ * perServerMapLength_;
    uint32_t expertOffsetCntSize = axisBS_ * Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN;
    uint8_t commRepeatTime = static_cast<uint8_t>(Ceil(maxBlockCnt_ * UB_ALIGN, SIZE_ALIGN_256));
    uint32_t commFlagSize = commRepeatTime * SIZE_ALIGN_256;
    uint64_t mask[1] = {0x0101010101010101};

    tpipe_->InitBuffer(serverCountBuf_, serverBuferLength * sizeof(uint32_t));
    tpipe_->InitBuffer(serverMapBuf_, serverMapLength); // max server = 8, per server = 2K
    tpipe_->InitBuffer(expertOffsetCntBuf_, expertOffsetCntSize); // 16K
    tpipe_->InitBuffer(flagBuf_, commFlagSize); // 16K
    tpipe_->InitBuffer(waitWriteBuf_, UB_ALIGN); // 32B

    serverCountTensor_ = serverCountBuf_.Get<uint32_t>(); // 16K
    waitWriteTensor_ = waitWriteBuf_.Get<uint32_t>(); // 32B
    tokenSendMap_ = serverMapBuf_.Get<int32_t>();
    expertOffsetCntTensor_ = expertOffsetCntBuf_.Get<int16_t>();  // TODO :
    commFlagTensor_ = flagBuf_.Get<uint32_t>();

    Duplicate<uint32_t>(serverCountTensor_, uint32_t(0), serverBuferLength);
    Duplicate<int32_t>(tokenSendMap_, int32_t(-1), serverMapLength);
    Duplicate<uint32_t>(commFlagTensor_, 0x3F800000, mask, commRepeatTime, uint16_t(1), uint8_t(8)); // 0x3F800000是float的1
    Duplicate<int16_t>(expertOffsetCntTensor_, static_cast<int16_t>(0), expertIdsCnt_);
    
    //LOG_INFO("[blockCntPerToken_]", blockCntPerToken_);
    //LOG_INFO("[blockCntExpertPerServer_]", blockCntExpertPerServer_);
    LOG_INFO("[blockCntPerTokenToExpert_]", blockCntPerTokenToExpert_);
    //LOG_INFO("[maxBlockCnt_]", maxBlockCnt_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitCommInfo()
{
    uint32_t epRankIdHccl = mc2Context_->epRankId;
    statusDataSpaceGm_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankIdHccl]);

    selfRankWinInGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(statusDataSpaceGm_));
    selfDataStatusGMTensor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t *>(statusDataSpaceGm_ + DISPATCH_STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeOriginal_, epRankIdOriginal_, moeExpertNum_,
                              epWorldSizeOriginal_, globalBS_, dataStateBuf);            
    statusSpaceGm_ = GetWindStateAddrByRankId(COMM_EP_IDX, epRankIdOriginal_);
    for (uint32_t i = 0; i < serverNum_; i++) {
        dpuCommHandle_[i] = reinterpret_cast<ChannelEntity *>(mc2Context_->dpuCommHandle_[i]); // todo:需要确认
    }
    tpipe_->InitBuffer(dpuCommInitBufBuf_, SIZE_ALIGN_256);
    dpuCommInitBufTensor_ = dpuCommInitBufBuf_.Get<uint8_t>();
    hcommHandle_.Init(reinterpret_cast<__ub__ uint8_t *>(dpuCommInitBufTensor_.GetPhyAddr()),SIZE_ALIGN_256); // todo:需要确认
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    elasticInfoGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(elasticInfo));
    xGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(x));
    xActiveMaskGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(xActiveMask));
    expertScalesGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(expertScales));
    // // //LOG_INFO("[expertScalesGMTensor_]",expertScalesGMTensor_,{axisBS_,axisK_});
    expandScalesOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(expandScalesOut));
    expertIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertIds));
    dynamicScalesOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dynamicScalesOut));
    expertTokenNumsOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(expertTokenNumsOut));
    expandIdxGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expandIdxOut));
    workspaceGM_ = workspaceGM;
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut; // 无GlobalTensor
    sendTpCountOutGM_ = tpSendCountsOut;
    mc2Context_ = (__gm__ Mc2MoeContext *)mc2Context;
    if constexpr (IsSmoothScaleExist) {
        scalesGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scales));
    }
    InitRecieveTilingContext(pipe, tilingData);
    InitCommInfo();
    InitComputeInfo();
    InitSetWindows(tilingData);
    InitExtraInfo();
    InitMaskInfo();
    InitDispatchBetweenServerInfo();
}


template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t startAivId, uint32_t curUseAivNum, uint32_t &startTokenId,
    uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t relativeAivId = aivId_ - startAivId;
    if(relativeAivId < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId = relativeAivId * sendTokenNum;
    } else {
        startTokenId = sendTokenNum * relativeAivId + remainderTokenNum;
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


/**
 * @brief ExpertActiveMaskCal 
 * @details 当isExpertMaskFlag_为true时，将所有不需要发送的专家ID全部设置为moeExpertNum，
 *          处理完之后后续计算不需要考虑二维Mask影响
 */
template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ExpertActiveMaskCal()
{
    //PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    ////LOG_INFO("ExpertActiveMaskCal expertMaskInputTensor_",expertMaskInputTensor_,{axisBS_, axisK_});
    LocalTensor<int8_t> maskStrideInt8Tensor = expertMaskInputTensor_.ReinterpretCast<int8_t>();
    LocalTensor<half> maskStrideFp16Tensor = subExpBuf_.Get<half>();
    LocalTensor<int32_t> maskStrideInt32Tensor = gatherMaskTBuf_.Get<int32_t>();
    Adds(expertIdsTensor_, expertIdsTensor_, moeExpertNum_, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    Cast(maskStrideFp16Tensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    ////LOG_INFO("ExpertActiveMaskCal maskStrideFp16Tensor",maskStrideFp16Tensor,{axisBS_, axisK_});
    Cast(maskStrideInt32Tensor, maskStrideFp16Tensor, RoundMode::CAST_RINT, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    ////LOG_INFO("ExpertActiveMaskCal maskStrideInt32Tensor",maskStrideInt32Tensor,{axisBS_, axisK_});
    Mul(expertIdsTensor_, expertIdsTensor_, maskStrideInt32Tensor, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    Adds(expertIdsTensor_, expertIdsTensor_, -moeExpertNum_, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    Abs(expertIdsTensor_, expertIdsTensor_, expertIdsCnt_);
    PipeBarrier<PIPE_ALL>();
    ////LOG_INFO("ExpertActiveMaskCal expertIdsTensor_",expertIdsTensor_,{axisBS_, axisK_});
}

/**
 * @brief InitMaskReSource 
 * @details 初始化一维mask和二维mask计算所需的资源
 */
template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitMaskReSource()
{
    if (isExpertMaskFlag_) {
        DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
        DataCopyPad(expertMaskInputTensor_, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    }

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams); // copy expertid
    SyncFunc<AscendC::HardEvent::MTE2_V>(); // 后续全部都需要依赖此数据，在这里全部一起等待
}

/**
 * @brief CalTokenMask 
 * @details 当isTokenMaskFlag_或者isExpertMaskFlag_为true时，计算需要发送多少Token，其中当二维为true时
 *          只将不发送的专家ID设置值为moeExpertNum_，不计算具体的Token数
 */
template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalTokenMask()
{
    activeMaskBsCnt_ = axisBS_;
    InitMaskReSource();
    if (isTokenMaskFlag_) { // 1维
        TokenActiveMaskCal();
    }

    if (isExpertMaskFlag_) { // 2维
        ExpertActiveMaskCal();
    }
}


template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::FillQuadruple(LocalTensor<XType> &xOutTensor, uint32_t startTokenIndex, uint32_t tokenIndex)
{
    uint32_t realIndex = tokenIndex - startTokenIndex;
    // //LOG_INFO("FillQuadruple realIndex:", realIndex);
    // //LOG_INFO("FillQuadruple startTokenIndex:", startTokenIndex);
    // //LOG_INFO("FillQuadruple tokenIndex:", tokenIndex);
    uint64_t mask = axisK_;
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    LocalTensor<int16_t> xOutTint16 = xOutTensor.template ReinterpretCast<int16_t>();
    LocalTensor<float> xoutTfloat32 = xOutTensor.template ReinterpretCast<float>();
    xOutTint32(sendXTypeElemAlign_) = epRankId_;      // 源rankId
    xOutTint32(sendXTypeElemAlign_ + 1) = tokenIndex; // TokenID ,
    
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(axisK_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> CopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xOutTint32[moeListAlign_], expertIdsGMTensor_[tokenIndex * axisK_], copyParams, CopyPadParams);
    PipeBarrier<PIPE_V>();
    DataCopy(xOutTint16[moeCntAlign_], expertOffsetCntTensor_[realIndex * moeCntRowAlign_], moeCntRowAlign_);
    DataCopyPadExtParams<float> expertScalePadParams{false, 0U, 0U, 0U};
    DataCopyPad(xoutTfloat32[moeExpertScalesAlign_], expertScalesGMTensor_[tokenIndex * axisK_], copyParams, expertScalePadParams);

    // for(uint32_t index =0; index < axisK_; index++) {
    //     //LOG_INFO("index: ", index);
    //     //LOG_INFO("xOutTint16: ", xOutTint16(index + moeCntAlign_));
    // }

    if(isExpertMaskFlag_) {
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t index = 0; index < axisK_; index++) {
            uint32_t expertPos = moeListAlign_ + index;
            uint32_t offset = tokenIndex * axisK_ + index;
            if (!expertMaskInputTensor_(offset)) {
                xOutTint32(expertPos) = moeExpertNum_;
            }
        }
    }

}

template <TemplateDispatchKFCTypeClass>
template <typename T>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WriteDataToWinOut(LocalTensor<T> &xOutTensor,
                                                                       GM_ADDR rankGM, uint32_t blockCnt)
{
    GlobalTensor<T> dataDstWinGMTensor;
    GlobalTensor<uint32_t> flagDstWinGMTensor;
    dataDstWinGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rankGM));
    flagDstWinGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(rankGM));
    DataCopyExtParams dataCopyOutParams = {static_cast<uint16_t>(blockCnt), SPLIT_BLOCK_DATA_SIZE, 0U,
                                           UB_ALIGN, 0U};
    DataCopyPad(dataDstWinGMTensor, xOutTensor, dataCopyOutParams);
    PipeBarrier<PIPE_MTE3>(); // TODO : 等待HCCL接口(后续抽成模版)
    DataCopyExtParams flagCopyOutParams = {static_cast<uint16_t>(blockCnt), UB_ALIGN, 0U,
                                           SPLIT_BLOCK_DATA_SIZE, 0U};
    DataCopyPad(flagDstWinGMTensor[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)], commFlagTensor_, flagCopyOutParams);
}


template <TemplateDispatchKFCTypeClass>
template <typename T>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WriteDataToWinOut(LocalTensor<T> &xOutTensor,  LocalTensor<T> &xTempTensor,
                                                                       GM_ADDR rankGM, uint32_t blockCnt)
{
    DataCopyParams insertDataParams = {static_cast<uint16_t>(blockCnt), COMM_COPY_BLOCK, 0U, INSERT_DATA_STRIDE};
    DataCopyParams insertFlagParams = {static_cast<uint16_t>(blockCnt), FLAG_COPY_BLOCK, 0U, INSERT_FLAG_STRIDE};
    LocalTensor<uint32_t> xFlagTensor = xTempTensor.template ReinterpretCast<uint32_t>();
    DataCopy(xTempTensor, xOutTensor, insertDataParams);
    DataCopy(xFlagTensor[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)], commFlagTensor_, insertFlagParams);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    uint32_t copyCount = blockCnt * SERVER_STATE_ALIGN / sizeof(T);
    GlobalTensor<T> dataDstWinGMTensor;
    dataDstWinGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rankGM));
    DataCopy(dataDstWinGMTensor, xTempTensor, copyCount);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SingleTokenProcess(uint32_t startTokenIndex, uint32_t tokenIndex,
                                                                                               uint32_t dstServerId,
                                                                                               uint32_t cnt)
{
    LocalTensor<XType> sendTokenTensor = xSendBuf_.Get<XType>();
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopyPadParams copyPadExtParams{true, 0U, 0U, 0U};
    DataCopyPad(sendTokenTensor, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
    FillQuadruple(sendTokenTensor, startTokenIndex, tokenIndex);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    PipeBarrier<PIPE_V>();
    GM_ADDR rankGM = (GetDataWriteAddrFromLocalServer(dstServerId) +
                                                                 SERVER_STATE_ALIGN + cnt * sendTokenLengthAlign_);
    WriteDataToWinOut(sendTokenTensor, rankGM, blockCntPerToken_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ScatterToken(uint32_t startIndex, uint32_t endIndex, uint32_t dstServerId)
{
    for(uint32_t tokenIndex = startIndex; tokenIndex < endIndex; tokenIndex++) {
        uint32_t offset = dstServerId * perServerMapCnt_ + tokenIndex;
        int32_t tokenSendOffset = tokenSendMap_.GetValue(offset);
        ////LOG_INFO("ScatterToken tokenSendOffset ",tokenSendOffset);
        if(tokenSendOffset == -1) {
            ////LOG_INFO("ScatterToken not send");
            continue;
        }
        SingleTokenProcess(startIndex, tokenIndex, dstServerId, tokenSendOffset); // 将数据写入对应的发送区
    }
}

// template <TemplateDispatchKFCTypeClass>
// __aicore__ inline bool MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::HasSharedExpertInServer(uint32_t serverId)
// {
//         uint32_t groupId = epRankId_ % sharedExpertNum_;
//         for (uint32_t sharedIndex = 0; sharedIndex < sharedExpertNum_; ++sharedIndex) {
//             uint32_t dstSharedExpertRankId = groupId + sharedIndex * rankNumPerSharedExpert_;
//             uint32_t dstServerId = dstSharedExpertRankId / rankSizePerServer_;
//             if(dstServerId == serverId) {
//                 return true;
//             }
//         }
//         return false;
// }

template <TemplateDispatchKFCTypeClass>
template <bool isContinueWait>
__aicore__ inline bool MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitFlag(GM_ADDR flagAddr, LocalTensor<float32_t> flagInLTTensor,
                                                                                             uint32_t stride, uint32_t waitNum) 
{
    // LOG_INFO("WaitFlag waitNum", waitNum);
    // LOG_INFO("WaitFlag stride", stride);
    uint32_t copyNum = 1; // 如果stride为0，表示所有数据全部在一起，直接一起拷贝进来
    uint32_t copyBlockLength = waitNum * sizeof(float32_t);
    uint32_t calCnt = waitNum;

    if (stride != 0) {
        copyNum = waitNum;
        copyBlockLength = sizeof(float32_t);
        calCnt = waitNum * UB_ALIGN / sizeof(float32_t); // 每次拷贝会自动填充满足32B对齐
    }

    GlobalTensor<float32_t> flagInGMTensor;
    flagInGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(flagAddr));
    DataCopyExtParams flagCopyParams{static_cast<uint16_t>(copyNum), copyBlockLength, static_cast<uint32_t>(stride), 0, 0};
    DataCopyPadExtParams<float32_t> flagPadParams{true, 0U, 0U, 0U}; // 全部填充为0
    uint32_t rcvCnt = 0;
    uint64_t count =0; //TODO:
    do{
        count++;
        if(count > 1000000){
            PRINTF("error WaitFlag,error count\n");
            break;
        }
        DataCopyPad(flagInLTTensor, flagInGMTensor, flagCopyParams, flagPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ////LOG_INFO("WaitFlag flagInLTTensor", flagInLTTensor, {1, waitNum});
        ReduceSum(flagInLTTensor, flagInLTTensor, flagInLTTensor, calCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        rcvCnt = static_cast<uint32_t>(flagInLTTensor.GetValue(0));
        if constexpr(isContinueWait) {
            if(rcvCnt == waitNum) { // 循环等待直到全部到达
                //LOG_INFO("WaitFlag success");
                break;
            }
        } 

    }while(isContinueWait);

    LOG_INFO("WaitFlag flagInLTTensor.GetValue(0)",flagInLTTensor.GetValue(0));
    return rcvCnt == waitNum;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::HcclCommSend(uint32_t dstServerId) 
{
    uint32_t serverTokenNum = serverCountTensor_(dstServerId);
    uint32_t tokenIndex = 0;
    uint32_t completeTokenNum = 0;
    LocalTensor<uint32_t> tokenSendMapTensor = dstExpBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(tokenSendMapTensor, static_cast<uint32_t>(0), serverTokenNum);

    while (true) {
        if (tokenSendMapTensor.GetValue(tokenIndex) != 0) {
            tokenIndex = (tokenIndex + 1) % serverTokenNum;
            continue;
        }
        GM_ADDR localReadGM = reinterpret_cast<GM_ADDR>((GetDataWriteAddrFromLocalServer<ExecutMode::COMM>(dstServerId)) + SERVER_STATE_ALIGN + tokenIndex * sendTokenLength_);
        GM_ADDR remoteWriteGM = reinterpret_cast<GM_ADDR>((GetDataWriteAddrFromRemoteServer<ExecutMode::COMM>(dstServerId)) + SERVER_STATE_ALIGN + tokenIndex * sendTokenLength_);
        GM_ADDR flagAddr = reinterpret_cast<GM_ADDR>(localReadGM + SPLIT_BLOCK_DATA_SIZE);
        bool isSendComplete = WaitFlag<false>(flagAddr, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, blockCntPerToken_);
        if (isSendComplete) {
            hcommHandle_.WriteNbi(dpuCommHandle_[GetServerCommHandleIndex(dstServerId)], remoteWriteGM, localReadGM, SERVER_STATE_ALIGN);
            tokenSendMapTensor.SetValue(tokenIndex, 1);
            tokenIndex = (tokenIndex + 1) % serverTokenNum;
            completeTokenNum++;
            if(completeTokenNum == serverTokenNum) {
                break;
            }
        }
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalExpertOffset(uint32_t startTokenId, uint32_t endTokenId)
{
    uint32_t offsetPerMask = Ceil(expertIdsCnt_ * sizeof(uint16_t), UB_ALIGN) * UB_ALIGN;
    uint64_t dstFlag = (aivNum_ - aivUsedCumSum_) * offsetPerMask;
    uint64_t usedCoreNum =  (aivNum_ - aivUsedCumSum_) > moeExpertNum_? moeExpertNum_ : (aivNum_ - aivUsedCumSum_);
    uint64_t dstOffset = startTokenId * axisK_ * sizeof(int16_t);
    uint32_t computeRow = endTokenId - startTokenId;
    uint32_t expertOffsetNum = computeRow * moeCntRowAlign_; //32B对齐
    ////LOG_INFO("expertOffsetNum",expertOffsetNum);
    GM_ADDR flagAddr = reinterpret_cast<GM_ADDR>(gmExpertOffsetCnt + dstFlag);
    LocalTensor<float32_t> flagInLTTensor = dstExpBuf_.Get<float32_t>();
    LocalTensor<int16_t> expertCntTensor = dstExpBuf_.Get<int16_t>();
    GlobalTensor<int16_t> expertOffsetCntGMTensor;
    DataCopyParams expertCntCopyInParams{static_cast<uint16_t>(computeRow), static_cast<uint16_t>(sizeof(int16_t) * axisK_), 0, 0};
    DataCopyPadParams expertCntPadParams{false, 0, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    Duplicate<int16_t>(expertOffsetCntTensor_, static_cast<int16_t>(0), expertOffsetNum);
    ////LOG_INFO("CalExpertOffset expertOffsetCntTensor_", expertOffsetCntTensor_, {axisBS_, moeCntRowAlign_});
    ////LOG_INFO("CalExpertOffset start waitFlg");
    // WaitFlag(flagAddr, flagInLTTensor,  usedCoreNum);
    WaitFlag<true>(flagAddr, flagInLTTensor, EXPERT_CNT_FLAG_SPACE, usedCoreNum);
    ////LOG_INFO("CalExpertOffset end waitFlg");
    ////LOG_INFO("usedCoreNum",usedCoreNum);
    PipeBarrier<PIPE_V>();
    for(uint32_t index =0 ;index < usedCoreNum; index++) {
        // 计算每个核心的专家偏移量
        uint64_t expertOffset = dstOffset + index * offsetPerMask;
        expertOffsetCntGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(gmExpertOffsetCnt + expertOffset));
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(expertCntTensor, expertOffsetCntGMTensor, expertCntCopyInParams, expertCntPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        // expertOffsetCntGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(gmExpertOffsetCnt + index * offsetPerMask));
        // uint32_t t= expertOffset - dstOffset;
        // //LOG_INFO("CalExpertOffset index",index);
        // //LOG_INFO("CalExpertOffset expertOffset",t);
        // //LOG_INFO("CalExpertOffset expertOffsetCntGMTensor",expertOffsetCntGMTensor, {axisBS_, axisK_});
        // //LOG_INFO("CalExpertOffset expertCntTensor", expertCntTensor, {axisBS_, moeCntRowAlign_});
        Add(expertOffsetCntTensor_, expertOffsetCntTensor_, expertCntTensor, expertOffsetNum);
        ////LOG_INFO("CalExpertOffset for expertOffsetCntTensor_", expertOffsetCntTensor_, {axisBS_, moeCntRowAlign_});
    }
    ////LOG_INFO("CalExpertOffset expertOffsetCntTensor_", expertOffsetCntTensor_, {axisBS_, moeCntRowAlign_});
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetServerFlag(uint32_t dstServerId)
{
    LOG_INFO("SetServerFlag start"); // todo:默认核数大于serverNum_，需要处理
    GlobalTensor<uint32_t> dstStateGMTensor;
    TBuf<> tempbuf;
    tpipe_->InitBuffer(tempbuf, SERVER_STATE_ALIGN);
    LocalTensor<uint32_t> outTensor = tempbuf.Get<uint32_t>();
    uint32_t flagOffset = SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t); // 前面的状态区的最后32B的第一个元素是flag位为1
    GM_ADDR localWriteGM = reinterpret_cast<GM_ADDR>((GetDataWriteAddrFromLocalServer<ExecutMode::COMM>(dstServerId)));
    GM_ADDR remoteWriteGM = reinterpret_cast<GM_ADDR>((GetDataWriteAddrFromRemoteServer<ExecutMode::COMM>(dstServerId)));
    outTensor(flagOffset) = 1;
    SyncFunc<AscendC::HardEvent::V_S>();
    dstStateGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(localWriteGM));
    outTensor(0) = serverCountTensor_(dstServerId);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    //LOG_INFO("SetServerFlag outTensor",outTensor,{16,8});
    DataCopy(dstStateGMTensor, outTensor, SERVER_STATE_ALIGN/sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    waitWriteTensor_.SetValue(0, 1); // TODO：目前临时使用这种方式来等待写入完成，后续需要优化为其他方式
    hcommHandle_.WriteNbi(dpuCommHandle_[GetServerCommHandleIndex(dstServerId)], remoteWriteGM, localWriteGM, SERVER_STATE_ALIGN); 
    LOG_INFO("SetServerFlag end");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitScatterBuffer(uint32_t serverId)
{
    if (serverId == localServerId_) {
        InitQuantBuffer<CommMode::INTRA_NODE>(); // 初始化量化参数
        tpipe_->InitBuffer(xInQueue_, 1, hAlignSize_); // 17.5K 
        tpipe_->InitBuffer(xOutQueue_, 1, hOutAlignUbSize_); //16.03K
        tpipe_->InitBuffer(expertScalesBuf_, axisK_ * sizeof(float)); //16 * 4
        tpipe_->InitBuffer(xTempQueue_, 1, hCommuSize_); // 17.5KB
        tokenData_ = xInQueue_.AllocTensor<XType>();
        insertFlagTensor_ = xTempQueue_.AllocTensor<ExpandXOutType>();
        expertScalesTensor_ = expertScalesBuf_.Get<float>();
    } else {
        tpipe_->InitBuffer(xSendBuf_,sendTokenLength_); //17K
    }
}

template <TemplateDispatchKFCTypeClass>
template <bool isStoreOffset>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalServerSendCnt()
{
    uint32_t calCnt = isTokenMaskFlag_? activeMaskBsCnt_ * axisK_ : expertIdsCnt_; 
    uint32_t activeMaskBsCnt = static_cast<uint32_t>(activeMaskBsCnt_);
    uint32_t sharedBufferXor = SIZE_ALIGN_256; //TODO:需要重新计算这个
    uint32_t compareCount = Ceil(activeMaskBsCnt * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    uint32_t maskSize = Ceil(compareCount / BITS_PER_BYTE, UB_ALIGN) * UB_ALIGN; // 为了匹配CompareScalar接口1bit一共8位
    uint64_t sendTokenCnt = 0;
    TBuf<> sharedTmpBuf;
    tpipe_->InitBuffer(sharedTmpBuf, sharedBufferXor); // TODO: 需要注意 ，可以只选256

    LocalTensor<float32_t> dstServerIdFp32 = dstExpBuf_.Get<float32_t>();
    LocalTensor<float32_t> tempTensorFp32 = subExpBuf_.Get<float32_t>();
    LocalTensor<half> tempTensorFp16 = subExpBuf_.Get<half>();
    LocalTensor<float32_t> gTempTensorFp32 = gatherMaskTBuf_.Get<float32_t>();
    LocalTensor<int32_t> dstServerIdInt32 = dstExpBuf_.Get<int32_t>();
    LocalTensor<int16_t> dstServerIdInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensorInt16 = subExpBuf_.Get<int16_t>(); 
    LocalTensor<int16_t> compareInt16 = gatherMaskTBuf_.Get<int16_t>();
    LocalTensor<uint8_t> sharedTensorInt8 = sharedTmpBuf.Get<uint8_t>();
    LocalTensor<int32_t> indexTensor = subExpBuf_.Get<int32_t>();
    LocalTensor<uint8_t> expertMaskTensorU8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> expertMaskTensorU32 = gatherMaskTBuf_.Get<uint32_t>();
    LocalTensor<int32_t>  gatherIndex = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t>  dstOffset = dstExpBuf_.Get<uint32_t>();

    // SyncFunc<AscendC::HardEvent::MTE2_V>();
    ////LOG_INFO("expertIdsTensor_",expertIdsTensor_,{static_cast<uint32_t>(activeMaskBsCnt_), axisK_});
    PipeBarrier<PIPE_V>();
    Cast(dstServerIdFp32, expertIdsTensor_, RoundMode::CAST_NONE, calCnt);
    Duplicate<float32_t>(tempTensorFp32, static_cast<float32_t>(moeExpertNumPerRank_), calCnt);
    PipeBarrier<PIPE_V>();
    ////LOG_INFO("dstServerIdFp32",dstServerIdFp32,{static_cast<uint32_t>(activeMaskBsCnt_), axisK_});
    Div(dstServerIdFp32, dstServerIdFp32, tempTensorFp32, calCnt);
    PipeBarrier<PIPE_V>();
    ////LOG_INFO("dstServerIdFp32",dstServerIdFp32,{static_cast<uint32_t>(activeMaskBsCnt_), axisK_});
    Adds(dstServerIdFp32, dstServerIdFp32, static_cast<float32_t>(sharedExpertRankNum_), calCnt);
    PipeBarrier<PIPE_V>();
    ////LOG_INFO("dstServerIdFp32",dstServerIdFp32,{static_cast<uint32_t>(activeMaskBsCnt_), axisK_});
    Duplicate<float32_t>(tempTensorFp32, static_cast<float32_t>(rankSizePerServer_), calCnt);
    PipeBarrier<PIPE_V>();
    Div(dstServerIdFp32, dstServerIdFp32, tempTensorFp32, calCnt); //计算每个专家所在的serverID
    PipeBarrier<PIPE_V>();
    ////LOG_INFO("dstServerIdFp32",dstServerIdFp32,{static_cast<uint32_t>(activeMaskBsCnt_), axisK_});
    Cast(dstServerIdInt32, dstServerIdFp32, RoundMode::CAST_TRUNC, calCnt); 
    PipeBarrier<PIPE_V>();
    ////LOG_INFO("dstServerIdInt32",dstServerIdInt32,{static_cast<uint32_t>(activeMaskBsCnt_), axisK_});

    ////LOG_INFO("CalServerSendCnt expertMaskInputTensor_",expertMaskInputTensor_,{axisBS_,axisK_});
    for(int32_t serverIndex = 0; serverIndex < serverNum_; serverIndex++) {
        int32_t dstServerId = serverIndex;
        Adds(dstServerIdInt32, dstServerIdInt32, -dstServerId, calCnt);
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("Adds dstServerIdInt32",dstServerIdInt32,{static_cast<uint32_t>(activeMaskBsCnt_), axisK_});
        Abs(dstServerIdFp32, dstServerIdFp32, calCnt);
        PipeBarrier<PIPE_V>();
        Mins(dstServerIdInt32, dstServerIdInt32, 1, calCnt);
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("Mins dstServerIdInt32",dstServerIdInt32,{static_cast<uint32_t>(activeMaskBsCnt_), axisK_});
        Cast(dstServerIdInt16, dstServerIdInt32, RoundMode::CAST_NONE, calCnt);
        Duplicate<int16_t>(tempTensorInt16, 1, calCnt);
        PipeBarrier<PIPE_V>();
        Xor(compareInt16, dstServerIdInt16, tempTensorInt16, sharedTensorInt8, calCnt);
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("Xor compareInt16",compareInt16,{static_cast<uint32_t>(activeMaskBsCnt_), axisK_});
        Cast(dstServerIdFp32, compareInt16, RoundMode::CAST_NONE, calCnt);//这里是 0 1 结果
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("Cast dstServerIdFp32",dstServerIdFp32,{static_cast<uint32_t>(activeMaskBsCnt_), axisK_});
        uint32_t srcShape[2] = {activeMaskBsCnt, axisK_};
        ReduceSum<float32_t, Pattern::Reduce::AR, true>(tempTensorFp32, dstServerIdFp32, sharedTensorInt8, srcShape, false);
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("ReduceSum tempTensorFp32",tempTensorFp32,{1,static_cast<uint32_t>(activeMaskBsCnt_)});
        Mins(dstServerIdFp32, tempTensorFp32, float32_t(1.0), activeMaskBsCnt); // 得到需要发送到这个server的TokenID
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("Mins dstServerIdFp32",tempTensorFp32,{1,static_cast<uint32_t>(activeMaskBsCnt_)});
        Cast(dstServerIdInt32, dstServerIdFp32, RoundMode::CAST_RINT, activeMaskBsCnt);
        CreateVecIndex(indexTensor, 0, activeMaskBsCnt);
        Duplicate<uint32_t>(expertMaskTensorU32, 0, maskSize);
        PipeBarrier<PIPE_V>();
        CompareScalar(expertMaskTensorU8, dstServerIdInt32, 1, CMPMODE::EQ, compareCount);
        PipeBarrier<PIPE_V>();
        GatherMask(gatherIndex, indexTensor, expertMaskTensorU32, true, activeMaskBsCnt, {1, 1, 0, 0}, sendTokenCnt); // 是否可以简化计算
        PipeBarrier<PIPE_V>();
        if(sendTokenCnt == 0) {
            continue;
        }
        ////LOG_INFO("GatherMask gatherIndex",gatherIndex,{1,static_cast<uint32_t>(sendTokenCnt)});
        serverCountTensor_.SetValue(serverIndex, static_cast<uint32_t>(sendTokenCnt));
        if constexpr (isStoreOffset) {
            Muls(gatherIndex, gatherIndex, static_cast<int32_t>(sizeof(int32_t)), static_cast<uint32_t>(sendTokenCnt));
            PipeBarrier<PIPE_V>();
            Scatter(tokenSendMap_[serverIndex * perServerMapCnt_], indexTensor, dstOffset, 0, static_cast<uint32_t>(sendTokenCnt)); // 32字节对齐
            PipeBarrier<PIPE_V>();
        }
    }
    PipeBarrier<PIPE_ALL>(); //todo：这里必须这样，或者后面的清理可以做
    //LOG_INFO("tokenSendMap_", tokenSendMap_,{1, static_cast<uint32_t>(activeMaskBsCnt_)});
    //LOG_INFO("serverCountTensor_", serverCountTensor_,{1, serverNum_});
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::QuantCleanTensor()
{
     #if defined(__DAV_C310__)
        if constexpr (QuantMode > UNQUANT && ((QuantMode == MX_QUANT) || (QuantMode == PERGROUP_DYNAMIC_QUANT))) {
            // 由于MX以及PERGROUP量化在计算scales时每次搬入256字节数据，所以在token搬入前需要对空间填0，避免引入脏数据
            LocalTensor<uint8_t> singleByteTok = tokenData_.template ReinterpretCast<uint8_t>();
            Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XType));
            PipeBarrier<PIPE_V>();
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    #endif
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendTokenToLocalServer(uint32_t firstAivId, uint32_t usedAivNum)
{
    LOG_INFO("[SendTokenToLocalServer] start");
    uint32_t totalSendCnt = activeMaskBsCnt_;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(axisK_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> expertScalePadParams{false, 0U, 0U, 0U};
    DataCopyPadParams copyPadExtParams{true, 0U, 0U, 0U};
    SplitToCore(totalSendCnt, firstAivId, usedAivNum, startTokenId, endTokenId, sendTokenNum);
    if (startTokenId >= endTokenId || sendTokenNum == 0) {
        return ;
    }
    LOG_INFO("[SendTokenToLocalServer] startTokenId: ", startTokenId);
    LOG_INFO("[SendTokenToLocalServer] endTokenId: ", endTokenId);
    LOG_INFO("[SendTokenToLocalServer] sendTokenNum: ", sendTokenNum);
    LOG_INFO("[SendTokenToLocalServer] firstAivId: ", firstAivId);
    LOG_INFO("[SendTokenToLocalServer] usedAivNum: ", usedAivNum);

    CalExpertOffset(startTokenId, endTokenId);
    SyncFunc<AscendC::HardEvent::V_S>();
    float expertScale = float(1.0);
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; tokenIndex++) {
        uint32_t offset = localServerId_ * perServerMapCnt_ + tokenIndex;
        int32_t tokenSendOffset = tokenSendMap_.GetValue(offset); // tdo:换成每个server只处理自己的，不在同一计算
        if(tokenSendOffset == -1) {
            continue;
        }
        QuantCleanTensor();
        xInQueue_.EnQue(tokenData_);
        tokenData_ = xInQueue_.DeQue<XType>();
        DataCopyPad(tokenData_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
        DataCopyPad(expertScalesTensor_, expertScalesGMTensor_[tokenIndex * axisK_], copyParams, expertScalePadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t topKIndex = 0; topKIndex < axisK_; topKIndex++) {
            uint32_t expertIndex = tokenIndex * axisK_ + topKIndex;
            uint32_t dstExpertId = expertIdsTensor_(expertIndex);
            uint32_t dstRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_; // 发往的卡号
            uint32_t expertCntIndex = (tokenIndex - startTokenId) * moeCntRowAlign_ + topKIndex;
            uint32_t curExpertCnt = expertOffsetCntTensor_(expertCntIndex);
            expertScale = expertScalesTensor_(topKIndex);
            GM_ADDR rankGM = reinterpret_cast<GM_ADDR>(GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                                            (expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ +
                                            dstExpertId % moeExpertNumPerRank_)) + hCommuSize_ * curExpertCnt); // 计算地址偏移
            ProcessToken(rankGM, tokenIndex, topKIndex, dstExpertId + sharedExpertNum_, epRankId_, expertScale);
       }

       for (int32_t shareIndex = 0; shareIndex < sharedExpertNum_; shareIndex++) { // 发共享专家
            uint32_t rankIDSharedGroup = epRankId_ % rankNumPerSharedExpert_;
            uint32_t dstRankId = shareIndex * rankNumPerSharedExpert_ + rankIDSharedGroup;
            uint32_t curExpertCnt = tokenIndex; // cnt即tokenid
            if (!IsInSameServer(dstRankId)) { // 不同server
                continue; 
            }
            GM_ADDR rankGM = reinterpret_cast<GM_ADDR>(GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                                                expertPerSizeOnWin_ * epRankId_ + hCommuSize_ * curExpertCnt);
            ProcessToken(rankGM, tokenIndex, axisK_ + shareIndex, shareIndex, epRankId_, expertScale);
        }
    }
    xInQueue_.FreeTensor<XType>(tokenData_);
    LOG_INFO("[SendTokenToLocalServer] end");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendTokenToRemoteServer(uint32_t serverIndex, uint32_t firstAivId, uint32_t usedAivNum)
{
    uint32_t totalSendCnt = activeMaskBsCnt_;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(totalSendCnt, firstAivId, usedAivNum, startTokenId, endTokenId, sendTokenNum);
    LOG_INFO("[SendTokenToRemoteServer] startTokenId",startTokenId);
    LOG_INFO("[SendTokenToRemoteServer] endTokenId",endTokenId);
    LOG_INFO("[SendTokenToRemoteServer] sendTokenNum",sendTokenNum);
    LOG_INFO("[SendTokenToRemoteServer] firstAivId",firstAivId);
    LOG_INFO("[SendTokenToRemoteServer] usedAivNum",usedAivNum);
    if (startTokenId >= endTokenId || sendTokenNum == 0) {
        return ;
    }
    CalExpertOffset(startTokenId, endTokenId);
    ScatterToken(startTokenId, endTokenId, serverIndex);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ScatterTokenToRemote()
{
    LOG_INFO("[ScatterTokenToRemote] start");
    CalServerSendCnt<true>();
    if (aivId_ < aivUsedCumSum_ + aivUsedRemotComm_) {
        uint32_t relativeServerId = aivId_ - aivUsedCumSum_;
        if (relativeServerId == localServerId_) {
            return; // TODO： 先暂时不用这个核，后续处理
        }
        SetServerFlag(relativeServerId); // 需要先发送Token总数，每个核负责一个server
        //专门处理发送给其他server的Token
        HcclCommSend(relativeServerId); // 专门负责发送Token
        return ;
    }
    uint32_t startServerId, endServerId, useAivPerServer;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    uint32_t firstAivId;

    CalServerSplitCore(serverNum_, aivUsedCumSum_ + aivUsedRemotComm_, aivUsedScatter_, 
                       startServerId, endServerId, useAivPerServer, firstAivId);
    if (useAivPerServer == 0 || startServerId >= endServerId) { // 出现serverNum比核数大的情况(TODO：需要确认是否需要)
        return ;
    }
    LOG_INFO("[ScatterTokenToRemote] serverNum_",serverNum_);
    LOG_INFO("[ScatterTokenToRemote] startServerId_",startServerId);
    LOG_INFO("[ScatterTokenToRemote] endServerId_",endServerId);
    LOG_INFO("[ScatterTokenToRemote] useAivPerServer_",useAivPerServer);
    LOG_INFO("[ScatterTokenToRemote] firstAivId_",firstAivId);
    InitScatterBuffer(startServerId); // todo:
    for (uint32_t serverIndex = startServerId; serverIndex < endServerId; serverIndex++) {
        if (serverCountTensor_.GetValue(serverIndex) == 0){
            continue;
        }
        if (serverIndex == localServerId_) {
            SendTokenToLocalServer(firstAivId, useAivPerServer);
            continue;
        }
        LOG_INFO("[ScatterTokenToRemote] OUT continue");
        SendTokenToRemoteServer(serverIndex, firstAivId, useAivPerServer);
    }
    LOG_INFO("[ScatterTokenToRemote] end");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalServerSplitCore(
    uint32_t serverNum, uint32_t startAivId, uint32_t useAivNum, uint32_t &startServerId, uint32_t &endServerId, uint32_t &useAivPerServer, uint32_t &firstAivId)
{
    useAivPerServer = useAivNum / serverNum; // 每个server分几个核处理
    uint32_t remainderAivNum = useAivNum % serverNum; 
    if (useAivPerServer > 0) { // 一个server多个核处理
        uint32_t relativeAivId = aivId_ - startAivId; // 相对Id
        startServerId = relativeAivId / useAivPerServer; // 当前核需要处理的
        endServerId = startServerId + 1;
        firstAivId = startAivId + startServerId * useAivPerServer + remainderAivNum;
        if (startServerId < remainderAivNum) { // 如果有剩余的核
            useAivPerServer += 1; // 前面的server多分一个核
            firstAivId = startAivId + startServerId * useAivPerServer;
        }
    } else { // 一般不会出现serverNum比核数大的情况
        useAivPerServer = 0;
        startServerId = 0;
        endServerId = 0;
    }
}

template <TemplateDispatchKFCTypeClass>
template <CommMode mode>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitQuantBuffer()
{
    if constexpr (QuantMode > UNQUANT) {
        if constexpr (mode == CommMode::INTER_NODE) {
            tpipe_->InitBuffer(receiveDataCastFloatBuf_, maxSize_); // max{28K, BS * K * 4B}
            tpipe_->InitBuffer(smoothScalesBuf_, maxSize_); // max{28K, BS * K * 4B}
        } else {
            receiveDataCastFloatBuf_ = dstExpBuf_;
            smoothScalesBuf_ = subExpBuf_;          // 内存复用
        }
        floatLocalTemp_ = receiveDataCastFloatBuf_.Get<float>();
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
        if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN); // 32B
             rowMaxTensor_ = rowMaxBuf_.Get<float>();
        }
    }
    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf_, dynamicScalesOutGMTensor_);
}


template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitGatherTokenBuffer()
{
    tpipe_->Reset();
    InitQuantBuffer<CommMode::INTER_NODE>();
    uint32_t flagMaxRecvBuf = blockCntPerToken_ * UB_ALIGN;
    uint64_t mask[1] = {0x0101010101010101};
    uint8_t commRepeatTime = static_cast<uint8_t>(Ceil(maxBlockCnt_ * UB_ALIGN, SIZE_ALIGN_256));
    uint8_t tokenRepeatTime = static_cast<uint8_t>(Ceil(blockCntPerToken_ * UB_ALIGN, SIZE_ALIGN_256));
    uint32_t commFlagSize = commRepeatTime * SIZE_ALIGN_256;
    uint32_t tokenFlagSize = tokenRepeatTime * SIZE_ALIGN_256;
    TBuf<> tokenRcvStatusBuf;
    TBuf<> tokenFlagBuf;
    TBuf<> cleanBuf;
    TBuf<> dpuInitBufferBuf;

    tpipe_->InitBuffer(tokenInQueue_, BUFFER_NUM, sendTokenLengthAlign_); // token输入队列
    tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hAlignSize_); // 14K * 2
    tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hOutAlignUbSize_); // 7K * 2 + 32 + 6
    tpipe_->InitBuffer(flagBuf_, commFlagSize);
    tpipe_->InitBuffer(tokenRcvStatusBuf, TOKEN_RECV_CNT_MAX * sizeof(uint32_t)); // 一个server最多发512个token给
    tpipe_->InitBuffer(tokenFlagBuf, flagMaxRecvBuf); 
    tpipe_->InitBuffer(cleanBuf, tokenFlagSize); // 每个Token的flag个数为 blockCntPerToken_
    tpipe_->InitBuffer(xTempQueue_, BUFFER_NUM, hCommuSize_); // 17.5KB
    tpipe_->InitBuffer(dpuInitBufferBuf, SIZE_ALIGN_256); // 要求256B

    dpuInitBufferTensor_ = dpuInitBufferBuf.Get<uint32_t>();
    commFlagTensor_ = flagBuf_.Get<uint32_t>();
    tokenData_ = xInQueue_.AllocTensor<XType>();
    insertFlagTensor_ = xTempQueue_.AllocTensor<ExpandXOutType>();
    recvTmpTensor_ = tokenInQueue_.AllocTensor<XType>();
    tokenRcvStatus_ = tokenRcvStatusBuf.Get<uint32_t>();
    flagRecvTensor_ = tokenFlagBuf.Get<float32_t>();
    statusCleanFp32Tensor_ = cleanBuf.Get<float32_t>();

    Duplicate<uint32_t>(commFlagTensor_, 0x3F800000, mask, commRepeatTime, uint16_t(1), uint8_t(8)); // 0x3F800000是float的1
    Duplicate<float32_t>(statusCleanFp32Tensor_, float32_t(0), mask, tokenRepeatTime, uint16_t(1), uint8_t(8));
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CleanTokenFlag(uint32_t serverId, uint32_t startTokenId, uint32_t cleanTokenNum)
{
    uint32_t cleanUpNum = blockCntPerToken_ * cleanTokenNum;
    //GlobalTensor<uint32_t> statusCntGlobal;
    GM_ADDR dataCleanAddr = GetReadAddrFromLocalServer(serverId) + SERVER_STATE_ALIGN + startTokenId * sendTokenLengthAlign_ + SPLIT_BLOCK_DATA_SIZE;
    GM_ADDR flagCleanAddr = GetReadAddrFromLocalServer(serverId);
    //statusCntGlobal.SetGlobalBuffer((__gm__ uint32_t *)flagCleanAddr);
    GlobalTensor<float32_t> cleanGlobal;
    cleanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(dataCleanAddr));
    DataCopyExtParams cleanUpParams = {uint16_t(cleanUpNum), sizeof(float32_t), 0U,
                                    SERVER_STATE_ALIGN - sizeof(int32_t), 0U}; // 清理512B的最后4字节
    DataCopyPad(cleanGlobal, statusCleanFp32Tensor_, cleanUpParams);
    //statusCntGlobal.SetValue(SERVER_CNT_OFFSET, 0); // 清理状态 TODO:
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::GatherTokenFrRemote()
{
    // LOG_INFO("[GatherTokenFrRemote] start");
    InitGatherTokenBuffer();
    TBuf<> serverFlagBuf;
    uint32_t startServerId, endServerId, useAivPerServer;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    uint32_t firstAivId;
    CalServerSplitCore(serverNum_, aivUsedCumSum_ + aivUsedScatter_ + aivUsedRemotComm_, aivUsedGather_, 
                       startServerId, endServerId, useAivPerServer, firstAivId);
    // LOG_INFO("[GatherTokenFrRemote] serverNum_",serverNum_);
    // LOG_INFO("[GatherTokenFrRemote] startServerId_",startServerId);
    // LOG_INFO("[GatherTokenFrRemote] endServerId_",endServerId);
    // LOG_INFO("[GatherTokenFrRemote] useAivPerServer_",useAivPerServer);
    if (useAivPerServer == 0 || startServerId >= endServerId) { // 出现serverNum比核数大的情况 TODO：
        return ;
    }
    
    tpipe_->InitBuffer(serverFlagBuf, UB_ALIGN);
    //LOG_INFO("[GatherTokenFrRemote] relativeAivId_",relativeAivId);
    //LOG_INFO("[GatherTokenFrRemote] firstAivId_",firstAivId);
    for (uint32_t serverIndex = startServerId; serverIndex < endServerId; serverIndex++) {
        // if (serverIndex == localServerId_) { // 本server不用从其他server接收token，直接跳过
        //     continue;
        // }
        uint32_t tokenCnt;
        WaitStatusFlag(serverIndex, serverFlagBuf, tokenCnt);
        if (tokenCnt == 0) {
            continue;
        }
        SplitToCore(tokenCnt, firstAivId, useAivPerServer, startTokenId, endTokenId, sendTokenNum);
        if (startTokenId >= endTokenId || sendTokenNum == 0) {
            continue;   
        }
        LOG_INFO("[GatherTokenFrRemote] tokenCnt",tokenCnt);
        //LOG_INFO("[GatherTokenFrRemote] startTokenId_",startTokenId);
        //LOG_INFO("[GatherTokenFrRemote] endTokenId_",endTokenId);
        //LOG_INFO("[GatherTokenFrRemote] sendTokenNum_",sendTokenNum);
        Duplicate<uint32_t>(tokenRcvStatus_, 0, TOKEN_RECV_CNT_MAX);
        SyncFunc<AscendC::HardEvent::V_S>();
        tokenInQueue_.EnQue(recvTmpTensor_);
        WaitToken(sendTokenNum, serverIndex, startTokenId);
        CleanTokenFlag(serverIndex, startTokenId, sendTokenNum);
    }
    xInQueue_.FreeTensor<XType>(tokenData_);
    tokenInQueue_.FreeTensor<XType>(recvTmpTensor_);
    //LOG_INFO("[GatherTokenFrRemote] end");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitStatusFlag(
    uint32_t dstServerId, TBuf<> &serverFlagBuf, uint32_t &rcvTokenCnt)
{
    LocalTensor<uint32_t> statusFlagTensor, tokenCntTensor;
    GlobalTensor<uint32_t> statusCntGlobal;
    GM_ADDR wAddr = GetReadAddrFromLocalServer(dstServerId); // 对应server接收区地址
    statusCntGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(wAddr));
    statusFlagTensor = serverFlagBuf.Get<uint32_t>();
    tokenCntTensor = serverFlagBuf.Get<uint32_t>();
    uint32_t count = 0;
    while (true) {
        count++;
        if(count >10000000) {
            LOG_INFO("WaitStatusFlag error count");

            break;
        }
        DataCopy(statusFlagTensor, statusCntGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)],
                 (SERVER_STATE_ALIGN - SPLIT_BLOCK_DATA_SIZE) / sizeof(uint32_t)); // 搬运后续的8个数据
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint32_t flag = statusFlagTensor.GetValue(0);
        if (flag == 1) { // 等待flag 
            break;
        }
    }
    DataCopy(tokenCntTensor, statusCntGlobal, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    rcvTokenCnt = tokenCntTensor.GetValue(0);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CopyInAndOut(uint32_t serverIdx,
                                                                                                uint32_t dstPosition,
                                                                                                uint32_t processTokenNum)
{
    GM_ADDR dataAddr = GetReadAddrFromLocalServer(serverIdx) + SERVER_STATE_ALIGN + dstPosition * sendTokenLengthAlign_;

    GlobalTensor<XType> dataFlagGlobal;
    dataFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(dataAddr));

    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerToken_ * processTokenNum),
                                         static_cast<uint32_t>(SPLIT_BLOCK_DATA_SIZE), static_cast<uint32_t>(SERVER_STATE_ALIGN - SPLIT_BLOCK_DATA_SIZE), 0};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};

    recvTmpTensor_ = tokenInQueue_.DeQue<XType>();
    DataCopyPad(recvTmpTensor_, dataFlagGlobal, srcTokenCopyParams, srcTokenPadParams);
    tokenInQueue_.EnQue(recvTmpTensor_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitToken(uint32_t tokenCnt, uint32_t serverIdx,
                                                               uint32_t startTokenIdx)
{
    uint32_t rcvStatusIndex = 0;
    uint32_t rcvTokenSum = 0;
    uint32_t processTokenNumPerBatch = 1;
    // uint32_t flagCopyStride = WIN_ADDR_ALIGN - sizeof(float32_t);

    uint32_t count = 0;
    while (true) {
        count++;
        if (count == 1000000) {
            PRINTF("[BATCHWRITE][WaitToken] error count.\n");
            break;
        }
        if (tokenRcvStatus_(rcvStatusIndex) == 1) {
            rcvStatusIndex = (rcvStatusIndex + 1) % tokenCnt; // 轮询查询每个有效的index
            continue;
        }
        //LOG_INFO("[BATCHWRITE][WaitToken] rcvStatusIndex",rcvStatusIndex);
        GM_ADDR flagAddr = GetReadAddrFromLocalServer(serverIdx) + SERVER_STATE_ALIGN +
                            (rcvStatusIndex + startTokenIdx) * sendTokenLengthAlign_ + SPLIT_BLOCK_DATA_SIZE; // 第一个flag起始地址
        bool isArrived = WaitFlag<false>(flagAddr, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, blockCntPerToken_);
        if (isArrived) { // 处理到达的情况
            uint32_t dstPosition = rcvStatusIndex + startTokenIdx;
            uint32_t processTokenBlock = blockCntPerToken_ * processTokenNumPerBatch;
            GM_ADDR dataAddr = GetReadAddrFromLocalServer(serverIdx) +
                                                           SERVER_STATE_ALIGN + dstPosition * sendTokenLengthAlign_;
            recvTmpTensor_ = tokenInQueue_.DeQue<XType>();                                            
            ReadDataFromWindows<XType>(dataAddr, processTokenBlock, recvTmpTensor_);
            //CopyInAndOut(serverIdx, dstPosition, processTokenNumPerBatch);
            //LOG_INFO("[BATCHWRITE][WaitToken][true]after CopyInAndOut tokenid:", dstPosition);
            SendToExpert(dstPosition);
            tokenRcvStatus_(rcvStatusIndex) = 1;
            rcvTokenSum++;
            if (rcvTokenSum == tokenCnt) {
                LOG_INFO("[BATCHWRITE][WaitToken] finishNum:");
                break;
            }
            continue;
        }
        rcvStatusIndex = (rcvStatusIndex + 1) % tokenCnt;
    }
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
    SyncFunc<AscendC::HardEvent::S_V>();
    // SyncFunc<AscendC::HardEvent::S_MTE3>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::IsInSameServer(uint32_t targetRankId)
{
    return targetRankId / rankSizePerServer_ == localServerId_;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ProcessToken(GM_ADDR rankGM,
            uint32_t tokenIndex, uint32_t topKIndex, uint32_t expertIndex,
            uint32_t srcRankIndex, float expertScale)
{
    xInQueue_.EnQue(tokenData_);
    tokenData_ = xInQueue_.DeQue<XType>();
    xTempQueue_.EnQue(insertFlagTensor_);
    insertFlagTensor_ = xTempQueue_.DeQue<ExpandXOutType>();
    if constexpr (QuantMode > UNQUANT) {
        xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
        quantInst_.QuantProcess(xOutTensor_, tokenData_, expertIndex, scalesCount_, scalesGMTensor_); // 量化
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
        FillTriple(xOutTensor_, srcRankIndex, tokenIndex, topKIndex, expertScale);
        WriteDataToWinOut(xOutTensor_, insertFlagTensor_, rankGM, blockCntPerTokenToExpert_); //TODO:可以继续优化
        xOutQueue_.FreeTensor<ExpandXOutType>(xOutTensor_);
    } else {
        
        xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
        DataCopy(xOutTensor_, tokenData_, hAlignSize_ / sizeof(ExpandXOutType));
        #if defined(__DAV_C310__)
        if constexpr (IsSmoothScaleExist) {
            DataCopyPadParams padParams = {true, 0, 0, 0};
            DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};
            auto tmp = scalesGMTensor_.ReinterpretCast<uint8_t>();
            DataCopyPad(xOutTensor_[axisH_].template ReinterpretCast<uint8_t>(), tmp[tokenIndex * scaleInBytes_],
                scaleInParams, padParams);
        }
        #endif
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
        FillTriple(xOutTensor_, srcRankIndex, tokenIndex, topKIndex, expertScale);
        WriteDataToWinOut(xOutTensor_, insertFlagTensor_, rankGM, blockCntPerTokenToExpert_); 
        xOutQueue_.FreeTensor<ExpandXOutType>(xOutTensor_);
    }
}


template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendToExpert(uint32_t rcvCnt)
{
    //LOG_INFO("[SendToExpert][moe] rcvCnt",rcvCnt);
    QuantCleanTensor();
    tokenInQueue_.EnQue(recvTmpTensor_);  
    recvTmpTensor_ = tokenInQueue_.DeQue<XType>();
    xInQueue_.EnQue(tokenData_);
    tokenData_ = xInQueue_.DeQue<XType>();
    DataCopy(tokenData_, recvTmpTensor_, hAlignSize_ / sizeof(XType));
    LocalTensor<int16_t> xInTensorInt16 = recvTmpTensor_.template ReinterpretCast<int16_t>();
    LocalTensor<uint32_t> xInTensor = recvTmpTensor_.template ReinterpretCast<uint32_t>();
    LocalTensor<float> xInToFloatTensor = recvTmpTensor_.template ReinterpretCast<float>();
    uint32_t srcRankIndex = xInTensor(sendXTypeElemAlign_);
    uint32_t tokenIndex = xInTensor(sendXTypeElemAlign_ + 1);
    LOG_INFO("[SendToExpert][moe] srcRankIndex",srcRankIndex);
    LOG_INFO("[SendToExpert][moe] tokenIndex",tokenIndex);
    float expertScale = float(1.0);
    for (int32_t topKIndex = 0; topKIndex < axisK_; topKIndex++) {                      // 发moe专家
        uint32_t dstExpertId = xInTensor(moeListAlign_ + topKIndex);                    // 取出专家id
        uint32_t dstRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_; // 发往的卡号
        uint32_t curExpertCnt = xInTensorInt16(moeCntAlign_ + topKIndex);                    // 取出cnt
        expertScale = xInToFloatTensor(moeExpertScalesAlign_ + topKIndex); 
        LOG_INFO("[SendToExpert][moe] topKIndex",topKIndex);
        LOG_INFO("[SendToExpert][moe] dstExpertId",dstExpertId);
        LOG_INFO("[SendToExpert][moe] dstRankId",dstRankId);
        LOG_INFO("[SendToExpert][moe] curExpertCnt",curExpertCnt);
        LOG_INFO("[SendToExpert][moe] expertScale",expertScale);
        
        if (dstExpertId >= moeExpertNum_ || !IsInSameServer(dstRankId)) { // 无效数据或者不在本server
            continue;
        }
        GM_ADDR rankGM = reinterpret_cast<GM_ADDR>(GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                                            (expertPerSizeOnWin_ * (srcRankIndex * moeExpertNumPerRank_ +
                                            dstExpertId % moeExpertNumPerRank_)) + hCommuSize_ * curExpertCnt); // 计算地址偏移
        ProcessToken(rankGM, tokenIndex, topKIndex, dstExpertId + sharedExpertNum_, srcRankIndex, expertScale);
    }

    tokenInQueue_.EnQue(recvTmpTensor_);

    for (int32_t shareIndex = 0; shareIndex < sharedExpertNum_; shareIndex++) { // 发共享专家
        uint32_t rankIDSharedGroup = epRankId_ % rankNumPerSharedExpert_;
        uint32_t dstRankId = shareIndex * rankNumPerSharedExpert_ + rankIDSharedGroup;
        uint32_t curExpertCnt = rcvCnt; // cnt即tokenid
        if (!IsInSameServer(dstRankId)) { // 不同server
            continue; 
        }
        GM_ADDR rankGM = reinterpret_cast<GM_ADDR>(GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                                            expertPerSizeOnWin_ * srcRankIndex + hCommuSize_ * curExpertCnt);
        ProcessToken(rankGM, tokenIndex, axisK_ + shareIndex, shareIndex, srcRankIndex, expertScale);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitStatusTensor()
{
    uint32_t rcvExpertDataSize = SPLIT_BLOCK_DATA_SIZE * blockCntExpertPerServer_;
    uint32_t flagMaxRecvNum = blockCntExpertPerServer_;
    uint32_t flagBufferSize = flagMaxRecvNum * UB_ALIGN;
    uint32_t statusBufBufferSize = flagBufferSize > rcvExpertDataSize ? flagBufferSize : rcvExpertDataSize;
    uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    TBuf flagRecvBuf;
    tpipe_->InitBuffer(flagRecvBuf, statusBufBufferSize);
    tpipe_->InitBuffer(statusBuf_, statusBufCntAlign_ * UB_ALIGN);
    flagRecvTensor_ = flagRecvBuf.Get<float32_t>();
    statusBufCntAlign_ = Ceil(expertNumInServer_ * serverNum_, 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    statusTensor_ = statusBuf_.Get<int32_t>();                  // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, statusBufCntAlign_ * 8);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign_ / 8, 1, 8); // 0x3F800000是float的1
    PipeBarrier<PIPE_V>();
}

template <TemplateDispatchKFCTypeClass>
template <typename T>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ReadDataFromWindows(GM_ADDR dataAddr, uint32_t blockCnt, LocalTensor<T> &xInTensor)
{
    GlobalTensor<T> dataFlagGlobal;
    dataFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dataAddr));

    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCnt), static_cast<uint32_t>(SPLIT_BLOCK_DATA_SIZE),
                                      static_cast<uint32_t>(SERVER_STATE_ALIGN - SPLIT_BLOCK_DATA_SIZE), 0};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xInTensor, dataFlagGlobal, srcTokenCopyParams, srcTokenPadParams);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetStatus() //框间状态区接收 + 框内状态区发送
{
    uint32_t startServerId;
    uint32_t endServerId;
    uint32_t resvServerNum;
    SplitToCore(serverNum_, 0, aivUsedCumSum_, startServerId, endServerId, resvServerNum);
    if (startServerId >= endServerId || resvServerNum == 0) {
        return;
    }

    InitStatusTensor();
    
    for(int serverIdx = startServerId; serverIdx < endServerId; serverIdx++) {
        LocalTensor<uint32_t> rcvExpertCntTensor = flagRecvTensor_.template ReinterpretCast<uint32_t>(); // buf复用
        if (serverIdx != localServerId_) {
            GM_ADDR serverRcvAddr = GetReadAddrFromLocalServer(serverIdx);
            GM_ADDR flagAddr = reinterpret_cast<GM_ADDR>(serverRcvAddr + SPLIT_BLOCK_DATA_SIZE);
            WaitFlag<true>(flagAddr, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, blockCntExpertPerServer_);
            ReadDataFromWindows<uint32_t>(serverRcvAddr, blockCntExpertPerServer_, rcvExpertCntTensor);
            SyncFunc<AscendC::HardEvent::MTE2_S>(); 
        }
        for (uint32_t expertIndex = 0; expertIndex < expertNumInServer_; expertIndex ++) {
            if(serverIdx != localServerId_) {
                statusTensor_(expertIndex * 8 + 1) = rcvExpertCntTensor(expertIndex);
            } else {
                if (expertIndex < shareRankNumInServer_) { // 共享专家
                    statusTensor_(expertIndex * 8 + 1) = serverCountTensor_.GetValue(localServerId_);
                } else { // moe专家
                    uint32_t moeExpertIndex = startMoeExpertId_ + expertIndex - shareRankNumInServer_;
                    statusTensor_(expertIndex * 8 + 1) = tokenNumToExpertTensor_.GetValue(moeExpertIndex);
                }
            }
            
        }
        //LOG_INFO("[statusTensor_]",statusTensor_,{moeExpertNum_,8});
        PipeBarrier<PIPE_ALL>(); //todo: 优化PIPE_ALL
        SyncFunc<AscendC::HardEvent::S_MTE3>(); 
        GlobalTensor<int32_t> rankGMTensor;
        for (uint32_t expertIndex = 0; expertIndex < expertNumInServer_; expertIndex ++) {
            uint32_t srcRankId = serverIdx * rankSizePerServer_ + epRankId_ % rankSizePerServer_;
            uint32_t dstRankId = expertIndex; // 共享专家
            uint32_t offset = stateOffset_ * srcRankId;
            if (expertIndex >= shareRankNumInServer_) { // moe专家
                dstRankId =
                    (expertIndex - shareRankNumInServer_) / moeExpertNumPerRank_ + shareRankNumInServer_;
                offset += ((expertIndex - shareRankNumInServer_) % moeExpertNumPerRank_ * epWorldSize_ * stateOffset_);
            }
            GM_ADDR rankGM =
                reinterpret_cast<GM_ADDR>(GetWindStateAddrByRankId(COMM_EP_IDX, dstRankId) + offset); // 计算地址偏移
            rankGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rankGM));
            DataCopy<int32_t>(rankGMTensor, statusTensor_[expertIndex * 8], 8UL);
            // //LOG_INFO("[rankGM]", rankGM);
            // //LOG_INFO("[srcRankId]",srcRankId);
            // //LOG_INFO("[dstRankId]",dstRankId);
            // //LOG_INFO("[statusTensor_]",statusTensor_,{8,8});
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitCalCumSumBuffer()
{
    tpipe_->Reset();
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
        // //LOG_INFO("get statusFp32Tensor_", statusFp32Tensor_, {8, 8});
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
        // //LOG_INFO("sumOfFlag:", sumOfFlag);
        count ++;
        if(count == 10000000) {
            PRINTF("waitDispatch error count\n");
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
    GM_ADDR wAddr = reinterpret_cast<GM_ADDR>(recvCntWorkspaceGM_);
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(wAddr));
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
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendExpertCnt()
{
    PipeBarrier<PIPE_V>(); // 防止setvalue 和计算流水没有完成；
    //LOG_INFO("tokenNumToExpertTensor_ ",tokenNumToExpertTensor_, {1, moeExpertNum_});

    uint32_t startServerId, endServerId, sendServerNum;
    SplitToCore(serverNum_, 0, aivUsedCumSum_, startServerId, endServerId, sendServerNum);
    if(startServerId >= endServerId || sendServerNum == 0) {
        return;
    }
    LocalTensor<int32_t> sendTokenNumToexpert = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(sendTokenNumToexpert, 0, moeExpertNumPerRank_ * rankSizePerServer_);
    for(uint32_t serverIndex = startServerId; serverIndex < endServerId; serverIndex++) {
        uint32_t serverStartRankId = serverIndex * rankSizePerServer_;
        uint32_t moeOffset = 0;
        uint32_t moeIdstart = (serverStartRankId - sharedExpertRankNum_) * moeExpertNumPerRank_;
        uint32_t moeNumInServer =  moeExpertNumPerRank_ * rankSizePerServer_;
        uint32_t moeIdend = moeIdstart + moeNumInServer;
        if (serverStartRankId < sharedExpertRankNum_) { //目标server有共享专家
            moeIdstart = 0;
            moeNumInServer = (sharedExpertRankNum_ - serverStartRankId) * moeExpertNumPerRank_;
            if (sharedExpertRankNum_ - serverStartRankId >= rankSizePerServer_) { //全部都是共享专家
                moeNumInServer = 0;
            }
            moeIdend = moeIdstart + moeNumInServer;
            uint32_t groupId = epRankId_ % rankNumPerSharedExpert_;
            moeOffset += sharedExpertRankNum_ - serverStartRankId; // 当前server有几个共享专家卡
            for(uint32_t shareIndex = 0; shareIndex < sharedExpertNum_; shareIndex++) {
                uint32_t dstRankId = shareIndex * rankNumPerSharedExpert_ + groupId;
                uint32_t dstServerId = dstRankId / rankSizePerServer_;
                if (dstServerId != serverIndex) {
                    continue;
                }
                uint32_t dstOffsetIndex = dstRankId - serverStartRankId;
                sendTokenNumToexpert.SetValue(dstOffsetIndex, serverCountTensor_.GetValue(dstServerId));
            }
        }

        for (uint32_t moeIndex = moeIdstart; moeIndex < moeIdend; moeIndex++) {
            uint32_t dst = moeOffset + moeIndex - moeIdstart;
            sendTokenNumToexpert.SetValue(dst, tokenNumToExpertTensor_.GetValue(moeIndex));
        }
        if (serverIndex == localServerId_) {
            continue;
        }

        GM_ADDR localWriteGM = reinterpret_cast<GM_ADDR>((GetDataWriteAddrFromLocalServer<ExecutMode::CUMSUM>(serverIndex)));
        WriteDataToWinOut(sendTokenNumToexpert, localWriteGM, blockCntExpertPerServer_);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        waitWriteTensor_.SetValue(0, 1); // TODO：目前临时使用这种方式来等待写入完成，后续需要优化为其他方式
        GM_ADDR remoteWriteGM = reinterpret_cast<GM_ADDR>((GetDataWriteAddrFromRemoteServer<ExecutMode::CUMSUM>(serverIndex)));
        hcommHandle_.WriteNbi(dpuCommHandle_[GetServerCommHandleIndex(serverIndex)], remoteWriteGM, localWriteGM, blockCntExpertPerServer_ * SERVER_STATE_ALIGN); 
    }
}

    

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalExpertTotalSendCnt() 
{
    uint64_t maskCnt = 0;
    uint32_t compareCount = 0;
    uint32_t maskNumPerExpert = 0;
    uint32_t mask = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    // //LOG_INFO("CalExpertTotalSendCnt mask",mask);

    compareCount = Ceil(mask * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t); // 一共多少元素
    maskNumPerExpert = compareCount / BITS_PER_BYTE / sizeof(int32_t); // 为了匹配CompareScalar接口1bit一共8位
    LocalTensor<uint8_t> expertMaskTensorU8 = subExpBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> expertMaskTensorU32 = subExpBuf_.Get<uint32_t>();
    LocalTensor<int32_t> gatherTempTensor = gatherMaskTBuf_.Get<int32_t>();
    tokenNumToExpertTensor_ = dstExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(tokenNumToExpertTensor_, 0, moeExpertNum_);
    LOG_INFO("CalExpertTotalSendCnt expertIdsTensor_",expertIdsTensor_,{axisBS_,axisK_});
    PipeBarrier<PIPE_V>();
    for (int32_t expertIndex = 0; expertIndex < moeExpertNum_; expertIndex++) { //处理moe专家
        int32_t dstExpertId = expertIndex;
        Duplicate<uint32_t>(expertMaskTensorU32, 0, maskNumPerExpert);
        PipeBarrier<PIPE_V>();
        CompareScalar(expertMaskTensorU8, expertIdsTensor_, dstExpertId, CMPMODE::EQ, compareCount);
        PipeBarrier<PIPE_V>();
        GatherMask(gatherTempTensor, expertIdsTensor_, expertMaskTensorU32, true, mask, {1, 1, 0, 0}, maskCnt);
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("CalExpertTotalSendCnt maskCnt",maskCnt);
        tokenNumToExpertTensor_.SetValue(expertIndex, static_cast<uint32_t>(maskCnt)); //todo :性能优化，大量使用SetValue
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalCumSum()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    CalServerSendCnt<false>();
    CalExpertTotalSendCnt();
    SendExpertCnt();
    SetStatus();
    
    SplitToCore(rscvStatusNum_, 0, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_);

    // //LOG_INFO("startStatusIndex_", startStatusIndex_);
    // //LOG_INFO("endStatusIndex_", endStatusIndex_);
    // //LOG_INFO("recStatusNumPerCore_", recStatusNumPerCore_);
    
    InitCalCumSumBuffer();
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
        if(count == 1000000) {
            PRINTF("WaitCumSumFlag error count \n");
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
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetValidExpertInfo(uint32_t startProcessId, uint32_t endProcessId, 
                                                                                                       uint32_t processNum, uint32_t &validNum)
{
    // 获取cumSum
    uint32_t expInfoSize = Ceil(processNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(recvCntWorkspaceGM_));
    DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    PipeBarrier<PIPE_ALL>();

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    for (uint32_t index = startProcessId; index < endProcessId; index++) { // 从sendCnt中挑选当前有发送过来的卡的token数量
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
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CopyOut(
    LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount)
{
    GlobalTensor<ExpandXOutType> dataFlagGlobal, expandXOutGlobal;
    dataFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ExpandXOutType *>(wAddr));
    expandXOutGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ExpandXOutType *>(expandXOutGM_) + (dstPosition) * axisH_);
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
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitAndFormatOutput(uint32_t validNum)
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
    TBuf<>  tBuf;
    tpipe_->InitBuffer(tBuf, tBufRealSize_);
    xTmpTensor_ = tBuf.Get<ExpandXOutType>();
    flagRecvTensor_ = tBuf.GetWithOffset<float32_t>(flagMaxRecvNum, gatherOutSize);  // buf复用
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor_.template ReinterpretCast<int32_t>();
    //LOG_INFO("WaitAndFormatOutput validNum:", validNum);
    //LOG_INFO("expertMapTensor_", expertMapTensor_,{1, 8});
    int count = 0;
    while (true) {
        count ++;
        if(count > 1000000) {
            PRINTF("WaitAndFormatOutput error count\n");
            // break;
        }
        if (expertLeftNumTensor_(index) == 0) { // 当前核负责的不需要收集
            index = (index + 1) % validNum; // 轮询查询每个有效的index
            continue;
        }
        srcExpRankId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ? maxCopyTokenCnt : expertLeftNumTensor_(index); // 按照ub大小一次搬入多个token
        srcDataBlockIdx = srcExpRankId % epWorldSize_ * localExpertNum + srcExpRankId / epWorldSize_; // 转换成数据区的排布偏移
        int32_t beginIdx = expertFinishNumTensor_(index);
        GM_ADDR flagAddr = reinterpret_cast<GM_ADDR>(windowGM_ + srcDataBlockIdx * expertPerSizeOnWin_ +
                                                     beginIdx * hCommuSize_ + SPLIT_BLOCK_DATA_SIZE);  //TODO:
        uint32_t flagNum = blockCntPerTokenToExpert_ * uint32_t(copyCnt);  
        // LOG_INFO("[WaitAndFormatOutput] start");
        bool isArrived = WaitFlag<false>(flagAddr, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, flagNum);
        // LOG_INFO("[WaitAndFormatOutput] end");
        // LOG_INFO("WaitAndFormatOutput index:", index);
        if (isArrived) {
            //LOG_INFO("WaitAndFormatOutput index:", index);
            //LOG_INFO("arriveCount:", arriveCount);
            //LOG_INFO("copyCnt:", copyCnt);
            dstPosition = srcExpRankId != 0 ? sendCntTensor_(srcExpRankId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);
            GM_ADDR wAddr = reinterpret_cast<GM_ADDR>(windowGM_ + srcDataBlockIdx * expertPerSizeOnWin_);
            CopyOut(xOutInt32Tensor, wAddr, index, dstPosition, copyCnt);
            // finish更新并clean
            expertFinishNumTensor_(index) += copyCnt;
            expertLeftNumTensor_(index) -= copyCnt;
            if (expertLeftNumTensor_(index) == 0) {
                // uint32_t cleanUpNum = expertFinishNumTensor_(index) * blockCntPerTokenToExpert_;
                // DataCopyExtParams cleanUoParams = {uint16_t(cleanUpNum), sizeof(int32_t), 0U, SERVER_STATE_ALIGN - sizeof(int32_t), 0U};
                // LocalTensor<int32_t> cleanTensor = tBuf.GetWithOffset<int32_t>(cleanUpNum * UB_ALIGN, 0); // 在0偏移位置存放比较结果
                // cleanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(wAddr));
                // SyncFunc<AscendC::HardEvent::MTE3_V>();
                // Duplicate<int32_t>(cleanTensor, 0, UB_ALIGN_DATA_COUNT);
                // SyncFunc<AscendC::HardEvent::V_MTE3>();
                //DataCopyPad(cleanGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(int32_t)], cleanTensor, cleanUoParams); //todo:需要重写
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
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitLocalCopyBuffer(uint32_t processNum)
{
    tpipe_->Reset();
    TBuf<> cumSumBuf, statusWaitBuf, statusCleanBuf, expertMapBuf, expertFinishBuf, expertLeftBuf, flagMaskBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // todo:注意UB越界
    uint32_t expInfoSize = Ceil(processNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tBufRealSize_ = MAX_UB_SIZE - (UB_ALIGN + rscvNumAlign + 2 * aivUsedCumSum_ * UB_ALIGN) -
        (expInfoSize * 3) - BUFFER_NUM * UB_ALIGN; // 3为expInfoSize大小buffer申请个数

    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusWaitBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(statusCleanBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);
    tpipe_->InitBuffer(flagMaskBuf, BUFFER_NUM * UB_ALIGN);  // max CompareScalar

    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    statusCleanFp32Tensor_ = statusCleanBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();

    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();
    flagCompResultU8_ = flagMaskBuf.Get<uint8_t>();
    flagCompResultLtU64_ = flagMaskBuf.Get<uint64_t>();
    flagRecvGatherMask_ = statusCleanBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);

    LocalTensor<uint32_t> flagCompResultLtU32 = flagMaskBuf.Get<uint32_t>();
    Duplicate<uint32_t>(flagCompResultLtU32, 0, BUFFER_NUM * UB_ALIGN / sizeof(uint32_t));
    Duplicate<uint32_t>(flagRecvGatherMask_, 0, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::V_S>();
    flagRecvGatherMask_.SetValue(0, 1);
    SyncFunc<AscendC::HardEvent::S_V>();    
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::LocalWindowCopy()
{
    uint32_t startProcessId, endProcessId, processNum;
    uint32_t validNum = 0;
    SplitToCore(rscvStatusNum_, 0, aivNum_, startProcessId, endProcessId, processNum);
    if (processNum == 0) {
        return;
    }
    InitLocalCopyBuffer(processNum);
    WaitCumSumFlag(); // 软同步
    // //LOG_INFO("after WaitCumSumFlag");

    SetValidExpertInfo(startProcessId, endProcessId, processNum, validNum);
    if (validNum == 0) { // 本核负责的Expert对应rank收到数据
        return;
    }
    WaitAndFormatOutput(validNum);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WriteExpertOffset()
{
    //LOG_INFO("start WriteExpertOffset");
    uint32_t newAivId = aivId_ - aivUsedCumSum_;
    uint32_t offsetPerMask = Ceil(expertIdsCnt_ * sizeof(uint16_t), UB_ALIGN) * UB_ALIGN;
    uint64_t dstOffset = newAivId * offsetPerMask; //gmExpertOffsetCnt
    uint64_t dstFlag = (aivNum_ - aivUsedCumSum_) * offsetPerMask + sizeof(uint32_t) * newAivId;
    GlobalTensor<int16_t> expertOffsetCntGMTensor;
    GlobalTensor<float32_t> expertOffsetFlagGMTensor;
    LocalTensor<float32_t> writeCntTensor = dstExpBuf_.Get<float32_t>();
    expertOffsetCntGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(gmExpertOffsetCnt + dstOffset));
    expertOffsetFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(gmExpertOffsetCnt + dstFlag));
    writeCntTensor.SetValue(0,1);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyExtParams expertOffsetCopyOutParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int16_t)), 0U, 0U, 0U};
    DataCopyPad(expertOffsetCntGMTensor, expertOffsetCntTensor_, expertOffsetCopyOutParams);
    //SyncFunc<AscendC::HardEvent::MTE3_MTE3>();
    // //LOG_INFO("WriteExpertOffset dstOffset",dstOffset);
    // //LOG_INFO("WriteExpertOffset expertOffsetCntGMTensor", expertOffsetCntGMTensor, {axisBS_, axisK_});
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams expertCntCopyOutParams{1U, static_cast<uint32_t>(sizeof(float32_t)), 0U, 0U, 0U};
    DataCopyPad(expertOffsetFlagGMTensor, writeCntTensor, expertCntCopyOutParams);
    expertOffsetFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(gmExpertOffsetCnt + (aivNum_ - aivUsedCumSum_) * offsetPerMask));
    PipeBarrier<PIPE_ALL>();
    // //LOG_INFO("WriteExpertOffset dstFlag", dstFlag);
    // //LOG_INFO("WriteExpertOffset expertOffsetFlagGMTensor", expertOffsetFlagGMTensor, {1, aivNum_});
    // //LOG_INFO("end WriteExpertOffset");
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalTokenSendExpertOffset()
{
    uint64_t starttime = GetSystemCycle();
    uint32_t startExpertId, endExpertId, sendExpertNum;
    SplitToCore(moeExpertNum_, aivUsedCumSum_, aivNum_ - aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum);
    if (startExpertId >= endExpertId || sendExpertNum == 0) {
        return ;
    }
    // //LOG_INFO("CalTokenSendExpertOffset startExpertId", startExpertId);
    // //LOG_INFO("CalTokenSendExpertOffset endExpertId", endExpertId);
    // //LOG_INFO("CalTokenSendExpertOffset sendExpertNum", sendExpertNum);
    TBuf<> tempBuf;
    TBuf<> indexBuf;
    
    uint32_t activeBsCnt = activeMaskBsCnt_;
    uint32_t calCnt = isTokenMaskFlag_? activeBsCnt * axisK_ : expertIdsCnt_; // 二维mask时，为了对齐需要全部读取expertID
    uint32_t expertNumAlign256 = Ceil(calCnt * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint64_t sendNumperExpert = 0;

    tpipe_->InitBuffer(tempBuf,  activeBsCnt * sizeof(uint32_t));
    tpipe_->InitBuffer(indexBuf,  calCnt * sizeof(uint16_t));

    LocalTensor<int16_t> dstExpIdTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> expertIdsInt16 = subExpBuf_.Get<int16_t>();
    LocalTensor<half> dstExpIdTensorFP16 = dstExpBuf_.Get<half>();
    LocalTensor<int16_t> XorTensorInt16 = subExpBuf_.Get<int16_t>();
    LocalTensor<uint8_t> sharedTensorInt8 = gatherMaskTBuf_.Get<uint8_t>(); //最小需要256B，最大是element * typesize
    LocalTensor<half> maskInputFp16Tensor = gatherMaskTBuf_.Get<half>();
    LocalTensor<int16_t> maskInputInt16Tensor = gatherMaskTBuf_.Get<int16_t>();
    LocalTensor<int16_t> expertsIndexTensor = indexBuf.Get<int16_t>();
    LocalTensor<uint16_t> gatherMaskTensor = gatherMaskTBuf_.Get<uint16_t>();
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> tempOffset = gatherMaskTBuf_.Get<int32_t>();
    LocalTensor<int16_t> validExpertIndexTensor = tempBuf.Get<int16_t>();
    LocalTensor<uint32_t> dstoffset = gatherMaskTBuf_.Get<uint32_t>();
    

    CreateVecIndex(expertsIndexTensor, static_cast<int16_t>(0), calCnt);
    //SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    for(uint32_t Index = startExpertId; Index < endExpertId; Index++) {
        int16_t dstExpertId = static_cast<uint16_t>(Index);
        Cast(expertIdsInt16, expertIdsTensor_, RoundMode::CAST_NONE, calCnt); // todo: 需要单独存放
        ////LOG_INFO("dstExpertId", dstExpertId);
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("expertIdsInt16", expertIdsInt16, {activeBsCnt, axisK_});
        Adds(dstExpIdTensorInt16, expertIdsInt16, -dstExpertId, calCnt);
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("dstExpIdTensorInt16",dstExpIdTensorInt16,{activeBsCnt, axisK_});
        Abs(dstExpIdTensorFP16, dstExpIdTensorFP16, calCnt);
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("dstExpIdTensorInt16",dstExpIdTensorInt16,{activeBsCnt, axisK_});
        Mins(dstExpIdTensorInt16, dstExpIdTensorInt16, 1, calCnt);
        Duplicate<int16_t>(XorTensorInt16, 1, calCnt); // todo:单独放
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("dstExpIdTensorInt16",dstExpIdTensorInt16,{activeBsCnt, axisK_});
        Xor(dstExpIdTensorInt16, dstExpIdTensorInt16, XorTensorInt16, sharedTensorInt8, calCnt);
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("dstExpIdTensorInt16",dstExpIdTensorInt16,{activeBsCnt, axisK_});
        Duplicate<uint16_t>(gatherMaskTensor, 0,
                        Ceil(calCnt, ALIGNED_LEN_256) * ALIGNED_LEN_256 / BITS_PER_BYTE / sizeof(uint16_t)); // 获取索引需要放在这里，因为需要256对齐
        Cast(dstExpIdTensorFP16, dstExpIdTensorInt16, RoundMode::CAST_RINT, calCnt);
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("dstExpIdTensorFP16",dstExpIdTensorFP16,{activeBsCnt, axisK_});
        CompareScalar(gatherMaskTensorInt8, dstExpIdTensorFP16, static_cast<half>(1), AscendC::CMPMODE::EQ, expertNumAlign256);
        PipeBarrier<PIPE_V>();
        GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor, true, calCnt, {1, 1, 0, 0},
                sendNumperExpert); // 当前需要发送的专家的索引以及需要发送的有效专家总数
        PipeBarrier<PIPE_V>();
        if(sendNumperExpert == 0){
            continue;
        }
        // //LOG_INFO("validExpertIndexTensor",validExpertIndexTensor,{1, static_cast<uint32_t>(sendNumperExpert)});
        // //LOG_INFO("sendNumperExpert",sendNumperExpert);
        Cast(tempOffset, validExpertIndexTensor, RoundMode::CAST_NONE, static_cast<uint32_t>(sendNumperExpert));
        PipeBarrier<PIPE_V>();
        ////LOG_INFO("tempOffset",tempOffset, {1, static_cast<uint32_t>(sendNumperExpert)});
        Muls(tempOffset, tempOffset, static_cast<int32_t>(sizeof(half)), static_cast<uint32_t>(sendNumperExpert));
        PipeBarrier<PIPE_V>();
        // //LOG_INFO("tempOffset",tempOffset, {1, static_cast<uint32_t>(sendNumperExpert)});
        // //LOG_INFO("dstoffset",dstoffset, {1, static_cast<uint32_t>(sendNumperExpert)});
        Scatter(expertOffsetCntTensor_, expertsIndexTensor, dstoffset, 0, static_cast<uint32_t>(sendNumperExpert));
        PipeBarrier<PIPE_V>();
    }
    uint64_t endtime = GetSystemCycle();
    uint64_t usetime = endtime -starttime;
    // //LOG_INFO("CalTokenSendExpertOffset use time",usetime);
    ////LOG_INFO("CalTokenSendExpertOffset expertOffsetCntTensor_",expertOffsetCntTensor_,{activeBsCnt, axisK_});
    WriteExpertOffset(); // 计算完成将结果写入GM并写flag
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::Process()
{   
    if ASCEND_IS_AIV { //todo:最开始清理server cnt flag
        CalTokenMask();
        if (aivId_ < aivUsedCumSum_) { //cnt计算 + cumsum
            CalCumSum();
        } else {
            CalTokenSendExpertOffset();
            if(aivId_ < aivUsedCumSum_ + aivUsedRemotComm_ + aivUsedScatter_) {
                ScatterTokenToRemote();  //4 20  5
            } else {
                GatherTokenFrRemote();
            }
        }
        //localWindowCopy中包含reset操作，需确保前面操作完成
        PipeBarrier<PIPE_ALL>();
        LocalWindowCopy();      // 本卡上专家数据连续化，输出expandX/scales/expandIdx
    }
}

} // namespace MoeDistributeDispatchV2HostKfcImpl
#endif // MOE_DISTRIBUTE_DISPATCH_V2_H