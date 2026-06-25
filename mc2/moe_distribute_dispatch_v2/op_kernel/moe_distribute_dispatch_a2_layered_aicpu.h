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
 * \file moe_distribute_dispatch_a2_layered_aicpu.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_AICPU_H
#define MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_AICPU_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_tiling.h"
#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

namespace MoeDistributeDispatchA2Impl {

#define TemplateMC2TypeA2layeredAicpuClass typename XType, typename ExpandXOutType,bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist
#define TemplateMC2TypeA2layeredAicpuFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist

using namespace AscendC;
template <TemplateMC2TypeA2layeredAicpuClass>
class MoeDistributeDispatchA2LayeredAicpu {
public:
    constexpr static uint32_t STATE_OFFSET = 512; // 状态空间偏移地址
    constexpr static uint32_t STATUS_SIZE_LAYERED = 1024 * 1024; // 1M
    constexpr static uint32_t B16_PER_BLOCK = 16;
    constexpr static uint32_t ONE_REPEAT_SORT_NUM = 32;
    constexpr static uint32_t SERVER_INFO_ALIGN = 512;
    constexpr static uint32_t RDMA_BUFFER_ALIGN = 4 * 1024;
    constexpr static uint32_t SELF_STATE_OFFSET = 512 * 1024; // 本卡状态空间偏移地址
    constexpr static uint32_t SERVER_RANK_SIZE = 8;
    constexpr static uint32_t INFO_NUM_IN_TOKENSTRUCK = 4; // 在Token后加入3种信息:expIds, weights, tokenIdx, scales
    constexpr static uint32_t STATUS_COUNT_OFFSET = 8; // count数组偏移为8保证对齐
    constexpr static uint32_t TYPE_MAP[4] = {static_cast<uint32_t>(AscendC::HcclDataType::HCCL_DATA_TYPE_INT8),
        static_cast<uint32_t>(AscendC::HcclDataType::HCCL_DATA_TYPE_FP16),
        static_cast<uint32_t>(AscendC::HcclDataType::HCCL_DATA_TYPE_FP32),
        static_cast<uint32_t>(AscendC::HcclDataType::HCCL_DATA_TYPE_INT64)};
    constexpr static uint32_t B64_PER_BLOCK = 4;
    constexpr static uint32_t PER_MSG_RDMA_SEND_TIME = 2;
    constexpr static uint32_t B32_PER_BLOCK = 8;
    constexpr static uint32_t UB_32B_ALIGN = 32;
    constexpr static uint32_t EXP_TOKEN_COUNT_FLAG_CNT = UB_32B_ALIGN / sizeof(int32_t);  // 8
    constexpr static uint32_t TBUF_SIZE = 190 * 1024;
    constexpr static uint32_t IPC_MAGIC_OFFSET = 2 * 1024 * 1024 - 64 * 32;
    constexpr static uint32_t IPC_FLAG_OFFSET = 1 * 1024 * 1024;
    constexpr static uint32_t IPC_TOKEN_CNT_OFFSET = 2 * 1024 * 1024;
    constexpr static uint32_t IPC_DATA_OFFSET = 4 * 1024 * 1024;
    constexpr static uint32_t WIN_SIZE_ALIGN = 1 * 1024 * 1024;
    constexpr static uint32_t IPC_BUFF_ALIGN = 512;
    constexpr static uint32_t TOKEN_COUNT_SIZE = 32;
    constexpr static uint32_t FLAG_U32_CNT = TOKEN_COUNT_SIZE / 4;
    constexpr static uint64_t  IPC_FLAG_STEP_1 = 1ULL;
    constexpr static uint64_t  IPC_FLAG_STEP_2 = 2ULL;
    constexpr static uint32_t TBUF_TEMP_OFFSET = 8 * 1024;
    constexpr static uint32_t TBUF_OFFSET_ALIGN = 2*1024;
    constexpr static uint32_t MAX_BS_NUM = 256;
    constexpr static uint32_t TBUF_OFFSET_ALIGN_B32_CNT = 2*1024 / sizeof(int32_t);
    constexpr static uint32_t RDMA_DATA_SIZE = 100U * 1024U * 1024U;
    constexpr static uint32_t EXTRA_TOKEN_INFO_NUM = 4U; // 专家信息 权重信息 量化Scale 到达标志位

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align) {
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

public:
    __aicore__ inline MoeDistributeDispatchA2LayeredAicpu() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expertScales, GM_ADDR expandXOut,
        GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut,
        GM_ADDR expandScales, GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM, GM_ADDR contextGM0);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReorderTokens();
    __aicore__ inline uint32_t GetExpRank(uint32_t expertId);
    __aicore__ inline bool IsInSameServer(uint32_t targetRankId);
    __aicore__ inline void SetTokenCnt(GlobalTensor<int32_t> globalSet);
    __aicore__ inline void CopyTokenToWinOut(uint32_t localTokenIdx, uint32_t tokenIdx, uint32_t dstServerId);
    __aicore__ inline void WaitWindow();
    __aicore__ inline void CreateInnerReduceInfo();
    __aicore__ inline void CreateOuterReduceInfo();

    __aicore__ inline void Win2Ipc();
    __aicore__ inline void Ipc2Out();
    __aicore__ inline void DispatchBetweenServer();
    __aicore__ inline void ConstructDataAndFlagBatchWriteInfo();
    __aicore__ inline void WaitIpcFlag(uint64_t flagVal = 1ULL);
    __aicore__ inline void SetIpcFlag(uint64_t flagVal = 1ULL);
    __aicore__ inline void GatherAndWriteCntInfo();
    __aicore__ inline void CleanUp();
    __aicore__ inline void QuantProcess(uint32_t sendTokenNum, LocalTensor<XType> xTokenLt,
                                        LocalTensor<float> tokenCastLt);
    __aicore__ inline uint64_t MergeMagicWithValue(uint64_t magic, uint64_t value);

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<float> weightsOutGt;
    GlobalTensor<uint64_t> dataBatchWriteInfoTensor_;
    GlobalTensor<int32_t> sendStatusTensor_;
    GlobalTensor<uint8_t> readTokensU8Tensor_;
    GlobalTensor<uint8_t> sendTokensU8Tensor_;
    GlobalTensor<uint32_t> sendTokensU32Tensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<uint32_t> expertToServerGlobalTensor_;
    GlobalTensor<int32_t> readStatusTensor_;

    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<uint64_t> batchWriteU64Tensor_;
    LocalTensor<uint32_t> batchWriteU32Tensor_;
    LocalTensor<uint32_t> expertToServerCntTensor_;
    LocalTensor<uint32_t> expertToServerIdxTensor_;

    TBuf<> expertCountBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> batchWriteInfoBuf_;
    TBuf<> expertToServerCntsBuf_;  // 总表，int类型只写1/0
    TBuf<> expertToServerIdxBuf_;
    TBuf<QuePosition::VECCALC> tBuf;

    GM_ADDR expandXGM_;
    GM_ADDR expandIdxGM_;
    GM_ADDR weightsGM_;
    GM_ADDR expertTokenNumsOutGM_;
    GM_ADDR epRecvCountsGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR dataBatchWriteInfo_;
    GM_ADDR expertToServerCntGM_;
    GM_ADDR shareAddrs[8];

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t globalBs_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};   // 真实的K值
    uint32_t alignK_{0};  // axisK_与 BITS32_PER_BLOCK 对齐
    uint32_t aivNum_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t aivId_{0}; // aiv id
    uint32_t moeExpertNum_{0}; // moe专家卡数, 等于worldSize_ - 共享专家卡数
    uint32_t moeExpertNumInServer_{0};
    uint32_t localMoeExpertNum_{0};
    uint32_t SERVER_SIZE_ON_WIN{0};
    uint32_t RANK_SIZE_ON_IPC{0};
    uint32_t WIN_SIZE{0};
    uint32_t bufferId_{0};
    uint32_t totalSize_{0};
    uint32_t totalWinSize_{0};
    uint32_t halfWinSize_{0};
    uint32_t serverNum{0};
    uint32_t expertTokenNumsType_{0};
    uint32_t shareMemOffset_{0};
    uint32_t tokenUbSize_{0};
    // TokenStruck相关
    uint32_t tokenGapInStruct_{0};
    uint32_t infoGapInStruct_{0};
    uint32_t tokenStructLen_{0};
    uint32_t tokenLenInStruct_{0};
    uint32_t expLenInStruct_{0};
    uint32_t weightLenInStruct_{0};
    uint32_t cntLenInStruct_{0};
    uint32_t scaleLenInStruct_{0};
    uint32_t realLenInStruct_{0};
    uint32_t tokenOffsetInStruct_{0};
    uint32_t expOffsetInStruct_{0};
    uint32_t weightOffsetInStruct_{0};
    uint32_t cntOffsetInStruct_{0};
    uint32_t scaleOffsetInStruct_{0};
    uint64_t magicVal_{0};

    uint64_t combineInnerCntOffset;
    uint64_t combineInnerCntIndexOffset;
    uint64_t combineOuterCntOffset;
    uint64_t combineOuterCntIndexOffset;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclA2CombineOpParam *winContext_{nullptr};
};

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut, GM_ADDR expandScales,
    GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM, GM_ADDR contextGM0)
{
    tpipe_ = pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);

    hccl_.InitV2(contextGM0, &tilingData);
    hccl_.SetCcTilingV2(offsetof(MoeDistributeDispatchA2TilingData, mc2CcTiling));

    winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
    rankId_ = tilingData.moeDistributeDispatchInfo.epRankId;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_);

    axisBS_ = tilingData.moeDistributeDispatchInfo.bs;
    globalBs_ = tilingData.moeDistributeDispatchInfo.globalBs;
    axisH_ = tilingData.moeDistributeDispatchInfo.h;
    axisK_ = tilingData.moeDistributeDispatchInfo.k;
    alignK_ = RoundUp(axisK_, BITS32_PER_BLOCK);
    aivNum_ = tilingData.moeDistributeDispatchInfo.aivNum;
    worldSize_ = tilingData.moeDistributeDispatchInfo.epWorldSize;
    moeExpertNum_ = tilingData.moeDistributeDispatchInfo.moeExpertNum;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    totalSize_ = winContext_->winSize;
    totalWinSize_ =  100 * 1024 * 1024; //RDMA 100 MB空间
    shareMemOffset_ = totalWinSize_;
    halfWinSize_ = totalWinSize_ / 2;
    WIN_SIZE = halfWinSize_ - STATUS_SIZE_LAYERED;
    expertTokenNumsType_ = tilingData.moeDistributeDispatchInfo.expertTokenNumsType;

    uint64_t winSizeMin = moeExpertNum_ * axisBS_ * (axisH_ * sizeof(XType) + EXTRA_TOKEN_INFO_NUM * alignK_ * sizeof(uint32_t)) +
        IPC_DATA_OFFSET + RDMA_DATA_SIZE; // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小

    for (int i = 0; i < SERVER_RANK_SIZE; i++) {
        shareAddrs[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(hccl_.GetWindowsInAddr(
            rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + shareMemOffset_));
    }

    // struct相关信息初始化计算
    tokenStructLen_ = axisH_ * sizeof(ExpandXOutType) + INFO_NUM_IN_TOKENSTRUCK * (axisK_ * sizeof(uint32_t));
    tokenLenInStruct_ = axisH_ * sizeof(ExpandXOutType);
    expLenInStruct_ = alignK_ * sizeof(uint32_t);
    weightLenInStruct_ = alignK_ * sizeof(uint32_t);
    cntLenInStruct_ = alignK_ * sizeof(uint32_t);
    scaleLenInStruct_ = UB_32B_ALIGN;
    realLenInStruct_ = axisK_ * sizeof(uint32_t);   // 内存中实际有效部分，跟 axisK_ 有关
    tokenStructLen_ = tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_ + cntLenInStruct_ + scaleLenInStruct_;
    tokenOffsetInStruct_ = 0;
    expOffsetInStruct_ = tokenLenInStruct_;
    weightOffsetInStruct_ = tokenLenInStruct_ + expLenInStruct_;
    cntOffsetInStruct_ = tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_;
    scaleOffsetInStruct_ = tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_ + cntLenInStruct_;
    tokenGapInStruct_ = (tokenStructLen_ - tokenLenInStruct_) / UB_32B_ALIGN;
    infoGapInStruct_ = (tokenStructLen_ - expLenInStruct_) / UB_32B_ALIGN;

    RANK_SIZE_ON_IPC = (totalSize_ - totalWinSize_ - IPC_DATA_OFFSET) / (localMoeExpertNum_ * worldSize_);
    RANK_SIZE_ON_IPC = (RANK_SIZE_ON_IPC / IPC_BUFF_ALIGN) * IPC_BUFF_ALIGN;

    aivId_ = GetBlockIdx();
    expertIdsCnt_ = axisBS_ * axisK_;
    serverNum = worldSize_ / SERVER_RANK_SIZE;
    SERVER_SIZE_ON_WIN = WIN_SIZE / serverNum;
    SERVER_SIZE_ON_WIN = (SERVER_SIZE_ON_WIN / RDMA_BUFFER_ALIGN) * RDMA_BUFFER_ALIGN;

    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t*)(windowInGM_ + WIN_SIZE + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferChosenGlobal_(0);

    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = windowOutGM_ + halfWinSize_ * bufferId_;

    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType*)(expandXOut),
                                        worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float*)(dynamicScalesOut));

    weightsOutGt.SetGlobalBuffer((__gm__ float*)(expandScales));

    sendTokensU8Tensor_.SetGlobalBuffer((__gm__ uint8_t*)(windowOutGM_));
    readTokensU8Tensor_.SetGlobalBuffer((__gm__ uint8_t*)(windowInGM_));
    sendTokensU32Tensor_.SetGlobalBuffer((__gm__ uint32_t*)(windowOutGM_));
    sendStatusTensor_.SetGlobalBuffer((__gm__ int32_t*)(windowOutGM_ + WIN_SIZE));
    readStatusTensor_.SetGlobalBuffer((__gm__ int32_t*)(windowInGM_ + WIN_SIZE));

    expertTokenNumsOutGM_ = expertTokenNumsOut; // 无GlobalTensor
    epRecvCountsGM_ = epRecvCountsOut; // 无GlobalTensor
    statusSpaceGm_ = windowInGM_ + WIN_SIZE;

    expandXGM_ = x;
    expandIdxGM_ = expertIds;
    weightsGM_ = expertScales;

    dataBatchWriteInfo_ = workspaceGM;
    dataBatchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint64_t*)(dataBatchWriteInfo_),
                                            serverNum * PER_MSG_RDMA_SEND_TIME * B64_PER_BLOCK);

    expertToServerCntGM_ = dataBatchWriteInfo_ + serverNum * PER_MSG_RDMA_SEND_TIME * B64_PER_BLOCK * sizeof(uint64_t);
    expertToServerGlobalTensor_.SetGlobalBuffer((__gm__ uint32_t*)(expertToServerCntGM_),
                                                RoundUp(axisBS_ * serverNum, B32_PER_BLOCK));

    combineInnerCntOffset = localMoeExpertNum_ * serverNum * SERVER_RANK_SIZE * sizeof(int32_t);
    combineInnerCntIndexOffset = combineInnerCntOffset + globalBs_ * serverNum * sizeof(int16_t);
    combineOuterCntOffset = combineInnerCntIndexOffset + globalBs_ * axisK_ * serverNum * sizeof(int32_t);
    combineOuterCntIndexOffset = combineOuterCntOffset + axisBS_ * sizeof(int32_t);
    moeExpertNumInServer_ = SERVER_RANK_SIZE * localMoeExpertNum_;

    tpipe_->InitBuffer(batchWriteInfoBuf_, PER_MSG_RDMA_SEND_TIME * BW_ITEM_SIZE);

    batchWriteU64Tensor_ = batchWriteInfoBuf_.Get<uint64_t>();
    batchWriteU32Tensor_ = batchWriteU64Tensor_.template ReinterpretCast<uint32_t>();

    tpipe_->InitBuffer(statusBuf_, UB_32B_ALIGN);

    tpipe_->InitBuffer(expertToServerIdxBuf_, serverNum * sizeof(uint32_t));
    expertToServerIdxTensor_ = expertToServerIdxBuf_.Get<uint32_t>();

    tpipe_->InitBuffer(tBuf, TBUF_SIZE);

    uint32_t expertIdsSize = RoundUp(expertIdsCnt_ * static_cast<uint32_t>(sizeof(int32_t)), UB_32B_ALIGN);
    uint32_t expertCountSize = RoundUp(moeExpertNum_ * static_cast<uint32_t>(sizeof(int32_t)), UB_32B_ALIGN);
    uint32_t expertToServerCntSize = RoundUp(axisBS_ * serverNum * static_cast<uint32_t>(sizeof(int32_t)), UB_32B_ALIGN);
    uint32_t expertIdsOffset = TBUF_SIZE - expertIdsSize;
    uint32_t expertCountOffset = expertIdsOffset - expertCountSize;
    uint32_t expertToServerCntOffset = expertCountOffset - expertToServerCntSize;
    tokenUbSize_ = TBUF_SIZE - TBUF_TEMP_OFFSET - expertIdsSize - expertCountSize - expertToServerCntSize;

    expertIdsTensor_ = tBuf.GetWithOffset<int32_t>(axisBS_ * axisK_, expertIdsOffset);
    expertCountTensor_ = tBuf.GetWithOffset<int32_t>(moeExpertNum_, expertCountOffset);
    expertToServerCntTensor_ = tBuf.GetWithOffset<uint32_t>(RoundUp(axisBS_ * serverNum, B32_PER_BLOCK), expertToServerCntOffset);
    Duplicate<uint32_t>(expertToServerCntTensor_, 0, RoundUp(axisBS_ * serverNum, B32_PER_BLOCK));
    Duplicate<int32_t>(expertCountTensor_, 0, moeExpertNum_);

    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t*)(statusSpaceGm_ + SELF_STATE_OFFSET));
    int32_t state = selfStatusTensor(aivId_ * UB_32B_ALIGN);
    PipeBarrier<PIPE_ALL>();

    if (aivId_ == 0) {
        sendStatusTensor_.SetValue(0, FLAG_VALUE);
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(sendStatusTensor_);
    }

    LocalTensor<uint64_t> tempLocal = tBuf.Get<uint64_t>();

    // 每次调用magic++,用来区分不同轮次
    GlobalTensor<uint64_t> magicGt;
    magicGt.SetGlobalBuffer((__gm__ uint64_t*)(shareAddrs[rankId_ % SERVER_RANK_SIZE] + IPC_MAGIC_OFFSET) +
        aivId_ * UB_32B_ALIGN / sizeof(uint64_t));
    DataCopy(tempLocal, magicGt, UB_32B_ALIGN / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
    tempLocal(0) += 1ULL;
    magicVal_ = tempLocal(0);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(magicGt, tempLocal, UB_32B_ALIGN / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
}


template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::CreateInnerReduceInfo()
{
    // 最后serverNum个Core加入本函数
    uint32_t curServerId = aivNum_ - aivId_ - 1;
    uint32_t currServerExpBegin = rankId_ / 8 * moeExpertNumInServer_;    // 目标Server的起始专家
    uint32_t currServerExpEnd = currServerExpBegin + moeExpertNumInServer_; // 目标Server的结束专家
    uint32_t tokenOccurNum = 0;
    uint32_t expOccurNum = 0;
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    __gm__ uint8_t *tokenCntGlobalAddr;
    if (curServerId == rankId_ / SERVER_RANK_SIZE) {
        tokenCntGlobalAddr = (__gm__ uint8_t*)(windowOutGM_) + curServerId * SERVER_SIZE_ON_WIN;
    } else {
        tokenCntGlobalAddr = (__gm__ uint8_t*)(windowInGM_) + curServerId * SERVER_SIZE_ON_WIN;
    }
    GlobalTensor<int32_t> tokenCntGlobalTensor;
    tokenCntGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)(tokenCntGlobalAddr));
    uint32_t realBS = tokenCntGlobalTensor.GetValue(0);
    PipeBarrier<PIPE_ALL>();

    if(realBS == 0){
        uint32_t copyTokenNum = aivNum_ < globalBs_ ? aivNum_ : globalBs_;
        LocalTensor<int16_t> zeroTemp = tBuf.GetWithOffset<int16_t>(copyTokenNum * sizeof(int16_t), 0);
        Duplicate<int16_t>(zeroTemp, 0, RoundUp(copyTokenNum, B16_PER_BLOCK));
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<int16_t> combineInnerCnt;
        combineInnerCnt.SetGlobalBuffer((__gm__ int16_t*)(epRecvCountsGM_ + combineInnerCntOffset +
                                        globalBs_* curServerId * sizeof(int16_t)));
        DataCopyExtParams innerCntWriteCountsParams{1, static_cast<uint32_t>(copyTokenNum * sizeof(int16_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(combineInnerCnt, zeroTemp, innerCntWriteCountsParams);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    LocalTensor<int32_t> localUB = tBuf.GetWithOffset<int32_t>(RoundUp(realBS * alignK_, BITS32_PER_BLOCK),
        baseBuffOffset);

    baseBuffOffset += sizeof(int32_t) * RoundUp(realBS * alignK_, TBUF_OFFSET_ALIGN_B32_CNT);
    LocalTensor<int32_t> combineReduceInfo = tBuf.GetWithOffset<int32_t>(moeExpertNumInServer_ * realBS,
        baseBuffOffset);

    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNumInServer_ * realBS, TBUF_OFFSET_ALIGN_B32_CNT);
    LocalTensor<int32_t> expCntMap = tBuf.GetWithOffset<int32_t>(moeExpertNumInServer_, baseBuffOffset);

    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNumInServer_, TBUF_OFFSET_ALIGN_B32_CNT);
    LocalTensor<int32_t> tokenOffset = tBuf.GetWithOffset<int32_t>(RoundUp(realBS * axisK_, BITS32_PER_BLOCK),
        baseBuffOffset);

    baseBuffOffset += sizeof(int32_t) * RoundUp(realBS * axisK_, TBUF_OFFSET_ALIGN_B32_CNT);
    LocalTensor<int32_t> innerOffsetLt = tBuf.GetWithOffset<int32_t>(RoundUp(realBS * axisK_, BITS32_PER_BLOCK),
        baseBuffOffset);

    baseBuffOffset += sizeof(int32_t) * RoundUp(realBS * axisK_, TBUF_OFFSET_ALIGN_B32_CNT);
    LocalTensor<int16_t> innerCntLt = tBuf.GetWithOffset<int16_t>(RoundUp(realBS + aivNum_, B16_PER_BLOCK),
        baseBuffOffset);

    int32_t invalidInfoData = globalBs_;
    Duplicate<int32_t>(combineReduceInfo, invalidInfoData, moeExpertNumInServer_ * realBS);
    Duplicate<int32_t>(expCntMap, int32_t(0), moeExpertNumInServer_);
    Duplicate<int32_t>(tokenOffset, int32_t(0), realBS * axisK_);
    Duplicate<int16_t>(innerCntLt, 0, RoundUp(realBS + aivNum_, B16_PER_BLOCK));
    Duplicate<int32_t>(innerOffsetLt, 0, (realBS) * axisK_);

    for (uint32_t tokenIdx=0; tokenIdx < realBS; tokenIdx++) {
        uint32_t srcCopyOffset =TOKEN_COUNT_SIZE + tokenIdx * tokenStructLen_ + expOffsetInStruct_;
        uint32_t dstCopyOffset = expLenInStruct_ * tokenIdx;
        DataCopy(localUB[dstCopyOffset / sizeof(uint32_t)], tokenCntGlobalTensor[srcCopyOffset / sizeof(uint32_t)],
                expLenInStruct_ / sizeof(uint32_t));
    }

    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIdx=0; tokenIdx < realBS; tokenIdx++) {
        for (uint32_t expIdx = 0; expIdx < axisK_; expIdx++) {
            int32_t expId = localUB(tokenIdx * alignK_ + expIdx);
            if (expId >= currServerExpBegin && expId < currServerExpEnd) {
                int32_t expIdInServer = expId % moeExpertNumInServer_;
                uint32_t offsetInExp = expCntMap(expIdInServer);
                expCntMap(expIdInServer) += 1;
                combineReduceInfo(expIdInServer * realBS+ offsetInExp) = tokenIdx;
                tokenOffset(tokenIdx * axisK_ + expIdx) = offsetInExp;
            }
        }
    }

    for (uint32_t expIdx = 0; expIdx < moeExpertNumInServer_; expIdx++) {
        if (expIdx % localMoeExpertNum_ == 0) {
            continue;
        }
        expCntMap(expIdx) += expCntMap(expIdx - 1);
    }

    for (uint32_t expBlockId=0; expBlockId < moeExpertNumInServer_; expBlockId++) {
        uint32_t validCnt = (expBlockId % localMoeExpertNum_ == 0) ? expCntMap(expBlockId) : (expCntMap(expBlockId) -
            expCntMap(expBlockId-1));
        for (uint32_t tokenIdx=0; tokenIdx < validCnt; tokenIdx++) {
            int32_t tokenId = combineReduceInfo(expBlockId * realBS + tokenIdx);
            if (tokenId == invalidInfoData) {
                continue;
            }
            for (uint32_t expIdx = 0; expIdx < axisK_; expIdx++) {
                uint32_t expId = localUB(tokenId * alignK_ + expIdx);
                if (expId >= currServerExpBegin && expId < currServerExpEnd) {
                    uint32_t expIdInServer = expId % moeExpertNumInServer_;
                    uint32_t rankIdInServer = expIdInServer / localMoeExpertNum_;
                    combineReduceInfo(expIdInServer * realBS + tokenOffset(tokenId * axisK_ + expIdx)) = invalidInfoData;
                    innerCntLt(tokenOccurNum) += 1;
                    innerOffsetLt(expOccurNum) =
                        (expIdInServer % localMoeExpertNum_== 0) ? 0 : expCntMap(expIdInServer - 1);
                    innerOffsetLt(expOccurNum) += rankIdInServer * globalBs_ * axisK_;
                    innerOffsetLt(expOccurNum) += tokenOffset(tokenId * axisK_ + expIdx);
                    expOccurNum += 1;
                }
            }
            tokenOccurNum += 1;
        }
    }
    for (uint32_t tokenIdx = 1; tokenIdx < realBS; ++tokenIdx) {
        innerCntLt(tokenIdx) += innerCntLt(tokenIdx - 1);
    }

    GlobalTensor<int16_t> combineInnerCnt;
    combineInnerCnt.SetGlobalBuffer((__gm__ int16_t*)(epRecvCountsGM_ + combineInnerCntOffset +
                                                      globalBs_* curServerId * sizeof(int16_t)));
    uint32_t copyTokenNum = (realBS + aivNum_) < globalBs_ ? (realBS + aivNum_) : globalBs_;
    DataCopyExtParams innerCntWriteCountsParams{1, static_cast<uint32_t>(copyTokenNum * sizeof(int16_t)), 0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineInnerCnt, innerCntLt, innerCntWriteCountsParams);
    PipeBarrier<PIPE_ALL>(); // 不确定连续两个GMdatacopypad是否会有影响，先隔离
    GlobalTensor<int32_t> combineInnerOffset;
    combineInnerOffset.SetGlobalBuffer((__gm__ int32_t*)(epRecvCountsGM_ + combineInnerCntIndexOffset +
                                                 globalBs_* axisK_ * curServerId * sizeof(int32_t)));

    DataCopyExtParams innerOffsetWriteCountsParams{1, static_cast<uint32_t>(realBS * axisK_ * sizeof(int32_t)),
                                                   0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineInnerOffset, innerOffsetLt, innerOffsetWriteCountsParams);
    PipeBarrier<PIPE_ALL>();
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::CreateOuterReduceInfo()
{
    // 仅最后一个核进去该逻辑
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;

    LocalTensor<int32_t> miniExpIds = tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> miniServerExpIds = tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK),
        baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * serverNum, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> combineCnt_ = tBuf.GetWithOffset<int32_t>(moeExpertNum_, baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNum_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> combineCntIdx_ = tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_, BITS32_PER_BLOCK),
        baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> combineOffset_ = tBuf.GetWithOffset<int32_t>(moeExpertNum_, baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNum_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> combineOffsetIdx_ = tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK),
        baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * serverNum, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> outerCntLt = tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> outerOffsetLt = tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * axisK_, BITS32_PER_BLOCK),
        baseBuffOffset);

    DataCopyExtParams expCopyParams{1, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> expPadParams;
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expCopyParams, expPadParams);

    Duplicate<int32_t>(miniExpIds, int32_t(moeExpertNum_), RoundUp(axisBS_, BITS32_PER_BLOCK));
    Duplicate<int32_t>(miniServerExpIds, int32_t(moeExpertNum_), RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK));
    Duplicate<int32_t>(combineCnt_, int32_t(0), moeExpertNum_);
    Duplicate<int32_t>(combineOffset_, int32_t(0), moeExpertNum_);
    Duplicate<int32_t>(outerCntLt, 0, RoundUp(axisBS_, BITS32_PER_BLOCK));
    Duplicate<int32_t>(outerOffsetLt, 0, RoundUp(axisBS_ * axisK_, BITS32_PER_BLOCK));

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    SyncFunc<AscendC::HardEvent::V_S>();

    // ServerIdx，统计token去往了哪些server，以及在server上的偏移，统计目的专家信息
    for (uint32_t expertIndex = 0; expertIndex < expertIdsCnt_; ++expertIndex) {
        uint32_t tokenIdx = expertIndex / axisK_;
        uint32_t expId = expertIdsTensor_(expertIndex);
        uint32_t expServerId = expId / moeExpertNumInServer_; // 专家在第几个server

        // 获取当前token中最小的一个expId,用于后续计算该token出现的位置
        uint32_t miniExpId = miniExpIds(tokenIdx);
        miniExpIds(tokenIdx) = (expId < miniExpId) ? expId : miniExpId;

        // 当前token每个目的server,统计其最小expId
        if (miniServerExpIds(tokenIdx * serverNum + expServerId) > expId) {
            miniServerExpIds(tokenIdx * serverNum + expServerId) = expId;
        }

        if (expertIndex % axisK_ != axisK_ - 1) {
            continue;
        }
        // token的最后一个expID，将上述信息进行记录
        combineCntIdx_(tokenIdx) = combineCnt_(miniExpId);
        combineCnt_(miniExpId) += 1;

        for (uint32_t serverIdx = 0; serverIdx < serverNum; ++serverIdx) {
            uint32_t miniServerExpId = miniServerExpIds(tokenIdx * serverNum + serverIdx);
            if (miniServerExpId != moeExpertNum_) {
                combineOffsetIdx_(tokenIdx * serverNum + serverIdx) = combineOffset_(miniServerExpId);
                combineOffset_(miniServerExpId) += 1;
            }
        }
    }
    // 计算前序和
    for (uint32_t expertIndex = 1; expertIndex < moeExpertNum_; ++expertIndex) {
        combineCnt_(expertIndex) += combineCnt_(expertIndex - 1);
        combineOffset_(expertIndex) += combineOffset_(expertIndex - 1);
    }

    // 第三次遍历，填充bs个token的Reduceinfo
    uint32_t outerOffsetIdx = 0;
    for (uint32_t tokenIdx = 0; tokenIdx < axisBS_; ++tokenIdx) {
        uint32_t miniExpId = miniExpIds(tokenIdx);
        // 将cnt,offset填写到InfoTensor对应的位置
        for (uint32_t serverIdx = 0; serverIdx < serverNum; ++serverIdx) {
            // 对于无效server跳过
            uint32_t miniServerExpId = miniServerExpIds(tokenIdx * serverNum + serverIdx);
            if (miniServerExpId == moeExpertNum_) {
                continue;
            }
            outerCntLt(tokenIdx) += 1;
            uint32_t preServerCnt = (serverIdx == 0) ? 0 : combineOffset_(serverIdx * moeExpertNumInServer_ -1);
            uint32_t serverBaseCnt = serverIdx * axisBS_;
            uint32_t preTokenCnt = (miniServerExpId == 0)? 0 : combineOffset_(miniServerExpId - 1);
            uint32_t tokenOffset = preTokenCnt - preServerCnt + combineOffsetIdx_(tokenIdx * serverNum + serverIdx) +
                serverBaseCnt;
            outerOffsetLt(outerOffsetIdx) = tokenOffset;
            outerOffsetIdx++;
        }
    }

    // 第四次遍历获取累加和
    for (uint32_t tokenIdx = 1; tokenIdx < axisBS_; ++tokenIdx) {
        outerCntLt(tokenIdx) += outerCntLt(tokenIdx - 1);
    }

    GlobalTensor<int32_t> combineOuterCnt;
    combineOuterCnt.SetGlobalBuffer((__gm__ int32_t*)(epRecvCountsGM_ + combineOuterCntOffset));

    DataCopyExtParams outerCntWriteCountsParams{1, static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineOuterCnt, outerCntLt, outerCntWriteCountsParams);

    PipeBarrier<PIPE_ALL>(); // 不确定连续两个GMdatacopypad是否会有影响，先隔离

    GlobalTensor<int32_t> combineOuterOffset;
    combineOuterOffset.SetGlobalBuffer((__gm__ int32_t*)(epRecvCountsGM_ + combineOuterCntIndexOffset));

    DataCopyExtParams outerOffsetWriteCountsParams{1, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(int32_t)),
        0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineOuterOffset, outerOffsetLt, outerOffsetWriteCountsParams);
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::ReorderTokens()
{
    uint32_t sendTokenNum = axisBS_ / aivNum_;
    uint32_t remainderTokenNum = axisBS_ % aivNum_;
    uint32_t startTokenId = sendTokenNum * aivId_;
    // 分核，每个Core处理sendTokenNum个Token的遍历
    if (aivId_ < remainderTokenNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    uint32_t endTokenId = startTokenId + sendTokenNum;

    if (sendTokenNum == 0) {
        return;
    }
    uint32_t dstServerId = 0;
    uint32_t tokenIndex = 0;
    // singleTokenUBSize 单token需要UB字节数
    uint32_t singleTokenUBSize = tokenStructLen_ > axisH_ * sizeof(XType) ? tokenStructLen_ : axisH_ * sizeof(XType);
    uint32_t quantTokenUBSize = 0;  // 量化时单token额外需要的UB字节数
    if constexpr (DynamicQuant || StaticQuant) {
        quantTokenUBSize = axisH_ * sizeof(float);
    }
    uint32_t maxTokenNumInUB = tokenUbSize_ / (singleTokenUBSize + quantTokenUBSize);
    uint32_t batchNum = (sendTokenNum + maxTokenNumInUB - 1) / maxTokenNumInUB;
    // 这几个tensor是相同的地址空间，只是数据类型不一样
    LocalTensor<uint8_t> tokenTempTensorU8_ =
        tBuf.GetWithOffset<uint8_t>(((singleTokenUBSize * maxTokenNumInUB) / sizeof(uint8_t)), TBUF_TEMP_OFFSET);
    LocalTensor<uint32_t> tokenTempTensorU32_ =
        tBuf.GetWithOffset<uint32_t>(((singleTokenUBSize * maxTokenNumInUB) / sizeof(uint32_t)), TBUF_TEMP_OFFSET);
    LocalTensor<XType> tokenLt =
        tBuf.GetWithOffset<XType>(((singleTokenUBSize * maxTokenNumInUB) / sizeof(XType)), TBUF_TEMP_OFFSET);

    // 仅用于量化cast
    LocalTensor<float> tokenCastLt = tBuf.GetWithOffset<float>(axisH_ * sendTokenNum,
        RoundUp(TBUF_TEMP_OFFSET + singleTokenUBSize * maxTokenNumInUB, B32_PER_BLOCK));
    GlobalTensor<uint8_t> xGMTensorU8_;
    xGMTensorU8_.SetGlobalBuffer((__gm__ uint8_t*)expandXGM_);
    GlobalTensor<uint8_t> expertIdsGMTensorU8_;
    expertIdsGMTensorU8_.SetGlobalBuffer((__gm__ uint8_t*)expandIdxGM_);

    GlobalTensor<uint32_t> expertIdsGMTensorU32_;
    expertIdsGMTensorU32_.SetGlobalBuffer((__gm__ uint32_t*)expandIdxGM_);

    GlobalTensor<uint8_t> weightGt;
    weightGt.SetGlobalBuffer((__gm__ uint8_t*)weightsGM_);

    DataCopyExtParams expCopyParams{1, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> expPadParams;
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expCopyParams, expPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t batchIndex = 0; batchIndex < batchNum; batchIndex++) {
        uint32_t currentTokenNum = sendTokenNum > maxTokenNumInUB ? maxTokenNumInUB : sendTokenNum;
        // token进行拷贝(量化)
        if constexpr (DynamicQuant || StaticQuant) {
            DataCopy(tokenTempTensorU8_, xGMTensorU8_[startTokenId * axisH_ * sizeof(XType)],
                currentTokenNum * axisH_ * sizeof(XType));
            QuantProcess(currentTokenNum, tokenLt, tokenCastLt);
        } else {
            DataCopyExtParams tokenCopyParams{static_cast<uint16_t>(currentTokenNum),
                 static_cast<uint32_t>(axisH_ * sizeof(XType)), 0, static_cast<uint32_t>(tokenGapInStruct_), 0};
            DataCopyPadExtParams<uint8_t> tokenPadParams;
            DataCopyPad(tokenTempTensorU8_, xGMTensorU8_[startTokenId * tokenLenInStruct_],
                tokenCopyParams, tokenPadParams);
        }
        // Expert进行拷贝
        DataCopyExtParams expCopyParams{static_cast<uint16_t>(currentTokenNum), static_cast<uint32_t>(realLenInStruct_),
            0, static_cast<uint32_t>(infoGapInStruct_), 0};
        DataCopyPadExtParams<uint8_t> expPadParams;
        DataCopyPad(tokenTempTensorU8_[expOffsetInStruct_], expertIdsGMTensorU8_[startTokenId * realLenInStruct_],
                    expCopyParams, expPadParams);
        // Weights进行拷贝
        DataCopyExtParams weightCopyParams{static_cast<uint16_t>(currentTokenNum), static_cast<uint16_t>(realLenInStruct_),
            0, static_cast<uint16_t>(infoGapInStruct_), 0};
        DataCopyPadExtParams<uint8_t> weightPadParams;
        DataCopyPad(tokenTempTensorU8_[weightOffsetInStruct_], weightGt[startTokenId * realLenInStruct_],
                    weightCopyParams, weightPadParams);
        uint32_t startExpId = startTokenId * axisK_;
        uint32_t endExpId = startExpId + currentTokenNum * axisK_;
        int32_t currentSum = 0;
        if (batchIndex == 0) {
            for (uint32_t expertIndex = 0; expertIndex < startExpId; ++expertIndex) {
                tokenIndex = expertIndex / axisK_;
                uint32_t expertId = (uint32_t)(expertIdsTensor_(expertIndex));   // 读取expId
                currentSum = expertCountTensor_(expertId);  // 读取已经往该Exp发送的token个数
                uint32_t dstServerId = expertId / moeExpertNumInServer_;  // 该token去往哪个Server
                // 覆盖写，确保去往同一个Server，该Token只统计一次
                expertToServerCntTensor_(tokenIndex * serverNum + dstServerId) = (int8_t)1;
                expertCountTensor_(expertId) = currentSum + 1;
            }
        }
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t expertIndex = startExpId; expertIndex < endExpId; ++expertIndex) {
            // 本Aiv要处理的token信息需要特殊计算
            tokenIndex = expertIndex / axisK_;
            uint32_t expertId = (uint32_t)(expertIdsTensor_(expertIndex));
            currentSum = expertCountTensor_(expertId);
            uint32_t dstServerId = expertId / moeExpertNumInServer_;
            expertToServerCntTensor_(tokenIndex * serverNum + dstServerId) = (int8_t)1;
            expertCountTensor_(expertId) = currentSum + 1;

            uint32_t sendTokenIdx = expertIndex / axisK_ - startTokenId;
            // 当前处理的Token在UB中的第几个
            uint32_t curCntOffset = (sendTokenIdx * tokenStructLen_ + cntOffsetInStruct_) / sizeof(uint32_t);
            // 索引到Cnt的位置
            tokenTempTensorU32_(curCntOffset + expertIndex % axisK_) = currentSum;
            // 写去往该Exp的token数,写在第ExpIdx位置
            expertToServerIdxTensor_(dstServerId) = expertIndex;
            // 覆盖写，记录本token中最后一个想去往该Server的Expidx
            if (expertIndex % axisK_ != axisK_ - 1) {
                continue;
            }
            // 轮询到需要搬移的token的最后一个ExpId，则进行数据拷贝逻辑
            for (uint32_t reviewExpIdx = expertIndex + 1 - axisK_; reviewExpIdx < expertIndex + 1; reviewExpIdx++) {
                uint32_t reviewExpertId = (uint32_t)(expertIdsTensor_(reviewExpIdx));
                // 往前回顾axisK_个Expid
                uint32_t reviewServerId = reviewExpertId / moeExpertNumInServer_;
                if (expertToServerIdxTensor_(reviewServerId) == reviewExpIdx) {
                    PipeBarrier<PIPE_ALL>();
                    CopyTokenToWinOut(sendTokenIdx, reviewExpIdx / axisK_, reviewServerId);
                }
            }
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        if (endExpId == expertIdsCnt_) {
            // 最后一个token的最后一个ExpId，需要负责写每个token去往每个Server的信息表
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopy(expertToServerGlobalTensor_, expertToServerCntTensor_,
                RoundUp(axisBS_ * serverNum, B32_PER_BLOCK));
        }
        startTokenId += currentTokenNum;
        sendTokenNum -= currentTokenNum;
    }
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::
    QuantProcess(uint32_t sendTokenNum, LocalTensor<XType> xTokenLt, LocalTensor<float> tokenCastLt) {
    constexpr uint32_t maxArrUbOffset = 6 * 1024;
    constexpr uint32_t maxArrLen = 3;
    constexpr uint32_t maxValOffset = 0;
    constexpr uint32_t minValOffset = 1;
    constexpr uint32_t resValOffset = 2;
    constexpr float quantMax = 127.0f;
    const half deqScale = static_cast<half>(1.000000e+00f);
    float dynamicScale = 0.0;
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> workLt = tBuf.GetWithOffset<float>(maxArrUbOffset / sizeof(float), 0);
    LocalTensor<float> maxLt = tBuf.GetWithOffset<float>(maxArrLen, maxArrUbOffset);
    Cast(tokenCastLt, xTokenLt, RoundMode::CAST_NONE, sendTokenNum * axisH_);
    for (int32_t i = 0; i < sendTokenNum; ++i) {
        PipeBarrier<PIPE_V>();
        if constexpr(DynamicQuant) {
            ReduceMax(maxLt[maxValOffset], tokenCastLt[i * axisH_], workLt, axisH_, false);
            SyncFunc<AscendC::HardEvent::V_S>();
            PipeBarrier<PIPE_V>();
            ReduceMin(maxLt[minValOffset], tokenCastLt[i * axisH_], workLt, axisH_, false);
            PipeBarrier<PIPE_V>();
            Abs(maxLt, maxLt, maxArrLen - 1);
            PipeBarrier<PIPE_V>();
            ReduceMax(maxLt[resValOffset], maxLt, workLt, maxArrLen - 1, false);

            SyncFunc<AscendC::HardEvent::V_S>();
            float maxVal = maxLt(resValOffset);
            dynamicScale = float(quantMax) / float(maxVal);
            SyncFunc<AscendC::HardEvent::S_V>();
            Muls(tokenCastLt[i * axisH_], tokenCastLt[i * axisH_], dynamicScale, axisH_);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<half> halfLocalTemp = tokenCastLt[i * axisH_].template ReinterpretCast<half>();
        LocalTensor<int32_t> int32LocalTemp = tokenCastLt[i * axisH_].template ReinterpretCast<int32_t>();
        Cast(int32LocalTemp, tokenCastLt[i * axisH_], RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
        SetDeqScale(deqScale);
        PipeBarrier<PIPE_V>();

        Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

        PipeBarrier<PIPE_V>();
        LocalTensor<ExpandXOutType> xOutTensor;
        LocalTensor<uint8_t> tokenUnitLt;
        tokenUnitLt = xTokenLt.template ReinterpretCast<uint8_t>();
        xOutTensor = tokenUnitLt[i * tokenStructLen_].template ReinterpretCast<ExpandXOutType>();
        Cast(xOutTensor, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

        LocalTensor<float> scaleTensor = tokenUnitLt[i * tokenStructLen_ +
                                                    scaleOffsetInStruct_].template ReinterpretCast<float>();
        scaleTensor.SetValue(0, float(1.0) / dynamicScale); // int8->float32
    }
}


template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::CopyTokenToWinOut(uint32_t localTokenIdx,
    uint32_t globalTokenIdx, uint32_t dstServerId)
{
    uint32_t curServerId = rankId_ / SERVER_RANK_SIZE;
    uint32_t toServerCntSum = 0;
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIdx = 0; tokenIdx < globalTokenIdx ; tokenIdx++) {
        uint32_t tensorOffset = tokenIdx * serverNum + dstServerId;
        toServerCntSum += expertToServerCntTensor_(tensorOffset);
    }

    LocalTensor<uint8_t> tokenTempTensorU8_ =
        tBuf.GetWithOffset<uint8_t>(tokenUbSize_, TBUF_TEMP_OFFSET);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    uint32_t destOffset = dstServerId * SERVER_SIZE_ON_WIN + tokenStructLen_ * toServerCntSum + TOKEN_COUNT_SIZE;
    DataCopy(sendTokensU8Tensor_[destOffset], tokenTempTensorU8_[localTokenIdx * tokenStructLen_],
        tokenStructLen_);
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::GatherAndWriteCntInfo()
{
    uint32_t destServerNum = serverNum / aivNum_;  // 每个AIV要处理的server数
    uint32_t remaServerNum = serverNum % aivNum_;
    uint32_t startServerId = destServerNum * aivId_;
    if (aivId_ < remaServerNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        destServerNum += 1;
        startServerId += aivId_;
    } else {
        startServerId += remaServerNum;
    }
    if (destServerNum == 0) {
        return;
    }
    DataCopy(expertToServerCntTensor_, expertToServerGlobalTensor_, RoundUp(axisBS_ * serverNum, B32_PER_BLOCK));
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t dstServerId = startServerId; dstServerId < startServerId + destServerNum; ++dstServerId) {
        uint32_t dstServerCnt = 0;

        for (uint32_t tokenIdx = 0; tokenIdx <axisBS_ ; ++tokenIdx) {
            dstServerCnt += expertToServerCntTensor_(serverNum * tokenIdx + dstServerId);
        }
        PipeBarrier<PIPE_ALL>();
        expertToServerIdxTensor_(dstServerId)=dstServerCnt;
        LocalTensor<uint32_t> writeCntLt = tBuf.GetWithOffset<uint32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
        writeCntLt.SetValue(0, dstServerCnt);
        uint32_t destOffset = (dstServerId * SERVER_SIZE_ON_WIN) / sizeof(uint32_t);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(sendTokensU32Tensor_[destOffset], writeCntLt, EXP_TOKEN_COUNT_FLAG_CNT);
    }
}

// 构建发往其他server的所有data报文
template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::ConstructDataAndFlagBatchWriteInfo()
{
    // 计算当前core要处理的server
    uint32_t batchWriteItemNum = serverNum / aivNum_;  // 一个aiv负责的server数量
    uint32_t remainderItemNum = serverNum % aivNum_;  // 多出来的server没人处理
    uint32_t startServerId = batchWriteItemNum * aivId_;  // 当前aiv负责[startServerId,endServerId)个server
    uint32_t curServerId = rankId_ / SERVER_RANK_SIZE;  // 当前serverId

    if (aivId_ < remainderItemNum) {
        startServerId += aivId_; // aiv0:1*0+0=0，aiv1:1*1+1=2，aiv2:1*2+2=4，... aiv23:1*23+23=46，
        batchWriteItemNum += 1; // 前remainderItemNum个aiv需要多处理1个server的数据
    } else {
        startServerId += remainderItemNum;  // aiv24:1*24+24=48, aiv25:1*25+24=49
    }
    uint32_t endServerId = startServerId + batchWriteItemNum;
    if (batchWriteItemNum == 0) {
        return;
    }
    // 当前aiv负责 [startServerId,endServerId) 个 server
    for (uint32_t dstserverInd = startServerId; dstserverInd < endServerId; ++dstserverInd) {
        uint32_t sendIdx = dstserverInd - startServerId;
        uint32_t dstRankId = rankId_ % SERVER_RANK_SIZE + dstserverInd *  SERVER_RANK_SIZE;  // 目标Rank
        PipeBarrier<PIPE_ALL>();
        uint64_t dstDataRdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + halfWinSize_ * bufferId_ +
            curServerId * SERVER_SIZE_ON_WIN);
        // src卡GetWindowsInAddr地址, 要发给serverIndex，即是本端的rdma地址
        uint64_t srcDataRdmaAddr = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ +
            dstserverInd * SERVER_SIZE_ON_WIN);
        // 去往该Server的传输的数据量
        uint32_t validTokenCount = expertToServerIdxTensor_(dstserverInd);
        uint32_t validDataLength = TOKEN_COUNT_SIZE + validTokenCount * tokenStructLen_;
        uint64_t winInAddr = (uint64_t)(hccl_.GetWindowsInAddr(rankId_));
        uint64_t winOutAddr = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_));
        PipeBarrier<PIPE_ALL>();
        batchWriteU64Tensor_(0) = srcDataRdmaAddr;     // 源地址
        batchWriteU64Tensor_(1) = dstDataRdmaAddr;  // 目的地址
        batchWriteU64Tensor_(2) = validDataLength;   // 数据长度
        batchWriteU32Tensor_(6) = HcclDataType::HCCL_DATA_TYPE_INT8;
        batchWriteU32Tensor_(7) = dstRankId;        // dst卡

        uint64_t dstFlagRdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + halfWinSize_ * bufferId_ +
            WIN_SIZE + curServerId * STATE_OFFSET);

        // src卡，即是本端的rdma地址
        uint64_t srcFlagRdmaAddr = (uint64_t)(sendStatusTensor_.GetPhyAddr());
        uint32_t flagLen = TOKEN_COUNT_SIZE;
        PipeBarrier<PIPE_ALL>();
        batchWriteU64Tensor_(4) = srcFlagRdmaAddr;      // 源地址
        batchWriteU64Tensor_(5) = dstFlagRdmaAddr;   // 目的地址
        batchWriteU64Tensor_(6) = flagLen;      // 数据长度
        batchWriteU32Tensor_(14) = HcclDataType::HCCL_DATA_TYPE_INT8;
        batchWriteU32Tensor_(15) = dstRankId;          // dst卡

        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t dstServerOffset = dstserverInd;
        uint32_t sendInfoCount = B64_PER_BLOCK * PER_MSG_RDMA_SEND_TIME;
        DataCopy(dataBatchWriteInfoTensor_[dstServerOffset * sendInfoCount], batchWriteU64Tensor_, sendInfoCount);
    }
}

// 机间同平面RDMA通信
template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::DispatchBetweenServer()
{
    ConstructDataAndFlagBatchWriteInfo();
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    if ASCEND_IS_AIV {
        if (aivId_ == 0) {
            HcclHandle batchWriteResultData = hccl_.BatchWrite<true>((GM_ADDR)(dataBatchWriteInfoTensor_.GetPhyAddr()),
                                                                    serverNum * PER_MSG_RDMA_SEND_TIME);
            bufferChosenGlobal_(0) = bufferId_ ^ 1;
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(bufferChosenGlobal_);
        }
        if (aivId_ == aivNum_ - 1) {
            CreateOuterReduceInfo();
        }
    }
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline uint32_t MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::GetExpRank(uint32_t expertId)
{
    return expertId / localMoeExpertNum_;
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline bool MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::IsInSameServer(uint32_t targetRankId)
{
    return targetRankId / SERVER_RANK_SIZE == rankId_ / SERVER_RANK_SIZE;
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline uint64_t MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::MergeMagicWithValue(uint64_t magic,
                                                                                                uint64_t value)
{
    return (magic * 2ULL + value);
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::SetIpcFlag(uint64_t flagVal)
{
    if (aivId_ >= SERVER_RANK_SIZE) {
        return;
    }
    uint32_t destRankIdx = aivId_;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<uint64_t> globalSet;
    globalSet.SetGlobalBuffer((__gm__ uint64_t*)(shareAddrs[destRankIdx] + IPC_FLAG_OFFSET) +
        localRankId * B64_PER_BLOCK);
    LocalTensor<uint64_t> localSet = tBuf.GetWithOffset<uint64_t>(B64_PER_BLOCK, 0);
    uint64_t setVal = MergeMagicWithValue(magicVal_, flagVal);
    localSet.SetValue(0, setVal);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(globalSet, localSet, B64_PER_BLOCK);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::WaitIpcFlag(uint64_t flagVal)
{
    uint64_t waitVal = MergeMagicWithValue(magicVal_, flagVal);
    if (aivId_ >= SERVER_RANK_SIZE) {
        return;
    }
    LocalTensor<uint64_t> localWait = tBuf.GetWithOffset<uint64_t>(B64_PER_BLOCK, 0);
    bool isSync = true;
    uint32_t destRankIdx = aivId_;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<uint64_t> flagIpcGt;
    flagIpcGt.SetGlobalBuffer((__gm__ uint64_t*)(shareAddrs[localRankId] + IPC_FLAG_OFFSET) +
        destRankIdx * B64_PER_BLOCK);
    PipeBarrier<PIPE_ALL>();
    do {
        DataCopy(localWait, flagIpcGt, B64_PER_BLOCK);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        // 当有core未达到checkValue的阶段时，继续等待
        uint64_t tempVal = localWait.GetValue(0);
        if (tempVal >= waitVal) {
            break;
        }
    } while (isSync);
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::SetTokenCnt(GlobalTensor<int32_t> globalSet)
{
    AscendC::SetAtomicAdd<int32_t>();
    LocalTensor<int32_t> localSet = tBuf.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
    localSet(0) = 1;    // AtomicAdd每次+1
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(globalSet, localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::WaitWindow()
{
    // 前ServerNum个卡进行等待，其中等待本服务器的卡可以直接return
    if (aivId_ >= serverNum || aivId_ == (rankId_ / SERVER_RANK_SIZE)) {
        return;
    }
    uint32_t waitFlagIdx = aivId_;
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> statusTensor = statusBuf_.Get<int32_t>();
    while (true) {
        DataCopy(statusTensor, readStatusTensor_[(waitFlagIdx) * STATE_OFFSET / sizeof(int32_t)], FLAG_U32_CNT);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        int32_t sumOfFlag = statusTensor.GetValue(0);
        if (sumOfFlag == FLAG_VALUE) {
            break;
        }
    }
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::Win2Ipc()
{
    uint32_t coresPerServer = (aivNum_ - serverNum) / serverNum; // 48/2 = 24
    if (aivId_ >= coresPerServer * serverNum) {
        return;
    }
    // 计算本core需要处理的ServerId
    uint32_t formServerId = aivId_ / coresPerServer; // 前24处理0， 后24处理1

    // 获取tokenCnt,计算本卡收到对端server多少Token，用于后续分核计算
    __gm__ uint8_t *tokenCntGlobalAddr;
    if (formServerId == rankId_ / SERVER_RANK_SIZE) {
        tokenCntGlobalAddr = (__gm__ uint8_t*)(windowOutGM_) + formServerId * SERVER_SIZE_ON_WIN;
    } else {
        tokenCntGlobalAddr = (__gm__ uint8_t*)(windowInGM_) + formServerId * SERVER_SIZE_ON_WIN;
    }
    GlobalTensor<uint32_t> tokenCntGlobalTensor;
    tokenCntGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)(tokenCntGlobalAddr));
    LocalTensor<uint32_t> localWait = tBuf.GetWithOffset<uint32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);

    DataCopy(localWait, tokenCntGlobalTensor, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t tokenCnt = localWait.GetValue(0);

    GlobalTensor<uint8_t> targetTokenIpcGt;  // 对端IPC的TokenTensor，写数据用

    uint32_t WinInTokenOffset = formServerId * SERVER_SIZE_ON_WIN + TOKEN_COUNT_SIZE;
    uint32_t localAivId = aivId_ % coresPerServer; // 0,1，2,3...19
    // 平均每个核处理多少token
    uint32_t tokenCntPerAiv = tokenCnt / coresPerServer; // 16/20
    // 平分后剩下多少token
    uint32_t tokenCntRemain = tokenCnt % coresPerServer; // 16%20
    // 前面的核共分到了多少剩余
    uint32_t tokenCntPreRemain = (localAivId < tokenCntRemain) ? localAivId : tokenCntRemain; // 小于16为
    // 当前核分到多少token
    uint32_t tokenCntCurAiv = (localAivId < tokenCntRemain) ? (tokenCntPerAiv + 1) : tokenCntPerAiv;

    LocalTensor<uint8_t> localUB = tBuf.GetWithOffset<uint8_t>(tokenUbSize_ / sizeof(uint8_t),
        TBUF_TEMP_OFFSET);
    uint32_t tokenCntInUB = tokenUbSize_ / tokenStructLen_;
    // ceil div
    uint32_t batchCnt = (tokenCntCurAiv + tokenCntInUB - 1) / tokenCntInUB;
    for (uint32_t batchIdx = 0; batchIdx < batchCnt; ++batchIdx) {
        uint32_t tokenCntInBatch = tokenCntInUB;
        if (batchIdx == batchCnt - 1) {
            tokenCntInBatch = tokenCntCurAiv - (batchCnt - 1) * tokenCntInUB;
        }
        // 计算当前Core处理的Token偏移
        uint32_t tokenStructIdx = localAivId * tokenCntPerAiv + tokenCntPreRemain + batchIdx * tokenCntInUB;
        // 等待GM->UB
        if (formServerId == rankId_ / SERVER_RANK_SIZE) {
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            DataCopy(localUB, sendTokensU8Tensor_[WinInTokenOffset + tokenStructIdx * tokenStructLen_],
                    tokenCntInBatch * tokenStructLen_);
        } else {
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            DataCopy(localUB, readTokensU8Tensor_[WinInTokenOffset + tokenStructIdx * tokenStructLen_],
                tokenCntInBatch * tokenStructLen_);
        }
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t tokenIdx = 0; tokenIdx < tokenCntInBatch; ++tokenIdx) {
            // 逐个处理Token to Ipc
            uint32_t expPos = tokenIdx * tokenStructLen_ + expOffsetInStruct_;
            LocalTensor<uint32_t> expInfoTensor = localUB[expPos].ReinterpretCast<uint32_t>();
            // 当前Token的ExpIds信息
            uint32_t tokenCntPos = tokenIdx * tokenStructLen_ + cntOffsetInStruct_;
            LocalTensor<uint32_t> cntInfoTensor = localUB[tokenCntPos].ReinterpretCast<uint32_t>();
            // 当前Token的Cnt信息
            for (uint32_t expIdx = 0; expIdx < axisK_; ++expIdx) {
                uint32_t targetexpertId = expInfoTensor[expIdx].GetValue(0);
                uint32_t targetRankId = GetExpRank(targetexpertId);
                if (!IsInSameServer(targetRankId)) {
                    continue;
                }
                uint32_t tokenPosInBlock = cntInfoTensor(expIdx);
                PipeBarrier<PIPE_ALL>();
                // 在IPC的当前Block中，前面还有tokenPosInBlock个Token
                uint32_t targetExpOffset = (targetexpertId % localMoeExpertNum_) * worldSize_ *
                    RANK_SIZE_ON_IPC;
                // 第几个Exp段
                uint32_t targetServerOffset = formServerId * SERVER_RANK_SIZE * RANK_SIZE_ON_IPC;
                // 第几个Server段
                uint32_t targetRankOffset = (rankId_ % SERVER_RANK_SIZE) * RANK_SIZE_ON_IPC;
                // 第几个Rank段
                uint32_t targetTokenOffset = tokenPosInBlock * tokenStructLen_;  // 第几个Token位
                uint32_t targetOffset = targetExpOffset + targetServerOffset + targetRankOffset + targetTokenOffset;

                targetTokenIpcGt.SetGlobalBuffer((__gm__ uint8_t*)(shareAddrs[targetRankId % SERVER_RANK_SIZE] +
                    IPC_DATA_OFFSET + targetOffset));
                PipeBarrier<PIPE_ALL>();
                DataCopy(targetTokenIpcGt, localUB[tokenIdx * tokenStructLen_], tokenStructLen_);
                // 对应token个数加1
                GlobalTensor<int32_t> targetCntIpcGt;    // 对端IPC的CntTensor，统计对端收到的次数
                targetCntIpcGt.SetGlobalBuffer((__gm__ int32_t*)(shareAddrs[targetRankId % SERVER_RANK_SIZE] +
                    IPC_TOKEN_CNT_OFFSET));
                uint32_t setTokenCntOffset = (targetexpertId % localMoeExpertNum_) * worldSize_ +
                    formServerId * SERVER_RANK_SIZE + (rankId_ % SERVER_RANK_SIZE);
                SetTokenCnt(targetCntIpcGt[EXP_TOKEN_COUNT_FLAG_CNT * setTokenCntOffset]);
            }
        }
    }
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::Ipc2Out()
{
    uint32_t coresPerExp = aivNum_ / localMoeExpertNum_;
    if (aivId_ >= coresPerExp * localMoeExpertNum_) {
        return;
    }
    uint32_t coresPerServer = aivNum_ / serverNum;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<int32_t> flagIpcGt;
    flagIpcGt.SetGlobalBuffer((__gm__ int32_t*)(shareAddrs[rankId_ % SERVER_RANK_SIZE]));
    uint32_t curExpIdx = aivId_ / coresPerExp;   // 当前处理的专家在本卡上的Idx
    uint32_t localAivId = aivId_ % coresPerExp;  // 处理本专家的同一批Core中，本Core的Idx
    // 每个exp对应ranksize行
    uint32_t srCntPerExp = serverNum * SERVER_RANK_SIZE;
    // 平均每个核处理多少行
    uint32_t srCntPerCore = srCntPerExp / coresPerExp;
    // 平分后还剩多少行
    uint32_t srCntRemain = srCntPerExp % coresPerExp;
    // 前面的核共分到了多少剩余
    uint32_t srCntPreRemain = (localAivId < srCntRemain) ? localAivId : srCntRemain;
    // 当前核分到多少行
    uint32_t srCntCurCore = (localAivId < srCntRemain) ? (srCntPerCore + 1) : srCntPerCore;

    GlobalTensor<int32_t> tokenCntIpcGt;
    tokenCntIpcGt.SetGlobalBuffer((__gm__ int32_t*)(shareAddrs[rankId_ % SERVER_RANK_SIZE] + IPC_TOKEN_CNT_OFFSET));

    // tBuf 内存分配
    // 4k ~ 6k 保存按expert统计的token个数信息
    LocalTensor<int64_t> tokenCntByExpUB = tBuf.GetWithOffset<int64_t>(2 * 1024 / sizeof(int64_t), 4 * 1024);
    // 6k ~ 8k 保存token个数统计信息
    LocalTensor<int32_t> tokenCntUB = tBuf.GetWithOffset<int32_t>(2 * 1024 / sizeof(int32_t), 6 * 1024);
    // 2k ~ 4k 保存权重信息
    LocalTensor<float>  weightLt = tBuf.GetWithOffset<float>(2 * 1024 / sizeof(float), 2 * 1024);

    DataCopyExtParams copyExpertIdsParams{1, static_cast<uint32_t>(serverNum * SERVER_RANK_SIZE *
        localMoeExpertNum_ * EXP_TOKEN_COUNT_FLAG_CNT * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams;
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(tokenCntUB, tokenCntIpcGt, copyExpertIdsParams, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    int32_t cntSum = 0;
    const int tempSize = serverNum * SERVER_RANK_SIZE * localMoeExpertNum_;
    int log2WorldSize = ScalarGetSFFValue<1>(worldSize_);
#pragma unroll 8
    for (uint32_t i = 0; i < tempSize; ++i) {
        cntSum += tokenCntUB(i << 3);
        tokenCntUB(i) = cntSum;
    }
    for (uint32_t i = 0; i < localMoeExpertNum_; ++i){
        if (expertTokenNumsType_ == 1) {
            int32_t preValue = (i == 0) ? 0 : tokenCntUB(i * worldSize_ - 1);
            tokenCntByExpUB(i) = static_cast<int64_t>(tokenCntUB(i * worldSize_ + worldSize_ - 1) - preValue);
        } else {
            tokenCntByExpUB(i) = static_cast<int64_t>(tokenCntUB(i * worldSize_ + worldSize_ - 1));
        }
    }

    uint32_t srPreCnt = curExpIdx * srCntPerExp + localAivId * srCntPerCore + srCntPreRemain;
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<uint8_t> srcIpcGt;
    srcIpcGt.SetGlobalBuffer((__gm__ uint8_t*)(shareAddrs[rankId_ % SERVER_RANK_SIZE] + IPC_DATA_OFFSET));
    constexpr uint32_t tokenUbSize = TBUF_SIZE - TBUF_TEMP_OFFSET;

    LocalTensor<uint8_t> localUB = tBuf.GetWithOffset<uint8_t>(tokenUbSize / sizeof(uint8_t),
        TBUF_TEMP_OFFSET);
    LocalTensor<float> localUBfloat = tBuf.GetWithOffset<float>(tokenUbSize / sizeof(float),
        TBUF_TEMP_OFFSET);
    LocalTensor<int32_t> localUBint32 = tBuf.GetWithOffset<int32_t>(tokenUbSize / sizeof(int32_t),
        TBUF_TEMP_OFFSET);

    int32_t sumTokenCnt = (0 == srPreCnt) ? 0 : tokenCntUB(srPreCnt - 1);
    for (uint32_t idx = 0; idx < srCntCurCore; ++idx) {
        // 循环本Core需要处理的Rank数
        uint32_t srIdx = srPreCnt + idx;
        int32_t curSrTokenCnt = tokenCntUB(srIdx) - (srIdx == 0 ? 0 : tokenCntUB(srIdx - 1));
        if (curSrTokenCnt == 0) {
            continue;
            // 目标Rank没Token发来则跳过
        }
        uint32_t tokenCntInUB = tokenUbSize / tokenStructLen_;
        // 单次能搬移的token数据量
        uint32_t batchCnt = (curSrTokenCnt + tokenCntInUB - 1) / tokenCntInUB;
        // 循环搬运次数
        // 分批逻辑待修改，应该是先收集所有待处理Rank的Token，再写out
        for (uint32_t batchIdx = 0; batchIdx < batchCnt; ++batchIdx) {
            uint32_t tokenCntInBatch = tokenCntInUB;
            if (batchIdx == batchCnt - 1) {
                tokenCntInBatch = curSrTokenCnt - (batchCnt - 1) * tokenCntInUB;
            }
            DataCopyExtParams copyTokenParams{static_cast<uint16_t>(1),
                static_cast<uint32_t>(tokenCntInBatch * tokenStructLen_), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padParams;
            uint32_t srcIpcOffset = srIdx * RANK_SIZE_ON_IPC + batchIdx * tokenCntInUB * tokenStructLen_;
            DataCopyPad(localUB, srcIpcGt[srcIpcOffset], copyTokenParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopyExtParams writeTokenParams{static_cast<uint16_t>(tokenCntInBatch),
                static_cast<uint32_t>(sizeof(ExpandXOutType) * axisH_),
                static_cast<uint32_t>(tokenGapInStruct_), 0, 0};
            LocalTensor<ExpandXOutType> outUB = localUB.ReinterpretCast<ExpandXOutType>();
            DataCopyPad(expandXOutGMTensor_[(sumTokenCnt + batchIdx * tokenCntInUB) * axisH_], outUB, writeTokenParams);
            PipeBarrier<PIPE_ALL>();

            for (uint32_t tokenIdx = 0; tokenIdx < tokenCntInBatch; tokenIdx++) {
                for (uint32_t expIdx = 0; expIdx < axisK_; expIdx++) {
                    uint32_t expOffset = (tokenIdx * tokenStructLen_ + expOffsetInStruct_) / sizeof(int32_t) + expIdx;
                    if (curExpIdx + rankId_ * localMoeExpertNum_ == localUBint32(expOffset)) {
                        uint32_t weightOffset = expOffset + alignK_;
                        weightLt(tokenIdx) = localUBfloat(weightOffset);
                        break;
                    }
                }
                LocalTensor<float> pintfLt = localUBfloat[(tokenIdx * tokenStructLen_ +
                                                        weightOffsetInStruct_) / sizeof(float)];
            }
            // weight output
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams weightTokenParams{static_cast<uint16_t>(1),
                static_cast<uint32_t>(tokenCntInBatch * sizeof(float)), 0, 0, 0};
            DataCopyPad(weightsOutGt[(sumTokenCnt + batchIdx * tokenCntInUB)], weightLt, weightTokenParams);
            PipeBarrier<PIPE_ALL>();
            // dynamic scales to output
            if constexpr (DynamicQuant) {
                DataCopyExtParams quantTokenParams{static_cast<uint16_t>(tokenCntInBatch),
                    static_cast<uint32_t>(sizeof(float)),
                    static_cast<uint32_t>((tokenStructLen_ - UB_32B_ALIGN) / UB_32B_ALIGN), 0, 0};

                LocalTensor<float> quantTempUB = localUB[scaleOffsetInStruct_].ReinterpretCast<float>();
                DataCopyPad(dynamicScalesOutGMTensor_[(sumTokenCnt + batchIdx * tokenCntInUB)], quantTempUB,
                            quantTokenParams);
            }
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        }
        sumTokenCnt += curSrTokenCnt;
    }
    if (aivId_ == 0) {
        // 搬运token统计信息到output
        GlobalTensor<int32_t> tokenNumsGlobal;
        tokenNumsGlobal.SetGlobalBuffer((__gm__ int32_t*)(epRecvCountsGM_));
        DataCopyExtParams countsParams{1,
            static_cast<uint32_t>(localMoeExpertNum_ * serverNum * SERVER_RANK_SIZE * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(tokenNumsGlobal, tokenCntUB, countsParams);

        // 搬运按expert的token信息到output
        GlobalTensor<int64_t> expertTokenNumsGlobal;
        expertTokenNumsGlobal.SetGlobalBuffer((__gm__ int64_t*)(expertTokenNumsOutGM_));
        DataCopyExtParams writeCountsParams{1,
            static_cast<uint32_t>(localMoeExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(expertTokenNumsGlobal, tokenCntByExpUB, writeCountsParams);
    }
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::CleanUp()
{
    uint32_t cleanBuffSize = worldSize_ * localMoeExpertNum_ * TOKEN_COUNT_SIZE;
    if (cleanBuffSize < STATE_OFFSET * serverNum) {
        cleanBuffSize = STATE_OFFSET * serverNum;
    }
    LocalTensor<int32_t> cleanTempLt_ = tBuf.GetWithOffset<int32_t>(cleanBuffSize / sizeof(int32_t), TBUF_TEMP_OFFSET);
    GlobalTensor<int32_t> flagIpcGt;
    Duplicate<int32_t>(cleanTempLt_, 0, cleanBuffSize / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>();
    flagIpcGt.SetGlobalBuffer((__gm__ int32_t*)(shareAddrs[rankId_ % SERVER_RANK_SIZE]));
    PipeBarrier<PIPE_ALL>();
    DataCopy(readStatusTensor_, cleanTempLt_, cleanBuffSize / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>();
    DataCopy(flagIpcGt[IPC_TOKEN_CNT_OFFSET / sizeof(int32_t)], cleanTempLt_, cleanBuffSize / sizeof(int32_t));
}

template <TemplateMC2TypeA2layeredAicpuClass>
__aicore__ inline void MoeDistributeDispatchA2LayeredAicpu<TemplateMC2TypeA2layeredAicpuFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        ReorderTokens();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        GatherAndWriteCntInfo();
        DispatchBetweenServer();
        WaitWindow();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();

        // 最后serverNum个核不参与Win2Ipc，只进行reduceInfo计算
        if (aivId_ < aivNum_ - serverNum) {
            Win2Ipc();
        } else {
            CreateInnerReduceInfo();
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        SetIpcFlag(IPC_FLAG_STEP_1);
        WaitIpcFlag(IPC_FLAG_STEP_1);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        Ipc2Out();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();

        if (aivId_ == 0) {
            CleanUp();
        }
        PipeBarrier<PIPE_ALL>();
        SetIpcFlag(IPC_FLAG_STEP_2);
        WaitIpcFlag(IPC_FLAG_STEP_2);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        hccl_.Finalize();
    }
}
} // MoeDistributeDispatchA2Impl
#endif // MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_AICPU_H

