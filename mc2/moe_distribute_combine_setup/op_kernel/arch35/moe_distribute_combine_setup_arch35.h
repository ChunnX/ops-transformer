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
 * \file moe_distribute_combine_setup_arch35.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H
#define MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hcomm/hcomm.h"
#include "kernel_tiling/kernel_tiling.h"

#if __has_include("../common/op_kernel/mc2_kernel_utils.h")
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "../moe_distribute_base.h"
#include "../moe_distribute_combine_setup_tiling_data.h"
#include "../../../common/op_kernel/mc2_moe_context.h"

namespace MoeDistributeCombineSetupImpl {

struct WriteWithNotifySQEInfoParams {
    uint64_t dataSrcAddr;
    uint64_t dataDstAddr;
    uint32_t length;
    uint64_t notifyAddr;
    uint64_t notifyData;
    uint8_t cqe;
};

#define TemplateMC2TypeClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeFunc ExpandXType, ExpandIdxType

#define TOKEN_BY_TOKEN
//#define DETECTION_TIME    // 统计时延开关

using namespace AscendC;
using namespace Mc2Aclnn;

template <TemplateMC2TypeClass>
class MoeDistributeCombineSetup {
    constexpr static uint8_t BUFFER_NUM = 2;                // 多buf
    constexpr static uint64_t STATE_OFFSET = 512U;          // 状态空间偏移地址
    constexpr static uint32_t STATE_SIZE = 1024U * 1024U;   // 1M
    constexpr static uint32_t UB_ALIGN = 32U;               // UB按32字节对齐
    constexpr static uint64_t STATE_SIZE_PER_CORE = 512U;   // 数据和状态的0/1区标识占用空间
    constexpr static uint64_t COMBINE_STATE_OFFSET = 0U;    // 本卡状态空间偏移地址，前面的地址给dispatch用
    constexpr static uint32_t STATE_COUNT_THRESHOLD = 512U; // moeExpertNumPerRank*epWorldSize状态数阈值

public:
    __aicore__ inline MoeDistributeCombineSetup(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                GM_ADDR quantExpandX, GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling,
                                __gm__ void *mc2CcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitCoreCalForExpert();
    __aicore__ inline void SplitCoreCal(); // 按卡分核，用于敲doorbell
    __aicore__ inline void CurRankComm(const WriteWithNotifySQEInfoParams &params);
    __aicore__ inline void Communication();
    __aicore__ inline void BuffInit();
    __aicore__ inline void AssistInfoLocalCopy();

    __aicore__ GM_ADDR GetWinAddrByRankId(const uint32_t rankId, const uint8_t expertLocalId = 0U)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + winDataSizeOffset_ +
               expertPerSizeOnWin_ * static_cast<uint64_t>(expertLocalId);
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(uint32_t rankId)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + COMBINE_STATE_OFFSET +
               WIN_STATE_OFFSET * static_cast<uint64_t>(dataState_);
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> assistInfoForCombineGlobal_;
    GM_ADDR epWindowGM_;
    GM_ADDR epStatusSpaceGM_;
    GM_ADDR expandXGM_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uint32_t进行处理
    const MoeDistributeCombineSetupInfo *moeDistributeCombineSetupInfo_{nullptr};
    uint32_t axisMaxBS_{0};
    uint32_t coreIdx_{0};    // aiv id
    uint32_t moeSendNum_{0}; // moeExpertPerRankNum * epWorldSize
    uint32_t curRankExpertNum_{0};
    uint64_t epDataOffsetOnWin_{0};
    uint64_t epStateOffsetOnWin_{0};
    uint64_t axisHExpandXTypeSize_{0};
    uint32_t startExpertId_{0}; // 当前核处理的起始专家序号（组装SQE阶段）
    uint32_t endExpertId_{0};   // 当前核处理的结束专家序号（组装SQE阶段）
    uint32_t sendExpertNum_{0};
    uint32_t startRankId_{0}; // 当前核负责敲doorbell的起始卡号
    uint32_t endRankId_{0};   // 当前核负责敲doorbell的结束卡号
    uint32_t sendRankNum_{0}; // 当前核负责敲doorbell的卡数
    uint32_t dataState_{0};
    uint64_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t remain_ub_space{0};
    bool isShardExpert_{false};

    TQue<QuePosition::VECIN, 1> assistInfoQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1>
        expertTokenTmpQueue_; // URMA本卡回环通信拷贝数据Local Memory暂存

    // URMA通信新增变量、Buf
    TBuf<> urmaSqInfoBuf_;  // 80B
    TBuf<> urmaCqInfoBuf_;  // 48B
    TBuf<> cqeBuf_;         // 32B * CQ_DEPTH_256
    TBuf<> templateSqeBuf_; // 96B
    TBuf<> piCiBuf_;        // 32B，用于 GetPICI 的 MTE 搬运

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    //=============================== Urma ===============================//
    TBuf<> hcommBuf_;
    LocalTensor<uint8_t> hcommTensor_;

    constexpr static uint64_t HCOMM_INIT_SIZE = 512UL;
    constexpr static uint64_t WIN_ADDR_ALIGN = 512UL;
    constexpr static uint32_t MAX_UB_SIZE = 170U * 1024U;
    constexpr static uint8_t BUFFER_SINGLE = 1;

    //uint32_t hOutSize_{0};         // 输出hidden size (字节)
    //uint32_t hAlignWinCnt_{0};     // 窗口对齐后的元素个数
    //uint32_t hOutAlignUbSize_{0};  // UB对齐后的hidden size
    //uint32_t bufferNum_{0};        // 缓冲区数量
    //TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列

    uint32_t hAlignSize_{0};       // UB对齐后的hidden size
    uint32_t hAlignWinSize_{0};    // 窗口对齐后的hidden size
    __gm__ ChannelEntity* urmaCommHandle_[HCCL_MAX_RANK_SIZE];
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文

    // GetUrmaAddrByRankId中是否包含 winDataSizeOffset_??????
    __aicore__ inline GM_ADDR GetUrmaAddrByRankId(const int32_t rankId, const int32_t epRankId)
    {
        if (rankId == epRankId) {
            // ---todo 需要确认 rma.addr + STATE_SIZE 是否已经等价于目标 rank 当前 winDataSizeOffset_ 对应的数据半区。如果不是，remoteAddr 还需要补 winDataSizeOffset
            return (GM_ADDR)urmaCommHandle_[0]->localBufferAddr->bufferInfo.rma.addr + STATE_SIZE;
        }
        uint32_t currRankId = rankId > epRankId? rankId - 1 : rankId;
        return (GM_ADDR)urmaCommHandle_[currRankId]->remoteBufferAddr->bufferInfo.rma.addr + STATE_SIZE;
    }

    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(const int32_t rankId, const int32_t epRankId)
    {
        if (rankId == epRankId) {
            return (GM_ADDR)urmaCommHandle_[0]->localBufferAddr->bufferInfo.rma.addr;// + dataState_ * WIN_STATE_OFFSET;
        }
        uint32_t currRankId = rankId > epRankId? rankId - 1 : rankId;
        return (GM_ADDR)urmaCommHandle_[currRankId]->remoteBufferAddr->bufferInfo.rma.addr;// + dataState_ * WIN_STATE_OFFSET;
    }
    //=============================== Urma ===============================//

};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Init(GM_ADDR context,
    GM_ADDR expandX, GM_ADDR /*expertIds*/, GM_ADDR assistInfoForCombine, GM_ADDR /*quantExpandX*/,
    GM_ADDR /*commCmdInfoOut*/, GM_ADDR /*workspaceGM*/, TPipe *pipe,
    const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    moeDistributeCombineSetupInfo_ = &(tilingData->moeDistributeCombineSetupInfo);
    mc2Context_ = (__gm__ Mc2MoeContext* )context;

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    // 获取win状态区地址，并保证数据一致
    // 在1M中选择512K偏移后的1.5k空间记录本卡历史状态
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(mc2Context_->epHcclBuffer_[mc2Context_->epRankId]);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET +
                                                            STATE_SIZE_PER_CORE * static_cast<uint64_t>(coreIdx_)));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);

    dataState_ = selfDataStatusTensor(0); // win状态区的0/1标识
    expandXGM_ = expandX;
    assistInfoForCombineGlobal_.SetGlobalBuffer((__gm__ int32_t *)assistInfoForCombine);

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    axisMaxBS_ = moeDistributeCombineSetupInfo_->globalBs / moeDistributeCombineSetupInfo_->epWorldSize;
    moeSendNum_ = moeDistributeCombineSetupInfo_->epWorldSize * moeDistributeCombineSetupInfo_->moeExpertPerRankNum;
    isShardExpert_ = (moeDistributeCombineSetupInfo_->epRankId <
                      moeDistributeCombineSetupInfo_->sharedExpertRankNum); // 当前rank是否为共享专家
    curRankExpertNum_ = (isShardExpert_) ? 1U : moeDistributeCombineSetupInfo_->moeExpertPerRankNum; // 本卡专家数

    axisHExpandXTypeSize_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->h) *
                            static_cast<uint64_t>(sizeof(ExpandXType));              // 一个token占用内存
    expertPerSizeOnWin_ = static_cast<uint64_t>(axisMaxBS_) * axisHExpandXTypeSize_; // 单个专家可能占用的最大空间

    winDataSizeOffset_ =
        static_cast<uint64_t>(dataState_) * (static_cast<uint64_t>(moeDistributeCombineSetupInfo_->totalWinSize) >> 1);
    stateOffset_ = (moeSendNum_ > STATE_COUNT_THRESHOLD) ? (STATE_OFFSET >> 1) : STATE_OFFSET;
    epStateOffsetOnWin_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->epRankId) * stateOffset_;
    epDataOffsetOnWin_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->epRankId) *
                         static_cast<uint64_t>(moeDistributeCombineSetupInfo_->moeExpertPerRankNum) *
                         expertPerSizeOnWin_; // 前面rank数据区占用内存地址偏移

    epWindowGM_ = GetWinAddrByRankId(moeDistributeCombineSetupInfo_->epRankId);
    epStatusSpaceGM_ = GetWinStateAddrByRankId(moeDistributeCombineSetupInfo_->epRankId);

    //=============================== Urma ===============================//
    hAlignSize_ = Ceil(moeDistributeCombineSetupInfo_->h * sizeof(ExpandXType), UB_ALIGN) * UB_ALIGN; // UB 32字节对齐
    hAlignWinSize_ = Ceil(hAlignSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // 窗口512字节对齐

    //hOutSize_ = moeDistributeCombineSetupInfo_->h * sizeof(ExpandXType);                             // 原始hidden size
    //hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXType);
    //hOutAlignUbSize_ = hAlignSize_;
    //uint32_t tmpTotalUB = moeDistributeCombineSetupInfo_->totalUbSize + hOutAlignUbSize_ * BUFFER_NUM;
    //bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
    //tpipe_->InitBuffer(xQueue_, bufferNum_, hOutAlignUbSize_);

    for (uint32_t i = 0; i < moeDistributeCombineSetupInfo_->epWorldSize -1; i++) {
        urmaCommHandle_[i] = reinterpret_cast<__gm__ ChannelEntity* >(mc2Context_->hcommHandle_[i]);
    }
    //=============================== Urma ===============================//

#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange<ExpandXType>((__gm__ ExpandXType *)(epWindowGM_), moeDistributeCombineSetupInfo_->totalWinSize);
    OOMCheckAddrRange<float>((__gm__ float *)(epStatusSpaceGM_), STATE_SIZE);
#endif

    // 分核计算
    SplitCoreCalForExpert();
    SplitCoreCal();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::SplitCoreCalForExpert()
{
    // 按epWorldSize*本卡专家数分核
    uint32_t coreIdxNew =
        (coreIdx_ + moeDistributeCombineSetupInfo_->epRankId) % moeDistributeCombineSetupInfo_->aivNum;
    sendExpertNum_ =
        moeDistributeCombineSetupInfo_->epWorldSize * curRankExpertNum_ / moeDistributeCombineSetupInfo_->aivNum;
    uint32_t remainderExpertNum =
        moeDistributeCombineSetupInfo_->epWorldSize * curRankExpertNum_ % moeDistributeCombineSetupInfo_->aivNum;
    startExpertId_ = sendExpertNum_ * coreIdxNew;
    if (coreIdxNew < remainderExpertNum) {
        ++sendExpertNum_;
        startExpertId_ += coreIdxNew;
    } else {
        startExpertId_ += remainderExpertNum;
    }
    endExpertId_ = startExpertId_ + sendExpertNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::SplitCoreCal()
{
    // 按卡数分核，用于敲doorbell阶段
    uint32_t coreIdxNew =
        (coreIdx_ + moeDistributeCombineSetupInfo_->epRankId) % moeDistributeCombineSetupInfo_->aivNum;
    sendRankNum_ = moeDistributeCombineSetupInfo_->epWorldSize / moeDistributeCombineSetupInfo_->aivNum;
    uint32_t remainderRankNum = moeDistributeCombineSetupInfo_->epWorldSize % moeDistributeCombineSetupInfo_->aivNum;
    startRankId_ = sendRankNum_ * coreIdxNew;
    if (coreIdxNew < remainderRankNum) {
        ++sendRankNum_;
        startRankId_ += coreIdxNew;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::BuffInit()
{
    uint32_t assistInfoQueueSize = Ceil(moeSendNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t urmaSqInfoBufSize = Ceil(static_cast<uint32_t>(sizeof(HcclAiRMAWQ)), UB_ALIGN) * UB_ALIGN;
    uint32_t urmaCqInfoBufSize = Ceil(static_cast<uint32_t>(sizeof(HcclAiRMACQ)), UB_ALIGN) * UB_ALIGN;
    uint32_t cqeBufSize = UB_ALIGN * CQ_DEPTH_256;
    uint32_t templateSqeBufSize = Ceil(WRITE_WITH_NOTIFY_SQE_SIZE, UB_ALIGN) * UB_ALIGN;
    uint32_t piCiBufSize = UB_ALIGN;
    remain_ub_space = moeDistributeCombineSetupInfo_->totalUbSize - assistInfoQueueSize - urmaSqInfoBufSize -
                      urmaCqInfoBufSize - cqeBufSize - templateSqeBufSize - piCiBufSize;

    tpipe_->Reset();
    tpipe_->InitBuffer(assistInfoQueue_, 1, assistInfoQueueSize); // epWorldSize * moeExpertPerRankNum * 4

    // 初始化urma相关buf
    tpipe_->InitBuffer(urmaSqInfoBuf_, urmaSqInfoBufSize);
    tpipe_->InitBuffer(urmaCqInfoBuf_, urmaCqInfoBufSize);
    tpipe_->InitBuffer(cqeBuf_, cqeBufSize);
    tpipe_->InitBuffer(templateSqeBuf_, templateSqeBufSize);
    tpipe_->InitBuffer(piCiBuf_, piCiBufSize);

    tpipe_->InitBuffer(expertTokenTmpQueue_, 1, static_cast<uint32_t>(remain_ub_space));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::AssistInfoLocalCopy()
{
    LocalTensor<int32_t> assistInfoForCombineLocal = assistInfoQueue_.AllocTensor<int32_t>();

    DataCopyExtParams epSendCntParams;
    if (isShardExpert_) {
        // 对于共享专家来说assistInfoForCombine输入维度为epWorldSize个
        epSendCntParams = {1U, moeDistributeCombineSetupInfo_->epWorldSize * static_cast<uint32_t>(sizeof(uint32_t)),
                           0U, 0U, 0U};
    } else {
        epSendCntParams = {1U, moeSendNum_ * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
    }
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(assistInfoForCombineLocal, assistInfoForCombineGlobal_, epSendCntParams, copyPadParams);
    assistInfoQueue_.EnQue(assistInfoForCombineLocal);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Communication()
{
    AssistInfoLocalCopy();
    LocalTensor<int32_t> assistInfoForCombineLocal = assistInfoQueue_.DeQue<int32_t>();
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>(); // 等assistInfoQueue_的GM->Local，后续标量读

    for (uint32_t expertIdx = startExpertId_; expertIdx < endExpertId_; ++expertIdx) {
        // 计算当前处理的token块应发送到哪个卡的哪个专家
        uint32_t sendRankId = expertIdx % moeDistributeCombineSetupInfo_->epWorldSize;
        uint32_t sendExpertId = expertIdx / moeDistributeCombineSetupInfo_->epWorldSize; // 每卡逻辑专家序号

        uint32_t preCount = 0U;
        uint32_t assistInfoIdx = expertIdx; // 按epWorldSize*curRankExpertNum_分核等价于对assistInfo分核
        if (likely(assistInfoIdx > 0U)) {
            // 计算其他卡或专家已经发了多少token
            preCount = assistInfoForCombineLocal.GetValue(assistInfoIdx - 1U);
        }
        // 当前要发送的token数量
        uint32_t curTokenNum = assistInfoForCombineLocal.GetValue(assistInfoIdx) - preCount;
        if(0 == curTokenNum) {
            continue;
        }

        if (unlikely(sendRankId == moeDistributeCombineSetupInfo_->epRankId)) { // 本卡通信：仅搬运数据
            if (likely(curTokenNum > 0U)) {
                WriteWithNotifySQEInfoParams notifySqeInfo = {
                    (uint64_t)expandXGM_ +
                        (likely(curTokenNum > 0U) ? static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_ : 0U),
                    (uint64_t)(GetWinAddrByRankId(sendRankId, sendExpertId) + epDataOffsetOnWin_),
                    static_cast<uint32_t>(axisHExpandXTypeSize_) * max(curTokenNum, 1U),
                    0U, 0U, 0U};
                if (unlikely(sendExpertId == curRankExpertNum_ - 1U)) {
                    notifySqeInfo.notifyAddr -= sizeof(uint64_t);
                    notifySqeInfo.notifyData = 0x3f800000;
                    notifySqeInfo.cqe = 1U;
                }

                CurRankComm(notifySqeInfo);
            }
            continue;
        }

        //非本卡通信：统一敲doorbell
        uint32_t currRankId = sendRankId > moeDistributeCombineSetupInfo_->epRankId ? sendRankId - 1 : sendRankId;
#ifdef TOKEN_BY_TOKEN
        // WriteNbi 逐 token
        for(uint32_t tokenIndex = 0; tokenIndex < curTokenNum; ++tokenIndex) {
            GM_ADDR rankGM = (GM_ADDR)(expandXGM_ + (static_cast<uint64_t>(preCount) + tokenIndex) * axisHExpandXTypeSize_);
            GM_ADDR remoteAddr = (GM_ADDR)(
                GetUrmaAddrByRankId(sendRankId, moeDistributeCombineSetupInfo_->epRankId) + // GetUrmaAddrByRankId中是否包含 winDataSizeOffset_??????
                epDataOffsetOnWin_ +
                static_cast<uint64_t>(sendExpertId) * expertPerSizeOnWin_ +
                static_cast<uint64_t>(tokenIndex) * axisHExpandXTypeSize_);

            hcomm_.WriteNbi<false>(
                mc2Context_->hcommHandle_[currRankId],
                remoteAddr,
                rankGM,
                static_cast<uint32_t>(axisHExpandXTypeSize_));
        }
#else
        // ---todo 如果 WriteNbi 支持一次写整段
        GM_ADDR rankGM = (GM_ADDR)(expandXGM_ + static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_);
        GM_ADDR remoteAddr = (GM_ADDR)(
            GetUrmaAddrByRankId(sendRankId, moeDistributeCombineSetupInfo_->epRankId) +
            epDataOffsetOnWin_ +
            static_cast<uint64_t>(sendExpertId) * expertPerSizeOnWin_);

        hcomm_.WriteNbi<false>(
            mc2Context_->hcommHandle_[currRankId],
            remoteAddr,
            rankGM,
            static_cast<uint32_t>(curTokenNum * axisHExpandXTypeSize_));
#endif

#ifdef DETECTION_TIME
        AscendC::PipeBarrier<PIPE_ALL>();                       // 1.清空流水线，确保之前的任务不干扰本次测量
        int64_t g0 = AscendC::GetSystemCycle();                 // 2.记录起始时间
#endif
        hcomm_.Commit(mc2Context_->hcommHandle_[currRankId]);
        //hcomm_.Drain(mc2Context_->hcommHandle_[currRankId]);
#ifdef DETECTION_TIME
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t g1 = AscendC::GetSystemCycle();
        int64_t costCycles = g1 - g0;
        op_LOGD("Communication cost cycles:%ld\n", costCycles);
#endif
    }
    SyncAll<true>();
    assistInfoQueue_.FreeTensor<int32_t>(assistInfoForCombineLocal);
}

template <TemplateMC2TypeClass>
__aicore__ inline void
MoeDistributeCombineSetup<TemplateMC2TypeFunc>::CurRankComm(const WriteWithNotifySQEInfoParams &params)
{
    // expandX在GM上，winIn也属于GM，数据需要GM -> Local -> winIn
    GlobalTensor<uint8_t> selfDataSrcTensor;
    GlobalTensor<uint8_t> selfDataDstTensor;
    selfDataSrcTensor.SetGlobalBuffer((__gm__ uint8_t *)params.dataSrcAddr);
    selfDataDstTensor.SetGlobalBuffer((__gm__ uint8_t *)params.dataDstAddr);
    LocalTensor<uint8_t> expertTokenTmpU8 = expertTokenTmpQueue_.AllocTensor<uint8_t>();

    uint64_t sendBytes = static_cast<uint64_t>(params.length);
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(remain_ub_space), 0U, 0U, 0U};
    DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};

    uint64_t offset = 0U;
    for (; sendBytes > remain_ub_space; sendBytes -= remain_ub_space) {
        DataCopyPad(expertTokenTmpU8, selfDataSrcTensor[offset], copyParams, padParams);
        expertTokenTmpQueue_.EnQue(expertTokenTmpU8);
        expertTokenTmpU8 = expertTokenTmpQueue_.DeQue<uint8_t>();
        DataCopyPad(selfDataDstTensor[offset], expertTokenTmpU8, copyParams);
        offset += remain_ub_space;
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    copyParams.blockLen = static_cast<uint32_t>(sendBytes);
    DataCopyPad(expertTokenTmpU8, selfDataSrcTensor[offset], copyParams, padParams);
    expertTokenTmpQueue_.EnQue(expertTokenTmpU8);
    expertTokenTmpU8 = expertTokenTmpQueue_.DeQue<uint8_t>();
    DataCopyPad(selfDataDstTensor[offset], expertTokenTmpU8, copyParams);
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

    expertTokenTmpQueue_.FreeTensor<uint8_t>(expertTokenTmpU8);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        // URMA 通信相关Buffer初始化
        BuffInit();

        // Phase1: 组装SQE（按卡数*本卡专家数分核）
        Communication();
    }
}
} // namespace MoeDistributeCombineSetupImpl

#endif // MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H
