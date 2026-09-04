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
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "../moe_distribute_combine_setup_base.h"
#include "../moe_distribute_combine_setup_tiling_data.h"
#include "../../../common/op_kernel/mc2_moe_context.h"

// arch35 性能优化开关（仅本算子）。全 0 时 Communication/CurRankComm/BuffInit 回原逻辑。
#ifndef MC2_CSU_OPT_OVERLAP_REMOTE
// 仅 bs<=16：先远端 WriteNbi 再本卡拷贝。bs256 必须走独立的「先本卡」函数，不能和这条编在一起。
#define MC2_CSU_OPT_OVERLAP_REMOTE 1
#endif
#ifndef MC2_CSU_OPT_LOCAL_COPY_PIPELINE
// 默认关闭。TQueBind 深度为 1，InitBuffer(2) 在 bs256 上只会拆半 UB、增加 chunk，不能真流水。
#define MC2_CSU_OPT_LOCAL_COPY_PIPELINE 0
#endif

namespace MoeDistributeCombineSetupImpl {

#define TemplateMC2TypeClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeFunc ExpandXType, ExpandIdxType

using namespace AscendC;
using namespace Mc2Aclnn;

template <TemplateMC2TypeClass>
class MoeDistributeCombineSetup {
    constexpr static uint8_t BUFFER_NUM = 2;              // 多buf
    constexpr static uint64_t STATE_OFFSET = 512U;        // 状态空间偏移地址
    constexpr static uint32_t STATE_SIZE = 1024U * 1024U; // 1M
    constexpr static uint32_t UB_ALIGN = 32U;             // UB按32字节对齐
    // 每 aiv 控制块大小（win [950KB,1MB) 区内的 D_S/C_S/C flag 步长）
    constexpr static uint64_t STATE_SIZE_PER_CORE = 512U;
    constexpr static uint32_t STATE_COUNT_THRESHOLD = 512U; // moeExpertNumPerRank*epWorldSize状态数阈值
    constexpr static uint64_t COMM_CMD_INFO_BYTES = 64U;    // commCmdInfoOut 每个 (a+rank) 条目 16 * int32

public:
    __aicore__ inline MoeDistributeCombineSetup(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                GM_ADDR quantExpandX, GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling,
                                __gm__ void *mc2CcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline __attribute__((always_inline)) void CurRankComm(
        const LocalTensor<int32_t> &assistInfoForCombineLocal, uint32_t curRankExpertNum);
    __aicore__ inline void Communication();
    __aicore__ inline __attribute__((always_inline)) void IssueRemoteWrites(
        const LocalTensor<int32_t> &assistInfoForCombineLocal, uint32_t curRankExpertNum);
    // bs256：独立函数，禁止内联，避免带上小 bs 的远端先行代码
    __aicore__ __attribute__((noinline)) void CommLocalThenRemote(const LocalTensor<int32_t> &assistInfoForCombineLocal,
                                                                  uint32_t curRankExpertNum);
    __aicore__ inline void BuffInit();
    __aicore__ inline void AssistInfoLocalCopy();

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> assistInfoForCombineGlobal_;
    GM_ADDR epWindowGM_;
    GM_ADDR epStatusSpaceGM_;
    GM_ADDR expandXGM_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    const MoeDistributeCombineSetupInfo *moeDistributeCombineSetupInfo_{nullptr};
    uint32_t axisMaxBS_{0};
    uint32_t moeSendNum_{0}; // moeExpertPerRankNum * epWorldSize
    uint64_t epDataOffsetOnWin_{0};
    uint64_t epStateOffsetOnWin_{0};
    uint64_t axisHExpandXTypeSize_{0};
    uint32_t startRankId_{0}; // 当前核处理的起始卡号
    uint32_t endRankId_{0};   // 当前核处理的结束卡号
    uint32_t sendRankNum_{0};
    uint32_t dataState_{0};
    uint64_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t remain_ub_space{0};
    bool isShardExpert_{false};

    TQue<QuePosition::VECIN, 1> assistInfoQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> expertTokenTmpQueue_;

    // =============================== Urma =============================== //
    uint32_t aivId_{0};
    uint32_t epRankId_{0};

    GM_ADDR quantExpandXGM_{nullptr}; // 仅用于结束时打印输出
    GM_ADDR statusDataSpaceGM_;
    GM_ADDR statusFlagGM_; // 本卡状态flag源地址，用于远端WriteNbi
    constexpr static uint64_t HCOMM_INIT_SIZE = 512UL;
    constexpr static uint32_t STATUS_FLAG_VALUE = 0x3f800000U; // float 1.0

    LocalTensor<uint8_t> hcommTensor_;

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文

    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId)
    {
        uint32_t index = rankId > epRankId_ ? rankId - 1 : rankId;
        // hcommHandle_ 在 GM(context) 上，读前 DCCI 单 cacheline，避免脏/陈旧 handle
        GlobalTensor<uint64_t> handleGT;
        handleGT.SetGlobalBuffer((__gm__ uint64_t *)&mc2Context_->hcommHandle_[index]);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(handleGT);
        return handleGT(0);
    }

    __aicore__ inline GM_ADDR GetUrmaAddrByRankId(const int32_t rankId, const uint8_t expertLocalId = 0U)
    {
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffset_ +
                         expertPerSizeOnWin_ * static_cast<uint64_t>(expertLocalId));
    }

    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(const int32_t rankId)
    {
        // C 专用状态区，与 D bank 物理隔离；C_S 独立乒乓
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[rankId] + COMBINE_STATUS_BASE +
                         dataState_ * COMBINE_STATE_BANK_STRIDE);
    }
    // =============================== Urma =============================== //
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Init(
    GM_ADDR context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR quantExpandX,
    GM_ADDR /* commCmdInfoOut */, GM_ADDR /* workspaceGM */, TPipe *pipe,
    const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    moeDistributeCombineSetupInfo_ = &(tilingData->moeDistributeCombineSetupInfo);
    mc2Context_ = (__gm__ Mc2MoeContext *)context;
    epRankId_ = mc2Context_->epRankId;

    quantExpandXGM_ = quantExpandX;

    // 获取 win 控制区：本算子只读写 C_S，与 D_S 独立，避免联跑抢乒乓
    GlobalTensor<int32_t> selfDataStatusTensor;
    statusDataSpaceGM_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankId_]);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGM_ + STATE_WIN_OFFSET +
                                                            STATE_SIZE_PER_CORE * static_cast<uint64_t>(aivId_) +
                                                            CTRL_COMBINE_STATE_OFFSET));

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState_ = selfDataStatusTensor(0); // C_S：combine 专用 0/1
    // 本卡 status flag 源（控制块 +64），供远端 WriteNbi；须在 epHcclBuffer_ 注册窗内
    statusFlagGM_ = statusDataSpaceGM_ + STATE_WIN_OFFSET + STATE_SIZE_PER_CORE * static_cast<uint64_t>(aivId_) +
                    CTRL_COMBINE_FLAG_OFFSET;

    GlobalTensor<int32_t> statusFlagTensor;
    statusFlagTensor.SetGlobalBuffer((__gm__ int32_t *)statusFlagGM_);
    statusFlagTensor.SetValue(0, static_cast<int32_t>(STATUS_FLAG_VALUE));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusFlagTensor);

    expandXGM_ = expandX;
    assistInfoForCombineGlobal_.SetGlobalBuffer((__gm__ int32_t *)assistInfoForCombine);

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    axisMaxBS_ = moeDistributeCombineSetupInfo_->globalBs / moeDistributeCombineSetupInfo_->epWorldSize;
    moeSendNum_ = moeDistributeCombineSetupInfo_->epWorldSize * moeDistributeCombineSetupInfo_->moeExpertPerRankNum;
    isShardExpert_ = (epRankId_ < moeDistributeCombineSetupInfo_->sharedExpertRankNum); // 当前rank是否为共享专家

    axisHExpandXTypeSize_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->h) *
                            static_cast<uint64_t>(sizeof(ExpandXType));              // 一个token占用内存
    expertPerSizeOnWin_ = static_cast<uint64_t>(axisMaxBS_) * axisHExpandXTypeSize_; // 单个专家可能占用的最大空间

    // C 固定占用数据区后半，区内按 C_S 做 totalWin/4 乒乓
    {
        const uint64_t totalWin = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->totalWinSize);
        winDataSizeOffset_ = (totalWin >> 1) + static_cast<uint64_t>(dataState_) * (totalWin >> 2);
    }
    stateOffset_ = (moeSendNum_ > STATE_COUNT_THRESHOLD) ? (STATE_OFFSET >> 1) : STATE_OFFSET;
    epStateOffsetOnWin_ = static_cast<uint64_t>(epRankId_) * stateOffset_;
    epDataOffsetOnWin_ = static_cast<uint64_t>(epRankId_) *
                         static_cast<uint64_t>(moeDistributeCombineSetupInfo_->moeExpertPerRankNum) *
                         expertPerSizeOnWin_; // 前面rank数据区占用内存地址偏移

    epWindowGM_ = GetUrmaAddrByRankId(epRankId_);
    epStatusSpaceGM_ = GetUrmaStateAddrByRankId(epRankId_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange<ExpandXType>((__gm__ ExpandXType *)(epWindowGM_),
                                   moeDistributeCombineSetupInfo_->totalWinSize / 4U);
    OOMCheckAddrRange<float>((__gm__ float *)(epStatusSpaceGM_), COMBINE_STATE_BANK_STRIDE);
#endif

    SplitCoreCal(); // 分核计算
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::SplitCoreCal()
{
    // 对worldSize按卡分核，得到每个核上处理的卡的数量，保证一个jetty只由一个核维护
    uint32_t coreIdxNew = (aivId_ + epRankId_) % moeDistributeCombineSetupInfo_->aivNum; // 按照卡去进行偏移
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
    uint64_t totalUbSize = moeDistributeCombineSetupInfo_->totalUbSize;
    uint32_t assistInfoQueueSize = Ceil(moeSendNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint64_t reservedUbSize = HCOMM_INIT_SIZE + assistInfoQueueSize;
    if (reservedUbSize > totalUbSize) {
        printf("[ERROR] hcomm_.Init UB space is insufficient, reserved=%llu total=%u\n", reservedUbSize, totalUbSize);
    }
    remain_ub_space = totalUbSize - reservedUbSize;
    tpipe_->Reset();
    if (!tpipe_->InitBuffer(assistInfoQueue_, 1, static_cast<uint32_t>(reservedUbSize)) ||
        !tpipe_->InitBuffer(expertTokenTmpQueue_, 1, static_cast<uint32_t>(remain_ub_space))) {
        printf("[ERROR] hcomm_.Init communication buffers failed\n");
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::AssistInfoLocalCopy()
{
    LocalTensor<int32_t> assistInfoForCombineLocal = assistInfoQueue_.AllocTensor<int32_t>();

    DataCopyExtParams epSendCntParams;
    if (isShardExpert_) {
        // 对于共享专家来说assistInfoForCombine输入维度为epWordSize个
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
__aicore__ inline __attribute__((always_inline)) void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::IssueRemoteWrites(
    const LocalTensor<int32_t> &assistInfoForCombineLocal, uint32_t curRankExpertNum)
{
    // 数据和状态分离：有token才发数据，每个dstRank始终单独发一次状态
    uint32_t middleRank = epRankId_ < startRankId_ ? startRankId_ : epRankId_;
    for (uint32_t dstRankId = middleRank; dstRankId < endRankId_; ++dstRankId) {
        if (unlikely(dstRankId == epRankId_)) {
            continue;
        }
        auto channelId = GetCommHandle(dstRankId);
        GM_ADDR dstStatusAddr = GetUrmaStateAddrByRankId(dstRankId) + epStateOffsetOnWin_;
        for (uint32_t expertIdx = 0U; expertIdx < curRankExpertNum; ++expertIdx) {
            uint32_t preCount = 0U;
            uint32_t assistInfoIdx = expertIdx * moeDistributeCombineSetupInfo_->epWorldSize + dstRankId;
            if (likely(assistInfoIdx > 0U)) {
                preCount = assistInfoForCombineLocal.GetValue(assistInfoIdx - 1U);
            }
            const uint32_t curCount = assistInfoForCombineLocal.GetValue(assistInfoIdx);
            uint32_t curTokenNum = curCount - preCount;
            if (curTokenNum == 0U) {
                continue;
            }
            GM_ADDR srcAddr = expandXGM_ + static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_;
            GM_ADDR dstAddr = GetUrmaAddrByRankId(dstRankId, expertIdx) + epDataOffsetOnWin_;
            uint64_t lengthU64 = axisHExpandXTypeSize_ * static_cast<uint64_t>(curTokenNum);
            int32_t dataRet = hcomm_.WriteNbi<false>(channelId, dstAddr, srcAddr, static_cast<uint32_t>(lengthU64));
            (void)dataRet;
        }
        hcomm_.WriteNbi<false>(channelId, dstStatusAddr, statusFlagGM_, UB_ALIGN);
        hcomm_.Commit(channelId);
    }

    middleRank = epRankId_ < endRankId_ ? epRankId_ : endRankId_;
    for (uint32_t dstRankId = startRankId_; dstRankId < middleRank; ++dstRankId) {
        if (unlikely(dstRankId == epRankId_)) {
            continue;
        }
        auto channelId = GetCommHandle(dstRankId);
        GM_ADDR dstStatusAddr = GetUrmaStateAddrByRankId(dstRankId) + epStateOffsetOnWin_;
        for (uint32_t expertIdx = 0U; expertIdx < curRankExpertNum; ++expertIdx) {
            uint32_t preCount = 0U;
            uint32_t assistInfoIdx = expertIdx * moeDistributeCombineSetupInfo_->epWorldSize + dstRankId;
            if (likely(assistInfoIdx > 0U)) {
                preCount = assistInfoForCombineLocal.GetValue(assistInfoIdx - 1U);
            }
            const uint32_t curCount = assistInfoForCombineLocal.GetValue(assistInfoIdx);
            uint32_t curTokenNum = curCount - preCount;
            if (curTokenNum == 0U) {
                continue;
            }
            GM_ADDR srcAddr = expandXGM_ + static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_;
            GM_ADDR dstAddr = GetUrmaAddrByRankId(dstRankId, expertIdx) + epDataOffsetOnWin_;
            uint64_t lengthU64 = axisHExpandXTypeSize_ * static_cast<uint64_t>(curTokenNum);
            int32_t dataRet = hcomm_.WriteNbi<false>(channelId, dstAddr, srcAddr, static_cast<uint32_t>(lengthU64));
            (void)dataRet;
        }
        hcomm_.WriteNbi<false>(channelId, dstStatusAddr, statusFlagGM_, UB_ALIGN);
        hcomm_.Commit(channelId);
    }
}

template <TemplateMC2TypeClass>
__aicore__ __attribute__((noinline)) void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::CommLocalThenRemote(
    const LocalTensor<int32_t> &assistInfoForCombineLocal, uint32_t curRankExpertNum)
{
    if (unlikely(epRankId_ >= startRankId_ && epRankId_ < endRankId_)) {
        CurRankComm(assistInfoForCombineLocal, curRankExpertNum);
    }
    IssueRemoteWrites(assistInfoForCombineLocal, curRankExpertNum);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Communication()
{
    uint32_t curRankExpertNum = (isShardExpert_) ? 1U : moeDistributeCombineSetupInfo_->moeExpertPerRankNum;
    AssistInfoLocalCopy();
    LocalTensor<int32_t> assistInfoForCombineLocal = assistInfoQueue_.DeQue<int32_t>();
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();

    uint32_t assistInfoSize = Ceil(moeSendNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    LocalTensor<uint8_t> assistStorage = assistInfoForCombineLocal.ReinterpretCast<uint8_t>();
    hcommTensor_ = assistStorage[assistInfoSize];
    int32_t ret = hcomm_.Init(hcommTensor_, static_cast<uint32_t>(HCOMM_INIT_SIZE));
    if (ret != HCOMM_SUCCESS) {
        printf("[ERROR] hcomm_.Init Init hcomm failed, ret=%d\n", ret);
        assistInfoQueue_.FreeTensor<int32_t>(assistInfoForCombineLocal);
        return;
    }

#if MC2_CSU_OPT_OVERLAP_REMOTE
    if (moeDistributeCombineSetupInfo_->bs <= 16U) {
        // bs8/16：与达标版本相同，远端先行 + 单 buffer，全部强制内联进本函数
        IssueRemoteWrites(assistInfoForCombineLocal, curRankExpertNum);
        if (unlikely(epRankId_ >= startRankId_ && epRankId_ < endRankId_)) {
            CurRankComm(assistInfoForCombineLocal, curRankExpertNum);
        }
    } else {
        CommLocalThenRemote(assistInfoForCombineLocal, curRankExpertNum);
    }
#else
    CommLocalThenRemote(assistInfoForCombineLocal, curRankExpertNum);
#endif

    assistInfoQueue_.FreeTensor<int32_t>(assistInfoForCombineLocal);
}

template <TemplateMC2TypeClass>
__aicore__ inline __attribute__((always_inline)) void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::CurRankComm(
    const LocalTensor<int32_t> &assistInfoForCombineLocal, uint32_t curRankExpertNum)
{
    uint32_t curTokenNum = 0;
    for (uint32_t expertIdx = 0U; expertIdx < curRankExpertNum; ++expertIdx) {
        uint32_t preCount = 0U;
        uint32_t assistInfoIdx = expertIdx * moeDistributeCombineSetupInfo_->epWorldSize + epRankId_;
        if (likely(assistInfoIdx > 0U)) {
            preCount = assistInfoForCombineLocal.GetValue(assistInfoIdx - 1U);
        }
        curTokenNum = assistInfoForCombineLocal.GetValue(assistInfoIdx) - preCount;
        if (unlikely(curTokenNum == 0U)) {
            continue;
        }

        GM_ADDR srcAddr = expandXGM_ + static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_;
        GM_ADDR dstAddr = GetUrmaAddrByRankId(epRankId_, expertIdx) + epDataOffsetOnWin_;

        GlobalTensor<uint8_t> selfDataSrcTensor;
        GlobalTensor<uint8_t> selfDataDstTensor;
        selfDataSrcTensor.SetGlobalBuffer((__gm__ uint8_t *)srcAddr);
        selfDataDstTensor.SetGlobalBuffer((__gm__ uint8_t *)dstAddr);

        uint64_t sendBytes = static_cast<uint64_t>(curTokenNum) * axisHExpandXTypeSize_;
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        LocalTensor<uint8_t> expertTokenTmpU8 = expertTokenTmpQueue_.AllocTensor<uint8_t>();
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(remain_ub_space), 0U, 0U, 0U};

        uint64_t i = 0;
        for (; sendBytes > remain_ub_space; sendBytes -= remain_ub_space) {
            DataCopyPad(expertTokenTmpU8, selfDataSrcTensor[i * remain_ub_space], copyParams, padParams);
            expertTokenTmpQueue_.EnQue(expertTokenTmpU8);
            expertTokenTmpU8 = expertTokenTmpQueue_.DeQue<uint8_t>();
            DataCopyPad(selfDataDstTensor[i * remain_ub_space], expertTokenTmpU8, copyParams);
            ++i;

            AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        }

        copyParams.blockLen = sendBytes;
        DataCopyPad(expertTokenTmpU8, selfDataSrcTensor[i * remain_ub_space], copyParams, padParams);
        expertTokenTmpQueue_.EnQue(expertTokenTmpU8);
        expertTokenTmpU8 = expertTokenTmpQueue_.DeQue<uint8_t>();
        DataCopyPad(selfDataDstTensor[i * remain_ub_space], expertTokenTmpU8, copyParams);

        expertTokenTmpQueue_.FreeTensor<uint8_t>(expertTokenTmpU8);
    }

    AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>();
    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(GetUrmaStateAddrByRankId(epRankId_) + epStateOffsetOnWin_));
    selfStatusTensor.SetValue(0, static_cast<int32_t>(STATUS_FLAG_VALUE));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        // 空闲核不参与通信，但仍要走到下面的 SyncAll，否则其他核会等不到它
        if (startRankId_ < moeDistributeCombineSetupInfo_->epWorldSize) {
            BuffInit();
            Communication();
        }
    }
}
} // namespace MoeDistributeCombineSetupImpl

#endif // MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H
