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
 * \file moe_distribute_combine_teardown_arch35.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_TEARDOWN_ARCH35_H
#define MOE_DISTRIBUTE_COMBINE_TEARDOWN_ARCH35_H

#if __has_include("../../common/mc2_kernel_utils.h")
#include "../../common/mc2_kernel_utils.h"
#else
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "kernel_operator.h"
#include "adv_api/hcomm/hcomm.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_combine_teardown_tiling_data.h"
#if __has_include("../../moe_distribute_combine_setup/moe_distribute_base.h")
#include "../../moe_distribute_combine_setup/moe_distribute_base.h"
#else
#include "../../../moe_distribute_combine_setup/op_kernel/moe_distribute_base.h"
#endif
#include "../../../common/op_kernel/mc2_moe_context.h"

namespace MoeDistributeCombineTeardownImpl {

// 置 1 打开状态通路调试：打印本核 C_S / 等待的状态槽，并在长时间等不到时 dump 一次实际读到的值。
// 与 combine_setup 的 COMBINE_SETUP_DEBUG_DST_STATUS 配套使用，用于比对两侧 C_S 是否同步。
#ifndef COMBINE_TEARDOWN_DEBUG_WAIT
#define COMBINE_TEARDOWN_DEBUG_WAIT 0
#endif
// 轮询多少次还没等到就 dump 一次现场
#define COMBINE_TEARDOWN_WAIT_DUMP_THRESHOLD 1000000U

using namespace AscendC;
using namespace Mc2Aclnn;

#define TemplateMC2TypeClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeFunc ExpandXType, ExpandIdxType

template <TemplateMC2TypeClass>
class MoeDistributeCombineTeardown {
    constexpr static uint8_t BUFFER_NUM = 2;              // 多buf
    constexpr static uint64_t STATE_OFFSET = 512U;        // 状态空间偏移地址
    constexpr static uint32_t STATE_SIZE = 1024U * 1024;  // 1M
    constexpr static uint32_t UB_ALIGN = 32U;             // UB按32字节对齐
    constexpr static uint64_t STATE_SIZE_PER_CORE = 512U; // 每 aiv 控制块大小（D_S/C_S/flag）
    constexpr static uint32_t STATE_COUNT_THRESHOLD = 512U; // moeExpertNumPerRank*epWorldSize状态数阈值

public:
    __aicore__ inline MoeDistributeCombineTeardown(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR expandX, GM_ADDR quantExpandX, GM_ADDR expertIds,
        GM_ADDR expandIdx, GM_ADDR expertScales, GM_ADDR commCmdInfo, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
        GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineTeardownTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AlltoAllBuffInit();
    __aicore__ inline void CopyIn();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void TokenSplitCoreCal();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void FlipDataState();

    __aicore__ GM_ADDR GetUrmaAddrByRankId(const int32_t rankId, const uint8_t expertLocalId = 0U)
    {
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[rankId]) + STATE_SIZE + winDataSizeOffset_ +
            expertPerSizeOnWin_ * static_cast<uint64_t>(expertLocalId);
    }

    __aicore__ GM_ADDR GetUrmaStateAddrByRankId(const int32_t rankId)
    {
        // C 专用状态区，与 D bank 物理隔离；本轮使用 C_S
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[rankId]) + COMBINE_STATUS_BASE +
            COMBINE_STATE_BANK_STRIDE * static_cast<uint64_t>(dataState_);
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> expertIdsGlobal_;
    GlobalTensor<ExpandIdxType> expandIdxGlobal_;
    GlobalTensor<float> expertScalesGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;
    GlobalTensor<ExpandXType> rowTmpGlobal_;
    GM_ADDR workspaceGM_;
    GM_ADDR epWindowGM_;
    GM_ADDR epStatusSpaceGM_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    const MoeDistributeCombineTeardownInfo *moeDistributeCombineTeardownInfo_{nullptr};
    uint32_t axisMaxBS_{0};
    uint32_t coreIdx_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t moeSendNum_{0}; // moeExpertPerRankNum * epWorldSize
    uint64_t axisHFloatSize_{0};
    uint64_t axisHExpandXTypeSize_{0};
    uint32_t bsKNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t sendRankNum_{0};
    uint32_t beginIndex_{0};
    uint32_t endIndex_{0};
    uint32_t dataState_{0};
    uint64_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t sharedExpertDataSizeOffset_{0};
    uint32_t epRankId_{0};
    GM_ADDR dataStateGM_{nullptr}; // 本核 dataState 0/1 标识地址，WaitDispatch 成功后再翻转

    TQue<QuePosition::VECIN, 1> moeSumQueue_;
    TQue<QuePosition::VECIN, 1> expertIdsQueue_;
    TQue<QuePosition::VECIN, 1> expandScalesQueue_;
    TQue<QuePosition::VECIN, 1> indexCountsQueue_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> tokenBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_; // gather mask输出buf
    TBuf<> gatherTmpBuf_;
    TBuf<> statusSumOutBuf_;

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::Init(GM_ADDR context,
    GM_ADDR /*expandX*/, GM_ADDR /*quantExpandX*/, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR expertScales,
    GM_ADDR /*commCmdInfo*/, GM_ADDR /*xActiveMask*/, GM_ADDR /*sharedExpertX*/, GM_ADDR XOut, GM_ADDR /*workspaceGM*/,
    TPipe *pipe, const MoeDistributeCombineTeardownTilingData *tilingData)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    moeDistributeCombineTeardownInfo_ = &(tilingData->moeDistributeCombineTeardownInfo);
    mc2Context_ = (__gm__ Mc2MoeContext* )context;
    epRankId_ = mc2Context_->epRankId;

    // 获取win状态区地址，并保证数据一致
    // 在1M中选择512K偏移后的1.5k空间记录本卡历史状态
    // C_S 仅在 WaitDispatch 成功后再翻转，与 D_S 独立
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankId_]);
    dataStateGM_ = statusDataSpaceGm + STATE_WIN_OFFSET + STATE_SIZE_PER_CORE * static_cast<uint64_t>(coreIdx_) +
                   CTRL_COMBINE_STATE_OFFSET;
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)dataStateGM_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);

    dataState_ = selfDataStatusTensor(0); // C_S，本轮仍使用该 bank

    expertIdsGlobal_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandIdxGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    expertScalesGlobal_.SetGlobalBuffer((__gm__ float *)expertScales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);

    axisMaxBS_ = moeDistributeCombineTeardownInfo_->globalBs / moeDistributeCombineTeardownInfo_->epWorldSize;
    moeSendNum_ =
        moeDistributeCombineTeardownInfo_->epWorldSize * moeDistributeCombineTeardownInfo_->moeExpertPerRankNum;

    bsKNum_ = moeDistributeCombineTeardownInfo_->bs * moeDistributeCombineTeardownInfo_->k;
    axisHFloatSize_ = static_cast<uint64_t>(moeDistributeCombineTeardownInfo_->h) *
                      static_cast<uint64_t>(sizeof(float)); // 一个token占用内存(float)
    axisHExpandXTypeSize_ = static_cast<uint64_t>(moeDistributeCombineTeardownInfo_->h) *
                            static_cast<uint64_t>(sizeof(ExpandXType)); // 一个token占用内存(输入type)
    expertPerSizeOnWin_ = static_cast<uint64_t>(axisMaxBS_) * axisHExpandXTypeSize_; // 每个卡的数据在win区占用空间
    sharedExpertDataSizeOffset_ = static_cast<uint64_t>(moeDistributeCombineTeardownInfo_->moeExpertPerRankNum) *
                                  static_cast<uint64_t>(moeDistributeCombineTeardownInfo_->sharedExpertRankNum) *
                                  expertPerSizeOnWin_; // 共享专家占用空间

    // C 固定占用数据区后半，区内按 C_S 做 totalWin/4 乒乓
    {
        const uint64_t totalWin = static_cast<uint64_t>(moeDistributeCombineTeardownInfo_->totalWinSize);
        winDataSizeOffset_ = (totalWin / 2ULL) + static_cast<uint64_t>(dataState_) * (totalWin / 4ULL);
    }
    stateOffset_ = (moeSendNum_ > STATE_COUNT_THRESHOLD) ? (STATE_OFFSET / 2ULL) : STATE_OFFSET;

    epWindowGM_ = GetUrmaAddrByRankId(epRankId_);
    epStatusSpaceGM_ = GetUrmaStateAddrByRankId(epRankId_);
#if COMBINE_TEARDOWN_DEBUG_WAIT
    // 与 combine_setup 的 [C_S] 行逐核比对：同一 aiv 上两边的 C_S 必须一致，
    // 且本卡这里的 statusBank 必须等于对端 setup 打印的 addr 去掉 stateOff 后的基址。
    AscendC::printf("[C_S] rank=%u aiv=%u C_S=%u statusBank=0x%llx winDataOff=0x%llx totalWin=0x%llx\n", epRankId_,
                    coreIdx_, dataState_, static_cast<unsigned long long>(static_cast<uint64_t>(epStatusSpaceGM_)),
                    static_cast<unsigned long long>(winDataSizeOffset_),
                    static_cast<unsigned long long>(moeDistributeCombineTeardownInfo_->totalWinSize));
#endif
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange<ExpandXType>((__gm__ ExpandXType *)(epWindowGM_),
                                   moeDistributeCombineTeardownInfo_->totalWinSize / 4ULL);
    OOMCheckAddrRange<float>((__gm__ float *)(epStatusSpaceGM_), COMBINE_STATE_BANK_STRIDE);
#endif

    SplitCoreCal();      // rank分核计算
    TokenSplitCoreCal(); // token分核计算
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::SplitCoreCal()
{
    // 对worldSize按卡分核，得到每个核上处理的卡的数量
    sendRankNum_ = moeDistributeCombineTeardownInfo_->epWorldSize / moeDistributeCombineTeardownInfo_->aivNum;
    uint32_t remainderRankNum =
        moeDistributeCombineTeardownInfo_->epWorldSize % moeDistributeCombineTeardownInfo_->aivNum;
    startRankId_ = sendRankNum_ * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        ++sendRankNum_;
        startRankId_ += coreIdx_;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::TokenSplitCoreCal()
{
    // token分核计算
    uint32_t tokenPerAivNum = moeDistributeCombineTeardownInfo_->bs / moeDistributeCombineTeardownInfo_->aivNum;
    uint32_t remainderToken = moeDistributeCombineTeardownInfo_->bs % moeDistributeCombineTeardownInfo_->aivNum;
    beginIndex_ = tokenPerAivNum * coreIdx_;
    if (coreIdx_ < remainderToken) {
        ++tokenPerAivNum;
        beginIndex_ += coreIdx_;
    } else {
        beginIndex_ += remainderToken;
    }
    endIndex_ = beginIndex_ + tokenPerAivNum;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::AlltoAllBuffInit()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(indexCountsQueue_, BUFFER_NUM, bsKNum_ * static_cast<uint32_t>(sizeof(ExpandIdxType)));
    tpipe_->InitBuffer(expertIdsQueue_, BUFFER_NUM, bsKNum_ * static_cast<uint32_t>(sizeof(int32_t)));
    tpipe_->InitBuffer(expandScalesQueue_, BUFFER_NUM, bsKNum_ * static_cast<uint32_t>(sizeof(float)));
    tpipe_->InitBuffer(moeSumQueue_, BUFFER_NUM, axisHExpandXTypeSize_);

    tpipe_->InitBuffer(statusBuf_, sendRankNum_ * UB_ALIGN); // 等状态需要占用sendRankNum_个DataBlock(32Bytes)
    tpipe_->InitBuffer(tokenBuf_, static_cast<uint32_t>(axisHExpandXTypeSize_));
    tpipe_->InitBuffer(rowTmpFloatBuf_, static_cast<uint32_t>(axisHFloatSize_));
    tpipe_->InitBuffer(sumFloatBuf_, static_cast<uint32_t>(axisHFloatSize_));
    tpipe_->InitBuffer(gatherTmpBuf_, static_cast<uint32_t>(sizeof(uint32_t)));
    tpipe_->InitBuffer(statusSumOutBuf_, static_cast<uint32_t>(sizeof(float)));

    uint32_t waitStatusMaxSize = moeDistributeCombineTeardownInfo_->epWorldSize * static_cast<uint32_t>(sizeof(float));
    tpipe_->InitBuffer(gatherMaskOutBuf_, waitStatusMaxSize);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::WaitDispatch()
{
    if (startRankId_ >= moeDistributeCombineTeardownInfo_->epWorldSize) {
        return;
    }

    LocalTensor<float> statusTensor = statusBuf_.Get<float>();
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = gatherTmpBuf_.Get<uint32_t>();
    LocalTensor<float> statusSumOutTensor = statusSumOutBuf_.Get<float>();
    GlobalTensor<float> epStatusSpaceGlobal;
    epStatusSpaceGlobal.SetGlobalBuffer((__gm__ float *)epStatusSpaceGM_);

    // 计算当前核期望target值
    gatherTmpTensor.SetValue(0, 1);
    AscendC::SyncFunc<AscendC::HardEvent::S_V>(); // 等gatherTmpTensor标量写，后续GatherMask使用

    uint64_t rsvdCnt = 0;
    DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1,
                               static_cast<uint16_t>((moeSendNum_ > STATE_COUNT_THRESHOLD) ? 7 : 15),
                               0}; // srcStride为15个DataBlock
    DataCopyParams clearParams{
        static_cast<uint16_t>(sendRankNum_), 1, 0,
        static_cast<uint16_t>((moeSendNum_ > STATE_COUNT_THRESHOLD) ? 7 : 15)}; // dstStride为15个DataBlock

    float sumTarget = static_cast<float>(sendRankNum_);
    float sumOfFlag = -1.0f;
    float minTarget = sumTarget - static_cast<float>(0.5);
    float maxTarget = sumTarget + static_cast<float>(0.5);
    SumParams sumParams{1,
                        Ceil(sendRankNum_ * static_cast<uint32_t>(sizeof(float)), UB_ALIGN) * UB_ALIGN /
                            static_cast<uint32_t>(sizeof(float)),
                        sendRankNum_};

    // 循环，累加本卡状态区状态
    auto index = static_cast<uint64_t>(startRankId_) * stateOffset_ / static_cast<uint64_t>(sizeof(float));
#if COMBINE_TEARDOWN_DEBUG_WAIT
    uint32_t pollCount = 0U;
    bool dumped = false;
#endif
    while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
        DataCopy(statusTensor, epStatusSpaceGlobal[index], intriParams);
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(gatherMaskOutTensor, statusTensor, gatherTmpTensor, true, 1,
                   {1, static_cast<uint16_t>(sendRankNum_), 1, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
        AscendC::SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
#if COMBINE_TEARDOWN_DEBUG_WAIT
        // 卡住时 dump 一次现场：逐槽打印，缺哪个 srcRank 的 flag 一目了然
        if ((++pollCount >= COMBINE_TEARDOWN_WAIT_DUMP_THRESHOLD) && !dumped) {
            dumped = true;
            AscendC::printf("[WAIT] rank=%u aiv=%u C_S=%u startRank=%u num=%u bank=0x%llx sum=%d target=%d\n",
                            epRankId_, coreIdx_, dataState_, startRankId_, sendRankNum_,
                            static_cast<unsigned long long>(static_cast<uint64_t>(epStatusSpaceGM_)),
                            static_cast<int32_t>(sumOfFlag), static_cast<int32_t>(sumTarget));
            LocalTensor<int32_t> rawStatus = statusTensor.ReinterpretCast<int32_t>();
            for (uint32_t i = 0U; i < sendRankNum_; ++i) {
                // 每个槽在 UB 里占一个 32B DataBlock，取其首个 int32
                AscendC::printf("[WAIT]   srcRank=%u slot=0x%x raw=0x%x\n", startRankId_ + i,
                                static_cast<uint32_t>((startRankId_ + i) * stateOffset_),
                                static_cast<uint32_t>(rawStatus.GetValue(i * (UB_ALIGN / sizeof(int32_t)))));
            }
        }
#endif
    }

    Duplicate<float>(statusTensor, 0, sendRankNum_ * UB_ALIGN / sizeof(float));
    AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(epStatusSpaceGlobal[index], statusTensor, clearParams); // 清状态
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::FlipDataState()
{
    // WaitDispatch 成功（或空闲核跳过等待）后再切换 C_S，供下一轮 combine_setup 使用
    GlobalTensor<int32_t> selfDataStatusTensor;
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)dataStateGM_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    selfDataStatusTensor(0) = static_cast<int32_t>(1U - dataState_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::CopyIn()
{
    LocalTensor<ExpandIdxType> indexCountsTensor = indexCountsQueue_.AllocTensor<ExpandIdxType>();
    LocalTensor<int32_t> expertIdsTensor = expertIdsQueue_.AllocTensor<int32_t>();
    LocalTensor<float> expandScalesTensor = expandScalesQueue_.AllocTensor<float>();
    DataCopyExtParams bskParams = {1U, static_cast<uint32_t>(bsKNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    DataCopyPad(indexCountsTensor, expandIdxGlobal_, bskParams, copyPadParams);
    DataCopyPad(expertIdsTensor, expertIdsGlobal_, bskParams, copyPadParams);
    DataCopyPad(expandScalesTensor, expertScalesGlobal_, bskParams, copyPadFloatParams);
    indexCountsQueue_.EnQue(indexCountsTensor);
    expertIdsQueue_.EnQue(expertIdsTensor);
    expandScalesQueue_.EnQue(expandScalesTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::LocalWindowCopy()
{
    LocalTensor<ExpandIdxType> indexCountsTensor = indexCountsQueue_.DeQue<ExpandIdxType>();
    LocalTensor<int32_t> expertIdsTensor = expertIdsQueue_.DeQue<int32_t>();
    LocalTensor<float> expandScalesTensor = expandScalesQueue_.DeQue<float>();
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();

    LocalTensor<float> rowTmpFloatTensor = rowTmpFloatBuf_.Get<float>();
    LocalTensor<float> sumFloatBufTensor = sumFloatBuf_.Get<float>();

    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(axisHExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyPadExtParams<ExpandXType> padParams{false, 0U, 0U, 0U};

    for (uint32_t tokenIndex = beginIndex_; tokenIndex < endIndex_; ++tokenIndex) { // 处理当前核要处理的token
        uint32_t bsKStartIdx = tokenIndex * moeDistributeCombineTeardownInfo_->k;
        ExpandIdxType indexCount = 0;
        int32_t moeExpert = 0;
        float scaleVal = 0.0f;

        Duplicate(sumFloatBufTensor, 0.0f,
                  moeDistributeCombineTeardownInfo_->h); // 清零，sumFloatBufLocal保存累加结果
        LocalTensor<ExpandXType> tmpUb;
        for (uint32_t idx = bsKStartIdx; idx < bsKStartIdx + moeDistributeCombineTeardownInfo_->k; ++idx) {
            // 循环k次，累加token
            indexCount = indexCountsTensor.GetValue(idx); // 根据expandIdx获取当前是第几个专家
            moeExpert = expertIdsTensor.GetValue(idx);    // 当前token的专家id
            scaleVal = expandScalesTensor.GetValue(idx);  // 当前专家的量化参数
            GM_ADDR wAddr = epWindowGM_ + sharedExpertDataSizeOffset_ +
                            expertPerSizeOnWin_ * static_cast<uint64_t>(moeExpert) +
                            axisHExpandXTypeSize_ * static_cast<uint64_t>(indexCount); // 当前token的值对应的win区地址

            rowTmpGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)wAddr);
            tmpUb = moeSumQueue_.AllocTensor<ExpandXType>();
            DataCopyPad(tmpUb, rowTmpGlobal_, copyParams, padParams);
            moeSumQueue_.EnQue(tmpUb);
            tmpUb = moeSumQueue_.DeQue<ExpandXType>();
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();

            // 转为float
            Cast(rowTmpFloatTensor, tmpUb, AscendC::RoundMode::CAST_NONE, moeDistributeCombineTeardownInfo_->h);
            PipeBarrier<PIPE_V>();
            // 乘量化系数
            Muls(rowTmpFloatTensor, rowTmpFloatTensor, scaleVal, moeDistributeCombineTeardownInfo_->h);
            PipeBarrier<PIPE_V>();
            // 累加token
            Add(sumFloatBufTensor, sumFloatBufTensor, rowTmpFloatTensor, moeDistributeCombineTeardownInfo_->h);

            moeSumQueue_.FreeTensor<ExpandXType>(tmpUb);
        }

        // 结果搬出
        PipeBarrier<PIPE_V>(); // 等sumFloatBufTensor的vector操作
        LocalTensor<ExpandXType> sumBufLocal = tokenBuf_.Get<ExpandXType>();
        Cast(sumBufLocal, sumFloatBufTensor, AscendC::RoundMode::CAST_RINT,
             moeDistributeCombineTeardownInfo_->h); // 转成对应数据类型
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(expandOutGlobal_[tokenIndex * moeDistributeCombineTeardownInfo_->h], sumBufLocal, copyParams);
    }

    indexCountsQueue_.FreeTensor<ExpandIdxType>(indexCountsTensor);
    expertIdsQueue_.FreeTensor<int32_t>(expertIdsTensor);
    expandScalesQueue_.FreeTensor<float>(expandScalesTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理{
        AlltoAllBuffInit();
        WaitDispatch();
        FlipDataState();
        SyncAll<true>();

        CopyIn();
        LocalWindowCopy(); // 拷出数据
    }
}

} // namespace MoeDistributeCombineTeardownImpl

#endif // MOE_DISTRIBUTE_COMBINE_TEARDOWN_ARCH35_H
