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
 * \file block_prologue_all_gather.h
 * \brief AllGather MTE communication Prologue for Blaze GemmUniversal.
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "inc/kernel/moe_distribute_base.h"
#include "tensor_api/tensor.h"
#include "all_gather_matmul_tiling_data.h"

namespace Blaze {
namespace Gemm {
namespace Block {

using namespace AscendC;

template <typename CommunicationType, typename ScalesType, uint32_t SyncMode>
class BlockPrologueAllGather {
public:
    static constexpr uint32_t NO_TILE_K                       = 1U;
    static constexpr uint64_t UB_ALIGN_BYTES                  = 32U;
    static constexpr uint32_t FLOAT_UB_ALIGN_NUM              = 8U;
    static constexpr uint32_t BUFFER_NUM                      = 2U;
    static constexpr uint64_t X_BLOCK_BYTES                   = 512U;
    static constexpr uint32_t X_PER_BLOCK_NUM                 = 512U;
    static constexpr uint64_t WIN_ADDR_ALIGN                  = 512UL;
    static constexpr uint64_t DATA_START_OFFSET               = 1024UL * 1024UL;
    static constexpr uint64_t UB_RESERVE                      = 10UL * 1024UL;
    static constexpr uint32_t kSyncMode                       = SyncMode;
    static constexpr uint32_t WIN_FLAG_MAX_VAL                = 1UL;
    static constexpr uint16_t CROSS_CORE_FLAG_NUM_NINE        = 9;
    static constexpr uint16_t CROSS_CORE_INNER_CUBE_VEC_SYNC  = 0x2;
    static constexpr float    STATUS_FLAG_VAL                 = 1.0f;
    static constexpr float    STATUS_FLAG_THREDSHOLD          = 0.5f;
    static constexpr uint32_t STATUS_POLL_MAX_RETRY           = 1000000UL;

    static constexpr uint16_t EVT_ID = 1;
    static constexpr uint16_t EVT_ID_PING = 2;
    static constexpr uint16_t EVT_ID_PONG = 3;
    static constexpr uint16_t EVT_ID_SCALE_PING = 4;
    static constexpr uint16_t EVT_ID_SCALE_PONG = 5;

    using MakeLayoutGM = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn,
                                                        AscendC::Te::LayoutTraitDefault<int8_t>>;   // todo: 不确定是否可以直接用fp8_e4m3fn_t
    using MakeLayoutUB = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn,
                                                        AscendC::Te::LayoutTraitDefault<int8_t>>;
    using MakeLayoutScaleGM = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn,
                                                             AscendC::Te::LayoutTraitDefault<ScalesType>>;
    using MakeLayoutScaleUB = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn,
                                                             AscendC::Te::LayoutTraitDefault<ScalesType>>;
    using MakeLayoutStatusGM = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn,  
                                                              AscendC::Te::LayoutTraitDefault<float>>;  // 用于状态区逻辑
    using MakeLayoutStatusUB = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn,
                                                              AscendC::Te::LayoutTraitDefault<float>>;

    using CopyGM2UB_t = decltype(AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{}));
    using CopyUB2GM_t = decltype(AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{}));

public:
    struct Params {
        GM_ADDR inputXAddr{nullptr};
        GM_ADDR inputXScaleAddr{nullptr};
        GM_ADDR allGatherDataOutAddr{nullptr};
        GM_ADDR allGatherScalesOutAddr{nullptr};
        __gm__ allGatherTilingData* tilingData{nullptr};
        GM_ADDR hcclContext{nullptr};
    };

    __aicore__ inline BlockPrologueAllGather()  {}
    __aicore__ inline ~BlockPrologueAllGather() {}

    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run();
    __aicore__ inline GM_ADDR GetSelfDataAddrGm()  { return selfRankAddr_; }
    __aicore__ inline GM_ADDR GetSelfScaleAddrGm() { return selfScaleAddr_; }

private:
    __aicore__ inline GM_ADDR WinDataAddr(uint32_t rankId);
    __aicore__ inline GM_ADDR WinStatusAddr(uint32_t rankId);

    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void CopyLocalData();
    __aicore__ inline void ExecuteAllGather();
    __aicore__ inline void ReadDataBlock(uint64_t curXOffset, uint32_t mCnt, uint32_t bufIdx = 0);

    __gm__ Mc2Kernel::HcclOpParam* hcclContext_{nullptr};
    uint32_t aivId_{0};
    uint32_t aicId_{0};
    uint32_t curRankId_{0};
    uint32_t rankSize_{0};
    uint64_t winDataSize_{0};

    uint64_t M_{0};
    uint64_t K_{0};
    uint64_t xSize_{0};
    uint64_t scaleSize_{0};
    uint64_t scaleChunkSize_{0};
    uint32_t scaleTotalChunks_{0};
    uint64_t scaleUbPaddedChunkN_{0};

    uint32_t sendCoreNumPerRank_{0};
    uint32_t remoteRankId_{0};
    uint64_t tileK_{0};
    uint32_t kDim_{0};
    uint32_t mDim_{0};
    uint32_t mBlockIdx_{0};
    uint32_t kBlockIdx_{0};
    uint32_t mMteCoreM_{0};
    uint32_t kMteCoreK_{0};
    uint32_t coreInnerMIndex_{0};
    uint32_t kStartIndex_{0};

    uint64_t ubWriteStateOffset_{0};
    uint64_t ubReadStateOffset_{0};
    uint64_t ubResetStateOffset_{0};
    uint64_t ubScaleOffset_{0};
    uint64_t ubScalePingOffset_{0};
    uint64_t ubScalePongOffset_{0};
    uint64_t ubXPingOffset_{0};
    uint64_t ubXPongOffset_{0};
    uint64_t xBufSize_{0};
    uint64_t paddingK_{0};

    GM_ADDR selfRankAddr_{nullptr};
    GM_ADDR selfScaleAddr_{nullptr};

    GM_ADDR inputXAddr_{nullptr};
    GM_ADDR inputXScaleAddr_{nullptr};
    GM_ADDR allGatherDataOutAddr_{nullptr};
    GM_ADDR allGatherScalesOutAddr_{nullptr};

    CopyGM2UB_t copyGM2UB_ = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
    CopyUB2GM_t copyUB2GM_ = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});

    AscendC::GlobalTensor<uint32_t> selfWinFlagGMTensor_;
};

template <typename CommunicationType, typename ScalesType, uint32_t SyncMode>
__aicore__ inline GM_ADDR BlockPrologueAllGather<CommunicationType, ScalesType, SyncMode>::WinDataAddr(
    uint32_t rankId)
{
    return (GM_ADDR)(hcclContext_->windowsIn[rankId] + DATA_START_OFFSET);
}

template <typename CommunicationType, typename ScalesType, uint32_t SyncMode>
__aicore__ inline GM_ADDR BlockPrologueAllGather<CommunicationType, ScalesType, SyncMode>::WinStatusAddr(
    uint32_t rankId)
{
    return (GM_ADDR)(hcclContext_->windowsIn[rankId]);
}

template <typename CommunicationType, typename ScalesType, uint32_t SyncMode>
__aicore__ inline void BlockPrologueAllGather<CommunicationType, ScalesType, SyncMode>::Init(
    const Params& params)
{
    hcclContext_ = (__gm__ Mc2Kernel::HcclOpParam*)params.hcclContext;
    inputXAddr_ = params.inputXAddr;
    inputXScaleAddr_ = params.inputXScaleAddr;
    allGatherDataOutAddr_ = params.allGatherDataOutAddr;
    allGatherScalesOutAddr_ = params.allGatherScalesOutAddr;

    M_ = params.tilingData->M;
    K_ = params.tilingData->Ka;
    xSize_ = params.tilingData->xSize;
    scaleSize_ = params.tilingData->scaleSize;
    sendCoreNumPerRank_ = params.tilingData->sendCoreNumPerRank;
    tileK_ = params.tilingData->mteTileK;

    uint64_t totalUb = params.tilingData->ubSize;
    uint64_t availableUb = totalUb - UB_RESERVE;

    aivId_ = GetBlockIdx();
    aicId_ = aivId_ / GetTaskRation();

    rankSize_ = Mc2Kernel::GetRankDim(hcclContext_);
    curRankId_ = Mc2Kernel::GetRankId(hcclContext_);
    winDataSize_ = Blaze::Gemm::CeilAlign(rankSize_ * xSize_, WIN_ADDR_ALIGN);

    kDim_ = params.tilingData->mteKSplitNum;
    mDim_ = params.tilingData->mteMSplitNum;
    uint32_t modCoreIndex = aicId_ % sendCoreNumPerRank_;
    remoteRankId_ = aicId_ / sendCoreNumPerRank_;
    mBlockIdx_ = modCoreIndex % mDim_;
    kBlockIdx_ = modCoreIndex / mDim_;

    mMteCoreM_ = params.tilingData->mteMSplitSize;
    kMteCoreK_ = params.tilingData->mteKSplitSize;

    uint32_t mTailNum = params.tilingData->mteMSplitLargeBlockNum;
    if (mBlockIdx_ < mTailNum) {
        mMteCoreM_ += 1;
        coreInnerMIndex_ = mBlockIdx_ * mMteCoreM_;
    } else {
        coreInnerMIndex_ = mBlockIdx_ * mMteCoreM_ + mTailNum;
    }
    kStartIndex_ = kBlockIdx_ * X_PER_BLOCK_NUM;

    // 状态区ub
    ubWriteStateOffset_ = 0;
    ubReadStateOffset_  = UB_ALIGN_BYTES;
    ubResetStateOffset_ = UB_ALIGN_BYTES + rankSize_ * UB_ALIGN_BYTES;

    uint64_t workStart = UB_ALIGN_BYTES + 2 * rankSize_ * UB_ALIGN_BYTES;
    ubScaleOffset_ = workStart;
    uint64_t workAvailable = (availableUb > workStart) ? (availableUb - workStart) : 0;

    // scale 搬运：double buffer，工作区一分为二，各 buffer 吃满一半
    uint64_t scaleHalf = (workAvailable / BUFFER_NUM / UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    scaleChunkSize_ = scaleHalf;
    if (scaleChunkSize_ == 0) {
        scaleChunkSize_ = UB_ALIGN_BYTES;
    }
    scaleTotalChunks_ = (scaleSize_ + scaleChunkSize_ - 1) / scaleChunkSize_;
    ubScalePingOffset_ = workStart;
    ubScalePongOffset_ = workStart + scaleChunkSize_;

    scaleUbPaddedChunkN_ = scaleChunkSize_ / sizeof(ScalesType);
    paddingK_ = Blaze::Gemm::CeilAlign(K_ * sizeof(CommunicationType), UB_ALIGN_BYTES) / sizeof(CommunicationType);

    // x 搬运：double buffer，同样吃满整个工作区（各 buffer 为工作区的一半）
    uint64_t totalAvailable = workAvailable / X_BLOCK_BYTES * X_BLOCK_BYTES;
    uint64_t demandSpace = Blaze::Gemm::CeilAlign(xSize_, X_BLOCK_BYTES);
    uint64_t availableSpace = totalAvailable / BUFFER_NUM / X_BLOCK_BYTES * X_BLOCK_BYTES;
    xBufSize_ = (availableSpace < demandSpace) ? availableSpace : demandSpace;
    if (xBufSize_ < X_BLOCK_BYTES) {
        xBufSize_ = X_BLOCK_BYTES;
    }
    ubXPingOffset_ = workStart;
    ubXPongOffset_ = workStart + xBufSize_;

    uint64_t xTotalBytes = M_ * K_ * sizeof(int8_t);
    uint64_t winScaleOffset = Blaze::Gemm::CeilDiv(rankSize_ * xTotalBytes, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    selfRankAddr_  = WinDataAddr(curRankId_);
    selfScaleAddr_ = (GM_ADDR)selfRankAddr_ + winScaleOffset;
}

template <typename CommunicationType, typename ScalesType, uint32_t SyncMode>
__aicore__ inline void BlockPrologueAllGather<CommunicationType, ScalesType, SyncMode>::Run()
{
    CopyLocalData();
    SyncAll<true>();
    WriteStatusToWin();
    ReadStatus();
    ExecuteAllGather();
}

template <typename CommunicationType, typename ScalesType, uint32_t SyncMode>
__aicore__ inline void BlockPrologueAllGather<CommunicationType, ScalesType, SyncMode>::WriteStatusToWin()
{
    uint64_t curOffset = (aivId_ % sendCoreNumPerRank_ + curRankId_ * sendCoreNumPerRank_) * FLOAT_UB_ALIGN_NUM;

    auto statusUbLayout = MakeLayoutStatusUB{}(1, FLOAT_UB_ALIGN_NUM);
    auto statusUbTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubWriteStateOffset_),
        statusUbLayout);

    statusUbTensor[AscendC::Te::MakeCoord(0, 0)] = STATUS_FLAG_VAL;

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);

    GM_ADDR dstAddr = WinStatusAddr(remoteRankId_) + curOffset * sizeof(float);
    auto dstLayout = MakeLayoutStatusGM{}(1, FLOAT_UB_ALIGN_NUM);
    auto dstTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ float*)(dstAddr)),
        dstLayout);
    AscendC::Te::Copy(copyUB2GM_, dstTensor, statusUbTensor);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVT_ID);
}

template <typename CommunicationType, typename ScalesType, uint32_t SyncMode>
__aicore__ inline void BlockPrologueAllGather<CommunicationType, ScalesType, SyncMode>::ReadStatus()
{
    // each core reads its OWN slot from local status window
    uint32_t offset = aivId_ * FLOAT_UB_ALIGN_NUM;
    GM_ADDR statusAddr = WinStatusAddr(curRankId_) + offset * sizeof(float);

    auto statusGmLayout = MakeLayoutStatusGM{}(1, FLOAT_UB_ALIGN_NUM);
    auto statusGmTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ float*)(statusAddr)),
        statusGmLayout);

    auto statusUbLayout = MakeLayoutStatusUB{}(1, FLOAT_UB_ALIGN_NUM);
    auto statusUbTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubReadStateOffset_),
        statusUbLayout);

    float minTarget = STATUS_FLAG_VAL - STATUS_FLAG_THREDSHOLD;
    float maxTarget = STATUS_FLAG_VAL + STATUS_FLAG_THREDSHOLD;
    float flag = -1.0f;
    uint32_t retryCnt = 0;
    while ((flag < minTarget) || (flag > maxTarget)) {
        if (retryCnt > STATUS_POLL_MAX_RETRY) {
            break;
        }
        retryCnt++;
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVT_ID);
        AscendC::Te::Copy(copyGM2UB_, statusUbTensor, statusGmTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVT_ID);
        flag = statusUbTensor[AscendC::Te::MakeCoord(0, 0)];
    }
    // reset status to 0
    statusUbTensor[AscendC::Te::MakeCoord(0, 0)] = 0.0f;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);
    AscendC::Te::Copy(copyUB2GM_, statusGmTensor, statusUbTensor);
}

template <typename CommunicationType, typename ScalesType, uint32_t SyncMode>
__aicore__ inline void BlockPrologueAllGather<CommunicationType, ScalesType, SyncMode>::CopyLocalData()
{
    GM_ADDR localWinGm = WinDataAddr(curRankId_);
    uint64_t selfDataOffset = curRankId_ * xSize_;

    uint32_t copyXBlockNum = GetBlockNum() / GetTaskRation() - 1;
    if (aivId_ == copyXBlockNum) {
        uint64_t winScaleBase = winDataSize_ + curRankId_ * scaleSize_;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PING);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PONG);

        for (uint32_t chunkIdx = 0; chunkIdx < scaleTotalChunks_; ++chunkIdx) {
            uint16_t evtId = (chunkIdx % 2 == 0) ? EVT_ID_SCALE_PING : EVT_ID_SCALE_PONG;
            uint64_t ubOff = (chunkIdx % 2 == 0) ? ubScalePingOffset_ : ubScalePongOffset_;
            uint64_t chunkOffset = chunkIdx * scaleChunkSize_;
            uint64_t curChunkSize = (chunkOffset + scaleChunkSize_ > scaleSize_)
                ? (scaleSize_ - chunkOffset) : scaleChunkSize_; // 尾巴scale逻辑
            uint64_t curChunkPaddedN = Blaze::Gemm::CeilAlign(curChunkSize, UB_ALIGN_BYTES) / sizeof(ScalesType);

            auto scaleSrcLayout = MakeLayoutScaleGM{}(1, curChunkSize);
            auto scaleSrcTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                    (__gm__ ScalesType*)(inputXScaleAddr_ + chunkOffset)),
                scaleSrcLayout);

            auto scaleUbLayout = MakeLayoutScaleUB{}(1, curChunkPaddedN);
            auto scaleUbTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, ScalesType>(ubOff),
                scaleUbLayout);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
            AscendC::Te::Copy(copyGM2UB_, scaleUbTensor, scaleSrcTensor);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);

            auto scaleDstLayout = MakeLayoutScaleGM{}(1, curChunkSize);
            auto scaleDstTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                    (__gm__ ScalesType*)(localWinGm + winScaleBase + chunkOffset)),
                scaleDstLayout);
            AscendC::Te::Copy(copyUB2GM_, scaleDstTensor, scaleUbTensor);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PING);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PONG);
        return;
    }
    if (aicId_ > copyXBlockNum) {
        return;
    }
    // 其他v核搬运x数据到本卡winData
    uint32_t mPerCore = M_ / copyXBlockNum;
    uint32_t remainderNum = M_ % copyXBlockNum;
    uint32_t mStartId;
    if (aicId_ < remainderNum) {
        mPerCore += 1;
        mStartId = aicId_ * mPerCore;
    } else {
        mStartId = remainderNum * (mPerCore + 1) + (aicId_ - remainderNum) * mPerCore;
    }

    // chunk-based 搬运：按 xBufSize_ 大小线性迭代
    uint64_t totalElements = static_cast<uint64_t>(mPerCore) * K_;
    uint64_t startGlobalOffset = static_cast<uint64_t>(mStartId) * K_;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

    for (uint64_t done = 0; done < totalElements; done += xBufSize_) {
        uint32_t chunkIdx = static_cast<uint32_t>(done / xBufSize_);
        uint16_t evtId = (chunkIdx % 2 == 0) ? EVT_ID_PING : EVT_ID_PONG;
        uint64_t ubOff = (chunkIdx % 2 == 0) ? ubXPingOffset_ : ubXPongOffset_;
        uint32_t chunkElems = static_cast<uint32_t>(
            (done + xBufSize_ > totalElements) ? (totalElements - done) : xBufSize_);
        uint64_t globalOffset = startGlobalOffset + done;

        // 1D tensor: 搬连续 chunkElems 个元素，类似 scale 的 chunk 搬运
        auto xUbLayout = MakeLayoutUB{}(1, chunkElems);
        auto xUbTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int8_t>(ubOff),
            xUbLayout);

        auto xSrcLayout = MakeLayoutGM{}(1, chunkElems);
        auto xSrcTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                (__gm__ int8_t*)(inputXAddr_ + globalOffset)),
            xSrcLayout);

        auto xDstLayout = MakeLayoutGM{}(1, chunkElems);
        auto xDstTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                (__gm__ int8_t*)(localWinGm + selfDataOffset + globalOffset)),
            xDstLayout);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        AscendC::Te::Copy(copyGM2UB_, xUbTensor, xSrcTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        AscendC::Te::Copy(copyUB2GM_, xDstTensor, xUbTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);
}

template <typename CommunicationType, typename ScalesType, uint32_t SyncMode>
__aicore__ inline void BlockPrologueAllGather<CommunicationType, ScalesType, SyncMode>::ExecuteAllGather()
{
    GM_ADDR remoteWinGm = WinDataAddr(remoteRankId_);
    GM_ADDR localWinGm  = WinDataAddr(curRankId_);

    uint32_t modCoreIndex = aicId_ % sendCoreNumPerRank_;
    if (modCoreIndex == sendCoreNumPerRank_ - 1) {
        GM_ADDR remoteScaleAddr = remoteWinGm + winDataSize_ + remoteRankId_ * scaleSize_;
        GM_ADDR localScaleAddr  = localWinGm + winDataSize_ + remoteRankId_ * scaleSize_;
        GM_ADDR allGatherScaleAddr = allGatherScalesOutAddr_ + remoteRankId_ * scaleSize_;

        if (remoteRankId_ != curRankId_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PING);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PONG);

            for (uint32_t chunkIdx = 0; chunkIdx < scaleTotalChunks_; ++chunkIdx) {
                uint16_t evtId = (chunkIdx % 2 == 0) ? EVT_ID_SCALE_PING : EVT_ID_SCALE_PONG;
                uint64_t ubOff = (chunkIdx % 2 == 0) ? ubScalePingOffset_ : ubScalePongOffset_;
                uint64_t chunkOffset = chunkIdx * scaleChunkSize_;
                uint64_t curChunkSize = (chunkOffset + scaleChunkSize_ > scaleSize_)
                    ? (scaleSize_ - chunkOffset) : scaleChunkSize_;
                uint64_t curChunkPaddedN = Blaze::Gemm::CeilAlign(curChunkSize, UB_ALIGN_BYTES) / sizeof(ScalesType);

                auto scaleSrcLayout = MakeLayoutScaleGM{}(1, curChunkSize);
                auto scaleSrcTensor = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                        (__gm__ ScalesType*)(remoteScaleAddr + chunkOffset)),
                    scaleSrcLayout);

                auto scaleUbLayout = MakeLayoutScaleUB{}(1, curChunkPaddedN);
                auto scaleUbTensor = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, ScalesType>(ubOff),
                    scaleUbLayout);

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
                AscendC::Te::Copy(copyGM2UB_, scaleUbTensor, scaleSrcTensor);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);

                auto localScaleLayout = MakeLayoutScaleGM{}(1, curChunkSize);
                auto localScaleTensor = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                        (__gm__ ScalesType*)(localScaleAddr + chunkOffset)),
                    localScaleLayout);
                auto allGatherScaleLayout = MakeLayoutScaleGM{}(1, curChunkSize);
                auto allGatherScaleTensor = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                        (__gm__ ScalesType*)(allGatherScaleAddr + chunkOffset)),
                    allGatherScaleLayout);

                AscendC::Te::Copy(copyUB2GM_, localScaleTensor, scaleUbTensor);
                AscendC::Te::Copy(copyUB2GM_, allGatherScaleTensor, scaleUbTensor);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PING);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PONG);
        }
    }

    if constexpr (SyncMode == NO_TILE_K) {
        uint32_t mPerCore = M_ / sendCoreNumPerRank_;
        uint32_t remainderNum = M_ % sendCoreNumPerRank_;
        uint32_t mStartId = mPerCore * (aicId_ % sendCoreNumPerRank_);
        if (aicId_ % sendCoreNumPerRank_ >= sendCoreNumPerRank_ - remainderNum) {
            mPerCore++;
            mStartId += aicId_ % sendCoreNumPerRank_ - (sendCoreNumPerRank_ - remainderNum);
        }

        // chunk-based: 按 xBufSize_ 线性迭代
        uint64_t totalElements = static_cast<uint64_t>(mPerCore) * K_;
        uint64_t startGlobalOffset = static_cast<uint64_t>(mStartId) * K_;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

        for (uint64_t done = 0; done < totalElements; done += xBufSize_) {
            uint32_t chunkIdx = static_cast<uint32_t>(done / xBufSize_);
            uint32_t chunkElems = static_cast<uint32_t>(
                (done + xBufSize_ > totalElements) ? (totalElements - done) : xBufSize_);
            uint64_t globalOffset = startGlobalOffset + done;
            ReadDataBlock(globalOffset, chunkElems, chunkIdx % 2);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

        SyncAll<true>();
        CrossCoreSetFlag<CROSS_CORE_INNER_CUBE_VEC_SYNC, PIPE_MTE3>(CROSS_CORE_FLAG_NUM_NINE);
    } else {
        uint32_t kLoop = tileK_ / kDim_;
        if (kBlockIdx_ < tileK_ % kDim_) {
            kLoop++;
        }
        uint32_t curXOffset = coreInnerMIndex_ * K_ + kStartIndex_;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

        for (uint64_t curKBlock = 0; curKBlock < kLoop; ++curKBlock) {
            uint64_t innerCurXOffset = curXOffset + curKBlock * X_PER_BLOCK_NUM * 2;
            ReadDataBlock(innerCurXOffset, mMteCoreM_, curKBlock % 2);
            PipeBarrier<PIPE_MTE3>();
            SyncAll<true>();
            CrossCoreSetFlag<CROSS_CORE_INNER_CUBE_VEC_SYNC, PIPE_MTE3>(CROSS_CORE_FLAG_NUM_NINE);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

        if (kLoop < ((tileK_ + kDim_ - 1) / kDim_)) {
            SyncAll<true>();
            CrossCoreSetFlag<CROSS_CORE_INNER_CUBE_VEC_SYNC, PIPE_MTE3>(CROSS_CORE_FLAG_NUM_NINE);
        }
    }
}

template <typename CommunicationType, typename ScalesType, uint32_t SyncMode>
__aicore__ inline void BlockPrologueAllGather<CommunicationType, ScalesType, SyncMode>::ReadDataBlock(
    uint64_t curXOffset, uint32_t mCnt, uint32_t bufIdx)
{
    if (remoteRankId_ == curRankId_) {
        return;
    }
    if (kBlockIdx_ >= kDim_ || mBlockIdx_ >= mDim_) {
        return;
    }

    uint16_t evtId = (bufIdx % 2 == 0) ? EVT_ID_PING : EVT_ID_PONG;
    uint64_t ubOff = (bufIdx % 2 == 0) ? ubXPingOffset_ : ubXPongOffset_;

    GM_ADDR remoteWinGm = WinDataAddr(remoteRankId_);
    GM_ADDR localWinGm  = WinDataAddr(curRankId_);

    GM_ADDR remoteXAddr = remoteWinGm + remoteRankId_ * xSize_;
    GM_ADDR localXAddr  = localWinGm  + remoteRankId_ * xSize_;
    GM_ADDR allGatherXAddr = allGatherDataOutAddr_ + remoteRankId_ * xSize_;

    if constexpr (SyncMode == NO_TILE_K) {
        // chunk-based 1D: curXOffset 为 flat element offset, mCnt 为 chunk 元素数
        uint64_t globalOffset = curXOffset;

        auto ubLayout = MakeLayoutUB{}(1, mCnt);
        auto ubTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int8_t>(ubOff),
            ubLayout);

        auto srcLayout = MakeLayoutGM{}(1, mCnt);
        auto srcTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                (__gm__ int8_t*)(remoteXAddr + globalOffset)),
            srcLayout);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        AscendC::Te::Copy(copyGM2UB_, ubTensor, srcTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);

        auto localLayout = MakeLayoutGM{}(1, mCnt);
        auto localTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                (__gm__ int8_t*)(localXAddr + globalOffset)),
            localLayout);

        auto agLayout = MakeLayoutGM{}(1, mCnt);
        auto agTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                (__gm__ int8_t*)(allGatherXAddr + globalOffset)),
            agLayout);

        AscendC::Te::Copy(copyUB2GM_, localTensor, ubTensor);
        AscendC::Te::Copy(copyUB2GM_, agTensor, ubTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
    } else {
        // split-K: 每次搬 mCnt 行 × X_PER_BLOCK_NUM(512) 列。
        // 单 slot 容量为 xBufSize_，最多容纳 rowsPerSlot 行；mCnt 可能远大于此（大 M），
        // 故对 M 维再分块，逐子块复用同一 ping/pong slot（evtId 串行同步），避免 UB 越界。
        uint64_t rowStartBase = curXOffset / K_;
        uint64_t colStart = curXOffset % K_;
        uint32_t rowsPerSlot = static_cast<uint32_t>(xBufSize_ / X_PER_BLOCK_NUM);
        if (rowsPerSlot == 0) {
            rowsPerSlot = 1;
        }

        auto srcTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ int8_t*)(remoteXAddr)),
            MakeLayoutGM{}(M_, K_));
        auto localTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ int8_t*)(localXAddr)),
            MakeLayoutGM{}(M_, K_));
        auto agTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ int8_t*)(allGatherXAddr)),
            MakeLayoutGM{}(M_, K_));

        for (uint32_t rowDone = 0; rowDone < mCnt; rowDone += rowsPerSlot) {
            uint32_t curRows = (rowDone + rowsPerSlot > mCnt) ? (mCnt - rowDone) : rowsPerSlot;
            uint64_t rowStart = rowStartBase + rowDone;

            auto srcSlice = srcTensor.Slice(
                AscendC::Te::MakeCoord(rowStart, colStart),
                AscendC::Te::MakeShape(static_cast<int64_t>(curRows), static_cast<int64_t>(X_PER_BLOCK_NUM)));
            auto ubTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int8_t>(ubOff),
                MakeLayoutUB{}(curRows, static_cast<int64_t>(X_PER_BLOCK_NUM)));

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
            AscendC::Te::Copy(copyGM2UB_, ubTensor, srcSlice);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);

            auto localSlice = localTensor.Slice(
                AscendC::Te::MakeCoord(rowStart, colStart),
                AscendC::Te::MakeShape(static_cast<int64_t>(curRows), static_cast<int64_t>(X_PER_BLOCK_NUM)));
            auto agSlice = agTensor.Slice(
                AscendC::Te::MakeCoord(rowStart, colStart),
                AscendC::Te::MakeShape(static_cast<int64_t>(curRows), static_cast<int64_t>(X_PER_BLOCK_NUM)));

            AscendC::Te::Copy(copyUB2GM_, localSlice, ubTensor);
            AscendC::Te::Copy(copyUB2GM_, agSlice, ubTensor);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        }
    }
}

} // namespace Block
} // namespace Gemm
} // namespace Blaze
