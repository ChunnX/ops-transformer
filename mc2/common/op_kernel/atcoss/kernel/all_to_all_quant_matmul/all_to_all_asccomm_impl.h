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
 * \file all_to_all_asccomm_impl.h
 * \brief CCU 四段式通信接口 — Init -> Commit -> Wait -> Finalize
 *
 * 从 arch35 的 MxQuantAlltoAllComm + HcclCommunicator 提炼，
 * 将 Init/PrepareAll/Process/End 改造为四段式 API。
 * 单 HCCL 引擎，DataA/ScaleA 通过 SetContext 切换。
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "atcoss/tiling/comm_tiling_data.h"

namespace Atcoss {
namespace AivComm {
namespace Ccu {

__aicore__ inline uint64_t CeilDivU64(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

struct CcuCommContext {
    GM_ADDR sendBuffer;
    GM_ADDR recvBuffer;
    uint64_t sendCount;
    uint64_t strideCount;
    uint64_t hcclDataType;
    uint8_t repeat{1};
};

template <typename TilingDataType>
class CcuAllToAllComm {
public:
    __aicore__ inline CcuAllToAllComm() : tiling_(nullptr) {}
    __aicore__ inline ~CcuAllToAllComm() {}

    __aicore__ inline uint64_t Init(
        TilingDataType *tiling,
        GM_ADDR sendBuffer, GM_ADDR recvBuffer,
        const Atcoss::Utils::CommTilingData &commTilingData,
        uint64_t hcclDataType,
        uint64_t elementSize,
        uint64_t winOffset = 0);

    __aicore__ inline void Commit();
    __aicore__ inline void Wait(bool waitLast = false);
    __aicore__ inline void Finalize();

    __aicore__ inline void SetContext(
        GM_ADDR sendBuffer, GM_ADDR recvBuffer,
        const Atcoss::Utils::CommTilingData &commTilingData,
        uint64_t hcclDataType,
        uint64_t elementSize,
        uint64_t winOffset = 0);

    __aicore__ inline uint64_t GetTotalTiles() const { return totalTiles_; }
    __aicore__ inline bool IsNotifyCore() const { return notifyFlag_; }

private:
    static constexpr AscendC::HcclServerType kServerType =
        AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU;
    AscendC::Hccl<kServerType> hccl_;

    TilingDataType *tiling_;
    Atcoss::Utils::CommTilingData commTilingData_;
    CcuCommContext ctx_;
    uint64_t winOffset_{0};
    uint64_t elementSize_{0};

    uint64_t currentTileIdx_{0};
    uint64_t totalTiles_{0};
    uint64_t remainingChunkSize_{0};

    AscendC::HcclHandle curHandle_{0};
    bool hasPendingHandle_{false};

    bool notifyFlag_{false};

    uint64_t totalSplitAxisSize_{0};
    uint64_t nonSplitAxisBytes_{0};
    uint64_t tileByteOffset_{0};

    __aicore__ inline void SetupContext_(
        GM_ADDR sendBuffer, GM_ADDR recvBuffer,
        const Atcoss::Utils::CommTilingData &commTilingData,
        uint64_t hcclDataType,
        uint64_t elementSize,
        uint64_t winOffset);
};

template <typename TilingDataType>
__aicore__ inline void
CcuAllToAllComm<TilingDataType>::SetupContext_(
    GM_ADDR sendBuffer, GM_ADDR recvBuffer,
    const Atcoss::Utils::CommTilingData &commTilingData,
    uint64_t hcclDataType,
    uint64_t elementSize,
    uint64_t winOffset)
{
    ctx_.sendBuffer = sendBuffer;
    ctx_.recvBuffer = recvBuffer;
    ctx_.hcclDataType = hcclDataType;
    ctx_.repeat = 1;
    winOffset_ = winOffset;
    elementSize_ = elementSize;

    commTilingData_ = commTilingData;

    totalTiles_ = commTilingData.splitAxisTileCnt + commTilingData.splitAxisTailCnt;
    totalSplitAxisSize_ = commTilingData.splitAxisTileSize * commTilingData.splitAxisTileCnt +
                          commTilingData.splitAxisTailSize * commTilingData.splitAxisTailCnt;
    nonSplitAxisBytes_ = commTilingData.nonSplitAxisSize * elementSize_;
    ctx_.strideCount = totalSplitAxisSize_ * commTilingData.nonSplitAxisSize;

    currentTileIdx_ = 0;
    tileByteOffset_ = 0;
    remainingChunkSize_ = totalSplitAxisSize_;
    hasPendingHandle_ = false;
}

template <typename TilingDataType>
__aicore__ inline uint64_t
CcuAllToAllComm<TilingDataType>::Init(
    TilingDataType *tiling,
    GM_ADDR sendBuffer, GM_ADDR recvBuffer,
    const Atcoss::Utils::CommTilingData &commTilingData,
    uint64_t hcclDataType,
    uint64_t elementSize,
    uint64_t winOffset)
{
    tiling_ = tiling;

    notifyFlag_ = false;
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            notifyFlag_ = true;
        }
    }

    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tiling_->mc2InitTiling));
    hccl_.SetCcTilingV2(offsetof(TilingDataType, mc2CcTiling));

    SetupContext_(sendBuffer, recvBuffer, commTilingData, hcclDataType, elementSize, winOffset);

    return totalTiles_;
}

template <typename TilingDataType>
__aicore__ inline void
CcuAllToAllComm<TilingDataType>::SetContext(
    GM_ADDR sendBuffer, GM_ADDR recvBuffer,
    const Atcoss::Utils::CommTilingData &commTilingData,
    uint64_t hcclDataType,
    uint64_t elementSize,
    uint64_t winOffset)
{
    SetupContext_(sendBuffer, recvBuffer, commTilingData, hcclDataType, elementSize, winOffset);
}

template <typename TilingDataType>
__aicore__ inline void
CcuAllToAllComm<TilingDataType>::Commit()
{
    if (!notifyFlag_ || remainingChunkSize_ <= 0 || hasPendingHandle_) {
        return;
    }

    uint64_t currentTileSize;
    if (currentTileIdx_ < commTilingData_.splitAxisTileCnt) {
        currentTileSize = commTilingData_.splitAxisTileSize;
    } else {
        currentTileSize = commTilingData_.splitAxisTailSize;
    }
    if (currentTileSize > remainingChunkSize_) {
        currentTileSize = remainingChunkSize_;
    }

    uint64_t currentTileByteSize = currentTileSize * nonSplitAxisBytes_;

    GM_ADDR sendAddr = ctx_.sendBuffer + tileByteOffset_;
    GM_ADDR recvAddr = ctx_.recvBuffer + winOffset_ + tileByteOffset_;

    ctx_.sendCount = currentTileSize * commTilingData_.nonSplitAxisSize;

    curHandle_ = hccl_.template AlltoAll<false>(
        sendAddr, recvAddr,
        ctx_.sendCount,
        static_cast<AscendC::HcclDataType>(static_cast<uint8_t>(ctx_.hcclDataType)),
        ctx_.strideCount,
        ctx_.repeat);
    hccl_.Commit(curHandle_);
    hasPendingHandle_ = true;

    currentTileIdx_++;
    tileByteOffset_ += currentTileByteSize;
    remainingChunkSize_ -= currentTileSize;
}

template <typename TilingDataType>
__aicore__ inline void
CcuAllToAllComm<TilingDataType>::Wait(bool waitLast)
{
    if (!notifyFlag_ || !hasPendingHandle_) {
        return;
    }
    hccl_.Wait(curHandle_);
    hasPendingHandle_ = false;
}

template <typename TilingDataType>
__aicore__ inline void
CcuAllToAllComm<TilingDataType>::Finalize()
{
    if (!notifyFlag_) {
        return;
    }
    if (hasPendingHandle_) {
        hccl_.Wait(curHandle_);
        hasPendingHandle_ = false;
    }
    hccl_.Finalize();
}

} // namespace Ccu
} // namespace AivComm
} // namespace Atcoss
