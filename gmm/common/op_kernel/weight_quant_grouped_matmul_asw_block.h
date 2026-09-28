/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_grouped_matmul_asw_block.h
 * \brief
 */
#pragma once

#include "weight_quant_tool_asw.h"

using AscendC::BLOCK_CUBE;
using namespace WeightQuantTools;

namespace WeightQuantGroupedMatmul::Arch35 {
struct ASWTilingParam {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t aScaleK;
    uint64_t bScaleK;
    uint64_t baseM;
    uint64_t splitBaseN;
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t totalCnt;
    uint64_t mCoreNum;
    uint64_t mTailCoreNum;
    uint64_t mBaseTail;
    uint64_t nBaseTail;
    uint64_t mIndex;
    uint64_t nIndex;
    uint64_t mSplitAddrOffset;
    uint64_t nSplitAddrOffset;
    uint64_t xGroupAddrOffset;
    uint64_t wGroupAddrOffset;
    uint64_t yGroupAddrOffset;
    uint64_t yScaleGroupAddrOffset;
    uint64_t pertokenScaleGroupAddrOffset;
    uint64_t antiquantScaleGroupAddrOffset;
    uint64_t biasGroupAddrOffset;
    uint64_t mTailTile;
    uint64_t nTailTile;
    uint64_t mainRow;
    uint64_t round;
    uint64_t index;
};

// 每个基本块在当前group的GlobalTensor上的偏移
struct ASWOffsetParam {
    uint64_t offsetM;
    uint64_t offsetN;
    uint64_t offsetX;
    uint64_t offsetWeight;
    uint64_t offsetY;
    uint64_t offsetAntiquantScale;
    uint64_t offsetBias;
    uint64_t offsetPerTokenScale;
    uint64_t offsetYScale;
};

struct DefaultGroupAddrUpdater {
    __aicore__ inline void operator()(ASWTilingParam &params)
    {
        params.yGroupAddrOffset += params.m * params.n;
    }
};

struct SwiGLUGroupAddrUpdater {
    __aicore__ inline void operator()(ASWTilingParam &params)
    {
        params.yGroupAddrOffset += params.m * (params.n / SWIGLU_N_HALF_FACTOR);
        uint64_t outputScaleN =
            Ops::Base::CeilDiv(params.n / SWIGLU_N_HALF_FACTOR, MX_SCALE_DIVISOR_SIZE) * MX_SCALE_LAST_DIM_2;
        params.yScaleGroupAddrOffset += params.m * outputScaleN;
    }
};

struct DefaultBlockOffsetUpdater {
    __aicore__ inline void operator()(ASWTilingParam &params, ASWOffsetParam &offset, uint64_t mOffset,
                                      uint64_t nOffset)
    {
        offset.offsetY = mOffset * params.n + nOffset;
    }
};

struct SwiGLUBlockOffsetUpdater {
    __aicore__ inline void operator()(ASWTilingParam &params, ASWOffsetParam &offset, uint64_t mOffset,
                                      uint64_t nOffset)
    {
        offset.offsetY = mOffset * (params.n / SWIGLU_N_HALF_FACTOR) + nOffset;
        uint64_t outputScaleN =
            Ops::Base::CeilDiv(params.n / SWIGLU_N_HALF_FACTOR, MX_SCALE_DIVISOR_SIZE) * MX_SCALE_LAST_DIM_2;
        uint64_t nScaleOffset = (nOffset / MX_SCALE_DIVISOR_SIZE) * MX_SCALE_LAST_DIM_2;
        offset.offsetYScale = mOffset * outputScaleN + nScaleOffset;
    }
};

class WeightQuantASWBlock {
public:
    __aicore__ inline WeightQuantASWBlock()
    {
    }
    __aicore__ inline void Init(const TCubeTiling *__restrict &tilingData, uint32_t blockIdx);
    // 每一个group需要更新mm的group偏移和MNK
    template <bool wTrans, typename xType, typename wType, typename antiquantScaleType, typename yType,
              CubeFormat wFormat = CubeFormat::ND, typename GroupAddrUpdater = DefaultGroupAddrUpdater>
    __aicore__ inline void UpdateGroupOffset(uint64_t m, uint64_t n, uint64_t k, uint32_t groupIdx);

    template <bool isSwiGLU = false>
    __aicore__ inline void UpdateGroupParams(bool isLastGroup); // 每一个group需要更新mm的参数

    template <bool xTrans = false, bool wTrans = false, bool isSwiGLU = false>
    __aicore__ inline void CalcTailTile();

    __aicore__ inline void UpdateTailTile();
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx, bool isLastGroupRound);

    template <bool xTrans, bool wTrans, CubeFormat wFormat = CubeFormat::ND, bool isSwiGLU = false>
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx, bool isLastGroupRound = true);

    template <bool wTrans, typename xType, typename wType, typename antiquantScaleType = AscendC::fp8_e8m0_t,
              CubeFormat wFormat = CubeFormat::ND, typename BlockOffsetUpdater = DefaultBlockOffsetUpdater>
    __aicore__ inline void CalcGMOffset();

    __aicore__ inline uint32_t GetEndBlockIdx() const;

public:
    ASWTilingParam params_;
    ASWOffsetParam offset_;
    const TCubeTiling *__restrict tilingData_;

private:
    constexpr static uint64_t WINDOW_LEN = 8UL;
    constexpr static uint64_t GMMSQ_MIN_SPLIT_BASE_N = 64UL;
    uint32_t blockIdx_;
    uint32_t startBlockIdx_;
    uint32_t endBlockIdx_;
    bool isLastGroupCrossRound_; // 最后一组是否存在工作的核轮数不一样（跨越0核）
};

__aicore__ inline void WeightQuantASWBlock::Init(const TCubeTiling *__restrict &tilingData, uint32_t blockIdx)
{
    blockIdx_ = blockIdx;
    tilingData_ = tilingData;

    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
    params_.mTailTile = 1; // 1 尾块不切分
    params_.nTailTile = 1; // 1 尾块不切分

    startBlockIdx_ = 0;                          // 每个group开始的核索引
    endBlockIdx_ = tilingData_->usedCoreNum - 1; // 上个group结束的核索引

    params_.m = 0;
    params_.n = 0;
    params_.k = 0;
    params_.aScaleK = 1;
    params_.bScaleK = 1;

    // 不同group的偏移量
    params_.xGroupAddrOffset = 0;
    params_.wGroupAddrOffset = 0;
    params_.yGroupAddrOffset = 0;
    params_.pertokenScaleGroupAddrOffset = 0;
    params_.antiquantScaleGroupAddrOffset = 0;
    params_.biasGroupAddrOffset = 0;
    params_.yScaleGroupAddrOffset = 0;
}

template <bool wTrans, typename xType, typename wType, typename antiquantScaleType, typename yType, CubeFormat wFormat,
          typename GroupAddrUpdater>
__aicore__ inline void WeightQuantASWBlock::UpdateGroupOffset(uint64_t m, uint64_t n, uint64_t k, uint32_t groupIdx)
{
    if (groupIdx > 0) { // groupIdx==0时，起始点均为0，无需计算，减少scalar
        params_.xGroupAddrOffset += params_.m * params_.k;

        uint64_t wGroupOffset = 0;
        if constexpr (wFormat == CubeFormat::NZ) {
            if constexpr (wTrans) {
                wGroupOffset = Ops::Base::CeilDiv(params_.k, C0<wType>::value) *
                               Ops::Base::CeilDiv(params_.n, static_cast<uint64_t>(BLOCK_CUBE)) * BLOCK_CUBE *
                               C0<wType>::value;
            } else {
                wGroupOffset = Ops::Base::CeilDiv(params_.n, C0<wType>::value) *
                               Ops::Base::CeilDiv(params_.k, static_cast<uint64_t>(BLOCK_CUBE)) * BLOCK_CUBE *
                               C0<wType>::value;
            }
        } else {
            wGroupOffset = params_.n * params_.k;
        }

        if constexpr (AscendC::IsSupportB4<wType>()) {
            wGroupOffset = wGroupOffset >> 1;
        }

        params_.wGroupAddrOffset += wGroupOffset;

        params_.antiquantScaleGroupAddrOffset += params_.n * params_.bScaleK;
        if constexpr (WeightQuantTools::IsMxA8W4<xType, wType, antiquantScaleType>() ||
                      WeightQuantTools::IsMxA8HiF4<xType, wType, antiquantScaleType>()) {
            params_.pertokenScaleGroupAddrOffset += params_.m * params_.aScaleK;
        }
        params_.biasGroupAddrOffset += params_.n;

        GroupAddrUpdater updater;
        updater(params_);
    }

    params_.m = m;
    params_.n = n;
    params_.k = k;

    if constexpr (WeightQuantTools::IsMxA8W4<xType, wType, antiquantScaleType>() ||
                  WeightQuantTools::IsMxA8HiF4<xType, wType, antiquantScaleType>()) {
        // mx a scale (m, k/64, 2)
        params_.aScaleK = Ops::Base::CeilDiv(params_.k, MX_SCALE_DIVISOR_SIZE) * MX_SCALE_LAST_DIM_2;
    }

    if constexpr (IsSameType<antiquantScaleType, fp8_e8m0_t>::value) {
        // mxfp4 b scale (n, k/64, 2)
        params_.bScaleK = Ops::Base::CeilDiv(params_.k, MX_SCALE_DIVISOR_SIZE) * MX_SCALE_LAST_DIM_2;
    } else if (IsSameType<antiquantScaleType, hif4_scale>::value) {
        // mxhif4 b scale (n, k/64, 1)
        params_.bScaleK = Ops::Base::CeilDiv(params_.k, MX_SCALE_DIVISOR_SIZE);
    }
}

template <bool isSwiGLU>
__aicore__ inline void WeightQuantASWBlock::UpdateGroupParams(bool isLastGroup)
{
    uint64_t mBlkNum = Ops::Base::CeilDiv(params_.m, static_cast<uint64_t>(tilingData_->baseM));
    params_.baseM = Ops::Base::CeilDiv(params_.m, mBlkNum);
    uint64_t actualN;
    if constexpr (isSwiGLU) {
        // SwiGLU将N分为gate和val两部分，gmmsq按照N/2和baseN/2分核
        params_.splitBaseN = tilingData_->baseN / SWIGLU_N_HALF_FACTOR;
        actualN = params_.n / SWIGLU_N_HALF_FACTOR;
    } else {
        params_.splitBaseN = tilingData_->baseN;
        actualN = params_.n;
    }

    params_.mCnt = Ops::Base::CeilDiv(params_.m, params_.baseM);
    params_.nCnt = Ops::Base::CeilDiv(actualN, params_.splitBaseN);
    params_.totalCnt = params_.mCnt * params_.nCnt;

    // 计算当前group的mm最后一轮计算空闲的核的索引
    // 新group开始的空闲的核索引
    startBlockIdx_ = endBlockIdx_ == tilingData_->usedCoreNum - 1 ? 0 : (endBlockIdx_ + 1);
    // 当前group结束的空闲的核索引
    endBlockIdx_ = (params_.totalCnt + startBlockIdx_ - 1) % tilingData_->usedCoreNum;

    // 2：最后一组占用最后一轮核数超一半且未分满核时重新规划baseN，没超一半按原来的逻辑尾块重切分
    if (isLastGroup && (endBlockIdx_ > tilingData_->usedCoreNum / 2) && (endBlockIdx_ < tilingData_->usedCoreNum - 1)) {
        uint64_t availableTotalCnt = params_.totalCnt + (tilingData_->usedCoreNum - endBlockIdx_ - 1);
        params_.splitBaseN = Ops::Base::CeilDiv(actualN, availableTotalCnt / params_.mCnt);
        if constexpr (isSwiGLU) {
            params_.splitBaseN = Ops::Base::CeilAlign(params_.splitBaseN, GMMSQ_MIN_SPLIT_BASE_N);
        } else {
            params_.splitBaseN = Ops::Base::CeilAlign(params_.splitBaseN, static_cast<uint64_t>(BLOCK_CUBE));
        }
        params_.nCnt = Ops::Base::CeilDiv(actualN, params_.splitBaseN);
        params_.totalCnt = params_.mCnt * params_.nCnt;
        endBlockIdx_ = (params_.totalCnt + startBlockIdx_ - 1) % tilingData_->usedCoreNum;
    }

    isLastGroupCrossRound_ = isLastGroup && !(startBlockIdx_ < endBlockIdx_ &&
                                              params_.totalCnt < static_cast<uint64_t>(tilingData_->usedCoreNum));

    params_.mBaseTail = params_.m - (params_.mCnt - 1) * params_.baseM;
    params_.nBaseTail = actualN - (params_.nCnt - 1) * params_.splitBaseN;
    params_.mCoreNum = WeightQuantTools::Min(WINDOW_LEN, params_.mCnt);
    // 计算round数还是按照实际，使用核数按照startBlockIdx开始
    params_.round = Ops::Base::CeilDiv(params_.totalCnt, static_cast<uint64_t>(tilingData_->usedCoreNum));
    params_.mainRow = params_.mCnt / params_.mCoreNum - 1;
    params_.mTailCoreNum = params_.mCnt - params_.mCoreNum * params_.mainRow;

    // 如果当前group不是最后一个group，则空闲核不参与最后一轮计算，需要留给下一个group使用
    if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
        params_.round -= 1;
    } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
        params_.round -= 1;
    }
}

template <bool xTrans, bool wTrans, bool isSwiGLU>
__aicore__ inline void WeightQuantASWBlock::CalcTailTile()
{
    // 最后一组核索引跨越0，切分最后一轮0之后的基本块，否则切分所有基本块
    // 可切分数 remainTile = 可用核数 / 待切分的基本块数
    uint64_t remainTile = isLastGroupCrossRound_ ? tilingData_->usedCoreNum / (endBlockIdx_ + 1) :
                                                   (tilingData_->usedCoreNum - startBlockIdx_) / params_.totalCnt;

    if (remainTile <= 1) {
        return;
    }

    // 初始化最小 tile 大小
    uint64_t mMin = BLOCK_CUBE;
    uint64_t nMin = BLOCK_CUBE;

    if constexpr (isSwiGLU) {
        nMin = GMMSQ_MIN_SPLIT_BASE_N;
    }

    // 计算 mTile 和 nTile，尽可能让m,n方向切分数一致
    uint64_t mTile = WeightQuantTools::Min(Ops::Base::CeilDiv(params_.mBaseTail, mMin), remainTile);
    uint64_t nTile = WeightQuantTools::Min(Ops::Base::CeilDiv(params_.nBaseTail, nMin), remainTile);
    while (mTile * nTile > remainTile) {
        if (mTile >= nTile) {
            mTile -= 1;
        } else {
            nTile -= 1;
        }
    }
    params_.mTailTile = mTile;
    params_.nTailTile = nTile;
}

__aicore__ inline void WeightQuantASWBlock::UpdateTailTile()
{
    // 尾块切分后，更新结束的核索引和round数
    // 最后一组核索引跨越0，结束核索引从0核开始更新，否则从原来的起始核索引重新计算
    uint64_t newEndBlockIdx = isLastGroupCrossRound_ ?
                                  params_.mTailTile * params_.nTailTile * (endBlockIdx_ + 1) - 1 :
                                  startBlockIdx_ + params_.mTailTile * params_.nTailTile * params_.totalCnt - 1;
    if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
        params_.round += 1;
    }
    endBlockIdx_ = newEndBlockIdx;
}

__aicore__ inline void WeightQuantASWBlock::UpdateBasicIndex(uint64_t roundIdx, bool isLastGroupRound)
{
    if (!isLastGroupRound || (isLastGroupRound && isLastGroupCrossRound_)) {
        uint64_t newBlockIdx = isLastGroupRound ? (blockIdx_ / (params_.mTailTile * params_.nTailTile)) : blockIdx_;
        params_.index = newBlockIdx + roundIdx * tilingData_->usedCoreNum;

        // GMM当前group的startBlockIdx_不一定从0开始，要进行计算过的base块数修正
        if (blockIdx_ < startBlockIdx_) {
            params_.index += tilingData_->usedCoreNum - startBlockIdx_; // 加上最开始从startBlockIdx_开始的base数
        } else {
            params_.index -= startBlockIdx_; // 减去roundIdx * CoreNum的未包含的最开始0-(startBlockIdx_-1)的base数
        }
    } else {
        // 最后一组核索引没有跨越0核，blockIdx_一定大于等于startBlockIdx_，且每个block只算一轮
        params_.index = (blockIdx_ - startBlockIdx_) / (params_.mTailTile * params_.nTailTile);
    }

    uint64_t rowIdx = params_.index / params_.nCnt / params_.mCoreNum;
    if (rowIdx < params_.mainRow) {
        params_.mIndex = rowIdx * params_.mCoreNum + params_.index % params_.mCoreNum;
        params_.nIndex = (params_.index / params_.mCoreNum) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = params_.index - params_.mainRow * params_.mCoreNum * params_.nCnt;
        params_.mIndex = params_.mainRow * params_.mCoreNum + tailIndex % params_.mTailCoreNum;
        params_.nIndex = (tailIndex / params_.mTailCoreNum) % params_.nCnt;
    }

    if (rowIdx & 1) {
        params_.nIndex = params_.nCnt - 1 - params_.nIndex;
    }
}

// 返回给kernel的开始空闲核索引，帮助最后一个group的尾块细分
__aicore__ inline uint32_t WeightQuantASWBlock::GetEndBlockIdx() const
{
    return endBlockIdx_;
}

template <bool xTrans, bool wTrans, CubeFormat wFormat, bool isSwiGLU>
__aicore__ inline void WeightQuantASWBlock::UpdateBlockParams(uint64_t roundIdx, bool isLastGroupRound)
{
    params_.singleCoreM = params_.mIndex != (params_.mCnt - 1) ? params_.baseM : params_.mBaseTail;
    params_.singleCoreN = params_.nIndex != (params_.nCnt - 1) ? params_.splitBaseN : params_.nBaseTail;

    if (!isLastGroupRound || (params_.mTailTile == 1 && params_.nTailTile == 1)) {
        if constexpr (isSwiGLU) {
            params_.singleCoreN *= SWIGLU_N_HALF_FACTOR; // gmmsq单核总计算量为2splitBaseN
        }
        return;
    }

    if (roundIdx == params_.round - 1) {
        uint64_t singleCoreMSplit = (params_.singleCoreM + params_.mTailTile - 1) / params_.mTailTile;
        uint64_t singleCoreNSplit = (params_.singleCoreN + params_.nTailTile - 1) / params_.nTailTile;
        if constexpr (!wTrans) {
            singleCoreNSplit = Ops::Base::CeilAlign(singleCoreNSplit, WeightQuantTools::INNER_AXIS_MIN_SPLIT_VAL);
        } else {
            if constexpr (wFormat == CubeFormat::NZ) {
                singleCoreNSplit = Ops::Base::CeilAlign(singleCoreNSplit, static_cast<uint64_t>(BLOCK_CUBE));
            }
        }
        if constexpr (isSwiGLU) {
            singleCoreNSplit = Ops::Base::CeilAlign(singleCoreNSplit, GMMSQ_MIN_SPLIT_BASE_N);
        }

        uint64_t totalTailTile = params_.mTailTile * params_.nTailTile;
        uint64_t mSplitIdx = (blockIdx_ % totalTailTile) % params_.mTailTile;
        uint64_t nSplitIdx = (blockIdx_ % totalTailTile) / params_.mTailTile;
        params_.mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        params_.nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (params_.mSplitAddrOffset >= params_.singleCoreM || params_.nSplitAddrOffset >= params_.singleCoreN) {
            params_.singleCoreM = 0;
            params_.singleCoreN = 0;
            return;
        }
        if (params_.mSplitAddrOffset + singleCoreMSplit > params_.singleCoreM) {
            params_.singleCoreM = params_.singleCoreM - singleCoreMSplit * mSplitIdx;
        } else {
            params_.singleCoreM = singleCoreMSplit;
        }
        if (params_.nSplitAddrOffset + singleCoreNSplit > params_.singleCoreN) {
            params_.singleCoreN = params_.singleCoreN - singleCoreNSplit * nSplitIdx;
        } else {
            params_.singleCoreN = singleCoreNSplit;
        }
    }

    if constexpr (isSwiGLU) {
        params_.singleCoreN *= SWIGLU_N_HALF_FACTOR; // gmmsq单核总计算量为2splitBaseN
    }
}

template <bool wTrans, typename xType, typename wType, typename antiquantScaleType, CubeFormat wFormat,
          typename BlockOffsetUpdater>
__aicore__ inline void WeightQuantASWBlock::CalcGMOffset()
{
    uint64_t mOffset = params_.mIndex * params_.baseM + params_.mSplitAddrOffset;
    uint64_t nOffset = params_.nIndex * params_.splitBaseN + params_.nSplitAddrOffset;
    offset_.offsetM = mOffset;
    offset_.offsetN = nOffset;

    offset_.offsetX = mOffset * params_.k;

    if constexpr (wFormat == CubeFormat::NZ) {
        if constexpr (wTrans) {
            offset_.offsetWeight = nOffset * C0<wType>::value;
        } else {
            offset_.offsetWeight = nOffset * Ops::Base::CeilDiv(params_.k, static_cast<uint64_t>(BLOCK_CUBE)) *
                                   static_cast<uint64_t>(BLOCK_CUBE);
        }
    } else {
        if constexpr (wTrans) {
            offset_.offsetWeight = nOffset * params_.k;
        } else {
            offset_.offsetWeight = nOffset;
        }
    }

    offset_.offsetAntiquantScale = nOffset * params_.bScaleK;
    if constexpr (WeightQuantTools::IsMxA8W4<xType, wType, antiquantScaleType>() ||
                  WeightQuantTools::IsMxA8HiF4<xType, wType, antiquantScaleType>()) {
        offset_.offsetPerTokenScale = mOffset * params_.aScaleK;
    }

    offset_.offsetBias = nOffset;

    BlockOffsetUpdater updater;
    updater(params_, offset_, mOffset, nOffset);
}

} // namespace WeightQuantGroupedMatmul::Arch35