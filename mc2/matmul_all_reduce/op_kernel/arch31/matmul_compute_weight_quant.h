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
 * \file matmul_compute_weight_quant.h
 * \brief
 */

#ifndef MC2_MATMUL_COMPUTE_WEIGHT_QUANT_H
#define MC2_MATMUL_COMPUTE_WEIGHT_QUANT_H

#include "matmul_compute.h"
namespace AscendC {
template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache = false, bool WeightQuant = false,
    AntiQuantType antiQuantType = AntiQuantType::NONE, bool hasAntiQuantOffset = false>
class MatmulComputeWeightQuant
    : public MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType, hasAntiQuantOffset>
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline MatmulComputeWeightQuant()
    {}
    __aicore__ inline void Init(
        AscendC::tiling::TCubeTiling& tiling, Mc2Tiling::RCSTiling& cfg, Mc2Tiling::Mc2L2cacheTilePara& tileL2cacheTiling,
        const LocalTensor<uint8_t>& mmFmtUb12);
    __aicore__ inline void InitGlobalBTensor(
        GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR antiquantScale, GM_ADDR antiquantOffset);
    __aicore__ inline void Compute(uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void ComputeWithL2Cache(uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void ComputeWithL2CacheOdd(int32_t mTileIdx, uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void ComputeWithL2CacheEven(int32_t mTileIdx, uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void ComputeWithNorm(uint32_t index = 0, uint8_t enAtomic = 0);

private:
    __aicore__ inline void CopyInAntiQuantParam();

private:
    GlobalTensor<A_T> addGlobal_;
    GlobalTensor<A_T> mulGlobal_;
    LocalTensor<A_T> antiquantScaleLocal_;
    LocalTensor<A_T> antiquantOffsetLocal_;
};

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
    AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulComputeWeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType, hasAntiQuantOffset>::
    Init(AscendC::tiling::TCubeTiling& tiling,
        Mc2Tiling::RCSTiling& cfg, Mc2Tiling::Mc2L2cacheTilePara& tileL2cacheTiling, const LocalTensor<uint8_t>& mmFmtUb12)
{
    MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType, hasAntiQuantOffset>::Init(
        tiling, cfg, tileL2cacheTiling, mmFmtUb12);
    antiquantScaleLocal_ = mmFmtUb12[this->block.tiling.transLength - this->block.tiling.baseN * 2 - 32 * 2]
                               .template ReinterpretCast<A_T>();
    antiquantOffsetLocal_ = mmFmtUb12[this->block.tiling.transLength * 2 - this->block.tiling.baseN * 2 - 32 * 2]
                                .template ReinterpretCast<A_T>();
    // bankconflict pad size is 32 point
    antiquantScaleLocal_.SetSize(this->block.tiling.baseN + 32);
    antiquantOffsetLocal_.SetSize(this->block.tiling.baseN + 32);
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
    AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<
    A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType, hasAntiQuantOffset>::CopyInAntiQuantParam()
{
    DataCopyParams intriParamsExtra;
    intriParamsExtra.blockCount = 1;
    intriParamsExtra.srcStride = 0;
    intriParamsExtra.dstStride = 0;
    if constexpr (antiQuantType == AntiQuantType::PER_TENSOR) {
        intriParamsExtra.blockLen = 1; // 拷贝一个Block
    } else if constexpr (antiQuantType == AntiQuantType::PER_CHANNEL) {
        intriParamsExtra.blockLen = this->block.tiling.baseN * sizeof(A_T) / 32;
    }
    int32_t paramOffset = this->block.offset.offsetBias;
    if constexpr (antiQuantType == AntiQuantType::PER_TENSOR) {
        paramOffset = 0;
    }

    if constexpr (hasAntiQuantOffset) {
        // 搬运offset
        DataCopy(antiquantOffsetLocal_, addGlobal_[paramOffset], intriParamsExtra);
    }
    // 搬运scale
    DataCopy(antiquantScaleLocal_, mulGlobal_[paramOffset], intriParamsExtra);
    if constexpr (antiQuantType == AntiQuantType::PER_TENSOR) {
        SetFlag<HardEvent::MTE2_S>(EVENT_ID7);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID7);
        this->mm.SetAntiQuantScalar(antiquantOffsetLocal_.GetValue(0), antiquantScaleLocal_.GetValue(0));
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
    } else {
        this->mm.SetAntiQuantVector(antiquantOffsetLocal_, antiquantScaleLocal_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);
    }
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
    AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<
    A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
    hasAntiQuantOffset>::InitGlobalBTensor(GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR antiquantScale, GM_ADDR antiquantOffset)
{
    // MC2的计算流中默认B矩阵不变，GM地址无需偏移
    this->bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(bGM), this->block.tiling.Kb * this->block.tiling.N);
    this->biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGM), this->block.tiling.N);
    this->mulGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(antiquantScale), this->block.tiling.N);
    this->addGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(antiquantOffset), this->block.tiling.N);
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
    AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<
    A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
    hasAntiQuantOffset>::Compute(uint32_t index, uint8_t enAtomic)
{
    if constexpr (Mc2L2Cache) {
        ComputeWithL2Cache(index, enAtomic);
    } else {
        ComputeWithNorm(index, enAtomic);
    }
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
    AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<
    A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
    hasAntiQuantOffset>::ComputeWithNorm(uint32_t index, uint8_t enAtomic)
{
    // 每次block循环开始前需要计算初始blockIndex
    this->block.UpdateBlockCnt(index);
    for (uint32_t idx = 0; idx < this->block.blockCnt; i++) {
        if (this->block.blockIndex < this->block.totalBlockCnt) {
            this->block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
            this->SetSingleCoreShape();
            this->mm.SetTensorA(this->aGlobal[this->block.offset.offsetA], this->block.isTransposeA);
            CopyInAntiQuantParam();
            this->mm.SetTensorB(this->bGlobal[this->block.offset.offsetB], this->block.isTransposeB);
            if (this->block.tiling.isBias) {
                this->mm.SetBias(this->biasGlobal[this->block.offset.offsetBias]);
            }
            this->mm.Iterate();
            this->mm.GetTensorC(this->cGl5[this->block.offset.offsetC], enAtomic);
#if __CCE_AICORE__ == 220
#else
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
#endif
        }
        this->block.UpdateBlockIndex();
    }
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
    AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<
    A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
    hasAntiQuantOffset>::ComputeWithL2Cache(uint32_t index, uint8_t enAtomic)
{
    for (int32_t mTileIdx = 0; mTileIdx < this->block.tilingL2.mTileCntL2; mTileIdx++) {
        if (mTileIdx % 2 == 0) {
            ComputeWithL2CacheEven(mTileIdx, index, enAtomic);
        } else {
            ComputeWithL2CacheOdd(mTileIdx, index, enAtomic);
        }
    }
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
    AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<
    A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
    hasAntiQuantOffset>::ComputeWithL2CacheOdd(int32_t mTileIdx, uint32_t index, uint8_t enAtomic)
{
    for (int32_t nTileIdx = this->block.tilingL2.nTileCntL2 - 1; nTileIdx >= 0; nTileIdx--) {
        this->block.UpdateBlockCnt(0, mTileIdx, nTileIdx);
        for (uint32_t j = 0; j < this->block.blockCnt; j++) {
            if (index < this->block.totalTileCnt) {
                this->block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIdx, nTileIdx);
                this->mm.SetSingleShape(
                    this->block.singleCoreM, this->block.singleCoreN, this->block.tiling.singleCoreK);
                this->mm.SetTensorA(this->aGlobal[this->block.offset.offsetA], this->block.isTransposeA);
                CopyInAntiQuantParam();
                this->mm.SetTensorB(this->bGlobal[this->block.offset.offsetB], this->block.isTransposeB);
                if (this->block.tiling.isBias) {
                    this->mm.SetBias(this->biasGlobal[this->block.offset.offsetBias]);
                }
                this->mm.Iterate();
                this->mm.GetTensorC(this->cGl5[this->block.offset.offsetC], enAtomic);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            }
            this->block.UpdateBlockIndex();
        }
    }
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
    AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<
    A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
    hasAntiQuantOffset>::ComputeWithL2CacheEven(int32_t mTileIdx, uint32_t index, uint8_t enAtomic)
{
    for (int32_t nTileIdx = 0; nTileIdx < this->block.tilingL2.nTileCntL2; nTileIdx++) {
        this->block.UpdateBlockCnt(0, mTileIdx, nTileIdx);
        for (uint32_t j = 0; j < this->block.blockCnt; j++) {
            if (index < this->block.totalTileCnt) {
                this->block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIdx, nTileIdx);
                this->mm.SetSingleShape(
                    this->block.singleCoreM, this->block.singleCoreN, this->block.tiling.singleCoreK);
                this->mm.SetTensorA(this->aGlobal[this->block.offset.offsetA], this->block.isTransposeA);
                CopyInAntiQuantParam();
                this->mm.SetTensorB(this->bGlobal[this->block.offset.offsetB], this->block.isTransposeB);
                if (this->block.tiling.isBias) {
                    this->mm.SetBias(this->biasGlobal[this->block.offset.offsetBias]);
                }
                this->mm.Iterate();
                this->mm.GetTensorC(this->cGl5[this->block.offset.offsetC], enAtomic);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
            }
            this->block.UpdateBlockIndex();
        }
    }
}
} // namespace AscendC
#endif // MC2_MATMUL_COMPUTE_WEIGHT_QUANT_H