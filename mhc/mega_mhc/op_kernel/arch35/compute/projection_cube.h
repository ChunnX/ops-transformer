/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_PROJECTION_CUBE_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_PROJECTION_CUBE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MegaMhc::Cube {
using namespace AscendC;

/** Execute host-described FP32/HF32 projection tiles; this class owns only the AIC Cube micro-pipeline. */
template <bool USE_HF32>
class ProjectionCube {
public:
    /** Allocate the exact planned L1/L0 slots and select HF32 mode on the Cube participant. */
    __aicore__ inline void Init(TPipe *pipe, const MegaMhcTilingData *tiling)
    {
        tiling_ = tiling;
        pipe->InitBuffer(l1Buffer_, static_cast<uint32_t>(tiling_->l1PeakBytes));
        LocalTensor<float> l1Base = l1Buffer_.Get<float>();
        aL1_ = l1Base;
        bL1_ = l1Base[tiling_->l1BOffsetElements];

        aL0_[0] = LocalTensor<float>(TPosition::A2, 0, tiling_->l0ASlotElements);
        aL0_[1] = LocalTensor<float>(TPosition::A2, static_cast<uint64_t>(tiling_->l0ASlotElements) * sizeof(float),
                                     tiling_->l0ASlotElements);
        bL0_[0] = LocalTensor<float>(TPosition::B2, 0, tiling_->l0BSlotElements);
        bL0_[1] = LocalTensor<float>(TPosition::B2, static_cast<uint64_t>(tiling_->l0BSlotElements) * sizeof(float),
                                     tiling_->l0BSlotElements);
        cL0_[0] = LocalTensor<float>(TPosition::CO1, 0, tiling_->l0CSlotElements);
        cL0_[1] = LocalTensor<float>(TPosition::CO1, static_cast<uint64_t>(tiling_->l0CSlotElements) * sizeof(float),
                                     tiling_->l0CSlotElements);

        SetFlag<HardEvent::MTE1_MTE2>(0);
        SetFlag<HardEvent::MTE1_MTE2>(1);
        SetFlag<HardEvent::MTE1_MTE2>(2);
        SetFlag<HardEvent::MTE1_MTE2>(3);
        SetFlag<HardEvent::M_MTE1>(3);
        SetFlag<HardEvent::M_MTE1>(4);
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::FIX_M>(1);
        if ASCEND_IS_AIC {
            if constexpr (USE_HF32) {
                SetHF32Mode(1);
                SetHF32TransMode(1);
            }
        }
    }

    /** Copy the planned [M,K] FP32 activation slice from GM staging into an L1 NZ slot. */
    __aicore__ inline void LoadA(const GlobalTensor<float> &stageGm, uint64_t stageOffset, uint32_t mRows, uint32_t nzM,
                                 uint32_t kCount, uint32_t slot)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(slot + 2);
        LocalTensor<float> destination = aL1_[slot * tiling_->l1ASlotElements];
        Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = mRows;
        params.dValue = kCount;
        params.srcNdMatrixStride = 1;
        params.srcDValue = tiling_->kL1;
        params.dstNzC0Stride = nzM;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        DataCopy(destination, stageGm[stageOffset], params);
        SetFlag<HardEvent::MTE2_MTE1>(slot + 2);
        WaitFlag<HardEvent::MTE2_MTE1>(slot + 2);
    }

    /** Consume the local reuse token for an A slot already filled by the paired AIV MTE3. */
    __aicore__ inline void AcceptDirectA(uint32_t slot)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(slot + 2U);
    }

    /** Copy the planned [Q,K] FP32 projection-weight slice into its transposed L1 NZ slot. */
    __aicore__ inline void LoadB(const GlobalTensor<float> &phiGm, uint32_t kStart, uint32_t kCount, uint32_t slot)
    {
        LocalTensor<float> destination = bL1_[slot * tiling_->l1BSlotElements];
        WaitFlag<HardEvent::MTE1_MTE2>(slot);
        Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = tiling_->q;
        params.dValue = kCount;
        params.srcNdMatrixStride = 1;
        params.srcDValue = tiling_->k;
        params.dstNzC0Stride = tiling_->nzN;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        DataCopy(destination, phiGm[kStart], params);
        SetFlag<HardEvent::MTE2_MTE1>(slot);
        WaitFlag<HardEvent::MTE2_MTE1>(slot);
    }

    /** Compute one host-sized L1 tile and Fixpipe the completed K partial to padded [group,M,NZ_N]. */
    __aicore__ inline void ComputeL1Tile(uint32_t mRows, uint32_t nzM, uint32_t l0FullLoops, uint32_t l0Tail,
                                         uint32_t tailLoadKStep, uint32_t fullMmadLoops, uint32_t tailMmadLoops,
                                         uint32_t tailMmadKStep, uint32_t tailMmadFinalKStep, bool firstInPartial,
                                         bool lastInPartial, uint32_t l1Slot, const GlobalTensor<float> &partialGm,
                                         uint64_t partialOffset)
    {
        if (firstInPartial) {
            WaitFlag<HardEvent::FIX_M>(cSlot_);
        }
        uint32_t l0Slot = 0;
        const uint32_t l0Count = l0FullLoops + static_cast<uint32_t>(l0Tail != 0);
        for (uint32_t l0Index = 0; l0Index < l0Count; ++l0Index) {
            const bool l0IsTail = l0Index >= l0FullLoops;
            const uint32_t loadKStep = l0IsTail ? tailLoadKStep : tiling_->fullL0LoadKStep;
            const uint32_t madLoops = l0IsTail ? tailMmadLoops : fullMmadLoops;
            const uint32_t aOffset = l0Index * tiling_->baseK * nzM;
            const uint32_t bOffset = l0Index * tiling_->baseK * tiling_->nzN;
            WaitFlag<HardEvent::M_MTE1>(l0Slot + 3);

            LoadA2(aL1_[l1Slot * tiling_->l1ASlotElements + aOffset], aL0_[l0Slot], loadKStep);
            LoadB2(bL1_[l1Slot * tiling_->l1BSlotElements + bOffset], bL0_[l0Slot], loadKStep);
            SetFlag<HardEvent::MTE1_M>(l0Slot);
            WaitFlag<HardEvent::MTE1_M>(l0Slot);

            uint32_t innerKOffset = 0;
            for (uint32_t inner = 0; inner < madLoops; ++inner) {
                const bool innerIsTail = l0IsTail && inner + 1 == madLoops && tailMmadLoops != 0;
                const uint32_t madK =
                    innerIsTail ? tailMmadFinalKStep :
                                  (!USE_HF32 ? tiling_->fullMmadKStep : (l0IsTail ? tailMmadKStep : tiling_->baseK));
                MmadParams params;
                params.m = nzM;
                // Compute the allocated NZ width; only the first q columns are logical outputs.
                params.n = tiling_->nzN;
                params.k = madK;
                params.disableGemv = true;
                params.cmatrixInitVal = firstInPartial && l0Index == 0 && inner == 0;
                params.cmatrixSource = false;
                // M_FIX/FIX_M own L0C synchronization. Fixpipe may emit fewer M rows
                // than the padded Mmad tile, so it must not leave unit flags occupied.
                params.unitFlag = 0;
                Mmad(cL0_[cSlot_], aL0_[l0Slot][innerKOffset * nzM], bL0_[l0Slot][innerKOffset * tiling_->nzN], params);
                if (tiling_->mmadNeedsBarrier != 0U) {
                    PipeBarrier<PIPE_M>();
                }
                innerKOffset += madK;
            }
            SetFlag<HardEvent::M_MTE1>(l0Slot + 3);
            l0Slot ^= 1U;
        }

        if (lastInPartial) {
            WritePartial(partialGm[partialOffset], mRows, nzM);
        }
        SetFlag<HardEvent::MTE1_MTE2>(l1Slot + 2);
        SetFlag<HardEvent::MTE1_MTE2>(l1Slot);
    }

    /** Drain both L0/L1 pipelines and restore the Cube precision mode before returning. */
    __aicore__ inline void End()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::MTE1_MTE2>(1);
        WaitFlag<HardEvent::MTE1_MTE2>(2);
        WaitFlag<HardEvent::MTE1_MTE2>(3);
        WaitFlag<HardEvent::M_MTE1>(3);
        WaitFlag<HardEvent::M_MTE1>(4);
        WaitFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(1);
        if ASCEND_IS_AIC {
            if constexpr (USE_HF32) {
                SetHF32Mode(0);
                SetHF32TransMode(0);
            }
        }
    }

private:
    /** Load one host-sized A L0 tile from NZ L1 using the supplied M/K dimensions. */
    __aicore__ inline void LoadA2(const LocalTensor<float> &source, const LocalTensor<float> &destination,
                                  uint32_t loadKStep)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = tiling_->loadMStride;
        params.kStep = loadKStep;
        params.srcStride = tiling_->loadMStride;
        params.dstStride = tiling_->loadMStride;
        params.ifTranspose = false;
        LoadData(destination, source, params);
    }

    /** Load one host-sized B L0 tile from NZ L1 using the supplied Q/K dimensions. */
    __aicore__ inline void LoadB2(const LocalTensor<float> &source, const LocalTensor<float> &destination,
                                  uint32_t loadKStep)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = tiling_->loadNStride;
        params.kStep = loadKStep;
        params.srcStride = tiling_->loadNStride;
        params.dstStride = tiling_->loadNStride;
        params.ifTranspose = false;
        LoadData(destination, source, params);
    }

    /** Store partial = A * B^T, then rotate L0C so the next partial can overlap this Fixpipe. */
    __aicore__ inline void WritePartial(const GlobalTensor<float> &destination, uint32_t mRows, uint32_t nzM)
    {
        SetFlag<HardEvent::M_FIX>(cSlot_);
        WaitFlag<HardEvent::M_FIX>(cSlot_);
        DataCopyCO12DstParams params;
        params.nSize = tiling_->fixpipeRowStride;
        params.mSize = mRows;
        params.dstStride = tiling_->fixpipeRowStride;
        params.srcStride = nzM;
        params.quantPre = QuantMode_t::NoQuant;
        params.nz2ndEn = true;
        params.unitFlag = 0;
        SetFixpipeNz2ndFlag(1, 1, 1);
        DataCopy(destination, cL0_[cSlot_], params);
        SetFlag<HardEvent::FIX_M>(cSlot_);
        cSlot_ ^= 1U;
    }

    const MegaMhcTilingData *tiling_ = nullptr;
    TBuf<TPosition::A1> l1Buffer_;
    LocalTensor<float> aL1_;
    LocalTensor<float> bL1_;
    LocalTensor<float> aL0_[2];
    LocalTensor<float> bL0_[2];
    LocalTensor<float> cL0_[2];
    uint32_t cSlot_ = 0;
};

} // namespace MegaMhc::Cube

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_PROJECTION_CUBE_H
