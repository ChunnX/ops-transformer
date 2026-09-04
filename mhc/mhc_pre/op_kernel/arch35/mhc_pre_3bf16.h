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
 * \file mhc_pre_3bf16.h
 * \brief Packed 3BF16 experimental MHC Pre kernel
 */

#ifndef MHC_PRE_3BF16_H
#define MHC_PRE_3BF16_H

#include "mhc_pre_vector_compute.h"

namespace MhcPre {

using MhcPre3Bf16AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using MhcPre3Bf16BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using MhcPre3Bf16CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
constexpr MatmulConfig MHC_PRE_3BF16_MM_CONFIG = GetMDLConfig(false, false, 0, false, false, false, true);
using MhcPre3Bf16Matmul =
    matmul::MatmulImpl<MhcPre3Bf16AType, MhcPre3Bf16BType, MhcPre3Bf16CType, MhcPre3Bf16CType, MHC_PRE_3BF16_MM_CONFIG>;

constexpr uint32_t MHC_PRE_3BF16_PART_COUNT = 3U;
constexpr float MHC_PRE_3BF16_MID_SCALE = 0.00390625F;
constexpr float MHC_PRE_3BF16_LOW_SCALE = 0.0000152587890625F;

template <class T, class P, int8_t RESI_MODE>
class MhcPre3Bf16 {
public:
    __aicore__ inline explicit MhcPre3Bf16(MhcPre3Bf16Matmul &matmul)
        : mm_(matmul)
    {}

    __aicore__ inline void Init(InitParams initParams)
    {
        vector_.BindGlobalTensors(initParams);
        vector_.InitFromTilingData(initParams.tilingData);
        vector_.InitMNConfig();
        vector_.InitHMixBuffer(initParams);
        vector_.InitPipeAndCoreIdx(initParams.tPipeIn);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.x));
        phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.phi));
        hMix3Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace));

        InitUbBuffers();
        SyncAll<false>();
        if ASCEND_IS_AIV {
            vector_.coreIdx_ = GetBlockIdx() / vector_.kDoubleBufferCount;
            vector_.AIVPreLoad();
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t blockCount = Ceil(vector_.totalLength_, vector_.chunkTSize_);
        for (uint64_t block = vector_.coreIdx_; block < blockCount; block += vector_.coreNum_) {
            vector_.InitBlockParams(block, blockCount);
            ConfigureRowPartition();
            if ASCEND_IS_AIV {
                ProcessRms(block, blockCount);
            } else {
                if (matmulCount_ >= vector_.kDoubleBufferCount) {
                    CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_3BF16_CONSUMED_FLAG);
                    CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_3BF16_CONSUMED_FLAG +
                                                                              MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                }
                ProcessMatmul();
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_MM_READY_FLAG);
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_MM_READY_FLAG +
                                                                         MHC_PRE_SUBBLOCK_FLAG_OFFSET);
            }
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_MM_READY_FLAG);
                ProcessPost();
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_3BF16_CONSUMED_FLAG);
                vector_.vec1Count_++;
            } else {
                matmulCount_++;
            }
        }
        if ASCEND_IS_AIV {
            vector_.invRmsOutQueue_.FreeTensor(vector_.invRmsUb_);
            vector_.biasInQue_.FreeTensor(vector_.biasInUb_);
        }
    }

private:
    __aicore__ inline void InitUbBuffers()
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        vector_.pipe_->InitBuffer(vector_.xInQueue_, vector_.kDoubleBufferCount, vector_.kXInQueueBufferBytes);
        vector_.pipe_->InitBuffer(vector_.outQueue_, vector_.kDoubleBufferCount, vector_.kOutQueueBufferBytes);
        uint32_t invRmsRows = (vector_.curSingleT_ + 1U) / 2U;
        uint32_t invRmsBytes = MhcPreAlign(invRmsRows, vector_.kAlignmentBytes / sizeof(P)) * sizeof(P);
        vector_.pipe_->InitBuffer(vector_.invRmsOutQueue_, 1U, invRmsBytes);
        uint32_t parameterBytes = MhcPreAlign(vector_.mnConfig_.n * sizeof(P), vector_.kAlignmentBytes);
        vector_.pipe_->InitBuffer(vector_.biasInQue_, 1U, parameterBytes);
        vector_.pipe_->InitBuffer(vector_.alphaBuf_, parameterBytes);
        vector_.alphaInUb_ = vector_.alphaBuf_.template Get<P>();
        uint32_t segmentBytes = MhcPreAlign(vector_.V1_BASE_T * vector_.N_ * sizeof(P), vector_.kAlignmentBytes);
        vector_.pipe_->InitBuffer(vector_.tmpBuff_, 2U * segmentBytes);
        vector_.hPreBuff_ = vector_.tmpBuff_.template GetWithOffset<P>(vector_.V1_BASE_T * vector_.N_, 0U);
        vector_.hPostBuff_ = vector_.tmpBuff_.template GetWithOffset<P>(vector_.V1_BASE_T * vector_.N_, segmentBytes);
    }

    __aicore__ inline void ConfigureRowPartition()
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        uint64_t firstRows = (vector_.curSingleT_ + 1U) / 2U;
        if (vector_.subBlockIdx_ == 0U) {
            vector_.vectorOffset_.singleCoreM = firstRows;
            vector_.vectorOffset_.offsetMStart = 0U;
            vector_.vectorOffset_.offsetMEnd = firstRows;
        } else {
            vector_.vectorOffset_.singleCoreM = vector_.curSingleT_ - firstRows;
            vector_.vectorOffset_.offsetMStart = firstRows;
            vector_.vectorOffset_.offsetMEnd = vector_.curSingleT_;
        }
    }

    __aicore__ inline void ProcessRms(uint32_t block, uint32_t blockCount)
    {
        if (vector_.vectorOffset_.singleCoreM == 0U) {
            return;
        }
        for (uint32_t offsetK = 0U; offsetK < vector_.matrixInfo_.nD; offsetK += vector_.ND_LENGTH) {
            uint32_t currentK = MhcPreMin(vector_.ND_LENGTH, vector_.matrixInfo_.nD - offsetK);
            for (uint32_t offsetM = vector_.vectorOffset_.offsetMStart; offsetM < vector_.vectorOffset_.offsetMEnd;
                 offsetM += vector_.V0_BASE_T) {
                uint32_t currentM =
                    MhcPreMin(vector_.V0_BASE_T, static_cast<uint32_t>(vector_.vectorOffset_.offsetMEnd - offsetM));
                uint32_t rmsOffset = offsetM - vector_.vectorOffset_.offsetMStart;
                LocalTensor<P> rmsLocal = vector_.invRmsUb_[rmsOffset];
                vector_.xLocal_ = vector_.xInQueue_.template AllocTensor<T>();
                vector_.DataCopyX(currentM, currentK, offsetM, offsetK);
                LocalTensor<T> xLocal = vector_.xInQueue_.template DeQue<T>();
                LocalTensor<P> scratch = vector_.outQueue_.template AllocTensor<P>();
                if (offsetK == 0U) {
                    vector_.template VFDoV0ProcessXIn<false, true>(
                        (__ubuf__ P *)scratch.GetPhyAddr(), (__ubuf__ P *)rmsLocal.GetPhyAddr(),
                        (__ubuf__ T *)xLocal.GetPhyAddr(), nullptr, currentM, currentK);
                } else {
                    vector_.template VFDoV0ProcessXIn<false, false>(
                        (__ubuf__ P *)scratch.GetPhyAddr(), (__ubuf__ P *)rmsLocal.GetPhyAddr(),
                        (__ubuf__ T *)xLocal.GetPhyAddr(), nullptr, currentM, currentK);
                }
                vector_.xInQueue_.FreeTensor(xLocal);
                vector_.outQueue_.FreeTensor(scratch);
            }
        }
        vector_.V0PostProcess(block, blockCount);
    }

    __aicore__ inline void ProcessMatmul()
    {
        uint64_t packedFusion = MHC_PRE_3BF16_PART_COUNT * vector_.matrixInfo_.fusionSize;
        uint64_t outputOffset = (matmulCount_ & 1U) * vector_.coreNum_ * vector_.chunkTSize_ * packedFusion +
                                vector_.coreIdx_ * vector_.chunkTSize_ * packedFusion;
        uint64_t inputOffset = vector_.globalOffsetM_ * vector_.matrixInfo_.nD;
        mm_.SetOrgShape(vector_.mnConfig_.curSingleCoreM, packedFusion, vector_.matrixInfo_.nD, vector_.matrixInfo_.nD);
        mm_.SetSingleShape(vector_.mnConfig_.curSingleCoreM, packedFusion, vector_.matrixInfo_.nD);
        mm_.SetTensorA(xGm_[inputOffset]);
        mm_.SetTensorB(phiGm_, true);
        mm_.template IterateAll<false>(hMix3Gm_[outputOffset]);
        mm_.End();
    }

    __aicore__ inline void Reconstruct(const LocalTensor<P> &dst, LocalTensor<P> &src, uint32_t rows)
    {
        uint32_t fusion = vector_.matrixInfo_.fusionSize;
        uint32_t packedFusion = MHC_PRE_3BF16_PART_COUNT * fusion;
        for (uint32_t row = 0U; row < rows; ++row) {
            uint32_t srcOffset = row * packedFusion;
            uint32_t dstOffset = row * fusion;
            Muls(dst[dstOffset], src[srcOffset + fusion], MHC_PRE_3BF16_MID_SCALE, fusion);
            PipeBarrier<PIPE_V>();
            Add(dst[dstOffset], dst[dstOffset], src[srcOffset], fusion);
            PipeBarrier<PIPE_V>();
            Muls(src[srcOffset + 2U * fusion], src[srcOffset + 2U * fusion], MHC_PRE_3BF16_LOW_SCALE, fusion);
            PipeBarrier<PIPE_V>();
            Add(dst[dstOffset], dst[dstOffset], src[srcOffset + 2U * fusion], fusion);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessPost()
    {
        uint32_t fusion = vector_.matrixInfo_.fusionSize;
        uint32_t packedFusion = MHC_PRE_3BF16_PART_COUNT * fusion;
        uint64_t pingBase = (vector_.vec1Count_ & 1U) * vector_.coreNum_ * vector_.chunkTSize_ * packedFusion;
        uint64_t coreBase = vector_.coreIdx_ * vector_.chunkTSize_ * packedFusion;
        uint32_t localRmsOffset = 0U;
        for (uint32_t offsetM = vector_.vectorOffset_.offsetMStart; offsetM < vector_.vectorOffset_.offsetMEnd;
             offsetM += vector_.V1_BASE_T) {
            uint32_t rows =
                MhcPreMin(static_cast<uint64_t>(vector_.V1_BASE_T), vector_.vectorOffset_.offsetMEnd - offsetM);
            LocalTensor<P> packedLocal = vector_.xInQueue_.template AllocTensor<P>();
            DataCopyExtParams packedCopy;
            packedCopy.blockCount = 1U;
            packedCopy.blockLen = rows * packedFusion * sizeof(P);
            packedCopy.srcStride = 0U;
            packedCopy.dstStride = 0U;
            DataCopyPadExtParams<P> noPad{true, 0U, 0U, 0};
            DataCopyPad(packedLocal, hMix3Gm_[pingBase + coreBase + offsetM * packedFusion], packedCopy, noPad);
            vector_.xInQueue_.template EnQue<P>(packedLocal);
            packedLocal = vector_.xInQueue_.template DeQue<P>();
            LocalTensor<P> compactLocal = vector_.outQueue_.template AllocTensor<P>();
            Reconstruct(compactLocal, packedLocal, rows);
            if (vector_.outFlag_) {
                DataCopyExtParams outputCopy;
                outputCopy.blockCount = 1U;
                outputCopy.blockLen = rows * fusion * sizeof(P);
                outputCopy.srcStride = 0U;
                outputCopy.dstStride = 0U;
                uint64_t outputOffset = (vector_.globalOffsetM_ + offsetM) * fusion;
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(vector_.hMixGm_[outputOffset], compactLocal, outputCopy);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            vector_.template AIV1PostProcessTile<true>(compactLocal, offsetM, rows, localRmsOffset, fusion);
            vector_.outQueue_.FreeTensor(compactLocal);
            vector_.xInQueue_.FreeTensor(packedLocal);
            localRmsOffset += rows;
        }
    }

    MhcPreVectorCompute<T, P, RESI_MODE> vector_;
    MhcPre3Bf16Matmul &mm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> phiGm_;
    GlobalTensor<P> hMix3Gm_;
    uint32_t matmulCount_ = 0U;
};

} // namespace MhcPre

#endif // MHC_PRE_3BF16_H
