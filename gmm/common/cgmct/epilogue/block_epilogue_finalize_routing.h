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
 * \file block_epilogue_finalize_routing.h
 * \brief
 */

#ifndef BLOCK_EPILOGUE_FINALIZE_ROUTING_H
#define BLOCK_EPILOGUE_FINALIZE_ROUTING_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/cgmct_common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../tile/tile_copy_policy.h"

namespace Cgmct {
namespace Gemm {
namespace Block {
namespace {
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t LOGIT_INDEX = 4;
constexpr uint32_t MAX_TILE_M = 256;
constexpr uint32_t MAX_TILE_N = 256;
constexpr uint32_t MAX_TILE_MN = MAX_TILE_M * MAX_TILE_N;
constexpr uint32_t BLOCK_BYTES = 256;
// 248 KiB / 1:2 and 376 KiB / 1:1 both leave about 114 KiB after the
// per-AIV L0C, logit and row-index regions. Two 56 x 256 FP32 output buffers
// consume 112 KiB and fit both products.
constexpr uint64_t MAX_OUTPUT_M_UB = 56;
constexpr uint32_t OUTPUT_BUFFER_ELEMENTS = MAX_OUTPUT_M_UB * MAX_TILE_N;
constexpr uint64_t BLOCK_ELEMENTS_FP32 = 8;
constexpr uint32_t VECTOR_UB_CAPACITY_BYTES = AIC_AIV_CORE_RATIO == 1U ? 376U * 1024U : 248U * 1024U;
constexpr uint32_t EPILOGUE_UB_BYTES = MAX_TILE_MN / AIC_AIV_CORE_RATIO * sizeof(float) +
                                       2U * BLOCK_BYTES * sizeof(float) + 2U * BLOCK_BYTES * sizeof(int64_t) +
                                       2U * OUTPUT_BUFFER_ELEMENTS * sizeof(float);
static_assert(EPILOGUE_UB_BYTES <= VECTOR_UB_CAPACITY_BYTES, "Finalize-routing epilogue exceeds vector UB");
} // namespace

using namespace AscendC;

#define GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS template <typename DataTypeOut_>
#define GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS DataTypeOut_

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
class BlockEpilogueFinalizeRouting {
public:
    __aicore__ inline BlockEpilogueFinalizeRouting();
    __aicore__ inline ~BlockEpilogueFinalizeRouting();

    struct Arguments {
        GM_ADDR yGMAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR logitGmAddr{nullptr};
        GM_ADDR rowIndexGmAddr{nullptr};
        int32_t baseM = 256;
        int32_t baseN = 256;
        Arguments() = default;
    };

    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>; // blk_m, blk_n, blk_k, _
    using BlockCoord =
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>; // y, _, _, _, logit, rowIndex
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t>;           // m, n, k

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);

private:
    __aicore__ inline void CopyInLogit(uint32_t curBaseM, uint64_t offsetM, LocalTensor<float> logitUb);
    __aicore__ inline void CopyInRowIndex(uint32_t curBaseM, uint64_t offsetM, LocalTensor<int64_t> rowIndexUb);
    __aicore__ inline void VFDoLogitMuls(uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit,
                                         uint16_t repeatTimesRe, __ubuf__ DataTypeOut *outUbAddr,
                                         LocalTensor<float> logitUb);
    __aicore__ inline void VectorAtomicProcess(uint32_t curBaseN, uint32_t curVecBaseM, uint32_t rowIndexOffset,
                                               uint64_t yOffset, LocalTensor<DataTypeOut> yLocal,
                                               LocalTensor<int64_t> rowIndexUb);

    // GM ADDR
    AscendC::GlobalTensor<float> logitGlobal_;
    AscendC::GlobalTensor<int64_t> rowIndexGlobal_;
    AscendC::GlobalTensor<DataTypeOut> yGlobal_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeOut> l0cOutUb_;
    AscendC::LocalTensor<float> logitUbPing_;
    AscendC::LocalTensor<float> logitUbPong_;
    AscendC::LocalTensor<int64_t> rowIndexUbPing_;
    AscendC::LocalTensor<int64_t> rowIndexUbPong_;
    AscendC::LocalTensor<DataTypeOut> outUbPing_;
    AscendC::LocalTensor<DataTypeOut> outUbPong_;

    const Params *params_;
    int64_t n_;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint64_t singleM_;
    uint64_t singleN_;
    uint64_t alignN_;
    uint64_t vlForFloatNumber_;
    uint32_t maxL0cElementsPerAiv_;
    uint32_t repeatTimesLine_;
    uint16_t yCrossPingPongID_ = 0;
    uint16_t logitCrossPingPongID_ = 0;
};

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline BlockEpilogueFinalizeRouting<
    GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::BlockEpilogueFinalizeRouting()
{
    constexpr uint32_t taskRatio = AIC_AIV_CORE_RATIO;
    maxL0cElementsPerAiv_ = static_cast<uint32_t>(CeilDiv(MAX_TILE_MN, static_cast<uint64_t>(taskRatio)));
    l0cOutUb_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECIN, 0, maxL0cElementsPerAiv_);
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline BlockEpilogueFinalizeRouting<
    GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::~BlockEpilogueFinalizeRouting()
{
    if ASCEND_IS_AIV {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        SetAtomicNone();
    }
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::Init(
    Params const &params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    yGlobal_.SetGlobalBuffer((__gm__ DataTypeOut *)params_->yGMAddr);
    // input
    uint32_t afterFirstIn = maxL0cElementsPerAiv_ * sizeof(DataTypeOut);
    logitUbPing_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, afterFirstIn, BLOCK_BYTES);
    uint32_t afterSecondIn = afterFirstIn + BLOCK_BYTES * sizeof(float);
    logitUbPong_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, afterSecondIn, BLOCK_BYTES);
    uint32_t afterLogit = afterSecondIn + BLOCK_BYTES * sizeof(float);
    rowIndexUbPing_ = AscendC::LocalTensor<int64_t>(AscendC::TPosition::VECIN, afterLogit, BLOCK_BYTES);
    uint32_t afterRowIndexPing = afterLogit + BLOCK_BYTES * sizeof(int64_t);
    rowIndexUbPong_ = AscendC::LocalTensor<int64_t>(AscendC::TPosition::VECIN, afterRowIndexPing, BLOCK_BYTES);
    uint32_t afterRowIndexPong = afterRowIndexPing + BLOCK_BYTES * sizeof(int64_t);
    outUbPing_ =
        AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECOUT, afterRowIndexPong, OUTPUT_BUFFER_ELEMENTS);
    outUbPong_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECOUT,
                                                   afterRowIndexPong + OUTPUT_BUFFER_ELEMENTS * sizeof(DataTypeOut),
                                                   OUTPUT_BUFFER_ELEMENTS);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    SetAtomicAdd<float>();
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline auto
BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::GetL0c2UbTensor()
{
    return l0cOutUb_;
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<
    GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::UpdateNextProblem(const ProblemShape &problemShape)
{
    n_ = Get<MNK_N>(problemShape);
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<
    GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(const BlockCoord &baseOffset)
{
    if ASCEND_IS_AIV {
        logitGlobal_.SetGlobalBuffer((__gm__ float *)params_->logitGmAddr + Get<LOGIT_INDEX>(baseOffset));
        rowIndexGlobal_.SetGlobalBuffer((__gm__ int64_t *)params_->rowIndexGmAddr + Get<LOGIT_INDEX>(baseOffset));
    }
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::CopyInLogit(
    uint32_t curBaseM, uint64_t offsetM, LocalTensor<float> logitUb)
{
    DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams;
    DataCopyPad(logitUb, logitGlobal_[offsetM], perTokenScaleParams, padParams);
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::CopyInRowIndex(
    uint32_t curBaseM, uint64_t offsetM, LocalTensor<int64_t> rowIndexUb)
{
    DataCopyExtParams rowIndexParams{1, static_cast<uint32_t>(curBaseM * sizeof(int64_t)), 0, 0, 0};
    DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
    DataCopyPad(rowIndexUb, rowIndexGlobal_[offsetM], rowIndexParams, padParams);
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VectorAtomicProcess(
    uint32_t curBaseN, uint32_t curVecBaseM, uint32_t rowIndexOffset, uint64_t yOffset, LocalTensor<DataTypeOut> yLocal,
    LocalTensor<int64_t> rowIndexUb)
{
    const uint32_t blockLenBytes = static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut));
    // UB-to-GM srcStride is measured in 32-byte data blocks. Each source row
    // already occupies Align(blockLenBytes, 32) bytes, so the inter-block gap is zero.
    constexpr uint32_t srcStrideBlocks = 0;
    uint32_t row = 0;
    while (row < curVecBaseM) {
        const int64_t outRow = rowIndexUb.GetValue(rowIndexOffset + row);
        if (outRow < 0) {
            ++row;
            continue;
        }
        uint32_t runLength = 1;
        uint64_t dstStrideBytes = 0;

        if (row + 1 < curVecBaseM) {
            const int64_t nextOutRow = rowIndexUb.GetValue(rowIndexOffset + row + 1);
            if (nextOutRow > outRow) {
                const uint64_t rowDelta = static_cast<uint64_t>(nextOutRow - outRow);
                const uint64_t dstPitchBytes =
                    rowDelta * static_cast<uint64_t>(n_) * static_cast<uint64_t>(sizeof(DataTypeOut));
                if (dstPitchBytes >= blockLenBytes && dstPitchBytes - blockLenBytes <= UINT32_MAX) {
                    dstStrideBytes = dstPitchBytes - blockLenBytes;
                    runLength = 2;
                    int64_t previousOutRow = nextOutRow;
                    while (row + runLength < curVecBaseM) {
                        const int64_t candidateOutRow = rowIndexUb.GetValue(rowIndexOffset + row + runLength);
                        if (candidateOutRow <= previousOutRow ||
                            static_cast<uint64_t>(candidateOutRow - previousOutRow) != rowDelta) {
                            break;
                        }
                        previousOutRow = candidateOutRow;
                        ++runLength;
                    }
                }
            }
        }

        DataCopyExtParams paramsOut{static_cast<uint16_t>(runLength), blockLenBytes, srcStrideBlocks,
                                    static_cast<uint32_t>(dstStrideBytes), 0};
        DataCopyPad(yGlobal_[static_cast<uint64_t>(outRow) * n_ + yOffset], yLocal[row * alignN_], paramsOut);
        row += runLength;
    }
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VFDoLogitMuls(
    uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit, uint16_t repeatTimesRe,
    __ubuf__ DataTypeOut *outUbAddr, LocalTensor<float> logitUb)
{
    __ubuf__ DataTypeOut *l0cOutUbAddr = (__ubuf__ DataTypeOut *)l0cOutUb_.GetPhyAddr();
    __ubuf__ DataTypeOut *logitUbAddr = (__ubuf__ DataTypeOut *)logitUb.GetPhyAddr();
    l0cOutUbAddr += offsetRe;
    logitUbAddr += offsetLogit;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vregLogit;
        AscendC::Reg::RegTensor<DataTypeOut> vregRe, vDstReg;
        AscendC::Reg::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimesLogit; ++i) {
            DataCopy<DataTypeOut, Reg::LoadDist::DIST_BRC_B32>(vregLogit, logitUbAddr + i);
            uint32_t elementNum = singleN_;
            for (uint16_t j = 0; j < repeatTimesRe; ++j) {
                mask = AscendC::Reg::UpdateMask<DataTypeOut>(elementNum);
                AscendC::Reg::DataCopy(vregRe, l0cOutUbAddr + i * alignN_ + j * vlForFloatNumber_);
                AscendC::Reg::Mul(vDstReg, vregLogit, vregRe, mask);
                AscendC::Reg::DataCopy(outUbAddr + i * alignN_ + j * vlForFloatNumber_, vDstReg, mask);
            }
        }
    }
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape &blockShape, const BlockCoord &blockCoord)
{
    singleM_ = Get<MNK_M>(blockShape);
    singleN_ = Get<MNK_N>(blockShape);
    constexpr uint32_t taskRatio = AIC_AIV_CORE_RATIO;
    const uint64_t singleMPerVec = CeilDiv(singleM_, static_cast<uint64_t>(taskRatio));
    alignN_ = CeilDiv(singleN_, BLOCK_ELEMENTS_FP32) * BLOCK_ELEMENTS_FP32;
    const uint64_t mOffset = static_cast<uint64_t>(subBlockIdx_) * singleMPerVec;
    if (mOffset >= singleM_) {
        return;
    }
    const uint64_t singleMInVec = AscendC::Std::min(singleMPerVec, singleM_ - mOffset);
    vlForFloatNumber_ = BLOCK_BYTES / sizeof(DataTypeOut);
    auto repeatTimesRe = CeilDiv(singleN_, vlForFloatNumber_);
    uint64_t logitOffset = Get<LOGIT_INDEX>(blockCoord) + mOffset;
    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint16_t remainRepeatTimesLogit =
        (singleMInVec % MAX_OUTPUT_M_UB != 0) ? singleMInVec % MAX_OUTPUT_M_UB : MAX_OUTPUT_M_UB;
    uint16_t metaPingPongID = logitCrossPingPongID_;
    auto logitUb = metaPingPongID == 0 ? logitUbPing_ : logitUbPong_;
    auto rowIndexUb = metaPingPongID == 0 ? rowIndexUbPing_ : rowIndexUbPong_;
    CopyInLogit(singleMInVec, logitOffset, logitUb);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(metaPingPongID);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(metaPingPongID);
    CopyInRowIndex(singleMInVec, logitOffset, rowIndexUb);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(metaPingPongID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(metaPingPongID);
    logitCrossPingPongID_ = (logitCrossPingPongID_ + 1) & 1;
    uint32_t loopNumY = CeilDiv(singleMInVec, MAX_OUTPUT_M_UB);
    for (int32_t i = 0; i < loopNumY; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(yCrossPingPongID_);
        repeatTimesLine_ = (i == loopNumY - 1) ? remainRepeatTimesLogit : MAX_OUTPUT_M_UB;
        __ubuf__ DataTypeOut *outUbAddr = yCrossPingPongID_ == 0 ? (__ubuf__ DataTypeOut *)outUbPing_.GetPhyAddr() :
                                                                   (__ubuf__ DataTypeOut *)outUbPong_.GetPhyAddr();
        VFDoLogitMuls(i * MAX_OUTPUT_M_UB * alignN_, i * MAX_OUTPUT_M_UB, repeatTimesLine_, repeatTimesRe, outUbAddr,
                      logitUb);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(yCrossPingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(yCrossPingPongID_);
        if (i == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(metaPingPongID);
        }
        auto yLocal = yCrossPingPongID_ == 0 ? outUbPing_ : outUbPong_;
        VectorAtomicProcess(singleN_, repeatTimesLine_, i * MAX_OUTPUT_M_UB, yOffset, yLocal, rowIndexUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(yCrossPingPongID_);
        yCrossPingPongID_ = (yCrossPingPongID_ + 1) & 1;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(metaPingPongID);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(logitCrossPingPongID_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(logitCrossPingPongID_);
    return;
}
} // namespace Block
} // namespace Gemm
} // namespace Cgmct
#endif
#endif
