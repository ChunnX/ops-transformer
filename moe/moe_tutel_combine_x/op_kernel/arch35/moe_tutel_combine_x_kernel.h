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
 * \file moe_tutel_combine_x_kernel.h
 * \brief Ascend C kernel of MoeTutelCombineX for arch35 (Ascend 950).
 *
 * Scatter-combine: x_grad[b, t, :] = y_grad[b, idx0 * C + idx1, :] * gates[b, t]
 * (each output row is contributed by exactly one token, so the "accumulate"
 * equals an overwrite when the caller starts from an uninitialized buffer).
 */
#ifndef MOE_TUTEL_COMBINE_X_KERNEL_H
#define MOE_TUTEL_COMBINE_X_KERNEL_H

#include "kernel_operator.h"
#include "moe_tutel_combine_x_tiling_struct.h"

// ---------------------------------------------------------------------------
// Empty branch: zero the whole x_grad output.
// ---------------------------------------------------------------------------
template <typename T>
class ScatterAddEmptyKernel {
public:
    static constexpr int64_t kZeroChunkElems = 8192;

    __aicore__ inline void Init(GM_ADDR y_grad, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations,
                                GM_ADDR x_grad, MoetutelcombinexTilingData *td, AscendC::TPipe *pipe)
    {
        td_ = td;
        gmXGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x_grad));
        if (pipe != nullptr) {
            pipe->InitBuffer(zeroBuf_, kZeroChunkElems * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (td_ == nullptr) {
            return;
        }
        // Empty tensor: scatter-combine contributes nothing, so the output
        // x_grad must be all zeros. batchSize/samples/hidden are filled by the
        // host tiling so the kernel can compute outputElems.
        int64_t totalElems = td_->batchSize * td_->samples * td_->hidden;
        if (totalElems <= 0) {
            return;
        }
        ZeroOutput(totalElems);
    }

private:
    __aicore__ inline void ZeroOutput(int64_t totalElems)
    {
        AscendC::LocalTensor<T> zero = zeroBuf_.Get<T>();
        AscendC::Duplicate(zero, static_cast<T>(0), kZeroChunkElems);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t off = 0; off < totalElems; off += kZeroChunkElems) {
            int64_t cur = (totalElems - off < kZeroChunkElems) ? (totalElems - off) : kZeroChunkElems;
            uint32_t curBytes = static_cast<uint32_t>(cur * sizeof(T));
            AscendC::DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = curBytes;
            p.srcStride = 0;
            p.dstStride = 0;
            AscendC::DataCopyPad(gmXGrad_[off], zero, p);
        }
    }

    const MoetutelcombinexTilingData *td_ = nullptr;
    AscendC::GlobalTensor<T> gmXGrad_;
    AscendC::TBuf<AscendC::TPosition::VECIN> zeroBuf_;
};

// ---------------------------------------------------------------------------
// VF helpers: xGradRow = yGradRow * gateScalar (overwrite, no accumulate).
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void ScatterMulVfF32(
    __ubuf__ T *xGradRow, __ubuf__ T *yGradRow,
    float gateScalar, uint32_t totalElems, uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<T> yReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    constexpr uint32_t kStep = 64;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, kStep);
        uint32_t remain = totalElems - i * kStep;
        mask = AscendC::Reg::UpdateMask<T>(remain);
        AscendC::Reg::LoadAlign(yReg, yGradRow, aReg);
        AscendC::Reg::Muls(yReg, yReg, gateScalar, mask);
        AscendC::Reg::StoreAlign(xGradRow, yReg, aReg, mask);
    }
}

constexpr AscendC::Reg::CastTrait kCastB162B32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait kCastB322B16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <typename T>
__simd_vf__ inline void ScatterMulVfB16(
    __ubuf__ T *xGradRow, __ubuf__ T *yGradRow,
    float gateScalar, uint32_t totalElems, uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<T> yRegT;
    AscendC::Reg::RegTensor<float> yRegF32;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t kStep = 64;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        uint32_t off = i * kStep;
        uint32_t remain = totalElems - off;
        mask = AscendC::Reg::UpdateMask<uint32_t>(remain);
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(yRegT, yGradRow + off);
        AscendC::Reg::Cast<float, T, kCastB162B32>(yRegF32, yRegT, mask);
        AscendC::Reg::Muls(yRegF32, yRegF32, gateScalar, mask);
        AscendC::Reg::Cast<T, float, kCastB322B16>(yRegT, yRegF32, mask);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(xGradRow + off, yRegT, mask);
    }
}

// ---------------------------------------------------------------------------
// Normal branches (no-batch / batched). The hasBatch distinction is resolved
// at runtime through template dispatch so the inner code keeps the
// compile-time specialization of the index arithmetic.
// ---------------------------------------------------------------------------
template <typename T>
class MoeTutelCombineXKernel {
public:
    __aicore__ inline void Init(GM_ADDR inputs[kMaxInputSlots],
                                GM_ADDR outputs[kMaxOutputSlots],
                                const MoetutelcombinexTilingData *td)
    {
        td_ = td;
        gmYGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputs[0]));
        gmGates_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputs[1]));
        gmIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(inputs[2]));
        gmLocations_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(inputs[3]));
        gmXGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outputs[0]));

        rawIndices_ = reinterpret_cast<__gm__ int32_t *>(inputs[2]);
        rawLocations_ = reinterpret_cast<__gm__ int32_t *>(inputs[3]);
        rawGates_ = reinterpret_cast<__gm__ T *>(inputs[1]);

        int32_t coreId = AscendC::GetBlockIdx();
        int64_t cTB = td_->coreTokenBase;
        int32_t bC = td_->bigCoreCnt;
        coreTokenStart_ = (coreId < bC) ? static_cast<int64_t>(coreId) * (cTB + 1)
                                        : static_cast<int64_t>(bC) * (cTB + 1) + static_cast<int64_t>(coreId - bC) * cTB;
        coreTokenEnd_ = coreTokenStart_ + ((coreId < bC) ? (cTB + 1) : cTB);

        // VECIN queues for DataCopyPad (byte-precision DMA) + Reg API (VF)
        uint32_t hChunk = static_cast<uint32_t>(td_->hChunk);
        uint32_t allocElems = ((hChunk + 63u) / 64u) * 64u;
        pipe_.InitBuffer(inQueueY, 1, allocElems * sizeof(T));
        pipe_.InitBuffer(inQueueX, 1, allocElems * sizeof(T));
        // VECCALC fp32 gate scalar + temp
        pipe_.InitBuffer(gateS_, sizeof(float));
        pipe_.InitBuffer(gateTmp_, 32);
        // VECIN zero buffer used to zero the x_grad rows of invalid tokens
        // (idx0<0 / idx1>=capacity / srcRow>=eC).
        zeroChunkElems_ = (td_->hidden < kZeroChunkElems) ? td_->hidden : kZeroChunkElems;
        if (zeroChunkElems_ < 1) {
            zeroChunkElems_ = 1;
        }
        pipe_.InitBuffer(zeroBuf_, zeroChunkElems_ * sizeof(T));

        samples_ = td_->samples;
        eC_ = td_->eC;
        hidden_ = td_->hidden;

        // Kernel-side capacity correction for the E=1 fallback (ACLNN path).
        if (td_->capacity == td_->eC && td_->eC > 0) {
            int64_t totalTokens = td_->tokenTotal > 0 ? td_->tokenTotal : 1;
            int32_t maxIdx = -1;
            for (int64_t i = 0; i < totalTokens; i++) {
                int32_t v = AscendC::ReadGmByPassDCache<int32_t>(rawIndices_ + i);
                if (v > maxIdx) {
                    maxIdx = v;
                }
            }
            int64_t E = (maxIdx >= 0) ? (static_cast<int64_t>(maxIdx) + 1) : 1;
            capacity_ = (E > 0) ? (td_->eC / E) : 0;
        } else {
            capacity_ = td_->capacity;
        }
    }

    __aicore__ inline void Process(bool hasBatch)
    {
        if (hasBatch) {
            ProcessImpl<true>();
        } else {
            ProcessImpl<false>();
        }
    }

private:
    static constexpr int64_t kZeroChunkElems = 8192;

    template <bool hasBatch>
    __aicore__ inline void ProcessImpl()
    {
        for (int64_t token = coreTokenStart_; token < coreTokenEnd_; token++) {
            int64_t b = 0;
            int64_t t = token;
            if constexpr (hasBatch) {
                b = token / td_->samples;
                t = token % td_->samples;
            }
            int64_t metaIdx = (hasBatch) ? (b * td_->samples + t) : t;

            // Scalar reads via ReadGmByPassDCache (byte-addressable)
            int32_t idx0 = AscendC::ReadGmByPassDCache<int32_t>(rawIndices_ + metaIdx);
            int32_t idx1 = AscendC::ReadGmByPassDCache<int32_t>(rawLocations_ + metaIdx);
            float gateVal;
            if constexpr (std::is_same_v<T, float>) {
                int32_t gb = AscendC::ReadGmByPassDCache<int32_t>(
                    reinterpret_cast<__gm__ int32_t *>(rawGates_ + metaIdx));
                gateVal = *reinterpret_cast<float *>(&gb);
            } else {
                int16_t gb = AscendC::ReadGmByPassDCache<int16_t>(
                    reinterpret_cast<__gm__ int16_t *>(rawGates_ + metaIdx));
                T gt;
                *reinterpret_cast<int16_t *>(&gt) = gb;
                gateTmp_.Get<T>().SetValue(0, gt);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Cast(gateS_.Get<float>(), gateTmp_.Get<T>(), AscendC::RoundMode::CAST_NONE, 1);
                AscendC::PipeBarrier<PIPE_ALL>();
                gateVal = gateS_.Get<float>().GetValue(0);
            }

            // Each output row is written by exactly one token; invalid entries
            // must produce an all-zero row (the caller may not have zeroed
            // x_grad).
            if (idx0 < 0 || static_cast<int64_t>(idx1) >= capacity_) {
                ZeroOutputRow<hasBatch>(b, t);
                continue;
            }
            int64_t srcRow = static_cast<int64_t>(idx0) * capacity_ + static_cast<int64_t>(idx1);
            if (srcRow < 0 || srcRow >= eC_) {
                ZeroOutputRow<hasBatch>(b, t);
                continue;
            }

            for (int64_t c = 0; c < td_->hChunkCnt; c++) {
                int64_t hOff = c * td_->hChunk;
                int64_t curLen = (c == td_->hChunkCnt - 1) ? td_->hChunkTail : td_->hChunk;
                uint32_t curBytes = static_cast<uint32_t>(curLen * sizeof(T));

                // CopyIn: DataCopyPad GM->UB (byte precision, VECIN dst)
                AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
                CopyInYChunk<hasBatch>(yLocal, b, srcRow, hOff, curBytes);
                inQueueY.EnQue(yLocal);
                AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                inQueueX.EnQue(xLocal);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);

                // Compute: DeQue -> VF (overwrite xLocal = yLocal * gate)
                yLocal = inQueueY.DeQue<T>();
                xLocal = inQueueX.DeQue<T>();
                ComputeChunk(yLocal, xLocal, static_cast<uint32_t>(curLen), gateVal);
                inQueueY.FreeTensor(yLocal);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);

                // CopyOut: DataCopyPad UB->GM (byte precision, VECIN src)
                CopyOutXChunk<hasBatch>(xLocal, b, t, hOff, curBytes);
                inQueueX.FreeTensor(xLocal);
            }
        }
    }

    template <bool hasBatch>
    __aicore__ inline void ZeroOutputRow(int64_t b, int64_t t)
    {
        int64_t rowOff = t * hidden_;
        if constexpr (hasBatch) {
            rowOff += b * samples_ * hidden_;
        }
        int64_t total = hidden_;
        if (total <= 0) {
            return;
        }
        AscendC::LocalTensor<T> zero = zeroBuf_.Get<T>();
        AscendC::Duplicate(zero, static_cast<T>(0), zeroChunkElems_);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t off = 0; off < total; off += zeroChunkElems_) {
            int64_t cur = (total - off < zeroChunkElems_) ? (total - off) : zeroChunkElems_;
            uint32_t curBytes = static_cast<uint32_t>(cur * sizeof(T));
            AscendC::DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = curBytes;
            p.srcStride = 0;
            p.dstStride = 0;
            AscendC::DataCopyPad(gmXGrad_[rowOff + off], zero, p);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <bool hasBatch>
    __aicore__ inline void CopyInYChunk(AscendC::LocalTensor<T> &dst, int64_t b, int64_t srcRow,
                                        int64_t hOff, uint32_t bytes)
    {
        int64_t gmOff = srcRow * td_->hidden + hOff;
        if constexpr (hasBatch) {
            gmOff += b * td_->eC * td_->hidden;
        }
        AscendC::DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = bytes;
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, gmYGrad_[gmOff], p, pad);
    }

    __aicore__ inline void ComputeChunk(AscendC::LocalTensor<T> &yLocal, AscendC::LocalTensor<T> &xLocal,
                                        uint32_t len, float gateVal)
    {
        constexpr uint32_t kStep = 64;
        uint16_t rt = static_cast<uint16_t>((len + kStep - 1) / kStep);
        if constexpr (std::is_same_v<T, float>) {
            asc_vf_call<ScatterMulVfF32<T>>(
                (__ubuf__ T *)xLocal.GetPhyAddr(), (__ubuf__ T *)yLocal.GetPhyAddr(), gateVal, len, rt);
        } else {
            asc_vf_call<ScatterMulVfB16<T>>(
                (__ubuf__ T *)xLocal.GetPhyAddr(), (__ubuf__ T *)yLocal.GetPhyAddr(), gateVal, len, rt);
        }
    }

    template <bool hasBatch>
    __aicore__ inline void CopyOutXChunk(AscendC::LocalTensor<T> &src, int64_t b, int64_t t,
                                         int64_t hOff, uint32_t bytes)
    {
        int64_t gmOff = t * td_->hidden + hOff;
        if constexpr (hasBatch) {
            gmOff += b * td_->samples * td_->hidden;
        }
        AscendC::DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = bytes;
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopyPad(gmXGrad_[gmOff], src, p);
    }

    const MoetutelcombinexTilingData *td_;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> gmYGrad_;
    AscendC::GlobalTensor<T> gmGates_;
    AscendC::GlobalTensor<T> gmXGrad_;
    AscendC::GlobalTensor<int32_t> gmIndices_;
    AscendC::GlobalTensor<int32_t> gmLocations_;
    __gm__ int32_t *rawIndices_;
    __gm__ int32_t *rawLocations_;
    __gm__ T *rawGates_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateS_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateTmp_;
    AscendC::TBuf<AscendC::TPosition::VECIN> zeroBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    int64_t coreTokenStart_ = 0;
    int64_t coreTokenEnd_ = 0;
    int64_t samples_ = 0;
    int64_t eC_ = 0;
    int64_t hidden_ = 0;
    int64_t capacity_ = 0;
    int64_t zeroChunkElems_ = 0;
};

#endif // MOE_TUTEL_COMBINE_X_KERNEL_H
