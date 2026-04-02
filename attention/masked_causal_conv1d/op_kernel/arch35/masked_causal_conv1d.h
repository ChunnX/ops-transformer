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
 * \file masked_causal_conv1d.h
 * \brief MaskedCausalConv1d AIV kernel implementation
 *
 * Shape: input [S,B,H], weight [3,H], mask [B,S] -> output [S,B,H]
 * Strategy: H×B×S 3D inter-core split, H outer loop, B middle, S inner;
 *           unified ioQueue layout [bUb][sUb+2][hUb]:
 *             rows 0,1        = prefix x[s0-2], x[s0-1]  (loaded fresh from GM each tile)
 *             rows 2..sUb+1   = current x[s0..s0+sUb-1]
 *           VF writes y[s] in-place to row s; CopyOut writes rows 0..sCur-1.
 */

#ifndef MASKED_CAUSAL_CONV1D_H
#define MASKED_CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "./vf/masked_conv1d_vf.h"
#include "masked_causal_conv1d_struct.h"

namespace MaskedCausalConv1dNs {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;    // double buffer for ioQueue
constexpr uint32_t ALIGN_BYTES = 32; // DataCopy 32-byte alignment unit

// ============================================================================
// MaskedCausalConv1d Kernel
// ============================================================================
template <typename T>
class MaskedCausalConv1d {
public:
    __aicore__ inline MaskedCausalConv1d() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR mask, GM_ADDR y,
                                const MaskedCausalConv1dTilingData* tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const MaskedCausalConv1dTilingData* tiling);
    __aicore__ inline void LoadMask(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur);
    __aicore__ inline void CopyIn(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t h0);
    __aicore__ inline void Compute(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur);
    __aicore__ inline void CopyOut(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t h0);

    // Helper: align up to align bytes
    static __aicore__ inline uint32_t AlignUp(uint32_t n, uint32_t align) {
        return (n + align - 1) / align * align;
    }

private:
    // -------------------------------------------------------------------------
    // Tiling parameters
    // -------------------------------------------------------------------------
    uint32_t S_, B_, H_;

    // H-dim inter-core
    uint32_t hCoreCnt_, hMainCnt_, hBlockFactor_, hBlockTailFactor_;
    // B-dim inter-core
    uint32_t bCoreCnt_, bMainCnt_, bBlockFactor_, bBlockTailFactor_;
    // S-dim inter-core
    uint32_t sCoreCnt_, sMainCnt_, sBlockFactor_, sBlockTailFactor_;

    // UB tile
    uint32_t hUb_;        // = H_REG = 64
    uint32_t ubFactorB_;
    uint32_t ubFactorS_;

    // Main-core loop params
    uint32_t loopNumH_,  ubTailFactorH_;
    uint32_t loopNumB_,  ubTailFactorB_;
    uint32_t loopNumS_,  ubTailFactorS_;

    // Tail-core loop params
    uint32_t tailBlockLoopNumH_,  tailBlockUbTailFactorH_;
    uint32_t tailBlockLoopNumB_,  tailBlockUbTailFactorB_;
    uint32_t tailBlockLoopNumS_,  tailBlockUbTailFactorS_;

    // Stride for non-contiguous x
    uint32_t xSStride_;  // S-dim stride (elements)
    uint32_t xBStride_;  // B-dim stride (elements)

    uint32_t realCoreNum_;

    // Core 3D index (decoded from blockIdx)
    uint32_t hCoreIdx_;
    uint32_t bCoreIdx_;
    uint32_t sCoreIdx_;

    // Current core's assigned ranges
    uint32_t hStart_, hLen_;
    uint32_t bStart_, bLen_;
    uint32_t sStart_, sLen_;

    // Loop params selected for this core (main vs tail)
    uint32_t myLoopNumH_,  myUbTailFactorH_;
    uint32_t myLoopNumB_,  myUbTailFactorB_;
    uint32_t myLoopNumS_,  myUbTailFactorS_;

    // -------------------------------------------------------------------------
    // Pipeline & Global Tensors
    // -------------------------------------------------------------------------
    TPipe pipe_;

    GlobalTensor<T>       xGM_;
    GlobalTensor<T>       weightGM_;
    GlobalTensor<uint8_t> maskGM_;   // [B][S] bool (uint8_t)
    GlobalTensor<T>       yGM_;

    // -------------------------------------------------------------------------
    // UB buffers
    // ioQueue   : [bUb][sUb+2][hUb] T, double buffer (ping-pong)
    //             rows 0,1 = prefix x[s0-2], x[s0-1]; rows 2..sUb+1 = x[s0..]
    // weightQueue: [3][hUb] T, single, loaded once per H-tile
    // maskQueue  : [bUb][sUb] uint8_t, single, loaded per S-tile
    // -------------------------------------------------------------------------
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> ioQueue_;
    TQue<QuePosition::VECIN, 1> weightQueue_;   // [3][hUb]
    TQue<QuePosition::VECIN, 1> maskQueue_;     // [bUb][sUb] bytes
};

// ============================================================================
// ParseTilingData
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::ParseTilingData(
    const MaskedCausalConv1dTilingData* tiling)
{
    S_ = tiling->S;  B_ = tiling->B;  H_ = tiling->H;

    hCoreCnt_ = tiling->hCoreCnt;  hMainCnt_ = tiling->hMainCnt;
    hBlockFactor_ = tiling->hBlockFactor;  hBlockTailFactor_ = tiling->hBlockTailFactor;

    bCoreCnt_ = tiling->bCoreCnt;  bMainCnt_ = tiling->bMainCnt;
    bBlockFactor_ = tiling->bBlockFactor;  bBlockTailFactor_ = tiling->bBlockTailFactor;

    sCoreCnt_ = tiling->sCoreCnt;  sMainCnt_ = tiling->sMainCnt;
    sBlockFactor_ = tiling->sBlockFactor;  sBlockTailFactor_ = tiling->sBlockTailFactor;

    hUb_ = tiling->hUb;
    ubFactorB_ = tiling->ubFactorB;
    ubFactorS_ = tiling->ubFactorS;

    loopNumH_ = tiling->loopNumH;  ubTailFactorH_ = tiling->ubTailFactorH;
    loopNumB_ = tiling->loopNumB;  ubTailFactorB_ = tiling->ubTailFactorB;
    loopNumS_ = tiling->loopNumS;  ubTailFactorS_ = tiling->ubTailFactorS;

    tailBlockLoopNumH_ = tiling->tailBlockLoopNumH;
    tailBlockUbTailFactorH_ = tiling->tailBlockUbTailFactorH;
    tailBlockLoopNumB_ = tiling->tailBlockLoopNumB;
    tailBlockUbTailFactorB_ = tiling->tailBlockUbTailFactorB;
    tailBlockLoopNumS_ = tiling->tailBlockLoopNumS;
    tailBlockUbTailFactorS_ = tiling->tailBlockUbTailFactorS;

    xSStride_ = tiling->xSStride;
    xBStride_ = tiling->xBStride;
    realCoreNum_ = tiling->realCoreNum;
}

// ============================================================================
// Init
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR mask, GM_ADDR y,
    const MaskedCausalConv1dTilingData* tiling)
{
    ParseTilingData(tiling);

    // Decode 3D core index: blockIdx = hCoreIdx + bCoreIdx*hCoreCnt + sCoreIdx*(hCoreCnt*bCoreCnt)
    uint32_t blockIdx = GetBlockIdx();

    hCoreIdx_ = blockIdx % hCoreCnt_;
    bCoreIdx_ = (blockIdx / hCoreCnt_) % bCoreCnt_;
    sCoreIdx_ = blockIdx / (hCoreCnt_ * bCoreCnt_);

    // Compute this core's H range
    if (hCoreIdx_ < hMainCnt_) {
        hStart_ = hCoreIdx_ * hBlockFactor_;
        hLen_   = hBlockFactor_;
    } else {
        hStart_ = hMainCnt_ * hBlockFactor_ + (hCoreIdx_ - hMainCnt_) * hBlockTailFactor_;
        hLen_   = hBlockTailFactor_;
    }

    // Compute this core's B range
    if (bCoreIdx_ < bMainCnt_) {
        bStart_ = bCoreIdx_ * bBlockFactor_;
        bLen_   = bBlockFactor_;
    } else {
        bStart_ = bMainCnt_ * bBlockFactor_ + (bCoreIdx_ - bMainCnt_) * bBlockTailFactor_;
        bLen_   = bBlockTailFactor_;
    }

    // Compute this core's S range
    if (sCoreIdx_ < sMainCnt_) {
        sStart_ = sCoreIdx_ * sBlockFactor_;
        sLen_   = sBlockFactor_;
    } else {
        sStart_ = sMainCnt_ * sBlockFactor_ + (sCoreIdx_ - sMainCnt_) * sBlockTailFactor_;
        sLen_   = sBlockTailFactor_;
    }

    // Select loop params for this core
    bool isHTail = (hCoreIdx_ >= hMainCnt_);
    bool isBTail = (bCoreIdx_ >= bMainCnt_);
    bool isSTail = (sCoreIdx_ >= sMainCnt_);

    myLoopNumH_       = isHTail ? tailBlockLoopNumH_       : loopNumH_;
    myUbTailFactorH_  = isHTail ? tailBlockUbTailFactorH_  : ubTailFactorH_;
    myLoopNumB_       = isBTail ? tailBlockLoopNumB_       : loopNumB_;
    myUbTailFactorB_  = isBTail ? tailBlockUbTailFactorB_  : ubTailFactorB_;
    myLoopNumS_       = isSTail ? tailBlockLoopNumS_       : loopNumS_;
    myUbTailFactorS_  = isSTail ? tailBlockUbTailFactorS_  : ubTailFactorS_;

    // Bind GM
    xGM_.SetGlobalBuffer((__gm__ T*)x, (uint64_t)S_ * xSStride_);
    weightGM_.SetGlobalBuffer((__gm__ T*)weight, (uint64_t)3 * H_);
    maskGM_.SetGlobalBuffer((__gm__ uint8_t*)mask);
    yGM_.SetGlobalBuffer((__gm__ T*)y, (uint64_t)S_ * B_ * H_);

    // ioQueue: [bUb][sUb+2][hUb] T — includes 2 prefix rows per batch
    uint32_t ioBufBytes     = AlignUp(ubFactorB_ * (ubFactorS_ + 2) * hUb_ * sizeof(T), ALIGN_BYTES);
    uint32_t weightBufBytes = AlignUp(3 * hUb_ * sizeof(T),                             ALIGN_BYTES);
    uint32_t maskBufBytes   = AlignUp(ubFactorB_ * ubFactorS_ * sizeof(uint8_t),        ALIGN_BYTES);

    pipe_.InitBuffer(ioQueue_,     BUFFER_NUM, ioBufBytes);
    pipe_.InitBuffer(weightQueue_, 1,          weightBufBytes);
    pipe_.InitBuffer(maskQueue_,   1,          maskBufBytes);
}

// ============================================================================
// LoadMask: load bUbCur × sCur mask values into maskQueue (per S-tile)
// mask GM layout: [B][S] (contiguous, stride = S)
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::LoadMask(
    uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur)
{
    LocalTensor<uint8_t> mBuf = maskQueue_.AllocTensor<uint8_t>();
    maskQueue_.EnQue(mBuf);
}

// ============================================================================
// CopyIn: load [bUb][sCur+2][hUb] into ioQueue
//
// Buffer layout per batch b (element offsets from ioBuf start):
//   row 0:        x[s0-2]  (prefix-2)
//   row 1:        x[s0-1]  (prefix-1)
//   rows 2..sCur+1: x[s0..s0+sCur-1]
//
// Boundary (s0 < 2, only when sCoreIdx=0 && sIter=0):
//   s0 == 0: rows 0,1 = 0; load sCur   rows starting at GM[0]
//   s0 == 1: row 0   = 0; load sCur+1 rows starting at GM[0]
//   s0 >= 2: load all sCur+2 rows starting at GM[s0-2]
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::CopyIn(
    uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t h0)
{
    LocalTensor<T> ioBuf = ioQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    uint32_t rowElems    = hUb_;
    uint32_t bRowStride  = (sCur + 2) * rowElems;  // element stride per batch in ioQueue
    uint32_t srcSkipBytes = (xSStride_ - hUb_) * static_cast<uint32_t>(sizeof(T));

    for (uint32_t b = 0; b < bUbCur; ++b) {
        uint32_t dstBase = b * bRowStride;  // element offset for this batch in ioQueue

        if (s0 >= 2) {
            // Common case: load sCur+2 rows starting at x[s0-2]
            uint64_t gmOff = (uint64_t)(s0 - 2) * xSStride_ + (uint64_t)(b0 + b) * xBStride_ + h0;
            DataCopyExtParams xcp{
                static_cast<uint16_t>(sCur + 2),
                static_cast<uint16_t>(rowElems * sizeof(T)),
                srcSkipBytes, 0, 0
            };
            DataCopyPad(ioBuf[dstBase], xGM_[gmOff], xcp, padParams);
            PipeBarrier<PIPE_ALL>();
        } else if (s0 == 0) {
            // Prefix rows 0,1 = zero; load sCur rows at rows 2..sCur+1
            Duplicate(ioBuf[dstBase], static_cast<T>(0), 2 * rowElems);
            uint64_t gmOff = (uint64_t)0 * xSStride_ + (uint64_t)(b0 + b) * xBStride_ + h0;
            DataCopyExtParams xcp{
                static_cast<uint16_t>(sCur),
                static_cast<uint16_t>(rowElems * sizeof(T)),
                srcSkipBytes, 0, 0
            };
            DataCopyPad(ioBuf[dstBase + 2 * rowElems], xGM_[gmOff], xcp, padParams);
            PipeBarrier<PIPE_ALL>();
        } else {
            // s0 == 1: row 0 = zero; load sCur+1 rows starting at GM[0] into rows 1..sCur+1
            Duplicate(ioBuf[dstBase], static_cast<T>(0), rowElems);
            uint64_t gmOff = (uint64_t)0 * xSStride_ + (uint64_t)(b0 + b) * xBStride_ + h0;
            DataCopyExtParams xcp{
                static_cast<uint16_t>(sCur + 1),
                static_cast<uint16_t>(rowElems * sizeof(T)),
                srcSkipBytes, 0, 0
            };
            DataCopyPad(ioBuf[dstBase + rowElems], xGM_[gmOff], xcp, padParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

    ioQueue_.EnQue(ioBuf);
}

// ============================================================================
// Compute: VF unified-buffer convolution, then apply mask at __aicore__ level
//
// After VF: y[s] is at ioBuf[b*(sCur+2)*hUb_ + s*hUb_]  (row s, in-place)
// Mask zeroing: if mask[b0+b, s0+s] == 0, zero y[s] at that offset.
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::Compute(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur)
{
    LocalTensor<T>       ioBuf = ioQueue_.DeQue<T>();
    LocalTensor<T>       wBuf  = weightQueue_.DeQue<T>();
    LocalTensor<uint8_t> mBuf  = maskQueue_.DeQue<uint8_t>();

    // VF convolution — no mask inside VF (avoid b8 SLD)
    CallMaskedConv1dVF<T>(ioBuf, wBuf, sCur, bUbCur);
    PipeBarrier<PIPE_ALL>();
    // Apply mask at __aicore__ level: zero y[s] where mask == 0
    // y[s] for batch b is at ioBuf[b*(sCur+2)*hUb_ + s*hUb_]
    for (uint32_t b = 0; b < bUbCur; ++b) {
        for (uint32_t s = 0; s < sCur; ++s) {
            uint8_t maskVal = maskGM_.GetValue((uint64_t)(b0 + b) * S_ + s0 + s);
            // uint8_t maskVal = mBuf.GetValue(b * sCur + s);  // UB read, ~1 cycle
            if (maskVal == 0) {
                uint32_t off = b * (sCur + 2) * hUb_ + s * hUb_;
                Duplicate(ioBuf[off], static_cast<T>(0), hUb_);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    maskQueue_.FreeTensor(mBuf);
    weightQueue_.FreeTensor(wBuf);
    ioQueue_.EnQue(ioBuf);
}

// ============================================================================
// CopyOut: write y[0..sCur-1] back to GM
//
// y[s] is at ioBuf[b*(sCur+2)*hUb_ + s*hUb_]  (rows 0..sCur-1 in unified buffer)
// Output y GM layout: [S][B][H] contiguous (S-stride = B*H, B-stride = H).
// Single DataCopyPad per batch writes sCur contiguous rows.
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::CopyOut(
    uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t h0)
{
    LocalTensor<T> ioBuf = ioQueue_.DeQue<T>();

    // Output is [S][B][H]: skip between consecutive S rows = (B_*H_ - hUb_) elements
    uint32_t dstSkipBytes = (B_ * H_ - hUb_) * static_cast<uint32_t>(sizeof(T));

    for (uint32_t b = 0; b < bUbCur; ++b) {
        // y[0..sCur-1] start at row 0 of this batch's region in the unified buffer
        uint32_t ioOff = b * (sCur + 2) * hUb_;

        uint64_t dstOff = (uint64_t)s0 * B_ * H_ + (uint64_t)(b0 + b) * H_ + h0;
        DataCopyExtParams ycp{
            static_cast<uint16_t>(sCur),
            static_cast<uint16_t>(hUb_ * sizeof(T)),
            0, dstSkipBytes, 0
        };
        DataCopyPad(yGM_[dstOff], ioBuf[ioOff], ycp);
    }
    PipeBarrier<PIPE_ALL>();
    ioQueue_.FreeTensor(ioBuf);
}

// ============================================================================
// Process: main compute loop  H(outer) -> B(middle) -> S(inner)
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::Process()
{
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= realCoreNum_) {
        return;
    }

    // ---- H outer loop ----
    for (uint32_t hIter = 0; hIter < myLoopNumH_; ++hIter) {
        uint32_t hUbCur = (hIter < myLoopNumH_ - 1) ? hUb_ : myUbTailFactorH_;
        uint32_t h0     = hStart_ + hIter * hUb_;
        // Load weight for this H-tile: weight[0..2, h0:h0+hUbCur]
        {
            LocalTensor<T> wBuf = weightQueue_.AllocTensor<T>();
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            uint32_t wSkipBytes = (H_ - hUbCur) * static_cast<uint32_t>(sizeof(T));
            DataCopyExtParams wcp{3,
                static_cast<uint16_t>(hUbCur * sizeof(T)),
                wSkipBytes, 0, 0};
            DataCopyPad(wBuf[0], weightGM_[h0], wcp, padParams);
            weightQueue_.EnQue(wBuf);
        }
        PipeBarrier<PIPE_ALL>();
        // ---- B middle loop ----
        for (uint32_t bIter = 0; bIter < myLoopNumB_; ++bIter) {
            uint32_t bUbCur = (bIter < myLoopNumB_ - 1) ? ubFactorB_ : myUbTailFactorB_;
            uint32_t b0     = bStart_ + bIter * ubFactorB_;

            // ---- S inner loop ----
            for (uint32_t sIter = 0; sIter < myLoopNumS_; ++sIter) {
                uint32_t sCur = (sIter < myLoopNumS_ - 1) ? ubFactorS_ : myUbTailFactorS_;
                uint32_t s0   = sStart_ + sIter * ubFactorS_;

                // CopyIn loads sCur+2 rows (rows 0,1 = prefix; rows 2..sCur+1 = x[s0..])
                LoadMask(s0, sCur, b0, bUbCur);
                CopyIn(s0, sCur, b0, bUbCur, h0);
                Compute(s0, sCur, b0, bUbCur);
                CopyOut(s0, sCur, b0, bUbCur, h0);
            }
        }
    }
}

} // namespace MaskedCausalConv1dNs

#endif // MASKED_CAUSAL_CONV1D_H
