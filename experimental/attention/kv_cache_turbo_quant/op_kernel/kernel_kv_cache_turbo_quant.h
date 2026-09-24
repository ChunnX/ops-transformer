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
 * \file kernel_kv_cache_turbo_quant.h
 * \brief Encode MIX：fp32 Cube + TILE Prefetch + 向量 Pack（Mul+ReduceSum，少 GetValue）。
 */
#ifndef KERNEL_KV_CACHE_TURBO_QUANT_H_
#define KERNEL_KV_CACHE_TURBO_QUANT_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

#define KV_ENC_INLINE __aicore__ inline
#include "encode_one_core.h"
#undef KV_ENC_INLINE

namespace KvCacheTurboQuantKernel {

constexpr int32_t kHeadDimI32 = 128;
constexpr int64_t kHeadDim = 128;
constexpr uint64_t kMatmulSysWorkspace = 32ULL * 1024ULL * 1024ULL;
constexpr uint32_t kPrefetchTile = 64U;
constexpr uint32_t kGemmTile = 128U;
constexpr uint32_t kNormTile = 16U;
constexpr uint32_t kNormCap = 2048U;
// P1：多行拼成一次 Compare/Select，提高向量利用率（语义仍按行 ≡ golden 中点）
constexpr uint32_t kQuantBatch = 32U;
constexpr int32_t kElemPerBlkFp32 = 8;
constexpr int32_t kQuantBatchDim = static_cast<int32_t>(kQuantBatch) * kHeadDimI32;

struct TilingView {
    int64_t mseBits;
    int64_t idxBytes;
    int64_t qjlBytes;
    uint32_t totalRows;
    uint32_t coreRows;
    uint32_t vecCoreNum;
};

struct DeviceSqrt {
    __aicore__ inline float operator()(float x) const
    {
        if (x <= 0.0f) {
            return 0.0f;
        }
        float y = x;
        if (y < 1.0f) {
            y = 1.0f;
        }
        for (int k = 0; k < 8; ++k) {
            y = 0.5f * (y + x / y);
        }
        return y;
    }
};

typedef MatmulType<TPosition::GM, CubeFormat::ND, float> AType;
typedef MatmulType<TPosition::GM, CubeFormat::ND, float, true> BType;
typedef MatmulType<TPosition::GM, CubeFormat::ND, float> CType;
typedef MatmulType<TPosition::GM, CubeFormat::ND, float> BiasType;

template <typename MM>
class KernelKvCacheTurboQuant {
public:
    __aicore__ inline KernelKvCacheTurboQuant() {}

    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR rotation, GM_ADDR qjl, GM_ADDR quantIdx, GM_ADDR quantQjl,
                                GM_ADDR quantNorm, GM_ADDR quantGamma, GM_ADDR workspace, const TilingView &tiling,
                                TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;

        kvGm_.SetGlobalBuffer((__gm__ bfloat16_t *)kv);
        rotGm_.SetGlobalBuffer((__gm__ float *)rotation);
        qjlGm_.SetGlobalBuffer((__gm__ float *)qjl);
        idxGm_.SetGlobalBuffer((__gm__ uint8_t *)quantIdx);
        qjlOutGm_.SetGlobalBuffer((__gm__ uint8_t *)quantQjl);
        normGm_.SetGlobalBuffer((__gm__ bfloat16_t *)quantNorm);
        gammaGm_.SetGlobalBuffer((__gm__ bfloat16_t *)quantGamma);

        const uint64_t n = static_cast<uint64_t>(tiling_.totalRows);
        __gm__ float *wsBase = (__gm__ float *)(workspace + kMatmulSysWorkspace);
        unitsGm_.SetGlobalBuffer(wsBase);
        matOutGm_.SetGlobalBuffer(wsBase + n * kHeadDim);
    }

    __aicore__ inline void InitBuffers()
    {
        pipe_->InitBuffer(xFp32Buf_, kPrefetchTile * kHeadDim * sizeof(float));
        pipe_->InitBuffer(kvBf16Buf_, kPrefetchTile * kHeadDim * sizeof(bfloat16_t));
        // Quant 批量：rotated / thr / idxf / tmp / primary / residual
        pipe_->InitBuffer(unitBuf_, kQuantBatch * kHeadDim * sizeof(float));
        pipe_->InitBuffer(outBuf_, kQuantBatch * kHeadDim * sizeof(float));
        pipe_->InitBuffer(mulTmpBuf_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(reduceTmpBuf_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(reduceDstBuf_, 8 * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, kQuantBatch * kHeadDim * sizeof(float));
        pipe_->InitBuffer(idxfBuf_, kQuantBatch * kHeadDim * sizeof(float));
        pipe_->InitBuffer(thrBuf_, kQuantBatch * kHeadDim * sizeof(float));
        pipe_->InitBuffer(primaryBuf_, kQuantBatch * kHeadDim * sizeof(float));
        pipe_->InitBuffer(wgtBuf_, 8 * sizeof(float));
        // Compare mask：32*128=4096 elem → 512B，留余量
        pipe_->InitBuffer(maskBuf_, 1024 * sizeof(uint8_t));
        pipe_->InitBuffer(mask2Buf_, 1024 * sizeof(uint8_t));
        pipe_->InitBuffer(idxU8Buf_, kQuantBatch * 64 * sizeof(uint8_t));
        pipe_->InitBuffer(qjlU8Buf_, kQuantBatch * 16 * sizeof(uint8_t));
        pipe_->InitBuffer(bf16ScalBuf_, kPrefetchTile * sizeof(bfloat16_t));
        pipe_->InitBuffer(scalFp32Buf_, kPrefetchTile * sizeof(float));
        pipe_->InitBuffer(gammaTileBuf_, kPrefetchTile * sizeof(float));
        // 保留与绿基线相同的 UB 占位（曾用于 PrefetchNorms；去掉后大 case 静默偏）
        pipe_->InitBuffer(accTileBuf_, kNormTile * 64 * sizeof(float));
        pipe_->InitBuffer(sumsBuf_, kNormTile * sizeof(float));
        pipe_->InitBuffer(onesBuf_, kNormTile * sizeof(float));
        pipe_->InitBuffer(invBuf_, kNormCap * sizeof(float));
        pipe_->InitBuffer(normBfCoreBuf_, kNormCap * sizeof(bfloat16_t));
        // Gather 质心查表（正本：turboquant_kv_decompress）
        pipe_->InitBuffer(centBuf_, 64 * sizeof(float));
        pipe_->InitBuffer(idxU32Buf_, kQuantBatch * kHeadDim * sizeof(uint32_t));
        // Quant 双缓冲：按 batch 行预取
        pipe_->InitBuffer(rotInQue_, 2, kQuantBatch * kHeadDim * sizeof(float));
        pipe_->InitBuffer(idxOutQue_, 2, kQuantBatch * 64 * sizeof(uint8_t));
    }

    __aicore__ inline void PrefetchUnits()
    {
        uint32_t blk = GetBlockIdx();
        uint32_t rowStart = blk * tiling_.coreRows;
        if (rowStart >= tiling_.totalRows) {
            return;
        }
        uint32_t rowEnd = rowStart + tiling_.coreRows;
        if (rowEnd > tiling_.totalRows) {
            rowEnd = tiling_.totalRows;
        }
        LocalTensor<bfloat16_t> kvBf16 = kvBf16Buf_.Get<bfloat16_t>();
        LocalTensor<float> xLocal = xFp32Buf_.Get<float>();
        for (uint32_t r0 = rowStart; r0 < rowEnd; r0 += kPrefetchTile) {
            uint32_t tc = kPrefetchTile;
            if (r0 + tc > rowEnd) {
                tc = rowEnd - r0;
            }
            const int64_t base = static_cast<int64_t>(r0) * kHeadDim;
            const uint32_t nElem = tc * static_cast<uint32_t>(kHeadDim);
            {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(nElem * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, bfloat16_t(0)};
                DataCopyPad(kvBf16, kvGm_[base], copyParams, padParams);
            }
            PipeBarrier<PIPE_ALL>();
            Cast(xLocal, kvBf16, RoundMode::CAST_NONE, static_cast<int32_t>(nElem));
            PipeBarrier<PIPE_ALL>();
            {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(nElem * sizeof(float)), 0, 0, 0};
                DataCopyPad(unitsGm_[base], xLocal, copyParams);
            }
        }
    }

    __aicore__ inline void Rotate(MM &mm)
    {
        RunGemm(mm, unitsGm_, rotGm_, matOutGm_);
    }

    __aicore__ inline void QuantizeAndResUnit()
    {
        uint32_t blk = GetBlockIdx();
        uint32_t rowStart = blk * tiling_.coreRows;
        if (rowStart >= tiling_.totalRows) {
            return;
        }
        uint32_t rowEnd = rowStart + tiling_.coreRows;
        if (rowEnd > tiling_.totalRows) {
            rowEnd = tiling_.totalRows;
        }

        const int mseBits = static_cast<int>(tiling_.mseBits);
        int nCen = 0;
        const float *cenPtr = KvEncodeCore::Centroids(mseBits, &nCen);
        LocalTensor<float> cent = centBuf_.Get<float>();
        Duplicate(cent, 0.0f, 32);
        PipeBarrier<PIPE_V>();
        for (int c = 0; c < nCen; ++c) {
            cent.SetValue(c, cenPtr[c]);
        }
        for (int c = 0; c < nCen - 1; ++c) {
            cent.SetValue(16 + c, 0.5f * (cenPtr[c] + cenPtr[c + 1]));
        }
        PipeBarrier<PIPE_ALL>();

        for (uint32_t r0 = rowStart; r0 < rowEnd; r0 += kQuantBatch) {
            uint32_t bc = kQuantBatch;
            if (r0 + bc > rowEnd) {
                bc = rowEnd - r0;
            }
            QuantBatch(r0, bc, rowStart, rowEnd - rowStart, cent, cenPtr, nCen, mseBits);
        }
    }

    __aicore__ inline void Project(MM &mm)
    {
        RunGemm(mm, unitsGm_, qjlGm_, matOutGm_);
    }

    __aicore__ inline void PackQjlSigns()
    {
        uint32_t blk = GetBlockIdx();
        uint32_t rowStart = blk * tiling_.coreRows;
        if (rowStart >= tiling_.totalRows) {
            return;
        }
        uint32_t rowEnd = rowStart + tiling_.coreRows;
        if (rowEnd > tiling_.totalRows) {
            rowEnd = tiling_.totalRows;
        }
        for (uint32_t r0 = rowStart; r0 < rowEnd; r0 += kQuantBatch) {
            uint32_t bc = kQuantBatch;
            if (r0 + bc > rowEnd) {
                bc = rowEnd - r0;
            }
            PackQjlBatch(r0, bc);
        }
    }

private:
    __aicore__ inline void RunGemm(MM &mm, GlobalTensor<float> &aGm, GlobalTensor<float> &bGm, GlobalTensor<float> &cGm)
    {
        uint32_t blk = GetBlockIdx();
        uint32_t rowStart = blk * tiling_.coreRows;
        if (rowStart >= tiling_.totalRows) {
            return;
        }
        uint32_t cur = tiling_.coreRows;
        if (rowStart + cur > tiling_.totalRows) {
            cur = tiling_.totalRows - rowStart;
        }
        for (uint32_t offRows = 0; offRows < cur; offRows += kGemmTile) {
            uint32_t rows = kGemmTile;
            if (offRows + rows > cur) {
                rows = cur - offRows;
            }
            uint64_t off = (static_cast<uint64_t>(rowStart) + offRows) * kHeadDim;
            mm.SetTensorA(aGm[off]);
            mm.SetTensorB(bGm, true);
            mm.SetTail(static_cast<int>(rows), kHeadDimI32, kHeadDimI32);
            mm.IterateAll(cGm[off], 0, false, true);
        }
        mm.End();
    }

    __aicore__ inline float DotSelf(LocalTensor<float> &x)
    {
        LocalTensor<float> mulTmp = mulTmpBuf_.Get<float>();
        LocalTensor<float> reduceTmp = reduceTmpBuf_.Get<float>();
        LocalTensor<float> reduceDst = reduceDstBuf_.Get<float>();
        Mul(mulTmp, x, x, kHeadDimI32);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceDst, mulTmp, reduceTmp, kHeadDimI32);
        PipeBarrier<PIPE_V>();
        Sqrt(reduceDst, reduceDst, 8);
        PipeBarrier<PIPE_ALL>();
        return reduceDst.GetValue(0);
    }

    __aicore__ inline void StoreFloatAsBf16(GlobalTensor<bfloat16_t> &dst, int64_t idx, float v)
    {
        LocalTensor<float> fLocal = scalFp32Buf_.Get<float>();
        LocalTensor<bfloat16_t> bLocal = bf16ScalBuf_.Get<bfloat16_t>();
        fLocal.SetValue(0, v);
        PipeBarrier<PIPE_ALL>();
        Cast(bLocal, fLocal, RoundMode::CAST_RINT, 1);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(dst[idx], bLocal, copyParams);
    }

    // PackBits LSB-first（bit2/3/4 特化；禁 Gather×8 版——曾更慢）
    __aicore__ inline void PackFromIdxI32(LocalTensor<int32_t> &idxI, int bitWidth, LocalTensor<uint8_t> &out,
                                          int outBytes)
    {
        const int groups = (kHeadDimI32 + 7) / 8;
        if (bitWidth == 2) {
            for (int g = 0; g < groups; ++g) {
                const int base = g * 8;
                int word = idxI.GetValue(base);
                word |= (idxI.GetValue(base + 1) << 2);
                word |= (idxI.GetValue(base + 2) << 4);
                word |= (idxI.GetValue(base + 3) << 6);
                word |= (idxI.GetValue(base + 4) << 8);
                word |= (idxI.GetValue(base + 5) << 10);
                word |= (idxI.GetValue(base + 6) << 12);
                word |= (idxI.GetValue(base + 7) << 14);
                const int o = g * 2;
                out.SetValue(o, static_cast<uint8_t>(word & 0xFF));
                out.SetValue(o + 1, static_cast<uint8_t>((word >> 8) & 0xFF));
            }
            return;
        }
        if (bitWidth == 3) {
            for (int g = 0; g < groups; ++g) {
                const int base = g * 8;
                int word = idxI.GetValue(base);
                word |= (idxI.GetValue(base + 1) << 3);
                word |= (idxI.GetValue(base + 2) << 6);
                word |= (idxI.GetValue(base + 3) << 9);
                word |= (idxI.GetValue(base + 4) << 12);
                word |= (idxI.GetValue(base + 5) << 15);
                word |= (idxI.GetValue(base + 6) << 18);
                word |= (idxI.GetValue(base + 7) << 21);
                const int o = g * 3;
                out.SetValue(o, static_cast<uint8_t>(word & 0xFF));
                out.SetValue(o + 1, static_cast<uint8_t>((word >> 8) & 0xFF));
                out.SetValue(o + 2, static_cast<uint8_t>((word >> 16) & 0xFF));
            }
            return;
        }
        if (bitWidth == 4) {
            for (int g = 0; g < groups; ++g) {
                const int base = g * 8;
                int word = idxI.GetValue(base);
                word |= (idxI.GetValue(base + 1) << 4);
                word |= (idxI.GetValue(base + 2) << 8);
                word |= (idxI.GetValue(base + 3) << 12);
                word |= (idxI.GetValue(base + 4) << 16);
                word |= (idxI.GetValue(base + 5) << 20);
                word |= (idxI.GetValue(base + 6) << 24);
                word |= (idxI.GetValue(base + 7) << 28);
                const int o = g * 4;
                out.SetValue(o, static_cast<uint8_t>(word & 0xFF));
                out.SetValue(o + 1, static_cast<uint8_t>((word >> 8) & 0xFF));
                out.SetValue(o + 2, static_cast<uint8_t>((word >> 16) & 0xFF));
                out.SetValue(o + 3, static_cast<uint8_t>((word >> 24) & 0xFF));
            }
            return;
        }
        for (int i = 0; i < outBytes; ++i) {
            out.SetValue(i, static_cast<uint8_t>(0));
        }
        for (int g = 0; g < groups; ++g) {
            int word = 0;
            for (int lane = 0; lane < 8; ++lane) {
                const int v = idxI.GetValue(g * 8 + lane);
                word |= (v << (lane * bitWidth));
            }
            if (bitWidth == 1) {
                out.SetValue(g, static_cast<uint8_t>(word & 0xFF));
            } else {
                for (int b = 0; b < bitWidth; ++b) {
                    out.SetValue(g * bitWidth + b, static_cast<uint8_t>((word >> (8 * b)) & 0xFF));
                }
            }
        }
    }

    __aicore__ inline void FlushScalTile(GlobalTensor<bfloat16_t> &dstGm, LocalTensor<float> &tileF, uint32_t absBase,
                                         uint32_t count)
    {
        LocalTensor<bfloat16_t> tileB = bf16ScalBuf_.Get<bfloat16_t>();
        PipeBarrier<PIPE_ALL>();
        Cast(tileB, tileF, RoundMode::CAST_RINT, static_cast<int32_t>(count));
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams np{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(dstGm[static_cast<int64_t>(absBase)], tileB, np);
    }

    // 批量中点二分（nElem=bc*128）；语义 ≡ 按行线性/二分
    __aicore__ inline void BinaryQuantBatch(LocalTensor<float> &rotated, LocalTensor<float> &idxf,
                                            LocalTensor<float> &thr, LocalTensor<float> &thrB, LocalTensor<float> &tmp,
                                            LocalTensor<uint8_t> &mask, LocalTensor<uint8_t> &mask2, const float *cen,
                                            int nCen, int32_t nElem)
    {
        Duplicate(idxf, 0.0f, nElem);
        PipeBarrier<PIPE_V>();
        if (nCen == 8) {
            const float t0 = 0.5f * (cen[0] + cen[1]);
            const float t1 = 0.5f * (cen[1] + cen[2]);
            const float t2 = 0.5f * (cen[2] + cen[3]);
            const float t3 = 0.5f * (cen[3] + cen[4]);
            const float t4 = 0.5f * (cen[4] + cen[5]);
            const float t5 = 0.5f * (cen[5] + cen[6]);
            const float t6 = 0.5f * (cen[6] + cen[7]);
            Duplicate(thr, t3, nElem);
            PipeBarrier<PIPE_V>();
            Compare(mask, rotated, thr, CMPMODE::GT, nElem);
            PipeBarrier<PIPE_V>();
            Duplicate(tmp, 4.0f, nElem);
            Select(idxf, mask, tmp, idxf, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
            Duplicate(thr, t1, nElem);
            Duplicate(thrB, t5, nElem);
            PipeBarrier<PIPE_V>();
            Select(thr, mask, thrB, thr, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
            PipeBarrier<PIPE_V>();
            Compare(mask2, rotated, thr, CMPMODE::GT, nElem);
            PipeBarrier<PIPE_V>();
            Adds(tmp, idxf, 2.0f, nElem);
            Select(idxf, mask2, tmp, idxf, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
            Duplicate(thr, t0, nElem);
            Duplicate(thrB, t2, nElem);
            PipeBarrier<PIPE_V>();
            Select(thr, mask2, thrB, thr, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
            Duplicate(tmp, t4, nElem);
            Duplicate(thrB, t6, nElem);
            PipeBarrier<PIPE_V>();
            Select(thrB, mask2, thrB, tmp, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
            PipeBarrier<PIPE_V>();
            Select(thr, mask, thrB, thr, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
            PipeBarrier<PIPE_V>();
            Compare(mask, rotated, thr, CMPMODE::GT, nElem);
            PipeBarrier<PIPE_V>();
            Adds(tmp, idxf, 1.0f, nElem);
            Select(idxf, mask, tmp, idxf, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
            PipeBarrier<PIPE_V>();
        } else if (nCen == 4) {
            const float t0 = 0.5f * (cen[0] + cen[1]);
            const float t1 = 0.5f * (cen[1] + cen[2]);
            const float t2 = 0.5f * (cen[2] + cen[3]);
            Duplicate(thr, t1, nElem);
            PipeBarrier<PIPE_V>();
            Compare(mask, rotated, thr, CMPMODE::GT, nElem);
            Duplicate(tmp, 2.0f, nElem);
            PipeBarrier<PIPE_V>();
            Select(idxf, mask, tmp, idxf, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
            Duplicate(thr, t0, nElem);
            Duplicate(thrB, t2, nElem);
            PipeBarrier<PIPE_V>();
            Select(thr, mask, thrB, thr, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
            PipeBarrier<PIPE_V>();
            Compare(mask, rotated, thr, CMPMODE::GT, nElem);
            Adds(tmp, idxf, 1.0f, nElem);
            PipeBarrier<PIPE_V>();
            Select(idxf, mask, tmp, idxf, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
        } else {
            // midpoints laid contiguously at cent[16..16+nCen-2]; thrTable aliases that span.
            // AscendC Gather srcOffset is in *bytes* (not elements); float → ×sizeof(float)=4.
            // e.g. nCen=16 first round: elemIdx=(0+8-1)=7 → byteOff=28 → thrTable[7]=cent[23], in-range.
            LocalTensor<float> thrTable = centBuf_.GetWithOffset<float>(16, 16 * sizeof(float));
            for (int bit = nCen >> 1; bit >= 1; bit >>= 1) {
                Adds(tmp, idxf, static_cast<float>(bit - 1), nElem);
                PipeBarrier<PIPE_V>();
                Muls(tmp, tmp, 4.0f, nElem);
                PipeBarrier<PIPE_V>();
                {
                    LocalTensor<int32_t> offI = idxU32Buf_.Get<int32_t>();
                    Cast(offI, tmp, RoundMode::CAST_ROUND, nElem);
                }
                PipeBarrier<PIPE_V>();
                {
                    LocalTensor<uint32_t> offU = idxU32Buf_.Get<uint32_t>();
                    Gather(thr, thrTable, offU, static_cast<uint32_t>(0), nElem);
                }
                PipeBarrier<PIPE_V>();
                Compare(mask, rotated, thr, CMPMODE::GT, nElem);
                PipeBarrier<PIPE_V>();
                Adds(tmp, idxf, static_cast<float>(bit), nElem);
                PipeBarrier<PIPE_V>();
                Select(idxf, mask, tmp, idxf, SELMODE::VSEL_TENSOR_TENSOR_MODE, nElem);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void QuantBatch(uint32_t absBase, uint32_t bc, uint32_t rowStart, uint32_t numRows,
                                      LocalTensor<float> &cent, const float *cen, int nCen, int mseBits)
    {
        const int32_t nElem = static_cast<int32_t>(bc * static_cast<uint32_t>(kHeadDimI32));
        LocalTensor<float> hx = unitBuf_.Get<float>();
        {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(nElem * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            DataCopyPad(hx, matOutGm_[static_cast<int64_t>(absBase) * kHeadDim], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();

        // 按行单位化：一次 Mul 全 batch 平方
        float norms[kQuantBatch];
        {
            LocalTensor<float> sq = primaryBuf_.Get<float>();
            Mul(sq, hx, hx, nElem);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceTmp = reduceTmpBuf_.Get<float>();
            LocalTensor<float> reduceDst = reduceDstBuf_.Get<float>();
            for (uint32_t i = 0; i < bc; ++i) {
                LocalTensor<float> rowSq = primaryBuf_.GetWithOffset<float>(kHeadDimI32, i * kHeadDim * sizeof(float));
                ReduceSum(reduceDst, rowSq, reduceTmp, kHeadDimI32);
                PipeBarrier<PIPE_V>();
                Sqrt(reduceDst, reduceDst, 8);
                PipeBarrier<PIPE_ALL>();
                norms[i] = reduceDst.GetValue(0);
                LocalTensor<float> row = unitBuf_.GetWithOffset<float>(kHeadDimI32, i * kHeadDim * sizeof(float));
                if (norms[i] > 0.f) {
                    Muls(row, row, 1.f / (norms[i] < 1e-30f ? 1e-30f : norms[i]), kHeadDimI32);
                } else {
                    Duplicate(row, 0.0f, kHeadDimI32);
                }
                PipeBarrier<PIPE_V>();
            }
        }

        LocalTensor<float> idxf = idxfBuf_.Get<float>();
        LocalTensor<float> thr = thrBuf_.Get<float>();
        LocalTensor<float> thrB = outBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> mask2 = mask2Buf_.Get<uint8_t>();
        BinaryQuantBatch(hx, idxf, thr, thrB, tmp, mask, mask2, cen, nCen, nElem);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<int32_t> idxI = idxU32Buf_.Get<int32_t>();
        Cast(idxI, idxf, RoundMode::CAST_RINT, nElem);
        PipeBarrier<PIPE_V>();

        const int64_t idxBytes = tiling_.idxBytes;
        LocalTensor<uint8_t> idxPacked = idxU8Buf_.Get<uint8_t>();
        for (uint32_t i = 0; i < bc; ++i) {
            LocalTensor<int32_t> idxRow =
                idxU32Buf_.GetWithOffset<int32_t>(kHeadDimI32, i * kHeadDim * sizeof(int32_t));
            LocalTensor<uint8_t> idxOut = idxU8Buf_.GetWithOffset<uint8_t>(
                static_cast<int32_t>(idxBytes), static_cast<uint32_t>(i) * static_cast<uint32_t>(idxBytes));
            PackFromIdxI32(idxRow, mseBits, idxOut, static_cast<int>(idxBytes));
        }
        PipeBarrier<PIPE_ALL>();
        {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(bc * static_cast<uint32_t>(idxBytes)), 0, 0, 0};
            DataCopyPad(idxGm_[static_cast<int64_t>(absBase) * idxBytes], idxPacked, copyParams);
        }

        ShiftLeft(idxI, idxI, static_cast<int32_t>(2), nElem);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> primary = primaryBuf_.Get<float>();
        {
            LocalTensor<uint32_t> idxU = idxU32Buf_.Get<uint32_t>();
            Gather(primary, cent, idxU, static_cast<uint32_t>(0), nElem);
        }
        PipeBarrier<PIPE_ALL>();

        LocalTensor<float> residual = outBuf_.Get<float>();
        Sub(residual, hx, primary, nElem);
        PipeBarrier<PIPE_V>();

        // residual 范数：一次 Mul 全 batch
        {
            LocalTensor<float> sq = primaryBuf_.Get<float>();
            Mul(sq, residual, residual, nElem);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> normTile = scalFp32Buf_.Get<float>();
        LocalTensor<float> gammaTile = gammaTileBuf_.Get<float>();
        LocalTensor<float> unitsBat = thrBuf_.Get<float>();
        LocalTensor<float> reduceTmp = reduceTmpBuf_.Get<float>();
        LocalTensor<float> reduceDst = reduceDstBuf_.Get<float>();
        for (uint32_t i = 0; i < bc; ++i) {
            const uint32_t absRow = absBase + i;
            const uint32_t li = absRow - rowStart;
            const uint32_t pos = li % kPrefetchTile;
            const uint32_t tileBaseLi = li - pos;

            LocalTensor<float> rowSq = primaryBuf_.GetWithOffset<float>(kHeadDimI32, i * kHeadDim * sizeof(float));
            ReduceSum(reduceDst, rowSq, reduceTmp, kHeadDimI32);
            PipeBarrier<PIPE_V>();
            Sqrt(reduceDst, reduceDst, 8);
            PipeBarrier<PIPE_ALL>();
            const float resNorm = reduceDst.GetValue(0);
            const float gamma = norms[i] * resNorm;
            LocalTensor<float> resRow = outBuf_.GetWithOffset<float>(kHeadDimI32, i * kHeadDim * sizeof(float));
            LocalTensor<float> resUnit = thrBuf_.GetWithOffset<float>(kHeadDimI32, i * kHeadDim * sizeof(float));
            if (resNorm > 0.f) {
                Muls(resUnit, resRow, 1.f / (resNorm < 1e-30f ? 1e-30f : resNorm), kHeadDimI32);
            } else {
                Duplicate(resUnit, 0.0f, kHeadDimI32);
            }
            PipeBarrier<PIPE_V>();
            normTile.SetValue(static_cast<int32_t>(pos), norms[i]);
            gammaTile.SetValue(static_cast<int32_t>(pos), gamma);

            const bool flush = (pos + 1U == kPrefetchTile) || (li + 1U == numRows);
            if (flush) {
                PipeBarrier<PIPE_ALL>();
                const uint32_t count = pos + 1U;
                const uint32_t absFlush = rowStart + tileBaseLi;
                FlushScalTile(normGm_, normTile, absFlush, count);
                FlushScalTile(gammaGm_, gammaTile, absFlush, count);
            }
        }
        PipeBarrier<PIPE_ALL>();
        {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(nElem * sizeof(float)), 0, 0, 0};
            DataCopyPad(unitsGm_[static_cast<int64_t>(absBase) * kHeadDim], unitsBat, copyParams);
        }
    }

    __aicore__ inline void PackQjlBatch(uint32_t absBase, uint32_t bc)
    {
        const int32_t nElem = static_cast<int32_t>(bc * static_cast<uint32_t>(kHeadDimI32));
        LocalTensor<float> projected = unitBuf_.Get<float>();
        {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(nElem * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            DataCopyPad(projected, matOutGm_[static_cast<int64_t>(absBase) * kHeadDim], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();

        LocalTensor<float> thr = thrBuf_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        Duplicate(thr, 0.0f, nElem);
        PipeBarrier<PIPE_V>();
        Compare(mask, projected, thr, CMPMODE::GE, nElem);
        PipeBarrier<PIPE_V>();
        {
            const int64_t qjlBytes = tiling_.qjlBytes;
            // 行连续且每行 16B：一次搬 bc 行
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(bc * static_cast<uint32_t>(qjlBytes)), 0, 0, 0};
            DataCopyPad(qjlOutGm_[static_cast<int64_t>(absBase) * qjlBytes], mask, copyParams);
        }
    }

    TilingView tiling_{};
    TPipe *pipe_{nullptr};

    GlobalTensor<bfloat16_t> kvGm_;
    GlobalTensor<float> rotGm_;
    GlobalTensor<float> qjlGm_;
    GlobalTensor<uint8_t> idxGm_;
    GlobalTensor<uint8_t> qjlOutGm_;
    GlobalTensor<bfloat16_t> normGm_;
    GlobalTensor<bfloat16_t> gammaGm_;
    GlobalTensor<float> unitsGm_;
    GlobalTensor<float> matOutGm_;

    TBuf<TPosition::VECCALC> xFp32Buf_;
    TBuf<TPosition::VECCALC> kvBf16Buf_;
    TBuf<TPosition::VECCALC> unitBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> mulTmpBuf_;
    TBuf<TPosition::VECCALC> reduceTmpBuf_;
    TBuf<TPosition::VECCALC> reduceDstBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> idxfBuf_;
    TBuf<TPosition::VECCALC> thrBuf_;
    TBuf<TPosition::VECCALC> primaryBuf_;
    TBuf<TPosition::VECCALC> wgtBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> mask2Buf_;
    TBuf<TPosition::VECCALC> idxU8Buf_;
    TBuf<TPosition::VECCALC> qjlU8Buf_;
    TBuf<TPosition::VECCALC> bf16ScalBuf_;
    TBuf<TPosition::VECCALC> scalFp32Buf_;
    TBuf<TPosition::VECCALC> gammaTileBuf_;
    TBuf<TPosition::VECCALC> accTileBuf_;
    TBuf<TPosition::VECCALC> sumsBuf_;
    TBuf<TPosition::VECCALC> onesBuf_;
    TBuf<TPosition::VECCALC> invBuf_;
    TBuf<TPosition::VECCALC> normBfCoreBuf_;
    TBuf<TPosition::VECCALC> centBuf_;
    TBuf<TPosition::VECCALC> idxU32Buf_;
    TQue<TPosition::VECIN, 2> rotInQue_;
    TQue<TPosition::VECOUT, 2> idxOutQue_;
};

} // namespace KvCacheTurboQuantKernel

#endif // KERNEL_KV_CACHE_TURBO_QUANT_H_
