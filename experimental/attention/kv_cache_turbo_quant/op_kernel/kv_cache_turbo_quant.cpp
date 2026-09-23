/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Vector stages of KvCacheTurboQuant.
 * head_dim = qjl_dim = 128, mse_bits in {2, 3, 4}.
 *
 * cast:     BF16 kv -> FP32. The host then matches torch with aclnn vector_norm and div.
 * quantize: Lloyd-Max indices and the raw residual y - y_hat
 * store:    FP32 norm and gamma -> BF16, round-to-nearest-even
 * sign:     1-bit QJL pack from an already projected residual
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int32_t kDim = 128;
constexpr int32_t kTile = 64;
constexpr float kPow2[29] = {1.f,         2.f,         4.f,         8.f,        16.f,       32.f,        64.f,
                             128.f,       256.f,       512.f,       1024.f,     2048.f,     4096.f,      8192.f,
                             16384.f,     32768.f,     65536.f,     131072.f,   262144.f,   524288.f,    1048576.f,
                             2097152.f,   4194304.f,   8388608.f,   16777216.f, 33554432.f, 67108864.f,  134217728.f,
                             268435456.f};

constexpr float kCent2[4] = {-1.335033178e-01f, -4.002048075e-02f, 4.002048075e-02f, 1.335033178e-01f};
constexpr float kBnd2[3] = {-8.676189929e-02f, 0.0f, 8.676189929e-02f};
constexpr float kCent3[8] = {-1.902069300e-01f, -1.187859178e-01f, -6.682205945e-02f, -2.166347019e-02f,
                             2.166347019e-02f,  6.682205945e-02f,  1.187859178e-01f,  1.902069300e-01f};
constexpr float kBnd3[7] = {-1.544964314e-01f, -9.280398488e-02f, -4.424276575e-02f, 0.0f,
                            4.424276575e-02f,  9.280398488e-02f,  1.544964314e-01f};
constexpr float kCent4[16] = {-2.414890379e-01f, -1.828317791e-01f, -1.429702938e-01f, -1.109927073e-01f,
                              -8.325428516e-02f, -5.802082643e-02f, -3.428063914e-02f, -1.134236995e-02f,
                              1.134236995e-02f,  3.428063914e-02f,  5.802082643e-02f,  8.325428516e-02f,
                              1.109927073e-01f,  1.429702938e-01f,  1.828317791e-01f,  2.414890379e-01f};
constexpr float kBnd4[15] = {-2.121604085e-01f, -1.629010439e-01f, -1.269814968e-01f, -9.712349623e-02f,
                             -7.063755393e-02f, -4.615073279e-02f, -2.281150408e-02f, 0.0f,
                             2.281150408e-02f,  4.615073279e-02f,  7.063755393e-02f,  9.712349623e-02f,
                             1.269814968e-01f,  1.629010439e-01f,  2.121604085e-01f};

__aicore__ inline void RowRange(uint32_t rowCount, uint32_t coreNum, uint32_t blockIdx, uint32_t &rowBegin,
                                uint32_t &rowEnd)
{
    const uint32_t base = rowCount / coreNum;
    const uint32_t rem = rowCount % coreNum;
    const uint32_t mine = base + (blockIdx < rem ? 1u : 0u);
    rowBegin = blockIdx * base + (blockIdx < rem ? blockIdx : rem);
    rowEnd = rowBegin + mine;
}

__aicore__ inline void LoadRows(const GlobalTensor<float> &src, const LocalTensor<float> &dst, uint32_t row0,
                                int32_t nRows)
{
    for (int32_t i = 0; i < nRows; ++i) {
        DataCopy(dst[i * kDim], src[(row0 + static_cast<uint32_t>(i)) * kDim], kDim);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void StoreRows(const GlobalTensor<float> &dst, const LocalTensor<float> &src, uint32_t row0,
                                 int32_t nRows)
{
    PipeBarrier<PIPE_ALL>();
    for (int32_t i = 0; i < nRows; ++i) {
        DataCopy(dst[(row0 + static_cast<uint32_t>(i)) * kDim], src[i * kDim], kDim);
    }
    PipeBarrier<PIPE_ALL>();
}

} // namespace

extern "C" __global__ __aicore__ void kvtq_cast_fp32(GM_ADDR kv, GM_ADDR fp32, uint32_t rowCount, uint32_t coreNum)
{
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= coreNum) {
        return;
    }
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    RowRange(rowCount, coreNum, blockIdx, rowBegin, rowEnd);
    if (rowBegin >= rowEnd) {
        return;
    }

    GlobalTensor<bfloat16_t> kvGm;
    GlobalTensor<float> fpGm;
    kvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(kv), rowCount * kDim);
    fpGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(fp32), rowCount * kDim);

    TPipe pipe;
    TBuf<TPosition::VECCALC> xBfBuf, xBuf;
    pipe.InitBuffer(xBfBuf, kTile * kDim * sizeof(bfloat16_t));
    pipe.InitBuffer(xBuf, kTile * kDim * sizeof(float));
    LocalTensor<bfloat16_t> xbf = xBfBuf.Get<bfloat16_t>();
    LocalTensor<float> x = xBuf.Get<float>();

    for (uint32_t base = rowBegin; base < rowEnd; base += kTile) {
        const uint32_t remain = rowEnd - base;
        const int32_t nRows =
            static_cast<int32_t>(remain < static_cast<uint32_t>(kTile) ? remain : static_cast<uint32_t>(kTile));
        for (int32_t i = 0; i < nRows; ++i) {
            DataCopy(xbf[i * kDim], kvGm[(base + static_cast<uint32_t>(i)) * kDim], kDim);
        }
        PipeBarrier<PIPE_ALL>();
        Cast(x, xbf, RoundMode::CAST_NONE, nRows * kDim);
        StoreRows(fpGm, x, base, nRows);
    }
}

extern "C" __global__ __aicore__ void kvtq_quantize(GM_ADDR rotated, GM_ADDR quantIdx, GM_ADDR residual,
                                                   uint32_t rowCount, uint32_t mseBits, uint32_t coreNum)
{
    const uint32_t blockIdx = GetBlockIdx();
    const int32_t bits = static_cast<int32_t>(mseBits);
    if (blockIdx >= coreNum || bits < 2 || bits > 4) {
        return;
    }
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    RowRange(rowCount, coreNum, blockIdx, rowBegin, rowEnd);
    if (rowBegin >= rowEnd) {
        return;
    }

    const float *cent = bits == 2 ? kCent2 : (bits == 3 ? kCent3 : kCent4);
    const float *bnd = bits == 2 ? kBnd2 : (bits == 3 ? kBnd3 : kBnd4);
    const int32_t nBnd = (1 << bits) - 1;
    const int32_t idxBytes = kDim * bits / 8;

    GlobalTensor<float> yGm;
    GlobalTensor<uint8_t> idxGm;
    GlobalTensor<float> rGm;
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rotated), rowCount * kDim);
    idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantIdx), rowCount * idxBytes);
    rGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(residual), rowCount * kDim);

    TPipe pipe;
    TBuf<TPosition::VECCALC> yBuf, idxBuf, yhatBuf, tmpBuf, oneBuf, wordBuf, wordHiBuf, iBuf, iHiBuf, maskBuf, packBuf,
        wLoBuf, wHiBuf;
    pipe.InitBuffer(yBuf, kTile * kDim * sizeof(float));
    pipe.InitBuffer(idxBuf, kTile * kDim * sizeof(float));
    pipe.InitBuffer(yhatBuf, kTile * kDim * sizeof(float));
    pipe.InitBuffer(tmpBuf, kTile * kDim * sizeof(float));
    pipe.InitBuffer(oneBuf, kTile * kDim * sizeof(float));
    pipe.InitBuffer(wordBuf, kTile * 16 * sizeof(float));
    pipe.InitBuffer(wordHiBuf, kTile * 16 * sizeof(float));
    pipe.InitBuffer(iBuf, kTile * 16 * sizeof(int32_t));
    pipe.InitBuffer(iHiBuf, kTile * 16 * sizeof(int32_t));
    pipe.InitBuffer(maskBuf, kTile * kDim / 8 + 32);
    pipe.InitBuffer(packBuf, kTile * 64 + 32);
    pipe.InitBuffer(wLoBuf, kDim * sizeof(float));
    pipe.InitBuffer(wHiBuf, kDim * sizeof(float));

    LocalTensor<float> y = yBuf.Get<float>();
    LocalTensor<float> idx = idxBuf.Get<float>();
    LocalTensor<float> yhat = yhatBuf.Get<float>();
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    LocalTensor<float> ones = oneBuf.Get<float>();
    LocalTensor<float> words = wordBuf.Get<float>();
    LocalTensor<float> wordsHi = wordHiBuf.Get<float>();
    LocalTensor<int32_t> wordsI = iBuf.Get<int32_t>();
    LocalTensor<int32_t> wordsHiI = iHiBuf.Get<int32_t>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<uint8_t> packed = packBuf.Get<uint8_t>();
    LocalTensor<float> wLo = wLoBuf.Get<float>();
    LocalTensor<float> wHi = wHiBuf.Get<float>();

    for (int32_t i = 0; i < kDim; ++i) {
        const int32_t lane = i & 7;
        if (bits == 4) {
            wLo.SetValue(i, lane < 4 ? kPow2[lane * 4] : 0.0f);
            wHi.SetValue(i, lane >= 4 ? kPow2[(lane - 4) * 4] : 0.0f);
        } else {
            wLo.SetValue(i, kPow2[lane * bits]);
            wHi.SetValue(i, 0.0f);
        }
    }
    PipeBarrier<PIPE_ALL>();

    for (uint32_t base = rowBegin; base < rowEnd; base += static_cast<uint32_t>(kTile)) {
        const uint32_t remain = rowEnd - base;
        const int32_t nRows = static_cast<int32_t>(remain < static_cast<uint32_t>(kTile) ? remain : static_cast<uint32_t>(kTile));
        const int32_t count = nRows * kDim;
        LoadRows(yGm, y, base, nRows);
        Duplicate(idx, 0.0f, count);
        Duplicate(ones, 1.0f, count);
        PipeBarrier<PIPE_V>();
        for (int32_t b = 0; b < nBnd; ++b) {
            Compares(mask, y, bnd[b], CMPMODE::GT, count);
            Select(tmp, mask, ones, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, static_cast<uint32_t>(count));
            Add(idx, idx, tmp, count);
        }
        PipeBarrier<PIPE_V>();
        // Table lookup, not a sum of centroid gaps. The summed gaps differ from the
        // table by a few ulps and can flip a QJL sign that sits on zero.
        Duplicate(yhat, cent[0], count);
        const int32_t nLevel = nBnd + 1;
        for (int32_t level = 1; level < nLevel; ++level) {
            Compares(mask, idx, static_cast<float>(level), CMPMODE::EQ, count);
            Duplicate(tmp, cent[level], count);
            Select(ones, mask, tmp, yhat, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            Adds(yhat, ones, 0.0f, count);
            PipeBarrier<PIPE_V>();
        }
        Duplicate(ones, 1.0f, count);
        PipeBarrier<PIPE_V>();

        set_mask_norm();
        const BinaryRepeatParams mulRp(1, 1, 1, 16, 16, 0);
        Mul(tmp, idx, wLo, 64, static_cast<uint8_t>(nRows), mulRp);
        Mul(tmp[64], idx[64], wLo[64], 64, static_cast<uint8_t>(nRows), mulRp);
        PipeBarrier<PIPE_V>();
        BlockReduceSum(words, tmp, nRows * 2, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        const int32_t nWords = nRows * 16;
        const int32_t nCast = (nWords + 7) & ~7;
        Cast(wordsI, words, RoundMode::CAST_RINT, nCast);
        if (bits == 4) {
            Mul(tmp, idx, wHi, 64, static_cast<uint8_t>(nRows), mulRp);
            Mul(tmp[64], idx[64], wHi[64], 64, static_cast<uint8_t>(nRows), mulRp);
            PipeBarrier<PIPE_V>();
            BlockReduceSum(wordsHi, tmp, nRows * 2, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            Cast(wordsHiI, wordsHi, RoundMode::CAST_RINT, nCast);
        }
        PipeBarrier<PIPE_ALL>();
        for (int32_t w = 0; w < nWords; ++w) {
            uint32_t word = static_cast<uint32_t>(wordsI.GetValue(w));
            if (bits == 4) {
                word += static_cast<uint32_t>(wordsHiI.GetValue(w)) << 16;
            }
            for (int32_t b = 0; b < bits; ++b) {
                packed.SetValue(w * bits + b, static_cast<uint8_t>((word >> (8 * b)) & 0xffu));
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams idxParams(1, static_cast<uint32_t>(nRows * idxBytes), 0, 0, 0);
        DataCopyPad(idxGm[base * static_cast<uint32_t>(idxBytes)], packed, idxParams);
        PipeBarrier<PIPE_ALL>();

        Sub(y, y, yhat, count);
        StoreRows(rGm, y, base, nRows);
    }
}

extern "C" __global__ __aicore__ void kvtq_store_bf16(GM_ADDR normFp, GM_ADDR gammaFp, GM_ADDR normBf, GM_ADDR gammaBf,
                                                     uint32_t rowCount, uint32_t coreNum)
{
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= coreNum) {
        return;
    }
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    RowRange(rowCount, coreNum, blockIdx, rowBegin, rowEnd);
    if (rowBegin >= rowEnd) {
        return;
    }

    GlobalTensor<float> normFpGm;
    GlobalTensor<float> gammaFpGm;
    GlobalTensor<bfloat16_t> normBfGm;
    GlobalTensor<bfloat16_t> gammaBfGm;
    normFpGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(normFp), rowCount);
    gammaFpGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gammaFp), rowCount);
    normBfGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(normBf), rowCount);
    gammaBfGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gammaBf), rowCount);

    constexpr int32_t kChunk = 64;
    TPipe pipe;
    TBuf<TPosition::VECCALC> fpBuf, bfBuf;
    pipe.InitBuffer(fpBuf, kChunk * 2 * sizeof(float));
    pipe.InitBuffer(bfBuf, kChunk * 2 * sizeof(bfloat16_t));
    LocalTensor<float> fp = fpBuf.Get<float>();
    LocalTensor<bfloat16_t> bf = bfBuf.Get<bfloat16_t>();

    for (uint32_t base = rowBegin; base < rowEnd; base += static_cast<uint32_t>(kChunk)) {
        const uint32_t remain = rowEnd - base;
        const int32_t nRows =
            static_cast<int32_t>(remain < static_cast<uint32_t>(kChunk) ? remain : static_cast<uint32_t>(kChunk));
        DataCopyPad(fp, normFpGm[base], DataCopyExtParams(1, static_cast<uint32_t>(nRows) * sizeof(float), 0, 0, 0),
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(fp[kChunk], gammaFpGm[base],
                    DataCopyExtParams(1, static_cast<uint32_t>(nRows) * sizeof(float), 0, 0, 0),
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < nRows; ++i) {
            bf.SetValue(i, Cast(fp.GetValue(i)));
            bf.SetValue(kChunk + i, Cast(fp.GetValue(kChunk + i)));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(normBfGm[base], bf, DataCopyExtParams(1, static_cast<uint32_t>(nRows) * sizeof(bfloat16_t), 0, 0, 0));
        DataCopyPad(gammaBfGm[base], bf[kChunk],
                    DataCopyExtParams(1, static_cast<uint32_t>(nRows) * sizeof(bfloat16_t), 0, 0, 0));
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ void kvtq_sign(GM_ADDR projected, GM_ADDR quantQjl, uint32_t rowCount, uint32_t coreNum)
{
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= coreNum) {
        return;
    }
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    RowRange(rowCount, coreNum, blockIdx, rowBegin, rowEnd);
    if (rowBegin >= rowEnd) {
        return;
    }

    constexpr int32_t kQjlBytes = kDim / 8;
    GlobalTensor<float> pGm;
    GlobalTensor<uint8_t> qjlGm;
    pGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(projected), rowCount * kDim);
    qjlGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantQjl), rowCount * kQjlBytes);

    TPipe pipe;
    TBuf<TPosition::VECCALC> pBuf, maskBuf;
    pipe.InitBuffer(pBuf, kTile * kDim * sizeof(float));
    pipe.InitBuffer(maskBuf, kTile * kQjlBytes + 32);
    LocalTensor<float> p = pBuf.Get<float>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();

    for (uint32_t base = rowBegin; base < rowEnd; base += static_cast<uint32_t>(kTile)) {
        const uint32_t remain = rowEnd - base;
        const int32_t nRows = static_cast<int32_t>(remain < static_cast<uint32_t>(kTile) ? remain : static_cast<uint32_t>(kTile));
        const int32_t count = nRows * kDim;
        LoadRows(pGm, p, base, nRows);
        Compares(mask, p, 0.0f, CMPMODE::GE, count);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams params(1, static_cast<uint32_t>(nRows * kQjlBytes), 0, 0, 0);
        DataCopyPad(qjlGm[base * kQjlBytes], mask, params);
        PipeBarrier<PIPE_ALL>();
    }
}
