/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TURBO_QUANT_ATTENTION_SCORE_PREFILL_H
#define TURBO_QUANT_ATTENTION_SCORE_PREFILL_H
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "turbo_quant_attention_score_impl.h"

namespace TurboQuant {
// The two FP16 components retain substantially more coefficient precision than
// a single FP16/BF16 conversion; accumulation and the public output stay FP32.
// BF16 queries within FP16 range are exact except for <= 2^-25 near zero.
template <typename NormType>
class PrefillPrepare {
public:
    __aicore__ inline explicit PrefillPrepare(Kernel<NormType, true> &kernel)
        : k_(kernel)
    {}
    __aicore__ inline bool PrepareQueries(GM_ADDR ap)
    {
        if (k_.lens_.GetValue(0) != k_.maxLen_)
            return false;
        GlobalTensor<half> am;
        am.SetGlobalBuffer((__gm__ half *)ap);
        const uint32_t qt = k_.tokens_ / ROWS;
        for (uint32_t task = GetBlockIdx(); task < k_.qHeads_ * qt; task += GetBlockNum() * GetTaskRation()) {
            uint32_t h = task / qt, t = task % qt * ROWS;
            auto raw = k_.a_.template ReinterpretCast<bfloat16_t>();
            auto rr = k_.b_.template ReinterpretCast<bfloat16_t>();
            DataCopyExtParams load{ROWS, DIM * 2, (k_.qHeads_ - 1) * DIM * 2, 0, 0};
            DataCopyPad(raw, k_.q_[(static_cast<uint64_t>(t) * k_.qHeads_ + h) * DIM], load,
                        DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
            DataCopyPad(rr, k_.qr_[(static_cast<uint64_t>(t) * k_.qHeads_ + h) * DIM], load,
                        DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Cast(k_.primary_, raw, RoundMode::CAST_NONE, ELEMS);
            Cast(k_.signs_, rr, RoundMode::CAST_NONE, ELEMS);
            PipeBarrier<PIPE_V>();
            auto halfQ = k_.a_.template ReinterpretCast<half>(),
                 halfR = k_.a_[ELEMS / 2].template ReinterpretCast<half>();
            Cast(halfQ, k_.primary_, RoundMode::CAST_RINT, ELEMS);
            Cast(halfR, k_.signs_, RoundMode::CAST_RINT, ELEMS);
            PipeBarrier<PIPE_V>();
            if (!SafeQueries())
                return false;
            auto packed = k_.b_.template ReinterpretCast<half>();
            for (uint32_t part = 0; part < 4; ++part) {
                DataCopy(packed[part * DIM], part < 2 ? halfQ : halfR,
                         DataCopyParams{ROWS, DIM * 2 / 32, 0, DIM * 6 / 32});
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(am[(static_cast<uint64_t>(h) * k_.tokens_ + t) * 512], packed, ROWS * 512);
            PipeBarrier<PIPE_ALL>();
        }
        return true;
    }
    __aicore__ inline bool PrepareBlock(uint32_t h, uint32_t k, const GlobalTensor<half> &bm,
                                        const LocalTensor<uint32_t> &expanded)
    {
        int32_t physical = k_.table_.GetValue(k / DIM);
        if (physical < 0 || static_cast<uint32_t>(physical) >= k_.blocks_)
            return false;
        uint64_t row = (static_cast<uint64_t>(physical) * DIM + k % DIM) * k_.kvHeads_ + h;
        k_.Decode(row, ROWS);
        Abs(k_.a_, k_.normCompact_, 2 * ROWS);
        Duplicate(k_.bytes_.template ReinterpretCast<uint32_t>(), 0xffffffffU, 8);
        PipeBarrier<PIPE_V>();
        CompareScalar(k_.bytes_, k_.a_, 65504.0F, CMPMODE::LE, 2 * ROWS);
        PipeBarrier<PIPE_V>();
        And(k_.reducedMain_.template ReinterpretCast<uint16_t>(), k_.reducedMain_.template ReinterpretCast<uint16_t>(),
            k_.bytes_.template ReinterpretCast<uint16_t>(), 16);
        PipeBarrier<PIPE_V>();
        auto raw = k_.a_.template ReinterpretCast<half>();
        Gather(k_.b_, k_.normCompact_, expanded, 0, ELEMS);
        PipeBarrier<PIPE_V>();
        Mul(k_.primary_, k_.primary_, k_.b_, ELEMS);
        Muls(k_.gammaCompact_, k_.gammaCompact_, COEFFICIENT, ROWS);
        PipeBarrier<PIPE_V>();
        Gather(k_.b_, k_.gammaCompact_, expanded, 0, ELEMS);
        PipeBarrier<PIPE_V>();
        Muls(k_.c_, k_.b_, -1.0F, ELEMS);
        PipeBarrier<PIPE_V>();
        Select(k_.signs_, k_.bytesFloat_.template ReinterpretCast<uint8_t>(), k_.b_, k_.c_,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, 64, ROWS, {1, 1, 1, 16, 16, 16});
        Select(k_.signs_[64], k_.bytesFloat_[ROWS * 2].template ReinterpretCast<uint8_t>(), k_.b_[64], k_.c_[64],
               SELMODE::VSEL_TENSOR_TENSOR_MODE, 64, ROWS, {1, 1, 1, 16, 16, 16});
        PipeBarrier<PIPE_V>();
        auto packed = k_.b_.template ReinterpretCast<half>();
        for (uint32_t part = 0; part < 4; ++part) {
            auto value = part < 2 ? k_.primary_ : k_.signs_;
            Cast(raw, value, RoundMode::CAST_RINT, ELEMS);
            PipeBarrier<PIPE_V>();
            DataCopy(packed[part * DIM], raw, DataCopyParams{ROWS, DIM * 2 / 32, 0, DIM * 6 / 32});
            PipeBarrier<PIPE_V>();
            if (part % 2 == 0) {
                Cast(k_.queryResidualCache_, raw, RoundMode::CAST_NONE, ELEMS);
                PipeBarrier<PIPE_V>();
                Sub(value, value, k_.queryResidualCache_, ELEMS);
                PipeBarrier<PIPE_V>();
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(bm[(static_cast<uint64_t>(h) * k_.maxLen_ + k) * 512], packed, ROWS * 512);
        PipeBarrier<PIPE_ALL>();
        return true;
    }
    __aicore__ inline void PrepareKeys(GM_ADDR bp, GM_ADDR fp, uint32_t nt, uint32_t tileN, uint32_t wave)
    {
        GlobalTensor<half> bm;
        bm.SetGlobalBuffer((__gm__ half *)bp);
        GlobalTensor<uint32_t> flags;
        flags.SetGlobalBuffer((__gm__ uint32_t *)fp);
        auto expanded = k_.queryCache_.template ReinterpretCast<uint32_t>();
        for (uint32_t row = 0; row < ROWS; ++row)
            Duplicate(expanded[row * DIM], row * 4, DIM);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t task = wave * GetBlockNum() * GetTaskRation() + GetBlockIdx();
             task < k_.kvHeads_ * nt && task < (wave + 1) * GetBlockNum() * GetTaskRation();
             task += GetBlockNum() * GetTaskRation()) {
            const uint32_t h = task / nt, k = task % nt * tileN;
            Duplicate(k_.reducedMain_.template ReinterpretCast<uint32_t>(), 0xffffffffU, 8);
            PipeBarrier<PIPE_V>();
            bool safe = true;
            for (uint32_t offset = 0; offset < tileN && safe; offset += ROWS)
                safe = PrepareBlock(h, k + offset, bm, expanded);
            PipeBarrier<PIPE_ALL>();
            safe = safe && k_.reducedMain_.template ReinterpretCast<uint32_t>().GetValue(0) == 0xffffffffU &&
                   k_.reducedMain_.template ReinterpretCast<uint32_t>().GetValue(1) == 0xffffffffU;
            if (!safe)
                MarkFallback(fp);
            PipeBarrier<PIPE_ALL>();
            flags.SetValue((task + 1) * 128, safe ? 1U : 2U);
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                flags[(task + 1) * 128]);
        }
    }

    __aicore__ inline void MarkFallback(GM_ADDR flag)
    {
        GlobalTensor<int32_t> status;
        status.SetGlobalBuffer((__gm__ int32_t *)flag);
        auto local = k_.output_.template ReinterpretCast<int32_t>();
        Duplicate(local, static_cast<int32_t>(1), 8);
        PipeBarrier<PIPE_ALL>();
        SetAtomicMax<int32_t>();
        DataCopy(status, local, 8);
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline bool SafeQueries()
    {
        Abs(k_.queryResidualCache_, k_.primary_, ELEMS);
        PipeBarrier<PIPE_V>();
        CompareScalar(k_.bytes_, k_.queryResidualCache_, 65504.0F, CMPMODE::LE, ELEMS);
        PipeBarrier<PIPE_V>();
        Abs(k_.queryResidualCache_, k_.signs_, ELEMS);
        PipeBarrier<PIPE_V>();
        CompareScalar(k_.bytes_[ELEMS / 8], k_.queryResidualCache_, 65504.0F, CMPMODE::LE, ELEMS);
        PipeBarrier<PIPE_V>();
        And(k_.bytes_.template ReinterpretCast<uint16_t>(), k_.bytes_.template ReinterpretCast<uint16_t>(),
            k_.bytes_[ELEMS / 8].template ReinterpretCast<uint16_t>(), ELEMS / 16);
        PipeBarrier<PIPE_V>();
        Not(k_.bytes_.template ReinterpretCast<uint16_t>(), k_.bytes_.template ReinterpretCast<uint16_t>(), ELEMS / 16);
        PipeBarrier<PIPE_V>();
        Cast(k_.bytesHalf_, k_.bytes_, RoundMode::CAST_NONE, ELEMS / 8);
        PipeBarrier<PIPE_V>();
        ReduceMax(k_.normFloat_.template ReinterpretCast<half>(), k_.bytesHalf_,
                  k_.gammaFloat_.template ReinterpretCast<half>(), ELEMS / 8, false);
        PipeBarrier<PIPE_ALL>();
        return static_cast<float>(k_.normFloat_.template ReinterpretCast<half>().GetValue(0)) == 0.0F;
    }

private:
    Kernel<NormType, true> &k_;
};
using PrefillA = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using PrefillB = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;
using PrefillC = MatmulType<TPosition::GM, CubeFormat::ND, float>;

template <typename TilingData>
__aicore__ inline void PrefillCompute(GM_ADDR ap, GM_ADDR bp, GM_ADDR out, const TilingData &data, GM_ADDR fp,
                                      bool querySafe)
{
    if ASCEND_IS_AIV {
        return;
    }
    TCubeTiling cube = data.cube;
    TPipe pipe;
    matmul::Matmul<PrefillA, PrefillB, PrefillC, PrefillC, CFG_MDL> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &cube);
    GlobalTensor<half> am, bm;
    GlobalTensor<float> output;
    am.SetGlobalBuffer((__gm__ half *)ap);
    bm.SetGlobalBuffer((__gm__ half *)bp);
    output.SetGlobalBuffer((__gm__ float *)out);
    GlobalTensor<uint32_t> flags;
    flags.SetGlobalBuffer((__gm__ uint32_t *)fp);
    const uint32_t nt = data.maxLen / cube.singleCoreN;
    const uint32_t waves = (data.kvHeads * nt + GetBlockNum() * 2 - 1) / (GetBlockNum() * 2);
    for (uint32_t wave = 0; wave <= waves; ++wave) {
        if (querySafe && wave > 0) {
            for (uint32_t task = GetBlockIdx(); task < data.qHeads * nt; task += GetBlockNum()) {
                const uint32_t h = task / nt, n = task % nt * cube.singleCoreN, kh = h / (data.qHeads / data.kvHeads);
                const uint32_t keyTask = kh * nt + n / cube.singleCoreN;
                const uint32_t waveSize = GetBlockNum() * 2;
                if (keyTask < (wave - 1) * waveSize || keyTask >= wave * waveSize)
                    continue;
                const uint32_t flagIndex = (keyTask + 1) * 128;
                DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(
                    flags[flagIndex]);
                if (flags.GetValue(flagIndex) != 1)
                    continue;
                mm.SetOrgShape(data.tokens, data.qHeads * data.maxLen, 512);
                mm.SetSingleShape(data.tokens, cube.singleCoreN, 512);
                mm.SetTensorA(am[(uint64_t)h * data.tokens * 512]);
                mm.SetTensorB(bm[((uint64_t)kh * data.maxLen + n) * 512], true);
                mm.IterateAll(output[(uint64_t)h * data.maxLen + n]);
                mm.End();
            }
        }
        SyncAll<false>();
    }
}

template <typename NormType, typename TilingData>
__aicore__ inline void RunPrefill(GM_ADDR q, GM_ADDR qr, GM_ADDR idx, GM_ADDR sign, GM_ADDR norm, GM_ADDR gamma,
                                  GM_ADDR table, GM_ADDR lens, GM_ADDR out, GM_ADDR user, const TilingData &t)
{
    const uint32_t tokens = t.tokens, heads = t.qHeads, length = t.maxLen, bits = t.bits;
    const uint32_t kvheads = t.kvHeads, blocks = t.blocks, width = t.tableWidth;
    auto ap = user;
    auto bp = ap + (uint64_t)heads * tokens * 512 * 2;
    auto fp = bp + (uint64_t)kvheads * length * 512 * 2;
    GlobalTensor<uint32_t> flags;
    flags.SetGlobalBuffer((__gm__ uint32_t *)fp);
    uint32_t nt = length / 256;
    bool safe = true;
    if ASCEND_IS_AIV {
        Kernel<NormType, true> op;
        op.Init(q, qr, idx, sign, norm, gamma, table, lens, out, tokens, heads, kvheads, 1, blocks, width, length,
                bits);
        PrefillPrepare<NormType> prepare(op);
        if (GetBlockIdx() == 0) {
            for (uint32_t i = 0; i < 8; ++i)
                flags.SetValue(i, 0U);
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(flags);
        }
        for (uint32_t task = GetBlockIdx(); task < kvheads * nt; task += GetBlockNum() * GetTaskRation()) {
            flags.SetValue((task + 1) * 128, 0U);
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                flags[(task + 1) * 128]);
        }
        SyncAll<false>();
        if (!prepare.PrepareQueries(ap))
            prepare.MarkFallback(fp);
        SyncAll<false>();
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(flags);
        const bool querySafe = flags.GetValue(0) == 0;
        SyncAll<false>();
        const uint32_t waves = (kvheads * nt + GetBlockNum() * GetTaskRation() - 1) / (GetBlockNum() * GetTaskRation());
        for (uint32_t wave = 0; wave <= waves; ++wave) {
            if (querySafe && wave < waves)
                prepare.PrepareKeys(bp, fp, nt, 256, wave);
            SyncAll<false>();
        }
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(flags);
        safe = flags.GetValue(0) == 0;
    }
    if ASCEND_IS_AIC {
        SyncAll<false>();
        SyncAll<false>();
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(flags);
        const bool querySafe = flags.GetValue(0) == 0;
        SyncAll<false>();
        PrefillCompute(ap, bp, out, t, fp, querySafe);
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(flags);
        safe = flags.GetValue(0) == 0;
    }
    if (!safe) {
        if ASCEND_IS_AIV {
            Kernel<NormType> fallback;
            fallback.Init(q, qr, idx, sign, norm, gamma, table, lens, out, tokens, heads, kvheads, 1, blocks, width,
                          length, bits);
            fallback.Process(GetBlockNum() * GetTaskRation());
        }
    }
}
} // namespace TurboQuant
#endif
