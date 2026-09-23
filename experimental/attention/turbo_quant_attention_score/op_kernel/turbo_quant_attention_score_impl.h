/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef TURBO_QUANT_ATTENTION_SCORE_IMPL_H
#define TURBO_QUANT_ATTENTION_SCORE_IMPL_H
#include "kernel_operator.h"

namespace TurboQuant {
using namespace AscendC;
constexpr uint32_t DIM = 128;
constexpr uint32_t ROWS = 16;
constexpr uint32_t ELEMS = ROWS * DIM;
constexpr float COEFFICIENT = 0.009791516697777346F;

template <typename NormType>
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR qr, GM_ADDR idx, GM_ADDR sign,
        GM_ADDR norm, GM_ADDR gamma, GM_ADDR table, GM_ADDR lens, GM_ADDR out,
        uint32_t tokens, uint32_t qHeads, uint32_t kvHeads, uint32_t batch,
        uint32_t blocks, uint32_t width, uint32_t maxLen, uint32_t bits)
    {
        tokens_ = tokens; qHeads_ = qHeads; kvHeads_ = kvHeads; batch_ = batch;
        blocks_ = blocks; width_ = width; maxLen_ = maxLen; bits_ = bits;
        q_.SetGlobalBuffer((__gm__ bfloat16_t *)q);
        qr_.SetGlobalBuffer((__gm__ bfloat16_t *)qr);
        idx_.SetGlobalBuffer((__gm__ uint8_t *)idx);
        sign_.SetGlobalBuffer((__gm__ uint8_t *)sign);
        norm_.SetGlobalBuffer((__gm__ NormType *)norm);
        gamma_.SetGlobalBuffer((__gm__ NormType *)gamma);
        table_.SetGlobalBuffer((__gm__ int32_t *)table);
        lens_.SetGlobalBuffer((__gm__ DTYPE_SEQ_LENS *)lens);
        out_.SetGlobalBuffer((__gm__ float *)out);
        pipe_.InitBuffer(storage_, 160 * 1024);
        auto mem = storage_.Get<float>();
        primary_ = mem[0]; signs_ = mem[ELEMS]; a_ = mem[2 * ELEMS];
        b_ = mem[3 * ELEMS]; c_ = mem[4 * ELEMS];
        offset0_ = mem[5 * ELEMS].ReinterpretCast<uint32_t>();
        offset1_ = mem[6 * ELEMS].ReinterpretCast<uint32_t>();
        offsetSign_ = mem[7 * ELEMS].ReinterpretCast<uint32_t>();
        scale_ = mem[8 * ELEMS]; scaleSign_ = mem[9 * ELEMS];
        bytesFloat_ = mem[10 * ELEMS];
        bytesHalf_ = mem[11 * ELEMS].ReinterpretCast<half>();
        bytes_ = mem[12 * ELEMS].ReinterpretCast<uint8_t>();
        query_ = mem[13 * ELEMS]; queryResidual_ = query_[DIM];
        queryRaw_ = query_[2 * DIM].ReinterpretCast<bfloat16_t>();
        normFloat_ = mem[14 * ELEMS]; gammaFloat_ = normFloat_[256];
        normRaw_ = normFloat_[512].ReinterpretCast<NormType>();
        gammaRaw_ = normFloat_[768].ReinterpretCast<NormType>();
        reducedMain_ = mem[15 * ELEMS]; reducedSign_ = reducedMain_[64];
        output_ = reducedMain_[128]; centroids_ = reducedMain_[160];
        normCompact_ = reducedMain_[192]; gammaCompact_ = reducedMain_[224];
        evenOffsets_ = reducedMain_[256].ReinterpretCast<uint32_t>();
        oddOffsets_ = reducedMain_[288].ReinterpretCast<uint32_t>();
        normOffsets_ = reducedMain_[320].ReinterpretCast<uint32_t>();
        queryCache_ = mem[16 * ELEMS]; queryResidualCache_ = queryCache_[4096];
        InitDecode();
    }

    __aicore__ inline void Process()
    {
        const uint64_t tiles = (maxLen_ + ROWS - 1) / ROWS;
        // Each task owns one key tile and reuses its decoded keys across GQA heads.
        // Prefill is split into groups of 8 queries to expose enough parallelism.
        const uint32_t queryGroups = batch_ == 1 ? (tokens_ + 7) / 8 : batch_;
        const uint64_t tasks = tiles * kvHeads_ * queryGroups;
        for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
            const uint32_t start = (task % tiles) * ROWS;
            const uint32_t head = (task / tiles) % kvHeads_;
            const uint32_t group = task / (tiles * kvHeads_);
            const uint32_t batch = batch_ == 1 ? 0 : group;
            const int64_t rawLength = lens_.GetValue(batch);
            const uint32_t length = rawLength > 0 && rawLength <= maxLen_ ? static_cast<uint32_t>(rawLength) : 0;
            const uint32_t count = maxLen_ - start < ROWS ? maxLen_ - start : ROWS;
            int32_t physical = -1;
            if (start < length) {
                physical = table_.GetValue(static_cast<uint64_t>(batch) * width_ + start / DIM);
            }
            const bool valid = physical >= 0 && static_cast<uint32_t>(physical) < blocks_;
            if (valid) {
                const uint32_t validRows = length - start < count ? length - start : count;
                const uint64_t cacheRow = (static_cast<uint64_t>(physical) * DIM + start % DIM) * kvHeads_ + head;
                Decode(cacheRow, validRows);
            }
            const uint32_t first = batch_ == 1 ? group * 8 : group;
            const uint32_t end = batch_ == 1 ? (first + 8 < tokens_ ? first + 8 : tokens_) : first + 1;
            if (valid && qHeads_ / kvHeads_ <= 4) {
                LoadQueries(first, end - first, head);
            }
            for (uint32_t token = first; token < end; ++token) {
                for (uint32_t g = 0; g < qHeads_ / kvHeads_; ++g) {
                    const uint64_t queryRow = static_cast<uint64_t>(token) * qHeads_ + head * (qHeads_ / kvHeads_) + g;
                    if (valid) {
                        Compute(queryRow, (token - first) * (qHeads_ / kvHeads_) + g);
                    }
                    const uint32_t validRows = valid ? (length - start < count ? length - start : count) : 0;
                    if (validRows < count) {
                        PipeBarrier<PIPE_ALL>();
                        const uint32_t negativeInfinity = 0xff800000U;
                        const float fill = *reinterpret_cast<const float *>(&negativeInfinity);
                        for (uint32_t r = validRows; r < count; ++r) {
                            output_.SetValue(r, fill);
                        }
                    }
                    PipeBarrier<PIPE_ALL>();
                    DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
                    DataCopyPad(out_[queryRow * maxLen_ + start], output_, copy);
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }

private:
    __aicore__ inline void InitDecode()
    {
        const float c2[] = {-0.1335033178F, -0.04002048075F, 0.04002048075F, 0.1335033178F};
        const float c3[] = {-0.19020693F, -0.1187859178F, -0.06682205945F, -0.02166347019F,
            0.02166347019F, 0.06682205945F, 0.1187859178F, 0.19020693F};
        const float c4[] = {-0.2414890379F, -0.1828317791F, -0.1429702938F, -0.1109927073F,
            -0.08325428516F, -0.05802082643F, -0.03428063914F, -0.01134236995F,
            0.01134236995F, 0.03428063914F, 0.05802082643F, 0.08325428516F,
            0.1109927073F, 0.1429702938F, 0.1828317791F, 0.2414890379F};
        for (uint32_t i = 0; i < (1U << bits_); ++i) {
            centroids_.SetValue(i, bits_ == 2 ? c2[i] : (bits_ == 3 ? c3[i] : c4[i]));
        }
        const uint32_t stride = bits_ == 2 ? 32 : 64;
        const float inversePowers[] = {1.0F, 0.5F, 0.25F, 0.125F, 0.0625F, 0.03125F, 0.015625F, 0.0078125F};
        for (uint32_t r = 0; r < ROWS; ++r) {
            evenOffsets_.SetValue(r, r * 2 * 4);
            oddOffsets_.SetValue(r, (r * 2 + 1) * 4);
            normOffsets_.SetValue(r, r * 16 * 4);
            for (uint32_t d = 0; d < DIM; ++d) {
                const uint32_t i = r * DIM + d, byte = d * bits_ / 8;
                const uint32_t next = byte + 1 < bits_ * 16 ? byte + 1 : byte;
                offset0_.SetValue(i, (r * stride + byte) * 4);
                offset1_.SetValue(i, (r * stride + next) * 4);
                offsetSign_.SetValue(i, (r * 32 + d / 8) * 4);
                scale_.SetValue(i, inversePowers[d * bits_ % 8]);
                scaleSign_.SetValue(i, inversePowers[d % 8]);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // Integer division/modulo via exact binary FP32 operations. Inputs are at
    // most 65535, so unpacking is exact, including codes spanning two bytes.
    __aicore__ inline void Modulo(float divisor)
    {
        Cast(c_.ReinterpretCast<int32_t>(), a_, RoundMode::CAST_FLOOR, ELEMS);
        PipeBarrier<PIPE_V>();
        Cast(a_, c_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, ELEMS);
        PipeBarrier<PIPE_V>();
        Muls(b_, a_, 1.0F / divisor, ELEMS);
        PipeBarrier<PIPE_V>();
        Cast(c_.ReinterpretCast<int32_t>(), b_, RoundMode::CAST_FLOOR, ELEMS);
        PipeBarrier<PIPE_V>();
        Cast(b_, c_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, ELEMS);
        PipeBarrier<PIPE_V>();
        Muls(b_, b_, divisor, ELEMS);
        PipeBarrier<PIPE_V>();
        Sub(a_, a_, b_, ELEMS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadBytes(const GlobalTensor<uint8_t> &src, uint32_t rowBytes, uint32_t rows)
    {
        DataCopyExtParams copy{static_cast<uint16_t>(rows), rowBytes, (kvHeads_ - 1) * rowBytes, 0, 0};
        DataCopyPad(bytes_, src, copy, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        const uint32_t count = ROWS * ((rowBytes + 31) / 32 * 32);
        Cast(bytesHalf_, bytes_, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Cast(bytesFloat_, bytesHalf_, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Decode(uint64_t row, uint32_t rows)
    {
        LoadBytes(idx_[row * bits_ * 16], bits_ * 16, rows);
        Gather(a_, bytesFloat_, offset0_, 0, ELEMS);
        Gather(b_, bytesFloat_, offset1_, 0, ELEMS);
        PipeBarrier<PIPE_V>();
        Muls(b_, b_, 256.0F, ELEMS);
        PipeBarrier<PIPE_V>();
        Add(a_, a_, b_, ELEMS);
        PipeBarrier<PIPE_V>();
        Mul(a_, a_, scale_, ELEMS);
        PipeBarrier<PIPE_V>();
        Modulo(bits_ == 2 ? 4.0F : (bits_ == 3 ? 8.0F : 16.0F));
        Muls(a_, a_, 4.0F, ELEMS);
        PipeBarrier<PIPE_V>();
        Cast(c_.ReinterpretCast<int32_t>(), a_, RoundMode::CAST_RINT, ELEMS);
        PipeBarrier<PIPE_V>();
        Gather(primary_, centroids_, c_.ReinterpretCast<uint32_t>(), 0, ELEMS);
        PipeBarrier<PIPE_ALL>();
        LoadBytes(sign_[row * 16], 16, rows);
        Gather(a_, bytesFloat_, offsetSign_, 0, ELEMS);
        PipeBarrier<PIPE_V>();
        Mul(a_, a_, scaleSign_, ELEMS);
        PipeBarrier<PIPE_V>();
        Modulo(2);
        Muls(signs_, a_, 2.0F, ELEMS);
        PipeBarrier<PIPE_V>();
        Adds(signs_, signs_, -1.0F, ELEMS);
        DataCopyExtParams copy{static_cast<uint16_t>(rows), 2, (kvHeads_ - 1) * 2, 0, 0};
        DataCopyPad(normRaw_, norm_[row], copy, DataCopyPadExtParams<NormType>{false, 0, 0, 0});
        DataCopyPad(gammaRaw_, gamma_[row], copy, DataCopyPadExtParams<NormType>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Cast(normFloat_, normRaw_, RoundMode::CAST_NONE, 256);
        Cast(gammaFloat_, gammaRaw_, RoundMode::CAST_NONE, 256);
        PipeBarrier<PIPE_V>();
        Gather(normCompact_, normFloat_, normOffsets_, 0, ROWS);
        Gather(gammaCompact_, gammaFloat_, normOffsets_, 0, ROWS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Dot(const LocalTensor<float> &keys, const LocalTensor<float> &query,
                              const LocalTensor<float> &result)
    {
        Mul(a_, keys, query, 64, ROWS, {1, 1, 1, 16, 16, 0});
        Mul(a_[64], keys[64], query[64], 64, ROWS, {1, 1, 1, 16, 16, 0});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(result, a_, 64, ROWS * 2, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadQueries(uint32_t first, uint32_t tokens, uint32_t head)
    {
        const uint32_t groups = qHeads_ / kvHeads_;
        const uint64_t offset = (static_cast<uint64_t>(first) * qHeads_ + head * groups) * DIM;
        // Byte decode temporaries are dead until the next key tile; reuse them
        // for batched BF16 query loads without increasing the UB footprint.
        auto raw = bytesFloat_.ReinterpretCast<bfloat16_t>();
        auto residualRaw = bytesHalf_.ReinterpretCast<bfloat16_t>();
        DataCopyExtParams copy{static_cast<uint16_t>(tokens), groups * DIM * 2, (qHeads_ - groups) * DIM * 2, 0, 0};
        DataCopyPad(raw, q_[offset], copy, DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        DataCopyPad(residualRaw, qr_[offset], copy, DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Cast(queryCache_, raw, RoundMode::CAST_NONE, tokens * groups * DIM);
        Cast(queryResidualCache_, residualRaw, RoundMode::CAST_NONE, tokens * groups * DIM);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint64_t row, uint32_t slot)
    {
        if (qHeads_ / kvHeads_ <= 4) {
            Dot(primary_, queryCache_[slot * DIM], reducedMain_);
            Dot(signs_, queryResidualCache_[slot * DIM], reducedSign_);
        } else {
            DataCopy(queryRaw_, q_[row * DIM], DIM);
            DataCopy(queryRaw_[DIM], qr_[row * DIM], DIM);
            PipeBarrier<PIPE_ALL>();
            Cast(query_, queryRaw_, RoundMode::CAST_NONE, DIM);
            Cast(queryResidual_, queryRaw_[DIM], RoundMode::CAST_NONE, DIM);
            PipeBarrier<PIPE_V>();
            Dot(primary_, query_, reducedMain_);
            Dot(signs_, queryResidual_, reducedSign_);
        }
        Gather(output_, reducedMain_, evenOffsets_, 0, ROWS);
        Gather(a_, reducedMain_, oddOffsets_, 0, ROWS);
        Gather(b_, reducedSign_, evenOffsets_, 0, ROWS);
        Gather(c_, reducedSign_, oddOffsets_, 0, ROWS);
        PipeBarrier<PIPE_V>();
        Add(output_, output_, a_, ROWS);
        Add(b_, b_, c_, ROWS);
        PipeBarrier<PIPE_V>();
        Mul(output_, output_, normCompact_, ROWS);
        Mul(b_, b_, gammaCompact_, ROWS);
        PipeBarrier<PIPE_V>();
        Muls(b_, b_, COEFFICIENT, ROWS);
        PipeBarrier<PIPE_V>();
        Add(output_, output_, b_, ROWS);
        PipeBarrier<PIPE_V>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> storage_;
    GlobalTensor<bfloat16_t> q_, qr_;
    GlobalTensor<uint8_t> idx_, sign_;
    GlobalTensor<NormType> norm_, gamma_;
    GlobalTensor<int32_t> table_;
    GlobalTensor<DTYPE_SEQ_LENS> lens_;
    GlobalTensor<float> out_;
    LocalTensor<float> primary_, signs_, a_, b_, c_, scale_, scaleSign_, bytesFloat_;
    LocalTensor<float> query_, queryResidual_, normFloat_, gammaFloat_, reducedMain_, reducedSign_, output_, centroids_;
    LocalTensor<uint32_t> offset0_, offset1_, offsetSign_;
    LocalTensor<uint32_t> evenOffsets_, oddOffsets_, normOffsets_;
    LocalTensor<float> normCompact_, gammaCompact_;
    LocalTensor<float> queryCache_, queryResidualCache_;
    LocalTensor<half> bytesHalf_;
    LocalTensor<uint8_t> bytes_;
    LocalTensor<bfloat16_t> queryRaw_;
    LocalTensor<NormType> normRaw_, gammaRaw_;
    uint32_t tokens_, qHeads_, kvHeads_, batch_, blocks_, width_, maxLen_, bits_;
};
} // namespace TurboQuant
#endif
