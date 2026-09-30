/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TURBO_QUANT_ATTENTION_SCORE_IMPL_H
#define TURBO_QUANT_ATTENTION_SCORE_IMPL_H
#include "kernel_operator.h"

namespace TurboQuant {
using namespace AscendC;
constexpr uint32_t DIM = 128;
constexpr uint32_t ROWS = 32;
constexpr uint32_t ELEMS = ROWS * DIM;
constexpr uint32_t UB_BYTES = 185 * 1024;
constexpr uint32_t QUERY_CACHE_ELEMS = 32 * DIM;
// Eight full tile buffers, compact decode scratch, and two query caches.
constexpr uint32_t UB_FLOATS =
    8 * ELEMS + 2 * DIM + 112 * ROWS + 3 * DIM + 48 * ROWS + 14 * ROWS + 96 + 2 * QUERY_CACHE_ELEMS;
static_assert(UB_FLOATS * sizeof(float) <= UB_BYTES, "UB layout exceeds its allocation");
constexpr float COEFFICIENT = 0.009791516697777346F;

template <typename NormType>
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR qr, GM_ADDR idx, GM_ADDR sign, GM_ADDR norm, GM_ADDR gamma,
                                GM_ADDR table, GM_ADDR lens, GM_ADDR out, uint32_t tokens, uint32_t qHeads,
                                uint32_t kvHeads, uint32_t batch, uint32_t blocks, uint32_t width, uint32_t maxLen,
                                uint32_t bits)
    {
        tokens_ = tokens;
        qHeads_ = qHeads;
        kvHeads_ = kvHeads;
        batch_ = batch;
        blocks_ = blocks;
        width_ = width;
        maxLen_ = maxLen;
        bits_ = bits;
        q_.SetGlobalBuffer((__gm__ bfloat16_t *)q);
        qr_.SetGlobalBuffer((__gm__ bfloat16_t *)qr);
        idx_.SetGlobalBuffer((__gm__ uint8_t *)idx);
        sign_.SetGlobalBuffer((__gm__ uint8_t *)sign);
        norm_.SetGlobalBuffer((__gm__ NormType *)norm);
        gamma_.SetGlobalBuffer((__gm__ NormType *)gamma);
        table_.SetGlobalBuffer((__gm__ int32_t *)table);
        lens_.SetGlobalBuffer((__gm__ DTYPE_SEQ_LENS *)lens);
        out_.SetGlobalBuffer((__gm__ float *)out);
        pipe_.InitBuffer(storage_, UB_BYTES);
        auto mem = storage_.Get<float>();
        primary_ = mem[0];
        signs_ = mem[ELEMS];
        a_ = mem[2 * ELEMS];
        b_ = mem[3 * ELEMS];
        c_ = mem[4 * ELEMS];
        offset0_ = mem[5 * ELEMS].ReinterpretCast<uint32_t>();
        offset1_ = mem[6 * ELEMS].ReinterpretCast<uint32_t>();
        offsetSign_ = mem[7 * ELEMS].ReinterpretCast<uint32_t>();
        // All offsets below are in FP32 elements and remain 32-byte aligned.
        uint32_t offset = 8 * ELEMS;
        scale_ = mem[offset];
        offset += DIM;
        scaleSign_ = mem[offset];
        offset += DIM;
        bytesFloat_ = mem[offset];
        offset += ROWS * 64;
        bytesHalf_ = mem[offset].ReinterpretCast<half>();
        offset += ROWS * 32;
        bytes_ = mem[offset].ReinterpretCast<uint8_t>();
        offset += ROWS * 16;
        query_ = mem[offset];
        queryResidual_ = mem[offset + DIM];
        offset += 2 * DIM;
        queryRaw_ = mem[offset].ReinterpretCast<bfloat16_t>();
        queryResidualRaw_ = mem[offset + DIM / 2].ReinterpretCast<bfloat16_t>();
        offset += DIM;
        normFloat_ = mem[offset];
        offset += ROWS * 16;
        gammaFloat_ = mem[offset];
        offset += ROWS * 16;
        normRaw_ = mem[offset].ReinterpretCast<NormType>();
        offset += ROWS * 8;
        gammaRaw_ = mem[offset].ReinterpretCast<NormType>();
        offset += ROWS * 8;
        reducedMain_ = mem[offset];
        offset += 4 * ROWS;
        reducedSign_ = mem[offset];
        offset += 4 * ROWS;
        output_ = mem[offset];
        offset += 4 * ROWS;
        centroids_ = mem[offset];
        offset += 32;
        normCompact_ = mem[offset];
        offset += ROWS;
        gammaCompact_ = mem[offset];
        offset += ROWS;
        normOffsets_ = mem[offset].ReinterpretCast<uint32_t>();
        offset += 64;
        queryCache_ = mem[offset];
        offset += QUERY_CACHE_ELEMS;
        queryResidualCache_ = mem[offset];
        offset += QUERY_CACHE_ELEMS;
        InitDecode();
    }

    __aicore__ inline void Process()
    {
        const uint64_t tiles = (maxLen_ + ROWS - 1) / ROWS;
        // Each task owns one key tile and reuses its decoded keys across GQA heads.
        // Prefill is split into groups of 8 queries to expose enough parallelism.
        const uint32_t queryGroups = batch_ == 1 ? (tokens_ + 7) / 8 : batch_;
        const uint64_t tasks = tiles * kvHeads_ * queryGroups;
        uint64_t cachedQuery = static_cast<uint64_t>(-1);
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
            if (valid && qHeads_ / kvHeads_ <= 4 && cachedQuery != static_cast<uint64_t>(group) * kvHeads_ + head) {
                LoadQueries(first, end - first, head);
                cachedQuery = static_cast<uint64_t>(group) * kvHeads_ + head;
            }
            const uint32_t groups = qHeads_ / kvHeads_;
            const bool batchedOutput = groups <= 4 && maxLen_ <= 0x3fffffffU;
            const uint32_t validRows = valid ? (length - start < count ? length - start : count) : 0;
            for (uint32_t token = first; token < end; ++token) {
                const uint64_t firstQuery = static_cast<uint64_t>(token) * qHeads_ + head * groups;
                if (valid && batchedOutput) {
                    ComputeGroup((token - first) * groups, groups);
                }
                for (uint32_t g = 0; g < groups; ++g) {
                    auto result = output_[batchedOutput ? g * ROWS : 0];
                    if (valid && !batchedOutput) {
                        Compute(firstQuery + g, (token - first) * groups + g, result);
                    }
                    if (validRows < count) {
                        PipeBarrier<PIPE_ALL>();
                        const uint32_t negativeInfinity = 0xff800000U;
                        const float fill = *reinterpret_cast<const float *>(&negativeInfinity);
                        for (uint32_t row = validRows; row < count; ++row) {
                            result.SetValue(row, fill);
                        }
                    }
                    if (!batchedOutput) {
                        PipeBarrier<PIPE_ALL>();
                        DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
                        DataCopyPad(out_[(firstQuery + g) * maxLen_ + start], result, copy);
                        PipeBarrier<PIPE_ALL>();
                    }
                }
                if (batchedOutput) {
                    PipeBarrier<PIPE_ALL>();
                    const uint32_t srcStride = (ROWS - (count + 7) / 8 * 8) / 8;
                    DataCopyExtParams copy{static_cast<uint16_t>(groups), count * 4U, srcStride, (maxLen_ - count) * 4U,
                                           0};
                    DataCopyPad(out_[firstQuery * maxLen_ + start], output_, copy);
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }

private:
    __aicore__ inline void InitDecode()
    {
        // Gather reads a full vector of offsets even for a short output count.
        // Initialize padding in the short norm/gamma offset table.
        Duplicate(normOffsets_, static_cast<uint32_t>(0), 64);
        PipeBarrier<PIPE_ALL>();
        const float c2[] = {-0.1335033178F, -0.04002048075F, 0.04002048075F, 0.1335033178F};
        const float c3[] = {-0.19020693F,   -0.1187859178F, -0.06682205945F, -0.02166347019F,
                            0.02166347019F, 0.06682205945F, 0.1187859178F,   0.19020693F};
        const float c4[] = {-0.2414890379F,  -0.1828317791F,  -0.1429702938F,  -0.1109927073F,
                            -0.08325428516F, -0.05802082643F, -0.03428063914F, -0.01134236995F,
                            0.01134236995F,  0.03428063914F,  0.05802082643F,  0.08325428516F,
                            0.1109927073F,   0.1429702938F,   0.1828317791F,   0.2414890379F};
        for (uint32_t i = 0; i < (1U << bits_); ++i) {
            centroids_.SetValue(i, bits_ == 2 ? c2[i] : (bits_ == 3 ? c3[i] : c4[i]));
        }
        const uint32_t stride = bits_ == 2 ? 32 : 64;
        const float inversePowers[] = {1.0F, 0.5F, 0.25F, 0.125F, 0.0625F, 0.03125F, 0.015625F, 0.0078125F};
        for (uint32_t r = 0; r < ROWS; ++r) {
            normOffsets_.SetValue(r, r * (kvHeads_ <= 16 ? kvHeads_ : 16) * 4);
        }
        // Build one dimension pattern, then replicate with vector row offsets.
        for (uint32_t d = 0; d < DIM; ++d) {
            const uint32_t byte = d * bits_ / 8;
            const uint32_t next = byte + 1 < bits_ * 16 ? byte + 1 : byte;
            offset0_.SetValue(d, byte * 4);
            offset1_.SetValue(d, next * 4);
            offsetSign_.SetValue(d, (d / 8) * 4);
            scale_.SetValue(d, inversePowers[d * bits_ % 8]);
            scaleSign_.SetValue(d, inversePowers[d % 8]);
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t r = 1; r < ROWS; ++r) {
            Adds(offset0_[r * DIM].ReinterpretCast<int32_t>(), offset0_.ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(r * stride * 4), DIM);
            Adds(offset1_[r * DIM].ReinterpretCast<int32_t>(), offset1_.ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(r * stride * 4), DIM);
            Adds(offsetSign_[r * DIM].ReinterpretCast<int32_t>(), offsetSign_.ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(r * 32 * 4), DIM);
        }
        PipeBarrier<PIPE_ALL>();
    }

    // Floor nonnegative unpack values, then mask their low integer bits.
    // Values are at most 65535; conversion is exact after rounding down.
    __aicore__ inline void Modulo(float divisor)
    {
        Cast(c_.ReinterpretCast<int32_t>(), a_, RoundMode::CAST_FLOOR, ELEMS);
        Duplicate(b_.ReinterpretCast<int32_t>(), static_cast<int32_t>(divisor) - 1, ELEMS);
        PipeBarrier<PIPE_V>();
        And(c_.ReinterpretCast<uint16_t>(), c_.ReinterpretCast<uint16_t>(), b_.ReinterpretCast<uint16_t>(), ELEMS * 2);
        PipeBarrier<PIPE_V>();
        Cast(a_, c_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, ELEMS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadBytes(const GlobalTensor<uint8_t> &src, uint32_t rowBytes, uint32_t rows)
    {
        DataCopyExtParams copy{static_cast<uint16_t>(rows), rowBytes, (kvHeads_ - 1) * rowBytes, 0, 0};
        DataCopyPad(bytes_, src, copy, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        const uint32_t rowStride = (rowBytes + 31) / 32 * 32;
        const uint32_t count = ROWS * rowStride;
        if (rows < ROWS) {
            for (uint32_t i = rows * rowStride; i < count; ++i) {
                bytes_.SetValue(i, 0);
            }
            PipeBarrier<PIPE_ALL>();
        }
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
        Mul(a_, a_, scale_, 64, ROWS, {1, 1, 1, 16, 16, 0});
        Mul(a_[64], a_[64], scale_[64], 64, ROWS, {1, 1, 1, 16, 16, 0});
        PipeBarrier<PIPE_V>();
        Cast(c_.ReinterpretCast<int32_t>(), a_, RoundMode::CAST_FLOOR, ELEMS);
        Duplicate(b_.ReinterpretCast<int32_t>(), static_cast<int32_t>((1U << bits_) - 1), ELEMS);
        PipeBarrier<PIPE_V>();
        And(c_.ReinterpretCast<uint16_t>(), c_.ReinterpretCast<uint16_t>(), b_.ReinterpretCast<uint16_t>(), ELEMS * 2);
        PipeBarrier<PIPE_V>();
        ShiftLeft(c_.ReinterpretCast<uint32_t>(), c_.ReinterpretCast<uint32_t>(), 2U, ELEMS);
        PipeBarrier<PIPE_V>();
        Gather(primary_, centroids_, c_.ReinterpretCast<uint32_t>(), 0, ELEMS);
        PipeBarrier<PIPE_ALL>();
        LoadBytes(sign_[row * 16], 16, rows);
        Gather(a_, bytesFloat_, offsetSign_, 0, ELEMS);
        PipeBarrier<PIPE_V>();
        Mul(a_, a_, scaleSign_, 64, ROWS, {1, 1, 1, 16, 16, 0});
        Mul(a_[64], a_[64], scaleSign_[64], 64, ROWS, {1, 1, 1, 16, 16, 0});
        PipeBarrier<PIPE_V>();
        Modulo(2);
        Muls(signs_, a_, 2.0F, ELEMS);
        PipeBarrier<PIPE_V>();
        Adds(signs_, signs_, -1.0F, ELEMS);
        // A contiguous span replaces ROWS tiny strided transfers when it fits.
        // Stop at the last selected head: do not read past the physical cache.
        const uint32_t normStride = kvHeads_ <= 16 ? kvHeads_ : 16;
        DataCopyExtParams copy = kvHeads_ <= 16 ?
                                     DataCopyExtParams{1, ((rows - 1) * kvHeads_ + 1) * 2, 0, 0, 0} :
                                     DataCopyExtParams{static_cast<uint16_t>(rows), 2, (kvHeads_ - 1) * 2, 0, 0};
        DataCopyPad(normRaw_, norm_[row], copy, DataCopyPadExtParams<NormType>{false, 0, 0, 0});
        DataCopyPad(gammaRaw_, gamma_[row], copy, DataCopyPadExtParams<NormType>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        if (rows < ROWS) {
            for (uint32_t i = rows * normStride; i < ROWS * normStride; ++i) {
                normRaw_.SetValue(i, static_cast<NormType>(0.0F));
                gammaRaw_.SetValue(i, static_cast<NormType>(0.0F));
            }
            PipeBarrier<PIPE_ALL>();
        }
        Cast(normFloat_, normRaw_, RoundMode::CAST_NONE, ROWS * normStride);
        Cast(gammaFloat_, gammaRaw_, RoundMode::CAST_NONE, ROWS * normStride);
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
        Add(a_, a_, a_[64], 64, ROWS, {1, 1, 1, 16, 16, 16});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(result, a_, 64, ROWS, 1, 1, 16);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadQueries(uint32_t first, uint32_t tokens, uint32_t head)
    {
        const uint32_t groups = qHeads_ / kvHeads_;
        const uint64_t offset = (static_cast<uint64_t>(first) * qHeads_ + head * groups) * DIM;
        // Decode arithmetic scratch is dead until Dot; use it for the
        // batched BF16 loads before converting into persistent query caches.
        auto raw = a_.ReinterpretCast<bfloat16_t>();
        auto residualRaw = b_.ReinterpretCast<bfloat16_t>();
        DataCopyExtParams copy{static_cast<uint16_t>(tokens), groups * DIM * 2, (qHeads_ - groups) * DIM * 2, 0, 0};
        DataCopyPad(raw, q_[offset], copy, DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        DataCopyPad(residualRaw, qr_[offset], copy, DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Cast(queryCache_, raw, RoundMode::CAST_NONE, tokens * groups * DIM);
        Cast(queryResidualCache_, residualRaw, RoundMode::CAST_NONE, tokens * groups * DIM);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeGroup(uint32_t slot, uint32_t groups)
    {
        for (uint32_t g = 0; g < groups; ++g) {
            Dot(primary_, queryCache_[(slot + g) * DIM], reducedMain_[g * ROWS]);
            Dot(signs_, queryResidualCache_[(slot + g) * DIM], reducedSign_[g * ROWS]);
        }
        Mul(output_, reducedMain_, normCompact_, ROWS, groups, {1, 1, 1, ROWS / 8, ROWS / 8, 0});
        Mul(b_, reducedSign_, gammaCompact_, ROWS, groups, {1, 1, 1, ROWS / 8, ROWS / 8, 0});
        PipeBarrier<PIPE_V>();
        Muls(b_, b_, COEFFICIENT, groups * ROWS);
        PipeBarrier<PIPE_V>();
        Add(output_, output_, b_, groups * ROWS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint64_t row, uint32_t slot, const LocalTensor<float> &result)
    {
        if (qHeads_ / kvHeads_ <= 4) {
            Dot(primary_, queryCache_[slot * DIM], reducedMain_);
            Dot(signs_, queryResidualCache_[slot * DIM], reducedSign_);
        } else {
            DataCopy(queryRaw_, q_[row * DIM], DIM);
            DataCopy(queryResidualRaw_, qr_[row * DIM], DIM);
            PipeBarrier<PIPE_ALL>();
            Cast(query_, queryRaw_, RoundMode::CAST_NONE, 2 * DIM);
            PipeBarrier<PIPE_V>();
            Dot(primary_, query_, reducedMain_);
            Dot(signs_, queryResidual_, reducedSign_);
        }
        Mul(result, reducedMain_, normCompact_, ROWS);
        Mul(b_, reducedSign_, gammaCompact_, ROWS);
        PipeBarrier<PIPE_V>();
        Muls(b_, b_, COEFFICIENT, ROWS);
        PipeBarrier<PIPE_V>();
        Add(result, result, b_, ROWS);
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
    LocalTensor<uint32_t> normOffsets_;
    LocalTensor<float> normCompact_, gammaCompact_;
    LocalTensor<float> queryCache_, queryResidualCache_;
    LocalTensor<half> bytesHalf_;
    LocalTensor<uint8_t> bytes_;
    LocalTensor<bfloat16_t> queryRaw_, queryResidualRaw_;
    LocalTensor<NormType> normRaw_, gammaRaw_;
    uint32_t tokens_, qHeads_, kvHeads_, batch_, blocks_, width_, maxLen_, bits_;
};
} // namespace TurboQuant
#endif
