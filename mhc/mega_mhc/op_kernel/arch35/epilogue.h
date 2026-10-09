/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_EPILOGUE_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_EPILOGUE_H

#include "compute/mhc_pre.h"
#include "compute/rms_norm_1.h"
#include "mega_mhc_kernel_context.h"

namespace MegaMhc::Schedule {
using namespace AscendC;

/** Load consecutive rows from each projection group; UB layout is [group, planned rows, stride]. */
template <typename T, bool WAIT_READY = true>
__aicore__ inline void LoadPrePartialRows(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, uint64_t row,
                                          uint32_t rows, const LocalTensor<float> &partialBuffer)
{
    const uint64_t gap = static_cast<uint64_t>(tiling.totalRows - rows) * tiling.fixpipeRowStride * sizeof(float);
    if (gap <= 0xFFFFFFFFULL) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyExtParams copyParams{static_cast<uint16_t>(tiling.partialGroupCount),
                                     static_cast<uint32_t>(rows * tiling.fixpipeRowStride * sizeof(float)),
                                     static_cast<uint32_t>(gap),
                                     (tiling.reduceRows - rows) * tiling.fixpipeRowStride / 8U, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(partialBuffer, tensors.partialMm[row * tiling.fixpipeRowStride], copyParams, padParams);
        if constexpr (WAIT_READY) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    } else {
        for (uint32_t group = 0; group < tiling.partialGroupCount; ++group) {
            Kernel::CopyGlobalToUb<float>(
                partialBuffer[group * tiling.reduceRows * tiling.fixpipeRowStride], tensors.partialMm,
                (static_cast<uint64_t>(group) * tiling.totalRows + row) * tiling.fixpipeRowStride,
                rows * tiling.fixpipeRowStride);
        }
    }
}

/** Load all RMS shards once per row batch; each shard retains its increasing-index ownership. */
template <typename T>
__aicore__ inline void PrepareEpilogueRows(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, uint64_t row,
                                           uint32_t rows, const LocalTensor<float> &partial,
                                           const LocalTensor<float> &rms, const LocalTensor<float> &coefficients,
                                           const LocalTensor<float> &alpha, const LocalTensor<float> &bias)
{
    LoadPrePartialRows<T, false>(tensors, tiling, row, rows, partial);
    const uint64_t gap = static_cast<uint64_t>(tiling.totalRows - rows) * sizeof(float);
    if (gap <= 0xFFFFFFFFULL) {
        // One DMA covers all K shards; destination rows retain the host-planned aligned stride.
        const uint32_t paddedRows = (rows + 7U) / 8U * 8U;
        DataCopyExtParams params{static_cast<uint16_t>(tiling.kGrid), static_cast<uint32_t>(rows * sizeof(float)),
                                 static_cast<uint32_t>(gap), (tiling.reduceRowsAligned - paddedRows) / 8U, 0U};
        DataCopyPadExtParams<float> padding{true, 0U, static_cast<uint8_t>(paddedRows - rows), 0.0f};
        DataCopyPad(rms, tensors.partialRms[row], params, padding);
    } else {
        for (uint32_t shard = 0; shard < tiling.kGrid; ++shard) {
            Kernel::CopyGlobalToUb<float>(rms[shard * tiling.reduceRowsAligned], tensors.partialRms,
                                          static_cast<uint64_t>(shard) * tiling.totalRows + row, rows);
        }
    }
    // Both projection and RMS partials are ready before the first reduction VF.
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Compute::NormalizeProjectionRows(
        reinterpret_cast<__ubuf__ float *>(partial.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(rms.GetPhyAddr()),
        rows, tiling.fixpipeRowStride, tiling.reduceRows * tiling.fixpipeRowStride, tiling.partialGroupCount,
        tiling.reduceRowsAligned, tiling.kGrid, tiling.k, tiling.mhcPreNormEps);
    Compute::ComputeGateRows(reinterpret_cast<__ubuf__ float *>(partial.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ float *>(alpha.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ float *>(bias.GetPhyAddr()), tiling.hcEps, rows,
                             tiling.fixpipeRowStride, tiling.reduceRowsAligned,
                             reinterpret_cast<__ubuf__ float *>(coefficients.GetPhyAddr()));
    Kernel::CopyUbToGlobal<float>(tensors.hPostNew, row * 4U, coefficients[tiling.reduceRowsAligned * 4U], rows * 4U);
    Kernel::CopyUbToGlobal<float>(tensors.hResNew, row * 16U, coefficients[tiling.reduceRowsAligned * 8U], rows * 16U);
}

/** Mix x2 using precomputed gates, round h_in, then normalize: output=RMSNorm(Cast<T>(sum_n pre[n]*x2[n])). */
template <typename T>
__aicore__ inline void FinalizeMhcRow(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, uint64_t row,
                                      const LocalTensor<T> &x2Row, const LocalTensor<T> &hIn,
                                      const LocalTensor<T> &hInPrime, const LocalTensor<float> &gamma1,
                                      const LocalTensor<float> &pre, const LocalTensor<float> &statsBuffer)
{
    __ubuf__ float *stats = reinterpret_cast<__ubuf__ float *>(statsBuffer.GetPhyAddr());
    Kernel::CopyGlobalToUb<T>(x2Row, tensors.x2Out, row * tiling.k, tiling.k);
    Compute::ComputeHInTile<T>(reinterpret_cast<__ubuf__ float *>(pre.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ T *>(x2Row.GetPhyAddr()), tiling.d, 0U, tiling.d,
                               reinterpret_cast<__ubuf__ T *>(hIn.GetPhyAddr()), stats);
    Compute::ComputeRmsNorm1FromSquares<T>(
        reinterpret_cast<__ubuf__ T *>(hIn.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(gamma1.GetPhyAddr()),
        tiling.d, tiling.normEps, stats, stats + 1, reinterpret_cast<__ubuf__ T *>(hInPrime.GetPhyAddr()));
    Kernel::CopyUbToGlobal<T>(tensors.hInPrime, row * tiling.d, hInPrime, tiling.d);
}

/** Issue one complete x2 row load into an available ping-pong slot; x2[r,n,d] stays in input dtype. */
template <typename T>
__aicore__ inline void PrefetchEpilogueRow(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, uint64_t row,
                                           const LocalTensor<T> &buffer, event_t event)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopyExtParams params{1, static_cast<uint32_t>(tiling.k * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, 0};
    DataCopyPad(buffer, tensors.x2Out[row * tiling.k], params, padding);
    SetFlag<HardEvent::MTE2_V>(event);
}

/** Finalize a precomputed gate batch while overlapping next-row loads and previous-row output stores.
 * Formula: y[r]=RMSNorm(Cast<T>(sum_n pre[r,n]*x2[r,n])); both explicit input-type roundings remain.
 * The host enables reuse of dead partial storage only when it fits a complete x2 row.
 */
template <typename T>
__aicore__ inline void FinalizeMhcRows(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, uint64_t start,
                                       uint32_t rows, const LocalTensor<T> &x2, const LocalTensor<T> &hIn,
                                       const LocalTensor<T> &hInPrime, const LocalTensor<float> &gamma,
                                       const LocalTensor<float> &partial, const LocalTensor<float> &coefficients,
                                       const LocalTensor<float> &statsBuffer)
{
    if (tiling.epiloguePrefetch == 0U) {
        for (uint32_t row = 0; row < rows; ++row) {
            FinalizeMhcRow(tensors, tiling, start + row, x2, hIn, hInPrime, gamma, coefficients[row * 4U], statsBuffer);
        }
        return;
    }
    LocalTensor<T> alternate = partial.template ReinterpretCast<T>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
    PrefetchEpilogueRow(tensors, tiling, start, x2, EVENT_ID4);
    __ubuf__ float *stats = reinterpret_cast<__ubuf__ float *>(statsBuffer.GetPhyAddr());
    for (uint32_t row = 0; row < rows; ++row) {
        const event_t event = (row & 1U) == 0U ? EVENT_ID4 : EVENT_ID5;
        const event_t nextEvent = (row & 1U) == 0U ? EVENT_ID5 : EVENT_ID4;
        const LocalTensor<T> input = (row & 1U) == 0U ? x2 : alternate;
        const LocalTensor<T> next = (row & 1U) == 0U ? alternate : x2;
        const LocalTensor<T> output = (row & 1U) == 0U ? hIn : hInPrime;
        WaitFlag<HardEvent::MTE2_V>(event);
        if (row + 1U < rows) {
            PrefetchEpilogueRow(tensors, tiling, start + row + 1U, next, nextEvent);
        }
        WaitFlag<HardEvent::MTE3_V>(event);
        Compute::ComputeHInTile<T>(reinterpret_cast<__ubuf__ float *>(coefficients.GetPhyAddr()) + row * 4U,
                                   reinterpret_cast<__ubuf__ T *>(input.GetPhyAddr()), tiling.d, 0U, tiling.d,
                                   reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()), stats);
        SetFlag<HardEvent::V_MTE2>(event);
        Compute::ComputeRmsNorm1FromSquares<T>(
            reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(gamma.GetPhyAddr()),
            tiling.d, tiling.normEps, stats, stats + 1, reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        DataCopyExtParams params{1, static_cast<uint32_t>(tiling.d * sizeof(T)), 0, 0, 0};
        DataCopyPad(tensors.hInPrime[(start + row) * tiling.d], output, params);
        SetFlag<HardEvent::MTE3_V>(event);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
}

/** Assign complete final rows to AIVs after all Cube and statistic partials are globally visible. */
template <typename T>
__aicore__ inline void RunEpilogue(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> x2Buffer;
        TBuf<TPosition::VECCALC> hInBuffer;
        TBuf<TPosition::VECCALC> hInPrimeBuffer;
        TBuf<TPosition::VECCALC> gamma1Buffer;
        TBuf<TPosition::VECCALC> partialBuffer;
        TBuf<TPosition::VECCALC> rmsPartialBuffer;
        TBuf<TPosition::VECCALC> coefficientBuffer;
        TBuf<TPosition::VECCALC> alphaBuffer;
        TBuf<TPosition::VECCALC> biasBuffer;
        TBuf<TPosition::VECCALC> statsBuffer;
        Kernel::InitUbBuffer(pipe, x2Buffer, tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInPrimeBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, gamma1Buffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(
            pipe, partialBuffer,
            (tiling.partialGroupCount * tiling.reduceRows * tiling.fixpipeRowStride + 64U) * sizeof(float));
        Kernel::InitUbBuffer(pipe, rmsPartialBuffer, tiling.reduceRowsAligned * tiling.kGrid * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer, tiling.reduceRowsAligned * 24U * sizeof(float));
        Kernel::InitUbBuffer(pipe, alphaBuffer, 4U * sizeof(float));
        Kernel::InitUbBuffer(pipe, biasBuffer, 32U * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, 4U * sizeof(float));

        LocalTensor<T> x2 = x2Buffer.Get<T>();
        LocalTensor<T> hIn = hInBuffer.Get<T>();
        LocalTensor<T> hInPrime = hInPrimeBuffer.Get<T>();
        LocalTensor<float> gamma1 = gamma1Buffer.Get<float>();
        LocalTensor<float> partial = partialBuffer.Get<float>();
        LocalTensor<float> rmsPartials = rmsPartialBuffer.Get<float>();
        LocalTensor<float> coefficients = coefficientBuffer.Get<float>();
        LocalTensor<float> alpha = alphaBuffer.Get<float>();
        LocalTensor<float> bias = biasBuffer.Get<float>();
        LocalTensor<float> stats = statsBuffer.Get<float>();
        Kernel::CopyGlobalToUb<float>(gamma1, tensors.gamma1, 0, tiling.d);
        Kernel::CopyGlobalToUb<float>(alpha, tensors.alpha, 0, 3U);
        Kernel::CopyGlobalToUb<float>(bias, tensors.bias, 0, tiling.q);

        const uint32_t aivIndex = GetBlockIdx();
        for (uint64_t start = static_cast<uint64_t>(aivIndex) * tiling.reduceRows; start < tiling.totalRows;
             start += static_cast<uint64_t>(tiling.activeAiv) * tiling.reduceRows) {
            const uint32_t rows = tiling.totalRows - start < tiling.reduceRows ?
                                      static_cast<uint32_t>(tiling.totalRows - start) :
                                      tiling.reduceRows;
            PrepareEpilogueRows(tensors, tiling, start, rows, partial, rmsPartials, coefficients, alpha, bias);
            FinalizeMhcRows(tensors, tiling, start, rows, x2, hIn, hInPrime, gamma1, partial, coefficients, stats);
        }
    }
}

template <typename T>
class WaveEpilogue {
public:
    /** Allocate the host-budgeted epilogue state and load immutable parameters once. */
    __aicore__ inline void Init(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &sourceTiling, TPipe *pipe)
    {
        plan_ = &sourceTiling;
        const auto &tiling = *plan_;
        Kernel::InitUbBuffer(pipe, x2Buffer, tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInPrimeBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, gamma1Buffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(
            pipe, partialBuffer,
            (tiling.partialGroupCount * tiling.reduceRows * tiling.fixpipeRowStride + 64U) * sizeof(float));
        Kernel::InitUbBuffer(pipe, rmsPartialBuffer, tiling.reduceRowsAligned * tiling.kGrid * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer, tiling.reduceRowsAligned * 24U * sizeof(float));
        Kernel::InitUbBuffer(pipe, alphaBuffer, 4U * sizeof(float));
        Kernel::InitUbBuffer(pipe, biasBuffer, 32U * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, 4U * sizeof(float));

        x2 = x2Buffer.Get<T>();
        hIn = hInBuffer.Get<T>();
        hInPrime = hInPrimeBuffer.Get<T>();
        gamma1 = gamma1Buffer.Get<float>();
        partial = partialBuffer.Get<float>();
        rmsPartials = rmsPartialBuffer.Get<float>();
        coefficients = coefficientBuffer.Get<float>();
        alpha = alphaBuffer.Get<float>();
        bias = biasBuffer.Get<float>();
        stats = statsBuffer.Get<float>();
        // Immutable parameters have disjoint destinations: issue all loads before one readiness wait.
        Kernel::EnqueueGlobalToUb(gamma1, tensors.gamma1, 0U, tiling.d);
        Kernel::EnqueueGlobalToUb(alpha, tensors.alpha, 0U, 3U);
        Kernel::EnqueueGlobalToUb(bias, tensors.bias, 0U, tiling.q);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    /** Bind complete rows of a globally completed wave; partials are read in host-sized batches. */
    __aicore__ inline void Begin(uint32_t wave)
    {
        const auto &t = *plan_;
        const uint32_t member = GetBlockIdx();
        first_ = static_cast<uint64_t>(wave) * t.rowsPerWave +
                 static_cast<uint64_t>(member / t.waveGroupSize) * t.mTile +
                 (member % t.waveGroupSize) * t.waveProducerRows;
        count_ = wave + 1U != t.mWaveCount || member < t.lastWaveFullProducers ?
                     t.waveProducerRows :
                     (member == t.lastWaveFullProducers ? t.lastWaveProducerTailRows : 0U);
        done_ = 0U;
        batchStart_ = 0U;
        batchRows_ = 0U;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
    }

    /** Advance one output row: gates=Pre(partials), output=RMS1(Cast(sum(pre*x2))). */
    __aicore__ inline void Tick(Kernel::Tensors<T> &tensors)
    {
        if (done_ >= count_) {
            return;
        }
        if (done_ == batchStart_ + batchRows_) {
            batchStart_ = done_;
            batchRows_ = count_ - done_ < plan_->reduceRows ? count_ - done_ : plan_->reduceRows;
            PrepareEpilogueRows(tensors, *plan_, first_ + done_, batchRows_, partial, rmsPartials, coefficients, alpha,
                                bias);
        }
        if (done_ == 0U) {
            PrefetchEpilogueRow(tensors, *plan_, first_, x2, EVENT_ID1);
        }
        const event_t outputEvent = (done_ & 1U) == 0U ? EVENT_ID4 : EVENT_ID5;
        const auto output = (done_ & 1U) == 0U ? hIn : hInPrime;
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(outputEvent);
        auto statsPtr = reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr());
        Compute::ComputeHInTile<T>(
            reinterpret_cast<__ubuf__ float *>(coefficients.GetPhyAddr()) + (done_ - batchStart_) * 4U,
            reinterpret_cast<__ubuf__ T *>(x2.GetPhyAddr()), plan_->d, 0U, plan_->d,
            reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()), statsPtr);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        // The next x2 load overlaps RMS1 and the following projection work; input is no longer read by RMS1.
        if (done_ + 1U < count_) {
            PrefetchEpilogueRow(tensors, *plan_, first_ + done_ + 1U, x2, EVENT_ID1);
        }
        Compute::ComputeRmsNorm1FromSquares<T>(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float *>(gamma1.GetPhyAddr()), plan_->d,
                                               plan_->normEps, statsPtr, statsPtr + 1U,
                                               reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()));
        SetFlag<HardEvent::V_MTE3>(outputEvent);
        WaitFlag<HardEvent::V_MTE3>(outputEvent);
        DataCopyExtParams copy{1U, static_cast<uint32_t>(plan_->d * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(tensors.hInPrime[(first_ + done_) * plan_->d], output, copy);
        SetFlag<HardEvent::MTE3_V>(outputEvent);
        ++done_;
    }

    /** Drain any remaining rows before reusing the epilogue state or leaving the final wave. */
    __aicore__ inline void Finish(Kernel::Tensors<T> &tensors)
    {
        while (done_ < count_) {
            Tick(tensors);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
    }

private:
    const MegaMhcTilingData *plan_ = nullptr;
    uint64_t first_;
    uint32_t count_, done_, batchStart_, batchRows_;
    TBuf<TPosition::VECCALC> x2Buffer, hInBuffer, hInPrimeBuffer, gamma1Buffer, partialBuffer;
    TBuf<TPosition::VECCALC> rmsPartialBuffer, coefficientBuffer, alphaBuffer, biasBuffer, statsBuffer;
    LocalTensor<T> x2, hIn, hInPrime;
    LocalTensor<float> gamma1, partial, rmsPartials, coefficients, alpha, bias, stats;
};

} // namespace MegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_EPILOGUE_H
