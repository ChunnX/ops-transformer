/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is distributed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_MK_FIVE_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_MK_FIVE_H

#include "../compute/mhc_post.h"
#include "../compute/rms_norm_0.h"
#include "../compute/rms_norm_mid.h"
#include "../mega_mhc_kernel_context.h"
#include "mk_projection.h"

namespace AiInfraMegaMhc::Schedule {
using namespace AscendC;

/** Enqueue one complete producer input row: hOut=x[row], residual=r[row], coefficients=H[row].
 * The caller waits for the slot's V_MTE2 credit; MTE2_V publishes all inputs together.
 */
template <typename T>
__aicore__ inline void PrefetchStagedProducer(Kernel::Tensors<T> &tensors, uint64_t row,
    const MegaMhcTilingData &tiling, const LocalTensor<T> &hOut, const LocalTensor<T> &residual,
    const LocalTensor<float> &coefficients, event_t event)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    Kernel::EnqueueGlobalToUb(hOut, tensors.hOut, row * tiling.d, tiling.d);
    Kernel::EnqueueGlobalToUb(residual, tensors.residual, row * tiling.k, tiling.k);
    Kernel::EnqueueGlobalToUb(coefficients, tensors.hPost, row * 4U, 4U);
    Kernel::EnqueueGlobalToUb(coefficients[64U], tensors.hRes, row * 16U, 16U);
    SetFlag<HardEvent::MTE2_V>(event);
}

/** Produce x2[row]=Post(Cast<T>(RMS0(hOut[row])), residual[row]) without Mid.
 * Two complete input slots overlap next-row DMA with current-row RegBase computation.
 * Post output retains its input-type rounding before GM publication; the phase barrier remains.
 */
template <typename T>
__aicore__ inline void RunFourStagedProducer(Kernel::Tensors<T> &tensors,
    const MegaMhcTilingData &tiling, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> inputBuffer, residualBuffer, x1Buffer, postBuffer;
        TBuf<TPosition::VECCALC> gammaBuffer, coefficientBuffer, statsBuffer;
        Kernel::InitUbBuffer(pipe, inputBuffer, 2U * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, residualBuffer, 2U * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, x1Buffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, postBuffer, tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, gammaBuffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer, 256U * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, 2U * sizeof(float));
        auto input = inputBuffer.Get<T>();
        auto residual = residualBuffer.Get<T>();
        auto x1 = x1Buffer.Get<T>();
        auto post = postBuffer.Get<T>();
        auto gamma = gammaBuffer.Get<float>();
        auto coefficients = coefficientBuffer.Get<float>();
        auto stats = statsBuffer.Get<float>();
        Kernel::CopyGlobalToUb(gamma, tensors.gamma0, 0U, tiling.d);
        const uint32_t firstRow = GetBlockIdx();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
        if (firstRow < tiling.totalRows) {
            PrefetchStagedProducer(tensors, firstRow, tiling, input, residual, coefficients, EVENT_ID4);
        }
        uint32_t slot = 0U;
        for (uint64_t row = firstRow; row < tiling.totalRows; row += tiling.activeAiv) {
            const event_t event = slot == 0U ? EVENT_ID4 : EVENT_ID5;
            WaitFlag<HardEvent::MTE2_V>(event);
            const uint32_t next = slot ^ 1U;
            if (row + tiling.activeAiv < tiling.totalRows) {
                PrefetchStagedProducer(tensors, row + tiling.activeAiv, tiling,
                    input[next * tiling.d], residual[next * tiling.k], coefficients[next * 128U],
                    next == 0U ? EVENT_ID4 : EVENT_ID5);
            }
            Compute::ComputeRmsNorm0Row<T>(reinterpret_cast<__ubuf__ T *>(input[slot * tiling.d].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(gamma.GetPhyAddr()), tiling.d, tiling.normEps,
                reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()) + 1U,
                reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T *>(x1.GetPhyAddr()));
            Compute::ComputePostFourHeads<T>(reinterpret_cast<__ubuf__ T *>(x1.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T *>(residual[slot * tiling.k].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(coefficients[slot * 128U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(coefficients[slot * 128U + 64U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ T *>(post.GetPhyAddr()), 1U, tiling.d, 0U, tiling.d);
            SetFlag<HardEvent::V_MTE2>(event);
            Kernel::CopyUbToGlobal(tensors.x2Out, row * tiling.k, post, tiling.k);
            slot = next;
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    }
}

/** Produce x2[row]=Post(Cast<T>(RMS0(hOut[row])), residual[row]) followed by RMSMid, with both input-type stage roundings.
 * Two complete input slots overlap next-row DMA with current-row RegBase computation.
 * Post output retains its input-type rounding before GM publication; the phase barrier remains.
 */
template <typename T>
__aicore__ inline void RunPrefetchedFiveProducer(Kernel::Tensors<T> &tensors,
    const MegaMhcTilingData &tiling, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> inputBuffer, residualBuffer, x1Buffer, gamma2Buffer;
        TBuf<TPosition::VECCALC> gammaBuffer, coefficientBuffer, statsBuffer;
        Kernel::InitUbBuffer(pipe, inputBuffer, 2U * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, residualBuffer, 2U * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, x1Buffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, gamma2Buffer, tiling.k * sizeof(float));
        Kernel::InitUbBuffer(pipe, gammaBuffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer, 256U * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, 2U * sizeof(float));
        auto input = inputBuffer.Get<T>();
        auto residual = residualBuffer.Get<T>();
        auto x1 = x1Buffer.Get<T>();
        auto gamma2 = gamma2Buffer.Get<float>();
        auto gamma = gammaBuffer.Get<float>();
        auto coefficients = coefficientBuffer.Get<float>();
        auto stats = statsBuffer.Get<float>();
        Kernel::CopyGlobalToUb(gamma, tensors.gamma0, 0U, tiling.d);
        Kernel::CopyGlobalToUb(gamma2, tensors.gamma2, 0U, tiling.k);
        const uint32_t firstRow = GetBlockIdx();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
        if (firstRow < tiling.totalRows) {
            PrefetchStagedProducer(tensors, firstRow, tiling, input, residual, coefficients, EVENT_ID4);
        }
        uint32_t slot = 0U;
        for (uint64_t row = firstRow; row < tiling.totalRows; row += tiling.activeAiv) {
            const event_t event = slot == 0U ? EVENT_ID4 : EVENT_ID5;
            WaitFlag<HardEvent::MTE2_V>(event);
            const uint32_t next = slot ^ 1U;
            if (row + tiling.activeAiv < tiling.totalRows) {
                PrefetchStagedProducer(tensors, row + tiling.activeAiv, tiling,
                    input[next * tiling.d], residual[next * tiling.k], coefficients[next * 128U],
                    next == 0U ? EVENT_ID4 : EVENT_ID5);
            }
            Compute::ComputeRmsNorm0Row<T>(reinterpret_cast<__ubuf__ T *>(input[slot * tiling.d].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(gamma.GetPhyAddr()), tiling.d, tiling.normEps,
                reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()) + 1U,
                reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T *>(x1.GetPhyAddr()));
            auto post = residual[slot * tiling.k];
            Compute::ComputePostFourHeads<T, true>(reinterpret_cast<__ubuf__ T *>(x1.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T *>(residual[slot * tiling.k].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(coefficients[slot * 128U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(coefficients[slot * 128U + 64U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ T *>(post.GetPhyAddr()), 1U, tiling.d, 0U, tiling.d,
                reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()));
            Compute::ComputeRmsNormMidFromSquares<T>(reinterpret_cast<__ubuf__ T *>(post.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(gamma2.GetPhyAddr()), tiling.k, tiling.normEps,
                reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()) + 1U,
                reinterpret_cast<__ubuf__ T *>(post.GetPhyAddr()));
            Kernel::CopyUbToGlobal(tensors.x2Out, row * tiling.k, post, tiling.k);
            // Residual is also the output: return input credit after MTE3 completes.
            SetFlag<HardEvent::V_MTE2>(event);
            slot = next;
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    }
}

/** Produce each complete Post->RMSMid row before the five-op m-k projection phase. */
template <typename T, bool WITH_MID = true>
__aicore__ inline void RunFiveProducer(Kernel::Tensors<T> &tensors,
                                       const MegaMhcTilingData &tiling, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        const uint32_t producerRows = tiling.postRows;
        TBuf<TPosition::VECCALC> hOutBuffer;
        TBuf<TPosition::VECCALC> x1Buffer;
        TBuf<TPosition::VECCALC> residualBuffer;
        TBuf<TPosition::VECCALC> postBuffer;
        TBuf<TPosition::VECCALC> gamma0Buffer;
        TBuf<TPosition::VECCALC> gamma2Buffer;
        TBuf<TPosition::VECCALC> hPostBuffer;
        TBuf<TPosition::VECCALC> hResBuffer;
        TBuf<TPosition::VECCALC> statsBuffer;
        Kernel::InitUbBuffer(pipe, hOutBuffer, producerRows * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, x1Buffer, producerRows * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, residualBuffer, producerRows * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, postBuffer, producerRows * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, gamma0Buffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(pipe, gamma2Buffer, tiling.k * sizeof(float));
        Kernel::InitUbBuffer(pipe, hPostBuffer, producerRows * 4U * sizeof(float));
        Kernel::InitUbBuffer(pipe, hResBuffer, producerRows * 16U * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, producerRows * 2U * sizeof(float));

        LocalTensor<T> hOut = hOutBuffer.Get<T>();
        LocalTensor<T> x1 = x1Buffer.Get<T>();
        LocalTensor<T> residual = residualBuffer.Get<T>();
        LocalTensor<T> postRow = postBuffer.Get<T>();
        LocalTensor<float> gamma0 = gamma0Buffer.Get<float>();
        LocalTensor<float> gamma2 = gamma2Buffer.Get<float>();
        LocalTensor<float> hPost = hPostBuffer.Get<float>();
        LocalTensor<float> hRes = hResBuffer.Get<float>();
        LocalTensor<float> stats = statsBuffer.Get<float>();
        __ubuf__ float *statsPtr = reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr());
        Kernel::CopyGlobalToUb<float>(gamma0, tensors.gamma0, 0, tiling.d);
        if constexpr (WITH_MID) {
            Kernel::CopyGlobalToUb<float>(gamma2, tensors.gamma2, 0, tiling.k);
        }

        const uint32_t aivIndex = GetBlockIdx();
        __ubuf__ T *hOutPtr = reinterpret_cast<__ubuf__ T *>(hOut.GetPhyAddr());
        __ubuf__ T *x1Ptr = reinterpret_cast<__ubuf__ T *>(x1.GetPhyAddr());
        __ubuf__ T *residualPtr = reinterpret_cast<__ubuf__ T *>(residual.GetPhyAddr());
        __ubuf__ T *postRowPtr = reinterpret_cast<__ubuf__ T *>(postRow.GetPhyAddr());
        __ubuf__ float *hPostPtr = reinterpret_cast<__ubuf__ float *>(hPost.GetPhyAddr());
        __ubuf__ float *hResPtr = reinterpret_cast<__ubuf__ float *>(hRes.GetPhyAddr());
        for (uint64_t batchStart = static_cast<uint64_t>(aivIndex) * producerRows; batchStart < tiling.totalRows;
             batchStart += tiling.activeAiv * producerRows) {
            const uint32_t batchRows = tiling.totalRows - batchStart < producerRows ?
                                           tiling.totalRows - batchStart : producerRows;
            Kernel::CopyGlobalToUb<float>(hPost, tensors.hPost, batchStart * 4U, batchRows * 4U);
            for (uint32_t batchRow = 0; batchRow < batchRows; ++batchRow) {
                const uint64_t row = batchStart + batchRow;
                Kernel::LoadProducerRow<T, false>(tensors, row, tiling.d,
                    hOut[batchRow * tiling.d], residual[batchRow * tiling.k],
                    hPost[batchRow * 4U], hRes[batchRow * 16U]);
                Compute::ComputeRmsNorm0Row<T>(hOutPtr + static_cast<uint64_t>(batchRow) * tiling.d,
                    reinterpret_cast<__ubuf__ float *>(gamma0.GetPhyAddr()), tiling.d, tiling.normEps,
                    statsPtr + batchRow * 2U + 1U, statsPtr + batchRow * 2U, x1Ptr + static_cast<uint64_t>(batchRow) * tiling.d);
            }
            // Keep each Post invocation row-local; the scheduler owns batching.
            for (uint32_t r = 0; r < batchRows; ++r) {
                Compute::ComputePostFourHeads<T, WITH_MID>(x1Ptr + r * tiling.d, residualPtr + r * tiling.k,
                    hPostPtr + r * 4U, hResPtr + r * 16U, postRowPtr + r * tiling.k,
                    1U, tiling.d, 0U, tiling.d, statsPtr + r * 2U);
            }
            for (uint32_t batchRow = 0; batchRow < batchRows; ++batchRow) {
                __ubuf__ T *row = postRowPtr + static_cast<uint64_t>(batchRow) * tiling.k;
                __ubuf__ float *rowStats = statsPtr + batchRow * 2U;
                if constexpr (WITH_MID) {
                    Compute::ComputeRmsNormMidFromSquares<T>(row,
                        reinterpret_cast<__ubuf__ float *>(gamma2.GetPhyAddr()), tiling.k,
                        tiling.normEps, rowStats, rowStats + 1, row);
                }
                Kernel::CopyUbToGlobal<T>(tensors.x2Out,
                    static_cast<uint64_t>(batchStart + batchRow) * tiling.k,
                    postRow[static_cast<uint64_t>(batchRow) * tiling.k], tiling.k);
            }
        }
    }
}

/** Keep the five-op schedule staged: full Post/Mid, projection, then shared row finalization. */
template <typename T, bool WITH_MID = true>
__aicore__ inline void RunMkFiveSchedule(Kernel::Tensors<T> &tensors,
                                         const MegaMhcTilingData &tiling, TPipe *pipe)
{
    if constexpr (!WITH_MID) {
        if (tiling.wavePipeline != 0U) {
            TPipe initPipe;
            InitWaveEpochs(tensors, &initPipe);
            initPipe.Destroy();
            RunMKScheduler<T, false, Transfer::StagePolicyT<kFiveProjectionTransport>, false, true, true>(
                tensors, tiling, pipe);
            pipe->Destroy();
            SyncAll<false>();
            return;
        }
    }
    if constexpr (WITH_MID) {
        RunPrefetchedFiveProducer<T>(tensors, tiling, pipe);
    } else {
        RunFourStagedProducer<T>(tensors, tiling, pipe);
    }
    pipe->Destroy();
    SyncAll<false>();

    TPipe projectionPipe;
    RunMKScheduler<T, false, Transfer::StagePolicyT<kFiveProjectionTransport>, false, false, true>(
        tensors, tiling, &projectionPipe);
    projectionPipe.Destroy();
    SyncAll<false>();
}

}  // namespace AiInfraMegaMhc::Schedule

#endif  // OPS_TRANSFORMER_MHC_MEGA_MHC_MK_FIVE_H
