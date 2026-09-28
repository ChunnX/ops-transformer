/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_MK_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_MK_H

#include "schedule_common.h"
#include "epilogue.h"

namespace MegaMhc::Schedule {
using namespace AscendC;

// Modes 0/2 share flags: use distinct IDs and avoid SyncAll's reserved IDs 11..14.
// Existing transfer credits use the independent mode-4 flag space.
constexpr uint16_t kMkRowsReadyFlag = 8;
constexpr uint16_t kMkCubeReadyFlag = 9;
constexpr uint16_t kMkProjectionReadyFlag = 10;

/** Publish all AIV stores, then gate subsequent loads on their visibility.
 * Mode 0 collects AIV arrivals only; AIC proceeds independently through paired READY flags.
 * Complete-token redistribution still requires every AIV producer to arrive.
 */
__aicore__ inline void WaitMkRowsReady()
{
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<0, PIPE_MTE3>(kMkRowsReadyFlag);
        CrossCoreWaitFlag<0, PIPE_MTE2>(kMkRowsReadyFlag);
    }
}

/** Publish completed Cube partials to both paired AIVs, then collect all K-shard completions.
 * FIX completion protects partialMm; AIV MTE3 completion protects partialRms.
 * The current epilogue redistributes tokens, so its MTE2 loads need all AIV arrivals.
 */
__aicore__ inline void WaitMkProjectionReady()
{
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<2, PIPE_FIX>(kMkCubeReadyFlag);
    }
    if ASCEND_IS_AIV {
        // Order the MTE3 notification after both local stores and paired Cube completion.
        CrossCoreWaitFlag<2, PIPE_MTE3>(kMkCubeReadyFlag);
        CrossCoreSetFlag<0, PIPE_MTE3>(kMkProjectionReadyFlag);
        CrossCoreWaitFlag<0, PIPE_MTE2>(kMkProjectionReadyFlag);
    }
}

/** Initialize the four-op completion epochs to zero before any group starts producing.
 * Each AIV owns one 128-byte region in otherwise unused RMS0 workspace.
 */
template <typename T>
__aicore__ inline void InitWaveEpochs(Kernel::Tensors<T> &tensors)
{
    TPipe pipe;
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> buffer;
        Kernel::InitUbBuffer(&pipe, buffer, 128U);
        auto scratch = buffer.Get<float>();
        Kernel::ZeroFp32Values(reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr()), 32U);
        Kernel::CopyUbToGlobal(tensors.invRms0, static_cast<uint64_t>(GetBlockIdx()) * 32U, scratch, 32U);
    }
    WaitMkRowsReady();
}

/** Stateful row producer used only by the outer projection schedule.
 * x2=Cast<T>(Post(Cast<T>(RMS0(hOut)),residual)); arithmetic stays in shared RegBase functions.
 * Producer buffers coexist with projection buffers; host owns row counts and tick spacing.
 */
template <typename T>
class WavePostProducer {
public:
    /** Allocate the host-budgeted two input slots and bind one host-planned AIV M group. */
    __aicore__ inline void Init(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, TPipe *pipe)
    {
        tensors_ = &tensors;
        tiling_ = &tiling;
        Kernel::InitUbBuffer(pipe, inputBuffer_, 2U * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, residualBuffer_, 2U * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, x1Buffer_, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, gammaBuffer_, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientsBuffer_, 256U * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer_, 2U * sizeof(float));
        Kernel::InitUbBuffer(pipe, syncBuffer_, 32U);
        input_ = inputBuffer_.template Get<T>();
        residual_ = residualBuffer_.template Get<T>();
        x1_ = x1Buffer_.template Get<T>();
        gamma_ = gammaBuffer_.template Get<float>();
        coefficients_ = coefficientsBuffer_.template Get<float>();
        stats_ = statsBuffer_.template Get<float>();
        sync_ = syncBuffer_.template Get<int32_t>();
        Kernel::CopyGlobalToUb(gamma_, tensors.gamma0, 0U, tiling.d);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
    }

    /** Begin a producer batch: full members own R rows; Host supplies the final member count and tail.
     * Empty tail members skip data access but still publish completion and participate in group synchronization.
     */
    __aicore__ inline void Begin(uint32_t wave)
    {
        const auto &t = *tiling_;
        row_ = static_cast<uint64_t>(wave) * t.rowsPerWave +
               static_cast<uint64_t>(GetBlockIdx() / t.waveGroupSize) * t.mTile +
               (GetBlockIdx() % t.waveGroupSize) * t.waveProducerRows;
        produced_ = 0U;
        slot_ = 0U;
        nextTick_ = 1U;
        const uint32_t member = GetBlockIdx();
        rows_ = wave + 1U != t.mWaveCount || member < t.lastWaveFullProducers ?
                    t.waveProducerRows :
                    (member == t.lastWaveFullProducers ? t.lastWaveProducerTailRows : 0U);
        if (rows_ != 0U) {
            Prefetch(row_, 0U);
        }
    }

    /** Prefetch hOut, residual and coefficients for one row into a free input slot. */
    __aicore__ inline void Prefetch(uint64_t row, uint32_t slot)
    {
        const auto &t = *tiling_;
        const event_t event = slot == 0U ? EVENT_ID4 : EVENT_ID5;
        WaitFlag<HardEvent::MTE3_MTE2>(event);
        Kernel::EnqueueGlobalToUb(input_[slot * t.d], tensors_->hOut, row * t.d, t.d);
        Kernel::EnqueueGlobalToUb(residual_[slot * t.k], tensors_->residual, row * t.k, t.k);
        Kernel::EnqueueGlobalToUb(coefficients_[slot * 128U], tensors_->hPost, row * 4U, 4U);
        Kernel::EnqueueGlobalToUb(coefficients_[slot * 128U + 64U], tensors_->hRes, row * 16U, 16U);
        SetFlag<HardEvent::MTE2_V>(event);
    }

    /** Compute one rounded RMS0/Post row and enqueue x2[row,:]; slot reuse waits for its store. */
    __aicore__ inline void ProduceOne()
    {
        const auto &t = *tiling_;
        const event_t event = slot_ == 0U ? EVENT_ID4 : EVENT_ID5;
        WaitFlag<HardEvent::MTE2_V>(event);
        const uint32_t next = slot_ ^ 1U;
        if (produced_ + 1U < rows_) {
            Prefetch(row_ + 1U, next);
        }
        Compute::ComputeRmsNorm0Row<T>(reinterpret_cast<__ubuf__ T *>(input_[slot_ * t.d].GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ float *>(gamma_.GetPhyAddr()), t.d, t.normEps,
                                       reinterpret_cast<__ubuf__ float *>(stats_.GetPhyAddr()) + 1U,
                                       reinterpret_cast<__ubuf__ float *>(stats_.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ T *>(x1_.GetPhyAddr()));
        Compute::ComputePostFourHeads<T>(
            reinterpret_cast<__ubuf__ T *>(x1_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ T *>(residual_[slot_ * t.k].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(coefficients_[slot_ * 128U].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(coefficients_[slot_ * 128U + 64U].GetPhyAddr()),
            reinterpret_cast<__ubuf__ T *>(residual_[slot_ * t.k].GetPhyAddr()), 1U, t.d, 0U, t.d);
        // All four sources for each column are read before stores in the shared Post VF.
        // Return input-slot credit only after MTE3 finishes reading this aliased output.
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copy{1U, static_cast<uint32_t>(t.k * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(tensors_->x2Out[row_ * t.k], residual_[slot_ * t.k], copy);
        SetFlag<HardEvent::MTE3_MTE2>(event);
        slot_ = next;
        ++row_;
        ++produced_;
    }

    /** Fill the prologue batch completely before the first projection tile can read it. */
    __aicore__ inline void Fill()
    {
        while (produced_ < rows_) {
            ProduceOne();
        }
    }

    /** Interleave a host-sized next-wave row batch after a tile is published to Cube.
     * The spacing is host planned; no tiling search or arithmetic VF branching occurs here.
     */
    __aicore__ inline void Tick(uint32_t tile)
    {
        if (tile >= nextTick_ && produced_ < rows_) {
            for (uint32_t row = 0U; row < tiling_->waveProducerRowsPerTick && produced_ < rows_; ++row) {
                ProduceOne();
            }
            nextTick_ += tiling_->waveProducerTileStride;
        }
    }

    /** Publish completed Post stores, then gate projection loads on all AIV arrivals.
     * Alternate two hardware notifications; immutable x2 rows need no reuse acknowledgement.
     */
    __aicore__ inline void WaitReady(uint32_t wave)
    {
        const uint16_t ready = 6U + (wave & 1U);
        CrossCoreSetFlag<0, PIPE_MTE3>(ready);
        CrossCoreWaitFlag<0, PIPE_MTE2>(ready);
    }

    /** Consume both input-slot credits after the final producer batch to close event lifetimes. */
    __aicore__ inline void End()
    {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
    }

private:
    Kernel::Tensors<T> *tensors_;
    const MegaMhcTilingData *tiling_;
    TBuf<TPosition::VECCALC> inputBuffer_, residualBuffer_, x1Buffer_;
    TBuf<TPosition::VECCALC> gammaBuffer_, coefficientsBuffer_, statsBuffer_, syncBuffer_;
    LocalTensor<T> input_, residual_, x1_;
    LocalTensor<float> gamma_, coefficients_, stats_;
    LocalTensor<int32_t> sync_;
    uint64_t row_;
    uint32_t produced_, slot_, nextTick_, rows_;
};

/** Materialize x2=Post(RMS0(hOut),residual), optionally followed by RMSMid.
 * Phase-local resources are released on return, before projection reuses UB.
 */
template <typename T, bool WITH_MID>
__aicore__ inline void RunMkProducerPhase(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling)
{
    TPipe pipe;
    if constexpr (WITH_MID) {
        RunPrefetchedFiveProducer<T>(tensors, tiling, &pipe);
    } else {
        RunFourStagedProducer<T>(tensors, tiling, &pipe);
    }
}

/** Compute partialMm[m,q]=sum_k(float(x2[m,k])*phi[q,k]) and partialRms=sum_k(x2*x2) over each K shard.
 * The wave variant also produces the next Post batch; both share this phase's pipe.
 */
template <typename T, typename StageTransfer, bool USE_HF32, bool WAVE_PRODUCER = false>
__aicore__ inline void RunMkProjectionPhase(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling)
{
    TPipe pipe;
    if constexpr (WAVE_PRODUCER) {
        RunProjectionSchedule<T, StageTransfer, USE_HF32, true, WavePostProducer<T>>(tensors, tiling, &pipe);
    } else {
        RunProjectionSchedule<T, StageTransfer, USE_HF32>(tensors, tiling, &pipe);
    }
}

/** Reduce projection partials, compute Pre weights and finish hInPrime=RMS1(weighted_sum(x2)).
 * The final phase owns its pipe until all output work has completed on return.
 */
template <typename T>
__aicore__ inline void RunMkEpiloguePhase(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling)
{
    TPipe pipe;
    RunEpilogue<T>(tensors, tiling, &pipe);
}

/** M-K owns a token tile and K shard; four-op may overlap the next Post batch.
 * Formula: x2=Post(RMS0(hOut),residual), optionally RMSMid(x2); y=RMS1(Pre(x2)).
 * Host owns all tile sizes. Phase functions own pipe lifetimes; this entry owns ordering and cross-core
 * synchronization.
 */
template <typename T, bool WITH_MID, bool USE_HF32>
__aicore__ inline void RunMkSchedule(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling)
{
    using StageTransfer = Transfer::StagePolicyT<WITH_MID ? kFiveProjectionTransport : kFourProjectionTransport>;
    if constexpr (!WITH_MID) {
        if (tiling.wavePipeline != 0U) {
            // Hardware notifications need no GM epoch initialization.
            RunMkProjectionPhase<T, StageTransfer, USE_HF32, true>(tensors, tiling);
            WaitMkProjectionReady();
            RunMkEpiloguePhase<T>(tensors, tiling);
            return;
        }
    }
    RunMkProducerPhase<T, WITH_MID>(tensors, tiling);
    WaitMkRowsReady();

    RunMkProjectionPhase<T, StageTransfer, USE_HF32>(tensors, tiling);
    WaitMkProjectionReady();

    RunMkEpiloguePhase<T>(tensors, tiling);
}

} // namespace MegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_MK_H
