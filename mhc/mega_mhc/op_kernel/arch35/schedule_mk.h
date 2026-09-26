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

#include "compute/mhc_post.h"
#include "compute/rms_norm_0.h"
#include "compute/rms_norm_mid.h"
#include "compute/mhc_pre.h"
#include "mega_mhc_kernel_context.h"
#include "compute/projection_cube.h"
#include "transfer.h"

namespace AiInfraMegaMhc::Schedule {
using namespace AscendC;

/** Return the number of valid rows from the host-planned M wave and tile index. */
__aicore__ inline uint32_t GetWaveRows(const MegaMhcTilingData &tiling, uint32_t mIndex, uint32_t wave)
{
    if (wave + 1U != tiling.mWaveCount) {
        return tiling.mTile;
    }
    if (mIndex >= tiling.lastWaveMGrid) {
        return 0U;
    }
    return mIndex + 1U == tiling.lastWaveMGrid ? tiling.lastMCoreRows : tiling.mTile;
}

/** Load a packed row block asynchronously; the event owns the destination until VF consumes it. */
template <typename T>
__aicore__ inline void PrefetchProjectionRows(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling,
                                              uint64_t firstRow, uint32_t rows, uint64_t kStart, uint32_t kCount,
                                              const LocalTensor<T> &input, event_t event)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopyExtParams copyParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(kCount * sizeof(T)),
                                 static_cast<uint32_t>((tiling.k - kCount) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(input, tensors.x2Out[firstRow * tiling.k + kStart], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(event);
}

/** Transfer a host-sized row block: packed x2 GM -> UB -> FP32 stage GM.
 * One strided load and one store replace per-row DMA handshakes; statistics share the conversion VF.
 */
template <typename T, typename StageTransfer, bool ASYNC_STAGE = false>
__aicore__ inline void StageProjectionRows(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling,
                                           uint64_t firstRow, uint32_t rows, uint64_t kStart, uint32_t kCount,
                                           uint64_t stageOffset, const LocalTensor<T> &input0,
                                           const LocalTensor<T> &input1, const LocalTensor<float> &stage,
                                           const LocalTensor<float> &squares, uint32_t tile, uint32_t tileCount,
                                           uint32_t kIndex)
{
    if (rows == 0U) {
        return;
    }
    const auto input = (tile & 1U) == 0U ? input0 : input1;
    const auto next = (tile & 1U) == 0U ? input1 : input0;
    const event_t event = (tile & 1U) == 0U ? EVENT_ID2 : EVENT_ID3;
    const event_t nextEvent = (tile & 1U) == 0U ? EVENT_ID3 : EVENT_ID2;
    if (tile == 0U) {
        PrefetchProjectionRows(tensors, tiling, firstRow, rows, kStart, kCount, input, event);
    }
    WaitFlag<HardEvent::MTE2_V>(event);
    if (tile + 1U < tileCount) {
        PrefetchProjectionRows(tensors, tiling, firstRow, rows, kStart + tiling.kL1,
                               Kernel::PlannedL1TileKCount(tiling, kIndex, tile + 1U), next, nextEvent);
    }
    if constexpr (ASYNC_STAGE) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
    }
    Compute::PrepareProjectionTile<T>(reinterpret_cast<__ubuf__ T *>(input.GetPhyAddr()), rows, kCount, tiling.kL1,
                                      reinterpret_cast<__ubuf__ float *>(stage.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ float *>(squares.GetPhyAddr()));
    SetFlag<HardEvent::V_MTE2>(event);
    if constexpr (ASYNC_STAGE) {
        StageTransfer::StoreAivRowAsync(tensors.stage, stageOffset, stage, rows * tiling.kL1, tiling);
    } else {
        StageTransfer::StoreAivRow(tensors.stage, stageOffset, stage, rows * tiling.kL1, tiling);
    }
}

/** Initialize the four-op completion epochs to zero before any group starts producing.
 * Each AIV owns one 128-byte region in otherwise unused RMS0 workspace.
 */
template <typename T>
__aicore__ inline void InitWaveEpochs(Kernel::Tensors<T> &tensors, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> buffer;
        Kernel::InitUbBuffer(pipe, buffer, 128U);
        auto scratch = buffer.Get<float>();
        Kernel::ZeroFp32Values(reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr()), 32U);
        Kernel::CopyUbToGlobal(tensors.invRms0, static_cast<uint64_t>(GetBlockIdx()) * 32U, scratch, 32U);
    }
    SyncAll<false>();
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

    /** Begin one planned producer batch: rows belong to member%groupSize within the M group. */
    __aicore__ inline void Begin(uint32_t wave)
    {
        const auto &t = *tiling_;
        row_ = static_cast<uint64_t>(wave) * t.rowsPerWave +
               static_cast<uint64_t>(GetBlockIdx() / t.waveGroupSize) * t.mTile +
               (GetBlockIdx() % t.waveGroupSize) * t.waveProducerRows;
        produced_ = 0U;
        slot_ = 0U;
        nextTick_ = 1U;
        Prefetch(row_, 0U);
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
        if (produced_ + 1U < t.waveProducerRows) {
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
        while (produced_ < tiling_->waveProducerRows) {
            ProduceOne();
        }
    }

    /** Interleave at most one next-wave row after a tile is published to Cube.
     * The spacing is host planned; no tiling search or arithmetic VF branching occurs here.
     */
    __aicore__ inline void Tick(uint32_t tile)
    {
        if (tile >= nextTick_ && produced_ < tiling_->waveProducerRows) {
            ProduceOne();
            nextTick_ += tiling_->waveProducerTileStride;
        }
    }

    /** Publish phase=wave+1 after all local rows are stored, then wait for all group members.
     * Completion epochs are monotone; immutable x2 rows need no acknowledgement or flag clearing.
     */
    __aicore__ inline void WaitReady(uint32_t wave)
    {
        const int32_t phase = static_cast<int32_t>(wave + 1U);
        const uint32_t member = GetBlockIdx();
        const uint32_t base = member / tiling_->waveGroupSize * tiling_->waveGroupSize;
        auto flags = tensors_->invRms0.template ReinterpretCast<int32_t>();
        sync_.SetValue(0, phase);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID6);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID6);
        DataCopy(flags[member * 32U], sync_, 8U);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID6);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID6);
        for (uint32_t sender = base; sender < base + tiling_->waveGroupSize; ++sender) {
            if (sender != member) {
                int32_t observed;
                do {
                    DataCopy(sync_, flags[sender * 32U], 8U);
                    SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
                    observed = sync_.GetValue(0);
                } while (observed < phase);
            }
        }
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
    uint32_t produced_, slot_, nextTick_;
};

/**
 * Run the shared M/K projection loop. The template policy selects only its producer:
 * Four-op optionally produces the next batch of complete Post rows during projection.
 * Otherwise both variants project previously materialized x2 rows.
 */
template <typename T, typename StageTransfer, bool WAVE_PRODUCER = false, bool ASYNC_STAGE = true>
__aicore__ inline void RunProjectionSchedule(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, TPipe *pipe)
{
    Cube::ProjectionCube cube;
    if ASCEND_IS_AIC {
        cube.Init(pipe, &tiling, tiling.projectionMode);
        StageTransfer::SeedPairCredits();
        const uint32_t aicIndex = GetBlockIdx();
        const uint32_t mIndex = aicIndex / tiling.kGrid;
        const uint32_t kIndex = aicIndex % tiling.kGrid;
        const uint32_t l1TileCount = Kernel::PlannedL1TileCount(tiling, kIndex);

        for (uint32_t wave = 0; wave < tiling.mWaveCount; ++wave) {
            const uint32_t mRows = GetWaveRows(tiling, mIndex, wave);
            const uint64_t mStart =
                static_cast<uint64_t>(wave) * tiling.rowsPerWave + static_cast<uint64_t>(mIndex) * tiling.mTile;
            const uint64_t kStart = static_cast<uint64_t>(kIndex) * tiling.splitK;
            for (uint32_t tile = 0; tile < l1TileCount; ++tile) {
                uint32_t kCount = 0;
                uint32_t l0FullLoops = 0;
                uint32_t l0Tail = 0;
                uint32_t tailLoadKStep = 0;
                uint32_t fullMmadLoops = 0;
                uint32_t tailMmadLoops = 0;
                uint32_t tailMmadKStep = 0;
                uint32_t tailMmadFinalKStep = 0;
                Kernel::GetPlannedL1Tile(tiling, kIndex, tile, kCount, l0FullLoops, l0Tail, tailLoadKStep,
                                         fullMmadLoops, tailMmadLoops, tailMmadKStep, tailMmadFinalKStep);
                const uint64_t kTileStart = kStart + static_cast<uint64_t>(tile) * tiling.kL1;
                const uint32_t activeK = kCount;
                const uint32_t l1Slot = (wave * l1TileCount + tile) & 1U;
                const uint64_t stageSlot =
                    (static_cast<uint64_t>(aicIndex) * tiling.slotCount + l1Slot) * tiling.mTile * tiling.kL1;

                if (mRows != 0U) {
                    cube.LoadB(tensors.phi, static_cast<uint32_t>(kTileStart), static_cast<uint32_t>(activeK), l1Slot);
                }
                StageTransfer::WaitReady(0U);
                StageTransfer::WaitReady(1U);
                if (mRows != 0U) {
                    StageTransfer::LoadA1(cube, tensors.stage, stageSlot, mRows, tiling.nzM,
                                          static_cast<uint32_t>(activeK), l1Slot);

                    const uint32_t groupIndex = static_cast<uint32_t>(kTileStart / tiling.partialGroupK);
                    const uint64_t groupEnd = static_cast<uint64_t>(groupIndex + 1U) * tiling.partialGroupK;
                    const bool firstInPartial = kTileStart == static_cast<uint64_t>(groupIndex) * tiling.partialGroupK;
                    const bool lastInPartial = kTileStart + activeK >= groupEnd || kTileStart + activeK == tiling.k;
                    const uint64_t partialOffset =
                        static_cast<uint64_t>(groupIndex) * tiling.totalRows * tiling.fixpipeRowStride +
                        mStart * tiling.fixpipeRowStride;
                    cube.ComputeL1Tile(mRows, tiling.nzM, l0FullLoops, l0Tail, tailLoadKStep, fullMmadLoops,
                                       tailMmadLoops, tailMmadKStep, tailMmadFinalKStep, firstInPartial, lastInPartial,
                                       l1Slot, tensors.partialMm, partialOffset);
                }
                StageTransfer::ReturnCredit(0U);
                StageTransfer::ReturnCredit(1U);
            }
        }
        cube.End();
    }

    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> rmsRowsBuffer;
        TBuf<TPosition::VECCALC> stageTileBuffer;
        TBuf<TPosition::VECCALC> prefetchBuffer;
        Kernel::InitUbBuffer(pipe, rmsRowsBuffer, tiling.projectionRows * sizeof(float));
        Kernel::InitUbBuffer(pipe, stageTileBuffer, tiling.projectionRows * tiling.kL1 * sizeof(float));
        Kernel::InitUbBuffer(pipe, prefetchBuffer, tiling.projectionRows * tiling.kL1 * sizeof(T));
        LocalTensor<T> prefetchInput = prefetchBuffer.Get<T>();
        if constexpr (ASYNC_STAGE) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
        LocalTensor<float> rmsRows = rmsRowsBuffer.Get<float>();
        LocalTensor<float> stageTile = stageTileBuffer.Get<float>();
        __ubuf__ float *rmsRowsPtr = reinterpret_cast<__ubuf__ float *>(rmsRows.GetPhyAddr());

        const uint32_t aivBlockIndex = GetBlockIdx();
        const uint32_t aicIndex = aivBlockIndex / 2U;
        const uint32_t subblock = GetSubBlockIdx();
        const uint32_t mIndex = aicIndex / tiling.kGrid;
        const uint32_t kIndex = aicIndex % tiling.kGrid;
        const uint32_t firstRows = tiling.projectionRows;
        const uint32_t l1TileCount = Kernel::PlannedL1TileCount(tiling, kIndex);
        const uint64_t kStart = static_cast<uint64_t>(kIndex) * tiling.splitK;

        TBuf<TPosition::VECCALC> x2TileBuffer;
        const uint32_t tileElements = tiling.projectionRows * tiling.kL1;
        Kernel::InitUbBuffer(pipe, x2TileBuffer, (tiling.k > tileElements ? tiling.k : tileElements) * sizeof(T));
        LocalTensor<T> x2Tile = x2TileBuffer.Get<T>();
        WavePostProducer<T> producer;
        if constexpr (WAVE_PRODUCER) {
            producer.Init(tensors, tiling, pipe);
            producer.Begin(0U);
            producer.Fill();
        }
        for (uint32_t wave = 0; wave < tiling.mWaveCount; ++wave) {
            if constexpr (WAVE_PRODUCER) {
                producer.WaitReady(wave);
                if (wave + 1U < tiling.mWaveCount) {
                    producer.Begin(wave + 1U);
                }
            }
            const uint32_t mRows = GetWaveRows(tiling, mIndex, wave);
            const uint64_t mStart =
                static_cast<uint64_t>(wave) * tiling.rowsPerWave + static_cast<uint64_t>(mIndex) * tiling.mTile;
            const uint32_t rowStart = subblock == 0U ? 0U : firstRows;
            const uint32_t rowCount =
                subblock == 0U ? (mRows < firstRows ? mRows : firstRows) : (mRows > firstRows ? mRows - firstRows : 0U);
            if (rowCount != 0U) {
                Kernel::ZeroFp32Values(rmsRowsPtr, rowCount);
            }

            for (uint32_t tile = 0; tile < l1TileCount; ++tile) {
                const uint32_t kCount = Kernel::PlannedL1TileKCount(tiling, kIndex, tile);
                const uint64_t kTileStart = kStart + static_cast<uint64_t>(tile) * tiling.kL1;
                const uint32_t activeK = kCount;
                const uint32_t slot = (wave * l1TileCount + tile) & 1U;
                const uint64_t stageSlot =
                    (static_cast<uint64_t>(aicIndex) * tiling.slotCount + slot) * tiling.mTile * tiling.kL1 +
                    static_cast<uint64_t>(rowStart) * tiling.kL1;

                StageTransfer::AcquireSlot();
                StageProjectionRows<T, StageTransfer, ASYNC_STAGE>(
                    tensors, tiling, mStart + rowStart, rowCount, kTileStart, activeK, stageSlot, x2Tile, prefetchInput,
                    stageTile, rmsRows, tile, l1TileCount, kIndex);
                StageTransfer::PublishReady();
                if constexpr (WAVE_PRODUCER) {
                    if (wave + 1U < tiling.mWaveCount) {
                        producer.Tick(tile);
                    }
                }
            }

            if (rowCount != 0U) {
                Kernel::CopyUbToGlobal<float>(tensors.partialRms,
                                              static_cast<uint64_t>(kIndex) * tiling.totalRows + mStart + rowStart,
                                              rmsRows, rowCount);
            }
        }
        if constexpr (WAVE_PRODUCER) {
            producer.End();
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        StageTransfer::DrainCredits();
        if constexpr (ASYNC_STAGE) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
        }
    }
}

/** Enqueue one complete producer input row: hOut=x[row], residual=r[row], coefficients=H[row].
 * Wait for V_MTE2 input credit or MTE3_MTE2 aliased output credit; MTE2_V publishes all inputs together.
 */
template <typename T, bool OUTPUT_CREDIT = false>
__aicore__ inline void PrefetchStagedProducer(Kernel::Tensors<T> &tensors, uint64_t row,
                                              const MegaMhcTilingData &tiling, const LocalTensor<T> &hOut,
                                              const LocalTensor<T> &residual, const LocalTensor<float> &coefficients,
                                              event_t event)
{
    if constexpr (OUTPUT_CREDIT) {
        WaitFlag<HardEvent::MTE3_MTE2>(event);
    } else {
        WaitFlag<HardEvent::V_MTE2>(event);
    }
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
__aicore__ inline void RunFourStagedProducer(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, TPipe *pipe)
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
                PrefetchStagedProducer(tensors, row + tiling.activeAiv, tiling, input[next * tiling.d],
                                       residual[next * tiling.k], coefficients[next * 128U],
                                       next == 0U ? EVENT_ID4 : EVENT_ID5);
            }
            Compute::ComputeRmsNorm0Row<T>(reinterpret_cast<__ubuf__ T *>(input[slot * tiling.d].GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ float *>(gamma.GetPhyAddr()), tiling.d,
                                           tiling.normEps, reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()) + 1U,
                                           reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ T *>(x1.GetPhyAddr()));
            Compute::ComputePostFourHeads<T>(
                reinterpret_cast<__ubuf__ T *>(x1.GetPhyAddr()),
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

/** Produce x2[row]=Post(Cast<T>(RMS0(hOut[row])), residual[row]) followed by RMSMid, with both input-type stage
 * roundings. Two complete input slots overlap next-row DMA with current-row RegBase computation. Post output retains
 * its input-type rounding before GM publication; the phase barrier remains.
 */
template <typename T>
__aicore__ inline void RunPrefetchedFiveProducer(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling,
                                                 TPipe *pipe)
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
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        if (firstRow < tiling.totalRows) {
            PrefetchStagedProducer<T, true>(tensors, firstRow, tiling, input, residual, coefficients, EVENT_ID4);
        }
        uint32_t slot = 0U;
        for (uint64_t row = firstRow; row < tiling.totalRows; row += tiling.activeAiv) {
            const event_t event = slot == 0U ? EVENT_ID4 : EVENT_ID5;
            WaitFlag<HardEvent::MTE2_V>(event);
            const uint32_t next = slot ^ 1U;
            if (row + tiling.activeAiv < tiling.totalRows) {
                PrefetchStagedProducer<T, true>(tensors, row + tiling.activeAiv, tiling, input[next * tiling.d],
                                                residual[next * tiling.k], coefficients[next * 128U],
                                                next == 0U ? EVENT_ID4 : EVENT_ID5);
            }
            Compute::ComputeRmsNorm0Row<T>(reinterpret_cast<__ubuf__ T *>(input[slot * tiling.d].GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ float *>(gamma.GetPhyAddr()), tiling.d,
                                           tiling.normEps, reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()) + 1U,
                                           reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ T *>(x1.GetPhyAddr()));
            auto post = residual[slot * tiling.k];
            Compute::ComputePostFourHeads<T, true>(
                reinterpret_cast<__ubuf__ T *>(x1.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T *>(residual[slot * tiling.k].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(coefficients[slot * 128U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(coefficients[slot * 128U + 64U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ T *>(post.GetPhyAddr()), 1U, tiling.d, 0U, tiling.d,
                reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()));
            Compute::ComputeRmsNormMidFromSquares<T>(reinterpret_cast<__ubuf__ T *>(post.GetPhyAddr()),
                                                     reinterpret_cast<__ubuf__ float *>(gamma2.GetPhyAddr()), tiling.k,
                                                     tiling.normEps,
                                                     reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()),
                                                     reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()) + 1U,
                                                     reinterpret_cast<__ubuf__ T *>(post.GetPhyAddr()));
            // x2[row,k]=post[k]. Publish slot credit on MTE3; only reuse waits for completion.
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams params{1U, static_cast<uint32_t>(tiling.k * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(tensors.x2Out[row * tiling.k], post, params);
            SetFlag<HardEvent::MTE3_MTE2>(event);
            slot = next;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        // Drain both output reads before releasing UB, including unused/empty input slots.
        SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
    }
}

/** Schedule RMS0/Post -> optional Mid -> projection; all variants finalize Pre/RMS1 after the barrier.
 * Four-op waves publish complete Post rows while earlier rows project; five-op completes Mid first.
 */
template <typename T, bool WITH_MID = true>
__aicore__ inline void RunMkSchedule(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, TPipe *pipe)
{
    using StageTransfer = Transfer::StagePolicyT<WITH_MID ? kFiveProjectionTransport : kFourProjectionTransport>;
    if constexpr (!WITH_MID) {
        if (tiling.wavePipeline != 0U) {
            TPipe initPipe;
            InitWaveEpochs(tensors, &initPipe);
            initPipe.Destroy();
            RunProjectionSchedule<T, StageTransfer, true>(tensors, tiling, pipe);
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
    RunProjectionSchedule<T, StageTransfer>(tensors, tiling, &projectionPipe);
    projectionPipe.Destroy();
    SyncAll<false>();
}

} // namespace AiInfraMegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_MK_H
