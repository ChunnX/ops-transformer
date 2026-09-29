/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_COMMON_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_COMMON_H

#include "compute/mhc_post.h"
#include "compute/rms_norm_0.h"
#include "compute/rms_norm_mid.h"
#include "compute/mhc_pre.h"
#include "mega_mhc_kernel_context.h"
#include "compute/projection_cube.h"
#include "transfer.h"
#include "epilogue.h"

namespace MegaMhc::Schedule {
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
    const event_t stageEvent = tiling.stageUbSlots == 2U && (tile & 1U) == 0U ? EVENT_ID6 : EVENT_ID7;
    if constexpr (ASYNC_STAGE) {
        WaitFlag<HardEvent::MTE3_V>(stageEvent);
    }
    Compute::PrepareProjectionTile<T, false, StageTransfer::kPackedNz>(
        reinterpret_cast<__ubuf__ T *>(input.GetPhyAddr()), rows, kCount,
        StageTransfer::kPackedNz ? tiling.stageRowStride : tiling.kL1,
        reinterpret_cast<__ubuf__ float *>(stage.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(squares.GetPhyAddr()));
    SetFlag<HardEvent::V_MTE2>(event);
    if constexpr (ASYNC_STAGE) {
        StageTransfer::StoreAivRowAsync(tensors.stage, stageOffset, stage, rows * tiling.kL1, tiling, stageEvent);
    } else {
        StageTransfer::StoreAivRow(tensors.stage, stageOffset, stage, rows * tiling.kL1, tiling);
    }
}

/** Empty producer policy for projection of already materialized x2 rows. */
struct MaterializedRows {};

/**
 * Run the shared M/K projection loop. The template policy selects only its producer:
 * Four-op optionally produces the next batch of complete Post rows during projection.
 * Otherwise both variants project previously materialized x2 rows.
 */
template <typename T, typename StageTransfer, bool USE_HF32, bool WAVE_PRODUCER = false,
          typename Producer = MaterializedRows, bool ASYNC_STAGE = true>
__aicore__ inline void RunProjectionScheduleImpl(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling,
                                                 TPipe *pipe)
{
    Cube::ProjectionCube<USE_HF32> cube;
    if ASCEND_IS_AIC {
        cube.Init(pipe, &tiling);
        StageTransfer::SeedPairCredits();
        const uint32_t aicIndex = GetBlockIdx();
        const uint32_t mIndex = aicIndex / tiling.kGrid;
        const uint32_t kIndex = aicIndex % tiling.kGrid;
        const uint32_t l1TileCount = Kernel::PlannedL1TileCount(tiling, kIndex);

        for (uint32_t wave = 0; wave < tiling.mWaveCount; ++wave) {
            const uint32_t mRows = GetWaveRows(tiling, mIndex, wave);
            const uint64_t mStart =
                static_cast<uint64_t>(wave) * tiling.rowsPerWave + static_cast<uint64_t>(mIndex) * tiling.mTile;
            const uint64_t kStart = Kernel::PlannedKStart(tiling, kIndex);
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
                }
                if constexpr (StageTransfer::kReturnAfterLoad) {
                    StageTransfer::ReturnCredit(0U);
                    StageTransfer::ReturnCredit(1U);
                }
                if (mRows != 0U) {
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
                if constexpr (!StageTransfer::kReturnAfterLoad) {
                    StageTransfer::ReturnCredit(0U);
                    StageTransfer::ReturnCredit(1U);
                }
            }
            if constexpr (WAVE_PRODUCER) {
                if (tiling.waveEpilogueRows != 0U) {
                    CrossCoreSetFlag<2, PIPE_FIX>(9U);
                }
            }
        }
        cube.End();
    }

    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> rmsRowsBuffer;
        TBuf<TPosition::VECCALC> stageTileBuffer;
        TBuf<TPosition::VECCALC> prefetchBuffer;
        Kernel::InitUbBuffer(pipe, rmsRowsBuffer, tiling.projectionRows * sizeof(float));
        Kernel::InitUbBuffer(pipe, stageTileBuffer,
                             tiling.stageUbSlots * tiling.stageRowStride * tiling.kL1 * sizeof(float));
        Kernel::InitUbBuffer(pipe, prefetchBuffer, tiling.projectionRows * tiling.kL1 * sizeof(T));
        LocalTensor<T> prefetchInput = prefetchBuffer.Get<T>();
        if constexpr (ASYNC_STAGE) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
            if (tiling.stageUbSlots == 2U) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
            }
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
        const uint64_t kStart = Kernel::PlannedKStart(tiling, kIndex);

        TBuf<TPosition::VECCALC> x2TileBuffer;
        const uint32_t tileElements = tiling.projectionRows * tiling.kL1;
        Kernel::InitUbBuffer(pipe, x2TileBuffer, tileElements * sizeof(T));
        LocalTensor<T> x2Tile = x2TileBuffer.Get<T>();
        Producer producer;
        WaveEpilogue<T> epilogue;
        if constexpr (WAVE_PRODUCER) {
            if (tiling.waveEpilogueRows != 0U) {
                epilogue.Init(tensors, tiling, pipe);
            }
            producer.Init(tensors, tiling, pipe);
            producer.Begin(0U);
            producer.Fill();
        }
        for (uint32_t wave = 0; wave < tiling.mWaveCount; ++wave) {
            if constexpr (WAVE_PRODUCER) {
                producer.WaitReady(wave);
                if (tiling.waveEpilogueRows != 0U && wave != 0U) {
                    epilogue.Begin(wave - 1U);
                }
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
                    stageTile[tiling.stageUbSlots == 2U ? (tile & 1U) * tiling.stageRowStride * tiling.kL1 : 0U],
                    rmsRows, tile, l1TileCount, kIndex);
                StageTransfer::PublishReady();
                if constexpr (WAVE_PRODUCER) {
                    if (wave + 1U < tiling.mWaveCount) {
                        producer.Tick(tile);
                    }
                    if (tiling.waveEpilogueRows != 0U && wave != 0U && tile >= 2U) {
                        epilogue.Tick(tensors);
                    }
                }
            }

            if (rowCount != 0U) {
                Kernel::CopyUbToGlobal<float>(tensors.partialRms,
                                              static_cast<uint64_t>(kIndex) * tiling.totalRows + mStart + rowStart,
                                              rmsRows, rowCount);
            }
            if constexpr (WAVE_PRODUCER) {
                if (tiling.waveEpilogueRows != 0U) {
                    if (wave != 0U) {
                        epilogue.Finish(tensors);
                    }
                    // FIX protects all projection partials; MTE3 protects the RMS shards.
                    CrossCoreWaitFlag<2, PIPE_MTE3>(9U);
                    CrossCoreSetFlag<0, PIPE_MTE3>(10U);
                    CrossCoreWaitFlag<0, PIPE_MTE2>(10U);
                }
            }
        }
        if constexpr (WAVE_PRODUCER) {
            producer.End();
            if (tiling.waveEpilogueRows != 0U) {
                epilogue.Begin(tiling.mWaveCount - 1U);
                epilogue.Finish(tensors);
            }
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        StageTransfer::DrainCredits();
        if constexpr (ASYNC_STAGE) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
            if (tiling.stageUbSlots == 2U) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
            }
        }
    }
}

/** Dispatch the host-selected transport once; each projection VF remains a compile-time layout specialization. */
template <typename T, typename StageTransfer, bool USE_HF32, bool WAVE_PRODUCER = false,
          typename Producer = MaterializedRows, bool ASYNC_STAGE = true>
__aicore__ inline void RunProjectionSchedule(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling, TPipe *pipe)
{
    if constexpr (StageTransfer::kPackedNz) {
        if (tiling.projectionTransport == static_cast<uint32_t>(ProjectionTransport::GmStage)) {
            RunProjectionScheduleImpl<T, Transfer::GmStage, USE_HF32, WAVE_PRODUCER, Producer, ASYNC_STAGE>(
                tensors, tiling, pipe);
            return;
        }
    }
    RunProjectionScheduleImpl<T, StageTransfer, USE_HF32, WAVE_PRODUCER, Producer, ASYNC_STAGE>(tensors, tiling, pipe);
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
        Kernel::InitUbBuffer(pipe, gamma2Buffer, tiling.midGammaTile * sizeof(float));
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
        if (tiling.midGammaTileCount == 1U) {
            Kernel::CopyGlobalToUb(gamma2, tensors.gamma2, 0U, tiling.k);
        }
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
            if (tiling.midGammaTileCount == 1U) {
                Compute::ComputeRmsNormMidFromSquares<T>(reinterpret_cast<__ubuf__ T *>(post.GetPhyAddr()),
                                                         reinterpret_cast<__ubuf__ float *>(gamma2.GetPhyAddr()),
                                                         tiling.k, tiling.normEps,
                                                         reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()),
                                                         reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr()) + 1U,
                                                         reinterpret_cast<__ubuf__ T *>(post.GetPhyAddr()));
            } else {
                // invRms uses all K elements once; gamma chunks preserve the same rounding point.
                auto sum = reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr());
                Compute::FinalizeInverseRms(sum, tiling.k, tiling.normEps, sum + 1U);
                for (uint32_t chunk = 0U; chunk < tiling.midGammaTileCount; ++chunk) {
                    const uint32_t offset = chunk * tiling.midGammaTile;
                    const uint32_t count =
                        chunk + 1U == tiling.midGammaTileCount ? tiling.midGammaLastTile : tiling.midGammaTile;
                    Kernel::CopyGlobalToUb(gamma2, tensors.gamma2, offset, count);
                    auto rowPart = reinterpret_cast<__ubuf__ T *>(post[offset].GetPhyAddr());
                    Compute::ApplyGammaAndCastRow<T>(rowPart, reinterpret_cast<__ubuf__ float *>(gamma2.GetPhyAddr()),
                                                     count, sum + 1U, rowPart);
                }
            }
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

} // namespace MegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_COMMON_H
