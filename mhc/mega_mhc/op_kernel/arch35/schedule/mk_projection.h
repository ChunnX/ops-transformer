/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is distributed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_MK_PROJECTION_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_MK_PROJECTION_H

#include "../compute/mhc_post.h"
#include "../compute/rms_norm_0.h"
#include "../compute/projection_stage.h"
#include "../mega_mhc_kernel_context.h"
#include "../projection_cube.h"
#include "../transfer/stage_policy.h"
#include "mk_epilogue.h"

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

/** Map a flattened K interval onto head and D views without changing the planned interval. */
template <typename T>
__aicore__ inline void ComputePostKRange(__ubuf__ T *x1, __ubuf__ T *residual,
                                         __ubuf__ float *hPost, __ubuf__ float *hRes,
                                         __ubuf__ T *x2Rows, uint32_t rows, uint32_t d,
                                         uint32_t kStart, uint32_t kCount)
{
    uint32_t completed = 0;
    while (completed < kCount) {
        const uint32_t kPosition = kStart + completed;
        const uint32_t outputHead = kPosition / d;
        const uint32_t column = kPosition % d;
        const uint32_t headRemaining = d - column;
        const uint32_t remaining = kCount - completed;
        const uint32_t segment = remaining < headRemaining ? remaining : headRemaining;
        Compute::ComputePostTile<T>(x1, residual, hPost, hRes, x2Rows, rows, d,
                                    outputHead, 1U, column, segment);
        completed += segment;
    }
}

/** Enqueue hOut[row,:] and residual[row,:,:] into a free producer input slot.
 * The slot's V_MTE2 credit protects reuse; MTE2_V publishes both arrays to RMS0/Post.
 */
template <typename T>
__aicore__ inline void PrefetchProducerInput(Kernel::Tensors<T> &tensors, uint64_t row, uint32_t d,
    const LocalTensor<T> &hOut, const LocalTensor<T> &residual, event_t event)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    Kernel::EnqueueGlobalToUb(hOut, tensors.hOut, row * d, d);
    Kernel::EnqueueGlobalToUb(residual, tensors.residual, row * 4U * d, 4U * d);
    SetFlag<HardEvent::MTE2_V>(event);
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
    uint64_t firstRow, uint32_t rows, uint64_t kStart, uint32_t kCount, uint64_t stageOffset,
    const LocalTensor<T> &input0, const LocalTensor<T> &input1,
    const LocalTensor<float> &stage, const LocalTensor<float> &squares,
    uint32_t tile, uint32_t tileCount, uint32_t kIndex)
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
    Compute::PrepareProjectionTile<T>(reinterpret_cast<__ubuf__ T *>(input.GetPhyAddr()), rows,
        kCount, tiling.kL1, reinterpret_cast<__ubuf__ float *>(stage.GetPhyAddr()),
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
        Compute::ComputePostFourHeads<T>(reinterpret_cast<__ubuf__ T *>(x1_.GetPhyAddr()),
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
 * Four-op caches complete Post rows per M wave when K has one owner, otherwise
 * produces disjoint K ranges. Five-op stages the previously materialized x2 rows.
 */
template <typename T, bool FUSED_POST, typename StageTransfer, bool OVERLAP_EPILOGUE = false, bool WAVE_PRODUCER = false, bool ASYNC_STAGE = false>
__aicore__ inline void RunMKScheduler(Kernel::Tensors<T> &tensors,
                                      const MegaMhcTilingData &tiling, TPipe *pipe)
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
            const uint64_t mStart = static_cast<uint64_t>(wave) * tiling.rowsPerWave +
                                    static_cast<uint64_t>(mIndex) * tiling.mTile;
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
                                         fullMmadLoops, tailMmadLoops, tailMmadKStep,
                                         tailMmadFinalKStep);
                const uint64_t kTileStart = kStart + static_cast<uint64_t>(tile) * tiling.kL1;
                const uint32_t activeK = kCount;
                const uint32_t l1Slot = (wave * l1TileCount + tile) & 1U;
                const uint64_t stageSlot =
                    (static_cast<uint64_t>(aicIndex) * tiling.slotCount + l1Slot) *
                    tiling.mTile * tiling.kL1;

                if (mRows != 0U) {
                    cube.LoadB(tensors.phi, static_cast<uint32_t>(kTileStart),
                               static_cast<uint32_t>(activeK), l1Slot);
                }
                StageTransfer::WaitReady(0U);
                StageTransfer::WaitReady(1U);
                if (mRows != 0U) {
                    StageTransfer::LoadA1(cube, tensors.stage, stageSlot, mRows,
                                          tiling.nzM, static_cast<uint32_t>(activeK), l1Slot);

                    const uint32_t groupIndex = static_cast<uint32_t>(kTileStart / tiling.partialGroupK);
                    const uint64_t groupEnd = static_cast<uint64_t>(groupIndex + 1U) * tiling.partialGroupK;
                    const bool firstInPartial = kTileStart ==
                                                static_cast<uint64_t>(groupIndex) * tiling.partialGroupK;
                    const bool lastInPartial = kTileStart + activeK >= groupEnd ||
                                               kTileStart + activeK == tiling.k;
                    const uint64_t partialOffset =
                        static_cast<uint64_t>(groupIndex) * tiling.totalRows * tiling.fixpipeRowStride +
                        mStart * tiling.fixpipeRowStride;
                    cube.ComputeL1Tile(mRows, tiling.nzM,
                                       l0FullLoops, l0Tail, tailLoadKStep, fullMmadLoops, tailMmadLoops,
                                       tailMmadKStep, tailMmadFinalKStep, firstInPartial,
                                       lastInPartial, l1Slot,
                                       tensors.partialMm, partialOffset);
                }
                StageTransfer::ReturnCredit(0U);
                StageTransfer::ReturnCredit(1U);
            }
            if constexpr (OVERLAP_EPILOGUE) {
                if (tiling.kGrid == 1U) {
                    CrossCoreSetFlag<Kernel::kPairSyncMode, PIPE_FIX>(Kernel::kPairWaveDoneFlag);
                    CrossCoreSetFlag<Kernel::kPairSyncMode, PIPE_FIX>(Kernel::kPairWaveDoneFlag + Kernel::kSubblockFlagOffset);
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
        Kernel::InitUbBuffer(pipe, stageTileBuffer, tiling.projectionRows * tiling.kL1 * sizeof(float));
        Kernel::InitUbBuffer(pipe, prefetchBuffer, tiling.projectionRows * tiling.kL1 * sizeof(T));
        LocalTensor<T> prefetchInput = prefetchBuffer.Get<T>();
        if constexpr (ASYNC_STAGE) { SetFlag<HardEvent::MTE3_V>(EVENT_ID7); }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
        LocalTensor<float> rmsRows = rmsRowsBuffer.Get<float>();
        LocalTensor<float> stageTile = stageTileBuffer.Get<float>();
        __ubuf__ float *rmsRowsPtr = reinterpret_cast<__ubuf__ float *>(rmsRows.GetPhyAddr());
        __ubuf__ float *stageTilePtr = reinterpret_cast<__ubuf__ float *>(stageTile.GetPhyAddr());

        const uint32_t aivBlockIndex = GetBlockIdx();
        const uint32_t aicIndex = aivBlockIndex / 2U;
        const uint32_t subblock = GetSubBlockIdx();
        const uint32_t mIndex = aicIndex / tiling.kGrid;
        const uint32_t kIndex = aicIndex % tiling.kGrid;
        const uint32_t firstRows = tiling.projectionRows;
        const uint32_t l1TileCount = Kernel::PlannedL1TileCount(tiling, kIndex);
        const uint64_t kStart = static_cast<uint64_t>(kIndex) * tiling.splitK;

        if constexpr (FUSED_POST) {
            const uint32_t producerRows = tiling.postRows;
            TBuf<TPosition::VECCALC> hOutBuffer;
            TBuf<TPosition::VECCALC> producerInputBuffer;
            TBuf<TPosition::VECCALC> x1Buffer;
            TBuf<TPosition::VECCALC> residualBuffer;
            TBuf<TPosition::VECCALC> fusedX2TileBuffer;
            TBuf<TPosition::VECCALC> gamma0Buffer;
            TBuf<TPosition::VECCALC> hPostBuffer;
            TBuf<TPosition::VECCALC> hResBuffer;
            TBuf<TPosition::VECCALC> scalarBuffer;
            TBuf<TPosition::VECCALC> alphaBuffer;
            TBuf<TPosition::VECCALC> biasBuffer;
            Kernel::InitUbBuffer(pipe, hOutBuffer, producerRows * tiling.d * sizeof(T));
            if (tiling.producerPrefetch != 0U) {
                Kernel::InitUbBuffer(pipe, producerInputBuffer, producerRows * (tiling.d + tiling.k) * sizeof(T));
            }
            Kernel::InitUbBuffer(pipe, x1Buffer, producerRows * tiling.d * sizeof(T));
            Kernel::InitUbBuffer(pipe, residualBuffer, producerRows * tiling.k * sizeof(T));
            Kernel::InitUbBuffer(pipe, fusedX2TileBuffer, producerRows * tiling.k * sizeof(T));
            Kernel::InitUbBuffer(pipe, gamma0Buffer, tiling.d * sizeof(float));
            Kernel::InitUbBuffer(pipe, hPostBuffer, (OVERLAP_EPILOGUE ? tiling.projectionRows * 24U : producerRows * 4U) * sizeof(float));
            Kernel::InitUbBuffer(pipe, hResBuffer, (OVERLAP_EPILOGUE && tiling.projectionRows > producerRows * 16U ? tiling.projectionRows : producerRows * 16U) * sizeof(float));
            Kernel::InitUbBuffer(pipe, scalarBuffer, 2U * sizeof(float));
            if constexpr (OVERLAP_EPILOGUE) {
                Kernel::InitUbBuffer(pipe, alphaBuffer, 4U * sizeof(float));
                Kernel::InitUbBuffer(pipe, biasBuffer, 32U * sizeof(float));
            }
            LocalTensor<T> hOut = hOutBuffer.Get<T>();
            LocalTensor<T> x1 = x1Buffer.Get<T>();
            LocalTensor<T> residual = residualBuffer.Get<T>();
            LocalTensor<T> fusedX2Tile = fusedX2TileBuffer.Get<T>();
            LocalTensor<float> gamma0 = gamma0Buffer.Get<float>();
            LocalTensor<float> hPost = hPostBuffer.Get<float>();
            LocalTensor<float> hRes = hResBuffer.Get<float>();
            LocalTensor<float> scalar = scalarBuffer.Get<float>();
            __ubuf__ T *hOutPtr = reinterpret_cast<__ubuf__ T *>(hOut.GetPhyAddr());
            __ubuf__ T *x1Ptr = reinterpret_cast<__ubuf__ T *>(x1.GetPhyAddr());
            __ubuf__ T *residualPtr = reinterpret_cast<__ubuf__ T *>(residual.GetPhyAddr());
            __ubuf__ T *fusedX2TilePtr = reinterpret_cast<__ubuf__ T *>(fusedX2Tile.GetPhyAddr());
            __ubuf__ float *hPostPtr = reinterpret_cast<__ubuf__ float *>(hPost.GetPhyAddr());
            __ubuf__ float *hResPtr = reinterpret_cast<__ubuf__ float *>(hRes.GetPhyAddr());
            __ubuf__ float *scalarPtr = reinterpret_cast<__ubuf__ float *>(scalar.GetPhyAddr());
            Kernel::CopyGlobalToUb<float>(gamma0, tensors.gamma0, 0, tiling.d);

            for (uint32_t wave = 0; wave < tiling.mWaveCount; ++wave) {
                const uint32_t mRows = GetWaveRows(tiling, mIndex, wave);
                const uint64_t mStart = static_cast<uint64_t>(wave) * tiling.rowsPerWave +
                                        static_cast<uint64_t>(mIndex) * tiling.mTile;
                const uint32_t rowStart = subblock == 0U ? 0U : firstRows;
                const uint32_t rowCount = subblock == 0U ?
                                              (mRows < firstRows ? mRows : firstRows) :
                                              (mRows > firstRows ? mRows - firstRows : 0U);
                if (rowCount != 0U) {
                    Kernel::ZeroFp32Values(rmsRowsPtr, rowCount);
                }

                // With one K owner, produce each complete token exactly once. Cube may
                // finish the previous M wave while this AIV prepares the next wave.
                // x2Out is the persistent row cache; the existing two-slot protocol
                // still publishes only the host-planned K tiles consumed by Cube.
                if (tiling.kGrid == 1U) {
                    if (tiling.producerPrefetch != 0U) {
                        // Two host-budgeted input slots: prefetch row r+1 while RMS0/Post consumes row r.
                        LocalTensor<T> nextHOut = producerInputBuffer.Get<T>();
                        LocalTensor<T> nextResidual = nextHOut[producerRows * tiling.d];
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
                        if (rowCount != 0U) {
                            PrefetchProducerInput(tensors, mStart + rowStart, tiling.d, hOut, residual, EVENT_ID4);
                        }
                        for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
                            const uint64_t row = mStart + rowStart + localRow;
                            const bool firstSlot = (localRow & 1U) == 0U;
                            const auto currentHOut = firstSlot ? hOut : nextHOut;
                            const auto currentResidual = firstSlot ? residual : nextResidual;
                            const event_t currentEvent = firstSlot ? EVENT_ID4 : EVENT_ID5;
                            WaitFlag<HardEvent::MTE2_V>(currentEvent);
                            // Finish coefficient reads before enqueuing the next input, so they cannot fence that prefetch.
                            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                            Kernel::EnqueueGlobalToUb(hPost, tensors.hPost, row * 4U, 4U);
                            Kernel::EnqueueGlobalToUb(hRes, tensors.hRes, row * 16U, 16U);
                            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                            if (localRow + 1U < rowCount) {
                                PrefetchProducerInput(tensors, row + 1U, tiling.d,
                                    firstSlot ? nextHOut : hOut, firstSlot ? nextResidual : residual,
                                    firstSlot ? EVENT_ID5 : EVENT_ID4);
                            }
                            Compute::ComputeRmsNorm0Row<T>(reinterpret_cast<__ubuf__ T *>(currentHOut.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float *>(gamma0.GetPhyAddr()), tiling.d,
                                tiling.normEps, scalarPtr + 1, scalarPtr, x1Ptr);
                            Compute::ComputePostTile<T>(x1Ptr, reinterpret_cast<__ubuf__ T *>(currentResidual.GetPhyAddr()),
                                hPostPtr, hResPtr, fusedX2TilePtr, 1U, tiling.d, 0U, 4U, 0U, tiling.d);
                            SetFlag<HardEvent::V_MTE2>(currentEvent);
                            Kernel::CopyUbToGlobal<T>(tensors.x2Out, row * tiling.k, fusedX2Tile, tiling.k);
                        }
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
                    } else {
                        for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
                            const uint64_t row = mStart + rowStart + localRow;
                            Kernel::LoadProducerRow<T, true>(tensors, row, tiling.d, hOut, residual, hPost, hRes);
                            Compute::ComputeRmsNorm0Row<T>(hOutPtr,
                                reinterpret_cast<__ubuf__ float *>(gamma0.GetPhyAddr()), tiling.d,
                                tiling.normEps, scalarPtr + 1, scalarPtr, x1Ptr);
                            Compute::ComputePostTile<T>(x1Ptr, residualPtr, hPostPtr, hResPtr,
                                fusedX2TilePtr, 1U, tiling.d, 0U, 4U, 0U, tiling.d);
                            Kernel::CopyUbToGlobal<T>(tensors.x2Out, row * tiling.k,
                                                      fusedX2Tile, tiling.k);
                        }
                    }
                    // Close the local GM producer/consumer edge before re-reading x2.
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                }

                for (uint32_t tile = 0; tile < l1TileCount; ++tile) {
                    const uint32_t kCount = Kernel::PlannedL1TileKCount(tiling, kIndex, tile);
                    const uint64_t kTileStart = kStart + static_cast<uint64_t>(tile) * tiling.kL1;
                    const uint32_t activeK = kCount;
                    const uint32_t slot = (wave * l1TileCount + tile) & 1U;
                    const uint64_t stageSlot =
                        (static_cast<uint64_t>(aicIndex) * tiling.slotCount + slot) *
                        tiling.mTile * tiling.kL1 + static_cast<uint64_t>(rowStart) * tiling.kL1;

                    StageTransfer::AcquireSlot();
                    if (tiling.kGrid == 1U) {
                        StageProjectionRows<T, StageTransfer, ASYNC_STAGE>(tensors, tiling, mStart + rowStart,
                            rowCount, kTileStart, activeK, stageSlot, fusedX2Tile, prefetchInput,
                            stageTile, rmsRows, tile, l1TileCount, kIndex);
                    } else {
                        for (uint32_t batchStart = 0; batchStart < rowCount; batchStart += producerRows) {
                            const uint32_t batchRows = rowCount - batchStart < producerRows ?
                                                           rowCount - batchStart : producerRows;
                            Kernel::CopyGlobalToUb<float>(hPost, tensors.hPost,
                                (mStart + rowStart + batchStart) * 4U, batchRows * 4U);
                            for (uint32_t batchRow = 0; batchRow < batchRows; ++batchRow) {
                                const uint32_t row = static_cast<uint32_t>(mStart) + rowStart + batchStart + batchRow;
                                Kernel::CopyGlobalToUb<T>(hOut[batchRow * tiling.d], tensors.hOut,
                                    static_cast<uint64_t>(row) * tiling.d, tiling.d);
                                Kernel::CopyGlobalToUb<T>(residual[batchRow * tiling.k], tensors.residual,
                                    static_cast<uint64_t>(row) * tiling.k, tiling.k);
                                Kernel::CopyGlobalToUb<float>(hRes[batchRow * 16U], tensors.hRes,
                                    static_cast<uint64_t>(row) * 16U, 16U);
                                Kernel::CopyGlobalToUb<float>(scalar, tensors.invRms0, row, 1U);
                                Compute::ApplyRmsNorm0Gamma<T>(hOutPtr + batchRow * tiling.d,
                                    reinterpret_cast<__ubuf__ float *>(gamma0.GetPhyAddr()), tiling.d,
                                    scalarPtr, x1Ptr + batchRow * tiling.d);
                            }
                            ComputePostKRange<T>(x1Ptr, residualPtr, hPostPtr, hResPtr,
                                fusedX2TilePtr, batchRows, tiling.d,
                                static_cast<uint32_t>(kTileStart), activeK);

                            for (uint32_t batchRow = 0; batchRow < batchRows; ++batchRow) {
                                const uint32_t rowInBlock = batchStart + batchRow;
                                const uint32_t row = static_cast<uint32_t>(mStart) + rowStart + rowInBlock;
                                const uint64_t rowTileOffset = static_cast<uint64_t>(batchRow) * tiling.k + kTileStart;
                                Compute::AccumulateValuesSquares<T, false>(fusedX2TilePtr + rowTileOffset, activeK,
                                    rmsRowsPtr + rowInBlock);
                                Kernel::CopyUbToGlobal<T>(tensors.x2Out,
                                    static_cast<uint64_t>(row) * tiling.k + kTileStart,
                                    fusedX2Tile[rowTileOffset], activeK);
                                Kernel::ConvertInputTileToFloat<T>(fusedX2TilePtr + rowTileOffset, activeK,
                                    stageTilePtr);
                                StageTransfer::StoreAivRow(tensors.stage,
                                    stageSlot + static_cast<uint64_t>(rowInBlock) * tiling.kL1,
                                    stageTile, activeK, tiling);
                            }
                        }
                    }
                    StageTransfer::PublishReady();
                    if constexpr (OVERLAP_EPILOGUE) {
                        if (tiling.kGrid == 1U && wave != 0U &&
                            (tile == 1U || l1TileCount == 1U)) {
                            CrossCoreWaitFlag<Kernel::kPairSyncMode, PIPE_MTE2>(Kernel::kPairWaveDoneFlag);
                            const uint32_t priorRows = GetWaveRows(tiling, mIndex, wave - 1U);
                            const uint32_t ownedRows = subblock == 0U ?
                                (priorRows < firstRows ? priorRows : firstRows) :
                                (priorRows > firstRows ? priorRows - firstRows : 0U);
                            FinalizePairedRows(tensors, tiling,
                                mStart - tiling.rowsPerWave + rowStart, ownedRows,
                                residual, hOut, x1, gamma0, stageTile, hPost, hRes,
                                alphaBuffer.Get<float>(), biasBuffer.Get<float>(), scalar);
                            Kernel::CopyGlobalToUb<float>(gamma0, tensors.gamma0, 0U, tiling.d);
                        }
                    }
                }

                if (rowCount != 0U) {
                    Kernel::CopyUbToGlobal<float>(tensors.partialRms,
                        static_cast<uint64_t>(kIndex) * tiling.totalRows + mStart + rowStart,
                        rmsRows, rowCount);
                }
                if constexpr (OVERLAP_EPILOGUE) {
                    if (tiling.kGrid == 1U && wave + 1U == tiling.mWaveCount) {
                        CrossCoreWaitFlag<Kernel::kPairSyncMode, PIPE_MTE2>(Kernel::kPairWaveDoneFlag);
                        FinalizePairedRows(tensors, tiling, mStart + rowStart, rowCount,
                            residual, hOut, x1, gamma0, stageTile, hPost, hRes,
                            alphaBuffer.Get<float>(), biasBuffer.Get<float>(), scalar);
                    }
                }
            }
        } else {
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
                    if (wave + 1U < tiling.mWaveCount) { producer.Begin(wave + 1U); }
                }
                const uint32_t mRows = GetWaveRows(tiling, mIndex, wave);
                const uint64_t mStart = static_cast<uint64_t>(wave) * tiling.rowsPerWave +
                                        static_cast<uint64_t>(mIndex) * tiling.mTile;
                const uint32_t rowStart = subblock == 0U ? 0U : firstRows;
                const uint32_t rowCount = subblock == 0U ?
                                              (mRows < firstRows ? mRows : firstRows) :
                                              (mRows > firstRows ? mRows - firstRows : 0U);
                if (rowCount != 0U) {
                    Kernel::ZeroFp32Values(rmsRowsPtr, rowCount);
                }

                for (uint32_t tile = 0; tile < l1TileCount; ++tile) {
                    const uint32_t kCount = Kernel::PlannedL1TileKCount(tiling, kIndex, tile);
                    const uint64_t kTileStart = kStart + static_cast<uint64_t>(tile) * tiling.kL1;
                    const uint32_t activeK = kCount;
                    const uint32_t slot = (wave * l1TileCount + tile) & 1U;
                    const uint64_t stageSlot =
                        (static_cast<uint64_t>(aicIndex) * tiling.slotCount + slot) *
                        tiling.mTile * tiling.kL1 + static_cast<uint64_t>(rowStart) * tiling.kL1;

                    StageTransfer::AcquireSlot();
                    StageProjectionRows<T, StageTransfer, ASYNC_STAGE>(tensors, tiling, mStart + rowStart,
                        rowCount, kTileStart, activeK, stageSlot, x2Tile, prefetchInput,
                        stageTile, rmsRows, tile, l1TileCount, kIndex);
                    StageTransfer::PublishReady();
                    if constexpr (WAVE_PRODUCER) {
                        if (wave + 1U < tiling.mWaveCount) { producer.Tick(tile); }
                    }
                }

                if (rowCount != 0U) {
                    Kernel::CopyUbToGlobal<float>(tensors.partialRms,
                        static_cast<uint64_t>(kIndex) * tiling.totalRows + mStart + rowStart,
                        rmsRows, rowCount);
                }
            }
            if constexpr (WAVE_PRODUCER) { producer.End(); }
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        StageTransfer::DrainCredits();
        if constexpr (ASYNC_STAGE) { WaitFlag<HardEvent::MTE3_V>(EVENT_ID7); }
    }
}

}  // namespace AiInfraMegaMhc::Schedule

#endif  // OPS_TRANSFORMER_MHC_MEGA_MHC_MK_PROJECTION_H
