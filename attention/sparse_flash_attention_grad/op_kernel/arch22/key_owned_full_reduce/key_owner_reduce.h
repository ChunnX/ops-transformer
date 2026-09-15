#pragma once
#include "basic_modules/sparse_flash_attention_grad_common_header.h"
#include "layout.h"

namespace SFAG_KEY_OWNED_FULL_REDUCE {

__aicore__ inline void KeyOwnedMte3ScalarDrain(event_t event)
{
    SetFlag<HardEvent::MTE3_S>(event);
    WaitFlag<HardEvent::MTE3_S>(event);
}

template <typename TILING_CLASS>
__aicore__ inline void BuildKeyOwnedMetadata(GM_ADDR indices, GM_ADDR workspace, const TILING_CLASS *td, TPipe &pipe,
                                             uint32_t queryRangeBegin, uint32_t queryRangeEnd)
{
    LocalTensor<int32_t> input(TPosition::VECCALC, 0, 2048);
    LocalTensor<int32_t> sortScoreInt(TPosition::VECCALC, 8192, 2048);
    LocalTensor<float> sortScore(TPosition::VECCALC, 8192, 2048);
    LocalTensor<float> sorted(TPosition::VECCALC, 16384, 4096);
    LocalTensor<float> sortTmp(TPosition::VECCALC, 32768, 4096);
    LocalTensor<int32_t> ordered(TPosition::VECCALC, 49152, 2048);
    LocalTensor<int32_t> positions(TPosition::VECCALC, 49152, 2048);
    LocalTensor<uint32_t> header(TPosition::VECCALC, 57344, 2048);
    LocalTensor<float> coreTotals(TPosition::VECCALC, 65536, 2048);
    LocalTensor<int32_t> headerCounts(TPosition::VECCALC, 0, 256);
    LocalTensor<float> headerCountsFloat(TPosition::VECCALC, 1024, 256);
    // Keep the hoisted gather pattern outside input's [0, 8192) UB window.
    // Otherwise every per-query DataCopyPad(input, ...) overwrites the mask
    // and makes bucket totals depend on sparse_indices payload values.
    LocalTensor<uint32_t> gatherMask(TPosition::VECCALC, 73728, 2);
    GlobalTensor<int32_t> inputGm;
    GlobalTensor<int32_t> orderedGm;
    GlobalTensor<uint32_t> headersGm;
    GlobalTensor<float> coreTotalsGm;
    inputGm.SetGlobalBuffer((__gm__ int32_t *)indices);
    headersGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + KeyOwnedHeaderByteOffset(td)));
    coreTotalsGm.SetGlobalBuffer((__gm__ float *)(workspace + KeyOwnedCoreBucketTotalsByteOffset(td)));
    orderedGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + KeyOwnedSortedIndicesByteOffset(td)));
    const event_t scalarVector = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_V>());
    const event_t vectorScalar = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_S>());
    const event_t scalarMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_MTE2>());
    const event_t mte2Vector = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
    const event_t scalarMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_MTE3>());
    const event_t mte3Scalar = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_S>());
    const event_t mte3Vector = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
    const uint32_t vectorCoreCount = KeyOwnedVectorCoreCount(td);
    const uint32_t queryCount = static_cast<uint32_t>(KeyOwnedQueryCount(td));
    const uint32_t selectedCount = static_cast<uint32_t>(KeyOwnedSelectedCount(td));
    const uint32_t selectedCountAligned = (selectedCount + 31U) / 32U * 32U;
    const uint32_t sortRepeats = selectedCountAligned / 32U;
    const uint32_t bucketCount = static_cast<uint32_t>(KeyOwnedBucketCount(td));
    const bool accumulateBucketTotals = td->opInfo.B == 1;
    SetAtomicNone();
    if (accumulateBucketTotals) {
        SetFlag<HardEvent::S_V>(scalarVector);
        WaitFlag<HardEvent::S_V>(scalarVector);
        Duplicate<float>(coreTotals, 0.0f, bucketCount);
        SetFlag<HardEvent::V_S>(vectorScalar);
        WaitFlag<HardEvent::V_S>(vectorScalar);
        gatherMask.SetValue(0, 0x02020202U);
        gatherMask.SetValue(1, 0x02020202U);
    }
    for (uint32_t q = queryRangeBegin + GetBlockIdx(); q < queryRangeEnd; q += vectorCoreCount) {
        SetFlag<HardEvent::S_MTE2>(scalarMte2);
        WaitFlag<HardEvent::S_MTE2>(scalarMte2);
        DataCopyPad(input, inputGm[static_cast<uint64_t>(q) * selectedCount],
                    DataCopyExtParams{1, selectedCount * sizeof(int32_t), 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(mte2Vector);
        WaitFlag<HardEvent::MTE2_V>(mte2Vector);
        SetFlag<HardEvent::S_V>(scalarVector);
        WaitFlag<HardEvent::S_V>(scalarVector);
        Duplicate<float>(sortScore, -3.402823466e+38F, selectedCountAligned);
        PipeBarrier<PIPE_V>();
        ShiftRight(sortScoreInt, input, 6, selectedCount);
        PipeBarrier<PIPE_V>();
        Muls(sortScoreInt, sortScoreInt, 2048, selectedCount);
        PipeBarrier<PIPE_V>();
        CreateVecIndex(positions, 0, selectedCount);
        PipeBarrier<PIPE_V>();
        Add(sortScoreInt, sortScoreInt, positions, selectedCount);
        PipeBarrier<PIPE_V>();
        Cast(sortScore, sortScoreInt, RoundMode::CAST_NONE, selectedCount);
        PipeBarrier<PIPE_V>();
        Muls(sortScore, sortScore, -1.0f, selectedCount);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(sorted, sortScore, input.ReinterpretCast<uint32_t>(), sortTmp, sortRepeats);
        PipeBarrier<PIPE_V>();
        Extract(sortScore, ordered.ReinterpretCast<uint32_t>(), sorted, sortRepeats);
        SetFlag<HardEvent::V_S>(vectorScalar);
        WaitFlag<HardEvent::V_S>(vectorScalar);
        SetFlag<HardEvent::S_MTE3>(scalarMte3);
        WaitFlag<HardEvent::S_MTE3>(scalarMte3);
        DataCopyPad(orderedGm[static_cast<uint64_t>(q) * selectedCount], ordered,
                    DataCopyExtParams{1, selectedCount * sizeof(int32_t), 0, 0, 0});
        uint32_t cursor = 0;
        for (uint32_t chunk = 0; chunk < bucketCount; chunk += 256) {
            KeyOwnedMte3ScalarDrain(mte3Scalar);
            const uint32_t chunkBuckets = Min<uint32_t>(256, bucketCount - chunk);
            for (uint32_t record = 0; record < chunkBuckets; ++record) {
                const uint32_t bucket = chunk + record;
                const uint32_t start = cursor;
                while (cursor < selectedCount &&
                       (static_cast<uint32_t>(ordered.GetValue(cursor)) >> 6) == bucket) {
                    ++cursor;
                }
                const uint32_t count = cursor - start;
                const uint32_t headerOffset = record * 8;
                header.SetValue(headerOffset, start);
                header.SetValue(headerOffset + 1, count);
                const uint32_t inlineCount = Min<uint32_t>(count, 4);
                for (uint32_t inlineIndex = 0; inlineIndex < inlineCount; ++inlineIndex) {
                    header.SetValue(headerOffset + inlineIndex + 2,
                                    static_cast<uint32_t>(ordered.GetValue(start + inlineIndex)));
                }
            }
            SetFlag<HardEvent::S_MTE3>(scalarMte3);
            WaitFlag<HardEvent::S_MTE3>(scalarMte3);
            const uint64_t headerWordOffset = (static_cast<uint64_t>(chunk) * queryCount + q) * 8;
            DataCopyPad(headersGm[headerWordOffset], header,
                        DataCopyExtParams{static_cast<uint16_t>(chunkBuckets), 32, 0,
                                          static_cast<uint32_t>(queryCount * 32 - 32), 0});
            if (accumulateBucketTotals) {
                SetFlag<HardEvent::MTE3_V>(mte3Vector);
                WaitFlag<HardEvent::MTE3_V>(mte3Vector);
                GatherMaskParams gatherParams{1, static_cast<uint8_t>((chunkBuckets + 7) / 8), 8, 0};
                uint64_t selected = 0;
                GatherMask(headerCounts, header.ReinterpretCast<int32_t>(), gatherMask, false, 0, gatherParams,
                           selected);
                PipeBarrier<PIPE_V>();
                Cast(headerCountsFloat, headerCounts, RoundMode::CAST_NONE, chunkBuckets);
                PipeBarrier<PIPE_V>();
                Add(coreTotals[chunk], coreTotals[chunk], headerCountsFloat, chunkBuckets);
                SetFlag<HardEvent::V_S>(vectorScalar);
                WaitFlag<HardEvent::V_S>(vectorScalar);
            }
        }
        KeyOwnedMte3ScalarDrain(mte3Scalar);
    }
    if (accumulateBucketTotals) {
        SetFlag<HardEvent::S_MTE3>(scalarMte3);
        WaitFlag<HardEvent::S_MTE3>(scalarMte3);
        DataCopyPad(coreTotalsGm[static_cast<uint64_t>(GetBlockIdx()) * bucketCount], coreTotals,
                    DataCopyExtParams{1, bucketCount * sizeof(uint32_t), 0, 0, 0});
        KeyOwnedMte3ScalarDrain(mte3Scalar);
    }
    KeyOwnedMte3ScalarDrain(mte3Scalar);
}

template <typename TILING_CLASS>
__aicore__ inline void BuildKeyOwnedBucketTotalsBs1(GM_ADDR workspace, const TILING_CLASS *td, TPipe &pipe,
                                                    uint32_t queryRangeBegin, uint32_t queryRangeEnd)
{
    constexpr uint32_t bucketsPerChunk = 64;
    LocalTensor<float> totals(TPosition::VECCALC, 0, bucketsPerChunk);
    LocalTensor<float> coreCounts(TPosition::VECCALC, 256, bucketsPerChunk);
    LocalTensor<uint32_t> outputCounts(TPosition::VECCALC, 512, bucketsPerChunk);
    GlobalTensor<float> coreTotalsGm;
    GlobalTensor<uint32_t> totalsGm;
    coreTotalsGm.SetGlobalBuffer((__gm__ float *)(workspace + KeyOwnedCoreBucketTotalsByteOffset(td)));
    totalsGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + KeyOwnedBucketTotalsByteOffset(td)));
    const event_t scalarVector = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_V>());
    const event_t vectorMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
    const event_t mte2Vector = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
    const event_t vectorMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
    const event_t mte3Vector = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
    const uint32_t vectorCoreCount = KeyOwnedVectorCoreCount(td);
    const uint32_t bucketCount = static_cast<uint32_t>(KeyOwnedBucketCount(td));
    (void)queryRangeBegin;
    (void)queryRangeEnd;

    for (uint32_t bucketBase = GetBlockIdx() * bucketsPerChunk; bucketBase < bucketCount;
         bucketBase += vectorCoreCount * bucketsPerChunk) {
        const uint32_t chunkBuckets = Min<uint32_t>(bucketsPerChunk, bucketCount - bucketBase);
        SetFlag<HardEvent::S_V>(scalarVector);
        WaitFlag<HardEvent::S_V>(scalarVector);
        Duplicate<float>(totals, 0.0f, chunkBuckets);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(vectorMte2);
        for (uint32_t core = 0; core < vectorCoreCount; ++core) {
            WaitFlag<HardEvent::V_MTE2>(vectorMte2);
            DataCopyPad(coreCounts, coreTotalsGm[static_cast<uint64_t>(core) * bucketCount + bucketBase],
                        DataCopyExtParams{1, chunkBuckets * sizeof(uint32_t), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(mte2Vector);
            WaitFlag<HardEvent::MTE2_V>(mte2Vector);
            Add(totals, totals, coreCounts, chunkBuckets);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(vectorMte2);
        }
        WaitFlag<HardEvent::V_MTE2>(vectorMte2);
        Cast(outputCounts.ReinterpretCast<int32_t>(), totals, RoundMode::CAST_RINT, chunkBuckets);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(vectorMte3);
        WaitFlag<HardEvent::V_MTE3>(vectorMte3);
        DataCopyPad(totalsGm[bucketBase], outputCounts,
                    DataCopyExtParams{1, chunkBuckets * sizeof(uint32_t), 0, 0, 0});
        SetFlag<HardEvent::MTE3_V>(mte3Vector);
        WaitFlag<HardEvent::MTE3_V>(mte3Vector);
    }
    SyncAll();
}

template <typename TILING_CLASS>
__aicore__ inline void BuildKeyOwnedBucketTotalsGeneric(GM_ADDR workspace, const TILING_CLASS *td, TPipe &pipe,
                                                        uint32_t queryRangeBegin, uint32_t queryRangeEnd)
{
    constexpr uint32_t ownersPerChunk = 256;
    constexpr uint32_t queriesPerChunk = 1024;
    LocalTensor<uint32_t> headers(TPosition::VECCALC, 0, queriesPerChunk * 8);
    LocalTensor<uint32_t> totals(TPosition::VECCALC, 32768, ownersPerChunk);
    GlobalTensor<uint32_t> headersGm, totalsGm;
    headersGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + KeyOwnedHeaderByteOffset(td)));
    totalsGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + KeyOwnedBucketTotalsByteOffset(td)));
    const event_t scalarMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_MTE2>());
    const event_t mte2Scalar = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_S>());
    const event_t scalarMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_MTE3>());
    const event_t mte3Scalar = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_S>());
    const uint32_t vectorCoreCount = KeyOwnedVectorCoreCount(td);
    const uint32_t queryCount = static_cast<uint32_t>(KeyOwnedQueryCount(td));
    const uint32_t queryBatchSize = td->opInfo.S1;
    const uint32_t bucketCount = static_cast<uint32_t>(KeyOwnedBucketCount(td));
    const uint32_t ownerCount = static_cast<uint32_t>(KeyOwnedOwnerCount(td));

    for (uint32_t ownerBase = GetBlockIdx() * ownersPerChunk; ownerBase < ownerCount;
         ownerBase += vectorCoreCount * ownersPerChunk) {
        const uint32_t ownerEnd = Min<uint32_t>(ownerCount, ownerBase + ownersPerChunk);
        for (uint32_t owner = ownerBase; owner < ownerEnd; ++owner) {
            const uint32_t bucket = owner % bucketCount;
            const uint32_t batch = owner / bucketCount;
            const uint32_t ownerQBegin = batch * queryBatchSize;
            const uint32_t ownerQEnd = Min<uint32_t>(queryCount, ownerQBegin + queryBatchSize);
            const uint32_t qBegin = Max<uint32_t>(ownerQBegin, queryRangeBegin);
            const uint32_t qEnd = Min<uint32_t>(ownerQEnd, queryRangeEnd);
            uint32_t total = 0;
            for (uint32_t qBase = qBegin; qBase < qEnd; qBase += queriesPerChunk) {
                const uint32_t qRows = Min<uint32_t>(queriesPerChunk, qEnd - qBase);
                SetFlag<HardEvent::S_MTE2>(scalarMte2);
                WaitFlag<HardEvent::S_MTE2>(scalarMte2);
                DataCopyPad(headers, headersGm[(static_cast<uint64_t>(bucket) * queryCount + qBase) * 8],
                            DataCopyExtParams{1, qRows * 8 * sizeof(uint32_t), 0, 0, 0},
                            DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_S>(mte2Scalar);
                WaitFlag<HardEvent::MTE2_S>(mte2Scalar);
                for (uint32_t localQ = 0; localQ < qRows; ++localQ) {
                    total += headers.GetValue(localQ * 8 + 1);
                }
            }
            totals.SetValue(owner - ownerBase, total);
        }
        SetFlag<HardEvent::S_MTE3>(scalarMte3);
        WaitFlag<HardEvent::S_MTE3>(scalarMte3);
        DataCopyPad(totalsGm[ownerBase], totals,
                    DataCopyExtParams{1, (ownerEnd - ownerBase) * sizeof(uint32_t), 0, 0, 0});
        KeyOwnedMte3ScalarDrain(mte3Scalar);
    }
    KeyOwnedMte3ScalarDrain(mte3Scalar);
    SyncAll();
}

template <typename TILING_CLASS>
__aicore__ inline void BuildKeyOwnedBucketTotals(GM_ADDR workspace, const TILING_CLASS *td, TPipe &pipe,
                                                 uint32_t queryRangeBegin, uint32_t queryRangeEnd)
{
    if (td->opInfo.B == 1) {
        BuildKeyOwnedBucketTotalsBs1(workspace, td, pipe, queryRangeBegin, queryRangeEnd);
    } else {
        BuildKeyOwnedBucketTotalsGeneric(workspace, td, pipe, queryRangeBegin, queryRangeEnd);
    }
}

__aicore__ inline void ReduceLoadedKeyOwnedBatch(LocalTensor<float> &accumulator, LocalTensor<float> &gradients,
                                                 LocalTensor<uint32_t> &batchKeys, uint32_t rows, uint32_t width,
                                                 event_t ready, event_t release)
{
    SetFlag<HardEvent::MTE2_V>(ready);
    WaitFlag<HardEvent::MTE2_V>(ready);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t localKey = batchKeys.GetValue(row);
        Add(accumulator[localKey * width], accumulator[localKey * width], gradients[row * width], width);
        // Repeated keys must retain the stable q,t accumulation order.
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE2>(release);
}

__aicore__ inline void WriteKeyOwnedAccumulator(GlobalTensor<float> &outputGm, LocalTensor<float> &accumulator,
                                                uint64_t outputOffset, uint32_t outputRows, uint32_t edgeWidth,
                                                event_t vectorMte3, event_t mte3Vector, event_t release)
{
    SetFlag<HardEvent::V_MTE3>(vectorMte3);
    WaitFlag<HardEvent::V_MTE3>(vectorMte3);
    const uint32_t firstRows = Min<uint32_t>(32, outputRows);
    if (firstRows > 0) {
        DataCopy(outputGm[outputOffset], accumulator, firstRows * edgeWidth);
    }
    if (outputRows > 32) {
        DataCopy(outputGm[outputOffset + 32 * edgeWidth], accumulator[32 * edgeWidth],
                 (outputRows - 32) * edgeWidth);
    }
    SetFlag<HardEvent::MTE3_V>(mte3Vector);
    WaitFlag<HardEvent::MTE3_V>(mte3Vector);
    SetFlag<HardEvent::V_MTE2>(release);
}

__aicore__ inline void ReadKeyOwnedAccumulator(GlobalTensor<float> &outputGm, LocalTensor<float> &accumulator,
                                               uint64_t outputOffset, uint32_t outputRows, uint32_t edgeWidth,
                                               event_t ready, event_t release)
{
    WaitFlag<HardEvent::V_MTE2>(release);
    const uint32_t firstRows = Min<uint32_t>(32, outputRows);
    if (firstRows > 0) {
        DataCopy(accumulator, outputGm[outputOffset], firstRows * edgeWidth);
    }
    if (outputRows > 32) {
        DataCopy(accumulator[32 * edgeWidth], outputGm[outputOffset + 32 * edgeWidth],
                 (outputRows - 32) * edgeWidth);
    }
    SetFlag<HardEvent::MTE2_V>(ready);
    WaitFlag<HardEvent::MTE2_V>(ready);
    SetFlag<HardEvent::V_MTE2>(release);
}

template <typename TILING_CLASS>
__aicore__ inline void ReduceKeyOwnedEdgeRange(
    GlobalTensor<float> &edgeGm, GlobalTensor<int32_t> &orderedGm, GlobalTensor<uint32_t> &headersGm,
    GlobalTensor<float> &outputGm, const TILING_CLASS *td, uint32_t selectedCount, uint32_t owner,
    uint32_t queryRangeBegin, uint32_t queryRangeEnd, uint32_t edgeQueryBase, uint64_t rangeBegin,
    uint64_t rangeEnd, uint64_t outputOffset, uint32_t outputRows, bool accumulateOutput,
    LocalTensor<float> &accumulator, LocalTensor<uint32_t> &headers,
    LocalTensor<int32_t> &ordered, LocalTensor<float> &gradients, LocalTensor<uint32_t> &batchKeys,
    event_t ready, event_t release, event_t scalarMte2, event_t mte2Scalar, event_t vectorMte3,
    event_t mte3Vector)
{
    const uint32_t queryCount = static_cast<uint32_t>(KeyOwnedQueryCount(td));
    const uint32_t bucketCount = static_cast<uint32_t>(KeyOwnedBucketCount(td));
    const uint32_t queryBatchSize = td->opInfo.S1;
    const uint32_t edgeWidth = static_cast<uint32_t>((static_cast<uint64_t>(td->opInfo.D) + td->opInfo.ropeD + 15) / 16 * 16);
    const uint32_t bucket = owner % bucketCount;
    const uint32_t batch = owner / bucketCount;
    const uint32_t ownerQBegin = batch * queryBatchSize;
    const uint32_t ownerQEnd = Min<uint32_t>(queryCount, ownerQBegin + queryBatchSize);
    const uint32_t qBegin = Max<uint32_t>(ownerQBegin, queryRangeBegin);
    const uint32_t qEnd = Min<uint32_t>(ownerQEnd, queryRangeEnd);
    if (accumulateOutput) {
        ReadKeyOwnedAccumulator(outputGm, accumulator, outputOffset, outputRows, edgeWidth, ready, release);
    } else {
        Duplicate(accumulator, 0.0f, KeyOwnedBucketWidth * edgeWidth);
        PipeBarrier<PIPE_V>();
    }
    uint32_t batchRows = 0;
    uint64_t seenRows = 0;
    bool complete = rangeBegin >= rangeEnd;
    if (!complete) {
        for (uint32_t qBase = qBegin; qBase < qEnd; qBase += 256) {
            const uint32_t qRows = Min<uint32_t>(256, qEnd - qBase);
            SetFlag<HardEvent::S_MTE2>(scalarMte2);
            WaitFlag<HardEvent::S_MTE2>(scalarMte2);
            DataCopyPad(headers, headersGm[(static_cast<uint64_t>(bucket) * queryCount + qBase) * 8],
                        DataCopyExtParams{1, qRows * 8 * sizeof(uint32_t), 0, 0, 0},
                        DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(mte2Scalar);
            WaitFlag<HardEvent::MTE2_S>(mte2Scalar);
            for (uint32_t localQ = 0; localQ < qRows; ++localQ) {
                const uint32_t q = qBase + localQ;
                const uint32_t start = headers.GetValue(localQ * 8);
                const uint32_t count = headers.GetValue(localQ * 8 + 1);
                const uint64_t nextSeenRows = seenRows + count;
                if (nextSeenRows <= rangeBegin) {
                    seenRows = nextSeenRows;
                    continue;
                }
                if (seenRows >= rangeEnd) {
                    complete = true;
                    break;
                }
                const uint32_t rowBegin = rangeBegin > seenRows ? static_cast<uint32_t>(rangeBegin - seenRows) : 0;
                const uint32_t rowEnd = rangeEnd < nextSeenRows ? static_cast<uint32_t>(rangeEnd - seenRows) : count;
                for (uint32_t row = rowBegin; row < rowEnd;) {
                    if (batchRows == 0) {
                        WaitFlag<HardEvent::V_MTE2>(release);
                    }
                    const uint32_t availableRows = 16 - batchRows;
                    const uint32_t remainingRows = rowEnd - row;
                    const uint32_t rows = remainingRows < availableRows ? remainingRows : availableRows;
                    if (count > 4) {
                        SetFlag<HardEvent::S_MTE2>(scalarMte2);
                        WaitFlag<HardEvent::S_MTE2>(scalarMte2);
                        DataCopyPad(ordered, orderedGm[static_cast<uint64_t>(q) * selectedCount + start + row],
                                    DataCopyExtParams{1, static_cast<uint32_t>(rows * sizeof(int32_t)), 0, 0, 0},
                                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                        SetFlag<HardEvent::MTE2_S>(mte2Scalar);
                        WaitFlag<HardEvent::MTE2_S>(mte2Scalar);
                    }
                    for (uint32_t i = 0; i < rows; ++i) {
                        const uint32_t key = count <= 4 ? headers.GetValue(localQ * 8 + row + i + 2) :
                                                        static_cast<uint32_t>(ordered.GetValue(i));
                        batchKeys.SetValue(batchRows + i, key & 63U);
                    }
                    const uint64_t edgeOffset =
                        (static_cast<uint64_t>(q - edgeQueryBase) * selectedCount + start + row) * edgeWidth;
                    DataCopy(gradients[batchRows * edgeWidth], edgeGm[edgeOffset], rows * edgeWidth);
                    batchRows += rows;
                    row += rows;
                    if (batchRows == 16) {
                        ReduceLoadedKeyOwnedBatch(accumulator, gradients, batchKeys, batchRows, edgeWidth,
                                                 ready, release);
                        batchRows = 0;
                    }
                }
                seenRows = nextSeenRows;
                if (seenRows >= rangeEnd) {
                    complete = true;
                    break;
                }
            }
            if (complete) {
                break;
            }
        }
    }
    if (batchRows != 0) {
        ReduceLoadedKeyOwnedBatch(accumulator, gradients, batchKeys, batchRows, edgeWidth, ready, release);
    }
    WaitFlag<HardEvent::V_MTE2>(release);
    WriteKeyOwnedAccumulator(outputGm, accumulator, outputOffset, outputRows, edgeWidth, vectorMte3, mte3Vector,
                             release);
}

__aicore__ inline uint32_t KeyOwnedPartitionCount(uint32_t totalRows, uint64_t targetRows,
                                                  uint32_t vectorCoreCount)
{
    if (totalRows <= targetRows) {
        return 1;
    }
    const uint64_t partitions = (static_cast<uint64_t>(totalRows) + targetRows - 1) / targetRows;
    return static_cast<uint32_t>(Min<uint64_t>(partitions, vectorCoreCount));
}

__aicore__ inline void ReduceKeyOwnedPartials(
    GlobalTensor<float> &partialGm, GlobalTensor<float> &dkGm, uint64_t partialBase, uint32_t partitions,
    uint64_t outputOffset, uint32_t outputRows, uint32_t edgeWidth, LocalTensor<float> &accumulator,
    LocalTensor<float> &gradients, event_t ready, event_t release, event_t vectorMte3, event_t mte3Vector,
    bool accumulateOutput)
{
    if (accumulateOutput) {
        ReadKeyOwnedAccumulator(dkGm, accumulator, outputOffset, outputRows, edgeWidth, ready, release);
    } else {
        Duplicate(accumulator, 0.0f, KeyOwnedBucketWidth * edgeWidth);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t partition = 0; partition < partitions; ++partition) {
        const uint64_t partialOffset = (partialBase + partition) * KeyOwnedBucketWidth * edgeWidth;
        for (uint32_t row = 0; row < KeyOwnedBucketWidth; row += 16) {
            const uint32_t rows = Min<uint32_t>(16, KeyOwnedBucketWidth - row);
            WaitFlag<HardEvent::V_MTE2>(release);
            DataCopy(gradients, partialGm[partialOffset + row * edgeWidth], rows * edgeWidth);
            SetFlag<HardEvent::MTE2_V>(ready);
            WaitFlag<HardEvent::MTE2_V>(ready);
            Add(accumulator[row * edgeWidth], accumulator[row * edgeWidth], gradients, rows * edgeWidth);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(release);
        }
    }
    WaitFlag<HardEvent::V_MTE2>(release);
    WriteKeyOwnedAccumulator(dkGm, accumulator, outputOffset, outputRows, edgeWidth, vectorMte3, mte3Vector, release);
}

template <typename TILING_CLASS>
__aicore__ inline void ReduceKeyOwnedEdges(GM_ADDR workspace, const TILING_CLASS *td, TPipe &pipe,
                                           uint32_t queryRangeBegin, uint32_t queryRangeEnd, bool firstTile)
{
    LocalTensor<float> accumulator(TPosition::VECCALC, 0, 64 * 576);
    LocalTensor<uint32_t> headers(TPosition::VECCALC, 147456, 256 * 8);
    LocalTensor<int32_t> ordered(TPosition::VECCALC, 155648, 16);
    LocalTensor<float> gradients(TPosition::VECCALC, 155712, 16 * 576);
    LocalTensor<uint32_t> batchKeys(TPosition::VECCALC, 192576, 16);
    GlobalTensor<float> edgeGm, dkGm, partialGm;
    GlobalTensor<int32_t> orderedGm;
    GlobalTensor<uint32_t> headersGm, totalsGm;
    __gm__ uint8_t *extension = (__gm__ uint8_t *)workspace + KeyOwnedExtensionByteOffset(td);
    edgeGm.SetGlobalBuffer((__gm__ float *)extension);
    headersGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + KeyOwnedHeaderByteOffset(td)));
    orderedGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + KeyOwnedSortedIndicesByteOffset(td)));
    totalsGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + KeyOwnedBucketTotalsByteOffset(td)));
    partialGm.SetGlobalBuffer((__gm__ float *)(workspace + KeyOwnedPartialByteOffset(td)));
    dkGm.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)workspace + KeyOwnedPostByteOffset(td) +
                                       static_cast<uint64_t>(td->opInfo.dqWorkspaceLen)));
    const event_t ready = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
    const event_t release = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
    const event_t scalarMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_MTE2>());
    const event_t mte2Scalar = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_S>());
    const event_t vectorMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
    const event_t mte3Vector = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
    const uint32_t vectorCoreCount = KeyOwnedVectorCoreCount(td);
    const uint32_t bucketCount = static_cast<uint32_t>(KeyOwnedBucketCount(td));
    const uint32_t ownerCount = static_cast<uint32_t>(KeyOwnedOwnerCount(td));
    const uint32_t selectedCount = static_cast<uint32_t>(KeyOwnedSelectedCount(td));
    const uint32_t s2 = td->opInfo.S2;
    const uint32_t d = td->opInfo.D;
    const uint32_t ropeD = td->opInfo.ropeD;
    const uint32_t edgeWidth = (d + ropeD + 15) / 16 * 16;
    const uint64_t totalRows = static_cast<uint64_t>(queryRangeEnd - queryRangeBegin) * selectedCount;
    const uint64_t targetRows = (totalRows + vectorCoreCount - 1) / vectorCoreCount;
    const uint32_t block = GetBlockIdx();
    SetAtomicNone();
    SetFlag<HardEvent::V_MTE2>(release);

    uint64_t partialBase = 0;
    for (uint32_t owner = 0; owner < ownerCount; ++owner) {
        const uint32_t ownerRows = totalsGm.GetValue(owner);
        if (ownerRows == 0 && !firstTile) {
            continue;
        }
        const uint32_t partitions = KeyOwnedPartitionCount(ownerRows, targetRows, vectorCoreCount);
        const uint32_t bucket = owner % bucketCount;
        const uint32_t batch = owner / bucketCount;
        const uint32_t outputRows = Min<uint32_t>(KeyOwnedBucketWidth, s2 - bucket * KeyOwnedBucketWidth);
        const uint64_t outputOffset =
            (static_cast<uint64_t>(batch) * s2 + bucket * KeyOwnedBucketWidth) * edgeWidth;
        if (partitions == 1) {
            if (owner % vectorCoreCount == block) {
                ReduceKeyOwnedEdgeRange(edgeGm, orderedGm, headersGm, dkGm, td, selectedCount, owner,
                                        queryRangeBegin, queryRangeEnd, queryRangeBegin, 0, ownerRows, outputOffset,
                                        outputRows, !firstTile, accumulator, headers, ordered, gradients, batchKeys,
                                        ready, release, scalarMte2, mte2Scalar, vectorMte3, mte3Vector);
            }
            continue;
        }

        const uint32_t partition = (block + vectorCoreCount - owner % vectorCoreCount) % vectorCoreCount;
        if (partition < partitions) {
            const uint64_t rangeBegin = static_cast<uint64_t>(ownerRows) * partition / partitions;
            const uint64_t rangeEnd = static_cast<uint64_t>(ownerRows) * (partition + 1) / partitions;
            const uint64_t partialOffset = (partialBase + partition) * KeyOwnedBucketWidth * edgeWidth;
            ReduceKeyOwnedEdgeRange(edgeGm, orderedGm, headersGm, partialGm, td, selectedCount, owner,
                                    queryRangeBegin, queryRangeEnd, queryRangeBegin, rangeBegin, rangeEnd,
                                    partialOffset, KeyOwnedBucketWidth, false, accumulator, headers, ordered,
                                    gradients, batchKeys, ready, release, scalarMte2, mte2Scalar, vectorMte3,
                                    mte3Vector);
        }
        partialBase += partitions;
    }

    WaitFlag<HardEvent::V_MTE2>(release);
    SyncAll();
    SetFlag<HardEvent::V_MTE2>(release);
    partialBase = 0;
    for (uint32_t owner = 0; owner < ownerCount; ++owner) {
        const uint32_t ownerRows = totalsGm.GetValue(owner);
        if (ownerRows == 0 && !firstTile) {
            continue;
        }
        const uint32_t partitions = KeyOwnedPartitionCount(ownerRows, targetRows, vectorCoreCount);
        if (partitions == 1) {
            continue;
        }
        if (owner % vectorCoreCount == block) {
            const uint32_t bucket = owner % bucketCount;
            const uint32_t batch = owner / bucketCount;
            const uint32_t outputRows =
                Min<uint32_t>(KeyOwnedBucketWidth, s2 - bucket * KeyOwnedBucketWidth);
            const uint64_t outputOffset =
                (static_cast<uint64_t>(batch) * s2 + bucket * KeyOwnedBucketWidth) * edgeWidth;
            ReduceKeyOwnedPartials(partialGm, dkGm, partialBase, partitions, outputOffset, outputRows, edgeWidth,
                                   accumulator, gradients, ready, release, vectorMte3, mte3Vector, !firstTile);
        }
        partialBase += partitions;
    }
    WaitFlag<HardEvent::V_MTE2>(release);
}

} // namespace SFAG_KEY_OWNED_FULL_REDUCE
