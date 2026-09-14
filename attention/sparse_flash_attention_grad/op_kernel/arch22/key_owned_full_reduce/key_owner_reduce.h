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
__aicore__ inline void BuildKeyOwnedMetadata(GM_ADDR indices, GM_ADDR workspace, const TILING_CLASS *td, TPipe &pipe)
{
    TBuf<> buffer;
    pipe.InitBuffer(buffer, 32768);
    LocalTensor<int32_t> input = buffer.GetWithOffset<int32_t>(2048, 0);
    LocalTensor<uint32_t> counts = buffer.GetWithOffset<uint32_t>(2048, 8192);
    LocalTensor<int32_t> ordered = buffer.GetWithOffset<int32_t>(2048, 16384);
    LocalTensor<uint32_t> header = buffer.GetWithOffset<uint32_t>(2048, 24576);
    GlobalTensor<int32_t> inputGm;
    GlobalTensor<int32_t> orderedGm;
    GlobalTensor<uint32_t> headersGm;
    inputGm.SetGlobalBuffer((__gm__ int32_t *)indices);
    __gm__ uint8_t *extension = (__gm__ uint8_t *)workspace + KeyOwnedExtensionByteOffset(td);
    headersGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + KeyOwnedHeaderByteOffset(td)));
    orderedGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + KeyOwnedSortedIndicesByteOffset(td)));
    const event_t scalarVector = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_V>());
    const event_t vectorScalar = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_S>());
    const event_t scalarMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_MTE2>());
    const event_t mte2Scalar = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_S>());
    const event_t scalarMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_MTE3>());
    const event_t mte3Scalar = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_S>());
    const uint32_t vectorCoreCount = KeyOwnedVectorCoreCount(td);
    const uint32_t queryCount = static_cast<uint32_t>(KeyOwnedQueryCount(td));
    const uint32_t selectedCount = static_cast<uint32_t>(KeyOwnedSelectedCount(td));
    const uint32_t bucketCount = static_cast<uint32_t>(KeyOwnedBucketCount(td));
    SetAtomicNone();
    for (uint32_t q = GetBlockIdx(); q < queryCount; q += vectorCoreCount) {
        SetFlag<HardEvent::S_V>(scalarVector);
        WaitFlag<HardEvent::S_V>(scalarVector);
        Duplicate<uint32_t>(counts, 0, 2048);
        SetFlag<HardEvent::V_S>(vectorScalar);
        WaitFlag<HardEvent::V_S>(vectorScalar);
        SetFlag<HardEvent::S_MTE2>(scalarMte2);
        WaitFlag<HardEvent::S_MTE2>(scalarMte2);
        DataCopyPad(input, inputGm[static_cast<uint64_t>(q) * selectedCount],
                    DataCopyExtParams{1, selectedCount * sizeof(int32_t), 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(mte2Scalar);
        WaitFlag<HardEvent::MTE2_S>(mte2Scalar);
        for (uint32_t t = 0; t < selectedCount; ++t) {
            const uint32_t bucket = static_cast<uint32_t>(input.GetValue(t)) >> 6;
            counts.SetValue(bucket, counts.GetValue(bucket) + 1);
        }
        uint32_t next = 0;
        for (uint32_t bucket = 0; bucket < bucketCount; ++bucket) {
            const uint32_t count = counts.GetValue(bucket);
            counts.SetValue(bucket, next);
            next += count;
        }
        for (uint32_t t = 0; t < selectedCount; ++t) {
            const int32_t key = input.GetValue(t);
            const uint32_t bucket = static_cast<uint32_t>(key) >> 6;
            const uint32_t cursor = counts.GetValue(bucket);
            ordered.SetValue(cursor, key);
            counts.SetValue(bucket, cursor + 1);
        }
        SetFlag<HardEvent::S_MTE3>(scalarMte3);
        WaitFlag<HardEvent::S_MTE3>(scalarMte3);
        DataCopyPad(orderedGm[static_cast<uint64_t>(q) * selectedCount], ordered,
                    DataCopyExtParams{1, selectedCount * sizeof(int32_t), 0, 0, 0});
        for (uint32_t chunk = 0; chunk < bucketCount; chunk += 256) {
            KeyOwnedMte3ScalarDrain(mte3Scalar);
            const uint32_t chunkBuckets = Min<uint32_t>(256, bucketCount - chunk);
            for (uint32_t record = 0; record < chunkBuckets; ++record) {
                const uint32_t bucket = chunk + record;
                const uint32_t end = counts.GetValue(bucket);
                const uint32_t start = bucket == 0 ? 0 : counts.GetValue(bucket - 1);
                const uint32_t count = end - start;
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
        }
        KeyOwnedMte3ScalarDrain(mte3Scalar);
    }
    KeyOwnedMte3ScalarDrain(mte3Scalar);
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

template <typename TILING_CLASS>
__aicore__ inline void ReduceKeyOwnedEdges(GM_ADDR workspace, const TILING_CLASS *td, TPipe &pipe)
{
    TBuf<> buffer;
    pipe.InitBuffer(buffer, 192640);
    LocalTensor<float> accumulator = buffer.GetWithOffset<float>(64 * 576, 0);
    LocalTensor<uint32_t> headers = buffer.GetWithOffset<uint32_t>(256 * 8, 147456);
    LocalTensor<int32_t> ordered = buffer.GetWithOffset<int32_t>(16, 155648);
    LocalTensor<float> gradients = buffer.GetWithOffset<float>(16 * 576, 155712);
    LocalTensor<uint32_t> batchKeys = buffer.GetWithOffset<uint32_t>(16, 192576);
    GlobalTensor<float> edgeGm, dkGm;
    GlobalTensor<int32_t> orderedGm;
    GlobalTensor<uint32_t> headersGm;
    __gm__ uint8_t *extension = (__gm__ uint8_t *)workspace + KeyOwnedExtensionByteOffset(td);
    edgeGm.SetGlobalBuffer((__gm__ float *)extension);
    headersGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + KeyOwnedHeaderByteOffset(td)));
    orderedGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + KeyOwnedSortedIndicesByteOffset(td)));
    dkGm.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)workspace + KeyOwnedPostByteOffset(td) +
                                       static_cast<uint64_t>(td->opInfo.dqWorkspaceLen)));
    const event_t ready = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
    const event_t release = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
    const event_t scalarMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_MTE2>());
    const event_t mte2Scalar = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_S>());
    const event_t vectorMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
    const event_t mte3Vector = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
    const uint32_t vectorCoreCount = KeyOwnedVectorCoreCount(td);
    const uint32_t queryCount = static_cast<uint32_t>(KeyOwnedQueryCount(td));
    const uint32_t bucketCount = static_cast<uint32_t>(KeyOwnedBucketCount(td));
    const uint32_t queryBatchSize = td->opInfo.S1;
    const uint32_t edgeWidth = static_cast<uint32_t>((static_cast<uint64_t>(td->opInfo.D) + td->opInfo.ropeD + 15) / 16 * 16);
    const uint32_t ownerCount = td->opInfo.B * bucketCount;
    SetAtomicNone();
    SetFlag<HardEvent::V_MTE2>(release);
    for (uint32_t owner = GetBlockIdx(); owner < ownerCount; owner += vectorCoreCount) {
        const uint32_t bucket = owner % bucketCount;
        const uint32_t batch = owner / bucketCount;
        const uint32_t qBegin = batch * queryBatchSize;
        const uint32_t qEnd = qBegin + queryBatchSize;
        Duplicate(accumulator, 0.0f, KeyOwnedBucketWidth * edgeWidth);
        PipeBarrier<PIPE_V>();
        uint32_t batchRows = 0;
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
                if (count == 0) {
                    continue;
                }
                for (uint32_t row = 0; row < count;) {
                    if (batchRows == 0) {
                        WaitFlag<HardEvent::V_MTE2>(release);
                    }
                    const uint32_t availableRows = 16 - batchRows;
                    const uint32_t remainingRows = count - row;
                    const uint32_t rows = remainingRows < availableRows ? remainingRows : availableRows;
                    if (count > 4) {
                        SetFlag<HardEvent::S_MTE2>(scalarMte2);
                        WaitFlag<HardEvent::S_MTE2>(scalarMte2);
                        DataCopyPad(ordered, orderedGm[static_cast<uint64_t>(q) * KeyOwnedSelectedCount(td) + start + row],
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
                    const uint64_t edgeOffset = (static_cast<uint64_t>(q) * td->opInfo.selectedBlockCount + start + row) * edgeWidth;
                    DataCopy(gradients[batchRows * edgeWidth], edgeGm[edgeOffset], rows * edgeWidth);
                    batchRows += rows;
                    row += rows;
                    if (batchRows == 16) {
                        ReduceLoadedKeyOwnedBatch(accumulator, gradients, batchKeys, batchRows, edgeWidth,
                                                 ready, release);
                        batchRows = 0;
                    }
                }
            }
        }
        if (batchRows != 0) {
            ReduceLoadedKeyOwnedBatch(accumulator, gradients, batchKeys, batchRows, edgeWidth, ready, release);
        }
        WaitFlag<HardEvent::V_MTE2>(release);
        SetFlag<HardEvent::V_MTE3>(vectorMte3);
        WaitFlag<HardEvent::V_MTE3>(vectorMte3);
        const uint32_t outputRows = Min<uint32_t>(64, td->opInfo.S2 - bucket * 64);
        const uint64_t outputOffset = (static_cast<uint64_t>(batch) * td->opInfo.S2 + bucket * 64) * edgeWidth;
        const uint32_t firstRows = Min<uint32_t>(32, outputRows);
        if (firstRows > 0) {
            DataCopy(dkGm[outputOffset], accumulator, firstRows * edgeWidth);
        }
        if (outputRows > 32) {
            DataCopy(dkGm[outputOffset + 32 * edgeWidth], accumulator[32 * edgeWidth], (outputRows - 32) * edgeWidth);
        }
        SetFlag<HardEvent::MTE3_V>(mte3Vector);
        WaitFlag<HardEvent::MTE3_V>(mte3Vector);
        SetFlag<HardEvent::V_MTE2>(release);
    }
    WaitFlag<HardEvent::V_MTE2>(release);
}

} // namespace SFAG_KEY_OWNED_FULL_REDUCE
