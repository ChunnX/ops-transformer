/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0
 */

#ifndef SERVICE_SPLIT_KN_ACCUM_H
#define SERVICE_SPLIT_KN_ACCUM_H

#include "mla_prolog_comm.h"
#include "kernel_operator.h"

namespace MlaProlog {

template<typename AccumType>
__aicore__ inline void AccumulateKnPartials(
    const LocalTensor<AccumType> &output,
    const GlobalTensor<AccumType> &partials,
    uint32_t mStart,
    uint32_t mLen,
    uint32_t nLen,
    uint32_t kGroups,
    uint64_t partialStride,
    const LocalTensor<AccumType> &temp)
{
    if (nLen == 0 || mLen == 0 || kGroups == 0) {
        return;
    }

    uint32_t chunkElems = Align(nLen, static_cast<uint32_t>(ALIGN_BLOCK_SIZE));
    // Each chunk needs chunkElems * sizeof(AccumType) bytes
    // // We need 2 chunks: accumulator and temp
    // LocalTensor<AccumType> temp = buffer.template Get<AccumType>(chunkElems);

    for (uint32_t m = mStart; m < mStart + mLen; m++) {
        // Step 1: Load first K-group's partial into accumulator
        // partials[k=0][m][nStart : nStart+nLen]
        uint64_t srcOffset = m * nLen;
        uint64_t dstOffset = (m - mStart) * nLen;
        DataCopy(output[dstOffset], partials[srcOffset], chunkElems);
        SetWitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        // Step 2: Sequentially add remaining K-groups (deterministic order)
        for (uint32_t k = 1; k < kGroups; k++) {
            uint64_t srcOffsetK = k * partialStride + srcOffset;
            DataCopy(temp, partials[srcOffsetK], chunkElems);
            SetWitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            // Vector add: accum += temp
            Add(output[dstOffset], output[dstOffset], temp, chunkElems);
            SetWitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

/**
 * Accumulate k_groups partial sums in sequential order for deterministic precision.
 * Each vector core handles a slice of the N dimension (columns).
 * If N < vec_cores, also splits M dimension.
 *
 * Memory layout of partials: [k_groups][M][N_total] in float32
 * Each K-group's slice is contiguous: partials[k * M * N_total]
 *
 * @tparam AccumType   Type of partial sums (typically float)
 * @tparam OutputType  Type of accumulated output (bf16 or int32)
 */
// template<typename AccumType, typename OutputType>
// __aicore__ inline void AccumulateKnPartials(
//     GlobalTensor<OutputType> &output,
//     const GlobalTensor<AccumType> &partials,
//     uint32_t M,
//     uint32_t N_total,
//     uint32_t nStart,
//     uint32_t nLen,
//     uint32_t mStart,
//     uint32_t mLen,
//     uint32_t kGroups,
//     uint32_t partialStride,
//     TBuf<TPosition::VECCALC> &buffer)
// {
//     if (nLen == 0 || mLen == 0 || kGroups == 0) {
//         return;
//     }

//     // UB buffer allocation: need 2 chunks for ping-pong (accum + temp)
//     constexpr uint32_t ALIGN = 32;  // 32-byte alignment
//     uint32_t chunkElems = (nLen + ALIGN - 1) / ALIGN * ALIGN;
//     // Each chunk needs chunkElems * sizeof(AccumType) bytes
//     // We need 2 chunks: accumulator and temp
//     LocalTensor<AccumType> accumBuf = buffer.template Get<AccumType>(chunkElems * 2);
//     LocalTensor<AccumType> accum = accumBuf[0];      // [0, chunkElems)
//     LocalTensor<AccumType> temp = accumBuf[chunkElems]; // [chunkElems, 2*chunkElems)

//     for (uint32_t m = mStart; m < mStart + mLen; m++) {
//         // Step 1: Load first K-group's partial into accumulator
//         // partials[k=0][m][nStart : nStart+nLen]
//         uint32_t srcOffset0 = m * N_total + nStart;
//         DataCopy(accum, partials[srcOffset0], chunkElems);
//         SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
//         WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

//         // Step 2: Sequentially add remaining K-groups (deterministic order)
//         for (uint32_t k = 1; k < kGroups; k++) {
//             uint32_t srcOffsetK = k * partialStride + m * N_total + nStart;
//             SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
//             WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
//             DataCopy(temp, partials[srcOffsetK], chunkElems);
//             SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
//             WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

//             // Vector add: accum += temp
//             Add(accum, accum, temp, chunkElems);
//             PipeBarrier<PIPE_V>();
//         }

//         // Step 3: Cast to output type if needed and store
//         if constexpr (std::is_same<AccumType, OutputType>::value) {
//             // Same type: direct store
//             SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
//             WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
//             uint32_t dstOffset = m * N_total + nStart;
//             DataCopy(output[dstOffset], accum, chunkElems);
//         } else {
//             // Need type cast (float -> bf16 or float -> int32)
//             // Reuse temp's memory for cast output (temp is no longer needed after accumulation)
//             LocalTensor<OutputType> castBuf = temp.template ReinterpretCast<OutputType>();
//             Cast(castBuf, accum, RoundMode::CAST_RINT, chunkElems);
//             SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
//             WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
//             uint32_t dstOffset = m * N_total + nStart;
//             DataCopy(output[dstOffset], castBuf, chunkElems);
//         }
//         SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
//         WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
//     }
// }

} // namespace MlaProlog

#endif // SERVICE_SPLIT_KN_ACCUM_H
