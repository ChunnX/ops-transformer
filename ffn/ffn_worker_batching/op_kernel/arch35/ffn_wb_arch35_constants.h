/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Shared compile-time layout specifications. Host budgets and kernel queues
// must use the same values; these constants do not add fields to the tiling ABI.
#ifndef FFN_WB_ARCH35_CONSTANTS_H
#define FFN_WB_ARCH35_CONSTANTS_H

#include <cstdint>

namespace FfnWbArch35Config {
// Hardware/producer contracts, kept distinct even when their values coincide.
constexpr int64_t UB_BLOCK_BYTES = 32;
constexpr int64_t VECTOR_REPEAT_BYTES = 256;
constexpr int64_t VECTOR_REPEAT_BLOCKS = VECTOR_REPEAT_BYTES / UB_BLOCK_BYTES;
constexpr int64_t COMPARE_FP32_ELEMENTS = VECTOR_REPEAT_BYTES / sizeof(float);
constexpr int64_t TOKEN_STRIDE_ALIGNMENT = 512;
constexpr int64_t MX_SCALE_BLOCK_ELEMENTS = 32;
constexpr int64_t FP4_VALUES_PER_BYTE = 2;
constexpr int64_t SORT_PROPOSALS_PER_REPEAT = 32;
constexpr int64_t EXTRACT_MAX_REPEATS = 255; // Extract API repeatTime limit

// Public token_dtype attribute values.
constexpr int64_t TOKEN_FP16 = 0;
constexpr int64_t TOKEN_BF16 = 1;
constexpr int64_t TOKEN_INT8 = 2;
constexpr int64_t TOKEN_FP8_E5M2 = 3;
constexpr int64_t TOKEN_FP8_E4M3 = 4;
constexpr int64_t TOKEN_FP4_E2M1 = 5;

// Gather: two queues, each processing up to 128 output rows per iteration.
constexpr int64_t GATHER_QUEUE_DEPTH = 2;
constexpr int64_t GATHER_ROWS_PER_LOOP = 128;
constexpr int64_t GATHER_OUTPUT_LANES = 5;      // four indices and dynamic scale
constexpr int64_t GATHER_FLOAT_INDEX_LANES = 7; // src, qa, a, remainder, qb, b, k
constexpr int64_t GATHER_INT_INDEX_LANES = 3;   // a, b, k

constexpr int64_t GROUP_VECTOR_BUFFERS = 10; // cur/prev/next/idx/diff/diffF and four compacted arrays
constexpr int64_t GROUP_MASK_BUFFERS = 2;    // run starts and ends
constexpr int64_t GROUP_ROW_FIELDS = 2;      // expert ID and token count (int64)
constexpr int64_t GROUP_MIN_ROWS = 2;
constexpr int64_t BITS_PER_BYTE = 8;
constexpr int64_t RECV_WAITER_FLAG_BUFFERS = 2; // int32 flags and FP32 reduction scratch

// Extra InitBuffer blocks in addition to element storage.
constexpr int64_t SORT_BUFFER_COUNT = 5;            // input, concat, temporary, sorted, mask
constexpr int64_t MERGE_BUFFER_COUNT = 3;           // input, output, count cache
constexpr int64_t EXTRACT_BUFFER_COUNT = 3;         // proposals, expert IDs, source indices
constexpr int64_t MERGE_COUNT_CACHE_SEGMENTS = 128; // bounded cache policy, not a hardware limit
} // namespace FfnWbArch35Config
#endif // FFN_WB_ARCH35_CONSTANTS_H
