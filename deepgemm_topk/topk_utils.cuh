#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

namespace deep_gemm::topk {

// =============================================================================
// Warp-level batch TopK (Phase 2 of two-phase design)
//
// Operates on a fully-computed logits array (written by Phase 1).
// One warp processes one Q token's logits and selects top-kSparseCount indices.
//
// Algorithm:
//   1. Initialize SMEM top-K buffer with -inf
//   2. Fill initial kSparseCount logits and sort (single-thread bubble sort)
//   3. Scan remaining logits in chunks of 32:
//      a. Each lane loads one logit
//      b. __ballot_sync finds qualifying lanes (val > current min)
//      c. Serial insertion: each qualifying lane inserts via binary search + shift
//   4. Output sorted indices
//
// SMEM requirement: kSparseCount * (sizeof(float) + sizeof(int32_t)) bytes
// =============================================================================
template <uint32_t kSparseCount>
CUTLASS_DEVICE void warp_batch_topk(
    const float* __restrict__ logits,     // [k_len] logits for one Q token (computed in Phase 1)
    const uint32_t k_len,                 // number of valid logits
    const uint32_t kv_offset_base,        // base KV index (to convert local idx to global)
    int32_t* __restrict__ out_indices,    // [kSparseCount] output global KV indices
    float* __restrict__ smem_vals,        // [kSparseCount] SMEM scratch (sorted desc)
    int32_t* __restrict__ smem_idxs,      // [kSparseCount] SMEM scratch (corresponding indices)
    const uint32_t lane_idx               // threadIdx.x % 32
) {
    // -------------------------------------------------------------------------
    // Step 1: Initialize SMEM top-K buffer
    // -------------------------------------------------------------------------
    #pragma unroll
    for (uint32_t i = lane_idx; i < kSparseCount; i += 32) {
        smem_vals[i] = -CUDART_INF_F;
        smem_idxs[i] = -1;
    }
    __syncwarp();

    // -------------------------------------------------------------------------
    // Step 2: Fill initial top-K with first min(kSparseCount, k_len) logits
    //         and sort them in descending order (bubble sort by lane 0)
    // -------------------------------------------------------------------------
    const uint32_t init_count = cute::min(kSparseCount, k_len);

    #pragma unroll
    for (uint32_t i = lane_idx; i < init_count; i += 32) {
        smem_vals[i] = logits[i];
        smem_idxs[i] = static_cast<int32_t>(kv_offset_base + i);
    }
    __syncwarp();

    // Sort initial fill (simple bubble sort, single thread)
    // For kSparseCount <= 2048, this is O(k^2) but only runs once
    if (lane_idx == 0) {
        #pragma unroll 1
        for (uint32_t i = 0; i < init_count; ++i) {
            #pragma unroll 1
            for (uint32_t j = i + 1; j < init_count; ++j) {
                if (smem_vals[i] < smem_vals[j]) {
                    const float tv = smem_vals[i];
                    smem_vals[i] = smem_vals[j];
                    smem_vals[j] = tv;
                    const int32_t ti = smem_idxs[i];
                    smem_idxs[i] = smem_idxs[j];
                    smem_idxs[j] = ti;
                }
            }
        }
    }
    __syncwarp();

    // -------------------------------------------------------------------------
    // Step 3: Scan remaining logits in chunks of 32
    //         Each lane loads one logit; qualifying lanes insert serially
    // -------------------------------------------------------------------------
    #pragma unroll 1
    for (uint32_t chunk_start = init_count; chunk_start < k_len; chunk_start += 32) {
        const uint32_t idx = chunk_start + lane_idx;
        const float val = (idx < k_len) ? logits[idx] : -CUDART_INF_F;
        const uint32_t global_kv_idx = kv_offset_base + idx;

        // Check if this value qualifies (beats current minimum in top-K)
        const float current_min = smem_vals[kSparseCount - 1];
        const bool qualifies = val > current_min;
        uint32_t mask = __ballot_sync(0xFFFFFFFF, qualifies);

        // Process qualifying lanes one at a time (serial insertion)
        // Each qualifying lane does binary search + shift insertion
        while (mask != 0) {
            const uint32_t winner_lane = __ffs(mask) - 1;

            if (lane_idx == winner_lane) {
                // Binary search for insertion position in sorted (desc) array
                uint32_t lo = 0, hi = kSparseCount;
                while (lo < hi) {
                    const uint32_t mid = (lo + hi) >> 1;
                    if (smem_vals[mid] > val) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                // lo is the insertion position
                // Shift elements [lo, kSparseCount-2] down by 1 to make room
                #pragma unroll 1
                for (uint32_t j = kSparseCount - 1; j > lo; --j) {
                    smem_vals[j] = smem_vals[j - 1];
                    smem_idxs[j] = smem_idxs[j - 1];
                }
                smem_vals[lo] = val;
                smem_idxs[lo] = static_cast<int32_t>(global_kv_idx);
            }
            __syncwarp();

            // Clear this lane's bit and move to next qualifier
            mask &= ~(1u << winner_lane);
        }
    }

    // -------------------------------------------------------------------------
    // Step 4: Output sorted indices (descending by logit value)
    // -------------------------------------------------------------------------
    #pragma unroll
    for (uint32_t i = lane_idx; i < kSparseCount; i += 32) {
        out_indices[i] = smem_idxs[i];
    }
    __syncwarp();
}

// =============================================================================
// Warp-level batch TopK for compressed logits (variable-length per token)
//
// Handles the case where each Q token has a different KV range [k_start, k_end).
// Only processes logits within the valid range, outputs indices relative to
// the KV cache base (not relative to k_start).
// =============================================================================
template <uint32_t kSparseCount>
CUTLASS_DEVICE void warp_batch_topk_compressed(
    const float* __restrict__ logits,        // [stride] logits row for one Q token
    const uint32_t stride,                    // row stride in logits buffer
    const uint32_t k_start,                   // valid KV start index
    const uint32_t k_end,                     // valid KV end index
    int32_t* __restrict__ out_indices,        // [kSparseCount] output
    float* __restrict__ smem_vals,            // [kSparseCount] SMEM scratch
    int32_t* __restrict__ smem_idxs,          // [kSparseCount] SMEM scratch
    const uint32_t lane_idx
) {
    const uint32_t k_len = k_end > k_start ? (k_end - k_start) : 0;

    // Edge case: no valid KV tokens
    if (k_len == 0) {
        if (lane_idx < kSparseCount) {
            out_indices[lane_idx] = -1;
        }
        __syncwarp();
        return;
    }

    // Edge case: fewer valid tokens than kSparseCount — output all, pad with -1
    if (k_len <= kSparseCount) {
        for (uint32_t i = lane_idx; i < kSparseCount; i += 32) {
            if (i < k_len) {
                out_indices[i] = static_cast<int32_t>(k_start + i);
            } else {
                out_indices[i] = -1;
            }
        }
        __syncwarp();
        return;
    }

    // Normal case: use batch TopK with the logits sub-range
    // logits pointer adjusted to k_start
    warp_batch_topk<kSparseCount>(
        logits + k_start,           // start of valid logits
        k_len,                       // number of valid logits
        k_start,                     // base KV index (output absolute indices)
        out_indices,
        smem_vals,
        smem_idxs,
        lane_idx
    );
}

} // namespace deep_gemm::topk
