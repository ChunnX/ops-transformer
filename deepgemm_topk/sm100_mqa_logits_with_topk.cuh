#pragma once

// =============================================================================
// sm100_mqa_logits_with_topk.cuh
//
// Based EXACTLY on sm100_mqa_logits.cuh from DeepGEMM.
// Only additions:
//   1. kSparseCount template parameter
//   2. logits_buffer (float*) replaces logits output (Phase 1 writes here)
//   3. sparse_indices (int32*) output (Phase 2 TopK writes here)
//   4. TopK code appended at end of kernel (pure CUDA, no new includes)
// =============================================================================

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/mqa_logits.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

// For CUDART_INF_F used in TopK
#include <math_constants.h>

namespace deep_gemm {

// Ring-buffer counter avoiding `% kNumStages`, which ptxas can lower poorly for TMEM paths
template <uint32_t kNumStages>
struct RingPipeline {
    uint32_t stage_idx = 0, phase = 0;

    CUTLASS_DEVICE cute::tuple<uint32_t, uint32_t> advance(const uint32_t& step = 1) {
        const uint32_t current_stage_idx = stage_idx, current_phase = phase;
        stage_idx += step;
        if (stage_idx >= kNumStages) {
            stage_idx -= kNumStages;
            phase ^= 1;
        }
        return {current_stage_idx, current_phase};
    }
};

// Convert runtime valid-token count to `cute::Int` so token loops stay compile-time constant
template <uint32_t kBlockQ, uint32_t kCandidate = kBlockQ, typename Fn>
CUTLASS_DEVICE void dispatch_num_block_tokens(const uint32_t& num_block_tokens, Fn&& fn) {
    if constexpr (kCandidate <= 1) {
        fn(cute::Int<1>{});
    } else if (num_block_tokens >= kCandidate) {
        fn(cute::Int<kCandidate>{});
    } else {
        dispatch_num_block_tokens<kBlockQ, kCandidate - 1>(num_block_tokens, static_cast<Fn&&>(fn));
    }
}

// =============================================================================
// Warp-level 8-bit radix-select TopK
// 4 rounds: histogram on each byte → find cutoff → narrow down
// Final: collect definite (value > kth) + boundary (value == kth)
// Fully parallel, no serial insertion, no bubble sort.
// =============================================================================
template <uint32_t kSparseCount>
CUTLASS_DEVICE void warp_bucket_topk(
    const float* __restrict__ logits,
    const uint32_t k_start,
    const uint32_t k_end,
    int32_t* __restrict__ out_indices,
    float* __restrict__ smem_vals,
    int32_t* __restrict__ smem_idxs,
    const uint32_t lane_idx
) {
    const uint32_t k_len = k_end > k_start ? (k_end - k_start) : 0;

    if (k_len == 0) {
        if (lane_idx < kSparseCount) out_indices[lane_idx] = -1;
        __syncwarp();
        return;
    }

    if (k_len <= kSparseCount) {
        for (uint32_t i = lane_idx; i < kSparseCount; i += 32)
            out_indices[i] = (i < k_len) ? static_cast<int32_t>(k_start + i) : -1;
        __syncwarp();
        return;
    }

    auto to_sortable = [](float f) -> uint32_t {
        uint32_t u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
    };

    // SMEM: 256-bucket histogram + candidate index buffer
    uint32_t* histogram = reinterpret_cast<uint32_t*>(smem_vals);

    uint32_t kth_value = 0;
    uint32_t definite_count = 0;
    uint32_t remaining = kSparseCount;

    // === 4 rounds of 8-bit radix select ===
    for (int round = 0; round < 4 && remaining > 0; ++round) {
        const int shift = (3 - round) * 8;
        const uint32_t prefix_mask = (round == 0) ? 0u : (0xFFFFFFFFu << (32 - round * 8));
        const uint32_t prefix_val = kth_value & prefix_mask;

        // Clear histogram via SMEM
        for (uint32_t i = lane_idx; i < 256; i += 32)
            histogram[i] = 0;
        __syncwarp();

        // Build histogram with SMEM atomicAdd
        for (uint32_t i = lane_idx; i < k_len; i += 32) {
            uint32_t u = to_sortable(logits[k_start + i]);
            if ((u & prefix_mask) == prefix_val) {
                uint32_t bucket = (u >> shift) & 0xFF;
                atomicAdd(&histogram[bucket], 1u);
            }
        }
        __syncwarp();

        // Find cutoff: each lane reads 8 buckets [lane*8 .. lane*8+7]
        uint32_t my_counts[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            my_counts[i] = histogram[lane_idx * 8 + i];

        uint32_t my_sum = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) my_sum += my_counts[i];

        // Inclusive prefix sum (lane 0 = lowest buckets, lane 31 = highest)
        uint32_t incl = my_sum;
        for (int off = 1; off < 32; off <<= 1) {
            uint32_t t = __shfl_up_sync(0xFFFFFFFF, incl, off);
            if (lane_idx >= off) incl += t;
        }
        const uint32_t total = __shfl_sync(0xFFFFFFFF, incl, 31);

        const uint32_t count_ge_me = incl;
        const uint32_t count_above_me = total - count_ge_me;

        uint32_t prefix_in_lane = 0;
        int32_t my_cutoff = -1;
        uint32_t my_count_above_cutoff = 0;

        #pragma unroll
        for (int i = 7; i >= 0; i--) {
            uint32_t count_above = count_above_me + prefix_in_lane;
            uint32_t count_ge = count_above + my_counts[i];
            if (count_above < remaining && count_ge >= remaining) {
                my_cutoff = static_cast<int32_t>(lane_idx * 8 + i);
                my_count_above_cutoff = count_above;
            }
            prefix_in_lane += my_counts[i];
        }

        uint32_t cutoff_mask = __ballot_sync(0xFFFFFFFF, my_cutoff >= 0);
        if (cutoff_mask == 0) break;
        uint32_t cutoff_lane = __ffs(cutoff_mask) - 1;
        uint32_t cutoff_bucket = static_cast<uint32_t>(
            __shfl_sync(0xFFFFFFFF, static_cast<uint32_t>(my_cutoff), cutoff_lane));
        uint32_t count_above_cutoff = __shfl_sync(0xFFFFFFFF, my_count_above_cutoff, cutoff_lane);

        kth_value |= (cutoff_bucket << shift);
        definite_count += count_above_cutoff;
        remaining = kSparseCount - definite_count;
    }

    // === Final pass: write definite (value > kth) + boundary (value == kth) ===
    const uint32_t kth_full = kth_value;

    // Count definite
    uint32_t my_def = 0;
    for (uint32_t i = lane_idx; i < k_len; i += 32) {
        uint32_t u = to_sortable(logits[k_start + i]);
        if (u > kth_full) my_def++;
    }

    // Prefix sum for output offset
    uint32_t def_prefix = my_def;
    for (int offset = 1; offset < 32; offset <<= 1) {
        uint32_t t = __shfl_up_sync(0xFFFFFFFF, def_prefix, offset);
        if (lane_idx >= offset) def_prefix += t;
    }
    const uint32_t def_exclusive = def_prefix - my_def;
    const uint32_t total_definite = __shfl_sync(0xFFFFFFFF, def_prefix, 31);

    // Write definite
    uint32_t def_pos = def_exclusive;
    for (uint32_t i = lane_idx; i < k_len; i += 32) {
        uint32_t u = to_sortable(logits[k_start + i]);
        if (u > kth_full) {
            out_indices[def_pos] = static_cast<int32_t>(k_start + i);
            def_pos++;
        }
    }
    __syncwarp();

    // Collect boundary (value == kth)
    constexpr uint32_t kPerLane = (kSparseCount + 31) / 32;
    uint32_t my_collected = 0;
    for (uint32_t i = lane_idx; i < k_len && my_collected < kPerLane; i += 32) {
        uint32_t u = to_sortable(logits[k_start + i]);
        if (u == kth_full) {
            smem_idxs[lane_idx * kPerLane + my_collected] = static_cast<int32_t>(k_start + i);
            my_collected++;
        }
    }

    uint32_t col_prefix = my_collected;
    for (int offset = 1; offset < 32; offset <<= 1) {
        uint32_t t = __shfl_up_sync(0xFFFFFFFF, col_prefix, offset);
        if (lane_idx >= offset) col_prefix += t;
    }
    const uint32_t col_exclusive = col_prefix - my_collected;
    const uint32_t total_collected = __shfl_sync(0xFFFFFFFF, col_prefix, 31);

    const uint32_t remaining_slots = kSparseCount - total_definite;
    for (uint32_t i = 0; i < my_collected; ++i) {
        uint32_t out_pos = total_definite + col_exclusive + i;
        if (out_pos < kSparseCount)
            out_indices[out_pos] = smem_idxs[lane_idx * kPerLane + i];
    }

    if (lane_idx == 0) {
        const uint32_t filled = total_definite +
            (total_collected < remaining_slots ? total_collected : remaining_slots);
        for (uint32_t i = filled; i < kSparseCount; ++i)
            out_indices[i] = -1;
    }
    __syncwarp();
}

// =============================================================================
// Core implementation (modified from sm100_mqa_logits_core_impl)
// Changes: kSparseCount param, logits_buffer + sparse_indices outputs, TopK at end
// =============================================================================
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF, bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kSparseCount,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t, typename MakeScheduler,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_DEVICE void sm100_mqa_logits_with_topk_core_impl(
    const uint32_t logits_stride,
    float* __restrict__ logits_buffer,                 // Phase 1 output (temp)
    int32_t* __restrict__ sparse_indices,              // Phase 2 output (final)
    const cute::TmaDescriptor& tensor_map_q,
    const cute::TmaDescriptor& tensor_map_sf_q,
    const cute::TmaDescriptor& tensor_map_kv,
    const cute::TmaDescriptor& tensor_map_sf_kv,
    const cute::TmaDescriptor& tensor_map_weights,
    const MakeScheduler& make_scheduler
) {
    constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;

    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    static constexpr uint32_t kNumTmemStages = 3;
    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;
    static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    static constexpr uint32_t kNumSFQ  = kIsMXSF ? math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kNumSFKV = kIsMXSF ? math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kRealNumSFQ = BLOCK_Q * kNumHeads;
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = kIsMXSF ? (kRealNumSFQ * sizeof(int)) : 0;
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kIsMXSF ? (kNumSFKV * sizeof(int)) : (SPLIT_KV * sizeof(float));
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(reduce_dtype_t);

    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");

    using SharedStorage = layout::MQALogitsSharedStorage<kNumHeads, kHeadDim, kIsMXSF, BLOCK_Q, SPLIT_KV,
                                                         kNumQStages, kNumKVStages, kNumTmemStages, qk_dtype_t, reduce_dtype_t>;
    extern __shared__ __align__(SharedStorage::kSwizzleAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    constexpr uint32_t kNumAccumTmemCols = BLOCK_Q * kNumHeads * kNumTmemStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 + kNumSFKV / 32>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            smem.full_q_barriers[i].init(1);
            smem.empty_q_barriers[i].init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            smem.full_kv_barriers[i].init(1);
            smem.empty_kv_barriers[i].init(kIsMXSF ? 1 : kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++i) {
            smem.full_tmem_barriers[i].init(1);
            smem.empty_tmem_barriers[i].init(128);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();

    if (warp_idx == kSpecWarpStart + 2)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
    __syncthreads();

    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];

    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;

    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumMathRegisters = 224;

    cudaGridDependencySynchronize();

    // ====== Phase 1: TMA Q Load Warp ======
    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (cute::elect_one_sync()) {
            auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

                const uint32_t q_token_base = scheduler.get_q_tma_token_base(q_block_idx);
                tma::copy<kNumQKBytesPerToken, BLOCK_Q * kNumHeads, 0>(
                    &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                    smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
                if constexpr (kIsMXSF)
                    tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx], smem.smem_sf_q[q_stage_idx], 0, q_token_base);
                tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, &smem.full_q_barriers[q_stage_idx], smem.smem_weights[q_stage_idx], 0, q_token_base);
                smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        }
        __syncwarp();

    // ====== Phase 1: TMA KV Load Warp ======
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        uint32_t cached_kv_page_base = 0;
        uint32_t cached_kv_page_coord = 0;
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            cached_kv_page_base = cute::numeric_limits<uint32_t>::max();
            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                if constexpr (decltype(scheduler)::kIsPaged) {
                    constexpr uint32_t kPageKV = decltype(scheduler)::kPageKV;
                    constexpr uint32_t kNumPagesPerSplit = decltype(scheduler)::kNumPagesPerSplit;
                    DG_STATIC_ASSERT(kNumPagesPerSplit <= 32, "Split spans more pages than a warp can cache");

                    const uint32_t kv_page_base = (kv_base + kv_split_idx) * kNumPagesPerSplit;
                    if (kv_page_base < cached_kv_page_base or kv_page_base + kNumPagesPerSplit > cached_kv_page_base + 32) {
                        cached_kv_page_base = (kv_page_base / 32) * 32;
                        cached_kv_page_coord = scheduler.get_kv_page_coord_by_page_offset(cached_kv_page_base + lane_idx);
                    }

                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    if (cute::elect_one_sync())
                        smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
                    __syncwarp();

                    int page_coords[kNumPagesPerSplit];
                    #pragma unroll
                    for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx) {
                        const auto src_lane = static_cast<int>(kv_page_base - cached_kv_page_base + page_idx);
                        page_coords[page_idx] = __shfl_sync(0xffffffff, cached_kv_page_coord, src_lane);
                    }

                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx) {
                            tma::copy<kNumQKBytesPerToken, kPageKV, 0, qk_dtype_t, true>(
                                &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                                smem.smem_kv[kv_stage_idx] + page_idx * kPageKV * kNumQKBytesPerToken,
                                0, 0, 1, page_coords[page_idx]);
                            tma::copy<kPageKV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                                     smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                                                     0, page_coords[page_idx]);
                        }
                        smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                    }
                    __syncwarp();
                } else if (cute::elect_one_sync()) {
                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);

                    const uint32_t kv_tma_offset = scheduler.get_kv_tma_offset(kv_base, kv_split_idx);
                    tma::copy<kNumQKBytesPerToken, SPLIT_KV, 0>(
                        &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                        smem.smem_kv[kv_stage_idx], 0, kv_tma_offset);
                    tma::copy<SPLIT_KV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                              smem.smem_sf_kv[kv_stage_idx], kv_tma_offset, 0);
                    smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        }

    // ====== Phase 1: MMA Issue Warp ======
    } else if (warp_idx == kSpecWarpStart + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);

        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            if constexpr (kIsMXSF) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                }
                cutlass::arch::fence_view_async_shared();
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                    mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                    if (cute::elect_one_sync())
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
                    __syncwarp();
                }
            }

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

                if constexpr (kIsMXSF) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                        utccp_required_smem_warp_transpose(smem_ptr);
                    }
                    cutlass::arch::fence_view_async_shared();
                }

                if (cute::elect_one_sync()) {
                    if constexpr (kIsMXSF) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
                        }
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                        CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

                        smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

                        if constexpr (kIsMXSF) {
                            DG_STATIC_ASSERT((not kIsFP4 and kHeadDim == 32) or kHeadDim == 64 or kHeadDim == 128, "Invalid head dim");

                            constexpr uint32_t kPackFactor = kIsFP4 ? 2 : 1;
                            constexpr uint32_t kQKSwizzleMode = kHeadDim / kPackFactor;

                            using mma_op_t = cute::conditional_t<kIsFP4, ptx::SM100_MMA_MXF4_SS, ptx::SM100_MMA_MXF8F6F4_SS>;
                            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<qk_dtype_t, qk_dtype_t, float, cutlass::float_ue8m0_t,
                                                                                       UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                            #pragma unroll
                            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                                auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * kPackFactor, k * kPackFactor);
                                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(
                                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(
                                    smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                                mma_op_t::fma(
                                    a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                                    kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
                            }
                        } else {
                            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                                            UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                            auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                            #pragma unroll
                            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                    smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                                ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, k, runtime_instr_desc);
                            }
                        }

                        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                     ::"r"(cute::cast_smem_ptr_to_uint(&smem.full_tmem_barriers[tmem_stage_idx])));
                    }
                }
                __syncwarp();
                if constexpr (kIsMXSF)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_kv_barriers[kv_stage_idx]));
            }
            smem.empty_q_barriers[q_stage_idx].arrive();
        }

    // ====== Phase 1: Idle Warp ======
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    // ====== Phase 1: Math Warpgroups ======
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_warpgroup_idx = warpgroup_idx;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;
        DG_STATIC_ASSERT(kNumMathWarpGroups <= kNumTmemStages, "Math warp groups exceed TMEM stages");
        tmem_pipeline.advance(math_warpgroup_idx);

        constexpr bool kIsReduceBF16 = not cute::is_same_v<reduce_dtype_t, float>;
        DG_STATIC_ASSERT(not kIsReduceBF16 or kNumHeads % 2 == 0, "bf16 weights need even kNumHeads");
        DG_STATIC_ASSERT(kNumHeads == 4 or kNumHeads == 8 or kNumHeads == 16 or kNumHeads == 32 or kNumHeads == 64,
                         "Unsupported TMEM load size");
        using weights_reg_dtype_t = cute::conditional_t<kIsReduceBF16, nv_bfloat162, float>;
        constexpr uint32_t kNumWeightsRegPerToken = kIsReduceBF16 ? (kNumHeads / 2) : kNumHeads;
        weights_reg_dtype_t weights[BLOCK_Q][kNumWeightsRegPerToken];
        float accum[kNumHeads];

        auto tmem_load_no_fence = [](auto num_elems_t, const uint32_t& addr, float* load_dst) {
            constexpr uint32_t N = decltype(num_elems_t)::value;
            using Loader = cute::conditional_t<N == 2,  cute::SM100_TMEM_LOAD_32dp32b2x,
                           cute::conditional_t<N == 4,  cute::SM100_TMEM_LOAD_32dp32b4x,
                           cute::conditional_t<N == 8,  cute::SM100_TMEM_LOAD_32dp32b8x,
                           cute::conditional_t<N == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                           cute::conditional_t<N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                                                        cute::SM100_TMEM_LOAD_32dp32b64x>>>>>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(addr, reinterpret_cast<uint32_t*>(load_dst)[Is]...);
            }(cute::make_index_sequence<N>{});
        };

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            const auto process_q_block = [&](auto num_valid_tokens_t) {
                constexpr uint32_t kNumValidTokens = decltype(num_valid_tokens_t)::value;

                #pragma unroll
                for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                    const auto smem_weights_row = smem.smem_weights[q_stage_idx] + i * kNumHeads;
                    if constexpr (kIsReduceBF16) {
                        const auto packed_row = reinterpret_cast<const uint32_t*>(smem_weights_row);
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads / 2; ++ j) {
                            const auto packed = ptx::ld_shared(packed_row + j);
                            weights[i][j] = ptx::exchange(*reinterpret_cast<const nv_bfloat162*>(&packed), 0);
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads; ++ j)
                            weights[i][j] = ptx::ld_shared(smem_weights_row + j);
                    }
                }

                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                    auto kv_offset = scheduler.get_logits_col(kv_base, kv_split_idx, math_thread_idx);

                    reduce_dtype_t scale_kv = static_cast<reduce_dtype_t>(0.0f);
                    uint32_t kv_stage_idx = 0;
                    if constexpr (not kIsMXSF) {
                        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx_local, kv_phase);
                        kv_stage_idx = kv_stage_idx_local;
                        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
                        scale_kv = static_cast<reduce_dtype_t>(ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + math_thread_idx));
                    }

                    CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                    smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                    ptx::tcgen05_after_thread_sync();

                    if constexpr (not kIsMXSF)
                        smem.empty_kv_barriers[kv_stage_idx].arrive();

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                        if constexpr (kNumHeads == 8) {
                            tmem_load_no_fence(cute::Int<kNumHeads>{}, tmem_addr, accum);
                            cutlass::arch::fence_view_async_tmem_load();
                        } else if constexpr (kNumHeads == 16) {
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr, accum);
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2, accum + kNumHeads / 2);
                            cutlass::arch::fence_view_async_tmem_load();
                        } else {
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr, accum);
                            cutlass::arch::fence_view_async_tmem_load();
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2, accum + kNumHeads / 2);
                            cutlass::arch::fence_view_async_tmem_load();
                        }

                        if (i == kNumValidTokens - 1) {
                            ptx::tcgen05_before_thread_sync();
                            smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                        }

                        reduce_dtype_t reduced;
                        if constexpr (kIsReduceBF16) {
                            auto sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                            auto sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                            const auto transform = [&](const uint32_t& j, const nv_bfloat162& sum) {
                                const auto a = ptx::cvt_relu_bf16x2_f32(make_float2(accum[j], accum[j + 1]));
                                const auto b = weights[i][j / 2];
                                return __hfma2(a, b, sum);
                            };

                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads; j += 4) {
                                sum_0 = transform(j, sum_0);
                                sum_1 = transform(j + 2, sum_1);
                            }

                            auto sum = __hadd2_rn(sum_0, sum_1);
                            reduced = __hadd_rn(sum.x, sum.y);
                        } else {
                            auto sum_0 = make_float2(0, 0);
                            auto sum_1 = make_float2(0, 0);
                            const auto transform = [&](const uint32_t& j, const float2& sum) {
                                auto a_0 = make_float2(accum[j], accum[j + 1]);
                                auto a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));
                                auto b = make_float2(weights[i][j], weights[i][j + 1]);
                                return __ffma2_rn(__fadd2_rn(a_0, a_1), b, sum);
                            };

                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads; j += 4) {
                                sum_0 = transform(j, sum_0);
                                sum_1 = transform(j + 2, sum_1);
                            }

                            auto sum = __fadd2_rn(sum_0, sum_1);
                            reduced = (sum.x + sum.y) / 2;
                        }

                        // Phase 1 store: write to logits_buffer (float)
                        auto result = static_cast<float>(
                            kIsMXSF ? reduced : reduced * scale_kv);

                        const auto q_offset = scheduler.get_logits_row(q_block_idx, i) * static_cast<uint64_t>(logits_stride);
                        if constexpr (kIsCompressedLogits) {
                            const uint32_t rel_kv = kv_offset - seq_k_start[i];
                            const uint32_t len = seq_k_end[i] - seq_k_start[i];
                            if (rel_kv < len)
                                logits_buffer[q_offset + rel_kv] = result;
                        } else {
                            if (kv_offset < logits_stride)
                                logits_buffer[q_offset + kv_offset] = result;
                        }
                    }
                }
            };

            if constexpr (decltype(scheduler)::kHasPartialBlock)
                dispatch_num_block_tokens<BLOCK_Q>(scheduler.get_num_block_tokens(q_block_idx), process_q_block);
            else
                process_q_block(cute::Int<BLOCK_Q>{});

            smem.empty_q_barriers[q_stage_idx].arrive();
        }

        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }

    // ========================================================================
    // PHASE 1 → PHASE 2 SYNC
    // ========================================================================
    __syncthreads();

    // ========================================================================
    // PHASE 2: TopK selection
    // All warps (math + specialized) participate
    // ========================================================================
    {
        const uint32_t total_warps = kSpecWarpStart + 4;  // math warps + 4 specialized warps
        const uint32_t topk_scratch_per_warp = kSparseCount * 8;
        struct TopKScratch {
            float vals[kSparseCount];
            int32_t idxs[kSparseCount];
        };
        TopKScratch* scratch = reinterpret_cast<TopKScratch*>(smem_buffer + warp_idx * topk_scratch_per_warp);
        float* topk_vals = scratch->vals;
        int32_t* topk_idxs = scratch->idxs;

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        const uint32_t num_q_tokens = [&] {
            if constexpr (decltype(scheduler)::kIsPaged) return scheduler.num_q_tokens_total;
            else return scheduler.num_q_tokens;
        }();

        const uint32_t num_sms = gridDim.x;
        for (uint32_t q_token = sm_idx * total_warps + warp_idx;
             q_token < num_q_tokens;
             q_token += num_sms * total_warps) {
            uint32_t k_start, k_end;

            if constexpr (decltype(scheduler)::kIsPaged) {
                k_start = 0;
                k_end = scheduler.context_lens[q_token];
            } else {
                k_start = cute::min(scheduler.cu_seq_len_k_start[q_token], scheduler.num_kv_tokens);
                k_end = cute::min(scheduler.cu_seq_len_k_end[q_token], scheduler.num_kv_tokens);
            }

            const float* logits_row = logits_buffer + static_cast<uint64_t>(q_token) * logits_stride;
            int32_t* out_ptr = sparse_indices + static_cast<uint64_t>(q_token) * kSparseCount;

            if (k_end > k_start) {
                warp_bucket_topk<kSparseCount>(
                    logits_row, k_start, k_end,
                    out_ptr, topk_vals, topk_idxs, lane_idx);
            } else {
                if (lane_idx < kSparseCount)
                    out_ptr[lane_idx] = -1;
                __syncwarp();
            }
        }
    }

    __syncthreads();
}

// =============================================================================
// Entry point: Non-paged (contiguous KV) with TopK
// =============================================================================
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF, bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kSparseCount,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_mqa_logits_with_topk(
    const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
    const uint32_t logits_stride,
    const uint32_t* cu_seq_len_k_start,
    const uint32_t* cu_seq_len_k_end,
    float* __restrict__ logits_buffer,
    int32_t* __restrict__ sparse_indices,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights
) {
    const auto make_scheduler = [&](const uint32_t& sm_idx, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        return sched::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs>(
            sm_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, seq_k_start, seq_k_end);
    };

    sm100_mqa_logits_with_topk_core_impl<
        kNumHeads, kHeadDim, kIsMXSF, kIsCompressedLogits, BLOCK_Q, SPLIT_KV,
        kNumQStages, kNumKVStages, kNumSMs,
        kNumSpecializedThreads, kNumMathThreads,
        kSparseCount,
        qk_dtype_t, logits_dtype_t, reduce_dtype_t,
        decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits_buffer, sparse_indices,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

// =============================================================================
// Entry point: Paged KV with TopK
// =============================================================================
template <uint32_t kTokensPerRequest, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t PAGE_KV,
          bool kIsMXSF, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV, uint32_t kSplitsPerChunk,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kSparseCount,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_paged_mqa_logits_with_topk(
    const uint32_t num_q_tokens_total,
    const uint32_t logits_stride, const uint32_t block_table_stride,
    const uint32_t* context_lens, float* __restrict__ logits_buffer,
    int32_t* __restrict__ sparse_indices,
    const uint32_t* block_table, const uint32_t* indices,
    const uint32_t* schedule_meta,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights
) {
    static constexpr uint32_t BLOCK_Q = 128 / kNumHeads;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    DG_STATIC_ASSERT(SPLIT_KV == PAGE_KV * kNumPagesPerSplit, "Invalid split/page size");

    const auto make_scheduler = [&](const uint32_t& sm_idx, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        return sched::SM100PagedMQALogitsScheduler<kTokensPerRequest, kIsContextLens2D, kIsVarlen,
                                                   kNumHeads, SPLIT_KV, PAGE_KV, kSplitsPerChunk>(
            sm_idx, context_lens, schedule_meta, indices,
            block_table, block_table_stride, num_q_tokens_total);
    };

    sm100_mqa_logits_with_topk_core_impl<
        kNumHeads, kHeadDim, kIsMXSF, false, BLOCK_Q, SPLIT_KV,
        kNumQStages, kNumKVStages, 0,
        kNumSpecializedThreads, kNumMathThreads,
        kSparseCount,
        qk_dtype_t, logits_dtype_t, reduce_dtype_t,
        decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits_buffer, sparse_indices,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

} // namespace deep_gemm

#pragma clang diagnostic pop
