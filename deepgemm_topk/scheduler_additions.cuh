// =============================================================================
// scheduler_additions.cuh
//
// Additions to SM100MQALogitsScheduler and SM100PagedMQALogitsScheduler
// to support Phase 2 (TopK) querying per-token KV ranges.
//
// These methods are needed by topk_phase() in sm100_mqa_logits_with_topk.cuh
// to get the KV range [k_start, k_end) for each Q token.
// =============================================================================

// =====================================================================
// Add to: deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits.cuh
// File: SM100MQALogitsScheduler
// =====================================================================

// --- Add these methods to the struct ---

// Get total number of Q tokens (for Phase 2 round-robin assignment)
CUTLASS_DEVICE uint32_t get_num_q_tokens() const {
    return num_q_tokens;
}

// Get KV start index for a specific Q token (for Phase 2 TopK)
CUTLASS_DEVICE uint32_t get_k_start_for_token(const uint32_t& q_token) const {
    const auto row_idx = cute::min(q_token, num_q_tokens - 1);
    return cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
}

// Get KV end index for a specific Q token (for Phase 2 TopK)
CUTLASS_DEVICE uint32_t get_k_end_for_token(const uint32_t& q_token) const {
    const auto row_idx = cute::min(q_token, num_q_tokens - 1);
    return cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
}


// =====================================================================
// Add to: deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh
// File: SM100PagedMQALogitsScheduler
// =====================================================================

// --- Add these methods to the struct ---

// Get total number of Q tokens
CUTLASS_DEVICE uint32_t get_num_q_tokens() const {
    return num_q_tokens_total;
}

// Get KV start index for a specific Q token (always 0 for paged mode)
CUTLASS_DEVICE uint32_t get_k_start_for_token(const uint32_t& q_token) const {
    return 0;
}

// Get KV end index for a specific Q token
// For paged mode, this is the context length of the corresponding request
CUTLASS_DEVICE uint32_t get_k_end_for_token(const uint32_t& q_token) const {
    // Determine which request this Q token belongs to
    if constexpr (kIsVarlen) {
        // Varlen: all tokens in a request share the same context_len
        // Find the request by scanning indices (or use Info)
        const auto info = Info::from_q_token(q_token, num_q_tokens_total, context_lens, get_indices());
        // context_len is encoded in num_kv_splits: context_len = num_kv_splits * SPLIT_KV (upper bound)
        // But for exact value, read context_lens directly
        const uint32_t lens_idx = kIsContextLens2D ? info.q_token_start + info.num_q_tokens - 1 : info.get_block_table_row();
        return context_lens[lens_idx];
    } else {
        // Regular: request_id = q_token / kNextN
        const uint32_t request_id = q_token / kNextN;
        const uint32_t lens_idx = kIsContextLens2D ? request_id * kNextN + kNextN - 1 : request_id;
        return context_lens[lens_idx];
    }
}
