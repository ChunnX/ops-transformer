// =============================================================================
// attention_with_topk.hpp
//
// Host-side API for MQA logits + TopK on SM100 (B200).
// Two independent kernel launches:
//   Phase 1: Original sm100_mqa_logits → logits_buffer (float32)
//   Phase 2: Independent TopK kernel → sparse_indices (int32)
// =============================================================================

#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_mqa_logits_with_topk.hpp"
#include "../jit_kernels/impls/sm100_mqa_logits.hpp"
#endif

#include "layout.hpp"

namespace deep_gemm::attention {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

// =============================================================================
// Non-paged: fp8_fp4_mqa_logits_with_topk
// =============================================================================
static torch::Tensor fp8_fp4_mqa_logits_with_topk(
    const std::tuple<torch::Tensor, std::optional<torch::Tensor>>& q,
    const std::tuple<torch::Tensor, torch::Tensor>& kv,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    const int& sparse_count,
    const int& max_seqlen_k
) {
    const auto [q_fp, q_sf] = q;
    const auto [kv_fp, kv_sf] = kv;
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;
    const bool is_mx_sf = q_sf.has_value();
    DG_HOST_ASSERT(not is_fp4 or is_mx_sf);
    int seq_len, seq_len_kv, num_heads, head_dim;

    const auto arch_major = device_runtime->get_arch_major();

    std::tie(seq_len, num_heads, head_dim) = get_logical_shape<3>(q_fp);
    DG_HOST_ASSERT((not is_fp4 and head_dim == 32) or head_dim == 64 or head_dim == 128);
    DG_HOST_ASSERT(q_fp.is_contiguous());
    DG_HOST_ASSERT(q_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn));

    if (is_mx_sf) {
        DG_HOST_ASSERT(arch_major == 10);
        auto [_s, _h] = get_shape<2>(q_sf.value());
        DG_HOST_ASSERT(seq_len == _s and num_heads == _h);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::kInt32);
    }

    int _head_dim;
    std::tie(seq_len_kv, _head_dim) = get_logical_shape<2>(kv_fp);
    DG_HOST_ASSERT(head_dim == _head_dim);
    DG_HOST_ASSERT(kv_fp.is_contiguous());
    DG_HOST_ASSERT(kv_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn));

    auto [_s_kv] = get_shape<1>(kv_sf);
    DG_HOST_ASSERT(seq_len_kv == _s_kv);
    DG_HOST_ASSERT(kv_sf.is_contiguous());
    DG_HOST_ASSERT(kv_sf.scalar_type() == (is_mx_sf ? torch::kInt32 : torch::kFloat));

    auto [_s, _h] = get_shape<2>(weights);
    DG_HOST_ASSERT(seq_len == _s and num_heads == _h);
    DG_HOST_ASSERT(weights.stride(1) == 1);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat
                   or (arch_major == 10 and weights.scalar_type() == torch::kBFloat16));

    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);

    DG_HOST_ASSERT(sparse_count >= 1 and sparse_count <= 2048);
    DG_HOST_ASSERT(arch_major == 10);
    DG_HOST_ASSERT(num_heads == 8 or num_heads == 16 or num_heads == 32 or num_heads == 64);

    // Allocate logits_buffer (temp, float32)
    constexpr int block_qh = 128;
    constexpr int block_kv = 256;
    const int block_q = block_qh / num_heads;
    const int stride_logits_alignment = 1024 / static_cast<int>(sizeof(float));
    int stride_logits;
    if (max_seqlen_k == 0) {
        stride_logits = align(seq_len_kv + block_kv, stride_logits_alignment);
    } else {
        stride_logits = align(align(max_seqlen_k, block_kv), stride_logits_alignment);
    }
    auto logits_buffer = torch::empty({align(seq_len, block_q), stride_logits},
        q_fp.options().dtype(torch::kFloat32));

    // Allocate sparse_indices (output, int32)
    auto sparse_indices = torch::empty({seq_len, sparse_count},
        q_fp.options().dtype(torch::kInt32));

    // Dispatch
    sm100_mqa_logits_with_topk(
        q_fp, q_sf, kv_fp, kv_sf, weights,
        cu_seq_len_k_start, cu_seq_len_k_end,
        logits_buffer, sparse_indices,
        seq_len, seq_len_kv, max_seqlen_k, stride_logits,
        num_heads, head_dim, block_q, block_kv,  // block_kv=256 is used as split_kv
        sparse_count, is_mx_sf, qk_dtype);

    return sparse_indices;
}

// =============================================================================
// Paged: fp8_fp4_paged_mqa_logits_with_topk
// =============================================================================
static torch::Tensor fp8_fp4_paged_mqa_logits_with_topk(
    const std::tuple<torch::Tensor, std::optional<torch::Tensor>>& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const torch::Tensor& schedule_meta,
    const int& max_context_len,
    const int& sparse_count,
    const std::optional<torch::Tensor>& indices
) {
    const auto [q_fp, q_sf] = q;
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;
    const bool is_mx_sf = q_sf.has_value();
    DG_HOST_ASSERT(not is_fp4 or is_mx_sf);

    int batch_size, next_n, num_heads, head_dim;
    int num_kv_blocks, block_kv;
    int block_table_stride = block_table.stride(0);
    int num_sms = device_runtime->get_num_sms();
    const auto arch_major = device_runtime->get_arch_major();

    std::tie(batch_size, next_n, num_heads, head_dim) = get_logical_shape<4>(q_fp);
    DG_HOST_ASSERT((not is_fp4 and head_dim == 32) or head_dim == 64 or head_dim == 128);
    DG_HOST_ASSERT(q_fp.is_contiguous());
    DG_HOST_ASSERT(q_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn));
    DG_HOST_ASSERT(next_n >= 1);

    if (is_mx_sf) {
        DG_HOST_ASSERT(arch_major == 10);
        auto [_b, _n, _h] = get_shape<3>(q_sf.value());
        DG_HOST_ASSERT(batch_size == _b and next_n == _n and num_heads == _h);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::kInt32);
    }

    int num_heads_kv, head_dim_with_sf;
    std::tie(num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf) = get_shape<4>(fused_kv_cache);
    DG_HOST_ASSERT(arch_major == 10 and (block_kv == 32 or block_kv == 64 or block_kv == 128));
    const int kv_head_dim = is_fp4 ? head_dim / 2 : head_dim;
    const int sf_bytes = static_cast<int>(is_mx_sf ? sizeof(int) : sizeof(float));
    DG_HOST_ASSERT(num_heads_kv == 1 and head_dim_with_sf == kv_head_dim + sf_bytes);
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf and fused_kv_cache.stride(3) == 1);
    DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte);

    auto kv_cache_stride_bytes = fused_kv_cache.stride(0);
    DG_HOST_ASSERT(kv_cache_stride_bytes % sf_bytes == 0);
    auto kv_cache = torch::from_blob(
        fused_kv_cache.data_ptr(),
        {num_kv_blocks, block_kv, kv_head_dim},
        {kv_cache_stride_bytes, kv_head_dim, 1},
        torch::TensorOptions().dtype(is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn));
    auto kv_cache_sf = torch::from_blob(
        fused_kv_cache.data_ptr<uint8_t>() + block_kv * kv_head_dim,
        {num_kv_blocks, block_kv},
        {kv_cache_stride_bytes / sf_bytes, 1},
        torch::TensorOptions().dtype(is_mx_sf ? torch::kInt32 : torch::kFloat32));

    auto [_bsn, _h] = get_shape<2>(weights);
    DG_HOST_ASSERT(_bsn == batch_size * next_n and _h == num_heads);
    DG_HOST_ASSERT(weights.stride(1) == 1);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat
                   or (arch_major == 10 and weights.scalar_type() == torch::kBFloat16));

    auto [_b, _mbl] = get_shape<2>(block_table);
    DG_HOST_ASSERT(_b == batch_size);
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);

    DG_HOST_ASSERT(context_lens.dim() == 2);
    const auto [__b, _n2] = get_shape<2>(context_lens);
    DG_HOST_ASSERT(batch_size == __b and next_n == _n2);
    DG_HOST_ASSERT(context_lens.is_contiguous());
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);

    auto [_sms, _mi] = get_shape<2>(schedule_meta);
    DG_HOST_ASSERT(_sms == num_sms + 1 and _mi == 2);
    DG_HOST_ASSERT(schedule_meta.is_contiguous());
    DG_HOST_ASSERT(schedule_meta.scalar_type() == torch::kInt);

    DG_HOST_ASSERT(sparse_count >= 1 and sparse_count <= 2048);
    DG_HOST_ASSERT(arch_major == 10);
    DG_HOST_ASSERT(num_heads == 8 or num_heads == 16 or num_heads == 32 or num_heads == 64);

    // Check indices (same as original)
    const bool is_varlen = indices.has_value();
    const auto indices_tensor = indices.value_or(torch::Tensor());
    if (is_varlen) {
        DG_HOST_ASSERT(arch_major == 10 and next_n == 1);
        DG_HOST_ASSERT(indices_tensor.dim() == 1 and indices_tensor.size(0) == batch_size);
        DG_HOST_ASSERT(indices_tensor.is_contiguous());
        DG_HOST_ASSERT(indices_tensor.scalar_type() == torch::kInt);
    }

    constexpr int split_kv = 256;
    const int stride_logits_alignment = 1024 / static_cast<int>(sizeof(float));
    const auto aligned_max_context_len = align(align(max_context_len, split_kv), stride_logits_alignment);
    auto logits_buffer = torch::empty({batch_size * next_n, aligned_max_context_len},
        q_fp.options().dtype(torch::kFloat32));

    auto sparse_indices = torch::empty({batch_size * next_n, sparse_count},
        q_fp.options().dtype(torch::kInt32));

    constexpr int splits_per_chunk = 16;
    sm100_paged_mqa_logits_with_topk(
        q_fp, q_sf, kv_cache, kv_cache_sf, weights, context_lens,
        logits_buffer, sparse_indices, block_table,
        indices_tensor, schedule_meta,
        batch_size, batch_size * next_n, next_n, num_heads, head_dim,
        num_kv_blocks, block_kv,
        true, is_varlen,
        aligned_max_context_len, block_table_stride, num_sms,
        split_kv, splits_per_chunk,
        sparse_count, is_mx_sf, qk_dtype);

    return sparse_indices;
}

// Legacy wrappers
static torch::Tensor fp8_mqa_logits_with_topk(
    const torch::Tensor& q,
    const std::tuple<torch::Tensor, torch::Tensor>& kv,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    const int& sparse_count,
    const int& max_seqlen_k
) {
    return fp8_fp4_mqa_logits_with_topk(
        std::make_tuple(q, std::nullopt), kv, weights,
        cu_seq_len_k_start, cu_seq_len_k_end,
        sparse_count, max_seqlen_k);
}

static torch::Tensor fp8_paged_mqa_logits_with_topk(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const torch::Tensor& schedule_meta,
    const int& max_context_len,
    const int& sparse_count,
    const std::optional<torch::Tensor>& indices
) {
    return fp8_fp4_paged_mqa_logits_with_topk(
        std::make_tuple(q, std::nullopt), fused_kv_cache, weights,
        context_lens, block_table, schedule_meta,
        max_context_len, sparse_count, indices);
}

#endif // DG_FP8_COMPATIBLE

static void register_topk_apis(pybind11::module_& m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def("fp8_fp4_mqa_logits_with_topk", &fp8_fp4_mqa_logits_with_topk,
          py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("sparse_count"),
          py::arg("max_seqlen_k") = 0);

    m.def("fp8_fp4_paged_mqa_logits_with_topk", &fp8_fp4_paged_mqa_logits_with_topk,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"),
          py::arg("sparse_count"),
          py::arg("indices") = std::nullopt);

    m.def("fp8_mqa_logits_with_topk", &fp8_mqa_logits_with_topk,
          py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("sparse_count"),
          py::arg("max_seqlen_k") = 0);

    m.def("fp8_paged_mqa_logits_with_topk", &fp8_paged_mqa_logits_with_topk,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"),
          py::arg("sparse_count"),
          py::arg("indices") = std::nullopt);
#endif
}

} // namespace deep_gemm::attention
