// =============================================================================
// sm100_mqa_logits_with_topk.hpp
//
// EXACT copy of sm100_mqa_logits.hpp, only changes:
//   1. #include <deep_gemm/impls/sm100_mqa_logits_with_topk.cuh> (instead of sm100_mqa_logits.cuh)
//   2. kSparseCount template parameter added to generate_impl
//   3. logits replaced by logits_buffer (float*) + sparse_indices (int*)
//   4. kernel name: sm100_mqa_logits_with_topk (instead of sm100_mqa_logits)
// =============================================================================

#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"
#include "sm100_mqa_logits.hpp"  // reuse get_mqa_logits_smem_size and make_tma_2d_desc
#include <deep_gemm/layout/mqa_logits.cuh>

namespace deep_gemm {

// =============================================================================
// Non-paged runtime
// =============================================================================
class SM100MQALogitsWithTopKRuntime final: public LaunchRuntime<SM100MQALogitsWithTopKRuntime> {
public:
    struct Args {
        int num_q_tokens;
        int num_kv_tokens;
        int stride_logits;
        int num_heads, head_dim;
        bool is_mx_sf, is_compressed_logits;

        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int split_kv;
        int sparse_count;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        float* logits_buffer;
        int* sparse_indices;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_sf_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_sf_kv;
        CUtensorMap tensor_map_weights;
        at::ScalarType qk_dtype;
        at::ScalarType weights_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(128 % args.num_heads == 0);

        return fmt::format(R"(
#include <deep_gemm/impls/sm100_mqa_logits_with_topk.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mqa_logits_with_topk<
        {}, {},
        {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {},
        {}, {}, {}
    >);
}};
)", args.num_heads, args.head_dim,
    args.is_mx_sf ? "true" : "false",
    args.is_compressed_logits,
    args.block_q, args.split_kv,
    args.num_q_stages, args.num_kv_stages,
    args.launch_args.grid_dim.first,
    args.num_specialized_threads, args.num_math_threads,
    args.sparse_count,
    args.qk_dtype == kPackedFP4 ? "cutlass::float_e2m1_t" : "cutlass::float_e4m3_t",
    "float",
    args.weights_dtype == torch::kBFloat16 ? "__nv_bfloat16" : "float");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.num_q_tokens, args.num_kv_tokens,
            args.stride_logits,
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits_buffer, args.sparse_indices,
            args.tensor_map_q, args.tensor_map_sf_q,
            args.tensor_map_kv, args.tensor_map_sf_kv,
            args.tensor_map_weights
        ));
    }
};

// =============================================================================
// Host wrapper: non-paged
// =============================================================================
static void sm100_mqa_logits_with_topk(
    const torch::Tensor& q, const std::optional<torch::Tensor>& sf_q,
    const torch::Tensor& kv, const torch::Tensor& sf_kv,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start, const torch::Tensor& cu_seq_len_k_end,
    torch::Tensor& logits_buffer, torch::Tensor& sparse_indices,
    const int& num_q_tokens, const int& num_kv_tokens,
    const int& max_seqlen_k, const int& stride_logits,
    const int& num_heads, const int& head_dim,
    const int& block_q, const int& split_kv,
    const int& sparse_count,
    const bool& is_mx_sf, const at::ScalarType& qk_dtype) {

    const bool is_fp4 = qk_dtype == kPackedFP4;

    constexpr int num_specialized_threads = 128;
    const int num_math_threads = 2 * 128;
    const int num_q_stages = 3;
    const int num_kv_stages = is_fp4 ? 10 : 5;
    const bool is_compressed_logits = (max_seqlen_k > 0);

    // Build TMA descriptors (identical to original sm100_mqa_logits)
    CUtensorMap tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv;
    if (is_fp4)
        DG_HOST_ASSERT(head_dim == 64 or head_dim == 128);
    else
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);

    const int swizzle_mode = is_fp4 ? head_dim / 2 : head_dim;
    tensor_map_q = make_tma_2d_desc(q, head_dim, num_q_tokens * num_heads,
                                    head_dim, block_q * num_heads,
                                    static_cast<int>(q.stride(1)),
                                    swizzle_mode, 0, false, not is_fp4);
    tensor_map_kv = make_tma_2d_desc(kv, head_dim, num_kv_tokens,
                                     head_dim, split_kv,
                                     static_cast<int>(kv.stride(0)),
                                     swizzle_mode, 0, false, not is_fp4);
    tensor_map_sf_kv = make_tma_2d_desc(sf_kv,
                                        get_tma_aligned_size(num_kv_tokens, static_cast<int>(sf_kv.element_size())), 1,
                                        split_kv, 1, 0, 0);
    if (is_mx_sf) {
        tensor_map_sf_q = make_tma_2d_desc(sf_q.value(), num_heads, num_q_tokens,
                                           num_heads, block_q,
                                           static_cast<int>(sf_q.value().stride(0)), 0);
    } else {
        tensor_map_sf_q = tensor_map_sf_kv;
    }
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, num_q_tokens,
                                                     num_heads, block_q,
                                                     static_cast<int>(weights.stride(0)), 0);

    // SMEM size: original + TopK scratch (kSparseCount * 8 bytes)
    const int smem_size = get_mqa_logits_smem_size(num_heads, head_dim, is_mx_sf, qk_dtype);
    // Phase 2: kSpecWarpStart(=8) warps, each needs sparse_count * 8 bytes scratch
    const int topk_smem = 8 * sparse_count * 8;
    const int total_smem = std::max(smem_size, topk_smem);
    DG_HOST_ASSERT(total_smem <= SM100ArchSpec::smem_capacity);

    const SM100MQALogitsWithTopKRuntime::Args args = {
        .num_q_tokens = num_q_tokens,
        .num_kv_tokens = num_kv_tokens,
        .stride_logits = stride_logits,
        .num_heads = num_heads, .head_dim = head_dim,
        .is_mx_sf = is_mx_sf,
        .is_compressed_logits = is_compressed_logits,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .split_kv = split_kv,
        .sparse_count = sparse_count,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits_buffer = logits_buffer.data_ptr<float>(),
        .sparse_indices = sparse_indices.data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_sf_q = tensor_map_sf_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_sf_kv = tensor_map_sf_kv,
        .tensor_map_weights = tensor_map_weights,
        .qk_dtype = qk_dtype,
        .weights_dtype = weights.scalar_type(),
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads,
                                  total_smem)
    };
    const auto code = SM100MQALogitsWithTopKRuntime::generate(args);
    const auto runtime = compiler->build("sm100_mqa_logits_with_topk", code);
    SM100MQALogitsWithTopKRuntime::launch(runtime, args);
}

// =============================================================================
// Paged runtime
// =============================================================================
class SM100PagedMQALogitsWithTopKRuntime final: public LaunchRuntime<SM100PagedMQALogitsWithTopKRuntime> {
public:
    struct Args {
        int num_q_tokens_total;
        int tokens_per_request;
        int num_heads;
        int head_dim;
        int page_kv;
        bool is_mx_sf;
        bool is_context_lens_2d;
        bool is_varlen;
        int block_table_stride;
        int logits_stride;
        int sparse_count;

        int num_q_stages;
        int num_kv_stages;
        int split_kv;
        int splits_per_chunk;

        int* context_lens;
        float* logits_buffer;
        int* sparse_indices;
        int* block_table;
        int* indices;
        int* schedule_meta;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_sf_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_sf_kv;
        CUtensorMap tensor_map_weights;
        at::ScalarType qk_dtype;
        at::ScalarType weights_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_mqa_logits_with_topk.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits_with_topk<
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {}, {}, {}
    >);
}};
)", args.tokens_per_request, args.num_heads,
    args.head_dim, args.page_kv,
    args.is_mx_sf ? "true" : "false",
    args.is_context_lens_2d, args.is_varlen ? "true" : "false",
    args.num_q_stages, args.num_kv_stages,
    args.split_kv, args.splits_per_chunk,
    args.num_specialized_threads, args.num_math_threads,
    args.sparse_count,
    args.qk_dtype == kPackedFP4 ? "cutlass::float_e2m1_t" : "cutlass::float_e4m3_t",
    "float",
    args.weights_dtype == torch::kBFloat16 ? "__nv_bfloat16" : "float");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.num_q_tokens_total,
            args.logits_stride, args.block_table_stride,
            args.context_lens, args.logits_buffer, args.sparse_indices,
            args.block_table, args.indices, args.schedule_meta,
            args.tensor_map_q, args.tensor_map_sf_q,
            args.tensor_map_kv, args.tensor_map_sf_kv,
            args.tensor_map_weights
        ));
    }
};

// =============================================================================
// Host wrapper: paged
// =============================================================================
static void sm100_paged_mqa_logits_with_topk(
    const torch::Tensor& q, const std::optional<torch::Tensor>& sf_q,
    const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_sf,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    torch::Tensor& logits_buffer, torch::Tensor& sparse_indices,
    const torch::Tensor& block_table, const torch::Tensor& indices,
    const torch::Tensor& schedule_meta,
    const int& num_requests, const int& num_q_tokens_total,
    const int& tokens_per_request,
    const int& num_heads, const int& head_dim,
    const int& num_kv_blocks, const int& page_kv,
    const bool& is_context_lens_2d, const bool& is_varlen,
    const int& logits_stride, const int& block_table_stride,
    const int& num_sms, const int& split_kv, const int& splits_per_chunk,
    const int& sparse_count,
    const bool& is_mx_sf, const at::ScalarType& qk_dtype) {

    const bool is_fp4 = qk_dtype == kPackedFP4;
    const int num_specialized_threads = 128;
    const int num_math_threads = 2 * 128;
    DG_HOST_ASSERT(split_kv == 256 and logits_stride % split_kv == 0);
    const int num_q_stages = 3;
    const int num_kv_stages = is_fp4 ? 10 : 5;
    DG_HOST_ASSERT(128 % num_heads == 0);
    const int block_q = 128 / num_heads;

    // Build TMA descriptors (identical to original sm100_paged_mqa_logits)
    CUtensorMap tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv;
    if (is_fp4)
        DG_HOST_ASSERT(head_dim == 64 or head_dim == 128);
    else
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);

    const int swizzle_mode = is_fp4 ? head_dim / 2 : head_dim;
    tensor_map_q = make_tma_2d_desc(q, head_dim, num_requests * tokens_per_request * num_heads,
                                    head_dim, block_q * num_heads,
                                    static_cast<int>(q.stride(2)),
                                    swizzle_mode, 0, false, not is_fp4);
    tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, page_kv, num_kv_blocks,
                                     head_dim, page_kv, 1,
                                     static_cast<int>(kv_cache.stride(1)),
                                     static_cast<int>(kv_cache.stride(0)),
                                     swizzle_mode, 0, false, not is_fp4);
    tensor_map_sf_kv = make_tma_2d_desc(kv_cache_sf, page_kv, num_kv_blocks,
                                        page_kv, 1,
                                        static_cast<int>(kv_cache_sf.stride(0)), 0);
    if (is_mx_sf) {
        tensor_map_sf_q = make_tma_2d_desc(sf_q.value(), num_heads, num_requests * tokens_per_request,
                                           num_heads, block_q,
                                           static_cast<int>(sf_q.value().stride(1)), 0);
    } else {
        tensor_map_sf_q = tensor_map_sf_kv;
    }
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, num_requests * tokens_per_request,
                                                     num_heads, block_q,
                                                     static_cast<int>(weights.stride(0)), 0);

    const int smem_size = get_mqa_logits_smem_size(num_heads, head_dim, is_mx_sf, qk_dtype);
    const int topk_smem = 8 * sparse_count * 8;
    const int total_smem = std::max(smem_size, topk_smem);
    DG_HOST_ASSERT(total_smem <= SM100ArchSpec::smem_capacity);

    const SM100PagedMQALogitsWithTopKRuntime::Args args = {
        .num_q_tokens_total = num_q_tokens_total,
        .tokens_per_request = tokens_per_request,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .page_kv = page_kv,
        .is_mx_sf = is_mx_sf,
        .is_context_lens_2d = is_context_lens_2d,
        .is_varlen = is_varlen,
        .block_table_stride = block_table_stride,
        .logits_stride = logits_stride,
        .sparse_count = sparse_count,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .split_kv = split_kv,
        .splits_per_chunk = splits_per_chunk,
        .context_lens = context_lens.data_ptr<int>(),
        .logits_buffer = logits_buffer.data_ptr<float>(),
        .sparse_indices = sparse_indices.data_ptr<int>(),
        .block_table = block_table.data_ptr<int>(),
        .indices = is_varlen ? indices.data_ptr<int>() : nullptr,
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_sf_q = tensor_map_sf_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_sf_kv = tensor_map_sf_kv,
        .tensor_map_weights = tensor_map_weights,
        .qk_dtype = qk_dtype,
        .weights_dtype = weights.scalar_type(),
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_sms,
                                  num_specialized_threads + num_math_threads,
                                  total_smem)
    };
    const auto code = SM100PagedMQALogitsWithTopKRuntime::generate(args);
    const auto runtime = compiler->build("sm100_paged_mqa_logits_with_topk", code);
    SM100PagedMQALogitsWithTopKRuntime::launch(runtime, args);
}

} // namespace deep_gemm
