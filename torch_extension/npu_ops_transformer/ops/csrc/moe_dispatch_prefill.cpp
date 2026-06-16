/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_dispatch_prefill.cpp
 * \brief Fused MoE EP prefill: notify_dispatch + dispatch_normal (Mc2 context).
 *
 * Internal flow mirrors sgl-kernel-npu deep_ep Buffer::intranode_dispatch (notify then
 * dispatch_normal). Mc2 uses context/ccl_buffer_size instead of HCCL comm group name.
 */

#include <algorithm>
#include <iostream>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t QUANT_MODE_NO_QUANT = 0;
constexpr int64_t QUANT_MODE_DYNAMIC_SCALES = 2;
constexpr int64_t EXPAND_IDX_COUNT_PER_GROUP = 3;
constexpr int64_t NUM_RANKS_MIN = 2;
constexpr int64_t NUM_RANKS_MAX = 384;
// sgl deep_ep.cpp: EP_HOST_ASSERT(r * t <= 131072)
constexpr int64_t MAX_TOTAL_TOKENS = 131072;

tensor_list npu_moe_dispatch_prefill(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &topk_idx,
    const std::vector<at::Tensor> &ext_handle_data, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t local_ranksize, int64_t moe_expert_num, int64_t ccl_buffer_size,
    const c10::optional<at::Tensor> &x_scales, const c10::optional<at::Tensor> &topk_weights,
    const c10::optional<at::Tensor> &num_tokens_per_rank, const c10::optional<at::Tensor> &is_token_in_rank,
    const c10::optional<at::Tensor> &num_tokens_per_expert,
    const c10::optional<at::Tensor> &dispatch_wait_recv_cost_stats, int64_t num_worst_tokens,
    int64_t quant_mode, int64_t round, int64_t per_round_tokens)
{
    (void)x_scales;
    (void)is_token_in_rank;
    (void)ext_handle_data[0];  // layout scratch; notify sendData allocated below (umdk-style).

    TORCH_CHECK(num_tokens_per_expert.has_value(), "num_tokens_per_expert is required.");
    TORCH_CHECK(ext_handle_data.size() == 2,
                "ext_handle_data must be [layout_scratch, send_token_idx_small].");
    TORCH_CHECK(ep_world_size >= NUM_RANKS_MIN && ep_world_size <= NUM_RANKS_MAX,
                "ep_world_size must be in [", NUM_RANKS_MIN, ", ", NUM_RANKS_MAX, "], got ", ep_world_size);
    TORCH_CHECK((moe_expert_num % ep_world_size) == 0,
                "moe_expert_num must be divisible by ep_world_size.");
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size), "ep_rank_id out of range.");
    TORCH_CHECK(ccl_buffer_size > 0, "ccl_buffer_size must be positive.");
    TORCH_CHECK(round > 0 && per_round_tokens > 0, "round and per_round_tokens must be positive.");
    TORCH_CHECK(round * per_round_tokens <= MAX_TOTAL_TOKENS,
                "round * per_round_tokens must be <= ", MAX_TOTAL_TOKENS, ", got round=", round,
                ", per_round_tokens=", per_round_tokens);
    TORCH_CHECK((quant_mode == QUANT_MODE_NO_QUANT) || (quant_mode == QUANT_MODE_DYNAMIC_SCALES),
                "quant_mode must be 0 or 2.");

    TORCH_CHECK(x.dim() == DIM_TWO && x.is_contiguous(), "x must be contiguous 2D.");
    TORCH_CHECK(topk_idx.dim() == DIM_TWO && topk_idx.is_contiguous(), "topk_idx must be contiguous 2D.");

    const at::Tensor &tokens_per_expert = num_tokens_per_expert.value();
    TORCH_CHECK(tokens_per_expert.scalar_type() == at::kInt, "num_tokens_per_expert must be int32.");
    TORCH_CHECK(tokens_per_expert.dim() == DIM_ONE && tokens_per_expert.is_contiguous(),
                "num_tokens_per_expert must be contiguous 1D.");
    TORCH_CHECK((tokens_per_expert.size(0) % ep_world_size) == 0,
                "num_tokens_per_expert length must be divisible by ep_world_size.");
    TORCH_CHECK(tokens_per_expert.size(0) == moe_expert_num * round,
                "num_tokens_per_expert size must equal moe_expert_num * round.");

    const at::Tensor &send_token_idx_small = ext_handle_data[1];
    TORCH_CHECK(send_token_idx_small.dim() == DIM_TWO && send_token_idx_small.is_contiguous(),
                "send_token_idx_small must be contiguous 2D.");

    if (num_tokens_per_rank.has_value()) {
        const at::Tensor &ntr = num_tokens_per_rank.value();
        TORCH_CHECK(ntr.dim() == DIM_ONE && ntr.is_contiguous(), "num_tokens_per_rank must be contiguous 1D.");
        TORCH_CHECK(ntr.size(0) == ep_world_size, "num_tokens_per_rank size must equal ep_world_size.");
    }

    const int64_t num_tokens = x.size(0);
    const int64_t hidden = x.size(1);
    const int64_t num_topk = topk_idx.size(1);
    TORCH_CHECK(num_tokens == topk_idx.size(0), "topk_idx batch must match x.");
    // sgl get_dispatch_layout: num_tokens <= round * per_round_tokens
    TORCH_CHECK(num_tokens <= round * per_round_tokens,
                "num_tokens must be <= round * per_round_tokens (capacity), got num_tokens=", num_tokens,
                ", round=", round, ", per_round_tokens=", per_round_tokens,
                ", capacity=", round * per_round_tokens);
    TORCH_CHECK(num_tokens == send_token_idx_small.size(0), "send_token_idx_small batch must match x.");
    TORCH_CHECK(num_topk == send_token_idx_small.size(1), "send_token_idx_small topk must match topk_idx.");

    if (topk_weights.has_value()) {
        const at::Tensor &tw = topk_weights.value();
        TORCH_CHECK(tw.dim() == DIM_TWO && tw.is_contiguous(), "topk_weights must be contiguous 2D.");
        TORCH_CHECK(num_tokens == tw.size(0) && num_topk == tw.size(1),
                    "topk_weights shape must match topk_idx.");
    }

    // umdk: topkIdx -> int32 expertIds before dispatch_normal
    at::Tensor topk_idx_int32 =
        (topk_idx.scalar_type() == at::kInt) ? topk_idx : topk_idx.to(at::kInt);

    const int64_t num_experts = moe_expert_num;
    const int64_t num_local_experts = num_experts / ep_world_size;
    const int64_t send_per_group = EXPAND_IDX_COUNT_PER_GROUP;
    // umdk: sendCount = sendPerGroup * numLocalExperts * numRanks (= sendPerGroup * numExperts, round=1)
    const int64_t send_count = send_per_group * num_local_experts * ep_world_size * round;
    // A3 intranode notify: local_ranksize must equal ep_world_size (sgl deep_ep.cpp intranode path).
    TORCH_CHECK(local_ranksize == ep_world_size,
                "local_ranksize for notify must equal ep_world_size on A3 intranode, got local_ranksize=",
                local_ranksize, ", ep_world_size=", ep_world_size);
    const int64_t notify_local_rank_size = local_ranksize;
    const int64_t notify_local_rank_id = ep_rank_id % notify_local_rank_size;

    auto int32_opts = at::TensorOptions()
                          .dtype(at::kInt)
                          .device(x.device())
                          .memory_format(c10::MemoryFormat::Contiguous);

    // --- notify_dispatch (sgl: [round, num_experts * send_per_group] buffers) ---
    at::Tensor send_data = at::empty({round, num_experts * send_per_group}, int32_opts);
    at::Tensor send_data_offset = at::empty({round, num_experts}, int32_opts);
    at::Tensor recv_data = at::empty({round, num_experts * send_per_group}, int32_opts);
    at::Tensor total_recv_token = at::empty({1}, int32_opts);
    at::Tensor recv_count = at::empty({round, num_experts}, int32_opts);
    at::Tensor recv_offset = at::empty({round, num_experts}, int32_opts);
    at::Tensor max_bs = at::empty({1}, int32_opts);
    at::Tensor recv_tokens_per_expert = at::empty({round * num_local_experts}, int32_opts);
    at::Tensor expert_global_offset = at::empty({num_local_experts}, int32_opts);
    at::Tensor srcrank_in_expert_offset = at::empty({num_local_experts * ep_world_size}, int32_opts);
    at::Tensor r_in_srcrank_offset = at::empty({num_local_experts * ep_world_size * round}, int32_opts);
    at::Tensor notify_ext_handle = at::empty({0}, int32_opts);

    at::TensorList recv_offset_ref = {recv_offset, recv_count};
    at::TensorList expert_output_offset_ref = {expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset};

    TORCH_CHECK(context.scalar_type() == at::kInt && context.dim() == DIM_ONE, "context must be 1D int32.");

    std::cout << "[cpp prefill before NotifyDispatch] "
              << "ep_world_size=" << ep_world_size << " ep_rank_id=" << ep_rank_id
              << " local_ranksize=" << local_ranksize << " moe_expert_num=" << moe_expert_num
              << " ccl_buffer_size=" << ccl_buffer_size << " num_worst_tokens=" << num_worst_tokens
              << " quant_mode=" << quant_mode << " round=" << round << " per_round_tokens=" << per_round_tokens
              << " send_count=" << send_count << " num_tokens=" << num_tokens
              << " topk_idx.sizes=" << topk_idx.sizes()
              << " num_tokens_per_rank.sizes="
              << (num_tokens_per_rank.has_value() ? num_tokens_per_rank.value().sizes() : c10::IntArrayRef{})
              << " is_token_in_rank.sizes="
              << (is_token_in_rank.has_value() ? is_token_in_rank.value().sizes() : c10::IntArrayRef{})
              << " num_tokens_per_expert.sizes=" << tokens_per_expert.sizes()
              << " send_data.sizes=" << send_data.sizes() << " send_token_idx_small.sizes=" << send_token_idx_small.sizes()
              << std::endl;
    std::cout << "[cpp ext_handle_data] len=" << ext_handle_data.size() << std::endl;
    for (size_t i = 0; i < ext_handle_data.size(); ++i) {
        const at::Tensor &t = ext_handle_data[i];
        std::cout << "  ext_handle_data[" << i << "] sizes=" << t.sizes() << " dtype=" << t.scalar_type()
                  << " device=" << t.device() << std::endl;
    }

    ACLNN_CMD(aclnnNotifyDispatch, context, send_data, tokens_per_expert, send_count, num_tokens, ep_world_size,
              ep_rank_id, notify_local_rank_size, notify_local_rank_id, ccl_buffer_size, round, per_round_tokens,
              send_data_offset, recv_data, recv_offset_ref, expert_output_offset_ref, total_recv_token, max_bs,
              recv_tokens_per_expert, notify_ext_handle);

    // sgl intranode_dispatch: real_max_bs then global_bs capped by per_round_tokens
    int64_t real_max_bs = std::max(max_bs.item<int64_t>(), num_worst_tokens);
    int64_t global_bs =
        static_cast<int64_t>(std::min(per_round_tokens, real_max_bs) * ep_world_size);
    TORCH_CHECK(global_bs > 0, "global_bs must be positive after notify.");

    int64_t trt = total_recv_token.item<int64_t>();
    int64_t num_recv_tokens = (trt == 0) ? 1 : trt;

    // --- dispatch_normal (umdk: expandxOut / dynamicScales / expandIdx) ---
    at::Tensor expand_x;
    if (quant_mode == QUANT_MODE_DYNAMIC_SCALES) {
        expand_x = at::empty({num_recv_tokens, hidden},
                             at::TensorOptions().dtype(at::kChar).device(x.device()).memory_format(
                                 c10::MemoryFormat::Contiguous));
    } else {
        expand_x = at::empty({num_recv_tokens, hidden}, x.options());
    }
    at::Tensor dynamic_scales_out = at::empty({num_recv_tokens},
                                              at::TensorOptions().dtype(at::kFloat).device(x.device()).memory_format(
                                                  c10::MemoryFormat::Contiguous));
    at::Tensor expand_idx_out = at::empty({num_recv_tokens * EXPAND_IDX_COUNT_PER_GROUP}, int32_opts);
    at::Tensor dispatch_ext_handle = at::empty({0}, int32_opts);

    at::Tensor dispatch_wait_recv_cost_stats_out;
    if (dispatch_wait_recv_cost_stats.has_value()) {
        const at::Tensor &stats = dispatch_wait_recv_cost_stats.value();
        TORCH_CHECK(stats.scalar_type() == at::kInt && stats.dim() == DIM_ONE && stats.is_contiguous(),
                    "dispatch_wait_recv_cost_stats must be contiguous 1D int32.");
        TORCH_CHECK(stats.size(0) == ep_world_size,
                    "dispatch_wait_recv_cost_stats size must equal ep_world_size.");
        dispatch_wait_recv_cost_stats_out = stats;
    }

    ACLNN_CMD(aclnnDispatchNormal, context, x, topk_idx_int32, send_data_offset, send_token_idx_small, recv_offset_ref,
              expert_output_offset_ref, ep_world_size, ep_rank_id, moe_expert_num, ccl_buffer_size, quant_mode,
              real_max_bs, global_bs, round, per_round_tokens, expand_x,
              dynamic_scales_out, expand_idx_out, dispatch_wait_recv_cost_stats_out, dispatch_ext_handle);

    // sgl: recv_count_one_dim = recv_count.sum(0) for combine send_head / ep_recv_counts
    at::Tensor recv_count_out = recv_count.sum(0).to(at::kInt);

    return std::make_tuple(expand_x, dynamic_scales_out, expand_idx_out, recv_count_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_dispatch_prefill", &npu_moe_dispatch_prefill, "moe_dispatch_prefill");
}
}  // namespace op_api
