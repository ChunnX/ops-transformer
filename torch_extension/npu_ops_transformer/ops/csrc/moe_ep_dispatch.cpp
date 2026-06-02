#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
const int DIM_TWO = 2;

tensor_list npu_moe_ep_dispatch(
    const at::Tensor &context,
    const at::Tensor &x,
    const c10::optional<at::Tensor> &topk_idx,
    const c10::optional<at::Tensor> &topk_weights,
    const c10::optional<at::Tensor> &cached_handle_dst_buffer_slot_idx,
    int64_t ep_world_size, int64_t ep_rank_id,
    int64_t num_experts, int64_t num_max_tokens_per_rank, int64_t ccl_buffer_size,
    int64_t expert_alignment, bool do_cpu_sync)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x must be 2D");
    TORCH_CHECK(ep_world_size > 1, "ep_world_size must be > 1");
    TORCH_CHECK(ep_rank_id >= 0 && ep_rank_id < ep_world_size, "ep_rank_id out of range");
    TORCH_CHECK(num_experts % ep_world_size == 0, "num_experts must be divisible by ep_world_size");
    TORCH_CHECK(x.dtype() == at::kBFloat16, "x must be bfloat16");

    bool any_cached = cached_handle_dst_buffer_slot_idx.has_value();
    TORCH_CHECK(!(any_cached && do_cpu_sync), "cached mode is incompatible with do_cpu_sync=True");
    TORCH_CHECK(any_cached || (topk_idx.has_value() && topk_weights.has_value()),
                "non-cached path requires topk_idx and topk_weights");

    auto x_size = x.sizes();
    int64_t num_tokens = x_size[0];
    int64_t hidden = x_size[1];
    int64_t top_k = 8;
    if (topk_idx.has_value()) {
        top_k = topk_idx->size(1);
    } else if (cached_handle_dst_buffer_slot_idx.has_value()) {
        top_k = cached_handle_dst_buffer_slot_idx->size(1);
    }
    int64_t num_local_experts = num_experts / ep_world_size;
    int64_t upper_a = ep_world_size * num_max_tokens_per_rank;

    at::Tensor num_recv_per_rank_hp;
    at::Tensor num_recv_per_expert_hp;
    at::Tensor dst_slot;
    at::Tensor recv_src_meta_internal;
    {
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        num_recv_per_rank_hp = at::empty({ep_world_size}, x.options().dtype(at::kInt));
        num_recv_per_expert_hp = at::empty({num_local_experts}, x.options().dtype(at::kLong));
        dst_slot = at::empty({num_tokens, top_k}, x.options().dtype(at::kInt));
        recv_src_meta_internal = at::empty({upper_a, 3}, x.options().dtype(at::kInt));
    }

    at::Tensor topk_idx_tensor = topk_idx.has_value() ? *topk_idx : at::Tensor();
    at::Tensor topk_weights_tensor = topk_weights.has_value() ? *topk_weights : at::Tensor();
    at::Tensor cached_slot_tensor = cached_handle_dst_buffer_slot_idx.has_value() ? *cached_handle_dst_buffer_slot_idx : at::Tensor();

    ACLNN_CMD(aclnnMoeEpDispatch,
        context, x, topk_idx_tensor, topk_weights_tensor,
        cached_slot_tensor,
        ep_world_size, ep_rank_id,
        num_experts, num_max_tokens_per_rank, ccl_buffer_size, expert_alignment, do_cpu_sync,
        num_recv_per_rank_hp, num_recv_per_expert_hp,
        dst_slot, recv_src_meta_internal);

    return std::tie(num_recv_per_rank_hp, num_recv_per_expert_hp, dst_slot, recv_src_meta_internal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_moe_ep_dispatch", &npu_moe_ep_dispatch, "moe_ep_dispatch");
}

}  // namespace op_api