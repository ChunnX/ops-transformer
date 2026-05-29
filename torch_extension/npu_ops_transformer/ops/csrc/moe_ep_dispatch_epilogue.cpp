#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
const int DIM_TWO = 2;

tensor_list npu_moe_ep_dispatch_epilogue(
    const at::Tensor &dst_buffer_slot_idx,
    const at::Tensor &recv_src_metadata_internal,
    const at::Tensor &num_recv_per_rank_hp,
    const at::Tensor &num_recv_per_expert_hp,
    const std::string &group_ep,
    int64_t ep_world_size, int64_t ep_rank_id,
    int64_t num_experts, int64_t num_max_tokens_per_rank,
    int64_t expert_alignment)
{
    TORCH_CHECK(dst_buffer_slot_idx.dim() == DIM_TWO, "dst_buffer_slot_idx must be 2D");

    int64_t top_k = dst_buffer_slot_idx.size(1);
    int64_t num_local_experts = num_experts / ep_world_size;
    int64_t upper_a = ep_world_size * num_max_tokens_per_rank;

    at::Tensor recv_x;
    at::Tensor recv_topk_idx;
    at::Tensor recv_topk_weights;
    at::Tensor recv_src_metadata;
    at::Tensor num_recv_per_rank;
    at::Tensor num_recv_per_expert;
    {
        auto localDevice = c10::Device(dst_buffer_slot_idx.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        recv_x = at::empty({upper_a, -1}, dst_buffer_slot_idx.options().dtype(at::kBFloat16));
        recv_topk_idx = at::empty({upper_a, top_k}, dst_buffer_slot_idx.options().dtype(at::kInt));
        recv_topk_weights = at::empty({upper_a, top_k}, dst_buffer_slot_idx.options().dtype(at::kFloat));
        recv_src_metadata = at::empty({upper_a, 3}, dst_buffer_slot_idx.options().dtype(at::kInt));
        num_recv_per_rank = at::empty({ep_world_size}, dst_buffer_slot_idx.options().dtype(at::kInt));
        num_recv_per_expert = at::empty({num_local_experts}, dst_buffer_slot_idx.options().dtype(at::kLong));
    }

    char *group_ep_ptr = const_cast<char *>(group_ep.c_str());

    ACLNN_CMD(aclnnMoeEpDispatchEpilogue,
        dst_buffer_slot_idx, recv_src_metadata_internal,
        num_recv_per_rank_hp, num_recv_per_expert_hp,
        group_ep_ptr, ep_world_size, ep_rank_id,
        num_experts, num_max_tokens_per_rank, expert_alignment,
        recv_x, recv_topk_idx, recv_topk_weights,
        recv_src_metadata, num_recv_per_rank, num_recv_per_expert);

    return std::tie(recv_x, recv_topk_idx, recv_topk_weights,
                    recv_src_metadata, num_recv_per_rank, num_recv_per_expert);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_moe_ep_dispatch_epilogue", &npu_moe_ep_dispatch_epilogue, "moe_ep_dispatch_epilogue");
}

}  // namespace op_api