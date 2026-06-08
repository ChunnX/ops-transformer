#include "../moe_distribute_combine_v2/ascend950/moe_distribute_combine_v2_a5_entry.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {
namespace MoeDistributeCombineV2A5 {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("npu_moe_distribute_combine_v2_a5(Tensor expand_x, Tensor expert_ids, "
          "Tensor assist_info, Tensor ep_send_counts, Tensor expert_scales, "
          "str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, "
          "Tensor mc2_context, int total_winsize_ep, "
          "*, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, "
          "Tensor? shared_expert_x=None, Tensor? elastic_info=None, "
          "Tensor? ori_x=None, Tensor? const_expert_alpha_1=None, "
          "Tensor? const_expert_alpha_2=None, Tensor? const_expert_v=None, "
          "Tensor? performance_info=None, int global_bs=0) -> Tensor");
}

static void calculate_tilingkey(int32_t &tilingKey, at::ScalarType xType)
{
    tilingKey = (xType == at::kBFloat16) ? 1 : 0;
}

static void calculate_tilingdata(MoeDistributeCombineV2Info &td,
    int64_t ep_world, int64_t ep_rank, int64_t moe_exp, int64_t winsize_ep,
    int64_t global_bs, int64_t bs, int64_t h, int64_t k, int64_t a)
{
    memset(&td, 0, sizeof(td));
    td.epWorldSize    = static_cast<uint32_t>(ep_world);
    td.epRankId       = static_cast<uint32_t>(ep_rank);
    td.moeExpertNum   = static_cast<uint32_t>(moe_exp);
    td.globalBs       = static_cast<uint32_t>(global_bs);
    td.bs = static_cast<uint32_t>(bs);
    td.k  = static_cast<uint32_t>(k);
    td.h  = static_cast<uint32_t>(h);
    td.a  = static_cast<uint32_t>(a);
    td.aivNum = 0;
    td.isMc2Context = true;
    td.totalWinSizeEp = static_cast<uint64_t>(winsize_ep);
}

at::Tensor npu_moe_distribute_combine_v2_a5_meta(
    const at::Tensor &expand_x, const at::Tensor &expert_ids,
    const at::Tensor &assist_info, const at::Tensor &ep_send_counts,
    const at::Tensor &expert_scales,
    c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t moe_expert_num, const at::Tensor &mc2_context, int64_t total_winsize_ep,
    const c10::optional<at::Tensor> &tp_send_counts,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &shared_expert_x,
    const c10::optional<at::Tensor> &elastic_info,
    const c10::optional<at::Tensor> &ori_x,
    const c10::optional<at::Tensor> &const_expert_alpha_1,
    const c10::optional<at::Tensor> &const_expert_alpha_2,
    const c10::optional<at::Tensor> &const_expert_v,
    const c10::optional<at::Tensor> &performance_info,
    int64_t global_bs)
{
    return at::empty({expert_ids.size(0), expand_x.size(1)}, expand_x.options());
}

static void* _opt(const c10::optional<at::Tensor> &t) {
    return (t.has_value() && t->defined() ? t->data_ptr() : nullptr);
}

at::Tensor npu_moe_distribute_combine_v2_a5(
    const at::Tensor &expand_x, const at::Tensor &expert_ids,
    const at::Tensor &assist_info, const at::Tensor &ep_send_counts,
    const at::Tensor &expert_scales,
    c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t moe_expert_num, const at::Tensor &mc2_context, int64_t total_winsize_ep,
    const c10::optional<at::Tensor> &tp_send_counts,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &shared_expert_x,
    const c10::optional<at::Tensor> &elastic_info,
    const c10::optional<at::Tensor> &ori_x,
    const c10::optional<at::Tensor> &const_expert_alpha_1,
    const c10::optional<at::Tensor> &const_expert_alpha_2,
    const c10::optional<at::Tensor> &const_expert_v,
    const c10::optional<at::Tensor> &performance_info,
    int64_t global_bs)
{
    const c10::OptionalDeviceGuard guard(expand_x.device());
    int64_t bs = expert_ids.size(0);
    int64_t h = expand_x.size(1);
    int64_t k = expert_ids.size(1);
    int64_t gbs = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = gbs * k;

    at::Tensor out = at::empty({bs, h}, expand_x.options());

    MoeDistributeCombineV2Info td;
    calculate_tilingdata(td, ep_world_size, ep_rank_id, moe_expert_num,
        total_winsize_ep, gbs, bs, h, k, a);

    int32_t tilingKey = 0;
    calculate_tilingkey(tilingKey, expand_x.scalar_type());

    auto s = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() mutable -> int {
        moe_distribute_combine_v2_a5_entry(tilingKey, td.aivNum == 0 ? 8u : td.aivNum, s,
            expand_x.data_ptr(),
            expert_ids.data_ptr(),
            assist_info.data_ptr(),
            ep_send_counts.data_ptr(),
            expert_scales.data_ptr(),
            _opt(tp_send_counts),
            _opt(x_active_mask),
            _opt(shared_expert_x),
            _opt(elastic_info),
            _opt(ori_x),
            _opt(const_expert_alpha_1),
            _opt(const_expert_alpha_2),
            _opt(const_expert_v),
            _opt(performance_info),
            out.data_ptr(),
            nullptr,  // workspace
            mc2_context.data_ptr(),
            td);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("moeDistributeCombineV2A5", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("npu_moe_distribute_combine_v2_a5", TORCH_FN(npu_moe_distribute_combine_v2_a5));
}

TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("npu_moe_distribute_combine_v2_a5", &npu_moe_distribute_combine_v2_a5_meta);
}

} // namespace MoeDistributeCombineV2A5
} // namespace ascend_ops
