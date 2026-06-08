#include "../moe_distribute_dispatch_v2/ascend950/moe_distribute_dispatch_v2_a5_entry.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {
namespace MoeDistributeDispatchV2A5 {

constexpr int DIM_ONE = 1;
constexpr int DIM_TWO = 2;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("npu_moe_distribute_dispatch_v2_a5(Tensor x, Tensor expert_ids, "
          "str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, "
          "Tensor mc2_context, int total_winsize_ep, "
          "*, Tensor? scales=None, Tensor? x_active_mask=None, "
          "Tensor? expert_scales=None, Tensor? performance_info=None, "
          "int quant_mode=0, int global_bs=0, int tp_world_size=0) "
          "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

static void calculate_tilingkey(int32_t &tilingKey, at::ScalarType xType, int quantMode)
{
    tilingKey = quantMode;
    if (xType == at::kBFloat16) tilingKey += 1;
}

static void calculate_tilingdata(MoeDistributeDispatchV2Info &td,
    int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num,
    int64_t total_winsize_ep, int64_t global_bs, int64_t bs, int64_t h, int64_t k, int64_t a,
    int64_t quant_mode, const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &performance_info)
{
    memset(&td, 0, sizeof(td));
    td.epWorldSize    = static_cast<uint32_t>(ep_world_size);
    td.epRankId       = static_cast<uint32_t>(ep_rank_id);
    td.moeExpertNum   = static_cast<uint32_t>(moe_expert_num);
    td.quantMode      = static_cast<uint32_t>(quant_mode);
    td.globalBs       = static_cast<uint32_t>(global_bs);
    td.bs = static_cast<uint32_t>(bs);
    td.k  = static_cast<uint32_t>(k);
    td.h  = static_cast<uint32_t>(h);
    td.a  = static_cast<uint32_t>(a);
    td.aivNum = 0;
    td.isMc2Context = true;  // A3方案：走 Mc2Context 路径
    td.totalWinSizeEp = static_cast<uint64_t>(total_winsize_ep);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_moe_distribute_dispatch_v2_a5_meta(
    const at::Tensor &x, const at::Tensor &expert_ids, c10::string_view group_ep,
    int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num,
    const at::Tensor &mc2_context, int64_t total_winsize_ep,
    const c10::optional<at::Tensor> &scales,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &expert_scales,
    const c10::optional<at::Tensor> &performance_info,
    int64_t quant_mode, int64_t global_bs, int64_t tp_world_size)
{
    int64_t bs = x.size(0);
    int64_t h = x.size(1);
    int64_t gbs = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = gbs * expert_ids.size(1);
    int64_t local_moe_exp = moe_expert_num / ep_world_size;
    auto out_dtype = (quant_mode == 0) ? x.scalar_type() : at::kChar;
    at::Tensor expand_x = at::empty({a, h}, x.options().dtype(out_dtype));
    at::Tensor dyn_scales = at::empty({a}, x.options().dtype(at::kFloat));
    at::Tensor assist = at::empty({std::max(bs * (int64_t)expert_ids.size(1), a * 128L)}, x.options().dtype(at::kInt));
    at::Tensor token_nums = at::empty({local_moe_exp}, x.options().dtype(at::kLong));
    at::Tensor ep_cnts = at::empty({ep_world_size * local_moe_exp}, x.options().dtype(at::kInt));
    at::Tensor exp_scales = at::empty({0}, x.options().dtype(at::kFloat));
    return std::tie(expand_x, dyn_scales, assist, token_nums, ep_cnts, exp_scales);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_moe_distribute_dispatch_v2_a5(
    const at::Tensor &x, const at::Tensor &expert_ids, c10::string_view group_ep,
    int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num,
    const at::Tensor &mc2_context, int64_t total_winsize_ep,
    const c10::optional<at::Tensor> &scales,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &expert_scales,
    const c10::optional<at::Tensor> &performance_info,
    int64_t quant_mode, int64_t global_bs, int64_t tp_world_size)
{
    const c10::OptionalDeviceGuard guard(x.device());
    int64_t bs = x.size(0);
    int64_t h = x.size(1);
    int64_t k = expert_ids.size(1);
    int64_t gbs = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = gbs * k;
    int64_t local_moe_exp = moe_expert_num / ep_world_size;

    auto out_dtype = (quant_mode == 0) ? x.scalar_type() : at::kChar;
    at::Tensor expand_x = at::empty({a, h}, x.options().dtype(out_dtype));
    at::Tensor dyn_scales = at::empty({a}, x.options().dtype(at::kFloat));
    at::Tensor assist = at::empty({std::max(bs * k, a * 128L)}, x.options().dtype(at::kInt));
    at::Tensor token_nums = at::empty({local_moe_exp}, x.options().dtype(at::kLong));
    at::Tensor ep_cnts = at::empty({ep_world_size * local_moe_exp}, x.options().dtype(at::kInt));
    at::Tensor exp_scales = at::empty({0}, x.options().dtype(at::kFloat));

    MoeDistributeDispatchV2Info td;
    calculate_tilingdata(td, ep_world_size, ep_rank_id, moe_expert_num,
        total_winsize_ep, gbs, bs, h, k, a, quant_mode, x_active_mask, performance_info);

    int32_t tilingKey = 0;
    calculate_tilingkey(tilingKey, x.scalar_type(), static_cast<int>(quant_mode));

    auto s = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() mutable -> int {
        moe_distribute_dispatch_v2_a5_entry(tilingKey, td.aivNum == 0 ? 8u : td.aivNum, s,
            x.data_ptr(),
            expert_ids.data_ptr(),
            (scales.has_value() && scales->defined() ? scales->data_ptr() : nullptr),
            (x_active_mask.has_value() && x_active_mask->defined() ? x_active_mask->data_ptr() : nullptr),
            (expert_scales.has_value() && expert_scales->defined() ? expert_scales->data_ptr() : nullptr),
            (performance_info.has_value() && performance_info->defined() ? performance_info->data_ptr() : nullptr),
            expand_x.data_ptr(),
            dyn_scales.data_ptr(),
            assist.data_ptr(),
            token_nums.data_ptr(),
            ep_cnts.data_ptr(),
            exp_scales.data_ptr(),
            nullptr,  // workspace
            mc2_context.data_ptr(),  // mc2Context 真实地址
            td);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("moeDistributeDispatchV2A5", acl_call);
    return std::tie(expand_x, dyn_scales, assist, token_nums, ep_cnts, exp_scales);
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("npu_moe_distribute_dispatch_v2_a5", TORCH_FN(npu_moe_distribute_dispatch_v2_a5));
}

TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("npu_moe_distribute_dispatch_v2_a5", &npu_moe_distribute_dispatch_v2_a5_meta);
}

} // namespace MoeDistributeDispatchV2A5
} // namespace ascend_ops
