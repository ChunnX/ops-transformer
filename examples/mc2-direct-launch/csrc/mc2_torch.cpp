/**
 * @file mc2_torch.cpp
 * @brief pybind11 Python interface for MC2 <<<>>> direct launch
 */
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <cstring>
#include "tiling_host.h"

extern "C" void mc2_dispatch_launch(int32_t tilingKey, uint32_t blockDim, void *stream,
    void *mc2Context, void *x, void *expertIds,
    void *scales, void *xActiveMask, void *elasticInfo,
    void *performanceInfo, void *expandXOut, void *dynamicScalesOut,
    void *expandIdxOut, void *expertTokenNumsOut, void *epSendCountsOut,
    void *tpSendCountsOut, void *expandScalesOut, void *workspaceGM,
    void *tilingInfoBytesPtr);

extern "C" void mc2_combine_launch(int32_t tilingKey, uint32_t blockDim, void *stream,
    void *mc2Context, void *expandX, void *expertIds,
    void *expandIdx, void *epSendCount,
    void *tpSendCount, void *residualX, void *gamma,
    void *expertScales, void *xActiveMask, void *sharedExpertX,
    void *elasticInfo, void *oriX, void *constExpertAlpha1,
    void *constExpertAlpha2, void *constExpertV, void *performanceInfo,
    void *yOut, void *rstdOut, void *XOut, void *workspaceGM,
    void *tilingInfoBytesPtr);

static void *ptr_or_null(const c10::optional<at::Tensor> &t) {
    return (t.has_value() && t->defined()) ? t->data_ptr() : nullptr;
}

void npu_mc2_dispatch(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &expert_ids,
    int64_t tiling_key, int64_t block_dim,
    const c10::optional<at::Tensor> &scales,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &elastic_info,
    const c10::optional<at::Tensor> &performance_info,
    const c10::optional<at::Tensor> &expert_scales,
    at::Tensor &expand_x, at::Tensor &dynamic_scales,
    at::Tensor &assist_info, at::Tensor &expert_token_nums,
    at::Tensor &ep_send_counts, at::Tensor &tp_send_counts,
    at::Tensor &expand_scales, at::Tensor &workspace,
    const at::Tensor &tiling_tensor)
{
    TORCH_CHECK(tiling_tensor.nbytes() >= sizeof(MoeDistributeDispatchV2Info),
        "tiling tensor too small");
    MoeDistributeDispatchV2Info tiling_info;
    std::memcpy(&tiling_info, tiling_tensor.data_ptr(), sizeof(MoeDistributeDispatchV2Info));

    auto stream = c10_npu::getCurrentNPUStream().stream();
    mc2_dispatch_launch(
        static_cast<int32_t>(tiling_key), static_cast<uint32_t>(block_dim), stream,
        context.data_ptr(), x.data_ptr(), expert_ids.data_ptr(),
        ptr_or_null(scales), ptr_or_null(x_active_mask), ptr_or_null(elastic_info),
        ptr_or_null(performance_info), expand_x.data_ptr(), dynamic_scales.data_ptr(),
        assist_info.data_ptr(), expert_token_nums.data_ptr(),
        ep_send_counts.data_ptr(), tp_send_counts.data_ptr(),
        expand_scales.data_ptr(), workspace.data_ptr(), &tiling_info);
}

void npu_mc2_combine(
    const at::Tensor &context, const at::Tensor &expand_x,
    const at::Tensor &expert_ids, const at::Tensor &assist_info,
    const at::Tensor &ep_send_count, const at::Tensor &expert_scales,
    int64_t tiling_key, int64_t block_dim,
    const c10::optional<at::Tensor> &tp_send_count,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &shared_expert_x,
    const c10::optional<at::Tensor> &elastic_info,
    const c10::optional<at::Tensor> &ori_x,
    const c10::optional<at::Tensor> &performance_info,
    at::Tensor &x_out, at::Tensor &workspace,
    const at::Tensor &tiling_tensor)
{
    TORCH_CHECK(tiling_tensor.nbytes() >= sizeof(MoeDistributeCombineV2Info),
        "tiling tensor too small");
    MoeDistributeCombineV2Info tiling_info;
    std::memcpy(&tiling_info, tiling_tensor.data_ptr(), sizeof(MoeDistributeCombineV2Info));

    auto stream = c10_npu::getCurrentNPUStream().stream();
    mc2_combine_launch(
        static_cast<int32_t>(tiling_key), static_cast<uint32_t>(block_dim), stream,
        context.data_ptr(), expand_x.data_ptr(), expert_ids.data_ptr(),
        assist_info.data_ptr(), ep_send_count.data_ptr(),
        ptr_or_null(tp_send_count),
        nullptr, nullptr,                        // residualX, gamma
        expert_scales.data_ptr(),
        ptr_or_null(x_active_mask), ptr_or_null(shared_expert_x),
        ptr_or_null(elastic_info), ptr_or_null(ori_x),
        nullptr, nullptr, nullptr,               // constExpertAlpha1/2/v
        ptr_or_null(performance_info),
        nullptr, nullptr,                        // yOut, rstdOut
        x_out.data_ptr(), workspace.data_ptr(), &tiling_info);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mc2_dispatch",  &npu_mc2_dispatch,  "MC2 dispatch <<<>>> launch");
    m.def("mc2_combine",   &npu_mc2_combine,   "MC2 combine <<<>>> launch");
}
