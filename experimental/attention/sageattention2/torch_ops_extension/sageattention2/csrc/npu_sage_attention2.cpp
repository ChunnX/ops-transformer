/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <tuple>
#include <torch/library.h>
#include "ops_common.h"

namespace sageattention2 {
using namespace at_npu::native;

std::tuple<at::Tensor, at::Tensor> MakeOutputs(
    const at::Tensor &query, c10::string_view tensorLayout, bool returnLse)
{
    TORCH_CHECK(query.dim() == 4, "query must be rank 4, got ", query.dim());
    const std::string layout(tensorLayout);
    TORCH_CHECK(layout == "HND" || layout == "NHD", "tensor_layout must be HND or NHD, got ", layout);
    const int64_t headAxis = layout == "HND" ? 1 : 2;
    const int64_t seqAxis = layout == "HND" ? 2 : 1;
    at::Tensor output = at::empty(query.sizes(), query.options());
    at::Tensor lse = returnLse
        ? at::empty({query.size(0), query.size(headAxis), query.size(seqAxis)}, query.options().dtype(at::kFloat))
        : at::empty({0}, query.options().dtype(at::kFloat));
    return {output, lse};
}

std::tuple<at::Tensor, at::Tensor> SageAttention2Npu(const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, c10::string_view tensorLayout, bool isCausal, c10::string_view qkQuantGran,
    double smScale, c10::string_view pvAccumDtype, bool smoothK, bool smoothV, bool returnLse)
{
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
        "query, key, and value must all be rank 4");
    TORCH_CHECK(query.device() == key.device() && query.device() == value.device(),
        "query, key, and value must be on the same device");
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(),
        "query, key, and value must have the same dtype");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
        "query, key, and value must be float16 or bfloat16");
    TORCH_CHECK(query.size(-1) > 0 && query.size(-1) <= 128, "head_dim must be in [1, 128]");
    TORCH_CHECK(query.stride(-1) == 1 && key.stride(-1) == 1 && value.stride(-1) == 1,
        "the last dimension of query, key, and value must be contiguous");
    const std::string layout(tensorLayout);
    const std::string gran(qkQuantGran);
    const std::string accum(pvAccumDtype);
    TORCH_CHECK(layout == "HND" || layout == "NHD", "tensor_layout must be HND or NHD, got ", layout);
    TORCH_CHECK(gran == "per_warp" || gran == "per_thread",
        "qk_quant_gran must be per_warp or per_thread, got ", gran);
    TORCH_CHECK(accum == "fp32" || accum == "fp32+fp32" || accum == "fp32+fp16",
        "pv_accum_dtype must be fp32, fp32+fp32, or fp32+fp16, got ", accum);
    const int64_t headAxis = layout == "HND" ? 1 : 2;
    const int64_t seqAxis = layout == "HND" ? 2 : 1;
    TORCH_CHECK(query.size(0) > 0 && query.size(headAxis) > 0 && query.size(seqAxis) > 0 &&
            key.size(0) > 0 && key.size(headAxis) > 0 && key.size(seqAxis) > 0 && value.size(0) > 0 &&
            value.size(headAxis) > 0 && value.size(seqAxis) > 0,
        "batch, head, and sequence dimensions must be positive");
    TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0), "batch dimensions must match");
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value shapes must match");
    TORCH_CHECK(query.size(3) == key.size(3), "query and key head_dim must match");
    TORCH_CHECK(query.size(headAxis) % key.size(headAxis) == 0, "q_heads must be divisible by kv_heads");
    TORCH_CHECK(!isCausal || query.size(seqAxis) == key.size(seqAxis),
        "causal mode requires q_seq == kv_seq");

    auto outputs = MakeOutputs(query, tensorLayout, returnLse);
    at::Tensor output = std::get<0>(outputs);
    at::Tensor lse = std::get<1>(outputs);
    std::string layoutStorage(layout);
    std::string granStorage(gran);
    std::string accumStorage(accum);
    char *layoutData = layoutStorage.data();
    char *granData = granStorage.data();
    char *accumData = accumStorage.data();
    EXEC_NPU_CMD_V1(aclnnSageAttention2, query, key, value, layoutData, isCausal, granData,
        smScale, accumData, smoothK, smoothV, returnLse, output, lse);
    return {output, lse};
}

std::tuple<at::Tensor, at::Tensor> SageAttention2Meta(const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, c10::string_view tensorLayout, bool isCausal, c10::string_view qkQuantGran,
    double smScale, c10::string_view pvAccumDtype, bool smoothK, bool smoothV, bool returnLse)
{
    return MakeOutputs(query, tensorLayout, returnLse);
}
} // namespace sageattention2

TORCH_LIBRARY_IMPL(sageattention2, PrivateUse1, m)
{
    m.impl("sage_attention2", &sageattention2::SageAttention2Npu);
}

TORCH_LIBRARY_IMPL(sageattention2, Meta, m)
{
    m.impl("sage_attention2", &sageattention2::SageAttention2Meta);
}
