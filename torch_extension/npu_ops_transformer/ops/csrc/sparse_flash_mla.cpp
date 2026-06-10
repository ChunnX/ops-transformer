/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#include <cmath>
#include <string>
#include <tuple>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;

const at::Tensor &OptionalTensorValue(const c10::optional<at::Tensor> &tensor)
{
    static const at::Tensor emptyTensor;
    return tensor.has_value() ? tensor.value() : emptyTensor;
}

int64_t GetStride0(const at::Tensor &tensor)
{
    return tensor.defined() && tensor.dim() > 0 ? tensor.stride(0) : 0;
}

std::tuple<at::Tensor, at::Tensor> MakeOutputs(const at::Tensor &q, const at::Tensor &oriKv,
                                               const std::string &layoutKv, bool returnSoftmaxLse)
{
    TORCH_CHECK(q.defined() && q.numel() > 0, "Tensor q must be defined and non-empty.");
    TORCH_CHECK(oriKv.defined(), "Tensor ori_kv must be defined.");
    at::Tensor attnOut = at::empty_like(q);
    at::Tensor softmaxLse;
    if (!returnSoftmaxLse) {
        softmaxLse = at::empty({0}, q.options().dtype(at::kFloat));
        return std::make_tuple(attnOut, softmaxLse);
    }

    if (q.dim() == 3) {
        int64_t n2 = layoutKv == "PA_BNBD" ? oriKv.size(DIM_2) : oriKv.size(DIM_1);
        softmaxLse = at::empty({n2, q.size(0), q.size(1) / n2}, q.options().dtype(at::kFloat));
    } else {
        int64_t n2 = oriKv.size(DIM_2);
        softmaxLse = at::empty({q.size(0), n2, q.size(1), q.size(2) / n2}, q.options().dtype(at::kFloat));
    }
    return std::make_tuple(attnOut, softmaxLse);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_sparse_flash_mla(
    const at::Tensor &q,
    const c10::optional<at::Tensor> &oriKv, const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices, const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &oriBlockTable, const c10::optional<at::Tensor> &cmpBlockTable,
    const c10::optional<at::Tensor> &cuSeqLensQ, const c10::optional<at::Tensor> &cuSeqLensOriKv,
    const c10::optional<at::Tensor> &cuSeqLensCmpKv, const c10::optional<at::Tensor> &seqUsedQ,
    const c10::optional<at::Tensor> &seqUsedOriKv, const c10::optional<at::Tensor> &seqUsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv, const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, c10::optional<double> softmaxScale,
    c10::optional<int64_t> cmpRatio, c10::optional<int64_t> oriMaskMode, c10::optional<int64_t> cmpMaskMode,
    c10::optional<int64_t> oriWinLeft, c10::optional<int64_t> oriWinRight,
    c10::optional<c10::string_view> layoutQ, c10::optional<c10::string_view> layoutKv,
    c10::optional<int64_t> topkValueMode, c10::optional<bool> returnSoftmaxLse)
{
    const at::Tensor &oriKvValue = OptionalTensorValue(oriKv);
    const at::Tensor &cmpKvValue = OptionalTensorValue(cmpKv);

    std::string layoutQStr = std::string(layoutQ.value_or("BSND"));
    std::string layoutKvStr = std::string(layoutKv.value_or("BSND"));
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    auto outputs = MakeOutputs(q, oriKvValue, layoutKvStr, returnSoftmaxLse.value_or(false));
    at::Tensor attnOut = std::get<0>(outputs);
    at::Tensor softmaxLse = std::get<1>(outputs);

    const double softmaxScaleValue = softmaxScale.value_or(1.0 / std::sqrt(static_cast<double>(q.size(q.dim() - 1))));
    const int64_t cmpRatioValue = cmpRatio.value_or(1);
    const int64_t oriMaskModeValue = oriMaskMode.value_or(0);
    const int64_t cmpMaskModeValue = cmpMaskMode.value_or(0);
    const int64_t oriWinLeftValue = oriWinLeft.value_or(-1);
    const int64_t oriWinRightValue = oriWinRight.value_or(-1);
    const int64_t topkValueModeValue = topkValueMode.value_or(1);
    const int64_t oriKvStride = GetStride0(oriKvValue);
    const int64_t cmpKvStride = GetStride0(cmpKvValue);
    const bool returnSoftmaxLseValue = returnSoftmaxLse.value_or(false);

    ACLNN_CMD(aclnnSparseFlashMla, q, oriKv, cmpKv, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
              cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv, seqUsedQ, seqUsedOriKv, seqUsedCmpKv, cmpResidualKv,
              oriTopkLength, cmpTopkLength, sinks, metadata, softmaxScaleValue, cmpRatioValue, oriMaskModeValue,
              cmpMaskModeValue, oriWinLeftValue, oriWinRightValue, layoutQPtr, layoutKvPtr, topkValueModeValue,
              oriKvStride, cmpKvStride, returnSoftmaxLseValue, attnOut, softmaxLse);
    return std::make_tuple(attnOut, softmaxLse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_flash_mla", &npu_sparse_flash_mla, "npu_sparse_flash_mla");
}
} // namespace op_api
