/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 */

/*!
 * \file generic_sparse_attention_grad.cpp
 * \brief PyTorch ACLNN binding for GenericSparseAttentionGrad.
 */

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor, at::Tensor> GenericSparseAttentionGrad(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &dout,
    const at::Tensor &out, const at::Tensor &lse, const at::Tensor &rsvdBlockIdx,
    const at::Tensor &rsvdBlockCount, const at::Tensor &metadata, const c10::optional<at::Tensor> &attenMask,
    const c10::optional<at::Tensor> &cuSeqLengths, const c10::optional<at::Tensor> &cuSeqLengthsKv,
    at::IntArrayRef blockShape, int64_t isPackedGqa, const std::string &qInputLayout,
    const std::string &kvInputLayout, double scaleValue, int64_t maskType, int64_t softmaxPrecision,
    int64_t windowSizeLeft, int64_t windowSizeRight)
{
    const at::Tensor &attenMaskValue = attenMask.value_or(at::Tensor());
    const at::Tensor &cuSeqLengthsValue = cuSeqLengths.value_or(at::Tensor());
    const at::Tensor &cuSeqLengthsKvValue = cuSeqLengthsKv.value_or(at::Tensor());

    at::Tensor dQuery = at::empty_like(query);
    at::Tensor dKey = at::empty_like(key);
    at::Tensor dValue = at::empty_like(value);

    char *qInputLayoutPtr = const_cast<char *>(qInputLayout.c_str());
    char *kvInputLayoutPtr = const_cast<char *>(kvInputLayout.c_str());

    ACLNN_CMD(aclnnGenericSparseAttentionGrad, query, key, value, dout, out, lse, rsvdBlockIdx, rsvdBlockCount,
              metadata, attenMaskValue, cuSeqLengthsValue, cuSeqLengthsKvValue, blockShape, isPackedGqa,
              qInputLayoutPtr, kvInputLayoutPtr, scaleValue, maskType, softmaxPrecision, windowSizeLeft,
              windowSizeRight, dQuery, dKey, dValue);

    return std::make_tuple(dQuery, dKey, dValue);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generic_sparse_attention_grad", &GenericSparseAttentionGrad, "generic_sparse_attention_grad");
}
} // namespace op_api
