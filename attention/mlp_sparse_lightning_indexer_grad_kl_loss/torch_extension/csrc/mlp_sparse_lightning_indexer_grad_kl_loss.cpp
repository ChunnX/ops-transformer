/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <string>
#include <tuple>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mlp_sparse_lightning_indexer_grad_kl_loss(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &queryIndex,
    const at::Tensor &keyIndex,
    const at::Tensor &weight,
    const at::Tensor &sparseIndices,
    const at::Tensor &softmaxMax,
    const at::Tensor &softmaxSum,
    const c10::optional<at::Tensor> &queryRope,
    const c10::optional<at::Tensor> &keyRope,
    const c10::optional<at::Tensor> &mask,
    const c10::optional<at::IntArrayRef> &curSeqLengthsQuery,
    const c10::optional<at::IntArrayRef> &curSeqLengthsKey,
    double scaleValue,
    std::string layout,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t blockSize,
    bool deterministic,
    int64_t validTokenNum)
{
    at::Tensor dQueryIndex = at::zeros_like(queryIndex);
    at::Tensor dKeyIndex = at::zeros_like(keyIndex);
    at::Tensor dWeight = at::zeros(weight.sizes(), weight.options().dtype(at::kFloat));
    at::Tensor loss = at::zeros({1}, weight.options().dtype(at::kFloat));
    // SDK op_build generates a double parameter for this ACLNN Float attribute.
    char *layoutPtr = const_cast<char *>(layout.c_str());

    ACLNN_CMD(aclnnMlpSparseLightningIndexerGradKLLoss,
              query, key, queryIndex, keyIndex, weight, sparseIndices, softmaxMax, softmaxSum,
              queryRope, keyRope, mask, curSeqLengthsQuery, curSeqLengthsKey,
              scaleValue, layoutPtr, sparseMode, preTokens, nextTokens,
              blockSize, deterministic, validTokenNum,
              dQueryIndex, dKeyIndex, dWeight, loss);
    return std::make_tuple(dQueryIndex, dKeyIndex, dWeight, loss);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mlp_sparse_lightning_indexer_grad_kl_loss",
          &mlp_sparse_lightning_indexer_grad_kl_loss,
          "MLP sparse lightning indexer grad KL loss (synchronization-only)");
}
} // namespace op_api
