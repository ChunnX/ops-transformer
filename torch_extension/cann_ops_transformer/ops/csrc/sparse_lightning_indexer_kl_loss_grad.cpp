/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <string>
#include <tuple>
#include "aclnn_common.h"

namespace op_api {
constexpr int64_t SLI_KL_LOSS_GRAD_METADATA_SIZE = 64;

at::Device GetSparseLightningKlLossGradMetadataDevice(
    const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensK,
    const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedK,
    const c10::optional<at::Tensor> &cmpResidualK)
{
    if (cuSeqlensQ.has_value() && cuSeqlensQ.value().defined()) {
        return cuSeqlensQ.value().device();
    }
    if (cuSeqlensK.has_value() && cuSeqlensK.value().defined()) {
        return cuSeqlensK.value().device();
    }
    if (sequsedQ.has_value() && sequsedQ.value().defined()) {
        return sequsedQ.value().device();
    }
    if (sequsedK.has_value() && sequsedK.value().defined()) {
        return sequsedK.value().device();
    }
    if (cmpResidualK.has_value() && cmpResidualK.value().defined()) {
        return cmpResidualK.value().device();
    }
    return at::Device(torch_npu::utils::get_npu_device_type());
}

at::Tensor SparseLightningIndexerKlLossGradMetadata(
    int64_t numHeadsQ,
    int64_t numHeadsK,
    int64_t headDim,
    const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensK,
    const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedK,
    const c10::optional<at::Tensor> &cmpResidualK,
    int64_t batchSize,
    int64_t maxSeqlenQ,
    int64_t maxSeqlenK,
    int64_t topk,
    std::string layoutQ,
    std::string layoutK,
    int64_t maskMode,
    int64_t cmpRatio)
{
    at::Tensor output = at::empty(
        {SLI_KL_LOSS_GRAD_METADATA_SIZE},
        at::TensorOptions().dtype(at::kInt).device(GetSparseLightningKlLossGradMetadataDevice(
            cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK)));

    char *layoutQPtr = const_cast<char *>(layoutQ.c_str());
    char *layoutKPtr = const_cast<char *>(layoutK.c_str());

    ACLNN_CMD(aclnnSparseLightningIndexerKLLossGradMetadata,
        cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK,
        batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim, topk,
        layoutQPtr, layoutKPtr, maskMode, cmpRatio, output);

    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> SparseLightningIndexerKlLossGrad(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &w,
    const at::Tensor &sparseIndices,
    const at::Tensor &attnSoftmaxL1Norm,
    const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensK,
    const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedK,
    const c10::optional<at::Tensor> &cmpResidualK,
    const c10::optional<at::Tensor> &metadata,
    std::string layoutQ,
    std::string layoutK,
    int64_t maskMode,
    int64_t cmpRatio)
{
    at::Tensor dq = at::empty_like(q);
    at::Tensor dk = at::empty_like(k);
    at::Tensor dw = at::empty_like(w);
    at::Tensor softmaxOut = at::empty_like(attnSoftmaxL1Norm);

    char *layoutQPtr = const_cast<char *>(layoutQ.c_str());
    char *layoutKPtr = const_cast<char *>(layoutK.c_str());

    ACLNN_CMD(aclnnSparseLightningIndexerKLLossGrad,
        q, k, w, sparseIndices, attnSoftmaxL1Norm, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK,
        cmpResidualK, metadata, layoutQPtr, layoutKPtr, maskMode, cmpRatio,
        dq, dk, dw, softmaxOut);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(dq, dk, dw, softmaxOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "sparse_lightning_indexer_kl_loss_grad_metadata",
        &SparseLightningIndexerKlLossGradMetadata,
        "sparse_lightning_indexer_kl_loss_grad_metadata");
    m.def(
        "sparse_lightning_indexer_kl_loss_grad",
        &SparseLightningIndexerKlLossGrad,
        "sparse_lightning_indexer_kl_loss_grad");
}

}  // namespace op_api
