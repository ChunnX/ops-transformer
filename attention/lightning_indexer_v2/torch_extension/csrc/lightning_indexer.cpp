/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"
#include "lightning_indexer_common.h"

namespace op_api {
using namespace at_npu::native;

constexpr int64_t LI_V2_METADATA_SIZE = 1024;

at::Tensor LightningIndexerMetadata(int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
                                    const c10::optional<at::Tensor> &cuSeqlensQ,
                                    const c10::optional<at::Tensor> &cuSeqlensK,
                                    const c10::optional<at::Tensor> &sequsedQ,
                                    const c10::optional<at::Tensor> &sequsedK,
                                    const c10::optional<at::Tensor> &cmpResidualK, int64_t batchSize,
                                    int64_t maxSeqlenQ, int64_t maxSeqlenK, c10::string_view layoutQ,
                                    c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio)
{
    at::Device outputDevice = at::Device(std::string("npu"));
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else if (cuSeqlensK.has_value()) {
        outputDevice = cuSeqlensK.value().device();
    } else if (sequsedQ.has_value()) {
        outputDevice = sequsedQ.value().device();
    } else if (sequsedK.has_value()) {
        outputDevice = sequsedK.value().device();
    } else if (cmpResidualK.has_value()) {
        outputDevice = cmpResidualK.value().device();
    }

    at::Tensor output = torch::empty({LI_V2_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = get_valid_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensKVal = get_valid_tensor(cuSeqlensK, outputDevice);
    auto sequsedQVal = get_valid_tensor(sequsedQ, outputDevice);
    auto sequsedKVal = get_valid_tensor(sequsedK, outputDevice);
    auto cmpResidualKVal = get_valid_tensor(cmpResidualK, outputDevice);

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKStr = std::string(layoutK);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKPtr = const_cast<char *>(layoutKStr.c_str());

    ACLNN_CMD(aclnnLightningIndexerV2Metadata, cuSeqlensQVal, cuSeqlensKVal, sequsedQVal, sequsedKVal, cmpResidualKVal,
              numHeadsQ, numHeadsK, headDim, topk, batchSize, maxSeqlenQ, maxSeqlenK, layoutQPtr, layoutKPtr, maskMode,
              cmpRatio, output);
    return output;
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lightning_indexer_metadata", &LightningIndexerMetadata, "lightning_indexer_metadata");
    m.def("lightning_indexer", &LightningIndexer<false>, "lightning_indexer");
}
} // namespace op_api
