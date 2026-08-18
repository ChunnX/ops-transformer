/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_key_indexer.cpp
 * \brief PoolKeyIndexer torch extension C++ backend.
 *        Bridges PyTorch tensors to aclnnPoolKeyIndexer via ACLNN_CMD.
 *        Reference: ops/csrc/lightning_indexer.cpp (LIV2 non-quant path)
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;

// Derive output shapes from query shape + layout.
//   BSND: query (B,S1,N1,D) -> indices (B,S1,topk+poolSize-1) / values (B,S1,topk/poolSize)
//   TND:  query (T1,N1,D)   -> indices (T1,topk+poolSize-1)     / values (T1,topk/poolSize)
// Note: PKI output has no N2 dimension (N2 is fixed to 1), unlike LIV2 which emits keyHeadNum.
std::tuple<at::Tensor, at::Tensor> ConstructPoolKeyIndexerOutputTensor(
    const at::Tensor &query, int64_t topk, int64_t poolSize,
    const std::string &queryLayoutStr, bool returnValue)
{
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);
    TORCH_CHECK(poolSize > 0, "pool_size should be greater than 0, but now is ", poolSize);
    TORCH_CHECK(topk % poolSize == 0, "topk(", topk,
                ") should be divisible by pool_size(", poolSize, ").");

    int64_t indicesLen = topk + poolSize - 1;
    int64_t valuesLen = topk / poolSize;

    at::SmallVector<int64_t, SIZE> indicesShape;
    at::SmallVector<int64_t, SIZE> valuesShape;

    if (queryLayoutStr == "BSND") {
        indicesShape = {query.size(DIM_0), query.size(DIM_1), indicesLen};
        valuesShape = {query.size(DIM_0), query.size(DIM_1), valuesLen};
    } else {
        indicesShape = {query.size(DIM_0), indicesLen};
        valuesShape = {query.size(DIM_0), valuesLen};
    }

    at::Tensor sparseIndicesOut = at::empty(indicesShape, query.options().dtype(at::kInt));
    at::Tensor sparseValuesOut;
    if (returnValue) {
        // Spec requires sparseValuesOut to be FLOAT (not query dtype)
        sparseValuesOut = at::empty(valuesShape, query.options().dtype(at::kFloat));
    } else {
        sparseValuesOut = at::empty({0}, query.options().dtype(at::kFloat));
    }
    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

std::tuple<at::Tensor, at::Tensor>
PoolKeyIndexer(const at::Tensor &query, const at::Tensor &poolKey, const at::Tensor &weights,
              const at::Tensor &poolTailK,
              const c10::optional<at::Tensor> &actualSeqQ,
              const c10::optional<at::Tensor> &actualSeqK,
              const c10::optional<at::Tensor> &blockTable,
              const c10::optional<at::Tensor> &qDescale,
              const c10::optional<at::Tensor> &kDescale,
              c10::string_view layoutQ, c10::string_view layoutK,
              int64_t topk, int64_t poolSize, int64_t maskMode,
              int64_t quantMode, bool returnValue)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(poolKey.numel() > 0, "Tensor pool_key is empty.")
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.")
    TORCH_CHECK(poolTailK.numel() > 0, "Tensor pool_tail_k is empty.")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    std::tuple<at::Tensor, at::Tensor> poolKeyIndexerOutput =
        ConstructPoolKeyIndexerOutputTensor(query, topk, poolSize, queryLayoutStr, returnValue);
    at::Tensor sparseIndicesOut = std::get<0>(poolKeyIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(poolKeyIndexerOutput);

    at::Device outputDevice = query.device();
    auto actualSeqQVal = get_valid_tensor(actualSeqQ, outputDevice);
    auto actualSeqKVal = get_valid_tensor(actualSeqK, outputDevice);
    auto blockTableVal = get_valid_tensor(blockTable, outputDevice);
    auto qDescaleVal = get_valid_tensor(qDescale, outputDevice);
    auto kDescaleVal = get_valid_tensor(kDescale, outputDevice);

    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());

    // TODO(step4): When quant_mode=0 (FP8 per-token-head) or quant_mode=1 (mxFP8),
    // implement FixPkiAclDtypes to convert query/key/descale to proper ACL dtypes
    // (FP8_E4M3FN / FLOAT8_E8M0) using TensorWrapper, similar to quant_lightning_indexer.cpp.
    // Currently only quant_mode=-1 (no quantization) is fully supported.

    ACLNN_CMD(aclnnPoolKeyIndexer,
              query, poolKey, weights, poolTailK,
              actualSeqQVal, actualSeqKVal, blockTableVal,
              qDescaleVal, kDescaleVal,
              queryLayoutPtr, keyLayoutPtr,
              topk, poolSize, maskMode, quantMode, returnValue,
              sparseIndicesOut, sparseValuesOut);

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pool_key_indexer", &PoolKeyIndexer, "pool_key_indexer");
}
} // namespace op_api
