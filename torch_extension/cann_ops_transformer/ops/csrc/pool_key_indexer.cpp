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

std::vector<bool> IsContiguousAxes(const at::Tensor &tensor)
{
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t ndim = sizes.size();
    if (ndim == 0) {
        return {};
    }
    std::vector<bool> result(ndim, false);

    std::vector<int64_t> contiguousStride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        contiguousStride[i] = contiguousStride[i + 1] * sizes[i + 1];
    }

    for (int64_t i = 0; i < ndim; i++) {
        result[i] = (strides[i] == contiguousStride[i]);
    }
    return result;
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
    (void)outputDevice;

    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());

    // pool_key 0轴非连续支持(参考 compressor 方案): eager/aclnn 链路的
    // tiling context 不上报运行时 stride, 故在 torch extension 层直读
    // at::Tensor::stride(0) 作为 key_stride0 属性下传; 非 0 轴必须连续。
    // PA_BBND 下 0 轴 stride >= blockSize*N2*D 时按非连续寻址, 否则等价连续。
    int64_t keyStride0 = -1;
    if (!poolKey.is_contiguous()) {
        auto contiguousAxes = IsContiguousAxes(poolKey);
        bool isPaBbnd = (keyLayoutStr == "PA_BBND");
        for (int64_t i = 1; i < static_cast<int64_t>(contiguousAxes.size()); i++) {
            TORCH_CHECK(contiguousAxes[i],
                        "pool_key only supports non-contiguous tensor on the 0-axis, axis[",
                        i, "] stride is not contiguous");
        }
        keyStride0 = poolKey.stride(0);
        if (isPaBbnd) {
            int64_t contiguousStride0 = poolKey.size(1) * poolKey.size(2) * poolKey.size(3);
            TORCH_CHECK(keyStride0 >= contiguousStride0,
                        "pool_key stride0(", keyStride0,
                        ") must be >= contiguous stride(", contiguousStride0, ") in PA_BBND scenarios");
        }
    }

    // pool_tail_k / actual_seq_q / actual_seq_k are ValueDepend(OPTIONAL) in the op
    // definition, so the auto-generated aclnn API expects aclIntArray* (not aclTensor*).
    // Convert 1-D int64 tensors to at::IntArrayRef so ConvertType produces aclIntArray*.
    at::Tensor poolTailKCpu = poolTailK.to(at::kLong).cpu().contiguous();
    at::IntArrayRef poolTailKArr(poolTailKCpu.data_ptr<int64_t>(), poolTailKCpu.numel());

    at::Tensor actualSeqQCpu;
    c10::optional<at::IntArrayRef> actualSeqQArr = c10::nullopt;
    if (actualSeqQ.has_value() && actualSeqQ.value().defined()) {
        actualSeqQCpu = actualSeqQ.value().to(at::kLong).cpu().contiguous();
        actualSeqQArr = at::IntArrayRef(actualSeqQCpu.data_ptr<int64_t>(), actualSeqQCpu.numel());
    }

    at::Tensor actualSeqKCpu;
    c10::optional<at::IntArrayRef> actualSeqKArr = c10::nullopt;
    if (actualSeqK.has_value() && actualSeqK.value().defined()) {
        actualSeqKCpu = actualSeqK.value().to(at::kLong).cpu().contiguous();
        actualSeqKArr = at::IntArrayRef(actualSeqKCpu.data_ptr<int64_t>(), actualSeqKCpu.numel());
    }

    ACLNN_CMD(aclnnPoolKeyIndexer,
              query, poolKey, weights, poolTailKArr,
              actualSeqQArr, actualSeqKArr, blockTable,
              qDescale, kDescale,
              topk, poolSize, queryLayoutPtr, keyLayoutPtr,
              maskMode, quantMode, returnValue, keyStride0,
              sparseIndicesOut, sparseValuesOut);

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pool_key_indexer", &PoolKeyIndexer, "pool_key_indexer");
}
} // namespace op_api
