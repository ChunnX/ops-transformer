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
 * \file compressor.cpp
 * \brief Compressor operator implementation for PyTorch NPU extension
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;
const int64_t VALUE_0 = 0;
const int64_t VALUE_1 = 1;
const int64_t VALUE_2 = 2;
const int64_t VALUE_128 = 128;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ConstructCompressorGradOutputTensor(
    const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &softmaxScore,
    int64_t cmpRatio, int64_t coff)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto xDim = x.dim();
    at::SmallVector<int64_t, MAX_DIM_SIZE> dXSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> dWkvSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> dWgateSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> dApeSize;
    at::Tensor dX;
    at::Tensor dWkv;
    at::Tensor dWgate;
    at::Tensor dApe;

    int64_t t = 0;
    int64_t h = 0;
    int64_t coffD = 0;

    int64_t batchSize = 0;
    int64_t seqSize = 0;

    TORCH_CHECK(wkv.defined(), "Check wkv != nullptr failed");
    auto wkvDim = wkv.dim();
    TORCH_CHECK(wkvDim == DIM_TWO, "wkv dim num[", wkvDim, "] should be 2");
    TORCH_CHECK(softmaxScore.defined(), "Check softmaxScore != nullptr failed");

    if (xDim == DIM_TWO) {
        t = x.size(0);
        h = x.size(1);
        coffD = wkv.size(0);
        dXSize = {t, h};
        dWkvSize = {coffD, h};
        dWgateSize = {coffD, h};
        dApeSize = {cmpRatio, coffD};
    } else {
        batchSize = x.size(0);
        seqSize = x.size(1);
        h = x.size(2);
        coffD = wkv.size(0);
        dXSize = {batchSize, seqSize, h};
        dWkvSize = {coffD, h};
        dWgateSize = {coffD, h};
        dApeSize = {cmpRatio, coffD};
    }

    dX = at::empty(dXSize, x.options().dtype(x.dtype()));
    dWkv = at::empty(dWkvSize, x.options().dtype(x.dtype()));
    dWgate = at::empty(dWgateSize, x.options().dtype(x.dtype()));
    dApe = at::empty(dApeSize, softmaxScore.options().dtype(softmaxScore.dtype()));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(dX, dWkv, dWgate, dApe);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> CompressorGrad(
    const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
    const at::Tensor &dCmpKv, const at::Tensor &softmaxScore, const at::Tensor &kv, int64_t cmpRatio, int64_t coff,
    const c10::optional<at::Tensor> &cuSeqlens, const c10::optional<at::Tensor> &seqused,
    const c10::optional<at::Tensor> &startPos)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto xDim = x.dim();
    TORCH_CHECK(xDim == DIM_TWO || xDim == DIM_THREE, "x dim num[", xDim, "] should be 2 or 3");

    TORCH_CHECK(cmpRatio >= VALUE_2 && cmpRatio <= VALUE_128, "cmp_ratio should be in range 2~128");

    TORCH_CHECK(coff == VALUE_1 || coff == VALUE_2, "coff value[", coff, "] should be 1 or 2");

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> compressorGradResult =
        ConstructCompressorGradOutputTensor(x, wkv, softmaxScore, cmpRatio, coff);
    
    at::Tensor dX = std::get<0>(compressorGradResult);
    at::Tensor dWkv = std::get<1>(compressorGradResult);
    at::Tensor dWgate = std::get<2>(compressorGradResult);
    at::Tensor dApe = std::get<3>(compressorGradResult);

    ACLNN_CMD(aclnnCompressorGrad, x, wkv, wgate, dCmpKv, softmaxScore,
              kv, cuSeqlens, seqused, startPos, cmpRatio, coff, dX, dWkv, dWgate, dApe);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(dX, dWkv, dWgate, dApe);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compressor_grad", &CompressorGrad, "compressor_grad");
}
} // namespace op_api
