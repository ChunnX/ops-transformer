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
 * \file mla_preprocess.cpp
 * \brief torch extension adaptation for aclnnMlaPreprocess
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

// Wrapper to pass a tensor with explicit FRACTAL_NZ format to ACLNN_CMD.
// The ConvertType overload below is found via ADL when ConvertTypes expands.
struct NzTensor {
    const at::Tensor &tensor;
};

// Custom ConvertType for NZ-format tensors: uses ACL_FORMAT_FRACTAL_NZ
// instead of the dim-based format detection in aclnn_common.h.
inline aclTensor *ConvertType(const NzTensor &nz)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }
    const at::Tensor &at_tensor = nz.tensor;
    if (!at_tensor.defined()) {
        return nullptr;
    }
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(
        acl_data_type != ACL_DT_UNDEFINED,
        std::string(c10::toString(scalar_data_type)) + " has not been supported")

    c10::SmallVector<int64_t, 5> storageDims;
    if (at_tensor.itemsize() == 0) {
        AT_ERROR("When ConvertType, tensor item size of cannot be zero.");
        return nullptr;
    }
    if (acl_data_type != ACL_STRING) {
        storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
    }

    auto acl_tensor = aclCreateTensor(
        at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type,
        at_tensor.strides().data(), at_tensor.storage_offset(),
        ACL_FORMAT_FRACTAL_NZ,
        storageDims.data(), storageDims.size(),
        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_mla_preprocess(
    const at::Tensor &input,
    const at::Tensor &gamma0,
    const at::Tensor &beta0,
    const at::Tensor &quantScale0,
    const at::Tensor &quantOffset0,
    const at::Tensor &wdqkv,
    const at::Tensor &deScale0,
    const at::Tensor &bias0,
    const at::Tensor &gamma1,
    const at::Tensor &beta1,
    const at::Tensor &quantScale1,
    const at::Tensor &quantOffset1,
    const at::Tensor &wuq,
    const at::Tensor &deScale1,
    const at::Tensor &bias1,
    const at::Tensor &gamma2,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const at::Tensor &wuk,
    const at::Tensor &kvCache,
    const at::Tensor &slotMapping,
    const c10::optional<at::Tensor> &kvCacheRope,
    const c10::optional<at::Tensor> &ctkvScale,
    const c10::optional<at::Tensor> &qNopeScale,
    int64_t wdqDim,
    int64_t qRopeDim,
    int64_t kRopeDim,
    double epsilon,
    int64_t qRotaryCoeff,
    int64_t kRotaryCoeff,
    bool transposeWdq,
    bool transposeWuq,
    bool transposeWuk,
    int64_t cacheMode,
    int64_t quantMode,
    bool doRmsNorm,
    int64_t wdkvSplitCount)
{
    int64_t tokenNum = input.size(0);
    int64_t headNum = wuk.size(0);

    int64_t qOutLastDim = (cacheMode == 0) ? 576 : 512;
    at::Tensor qOut = at::empty({tokenNum, headNum, qOutLastDim},
        (cacheMode == 2) ? input.options().dtype(at::kChar) : input.options());

    at::Tensor qRopeOut;
    if (cacheMode == 0) {
        qRopeOut = at::empty({0}, input.options());
    } else {
        qRopeOut = at::empty({tokenNum, headNum, 64}, input.options());
    }

    at::Tensor krCacheOut;
    if (cacheMode == 0) {
        krCacheOut = at::empty({0}, input.options());
    } else {
        // krCacheOut must have the same stride0 as kvCacheRope.
        // The kernel uses kv_cache_rope_stride0_ for both tensors.
        TORCH_CHECK(kvCacheRope.has_value() && kvCacheRope.value().defined(),
            "kvCacheRope is required for cacheMode > 0");
        auto ropeStrides = kvCacheRope.value().strides();
        int64_t krStride0 = ropeStrides[0];
        c10::SmallVector<int64_t, 5> krSizes;
        if (cacheMode == 1) {
            krSizes = {kvCache.size(0), kvCache.size(1), 1, 64};
        } else {
            krSizes = {kvCache.size(0), 4, kvCache.size(2), 16};
        }
        int64_t physSize = krSizes[0] * krStride0;
        auto krBuf = at::zeros({physSize}, input.options());
        krCacheOut = krBuf.as_strided(krSizes, ropeStrides);
    }

    NzTensor wdqkv_nz{wdqkv};
    NzTensor wuq_nz{wuq};

    if (cacheMode >= 2) {
        NzTensor kvCache_nz{kvCache};
        at::Tensor kcr_ref = (kvCacheRope.has_value() && kvCacheRope.value().defined())
                            ? kvCacheRope.value()
                            : at::Tensor();
        NzTensor kvCacheRope_nz{kcr_ref};
        NzTensor kvCacheOut_nz{kvCache};
        NzTensor krCacheOut_nz{krCacheOut};
        ACLNN_CMD(aclnnMlaPreprocess,
            input, gamma0, beta0, quantScale0, quantOffset0,
            wdqkv_nz, deScale0, bias0, gamma1, beta1, quantScale1, quantOffset1,
            wuq_nz, deScale1, bias1, gamma2, cos, sin, wuk,
            kvCache_nz, kvCacheRope_nz, slotMapping, ctkvScale, qNopeScale,
            wdqDim, qRopeDim, kRopeDim, epsilon,
            qRotaryCoeff, kRotaryCoeff,
            transposeWdq, transposeWuq, transposeWuk,
            cacheMode, quantMode, doRmsNorm, wdkvSplitCount,
            qOut, kvCacheOut_nz, qRopeOut, krCacheOut_nz);
    } else {
        ACLNN_CMD(aclnnMlaPreprocess,
            input, gamma0, beta0, quantScale0, quantOffset0,
            wdqkv_nz, deScale0, bias0, gamma1, beta1, quantScale1, quantOffset1,
            wuq_nz, deScale1, bias1, gamma2, cos, sin, wuk,
            kvCache, kvCacheRope, slotMapping, ctkvScale, qNopeScale,
            wdqDim, qRopeDim, kRopeDim, epsilon,
            qRotaryCoeff, kRotaryCoeff,
            transposeWdq, transposeWuq, transposeWuk,
            cacheMode, quantMode, doRmsNorm, wdkvSplitCount,
            qOut, kvCache, qRopeOut, krCacheOut);
    }

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        qOut, kvCache, qRopeOut, krCacheOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_mla_preprocess", &npu_mla_preprocess, "mla_preprocess");
}

} // namespace op_api
