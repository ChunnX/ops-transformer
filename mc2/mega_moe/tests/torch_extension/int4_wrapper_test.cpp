// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <torch/extension.h>
#include "mega_moe_weight_utils.h"
#include <aclnn/acl_meta.h>
#include <memory>

pybind11::dict Describe(const at::Tensor &tensor, int64_t dtype)
{
    auto descriptor = ConvertType(TensorWrapper{tensor, static_cast<aclDataType>(dtype)});
    TORCH_CHECK(descriptor != nullptr, "Failed to create ACL tensor");
    std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)> guard(descriptor, aclDestroyTensor);
    pybind11::dict result;
    auto readDims = [&](auto query) {
        int64_t *values = nullptr;
        uint64_t count = 0;
        TORCH_CHECK(query(descriptor, &values, &count) == 0, "ACL descriptor query failed");
        std::unique_ptr<int64_t[]> valuesGuard(values);
        return std::vector<int64_t>(values, values + count);
    };
    result["shape"] = readDims(aclGetViewShape);
    result["strides"] = readDims(aclGetViewStrides);
    result["storage_shape"] = readDims(aclGetStorageShape);
    int64_t offset;
    aclFormat format;
    aclDataType actualDtype;
    void *addr = nullptr;
    TORCH_CHECK(aclGetViewOffset(descriptor, &offset) == 0, "offset query failed");
    TORCH_CHECK(aclGetFormat(descriptor, &format) == 0, "format query failed");
    TORCH_CHECK(aclGetDataType(descriptor, &actualDtype) == 0, "dtype query failed");
    TORCH_CHECK(aclGetRawTensorAddr(descriptor, &addr) == 0, "address query failed");
    result["offset"] = offset;
    result["format"] = static_cast<int>(format);
    result["dtype"] = static_cast<int>(actualDtype);
    result["address"] = reinterpret_cast<uintptr_t>(addr);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("describe", &Describe);
    m.def("resolve", [](const std::vector<at::Tensor> &weights, c10::optional<int64_t> dtype, bool arch22) {
        return static_cast<int>(op_api::ResolveMegaMoeWeightDtype(weights, dtype, arch22, "weight"));
    });
    m.def("check_shared", [](const std::vector<at::Tensor> &weights, int64_t dtype) {
        op_api::CheckMegaMoeInt4Weights(weights, static_cast<aclDataType>(dtype), "shared_weight");
    });
    m.attr("INT4") = static_cast<int>(ACL_INT4);
    m.attr("INT32") = static_cast<int>(ACL_INT32);
    m.attr("INT8") = static_cast<int>(ACL_INT8);
    m.attr("BF16") = static_cast<int>(ACL_BF16);
    m.attr("FP4") = static_cast<int>(ACL_FLOAT4_E2M1);
    m.attr("NZ") = static_cast<int>(ACL_FORMAT_FRACTAL_NZ);
    m.attr("DTYPE_INT4") = static_cast<int>(DType::INT4);
    m.attr("DTYPE_INT32") = static_cast<int>(DType::INT32);
    m.attr("DTYPE_FP4") = static_cast<int>(DType::FLOAT4_E2M1);
}
