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
#include <c10/core/DeviceGuard.h>
#include <acl/acl_base.h>

#include <memory>

// Exercise production Torch output allocation shapes and converters on CPU. Only the ACL
// descriptor/runtime boundary is replaced; no NPU kernel is executed by this UT.
#define CANN_OPS_TRANSFORMER_ACLNN_COMMON_H
struct aclTensor {
    std::vector<int64_t> view;
    std::vector<int64_t> strides;
    std::vector<int64_t> storage;
    int64_t offset;
    aclFormat format;
    void *data;
};
struct aclTensorList {
    std::vector<const aclTensor *> tensors;
    ~aclTensorList()
    {
        for (auto tensor : tensors) {
            delete tensor;
        }
    }
};

aclTensor *aclCreateTensor(const int64_t *shape, uint64_t rank, aclDataType, const int64_t *strides, int64_t offset,
                           aclFormat format, const int64_t *storage, uint64_t storageRank, void *data)
{
    return new aclTensor{
        {shape, shape + rank}, {strides, strides + rank}, {storage, storage + storageRank}, offset, format, data};
}
aclTensorList *aclCreateTensorList(const aclTensor *const *tensors, uint64_t count)
{
    return new aclTensorList{{tensors, tensors + count}};
}
void Release(const aclTensor *tensor)
{
    delete tensor;
}
bool IsOpInputBaseFormat(const at::Tensor &tensor)
{
    return tensor.layout() == at::kStrided;
}
aclDataType ConvertToAclDataType(at::ScalarType type)
{
    switch (type) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kChar:
            return ACL_INT8;
        case at::kLong:
            return ACL_INT64;
        case at::kInt:
            return ACL_INT32;
        case at::kFloat:
            return ACL_FLOAT;
        default:
            return ACL_DT_UNDEFINED;
    }
}
#define GET_OP_API_FUNC(name) (&name)
void *GetOpApiFuncAddr(const char *)
{
    return nullptr;
}
#define ACLNN_CMD(name, ...) TORCH_CHECK(false, "Device execution is outside the CPU Torch UT")

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, m) static void BindProductionModule(pybind11::module_ &m)
#include "../../../torch_extension/csrc/flash_mla.cpp"
#pragma pop_macro("PYBIND11_MODULE")

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("output_shapes", &op_api::output_shapes);
    m.def("normalize_cache", &op_api::normalize_cache);
    m.def("check_sequence_descriptor", &op_api::check_sequence_descriptor);
    m.def("check_metadata_descriptor", &op_api::check_metadata_descriptor);
    m.def("describe", [](const at::Tensor &tensor) {
        std::unique_ptr<aclTensor> desc(op_api::ConvertType(op_api::NdTensor{tensor}));
        pybind11::dict result;
        result["view"] = desc->view;
        result["storage"] = desc->storage;
        result["strides"] = desc->strides;
        result["offset"] = desc->offset;
        result["format"] = static_cast<int>(desc->format);
        result["base"] = reinterpret_cast<uintptr_t>(desc->data);
        return result;
    });
}
