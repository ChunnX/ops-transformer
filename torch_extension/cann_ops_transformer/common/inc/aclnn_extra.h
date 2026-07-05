/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_extra.h
 * \brief Standalone utilities not required by the core ACLNN_CMD macro.
 *        Include this header only when you directly use QuantMode,
 *        StorageShapeTensor, get_valid_tensor, AddParamToBuf or CalcHashId.
 *
 *        Requires "aclnn_common.h" to be included before this header.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_EXTRA_H
#define CANN_OPS_TRANSFORMER_ACLNN_EXTRA_H

enum QuantMode {
    QUANT_MODE_NO_QUANT = 0,
    QUANT_MODE_STATIC = 1,
    QUANT_MODE_PERTOKEN = 2,
    QUANT_MODE_PERGROUP = 3,
    QUANT_MODE_MX = 4,
};

typedef struct {
    const at::Tensor &tensor_;
} StorageShapeTensor;

inline aclTensor *ConvertType(const StorageShapeTensor &storage_shape_tensor)
{
    const at::Tensor &at_tensor = storage_shape_tensor.tensor_;
    if (!at_tensor.defined()) {
        return nullptr;
    }
    aclDataType acl_data_type = ConvertToAclDataType(at_tensor.scalar_type());
    if (!at_tensor.is_contiguous() || !IsOpInputBaseFormat(at_tensor) || Is4BitDtype(acl_data_type)) {
        return ConvertType(at_tensor);
    }
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
        std::string(c10::toString(at_tensor.scalar_type())) + " has not been supported")
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector(at_tensor.sizes());
    std::vector<int64_t> storageDims(at_tensor.sizes().begin(), at_tensor.sizes().end());
    aclFormat format = ACL_FORMAT_ND;
    switch (at_tensor.sizes().size()) {
        case 3:
            format = ACL_FORMAT_NCL;
            break;
        case 4:
            format = ACL_FORMAT_NCHW;
            break;
        case 5:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
    }
    return aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
                           at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                           const_cast<void *>(at_tensor.storage().data()));
}

inline const c10::optional<at::Tensor> get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device)
{
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

template <std::size_t N>
void AddParamToBuf(const std::array<bool, N> &value)
{
    MEMCPY_TO_BUF(value.data(), value.size() * sizeof(bool));
}

template <typename T>
void AddParamToBuf(const T &value)
{
    MEMCPY_TO_BUF(&value, sizeof(T));
}

void AddParamToBuf(const at::Tensor &);
void AddParamToBuf(const at::Scalar &);
void AddParamToBuf(const at::IntArrayRef &);
void AddParamToBuf(const at::ArrayRef<bool> &);
void AddParamToBuf(const at::TensorList &);
void AddParamToBuf(const c10::optional<at::Tensor> &);
void AddParamToBuf(const c10::optional<at::IntArrayRef> &);
void AddParamToBuf(const c10::optional<at::Scalar> &);
void AddParamToBuf(const at::ScalarType);
void AddParamToBuf(const string &);
void AddParamToBuf();

template <typename T, typename... Args>
void AddParamToBuf(const T &arg, Args &...args)
{
    AddParamToBuf(arg);
    AddParamToBuf(args...);
}

uint64_t CalcHashId();

#endif  // CANN_OPS_TRANSFORMER_ACLNN_EXTRA_H
