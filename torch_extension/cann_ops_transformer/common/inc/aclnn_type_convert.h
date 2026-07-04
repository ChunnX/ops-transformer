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
 * \file aclnn_type_convert.h
 * \brief Type conversion functions (at::Tensor → aclTensor*, etc.) and related helpers.
 *        This is the heaviest sub-header due to torch_npu format/device utilities.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_TYPE_CONVERT_H
#define CANN_OPS_TRANSFORMER_ACLNN_TYPE_CONVERT_H

#include <functional>
#include <unordered_map>
#include <acl/acl_rt.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/Device.h>
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

#include "aclnn_common_types.h"
#include "aclnn_opapi_loader.h"

inline bool Is4BitDtype(const aclDataType acl_data_type)
{
    return acl_data_type == ACL_FLOAT4_E2M1 || acl_data_type == ACL_FLOAT4_E1M2 || acl_data_type == ACL_INT4;
}

static std::unordered_map<aclFormat, aclFormat> FORMAT_FAKE_TO_REAL {
    { ACL_FORMAT_FRACTAL_NZ_C0_16, ACL_FORMAT_FRACTAL_NZ_C0_32 },
    { ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ }
};

static inline bool IsOpInputBaseFormat(const at::Tensor &at_tensor)
{
    if (!torch_npu::utils::is_npu(at_tensor)) {
        return true;
    }
    const auto format = static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
    return (format == ACL_FORMAT_ND) || (format == ACL_FORMAT_NCHW) ||
           (format == ACL_FORMAT_NHWC) || (format == ACL_FORMAT_NCDHW);
}

inline void CollectB4ShapeInfo(const at::Tensor &at_tensor,
                               c10::SmallVector<int64_t, MAX_DIM_NUM>& wrapperStride,
                               c10::SmallVector<int64_t, MAX_DIM_NUM>& wrapperShape)
{
    int64_t nDim = at_tensor.sizes().size();
    if (nDim == 1) {
        wrapperShape[0] = wrapperShape[0] * FP4_IN_INT8;
    } else if (nDim > 1) {
        if (wrapperStride[nDim - 1] == 1 && wrapperStride[nDim - PENULTIMATE_DIM] == 1) {
            if (wrapperShape[nDim - PENULTIMATE_DIM] == 1) {
                wrapperStride[nDim - 1] = wrapperStride[nDim - 1] * FP4_IN_INT8;
                wrapperShape[nDim - PENULTIMATE_DIM] = wrapperShape[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
            } else if (wrapperShape[nDim - 1] == 1) {
                wrapperStride[nDim - PENULTIMATE_DIM] = wrapperStride[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
                wrapperShape[nDim - 1] = wrapperShape[nDim - 1] * FP4_IN_INT8;
            }
        } else if (wrapperStride[nDim - 1] == 1) {
            wrapperStride[nDim - PENULTIMATE_DIM] =
                wrapperStride[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
            wrapperShape[nDim - 1] = wrapperShape[nDim - 1] * FP4_IN_INT8;
        } else if (wrapperStride[nDim - PENULTIMATE_DIM] == 1) {
            wrapperStride[nDim - 1] =
                wrapperStride[nDim - 1] * FP4_IN_INT8;
            wrapperShape[nDim - PENULTIMATE_DIM] =
                wrapperShape[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
        }

        for (auto i = 0; i < nDim - PENULTIMATE_DIM; i++) {
            wrapperStride[i] = wrapperStride[i] * FP4_IN_INT8;
        }
    } else {
        TORCH_CHECK(false, "unsupported tensor size() in 4-bit dtype.");
    }
}

aclDataType ConvertToAclDataType(const at::ScalarType &data_type)
{
    int64_t dtype_index = static_cast<int64_t>(data_type);
    TORCH_CHECK(dtype_index >= 0 && dtype_index < static_cast<int64_t>(at::ScalarType::NumOptions) + 1,
                "data_type enum value (",
                dtype_index,
                ") is out of range: [0, ",
                static_cast<int64_t>(at::ScalarType::NumOptions),
                "]")
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[dtype_index];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
                std::string(c10::toString(data_type)) + " has not been supported")
    return acl_dtype;
}

inline aclDataType GetAclDataType(int64_t t)
{
    if (t >= g_toAclOffset) {
        return static_cast<aclDataType>(t - g_toAclOffset);
    }
    return ConvertToAclDataType(
        static_cast<at::ScalarType>(t));
}

inline c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor)
{
    c10::Scalar expScalar;
    const at::Tensor *aclInput = &tensor;
    if (aclInput->scalar_type() == at::ScalarType::Double) {
        double value = *(double *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Long) {
        int64_t value = *(int64_t *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Float) {
        float value = *(float *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Int) {
        int value = *(int *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Half) {
        c10::Half value = *(c10::Half *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Bool) {
        int8_t value = *(int8_t *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::ComplexDouble) {
        c10::complex<double> value = *(c10::complex<double> *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::ComplexFloat) {
        c10::complex<float> value = *(c10::complex<float> *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::BFloat16) {
        c10::BFloat16 value = *(c10::BFloat16 *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else {
        ASCEND_LOGE("unsupported scalar type! ");
    }
    return expScalar;
}

inline at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor)
{
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    return cpuPinMemTensor.to(
        c10::Device(torch_npu::utils::get_npu_device_type(), deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type)
{
    return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

inline aclTensor *ConvertType(const at::Tensor &at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (!at_tensor.defined()) {
        return nullptr;
    }
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = ConvertToAclDataType(scalar_data_type);
    TORCH_CHECK(
        acl_data_type != ACL_DT_UNDEFINED, std::string(c10::toString(scalar_data_type)) + " has not been supported")
    std::vector<int64_t> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector(at_tensor.sizes());

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    const bool isBaseFormat = IsOpInputBaseFormat(at_tensor);
    if (!isBaseFormat) {
        format = static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            storageDims = at_npu::native::get_npu_storage_sizes(at_tensor);
        }
    } else {
        switch (dimNum) {
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
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
        }
    }

    if (acl_data_type != ACL_STRING && Is4BitDtype(acl_data_type)) {
        CollectB4ShapeInfo(at_tensor, wrapperStride, wrapperShape);
        storageDims.back() *= FP4_IN_INT8;
        if (!isBaseFormat) {
            auto realFormat = FORMAT_FAKE_TO_REAL.find(format);
            TORCH_CHECK(realFormat != FORMAT_FAKE_TO_REAL.end(), "not support convert ", format, ".");
            format = realFormat->second;
        }
    }

    if (at_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        c10::Scalar expScalar = ConvertTensorToScalar(at_tensor);
        at::Tensor aclInput = CopyScalarToDevice(expScalar, scalar_data_type);
        return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), acl_data_type,
                               aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(),
                               storageDims.size(), const_cast<void *>(aclInput.storage().data()));
    }

    auto acl_tensor =
        aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

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

inline aclScalar *ConvertType(const at::Scalar &at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = ConvertToAclDataType(scalar_data_type);
    TORCH_CHECK(
        acl_data_type != ACL_DT_UNDEFINED, std::string(c10::toString(scalar_data_type)) + " has not been supported")
    aclScalar *acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double: {
                double value = at_scalar.toDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Long: {
                int64_t value = at_scalar.toLong();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Bool: {
                bool value = at_scalar.toBool();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::ComplexDouble: {
                auto value = at_scalar.toComplexDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        default:
            acl_scalar = nullptr;
            break;
    }
    return acl_scalar;
}

inline aclIntArray *ConvertType(const at::IntArrayRef &at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto array = aclCreateIntArray(at_array.data(), at_array.size());
    return array;
}

template <std::size_t N>
inline aclBoolArray *ConvertType(const std::array<bool, N> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclBoolArray *ConvertType(const at::ArrayRef<bool> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclTensorList *ConvertType(const at::TensorList &at_tensor_list)
{
    if (at_tensor_list.size() == 0) {
        return nullptr;
    }
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(at_tensor_list[i]);
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

inline aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return ConvertType(opt_tensor.value());
    }
    return nullptr;
}

inline aclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }
    return nullptr;
}

inline aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertType(opt_scalar.value());
    }
    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType)
{
    return ConvertToAclDataType(scalarType);
}

inline const char* ConvertType(const std::string& str)
{
    return str.c_str();
}

aclTensor *ConvertType(const TensorWrapper &tensor_r)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    const at::Tensor &at_tensor = tensor_r.tensor_;

    if (!at_tensor.defined()) {
        return nullptr;
    }

    aclDataType acl_data_type = tensor_r.dtype;
    std::vector<int64_t> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector(at_tensor.sizes());

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    const bool isBaseFormat = IsOpInputBaseFormat(at_tensor);
    if (!isBaseFormat) {
        format = static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            storageDims = at_npu::native::get_npu_storage_sizes(at_tensor);
        }
    } else {
        switch (dimNum) {
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
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
        }
    }

    if (acl_data_type != ACL_STRING && Is4BitDtype(acl_data_type)) {
        CollectB4ShapeInfo(at_tensor, wrapperStride, wrapperShape);
        storageDims.back() *= FP4_IN_INT8;
        if (!isBaseFormat) {
            auto realFormat = FORMAT_FAKE_TO_REAL.find(format);
            TORCH_CHECK(realFormat != FORMAT_FAKE_TO_REAL.end(), "not support convert ", format, ".");
            format = realFormat->second;
        }
    }

    auto acl_tensor =
        aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

aclTensorList *ConvertType(const TensorListWrapper &tensor_list_wrapper)
{
    if (tensor_list_wrapper.tensor_list_.size() == 0) {
        return nullptr;
    }
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(tensor_list_wrapper.tensor_list_.size());
    for (size_t i = 0; i < tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(TensorWrapper{
            tensor_list_wrapper.tensor_list_[i], tensor_list_wrapper.dtype});
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

#endif  // CANN_OPS_TRANSFORMER_ACLNN_TYPE_CONVERT_H
