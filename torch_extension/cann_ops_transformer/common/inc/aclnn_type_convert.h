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
 * \brief Type conversion functions (ConvertType series), scalar/tensor copy utilities,
 *        Release functions, and related helpers for ACLNN operator wrappers.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_TYPE_CONVERT_H
#define CANN_OPS_TRANSFORMER_ACLNN_TYPE_CONVERT_H

#include "aclnn_api_loader.h"

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

inline const char* ConvertType(const string& str)
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

template <typename T>
T ConvertType(T value)
{
    return value;
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr, std::index_sequence<I...>)
{
    using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

inline void Release(aclTensor *p)
{
    static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
    if (aclDestroyTensor == nullptr) {
        return;
    }
    aclDestroyTensor(p);
}

inline void Release(aclScalar *p)
{
    static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
    if (aclDestroyScalar == nullptr) {
        return;
    }
    aclDestroyScalar(p);
}

inline void Release(aclIntArray *p)
{
    static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
    if (aclDestroyIntArray == nullptr) {
        return;
    }

    aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p)
{
    static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
    if (aclDestroyBoolArray == nullptr) {
        return;
    }

    aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p)
{
    static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
    if (aclDestroyTensorList == nullptr) {
        return;
    }

    aclDestroyTensorList(p);
}

inline const c10::optional<at::Tensor> get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device)
{
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

template <typename T>
void Release(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple &t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

/**
 * check arg is at::Tensor ?
 */
template<typename T>
struct is_at_tensor : std::false_type {};

template<>
struct is_at_tensor<at::Tensor> : std::true_type {};

/**
 * check arg is at::TensorList ?
 */
template<typename T>
struct is_at_tensor_list : std::false_type {};

template<>
struct is_at_tensor_list<at::TensorList> : std::true_type {};

/**
 * find first at::Tensor
 */
template <std::size_t I = 0, typename...Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type GetFirstTensor(const std::tuple<Ts...>& t, at::Tensor& res) {}

template <std::size_t I = 0, typename... Ts>
    typename std::enable_if < I<sizeof...(Ts), void>::type GetFirstTensor(const std::tuple<Ts...> &t, at::Tensor &res)
{
    if constexpr (is_at_tensor<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t);
        return;
    } else if constexpr (is_at_tensor_list<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t)[0];
        return;
    }
    return GetFirstTensor<I + 1, Ts...>(t, res);
}

/**
 * get the device
 */
template <typename... Ts>
auto DecodeDevice(Ts&... args) -> at::Device
{
    auto tp = std::make_tuple(args...);
    at::Tensor ft;
    GetFirstTensor(tp, ft);
    return ft.device();
}

#endif  // CANN_OPS_TRANSFORMER_ACLNN_TYPE_CONVERT_H