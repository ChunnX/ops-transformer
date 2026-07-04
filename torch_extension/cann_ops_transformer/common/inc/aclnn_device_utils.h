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
 * \file aclnn_device_utils.h
 * \brief Device detection helpers: is_at_tensor, is_at_tensor_list, GetFirstTensor, DecodeDevice.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_DEVICE_UTILS_H
#define CANN_OPS_TRANSFORMER_ACLNN_DEVICE_UTILS_H

#include <type_traits>
#include <ATen/Tensor.h>

template<typename T>
struct is_at_tensor : std::false_type {};

template<>
struct is_at_tensor<at::Tensor> : std::true_type {};

template<typename T>
struct is_at_tensor_list : std::false_type {};

template<>
struct is_at_tensor_list<at::TensorList> : std::true_type {};

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

template <typename... Ts>
auto DecodeDevice(Ts&... args) -> at::Device
{
    auto tp = std::make_tuple(args...);
    at::Tensor ft;
    GetFirstTensor(tp, ft);
    return ft.device();
}

#endif  // CANN_OPS_TRANSFORMER_ACLNN_DEVICE_UTILS_H
