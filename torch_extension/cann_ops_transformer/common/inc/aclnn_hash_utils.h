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
 * \file aclnn_hash_utils.h
 * \brief Hash buffer, parameter serialization (AddParamToBuf), and CalcHashId.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_HASH_UTILS_H
#define CANN_OPS_TRANSFORMER_ACLNN_HASH_UTILS_H

#include <cstring>
#include <string>
#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#include "aclnn_common_types.h"

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;
extern thread_local char g_hashBuf[kHashBufSize];
extern thread_local int g_hashOffset;

#define MEMCPY_TO_BUF(data_expression, size_expression)                                    \
    if (g_hashOffset + (size_expression) > kHashBufSize) {                                 \
        g_hashOffset = kHashBufMaxSize;                                                    \
        return;                                                                            \
    }                                                                                      \
    auto ret = memcpy_s(g_hashBuf + g_hashOffset, size_expression, data_expression, size_expression); \
    TORCH_CHECK(ret == 0, "memcpy_s failed, error:", ret);                               \
    g_hashOffset += size_expression;

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
void AddParamToBuf(const std::string &);
void AddParamToBuf();

template <typename T, typename... Args>
void AddParamToBuf(const T &arg, Args &...args)
{
    AddParamToBuf(arg);
    AddParamToBuf(args...);
}

uint64_t CalcHashId();

#endif  // CANN_OPS_TRANSFORMER_ACLNN_HASH_UTILS_H
