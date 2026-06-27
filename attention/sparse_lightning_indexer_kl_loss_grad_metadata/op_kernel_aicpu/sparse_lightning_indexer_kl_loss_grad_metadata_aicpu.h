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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.h
 * \brief
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_H

#include <cstdint>
#include <string>
#include <type_traits>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {

enum class SliSparseMode : int32_t {
    NO_MASK = 0,
    RIGHT_DOWN_CAUSAL = 3,
};

enum class SliLayout {
    BSND = 0,
    TND = 1,
};

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, bool>::type GetAttrValue(CpuKernelContext &ctx,
                                                                               const std::string &name, T &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = static_cast<T>(attr->GetInt());
    return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name, std::string &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = attr->GetString();
    return true;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, void>::type GetAttrValueOpt(CpuKernelContext &ctx,
                                                                                  const std::string &name, T &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = static_cast<T>(attr->GetInt());
    }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name, std::string &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = attr->GetString();
    }
} // namespace aicpu
}

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_H
