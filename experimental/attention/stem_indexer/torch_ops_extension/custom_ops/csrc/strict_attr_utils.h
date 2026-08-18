/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STEM_INDEXER_STRICT_ATTR_UTILS_H
#define STEM_INDEXER_STRICT_ATTR_UTILS_H

#include <ATen/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

namespace custom {
inline bool ParseBoolAttr(const at::Scalar &value, const char *name)
{
    TORCH_CHECK(value.type() == at::ScalarType::Bool, name, " must be bool, but got ",
                c10::toString(value.type()), ".");
    return value.toBool();
}

inline int64_t ParseIntAttr(const at::Scalar &value, const char *name)
{
    TORCH_CHECK(value.type() == at::ScalarType::Long, name, " must be int, but got ",
                c10::toString(value.type()), ".");
    return value.toLong();
}

inline double ParseFloatAttr(const at::Scalar &value, const char *name)
{
    TORCH_CHECK(value.type() == at::ScalarType::Double, name, " must be float, but got ",
                c10::toString(value.type()), ".");
    return value.toDouble();
}
} // namespace custom

#endif // STEM_INDEXER_STRICT_ATTR_UTILS_H
