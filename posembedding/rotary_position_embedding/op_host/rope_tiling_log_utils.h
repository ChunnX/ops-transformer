/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTARY_POSITION_EMBEDDING_ROPE_TILING_LOG_UTILS_H
#define ROTARY_POSITION_EMBEDDING_ROPE_TILING_LOG_UTILS_H

#include <string>

#include "log/log.h"

namespace optiling {
namespace rope_log {

inline std::string ShapeValue(const gert::Shape &shape)
{
    std::string value = Ops::Base::ToString(shape);
    if (value.size() >= 2 && value.front() == '[' && value.back() == ']') {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

} // namespace rope_log
} // namespace optiling

#endif // ROTARY_POSITION_EMBEDDING_ROPE_TILING_LOG_UTILS_H
