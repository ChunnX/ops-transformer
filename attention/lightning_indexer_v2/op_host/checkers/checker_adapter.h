/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEXER_V2_CHECKER_ADAPTER_H
#define INDEXER_V2_CHECKER_ADAPTER_H

#include "base_checker_indexer_v2.h"

namespace optiling {
namespace indexer_v2_checker {

inline CheckerTensor MakeRequiredTensor(gert::TilingContext *context, uint32_t index)
{
    if (context == nullptr) {
        return {};
    }
    return {context->GetInputDesc(index), context->GetInputShape(index), nullptr,
            context->GetDynamicInputStride(index, 0)};
}

inline CheckerTensor MakeOptionalTensor(gert::TilingContext *context, uint32_t index)
{
    if (context == nullptr) {
        return {};
    }
    return {context->GetOptionalInputDesc(index), context->GetOptionalInputShape(index),
            context->GetOptionalInputTensor(index), context->GetDynamicInputStride(index, 0)};
}

inline CheckerTensor MakeOutputTensor(gert::TilingContext *context, uint32_t index)
{
    if (context == nullptr) {
        return {};
    }
    return {context->GetOutputDesc(index), context->GetOutputShape(index), nullptr, nullptr};
}

} // namespace indexer_v2_checker
} // namespace optiling

#endif // INDEXER_V2_CHECKER_ADAPTER_H
