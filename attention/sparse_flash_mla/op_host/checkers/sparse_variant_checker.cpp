/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_variant_checker.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseFlashMla" : context.opName;
}
} // namespace

ge::graphStatus SparseVariantChecker::CheckFeature(const CheckContext &context) const
{
    if (context.isAscend950) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t expectedRatio = 1;
    const char *mode = "SWA";
    if (context.cmpKv.present && context.cmpSparseIndices.present) {
        expectedRatio = 4;
        mode = "CSA";
    } else if (context.cmpKv.present) {
        expectedRatio = 128;
        mode = "HCA";
    }
    OP_CHECK_IF(context.cmpRatio != expectedRatio,
                OP_LOGE(Op(context), "On Atlas A2/A3, %s requires cmp_ratio=%ld.", mode, expectedRatio),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
