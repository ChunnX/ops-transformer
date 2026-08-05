/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mixed_quant_variant_checker.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "MixedQuantSparseFlashMla" : context.opName;
}
} // namespace

ge::graphStatus MixedQuantVariantChecker::CheckSinglePara(const CheckContext &context) const
{
    OP_CHECK_IF(context.quantMode != 1 && context.quantMode != 2,
                OP_LOGE(Op(context), "quant_mode only supports 1 and 2."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.ropeHeadDim != 64,
                OP_LOGE(Op(context), "rope_head_dim only supports 64, but got %ld.", context.ropeHeadDim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
