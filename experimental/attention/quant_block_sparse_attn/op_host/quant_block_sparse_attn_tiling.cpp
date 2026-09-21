/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file quant_block_sparse_attn_tiling.cpp
 *  \brief Architecture and quantization-mode routing for QuantBlockSparseAttn.
 */

#include <limits>
#include <memory>
#include <string>

#include <tiling/platform/platform_ascendc.h>
#include "arch35/qbsa_fp8/qbsa_fp8_tiling.h"
#include "arch35/qbsa_mxfp8/qbsa_mxfp8_tiling.h"
#include "log/log.h"
#include "quant_block_sparse_attn_tiling.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";

ge::graphStatus GetQuantMode(const gert::RuntimeAttrs *attrs, uint32_t &quantMode)
{
    if (attrs == nullptr) {
        OP_LOGE(kOpName, "Tiling attrs are nullptr");
        return ge::GRAPH_FAILED;
    }
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(QBSA_QUANT_MODE_ATTR_INDEX);
    const int64_t rawQuantMode = quantModePtr == nullptr ? static_cast<int64_t>(QBSA_QUANT_MODE_FP8) : *quantModePtr;
    if (rawQuantMode < 0 || rawQuantMode > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "quant_mode", std::to_string(rawQuantMode),
                                              "Must be representable as uint32_t");
        return ge::GRAPH_FAILED;
    }
    quantMode = static_cast<uint32_t>(rawQuantMode);
    return ge::GRAPH_SUCCESS;
}

std::unique_ptr<QbsaTilingStrategy> CreateArch35Strategy(gert::TilingContext *context, uint32_t quantMode)
{
    switch (quantMode) {
        case QBSA_QUANT_MODE_FP8:
            return std::unique_ptr<QbsaTilingStrategy>(new QbsaFp8Arch35Tiling(context));
        case QBSA_QUANT_MODE_MXFP8_FULL_QUANT:
            return std::unique_ptr<QbsaTilingStrategy>(new QbsaMxfp8Arch35Tiling(context));
        default:
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "quant_mode", std::to_string(quantMode),
                                                  "No arch35 strategy is implemented for this quant_mode");
            return nullptr;
    }
}

std::unique_ptr<QbsaTilingStrategy> CreateStrategy(gert::TilingContext *context, uint32_t quantMode)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(kOpName, "Platform info is nullptr");
        return nullptr;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const NpuArch npuArch = platform.GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_3510:
            return CreateArch35Strategy(context, quantMode);
        default:
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "npu_arch", std::to_string(static_cast<int32_t>(npuArch)),
                                                  "QuantBlockSparseAttn currently supports DAV_3510 only");
            return nullptr;
    }
}
} // namespace

ge::graphStatus TilingQuantBlockSparseAttn(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    uint32_t quantMode = QBSA_QUANT_MODE_FP8;
    if (GetQuantMode(context->GetAttrs(), quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto strategy = CreateStrategy(context, quantMode);
    return strategy == nullptr ? ge::GRAPH_FAILED : strategy->Run();
}
} // namespace optiling

namespace ge {
graphStatus TilingPrepareQuantBlockSparseAttn(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        OP_LOGE("QuantBlockSparseAttn", "TilingPrepare context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class QuantBlockSparseAttnTiling {
public:
    static ge::graphStatus Tiling(gert::TilingContext *context)
    {
        return optiling::TilingQuantBlockSparseAttn(context);
    }
};
} // namespace ops

namespace optiling {
IMPL_OP_OPTILING(QuantBlockSparseAttn).Tiling(TilingQuantBlockSparseAttn);
} // namespace optiling
