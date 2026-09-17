/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_matmul_aiv_mode_common.h"

#include "mc2_log.h"
#include "mc2_tiling_utils.h"
#include "platform/platform_ascendc.h"

namespace mc2tiling {

bool MknRange::Contains(int32_t m, int32_t k, int32_t n) const
{
    return m > mStart && m <= mEnd && k > kStart && k <= kEnd && n > nStart && n <= nEnd;
}

int32_t GetValueFromMknConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue,
                                    const MknConditionMap &conditionMap)
{
    for (const auto &item : conditionMap) {
        for (const auto &range : item.second) {
            if (range.Contains(m, k, n)) {
                return item.first;
            }
        }
    }
    return defaultValue;
}

AivPlatformInfo GetAivPlatformInfo(const gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    AivPlatformInfo info{platform.GetCoreNumAiv(), 0U};
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    return info;
}

AivScaleDtypeResult CheckAivMatmulScaleDtype(const gert::Tensor *scale, ge::DataType outputType)
{
    if (scale == nullptr) {
        return {false, false};
    }
    const auto scaleType = scale->GetDataType();
    const bool isInt64 = outputType == ge::DT_FLOAT16 && scaleType == ge::DT_INT64;
    return {scaleType == ge::DT_FLOAT || isInt64, isInt64};
}

ge::graphStatus ValidateCclBufferSizeIfAvailable(const char *group, const char *opName, uint64_t minBufferSize)
{
    uint64_t cclBufferSize = 0ULL;
    if (GetCclBufferSize(group, &cclBufferSize, opName) != ge::GRAPH_SUCCESS) {
        OP_LOGW(opName, "Can't get HCCL_BUFFSIZE, skip CCL buffer size validation.");
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(cclBufferSize < minBufferSize,
                    OP_LOGE(opName, "HCCL_BUFFSIZE (%lu Bytes) too small, min required %lu Bytes", cclBufferSize,
                            minBufferSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace mc2tiling
