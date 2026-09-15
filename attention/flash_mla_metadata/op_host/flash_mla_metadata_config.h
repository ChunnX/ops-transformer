/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_METADATA_CONFIG_H
#define FLASH_MLA_METADATA_CONFIG_H

#include "../../flash_mla/op_host/flash_mla_shape.h"
#include "../../flash_mla/op_kernel/flash_mla_schedule_config.h"

namespace FlashMla {
// The internal ACLNN ABI uses a signed sentinel for omitted Python dimensions.
// Resolve it before storing dimensions in the unsigned scheduling config.
constexpr int64_t kMlaInferDimension = -1;

// blockSize must already be populated. Output-layout suffixes have no effect
// on value scheduling. FP16 and BF16 share the same nonquant scheduling path.
inline bool ConfigureMlaMetadata(const char *inputLayer, int64_t quantMode, MlaScheduleConfig &config)
{
    std::string queryLayout, outputLayout;
    if ((quantMode != kMlaNoQuant && quantMode != kMlaInt8FullQuant) ||
        !SplitMlaLayout(inputLayer, queryLayout, outputLayout) ||
        !ParseMlaLayout(queryLayout.c_str(), config.layoutQ)) {
        return false;
    }
    config.queryDataType = quantMode == kMlaInt8FullQuant ? kMlaInt8 : kMlaFp16;
    config.layoutKv = MlaScheduleKvLayout(config);
    return true;
}
} // namespace FlashMla
#endif // FLASH_MLA_METADATA_CONFIG_H
