/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../flash_mla_value_tiling.h"
#include "flash_mla_value_tiling_common.h"
#include "flash_mla_value_tiling_nonquant.h"
#include "flash_mla_value_tiling_legacy.h"

namespace FlashMla {
MlaStatus ComputeMlaValueTiling(const MlaScheduleConfig &inputConfig, const MlaPlatformInfo &platform, MlaSeqView seqQ,
                                MlaSeqView seqKv, MlaValuePatch &patch)
{
    MlaScheduleConfig config = inputConfig;
    auto normalizeStatus = NormalizeMlaScheduleConfig(config);
    if (normalizeStatus != MlaStatus::SUCCESS) {
        InitializeMlaPatch(config, platform, SelectMlaFamily(config), patch);
        return patch.header.status = normalizeStatus;
    }
    auto status = ValidateMlaSeqViews(config, platform, seqQ, seqKv);
    if (status != MlaStatus::SUCCESS) {
        InitializeMlaPatch(config, platform, SelectMlaFamily(config), patch);
        return patch.header.status = status;
    }
    if (config.batchSize == 0 || config.qSeqSize == 0 || config.kvSeqSize == 0) {
        InitializeMlaPatch(config, platform, MlaFamily::NO_SCHEDULE, patch);
        return patch.header.status = MlaStatus::SUCCESS;
    }
    try {
        if (SelectMlaFamily(config) == MlaFamily::ARCH22_MLA) {
            Arch22::MlaNonQuantValueState tiling;
            return tiling.Compute(config, platform, seqQ, seqKv, patch);
        }
        Arch22::MlaLegacyValueState tiling;
        return tiling.Compute(config, platform, seqQ, seqKv, patch);
    } catch (const std::bad_alloc &) {
        return patch.header.status = MlaStatus::CAPACITY_EXCEEDED;
    }
}

MlaStatus ComputeMlaWorkspaceCapacity(const MlaScheduleConfig &inputConfig, const MlaPlatformInfo &platform,
                                      uint64_t &capacity)
{
    MlaScheduleConfig config = inputConfig;
    const auto status = NormalizeMlaScheduleConfig(config);
    if (status != MlaStatus::SUCCESS) {
        return status;
    }
    const auto dimensionsStatus = ValidateMlaValueDimensions(config);
    if (dimensionsStatus != MlaStatus::SUCCESS) {
        return dimensionsStatus;
    }
    if (platform.aicNum == 0 || platform.aicNum > kMlaMaxAicNum ||
        (platform.aivNum != platform.aicNum && platform.aivNum != 2 * platform.aicNum)) {
        return MlaStatus::INVALID_ARGUMENT;
    }
    if (config.batchSize == 0 || config.qSeqSize == 0 || config.kvSeqSize == 0) {
        capacity = platform.sysWorkspaceSize;
        return MlaStatus::SUCCESS;
    }
    if (SelectMlaFamily(config) == MlaFamily::ARCH22_MLA) {
        Arch22::MlaNonQuantValueState state;
        return state.WorkspaceCapacity(config, platform, capacity);
    }
    Arch22::MlaLegacyValueState state;
    return state.WorkspaceCapacity(config, platform, capacity);
}
} // namespace FlashMla
