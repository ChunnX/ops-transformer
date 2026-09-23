/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_comm_algo_selector.h"
#include "mc2_log_compat.h"
#include <vector>

namespace Mc2Hcom {

uint32_t Mc2CommAlgoSelector::GetL0TopoType(const char *group)
{
    uint32_t topoType = INVALID_TOPO_TYPE;
    HcclResult ret = MC2HcomTopology::TryGetGroupTopoType(group, &topoType);
    if (ret != HCCL_SUCCESS) {
        OP_LOGW("", "[CommAlgoSelector] TryGetGroupTopoType failed, ret=%d", static_cast<int>(ret));
        return INVALID_TOPO_TYPE;
    }
    return topoType;
}

uint32_t Mc2CommAlgoSelector::GetLayerNum(const char *group)
{
    std::vector<uint32_t> layers;
    HcclResult ret = MC2HcomTopology::CommGetNetLayersByGroup(group, layers);
    if (ret != HCCL_SUCCESS) {
        OP_LOGW("", "[CommAlgoSelector] CommGetNetLayersByGroup failed, ret=%d", static_cast<int>(ret));
        return 0;
    }
    return static_cast<uint32_t>(layers.size());
}

std::string Mc2CommAlgoSelector::SelectAlgoName(const std::string &opName, const char *group, uint8_t commEngine,
                                                uint64_t commDataBytes, uint32_t rankSize, const CommAlgoEntry *entries,
                                                uint32_t entryCount, const std::string &defaultAlgoName)
{
    if (group == nullptr || entries == nullptr || entryCount == 0) {
        OP_LOGW(opName.c_str(), "[CommAlgoSelector] invalid input, fallback to default: %s", defaultAlgoName.c_str());
        return defaultAlgoName;
    }

    uint32_t n = entryCount;
    if (n > MAX_COMM_ALGO_CANDIDATES) {
        OP_LOGW(opName.c_str(), "[CommAlgoSelector] entries size %u exceeds max %u, truncated", n,
                MAX_COMM_ALGO_CANDIDATES);
        n = MAX_COMM_ALGO_CANDIDATES;
    }

    uint32_t topoType = GetL0TopoType(group);
    uint32_t layerNum = GetLayerNum(group);
    OP_LOGI(opName.c_str(),
            "[CommAlgoSelector] group=%s, topoType=%u(0x%x), layerNum=%u, commEngine=%u, "
            "commDataBytes=%llu, rankSize=%u, entries=%u",
            group, topoType, topoType, layerNum, commEngine, commDataBytes, rankSize, n);

    // 表项各维度（引擎/卡数/层数/拓扑/数据量）均为硬约束，全部匹配才成为候选；
    // 引擎0为通配引擎；topoType/layerNum查询失败（未知）时对应维度不做约束
    const CommAlgoEntry *best = nullptr;
    for (uint32_t i = 0; i < n; i++) {
        const CommAlgoEntry *entry = &entries[i];
        if (entry->commEngine != 0 && entry->commEngine != commEngine) {
            continue;
        }
        if (rankSize < entry->minRankSize || rankSize > entry->maxRankSize) {
            continue;
        }
        if (layerNum != 0 && (entry->minLayers > layerNum || entry->maxLayers < layerNum)) {
            continue;
        }
        if (topoType != INVALID_TOPO_TYPE && entry->topoType != WILDCARD_TOPO_TYPE && entry->topoType != topoType) {
            continue;
        }
        if (commDataBytes < entry->minBytes || commDataBytes > entry->maxBytes) {
            continue;
        }
        if (best == nullptr || entry->priority > best->priority) {
            best = entry;
        }
    }

    if (best == nullptr) {
        OP_LOGW(opName.c_str(), "[CommAlgoSelector] no candidate matched all constraints, fallback to default: %s",
                defaultAlgoName.c_str());
        return defaultAlgoName;
    }

    OP_LOGI(opName.c_str(), "[CommAlgoSelector] selected algo: %s (priority=%d)", best->algoName, best->priority);
    return best->algoName;
}

} // namespace Mc2Hcom
