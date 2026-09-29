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
#include <algorithm>
#include <vector>

namespace Mc2Hcom {

bool Mc2CommAlgoSelector::GetTopoInfo(const char* group, Mc2TopoInfo& topoInfo)
{
    if (group == nullptr) {
        return false;
    }
    topoInfo = Mc2TopoInfo{};
    HcclResult ret = MC2HcomTopology::CommGetTopoInfoByGroup(group, topoInfo);
    if (ret != HCCL_SUCCESS) {
        OP_LOGW("", "[CommAlgoSelector] CommGetTopoInfoByGroup failed, ret=%d", static_cast<int>(ret));
        return false;
    }
    return true;
}

// 由装配事实按位拼接组网标志：层数标志来自最小覆盖层数topoLevelNums(0未知不置位)；
// PCIE混联取原始事实；L0实例结构位由拓扑实例聚合直接拼接(HAS_MESH/HAS_CLOS/MESH_COVER/CLOS_COVER)，
// 实例聚合接口缺失时结构位保持false，具体算法表项不命中、由兜底表项承接回退默认算法
uint32_t Mc2CommAlgoSelector::CalcTopoFlags(const Mc2TopoInfo& topoInfo)
{
    uint32_t flags = TOPO_FLAG_NONE;
    if (topoInfo.topoLevelNums > 1) {
        flags |= TOPO_FLAG_MULTI_LEVEL;
    } else if (topoInfo.topoLevelNums == 1) {
        flags |= TOPO_FLAG_SINGLE_LEVEL;
    }
    if (topoInfo.level0PcieMix) {
        flags |= TOPO_FLAG_PCIE_MIX;
    }
    if (topoInfo.l0HasMesh) {
        flags |= TOPO_FLAG_L0_HAS_MESH;
    }
    if (topoInfo.l0HasClos) {
        flags |= TOPO_FLAG_L0_HAS_CLOS;
    }
    if (topoInfo.l0MeshCoverL0) {
        flags |= TOPO_FLAG_L0_MESH_COVER;
    }
    if (topoInfo.l0ClosCoverL0) {
        flags |= TOPO_FLAG_L0_CLOS_COVER;
    }
    return flags;
}

std::string Mc2CommAlgoSelector::SelectAlgoName(const std::string& opName, const char* group, uint8_t commEngine,
                                                uint64_t commDataBytes, uint32_t rankSize, const CommAlgoEntry* entries,
                                                uint32_t entryCount, const std::string& defaultAlgoName)
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
    // 一次性装配组网整体信息，拼出组网标志位匹配量(对齐参考实现TopoInfoWithNetLayerDetails的
    // 装配+比较方式)；装配失败时相应标志不置位，具体表项不命中、由兜底表项承接回退默认算法
    Mc2TopoInfo topoInfo;
    const bool topoValid = GetTopoInfo(group, topoInfo);
    const uint32_t topoFlags = CalcTopoFlags(topoInfo);
    OP_LOGI(opName.c_str(),
            "[CommAlgoSelector] group=%s, topoValid=%d, topoFlags=0x%x, level0Topo=%u, levelNum(cover)=%u, "
            "level0RankSize=%u, level0PcieMix=%d, userRankSize=%u, commEngine=%u, commDataBytes=%llu, rankSize=%u, "
            "entries=%u",
            group, static_cast<int>(topoValid), topoFlags, static_cast<uint32_t>(topoInfo.level0Topo),
            topoInfo.topoLevelNums, topoInfo.level0RankSize, static_cast<int>(topoInfo.level0PcieMix),
            topoInfo.userRankSize, commEngine, commDataBytes, rankSize, n);

    const CommAlgoEntry* buf[2][MAX_COMM_ALGO_CANDIDATES];
    uint32_t cnt[2] = {n, 0};
    uint32_t src = 0;
    uint32_t dst = 1;
    for (uint32_t i = 0; i < n; i++) {
        buf[0][i] = &entries[i];
    }

    // 硬过滤：任一维度清空候选集即判定无匹配，候选集保持为空并最终回退默认算法；
    // 维度未知(组网形态UNKNOWN/标志未置)时该维度不筛除候选，由兜底表项承接回退默认算法
    auto applyFilter = [&](const char* name, auto&& pred) {
        if (cnt[src] == 0) {
            return; // 已有维度无匹配，后续维度无需再过滤
        }
        cnt[dst] = 0;
        for (uint32_t i = 0; i < cnt[src]; i++) {
            if (pred(buf[src][i])) {
                buf[dst][cnt[dst]++] = buf[src][i];
            }
        }
        if (cnt[dst] == 0) {
            OP_LOGW(opName.c_str(),
                    "[CommAlgoSelector] filter '%s' removed all %u candidates, no match, fallback to default: %s", name,
                    cnt[src], defaultAlgoName.c_str());
            cnt[src] = 0; // 硬过滤：候选集清空后保持为空
            return;
        }
        std::swap(src, dst);
        OP_LOGD(opName.c_str(), "[CommAlgoSelector] after '%s': %u candidates remain", name, cnt[src]);
    };

    applyFilter("commEngine",
                [&](const CommAlgoEntry* e) { return e->commEngine == 0 || e->commEngine == commEngine; });
    applyFilter("rankSize",
                [&](const CommAlgoEntry* e) { return rankSize >= e->minRankSize && rankSize <= e->maxRankSize; });
    // 组网标志维度：层数/PCIE混联/L0实例结构等组网事实拼成标志位按位与匹配(对齐参考实现MachineFlag机制)，
    // 表项要求TOPO_FLAG_NONE时不限制；装配未知(标志未置)时具体表项不命中，由兜底表项承接回退默认算法
    applyFilter("topoFlags", [&](const CommAlgoEntry* e) { return (topoFlags & e->topoFlags) == e->topoFlags; });
    applyFilter("commDataBytes",
                [&](const CommAlgoEntry* e) { return commDataBytes >= e->minBytes && commDataBytes <= e->maxBytes; });

    if (cnt[src] == 0) {
        OP_LOGW(opName.c_str(), "[CommAlgoSelector] no candidate survived all filters, fallback to default: %s",
                defaultAlgoName.c_str());
        return defaultAlgoName;
    }

    const CommAlgoEntry* best = buf[src][0];
    for (uint32_t i = 1; i < cnt[src]; i++) {
        if (buf[src][i]->priority > best->priority) {
            best = buf[src][i];
        }
    }

    OP_LOGI(opName.c_str(), "[CommAlgoSelector] selected algo: %s (priority=%d, candidates=%u)", best->algoName,
            best->priority, cnt[src]);
    return best->algoName;
}

} // namespace Mc2Hcom
