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

// 由装配信息按位推导组网标志，对齐参考实现external_alg_rules的MachineFlag语义：
// 单层/多层标志来自最小覆盖层数topoLevelNums(0未知不置标志)；
// PCIE混联与UBX标志按参考FlagPcieMix/FlagUbx均为"MESH_1D_CLOS机型+是否含PCIE链路"的细分
// (参考FlagPcieMix的IsLayerAllConnetedWithTopo子条件依赖按拓扑实例聚合的实验接口，此处不感知)
uint32_t Mc2CommAlgoSelector::CalcTopoFlags(const Mc2TopoInfo& topoInfo)
{
    uint32_t flags = TOPO_FLAG_NONE;
    if (topoInfo.topoLevelNums > 1) {
        flags |= TOPO_FLAG_MULTI_LEVEL;
    } else if (topoInfo.topoLevelNums == 1) {
        flags |= TOPO_FLAG_SINGLE_LEVEL;
    }
    if (topoInfo.level0Topo == Mc2Level0Topo::MESH_1D_CLOS) {
        flags |= topoInfo.level0PcieMix ? TOPO_FLAG_PCIE_MIX : TOPO_FLAG_UBX;
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
    // 一次性装配组网整体信息，再拼出L0组网形态与组网标志位两个比较量
    // (对齐参考实现TopoInfoWithNetLayerDetails的装配+比较方式)；装配失败时相应维度按未知处理(不参与过滤)
    Mc2TopoInfo topoInfo;
    const bool topoValid = GetTopoInfo(group, topoInfo);
    const uint32_t level0Topo = static_cast<uint32_t>(topoInfo.level0Topo); // 未知时为INVALID_TOPO_TYPE
    const uint32_t topoFlags = CalcTopoFlags(topoInfo);
    OP_LOGI(opName.c_str(),
            "[CommAlgoSelector] group=%s, topoValid=%d, level0Topo=%u, topoFlags=0x%x, levelNum(cover)=%u, "
            "level0RankSize=%u, level0PcieMix=%d, userRankSize=%u, commEngine=%u, commDataBytes=%llu, rankSize=%u, "
            "entries=%u",
            group, static_cast<int>(topoValid), level0Topo, topoFlags, topoInfo.topoLevelNums, topoInfo.level0RankSize,
            static_cast<int>(topoInfo.level0PcieMix), topoInfo.userRankSize, commEngine, commDataBytes, rankSize, n);

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
    // 组网标志维度：层数/混联/UBX等组网信息拼成标志位按位与匹配(对齐参考实现MachineFlag机制)，
    // 表项要求TOPO_FLAG_NONE时不限制；装配未知(无标志)时具体表项不命中，由兜底表项承接回退默认算法
    applyFilter("topoFlags", [&](const CommAlgoEntry* e) { return (topoFlags & e->topoFlags) == e->topoFlags; });
    // L0组网形态维度：与装配出的Mc2Level0Topo值比较(对齐参考实现level0Topo==Level0Shape::MESH_1D的比较方式)，
    // 表项可指定具体形态、mesh族(MESH_ANY_TOPO_TYPE)或通配(WILDCARD_TOPO_TYPE)，未知跳过
    applyFilter("level0Topo", [&](const CommAlgoEntry* e) {
        if (level0Topo == INVALID_TOPO_TYPE || e->level0Topo == WILDCARD_TOPO_TYPE) {
            return true;
        }
        if (e->level0Topo == MESH_ANY_TOPO_TYPE) {
            return topoInfo.IsMeshTopo();
        }
        return e->level0Topo == level0Topo;
    });
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
