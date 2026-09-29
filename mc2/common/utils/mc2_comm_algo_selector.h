/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_COMM_ALGO_SELECTOR_H
#define MC2_COMM_ALGO_SELECTOR_H

#include <cstdint>
#include <string>
#include "mc2_hcom_topo_info.h"

namespace Mc2Hcom {

constexpr uint32_t MAX_COMM_ALGO_CANDIDATES = 32;
// 等值于Mc2Level0Topo::UNKNOWN，表示组网形态未知，topo维度过滤跳过
constexpr uint32_t INVALID_TOPO_TYPE = 0xFFFFFFFFU;
// topo维度通配，任意L0组网形态均命中
constexpr uint32_t WILDCARD_TOPO_TYPE = 0xFFFFFFFEU;
// 匹配mesh族形态(MESH_1D或MESH_1D_CLOS)
constexpr uint32_t MESH_ANY_TOPO_TYPE = 0xFFFFFFFDU;

// 组网标志位，由Mc2TopoInfo一次性装配推导(对齐参考实现external_alg_rules的MachineFlag机制)，
// 表项按位与匹配：机器当前标志包含表项要求的全部标志位才命中
constexpr uint32_t TOPO_FLAG_NONE = 0U;
constexpr uint32_t TOPO_FLAG_SINGLE_LEVEL = 0b1U; ///< 单层网络(topoLevelNums==1)
constexpr uint32_t TOPO_FLAG_MULTI_LEVEL = 0b10U; ///< 多层网络(topoLevelNums>1)，对齐参考FlagMultiLevel
constexpr uint32_t TOPO_FLAG_PCIE_MIX = 0b100U;   ///< Mesh+CLOS且含PCIE混联链路，对齐参考FlagPcieMix
// UBX机型(Mesh+CLOS且非PCIE混联)：L0按拓扑实例聚合判定为MESH_1D_CLOS
// (1个CLOS实例覆盖L0全部rank+1个1DMESH实例)，对齐参考FlagUbx与CalcLevel0TopoShape
constexpr uint32_t TOPO_FLAG_UBX = 0b1000U;

struct CommAlgoEntry {
    uint8_t commEngine;
    uint32_t level0Topo; ///< L0组网形态(Mc2Level0Topo取值)或通配值(WILDCARD_TOPO_TYPE/MESH_ANY_TOPO_TYPE)
    uint32_t topoFlags;  ///< 组网标志位要求(TOPO_FLAG_*按位组合，TOPO_FLAG_NONE表示不要求)
    uint64_t minBytes;
    uint64_t maxBytes;
    uint32_t minRankSize;
    uint32_t maxRankSize;
    int32_t priority;
    const char* algoName;
};

class Mc2CommAlgoSelector {
public:
    static std::string SelectAlgoName(const std::string& opName, const char* group, uint8_t commEngine,
                                      uint64_t commDataBytes, uint32_t rankSize, const CommAlgoEntry* entries,
                                      uint32_t entryCount, const std::string& defaultAlgoName);

private:
    // 一次性装配组网整体信息(Mc2TopoInfo)，装配失败返回false，调用方按维度未知处理
    static bool GetTopoInfo(const char* group, Mc2TopoInfo& topoInfo);
    // 由装配信息按位推导组网标志(单层/多层/PCIE混联/UBX)，装配失败(topoLevelNums==0/形态UNKNOWN)不置标志
    static uint32_t CalcTopoFlags(const Mc2TopoInfo& topoInfo);
};

} // namespace Mc2Hcom

#endif // MC2_COMM_ALGO_SELECTOR_H
