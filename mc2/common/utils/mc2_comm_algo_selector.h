/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_COMM_ALGO_SELECTOR_H
#define MC2_COMM_ALGO_SELECTOR_H

#include <cstdint>
#include <string>
#include "mc2_hcom_topo_info.h"

namespace Mc2Hcom {

constexpr uint32_t MAX_COMM_ALGO_CANDIDATES = 32;

// 组网标志位：由Mc2TopoInfo一次性装配的事实按位拼接(对齐参考实现external_alg_rules的MachineFlag机制)，
// 表项按位与匹配：机器当前标志包含表项要求的全部标志位才命中
constexpr uint32_t TOPO_FLAG_NONE = 0U;
constexpr uint32_t TOPO_FLAG_SINGLE_LEVEL = 0b1U; ///< 单层网络(topoLevelNums==1)
constexpr uint32_t TOPO_FLAG_MULTI_LEVEL = 0b10U; ///< 多层网络(topoLevelNums>1)，对齐参考FlagMultiLevel
constexpr uint32_t TOPO_FLAG_PCIE_MIX = 0b100U;   ///< L0存在PCIE协议链路(原始事实)
// L0实例结构事实位：按拓扑实例聚合直接拼接(对齐参考实现CalcLevel0TopoShape的数据来源)，各形态唯一编码
constexpr uint32_t TOPO_FLAG_L0_HAS_MESH = 0b1000U;      ///< L0存在1DMESH实例(任意mesh族形态均置位)
constexpr uint32_t TOPO_FLAG_L0_HAS_CLOS = 0b10000U;     ///< L0存在CLOS实例
constexpr uint32_t TOPO_FLAG_L0_MESH_COVER = 0b100000U;  ///< 存在覆盖L0全部rank的1DMESH实例(纯mesh单机)
constexpr uint32_t TOPO_FLAG_L0_CLOS_COVER = 0b1000000U; ///< 存在覆盖L0全部rank的CLOS实例
// UBX机型组合常量：Mesh+CLOS混联且CLOS实例覆盖L0(die内mesh不覆盖L0全部rank)
constexpr uint32_t TOPO_FLAG_UBX = TOPO_FLAG_L0_HAS_MESH | TOPO_FLAG_L0_HAS_CLOS | TOPO_FLAG_L0_CLOS_COVER;

struct CommAlgoEntry {
    uint8_t commEngine;
    uint32_t topoFlags; ///< 组网标志位要求(TOPO_FLAG_*按位组合，TOPO_FLAG_NONE表示不要求)
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
    // 由装配事实按位拼接组网标志(层数/PCIE混联/L0实例结构)，装配失败(层数未知/结构事实未置)相应位不置位
    static uint32_t CalcTopoFlags(const Mc2TopoInfo& topoInfo);
};

} // namespace Mc2Hcom

#endif // MC2_COMM_ALGO_SELECTOR_H
