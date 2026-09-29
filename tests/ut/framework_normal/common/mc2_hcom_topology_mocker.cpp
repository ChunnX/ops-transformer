/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mock_mc2_hcom_topo_info.cpp
 * \brief
 */

#include "mc2_hcom_topo_info.h"
#include "mc2_hcom_topology_mocker.h"
namespace Mc2Hcom {
MC2HcomTopologyMocker& MC2HcomTopologyMocker::GetInstance()
{
    static MC2HcomTopologyMocker instance;
    return instance;
}

void MC2HcomTopologyMocker::SetValue(const char* key, uint64_t value)
{
    mockValue_[key] = value;
}

void MC2HcomTopologyMocker::SetValues(const MockValues& values)
{
    for (auto& [key, value] : values) {
        SetValue(key, value);
    }
}

uint64_t MC2HcomTopologyMocker::GetValue(const char* key, uint64_t defaultValue) const
{
    auto it = mockValue_.find(key);
    if (it == mockValue_.end()) {
        return defaultValue;
    }
    return it->second;
}

void MC2HcomTopologyMocker::Reset()
{
    mockValue_.clear();
}

// mock mc2/common/inc/mc2_hcom_topo_info.h ----------------------------------------------------------------------------
constexpr static uint64_t DEFAULT_RANK_NUM = 8;
constexpr static uint64_t DEFAULT_CCL_BUFFER_SIZE = 6000ULL * 1024ULL * 1024ULL;

// class MC2HcomTopology
// public:
HcclResult MC2HcomTopology::CommGetInstSizeByGroup([[maybe_unused]] const char* group, uint32_t* rankNum)
{
    if (rankNum == nullptr) {
        return HCCL_E_PARA;
    }
    *rankNum = static_cast<uint32_t>(MC2HcomTopologyMocker::GetInstance().GetValue("rankNum", DEFAULT_RANK_NUM));
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType([[maybe_unused]] const char* group, uint32_t* topoType)
{
    if (topoType == nullptr) {
        return HCCL_E_PARA;
    }
    *topoType = static_cast<uint32_t>(MC2HcomTopologyMocker::GetInstance().GetValue("topoType", 0));
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetCclBufferSizeByGroup([[maybe_unused]] const char* group, uint64_t* cclBufferSize,
                                                        [[maybe_unused]] HcclComm* hcclComm)
{
    if (cclBufferSize == nullptr) {
        return HCCL_E_PARA;
    }
    *cclBufferSize = MC2HcomTopologyMocker::GetInstance().GetValue("cclBufferSize", DEFAULT_CCL_BUFFER_SIZE);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize([[maybe_unused]] const char* group, uint64_t* cclBufferSize)
{
    if (cclBufferSize == nullptr) {
        return HCCL_E_PARA;
    }
    *cclBufferSize = MC2HcomTopologyMocker::GetInstance().GetValue("cclBufferSize", DEFAULT_CCL_BUFFER_SIZE);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetHcclBufferByGroup([[maybe_unused]] const char* group, [[maybe_unused]] void** buffer,
                                                     [[maybe_unused]] uint64_t* size)
{
    if (buffer == nullptr || size == nullptr) {
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetNetLayersByGroup([[maybe_unused]] const char* group, std::vector<uint32_t>& layers)
{
    layers.clear();
    uint64_t layerNum = MC2HcomTopologyMocker::GetInstance().GetValue("layerNum", 1);
    for (uint64_t i = 0; i < layerNum; i++) {
        std::string key = "layer" + std::to_string(i);
        layers.push_back(static_cast<uint32_t>(MC2HcomTopologyMocker::GetInstance().GetValue(key.c_str(), 0)));
    }
    return static_cast<HcclResult>(MC2HcomTopologyMocker::GetInstance().GetValue("getLayersRet", HCCL_SUCCESS));
}

HcclResult MC2HcomTopology::CommGetTopoTypeByLayer([[maybe_unused]] const char* group, uint32_t layer,
                                                   uint32_t* topoType)
{
    if (topoType == nullptr) {
        return HCCL_E_PARA;
    }
    std::string key = "layer" + std::to_string(layer) + "TopoType";
    *topoType = static_cast<uint32_t>(MC2HcomTopologyMocker::GetInstance().GetValue(key.c_str(), COMM_MESH));
    return static_cast<HcclResult>(MC2HcomTopologyMocker::GetInstance().GetValue("getTopoTypeRet", HCCL_SUCCESS));
}

HcclResult MC2HcomTopology::CommGetTopoInfoByGroup([[maybe_unused]] const char* group, Mc2TopoInfo& topoInfo)
{
    if (group == nullptr) {
        return HCCL_E_PARA;
    }
    const MC2HcomTopologyMocker& mocker = MC2HcomTopologyMocker::GetInstance();
    topoInfo = Mc2TopoInfo{};
    topoInfo.userRankSize =
        static_cast<uint32_t>(mocker.GetValue("userRankSize", mocker.GetValue("rankNum", DEFAULT_RANK_NUM)));
    const uint64_t layerNum = mocker.GetValue("layerNum", 1);
    for (uint64_t i = 0; i < layerNum; i++) {
        std::string key = "layer" + std::to_string(i);
        topoInfo.netLayers.push_back(static_cast<uint32_t>(mocker.GetValue(key.c_str(), 0)));
    }
    topoInfo.topoLevelNums = static_cast<uint32_t>(mocker.GetValue("topoLevelNums", layerNum));
    // L0实例结构事实(默认模拟A5 UBX单机：Mesh+CLOS混联且CLOS覆盖L0，die内mesh不覆盖)
    topoInfo.l0HasMesh = mocker.GetValue("l0HasMesh", 1) != 0;
    topoInfo.l0HasClos = mocker.GetValue("l0HasClos", 1) != 0;
    topoInfo.l0MeshCoverL0 = mocker.GetValue("l0MeshCover", 0) != 0;
    topoInfo.l0ClosCoverL0 = mocker.GetValue("l0ClosCover", 1) != 0;
    // 组网形态汇总值(仅日志/comm_sets映射用，不参与算法表匹配)
    topoInfo.level0Topo =
        static_cast<Mc2Level0Topo>(mocker.GetValue("level0Topo", static_cast<uint64_t>(Mc2Level0Topo::MESH_1D_CLOS)));
    topoInfo.level0PcieMix = mocker.GetValue("level0PcieMix", 0) != 0;
    topoInfo.level0RankSize = static_cast<uint32_t>(mocker.GetValue("level0RankSize", topoInfo.userRankSize));
    return static_cast<HcclResult>(mocker.GetValue("getTopoInfoRet", HCCL_SUCCESS));
}

// private:
MC2HcomTopology::MC2HcomTopology() {}

MC2HcomTopology::~MC2HcomTopology() {}

HcclResult MC2HcomTopology::CallHcomGetCommHandleByGroup([[maybe_unused]] const char* group,
                                                         [[maybe_unused]] HcclComm* commHandle) const
{
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallCommGetCCLBufSizeCfg([[maybe_unused]] HcclComm comm, uint64_t* cclBufferSize) const
{
    if (cclBufferSize == nullptr) {
        return HCCL_E_PARA;
    }
    *cclBufferSize = MC2HcomTopologyMocker::GetInstance().GetValue("cclBufferSize", DEFAULT_CCL_BUFFER_SIZE);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallHcomGetRankSizeEx([[maybe_unused]] const char* group,
                                                  [[maybe_unused]] uint32_t* ranksize,
                                                  [[maybe_unused]] uint32_t flag) const
{
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallCommGetNetLayers([[maybe_unused]] HcclComm comm, uint32_t** netLayer,
                                                 uint32_t* netLayerNum) const
{
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallCommGetInstTopoTypeByNetLayer([[maybe_unused]] HcclComm comm,
                                                              [[maybe_unused]] uint32_t netLayer,
                                                              uint32_t* topoType) const
{
    if (topoType == nullptr) {
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

} // namespace Mc2Hcom
