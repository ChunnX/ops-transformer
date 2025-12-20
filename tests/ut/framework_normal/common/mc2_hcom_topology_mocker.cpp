/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

void MC2HcomTopologyMocker::SetValue(const char* key, uint32_t value)
{
    mockValue_[key] = value;
}

void MC2HcomTopologyMocker::SetValues(const MockValues& values)
{
    for (auto &[key, value] : values) {
        SetValue(key, value);
    }
}

uint32_t MC2HcomTopologyMocker::GetValue(const char* key, uint32_t defaultValue) const
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

MC2HcomTopology::MC2HcomTopology(const char *libPath)
{
}

MC2HcomTopology &MC2HcomTopology::GetInstance()
{
    static MC2HcomTopology instance("");
    return instance;
}

HcclResult MC2HcomTopology::CallHcomGetCommHandleByGroup(const char *group, HcclComm *commHandle)
{
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallCommGetNetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
{
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *topoType)
{
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum)
{
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    constexpr static uint32_t DEFAULT_RANK_NUM = 8;
    *rankNum = MC2HcomTopologyMocker::GetInstance().GetValue("rankNum", DEFAULT_RANK_NUM);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    return HCCL_SUCCESS;
}
}  // namespace Mc2Hcom
