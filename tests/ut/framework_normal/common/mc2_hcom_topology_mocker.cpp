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

void MC2HcomTopologyMocker::SetValue(const char* key, uint64_t value)
{
    mockValue_[key] = value;
}

void MC2HcomTopologyMocker::SetValues(const MockValues& values)
{
    for (auto &[key, value] : values) {
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
HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    *rankNum = static_cast<uint32_t>(MC2HcomTopologyMocker::GetInstance().GetValue("rankNum", DEFAULT_RANK_NUM));
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetCclBufferSizeByGroup(const char *group, uint64_t *cclBufferSize, HcclComm *hcclComm)
{
    *cclBufferSize = MC2HcomTopologyMocker::GetInstance().GetValue("cclBufferSize", DEFAULT_CCL_BUFFER_SIZE);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize(const char *group, uint64_t* cclBufferSize)
{
    *cclBufferSize = MC2HcomTopologyMocker::GetInstance().GetValue("cclBufferSize", DEFAULT_CCL_BUFFER_SIZE);
    return HCCL_SUCCESS;
}

// private:
MC2HcomTopology &MC2HcomTopology::GetInstance()
{
    static MC2HcomTopology instance("");
    return instance;
}

MC2HcomTopology::MC2HcomTopology(const char *libPath)
{
}

HcclResult MC2HcomTopology::CallHcomGetCommHandleByGroup(const char *group, HcclComm *commHandle) const
{
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallCommGetCCLBufSizeCfg(HcclComm comm, uint64_t *cclBufferSize) const
{
    *cclBufferSize = MC2HcomTopologyMocker::GetInstance().GetValue("cclBufferSize", DEFAULT_CCL_BUFFER_SIZE);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallHcomGetRankSizeEx(const char *group, uint32_t *ranksize, uint32_t flag) const
{
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallHcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t flag) const
{
    return HCCL_SUCCESS;
}

}  // namespace Mc2Hcom
