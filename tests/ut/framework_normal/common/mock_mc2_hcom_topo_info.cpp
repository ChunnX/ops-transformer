/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mock_mc2_hcom_topo_info.cpp
 * \brief
 */

#include "mc2_hcom_topo_info.h"
namespace Mc2Hcom {

MC2HcomTopology::MC2HcomTopology(const char *libPath)
{
    (void) libPath;
}

MC2HcomTopology &MC2HcomTopology::GetInstance()
{
    static const char *libPath = "";
    static MC2HcomTopology instance(libPath);
    return instance;
}

HcclResult MC2HcomTopology::CallHcomGetCommHandleByGroup(const char *group, HcclComm *commHandle)
{
    (void) group;
    (void) commHandle;
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallCommGetNetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
{
    (void) comm;
    (void) netLayers;
    (void) netLayerNum;
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *topoType)
{
    (void) comm;
    (void) netLayer;
    (void) topoType;
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum)
{
    (void) comm;
    (void) netLayer;
    (void) rankNum;
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    (void) group;
    constexpr uint32_t DEFAULT_RANK_NUM = 8; // 默认的rank数量
    *rankNum = DEFAULT_RANK_NUM;
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    (void) group;
    (void) topoType;
    return HCCL_SUCCESS;
}
}  // namespace Mc2Hcom
