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
 * \file mc2_hcom_topo_info.cc
 * \brief
 */

#include <cstdlib>
#include <string>
#include <dlfcn.h>
#include "log/log.h"
#include "mc2_hcom_topo_info.h"
#ifndef BUILD_OPEN_PROJECT
#include "hcom/hcom_topo_info.h"
#endif

namespace Mc2Hcom {
const std::string HCOM_GET_COMM_FUNC_NAME = "HcomGetCommHandleByGroup";
const std::string COMM_GET_NET_LAYERS_NAME = "CommGetNetLayers";
const std::string COMM_GET_TOPO_TYPE_NAME = "CommGetInstTopoTypeByNetLayer";
const std::string COMM_GET_SIZE_NAME = "CommGetInstSizeByNetLayer";

static const char *GetLibPath()
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OP_LOGE("", "Ascend home path doesn't exist.");
        return nullptr;
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl.so";
#else
    return nullptr;
#endif
    std::string fullPath = ascendPath + hcclPathPostfix;
    OP_LOGI("", "Loading lib in path %s.", fullPath.c_str());
    return fullPath.c_str();
}

template <typename T>
T GetHcclLibFunc(void *handle, const std::string &funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        OP_LOGE("", "Load func=%s error=%s in lib hccl failed.", funcName.c_str(), dlerror());
    }
    return func;
}

MC2HcomTopology::MC2HcomTopology(const char *libPath)
{
    handle_ = dlopen(libPath, RTLD_NOW);
    if (handle_ == nullptr) {
        OP_LOGE("", "Load lib hccl failed.");
        return;
    }

    getCommHandle_ = GetHcclLibFunc<FuncGetHandle>(handle_, HCOM_GET_COMM_FUNC_NAME);
    getNetLayers_ = GetHcclLibFunc<FuncGetNetLayers>(handle_, COMM_GET_NET_LAYERS_NAME);
    getTopoType_ = GetHcclLibFunc<FuncGetTopoType>(handle_, COMM_GET_TOPO_TYPE_NAME);
    getInstSize_ = GetHcclLibFunc<FuncGetInstSize>(handle_, COMM_GET_SIZE_NAME);
    if (getCommHandle_ == nullptr || getNetLayers_ == nullptr || getTopoType_ == nullptr || getInstSize_ == nullptr) {
        OP_LOGE("", "Lib load new topo functions failed.");
        getCommHandle_ = nullptr;
        getNetLayers_ = nullptr;
        getTopoType_ = nullptr;
        getInstSize_ = nullptr;
        return;
    }

    OP_LOGI("", "Init MC2HcomTopoLogy Success.");
}

MC2HcomTopology &MC2HcomTopology::GetInstance()
{
    static const char *libPath = GetLibPath();
    static MC2HcomTopology loader(libPath);
    return loader;
}

HcclResult MC2HcomTopology::CallHcomGetCommHandleByGroup(const char *group, HcclComm *commHandle)
{
    if (getCommHandle_ == nullptr) {
        OP_LOGE("", "Failed to get comm handle, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getCommHandle_(group, commHandle));
}

HcclResult MC2HcomTopology::CallCommGetNetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
{
    if (getNetLayers_ == nullptr) {
        OP_LOGE("", "Failed to get net layers, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getNetLayers_(comm, netLayers, netLayerNum));
}

HcclResult MC2HcomTopology::CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *topoType)
{
    if (getTopoType_ == nullptr) {
        OP_LOGE("", "Failed to get topo type, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getTopoType_(comm, netLayer, topoType));
}

HcclResult MC2HcomTopology::CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum)
{
    if (getInstSize_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getInstSize_(comm, netLayer, rankNum));
}

#ifdef BUILD_OPEN_PROJECT
HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    HcclComm hcclComm = nullptr;
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    if (group == nullptr || rankNum == nullptr) {
        OP_LOGE("", "Group or rank num is nullptr.");
        return HCCL_E_PTR;
    }
    HcclResult ret = GetInstance().CallHcomGetCommHandleByGroup(group, &hcclComm);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get comm handle.");
        return ret;
    }
    ret = GetInstance().CallCommGetNetLayers(hcclComm, &netLayers, &netLayerNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get net layers.");
        return ret;
    }
    ret = GetInstance().CallCommGetInstSizeByNetLayer(hcclComm, *netLayers, rankNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get inst size.");
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    HcclComm hcclComm = nullptr;
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    if (group == nullptr || topoType == nullptr) {
        OP_LOGE("", "Group or topo type is nullptr.");
        return HCCL_E_PTR;
    }
    HcclResult ret = GetInstance().CallHcomGetCommHandleByGroup(group, &hcclComm);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get comm handle.");
        return ret;
    }
    ret = GetInstance().CallCommGetNetLayers(hcclComm, &netLayers, &netLayerNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get net layers.");
        return ret;
    }
    ret = GetInstance().CallCommGetInstTopoTypeByNetLayer(hcclComm, *netLayers, topoType);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get topo type.");
        return ret;
    }
    return HCCL_SUCCESS;
}
#else
HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    int64_t rankSize = *rankNum;
    uint32_t ret = ge::HcomTopoInfo::Instance().GetGroupRankSize(group, rankSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return HCCL_E_NOT_FOUND;
    }
    *rankNum = static_cast<uint32_t>(rankSize);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    ge::HcomTopoInfo::TopoInfo topoInfo;
    if (!ge::HcomTopoInfo::Instance().TryGetGroupTopoInfo(group, topoInfo)) {
        OP_LOGW("", "GroupTopoInfo not set.");
        *topoType = COMM_ALG_DEFAULT;
        return HCCL_E_NOT_FOUND;
    }
    *topoType = topoInfo.topo_level_descs[static_cast<int32_t>(ge::HcomTopoInfo::TopoLevel::L0)].comm_sets;
    OP_LOGD("", "comm_sets from TopoInfo is %u, COMM_MESH is %u", *topoType, COMM_MESH);
    return HCCL_SUCCESS;
}
#endif
}  // namespace Mc2Hcom
