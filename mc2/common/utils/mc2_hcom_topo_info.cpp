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
 * \file mc2_hcom_topo_info.cpp
 * \brief
 */

#include <cstdlib>
#include <dlfcn.h>
#include "log/log.h"
#include "mc2_hcom_topo_info.h"
#include "ops_legacy/op_tiling/hcom_topo_info.h"
#include "mc2_get_hccl_buffer_size.h"
#ifndef BUILD_OPEN_PROJECT
#include "hcom/hcom_topo_info.h"
#endif

namespace {
    static constexpr uint64_t CTX_DEFAULT_COPY_OFFSET = 0;  // 默认从最开始拷贝
    static constexpr uint64_t CONTEXT_TAG_MAX_SIZE = 255;   // 最大上下文标签大小
    static constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1; // 当走MTE或者CCU通信时， hccl获取的组网层数应该为1
    static constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;  // 当走UB_MEM通信时， 获取link时应该传0
}

namespace Mc2Hcom {
const std::string HCOM_GET_COMM_FUNC_NAME = "HcomGetCommHandleByGroup";
#ifdef BUILD_OPEN_PROJECT
const std::string HCOM_GET_RANK_SIZE_EX = "HcomGetRankSizeEx";
const std::string HCOM_GET_L0_TOPO_TYPE_EX = "HcomGetL0TopoTypeEx";
#else
const std::string COMM_GET_NET_LAYERS_NAME = "CommGetNetLayers";
const std::string COMM_GET_TOPO_TYPE_NAME = "CommGetInstTopoTypeByNetLayer";
const std::string COMM_GET_SIZE_NAME = "CommGetInstSizeByNetLayer";
#endif
const std::string COMM_GET_CCL_BUFFER_SIZE_NAME = "CommGetCCLBufSizeCfg";

static const std::string GetLibPath(std::string &createTensorLibPath_)
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OP_LOGE("", "Ascend home path doesn't exist.");
        return nullptr;
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
    std::string createTensorLibPath = "/x86_64-linux/lib64/libnnopbase.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
    std::string createTensorLibPath = "/aarch64-linux/lib64/libnnopbase.so";
#else
    return nullptr;
#endif
    createTensorLibPath_ = std::string(ascendPath) + createTensorLibPath;
    std::string fullPath = std::string(ascendPath) + hcclPathPostfix;
    OP_LOGI("", "Loading lib in path %s.", fullPath.c_str());
    return fullPath;
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
#ifdef BUILD_OPEN_PROJECT
    getRankSizeEx_ = GetHcclLibFunc<FuncGetRankSizeEx>(handle_, HCOM_GET_RANK_SIZE_EX);
    getL0TopoTypeEx_ = GetHcclLibFunc<FuncGetL0TopoTypeEx>(handle_, HCOM_GET_L0_TOPO_TYPE_EX);
#else
    getNetLayers_ = GetHcclLibFunc<FuncGetNetLayers>(handle_, COMM_GET_NET_LAYERS_NAME);
    getTopoType_ = GetHcclLibFunc<FuncGetTopoType>(handle_, COMM_GET_TOPO_TYPE_NAME);
    getInstSize_ = GetHcclLibFunc<FuncGetInstSize>(handle_, COMM_GET_SIZE_NAME);
#endif
    getCclBufferSize_ = GetHcclLibFunc<FuncGetCclBufferSize>(handle_, COMM_GET_CCL_BUFFER_SIZE_NAME);

#ifdef BUILD_OPEN_PROJECT
    if (getCommHandle_ == nullptr || getRankSizeEx_ == nullptr || getL0TopoTypeEx_ == nullptr ||
        getCclBufferSize_ == nullptr) {
        OP_LOGE("", "Lib load new topo functions failed.");
        getCommHandle_ = nullptr;
        getCclBufferSize_ = nullptr;
        getRankSizeEx_ = nullptr;
        getL0TopoTypeEx_ = nullptr;
        return;
    }
#else
    if (getCommHandle_ == nullptr || getNetLayers_ == nullptr || getTopoType_ == nullptr || getInstSize_ == nullptr ||
        getCclBufferSize_ == nullptr) {
        OP_LOGE("", "Lib load new topo functions failed.");
        getCommHandle_ = nullptr;
        getNetLayers_ = nullptr;
        getTopoType_ = nullptr;
        getInstSize_ = nullptr;
        getCclBufferSize_ = nullptr;
        return;
    }
#endif

    OP_LOGI("", "Init MC2HcomTopoLogy Success.");
}

MC2HcomTopology &MC2HcomTopology::GetInstance()
{
    std::string createTensorLibPath;
    static const char *libPath = GetLibPath(createTensorLibPath).c_str();
    static MC2HcomTopology loader(libPath);
    return loader;
}

HcclResult MC2HcomTopology::CallHcomGetCommHandleByGroup(const char *group, HcclComm *commHandle) const
{
    if (getCommHandle_ == nullptr) {
        OP_LOGE("", "Failed to get comm handle, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getCommHandle_(group, commHandle));
}

#ifdef BUILD_OPEN_PROJECT
HcclResult MC2HcomTopology::CallHcomGetRankSizeEx(const char *group, uint32_t *ranksize, uint32_t flag) const
{
    if (getRankSizeEx_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getRankSizeEx_(group, ranksize, flag));
}

HcclResult MC2HcomTopology::CallHcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t flag) const
{
    if (getL0TopoTypeEx_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getL0TopoTypeEx_(group, topoType, flag));
}
#else
HcclResult MC2HcomTopology::CallCommGetNetLayers(HcclComm comm, uint32_t **netLayer, uint32_t *netLayerNum) const
{
    if (getNetLayers_ == nullptr) {
        OP_LOGE("", "Failed to get net layers, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getNetLayers_(comm, netLayer, netLayerNum));
}

HcclResult MC2HcomTopology::CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer,
                                                              uint32_t *topoType) const
{
    if (getTopoType_ == nullptr) {
        OP_LOGE("", "Failed to get topo type, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getTopoType_(comm, netLayer, topoType));
}

HcclResult MC2HcomTopology::CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum) const
{
    if (getInstSize_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getInstSize_(comm, netLayer, rankNum));
}
#endif
HcclResult MC2HcomTopology::CallCommGetCCLBufSizeCfg(HcclComm comm, uint64_t *cclBufferSize) const
{
    if (getCclBufferSize_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getCclBufferSize_(comm, cclBufferSize));
}

HcclResult MC2HcomTopology::CommGetCclBufferSizeByGroup(const char *group, uint64_t *cclBufferSize, HcclComm *hcclComm)
{
    if (group == nullptr || cclBufferSize == nullptr || hcclComm == nullptr) {
        OP_LOGE("", "Group or rank num or hcclComm is nullptr.");
        return HCCL_E_PTR;
    }
    HcclResult ret = GetInstance().CallHcomGetCommHandleByGroup(group, hcclComm);
    if (ret != HCCL_SUCCESS) {
        OP_LOGI("", "get nullptr comm handle.");
        *hcclComm = nullptr;
        return HCCL_SUCCESS;
    }
    ret = GetInstance().CallCommGetCCLBufSizeCfg(*hcclComm, cclBufferSize);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get buffer size.");
        return ret;
    }
    OP_LOGI("", "cclBufferSize is %lu", *cclBufferSize);
    return HCCL_SUCCESS;
}

#ifdef BUILD_OPEN_PROJECT
HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize(const char *group, uint64_t *cclBufferSize)
{
    if (ge::HcomTopoInfo::Instance().GetGroupLocalWindowSize(group, *cclBufferSize) != ge::GRAPH_SUCCESS) {
        OP_LOGD("", "Get winSize from GetGroupLocalWindowSize=%lu", *cclBufferSize);
        *cclBufferSize = mc2tiling::GetHcclBufferMaxSize();
        return HCCL_SUCCESS;
    }
    OP_LOGD("", "Get winSize from GetMaxWindowSize=%lu", *cclBufferSize);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    if (group == nullptr || rankNum == nullptr) {
        OP_LOGE("", "Group or rank num is nullptr.");
        return HCCL_E_PTR;
    }
    HcclResult ret = GetInstance().CallHcomGetRankSizeEx(group, rankNum, COMM_IS_NOT_SET_DEVICE);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get rank size.");
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    if (group == nullptr || topoType == nullptr) {
        OP_LOGE("", "Group or topo type is nullptr.");
        return HCCL_E_PTR;
    }
    CommTopo topoRet;
    HcclResult ret = GetInstance().CallHcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get topo type.");
        return ret;
    }
    *topoType = static_cast<uint32_t>(topoRet);
    return HCCL_SUCCESS;
}
#else
HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize(const char *group, uint64_t *cclBufferSize)
{
    uint32_t ret = ge::HcomTopoInfo::Instance().GetGroupLocalWindowSize(group, *cclBufferSize);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE("", "cclBufferSize not set.");
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

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

MC2HcomTopology::~MC2HcomTopology()
{
    if (handle_ != nullptr) {
        dlclose(handle_);
        handle_ = nullptr;
    }
    if (hcclLibHandle_ != nullptr) {
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
    }
}

HcclResult MC2HcomTopology::LoadHcclSymbols()
{
    OP_LOGD("", "start to load HCCL library and symbols");
    if (hcclLibHandle_ != nullptr)
        return HCCL_SUCCESS;
    std::string createTensorLibPath;
    const char *libPath = GetLibPath(createTensorLibPath).c_str();
    if (libPath == nullptr) {
        OP_LOGE("", "Failed to get HCCL library path");
        return HCCL_E_PTR;
    }

    hcclLibHandle_ = dlopen(libPath, RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        OP_LOGE("", "dlopen HCCL library failed: %s, error: %s", libPath, dlerror());
        return HCCL_E_PTR;
    }

    struct SymbolInfo {
        void **ptr;
        const char *name;
    };

    SymbolInfo symbols[] = {{reinterpret_cast<void **>(&HcclRankGraphGetLinks), "HcclRankGraphGetLinks"},
                            {reinterpret_cast<void **>(&HcclRankGraphGetLayers), "HcclRankGraphGetLayers"},
                            {reinterpret_cast<void **>(&HcclChannelAcquire), "HcclChannelAcquire"},
                            {reinterpret_cast<void **>(&HcclGetHcclBuffer), "HcclGetHcclBuffer"},
                            {reinterpret_cast<void **>(&HcclChannelGetHcclBuffer), "HcclChannelGetHcclBuffer"},
                            {reinterpret_cast<void **>(&HcclEngineCtxCreate), "HcclEngineCtxCreate"},
                            {reinterpret_cast<void **>(&HcclEngineCtxGet), "HcclEngineCtxGet"},
                            {reinterpret_cast<void **>(&HcclEngineCtxCopy), "HcclEngineCtxCopy"},
                            {reinterpret_cast<void **>(&HcclGetRankId), "HcclGetRankId"},
                            {reinterpret_cast<void **>(&HcclGetRankSize), "HcclGetRankSize"}};

    for (auto &sym : symbols) {
        *(sym.ptr) = GetHcclLibFunc<void *>(hcclLibHandle_, sym.name);
        if (*(sym.ptr) == nullptr) {
            OP_LOGE("", "Failed to load %s symbol", sym.name);
            dlclose(hcclLibHandle_);
            hcclLibHandle_ = nullptr;
            return HCCL_E_PTR;
        }
    }
    OP_LOGD("", "end to load HCCL library and symbols");
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::GetCommHandle(const char *groupEp)
{
    OP_LOGD("", "start to get HCCL communication handle, groupEp: %s", groupEp);
    HcclResult ret = getCommHandle_(groupEp, &hcclHandle_);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Get HCCL handle failed, groupEp: %s", groupEp);
        return HCCL_E_PTR;
    }
    OP_LOGD("", "Get HCCL communication handle success hcclHandle_ is: %p", hcclHandle_);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::GetHcclCommLink(uint32_t netLayers, uint32_t srcRankId, uint32_t dstRankId,
                                             const CommProtocol &protocol, CommLink *&links)
{
    OP_LOGD("", "start to get HCCL communication link");
    if (HcclRankGraphGetLinks == nullptr) {
        OP_LOGE("", "HcclRankGraphGetLinks is null");
        return HCCL_E_PTR;
    }

    CommLink *linksList = nullptr;
    uint32_t netLinkNum = 0;
    HcclResult hcclRet = HcclRankGraphGetLinks(hcclHandle_, netLayers, srcRankId, dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE("", "Get HCCL link failed");
        return HCCL_E_PTR;
    }
    OP_LOGD("", "Get HCCL communication link success, netLinkNum is: %u", netLinkNum);
    if (netLinkNum == 0) {
        OP_LOGE("", "No available HCCL links");
        return HCCL_E_PTR;
    }

    uint32_t index = 0;
    for (; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == protocol) {
            links = &linksList[index];
            break;
        }
    }

    if (index == netLinkNum) {
        OP_LOGE("", "No matching communication protocol found in HCCL links protocol is %d", protocol);
        return HCCL_E_PTR;
    }
    OP_LOGD("", "Get HCCL communication link success, protocol is: %d", protocol);

    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::GetHcclCommChannel(uint32_t rankDim, uint32_t srcRankId, const CommProtocol &protocol,
                                                const CommEngine &engine, std::vector<ChannelHandle> &channels)
{
    OP_LOGD("", "start to get HCCL communication channel");
    if (HcclRankGraphGetLayers == nullptr || HcclChannelAcquire == nullptr) {
        OP_LOGE("", "HCCL channel function pointer is null");
        return HCCL_E_PTR;
    }

    uint32_t channelNum = rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);
    CommLink *links = nullptr;
    uint32_t netLayerNum = 0;
    uint32_t *netLayerList = nullptr;

    HcclResult hcclRet = HcclRankGraphGetLayers(hcclHandle_, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE("", "Get HCCL layers failed");
        return HCCL_E_PTR;
    }
    OP_LOGD("", "Get HCCL layers success, netLayerNum is: %u", netLayerNum);

    uint32_t netLayers = netLayerNum - 1;
    if (protocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
        netLayers = HCCL_COMM_LAYERS_UB_MEM; // UB_MEM协议只走最底层的通信，直接指定netLayers为0
    }
    hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE("", "HCCL channel init failed");
        return HCCL_E_PTR;
    }
    OP_LOGD("", "HCCL channel init success");

    for (uint32_t i = 0; i < rankDim; ++i) {
        if (i == srcRankId) {
            continue;
        }
        uint32_t dstRank = i;
        uint32_t channelId = (i > srcRankId) ? (i - 1) : i;
        HcclResult ret = GetHcclCommLink(netLayers, srcRankId, dstRank, protocol, links);
        if (ret != HCCL_SUCCESS)
            return ret;

        channelDesc[channelId].channelProtocol = protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].notifyNum = channelNum;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    OP_LOGD("", "Get HCCL communication channel description success");

    hcclRet = HcclChannelAcquire(hcclHandle_, engine, channelDesc.data(), channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE("", "Acquire HCCL channel failed");
        return HCCL_E_PTR;
    }
    OP_LOGD("", "Acquire HCCL communication channel success");

    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::GetHcclCommResource(const CommEngine &engine, const CommProtocol &protocol,
                                                 Mc2MoeContext *mc2ContextStruct)
{
    OP_LOGD("", "start to get HCCL communication resource");
    if (HcclGetHcclBuffer == nullptr || HcclChannelGetHcclBuffer == nullptr) {
        OP_LOGE("", "HCCL buffer function pointer is null");
        return HCCL_E_PTR;
    }

    uint32_t rankId = mc2ContextStruct->epRankId;
    uint32_t rankSize = mc2ContextStruct->epRankSize;
    std::vector<ChannelHandle> channels;

    HcclResult ret = GetHcclCommChannel(rankSize, rankId, protocol, engine, channels);
    if (ret != HCCL_SUCCESS)
        return ret;
    OP_LOGD("", "Get HCCL communication channel success, channel num is: %zu", channels.size());

    for (uint32_t i = 0; i < rankSize; ++i) {
        void *tempBuffer = nullptr;
        uint64_t bufSize = 0;
        HcclResult hcclRet;

        if (i == rankId) {
            hcclRet = HcclGetHcclBuffer(hcclHandle_, &tempBuffer, &mc2ContextStruct->winSize);
        } else {
            uint32_t idx = (i < rankId) ? i : (i - 1);
            hcclRet = HcclChannelGetHcclBuffer(hcclHandle_, channels[idx], &tempBuffer, &bufSize);
        }

        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE("", "Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return HCCL_E_PTR;
        }

        mc2ContextStruct->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer);
    }
    OP_LOGD("", "Get HCCL CommResource success");
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CreatMc2Context(const std::string &mc2ContextTag, const CommEngine &engine,
                                             const CommProtocol &protocol, void *&ctx, Mc2MoeContext *mc2ContextStruct)
{
    OP_LOGD("", "start to create HCCL context");
    if (HcclEngineCtxCreate == nullptr || HcclGetRankId == nullptr || HcclGetRankSize == nullptr ||
        HcclEngineCtxCopy == nullptr) {
        OP_LOGE("", "MC2 context function pointer is null");
        return HCCL_E_PTR;
    }
    OP_LOGD("", "hcclHandle_ is: %p", hcclHandle_);
    uint64_t ctxSize = sizeof(Mc2MoeContext);
    HcclResult hcclRet = HcclEngineCtxCreate(hcclHandle_, mc2ContextTag.c_str(), engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE("", "Create HCCL context memory failed");
        return HCCL_E_PTR;
    }
    OP_LOGD("", "Create HCCL context success, ctx: %p", ctx);

    hcclRet = HcclGetRankId(hcclHandle_, &mc2ContextStruct->epRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE("", "Get rank ID failed");
        return HCCL_E_PTR;
    }
    OP_LOGD("", "Get rank ID success, rankId is: %u", mc2ContextStruct->epRankId);

    hcclRet = HcclGetRankSize(hcclHandle_, &mc2ContextStruct->epRankSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE("", "Get rank size failed");
        return HCCL_E_PTR;
    }
    OP_LOGD("", "Get rank size success, rankSize is: %u", mc2ContextStruct->epRankSize);

    HcclResult ret = GetHcclCommResource(engine, protocol, mc2ContextStruct);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Get HCCL communication resource failed");
        return ret;
    }

    hcclRet = HcclEngineCtxCopy(hcclHandle_, engine, mc2ContextTag.c_str(), mc2ContextStruct, ctxSize,
                                CTX_DEFAULT_COPY_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE("", "Copy context from host to device failed");
        return HCCL_E_PTR;
    }
    OP_LOGD("", "Copy context from host to device success");
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::GetaclCreateTensorSymbol()
{
    std::string createTensorLibPath;
    GetLibPath(createTensorLibPath);
    const char *libPath = createTensorLibPath.c_str();
    if (libPath == nullptr) {
        OP_LOGE("", "Failed to get aclCreateTensor library path");
        return HCCL_E_PTR;
    }

    void *createHandle = dlopen(libPath, RTLD_NOW | RTLD_GLOBAL);
    if (createHandle == nullptr) {
        OP_LOGE("", "dlopen aclCreateTensor library failed: %s, error: %s", libPath, dlerror());
        return HCCL_E_PTR;
    }

    aclCreateTensor =
        reinterpret_cast<decltype(aclCreateTensor)>(GetHcclLibFunc<void *>(createHandle, "aclCreateTensor"));

    if (aclCreateTensor == nullptr) {
        OP_LOGE("", "Failed to load aclCreateTensor symbol");
        dlclose(createHandle);
        createHandle = nullptr;
        return HCCL_E_PTR;
    }
    OP_LOGI("", "Load aclCreateTensor symbol success");
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CreatMc2ContextTensor(void *ctx, aclTensor *&mc2Context)
{
    OP_LOGD("", "start to create MC2 context ACL tensor");
    if (ctx == nullptr) {
        OP_LOGE("", "Context pointer is null");
        return HCCL_E_PTR;
    }
    auto ret = GetaclCreateTensorSymbol();
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Get aclCreateTensor symbol failed");
        return HCCL_E_PTR;
    }
    OP_LOGD("", "Get aclCreateTensor symbol success");
    uint64_t length = sizeof(Mc2MoeContext);
    int64_t shape[1] = {static_cast<int64_t>(length / sizeof(uint32_t))};
    int64_t strides[1] = {1};

    mc2Context = aclCreateTensor(shape, 1, ACL_INT32, strides, 0, ACL_FORMAT_ND, shape, 1, ctx);
    if (mc2Context == nullptr) {
        OP_LOGE("", "Create ACL tensor failed");
        return HCCL_E_PTR;
    }

    OP_LOGD("", "Create ACL tensor success");
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::GetHcclBufferSize(uint64_t &hcclBuffSize)
{
    if (HcclGetHcclBuffer == nullptr) {
        OP_LOGE("", "HcclGetHcclBuffer is null");
        return HCCL_E_PTR;
    }

    void *tempBuf = nullptr;
    HcclResult ret = HcclGetHcclBuffer(hcclHandle_, &tempBuf, &hcclBuffSize);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Get HCCL buffer size failed");
        return HCCL_E_PTR;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::GetCommProtocol(CommProtocol &protocol)
{
    OP_LOGD("", "start to get HCCL communication protocol");
    uint32_t layerNum = 0;
    uint32_t *layerList = nullptr;
    HcclResult ret = HcclRankGraphGetLayers(hcclHandle_, &layerList, &layerNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Get HCCL layers failed");
        return HCCL_E_PTR;
    }

    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return HCCL_SUCCESS;
    } else {
        uint32_t epRankSize = 0;
        ret = HcclGetRankSize(hcclHandle_, &epRankSize); // 获取通讯域存在多少卡
        if (ret != HCCL_SUCCESS) {
            OP_LOGE("", "Get HCCL rank size failed");
            return HCCL_E_PTR;
        }
        uint32_t netLayer = HCCL_COMM_LAYERS_UB_MEM, srcRankId = 0, dstRankId = 0, netLinkNum = 0;
        CommLink *linksList = nullptr;
        for (uint32_t i = 1; i < epRankSize; ++i) { // 判断卡0是否可以和所有的其他卡通信，并且通信协议是否支持UB_MEM
            dstRankId = i;
            HcclResult hcclRet =
                HcclRankGraphGetLinks(hcclHandle_, netLayer, srcRankId, dstRankId, &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS) {
                OP_LOGE("", "Get HCCL links failed when checking protocol support");
                return HCCL_E_PTR;
            }
            if (netLinkNum == 0) {
                OP_LOGE("", "No available HCCL links found");
                return HCCL_E_PTR;
            }
            bool isFoundUbMemProtocol = false;
            for (uint32_t j = 0; j < netLinkNum; ++j) { // 判断卡0是否可以和所有的其他卡通信，并且通信协议是否支持UB_MEM
                if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
                    isFoundUbMemProtocol = true;
                    break;
                }
            }
            if (!isFoundUbMemProtocol) {
                 OP_LOGE("", "No HCCL links support UB_MEM srcRankID %d, dstRankID %d", srcRankId, dstRankId);
                 return HCCL_E_PTR;
            }
        }
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return HCCL_SUCCESS;
    }
    OP_LOGE("", "Only MTE communication mode is supported");
    return HCCL_E_PTR;
}

HcclResult MC2HcomTopology::GetMc2ContextTensor(const char *groupEp, const char *opName, uint64_t &hcclBuffSize,
                                                 aclTensor *&mc2Context)
{
    OP_LOGD("", "start to get MC2 context tensor, groupEp: %s, opName: %s", groupEp, opName);
    auto instance = GetInstance();
    HcclResult ret = instance.LoadHcclSymbols();
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    CommProtocol protocol;
    Mc2MoeContext contextStruct;
    std::string tag = std::string(groupEp) + opName;

    if (tag.size() > CONTEXT_TAG_MAX_SIZE) {
        OP_LOGE("", "Context tag too long, max: %u, current: %zu", CONTEXT_TAG_MAX_SIZE, tag.size());
        return HCCL_E_PTR;
    }

    ret = instance.GetCommHandle(groupEp);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Get HCCL communication handle failed");
        return ret;
    }

    ret = instance.GetCommProtocol(protocol);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Get HCCL communication protocol failed");
        return ret;
    }

    HcclResult hcclRet = instance.HcclEngineCtxGet(instance.hcclHandle_, tag.c_str(), engine, &ctx, &ctxSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGD("", "Get HCCL engine context failed start to create context");
        ret = instance.CreatMc2Context(tag, engine, protocol, ctx, &contextStruct);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        hcclBuffSize = contextStruct.winSize;
    } else {
        OP_LOGD("", "Get HCCL engine context success");
        ret = instance.GetHcclBufferSize(hcclBuffSize);
        if (ret != HCCL_SUCCESS) {
            OP_LOGE("", "Get HCCL buffer size failed");
            return ret;
        }
    }

    ret = instance.CreatMc2ContextTensor(ctx, mc2Context);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Create MC2 context tensor failed");
        return ret;
    }
    OP_LOGD("", "Get MC2 context tensor success");
    return HCCL_SUCCESS;
}

} // namespace Mc2Hcom