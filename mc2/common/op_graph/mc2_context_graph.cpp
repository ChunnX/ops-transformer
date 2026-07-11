/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_context_graph.cpp
 * \brief MC2 Context management for graph-level fusion pass.
 *        HCCL call chain aligned with mc2/common/op_api/mc2_context.cpp (authoritative).
 *        Differences: outputs ge::GNode (Const) instead of aclTensor.
 */

#include "op_graph/mc2_context_graph.h"
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

#include <cstdlib>
#include <dlfcn.h>
#include "mc2_common_log.h"

namespace {
constexpr uint64_t KOPY_DEFAULT_CTX_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t EP_RANK_OFFSET_STEP = 1024;
}  // namespace

namespace ops {

Mc2ContextGraph::~Mc2ContextGraph()
{
    if (hcclLibHandle_ != nullptr) {
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
    }
}

Mc2ContextGraph &Mc2ContextGraph::GetInstance()
{
    static Mc2ContextGraph instance;
    return instance;
}

const std::string Mc2ContextGraph::GetLibPath()
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OPS_LOG_E("Mc2ContextGraph", "Ascend home path doesn't exist.");
        return "";
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
#endif
    std::string fullPath = std::string(ascendPath) + hcclPathPostfix;
    OPS_LOG_I("Mc2ContextGraph", "Loading lib in path %s.", fullPath.c_str());
    return fullPath;
}

template <typename T>
T Mc2ContextGraph::GetHcclLibFunc(void *handle, const std::string &funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        OPS_LOG_E("Mc2ContextGraph", "Load func=%s error=%s in lib hccl failed.", funcName.c_str(), dlerror());
    }
    return func;
}

ge::graphStatus Mc2ContextGraph::LoadHcclSymbols()
{
    if (hcclLibHandle_ != nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string libPath = GetLibPath();
    if (libPath.empty()) {
        OPS_LOG_E("Mc2ContextGraph", "Failed to get HCCL library path");
        return ge::GRAPH_FAILED;
    }
    hcclLibHandle_ = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        OPS_LOG_E("Mc2ContextGraph", "dlopen HCCL library failed: %s, error: %s", libPath.c_str(), dlerror());
        return ge::GRAPH_FAILED;
    }

    struct SymbolInfo {
        void **ptr;
        const char *name;
    };
    SymbolInfo symbols[] = {
        {reinterpret_cast<void **>(&HcomGetCommHandleByGroup), "HcomGetCommHandleByGroup"},
        {reinterpret_cast<void **>(&HcclRankGraphGetLinks), "HcclRankGraphGetLinks"},
        {reinterpret_cast<void **>(&HcclRankGraphGetLayers), "HcclRankGraphGetLayers"},
        {reinterpret_cast<void **>(&HcclRankGraphGetRankSizeByLayer), "HcclRankGraphGetRankSizeByLayer"},
        {reinterpret_cast<void **>(&HcclChannelAcquire), "HcclChannelAcquire"},
        {reinterpret_cast<void **>(&HcclGetHcclBuffer), "HcclGetHcclBuffer"},
        {reinterpret_cast<void **>(&HcclChannelGetHcclBuffer), "HcclChannelGetHcclBuffer"},
        {reinterpret_cast<void **>(&HcclEngineCtxCreate), "HcclEngineCtxCreate"},
        {reinterpret_cast<void **>(&HcclEngineCtxCopy), "HcclEngineCtxCopy"},
        {reinterpret_cast<void **>(&HcclGetRankId), "HcclGetRankId"},
        {reinterpret_cast<void **>(&HcclGetRankSize), "HcclGetRankSize"},
        {reinterpret_cast<void **>(&HcclRankGraphGetRanksByLayer), "HcclRankGraphGetRanksByLayer"},
    };
    for (auto &sym : symbols) {
        *(sym.ptr) = GetHcclLibFunc<void *>(hcclLibHandle_, sym.name);
        if (*(sym.ptr) == nullptr) {
            OPS_LOG_E("Mc2ContextGraph", "Failed to load %s symbol", sym.name);
            dlclose(hcclLibHandle_);
            hcclLibHandle_ = nullptr;
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::GetCommHandle(const char *groupEp, HcclComm &hcclHandle)
{
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if (ret != HCCL_SUCCESS) {
        OPS_LOG_E("Mc2ContextGraph", "Get HCCL handle failed, groupEp: %s", groupEp);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::GetHcclCommLink(const HcclComm &hcclHandle, uint32_t netLayerId,
                                                  uint32_t srcRankId, uint32_t dstRankId,
                                                  const CommProtocol &protocol, CommLink *&links)
{
    CommLink *linksList = nullptr;
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS || netLinkNum == 0) {
        OPS_LOG_E("Mc2ContextGraph", "Get HCCL Communication link failed");
        return ge::GRAPH_FAILED;
    }
    uint32_t index = 0;
    for (; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == protocol) {
            links = &linksList[index];
            break;
        }
    }
    if (index == netLinkNum) {
        OPS_LOG_E("Mc2ContextGraph", "No matching protocol found");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::GetRankSizePerServer(const HcclComm &hcclHandle, uint32_t netLayers)
{
    auto hcclRet = HcclRankGraphGetRankSizeByLayer(hcclHandle, netLayers, &rankSizePerServer_);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E("Mc2ContextGraph", "Get rank size per server failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::InitHcclChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                                 const CommProtocol &protocol,
                                                 std::vector<HcclChannelDesc> &channelDesc)
{
    uint32_t channelNum = channelDesc.size();
    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E("Mc2ContextGraph", "HCCL channel init failed");
        return ge::GRAPH_FAILED;
    }
    uint32_t netLayerNum = 0;
    uint32_t *netLayerList = nullptr;
    auto hcclRetLayers = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRetLayers != HCCL_SUCCESS || netLayerNum == 0) {
        OPS_LOG_E("Mc2ContextGraph", "Get HCCL net layers failed");
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < rankDim; ++i) {
        if (i == srcRankId) {
            continue;
        }
        uint32_t dstRank = i;
        uint32_t channelId = (i > srcRankId) ? (i - 1) : i;
        CommLink *links = nullptr;
        uint32_t layerId = (netLayerNum == 1) ? netLayerList[HCCL_COMM_LAYERS_UB_MEM] : layerMap_[dstRank];
        auto ret = GetHcclCommLink(hcclHandle, layerId, srcRankId, dstRank, protocol, links);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        channelDesc[channelId].channelProtocol = protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::GetHcclCommChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                                    const CommProtocol &protocol, const CommEngine &engine,
                                                    std::vector<ChannelHandle> &channels)
{
    uint32_t channelNum = rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);

    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    auto hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
    auto ret = GetRankSizePerServer(hcclHandle, netLayers);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = InitHcclChannel(hcclHandle, rankDim, srcRankId, protocol, channelDesc);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    hcclRet = HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E("Mc2ContextGraph", "Acquire HCCL channel failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::FillContextBuffer(const HcclComm &hcclHandle, const CommEngine &engine,
                                                   Mc2Aclnn::Mc2MoeContext *context, uint32_t rankId,
                                                   const std::vector<ChannelHandle> &channels)
{
    for (uint32_t i = 0; i < epRankSize_; ++i) {
        void *tempBuffer = nullptr;
        uint64_t bufSize = 0;
        HcclResult hcclRet;
        if (i == rankId) {
            hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize_);
        } else {
            uint32_t idx = (i < rankId) ? i : (i - 1);
            hcclRet = HcclChannelGetHcclBuffer(hcclHandle, channels[idx], &tempBuffer, &bufSize);
        }
        if (hcclRet != HCCL_SUCCESS || tempBuffer == nullptr) {
            OPS_LOG_E("Mc2ContextGraph", "Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return ge::GRAPH_FAILED;
        }
        context->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer) + i * EP_RANK_OFFSET_STEP;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::GetHcclCommResource(const HcclComm &hcclHandle, const CommEngine &engine,
                                                     const CommProtocol &protocol,
                                                     Mc2Aclnn::Mc2MoeContext *context)
{
    uint32_t rankId = context->epRankId;
    std::vector<ChannelHandle> channels;
    auto ret = GetHcclCommChannel(hcclHandle, epRankSize_, rankId, protocol, engine, channels);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    context->rankSizePerServer = rankSizePerServer_;
    ret = FillContextBuffer(hcclHandle, engine, context, rankId, channels);
    return ret;
}

ge::graphStatus Mc2ContextGraph::CreatMc2Context(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                                 const CommEngine &engine, const CommProtocol &protocol,
                                                 Mc2Aclnn::Mc2MoeContext *context, void *&ctx, uint64_t &hcclBuffSize)
{
    uint64_t ctxSize = sizeof(Mc2Aclnn::Mc2MoeContext);
    auto hcclRet = HcclEngineCtxCreate(hcclHandle, mc2ContextTag.c_str(), engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E("Mc2ContextGraph", "Create HCCL context memory failed");
        return ge::GRAPH_FAILED;
    }
    hcclRet = HcclGetRankId(hcclHandle, &context->epRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E("Mc2ContextGraph", "Get rank ID failed");
        return ge::GRAPH_FAILED;
    }
    hcclRet = HcclGetRankSize(hcclHandle, &epRankSize_);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E("Mc2ContextGraph", "Get rank size failed");
        return ge::GRAPH_FAILED;
    }
    auto ret = GetHcclCommResource(hcclHandle, engine, protocol, context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    hcclRet = HcclEngineCtxCopy(hcclHandle, engine, mc2ContextTag.c_str(), context, ctxSize, KOPY_DEFAULT_CTX_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E("Mc2ContextGraph", "Copy context from host to device failed");
        return ge::GRAPH_FAILED;
    }
    hcclBuffSize = hcclBuffSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::CheckLinks(uint32_t &netLinkNum, CommLink *linksList)
{
    if (linksList == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus Mc2ContextGraph::CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList,
                                                      uint32_t &layerNum)
{
    uint32_t srcRankId = 0;
    auto hcclRet = HcclGetRankId(hcclHandle, &srcRankId);
    if (hcclRet != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        uint32_t rankNumInLayer = 0;
        uint32_t *rankIdLists = nullptr;
        hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if (hcclRet != HCCL_SUCCESS || rankNumInLayer == 0 || rankIdLists == nullptr) {
            return ge::GRAPH_FAILED;
        }
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if (rankIdLists[rankId] == srcRankId || layerMap_.find(rankIdLists[rankId]) != layerMap_.end()) {
                continue;
            }
            uint32_t netLinkNum = 0;
            CommLink *linksList = nullptr;
            hcclRet = HcclRankGraphGetLinks(hcclHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                            &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS || netLinkNum == 0 || linksList == nullptr) {
                return ge::GRAPH_FAILED;
            }
            if (CheckLinks(netLinkNum, linksList) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            layerMap_[rankIdLists[rankId]] = layerList[layerIndex];
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol)
{
    uint32_t layerNum = 0;
    uint32_t *layerList = nullptr;
    auto hcclRet = HcclRankGraphGetLayers(hcclHandle, &layerList, &layerNum);
    if (hcclRet != HCCL_SUCCESS || layerNum == 0) {
        return ge::GRAPH_FAILED;
    }
    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return ge::GRAPH_SUCCESS;
    }
    auto status = CheckProtocolSupport(hcclHandle, layerList, layerNum);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::ValidateContextTag(const std::string &mc2ContextTag)
{
    if (mc2ContextTag.size() > MAX_CONTEXT_TAG_SIZE) {
        OPS_LOG_E("Mc2ContextGraph", "Mc2ContextTag too long, max %llu, got %zu", MAX_CONTEXT_TAG_SIZE,
                mc2ContextTag.size());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2ContextGraph::GetMc2ContextData(const std::string &groupEp, const std::string &opName,
                                                   Mc2Aclnn::Mc2MoeContext &context, int64_t &hcclBuffSize)
{
    std::string mc2ContextTag = groupEp + opName;
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    hcclBuffSize = 0;

    auto &manager = GetInstance();

    // Cache: same process, same tag (dispatch+combine share) → reuse cached struct + buffSize
    std::string cacheKey = mc2ContextTag;
    auto cacheIt = manager.contextCache_.find(cacheKey);
    if (cacheIt != manager.contextCache_.end()) {
        context = cacheIt->second.context;
        hcclBuffSize = cacheIt->second.hcclBuffSize;
        OPS_LOG_I("Mc2ContextGraph", "Reusing cached context for tag %s, hcclBuffSize=%lld", mc2ContextTag.c_str(),
                hcclBuffSize);
        return ge::GRAPH_SUCCESS;
    }

    // Graph mode and aclnn do not coexist — no HCCL engine cache to check.
    // Always create new context via CreatMc2Context (aligned with mc2_context.cpp non-cache path).
    auto status = manager.LoadHcclSymbols();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = manager.ValidateContextTag(mc2ContextTag);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    HcclComm hcclHandle;
    status = manager.GetCommHandle(groupEp.c_str(), hcclHandle);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    CommProtocol protocol;
    status = manager.GetCommProtocol(hcclHandle, protocol);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    void *ctx = nullptr;
    uint64_t buffSize = 0;
    status = manager.CreatMc2Context(hcclHandle, mc2ContextTag, engine, protocol, &context, ctx, buffSize);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    hcclBuffSize = static_cast<int64_t>(buffSize);

    ContextCacheEntry entry;
    entry.context = context;
    entry.hcclBuffSize = hcclBuffSize;
    manager.contextCache_[cacheKey] = entry;
    return ge::GRAPH_SUCCESS;
}

template void *Mc2ContextGraph::GetHcclLibFunc<void *>(void *handle, const std::string &funcName);

}  // namespace ops
#endif  // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
