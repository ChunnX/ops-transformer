/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_moe_context.cpp
 * \brief Moe Context Manager implementation for graph fusion in mc2 operator repository
 */
#include "mc2_moe_context.h"

#if defined(HCOMM_VERSION_NUM) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

#include <cstdlib>
#include <dlfcn.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"

namespace {
constexpr uint64_t CTX_COPY_DEFAULT_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t MAX_EP_RANK_SIZE = 1024;
} // namespace

namespace ops {

Mc2MoeContextManager::~Mc2MoeContextManager()
{
    if (hcclLibHandle_ != nullptr) {
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
    }
}

Mc2MoeContextManager &Mc2MoeContextManager::GetInstance()
{
    static Mc2MoeContextManager instance;
    return instance;
}

const std::string Mc2MoeContextManager::GetLibPath()
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        return "";
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
#endif
    std::string fullPath = std::string(ascendPath) + hcclPathPostfix;
    return fullPath;
}

template <typename T>
T Mc2MoeContextManager::GetHcclLibFunc(void *handle, const std::string &funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        const char *dlErr = dlerror();
    }
    return func;
}

ge::graphStatus Mc2MoeContextManager::LoadSymbol(void **ptr, const char *name)
{
    *ptr = GetHcclLibFunc<void *>(hcclLibHandle_, name);
    if (*ptr == nullptr) {
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::LoadHcclSymbols()
{
    const std::string libPath = GetLibPath();
    if (libPath.empty()) {
        return ge::GRAPH_FAILED;
    }

    hcclLibHandle_ = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        const char *dlErr = dlerror();
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
        {reinterpret_cast<void **>(&HcclEngineCtxGet), "HcclEngineCtxGet"},
        {reinterpret_cast<void **>(&HcclEngineCtxCopy), "HcclEngineCtxCopy"},
        {reinterpret_cast<void **>(&HcclGetRankId), "HcclGetRankId"},
        {reinterpret_cast<void **>(&HcclGetRankSize), "HcclGetRankSize"},
        {reinterpret_cast<void **>(&HcclRankGraphGetRanksByLayer), "HcclRankGraphGetRanksByLayer"},
    };

    for (auto &sym : symbols) {
        if (LoadSymbol(sym.ptr, sym.name) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetCommHandle(const char *groupEp, HcclComm &hcclHandle)
{
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if (ret != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetHcclCommLink(const HcclComm &hcclHandle, const LinkQueryParams &params,
                                                      CommLink *&links)
{
    CommLink *linksList = nullptr;
    uint32_t netLinkNum = 0;

    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, params.netLayerIndex, params.srcRankId, params.dstRankId,
                                         &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (netLinkNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t index = 0;
    for (; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == params.protocol) {
            links = &linksList[index];
            break;
        }
    }

    if (index == netLinkNum) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetRankSizePerServer(const HcclComm &hcclHandle, uint32_t netLayers)
{
    auto hcclRet = HcclRankGraphGetRankSizeByLayer(hcclHandle, netLayers, &rankSizePerServer_);
    if (hcclRet != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::SetupChannelDesc(const HcclComm &hcclHandle, const ChannelInitParams &params,
                                                       uint32_t *netLayerList, uint32_t netLayerNum,
                                                       std::vector<HcclChannelDesc> &channelDesc)
{
    for (uint32_t i = 0; i < params.rankDim; ++i) {
        if (i == params.srcRankId) {
            continue;
        }

        uint32_t dstRank = i;
        uint32_t channelId = (i > params.srcRankId) ? (i - 1) : i;
        CommLink *links = nullptr;
        uint32_t netLayerIndex = netLayerNum == 1 ? netLayerList[HCCL_COMM_LAYERS_UB_MEM] : layerMap[dstRank];
        LinkQueryParams query{netLayerIndex, params.srcRankId, dstRank, params.protocol};
        auto ret = GetHcclCommLink(hcclHandle, query, links);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }

        channelDesc[channelId].channelProtocol = params.protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::InitHcclChannel(const HcclComm &hcclHandle, const ChannelInitParams &params,
                                                      std::vector<HcclChannelDesc> &channelDesc)
{
    uint32_t channelNum = channelDesc.size();
    uint32_t netLayerIndex = 0;
    uint32_t netLayerNum = 0;
    uint32_t *netLayerList = nullptr;

    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS || netLayerNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
    auto status = GetRankSizePerServer(hcclHandle, netLayers);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = SetupChannelDesc(hcclHandle, params, netLayerList, netLayerNum, channelDesc);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetHcclCommChannel(const HcclComm &hcclHandle, const ChannelAcquireParams &params,
                                                         std::vector<ChannelHandle> &channels)
{
    uint32_t channelNum = params.rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);

    ChannelInitParams initParams{params.rankDim, params.srcRankId, params.protocol};
    auto status = InitHcclChannel(hcclHandle, initParams, channelDesc);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    auto hcclRet = HcclChannelAcquire(hcclHandle, params.engine, channelDesc.data(), channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::FillContextBuffer(const HcclComm &hcclHandle, const CommEngine &engine,
                                                        Mc2MoeContext *context, uint32_t rankId,
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

        if (hcclRet != HCCL_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        context->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetHcclCommResource(const HcclComm &hcclHandle, const CommEngine &engine,
                                                          const CommProtocol &protocol, Mc2MoeContext *context)
{
    uint32_t rankId = context->epRankId;
    std::vector<ChannelHandle> channels;

    ChannelAcquireParams params{epRankSize_, rankId, protocol, engine};
    auto status = GetHcclCommChannel(hcclHandle, params, channels);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    context->rankSizePerServer = rankSizePerServer_;

    status = FillContextBuffer(hcclHandle, engine, context, rankId, channels);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::CreatMc2Context(const ContextCreateParams &params, Mc2MoeContext *context)
{
    void *ctx = nullptr;
    uint64_t ctxSize = sizeof(Mc2MoeContext);

    auto hcclRet = HcclEngineCtxCreate(params.hcclHandle, params.mc2ContextTag.c_str(), params.engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    hcclRet = HcclGetRankId(params.hcclHandle, &context->epRankId);
    if (hcclRet != HCCL_SUCCESS || context->epRankId >= MAX_EP_RANK_SIZE) {
        return ge::GRAPH_FAILED;
    }

    hcclRet = HcclGetRankSize(params.hcclHandle, &epRankSize_);
    if (hcclRet != HCCL_SUCCESS || epRankSize_ == 0 || epRankSize_ >= MAX_EP_RANK_SIZE) {
        return ge::GRAPH_FAILED;
    }

    auto status = GetHcclCommResource(params.hcclHandle, params.engine, params.protocol, context);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    hcclRet = HcclEngineCtxCopy(params.hcclHandle, params.engine, params.mc2ContextTag.c_str(), context, ctxSize,
                                CTX_COPY_DEFAULT_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::CheckLinks(uint32_t &netLinkNum, CommLink *linksList)
{
    bool isFoundUbMemProtocol = false;
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            isFoundUbMemProtocol = true;
            break;
        }
    }
    return isFoundUbMemProtocol ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus Mc2MoeContextManager::CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList,
                                                           uint32_t &layerNum)
{
    uint32_t srcRankId = 0;
    uint32_t dstRankId = 0;
    uint32_t netLinkNum = 0;
    uint32_t rankNumInLayer = 0;
    uint32_t *rankIdLists = nullptr;
    CommLink *linksList = nullptr;

    auto hcclRet = HcclGetRankId(hcclHandle, &srcRankId);
    if (hcclRet != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if (hcclRet != HCCL_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if (rankIdLists[rankId] == srcRankId || layerMap.find(rankIdLists[rankId]) != layerMap.end()) {
                continue;
            }
            hcclRet = HcclRankGraphGetLinks(hcclHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                            &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (netLinkNum == 0) {
                return ge::GRAPH_FAILED;
            }
            if (CheckLinks(netLinkNum, linksList) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            layerMap[rankIdLists[rankId]] = layerList[layerIndex];
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol)
{
    uint32_t layerNum = 0;
    uint32_t *layerList = nullptr;

    auto ret = HcclRankGraphGetLayers(hcclHandle, &layerList, &layerNum);
    if (ret != HCCL_SUCCESS || layerNum == 0) {
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

ge::graphStatus Mc2MoeContextManager::ValidateContextTag(const std::string &mc2ContextTag)
{
    if (mc2ContextTag.size() > MAX_CONTEXT_TAG_SIZE) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetMc2Context(const ContextAcquireParams &params, Mc2MoeContext *context,
                                                    int64_t &hcclBuffSize)
{
    ContextCreateParams createParams{params.hcclHandle, params.engine, params.protocol, params.mc2ContextTag};
    auto status = CreatMc2Context(createParams, context);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    hcclBuffSize = static_cast<int64_t>(hcclBuffSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetHcclContext(const std::string &graphName, const std::string &groupEp,
                                                     const std::string &nodeName, Mc2MoeContext *context,
                                                     int64_t &hcclBuffSize)
{
    auto status = LoadHcclSymbols();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    CommProtocol protocol;
    std::string mc2ContextTag = graphName + groupEp + nodeName;

    status = ValidateContextTag(mc2ContextTag);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    HcclComm hcclHandle;
    status = GetCommHandle(groupEp.c_str(), hcclHandle);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    status = GetCommProtocol(hcclHandle, protocol);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    ContextAcquireParams params{hcclHandle, engine, protocol, mc2ContextTag};
    status = GetMc2Context(params, context, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::ExtractNodeInfo(const ge::NodePtr &originalNode, const std::string &epName,
                                                      std::string &groupEp)
{
    auto opDesc = originalNode->GetOpDesc();
    if (opDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (!ge::AttrUtils::GetStr(opDesc, epName.c_str(), groupEp) || groupEp.empty()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::CheckCache(const std::string &uniqueKey, ge::NodePtr &constNode,
                                                 int64_t &hcclBuffSize)
{
    auto nodeIt = contextNodeMapA5_.find(uniqueKey);
    auto bufferIt = bufferSizeMapA5_.find(uniqueKey);
    if (nodeIt != contextNodeMapA5_.end() && bufferIt != bufferSizeMapA5_.end()) {
        constNode = nodeIt->second;
        hcclBuffSize = bufferIt->second;
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus Mc2MoeContextManager::UpdateCache(const std::string &uniqueKey, const ge::NodePtr &constNode,
                                                  int64_t hcclBuffSize)
{
    contextNodeMapA5_[uniqueKey] = constNode;
    bufferSizeMapA5_[uniqueKey] = hcclBuffSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::CreateContextNode(ge::ComputeGraph &graph, const Mc2MoeContext &context,
                                                        ge::NodePtr &constNode)
{
    ge::GeShape shape({static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t))});
    ge::GeTensorDesc tensorDesc = ge::GeTensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    auto geTensor =
        std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<const uint8_t *>(&context), sizeof(Mc2MoeContext));

    ge::OpDescPtr constOpDesc = ge::OpDescUtils::CreateConstOp(geTensor);
    if (constOpDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    constNode = graph.AddNode(constOpDesc);
    if (constNode == nullptr) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetMc2ContextA5(ge::ComputeGraph &graph, const std::string &nodeName,
                                                      const std::string &epName, const ge::NodePtr &originalNode,
                                                      ge::NodePtr &constNode, int64_t &hcclBuffSize)
{
    if (nodeName.empty() || epName.empty()) {
        return ge::GRAPH_FAILED;
    }

    Mc2MoeContextManager &manager = GetInstance();

    std::string groupEp;
    auto status = manager.ExtractNodeInfo(originalNode, epName, groupEp);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    std::string graphName = graph.GetName();
    if (graphName.empty()) {
        return ge::GRAPH_FAILED;
    }

    std::string uniqueKey = graphName + "_" + groupEp + "_" + nodeName;
    status = manager.CheckCache(uniqueKey, constNode, hcclBuffSize);
    if (status == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    Mc2MoeContext context{0};
    status = manager.GetHcclContext(graphName, groupEp, nodeName, &context, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = manager.CreateContextNode(graph, context, constNode);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = manager.UpdateCache(uniqueKey, constNode, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::CheckContextCache(const std::string &uniqueKey, Mc2MoeContext &context,
                                                        int64_t &hcclBuffSize)
{
    auto dataIt = contextDataMapA5_.find(uniqueKey);
    auto bufferIt = bufferSizeMapA5_.find(uniqueKey);
    if (dataIt != contextDataMapA5_.end() && bufferIt != bufferSizeMapA5_.end()) {
        context = dataIt->second;
        hcclBuffSize = bufferIt->second;
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus Mc2MoeContextManager::UpdateContextCache(const std::string &uniqueKey, const Mc2MoeContext &context,
                                                         int64_t hcclBuffSize)
{
    contextDataMapA5_[uniqueKey] = context;
    bufferSizeMapA5_[uniqueKey] = hcclBuffSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetMc2ContextData(const std::string &nodeName, const std::string &groupEp,
                                                        Mc2MoeContext &context, int64_t &hcclBuffSize)
{
    if (nodeName.empty() || groupEp.empty()) {
        return ge::GRAPH_FAILED;
    }

    Mc2MoeContextManager &manager = GetInstance();

    std::string uniqueKey = groupEp + "_" + nodeName;
    auto status = manager.CheckContextCache(uniqueKey, context, hcclBuffSize);
    if (status == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    status = manager.LoadHcclSymbols();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    std::string mc2ContextTag = "fusion_" + groupEp + "_" + nodeName;
    status = manager.ValidateContextTag(mc2ContextTag);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    HcclComm hcclHandle;
    status = manager.GetCommHandle(groupEp.c_str(), hcclHandle);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    CommProtocol protocol;
    status = manager.GetCommProtocol(hcclHandle, protocol);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    CommEngine engine = CommEngine::COMM_ENGINE_AIV;

    HcclResult hcclRet = manager.HcclGetRankId(hcclHandle, &context.epRankId);
    if (hcclRet != HCCL_SUCCESS || context.epRankId >= MAX_EP_RANK_SIZE) {
        return ge::GRAPH_SUCCESS;
    }

    hcclRet = manager.HcclGetRankSize(hcclHandle, &manager.epRankSize_);
    if (hcclRet != HCCL_SUCCESS || manager.epRankSize_ == 0 || manager.epRankSize_ >= MAX_EP_RANK_SIZE) {
        return ge::GRAPH_SUCCESS;
    }

    std::vector<ChannelHandle> channels;
    ChannelAcquireParams params{manager.epRankSize_, context.epRankId, protocol, engine};
    status = manager.GetHcclCommChannel(hcclHandle, params, channels);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    uint32_t netLayerNum = 0;
    uint32_t *netLayerList = nullptr;
    hcclRet = manager.HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS || netLayerNum == 0) {
        return ge::GRAPH_SUCCESS;
    }

    uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
    status = manager.GetRankSizePerServer(hcclHandle, netLayers);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    context.rankSizePerServer = manager.rankSizePerServer_;

    for (uint32_t i = 0; i < manager.epRankSize_; ++i) {
        void *tempBuffer = nullptr;
        uint64_t bufSize = 0;

        if (i == context.epRankId) {
            hcclRet = manager.HcclGetHcclBuffer(hcclHandle, &tempBuffer, &manager.hcclBuffSize_);
        } else {
            uint32_t idx = (i < context.epRankId) ? i : (i - 1);
            hcclRet = manager.HcclChannelGetHcclBuffer(hcclHandle, channels[idx], &tempBuffer, &bufSize);
        }

        if (hcclRet != HCCL_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }

        context.epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer);
    }

    hcclBuffSize = static_cast<int64_t>(manager.hcclBuffSize_);

    status = manager.UpdateContextCache(uniqueKey, context, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeContextManager::GetMc2ContextDataFromNode(const ge::GNode &node, const std::string &epName,
                                                                Mc2MoeContext &context, int64_t &hcclBuffSize)
{
    ge::AscendString groupEpStr;
    if (!node.GetAttr(epName.c_str(), groupEpStr)) {
        return ge::GRAPH_FAILED;
    }

    std::string groupEp(groupEpStr.GetString());
    std::string nodeName = node.GetName().GetString();

    return GetMc2ContextData(nodeName, groupEp, context, hcclBuffSize);
}

ge::graphStatus GetMc2ContextDataLite(const std::string &nodeName, const std::string &groupEp,
                                      Mc2MoeContextLite &context, int64_t &hcclBuffSize)
{
    Mc2MoeContext fullContext{0};
    auto status = Mc2MoeContextManager::GetMc2ContextData(nodeName, groupEp, fullContext, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    context.epRankId = fullContext.epRankId;
    context.rankSizePerServer = fullContext.rankSizePerServer;
    context.kfcContextAddr = fullContext.kfcContextAddr;
    for (uint32_t i = 0; i < HCCL_MAX_RANK_SIZE; ++i) {
        context.epHcclBuffer_[i] = fullContext.epHcclBuffer_[i];
    }

    return ge::GRAPH_SUCCESS;
}

template void *Mc2MoeContextManager::GetHcclLibFunc<void *>(void *handle, const std::string &funcName);

} // namespace ops
#endif // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION