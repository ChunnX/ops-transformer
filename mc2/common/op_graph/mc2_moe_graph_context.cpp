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
 * \file mc2_moe_graph_context.cpp
 * \brief MC2 MoE Context Manager implementation for graph-level fusion pass
 */
#include "op_graph/mc2_moe_graph_context.h"
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

#include <cstdlib>
#include <climits>
#include <dlfcn.h>
#include <atomic>
#include <thread>
#include "graph/operator_factory.h"
#include "mc2_common_log.h"

namespace {

constexpr uint64_t CTX_COPY_DEFAULT_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t MAX_EP_RANK_SIZE = 1024;
constexpr uint32_t EP_RANK_OFFSET_STEP = 8192;

}  // namespace

namespace ops {

Mc2MoeGraphContext::~Mc2MoeGraphContext()
{
    if (hcclLibHandle_ != nullptr) {
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
    }
    ResetAllFunctionPointers();
}

void Mc2MoeGraphContext::ResetAllFunctionPointers()
{
    HcomGetCommHandleByGroup = nullptr;
    HcclRankGraphGetLinks = nullptr;
    HcclRankGraphGetLayers = nullptr;
    HcclRankGraphGetRankSizeByLayer = nullptr;
    HcclChannelAcquire = nullptr;
    HcclGetHcclBuffer = nullptr;
    HcclChannelGetHcclBuffer = nullptr;
    HcclEngineCtxCreate = nullptr;
    HcclEngineCtxGet = nullptr;
    HcclEngineCtxCopy = nullptr;
    HcclEngineCtxDestroy = nullptr;
    HcclGetRankId = nullptr;
    HcclGetRankSize = nullptr;
    HcclRankGraphGetRanksByLayer = nullptr;
}

Mc2MoeGraphContext &Mc2MoeGraphContext::GetInstance()
{
    static Mc2MoeGraphContext instance;
    return instance;
}

const std::string Mc2MoeGraphContext::GetLibPath()
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OPS_LOG_E(LOG_TAG, "Ascend home path doesn't exist.");
        return "";
    }
    std::string fullPath = std::string(ascendPath) + "/lib64/libhccl_fwk.so";
    OPS_LOG_D(LOG_TAG, "Loading lib in path %s.", fullPath.c_str());
    return fullPath;
}

template <typename T>
T Mc2MoeGraphContext::GetHcclLibFunc(void *handle, const std::string &funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        const char *dlErr = dlerror();
        OPS_LOG_E(LOG_TAG, "Load func=%s error=%s in lib hccl failed.", funcName.c_str(), dlErr ? dlErr : "unknown");
    }
    return func;
}

ge::graphStatus Mc2MoeGraphContext::LoadSymbol(void **ptr, const char *name)
{
    *ptr = GetHcclLibFunc<void *>(hcclLibHandle_, name);
    if (*ptr == nullptr) {
        OPS_LOG_E(LOG_TAG, "Failed to load %s symbol", name);
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
        ResetAllFunctionPointers();
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::LoadHcclSymbols()
{
    if (hcclLibHandle_ != nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string libPath = GetLibPath();
    if (libPath.empty()) {
        OPS_LOG_E(LOG_TAG, "Failed to get HCCL library path");
        return ge::GRAPH_FAILED;
    }
    char resolvedPath[PATH_MAX];
    if (realpath(libPath.c_str(), resolvedPath) == nullptr) {
        OPS_LOG_E(LOG_TAG, "realpath failed for HCCL library: %s, errno=%d", libPath.c_str(), errno);
        return ge::GRAPH_FAILED;
    }
    hcclLibHandle_ = dlopen(resolvedPath, RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        const char *dlErr = dlerror();
        OPS_LOG_E(LOG_TAG, "dlopen HCCL library failed: %s, error: %s", libPath.c_str(), dlErr ? dlErr : "unknown");
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
        {reinterpret_cast<void **>(&HcclEngineCtxDestroy), "HcclEngineCtxDestroy"},
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

ge::graphStatus Mc2MoeGraphContext::GetCommHandle(const char *groupEp, HcclComm &hcclHandle)
{
    OPS_LOG_I(LOG_TAG, "Get HCCL comm handle, groupEp: %s", groupEp);
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if (ret != HCCL_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Get HCCL handle failed, groupEp: %s", groupEp);
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(LOG_TAG, "Get HCCL comm handle success");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::GetHcclCommLink(const HcclComm &hcclHandle, const LinkQueryParams &params,
                                                    CommLink *&links)
{
    OPS_LOG_I(LOG_TAG, "Start to get HCCL communication link");
    CommLink *linksList = nullptr;
    uint32_t netLinkNum = 0;

    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, params.netLayerIndex, params.srcRankId, params.dstRankId,
                                         &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Get HCCL Communication link failed");
        return ge::GRAPH_FAILED;
    }
    if ((netLinkNum == 0) || (linksList == nullptr)) {
        OPS_LOG_E(LOG_TAG, "The Net Link Is nullptr or count is 0, netLinkNum=%u", netLinkNum);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(LOG_TAG, "Get HCCL Rank Links Success Links Num is: %u", netLinkNum);
    uint32_t index = 0;
    for (; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == params.protocol) {
            links = &linksList[index];
            break;
        }
    }

    if (index == netLinkNum) {
        OPS_LOG_E(LOG_TAG, "No matching communication protocol found in HCCL links, protocol is %d, "
                  "srcRankId is %u, dstRankId is %u, netLayerIndex is %u",
                  params.protocol, params.srcRankId, params.dstRankId, params.netLayerIndex);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_I(LOG_TAG, "Get HCCL communication link success, protocol is: %d", params.protocol);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::GetRankSizePerServer(const HcclComm &hcclHandle, uint32_t netLayers)
{
    auto hcclRet = HcclRankGraphGetRankSizeByLayer(hcclHandle, netLayers, &rankSizePerServer_);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Get HCCL rank size per server failed");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(LOG_TAG, "Get HCCL rank size per server success, rankSizePerServer_ is: %u", rankSizePerServer_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::SetupChannelDesc(const HcclComm &hcclHandle, const ChannelInitParams &params,
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
        uint32_t netLayerIndex =
            netLayerNum == 1 ? netLayerList[HCCL_COMM_LAYERS_UB_MEM] : layerMap_[dstRank];
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

ge::graphStatus Mc2MoeGraphContext::InitHcclChannel(const HcclComm &hcclHandle, const ChannelInitParams &params,
                                                    std::vector<HcclChannelDesc> &channelDesc)
{
    uint32_t channelNum = channelDesc.size();
    uint32_t netLayerNum = 0;
    uint32_t *netLayerList = nullptr;

    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "HCCL channel init failed");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(LOG_TAG, "HCCL channel init success");

    hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if ((hcclRet != HCCL_SUCCESS) || (netLayerNum == 0) || (netLayerList == nullptr)) {
        OPS_LOG_E(LOG_TAG, "Get HCCL GetNetLayers failed");
        return ge::GRAPH_FAILED;
    }

    uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
    auto status = GetRankSizePerServer(hcclHandle, netLayers);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Get HCCL GetRankSizePerServer failed");
        return status;
    }

    status = SetupChannelDesc(hcclHandle, params, netLayerList, netLayerNum, channelDesc);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Get HCCL SetupChannelDesc failed");
        return status;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::GetHcclCommChannel(const HcclComm &hcclHandle, const ChannelAcquireParams &params,
                                                       std::vector<ChannelHandle> &channels)
{
    OPS_LOG_I(LOG_TAG, "Start to get HCCL communication channel");
    if ((params.rankDim == 0) || (params.rankDim >= MAX_EP_RANK_SIZE)) {
        OPS_LOG_E(LOG_TAG, "Invalid rankDim: %u", params.rankDim);
        return ge::GRAPH_FAILED;
    }
    uint32_t channelNum = params.rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);

    ChannelInitParams initParams{params.rankDim, params.srcRankId, params.protocol};
    auto status = InitHcclChannel(hcclHandle, initParams, channelDesc);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Get HCCL InitHcclChannel failed");
        return status;
    }

    auto hcclRet = HcclChannelAcquire(hcclHandle, params.engine, channelDesc.data(), channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Acquire HCCL channel failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::FillContextBuffer(const HcclComm &hcclHandle, const CommEngine &engine,
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

        if ((hcclRet != HCCL_SUCCESS) || (tempBuffer == nullptr)) {
            OPS_LOG_E(LOG_TAG, "Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return ge::GRAPH_FAILED;
        }

        context->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer) + i * EP_RANK_OFFSET_STEP;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::GetHcclCommResource(const HcclComm &hcclHandle, const CommEngine &engine,
                                                        const CommProtocol &protocol, Mc2MoeContext *context)
{
    OPS_LOG_I(LOG_TAG, "Start to get HCCL communication resource");
    uint32_t rankId = context->epRankId;
    std::vector<ChannelHandle> channels;

    ChannelAcquireParams params{epRankSize_, rankId, protocol, engine};
    auto status = GetHcclCommChannel(hcclHandle, params, channels);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Get HCCL communication channel failed");
        return status;
    }

    context->rankSizePerServer = rankSizePerServer_;
    OPS_LOG_I(LOG_TAG, "Get HCCL communication channel success, channel num is: %zu", channels.size());

    status = FillContextBuffer(hcclHandle, engine, context, rankId, channels);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "FillContextBuffer failed");
        return status;
    }
    OPS_LOG_I(LOG_TAG, "Fill context buffer success");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::CreatMc2Context(const ContextCreateParams &params, Mc2MoeContext *context)
{
    void *ctx = nullptr;
    uint64_t ctxSize = sizeof(Mc2MoeContext);
    auto hcclRet = HcclEngineCtxCreate(params.hcclHandle, params.mc2ContextTag.c_str(), params.engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Create HCCL context memory failed");
        return ge::GRAPH_FAILED;
    }
    auto cleanup = [&]() {
        if (ctx != nullptr) {
            HcclEngineCtxDestroy(params.hcclHandle, params.mc2ContextTag.c_str(), params.engine);
        }
    };

    hcclRet = HcclGetRankId(params.hcclHandle, &context->epRankId);
    if ((hcclRet != HCCL_SUCCESS) || (context->epRankId >= MAX_EP_RANK_SIZE)) {
        OPS_LOG_E(LOG_TAG, "Get rank ID failed, rankId=%u", context->epRankId);
        cleanup();
        return ge::GRAPH_FAILED;
    }
    hcclRet = HcclGetRankSize(params.hcclHandle, &epRankSize_);
    if ((hcclRet != HCCL_SUCCESS) || (epRankSize_ == 0) || (epRankSize_ >= MAX_EP_RANK_SIZE)) {
        OPS_LOG_E(LOG_TAG, "Get rank size failed, rankSize=%u", epRankSize_);
        cleanup();
        return ge::GRAPH_FAILED;
    }

    auto status = GetHcclCommResource(params.hcclHandle, params.engine, params.protocol, context);
    if (status != ge::GRAPH_SUCCESS) {
        cleanup();
        OPS_LOG_E(LOG_TAG, "Get HCCL communication resource failed");
        return status;
    }
    hcclRet = HcclEngineCtxCopy(params.hcclHandle, params.engine, params.mc2ContextTag.c_str(), context, ctxSize,
                                CTX_COPY_DEFAULT_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        cleanup();
        OPS_LOG_E(LOG_TAG, "Copy context from host to device failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::CheckLinks(uint32_t &netLinkNum, CommLink *linksList)
{
    if (linksList == nullptr) {
        OPS_LOG_E(LOG_TAG, "linksList is nullptr");
        return ge::GRAPH_FAILED;
    }
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus Mc2MoeGraphContext::CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList,
                                                         uint32_t &layerNum)
{
    uint32_t srcRankId = 0;
    uint32_t netLinkNum = 0;
    uint32_t rankNumInLayer = 0;
    uint32_t *rankIdLists = nullptr;
    CommLink *linksList = nullptr;

    auto hcclRet = HcclGetRankId(hcclHandle, &srcRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "CheckProtocolSupport Get rank ID failed");
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_I(LOG_TAG, "CheckProtocolSupport Get rank ID success, rankId is: %u", srcRankId);
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        OPS_LOG_I(LOG_TAG, "CheckProtocolSupport Check layer %u", layerList[layerIndex]);
        hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if ((hcclRet != HCCL_SUCCESS) || (rankNumInLayer == 0) || (rankIdLists == nullptr)) {
            OPS_LOG_E(LOG_TAG, "Get rank IDs by layer failed");
            return ge::GRAPH_FAILED;
        }
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if ((rankIdLists[rankId] == srcRankId) || (layerMap_.find(rankIdLists[rankId]) != layerMap_.end())) {
                continue;
            }
            hcclRet = HcclRankGraphGetLinks(hcclHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                            &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS) {
                OPS_LOG_E(LOG_TAG, "Get HCCL links failed");
                return ge::GRAPH_FAILED;
            }
            if ((netLinkNum == 0) || (linksList == nullptr)) {
                OPS_LOG_E(LOG_TAG, "No available HCCL links found, srcRankID %u, dstRankID %u layer is %u", srcRankId,
                          rankIdLists[rankId], layerList[layerIndex]);
                return ge::GRAPH_FAILED;
            }
            if (CheckLinks(netLinkNum, linksList) != ge::GRAPH_SUCCESS) {
                OPS_LOG_E(LOG_TAG, "No HCCL links support UB_MEM srcRankID %u, dstRankID %u layer is %u", srcRankId,
                          rankIdLists[rankId], layerList[layerIndex]);
                return ge::GRAPH_FAILED;
            }
            layerMap_[rankIdLists[rankId]] = layerList[layerIndex];
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol)
{
    OPS_LOG_I(LOG_TAG, "Start to get HCCL communication protocol");
    uint32_t layerNum = 0;
    uint32_t *layerList = nullptr;

    auto ret = HcclRankGraphGetLayers(hcclHandle, &layerList, &layerNum);
    if ((ret != HCCL_SUCCESS) || (layerNum == 0)) {
        OPS_LOG_E(LOG_TAG, "Get HCCL layers failed");
        return ge::GRAPH_FAILED;
    }

    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return ge::GRAPH_SUCCESS;
    }

    OPS_LOG_I(LOG_TAG, "start CheckProtocolSupport, layerNum is %u", layerNum);
    auto status = CheckProtocolSupport(hcclHandle, layerList, layerNum);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "CheckProtocolSupport failed");
        return status;
    }

    OPS_LOG_I(LOG_TAG, "CheckProtocolSupport success!");
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::ValidateContextTag(const std::string &mc2ContextTag)
{
    if (mc2ContextTag.size() > MAX_CONTEXT_TAG_SIZE) {
        OPS_LOG_E(LOG_TAG, "Mc2ContextTag is too long, max size is %llu, but current size is %zu", MAX_CONTEXT_TAG_SIZE,
                  mc2ContextTag.size());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::GetMc2Context(const ContextAcquireParams &params, Mc2MoeContext *context,
                                                  int64_t &hcclBuffSize)
{
    ContextCreateParams createParams{params.hcclHandle, params.engine, params.protocol, params.mc2ContextTag};
    auto status = CreatMc2Context(createParams, context);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "CreatMc2Context failed");
        return status;
    }

    hcclBuffSize = static_cast<int64_t>(hcclBuffSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::GetHcclContext(const std::string &contextTag, const std::string &groupEp,
                                                   Mc2MoeContext *context, int64_t &hcclBuffSize)
{
    OPS_LOG_I(LOG_TAG, "GetHcclContext start, tag=%s, groupEp=%s", contextTag.c_str(), groupEp.c_str());
    auto status = LoadHcclSymbols();
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "LoadHcclSymbols failed");
        return ge::GRAPH_FAILED;
    }

    CommProtocol protocol;
    static std::atomic<uint64_t> s_ctxVersion{0};
    std::string mc2ContextTag = contextTag + "_" + groupEp + "_" +
                                std::to_string(s_ctxVersion.fetch_add(1, std::memory_order_relaxed));

    status = ValidateContextTag(mc2ContextTag);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "ValidateContextTag failed");
        return ge::GRAPH_FAILED;
    }

    HcclComm hcclHandle;
    status = GetCommHandle(groupEp.c_str(), hcclHandle);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "GetCommHandle failed");
        return ge::GRAPH_FAILED;
    }

    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    OPS_LOG_I(LOG_TAG, "GetCommProtocol...");
    status = GetCommProtocol(hcclHandle, protocol);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "GetCommProtocol failed");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(LOG_TAG, "GetCommProtocol done, protocol=%d", protocol);

    ContextAcquireParams params{hcclHandle, engine, protocol, mc2ContextTag};
    OPS_LOG_I(LOG_TAG, "GetMc2Context...");
    status = GetMc2Context(params, context, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "GetMc2Context failed");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(LOG_TAG, "GetHcclContext done, hcclBuffSize=%lld", hcclBuffSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MoeGraphContext::CreateContextNode(ge::Graph& graph, const Mc2MoeContext& context,
                                                      ge::GNode& constNode)
{
    static const std::string kContextConstName = "mc2_moe_context";
    static const uint32_t IDX_0 = 0;

    ge::Shape shape({static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t))});
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor tensorValue(tensorDesc, reinterpret_cast<const uint8_t*>(&context), sizeof(Mc2MoeContext));

    auto constOp = ge::OperatorFactory::CreateOperator(kContextConstName, "Const");
    if (constOp.IsEmpty()) {
        OPS_LOG_E(LOG_TAG, "Const op not found in factory");
        return ge::GRAPH_FAILED;
    }
    constOp.UpdateOutputDesc(IDX_0, tensorDesc);
    constOp.SetAttr("value", tensorValue);

    constNode = graph.AddNodeByOp(constOp);
    ge::AscendString ctxNodeName;
    constNode.GetName(ctxNodeName);
    OPS_LOG_I(LOG_TAG, "Context Const node added as %s", ctxNodeName.GetString());

    return ge::GRAPH_SUCCESS;
}

bool Mc2MoeGraphContext::TryGetBufferSizeCache(const std::string& key, int64_t& hcclBuffSize)
{
    auto it = hcclBuffSizeCacheA5_.find(key);
    if (it != hcclBuffSizeCacheA5_.end()) {
        hcclBuffSize = it->second;
        return true;
    }
    return false;
}

void Mc2MoeGraphContext::UpdateBufferSizeCache(const std::string& key, int64_t hcclBuffSize)
{
    hcclBuffSizeCacheA5_[key] = hcclBuffSize;
}
ge::graphStatus Mc2MoeGraphContext::GetMc2ContextA5(
    ge::Graph& graph, const std::string& contextTag, const std::string& epName,
    const ge::GNode& originalNode, ge::GNode& constNode, int64_t& hcclBuffSize)
{
    if ((contextTag.empty()) || (epName.empty())) {
        OPS_LOG_E(LOG_TAG, "Context tag or ep name is empty");
        return ge::GRAPH_FAILED;
    }
    ge::AscendString groupEpStr;
    if ((originalNode.GetAttr(ge::AscendString(epName.c_str()), groupEpStr) != ge::GRAPH_SUCCESS) ||
        (groupEpStr.GetString() == nullptr) || (*groupEpStr.GetString() == '\0')) {
        OPS_LOG_E(LOG_TAG, "Failed to get %s attr", epName.c_str());
        return ge::GRAPH_FAILED;
    }
    std::string groupEp(groupEpStr.GetString());
    Mc2MoeGraphContext& manager = GetInstance();
    ge::AscendString graphNameStr;
    graph.GetName(graphNameStr);
    std::string uniqueKey = contextTag + "_" + groupEp + "_" +
        (graphNameStr.GetString() ? graphNameStr.GetString() : "unknown");
    auto nodeIt = manager.contextNodeMapA5_.find(uniqueKey);
    if (nodeIt != manager.contextNodeMapA5_.end()) {
        ge::AscendString cachedName;
        if (nodeIt->second.GetName(cachedName) == ge::GRAPH_SUCCESS) {
            constNode = nodeIt->second;
            if (manager.TryGetBufferSizeCache(uniqueKey, hcclBuffSize)) {
                return ge::GRAPH_SUCCESS;
            }
            Mc2MoeContext context{0};
            auto status = manager.GetHcclContext(contextTag, groupEp, &context, hcclBuffSize);
            if (status != ge::GRAPH_SUCCESS) {
                OPS_LOG_E(LOG_TAG, "Get HCCL context failed for reuse");
                return status;
            }
            manager.UpdateBufferSizeCache(uniqueKey, hcclBuffSize);
            return ge::GRAPH_SUCCESS;
        }
        manager.contextNodeMapA5_.erase(nodeIt);
    }
    return manager.CreateNewMc2Context(graph, contextTag, groupEp, uniqueKey, constNode, hcclBuffSize);
}

ge::graphStatus Mc2MoeGraphContext::CreateNewMc2Context(
    ge::Graph& graph, const std::string& contextTag, const std::string& groupEp,
    const std::string& uniqueKey, ge::GNode& constNode, int64_t& hcclBuffSize)
{
    Mc2MoeContext context{0};
    auto status = GetHcclContext(contextTag, groupEp, &context, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Get HCCL context failed");
        return status;
    }
    UpdateBufferSizeCache(uniqueKey, hcclBuffSize);
    status = CreateContextNode(graph, context, constNode);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(LOG_TAG, "Create context node failed");
        return status;
    }
    contextNodeMapA5_[uniqueKey] = constNode;
    return ge::GRAPH_SUCCESS;
}

template void *Mc2MoeGraphContext::GetHcclLibFunc<void *>(void *handle, const std::string &funcName);

}  // namespace ops

#endif  // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#endif  // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
