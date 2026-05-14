/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_moe_context.cpp
 * \brief
 */
#include "mc2_moe_context.h"
#if defined(HCOMM_VERSION_NUM) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

#include <cstdlib>
#include <dlfcn.h>
#include "mc2_platform_info.h"
#include "mc2_common_log.h"

namespace {
constexpr uint64_t CTX_COPY_DEFAULT_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t MAX_EP_RANK_SIZE = 1024;
} // namespace

namespace fe {

MoeContextManager::~MoeContextManager()
{
    if (hcclLibHandle_ != nullptr) {
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
    }
}

MoeContextManager& MoeContextManager::GetInstance()
{
    static MoeContextManager instance;
    return instance;
}

const std::string MoeContextManager::GetLibPath()
{
    const char* ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Ascend home path doesn't exist.");
        return "";
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
#endif
    std::string fullPath = std::string(ascendPath) + hcclPathPostfix;
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Loading lib in path %s.", fullPath.c_str());
    return fullPath;
}

template <typename T>
T MoeContextManager::GetHcclLibFunc(void* handle, const std::string& funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        const char* dlErr = dlerror();
        OP_LOGE(
            FUSED_OP_TYPE2_A5.c_str(), "Load func=%s error=%s in lib hccl failed.", funcName.c_str(),
            dlErr ? dlErr : "unknown");
    }
    return func;
}

ge::graphStatus MoeContextManager::LoadSymbol(void** ptr, const char* name)
{
    *ptr = GetHcclLibFunc<void*>(hcclLibHandle_, name);
    if (*ptr == nullptr) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Failed to load %s symbol", name);
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::LoadHcclSymbols()
{
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Start to load HCCL library and symbols");
    const std::string libPath = GetLibPath();
    if (libPath.empty()) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Failed to get HCCL library path");
        return ge::GRAPH_FAILED;
    }

    hcclLibHandle_ = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        const char* dlErr = dlerror();
        OP_LOGE(
            FUSED_OP_TYPE2_A5.c_str(), "dlopen HCCL library failed: %s, error: %s", libPath.c_str(),
            dlErr ? dlErr : "unknown");
        return ge::GRAPH_FAILED;
    }

    struct SymbolInfo {
        void** ptr;
        const char* name;
    };

    SymbolInfo symbols[] = {
        {reinterpret_cast<void**>(&HcomGetCommHandleByGroup), "HcomGetCommHandleByGroup"},
        {reinterpret_cast<void**>(&HcclRankGraphGetLinks), "HcclRankGraphGetLinks"},
        {reinterpret_cast<void**>(&HcclRankGraphGetLayers), "HcclRankGraphGetLayers"},
        {reinterpret_cast<void**>(&HcclRankGraphGetRankSizeByLayer), "HcclRankGraphGetRankSizeByLayer"},
        {reinterpret_cast<void**>(&HcclChannelAcquire), "HcclChannelAcquire"},
        {reinterpret_cast<void**>(&HcclGetHcclBuffer), "HcclGetHcclBuffer"},
        {reinterpret_cast<void**>(&HcclChannelGetHcclBuffer), "HcclChannelGetHcclBuffer"},
        {reinterpret_cast<void**>(&HcclEngineCtxCreate), "HcclEngineCtxCreate"},
        {reinterpret_cast<void**>(&HcclEngineCtxGet), "HcclEngineCtxGet"},
        {reinterpret_cast<void**>(&HcclEngineCtxCopy), "HcclEngineCtxCopy"},
        {reinterpret_cast<void**>(&HcclGetRankId), "HcclGetRankId"},
        {reinterpret_cast<void**>(&HcclGetRankSize), "HcclGetRankSize"},
        {reinterpret_cast<void**>(&HcclRankGraphGetRanksByLayer), "HcclRankGraphGetRanksByLayer"},
    };

    for (auto& sym : symbols) {
        if (LoadSymbol(sym.ptr, sym.name) != SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "All HCCL symbols loaded successfully");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetCommHandle(const char* groupEp, HcclComm& hcclHandle)
{
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Start to get HCCL communication handle, groupEp: %s", groupEp);
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL handle failed, groupEp: %s", groupEp);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL communication handle success hcclHandle is: %p", hcclHandle);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetHcclCommLink(const HcclComm& hcclHandle, const LinkQueryParams& params, CommLink*& links)
{
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Start to get HCCL communication link");
    CommLink* linksList = nullptr;
    uint32_t netLinkNum = 0;

    auto hcclRet = HcclRankGraphGetLinks(
        hcclHandle, params.netLayerIndex, params.srcRankId, params.dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL Communication link failed");
        return ge::GRAPH_FAILED;
    }
    if (netLinkNum == 0) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "The Net Link Is nullptr.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL Rank Links Success Links Num is: %u", netLinkNum);
    uint32_t index = 0;
    for (; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == params.protocol) {
            links = &linksList[index];
            break;
        }
    }

    if (index == netLinkNum) {
        OP_LOGE(
            FUSED_OP_TYPE2_A5.c_str(),
            "No matching communication protocol found in HCCL links protocol is %d, srcRankId is %u, dstRankId is %u, "
            "netLayerIndex is %u",
            params.protocol, params.srcRankId, params.dstRankId, params.netLayerIndex);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL communication link success, protocol is: %d", params.protocol);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetRankSizePerServer(const HcclComm& hcclHandle, uint32_t netLayers)
{
    auto hcclRet = HcclRankGraphGetRankSizeByLayer(hcclHandle, netLayers, &rankSizePerServer_);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL rank size per server failed");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(
        FUSED_OP_TYPE2_A5.c_str(), "Get HCCL rank size per server success, rankSizePerServer_ is: %u",
        rankSizePerServer_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::SetupChannelDesc(
    const HcclComm& hcclHandle, const ChannelInitParams& params, uint32_t* netLayerList, uint32_t netLayerNum,
    std::vector<HcclChannelDesc>& channelDesc)
{
    for (uint32_t i = 0; i < params.rankDim; ++i) {
        if (i == params.srcRankId) {
            continue;
        }

        uint32_t dstRank = i;
        uint32_t channelId = (i > params.srcRankId) ? (i - 1) : i;
        CommLink* links = nullptr;
        uint32_t netLayerIndex = netLayerNum == 1 ? netLayerList[HCCL_COMM_LAYERS_UB_MEM] : layerMap[dstRank];
        LinkQueryParams query{netLayerIndex, params.srcRankId, dstRank, params.protocol};
        auto ret = GetHcclCommLink(hcclHandle, query, links);
        if (ret != SUCCESS) {
            return ret;
        }

        channelDesc[channelId].channelProtocol = params.protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::InitHcclChannel(
    const HcclComm& hcclHandle, const ChannelInitParams& params, std::vector<HcclChannelDesc>& channelDesc)
{
    uint32_t channelNum = channelDesc.size();
    uint32_t netLayerIndex = 0;
    uint32_t netLayerNum = 0;
    uint32_t* netLayerList = nullptr;

    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "HCCL channel init failed");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "HCCL channel init success");

    hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS || netLayerNum == 0) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL GetNetLayers failed");
        return ge::GRAPH_FAILED;
    }

    uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
    auto status = GetRankSizePerServer(hcclHandle, netLayers);
    FUSION_PASS_CHECK(
        status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL GetRankSizePerServer failed"), return status);

    status = SetupChannelDesc(hcclHandle, params, netLayerList, netLayerNum, channelDesc);
    FUSION_PASS_CHECK(
        status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL SetupChannelDesc failed"), return status);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetHcclCommChannel(
    const HcclComm& hcclHandle, const ChannelAcquireParams& params, std::vector<ChannelHandle>& channels)
{
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Start to get HCCL communication channel");
    uint32_t channelNum = params.rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);

    ChannelInitParams initParams{params.rankDim, params.srcRankId, params.protocol};
    auto status = InitHcclChannel(hcclHandle, initParams, channelDesc);
    FUSION_PASS_CHECK(
        status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL InitHcclChannel failed"), return status);

    auto hcclRet = HcclChannelAcquire(hcclHandle, params.engine, channelDesc.data(), channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Acquire HCCL channel failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::FillContextBuffer(
    const HcclComm& hcclHandle, const CommEngine& engine, Mc2MoeContext* context, uint32_t rankId,
    const std::vector<ChannelHandle>& channels)
{
    for (uint32_t i = 0; i < epRankSize_; ++i) {
        void* tempBuffer = nullptr;
        uint64_t bufSize = 0;
        HcclResult hcclRet;

        if (i == rankId) {
            hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize_);
        } else {
            uint32_t idx = (i < rankId) ? i : (i - 1);
            hcclRet = HcclChannelGetHcclBuffer(hcclHandle, channels[idx], &tempBuffer, &bufSize);
        }

        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return ge::GRAPH_FAILED;
        }

        context->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetHcclCommResource(
    const HcclComm& hcclHandle, const CommEngine& engine, const CommProtocol& protocol, Mc2MoeContext* context)
{
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Start to get HCCL communication resource");
    uint32_t rankId = context->epRankId;
    std::vector<ChannelHandle> channels;

    ChannelAcquireParams params{epRankSize_, rankId, protocol, engine};
    auto status = GetHcclCommChannel(hcclHandle, params, channels);
    FUSION_PASS_CHECK(
        status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL communication channel failed"), return status);

    context->rankSizePerServer = rankSizePerServer_;
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL communication channel success, channel num is: %u", channels.size());

    status = FillContextBuffer(hcclHandle, engine, context, rankId, channels);
    FUSION_PASS_CHECK(status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "FillContextBuffer failed"), return status);
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Fill context buffer success");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::CreatMc2Context(const ContextCreateParams& params, Mc2MoeContext* context)
{
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Start to create HCCL context");
    void* ctx = nullptr;
    uint64_t ctxSize = sizeof(Mc2MoeContext);

    auto hcclRet = HcclEngineCtxCreate(params.hcclHandle, params.mc2ContextTag.c_str(), params.engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Create HCCL context memory failed");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Create HCCL context success, ctx: %p", ctx);

    hcclRet = HcclGetRankId(params.hcclHandle, &context->epRankId);
    if (hcclRet != HCCL_SUCCESS || context->epRankId >= MAX_EP_RANK_SIZE) {
        OP_LOGE(
            FUSED_OP_TYPE2_A5.c_str(), "Get rank ID failed or rank ID is invalid, rankId is: %u", context->epRankId);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Get rank ID success, rankId is: %u", context->epRankId);

    hcclRet = HcclGetRankSize(params.hcclHandle, &epRankSize_);
    if (hcclRet != HCCL_SUCCESS || epRankSize_ == 0 || epRankSize_ >= MAX_EP_RANK_SIZE) {
        OP_LOGE(
            FUSED_OP_TYPE2_A5.c_str(), "Get rank size failed or rank size is invalid, rankSize is: %u", epRankSize_);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Get rank size success, rankSize is: %u", epRankSize_);

    auto status = GetHcclCommResource(params.hcclHandle, params.engine, params.protocol, context);
    FUSION_PASS_CHECK(
        status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL communication resource failed"), return status);

    hcclRet = HcclEngineCtxCopy(
        params.hcclHandle, params.engine, params.mc2ContextTag.c_str(), context, ctxSize, CTX_COPY_DEFAULT_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Copy context from host to device failed");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Copy context from host to device success");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::CheckLinks(uint32_t& netLinkNum, CommLink* linksList)
{
    bool isFoundUbMemProtocol = false;
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            isFoundUbMemProtocol = true;
            break;
        }
    }
    return isFoundUbMemProtocol ? SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus MoeContextManager::CheckProtocolSupport(const HcclComm& hcclHandle, uint32_t*& layerList, uint32_t& layerNum)
{
    uint32_t srcRankId = 0;
    uint32_t dstRankId = 0;
    uint32_t netLinkNum = 0;
    uint32_t rankNumInLayer = 0;
    uint32_t* rankIdLists = nullptr;
    CommLink* linksList = nullptr;

    auto hcclRet = HcclGetRankId(hcclHandle, &srcRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "CheckProtocolSupport Get rank ID failed");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD("CheckProtocolSupport Get rank ID success, rankId is: %d", srcRankId);
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        OP_LOGD("CheckProtocolSupport Check layer %d", layerList[layerIndex]);
        hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get rank IDs by layer failed");
            return ge::GRAPH_FAILED;
        }
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if (rankIdLists[rankId] == srcRankId ||
                layerMap.find(rankIdLists[rankId]) != layerMap.end()) { // 本卡或者已经校验过的卡跳过
                continue;
            }
            hcclRet = HcclRankGraphGetLinks(
                hcclHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId], &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS) {
                OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL links failed");
                return ge::GRAPH_FAILED;
            }
            if (netLinkNum == 0) {
                OP_LOGE(
                    FUSED_OP_TYPE2_A5.c_str(), "No available HCCL links found, srcRankID %d, dstRankID %d layer is %d",
                    srcRankId, rankIdLists[rankId], layerList[layerIndex]);
                return ge::GRAPH_FAILED;
            }
            if (CheckLinks(netLinkNum, linksList) != SUCCESS) {
                OP_LOGE(
                    FUSED_OP_TYPE2_A5.c_str(), "No HCCL links support UB_MEM srcRankID %d, dstRankID %d layer is %d",
                    srcRankId, dstRankId, layerList[layerIndex]);
                return ge::GRAPH_FAILED;
            }
            layerMap[rankIdLists[rankId]] = layerList[layerIndex];
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetCommProtocol(const HcclComm& hcclHandle, CommProtocol& protocol)
{
    OP_LOGD(FUSED_OP_TYPE2_A5.c_str(), "Start to get HCCL communication protocol");
    uint32_t layerNum = 0;
    uint32_t* layerList = nullptr;

    auto ret = HcclRankGraphGetLayers(hcclHandle, &layerList, &layerNum);
    if (ret != HCCL_SUCCESS || layerNum == 0) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL layers failed");
        return ge::GRAPH_FAILED;
    }

    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGD("start CheckProtocolSupport, layerNum is %d", layerNum);
    auto status = CheckProtocolSupport(hcclHandle, layerList, layerNum);
    FUSION_PASS_CHECK(
        status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "CheckProtocolSupport failed"), return status);

    OP_LOGD("CheckProtocolSupport success!");
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::ValidateContextTag(const std::string& mc2ContextTag)
{
    if (mc2ContextTag.size() > MAX_CONTEXT_TAG_SIZE) {
        OP_LOGE(
            FUSED_OP_TYPE2_A5.c_str(), "Mc2ContextTag is too long, max size is %d, but current size is %d",
            MAX_CONTEXT_TAG_SIZE, mc2ContextTag.size());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetMc2Context(
    const ContextAcquireParams& params, Mc2MoeContext* context, int64_t& hcclBuffSize)
{
    ContextCreateParams createParams{params.hcclHandle, params.engine, params.protocol, params.mc2ContextTag};
    auto status = CreatMc2Context(createParams, context);
    FUSION_PASS_CHECK(status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "CreatMc2Context failed"), return status);

    hcclBuffSize = static_cast<int64_t>(hcclBuffSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetHcclContext(
    const std::string& graphName, const std::string& groupEp, const std::string& nodeName, Mc2MoeContext* context,
    int64_t& hcclBuffSize)
{
    auto status = LoadHcclSymbols();
    FUSION_PASS_CHECK(status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "LoadHcclSymbols failed"), return ge::GRAPH_SUCCESS);

    CommProtocol protocol;
    std::string mc2ContextTag = graphName + groupEp + nodeName;

    status = ValidateContextTag(mc2ContextTag);
    FUSION_PASS_CHECK(
        status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "ValidateContextTag failed"), return ge::GRAPH_SUCCESS);

    HcclComm hcclHandle;
    status = GetCommHandle(groupEp.c_str(), hcclHandle);
    FUSION_PASS_CHECK(status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "GetCommHandle failed"), return ge::GRAPH_SUCCESS);

    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    status = GetCommProtocol(hcclHandle, protocol);
    FUSION_PASS_CHECK(status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "GetCommProtocol failed"), return ge::GRAPH_SUCCESS);

    ContextAcquireParams params{hcclHandle, engine, protocol, mc2ContextTag};
    status = GetMc2Context(params, context, hcclBuffSize);
    FUSION_PASS_CHECK(status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "GetMc2Context failed"), return ge::GRAPH_SUCCESS);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::ExtractNodeInfo(
    const ge::NodePtr& originalNode, const std::string& epName, std::string& groupEp)
{
    auto opDesc = originalNode->GetOpDesc();
    FUSION_PASS_CHECK(
        opDesc == nullptr, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "originalNode has no opDesc"), return ge::GRAPH_FAILED);

    if (!ge::AttrUtils::GetStr(opDesc, epName.c_str(), groupEp) || groupEp.empty()) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Failed to get %s attr", epName.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::CheckCache(const std::string& uniqueKey, ge::NodePtr& constNode, int64_t& hcclBuffSize)
{
    auto nodeIt = contextNodeMapA5.find(uniqueKey);
    auto bufferIt = bufferSizeMapA5.find(uniqueKey);
    if (nodeIt != contextNodeMapA5.end() && bufferIt != bufferSizeMapA5.end()) {
        constNode = nodeIt->second;
        hcclBuffSize = bufferIt->second;
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus MoeContextManager::UpdateCache(const std::string& uniqueKey, const ge::NodePtr& constNode, int64_t hcclBuffSize)
{
    contextNodeMapA5[uniqueKey] = constNode;
    bufferSizeMapA5[uniqueKey] = hcclBuffSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::CreateContextNode(
    ge::ComputeGraph& graph, const Mc2MoeContext& context, ge::NodePtr& constNode)
{
    ge::GeShape shape({static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t))});
    ge::GeTensorDesc tensorDesc = ge::GeTensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    auto geTensor =
        std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<const uint8_t*>(&context), sizeof(Mc2MoeContext));

    ge::OpDescPtr constOpDesc = ge::OpDescUtils::CreateConstOp(geTensor);
    FUSION_PASS_CHECK(
        constOpDesc == nullptr, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Create const op desc failed"), return ge::GRAPH_FAILED);

    constNode = graph.AddNode(constOpDesc);
    FUSION_PASS_CHECK(constNode == nullptr, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Add const node failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetMc2ContextA5(
    ge::ComputeGraph& graph, const std::string& nodeName, const std::string& epName, const ge::NodePtr& originalNode,
    ge::NodePtr& constNode, int64_t& hcclBuffSize)
{
    if (nodeName.empty() || epName.empty()) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Node name or ep name is empty");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(
        FUSED_OP_TYPE2_A5.c_str(), "Start to get or create Mc2Context for node %s in graph %s", nodeName.c_str(),
        graph.GetName().c_str());

    MoeContextManager& manager = GetInstance();

    std::string groupEp;
    auto status = manager.ExtractNodeInfo(originalNode, epName, groupEp);
    FUSION_PASS_CHECK(status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "get groupEp failed"), return status);

    std::string graphName = graph.GetName();
    if (graphName.empty()) {
        OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Failed to get graph name");
        return ge::GRAPH_FAILED;
    }

    std::string uniqueKey = graphName + "_" + groupEp + "_" + nodeName;
    status = manager.CheckCache(uniqueKey, constNode, hcclBuffSize);
    if (status == SUCCESS) {
        OP_LOGD(
            FUSED_OP_TYPE2_A5.c_str(), "Found existing context node in cache for group:%s, graph:%s, reuse it.",
            groupEp.c_str(), graphName.c_str());
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGD(
        FUSED_OP_TYPE2_A5.c_str(), "Generate new context node for group:%s, graph:%s", groupEp.c_str(),
        graphName.c_str());

    Mc2MoeContext context{0};
    status = manager.GetHcclContext(graphName, groupEp, nodeName, &context, hcclBuffSize);
    FUSION_PASS_CHECK(status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL context failed"), return status);

    status = manager.CreateContextNode(graph, context, constNode);
    FUSION_PASS_CHECK(
        status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Create context node failed"), return status);

    status = manager.UpdateCache(uniqueKey, constNode, hcclBuffSize);
    FUSION_PASS_CHECK(status != SUCCESS, OP_LOGE(FUSED_OP_TYPE2_A5.c_str(), "Update cache failed"), return status);

    OP_LOGD(
        FUSED_OP_TYPE2_A5.c_str(), "End to get or create Mc2Context for node %s in graph %s", nodeName.c_str(),
        graph.GetName().c_str());
    return ge::GRAPH_SUCCESS;
}

template void* MoeContextManager::GetHcclLibFunc<void*>(void* handle, const std::string& funcName);
std::string MoeContextManager::FUSED_OP_TYPE2_A5 = "UpdateContextA5";

} // namespace fe
#endif // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION