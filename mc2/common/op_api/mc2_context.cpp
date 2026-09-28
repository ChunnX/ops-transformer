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
 * \file mc2_context.cpp
 * \brief MC2 Context management implementation
 */

#include <cstdlib>
#include <dlfcn.h>
#include <utility>
#include "mc2_context.h"
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

namespace {
constexpr uint64_t KOPY_DEFAULT_CTX_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t EP_RANK_OFFSET_STEP = 1024;
} // namespace

namespace Mc2Aclnn {

const std::string Mc2Context::GetLibPath()
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Ascend home path doesn't exist.");
        return "";
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
#endif
    std::string fullPath = std::string(ascendPath) + hcclPathPostfix;
    OP_LOGI("Loading lib in path %s.", fullPath.c_str());
    return fullPath;
}

template <typename T>
T Mc2Context::GetHcclLibFunc(void *handle, const std::string &funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Load func=%s error=%s in lib hccl failed.", funcName.c_str(), dlerror());
    }
    return func;
}

Mc2Context::Mc2Context()
{
    OP_LOGI("Init Mc2Context Success!");
}

Mc2Context::~Mc2Context()
{
    if (hcclLibHandle_ != nullptr) {
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
    }
}

aclnnStatus Mc2Context::LoadHcclSymbols()
{
    OP_LOGD("Start to load HCCL library and symbols");
    const std::string libPath = GetLibPath();
    if (libPath.empty()) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Failed to get HCCL library path");
        return ACLNN_ERR_INNER;
    }

    hcclLibHandle_ = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "dlopen HCCL library failed: %s, error: %s", libPath.c_str(), dlerror());
        return ACLNN_ERR_INNER;
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
        *(sym.ptr) = GetHcclLibFunc<void *>(hcclLibHandle_, sym.name);
        if (*(sym.ptr) == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Failed to load %s symbol", sym.name);
            dlclose(hcclLibHandle_);
            hcclLibHandle_ = nullptr;
            return ACLNN_ERR_INNER;
        }
    }
    OP_LOGD("All HCCL symbols loaded successfully");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetCommHandle(const char *groupEp, HcclComm &hcclHandle)
{
    OP_LOGI("Start to get HCCL communication handle, groupEp: %s", groupEp);
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL handle failed, groupEp: %s", groupEp);
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get HCCL communication handle success hcclHandle is: %p", hcclHandle);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclCommLink(const HcclComm &hcclHandle, uint32_t netLayerId, uint32_t srcRankId,
                                        uint32_t dstRankId, const CommProtocol &protocol, CommLink *&links)
{
    OP_LOGD("Start to get HCCL communication link");
    CommLink *linksList = nullptr;
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL Communication link failed");
        return ACLNN_ERR_INNER;
    }
    if (linksList == nullptr || netLinkNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                                "The Net Link Is nullptr. srcRankId is %u, dstRankId is %u, layerId is %u", srcRankId,
                                dstRankId, netLayerId);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Rank Links Success Links Num is: %u", netLinkNum);
    uint32_t index = 0;
    for (; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == protocol) {
            links = &linksList[index];
            break;
        }
    }
    if (index == netLinkNum) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "No matching communication protocol found in HCCL links, "
                               "protocol: %d, srcRankId: %u, dstRankId: %u", protocol, srcRankId, dstRankId);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL communication link success, protocol: %d, srcRankId: %u, dstRankId: %u",
            protocol, srcRankId, dstRankId);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetNetLayers(const HcclComm &hcclHandle, uint32_t *&netLayerList, uint32_t &netLayerNum)
{
    auto hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS || netLayerList == nullptr || netLayerNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL layers failed or returned an empty layer list");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get HCCL layers success, netLayerNum is: %u", netLayerNum);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetRankSizePerServer(const HcclComm &hcclHandle, uint32_t netLayers)
{
    auto hcclRet = HcclRankGraphGetRankSizeByLayer(hcclHandle, netLayers, &rankSizePerServer_);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL rank size per server failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get HCCL rank size per server success, rankSizePerServer_ is: %u", rankSizePerServer_);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::ResolveLinkInfo(uint32_t dstRank, const CommProtocol &protocol, const uint32_t *netLayerList,
                                     uint32_t netLayerNum, RankLinkInfo &linkInfo) const
{
    auto linkIter = rankLinkMap_.find(dstRank);
    if (linkIter != rankLinkMap_.end()) {
        linkInfo = linkIter->second;
        return ACLNN_SUCCESS;
    }
    if (netLayerNum != HCCL_COMM_LAYERS_MTE_CCU) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Topology info not found for dstRank %u in a multi-layer topology",
                               dstRank);
        return ACLNN_ERR_INNER;
    }
    linkInfo = {protocol, netLayerList[HCCL_COMM_LAYERS_UB_MEM]};
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::InitHcclChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                        const CommProtocol &protocol, std::vector<HcclChannelDesc> &channelDesc)
{
    uint32_t channelNum = channelDesc.size();
    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "HCCL channel init failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("HCCL channel init success");

    uint32_t netLayerNum = 0;
    uint32_t *netLayerList = nullptr;
    auto ret = GetNetLayers(hcclHandle, netLayerList, netLayerNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    for (uint32_t dstRank = 0; dstRank < rankDim; ++dstRank) {
        if (dstRank == srcRankId) {
            continue;
        }
        uint32_t channelId = (dstRank > srcRankId) ? (dstRank - 1) : dstRank;
        RankLinkInfo linkInfo;
        ret = ResolveLinkInfo(dstRank, protocol, netLayerList, netLayerNum, linkInfo);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        CommLink *links = nullptr;
        ret = GetHcclCommLink(hcclHandle, linkInfo.layer, srcRankId, dstRank, linkInfo.protocol, links);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        channelDesc[channelId].channelProtocol = linkInfo.protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclCommChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                           const CommProtocol &protocol, const CommEngine &engine,
                                           std::vector<ChannelHandle> &channels)
{
    OP_LOGD("Start to get HCCL communication channel");
    uint32_t channelNum = rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);

    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    auto ret = GetNetLayers(hcclHandle, netLayerList, netLayerNum);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER]; // 获取本server卡数
    ret = GetRankSizePerServer(hcclHandle, netLayers);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = InitHcclChannel(hcclHandle, rankDim, srcRankId, protocol, channelDesc);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto hcclRet = HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Acquire HCCL channel failed");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclCommResource(const HcclComm &hcclHandle, const CommEngine &engine,
                                            const CommProtocol &protocol, Mc2MoeContext *mc2ContextStruct)
{
    OP_LOGD("Start to get HCCL communication resource");
    uint32_t rankId = mc2ContextStruct->epRankId;
    std::vector<ChannelHandle> channels;

    auto ret = GetHcclCommChannel(hcclHandle, epRankSize_, rankId, protocol, engine, channels);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    // 将rankSizePerServer_写入通信资源结构体中，后续DPU直驱需要使用
    mc2ContextStruct->rankSizePerServer = rankSizePerServer_;
    OP_LOGD("Get HCCL communication channel success, channel num is: %u", channels.size());

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
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return ACLNN_ERR_INNER;
        }

        mc2ContextStruct->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer) + i * EP_RANK_OFFSET_STEP;
    }
    OP_LOGD("Get HCCL CommResource success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CreatMc2Context(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                        const CommEngine &engine, const CommProtocol &protocol,
                                        Mc2MoeContext *mc2ContextStruct, void *&ctx, uint64_t &hcclBuffSize)
{
    OP_LOGD("Start to create HCCL context");
    uint64_t ctxSize = sizeof(Mc2MoeContext);
    auto hcclRet = HcclEngineCtxCreate(hcclHandle, mc2ContextTag.c_str(), engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Create HCCL context memory failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Create HCCL context success, ctx: %p", ctx);

    hcclRet = HcclGetRankId(hcclHandle, &mc2ContextStruct->epRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank ID failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get rank ID success, rankId is: %u", mc2ContextStruct->epRankId);

    hcclRet = HcclGetRankSize(hcclHandle, &epRankSize_);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank size failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get rank size success, rankSize is: %u", epRankSize_);

    if (epRankSize_ > HCCL_MAX_RANK_SIZE) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Rank size [%u] exceeds max rank size [%u]", epRankSize_,
                                HCCL_MAX_RANK_SIZE);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto ret = GetHcclCommResource(hcclHandle, engine, protocol, mc2ContextStruct);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL communication resource failed");
        return ret;
    }

    hcclRet = HcclEngineCtxCopy(hcclHandle, engine, mc2ContextTag.c_str(), mc2ContextStruct, ctxSize,
                                KOPY_DEFAULT_CTX_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Copy context from host to device failed");
        return ACLNN_ERR_INNER;
    }
    hcclBuffSize = hcclBuffSize_;
    OP_LOGD("Copy context from host to device success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CreatMc2ContextTensor(void *ctx, aclTensor *&mc2Context)
{
    OP_LOGD("Start to create Mc2Context Tensor");
    if (ctx == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Create Mc2Context Tensor failed ctx is nullptr.");
        return ACLNN_ERR_INNER;
    }

    uint64_t mc2ContextLength = sizeof(Mc2MoeContext);
    int64_t shape[1] = {static_cast<int64_t>(mc2ContextLength / sizeof(uint32_t))};
    int64_t strides[1] = {1};

    mc2Context = aclCreateTensor(shape, 1, ACL_INT32, strides, 0, ACL_FORMAT_ND, shape, 1, ctx);
    if (mc2Context == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Create Mc2Context Tensor failed.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("CreatMc2ContextTensor Success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclBufferSize(const HcclComm &hcclHandle, uint64_t &hcclBuffSize)
{
    void *tempBuffer = nullptr;
    auto hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL Buffer Size failed");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetRankId(const HcclComm &hcclHandle, uint32_t &rankId) const
{
    auto hcclRet = HcclGetRankId(hcclHandle, &rankId);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank ID failed, ret: %d", hcclRet);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetRankSize(const HcclComm &hcclHandle, uint32_t &rankSize) const
{
    auto hcclRet = HcclGetRankSize(hcclHandle, &rankSize);
    if (hcclRet != HCCL_SUCCESS || rankSize == 0 || rankSize > HCCL_MAX_RANK_SIZE) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get valid HCCL rank size failed, ret: %d, rank size: %u",
                               hcclRet, rankSize);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetLayerRanks(const HcclComm &hcclHandle, uint32_t layerId, LayerRanks &layer) const
{
    uint32_t rankNum = 0;
    uint32_t *rankList = nullptr;
    auto hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerId, &rankList, &rankNum);
    if (hcclRet != HCCL_SUCCESS || rankList == nullptr || rankNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank IDs by layer %u failed or returned an empty rank list",
                               layerId);
        return ACLNN_ERR_INNER;
    }
    layer = {layerId, std::vector<uint32_t>(rankList, rankList + rankNum)};
    return ACLNN_SUCCESS;
}

bool Mc2Context::HasLinkWithProtocol(const CommLink *linksList, uint32_t netLinkNum, const CommProtocol &protocol)
{
    for (uint32_t linkIdx = 0; linkIdx < netLinkNum; ++linkIdx) {
        if (linksList[linkIdx].linkAttr.linkProtocol == protocol) {
            return true;
        }
    }
    return false;
}

// Protocol discovery may probe layers that do not connect the rank pair; treat those probes as unsupported.
bool Mc2Context::SupportsProtocol(const HcclComm &hcclHandle, uint32_t layerId, uint32_t srcRankId, uint32_t dstRankId,
                                 const CommProtocol &protocol) const
{
    CommLink *linksList = nullptr;
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, layerId, srcRankId, dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS || linksList == nullptr || netLinkNum == 0) {
        OP_LOGW("Get HCCL links failed when checking protocol support, srcRankId: %u, dstRankId: %u, "
                "layerId: %u, ret: %d, netLinkNum: %u", srcRankId, dstRankId, layerId, hcclRet, netLinkNum);
        return false;
    }
    return HasLinkWithProtocol(linksList, netLinkNum, protocol);
}

bool Mc2Context::SupportsDomainProtocolWithAllRanks(const HcclComm &hcclHandle, const LayerRanks &layer,
                                                  const CommProtocol &domainProtocol, uint32_t srcRankId) const
{
    for (uint32_t dstRank : layer.ranks) {
        if (dstRank == srcRankId || rankLinkMap_.count(dstRank) > 0) {
            continue;
        }
        if (!SupportsProtocol(hcclHandle, layer.layer, srcRankId, dstRank, domainProtocol)) {
            return false;
        }
    }
    return true;
}

// Record a peer once so its innermost usable layer is retained.
void Mc2Context::RecordDomainLayerRanks(const CommProtocol &protocol, const LayerRanks &layer, uint32_t srcRankId)
{
    for (uint32_t dstRank : layer.ranks) {
        if (dstRank != srcRankId && rankLinkMap_.count(dstRank) == 0) {
            rankLinkMap_[dstRank] = {protocol, layer.layer};
        }
    }
}

// Extend the UB domain through consecutive inner-to-outer layers; record each peer's innermost usable layer.
aclnnStatus Mc2Context::FindUbDomain(const HcclComm &hcclHandle, const uint32_t *layerList, uint32_t layerNum,
                                  const CommProtocol &domainProtocol, uint32_t srcRankId, LayerRanks &ubDomain)
{
    bool hasDomainLayer = false;
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        LayerRanks layer;
        auto ret = GetLayerRanks(hcclHandle, layerList[layerIndex], layer);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        if (!SupportsDomainProtocolWithAllRanks(hcclHandle, layer, domainProtocol, srcRankId)) {
            break;
        }
        RecordDomainLayerRanks(domainProtocol, layer, srcRankId);
        ubDomain = std::move(layer);
        hasDomainLayer = true;
    }
    if (!hasDomainLayer) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Failed to determine UB domain for rank %u, protocol %d",
                               srcRankId, domainProtocol);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::ValidateRankLinkMap(uint32_t rankSize, uint32_t srcRankId) const
{
    if (srcRankId >= rankSize) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Invalid local rank %u for rank size %u", srcRankId, rankSize);
        return ACLNN_ERR_INNER;
    }
    for (const auto &linkEntry : rankLinkMap_) {
        if (linkEntry.first >= rankSize || linkEntry.first == srcRankId) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Invalid topology rank %u for rank size %u",
                                   linkEntry.first, rankSize);
            return ACLNN_ERR_INNER;
        }
    }
    return ACLNN_SUCCESS;
}

// Search all layers because the cross-domain protocol may differ from the protocol used to identify the UB domain.
aclnnStatus Mc2Context::CheckCrossUbDomainProtocols(const HcclComm &hcclHandle, const uint32_t *layerList,
                                                uint32_t layerNum, uint32_t srcRankId)
{
    bool hasCrossDomainPcieLink = false;
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        LayerRanks layer;
        auto ret = GetLayerRanks(hcclHandle, layerList[layerIndex], layer);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        for (uint32_t dstRank : layer.ranks) {
            if (dstRank == srcRankId || rankLinkMap_.count(dstRank) > 0) {
                continue;
            }
            if (SupportsProtocol(hcclHandle, layer.layer, srcRankId, dstRank, CommProtocol::COMM_PROTOCOL_PCIE)) {
                OP_LOGI("Rank %u supports PCIE with cross-domain rank %u in layer %u", srcRankId, dstRank, layer.layer);
                hasCrossDomainPcieLink = true;
                rankLinkMap_[dstRank] = {CommProtocol::COMM_PROTOCOL_PCIE, layer.layer};
            }
        }
    }
    if (!hasCrossDomainPcieLink) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Rank %u has no PCIE link to a peer outside its UB domain", srcRankId);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckProtocolSupport(const HcclComm &hcclHandle, const uint32_t *layerList, uint32_t layerNum,
                                           const CommProtocol &protocol)
{
    uint32_t srcRankId = 0;
    uint32_t rankSize = 0;
    auto ret = GetRankId(hcclHandle, srcRankId);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = GetRankSize(hcclHandle, rankSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    rankLinkMap_.clear();
    rankNumPerUbDomain_ = 0;

    LayerRanks ubDomain;
    ret = FindUbDomain(hcclHandle, layerList, layerNum, protocol, srcRankId, ubDomain);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ValidateRankLinkMap(rankSize, srcRankId);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    rankNumPerUbDomain_ = static_cast<uint32_t>(ubDomain.ranks.size());
    if (rankNumPerUbDomain_ > rankSize) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "UB domain rank count %u exceeds rank size %u",
                               rankNumPerUbDomain_, rankSize);
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Layer %u is current rank's UB domain, rankNumPerUbDomain_: %u", ubDomain.layer, rankNumPerUbDomain_);
    if (rankNumPerUbDomain_ != rankSize) {
        // UB_MEM links inside the domain are already resolved by FindUbDomain.
        ret = CheckCrossUbDomainProtocols(hcclHandle, layerList, layerNum, srcRankId);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = ValidateRankLinkMap(rankSize, srcRankId);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    if (rankLinkMap_.size() != rankSize - 1) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Incomplete topology for rank %u, recorded %zu of %u peers",
                               srcRankId, rankLinkMap_.size(), rankSize - 1);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol)
{
    OP_LOGD("Start to get HCCL communication protocol");
    uint32_t layerNum = 0;
    uint32_t *layerList = nullptr;
    auto ret = GetNetLayers(hcclHandle, layerList, layerNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        OP_LOGI("HCCL communication layerNum is %u,so set protocol to UB_MEM", layerNum);
        return ACLNN_SUCCESS;
    }
    return CheckProtocolSupport(hcclHandle, layerList, layerNum, protocol);
}

aclnnStatus Mc2Context::ValidateContextTag(const std::string &mc2ContextTag)
{
    if (mc2ContextTag.size() > MAX_CONTEXT_TAG_SIZE) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Mc2ContextTag is too long, max size is %u, but current size is %u",
                                MAX_CONTEXT_TAG_SIZE, mc2ContextTag.size());
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckContextCache(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                          const CommEngine &engine, void *&ctx, uint64_t &hcclBuffSize)
{
    uint64_t ctxSize = 0;
    auto hcclRet = HcclEngineCtxGet(hcclHandle, mc2ContextTag.c_str(), engine, &ctx, &ctxSize);
    if (hcclRet != HCCL_SUCCESS) { // 没找到缓存，创建context
        hcclBuffSize = 0;
        OP_LOGI("Context cache not found, need to create");
        return ACLNN_SUCCESS;
    }
    auto aclnnRet = GetHcclBufferSize(hcclHandle, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetMc2ContextTensor(const char *groupEp, const char *opName, uint64_t &hcclBuffSize,
                                            aclTensor *&mc2Context)
{
    OP_LOGI("Start to get Mc2MoeContext Tensor");
    Mc2Context instance;

    auto aclnnRet = instance.LoadHcclSymbols();
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    void *ctx = nullptr;
    CommProtocol protocol;
    std::string mc2ContextTag = std::string(groupEp) + std::string(opName);
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    hcclBuffSize = 0; // Default to 0, will be updated in CheckContextCache

    aclnnRet = instance.ValidateContextTag(mc2ContextTag);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    HcclComm hcclHandle;
    aclnnRet = instance.GetCommHandle(groupEp, hcclHandle);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    aclnnRet = instance.CheckContextCache(hcclHandle, mc2ContextTag, engine, ctx, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    if (hcclBuffSize != 0) { // Cache not found, need to create context
        aclnnRet = instance.CreatMc2ContextTensor(ctx, mc2Context);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        OP_LOGI("Found context cache, Get Mc2MoeContext Tensor Success");
        return ACLNN_SUCCESS;
    }

    aclnnRet = instance.GetCommProtocol(hcclHandle, protocol);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    Mc2MoeContext mc2ContextStruct;
    aclnnRet =
        instance.CreatMc2Context(hcclHandle, mc2ContextTag, engine, protocol, &mc2ContextStruct, ctx, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    aclnnRet = instance.CreatMc2ContextTensor(ctx, mc2Context);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    OP_LOGI("Get Mc2MoeContext Tensor Success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetMc2RankSize(const char *groupEp, uint32_t &rankSize)
{
    OP_LOGI("Start to get Mc2RankSize Tensor");
    Mc2Context instance;

    auto aclnnRet = instance.LoadHcclSymbols();
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    HcclComm hcclHandle;
    aclnnRet = instance.GetCommHandle(groupEp, hcclHandle);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    auto hcclRet = instance.HcclGetRankSize(hcclHandle, &rankSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank size failed");
        return ACLNN_ERR_INNER;
    }

    OP_LOGI("Get rank size success, rankSize is: %u", rankSize);
    return ACLNN_SUCCESS;
}

// 模板函数显式实例化
template void *Mc2Context::GetHcclLibFunc<void *>(void *handle, const std::string &funcName);

} // namespace Mc2Aclnn
#endif
