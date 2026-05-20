/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * ...
 */

#include "mc2_moe_context.h"
#if defined(HCOMM_VERSION_NUM) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

#include <cstdlib>
#include <dlfcn.h>

namespace {
constexpr uint64_t CTX_COPY_DEFAULT_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t MAX_EP_RANK_SIZE = 1024;
} // namespace

namespace ops {
std::string MoeContextManager::FUSED_OP_TYPE2_A5 = "UpdateContextA5";

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
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Ascend home path doesn't exist.");
        return "";
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
#endif
    return std::string(ascendPath) + hcclPathPostfix;
}

template <typename T>
T MoeContextManager::GetHcclLibFunc(void* handle, const std::string& funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        const char* dlErr = dlerror();
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Load func=%s error=%s in lib hccl failed.",
                  funcName.c_str(), dlErr ? dlErr : "unknown");
    }
    return func;
}

ge::graphStatus MoeContextManager::LoadSymbol(void** ptr, const char* name)
{
    *ptr = GetHcclLibFunc<void*>(hcclLibHandle_, name);
    if (*ptr == nullptr) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Failed to load %s symbol", name);
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::LoadHcclSymbols()
{
    const std::string libPath = GetLibPath();
    if (libPath.empty()) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Failed to get HCCL library path");
        return ge::GRAPH_FAILED;
    }
    hcclLibHandle_ = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        const char* dlErr = dlerror();
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "dlopen HCCL library failed: %s, error: %s",
                  libPath.c_str(), dlErr ? dlErr : "unknown");
        return ge::GRAPH_FAILED;
    }

    HcomGetCommHandleByGroup = GetHcclLibFunc<decltype(HcomGetCommHandleByGroup)>(hcclLibHandle_, "HcomGetCommHandleByGroup");
    if (HcomGetCommHandleByGroup == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclRankGraphGetLinks = GetHcclLibFunc<decltype(HcclRankGraphGetLinks)>(hcclLibHandle_, "HcclRankGraphGetLinks");
    if (HcclRankGraphGetLinks == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclRankGraphGetLayers = GetHcclLibFunc<decltype(HcclRankGraphGetLayers)>(hcclLibHandle_, "HcclRankGraphGetLayers");
    if (HcclRankGraphGetLayers == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclRankGraphGetRankSizeByLayer = GetHcclLibFunc<decltype(HcclRankGraphGetRankSizeByLayer)>(hcclLibHandle_, "HcclRankGraphGetRankSizeByLayer");
    if (HcclRankGraphGetRankSizeByLayer == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclChannelAcquire = GetHcclLibFunc<decltype(HcclChannelAcquire)>(hcclLibHandle_, "HcclChannelAcquire");
    if (HcclChannelAcquire == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclGetHcclBuffer = GetHcclLibFunc<decltype(HcclGetHcclBuffer)>(hcclLibHandle_, "HcclGetHcclBuffer");
    if (HcclGetHcclBuffer == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclChannelGetHcclBuffer = GetHcclLibFunc<decltype(HcclChannelGetHcclBuffer)>(hcclLibHandle_, "HcclChannelGetHcclBuffer");
    if (HcclChannelGetHcclBuffer == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclEngineCtxCreate = GetHcclLibFunc<decltype(HcclEngineCtxCreate)>(hcclLibHandle_, "HcclEngineCtxCreate");
    if (HcclEngineCtxCreate == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclEngineCtxGet = GetHcclLibFunc<decltype(HcclEngineCtxGet)>(hcclLibHandle_, "HcclEngineCtxGet");
    if (HcclEngineCtxGet == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclEngineCtxCopy = GetHcclLibFunc<decltype(HcclEngineCtxCopy)>(hcclLibHandle_, "HcclEngineCtxCopy");
    if (HcclEngineCtxCopy == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclGetRankId = GetHcclLibFunc<decltype(HcclGetRankId)>(hcclLibHandle_, "HcclGetRankId");
    if (HcclGetRankId == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclGetRankSize = GetHcclLibFunc<decltype(HcclGetRankSize)>(hcclLibHandle_, "HcclGetRankSize");
    if (HcclGetRankSize == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    HcclRankGraphGetRanksByLayer = GetHcclLibFunc<decltype(HcclRankGraphGetRanksByLayer)>(hcclLibHandle_, "HcclRankGraphGetRanksByLayer");
    if (HcclRankGraphGetRanksByLayer == nullptr) { dlclose(hcclLibHandle_); hcclLibHandle_ = nullptr; return ge::GRAPH_FAILED; }
    OPS_LOG_D(FUSED_OP_TYPE2_A5.c_str(), "All HCCL symbols loaded successfully");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetCommHandle(const char* groupEp, HcclComm& hcclHandle)
{
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if (ret != HCCL_SUCCESS) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL handle failed, groupEp: %s", groupEp);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetHcclCommLink(const HcclComm& hcclHandle, const LinkQueryParams& params, CommLink*& links)
{
    CommLink* linksList = nullptr;
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, params.netLayerIndex, params.srcRankId, params.dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS || netLinkNum == 0) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL Communication link failed, netLinkNum=%u", netLinkNum);
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
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "No matching protocol found, protocol=%d", params.protocol);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetRankSizePerServer(const HcclComm& hcclHandle, uint32_t netLayers)
{
    auto hcclRet = HcclRankGraphGetRankSizeByLayer(hcclHandle, netLayers, &rankSizePerServer_);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL rank size per server failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::SetupChannelDesc(const HcclComm& hcclHandle, const ChannelInitParams& params,
    uint32_t* netLayerList, uint32_t netLayerNum, std::vector<HcclChannelDesc>& channelDesc)
{
    for (uint32_t i = 0; i < params.rankDim; ++i) {
        if (i == params.srcRankId) continue;
        uint32_t dstRank = i;
        uint32_t channelId = (i > params.srcRankId) ? (i - 1) : i;
        CommLink* links = nullptr;
        uint32_t netLayerIndex = netLayerNum == 1 ? netLayerList[HCCL_COMM_LAYERS_UB_MEM] : layerMap[dstRank];
        LinkQueryParams query{netLayerIndex, params.srcRankId, dstRank, params.protocol};
        auto ret = GetHcclCommLink(hcclHandle, query, links);
        if (ret != ge::GRAPH_SUCCESS) return ret;
        channelDesc[channelId].channelProtocol = params.protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::InitHcclChannel(const HcclComm& hcclHandle, const ChannelInitParams& params,
    std::vector<HcclChannelDesc>& channelDesc)
{
    uint32_t channelNum = static_cast<uint32_t>(channelDesc.size());
    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "HCCL channel init failed");
        return ge::GRAPH_FAILED;
    }
    uint32_t netLayerNum = 0;
    uint32_t* netLayerList = nullptr;
    hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS || netLayerNum == 0) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL GetNetLayers failed");
        return ge::GRAPH_FAILED;
    }
    uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
    auto status = GetRankSizePerServer(hcclHandle, netLayers);
    if (status != ge::GRAPH_SUCCESS) return status;
    return SetupChannelDesc(hcclHandle, params, netLayerList, netLayerNum, channelDesc);
}

ge::graphStatus MoeContextManager::GetHcclCommChannel(const HcclComm& hcclHandle, const ChannelAcquireParams& params,
    std::vector<ChannelHandle>& channels)
{
    uint32_t channelNum = params.rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);
    ChannelInitParams initParams{params.rankDim, params.srcRankId, params.protocol};
    auto status = InitHcclChannel(hcclHandle, initParams, channelDesc);
    if (status != ge::GRAPH_SUCCESS) return status;
    auto hcclRet = HcclChannelAcquire(hcclHandle, params.engine, channelDesc.data(), channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Acquire HCCL channel failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::FillContextBuffer(const HcclComm& hcclHandle, const CommEngine& engine,
    Mc2MoeContext* context, uint32_t rankId, const std::vector<ChannelHandle>& channels)
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
            OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return ge::GRAPH_FAILED;
        }
        context->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetHcclCommResource(const HcclComm& hcclHandle, const CommEngine& engine,
    const CommProtocol& protocol, Mc2MoeContext* context)
{
    uint32_t rankId = context->epRankId;
    std::vector<ChannelHandle> channels;
    ChannelAcquireParams params{epRankSize_, rankId, protocol, engine};
    auto status = GetHcclCommChannel(hcclHandle, params, channels);
    if (status != ge::GRAPH_SUCCESS) return status;
    context->rankSizePerServer = rankSizePerServer_;
    return FillContextBuffer(hcclHandle, engine, context, rankId, channels);
}

ge::graphStatus MoeContextManager::CreatMc2Context(const ContextCreateParams& params, Mc2MoeContext* context)
{
    void* ctx = nullptr;
    uint64_t ctxSize = sizeof(Mc2MoeContext);
    auto hcclRet = HcclEngineCtxCreate(params.hcclHandle, params.mc2ContextTag.c_str(), params.engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Create HCCL context memory failed");
        return ge::GRAPH_FAILED;
    }
    hcclRet = HcclGetRankId(params.hcclHandle, &context->epRankId);
    if (hcclRet != HCCL_SUCCESS || context->epRankId >= MAX_EP_RANK_SIZE) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Get rank ID failed, rankId=%u", context->epRankId);
        return ge::GRAPH_FAILED;
    }
    hcclRet = HcclGetRankSize(params.hcclHandle, &epRankSize_);
    if (hcclRet != HCCL_SUCCESS || epRankSize_ == 0 || epRankSize_ >= MAX_EP_RANK_SIZE) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Get rank size failed, rankSize=%u", epRankSize_);
        return ge::GRAPH_FAILED;
    }
    auto status = GetHcclCommResource(params.hcclHandle, params.engine, params.protocol, context);
    if (status != ge::GRAPH_SUCCESS) return status;
    hcclRet = HcclEngineCtxCopy(params.hcclHandle, params.engine, params.mc2ContextTag.c_str(),
                                context, ctxSize, CTX_COPY_DEFAULT_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Copy context from host to device failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::CheckLinks(uint32_t& netLinkNum, CommLink* linksList)
{
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus MoeContextManager::CheckProtocolSupport(const HcclComm& hcclHandle, uint32_t*& layerList, uint32_t& layerNum)
{
    uint32_t srcRankId = 0;
    auto hcclRet = HcclGetRankId(hcclHandle, &srcRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "CheckProtocolSupport Get rank ID failed");
        return ge::GRAPH_FAILED;
    }
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        uint32_t rankNumInLayer = 0;
        uint32_t* rankIdLists = nullptr;
        hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if (hcclRet != HCCL_SUCCESS) {
            OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Get rank IDs by layer failed");
            return ge::GRAPH_FAILED;
        }
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if (rankIdLists[rankId] == srcRankId || layerMap.find(rankIdLists[rankId]) != layerMap.end()) continue;
            uint32_t netLinkNum = 0;
            CommLink* linksList = nullptr;
            hcclRet = HcclRankGraphGetLinks(hcclHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                            &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS || netLinkNum == 0) return ge::GRAPH_FAILED;
            if (CheckLinks(netLinkNum, linksList) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
            layerMap[rankIdLists[rankId]] = layerList[layerIndex];
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetCommProtocol(const HcclComm& hcclHandle, CommProtocol& protocol)
{
    uint32_t layerNum = 0;
    uint32_t* layerList = nullptr;
    auto ret = HcclRankGraphGetLayers(hcclHandle, &layerList, &layerNum);
    if (ret != HCCL_SUCCESS || layerNum == 0) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Get HCCL layers failed");
        return ge::GRAPH_FAILED;
    }
    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return ge::GRAPH_SUCCESS;
    }
    auto status = CheckProtocolSupport(hcclHandle, layerList, layerNum);
    if (status != ge::GRAPH_SUCCESS) return status;
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::ValidateContextTag(const std::string& mc2ContextTag)
{
    if (mc2ContextTag.size() > MAX_CONTEXT_TAG_SIZE) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Mc2ContextTag too long, max=%d, current=%zu",
                  MAX_CONTEXT_TAG_SIZE, mc2ContextTag.size());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetMc2Context(const ContextAcquireParams& params, Mc2MoeContext* context,
                                                  int64_t& hcclBuffSize)
{
    ContextCreateParams createParams{params.hcclHandle, params.engine, params.protocol, params.mc2ContextTag};
    auto status = CreatMc2Context(createParams, context);
    if (status != ge::GRAPH_SUCCESS) return status;
    hcclBuffSize = static_cast<int64_t>(hcclBuffSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeContextManager::GetHcclContext(const std::string& graphName, const std::string& groupEp,
    const std::string& nodeName, Mc2MoeContext* context, int64_t& hcclBuffSize)
{
    auto status = LoadHcclSymbols();
    if (status != ge::GRAPH_SUCCESS) return status;
    std::string mc2ContextTag = graphName + groupEp + nodeName;
    status = ValidateContextTag(mc2ContextTag);
    if (status != ge::GRAPH_SUCCESS) return status;
    HcclComm hcclHandle;
    status = GetCommHandle(groupEp.c_str(), hcclHandle);
    if (status != ge::GRAPH_SUCCESS) return status;
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    CommProtocol protocol;
    status = GetCommProtocol(hcclHandle, protocol);
    if (status != ge::GRAPH_SUCCESS) return status;
    ContextAcquireParams params{hcclHandle, engine, protocol, mc2ContextTag};
    return GetMc2Context(params, context, hcclBuffSize);
}

ge::graphStatus MoeContextManager::GetMc2ContextA5(const std::string& nodeName, const std::string& epName,
    Mc2MoeContext& context, int64_t& hcclBuffSize)
{
    if (nodeName.empty()) {
        OPS_LOG_E(FUSED_OP_TYPE2_A5.c_str(), "Node name is empty");
        return ge::GRAPH_FAILED;
    }
    MoeContextManager& manager = GetInstance();
    Mc2MoeContext ctx{};
    auto status = manager.GetHcclContext("default", epName, nodeName, &ctx, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) return status;
    context = ctx;
    return ge::GRAPH_SUCCESS;
}

template void* MoeContextManager::GetHcclLibFunc<void*>(void* handle, const std::string& funcName);

} // namespace ops
#endif // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
