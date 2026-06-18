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
#include "mc2_context.h"
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

namespace {
constexpr uint64_t KOPY_DEFAULT_CTX_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t EP_RANK_OFFSET_STEP = 8192;
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
    if (netLinkNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "The Net Link Is nullptr. srcRankId is %u, dstRankId is %u, layerId is %u", srcRankId,
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
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "No matching communication protocol found in HCCL links protocol is %d", protocol);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL communication link success, protocol is: %d", protocol);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetNetLayers(const HcclComm &hcclHandle, uint32_t *&netLayerList, uint32_t &netLayerNum)
{
    auto hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL layers failed");
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
    uint32_t layerId = 0;
    uint32_t *netLayerList = nullptr;
    auto ret = GetNetLayers(hcclHandle, netLayerList, netLayerNum);
    if (ret != ACLNN_SUCCESS || netLayerNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL net layers failed netLayerNum is: %u", netLayerNum);
        return ret;
    }

    for (uint32_t i = 0; i < rankDim; ++i) {
        if (i == srcRankId) {
            continue;
        }
        uint32_t dstRank = i;
        uint32_t channelId = (i > srcRankId) ? (i - 1) : i;
        CommLink *links = nullptr;
        layerId = netLayerNum == 1 ?
                      netLayerList[HCCL_COMM_LAYERS_UB_MEM] :
                      layerMap[dstRank]; // 如果只有一层通信，直接使用该层；如果有多层通信，使用之前记录的通信层
        ret = GetHcclCommLink(hcclHandle, layerId, srcRankId, dstRank, protocol, links);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        channelDesc[channelId].channelProtocol = protocol;
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

aclnnStatus Mc2Context::CheckLinks(uint32_t &netLinkNum, CommLink *linksList)
{
    bool isFoundUbMemProtocol = false;
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            isFoundUbMemProtocol = true;
            break;
        }
    }
    return isFoundUbMemProtocol ? ACLNN_SUCCESS : ACLNN_ERR_INNER;
}

/**
 * 检查协议支持 & 跨超节点判断，分三步：
 * 步骤3-4：逐层遍历组网层，检查本卡与每层所有卡是否全部支持UB_MEM协议，
 *          若全支持则记录本UB域信息(rankNumPerUbDomain_/layerNumMyUbDomain_)，
 *          若不支持则判断已访问卡数是否等于整个域卡数：
 *            - 等于：全域仅支持UB_MEM，isCrossServer=false，直接返回
 *            - 不等：跳出循环，进入URMA校验
 * 步骤5：验证rankNumPerUbDomain_ < rankSize且rankSize能被rankNumPerUbDomain_整除
 * 步骤6：在最大组网层上，排除本UB域的卡，检查剩余跨超卡是否全部支持URMA协议
 */
aclnnStatus Mc2Context::CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList, uint32_t &layerNum,
                                               bool &isCrossServer)
{
    isCrossServer = false;
    uint32_t srcRankId = 0;
    uint32_t rankNumInLayer = 0;
    uint32_t *rankIdLists = nullptr;
    CommLink *linksList = nullptr;
    uint32_t netLinkNum = 0;
    rankNumPerUbDomain_ = 0; // 本卡所属UB域的卡数，初始为0
    layerNumMyUbDomain_ = 0; // 本卡所属UB域对应的组网层号，初始为0

    auto hcclRet = HcclGetRankId(hcclHandle, &srcRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "CheckProtocolSupport Get rank ID failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("CheckProtocolSupport Get rank ID success, rankId is: %u", srcRankId);

    // 步骤3：按HcclRankGraphGetLayers获取到的组网层数逐层遍历
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        OP_LOGD("CheckProtocolSupport Check layer %u", layerList[layerIndex]);
        // 步骤4：调用HcclRankGraphGetRanksByLayer获取当前层的所有卡ID
        hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank IDs by layer failed");
            return ACLNN_ERR_INNER;
        }

        // 步骤4：检查本卡与该层所有卡之间是否全部支持UB_MEM协议
        bool allSupportUbMem = true;
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if (rankIdLists[rankId] == srcRankId ||
                layerMap.find(rankIdLists[rankId]) != layerMap.end()) {
                continue;
            }
            hcclRet = HcclRankGraphGetLinks(hcclHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                            &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS) {
                OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL links failed");
                return ACLNN_ERR_INNER;
            }
            if (netLinkNum == 0) {
                OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                        "No available HCCL links found, srcRankID %u, dstRankID %u layer is %u",
                        srcRankId, rankIdLists[rankId], layerList[layerIndex]);
                return ACLNN_ERR_INNER;
            }
            // 复用现有CheckLinks逻辑判断UB_MEM协议是否支持
            if (CheckLinks(netLinkNum, linksList) != ACLNN_SUCCESS) {
                OP_LOGD("Mc2Context",
                        "UB_MEM not supported for rank %u on layer %u, checking if all visited ranks cover domain",
                        rankIdLists[rankId], layerList[layerIndex]);
                allSupportUbMem = false;
                break;
            }
            layerMap[rankIdLists[rankId]] = layerList[layerIndex];
        }

        // 步骤4续：若该层所有卡都支持UB_MEM，更新本UB域信息
        if (allSupportUbMem) {
            // rankNumPerUbDomain_记录本UB域卡数，layerNumMyUbDomain_记录对应组网层号
            rankNumPerUbDomain_ = rankNumInLayer;
            layerNumMyUbDomain_ = layerList[layerIndex];
        } else {
            // 步骤4续：若不支持，判断已访问卡数(rankNumPerUbDomain_)是否等于整个域卡数(epRankSize_)
            // 若相等则代表整个通信域只支持UB_MEM协议，直接返回成功
            if (rankNumPerUbDomain_ == epRankSize_) {
                OP_LOGI("All visited ranks cover entire domain, only UB_MEM supported");
                isCrossServer = false;
                return ACLNN_SUCCESS;
            }
            // 若不等则跳出循环，进入步骤5-6的URMA校验
            OP_LOGD("Not all ranks support UB_MEM, breaking to check URMA");
            break;
        }
    }

    // 步骤4续：若不存在任何UB域层，报错退出
    if (rankNumPerUbDomain_ == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "No UB domain found where all ranks support UB_MEM");
        return ACLNN_ERR_INNER;
    }

    // 步骤5：验证rankNumPerUbDomain_ < rankSize且rankSize能被rankNumPerUbDomain_整除
    // 若不满足，说明UB域卡数与总卡数不匹配，报错退出
    if (rankNumPerUbDomain_ >= epRankSize_ || epRankSize_ % rankNumPerUbDomain_ != 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                "Invalid UB domain size: rankNumPerUbDomain_=%u, rankSize=%u",
                rankNumPerUbDomain_, epRankSize_);
        return ACLNN_ERR_INNER;
    }

    // 步骤6：在最大组网层上获取全网卡ID，排除本UB域卡，逐一检查跨超卡是否支持URMA协议
    uint32_t maxLayer = layerList[layerNum - 1]; // 取layerList中最大层数值
    uint32_t *allRankIds = nullptr;
    uint32_t allRankNum = 0;
    // 获取整个组网的卡ID
    hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, maxLayer, &allRankIds, &allRankNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank IDs by max layer failed");
        return ACLNN_ERR_INNER;
    }

    // 获取本UB域的卡ID列表，用于排除域内卡
    uint32_t *myUbDomainRankIds = nullptr;
    uint32_t myUbDomainRankNum = 0;
    hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerNumMyUbDomain_, &myUbDomainRankIds, &myUbDomainRankNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get UB domain rank IDs failed");
        return ACLNN_ERR_INNER;
    }

    // 步骤6续：遍历全网卡ID，排除本UB域的卡，检查剩余跨超卡是否均支持URMA协议
    for (uint32_t i = 0; i < allRankNum; ++i) {
        if (allRankIds[i] == srcRankId) {
            continue;
        }

        // 判断该卡是否属于本UB域，域内卡跳过
        bool isInMyUbDomain = false;
        for (uint32_t j = 0; j < myUbDomainRankNum; ++j) {
            if (allRankIds[i] == myUbDomainRankIds[j]) {
                isInMyUbDomain = true;
                break;
            }
        }
        if (isInMyUbDomain) {
            continue;
        }

        // 跨超卡：检查本卡与该卡之间是否支持URMA协议（复用现有linkProtocol判断逻辑）
        hcclRet = HcclRankGraphGetLinks(hcclHandle, maxLayer, srcRankId, allRankIds[i],
                                        &linksList, &netLinkNum);
        if (hcclRet != HCCL_SUCCESS || netLinkNum == 0) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                    "Get HCCL links failed for cross-server rank %u", allRankIds[i]);
            return ACLNN_ERR_INNER;
        }

        bool supportUrma = false;
        for (uint32_t j = 0; j < netLinkNum; ++j) {
            if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_URMA) {
                supportUrma = true;
                break;
            }
        }
        if (!supportUrma) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                    "URMA protocol not supported for cross-server rank %u", allRankIds[i]);
            return ACLNN_ERR_INNER;
        }
        layerMap[allRankIds[i]] = maxLayer;
    }

    // 确认跨超组网，isCrossServer设置为true
    isCrossServer = true;
    OP_LOGI("Cross-server confirmed with URMA support, rankNumPerUbDomain_=%u, layerNumMyUbDomain_=%u",
            rankNumPerUbDomain_, layerNumMyUbDomain_);
    return ACLNN_SUCCESS;
}

/**
 * 获取通信协议 & 判断是否跨超节点：
 * 步骤1：调用HcclRankGraphGetLayers获取组网层数layerNum
 * 步骤2：若layerNum==1则不跨超，直接选用UB_MEM协议
 *         若layerNum>=2则调用CheckProtocolSupport进一步判断是否跨超及URMA支持
 */
aclnnStatus Mc2Context::GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol, bool &isCrossServer)
{
    OP_LOGD("Start to get HCCL communication protocol");
    isCrossServer = false;
    // 步骤1：调用HcclRankGraphGetLayers获取组网层数
    uint32_t layerNum = 0;
    uint32_t *layerList = nullptr;
    auto ret = HcclRankGraphGetLayers(hcclHandle, &layerList, &layerNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL layers failed");
        return ACLNN_ERR_INNER;
    }

    // 步骤2：若layerNum==1，表示只有一个UB域，不跨超，直接选用UB_MEM协议
    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        OP_LOGI("HCCL communication layerNum is %u, not cross-server, set protocol to UB_MEM", layerNum);
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return ACLNN_SUCCESS;
    }

    // 步骤2续：若layerNum>=2，需进一步判断是否跨超节点
    OP_LOGD("start CheckProtocolSupport, layerNum is %u", layerNum);

    auto aclnnRet = CheckProtocolSupport(hcclHandle, layerList, layerNum, isCrossServer);
    if (aclnnRet != ACLNN_SUCCESS) {
        return aclnnRet;
    }

    if (isCrossServer) {
        OP_LOGI("Cross-server scenario confirmed, protocol set to UB_MEM for intra-server, URMA for cross-server");
    } else {
        OP_LOGI("Not cross-server, protocol set to UB_MEM");
    }
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    return ACLNN_SUCCESS;
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

/**
 * 获取URMA通信句柄 & 跨超通信资源（步骤7）：
 * 确认跨超组网后，仅对跨超rank通过HcclChannelAcquire获取URMA channel handle，
 * 保存至hcommHandle_中随context传入kernel；
 * 同时通过HcclChannelGetHcclBuffer获取URMA通信buffer地址，更新epHcclBuffer_中跨超rank的地址
 */
aclnnStatus Mc2Context::GetHcclCommHandle(const HcclComm &hcclHandle, const CommProtocol &protocol,
                                            Mc2MoeContext *mc2ContextStruct)
{
    OP_LOGD("Start to get HCCL URMA communication handle for cross-server ranks");
    uint32_t rankId = mc2ContextStruct->epRankId;

    // 步骤7：确认跨超组网后，通过layerMap和layerNumMyUbDomain_确定哪些rank是跨超rank
    // 跨超rank的特征：layerMap[i] != layerNumMyUbDomain_（不在本UB域内）
    std::vector<uint32_t> crossServerRanks;
    for (uint32_t i = 0; i < epRankSize_; ++i) {
        if (i == rankId) {
            continue;
        }
        if (layerMap[i] != layerNumMyUbDomain_) {
            crossServerRanks.push_back(i);
        }
    }

    uint32_t channelNum = crossServerRanks.size();
    if (channelNum == 0) {
        OP_LOGI("No cross-server ranks found, skip URMA channel acquisition");
        return ACLNN_SUCCESS;
    }

    // 步骤7续：为每个跨超rank创建URMA channel描述并获取对应通信链路
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "URMA channel desc init failed");
        return ACLNN_ERR_INNER;
    }

    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        uint32_t dstRank = crossServerRanks[idx];
        CommLink *links = nullptr;
        auto ret = GetHcclCommLink(hcclHandle, layerMap[dstRank], rankId, dstRank, protocol, links);
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get URMA comm link failed for rank %u", dstRank);
            return ret;
        }
        channelDesc[idx].channelProtocol = protocol;
        channelDesc[idx].remoteRank = dstRank;
        channelDesc[idx].localEndpoint = links->srcEndpointDesc;
        channelDesc[idx].remoteEndpoint = links->dstEndpointDesc;
    }

    // 调用HcclChannelAcquire获取URMA channel，返回的句柄即为所需handle
    std::vector<ChannelHandle> channels(channelNum);
    hcclRet = HcclChannelAcquire(hcclHandle, CommEngine::COMM_ENGINE_AIV, channelDesc.data(),
                                  channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Acquire URMA channel failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Acquire URMA channel success, channel num is: %u", channelNum);

    // 步骤7续：初始化hcommHandle_全部为0，域内rank不需要URMA handle
    for (uint32_t i = 0; i < epRankSize_; ++i) {
        mc2ContextStruct->hcommHandle_[i] = 0; // 域内rank URMA handle置0
    }

    // 步骤7续：对每个跨超rank，保存URMA channel handle至hcommHandle_，
    // 并通过HcclChannelGetHcclBuffer获取URMA通信buffer地址更新epHcclBuffer_
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        uint32_t dstRank = crossServerRanks[idx];
        // 保存HcclChannelAcquire返回的句柄至hcommHandle_[dstRank]，随context传入kernel
        mc2ContextStruct->hcommHandle_[dstRank] = reinterpret_cast<uint64_t>(channels[idx]);

        // 步骤7续：通过HcclChannelGetHcclBuffer获取URMA通信所需的buffer地址
        void *tempBuffer = nullptr;
        uint64_t bufSize = 0;
        hcclRet = HcclChannelGetHcclBuffer(hcclHandle, channels[idx], &tempBuffer, &bufSize);
        if (hcclRet != HCCL_SUCCESS || tempBuffer == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get URMA buffer failed, src: %u, dst: %u", rankId, dstRank);
            return ACLNN_ERR_INNER;
        }
        // 更新跨超rank的epHcclBuffer_为URMA buffer地址
        mc2ContextStruct->epHcclBuffer_[dstRank] = reinterpret_cast<uint64_t>(tempBuffer) + dstRank * EP_RANK_OFFSET_STEP;
    }

    OP_LOGI("Get HCCL URMA communication handle success, cross-server rank num: %u", channelNum);
    return ACLNN_SUCCESS;
}

/**
 * 获取Mc2MoeContext Tensor主流程：
 * 1. 判断是否跨超节点（通过GetCommProtocol内部CheckProtocolSupport完成）
 * 2. 若跨超，进一步验证所有卡之间是否都支持URMA通信（CheckProtocolSupport步骤6）
 * 3. 分别获取对应通信地址：
 *    - CreatMc2Context获取UB_MEM通信资源（域内rank的epHcclBuffer_）
 *    - GetHcclCommHandle获取URMA通信资源（跨超rank的hcommHandle_和epHcclBuffer_）
 */
aclnnStatus Mc2Context::GetMc2ContextTensor(const char *groupEp, const char *opName, uint64_t &hcclBuffSize,
                                            aclTensor *&mc2Context, bool &isCrossServer)
{
    OP_LOGI("Start to get Mc2MoeContext Tensor");
    Mc2Context instance;

    auto aclnnRet = instance.LoadHcclSymbols();
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    void *ctx = nullptr;
    CommProtocol protocol;
    std::string mc2ContextTag = std::string(groupEp) + std::string(opName);
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    hcclBuffSize = 0;
    isCrossServer = false;

    aclnnRet = instance.ValidateContextTag(mc2ContextTag);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    HcclComm hcclHandle;
    aclnnRet = instance.GetCommHandle(groupEp, hcclHandle);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    aclnnRet = instance.CheckContextCache(hcclHandle, mc2ContextTag, engine, ctx, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    if (hcclBuffSize != 0) {
        aclnnRet = instance.CreatMc2ContextTensor(ctx, mc2Context);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        OP_LOGI("Found context cache, Get Mc2MoeContext Tensor Success");
        return ACLNN_SUCCESS;
    }

    // 流程1-2：判断是否跨超节点 & 验证URMA通信支持，isCrossServer出参标识结果
    aclnnRet = instance.GetCommProtocol(hcclHandle, protocol, isCrossServer);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    // 流程3a：获取UB_MEM通信资源（域内rank的epHcclBuffer_和epHcclBuffer_初始化）
    Mc2MoeContext mc2ContextStruct;
    aclnnRet = instance.CreatMc2Context(hcclHandle, mc2ContextTag, engine, protocol,
                                        &mc2ContextStruct, ctx, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    // 流程3b：若跨超，获取URMA通信资源（跨超rank的hcommHandle_和epHcclBuffer_更新）
    if (isCrossServer) {
        aclnnRet = instance.GetHcclCommHandle(hcclHandle, CommProtocol::COMM_PROTOCOL_URMA, &mc2ContextStruct);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    }

    aclnnRet = instance.CreatMc2ContextTensor(ctx, mc2Context);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    OP_LOGI("Get Mc2MoeContext Tensor Success, isCrossServer: %d", isCrossServer);
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
