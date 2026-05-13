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
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

namespace {
constexpr uint64_t KOPY_DEFAULT_CTX_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t INIT_CHANNEL_NUM_ONE = 1;
constexpr uint32_t INIT_CHANNEL_NUM_TWO = 2;
 
} // namespace

namespace Mc2Aclnn {

const std::string Mc2Context::GetLibPath()
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Ascend home path doesn't exist.");
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
        OP_LOGE(ACLNN_ERR_INNER, "Load func=%s error=%s in lib hccl failed.", funcName.c_str(), dlerror());
    }
    return func;
}

Mc2Context &Mc2Context::GetInstance()
{
    static Mc2Context instance;
    return instance;
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
        OP_LOGE(ACLNN_ERR_INNER, "Failed to get HCCL library path");
        return ACLNN_ERR_INNER;
    }

    hcclLibHandle_ = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "dlopen HCCL library failed: %s, error: %s", libPath.c_str(), dlerror());
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
            OP_LOGE(ACLNN_ERR_INNER, "Failed to load %s symbol", sym.name);
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
    OP_LOGD("Start to get HCCL communication handle, groupEp: %s", groupEp);
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL handle failed, groupEp: %s", groupEp);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL communication handle success hcclHandle is: %p", hcclHandle);
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
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Communication link failed");
        return ACLNN_ERR_INNER;
    }
    if (netLinkNum == 0) {
        OP_LOGE(ACLNN_ERR_INNER, "The Net Link Is nullptr. srcRankId is %u, dstRankId is %u, layerId is %u", srcRankId,
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
        OP_LOGE(ACLNN_ERR_INNER, "No matching communication protocol found in HCCL links protocol is %d", protocol);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL communication link success, protocol is: %d", protocol);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetNetLayers(const HcclComm &hcclHandle, uint32_t *&netLayerList, uint32_t &netLayerNum)
{
    auto hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL layers failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL layers success, netLayerNum is: %u", netLayerNum);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::InitHcclChannel(const HcclComm &hcclHandle, uint32_t srcRankId,const CommProtocol &protocol,
                                        std::vector<std::vector<HcclChannelDesc>> &channelDesc)
{
    for (auto &channel : channelDesc) {
        uint32_t channelNum = channel.size();
        auto hcclRet = HcclChannelDescInit(channel.data(), channelNum);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "HCCL channel init failed");
            return ACLNN_ERR_INNER;
        }
    }
    OP_LOGD("HCCL channel init success");
    // 初始化UB域内通信通道描述符
    uint32_t netLayerNum = 0;
    uint32_t layerId = 0;
    uint32_t *netLayerList = nullptr;
    auto ret = GetNetLayers(hcclHandle, netLayerList, netLayerNum);
    if (ret != ACLNN_SUCCESS || netLayerNum == 0) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL net layers failed netLayerNum is: %u", netLayerNum);
        return ret;
    }
    for (uint32_t i = 0; i < channelDesc[INIT_CHANNEL_NUM_ONE].size(); ++i) {
        if (i == srcRankId) {
            continue;
        }
        uint32_t dstRank = i;
        uint32_t channelId = (i > srcRankId) ? (i - 1) : i;
        CommLink *links = nullptr;
        layerId = netLayerNum == 1 ?
                      netLayerList[HCCL_COMM_LAYERS_UB_MEM] :
                      layerUBMEMMap_[dstRank]; // 如果只有一层通信，直接使用该层；如果有多层通信，使用之前记录的通信层
        ret = GetHcclCommLink(hcclHandle, layerId, srcRankId, dstRank, CommProtocol::COMM_PROTOCOL_UB_MEM, links);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        channelDesc[INIT_CHANNEL_NUM_ONE][channelId].channelProtocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        channelDesc[INIT_CHANNEL_NUM_ONE][channelId].remoteRank = dstRank;
        channelDesc[INIT_CHANNEL_NUM_ONE][channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[INIT_CHANNEL_NUM_ONE][channelId].remoteEndpoint = links->dstEndpointDesc;
    }

    // 初始化ROCE通信通道描述符
    if (protocol == CommProtocol::COMM_PROTOCOL_ROCE) {
        for (uint32_t i = 0; i < ubDomainNum_; i++) {
            if (i == srcRankId / rankNumPerUbDomain_) {
                continue;
            }
            uint32_t dstRank = i * rankNumPerUbDomain_ + srcRankId % rankNumPerUbDomain_;
            uint32_t channelId = (dstRank > srcRankId) ? (i - 1) : i;
            CommLink *links = nullptr;
            layerId = layerROCEMap_[dstRank]; // 从ROCE通信层映射表中获取ROCE通信层ID
            ret = GetHcclCommLink(hcclHandle, layerId, srcRankId, dstRank, CommProtocol::COMM_PROTOCOL_ROCE, links);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
            channelDesc[INIT_CHANNEL_NUM_TWO][channelId].channelProtocol = CommProtocol::COMM_PROTOCOL_ROCE;
            channelDesc[INIT_CHANNEL_NUM_TWO][channelId].remoteRank = dstRank;
            channelDesc[INIT_CHANNEL_NUM_TWO][channelId].localEndpoint = links->srcEndpointDesc;
            channelDesc[INIT_CHANNEL_NUM_TWO][channelId].remoteEndpoint = links->dstEndpointDesc;
        }
    }
    return ACLNN_SUCCESS;
}



aclnnStatus Mc2Context::GetHcclCommChannel(const HcclComm &hcclHandle, uint32_t srcRankId,
                                           const CommProtocol &protocol, const CommEngine &engine,
                                           std::vector<std::vector<ChannelHandle>> &channels)
{
    OP_LOGD("Start to get HCCL communication channel");
    uint32_t initChannelNum = protocol == CommProtocol::COMM_PROTOCOL_UB_MEM ?
                                          INIT_CHANNEL_NUM_ONE : INIT_CHANNEL_NUM_TWO;
    uint32_t ubMemChannelArrNum = rankNumPerUbDomain_ - 1; // ub_mem通信通道数组大小为卡数减1
    uint32_t roceChannelArrNum = ubDomainNum_ - 1; // roce通信通道数组大小为ub域数减1，因为本卡只会与其他UB域内同层次的卡通信
    channels.resize(initChannelNum);
    std::vector<std::vector<HcclChannelDesc>> channelDesc(initChannelNum);
    channels[INIT_CHANNEL_NUM_ONE].resize(ubMemChannelArrNum);
    channelDesc[INIT_CHANNEL_NUM_ONE].resize(ubMemChannelArrNum);
    if (protocol == CommProtocol::COMM_PROTOCOL_ROCE) {
        channels[INIT_CHANNEL_NUM_TWO].resize(roceChannelArrNum);
        channelDesc[INIT_CHANNEL_NUM_TWO].resize(roceChannelArrNum);
    }

    auto ret = InitHcclChannel(hcclHandle, srcRankId, protocol, channelDesc);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    
    auto hcclRet = HcclChannelAcquire(hcclHandle, engine, channelDesc[INIT_CHANNEL_NUM_ONE].data(),
                                      ubMemChannelArrNum, channels[INIT_CHANNEL_NUM_ONE].data());
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Acquire HCCL ub_mem channel failed");
        return ACLNN_ERR_INNER;
    }
    if (protocol == CommProtocol::COMM_PROTOCOL_ROCE) {
        hcclRet = HcclChannelAcquire(hcclHandle, engine, channelDesc[INIT_CHANNEL_NUM_TWO].data(), 
                                     roceChannelArrNum, channels[INIT_CHANNEL_NUM_TWO].data());
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Acquire HCCL roce channel failed");
            return ACLNN_ERR_INNER;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclCommResource(const HcclComm &hcclHandle, const CommEngine &engine,
                                            const CommProtocol &protocol, Mc2MoeContext *mc2ContextStruct)
{
    OP_LOGD("Start to get HCCL communication resource");
    uint32_t rankId = mc2ContextStruct->epRankId;
    uint32_t relativeIndex = rankId % rankNumPerUbDomain_;
    std::vector<std::vector<ChannelHandle>> Channels;
    HcclResult hcclRet;

    auto ret = GetHcclCommChannel(hcclHandle, rankId, protocol, engine, Channels);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    OP_LOGD("Get HCCL communication channel success");

    for (uint32_t i = 0; i < rankNumPerUbDomain_; ++i) {
        void *tempBuffer = nullptr;
        uint64_t bufSize = 0;

        if (i == relativeIndex) {
            hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize_);
        } else {
            uint32_t idx = (i < relativeIndex) ? i : (i - 1);
            hcclRet = HcclChannelGetHcclBuffer(hcclHandle, Channels[INIT_CHANNEL_NUM_ONE][idx], &tempBuffer, &bufSize);
        }

        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return ACLNN_ERR_INNER;
        }
        mc2ContextStruct->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer);
    }
    if (protocol == CommProtocol::COMM_PROTOCOL_ROCE) {
        hcclRet = HcomChannelGetPtrByHandle(Channels[INIT_CHANNEL_NUM_TWO].data(), Channels[INIT_CHANNEL_NUM_TWO].size(), &mc2ContextStruct->dpuCommHandle_);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Get HCCL roce channel ptr failed");
            return ACLNN_ERR_INNER;
        }
    }
    OP_LOGD("Get HCCL CommResource success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CreateMc2Context(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                        const CommEngine &engine, const CommProtocol &protocol, void *&ctx,
                                        Mc2MoeContext *mc2ContextStruct, uint64_t &hcclBuffSize)
{
    OP_LOGD("Start to create HCCL context");
    uint64_t ctxSize = sizeof(Mc2MoeContext);
    auto hcclRet = HcclEngineCtxCreate(hcclHandle, mc2ContextTag.c_str(), engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Create HCCL context memory failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Create HCCL context success, ctx: %p", ctx);

    if (protocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
        rankSizePerServer_ = epRankSize_; // 全部都是UB_MEM协议下，单个server卡数为epRankSize_
    } else  if (protocol == CommProtocol::COMM_PROTOCOL_ROCE) {
        rankSizePerServer_ = rankNumPerUbDomain_; // ROCE协议下，单个server卡数为UB域卡数
    }
    ubDomainNum_ = epRankSize_ / rankNumPerUbDomain_; // 计算UB域数量
    mc2ContextStruct->epRankId = epRankId_; // 记录本卡的rankId
    mc2ContextStruct->rankSizePerServer = rankSizePerServer_; // 记录单个ub域卡数
    OP_LOGD("ubDomainNum_ is: %u, rankSizePerServer_ is: %u,", ubDomainNum_, rankSizePerServer_);

    auto ret = GetHcclCommResource(hcclHandle, engine, protocol, mc2ContextStruct);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL communication resource failed");
        return ret;
    }

    hcclBuffSize = hcclBuffSize_;
    OP_LOGD("hcclBuffSize_ is: %u", hcclBuffSize_);

    hcclRet = HcclEngineCtxCopy(hcclHandle, engine, mc2ContextTag.c_str(), mc2ContextStruct, ctxSize,
                                KOPY_DEFAULT_CTX_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Copy context from host to device failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Copy context from host to device success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CreatMc2ContextTensor(void *ctx, aclTensor *&mc2Context)
{
    OP_LOGD("Start to create Mc2Context Tensor");
    if (ctx == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Create Mc2Context Tensor failed ctx is nullptr.");
        return ACLNN_ERR_INNER;
    }

    uint64_t mc2ContextLength = sizeof(Mc2MoeContext);
    int64_t shape[1] = {static_cast<int64_t>(mc2ContextLength / sizeof(uint32_t))};
    int64_t strides[1] = {1};

    mc2Context = aclCreateTensor(shape, 1, ACL_INT32, strides, 0, ACL_FORMAT_ND, shape, 1, ctx);
    if (mc2Context == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Create Mc2Context Tensor failed.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("CreatMc2ContextTensor Success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclBufferSize(const HcclComm &hcclHandle, uint64_t &hcclBuffSize)
{
    void *tempBuffer = nullptr;
    auto hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Buffer Size failed");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckLinks(uint32_t &netLinkNum, CommLink *linksList, CommProtocol &protocol)
{
    bool isFoundSupportProtocol = false;
    if (protocol == CommProtocol::COMM_PROTOCOL_UB_MEM) { //目前还在UB域中，需要校验是否有UB_MEM协议,若是有ROCE协议,则切换到ROCE域
        for (uint32_t j = 0; j < netLinkNum; ++j) {
            if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
                isFoundSupportProtocol = true;
                break;
            }
            if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_ROCE) {
                protocol = CommProtocol::COMM_PROTOCOL_ROCE;
                isFoundSupportProtocol = true;
                break;
            }
        }
        //  如果是转换到UB域间了，第一次判断时有可能两张卡即不支持UB_MEM, 也不支持ROCE协议，这种情况下不应该报错
        //  正常情况下，这种情况netLinkNum应该为0，但是为了防止不标准的情况，这里做一个冗余防御处理
        //  这种情况下，直接将protocol设置为ROCE协议，
        if (!isFoundSupportProtocol) {
            protocol = CommProtocol::COMM_PROTOCOL_ROCE;
            isFoundSupportProtocol = true;
        }
    } 
    if (protocol == CommProtocol::COMM_PROTOCOL_ROCE) { // 在ROCE域中，需要校验是否有ROCE协议，若没有则报错
        for (uint32_t j = 0; j < netLinkNum; ++j) {
            if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_ROCE) {
                isFoundSupportProtocol = true;
                break;
            }
        }
        if (!isFoundSupportProtocol) {
            OP_LOGE(ACLNN_ERR_INNER, "No ROCE or UB_MEM protocol found in links, please check the Hccl Configuration.");
        }
    }
    return isFoundSupportProtocol ? ACLNN_SUCCESS : ACLNN_ERR_INNER;
}

aclnnStatus Mc2Context::GetEpRankIdAndSize(const HcclComm &hcclHandle)
{
    auto hcclRet = HcclGetRankId(hcclHandle, &epRankId_);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get rank ID failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get rank ID success, rankId is: %u", epRankId_);

    hcclRet = HcclGetRankSize(hcclHandle, &epRankSize_);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get rank size failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get rank size success, rankSize is: %u", epRankSize_);
}

aclnnStatus Mc2Context::CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList, uint32_t &layerNum,
                                             CommProtocol &protocol)
{
    uint32_t netLinkNum = 0;
    uint32_t rankNumInLayer = 0;
    uint32_t *rankIdLists = nullptr;
    CommLink *linksList = nullptr;
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM; // 假设所有卡默认支持UB_MEM协议

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        OP_LOGD("CheckProtocolSupport Check layer %u", layerList[layerIndex]);
        auto hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Get rank IDs by layer failed");
            return ACLNN_ERR_INNER;
        }
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if (rankIdLists[rankId] == epRankId_ ||
                layerUBMEMMap_.find(rankIdLists[rankId]) != layerUBMEMMap_.end()) { // 本卡或者已经校验过的卡跳过
                continue;
            }
            if (protocol == CommProtocol::COMM_PROTOCOL_ROCE) {
               if(rankIdLists[rankId] % rankNumPerUbDomain_ != epRankId_ % rankNumPerUbDomain_ || 
                 layerROCEMap_.find(rankIdLists[rankId]) != layerROCEMap_.end()) { // DPU只需要知道每个UE域中同级别的卡地址
                    continue;
               }
            }
            hcclRet = HcclRankGraphGetLinks(hcclHandle, layerList[layerIndex], epRankId_, rankIdLists[rankId],
                                            &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS) {
                OP_LOGE(ACLNN_ERR_INNER, "Get HCCL links failed");
                return ACLNN_ERR_INNER;
            }
            if (netLinkNum == 0) { // DPU场景下, 需要考虑第一次跨UE域时获取链接时可能存在没有链接的情况，这种情况并不算错误，需要跳过
                if (protocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
                    if ((rankNumInLayer == epRankSize_) && (rankId == rankNumInLayer - 1)) {
                        OP_LOGD("The last rank cannot use ROCE protocol, please check the hccl Configuration.");
                        return ACLNN_ERR_INNER;
                    }
                    protocol = CommProtocol::COMM_PROTOCOL_ROCE;
                    OP_LOGD("protocol is UB_MEM, but no available HCCL links found, set protocol to ROCE_DPU");
                    continue; 
                }
                OP_LOGE("No available HCCL links found, layer is %u, protocol is %u", layerList[layerIndex], protocol);
                return ACLNN_ERR_INNER; //  如果已经是ROCE协议，且没有链接，返回错误
            }
            if (CheckLinks(netLinkNum, linksList, protocol) != ACLNN_SUCCESS) {
                OP_LOGE(ACLNN_ERR_INNER, "No HCCL links support %s, srcRankID %u, dstRankID %u, layer is %u",
                        protocol == CommProtocol::COMM_PROTOCOL_UB_MEM ? "UB_MEM" : "ROCE_DPU",
                        epRankId_, rankIdLists[rankId], layerList[layerIndex]);
                return ACLNN_ERR_INNER;
            }
            if (protocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
                layerUBMEMMap_[rankIdLists[rankId]] = layerList[layerIndex];
            } else if (protocol == CommProtocol::COMM_PROTOCOL_ROCE && 
                       rankIdLists[rankId] % rankNumPerUbDomain_ == epRankId_ % rankNumPerUbDomain_) {
                layerROCEMap_[rankIdLists[rankId]] = layerList[layerIndex];
            }
        }
        if (protocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            rankNumPerUbDomain_ = rankNumInLayer; // 记录当前UB域中卡的数量
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol)
{
    OP_LOGD("Start to get HCCL communication protocol");
    uint32_t layerNum = 0;
    uint32_t *layerList = nullptr;

    auto aclnnRet = GetEpRankIdAndSize(hcclHandle);
    if (aclnnRet != ACLNN_SUCCESS) {
        return aclnnRet;
    }

    auto ret = HcclRankGraphGetLayers(hcclHandle, &layerList, &layerNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL layers failed");
        return ACLNN_ERR_INNER;
    }

    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM; // COMM_PROTOCOL_UB_MEM
        return ACLNN_SUCCESS;
    }

    OP_LOGD("start CheckProtocolSupport, layerNum is %u", layerNum);

    aclnnRet = CheckProtocolSupport(hcclHandle, layerList, layerNum, protocol);
    if (aclnnRet != ACLNN_SUCCESS) {
        return aclnnRet;
    }

    std::string protocolStr = protocol == CommProtocol::COMM_PROTOCOL_UB_MEM ? "MTE_UB_MEM" : "ROCE_DPU";
    OP_LOGD("CheckProtocolSupport success!, protocol is %s", protocolStr.c_str());
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::ValidateContextTag(const std::string &mc2ContextTag)
{
    if (mc2ContextTag.size() > MAX_CONTEXT_TAG_SIZE) {
        OP_LOGE(ACLNN_ERR_INNER, "Mc2ContextTag is too long, max size is %d, but current size is %d",
                MAX_CONTEXT_TAG_SIZE, mc2ContextTag.size());
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckHcclContext(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                         const CommEngine &engine, void *&ctx, uint64_t &hcclBuffSize)
{
    uint64_t ctxSize = 0;
    auto hcclRet = HcclEngineCtxGet(hcclHandle, mc2ContextTag.c_str(), engine, &ctx, &ctxSize);
    if (hcclRet == HCCL_SUCCESS) {
        auto aclnnRet = GetHcclBufferSize(hcclHandle, hcclBuffSize);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        return ACLNN_SUCCESS;
    }
    hcclBuffSize = 0; // 为了做一个标志位，表示是否需要创建Mc2ContextTensor
    return ACLNN_SUCCESS;
}


aclnnStatus Mc2Context::GetMc2ContextTensor(const char *groupEp, const char *opName, uint64_t &hcclBuffSize,
                                            aclTensor *&mc2Context)
{
    OP_LOGD("Start to get Mc2MoeContext Tensor");
    auto &instance = GetInstance();

    auto aclnnRet = instance.LoadHcclSymbols();
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    void *ctx = nullptr;
    CommProtocol protocol;
    std::string mc2ContextTag = std::string(groupEp) + std::string(opName);
    CommEngine engine = CommEngine::COMM_ENGINE_AIV; // 目前DPU与MTE通信仅支持AIV引擎

    aclnnRet = instance.ValidateContextTag(mc2ContextTag);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    HcclComm hcclHandle;
    aclnnRet = instance.GetCommHandle(groupEp, hcclHandle);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    aclnnRet = instance.CheckHcclContext(hcclHandle, mc2ContextTag, engine, ctx, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    if (hcclBuffSize != 0) { // 如果有缓存，直接创建Mc2ContextTensor
        aclnnRet = instance.CreatMc2ContextTensor(ctx, mc2Context);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        OP_LOGD("Found context cache, Get Mc2MoeContext Tensor Success");
        return ACLNN_SUCCESS;
    }   

    aclnnRet = instance.GetCommProtocol(hcclHandle, protocol);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    Mc2MoeContext mc2ContextStruct;
    auto aclnnRet = instance.CreateMc2Context(hcclHandle, mc2ContextTag, engine, protocol, ctx, &mc2ContextStruct,
                                              hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    
    aclnnRet = instance.CreatMc2ContextTensor(ctx, mc2Context);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    OP_LOGD("Create Mc2MoeContext Tensor Success");
    return ACLNN_SUCCESS;
}

// 模板函数显式实例化
template void *Mc2Context::GetHcclLibFunc<void *>(void *handle, const std::string &funcName);

} // namespace Mc2Aclnn
#endif // #if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION