// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

/*!
 * \file moe_dispatch_ffn_combine_context.cpp
 * \brief MC2 MoE Context implementation using HCCL EngineCtx API
 */

#include <torch/extension.h>
#include "hccl_common.h"

namespace op_api {

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint64_t KopyDefaultCtxOffset = 0;
// 记录本卡与其他卡的通信层数，key为其他卡的rankId，value为通信层数
std::unordered_map<uint32_t, uint32_t> layerMap;

// Mc2MoeContext结构体 (参考mc2_moe_context.h)
struct Mc2MoeContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
};


static int32_t GetHcclBufferSize(const HcclComm &commHandle, uint64_t &hcclBuffSize)
{
    void *tempBuffer = nullptr;
    auto hcclRet = HcclGetHcclBufferFunc(commHandle, &tempBuffer, &hcclBuffSize);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Get HCCL Buffer Size failed");
        return -1;
    }
    return 0;
}

static int32_t GetNetLayers(const HcclComm &commHandle, uint32_t *&netLayerList, uint32_t &netLayerNum)
{
    auto hcclRet = HcclRankGraphGetLayersFunc(commHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Get HCCL layers failed");
        return -1;
    }
    ASCEND_LOGI("Get HCCL layers success, netLayerNum is: %u", netLayerNum);
    return 0;
}

static int32_t GetRankSizePerServer(const HcclComm &commHandle, uint32_t netLayers, uint32_t &rankSizePerServer)
{
    auto hcclRet = HcclRankGraphGetRankSizeByLayerFunc(commHandle, netLayers, &rankSizePerServer);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Get HCCL rank size per server failed");
        return -1;
    }
    ASCEND_LOGI("Get HCCL rank size per server success, rankSizePerServer is: %u", rankSizePerServer);
    return 0;
}

static int32_t GetHcclCommLink(const HcclComm &commHandle, uint32_t netLayerId, uint32_t srcRankId,
                               uint32_t dstRankId, const CommProtocol &protocol, CommLink *&links)
{
    CommLink *linksList = nullptr;
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Get HCCL Communication link failed");
        return -1;
    }
    if (netLinkNum == 0) {
        ASCEND_LOGE("The Net Link Is nullptr. srcRankId is %u, dstRankId is %u, layerId is %u",
            srcRankId, dstRankId, netLayerId);
        return -1;
    }
    ASCEND_LOGI("Get HCCL Rank Links Success Links Num is: %u", netLinkNum);
    uint32_t index = 0;
    for (; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == protocol) {
            links = &linksList[index];
            break;
        }
    }
    if (index == netLinkNum) {
        ASCEND_LOGE("No matching communication protocol found in HCCL links protocol is %d", protocol);
        return -1;
    }
    return 0;
}

static int32_t InitHcclChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                               const CommProtocol &protocol, std::vector<HcclChannelDesc> &channelDesc)
{
    uint32_t channelNum = channelDesc.size();
    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("HCCL channel init failed");
        return -1;
    }
    ASCEND_LOGI("HCCL channel init success");

    uint32_t netLayerNum = 0;
    uint32_t layerId = 0;
    uint32_t *netLayerList = nullptr;
    auto ret = GetNetLayers(commHandle, netLayerList, netLayerNum);
    if (ret != 0 || netLayerNum == 0) {
        ASCEND_LOGE("Get HCCL net layers failed netLayerNum is: %u", netLayerNum);
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
        ret = GetHcclCommLink(commHandle, layerId, srcRankId, dstRank, protocol, links);
        if (ret != 0) {
            return ret;
        }
        channelDesc[channelId].channelProtocol = protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].notifyNum = channelNum;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;   // srcEndpointDesc offset
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;   // dstEndpointDesc offset
    }
    return 0;
}

static int32_t GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                                  const CommProtocol &protocol, const CommEngine &engine,
                                  std::vector<ChannelHandle> &channels, uint32_t &rankSizePerServer)
{
    ASCEND_LOGI("Start to get HCCL communication channel");
    uint32_t channelNum = rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);

    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    auto ret = GetNetLayers(commHandle, netLayerList, netLayerNum);
    if (ret != 0) {
        return ret;
    }

    uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
    
    ret = GetRankSizePerServer(commHandle, netLayers, rankSizePerServer);
    if (ret != 0) {
        return ret;
    }

    ret = InitHcclChannel(commHandle, rankDim, srcRankId, protocol, channelDesc);
    if (ret != 0) {
        return ret;
    }

    auto hcclRet = HcclChannelAcquireFunc(commHandle, engine, channelDesc.data(), channelNum, channels.data());
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Acquire HCCL channel failed");
        return -1;
    }
    return 0;
}

static int32_t GetHcclCommResource(const HcclComm &commHandle, const CommEngine &engine, const CommProtocol &protocol,
                                   Mc2MoeContext *mc2ContextStruct, uint32_t epRankSize, uint64_t &hcclBuffSize)
{
    ASCEND_LOGI("Start to get HCCL communication resource");
    uint32_t rankId = mc2ContextStruct->epRankId;
    std::vector<ChannelHandle> channels;

    uint32_t rankSizePerServer = 0;
    auto ret = GetHcclCommChannel(commHandle, epRankSize, rankId, protocol, engine, channels, rankSizePerServer);
    if (ret != 0) {
        return ret;
    }
    mc2ContextStruct->rankSizePerServer = rankSizePerServer;
    ASCEND_LOGI("Get HCCL communication channel success, channel num is: %zu", channels.size());

    for (uint32_t i = 0; i < epRankSize; ++i) {
        void *tempBuffer = nullptr;
        uint64_t bufSize = 0;
        HcclResult hcclRet;

        if (i == rankId) {
            hcclRet = HcclGetHcclBufferFunc(commHandle, &tempBuffer, &hcclBuffSize);
        } else {
            uint32_t idx = (i < rankId) ? i : (i - 1);
            hcclRet = HcclChannelGetHcclBufferFunc(commHandle, channels[idx], &tempBuffer, &bufSize);
        }

        if (hcclRet != HCCL_SUCCESS) {
            ASCEND_LOGE("Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return -1;
        }

        mc2ContextStruct->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer);
    }
    ASCEND_LOGI("Get HCCL CommResource success");
    return 0;
}

static int32_t CheckLinks(uint32_t &netLinkNum, CommLink *linksList)
{
    bool isFoundUbMemProtocol = false;
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            isFoundUbMemProtocol = true;
            break;
        }
    }
    if (!isFoundUbMemProtocol) {
        return -1;
    }
    return 0;
}

static int32_t CheckProtocolSupport(const HcclComm &commHandle, uint32_t *&layerList, uint32_t &layerNum)
{
    uint32_t srcRankId = 0;
    uint32_t dstRankId = 0;
    uint32_t netLinkNum = 0;
    uint32_t rankNumInLayer = 0;
    uint32_t *rankIdLists = nullptr;
    CommLink *linksList = nullptr;

    auto hcclRet = HcclGetRankIdFunc(commHandle, &srcRankId);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("CheckProtocolSupport Get rank ID failed");
        return -1;
    }
    ASCEND_LOGE("CheckProtocolSupport Get rank ID success, rankId is: %d", srcRankId);

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        ASCEND_LOGI("CheckProtocolSupport Check layer %d", layerList[layerIndex]);
        hcclRet = HcclRankGraphGetRanksByLayerFunc(commHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if (hcclRet != HCCL_SUCCESS) {
            ASCEND_LOGE("Get rank IDs by layer failed");
            return -1;
        }
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if (rankIdLists[rankId] == srcRankId ||
                layerMap.find(rankIdLists[rankId]) != layerMap.end()) { // 本卡或者已经校验过的卡跳过
                continue;
            }
            hcclRet = HcclRankGraphGetLinksFunc(commHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                                &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS) {
                ASCEND_LOGE("Get HCCL links failed when checking protocol support");
                return -1;
            }
            if (netLinkNum == 0) {
                ASCEND_LOGE("No available HCCL links found");
                return -1;
            }
            if (CheckLinks(netLinkNum, linksList) != 0) {
                ASCEND_LOGE("No HCCL links support UB_MEM srcRankID %d, dstRankID %d layer is %d",
                            srcRankId, dstRankId, layerList[layerIndex]);
                return -1;
            }
            layerMap[rankIdLists[rankId]] = layerList[layerIndex];
        }
    }
    return 0;
}

static int32_t GetCommProtocol(const HcclComm &commHandle, CommProtocol &protocol)
{
    ASCEND_LOGI("Start to get HCCL communication protocol");
    uint32_t layerNum = 0;
    uint32_t *layerList = nullptr;
    auto ret = HcclRankGraphGetLayersFunc(commHandle, &layerList, &layerNum);
    if (ret != HCCL_SUCCESS) {
        ASCEND_LOGE("Get HCCL layers failed");
        return -1;
    }

    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return 0;
    }

    ASCEND_LOGI("start CheckProtocolSupport, layerNum is %d", layerNum);
    auto aclnnRet = CheckProtocolSupport(commHandle, layerList, layerNum);
    if (aclnnRet != 0) {
        return aclnnRet;
    }

    ASCEND_LOGI("CheckProtocolSupport success!");
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    return 0;
}

static int32_t CreatMc2Context(const HcclComm &commHandle, const std::string &mc2ContextTag,
                               const CommEngine &engine, const CommProtocol &protocol, void *&ctx,
                               Mc2MoeContext *mc2ContextStruct, uint64_t &hcclBuffSize)
{
    ASCEND_LOGI("Start to create HCCL context");
    uint64_t mc2ContextSize = sizeof(Mc2MoeContext);
    auto hcclRet = HcclEngineCtxCreateFunc(commHandle, mc2ContextTag.c_str(), engine, mc2ContextSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Create HCCL context memory failed");
        return -1;
    }
    ASCEND_LOGI("Create HCCL context success, ctx: %p", ctx);

    hcclRet = HcclGetRankIdFunc(commHandle, &mc2ContextStruct->epRankId);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Get rank ID failed");
        return -1;
    }
    ASCEND_LOGI("Get rank ID success, rankId is: %u", mc2ContextStruct->epRankId);

    uint32_t epRankSize = 0;
    hcclRet = HcclGetRankSizeFunc(commHandle, &epRankSize);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Get rank size failed");
        return -1;
    }
    ASCEND_LOGI("Get rank size success, rankSize is: %u", epRankSize);

    auto ret = GetHcclCommResource(commHandle, engine, protocol, mc2ContextStruct, epRankSize, hcclBuffSize);
    if (ret != 0) {
        ASCEND_LOGE("Get HCCL communication resource failed");
        return ret;
    }

    hcclRet = HcclEngineCtxCopyFunc(commHandle, engine, mc2ContextTag.c_str(), mc2ContextStruct, mc2ContextSize,
                                    KopyDefaultCtxOffset);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Copy context from host to device failed");
        return -1;
    }
    ASCEND_LOGI("Copy context from host to device success");
    return 0;
}

static int32_t GetOrCreateMc2Context(const HcclComm &commHandle, const std::string &mc2ContextTag,
                                     const CommEngine &engine, const CommProtocol &protocol, void *&ctx,
                                     uint64_t &hcclBuffSize, Mc2MoeContext &mc2ContextStruct,
                                     uint32_t &epWorldSize)
{
    uint64_t ctxSize = 0;
    auto hcclRet = HcclEngineCtxGetFunc(commHandle, mc2ContextTag.c_str(), engine, &ctx, &ctxSize);
    if (hcclRet != HCCL_SUCCESS) {
        auto aclnnRet = CreatMc2Context(commHandle, mc2ContextTag, engine, protocol, ctx,
                                        &mc2ContextStruct, hcclBuffSize);
        if (aclnnRet != 0) {
            return aclnnRet;
        }
    } else {
        auto aclnnRet = GetHcclBufferSize(commHandle, hcclBuffSize);
        if (aclnnRet != 0) {
            return aclnnRet;
        }
    }
    hcclRet = HcclGetRankSizeFunc(commHandle, &epWorldSize);
    if (hcclRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Get HCCL rank size failed");
        return -1;
    }
    return 0;
}

bool update_context_by_hccl_channel(std::string group_ep, py::object ep_world_size, py::object ccl_buffer_size,
                                    at::Tensor &context)
{
    ASCEND_LOGI("Start to get Mc2MoeContext Tensor, groupEp: %s", group_ep.c_str());
    InitHcclEngineCtxFunctions();

    uint32_t epWorldSize = 0;
    uint64_t cclBufferSize = 0;

    void *ctx = nullptr;
    CommProtocol protocol;

    std::string mc2ContextTag = std::string(group_ep) + "moe_dispatch_ffn_combine";
    if (mc2ContextTag.size() > 255) {
        ASCEND_LOGE("Mc2ContextTag is too long, max size is 255");
        return false;
    }

    HcclComm hcclHandle;
    auto aclnnRet = HcomGetCommHandleByGroupFunc(group_ep.c_str(), &hcclHandle);
    if (aclnnRet != HCCL_SUCCESS) {
        ASCEND_LOGE("Get HCCL handle failed, groupEp: %s", group_ep.c_str());
        return false;
    }
    ASCEND_LOGI("Get HCCL communication handle success hcclHandle is: %p", hcclHandle);

    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    auto ret = GetCommProtocol(hcclHandle, protocol);
    if (ret != 0) {
        ASCEND_LOGE("GetCommProtocol failed");
        return false;
    }

    Mc2MoeContext mc2ContextStruct;
    ret = GetOrCreateMc2Context(hcclHandle, mc2ContextTag, engine, protocol, ctx, cclBufferSize,
                                mc2ContextStruct, epWorldSize);
    if (ret != 0) {
        ASCEND_LOGE("GetOrCreateMc2Context failed");
        return false;
    }

    ep_world_size.attr("value") = epWorldSize;
    ccl_buffer_size.attr("value") = cclBufferSize;
    
    // 将context拷贝到输出的tensor中
    at::Tensor hostContext = at::from_blob(&mc2ContextStruct, {sizeof(Mc2MoeContext) / sizeof(int32_t)}, at::kInt);
    context.copy_(hostContext);
    ASCEND_LOGI("Get Mc2MoeContext Tensor Success, groupEp: %s, ep_world_size: %d, ccl_buffer_size: %d",
                group_ep.c_str(), ep_world_size.attr("value"), ccl_buffer_size.attr("value"));

    return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("update_context_by_hccl_channel", &update_context_by_hccl_channel,
          "update_context_by_hccl_channel", py::arg("group_ep"), py::arg("ep_world_size"),
          py::arg("ccl_buffer_size"), py::arg("context").noconvert());
}
} // op_api
