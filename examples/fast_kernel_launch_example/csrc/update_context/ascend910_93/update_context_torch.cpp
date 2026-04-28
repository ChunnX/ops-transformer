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
* \file update_context_torch.cpp
* \brief Update context using channel-based address acquisition
*/

#include "update_context_torch.h"

namespace ascend_ops {
namespace Update_Context {

constexpr uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t OP_TYPE_ALL_TO_ALLV = 8;
constexpr uint32_t EP_OP_TYPE = OP_TYPE_ALL_TO_ALLV;
const std::string EP_ALG_CONFIG = "AlltoAll=level0:fullmesh;level1:pairwise";
static bool isInit = false;
static std::string last_group_ep_str = "";
static at::Tensor output;
int64_t bufferSize = 0;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("updateContext(Tensor x, str group_ep, int ep_world_size) -> (Tensor, int)");
}

static int32_t GetHcclCommLink(const HcclComm& hcclHandle, const uint32_t netLayers,
                                const uint32_t srcRankId, const uint32_t dstRankId,
                                CommLink*& links, CommProtocol& protocol)
{
    CommLink* linksList = nullptr;
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, netLayers, srcRankId, dstRankId,
                                         &linksList, &netLinkNum);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL Communication link failed, ret=", hcclRet);
    TORCH_CHECK(netLinkNum > 0, "The Net Link Is nullptr.");

    links = &linksList[0];
    protocol = linksList[0].linkAttr.linkProtocol;
    return 0;
}

static int32_t GetHcclCommChannel(const HcclComm& hcclHandle, const uint32_t rankDim,
                                   const uint32_t srcRankId, const CommEngine& engine,
                                   std::vector<ChannelHandle>& channels)
{
    uint32_t channelNum = rankDim;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);

    HcclResult hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Hccl Channel Init failed, ret=", hcclRet);

    uint32_t* netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL Rank Graph Layers failed, ret=", hcclRet);
    TORCH_CHECK(netLayerNum > 0, "Net Layer Num is 0.");

    constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
    uint32_t layerId = 0;

    for (uint32_t i = 0; i < rankDim; ++i) {
        if (i == srcRankId) {
            continue;
        }
        uint32_t dstRankId = i;
        uint32_t channelId = i;
        layerId = netLayerList[HCCL_COMM_LAYERS_UB_MEM];
        CommLink* links = nullptr;
        CommProtocol protocol;
        int32_t ret = GetHcclCommLink(hcclHandle, layerId, srcRankId, dstRankId, links, protocol);
        TORCH_CHECK(ret == 0, "GetHcclCommLink failed, ret=", ret);

        channelDesc[channelId].channelProtocol = protocol;
        channelDesc[channelId].remoteRank = dstRankId;
        channelDesc[channelId].notifyNum = channelNum;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;

        if ((dstRankId ^ 1) == srcRankId) {
            channelDesc[srcRankId].channelProtocol = protocol;
            channelDesc[srcRankId].remoteRank = dstRankId;
            channelDesc[srcRankId].notifyNum = channelNum;
            channelDesc[srcRankId].localEndpoint = links->srcEndpointDesc;
            channelDesc[srcRankId].remoteEndpoint = links->dstEndpointDesc;
        }
    }

    hcclRet = HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), channelNum, channels.data());
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Acquire HCCL channel failed, ret=", hcclRet);
    return 0;
}

static int32_t GetHcclCommResource(const HcclComm& hcclHandle, const uint32_t rankId,
                                   const uint32_t epRankSize, const CommEngine& engine,
                                   Mc2ContextStru* mc2Context, int64_t& cclBufferSize)
{
    std::vector<ChannelHandle> channels;
    int32_t ret = GetHcclCommChannel(hcclHandle, epRankSize, rankId, engine, channels);
    TORCH_CHECK(ret == 0, "GetHcclCommChannel failed, ret=", ret);

    for (uint32_t i = 0; i < epRankSize; ++i) {
        void* tempBuffer = nullptr;
        uint64_t bufSize = 0;
        HcclResult hcclRet;

        if (i == rankId) {
            hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &bufSize);
            cclBufferSize = bufSize;
        } else {
            uint32_t channelIndex = i;
            hcclRet = HcclChannelGetHcclBuffer(hcclHandle, channels[channelIndex], &tempBuffer, &bufSize);
        }
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get Hccl Buffer failed, src=", rankId, " dst=", i);
        mc2Context[0].epHcclBuffer[i] = reinterpret_cast<uint64_t>(tempBuffer);
    }

    void* tempBuffer = nullptr;
    uint64_t bufSize = 0;
    HcclResult hcclRet;
    hcclRet = HcclChannelGetHcclBuffer(hcclHandle, channels[rankId], &tempBuffer, &bufSize);
    mc2Context[0].HccsAddr = reinterpret_cast<uint64_t>(tempBuffer);
    return 0;
}

static int32_t CreatMc2Context(HcclComm& comm, int64_t worldSize, int64_t& cclBufferSize,
                                Mc2ContextStru* mc2Context)
{
    uint32_t rankId;
    HcclResult ret = HcclGetRankId(comm, &rankId);
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret=", ret);
    mc2Context[0].epRankId = rankId;

    CommEngine engine = static_cast<CommEngine>(COMM_ENGINE_AIV);

    int32_t ctxRet = GetHcclCommResource(comm, rankId, static_cast<uint32_t>(worldSize),
                                          engine, mc2Context, cclBufferSize);
    TORCH_CHECK(ctxRet == 0, "GetHcclCommResource failed, ret=", ctxRet);
    return 0;
}

static int32_t CreateHcclContext(HcclComm& commHandle, int64_t worldSize, const char* groupName)
{
    HcclResult ret = HcclCommGetHandleWithName(groupName, &commHandle);
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclCommGetHandleWithName failed, ret=", ret);

    uint32_t rankId = 0;
    uint32_t worldSizeHccl = 0;
    ret = HcclGetRankSize(commHandle, &worldSizeHccl);
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankSize failed, ret=", ret);

    ret = HcclGetRankId(commHandle, &rankId);
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret=", ret);

    TORCH_CHECK(rankId < worldSizeHccl && worldSize == worldSizeHccl,
                "rank validation failed: rankId=", rankId, " worldSizeHccl=", worldSizeHccl,
                " worldSize=", worldSize);
    return 0;
}

static int32_t GetMc2Context(Mc2ContextStru* mc2ContextHost, int64_t epWorldSize,
                             int64_t& cclBufferSize, const char* groupEpStr)
{
    HcclComm epCommHandle;
    int32_t contextRet = CreateHcclContext(epCommHandle, epWorldSize, groupEpStr);
    TORCH_CHECK(contextRet == 0, "CreateHcclContext failed, ret=", contextRet);

    contextRet = CreatMc2Context(epCommHandle, epWorldSize, cclBufferSize, mc2ContextHost);
    TORCH_CHECK(contextRet == 0, "CreatMc2Context failed, ret=", contextRet);
    return 0;
}

std::tuple<at::Tensor, int64_t> update_context(const at::Tensor &x, c10::string_view group_ep,
                                                int64_t ep_world_size)
{
    std::string new_group_ep_str = std::string(group_ep);
    bool isSameGroup = (new_group_ep_str == last_group_ep_str);
    if (isInit && isSameGroup) {
        return std::make_tuple(output, bufferSize);
    }

    Mc2ContextStru mc2ContextHost[2];
    int32_t ret = GetMc2Context(mc2ContextHost, ep_world_size, bufferSize, new_group_ep_str.c_str());
    TORCH_CHECK(ret == 0, "GetMc2Context failed, ret=", ret);

    at::Tensor hostContext = at::from_blob(&mc2ContextHost, {sizeof(Mc2ContextStru) * 2 / sizeof(int32_t)}, at::kInt);
    output = at::empty({sizeof(Mc2ContextStru) * 2 / sizeof(int32_t)},
                       at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1)
                       .memory_format(c10::MemoryFormat::Contiguous));
    output.copy_(hostContext);
    last_group_ep_str = new_group_ep_str;
    isInit = true;
    return std::make_tuple(output, bufferSize);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("updateContext", update_context);
}

}
}