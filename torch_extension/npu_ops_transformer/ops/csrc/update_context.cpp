/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_context.cpp
 * \brief Unified update_context implementation supporting both KFC and HCCL Channel modes.
 *
 * Architecture:
 *   CommContext               - 统一上下文结构体，两种模式共用
 *   CopyContextToTensor()     - 提取方法，将CommContext拷贝到tensor (重复 → 提取方法)
 *   KfcContextBuilder         - KFC模式单一职责类 (过大类/发散式变化 → 提取类/按职责拆分)
 *   HcclChannelContextBuilder - HCCL Channel模式单一职责类 (过大类/发散式变化 → 提取类/按职责拆分)
 *   CommContextManager        - pybind11类接口，封装参数和输出，消除Python侧魔鬼数字和IntWrapper
 */

#include <torch/extension.h>
#include "hccl_common.h"

namespace op_api {

// ======================== Common constants ========================

constexpr static uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t OP_TYPE_ALL_TO_ALLV = 8;
constexpr uint32_t EP_OP_TYPE = OP_TYPE_ALL_TO_ALLV;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
const std::string EP_ALG_CONFIG = "AlltoAll=level0:fullmesh;level1:pairwise";

constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint64_t KopyDefaultCtxOffset = 0;

// ======================== Common Types and Utilities ========================

// 统一上下文结构体：两种模式共用，hcclContextAddr在HCCL Channel模式下为0
struct CommContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0; // host kfc方案中，需要传递通信API所需的地址
    uint64_t hcclContextAddr = 0;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE] = {};
};

// 提取方法：将CommContext拷贝到输出的tensor中 (消除两模式的重复代码)
static void CopyContextToTensor(const CommContext &context, at::Tensor &tensor)
{
    at::Tensor hostContext = at::from_blob(const_cast<CommContext *>(&context),
                                           {sizeof(CommContext) / sizeof(int32_t)}, at::kInt);
    tensor.copy_(hostContext);
}

// ======================== KFC Mode ========================

// KFC模式单一职责类：封装所有KFC相关逻辑 (过大类 → 提取类，特性依恋 → 将方法移到合适的类)
class KfcContextBuilder {
public:
    bool Build(const std::string &groupEp, int64_t epWorldSize,
               int64_t &cclBufferSize, at::Tensor &contextTensor)
    {
        CommContext mc2ContextHost;
        int32_t ret = GetMc2Context(mc2ContextHost, epWorldSize, cclBufferSize, groupEp.c_str());
        TORCH_CHECK(ret == 0, "GetMc2Context failed, ret:", ret);

        CopyContextToTensor(mc2ContextHost, contextTensor);
        return true;
    }

private:
    int32_t CollectRankBuffers(HcclComm &comm, int64_t worldSize, int64_t &cclBufferSize, CommContext &mc2Context)
    {
        uint32_t ctxIndex = 0;
        uint32_t rankId;
        (void)HcclGetRankIdFunc(comm, &rankId);
        mc2Context.epRankId = rankId;
        for (uint64_t remoteRankId = 0; remoteRankId < worldSize; remoteRankId++) {
            void *remoteAddr = nullptr;
            uint64_t commSize = 0;
            HcclResult ret;
            if (rankId == remoteRankId) {
                ret = static_cast<HcclResult>(HcclGetHcclBufferFunc(comm, &remoteAddr, &commSize));
                cclBufferSize = commSize;
            } else {
                ret = static_cast<HcclResult>(HcclGetRemoteIpcHcclBufFunc(comm, remoteRankId, &remoteAddr, &commSize));
            }
            TORCH_CHECK((ret == HCCL_SUCCESS), "Get HcclBufferSize failed, ret=", ret);
            mc2Context.epHcclBuffer_[remoteRankId] = (uint64_t)remoteAddr;
        }

        return 0;
    }

    int32_t CreateHcclContext(HcclComm &commHandle, void *opArgs, int64_t worldSize, const char *groupName,
        std::string algConfig, uint32_t opType, CommContext &mc2Context)
    {
        HcclResult ret = static_cast<HcclResult>(HcclKfcOpArgsSetAlgConfigFunc(opArgs, const_cast<char *>(algConfig.c_str())));  // 设置通信类型
        TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetAlgConfig failed, ret:", ret);
        ret = static_cast<HcclResult>(HcclCommGetHandleWithNameFunc(groupName, &commHandle));  // 通过groupName获取groupHandle
        TORCH_CHECK(ret == 0, "HcclGetCommHandle failed, ret:", ret);
        void *opsResCtx;
        ret = static_cast<HcclResult>(HcclCreateOpResCtxFunc(commHandle, opType, opArgs, &opsResCtx)); // 创建HcclContext
        TORCH_CHECK(ret == 0, "HcclCreateOpResCtx failed, ret:", ret);
        mc2Context.hcclContextAddr = (uint64_t)opsResCtx;

        // Get Comm world Size and rank
        uint32_t rankId = 0;
        uint32_t worldSizeHccl = 0;
        ret = static_cast<HcclResult>(HcclGetRankSizeFunc(commHandle, &worldSizeHccl)); // 获取通信域大小
        TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankSize failed, ret:", ret);
        ret = static_cast<HcclResult>(HcclGetRankIdFunc(commHandle, &rankId)); // 获取卡号
        TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret:", ret);
        if ((rankId >= worldSizeHccl) || (worldSize != worldSizeHccl)) {
            TORCH_CHECK(rankId < worldSizeHccl, "rankId:", rankId, " worldSizeHccl:", worldSizeHccl, " worldSize:", worldSize);
            return -1;
        }
        return 0;
    }

    int32_t GetMc2Context(CommContext &mc2ContextHost, int64_t epWorldSize, int64_t &cclBufferSize, const char *groupEpStr)
    {
        InitHcclFunctions();
        void *opArgs = nullptr;
        HcclResult ret = static_cast<HcclResult>(HcclKfcAllocOpArgsFunc(&opArgs));  // 通信配置对象创建
        TORCH_CHECK(ret == 0, "HcclKfcAllocOpArgs failed, ret:", ret);
        uint8_t commEngine = COMM_ENGINE_AIV;
        ret = static_cast<HcclResult>(HcclKfcOpArgsSetCommEngineFunc(opArgs, (uint8_t)commEngine)); // 设置通信方式
        TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetCommEngine failed, ret:", ret);
        HcclComm epCommHandle;
        int32_t contextRet = CreateHcclContext(epCommHandle, opArgs, epWorldSize, groupEpStr, EP_ALG_CONFIG, EP_OP_TYPE, mc2ContextHost);
        TORCH_CHECK(contextRet == 0, "CreateHcclContext failed, ret:", contextRet);
        ret = static_cast<HcclResult>(HcclKfcFreeOpArgsFunc(opArgs)); // 释放通信配置对象
        TORCH_CHECK(ret == 0, "getHcclKfcFreeOpArgs failed, ret:", ret);
        contextRet = CollectRankBuffers(epCommHandle, epWorldSize, cclBufferSize, mc2ContextHost);
        TORCH_CHECK(contextRet == 0, "CollectRankBuffers failed, ret:", contextRet);

        return 0;
    }
};

// ======================== HCCL Channel Mode ========================

// HCCL Channel模式单一职责类：封装所有HCCL Channel相关逻辑
// (过大类 → 提取类，发散式变化 → 按职责拆分，全局可变状态 → 封装layerMap_)
class HcclChannelContextBuilder {
public:
    bool Build(const std::string &groupEp, int64_t &cclBufferSize, at::Tensor &contextTensor)
    {
        ASCEND_LOGI("Start to get CommContext Tensor, groupEp: %s", groupEp.c_str());
        InitHcclEngineCtxFunctions();

        HcclComm hcclHandle;
        if (!AcquireHcclHandle(groupEp, hcclHandle)) {
            return false;
        }

        CommProtocol protocol;
        if (!NegotiateProtocol(hcclHandle, protocol)) {
            return false;
        }

        CommContext commContextStruct;
        if (!BuildContext(hcclHandle, groupEp, protocol, commContextStruct, cclBufferSize)) {
            return false;
        }

        // 将context拷贝到输出的tensor中
        CopyContextToTensor(commContextStruct, contextTensor);
        ASCEND_LOGI("Get CommContext Tensor Success, groupEp: %s, ccl_buffer_size: %d",
                    groupEp.c_str(), cclBufferSize);
        return true;
    }

private:
    // 过长方法 → 提取方法：获取HCCL通信句柄
    bool AcquireHcclHandle(const std::string &groupEp, HcclComm &hcclHandle)
    {
        auto aclnnRet = HcomGetCommHandleByGroupFunc(groupEp.c_str(), &hcclHandle);
        if (aclnnRet != HCCL_SUCCESS) {
            ASCEND_LOGE("Get HCCL handle failed, groupEp: %s", groupEp.c_str());
            return false;
        }
        ASCEND_LOGI("Get HCCL communication handle success hcclHandle is: %p", hcclHandle);
        return true;
    }

    // 过长方法 → 提取方法：协商通信协议
    bool NegotiateProtocol(const HcclComm &hcclHandle, CommProtocol &protocol)
    {
        auto ret = GetCommProtocol(hcclHandle, protocol);
        if (ret != 0) {
            ASCEND_LOGE("GetCommProtocol failed");
            return false;
        }
        return true;
    }

    // 过长方法 → 提取方法：构建上下文
    bool BuildContext(const HcclComm &hcclHandle, const std::string &groupEp,
                      const CommProtocol &protocol, CommContext &commContextStruct,
                      int64_t &cclBufferSize)
    {
        std::string mc2ContextTag = std::string(groupEp) + "moe_dispatch_ffn_combine";
        if (mc2ContextTag.size() > 255) {
            ASCEND_LOGE("Mc2ContextTag is too long, max size is 255");
            return false;
        }

        CommEngine engine = CommEngine::COMM_ENGINE_AIV;
        void *ctx = nullptr;
        uint64_t hcclBuffSize = 0;

        auto ret = GetOrCreateContext(hcclHandle, mc2ContextTag, engine, protocol,
                                      ctx, hcclBuffSize, commContextStruct);
        if (ret != 0) {
            ASCEND_LOGE("GetOrCreateContext failed");
            return false;
        }

        cclBufferSize = hcclBuffSize;
        return true;
    }

    // ---- Protocol negotiation helpers ----

    int32_t GetCommProtocol(const HcclComm &commHandle, CommProtocol &protocol)
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

    int32_t CheckProtocolSupport(const HcclComm &commHandle, uint32_t *&layerList, uint32_t &layerNum)
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
                    layerMap_.find(rankIdLists[rankId]) != layerMap_.end()) { // 本卡或者已经校验过的卡跳过
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
                layerMap_[rankIdLists[rankId]] = layerList[layerIndex];
            }
        }
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

    // ---- Channel management helpers ----

    int32_t InitHcclChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
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
                    layerMap_[dstRank]; // 如果只有一层通信，直接使用该层；如果有多层通信，使用之前记录的通信层
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

    int32_t GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
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

    // ---- Resource management helpers ----

    int32_t GetHcclCommResource(const HcclComm &commHandle, const CommEngine &engine, const CommProtocol &protocol,
                                CommContext *commContextStruct, uint32_t epRankSize, uint64_t &hcclBuffSize)
    {
        ASCEND_LOGI("Start to get HCCL communication resource");
        uint32_t rankId = commContextStruct->epRankId;
        std::vector<ChannelHandle> channels;

        uint32_t rankSizePerServer = 0;
        auto ret = GetHcclCommChannel(commHandle, epRankSize, rankId, protocol, engine, channels, rankSizePerServer);
        if (ret != 0) {
            return ret;
        }
        commContextStruct->rankSizePerServer = rankSizePerServer;
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

            commContextStruct->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer);
        }
        ASCEND_LOGI("Get HCCL CommResource success");
        return 0;
    }

    // ---- Context lifecycle helpers ----

    int32_t CreateContext(const HcclComm &commHandle, const std::string &mc2ContextTag,
                          const CommEngine &engine, const CommProtocol &protocol, void *&ctx,
                          CommContext *commContextStruct, uint64_t &hcclBuffSize)
    {
        ASCEND_LOGI("Start to create HCCL context");
        uint64_t commContextSize = sizeof(CommContext);
        auto hcclRet = HcclEngineCtxCreateFunc(commHandle, mc2ContextTag.c_str(), engine, commContextSize, &ctx);
        if (hcclRet != HCCL_SUCCESS) {
            ASCEND_LOGE("Create HCCL context memory failed");
            return -1;
        }
        ASCEND_LOGI("Create HCCL context success, ctx: %p", ctx);

        hcclRet = HcclGetRankIdFunc(commHandle, &commContextStruct->epRankId);
        if (hcclRet != HCCL_SUCCESS) {
            ASCEND_LOGE("Get rank ID failed");
            return -1;
        }
        ASCEND_LOGI("Get rank ID success, rankId is: %u", commContextStruct->epRankId);

        uint32_t epRankSize = 0;
        hcclRet = HcclGetRankSizeFunc(commHandle, &epRankSize);
        if (hcclRet != HCCL_SUCCESS) {
            ASCEND_LOGE("Get rank size failed");
            return -1;
        }
        ASCEND_LOGI("Get rank size success, rankSize is: %u", epRankSize);

        auto ret = GetHcclCommResource(commHandle, engine, protocol, commContextStruct, epRankSize, hcclBuffSize);
        if (ret != 0) {
            ASCEND_LOGE("Get HCCL communication resource failed");
            return ret;
        }

        hcclRet = HcclEngineCtxCopyFunc(commHandle, engine, mc2ContextTag.c_str(), commContextStruct, commContextSize,
                                         KopyDefaultCtxOffset);
        if (hcclRet != HCCL_SUCCESS) {
            ASCEND_LOGE("Copy context from host to device failed");
            return -1;
        }
        ASCEND_LOGI("Copy context from host to device success");
        return 0;
    }

    int32_t GetOrCreateContext(const HcclComm &commHandle, const std::string &mc2ContextTag,
                               const CommEngine &engine, const CommProtocol &protocol, void *&ctx,
                               uint64_t &hcclBuffSize, CommContext &commContextStruct)
    {
        uint64_t ctxSize = 0;
        auto hcclRet = HcclEngineCtxGetFunc(commHandle, mc2ContextTag.c_str(), engine, &ctx, &ctxSize);
        if (hcclRet != HCCL_SUCCESS) {
            auto aclnnRet = CreateContext(commHandle, mc2ContextTag, engine, protocol, ctx,
                                          &commContextStruct, hcclBuffSize);
            if (aclnnRet != 0) {
                return aclnnRet;
            }
        } else {
            auto aclnnRet = GetHcclBufferSize(commHandle, hcclBuffSize);
            if (aclnnRet != 0) {
                return aclnnRet;
            }
        }
        return 0;
    }

    // ---- Static HCCL query helpers ----

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

    // 记录本卡与其他卡的通信层数，key为其他卡的rankId，value为通信层数 (全局可变状态 → 封装为类成员)
    std::unordered_map<uint32_t, uint32_t> layerMap_;
};

// ======================== CommContextManager (pybind11类接口) ========================

// 基本类型偏执 → 引入类接口：封装参数和输出，消除Python侧魔鬼数字和IntWrapper
// CommContextManager通过pybind11暴露为Python类，提供：
//   - context_tensor_size() 静态方法：从C++结构体布局推导tensor大小，消除魔鬼数字
//   - update(context_tensor) 方法：执行上下文更新
//   - ccl_buffer_size 属性：直接返回int64_t，消除IntWrapper
class CommContextManager {
public:
    CommContextManager(const std::string &groupEp, int64_t epWorldSize,
                       const std::string &commMode = "kfc")
        : groupEp_(groupEp), epWorldSize_(epWorldSize), commMode_(commMode),
          cclBufferSize_(0) {}

    bool update(at::Tensor &contextTensor)
    {
        if (commMode_ == "hccl_channel") {
            HcclChannelContextBuilder builder;
            if (!builder.Build(groupEp_, cclBufferSize_, contextTensor)) {
                return false;
            }
        } else {
            KfcContextBuilder builder;
            if (!builder.Build(groupEp_, epWorldSize_, cclBufferSize_, contextTensor)) {
                return false;
            }
        }
        return true;
    }

    int64_t ccl_buffer_size() const { return cclBufferSize_; }
    int64_t ep_world_size() const { return epWorldSize_; }
    std::string group_name() const { return groupEp_; }
    std::string comm_mode() const { return commMode_; }

    // 从C++结构体布局推导tensor大小，消除Python侧魔鬼数字 (3+1024) 和魔鬼公式 ((x*8)+3)//4
    static int64_t context_tensor_size()
    {
        return (sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
    }

private:
    std::string groupEp_;
    int64_t epWorldSize_;
    std::string commMode_;
    int64_t cclBufferSize_ = 0;
};

// ======================== Unified Entry Point ========================

// Bind the CommContextManager class to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<CommContextManager>(m, "CommContextManager")
        .def(py::init<const std::string &, int64_t, const std::string &>(),
             py::arg("groupEp"), py::arg("epWorldSize"), py::arg("commMode") = std::string("kfc"))
        .def("update", &CommContextManager::update, py::arg("contextTensor").noconvert())
        .def_property_readonly("ccl_buffer_size", &CommContextManager::ccl_buffer_size)
        .def_property_readonly("ep_world_size", &CommContextManager::ep_world_size)
        .def_property_readonly("group_name", &CommContextManager::group_name)
        .def_property_readonly("comm_mode", &CommContextManager::comm_mode)
        .def_static("context_tensor_size", &CommContextManager::context_tensor_size);
}
} // op_api
