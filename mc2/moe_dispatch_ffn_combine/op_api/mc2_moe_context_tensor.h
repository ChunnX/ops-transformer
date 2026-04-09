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
 * \file mc2_moe_context_tensor.h
 * \brief
 */

#include "mc2_moe_context.h"
#include "opdev/platform.h"
#include "hccl/hccl_comm.h"
#include "hccl/hcom.h"
#include "hccl/hccl_rank_graph.h"

namespace Mc2MoeDistributeContext {

constexpr uint64_t kDefaultCtxOffset = 0; // 默认从最开始拷贝
constexpr uint64_t MaxContextTagSize = 255; // 最大上下文标签大小
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1; // 当走MTE或者CCU通信时， hccl获取的组网层数应该为1

static aclnnStatus GetCommHandle(const char* groupEp, HcclComm& hcclHandle)
{
    OP_LOGD("Start to get HCCL communication handle");
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle); // 获取HCCL通信句柄
    if(ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Communication handle failed groupEp is:%s", groupEp);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Communication Handle Success");
    return ACLNN_SUCCESS;
}

static aclnnStatus GetHcclCommLink(const HcclComm& hcclHandle, const uint32_t netLayers, const uint32_t srcRankId,
                                   const uint32_t dstRankId, const CommProtocol& protocol, CommLink*& links)
{
    OP_LOGD("Start to get HCCL communication link");
    CommLink* linksList{nullptr};
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, netLayers, srcRankId, dstRankId, &linksList, &netLinkNum);
    if(hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Communication link failed");
        return ACLNN_ERR_INNER;
    }
    if (netLinkNum == 0) {
        OP_LOGE(ACLNN_ERR_INNER, "The Net Link Is nullptr.");
        return ACLNN_ERR_INNER;
    }
    uint32_t linksIndex = 0;
    while(linksIndex < netLinkNum) { // 遍历组网支持的协议
        if (linksList[linksIndex].linkAttr.linkProtocol == protocol) { // 如果与目标协议相同返回对应的link
            links = &linksList[linksIndex];
            break;
        }
        linksIndex++;
    }
    if (linksIndex == netLinkNum) { // 遍历完没有找到匹配的协议
        OP_LOGE(ACLNN_ERR_INNER, "Failed to obtain communication handle: No matching protocol \
                found in the connection configuration");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Communication Link Success");
    return ACLNN_SUCCESS;
}

static aclnnStatus GetHcclCommChannel(const HcclComm hcclHandle, const uint32_t rankDim, const uint32_t srcRankId,
                                     const CommProtocol& protocol, const CommEngine& engine,
                                     std::vector<ChannelHandle>& channeles)
{
    OP_LOGD("Start to get HCCL communication channel");
    uint32_t channelNum = rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc;
    channelDesc.resize(channelNum);
    channeles.resize(channelNum);
    CommLink * links{nullptr};
    HcclResult hcclRet;
    aclnnStatus aclnnRet;
    uint32_t netLayerNum{0};
    uint32_t netLayers{0};
    uint32_t* netLayerList{nullptr};

    hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Rank Graph Layers failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Rank Graph Layers Success, netLayerNum:%d", netLayerNum);
    netLayers = 0; // 组网拓扑层级，从0开始，则最大为num - 1

    hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if(hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Hccl Channel Init failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Hccl Channel Init Success");

    for (uint32_t channelIndex = 0; channelIndex < rankDim; channelIndex++) {
        if (channelIndex == srcRankId) { // 本卡不需要初始化channel,本卡hccl buffer地址调用接口获取
            continue;
        }
        uint32_t dstRankId = channelIndex; // 目标rank id
        uint32_t channelId = channelIndex > srcRankId ? channelIndex - 1 : channelIndex; // 通道id,比本卡Id大的卡Id全部左移一位
        aclnnRet = GetHcclCommLink(hcclHandle, netLayers, srcRankId, dstRankId, protocol, links); // 遍历组网支持的所有通信协议
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        channelDesc[channelId].channelProtocol = protocol; // 通信协议
        channelDesc[channelId].remoteRank = dstRankId; // 目标rank id
        channelDesc[channelId].notifyNum = channelNum; // 通信的notify数量
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    hcclRet = HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), channelNum, channeles.data());
    if(hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Channel Resource failed, engine is:%d", engine);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Channel Resource success");
    return ACLNN_SUCCESS;
}

static aclnnStatus GetHcclCommResource(const HcclComm hcclHandle, const CommEngine& engine,
                                       const CommProtocol& protocol, Mc2MoeContext* mc2_context)
{
    OP_LOGD("Start to get HCCL communication resource");
    HcclResult hcclRet;
    void * tempBuffer = nullptr;
    uint64_t buffersize = 0;
    uint32_t rankId = mc2_context->epRankId;
    uint32_t epRankSize = mc2_context->epRankSize;
    std::vector<ChannelHandle> channels;
    // 获取HCCL通信通道，后续获取远端通信地址时需要使用
    auto aclnnRet = GetHcclCommChannel(hcclHandle, epRankSize, rankId, protocol, engine, channels);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    OP_LOGD("Create HCCL Communication Channel success");
    for (uint32_t rankIdIndex = 0; rankIdIndex < epRankSize; rankIdIndex++) { // 获取对应的每张卡的通信地址
        if (rankIdIndex == rankId) { // 获取本卡的通信地址
            hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &mc2_context->winSize);
        } else { // 获取其他卡的通信地址
            uint32_t channelIndex = rankIdIndex < rankId ? rankIdIndex : rankIdIndex - 1;
            hcclRet = HcclChannelGetHcclBuffer(hcclHandle, channels[channelIndex], &tempBuffer, &buffersize);
        }
        if(hcclRet != HCCL_SUCCESS) { 
            OP_LOGE(ACLNN_ERR_INNER, "Get Hccl Communicate Buffer Failed,\
                    srcRankId:%d, dstRankId:%d", rankId, rankIdIndex);
            return ACLNN_ERR_INNER;
        }
        mc2_context->epHcclBuffer_[rankIdIndex] =  reinterpret_cast<uint64_t>(tempBuffer);
    }
    OP_LOGD("Get Hccl Communicate Buffer Success");
    return ACLNN_SUCCESS;
}

static aclnnStatus CreatMc2Context(const HcclComm hcclHandle, const std::string& mc2ContextTag,
                                   const CommEngine& engine, const CommProtocol& protocol,
                                   void* &ctx, Mc2MoeContext*  mc2_context)
{
    OP_LOGD("Start to create MC2 Context");
    HcclResult hcclRet;
    aclnnStatus aclnnRet;
    uint64_t dstCtxOffset = kDefaultCtxOffset; // 拷贝地址的偏移量，即将数据从host侧拷贝到device侧时从哪个偏移地址开始拷贝，默认从0开始，全部拷贝
    uint64_t ctxSize = sizeof(Mc2MoeContext); // 需要分配的context结构体大小

    hcclRet = HcclEngineCtxCreate(hcclHandle, mc2ContextTag.c_str(), engine, ctxSize, &ctx);
    if(hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Context Memory failed, mc2ContextTag is:%s, engine is:%d",\
                mc2ContextTag.c_str(), engine);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Context Memory success");
    hcclRet = HcclGetRankId(hcclHandle, &mc2_context->epRankId); // 获取本卡ID
    if(hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Rank Id failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Rank Id success");
    hcclRet = HcclGetRankSize(hcclHandle, &mc2_context->epRankSize); // 获取通讯域存在多少卡
    if(hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Hccl Get Rank Size failed.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Rank Size success %d", mc2_context->epRankSize);
    aclnnRet = GetHcclCommResource(hcclHandle, engine, protocol, mc2_context); // 获取所需的HCCL通信资源，即通信buffer地址
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    // 把host对应的数据拷贝到device侧
    hcclRet = HcclEngineCtxCopy(hcclHandle, engine, mc2ContextTag.c_str(), mc2_context, ctxSize, dstCtxOffset);
    if(hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Failed to copy Mc2MoeContext from host to device");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("CreatMc2Context Success");
    return ACLNN_SUCCESS;
}

static aclnnStatus CreatMc2ContextTensor(void* ctx, aclTensor* &mc2Context)
{
    OP_LOGD("Start to create Mc2Context Tensor");
    if(ctx == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Create Mc2Context Tensor failed ctx is nullptr.");
        return ACLNN_ERR_INNER;
    }
    uint64_t mc2ContextLength = sizeof(Mc2MoeContext);
    int64_t shap[1] = {mc2ContextLength / sizeof(uint32_t)}; // 默认1维
    int64_t strides[1] = {1};
    mc2Context = aclCreateTensor(
        shap, 1, aclDataType::ACL_INT32, strides, 0, 
        aclFormat::ACL_FORMAT_ND, shap, 1, ctx); // 创建mc2Context Tensor
    if(mc2Context == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Create Mc2Context Tensor failed.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("CreatMc2ContextTensor Success");
    return ACLNN_SUCCESS;
}

static aclnnStatus GetHcclBufferSize(const HcclComm& hcclHandle, uint64_t& hcclBuffSize) 
{
    // 多轮次调用的时候，host侧没有保存mc2contxet结构体的数据，需要重新获取hccl buffer 大小
    void * tempBuffer = nullptr;
    auto hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Buffer Size failed");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}


static aclnnStatus GetCommEngine(const HcclComm& hcclHandle, CommEngine& engine, CommProtocol& protocol)
{
    OP_LOGD("Start to get CommEngine");
    aclnnStatus aclnnRet;
    uint32_t netLayerNum{0};
    uint32_t* netLayerList{nullptr};

    auto hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Rank Graph Layers failed");
        return ACLNN_ERR_INNER;
    }
    engine = CommEngine::COMM_ENGINE_AIV;
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    // if (netLayerNum == HCCL_COMM_LAYERS_MTE_CCU) { // 目前CCU走原有方案,通过别的方式判断是否走CCU,目前只支持MTE
    //     engine = CommEngine::COMM_ENGINE_AIV;
    //     protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    //     OP_LOGD("Get CommEngine Success engine is:%d, protocol is:%d", engine, protocol);
    //     return ACLNN_SUCCESS;
    // }
    // OP_LOGE(ACLNN_ERR_INNER, "Current version only supports MTE communication mode, other modes are not supported.");
    return ACLNN_SUCCESS;
}

extern inline aclnnStatus GetMc2ContextTensor(const char* groupEp, const char* opName, uint64_t& hcclBuffSize, aclTensor* &mc2Context)
{
    OP_LOGD("Start to get Mc2MoeContext Tensor");
    HcclComm hcclHandle;
    aclnnStatus aclnnRet;
    void * ctx{nullptr};
    uint64_t ctxSize{0};
    CommEngine engine;
    CommProtocol protocol;
    Mc2MoeContext mc2_context;
    std::string mc2ContextTag = std::string(groupEp) + std::string(opName);
    if (mc2ContextTag.size() > MaxContextTagSize) { // 检查上下文标签大小最大允许255个字符串长度
        OP_LOGE(ACLNN_ERR_INNER, "Mc2ContextTag is too long, max size is %d, but current size is %d",\
                MaxContextTagSize, mc2ContextTag.size());
        return ACLNN_ERR_INNER;
    }
    aclnnRet = GetCommHandle(groupEp, hcclHandle); // 根据EP域名称获取通信上下文句柄
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    aclnnRet = GetCommEngine(hcclHandle, engine, protocol); // 获取对应的通信引擎和通信协议
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    auto hcclRet = HcclEngineCtxGet(hcclHandle, mc2ContextTag.c_str(), engine, &ctx, &ctxSize); // 检查是否已经存在context结构体，存在则直接跳过，否则进行创建
    if(hcclRet != HCCL_SUCCESS) {
        aclnnRet = CreatMc2Context(hcclHandle, mc2ContextTag, engine, protocol, ctx, &mc2_context); // 不存在context结构体，进行创建
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        hcclBuffSize = mc2_context.winSize; // 从mc2结构体中获取对应的buffer大小
    } else {
        aclnnRet = GetHcclBufferSize(hcclHandle, hcclBuffSize); // 已经存在context结构体，则hccl buffer的大小单独进行获取
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    }
    aclnnRet = CreatMc2ContextTensor(ctx, mc2Context); // 创建对应的输入aclnn Tensor
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    OP_LOGD("Get Mc2MoeContext Tensor Success");
    return ACLNN_SUCCESS;
}

}