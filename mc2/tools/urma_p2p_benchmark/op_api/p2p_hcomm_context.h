/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file p2p_hcomm_context.h
 * \brief P2P Hcomm模式本地5参数版GetP2pHcommContextTensor
 *
 * 照着bandwidth_test的5参数版GetMc2ContextTensor实现，使用P2pHcommContext
 * (含hcommHandle_)作为通信上下文结构体，支持COMM_PROTOCOL_UBC_CTP协议。
 *
 * 独立namespace P2pHcommAclnn，避免与ops-trans-coke common的Mc2Aclnn::Mc2Context冲突。
 */

#ifndef P2P_HCOMM_CONTEXT_H
#define P2P_HCOMM_CONTEXT_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include "hccl/hccl_rank_graph.h"
#include "aclnn/aclnn_base.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

// P2pHcommContext: host端通信上下文结构体（与kernel端p2p_hcomm_comm.h中的P2pHcommContext布局一致）
// 布局与bandwidth_test的Mc2Aclnn::Mc2MoeContext完全相同（含hcommHandle_）
namespace P2pHcommAclnn {
constexpr uint32_t P2P_HCCL_MAX_RANK_SIZE = 1024;

struct P2pHcommContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer_[P2P_HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[P2P_HCCL_MAX_RANK_SIZE];
};

class P2pHcommContextHelper {
public:
    // 5参数版GetP2pHcommContextTensor（照着bandwidth_test的5参数版GetMc2ContextTensor实现）
    // 使用COMM_PROTOCOL_UBC_CTP协议创建包含hcommHandle_的完整通信上下文
    static aclnnStatus GetP2pHcommContextTensor(const char *groupEp, const char *opName, uint64_t &hcclBuffSize,
                                                 aclTensor *&mc2Context, CommProtocol protocol);

private:
    explicit P2pHcommContextHelper();
    ~P2pHcommContextHelper();

    // 以下方法照着bandwidth_test的Mc2Context类实现，逻辑完全一致
    aclnnStatus LoadHcclSymbols();
    aclnnStatus GetCommHandle(const char *groupEp, HcclComm &hcclHandle);
    aclnnStatus GetHcclCommLink(const HcclComm &hcclHandle, uint32_t netLayers, uint32_t srcRankId,
                                uint32_t dstRankId, const CommProtocol &protocol, CommLink *&links);
    aclnnStatus GetNetLayers(const HcclComm &hcclHandle, uint32_t *&netLayerList, uint32_t &netLayerNum);
    aclnnStatus GetRankSizePerServer(const HcclComm &hcclHandle, uint32_t netLayers);
    aclnnStatus InitHcclChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                const CommProtocol &protocol, std::vector<HcclChannelDesc> &channelDesc);
    // 7参数版GetHcclCommChannel（照着bandwidth_test，传入P2pHcommContext*以填充hcommHandle_）
    aclnnStatus GetHcclCommChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                   P2pHcommContext *p2pContextStruct, const CommProtocol &protocol,
                                   const CommEngine &engine, std::vector<ChannelHandle> &channels);
    aclnnStatus GetHcclCommResource(const HcclComm &hcclHandle, const CommEngine &engine,
                                     const CommProtocol &protocol, P2pHcommContext *p2pContextStruct);
    aclnnStatus CreatP2pHcommContext(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                      const CommEngine &engine, const CommProtocol &protocol,
                                      P2pHcommContext *p2pContextStruct, void *&ctx, uint64_t &hcclBuffSize);
    aclnnStatus CreatP2pHcommContextTensor(void *ctx, aclTensor *&mc2Context);
    aclnnStatus GetHcclBufferSize(const HcclComm &hcclHandle, uint64_t &hcclBuffSize);
    // 4参数版CheckProtocolSupport（照着bandwidth_test，传入CommProtocol）
    aclnnStatus CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList, uint32_t &layerNum,
                                      const CommProtocol &protocol);
    aclnnStatus GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol);
    aclnnStatus ValidateContextTag(const std::string &mc2ContextTag);
    // 3参数版CheckLinks（照着bandwidth_test，传入CommProtocol）
    aclnnStatus CheckLinks(uint32_t &netLinkNum, CommLink *linksList, CommProtocol protocol);
    aclnnStatus CheckContextCache(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                  const CommEngine &engine, void *&ctx, uint64_t &hcclBuffSize);
    const std::string GetLibPath();
    template <typename T>
    T GetHcclLibFunc(void *handle, const std::string &funcName);

    void *hcclLibHandle_ = nullptr;
    uint64_t hcclBuffSize_ = 0;
    uint32_t epRankSize_ = 0;
    uint32_t rankSizePerServer_ = 0;
    std::unordered_map<uint32_t, uint32_t> layerMap;

    // HCCL动态加载函数指针（与bandwidth_test的Mc2Context完全一致）
    HcclResult (*HcomGetCommHandleByGroup)(const char *, HcclComm *) = nullptr;
    HcclResult (*HcclRankGraphGetLinks)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink **, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetLayers)(HcclComm, uint32_t **, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetRankSizeByLayer)(HcclComm, uint32_t, uint32_t *) = nullptr;
    HcclResult (*HcclChannelAcquire)(HcclComm, CommEngine, HcclChannelDesc *, uint32_t, ChannelHandle *) = nullptr;
    HcclResult (*HcclGetHcclBuffer)(HcclComm, void **, uint64_t *) = nullptr;
    HcclResult (*HcclChannelGetHcclBuffer)(HcclComm, ChannelHandle, void **, uint64_t *) = nullptr;
    HcclResult (*HcclEngineCtxCreate)(HcclComm, const char *, CommEngine, uint64_t, void **) = nullptr;
    HcclResult (*HcclEngineCtxGet)(HcclComm, const char *, CommEngine, void **, uint64_t *) = nullptr;
    HcclResult (*HcclEngineCtxCopy)(HcclComm, CommEngine, const char *, void *, uint64_t, uint64_t) = nullptr;
    HcclResult (*HcclGetRankId)(HcclComm, uint32_t *) = nullptr;
    HcclResult (*HcclGetRankSize)(HcclComm, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetRanksByLayer)(HcclComm, uint32_t, uint32_t **, uint32_t *) = nullptr;
};

} // namespace P2pHcommAclnn
#endif // P2P_HCOMM_CONTEXT_H
