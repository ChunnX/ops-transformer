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
 * \file mc2_context_graph.h
 * \brief MC2 Context management for graph-level fusion pass (output GNode instead of aclTensor)
 */
#ifndef MC2_CONTEXT_GRAPH_H
#define MC2_CONTEXT_GRAPH_H

#define HCCL_CHANNEL_SUPPORT_VERSION 89999700
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
#define HCOMM_VERSION_NUM (HCCL_CHANNEL_SUPPORT_VERSION)
#endif
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

#include <string>
#include <unordered_map>
#include "graph/ge_error_codes.h"
#include "mc2_moe_context.h"

namespace ops {

class Mc2ContextGraph {
public:
    static ge::graphStatus GetMc2ContextData(const std::string &groupEp, const std::string &opName,
                                             Mc2Aclnn::Mc2MoeContext &context, int64_t &hcclBuffSize);

private:
    Mc2ContextGraph() = default;
    ~Mc2ContextGraph();
    Mc2ContextGraph(const Mc2ContextGraph &) = delete;
    Mc2ContextGraph &operator=(const Mc2ContextGraph &) = delete;
    static Mc2ContextGraph &GetInstance();

    ge::graphStatus LoadHcclSymbols();
    ge::graphStatus GetCommHandle(const char *groupEp, HcclComm &hcclHandle);
    ge::graphStatus GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol);
    ge::graphStatus CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList, uint32_t &layerNum);
    ge::graphStatus CheckLinks(uint32_t &netLinkNum, CommLink *linksList);
    ge::graphStatus GetHcclCommLink(const HcclComm &hcclHandle, uint32_t netLayerId, uint32_t srcRankId,
                                    uint32_t dstRankId, const CommProtocol &protocol, CommLink *&links);
    ge::graphStatus GetRankSizePerServer(const HcclComm &hcclHandle, uint32_t netLayers);
    ge::graphStatus InitHcclChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                    const CommProtocol &protocol, std::vector<HcclChannelDesc> &channelDesc);
    ge::graphStatus GetHcclCommChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                       const CommProtocol &protocol, const CommEngine &engine,
                                       std::vector<ChannelHandle> &channels);
    ge::graphStatus FillContextBuffer(const HcclComm &hcclHandle, const CommEngine &engine,
                                      Mc2Aclnn::Mc2MoeContext *context, uint32_t rankId,
                                      const std::vector<ChannelHandle> &channels);
    ge::graphStatus GetHcclCommResource(const HcclComm &hcclHandle, const CommEngine &engine,
                                        const CommProtocol &protocol, Mc2Aclnn::Mc2MoeContext *context);
    ge::graphStatus CreatMc2Context(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                    const CommEngine &engine, const CommProtocol &protocol,
                                    Mc2Aclnn::Mc2MoeContext *context, void *&ctx, uint64_t &hcclBuffSize);
    ge::graphStatus ValidateContextTag(const std::string &mc2ContextTag);
    const std::string GetLibPath();
    template <typename T>
    T GetHcclLibFunc(void *handle, const std::string &funcName);

    void *hcclLibHandle_ = nullptr;
    uint64_t hcclBuffSize_ = 0;
    uint32_t epRankSize_ = 0;
    uint32_t rankSizePerServer_ = 0;
    std::unordered_map<uint32_t, uint32_t> layerMap_;

    HcclResult (*HcomGetCommHandleByGroup)(const char *, HcclComm *) = nullptr;
    HcclResult (*HcclRankGraphGetLinks)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink **, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetLayers)(HcclComm, uint32_t **, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetRankSizeByLayer)(HcclComm, uint32_t, uint32_t *) = nullptr;
    HcclResult (*HcclChannelAcquire)(HcclComm, CommEngine, HcclChannelDesc *, uint32_t, ChannelHandle *) = nullptr;
    HcclResult (*HcclGetHcclBuffer)(HcclComm, void **, uint64_t *) = nullptr;
    HcclResult (*HcclChannelGetHcclBuffer)(HcclComm, ChannelHandle, void **, uint64_t *) = nullptr;
    HcclResult (*HcclEngineCtxCreate)(HcclComm, const char *, CommEngine, uint64_t, void **) = nullptr;
    HcclResult (*HcclEngineCtxCopy)(HcclComm, CommEngine, const char *, void *, uint64_t, uint64_t) = nullptr;
    HcclResult (*HcclGetRankId)(HcclComm, uint32_t *) = nullptr;
    HcclResult (*HcclGetRankSize)(HcclComm, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetRanksByLayer)(HcclComm, uint32_t, uint32_t **, uint32_t *) = nullptr;

    struct ContextCacheEntry {
        Mc2Aclnn::Mc2MoeContext context;
        int64_t hcclBuffSize = 0;
    };
    std::unordered_map<std::string, ContextCacheEntry> contextCache_;
};

}  // namespace ops

#endif  // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#endif  // MC2_CONTEXT_GRAPH_H
