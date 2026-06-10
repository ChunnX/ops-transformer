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
 * \file mc2_moe_graph_context.h
 * \brief MC2 MoE Graph Context Manager for graph-level fusion pass
 */
#ifndef OPS_TRANSFORMER_MC2_MOE_GRAPH_CONTEXT_H
#define OPS_TRANSFORMER_MC2_MOE_GRAPH_CONTEXT_H

#include "version/cann_version.h"
#define GRAPH_FUSION_SUPPORT_VERSION 90000000
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION

#define HCCL_CHANNEL_SUPPORT_VERSION 90000000
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
#define HCOMM_VERSION_NUM (HCCL_CHANNEL_SUPPORT_VERSION)
#endif
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

#include "hccl/hccl_rank_graph.h"
#include "graph/ge_error_codes.h"
#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/operator.h"
#include "graph/attr_value.h"
#include "graph/tensor.h"
#include <string>
#include <vector>
#include <numeric>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>

namespace ops {
namespace {
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
}

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
};

struct LinkQueryParams {
    uint32_t netLayerIndex;
    uint32_t srcRankId;
    uint32_t dstRankId;
    CommProtocol protocol;
};

struct ChannelInitParams {
    uint32_t rankDim;
    uint32_t srcRankId;
    CommProtocol protocol;
};

struct ChannelAcquireParams {
    uint32_t rankDim;
    uint32_t srcRankId;
    CommProtocol protocol;
    CommEngine engine;
};

struct ContextCreateParams {
    HcclComm hcclHandle;
    CommEngine engine;
    CommProtocol protocol;
    const std::string& mc2ContextTag;
};

struct ContextAcquireParams {
    HcclComm hcclHandle;
    CommEngine engine;
    CommProtocol protocol;
    const std::string& mc2ContextTag;
};

class Mc2MoeGraphContext {
public:
    static ge::graphStatus GetMc2ContextA5(
        ge::Graph& graph, const std::string& contextTag, const std::string& epName,
        const ge::GNode& originalNode, ge::GNode& constNode, int64_t& hcclBuffSize);

private:
    Mc2MoeGraphContext() = default;
    ~Mc2MoeGraphContext();
    Mc2MoeGraphContext(const Mc2MoeGraphContext&) = delete;
    Mc2MoeGraphContext& operator=(const Mc2MoeGraphContext&) = delete;
    static Mc2MoeGraphContext& GetInstance();
    static constexpr const char* LOG_TAG = "Mc2MoeContext";

    ge::graphStatus LoadHcclSymbols();
    ge::graphStatus LoadSymbol(void** ptr, const char* name);
    ge::graphStatus GetCommHandle(const char* groupEp, HcclComm& hcclHandle);
    ge::graphStatus GetHcclContext(const std::string& contextTag, const std::string& groupEp,
                                   Mc2MoeContext* context, int64_t& hcclBuffSize);
    ge::graphStatus GetHcclCommLink(const HcclComm& hcclHandle, const LinkQueryParams& params, CommLink*& links);
    ge::graphStatus GetRankSizePerServer(const HcclComm& hcclHandle, uint32_t netLayers);
    ge::graphStatus InitHcclChannel(const HcclComm& hcclHandle, const ChannelInitParams& params,
                                    std::vector<HcclChannelDesc>& channelDesc);
    ge::graphStatus SetupChannelDesc(const HcclComm& hcclHandle, const ChannelInitParams& params,
                                     uint32_t* netLayerList, uint32_t netLayerNum,
                                     std::vector<HcclChannelDesc>& channelDesc);
    ge::graphStatus GetHcclCommChannel(const HcclComm& hcclHandle, const ChannelAcquireParams& params,
                                       std::vector<ChannelHandle>& channels);
    ge::graphStatus GetHcclCommResource(const HcclComm& hcclHandle, const CommEngine& engine,
                                         const CommProtocol& protocol, Mc2MoeContext* context);
    ge::graphStatus FillContextBuffer(const HcclComm& hcclHandle, const CommEngine& engine, Mc2MoeContext* context,
                                      uint32_t rankId, const std::vector<ChannelHandle>& channels);
    ge::graphStatus CreatMc2Context(const ContextCreateParams& params, Mc2MoeContext* context);
    ge::graphStatus CheckLinks(uint32_t& netLinkNum, CommLink* linksList);
    ge::graphStatus CheckProtocolSupport(const HcclComm& hcclHandle, uint32_t*& layerList, uint32_t& layerNum);
    ge::graphStatus GetCommProtocol(const HcclComm& hcclHandle, CommProtocol& protocol);
    ge::graphStatus ValidateContextTag(const std::string& mc2ContextTag);
    ge::graphStatus GetMc2Context(const ContextAcquireParams& params, Mc2MoeContext* context, int64_t& hcclBuffSize);
    const std::string GetLibPath();
    template <typename T>
    T GetHcclLibFunc(void* handle, const std::string& funcName);

    ge::graphStatus CreateContextNode(ge::Graph& graph, const Mc2MoeContext& context, ge::GNode& constNode);
    ge::graphStatus CreateNewMc2Context(ge::Graph& graph, const std::string& contextTag,
                                        const std::string& groupEp, const std::string& uniqueKey,
                                        ge::GNode& constNode, int64_t& hcclBuffSize);
    bool TryGetBufferSizeCache(const std::string& key, int64_t& hcclBuffSize);
    void UpdateBufferSizeCache(const std::string& key, int64_t hcclBuffSize);
    void ResetAllFunctionPointers();

    void* hcclLibHandle_{nullptr};
    uint64_t hcclBuffSize_{0};
    uint32_t epRankSize_{0};
    uint32_t rankSizePerServer_{0};

    HcclResult (*HcomGetCommHandleByGroup)(const char*, HcclComm*){nullptr};
    HcclResult (*HcclRankGraphGetLinks)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*){nullptr};
    HcclResult (*HcclRankGraphGetLayers)(HcclComm, uint32_t**, uint32_t*){nullptr};
    HcclResult (*HcclRankGraphGetRankSizeByLayer)(HcclComm, uint32_t, uint32_t*){nullptr};
    HcclResult (*HcclChannelAcquire)(HcclComm, CommEngine, HcclChannelDesc*, uint32_t, ChannelHandle*){nullptr};
    HcclResult (*HcclGetHcclBuffer)(HcclComm, void**, uint64_t*){nullptr};
    HcclResult (*HcclChannelGetHcclBuffer)(HcclComm, ChannelHandle, void**, uint64_t*){nullptr};
    HcclResult (*HcclEngineCtxCreate)(HcclComm, const char*, CommEngine, uint64_t, void**){nullptr};
    HcclResult (*HcclEngineCtxGet)(HcclComm, const char*, CommEngine, void**, uint64_t*){nullptr};
    HcclResult (*HcclEngineCtxCopy)(HcclComm, CommEngine, const char*, void*, uint64_t, uint64_t){nullptr};
    HcclResult (*HcclEngineCtxDestroy)(HcclComm, const char*, CommEngine){nullptr};
    HcclResult (*HcclGetRankId)(HcclComm, uint32_t*){nullptr};
    HcclResult (*HcclGetRankSize)(HcclComm, uint32_t*){nullptr};
    HcclResult (*HcclRankGraphGetRanksByLayer)(HcclComm, uint32_t, uint32_t**, uint32_t*){nullptr};

    std::unordered_map<uint32_t, uint32_t> layerMap_;
    std::unordered_map<std::string, ge::GNode> contextNodeMapA5_;
    std::unordered_map<std::string, int64_t> hcclBuffSizeCacheA5_;
};

}  // namespace ops

#endif  // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#endif  // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#endif  // OPS_TRANSFORMER_MC2_MOE_GRAPH_CONTEXT_H
