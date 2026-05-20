/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file mc2_moe_context.h
 * \brief MoeContextManager - public API version
 */
#ifndef OPS_TRANSFORMER_MC2_MOE_CONTEXT_H
#define OPS_TRANSFORMER_MC2_MOE_CONTEXT_H
#define HCCL_CHANNEL_SUPPORT_VERSION 90000000

#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/operator.h"
#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "graph/ge_error_codes.h"
#include "hccl/hccl_rank_graph.h"
#include "mc2_common_log.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>

namespace ops {

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer_[1024];
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

class MoeContextManager
{
public:
    static ge::graphStatus GetMc2ContextA5(
        const std::string& nodeName, const std::string& epName,
        Mc2MoeContext& context, int64_t& hcclBuffSize);

private:
    MoeContextManager() = default;
    ~MoeContextManager();
    MoeContextManager(const MoeContextManager&) = delete;
    MoeContextManager& operator=(const MoeContextManager&) = delete;
    static MoeContextManager& GetInstance();
    static std::string FUSED_OP_TYPE2_A5;

    ge::graphStatus LoadHcclSymbols();
    ge::graphStatus LoadSymbol(void** ptr, const char* name);
    ge::graphStatus GetCommHandle(const char* groupEp, HcclComm& hcclHandle);
    ge::graphStatus GetHcclContext(const std::string& graphName, const std::string& groupEp,
        const std::string& nodeName, Mc2MoeContext* context, int64_t& hcclBuffSize);
    ge::graphStatus GetHcclCommLink(const HcclComm& hcclHandle, const LinkQueryParams& params, CommLink*& links);
    ge::graphStatus GetRankSizePerServer(const HcclComm& hcclHandle, uint32_t netLayers);
    ge::graphStatus InitHcclChannel(const HcclComm& hcclHandle, const ChannelInitParams& params,
        std::vector<HcclChannelDesc>& channelDesc);
    ge::graphStatus SetupChannelDesc(const HcclComm& hcclHandle, const ChannelInitParams& params,
        uint32_t* netLayerList, uint32_t netLayerNum, std::vector<HcclChannelDesc>& channelDesc);
    ge::graphStatus GetHcclCommChannel(const HcclComm& hcclHandle, const ChannelAcquireParams& params,
        std::vector<ChannelHandle>& channels);
    ge::graphStatus GetHcclCommResource(const HcclComm& hcclHandle, const CommEngine& engine,
        const CommProtocol& protocol, Mc2MoeContext* context);
    ge::graphStatus FillContextBuffer(const HcclComm& hcclHandle, const CommEngine& engine,
        Mc2MoeContext* context, uint32_t rankId, const std::vector<ChannelHandle>& channels);
    ge::graphStatus CreatMc2Context(const ContextCreateParams& params, Mc2MoeContext* context);
    ge::graphStatus CheckLinks(uint32_t& netLinkNum, CommLink* linksList);
    ge::graphStatus CheckProtocolSupport(const HcclComm& hcclHandle, uint32_t*& layerList, uint32_t& layerNum);
    ge::graphStatus GetCommProtocol(const HcclComm& hcclHandle, CommProtocol& protocol);
    ge::graphStatus ValidateContextTag(const std::string& mc2ContextTag);
    ge::graphStatus GetMc2Context(const ContextAcquireParams& params, Mc2MoeContext* context, int64_t& hcclBuffSize);
    const std::string GetLibPath();
    template <typename T>
    T GetHcclLibFunc(void* handle, const std::string& funcName);

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
    HcclResult (*HcclGetRankId)(HcclComm, uint32_t*){nullptr};
    HcclResult (*HcclGetRankSize)(HcclComm, uint32_t*){nullptr};
    HcclResult (*HcclRankGraphGetRanksByLayer)(HcclComm, uint32_t, uint32_t**, uint32_t*){nullptr};

    std::unordered_map<uint32_t, uint32_t> layerMap;
    std::unordered_map<std::string, Mc2MoeContext> contextCacheMapA5;
    std::unordered_map<std::string, int64_t> bufferSizeMapA5;
};

} // namespace ops
#endif // OPS_TRANSFORMER_MC2_MOE_CONTEXT_H
