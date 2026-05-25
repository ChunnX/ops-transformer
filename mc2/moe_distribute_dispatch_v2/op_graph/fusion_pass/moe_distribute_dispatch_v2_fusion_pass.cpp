/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * MoeDistributeDispatchV2 -> MoeDistributeDispatchV3 fusion pass (ES + SubgraphRewriter)
 *
 *   x expert_ids scales x_active_mask expert_scales ...
 *                         |
 *                MoeDistributeDispatchV2
 *                         |
 *   expand_x dynamic_scales assist_info_for_combine expert_token_nums
 *   ep_recv_counts tp_recv_counts expand_scales
 *
 *   After fusion (A5 only, non-ccu):
 *   context x expert_ids scales x_active_mask expert_scales ...
 *                         |
 *                MoeDistributeDispatchV3
 *                         |
 *   (same outputs)
 */
#include "moe_distribute_dispatch_v2_fusion_pass.h"
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/operator_factory.h"
#include "graph/ge_error_codes.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/graph_rewriter.h"
#include "es_MoeDistributeDispatchV3.h"
#include "mc2_common_log.h"
#include "mc2_platform_info.h"
#include "op_graph/mc2_moe_context.h"

namespace ops {
namespace {
const std::string PASS_NAME = "MoeDistributeDispatchV2FusionPass";
const std::string FUSED_OP_TYPE2 = "MoeDistributeDispatchV2";
const std::string MOE_CONTEXT_TAG = "MOE_DISPATCH_COMBINE_V2_CONTEXT";
const std::string MOE_EP_NAME = "group_ep";

constexpr size_t kV2TotalOutputs = 7;
}  // namespace

static bool BuildDispatchReplacement(const ge::GNode &moeNode, const Mc2MoeContext &mc2Ctx, int64_t cclBufferSize,
                                     std::unique_ptr<ge::Graph> &replacement)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Building dispatch V3 replacement graph");

    auto builder = ge::es::EsGraphBuilder("dispatch_v3_replace");

    size_t v2InputCount = moeNode.GetInputsSize();
    auto inputs = builder.CreateInputs(v2InputCount);

    int64_t ctxLen = static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t));
    auto ctxPtr = reinterpret_cast<const int32_t *>(&mc2Ctx);
    std::vector<int32_t> ctxData(ctxPtr, ctxPtr + ctxLen);
    auto contextConst = builder.CreateConst(ctxData, {ctxLen}, ge::DT_INT32);
    OPS_LOG_I(PASS_NAME.c_str(), "Context Const created (size=%ld, dtype=DT_INT32)", ctxLen);

    auto getIntAttr = [&](const char *name, int64_t &val) {
        ge::AscendString ascName(name);
        return moeNode.GetAttr(ascName, val) == ge::GRAPH_SUCCESS;
    };

    int64_t epWorldSize = 0, epRankId = 0, moeExpertNum = 0;
    if (!getIntAttr("ep_world_size", epWorldSize) ||
        !getIntAttr("ep_rank_id", epRankId) ||
        !getIntAttr("moe_expert_num", moeExpertNum)) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to get required attrs from V2");
        return false;
    }

    int64_t tpWorldSize = 0, tpRankId = 0, expertShardType = 0, sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 0, quantMode = 0, globalBs = 0, expertTokenNumsType = 1;
    int64_t zeroExpertNum = 0, copyExpertNum = 0, constExpertNum = 0, yDtype = 28;
    getIntAttr("tp_world_size", tpWorldSize);
    getIntAttr("tp_rank_id", tpRankId);
    getIntAttr("expert_shard_type", expertShardType);
    getIntAttr("shared_expert_num", sharedExpertNum);
    getIntAttr("shared_expert_rank_num", sharedExpertRankNum);
    getIntAttr("quant_mode", quantMode);
    getIntAttr("global_bs", globalBs);
    getIntAttr("expert_token_nums_type", expertTokenNumsType);
    getIntAttr("zero_expert_num", zeroExpertNum);
    getIntAttr("copy_expert_num", copyExpertNum);
    getIntAttr("const_expert_num", constExpertNum);
    getIntAttr("y_dtype", yDtype);

    std::string commAlg;
    ge::AscendString commAlgAsc;
    if (moeNode.GetAttr(ge::AscendString("comm_alg"), commAlgAsc) == ge::GRAPH_SUCCESS) {
        commAlg = commAlgAsc.GetString();
    }

    OPS_LOG_I(PASS_NAME.c_str(), "V2 attrs: epWorld=%ld epRank=%ld experts=%ld cclBuf=%ld",
              epWorldSize, epRankId, moeExpertNum, cclBufferSize);

    auto output = ge::es::MoeDistributeDispatchV3(
        contextConst,
        inputs[0],
        inputs[1],
        v2InputCount > 2 ? inputs[2] : ge::es::EsTensorHolder(),
        v2InputCount > 3 ? inputs[3] : ge::es::EsTensorHolder(),
        v2InputCount > 4 ? inputs[4] : ge::es::EsTensorHolder(),
        v2InputCount > 5 ? inputs[5] : ge::es::EsTensorHolder(),
        v2InputCount > 6 ? inputs[6] : ge::es::EsTensorHolder(),
        epWorldSize, epRankId, moeExpertNum, cclBufferSize,
        tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
        sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType,
        commAlg.c_str(), zeroExpertNum, copyExpertNum, constExpertNum, yDtype);

    replacement = builder.BuildAndReset({
        output.expand_x, output.dynamic_scales, output.assist_info_for_combine,
        output.expert_token_nums, output.ep_recv_count, output.tp_recv_count, output.expand_scales});
    if (replacement == nullptr) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to build replacement graph");
        return false;
    }

    OPS_LOG_I(PASS_NAME.c_str(), "Dispatch replacement graph built OK");
    return true;
}

static bool BuildDispatchBoundary(const ge::GNode &moeNode,
                                  std::unique_ptr<ge::fusion::SubgraphBoundary> &boundary)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Building dispatch boundary");
    boundary = std::make_unique<ge::fusion::SubgraphBoundary>();

    size_t v2InputCount = moeNode.GetInputsSize();
    for (size_t i = 0; i < v2InputCount; i++) {
        ge::fusion::SubgraphInput si;
        si.AddInput({moeNode, static_cast<int64_t>(i)});
        if (boundary->AddInput(static_cast<int64_t>(i), std::move(si)) != ge::SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "AddInput idx=%zu failed", i);
            return false;
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "Boundary inputs: %zu", v2InputCount);

    size_t mapped = 0;
    for (size_t i = 0; i < kV2TotalOutputs; i++) {
        auto consumers = moeNode.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (consumers.empty()) {
            OPS_LOG_W(PASS_NAME.c_str(), "V2 output %zu has no consumers, boundary output will be empty", i);
            mapped++;
            continue;
        }
        ge::fusion::SubgraphOutput so({*consumers[0].first, consumers[0].second});
        if (boundary->AddOutput(static_cast<int64_t>(i), std::move(so)) != ge::SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "AddOutput idx=%zu failed", i);
            return false;
        }
        mapped++;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "Boundary outputs: %zu/%zu", mapped, kV2TotalOutputs);

    return true;
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::Fusion(
    ge::Graph &graph, ge::GNode &moeNode)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Fusion start");
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    Mc2MoeContext mc2Ctx;
    int64_t cclBufferSize = 0;
    if (MoeContextManager::GetMc2ContextA5(MOE_CONTEXT_TAG, MOE_EP_NAME, moeNode, mc2Ctx, cclBufferSize) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_I(PASS_NAME.c_str(), "GetMc2ContextA5 not applicable, skip");
        return ge::GRAPH_SUCCESS;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "GetMc2ContextA5 OK, cclBufferSize=%ld", cclBufferSize);

    std::unique_ptr<ge::Graph> replacement;
    if (!BuildDispatchReplacement(moeNode, mc2Ctx, cclBufferSize, replacement)) {
        OPS_LOG_E(PASS_NAME.c_str(), "BuildDispatchReplacement failed");
        return ge::GRAPH_FAILED;
    }

    std::unique_ptr<ge::fusion::SubgraphBoundary> boundary;
    if (!BuildDispatchBoundary(moeNode, boundary)) {
        OPS_LOG_E(PASS_NAME.c_str(), "BuildDispatchBoundary failed");
        return ge::GRAPH_FAILED;
    }

    if (ge::fusion::SubgraphRewriter::Replace(*boundary, *replacement) != ge::SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "SubgraphRewriter::Replace failed");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "Dispatch V2->V3 fusion SUCCESS");
#endif
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::Run(
    ge::GraphPtr &graph, ge::CustomPassContext &pass_context)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Run start");
    (void)pass_context;

    if (IsTargetPlatformSocVersion(PASS_NAME.c_str(), PLATFORM_A2)) {
        OPS_LOG_I(PASS_NAME.c_str(), "A2 platform, skip");
        return ge::GRAPH_SUCCESS;
    }
    if (!IsTargetPlatformNpuArch(PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_I(PASS_NAME.c_str(), "Non-A5 platform, skip");
        return ge::GRAPH_SUCCESS;
    }

    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if (gNode.GetType(nodeType) != ge::GRAPH_SUCCESS || nodeType != FUSED_OP_TYPE2.c_str()) {
            continue;
        }

        ge::AscendString nameStr;
        gNode.GetName(nameStr);
        OPS_LOG_I(PASS_NAME.c_str(), "Found V2 node: %s", nameStr.GetString());

        ge::AscendString commAlg;
        if (gNode.GetAttr(ge::AscendString("comm_alg"), commAlg) == ge::GRAPH_SUCCESS &&
            std::string(commAlg.GetString()) == "ccu") {
            OPS_LOG_I(PASS_NAME.c_str(), "comm_alg=ccu, skip");
            continue;
        }

        auto ret = Fusion(*graph, gNode);
        if (ret == ge::GRAPH_SUCCESS) {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: fusion SUCCESS", nameStr.GetString());
        } else {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: fusion FAILED (ret=%d)", nameStr.GetString(), ret);
        }
    }

    OPS_LOG_I(PASS_NAME.c_str(), "Run end");
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(MoeDistributeDispatchV2FusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
}  // namespace ops
#endif  // GRAPH_FUSION_SUPPORT_VERSION
