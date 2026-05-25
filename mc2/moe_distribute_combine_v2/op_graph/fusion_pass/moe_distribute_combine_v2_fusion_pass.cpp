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
 * MoeDistributeCombineV2 -> MoeDistributeCombineV3 fusion pass (ES + SubgraphRewriter)
 *
 *   expand_x expert_ids assist_info_for_combine ep_send_counts expert_scales ...
 *                                   |
 *                          MoeDistributeCombineV2
 *                                   |
 *                                   x
 *
 *   After fusion (A5 only, non-ccu):
 *   context expand_x expert_ids assist_info_for_combine ep_send_counts expert_scales ...
 *                                   |
 *                          MoeDistributeCombineV3
 *                                   |
 *                                   x
 */
#include "moe_distribute_combine_v2_fusion_pass.h"
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/operator_factory.h"
#include "graph/ge_error_codes.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/graph_rewriter.h"
#include "es_MoeDistributeCombineV3.h"
#include "mc2_common_log.h"
#include "mc2_platform_info.h"
#include "op_graph/mc2_moe_context.h"

namespace ops {
namespace {
const std::string PASS_NAME = "MoeDistributeCombineV2FusionPass";
const std::string FUSED_OP_TYPE2 = "MoeDistributeCombineV2";
const std::string MOE_CONTEXT_TAG = "MOE_DISPATCH_COMBINE_V2_CONTEXT";
const std::string MOE_EP_NAME = "group_ep";

constexpr size_t kV2TotalInputs = 18;
constexpr size_t kV2TotalOutputs = 1;
}  // namespace

static bool BuildCombineReplacement(const ge::GNode &moeNode, const Mc2MoeContext &mc2Ctx, int64_t cclBufferSize,
                                    std::unique_ptr<ge::Graph> &replacement)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Building combine V3 replacement graph");

    auto builder = ge::es::EsGraphBuilder("combine_v3_replace");

    size_t v2InputCount = moeNode.GetInputsSize();
    auto inputs = builder.CreateInputs(v2InputCount);

    int64_t ctxLen = static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t));
    auto ctxPtr = reinterpret_cast<const int32_t *>(&mc2Ctx);
    std::vector<int32_t> ctxData(ctxPtr, ctxPtr + ctxLen);
    auto contextConst = builder.CreateConst(ctxData, {ctxLen}, ge::DT_INT32);
    OPS_LOG_I(PASS_NAME.c_str(), "Context Const created (size=%ld)", ctxLen);

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
    int64_t sharedExpertRankNum = 0, globalBs = 0, outDtype = 0, commQuantMode = 0;
    int64_t groupListType = 0, zeroExpertNum = 0, copyExpertNum = 0, constExpertNum = 0;
    getIntAttr("tp_world_size", tpWorldSize);
    getIntAttr("tp_rank_id", tpRankId);
    getIntAttr("expert_shard_type", expertShardType);
    getIntAttr("shared_expert_num", sharedExpertNum);
    getIntAttr("shared_expert_rank_num", sharedExpertRankNum);
    getIntAttr("global_bs", globalBs);
    getIntAttr("out_dtype", outDtype);
    getIntAttr("comm_quant_mode", commQuantMode);
    getIntAttr("group_list_type", groupListType);
    getIntAttr("zero_expert_num", zeroExpertNum);
    getIntAttr("copy_expert_num", copyExpertNum);
    getIntAttr("const_expert_num", constExpertNum);

    std::string commAlg;
    ge::AscendString commAlgAsc;
    if (moeNode.GetAttr(ge::AscendString("comm_alg"), commAlgAsc) == ge::GRAPH_SUCCESS) {
        commAlg = commAlgAsc.GetString();
    }

    OPS_LOG_I(PASS_NAME.c_str(), "V2 attrs: epWorld=%ld epRank=%ld experts=%ld cclBuf=%ld",
              epWorldSize, epRankId, moeExpertNum, cclBufferSize);

    // V2 ports: 0=expand_x 1=expert_ids 2=assist_info_for_combine 3=ep_send_counts
    //           4=expert_scales 5-17 optional
    auto output = ge::es::MoeDistributeCombineV3(
        contextConst,
        inputs[0],                                             // expand_x
        inputs[1],                                             // expert_ids
        inputs[2],                                             // assist_info_for_combine
        inputs[3],                                             // ep_send_counts
        inputs[4],                                             // expert_scales
        v2InputCount > 5 ? inputs[5] : ge::es::EsTensorHolder(),    // tp_send_counts
        v2InputCount > 6 ? inputs[6] : ge::es::EsTensorHolder(),    // x_active_mask
        v2InputCount > 7 ? inputs[7] : ge::es::EsTensorHolder(),    // activation_scale
        v2InputCount > 8 ? inputs[8] : ge::es::EsTensorHolder(),    // weight_scale
        v2InputCount > 9 ? inputs[9] : ge::es::EsTensorHolder(),    // group_list
        v2InputCount > 10 ? inputs[10] : ge::es::EsTensorHolder(),  // expand_scales
        v2InputCount > 11 ? inputs[11] : ge::es::EsTensorHolder(),  // shared_expert_x
        v2InputCount > 12 ? inputs[12] : ge::es::EsTensorHolder(),  // elastic_info
        v2InputCount > 13 ? inputs[13] : ge::es::EsTensorHolder(),  // ori_x
        v2InputCount > 14 ? inputs[14] : ge::es::EsTensorHolder(),  // const_expert_alpha_1
        v2InputCount > 15 ? inputs[15] : ge::es::EsTensorHolder(),  // const_expert_alpha_2
        v2InputCount > 16 ? inputs[16] : ge::es::EsTensorHolder(),  // const_expert_v
        v2InputCount > 17 ? inputs[17] : ge::es::EsTensorHolder(),  // performance_info
        epWorldSize, epRankId, moeExpertNum, cclBufferSize,
        tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
        sharedExpertRankNum, globalBs, outDtype, commQuantMode,
        groupListType, commAlg.c_str(), zeroExpertNum, copyExpertNum, constExpertNum);

    replacement = builder.BuildAndReset({output});
    if (replacement == nullptr) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to build replacement graph");
        return false;
    }

    OPS_LOG_I(PASS_NAME.c_str(), "Combine replacement graph built OK");
    return true;
}

static bool BuildCombineBoundary(const ge::GNode &moeNode,
                                 std::unique_ptr<ge::fusion::SubgraphBoundary> &boundary)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Building combine boundary");
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

    for (size_t i = 0; i < kV2TotalOutputs; i++) {
        auto consumers = moeNode.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (consumers.empty()) continue;
        ge::fusion::SubgraphOutput so({*consumers[0].first, consumers[0].second});
        if (boundary->AddOutput(static_cast<int64_t>(i), std::move(so)) != ge::SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "AddOutput idx=%zu failed", i);
            return false;
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "Boundary outputs: %zu", kV2TotalOutputs);

    return true;
}

ge::graphStatus MoeDistributeCombineV2FusionPass::Fusion(
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
    if (!BuildCombineReplacement(moeNode, mc2Ctx, cclBufferSize, replacement)) {
        OPS_LOG_E(PASS_NAME.c_str(), "BuildCombineReplacement failed");
        return ge::GRAPH_FAILED;
    }

    std::unique_ptr<ge::fusion::SubgraphBoundary> boundary;
    if (!BuildCombineBoundary(moeNode, boundary)) {
        OPS_LOG_E(PASS_NAME.c_str(), "BuildCombineBoundary failed");
        return ge::GRAPH_FAILED;
    }

    if (ge::fusion::SubgraphRewriter::Replace(*boundary, *replacement) != ge::SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "SubgraphRewriter::Replace failed");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "Combine V2->V3 fusion SUCCESS");
#endif
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2FusionPass::Run(
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

REG_FUSION_PASS(MoeDistributeCombineV2FusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
}  // namespace ops
#endif  // GRAPH_FUSION_SUPPORT_VERSION
