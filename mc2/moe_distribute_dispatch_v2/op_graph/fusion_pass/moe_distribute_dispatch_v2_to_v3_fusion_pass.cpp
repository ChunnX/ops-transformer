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
 * 将图中的 MoeDistributeDispatchV2 节点 1:1 替换为 MoeDistributeDispatchV3（仅 A5，comm_alg != "ccu"）。
 * V3 相比 V2：头部新增 context 输入（Mc2MoeContext 序列化数据，以 Const 节点内嵌于替换子图），
 * 属性删除 group_ep/group_tp，新增 ccl_buffer_size。与 Combine 侧 pass 共用 Mc2MoeContextManager，
 * 同一 groupEp 得到的 context 内容一致，保证 dispatch/combine 收发同一块 HCCL window buffer。
 *
 * pattern 覆盖 A5 上符合 README 的全部输入组合（共 32 种）：5 个可选输入
 * scales、x_active_mask、expert_scales、elastic_info、performance_info 任意组合。
 */
#include "moe_distribute_dispatch_v2_to_v3_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include <array>
#include "es_MoeDistributeDispatchV2.h"
#include "es_MoeDistributeDispatchV3.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "op_graph/mc2_moe_context_manager.h"
#include "ge/ge_utils.h"
#include "acl/acl_rt.h"

namespace ops {
namespace {
const std::string FUSION_PASS_NAME = "MoeDistributeDispatchV2FusionPass";
const int64_t DISPATCH_V2_CAPTURE_IDX = 0l;
const size_t DISPATCH_REQUIRED_INPUT_NUM = 2;
const size_t DISPATCH_OPTIONAL_INPUT_NUM = 5;
const uint32_t OPTIONAL_VARIANT_NUM = 1U << DISPATCH_OPTIONAL_INPUT_NUM; // 32

// 可选输入槽位（es 参数序，紧随 2 个必选输入之后），每个槽位对应一个自由 bit
const size_t SLOT_SCALES = 0;
const size_t SLOT_X_ACTIVE_MASK = 1;
const size_t SLOT_EXPERT_SCALES = 2;
const size_t SLOT_ELASTIC_INFO = 3;
const size_t SLOT_PERFORMANCE_INFO = 4;

ge::CustomPassStage GetDispatchV2ToV3FusionPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}

int64_t GetIntAttrOrDefault(const ge::GNode &node, const char *attrName, int64_t defaultValue)
{
    int64_t attrValue = defaultValue;
    if (node.GetAttr(attrName, attrValue) != ge::GRAPH_SUCCESS) {
        return defaultValue;
    }
    return attrValue;
}

std::string GetStrAttrOrDefault(const ge::GNode &node, const char *attrName, const std::string &defaultValue)
{
    ge::AscendString attrValue;
    if (node.GetAttr(attrName, attrValue) != ge::GRAPH_SUCCESS) {
        return defaultValue;
    }
    return attrValue.GetString() == nullptr ? defaultValue : attrValue.GetString();
}

std::string MakeVariantName(uint32_t optionalMask) { return FUSION_PASS_NAME + "_m" + std::to_string(optionalMask); }

bool ParseVariantName(const std::string &name, uint32_t &optionalMask)
{
    auto mPos = name.rfind("_m");
    if (mPos == std::string::npos || mPos + 2 >= name.size()) {
        return false;
    }
    optionalMask = static_cast<uint32_t>(std::stoul(name.substr(mPos + 2)));
    return (optionalMask < OPTIONAL_VARIANT_NUM);
}

std::array<bool, DISPATCH_OPTIONAL_INPUT_NUM> CalcOptionalPresence(uint32_t optionalMask)
{
    std::array<bool, DISPATCH_OPTIONAL_INPUT_NUM> presence{};
    for (size_t slot = 0; slot < DISPATCH_OPTIONAL_INPUT_NUM; ++slot) {
        presence[slot] = ((optionalMask >> slot) & 1U) != 0;
    }
    return presence;
}

ge::fusion::PatternUniqPtr MakePatternForDispatchV2(uint32_t optionalMask)
{
    auto graphBuilder = ge::es::EsGraphBuilder(MakeVariantName(optionalMask).c_str());
    auto presence = CalcOptionalPresence(optionalMask);
    auto holders =
        graphBuilder.CreateInputs(DISPATCH_REQUIRED_INPUT_NUM + static_cast<size_t>(__builtin_popcount(optionalMask)));
    size_t holderIdx = 0;
    ge::es::EsTensorHolder x = holders[holderIdx++];
    ge::es::EsTensorHolder expertIds = holders[holderIdx++];
    ge::es::EsTensorHolder optional[DISPATCH_OPTIONAL_INPUT_NUM];
    for (size_t slot = 0; slot < DISPATCH_OPTIONAL_INPUT_NUM; ++slot) {
        if (presence[slot]) {
            optional[slot] = holders[holderIdx++];
        }
    }
    auto dispatch = ge::es::MoeDistributeDispatchV2(
        x, expertIds, presence[SLOT_SCALES] ? optional[SLOT_SCALES] : nullptr,
        presence[SLOT_X_ACTIVE_MASK] ? optional[SLOT_X_ACTIVE_MASK] : nullptr,
        presence[SLOT_EXPERT_SCALES] ? optional[SLOT_EXPERT_SCALES] : nullptr,
        presence[SLOT_ELASTIC_INFO] ? optional[SLOT_ELASTIC_INFO] : nullptr,
        presence[SLOT_PERFORMANCE_INFO] ? optional[SLOT_PERFORMANCE_INFO] : nullptr, "", 0, 0, 0);
    auto graph = graphBuilder.BuildAndReset({dispatch.expand_x, dispatch.dynamic_scales,
                                             dispatch.assist_info_for_combine, dispatch.expert_token_nums,
                                             dispatch.ep_recv_count, dispatch.tp_recv_count, dispatch.expand_scales});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*dispatch.expand_x.GetProducer(), 0});
    return pattern;
}

bool CreateReplacementInputs(std::vector<ge::es::EsTensorHolder> &inputs, ge::es::EsGraphBuilder &replaceGraphBuilder,
                             const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    static const char *orderedInputNames[] = {
        "x", "expert_ids", "scales", "x_active_mask", "expert_scales", "elastic_info", "performance_info"};
    for (size_t i = 0; i < subgraphInputs.size(); ++i) {
        auto matchNodes = subgraphInputs[i].GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Subgraph input %zu has no producer.", i);
            return false;
        }
        ge::TensorDesc tmpDesc;
        if (matchNodes.at(0).node.GetInputDesc(matchNodes.at(0).index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "GetInputDesc for input %zu failed.", i);
            return false;
        }
        inputs.emplace_back(replaceGraphBuilder.CreateInput(static_cast<int64_t>(i), orderedInputNames[i],
                                                            tmpDesc.GetDataType(), tmpDesc.GetOriginFormat(),
                                                            tmpDesc.GetShape().GetDims()));
    }
    return true;
}

bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr &replaceGraph,
                            const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    std::vector<ge::Shape> inputShapes;
    for (const auto &subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            continue;
        }
        ge::TensorDesc tmpDesc;
        if (matchNodes.at(0).node.GetInputDesc(matchNodes.at(0).index, tmpDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
    }
    return (ge::GeUtils::InferShape(*replaceGraph, inputShapes) == ge::SUCCESS);
}
} // namespace

std::vector<ge::fusion::PatternUniqPtr> MoeDistributeDispatchV2FusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns for MoeDistributeDispatchV2FusionPass.");
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    for (uint32_t optionalMask = 0; optionalMask < OPTIONAL_VARIANT_NUM; ++optionalMask) {
        patternGraphs.emplace_back(MakePatternForDispatchV2(optionalMask));
    }
    return patternGraphs;
}

bool MoeDistributeDispatchV2FusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip when cann version not compatible.");
        return false;
    }

    // 仅 A5 支持 MoeDistributeDispatchV3
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Check target platform fail, only A5 supports V3.");
        return false;
    }

    ge::fusion::NodeIo dispatchNodeIo;
    if (matchResult->GetCapturedTensor(DISPATCH_V2_CAPTURE_IDX, dispatchNodeIo) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get MoeDistributeDispatchV2 node failed.");
        return false;
    }
    ge::GNode dispatchNode = dispatchNodeIo.node;

    // ccu 通信算法不支持 mc2Context，保持 V2 不变
    if (GetStrAttrOrDefault(dispatchNode, "comm_alg", "") == "ccu") {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "comm_alg is ccu, skip V2 to V3 fusion.");
        return false;
    }

    std::string groupEp = GetStrAttrOrDefault(dispatchNode, "group_ep", "");
    std::vector<int32_t> contextData;
    int64_t hcclBuffSize = 0;
    if (!ops::Mc2MoeContextManager::GetContextData(groupEp, contextData, hcclBuffSize)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Get mc2 context data failed, skip V2 to V3 fusion.");
        return false;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found one MoeDistributeDispatchV2 node that meets requirements.");
    return true;
}

ge::fusion::GraphUniqPtr MoeDistributeDispatchV2FusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::AscendString patternName;
    if (matchResult->GetPatternGraph().GetName(patternName) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    uint32_t optionalMask = 0;
    if (!ParseVariantName(patternName.GetString(), optionalMask)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Parse pattern variant name failed in Replacement.");
        return nullptr;
    }
    auto presence = CalcOptionalPresence(optionalMask);
    const size_t totalInputNum = DISPATCH_REQUIRED_INPUT_NUM + static_cast<size_t>(__builtin_popcount(optionalMask));

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS ||
        subgraphInputs.size() != totalInputNum) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed or input num %zu != expected %zu.",
                  subgraphInputs.size(), totalInputNum);
        return nullptr;
    }

    ge::fusion::NodeIo dispatchNodeIo;
    if (matchResult->GetCapturedTensor(DISPATCH_V2_CAPTURE_IDX, dispatchNodeIo) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Capture MoeDistributeDispatchV2 node failed.");
        return nullptr;
    }
    ge::GNode dispatchNode = dispatchNodeIo.node;

    std::string groupEp = GetStrAttrOrDefault(dispatchNode, "group_ep", "");
    std::vector<int32_t> contextData;
    int64_t hcclBuffSize = 0;
    if (!ops::Mc2MoeContextManager::GetContextData(groupEp, contextData, hcclBuffSize)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get mc2 context data failed in Replacement.");
        return nullptr;
    }

    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");
    std::vector<ge::es::EsTensorHolder> orderedInputs;
    if (!CreateReplacementInputs(orderedInputs, replaceGraphBuilder, subgraphInputs)) {
        return nullptr;
    }
    auto contextConst = replaceGraphBuilder.CreateConst(contextData, {static_cast<int64_t>(contextData.size())});

    // orderedInputs 按（2 必选 + 存在槽位）顺序排列，回填到 5 个可选槽位
    size_t inputIdx = DISPATCH_REQUIRED_INPUT_NUM;
    ge::es::EsTensorHolder optional[DISPATCH_OPTIONAL_INPUT_NUM];
    for (size_t slot = 0; slot < DISPATCH_OPTIONAL_INPUT_NUM; ++slot) {
        if (presence[slot]) {
            optional[slot] = orderedInputs[inputIdx++];
        }
    }

    std::string commAlg = GetStrAttrOrDefault(dispatchNode, "comm_alg", "");
    auto dispatchV3 = ge::es::MoeDistributeDispatchV3(
        contextConst, orderedInputs[0], orderedInputs[1], presence[SLOT_SCALES] ? optional[SLOT_SCALES] : nullptr,
        presence[SLOT_X_ACTIVE_MASK] ? optional[SLOT_X_ACTIVE_MASK] : nullptr,
        presence[SLOT_EXPERT_SCALES] ? optional[SLOT_EXPERT_SCALES] : nullptr,
        presence[SLOT_ELASTIC_INFO] ? optional[SLOT_ELASTIC_INFO] : nullptr,
        presence[SLOT_PERFORMANCE_INFO] ? optional[SLOT_PERFORMANCE_INFO] : nullptr,
        GetIntAttrOrDefault(dispatchNode, "ep_world_size", 0), GetIntAttrOrDefault(dispatchNode, "ep_rank_id", 0),
        GetIntAttrOrDefault(dispatchNode, "moe_expert_num", 0), hcclBuffSize,
        GetIntAttrOrDefault(dispatchNode, "tp_world_size", 0), GetIntAttrOrDefault(dispatchNode, "tp_rank_id", 0),
        GetIntAttrOrDefault(dispatchNode, "expert_shard_type", 0),
        GetIntAttrOrDefault(dispatchNode, "shared_expert_num", 1),
        GetIntAttrOrDefault(dispatchNode, "shared_expert_rank_num", 0),
        GetIntAttrOrDefault(dispatchNode, "quant_mode", 0), GetIntAttrOrDefault(dispatchNode, "global_bs", 0),
        GetIntAttrOrDefault(dispatchNode, "expert_token_nums_type", 1), commAlg.c_str(),
        GetIntAttrOrDefault(dispatchNode, "zero_expert_num", 0),
        GetIntAttrOrDefault(dispatchNode, "copy_expert_num", 0),
        GetIntAttrOrDefault(dispatchNode, "const_expert_num", 0),
        GetIntAttrOrDefault(dispatchNode, "y_dtype", static_cast<int64_t>(ge::DT_UNDEFINED)));

    std::vector<ge::es::EsTensorHolder> outputs = {
        dispatchV3.expand_x,          dispatchV3.dynamic_scales, dispatchV3.assist_info_for_combine,
        dispatchV3.expert_token_nums, dispatchV3.ep_recv_count,  dispatchV3.tp_recv_count,
        dispatchV3.expand_scales};
    auto replaceGraph = replaceGraphBuilder.BuildAndReset(outputs);
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replace MoeDistributeDispatchV2 with MoeDistributeDispatchV3 success.");
    return replaceGraph;
}

REG_FUSION_PASS(MoeDistributeDispatchV2FusionPass).Stage(GetDispatchV2ToV3FusionPassStage());
} // namespace ops
#endif // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
