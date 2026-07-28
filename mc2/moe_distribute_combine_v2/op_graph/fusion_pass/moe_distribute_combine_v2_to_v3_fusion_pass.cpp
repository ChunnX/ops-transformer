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
 * 将图中的 MoeDistributeCombineV2 节点 1:1 替换为 MoeDistributeCombineV3（仅 A5，comm_alg != "ccu"）。
 * V3 相比 V2：头部新增 context 输入（Mc2MoeContext 序列化数据，以 Const 节点内嵌于替换子图），
 * 属性删除 group_ep/group_tp，新增 ccl_buffer_size。通信域 window buffer 地址在编译期通过
 * Mc2MoeContextManager 查询 HCCL 获得，与单算子路径（mc2_context.cpp）语义一致。
 *
 * pattern 覆盖 A5 上符合 README 的全部输入组合（共 48 种）：
 * - 自由位（4 bit）：x_active_mask、elastic_info、performance_info、shared_expert_x
 *   （shared_expert_x 存在时要求 sharedExpertRankNum != 0，MeetRequirements 校验）
 * - 特殊专家（3 态）：无 / copy 专家（ori_x）/ const 专家（ori_x + const_expert_alpha_1/2 + const_expert_v），
 *   由属性 copyExpertNum、constExpertNum 决定，MeetRequirements 校验输入与属性一致性
 * tp_send_counts、activation_scale、weight_scale、group_list、expand_scales 在 A5 不支持或被 tiling 禁用，
 * 含这些输入的节点不匹配、保持 V2 不变。
 */
#include "moe_distribute_combine_v2_to_v3_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include <array>
#include "es_MoeDistributeCombineV2.h"
#include "es_MoeDistributeCombineV3.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "op_graph/mc2_moe_context_manager.h"
#include "ge/ge_utils.h"
#include "acl/acl_rt.h"

namespace ops {
namespace {
const std::string FUSION_PASS_NAME = "MoeDistributeCombineV2ToV3FusionPass";
const int64_t COMBINE_V2_CAPTURE_IDX = 0l;
const size_t COMBINE_REQUIRED_INPUT_NUM = 5;
const size_t COMBINE_OPTIONAL_INPUT_NUM = 13;
const uint32_t FREE_BIT_NUM = 4; // x_active_mask、elastic_info、performance_info、shared_expert_x
const uint32_t FREE_VARIANT_NUM = 1U << FREE_BIT_NUM; // 16
const uint32_t EXPERT_BUNDLE_NUM = 3;                 // 无特殊专家 / copy 专家 / const 专家

// 自由位定义（bit 位置 -> 13 个可选输入在 es 参数序中的槽位）
const uint32_t FREE_BIT_X_ACTIVE_MASK = 0;
const uint32_t FREE_BIT_ELASTIC_INFO = 1;
const uint32_t FREE_BIT_PERFORMANCE_INFO = 2;
const uint32_t FREE_BIT_SHARED_EXPERT_X = 3;

// 可选输入槽位（es 参数序，紧随 5 个必选输入之后）
const size_t SLOT_TP_SEND_COUNTS = 0; // A5 无 TP 域，不生成
const size_t SLOT_X_ACTIVE_MASK = 1;
const size_t SLOT_ACTIVATION_SCALE = 2; // tiling 禁用，不生成
const size_t SLOT_WEIGHT_SCALE = 3;     // tiling 禁用，不生成
const size_t SLOT_GROUP_LIST = 4;       // tiling 禁用，不生成
const size_t SLOT_EXPAND_SCALES = 5;    // A5 不支持，不生成
const size_t SLOT_SHARED_EXPERT_X = 6;
const size_t SLOT_ELASTIC_INFO = 7;
const size_t SLOT_ORI_X = 8;
const size_t SLOT_CONST_EXPERT_ALPHA_1 = 9;
const size_t SLOT_CONST_EXPERT_ALPHA_2 = 10;
const size_t SLOT_CONST_EXPERT_V = 11;
const size_t SLOT_PERFORMANCE_INFO = 12;

enum ExpertBundle : uint32_t {
    EXPERT_BUNDLE_NONE = 0,  // 无特殊专家输入
    EXPERT_BUNDLE_COPY = 1,  // copyExpertNum > 0 且 constExpertNum == 0：仅 ori_x
    EXPERT_BUNDLE_CONST = 2, // constExpertNum > 0：ori_x + const_expert_alpha_1/2 + const_expert_v
};

ge::CustomPassStage GetCombineV2ToV3FusionPassStage()
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

std::string MakeVariantName(uint32_t freeMask, uint32_t bundle)
{
    return FUSION_PASS_NAME + "_f" + std::to_string(freeMask) + "_b" + std::to_string(bundle);
}

bool ParseVariantName(const std::string &name, uint32_t &freeMask, uint32_t &bundle)
{
    auto fPos = name.rfind("_f");
    auto bPos = name.rfind("_b");
    if (fPos == std::string::npos || bPos == std::string::npos || bPos <= fPos + 2) {
        return false;
    }
    freeMask = static_cast<uint32_t>(std::stoul(name.substr(fPos + 2, bPos - fPos - 2)));
    bundle = static_cast<uint32_t>(std::stoul(name.substr(bPos + 2)));
    return (freeMask < FREE_VARIANT_NUM && bundle < EXPERT_BUNDLE_NUM);
}

// 计算指定变体下 13 个可选输入槽位的存在性
std::array<bool, COMBINE_OPTIONAL_INPUT_NUM> CalcOptionalPresence(uint32_t freeMask, uint32_t bundle)
{
    std::array<bool, COMBINE_OPTIONAL_INPUT_NUM> presence{};
    presence[SLOT_X_ACTIVE_MASK] = ((freeMask >> FREE_BIT_X_ACTIVE_MASK) & 1U) != 0;
    presence[SLOT_ELASTIC_INFO] = ((freeMask >> FREE_BIT_ELASTIC_INFO) & 1U) != 0;
    presence[SLOT_PERFORMANCE_INFO] = ((freeMask >> FREE_BIT_PERFORMANCE_INFO) & 1U) != 0;
    presence[SLOT_SHARED_EXPERT_X] = ((freeMask >> FREE_BIT_SHARED_EXPERT_X) & 1U) != 0;
    presence[SLOT_ORI_X] = (bundle != EXPERT_BUNDLE_NONE);
    presence[SLOT_CONST_EXPERT_ALPHA_1] = (bundle == EXPERT_BUNDLE_CONST);
    presence[SLOT_CONST_EXPERT_ALPHA_2] = (bundle == EXPERT_BUNDLE_CONST);
    presence[SLOT_CONST_EXPERT_V] = (bundle == EXPERT_BUNDLE_CONST);
    return presence;
}

ge::fusion::PatternUniqPtr MakePatternForCombineV2(uint32_t freeMask, uint32_t bundle)
{
    auto graphBuilder = ge::es::EsGraphBuilder(MakeVariantName(freeMask, bundle).c_str());
    auto presence = CalcOptionalPresence(freeMask, bundle);
    auto holders =
        graphBuilder.CreateInputs(COMBINE_REQUIRED_INPUT_NUM + static_cast<size_t>(__builtin_popcount(freeMask)) +
                                  ((bundle == EXPERT_BUNDLE_NONE) ? 0 : ((bundle == EXPERT_BUNDLE_COPY) ? 1 : 4)));
    size_t holderIdx = 0;
    ge::es::EsTensorHolder required[COMBINE_REQUIRED_INPUT_NUM];
    for (size_t i = 0; i < COMBINE_REQUIRED_INPUT_NUM; ++i) {
        required[i] = holders[holderIdx++];
    }
    ge::es::EsTensorHolder optional[COMBINE_OPTIONAL_INPUT_NUM];
    for (size_t slot = 0; slot < COMBINE_OPTIONAL_INPUT_NUM; ++slot) {
        if (presence[slot]) {
            optional[slot] = holders[holderIdx++];
        }
    }
    auto combine = ge::es::MoeDistributeCombineV2(
        required[0], required[1], required[2], required[3], required[4],
        presence[SLOT_TP_SEND_COUNTS] ? optional[SLOT_TP_SEND_COUNTS] : nullptr,
        presence[SLOT_X_ACTIVE_MASK] ? optional[SLOT_X_ACTIVE_MASK] : nullptr,
        presence[SLOT_ACTIVATION_SCALE] ? optional[SLOT_ACTIVATION_SCALE] : nullptr,
        presence[SLOT_WEIGHT_SCALE] ? optional[SLOT_WEIGHT_SCALE] : nullptr,
        presence[SLOT_GROUP_LIST] ? optional[SLOT_GROUP_LIST] : nullptr,
        presence[SLOT_EXPAND_SCALES] ? optional[SLOT_EXPAND_SCALES] : nullptr,
        presence[SLOT_SHARED_EXPERT_X] ? optional[SLOT_SHARED_EXPERT_X] : nullptr,
        presence[SLOT_ELASTIC_INFO] ? optional[SLOT_ELASTIC_INFO] : nullptr,
        presence[SLOT_ORI_X] ? optional[SLOT_ORI_X] : nullptr,
        presence[SLOT_CONST_EXPERT_ALPHA_1] ? optional[SLOT_CONST_EXPERT_ALPHA_1] : nullptr,
        presence[SLOT_CONST_EXPERT_ALPHA_2] ? optional[SLOT_CONST_EXPERT_ALPHA_2] : nullptr,
        presence[SLOT_CONST_EXPERT_V] ? optional[SLOT_CONST_EXPERT_V] : nullptr,
        presence[SLOT_PERFORMANCE_INFO] ? optional[SLOT_PERFORMANCE_INFO] : nullptr, "", 0, 0, 0);
    auto graph = graphBuilder.BuildAndReset({combine});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*combine.GetProducer(), 0});
    return pattern;
}

// 根据属性推导特殊专家输入应有的形态，与匹配到的变体比对，保证输入与属性符合 README 约束
bool CheckExpertBundleConsistent(const ge::GNode &node, uint32_t bundle)
{
    int64_t copyExpertNum = GetIntAttrOrDefault(node, "copy_expert_num", 0);
    int64_t constExpertNum = GetIntAttrOrDefault(node, "const_expert_num", 0);
    uint32_t expected = EXPERT_BUNDLE_NONE;
    if (constExpertNum > 0) {
        expected = EXPERT_BUNDLE_CONST;
    } else if (copyExpertNum > 0) {
        expected = EXPERT_BUNDLE_COPY;
    }
    if (expected != bundle) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "Expert bundle mismatch: copyExpertNum %ld, constExpertNum %ld, matched bundle %u.", copyExpertNum,
                  constExpertNum, bundle);
        return false;
    }
    return true;
}

bool CreateReplacementInputs(std::vector<ge::es::EsTensorHolder> &inputs, ge::es::EsGraphBuilder &replaceGraphBuilder,
                             const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    static const char *orderedInputNames[] = {
        "expand_x",        "expert_ids",           "assist_info_for_combine", "ep_send_counts",
        "expert_scales",   "x_active_mask",        "shared_expert_x",         "elastic_info",
        "ori_x",           "const_expert_alpha_1", "const_expert_alpha_2",    "const_expert_v",
        "performance_info"};
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

std::vector<ge::fusion::PatternUniqPtr> MoeDistributeCombineV2ToV3FusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns for MoeDistributeCombineV2ToV3FusionPass.");
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    for (uint32_t freeMask = 0; freeMask < FREE_VARIANT_NUM; ++freeMask) {
        for (uint32_t bundle = 0; bundle < EXPERT_BUNDLE_NUM; ++bundle) {
            patternGraphs.emplace_back(MakePatternForCombineV2(freeMask, bundle));
        }
    }
    return patternGraphs;
}

bool MoeDistributeCombineV2ToV3FusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip when cann version not compatible.");
        return false;
    }

    // 仅 A5 支持 MoeDistributeCombineV3
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Check target platform fail, only A5 supports V3.");
        return false;
    }

    ge::fusion::NodeIo combineNodeIo;
    if (matchResult->GetCapturedTensor(COMBINE_V2_CAPTURE_IDX, combineNodeIo) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get MoeDistributeCombineV2 node failed.");
        return false;
    }
    ge::GNode combineNode = combineNodeIo.node;

    // ccu 通信算法不支持 mc2Context，保持 V2 不变
    if (GetStrAttrOrDefault(combineNode, "comm_alg", "") == "ccu") {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "comm_alg is ccu, skip V2 to V3 fusion.");
        return false;
    }

    // 输入组合需与属性一致（README 约束）
    ge::AscendString patternName;
    if (matchResult->GetPatternGraph().GetName(patternName) != ge::GRAPH_SUCCESS) {
        return false;
    }
    uint32_t freeMask = 0;
    uint32_t bundle = 0;
    if (!ParseVariantName(patternName.GetString(), freeMask, bundle)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Parse pattern variant name failed.");
        return false;
    }
    if (!CheckExpertBundleConsistent(combineNode, bundle)) {
        return false;
    }
    // shared_expert_x 存在时要求 sharedExpertRankNum 非 0（tiling 约束）
    if (((freeMask >> FREE_BIT_SHARED_EXPERT_X) & 1U) != 0 &&
        GetIntAttrOrDefault(combineNode, "shared_expert_rank_num", 0) == 0) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "shared_expert_x present but sharedExpertRankNum is 0, skip.");
        return false;
    }

    std::string groupEp = GetStrAttrOrDefault(combineNode, "group_ep", "");
    std::vector<int32_t> contextData;
    int64_t hcclBuffSize = 0;
    if (!ops::Mc2MoeContextManager::GetContextData(groupEp, contextData, hcclBuffSize)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Get mc2 context data failed, skip V2 to V3 fusion.");
        return false;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found one MoeDistributeCombineV2 node that meets requirements.");
    return true;
}

ge::fusion::GraphUniqPtr
MoeDistributeCombineV2ToV3FusionPass::Replacement(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::AscendString patternName;
    if (matchResult->GetPatternGraph().GetName(patternName) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    uint32_t freeMask = 0;
    uint32_t bundle = 0;
    if (!ParseVariantName(patternName.GetString(), freeMask, bundle)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Parse pattern variant name failed in Replacement.");
        return nullptr;
    }
    auto presence = CalcOptionalPresence(freeMask, bundle);
    const size_t totalInputNum = COMBINE_REQUIRED_INPUT_NUM + static_cast<size_t>(__builtin_popcount(freeMask)) +
                                 ((bundle == EXPERT_BUNDLE_NONE) ? 0 : ((bundle == EXPERT_BUNDLE_COPY) ? 1 : 4));

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS ||
        subgraphInputs.size() != totalInputNum) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed or input num %zu != expected %zu.",
                  subgraphInputs.size(), totalInputNum);
        return nullptr;
    }

    ge::fusion::NodeIo combineNodeIo;
    if (matchResult->GetCapturedTensor(COMBINE_V2_CAPTURE_IDX, combineNodeIo) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Capture MoeDistributeCombineV2 node failed.");
        return nullptr;
    }
    ge::GNode combineNode = combineNodeIo.node;

    std::string groupEp = GetStrAttrOrDefault(combineNode, "group_ep", "");
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

    // orderedInputs 按（5 必选 + 存在槽位）顺序排列，回填到 13 个可选槽位
    size_t inputIdx = COMBINE_REQUIRED_INPUT_NUM;
    ge::es::EsTensorHolder optional[COMBINE_OPTIONAL_INPUT_NUM];
    for (size_t slot = 0; slot < COMBINE_OPTIONAL_INPUT_NUM; ++slot) {
        if (presence[slot]) {
            optional[slot] = orderedInputs[inputIdx++];
        }
    }

    std::string commAlg = GetStrAttrOrDefault(combineNode, "comm_alg", "");
    auto combineV3 = ge::es::MoeDistributeCombineV3(
        contextConst, orderedInputs[0], orderedInputs[1], orderedInputs[2], orderedInputs[3], orderedInputs[4],
        presence[SLOT_TP_SEND_COUNTS] ? optional[SLOT_TP_SEND_COUNTS] : nullptr,
        presence[SLOT_X_ACTIVE_MASK] ? optional[SLOT_X_ACTIVE_MASK] : nullptr,
        presence[SLOT_ACTIVATION_SCALE] ? optional[SLOT_ACTIVATION_SCALE] : nullptr,
        presence[SLOT_WEIGHT_SCALE] ? optional[SLOT_WEIGHT_SCALE] : nullptr,
        presence[SLOT_GROUP_LIST] ? optional[SLOT_GROUP_LIST] : nullptr,
        presence[SLOT_EXPAND_SCALES] ? optional[SLOT_EXPAND_SCALES] : nullptr,
        presence[SLOT_SHARED_EXPERT_X] ? optional[SLOT_SHARED_EXPERT_X] : nullptr,
        presence[SLOT_ELASTIC_INFO] ? optional[SLOT_ELASTIC_INFO] : nullptr,
        presence[SLOT_ORI_X] ? optional[SLOT_ORI_X] : nullptr,
        presence[SLOT_CONST_EXPERT_ALPHA_1] ? optional[SLOT_CONST_EXPERT_ALPHA_1] : nullptr,
        presence[SLOT_CONST_EXPERT_ALPHA_2] ? optional[SLOT_CONST_EXPERT_ALPHA_2] : nullptr,
        presence[SLOT_CONST_EXPERT_V] ? optional[SLOT_CONST_EXPERT_V] : nullptr,
        presence[SLOT_PERFORMANCE_INFO] ? optional[SLOT_PERFORMANCE_INFO] : nullptr,
        GetIntAttrOrDefault(combineNode, "ep_world_size", 0), GetIntAttrOrDefault(combineNode, "ep_rank_id", 0),
        GetIntAttrOrDefault(combineNode, "moe_expert_num", 0), hcclBuffSize,
        GetIntAttrOrDefault(combineNode, "tp_world_size", 0), GetIntAttrOrDefault(combineNode, "tp_rank_id", 0),
        GetIntAttrOrDefault(combineNode, "expert_shard_type", 0),
        GetIntAttrOrDefault(combineNode, "shared_expert_num", 1),
        GetIntAttrOrDefault(combineNode, "shared_expert_rank_num", 0), GetIntAttrOrDefault(combineNode, "global_bs", 0),
        GetIntAttrOrDefault(combineNode, "out_dtype", 0), GetIntAttrOrDefault(combineNode, "comm_quant_mode", 0),
        GetIntAttrOrDefault(combineNode, "group_list_type", 0), commAlg.c_str(),
        GetIntAttrOrDefault(combineNode, "zero_expert_num", 0), GetIntAttrOrDefault(combineNode, "copy_expert_num", 0),
        GetIntAttrOrDefault(combineNode, "const_expert_num", 0));

    auto replaceGraph = replaceGraphBuilder.BuildAndReset({combineV3});
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replace MoeDistributeCombineV2 with MoeDistributeCombineV3 success.");
    return replaceGraph;
}

REG_FUSION_PASS(MoeDistributeCombineV2ToV3FusionPass).Stage(GetCombineV2ToV3FusionPassStage());
} // namespace ops
#endif // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
