/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_distribute_combine_add_rms_norm_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_MoeDistributeCombineV2.h"
#include "es_MoeDistributeCombineAddRmsNorm.h"
#include "es_nn_ops.h"   // nn ops stub
#include "es_math_ops.h" // math ops stub
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include "acl/acl_rt.h"

namespace ops {
namespace {
const std::string PASS_NAME = "MoeDistributeCombineAddRmsNormFusionPass";
const std::string PATTERN_TP_SEND = "HasTpSend";
const std::string PATTERN_ACTIVE_MASK = "HasActiveMask";
const std::string PATTERN_ACT_SCALE = "HasActScale";
const std::string PATTERN_WEIGHT_SCALE = "HasWeightScale";
const std::string PATTERN_GROUP_LIST = "HasGroupList";
const std::string PATTERN_EXPAND_SCALES = "HasExpandScales";

const int64_t COMBINE_CAPTURE_IDX = 0l;
const int64_t ADD_CAPTURE_IDX = 1l;
const int64_t ARN_CAPTURE_IDX = 2l;

const int32_t COMBINE_EXPAND_X_IDX = 0;
const int32_t ADD_X2_IDX = 1;
const int32_t ARN_X1_IDX = 0;
const int32_t DATA_OUT_IDX = 0;

const size_t EXPAND_X_RANK = 2;
const size_t X_RANK = 3;
const int64_t MAX_H_SIZE = 7168;
const int64_t MAX_BS_SIZE = 192;
const int64_t MAX_EP_RANK_SIZE = 288;

// canndev Relink：Combine 可选口仅迁到 expand_scales；其后 X2→fusion.shared_expert_x
// elastic_info / ori_x / const_* / performance_info 固定 nullptr（canndev 未迁）
struct OriginalGraphInfo {
    bool hasTpSend = false;
    bool hasActiveMask = false;
    bool hasActScale = false;
    bool hasWeightScale = false;
    bool hasGroupList = false;
    bool hasExpandScales = false;
};

struct ReplaceGraphInputs {
    ge::es::EsTensorHolder rExpandX;
    ge::es::EsTensorHolder rExpertIds;
    ge::es::EsTensorHolder rAssist;
    ge::es::EsTensorHolder rEpSend;
    ge::es::EsTensorHolder rExpertScales;
    ge::es::EsTensorHolder rResidualX; // pattern X1 / InplaceAddRmsNorm.x1
    ge::es::EsTensorHolder rGamma;
    ge::es::EsTensorHolder rTpSend;
    ge::es::EsTensorHolder rActiveMask;
    ge::es::EsTensorHolder rActScale;
    ge::es::EsTensorHolder rWeightScale;
    ge::es::EsTensorHolder rGroupList;
    ge::es::EsTensorHolder rExpandScales;
    ge::es::EsTensorHolder rSharedExpertX; // pattern X2 / Add.x2（对齐 canndev Relink idx13）
};

ge::CustomPassStage GetPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}

size_t GetPatternInputNum(const OriginalGraphInfo &info)
{
    // 5 combine 必选 + residual_x + gamma + x2；再加 canndev 会迁的 6 个可选
    size_t n = 8;
    if (info.hasTpSend) {
        ++n;
    }
    if (info.hasActiveMask) {
        ++n;
    }
    if (info.hasActScale) {
        ++n;
    }
    if (info.hasWeightScale) {
        ++n;
    }
    if (info.hasGroupList) {
        ++n;
    }
    if (info.hasExpandScales) {
        ++n;
    }
    return n;
}

ge::fusion::PatternUniqPtr MakePattern(const OriginalGraphInfo &info)
{
    std::string patternName = PASS_NAME;
    if (info.hasTpSend) {
        patternName += PATTERN_TP_SEND;
    }
    if (info.hasActiveMask) {
        patternName += PATTERN_ACTIVE_MASK;
    }
    if (info.hasActScale) {
        patternName += PATTERN_ACT_SCALE;
    }
    if (info.hasWeightScale) {
        patternName += PATTERN_WEIGHT_SCALE;
    }
    if (info.hasGroupList) {
        patternName += PATTERN_GROUP_LIST;
    }
    if (info.hasExpandScales) {
        patternName += PATTERN_EXPAND_SCALES;
    }

    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto inputs = graphBuilder.CreateInputs(GetPatternInputNum(info));
    size_t idx = 0;
    auto expandX = inputs[idx++];
    auto expertIds = inputs[idx++];
    auto assist = inputs[idx++];
    auto epSend = inputs[idx++];
    auto expertScales = inputs[idx++];
    auto residualX = inputs[idx++]; // X1
    auto gamma = inputs[idx++];

    ge::es::EsTensorHolder tpSend = nullptr;
    if (info.hasTpSend) {
        tpSend = inputs[idx++];
    }
    ge::es::EsTensorHolder activeMask = nullptr;
    if (info.hasActiveMask) {
        activeMask = inputs[idx++];
    }
    ge::es::EsTensorHolder actScale = nullptr;
    if (info.hasActScale) {
        actScale = inputs[idx++];
    }
    ge::es::EsTensorHolder weightScale = nullptr;
    if (info.hasWeightScale) {
        weightScale = inputs[idx++];
    }
    ge::es::EsTensorHolder groupList = nullptr;
    if (info.hasGroupList) {
        groupList = inputs[idx++];
    }
    ge::es::EsTensorHolder expandScales = nullptr;
    if (info.hasExpandScales) {
        expandScales = inputs[idx++];
    }
    auto x2 = inputs[idx++]; // Add 的另一路，融后接到 shared_expert_x

    const char *groupEp = "";
    const int64_t epWorldSize = 0;
    const int64_t epRankId = 0;
    const int64_t moeExpertNum = 0;
    // shared_expert_x / elastic / ori_x / const_* / performance 固定 nullptr
    auto combineY = ge::es::MoeDistributeCombineV2(expandX, expertIds, assist, epSend, expertScales, tpSend, activeMask,
                                                   actScale, weightScale, groupList, expandScales, nullptr, nullptr,
                                                   nullptr, nullptr, nullptr, nullptr, nullptr, groupEp, epWorldSize,
                                                   epRankId, moeExpertNum);

    // shape Const 进 pattern 内部，不占 subgraph 边界（对齐 canndev 删掉 reshape const）
    auto reshaped = ge::es::Reshape(combineY, ge::es::EsTensorLike(std::vector<int64_t>{-1, 1, -1}));
    auto addY = ge::es::Add(reshaped, x2);
    auto arn = ge::es::InplaceAddRmsNorm(residualX, addY, gamma);

    auto graph = graphBuilder.BuildAndReset({arn.ref_x1, arn.rstd, arn.ref_x2});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*combineY.GetProducer(), 0})
        .CaptureTensor({*addY.GetProducer(), 0})
        .CaptureTensor({*arn.ref_x1.GetProducer(), 0});
    return pattern;
}

bool GetPatternNameStr(const std::unique_ptr<ge::fusion::MatchResult> &matchResult, std::string &patternNameStr)
{
    ge::AscendString patternName = "";
    if (matchResult->GetPatternGraph().GetName(patternName) != ge::SUCCESS) {
        OPS_LOG_W(PASS_NAME.c_str(), "Get pattern graph name failed.");
        return false;
    }
    patternNameStr = patternName.GetString() != nullptr ? patternName.GetString() : "";
    return true;
}

OriginalGraphInfo ParsePatternInfo(const std::string &patternNameStr)
{
    OriginalGraphInfo info;
    info.hasTpSend = patternNameStr.find(PATTERN_TP_SEND) != std::string::npos;
    info.hasActiveMask = patternNameStr.find(PATTERN_ACTIVE_MASK) != std::string::npos;
    info.hasActScale = patternNameStr.find(PATTERN_ACT_SCALE) != std::string::npos;
    info.hasWeightScale = patternNameStr.find(PATTERN_WEIGHT_SCALE) != std::string::npos;
    info.hasGroupList = patternNameStr.find(PATTERN_GROUP_LIST) != std::string::npos;
    info.hasExpandScales = patternNameStr.find(PATTERN_EXPAND_SCALES) != std::string::npos;
    return info;
}

bool GetCapturedNode(const std::unique_ptr<ge::fusion::MatchResult> &matchResult, int64_t captureIdx, ge::GNode &node)
{
    ge::fusion::NodeIo nodeIo;
    if (matchResult->GetCapturedTensor(captureIdx, nodeIo) != ge::SUCCESS) {
        OPS_LOG_W(PASS_NAME.c_str(), "Get captured tensor failed, idx=%ld.", captureIdx);
        return false;
    }
    node = nodeIo.node;
    return true;
}

bool CheckNodeShape(ge::GNode &combineNode, ge::GNode &addNode, ge::GNode &arnNode)
{
    ge::TensorDesc expandXDesc;
    ge::TensorDesc x2Desc;
    ge::TensorDesc x1Desc;
    if (combineNode.GetInputDesc(COMBINE_EXPAND_X_IDX, expandXDesc) != ge::GRAPH_SUCCESS ||
        addNode.GetInputDesc(ADD_X2_IDX, x2Desc) != ge::GRAPH_SUCCESS ||
        arnNode.GetInputDesc(ARN_X1_IDX, x1Desc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(PASS_NAME.c_str(), "GetInputDesc failed in CheckNodeShape.");
        return false;
    }
    const auto expandXDims = expandXDesc.GetShape().GetDims();
    const auto x2Dims = x2Desc.GetShape().GetDims();
    const auto x1Dims = x1Desc.GetShape().GetDims();
    if (expandXDims.size() != EXPAND_X_RANK) {
        OPS_LOG_D(PASS_NAME.c_str(), "expand_x shape is not right.");
        return false;
    }
    if (x2Dims.size() != X_RANK || x2Dims[0] > MAX_BS_SIZE) {
        OPS_LOG_D(PASS_NAME.c_str(), "x2 shape is not right.");
        return false;
    }
    if (x1Dims.size() != X_RANK || x1Dims[0] > MAX_BS_SIZE) {
        OPS_LOG_D(PASS_NAME.c_str(), "x1 shape is not right.");
        return false;
    }
    if (x2Dims[0] != x1Dims[0] || x2Dims[1] != x1Dims[1] || x2Dims[2] != x1Dims[2]) {
        OPS_LOG_D(PASS_NAME.c_str(), "x2 is not the same as x1.");
        return false;
    }
    if (expandXDims[1] != x1Dims[2]) {
        OPS_LOG_D(PASS_NAME.c_str(), "expand_x is not matched to x1.");
        return false;
    }
    if (expandXDims[0] > (MAX_BS_SIZE * MAX_EP_RANK_SIZE) || expandXDims[1] > MAX_H_SIZE) {
        OPS_LOG_D(PASS_NAME.c_str(), "expand_x is too huge to merge.");
        return false;
    }
    return true;
}

bool CheckNodeDataType(ge::GNode &combineNode, ge::GNode &addNode, ge::GNode &arnNode)
{
    ge::TensorDesc expandXDesc;
    ge::TensorDesc x2Desc;
    ge::TensorDesc x1Desc;
    if (combineNode.GetInputDesc(COMBINE_EXPAND_X_IDX, expandXDesc) != ge::GRAPH_SUCCESS ||
        addNode.GetInputDesc(ADD_X2_IDX, x2Desc) != ge::GRAPH_SUCCESS ||
        arnNode.GetInputDesc(ARN_X1_IDX, x1Desc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(PASS_NAME.c_str(), "GetInputDesc failed.");
        return false;
    }
    if (expandXDesc.GetDataType() != ge::DT_BF16) {
        OPS_LOG_D(PASS_NAME.c_str(), "expand_x dtype is not right.");
        return false;
    }
    if (x2Desc.GetDataType() != ge::DT_BF16) {
        OPS_LOG_D(PASS_NAME.c_str(), "x2 dtype is not right.");
        return false;
    }
    if (x1Desc.GetDataType() != ge::DT_BF16) {
        OPS_LOG_D(PASS_NAME.c_str(), "x1 dtype is not right.");
        return false;
    }
    return true;
}

bool CheckNodeDataAnchor(ge::GNode &combineNode, ge::GNode &addNode, ge::GNode &arnNode)
{
    if (combineNode.GetOutDataNodesAndPortIndexs(DATA_OUT_IDX).size() != 1U) {
        OPS_LOG_D(PASS_NAME.c_str(), "MoeDistributeCombineV2 output is not right.");
        return false;
    }
    if (addNode.GetInputsSize() != 2U) {
        OPS_LOG_D(PASS_NAME.c_str(), "add input is not right.");
        return false;
    }
    if (addNode.GetOutDataNodesAndPortIndexs(DATA_OUT_IDX).size() != 1U) {
        OPS_LOG_D(PASS_NAME.c_str(), "add output is not right.");
        return false;
    }
    if (arnNode.GetInputsSize() != 3U) {
        OPS_LOG_D(PASS_NAME.c_str(), "InplaceAddRmsNorm input is not right.");
        return false;
    }
    return true;
}

bool CollectSubgraphInputsInfo(const std::vector<ge::fusion::SubgraphInput> &subGraphInputs,
                               std::vector<ge::Shape> &inputShapes, std::vector<ge::DataType> &inputDTypes,
                               std::vector<ge::Format> &inputFormats)
{
    inputShapes.clear();
    inputDTypes.clear();
    inputFormats.clear();
    for (const auto &subGraphInput : subGraphInputs) {
        auto matchNodes = subGraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_E(PASS_NAME.c_str(), "CollectSubgraphInputsInfo: matchNodes is empty.");
            return false;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "CollectSubgraphInputsInfo: GetInputDesc failed.");
            return false;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
        inputDTypes.emplace_back(tmpDesc.GetDataType());
        inputFormats.emplace_back(tmpDesc.GetOriginFormat());
    }
    return true;
}

ge::es::EsTensorHolder CreateOneInput(ge::es::EsGraphBuilder &builder, int64_t &inputIdx, const char *name,
                                      const std::vector<ge::Shape> &shapes, const std::vector<ge::DataType> &dtypes,
                                      const std::vector<ge::Format> &formats)
{
    auto holder =
        builder.CreateInput(inputIdx, name, dtypes[inputIdx], formats[inputIdx], shapes[inputIdx].GetDims());
    ++inputIdx;
    return holder;
}

bool CreateReplaceGraphInputs(ReplaceGraphInputs &inputs, ge::es::EsGraphBuilder &replaceGraphBuilder,
                              const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                              const OriginalGraphInfo &info)
{
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    if (!CollectSubgraphInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats)) {
        return false;
    }
    if (inputShapes.size() != GetPatternInputNum(info)) {
        OPS_LOG_E(PASS_NAME.c_str(), "Replace input size=%zu, expect=%zu.", inputShapes.size(),
                  GetPatternInputNum(info));
        return false;
    }

    int64_t inputIdx = 0;
    inputs.rExpandX =
        CreateOneInput(replaceGraphBuilder, inputIdx, "expand_x", inputShapes, inputDTypes, inputFormats);
    inputs.rExpertIds =
        CreateOneInput(replaceGraphBuilder, inputIdx, "expert_ids", inputShapes, inputDTypes, inputFormats);
    inputs.rAssist = CreateOneInput(replaceGraphBuilder, inputIdx, "assist_info_for_combine", inputShapes, inputDTypes,
                                    inputFormats);
    inputs.rEpSend =
        CreateOneInput(replaceGraphBuilder, inputIdx, "ep_send_counts", inputShapes, inputDTypes, inputFormats);
    inputs.rExpertScales =
        CreateOneInput(replaceGraphBuilder, inputIdx, "expert_scales", inputShapes, inputDTypes, inputFormats);
    inputs.rResidualX =
        CreateOneInput(replaceGraphBuilder, inputIdx, "residual_x", inputShapes, inputDTypes, inputFormats);
    inputs.rGamma = CreateOneInput(replaceGraphBuilder, inputIdx, "gamma", inputShapes, inputDTypes, inputFormats);

    inputs.rTpSend = nullptr;
    if (info.hasTpSend) {
        inputs.rTpSend =
            CreateOneInput(replaceGraphBuilder, inputIdx, "tp_send_counts", inputShapes, inputDTypes, inputFormats);
    }
    inputs.rActiveMask = nullptr;
    if (info.hasActiveMask) {
        inputs.rActiveMask =
            CreateOneInput(replaceGraphBuilder, inputIdx, "x_active_mask", inputShapes, inputDTypes, inputFormats);
    }
    inputs.rActScale = nullptr;
    if (info.hasActScale) {
        inputs.rActScale =
            CreateOneInput(replaceGraphBuilder, inputIdx, "activation_scale", inputShapes, inputDTypes, inputFormats);
    }
    inputs.rWeightScale = nullptr;
    if (info.hasWeightScale) {
        inputs.rWeightScale =
            CreateOneInput(replaceGraphBuilder, inputIdx, "weight_scale", inputShapes, inputDTypes, inputFormats);
    }
    inputs.rGroupList = nullptr;
    if (info.hasGroupList) {
        inputs.rGroupList =
            CreateOneInput(replaceGraphBuilder, inputIdx, "group_list", inputShapes, inputDTypes, inputFormats);
    }
    inputs.rExpandScales = nullptr;
    if (info.hasExpandScales) {
        inputs.rExpandScales =
            CreateOneInput(replaceGraphBuilder, inputIdx, "expand_scales", inputShapes, inputDTypes, inputFormats);
    }
    // canndev：Add.x2 → fusion.shared_expert_x
    inputs.rSharedExpertX =
        CreateOneInput(replaceGraphBuilder, inputIdx, "shared_expert_x", inputShapes, inputDTypes, inputFormats);
    return true;
}

ge::fusion::GraphUniqPtr BuildReplaceGraph(const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                           const std::unique_ptr<ge::fusion::MatchResult> &matchResult,
                                           const OriginalGraphInfo &info)
{
    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");
    ReplaceGraphInputs inputTensors;
    if (!CreateReplaceGraphInputs(inputTensors, replaceGraphBuilder, subgraphInputs, info)) {
        OPS_LOG_E(PASS_NAME.c_str(), "CreateReplaceGraphInputs failed.");
        return nullptr;
    }

    ge::GNode combineNode;
    if (!GetCapturedNode(matchResult, COMBINE_CAPTURE_IDX, combineNode)) {
        return nullptr;
    }

    ge::AscendString groupEp;
    OP_LOGE_IF(combineNode.GetAttr("group_ep", groupEp) != ge::GRAPH_SUCCESS, nullptr, PASS_NAME.c_str(),
               "Get Attr group_ep failed.");
    int64_t epWorldSize = 0;
    OP_LOGE_IF(combineNode.GetAttr("ep_world_size", epWorldSize) != ge::GRAPH_SUCCESS, nullptr, PASS_NAME.c_str(),
               "Get Attr ep_world_size failed.");
    int64_t epRankId = 0;
    OP_LOGE_IF(combineNode.GetAttr("ep_rank_id", epRankId) != ge::GRAPH_SUCCESS, nullptr, PASS_NAME.c_str(),
               "Get Attr ep_rank_id failed.");
    int64_t moeExpertNum = 0;
    OP_LOGE_IF(combineNode.GetAttr("moe_expert_num", moeExpertNum) != ge::GRAPH_SUCCESS, nullptr, PASS_NAME.c_str(),
               "Get Attr moe_expert_num failed.");

    ge::AscendString groupTp("");
    (void)combineNode.GetAttr("group_tp", groupTp);
    int64_t tpWorldSize = 0;
    (void)combineNode.GetAttr("tp_world_size", tpWorldSize);
    int64_t tpRankId = 0;
    (void)combineNode.GetAttr("tp_rank_id", tpRankId);
    int64_t expertShardType = 0;
    (void)combineNode.GetAttr("expert_shard_type", expertShardType);
    int64_t sharedExpertNum = 1;
    (void)combineNode.GetAttr("shared_expert_num", sharedExpertNum);
    int64_t sharedExpertRankNum = 0;
    (void)combineNode.GetAttr("shared_expert_rank_num", sharedExpertRankNum);
    int64_t globalBs = 0;
    (void)combineNode.GetAttr("global_bs", globalBs);
    int64_t outDtype = 0;
    (void)combineNode.GetAttr("out_dtype", outDtype);
    int64_t commQuantMode = 0;
    (void)combineNode.GetAttr("comm_quant_mode", commQuantMode);
    int64_t groupListType = 0;
    (void)combineNode.GetAttr("group_list_type", groupListType);
    // 对齐 canndev AddAttr：不拷 comm_alg / zero|copy|const_expert_num / epsilon→norm_eps（用 IR 默认）

    auto out = ge::es::MoeDistributeCombineAddRmsNorm(
        inputTensors.rExpandX, inputTensors.rExpertIds, inputTensors.rAssist, inputTensors.rEpSend,
        inputTensors.rExpertScales, inputTensors.rResidualX, inputTensors.rGamma, inputTensors.rTpSend,
        inputTensors.rActiveMask, inputTensors.rActScale, inputTensors.rWeightScale, inputTensors.rGroupList,
        inputTensors.rExpandScales, inputTensors.rSharedExpertX, nullptr, nullptr, nullptr, nullptr, nullptr,
        groupEp.GetString(), epWorldSize, epRankId, moeExpertNum, groupTp.GetString(), tpWorldSize, tpRankId,
        expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType);
    return replaceGraphBuilder.BuildAndReset({out.y, out.rstd, out.x});
}

bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr &replaceGraph,
                            const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    std::vector<ge::Shape> inputShapes;
    for (const auto &subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_D(PASS_NAME.c_str(), "InferShapeReplaceGraph: matchNodes is empty.");
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_D(PASS_NAME.c_str(), "GetInputDesc failed.");
            continue;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
    }
    if (ge::GeUtils::InferShape(*replaceGraph, inputShapes) != ge::SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "GeUtils::InferShape failed for replace graph.");
        return false;
    }
    return true;
}
} // namespace

std::vector<ge::fusion::PatternUniqPtr> MoeDistributeCombineAddRmsNormFusionPass::Patterns()
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Patterns for MoeDistributeCombineAddRmsNormFusionPass");
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    // 对齐 canndev Relink 会迁的 6 个 Combine 可选口
    for (bool hasTpSend : {false, true}) {
        for (bool hasActiveMask : {false, true}) {
            for (bool hasActScale : {false, true}) {
                for (bool hasWeightScale : {false, true}) {
                    for (bool hasGroupList : {false, true}) {
                        for (bool hasExpandScales : {false, true}) {
                            OriginalGraphInfo info{hasTpSend, hasActiveMask, hasActScale,
                                                   hasWeightScale, hasGroupList, hasExpandScales};
                            patternGraphs.emplace_back(MakePattern(info));
                        }
                    }
                }
            }
        }
    }
    return patternGraphs;
}

bool MoeDistributeCombineAddRmsNormFusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter MeetRequirements for MoeDistributeCombineAddRmsNormFusionPass");

    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_D(PASS_NAME.c_str(), "Skip in MeetRequirements when cann version not compatible.");
        return false;
    }

    // canndev 本 pass 无平台门禁

    std::string patternNameStr;
    if (!GetPatternNameStr(matchResult, patternNameStr)) {
        return false;
    }
    const OriginalGraphInfo info = ParsePatternInfo(patternNameStr);
    const size_t expectNum = GetPatternInputNum(info);

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "Get subgraph inputs failed in MeetRequirements.");
        return false;
    }
    if (subgraphInputs.size() != expectNum) {
        OPS_LOG_D(PASS_NAME.c_str(), "Skip pattern=%s: subgraph size=%zu, expect=%zu.", patternNameStr.c_str(),
                  subgraphInputs.size(), expectNum);
        return false;
    }

    ge::GNode combineNode;
    ge::GNode addNode;
    ge::GNode arnNode;
    if (!GetCapturedNode(matchResult, COMBINE_CAPTURE_IDX, combineNode) ||
        !GetCapturedNode(matchResult, ADD_CAPTURE_IDX, addNode) ||
        !GetCapturedNode(matchResult, ARN_CAPTURE_IDX, arnNode)) {
        return false;
    }

    if (!CheckNodeShape(combineNode, addNode, arnNode) || !CheckNodeDataType(combineNode, addNode, arnNode) ||
        !CheckNodeDataAnchor(combineNode, addNode, arnNode)) {
        return false;
    }

    OPS_LOG_I(PASS_NAME.c_str(), "Found One pattern that meets requirements");
    return true;
}

ge::fusion::GraphUniqPtr MoeDistributeCombineAddRmsNormFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Replacement for MoeDistributeCombineAddRmsNormFusionPass");
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }

    std::string patternNameStr;
    if (!GetPatternNameStr(matchResult, patternNameStr)) {
        OPS_LOG_E(PASS_NAME.c_str(), "Get pattern graph name failed in Replacement.");
        return nullptr;
    }
    const OriginalGraphInfo info = ParsePatternInfo(patternNameStr);

    ge::fusion::GraphUniqPtr replaceGraph = BuildReplaceGraph(subgraphInputs, matchResult, info);
    if (replaceGraph == nullptr) {
        OPS_LOG_E(PASS_NAME.c_str(), "BuildReplaceGraph failed in Replacement.");
        return nullptr;
    }
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(PASS_NAME.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }
    return std::move(replaceGraph);
}

REG_FUSION_PASS(MoeDistributeCombineAddRmsNormFusionPass).Stage(GetPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION
