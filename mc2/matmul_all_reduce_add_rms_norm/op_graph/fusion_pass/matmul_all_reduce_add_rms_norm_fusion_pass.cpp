/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_all_reduce_add_rms_norm_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_MatmulAllReduce.h"
#include "es_AddRmsNorm.h"
#include "es_InplaceAddRmsNorm.h"
#include "es_MatmulAllReduceAddRmsNorm.h"
#include "es_InplaceMatmulAllReduceAddRmsNorm.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include "acl/acl_rt.h"

namespace ops {
namespace {
const std::string PASS_NAME = "MatmulAllReduceAddRmsNormFusionPass";
const std::string PATTERN_INPLACE = "Inplace";
const std::string PATTERN_MC2_TO_X1 = "Mc2ToX1";
const std::string PATTERN_MC2_TO_X2 = "Mc2ToX2";
const std::string PATTERN_BIAS = "HasBias";
const std::string PATTERN_SCALE = "HasScale";
const std::string PATTERN_OFFSET = "HasOffset";
const std::string PATTERN_DEQUANT = "HasDequant";

const int64_t MC2_CAPTURE_IDX = 0l;
const int64_t ARN_CAPTURE_IDX = 1l;
const int32_t ARN_OUT_RSTD_IDX = 1;
const int32_t MAR_OUT_IDX = 0;

enum class ArnKind {
    AddRmsNorm,
    InplaceAddRmsNorm
};

struct OriginalGraphInfo {
    ArnKind arnKind = ArnKind::AddRmsNorm;
    bool mc2ConnectToX1 = true; // true: MAR→ARN.x1，residual 为 ARN.x2；反之亦然
    bool hasBias = false;
    bool hasScale = false;
    bool hasOffset = false;
    bool hasDequant = false;
};

struct ReplaceGraphInputs {
    ge::es::EsTensorHolder rX1;
    ge::es::EsTensorHolder rX2;
    ge::es::EsTensorHolder rBias;
    ge::es::EsTensorHolder rResidual;
    ge::es::EsTensorHolder rGamma;
    ge::es::EsTensorHolder rAntiquantScale;
    ge::es::EsTensorHolder rAntiquantOffset;
    ge::es::EsTensorHolder rDequantScale;
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
    // x1/x2 + residual + gamma；可选 bias/scale/offset/dequant；x3 永不进 pattern（canndev 有 x3 则拒融）
    size_t n = 4;
    if (info.hasBias) {
        ++n;
    }
    if (info.hasScale) {
        ++n;
    }
    if (info.hasOffset) {
        ++n;
    }
    if (info.hasDequant) {
        ++n;
    }
    return n;
}

ge::fusion::PatternUniqPtr MakePattern(const OriginalGraphInfo &info)
{
    std::string patternName = PASS_NAME;
    if (info.arnKind == ArnKind::InplaceAddRmsNorm) {
        patternName += PATTERN_INPLACE;
    }
    patternName += info.mc2ConnectToX1 ? PATTERN_MC2_TO_X1 : PATTERN_MC2_TO_X2;
    if (info.hasBias) {
        patternName += PATTERN_BIAS;
    }
    if (info.hasScale) {
        patternName += PATTERN_SCALE;
    }
    if (info.hasOffset) {
        patternName += PATTERN_OFFSET;
    }
    if (info.hasDequant) {
        patternName += PATTERN_DEQUANT;
    }

    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto inputs = graphBuilder.CreateInputs(GetPatternInputNum(info));
    size_t idx = 0;
    auto x1 = inputs[idx++];
    auto x2 = inputs[idx++];
    ge::es::EsTensorHolder bias = nullptr;
    if (info.hasBias) {
        bias = inputs[idx++];
    }
    auto residual = inputs[idx++];
    auto gamma = inputs[idx++];
    ge::es::EsTensorHolder antiquantScale = nullptr;
    if (info.hasScale) {
        antiquantScale = inputs[idx++];
    }
    ge::es::EsTensorHolder antiquantOffset = nullptr;
    if (info.hasOffset) {
        antiquantOffset = inputs[idx++];
    }
    ge::es::EsTensorHolder dequantScale = nullptr;
    if (info.hasDequant) {
        dequantScale = inputs[idx++];
    }

    const char *group = "";
    // x3 / pertoken / comm_quant 固定 nullptr，与 canndev 不迁这些边一致
    auto marY = ge::es::MatmulAllReduce(x1, x2, bias, nullptr, antiquantScale, antiquantOffset, dequantScale, nullptr,
                                        nullptr, nullptr, group);

    std::unique_ptr<ge::fusion::Pattern> pattern;
    if (info.arnKind == ArnKind::AddRmsNorm) {
        ge::es::AddRmsNormOutput arn;
        if (info.mc2ConnectToX1) {
            arn = ge::es::AddRmsNorm(marY, residual, gamma);
        } else {
            arn = ge::es::AddRmsNorm(residual, marY, gamma);
        }
        // 输出顺序：ARN.y → fusion.norm_out；ARN.x → fusion.y（对齐 canndev RelinkOutputDataEdge）
        auto graph = graphBuilder.BuildAndReset({arn.y, arn.x});
        pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
        pattern->CaptureTensor({*marY.GetProducer(), 0}).CaptureTensor({*arn.y.GetProducer(), 0});
    } else {
        ge::es::InplaceAddRmsNormOutput arn;
        if (info.mc2ConnectToX1) {
            arn = ge::es::InplaceAddRmsNorm(marY, residual, gamma);
        } else {
            arn = ge::es::InplaceAddRmsNorm(residual, marY, gamma);
        }
        auto graph = graphBuilder.BuildAndReset({arn.ref_x1, arn.ref_x2});
        pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
        pattern->CaptureTensor({*marY.GetProducer(), 0}).CaptureTensor({*arn.ref_x1.GetProducer(), 0});
    }
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

size_t GetExpectedSubgraphInputNum(const std::string &patternNameStr)
{
    OriginalGraphInfo info;
    info.hasBias = patternNameStr.find(PATTERN_BIAS) != std::string::npos;
    info.hasScale = patternNameStr.find(PATTERN_SCALE) != std::string::npos;
    info.hasOffset = patternNameStr.find(PATTERN_OFFSET) != std::string::npos;
    info.hasDequant = patternNameStr.find(PATTERN_DEQUANT) != std::string::npos;
    return GetPatternInputNum(info);
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

bool IsInputConnected(ge::GNode &node, const char *inputName)
{
    int32_t localIdx = -1;
    if (node.GetInputIndexByName(ge::AscendString(inputName), localIdx) != ge::GRAPH_SUCCESS || localIdx < 0) {
        return false;
    }
    auto peer = node.GetInDataNodesAndPortIndexs(localIdx);
    return peer.first != nullptr;
}

bool CheckMc2OutDegree(ge::GNode &mc2Node, ge::GNode &arnNode)
{
    auto consumers = mc2Node.GetOutDataNodesAndPortIndexs(MAR_OUT_IDX);
    if (consumers.size() != 1U) {
        OPS_LOG_D(PASS_NAME.c_str(), "MatmulAllReduce should have only one out data node, but got %zu.",
                  consumers.size());
        return false;
    }
    if (consumers[0].first == nullptr) {
        return false;
    }
    ge::AscendString arnName;
    ge::AscendString consumerName;
    arnNode.GetName(arnName);
    consumers[0].first->GetName(consumerName);
    const char *arnNameStr = arnName.GetString();
    const char *consumerNameStr = consumerName.GetString();
    if (arnNameStr == nullptr || consumerNameStr == nullptr || std::string(arnNameStr) != consumerNameStr) {
        OPS_LOG_D(PASS_NAME.c_str(), "MatmulAllReduce consumer is not AddRmsNorm.");
        return false;
    }
    return true;
}

bool CheckArnRstdUnused(ge::GNode &arnNode)
{
    auto rstdConsumers = arnNode.GetOutDataNodesAndPortIndexs(ARN_OUT_RSTD_IDX);
    if (!rstdConsumers.empty()) {
        OPS_LOG_D(PASS_NAME.c_str(), "AddRmsNorm rstd output is connected, skip fusion.");
        return false;
    }
    return true;
}

bool CheckArnDtype(ge::GNode &arnNode)
{
    // 对齐 canndev CheckAddRmsNormType：输入不可为 FLOAT；输出除 rstd 外不可为 FLOAT
    const size_t inputSize = arnNode.GetInputsSize();
    for (size_t i = 0; i < inputSize; ++i) {
        ge::TensorDesc desc;
        if (arnNode.GetInputDesc(static_cast<int32_t>(i), desc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        if (desc.GetDataType() == ge::DT_FLOAT) {
            OPS_LOG_D(PASS_NAME.c_str(), "AddRmsNorm input %zu is FLOAT, skip fusion.", i);
            return false;
        }
    }
    const size_t outputSize = arnNode.GetOutputsSize();
    for (size_t i = 0; i < outputSize; ++i) {
        if (i == static_cast<size_t>(ARN_OUT_RSTD_IDX)) {
            continue;
        }
        ge::TensorDesc desc;
        if (arnNode.GetOutputDesc(static_cast<int32_t>(i), desc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        if (desc.GetDataType() == ge::DT_FLOAT) {
            OPS_LOG_D(PASS_NAME.c_str(), "AddRmsNorm output %zu is FLOAT, skip fusion.", i);
            return false;
        }
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

bool CreateReplaceGraphInputs(ReplaceGraphInputs &inputs, ge::es::EsGraphBuilder &replaceGraphBuilder,
                              const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                              const std::string &patternNameStr)
{
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    if (!CollectSubgraphInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats)) {
        return false;
    }
    const size_t expectNum = GetExpectedSubgraphInputNum(patternNameStr);
    if (inputShapes.size() != expectNum) {
        OPS_LOG_E(PASS_NAME.c_str(), "Replace input size=%zu, expect=%zu.", inputShapes.size(), expectNum);
        return false;
    }

    int64_t inputIdx = 0;
    inputs.rX1 = replaceGraphBuilder.CreateInput(inputIdx, "x1", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rX2 = replaceGraphBuilder.CreateInput(inputIdx, "x2", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    ++inputIdx;

    inputs.rBias = nullptr;
    if (patternNameStr.find(PATTERN_BIAS) != std::string::npos) {
        inputs.rBias = replaceGraphBuilder.CreateInput(inputIdx, "bias", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                       inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }

    inputs.rResidual = replaceGraphBuilder.CreateInput(inputIdx, "residual", inputDTypes[inputIdx],
                                                       inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rGamma = replaceGraphBuilder.CreateInput(inputIdx, "gamma", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                    inputShapes[inputIdx].GetDims());
    ++inputIdx;

    inputs.rAntiquantScale = nullptr;
    if (patternNameStr.find(PATTERN_SCALE) != std::string::npos) {
        inputs.rAntiquantScale =
            replaceGraphBuilder.CreateInput(inputIdx, "antiquant_scale", inputDTypes[inputIdx],
                                            inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }
    inputs.rAntiquantOffset = nullptr;
    if (patternNameStr.find(PATTERN_OFFSET) != std::string::npos) {
        inputs.rAntiquantOffset =
            replaceGraphBuilder.CreateInput(inputIdx, "antiquant_offset", inputDTypes[inputIdx],
                                            inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }
    inputs.rDequantScale = nullptr;
    if (patternNameStr.find(PATTERN_DEQUANT) != std::string::npos) {
        inputs.rDequantScale =
            replaceGraphBuilder.CreateInput(inputIdx, "dequant_scale", inputDTypes[inputIdx], inputFormats[inputIdx],
                                            inputShapes[inputIdx].GetDims());
    }
    return true;
}

ge::fusion::GraphUniqPtr BuildReplaceGraph(const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                           const std::unique_ptr<ge::fusion::MatchResult> &matchResult,
                                           const std::string &patternNameStr)
{
    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");
    ReplaceGraphInputs inputTensors;
    if (!CreateReplaceGraphInputs(inputTensors, replaceGraphBuilder, subgraphInputs, patternNameStr)) {
        OPS_LOG_E(PASS_NAME.c_str(), "CreateReplaceGraphInputs failed.");
        return nullptr;
    }

    ge::GNode mc2Node;
    ge::GNode arnNode;
    if (!GetCapturedNode(matchResult, MC2_CAPTURE_IDX, mc2Node) ||
        !GetCapturedNode(matchResult, ARN_CAPTURE_IDX, arnNode)) {
        return nullptr;
    }

    ge::AscendString group;
    OP_LOGE_IF(mc2Node.GetAttr("group", group) != ge::GRAPH_SUCCESS, nullptr, PASS_NAME.c_str(),
               "Get Attr group failed.");

    ge::AscendString reduceOp("sum");
    (void)mc2Node.GetAttr("reduce_op", reduceOp);

    bool isTransA = false;
    (void)mc2Node.GetAttr("is_trans_a", isTransA);
    bool isTransB = false;
    (void)mc2Node.GetAttr("is_trans_b", isTransB);

    int64_t commTurn = 0;
    (void)mc2Node.GetAttr("comm_turn", commTurn);
    int64_t antiquantGroupSize = 0;
    (void)mc2Node.GetAttr("antiquant_group_size", antiquantGroupSize);

    float epsilon = 1e-6f;
    (void)arnNode.GetAttr("epsilon", epsilon);

    const bool isInplace = patternNameStr.find(PATTERN_INPLACE) != std::string::npos;
    if (isInplace) {
        auto out = ge::es::InplaceMatmulAllReduceAddRmsNorm(
            inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, inputTensors.rResidual, inputTensors.rGamma,
            inputTensors.rAntiquantScale, inputTensors.rAntiquantOffset, inputTensors.rDequantScale, group.GetString(),
            reduceOp.GetString(), isTransA, isTransB, commTurn, antiquantGroupSize, epsilon);
        // 与 pattern 输出 [ref_x1, ref_x2] 对齐 → [norm_out, ref_residual]
        return replaceGraphBuilder.BuildAndReset({out.norm_out, out.ref_residual});
    }

    auto out = ge::es::MatmulAllReduceAddRmsNorm(
        inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, inputTensors.rResidual, inputTensors.rGamma,
        inputTensors.rAntiquantScale, inputTensors.rAntiquantOffset, inputTensors.rDequantScale, group.GetString(),
        reduceOp.GetString(), isTransA, isTransB, commTurn, antiquantGroupSize, epsilon);
    // 与 pattern 输出 [y, x] 对齐 → [norm_out, y]
    return replaceGraphBuilder.BuildAndReset({out.norm_out, out.y});
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

std::vector<ge::fusion::PatternUniqPtr> MatmulAllReduceAddRmsNormFusionPass::Patterns()
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Patterns for MatmulAllReduceAddRmsNormFusionPass");
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    // ArnKind × Mc2→x1/x2 × bias × scale × offset × dequant
    for (auto arnKind : {ArnKind::AddRmsNorm, ArnKind::InplaceAddRmsNorm}) {
        for (bool mc2ToX1 : {true, false}) {
            for (bool hasBias : {false, true}) {
                for (bool hasScale : {false, true}) {
                    for (bool hasOffset : {false, true}) {
                        for (bool hasDequant : {false, true}) {
                            OriginalGraphInfo info{arnKind, mc2ToX1, hasBias, hasScale, hasOffset, hasDequant};
                            patternGraphs.emplace_back(MakePattern(info));
                        }
                    }
                }
            }
        }
    }
    return patternGraphs;
}

bool MatmulAllReduceAddRmsNormFusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter MeetRequirements for MatmulAllReduceAddRmsNormFusionPass");

    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_D(PASS_NAME.c_str(), "Skip in MeetRequirements when cann version not compatible.");
        return false;
    }

    // 对齐 canndev：仅 DAV_2201（A2/A3）
    if (!IsTargetPlatformNpuArch(PASS_NAME.c_str(), NPUARCH_A2A3)) {
        OPS_LOG_D(PASS_NAME.c_str(), "Check target platform fail!");
        return false;
    }

    std::string patternNameStr;
    if (!GetPatternNameStr(matchResult, patternNameStr)) {
        return false;
    }
    const size_t expectNum = GetExpectedSubgraphInputNum(patternNameStr);

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

    ge::GNode mc2Node;
    ge::GNode arnNode;
    if (!GetCapturedNode(matchResult, MC2_CAPTURE_IDX, mc2Node) ||
        !GetCapturedNode(matchResult, ARN_CAPTURE_IDX, arnNode)) {
        return false;
    }

    // 有 x3 边则不融（对齐 canndev）
    if (IsInputConnected(mc2Node, "x3")) {
        OPS_LOG_D(PASS_NAME.c_str(), "MatmulAllReduce x3 is connected, skip fusion.");
        return false;
    }

    if (!CheckMc2OutDegree(mc2Node, arnNode) || !CheckArnRstdUnused(arnNode) || !CheckArnDtype(arnNode)) {
        return false;
    }

    OPS_LOG_I(PASS_NAME.c_str(), "Found One pattern that meets requirements");
    return true;
}

ge::fusion::GraphUniqPtr MatmulAllReduceAddRmsNormFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Replacement for MatmulAllReduceAddRmsNormFusionPass");
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

    ge::fusion::GraphUniqPtr replaceGraph = BuildReplaceGraph(subgraphInputs, matchResult, patternNameStr);
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

REG_FUSION_PASS(MatmulAllReduceAddRmsNormFusionPass).Stage(GetPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION
