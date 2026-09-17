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
 * \file rotary_position_embedding_fusion_pass.cpp
 * \brief RotaryMul --> RotaryPositionEmbedding(mode = 0), migrated from canndev built-in rope_fusion_pass.
 */

#include "rotary_position_embedding_fusion_pass.h"

#if CANN_VERSION_NUM >= ROTARY_POSITION_EMBEDDING_GRAPH_FUSION_SUPPORT_VERSION
#include <string>
#include <vector>

#include "es_RotaryPositionEmbedding.h" // es autogen header
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace ops {
namespace {
const std::string kPassName = "RotaryMul2RotaryPositionEmbeddingFusionPass";
const std::string kSrcOpType = "RotaryMul";
constexpr int64_t kCaptureTensorIdx = 0L;
constexpr int64_t kRotaryMulMode = 0L;
constexpr size_t kInputNum = 3;
constexpr size_t kXIdx = 0;
constexpr size_t kCosIdx = 1;
constexpr size_t kSinIdx = 2;

// Same gating as the built-in pass: fuse only on regbase platforms (NpuArch DAV_3510/DAV_5102).
// NpuArch 3510 maps to short_soc_version Ascend950/Ascend350, 5102 maps to MC62CM12A.
bool IsRegBasePlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        ge::SUCCESS) {
        OP_LOGW(kPassName, "Get platform info failed.");
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    OP_LOGD(kPassName, "Platform short soc version: %s.", soc.c_str());
    return soc == "Ascend950" || soc == "Ascend350" || soc == "MC62CM12A";
}

void GetInputsInfo(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs, std::vector<ge::Shape>& inputShapes,
                   std::vector<ge::DataType>& inputDtypes, std::vector<ge::Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OP_LOGD(kPassName, "GetInputsInfo: match nodes is empty.");
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tensorDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tensorDesc) != ge::GRAPH_SUCCESS) {
            OP_LOGD(kPassName, "GetInputsInfo: get input desc failed.");
            continue;
        }
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

ge::Status InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr& replaceGraph,
                                  const std::vector<ge::fusion::SubgraphInput>& subgraphInputs)
{
    std::vector<ge::Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tensorDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tensorDesc) != ge::GRAPH_SUCCESS) {
            OP_LOGD(kPassName, "InferShapeReplaceGraph: get input desc failed.");
            continue;
        }
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return ge::GeUtils::InferShape(*replaceGraph, inputShapes);
}
} // namespace

std::vector<ge::fusion::PatternUniqPtr> RotaryMul2RotaryPositionEmbeddingFusionPass::Patterns()
{
    OP_LOGD(kPassName, "Enter Patterns for RotaryMul2RotaryPositionEmbeddingFusionPass.");
    std::vector<ge::fusion::PatternUniqPtr> patterns;

    auto graphBuilder = ge::es::EsGraphBuilder(kPassName.c_str());
    auto inputs = graphBuilder.CreateInputs<kInputNum>();

    // RotaryMul has no ES API, build the pattern node with CompliantNodeBuilder.
    ge::Graph* graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();
    ge::GNode rotaryMulNode = ge::es::CompliantNodeBuilder(graphPtr)
                                  .OpType(kSrcOpType.c_str())
                                  .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                                {"r1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                                {"r2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                  .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                  .Build();
    for (size_t i = 0; i < kInputNum; ++i) {
        ge::GNode inputNode = *inputs[i].GetProducer();
        ge::es::AddEdgeAndUpdatePeerDesc(*graphPtr, inputNode, 0, rotaryMulNode, static_cast<int32_t>(i));
    }
    ge::es::EsTensorHolder y(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(rotaryMulNode, 0));

    auto graph = graphBuilder.BuildAndReset({y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*y.GetProducer(), kCaptureTensorIdx});
    patterns.emplace_back(std::move(pattern));
    return patterns;
}

bool RotaryMul2RotaryPositionEmbeddingFusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OP_LOGD(kPassName, "Enter MeetRequirements for RotaryMul2RotaryPositionEmbeddingFusionPass.");
    (void)matchResult;
    if (!IsRegBasePlatform()) {
        OP_LOGD(kPassName, "RotaryMul can not be fused to RotaryPositionEmbedding on current platform, skip.");
        return false;
    }
    return true;
}

ge::fusion::GraphUniqPtr RotaryMul2RotaryPositionEmbeddingFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OP_LOGD(kPassName, "Enter Replacement for RotaryMul2RotaryPositionEmbeddingFusionPass.");
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OP_LOGE(kPassName, "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    if (subgraphInputs.size() != kInputNum) {
        OP_LOGE(kPassName, "Subgraph input num %zu is not expected %zu.", subgraphInputs.size(), kInputNum);
        return nullptr;
    }

    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDtypes;
    std::vector<ge::Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);
    if (inputShapes.size() != kInputNum) {
        OP_LOGE(kPassName, "Get inputs info failed in Replacement.");
        return nullptr;
    }

    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");
    auto rX = replaceGraphBuilder.CreateInput(kXIdx, "x", inputDtypes[kXIdx], inputFormats[kXIdx],
                                              inputShapes[kXIdx].GetDims());
    auto rCos = replaceGraphBuilder.CreateInput(kCosIdx, "cos", inputDtypes[kCosIdx], inputFormats[kCosIdx],
                                                inputShapes[kCosIdx].GetDims());
    auto rSin = replaceGraphBuilder.CreateInput(kSinIdx, "sin", inputDtypes[kSinIdx], inputFormats[kSinIdx],
                                                inputShapes[kSinIdx].GetDims());

    auto y = ge::es::RotaryPositionEmbedding(rX, rCos, rSin, nullptr, kRotaryMulMode);
    ge::fusion::GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({y});
    if (InferShapeReplaceGraph(replaceGraph, subgraphInputs) != ge::SUCCESS) {
        OP_LOGE(kPassName, "InferShape for replacement graph failed.");
        return nullptr;
    }
    return replaceGraph;
}

REG_FUSION_PASS(RotaryMul2RotaryPositionEmbeddingFusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
} // namespace ops
#endif // CANN_VERSION_NUM >= ROTARY_POSITION_EMBEDDING_GRAPH_FUSION_SUPPORT_VERSION
