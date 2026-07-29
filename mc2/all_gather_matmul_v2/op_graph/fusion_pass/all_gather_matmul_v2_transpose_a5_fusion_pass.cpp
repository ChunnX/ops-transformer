/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_matmul_v2_transpose_a5_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_AllGatherMatmulV2.h" // es autogen header
#include "es_TransposeD.h"        // es autogen header for stub op
#include "es_math_ops.h"          // math ops stub
#include "common/utils/op_mc2.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>      // dlopen 动态加载
#include "acl/acl_rt.h" // 运行时判断cann ver

namespace ops {
const std::string FUSION_PASS_NAME = "AllGatherMatmulV2TransposeA5FusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_TRANSPOSE_D = "TransposeD";
const std::string PATTERN_BIAS = "HasBias";
const std::string PATTERN_SCALE = "HasScale";
const std::string PATTERN_X2SCALE_TRANSPOSE = "X2ScaleTranspose";

const int64_t AGMMV2_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_X2_CAPTURE_IDX = 1l;
const int64_t TRANSPOSE_X2SCALE_CAPTURE_IDX = 2l;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const size_t INPUT_NUM_NO_BIAS = 2;
const size_t INPUT_NUM_HAS_BIAS = 3;
const size_t INPUT_NUM_HAS_SCALE = 4; // x1_scale/x2_scale 成对出现；无 bias 至少 4 路
const size_t INPUT_NUM_HAS_BIAS_AND_SCALE = 5;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_ONE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;

struct ReplaceGraphInputs {
    ge::es::EsTensorHolder rX1;
    ge::es::EsTensorHolder rX2;
    ge::es::EsTensorHolder rBias;
    ge::es::EsTensorHolder rX1Scale;
    ge::es::EsTensorHolder rX2Scale;
    ge::es::EsTensorHolder rQuantScale;
};

struct Mc2ReplaceAttrs {
    ge::AscendString group;
    bool isTransA = false;
    bool isTransB = false;
    int64_t gatherIndex = 0;
    int64_t commTurn = 0;
    int64_t rankSize = 0;
    int64_t blockSize = 0;
    int64_t groupSize = 0;
    bool isGatherOut = true;
    bool isAmaxOut = false;
    int64_t yDtype = 0;
    ge::AscendString commMode;
};

// 加载 EsTranspose 符号
namespace {
typedef EsCTensorHolder *(*EsTransposeFunc)(EsCTensorHolder *, EsCTensorHolder *);

EsTransposeFunc GetEsTransposeFunc()
{
    void *handle = dlopen("libes_math.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        OPS_LOG_E("AllGatherMatmulV2TransposeA5FusionPass", "dlopen failed: %s", dlerror());
        return nullptr;
    }
    dlerror();
    auto func = reinterpret_cast<EsTransposeFunc>(dlsym(handle, "EsTranspose"));
    if (dlerror() != nullptr) {
        OPS_LOG_E("AllGatherMatmulV2TransposeA5FusionPass", "dlsym EsTranspose failed");
        return nullptr;
    }
    return func;
}

ge::es::EsTensorHolder TransposeDL(const ge::es::EsTensorLike &x, const ge::es::EsTensorLike &perm)
{
    static EsTransposeFunc func = GetEsTransposeFunc();
    auto *builder = ge::es::ResolveBuilder(x, perm);
    auto xHolder = x.ToTensorHolder(builder);
    if (func == nullptr) {
        OPS_LOG_E("AllGatherMatmulV2TransposeA5FusionPass", "EsTranspose is unavailable, fallback to TransposeD.");
        return ge::es::TransposeD(xHolder, std::vector<int64_t>{1, 0});
    }
    auto result = func(xHolder.GetCTensorHolder(), perm.ToTensorHolder(builder).GetCTensorHolder());
    return result;
}

ge::CustomPassStage GetAllGatherMatmulV2TransposeA5FusionPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}
} // namespace

static ge::fusion::PatternUniqPtr MakePatternForTranspose(bool hasBias, bool hasScale)
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE;
    if (hasBias) {
        patternName += PATTERN_BIAS;
    }
    if (hasScale) {
        patternName += PATTERN_SCALE;
    }
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto [x1, x2, bias, x1Scale, x2Scale] = graphBuilder.CreateInputs<5>();
    auto transpose = TransposeDL(x2, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    const char *group = "";

    auto mc2Out = ge::es::AllGatherMatmulV2(x1, transpose, hasBias ? bias : nullptr, hasScale ? x1Scale : nullptr,
                                            hasScale ? x2Scale : nullptr, nullptr, group);
    auto graph = graphBuilder.BuildAndReset({mc2Out.y, mc2Out.gather_out, mc2Out.amax_out});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2Out.y.GetProducer(), 0}).CaptureTensor({*transpose.GetProducer(), 0});
    return pattern;
}

static ge::fusion::PatternUniqPtr MakePatternForTransposeD(bool hasBias, bool hasScale)
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE_D;
    if (hasBias) {
        patternName += PATTERN_BIAS;
    }
    if (hasScale) {
        patternName += PATTERN_SCALE;
    }
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto [x1, x2, bias, x1Scale, x2Scale] = graphBuilder.CreateInputs<5>();
    auto transposeD = ge::es::TransposeD(x2, std::vector<int64_t>{1, 0});
    const char *group = "";

    auto mc2Out = ge::es::AllGatherMatmulV2(x1, transposeD, hasBias ? bias : nullptr, hasScale ? x1Scale : nullptr,
                                            hasScale ? x2Scale : nullptr, nullptr, group);
    auto graph = graphBuilder.BuildAndReset({mc2Out.y, mc2Out.gather_out, mc2Out.amax_out});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2Out.y.GetProducer(), 0}).CaptureTensor({*transposeD.GetProducer(), 0});
    return pattern;
}

static ge::fusion::PatternUniqPtr MakePatternForTransposeOnX2AndX2Scale(bool hasBias)
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE + PATTERN_X2SCALE_TRANSPOSE + PATTERN_SCALE;
    if (hasBias) {
        patternName += PATTERN_BIAS;
    }
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto [x1, x2, bias, x1Scale, x2Scale] = graphBuilder.CreateInputs<5>();
    auto transposeX2 = TransposeDL(x2, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    auto transposeX2Scale = TransposeDL(x2Scale, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    const char *group = "";

    auto mc2Out = ge::es::AllGatherMatmulV2(x1, transposeX2, hasBias ? bias : nullptr, x1Scale, transposeX2Scale,
                                            nullptr, group);
    auto graph = graphBuilder.BuildAndReset({mc2Out.y, mc2Out.gather_out, mc2Out.amax_out});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2Out.y.GetProducer(), 0})
        .CaptureTensor({*transposeX2.GetProducer(), 0})
        .CaptureTensor({*transposeX2Scale.GetProducer(), 0});
    return pattern;
}

static ge::fusion::PatternUniqPtr MakePatternForTransposeDOnX2AndX2Scale(bool hasBias)
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE_D + PATTERN_X2SCALE_TRANSPOSE + PATTERN_SCALE;
    if (hasBias) {
        patternName += PATTERN_BIAS;
    }
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto [x1, x2, bias, x1Scale, x2Scale] = graphBuilder.CreateInputs<5>();
    auto transposeX2 = ge::es::TransposeD(x2, std::vector<int64_t>{1, 0});
    auto transposeX2Scale = ge::es::TransposeD(x2Scale, std::vector<int64_t>{1, 0});
    const char *group = "";

    auto mc2Out = ge::es::AllGatherMatmulV2(x1, transposeX2, hasBias ? bias : nullptr, x1Scale, transposeX2Scale,
                                            nullptr, group);
    auto graph = graphBuilder.BuildAndReset({mc2Out.y, mc2Out.gather_out, mc2Out.amax_out});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2Out.y.GetProducer(), 0})
        .CaptureTensor({*transposeX2.GetProducer(), 0})
        .CaptureTensor({*transposeX2Scale.GetProducer(), 0});
    return pattern;
}

static bool GetPatternNameStr(const std::unique_ptr<ge::fusion::MatchResult> &matchResult, std::string &patternNameStr)
{
    ge::AscendString patternName = "";
    if (matchResult->GetPatternGraph().GetName(patternName) != ge::SUCCESS) {
        OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get pattern graph name failed.");
        return false;
    }
    patternNameStr = patternName.GetString();
    return true;
}

static bool PatternHasScaleInputs(const std::string &patternNameStr)
{
    return patternNameStr.find(PATTERN_SCALE) != std::string::npos;
}

static bool PatternHasX2ScaleTranspose(const std::string &patternNameStr)
{
    return patternNameStr.find(PATTERN_X2SCALE_TRANSPOSE) != std::string::npos;
}

static size_t GetMinRequiredSubgraphInputNum(const std::string &patternNameStr)
{
    const bool hasBias = patternNameStr.find(PATTERN_BIAS) != std::string::npos;
    if (PatternHasScaleInputs(patternNameStr)) {
        return hasBias ? INPUT_NUM_HAS_BIAS_AND_SCALE : INPUT_NUM_HAS_SCALE;
    }
    return hasBias ? INPUT_NUM_HAS_BIAS : INPUT_NUM_NO_BIAS;
}

static bool GetTransposePerm(const ge::GNode &transposeNode, std::vector<int64_t> &permValue)
{
    ge::AscendString nodeType("");
    transposeNode.GetType(nodeType);
    std::string nodeTypeStr = nodeType.GetString();
    if (nodeTypeStr == PATTERN_TRANSPOSE_D) { // return when node is TransposeD
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Now in TransposeD node for getting perm");
        if (transposeNode.GetAttr("perm", permValue) != ge::GRAPH_SUCCESS) {
            OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get perm Attr from TransposeD node failed.");
            return false;
        }
        return !permValue.empty();
    }

    // Transpose 的场景
    ge::TensorDesc permDesc;
    transposeNode.GetInputDesc(TRANSPOSE_PERM_IDX, permDesc);
    ge::Shape permShape = permDesc.GetShape();
    if (permShape.GetDimNum() != ONE_DIM_SIZE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation dim size must be 1, but got %zu.",
                  permShape.GetDimNum());
        return false;
    }

    ge::Tensor permTensor;
    if (transposeNode.GetInputConstData(TRANSPOSE_PERM_IDX, permTensor) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation const data.");
        return false;
    }

    ge::DataType permDType = permDesc.GetDataType();
    uint8_t *permData = permTensor.GetData();

    if (permData == nullptr) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation data is nullptr.");
        return false;
    }

    size_t size = 0;
    if (permDType == ge::DT_INT32) {
        size = permTensor.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(static_cast<int64_t>(*(reinterpret_cast<int32_t *>(permData) + i)));
        }
    } else if (permDType == ge::DT_INT64) {
        size = permTensor.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(*(reinterpret_cast<int64_t *>(permData) + i));
        }
    } else {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation dtype must be int32 or int64.");
        return false;
    }

    return !permValue.empty();
}

static bool IsX2TransposePermValid(const std::vector<int64_t> &permValue)
{
    const size_t permSize = permValue.size();
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "x2 transpose permutation size must be 2 or 3, but got %zu.", permSize);
        return false;
    }

    if ((permValue[permSize - PERM_SIZE_ONE] != static_cast<int64_t>(permSize - PERM_SIZE_TWO)) ||
        (permValue[permSize - PERM_SIZE_TWO] != static_cast<int64_t>(permSize - PERM_SIZE_ONE))) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "x2 transpose permutation last 2 dims must be [1, 0].");
        return false;
    }
    return true;
}

static bool IsX2ScaleTransposePermValid(const std::vector<int64_t> &permValue)
{
    if (permValue.size() < PERM_SIZE_TWO) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "x2_scale transpose permutation size must be at least 2, but got %zu.",
                  permValue.size());
        return false;
    }

    if (permValue[0] != 1 || permValue[1] != 0) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "x2_scale transpose permutation first 2 dims must be [1, 0].");
        return false;
    }
    return true;
}

static bool GetCapturedMc2Node(const std::unique_ptr<ge::fusion::MatchResult> &matchResult, ge::GNode &mc2Node)
{
    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(AGMMV2_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Capture AllGatherMatmulV2 node failed.");
    mc2Node = mc2NodeIo.node;
    return true;
}

static bool IsX2ScaleDirectConnectAllowed(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    std::string patternNameStr;
    if (!GetPatternNameStr(matchResult, patternNameStr)) {
        return false;
    }
    if (!PatternHasScaleInputs(patternNameStr) || PatternHasX2ScaleTranspose(patternNameStr)) {
        return true;
    }

    ge::GNode mc2Node;
    if (!GetCapturedMc2Node(matchResult, mc2Node)) {
        return false;
    }

    const int32_t x2ScaleInputIdx = static_cast<int32_t>(ops::MC2V2InputIdx::K_X2SCALE);
    ge::TensorDesc x2ScaleDesc;
    if (mc2Node.GetInputDesc(x2ScaleInputIdx, x2ScaleDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Get x2_scale input desc failed.");
        return false;
    }

    const size_t x2ScaleDimNum = x2ScaleDesc.GetShape().GetDimNum();
    if (x2ScaleDimNum > ONE_DIM_SIZE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "x2_scale dim num is %zu (>1) without x2_scale Transpose, fusion is skipped.", x2ScaleDimNum);
        return false;
    }
    return true;
}

static bool ValidateCapturedTransposePerm(const std::unique_ptr<ge::fusion::MatchResult> &matchResult,
                                          int64_t captureIdx, bool isX2ScaleTranspose)
{
    ge::fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(captureIdx, transposeOutput) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get captured Transpose node failed, capture idx %ld.", captureIdx);

    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeOutput.node, permValue)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation, capture idx %ld.", captureIdx);
        return false;
    }

    return isX2ScaleTranspose ? IsX2ScaleTransposePermValid(permValue) : IsX2TransposePermValid(permValue);
}

static bool IsTransposePermValid(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    std::string patternNameStr;
    if (!GetPatternNameStr(matchResult, patternNameStr)) {
        return false;
    }

    if (!ValidateCapturedTransposePerm(matchResult, TRANSPOSE_X2_CAPTURE_IDX, false)) {
        return false;
    }

    if (PatternHasX2ScaleTranspose(patternNameStr)) {
        return ValidateCapturedTransposePerm(matchResult, TRANSPOSE_X2SCALE_CAPTURE_IDX, true);
    }
    return true;
}

static bool CollectSubgraphInputsInfo(const std::vector<ge::fusion::SubgraphInput> &subGraphInputs,
                                      std::vector<ge::Shape> &inputShapes, std::vector<ge::DataType> &inputDTypes,
                                      std::vector<ge::Format> &inputFormats)
{
    inputShapes.clear();
    inputDTypes.clear();
    inputFormats.clear();
    for (const auto &subGraphInput : subGraphInputs) {
        auto matchNodes = subGraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CollectSubgraphInputsInfo: matchNodes is empty.");
            return false;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CollectSubgraphInputsInfo: GetInputDesc failed.");
            return false;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
        inputDTypes.emplace_back(tmpDesc.GetDataType());
        inputFormats.emplace_back(tmpDesc.GetOriginFormat());
    }
    return true;
}

static bool ValidateSubgraphInputNum(const std::string &patternNameStr, size_t inputNum)
{
    const size_t minRequiredInputs = GetMinRequiredSubgraphInputNum(patternNameStr);
    if (inputNum < minRequiredInputs) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(),
                  "Subgraph input nums do not meet requirement, expect at least %zu but got %zu.", minRequiredInputs,
                  inputNum);
        return false;
    }
    return true;
}

static bool CreateReplaceGraphInputs(ReplaceGraphInputs &inputs, ge::es::EsGraphBuilder &replaceGraphBuilder,
                                     const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                     const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to create inputs in Replacement.");
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    if (!CollectSubgraphInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CollectSubgraphInputsInfo failed in CreateReplaceGraphInputs.");
        return false;
    }

    ge::AscendString patternName = "";
    matchResult->GetPatternGraph().GetName(patternName);
    std::string patternNameStr = patternName.GetString();
    if (!ValidateSubgraphInputNum(patternNameStr, inputShapes.size())) {
        return false;
    }

    const bool hasScaleInputs = PatternHasScaleInputs(patternNameStr);

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

    inputs.rX1Scale = nullptr;
    inputs.rX2Scale = nullptr;
    if (hasScaleInputs) {
        inputs.rX1Scale = replaceGraphBuilder.CreateInput(inputIdx, "x1_scale", inputDTypes[inputIdx],
                                                          inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
        inputs.rX2Scale = replaceGraphBuilder.CreateInput(inputIdx, "x2_scale", inputDTypes[inputIdx],
                                                          inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    }
    inputs.rQuantScale = nullptr;
    return true;
}

static bool GetMc2ReplaceAttrs(const ge::GNode &mc2Node, Mc2ReplaceAttrs &attrs)
{
    OP_LOGE_IF(mc2Node.GetAttr("group", attrs.group) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr group failed.");
    OP_LOGE_IF(mc2Node.GetAttr("is_trans_a", attrs.isTransA) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr is_trans_a failed.");
    OP_LOGE_IF(mc2Node.GetAttr("is_trans_b", attrs.isTransB) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr is_trans_b failed.");
    // 移除 Transpose 节点后，is_trans_b 置反
    attrs.isTransB = !attrs.isTransB;

    OP_LOGE_IF(mc2Node.GetAttr("gather_index", attrs.gatherIndex) != ge::GRAPH_SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get Attr gather_index failed.");
    OP_LOGE_IF(mc2Node.GetAttr("comm_turn", attrs.commTurn) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_turn failed.");
    OP_LOGE_IF(mc2Node.GetAttr("rank_size", attrs.rankSize) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr rank_size failed.");
    OP_LOGE_IF(mc2Node.GetAttr("block_size", attrs.blockSize) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr block_size failed.");
    OP_LOGE_IF(mc2Node.GetAttr("group_size", attrs.groupSize) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr group_size failed.");
    OP_LOGE_IF(mc2Node.GetAttr("is_gather_out", attrs.isGatherOut) != ge::GRAPH_SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get Attr is_gather_out failed.");
    OP_LOGE_IF(mc2Node.GetAttr("is_amax_out", attrs.isAmaxOut) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr is_amax_out failed.");
    OP_LOGE_IF(mc2Node.GetAttr("y_dtype", attrs.yDtype) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr y_dtype failed.");
    OP_LOGE_IF(mc2Node.GetAttr("comm_mode", attrs.commMode) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_mode failed.");
    return true;
}

static ge::fusion::GraphUniqPtr
BuildReplaceGraph(const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                  const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to build replaceGraph EsGraphBuilder.");
    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");

    ReplaceGraphInputs inputTensors;
    if (!CreateReplaceGraphInputs(inputTensors, replaceGraphBuilder, subgraphInputs, matchResult)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CreateReplaceGraphInputs failed.");
        return nullptr;
    }

    ge::GNode mc2Node;
    if (!GetCapturedMc2Node(matchResult, mc2Node)) {
        return nullptr;
    }

    Mc2ReplaceAttrs attrs;
    if (!GetMc2ReplaceAttrs(mc2Node, attrs)) {
        return nullptr;
    }

    auto mc2Out = ge::es::AllGatherMatmulV2(
        inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, inputTensors.rX1Scale, inputTensors.rX2Scale,
        inputTensors.rQuantScale, attrs.group.GetString(), attrs.isTransA, attrs.isTransB, attrs.gatherIndex,
        attrs.commTurn, attrs.rankSize, attrs.blockSize, attrs.groupSize, attrs.isGatherOut, attrs.isAmaxOut,
        attrs.yDtype, attrs.commMode.GetString());

    return replaceGraphBuilder.BuildAndReset({mc2Out.y, mc2Out.gather_out, mc2Out.amax_out});
}

static bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr &replaceGraph,
                                   const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                   const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter InferShapeReplaceGraph for AllGatherMatmulV2TransposeA5FusionPass");
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    if (!CollectSubgraphInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CollectSubgraphInputsInfo failed in InferShapeReplaceGraph.");
        return false;
    }

    std::string patternNameStr;
    if (!GetPatternNameStr(matchResult, patternNameStr)) {
        return false;
    }
    if (!ValidateSubgraphInputNum(patternNameStr, inputShapes.size())) {
        return false;
    }

    return (ge::GeUtils::InferShape(*replaceGraph, inputShapes) == ge::SUCCESS);
}

std::vector<ge::fusion::PatternUniqPtr> AllGatherMatmulV2TransposeA5FusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns for AllGatherMatmulV2TransposeA5FusionPass");
    bool hasBias = true;
    bool hasScale = true;
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    patternGraphs.emplace_back(MakePatternForTranspose(hasBias, hasScale));
    patternGraphs.emplace_back(MakePatternForTranspose(!hasBias, hasScale));
    patternGraphs.emplace_back(MakePatternForTransposeD(hasBias, hasScale));
    patternGraphs.emplace_back(MakePatternForTransposeD(!hasBias, hasScale));
    patternGraphs.emplace_back(MakePatternForTranspose(hasBias, !hasScale));
    patternGraphs.emplace_back(MakePatternForTranspose(!hasBias, !hasScale));
    patternGraphs.emplace_back(MakePatternForTransposeD(hasBias, !hasScale));
    patternGraphs.emplace_back(MakePatternForTransposeD(!hasBias, !hasScale));
    patternGraphs.emplace_back(MakePatternForTransposeOnX2AndX2Scale(hasBias));
    patternGraphs.emplace_back(MakePatternForTransposeOnX2AndX2Scale(!hasBias));
    patternGraphs.emplace_back(MakePatternForTransposeDOnX2AndX2Scale(hasBias));
    patternGraphs.emplace_back(MakePatternForTransposeDOnX2AndX2Scale(!hasBias));
    return patternGraphs;
}

bool AllGatherMatmulV2TransposeA5FusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for AllGatherMatmulV2TransposeA5FusionPass");

    // 9.0.0 版本前运行降级stage 空跑
    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip in MeetRequirements when cann version not compatible.");
        return false;
    }

    // 是否ascend950
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Check target platform fail!");
        return false;
    }

    // perm 检查
    if (!IsTransposePermValid(matchResult)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation is not valid.");
        return false;
    }

    // x2_scale 维数 > 1 时必须有 x2_scale Transpose（对齐 canndev）
    if (!IsX2ScaleDirectConnectAllowed(matchResult)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "x2_scale direct connect check failed.");
        return false;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found One pattern that meets requirements");
    return true;
}

ge::fusion::GraphUniqPtr AllGatherMatmulV2TransposeA5FusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement for AllGatherMatmulV2TransposeA5FusionPass");
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    size_t subgraphInputSize = subgraphInputs.size();
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Subgraph input size is %zu.", subgraphInputSize);

    // build replace graph after removing Transpose nodes
    ge::fusion::GraphUniqPtr replaceGraph = BuildReplaceGraph(subgraphInputs, matchResult);
    if (replaceGraph == nullptr) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "BuildReplaceGraph failed in Replacement.");
        return nullptr;
    }
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs, matchResult)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }
    return std::move(replaceGraph);
}

REG_FUSION_PASS(AllGatherMatmulV2TransposeA5FusionPass)
    .Stage(GetAllGatherMatmulV2TransposeA5FusionPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION
