/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_to_all_matmul_transpose_a5_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_AlltoAllMatmul.h"
#include "es_TransposeD.h"
#include "es_math_ops.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>

namespace ops {
const std::string FUSION_PASS_NAME = "AllToAllMatmulTransposeA5FusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_TRANSPOSE_D = "TransposeD";
const std::string PATTERN_BIAS = "HasBias";
const int64_t ALL2ALLMATMUL_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_CAPTURE_IDX = 1l;
const int64_t MX_QUANT_MODE = 6;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const size_t INPUT_NUM_NO_BIAS = 4;
const size_t INPUT_NUM_HAS_BIAS = 5;
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
    ge::es::EsTensorHolder rCommScale;
    ge::es::EsTensorHolder rX1Offset;
    ge::es::EsTensorHolder rX2Offset;
};

namespace {
    typedef EsCTensorHolder* (*EsTransposeFunc)(EsCTensorHolder*, EsCTensorHolder*);
    
    EsTransposeFunc GetEsTransposeFunc()
    {
        void* handle = dlopen("libes_math.so", RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "dlopen failed: %s", dlerror());
            return nullptr;
        }
        dlerror();
        auto func = reinterpret_cast<EsTransposeFunc>(dlsym(handle, "EsTranspose"));
        if (dlerror() != nullptr) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "dlsym EsTranspose failed");
            return nullptr;
        }
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "EsTranspose function loaded successfully");
        return func;
    }
    
    ge::es::EsTensorHolder TransposeDL(const ge::es::EsTensorLike& x, const ge::es::EsTensorLike& perm)
    {
        static EsTransposeFunc func = GetEsTransposeFunc();
        auto* builder = ge::es::ResolveBuilder(x, perm);
        auto result = func(x.ToTensorHolder(builder).GetCTensorHolder(),
            perm.ToTensorHolder(builder).GetCTensorHolder());
        return result;
    }
}

static ge::fusion::PatternUniqPtr MakePatternForTranspose(bool hasBias)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MakePatternForTranspose: hasBias=%d", hasBias);
    
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE;
    if (hasBias) {
        patternName += PATTERN_BIAS;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Pattern name: %s", patternName.c_str());
    
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto [x1, x2, bias, x1Scale, x2Scale] = graphBuilder.CreateInputs<5>();
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating Transpose node with perm [1,0]");
    auto transpose = TransposeDL(x2, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    
    std::string group = std::string("");
    int64_t worldSize = 0;
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating AlltoAllMatmul node in pattern (hasBias=%d)", hasBias);
    auto mc2 = ge::es::AlltoAllMatmul(x1, transpose, hasBias ? bias : nullptr, x1Scale, x2Scale, nullptr, nullptr,
                                       nullptr, group.c_str(), worldSize);
    
    auto graph = graphBuilder.BuildAndReset({mc2.y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    
    pattern->CaptureTensor({*mc2.y.GetProducer(), 0});
    pattern->CaptureTensor({*transpose.GetProducer(), 0});
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MakePatternForTranspose completed");
    return pattern;
}

static ge::fusion::PatternUniqPtr MakePatternForTransposeD(bool hasBias)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MakePatternForTransposeD: hasBias=%d", hasBias);
    
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE_D;
    if (hasBias) {
        patternName += PATTERN_BIAS;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Pattern name: %s", patternName.c_str());
    
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto [x1, x2, bias, x1Scale, x2Scale] = graphBuilder.CreateInputs<5>();
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating TransposeD node with perm [1,0]");
    auto transposeD = ge::es::TransposeD(x2, std::vector<int64_t>{1, 0});
    
    std::string group = std::string("");
    int64_t worldSize = 0;
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating AlltoAllMatmul node in pattern (hasBias=%d)", hasBias);
    auto mc2 = ge::es::AlltoAllMatmul(x1, transposeD, hasBias ? bias : nullptr, x1Scale, x2Scale, nullptr, nullptr,
                                       nullptr, group.c_str(), worldSize);
    
    auto graph = graphBuilder.BuildAndReset({mc2.y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    
    pattern->CaptureTensor({*mc2.y.GetProducer(), 0});
    pattern->CaptureTensor({*transposeD.GetProducer(), 0});
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MakePatternForTransposeD completed");
    return pattern;
}

static bool IsMXQuantMode(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(ALL2ALLMATMUL_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get AlltoAllMatmul node failed.");
    ge::GNode mc2Node = mc2NodeIo.node;

    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    if (mc2Node.GetAttr("x1_quant_mode", x1QuantMode) != ge::GRAPH_SUCCESS ||
        mc2Node.GetAttr("x2_quant_mode", x2QuantMode) != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get quant mode Attr failed.");
        return false;
    }
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "IsMXQuantMode: x1QuantMode=%ld, x2QuantMode=%ld, MX_QUANT_MODE=%ld",
              x1QuantMode, x2QuantMode, MX_QUANT_MODE);
    
    return (x1QuantMode == MX_QUANT_MODE && x2QuantMode == MX_QUANT_MODE);
}

static bool GetTransposePerm(const ge::GNode &transposeNode, std::vector<int64_t> &permValue)
{
    ge::AscendString nodeType("");
    transposeNode.GetType(nodeType);
    std::string nodeTypeStr = nodeType.GetString();
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "GetTransposePerm: nodeType=%s", nodeTypeStr.c_str());
    
    if (nodeTypeStr == PATTERN_TRANSPOSE_D) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "TransposeD node: getting perm from attr");
        if (transposeNode.GetAttr("perm", permValue) != ge::GRAPH_SUCCESS) {
            OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get perm Attr from TransposeD node failed.");
            return false;
        }
        std::string permStr;
        for (size_t i = 0; i < permValue.size(); ++i) {
            permStr += std::to_string(permValue[i]) + " ";
        }
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "TransposeD perm: [%s]", permStr.c_str());
        return !permValue.empty();
    }

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

    std::string permStr;
    for (size_t i = 0; i < permValue.size(); ++i) {
        permStr += std::to_string(permValue[i]) + " ";
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Transpose perm from input: [%s]", permStr.c_str());

    return !permValue.empty();
}

static bool IsTransposePermValid(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "IsTransposePermValid: checking transpose perm");
    
    ge::fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(TRANSPOSE_CAPTURE_IDX, transposeOutput) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get Transpose node failed.");
    auto transposeNode = transposeOutput.node;
    
    ge::AscendString capturedNodeName, capturedNodeType;
    transposeNode.GetName(capturedNodeName);
    transposeNode.GetType(capturedNodeType);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Captured Transpose node: name=%s, type=%s, index=%d",
              capturedNodeName.GetString(), capturedNodeType.GetString(), transposeOutput.index);

    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeNode, permValue)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation.");
        return false;
    }

    const size_t permSize = permValue.size();
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Perm size: %zu", permSize);
    
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation size must be 2 or 3, but got %zu.", permSize);
        return false;
    }

    bool valid = (permValue[permSize - PERM_SIZE_ONE] == static_cast<int64_t>(permSize - PERM_SIZE_TWO)) &&
                 (permValue[permSize - PERM_SIZE_TWO] == static_cast<int64_t>(permSize - PERM_SIZE_ONE));
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Perm valid: %d", valid);
    
    return valid;
}

static void GetInputsInfo(const std::vector<ge::fusion::SubgraphInput> &subGraphInputs,
                          std::vector<ge::Shape> &inputShapes, std::vector<ge::DataType> &inputDTypes,
                          std::vector<ge::Format> &inputFormats)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter GetInputsInfo, subGraphInputs size=%zu", subGraphInputs.size());
    
    for (const auto &subGraphInput : subGraphInputs) {
        auto matchNodes = subGraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetInputsInfo: matchNodes is empty.");
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetInputDesc failed.");
            continue;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
        inputDTypes.emplace_back(tmpDesc.GetDataType());
        inputFormats.emplace_back(tmpDesc.GetOriginFormat());
    }
    
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetInputsInfo: collected %zu inputs", inputShapes.size());
}

static bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr &replaceGraph,
                                   const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter InferShapeReplaceGraph");
    std::vector<ge::Shape> inputShapes;
    for (const auto &subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
    }
    bool result = (ge::GeUtils::InferShape(*replaceGraph, inputShapes) == ge::SUCCESS);
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph result: %d", result);
    return result;
}

static bool CreateReplaceGraphInputs(ReplaceGraphInputs &inputs, ge::es::EsGraphBuilder &replaceGraphBuilder,
                                     const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                     const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to create inputs in Replacement.");
    
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats);
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Input shapes count: %zu, INPUT_NUM_NO_BIAS=%zu, INPUT_NUM_HAS_BIAS=%zu",
              inputShapes.size(), INPUT_NUM_NO_BIAS, INPUT_NUM_HAS_BIAS);
    
    if (inputShapes.size() < INPUT_NUM_NO_BIAS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement without bias.");
        return false;
    }
    
    ge::AscendString patternName = "";
    matchResult->GetPatternGraph().GetName(patternName);
    std::string patternNameStr = patternName.GetString();
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Pattern name: %s", patternNameStr.c_str());

    int64_t inputIdx = 0;
    inputs.rX1 = replaceGraphBuilder.CreateInput(inputIdx, "x1", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Created input x1 at index %ld", inputIdx);
    ++inputIdx;
    
    inputs.rX2 = replaceGraphBuilder.CreateInput(inputIdx, "x2", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Created input x2 at index %ld", inputIdx);
    ++inputIdx;
    
    inputs.rBias = nullptr;
    bool hasBias = (patternNameStr.find(PATTERN_BIAS) != std::string::npos);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Has bias: %d", hasBias);
    
    if (hasBias) {
        if (inputShapes.size() < INPUT_NUM_HAS_BIAS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement with bias.");
            return false;
        }
        inputs.rBias = replaceGraphBuilder.CreateInput(inputIdx, "bias", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                       inputShapes[inputIdx].GetDims());
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Created input bias at index %ld", inputIdx);
        ++inputIdx;
    }
    
    inputs.rX1Scale = replaceGraphBuilder.CreateInput(inputIdx, "x1_scale", inputDTypes[inputIdx],
                                                      inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Created input x1_scale at index %ld", inputIdx);
    ++inputIdx;
    
    inputs.rX2Scale = replaceGraphBuilder.CreateInput(inputIdx, "x2_scale", inputDTypes[inputIdx],
                                                      inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Created input x2_scale at index %ld", inputIdx);
    
    inputs.rCommScale = nullptr;
    inputs.rX1Offset = nullptr;
    inputs.rX2Offset = nullptr;
    
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "CreateReplaceGraphInputs completed successfully");
    return true;
}

static ge::fusion::GraphUniqPtr BuildReplaceGraph(const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                                  const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Start to build replaceGraph");
    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");

    ReplaceGraphInputs inputTensors;
    if (!CreateReplaceGraphInputs(inputTensors, replaceGraphBuilder, subgraphInputs, matchResult)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CreateReplaceGraphInputs failed.");
        return nullptr;
    }

    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(ALL2ALLMATMUL_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Capture AlltoAllMatmul node failed.");
    ge::GNode mc2Node = mc2NodeIo.node;
    
    ge::AscendString originalNodeName, originalNodeType;
    mc2Node.GetName(originalNodeName);
    mc2Node.GetType(originalNodeType);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original AlltoAllMatmul node: name=%s, type=%s",
              originalNodeName.GetString(), originalNodeType.GetString());

    ge::AscendString group = "";
    OP_LOGE_IF(mc2Node.GetAttr("group", group) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original group: %s", group.GetString());

    int64_t worldSize = -1;
    OP_LOGE_IF(mc2Node.GetAttr("world_size", worldSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr world_size failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original world_size: %ld", worldSize);

    std::vector<int64_t> all2allAxes = {-2, -1};
    OP_LOGE_IF(mc2Node.GetAttr("all2all_axes", all2allAxes) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr all2all_axes failed.");
    std::string axesStr;
    for (auto ax : all2allAxes) axesStr += std::to_string(ax) + " ";
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original all2all_axes: [%s]", axesStr.c_str());

    int64_t yDType = 28;
    OP_LOGE_IF(mc2Node.GetAttr("y_dtype", yDType) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr y_dtype failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original y_dtype: %ld", yDType);

    int64_t x1QuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("x1_quant_mode", x1QuantMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr x1_quant_mode failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original x1_quant_mode: %ld", x1QuantMode);

    int64_t x2QuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("x2_quant_mode", x2QuantMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr x2_quant_mode failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original x2_quant_mode: %ld", x2QuantMode);

    int64_t commQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("comm_quant_mode", commQuantMode) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr comm_quant_mode failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original comm_quant_mode: %ld", commQuantMode);

    int64_t x1QuantDType = 28;
    OP_LOGE_IF(mc2Node.GetAttr("x1_quant_dtype", x1QuantDType) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr x1_quant_dtype failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original x1_quant_dtype: %ld", x1QuantDType);

    int64_t commQuantDType = 28;
    OP_LOGE_IF(mc2Node.GetAttr("comm_quant_dtype", commQuantDType) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr comm_quant_dtype failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original comm_quant_dtype: %ld", commQuantDType);

    bool transposeX1 = false;
    OP_LOGE_IF(mc2Node.GetAttr("transpose_x1", transposeX1) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr transpose_x1 failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original transpose_x1: %d", transposeX1);

    bool transposeX2 = false;
    auto getAttrRet = mc2Node.GetAttr("transpose_x2", transposeX2);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "GetAttr transpose_x2 ret=%d, value=%d", getAttrRet, transposeX2);
    
    if (getAttrRet != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(FUSION_PASS_NAME.c_str(), "transpose_x2 attr not found, using default false");
        transposeX2 = false;
    }
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Before negation: transposeX2=%d", transposeX2);
    transposeX2 = !transposeX2;
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "After negation (will set to new node): transposeX2=%d", transposeX2);

    int64_t groupSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("group_size", groupSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group_size failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original group_size: %ld", groupSize);

    bool alltoallOutFlag = true;
    OP_LOGE_IF(mc2Node.GetAttr("alltoall_out_flag", alltoallOutFlag) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr alltoall_out_flag failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Original alltoall_out_flag: %d", alltoallOutFlag);

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating new AlltoAllMatmul node with transposeX2=%d", transposeX2);
    
    auto mc2 =
        ge::es::AlltoAllMatmul(inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, inputTensors.rX1Scale,
                               inputTensors.rX2Scale, inputTensors.rCommScale, inputTensors.rX1Offset,
                               inputTensors.rX2Offset, group.GetString(), worldSize, all2allAxes, yDType, x1QuantMode,
                               x2QuantMode, commQuantMode, x1QuantDType, commQuantDType, transposeX1, transposeX2,
                               groupSize, alltoallOutFlag);
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "New AlltoAllMatmul node created successfully");
    
    return replaceGraphBuilder.BuildAndReset({mc2.y});
}

std::vector<ge::fusion::PatternUniqPtr> AllToAllMatmulTransposeA5FusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns for AllToAllMatmulTransposeA5FusionPass");
    
    bool hasBias = true;
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating patterns for Transpose (hasBias=true)");
    patternGraphs.emplace_back(MakePatternForTranspose(hasBias));
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating patterns for Transpose (hasBias=false)");
    patternGraphs.emplace_back(MakePatternForTranspose(!hasBias));
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating patterns for TransposeD (hasBias=true)");
    patternGraphs.emplace_back(MakePatternForTransposeD(hasBias));
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating patterns for TransposeD (hasBias=false)");
    patternGraphs.emplace_back(MakePatternForTransposeD(!hasBias));
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Patterns created, total count=%zu", patternGraphs.size());
    
    return patternGraphs;
}

bool AllToAllMatmulTransposeA5FusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements");
    
    ge::AscendString patternGraphName;
    matchResult->GetPatternGraph().GetName(patternGraphName);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Matched pattern graph: %s", patternGraphName.GetString());

    ge::fusion::NodeIo mc2NodeIo;
    auto captureRet = matchResult->GetCapturedTensor(ALL2ALLMATMUL_CAPTURE_IDX, mc2NodeIo);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "GetCapturedTensor for ALL2ALLMATMUL: ret=%d", captureRet);
    
    if (captureRet == ge::SUCCESS) {
        ge::AscendString nodeName, nodeType;
        mc2NodeIo.node.GetName(nodeName);
        mc2NodeIo.node.GetType(nodeType);
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Captured AlltoAllMatmul node: name=%s, type=%s, index=%d",
                  nodeName.GetString(), nodeType.GetString(), mc2NodeIo.index);
    } else {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Failed to capture AlltoAllMatmul node");
        return false;
    }
    
    ge::fusion::NodeIo transposeIo;
    captureRet = matchResult->GetCapturedTensor(TRANSPOSE_CAPTURE_IDX, transposeIo);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "GetCapturedTensor for TRANSPOSE: ret=%d", captureRet);
    
    if (captureRet == ge::SUCCESS) {
        ge::AscendString nodeName, nodeType;
        transposeIo.node.GetName(nodeName);
        transposeIo.node.GetType(nodeType);
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Captured Transpose node: name=%s, type=%s, index=%d",
                  nodeName.GetString(), nodeType.GetString(), transposeIo.index);
    } else {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Failed to capture Transpose node");
        return false;
    }

    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Check target platform fail! Not A5 architecture.");
        return false;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Platform check passed: A5 architecture");

    if (!IsMXQuantMode(matchResult)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Quant mode is not MX QUANT, fusion will be skipped.");
        return false;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MX quant mode check passed");

    if (!IsTransposePermValid(matchResult)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation is not valid.");
        return false;
    }
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found One pattern that meets requirements");
    return true;
}

ge::fusion::GraphUniqPtr
AllToAllMatmulTransposeA5FusionPass::Replacement(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Replacement");
    
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Subgraph input size is %zu", subgraphInputs.size());

    ge::fusion::GraphUniqPtr replaceGraph = BuildReplaceGraph(subgraphInputs, matchResult);
    if (replaceGraph == nullptr) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "BuildReplaceGraph failed in Replacement.");
        return nullptr;
    }
    
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement completed successfully");
    return std::move(replaceGraph);
}

REG_FUSION_PASS(AllToAllMatmulTransposeA5FusionPass)
    .Stage(ge::CustomPassStage::kCompatibleInherited);
} // namespace ops
#endif // GRAPH_FUSION_SUPPORT_VERSION