/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allto_allv_quant_grouped_mat_mul_transpose_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_AlltoAllvQuantGroupedMatMul.h"
#include "es_TransposeD.h"
#include "es_math_ops.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>

namespace ops {
const std::string FUSION_PASS_NAME = "AlltoAllvQuantGroupedMatMulTransposeFusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_TRANSPOSE_D = "TransposeD";
const std::string PATTERN_GMM_WEIGHT = "HasGmmWeight";
const std::string PATTERN_MM_WEIGHT = "HasMmWeight";
const std::string PATTERN_GMM_WEIGHT_SCALE = "HasGmmWeightScale";
const std::string PATTERN_MM_WEIGHT_SCALE = "HasMmWeightScale";

const int64_t MC2_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_CAPTURE_IDX = 1l;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;
const size_t PERM_SIZE_FOUR = 4;
const size_t PERM_SIZE_ONE = 1;

static std::string ShapeToStr(const ge::Shape &shape) {
    std::string str = "[";
    auto dims = shape.GetDims();
    for (size_t i = 0; i < dims.size(); ++i) {
        str += std::to_string(dims[i]);
        if (i < dims.size() - 1) str += ", ";
    }
    str += "]";
    return str;
}

enum MC2InputIdx {
    K_GMM_X = 0,
    K_GMM_WEIGHT = 1,
    K_GMM_X_SCALE = 2,
    K_GMM_WEIGHT_SCALE = 3,
    K_SEND_COUNTS_TENSOR = 4,
    K_RECV_COUNTS_TENSOR = 5,
    K_MM_X = 6,
    K_MM_WEIGHT = 7,
    K_MM_X_SCALE = 8,
    K_MM_WEIGHT_SCALE = 9
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

static bool GetTransposePerm(const ge::GNode &transposeNode, std::vector<int64_t> &permValue)
{
    ge::AscendString nodeType("");
    transposeNode.GetType(nodeType);
    std::string nodeTypeStr = nodeType.GetString();
    if (nodeTypeStr == PATTERN_TRANSPOSE_D) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Now in TransposeD node for getting perm");
        if (transposeNode.GetAttr("perm", permValue) != ge::GRAPH_SUCCESS) {
            OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get perm Attr from TransposeD node failed.");
            return false;
        }
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

    return !permValue.empty();
}

static bool IsTransposePermValid(const ge::GNode &transposeNode, uint32_t parentIdx)
{
    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeNode, permValue)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation.");
        return false;
    }

    const size_t permSize = permValue.size();
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE && permSize != PERM_SIZE_FOUR) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation size must be 2/3/4, but got %zu.", permSize);
        return false;
    }

    if (parentIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE)) {
        if ((permValue[1] != static_cast<int64_t>(2)) || (permValue[2] != static_cast<int64_t>(1))) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GmmWeightScale perm dim must be [0,2,1,3].");
            return false;
        }
    } else if (parentIdx == static_cast<uint32_t>(MC2InputIdx::K_MM_WEIGHT_SCALE)) {
        if ((permValue[1] != static_cast<int64_t>(0)) || (permValue[0] != static_cast<int64_t>(1))) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "MmWeightScale perm dim must be [1,0,2].");
            return false;
        }
    } else {
        if ((permValue[permSize - PERM_SIZE_ONE] != static_cast<int64_t>(permSize - PERM_SIZE_TWO)) ||
            (permValue[permSize - PERM_SIZE_TWO] != static_cast<int64_t>(permSize - PERM_SIZE_ONE))) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation last 2 dims must be [1, 0].");
            return false;
        }
    }
    return true;
}

static ge::fusion::PatternUniqPtr MakePatternForTranspose(uint32_t inputIdx, const std::string &patternSuffix)
{
    std::string patternName = FUSION_PASS_NAME + patternSuffix;
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== MakePatternForTranspose: inputIdx=%u, suffix=%s ===", 
              inputIdx, patternSuffix.c_str());
    
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    
    // 参考 MatmulAlltoAll：只创建必选输入 + 有值可选输入的 CreateInputs
    // 原图中 sendCountsTensor/recvCountsTensor 为 nullptr，mm 系列可能为 nullptr 或有值
    // 这里先只创建 gmm 相关的4个必选输入
    auto [gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmX, mmWeight, mmXScale, mmWeightScale] = graphBuilder.CreateInputs<8>();
    
    std::string group = "";
    int64_t epWorldSize = 0;
    std::vector<int64_t> sendCounts = {};
    std::vector<int64_t> recvCounts = {};
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    int64_t mmXQuantMode = 0;
    int64_t mmWeightQuantMode = 0;
    int64_t groupSize = 0;
    int64_t yDtype = 28;
    int64_t mmDtype = 28;
    bool permuteOutFlag = false;
    
    ge::es::EsTensorHolder transposeOutput;
    std::vector<int64_t> permValue;
    
    if (inputIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT)) {
        permValue = {0, 2, 1};
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Pattern: Transpose on gmm_weight, perm=[0,2,1]");
        transposeOutput = TransposeDL(gmmWeight, ge::es::EsTensorLike(permValue));
    } else if (inputIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE)) {
        permValue = {0, 2, 1, 3};
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Pattern: Transpose on gmm_weight_scale, perm=[0,2,1,3]");
        transposeOutput = TransposeDL(gmmWeightScale, ge::es::EsTensorLike(permValue));
    }
    
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX,
        inputIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT) ? transposeOutput : gmmWeight,
        gmmXScale,
        inputIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE) ? transposeOutput : gmmWeightScale,
        nullptr, nullptr,
        mmX, mmWeight, mmXScale, mmWeightScale,
        group.c_str(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode, false, false, permuteOutFlag,
        mmXQuantMode, mmWeightQuantMode, groupSize, yDtype, mmDtype);
    
    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0});
    pattern->CaptureTensor({*transposeOutput.GetProducer(), 0});
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== Pattern created, captured 2 tensors ===");
    return pattern;
}

static ge::fusion::PatternUniqPtr MakePatternForTransposeD(uint32_t inputIdx, const std::string &patternSuffix)
{
    std::string patternName = FUSION_PASS_NAME + patternSuffix;
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== MakePatternForTransposeD: inputIdx=%u, suffix=%s ===", 
              inputIdx, patternSuffix.c_str());
    
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    
    auto [gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmX, mmWeight, mmXScale, mmWeightScale] = graphBuilder.CreateInputs<8>();
    
    std::string group = "";
    int64_t epWorldSize = 0;
    std::vector<int64_t> sendCounts = {};
    std::vector<int64_t> recvCounts = {};
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    int64_t mmXQuantMode = 0;
    int64_t mmWeightQuantMode = 0;
    int64_t groupSize = 0;
    int64_t yDtype = 28;
    int64_t mmDtype = 28;
    bool permuteOutFlag = false;
    
    ge::es::EsTensorHolder transposeOutput;
    
    if (inputIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT)) {
        auto transposeD = ge::es::TransposeD(gmmWeight, std::vector<int64_t>{0, 2, 1});
        transposeOutput = transposeD;
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Pattern: TransposeD on gmm_weight, perm=[0,2,1]");
    } else if (inputIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE)) {
        auto transposeD = ge::es::TransposeD(gmmWeightScale, std::vector<int64_t>{0, 2, 1, 3});
        transposeOutput = transposeD;
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Pattern: TransposeD on gmm_weight_scale, perm=[0,2,1,3]");
    }
    
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX,
        inputIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT) ? transposeOutput : gmmWeight,
        gmmXScale,
        inputIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE) ? transposeOutput : gmmWeightScale,
        nullptr, nullptr,
        mmX, mmWeight, mmXScale, mmWeightScale,
        group.c_str(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode, false, false, permuteOutFlag,
        mmXQuantMode, mmWeightQuantMode, groupSize, yDtype, mmDtype);
    
    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0});
    pattern->CaptureTensor({*transposeOutput.GetProducer(), 0});
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== Pattern created, captured 2 tensors ===");
    return pattern;
}

std::vector<ge::fusion::PatternUniqPtr> AlltoAllvQuantGroupedMatMulTransposeFusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== Enter Patterns for AlltoAllvQuantGroupedMatMulTransposeFusionPass ===");
    
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    
    // 添加 Transpose 类型的 patterns（最多3维：gmm_weight 3维 perm=[0,2,1]，gmm_weight_scale 4维 perm=[0,2,1,3]）
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== Creating Transpose patterns ===");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating Transpose pattern for GMM_WEIGHT (idx=1)");
    patternGraphs.emplace_back(MakePatternForTranspose(static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT), "_Transpose_GmmWeight"));
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating Transpose pattern for GMM_WEIGHT_SCALE (idx=3)");
    patternGraphs.emplace_back(MakePatternForTranspose(static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE), "_Transpose_GmmWeightScale"));
    
    // 添加 TransposeD 类型的 patterns
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== Creating TransposeD patterns ===");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating TransposeD pattern for GMM_WEIGHT (idx=1)");
    patternGraphs.emplace_back(MakePatternForTransposeD(static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT), "_TransposeD_GmmWeight"));
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Creating TransposeD pattern for GMM_WEIGHT_SCALE (idx=3)");
    patternGraphs.emplace_back(MakePatternForTransposeD(static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE), "_TransposeD_GmmWeightScale"));
    
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== Patterns DONE, total %zu patterns ===", patternGraphs.size());
    return patternGraphs;
}

bool AlltoAllvQuantGroupedMatMulTransposeFusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== Enter MeetRequirements ===");

    // 1. 平台检查
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MeetRequirements FAILED: Platform check failed (not A5)");
        return false;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MeetRequirements: Platform check passed (A5)");

    // 2. 获取 MC2 节点
    ge::fusion::NodeIo mc2NodeIo;
    if (matchResult->GetCapturedTensor(MC2_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MeetRequirements FAILED: Get MC2 node failed");
        return false;
    }
    ge::GNode mc2Node = mc2NodeIo.node;
    ge::AscendString mc2NodeName;
    mc2Node.GetName(mc2NodeName);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MeetRequirements: Got MC2 node, name=%s", 
              mc2NodeName.GetString());

    // 3. 获取 TransposeD 节点 (idx=1)
    ge::fusion::NodeIo transposeNodeIo;
    if (matchResult->GetCapturedTensor(TRANSPOSE_CAPTURE_IDX, transposeNodeIo) != ge::SUCCESS) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MeetRequirements FAILED: Get Transpose/TransposeD node failed");
        return false;
    }
    ge::GNode transposeNode = transposeNodeIo.node;
    ge::AscendString transposeNodeName;
    ge::AscendString transposeNodeType;
    transposeNode.GetName(transposeNodeName);
    transposeNode.GetType(transposeNodeType);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MeetRequirements: Got Transpose node, name=%s, type=%s",
              transposeNodeName.GetString(), transposeNodeType.GetString());

    // 4. 获取 Transpose 的 parentIdx（Transpose 连接到 MC2 的哪个输入）
    uint32_t parentIdx = transposeNodeIo.index;
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MeetRequirements: Transpose parentIdx=%u", parentIdx);

    // 5. 验证 perm 值（兼容 Transpose 和 TransposeD）
    if (!IsTransposePermValid(transposeNode, parentIdx)) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MeetRequirements FAILED: IsTransposePermValid check failed");
        return false;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MeetRequirements: IsTransposePermValid check passed");

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== MeetRequirements SUCCESS ===");
    return true;
}

static void GetInputsInfo(const std::vector<ge::fusion::SubgraphInput> &subGraphInputs,
                          std::vector<ge::Shape> &inputShapes, std::vector<ge::DataType> &inputDTypes,
                          std::vector<ge::Format> &inputFormats)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter GetInputsInfo");
    for (const auto &subGraphInput : subGraphInputs) {
        auto matchNodes = subGraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
        inputDTypes.emplace_back(tmpDesc.GetDataType());
        inputFormats.emplace_back(tmpDesc.GetOriginFormat());
    }
}

static bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr &replaceGraph,
                                   const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
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
    return (ge::GeUtils::InferShape(*replaceGraph, inputShapes) == ge::SUCCESS);
}

ge::fusion::GraphUniqPtr AlltoAllvQuantGroupedMatMulTransposeFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== Enter Replacement ===");
    
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed.");
        return nullptr;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: Got %zu subgraph inputs", subgraphInputs.size());

    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: Got %zu input shapes/dtypes", inputShapes.size());

    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");

    int64_t inputIdx = 0;
    auto rGmmX = replaceGraphBuilder.CreateInput(inputIdx, "gmm_x", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                  inputShapes[inputIdx].GetDims());
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: Created input gmm_x, idx=%d, shape=%s", 
              inputIdx, ShapeToStr(inputShapes[inputIdx]).c_str());
    ++inputIdx;
    
    auto rGmmWeight = replaceGraphBuilder.CreateInput(inputIdx, "gmm_weight", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                       inputShapes[inputIdx].GetDims());
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: Created input gmm_weight, idx=%d, shape=%s", 
              inputIdx, ShapeToStr(inputShapes[inputIdx]).c_str());
    ++inputIdx;
    
    auto rGmmXScale = replaceGraphBuilder.CreateInput(inputIdx, "gmm_x_scale", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                       inputShapes[inputIdx].GetDims());
    ++inputIdx;
    
    auto rGmmWeightScale = replaceGraphBuilder.CreateInput(inputIdx, "gmm_weight_scale", inputDTypes[inputIdx],
                                                           inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    ++inputIdx;
    
    ge::es::EsTensorHolder rSendCountsTensor = nullptr;
    ge::es::EsTensorHolder rRecvCountsTensor = nullptr;
    ge::es::EsTensorHolder rMmX = nullptr;
    ge::es::EsTensorHolder rMmWeight = nullptr;
    ge::es::EsTensorHolder rMmXScale = nullptr;
    ge::es::EsTensorHolder rMmWeightScale = nullptr;

    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(MC2_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Capture AlltoAllvQuantGroupedMatMul node failed.");
    ge::GNode mc2Node = mc2NodeIo.node;
    
    ge::fusion::NodeIo transposeNodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(TRANSPOSE_CAPTURE_IDX, transposeNodeIo) != ge::SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Capture Transpose node failed.");
    uint32_t transposeParentIdx = transposeNodeIo.index;
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: Transpose parentIdx=%u", transposeParentIdx);
    
    ge::AscendString mc2NodeName;
    mc2Node.GetName(mc2NodeName);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: Got MC2 node, name=%s", mc2NodeName.GetString());

    ge::AscendString group = "";
    OP_LOGE_IF(mc2Node.GetAttr("group", group) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group failed.");

    int64_t epWorldSize = -1;
    OP_LOGE_IF(mc2Node.GetAttr("ep_world_size", epWorldSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr ep_world_size failed.");

    std::vector<int64_t> sendCounts = {};
    OP_LOGE_IF(mc2Node.GetAttr("send_counts", sendCounts) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr send_counts failed.");

    std::vector<int64_t> recvCounts = {};
    OP_LOGE_IF(mc2Node.GetAttr("recv_counts", recvCounts) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr recv_counts failed.");

    int64_t gmmXQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("gmm_x_quant_mode", gmmXQuantMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr gmm_x_quant_mode failed.");

    int64_t gmmWeightQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("gmm_weight_quant_mode", gmmWeightQuantMode) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr gmm_weight_quant_mode failed.");

    bool transGmmWeight = false;
    OP_LOGE_IF(mc2Node.GetAttr("trans_gmm_weight", transGmmWeight) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr trans_gmm_weight failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: trans_gmm_weight before = %d", transGmmWeight);

    bool transMmWeight = false;
    OP_LOGE_IF(mc2Node.GetAttr("trans_mm_weight", transMmWeight) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr trans_mm_weight failed.");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: trans_mm_weight before = %d", transMmWeight);

    if (transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT) ||
        transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE)) {
        transGmmWeight = !transGmmWeight;
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: flipped trans_gmm_weight to %d (Transpose on idx=%u)",
                  transGmmWeight, transposeParentIdx);
    } else if (transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_MM_WEIGHT) ||
               transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_MM_WEIGHT_SCALE)) {
        transMmWeight = !transMmWeight;
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: flipped trans_mm_weight to %d (Transpose on idx=%u)",
                  transMmWeight, transposeParentIdx);
    }

    bool permuteOutFlag = false;
    OP_LOGE_IF(mc2Node.GetAttr("permute_out_flag", permuteOutFlag) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr permute_out_flag failed.");

    int64_t mmXQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("mm_x_quant_mode", mmXQuantMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr mm_x_quant_mode failed.");

    int64_t mmWeightQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("mm_weight_quant_mode", mmWeightQuantMode) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr mm_weight_quant_mode failed.");

    int64_t groupSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("group_size", groupSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group_size failed.");

    int64_t yDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("y_dtype", yDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr y_dtype failed.");

    int64_t mmDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("mm_dtype", mmDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr mm_dtype failed.");

    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        rGmmX, rGmmWeight, rGmmXScale, rGmmWeightScale,
        rSendCountsTensor, rRecvCountsTensor,
        rMmX, rMmWeight, rMmXScale, rMmWeightScale,
        group.GetString(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight, permuteOutFlag,
        mmXQuantMode, mmWeightQuantMode, groupSize, yDtype, mmDtype);

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: Created new MC2 node with flipped attributes");
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: trans_gmm_weight=%d, trans_mm_weight=%d", 
              transGmmWeight, transMmWeight);

    auto replaceGraph = replaceGraphBuilder.BuildAndReset({mc2.gmm_y});
    if (replaceGraph == nullptr) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Build replace graph failed.");
        return nullptr;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: BuildAndReset SUCCESS");

    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShape failed.");
        return nullptr;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement: InferShape SUCCESS");

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "=== Replacement SUCCESS - Transpose should be eliminated ===");
    return std::move(replaceGraph);
}

REG_FUSION_PASS(AlltoAllvQuantGroupedMatMulTransposeFusionPass)
    .Stage(ge::CustomPassStage::kCompatibleInherited);
} // namespace ops
#endif // GRAPH_FUSION_SUPPORT_VERSION