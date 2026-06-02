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
#include "es_AlltoAllvQuantGroupedMatMul.h" // es autogen header
#include "es_TransposeD.h"     // es autogen header for stub op
#include "es_math_ops.h"       // math ops stub
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h> // dlopen 动态加载

namespace ops {
const std::string FUSION_PASS_NAME = "AlltoAllvQuantGroupedMatMulTransposeFusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_TRANSPOSE_D = "TransposeD";

const int64_t MC2_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_CAPTURE_IDX = 1l;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const int64_t MX_QUANT_MODE = 6;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;
const size_t PERM_SIZE_FOUR = 4;
const size_t PERM_SIZE_ONE = 1;
const size_t INPUT_NUM_TOTAL = 8;

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

struct ReplaceGraphInputs {
    ge::es::EsTensorHolder rGmmX;
    ge::es::EsTensorHolder rGmmWeight;
    ge::es::EsTensorHolder rGmmXScale;
    ge::es::EsTensorHolder rGmmWeightScale;
    ge::es::EsTensorHolder rMmX;
    ge::es::EsTensorHolder rMmWeight;
    ge::es::EsTensorHolder rMmXScale;
    ge::es::EsTensorHolder rMmWeightScale;
};

// 加载 EsTranspose 符号
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

static bool IsTransposePermValid(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(TRANSPOSE_CAPTURE_IDX, transposeOutput) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get Transpose node failed.");
    auto transposeNode = transposeOutput.node;
    uint32_t parentIdx = transposeOutput.index;

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

static bool IsMXQuantMode(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(MC2_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get AlltoAllvQuantGroupedMatMul node failed.");
    ge::GNode mc2Node = mc2NodeIo.node;

    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    if (mc2Node.GetAttr("gmm_x_quant_mode", gmmXQuantMode) != ge::GRAPH_SUCCESS ||
        mc2Node.GetAttr("gmm_weight_quant_mode", gmmWeightQuantMode) != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get quant mode Attr failed.");
        return false;
    }
    return (gmmXQuantMode == MX_QUANT_MODE && gmmWeightQuantMode == MX_QUANT_MODE);
}

// Pattern for Transpose on gmm_weight (REQUIRED输入 + mm系列配套输入)
static ge::fusion::PatternUniqPtr MakePatternForTransposeGmmWeight()
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE + "_GmmWeight";
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    
    auto [gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmX, mmWeight, mmXScale, mmWeightScale] = graphBuilder.CreateInputs<8>();
    
    auto transpose = TransposeDL(gmmWeight, ge::es::EsTensorLike(std::vector<int64_t>{0, 2, 1}));
    
    std::string group = "";
    int64_t epWorldSize = 0;
    std::vector<int64_t> sendCounts = {};
    std::vector<int64_t> recvCounts = {};
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX, transpose, gmmXScale, gmmWeightScale,
        nullptr, nullptr,
        mmX, mmWeight, mmXScale, mmWeightScale,
        group.c_str(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode);
    
    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0}).CaptureTensor({*transpose.GetProducer(), 0});
    return pattern;
}

// Pattern for TransposeD on gmm_weight (REQUIRED输入 + mm系列配套输入)
static ge::fusion::PatternUniqPtr MakePatternForTransposeDGmmWeight()
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE_D + "_GmmWeight";
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    
    auto [gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmX, mmWeight, mmXScale, mmWeightScale] = graphBuilder.CreateInputs<8>();
    
    auto transposeD = ge::es::TransposeD(gmmWeight, std::vector<int64_t>{0, 2, 1});
    
    std::string group = "";
    int64_t epWorldSize = 0;
    std::vector<int64_t> sendCounts = {};
    std::vector<int64_t> recvCounts = {};
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX, transposeD, gmmXScale, gmmWeightScale,
        nullptr, nullptr,
        mmX, mmWeight, mmXScale, mmWeightScale,
        group.c_str(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode);
    
    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0}).CaptureTensor({*transposeD.GetProducer(), 0});
    return pattern;
}

// Pattern for Transpose on gmm_weight_scale (REQUIRED输入 + mm系列配套输入)
static ge::fusion::PatternUniqPtr MakePatternForTransposeGmmWeightScale()
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE + "_GmmWeightScale";
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    
    auto [gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmX, mmWeight, mmXScale, mmWeightScale] = graphBuilder.CreateInputs<8>();
    
    auto transpose = TransposeDL(gmmWeightScale, ge::es::EsTensorLike(std::vector<int64_t>{0, 2, 1, 3}));
    
    std::string group = "";
    int64_t epWorldSize = 0;
    std::vector<int64_t> sendCounts = {};
    std::vector<int64_t> recvCounts = {};
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX, gmmWeight, gmmXScale, transpose,
        nullptr, nullptr,
        mmX, mmWeight, mmXScale, mmWeightScale,
        group.c_str(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode);
    
    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0}).CaptureTensor({*transpose.GetProducer(), 0});
    return pattern;
}

// Pattern for TransposeD on gmm_weight_scale (REQUIRED输入 + mm系列配套输入)
static ge::fusion::PatternUniqPtr MakePatternForTransposeDGmmWeightScale()
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE_D + "_GmmWeightScale";
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    
    auto [gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmX, mmWeight, mmXScale, mmWeightScale] = graphBuilder.CreateInputs<8>();
    
    auto transposeD = ge::es::TransposeD(gmmWeightScale, std::vector<int64_t>{0, 2, 1, 3});
    
    std::string group = "";
    int64_t epWorldSize = 0;
    std::vector<int64_t> sendCounts = {};
    std::vector<int64_t> recvCounts = {};
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX, gmmWeight, gmmXScale, transposeD,
        nullptr, nullptr,
        mmX, mmWeight, mmXScale, mmWeightScale,
        group.c_str(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode);
    
    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0}).CaptureTensor({*transposeD.GetProducer(), 0});
    return pattern;
}

// Pattern for Transpose on mm_weight (REQUIRED输入 + mm系列配套输入)
static ge::fusion::PatternUniqPtr MakePatternForTransposeMmWeight()
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE + "_MmWeight";
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    
    // REQUIRED输入 + mm系列配套输入
    auto [gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmX, mmWeight, mmXScale, mmWeightScale] = graphBuilder.CreateInputs<8>();
    
    auto transpose = TransposeDL(mmWeight, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    
    std::string group = "";
    int64_t epWorldSize = 0;
    std::vector<int64_t> sendCounts = {};
    std::vector<int64_t> recvCounts = {};
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX, gmmWeight, gmmXScale, gmmWeightScale,
        nullptr, nullptr,
        mmX, transpose, mmXScale, mmWeightScale,
        group.c_str(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode);
    
    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0}).CaptureTensor({*transpose.GetProducer(), 0});
    return pattern;
}

// Pattern for TransposeD on mm_weight (REQUIRED输入 + mm系列配套输入)
static ge::fusion::PatternUniqPtr MakePatternForTransposeDMmWeight()
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE_D + "_MmWeight";
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    
    // REQUIRED输入 + mm系列配套输入
    auto [gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmX, mmWeight, mmXScale, mmWeightScale] = graphBuilder.CreateInputs<8>();
    
    auto transposeD = ge::es::TransposeD(mmWeight, std::vector<int64_t>{1, 0});
    
    std::string group = "";
    int64_t epWorldSize = 0;
    std::vector<int64_t> sendCounts = {};
    std::vector<int64_t> recvCounts = {};
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX, gmmWeight, gmmXScale, gmmWeightScale,
        nullptr, nullptr,
        mmX, transposeD, mmXScale, mmWeightScale,
        group.c_str(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode);
    
    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0}).CaptureTensor({*transposeD.GetProducer(), 0});
    return pattern;
}

// Pattern for Transpose on mm_weight_scale (REQUIRED输入 + mm系列配套输入)
static ge::fusion::PatternUniqPtr MakePatternForTransposeMmWeightScale()
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE + "_MmWeightScale";
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    
    // REQUIRED输入 + mm系列配套输入
    auto [gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmX, mmWeight, mmXScale, mmWeightScale] = graphBuilder.CreateInputs<8>();
    
    auto transpose = TransposeDL(mmWeightScale, ge::es::EsTensorLike(std::vector<int64_t>{1, 0, 2}));
    
    std::string group = "";
    int64_t epWorldSize = 0;
    std::vector<int64_t> sendCounts = {};
    std::vector<int64_t> recvCounts = {};
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX, gmmWeight, gmmXScale, gmmWeightScale,
        nullptr, nullptr,
        mmX, mmWeight, mmXScale, transpose,
        group.c_str(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode);
    
    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0}).CaptureTensor({*transpose.GetProducer(), 0});
    return pattern;
}

// Pattern for TransposeD on mm_weight_scale (REQUIRED输入 + mm系列配套输入)
static ge::fusion::PatternUniqPtr MakePatternForTransposeDMmWeightScale()
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE_D + "_MmWeightScale";
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    
    // REQUIRED输入 + mm系列配套输入
    auto [gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmX, mmWeight, mmXScale, mmWeightScale] = graphBuilder.CreateInputs<8>();
    
    auto transposeD = ge::es::TransposeD(mmWeightScale, std::vector<int64_t>{1, 0, 2});
    
    std::string group = "";
    int64_t epWorldSize = 0;
    std::vector<int64_t> sendCounts = {};
    std::vector<int64_t> recvCounts = {};
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX, gmmWeight, gmmXScale, gmmWeightScale,
        nullptr, nullptr,
        mmX, mmWeight, mmXScale, transposeD,
        group.c_str(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode);
    
    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0}).CaptureTensor({*transposeD.GetProducer(), 0});
    return pattern;
}

std::vector<ge::fusion::PatternUniqPtr> AlltoAllvQuantGroupedMatMulTransposeFusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns for AlltoAllvQuantGroupedMatMulTransposeFusionPass");
    
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    
    patternGraphs.emplace_back(MakePatternForTransposeGmmWeight());
    patternGraphs.emplace_back(MakePatternForTransposeGmmWeightScale());
    patternGraphs.emplace_back(MakePatternForTransposeMmWeight());
    patternGraphs.emplace_back(MakePatternForTransposeMmWeightScale());
    
    patternGraphs.emplace_back(MakePatternForTransposeDGmmWeight());
    patternGraphs.emplace_back(MakePatternForTransposeDGmmWeightScale());
    patternGraphs.emplace_back(MakePatternForTransposeDMmWeight());
    patternGraphs.emplace_back(MakePatternForTransposeDMmWeightScale());
    
    return patternGraphs;
}

bool AlltoAllvQuantGroupedMatMulTransposeFusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for AlltoAllvQuantGroupedMatMulTransposeFusionPass");

    // 是否ascend950
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Check target platform fail!");
        return false;
    }

    // 是否mx quant
    if (!IsMXQuantMode(matchResult)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Quant mode is not MX QUANT, fusion will be skipped.");
        return false;
    }

    // 是否transpose 条件吻合
    if (!IsTransposePermValid(matchResult)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation is not valid.");
        return false;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found One pattern that meets requirements");
    return true;
}

static void GetInputsInfo(const std::vector<ge::fusion::SubgraphInput> &subGraphInputs,
                          std::vector<ge::Shape> &inputShapes, std::vector<ge::DataType> &inputDTypes,
                          std::vector<ge::Format> &inputFormats)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter GetInputsInfo for AlltoAllvQuantGroupedMatMulTransposeFusionPass");
    for (const auto &subGraphInput : subGraphInputs) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetInputsInfo: collecting input datas.");
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
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Finish GetInputsInfo for AlltoAllvQuantGroupedMatMulTransposeFusionPass");
}

static std::vector<int64_t> ReverseTransposeDims(const std::vector<int64_t> &dims, uint32_t transposeParentIdx)
{
    std::vector<int64_t> reversed = dims;
    if (transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT) ||
        transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE)) {
        if (reversed.size() >= DIM_THREE) {
            std::swap(reversed[DIM_ONE], reversed[DIM_TWO]);
        }
    } else if (transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_MM_WEIGHT) ||
               transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_MM_WEIGHT_SCALE)) {
        if (reversed.size() >= DIM_TWO) {
            std::swap(reversed[DIM_ZERO], reversed[DIM_ONE]);
        }
    }
    return reversed;
}

static bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr &replaceGraph,
                                   const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                   uint32_t transposeParentIdx)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter InferShapeReplaceGraph for AlltoAllvQuantGroupedMatMulTransposeFusionPass");
    std::vector<ge::Shape> inputShapes;
    size_t idx = 0;
    for (const auto &subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph: matchNodes is empty.");
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetInputDesc failed.");
            continue;
        }
        auto dims = tmpDesc.GetShape().GetDims();
        if (idx == static_cast<size_t>(MC2InputIdx::K_GMM_WEIGHT) &&
            transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT)) {
            dims = ReverseTransposeDims(dims, transposeParentIdx);
        } else if (idx == static_cast<size_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE) &&
                   transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE)) {
            dims = ReverseTransposeDims(dims, transposeParentIdx);
        } else if (idx == static_cast<size_t>(MC2InputIdx::K_MM_WEIGHT) &&
                   transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_MM_WEIGHT)) {
            dims = ReverseTransposeDims(dims, transposeParentIdx);
        } else if (idx == static_cast<size_t>(MC2InputIdx::K_MM_WEIGHT_SCALE) &&
                   transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_MM_WEIGHT_SCALE)) {
            dims = ReverseTransposeDims(dims, transposeParentIdx);
        }
        inputShapes.emplace_back(ge::Shape(dims));
        ++idx;
    }
    return (ge::GeUtils::InferShape(*replaceGraph, inputShapes) == ge::SUCCESS);
}

static bool CreateReplaceGraphInputs(ReplaceGraphInputs &inputs, ge::es::EsGraphBuilder &replaceGraphBuilder,
                                     const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                     const std::unique_ptr<ge::fusion::MatchResult> &matchResult,
                                     uint32_t transposeParentIdx)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to create inputs in Replacement.");
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats);
    if (inputShapes.size() < INPUT_NUM_TOTAL) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement, need %zu but got %zu.",
                  INPUT_NUM_TOTAL, inputShapes.size());
        return false;
    }

    int64_t inputIdx = 0;
    inputs.rGmmX = replaceGraphBuilder.CreateInput(inputIdx, "gmm_x", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                    inputShapes[inputIdx].GetDims());
    ++inputIdx;

    auto gmmWeightDims = inputShapes[inputIdx].GetDims();
    if (transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT)) {
        gmmWeightDims = ReverseTransposeDims(gmmWeightDims, transposeParentIdx);
    }
    inputs.rGmmWeight = replaceGraphBuilder.CreateInput(inputIdx, "gmm_weight", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                         gmmWeightDims);
    ++inputIdx;
    
    inputs.rGmmXScale = replaceGraphBuilder.CreateInput(inputIdx, "gmm_x_scale", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                         inputShapes[inputIdx].GetDims());
    ++inputIdx;

    auto gmmWeightScaleDims = inputShapes[inputIdx].GetDims();
    if (transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT_SCALE)) {
        gmmWeightScaleDims = ReverseTransposeDims(gmmWeightScaleDims, transposeParentIdx);
    }
    inputs.rGmmWeightScale = replaceGraphBuilder.CreateInput(inputIdx, "gmm_weight_scale", inputDTypes[inputIdx],
                                                              inputFormats[inputIdx], gmmWeightScaleDims);
    ++inputIdx;
    
    inputs.rMmX = replaceGraphBuilder.CreateInput(inputIdx, "mm_x", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                   inputShapes[inputIdx].GetDims());
    ++inputIdx;

    auto mmWeightDims = inputShapes[inputIdx].GetDims();
    if (transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_MM_WEIGHT)) {
        mmWeightDims = ReverseTransposeDims(mmWeightDims, transposeParentIdx);
    }
    inputs.rMmWeight = replaceGraphBuilder.CreateInput(inputIdx, "mm_weight", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                        mmWeightDims);
    ++inputIdx;
    
    inputs.rMmXScale = replaceGraphBuilder.CreateInput(inputIdx, "mm_x_scale", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                        inputShapes[inputIdx].GetDims());
    ++inputIdx;

    auto mmWeightScaleDims = inputShapes[inputIdx].GetDims();
    if (transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_MM_WEIGHT_SCALE)) {
        mmWeightScaleDims = ReverseTransposeDims(mmWeightScaleDims, transposeParentIdx);
    }
    inputs.rMmWeightScale = replaceGraphBuilder.CreateInput(inputIdx, "mm_weight_scale", inputDTypes[inputIdx],
                                                             inputFormats[inputIdx], mmWeightScaleDims);
    
    return true;
}

static ge::fusion::GraphUniqPtr BuildReplaceGraph(const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                                  const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to build replaceGraph EsGraphBuilder.");
    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");

    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(MC2_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Capture AlltoAllvQuantGroupedMatMul node failed.");
    ge::GNode mc2Node = mc2NodeIo.node;
    
    ge::fusion::NodeIo transposeNodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(TRANSPOSE_CAPTURE_IDX, transposeNodeIo) != ge::SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Capture Transpose node failed.");
    uint32_t transposeParentIdx = transposeNodeIo.index;

    ReplaceGraphInputs inputTensors;
    if (!CreateReplaceGraphInputs(inputTensors, replaceGraphBuilder, subgraphInputs, matchResult, transposeParentIdx)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CreateReplaceGraphInputs failed.");
        return nullptr;
    }

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

    bool transMmWeight = false;
    OP_LOGE_IF(mc2Node.GetAttr("trans_mm_weight", transMmWeight) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr trans_mm_weight failed.");
    
    // 移除Transpose节点后，对应transpose属性置反
    if (transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_GMM_WEIGHT)) {
        transGmmWeight = !transGmmWeight;
    } else if (transposeParentIdx == static_cast<uint32_t>(MC2InputIdx::K_MM_WEIGHT)) {
        transMmWeight = !transMmWeight;
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
        inputTensors.rGmmX, inputTensors.rGmmWeight, inputTensors.rGmmXScale, inputTensors.rGmmWeightScale,
        nullptr, nullptr,
        inputTensors.rMmX, inputTensors.rMmWeight, inputTensors.rMmXScale, inputTensors.rMmWeightScale,
        group.GetString(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight, permuteOutFlag,
        mmXQuantMode, mmWeightQuantMode, groupSize, yDtype, mmDtype);

    return replaceGraphBuilder.BuildAndReset({mc2.gmm_y});
}

ge::fusion::GraphUniqPtr AlltoAllvQuantGroupedMatMulTransposeFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement for AlltoAllvQuantGroupedMatMulTransposeFusionPass");
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    size_t subgraphInputSize = subgraphInputs.size();
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Subgraph input size is %zu.", subgraphInputSize);

    ge::fusion::NodeIo transposeNodeIoForInfer;
    uint32_t transposeParentIdxForInfer = 0;
    if (matchResult->GetCapturedTensor(TRANSPOSE_CAPTURE_IDX, transposeNodeIoForInfer) == ge::SUCCESS) {
        transposeParentIdxForInfer = transposeNodeIoForInfer.index;
    }

    // build replace graph with transpose attr reversed
    ge::fusion::GraphUniqPtr replaceGraph = BuildReplaceGraph(subgraphInputs, matchResult);
    if (replaceGraph == nullptr) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "BuildReplaceGraph failed in Replacement.");
        return nullptr;
    }
    // infershape
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs, transposeParentIdxForInfer)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }
    return std::move(replaceGraph);
}

REG_FUSION_PASS(AlltoAllvQuantGroupedMatMulTransposeFusionPass)
    .Stage(ge::CustomPassStage::kCompatibleInherited); // 兼容性场景下指定stage
} // namespace ops
#endif // GRAPH_FUSION_SUPPORT_VERSION