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
#include "es_AlltoAllMatmul.h" // es autogen header
#include "es_TransposeD.h"     // es autogen header for stub op
#include "es_math_ops.h"       // math ops stub
#include "common/utils/op_mc2.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>      // dlopen 动态加载
#include "acl/acl_rt.h" // 运行时判断cann ver
#include <set>
#include <tuple>

namespace ops {
const std::string FUSION_PASS_NAME = "AllToAllMatmulTransposeA5FusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_TRANSPOSE_D = "TransposeD";
const std::string PATTERN_BIAS = "HasBias";
const std::string PATTERN_X1_SCALE = "HasX1Scale";
const std::string PATTERN_X2_SCALE = "HasX2Scale";
const std::string PATTERN_COMM_SCALE = "HasCommScale";
const std::string PATTERN_X1_OFFSET = "HasX1Offset";
const std::string PATTERN_X2_OFFSET = "HasX2Offset";

const int64_t MX_QUANT_MODE = 6;
const int64_t ALL2ALLMM_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_CAPTURE_IDX = 1l;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_ONE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;

enum class TransposeNode {
    Transpose,
    TransposeD
};

struct OriginalGraphInfo {
    bool hasBias = false;
    bool hasX1Scale = false;
    bool hasX2Scale = false;
    bool hasCommScale = false;
    bool hasX1Offset = false;
    bool hasX2Offset = false;
    TransposeNode node = TransposeNode::Transpose;

    bool operator<(const OriginalGraphInfo &rhs) const
    {
        return std::tie(hasBias, hasX1Scale, hasX2Scale, hasCommScale, hasX1Offset, hasX2Offset, node) <
               std::tie(rhs.hasBias, rhs.hasX1Scale, rhs.hasX2Scale, rhs.hasCommScale, rhs.hasX1Offset, rhs.hasX2Offset,
                        rhs.node);
    }
};

struct ReplaceGraphInputs {
    ge::es::EsTensorHolder rX1;
    ge::es::EsTensorHolder rX2;
    // option
    ge::es::EsTensorHolder rBias;
    ge::es::EsTensorHolder rX1Scale;
    ge::es::EsTensorHolder rX2Scale;
    ge::es::EsTensorHolder rCommScale;
    ge::es::EsTensorHolder rX1Offset;
    ge::es::EsTensorHolder rX2Offset;
};

struct Mc2ReplaceAttrs {
    ge::AscendString group;
    int64_t world_size;
    std::vector<int64_t> all2all_axes;
    int64_t y_dtype;
    int64_t x1_quant_mode;
    int64_t x2_quant_mode;
    int64_t comm_quant_mode;
    int64_t x1_quant_dtype;
    int64_t comm_quant_dtype;
    bool transpose_x1;
    bool transpose_x2;
    int64_t group_size;
    ge::AscendString comm_mode;
    bool alltoall_out_flag;
};

// 加载 EsTranspose 符号
namespace {
typedef EsCTensorHolder *(*EsTransposeFunc)(EsCTensorHolder *, EsCTensorHolder *);

EsTransposeFunc GetEsTransposeFunc()
{
    void *handle = dlopen("libes_math.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        OPS_LOG_E("AllToAllMatmulTransposeA5FusionPass", "dlopen failed: %s", dlerror());
        return nullptr;
    }
    dlerror();
    auto func = reinterpret_cast<EsTransposeFunc>(dlsym(handle, "EsTranspose"));
    if (dlerror() != nullptr) {
        OPS_LOG_E("AllToAllMatmulTransposeA5FusionPass", "dlsym EsTranspose failed");
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
        OPS_LOG_E("AllToAllMatmulTransposeA5FusionPass", "EsTranspose is unavailable, fallback to TransposeD.");
        return ge::es::TransposeD(xHolder, std::vector<int64_t>{1, 0});
    }
    auto result = func(xHolder.GetCTensorHolder(), perm.ToTensorHolder(builder).GetCTensorHolder());
    return result;
}

ge::CustomPassStage GetAllToAllMatmulTransposeA5FusionPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}
} // namespace

// [FUSION_DBG] 打印 AlltoAllMatmul 每个输入锚点的生产者类型/名字
static void DumpMc2NodeInputs(const char *tag, const std::string &extra, const ge::GNode &mc2Node)
{
    ge::AscendString mc2Name;
    mc2Node.GetName(mc2Name);
    const size_t inputSize = mc2Node.GetInputsSize();
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] %s %s mc2Name=%s inputsSize=%zu", tag, extra.c_str(),
              mc2Name.GetString(), inputSize);
    for (size_t i = 0; i < inputSize; ++i) {
        auto peer = mc2Node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (peer.first == nullptr) {
            OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] %s in[%zu]=<empty>", tag, i);
            continue;
        }
        ge::AscendString peerType;
        ge::AscendString peerName;
        peer.first->GetType(peerType);
        peer.first->GetName(peerName);
        OPS_LOG_I(FUSION_PASS_NAME.c_str(),
                  "[FUSION_DBG] %s in[%zu] producerType=%s producerName=%s peerOutPort=%d", tag, i,
                  peerType.GetString(), peerName.GetString(), peer.second);
    }
}

static void DumpAllAlltoAllMatmulInGraph(const ge::GraphPtr &graph)
{
    if (graph == nullptr) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] TARGET graph is null");
        return;
    }
    const auto nodes = graph->GetAllNodes();
    size_t mc2Count = 0;
    for (const auto &node : nodes) {
        ge::AscendString nodeType;
        node.GetType(nodeType);
        if (std::string(nodeType.GetString()) != "AlltoAllMatmul") {
            continue;
        }
        ++mc2Count;
        DumpMc2NodeInputs("TARGET", "before_match", node);
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] TARGET AlltoAllMatmul count=%zu", mc2Count);
}

ge::Status AllToAllMatmulTransposeA5FusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &passContext)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] Enter Run, dump TARGET AlltoAllMatmul inputs");
    DumpAllAlltoAllMatmulInGraph(graph);
    return ge::fusion::PatternFusionPass::Run(graph, passContext);
}

static ge::fusion::PatternUniqPtr MakePattern(const OriginalGraphInfo &info)
{
    std::string patternName = FUSION_PASS_NAME;
    patternName += info.node == TransposeNode::Transpose ? PATTERN_TRANSPOSE : PATTERN_TRANSPOSE_D;
    if (info.hasBias) {
        patternName += PATTERN_BIAS;
    }
    if (info.hasX1Scale) {
        patternName += PATTERN_X1_SCALE;
    }
    if (info.hasX2Scale) {
        patternName += PATTERN_X2_SCALE;
    }
    if (info.hasCommScale) {
        patternName += PATTERN_COMM_SCALE;
    }
    if (info.hasX1Offset) {
        patternName += PATTERN_X1_OFFSET;
    }
    if (info.hasX2Offset) {
        patternName += PATTERN_X2_OFFSET;
    }
    // [FUSION_DBG] 期望 AlltoAllMatmul 有效输入个数 = 2(x1,x2) + 可选
    const size_t expectMc2InputNum =
        2U + static_cast<size_t>(info.hasBias) + static_cast<size_t>(info.hasX1Scale) +
        static_cast<size_t>(info.hasX2Scale) + static_cast<size_t>(info.hasCommScale) +
        static_cast<size_t>(info.hasX1Offset) + static_cast<size_t>(info.hasX2Offset);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(),
              "[FUSION_DBG] MakePattern name=%s expectMc2InputNum=%zu flags={bias=%d,x1s=%d,x2s=%d,comm=%d,x1o=%d,x2o=%d,node=%s}",
              patternName.c_str(), expectMc2InputNum, info.hasBias, info.hasX1Scale, info.hasX2Scale, info.hasCommScale,
              info.hasX1Offset, info.hasX2Offset,
              info.node == TransposeNode::Transpose ? "Transpose" : "TransposeD");

    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto [x1, x2, bias, x1Scale, x2Scale, commScale, x1Offset, x2Offset] = graphBuilder.CreateInputs<8>();
    ge::es::EsTensorHolder transpose;
    if (info.node == TransposeNode::Transpose) {
        transpose = TransposeDL(x2, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    } else {
        transpose = ge::es::TransposeD(x2, std::vector<int64_t>{1, 0});
    }

    // attr
    const char *group = "";
    int64_t worldSize = 0;

    auto mc2 = ge::es::AlltoAllMatmul(x1, transpose, info.hasBias ? bias : nullptr, info.hasX1Scale ? x1Scale : nullptr,
                                      info.hasX2Scale ? x2Scale : nullptr, info.hasCommScale ? commScale : nullptr,
                                      info.hasX1Offset ? x1Offset : nullptr, info.hasX2Offset ? x2Offset : nullptr, group,
                                      worldSize);
    // 与 MatmulAlltoAll fusion 一致：pattern 只挂主输出 y，避免 all2all_out 未接入时匹配失败
    auto graph = graphBuilder.BuildAndReset({mc2.y});
    if (graph == nullptr) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] MakePattern BuildAndReset failed, name=%s", patternName.c_str());
        return nullptr;
    }
    auto *mc2Producer = mc2.y.GetProducer();
    auto *transposeProducer = transpose.GetProducer();
    OPS_LOG_I(FUSION_PASS_NAME.c_str(),
              "[FUSION_DBG] MakePattern built name=%s expectMc2InputNum=%zu mc2Producer=%p transposeProducer=%p",
              patternName.c_str(), expectMc2InputNum, static_cast<const void *>(mc2Producer),
              static_cast<const void *>(transposeProducer));
    if (expectMc2InputNum == 3U && mc2Producer != nullptr) {
        DumpMc2NodeInputs("PATTERN", patternName, *mc2Producer);
    }
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2Producer, 0}).CaptureTensor({*transposeProducer, 0});
    return pattern;
}

static bool IsMXQuantMode(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(ALL2ALLMM_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get AlltoAllMatmul node failed.");
    ge::GNode mc2Node = mc2NodeIo.node;

    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    if (mc2Node.GetAttr("x1_quant_mode", x1QuantMode) != ge::GRAPH_SUCCESS ||
        mc2Node.GetAttr("x2_quant_mode", x2QuantMode) != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get quant mode Attr failed.");
        return false;
    }
    return (x1QuantMode == MX_QUANT_MODE && x2QuantMode == MX_QUANT_MODE);
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

    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeNode, permValue)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation.");
        return false;
    }

    const size_t permSize = permValue.size();
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation size must be 2 or 3, but got %zu.", permSize);
        return false;
    }

    // x2 路径：最后两维必须互换，即 2维 [1,0] / 3维 [0,2,1]
    if ((permValue[permSize - PERM_SIZE_ONE] != static_cast<int64_t>(permSize - PERM_SIZE_TWO)) ||
        (permValue[permSize - PERM_SIZE_TWO] != static_cast<int64_t>(permSize - PERM_SIZE_ONE))) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation last 2 dims must be swapped.");
        return false;
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

static size_t GetMinRequiredSubgraphInputNum(const std::string &patternNameStr)
{
    bool hasBias = patternNameStr.find(PATTERN_BIAS) != std::string::npos;
    bool hasX1Scale = patternNameStr.find(PATTERN_X1_SCALE) != std::string::npos;
    bool hasX2Scale = patternNameStr.find(PATTERN_X2_SCALE) != std::string::npos;
    bool hasCommScale = patternNameStr.find(PATTERN_COMM_SCALE) != std::string::npos;
    bool hasX1Offset = patternNameStr.find(PATTERN_X1_OFFSET) != std::string::npos;
    bool hasX2Offset = patternNameStr.find(PATTERN_X2_OFFSET) != std::string::npos;

    // required input x1,x2
    size_t inputNum = 2;
    if (hasBias) {
        inputNum++;
    }
    if (hasX1Scale) {
        inputNum++;
    }
    if (hasX2Scale) {
        inputNum++;
    }
    if (hasCommScale) {
        inputNum++;
    }
    if (hasX1Offset) {
        inputNum++;
    }
    if (hasX2Offset) {
        inputNum++;
    }

    return inputNum;
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
    if (patternNameStr.find(PATTERN_X1_SCALE) != std::string::npos) {
        inputs.rX1Scale = replaceGraphBuilder.CreateInput(inputIdx, "x1_scale", inputDTypes[inputIdx],
                                                          inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }
    inputs.rX2Scale = nullptr;
    if (patternNameStr.find(PATTERN_X2_SCALE) != std::string::npos) {
        inputs.rX2Scale = replaceGraphBuilder.CreateInput(inputIdx, "x2_scale", inputDTypes[inputIdx],
                                                          inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }
    inputs.rCommScale = nullptr;
    if (patternNameStr.find(PATTERN_COMM_SCALE) != std::string::npos) {
        inputs.rCommScale = replaceGraphBuilder.CreateInput(inputIdx, "comm_scale", inputDTypes[inputIdx],
                                                            inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }
    inputs.rX1Offset = nullptr;
    if (patternNameStr.find(PATTERN_X1_OFFSET) != std::string::npos) {
        inputs.rX1Offset = replaceGraphBuilder.CreateInput(inputIdx, "x1_offset", inputDTypes[inputIdx],
                                                           inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }
    inputs.rX2Offset = nullptr;
    if (patternNameStr.find(PATTERN_X2_OFFSET) != std::string::npos) {
        inputs.rX2Offset = replaceGraphBuilder.CreateInput(inputIdx, "x2_offset", inputDTypes[inputIdx],
                                                           inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    }
    return true;
}

static bool GetCapturedMc2Node(const std::unique_ptr<ge::fusion::MatchResult> &matchResult, ge::GNode &mc2Node)
{
    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(ALL2ALLMM_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Capture AlltoAllMatmul node failed.");
    mc2Node = mc2NodeIo.node;
    return true;
}

static bool GetMc2ReplaceAttrs(const ge::GNode &mc2Node, Mc2ReplaceAttrs &attrs)
{
    OP_LOGE_IF(mc2Node.GetAttr("group", attrs.group) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr group failed.");
    OP_LOGE_IF(mc2Node.GetAttr("world_size", attrs.world_size) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr world_size failed.");
    OP_LOGE_IF(mc2Node.GetAttr("all2all_axes", attrs.all2all_axes) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr all2all_axes failed.");
    OP_LOGE_IF(mc2Node.GetAttr("y_dtype", attrs.y_dtype) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr y_dtype failed.");
    OP_LOGE_IF(mc2Node.GetAttr("x1_quant_mode", attrs.x1_quant_mode) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr x1_quant_mode failed.");
    OP_LOGE_IF(mc2Node.GetAttr("x2_quant_mode", attrs.x2_quant_mode) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr x2_quant_mode failed.");
    OP_LOGE_IF(mc2Node.GetAttr("comm_quant_mode", attrs.comm_quant_mode) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_quant_mode failed.");
    OP_LOGE_IF(mc2Node.GetAttr("x1_quant_dtype", attrs.x1_quant_dtype) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr x1_quant_dtype failed.");
    OP_LOGE_IF(mc2Node.GetAttr("comm_quant_dtype", attrs.comm_quant_dtype) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_quant_dtype failed.");
    OP_LOGE_IF(mc2Node.GetAttr("transpose_x1", attrs.transpose_x1) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr transpose_x1 failed.");
    OP_LOGE_IF(mc2Node.GetAttr("transpose_x2", attrs.transpose_x2) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr transpose_x2 failed.");
    OP_LOGE_IF(mc2Node.GetAttr("group_size", attrs.group_size) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr group_size failed.");
    OP_LOGE_IF(mc2Node.GetAttr("comm_mode", attrs.comm_mode) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_mode failed.");
    OP_LOGE_IF(mc2Node.GetAttr("alltoall_out_flag", attrs.alltoall_out_flag) != ge::GRAPH_SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get Attr alltoall_out_flag failed.");

    // 移除 Transpose 节点后，transpose_x2 置反
    attrs.transpose_x2 = !attrs.transpose_x2;
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

    auto mc2 = ge::es::AlltoAllMatmul(inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, inputTensors.rX1Scale,
                                      inputTensors.rX2Scale, inputTensors.rCommScale, inputTensors.rX1Offset, inputTensors.rX2Offset,
                                      attrs.group.GetString(), attrs.world_size, attrs.all2all_axes, attrs.y_dtype, attrs.x1_quant_mode,
                                      attrs.x2_quant_mode, attrs.comm_quant_mode, attrs.x1_quant_dtype, attrs.comm_quant_dtype,
                                      attrs.transpose_x1, attrs.transpose_x2, attrs.group_size, attrs.comm_mode.GetString(), attrs.alltoall_out_flag);
    // 与 MakePattern 保持一致，替换子图只暴露主输出 y
    return replaceGraphBuilder.BuildAndReset({mc2.y});
}

static bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr &replaceGraph,
                                   const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter InferShapeReplaceGraph for AllToAllMatmulTransposeA5FusionPass");
    std::vector<ge::Shape> inputShapes;
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
        inputShapes.emplace_back(tmpDesc.GetShape());
    }
    return (ge::GeUtils::InferShape(*replaceGraph, inputShapes) == ge::SUCCESS);
}

std::vector<ge::fusion::PatternUniqPtr> AllToAllMatmulTransposeA5FusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns for AllToAllMatmulTransposeA5FusionPass");
    // hasBias, hasX1Scale, hasX2Scale, hasCommScale, hasX1Offset, hasX2Offset, TransposeNode
    std::set<OriginalGraphInfo> infos;
    for (bool hasBias : {false, true}) {
        for (bool hasX1Scale : {false, true}) {
            for (bool hasX2Scale : {false, true}) {
                for (bool hasCommScale : {false, true}) {
                    for (bool hasX1Offset : {false, true}) {
                        for (bool hasX2Offset : {false, true}) {
                            for (auto node : {TransposeNode::Transpose, TransposeNode::TransposeD}) {
                                infos.insert({hasBias, hasX1Scale, hasX2Scale, hasCommScale, hasX1Offset, hasX2Offset,
                                              node});
                            }
                        }
                    }
                }
            }
        }
    }
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    size_t size3Count = 0;
    for (const auto &info : infos) {
        const size_t expectMc2InputNum =
            2U + static_cast<size_t>(info.hasBias) + static_cast<size_t>(info.hasX1Scale) +
            static_cast<size_t>(info.hasX2Scale) + static_cast<size_t>(info.hasCommScale) +
            static_cast<size_t>(info.hasX1Offset) + static_cast<size_t>(info.hasX2Offset);
        if (expectMc2InputNum == 3U) {
            ++size3Count;
        }
        patternGraphs.emplace_back(MakePattern(info));
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(),
              "[FUSION_DBG] Patterns done, total=%zu size3=%zu (infos=%zu)", patternGraphs.size(), size3Count,
              infos.size());
    return patternGraphs;
}

bool AllToAllMatmulTransposeA5FusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::AscendString patternName = "";
    matchResult->GetPatternGraph().GetName(patternName);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(),
              "[FUSION_DBG] Enter MeetRequirements, pattern=%s", patternName.GetString());
    ge::fusion::NodeIo mc2NodeIo;
    if (matchResult->GetCapturedTensor(ALL2ALLMM_CAPTURE_IDX, mc2NodeIo) == ge::SUCCESS) {
        DumpMc2NodeInputs("MATCHED", patternName.GetString(), mc2NodeIo.node);
    }

    // 9.0.0 版本前运行降级stage 空跑
    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(),
                  "[FUSION_DBG] Skip MeetRequirements: cann version not compatible, ver=%d", geCompilerVersion);
        return false;
    }

    // 是否ascend950
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] Skip MeetRequirements: platform not A5");
        return false;
    }

    // 是否mx quant
    if (!IsMXQuantMode(matchResult)) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] Skip MeetRequirements: not MX QUANT, pattern=%s",
                  patternName.GetString());
        return false;
    }

    // 是否transpose 条件吻合
    if (!IsTransposePermValid(matchResult)) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] Skip MeetRequirements: transpose perm invalid, pattern=%s",
                  patternName.GetString());
        return false;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(),
              "[FUSION_DBG] MeetRequirements PASS, pattern=%s", patternName.GetString());
    return true;
}

ge::fusion::GraphUniqPtr AllToAllMatmulTransposeA5FusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::AscendString patternName = "";
    matchResult->GetPatternGraph().GetName(patternName);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] Enter Replacement, pattern=%s", patternName.GetString());
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    size_t subgraphInputSize = subgraphInputs.size();
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] Replacement subgraph input size=%zu, pattern=%s",
              subgraphInputSize, patternName.GetString());

    // build replace graph after removing Transpose nodes
    ge::fusion::GraphUniqPtr replaceGraph = BuildReplaceGraph(subgraphInputs, matchResult);
    if (replaceGraph == nullptr) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] BuildReplaceGraph failed, pattern=%s",
                  patternName.GetString());
        return nullptr;
    }

    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] InferShapeReplaceGraph failed, pattern=%s",
                  patternName.GetString());
        return nullptr;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "[FUSION_DBG] Replacement SUCCESS, pattern=%s", patternName.GetString());
    return std::move(replaceGraph);
}

REG_FUSION_PASS(AllToAllMatmulTransposeA5FusionPass)
    .Stage(GetAllToAllMatmulTransposeA5FusionPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION
