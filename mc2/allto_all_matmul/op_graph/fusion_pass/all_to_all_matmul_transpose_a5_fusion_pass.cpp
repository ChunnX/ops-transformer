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
#include <string>
#include <cstring>
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

// AlltoAllMatmul IR 输入下标（与 op def / ES IrDefInputs 一致）
const int32_t IR_BIAS = 2;
const int32_t IR_X1_SCALE = 3;
const int32_t IR_X2_SCALE = 4;
const int32_t IR_COMM_SCALE = 5;
const int32_t IR_X1_OFFSET = 6;
const int32_t IR_X2_OFFSET = 7;

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
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    // 只创建实际会连上的 Data，避免 CreateInputs<8> 留下未使用输入
    const size_t inputNum = 2U + static_cast<size_t>(info.hasBias) + static_cast<size_t>(info.hasX1Scale) +
                            static_cast<size_t>(info.hasX2Scale) + static_cast<size_t>(info.hasCommScale) +
                            static_cast<size_t>(info.hasX1Offset) + static_cast<size_t>(info.hasX2Offset);
    auto inputs = graphBuilder.CreateInputs(inputNum);
    size_t idx = 0;
    auto x1 = inputs[idx++];
    auto x2 = inputs[idx++];
    ge::es::EsTensorHolder bias = nullptr;
    ge::es::EsTensorHolder x1Scale = nullptr;
    ge::es::EsTensorHolder x2Scale = nullptr;
    ge::es::EsTensorHolder commScale = nullptr;
    ge::es::EsTensorHolder x1Offset = nullptr;
    ge::es::EsTensorHolder x2Offset = nullptr;
    if (info.hasBias) {
        bias = inputs[idx++];
    }
    if (info.hasX1Scale) {
        x1Scale = inputs[idx++];
    }
    if (info.hasX2Scale) {
        x2Scale = inputs[idx++];
    }
    if (info.hasCommScale) {
        commScale = inputs[idx++];
    }
    if (info.hasX1Offset) {
        x1Offset = inputs[idx++];
    }
    if (info.hasX2Offset) {
        x2Offset = inputs[idx++];
    }

    ge::es::EsTensorHolder transpose;
    if (info.node == TransposeNode::Transpose) {
        transpose = TransposeDL(x2, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    } else {
        transpose = ge::es::TransposeD(x2, std::vector<int64_t>{1, 0});
    }

    const char *group = "";
    int64_t worldSize = 0;
    auto mc2 = ge::es::AlltoAllMatmul(x1, transpose, bias, x1Scale, x2Scale, commScale, x1Offset, x2Offset, group,
                                      worldSize);
    auto graph = graphBuilder.BuildAndReset({mc2.y, mc2.all2all_out});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.y.GetProducer(), 0}).CaptureTensor({*transpose.GetProducer(), 0});
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
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Begin CollectSubgraphInputsInfo.");
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
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "End CollectSubgraphInputsInfo.");
    return true;
}

static bool GetCapturedMc2Node(const std::unique_ptr<ge::fusion::MatchResult> &matchResult, ge::GNode &mc2Node);

static bool IsSameGNode(const ge::GNode &lhs, const ge::GNode &rhs)
{
    ge::AscendString lhsName;
    ge::AscendString rhsName;
    if (lhs.GetName(lhsName) != ge::GRAPH_SUCCESS || rhs.GetName(rhsName) != ge::GRAPH_SUCCESS) {
        return false;
    }
    const char *lhsStr = lhsName.GetString();
    const char *rhsStr = rhsName.GetString();
    if (lhsStr == nullptr || rhsStr == nullptr) {
        return false;
    }
    return std::strcmp(lhsStr, rhsStr) == 0;
}

static const char *IrInputName(int32_t irIndex)
{
    switch (irIndex) {
        case IR_BIAS:
            return "bias";
        case IR_X1_SCALE:
            return "x1_scale";
        case IR_X2_SCALE:
            return "x2_scale";
        case IR_COMM_SCALE:
            return "comm_scale";
        case IR_X1_OFFSET:
            return "x1_offset";
        case IR_X2_OFFSET:
            return "x2_offset";
        default:
            return "unknown";
    }
}

static void AssignOptionalInput(ReplaceGraphInputs &inputs, int32_t irIndex, const ge::es::EsTensorHolder &holder)
{
    switch (irIndex) {
        case IR_BIAS:
            inputs.rBias = holder;
            break;
        case IR_X1_SCALE:
            inputs.rX1Scale = holder;
            break;
        case IR_X2_SCALE:
            inputs.rX2Scale = holder;
            break;
        case IR_COMM_SCALE:
            inputs.rCommScale = holder;
            break;
        case IR_X1_OFFSET:
            inputs.rX1Offset = holder;
            break;
        case IR_X2_OFFSET:
            inputs.rX2Offset = holder;
            break;
        default:
            break;
    }
}

// 顺序必须与 MakePattern 中 CreateInputs / idx++ 一致
static std::vector<int32_t> GetOptionalIrsFromPatternName(const std::string &patternNameStr)
{
    std::vector<int32_t> irs;
    if (patternNameStr.find(PATTERN_BIAS) != std::string::npos) {
        irs.push_back(IR_BIAS);
    }
    if (patternNameStr.find(PATTERN_X1_SCALE) != std::string::npos) {
        irs.push_back(IR_X1_SCALE);
    }
    if (patternNameStr.find(PATTERN_X2_SCALE) != std::string::npos) {
        irs.push_back(IR_X2_SCALE);
    }
    if (patternNameStr.find(PATTERN_COMM_SCALE) != std::string::npos) {
        irs.push_back(IR_COMM_SCALE);
    }
    if (patternNameStr.find(PATTERN_X1_OFFSET) != std::string::npos) {
        irs.push_back(IR_X1_OFFSET);
    }
    if (patternNameStr.find(PATTERN_X2_OFFSET) != std::string::npos) {
        irs.push_back(IR_X2_OFFSET);
    }
    return irs;
}

// 从 subgraph 边界取 optional 实际接到 mc2 的 IR 下标（无 bias 时也不会把 scale 误判成 bias）
static bool CollectOptionalIrsFromSubgraph(const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                           const ge::GNode &mc2Node, std::vector<int32_t> &optionalIrs)
{
    optionalIrs.clear();
    // subgraph[0]=x1 -> mc2:0, subgraph[1]=x2 -> Transpose；optional 从下标 2 开始
    for (size_t i = 2; i < subgraphInputs.size(); ++i) {
        const auto matchNodes = subgraphInputs[i].GetAllInputs();
        bool found = false;
        for (const auto &nodeIo : matchNodes) {
            if (!IsSameGNode(nodeIo.node, mc2Node)) {
                continue;
            }
            const int32_t irIndex = static_cast<int32_t>(nodeIo.index);
            if (irIndex < IR_BIAS || irIndex > IR_X2_OFFSET) {
                OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Unexpected mc2 input index %d from subgraph[%zu].", irIndex, i);
                return false;
            }
            optionalIrs.push_back(irIndex);
            found = true;
            break;
        }
        if (!found) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Subgraph[%zu] does not connect to captured AlltoAllMatmul.", i);
            return false;
        }
    }
    return true;
}

static bool IsPatternOptionalMatchNode(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::GNode mc2Node;
    if (!GetCapturedMc2Node(matchResult, mc2Node)) {
        return false;
    }
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in IsPatternOptionalMatchNode.");
        return false;
    }
    std::vector<int32_t> connectedIrs;
    if (!CollectOptionalIrsFromSubgraph(subgraphInputs, mc2Node, connectedIrs)) {
        return false;
    }

    ge::AscendString patternName = "";
    matchResult->GetPatternGraph().GetName(patternName);
    const std::string patternNameStr = patternName.GetString() != nullptr ? patternName.GetString() : "";
    const auto patternIrs = GetOptionalIrsFromPatternName(patternNameStr);
    if (patternIrs != connectedIrs) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "Skip pattern=%s: optional IR mismatch (pattern_size=%zu, connected_size=%zu).",
                  patternNameStr.c_str(), patternIrs.size(), connectedIrs.size());
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

    ge::GNode mc2Node;
    if (!GetCapturedMc2Node(matchResult, mc2Node)) {
        return false;
    }
    std::vector<int32_t> optionalIrs;
    if (!CollectOptionalIrsFromSubgraph(subgraphInputs, mc2Node, optionalIrs)) {
        return false;
    }
    if (2U + optionalIrs.size() != inputShapes.size()) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Subgraph input size mismatch, subgraph=%zu, optional=%zu.",
                  inputShapes.size(), optionalIrs.size());
        return false;
    }

    ge::AscendString patternName = "";
    matchResult->GetPatternGraph().GetName(patternName);
    const std::string patternNameStr = patternName.GetString() != nullptr ? patternName.GetString() : "";
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "CreateReplaceGraphInputs pattern=%s, subgraph=%zu, optional=%zu.",
              patternNameStr.c_str(), inputShapes.size(), optionalIrs.size());

    int64_t inputIdx = 0;
    inputs.rX1 = replaceGraphBuilder.CreateInput(inputIdx, "x1", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rX2 = replaceGraphBuilder.CreateInput(inputIdx, "x2", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    ++inputIdx;

    inputs.rBias = nullptr;
    inputs.rX1Scale = nullptr;
    inputs.rX2Scale = nullptr;
    inputs.rCommScale = nullptr;
    inputs.rX1Offset = nullptr;
    inputs.rX2Offset = nullptr;
    for (int32_t irIndex : optionalIrs) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Map subgraph[%ld] -> %s(ir=%d), dtype=%d.", inputIdx, IrInputName(irIndex),
                  irIndex, static_cast<int32_t>(inputDTypes[inputIdx]));
        auto holder = replaceGraphBuilder.CreateInput(inputIdx, IrInputName(irIndex), inputDTypes[inputIdx],
                                                      inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        AssignOptionalInput(inputs, irIndex, holder);
        ++inputIdx;
    }
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "End to create inputs in Replacement.");
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

    // 与 MatmulAlltoAll fusion 一致：attr 全部用局部变量
    ge::AscendString group;
    OP_LOGE_IF(mc2Node.GetAttr("group", group) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group failed.");

    int64_t worldSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("world_size", worldSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr world_size failed.");

    std::vector<int64_t> all2allAxes = {-2, -1};
    OP_LOGE_IF(mc2Node.GetAttr("all2all_axes", all2allAxes) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr all2all_axes failed.");

    int64_t yDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("y_dtype", yDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr y_dtype failed.");

    int64_t x1QuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("x1_quant_mode", x1QuantMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr x1_quant_mode failed.");

    int64_t x2QuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("x2_quant_mode", x2QuantMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr x2_quant_mode failed.");

    int64_t commQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("comm_quant_mode", commQuantMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_quant_mode failed.");

    int64_t x1QuantDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("x1_quant_dtype", x1QuantDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr x1_quant_dtype failed.");

    int64_t commQuantDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("comm_quant_dtype", commQuantDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_quant_dtype failed.");

    bool transposeX1 = false;
    OP_LOGE_IF(mc2Node.GetAttr("transpose_x1", transposeX1) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr transpose_x1 failed.");

    bool transposeX2 = false;
    OP_LOGE_IF(mc2Node.GetAttr("transpose_x2", transposeX2) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr transpose_x2 failed.");
    // 移除 Transpose 节点后，transpose_x2 置反
    transposeX2 = !transposeX2;

    int64_t groupSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("group_size", groupSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group_size failed.");

    ge::AscendString commMode;
    OP_LOGE_IF(mc2Node.GetAttr("comm_mode", commMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_mode failed.");

    bool alltoallOutFlag = true;
    OP_LOGE_IF(mc2Node.GetAttr("alltoall_out_flag", alltoallOutFlag) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr alltoall_out_flag failed.");

    auto mc2 = ge::es::AlltoAllMatmul(inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, inputTensors.rX1Scale,
                                      inputTensors.rX2Scale, inputTensors.rCommScale, inputTensors.rX1Offset,
                                      inputTensors.rX2Offset, group.GetString(), worldSize, all2allAxes, yDtype,
                                      x1QuantMode, x2QuantMode, commQuantMode, x1QuantDtype, commQuantDtype,
                                      transposeX1, transposeX2, groupSize, commMode.GetString(), alltoallOutFlag);
    return replaceGraphBuilder.BuildAndReset({mc2.y, mc2.all2all_out});
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
    for (const auto &info : infos) {
        patternGraphs.emplace_back(MakePattern(info));
    }
    return patternGraphs;
}

bool AllToAllMatmulTransposeA5FusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for AllToAllMatmulTransposeA5FusionPass");

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

    // 同 optional 个数的错误 pattern（如 HasX1Offset 误匹配真实 x2_scale）会导致 FLOAT8 接到 x1_offset
    if (!IsPatternOptionalMatchNode(matchResult)) {
        return false;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found One pattern that meets requirements");
    return true;
}

ge::fusion::GraphUniqPtr AllToAllMatmulTransposeA5FusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement for AllToAllMatmulTransposeA5FusionPass");
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
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }

    return std::move(replaceGraph);
}

REG_FUSION_PASS(AllToAllMatmulTransposeA5FusionPass)
    .Stage(GetAllToAllMatmulTransposeA5FusionPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION
