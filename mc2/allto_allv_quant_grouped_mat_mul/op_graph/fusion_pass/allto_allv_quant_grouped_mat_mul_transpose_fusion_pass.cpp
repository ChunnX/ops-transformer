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
#include "es_TransposeD.h"                  // es autogen header for stub op
#include "es_math_ops.h"                    // math ops stub
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>      // dlopen 动态加载
#include "acl/acl_rt.h" // 运行时判断cann ver

namespace ops {
const std::string FUSION_PASS_NAME = "AlltoAllvQuantGroupedMatMulTransposeFusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_TRANSPOSE_D = "TransposeD";
const std::string PATTERN_HAS_MM = "HasMm";
const std::string PATTERN_GMM_WEIGHT_TRANSPOSE = "GmmWeightTranspose";
const std::string PATTERN_GMM_WEIGHT_SCALE_TRANSPOSE = "GmmWeightScaleTranspose";
const std::string PATTERN_MM_WEIGHT_TRANSPOSE = "MmWeightTranspose";
const std::string PATTERN_MM_WEIGHT_SCALE_TRANSPOSE = "MmWeightScaleTranspose";

// IR 必选：gmm_x / gmm_weight / gmm_x_scale / gmm_weight_scale
const size_t INPUT_NUM_NO_MM = 4;
// 可选 mm 路径齐套：mm_x / mm_weight / mm_x_scale / mm_weight_scale
const size_t INPUT_NUM_HAS_MM = 8;

const int64_t MC2_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_ONE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;
const size_t PERM_SIZE_FOUR = 4;

enum class TransposeNode {
    Transpose,
    TransposeD
};

enum class WeightTransposePort {
    GmmWeight,
    GmmWeightScale,
    MmWeight,
    MmWeightScale
};

// ES matcher：覆盖会改变入边/拓扑的组合
// - hasMm：是否挂可选 mm_* 输入
// - 四路 weight/scale 是否各自挂 Transpose（至少一路为 true）
struct OriginalGraphInfo {
    bool hasMm = false;
    bool gmmWeightTranspose = false;
    bool gmmWeightScaleTranspose = false;
    bool mmWeightTranspose = false;
    bool mmWeightScaleTranspose = false;
    TransposeNode node = TransposeNode::Transpose;
};

// 加载 EsTranspose 符号
namespace {
typedef EsCTensorHolder *(*EsTransposeFunc)(EsCTensorHolder *, EsCTensorHolder *);

EsTransposeFunc GetEsTransposeFunc()
{
    void *handle = dlopen("libes_math.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        OPS_LOG_E("AlltoAllvQuantGroupedMatMulTransposeFusionPass", "dlopen failed: %s", dlerror());
        return nullptr;
    }
    dlerror();
    auto func = reinterpret_cast<EsTransposeFunc>(dlsym(handle, "EsTranspose"));
    if (dlerror() != nullptr) {
        OPS_LOG_E("AlltoAllvQuantGroupedMatMulTransposeFusionPass", "dlsym EsTranspose failed");
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
        OPS_LOG_E("AlltoAllvQuantGroupedMatMulTransposeFusionPass",
                  "EsTranspose is unavailable, fallback to TransposeD.");
        return ge::es::TransposeD(xHolder, std::vector<int64_t>{1, 0});
    }
    auto result = func(xHolder.GetCTensorHolder(), perm.ToTensorHolder(builder).GetCTensorHolder());
    return result;
}

ge::CustomPassStage GetAlltoAllvQuantGroupedMatMulTransposeFusionPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}
} // namespace

static size_t GetPatternInputNum(const OriginalGraphInfo &info)
{
    return info.hasMm ? INPUT_NUM_HAS_MM : INPUT_NUM_NO_MM;
}

static ge::es::EsTensorHolder MakeTransposeNode(const OriginalGraphInfo &info, const ge::es::EsTensorHolder &x)
{
    if (info.node == TransposeNode::Transpose) {
        return TransposeDL(x, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    }
    return ge::es::TransposeD(x, std::vector<int64_t>{1, 0});
}

static bool HasAnyWeightTranspose(const OriginalGraphInfo &info)
{
    return info.gmmWeightTranspose || info.gmmWeightScaleTranspose || info.mmWeightTranspose ||
           info.mmWeightScaleTranspose;
}

static ge::fusion::PatternUniqPtr MakePattern(const OriginalGraphInfo &info)
{
    std::string patternName = FUSION_PASS_NAME;
    patternName += info.node == TransposeNode::Transpose ? PATTERN_TRANSPOSE : PATTERN_TRANSPOSE_D;
    if (info.hasMm) {
        patternName += PATTERN_HAS_MM;
    }
    if (info.gmmWeightTranspose) {
        patternName += PATTERN_GMM_WEIGHT_TRANSPOSE;
    }
    if (info.gmmWeightScaleTranspose) {
        patternName += PATTERN_GMM_WEIGHT_SCALE_TRANSPOSE;
    }
    if (info.mmWeightTranspose) {
        patternName += PATTERN_MM_WEIGHT_TRANSPOSE;
    }
    if (info.mmWeightScaleTranspose) {
        patternName += PATTERN_MM_WEIGHT_SCALE_TRANSPOSE;
    }

    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    // 必选 gmm_*；可选 mm_*；send/recv counts 不进 pattern（走 attr / nullptr）
    const size_t inputNum = GetPatternInputNum(info);
    auto inputs = graphBuilder.CreateInputs(inputNum);
    size_t idx = 0;
    auto gmmX = inputs[idx++];
    auto gmmWeight = inputs[idx++];
    auto gmmXScale = inputs[idx++];
    auto gmmWeightScale = inputs[idx++];

    ge::es::EsTensorHolder mmX = nullptr;
    ge::es::EsTensorHolder mmWeight = nullptr;
    ge::es::EsTensorHolder mmXScale = nullptr;
    ge::es::EsTensorHolder mmWeightScale = nullptr;
    if (info.hasMm) {
        mmX = inputs[idx++];
        mmWeight = inputs[idx++];
        mmXScale = inputs[idx++];
        mmWeightScale = inputs[idx++];
    }

    ge::es::EsTensorHolder transposeGmmWeight = nullptr;
    ge::es::EsTensorHolder gmmWeightIn = gmmWeight;
    if (info.gmmWeightTranspose) {
        transposeGmmWeight = MakeTransposeNode(info, gmmWeight);
        gmmWeightIn = transposeGmmWeight;
    }

    ge::es::EsTensorHolder transposeGmmWeightScale = nullptr;
    ge::es::EsTensorHolder gmmWeightScaleIn = gmmWeightScale;
    if (info.gmmWeightScaleTranspose) {
        transposeGmmWeightScale = MakeTransposeNode(info, gmmWeightScale);
        gmmWeightScaleIn = transposeGmmWeightScale;
    }

    ge::es::EsTensorHolder transposeMmWeight = nullptr;
    ge::es::EsTensorHolder mmWeightIn = mmWeight;
    if (info.mmWeightTranspose) {
        transposeMmWeight = MakeTransposeNode(info, mmWeight);
        mmWeightIn = transposeMmWeight;
    }

    ge::es::EsTensorHolder transposeMmWeightScale = nullptr;
    ge::es::EsTensorHolder mmWeightScaleIn = mmWeightScale;
    if (info.mmWeightScaleTranspose) {
        transposeMmWeightScale = MakeTransposeNode(info, mmWeightScale);
        mmWeightScaleIn = transposeMmWeightScale;
    }

    const char *group = "";
    const int64_t epWorldSize = 0;
    const std::vector<int64_t> sendCounts;
    const std::vector<int64_t> recvCounts;
    const int64_t gmmXQuantMode = 0;
    const int64_t gmmWeightQuantMode = 0;
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        gmmX, gmmWeightIn, gmmXScale, gmmWeightScaleIn, nullptr, nullptr, mmX, mmWeightIn, mmXScale, mmWeightScaleIn,
        group, epWorldSize, sendCounts, recvCounts, gmmXQuantMode, gmmWeightQuantMode);

    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y, mc2.mm_y, mc2.permute_out});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    // Capture：0=mc2；其后按固定顺序追加本 pattern 中存在的 Transpose，便于 MeetRequirements 校验 perm
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0});
    if (info.gmmWeightTranspose) {
        pattern->CaptureTensor({*transposeGmmWeight.GetProducer(), 0});
    }
    if (info.gmmWeightScaleTranspose) {
        pattern->CaptureTensor({*transposeGmmWeightScale.GetProducer(), 0});
    }
    if (info.mmWeightTranspose) {
        pattern->CaptureTensor({*transposeMmWeight.GetProducer(), 0});
    }
    if (info.mmWeightScaleTranspose) {
        pattern->CaptureTensor({*transposeMmWeightScale.GetProducer(), 0});
    }
    return pattern;
}

std::vector<ge::fusion::PatternUniqPtr> AlltoAllvQuantGroupedMatMulTransposeFusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns for AlltoAllvQuantGroupedMatMulTransposeFusionPass");
    // hasMm × 四路 Transpose 开关（至少一路）× Transpose/TransposeD
    // 无 mm：有效开关组合 3 种 × 2 = 6；有 mm：15 种 × 2 = 30；共 36 种
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    for (bool hasMm : {false, true}) {
        for (bool gmmWeightTranspose : {false, true}) {
            for (bool gmmWeightScaleTranspose : {false, true}) {
                for (bool mmWeightTranspose : {false, true}) {
                    for (bool mmWeightScaleTranspose : {false, true}) {
                        if (!hasMm && (mmWeightTranspose || mmWeightScaleTranspose)) {
                            continue;
                        }
                        OriginalGraphInfo info{hasMm, gmmWeightTranspose, gmmWeightScaleTranspose, mmWeightTranspose,
                                               mmWeightScaleTranspose, TransposeNode::Transpose};
                        if (!HasAnyWeightTranspose(info)) {
                            continue;
                        }
                        for (auto node : {TransposeNode::Transpose, TransposeNode::TransposeD}) {
                            info.node = node;
                            patternGraphs.emplace_back(MakePattern(info));
                        }
                    }
                }
            }
        }
    }
    return patternGraphs;
}

static bool GetTransposePerm(const ge::GNode &transposeNode, std::vector<int64_t> &permValue)
{
    ge::AscendString nodeType("");
    transposeNode.GetType(nodeType);
    std::string nodeTypeStr = nodeType.GetString();
    if (nodeTypeStr == PATTERN_TRANSPOSE_D) {
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

static bool IsWeightTransposePermValid(const std::vector<int64_t> &permValue)
{
    const size_t permSize = permValue.size();
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE && permSize != PERM_SIZE_FOUR) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "weight transpose permutation size must be 2/3/4, but got %zu.", permSize);
        return false;
    }
    // 最后两维必须互换
    if ((permValue[permSize - PERM_SIZE_ONE] != static_cast<int64_t>(permSize - PERM_SIZE_TWO)) ||
        (permValue[permSize - PERM_SIZE_TWO] != static_cast<int64_t>(permSize - PERM_SIZE_ONE))) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "weight transpose permutation last 2 dims must be swapped.");
        return false;
    }
    return true;
}

static bool IsGmmWeightScaleTransposePermValid(const std::vector<int64_t> &permValue)
{
    if (permValue.size() < PERM_SIZE_THREE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "gmm_weight_scale transpose permutation size must be at least 3, but got %zu.", permValue.size());
        return false;
    }
    // canndev：perm[1]==2 && perm[2]==1
    if (permValue[1] != static_cast<int64_t>(2) || permValue[2] != static_cast<int64_t>(1)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "gmm_weight_scale transpose perm[1]/perm[2] must be 2/1.");
        return false;
    }
    return true;
}

static bool IsMmWeightScaleTransposePermValid(const std::vector<int64_t> &permValue)
{
    if (permValue.size() < PERM_SIZE_TWO) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "mm_weight_scale transpose permutation size must be at least 2, but got %zu.", permValue.size());
        return false;
    }
    // canndev：perm[0]==1 && perm[1]==0
    if (permValue[0] != static_cast<int64_t>(1) || permValue[1] != static_cast<int64_t>(0)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "mm_weight_scale transpose first 2 dims must be [1, 0].");
        return false;
    }
    return true;
}

static bool ValidateCapturedTransposePerm(const std::unique_ptr<ge::fusion::MatchResult> &matchResult,
                                          int64_t captureIdx, WeightTransposePort port)
{
    ge::fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(captureIdx, transposeOutput) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get captured Transpose node failed, capture idx %ld.", captureIdx);

    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeOutput.node, permValue)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation, capture idx %ld.", captureIdx);
        return false;
    }

    switch (port) {
        case WeightTransposePort::GmmWeight:
        case WeightTransposePort::MmWeight:
            return IsWeightTransposePermValid(permValue);
        case WeightTransposePort::GmmWeightScale:
            return IsGmmWeightScaleTransposePermValid(permValue);
        case WeightTransposePort::MmWeightScale:
            return IsMmWeightScaleTransposePermValid(permValue);
    }
    return false;
}

static bool IsTransposePermValid(const std::unique_ptr<ge::fusion::MatchResult> &matchResult,
                                 const std::string &patternNameStr)
{
    // Capture 顺序与 MakePattern 一致：mc2=0，其后按 gmm_w → gmm_ws → mm_w → mm_ws
    int64_t captureIdx = MC2_CAPTURE_IDX + 1;
    if (patternNameStr.find(PATTERN_GMM_WEIGHT_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, WeightTransposePort::GmmWeight)) {
            return false;
        }
        ++captureIdx;
    }
    if (patternNameStr.find(PATTERN_GMM_WEIGHT_SCALE_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, WeightTransposePort::GmmWeightScale)) {
            return false;
        }
        ++captureIdx;
    }
    if (patternNameStr.find(PATTERN_MM_WEIGHT_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, WeightTransposePort::MmWeight)) {
            return false;
        }
        ++captureIdx;
    }
    if (patternNameStr.find(PATTERN_MM_WEIGHT_SCALE_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, WeightTransposePort::MmWeightScale)) {
            return false;
        }
    }
    return true;
}

bool AlltoAllvQuantGroupedMatMulTransposeFusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for AlltoAllvQuantGroupedMatMulTransposeFusionPass");

    // 9.0.0 版本前运行降级stage 空跑
    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip in MeetRequirements when cann version not compatible.");
        return false;
    }

    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Check target platform fail!");
        return false;
    }

    ge::AscendString patternName = "";
    matchResult->GetPatternGraph().GetName(patternName);
    const std::string patternNameStr = patternName.GetString() != nullptr ? patternName.GetString() : "";
    const bool hasMm = patternNameStr.find(PATTERN_HAS_MM) != std::string::npos;
    const size_t expectNum = hasMm ? INPUT_NUM_HAS_MM : INPUT_NUM_NO_MM;

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in MeetRequirements.");
        return false;
    }
    if (subgraphInputs.size() != expectNum) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip pattern=%s: subgraph size=%zu, expect=%zu.", patternNameStr.c_str(),
                  subgraphInputs.size(), expectNum);
        return false;
    }

    if (!IsTransposePermValid(matchResult, patternNameStr)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation is not valid.");
        return false;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found One pattern that meets requirements");

    return true;
}

ge::fusion::GraphUniqPtr AlltoAllvQuantGroupedMatMulTransposeFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement for AlltoAllvQuantGroupedMatMulTransposeFusionPass");
    (void)matchResult;
    // TODO: BuildReplaceGraph（吸收 Transpose 后翻转 trans_gmm_weight / trans_mm_weight）
    return nullptr;
}

REG_FUSION_PASS(AlltoAllvQuantGroupedMatMulTransposeFusionPass)
    .Stage(GetAlltoAllvQuantGroupedMatMulTransposeFusionPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION
