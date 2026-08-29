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
 * \file mega_moe_tiling.cpp
 * \brief
 */

#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <unordered_map>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_exception_dump.h"
#include "../mega_moe_tiling_host.h"
#include "../../../op_kernel/arch35/mega_moe_tiling.h"
#include "../../../op_kernel/arch35/mega_moe_tiling_key.h"
#include "../../../op_kernel/arch35/common/mega_moe_constants.h"
#include "../../../op_kernel/arch35/common/mega_moe_workspace.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
using namespace MegaMoeImpl;

namespace optiling {
namespace {
// SwiGLU 门控把 GMM1 的输出劈成 gate 和 up 两半，所以 weight1 的行数是 weight2 列数的两倍。
constexpr int64_t SWIGLU_GATE_UP_SPLIT_FACTOR = 2LL;
// MX 量化的 weight scale 每个专家末尾固定带 2 个 base 分量。
constexpr int64_t WEIGHT_SCALE_MULTI_BASE_DIM_SIZE = 2LL;
// 共享专家最多 4 个。
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4LL;
constexpr int64_t UB_BLOCK_SIZE = 32LL;

constexpr int64_t FOUR_DIMS = 4LL;
constexpr int64_t THREE_DIMS = 3LL;
constexpr int64_t TWO_DIMS = 2LL;
constexpr int64_t ONE_DIM = 1LL;
constexpr int64_t MIN_TOPK = 1LL;
constexpr int64_t MAX_TOPK = 32LL;
constexpr int64_t MIN_EXPERT_PER_RANK = 1LL;
constexpr int64_t MAX_EXPERT_PER_RANK = 1024LL;
constexpr int64_t MIN_H = 1024LL;
constexpr int64_t MAX_H = 8LL * 1024LL; // 8K
constexpr int64_t H_ALIGN = 32LL;
constexpr int64_t W4_K_ALIGN = 64LL;
constexpr int64_t HIDDEN_DIM_BASE = 1024LL;
constexpr int64_t MAX_HIDDEN_DIM = 8LL * 1024LL; // 8K
// MTE 路径 N 泛化：hiddenDim(=GMM1 输出维，含 gate/up 两半) 按 GMM_TILE_N(256) 粒度放宽，
// 下限 512 保证非交织模式 gate/up 半宽至少覆盖 1 个完整 256 tile；URMA 路径维持 1K 约束不变。
constexpr int64_t MTE_MIN_HIDDEN_DIM = 512LL;
constexpr int64_t MTE_HIDDEN_DIM_ALIGN = 256LL;
constexpr int64_t MIN_EP_WORLD_SIZE = 2LL;
constexpr int64_t MAX_EP_WORLD_SIZE = 1024LL;
constexpr int64_t MAX_MOE_EXPERT_NUM = 2048LL;
constexpr int64_t INPUT_WEIGHT_SCALES_CEIL_ALIGN = 64LL;
constexpr int64_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 50LL;
constexpr float DEFAULT_ACTIVATION_CLAMP = std::numeric_limits<float>::max();
constexpr float DEFAULT_SWIGLU_OAI_ALPHA = 1.702f;
constexpr float DEFAULT_SWIGLU_OAI_BETA = 1.0f;

constexpr uint32_t GMM_TILE_N = 256U;
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE = 4U;
constexpr uint32_t WEIGHT_MATRIX_ROW_DIM_INDEX = 0U;
constexpr uint32_t WEIGHT_MATRIX_COLUMN_DIM_INDEX = 1U;
constexpr uint32_t WEIGHT_SCALE_MATRIX_DIM_INDEX = 0U;
constexpr uint32_t WEIGHT_SCALE_GROUP_DIM_INDEX = 1U;
constexpr uint32_t WEIGHT_SCALE_MULTI_BASE_DIM_INDEX = 2U;
constexpr uint8_t INVALID_GMM_MODE = std::numeric_limits<uint8_t>::max();

static const std::unordered_map<ge::DataType, int64_t> EXPERT_QUANT_DTYPE_TO_MODE{
    {ge::DT_FLOAT8_E5M2, EXPERT_QUANT_OUT_DTYPE_E5M2},
    {ge::DT_FLOAT8_E4M3FN, EXPERT_QUANT_OUT_DTYPE_E4M3FN},
    {ge::DT_FLOAT4_E2M1, EXPERT_QUANT_OUT_DTYPE_E2M1},
};

struct ExpertWeightTensorGroupInputs {
    uint32_t weightOneIndex;
    uint32_t weightTwoIndex;
    uint32_t weightScalesOneIndex;
    uint32_t weightScalesTwoIndex;
    const char *expertTypeName;
};

struct ExpertParams {
    ExpertWeightTensorGroupInputs inputs;
    int64_t expertCount;
    ge::DataType activationDtype;
    uint8_t gmmMode;
};

struct MegaMoeExpertParams {
    ExpertParams moe;
    ExpertParams shared;
    bool isPerExpertWeightTensor;
};

uint32_t CalcMGroupsPerWave(const MegaMoeTilingData *tilingData, uint32_t aicNum)
{
    if (tilingData->hiddenDim == 0U || tilingData->h == 0U || aicNum == 0U) {
        return 1U;
    }

    /*
     * hiddenDim 包含 gate/up 两部分。交织模式每核至少调度 4 个独立 N tile；非交织模式
     * 每个物理任务成对处理 gate/up，原先每核 2 个物理任务同样等价于 4 个逻辑 N tile。
     * hiddenDim 已校验为 GMM_TILE_N(256) 的倍数（MTE；URMA 仍为 1024），CeilDiv 对
     * 非交织半宽产生的尾 tile 只影响 wave 粒度估算，不影响正确性，两种编译模式共用本公式。
     */
    uint64_t gmm1LogicalTilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->hiddenDim, GMM_TILE_N);
    uint64_t gmm2TilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->h, GMM_TILE_N);
    uint64_t gmm1RequiredMGroups = ops::CeilDiv<uint64_t>(
        static_cast<uint64_t>(aicNum) * GMM1_MIN_LOGICAL_TILES_PER_CORE, gmm1LogicalTilesPerMGroup);
    uint64_t gmm2RequiredMGroups = ops::CeilDiv<uint64_t>(static_cast<uint64_t>(aicNum), gmm2TilesPerMGroup);
    return static_cast<uint32_t>(std::max(gmm1RequiredMGroups, gmm2RequiredMGroups));
}

/*
 * 统计指定动态输入 (tensor list) 中的 tensor 数量。
 */
static uint32_t GetDynamicInputTensorCount(const gert::TilingContext *context, uint32_t inputIndex)
{
    uint32_t tensorCount = 0;
    while (context->GetDynamicInputShape(inputIndex, tensorCount) != nullptr) {
        ++tensorCount;
    }
    return tensorCount;
}

/*
 * 按权重布局返回专家数：逐专家布局取 TensorList 长度，堆叠布局取首个 tensor 的 dim0。
 * 可选输入未传入或首维为 0 时返回 0。
 */
static int64_t GetWeightExpertCount(const gert::TilingContext *context, uint32_t inputIndex,
                                    bool isPerExpertWeightTensor)
{
    const auto *firstTensorShape = context->GetDynamicInputShape(inputIndex, 0);
    if (firstTensorShape == nullptr || firstTensorShape->GetStorageShape().GetDimNum() == 0U ||
        firstTensorShape->GetStorageShape().GetDim(0) <= 0) {
        return 0;
    }
    if (isPerExpertWeightTensor) {
        return static_cast<int64_t>(GetDynamicInputTensorCount(context, inputIndex));
    }
    return firstTensorShape->GetStorageShape().GetDim(0);
}

/*
 * 以单专家视图读取指定维度；堆叠布局会自动跳过最外层的专家维。
 */
static int64_t GetSingleExpertTensorDimSize(const gert::StorageShape *tensorShape, uint32_t singleExpertDimIndex,
                                            bool isPerExpertTensor)
{
    uint32_t expertDimOffset = isPerExpertTensor ? 0U : 1U;
    return tensorShape->GetStorageShape().GetDim(expertDimOffset + singleExpertDimIndex);
}
} // namespace

void PrintMegaMoeTilingData(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);

    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");
    OP_LOGD(nodeName,
            "shape: bs=%u, numMaxTokensPerRank=%u, h=%u, hiddenDim=%u, topK=%u, maxOutputSize=%u, "
            "isPerExpertWeightTensor=%d",
            tilingData->bs, tilingData->numMaxTokensPerRank, tilingData->h, tilingData->hiddenDim, tilingData->topK,
            tilingData->maxOutputSize, tilingData->isPerExpertWeightTensor);
    OP_LOGD(nodeName,
            "topology: moeExpertPerRank=%u, sharedExpertNum=%u, epWorldSize=%u, aicNum=%u, blockAivNum=%u, "
            "blockNumPerEP=%u, topoType=%ld",
            tilingData->moeExpertPerRank, tilingData->sharedExpertNum, tilingData->epWorldSize, tilingData->aicNum,
            tilingData->blockAivNum, tilingData->blockNumPerEP, tilingData->topoType);
    OP_LOGD(nodeName, "mode: moeGmmMode=%u, sharedGmmMode=%u, combineQuantMode=%ld, clampLimit=%f",
            static_cast<uint32_t>(tilingData->moeGmmMode), static_cast<uint32_t>(tilingData->sharedGmmMode),
            tilingData->combineQuantMode, tilingData->clampLimit);
    OP_LOGD(nodeName, "combineSync: slotCountPerExpert=%lu", tilingData->combineSyncSlotCountPerExpert);

    const auto &dispatchConfig = tilingData->dispatchBufferConfig;
    OP_LOGD(nodeName, "dispatch: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, copyBufferBytes=%u",
            dispatchConfig.routeItemsPerBatch, dispatchConfig.routeBatchCount, dispatchConfig.bufferCount,
            dispatchConfig.copyBufferBytes);

    const auto &sendMaskConfigWithExtraExpert = tilingData->sendMaskConfigForCoreWithExtraExpert;
    const auto &sendMaskConfigWithoutExtraExpert = tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    OP_LOGD(nodeName, "sendMask: coreCountWithExtraExpert=%u", tilingData->sendMaskCoreCountWithExtraExpert);
    OP_LOGD(nodeName,
            "sendMaskWithExtraExpert: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, bufferBytes=%u",
            sendMaskConfigWithExtraExpert.routeItemsPerBatch, sendMaskConfigWithExtraExpert.routeBatchCount,
            sendMaskConfigWithExtraExpert.bufferCount, sendMaskConfigWithExtraExpert.bufferBytes);
    OP_LOGD(nodeName,
            "sendMaskWithoutExtraExpert: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, bufferBytes=%u",
            sendMaskConfigWithoutExtraExpert.routeItemsPerBatch, sendMaskConfigWithoutExtraExpert.routeBatchCount,
            sendMaskConfigWithoutExtraExpert.bufferCount, sendMaskConfigWithoutExtraExpert.bufferBytes);

    const auto &unpermuteFullChunkConfig = tilingData->unpermuteConfigForFullTokenChunk;
    const auto &unpermuteTailChunkConfig = tilingData->unpermuteConfigForTailTokenChunk;
    OP_LOGD(nodeName, "unpermute: fullTokenChunkCoreCount=%u", tilingData->unpermuteFullTokenChunkCoreCount);
    OP_LOGD(nodeName,
            "unpermuteFullChunk: tokensPerBatch=%d, inputBufferCount=%d, bf16SlotElements=%u, fp32SlotElements=%u, "
            "weightBufferBytes=%u, conversionBufferBytes=%u",
            unpermuteFullChunkConfig.tokensPerBatch, unpermuteFullChunkConfig.inputBufferCount,
            unpermuteFullChunkConfig.bf16SlotElementCount, unpermuteFullChunkConfig.fp32SlotElementCount,
            unpermuteFullChunkConfig.topKWeightsBufferBytes, unpermuteFullChunkConfig.topKWeightsConversionBufferBytes);
    OP_LOGD(nodeName,
            "unpermuteTailChunk: tokensPerBatch=%d, inputBufferCount=%d, bf16SlotElements=%u, fp32SlotElements=%u, "
            "weightBufferBytes=%u, conversionBufferBytes=%u",
            unpermuteTailChunkConfig.tokensPerBatch, unpermuteTailChunkConfig.inputBufferCount,
            unpermuteTailChunkConfig.bf16SlotElementCount, unpermuteTailChunkConfig.fp32SlotElementCount,
            unpermuteTailChunkConfig.topKWeightsBufferBytes, unpermuteTailChunkConfig.topKWeightsConversionBufferBytes);
    OP_LOGD(nodeName, "topkWeightsPrefetch is %d", tilingData->topkWeightsPrefetch);
    OP_LOGD(nodeName, "mGroupsPerWave is %u", tilingData->mGroupsPerWave);
}

void PrintWorkspaceInfo(const struct WorkspaceInfo *info, const char *nodeName)
{
    OP_LOGD(nodeName, "dispatchRevDataPtr:         %ld\n", info->dispatchRevDataPtr);
    OP_LOGD(nodeName, "dispatchRevScalePtr:        %ld\n", info->dispatchRevScalePtr);
    OP_LOGD(nodeName, "activationQuantDataPtr:         %ld\n", info->activationQuantDataPtr);
    OP_LOGD(nodeName, "activationQuantScalePtr:        %ld\n", info->activationQuantScalePtr);
    OP_LOGD(nodeName, "expertRevTokenNumsPtr:      %ld\n", info->expertRevTokenNumsPtr);
    OP_LOGD(nodeName, "metaInfoPtr:                %ld\n", info->metaInfoPtr);
    OP_LOGD(nodeName, "flagActivationToGmm2Ptr:        %ld\n", info->flagActivationToGmm2Ptr);
    OP_LOGD(nodeName, "flagDispatchToGmm1Ptr:      %ld\n", info->flagDispatchToGmm1Ptr);
    OP_LOGD(nodeName, "flagSendCntCalToUpdParamsPtr:      %ld\n", info->flagSendCntCalToUpdParamsPtr);
    OP_LOGD(nodeName, "flagGmmToEpiloguePtr: %ld\n", info->flagGmmToEpiloguePtr);
    OP_LOGD(nodeName, "gmm2ReadyPtr:               %ld\n", info->gmm2ReadyPtr);
    OP_LOGD(nodeName, "gmm2CombineSyncCounterPtr:  %ld\n", info->gmm2CombineSyncCounterPtr);
    OP_LOGD(nodeName, "gmm2MmadResPtr:             %ld\n", info->gmm2MmadResPtr);
    OP_LOGD(nodeName, "workspaceSize:              %ld\n", info->workspaceSize);
}

void PrintPeermemInfo(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_LOGD(nodeName, "========== PeermemInfo ==========");
    int64_t exceptionDumpRegionSize = tilingData->topoType == TOPO_TYPE_MTE ? EXCEPTION_DUMP_REGION_SIZE : 0;
    OP_LOGD(nodeName, "exceptionDumpRegionSize: {%ld}\n", exceptionDumpRegionSize);
    // 各区尺寸取自 kernel 侧同一份布局公式（common/mega_moe_peermem.h）。
    int64_t maskAlignSize = CalcDispatchMaskAlignSize(tilingData);
    int64_t maskRecvSize = CalcMaskRecvSize(maskAlignSize, static_cast<int64_t>(tilingData->moeExpertPerRank),
                                            static_cast<int64_t>(tilingData->epWorldSize));
    int64_t tokenScaleBytes =
        CalcQuantTokenScaleBytes(static_cast<int64_t>(tilingData->h), 1U, static_cast<int64_t>(tilingData->topK),
                                 tilingData->topkWeightsPrefetch == 1);
    int64_t quantTokenScaleSize = ops::CeilAlign(
        static_cast<int64_t>(tilingData->numMaxTokensPerRank) * tokenScaleBytes, static_cast<int64_t>(ALIGN_512));
    // 非量化 combine 每个 hidden 元素存一个 BF16（2 字节）。
    int64_t combineTokenBytes =
        CalcCombineTokenBytes(static_cast<int64_t>(tilingData->h), 2, tilingData->combineQuantMode != COMBINE_NO_QUANT);
    int64_t sendTotalNum = static_cast<int64_t>(tilingData->numMaxTokensPerRank) * tilingData->topK;
    int64_t combineSendSize = ops::CeilAlign(sendTotalNum * combineTokenBytes, static_cast<int64_t>(ALIGN_512));
    OP_LOGD(nodeName, "rankSyncInWorldSize: {%ld}\n", PEERMEM_DATA_OFFSET);
    OP_LOGD(nodeName, "maskRecvSize: {%ld}\n", maskRecvSize);
    OP_LOGD(nodeName, "quantTokenScaleSize: {%ld}\n", quantTokenScaleSize);
    OP_LOGD(nodeName, "combineSendSize: {%ld}\n", combineSendSize);
    OP_LOGD(nodeName, "total PeermemInfo Size: {%ld}\n",
            exceptionDumpRegionSize + PEERMEM_DATA_OFFSET + maskRecvSize + quantTokenScaleSize + combineSendSize);
}

static ge::graphStatus CheckAndSetSharedExpertCount(const gert::TilingContext *context, MegaMoeExpertParams &params,
                                                    const char *nodeName)
{
    int64_t weightOneExpertCount =
        GetWeightExpertCount(context, params.shared.inputs.weightOneIndex, params.isPerExpertWeightTensor);
    bool hasWeightOne = weightOneExpertCount > 0;
    bool hasWeightTwo =
        GetWeightExpertCount(context, params.shared.inputs.weightTwoIndex, params.isPerExpertWeightTensor) > 0;
    bool hasScaleOne =
        GetWeightExpertCount(context, params.shared.inputs.weightScalesOneIndex, params.isPerExpertWeightTensor) > 0;
    bool hasScaleTwo =
        GetWeightExpertCount(context, params.shared.inputs.weightScalesTwoIndex, params.isPerExpertWeightTensor) > 0;
    bool presenceMismatch = hasWeightOne != hasWeightTwo || hasWeightOne != hasScaleOne || hasWeightOne != hasScaleTwo;
    std::string presenceValues = "[" + std::to_string(hasWeightOne) + ", " + std::to_string(hasWeightTwo) + ", " +
                                 std::to_string(hasScaleOne) + ", " + std::to_string(hasScaleTwo) + "]";
    OP_TILING_CHECK(presenceMismatch,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        nodeName, "shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2",
                        presenceValues.c_str(), "Shared expert weights and weight scales must be provided together."),
                    return ge::GRAPH_FAILED);
    params.shared.expertCount = hasWeightOne ? weightOneExpertCount : 0;
    return ge::GRAPH_SUCCESS;
}

static int64_t GetCombineQuantModeByAttr(const gert::TilingContext *context, MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    if (combineQuantModePtr == nullptr) {
        return COMBINE_QUANT_OUT_TYPE_NO_QUANT;
    }
    return static_cast<int64_t>(*combineQuantModePtr);
}

static uint64_t CalTilingKey(const gert::TilingContext *context, MegaMoeConfig &config,
                             const MegaMoeExpertParams &expertParams)
{
    auto attrs = context->GetAttrs();
    auto moeWeightDesc = context->GetDynamicInputDesc(expertParams.moe.inputs.weightOneIndex, 0);
    auto sharedWeightDesc = expertParams.shared.expertCount > 0 ?
                                context->GetDynamicInputDesc(expertParams.shared.inputs.weightOneIndex, 0) :
                                moeWeightDesc;

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t combineQuantMode = GetCombineQuantModeByAttr(context, config);
    int64_t topoType = TILINGKEY_TPL_MTE;
    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    return GET_TPL_TILING_KEY(
        static_cast<int64_t>(moeWeightDesc->GetDataType()), static_cast<int64_t>(sharedWeightDesc->GetDataType()),
        static_cast<int64_t>(*dispatchQuantModePtr), EXPERT_QUANT_DTYPE_TO_MODE.at(expertParams.moe.activationDtype),
        EXPERT_QUANT_DTYPE_TO_MODE.at(expertParams.shared.activationDtype), combineQuantMode, topoType,
        topkWeightsType);
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    auto sharedExpertQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrSharedExpertQuantOutDtypeIndex));
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    auto topoTypePtr = attrs->GetAttrPointer<int64_t>((config.attrTopoTypeIndex));
    auto topkWeightsTypePtr = attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);

    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr || *cclBufferSizePtr < 0,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "cclBufferSize"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxRecvTokenNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "maxRecvTokenNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dispatchQuantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantOutDtypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dispatchQuantOutDtype"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertQuantOutDtypePtr == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "sharedExpertQuantOutDtype"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(combineQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "combineQuantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commAlg"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numMaxTokensPerRankPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numMaxTokensPerRank"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topoTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topoType"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkWeightsType"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(activationPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "activation"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(activationParamsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "activationParams"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * swiglu / swiglustep 只认一个可选参数 clamp——激活值的截断上限。
 * 不传就是不截断（默认取 float 最大值，见 SetActivationTilingData）。
 */
static ge::graphStatus CheckSwiGluActivationParams(const float *activationParams, size_t paramCount,
                                                   const char *nodeName)
{
    OP_TILING_CHECK(paramCount > 1U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize", std::to_string(paramCount).c_str(),
                                              "0 or 1 for swiglu/swiglustep"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 1U) {
        const float activationClamp = activationParams[0];
        OP_TILING_CHECK(activationClamp < 0.0f || std::isnan(activationClamp),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationClamp", std::to_string(activationClamp).c_str(),
                                                  "should be >= 0 and not NAN"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * swigluoai 要 clamp、alpha、beta 三个参数，三个一起给、或者一个都不给。
 * 不接受给一半：缺的那个没有单独的默认值可退，只有一个都不给时才整体走默认。
 */
static ge::graphStatus CheckSwiGluOaiActivationParams(const float *activationParams, size_t paramCount,
                                                      const char *nodeName)
{
    OP_TILING_CHECK(paramCount != 0U && paramCount != 3U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize", std::to_string(paramCount).c_str(),
                                              "0 or 3 for swigluoai [clamp, alpha, beta]"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 3U) {
        const float activationClamp = activationParams[0];
        OP_TILING_CHECK(activationClamp < 0.0f || std::isnan(activationClamp),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationClamp", std::to_string(activationClamp).c_str(),
                                                  ">= 0 and not NAN"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(!std::isfinite(activationParams[1]),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationAlpha",
                                                  std::to_string(activationParams[1]).c_str(), "finite"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(!std::isfinite(activationParams[2]),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationBeta",
                                                  std::to_string(activationParams[2]).c_str(), "finite"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * situglu 的 beta 必填：它要进分母，取 0 或非有限值算出来就是 NaN。
 * linear_beta 选填，给了就切到 LINEAR 子模式。
 * 注意个数校验必须排在取值之前：一个参数都不传时 activationParams 是空指针，
 * 先取 activationParams[0] 会直接段错误，而不是干净地报错返回。
 */
static ge::graphStatus CheckSituGluActivationParams(const float *activationParams, size_t paramCount,
                                                    const char *nodeName)
{
    OP_TILING_CHECK(paramCount < 1U || paramCount > 2U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize", std::to_string(paramCount).c_str(),
                                              "should be 1 or 2 for situglu"),
                    return ge::GRAPH_FAILED);
    const float beta = activationParams[0];
    OP_TILING_CHECK(!std::isfinite(beta) || beta == 0.0f,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "situglu_beta", std::to_string(beta).c_str(),
                                              "should be finite and non-zero"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 2U) {
        const float linearBeta = activationParams[1];
        OP_TILING_CHECK(!std::isfinite(linearBeta) || linearBeta == 0.0f,
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "situglu_linear_beta", std::to_string(linearBeta).c_str(),
                                                  "should be finite and non-zero"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * URMA 路径的 kernel 目前只实现了 swiglu，别的激活还没接上。
 * 在 host 就拦掉，免得跑到板上才发现这条路走不通。
 */
static ge::graphStatus CheckActivationTopoConstraint(int64_t topoType, const char *activationPtr, const char *nodeName)
{
    OP_TILING_CHECK(topoType == TOPO_TYPE_URMA && std::strcmp(activationPtr, "swiglu") != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr, "'swiglu' for URMA topology"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验通过后，把同一份激活属性写入 tilingData，避免校验和设置阶段重复读取 attrs。
 * 默认值必须先写入：swiglu 直接使用默认值，其余模式只覆盖各自需要的字段。
 */
static void SetActivationTilingData(const char *activation, const float *activationParams, size_t paramCount,
                                    MegaMoeTilingData *tilingData)
{
    tilingData->clampLimit = paramCount == 0U ? DEFAULT_ACTIVATION_CLAMP : activationParams[0];
    tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU);
    tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::DEFAULT);
    tilingData->activationAlpha = 1.0f;
    tilingData->activationBeta = 1.0f;

    if (std::strcmp(activation, "situglu") == 0) {
        tilingData->clampLimit = std::numeric_limits<float>::max();
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SITU);
        tilingData->activationBeta = activationParams[0];
        if (paramCount == 2U) {
            tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::LINEAR);
            tilingData->activationAlpha = activationParams[1];
        } else {
            tilingData->activationAlpha = 0.0f;
        }
    } else if (std::strcmp(activation, "swiglustep") == 0) {
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU_STEP);
    } else if (std::strcmp(activation, "swigluoai") == 0) {
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU_OAI);
        tilingData->activationAlpha = paramCount == 0U ? DEFAULT_SWIGLU_OAI_ALPHA : activationParams[1];
        tilingData->activationBeta = paramCount == 0U ? DEFAULT_SWIGLU_OAI_BETA : activationParams[2];
    }
}

static ge::graphStatus CheckAndSetActivationParams(const gert::TilingContext *context, MegaMoeConfig &config,
                                                   MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);
    const size_t paramCount = activationParamsPtr->GetSize();
    const float *activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();
    OP_TILING_CHECK(paramCount != 0U && activationParams == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "activationParamsData"), return ge::GRAPH_FAILED);

    const bool isSwiGlu = std::strcmp(activationPtr, "swiglu") == 0;
    const bool isSwiGluStep = std::strcmp(activationPtr, "swiglustep") == 0;
    const bool isSwiGluOai = std::strcmp(activationPtr, "swigluoai") == 0;
    const bool isSituGlu = std::strcmp(activationPtr, "situglu") == 0;
    OP_TILING_CHECK(!isSwiGlu && !isSwiGluStep && !isSwiGluOai && !isSituGlu,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr,
                                              "one of 'swiglu', 'swiglustep', 'swigluoai' or 'situglu'"),
                    return ge::GRAPH_FAILED);

    // 拓扑约束先判：URMA 只认 swiglu，激活类型不对就没必要再往下校验它的参数了。
    ge::graphStatus topoStatus = CheckActivationTopoConstraint(tilingData->topoType, activationPtr, nodeName);
    if (topoStatus != ge::GRAPH_SUCCESS) {
        return topoStatus;
    }

    // 上面已经把激活类型收敛成四选一，所以这里最后一个分支就是 situglu。
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    if (isSwiGlu || isSwiGluStep) {
        status = CheckSwiGluActivationParams(activationParams, paramCount, nodeName);
    } else if (isSwiGluOai) {
        status = CheckSwiGluOaiActivationParams(activationParams, paramCount, nodeName);
    } else {
        status = CheckSituGluActivationParams(activationParams, paramCount, nodeName);
    }
    if (status == ge::GRAPH_SUCCESS) {
        SetActivationTilingData(activationPtr, activationParams, paramCount, tilingData);
    }
    return status;
}

/*
 * 下面四组校验按属性的业务类别划分：通信域、专家、量化、容量。
 * 组间顺序服从数据依赖：通信域先给出 epWorldSize，专家组用它推出本卡专家数，
 * 量化组先把 combineQuantMode 校验掉（容量公式要用它决定窗口里要不要留量化输出的位置），
 * 最后容量组才用前面这些值去算 peermem 窗口该开多大。
 */

/*
 * 通信域这一组：EP 域里有多少张卡、用哪种通信算法。
 * epWorldSize 是后面两组都要用的枢纽值（专家组拿它做整除、容量组拿它算窗口），所以经出参带出去。
 */
static ge::graphStatus CheckCommDomainAttrs(const gert::TilingContext *context, MegaMoeConfig &config,
                                            const char *nodeName, int64_t &epWorldSize)
{
    auto attrs = context->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    epWorldSize = static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > MAX_EP_WORLD_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EP_WORLD_SIZE) + ", " +
                                               std::to_string(MAX_EP_WORLD_SIZE) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    // 目前只实现了默认那一种通信算法，传了别的名字就是配错了，在 host 直接拦下。
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    OP_TILING_CHECK(std::strcmp(commAlgPtr, "") != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "commAlg", commAlgPtr, "not support, need empty string"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 专家这一组：路由专家总数、由它推出的本卡专家数，以及共享专家个数。
 * 专家要能均分到每张卡，所以本卡专家数 = moeExpertNum / epWorldSize，除法必须排在整除校验之后，
 * 否则不整除的配置会先算出一个取整过的错值。算完再跟 weight1 实际给了几个专家对一下，
 * 对不上说明权重和 attr 配的不是同一套。
 * 共享专家不参与上面这条路由切分，它的个数是从 sharedWeight1 张量形状数出来的，这里只卡上限。
 */
static ge::graphStatus CheckExpertNumAttrs(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName, const MegaMoeExpertParams &expertParams,
                                           int64_t epWorldSize, int64_t &moeExpertPerRank)
{
    auto attrs = context->GetAttrs();

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    OP_TILING_CHECK((moeExpertNum < epWorldSize || moeExpertNum > MAX_MOE_EXPERT_NUM) || (moeExpertNum % epWorldSize),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertNum", std::to_string(moeExpertNum).c_str(),
                                              (std::string("should in [") + std::to_string(epWorldSize) + ", " +
                                               std::to_string(MAX_MOE_EXPERT_NUM) + "] and mod(..., epWorldSize(" +
                                               std::to_string(epWorldSize) + ")) == 0")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    moeExpertPerRank = moeExpertNum / epWorldSize;
    OP_TILING_CHECK(moeExpertPerRank < MIN_EXPERT_PER_RANK || moeExpertPerRank > MAX_EXPERT_PER_RANK,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertPerRank", std::to_string(moeExpertPerRank).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EXPERT_PER_RANK) + ", " +
                                               std::to_string(MAX_EXPERT_PER_RANK) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(expertParams.moe.expertCount != moeExpertPerRank,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "weight1 expert count", std::to_string(expertParams.moe.expertCount).c_str(),
                        (std::string("should equal the local MoE expert count (moeExpertNum / epWorldSize) = ") +
                         std::to_string(moeExpertPerRank))
                            .c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expertParams.shared.expertCount > MAX_SHARED_EXPERT_NUM,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum", std::to_string(expertParams.shared.expertCount).c_str(),
                                  "only support 0-4"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 量化这一组：分别解析 MoE/shared 的 activation dtype、weight dtype 和 weight format，
 * 再校验 combine 回来的时候要不要量化。
 * 排在容量组之前，是因为 combine 量化模式会决定 peermem 窗口里要不要给量化输出留位置。
 */
static uint8_t ResolveExpertGmmMode(ge::DataType activationDtype, ge::DataType weightDtype, ge::Format weightOneFormat,
                                    ge::Format weightTwoFormat)
{
    bool isNdFormat = weightOneFormat == ge::FORMAT_ND && weightTwoFormat == ge::FORMAT_ND;
    bool isNzFormat = weightOneFormat == ge::FORMAT_FRACTAL_NZ && weightTwoFormat == ge::FORMAT_FRACTAL_NZ;
    bool isNzC032Format =
        weightOneFormat == ge::FORMAT_FRACTAL_NZ_C0_32 && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;
    bool isNzAndNzC032Format =
        weightOneFormat == ge::FORMAT_FRACTAL_NZ && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;

    if (activationDtype == ge::DT_FLOAT8_E5M2 && weightDtype == ge::DT_FLOAT8_E5M2 && isNdFormat) {
        return GMM_MODE_A8W8_ND;
    }
    if (activationDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT8_E4M3FN && isNdFormat) {
        return GMM_MODE_A8W8_ND;
    }
    if (activationDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT8_E4M3FN && isNzFormat) {
        return GMM_MODE_A8W8_NZ;
    }
    if (activationDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT4_E2M1 && isNzC032Format) {
        return GMM_MODE_A8W4_NZ;
    }
    if (activationDtype == ge::DT_FLOAT4_E2M1 && weightDtype == ge::DT_FLOAT4_E2M1 &&
        weightOneFormat == ge::FORMAT_ND && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32) {
        return GMM_MODE_A4W4_ND;
    }
    if (activationDtype == ge::DT_FLOAT4_E2M1 && weightDtype == ge::DT_FLOAT4_E2M1 && isNzAndNzC032Format) {
        return GMM_MODE_A4W4_NZ;
    }
    return INVALID_GMM_MODE;
}

static ge::graphStatus CheckAndSetExpertQuantAttrs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                   MegaMoeExpertParams &params, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    OP_TILING_CHECK(
        *dispatchQuantModePtr != DISPATCH_QUANT_MODE_MXFP,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "dispatch_quant_mode", std::to_string(*dispatchQuantModePtr).c_str(),
                                  std::to_string(DISPATCH_QUANT_MODE_MXFP).c_str()),
        return ge::GRAPH_FAILED);
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    OP_TILING_CHECK(EXPERT_QUANT_DTYPE_TO_MODE.count(static_cast<ge::DataType>(*dispatchQuantOutDtypePtr)) == 0U,
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "dispatch_quant_out_dtype",
                                              std::to_string(*dispatchQuantOutDtypePtr).c_str(),
                                              "fp8_e5m2, fp8_e4m3fn or fp4_e2m1"),
                    return ge::GRAPH_FAILED);
    auto sharedDtypePtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertQuantOutDtypeIndex);
    bool invalidSharedDtype = *sharedDtypePtr != static_cast<int64_t>(ge::DT_UNDEFINED) &&
                              EXPERT_QUANT_DTYPE_TO_MODE.count(static_cast<ge::DataType>(*sharedDtypePtr)) == 0U;
    OP_TILING_CHECK(
        invalidSharedDtype,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "shared_expert_quant_out_dtype", std::to_string(*sharedDtypePtr).c_str(),
                                  "DT_UNDEFINED, E5M2, E4M3FN or E2M1"),
        return ge::GRAPH_FAILED);
    params.moe.activationDtype = static_cast<ge::DataType>(*dispatchQuantOutDtypePtr);
    params.shared.activationDtype =
        params.shared.expertCount <= 0 || *sharedDtypePtr == static_cast<int64_t>(ge::DT_UNDEFINED) ?
            params.moe.activationDtype :
            static_cast<ge::DataType>(*sharedDtypePtr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetExpertGmmMode(const gert::TilingContext *context, ExpertParams &expertParams,
                                                const char *nodeName)
{
    auto weightOneDesc = context->GetDynamicInputDesc(expertParams.inputs.weightOneIndex, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(expertParams.inputs.weightTwoIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);
    ge::DataType weightOneDtype = weightOneDesc->GetDataType();
    ge::DataType weightTwoDtype = weightTwoDesc->GetDataType();
    std::string weightNames = std::string(expertParams.inputs.expertTypeName) + " weight1, weight2";
    std::string weightDtypes =
        "[" + Ops::Base::ToString(weightOneDtype) + ", " + Ops::Base::ToString(weightTwoDtype) + "]";
    OP_TILING_CHECK(weightOneDtype != weightTwoDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        nodeName, weightNames.c_str(), weightDtypes.c_str(),
                        "weight1 and weight2 within the same expert group must have the same dtype."),
                    return ge::GRAPH_FAILED);
    ge::Format weightOneFormat = weightOneDesc->GetStorageFormat();
    ge::Format weightTwoFormat = weightTwoDesc->GetStorageFormat();
    expertParams.gmmMode =
        ResolveExpertGmmMode(expertParams.activationDtype, weightOneDtype, weightOneFormat, weightTwoFormat);
    std::string actualConfig = "[" + Ops::Base::ToString(expertParams.activationDtype) + ", " +
                               Ops::Base::ToString(weightOneDtype) + ", " + Ops::Base::ToString(weightOneFormat) +
                               ", " + Ops::Base::ToString(weightTwoFormat) + "]";
    OP_TILING_CHECK(
        expertParams.gmmMode == INVALID_GMM_MODE,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            nodeName, "activation dtype, weight dtype, weight1 format, weight2 format", actualConfig.c_str(),
            (std::string(expertParams.inputs.expertTypeName) + " configuration does not form a supported GMM mode.")
                .c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetExpertWeightParams(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                     MegaMoeExpertParams &params, const char *nodeName)
{
    params.moe.inputs = {config.weight1Index, config.weight2Index, config.weightScales1Index, config.weightScales2Index,
                         "MoE expert"};
    params.shared.inputs = {config.sharedWeight1Index, config.sharedWeight2Index, config.sharedWeightScales1Index,
                            config.sharedWeightScales2Index, "shared expert"};
    auto weightOneShape = context->GetDynamicInputShape(params.moe.inputs.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    uint32_t weightDimNum = weightOneShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        weightDimNum != TWO_DIMS && weightDimNum != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "weight1", (std::to_string(weightDimNum) + "D").c_str(),
                                                 "weight1 must be one stacked 3D tensor or a list of 2D tensors."),
        return ge::GRAPH_FAILED);
    params.isPerExpertWeightTensor = weightDimNum == TWO_DIMS;
    params.moe.expertCount =
        GetWeightExpertCount(context, params.moe.inputs.weightOneIndex, params.isPerExpertWeightTensor);
    if (CheckAndSetSharedExpertCount(context, params, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetCommAndExpertAttrs(const gert::TilingContext *context, MegaMoeConfig &config,
                                                     const MegaMoeExpertParams &params, MegaMoeTilingData *tilingData,
                                                     const char *nodeName)
{
    int64_t epWorldSize = 0;
    if (CheckCommDomainAttrs(context, config, nodeName, epWorldSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t moeExpertPerRank = 0;
    if (CheckExpertNumAttrs(context, config, nodeName, params, epWorldSize, moeExpertPerRank) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingData->epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData->moeExpertPerRank = static_cast<uint32_t>(moeExpertPerRank);
    tilingData->sharedExpertNum = static_cast<uint32_t>(params.shared.expertCount);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetExpertQuantParams(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                    MegaMoeExpertParams &params, int64_t topoType, const char *nodeName)
{
    if (CheckAndSetExpertQuantAttrs(context, config, params, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAndSetExpertGmmMode(context, params.moe, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    params.shared.gmmMode = params.moe.gmmMode;
    if (params.shared.expertCount > 0 &&
        CheckAndSetExpertGmmMode(context, params.shared, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    bool hasDifferentExpertQuantConfigs =
        params.shared.gmmMode != params.moe.gmmMode || params.shared.activationDtype != params.moe.activationDtype;
    OP_TILING_CHECK(
        topoType != TOPO_TYPE_MTE && hasDifferentExpertQuantConfigs,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "topo_type", std::to_string(topoType).c_str(),
                                  "MTE when MoE and shared experts use different quantization configurations"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetCombineQuantAttr(const gert::TilingContext *context, MegaMoeConfig &config,
                                                   MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    bool invalid = *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_NO_QUANT &&
                   *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E5M2 &&
                   *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E4M3FN;
    OP_TILING_CHECK(
        invalid,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "combineQuantMode", std::to_string(*combineQuantModePtr).c_str(),
                                  "only support no_quant(0), fp8_e5m2(3) and fp8_e4m3fn(4)"),
        return ge::GRAPH_FAILED);
    tilingData->combineQuantMode = static_cast<uint32_t>(*combineQuantModePtr);
    return ge::GRAPH_SUCCESS;
}

/*
 * 算 peermem 窗口至少要开多大。公式跟 kernel 侧的布局同源（common/mega_moe_peermem.h），
 * host 和 device 不各写一份，免得两边算出来的窗口对不上。
 * 校验这会儿还不知道激活是什么 dtype，elemsPerByte 取 1 是保守上界，与既有校验口径一致。
 * 专家数、通信域、Combine mode 和拓扑均使用前序已校验并写入 tilingData 的值。
 */
static int64_t CalcLeastCclBufferSize(int64_t numMaxTokensPerRank, int64_t yDtypeSize,
                                      const MegaMoeTilingData *tilingData, bool topkWeightsPrefetch)
{
    PeermemSizeParams peermemSizeParams{static_cast<int64_t>(numMaxTokensPerRank),
                                        static_cast<int64_t>(tilingData->topK),
                                        static_cast<int64_t>(tilingData->h),
                                        static_cast<int64_t>(tilingData->moeExpertPerRank),
                                        static_cast<int64_t>(tilingData->epWorldSize),
                                        static_cast<int64_t>(yDtypeSize),
                                        1U,
                                        topkWeightsPrefetch,
                                        tilingData->combineQuantMode != COMBINE_NO_QUANT};
    int64_t leastCclBufferSize = CalcPeermemLeastSize(peermemSizeParams);
    // MTE 路径会在 peermem 头部单独留一段异常 dump 区，窗口下限要把这段一起算进去。
    if (tilingData->topoType == TOPO_TYPE_MTE) {
        leastCclBufferSize += EXCEPTION_DUMP_REGION_SIZE;
    }
    return leastCclBufferSize;
}

/*
 * 容量这一组：单卡 token 数上界、topk 权重预取开关、peermem 窗口大小、能收多少 token。
 * topkWeightsType 归在这里，是因为开了预取之后 peermem 里每个 token 要多存一份 topk 权重，
 * 它直接决定窗口要开多大，所以先把它校验掉，再拿去算下面的容量。
 * numMaxTokensPerRank 传 0 表示"就按本卡 bs 算"，替换成 bs 之后的值才是后面公式该用的，
 * 所以默认化就在这个函数里做完，别的地方不要再去读一次 attr——那样会拿回 0 把容量算塌。
 */
static ge::graphStatus CheckAndSetCapacityAttrs(const gert::TilingContext *context, MegaMoeConfig &config,
                                                const char *nodeName, int64_t yDtypeSize, MegaMoeTilingData *tilingData)
{
    auto attrs = context->GetAttrs();
    int64_t epWorldSize = tilingData->epWorldSize;
    int64_t moeExpertPerRank = tilingData->moeExpertPerRank;

    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    int64_t numMaxTokensPerRank = static_cast<int64_t>(*numMaxTokensPerRankPtr);
    // 这个上界最后按 uint32 落库，越界值不拦会被截断成一个错误的上界，所以连 UINT32_MAX 一起卡。
    OP_TILING_CHECK(numMaxTokensPerRank < 0 || numMaxTokensPerRank > 0xFFFFFFFFLL ||
                        (numMaxTokensPerRank != 0 && tilingData->bs > numMaxTokensPerRank),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "numMaxTokensPerRank", std::to_string(numMaxTokensPerRank).c_str(),
                        (std::string("0 or in [bs, UINT32_MAX], bs is ") + std::to_string(tilingData->bs).c_str())),
                    return ge::GRAPH_FAILED);
    if (numMaxTokensPerRank == 0) {
        numMaxTokensPerRank = tilingData->bs;
    }

    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    OP_TILING_CHECK(topkWeightsType != 0 && topkWeightsType != 1,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topkWeightsType", std::to_string(topkWeightsType).c_str(),
                                              "only support 0(disabled) or 1(enabled)"),
                    return ge::GRAPH_FAILED);

    int64_t leastCclBufferSize =
        CalcLeastCclBufferSize(numMaxTokensPerRank, yDtypeSize, tilingData, topkWeightsType == 1);
    int64_t cclBufferSize = static_cast<int64_t>(*attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex)));
    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "cclBufferSize", std::to_string(cclBufferSize).c_str(),
                                              (std::string("should >= ") + std::to_string(leastCclBufferSize)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    int64_t maxRecvTokenNum = static_cast<int64_t>(*attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex)));
    int64_t maxRecvTokenCapacity =
        numMaxTokensPerRank * epWorldSize * std::min(static_cast<int64_t>(tilingData->topK), moeExpertPerRank);
    OP_TILING_CHECK(
        maxRecvTokenNum < 0 || maxRecvTokenNum > maxRecvTokenCapacity,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "maxRecvTokenNum", std::to_string(maxRecvTokenNum).c_str(),
                                  (std::string("should in [0, ") + std::to_string(maxRecvTokenCapacity) + "]").c_str()),
        return ge::GRAPH_FAILED);

    tilingData->numMaxTokensPerRank = static_cast<uint32_t>(numMaxTokensPerRank);
    tilingData->topkWeightsPrefetch = static_cast<int32_t>(topkWeightsType);
    uint32_t defaultMaxOutputSize = tilingData->numMaxTokensPerRank * tilingData->epWorldSize *
                                    std::min(tilingData->topK, tilingData->moeExpertPerRank);
    tilingData->maxOutputSize = maxRecvTokenNum != 0 ? static_cast<uint32_t>(maxRecvTokenNum) : defaultMaxOutputSize;
    return ge::GRAPH_SUCCESS;
}

/*
 * 设置 prefetch 状态位区上限与 wave 粒度。maxTilesPerExpert 先无条件置零、再由 prefetch 分支覆盖，
 * 非 prefetch 路径依赖这次置零，不可改成只在分支内赋值。
 * 本步读 maxOutputSize，必须排在 CheckAndSetCapacityAttrs 之后。
 */
static void SetPrefetchAndWaveParams(MegaMoeTilingData *tilingData, const uint32_t aicNum)
{
    // GMM1 tile 状态位区每 expert 的 tile 上限（仅 prefetch 软同步路径使用）。
    // 非交织调度只遍历 hiddenDim / 2，交织调度会遍历完整 hiddenDim；host 无法感知 kernel
    // 编译期开关，因此按完整 hiddenDim 预留，避免交织模式覆盖下一 expert 的状态槽。
    tilingData->maxTilesPerExpert = 0;
    if (tilingData->topkWeightsPrefetch == 1) {
        int64_t maxSchedulerN = static_cast<int64_t>(tilingData->hiddenDim);
        int64_t maxTilesM = ops::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(256));
        int64_t maxTilesN = ops::CeilDiv(maxSchedulerN, static_cast<int64_t>(256));
        tilingData->maxTilesPerExpert =
            static_cast<uint32_t>(ops::CeilAlign(maxTilesM * maxTilesN, static_cast<int64_t>(16)));
    }

    tilingData->mGroupsPerWave = CalcMGroupsPerWave(tilingData, aicNum);
}

static void SetAttrParams(const MegaMoeExpertParams &expertParams, MegaMoeTilingData *tilingData, const uint32_t aicNum)
{
    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->epWorldSize);

    tilingData->moeGmmMode = expertParams.moe.gmmMode;
    tilingData->sharedGmmMode = expertParams.shared.gmmMode;

    // 读 maxOutputSize，必须排在 CheckAndSetCapacityAttrs 之后。
    SetPrefetchAndWaveParams(tilingData, aicNum);
}

/*
 * 计算每个 dispatch ring slot 的量化 token 与 scale 拷贝区字节数（prefetch 模式再追加对齐后的 weight）。
 */
static uint32_t CalcDispatchCopyBufferBytes(const MegaMoeTilingData *tilingData, uint32_t activationElementsPerByte)
{
    // copyBufferBytes 是每个 dispatch slot 中量化 token 和 scale 的拷贝区大小。
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    uint32_t copyBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        uint32_t weightBytes =
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        copyBufferBytes += weightBytes;
    }
    return copyBufferBytes;
}

/*
 * 计算 dispatch 阶段不随 ring 深度与 route batch 变化的固定 UB 占用。
 */
static uint32_t CalcDispatchFixedBufferBytes(const MegaMoeTilingData *tilingData)
{
    // fixedBufferBytes 包含 expertTokenCntTensor_、cumsumInfoTensor_、sendCntTensor_ 和 expertTokenNumsOutTensor_。
    uint32_t fixedBufferBytes =
        static_cast<uint32_t>(ALIGN_32) +
        static_cast<uint32_t>(ops::CeilAlign(
            static_cast<uint64_t>(tilingData->epWorldSize) * tilingData->moeExpertPerRank * sizeof(int32_t),
            static_cast<uint64_t>(ALIGN_32))) +
        tilingData->epWorldSize * static_cast<uint32_t>(ALIGN_32) +
        static_cast<uint32_t>(ops::CeilAlign(static_cast<uint64_t>(tilingData->moeExpertPerRank) * sizeof(int32_t),
                                             static_cast<uint64_t>(ALIGN_32)));
    return fixedBufferBytes;
}

/*
 * 分两步定下 dispatch 阶段的 UB 分配：
 *   第一步：先按一个保守的 route batch 大小起步，把剩下的 UB 全拿去开 ring slot，slot 越多流水越深；
 *   第二步：ring 深度定死之后，UB 若还有富余，再反过来把 route batch 撑大，这样总批数更少。
 * 两处写法不能顺手改：预算减法用三目做饱和（UB 不够时得 0，直接相减会回绕成近 4GB）；
 * bufferCount 的钳位必须先 min(上限) 再 max(下限)，颠倒后在下限大于上限的配置下结果会反过来。
 */
static void SelectDispatchRingAndRouteBatch(MegaMoeDispatchBufferConfig &bufferConfig, uint64_t sendTotalNum,
                                            uint64_t alignedTotalRouteItems, uint32_t fixedBufferBytes,
                                            uint32_t dispatchSlotBytes, uint32_t availableUbBytes)
{
    // 第一步：用基准 batch 把 ring 深度定下来。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_RECV_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) / bufferConfig.routeItemsPerBatch);

    uint32_t maskBufferBytes = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8);
    // routeIndexBufferBytes 是单个 int32 route index tensor 的大小。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    // bytesWithoutDispatchSlots 包含固定 tensor、maskBatchTensor_、topkIndexTensor_ 和
    // validTopkIndexTensor_（此二者等大）。
    uint32_t bytesWithoutDispatchSlots = fixedBufferBytes + maskBufferBytes + 2U * routeIndexBufferBytes;
    // dispatchSlotBudgetBytes 是扣除非 ring tensor 后可用于分配 ring slot 的 UB。
    uint32_t dispatchSlotBudgetBytes =
        availableUbBytes > bytesWithoutDispatchSlots ? availableUbBytes - bytesWithoutDispatchSlots : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(dispatchSlotBudgetBytes / dispatchSlotBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_DISPATCH_BUFFER_COUNT);
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_DISPATCH_BUFFER_COUNT);

    // 第二步：ring 深度已定，剩余 UB 用来扩 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithDispatchSlots 包含固定 tensor 和已选中的 dispatch slot。
        uint32_t fixedBytesWithDispatchSlots =
            fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * dispatchSlotBytes;
        // routeItemBudgetBytes 是 maskBatchTensor_ 和两个 route index tensor 可使用的 UB。
        uint32_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithDispatchSlots ? availableUbBytes - fixedBytesWithDispatchSlots : 0U;
        // maskBatchTensor_ 占 1bit，topkIndexTensor_ 和 validTopkIndexTensor_ 各占一个 int32：
        // bytesPerItem = 1/8 + 4 + 4 = 65/8B，因此 maxItems = budgetBytes * 8 / 65。
        uint32_t expandedRouteItems = routeItemBudgetBytes * 8U / 65U;
        expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
        expandedRouteItems =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
        if (expandedRouteItems > static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) /
                                                                bufferConfig.routeItemsPerBatch);
        }
    }
}

static MegaMoeDispatchBufferConfig CalcDispatchBufferConfig(const MegaMoeTilingData *tilingData,
                                                            uint32_t activationElementsPerByte,
                                                            uint32_t availableUbBytes)
{
    MegaMoeDispatchBufferConfig bufferConfig{};
    // 发送批网格按上界 numMaxTokensPerRank*topK 划分, 保证每张卡的批数和 mask 推送字节数一致,
    // 对端 mask 槽才能被完整覆盖; 本卡真实路由数 bs*topK 可能小于网格, 装载长度在 kernel 内逐批夹紧
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->topK;
    uint64_t alignedTotalRouteItems =
        (sendTotalNum + static_cast<uint64_t>(ALIGN_256) - 1U) / static_cast<uint64_t>(ALIGN_256) * ALIGN_256;
    uint32_t copyBufferBytes = CalcDispatchCopyBufferBytes(tilingData, activationElementsPerByte);
    bufferConfig.copyBufferBytes = copyBufferBytes;

    uint32_t fixedBufferBytes = CalcDispatchFixedBufferBytes(tilingData);
    // dispatchSlotBytes 表示一个 ring slot：一份 token/scale copy buffer 加一条 32B triple。
    // Stage 1 与 Stage 2 用的是同一个值，故在此统一算出。
    uint32_t dispatchSlotBytes = copyBufferBytes + static_cast<uint32_t>(ALIGN_32);

    SelectDispatchRingAndRouteBatch(bufferConfig, sendTotalNum, alignedTotalRouteItems, fixedBufferBytes,
                                    dispatchSlotBytes, availableUbBytes);
    return bufferConfig;
}

static MegaMoeSendMaskBufferConfig CalcSendMaskBufferConfig(const MegaMoeTilingData *tilingData,
                                                            uint32_t fixedBufferBytes, uint32_t ownedExpertCount,
                                                            uint32_t availableUbBytes)
{
    MegaMoeSendMaskBufferConfig bufferConfig{};
    // 发送批网格按上界 numMaxTokensPerRank*topK 划分, 保证每张卡的批数和 mask 推送字节数一致,
    // 对端 mask 槽才能被完整覆盖; 本卡真实路由数 bs*topK 可能小于网格, 装载长度在 kernel 内逐批夹紧
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->topK;
    uint64_t alignedTotalRouteItems =
        (sendTotalNum + static_cast<uint64_t>(ALIGN_256) - 1U) / static_cast<uint64_t>(ALIGN_256) * ALIGN_256;

    // Stage 1：使用基准 batch 确定 mask push ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_SEND_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) / bufferConfig.routeItemsPerBatch);
    bufferConfig.bufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8) + static_cast<uint32_t>(ALIGN_32);

    // routeIndexBufferBytes 是单个 int32 route tensor 的大小。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    // bytesWithoutMaskBuffers 包含固定 tensor、topkIdsTensor_ 和 sendGatherOutTensor_。
    uint32_t bytesWithoutMaskBuffers = fixedBufferBytes + 2U * routeIndexBufferBytes;
    // maskBufferBudgetBytes 是扣除非 ring tensor 后可用于分配 mask slot 的 UB。
    uint32_t maskBufferBudgetBytes =
        availableUbBytes > bytesWithoutMaskBuffers ? availableUbBytes - bytesWithoutMaskBuffers : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(maskBufferBudgetBytes / bufferConfig.bufferBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_SEND_MASK_BUFFER_COUNT);

    // maskPushCount 是当前 core 实际执行的 mask push 次数，更多 slot 不会增加流水重叠。
    uint64_t maskPushCount = static_cast<uint64_t>(bufferConfig.routeBatchCount) * ownedExpertCount;
    if (maskPushCount > 0U && static_cast<uint64_t>(bufferConfig.bufferCount) > maskPushCount) {
        bufferConfig.bufferCount = static_cast<int32_t>(maskPushCount);
    }
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);

    // Stage 2：固定 ring 深度，用剩余 UB 扩大 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithMaskCount 包含固定 tensor 和所有 slot 的 32B count 尾部。
        uint32_t fixedBytesWithMaskCount =
            fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * static_cast<uint32_t>(ALIGN_32);
        // routeItemBudgetBytes 是两个 route tensor 和所有 slot 的 mask 可使用的 UB。
        uint32_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithMaskCount ? availableUbBytes - fixedBytesWithMaskCount : 0U;
        // topkIdsTensor_ 和 sendGatherOutTensor_ 各占一个 int32，N 个 sendMask slot 各占 1bit：
        // bytesPerItem = 4 + 4 + N/8 = (64 + N)/8B。
        uint32_t expandedRouteItems =
            routeItemBudgetBytes * 8U / (64U + static_cast<uint32_t>(bufferConfig.bufferCount));
        expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
        expandedRouteItems =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
        if (expandedRouteItems > static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) /
                                                                bufferConfig.routeItemsPerBatch);
            bufferConfig.bufferBytes =
                static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8) + static_cast<uint32_t>(ALIGN_32);
        }
    }
    return bufferConfig;
}

/*
 * 计算 unpermute 的 slot 与 scale 尺寸：单 token 的 BF16 搬入区 + FP32 计算区，以及 combine quant 的 scale 展开区。
 * dataSlotBytes 与 scaleBytes 经出参带出，供后续两个阶段共用。
 */
static void CalcUnpermuteSlotAndScaleBytes(const MegaMoeTilingData *tilingData,
                                           MegaMoeUnpermuteBufferConfig &bufferConfig, uint32_t &dataSlotBytes,
                                           uint32_t &scaleBytes)
{
    uint32_t bf16SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(uint16_t), static_cast<uint64_t>(ALIGN_32)));
    uint32_t fp32SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    // dataSlotBytes 是同一 token 的 BF16 搬入区和 FP32 计算区之和。
    dataSlotBytes = bf16SlotBytes + fp32SlotBytes;
    bufferConfig.bf16SlotElementCount = bf16SlotBytes / sizeof(uint16_t);
    bufferConfig.fp32SlotElementCount = fp32SlotBytes / sizeof(float);

    // scaleBytes 是 combine quant 使用的 BF16/FP32 scale 展开区大小。
    scaleBytes = 0U;
    if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleElementCount = (tilingData->h + ALIGN_32 - 1U) / ALIGN_32;
        scaleBytes = static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(uint16_t) * DEQUANT_BF16_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32))) +
                     static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32)));
    }
}

/*
 * 按 weight 元素个数刷新 unpermute 的 weight 区大小：FP32 主区必算，转换中转区只有需要 dtype
 * 转换时才占 UB。两个阶段定完 tokensPerBatch 后都要重算一次，所以抽出来复用。
 */
static void SetUnpermuteWeightBufferBytes(MegaMoeUnpermuteBufferConfig &bufferConfig, uint32_t weightElementCount,
                                          uint32_t topKWeightsConversionElementBytes)
{
    bufferConfig.topKWeightsBufferBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    if (topKWeightsConversionElementBytes > 0U) {
        bufferConfig.topKWeightsConversionBufferBytes = static_cast<uint32_t>(
            ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * topKWeightsConversionElementBytes,
                           static_cast<uint64_t>(ALIGN_32)));
    }
}

static MegaMoeUnpermuteBufferConfig CalcUnpermuteBufferConfig(const MegaMoeTilingData *tilingData,
                                                              uint32_t coreTokenCount,
                                                              uint32_t topKWeightsConversionElementBytes,
                                                              uint32_t availableUbBytes)
{
    MegaMoeUnpermuteBufferConfig bufferConfig{};
    if (coreTokenCount == 0U) {
        return bufferConfig;
    }

    uint32_t dataSlotBytes = 0U;
    uint32_t scaleBytes = 0U;
    CalcUnpermuteSlotAndScaleBytes(tilingData, bufferConfig, dataSlotBytes, scaleBytes);

    // 第一步：先按基准 weight batch 定住每批 token 数和 weight 区，再拿剩余 UB 开输入 ring。
    uint32_t baseTokensPerBatch = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / tilingData->topK;
    bufferConfig.tokensPerBatch = static_cast<int32_t>(std::min(baseTokensPerBatch, coreTokenCount));
    uint32_t weightElementCount = static_cast<uint32_t>(bufferConfig.tokensPerBatch) * tilingData->topK;
    SetUnpermuteWeightBufferBytes(bufferConfig, weightElementCount, topKWeightsConversionElementBytes);

    // bytesBeforeInputBuffers 包含 weight、scale 和一个累加/输出 data slot。
    uint32_t bytesBeforeInputBuffers = bufferConfig.topKWeightsBufferBytes +
                                       bufferConfig.topKWeightsConversionBufferBytes + scaleBytes + dataSlotBytes;
    uint32_t inputBufferBudgetBytes =
        availableUbBytes > bytesBeforeInputBuffers ? availableUbBytes - bytesBeforeInputBuffers : 0U;
    bufferConfig.inputBufferCount = static_cast<int32_t>(inputBufferBudgetBytes / dataSlotBytes);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, MAX_UNPERMUTE_INPUT_BUFFER_COUNT);
    int32_t accumulationItemCount =
        bufferConfig.tokensPerBatch * static_cast<int32_t>(tilingData->topK + tilingData->sharedExpertNum);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, accumulationItemCount);
    bufferConfig.inputBufferCount = std::max(bufferConfig.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);

    // 第二步：ring 深度已定，UB 还有富余就把 weight batch 撑大。
    if (baseTokensPerBatch < coreTokenCount) {
        // fixedBytes 包含 scale、一个累加/输出 slot 和已经选中的所有输入 slot。
        uint32_t fixedBytes = scaleBytes + (static_cast<uint32_t>(bufferConfig.inputBufferCount) + 1U) * dataSlotBytes;
        // weightBudgetBytes 是 FP32 weight 及可选转换中转区可使用的 UB。
        uint32_t weightBudgetBytes = availableUbBytes - fixedBytes - UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES;
        uint32_t weightBytesPerToken =
            tilingData->topK * (static_cast<uint32_t>(sizeof(float)) + topKWeightsConversionElementBytes);
        uint32_t expandedTokensPerBatch = std::min(weightBudgetBytes / weightBytesPerToken, coreTokenCount);
        if (expandedTokensPerBatch > static_cast<uint32_t>(bufferConfig.tokensPerBatch)) {
            bufferConfig.tokensPerBatch = static_cast<int32_t>(expandedTokensPerBatch);
            weightElementCount = expandedTokensPerBatch * tilingData->topK;
            SetUnpermuteWeightBufferBytes(bufferConfig, weightElementCount, topKWeightsConversionElementBytes);
        }
    }
    return bufferConfig;
}

static uint64_t CalcCombineSyncSlotCountPerExpert(const MegaMoeTilingData *tilingData)
{
    if (tilingData->combineQuantMode == COMBINE_NO_QUANT && tilingData->topoType != TOPO_TYPE_URMA ||
        tilingData->moeExpertPerRank == 0U) {
        return 0U;
    }

    uint64_t logicalCoreCount = tilingData->blockAivNum;
    bool useAiv1OnlyWaveCombine =
        tilingData->topoType == TOPO_TYPE_MTE &&
        (tilingData->moeGmmMode == GMM_MODE_A8W8_ND || tilingData->moeGmmMode == GMM_MODE_A8W8_NZ);
    if (tilingData->moeGmmMode == GMM_MODE_A8W4_NZ || tilingData->moeGmmMode == GMM_MODE_A4W4_ND ||
        tilingData->moeGmmMode == GMM_MODE_A4W4_NZ || useAiv1OnlyWaveCombine) {
        // 配对路径和 A8W8 wave 均只让 subBlockIdx=1 参与 Combine，因此逻辑核数是物理 AIV 数的一半。
        logicalCoreCount /= 2U;
    }
    // 同一 token 的 topK expert id 不重复，因此单 expert 从每张卡最多接收 bs 个 token。
    uint64_t maxTokenCountForOneExpert =
        static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->epWorldSize;
    uint64_t maxTokenGroupCountForOneExpert =
        ops::CeilDiv(maxTokenCountForOneExpert, static_cast<uint64_t>(COMBINE_TOKEN_GROUP_SIZE));
    // Workspace 在路由结果产生前分配，因此每个本卡 MoE expert 都按独立最坏情况预留 slot。
    return std::max(maxTokenGroupCountForOneExpert, logicalCoreCount);
}

static uint64_t CalcHostFlagElementCount(const MegaMoeTilingData *tilingData)
{
    bool useMteA8W8Wave = tilingData->topoType == TOPO_TYPE_MTE &&
                          (tilingData->moeGmmMode == GMM_MODE_A8W8_ND || tilingData->moeGmmMode == GMM_MODE_A8W8_NZ);
    bool useGroupGrainedActivationFlag = tilingData->topoType == TOPO_TYPE_MTE;
    bool useGroupSyncCounters =
        tilingData->topoType == TOPO_TYPE_URMA || (tilingData->combineQuantMode != COMBINE_NO_QUANT && !useMteA8W8Wave);
    uint64_t maxWavesPerExpert = ops::CeilDiv<uint64_t>(tilingData->maxOutputSize, L1_TILE_M_256);
    uint64_t waveFlagSlotsPerExpert = maxWavesPerExpert * INT_CACHELINE;
    uint64_t activationFlagSlotsPerExpert = useGroupGrainedActivationFlag ? waveFlagSlotsPerExpert : INT_CACHELINE;
    uint64_t moeExpertCount = tilingData->moeExpertPerRank;

    uint64_t flagElementCount = moeExpertCount * (activationFlagSlotsPerExpert + waveFlagSlotsPerExpert +
                                                  static_cast<uint64_t>(INT_CACHELINE) * tilingData->aicNum);
    if (tilingData->moeGmmMode == GMM_MODE_A8W4_NZ || tilingData->moeGmmMode == GMM_MODE_A4W4_ND ||
        tilingData->moeGmmMode == GMM_MODE_A4W4_NZ) {
        flagElementCount += static_cast<uint64_t>(tilingData->aicNum) * INT_CACHELINE;
    }
    if (useMteA8W8Wave) {
        flagElementCount += moeExpertCount * tilingData->aicNum * INT_CACHELINE;
    }
    if (useGroupSyncCounters) {
        flagElementCount += tilingData->combineSyncSlotCountPerExpert * moeExpertCount * INT_CACHELINE;
    }
    if (tilingData->sharedExpertNum > 0) {
        uint64_t tokenGroupCount = ops::CeilDiv<uint64_t>(tilingData->bs, L1_TILE_M_256);
        flagElementCount += tokenGroupCount * tilingData->sharedExpertNum * INT_CACHELINE;
    }
    return flagElementCount;
}

/*
 * 设置 SendMask 两套 UB 配置：先算与 ring 深度无关的固定占用，再按 expert 分核的两类 core 各算一套。
 * 本函数读 tilingData->combineSyncSlotCountPerExpert（经 CalcHostFlagElementCount），
 * 该字段必须在调用前写好。
 */
static void SetSendMaskBufferConfigs(MegaMoeTilingData *tilingData, uint32_t activationElementsPerByte,
                                     uint32_t availableUbBytes)
{
    uint64_t totalFlagElementCount = CalcHostFlagElementCount(tilingData);
    uint32_t resetElementCountPerCore =
        static_cast<uint32_t>(ops::CeilDiv(totalFlagElementCount, static_cast<uint64_t>(tilingData->blockAivNum)));
    uint32_t resetBatchElementCount = std::min(resetElementCountPerCore, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    uint32_t resetTensorBytes =
        ops::CeilAlign(resetBatchElementCount, static_cast<uint32_t>(INT32_PER_256B)) * sizeof(int32_t);
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    // 与 kernel xOutTensorSize 一致：token 和 scale 分别对齐，prefetch 模式再追加对齐后的 weight。
    uint32_t quantOutputBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        quantOutputBufferBytes +=
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    }
    uint32_t quantInputBufferBytes = ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_128)) * sizeof(uint16_t);
    // sendCntAccTensor_ 按本卡 MoE 专家数分配，与 kernel 地址布局一致。
    uint32_t maxExpertCountPerCore =
        ops::CeilDiv(tilingData->epWorldSize * tilingData->moeExpertPerRank, tilingData->blockAivNum);
    uint32_t sendCountAccumulatorBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(maxExpertCountPerCore) * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32)));
    // mxTempTensor_ 占 2KB，xOutTensor_ 和 xInTensor_ 各使用双 buffer。
    uint32_t sendMaskFixedBufferBytes = resetTensorBytes + 2U * 1024U + 2U * quantOutputBufferBytes +
                                        2U * quantInputBufferBytes + sendCountAccumulatorBytes;

    /*
     * 与 kernel SendMaskCal 的 expert 遍历一一对应：
     *   globalExpertId = aivCoreIdx + ownedIdx * blockAivNum。
     * totalExpertCount 除以 blockAivNum 后，前 remainder 个 AIV core 各多处理一个 expert，其余 core
     * 处理 quotient 个 expert，因此这里只需预计算两套配置。
     *
     * 若修改 SendMaskCal 的 expert 分核方式、ownedExpertCount 或 maskPushCount 计算，必须同步更新
     * 这里的两类 core 划分和 kernel 配置选择条件。
     */
    uint32_t totalExpertCount = tilingData->epWorldSize * tilingData->moeExpertPerRank;
    uint32_t expertCountPerCoreWithoutExtraExpert = totalExpertCount / tilingData->blockAivNum;
    tilingData->sendMaskCoreCountWithExtraExpert = totalExpertCount % tilingData->blockAivNum;
    uint32_t expertCountPerCoreWithExtraExpert = expertCountPerCoreWithoutExtraExpert + 1U;
    tilingData->sendMaskConfigForCoreWithExtraExpert = CalcSendMaskBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithExtraExpert, availableUbBytes);
    tilingData->sendMaskConfigForCoreWithoutExtraExpert = CalcSendMaskBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithoutExtraExpert, availableUbBytes);
}

/*
 * 设置 Unpermute 的完整 chunk 与 tail chunk 两套 UB 配置，并记录完整 chunk 对应的 core 数。
 */
static void SetUnpermuteBufferConfigs(MegaMoeTilingData *tilingData, ge::DataType topKWeightsDataType,
                                      uint32_t availableUbBytes)
{
    /*
     * 与 kernel Unpermute 开头的 TilingByCore(m_, ..., align=1) 一一对应。TilingByCore 使用：
     *   fullTokenChunkSize = ceil(bs / blockAivNum)
     * 为连续 core 分配等长完整 chunk，最后一个活跃 core 可能只处理 tail，后续 core 的 coreLen 为 0
     * 并在读取配置前返回。因此 host 只需预计算“完整 chunk”和“tail chunk”两套配置，并记录完整
     * chunk 对应的 core 数作为 kernel 选择边界。
     *
     * 若修改 TilingByCore、Unpermute 的 align 参数或分核方式，必须同步更新下面的 chunk 推导以及
     * kernel 中 UnpermuteBuffInit 的配置选择条件。
     */
    // FP32 可直接搬入计算 buffer；其他已支持类型按实际元素大小预留转换中转区。
    uint32_t topKWeightsConversionElementBytes =
        topKWeightsDataType == ge::DT_FLOAT ? 0U : static_cast<uint32_t>(ge::GetSizeByDataType(topKWeightsDataType));
    uint32_t fullTokenChunkSize = ops::CeilDiv(tilingData->bs, tilingData->blockAivNum);
    uint32_t activeCoreCount = ops::CeilDiv(tilingData->bs, fullTokenChunkSize);
    uint32_t tailTokenChunkSize = tilingData->bs - (activeCoreCount - 1U) * fullTokenChunkSize;
    bool tailIsFullTokenChunk = tailTokenChunkSize == fullTokenChunkSize;
    tilingData->unpermuteFullTokenChunkCoreCount = tailIsFullTokenChunk ? activeCoreCount : activeCoreCount - 1U;
    tilingData->unpermuteConfigForFullTokenChunk =
        CalcUnpermuteBufferConfig(tilingData, fullTokenChunkSize, topKWeightsConversionElementBytes, availableUbBytes);
    tilingData->unpermuteConfigForTailTokenChunk =
        tailIsFullTokenChunk ? MegaMoeUnpermuteBufferConfig{} :
                               CalcUnpermuteBufferConfig(tilingData, tailTokenChunkSize,
                                                         topKWeightsConversionElementBytes, availableUbBytes);
}

static void SetAdaptiveBufferConfigs(const gert::TilingContext *context, MegaMoeConfig &config,
                                     const MegaMoeExpertParams &expertParams, MegaMoeTilingData *tilingData,
                                     uint32_t availableUbBytes)
{
    auto topKWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    uint32_t activationElementsPerByte = expertParams.moe.activationDtype == ge::DT_FLOAT4_E2M1 ? 2U : 1U;
    // 所有 AIV 核共用同一套 dispatch UB 配置：这块布局和怎么分核无关。
    // kernel 那边如果改成按核拆 tensor，或者动了 copyTmp 的槽位布局，这里要跟着改。
    tilingData->dispatchBufferConfig =
        CalcDispatchBufferConfig(tilingData, activationElementsPerByte, availableUbBytes);

    // 先于 SendMask 组写入：CalcHostFlagElementCount 会读这个字段累加 flag 区大小，
    // 该依赖只经 tilingData 传递，调换顺序会静默改变 flag 区尺寸。
    tilingData->combineSyncSlotCountPerExpert = CalcCombineSyncSlotCountPerExpert(tilingData);

    SetSendMaskBufferConfigs(tilingData, activationElementsPerByte, availableUbBytes);

    ge::DataType topKWeightsDataType = topKWeightsDesc->GetDataType();
    SetUnpermuteBufferConfigs(tilingData, topKWeightsDataType, availableUbBytes);
}

static ge::graphStatus CheckAttrAndSetTilingData(const gert::TilingContext *context, MegaMoeConfig &config,
                                                 const MegaMoeExpertParams &expertParams, MegaMoeTilingData *tilingData,
                                                 const uint32_t aicNum)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckAndSetActivationParams(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check activation params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAndSetCombineQuantAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "combine quantization attribute is invalid."), return ge::GRAPH_FAILED);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    int64_t yDtypeSize = ge::GetSizeByDataType(yDesc->GetDataType());
    OP_TILING_CHECK(CheckAndSetCapacityAttrs(context, config, nodeName, yDtypeSize, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "capacity attributes are invalid."), return ge::GRAPH_FAILED);

    SetAttrParams(expertParams, tilingData, aicNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, WorkspaceInfo &workspaceInfo, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspace"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(workspaceInfo.workspaceSize == 0LL,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "workspaceSize",
                                              std::to_string(workspaceInfo.workspaceSize).c_str(), "non-zero"),
                    return ge::GRAPH_FAILED);

    int64_t workspaceSize = sysWorkspaceSize + workspaceInfo.workspaceSize + RESERVED_WORKSPACE_SIZE;
    workspace[0] = workspaceSize;

    OP_LOGD(nodeName, "sysWorkspaceSize: %ld \n", sysWorkspaceSize);
    OP_LOGD(nodeName, "mega_moe_tiling workspaceSize: %ld \n", workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context, MegaMoeConfig &config,
                                             const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsDesc);

    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(xActiveMaskDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "xActiveMask", "not null", "null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scalesDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "scales", "not null", "null"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorListEntryProperties(const gert::StorageShape *referenceShape,
                                                      const gert::CompileTimeTensorDesc *referenceDesc,
                                                      const gert::StorageShape *currentShape,
                                                      const gert::CompileTimeTensorDesc *currentDesc,
                                                      const std::string &tensorName, uint32_t tensorIndex,
                                                      uint32_t expectedDimNum, const char *nodeName)
{
    const std::string entryName = tensorName + "[" + std::to_string(tensorIndex) + "]";
    const uint32_t currentDimNum = currentShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        currentDimNum != expectedDimNum,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, entryName.c_str(), (std::to_string(currentDimNum) + "D").c_str(),
                                     (std::to_string(expectedDimNum) + "D").c_str()),
        return ge::GRAPH_FAILED);

    bool shapeMismatch = false;
    for (uint32_t dimIndex = 0; dimIndex < expectedDimNum && !shapeMismatch; ++dimIndex) {
        shapeMismatch =
            currentShape->GetStorageShape().GetDim(dimIndex) != referenceShape->GetStorageShape().GetDim(dimIndex);
    }
    OP_TILING_CHECK(shapeMismatch,
                    OP_LOGE_FOR_INVALID_SHAPE(nodeName, entryName.c_str(),
                                              Ops::Base::ToString(currentShape->GetStorageShape()).c_str(),
                                              Ops::Base::ToString(referenceShape->GetStorageShape()).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        currentDesc->GetDataType() != referenceDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, entryName.c_str(), Ops::Base::ToString(currentDesc->GetDataType()).c_str(),
                                  Ops::Base::ToString(referenceDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(currentDesc->GetStorageFormat() != referenceDesc->GetStorageFormat(),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, entryName.c_str(),
                                               Ops::Base::ToString(currentDesc->GetStorageFormat()).c_str(),
                                               Ops::Base::ToString(referenceDesc->GetStorageFormat()).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验同一 TensorList 内所有 tensor 均为预期维数，且 shape、dtype 和 format 一致。
 * 以第 0 项为基准，从第 1 项开始逐项比较。
 */
static ge::graphStatus CheckTensorPropertiesWithinList(const gert::TilingContext *context, uint32_t inputIndex,
                                                       const char *expertTypeName, const char *tensorRole,
                                                       uint32_t expectedDimNum, const char *nodeName)
{
    const std::string tensorName = std::string(expertTypeName) + " " + tensorRole;
    const uint32_t tensorCount = GetDynamicInputTensorCount(context, inputIndex);
    OP_TILING_CHECK(tensorCount == 0U, OP_LOGE_FOR_INVALID_TENSORNUM(nodeName, tensorName.c_str(), tensorCount, "> 0"),
                    return ge::GRAPH_FAILED);

    auto referenceShape = context->GetDynamicInputShape(inputIndex, 0);
    auto referenceDesc = context->GetDynamicInputDesc(inputIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceDesc);
    const uint32_t referenceDimNum = referenceShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(referenceDimNum != expectedDimNum,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, (tensorName + "[0]").c_str(),
                                                 (std::to_string(referenceDimNum) + "D").c_str(),
                                                 (std::to_string(expectedDimNum) + "D").c_str()),
                    return ge::GRAPH_FAILED);

    for (uint32_t tensorIdx = 1; tensorIdx < tensorCount; ++tensorIdx) {
        auto currentShape = context->GetDynamicInputShape(inputIndex, tensorIdx);
        auto currentDesc = context->GetDynamicInputDesc(inputIndex, tensorIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentDesc);

        if (CheckTensorListEntryProperties(referenceShape, referenceDesc, currentShape, currentDesc, tensorName,
                                           tensorIdx, expectedDimNum, nodeName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 MoE 专家或共享专家的 weight1、weight2、weightScales1 和 weightScales2 四个 TensorList。
 * 先校验每个 TensorList 内部的 tensor 属性一致，再校验四个 TensorList 的长度及堆叠布局下的专家数一致。
 */
static ge::graphStatus CheckExpertTensorListCounts(const gert::TilingContext *context,
                                                   const ExpertWeightTensorGroupInputs &inputs, uint32_t weightDimNum,
                                                   const char *nodeName)
{
    uint32_t weightOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightOneIndex);
    uint32_t weightTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightTwoIndex);
    uint32_t weightScalesOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesOneIndex);
    uint32_t weightScalesTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesTwoIndex);
    bool tensorCountsMismatch = weightTwoTensorCount != weightOneTensorCount ||
                                weightScalesOneTensorCount != weightOneTensorCount ||
                                weightScalesTwoTensorCount != weightOneTensorCount;
    const std::string tensorNames =
        std::string(inputs.expertTypeName) + " weight1, weight2, weight_scales1, weight_scales2";
    const std::string tensorCounts =
        "[" + std::to_string(weightOneTensorCount) + ", " + std::to_string(weightTwoTensorCount) + ", " +
        std::to_string(weightScalesOneTensorCount) + ", " + std::to_string(weightScalesTwoTensorCount) + "]";
    OP_TILING_CHECK(
        tensorCountsMismatch || (weightDimNum == THREE_DIMS && weightOneTensorCount != 1U),
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
            nodeName, tensorNames.c_str(), tensorCounts.c_str(),
            "The four TensorLists must have equal lengths, and stacked layout requires one tensor per list."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckStackedExpertCounts(const gert::TilingContext *context,
                                                const ExpertWeightTensorGroupInputs &inputs, const char *nodeName)
{
    auto weightOneShape = context->GetDynamicInputShape(inputs.weightOneIndex, 0);
    auto weightTwoShape = context->GetDynamicInputShape(inputs.weightTwoIndex, 0);
    auto scaleOneShape = context->GetDynamicInputShape(inputs.weightScalesOneIndex, 0);
    auto scaleTwoShape = context->GetDynamicInputShape(inputs.weightScalesTwoIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleTwoShape);
    int64_t expertCount = weightOneShape->GetStorageShape().GetDim(0);
    bool mismatch = weightTwoShape->GetStorageShape().GetDim(0) != expertCount ||
                    scaleOneShape->GetStorageShape().GetDim(0) != expertCount ||
                    scaleTwoShape->GetStorageShape().GetDim(0) != expertCount;
    const std::string tensorNames =
        std::string(inputs.expertTypeName) + " weight1, weight2, weight_scales1, weight_scales2";
    const std::string expertCounts = "[" + std::to_string(expertCount) + ", " +
                                     std::to_string(weightTwoShape->GetStorageShape().GetDim(0)) + ", " +
                                     std::to_string(scaleOneShape->GetStorageShape().GetDim(0)) + ", " +
                                     std::to_string(scaleTwoShape->GetStorageShape().GetDim(0)) + "]";
    OP_TILING_CHECK(mismatch,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        nodeName, tensorNames.c_str(), expertCounts.c_str(),
                        "The leading dimensions of stacked inputs must contain the same expert count."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertWeightInputGroupLayout(const gert::TilingContext *context,
                                                         const ExpertWeightTensorGroupInputs &inputs,
                                                         uint32_t weightDimNum, const char *nodeName)
{
    uint32_t scaleDimNum = weightDimNum + 1U;
    bool invalidProperties =
        CheckTensorPropertiesWithinList(context, inputs.weightOneIndex, inputs.expertTypeName, "weight1", weightDimNum,
                                        nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorPropertiesWithinList(context, inputs.weightTwoIndex, inputs.expertTypeName, "weight2", weightDimNum,
                                        nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorPropertiesWithinList(context, inputs.weightScalesOneIndex, inputs.expertTypeName, "weight_scales1",
                                        scaleDimNum, nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorPropertiesWithinList(context, inputs.weightScalesTwoIndex, inputs.expertTypeName, "weight_scales2",
                                        scaleDimNum, nodeName) != ge::GRAPH_SUCCESS;
    if (invalidProperties) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExpertTensorListCounts(context, inputs, weightDimNum, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return weightDimNum == THREE_DIMS ? CheckStackedExpertCounts(context, inputs, nodeName) : ge::GRAPH_SUCCESS;
}

/*
 * 以 MoE weight1[0] 的维数确定公共权重布局，2D 表示逐专家 TensorList，3D 表示单 tensor 堆叠。
 * 校验 MoE 的 weight1、weight2、weightScales1 和 weightScales2。
 * 检查共享专家的对应输入是否同时缺省或同时存在，并在存在时校验其布局与 MoE 一致。
 */
static ge::graphStatus CheckMoeAndSharedWeightInputLayouts(const gert::TilingContext *context,
                                                           const MegaMoeExpertParams &expertParams,
                                                           const char *nodeName)
{
    uint32_t weightDimNum = expertParams.isPerExpertWeightTensor ? TWO_DIMS : THREE_DIMS;
    if (CheckExpertWeightInputGroupLayout(context, expertParams.moe.inputs, weightDimNum, nodeName) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (expertParams.shared.expertCount > 0) {
        if (CheckExpertWeightInputGroupLayout(context, expertParams.shared.inputs, weightDimNum, nodeName) !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertWeightTensorDim(const gert::TilingContext *context,
                                                  const ExpertWeightTensorGroupInputs &inputs, uint32_t xIndex,
                                                  bool isPerExpertWeightTensor, const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(inputs.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    auto weightTwoStorageShape = context->GetDynamicInputShape(inputs.weightTwoIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);
    const int64_t weightOneRowCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightOneColumnCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightTwoRowCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightTwoColumnCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);

    const gert::StorageShape *xStorageShape = context->GetInputShape(xIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const int64_t xColumnCount = xStorageShape->GetStorageShape().GetDim(1);
    const std::string commonMatrixDimensionsString = "[" + std::to_string(weightOneColumnCount) + ", " +
                                                     std::to_string(weightTwoRowCount) + ", " +
                                                     std::to_string(xColumnCount) + "]";
    OP_TILING_CHECK(
        weightOneColumnCount != weightTwoRowCount || weightOneColumnCount != xColumnCount,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, inputs.expertTypeName, commonMatrixDimensionsString.c_str(),
                                               "weight1 column count and weight2 row count must equal x column count."),
        return ge::GRAPH_FAILED);

    const std::string weightRowColumnDimensionsString =
        "[" + std::to_string(weightOneRowCount) + ", " + std::to_string(weightTwoColumnCount) + "]";
    OP_TILING_CHECK(
        weightOneRowCount != weightTwoColumnCount * SWIGLU_GATE_UP_SPLIT_FACTOR,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, inputs.expertTypeName, weightRowColumnDimensionsString.c_str(),
                                               "weight1 row count must equal weight2 column count multiplied by 2."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightTensorDim(const gert::TilingContext *context, const MegaMoeExpertParams &expertParams,
                                            uint32_t xIndex, const char *nodeName)
{
    if (CheckExpertWeightTensorDim(context, expertParams.moe.inputs, xIndex, expertParams.isPerExpertWeightTensor,
                                   nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (expertParams.shared.expertCount <= 0) {
        return ge::GRAPH_SUCCESS;
    }
    return CheckExpertWeightTensorDim(context, expertParams.shared.inputs, xIndex, expertParams.isPerExpertWeightTensor,
                                      nodeName);
}

static ge::graphStatus CheckOutputTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                            const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    auto yStorageShape = context->GetOutputShape(config.yIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorageShape);
    OP_TILING_CHECK(
        yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "y", (std::to_string(yStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);
    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "y dim0 = %ld", yDim0);
    OP_LOGD(nodeName, "y dim1 = %ld", yDim1);

    OP_TILING_CHECK(
        yDim0 != bs || yDim1 != h,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "y", (std::string("[") + std::to_string(yDim0) + ", " + std::to_string(yDim1) + "]").c_str(),
            (std::string("The shape of y must be [bs, h] = [") + std::to_string(bs) + ", " + std::to_string(h) + "].")
                .c_str()),
        return ge::GRAPH_FAILED);

    auto expertTokenNumsStorageShape = context->GetOutputShape(config.expertTokenNumsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsStorageShape);
    OP_TILING_CHECK(
        expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "expert_token_nums",
            (std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsDim0);

    // expertTokenNums 仅报告 MoE 专家的 token 数，不包含共享专家。
    auto attrs = context->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    int64_t moeExpertPerRank = static_cast<int64_t>(*moeExpertNumPtr) / static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(
        expertTokenNumsDim0 != moeExpertPerRank,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "expertTokenNums", (std::string("dim0=") + std::to_string(expertTokenNumsDim0)).c_str(),
            (std::string("The shape [dim0] of expertTokenNums must be equal to moeExpertPerRank(") +
             std::to_string(moeExpertPerRank) + ").")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一组 weight 和它的 scale 形状对不对得上。weight 每 INPUT_WEIGHT_SCALES_CEIL_ALIGN 列共用一个
 * scale，所以 scale 的形状必然是 [weight 行数, ceil(weight 列数 / 该对齐值)]。
 * weight1 和 weight2 两组走的是同一套规则，所以共用这个函数。
 * scale 末尾的 multi-base 维在 CheckExpertWeightScalesTensorDim 中统一校验。
 */
static ge::graphStatus CheckWeightScaleTensorDim(const gert::StorageShape *weightScaleStorageShape,
                                                 const gert::StorageShape *weightStorageShape,
                                                 bool isPerExpertWeightTensor, const char *expertTypeName,
                                                 const char *weightScaleRole, const char *weightRole,
                                                 const char *nodeName)
{
    const int64_t weightScaleMatrixDimSize =
        GetSingleExpertTensorDimSize(weightScaleStorageShape, WEIGHT_SCALE_MATRIX_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightScaleGroupDimSize =
        GetSingleExpertTensorDimSize(weightScaleStorageShape, WEIGHT_SCALE_GROUP_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightRowCount =
        GetSingleExpertTensorDimSize(weightStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightColumnCount =
        GetSingleExpertTensorDimSize(weightStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);

    OP_TILING_CHECK(weightScaleMatrixDimSize != weightRowCount,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, (std::string(expertTypeName) + " " + weightScaleRole).c_str(),
                        (std::string("matrix dimension=") + std::to_string(weightScaleMatrixDimSize)).c_str(),
                        (std::string("The matrix dimension must equal the row count of ") + expertTypeName + " " +
                         weightRole + "(" + std::to_string(weightRowCount) + ").")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(weightScaleGroupDimSize != ops::CeilDiv(weightColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, (std::string(expertTypeName) + " " + weightScaleRole).c_str(),
                        (std::string("group dimension=") + std::to_string(weightScaleGroupDimSize)).c_str(),
                        (std::string("The group dimension must equal CeilDiv(") + expertTypeName + " " + weightRole +
                         " column count, INPUT_WEIGHT_SCALES_CEIL_ALIGN) = " +
                         std::to_string(ops::CeilDiv(weightColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN)) + ".")
                            .c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertWeightScalesTensorDim(const gert::TilingContext *context,
                                                        const ExpertWeightTensorGroupInputs &inputs,
                                                        bool isPerExpertWeightTensor, const char *nodeName)
{
    // CheckMoeAndSharedWeightInputLayouts 已保证本侧四个 TensorList 的首项 shape 存在且 rank 合法。
    auto weightScalesOneStorageShape = context->GetDynamicInputShape(inputs.weightScalesOneIndex, 0);
    auto weightScalesTwoStorageShape = context->GetDynamicInputShape(inputs.weightScalesTwoIndex, 0);
    auto weightOneStorageShape = context->GetDynamicInputShape(inputs.weightOneIndex, 0);
    auto weightTwoStorageShape = context->GetDynamicInputShape(inputs.weightTwoIndex, 0);

    ge::graphStatus checkStatus =
        CheckWeightScaleTensorDim(weightScalesOneStorageShape, weightOneStorageShape, isPerExpertWeightTensor,
                                  inputs.expertTypeName, "weight_scales1", "weight1", nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }
    checkStatus = CheckWeightScaleTensorDim(weightScalesTwoStorageShape, weightTwoStorageShape, isPerExpertWeightTensor,
                                            inputs.expertTypeName, "weight_scales2", "weight2", nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }
    const int64_t scaleOneMultiBase = GetSingleExpertTensorDimSize(
        weightScalesOneStorageShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t scaleTwoMultiBase = GetSingleExpertTensorDimSize(
        weightScalesTwoStorageShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, isPerExpertWeightTensor);
    const std::string scaleNames = std::string(inputs.expertTypeName) + " weight_scales1, weight_scales2";
    const std::string trailingDimensions =
        "[" + std::to_string(scaleOneMultiBase) + ", " + std::to_string(scaleTwoMultiBase) + "]";
    OP_TILING_CHECK(
        scaleOneMultiBase != WEIGHT_SCALE_MULTI_BASE_DIM_SIZE || scaleTwoMultiBase != WEIGHT_SCALE_MULTI_BASE_DIM_SIZE,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, scaleNames.c_str(), trailingDimensions.c_str(),
                                               "The trailing dimension of each weight scale tensor must be 2."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckContextAndXTensorDim(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                 int64_t &xDim0, const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "context", (std::to_string(contextStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of context must be 1D."),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "context dim0 = %ld", contextStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "x", (std::to_string(xStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of x must be 2D."),
                    return ge::GRAPH_FAILED);
    xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckRoutingTensorDims(const gert::TilingContext *context, const MegaMoeConfig &config,
                                              int64_t xDim0, const char *nodeName)
{
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    OP_TILING_CHECK(topkIdsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkIds"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "topkIds", (std::to_string(topkIdsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of topkIds must be 2D."),
        return ge::GRAPH_FAILED);
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkIds dim0 = %ld", topkIdsDim0);
    OP_LOGD(nodeName, "topkIds dim1 = %ld", topkIdsDim1);

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkWeights"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "topkWeights",
                        (std::to_string(topkWeightsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of topkWeights must be 2D."),
                    return ge::GRAPH_FAILED);
    const int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkWeights dim0 = %ld", topkWeightsDim0);
    OP_LOGD(nodeName, "topkWeights dim1 = %ld", topkWeightsDim1);

    OP_TILING_CHECK(xDim0 != topkIdsDim0 || xDim0 != topkWeightsDim0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "x, topkIds, topkWeights",
                        (std::string("[") + std::to_string(xDim0) + ", " + std::to_string(topkIdsDim0) + ", " +
                         std::to_string(topkWeightsDim0) + "]")
                            .c_str(),
                        "The shape [dim0] of x, topkIds, and topkWeights must be equal."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        topkIdsDim1 != topkWeightsDim1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "topkIds, topkWeights",
            (std::string("[") + std::to_string(topkIdsDim1) + ", " + std::to_string(topkWeightsDim1) + "]").c_str(),
            "The shape [dim1] of topkIds and topkWeights must be equal."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验基础输入张量的维度：context 为 1D，x / topkIds / topkWeights 为 2D，
 * 并校验三者之间的 dim0 / dim1 一致性。
 */
static ge::graphStatus CheckBasicInputTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                                const char *nodeName)
{
    int64_t xDim0 = 0;
    OP_TILING_CHECK(CheckContextAndXTensorDim(context, config, xDim0, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "context or x shape is invalid."), return ge::GRAPH_FAILED);
    return CheckRoutingTensorDims(context, config, xDim0, nodeName);
}

/*
 * 校验专家权重相关输入与输出的维度：权重布局、weight 形状、weight scale 形状、共享专家输入和输出形状。
 */
static ge::graphStatus CheckExpertWeightAndOutputTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                                           const MegaMoeExpertParams &expertParams,
                                                           const char *nodeName)
{
    OP_TILING_CHECK(CheckMoeAndSharedWeightInputLayouts(context, expertParams, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "expert weight tensor layout is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightTensorDim(context, expertParams, config.xIndex, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "weight params shape is invalid."), return ge::GRAPH_FAILED);

    if (expertParams.shared.expertCount > 0) {
        // CheckWeightTensorDim 已保证两侧 W1 column 和 W2 row 都等于同一个 x.H，且各自满足
        // W1 row = 2 * W2 column。因此跨侧只需比较 W1 row（即 GMM1 输出维 hiddenDim）。
        auto moeWeightOneShape = context->GetDynamicInputShape(expertParams.moe.inputs.weightOneIndex, 0);
        auto sharedWeightOneShape = context->GetDynamicInputShape(expertParams.shared.inputs.weightOneIndex, 0);
        int64_t moeHiddenDim = GetSingleExpertTensorDimSize(moeWeightOneShape, WEIGHT_MATRIX_ROW_DIM_INDEX,
                                                            expertParams.isPerExpertWeightTensor);
        int64_t sharedHiddenDim = GetSingleExpertTensorDimSize(sharedWeightOneShape, WEIGHT_MATRIX_ROW_DIM_INDEX,
                                                               expertParams.isPerExpertWeightTensor);
        OP_TILING_CHECK(sharedHiddenDim != moeHiddenDim,
                        OP_LOGE_FOR_INVALID_VALUE(
                            nodeName, "shared expert hiddenDim", std::to_string(sharedHiddenDim).c_str(),
                            (std::string("must equal MoE expert hiddenDim ") + std::to_string(moeHiddenDim)).c_str()),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(
        CheckExpertWeightScalesTensorDim(context, expertParams.moe.inputs, expertParams.isPerExpertWeightTensor,
                                         nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "MoE weight scale shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        expertParams.shared.expertCount > 0 &&
            CheckExpertWeightScalesTensorDim(context, expertParams.shared.inputs, expertParams.isPerExpertWeightTensor,
                                             nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "shared weight scale shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckOutputTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "output params shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验基础输入张量的 dtype：context / x / topkIds / topkWeights。
 */
static ge::graphStatus CheckBasicInputDataType(const gert::TilingContext *context, MegaMoeConfig &config,
                                               const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "context",
                                                          Ops::Base::ToString(contextDesc->GetDataType()).c_str(),
                                                          "The dtype of context must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        xDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "The dtype of x must be DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "topkIds",
                                                          Ops::Base::ToString(topkIdsDesc->GetDataType()).c_str(),
                                                          "The dtype of topkIds must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((topkWeightsDesc->GetDataType() != ge::DT_BF16) && (topkWeightsDesc->GetDataType() != ge::DT_FLOAT)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "topkWeights",
                                              Ops::Base::ToString(topkWeightsDesc->GetDataType()).c_str(),
                                              "The dtype of topkWeights must be DT_FLOAT or DT_BF16."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验输出张量的 dtype：y 与 expertTokenNums。
 */
static ge::graphStatus CheckOutputDataType(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(
        yDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "y", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                              "The dtype of y must be DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "expertTokenNums", Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str(),
                        "The dtype of expertTokenNums must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool IsNzPrimaryFormat(const gert::CompileTimeTensorDesc *desc)
{
    return static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ;
}

static ge::graphStatus CheckBasicTensorFormats(const gert::TilingContext *context, const MegaMoeConfig &config,
                                               const char *nodeName)
{
    const gert::CompileTimeTensorDesc *descs[] = {
        context->GetInputDesc(config.xIndex), context->GetInputDesc(config.topkIdsIndex),
        context->GetInputDesc(config.topkWeightsIndex), context->GetOutputDesc(config.yIndex),
        context->GetOutputDesc(config.expertTokenNumsIndex)};
    const char *names[] = {"x", "topk_ids", "topk_weights", "y", "expert_token_nums"};
    for (uint32_t index = 0; index < sizeof(descs) / sizeof(descs[0]); ++index) {
        OP_TILING_CHECK(IsNzPrimaryFormat(descs[index]),
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, names[index],
                                                   Ops::Base::ToString(descs[index]->GetStorageFormat()).c_str(),
                                                   "a non-FRACTAL_NZ format"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// GMM mode 解析已经校验 weight；这里继续校验两组 weight scale 固定为 E8M0 且不使用 NZ 主格式。
static ge::graphStatus CheckExpertScaleProperties(const gert::TilingContext *context,
                                                  const ExpertWeightTensorGroupInputs &inputs, const char *nodeName)
{
    auto scaleOneDesc = context->GetDynamicInputDesc(inputs.weightScalesOneIndex, 0);
    auto scaleTwoDesc = context->GetDynamicInputDesc(inputs.weightScalesTwoIndex, 0);
    const std::string scaleNames = std::string(inputs.expertTypeName) + " weight_scales1, weight_scales2";

    bool invalidDtype =
        scaleOneDesc->GetDataType() != ge::DT_FLOAT8_E8M0 || scaleTwoDesc->GetDataType() != ge::DT_FLOAT8_E8M0;
    const std::string scaleDtypes = "[" + Ops::Base::ToString(scaleOneDesc->GetDataType()) + ", " +
                                    Ops::Base::ToString(scaleTwoDesc->GetDataType()) + "]";
    OP_TILING_CHECK(invalidDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, scaleNames.c_str(), scaleDtypes.c_str(),
                                                           "Both weight scale dtypes must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);

    const std::string scaleFormats = "[" + Ops::Base::ToString(scaleOneDesc->GetStorageFormat()) + ", " +
                                     Ops::Base::ToString(scaleTwoDesc->GetStorageFormat()) + "]";
    OP_TILING_CHECK(
        IsNzPrimaryFormat(scaleOneDesc) || IsNzPrimaryFormat(scaleTwoDesc),
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(nodeName, scaleNames.c_str(), scaleFormats.c_str(),
                                                "Weight scale tensors must use a non-FRACTAL_NZ primary format."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMegaMoe(const gert::TilingContext *context, MegaMoeConfig &config,
                                          const MegaMoeExpertParams &expertParams, const char *nodeName)
{
    OP_TILING_CHECK(CheckBasicInputTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "basic input shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckExpertWeightAndOutputTensorDim(context, config, expertParams, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "expert weight or output shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckBasicInputDataType(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "basic input dtype is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckExpertScaleProperties(context, expertParams.moe.inputs, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "MoE expert weight scale properties are invalid."), return ge::GRAPH_FAILED);

    if (expertParams.shared.expertCount > 0) {
        OP_TILING_CHECK(CheckExpertScaleProperties(context, expertParams.shared.inputs, nodeName) == ge::GRAPH_FAILED,
                        OP_LOGE(nodeName, "shared expert weight scale properties are invalid."),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(CheckOutputDataType(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "output dtype is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckBasicTensorFormats(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "basic tensor format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 topK 与 token 隐藏维 H：取值范围，以及按拓扑和权重分形确定的 H 对齐要求。
 */
static ge::graphStatus CheckAndSetBasicInputParams(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                                   MegaMoeConfig &config, const MegaMoeExpertParams &expertParams,
                                                   const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);

    // topk范围
    OP_TILING_CHECK(
        topkIdsDim1 < MIN_TOPK || topkIdsDim1 > MAX_TOPK,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "topK", std::to_string(topkIdsDim1).c_str(), "only support [1, 32]"),
        return ge::GRAPH_FAILED);

    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    // 检查 H 范围 [1K, 8K]
    OP_TILING_CHECK(
        xDim1 < MIN_H || xDim1 > MAX_H,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "H", std::to_string(xDim1).c_str(),
            (std::string("should in [") + std::to_string(MIN_H) + ", " + std::to_string(MAX_H) + "]").c_str()),
        return ge::GRAPH_FAILED);
    bool usesW4C032WeightOne = expertParams.moe.gmmMode == GMM_MODE_A8W4_NZ ||
                               (expertParams.shared.expertCount > 0 && expertParams.shared.gmmMode == GMM_MODE_A8W4_NZ);
    // URMA/Layered 保持 1K 对齐；W4 的 NZ_C0_32 分形要求 GMM K 按 64 对齐；其余 MTE 路径按 32 对齐。
    int64_t requiredHAlignment =
        tilingData->topoType == TOPO_TYPE_URMA ? HIDDEN_DIM_BASE : (usesW4C032WeightOne ? W4_K_ALIGN : H_ALIGN);
    OP_TILING_CHECK(
        xDim1 % requiredHAlignment != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "H", std::to_string(xDim1).c_str(),
                                  (std::string("multiple of ") + std::to_string(requiredHAlignment)).c_str()),
        return ge::GRAPH_FAILED);

    tilingData->bs = static_cast<uint32_t>(bs);
    tilingData->h = static_cast<uint32_t>(xDim1);
    tilingData->topK = static_cast<uint32_t>(topkIdsDim1);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验并设置 GMM1 输出维 hiddenDim：取值范围与按拓扑确定的对齐要求。
 */
static ge::graphStatus CheckAndSetHiddenDim(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                            const MegaMoeExpertParams &expertParams, const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(expertParams.moe.inputs.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    int64_t hiddenDim = GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX,
                                                     expertParams.isPerExpertWeightTensor);
    // URMA 保持 1K 下限/1K 对齐；MTE 泛化到 [512, 8K] 且 256 对齐（gate/up 半宽按 128 对齐）。
    const int64_t minHiddenDim = tilingData->topoType == TOPO_TYPE_URMA ? HIDDEN_DIM_BASE : MTE_MIN_HIDDEN_DIM;
    const int64_t hiddenDimAlign = tilingData->topoType == TOPO_TYPE_URMA ? HIDDEN_DIM_BASE : MTE_HIDDEN_DIM_ALIGN;
    OP_TILING_CHECK(hiddenDim < minHiddenDim || hiddenDim > MAX_HIDDEN_DIM,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "hiddenDim", std::to_string(hiddenDim).c_str(),
                                              (std::string("should in [") + std::to_string(minHiddenDim) + ", " +
                                               std::to_string(MAX_HIDDEN_DIM) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hiddenDim % hiddenDimAlign != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "hiddenDim", std::to_string(hiddenDim).c_str(),
                                              (std::string("multiple of ") + std::to_string(hiddenDimAlign)).c_str()),
                    return ge::GRAPH_FAILED);

    tilingData->hiddenDim = static_cast<uint32_t>(hiddenDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetInput(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                        MegaMoeConfig &config, MegaMoeExpertParams &expertParams, const char *nodeName)
{
    OP_TILING_CHECK(CheckAndSetExpertWeightParams(context, config, expertParams, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expert weight parameters are invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAndSetExpertQuantParams(context, config, expertParams, tilingData->topoType, nodeName) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expert quantization parameters are invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(TilingCheckMegaMoe(context, config, expertParams, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check input failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        CheckAndSetCommAndExpertAttrs(context, config, expertParams, tilingData, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "communication or expert attributes are invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        CheckAndSetBasicInputParams(context, tilingData, config, expertParams, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "topK or H is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAndSetHiddenDim(context, tilingData, expertParams, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "hiddenDim is invalid."), return ge::GRAPH_FAILED);

    tilingData->isPerExpertWeightTensor = expertParams.isPerExpertWeightTensor;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetTopoType(const gert::TilingContext *context, const MegaMoeConfig &config,
                                           MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t topoType = *attrs->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    OP_TILING_CHECK(topoType != TOPO_TYPE_MTE && topoType != TOPO_TYPE_URMA,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topoType", std::to_string(topoType).c_str(),
                                              "only support MTE(0) or URMA(1)"),
                    return ge::GRAPH_FAILED);
    tilingData->topoType = topoType;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetPlatformParams(gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                                 uint32_t &aicNum, uint64_t &ubSize, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    aicNum = ascendcPlatform.GetCoreNumAic();
    OP_TILING_CHECK(aivNum == 0U || aicNum == 0U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "aivNum/aicNum",
                                              (std::to_string(aivNum) + ", " + std::to_string(aicNum)).c_str(),
                                              "should both be > 0"),
                    return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingData->aicNum = aicNum;
    tilingData->blockAivNum = aivNum;
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    context->SetScheduleMode(1); // batch model, all cores start at the same time
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FinalizeTiling(gert::TilingContext *context, MegaMoeConfig &config,
                                      const MegaMoeExpertParams &expertParams, MegaMoeTilingData *tilingData,
                                      const char *nodeName)
{
    uint64_t tilingKey = CalTilingKey(context, config, expertParams);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    WorkspaceInfo workspaceInfo((uint8_t *)0, tilingData);
    OP_TILING_CHECK(SetWorkspace(context, workspaceInfo, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Tiling set workspace Failed"), return ge::GRAPH_FAILED);

    PrintMegaMoeTilingData(tilingData, nodeName);
    PrintWorkspaceInfo(&workspaceInfo, nodeName);
    PrintPeermemInfo(tilingData, nodeName);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MegaMoe tiling check func.");

    MegaMoeTilingData *tilingData = context->GetTilingData<MegaMoeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorPtrNullptr(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "tensor pointer check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAttrPtrNullptr(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "attribute pointer check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetTopoType(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "topology type is invalid."), return ge::GRAPH_FAILED);

    MegaMoeExpertParams expertParams{};
    OP_TILING_CHECK(CheckAndSetInput(context, tilingData, config, expertParams, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Check and set input failed."), return ge::GRAPH_FAILED);

    uint32_t aicNum = 0U;
    uint64_t ubSize = 0U;
    OP_TILING_CHECK(CheckAndSetPlatformParams(context, tilingData, aicNum, ubSize, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "platform parameters are invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrAndSetTilingData(context, config, expertParams, tilingData, aicNum) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);
    SetAdaptiveBufferConfigs(context, config, expertParams, tilingData, static_cast<uint32_t>(ubSize));

    return FinalizeTiling(context, config, expertParams, tilingData, nodeName);
}

static ge::graphStatus MegaMoeTilingFunc(gert::TilingContext *context)
{
    MegaMoeConfig config;
    return MegaMoeTilingFuncImplPublic(context, config);
}

struct MegaMoeCompileInfo {};
static ge::graphStatus TilingParseForMegaMoe(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaMoe).Tiling(MegaMoeTilingFunc).TilingParse<MegaMoeCompileInfo>(TilingParseForMegaMoe);

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
inline void MegaMoeExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2Exception::Mc2ExceptionImpl(args, userdata, "MegaMoe");
}

__attribute__((constructor)) void RegisterMegaMoeExceptionFunc()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;

    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MegaMoe", "Get runtime version failed when register exception func.");
        return;
    }
    if (aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MegaMoe", "Get metadef version failed when register exception func.");
        return;
    }

    if (runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MegaMoe",
                "The runtime(%d) or metadata(%d) version is lower than the version(%d) supporting exception func.",
                runtimeVersionNum, metadefVersionNum, EXCEPTION_DUMP_SUPPORT_VERSION);
        return;
    }

    IMPL_OP(MegaMoe).ExceptionDumpParseFunc(MegaMoeExceptionImplWrapper);
}
#endif
} // namespace optiling
