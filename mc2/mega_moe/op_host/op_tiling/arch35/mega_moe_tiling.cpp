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
const static int64_t SWIGLU_GATE_UP_SPLIT_FACTOR = 2LL;
// MX 量化的 weight scale 每个专家末尾固定带 2 个 base 分量。
const static int64_t WEIGHT_SCALE_MULTI_BASE_DIM_SIZE = 2LL;
// 共享专家最多 4 个。
const static int64_t MAX_SHARED_EXPERT_NUM = 4LL;
const static int64_t UB_BLOCK_SIZE = 32LL;

const static int64_t FOUR_DIMS = 4LL;
const static int64_t THREE_DIMS = 3LL;
const static int64_t TWO_DIMS = 2LL;
const static int64_t ONE_DIM = 1LL;
const static int64_t MIN_TOPK = 1LL;
const static int64_t MAX_TOPK = 32LL;
const static int64_t MIN_EXPERT_PER_RANK = 1LL;
const static int64_t MAX_EXPERT_PER_RANK = 1024LL;
const static int64_t MIN_H = 1024LL;
const static int64_t MAX_H = 8LL * 1024LL; // 8K
const static int64_t H_ALIGN = 32LL;
const static int64_t W4_K_ALIGN = 64LL;
const static int64_t HIDDEN_DIM_BASE = 1024LL;
const static int64_t MAX_HIDDEN_DIM = 8LL * 1024LL; // 8K
// MTE 路径 N 泛化：hiddenDim(=GMM1 输出维，含 gate/up 两半) 按 GMM_TILE_N(256) 粒度放宽，
// 下限 512 保证非交织模式 gate/up 半宽至少覆盖 1 个完整 256 tile；URMA 路径维持 1K 约束不变。
const static int64_t MTE_MIN_HIDDEN_DIM = 512LL;
const static int64_t MTE_HIDDEN_DIM_ALIGN = 256LL;
const static int64_t MIN_EP_WORLD_SIZE = 2LL;
const static int64_t MAX_EP_WORLD_SIZE = 1024LL;
const static int64_t MAX_MOE_EXPERT_NUM = 2048LL;
const static int64_t INPUT_WEIGHT_SCALES_CEIL_ALIGN = 64LL;
const static int64_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 50LL;
constexpr float DEFAULT_ACTIVATION_CLAMP = std::numeric_limits<float>::max();
constexpr float DEFAULT_SWIGLU_OAI_ALPHA = 1.702f;
constexpr float DEFAULT_SWIGLU_OAI_BETA = 1.0f;

constexpr uint32_t GMM_TILE_N = 256U;
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE = 4U;
constexpr int64_t GMM_TILE_STATUS_COUNT_ALIGN = 16LL;
const static uint32_t WEIGHT_MATRIX_ROW_DIM_INDEX = 0U;
const static uint32_t WEIGHT_MATRIX_COLUMN_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MATRIX_DIM_INDEX = 0U;
const static uint32_t WEIGHT_SCALE_GROUP_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MULTI_BASE_DIM_INDEX = 2U;

struct ExpertWeightInputIndices {
    uint32_t weightOne;
    uint32_t weightTwo;
    uint32_t weightScalesOne;
    uint32_t weightScalesTwo;
};

struct ExpertParams {
    ExpertWeightInputIndices inputs;
    const char *expertTypeName;
    int64_t expertCount;
    ge::DataType quantOutDtype;
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

static void PrintMegaMoeCoreTilingData(const MegaMoeTilingData *tilingData, const char *nodeName)
{
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
    OP_LOGD(nodeName, "topkWeightsPrefetch is %d", tilingData->topkWeightsPrefetch);
    OP_LOGD(nodeName, "mGroupsPerWave is %u", tilingData->mGroupsPerWave);
}

static void PrintMegaMoeBufferConfigs(const MegaMoeTilingData *tilingData, const char *nodeName)
{
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
}

void PrintMegaMoeTilingData(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);
    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");
    PrintMegaMoeCoreTilingData(tilingData, nodeName);
    PrintMegaMoeBufferConfigs(tilingData, nodeName);
}

void PrintWorkspaceLayout(const struct WorkspaceLayout *layout, const char *nodeName)
{
    OP_LOGD(nodeName, "dispatchRevDataOffset:         %ld\n", layout->dispatchRevDataOffset);
    OP_LOGD(nodeName, "dispatchRevScaleOffset:        %ld\n", layout->dispatchRevScaleOffset);
    OP_LOGD(nodeName, "activationQuantDataOffset:     %ld\n", layout->activationQuantDataOffset);
    OP_LOGD(nodeName, "activationQuantScaleOffset:    %ld\n", layout->activationQuantScaleOffset);
    OP_LOGD(nodeName, "expertRevTokenNumsOffset:      %ld\n", layout->expertRevTokenNumsOffset);
    OP_LOGD(nodeName, "metaInfoOffset:                %ld\n", layout->metaInfoOffset);
    OP_LOGD(nodeName, "flagActivationToGmm2Offset:    %ld\n", layout->flagActivationToGmm2Offset);
    OP_LOGD(nodeName, "sharedActivationToGmm2Offset:  %ld\n", layout->sharedActivationToGmm2Offset);
    OP_LOGD(nodeName, "flagDispatchToGmm1Offset:      %ld\n", layout->flagDispatchToGmm1Offset);
    OP_LOGD(nodeName, "flagSendCntCalToUpdParamsOffset: %ld\n", layout->flagSendCntCalToUpdParamsOffset);
    OP_LOGD(nodeName, "flagGmmToEpilogueOffset:       %ld\n", layout->flagGmmToEpilogueOffset);
    OP_LOGD(nodeName, "gmm2ReadyOffset:               %ld\n", layout->gmm2ReadyOffset);
    OP_LOGD(nodeName, "gmm2CombineSyncCounterOffset:  %ld\n", layout->gmm2CombineSyncCounterOffset);
    OP_LOGD(nodeName, "gmm2MmadResOffset:             %ld\n", layout->gmm2MmadResOffset);
    OP_LOGD(nodeName, "workspaceSize:                 %ld\n", layout->workspaceSize);
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
    int64_t expertCountRecvSize = CalcExpertCountRecvSize(static_cast<int64_t>(tilingData->moeExpertPerRank),
                                                          static_cast<int64_t>(tilingData->epWorldSize));
    uint32_t activationElementsPerByte = IsA4W4GmmMode(tilingData->moeGmmMode) ? 2U : 1U;
    int64_t tokenScaleBytes =
        CalcQuantTokenScaleBytes(static_cast<int64_t>(tilingData->h), activationElementsPerByte,
                                 static_cast<int64_t>(tilingData->topK), tilingData->topkWeightsPrefetch == 1);
    int64_t quantTokenScaleSize = ops::CeilAlign(
        static_cast<int64_t>(tilingData->numMaxTokensPerRank) * tokenScaleBytes, static_cast<int64_t>(ALIGN_512));
    // 非量化 combine 每个 hidden 元素存一个 BF16（2 字节）。
    int64_t combineTokenBytes =
        CalcCombineTokenBytes(static_cast<int64_t>(tilingData->h), 2, tilingData->combineQuantMode != COMBINE_NO_QUANT);
    int64_t sendTotalNum = static_cast<int64_t>(tilingData->numMaxTokensPerRank) * tilingData->topK;
    int64_t combineSendSize = ops::CeilAlign(sendTotalNum * combineTokenBytes, static_cast<int64_t>(ALIGN_512));
    OP_LOGD(nodeName, "rankSyncInWorldSize: {%ld}\n", PEERMEM_DATA_OFFSET);
    OP_LOGD(nodeName, "maskRecvSize: {%ld}\n", maskRecvSize);
    OP_LOGD(nodeName, "expertCountRecvSize: {%ld}\n", expertCountRecvSize);
    OP_LOGD(nodeName, "quantTokenScaleSize: {%ld}\n", quantTokenScaleSize);
    OP_LOGD(nodeName, "combineSendSize: {%ld}\n", combineSendSize);
    OP_LOGD(nodeName, "total PeermemInfo Size: {%ld}\n",
            exceptionDumpRegionSize + PEERMEM_DATA_OFFSET + maskRecvSize + expertCountRecvSize + quantTokenScaleSize +
                combineSendSize);
}

static int64_t GetExpertQuantMode(ge::DataType quantOutDtype)
{
    if (quantOutDtype == ge::DT_FLOAT8_E5M2) {
        return EXPERT_QUANT_OUT_DTYPE_E5M2;
    }
    if (quantOutDtype == ge::DT_FLOAT8_E4M3FN) {
        return EXPERT_QUANT_OUT_DTYPE_E4M3FN;
    }
    return EXPERT_QUANT_OUT_DTYPE_E2M1;
}

static uint64_t CalcTilingKey(const gert::TilingContext *context, const MegaMoeExpertParams &expertParams,
                              const MegaMoeTilingData *tilingData)
{
    auto moeWeightDesc = context->GetDynamicInputDesc(expertParams.moe.inputs.weightOne, 0);
    auto sharedWeightDesc = expertParams.shared.expertCount > 0 ?
                                context->GetDynamicInputDesc(expertParams.shared.inputs.weightOne, 0) :
                                moeWeightDesc;

    int64_t topoType = TILINGKEY_TPL_MTE;

    return GET_TPL_TILING_KEY(
        static_cast<int64_t>(moeWeightDesc->GetDataType()), static_cast<int64_t>(sharedWeightDesc->GetDataType()),
        DISPATCH_QUANT_MODE_MXFP, GetExpertQuantMode(expertParams.moe.quantOutDtype),
        GetExpertQuantMode(expertParams.shared.quantOutDtype), static_cast<int64_t>(tilingData->combineQuantMode),
        topoType, static_cast<int64_t>(tilingData->topkWeightsPrefetch));
}

static ge::graphStatus CheckRequiredAttrPtrNullptr(const gert::TilingContext *context, const MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

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

    OP_CHECK_NULL_WITH_CONTEXT(context, moeExpertNumPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, epWorldSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, cclBufferSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxRecvTokenNumPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dispatchQuantModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dispatchQuantOutDtypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, sharedExpertQuantOutDtypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, combineQuantModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, commAlgPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, numMaxTokensPerRankPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, topoTypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsTypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, activationPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, activationParamsPtr);
    const size_t paramCount = activationParamsPtr->GetSize();
    const float *activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();
    if (paramCount != 0U) {
        OP_CHECK_NULL_WITH_CONTEXT(context, activationParams);
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * swiglu / swiglustep 只认一个可选参数 clamp——激活值的截断上限。
 * 不传就是不截断（默认取 float 最大值，见 SetActivationAttrParams）。
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
 * situglu 的 beta 必填：Ascend 950PR/Ascend 950DT 仅支持大于 0 的有限值。
 * linear_beta 选填，给了就切到 LINEAR 子模式，同样仅支持大于 0 的有限值。
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
    OP_TILING_CHECK(!std::isfinite(beta) || beta <= 0.0f,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "situglu_beta", std::to_string(beta).c_str(),
                                              "should be finite and greater than 0"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 2U) {
        const float linearBeta = activationParams[1];
        OP_TILING_CHECK(!std::isfinite(linearBeta) || linearBeta <= 0.0f,
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "situglu_linear_beta", std::to_string(linearBeta).c_str(),
                                                  "should be finite and greater than 0"),
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

static ge::graphStatus ValidateActivationParams(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                int64_t topoType, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);
    const size_t paramCount = activationParamsPtr->GetSize();
    const float *activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();

    const bool isSwiGlu = std::strcmp(activationPtr, "swiglu") == 0;
    const bool isSwiGluStep = std::strcmp(activationPtr, "swiglustep") == 0;
    const bool isSwiGluOai = std::strcmp(activationPtr, "swigluoai") == 0;
    const bool isSituGlu = std::strcmp(activationPtr, "situglu") == 0;
    OP_TILING_CHECK(!isSwiGlu && !isSwiGluStep && !isSwiGluOai && !isSituGlu,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr,
                                              "one of 'swiglu', 'swiglustep', 'swigluoai' or 'situglu'"),
                    return ge::GRAPH_FAILED);

    // 拓扑约束先判：URMA 只认 swiglu，激活类型不对就没必要再往下校验它的参数了。
    ge::graphStatus topoStatus = CheckActivationTopoConstraint(topoType, activationPtr, nodeName);
    if (topoStatus != ge::GRAPH_SUCCESS) {
        return topoStatus;
    }

    // 上面已经把激活类型收敛成四选一，所以这里最后一个分支就是 situglu。
    if (isSwiGlu || isSwiGluStep) {
        return CheckSwiGluActivationParams(activationParams, paramCount, nodeName);
    }
    if (isSwiGluOai) {
        return CheckSwiGluOaiActivationParams(activationParams, paramCount, nodeName);
    }
    return CheckSituGluActivationParams(activationParams, paramCount, nodeName);
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
static ge::graphStatus CheckCommDomainAttrs(const gert::TilingContext *context, const MegaMoeConfig &config,
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
static ge::graphStatus CheckExpertNumAttrs(const gert::TilingContext *context, const MegaMoeConfig &config,
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

static ge::graphStatus CheckAndSetCommAndExpertAttrs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                     const MegaMoeExpertParams &expertParams,
                                                     MegaMoeTilingData *tilingData, const char *nodeName)
{
    int64_t epWorldSize = 0;
    if (CheckCommDomainAttrs(context, config, nodeName, epWorldSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t moeExpertPerRank = 0;
    if (CheckExpertNumAttrs(context, config, nodeName, expertParams, epWorldSize, moeExpertPerRank) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingData->epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData->moeExpertPerRank = static_cast<uint32_t>(moeExpertPerRank);
    tilingData->sharedExpertNum = static_cast<uint32_t>(expertParams.shared.expertCount);
    return ge::GRAPH_SUCCESS;
}

static bool IsSupportedExpertQuantDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT8_E5M2 || dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT4_E2M1;
}

/*
 * 由单个专家组的量化输出类型、权重类型和两个权重的 format 确定 GMM 实现路径。
 * A4W4 ND 是内部保留能力：weight1 为 ND，weight2 仍为 NZ_C0_32；算子原型不因此扩展支持范围。
 */
static uint8_t ResolveExpertGmmMode(ge::DataType quantOutDtype, ge::DataType weightDtype, ge::Format weightOneFormat,
                                    ge::Format weightTwoFormat)
{
    bool isNdFormat = weightOneFormat == ge::FORMAT_ND && weightTwoFormat == ge::FORMAT_ND;
    bool isNzFormat = weightOneFormat == ge::FORMAT_FRACTAL_NZ && weightTwoFormat == ge::FORMAT_FRACTAL_NZ;
    bool isNzC032Format =
        weightOneFormat == ge::FORMAT_FRACTAL_NZ_C0_32 && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;
    bool isNdAndNzC032Format = weightOneFormat == ge::FORMAT_ND && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;
    bool isNzAndNzC032Format =
        weightOneFormat == ge::FORMAT_FRACTAL_NZ && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;

    if (quantOutDtype == ge::DT_FLOAT8_E5M2 && weightDtype == ge::DT_FLOAT8_E5M2 && isNdFormat) {
        return GMM_MODE_A8W8_ND;
    }
    if (quantOutDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT8_E4M3FN && isNdFormat) {
        return GMM_MODE_A8W8_ND;
    }
    if (quantOutDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT8_E4M3FN && isNzFormat) {
        return GMM_MODE_A8W8_NZ;
    }
    if (quantOutDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT4_E2M1 && isNzC032Format) {
        return GMM_MODE_A8W4_NZ;
    }
    if (quantOutDtype == ge::DT_FLOAT4_E2M1 && weightDtype == ge::DT_FLOAT4_E2M1 && isNdAndNzC032Format) {
        return GMM_MODE_A4W4_ND;
    }
    if (quantOutDtype == ge::DT_FLOAT4_E2M1 && weightDtype == ge::DT_FLOAT4_E2M1 && isNzAndNzC032Format) {
        return GMM_MODE_A4W4_NZ;
    }
    return std::numeric_limits<uint8_t>::max();
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
    ge::DataType moeQuantOutDtype = static_cast<ge::DataType>(*dispatchQuantOutDtypePtr);
    OP_TILING_CHECK(!IsSupportedExpertQuantDtype(moeQuantOutDtype),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "dispatch_quant_out_dtype",
                                              std::to_string(*dispatchQuantOutDtypePtr).c_str(),
                                              "fp8_e5m2, fp8_e4m3fn or fp4_e2m1"),
                    return ge::GRAPH_FAILED);

    auto sharedQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertQuantOutDtypeIndex);
    ge::DataType sharedQuantOutDtype = static_cast<ge::DataType>(*sharedQuantOutDtypePtr);
    bool hasExplicitSharedQuantDtype = sharedQuantOutDtype != ge::DT_UNDEFINED;
    OP_TILING_CHECK(hasExplicitSharedQuantDtype && !IsSupportedExpertQuantDtype(sharedQuantOutDtype),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "shared_expert_quant_out_dtype",
                                              std::to_string(*sharedQuantOutDtypePtr).c_str(),
                                              "DT_UNDEFINED, fp8_e5m2, fp8_e4m3fn or fp4_e2m1"),
                    return ge::GRAPH_FAILED);

    params.moe.quantOutDtype = moeQuantOutDtype;
    params.shared.quantOutDtype =
        params.shared.expertCount <= 0 || !hasExplicitSharedQuantDtype ? moeQuantOutDtype : sharedQuantOutDtype;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetExpertGmmMode(const gert::TilingContext *context, ExpertParams &expertParams,
                                                const char *nodeName)
{
    auto weightOneDesc = context->GetDynamicInputDesc(expertParams.inputs.weightOne, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(expertParams.inputs.weightTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);

    ge::DataType weightOneDtype = weightOneDesc->GetDataType();
    ge::DataType weightTwoDtype = weightTwoDesc->GetDataType();
    std::string weightNames = std::string(expertParams.expertTypeName) + " weight1, weight2";
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
        ResolveExpertGmmMode(expertParams.quantOutDtype, weightOneDtype, weightOneFormat, weightTwoFormat);
    std::string actualConfig = "[" + Ops::Base::ToString(expertParams.quantOutDtype) + ", " +
                               Ops::Base::ToString(weightOneDtype) + ", " + Ops::Base::ToString(weightOneFormat) +
                               ", " + Ops::Base::ToString(weightTwoFormat) + "]";
    OP_TILING_CHECK(
        expertParams.gmmMode == std::numeric_limits<uint8_t>::max(),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            nodeName, "quant output dtype, weight dtype, weight1 format, weight2 format", actualConfig.c_str(),
            (std::string(expertParams.expertTypeName) + " configuration does not form a supported GMM mode.").c_str()),
        return ge::GRAPH_FAILED);
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
        params.shared.gmmMode != params.moe.gmmMode || params.shared.quantOutDtype != params.moe.quantOutDtype;
    OP_TILING_CHECK(
        topoType != TOPO_TYPE_MTE && hasDifferentExpertQuantConfigs,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "topo_type", std::to_string(topoType).c_str(),
                                  "MTE when MoE and shared experts use different quantization configurations"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * combine 量化模式会决定 peermem 窗口里是否需要为量化输出留空间。
 */
static ge::graphStatus CheckAndSetCombineQuantAttr(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                   MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto combineQuantModePtr = context->GetAttrs()->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    OP_TILING_CHECK(
        *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_NO_QUANT &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E5M2 &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E4M3FN,
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
 * topology、专家数和 combine mode 均消费前序已经校验并写入 tilingData 的结果。
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

static ge::graphStatus CheckAndSetMaxTokensPerRankAttr(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                       MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
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
    tilingData->numMaxTokensPerRank = static_cast<uint32_t>(numMaxTokensPerRank);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetPeermemCapacityAttrs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                       int64_t yDtypeSize, MegaMoeTilingData *tilingData,
                                                       const char *nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    OP_TILING_CHECK(topkWeightsType != 0 && topkWeightsType != 1,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topkWeightsType", std::to_string(topkWeightsType).c_str(),
                                              "only support 0(disabled) or 1(enabled)"),
                    return ge::GRAPH_FAILED);

    int64_t leastCclBufferSize = CalcLeastCclBufferSize(static_cast<int64_t>(tilingData->numMaxTokensPerRank),
                                                        yDtypeSize, tilingData, topkWeightsType == 1);
    int64_t cclBufferSize = static_cast<int64_t>(*attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex)));
    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "cclBufferSize", std::to_string(cclBufferSize).c_str(),
                                              (std::string("should >= ") + std::to_string(leastCclBufferSize)).c_str()),
                    return ge::GRAPH_FAILED);
    tilingData->topkWeightsPrefetch = static_cast<int32_t>(topkWeightsType);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetMaxRecvTokenNumAttr(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                      MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t maxRecvTokenNum = static_cast<int64_t>(*attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex)));
    int64_t maxRecvTokenLimit = static_cast<int64_t>(tilingData->numMaxTokensPerRank) * tilingData->epWorldSize *
                                std::min(tilingData->topK, tilingData->moeExpertPerRank);
    OP_TILING_CHECK(
        maxRecvTokenNum < 0 || maxRecvTokenNum > maxRecvTokenLimit,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "maxRecvTokenNum", std::to_string(maxRecvTokenNum).c_str(),
                                  (std::string("should in [0, ") + std::to_string(maxRecvTokenLimit) + "]").c_str()),
        return ge::GRAPH_FAILED);

    uint32_t defaultMaxOutputSize = tilingData->numMaxTokensPerRank * tilingData->epWorldSize *
                                    std::min(tilingData->topK, tilingData->moeExpertPerRank);
    tilingData->maxOutputSize = maxRecvTokenNum != 0 ? static_cast<uint32_t>(maxRecvTokenNum) : defaultMaxOutputSize;
    return ge::GRAPH_SUCCESS;
}

/*
 * 容量属性按依赖顺序处理：先默认化单卡 token 上界，再校验 peermem 窗口，
 * 最后由接收 token 上限确定 maxOutputSize。
 */
static ge::graphStatus CheckAndSetCapacityAttrs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                MegaMoeTilingData *tilingData, const char *nodeName)
{
    if (CheckAndSetMaxTokensPerRankAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t yDtypeSize = ge::GetSizeByDataType(context->GetOutputDesc(config.yIndex)->GetDataType());
    if (CheckAndSetPeermemCapacityAttrs(context, config, yDtypeSize, tilingData, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckAndSetMaxRecvTokenNumAttr(context, config, tilingData, nodeName);
}

/*
 * 设置激活属性。前五行默认值与其后的 if 链是一个不可分割的顺序整体：
 * activation 为 swiglu 时正是靠默认值生效，if 链没有 else 分支，默认值块也不可挪到 if 链之后。
 */
static void SetActivationAttrParams(const gert::TilingContext *context, const MegaMoeConfig &config,
                                    MegaMoeTilingData *tilingData)
{
    auto attrs = context->GetAttrs();

    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);

    const size_t paramCount = activationParamsPtr->GetSize();
    const float *activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();
    tilingData->clampLimit = paramCount == 0U ? DEFAULT_ACTIVATION_CLAMP : activationParams[0];
    tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU);
    tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::DEFAULT);
    tilingData->activationAlpha = 1.0f;
    tilingData->activationBeta = 1.0f;

    if (std::strcmp(activationPtr, "situglu") == 0) {
        tilingData->clampLimit = std::numeric_limits<float>::max();
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SITU);
        tilingData->activationBeta = activationParams[0];
        if (activationParamsPtr->GetSize() == 2U) {
            tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::LINEAR);
            tilingData->activationAlpha = activationParams[1];
        } else {
            tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::DEFAULT);
            tilingData->activationAlpha = 0.0f;
        }
    } else if (std::strcmp(activationPtr, "swiglustep") == 0) {
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU_STEP);
    } else if (std::strcmp(activationPtr, "swigluoai") == 0) {
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU_OAI);
        tilingData->activationAlpha = paramCount == 0U ? DEFAULT_SWIGLU_OAI_ALPHA : activationParams[1];
        tilingData->activationBeta = paramCount == 0U ? DEFAULT_SWIGLU_OAI_BETA : activationParams[2];
    }
}

static ge::graphStatus CheckAndSetActivationParams(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                   MegaMoeTilingData *tilingData, const char *nodeName)
{
    ge::graphStatus status = ValidateActivationParams(context, config, tilingData->topoType, nodeName);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    SetActivationAttrParams(context, config, tilingData);
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
            static_cast<uint32_t>(ops::CeilAlign(maxTilesM * maxTilesN, GMM_TILE_STATUS_COUNT_ALIGN));
    }

    tilingData->mGroupsPerWave = CalcMGroupsPerWave(tilingData, aicNum);
}

static void SetDerivedTilingParams(const MegaMoeExpertParams &expertParams, MegaMoeTilingData *tilingData,
                                   const uint32_t aicNum)
{
    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->epWorldSize);
    tilingData->isPerExpertWeightTensor = expertParams.isPerExpertWeightTensor;
    tilingData->moeGmmMode = expertParams.moe.gmmMode;
    tilingData->sharedGmmMode = expertParams.shared.gmmMode;

    // 读 maxOutputSize，必须排在 CheckAndSetCapacityAttrs 之后。
    SetPrefetchAndWaveParams(tilingData, aicNum);
}

/*
 * 计算一个 token 的交织量化记录大小。dispatch peermem、SendMask UB 和共享专家量化都必须复用该口径。
 */
static uint32_t CalcQuantTokenAndScaleBytes(const MegaMoeTilingData *tilingData, uint32_t activationElementsPerByte)
{
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    return quantTokenBytes + quantScaleAlignBytes;
}

/*
 * 计算每个 dispatch ring slot 的量化 token 与 scale 拷贝区字节数（prefetch 模式再追加对齐后的 weight）。
 */
static uint32_t CalcDispatchCopyBufferBytes(const MegaMoeTilingData *tilingData, uint32_t activationElementsPerByte)
{
    uint32_t copyBufferBytes = CalcQuantTokenAndScaleBytes(tilingData, activationElementsPerByte);
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
    // fixedBufferBytes 包含 cumsumInfoTensor_ 和 expertTokenNumsOutTensor_。
    uint32_t fixedBufferBytes =
        static_cast<uint32_t>(ops::CeilAlign(
            static_cast<uint64_t>(tilingData->epWorldSize) * tilingData->moeExpertPerRank * sizeof(int32_t),
            static_cast<uint64_t>(ALIGN_32))) +
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

static void ExpandSendMaskRouteBatch(MegaMoeSendMaskBufferConfig &bufferConfig, uint64_t sendTotalNum,
                                     uint64_t alignedTotalRouteItems, uint32_t fixedBufferBytes,
                                     uint32_t availableUbBytes)
{
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) >= sendTotalNum) {
        return;
    }

    // ring 深度固定后，剩余 UB 用于两个 route tensor 和各 mask slot。
    uint32_t fixedBytesWithMaskCount =
        fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * static_cast<uint32_t>(ALIGN_32);
    uint32_t routeItemBudgetBytes =
        availableUbBytes > fixedBytesWithMaskCount ? availableUbBytes - fixedBytesWithMaskCount : 0U;
    // 两个 int32 route tensor 加 N 个 1bit mask slot：每项共 (64 + N) / 8 字节。
    uint32_t expandedRouteItems = routeItemBudgetBytes * 8U / (64U + static_cast<uint32_t>(bufferConfig.bufferCount));
    expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
    expandedRouteItems =
        static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
    if (expandedRouteItems <= static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
        return;
    }

    bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
    bufferConfig.routeBatchCount =
        static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) / bufferConfig.routeItemsPerBatch);
    bufferConfig.bufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8) + static_cast<uint32_t>(ALIGN_32);
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

    ExpandSendMaskRouteBatch(bufferConfig, sendTotalNum, alignedTotalRouteItems, fixedBufferBytes, availableUbBytes);
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
    // MTE 统一使用 per-expert AIC ready 表；group counter 只服务 layered 量化 Combine。
    if (tilingData->topoType != TOPO_TYPE_URMA || tilingData->combineQuantMode == COMBINE_NO_QUANT ||
        tilingData->moeExpertPerRank == 0U) {
        return 0U;
    }

    uint64_t logicalCoreCount = tilingData->blockAivNum;
    if (IsW4GmmMode(tilingData->moeGmmMode)) {
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
    bool useMteWaveCombine = tilingData->topoType == TOPO_TYPE_MTE;
    bool useGroupGrainedActivationFlag = tilingData->topoType == TOPO_TYPE_MTE;
    bool useGroupSyncCounters = tilingData->topoType == TOPO_TYPE_URMA;
    uint64_t maxWavesPerExpert = ops::CeilDiv<uint64_t>(tilingData->maxOutputSize, L1_TILE_M_256);
    uint64_t waveFlagSlotsPerExpert = maxWavesPerExpert * INT_CACHELINE;
    uint64_t activationFlagSlotsPerExpert = useGroupGrainedActivationFlag ? waveFlagSlotsPerExpert : INT_CACHELINE;
    uint64_t moeExpertCount = tilingData->moeExpertPerRank;

    uint64_t flagElementCount = moeExpertCount * (activationFlagSlotsPerExpert + waveFlagSlotsPerExpert +
                                                  static_cast<uint64_t>(INT_CACHELINE) * tilingData->aicNum);
    bool hasSharedW4Gmm = tilingData->sharedExpertNum > 0 && IsW4GmmMode(tilingData->sharedGmmMode);
    if (IsW4GmmMode(tilingData->moeGmmMode) || hasSharedW4Gmm) {
        flagElementCount += static_cast<uint64_t>(tilingData->aicNum) * INT_CACHELINE;
    }
    if (useMteWaveCombine) {
        flagElementCount += moeExpertCount * tilingData->aicNum * INT_CACHELINE;
    }
    if (useGroupSyncCounters) {
        flagElementCount += tilingData->combineSyncSlotCountPerExpert * moeExpertCount * INT_CACHELINE;
    }
    if (tilingData->sharedExpertNum > 0) {
        uint64_t tokenGroupCount = ops::CeilDiv<uint64_t>(tilingData->bs, L1_TILE_M_256);
        flagElementCount += tokenGroupCount * tilingData->sharedExpertNum * INT_CACHELINE;
        flagElementCount +=
            static_cast<uint64_t>(CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(tilingData->bs))) *
            tilingData->sharedExpertNum;
    }
    return flagElementCount;
}

static uint32_t CalcSendMaskFixedBufferBytes(const MegaMoeTilingData *tilingData, uint32_t moeActivationElementsPerByte,
                                             uint32_t sharedActivationElementsPerByte, bool needsIndependentSharedQuant)
{
    uint64_t totalFlagElementCount = CalcHostFlagElementCount(tilingData);
    uint32_t resetElementCountPerCore =
        static_cast<uint32_t>(ops::CeilDiv(totalFlagElementCount, static_cast<uint64_t>(tilingData->blockAivNum)));
    uint32_t resetBatchElementCount = std::min(resetElementCountPerCore, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    uint32_t resetTensorBytes =
        ops::CeilAlign(resetBatchElementCount, static_cast<uint32_t>(INT32_PER_256B)) * sizeof(int32_t);

    uint32_t moeQuantOutputBufferBytes = CalcDispatchCopyBufferBytes(tilingData, moeActivationElementsPerByte);
    uint32_t quantOutputBufferBytes = moeQuantOutputBufferBytes;
    if (needsIndependentSharedQuant) {
        // 两侧量化分时复用同一组 xOut UB，按较大的逐 token 记录预留；shared 不存 topK weight。
        uint32_t sharedQuantOutputBufferBytes =
            CalcQuantTokenAndScaleBytes(tilingData, sharedActivationElementsPerByte);
        quantOutputBufferBytes = std::max(moeQuantOutputBufferBytes, sharedQuantOutputBufferBytes);
    }

    uint32_t quantInputBufferBytes = ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_128)) * sizeof(uint16_t);
    uint32_t maxExpertCountPerCore =
        ops::CeilDiv(tilingData->epWorldSize * tilingData->moeExpertPerRank, tilingData->blockAivNum);
    uint32_t sendCountAccumulatorBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(maxExpertCountPerCore) * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32)));
    // mxTempTensor_ 占 2KB，xOutTensor_ 和 xInTensor_ 各使用双 buffer。
    return resetTensorBytes + 2U * 1024U + 2U * quantOutputBufferBytes + 2U * quantInputBufferBytes +
           sendCountAccumulatorBytes;
}

/*
 * 设置 SendMask 两套 UB 配置：先算与 ring 深度无关的固定占用，再按 expert 分核的两类 core 各算一套。
 * 本函数读 tilingData->combineSyncSlotCountPerExpert（经 CalcHostFlagElementCount），
 * 该字段必须在调用前写好。
 */
static void SetSendMaskBufferConfigs(MegaMoeTilingData *tilingData, uint32_t moeActivationElementsPerByte,
                                     uint32_t sharedActivationElementsPerByte, bool needsIndependentSharedQuant,
                                     uint32_t availableUbBytes)
{
    uint32_t sendMaskFixedBufferBytes = CalcSendMaskFixedBufferBytes(
        tilingData, moeActivationElementsPerByte, sharedActivationElementsPerByte, needsIndependentSharedQuant);

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

static ge::graphStatus SetAdaptiveBufferConfigs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                const MegaMoeExpertParams &expertParams, MegaMoeTilingData *tilingData,
                                                uint32_t availableUbBytes)
{
    const char *nodeName = context->GetNodeName();
    auto topKWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    uint32_t activationElementsPerByte = expertParams.moe.quantOutDtype == ge::DT_FLOAT4_E2M1 ? 2U : 1U;
    uint32_t sharedActivationElementsPerByte = expertParams.shared.quantOutDtype == ge::DT_FLOAT4_E2M1 ? 2U : 1U;
    bool needsIndependentSharedQuant =
        expertParams.shared.expertCount > 0 && expertParams.shared.quantOutDtype != expertParams.moe.quantOutDtype;
    // 所有 AIV 核共用同一套 dispatch UB 配置：这块布局和怎么分核无关。
    // kernel 那边如果改成按核拆 tensor，或者动了 copyTmp 的槽位布局，这里要跟着改。
    tilingData->dispatchBufferConfig =
        CalcDispatchBufferConfig(tilingData, activationElementsPerByte, availableUbBytes);

    // 先于 SendMask 组写入：CalcHostFlagElementCount 会读这个字段累加 flag 区大小，
    // 该依赖只经 tilingData 传递，调换顺序会静默改变 flag 区尺寸。
    tilingData->combineSyncSlotCountPerExpert = CalcCombineSyncSlotCountPerExpert(tilingData);

    SetSendMaskBufferConfigs(tilingData, activationElementsPerByte, sharedActivationElementsPerByte,
                             needsIndependentSharedQuant, availableUbBytes);

    ge::DataType topKWeightsDataType = topKWeightsDesc->GetDataType();
    SetUnpermuteBufferConfigs(tilingData, topKWeightsDataType, availableUbBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, const WorkspaceLayout &workspaceLayout,
                                    const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspace"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(workspaceLayout.workspaceSize == 0LL,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "workspaceSize",
                                              std::to_string(workspaceLayout.workspaceSize).c_str(), "non-zero"),
                    return ge::GRAPH_FAILED);

    int64_t workspaceSize = sysWorkspaceSize + workspaceLayout.workspaceSize + RESERVED_WORKSPACE_SIZE;
    workspace[0] = workspaceSize;

    OP_LOGD(nodeName, "sysWorkspaceSize: %ld \n", sysWorkspaceSize);
    OP_LOGD(nodeName, "mega_moe_tiling workspaceSize: %ld \n", workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckRequiredTensorPtrNullptr(const gert::TilingContext *context, const MegaMoeConfig &config)
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
    auto contextShape = context->GetInputShape(config.contextIndex);
    auto xShape = context->GetInputShape(config.xIndex);
    auto topkIdsShape = context->GetInputShape(config.topkIdsIndex);
    auto topkWeightsShape = context->GetInputShape(config.topkWeightsIndex);
    auto weightOneShape = context->GetDynamicInputShape(config.weight1Index, 0);
    auto weightTwoShape = context->GetDynamicInputShape(config.weight2Index, 0);
    auto weightScalesOneShape = context->GetDynamicInputShape(config.weightScales1Index, 0);
    auto weightScalesTwoShape = context->GetDynamicInputShape(config.weightScales2Index, 0);
    auto yShape = context->GetOutputShape(config.yIndex);
    auto expertTokenNumsShape = context->GetOutputShape(config.expertTokenNumsIndex);

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
    OP_CHECK_NULL_WITH_CONTEXT(context, contextShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);

    return ge::GRAPH_SUCCESS;
}

/*
 * arch35 当前不支持 xActiveMask 和 scales，可选输入一旦传入即拒绝。
 */
static ge::graphStatus CheckUnsupportedOptionalInputs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                      const char *nodeName)
{
    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(xActiveMaskDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "xActiveMask", "not null", "null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scalesDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "scales", "not null", "null"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDimNum(const gert::StorageShape *storageShape, uint32_t expectedDimNum,
                                         const char *tensorName, const char *nodeName)
{
    uint32_t actualDimNum = storageShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(actualDimNum != expectedDimNum,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, tensorName, (std::to_string(actualDimNum) + "D").c_str(),
                                                 (std::to_string(expectedDimNum) + "D").c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorListEntryMatchesReference(const gert::StorageShape *referenceShape,
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
static ge::graphStatus CheckTensorListConsistency(const gert::TilingContext *context, uint32_t inputIndex,
                                                  const char *inputRole, const char *expertTypeName,
                                                  uint32_t expectedDimNum, const char *nodeName)
{
    std::string inputName = std::string(inputRole) + " of " + expertTypeName;
    uint32_t tensorCount = GetDynamicInputTensorCount(context, inputIndex);
    OP_TILING_CHECK(tensorCount == 0U, OP_LOGE_WITH_INVALID_INPUT(nodeName, inputName.c_str()),
                    return ge::GRAPH_FAILED);

    auto referenceShape = context->GetDynamicInputShape(inputIndex, 0);
    auto referenceDesc = context->GetDynamicInputDesc(inputIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceDesc);
    uint32_t referenceDimNum = referenceShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(referenceDimNum != expectedDimNum,
                    OP_LOGE(nodeName, "%s[0] must be %uD.", inputName.c_str(), expectedDimNum),
                    return ge::GRAPH_FAILED);

    for (uint32_t tensorIdx = 1; tensorIdx < tensorCount; ++tensorIdx) {
        auto currentShape = context->GetDynamicInputShape(inputIndex, tensorIdx);
        auto currentDesc = context->GetDynamicInputDesc(inputIndex, tensorIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentDesc);

        if (CheckTensorListEntryMatchesReference(referenceShape, referenceDesc, currentShape, currentDesc, inputName,
                                                 tensorIdx, expectedDimNum, nodeName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertTensorListCounts(const gert::TilingContext *context, const ExpertParams &expertParams,
                                                   uint32_t weightDimNum, const char *nodeName)
{
    const auto &inputs = expertParams.inputs;
    uint32_t weightOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightOne);
    uint32_t weightTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightTwo);
    uint32_t weightScalesOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesOne);
    uint32_t weightScalesTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesTwo);
    bool tensorCountsMismatch = weightTwoTensorCount != weightOneTensorCount ||
                                weightScalesOneTensorCount != weightOneTensorCount ||
                                weightScalesTwoTensorCount != weightOneTensorCount;
    OP_TILING_CHECK(
        tensorCountsMismatch || (weightDimNum == THREE_DIMS && weightOneTensorCount != 1U),
        OP_LOGE(nodeName,
                "weight1, weight2, weight_scales1 and weight_scales2 of %s must contain the same number of tensors; "
                "stacked layout requires exactly one tensor.",
                expertParams.expertTypeName),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckStackedExpertCounts(const gert::TilingContext *context, const ExpertParams &expertParams,
                                                const char *nodeName)
{
    const auto &inputs = expertParams.inputs;
    auto weightOneShape = context->GetDynamicInputShape(inputs.weightOne, 0);
    auto weightTwoShape = context->GetDynamicInputShape(inputs.weightTwo, 0);
    auto weightScalesOneShape = context->GetDynamicInputShape(inputs.weightScalesOne, 0);
    auto weightScalesTwoShape = context->GetDynamicInputShape(inputs.weightScalesTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);

    int64_t expertCount = weightOneShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(weightTwoShape->GetStorageShape().GetDim(0) != expertCount ||
                        weightScalesOneShape->GetStorageShape().GetDim(0) != expertCount ||
                        weightScalesTwoShape->GetStorageShape().GetDim(0) != expertCount,
                    OP_LOGE(nodeName,
                            "Dim0 of weight1, weight2, weight_scales1 and weight_scales2 must have matching expert "
                            "counts for %s inputs.",
                            expertParams.expertTypeName),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 MoE 专家或共享专家的 weight1、weight2、weightScales1 和 weightScales2 四个 TensorList。
 */
static ge::graphStatus CheckExpertWeightInputGroupLayout(const gert::TilingContext *context,
                                                         const ExpertParams &expertParams, uint32_t weightDimNum,
                                                         const char *nodeName)
{
    const auto &inputs = expertParams.inputs;
    uint32_t scaleDimNum = weightDimNum + 1U;
    bool inconsistentTensorLists =
        CheckTensorListConsistency(context, inputs.weightOne, "weight1", expertParams.expertTypeName, weightDimNum,
                                   nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorListConsistency(context, inputs.weightTwo, "weight2", expertParams.expertTypeName, weightDimNum,
                                   nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorListConsistency(context, inputs.weightScalesOne, "weight_scales1", expertParams.expertTypeName,
                                   scaleDimNum, nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorListConsistency(context, inputs.weightScalesTwo, "weight_scales2", expertParams.expertTypeName,
                                   scaleDimNum, nodeName) != ge::GRAPH_SUCCESS;
    if (inconsistentTensorLists) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExpertTensorListCounts(context, expertParams, weightDimNum, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (weightDimNum == THREE_DIMS) {
        return CheckStackedExpertCounts(context, expertParams, nodeName);
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 以 MoE weight1[0] 的维数确定公共权重布局，2D 表示逐专家 TensorList，3D 表示单 tensor 堆叠。
 * 校验 MoE 的 weight1、weight2、weightScales1 和 weightScales2。
 * 检查共享专家的对应输入是否同时缺省或同时存在，并在存在时校验其布局与 MoE 一致。
 */
static ge::graphStatus CheckAndSetExpertWeightParams(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                     MegaMoeExpertParams &params, const char *nodeName)
{
    params.moe.inputs = {config.weight1Index, config.weight2Index, config.weightScales1Index,
                         config.weightScales2Index};
    params.moe.expertTypeName = "MoE expert";
    params.shared.inputs = {config.sharedWeight1Index, config.sharedWeight2Index, config.sharedWeightScales1Index,
                            config.sharedWeightScales2Index};
    params.shared.expertTypeName = "shared expert";

    auto weightOneShape = context->GetDynamicInputShape(params.moe.inputs.weightOne, 0);
    uint32_t weightDimNum = weightOneShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        weightDimNum != TWO_DIMS && weightDimNum != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "weight1", (std::to_string(weightDimNum) + "D").c_str(),
                                                 "weight1 must be one stacked 3D tensor or a list of 2D tensors."),
        return ge::GRAPH_FAILED);
    params.isPerExpertWeightTensor = weightDimNum == TWO_DIMS;

    OP_TILING_CHECK(CheckExpertWeightInputGroupLayout(context, params.moe, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "MoE expert weight layout is invalid."), return ge::GRAPH_FAILED);
    params.moe.expertCount = GetWeightExpertCount(context, params.moe.inputs.weightOne, params.isPerExpertWeightTensor);

    // 检查共享专家的四个 weight/scale 输入是否同时存在或同时缺省。
    params.shared.expertCount =
        GetWeightExpertCount(context, params.shared.inputs.weightOne, params.isPerExpertWeightTensor);
    bool hasSharedWeightOne = params.shared.expertCount > 0;
    bool hasSharedWeightTwo =
        GetWeightExpertCount(context, params.shared.inputs.weightTwo, params.isPerExpertWeightTensor) > 0;
    bool hasSharedWeightScalesOne =
        GetWeightExpertCount(context, params.shared.inputs.weightScalesOne, params.isPerExpertWeightTensor) > 0;
    bool hasSharedWeightScalesTwo =
        GetWeightExpertCount(context, params.shared.inputs.weightScalesTwo, params.isPerExpertWeightTensor) > 0;
    OP_TILING_CHECK(hasSharedWeightOne != hasSharedWeightTwo || hasSharedWeightOne != hasSharedWeightScalesOne ||
                        hasSharedWeightOne != hasSharedWeightScalesTwo,
                    OP_LOGE(nodeName, "Shared expert weights and weight scales must be provided together."),
                    return ge::GRAPH_FAILED);
    if (hasSharedWeightOne) {
        OP_TILING_CHECK(
            CheckExpertWeightInputGroupLayout(context, params.shared, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Shared expert weight layout is invalid."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * 建立完整的专家配置。三步存在严格依赖：
 * weight 布局决定专家输入组织方式，量化属性和 weight dtype/format 共同决定 GMM mode，
 * 通信域及专家数最后校验并写入与该配置匹配的本卡专家信息。
 */
static ge::graphStatus CheckAndSetExpertParams(const gert::TilingContext *context, const MegaMoeConfig &config,
                                               int64_t topoType, MegaMoeExpertParams &expertParams,
                                               MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(CheckAndSetExpertWeightParams(context, config, expertParams, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expert weight parameters are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAndSetExpertQuantParams(context, config, expertParams, topoType, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "expert quantization parameters are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAndSetCommAndExpertAttrs(context, config, expertParams, tilingData, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "communication or expert attributes are invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertWeightTensorDims(const gert::TilingContext *context, const ExpertParams &expertParams,
                                                   uint32_t xIndex, bool isPerExpertWeightTensor, const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightOne, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    auto weightTwoStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);

    // 去掉可选的专家维后，weight1 和 weight2 均按单专家二维矩阵读取行数和列数。
    const int64_t weightOneRowCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightOneColumnCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightTwoRowCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightTwoColumnCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);

    // 单专家 GMM 形状：weight1=[N, H]，weight2=[H, N/2]，x=[BS, H]。
    const gert::StorageShape *xStorageShape = context->GetInputShape(xIndex);
    const int64_t xColumnCount = xStorageShape->GetStorageShape().GetDim(1);
    const std::string commonMatrixDimensionsString = "[" + std::to_string(weightOneColumnCount) + ", " +
                                                     std::to_string(weightTwoRowCount) + ", " +
                                                     std::to_string(xColumnCount) + "]";
    OP_TILING_CHECK(weightOneColumnCount != weightTwoRowCount || weightOneColumnCount != xColumnCount,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, expertParams.expertTypeName, commonMatrixDimensionsString.c_str(),
                        "weight1 column count and weight2 row count must equal x column count."),
                    return ge::GRAPH_FAILED);

    const std::string weightRowColumnDimensionsString =
        "[" + std::to_string(weightOneRowCount) + ", " + std::to_string(weightTwoColumnCount) + "]";
    OP_TILING_CHECK(weightOneRowCount != weightTwoColumnCount * SWIGLU_GATE_UP_SPLIT_FACTOR,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, expertParams.expertTypeName, weightRowColumnDimensionsString.c_str(),
                        "weight1 row count must equal weight2 column count multiplied by 2."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightTensorDims(const gert::TilingContext *context,
                                             const MegaMoeExpertParams &expertParams, uint32_t xIndex,
                                             const char *nodeName)
{
    if (CheckExpertWeightTensorDims(context, expertParams.moe, xIndex, expertParams.isPerExpertWeightTensor,
                                    nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (expertParams.shared.expertCount <= 0) {
        return ge::GRAPH_SUCCESS;
    }
    return CheckExpertWeightTensorDims(context, expertParams.shared, xIndex, expertParams.isPerExpertWeightTensor,
                                       nodeName);
}

static ge::graphStatus CheckYTensorDims(const gert::TilingContext *context, const MegaMoeConfig &config,
                                        const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    auto yStorageShape = context->GetOutputShape(config.yIndex);
    if (CheckTensorDimNum(yStorageShape, TWO_DIMS, "y", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        yDim0 != bs || yDim1 != h,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "y", (std::string("[") + std::to_string(yDim0) + ", " + std::to_string(yDim1) + "]").c_str(),
            (std::string("The shape of y must be [bs, h] = [") + std::to_string(bs) + ", " + std::to_string(h) + "].")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertTokenNumsTensorDims(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                      uint32_t moeExpertPerRank, const char *nodeName)
{
    auto expertTokenNumsStorageShape = context->GetOutputShape(config.expertTokenNumsIndex);
    if (CheckTensorDimNum(expertTokenNumsStorageShape, ONE_DIM, "expert_token_nums", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    // expertTokenNums 仅报告 MoE 专家的 token 数，不包含共享专家。
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
static ge::graphStatus CheckWeightScaleTensorDims(const gert::StorageShape *weightScaleStorageShape,
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

static ge::graphStatus CheckExpertWeightScalesTensorDims(const gert::TilingContext *context,
                                                         const ExpertParams &expertParams, bool isPerExpertWeightTensor,
                                                         const char *nodeName)
{
    // CheckAndSetExpertWeightParams 已保证本侧四个 TensorList 的首项 shape 存在且 rank 合法。
    auto weightScalesOneStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightScalesOne, 0);
    auto weightScalesTwoStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightScalesTwo, 0);
    auto weightOneStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightOne, 0);
    auto weightTwoStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightTwo, 0);

    ge::graphStatus checkStatus =
        CheckWeightScaleTensorDims(weightScalesOneStorageShape, weightOneStorageShape, isPerExpertWeightTensor,
                                   expertParams.expertTypeName, "weight_scales1", "weight1", nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }
    checkStatus =
        CheckWeightScaleTensorDims(weightScalesTwoStorageShape, weightTwoStorageShape, isPerExpertWeightTensor,
                                   expertParams.expertTypeName, "weight_scales2", "weight2", nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    const int64_t scaleOneMultiBase = GetSingleExpertTensorDimSize(
        weightScalesOneStorageShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t scaleTwoMultiBase = GetSingleExpertTensorDimSize(
        weightScalesTwoStorageShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, isPerExpertWeightTensor);
    const std::string scaleNames = std::string(expertParams.expertTypeName) + " weight_scales1, weight_scales2";
    const std::string trailingDimensions =
        "[" + std::to_string(scaleOneMultiBase) + ", " + std::to_string(scaleTwoMultiBase) + "]";
    OP_TILING_CHECK(
        scaleOneMultiBase != WEIGHT_SCALE_MULTI_BASE_DIM_SIZE || scaleTwoMultiBase != WEIGHT_SCALE_MULTI_BASE_DIM_SIZE,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, scaleNames.c_str(), trailingDimensions.c_str(),
                                               "The trailing dimension of each weight scale tensor must be 2."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckXAndTopKTensorDims(const gert::TilingContext *context, const MegaMoeConfig &config,
                                               const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    if (CheckTensorDimNum(xStorageShape, TWO_DIMS, "x", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    if (CheckTensorDimNum(topkIdsStorageShape, TWO_DIMS, "topkIds", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    if (CheckTensorDimNum(topkWeightsStorageShape, TWO_DIMS, "topkWeights", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);

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
 * 校验非权重输入张量的维度：context 为 1D，x / topkIds / topkWeights 为 2D，
 * 并校验三者之间的 dim0 / dim1 一致性。
 */
static ge::graphStatus CheckNonWeightInputTensorDims(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                     const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    if (CheckTensorDimNum(contextStorageShape, ONE_DIM, "context", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckXAndTopKTensorDims(context, config, nodeName);
}

/*
 * 校验专家权重相关输入的维度：weight 形状、weight scale 形状以及 MoE/shared 的 hiddenDim 一致性。
 */
static ge::graphStatus CheckExpertTensorDims(const gert::TilingContext *context, const MegaMoeConfig &config,
                                             const MegaMoeExpertParams &expertParams, const char *nodeName)
{
    OP_TILING_CHECK(CheckWeightTensorDims(context, expertParams, config.xIndex, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "weight params shape is invalid."), return ge::GRAPH_FAILED);

    if (expertParams.shared.expertCount > 0) {
        // 两侧 W1 column 和 W2 row 已分别校验为 x.H，且各自满足 W1 row = 2 * W2 column。
        // 因此跨侧只需保证 W1 row（GMM1 输出维 hiddenDim）一致。
        auto moeWeightOneShape = context->GetDynamicInputShape(expertParams.moe.inputs.weightOne, 0);
        auto sharedWeightOneShape = context->GetDynamicInputShape(expertParams.shared.inputs.weightOne, 0);
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

    OP_TILING_CHECK(CheckExpertWeightScalesTensorDims(context, expertParams.moe, expertParams.isPerExpertWeightTensor,
                                                      nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "MoE weight scale shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        expertParams.shared.expertCount > 0 &&
            CheckExpertWeightScalesTensorDims(context, expertParams.shared, expertParams.isPerExpertWeightTensor,
                                              nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "shared weight scale shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验非权重输入张量的 dtype：context / x / topkIds / topkWeights。
 */
static ge::graphStatus CheckNonWeightInputDataTypes(const gert::TilingContext *context, const MegaMoeConfig &config,
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
static ge::graphStatus CheckOutputDataTypes(const gert::TilingContext *context, const MegaMoeConfig &config,
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

static ge::graphStatus CheckNonWeightTensorFormats(const gert::TilingContext *context, const MegaMoeConfig &config,
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

// GMM mode 解析已经校验 weight；这里只校验每组 weight scale 的 dtype 和 format。
static ge::graphStatus CheckExpertScaleDataTypesAndFormats(const gert::TilingContext *context,
                                                           const ExpertParams &expertParams, const char *nodeName)
{
    auto scaleOneDesc = context->GetDynamicInputDesc(expertParams.inputs.weightScalesOne, 0);
    auto scaleTwoDesc = context->GetDynamicInputDesc(expertParams.inputs.weightScalesTwo, 0);
    const std::string scaleNames = std::string(expertParams.expertTypeName) + " weight_scales1, weight_scales2";

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

static ge::graphStatus CheckTensorShapes(const gert::TilingContext *context, const MegaMoeConfig &config,
                                         const MegaMoeExpertParams &expertParams, uint32_t moeExpertPerRank,
                                         const char *nodeName)
{
    OP_TILING_CHECK(CheckNonWeightInputTensorDims(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "non-weight input shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckExpertTensorDims(context, config, expertParams, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expert tensor shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckYTensorDims(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "y tensor shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckExpertTokenNumsTensorDims(context, config, moeExpertPerRank, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expertTokenNums tensor shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDataTypesAndFormats(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                      const MegaMoeExpertParams &expertParams, const char *nodeName)
{
    OP_TILING_CHECK(CheckNonWeightInputDataTypes(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "non-weight input dtype is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckExpertScaleDataTypesAndFormats(context, expertParams.moe, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "MoE expert weight scale properties are invalid."), return ge::GRAPH_FAILED);

    if (expertParams.shared.expertCount > 0) {
        OP_TILING_CHECK(
            CheckExpertScaleDataTypesAndFormats(context, expertParams.shared, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "shared expert weight scale properties are invalid."), return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(CheckOutputDataTypes(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "output dtype is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckNonWeightTensorFormats(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "non-weight tensor format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验并设置 topK 与 token 隐藏维 H：取值范围，以及按拓扑和权重分形确定的 H 对齐要求。
 */
static ge::graphStatus CheckAndSetTokenShapeParams(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                                   const MegaMoeConfig &config, const MegaMoeExpertParams &expertParams,
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
    auto weightOneStorageShape = context->GetDynamicInputShape(expertParams.moe.inputs.weightOne, 0);
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

/*
 * shape 校验与维度提取必须作为同一阶段：只有完整 shape 契约通过后，才把 bs/topK/H/hiddenDim
 * 写入 tilingData，后续容量和 buffer 规划不再读取原始 shape。
 */
static ge::graphStatus CheckAndSetTensorDimensions(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                   const MegaMoeExpertParams &expertParams,
                                                   MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(
        CheckTensorShapes(context, config, expertParams, tilingData->moeExpertPerRank, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "tensor shapes are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAndSetTokenShapeParams(context, tilingData, config, expertParams, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "topK or H is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetHiddenDim(context, tilingData, expertParams, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "hiddenDim is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetTopoType(const gert::TilingContext *context, const MegaMoeConfig &config,
                                           MegaMoeTilingData *tilingData, const char *nodeName)
{
    int64_t topoType = *context->GetAttrs()->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
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
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CommitTilingResult(gert::TilingContext *context, const MegaMoeExpertParams &expertParams,
                                          MegaMoeTilingData *tilingData, const char *nodeName)
{
    WorkspaceLayout workspaceLayout(tilingData);
    OP_TILING_CHECK(SetWorkspace(context, workspaceLayout, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(
        ascendcPlatform.CalcTschBlockDim(tilingData->blockAivNum, tilingData->aicNum, tilingData->blockAivNum));
    context->SetScheduleMode(1); // batch model, all cores start at the same time
    uint64_t tilingKey = CalcTilingKey(context, expertParams, tilingData);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    PrintMegaMoeTilingData(tilingData, nodeName);
    PrintWorkspaceLayout(&workspaceLayout, nodeName);
    PrintPeermemInfo(tilingData, nodeName);
    return ge::GRAPH_SUCCESS;
}

/*
 * Tiling 阶段章程：
 * 1. 前置条件：集中检查必需 tensor/attr 指针，后续函数不重复判空。
 * 2. 模式配置：按 topology -> expert weight -> quant/GMM mode -> 通信与专家数的依赖顺序建立配置。
 * 3. Tensor 契约：presence、shape、dtype/format 分阶段校验；同类属性聚合，不写 tiling 结果。
 * 4. 问题规模：由已校验 tensor 提取 bs/topK/H/hiddenDim，后续只消费 tilingData，不重复读取 shape。
 * 5. 执行属性：activation -> combine quant -> capacity；容量计算消费前两阶段已经确定的模式和规模。
 * 6. 资源规划：最后读取平台资源，再计算派生字段、UB buffer 和 workspace。
 * 7. 结果提交：所有校验和计算成功后，统一写 tiling key、workspace 并输出诊断信息。
 *
 * 聚合只用于同一对象的完整不变量（如 expert 配置、capacity）；存在先后依赖或不同副作用的阶段保持分离。
 * Check 系列只校验，CheckAndSet 系列校验并写 tilingData，Set/Calc 系列不承担输入合法性校验。
 */
ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MegaMoe tiling check func.");

    MegaMoeTilingData *tilingData = context->GetTilingData<MegaMoeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    OP_TILING_CHECK(CheckRequiredTensorPtrNullptr(context, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "tensor pointer check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckUnsupportedOptionalInputs(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "unsupported optional input is present."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredAttrPtrNullptr(context, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "attribute pointer check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetTopoType(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "topology type is invalid."), return ge::GRAPH_FAILED);

    MegaMoeExpertParams expertParams{};
    OP_TILING_CHECK(CheckAndSetExpertParams(context, config, tilingData->topoType, expertParams, tilingData,
                                            nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expert configuration is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDataTypesAndFormats(context, config, expertParams, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "tensor dtypes or formats are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAndSetTensorDimensions(context, config, expertParams, tilingData, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "tensor dimensions are invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAndSetActivationParams(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "activation parameters are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetCombineQuantAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "combine quantization attribute is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAndSetCapacityAttrs(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "capacity attributes are invalid."), return ge::GRAPH_FAILED);

    uint32_t aicNum = 0U;
    uint64_t ubSize = 0U;
    OP_TILING_CHECK(CheckAndSetPlatformParams(context, tilingData, aicNum, ubSize, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "platform parameters are invalid."), return ge::GRAPH_FAILED);

    SetDerivedTilingParams(expertParams, tilingData, aicNum);
    OP_TILING_CHECK(SetAdaptiveBufferConfigs(context, config, expertParams, tilingData,
                                             static_cast<uint32_t>(ubSize)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Setting adaptive buffer configs failed."), return ge::GRAPH_FAILED);

    return CommitTilingResult(context, expertParams, tilingData, nodeName);
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
