/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "moe_distribute_dispatch_tiling_v2.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
namespace {
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t SCALES_INDEX = 2U;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
constexpr uint32_t EXPERT_SCALES_INDEX = 4U;
constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
constexpr uint32_t OUTPUT_ASSIST_INFO_INDEX = 2U;
constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;
constexpr uint32_t OUTPUT_EP_RECV_COUNTS_INDEX = 4U;
constexpr uint32_t OUTPUT_TP_RECV_COUNTS_INDEX = 5U;

constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t HALF_NUM = 2; // cumsum最多只能用一半的核
constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t OP_TYPE_ALL_GATHER = 6;

constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MAX_EP_WORLD_SIZE = 768L; // 384 * 2
constexpr int64_t MAX_EP_WORLD_SIZE_LAYERED = 256;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MAX_TP_WORLD_SIZE = 2;
constexpr int64_t MAX_TP_WORLD_SIZE_LAYERED = 1;
constexpr int64_t BS_UPPER_BOUND = 512;
constexpr int64_t BS_UPPER_BOUND_LAYERED = 256;
constexpr int64_t FULLMESH_BS_UPPER_BOUND = 256;
constexpr uint32_t H_BASIC_BLOCK_LAYERED = 32;
constexpr uint32_t RANK_NUM_PER_NODE = 16U;

constexpr uint32_t SIZE_OF_UNQUANT = 2;
constexpr uint32_t SIZE_OF_HALF = 2;
constexpr uint32_t SIZE_ALIGN_256 = 256;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t MOE_EXPERT_MAX_NUM_LAYERED = 512;
constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
constexpr int64_t K_MAX = 16;
constexpr int64_t FULLMESH_K_MAX = 12;
constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512;
constexpr uint32_t RANK_LIST_NUM = 2;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr int64_t H_MAX_LAYERED = 7168;
constexpr uint64_t TRIPLE = 3;
constexpr uint64_t ASSIST_NUM_PER_A = 128;
constexpr uint64_t EVEN_ALIGN = 2;
constexpr int64_t ELASTIC_METAINFO_OFFSET = 4;
constexpr int64_t CEIL_ALIGN32 = 8;

constexpr uint64_t STATIC_SCALE_DIM_0 = 1;
constexpr uint64_t ONE_DIM_SCALE_COL_NUM = 1;
constexpr uint64_t MX_BLOCK_SIZE = 32U;
constexpr uint64_t PERGROUP_BLOCK_SIZE = 128U;
} // namespace

// Supported x datatype in nonquant mode, the same as expandX
static const std::set<ge::DataType> NON_QUANT_DTYPE = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
                                                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeDispatchV2Info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeDispatchV2Info.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeDispatchV2Info.expertShardType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.moeDistributeDispatchV2Info.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeDispatchV2Info.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeDispatchV2Info.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeDispatchV2Info.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeDispatchV2Info.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeDispatchV2Info.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeDispatchV2Info.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", tilingData.moeDistributeDispatchV2Info.totalWinSizeEp);
    OP_LOGD(nodeName, "totalWinSizeTP is %lu.", tilingData.moeDistributeDispatchV2Info.totalWinSizeTp);
    OP_LOGD(nodeName, "hasElastic is %d.", tilingData.moeDistributeDispatchV2Info.hasElasticInfo);
    OP_LOGD(nodeName, "isPerformance is %d.", tilingData.moeDistributeDispatchV2Info.isPerformance);
    OP_LOGD(nodeName, "zeroComputeExpertNum is %d", tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum);
}

static bool CheckDynamicScalesDim(const gert::TilingContext *context, const char *nodeName, const uint32_t quantMode)
{
    const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamic_scales"),
                    return false);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT))) {
        OP_TILING_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_ONE_DIM_NUM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName, "dynamic_scales",
                            std::to_string(dynamicScalesStorageShape->GetStorageShape().GetDimNum()).c_str(),
                            (std::to_string(DYNAMIC_SCALE_ONE_DIM_NUM) + "D").c_str()),
                        return false);
        OP_LOGD(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    } else {
        OP_TILING_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_TWO_DIM_NUM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName, "dynamic_scales",
                            std::to_string(dynamicScalesStorageShape->GetStorageShape().GetDimNum()).c_str(),
                            (std::to_string(DYNAMIC_SCALE_TWO_DIM_NUM) + "D").c_str()),
                        return false);
        OP_LOGD(nodeName, "dynamicScales dim0=%ld, dim1=%ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0),
                dynamicScalesStorageShape->GetStorageShape().GetDim(1));
    }
    return true;
}

static bool CheckScaleTensorDim(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                const uint32_t quantMode, DispatchV2Config &config)
{
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scales"), return false);
        if (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) {
            // the cond is compatible with A2/A3 because static quant is only supported on A5
            OP_TILING_CHECK(scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                            OP_LOGE_FOR_INVALID_SHAPEDIM(
                                nodeName, "scales",
                                std::to_string(scalesStorageShape->GetStorageShape().GetDimNum()).c_str(), "2D"),
                            return false);
            OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
            OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
        } else {
            OP_TILING_CHECK((scalesStorageShape->GetStorageShape().GetDimNum() != ONE_DIM) &&
                                (scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS),
                            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                                nodeName, "scales",
                                std::to_string(scalesStorageShape->GetStorageShape().GetDimNum()).c_str(),
                                "scales dims must be 1D or 2D when quantMode is static quant"),
                            return false);
            auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
            OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expand_x"), return false);
            OP_TILING_CHECK((expandXDesc->GetDataType() == ge::DT_HIFLOAT8) &&
                                (scalesStorageShape->GetStorageShape().GetDimNum() != ONE_DIM),
                            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                                nodeName, "scales",
                                std::to_string(scalesStorageShape->GetStorageShape().GetDimNum()).c_str(),
                                "scales dims must be 1D when x dtype is hif8 in static quant"),
                            return false);
            OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
            if (scalesStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
                OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
            }
        }
    }
    return true;
}

// x, expertIds, scales维度校验
static bool CheckInputTensorDim(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                const uint32_t quantMode, const bool isLayered, DispatchV2Config &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "x", std::to_string(xStorageShape->GetStorageShape().GetDimNum()).c_str(), "2D"),
                    return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    if (isLayered) {
        const gert::StorageShape *expertScaleStorageShape = context->GetOptionalInputShape(config.expertScalesIndex);
        OP_TILING_CHECK(expertScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_scales"),
                        return false);
        OP_TILING_CHECK(expertScaleStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName, "expert_scales",
                            std::to_string(expertScaleStorageShape->GetStorageShape().GetDimNum()).c_str(), "2D"),
                        return false);
        const int64_t expertScalesDim0 = expertScaleStorageShape->GetStorageShape().GetDim(0);
        const int64_t expertScalesDim1 = expertScaleStorageShape->GetStorageShape().GetDim(1);
        OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesDim0);
        OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesDim1);
    }
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_ids"), return false);
    OP_TILING_CHECK(
        expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expert_ids",
                                     std::to_string(expertIdStorageShape->GetStorageShape().GetDimNum()).c_str(), "2D"),
        return false);
    const int64_t expertIdDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdDim0);
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdDim1);
    OP_TILING_CHECK(
        !CheckScaleTensorDim(context, nodeName, isScales, quantMode, config),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "scales", "invalid", "isScale Input param shape is invalid"),
        return false);
    return true;
}

// expertX, assistInfo, expertTokenNums, epRecvCount, tpRecvCount维度校验
static bool CheckCommonOutputTensorDim(const gert::TilingContext *context, const char *nodeName,
                                       const uint32_t quantMode)
{
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expand_x"), return false);
    OP_TILING_CHECK(
        expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expand_x",
                                     std::to_string(expandXStorageShape->GetStorageShape().GetDimNum()).c_str(), "2D"),
        return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        OP_TILING_CHECK(!CheckDynamicScalesDim(context, nodeName, quantMode),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "dynamic_scales", "invalid",
                                                               "CheckDynamicScalesDim failed"),
                        return false);
    }

    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "assist_info_for_combine"),
                    return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "assist_info_for_combine",
                        std::to_string(assistInfoStorageShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
                    return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_token_nums"),
                    return false);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "expert_token_nums",
                        std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
                    return false);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "ep_recv_count"),
                    return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "ep_recv_count",
                        std::to_string(epRecvCountStorageShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
                    return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tp_recv_count"),
                    return false);
    OP_TILING_CHECK(tpRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "tp_recv_count",
                        std::to_string(tpRecvCountStorageShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
                    return false);
    OP_LOGD(nodeName, "tpRecvCount dim0 = %ld", tpRecvCountStorageShape->GetStorageShape().GetDim(0));

    return true;
}

static bool CheckTensorDim(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                           const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                           const bool isPerformance, const bool isLayered, DispatchV2Config &config)
{
    OP_TILING_CHECK(
        !CheckInputTensorDim(context, nodeName, isScales, quantMode, isLayered, config),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "input_params", "invalid", "Input param shape is invalid"),
        return false);
    if (isActiveMask) {
        const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
        int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
        const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
        const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x_active_mask"),
                        return false);
        const int64_t xActiveMaskDimNum = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNum != ONE_DIM) && (xActiveMaskDimNum != TWO_DIMS)),
                        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x_active_mask",
                                                     std::to_string(xActiveMaskDimNum).c_str(), "1D or 2D"),
                        return false);
        OP_TILING_CHECK((xActiveMaskStorageShape->GetStorageShape().GetDim(0) != xDim0),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            nodeName, "x_active_mask",
                            std::to_string(xActiveMaskStorageShape->GetStorageShape().GetDim(0)).c_str(),
                            ("The dim0 of x_active_mask should equal to x dim0=" + std::to_string(xDim0)).c_str()),
                        return false);
        OP_TILING_CHECK(
            ((xActiveMaskDimNum == TWO_DIMS) && (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != expertIdDim1)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                nodeName, "x_active_mask", std::to_string(xActiveMaskStorageShape->GetStorageShape().GetDim(1)).c_str(),
                ("The dim1 of x_active_mask should equal to expert_ids dim1=" + std::to_string(expertIdDim1)).c_str()),
            return false);
    }
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "elastic_info"),
                        return false);
        OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName, "elastic_info",
                            std::to_string(elasticInfoStorageShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
                        return false);
        OP_LOGD(nodeName, "elasticInfo dim0 = %ld", elasticInfoStorageShape->GetStorageShape().GetDim(0));
    }
    if (isPerformance) {
        const gert::StorageShape *performanceInfoStorageShape =
            context->GetOptionalInputShape(config.performanceInfoIndex);
        OP_TILING_CHECK(performanceInfoStorageShape == nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(nodeName, "performance_info"), return false);
        OP_TILING_CHECK(performanceInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName, "performance_info",
                            std::to_string(performanceInfoStorageShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
                        return false);
        OP_LOGD(nodeName, "performanceInfo dim0 = %ld", performanceInfoStorageShape->GetStorageShape().GetDim(0));
    }

    OP_TILING_CHECK(
        !CheckCommonOutputTensorDim(context, nodeName, quantMode),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "output_params", "invalid", "Output param shape is invalid"),
        return false);

    return true;
}

static ge::graphStatus CheckQuantModeAndScales(const gert::TilingContext *context, const char *nodeName, bool isScales,
                                               const uint32_t quantMode, DispatchV2Config &config)
{
    OP_TILING_CHECK(!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "scales", "nullptr",
                        ("scales should not be nullptr when quant_mode is " + std::to_string(quantMode)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "scales", "not nullptr",
                        ("scales should be nullptr when quant_mode is " + std::to_string(quantMode)).c_str()),
                    return ge::GRAPH_FAILED);
    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
                        ((xDesc->GetDataType() == ge::DT_HIFLOAT8) || (xDesc->GetDataType() == ge::DT_FLOAT8_E5M2) ||
                         (xDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN) ||
                         (xDesc->GetDataType() == ge::DT_FLOAT4_E2M1) || (xDesc->GetDataType() == ge::DT_FLOAT4_E1M2)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "nullptr",
                                                          ("scales should not be nullptr when quant_mode is " +
                                                           std::to_string(quantMode) + " and x datatype is " +
                                                           Ops::Base::ToString(xDesc->GetDataType()))
                                                              .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
                        ((xDesc->GetDataType() == ge::DT_BF16) || (xDesc->GetDataType() == ge::DT_FLOAT16)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "not nullptr",
                                                          ("scales should be nullptr when quant_mode is " +
                                                           std::to_string(quantMode) + " and x datatype is " +
                                                           Ops::Base::ToString(xDesc->GetDataType()))
                                                              .c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeDispatchV2TilingFuncBase::CheckCommomOutputTensorDataType(const gert::TilingContext *context,
                                                                            const char *nodeName)
{
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "assist_info_for_combine"),
                    return false);
    OP_TILING_CHECK(
        assistInfoDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "assist_info_for_combine",
                                  std::to_string(static_cast<int64_t>(assistInfoDesc->GetDataType())).c_str(), "INT32"),
        return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_token_nums"),
                    return false);
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE(
                        nodeName, "expert_token_nums",
                        std::to_string(static_cast<int64_t>(expertTokenNumsDesc->GetDataType())).c_str(), "INT64"),
                    return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "ep_recv_count"), return false);
    OP_TILING_CHECK(epRecvCountsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(
                        nodeName, "ep_recv_count",
                        std::to_string(static_cast<int64_t>(epRecvCountsDesc->GetDataType())).c_str(), "INT32"),
                    return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tp_recv_count"), return false);
    OP_TILING_CHECK(tpRecvCountsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(
                        nodeName, "tp_recv_count",
                        std::to_string(static_cast<int64_t>(tpRecvCountsDesc->GetDataType())).c_str(), "INT32"),
                    return false);

    return true;
}

bool MoeDistributeDispatchV2TilingFuncBase::CheckCommomOtherInputTensorDataType(
    const gert::TilingContext *context, const char *nodeName, const bool isActiveMask, const bool hasElasticInfo,
    const bool isPerformance, DispatchV2Config &config)
{
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expert_ids",
                                              std::to_string(static_cast<int64_t>(expertIdDesc->GetDataType())).c_str(),
                                              "INT32"),
                    return false);

    if (isPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        OP_TILING_CHECK(performanceInfoDesc->GetDataType() != ge::DT_INT64,
                        OP_LOGE_FOR_INVALID_DTYPE(
                            nodeName, "performance_info",
                            std::to_string(static_cast<int64_t>(performanceInfoDesc->GetDataType())).c_str(), "INT64"),
                        return false);
    }
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL,
                        OP_LOGE_FOR_INVALID_DTYPE(
                            nodeName, "x_active_mask",
                            std::to_string(static_cast<int64_t>(xActiveMaskDesc->GetDataType())).c_str(), "BOOL"),
                        return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE(
                            nodeName, "elastic_info",
                            std::to_string(static_cast<int64_t>(elasticInfoDesc->GetDataType())).c_str(), "INT32"),
                        return false);
    }

    return true;
}

// 校验输入输出数据格式
static bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                              const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                              const bool isPerformance, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(
                        nodeName, "x", std::to_string(static_cast<int64_t>(xDesc->GetStorageFormat())).c_str(), "ND"),
                    return false);
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expert_ids",
                                   std::to_string(static_cast<int64_t>(expertIdDesc->GetStorageFormat())).c_str(),
                                   "ND"),
        return false);
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE_FOR_INVALID_FORMAT(
                nodeName, "scales", std::to_string(static_cast<int64_t>(scalesDesc->GetStorageFormat())).c_str(), "ND"),
            return false);
    }
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(
                            nodeName, "x_active_mask",
                            std::to_string(static_cast<int64_t>(xActiveMaskDesc->GetStorageFormat())).c_str(), "ND"),
                        return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(
                            nodeName, "elastic_info",
                            std::to_string(static_cast<int64_t>(elasticInfoDesc->GetStorageFormat())).c_str(), "ND"),
                        return false);
    }
    if (isPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(performanceInfoDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(
                            nodeName, "performance_info",
                            std::to_string(static_cast<int64_t>(performanceInfoDesc->GetStorageFormat())).c_str(),
                            "ND"),
                        return false);
    }
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expand_x",
                                   std::to_string(static_cast<int64_t>(expandXDesc->GetStorageFormat())).c_str(), "ND"),
        return false);
    if (quantMode >= static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(
                            nodeName, "dynamic_scales",
                            std::to_string(static_cast<int64_t>(dynamicScalesDesc->GetStorageFormat())).c_str(), "ND"),
                        return false);
    }
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "assist_info_for_combine",
                                   std::to_string(static_cast<int64_t>(assistInfoDesc->GetStorageFormat())).c_str(),
                                   "ND"),
        return false);
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(
                        nodeName, "expert_token_nums",
                        std::to_string(static_cast<int64_t>(expertTokenNumsDesc->GetStorageFormat())).c_str(), "ND"),
                    return false);
    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "ep_recv_count",
                                   std::to_string(static_cast<int64_t>(epRecvCountsDesc->GetStorageFormat())).c_str(),
                                   "ND"),
        return false);
    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "tp_recv_count",
                                   std::to_string(static_cast<int64_t>(tpRecvCountsDesc->GetStorageFormat())).c_str(),
                                   "ND"),
        return false);
    return true;
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context, const char *nodeName,
                                           std::string &groupEp, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrGroupEpIndex));
        OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                            (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "group_ep", groupEpPtr ? groupEpPtr : "nullptr",
                                                              "group_ep is invalid, length should be in [1, 128)"),
                        return ge::GRAPH_FAILED);
        groupEp = std::string(groupEpPtr);
    }
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrTpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrTpRankIdIndex);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrSharedExpertNumIndex));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertRankNumIndex);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    // 判空
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "comm_alg"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "ep_world_size"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tp_world_size"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "ep_rank_id"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tp_rank_id"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_shard_type"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "shared_expert_num"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "shared_expert_rank_num"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moe_expert_num"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "quant_mode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_token_nums_type"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "zero_expert_num"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "copy_expert_num"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "const_expert_num"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupAttrParams(const gert::TilingContext *context, const char *nodeName,
                                            std::string &groupTp, bool &isLayered, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrTpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrTpRankIdIndex);
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int64_t>(config.attrCommAlgIndex));
    int64_t epWorldSize = *epWorldSizePtr;
    isLayered = strcmp(commAlgPtr, "hierarchy") == 0; // isLayered赋值
    int64_t maxEpworldsize = isLayered ? MAX_EP_WORLD_SIZE_LAYERED : MAX_EP_WORLD_SIZE;
    int64_t maxTpworldsize = isLayered ? MAX_TP_WORLD_SIZE_LAYERED : MAX_TP_WORLD_SIZE;
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > maxEpworldsize),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(),
                        ("ep_world_size is invalid, only support [" + std::to_string(MIN_EP_WORLD_SIZE) + ", " +
                         std::to_string(maxEpworldsize) + "]")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (isLayered && (epWorldSize % RANK_NUM_PER_NODE != 0)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(),
            ("ep_world_size should be " + std::to_string(RANK_NUM_PER_NODE) + " aligned when using hierarchy").c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > maxTpworldsize),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "tp_world_size", std::to_string(*tpWorldSizePtr).c_str(),
                        ("tp_world_size is invalid, only support [0, " + std::to_string(maxTpworldsize) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "ep_rank_id", std::to_string(*epRankIdPtr).c_str(),
                        ("ep_rank_id is invalid, only support [0, " + std::to_string(epWorldSize) + ")").c_str()),
                    return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int64_t>(config.attrGroupTpIndex));
        OP_TILING_CHECK(
            (*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                nodeName, "tp_rank_id", std::to_string(*tpRankIdPtr).c_str(),
                ("tp_rank_id is invalid, only support [0, " + std::to_string(*tpWorldSizePtr) + ")").c_str()),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                            (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "group_tp", groupTpPtr ? groupTpPtr : "nullptr",
                                                              "group_tp is invalid, length should be in [1, 128)"),
                        return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "tp_rank_id",
                                                              std::to_string(*tpRankIdPtr).c_str(),
                                                              "tp_rank_id is invalid, no TP mode only support 0"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::CheckOtherAttrParams(const gert::TilingContext *context,
                                                                            const char *nodeName, bool &isSetFullMeshV2,
                                                                            DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    if (config.isMc2Context) {
        OP_TILING_CHECK((strcmp(commAlgPtr, "hierarchy") == 0),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "comm_alg", commAlgPtr,
                                                              "comm_alg doesn't support hierarchy when using context"),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(*expertShardPtr != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "expert_shard_type",
                                                          std::to_string(*expertShardPtr).c_str(),
                                                          "expert_shard_type is invalid, only support 0"),
                    return ge::GRAPH_FAILED);

    if (CheckQuantModePtr(quantModePtr, nodeName) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 0) && (*expertTokenNumsTypePtr != 1),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "expert_token_nums_type",
                                                          std::to_string(*expertTokenNumsTypePtr).c_str(),
                                                          "expert_token_nums_type is invalid, only support 0 or 1"),
                    return ge::GRAPH_FAILED);
    // A5 已作校验，这里只校验 A3
    if (CheckCommAlgPtr(commAlgPtr, nodeName) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    isSetFullMeshV2 = ((strcmp(commAlgPtr, "fullmesh_v2") == 0) ? true : false);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 isSetFullMeshV2 = %d\n", isSetFullMeshV2);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::CheckCommAttrParams(const gert::TilingContext *context,
                                                                           const char *nodeName, std::string &groupTp,
                                                                           bool &isSetFullMeshV2, bool &isLayered,
                                                                           DispatchV2Config &config)
{
    // 校验 epWorldSize tpWorldSize epRankId tpRankId
    OP_TILING_CHECK(
        CheckGroupAttrParams(context, nodeName, groupTp, isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "group_params", "invalid", "CheckGroupAttrParams failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOtherAttrParams(context, nodeName, isSetFullMeshV2, config) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "other_params", "invalid", "CheckOtherAttrParams failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertAttrParams(const gert::TilingContext *context, const char *nodeName, bool isLayered,
                                             DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrSharedExpertNumIndex));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertRankNumIndex);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    OP_TILING_CHECK((zeroExpertNum < 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "zero_expert_num",
                                                          std::to_string(zeroExpertNum).c_str(),
                                                          "zero_expert_num is invalid, should be >= 0"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((copyExpertNum < 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "copy_expert_num",
                                                          std::to_string(copyExpertNum).c_str(),
                                                          "copy_expert_num is invalid, should be >= 0"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((constExpertNum < 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "const_expert_num",
                                                          std::to_string(constExpertNum).c_str(),
                                                          "const_expert_num is invalid, should be >= 0"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "zeroExpertNum=%ld,copyExpertNum= %ld, constExpertNum=%ld", zeroExpertNum, copyExpertNum,
            constExpertNum);
    OP_TILING_CHECK(zeroComputeExpertNum + moeExpertNum > INT32_MAX,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "total_expert_num",
                                                          std::to_string(zeroComputeExpertNum + moeExpertNum).c_str(),
                                                          "total expert num exceeds INT32_MAX"),
                    return ge::GRAPH_FAILED);
    if (isLayered) {
        OP_TILING_CHECK((sharedExpertRankNum != 0),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "shared_expert_rank_num", std::to_string(sharedExpertRankNum).c_str(),
                            "shared_expert_rank_num is invalid in hierarchy mode, only support 0"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            (*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                nodeName, "shared_expert_num", std::to_string(*sharedExpertNumPtr).c_str(),
                ("shared_expert_num is invalid, only support [0, " + std::to_string(MAX_SHARED_EXPERT_NUM) + "]")
                    .c_str()),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                nodeName, "shared_expert_rank_num", std::to_string(sharedExpertRankNum).c_str(),
                ("shared_expert_rank_num is invalid, only support [0, " + std::to_string(epWorldSize) + ")").c_str()),
            return ge::GRAPH_FAILED);
    }
    int64_t moeExpertMaxNum = isLayered ? MOE_EXPERT_MAX_NUM_LAYERED : MOE_EXPERT_MAX_NUM;
    OP_TILING_CHECK(
        (moeExpertNum <= 0) || (moeExpertNum > moeExpertMaxNum),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "moe_expert_num", std::to_string(moeExpertNum).c_str(),
            ("moe_expert_num is invalid, only support (0, " + std::to_string(moeExpertMaxNum) + "]").c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, const char *nodeName, DispatchV2Config &config,
                                     MoeDistributeDispatchV2TilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrSharedExpertNumIndex));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertRankNumIndex);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrTpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrTpRankIdIndex);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));

    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    tilingData.moeDistributeDispatchV2Info.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData.moeDistributeDispatchV2Info.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeDispatchV2Info.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeDispatchV2Info.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeDispatchV2Info.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeDispatchV2Info.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchV2Info.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);
    tilingData.moeDistributeDispatchV2Info.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData.moeDistributeDispatchV2Info.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum = static_cast<int32_t>(zeroComputeExpertNum);
    if ((tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum == 0U) &&
        (tilingData.moeDistributeDispatchV2Info.sharedExpertNum == 1U)) {
        tilingData.moeDistributeDispatchV2Info.sharedExpertNum = 0U;
    }
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 zeroComputeExpertNum = %d\n",
            tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::GetAttrAndSetTilingData(
    const gert::TilingContext *context, const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData,
    std::string &groupEp, std::string &groupTp, bool &isSetFullMeshV2, bool &isLayered, DispatchV2Config &config)
{
    OP_TILING_CHECK(
        CheckAttrPtrNullptr(context, nodeName, groupEp, config) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "attr_params", "invalid", "params check null failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckCommAttrParams(context, nodeName, groupTp, isSetFullMeshV2, isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "comm_params", "invalid", "params shape is invalid"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckExpertAttrParams(context, nodeName, isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "expert_params", "invalid", "params dataType is invalid"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        SetAttrParams(context, nodeName, config, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "attr_params", "invalid", "set attr params failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool CheckSharedAttrs(const char *nodeName, const MoeDistributeDispatchV2TilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK((sharedExpertNum == 0U) && (sharedExpertRankNum > 0U),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "shared_expert_rank_num", std::to_string(sharedExpertRankNum).c_str(),
                        "shared_expert_rank_num is invalid, only support 0 when shared_expert_num is 0"),
                    return false);
    OP_TILING_CHECK((sharedExpertNum > 0U) && (sharedExpertRankNum == 0U),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "shared_expert_num", std::to_string(sharedExpertNum).c_str(),
                        "shared_expert_num is invalid, only support 0 when shared_expert_rank_num is 0"),
                    return false);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        OP_TILING_CHECK(((sharedExpertRankNum % sharedExpertNum) != 0U),
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            nodeName, "shared_expert_rank_num and shared_expert_num",
                            (std::to_string(sharedExpertRankNum) + " and " + std::to_string(sharedExpertNum)).c_str(),
                            "shared_expert_rank_num should be divisible by shared_expert_num"),
                        return false);
    }

    return true;
}

static bool CheckCommAlgAttrs(const gert::TilingContext *context, const char *nodeName,
                              const MoeDistributeDispatchV2TilingData &tilingData, bool isSetFullMeshV2,
                              DispatchV2Config &config, bool isLayered)
{
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    bool hasElasticInfo = tilingData.moeDistributeDispatchV2Info.hasElasticInfo;
    int32_t zeroComputeExpertNum = tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum;
    bool isExpertMask = tilingData.moeDistributeDispatchV2Info.isExpertMask;
    bool isPerformance = tilingData.moeDistributeDispatchV2Info.isPerformance;
    // 获取bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    uint32_t bs = static_cast<uint32_t>(xDim0);

    // 获取topk
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    uint32_t k = static_cast<uint32_t>(expertIdsDim1);

    // 检查comm_alg和tpWorldSize是否冲突
    OP_TILING_CHECK(isSetFullMeshV2 && (tpWorldSize == TP_WORLD_SIZE_TWO),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "comm_alg and tp_world_size", "fullmesh_v2 and 2",
                                                           "When comm_alg is fullmesh_v2, tp_world_size cannot be 2"),
                    return false);
    OP_TILING_CHECK(isSetFullMeshV2 && (bs > FULLMESH_BS_UPPER_BOUND),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "bs", std::to_string(bs).c_str(),
                                                          ("When comm_alg is fullmesh_v2, bs should be between [1, " +
                                                           std::to_string(FULLMESH_BS_UPPER_BOUND) + "]")
                                                              .c_str()),
                    return false);
    OP_TILING_CHECK(
        isSetFullMeshV2 && (k > FULLMESH_K_MAX),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "k", std::to_string(k).c_str(),
            ("When comm_alg is fullmesh_v2, k should be between [1, " + std::to_string(FULLMESH_K_MAX) + "]").c_str()),
        return false);
    OP_TILING_CHECK((isLayered && hasElasticInfo),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "comm_alg and elastic_info",
                                                           "hierarchy and enabled",
                                                           "Cannot support elastic_info when comm_alg is hierarchy"),
                    return false);
    OP_TILING_CHECK(
        (isLayered && (zeroComputeExpertNum > 0)),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "comm_alg and zero_compute_expert_num",
                                               "hierarchy and " + std::to_string(zeroComputeExpertNum),
                                               "Cannot support zero_compute_expert_num when comm_alg is hierarchy"),
        return false);
    OP_TILING_CHECK(
        (isLayered && isExpertMask),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "comm_alg and x_active_mask", "hierarchy and 2D",
                                               "Cannot support 2D x_active_mask when comm_alg is hierarchy"),
        return false);
    OP_TILING_CHECK(
        (isLayered && isPerformance),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "comm_alg and performance_info", "hierarchy and enabled",
                                               "Cannot support performance_info when comm_alg is hierarchy"),
        return false);

    return true;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext *context, const char *nodeName,
                                  MoeDistributeDispatchV2TilingData &tilingData, uint32_t &localMoeExpertNum,
                                  bool isActiveMask, bool isSetFullMeshV2, bool isLayered, DispatchV2Config &config)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    OP_TILING_CHECK(!CheckSharedAttrs(nodeName, tilingData),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "shared_expert_attrs", "invalid",
                                                           "Check shared expert related attributes failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckCommAlgAttrs(context, nodeName, tilingData, isSetFullMeshV2, config, isLayered),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "comm_alg_attrs", "invalid",
                                                           "Check comm_alg related attributes failed"),
                    return ge::GRAPH_FAILED);
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(
        moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            nodeName, "moe_expert_num and ep_world_size",
            (std::to_string(moeExpertNum) + " and " + std::to_string(epWorldSize - sharedExpertRankNum)).c_str(),
            "moe_expert_num should be divisible by (ep_world_size - shared_expert_rank_num)"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localMoeExpertNum <= 0) || (localMoeExpertNum * epWorldSize > LOCAL_EXPERT_MAX_SIZE),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "local_moe_expert_num",
                                                          std::to_string(localMoeExpertNum).c_str(),
                                                          ("local_moe_expert_num is invalid, should be in [1, " +
                                                           std::to_string(LOCAL_EXPERT_MAX_SIZE / epWorldSize) + "]")
                                                              .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tpWorldSize > 1) && (localMoeExpertNum > 1),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        nodeName, "tp_world_size and local_moe_expert_num",
                        (std::to_string(tpWorldSize) + " and " + std::to_string(localMoeExpertNum)).c_str(),
                        "Cannot support multi-moe_expert in a rank when tp_world_size > 1"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tpWorldSize > 1) && (tilingData.moeDistributeDispatchV2Info.hasElasticInfo),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "tp_world_size and elastic_info",
                                                           std::to_string(tpWorldSize).c_str(),
                                                           "Cannot support elastic_info when tp_world_size > 1"),
                    return ge::GRAPH_FAILED);
    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t bsUpperBound = isLayered ? BS_UPPER_BOUND_LAYERED : BS_UPPER_BOUND;
    OP_TILING_CHECK((xDim0 > bsUpperBound) || (xDim0 <= 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "bs", std::to_string(xDim0).c_str(),
                        ("bs is invalid, should be in [1, " + std::to_string(bsUpperBound) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.bs = static_cast<uint32_t>(xDim0);

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(config.attrGlobalBsIndex);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "global_bs"), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n", *globalBsPtr, xDim0,
            epWorldSize);
    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
                                            ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "global_bs", std::to_string(*globalBsPtr).c_str(),
                        ("global_bs is invalid, only support 0 or max_bs * ep_world_size, bs=" + std::to_string(xDim0) +
                         ", ep_world_size=" + std::to_string(epWorldSize))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*globalBsPtr > (xDim0 * static_cast<int64_t>(epWorldSize))) && isActiveMask),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        nodeName, "global_bs and x_active_mask",
                        (std::to_string(*globalBsPtr) + " and enabled").c_str(),
                        "Different bs on different rank cannot work when x_active_mask is enabled"),
                    return ge::GRAPH_FAILED);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTwoDimScalesShape(const gert::TilingContext *context, const char *nodeName,
                                              const MoeDistributeDispatchV2TilingData &tilingData,
                                              const int64_t scalesDim0, const int64_t scalesDim1,
                                              DispatchV2Config &config)
{
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    if (sharedExpertRankNum != 0U) {
        OP_TILING_CHECK(
            scalesDim0 != (moeExpertNum + sharedExpertNum),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                nodeName, "scales",
                (std::to_string(scalesDim0) + " and " + std::to_string(moeExpertNum + sharedExpertNum)).c_str(),
                "scales dim0 should equal to moe_expert_num + shared_expert_num"),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(scalesDim0 != moeExpertNum,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            nodeName, "scales",
                            (std::to_string(scalesDim0) + " and " + std::to_string(moeExpertNum)).c_str(),
                            "scales dim0 should equal to moe_expert_num"),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(xDim1 != scalesDim1,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "scales", (std::to_string(xDim1) + " and " + std::to_string(scalesDim1)).c_str(),
                        "scales dim1 should equal to x dim1"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(const gert::TilingContext *context, const char *nodeName,
                                        MoeDistributeDispatchV2TilingData &tilingData, const uint32_t quantMode,
                                        const bool isScales, const bool isSharedExpert, const bool hasElasticInfo,
                                        const bool isPerformance, const int64_t localMoeExpertNum, bool isLayered,
                                        DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;

    uint32_t A = 0U;
    uint32_t globalBs = tilingData.moeDistributeDispatchV2Info.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    int64_t hMax = isLayered ? H_MAX_LAYERED : H_MAX;
    int64_t hMin = isLayered ? 0 : H_MIN;
    OP_TILING_CHECK(
        (isLayered && (xDim1 % H_BASIC_BLOCK_LAYERED != 0)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "h", std::to_string(xDim1).c_str(),
            ("h should be " + std::to_string(H_BASIC_BLOCK_LAYERED) + " aligned when using hierarchy").c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((xDim1 < hMin) || (xDim1 > hMax),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "h", std::to_string(xDim1).c_str(),
                        ("h should be in [" + std::to_string(hMin) + ", " + std::to_string(hMax) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.h = static_cast<uint32_t>(xDim1);

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    if ((expandXDesc->GetDataType() == ge::DT_FLOAT4_E2M1) || (expandXDesc->GetDataType() == ge::DT_FLOAT4_E1M2)) {
        OP_TILING_CHECK(xDim1 % EVEN_ALIGN,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "h", std::to_string(xDim1).c_str(),
                            "When expandx data type is FLOAT4_E2M1/FLOAT4_E1M2, the h should be even"),
                        return ge::GRAPH_FAILED);
    }

    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "x and expert_ids",
                        (std::to_string(xDim0) + " and " + std::to_string(expertIdsDim0)).c_str(),
                        "x dim0 should equal to expert_ids dim0"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX) ||
                        (expertIdsDim1 > moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "k", std::to_string(expertIdsDim1).c_str(),
                        ("k should be in (0, min(" + std::to_string(K_MAX) +
                         ", moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num=" +
                         std::to_string(moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum) + ")]")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    if (isLayered) {
        const gert::StorageShape *expertScaleStorageShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);
        const int64_t expertScalesDim0 = expertScaleStorageShape->GetStorageShape().GetDim(0);
        const int64_t expertScalesDim1 = expertScaleStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK((expertScalesDim0 != expertIdsDim0) || (expertScalesDim1 != expertIdsDim1),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            nodeName, "expert_scales and expert_ids",
                            (std::to_string(expertScalesDim0) + "," + std::to_string(expertScalesDim1) + " and " +
                             std::to_string(expertIdsDim0) + "," + std::to_string(expertIdsDim1))
                                .c_str(),
                            "expert_scales shape should equal to expert_ids shape"),
                        return ge::GRAPH_FAILED);
    }
    tilingData.moeDistributeDispatchV2Info.k = static_cast<uint32_t>(expertIdsDim1);

    // 校验scales的维度
    uint64_t h = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bs = tilingData.moeDistributeDispatchV2Info.bs;
    uint64_t scalesRow = 0;
    uint64_t scalesCol = 0;
    uint32_t scalesTypeSize = 0;
    uint64_t scalesCount = 0;
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scales"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scales_desc"),
                        return ge::GRAPH_FAILED);
        size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        scalesRow = static_cast<uint64_t>(scalesDim0);
        scalesTypeSize = ge::GetSizeByDataType(scalesDesc->GetDataType());
        if (scalesDimNum == ONE_DIM) {
            // A3不会进此分支
            auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
            OP_TILING_CHECK(
                (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) &&
                    (expandXDesc->GetDataType() == ge::DT_INT8) && (scalesDim0 != static_cast<int64_t>(h)) &&
                    (scalesDim0 != static_cast<int64_t>(STATIC_SCALE_DIM_0)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales_dim0", std::to_string(scalesDim0).c_str(),
                                                      ("The expected scales_dim0 is " + std::to_string(h) + " or " +
                                                       std::to_string(STATIC_SCALE_DIM_0) + " in static quant")
                                                          .c_str()),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK((quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) &&
                                (expandXDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDim0 != STATIC_SCALE_DIM_0),
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                nodeName, "scales_dim0", std::to_string(scalesDim0).c_str(),
                                "The expected scales_dim0 is 1 when expand_x datatype is hif8 in static quant"),
                            return ge::GRAPH_FAILED);
            scalesCol = ONE_DIM_SCALE_COL_NUM;
            scalesCount = static_cast<uint64_t>(scalesDim0);
        } else if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(
                scalesDim0 != bs,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    nodeName, "scales_dim0", std::to_string(scalesDim0).c_str(),
                    ("The expected scales_dim0 is " + std::to_string(bs) + " when scales is not null in non-quant")
                        .c_str()),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                scalesDim1 > h,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales_dim1", std::to_string(scalesDim1).c_str(),
                                                      ("The expected scales_dim1 should be less than or equal to " +
                                                       std::to_string(h) + " when scales is not null in non-quant")
                                                          .c_str()),
                return ge::GRAPH_FAILED);
            scalesCol = static_cast<uint64_t>(scalesDim1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        } else {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(
                CheckTwoDimScalesShape(context, nodeName, tilingData, scalesDim0, scalesDim1, config) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "scales", "invalid", "CheckTwoDimScalesShape failed"),
                return ge::GRAPH_FAILED);
            scalesCol = static_cast<uint64_t>(scalesDim1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        }
    }
    tilingData.moeDistributeDispatchV2Info.scalesRow = scalesRow;
    tilingData.moeDistributeDispatchV2Info.scalesCol = scalesCol;
    tilingData.moeDistributeDispatchV2Info.scalesCount = scalesCount;
    tilingData.moeDistributeDispatchV2Info.scalesTypeSize = scalesTypeSize;

    uint32_t rankNumPerSharedExpert = 0;
    uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t maxBs = globalBs / epWorldSizeU32;
    uint32_t maxSharedGroupNum = 0;
    if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) { // 除零保护
        rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
    }
    if (isSharedExpert) { // 本卡为共享专家
        A = maxBs * maxSharedGroupNum;
    } else { // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, expertIdsDim1);
    }

    // 校验elasticInfo的维度，并更新一下最大输出的值
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        const int64_t elasticInfoDim0 = elasticInfoStorageShape->GetStorageShape().GetDim(0);
        const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);

        OP_TILING_CHECK(
            elasticInfoDim0 != (ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * epWorldSize),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                nodeName, "elastic_info_dim0", std::to_string(elasticInfoDim0).c_str(),
                ("elastic_info_dim0 should equal to " +
                 std::to_string(ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * epWorldSize) + " (4 + 2 * ep_world_size)")
                    .c_str()),
            return ge::GRAPH_FAILED);
        A = std::max(static_cast<int64_t>(maxBs * maxSharedGroupNum),
                     globalBs * std::min(localMoeExpertNum, expertIdsDim1));
    }
    tilingData.moeDistributeDispatchV2Info.a = A;

    if (isPerformance) {
        const gert::StorageShape *performanceInfoStorageShape =
            context->GetOptionalInputShape(config.performanceInfoIndex);
        const int64_t performanceInfoDim0 = performanceInfoStorageShape->GetStorageShape().GetDim(0);
        const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);

        OP_TILING_CHECK(
            performanceInfoDim0 != epWorldSize,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                nodeName, "performance_info_dim0", std::to_string(performanceInfoDim0).c_str(),
                ("performance_info_dim0 should equal to ep_world_size=" + std::to_string(epWorldSize)).c_str()),
            return ge::GRAPH_FAILED);
    }

    // 校验expandX的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        expandXDim0 < tpWorldSize * static_cast<int64_t>(A),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "expand_x_dim0", std::to_string(expandXDim0).c_str(),
            ("expand_x_dim0 should be greater than or equal to A*tp_world_size=" + std::to_string(tpWorldSize * A))
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        xDim1 != expandXDim1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "x and expand_x",
                                               (std::to_string(xDim1) + " and " + std::to_string(expandXDim1)).c_str(),
                                               "expand_x dim1 should equal to x dim1"),
        return ge::GRAPH_FAILED);

    // 校验dynamicScales的维度
    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        // Dim0
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(dynamicScalesDim0 < static_cast<int64_t>(A) * tpWorldSize,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "dynamic_scales_dim0", std::to_string(dynamicScalesDim0).c_str(),
                            ("dynamic_scales_dim0 should be greater than or equal to A*tp_world_size=" +
                             std::to_string(A * tpWorldSize))
                                .c_str()),
                        return ge::GRAPH_FAILED);
        if (quantMode != static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
            const uint64_t dynamicScalesDim1 =
                static_cast<uint64_t>(dynamicScalesStorageShape->GetStorageShape().GetDim(1));
            OP_TILING_CHECK(
                (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) &&
                    (dynamicScalesDim1 !=
                     ops::CeilAlign(static_cast<uint64_t>(ops::CeilDiv(h, MX_BLOCK_SIZE)), EVEN_ALIGN)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    nodeName, "dynamic_scales_dim1", std::to_string(dynamicScalesDim1).c_str(),
                    ("dynamic_scales_dim1 should be equal to " +
                     std::to_string(ops::CeilAlign(static_cast<uint64_t>(ops::CeilDiv(h, MX_BLOCK_SIZE)), EVEN_ALIGN)) +
                     " and even when quant_mode is MX")
                        .c_str()),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK((dynamicScalesDim1 != ops::CeilDiv(h, PERGROUP_BLOCK_SIZE)) &&
                                (quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)),
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                nodeName, "dynamic_scales_dim1", std::to_string(dynamicScalesDim1).c_str(),
                                ("dynamic_scales_dim1 should be equal to " +
                                 std::to_string(ops::CeilDiv(h, PERGROUP_BLOCK_SIZE)) + " when quant_mode is pergroup")
                                    .c_str()),
                            return ge::GRAPH_FAILED);
        }
    }

    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    const int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    int64_t minAssistInfoDim0 = static_cast<int64_t>(A * ASSIST_NUM_PER_A);
    OP_TILING_CHECK(assistInfoDim0 < minAssistInfoDim0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "assist_info_dim0", std::to_string(assistInfoDim0).c_str(),
                        ("assist_info_dim0 should be greater than or equal to min_assist_info_dim0=" +
                         std::to_string(minAssistInfoDim0))
                            .c_str()),
                    return ge::GRAPH_FAILED);

    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);

    if (hasElasticInfo) {
        OP_TILING_CHECK(expertTokenNumsDim0 != (localMoeExpertNum > 1 ? localMoeExpertNum : 1),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "expert_token_nums_dim0", std::to_string(expertTokenNumsDim0).c_str(),
                            ("elastic scaling expert_token_nums_dim0 should equal to max(local_moe_expert_num,1)=" +
                             std::to_string(localMoeExpertNum > 1 ? localMoeExpertNum : 1))
                                .c_str()),
                        return ge::GRAPH_FAILED);
    } else if (isSharedExpert) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "expert_token_nums_dim0",
                                                              std::to_string(expertTokenNumsDim0).c_str(),
                                                              "shared expert_token_nums_dim0 should equal to 1"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            expertTokenNumsDim0 != localMoeExpertNum,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                nodeName, "expert_token_nums_dim0", std::to_string(expertTokenNumsDim0).c_str(),
                ("moe expert_token_nums_dim0 should equal to local_moe_expert_num=" + std::to_string(localMoeExpertNum))
                    .c_str()),
            return ge::GRAPH_FAILED);
    }

    // 校验epRecvCount和tpRecvCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpRecvCountDim0 = tpRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = ((isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum);
    if (hasElasticInfo) {
        epRecvCount = std::max(epWorldSize, epWorldSize * localMoeExpertNum);
    }
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        epRecvCount *= tpWorldSize;
    }
    if (isLayered) {
        // 如果是分层方案，则需要校验全新的shape，额外的globalBs * 2 * k * epWorldSize / 8
        // 用来存储token的cnt信息与offset信息，为了兼容A2&A5 这里取/8。
        epRecvCount =
            epWorldSize * localMoeExpertNum + globalBs * 2 * expertIdsDim1 * epWorldSize / RANK_NUM_PER_NODE_A2;
        OP_TILING_CHECK(epRecvCountDim0 < epRecvCount,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "ep_recv_count_dim0", std::to_string(epRecvCountDim0).c_str(),
                            ("ep_recv_count_dim0 should be greater than or equal to " + std::to_string(epRecvCount) +
                             " (ep_world_size * local_moe_expert_num + global_bs * 2 * k * ep_world_size / 8)")
                                .c_str()),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(epRecvCountDim0 < epRecvCount,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep_recv_count_dim0",
                                                              std::to_string(epRecvCountDim0).c_str(),
                                                              ("ep_recv_count_dim0 should be greater than or equal to "
                                                               "ep_world_size * local_moe_expert_num * tp_world_size=" +
                                                               std::to_string(epRecvCount))
                                                                  .c_str()),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(tpRecvCountDim0 != tpWorldSize,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "tp_recv_count_dim0", std::to_string(tpRecvCountDim0).c_str(),
                        ("tp_recv_count_dim0 should equal to tp_world_size=" + std::to_string(tpWorldSize)).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验输入输出数据格式
static bool CheckTensorPtrNullptr(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                  const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                                  const bool isPerformance, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return false);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_ids"), return false);
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scales"), return false);
    }
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x_active_mask"),
                        return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "elastic_info"), return false);
    }
    if (isPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        OP_TILING_CHECK(performanceInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "performance_info"),
                        return false);
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expand_x"), return false);
    if (quantMode >= static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamic_scales"),
                        return false);
    }
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "assist_info_for_combine"),
                    return false);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_token_nums"),
                    return false);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "ep_recv_count"), return false);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tp_recv_count"), return false);

    return true;
}

static ge::graphStatus CheckMc2Context(gert::TilingContext *context, const char *nodeName, DispatchV2Config &config)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "context",
                                     std::to_string(contextStorageShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
        return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    auto contextDesc = context->GetInputDesc(config.contextIndex);
    OP_TILING_CHECK(contextDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "context"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "context",
                                              std::to_string(static_cast<int64_t>(contextDesc->GetDataType())).c_str(),
                                              "INT32"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "context",
                                   std::to_string(static_cast<int64_t>(contextDesc->GetStorageFormat())).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::TilingCheckMoeDistributeDispatch(
    gert::TilingContext *context, const char *nodeName, const bool isActiveMask, const bool isScales,
    const bool hasElasticInfo, const bool isPerformance, const uint32_t quantMode, const bool isLayered,
    DispatchV2Config &config)
{
    OP_TILING_CHECK(
        !CheckTensorPtrNullptr(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo, isPerformance,
                               config),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "tensor_params", "invalid", "params check null failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !CheckTensorDim(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo, isPerformance, isLayered,
                        config),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "tensor_params", "invalid", "params shape is invalid"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !CheckTensorDataType(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo, isPerformance,
                             config),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "tensor_params", "invalid", "params dataType is invalid"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !CheckTensorFormat(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo, isPerformance, config),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "tensor_params", "invalid", "params format is invalid"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData *tiling,
                                   const std::string groupEp, const std::string groupTp, const uint32_t tpWorldSize,
                                   bool isLayered)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 groupEp = %s", groupEp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_ALL_GATHER;
    std::string algConfigAllToAllStr =
        isLayered ? "AlltoAll=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigAllGatherStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE); // 通过不拉起AICPU，提高算子退出性能
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "mc2_init_tiling", "failed",
                                                          "mc2CcTilingConfig mc2InitTiling GetTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "mc2_cc_tiling1", "failed",
                                                          "mc2CcTilingConfig mc2CcTiling1 GetTiling failed"),
                    return ge::GRAPH_FAILED);

    if (tpWorldSize > 1) {
        OP_LOGD(nodeName, "MoeDistributeDispatchV2 groupTp = %s", groupTp.c_str());
        mc2CcTilingConfig.SetGroupName(groupTp);
        mc2CcTilingConfig.SetOpType(opType2);
        mc2CcTilingConfig.SetAlgConfig(algConfigAllGatherStr);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2) != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "mc2_cc_tiling2", "failed",
                                                              "mc2CcTilingConfig mc2CcTiling2 GetTiling failed"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndCalWinSize(const gert::TilingContext *context,
                                          MoeDistributeDispatchV2TilingData &tilingData, const char *nodeName,
                                          const bool isSetFullMeshV2, uint32_t localMoeExpertNum, bool isLayered,
                                          DispatchV2Config &config)
{
    CheckWinSizeData winSizeData;
    winSizeData.localMoeExpertNum = localMoeExpertNum;
    winSizeData.sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    winSizeData.a = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.a);
    winSizeData.h = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h);
    winSizeData.k = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.k);
    winSizeData.bs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.bs);
    winSizeData.epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    winSizeData.globalBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.globalBs);
    winSizeData.moeExpertNum = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    winSizeData.tpWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    winSizeData.isSetFullMeshV2 = isSetFullMeshV2;
    winSizeData.isLayered = isLayered;
    winSizeData.isMc2Context = config.isMc2Context;
    OP_TILING_CHECK(CheckWinSize(context, nodeName, winSizeData) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "win_size", "invalid", "Get WinSize failed"),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.totalWinSizeEp = winSizeData.totalWinSizeEp;
    if (winSizeData.tpWorldSize == TP_WORLD_SIZE_TWO) {
        tilingData.moeDistributeDispatchV2Info.totalWinSizeTp = winSizeData.totalWinSizeTp;
    }
    return ge::GRAPH_SUCCESS;
}

static uint32_t Ceil(uint32_t dividend, uint32_t divisor, const char *nodeName)
{
    if (divisor != 0) {
        return ((dividend + divisor - 1) / divisor) * divisor;
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "divisor", "0", "ceil divisor should not be 0");
    return 0;
}

static uint32_t SendToMoeExpertUsedBuffer(const gert::TilingContext *context,
                                          const MoeDistributeDispatchV2TilingData &tilingData,
                                          uint32_t expertIdsBufSize, uint32_t kSize, const char *nodeName)
{
    uint32_t UbForMoe = 0;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t totalExpertNum = sharedExpertRankNum + moeExpertNum;
    uint32_t aivUsedCumSum = totalExpertNum / 32; // 单核处理32个专家cnt发送
    aivUsedCumSum = (aivUsedCumSum == 0) ? 1 : aivUsedCumSum;
    aivUsedCumSum = (aivUsedCumSum >= (aivNum / HALF_NUM)) ? (aivNum / HALF_NUM) : aivUsedCumSum;
    uint32_t aivUsedAllToAll = aivNum - aivUsedCumSum;
    uint32_t sharedUsedAivNum = 0;
    if (sharedExpertRankNum != 0U) {
        sharedUsedAivNum = (aivUsedAllToAll * sharedExpertNum) / (kSize + sharedExpertNum);
        if (sharedUsedAivNum == 0) {
            sharedUsedAivNum = 1;
        }
    }
    uint32_t moeUsedAivNum = aivUsedAllToAll - sharedUsedAivNum;
    uint32_t maskSizePerExpert = Ceil((expertIdsBufSize / sizeof(int32_t)) / 8, UB_ALIGN, nodeName); // 8 is 1byte->8bit
    uint32_t expertMaskBufSize = maskSizePerExpert * Ceil(moeExpertNum, moeUsedAivNum, nodeName) / moeUsedAivNum;
    UbForMoe = UbForMoe + expertMaskBufSize;                                        // expertMaskBuf
    UbForMoe = UbForMoe + Ceil(moeExpertNum * sizeof(int32_t), UB_ALIGN, nodeName); // tokenNumToExpertBuf
    return UbForMoe;
}

static uint32_t AllToAllBasicUsedBuffer(const gert::TilingContext *context,
                                        const MoeDistributeDispatchV2TilingData &tilingData, uint32_t quantMode,
                                        uint32_t &expertIdsBufSize, uint32_t hSize, uint32_t expertIdsCnt,
                                        uint32_t &hOutSizeAlign, const char *nodeName)
{
    uint32_t UbForBasic = 0;
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    if (tilingData.moeDistributeDispatchV2Info.hasElasticInfo) {
        uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSize) * sizeof(int32_t);
        uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN, nodeName);
        UbForBasic = UbForBasic + elasticInfoSizeAlign; // elasticInfoBuf_
    }

    uint32_t sizeofX = 2U;
    uint32_t hAlignSize = Ceil(hSize * sizeofX, UB_ALIGN, nodeName);
    UbForBasic = UbForBasic + hAlignSize * BUFFER_NUM; // inQueue

    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    uint32_t hOutSize = (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) ? hSize * SIZE_OF_UNQUANT : hSize;
    hOutSizeAlign = Ceil(hOutSize, UB_ALIGN, nodeName);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) && isScales) {
        uint32_t scaleInBytes =
            tilingData.moeDistributeDispatchV2Info.scalesCol * tilingData.moeDistributeDispatchV2Info.scalesTypeSize;
        hOutSizeAlign += scaleInBytes;
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        hOutSizeAlign += sizeof(float);
    }
    uint32_t hScaleSizeAlign = Ceil(hOutSizeAlign, UB_ALIGN, nodeName);
    uint32_t tokenQuantAlign = hScaleSizeAlign / sizeof(int32_t);
    hOutSizeAlign = tokenQuantAlign * sizeof(int32_t) + UB_ALIGN;
    uint32_t blockCntPerToken = Ceil(hOutSizeAlign, SPLIT_BLOCK_DATA_SIZE, nodeName) / SPLIT_BLOCK_DATA_SIZE;
    uint32_t hCommuSize = blockCntPerToken * SPLIT_BLOCK_SIZE;
    UbForBasic = UbForBasic + hCommuSize * BUFFER_NUM; // outBuf

    expertIdsBufSize = Ceil(expertIdsCnt * sizeof(int32_t), SIZE_ALIGN_256, nodeName);
    UbForBasic = UbForBasic + expertIdsBufSize; // expertIdsBuf_
    return UbForBasic;
}

static ge::graphStatus CheckUBSize(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
                                   const char *nodeName)
{
    uint32_t ubSize = UB_ALIGN; // dataStateBuf

    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint32_t hSize = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bsSize = tilingData.moeDistributeDispatchV2Info.bs;
    uint32_t kSize = tilingData.moeDistributeDispatchV2Info.k;
    uint32_t expertIdsCnt = bsSize * kSize;
    uint32_t expertIdsBufSize = 0;
    uint32_t hOutSizeAlign = 0;
    ubSize = ubSize + AllToAllBasicUsedBuffer(context, tilingData, quantMode, expertIdsBufSize, hSize, expertIdsCnt,
                                              hOutSizeAlign, nodeName);

    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    bool isScales = (scalesStorageShape != nullptr);
    bool needMaskCalFlag = (isActiveMask || tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum != 0);
    if (needMaskCalFlag) {
        ubSize = ubSize + expertIdsBufSize; // gatherMaskTBuf_
    }

    uint32_t hFp32Size = Ceil(hSize * sizeof(float), UB_ALIGN, nodeName);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)) && isScales) {
        hFp32Size = Ceil(hSize, PERGROUP_BLOCK_SIZE, nodeName) * sizeof(float);
    }
    uint32_t bsKAlign256 = Ceil(expertIdsCnt * SIZE_OF_HALF, SIZE_ALIGN_256, nodeName);
    uint32_t expertIdsSize = Ceil(expertIdsCnt * sizeof(int32_t), UB_ALIGN, nodeName);
    uint32_t xActivateMaskSize = bsSize * Ceil(kSize * sizeof(bool), UB_ALIGN, nodeName) * SIZE_OF_HALF;
    uint32_t maxSizeForUbBuffer = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSizeForUbBuffer = maxSizeForUbBuffer > xActivateMaskSize ? maxSizeForUbBuffer : xActivateMaskSize;
    maxSizeForUbBuffer = maxSizeForUbBuffer > bsKAlign256 ? maxSizeForUbBuffer : bsKAlign256;
    tilingData.moeDistributeDispatchV2Info.maxSizeForUbBuffer = maxSizeForUbBuffer;
    if (quantMode > static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        uint32_t hOutAlignUbSize = Ceil(hOutSizeAlign, UB_ALIGN, nodeName);
        ubSize = ubSize + hOutAlignUbSize;
        ubSize = ubSize + BUFFER_NUM * maxSizeForUbBuffer; // receiveDataCastFloatBuf smoothScalesBuf
    } else if (needMaskCalFlag) {
        ubSize = ubSize + BUFFER_NUM * maxSizeForUbBuffer;
    }

    if (tilingData.moeDistributeDispatchV2Info.isExpertMask ||
        (tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum != 0)) {
        uint32_t axisBSAlign = Ceil(bsSize * sizeof(int32_t), UB_ALIGN, nodeName);
        ubSize = ubSize + axisBSAlign; // validBsIndexTBuf_
        uint32_t validBufferSize = expertIdsSize > xActivateMaskSize ? expertIdsSize : xActivateMaskSize;
        ubSize = ubSize + validBufferSize; // validExpertIndexBuf_
    }

    ubSize = ubSize + SendToMoeExpertUsedBuffer(context, tilingData, expertIdsBufSize, kSize, nodeName);
    uint64_t ubRealSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubRealSize);
    OP_TILING_CHECK(ubSize > ubRealSize,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "ub_size", std::to_string(ubSize).c_str(),
                        ("Current scenario exceeds UB size limit, BS=" + std::to_string(bsSize) +
                         ", H=" + std::to_string(hSize) + ", topK=" + std::to_string(kSize) +
                         ", moe_expert_num=" + std::to_string(tilingData.moeDistributeDispatchV2Info.moeExpertNum) +
                         ", aiv_num=" + std::to_string(ascendcPlatform.GetCoreNumAiv()))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspaces"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
MoeDistributeDispatchV2TilingFuncBase::MoeDistributeDispatchA3TilingFuncImplPublic(gert::TilingContext *context,
                                                                                   DispatchV2Config &config)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tiling_data"),
                    return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t quantMode = static_cast<uint32_t>(QuantModeA5::NON_QUANT);
    bool isScales = false;
    bool isActiveMask = false;
    bool hasElasticInfo = false;
    bool isPerformance = false;
    bool isSetFullMeshV2 = false;
    bool isLayered = false;
    uint32_t localMoeExpertNum = 1;
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchV2 tiling check func.");

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp, groupTp, isSetFullMeshV2,
                                            isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "attr_and_tiling", "invalid",
                                                           "Get attr and set tiling data failed"),
                    return ge::GRAPH_FAILED);

    // 获取scales
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
    isScales = (scalesStorageShape != nullptr);

    // 获取xActiveMask
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isTokenMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeDispatchV2Info.isExpertMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    // 获取elasticInfo
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    hasElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.hasElasticInfo = hasElasticInfo;

    // 获取performanceInfo
    const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(config.performanceInfoIndex);
    isPerformance = (performanceInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isPerformance = isPerformance;

    quantMode = tilingData->moeDistributeDispatchV2Info.quantMode;

    // 检查quantMode和scales是否匹配
    if (CheckQuantModeMatchScales(context, nodeName, isScales, quantMode, config) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    // 检查context输入
    if (config.isMc2Context) {
        OP_TILING_CHECK(
            CheckMc2Context(context, nodeName, config) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "context", "invalid", "Tiling check context failed"),
            return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(TilingCheckMoeDistributeDispatch(context, nodeName, isActiveMask, isScales, hasElasticInfo,
                                                     isPerformance, quantMode, isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "params", "invalid", "Tiling check param failed"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrs(context, nodeName, *tilingData, localMoeExpertNum, isActiveMask, isSetFullMeshV2,
                               isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "attrs", "invalid", "Check attr failed"),
                    return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeDistributeDispatchV2Info.epRankId;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    bool isSharedExpert = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(
        CheckTensorShape(context, nodeName, *tilingData, quantMode, isScales, isSharedExpert, hasElasticInfo,
                         isPerformance, static_cast<int64_t>(localMoeExpertNum), isLayered,
                         config) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "tensor_shape", "invalid", "Check tensor shape failed"),
        return ge::GRAPH_FAILED);

    if (isSetFullMeshV2) {
        OP_TILING_CHECK(
            CheckUBSize(context, *tilingData, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "ub_size", "invalid", "Tiling check UB size failed"),
            return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(
        CheckAndCalWinSize(context, *tilingData, nodeName, isSetFullMeshV2, localMoeExpertNum, isLayered, config) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "win_size", "invalid", "Tiling check window size failed"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "workspace", "invalid", "Tiling set workspace failed"),
        return ge::GRAPH_FAILED);
    uint32_t tpWorldSize = tilingData->moeDistributeDispatchV2Info.tpWorldSize;
    if (!config.isMc2Context) {
        OP_TILING_CHECK(
            SetHcommCfg(context, tilingData, groupEp, groupTp, tpWorldSize, isLayered) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "hcomm_cfg", "invalid", "Tiling set hcomm cfg failed"),
            return ge::GRAPH_FAILED);
    }
    uint64_t tilingKey = CalTilingKey(isScales, quantMode, tpWorldSize, isSetFullMeshV2, isLayered);
    tilingData->moeDistributeDispatchV2Info.isMc2Context = config.isMc2Context;

    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t epWorldSize = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    uint32_t serverNum = epWorldSize / RANK_NUM_PER_NODE;
    OP_TILING_CHECK((isLayered && ((aivNum < RANK_NUM_PER_NODE) || (aivNum <= 2 * serverNum))),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "aiv_num", std::to_string(aivNum).c_str(),
                                                          ("Layered should use aiv_num greater than " +
                                                           std::to_string(RANK_NUM_PER_NODE) +
                                                           " and 2 * serverNum=" + std::to_string(2 * serverNum))
                                                              .c_str()),
                    return ge::GRAPH_FAILED);
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    context->SetScheduleMode(1); // 设置为batch mode模式, 所有核同时启动
    tilingData->moeDistributeDispatchV2Info.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchV2Info.aivNum = aivNum;
    OP_LOGD(nodeName, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
// Register exception func
inline void MoeDistributeDispatchV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char *socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        return;
    }
    Mc2Exception::Mc2ExceptionImpl(args, userdata, "MoeDistributeDispatchV2");
}

__attribute__((constructor)) void RegisterMoeDistributeDispatchV2ExceptionFunc()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;

    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MoeDistributeDispatchV2", "Get runtime version failed when register exception func.");
        return;
    }
    if (aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MoeDistributeDispatchV2", "Get metadef version failed when register exception func.");
        return;
    }

    if (runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MoeDistributeDispatchV2",
                "The runtime(%d) or metadata(%d) version is lower than the version(%d) supporting exception func.",
                runtimeVersionNum, metadefVersionNum, EXCEPTION_DUMP_SUPPORT_VERSION);
        return;
    }

    IMPL_OP(MoeDistributeDispatchV2).ExceptionDumpParseFunc(MoeDistributeDispatchV2ExceptionImplWrapper);
}
#endif

} // namespace optiling