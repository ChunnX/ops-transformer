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
 * \file dispatch_v2_tiling.cpp
 * \brief MOE Distribute Dispatch V2 A3/A5 MTE 主流程
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
// 公共层组件
#include "moe_dispatch/defs.h"
#include "moe_dispatch/utils.h"
#include "moe_dispatch/quant_checker_factory.h"
#include "moe_dispatch/elastic_info_checker.h"
#include "moe_dispatch/active_mask_checker.h"
#include "moe_dispatch/performance_info_checker.h"
#include "moe_dispatch/fullmesh_checker.h"
#include "moe_dispatch/comm_cfg.h"
// Layout + TilingCtx
#include "dispatch_v2_layout.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
using namespace moe_dispatch_defs;  // 公共层常量命名空间
namespace {
    // ========== Tensor 索引常量（算子原型相关）==========
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

    // ========== 其他算子内部常量 ==========
    // 注：以下常量已使用公共层 defs.h 定义：
    //     - ALIGN_256, MAX_LOCAL_EXPERT_NUM, ASSIST_INFO_NUM_PER_A, ELASTIC_INFO_OFFSET
    //     - MAX_MOE_EXPERT_NUM_V2, MAX_H_DEFAULT, MAX_BS_DEFAULT 等公共常量
    constexpr uint32_t HALF_NUM = 2;                      // cumsum最多只能用一半的核
    constexpr uint32_t H_BASIC_BLOCK_LAYERED = 32;
    constexpr uint32_t RANK_NUM_PER_NODE = 16U;
    constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;          // A2 兼容用
    constexpr uint32_t SIZE_OF_UNQUANT = 2;
    constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
    constexpr uint32_t SPLIT_BLOCK_SIZE = 512UL;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint64_t TRIPLE = 3;
    constexpr int64_t CEIL_ALIGN32 = 8;
    constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512;
    constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;
    constexpr uint64_t STATIC_SCALE_DIM_0 = 1;
    constexpr uint64_t ONE_DIM_SCALE_COL_NUM = 1;
}

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

// 使用公共层 QuantChecker 校验 dynamicScales 维度
static bool CheckDynamicScalesDim(const gert::TilingContext *context,
    const char *nodeName, const uint32_t quantMode)
{
    const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesStorageShape == nullptr,
        OP_LOGE(nodeName, "dynamicScalesShape is null."), return false);

    // 创建量化校验器获取动态缩放维度数
    auto checker = moe_dispatch::CreateQuantChecker(quantMode, ge::DT_FLOAT8_E8M0);
    OP_TILING_CHECK(checker == nullptr,
        OP_LOGE(nodeName, "Invalid quantMode: %u.", quantMode), return false);

    uint32_t expectedDimNum = checker->GetDynamicScalesDimNum();
    uint32_t actualDimNum = static_cast<uint32_t>(dynamicScalesStorageShape->GetStorageShape().GetDimNum());
    OP_TILING_CHECK(actualDimNum != expectedDimNum,
        OP_LOGE(nodeName, "dynamicScalesShape dims must be %u when quantMode=%u, but current dim num is %u.",
        expectedDimNum, quantMode, actualDimNum), return false);

    OP_LOGD(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    if (expectedDimNum == moe_dispatch_defs::DYNAMIC_SCALE_TWO_DIM_NUM) {
        OP_LOGD(nodeName, "dynamicScales dim1 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(1));
    }
    return true;
}

// 使用公共层 QuantChecker 校验 scales 维度
static bool CheckScaleTensorDim(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, DispatchV2Config &config)
{
    if (!isScales) {
        return true;
    }

    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
    OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."), return false);

    // 创建量化校验器
    auto checker = moe_dispatch::CreateQuantChecker(quantMode, ge::DT_FLOAT8_E8M0);
    OP_TILING_CHECK(checker == nullptr,
        OP_LOGE(nodeName, "Invalid quantMode: %u.", quantMode), return false);

    // 获取 expandX dtype 用于静态量化 hif8 特殊校验
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    ge::DataType expandXDtype = expandXDesc->GetDataType();

    uint32_t actualDimNum = static_cast<uint32_t>(scalesStorageShape->GetStorageShape().GetDimNum());
    uint32_t expectedDimNum = checker->GetExpectedScalesDimNum(expandXDtype);

    if (checker->AllowScalesMultiDim()) {
        // 静态量化：允许 1D 或 2D，但 hif8 时强制 1D
        if (expectedDimNum == moe_dispatch_defs::ONE_DIM) {
            // hif8 强制 1D
            OP_TILING_CHECK(actualDimNum != moe_dispatch_defs::ONE_DIM,
                OP_LOGE(nodeName, "scalesShape dims must be 1 when x dtype is hif8 in static quant, but current dim num is %u.",
                actualDimNum), return false);
        } else {
            // 允许 1D 或 2D
            OP_TILING_CHECK(actualDimNum != moe_dispatch_defs::ONE_DIM && actualDimNum != moe_dispatch_defs::TWO_DIMS,
                OP_LOGE(nodeName, "scalesShape dims must be 1 or 2 when quantMode is %u, but current dim num is %u.",
                quantMode, actualDimNum), return false);
        }
    } else {
        // 其他模式：固定维度数
        OP_TILING_CHECK(actualDimNum != expectedDimNum,
            OP_LOGE(nodeName, "scales dims must be %u when quantMode=%u, but current dim num is %u.",
            expectedDimNum, quantMode, actualDimNum), return false);
    }

    OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
    if (actualDimNum == moe_dispatch_defs::TWO_DIMS) {
        OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
    }
    return true;
}

//x, expertIds, scales维度校验
static bool CheckInputTensorDim(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const bool isLayered, DispatchV2Config &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %lu.",
        xStorageShape->GetStorageShape().GetDimNum()), return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    if (isLayered) {
        const gert::StorageShape *expertScaleStorageShape = context->GetOptionalInputShape(config.expertScalesIndex);
        OP_TILING_CHECK(expertScaleStorageShape == nullptr, OP_LOGE(nodeName, "expertScaleShape is null."), return false);
        OP_TILING_CHECK(expertScaleStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertScaleShape dims must be 2, but current dim num is %lu.",
        expertScaleStorageShape->GetStorageShape().GetDimNum()), return false);
        const int64_t expertScalesDim0 = expertScaleStorageShape->GetStorageShape().GetDim(0);
        const int64_t expertScalesDim1 = expertScaleStorageShape->GetStorageShape().GetDim(1);
        OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesDim0);
        OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesDim1);
    }
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(nodeName, "expertIdShape is null."), return false);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertIdShape dims must be 2, but current dim num is %lu.",
        expertIdStorageShape->GetStorageShape().GetDimNum()), return false);
    const int64_t expertIdDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdDim0);
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdDim1);
    OP_TILING_CHECK(!CheckScaleTensorDim(context, nodeName, isScales, quantMode, config),
        OP_LOGE(nodeName, "isScale Input param shape is invalid."), return false);
    return true;
}

//expertX, assistInfo, expertTokenNums, epRecvCount, tpRecvCount维度校验
static bool CheckCommonOutputTensorDim(const gert::TilingContext *context, const char *nodeName,
    const uint32_t quantMode)
{
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandXShape is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expandXShape dims must be 2, but current dim num is %lu.",
        expandXStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    if ((quantMode != static_cast<uint32_t>(moe_dispatch_defs::QuantMode::UNQUANT)) && (quantMode != static_cast<uint32_t>(moe_dispatch_defs::QuantMode::STATIC_QUANT))) {
        OP_TILING_CHECK(!CheckDynamicScalesDim(context, nodeName, quantMode),OP_LOGE(nodeName, "CheckDynamicScalesDim failed."), return false);
    }

    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE(nodeName, "assistInfoShape is null."), return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "assistInfoShape dims must be 1, but current dim num is %lu.",
        assistInfoStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr,
        OP_LOGE(nodeName, "expertTokenNumsShape is null."), return false);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "expertTokenNumsShape dims must be 1, but current dim num is %lu.",
        expertTokenNumsStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "epRecvCountShape is null."), return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "epRecvCountShape dims must be 1, but current dim num is %lu.",
        epRecvCountStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr,
        OP_LOGE(nodeName, "tpRecvCountShape is null."), return false);
    OP_TILING_CHECK(tpRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "tpRecvCountShape dims must be 1, but current dim num is %lu.",
        tpRecvCountStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "tpRecvCount dim0 = %ld", tpRecvCountStorageShape->GetStorageShape().GetDim(0));

    return true;
}

// 使用 TilingCtx 校验 tensor 维度
static bool CheckTensorDim(const gert::TilingContext *context, const char *nodeName,
    const DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    OP_TILING_CHECK(!CheckInputTensorDim(context, nodeName, tc.hasScales, tc.quantMode, tc.isLayered, config),
        OP_LOGE(nodeName, "Input param shape is invalid."), return false);

    // 使用公共层 active_mask_check 校验
    if (tc.hasActiveMask) {
        const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
        uint32_t bs = static_cast<uint32_t>(xStorageShape->GetStorageShape().GetDim(0));
        const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
        uint32_t k = static_cast<uint32_t>(expertIdStorageShape->GetStorageShape().GetDim(1));

        ge::graphStatus ret = active_mask_check::CheckShape(
            const_cast<gert::TilingContext*>(context), config.xActiveMaskIndex, bs, k, nodeName);
        OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "activeMask shape check failed."), return false);
    }

    // 使用公共层 elastic_info_check 校验（注意：公共层包含精确值校验，需要 epWorldSize）
    // 主流程中 epWorldSize 在后续才能获取，这里只做基础维度校验
    if (tc.hasElastic) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoStorageShape == nullptr, OP_LOGE(nodeName, "elasticInfo is null."), return false);
        OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE(nodeName, "elasticInfo dim must be 1, but current dim num is %lu.",
            elasticInfoStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "elasticInfo dim0 = %ld", elasticInfoStorageShape->GetStorageShape().GetDim(0));
    }

    // 使用公共层 performance_info_check 校验
    if (tc.hasPerformance) {
        const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(config.performanceInfoIndex);
        OP_TILING_CHECK(performanceInfoStorageShape == nullptr, OP_LOGE(nodeName, "performanceInfo is null."), return false);
        OP_TILING_CHECK(performanceInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE(nodeName, "performanceInfo dim must be 1, but current dim num is %lu.",
            performanceInfoStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "performanceInfo dim0 = %ld", performanceInfoStorageShape->GetStorageShape().GetDim(0));
    }

    OP_TILING_CHECK(!CheckCommonOutputTensorDim(context, nodeName, tc.quantMode),
        OP_LOGE(nodeName, "Output param shape is invalid."), return false);

    return true;
}

// 使用公共层 QuantChecker 校验 quantMode 和 scales 的匹配关系
static ge::graphStatus CheckQuantModeAndScales(const gert::TilingContext *context, const char *nodeName,
    bool isScales, const uint32_t quantMode, DispatchV2Config &config)
{
    // 创建量化校验器（A5 使用 e8m0 作为 MX 的 dynamicScales dtype）
    auto checker = moe_dispatch::CreateQuantChecker(quantMode, ge::DT_FLOAT8_E8M0);
    OP_TILING_CHECK(checker == nullptr,
        OP_LOGE(nodeName, "Invalid quantMode: %u.", quantMode), return ge::GRAPH_FAILED);

    // 获取 x 的 dtype 用于 ShouldHaveScales 判断
    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return ge::GRAPH_FAILED);
    ge::DataType xDtype = xDesc->GetDataType();

    // 检查 scales 是否必须存在
    bool shouldHaveScales = checker->ShouldHaveScales(xDtype);
    if (shouldHaveScales && !isScales) {
        OP_LOGE(nodeName, "The scales should not be nullptr when quantMode is %u and X datatype is %s.",
            quantMode, Ops::Base::ToString(xDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    // 非量化模式下 scales 必须为空
    if (!shouldHaveScales && isScales) {
        OP_LOGE(nodeName, "The scales should be nullptr when quantMode is %u.", quantMode);
        return ge::GRAPH_FAILED;
    }

    // 如果 scales 存在，使用公共层 CheckScales 校验
    if (isScales) {
        auto scalesShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_TILING_CHECK(scalesShape == nullptr,
            OP_LOGE(nodeName, "scalesShape is null."), return ge::GRAPH_FAILED);
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);

        // 构造 ScalesTensorInfo
        moe_dispatch::ScalesTensorInfo info;
        info.dtype = scalesDesc != nullptr ? scalesDesc->GetDataType() : ge::DT_UNDEFINED;
        info.dimNum = static_cast<uint32_t>(scalesShape->GetStorageShape().GetDimNum());
        info.dim0 = scalesShape->GetStorageShape().GetDim(0);
        info.dim1 = info.dimNum > 1U ? scalesShape->GetStorageShape().GetDim(1) : 0UL;

        ge::graphStatus status = checker->CheckScales(
            const_cast<gert::TilingContext*>(context), info, 0, 0);
        OP_TILING_CHECK(status != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Scales tensor validation failed for quantMode %u.", quantMode),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeDispatchV2TilingFuncBase::CheckCommomOutputTensorDataType(
    const gert::TilingContext *context, const char *nodeName)
{
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(assistInfoDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "assistInfoForCombine dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(assistInfoDesc->GetDataType()).c_str()), return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."),
        return false);
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT64,
        OP_LOGE(nodeName, "expertTokenNums dataType is invalid, dataType should be int64, but is %s.",
        Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str()), return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(epRecvCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "epRecvCounts dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(epRecvCountsDesc->GetDataType()).c_str()), return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(tpRecvCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "tpRecvCounts dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(tpRecvCountsDesc->GetDataType()).c_str()), return false);

    return true;
}

bool MoeDistributeDispatchV2TilingFuncBase::CheckCommomOtherInputTensorDataType(
    const gert::TilingContext *context, const char *nodeName,
    const bool isActiveMask, const bool hasElasticInfo, const bool isPerformance, DispatchV2Config &config)
{
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "expertId dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(expertIdDesc->GetDataType()).c_str()), return false);

    if (isPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        OP_TILING_CHECK(performanceInfoDesc->GetDataType() != ge::DT_INT64, OP_LOGE(nodeName,
            "performanceInfoDesc dataType is invalid, dataType should be int64, but is %s.",
            Ops::Base::ToString(performanceInfoDesc->GetDataType()).c_str()), return false);
    }
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL, OP_LOGE(nodeName,
            "xActiveMask dataType is invalid, dataType should be bool, but is %s.",
            Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str()), return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName,
            "elasticInfoDesc dataType is invalid, dataType should be int32, but is %s.",
            Ops::Base::ToString(elasticInfoDesc->GetDataType()).c_str()), return false);
    }

    return true;
}

// 使用 TilingCtx 校验输入输出数据格式
static bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName,
    const DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex); // nullptr前面已check过
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "x format is invalid."), return false);
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertId format is invalid."), return false);
    if (tc.hasScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "scales format is invalid."), return false);
    }
    if (tc.hasActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "xActiveMask format is invalid."), return false);
    }
    if (tc.hasElastic) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "elasticInfo format is invalid."), return false);
    }
    if (tc.hasPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(performanceInfoDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "performanceInfoDesc format is invalid."), return false);
    }
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expandX format is invalid."), return false);
    if (tc.quantMode >= static_cast<uint32_t>(moe_dispatch_defs::QuantMode::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "dynamicScales format is invalid."), return false);
    }
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "assistInfoForCombine format is invalid."), return false);
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertTokenNums format is invalid."), return false);
    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "epRecvCounts format is invalid."), return false);
    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "tpRecvCounts format is invalid."), return false);
    return true;
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context, const char *nodeName,
    std::string &groupEp, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrGroupEpIndex));
        OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, moe_dispatch_defs::MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupEpPtr, moe_dispatch_defs::MAX_GROUP_NAME_LENGTH) == moe_dispatch_defs::MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "groupEpPtr is null."), return ge::GRAPH_FAILED);
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
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE(nodeName, "commAlgPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "expertShardPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertRankNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNumPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName, "quantModePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE(nodeName, "expertTokenNumsTypePtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(nodeName, "zeroExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(nodeName, "copyExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE(nodeName, "constExpertNumPtr is null."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 校验 group 属性参数
static ge::graphStatus CheckGroupAttrParams(const gert::TilingContext *context, const char *nodeName,
    std::string &groupTp, DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrTpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrTpRankIdIndex);
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int64_t>(config.attrCommAlgIndex));
    int64_t epWorldSize = *epWorldSizePtr;
    tc.isLayered = strcmp(commAlgPtr, "hierarchy") == 0; // isLayered赋值
    int64_t maxEpworldsize = tc.isLayered ? MAX_EP_WORLD_SIZE_LAYERED : MAX_EP_WORLD_SIZE;
    int64_t maxTpworldsize = tc.isLayered ? MAX_TP_WORLD_SIZE_LAYERED : MAX_TP_WORLD_SIZE;
    OP_TILING_CHECK((epWorldSize < moe_dispatch_defs::MIN_EP_WORLD_SIZE) || (epWorldSize > maxEpworldsize),
        OP_LOGE(nodeName, "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%ld.",
        moe_dispatch_defs::MIN_EP_WORLD_SIZE, maxEpworldsize, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tc.isLayered && (epWorldSize % RANK_NUM_PER_NODE != 0)),     // 校验epWorldSize是否是16整数倍
        OP_LOGE(nodeName, "epWorldSize should be %u Aligned, but got %ld.", RANK_NUM_PER_NODE, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > maxTpworldsize),
        OP_LOGE(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
        maxTpworldsize, *tpWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.",
        epWorldSize, *epRankIdPtr), return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int64_t>(config.attrGroupTpIndex));
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OP_LOGE(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
            *tpWorldSizePtr, *tpRankIdPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, moe_dispatch_defs::MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupTpPtr, moe_dispatch_defs::MAX_GROUP_NAME_LENGTH) == moe_dispatch_defs::MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "groupTpPtr is null."), return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0,
            OP_LOGE(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 校验其他属性参数
ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::CheckOtherAttrParams(
    const gert::TilingContext *context, const char *nodeName,
    DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    if (config.isMc2Context) {
        OP_TILING_CHECK((strcmp(commAlgPtr, "hierarchy") == 0),
            OP_LOGE(nodeName, "commAlgPtr %s doesn't support comm with context.", commAlgPtr),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(*expertShardPtr != 0,
        OP_LOGE(nodeName, "expertShardType is invalid, only support 0, but got expertShardType=%ld.",
        *expertShardPtr), return ge::GRAPH_FAILED);

    if (CheckQuantModePtr(quantModePtr, nodeName) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 0) && (*expertTokenNumsTypePtr != 1),
        OP_LOGE(nodeName, "expertTokenNumsType only support 0 or 1, but got expertTokenNumsType=%ld.",
        *expertTokenNumsTypePtr), return ge::GRAPH_FAILED);
    // A5 已作校验，这里只校验 A3
    if (CheckCommAlgPtr(commAlgPtr, nodeName) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    tc.isFullMeshV2 = ((strcmp(commAlgPtr, "fullmesh_v2") == 0) ? true : false);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 isSetFullMeshV2 = %d\n", tc.isFullMeshV2);

    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 校验通信属性参数
ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::CheckCommAttrParams(
    const gert::TilingContext *context, const char *nodeName,
    std::string &groupTp, DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    // 校验 epWorldSize tpWorldSize epRankId tpRankId
    OP_TILING_CHECK(CheckGroupAttrParams(context, nodeName, groupTp, tc, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckGroupAttrParams is failed."), return ge::GRAPH_FAILED);
    // 校验 expertSharedType quantMode expertTokenNumsType commAlg
    OP_TILING_CHECK(CheckOtherAttrParams(context, nodeName, tc, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckGroupAttrParams is failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 校验专家属性参数
static ge::graphStatus CheckExpertAttrParams(const gert::TilingContext *context, const char *nodeName,
    const DispatchV2TilingCtx &tc, DispatchV2Config &config)
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
    OP_TILING_CHECK((zeroExpertNum < 0), OP_LOGE(nodeName,
        "zeroExpertNum less than 0, zeroExpertNum is %ld.", zeroExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((copyExpertNum < 0), OP_LOGE(nodeName,
        "copyExpertNum less than 0, copyExpertNum is %ld.", copyExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((constExpertNum < 0), OP_LOGE(nodeName,
        "constExpertNum less than 0, constExpertNum is %ld.", constExpertNum), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "zeroExpertNum=%ld,copyExpertNum= %ld, constExpertNum=%ld", zeroExpertNum, copyExpertNum,
        constExpertNum);
    OP_TILING_CHECK(zeroComputeExpertNum + moeExpertNum > INT32_MAX, OP_LOGE(nodeName,
        "zeroExpertNum[%ld] + copyExpertNum[%ld] + constExpertNum[%ld] + moeExpertNum[%ld] exceed INT32_MAX.",
         zeroExpertNum, copyExpertNum, constExpertNum, moeExpertNum), return ge::GRAPH_FAILED);
    if (tc.isLayered) {
        OP_TILING_CHECK((sharedExpertRankNum != 0),
            OP_LOGE(nodeName, "sharedExpertNum is invalid in hierarchy mode, only support 0, but got sharedExpertNum=%ld, sharedExpertRankNum=%ld",
            *sharedExpertNumPtr, sharedExpertRankNum), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > moe_dispatch_defs::MAX_SHARED_EXPERT_NUM),
            OP_LOGE(nodeName, "sharedExpertNum is invalid, only support [0, %ld], but got sharedExpertNum=%ld.",
            moe_dispatch_defs::MAX_SHARED_EXPERT_NUM, *sharedExpertNumPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
            OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
            epWorldSize, sharedExpertRankNum), return ge::GRAPH_FAILED);
    }
    int64_t moeExpertMaxNum = tc.isLayered ? MAX_MOE_EXPERT_NUM_LAYERED : MAX_MOE_EXPERT_NUM_V2;
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > moeExpertMaxNum),
        OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
        moeExpertMaxNum, moeExpertNum), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, const char *nodeName,
    DispatchV2Config &config, MoeDistributeDispatchV2TilingData &tilingData)
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
    if ((tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum == 0U) && (tilingData.moeDistributeDispatchV2Info.sharedExpertNum == 1U)) {
        tilingData.moeDistributeDispatchV2Info.sharedExpertNum = 0U;
    }
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 zeroComputeExpertNum = %d\n",
        tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum);

    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 获取属性并设置 TilingData
ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::GetAttrAndSetTilingData(
    const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchV2TilingData &tilingData, std::string &groupEp, std::string &groupTp,
    DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    OP_TILING_CHECK(CheckAttrPtrNullptr(context, nodeName, groupEp, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckCommAttrParams(context, nodeName, groupTp, tc, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckExpertAttrParams(context, nodeName, tc, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetAttrParams(context, nodeName, config, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 校验通信算法属性
static bool CheckCommAlgAttrs(const gert::TilingContext *context, const char *nodeName,
    const MoeDistributeDispatchV2TilingData &tilingData, const DispatchV2TilingCtx &tc, DispatchV2Config &config)
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

    // 使用公共层 fullmesh_check 校验
    if (tc.isFullMeshV2) {
        ge::graphStatus ret = fullmesh_check::CheckConstraints(nodeName, bs, k, tpWorldSize);
        OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "fullmesh_v2 constraints check failed."), return false);
    }

    // 使用公共层 elastic_info_check 校验兼容性
    if (hasElasticInfo) {
        ge::graphStatus ret = elastic_info_check::CheckCompatibility(nodeName, tc.isLayered, tpWorldSize);
        OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Cannot support elasticInfo when comm_alg is hierarchy or tp>1"),
            return false);
    }

    // 校验特殊专家和分层不能同时启用
    OP_TILING_CHECK((tc.isLayered && (zeroComputeExpertNum > 0)),
        OP_LOGE(nodeName, "Cannot support zeroComputeExpert when comm_alg is hierarchy"),
        return false);

    // 使用公共层 active_mask_check 校验兼容性
    if (isExpertMask) {
        ge::graphStatus ret = active_mask_check::CheckCompatibility(nodeName, tc.isLayered, isExpertMask, bs);
        OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Cannot support 2D xActiveMask when comm_alg is hierarchy"),
            return false);
    }

    // 使用公共层 performance_info_check 校验兼容性
    if (isPerformance) {
        ge::graphStatus ret = performance_info_check::CheckCompatibility(nodeName, tc.isLayered);
        OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Cannot support isPerformance when comm_alg is hierarchy"),
            return false);
    }

    return true;
}

// 使用 TilingCtx 校验属性
static ge::graphStatus CheckAttrs(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchV2TilingData &tilingData, DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 使用公共层校验 shared expert 属性关系
    ge::graphStatus ret = moe_dispatch_utils::CheckSharedExpertRelation(
        sharedExpertNum, sharedExpertRankNum, epWorldSize, moeExpertNum);
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckSharedExpertRelation failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckCommAlgAttrs(context, nodeName, tilingData, tc, config),
        OP_LOGE(nodeName, "Check comm_alg related attributes failed."), return ge::GRAPH_FAILED);

    // moe专家数均分校验已在 CheckSharedExpertRelation 中完成
    tc.localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK((tc.localExpertNum <= 0) || (tc.localExpertNum * epWorldSize > MAX_LOCAL_EXPERT_NUM),OP_LOGE(nodeName, "localMoeExpertNum is invalid, "
        "localMoeExpertNum * epWorldSize must be less than or equal to 2048, and localMoeExpertNum must be greater than 0, "
        "but got localMoeExpertNum * epWorldSize = %u, localMoeExpertNum = %u", tc.localExpertNum * epWorldSize, tc.localExpertNum), return ge::GRAPH_FAILED);
    // 校验tp=2时单个moe卡上专家数是否等于1
    OP_TILING_CHECK((tpWorldSize > 1) && (tc.localExpertNum > 1), OP_LOGE(nodeName, "Cannot support multi-moeExpert %u "
        "in a rank when tpWorldSize = %u > 1", tc.localExpertNum, tpWorldSize), return ge::GRAPH_FAILED);
    // 注意：elasticInfo 与 tp>1 互斥已在 CheckCommAlgAttrs 中通过公共层 elastic_info_check 校验

    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t bsUpperBound = tc.isLayered ? MAX_BS_LAYERED : MAX_BS_DEFAULT;
    OP_TILING_CHECK((xDim0 > bsUpperBound) || (xDim0 <= 0), OP_LOGE(nodeName, "xDim0(BS) is invalid. Should be between "
        "[1, %ld], but got xDim0=%ld.", bsUpperBound, xDim0), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.bs = static_cast<uint32_t>(xDim0);
    tc.bs = static_cast<uint32_t>(xDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(config.attrGlobalBsIndex);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBsPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n",
        *globalBsPtr, xDim0, epWorldSize);
    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)), OP_LOGE(nodeName, "globalBS is invalid, only "
        "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, xDim0, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*globalBsPtr > (xDim0 * static_cast<int64_t>(epWorldSize))) && tc.hasActiveMask),
        OP_LOGE(nodeName, "Different bs on different rank cannot work when isActiveMask=true, globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, xDim0, epWorldSize), return ge::GRAPH_FAILED);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
        tc.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(*globalBsPtr);
        tc.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 校验二维 scales shape
static ge::graphStatus CheckTwoDimScalesShape(const gert::TilingContext *context, const char *nodeName,
    const MoeDistributeDispatchV2TilingData &tilingData, const DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);

    // 获取 scales 的维度
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
    OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."), return ge::GRAPH_FAILED);
    const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
    const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);

    if (sharedExpertRankNum != 0U) {
        OP_TILING_CHECK(scalesDim0 != (moeExpertNum + sharedExpertNum), OP_LOGE(nodeName,
            "scales's dim0 not equal to moeExpertNum + sharedExpertNum, scales's dim0=%ld, (moeExpertNum + sharedExpertNum)=%ld.",
            scalesDim0, moeExpertNum + sharedExpertNum), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(scalesDim0 != moeExpertNum, OP_LOGE(nodeName,
            "scales's dim0 not equal to moeExpertNum, scales's dim0=%ld, moeExpertNum=%ld.",
            scalesDim0, moeExpertNum), return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(xDim1 != scalesDim1, OP_LOGE(nodeName, "scales's dim1 not equal to xShape's dim1, "
        "xShape's dim1=%ld, scales's dim1=%ld.", xDim1, scalesDim1), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 校验 tensor shape
static ge::graphStatus CheckTensorShape(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchV2TilingData &tilingData, DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

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
    int64_t hMax = tc.isLayered ? MAX_H_LAYERED : MAX_H_DEFAULT;
    int64_t hMin = tc.isLayered ? 0 : H_MIN;
    OP_TILING_CHECK((tc.isLayered && (xDim1 % H_BASIC_BLOCK_LAYERED != 0)), OP_LOGE(nodeName,
        "xShape dims1(H) should be %u Aligned, but got %ld.", H_BASIC_BLOCK_LAYERED, xDim1), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((xDim1 < hMin) || (xDim1 > hMax), OP_LOGE(nodeName,
        "xShape dims1(H) should be in [%ld, %ld], but got %ld.", hMin, hMax, xDim1), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.h = static_cast<uint32_t>(xDim1);
    tc.h = static_cast<uint32_t>(xDim1);

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    if ((expandXDesc->GetDataType() == ge::DT_FLOAT4_E2M1) || (expandXDesc->GetDataType() == ge::DT_FLOAT4_E1M2)) {
        OP_TILING_CHECK(xDim1 % EVEN_ALIGN,
            OP_LOGE(nodeName,
            "When expandx data type is FLOAT_E2M1/FLOAT_E1M2, the last axis should be even, please check."),
            return ge::GRAPH_FAILED);
    }

    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0, OP_LOGE(nodeName, "xShape's dim0 not equal to expertIdShape's dim0, "
        "xShape's dim0 is %ld, expertIdShape's dim0 is %ld.", xDim0, expertIdsDim0), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX) || (expertIdsDim1 > moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum),
        OP_LOGE(nodeName, "expertIdShape's dim1(k) should be in (0, min(%ld, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum = %ld)], "
        "but got expertIdShape's dim1=%ld.", K_MAX, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum, expertIdsDim1), return ge::GRAPH_FAILED);
    if (tc.isLayered) {
        const gert::StorageShape *expertScaleStorageShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);
        const int64_t expertScalesDim0 = expertScaleStorageShape->GetStorageShape().GetDim(0);
        const int64_t expertScalesDim1 = expertScaleStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK((expertScalesDim0 != expertIdsDim0) || (expertScalesDim1 != expertIdsDim1),
            OP_LOGE(nodeName, "expertScaleShape's dim not equal to expertIdShape's dim, "
            "expertScaleShape's dim0 is %ld, expertIdShape's dim0 is %ld. expertScaleShape's dim1 is %ld, expertIdShape's dim1 is %ld.",
            expertScalesDim0, expertIdsDim0, expertScalesDim1, expertIdsDim1), return ge::GRAPH_FAILED);
    }
    tilingData.moeDistributeDispatchV2Info.k = static_cast<uint32_t>(expertIdsDim1);
    tc.k = static_cast<uint32_t>(expertIdsDim1);

    // 校验scales的维度
    uint64_t h = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bs = tilingData.moeDistributeDispatchV2Info.bs;
    tc.bs = bs;
    uint64_t scalesRow = 0;
    uint64_t scalesCol = 0;
    uint32_t scalesTypeSize = 0;
    uint64_t scalesCount = 0;
    if (tc.hasScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return ge::GRAPH_FAILED);
        size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        scalesRow = static_cast<uint64_t>(scalesDim0);
        scalesTypeSize = ge::GetSizeByDataType(scalesDesc->GetDataType());
        if (scalesDimNum == ONE_DIM) {
            //A3不会进此分支
            auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
            OP_TILING_CHECK((tc.quantMode == static_cast<uint32_t>(moe_dispatch_defs::QuantMode::STATIC_QUANT)) &&
            (expandXDesc->GetDataType() == ge::DT_INT8) &&
            (scalesDim0 != static_cast<int64_t>(h)) &&
            (scalesDim0 != static_cast<int64_t>(STATIC_SCALE_DIM_0)),
            OP_LOGE(nodeName, "The expected scalesDim0 is %lu or %lu in static quant, but got %ld",
            h, STATIC_SCALE_DIM_0, scalesDim0), return ge::GRAPH_FAILED);
            OP_TILING_CHECK((tc.quantMode == static_cast<uint32_t>(moe_dispatch_defs::QuantMode::STATIC_QUANT))
                && (expandXDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDim0 != STATIC_SCALE_DIM_0),
                OP_LOGE(nodeName,
                    "The expected scalesDim0 is 1 when expandX datatype is hif8 in static quant, but got %ld",
                    scalesDim0), return ge::GRAPH_FAILED);
            scalesCol = ONE_DIM_SCALE_COL_NUM;
            scalesCount = static_cast<uint64_t>(scalesDim0);
        } else if (tc.quantMode == static_cast<uint32_t>(moe_dispatch_defs::QuantMode::UNQUANT)) {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(scalesDim0 != bs,
                OP_LOGE(nodeName, "The expected scalesDim0 is %u when scales is not null in non-quant, but got %ld",
                bs, scalesDim0), return ge::GRAPH_FAILED);
            OP_TILING_CHECK(scalesDim1 > h,
                OP_LOGE(nodeName,
                "The expected scalesDim1 is less than or equal to %lu when scales is not null in non-quant, "
                "but got %ld", h, scalesDim1), return ge::GRAPH_FAILED);
            scalesCol = static_cast<uint64_t>(scalesDim1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        } else {
            OP_TILING_CHECK(
                CheckTwoDimScalesShape(context, nodeName, tilingData, tc, config)
                    != ge::GRAPH_SUCCESS,
                OP_LOGE(nodeName, "CheckTwoDimScalesShape failed."), return ge::GRAPH_FAILED);
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            scalesCol = static_cast<uint64_t>(scalesDim1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        }
    }
    tilingData.moeDistributeDispatchV2Info.scalesRow = scalesRow;
    tilingData.moeDistributeDispatchV2Info.scalesCol = scalesCol;
    tilingData.moeDistributeDispatchV2Info.scalesCount = scalesCount;
    tilingData.moeDistributeDispatchV2Info.scalesTypeSize = scalesTypeSize;
    tc.scalesRow = scalesRow;
    tc.scalesCol = scalesCol;
    tc.scalesTypeSize = scalesTypeSize;

    uint32_t rankNumPerSharedExpert = 0;
    uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t maxBs = globalBs / epWorldSizeU32;
    uint32_t maxSharedGroupNum = 0;
    if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) { // 除零保护
        rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
    }
    if (tc.isSharedRank) { // 本卡为共享专家
        A = maxBs * maxSharedGroupNum;
    } else {     // 本卡为moe专家
        A = globalBs * static_cast<uint32_t>(std::min(static_cast<int64_t>(tc.localExpertNum), expertIdsDim1));
    }

    // 使用公共层 elastic_info_check 校验 elasticInfo 维度精确值
    if (tc.hasElastic) {
        uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchV2Info.epWorldSize;
        ge::graphStatus ret = elastic_info_check::CheckShape(
            const_cast<gert::TilingContext*>(context), config.elasticInfoIndex, epWorldSizeU32, nodeName);
        OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "elasticInfo shape check failed, expected dim0 = 4 + 2 * epWorldSize."),
            return ge::GRAPH_FAILED);
        A = std::max(static_cast<int64_t>(maxBs * maxSharedGroupNum),
            globalBs * std::min(static_cast<int64_t>(tc.localExpertNum), expertIdsDim1));
    }
    tilingData.moeDistributeDispatchV2Info.a = A;
    tc.A = A;

    // 使用公共层 performance_info_check 校验 performanceInfo 维度精确值
    if (tc.hasPerformance) {
        uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchV2Info.epWorldSize;
        ge::graphStatus ret = performance_info_check::CheckShape(
            const_cast<gert::TilingContext*>(context), config.performanceInfoIndex, epWorldSizeU32, nodeName);
        OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "performanceInfo shape check failed, expected dim0 = epWorldSize."),
            return ge::GRAPH_FAILED);
    }

    // 校验expandX的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A), OP_LOGE(nodeName,
        "expandX's dim0 not greater than or equal to A*tpWorldSize, "
        "expandX's dim0 is %ld, A*tpWorldSize is %ld.", expandXDim0, tpWorldSize * A), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDim1 != expandXDim1, OP_LOGE(nodeName, "expandX's dim1 not equal to xShape's dim1, "
        "xShape's dim1 is %ld, expandX's dim1 is %ld.", xDim1, expandXDim1), return ge::GRAPH_FAILED);

    // 校验dynamicScales的维度
    if ((tc.quantMode != static_cast<uint32_t>(moe_dispatch_defs::QuantMode::UNQUANT)) && (tc.quantMode != static_cast<uint32_t>(moe_dispatch_defs::QuantMode::STATIC_QUANT))) {
        // Dim0
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(dynamicScalesDim0 < static_cast<int64_t>(A) * tpWorldSize, OP_LOGE(nodeName,
            "dynamicScales's dim0 should be equal to or greater than A*tpWorldSize, dynamicScales's dim0=%ld, A*tpWorldSize=%ld.",
            dynamicScalesDim0, A * tpWorldSize), return ge::GRAPH_FAILED);
        // Dim1, only for pergroup and mx
        if (tc.quantMode != static_cast<uint32_t>(moe_dispatch_defs::QuantMode::PERTOKEN_DYNAMIC_QUANT)) {
            const uint64_t dynamicScalesDim1 = static_cast<uint64_t>(dynamicScalesStorageShape->GetStorageShape().GetDim(1));
            OP_TILING_CHECK((tc.quantMode == static_cast<uint32_t>(moe_dispatch_defs::QuantMode::MX_QUANT))
                && (dynamicScalesDim1 != ops::CeilAlign(ops::CeilDiv(h, static_cast<uint64_t>(MX_BLOCK_SIZE)),
                    static_cast<uint64_t>(EVEN_ALIGN))),
                OP_LOGE(nodeName, "dynamicScales's dim1 should be equal to %lu and even when quantMode=%u, but got %lu.",
                ops::CeilAlign(ops::CeilDiv(h, static_cast<uint64_t>(MX_BLOCK_SIZE)),
                    static_cast<uint64_t>(EVEN_ALIGN)),
                    tc.quantMode, dynamicScalesDim1),
                    return ge::GRAPH_FAILED);
            OP_TILING_CHECK((dynamicScalesDim1 != ops::CeilDiv(h, static_cast<uint64_t>(PERGROUP_BLOCK_SIZE))) &&
                (tc.quantMode == static_cast<uint32_t>(moe_dispatch_defs::QuantMode::PERGROUP_DYNAMIC_QUANT)),
                OP_LOGE(nodeName, "dynamicScales's dim1 should be equal to %lu when quantMode=%u, but got %lu.",
                ops::CeilDiv(h, static_cast<uint64_t>(PERGROUP_BLOCK_SIZE)), tc.quantMode, dynamicScalesDim1),
                return ge::GRAPH_FAILED);
        }
    }

    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    const int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    int64_t minAssistInfoDim0 = static_cast<int64_t>(A * ASSIST_INFO_NUM_PER_A);
    OP_TILING_CHECK(assistInfoDim0 < minAssistInfoDim0, OP_LOGE(nodeName, "assistInfoDim0 < minAssistInfoDim0,"
        " assistInfoDim0 is %ld, minAssistInfoDim0 is %ld.", assistInfoDim0, minAssistInfoDim0), return ge::GRAPH_FAILED);

    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);

    if (tc.hasElastic) {
        OP_TILING_CHECK(expertTokenNumsDim0 != (tc.localExpertNum > 1 ? tc.localExpertNum : 1),  OP_LOGE(nodeName,
            "elastic scaling expertTokenNums's Dim0 not equal to max(localMoeExpertNum,1), expertTokenNumsDim0 is %ld, "
            "localMoeExpertNum is %ld.", expertTokenNumsDim0, tc.localExpertNum), return ge::GRAPH_FAILED);
    } else if (tc.isSharedRank) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1, OP_LOGE(nodeName, "shared expertTokenNums's dim0 %ld not equal to 1.",
            expertTokenNumsDim0), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(expertTokenNumsDim0 != tc.localExpertNum, OP_LOGE(nodeName,
            "moe expertTokenNums's Dim0 not equal to localMoeExpertNum, expertTokenNumsDim0 is %ld, "
            "localMoeExpertNum is %ld.", expertTokenNumsDim0, tc.localExpertNum), return ge::GRAPH_FAILED);
    }

    // 校验epRecvCount和tpRecvCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpRecvCountDim0 = tpRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = ((tc.isSharedRank) ? epWorldSize : epWorldSize * tc.localExpertNum);
    if (tc.hasElastic){
        epRecvCount = std::max(epWorldSize, epWorldSize * tc.localExpertNum);
    }
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        epRecvCount *= tpWorldSize;
    }
    if (tc.isLayered) {
        // 如果是分层方案，则需要校验全新的shape，额外的globalBs * 2 * k * epWorldSize / 8 用来存储token的cnt信息与offset信息，为了兼容A2&A5 这里取/8。
        epRecvCount = epWorldSize * tc.localExpertNum + globalBs * 2 * expertIdsDim1 * epWorldSize / RANK_NUM_PER_NODE_A2;
        OP_TILING_CHECK(epRecvCountDim0 < epRecvCount, OP_LOGE(nodeName,
        "dimension 0 of epRecvCount should be greater than or equal to epWorldSize * localMoeExpertNum + globalBs * 2 * k * epWorldSize / 8, "
        "but dimension 0 of epRecvCount is %ld, epWorldSize is %ld, localMoeExpertNum is %ld, k is %ld.",
        epRecvCountDim0, epWorldSize, tc.localExpertNum, expertIdsDim1), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(epRecvCountDim0 < epRecvCount, OP_LOGE(nodeName,
        "dimension 0 of epRecvCount should be greater than or equal to epWorldSize * localMoeExpertNum * tpWorldSize, "
        "but dimension 0 of epRecvCount is %ld, epWorldSize is %ld, localMoeExpertNum is %ld, tpWorldSize is %ld.",
        epRecvCountDim0, epWorldSize, tc.localExpertNum, tpWorldSize), return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(tpRecvCountDim0 != tpWorldSize, OP_LOGE(nodeName,
        "dimension 0 of tpRecvCount should be equal to tpWorldSize, but dimension 0 of tpRecvCount is %ld, "
        "tpWorldSize is %ld.", tpRecvCountDim0, tpWorldSize), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 校验输入输出 tensor 指针非空
static bool CheckTensorPtrNullptr(const gert::TilingContext *context, const char *nodeName,
    const DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    if (tc.hasScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
    }
    if (tc.hasActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
    }
    if (tc.hasElastic) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
    }
    if (tc.hasPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        OP_TILING_CHECK(performanceInfoDesc == nullptr, OP_LOGE(nodeName, "performanceInfoDesc is null."), return false);
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    if (tc.quantMode >= static_cast<uint32_t>(moe_dispatch_defs::QuantMode::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."),
            return false);
    }
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."), return false);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);

    return true;
}

static ge::graphStatus CheckMc2Context(gert::TilingContext *context, const char *nodeName, DispatchV2Config &config)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE(nodeName, "contextShape is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "contextShape dims must be 1, but current dim num is %lu.",
        contextStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    auto contextDesc = context->GetInputDesc(config.contextIndex);
    OP_TILING_CHECK(contextDesc == nullptr, OP_LOGE(nodeName, "contextDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "context dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(contextDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "context format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 执行完整校验流程
ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::TilingCheckMoeDistributeDispatch(
    gert::TilingContext *context, const char *nodeName,
    const DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    OP_TILING_CHECK(!CheckTensorPtrNullptr(context, nodeName, tc, config),
        OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, tc, config),
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, tc.hasScales, tc.quantMode,
        tc.hasActiveMask, tc.hasElastic, tc.hasPerformance, config),
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, tc, config),
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 使用 TilingCtx 校验并计算 WinSize
static ge::graphStatus CheckAndCalWinSize(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
    const char *nodeName, DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    CheckWinSizeData winSizeData;
    winSizeData.localMoeExpertNum = tc.localExpertNum;
    winSizeData.sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    winSizeData.a = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.a);
    winSizeData.h = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h);
    winSizeData.k = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.k);
    winSizeData.bs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.bs);
    winSizeData.epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    winSizeData.globalBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.globalBs);
    winSizeData.moeExpertNum = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    winSizeData.tpWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    winSizeData.isSetFullMeshV2 = tc.isFullMeshV2;
    winSizeData.isLayered = tc.isLayered;
    winSizeData.isMc2Context = config.isMc2Context;
    OP_TILING_CHECK(CheckWinSize(context, nodeName, winSizeData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get WinSize failed."), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.totalWinSizeEp = winSizeData.totalWinSizeEp;
    tc.totalWinSizeEp = winSizeData.totalWinSizeEp;
    if (winSizeData.tpWorldSize == TP_WORLD_SIZE_TWO) {
        tilingData.moeDistributeDispatchV2Info.totalWinSizeTp = winSizeData.totalWinSizeTp;
    }
    return ge::GRAPH_SUCCESS;
}

// Ceil 函数已替换为公共层 moe_dispatch_utils::AlignUp
static uint32_t SendToMoeExpertUsedBuffer(const gert::TilingContext *context,
    const MoeDistributeDispatchV2TilingData &tilingData, uint32_t expertIdsBufSize, uint32_t kSize, const char *nodeName)
{
    uint32_t UbForMoe = 0;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    auto platformInfo = moe_dispatch_utils::GetPlatformInfo(context);
    uint32_t aivNum = platformInfo.aivNum;
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
    uint32_t maskSizePerExpert = moe_dispatch_utils::AlignUp((expertIdsBufSize / sizeof(int32_t)) / 8, UB_ALIGN); // 8 is 1byte->8bit
    uint32_t expertMaskBufSize = maskSizePerExpert * moe_dispatch_utils::AlignUp(moeExpertNum, moeUsedAivNum) / moeUsedAivNum;
    UbForMoe = UbForMoe + expertMaskBufSize; // expertMaskBuf
    UbForMoe = UbForMoe + moe_dispatch_utils::AlignUp(moeExpertNum * sizeof(int32_t), UB_ALIGN); // tokenNumToExpertBuf
    return UbForMoe;
}

// 计算 AllToAll 基础 buffer 大小（简化参数，从 tilingData 获取所需值）
static uint32_t AllToAllBasicUsedBuffer(const gert::TilingContext *context, const MoeDistributeDispatchV2TilingData &tilingData,
    uint32_t &expertIdsBufSize, uint32_t &hOutSizeAlign, const char *nodeName)
{
    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint32_t hSize = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bsSize = tilingData.moeDistributeDispatchV2Info.bs;
    uint32_t kSize = tilingData.moeDistributeDispatchV2Info.k;
    uint32_t expertIdsCnt = bsSize * kSize;

    uint32_t UbForBasic = 0;
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    if (tilingData.moeDistributeDispatchV2Info.hasElasticInfo) {
        uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSize) * sizeof(int32_t);
        uint32_t elasticInfoSizeAlign = moe_dispatch_utils::AlignUp(elasticInfoSize, UB_ALIGN);
        UbForBasic = UbForBasic + elasticInfoSizeAlign; // elasticInfoBuf_
    }

    uint32_t sizeofX = 2U;
    uint32_t hAlignSize = moe_dispatch_utils::AlignUp(hSize * sizeofX, UB_ALIGN);
    UbForBasic = UbForBasic + hAlignSize * BUFFER_NUM; //inQueue

    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    uint32_t hOutSize = (quantMode == static_cast<uint32_t>(moe_dispatch_defs::QuantMode::UNQUANT)) ? hSize * SIZE_OF_UNQUANT : hSize;
    hOutSizeAlign = moe_dispatch_utils::AlignUp(hOutSize, UB_ALIGN);
    if ((quantMode == static_cast<uint32_t>(moe_dispatch_defs::QuantMode::UNQUANT)) && isScales) {
        uint32_t scaleInBytes = tilingData.moeDistributeDispatchV2Info.scalesCol *
                    tilingData.moeDistributeDispatchV2Info.scalesTypeSize;
        hOutSizeAlign += scaleInBytes;
    } else if (quantMode == static_cast<uint32_t>(moe_dispatch_defs::QuantMode::PERTOKEN_DYNAMIC_QUANT)) {
        hOutSizeAlign += sizeof(float);
    }
    uint32_t hScaleSizeAlign = moe_dispatch_utils::AlignUp(hOutSizeAlign, UB_ALIGN);
    uint32_t tokenQuantAlign = hScaleSizeAlign / sizeof(int32_t);
    hOutSizeAlign = tokenQuantAlign * sizeof(int32_t) + UB_ALIGN;
    uint32_t blockCntPerToken = moe_dispatch_utils::AlignUp(hOutSizeAlign, SPLIT_BLOCK_DATA_SIZE) / SPLIT_BLOCK_DATA_SIZE;
    uint32_t hCommuSize = blockCntPerToken * SPLIT_BLOCK_SIZE;
    UbForBasic = UbForBasic + hCommuSize * BUFFER_NUM; // outBuf

    expertIdsBufSize = moe_dispatch_utils::AlignUp(expertIdsCnt * sizeof(int32_t), ALIGN_256);
    UbForBasic = UbForBasic + expertIdsBufSize; //expertIdsBuf_
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
    ubSize = ubSize + AllToAllBasicUsedBuffer(context, tilingData, expertIdsBufSize, hOutSizeAlign, nodeName);

    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    bool isScales = (scalesStorageShape != nullptr);
    bool needMaskCalFlag = (isActiveMask || tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum != 0);
    if (needMaskCalFlag) {
        ubSize = ubSize + expertIdsBufSize; //gatherMaskTBuf_
    }

    uint32_t hFp32Size = moe_dispatch_utils::AlignUp(hSize * sizeof(float), UB_ALIGN);
    if ((quantMode == static_cast<uint32_t>(moe_dispatch_defs::QuantMode::PERGROUP_DYNAMIC_QUANT)) && isScales) {
        hFp32Size = moe_dispatch_utils::AlignUp(hSize, PERGROUP_BLOCK_SIZE) * sizeof(float);
    }
    uint32_t bsKAlign256 = moe_dispatch_utils::AlignUp(expertIdsCnt * SIZE_OF_HALF, ALIGN_256);
    uint32_t expertIdsSize = moe_dispatch_utils::AlignUp(expertIdsCnt * sizeof(int32_t), UB_ALIGN);
    uint32_t xActivateMaskSize = bsSize * moe_dispatch_utils::AlignUp(kSize * sizeof(bool), UB_ALIGN) * SIZE_OF_HALF;
    uint32_t maxSizeForUbBuffer = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSizeForUbBuffer = maxSizeForUbBuffer > xActivateMaskSize ? maxSizeForUbBuffer : xActivateMaskSize;
    maxSizeForUbBuffer = maxSizeForUbBuffer > bsKAlign256 ? maxSizeForUbBuffer : bsKAlign256;
    tilingData.moeDistributeDispatchV2Info.maxSizeForUbBuffer = maxSizeForUbBuffer;
    if (quantMode > static_cast<uint32_t>(moe_dispatch_defs::QuantMode::UNQUANT)) {
        uint32_t hOutAlignUbSize = moe_dispatch_utils::AlignUp(hOutSizeAlign, UB_ALIGN);
        ubSize = ubSize + hOutAlignUbSize;
        ubSize = ubSize + BUFFER_NUM * maxSizeForUbBuffer; // receiveDataCastFloatBuf smoothScalesBuf
    } else if (needMaskCalFlag) {
        ubSize = ubSize + BUFFER_NUM * maxSizeForUbBuffer;
    }

    if (tilingData.moeDistributeDispatchV2Info.isExpertMask || (tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum != 0)) {
        uint32_t axisBSAlign = moe_dispatch_utils::AlignUp(bsSize * sizeof(int32_t), UB_ALIGN);
        ubSize = ubSize +  axisBSAlign; //validBsIndexTBuf_
        uint32_t validBufferSize = expertIdsSize > xActivateMaskSize ? expertIdsSize : xActivateMaskSize;
        ubSize = ubSize + validBufferSize;  //validExpertIndexBuf_
    }

    ubSize = ubSize + SendToMoeExpertUsedBuffer(context, tilingData, expertIdsBufSize, kSize, nodeName);
    uint64_t ubRealSize = 0;
    auto platformInfo = moe_dispatch_utils::GetPlatformInfo(context);
    ubRealSize = platformInfo.ubSize;
    OP_TILING_CHECK(ubSize > ubRealSize, OP_LOGE(nodeName,
        "Current scenario is exceeds the size limit, BS = %u, H = %u, topK = %u, moeExpertNum = %u, aivNum available = %u, used UBsize = %u",
        bsSize, hSize, kSize, tilingData.moeDistributeDispatchV2Info.moeExpertNum, platformInfo.aivNum, ubSize), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    auto platformInfo = moe_dispatch_utils::GetPlatformInfo(context);
    uint32_t aivNum = platformInfo.aivNum;
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::MoeDistributeDispatchA3TilingFuncImplPublic(
    gert::TilingContext *context, DispatchV2Config &config)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    DispatchV2TilingCtx tc;  // 统一承载所有中间值
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchV2 tiling check func.");

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp, groupTp, tc, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    // 获取scales
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
    tc.hasScales = (scalesStorageShape != nullptr);

    // 获取xActiveMask
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    tc.hasActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isTokenMask = ((tc.hasActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeDispatchV2Info.isExpertMask = ((tc.hasActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));
    tc.isTokenMask = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    tc.isExpertMask = tilingData->moeDistributeDispatchV2Info.isExpertMask;

    // 获取elasticInfo
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    tc.hasElastic = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.hasElasticInfo = tc.hasElastic;

    // 获取performanceInfo
    const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(config.performanceInfoIndex);
    tc.hasPerformance = (performanceInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isPerformance = tc.hasPerformance;

    tc.quantMode = tilingData->moeDistributeDispatchV2Info.quantMode;

    // 检查quantMode和scales是否匹配
    if (CheckQuantModeMatchScales(context, nodeName, tc.hasScales, tc.quantMode, config) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    // 检查context输入
    if (config.isMc2Context) {
        OP_TILING_CHECK(
            CheckMc2Context(context, nodeName, config) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Tiling check context failed."), return ge::GRAPH_FAILED);
    }
    tc.isMc2Context = config.isMc2Context;

    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(
        TilingCheckMoeDistributeDispatch(context, nodeName, tc, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(CheckAttrs(context, nodeName, *tilingData, tc, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeDistributeDispatchV2Info.epRankId;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    tc.isSharedRank = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData, tc, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    // 校验UB大小
    if (tc.isFullMeshV2) {
        OP_TILING_CHECK(CheckUBSize(context, *tilingData, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Tiling check UB size failed."), return ge::GRAPH_FAILED);
    }

    // 校验win区大小
    OP_TILING_CHECK(CheckAndCalWinSize(context, *tilingData, nodeName, tc, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    uint32_t tpWorldSize = tilingData->moeDistributeDispatchV2Info.tpWorldSize;
    if (!config.isMc2Context) {
        // 使用公共层通信配置接口（v2 不需要 URMA 参数）
        moe_dispatch_comm_cfg::SetEpAllToAllCfg(tilingData->mc2InitTiling, tilingData->mc2CcTiling1, groupEp, tc.isLayered);
        if (tpWorldSize > 1) {
            moe_dispatch_comm_cfg::SetTpAllGatherCfg(tilingData->mc2InitTiling, tilingData->mc2CcTiling2, groupTp);
        }
    }
    uint64_t tilingKey = CalTilingKey(tc.hasScales, tc.quantMode, tpWorldSize, tc.isFullMeshV2, tc.isLayered);
    tilingData->moeDistributeDispatchV2Info.isMc2Context = config.isMc2Context;

    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t numBlocks = 1U;
    auto platformInfo = moe_dispatch_utils::GetPlatformInfo(context);
    uint32_t aivNum = platformInfo.aivNum;
    tc.aivNum = aivNum;
    uint32_t epWorldSize = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    uint32_t serverNum = epWorldSize / RANK_NUM_PER_NODE;
    OP_TILING_CHECK((tc.isLayered && ((aivNum < RANK_NUM_PER_NODE)||(aivNum <= 2 * serverNum))), OP_LOGE(nodeName,
        "Layered should use aiv num greater than 16 and 2 * serverNum, now is %u", aivNum), return ge::GRAPH_FAILED);
    uint64_t ubSize = platformInfo.ubSize;
    tc.totalUbSize = ubSize;
    numBlocks = platformInfo.blockDim;
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
    const char* socName = aclrtGetSocName();
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

    IMPL_OP(MoeDistributeDispatchV2)
        .ExceptionDumpParseFunc(MoeDistributeDispatchV2ExceptionImplWrapper);
}
#endif

} // namespace optiling