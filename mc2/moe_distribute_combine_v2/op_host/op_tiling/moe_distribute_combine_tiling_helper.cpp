/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_tiling_helper.cc
 * \brief
 */

#include "moe_distribute_combine_tiling_helper.h"
#include "mc2_log.h"

using namespace ge;

namespace optiling {
inline bool MoeDistributeCombineTilingHelper::CheckInputTensorDim(const gert::TilingContext *context,
                                                                  const char *nodeName)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandX"), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expandX", (std::to_string(expandXStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
                    return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIds"), return false);
    OP_TILING_CHECK(expertIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expertIds", (std::to_string(expertIdsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
                    return false);
    OP_LOGD(nodeName, "expertIds dim0 = %ld", expertIdsStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertIds dim1 = %ld", expertIdsStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expandIdxStorageShape = context->GetInputShape(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdx"), return false);
    OP_TILING_CHECK(expandIdxStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expandIdx", (std::to_string(expandIdxStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);
    OP_LOGD(nodeName, "expandIdx dim0 = %ld", expandIdxStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *sharedExpertX = context->GetOptionalInputShape(SHARED_EXPERT_X_INDEX);
    if (sharedExpertX != nullptr) {
        auto attrs = context->GetAttrs();
        auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
        OP_TILING_CHECK(*sharedExpertRankNumPtr != 0, OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "sharedExpertRankNum", "invalid", "sharedExpertX only support input None "\
            "when sharedExpertRankNum is non-zero."), return false);
        OP_TILING_CHECK(((sharedExpertX->GetStorageShape().GetDimNum() != TWO_DIMS) &&
                        (sharedExpertX->GetStorageShape().GetDimNum() != THREE_DIMS)),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x", "invalid", "sharedExpertX must be 2-dimension or 3-dimension, but got %lu dim"), return false);
    }
    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckInputSendCountsTensorDim(const gert::TilingContext *context,
                                                                            const char *nodeName)
{
    const gert::StorageShape *epSendCountsStorageShape = context->GetInputShape(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epSendCounts"), return false);
    OP_TILING_CHECK(epSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "epSendCounts", (std::to_string(epSendCountsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);
    OP_LOGD(nodeName, "epSendCounts dim0 = %ld", epSendCountsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpSendCountsStorageShape = context->GetOptionalInputShape(TP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(tpSendCountsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpSendCounts"), return false);
    OP_TILING_CHECK(tpSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "tpSendCounts", (std::to_string(tpSendCountsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);
    OP_LOGD(nodeName, "tpSendCounts dim0 = %ld", tpSendCountsStorageShape->GetStorageShape().GetDim(0));
    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckInputExpertScalesTensorDim(const gert::TilingContext *context,
                                                                              const char *nodeName)
{
    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScales"), return false);
    OP_TILING_CHECK(expertScalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expertScale", (std::to_string(expertScalesStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
                    return false);
    OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(1));
    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckOutputTensorDim(const gert::TilingContext *context,
                                                                   const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return false);
    OP_TILING_CHECK(
        xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x", (std::to_string(xStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return false);
    OP_LOGD(nodeName, "x dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    return true;
}

bool MoeDistributeCombineTilingHelper::CheckTensorDim(gert::TilingContext *context, const char *nodeName)
{
    OP_TILING_CHECK(!CheckInputTensorDim(context, nodeName) || !CheckInputSendCountsTensorDim(context, nodeName) ||
                    !CheckInputExpertScalesTensorDim(context, nodeName) || !CheckOutputTensorDim(context, nodeName),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "Input param", "invalid", "The shape of parameter Input param is invalid"), return false);

    // x_active_mask当前不支持传入
    // A3/A5 v1接口的x_active_mask不支持传入
    if (OpVersionManager::GetInstance().GetVersion() == OP_VERSION_1) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskStorageShape != nullptr, OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "x", "invalid", "x_active_mask only support input None."),
            return false);
    }

    OP_TILING_CHECK(!CheckOutputTensorDim(context, nodeName), OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "Output param", "invalid", "The shape of parameter Output param is invalid"),
                    return false);

    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckActiveMask(const gert::TilingContext *context, const char *nodeName)
{
    // Check Dim/DType/Format
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskStorageShape"),
                    return false);
    const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(xActiveMaskDimNums != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x", "invalid", "xActiveMask must be 1-dim, but current dim num is %ld."),
                    return false);
    auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
    OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskDesc"), return false);
    OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "xActiveMask", Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str(), "bool"),
                    return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "xActiveMask", Ops::Base::ToString(xActiveMaskDesc->GetStorageFormat()).c_str(), "not FORMAT_FRACTAL_NZ"), return false);

    return true;
}

// 校验数据类型
bool MoeDistributeCombineTilingHelper::CheckTensorDataType(const gert::TilingContext *context, const char *nodeName)
{
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandxDesc"), return false);
    OP_TILING_CHECK((expandXDesc->GetDataType() != ge::DT_BF16) && (expandXDesc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expandX", Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), "bf16 or float16"), return false);
    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdsDesc"), return false);
    OP_TILING_CHECK((expertIdsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expertIds", Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str(), "int32"), return false);
    auto expandIdxDesc = context->GetInputDesc(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdxDesc"), return false);
    OP_TILING_CHECK((expandIdxDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expandIdx", Ops::Base::ToString(expandIdxDesc->GetDataType()).c_str(), "int32"), return false);
    auto epSendCountsDesc = context->GetInputDesc(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epSendCountsDesc"), return false);
    OP_TILING_CHECK((epSendCountsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "epSendCounts", Ops::Base::ToString(epSendCountsDesc->GetDataType()).c_str(), "int32"), return false);
    auto tpSendCountsDesc = context->GetOptionalInputDesc(TP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(tpSendCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpSendCountsDesc"), return false);
    OP_TILING_CHECK((tpSendCountsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "tpSendCounts", Ops::Base::ToString(tpSendCountsDesc->GetDataType()).c_str(), "int32"), return false);
    auto expertScalesDesc = context->GetInputDesc(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScalesDesc"), return false);
    OP_TILING_CHECK((expertScalesDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expertScales", Ops::Base::ToString(expertScalesDesc->GetDataType()).c_str(), "float"), return false);
    auto sharedExpertXDesc = context->GetOptionalInputDesc(SHARED_EXPERT_X_INDEX);
    if (sharedExpertXDesc != nullptr) {
        OP_TILING_CHECK(sharedExpertXDesc->GetDataType() != expandXDesc->GetDataType(),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x", "invalid", "sharedExpertX dataType should be the same as expandX dataType, but got sharedExpertX"
            "dataType %s, expandX dataType %s."), return false);
    }
    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != expandXDesc->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), Ops::Base::ToString(expandXDesc->GetDataType()).c_str()),
                    return false);
    return true;
}

bool MoeDistributeCombineTilingHelper::CheckTensorFormat(const gert::TilingContext *context, const char *nodeName)
{
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandxDesc"), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expandX", Ops::Base::ToString(expandXDesc->GetStorageFormat()).c_str(), "not FORMAT_FRACTAL_NZ"), return false);

    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdsDesc"), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdsDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expertIds", Ops::Base::ToString(expertIdsDesc->GetStorageFormat()).c_str(), "not FORMAT_FRACTAL_NZ"), return false);

    auto expandIdxDesc = context->GetInputDesc(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdxDesc"), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandIdxDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expandIdx", Ops::Base::ToString(expandIdxDesc->GetStorageFormat()).c_str(), "not FORMAT_FRACTAL_NZ"), return false);

    auto epSendCountsDesc = context->GetInputDesc(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epSendCountsDesc"), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(epSendCountsDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "epSendCounts", Ops::Base::ToString(epSendCountsDesc->GetStorageFormat()).c_str(), "not FORMAT_FRACTAL_NZ"), return false);

    auto tpSendCountsDesc = context->GetOptionalInputDesc(TP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(tpSendCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpSendCountsDesc"), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpSendCountsDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "tpSendCounts", Ops::Base::ToString(tpSendCountsDesc->GetStorageFormat()).c_str(), "not FORMAT_FRACTAL_NZ"), return false);

    auto expertScalesDesc = context->GetInputDesc(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScalesDesc"), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expertScales", Ops::Base::ToString(expertScalesDesc->GetStorageFormat()).c_str(), "not FORMAT_FRACTAL_NZ"), return false);

    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x", Ops::Base::ToString(xDesc->GetStorageFormat()).c_str(), "not FORMAT_FRACTAL_NZ"), return false);
    return true;
}

ge::graphStatus MoeDistributeCombineTilingHelper::TilingCheckMoeDistributeCombine(gert::TilingContext *context,
                                                                                  const char *nodeName)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName), OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "param", "invalid", "The shape of parameter param is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName), OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "param", "invalid", "The dtype of parameter param is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName), OP_LOGE_FOR_INVALID_FORMAT(nodeName, "param", "invalid", "valid format"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingHelper::TilingCheckMoeDistributeCombineA5(gert::TilingContext *context,
    const char *nodeName, const bool isTokenMask)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName), OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "param", "invalid", "The shape of parameter param is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName), OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "param", "invalid", "The dtype of parameter param is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName), OP_LOGE_FOR_INVALID_FORMAT(nodeName, "param", "invalid", "valid format"),
                    return ge::GRAPH_FAILED);
    // 检查ActiveMask信息
    if ((OpVersionManager::GetInstance().GetVersion() != OP_VERSION_1) && isTokenMask) {
        OP_TILING_CHECK(!CheckActiveMask(context, nodeName), OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "x", "invalid", "xActiveMask is invalid."),
        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling