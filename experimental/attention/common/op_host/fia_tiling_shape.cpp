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
 * \file fia_tiling_shape.cpp
 * \brief
 */

#include <vector>
#include <algorithm>
#include "fia_tiling_shape.h"

namespace optiling {
static const std::map<FiaLayout, std::vector<FiaAxis>> FIA_LAYOUT_AXIS_MAP = {
    {FiaLayout::BSND, {FiaAxis::B, FiaAxis::S, FiaAxis::N, FiaAxis::D}},
    {FiaLayout::BNSD, {FiaAxis::B, FiaAxis::N, FiaAxis::S, FiaAxis::D}},
    {FiaLayout::BnBsH, {FiaAxis::Bn, FiaAxis::Bs, FiaAxis::H}},
    {FiaLayout::BnNBsD, {FiaAxis::Bn, FiaAxis::N, FiaAxis::Bs, FiaAxis::D}},
};

static ge::graphStatus GetLayoutAxes(std::vector<FiaAxis> &layoutAxes, const FiaLayout &layout,
    const std::string &opName, const std::string &funcName)
{
    auto it = FIA_LAYOUT_AXIS_MAP.find(layout);
    if (it == FIA_LAYOUT_AXIS_MAP.end()) {
        OP_LOGE(opName, "[%s] compare layout %s is unsupported.",
            funcName.c_str(), LayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    layoutAxes = it->second;
    return ge::GRAPH_SUCCESS;
}

bool FiaTilingShape::HasAxis(const FiaAxis &axis) const
{   
    const auto& layoutIt = FIA_LAYOUT_AXIS_MAP.find(layout_);
    if (layoutIt == FIA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<FiaAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }

    return true;
}

size_t FiaTilingShape::GetAxisIdx(const FiaAxis &axis) const
{
    if (HasAxis(axis)) {
        const std::vector<FiaAxis>& axes = FIA_LAYOUT_AXIS_MAP.find(layout_)->second;
        const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
        return std::distance(axes.begin(), axisIt);
    }
    return 0;
}

int64_t FiaTilingShape::GetAxisNum(const FiaAxis &axis) const
{
    return HasAxis(axis) ? shape_.GetDim(GetAxisIdx(axis)) : invalidDimValue_;
}

ge::graphStatus FiaTilingShape::CheckHasAxis(const FiaAxis &axis, const std::string &funcName) const
{
    if (shape_.GetDimNum() == 0) {
        OP_LOGE(opName_, "[%s] the dim number of %s is 0.", funcName.c_str(), name_.c_str());
        return ge::GRAPH_FAILED;
    }

    std::vector<FiaAxis> layoutAxes;
    if (GetLayoutAxes(layoutAxes, layout_, opName_, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shape_.GetDimNum() != layoutAxes.size()) {
        OP_LOGE(opName_,
            "[%s] %s shape dimension is %zu, expected shape dimension is %zu, layout(%s) axes size is %zu, they should be equal.",
            funcName.c_str(), name_.c_str(), shape_.GetDimNum(), layoutAxes.size(),
            LayoutToSerialString(layout_).c_str(), layoutAxes.size());
        return ge::GRAPH_FAILED;
    }

    if ((axis == FiaAxis::D)) {
        if (HasShapeD()) {
            return ge::GRAPH_SUCCESS;
        } else if (!HasShapeH()) {
            OP_LOGE(opName_, "[%s] %s's layout is %s, do not have D and H.",
                funcName.c_str(), name_.c_str(), LayoutToSerialString(layout_).c_str());
            return ge::GRAPH_FAILED;
        } else if (!hasSetN_) {
            OP_LOGE(opName_, "[%s] %s's N is not specified, cannot caculate D by H.", funcName.c_str(), name_.c_str());
            return ge::GRAPH_FAILED;
        } else if (N_ == 0) {
            OP_LOGE(opName_, "[%s] %s's N is 0.", funcName.c_str(), name_.c_str());
            return ge::GRAPH_FAILED;
        } else if (GetShapeH() % N_ != 0) {
            OP_LOGE(opName_, "[%s] %s's H(%ld) should be an integer multiple of N(%ld).",
            funcName.c_str(), name_.c_str(), GetShapeH(), N_);
            return ge::GRAPH_FAILED;
        }
    } else if (HasAxis(axis)) {
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(opName_, "[%s] %s's layout is %s, %s is not exists.",
        funcName.c_str(), name_.c_str(), LayoutToSerialString(layout_).c_str(),
        AxisToSerialString(axis).c_str());
    return ge::GRAPH_FAILED;
}
} // namespace optiling