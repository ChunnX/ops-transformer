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
 * \file fa_tiling_shape.cpp
 * \brief
 */

#include <vector>
#include <algorithm>
#include "fa_tiling_shape.h"


namespace optiling {
static const std::map<FaLayout, std::vector<FaAxis>> FA_LAYOUT_AXIS_MAP = {
    {FaLayout::BSH, {FaAxis::B, FaAxis::S, FaAxis::H}},
    {FaLayout::BSND, {FaAxis::B, FaAxis::S, FaAxis::N, FaAxis::D}},
    {FaLayout::BNSD, {FaAxis::B, FaAxis::N, FaAxis::S, FaAxis::D}},
    {FaLayout::NZ, {FaAxis::Bn, FaAxis::N, FaAxis::D1, FaAxis::Bs, FaAxis::D0}},
    {FaLayout::TND, {FaAxis::T, FaAxis::N, FaAxis::D}},
    {FaLayout::NBSD, {FaAxis::N, FaAxis::B, FaAxis::S, FaAxis::D}},
    {FaLayout::NTD, {FaAxis::N, FaAxis::T, FaAxis::D}},
    {FaLayout::BS2, {FaAxis::B, FaAxis::S2}},
    {FaLayout::S1S2, {FaAxis::S1, FaAxis::S2}},
    {FaLayout::BnBsH, {FaAxis::Bn, FaAxis::Bs, FaAxis::H}},
    {FaLayout::BnNBsD, {FaAxis::Bn, FaAxis::N, FaAxis::Bs, FaAxis::D}},
    {FaLayout::BNS1S2, {FaAxis::B, FaAxis::N, FaAxis::S1, FaAxis::S2}},
    {FaLayout::INS1S2, {FaAxis::CONST, FaAxis::N, FaAxis::S1, FaAxis::S2}},
    {FaLayout::BNS11, {FaAxis::B, FaAxis::N, FaAxis::S1, FaAxis::CONST}},
    {FaLayout::TN1, {FaAxis::T, FaAxis::N, FaAxis::CONST}},
    {FaLayout::BS1S2, {FaAxis::B, FaAxis::S1, FaAxis::S2}},
    {FaLayout::B1S1S2, {FaAxis::B, FaAxis::CONST, FaAxis::S1, FaAxis::S2}},
    {FaLayout::IS1S2, {FaAxis::CONST, FaAxis::S1, FaAxis::S2}},
    {FaLayout::I1S1S2, {FaAxis::CONST, FaAxis::CONST, FaAxis::S1, FaAxis::S2}},
    // {FaLayout::S1S1, {FaAxis::S1}},
};

static bool equal_to(const int64_t& a, const int64_t& b)
{
    return (a == b);
}

static bool greater(const int64_t& a, const int64_t& b)
{
    return (a > b);
}

static bool greater_equal(const int64_t& a, const int64_t& b)
{
    return (a >= b);
}

static bool less(const int64_t& a, const int64_t& b)
{
    return (a < b);
}

static bool less_equal(const int64_t& a, const int64_t& b)
{
    return (a <= b);
}

static bool not_equal_to(const int64_t& a, const int64_t& b)
{
    return (a != b);
}

static bool ignore_input(const int64_t& a, const int64_t& b)
{
    (void)a;
    (void)b;
    return true;
}

static ge::graphStatus GetLayoutAxes(std::vector<FaAxis> &layoutAxes, const FaLayout &layout,
    const std::string &opName, const std::string &funcName)
{
    auto it = FA_LAYOUT_AXIS_MAP.find(layout);
    if (it == FA_LAYOUT_AXIS_MAP.end()) {
        OP_LOGE(opName, "[%s] compare layout %s is unsupported.",
            funcName.c_str(), LayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    layoutAxes = it->second;
    return ge::GRAPH_SUCCESS;
}

const std::map<FaCompareType, CompareFunc<int64_t>> FaTilingShapeCompare::compareFuncMap_ = {
    {FaCompareType::EQUAL, equal_to},
    {FaCompareType::GREATER, greater},
    {FaCompareType::GREATER_EQUAL, greater_equal},
    {FaCompareType::LESS, less},
    {FaCompareType::LESS_EQUAL, less_equal},
    {FaCompareType::NOT_EQUAL, not_equal_to},
    {FaCompareType::IGNORE_INPUT, ignore_input}
    
};

bool FaTilingShape::HasAxis(const FaAxis &axis) const
{   
    const auto& layoutIt = FA_LAYOUT_AXIS_MAP.find(layout_);
    if (layoutIt == FA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<FaAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }

    return true;
}

size_t FaTilingShape::GetAxisIdx(const FaAxis &axis) const
{
    if (HasAxis(axis)) {
        const std::vector<FaAxis>& axes = FA_LAYOUT_AXIS_MAP.find(layout_)->second;
        const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
        return std::distance(axes.begin(), axisIt);
    }
    return 0;
}

int64_t FaTilingShape::GetAxisNum(const FaAxis &axis) const
{
    return HasAxis(axis) ? shape_.GetDim(GetAxisIdx(axis)) : invalidDimValue_;
}

ge::graphStatus FaTilingShape::CheckHasAxis(const FaAxis &axis, const std::string &funcName) const
{
    if (shape_.GetDimNum() == 0) {
        OP_LOGE(opName_, "[%s] the dim number of %s is 0.", funcName.c_str(), name_.c_str());
        return ge::GRAPH_FAILED;
    }

    std::vector<FaAxis> layoutAxes;
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

    if ((axis == FaAxis::D)) {
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

std::string FaTilingShapeCompare::CompareTypeToSerialString(const FaCompareType compareType) const
{
    switch (compareType) {
        case FaCompareType::EQUAL: 
            return "EQUAL";
        case FaCompareType::GREATER: 
            return "GREATER";
        case FaCompareType::GREATER_EQUAL: 
            return "GREATER_EQUAL";
        case FaCompareType::LESS: 
            return "LESS";
        case FaCompareType::LESS_EQUAL: 
            return "LESS_EQUAL";
        case FaCompareType::NOT_EQUAL: 
            return "NOT_EQUAL";
        default: 
            return "UNKNOWN";
    }
}

std::string FaTilingShapeCompare::CompareTypeToSerialSymbolString(const FaCompareType &compareType) const
{
    switch (compareType) {
        case FaCompareType::EQUAL: 
            return "==";
        case FaCompareType::GREATER: 
            return ">";
        case FaCompareType::GREATER_EQUAL: 
            return ">=";
        case FaCompareType::LESS: 
            return "<";
        case FaCompareType::LESS_EQUAL: 
            return "<=";
        case FaCompareType::NOT_EQUAL: 
            return "!=";
        default: 
            return "UNKNOWN";
    }
}

ge::graphStatus FaTilingShapeCompare::GetExpectedShapeSpecial(gert::Shape &shapeExpected,
    const FaTilingShapeCompareParam &param, const std::string &funcName) const
{
    if (layout_ == FaLayout::BNS1S2) {
        shapeExpected = gert::Shape({param.B, param.N, param.S1, param.S2});
    } else if (layout_ == FaLayout::INS1S2) {
        shapeExpected = gert::Shape({param.CONST, param.N, param.S1, param.S2});
    } else if (layout_ == FaLayout::BNS11) {
        shapeExpected = gert::Shape({param.B, param.N, param.S1, param.CONST});
    } else if (layout_ == FaLayout::TN1) {
        shapeExpected = gert::Shape({param.T, param.N, param.CONST});
    } else if (layout_ == FaLayout::BS2) {
        shapeExpected = gert::Shape({param.B, param.S2});
    } else if (layout_ == FaLayout::S1S2) {
        shapeExpected = gert::Shape({param.S1, param.S2});
    } else if (layout_ == FaLayout::BS1S2) {
        shapeExpected = gert::Shape({param.B, param.S1, param.S2});
    } else if (layout_ == FaLayout::B1S1S2) {
        shapeExpected = gert::Shape({param.B, param.CONST, param.S1, param.S2});
    } else if (layout_ == FaLayout::IS1S2) {
        shapeExpected = gert::Shape({param.CONST, param.S1, param.S2});
    } else if (layout_ == FaLayout::I1S1S2) {
        shapeExpected = gert::Shape({param.CONST, param.CONST, param.S1, param.S2});
    // } else if (layout_ == FaLayout::S1S1) {
    //     shapeExpected = gert::Shape({param.S1});
    } else {
        OP_LOGE(opName_, "[%s] layout %s is unsupported", funcName.c_str(), LayoutToSerialString(layout_).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaTilingShapeCompare::GetExpectedShape(gert::Shape &shapeExpected,
    const FaTilingShapeCompareParam &param, const std::string &funcName) const
{
    if (layout_ == FaLayout::BSH) {
        shapeExpected = gert::Shape({param.B, param.S, param.H});
    } else if (layout_ == FaLayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout_ == FaLayout::BNSD) {
        shapeExpected = gert::Shape({param.B, param.N, param.S, param.D});
    } else if (layout_ == FaLayout::BnBsH) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.H});
    } else if (layout_ == FaLayout::BnNBsD) {
        shapeExpected = gert::Shape({param.Bn, param.N, param.Bs, param.D});
    } else if (layout_ == FaLayout::NZ) {
        shapeExpected = gert::Shape({param.Bn, param.N, param.D / param.D0, param.Bs, param.D0});
    } else if (layout_ == FaLayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout_ == FaLayout::NBSD) {
        shapeExpected = gert::Shape({param.N, param.B, param.S, param.D});
    } else if (layout_ == FaLayout::NTD) {
        shapeExpected = gert::Shape({param.N, param.T, param.D});
    } else {
        return GetExpectedShapeSpecial(shapeExpected, param, funcName);
    }
    return ge::GRAPH_SUCCESS;
}

FaCompareType FaTilingShapeCompare::GetCompareType(const std::map<FaAxis, FaCompareType> &compareTypeMap,
    const FaAxis &axis) const
{
    auto it = compareTypeMap.find(axis);
    auto compareType = FaCompareType::EQUAL;
    if (it != compareTypeMap.end()) {
        compareType = it->second;
    }
    return compareType;
}

ge::graphStatus FaTilingShapeCompare::GetCompareFunc(const FaCompareType &compareType, 
    CompareFunc<int64_t> &compareFunc, const std::string &funcName) const
{
    auto it = compareFuncMap_.find(compareType);
    if (it == compareFuncMap_.end()) {
        OP_LOGE(opName_, "[%s] compare type %s is unsupported.", funcName.c_str(),
            CompareTypeToSerialString(compareType).c_str());
        return ge::GRAPH_FAILED;
    }
    compareFunc = it->second;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaTilingShapeCompare::CompareShape(FaTilingShapeCompareParam &param, const std::string &funcName) const
{
    param.H = param.N * param.D;
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    std::vector<FaAxis> layoutAxes;
    if (GetLayoutAxes(layoutAxes, layout_, opName_, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if ((shape_.GetDimNum() != shapeExpected.GetDimNum()) || (shape_.GetDimNum() != layoutAxes.size())) {
        OP_LOGE(opName_,
            "[%s] %s shape dimension is %zu, expected shape dimension is %zu, layout(%s) axes size is %zu, they should be equal.",
            funcName.c_str(), name_.c_str(), shape_.GetDimNum(), shapeExpected.GetDimNum(),
            LayoutToSerialString(layout_).c_str(), layoutAxes.size());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape_.GetDimNum(); i++) {
        auto axis = layoutAxes[i];
        auto compareType = GetCompareType(param.compareTypeMap, axis);
        CompareFunc<int64_t> compareFunc;
        if (GetCompareFunc(compareType, compareFunc, funcName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (!compareFunc(shape_.GetDim(i), shapeExpected.GetDim(i))) {
            if (param.compareTypeMap.empty()) {
                OP_LOGE(opName_, "[%s] %s layout is %s, shape %s should be equal to %s.",
                    funcName.c_str(), name_.c_str(), LayoutToSerialString(layout_).c_str(),
                    GetShapeStr(shape_).c_str(), GetShapeStr(shapeExpected).c_str());
            } else {
                OP_LOGE(opName_, "[%s] %s layout is %s, shape is %s, expected shape is %s, axis %s(%ld) should be %s expected %ld.",
                    funcName.c_str(), name_.c_str(), LayoutToSerialString(layout_).c_str(),
                    GetShapeStr(shape_).c_str(), GetShapeStr(shapeExpected).c_str(), AxisToSerialString(axis).c_str(),
                    shape_.GetDim(i), CompareTypeToSerialSymbolString(compareType).c_str(), shapeExpected.GetDim(i));
            }
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling