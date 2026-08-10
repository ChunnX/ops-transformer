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
 * \file base_checker.cpp
 * \brief Implementation of QSMLABaseChecker utility methods.
 */

#include "base_checker.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"

namespace optiling {

using std::string;
using std::to_string;
using std::vector;
using namespace ge;

static const char *QSMLA_OP_NAME = "QuantSparseFlashMla";

// ---- Dtype support: checks tensor dtype is in DTYPE_SUPPORT_MAP ----
ge::graphStatus QSMLABaseChecker::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                    const std::string &name) const
{
    if (desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto &it = DTYPE_SUPPORT_MAP.find(name);
    OP_CHECK_IF(
        it == DTYPE_SUPPORT_MAP.end(),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            QSMLA_OP_NAME, name.c_str(), ("dtype support list not found in DTYPE_SUPPORT_MAP for " + name).c_str()),
        return ge::GRAPH_FAILED);
    const auto &expectList = it->second;
    if (std::find(expectList.begin(), expectList.end(), desc->GetDataType()) == expectList.end()) {
        LogErrorDtypeSupport(expectList, desc->GetDataType(), name);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ---- Format support: checks tensor origin format is in FORMAT_SUPPORT_SET ----
ge::graphStatus QSMLABaseChecker::CheckFormatSupport(const gert::CompileTimeTensorDesc *desc,
                                                     const std::string &name) const
{
    if (desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (FORMAT_SUPPORT_SET.find(desc->GetOriginFormat()) == FORMAT_SUPPORT_SET.end()) {
        OP_LOGE_FOR_INVALID_FORMAT(QSMLA_OP_NAME, name.c_str(),
                                   ge::TypeUtils::FormatToSerialString(desc->GetOriginFormat()).c_str(), "ND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ---- Dim num support: checks tensor dimension count is in expected list ----
ge::graphStatus QSMLABaseChecker::CheckDimNumSupport(const gert::StorageShape *shape,
                                                     const std::vector<size_t> &expectList,
                                                     const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    size_t actualDimNum = shape->GetStorageShape().GetDimNum();
    if (std::find(expectList.begin(), expectList.end(), actualDimNum) == expectList.end()) {
        string expectStr;
        for (size_t i = 0; i < expectList.size(); ++i) {
            expectStr += to_string(expectList[i]);
            if (i < expectList.size() - 1)
                expectStr += ", ";
        }
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(QSMLA_OP_NAME, name.c_str(), to_string(actualDimNum).c_str(),
                                                 ("dim num should be [" + expectStr + "]").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ---- Dim num in layout: checks tensor dim count matches the layout's expected dim count ----
ge::graphStatus QSMLABaseChecker::CheckDimNumInLayoutSupport(QSMLALayout layout, const gert::StorageShape *shape,
                                                             const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto &dimIt = QSMLA_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == QSMLA_LAYOUT_DIM_MAP.end()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            QSMLA_OP_NAME, name.c_str(),
            ("layout " + LayoutToSerialString(layout) + " not found in QSMLA_LAYOUT_DIM_MAP").c_str());
        return ge::GRAPH_FAILED;
    }
    size_t expected = dimIt->second;
    size_t actual = shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(actual != expected,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(QSMLA_OP_NAME, name.c_str(), to_string(actual).c_str(),
                                                         ("When layout is " + LayoutToSerialString(layout) + ", " +
                                                          name + " dim should be " + to_string(expected))
                                                             .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---- Shape size non-zero (optional tensor): checks tensor is not empty ----
ge::graphStatus QSMLABaseChecker::CheckShapeSizeNonZero(const gert::Tensor *tensor, const std::string &name) const
{
    if (tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(tensor->GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(QSMLA_OP_NAME, name.c_str(), "0",
                                                          ("Any dim of input " + name + " cannot be 0").c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---- Shape size non-zero (required tensor): checks tensor is not empty ----
ge::graphStatus QSMLABaseChecker::CheckShapeSizeNonZero(const gert::StorageShape *shape, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(shape->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(QSMLA_OP_NAME, name.c_str(), "0",
                                                          ("Any dim of input " + name + " cannot be 0").c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---- Value support (template): checks attribute value is in expected list ----
template <typename T>
ge::graphStatus QSMLABaseChecker::CheckValueSupport(const T value, const std::vector<T> &expectList,
                                                    const std::string &name) const
{
    if (std::find(expectList.begin(), expectList.end(), value) == expectList.end()) {
        string expectStr;
        for (size_t i = 0; i < expectList.size(); ++i) {
            expectStr += to_string(expectList[i]);
            if (i < expectList.size() - 1)
                expectStr += ", ";
        }
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QSMLA_OP_NAME, name.c_str(), to_string(value).c_str(),
                                              ("only support [" + expectStr + "]").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ---- Axis helpers ----
bool QSMLABaseChecker::HasAxis(QSMLAAxis axis, QSMLALayout layout, const gert::Shape &shape) const
{
    const auto &layoutIt = QSMLA_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == QSMLA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }
    const auto &axes = layoutIt->second;
    if (std::find(axes.begin(), axes.end(), axis) == axes.end()) {
        return false;
    }
    const auto &dimIt = QSMLA_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == QSMLA_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t QSMLABaseChecker::GetAxisIdx(QSMLAAxis axis, QSMLALayout layout) const
{
    const auto &axes = QSMLA_LAYOUT_AXIS_MAP.find(layout)->second;
    return std::distance(axes.begin(), std::find(axes.begin(), axes.end(), axis));
}

int64_t QSMLABaseChecker::GetAxisNum(const gert::Shape &shape, QSMLAAxis axis, QSMLALayout layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : INVALID_DIM_VALUE;
}

// ---- Logging helpers ----
std::string QSMLABaseChecker::DataTypeToSerialString(ge::DataType type) const
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    return (it != DATATYPE_TO_STRING_MAP.end()) ? it->second : "UNDEFINED";
}

std::string QSMLABaseChecker::LayoutToSerialString(QSMLALayout layout) const
{
    switch (layout) {
        case QSMLALayout::BSND:
            return "BSND";
        case QSMLALayout::TND:
            return "TND";
        case QSMLALayout::PA_BBND:
            return "PA_BBND";
        default:
            return "UNKNOWN";
    }
}

std::string QSMLABaseChecker::GetShapeStr(const gert::Shape &shape) const
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        oss << shape.GetDim(i);
        if (i < shape.GetDimNum() - 1)
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

void QSMLABaseChecker::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectList, ge::DataType actual,
                                            const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectList.size(); ++i) {
        oss << DataTypeToSerialString(expectList[i]);
        if (i < expectList.size() - 1)
            oss << ", ";
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QSMLA_OP_NAME, name.c_str(), DataTypeToSerialString(actual).c_str(),
                                          ("only support [" + oss.str() + "]").c_str());
}

void QSMLABaseChecker::LogErrorLayoutSupport(const std::vector<QSMLALayout> &expectList, QSMLALayout actual,
                                             const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectList.size(); ++i) {
        oss << LayoutToSerialString(expectList[i]);
        if (i < expectList.size() - 1)
            oss << ", ";
    }
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
        QSMLA_OP_NAME, name.c_str(),
        ("only support layout [" + oss.str() + "], but got " + LayoutToSerialString(actual)).c_str());
}

// explicit instantiation
template ge::graphStatus QSMLABaseChecker::CheckValueSupport(const int64_t value,
                                                             const std::vector<int64_t> &expectList,
                                                             const std::string &name) const;
template ge::graphStatus QSMLABaseChecker::CheckValueSupport(const uint32_t value,
                                                             const std::vector<uint32_t> &expectList,
                                                             const std::string &name) const;
template ge::graphStatus QSMLABaseChecker::CheckValueSupport(const int32_t value,
                                                             const std::vector<int32_t> &expectList,
                                                             const std::string &name) const;
template ge::graphStatus QSMLABaseChecker::CheckValueSupport(const uint64_t value,
                                                             const std::vector<uint64_t> &expectList,
                                                             const std::string &name) const;

} // namespace optiling
