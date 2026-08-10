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
 * \file base_checker.h
 * \brief Base checker class providing common tensor/attribute validation utilities
 *        shared by all QSMLA sub-checkers. Each sub-checker inherits from QSMLABaseChecker
 *        and implements four virtual methods: CheckSinglePara, CheckParaExistence,
 *        CheckCrossFeature, CheckMultiParaConsistency.
 */

#ifndef QSMLA_BASE_CHECKER_H
#define QSMLA_BASE_CHECKER_H

#include <numeric>
#include <string>
#include <vector>
#include "tiling/tiling_api.h"
#include "qsmla_tiling_common.h"
#include "qsmla_tiling_info.h"
#include "qsmla_check_constants.h"

namespace optiling {

class QSMLABaseChecker {
public:
    QSMLABaseChecker() = default;
    virtual ~QSMLABaseChecker() = default;

    // Four canonical check phases every sub-checker must implement
    virtual ge::graphStatus CheckSinglePara(const QSMLATilingInfo &info) const = 0;
    virtual ge::graphStatus CheckParaExistence(const QSMLATilingInfo &info) const = 0;
    virtual ge::graphStatus CheckCrossFeature(const QSMLATilingInfo &info) const = 0;
    virtual ge::graphStatus CheckMultiParaConsistency(const QSMLATilingInfo &info) const = 0;

protected:
    // ---- Tensor validation utilities (shape / dtype / format only, no value checks) ----

    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckFormatSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape, const std::vector<size_t> &expectList,
                                       const std::string &name) const;
    ge::graphStatus CheckDimNumInLayoutSupport(QSMLALayout layout, const gert::StorageShape *shape,
                                               const std::string &name) const;
    ge::graphStatus CheckShapeSizeNonZero(const gert::Tensor *tensor, const std::string &name) const;
    ge::graphStatus CheckShapeSizeNonZero(const gert::StorageShape *shape, const std::string &name) const;

    // ---- Attribute validation utilities (value checks) ----
    template <typename T>
    ge::graphStatus CheckValueSupport(const T value, const std::vector<T> &expectList, const std::string &name) const;

    // ---- Axis helpers ----
    bool HasAxis(QSMLAAxis axis, QSMLALayout layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(QSMLAAxis axis, QSMLALayout layout) const;
    int64_t GetAxisNum(const gert::Shape &shape, QSMLAAxis axis, QSMLALayout layout) const;

    // ---- Logging helpers ----
    std::string DataTypeToSerialString(ge::DataType type) const;
    std::string LayoutToSerialString(QSMLALayout layout) const;
    std::string GetShapeStr(const gert::Shape &shape) const;
    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectList, ge::DataType actual,
                              const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<QSMLALayout> &expectList, QSMLALayout actual,
                               const std::string &name) const;

    static constexpr int64_t INVALID_DIM_VALUE = std::numeric_limits<int64_t>::min();
};

} // namespace optiling
#endif // QSMLA_BASE_CHECKER_H
