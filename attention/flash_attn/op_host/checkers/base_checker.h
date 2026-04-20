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
 * \brief Base checker class for npu_flash_attn parameters
 */

#ifndef BASE_CHECKER_H
#define BASE_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"

#include "../fa_tiling_info.h"
#include "../fa_tiling_shape.h"
#include "../flash_attn_tiling_utils.h"

namespace optiling {
class FABaseChecker {
public:
    FABaseChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
                enableNonQuant_(enableNonQuant),
                enableFullQuant_(enableFullQuant),
                enableAntiQuant_(enableAntiQuant) {}
    virtual ~FABaseChecker() = default;

    virtual ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) { return ge::GRAPH_SUCCESS; }
    virtual ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) { return ge::GRAPH_SUCCESS; }
    virtual ge::graphStatus CheckFeature(const FaTilingInfo &faInfo) { return ge::GRAPH_SUCCESS; }
    virtual ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) { return ge::GRAPH_SUCCESS; }

    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckFormatSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    template <typename T>
    ge::graphStatus CheckValueSupport(const T value, const std::vector<T> &expectValList) const;

    std::string DataTypeToSerialString(ge::DataType type);

protected:
    bool enableNonQuant_ = false;
    bool enableFullQuant_ = false;
    bool enableAntiQuant_ = false;
};
} // namespace optiling
#endif // BASE_CHECKER_H