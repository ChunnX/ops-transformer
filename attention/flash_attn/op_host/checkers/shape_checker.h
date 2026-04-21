/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file shape_checker.h
 * \brief
 */

#ifndef SHAPE_CHECKER_H
#define SHAPE_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

using std::map;
namespace optiling {
class ShapeChecker : public FABaseChecker {
public:
    ShapeChecker() = default;
    ~ShapeChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckFeature(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) override;

private:
    ge::graphStatus CheckPAKeyValue(const FaTilingInfo &faInfo);
    ge::graphStatus CheckAxis(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQueryOutConsistency(const FaTilingInfo &faInfo);
    ge::graphStatus CheckNonQuantDataType(const FaTilingInfo &faInfo);
    ge::graphStatus CheckNonQuantHeadNum(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDtypeCommon(const gert::CompileTimeTensorDesc *desc, const std::string &name,
                                     std::map<std::string, std::vector<ge::DataType>> dataMap);

private:
};

} // namespace optiling
#endif // SHAPE_CHECKER_H