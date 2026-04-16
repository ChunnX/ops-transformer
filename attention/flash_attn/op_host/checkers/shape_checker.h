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
class ShapeChecker : public BaseChecker {
public:
    ShapeChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~ShapeChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckFeature(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) override;

private:
    // 公共校验函数
    ge::graphStatus CheckInputFormat(const FaTilingInfo &faInfo);
    ge::graphStatus CheckParaExistenceImpl(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDtypeCommon(const gert::CompileTimeTensorDesc *desc, const std::string &name,
                                     std::map<std::string, std::vector<ge::DataType>> dataMap);
    ge::graphStatus CheckPAKeyValue(const FaTilingInfo &faInfo);
    bool CheckEmptyTensorList(const FaTilingInfo &faInfo);
    bool CheckNormalTensorList(const FaTilingInfo &faInfo);
    ge::graphStatus CheckTensorList(const FaTilingInfo &faInfo);
    ge::graphStatus CheckMultiDtype(const FaTilingInfo &faInfo);
    ge::graphStatus CheckAxis(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQueryOutConsistency(const FaTilingInfo &faInfo);
    ge::graphStatus CheckKeyValueConsistency(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQueryShape(const FaTilingInfo &faInfo);
    ge::graphStatus CheckKeyNHVaild(const FaTilingInfo &faInfo, const gert::Shape &keyShape);
    ge::graphStatus CheckKeyDVaild(const FaTilingInfo &faInfo, const gert::Shape &keyShape);
    ge::graphStatus CheckKeyShape(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQueryKeyConsistency(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQueryKeyTensorlistConsistency(const FaTilingInfo &faInfo);
    ge::graphStatus CheckMultiAttr(const FaTilingInfo &faInfo);
    void GetQueryDimAndOutDim(const gert::StorageShape* queryShape, const gert::StorageShape* outShape,
        const std::string &layoutStr, int64_t &tmpQueryDim, int64_t &outDim, uint32_t i);

    // enableNonQuant 相关校验函数
    ge::graphStatus CheckNonQuantDataType(const FaTilingInfo &faInfo);
    ge::graphStatus CheckNonQuantAttr(const FaTilingInfo &faInfo);
    ge::graphStatus CheckNonQuantHeadNum(const FaTilingInfo &faInfo);
    ge::graphStatus CheckNonQuantInputLayout(const FaTilingInfo &faInfo);
    ge::graphStatus CheckNonQuantInnerPrecise(const FaTilingInfo &faInfo);
    bool CheckTNDLayoutCrossover(const FaTilingInfo &faInfo);
    bool CheckNTDLayoutCrossover(const FaTilingInfo &faInfo);
    bool CheckTransposeLayoutCrossover(const FaTilingInfo &faInfo);

private:
};

} // namespace optiling
#endif // SHAPE_CHECKER_H