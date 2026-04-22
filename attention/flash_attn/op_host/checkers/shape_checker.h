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
#include <memory>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"
#include "../fa_tiling_shape.h"

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

    void SetFaShapeCompare(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQueryShape(const FaTilingInfo &faInfo) const;
    ge::graphStatus CheckKVShape(const FaTilingInfo &faInfo) const;
    ge::graphStatus CheckAttnOutShape(const FaTilingInfo &faInfo) const;
    ge::graphStatus CheckLseOutShape(const FaTilingInfo &faInfo) const;
    ge::graphStatus CheckShapeConsistency(const FaTilingInfo &faInfo);

private:
    ge::graphStatus CheckPAKeyValue(const FaTilingInfo &faInfo);
    ge::graphStatus CheckPAKeyValueDim3(const FaTilingInfo &faInfo);
    ge::graphStatus CheckPAKeyValueDim4(const FaTilingInfo &faInfo);
    ge::graphStatus CheckPAKeyValueDim5(const FaTilingInfo &faInfo);
    ge::graphStatus CheckAxis(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQueryOutConsistency(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQueryOutConsistencySemantic(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQueryOutConsistencyIndex(const FaTilingInfo &faInfo);
    ge::graphStatus CheckNonQuantDataType(const FaTilingInfo &faInfo);
    ge::graphStatus CheckNonQuantHeadNum(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDtypeConsistency(const FaTilingInfo &faInfo);
    ge::graphStatus CheckSoftmaxScale(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDeterministic(const FaTilingInfo &faInfo);
    ge::graphStatus CheckShapeLayout(const FaTilingInfo &faInfo);
    ge::graphStatus CheckShapeLayoutQuery(const FaTilingInfo &faInfo);
    ge::graphStatus CheckShapeLayoutKV(const FaTilingInfo &faInfo);

    std::shared_ptr<FaTilingShapeCompare> queryShapeCmp_;
    std::shared_ptr<FaTilingShapeCompare> keyShapeCmp_;
    std::shared_ptr<FaTilingShapeCompare> valueShapeCmp_;
    std::shared_ptr<FaTilingShapeCompare> attnOutShapeCmp_;
    std::shared_ptr<FaTilingShapeCompare> lseOutShapeCmp_;
};

} // namespace optiling
#endif // SHAPE_CHECKER_H