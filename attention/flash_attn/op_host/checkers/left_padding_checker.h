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
 * \file left_padding_checker.h
 * \brief
 */

#ifndef LEFT_PADDING_CHECKER_H
#define LEFT_PADDING_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace faAttn{

class LeftPaddingChecker : public BaseChecker {
public:
    LeftPaddingChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~LeftPaddingChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckFeature(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) override;

private:
    // 公共校验函数
    ge::graphStatus CheckShapeSupport(const gert::Tensor *tensor, const std::vector<int64_t> &expectShapeList) const;
    ge::graphStatus CheckExistenceDesc(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureActualLen(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureLayout(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureAlibiPse(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeaturePageAttention(const FaTilingInfo &faInfo);

    // enableNonQuant 相关校验函数
    ge::graphStatus CheckShapeAndDim(const FaTilingInfo &faInfo);

private:
};

}  // namespace optiling
}
#endif  // LEFT_PADDING_CHECKER_H