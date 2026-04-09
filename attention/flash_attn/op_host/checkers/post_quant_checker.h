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
 * \file post_quant_checker.h
 * \brief
 */

#ifndef POST_QUANT_CHECKER_H
#define POST_QUANT_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace faAttn{

class PostQuantChecker : public BaseChecker {
public:
    PostQuantChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~PostQuantChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckFeature(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) override;

private:
    // 公共校验函数
    ge::graphStatus CheckSingleDtype(const FaTilingInfo &faInfo);
    ge::graphStatus CheckExistenceQuantScale2(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureQueryDType(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureOutputEqual(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeaturePrefix(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureRowValid(const FaTilingInfo &faInfo);
    ge::graphStatus CheckMultiParaQuantOffset2(const FaTilingInfo &faInfo);
    ge::graphStatus CheckMultiParaShape(const FaTilingInfo &faInfo);

private:
};

} // namespace optiling
}
#endif // POST_QUANT_CHECKER_H