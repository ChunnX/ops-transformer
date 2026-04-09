/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pse_checker.h
 * \brief
 */

#ifndef PSE_CHECKER_H
#define PSE_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace faAttn{

class PSEChecker : public BaseChecker {
public:
    PSEChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~PSEChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckFeature(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) override;

private:
    // singlepara
    ge::graphStatus CheckPseType(const FaTilingInfo &faInfo);
    ge::graphStatus CheckPseShiftDataType(const FaTilingInfo &faInfo);
    ge::graphStatus CheckPseShiftShape(const FaTilingInfo &faInfo);
    // existence
    ge::graphStatus CheckPseShiftExistence(const FaTilingInfo &faInfo);
    // feature
    ge::graphStatus CheckFeaturePA(const FaTilingInfo &faInfo);
    ge::graphStatus CheckerFeatureCrossover(const FaTilingInfo &faInfo);
    // multipara
    ge::graphStatus CheckAlibiStartIdx(const FaTilingInfo &faInfo);

private:
};

} // namespace optiling
}
#endif // PSE_CHECKER_H