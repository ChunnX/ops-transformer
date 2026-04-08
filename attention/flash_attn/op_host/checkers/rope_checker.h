/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rope_checker.h
 * \brief
 */

#ifndef ROPE_CHECKER_H
#define ROPE_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optilingForBuildPKG {
class RopeChecker : public BaseChecker {
public:
    RopeChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~RopeChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckFeature(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) override;

private:
    // 公共校验函数
    ge::graphStatus CheckQDsizeSupport(const FaTilingInfo &faInfo);
    ge::graphStatus CheckRopeDSizeSupport(const FaTilingInfo &faInfo);
    ge::graphStatus CheckRopeDtype(const FaTilingInfo &faInfo);
    ge::graphStatus CheckRopeDtypeConsistency(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQKAndQKRopeShapeConsistency(const FaTilingInfo &faInfo,
        const gert::Shape shape, const gert::Shape ropeShape, const std::string &inputName);
    ge::graphStatus CheckPAKeyAndKeyRopeShapeConsistency(const FaTilingInfo &faInfo,
        const gert::Shape &keyShape, const gert::Shape &keyRopeShape);
    ge::graphStatus CheckTensorlistKeyAndKeyRopeShapeConsistency(const FaTilingInfo &faInfo);
    ge::graphStatus CheckRopeExistence(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureDecodeMLA(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureSupport(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureAntiQuant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckShapeSupport(const FaTilingInfo &faInfo);
    ge::graphStatus CheckAxisSupport(const FaTilingInfo &faInfo);

private:
};

} // namespace optilingForBuildPKG
#endif // ROPE_CHECKER_H