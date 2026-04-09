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
 * \file dequant_checker.h
 * \brief
 */

#ifndef DEQUANT_CHECKER_H
#define DEQUANT_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
class DequantChecker : public BaseChecker {
public:
    DequantChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~DequantChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckFeature(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) override;

private:
    // enableNonQuant 相关校验函数
    ge::graphStatus CheckExistenceNoquant(const FaTilingInfo &faInfo);

    // enableFullQuant 相关校验函数
    ge::graphStatus CheckDataTypeFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantScaleDtypeMLAFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantScaleDtypeGQAPerblock(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantScaleDtypeGQAPertensor(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantScaleDtypeFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantModeMLAFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantModeGQAPerblock(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantModeGQAPertensor(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantModeFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantScaleDimNumGQAPerblock(const FaTilingInfo &faInfo);

    ge::graphStatus CheckTensorExistFullquant(const FaTilingInfo &faInfo, const gert::Tensor *tensor,
        const std::string &quantModeName, const std::string &inputName);
    ge::graphStatus CheckTensorNotExistFullquant(const FaTilingInfo &faInfo, const gert::Tensor *tensor,
        const std::string &quantModeName, const std::string &inputName);
    ge::graphStatus CheckExistencePertensorFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckExistenceMLAFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckExistencePerblockFullquant(const FaTilingInfo &faInfo);

    ge::graphStatus CheckFeaturePertensorFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeaturePerblockFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureMLAFullquant(const FaTilingInfo &faInfo);

    ge::graphStatus CheckDequantScaleShapeMLAFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantScaleShapePertensor(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantScaleShapePerblock(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDequantScaleShapeFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckInputDTypeFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckInputLayoutPerblock(const FaTilingInfo &faInfo);
    ge::graphStatus CheckInputLayoutPertensor(const FaTilingInfo &faInfo);
    ge::graphStatus CheckInputLayoutMLAFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckInputLayoutFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckInputAxisFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckN1SizeFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckN2SizeFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQSSizeFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckGSizeFullquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDSizeFullquant(const FaTilingInfo &faInfo);

    // enableAntiQuant 相关校验函数
    // SinglePara
    ge::graphStatus CheckSingleParaForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckAntiquantModeForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckInputKVTypeForAntiquant(const FaTilingInfo &faInfo);
    
    // Existence
    ge::graphStatus CheckExistenceForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckScaleExistenceForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDescExistenceForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckOffsetExistenceForAntiquant(const FaTilingInfo &faInfo);

    // Feature
    ge::graphStatus CheckFeatureForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureExtendForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeaturePAForAntiquant(const FaTilingInfo &faInfo);

    // MultiPara
    ge::graphStatus CheckMultiParaForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckScaleTypeForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckScaleShapeForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckOffsetTypeForAntiquant(const FaTilingInfo &faInfo);
    ge::graphStatus CheckOffsetShapeForAntiquant(const FaTilingInfo &faInfo);

    ge::graphStatus CheckScaleShapeForPerChannelPerTensorMode(const FaTilingInfo &faInfo);
    ge::graphStatus CheckScaleShapeForPerTokenMode(const FaTilingInfo &faInfo);
    ge::graphStatus CheckKScaleShapeForPerChannelPerTensorMode(const FaTilingInfo &faInfo);
    ge::graphStatus CheckKScaleShapeForPerTokenMode(const FaTilingInfo &faInfo);
    ge::graphStatus CheckKScaleShapeForPerTensorHeadMode(const FaTilingInfo &faInfo);
    ge::graphStatus CheckKScaleShapeForPerTokenHeadMode(const FaTilingInfo &faInfo);
    ge::graphStatus CheckKScaleShapeForPerTokenPAMode(const FaTilingInfo &faInfo);
    ge::graphStatus CheckKScaleShapeForPerTokenHeadPAMode(const FaTilingInfo &faInfo);
    ge::graphStatus CheckKScaleShapeForPerTokenGroupMode(const FaTilingInfo &faInfo);
    ge::graphStatus CheckVScaleShapeForPerTokenMode(const FaTilingInfo &faInfo);

private:
    bool enablePerblockQuant_ = false;
    bool enablePertensorQuant_ = false;
    bool enableIFAMLAFullQuant_ = false;
};

} // namespace optiling
#endif // DEQUANT_CHECKER_H