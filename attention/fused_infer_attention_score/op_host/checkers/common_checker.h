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
 * \file common_checker.h
 * \brief
 */

#ifndef COMMON_CHECKER_H
#define COMMON_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

using std::map;
namespace optiling {
class CommonChecker : public BaseChecker {
public:
    CommonChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~CommonChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

private:
    ge::graphStatus CheckShapeConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKVStorageConsistency(const FiaTilingInfo &fiaInfo);
    // 公共校验函数
    ge::graphStatus CheckInputFormat(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckParaExistenceImpl(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDtypeCommon(const gert::CompileTimeTensorDesc *desc, const std::string &name,
                                     std::map<std::string, std::vector<ge::DataType>> dataMap);
    ge::graphStatus CheckPAKeyValue(const FiaTilingInfo &fiaInfo);
    bool CheckEmptyTensorList(const FiaTilingInfo &fiaInfo);
    bool CheckNormalTensorList(const FiaTilingInfo &fiaInfo);
    bool CheckNormalTensorListBSH(const FiaTilingInfo &fiaInfo);
    bool CheckNormalTensorListBNSD(const FiaTilingInfo &fiaInfo);
    bool CheckNormalTensorListBSND(const FiaTilingInfo &fiaInfo);
    bool CheckKeyRopeTensorListBSHConsistency(const FiaTilingInfo &fiaInfo, int64_t &keyRopeS, int64_t tmpIdx);
    bool CheckSingleKVPairBSH(const FiaTilingInfo &fiaInfo, int64_t tmpIdx, int64_t standardKH, int64_t standardVH);
    ge::graphStatus CheckTensorList(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckMultiDtype(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAxis(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAxisIfaMla(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAxisIsQKVDiff(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAxisTND(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQueryOutConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQueryOutDimsConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKeyValueConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKeyValueDimsConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckValueOutDConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQueryShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKeyNHVaild(const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape);
    ge::graphStatus CheckKeyDVaild(const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape);
    ge::graphStatus CheckKeyShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKVContiguous(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckKVContiguousBnBsH(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckKVContiguousBnNBsDNZ(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckQueryKeyConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQueryKeyTensorlistConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckMultiAttr(const FiaTilingInfo &fiaInfo);
    void GetQueryDimAndOutDim(const gert::StorageShape* queryShape, const gert::StorageShape* outShape,
        const std::string &layoutStr, int64_t &tmpQueryDim, int64_t &outDim, uint32_t i);

    // enableNonQuant 相关校验函数
    ge::graphStatus CheckNonQuantDataType(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAttr(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDimNum(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckHeadNum(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputLayout(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputLayoutGQA(const FiaTilingInfo &fiaInfo, const std::string &inputLayout);
    ge::graphStatus CheckInputLayoutDecodeMLA(const FiaTilingInfo &fiaInfo, const std::string &inputLayout);
    ge::graphStatus ValidateNoRopeLayoutDim(const FiaTilingInfo &fiaInfo, const std::string &inputLayout);
    ge::graphStatus CheckNoRopeBNSDBSNDLayoutDim(const FiaTilingInfo &fiaInfo, const std::string &inputLayout);
    ge::graphStatus CheckNoRopeLayoutDimC(const FiaTilingInfo &fiaInfo, const std::string &inputLayout);
    ge::graphStatus CheckNoRopeLayoutDimD(const FiaTilingInfo &fiaInfo, const std::string &inputLayout);
    ge::graphStatus CheckInnerPrecise(const FiaTilingInfo &fiaInfo);
    bool CheckTNDLayoutCrossover(const FiaTilingInfo &fiaInfo);
    bool CheckNTDLayoutCrossover(const FiaTilingInfo &fiaInfo);
    bool CheckTransposeLayoutCrossover(const FiaTilingInfo &fiaInfo);
    bool CheckTransposeLayoutBsNsdCrossover(const FiaTilingInfo &fiaInfo, const std::string &layoutStr, bool isGqa);
    bool CheckTransposeLayoutBnsdBsndCrossover(const FiaTilingInfo &fiaInfo, const std::string &layoutStr, bool isGqa);

private:
};

} // namespace optiling
#endif // SHAPE_CHECKER_H