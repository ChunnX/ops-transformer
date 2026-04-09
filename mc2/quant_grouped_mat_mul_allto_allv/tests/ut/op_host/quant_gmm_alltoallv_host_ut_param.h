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
 * \file quant_gmm_alltoallv_host_ut_param.h
 * \brief tiling ut
 */

#ifndef QUANT_GMM_ALLTOALLV_HOST_UT_PARAM_H
#define QUANT_GMM_ALLTOALLV_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace QuantGmmAlltoallvUT {

struct QuantGmmAlltoallvTilingUtParam {
    std::string case_name;
    gert::TilingContextPara::TensorDescription gmmX = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription gmmWeight = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription gmmXScale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription gmmWeightScale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription mmX = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription mmWeight = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription mmXScale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription mmWeightScale = TD_DEFAULT;
    std::vector<int64_t> sendCounts;
    std::vector<int64_t> recvCounts;
    gert::TilingContextPara::TensorDescription gmmY = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription mmY = TD_DEFAULT;
    int64_t gmmXQuantMode;
    int64_t gmmWeightQuantMode;
    int64_t mmXQuantMode;
    int64_t mmWeightQuantMode;
    int64_t groupSize;
    bool transGmmWeight;
    bool transMmWeight;
    int64_t worldSize;
    int64_t epWorldSize;
    ge::graphStatus expectResult;
    uint64_t expectTilingKey;

    explicit QuantGmmAlltoallvTilingUtParam(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "caseName");
        GetTensorGE(csvMap, "gmmXShape", "gmmXDtype", "gmmXFormat", this->gmmX);
        GetTensorGE(csvMap, "gmmWeightShape", "gmmWeightDtype", "gmmWeightFormat", this->gmmWeight);
        GetTensorGE(csvMap, "gmmXScaleShape", "gmmXScaleDtype", "gmmXScaleFormat", this->gmmXScale);
        GetTensorGE(csvMap, "gmmWeightScaleShape", "gmmWeightScaleDtype", "gmmWeightScaleFormat", this->gmmWeightScale);
        GetTensorGE(csvMap, "mmXShape", "mmXDtype", "mmXFormat", this->mmX);
        GetTensorGE(csvMap, "mmWeightShape", "mmWeightDtype", "mmWeightFormat", this->mmWeight);
        GetTensorGE(csvMap, "mmXScaleShape", "mmXScaleDtype", "mmXScaleFormat", this->mmXScale);
        GetTensorGE(csvMap, "mmWeightScaleShape", "mmWeightScaleDtype", "mmWeightScaleFormat", this->mmWeightScale);
        this->sendCounts = GetShapeArr(ReadMap(csvMap, "sendCounts"));
        this->recvCounts = GetShapeArr(ReadMap(csvMap, "recvCounts"));
        GetTensorGE(csvMap, "gmmYShape", "gmmYDtype", "gmmYFormat", this->gmmY);
        GetTensorGE(csvMap, "mmYShape", "mmYDtype", "mmYFormat", this->mmY);
        this->gmmXQuantMode = stoll(ReadMap(csvMap, "gmmXQuantMode", "0"));
        this->gmmWeightQuantMode = stoll(ReadMap(csvMap, "gmmWeightQuantMode", "0"));
        this->mmXQuantMode = stoll(ReadMap(csvMap, "mmXQuantMode", "0"));
        this->mmWeightQuantMode = stoll(ReadMap(csvMap, "mmWeightQuantMode", "0"));
        this->groupSize = stoll(ReadMap(csvMap, "groupSize", "0"));
        this->transGmmWeight = ReadMap(csvMap, "transGmmWeight", "false") == "true";
        this->transMmWeight = ReadMap(csvMap, "transMmWeight", "false") == "true";
        this->worldSize = stoll(ReadMap(csvMap, "worldSize", "2"));
        this->epWorldSize = stoll(ReadMap(csvMap, "epWorldSize", "2"));
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectStatus"));
        this->expectTilingKey = stoull(ReadMap(csvMap, "expectTilingKey", "0"));
    }
};

inline std::ostream& operator<<(std::ostream& os, const QuantGmmAlltoallvTilingUtParam& param)
{
    return os << param.case_name;
}

} // namespace QuantGmmAlltoallvUT

#endif // QUANT_GMM_ALLTOALLV_HOST_UT_PARAM_H
