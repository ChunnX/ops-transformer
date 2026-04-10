/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_HOST_UT_PARAM_H
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_HOST_UT_PARAM_H

#include "op_host_csv_case_loader.h"

namespace QuantGmmAlltoAllvUT {

struct QuantGmmAlltoAllvTilingUtParam {
    std::string case_name;

    std::vector<int64_t> gmmXShape;
    ge::DataType gmmXDtype;
    ge::Format gmmXFormat;

    std::vector<int64_t> gmmWeightShape;
    ge::DataType gmmWeightDtype;
    ge::Format gmmWeightFormat;

    std::vector<int64_t> gmmXScaleShape;
    ge::DataType gmmXScaleDtype;
    ge::Format gmmXScaleFormat;

    std::vector<int64_t> gmmWeightScaleShape;
    ge::DataType gmmWeightScaleDtype;
    ge::Format gmmWeightScaleFormat;

    std::vector<int64_t> mmXShape;
    ge::DataType mmXDtype;
    ge::Format mmXFormat;

    std::vector<int64_t> mmWeightShape;
    ge::DataType mmWeightDtype;
    ge::Format mmWeightFormat;

    std::vector<int64_t> mmXScaleShape;
    ge::DataType mmXScaleDtype;
    ge::Format mmXScaleFormat;

    std::vector<int64_t> mmWeightScaleShape;
    ge::DataType mmWeightScaleDtype;
    ge::Format mmWeightScaleFormat;

    std::vector<int64_t> sendCounts;
    std::vector<int64_t> recvCounts;

    std::vector<int64_t> gmmYShape;
    ge::DataType gmmYDtype;
    ge::Format gmmYFormat;

    std::vector<int64_t> mmYShape;
    ge::DataType mmYDtype;
    ge::Format mmYFormat;

    int64_t gmmXQuantMode;
    int64_t gmmWeightQuantMode;
    int64_t mmXQuantMode;
    int64_t mmWeightQuantMode;
    int64_t groupSize;

    bool transGmmWeight;
    bool transMmWeight;

    int64_t worldSize;
    int64_t epWorldSize;
    int64_t graphType;

    ge::graphStatus expectStatus;
    uint64_t expectTilingKey;

    explicit QuantGmmAlltoAllvTilingUtParam(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "caseName");

        this->gmmXShape = GetShapeArr(ReadMap(csvMap, "gmmXShape"));
        this->gmmXDtype = Str2DTypeGE(ReadMap(csvMap, "gmmXDtype"));
        this->gmmXFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "gmmXFormat"), ge::FORMAT_ND);

        this->gmmWeightShape = GetShapeArr(ReadMap(csvMap, "gmmWeightShape"));
        this->gmmWeightDtype = Str2DTypeGE(ReadMap(csvMap, "gmmWeightDtype"));
        this->gmmWeightFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "gmmWeightFormat"), ge::FORMAT_ND);

        this->gmmXScaleShape = GetShapeArr(ReadMap(csvMap, "gmmXScaleShape"));
        this->gmmXScaleDtype = Str2DTypeGE(ReadMap(csvMap, "gmmXScaleDtype"));
        this->gmmXScaleFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "gmmXScaleFormat"), ge::FORMAT_ND);

        this->gmmWeightScaleShape = GetShapeArr(ReadMap(csvMap, "gmmWeightScaleShape"));
        this->gmmWeightScaleDtype = Str2DTypeGE(ReadMap(csvMap, "gmmWeightScaleDtype"));
        this->gmmWeightScaleFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "gmmWeightScaleFormat"), ge::FORMAT_ND);

        this->mmXShape = GetShapeArr(ReadMap(csvMap, "mmXShape"));
        this->mmXDtype = Str2DTypeGE(ReadMap(csvMap, "mmXDtype"));
        this->mmXFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "mmXFormat"), ge::FORMAT_ND);

        this->mmWeightShape = GetShapeArr(ReadMap(csvMap, "mmWeightShape"));
        this->mmWeightDtype = Str2DTypeGE(ReadMap(csvMap, "mmWeightDtype"));
        this->mmWeightFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "mmWeightFormat"), ge::FORMAT_ND);

        this->mmXScaleShape = GetShapeArr(ReadMap(csvMap, "mmXScaleShape"));
        this->mmXScaleDtype = Str2DTypeGE(ReadMap(csvMap, "mmXScaleDtype"));
        this->mmXScaleFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "mmXScaleFormat"), ge::FORMAT_ND);

        this->mmWeightScaleShape = GetShapeArr(ReadMap(csvMap, "mmWeightScaleShape"));
        this->mmWeightScaleDtype = Str2DTypeGE(ReadMap(csvMap, "mmWeightScaleDtype"));
        this->mmWeightScaleFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "mmWeightScaleFormat"), ge::FORMAT_ND);

        this->sendCounts = GetShapeArr(ReadMap(csvMap, "sendCounts"));
        this->recvCounts = GetShapeArr(ReadMap(csvMap, "recvCounts"));

        this->gmmYShape = GetShapeArr(ReadMap(csvMap, "gmmYShape"));
        this->gmmYDtype = Str2DTypeGE(ReadMap(csvMap, "gmmYDtype"));
        this->gmmYFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "gmmYFormat"), ge::FORMAT_ND);

        this->mmYShape = GetShapeArr(ReadMap(csvMap, "mmYShape"));
        this->mmYDtype = Str2DTypeGE(ReadMap(csvMap, "mmYDtype"));
        this->mmYFormat = ReadMap(GE_FORMAT, ReadMap(csvMap, "mmYFormat"), ge::FORMAT_ND);

        this->gmmXQuantMode = stoll(ReadMap(csvMap, "gmmXQuantMode", "0"));
        this->gmmWeightQuantMode = stoll(ReadMap(csvMap, "gmmWeightQuantMode", "0"));
        this->mmXQuantMode = stoll(ReadMap(csvMap, "mmXQuantMode", "0"));
        this->mmWeightQuantMode = stoll(ReadMap(csvMap, "mmWeightQuantMode", "0"));
        this->groupSize = stoll(ReadMap(csvMap, "groupSize", "0"));

        this->transGmmWeight = ReadMap(csvMap, "transGmmWeight", "false") == "true";
        this->transMmWeight = ReadMap(csvMap, "transMmWeight", "false") == "true";

        this->worldSize = stoll(ReadMap(csvMap, "worldSize", "0"));
        this->epWorldSize = stoll(ReadMap(csvMap, "epWorldSize", "0"));
        this->graphType = stoll(ReadMap(csvMap, "graphType", "0"));

        this->expectStatus = Str2StatusGE(ReadMap(csvMap, "expectStatus"));
        this->expectTilingKey = stoull(ReadMap(csvMap, "expectTilingKey", "0"));
    }
};

inline std::ostream& operator<<(std::ostream& os, const QuantGmmAlltoAllvTilingUtParam& param)
{
    return os << param.case_name;
}

} // namespace QuantGmmAlltoAllvUT

#endif // QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_HOST_UT_PARAM_H
