/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_QUANT_MATMUL_REDUCE_SCATTER_HOST_UT_PARAM_H
#define TEST_QUANT_MATMUL_REDUCE_SCATTER_HOST_UT_PARAM_H

#include <sstream>
#include "quant_matmul_reduce_scatter_host_ut_param.h"

namespace QuantMatmulReduceScatterUT {

struct QuantMatmulReduceScatterInferShapeUtParam : public QuantMatmulReduceScatterHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription context = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x1Scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2Scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::vector<std::vector<int64_t>> expectOutputShape;

    explicit QuantMatmulReduceScatterInferShapeUtParam(const csv_map &csvMap)
        : QuantMatmulReduceScatterHostUtParamBase(csvMap)
    {
        // def 输入顺序: context(0) x1(1) x2(2) x1_scale(3) x2_scale(4)，context 恒实例化
        this->context = gert::InfershapeContextPara::TensorDescription(gert::StorageShape({100}, {100}), ge::DT_INT32,
                                                                       ge::FORMAT_ND);
        this->inputInstance.emplace_back(1);
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", this->x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", this->x2));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format", this->x1Scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format", this->x2Scale));
        this->y = gert::InfershapeContextPara::TensorDescription({}, Str2DTypeGE(ReadMap(csvMap, "y_dtype", "BF16")),
                                                                 ge::FORMAT_ND);
        // def 输出顺序: y(0)，单输出
        this->outputInstance = {1};
        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape.emplace_back(GetShapeArr(ReadMap(csvMap, "expect_y_shape")));
        }
    }
};

struct QuantMatmulReduceScatterInferDataTypeUtParam : public QuantMatmulReduceScatterHostUtParamBase {
    ge::DataType x1 = ge::DT_UNDEFINED;
    ge::DataType x2 = ge::DT_UNDEFINED;
    ge::DataType x1Scale = ge::DT_UNDEFINED;
    ge::DataType x2Scale = ge::DT_UNDEFINED;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    ge::DataType expectYDtype = ge::DT_UNDEFINED;

    explicit QuantMatmulReduceScatterInferDataTypeUtParam(const csv_map &csvMap)
        : QuantMatmulReduceScatterHostUtParamBase(csvMap)
    {
        GetDataTypeGE(csvMap, "x1_dtype", this->x1);
        GetDataTypeGE(csvMap, "x2_dtype", this->x2);
        GetDataTypeGE(csvMap, "x1_scale_dtype", this->x1Scale);
        GetDataTypeGE(csvMap, "x2_scale_dtype", this->x2Scale);
        // def 输入顺序: context(0) x1(1) x2(2) x1_scale(3) x2_scale(4)
        this->inputInstance = {1, 1, 1, 1, 1};
        // def 输出顺序: y(0)，单输出
        this->outputInstance = {1};
        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectYDtype = Str2DTypeGE(ReadMap(csvMap, "expect_y_dtype"));
        }
    }
};

} // namespace QuantMatmulReduceScatterUT

#endif // TEST_QUANT_MATMUL_REDUCE_SCATTER_HOST_UT_PARAM_H
