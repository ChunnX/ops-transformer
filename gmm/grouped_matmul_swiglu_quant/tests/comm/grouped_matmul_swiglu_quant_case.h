/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_case.h
 * \brief GroupedMatmulSwigluQuant 测试用例.
 */

#pragma once
#include <vector>
#include <cstdint>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>

namespace ops::adv::tests::grouped_matmul_swiglu_quant {
class GmmSwigluQuantCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

public:

    class Param {
    public:
        int64_t e = 0;
        int64_t m = 0;
        int64_t k = 0;
        int64_t n = 0;
        ge::DataType weightScaleDataType = ge::DataType::DT_FLOAT;
        ge::DataType xScaleDataType = ge::DataType::DT_FLOAT;
        Param();
        Param(int64_t e, int64_t m, int64_t k, int64_t n, ge::DataType weightScaleDataType);
    };

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *actualSeqLengthsTensor = nullptr;
    };

    Tensor x, weight, weightScale, xScale, groupList, y, yScale;
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc GmmSwigluQuantTilingFunc = nullptr;
    GmmSwigluQuantCase();
    GmmSwigluQuantCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam& tilingParam);
};

} // namespace ops::adv::tests::ifa
