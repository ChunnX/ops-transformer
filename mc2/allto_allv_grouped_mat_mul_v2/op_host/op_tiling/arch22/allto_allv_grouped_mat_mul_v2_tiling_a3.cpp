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
 * \file allto_allv_grouped_mat_mul_tiling_a3.cpp
 * \brief
 */

#include "allto_allv_grouped_mat_mul_tiling_a3.h"

namespace {
    static const char* A_INNER_DEBUG = "AlltoAllvGroupedMatMul Tiling";
}

namespace optiling {

    std::vector<int64_t> AlltoAllvGmmTilingA3::GetEpWorldSizeOptional() const
    {
        return {8, 16, 32, 64, 128}; //A3限制epWorldSize为{8，16，32，64, 128}
    }
    
    bool AlltoAllvGmmTilingA3::NeedToCheckCounts() const
    {
        return true;
    }

    ge::graphStatus AlltoAllvGmmTilingFuncA3::AlltoAllvGmmOpTilingFunc(gert::TilingContext* context)
    {
        AlltoAllvGmmTilingA3 tiling;
        OP_TILING_CHECK(
            tiling.Init(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM tiling init failed."),
            return ge::GRAPH_FAILED);
        return tiling.RunFusionKernelTiling(context);
    }

    ge::graphStatus AlltoAllvGmmTilingStructA3::DoOpTiling()
    {
        AlltoAllvGmmTilingFuncA3 funcA3;
        return funcA3.AlltoAllvGmmOpTilingFunc(context_);
    }

    REGISTER_OPS_TILING_TEMPLATE(AlltoAllvGroupedMatMul, AlltoAllvGmmTilingStructA3, 0);
}