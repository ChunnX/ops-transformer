/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_mat_mul_allto_allv_tiling_adapter.h
 * \brief
 */

#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_TILING_ADAPTER_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_TILING_ADAPTER_H

#pragma once
#include "securec.h"
#include "mc2_matmul_tiling_cfg.h"
#include "grouped_mat_mul_allto_allv_tiling_common.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "../../../../3rd/grouped_matmul/op_tiling/gmm_qbmm_tiling.h"
#include "../../../../3rd/grouped_matmul/op_tiling/grouped_matmul_host_util.h"
#include "../../../../3rd/grouped_matmul/op_tiling/grouped_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_mat_mul_allto_allv_tiling.h"
#include "register/tilingdata_base.h"

namespace optiling {

using namespace Mc2GroupedMatmulTiling::GmmConstant;
using Mc2GroupedMatmulTiling::Mc2GroupedQbmmTiling;
using Mc2GroupedMatmulTiling::QuantMode;

namespace Mc2GroupedMatmul {

class GroupedMatmulAllToAllvAdapter : public Mc2GroupedQbmmTiling {
public:
    explicit GroupedMatmulAllToAllvAdapter(gert::TilingContext *context) : Mc2GroupedQbmmTiling(context) {};

    ~GroupedMatmulAllToAllvAdapter() override = default;

    const Mc2GroupedMatmulTilingData::GMMQuantTilingData &GetGmmQuantTilingAdapterData() const
    {
        return tilingData_;
    }

    bool AnalyzeAttrs() override
    {
        return true;
    }
    bool AnalyzeDtype() override
    {
        return true;
    }
    bool AnalyzeInputs() override
    {
        return true;
    }
    void Reset() override
    {
    }

    ge::graphStatus Process();
    ge::graphStatus SetCommonInputParams(const GmmAlltoAllvParamsInfo &params);
    ge::graphStatus SetGroupExpertInputParameters(const GmmAlltoAllvParamsInfo &params, uint64_t gmmX);
    ge::graphStatus SetSharedExpertInputParameters(const GmmAlltoAllvParamsInfo &params);
};

} // namespace Mc2GroupedMatmul
} // namespace optiling

#endif