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
 * \file allto_allv_grouped_mat_mul_tiling_base.h
 * \brief
 */
#ifndef MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_H
#define MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/op_tiling/mc2_tiling_struct.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"

namespace optiling {

// 定义参数结构体
struct SetMMTilingParams {
    matmul_tiling::DataType matmulDtype;
    int32_t curMaxM;
    int32_t curMaxK;
    int32_t curMaxN;
    int32_t curBaseM;
    int32_t curBaseN;
    int32_t type;
};

struct MMTilingParams {
    int32_t curMaxM;
    int32_t curMaxK;
    int32_t curMaxN;
    int32_t* curBaseM;
    int32_t* curBaseK;
    int32_t* curBaseN;
};

class AlltoAllvGmmTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass
{
public:
    explicit AlltoAllvGmmTilingBase(gert::TilingContext* context) : Ops::Transformer::OpTiling::TilingBaseClass(context){};

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetPlatformInfo() override;

    NpuArch npuArch_;
};

class AlltoAllvGmmTilingCommon
{
public:
    virtual ge::graphStatus AlltoAllvGmmOpTilingFunc(gert::TilingContext* context) = 0;
};
} // namespace optiling

#endif // MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_H
