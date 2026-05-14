/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the the software repository for the full text of the License.
 */

/*!
 * \file all_gather_matmul_int8_tiling.h
 * \brief INT8 Tiling for A5 (950PR) with A5-specific optimized parameters
 */

#ifndef __ALL_GATHER_MATMUL_INT8_TILING_H__
#define __ALL_GATHER_MATMUL_INT8_TILING_H__

#pragma once

#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../../op_kernel/arch35/all_gather_matmul_tiling_arch35.h"
#include "../../../op_kernel/all_gather_matmul_v2_apt_tiling_key.h"

using Ops::Transformer::OpTiling::TilingBaseClass;

namespace optiling {

class AllGatherMatmulInt8Tiling : public TilingBaseClass {
public:
    explicit AllGatherMatmulInt8Tiling(gert::TilingContext *context);
    ~AllGatherMatmulInt8Tiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus CheckInput();
    ge::graphStatus SetTilingData();
    ge::graphStatus SetMc2Hcomm();
    void PrintTilingData();

    Mc2Tiling::AllGatherMatmulTilingDataInt8 *tilingData_;
    const char *opName_;
    const char *group_;
    int64_t rankSize_ = 0;
    uint32_t M_ = 0;
    uint32_t K_ = 0;
    uint32_t N_ = 0;
    uint32_t totalUbSize_ = 0;
    uint32_t aivNum_ = 0;
    uint32_t numBlocks_ = 0;
    bool isTransposeX1_ = false;
    bool isTransposeX2_ = false;
    bool isX2ScaleTypeInt64_ = false;
    int32_t dequantType_ = 0;
    ge::DataType x1Type_ = ge::DT_UNDEFINED;
    ge::DataType x2Type_ = ge::DT_UNDEFINED;
    ge::DataType yType_ = ge::DT_UNDEFINED;
    uint32_t elementSize_ = 0;
    uint32_t nElemAlign_ = 0;
};

} // namespace optiling

#endif // __ALL_GATHER_MATMUL_INT8_TILING_H__