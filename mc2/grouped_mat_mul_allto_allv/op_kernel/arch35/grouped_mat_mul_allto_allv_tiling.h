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
 * \file grouped_mat_mul_allto_allv_tiling.h
 * \brief Grouped MatMul AlltoAllV TilingData 定义
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_TILING_ARCH35_H__
#define GROUPED_MAT_MUL_ALLTO_ALLV_TILING_ARCH35_H__

#if __has_include("../../allto_allv_quant_grouped_mat_mul/mc2_templates/common/a2av_common_tiling.h")
#include "../../allto_allv_quant_grouped_mat_mul/mc2_templates/common/a2av_common_tiling.h"
#else
#include "../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"
#endif

#pragma once

using MC2KernelTemplate::GMMQuantTilingData;
using MC2KernelTemplate::TaskTilingInfo;

struct GmmA2avWorkspaceInfo {
    uint64_t wsGmmOutputSize;
    uint64_t wsGmmComputeWorkspaceSize;
    uint64_t wsSharedGmmComputeWorkspaceSize;
};

struct GmmA2avTilingData {
    MC2KernelTemplate::HcclA2avTilingInfo hcclA2avTiling;

    GmmA2avWorkspaceInfo workspaceInfo;

    TaskTilingInfo taskTilingInfo;

    GMMQuantTilingData sharedGmmTiling;

    GMMQuantTilingData gmmBaseTiling;

    bool isPermuteOut = false;
};

#endif