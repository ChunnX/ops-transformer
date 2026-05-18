/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_tiling_key_utils.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_KEY_UTILS_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_KEY_UTILS_H

#include "grouped_matmul_tiling_utils.h"

namespace optiling {
struct GMMTplTilingKeyParams {
    uint32_t xDType = GMM_TPL_INVALID;
    uint32_t weightDType = GMM_TPL_INVALID;
    uint32_t yDType = GMM_TPL_INVALID;
    uint32_t transposeX = 0U;
    uint32_t transposeWeight = 0U;
    uint32_t groupListType = 0U;
    uint32_t isStaticTilingApi = 0U;
    uint32_t a8w4KernelTemplate = GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE;
    uint32_t a16w8KernelTemplate = GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE;
    uint32_t aivAicRatio = GROUPED_MATMUL_AIV_AIC_RATIO_1;
    uint32_t isEnableFixedAxis = 0U;
};

static inline uint64_t MakeGMMTplTilingKey(const GMMTplTilingKeyParams& params)
{
    return GET_TPL_TILING_KEY(params.xDType, params.weightDType, params.yDType, params.transposeX,
                              params.transposeWeight, params.groupListType, params.isStaticTilingApi,
                              params.a8w4KernelTemplate, params.a16w8KernelTemplate, params.aivAicRatio,
                              params.isEnableFixedAxis);
}

static inline uint64_t MakeA8W4TplTilingKey(uint32_t yDType, uint32_t a8w4KernelTemplate, uint32_t aivAicRatio)
{
    return GET_TPL_TILING_KEY(GMM_TPL_INT8, GMM_TPL_INT4, yDType, 0, 0, GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, 0,
                              a8w4KernelTemplate, GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, aivAicRatio, 0);
}

static inline uint64_t MakeA16W4MsdTplTilingKey(uint32_t groupListType)
{
    return GET_TPL_TILING_KEY(GMM_TPL_BF16, GMM_TPL_INT4, GMM_TPL_BF16, 0, 1, groupListType, 0,
                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE,
                              GROUPED_MATMUL_A16W4_KERNEL_TEMPLATE_MSD_ANTIQUANT_GS32,
                              GROUPED_MATMUL_AIV_AIC_RATIO_2, 0);
}
}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_KEY_UTILS_H
