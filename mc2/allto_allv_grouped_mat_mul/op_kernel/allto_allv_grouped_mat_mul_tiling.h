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
 * \file allto_allv_grouped_mat_mul_tiling.h
 * \brief
 */
#ifndef __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__
#define __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"

constexpr uint32_t A2AVGMM_MAX_COUNT_NUM = 128U;

#pragma pack(push, 8)
struct AlltoAllvGmmInfo {
    uint32_t M = 0U;
    uint32_t K = 0U;
    uint32_t N = 0U;
    uint32_t rankSize = 0U;
    uint32_t expertPerRank = 0U;
    uint32_t maxOutputSize = 0U;
    uint32_t isTransposeB = 0U;
    uint32_t hasSharedExpert = 0U;
    uint32_t hasPermuteOut = 0U;
    uint32_t aivCoreNum = 0U;
    uint32_t aicCoreNum = 0U;
    uint32_t totalUbSize = 0U;
};

struct AlltoAllvGmmCoCTiling {
    uint32_t m0 = 0U;
    uint32_t k0 = 0U;
    uint32_t n0 = 0U;
    uint32_t ubMoveNum = 0U;
    uint32_t swizzlCount = 0U;
    uint32_t swizzlDirect = 0U;
};

struct AlltoAllvGmmExpertMeta {
    uint64_t recvTokenBase = 0U;
    uint32_t tokenCount = 0U;
    uint32_t reserved = 0U;
};

struct AlltoAllvGmmExpertSourceMeta {
    uint64_t dstTokenOffset = 0U;
    uint64_t srcTokenOffset = 0U;
    uint32_t tokenCount = 0U;
    uint32_t sourceRank = 0U;
};

struct AlltoAllvGmmAivTilingData {
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling hcclCcTiling;
    AlltoAllvGmmInfo gmmInfo;
    AlltoAllvGmmInfo mmInfo;
    AlltoAllvGmmCoCTiling gmmCocTiling;
    AlltoAllvGmmCoCTiling mmCocTiling;
    int64_t sendCounts[A2AVGMM_MAX_COUNT_NUM] = {};
    int64_t recvCounts[A2AVGMM_MAX_COUNT_NUM] = {};
    uint64_t recvTokenOffset = 0U;
    uint64_t countTableOffset = 0U;
    uint64_t expertMetaOffset = 0U;
    uint64_t expertReadyOffset = 0U;
    uint64_t userWorkspaceSize = 0U;
    uint32_t countNum = 0U;
    uint32_t reserved = 0U;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct AlltoAllvGmmTilingData {
    MC2KernelTemplate::HcclA2avTilingInfo hcclA2avTilingInfo;
    MC2KernelTemplate::TaskTilingInfo taskTilingInfo;
    bool isPermuteOut = false;
    bool isNeedMM = false;
    Mc2GroupedMatmulTilingData::GMMQuantTilingData gmmQuantTilingData;
    Mc2GroupedMatmulTilingData::GMMQuantTilingData mmQuantTilingData;
};
#pragma pack(pop)

#endif // __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__
