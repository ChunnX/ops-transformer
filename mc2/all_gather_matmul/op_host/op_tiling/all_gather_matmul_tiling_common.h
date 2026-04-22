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
 * \file all_gather_matmul_tiling_common.h
 * \brief
 */
#ifndef __ALL_GATHER_MATMUL_TILING_COMMON_H__
#define __ALL_GATHER_MATMUL_TILING_COMMON_H__

#include "vector"
#include "mc2_hcom_topo_info.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "all_gather_formulaic_tiling.h"
#include "mc2_log.h"
#include "ops_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../op_kernel/all_gather_matmul_tiling_key.h"
#include "../../op_kernel/all_gather_matmul_tiling.h"

namespace optiling {
    AllGatherPlusMM* GetTileFormulate(const mc2tiling::TilingArgs& args, uint32_t inputRankDim,
        KernelType inputKernelType, Mc2Tiling::AllGatherMatmulTilingData& tilingData);
    ge::graphStatus CheckValidRank(Mc2Tiling::AllGatherMatmulTilingData* tilingData,
        const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK, gert::TilingContext *context);
    void SetSocParam(Mc2Tiling::AllGatherMatmulTilingData* tilingData, const char* group);
    std::string GetAlgConfig(Mc2Tiling::AllGatherMatmulTilingData* tilingData);
}

#endif //__ALL_GATHER_MATMUL_TILING_COMMON_H__