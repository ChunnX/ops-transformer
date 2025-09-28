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
 * \file moe_init_routing_v3.cpp
 * \brief
 */
#include "moe_v3_mrgsort_out.h"
#include "moe_v3_mrgsort.h"
#include "moe_v3_sort_one_core.h"
#include "moe_v3_sort_multi_core.h"
#include "moe_v3_gather_sort_multi_core.h"
#include "moe_v3_expert_tokens_count.h"
#include "moe_v3_row_idx_gather.h"
#include "moe_v3_gather_out.h"
#include "moe_v3_gather_dynamic_quant.h"
#include "moe_v3_full_load.h"
#include "moe_v3_sort_actual_expert.h"
#include "moe_v3_sort_multi_core_performance.h"


/*
 * 高性能模板
 */
#define MOE_INIT_ROUTING_V3_PERFORMANCE 2000000

/*
 * 非量化、无Gather
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_GATHER 1000000  // 无Gather单核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER 1001000 // 无Gather单核排序、非量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER 1100000 // 无Gather多核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER 1101000 // 无Gather多核排序、非量化、SCATTER索引

/*
 * 动态量化、无Gather
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER 1020000 // 单核排序、动态量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER 1021000 // 单核排序、动态量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER 1120000 // 多核排序、动态量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER 1121000 // 多核排序、动态量化、GATHER索引

/*
 * 非量化、有Gather
 */
#define MOE_INIT_ROUTING_V3_GATHER_SORTONECORE_GATHER 1200000 // 单核Gather->单核或多核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_GATHER_SORTONECORE_SCATTER 1201000 // 单核Gather->单核或多核排序、非量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_GATHER 1300000 // 多核Gather->多核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_SCATTER 1301000 // 多核Gather->多核排序、非量化、SCATTER索引


using namespace AscendC;
using namespace MoeInitRoutingV3;
extern "C" __global__ __aicore__ void moe_init_routing_v3(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                                          GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                                          GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    auto t = &tilingData;

    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_PERFORMANCE)) {
        TPipe fullLoadPipe;
        MoeV3FullLoad op;
        op.Init(x, expertIdx, scale, offset, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, t,
                &fullLoadPipe);
        op.Process();
        fullLoadPipe.Destroy();
        return;
    }

    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTONECORE_SCATTER)) {
        TPipe sortActualExpertPipe;
        MoeSortActualExpert<DTYPE_X> op;
        bool isFinished = false;
        op.Init(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, userWS, t,
                &sortActualExpertPipe);
        isFinished = op.Process();
        sortActualExpertPipe.Destroy();
        if (isFinished) {
            return;
        }
    }

    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_SCATTER)) {
        TPipe gatherSortMultiCorePipe;
        MoeGatherSortMultiCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &gatherSortMultiCorePipe);
        op.Process();
        gatherSortMultiCorePipe.Destroy();
    }

    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_SCATTER)) {
        TPipe mergeSortMultiCorePipe;
        MoeSortMultiCorePerformance op;
        op.Init(expandedRowIdx, userWS, t, &mergeSortMultiCorePipe);
        op.Process();
        mergeSortMultiCorePipe.Destroy();
    }

    TPipe sortPipe;
    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER)) {
        // 单核排序
        MoeSortOneCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &sortPipe);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER)) {
        // 多核排序
        MoeSortMultiCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &sortPipe);
        op.Process();
    }
    sortPipe.Destroy();

    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_SCATTER)) {
        TPipe histogramPipe;
        ExpertTokensCount countOp;
        countOp.Init<true>(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
        countOp.Process();
        histogramPipe.Destroy();
    } else {
        TPipe histogramPipe;
        ExpertTokensCount countOp;
        countOp.Init<false>(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
        countOp.Process();
        histogramPipe.Destroy();
    }

    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER)) {
        // GATHER索引
        TPipe rowIdxPipe;
        RowIdxGather rowIdxGatherOp;
        rowIdxGatherOp.Init(expandedRowIdx, userWS, t, &rowIdxPipe);
        rowIdxGatherOp.Process();
        rowIdxPipe.Destroy();
    }

    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_GATHER_SORTMULTICORE_SCATTER)) {
        // 非量化
        TPipe gatherPipe;
        MoeGatherOut<DTYPE_X> gatherOp;
        gatherOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
        gatherOp.Process();
        gatherPipe.Destroy();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER)) {
        // 动态量化
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe gatherPipe;
            MoeGatherOutDynamicQuant<DTYPE_X> gatherDynamicQuantOp;
            gatherDynamicQuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherDynamicQuantOp.Process();
            gatherPipe.Destroy();
        }
    }
}