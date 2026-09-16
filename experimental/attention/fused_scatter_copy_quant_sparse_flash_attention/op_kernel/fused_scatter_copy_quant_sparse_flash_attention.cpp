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
 * \file fused_scatter_copy_quant_sparse_flash_attention.cpp
 * \brief
 */

#ifndef FUSED_SCATTER_COPY_QUANT_SPARSE_FLASH_ATTENTION_PREFETCH
#define FUSED_SCATTER_COPY_QUANT_SPARSE_FLASH_ATTENTION_PREFETCH 1
#endif

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fused_scatter_copy_quant_sparse_flash_attention_template_tiling_key.h"
#include "fused_scatter_copy_quant_sparse_flash_attention_copy.h"

using FusedScatterCopyQuantSparseFlashAttentionTilingDataMla = FusedScatterCopyQuantSparseFlashAttentionTilingData;
#include "arch35/fused_scatter_copy_quant_sparse_flash_attention_kernel_mla.h"

template<bool SPLIT_HEADS>
__global__ __aicore__ void fused_scatter_copy_quant_sparse_flash_attention(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
    GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR topkDstSlots,
    GM_ADDR topkSrcIds, GM_ADDR topkMissCounts,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR residentSeqLengths,
    GM_ADDR copySrcIds, GM_ADDR copyDstSlots, GM_ADDR copyCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)hbmKvAlias;
    GET_TILING_DATA_WITH_STRUCT(FusedScatterCopyQuantSparseFlashAttentionTilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const bool fallback = fused_scatter_copy_qsfa::RequiresUnionCopy(copyCounts, topkMissCounts,
        actualSeqLengthsQuery, tilingData);
    if (fallback) {
        fused_scatter_copy_qsfa::UnionCopy(pipe, hbmKvBytes, dramKvBytes, hbmBlockTable,
            dramBlockTable, copySrcIds, copyDstSlots, copyCounts, tilingData);
    }
    using QueryType = typename std::conditional<ORIG_DTYPE_QUERY == DT_BF16, bfloat16_t, half>::type;
    using Cube = typename std::conditional<g_coreType == AscendC::AIC,
        BaseApi::QSFAMatmulService<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>,
        BaseApi::QSFAMatmulServiceDummy<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>>::type;
    using Vector = typename std::conditional<g_coreType == AscendC::AIC,
        BaseApi::QSFAVectorServiceDummy<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>,
        BaseApi::QSFAVectorService<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>>::type;
    BaseApi::FusedScatterCopyQuantSparseFlashAttentionMla<Cube, Vector> op;
    op.Init(query, hbmKvBytes, hbmKvBytes, topkDstSlots, nullptr, nullptr, hbmBlockTable,
        actualSeqLengthsQuery, residentSeqLengths, attentionOut, userWorkspace, &tilingData, &pipe);
    op.InitSourceAwareGather(dramKvBytes, dramBlockTable, topkDstSlots, topkSrcIds, topkMissCounts, &tilingData,
        !fallback, userWorkspace + static_cast<uint64_t>(tilingData.singleCoreParams.usedCoreNum) * 3 * 128 * 576 * 2);
    op.Process();
}
