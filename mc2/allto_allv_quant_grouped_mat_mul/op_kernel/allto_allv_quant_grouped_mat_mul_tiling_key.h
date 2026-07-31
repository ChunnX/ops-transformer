/* *
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_quant_grouped_mat_mul_tiling_key.h
 * \brief
 */
#ifndef __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_KEY_H__
#define __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// communication mode
#ifndef TILINGKEY_TPL_CCU
#define TILINGKEY_TPL_CCU 0
#endif
#ifndef TILINGKEY_TPL_AICPU
#define TILINGKEY_TPL_AICPU 1
#endif

ASCENDC_TPL_ARGS_DECL(AlltoAllvQuantGroupedMatMul,
                      ASCENDC_TPL_BOOL_DECL( // gmm weight transpose
                          TILINGKEY_GMM_WEIGHT_TRANSPOSE, 0, 1),
                      ASCENDC_TPL_BOOL_DECL( // mm weight transpose
                          TILINGKEY_MM_WEIGHT_TRANSPOSE, 0, 1),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_COMM_MODE, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_TPL_CCU, TILINGKEY_TPL_AICPU),
                      ASCENDC_TPL_BOOL_DECL( // counts source: 0=attr list, 1=GM tensor
                          TILINGKEY_COUNTS_FROM_TENSOR, 0, 1), );

ASCENDC_TPL_SEL(
    // CCU: gmmTrans(0,1) x mmTrans(0,1) x countsFromTensor(0,1) = 8
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANSPOSE, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_MM_WEIGHT_TRANSPOSE, 0, 1),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_COUNTS_FROM_TENSOR, 0, 1)),
    // AICPU: gmmTrans(0,1) x mmTrans(0,1) x countsFromTensor(0,1) = 8
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANSPOSE, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_MM_WEIGHT_TRANSPOSE, 0, 1),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_AICPU),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_COUNTS_FROM_TENSOR, 0, 1)),
); // AlltoAllvQuantGroupedMatMul

#endif // __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_KEY_H__
