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
 * \file quant_block_sparse_attn_template_tiling_key.h
 * \brief QuantBlockSparseAttn template tiling key declarations.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_TEMPLATE_TILING_KEY_H
#define QUANT_BLOCK_SPARSE_ATTN_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define BSA_LAYOUT_TND 2
#define BSA_LAYOUT_NTD 5

#define BSA_KV_LAYOUT_CONTIGUOUS 0
#define BSA_KV_LAYOUT_PA_ND 1
#define BSA_KV_LAYOUT_PA_BNSD 3

#define BSA_MASK_NONE 0
#define BSA_MASK_CAUSAL 3

#define BSA_DTYPE_FP8_E4M3FN 0

#define BSA_TPL_4_BW 4


ASCENDC_TPL_ARGS_DECL(
    QuantBlockSparseAttn, ASCENDC_TPL_UINT_DECL(QKV_DTYPE, BSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, BSA_DTYPE_FP8_E4M3FN),
    ASCENDC_TPL_UINT_DECL(LAYOUT_T, BSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, BSA_LAYOUT_TND, BSA_LAYOUT_NTD),
    ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, BSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, BSA_KV_LAYOUT_CONTIGUOUS,
                          BSA_KV_LAYOUT_PA_BNSD),
    ASCENDC_TPL_UINT_DECL(MASK_MODE, BSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, BSA_MASK_NONE, BSA_MASK_CAUSAL),
    ASCENDC_TPL_BOOL_DECL(RETURN_SOFTMAX_LSE, 0, 1),);


ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QKV_DTYPE, ASCENDC_TPL_UI_LIST, BSA_DTYPE_FP8_E4M3FN),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, BSA_LAYOUT_TND, BSA_LAYOUT_NTD),
                         ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, BSA_KV_LAYOUT_CONTIGUOUS,
                                              BSA_KV_LAYOUT_PA_BNSD),
                         ASCENDC_TPL_UINT_SEL(MASK_MODE, ASCENDC_TPL_UI_LIST, BSA_MASK_NONE, BSA_MASK_CAUSAL),
                         ASCENDC_TPL_BOOL_SEL(RETURN_SOFTMAX_LSE, 0, 1),),);


#endif // QUANT_BLOCK_SPARSE_ATTN_TEMPLATE_TILING_KEY_H
