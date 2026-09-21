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
 * \file quant_block_sparse_attn_tiling_key.h
 * \brief Arch35 QuantBlockSparseAttn tiling-key field declaration.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_TILING_KEY_H
#define QUANT_BLOCK_SPARSE_ATTN_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define QBSA_LAYOUT_TND 2
#define QBSA_LAYOUT_NTD 5

#define QBSA_KV_LAYOUT_CONTIGUOUS 0
#define QBSA_KV_LAYOUT_PA_ND 1
#define QBSA_KV_LAYOUT_PA_BNSD 3

#define QBSA_MASK_NONE 0
#define QBSA_MASK_CAUSAL 3

#define QBSA_DTYPE_FP8_E4M3FN 0

// Neutral encoded values. Feature headers give these values their FP8/MXFP8 meaning.
#define QBSA_TILING_CONFIG_0 0
#define QBSA_TILING_CONFIG_1 1
#define QBSA_TILING_QUANT_MODE_1 1
#define QBSA_TILING_QUANT_MODE_2 2

#define QBSA_TPL_4_BW 4

ASCENDC_TPL_ARGS_DECL(
    QuantBlockSparseAttn, ASCENDC_TPL_UINT_DECL(QKV_DTYPE, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, QBSA_DTYPE_FP8_E4M3FN),
    ASCENDC_TPL_UINT_DECL(LAYOUT_T, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, QBSA_LAYOUT_TND, QBSA_LAYOUT_NTD),
    ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, QBSA_KV_LAYOUT_CONTIGUOUS,
                          QBSA_KV_LAYOUT_PA_BNSD),
    ASCENDC_TPL_UINT_DECL(MASK_MODE, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, QBSA_MASK_NONE, QBSA_MASK_CAUSAL),
    ASCENDC_TPL_BOOL_DECL(RETURN_SOFTMAX_LSE, 0, 1),
    ASCENDC_TPL_UINT_DECL(Config, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, QBSA_TILING_CONFIG_0, QBSA_TILING_CONFIG_1),
    ASCENDC_TPL_UINT_DECL(QUANT_MODE, QBSA_TPL_4_BW, ASCENDC_TPL_UI_LIST, QBSA_TILING_QUANT_MODE_1,
                          QBSA_TILING_QUANT_MODE_2), );

#endif // QUANT_BLOCK_SPARSE_ATTN_TILING_KEY_H
