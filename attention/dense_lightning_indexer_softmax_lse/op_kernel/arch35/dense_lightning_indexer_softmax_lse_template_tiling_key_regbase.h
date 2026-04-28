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
 * \file dense_lightning_indexer_softmax_lse_template_tiling_key_regbase.h
 * \brief Template tiling key definitions for A5 (arch35) implementation
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TEMPLATE_TILING_KEY_REGBASE_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TEMPLATE_TILING_KEY_REGBASE_H

#include "ascendc/host_api/tiling/template_argument.h"

enum class LayoutTypeRegbase : uint8_t {
    LAYOUT_BSND = 0,
    LAYOUT_TND = 1,
    LAYOUT_NONE
};

#define MQA_SPARSE 3
#define ASCENDC_TPL_4_BW 4

ASCENDC_TPL_ARGS_DECL(DenseLISoftmaxLse,
    ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::DenseLISoftmaxLseRegBaseTilingData)
    )
);
#endif