/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_v2_tiling_key.h
 * \brief Tiling key declaration using ASCENDC_TPL macros.
 */
#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_TILING_KEY_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_TILING_KEY_H

#define ASCENDC_TPL_1_BW 1

#include "ascendc/host_api/tiling/template_argument.h"

// 模板参数支持的范围定义
// 顺序与 pypto-pro TilingKeyField 声明顺序一致: isLongS2, hasSeqUsedQ, hasSeqUsedK
ASCENDC_TPL_ARGS_DECL(DenseLightningIndexerSoftmaxLseV2,
                      ASCENDC_TPL_UINT_DECL(IS_LONG_S2, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(HAS_SEQUSED_Q, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(HAS_SEQUSED_K, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1));

// 支持的模板参数组合
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(IS_LONG_S2, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HAS_SEQUSED_Q, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HAS_SEQUSED_K, ASCENDC_TPL_UI_LIST, 0, 1),),);

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_TILING_KEY_H
