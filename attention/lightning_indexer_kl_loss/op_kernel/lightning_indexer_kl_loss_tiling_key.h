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
 * \file lightning_indexer_kl_loss_tiling_key.h
 * \brief lightning_indexer_kl_loss tiling key declare
 */

#ifndef __LIGHTNING_INDEXER_KL_LOSS_TILING_KEY_H__
#define __LIGHTNING_INDEXER_KL_LOSS_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"


/* 模板参数 */
ASCENDC_TPL_ARGS_DECL(LightningIndexerKlLoss,
    // bit: 1   DeterSparseType
    //          0：NO_DETER
    //          1：DETER
    ASCENDC_TPL_BOOL_DECL(DeterType, 0, 1),
    // bit: 1   DataType
    //          0：FLOAT16/BFLOAT16
    //          1：FLOAT32
    ASCENDC_TPL_BOOL_DECL(DataType, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(DeterType, 0, 1),
        ASCENDC_TPL_BOOL_SEL(DataType, 0, 1)));

#endif