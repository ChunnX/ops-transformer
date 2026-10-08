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
 * \file compressor_grad_template_tiling_key.h
 * \brief CompressorGrad arch22 tiling key declaration. Only the io dtype is a
 *        template argument; coff/layout/cmp_ratio stay runtime tiling fields.
 */

#ifndef COMPRESSOR_GRAD_TEMPLATE_TILING_KEY_H
#define COMPRESSOR_GRAD_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "compressor_grad_tiling_data.h"

#define ASCENDC_TPL_2_BW 2 // 每个参数占用2个bit位

// 可表示的tilingkey范围为64bit，注意不可超过限制
ASCENDC_TPL_ARGS_DECL(CompressorGrad, // 算子唯一标识，与opType保持一致
                      // bit:0-1 x的dtype  0:BF16 1:FP16
                      ASCENDC_TPL_UINT_DECL(D_TYPE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1), );

ASCENDC_TPL_SEL(

    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(D_TYPE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::CompressorGradTilingDataA3)), );

#endif // COMPRESSOR_GRAD_TEMPLATE_TILING_KEY_H
