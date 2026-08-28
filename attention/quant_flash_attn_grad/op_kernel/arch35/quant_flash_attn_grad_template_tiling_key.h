/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON‑INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_grad_template_tiling_key.h
 * \brief QuantFlashAttnGrad (HIFLOAT8 only) 模板参数笛卡尔积声明
 */

#ifndef TEMPLATE_TILING_KEY_QUANT_FAG_H_
#define TEMPLATE_TILING_KEY_QUANT_FAG_H_
#include "ascendc/host_api/tiling/template_argument.h"
#include "quant_flash_attention_score_grad_tiling_data_regbase.h"

// kernel通过宏定义隔离dtype编译tilingkey，降低耗时。tiling侧没有相关宏
#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (-1)
#endif

#define ASCENDC_TPL_3_BW 3
#define ASCENDC_TPL_4_BW 4
#define ASCENDC_TPL_8_BW 8
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_12_BW 12

using QuantFagTiling = optiling::QuantFag::QuantFlashAttnGradTilingData;

// 可表示的tilingkey范围为64bit，注意不能超过限制
ASCENDC_TPL_ARGS_DECL(QuantFlashAttnGrad, // 算子唯一标识，可以opType保持一致
    ASCENDC_TPL_UINT_DECL(HasAttenMask, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(HasSink, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(S1TemplateNum, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 512),
    ASCENDC_TPL_UINT_DECL(S2TemplateNum, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 512),
    ASCENDC_TPL_UINT_DECL(DTemplateNum, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 128),
    ASCENDC_TPL_UINT_DECL(IsNEqual, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 1),
    ASCENDC_TPL_UINT_DECL(Layout, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
);

ASCENDC_TPL_SEL(
    // 量化主路径: HIFLOAT8
    #if (ORIG_DTYPE_QUERY == -1) || (ORIG_DTYPE_QUERY == DT_HIFLOAT8)
        ASCENDC_TPL_ARGS_SEL(
            ASCENDC_TPL_UINT_SEL(HasAttenMask, ASCENDC_TPL_UI_LIST, 0, 1),
            ASCENDC_TPL_UINT_SEL(HasSink, ASCENDC_TPL_UI_LIST, 0, 1),
            ASCENDC_TPL_UINT_SEL(S1TemplateNum, ASCENDC_TPL_UI_LIST, 512),
            ASCENDC_TPL_UINT_SEL(S2TemplateNum, ASCENDC_TPL_UI_LIST, 512),
            ASCENDC_TPL_UINT_SEL(DTemplateNum, ASCENDC_TPL_UI_LIST, 128),
            ASCENDC_TPL_UINT_SEL(IsNEqual, ASCENDC_TPL_UI_LIST, 1),
            ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0, 1, 2),
            ASCENDC_TPL_TILING_STRUCT_SEL(QuantFagTiling)
        ),
    #endif
);

#endif //TEMPLATE_TILING_KEY_QUANT_FA