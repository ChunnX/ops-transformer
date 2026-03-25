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
 * \file flash_attn_template_tiling_key.h
 * \brief FlashAttn TilingKey定义（非量化，仅FP16/BF16）
 */

#ifndef TEMPLATE_TILING_KEY_FLASH_ATTN_H_
#define TEMPLATE_TILING_KEY_FLASH_ATTN_H_

#include "ascendc/host_api/tiling/template_argument.h"

// kernel通过宏定义隔离dtype编译tilingkey，降低耗时
#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (-1)
#endif

#ifndef ASCENDC_TPL_1_BW
#define ASCENDC_TPL_1_BW 1
#endif
#ifndef ASCENDC_TPL_2_BW
#define ASCENDC_TPL_2_BW 2
#endif
#ifndef ASCENDC_TPL_4_BW
#define ASCENDC_TPL_4_BW 4
#endif
#ifndef ASCENDC_TPL_10_BW
#define ASCENDC_TPL_10_BW 10
#endif
#ifndef ASCENDC_TPL_12_BW
#define ASCENDC_TPL_12_BW 12
#endif

// FlashAttn TilingKey布局，总计约56bit
ASCENDC_TPL_ARGS_DECL(FlashAttn,
    // bit:1-0  KernelTypeKey: 0=正常计算, 1=空tensor场景
    ASCENDC_TPL_UINT_DECL(KernelTypeKey, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit:3-2  ImplMode: 0=高精度, 1=高性能, 2=invalid_line高精度
    ASCENDC_TPL_UINT_DECL(ImplMode, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
    // bit:7-4  Layout (Q的数据布局): 1=BSND, 2=BNSD, 3=TND (与flash_attn_score保持一致)
    ASCENDC_TPL_UINT_DECL(Layout, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4),
    // bit:17-8  S1TemplateType
    ASCENDC_TPL_UINT_DECL(S1TemplateType, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_LIST, 0, 64, 128),
    // bit:27-18  S2TemplateType
    ASCENDC_TPL_UINT_DECL(S2TemplateType, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_LIST, 0, 64, 128, 256),
    // bit:39-28  DTemplateType
    ASCENDC_TPL_UINT_DECL(DTemplateType, ASCENDC_TPL_12_BW, ASCENDC_TPL_UI_LIST, 0, 64, 128, 192, 256, 768),
    // bit:51-40  DvTemplateType (0表示与DTemplateType相同)
    ASCENDC_TPL_UINT_DECL(DvTemplateType, ASCENDC_TPL_12_BW, ASCENDC_TPL_UI_LIST, 0, 64, 128, 192, 256),
    // bit:52  HasAtten: 是否有attention mask（band/sliding window等场景）
    ASCENDC_TPL_UINT_DECL(HasAtten, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit:53  IsPA: 是否为分页注意力（PA_ND/PA_Nz layout）
    ASCENDC_TPL_UINT_DECL(IsPA, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit:54  IsSoftmaxLse: 是否输出softmax_lse（训练正向传播）
    ASCENDC_TPL_UINT_DECL(IsSoftmaxLse, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit:55  Regbase: 是否使用regbase接口
    ASCENDC_TPL_UINT_DECL(Regbase, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(KernelTypeKey, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ImplMode, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
        ASCENDC_TPL_UINT_SEL(S1TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(S2TemplateType, ASCENDC_TPL_UI_LIST, 128),
        ASCENDC_TPL_UINT_SEL(DTemplateType, ASCENDC_TPL_UI_LIST, 0, 64, 128, 192, 256, 768),
        ASCENDC_TPL_UINT_SEL(DvTemplateType, ASCENDC_TPL_UI_LIST, 0, 64),
        ASCENDC_TPL_UINT_SEL(HasAtten, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(IsPA, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(IsSoftmaxLse, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Regbase, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FlashAttentionScoreSimplifiedTilingData)
    ),
);

#endif // TEMPLATE_TILING_KEY_FLASH_ATTN_H_
