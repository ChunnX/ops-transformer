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
 * \file quant_flash_attn_template_tiling_key.h
 * \brief
 */

#ifndef TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_
#define TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "quant_flash_attn_tiling_data.h"

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_3_BW 3

#define LAYOUT_ENUM_BSND 0
#define LAYOUT_ENUM_BNSD 1
#define LAYOUT_ENUM_TND 3

enum class QFA_LAYOUT : uint32_t
{
    BSND = LAYOUT_ENUM_BSND,
    BNSD = LAYOUT_ENUM_BNSD,
    TND = LAYOUT_ENUM_TND,
};

template <typename QUANT_T, typename SCALE_T, typename OUT_T, const bool PAGE_ATTENTION = false,
          QFA_LAYOUT LAYOUT_T = QFA_LAYOUT::BSND, QFA_LAYOUT KV_LAYOUT_T = QFA_LAYOUT::BSND,
          const bool HAS_MASK = false, typename... Args>
struct QFAType {
    using quantType = QUANT_T;
    using scaleType = SCALE_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr QFA_LAYOUT qLayout = LAYOUT_T;
    static constexpr QFA_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr bool hasMask = HAS_MASK;
};

ASCENDC_TPL_ARGS_DECL(QuantFlashAttn,
    //    bit 1 PageAttention
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(PageAttention, 0, 1),
    //    bit 2 HasMask
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(HasMask, 0, 1),
    //    bit 3-10 QueryLayout
    //    0: BSND
    //    1: BNSD
    //    3: TND
    ASCENDC_TPL_UINT_DECL(QueryLayout, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, LAYOUT_ENUM_BSND, LAYOUT_ENUM_BNSD, LAYOUT_ENUM_TND),
    //    bit 11-18 kvLayout
    //    0: BSND, when PageAttention enable, BSND is PA_BBND
    //    1: BNSD, when PageAttention enable, BNSD is PA_BNBD
    //    3: TND
    ASCENDC_TPL_UINT_DECL(KvLayout, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, LAYOUT_ENUM_BSND, LAYOUT_ENUM_BNSD, LAYOUT_ENUM_TND),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(PageAttention, 0),
        ASCENDC_TPL_BOOL_SEL(HasMask, 0, 1),
        ASCENDC_TPL_UINT_SEL(QueryLayout, ASCENDC_TPL_UI_LIST, LAYOUT_ENUM_BSND, LAYOUT_ENUM_BNSD, LAYOUT_ENUM_TND),
        ASCENDC_TPL_UINT_SEL(KvLayout, ASCENDC_TPL_UI_LIST, LAYOUT_ENUM_BSND, LAYOUT_ENUM_BNSD, LAYOUT_ENUM_TND),
        ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(PageAttention, 1),
        ASCENDC_TPL_BOOL_SEL(HasMask, 0, 1),
        ASCENDC_TPL_UINT_SEL(QueryLayout, ASCENDC_TPL_UI_LIST, LAYOUT_ENUM_BSND, LAYOUT_ENUM_BNSD, LAYOUT_ENUM_TND),
        ASCENDC_TPL_UINT_SEL(KvLayout, ASCENDC_TPL_UI_LIST, LAYOUT_ENUM_BSND, LAYOUT_ENUM_BNSD),
        ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingData)
    ),
);

#endif // TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_
