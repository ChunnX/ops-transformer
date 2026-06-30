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
 * \file mixed_quant_flash_attn_template_tiling_key.h
 * \brief MixedQuantFlashAttn TilingKey定义（5字段）
 */

#ifndef MIXED_QUANT_FLASH_ATTN_TEMPLATE_TILING_KEY_H_
#define MIXED_QUANT_FLASH_ATTN_TEMPLATE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "../utils/flash_attn_common_def.h"
#include "../../../incre_flash_attention/op_kernel/arch35/incre_flash_attention_tiling_regbase.h"
#include "mixed_quant_flash_attn_tiling_data.h"

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT (DT_BF16)
#endif

ASCENDC_TPL_ARGS_DECL(FlashAttn,
                      // inputLayoutQ (8-bit), spec 3值
                      //    0: BSND
                      //    1: BNSD
                      //    2: TND
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 2),

                      // inputLayoutKV (8-bit), spec 2值
                      //    1: PA_BBND
                      //    2: PA_BNBD
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 1, 2),

                      // hasAttenMask (1-bit)
                      //    0: false
                      //    1: true
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, false, true),

                      // config (3-bit), spec 当前仅1值
                      //    config=0: D=128, sOuter=64, sInner=256
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 0),

                      // quantMode (8-bit), spec 当前仅1值
                      //    1: q不量化, kv mxfp4
                      ASCENDC_TPL_UINT_DECL(QuantMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 1, 1),);

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSND,
                                                          InOutLayoutType_BNSD, InOutLayoutType_TND),
                                     ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, 1, 2),
                                     ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
                                     ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(MixedQuantFlashAttnTilingData)), );

#endif // MIXED_QUANT_FLASH_ATTN_TEMPLATE_TILING_KEY_H_
