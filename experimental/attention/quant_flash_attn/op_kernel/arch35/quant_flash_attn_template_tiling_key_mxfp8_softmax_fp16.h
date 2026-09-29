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
 * \file quant_flash_attn_template_tiling_key.h
 * \brief QuantFlashAttn TilingKey定义（MxFP8 SOFTMAX FP16 单模板）
 */

#ifndef TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_
#define TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "quant_flash_attn_common_def_mxfp8_softmax_fp16.h"
#include "quant_flash_attn_tiling_data_mxfp8_softmax_fp16.h"

using namespace optiling;

ASCENDC_TPL_ARGS_DECL(QuantFlashAttn,
                      //    InOutLayoutType (8-bit)
                      //    1: InOutLayoutType_BNSD_BNSD
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255),
                      //    Config (10-bit)
                      //    5: Config_S1Aligned256_S2Aligned256_DAligned128_DVAligned128
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 1023),
                      //    QuantMode (5-bit)
                      //    3: QFA_MXFP8_SOFTMAX_FP16
                      ASCENDC_TPL_UINT_DECL(QuantMode, ASCENDC_TPL_5_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 31),
                      //    HasAttenMask
                      //    0 / 1
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, 0, 1),
                      //    KvLayoutType (2-bit)
                      //    0: KvLayoutType_NO_PA
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      //    IsFd
                      //    0 / 1
                      ASCENDC_TPL_BOOL_DECL(IsFd, 0, 1));

ASCENDC_TPL_SEL(
    // MxFP8 SOFTMAX FP16 单模板
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned256_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFA_MXFP8_SOFTMAX_FP16),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingData)));

#endif // TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_