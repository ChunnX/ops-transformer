/**
  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
  * CANN Open Software License Agreement Version 2.0 (the "License").
  * Please refer to the License for details. You may not use this file except in compliance with the License.
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
  * See LICENSE in the software repository for the full text of the License.
  */

/*!
 * \file quant_flash_attn_common_def.h
 * \brief QuantFlashAttn算子tiling和kernel共用的相关定义
 */

#ifndef QUANT_FLASH_ATTN_COMMON_DEF_H_
#define QUANT_FLASH_ATTN_COMMON_DEF_H_

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_3_BW 3

#define InOutLayoutType_BNSD_BNSD 1
#define InOutLayoutType_BSH_BSH 0
#define InOutLayoutType_TND_TND 2

#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_BBH 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

#define Config_S1Aligned128_S2Aligned512_DAligned64_DVAligned64 0
#define Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128 1

#endif // QUANT_FLASH_ATTN_COMMON_DEF_H_
