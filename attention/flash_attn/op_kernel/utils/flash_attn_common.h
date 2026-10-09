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
 * \file flash_attn_common.h
 * \brief flash_attn_metadata_aicpu.cpp、flash_attn_template_tiling_key.h、flash_attn_tiling.cpp公共依赖
 */

#ifndef FLASH_ATTN_COMMON_H_
#define FLASH_ATTN_COMMON_H_

#define FA_Template_ND 0
#define FA_Template_DN 1

#define KV_STORAGE_MODE_CONTINUE 0
#define KV_STORAGE_MODE_PA_BSND 1
#define KV_STORAGE_MODE_PA_BNSD 2
#define KV_STORAGE_MODE_PA_NZ 3

// q_out layout
//   0: BSND (BSND排布与BSH一样)
//   1: BNSD
//   2: TND
//   3: BNSD_BSND
#define InOutLayoutType_BSND 0
#define InOutLayoutType_BNSD 1
#define InOutLayoutType_TND 2
#define InOutLayoutType_BNSD_BSND 3
#define InOutLayoutType_NTD_NTD 4
// backward compat
#define InOutLayoutType_BSH_BSH InOutLayoutType_BSND

// KvLayoutType
#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_BBH 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

#endif // FLASH_ATTN_COMMON_H_
