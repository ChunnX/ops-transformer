/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mixed_quant_flash_attn_entry_regbase.h
 * \brief MixedQuantFlashAttn arch35 kernel类型推导与实例化（伪量化场景）
 *
 * 5字段 tiling key → 模板参数: inOutLayoutType, KvLayoutType, hasAttenMask, config, QuantMode
 */

#ifndef MIXED_QUANT_FLASH_ATTN_ENTRY_REGBASE_H_
#define MIXED_QUANT_FLASH_ATTN_ENTRY_REGBASE_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../utils/mixed_quant_flash_attn_common_def.h"
#include "mixed_quant_flash_attn_tiling_data.h"

template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask,
          uint8_t config, uint8_t QuantMode>
inline __aicore__ void
mixed_quant_flash_attn_kernel_run(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                  __gm__ uint8_t *kDescale, __gm__ uint8_t *vDescale, __gm__ uint8_t *attenMask,
                                  __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ,
                                  __gm__ uint8_t *sequsedKv, __gm__ uint8_t *blocktable, __gm__ uint8_t *learnableSink,
                                  __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                  __gm__ uint8_t *tiling, __gm__ uint8_t *runtimeMetaData)
{
    // TODO: 待后续填充实现
}

#endif // MIXED_QUANT_FLASH_ATTN_ENTRY_REGBASE_H_
