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
 * \file flash_attention_mixedquant_block_vec_flashdecode.h
 * \brief MixedQuantFlashAttn FlashDecode Vec Block骨架定义
 */

#ifndef FLASH_ATTENTION_MIXEDQUANT_BLOCK_VEC_FLASHDECODE_H_
#define FLASH_ATTENTION_MIXEDQUANT_BLOCK_VEC_FLASHDECODE_H_

#include "../utils/mixed_quant_flash_attn_common_def.h"

namespace BaseApi {

template <typename INPUT_T, typename ACC_T, typename OUT_T, inferFaLayOutTypeEnum inputLayoutType,
          inferFaLayOutTypeEnum outputLayoutType, inferS1TemplateType s1TemplateType,
          inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType, bool hasAttenMask, uint8_t KvLayoutType>
class FAMixedQuantGqaBlockVecFlashDecode {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_MIXEDQUANT_BLOCK_VEC_FLASHDECODE_H_
