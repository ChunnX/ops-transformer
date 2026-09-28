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
 * \file quant_flash_attn_block_vec_flashdecode.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H
#define QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "quant_flash_attn_common_def.h"
#include "memory_copy_arch92_quant_flash_attn.h"

namespace BaseApi {
template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128,
          bool hasAtten = false, uint8_t KvLayoutType = 0, bool useDn = false>
class QuantFlashAttnBlockVecFlashDecode {
public:
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecFlashDecode(ConstInfoX &constInfo){};
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128,
          bool hasAtten = false, uint8_t KvLayoutType = 0, bool useDn = false>
class QuantFlashAttnBlockVecFlashDecodeDummy {
public:
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecFlashDecodeDummy(ConstInfoX &constInfo){};
};

} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H
