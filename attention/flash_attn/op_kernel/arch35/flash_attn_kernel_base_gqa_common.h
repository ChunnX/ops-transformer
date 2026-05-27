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
 * \file flash_attn_kernel_base_gqa_common.h
 * \brief Flash Attention GQA common definitions (no state)
 */

#ifndef FLASH_ATTENTION_NOQUANT_GQA_COMMON_H_
#define FLASH_ATTENTION_NOQUANT_GQA_COMMON_H_

#include "../../../common/op_kernel/fia_public_define.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../utils/flash_attn_common_def.h"
#include "../utils/flash_attn_utils.h"

#include "flash_attn_block_cube_noquant_gqa.h"
#include "flash_attn_block_vec_noquant_gqa.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "flash_attention_noquant_block_vec_flashdecode.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "flash_attn_tiling_data.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_COMMON_H_