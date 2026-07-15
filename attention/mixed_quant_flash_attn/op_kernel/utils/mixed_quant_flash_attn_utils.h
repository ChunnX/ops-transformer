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
 * \file mixed_quant_flash_attn_common_def.h
 * \brief
 */

#ifndef MIXED_QUANT_FLASH_ATTN_UTILS_H_
#define MIXED_QUANT_FLASH_ATTN_UTILS_H_

#include "util.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mixed_quant_flash_attn_common_def.h"

using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

__aicore__ inline constexpr uint16_t MqfaAlignUp64(uint16_t data) {
    return (data + 63U) >> 6U << 6U;
}

__aicore__ inline constexpr uint32_t MqfaAlignUp16(uint32_t data) {
    return (data + 15U) >> 4U << 4U;
}

__aicore__ inline uint16_t MqfaGetRealDealSize(uint16_t realSize) {
    uint16_t dealSize = ((realSize >> 1) + 31U) >> 5U << 5U;
    return (dealSize > realSize) ? realSize : dealSize;
}

#endif  // FLASH_ATTN_COMMON_DEF_H_