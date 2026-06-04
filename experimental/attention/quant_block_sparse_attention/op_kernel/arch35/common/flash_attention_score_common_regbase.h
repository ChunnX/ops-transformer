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
 * \file flash_attention_score_common_regbase.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_COMMON_REGBASE_H
#define FLASH_ATTENTION_SCORE_COMMON_REGBASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "stdarg.h"
#include "util_regbase.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace regbaseutil;

constexpr uint64_t BLOCK_BYTE = 32;
constexpr int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
constexpr int32_t SOFTMAX_K_ALIGNED_SIZE = 64;
constexpr uint64_t DATACOPYPAD_PADDING_VALUE_ZERO = 0;
constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;
constexpr uint32_t NEGATIVE_MIN_VALUE_FP16 = 0xFBFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;
constexpr int64_t FP8_QUANT_BLOCK_SIZE = 128;

constexpr int64_t attenMaskBN2GS1S2 = 0;
constexpr int64_t attenMaskBS1S2 = 1;
constexpr int64_t attenMaskS1S2 = 2;
constexpr int64_t attenMaskTT = 99;
constexpr uint16_t PREFIX_N_MAX_B = 32;
constexpr int32_t fp32BaseSize = 8;

constexpr uint32_t attenMaskNoCompress = 0;
constexpr uint32_t attenMaskLeftUpCausalCompress = 1;
constexpr uint32_t attenMaskRightDownCausalCompress = 2;
constexpr uint16_t SHIFT_NUM_2 = 2;
constexpr uint16_t SHIFT_NUM_6 = 6;
constexpr uint16_t ADD_NUM_63 = 63;

constexpr uint32_t L0C_SHARED_SIZE_64K = 64 * 1024;
constexpr uint32_t L0C_SHARED_SIZE_128K = 128 * 1024;
constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;
constexpr uint64_t MM2_RES_INTRA_EVENT[2] = {7, 8}; // mm2ResIntraEvent
constexpr uint64_t MM1_RES_INTRA_EVENT[2] = {9, 10}; //mm1ResIntraEvent
constexpr uint64_t KB_TO_BYTES = 1024;
constexpr uint64_t L0C_SIZE = 256;
constexpr uint64_t MLA_L0A_SIZE = 64;
constexpr uint64_t MLA_L0B_SIZE = 64; 
constexpr uint64_t BASE_SIZE_128 = 128;
constexpr uint64_t FLOAT_BYTES = 4;
constexpr uint64_t BT_PP_SIZE = 4;
constexpr uint64_t BT_AVAL_SIZE = 16;
constexpr uint64_t BT_BVAL_SIZE = 16;


enum class ImplModeEnum {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

namespace BaseApi {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
};


__aicore__ constexpr uint16_t Align64Func(uint16_t data) {
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

__aicore__ constexpr bool IsDn() {
    return false;
}

// TODO: 判断逻辑重写
__aicore__ constexpr TPosition GetC2Position() {
    return TPosition::VECCALC;
}
}
#endif // FLASH_ATTENTION_SCORE_COMMON_REGBASE_H