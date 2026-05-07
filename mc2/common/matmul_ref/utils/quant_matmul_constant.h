/**
 * MX 量化专用常量（通用常量由 matmul_constant.h 提供）
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "matmul_constant.h"

// GM offset tuple indices for quant matmul helpers.
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_SCALEA_OFFSET = 2UL;
constexpr uint64_t IDX_SCALEB_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 4UL;

constexpr uint64_t B8_MIN_STEP = 2UL;
constexpr uint16_t SCALE_BUFFER_FLAG_0 = 4;
constexpr uint16_t SCALE_BUFFER_FLAG_1 = 5;
constexpr uint8_t  MTE1_MTE2_EVENT_ID_NUM = 6;
