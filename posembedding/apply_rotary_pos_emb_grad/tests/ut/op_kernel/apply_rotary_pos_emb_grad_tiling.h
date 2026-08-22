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
 * \file apply_rotary_pos_emb_grad_tiling.h
 * \brief kernel UT tiling shim.
 *
 * The apply_rotary_pos_emb_grad kernel is implemented by the top-level template
 * in apply_rotary_pos_emb_grad_apt.cpp. The kernel declares ApplyRopeGradTilingData
 * through REGISTER_TILING_DEFAULT and reads raw tiling GM bytes with
 * GET_TILING_DATA_WITH_STRUCT.
 *
 * The UT CMake force-includes this file to provide a byte-copy InitTilingData
 * without redefining the kernel-side tiling struct.
 */

#ifndef APPLY_ROTARY_POS_EMB_GRAD_TILING_H
#define APPLY_ROTARY_POS_EMB_GRAD_TILING_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#ifndef DTYPE_GRAD_QUERY_EMBED
#define DTYPE_GRAD_QUERY_EMBED float
#endif
#define __CCE_UT_TEST__

inline long max(const long x, const long y)
{
    return x > y ? x : y;
}

inline unsigned long max(const unsigned long x, const unsigned long y)
{
    return x > y ? x : y;
}

inline long min(const long x, const long y)
{
    return x < y ? x : y;
}

inline unsigned long min(const unsigned long x, const unsigned long y)
{
    return x < y ? x : y;
}

template <typename T>
inline void InitTilingData(const uint8_t* p_tilingdata, T* tilingdata)
{
    memcpy(tilingdata, p_tilingdata, sizeof(T));
}

#undef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif // APPLY_ROTARY_POS_EMB_GRAD_TILING_H
