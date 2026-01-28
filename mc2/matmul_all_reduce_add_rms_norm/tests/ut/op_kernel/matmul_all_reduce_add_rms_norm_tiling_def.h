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
 * \file matmul_all_reduce_add_rms_norm_tiling_def.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_DEF_H
#define MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "test_add_rms_norm.h"
#include "../../../../matmul_all_reduce/tests/ut/op_kernel/weight_quant_batch_matmul_v2_tiling.h"
#include "../../../op_kernel/matmul_all_reduce_add_rms_norm_tiling_data.h"
#if __has_include("../../../../common/inc/hccl_stub.h")
#include "../../../../../tests/ut/framework_normal/common/hccl_stub.h"
#endif

#ifndef _ADD_RMS_NORM_TILING_H_
#define _ADD_RMS_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddRMSNormTilingData {
    uint32_t num_row;
    uint32_t num_col;
    uint32_t block_factor;
    uint32_t row_factor;
    uint32_t ub_factor;
    float epsilon;
    float avg_factor;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)  \
    __ubuf__ tilingStruct* tilingDataPointer =                                 \
            reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)     \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
    AddRMSNormTilingData tilingData;                                               \
    INIT_TILING_DATA(AddRMSNormTilingData, tilingDataPointer, tilingPointer);  \
    (tilingData).num_row = tilingDataPointer->num_row;                              \
    (tilingData).num_col = tilingDataPointer->num_col;                          \
    (tilingData).block_factor = tilingDataPointer->block_factor;                  \
    (tilingData).row_factor = tilingDataPointer->row_factor;            \
    (tilingData).ub_factor = tilingDataPointer->ub_factor;              \
    (tilingData).epsilon = tilingDataPointer->epsilon;            \
    (tilingData).avg_factor = tilingDataPointer->avg_factor;
#endif

#ifdef __CCE_KT_TEST__
#include "kernel_log.h"
#endif

constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;

inline void InitMatmulAllReduceAddRmsNormTilingData(uint8_t* tiling,
                                                    Mc2Tiling::MatmulAllReduceAddRmsNormTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(Mc2Tiling::MatmulAllReduceAddRmsNormTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                        \
    MatmulAllReduceAddRmsNormTilingData tiling_data;                                                    \
    InitMatmulAllReduceAddRmsNormTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling)            \
    auto var = ((tiling_type *)((uint8_t*)AscendC::GmAlloc(1024)))->member; \
    size_t offset##var = (size_t)(&((tiling_type*)0)->member);              \
    InitTilingData<decltype(var)>(tiling + offset##var, &var);

#endif  // FOREACH_MINIMUM_SCALAR_TILING_DEF_H
