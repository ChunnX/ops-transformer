/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file mc2_grouped_matmul_utils.h
 * \brief
 */
#ifndef MC2_GROUPED_MATMUL_UTILS_H
#define MC2_GROUPED_MATMUL_UTILS_H

#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113))
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "../../../../common/include/op_kernel/grouped_matmul_utils_common.h"
#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113))
#if defined(ORIG_DTYPE_X) && defined(DT_INT8) && ORIG_DTYPE_X == DT_INT8
#define DTYPE_L0C_LOCAL int32_t
#else
#define DTYPE_L0C_LOCAL float
#endif
#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) && defined(DT_FLOAT8_E5M2) && defined(DT_FLOAT8_E4M3FN) && \
    defined(DT_HIFLOAT8) && defined(DT_INT8) && defined(DT_FLOAT4_E2M1) && defined(DT_FLOAT4_E1M2) &&              \
    defined(DT_INT4) &&                                                                                            \
    ((ORIG_DTYPE_X == DT_INT8 && ORIG_DTYPE_WEIGHT == DT_INT8) ||                                                  \
    (ORIG_DTYPE_X == DT_HIFLOAT8 && ORIG_DTYPE_WEIGHT == DT_HIFLOAT8) ||                                           \
    ((ORIG_DTYPE_X == DT_FLOAT8_E5M2 || ORIG_DTYPE_X == DT_FLOAT8_E4M3FN) &&                                       \
    (ORIG_DTYPE_WEIGHT == DT_FLOAT8_E5M2 || ORIG_DTYPE_WEIGHT == DT_FLOAT8_E4M3FN)) ||                             \
    ((ORIG_DTYPE_X == DT_FLOAT4_E2M1 || ORIG_DTYPE_X == DT_FLOAT4_E1M2) &&                                         \
    (ORIG_DTYPE_WEIGHT == DT_FLOAT4_E2M1 || ORIG_DTYPE_WEIGHT == DT_FLOAT4_E1M2)) ||                               \
    (ORIG_DTYPE_X == DT_INT4 && ORIG_DTYPE_WEIGHT == DT_INT4))
#define V310_GMM_QUANT
#if defined(ORIG_DTYPE_SCALE) && defined(DT_FLOAT8_E8M0) && ORIG_DTYPE_SCALE == DT_FLOAT8_E8M0
#define V310_GMM_QUANT_MX
#elif defined(ORIG_DTYPE_SCALE) && defined(DT_UINT64) && defined(DT_INT64) && \
    (ORIG_DTYPE_SCALE != DT_UINT64 && ORIG_DTYPE_SCALE != DT_INT64)
#define V310_GMM_QUANT_MIX
#define V310_GMM_QUANT_PERTENSOR_CUBE
#if (ORIG_DTYPE_X != DT_INT8 && ORIG_DTYPE_SCALE == DT_FLOAT)
#define V310_GMM_QUANT_PERTILE
#endif
#else
#define V310_GMM_QUANT_CUBE
#endif
#endif

#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_X != ORIG_DTYPE_WEIGHT
#if ((ORIG_DTYPE_X == DT_FLOAT16 || ORIG_DTYPE_X == DT_BF16) && (ORIG_DTYPE_WEIGHT == DT_FLOAT8_E5M2 ||            \
    ORIG_DTYPE_WEIGHT == DT_FLOAT8_E4M3FN || ORIG_DTYPE_WEIGHT == DT_HIFLOAT8 || ORIG_DTYPE_WEIGHT == DT_INT8 ||   \
    ORIG_DTYPE_WEIGHT == DT_FLOAT4_E2M1 || ORIG_DTYPE_WEIGHT == DT_FLOAT4_E1M2 || ORIG_DTYPE_WEIGHT == DT_FLOAT || \
    ORIG_DTYPE_WEIGHT == DT_INT32 || ORIG_DTYPE_WEIGHT == DT_INT4)) ||                                             \
    (ORIG_DTYPE_X == DT_INT8 && (ORIG_DTYPE_WEIGHT == DT_INT4 || ORIG_DTYPE_WEIGHT == DT_INT32)) ||                \
    (ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && (ORIG_DTYPE_WEIGHT == DT_FLOAT4_E2M1 || ORIG_DTYPE_WEIGHT == DT_FLOAT))
#define V310_GMM_ANTI_QUANT
#endif
#endif
#endif

#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) && defined(ORIG_DTYPE_Y) && defined(DT_INT8) && \
    defined(DT_BF16) && defined(DT_INT4)
#if ORIG_DTYPE_X == ORIG_DTYPE_WEIGHT
#if ORIG_DTYPE_X == DT_INT8
#if ORIG_DTYPE_Y == DT_BF16
#define GMM_QUANT_BF16
#define MM_DTYPE_Y int32_t
#elif ORIG_DTYPE_Y == DT_FLOAT16
#define GMM_QUANT_FLOAT16
#define MM_DTYPE_Y int32_t
#elif ORIG_DTYPE_Y == DT_INT32
#define GMM_QUANT_INT32
#else
#define GMM_QUANT_INT8
#endif
#elif ORIG_DTYPE_X == DT_INT4
#define GMM_A4W4
#define MM_DTYPE_Y half
#if ORIG_DTYPE_Y == DT_BF16
#define GMM_A4W4_BF16
#elif ORIG_DTYPE_Y == DT_FLOAT16
#define GMM_A4W4_FP16
#endif
#else
#define GMM_FLOAT
#endif
#else
#define GMM_ANTI_QUANT
#if ORIG_DTYPE_X == DT_INT8 && ORIG_DTYPE_WEIGHT == DT_INT4
#define GMM_ANTI_QUANT_A8W4_MSD
#define GMM_ANTI_QUANT_A8W4
#if ORIG_DTYPE_Y == DT_BF16
#define GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
#else
#define GMM_ANTI_QUANT_A8W4_MSD_OUT_FP16
#endif
#define MM_DTYPE_Y int32_t
#else
#define GMM_ANTI_QUANT
#endif
#endif
#endif

#if defined(DTYPE_Y) && !defined(MM_DTYPE_Y)
#define MM_DTYPE_Y DTYPE_Y
#endif

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
#ifdef GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
#undef DTYPE_SCALE
#define DTYPE_SCALE bfloat16_t
#elif defined(GMM_ANTI_QUANT_A8W4_MSD_OUT_FP16)
#undef DTYPE_SCALE
#define DTYPE_SCALE float
#endif
#endif

#if defined(CONST_TILING)
#define TILING_TYPE const int32_t
#else
#define TILING_TYPE __gm__ int32_t
#endif

#if defined(CONST_TILING)
#if defined(V310_GMM_ANTI_QUANT)
#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling)       \
    GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, member, obj, tiling); \
    const int32_t *(var) = (const int32_t *)((const uint8_t *)&obj);
#else
#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling) \
    GET_TILING_DATA_MEMBER(tilingType, member, obj, tiling);         \
    const int32_t *(var) = (const int32_t *)((const uint8_t *)&obj);
#endif
#else
#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling) \
    size_t offset##var = (size_t)(&((tilingType *)0)->member);       \
    __gm__ int32_t *(var) = (__gm__ int32_t *)((tiling) + (offset##var));
#endif

namespace MC2_GROUPED_MATMUL {
using namespace AscendC;
using namespace CommonGMMUtils;

#if ((defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) ||                        \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))) && \
    defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_X == DT_INT8 && ORIG_DTYPE_WEIGHT == DT_INT8
constexpr MatmulConfig NZ_CFG_MDL =
    GetMDLConfig(false, false, 0, true, false, false, true, true, true, false, false, true);
constexpr MatmulConfig matmulCFGUnitFlag{
    .doMultiDataLoad = true,
    .enUnitFlag = true,
    .enableKdimReorderLoad = true
};
#else
constexpr MatmulConfig NZ_CFG_MDL = GetMDLConfig(false, false, 0, true, false, false, true);
constexpr MatmulConfig matmulCFGUnitFlag{ false, false, true, 0, 0, 0, false, false, false, false,
    false, 0,     0,    0, 0, 0, 0,     0,     true };
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 310
__aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, int32_t &preOffset,
    const MC2GMMBaseParams *__restrict &gmmBaseParams, const GlobalTensor<int64_t> &groupListGm)
{
    int32_t splitValue = 0;
    if (likely(gmmBaseParams->groupType != -1)) { // -1: no  need to split
        if (gmmBaseParams->groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else {
            splitValue = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
        }
    }
    return splitValue;
}
#endif

} // namespace MC2_GROUPED_MATMUL

#endif // MC2_GROUPED_MATMUL_UTILS_H
