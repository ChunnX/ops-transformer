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
 * \file grouped_matmul_utils.h
 * \brief GMM-specific utilities (refactored: common parts moved to ascendc_utils.h & grouped_matmul_common.h).
 */
#ifndef ASCENDC_GROUPED_MATMUL_UTILS_H
#define ASCENDC_GROUPED_MATMUL_UTILS_H

#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113))
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#else
#include "kernel_operator.h"
#endif

#include "ascendc_utils.h"
#include "grouped_matmul_common.h"

namespace GROUPED_MATMUL {
using namespace AscendC;

constexpr uint64_t SYNC_AIV_AIC_FLAG = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG = 5;
constexpr uint64_t SYNC_MODE2 = 2;

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ != 310) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)
__aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, int32_t &preOffset,
                                                     const GMMBaseParams *__restrict &gmmBaseParams,
                                                     const GlobalTensor<int64_t> &groupListGm)
{
    int32_t splitValue = 0;
    if (likely(gmmBaseParams->groupType != -1)) {
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

} // namespace GROUPED_MATMUL

#endif // ASCENDC_GROUPED_MATMUL_UTILS_H
