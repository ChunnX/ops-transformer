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
 * \file grouped_matmul_tiling_a4w4.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_A4W4_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_A4W4_H

#include "grouped_matmul_tiling_utils.h"

namespace optiling {
bool GMMTiling::IsA4W4OptimizeCondition() {
    bool isKInRange = (maxK_ >= A4W4OPTIMIZE_K_LOWER && maxK_ <= A4W4OPTIMIZE_K_UPPER);
    bool isNInRange = (maxN_ >= A4W4OPTIMIZE_N_LOWER && maxN_ <= A4W4OPTIMIZE_N_UPPER);
    bool isGroupInRange = (groupNum_ >= A4W4OPTIMIZE_GROUP_NUM_LOWER && groupNum_ <= A4W4OPTIMIZE_GROUP_NUM_UPPER);
    bool isTuningInRange = (tuningConfig_ >= A4W4OPTIMIZE_PERM_LOWER) &&
                          (tuningConfig_ <= A4W4OPTIMIZE_PERM_UPPER);
    bool isPerGroupValid = isPerGroup_ && quantGroupSize_ == A4W4OPTIMIZE_QUANT_GROUP_SIZE;
    bool isDataTypeValid = yDtype_ == ge::DT_BF16 && scaleDtype_ == ge::DT_UINT64 && perTokenScaleDtype_ == ge::DT_FLOAT;
    bool isConfigValid = !transposeX_ && (splitItem_ == A4W4OPTIMIZE_SPLIT_ITEM2 || splitItem_ == A4W4OPTIMIZE_SPLIT_ITEM3)
                          && (groupListType_ == A4W4OPTIMIZE_GROUP_LIST_TYPE)
                          && (groupType_ == A4W4OPTIMIZE_GROUP_TYPE) && (actType_ == 0)
                          && !transposeWeight_;
    bool isFormatValid = (wFormat_ == matmul_tiling::CubeFormat::NZ);

    return isKInRange && isNInRange && isTuningInRange && isGroupInRange && isPerGroupValid && isA4W4_ &&
           isDataTypeValid && isConfigValid && !hasBias_ && isFormatValid;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlkA4W4() {
  ubDivideBlkNum_ = UB_A4W4_BLOCK_NUM;
  ubIoBlkNum_ = UB_A4W4_IO_USED_BLOCK_HALF;
  ubBlockAlign_ = UB_A4W4_PER_BLOCK_ALIGN;
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_A4W4_H
