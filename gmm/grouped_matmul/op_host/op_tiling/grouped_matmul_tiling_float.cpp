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
 * \file grouped_matmul_tiling_float.cpp
 * \brief
 */
#include "grouped_matmul_tiling_utils.h"

namespace optiling {
void GMMTiling::FullLoadK(const GMMCompileInfo* compileInfoPtr) {
  if (wFormat_ == matmul_tiling::CubeFormat::ND && !transposeWeight_ && !transposeX_ && xDType_ == weightDtype_ &&
      (weightDtype_ == ge::DT_FLOAT16 || weightDtype_ == ge::DT_BF16) &&
      maxM_ >= FULL_K_M_E_THRESHOLD * groupNum_ && maxN_ == FULL_K_N_THRESHOLD &&
      maxK_ <= FULL_K_MAX_K_THRESHOLD &&
      maxK_ >= FULL_K_MIN_K_THRESHOLD) {
    int64_t fullLoadStepKa = CeilDiv(maxK_ , baseK_);
    int64_t fullLoadStepKb = fullLoadStepKa / static_cast<int64_t>(QUEUE_DOUBLE_BUFFER);
    int64_t fullLoadDepthKa = fullLoadStepKa * static_cast<int64_t>(QUEUE_DOUBLE_BUFFER);
    int64_t fullLoadDepthKb = fullLoadStepKb * static_cast<int64_t>(QUEUE_DOUBLE_BUFFER);
    bool ifFullLoad = (maxM_ > FULL_K_M_THRESHOLD) && isAllSingleTensor_ && groupType_ == SPLIT_M &&
                      (((baseM_ * baseK_ * static_cast<int64_t>(mmDataTypeSize_)) * fullLoadDepthKa +
                        (baseN_ * baseK_ * static_cast<int64_t>(mmDataTypeSize_)) * fullLoadDepthKb) <=
                      static_cast<int64_t>(compileInfoPtr->l1Size));
    if (ifFullLoad) {
      tilingData.mmTilingData.set_stepKa(fullLoadStepKa);  // set precomputed mmStepKa
      tilingData.mmTilingData.set_depthA1(fullLoadDepthKa);  // set precomputed mmDepthA1
      tilingData.mmTilingData.set_stepKb(fullLoadStepKb);  // set precomputed mmStepKb
      tilingData.mmTilingData.set_depthB1(fullLoadDepthKb);  // set precomputed mmDepthB1
      tilingData.mmTilingData.set_iterateOrder(1);  // set precomputed stepN
      tilingData.gmmBaseParams.set_singleN(FULL_K_SINGLE_N);  // 0 is the default value
    }
  }
}

}  // namespace optiling
