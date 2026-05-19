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
 * \file grouped_matmul_tiling_a16w8.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_A16W8_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_A16W8_H

#include "grouped_matmul_tiling_utils.h"

namespace optiling {
ge::graphStatus GMMTiling::GetPerGroupNum(const gert::TilingContext* context) {
  auto antiquantScale = context->GetDynamicInputTensor(ANTIQUANT_SCALE_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, antiquantScale);
  auto antiquantScaleShape = antiquantScale->GetStorageShape();
  int64_t dimNum = antiquantScaleShape.GetDimNum();
  auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, wTensor);
  gert::Shape wShape = wTensor->GetOriginShape();
  size_t wDimNum = wShape.GetDimNum();
  if ((isSingleWeight_ && wDimNum > 2UL && dimNum == 3L) || (!isSingleWeight_ && dimNum == 2L)) {  // 2 and 3: dim threshold
    int64_t g = antiquantScaleShape.GetDim(dimNum - 2L);
    perTokenOrPerGroupSize_ = g > 1L ? static_cast<uint32_t>(kList_[0] / g) : 0U;
    tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
  }
  if (isSingleWeight_) {
      tilingData.gmmBaseParams.set_withOffset(GetWithOffset(context));
  } else {
      tilingData.gmmBaseParams.set_withOffset(1);  // 非单场景，offset默认存在
  }
  return ge::GRAPH_SUCCESS;
}

void GMMTiling::DivideUbAndSetWorkspaceAntiquant(size_t* workspaces, const uint32_t& aicNum, uint32_t &ubSize) {
    if (isA16W8Msd_) {
      // whole workspace for a16w8 msd scene is combined by workspace for global max of each row (m * 8),
      // workspace for local reduce sum of each row (m * aicNum), workspace for data after prerpocessing x (2 * m * k)
      // and workspace for matmul output (2 * m * n)
      // 32: need 32 byte to store global max
      workspaces[0] += totalM_ * (aicNum * sizeof(float) + 32UL +
                                  A16W8_MSD_STEP * (static_cast<uint64_t>(maxK_) * sizeof(int8_t) +
                                                    static_cast<uint64_t>(maxN_) * sizeof(int32_t)));
      // 7: make aicnum align up to 8
      uint32_t alignedAicNum = (aicNum + 7U) & (~7U);
      ubSize = static_cast<uint32_t>(ubSize_ - (static_cast<uint64_t>(baseM_) / A16W8_MSD_STEP) *
                                               alignedAicNum * sizeof(float));
      // workspacesSize in GMMBaseParams is size of matmul left input matrix of matmul.
      workspacesSize_ += A16W8_MSD_STEP * static_cast<uint64_t>(maxK_) * static_cast<uint64_t>(maxM_);
    } else {
      for (uint32_t i = 0; i < groupNum_; i++) {
        bool isAllSingleTensor = isSingleX_ && isSingleWeight_ && isSingleY_;
        int32_t kInList = isAllSingleTensor ? kList_[0] : kList_[i];  // in s-s-s case，k only exits in the first of the list
        int32_t nInList = isAllSingleTensor ? nList_[0] : nList_[i];  // in s-s-s case，n only exits in the first of the list
        int32_t k = kList_[0] == -1 ? static_cast<int32_t>(maxK_) : kInList;
        int32_t n = nList_[0] == -1 ? static_cast<int32_t>(maxN_) : nInList;
        minK_ = std::min(minK_, k);
        workspacesSize_ += static_cast<uint64_t>(k) * static_cast<uint64_t>(n);
      }
      // when minK * baseN * coreNum * sizeof(float16) > 12M, it goes into antiquantPerformance branch (12M is obtained by test).
      int32_t dimMN =
        CeilDiv(CeilDiv(maxM_, groupNum_), baseM_) * CeilDiv(maxN_, baseN_);
      bool goodCubeUtility = dimMN * (xDType_ == ge::DT_BF16 ? 2 : 1) >= static_cast<int32_t>(aicNum * 0.4);  // 0.4: a factor, in practice.
      antiquantPerformance_ =
        goodCubeUtility && static_cast<int64_t>(minK_) * baseN_ * static_cast<int64_t>(aicNum) >= ANTIQUANT_PERFORMANCE_THRESHOLD && !transposeWeight_;
      uint32_t maxUbBaseN = static_cast<uint32_t>(BEST_UB_BASEN);
      if (transposeWeight_) {
        maxUbBaseN = baseN_;
      } else if (antiquantPerformance_) {
        // 2: use 2 pieces of workspace in antiquantPerformance branch
        workspacesSize_ = static_cast<uint64_t>(maxN_) * static_cast<uint64_t>(maxK_) * 2UL;
      }
      // 2: 2 InQueue(antiquant_scale,antiquant_offset)
      ubSize = static_cast<uint32_t>(ubSize_ - 2U * maxUbBaseN * mmDataTypeSize_ * QUEUE_DOUBLE_BUFFER);
      workspaces[0] += workspacesSize_ * mmDataTypeSize_;
    }
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlkAntiquant() {
  if (isA16W8Msd_) {
    ubDivideBlkNum_ = UB_A16W8_MSD_BLOCK_NUM;
    ubIoBlkNum_ = UB_A16W8_MSD_IO_USED_BLOCK;
    ubBlockAlign_ = UB_A16W8_MSD_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  if (xDType_ == ge::DT_FLOAT16 && (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
    if (weightDtype_ == ge::DT_INT8) {
      ubDivideBlkNum_ = UB_A16W8_BLOCK_NUM_FP16;
      ubIoBlkNum_ = UB_A16W8_IO_USED_BLOCK_FP16;
    } else {  // int4
      ubDivideBlkNum_ = UB_A16W4_BLOCK_NUM_FP16;
      ubIoBlkNum_ = UB_A16W4_IO_USED_BLOCK_FP16;
    }
    ubBlockAlign_ = UB_ANTIQUANT_PER_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  if (xDType_ == ge::DT_BF16 && (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
    if (weightDtype_ == ge::DT_INT8) {
      ubDivideBlkNum_ = UB_A16W8_BLOCK_NUM_BF16;
      ubIoBlkNum_ = UB_A16W8_IO_USED_BLOCK_BF16;
    } else {
      ubDivideBlkNum_ = UB_A16W4_BLOCK_NUM_BF16;
      ubIoBlkNum_ = UB_A16W4_IO_USED_BLOCK_BF16;
    }
    ubBlockAlign_ = UB_ANTIQUANT_PER_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_A16W8_H
