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
 * \file grouped_matmul_tiling_a8w8.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_A8W8_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_A8W8_H

#include "grouped_matmul_tiling_utils.h"

namespace optiling {
int32_t GMMTiling::FindBestSingleN(const uint32_t& aicNum) {
  uint64_t quantGroupNum = tilingData.gmmBaseParams.get_quantGroupNum();
  // A8W8模式以及A4W4 Perchannel模式支持动态分块
  if(maxN_ < baseN_ || tuningConfig_ <= 0|| !(isA8W8_ || (isA4W4_ && quantGroupNum == 1)) ) {
    return baseN_;
  }
  int32_t mDim = CeilDiv(tuningConfig_ , baseM_);
  int32_t nDim = CeilDiv(maxN_, baseN_);
  int32_t taskNum = mDim * nDim * static_cast<int32_t>(groupNum_);
  int32_t taskNumPerCore = CeilDiv(taskNum, aicNum);
  // 每个核只需要做1个基本块的时候，任务量太少，无需处理
  if(taskNumPerCore <= 1) {
    return baseN_;
  }
  int32_t curNDim = 0;
  int32_t curTaskNum = 0;
  int32_t bestSingleN = baseN_;
  float ratio = 0;
  for (uint32_t i = 1; i <= aicNum; ++i) {
    if(wFormat_ == matmul_tiling::CubeFormat::NZ) {
      bestSingleN = CeilDiv(static_cast<int32_t>(maxN_), i);
      if(bestSingleN != maxN_ && bestSingleN % baseN_ != 0) {
        continue;
      }
    } else {
      //暂时只NZ格式开启动态分块
      return baseN_;
    }
    curNDim = CeilDiv(maxN_, bestSingleN);
    curTaskNum = mDim * curNDim * static_cast<int32_t>(groupNum_);
    ratio = static_cast<float>(curTaskNum) / AlignUp(static_cast<uint32_t>(curTaskNum), aicNum);
    if (ratio >= EFFECTIVE_TASK_RATIO) {
      return bestSingleN;
    }
  }
  return baseN_;
}

bool GMMTiling::TryFullLoadA(int32_t baseM,const GMMCompileInfo *compileInfoPtr) {
  auto l1Size = compileInfoPtr->l1Size;
  //暂时只支持A8W8和A4W4Perchannel模式
  float sizeofweightDtype = weightDtype_ == ge::DT_INT4 ? 0.5f : 1.0f;
  float sizeofxDtype = xDType_ == ge::DT_INT4 ? 0.5f : 1.0f;

  auto matBl1Size = static_cast<int32_t>(tilingData.mmTilingData.get_depthB1() * baseN_ * baseK_ * sizeofweightDtype);
  auto remainL1Size = l1Size - matBl1Size;
  if(hasBias_) {
    remainL1Size -= BIAS_REMAIN_SPACE;
  }
  int32_t newDepthA1 = CeilDiv(maxK_, baseK_);
  if(static_cast<int32_t>(newDepthA1 * baseM * baseK_ * sizeofxDtype) < static_cast<int32_t>(remainL1Size)) {
    tilingData.mmTilingData.set_stepKa(newDepthA1);
    tilingData.mmTilingData.set_depthA1(newDepthA1);
    return true;
  }
  return false;
}

ge::graphStatus GMMTiling::DynamicTilingSingleN(gert::TilingContext* context, const uint32_t& aicNum, const GMMCompileInfo *compileInfoPtr) {
  OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(
               context->GetNodeName(), "compileInfoPtr is nullptr."), return ge::GRAPH_FAILED);
  if (maxN_ < baseN_ || tuningConfig_ <= 0 || wFormat_ == matmul_tiling::CubeFormat::ND) {
    return ge::GRAPH_SUCCESS;
  }
  int32_t bestSingleN = FindBestSingleN(aicNum);
  if(bestSingleN == baseN_) {//没找到更优的singleN
    return ge::GRAPH_SUCCESS;
  }
  tilingData.gmmBaseParams.set_singleN(bestSingleN);
  //先不改看看baseM能否全载左矩阵
  if(TryFullLoadA(baseM_, compileInfoPtr)) {
    return ge::GRAPH_SUCCESS;
  }
  //可以尝试减小baseM来全载左矩阵
  int32_t newBaseM = static_cast<int32_t>(SixteenAlign(tuningConfig_, true));
  //防止不均匀情况
  newBaseM += MIN_BASE_M;
  //再看看能否全载左矩阵
  if(newBaseM < baseM_ && TryFullLoadA(newBaseM, compileInfoPtr)) {
    tilingData.mmTilingData.set_baseM(newBaseM);
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_SUCCESS;
}

int32_t GMMTiling::FindBestSingleNPertoken(const uint32_t aicNum) const {
  if (CeilDiv(maxN_, baseN_) * groupNum_ <= aicNum) {  // if all matmuls only occupy a part of cores
    return baseN_;
  }
  if (maxN_  >= 2048) {  // 2048: a threshold
    return 1024;  // 1024: max singleN
  }
  int32_t bestSingleN = baseN_;  // init bestSingleN
  uint32_t bestLastCycleCoreNum = (groupNum_ * CeilDiv(maxN_, bestSingleN)) % aicNum;  // init lastCycleCoreNum
  // 1024: max singleN
  for (int32_t tempSingleN = 1024 / baseN_ * baseN_; tempSingleN > baseN_; tempSingleN -= baseN_) {
    uint32_t lastCycleCoreNum = (groupNum_ * CeilDiv(maxN_, tempSingleN)) % aicNum;
    if (lastCycleCoreNum == 0U) {
      bestSingleN = tempSingleN;
      break;
    }
    if (lastCycleCoreNum > bestLastCycleCoreNum ||
      (lastCycleCoreNum == bestLastCycleCoreNum && maxN_ % tempSingleN == 0)) {
      bestSingleN = tempSingleN;
      bestLastCycleCoreNum = lastCycleCoreNum;
    }
  }
  return bestSingleN;
}

void GMMTiling::FindBestUsedCoreNumOneGroup(const uint32_t aicNum) {
  if (groupNum_ > 1U) {
    return;
  }
  uint32_t totalCoreNums = CeilDiv(maxN_, baseN_);
  // 3: if cube iterNum less or equal to 3, and more than half cores are unused in last iter, use less cores each iter
  if ((aicNum * 3U > totalCoreNums && totalCoreNums % aicNum <= aicNum / 2U) || totalCoreNums < aicNum) {  // 2: half of aicNum
    uint32_t  cubeIterNum = CeilDiv(totalCoreNums, aicNum);
    usedCoreNum_ = CeilDiv(totalCoreNums, cubeIterNum);
  }
}

ge::graphStatus GMMTiling::SetWorkspscesPerTokenQuant(const uint32_t aicNum, size_t* workspaces) {
  if (aicNum == 0U) {  // invaild value
    return ge::GRAPH_FAILED;
  }
  bool opt = (maxM_ <= 32 * groupNum_ && wFormat_ == matmul_tiling::CubeFormat::NZ) &&
             (!transposeWeight_ || maxN_ >= 2048);  // 32: a factor, 2048: a threshold.
  if (opt) {  // non-basic strategy. matmul output in non-continugous mode with singleN >= baseN
    int32_t bestSingleN = FindBestSingleNPertoken(aicNum);
    tilingData.gmmBaseParams.set_singleN(bestSingleN);
  }
  if (isA4W4_) {
    if (isA4W4Optimize_) {
      constexpr uint32_t BASEM_OPTIMIZE = 128;
      workspaces[0] += 2UL * BASEM_OPTIMIZE * BEST_BASEN * usedCoreNum_ * sizeof(short); // a4w4 mmout dtype is half
    } else {
      // 4： when do cv parallelism, four pieces of workspace are used for storing four cycles of matmul output
      workspaces[0] += 4UL * baseM_ * baseN_ * usedCoreNum_ * sizeof(short); // a4w4 mmout dtype is half
    }
  } else {
    // 4： when do cv parallelism, four pieces of workspace are used for storing four cycles of matmul output
    workspaces[0] += 4UL * baseM_ * baseN_ * usedCoreNum_ * sizeof(int32_t);
  }

  return ge::GRAPH_SUCCESS;
}

bool GMMTiling::StaticTilingProcess(gert::TilingContext *context) {
  // cond.1 A8W8
  // cond.2 singleX-singleW-singleY scenario
  // cond.3 without bias
  // cond.4 without activation
  // cond.5 no pretiling
  // cond.6 only support typeM
  // cond.7 tilingdata corresponds to expected value
  if ((xDType_ != ge::DT_INT8 || weightDtype_ != ge::DT_INT8) ||
      (isSingleX_ == 0 || isSingleWeight_ == 0 || isSingleY_ == 0) ||
      hasBias_ ||
      actType_ != 0U ||
      tilingData.gmmBaseParams.get_isPreTiling() != 0 ||
      tilingData.gmmBaseParams.get_groupType() != 0 ||
      !CheckTilingMatchStaticValue()) {
    return false;
  }
  // cond.8 only support milan platform
  auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
  OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "GMM tiling: compileInfoPtr is nullptr."),
              return false);
  if (!(compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
        compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910_93)) {
    return false;
  }

  tilingData.gmmBaseParams.set_m(tilingData.mmTilingData.get_M());
  tilingData.gmmBaseParams.set_n(tilingData.mmTilingData.get_N());
  tilingData.gmmBaseParams.set_k(tilingData.mmTilingData.get_Ka());
  return true;
}

bool GMMTiling::CheckTilingMatchStaticValue() {
  if (tilingData.mmTilingData.get_depthA1() != STATIC_TILING_DEPTH_A1_B1 ||
      tilingData.mmTilingData.get_depthB1() != STATIC_TILING_DEPTH_A1_B1 ||
      tilingData.mmTilingData.get_stepM() != 1 ||
      tilingData.mmTilingData.get_stepN() != 1 ||
      tilingData.mmTilingData.get_stepKa() != STATIC_TILING_STEP_KA_KB ||
      tilingData.mmTilingData.get_stepKb() != STATIC_TILING_STEP_KA_KB ||
      tilingData.mmTilingData.get_dbL0A() != DOUBLE_BUFFER_L0A_L0B ||
      tilingData.mmTilingData.get_dbL0B() != DOUBLE_BUFFER_L0A_L0B ||
      tilingData.mmTilingData.get_dbL0C() != 1 ||
      maxK_ > STATIC_TILING_MAX_K) {
    return false;
  }
  if (tilingData.mmTilingData.get_baseM() == BASIC_BLOCK_SIZE_128 &&
      tilingData.mmTilingData.get_baseN() == BASIC_BLOCK_SIZE_256 &&
      tilingData.mmTilingData.get_baseK() == BASIC_BLOCK_SIZE_128) {
        return true;
  }
  return false;
}

bool GMMTiling::IsAivAicRatioTwoRequired() const {
    // Must be valid group type (not 2)
    if (groupListType_ == GROUP_LIST_SPARSE_M) {
      return false;
    }

    // Condition 1: GELU activation (immediate match)
    if (actType_ == ACT_TYPE_GELU) {
      return true;
    }

    // Condition 2: Complex tuning configuration requiring:
    // - K dimension outside normal vectorization range
    // - Minimum tuning configuration threshold
    // - Valid token/group size
    const bool needs_double_vector = (maxK_ <= DOUBLE_VECTOT_THRESHOLD_K_LOWER) ||
                                    (maxK_ >= DOUBLE_VECTOT_THRESHOLD_K_UPPER);
    const bool has_sufficient_tuning = (tuningConfig_ >= SMALL_TUNING_CONFIG_THRESHOLD);
    const bool has_valid_workload = (perTokenOrPerGroupSize_ > 0U);

    return needs_double_vector && has_sufficient_tuning && has_valid_workload;
}

bool GMMTiling::IsFixedAxisMoveCondition() {
    bool isCorrectShape = (maxK_ == FIXAXISMOVE_K1 && maxN_ == FIXAXISMOVE_N1) ||
                          (maxK_ == FIXAXISMOVE_K2 && maxN_ == FIXAXISMOVE_N2);
    bool isGroupCorrect = (groupNum_ == FIXAXISMOVE_GROUP_NUM);
    bool isTuningInRange = (tuningConfig_ >= FIXAXISMOVE_PERM_LOWER) &&
                          (tuningConfig_ <= FIXAXISMOVE_PERM_UPPER);
    bool isDataTypeCorrect = yDtype_ == ge::DT_FLOAT16 && scaleDtype_ == ge::DT_FLOAT && perTokenScaleDtype_ == ge::DT_FLOAT;
    bool isConfigCorrect = !transposeX_ && (splitItem_ == FIXAXISMOVE_SPLIT_ITEM2 || splitItem_ == FIXAXISMOVE_SPLIT_ITEM3)
                          && (groupListType_ == FIXAXISMOVE_GROUP_LIST_TYPE)
                          && (groupType_ == FIXAXISMOVE_GROUP_TYPE) && (actType_ == 0)
                          && !transposeWeight_;
    bool isWorkspaceValid = (static_cast<int64_t>(FixedAxisMoveWorkspace_) <= tuningConfigWorkspace_) ||
                            (tuningConfigWorkspace_ == -1);
    bool isFormatValid = (wFormat_ == matmul_tiling::CubeFormat::NZ);

    return isCorrectShape && isTuningInRange && isGroupCorrect && isA8W8_ &&
           isDataTypeCorrect && isConfigCorrect && isWorkspaceValid && !hasBias_ && isFormatValid;
}

bool GMMTiling::IsIntDataType() const {
    return yDtype_ == ge::DT_INT8 || yDtype_ == ge::DT_INT32;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlkQuant() {
  if ((weightDtype_ == ge::DT_INT8 || isA8W4FakeA8W8_) && (perTokenOrPerGroupSize_ == 1 || actType_ != 0)) {
    // include case per-token without activation, per-token with activation and per-tensor with activation
    ubDivideBlkNum_ = UB_DYNAMIC_QUANT_BLOCK_NUM;
    ubIoBlkNum_ = UB_DUNAMIC_QUANT_IO_USED_BLOCK;
    ubBlockAlign_ = UB_QUANT_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  if ((weightDtype_ == ge::DT_INT8 || isA8W4FakeA8W8_) && perTokenOrPerGroupSize_ != 1) {
    // include case per-tensor without activation
    if (yDtype_ == ge::DT_FLOAT16) {
      ubDivideBlkNum_ = UB_STATIC_QUANT_BLOCK_NUM_FP16;
    } else {
      ubDivideBlkNum_ = UB_STATIC_QUANT_BLOCK_NUM_BF16;
    }
    ubIoBlkNum_ = UB_STATIC_QUANT_IO_USED_BLOCK;
    ubBlockAlign_ = UB_QUANT_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_A8W8_H
