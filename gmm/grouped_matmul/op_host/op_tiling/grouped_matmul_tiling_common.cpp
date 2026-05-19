/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_tiling_common.cpp
 * \brief
 */
#include "grouped_matmul_tiling_a16w4.h"
#include "grouped_matmul_tiling_a16w8.h"
#include "grouped_matmul_tiling_a4w4.h"
#include "grouped_matmul_tiling_a8w4.h"
#include "grouped_matmul_tiling_a8w8.h"
#include "grouped_matmul_tiling_float.h"
#include "grouped_matmul_tiling_key_utils.h"

namespace optiling {
enum class TensorListScenario : uint8_t {
  SSS = 0U,  // single x, single weight, single y
  SSM,
  SMS,
  SMM,
  MSS,
  MSM,
  MMS,
  MMM
};

static inline TensorListScenario MakeTensorListScenario(bool isSingleX, bool isSingleWeight, bool isSingleY)
{
  const uint8_t scenario = (isSingleX ? 4U : 0U) | (isSingleWeight ? 2U : 0U) | (isSingleY ? 1U : 0U);
  switch (scenario) {
    case 7U:
      return TensorListScenario::SSS;
    case 6U:
      return TensorListScenario::SSM;
    case 5U:
      return TensorListScenario::SMS;
    case 4U:
      return TensorListScenario::SMM;
    case 3U:
      return TensorListScenario::MSS;
    case 2U:
      return TensorListScenario::MSM;
    case 1U:
      return TensorListScenario::MMS;
    default:
      return TensorListScenario::MMM;
  }
}

static inline void LogUnsupportedTensorListScenario(const gert::TilingContext* context, int64_t groupType,
                                                    bool isSingleWeight, bool isSingleX, bool isSingleY)
{
  OP_LOGE(context->GetNodeName(),
          "GMM tiling: unsupported tensor-list scenario, groupType=%ld, singleWeight=%d, singleX=%d, singleY=%d.",
          groupType, isSingleWeight, isSingleX, isSingleY);
}


ge::graphStatus GMMTiling::CheckWeightNZShape(const gert::TilingContext* context, int64_t numInOneBlk) const {
  OP_CHECK_IF(numInOneBlk <= 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "numInOneBlk, the "
             "input of CheckWeightNZShape has an invaild value %ld", numInOneBlk), return ge::GRAPH_FAILED);
  size_t i = 0;
  while (true) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i++);
    if (wTensor == nullptr) { break; }
    gert::Shape wOriginShape = wTensor->GetOriginShape();
    int64_t lastDimValue = wOriginShape.GetDim(wOriginShape.GetDimNum() - 1);  // inner axis
    OP_CHECK_IF(lastDimValue % numInOneBlk != 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
               "the inner axis size of nz weight is expected to be a multiple of 32B, "
               "but now the inner axis size is %ld.", lastDimValue),
               return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::CheckMKN(const gert::TilingContext* context) {
  mmDataTypeSize_ = GetSizeByDataType(mmDType_);
  OP_CHECK_IF(mmDataTypeSize_ == 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMM get mm dtype[%s] size is 0.", TypeUtils::DataTypeToAscendString(mmDType_).GetString()),
             return ge::GRAPH_FAILED);
  uint32_t numInOneBlk = 0;
  if (isA4W4_) {
    numInOneBlk = static_cast<uint32_t>(ONE_BLK_SIZE / INT4_DATA_TYPE_SIZE);
  } else {
    numInOneBlk = ONE_BLK_SIZE / mmDataTypeSize_;
  }
  OP_CHECK_IF(numInOneBlk == 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMM numInOneBlk cannot be 0."), return ge::GRAPH_FAILED);
  int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
  OP_CHECK_IF(maxM_ > maxMKN || maxN_ > maxMKN || maxK_ > maxMKN,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "32B-aligned m, n or k axis is out of range int32!"),
             return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

void GMMTiling::SetTilingDataIsSingleTensor() {
  tilingData.gmmBaseParams.set_singleWeight(static_cast<uint32_t>(isSingleWeight_));
  tilingData.gmmBaseParams.set_singleX(static_cast<uint32_t>(isSingleX_));
  tilingData.gmmBaseParams.set_singleY(static_cast<uint32_t>(isSingleY_));
}

bool GMMTiling::CheckTensorListLength(const gert::TilingContext *context)
{
  auto xTensor = context->GetDynamicInputTensor(X_INDEX, MAX_TENSOR_CONT);
  auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, MAX_TENSOR_CONT);
  OP_CHECK_IF(xTensor != nullptr || wTensor != nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM tensorList length should between [1, %d]",
                                          MAX_TENSOR_CONT),
              return false);
  return true;
}

ge::graphStatus GMMTiling::PrepareTilingData(const gert::TilingContext* context) {
  // get transpose and groupType
  OP_CHECK_IF(GMMGetAttrs(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMMGetAttrs failed"),
             return ge::GRAPH_FAILED);
  // get the first tensor's shape of weight and x
  auto xTensor = context->GetDynamicInputTensor(X_INDEX, 0);  // 0: get first tensor
  OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
  gert::Shape xShape = xTensor->GetStorageShape();
  xDimNum_ = static_cast<uint32_t>(xShape.GetDimNum());
  xKDim_ = transposeX_ ? 0U : xDimNum_ - 1U;  // 0: when x is transposed, the first dim is k; -1：otherwise, the last dim is k

  auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, wTensor);
  gert::Shape wShape = wTensor->GetOriginShape();
  uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
  weightNDim_ = transposeWeight_ ? wDimNum - 2U : wDimNum - 1U;  // -2: when w is transposed, the last 2 dim is n; -1: otherwise, the last dim is n
  weightKDim_ = transposeWeight_ ? wDimNum - 1U : wDimNum - 2U;  // -2: when w is transposed, the last 1 dim is k; -1: otherwise, the last 2 dim is k
  nzFactor_ = 1;  // init
  if (wFormat_ == matmul_tiling::CubeFormat::NZ) {
    uint32_t numInOneBlk = isA4W4_ ? static_cast<uint32_t>(UB_BLOCK_UNIT_SIZE / INT4_DATA_TYPE_SIZE) :
                           UB_BLOCK_UNIT_SIZE / std::max(1, GetSizeByDataType(weightDtype_));
    if (isA8W4FakeA8W8_) {
      numInOneBlk = UB_BLOCK_UNIT_SIZE;
    }
    if (wDimNum >= 4U) {  // 4: least dim num of nz format tensor
      weightNDim_ = transposeWeight_ ? wDimNum - 3U : wDimNum - 4U;  // -3: when w is transposed, the last 3 dim is n/nzFactor; -4: when w has nz format, the last 4 dim is n/nzFactor
      // nzFactor_ is a factor used to compute n axis size. If weight is transposed, nzFactor_ is 16; otherwise nzFactor_ is 16 for bf16, 32 for int8
      nzFactor_ = transposeWeight_ ? 16 : static_cast<int32_t>(numInOneBlk);
    } else {
      OP_CHECK_IF(CheckWeightNZShape(context, static_cast<int64_t>(numInOneBlk)) != ge::GRAPH_SUCCESS,
                 OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the shape of nz weight is invaild."),
                 return ge::GRAPH_FAILED);
    }
  }
  isSingleWeight_ = (context->GetDynamicInputTensor(WEIGHT_INDEX, 1) == nullptr);
  isSingleX_ = (context->GetDynamicInputTensor(X_INDEX, 1) == nullptr);
  isSingleY_ = (splitItem_ == 2 || splitItem_ == 3);  // 2: when x is multi-tensor, y is single-tensor; 3: when x is single-tensor, y is single-tensor
  SetTilingDataIsSingleTensor();

  if (groupType_ == SPLIT_M) {
    return GMMGetTensorShapeSplitM(context, xShape, wShape);
  }
  if (groupType_ == SPLIT_K) {
    return GMMGetTensorShapeSplitK(context, xShape, wShape);
  }
  if (groupType_ == NO_SPLIT) {  // not split any axis
    if (isSingleWeight_ && wDimNum > 2U) {  // 2: dim of splited weight tensor
      return SeparatedXSingleWeight(context, wShape);
    }
    return SeparatedXSeparatedWeight(context);
  }
  LogUnsupportedTensorListScenario(context, groupType_, isSingleWeight_, isSingleX_, isSingleY_);
  return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMGetTensorShapeSplitM(const gert::TilingContext* context, const gert::Shape &xShape,
    const gert::Shape &wShape) {
    switch (MakeTensorListScenario(isSingleX_, isSingleWeight_, isSingleY_)) {
      case TensorListScenario::SSS:
        return SplitMSingleXSingleWeightSingleY(xShape, wShape);
      case TensorListScenario::SMS:
      case TensorListScenario::SMM:
        return SplitMSingleXSeparatedWeight(context, xShape);
      case TensorListScenario::MSS:
      case TensorListScenario::MSM:
        return SeparatedXSingleWeight(context, wShape);
      case TensorListScenario::MMS:
      case TensorListScenario::MMM:
        return SeparatedXSeparatedWeight(context);
      default:
        break;
    }
    LogUnsupportedTensorListScenario(context, groupType_, isSingleWeight_, isSingleX_, isSingleY_);
    return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMGetTensorShapeSplitK(const gert::TilingContext* context, const gert::Shape &xShape,
    const gert::Shape &wShape) {
    switch (MakeTensorListScenario(isSingleX_, isSingleWeight_, isSingleY_)) {
      case TensorListScenario::SSS:
        return SplitKSingleXSingleWeightSingleY(context, xShape, wShape);
      case TensorListScenario::SMM:
        return SplitKSingleXSeparatedWeight(context, xShape, wShape);
      case TensorListScenario::MSS:
      case TensorListScenario::MSM:
        return SeparatedXSingleWeight(context, wShape);
      default:
        break;
    }
    LogUnsupportedTensorListScenario(context, groupType_, isSingleWeight_, isSingleX_, isSingleY_);
    return ge::GRAPH_FAILED;
}

/** @brief split M：single-single-single(s-s-s)
*/
ge::graphStatus GMMTiling::SplitMSingleXSingleWeightSingleY(const gert::Shape &xShape, const gert::Shape &wShape) {
  groupNum_ = static_cast<int32_t>(wShape.GetDim(0));
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  int64_t n = wShape.GetDim(weightNDim_) * static_cast<int64_t>(nzFactor_);
  kList_[0] = static_cast<int32_t>(k);  // if split M axis, the K axis values of x tensorList are all the same.
  nList_[0] = static_cast<int32_t>(n);
  mList_[0] = -1;
  maxM_ = m;
  maxK_ = k;
  maxN_ = n;
  totalM_ = static_cast<uint32_t>(m);
  FixedAxisMoveWorkspace_ = maxM_ * maxN_ * sizeof(int32_t);
  return ge::GRAPH_SUCCESS;
}

/** @brief split M：single-multi-single(s-m-s)/single-multi-multi(s-m-m), share the same function.
*/
ge::graphStatus GMMTiling::SplitMSingleXSeparatedWeight(const gert::TilingContext* context, const gert::Shape &xShape) {
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i);
    if (wTensor == nullptr) { break; }  // when x has multi tensors, xTensor is allowed to be empty
    auto wShape = wTensor->GetOriginShape();

    groupNum_ += 1U;
    kList_[i] = static_cast<int32_t>(k);
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
    nList_[i] = static_cast<int32_t>(n);
    maxN_ = std::max(maxN_, n);
  }
  mList_[0] = -1;  // mList is unknown right now
  maxM_ = m;
  maxK_ = k;
  totalM_ = static_cast<uint32_t>(m);

  return ge::GRAPH_SUCCESS;
}

/** @brief split M：multi-multi-single(m-m-s); no split: multi-multi-multi(m-m-m), share the same function
*/
ge::graphStatus GMMTiling::SeparatedXSeparatedWeight(const gert::TilingContext* context) {
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i);
    auto xTensor = context->GetDynamicInputTensor(X_INDEX, i);
    if (wTensor == nullptr || xTensor == nullptr) { break; }
    auto wShape = wTensor->GetOriginShape();
    auto xShape = xTensor->GetStorageShape();
    groupNum_ += 1U;
    int64_t m = GMMGetBS(xShape);
    int64_t k = xShape.GetDim(xKDim_);
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
    mList_[i] = static_cast<int32_t>(m);
    kList_[i] = static_cast<int32_t>(k);
    nList_[i] = static_cast<int32_t>(n);
    maxM_ = std::max(maxM_, m);
    maxK_ = std::max(maxK_, k);
    maxN_ = std::max(maxN_, n);
    totalM_ += static_cast<uint32_t>(m);
  }
  groupType_ = NO_SPLIT;
  return ge::GRAPH_SUCCESS;
}

/** @brief split M : multi-single-multi(m-s-m), split K : multi-single-multi(m-s-m), share the same function
*/
ge::graphStatus GMMTiling::SeparatedXSingleWeight(const gert::TilingContext* context, const gert::Shape &wShape) {
  int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto xTensor = context->GetDynamicInputTensor(X_INDEX, i);
    if (xTensor == nullptr) { break; }  // when x has multi tensors, xTensor is allowed to be empty
    auto xShape = xTensor->GetStorageShape();
    groupNum_ += 1U;
    int64_t m = GMMGetBS(xShape);
    int64_t k = xShape.GetDim(xKDim_);
    mList_[i] = static_cast<int32_t>(m);
    kList_[i] = static_cast<int32_t>(k);
    nList_[i] = static_cast<int32_t>(n);
    maxM_ = std::max(maxM_, m);
    maxK_ = std::max(maxK_, k);
    totalM_ += static_cast<uint32_t>(m);
  }
  maxN_ = n;
  groupType_ = NO_SPLIT;
  return ge::GRAPH_SUCCESS;
}

/** @brief split K single-single-single
*/
ge::graphStatus GMMTiling::SplitKSingleXSingleWeightSingleY(const gert::TilingContext* context,
    const gert::Shape &xShape, const gert::Shape &wShape) {
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;

  auto groupListTensor = context->GetDynamicInputTensor(GROUPLIST_INDEX, 0);
  if (groupListTensor == nullptr) {
    OP_LOGE(context->GetNodeName(), "GMM tiling: groupList tensor is nullptr.");
    return ge::GRAPH_FAILED;
  }
  gert::Shape groupListShape = groupListTensor->GetStorageShape();
  groupNum_ = static_cast<int32_t>(groupListShape.GetDim(0));  // 0: the first dim of groupList is groupNum
  mList_[0] = static_cast<int32_t>(m);
  nList_[0] = static_cast<int32_t>(n);
  kList_[0] = -1;
  maxM_ = m;
  maxN_ = n;
  maxK_ = k;
  totalM_ = static_cast<uint32_t>(m);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::SplitKSingleXSeparatedWeight(const gert::TilingContext* context,
  const gert::Shape &xShape, const gert::Shape &wShape) {
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i);
    if (wTensor == nullptr) { break; }
    auto wTensorShape = wTensor->GetOriginShape();
    groupNum_ += 1;
    mList_[i] = static_cast<int32_t>(m);
    k = wTensorShape.GetDim(weightKDim_);
    kList_[i] = static_cast<int32_t>(k);
    maxK_ = std::max(maxK_, k);
    n = wTensorShape.GetDim(weightNDim_);
    nList_[i] = static_cast<int32_t>(n);
    maxN_ = std::max(maxN_, n);
  }
  maxM_ = m;
  totalM_ = static_cast<uint32_t>(m);
  groupType_ = NO_SPLIT;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::Init(const gert::TilingContext* context) {
  OP_CHECK_IF(!CheckTensorListLength(context),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Check tensorList length failed."),
             return ge::GRAPH_FAILED);
  OP_CHECK_IF(PrepareTilingData(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM PrepareTilingData failed."),
             return ge::GRAPH_FAILED);
  auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);  // check compileInfoPtr is not null

  // check tuningConfig_
  if (tuningConfig_ < 0 || tuningConfig_ > maxM_) {
    OP_LOGE(context->GetNodeName(), "GMM tiling: invalid tuningConfig=%ld, expected range is [0, %ld].",
            tuningConfig_, maxM_);
    return ge::GRAPH_FAILED;
  }
  // check whether x, weight and y are all single tensor
  isAllSingleTensor_ = isSingleX_ && isSingleWeight_ && isSingleY_;
  bool isA16W8 = (xDType_ == ge::DT_FLOAT16 || xDType_ == ge::DT_BF16) && weightDtype_ == ge::DT_INT8;
  // check whether k and n are supported in msd
  bool isKNForA16W8MSD = maxN_ % static_cast<int64_t>(A16W8_MSD_KN_BASE_BLOCK) == 0L &&
                         maxK_ % static_cast<int64_t>(A16W8_MSD_KN_BASE_BLOCK) == 0L &&
                         maxK_ <= static_cast<int64_t>(A16W8_MSD_MAX_K) &&
                         maxN_ >= static_cast<int64_t>(A16W8_MSD_MIN_N);
  // check whether total token num and average token num are supported in msd
  bool isMForA16W8MSD = totalM_ <= A16W8_MSD_AVERAGE_TOKEN_NUM * groupNum_;
  isA16W8Msd_ = isAllSingleTensor_ && groupType_ == SPLIT_M && isA16W8 && isKNForA16W8MSD && isMForA16W8MSD;
  mmDType_ = isA16W8Msd_ ? ge::DT_INT8 : xDType_;
  OP_CHECK_IF(CheckMKN(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM CheckMKN failed."),
             return ge::GRAPH_FAILED);
  auto biasPtr = context->GetDynamicInputTensor(BIAS_INDEX, 0);  // 0: obtain the first tensor of the tensorList
  hasBias_ = !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);
  if (isA4W4_) {
    uint64_t quantGroupNum = 0;
    uint32_t scaleDimNum = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape().GetDimNum();
    // 3: pergroup scale shape is [e,g,n]
    if (scaleDimNum == 3U) {
      quantGroupNum = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape().GetDim(1);
      quantGroupSize_ = quantGroupNum == 0 ? 0U : maxK_ / quantGroupNum;
      isPerGroup_ = true;
    // 2: perchannel scale shape is [e, n]
    } else if (scaleDimNum == 2U) {
      quantGroupNum = 1UL;
    } else {
      OP_LOGE(context->GetNodeName(), "GMM tiling: invalid A4W4 scale dim=%u, expected 2 or 3.", scaleDimNum);
      return ge::GRAPH_FAILED;
    }
    tilingData.gmmBaseParams.set_k(maxK_);
    tilingData.gmmBaseParams.set_n(maxN_);
    tilingData.gmmBaseParams.set_quantGroupNum(quantGroupNum);
    isA4W4Optimize_ = IsA4W4OptimizeCondition();
    if (isA4W4Optimize_) {
      tilingData.gmmBaseParams.set_isA4W4Optimize(1);
    }
  }
  if (isA8W4FakeA8W8_) {
    hasBias_ = false;
  }
  tilingData.gmmArray.set_mList(mList_);
  tilingData.gmmArray.set_kList(kList_);
  tilingData.gmmArray.set_nList(nList_);
  tilingData.gmmBaseParams.set_groupNum(groupNum_);
  tilingData.gmmBaseParams.set_m(totalM_);
  tilingData.gmmBaseParams.set_hasBias(static_cast<uint32_t>(hasBias_));
  tilingData.gmmBaseParams.set_groupType(static_cast<int32_t>(groupType_));
  tilingData.gmmBaseParams.set_activeType(actType_);
  tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
  tilingData.gmmBaseParams.set_groupListType(groupListType_);
  tilingData.gmmBaseParams.set_k(maxK_);
  tilingData.gmmBaseParams.set_n(maxN_);
  OP_LOGI(context->GetNodeName(), "GMM_tiling: groupNum_ is %u, maxM_ is %ld, maxK_ is %ld, maxN_ is %ld.",
            groupNum_, maxM_, maxK_, maxN_);
  return ge::GRAPH_SUCCESS;
}

/*
1、输出默认会存储在L2Cache中，可以考虑设置输出不走L2Cache从而提高输入的L2Cache命中率
2、后续算子可能会用到当前算子的输出，若当前输出不在L2Cache中可能会影响后续算子效率
3、当前算子输出如果超过L2Cache大小时，实测不会对后续算子造成影响，此时可以跳过L2Cache
*/
ge::graphStatus GMMTiling::IsOutputDisableL2Cache(gert::TilingContext* context, const GMMCompileInfo *compileInfoPtr) {
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(
                context->GetNodeName(), "compileInfoPtr is nullptr."), return ge::GRAPH_FAILED);
    //仅切K单x单weight场景可以Tiling侧知道输出矩阵大小
    if(groupType_ != SPLIT_K || !isSingleX_ || !isSingleWeight_) {
        tilingData.gmmBaseParams.set_isOutputDisableL2Cache(0);
        return ge::GRAPH_SUCCESS;
    }
    auto outputSize = static_cast<uint64_t>(GetSizeByDataType(yDtype_) * maxM_ * maxN_ * groupNum_);
    if(outputSize > compileInfoPtr->l2Size) {
        tilingData.gmmBaseParams.set_isOutputDisableL2Cache(1);
    } else {
        tilingData.gmmBaseParams.set_isOutputDisableL2Cache(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::DivideUbAndSetWorkspace(gert::TilingContext* context, const uint32_t& aicNum) {
  size_t* workspaces = context->GetWorkspaceSizes(1);  // get second variable
  OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);  // check workspaces is not null
  workspaces[0] = SYS_WORKSPACE_SIZE;  // default size
  if (weightDtype_ != ge::DT_INT8 && weightDtype_ != ge::DT_INT4) {
    return ge::GRAPH_SUCCESS;
  }
  uint32_t ubSize = static_cast<uint32_t>(ubSize_);
  if ((xDType_ == ge::DT_BF16 || xDType_ == ge::DT_FLOAT16)) {
    DivideUbAndSetWorkspaceAntiquant(workspaces, aicNum, ubSize);
    OP_CHECK_IF(GetPerGroupNum(context) != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(
               context->GetNodeName(), "GetPerGroupNum failed."), return ge::GRAPH_FAILED);
  } else if (xDType_ == ge::DT_INT8) {
    isFixedAxisMove_ = IsFixedAxisMoveCondition();
    if (isFixedAxisMove_) {
          workspaces[0] += FixedAxisMoveWorkspace_;
    } else {
      // if tuningConfig_ in [1,256], recompute coreNum
      constexpr int32_t tuningConfigLowerLimit = 1;
      constexpr int32_t tuningConfigUpperLimit = 256;
      if (tuningConfig_ >= tuningConfigLowerLimit && tuningConfig_ <= tuningConfigUpperLimit) {
        FindBestUsedCoreNumOneGroup(aicNum);
      }
      if (yDtype_ == ge::DT_INT32) {
        return ge::GRAPH_SUCCESS;
      }
      uint32_t scaleDataTypeSize = GetSizeByDataType(scaleDtype_);
      ubSize = perTokenOrPerGroupSize_ == 1U ?  // is perToken
        static_cast<uint32_t>(ubSize_ -
                              (static_cast<uint64_t>(baseN_) * scaleDataTypeSize +
                              static_cast<uint64_t>(baseM_) * sizeof(float)) * QUEUE_DOUBLE_BUFFER) :
        static_cast<uint32_t>(ubSize_ - baseN_ * scaleDataTypeSize * QUEUE_DOUBLE_BUFFER);
      OP_CHECK_IF(SetWorkspscesPerTokenQuant(aicNum, workspaces) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetWorkspscesPerTokenQuant failed."),
                return ge::GRAPH_FAILED);
      if (isA8W4FakeA8W8_) {
        workspaces[0] += A8W4noMsdSpace_;
      }
    }
  } else if (xDType_ == ge::DT_INT4) {
    ubSize = perTokenOrPerGroupSize_ == 1U ?  // is perToken
      static_cast<uint32_t>(ubSize_ - (static_cast<uint32_t>(baseM_) * sizeof(float)) * QUEUE_DOUBLE_BUFFER) : ubSize_;
    OP_CHECK_IF(SetWorkspscesPerTokenQuant(aicNum, workspaces) != ge::GRAPH_SUCCESS,
      OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetWorkspscesPerTokenQuant failed."),
      return ge::GRAPH_FAILED);
  }
  OP_CHECK_IF(GMMSetUbDivideBlk() != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMMSetUbDivideBlk failed."),
             return ge::GRAPH_FAILED);
  OP_CHECK_IF(GMMCalUbSize(context, ubSize) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMMCalUbSize failed."),
             return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}


ge::graphStatus GMMTiling::RunFusionKernelTiling(gert::TilingContext* context) {
  Reset(context);
  return DoTiling();
}

void GMMTiling::Reset(gert::TilingContext* context)
{
  Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
  compileInfoPtr_ = nullptr;
  aicNum_ = 0U;
  isSpecialTilingDone_ = false;
}

bool GMMTiling::IsCapable()
{
  return IsScenarioMatched();
}

ge::graphStatus GMMTiling::GetShapeAttrsInfo()
{
  OP_CHECK_IF(Init(context_) != ge::GRAPH_SUCCESS,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GMM tiling init failed"),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GetPlatformInfo()
{
  OP_LOGI(context_->GetNodeName(), "Begin Run GMM %s Tiling", GetScenarioName());
  compileInfoPtr_ = context_->GetCompileInfo<GMMCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr_);
  ubSize_ = compileInfoPtr_->ubSize;
  aicNum_ = compileInfoPtr_->aicNum;
  OP_CHECK_IF(aicNum_ == 0U,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GMM tiling: aicNum cannot be 0."),
              return ge::GRAPH_FAILED);
  usedCoreNum_ = aicNum_;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::DoOpTiling()
{
  return DoScenarioOpTiling();
}

ge::graphStatus GMMTiling::DoLibApiTiling()
{
  if (IsTilingComplete()) {
    return ge::GRAPH_SUCCESS;
  }
  return DoScenarioLibApiTiling();
}

ge::graphStatus GMMTiling::GetWorkspaceSize()
{
  if (IsTilingComplete()) {
    return ge::GRAPH_SUCCESS;
  }
  return GetScenarioWorkspaceSize();
}

ge::graphStatus GMMTiling::PostTiling()
{
  if (IsTilingComplete()) {
    return ge::GRAPH_SUCCESS;
  }
  return PostScenarioTiling();
}

uint64_t GMMTiling::GetTilingKey() const
{
  return context_ == nullptr ? 0UL : context_->GetTilingKey();
}

void GMMTiling::DumpTilingInfo()
{
  if (IsTilingComplete()) {
    return;
  }
  DumpScenarioTilingInfo();
}

const char *GMMTiling::GetScenarioName() const
{
  return "Common";
}

bool GMMTiling::IsScenarioMatched() const
{
  return true;
}

bool GMMTiling::IsTilingComplete() const
{
  return isSpecialTilingDone_;
}

ge::graphStatus GMMTiling::DoScenarioOpTiling()
{
  OP_CHECK_IF(CalMMTiling(context_, compileInfoPtr_) != ge::GRAPH_SUCCESS,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GMM CalMMTiling failed"),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::DoScenarioLibApiTiling()
{
  OP_CHECK_IF(GMMSetMMTiling(context_, compileInfoPtr_) != ge::GRAPH_SUCCESS,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GMM GMMSetMMTiling failed"),
              return ge::GRAPH_FAILED);
  tilingData.gmmBaseParams.set_singleN(0);
  FullLoadK(compileInfoPtr_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GetScenarioWorkspaceSize()
{
  OP_CHECK_IF(DivideUbAndSetWorkspace(context_, aicNum_) != ge::GRAPH_SUCCESS,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GMM DivideUbAndSetWorkspace failed"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(DynamicTilingSingleN(context_, usedCoreNum_, compileInfoPtr_) != ge::GRAPH_SUCCESS,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GMM DynamicTilingSingleN failed"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(IsOutputDisableL2Cache(context_, compileInfoPtr_) != ge::GRAPH_SUCCESS,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GMM IsOutputDisableL2Cache failed"),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::PostScenarioTiling()
{
  tilingData.gmmBaseParams.set_workspaceSize(workspacesSize_);
  tilingData.mmTilingData.set_usedCoreNum(usedCoreNum_);
  tilingData.gmmBaseParams.set_coreNum(usedCoreNum_);
  GMMSetTplTilingKey(context_);
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->SetBlockDim(usedCoreNum_);
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

void GMMTiling::DumpScenarioTilingInfo()
{
  PrintTilingInfo(context_);
}

uint32_t GMMTiling::GetScenarioA8W4KernelTemplate() const
{
  return GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE;
}

uint32_t GMMTiling::GetScenarioA16W8KernelTemplate() const
{
  return GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE;
}

uint32_t GMMTiling::GetScenarioAivAicRatio() const
{
  return antiquantPerformance_ ? static_cast<uint32_t>(GROUPED_MATMUL_AIV_AIC_RATIO_2) :
                                 static_cast<uint32_t>(GROUPED_MATMUL_AIV_AIC_RATIO_1);
}

uint32_t GMMTiling::GetScenarioEnableFixedAxis() const
{
  return 0U;
}

bool GMMTiling::ShouldSetScenarioScheduleMode() const
{
  return isA16W8Msd_ || antiquantPerformance_;
}

ge::graphStatus GMMTiling::SetScenarioUbDivideBlk()
{
  return GMMSetUbDivideBlkAntiquant();
}

int32_t GMMTiling::GetScenarioBaseN(bool tuningConfigFlag) const
{
  (void)tuningConfigFlag;
  return BEST_BASEN;
}

bool GMMTiling::TryScenarioSpecificBaseTiling(const GMMCompileInfo *compileInfoPtr, bool tuningConfigFlag)
{
  (void)compileInfoPtr;
  (void)tuningConfigFlag;
  return false;
}

int32_t GMMTiling::CalcScenarioBaseK(const GMMCompileInfo *compileInfoPtr) const
{
  int32_t baseK = static_cast<int32_t>((compileInfoPtr->l0BSize / DOUBLE_BUFFER_L0A_L0B) /
                                       (static_cast<uint32_t>(baseN_) * mmDataTypeSize_));
  return static_cast<int32_t>(SixteenAlign(static_cast<int64_t>(baseK)));
}

uint32_t GMMTiling::CalcScenarioBaseM(const GMMCompileInfo *compileInfoPtr, uint32_t maxBaseM) const
{
  return std::min<uint32_t>((compileInfoPtr->l0ASize / DOUBLE_BUFFER_L0A_L0B) /
                            (static_cast<uint32_t>(baseK_) * mmDataTypeSize_), maxBaseM);
}

int32_t GMMTiling::AlignScenarioBaseM(uint32_t baseM) const
{
  return baseM > maxM_ ? SixteenAlign(maxM_, true) : SixteenAlign(baseM);
}

uint32_t GMMTiling::CalcScenarioStepKa(uint64_t l1ASize) const
{
  return static_cast<uint32_t>((l1ASize / 2UL) /
                               (static_cast<uint64_t>(baseM_) * baseK_ * mmDataTypeSize_));
}

uint32_t GMMTiling::CalcScenarioStepKb(uint64_t l1BSize) const
{
  return static_cast<uint32_t>((l1BSize / 2UL) /
                               (static_cast<uint64_t>(baseN_) * baseK_ * mmDataTypeSize_));
}

int64_t GMMTiling::GetScenarioMInMM() const
{
  return maxM_;
}

bool GMMTiling::ShouldReserveQuantL1N() const
{
  return false;
}

bool GMMTiling::ShouldDisableBiasInMMTiling() const
{
  return false;
}

bool GMMTiling::CanUsePreTiling() const
{
  return false;
}

namespace {
bool IsGMMFloatDType(ge::DataType dtype)
{
  return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT;
}

bool IsGMMA16DType(ge::DataType dtype)
{
  return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}
}  // namespace

ge::graphStatus GMMA8W4MsdTiling::GetShapeAttrsInfo()
{
  auto xDesc = context_->GetDynamicInputDesc(X_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
  auto w0Desc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, w0Desc);
  xDType_ = xDesc->GetDataType();
  weightDtype_ = w0Desc->GetDataType();
  return ge::GRAPH_SUCCESS;
}

const char *GMMA8W4MsdTiling::GetScenarioName() const
{
  return "A8W4MSD";
}

bool GMMA8W4MsdTiling::IsScenarioMatched() const
{
  return xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4;
}

ge::graphStatus GMMA8W4MsdTiling::DoScenarioOpTiling()
{
  ge::graphStatus result = A8W4MsdTiling(context_, compileInfoPtr_);
  if (result == ge::GRAPH_SUCCESS) {
    isSpecialTilingDone_ = true;
  }
  return result;
}

const char *GMMA8W4FakeTiling::GetScenarioName() const
{
  return "A8W4Fake";
}

bool GMMA8W4FakeTiling::IsScenarioMatched() const
{
  return xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4;
}

ge::graphStatus GMMA8W4FakeTiling::DoScenarioOpTiling()
{
  return A8W4FakeTiling(context_, compileInfoPtr_);
}

uint32_t GMMA8W4FakeTiling::GetScenarioA8W4KernelTemplate() const
{
  return isA8W4FakeA8W8_ ? static_cast<uint32_t>(GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERCHANNEL_ANTIQUANT) :
                           static_cast<uint32_t>(GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE);
}

ge::graphStatus GMMA8W4FakeTiling::SetScenarioUbDivideBlk()
{
  return GMMSetUbDivideBlkQuant();
}

bool GMMA8W4FakeTiling::TryScenarioSpecificBaseTiling(const GMMCompileInfo *compileInfoPtr, bool tuningConfigFlag)
{
  (void)compileInfoPtr;
  if (!tuningConfigFlag) {
    return false;
  }
  baseN_ = BEST_BASEN_QUANT_ONE_GROUP;
  baseM_ = BEST_BASEM_QUANT_ONE_GROUP;
  baseK_ = BEST_BASEK_QUANT_ONE_GROUP;
  baseM_ = baseM_ > maxM_ ? static_cast<int32_t>(SixteenAlign(maxM_, true)) : baseM_;
  return true;
}

bool GMMA8W4FakeTiling::ShouldReserveQuantL1N() const
{
  return true;
}

bool GMMA8W4FakeTiling::CanUsePreTiling() const
{
  return true;
}

const char *GMMA8W8Tiling::GetScenarioName() const
{
  return "A8W8";
}

bool GMMA8W8Tiling::IsScenarioMatched() const
{
  return xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT8;
}

uint32_t GMMA8W8Tiling::GetScenarioAivAicRatio() const
{
  if (isFixedAxisMove_) {
    return static_cast<uint32_t>(GROUPED_MATMUL_AIV_AIC_RATIO_1);
  }
  if (IsAivAicRatioTwoRequired()) {
    return static_cast<uint32_t>(GROUPED_MATMUL_AIV_AIC_RATIO_2);
  }
  if (IsIntDataType()) {
    return static_cast<uint32_t>(GROUPED_MATMUL_CUBE_ONLY);
  }
  return static_cast<uint32_t>(GROUPED_MATMUL_AIV_AIC_RATIO_1);
}

uint32_t GMMA8W8Tiling::GetScenarioEnableFixedAxis() const
{
  return isFixedAxisMove_ ? 1U : 0U;
}

ge::graphStatus GMMA8W8Tiling::SetScenarioUbDivideBlk()
{
  return GMMSetUbDivideBlkQuant();
}

bool GMMA8W8Tiling::TryScenarioSpecificBaseTiling(const GMMCompileInfo *compileInfoPtr, bool tuningConfigFlag)
{
  (void)compileInfoPtr;
  if (!tuningConfigFlag) {
    return false;
  }
  baseN_ = BEST_BASEN_QUANT_ONE_GROUP;
  baseM_ = BEST_BASEM_QUANT_ONE_GROUP;
  baseK_ = BEST_BASEK_QUANT_ONE_GROUP;
  baseM_ = baseM_ > maxM_ ? static_cast<int32_t>(SixteenAlign(maxM_, true)) : baseM_;
  return true;
}

bool GMMA8W8Tiling::ShouldReserveQuantL1N() const
{
  return true;
}

bool GMMA8W8Tiling::CanUsePreTiling() const
{
  return true;
}

const char *GMMA4W4Tiling::GetScenarioName() const
{
  return "A4W4";
}

bool GMMA4W4Tiling::IsScenarioMatched() const
{
  return xDType_ == ge::DT_INT4 && weightDtype_ == ge::DT_INT4;
}

uint32_t GMMA4W4Tiling::GetScenarioAivAicRatio() const
{
  return static_cast<uint32_t>(GROUPED_MATMUL_AIV_AIC_RATIO_2);
}

ge::graphStatus GMMA4W4Tiling::SetScenarioUbDivideBlk()
{
  return GMMSetUbDivideBlkA4W4();
}

int32_t GMMA4W4Tiling::GetScenarioBaseN(bool tuningConfigFlag) const
{
  (void)tuningConfigFlag;
  return tuningConfig_ > 64 ? BEST_BASEN : BEST_BASEN_A4W4;
}

int32_t GMMA4W4Tiling::CalcScenarioBaseK(const GMMCompileInfo *compileInfoPtr) const
{
  int32_t baseK = static_cast<int32_t>((compileInfoPtr->l0BSize / DOUBLE_BUFFER_L0A_L0B) /
                                       static_cast<uint32_t>(baseN_ * INT4_DATA_TYPE_SIZE));
  return static_cast<int32_t>(SixteenAlign(static_cast<int64_t>(baseK)));
}

uint32_t GMMA4W4Tiling::CalcScenarioBaseM(const GMMCompileInfo *compileInfoPtr, uint32_t maxBaseM) const
{
  return std::min<uint32_t>((compileInfoPtr->l0ASize / DOUBLE_BUFFER_L0A_L0B) /
                            static_cast<uint32_t>(baseK_ * INT4_DATA_TYPE_SIZE), maxBaseM);
}

uint32_t GMMA4W4Tiling::CalcScenarioStepKa(uint64_t l1ASize) const
{
  return static_cast<uint32_t>((l1ASize / 2UL) /
                               static_cast<uint64_t>(INT4_DATA_TYPE_SIZE * static_cast<float>(baseM_) *
                                                     static_cast<float>(baseK_)));
}

uint32_t GMMA4W4Tiling::CalcScenarioStepKb(uint64_t l1BSize) const
{
  return static_cast<uint32_t>((l1BSize / 2UL) /
                               static_cast<uint64_t>(INT4_DATA_TYPE_SIZE * static_cast<float>(baseN_) *
                                                     static_cast<float>(baseK_)));
}

bool GMMA4W4Tiling::ShouldDisableBiasInMMTiling() const
{
  return true;
}

const char *GMMA16W8Tiling::GetScenarioName() const
{
  return "A16W8";
}

bool GMMA16W8Tiling::IsScenarioMatched() const
{
  return IsGMMA16DType(xDType_) && weightDtype_ == ge::DT_INT8;
}

uint32_t GMMA16W8Tiling::GetScenarioA16W8KernelTemplate() const
{
  return isA16W8Msd_ ? static_cast<uint32_t>(GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD) :
                       static_cast<uint32_t>(GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT);
}

ge::graphStatus GMMA16W8Tiling::SetScenarioUbDivideBlk()
{
  return GMMSetUbDivideBlkAntiquant();
}

int32_t GMMA16W8Tiling::GetScenarioBaseN(bool tuningConfigFlag) const
{
  (void)tuningConfigFlag;
  return isA16W8Msd_ && maxN_ >= 2048 && !transposeWeight_ ? BEST_BASEN_MSD : BEST_BASEN;
}

int32_t GMMA16W8Tiling::AlignScenarioBaseM(uint32_t baseM) const
{
  if (!isA16W8Msd_) {
    return GMMTiling::AlignScenarioBaseM(baseM);
  }
  return baseM > A16W8_MSD_STEP * maxM_ ?
      static_cast<int32_t>(SixteenAlign(static_cast<int64_t>(A16W8_MSD_STEP) * maxM_, true)) :
      SixteenAlign(static_cast<int64_t>(baseM));
}

int64_t GMMA16W8Tiling::GetScenarioMInMM() const
{
  return isA16W8Msd_ ? static_cast<int64_t>(A16W8_MSD_STEP) * maxM_ : maxM_;
}

bool GMMA16W8Tiling::ShouldDisableBiasInMMTiling() const
{
  return isA16W8Msd_;
}

const char *GMMA16W4Tiling::GetScenarioName() const
{
  return "A16W4";
}

bool GMMA16W4Tiling::IsScenarioMatched() const
{
  return IsGMMA16DType(xDType_) && weightDtype_ == ge::DT_INT4;
}

ge::graphStatus GMMA16W4Tiling::DoScenarioOpTiling()
{
  if (xDType_ == ge::DT_BF16) {
    ge::graphStatus a16w4MsdTilingResult = A16W4MsdTiling(context_, compileInfoPtr_);
    if (a16w4MsdTilingResult == ge::GRAPH_SUCCESS) {
      isSpecialTilingDone_ = true;
      return ge::GRAPH_SUCCESS;
    }
  }
  return GMMTiling::DoScenarioOpTiling();
}

ge::graphStatus GMMA16W4Tiling::SetScenarioUbDivideBlk()
{
  return GMMSetUbDivideBlkAntiquant();
}

const char *GMMFloatTiling::GetScenarioName() const
{
  return "Float";
}

bool GMMFloatTiling::IsScenarioMatched() const
{
  return xDType_ == weightDtype_ && IsGMMFloatDType(xDType_);
}

uint32_t GMMFloatTiling::GetScenarioAivAicRatio() const
{
  return !transposeX_ ? static_cast<uint32_t>(GROUPED_MATMUL_CUBE_ONLY) :
                        static_cast<uint32_t>(GROUPED_MATMUL_AIV_AIC_RATIO_1);
}

ge::graphStatus GMMFloatTiling::SetScenarioUbDivideBlk()
{
  return GMMSetUbDivideBlkAntiquant();
}

bool GMMFloatTiling::TryScenarioSpecificBaseTiling(const GMMCompileInfo *compileInfoPtr, bool tuningConfigFlag)
{
  (void)tuningConfigFlag;
  if (groupType_ == SPLIT_K && isSingleX_ && isSingleWeight_ && isA16W16_ &&
      (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
       compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910_93) &&
      maxM_ > (maxN_ * SPLITK_M_N_RATIO_THRESHOLD_2)) {
    baseM_ = SPLITK_BASEM_256;
    baseN_ = SPLITK_BASEN_128;
    baseK_ = SPLITK_BASEK_64;
    baseM_ = baseM_ > maxM_ ? static_cast<int32_t>(SixteenAlign(maxM_, true)) : baseM_;
    return true;
  }
  return false;
}

void GMMTiling::PrintTilingInfo(gert::TilingContext *context) {
  OP_LOGD(context->GetNodeName(), "End Run GMM Tiling");
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: usedCoreNum is %d.", tilingData.mmTilingData.get_usedCoreNum());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: bestSingleN is %u.", tilingData.gmmBaseParams.get_singleN());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: with_offset is %d.", tilingData.gmmBaseParams.get_withOffset());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: tuning_config is %ld.", tuningConfig_);
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: Ka is %d.", tilingData.mmTilingData.get_Ka());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: Kb is %d.", tilingData.mmTilingData.get_Kb());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: baseM is %d.", tilingData.mmTilingData.get_baseM());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: baseN is %d.", tilingData.mmTilingData.get_baseN());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: baseK is %d.", tilingData.mmTilingData.get_baseK());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: depthA1 is %d.", tilingData.mmTilingData.get_depthA1());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: depthB1 is %d.", tilingData.mmTilingData.get_depthB1());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: stepKa is %d.", tilingData.mmTilingData.get_stepKa());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: stepKb is %d.", tilingData.mmTilingData.get_stepKb());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: stepM is %d.", tilingData.mmTilingData.get_stepM());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: stepN is %d.", tilingData.mmTilingData.get_stepN());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: isBias is %d.", tilingData.mmTilingData.get_isBias());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: transLength is %d.", tilingData.mmTilingData.get_transLength());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: iterateOrder is %d.", tilingData.mmTilingData.get_iterateOrder());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: dbL0A is %d.", tilingData.mmTilingData.get_dbL0A());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: dbL0B is %d.", tilingData.mmTilingData.get_dbL0B());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: dbL0C is %d.", tilingData.mmTilingData.get_dbL0C());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: usedL1Size is %d.", tilingData.mmTilingData.get_shareL1Size());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: usedUBSize is %d.", tilingData.mmTilingData.get_shareUbSize());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: isOutputDisableL2Cache is %d.", tilingData.gmmBaseParams.get_isOutputDisableL2Cache());
  auto buf = (uint32_t *)context->GetRawTilingData()->GetData();
  auto bufLen = context->GetRawTilingData()->GetDataSize();
  std::ostringstream oss;
  oss << "Start to dump tiling info. tilingkey:" << context->GetTilingKey() << ", tiling data size:" << bufLen
      << ", content:";
  for (size_t i = 0; i < bufLen / sizeof(uint32_t); i++) {
      oss << *(buf + i) << ",";
      if (oss.str().length() > 640) { // Split according to 640 to avoid truncation
          OP_LOGD(context, "%s", oss.str().c_str());
          oss.str("");
      }
  }
  OP_LOGD(context, "%s", oss.str().c_str());
}

ge::graphStatus GMMTiling::GMMCalUbSize(const gert::TilingContext* context, uint32_t ubSize) {
  OP_CHECK_IF((ubDivideBlkNum_ == 0 || ubBlockAlign_ == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "ubDivideBlkNum and ubBlockAlign cannot be 0"),
             return ge::GRAPH_FAILED);
  uint32_t ubCalSize = ubSize / ubDivideBlkNum_;  // divide the UB into ubDivideBlkNum_ pieces
  ubCalSize = ubCalSize / ubBlockAlign_ * ubBlockAlign_;  // 16k/8k/4k align.
  uint32_t ubRestBytes = ubSize - ubCalSize * ubIoBlkNum_;  // compute the rest memory in UB space
  ubRestBytes = ubRestBytes / UB_BLOCK_UNIT_SIZE * UB_BLOCK_UNIT_SIZE;  // 32B align.
  OP_CHECK_IF((ubCalSize == 0 || ubRestBytes == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "ubCalSize and ubRestBytes cannot be 0"),
             return ge::GRAPH_FAILED);
  uint32_t ubBaseN = 0;  // init
  uint32_t ubBaseK = 0;  // init
  uint32_t ubBaseM = 0;  // init
  if (transposeWeight_) {
    ubBaseK = static_cast<uint32_t>(BEST_UB_BASEK);
    ubBaseN = ubCalSize / ubBaseK;
    uint32_t alignFactor = UB_BLOCK_UNIT_SIZE;
    if (weightDtype_ == ge::DT_INT4) {
      alignFactor <<= 1U;  // int4 need 64 elements algin.
    }
    ubBaseN = ubBaseN / alignFactor * alignFactor;
  } else {
    if ((xDType_ == ge::DT_BF16 || xDType_ == ge::DT_FLOAT16) &&
        (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
      if (perTokenOrPerGroupSize_ > 0U) {
        ubBaseK = perTokenOrPerGroupSize_;
        ubBaseN = std::min<uint32_t>(BEST_UB_BASEN, std::max<uint32_t>(MIN_UB_BASEN, (ubCalSize / ubBaseK + MIN_UB_BASEN - 1) / MIN_UB_BASEN * MIN_UB_BASEN));
      } else if (antiquantPerformance_) {
        ubBaseN = static_cast<uint32_t>(BEST_UB_BASEN);
      } else {
        ubBaseN = static_cast<uint32_t>(baseN_);
      }
    } else {
      ubBaseN = static_cast<uint32_t>(baseN_);
    }
    ubBaseK = ubCalSize / ubBaseN;  // ubCalSize is the number of elements, not in bytes unit.
    ubBaseM = ubCalSize / ubBaseN;
  }
  if (xDType_ == ge::DT_BF16 && (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4) && !isA16W8Msd_) {
    OP_CHECK_IF(ubBaseK == 0 || ubBaseN == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "ubBaseK or ubBaseN cannot be 0"),
               return ge::GRAPH_FAILED);
  }
  tilingData.gmmBaseParams.set_ubCalSize(ubCalSize);
  tilingData.gmmBaseParams.set_ubRestBytes(ubRestBytes);  // in byte unit
  tilingData.gmmBaseParams.set_ubBaseK(ubBaseK);
  tilingData.gmmBaseParams.set_ubBaseN(ubBaseN);
  tilingData.gmmBaseParams.set_vBaseM(ubBaseM);
  return ge::GRAPH_SUCCESS;
}

int64_t GMMTiling::GMMGetBS(const gert::Shape &xShape) const {
    int64_t bs = 0;  // init bs
    if (transposeX_) {
      bs = xShape.GetDim(1);  // x shape is [k, m] if x is transpose_
    } else {
      if (groupType_ == -1) {  // -1: no group case, may exits a situation that multi dims product equals to bs.
        bs = xShape.GetDim(0);  // 0: x first dim
        size_t bsDimNum = xDimNum_ >= 1U ? xDimNum_ - 1UL : 0UL;  // 1: x last dim k, the other dimensions are bs
        for (size_t i = 1; i < bsDimNum; i++) {
            bs *= xShape.GetDim(i);
        }
      } else {
        bs = xShape.GetDim(0);  // in group case，x's shapeis [m,k], 0 is the m axis.
      }
    }
    return bs;
}

uint32_t GMMTiling::GetTplDataType(const ge::DataType &dtype) {
  static const std::map<ge::DataType, uint32_t> dtype_map = {
    {ge::DT_INT4, GMM_TPL_INT4},
    {ge::DT_INT8, GMM_TPL_INT8},
    {ge::DT_FLOAT16, GMM_TPL_FLOAT16},
    {ge::DT_BF16, GMM_TPL_BF16},
    {ge::DT_INT32, GMM_TPL_INT32},
    {ge::DT_FLOAT, GMM_TPL_FLOAT}
  };
  auto it = dtype_map.find(dtype);
  if (it != dtype_map.end()) {
    return it->second;
  }
  return GMM_TPL_INVALID;
}

// A4W4该优化点只支持CV比例1:2，没有做1:1的适配
void GMMTiling::GMMSetTplTilingKey(gert::TilingContext* context) {
  uint32_t isStaticTilingApi = 0;
  uint32_t a8w4KernelTemplate = GetScenarioA8W4KernelTemplate();
  uint32_t a16w8KernelTemplate = GetScenarioA16W8KernelTemplate();
  uint32_t aivAicRatio = GetScenarioAivAicRatio();
  uint32_t isEnableFixedAxis = GetScenarioEnableFixedAxis();

  if (a8w4KernelTemplate == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE &&
      a16w8KernelTemplate == GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE &&
      aivAicRatio != GROUPED_MATMUL_AIV_AIC_RATIO_2 &&
      isEnableFixedAxis == 0U &&
      StaticTilingProcess(context)) {
    isStaticTilingApi = 1U;
  }

  GMMTplTilingKeyParams params;
  params.xDType = GetTplDataType(xDType_);
  params.weightDType = GetTplDataType(weightDtype_);
  params.yDType = GetTplDataType(yDtype_);
  params.transposeX = transposeX_ ? 1U : 0U;
  params.transposeWeight = transposeWeight_ ? 1U : 0U;
  params.groupListType = groupListType_;
  params.isStaticTilingApi = isStaticTilingApi;
  params.a8w4KernelTemplate = a8w4KernelTemplate;
  params.a16w8KernelTemplate = a16w8KernelTemplate;
  params.aivAicRatio = aivAicRatio;
  params.isEnableFixedAxis = isEnableFixedAxis;
  context->SetTilingKey(MakeGMMTplTilingKey(params));

  if (ShouldSetScenarioScheduleMode()) {
    context->SetScheduleMode(1);  // set as batchmod for template using SyncAll
  }
}

ge::graphStatus GMMTiling::GMMGetAttrs(const gert::TilingContext* context) {
  auto attr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attr);  // check attr is not null
  const bool* transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
  const bool* transposeXPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
  const int64_t* groupTypePtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPTYPE);
  const int64_t* splitItemPtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_SPLIT_ITEM);
  const int64_t* actTypePtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);
  const uint32_t* groupListTypePtr = attr->GetAttrPointer<uint32_t>(ATTR_INDEX_GROUP_LIST_TYPE);
  const auto tuningConfigPtr = attr->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_TUNING_CONFIG);
  transposeWeight_ = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
  transposeX_ = transposeXPtr != nullptr ? *transposeXPtr : false;
  groupType_ = groupTypePtr != nullptr ? *groupTypePtr : NO_SPLIT;
  splitItem_ = splitItemPtr != nullptr ? *splitItemPtr : 0U;  // 0: 默认split_item
  actType_ = actTypePtr != nullptr ? *actTypePtr : 0;
  groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;

  auto xDesc = context->GetDynamicInputDesc(X_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);  // check xDesc is not null
  xDType_ = xDesc->GetDataType();
  auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
  weightDtype_ = w0Desc->GetDataType();
  if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4) {
    const uint64_t n = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape().GetDim(2);
    const uint64_t k = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(1);
    const uint64_t groupNum = context->GetDynamicInputTensor(WEIGHT_INDEX, 0)->GetStorageShape().GetDim(0);
    const uint64_t quantGroupNum = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape().GetDim(1);
    isA8W4FakeA8W8_ = true;
    A8W4noMsdSpace_ = groupNum * k * n * sizeof(int8_t) + groupNum * n * sizeof(float);
    tilingData.gmmBaseParams.set_groupNum(groupNum);
    tilingData.gmmBaseParams.set_n(n);
    tilingData.gmmBaseParams.set_k(k);
    tilingData.gmmBaseParams.set_quantGroupNum(quantGroupNum);
  }
  isA8W8_ = (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT8);
  isA4W4_ = xDType_ == ge::DT_INT4 && weightDtype_ == ge::DT_INT4;
  isA16W16_ = (xDType_ == ge::DT_FLOAT16 && weightDtype_ == ge::DT_FLOAT16) || (xDType_ == ge::DT_BF16 && weightDtype_ == ge::DT_BF16);
  auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);  // check compileInfoPtr is not null
  if (groupListType_ == GROUP_LIST_SPARSE_M) {
    OP_CHECK_IF((!(compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
                  compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910_93)),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "This platform not support groupListType is 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!isA8W8_,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "Only both x and weight dtype are int8 support groupListType is 2"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupType_ != SPLIT_M,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "When groupListType is 2 only support groupType 0, but get groupType %ld",
                                           groupType_),
               return ge::GRAPH_FAILED);
  }

  auto perTokenScalePtr = context->GetOptionalInputTensor(PER_TOKEN_SCALE_INDEX);
  if (perTokenScalePtr != nullptr && perTokenScalePtr->GetStorageShape().GetShapeSize() != 0) {
    perTokenOrPerGroupSize_ = 1U;
  }
  auto pertokenScaleDesc = context->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
  if (pertokenScaleDesc != nullptr) {
    perTokenScaleDtype_ = pertokenScaleDesc->GetDataType();
  }
  tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
  auto yDesc = context->GetOutputDesc(Y_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
  yDtype_ = yDesc->GetDataType();
  if ((weightDtype_ == ge::DT_INT8 && xDType_ == ge::DT_INT8 && yDtype_ != ge::DT_INT32) || isA8W4FakeA8W8_ ||
      (xDType_ == ge::DT_FLOAT8_E4M3FN) || (xDType_ == ge::DT_FLOAT8_E5M2) || isA4W4_) {
      auto scale0Desc = context->GetDynamicInputDesc(SCALE_INDEX, 0);
      OP_CHECK_NULL_WITH_CONTEXT(context, scale0Desc);
      scaleDtype_ = scale0Desc->GetDataType();
  }

  auto wFormat0 = static_cast<ge::Format>(ge::GetPrimaryFormat(w0Desc->GetStorageFormat()));
  wFormat_ = wFormat0 == ge::FORMAT_FRACTAL_NZ ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND;
  tuningConfig_ = (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > TUNING_CONFIG_TOKEN_PER_EXPECT_INDEX) ?
                  (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[TUNING_CONFIG_TOKEN_PER_EXPECT_INDEX] : 0;
  tuningConfigWorkspace_ = (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > TUNING_CONFIG_ALLOW_WORKSPACE_INDEX) ?
                  (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[TUNING_CONFIG_ALLOW_WORKSPACE_INDEX] : 0;

  if (((compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
        compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910_93)) &&
    isA4W4_ ){
      OP_CHECK_IF( !((wFormat0 == ge::FORMAT_FRACTAL_NZ) || (wFormat0 != FORMAT_FRACTAL_NZ && transposeWeight_ == 0)),
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "A4W4 GMM currently supports only weight tensor nz transpose/untranspose input \
or nd format untranspose input."
                                           ),
                return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlk() {
  ubDivideBlkNum_ = 0U;  // init ubDivideBlkNum_
  ubIoBlkNum_ = 0U;  // init ubIoBlkNum_
  ubBlockAlign_ = 0U;  // init ubBlockAlign_
  return SetScenarioUbDivideBlk();
}

ge::graphStatus GMMTiling::SetBias(const gert::TilingContext* context, matmul_tiling::MultiCoreMatmulTiling& mm) const {
  if (!hasBias_ || ShouldDisableBiasInMMTiling()) {
    mm.SetBias(false);
  } else {
    mm.SetBias(true);
    auto biasTensor = context->GetDynamicInputTensor(BIAS_INDEX, 0);
    OP_CHECK_IF(biasTensor == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Get bias tensor failed."),
               return ge::GRAPH_FAILED);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   static_cast<matmul_tiling::DataType>(biasTensor->GetDataType()));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::CalcStepKaKb(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr,
                                        int64_t mInMM, uint32_t& mmStepKa, uint32_t& mmStepKb) {
  uint64_t availableL1Size = compileInfoPtr->l1Size;
  if (ShouldReserveQuantL1N()) {
    availableL1Size -= static_cast<uint64_t>(baseN_) * sizeof(uint64_t);
  }
  if (hasBias_) {
    availableL1Size -= static_cast<uint64_t>(baseN_) * static_cast<uint64_t>(4);  // 4: size of float32 or int32
  }
  if (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
    availableL1Size = BEST_L1_PARTA + BEST_L1_PARTB;
  }
  OP_CHECK_IF(availableL1Size < L1_PARTA_SIZE,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "availableL1Size is less than 256k"),
             return ge::GRAPH_FAILED);
  // according to double buffer, recompute the params used for data movement from GM to L1
  uint64_t l1ASize = baseM_ > baseN_ ? L1_PARTA_SIZE : availableL1Size - L1_PARTA_SIZE;
  uint64_t l1BSize = availableL1Size - l1ASize;
  mmStepKa = CalcScenarioStepKa(l1ASize);
  mmStepKb = CalcScenarioStepKb(l1BSize);
  if (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P &&
                                    wFormat_ == matmul_tiling::CubeFormat::NZ && mInMM <= baseM_) {
    mmStepKa = std::min<uint32_t>(mmStepKa, std::max(1, 128 / baseK_));  // 128: nz inner block size. In practice, baseK_*mmStepKa=128 makes performance better.
  }

  OP_CHECK_IF(mmStepKa == 0 || mmStepKb == 0,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "stepka or stepkb cannot be 0"),
             return ge::GRAPH_FAILED);

  if (mmStepKa > mmStepKb) {
    mmStepKa = mmStepKa / mmStepKb * mmStepKb;
  } else if (mmStepKa < mmStepKb) {
    mmStepKb = mmStepKb / mmStepKa * mmStepKa;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GMMSetMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr) {
  matmul_tiling::DataType matmulDtype = static_cast<matmul_tiling::DataType>(mmDType_);
  matmul_tiling::PlatformInfo platformInfo;
  InitPlatformInfo(compileInfoPtr, platformInfo);
  matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
  int64_t mInMM = GetScenarioMInMM();
  mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
  mm.SetBType(matmul_tiling::TPosition::GM, wFormat_, matmulDtype, false);
  mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, matmul_tiling::DataType::DT_FLOAT16);
  OP_CHECK_IF(SetBias(context, mm) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetBias failed."), return ge::GRAPH_FAILED);
  mm.SetOrgShape(mInMM, maxN_, maxK_);
  mm.SetShape(mInMM, baseN_, maxK_);
  mm.SetFixSplit(baseM_, baseN_, baseK_);
  mm.SetBufferSpace(compileInfoPtr->l1Size, compileInfoPtr->l0CSize, ubSize_);
  if(groupType_ == SPLIT_K && maxK_ == 0) {
    //切K场景K轴为0时不需要使用高阶API，无需tiling，kernel需要做清零处理，此处无需tiling
    OP_LOGD("GMM SplitK,and K == 0, no need tiling");
  } else {
    OP_CHECK_IF(mm.GetTiling(tilingData.mmTilingData) == -1,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "matmul getTiling failed."),
            return ge::GRAPH_FAILED);
  }


  uint32_t mmStepKa = 1;
  uint32_t mmStepKb = 1;
  OP_CHECK_IF(CalcStepKaKb(context, compileInfoPtr, mInMM, mmStepKa, mmStepKb) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "matmul calc stepka or stepkb failed."),
             return ge::GRAPH_FAILED);

  constexpr uint32_t stepM = 1;  // 1: stepM set fixed value 1
  constexpr uint32_t stepN = 1;  // 1: stepN set fixed value 1
  uint32_t mmDepthA1 = mmStepKa * DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
  uint32_t mmDepthB1 = mmStepKb * DOUBLE_BUFFER_STEPKA_STEPKB * stepN;
  tilingData.mmTilingData.set_shareMode(0);
  if (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
    tilingData.mmTilingData.set_shareUbSize(0);
    tilingData.mmTilingData.set_transLength(131072);  // 131072: 128KB size
  }
  tilingData.mmTilingData.set_dbL0C(1);  // disable double buffer for LOC
  tilingData.mmTilingData.set_baseM(baseM_);  // set precomputed baseM
  tilingData.mmTilingData.set_baseN(baseN_);  // set precomputed baseN
  tilingData.mmTilingData.set_baseK(baseK_);  // set precomputed baseK
  tilingData.mmTilingData.set_stepKa(mmStepKa);  // set precomputed mmStepKa
  tilingData.mmTilingData.set_depthA1(mmDepthA1);  // set precomputed mmDepthA1
  tilingData.mmTilingData.set_stepKb(mmStepKb);  // set precomputed mmStepKb
  tilingData.mmTilingData.set_depthB1(mmDepthB1);  // set precomputed mmDepthB1
  tilingData.mmTilingData.set_stepM(stepM);  // set precomputed stepM
  tilingData.mmTilingData.set_stepN(stepN);  // set precomputed stepN
  SetMMPreTiling();
  OP_LOGI(context->GetNodeName(), "GMM_tiling: baseM is %d, baseK is %d, baseN is %d.", baseM_, baseK_, baseN_);
  return ge::GRAPH_SUCCESS;
}

void GMMTiling::SetMMPreTiling() {
  uint64_t ispreTiling = 0;
  int64_t isNz = wFormat_ ==  matmul_tiling::CubeFormat::NZ ? 1 : 0;
  if (tuningConfig_ == 0L && CanUsePreTiling() && groupNum_ == 1U && usedCoreNum_ == A3_AIC_NUM) {
    ispreTiling = static_cast<uint64_t>(1); // 1: pretiling key
  } else {
    tilingData.gmmBaseParams.set_isPreTiling(ispreTiling);
    return;
  }
  std::array<int64_t, 4> mKNList = {maxM_, maxK_, maxN_, isNz}; // 4: input shape info size
  if (A8W8_PRETILING_WHITE_LIST.count(mKNList)) {
    int64_t newBaseM = A8W8_PRETILING_WHITE_LIST.at(mKNList)[0];
    int64_t bestSingleN = A8W8_PRETILING_WHITE_LIST.at(mKNList)[1];
    tilingData.mmTilingData.set_baseM(newBaseM);  // set pretiling baseM
    tilingData.mmTilingData.set_singleCoreN(bestSingleN);  // set pretiling singleN
    ispreTiling = static_cast<uint64_t>(2); // 2: white list pretiling key
  }
  tilingData.gmmBaseParams.set_isPreTiling(ispreTiling);
  return;
}

ge::graphStatus GMMTiling::CalMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr) {
  // if tuningConfig_ in (128, 256], recompute tiling.
  // or y:int32 and tuningConfig_ in (0, 128], k,n:(7168, 2048) or k,n:(7680, 2048), which got by actual measurement)
  constexpr int32_t tuningConfigBaseLowerLimit = 128;
  constexpr int32_t tuningConfigBaseUpperLimit = 256;
  constexpr int32_t maxKLimit = 7168;
  constexpr int32_t altKLimit = 7680;
  constexpr int32_t maxNLimit = 2048;

  bool tuningConfigFlag = (tuningConfig_ > tuningConfigBaseLowerLimit && tuningConfig_ <= tuningConfigBaseUpperLimit)
    || (yDtype_ == ge::DT_INT32 && tuningConfig_ > 0 && tuningConfig_ <= tuningConfigBaseLowerLimit && (maxK_ == maxKLimit || maxK_ == altKLimit) && maxN_ == maxNLimit);

  baseN_ = GetScenarioBaseN(tuningConfigFlag);
  if (TryScenarioSpecificBaseTiling(compileInfoPtr, tuningConfigFlag)) {
    return ge::GRAPH_SUCCESS;
  }
  // according to the double buffer enabled L0B, compute baseK
  baseK_ = CalcScenarioBaseK(compileInfoPtr);
  baseK_ = static_cast<int32_t>(SixteenAlign(static_cast<int64_t>(baseK_)));
  OP_CHECK_IF(baseK_ == 0,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "baseK_ cannot be 0."),
             return ge::GRAPH_FAILED);
  // according to the double buffer enabled L0A/L0C, compute baseM(cube)
  uint32_t maxBaseM = static_cast<uint32_t>(compileInfoPtr->l0CSize /
                                            (static_cast<uint32_t>(baseN_) * FP32_DATATYPE_SIZE));
  baseM_ = static_cast<int32_t>(CalcScenarioBaseM(compileInfoPtr, maxBaseM));
  baseM_ = AlignScenarioBaseM(baseM_);
  if (baseM_ > MAX_BASEM) {
    baseM_ = MAX_BASEM;
  }
  OP_CHECK_IF(baseM_ == 0,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "baseM_ cannot be 0."),
             return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
