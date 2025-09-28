/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_grouped_matmul.h"
#include "aclnn_grouped_matmul_v2.h"
#include "aclnn_grouped_matmul_v3.h"
#include "aclnn_grouped_matmul_v4.h"
#include "aclnn_grouped_matmul_v5.h"
#include "aclnn_grouped_matmul_weight_nz.h"

#include <dlfcn.h>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "grouped_matmul.h"
#include "aclnn_kernels/contiguous.h"
#include "../grouped_matmul_host_util.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#include "aclnn_grouped_matmul_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
  static constexpr int64_t X_Y_SEPARATED = 0L;  // x,y no split
  static constexpr int64_t Y_SEPARATED = 1L;   // x split
  static constexpr int64_t X_SEPARATED = 2L;   // y split
  static constexpr int64_t NO_SEPARATED = 3L;  // x,y split
  static constexpr int64_t MAX_GROUP_LIST_SIZE_ARRAY = 128L;
  static constexpr int64_t MAX_GROUP_LIST_SIZE_TENSOR = 1024L;
  static constexpr int64_t MAX_INNER_AXIS = 65535L;

  static constexpr size_t SEPARATED_WEIGHT_DIM = 2UL;
  static constexpr int64_t END_ACT_TYPE_ENUM = 6L;
  static constexpr size_t ALIGN_NZ_INT4_N = 64UL;
  static constexpr size_t ALIGN_NZ_INT8_N = 32UL;
  static constexpr size_t ALIGN_NZ_K = 16UL;

  static constexpr uint64_t INT4_PER_INT32 = 8UL;

  static constexpr size_t WEIGHT_DIM_A8W4 = 3UL;
  static constexpr size_t OFFSET_DIM_A8W4 = 3UL;
  static constexpr size_t BIAS_DIM_A8W4 = 2UL;
  static constexpr size_t PER_CHANNEL_SCALE_DIM = 2UL;
  static constexpr size_t PER_GROUP_SCALE_DIM = 3UL;
  static constexpr size_t DIMS_THREE_FOR_GMM = 3UL;

  static void UnpackInt32ToInt4(const aclTensorList *&tensorListS32, const std::string& tensorListType)
  {
    OP_LOGD("Unpack %s from int32 to int4 start.", tensorListType.c_str());
    auto tensorListS4 = const_cast<aclTensorList *>(tensorListS32);
    for (size_t i = 0; i < tensorListS4->Size();++i) {
      op::Shape tensorShape = (*tensorListS4)[i]->GetViewShape();
      auto viewShapeDim = tensorShape.GetDimNum();
      tensorShape[viewShapeDim - 1] = tensorShape[viewShapeDim - 1] * INT4_PER_INT32;
      (*tensorListS4)[i]->SetViewShape(tensorShape);
      (*tensorListS4)[i]->SetDataType(DataType::DT_INT4);
    }
    OP_LOGD("Unpack %s from int32 to int4 finished.", tensorListType.c_str());
  }

  bool IsQuant(const DataType &xDtype, const DataType &weightDtype)
  {
    if (xDtype == DataType::DT_FLOAT4_E1M2 || xDtype == DataType::DT_FLOAT4_E2M1 || xDtype == DataType::DT_INT4) {
      return true;
    }
    return ge::GetSizeByDataType(xDtype) == 1 && ge::GetSizeByDataType(weightDtype) == 1;
  }

  bool IsWeightQuant(const DataType &xDtype, const DataType &weightDtype)
  {
    return ge::GetSizeByDataType(xDtype) != ge::GetSizeByDataType(weightDtype);
  }
}

namespace {
static bool CheckSpecialTranspose(const aclTensorList *tensorList) {
  // if last two axis shape is (1, 1), gmm::IsTransposeLastTwoDims() api always return true,
  // when group type is 0 or -1, x is required to not be transposed. To ensure this case can execute normally,
  // transposeX is setted to false manually.
  // when groupType = 2, gmm::IsTransposeLastTwoDims() returns true, and weight requires to not be transposed,
  // transposeWeight need to set false
  int64_t loopNum = tensorList->Size();
  int64_t dimNum = 0;
  int64_t checkedAxisNum = 0;
  int64_t lastAxisSize = 0;
  bool transpose = false;
  for (int64_t i = 0; i < loopNum; ++i) {
    lastAxisSize = 1;
    transpose = gmm::IsTransposeLastTwoDims((*tensorList)[i]);
    auto shape = (*tensorList)[i]->GetViewShape();
    dimNum = shape.GetDimNum();
    checkedAxisNum = dimNum > 1 ? 2 : 1;  // 2:need to check last two axis' shape
    for (int64_t j = 1; j <= checkedAxisNum; ++j) {
      lastAxisSize *= shape.GetDim(dimNum - j);
    }
    transpose = transpose && (lastAxisSize != 1);
    if (transpose) {
      break;
    }
  }
  return transpose;
}
}

namespace {
static aclnnStatus CheckShapeSameLengthTensorList(const aclTensorList *tensorList1, const aclTensorList *tensorList2,
                                                  const std::vector<size_t>& dimIds, const int64_t innerAxisDimId,
                                                  const std::vector<std::string>& tensorType) {
  // Verify if the values of a specified dimension in each tensor of two tensor lists with equal lengths are consistent.
  uint64_t groupNum = tensorList1->Size();
  for (uint64_t i = 0; i < groupNum; i++) {
    int64_t dimValue1 = (*tensorList1)[i]->GetViewShape().GetDim(dimIds[0]);
    // tensorType[2] indicates whether to verify innerAxisDimId of tensorList1;if so, check if it's less than or equal to 65535.
    if (tensorType[2] == "true" && innerAxisDimId > -1) {
      int64_t innerAxisValue = (*tensorList1)[i]->GetViewShape().GetDim(innerAxisDimId);
      CHECK_COND(innerAxisValue <= MAX_INNER_AXIS, ACLNN_ERR_PARAM_INVALID, "Dim %lu value of %s[%lu] should less or equal to 65535, but now is %ld.",
                 dimIds[0], tensorType[0].c_str(), i, innerAxisValue);
    }
    int64_t dimValue2 = (*tensorList2)[i]->GetViewShape().GetDim(dimIds[1]);
    CHECK_COND(dimValue1 == dimValue2, ACLNN_ERR_PARAM_INVALID,
               "Dim %lu value of %s[%lu] should be equal with dim %lu value of %s[%lu], but now is %ld and %ld respectively.",
               dimIds[0], tensorType[0].c_str(), i, dimIds[1], tensorType[1].c_str(), i, dimValue1, dimValue2);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckShapeDiffLengthTensorList(const aclTensorList *longTensorList,
                                                  const aclTensorList *singleTensorList,
                                                  const std::vector<size_t>& dimIds,
                                                  const int64_t innerAxisdimId,
                                                  const std::vector<std::string>& tensorType) {
  // Check if the values of a specified axis in a tensor list of multiple tensor
  // match those in a tensor list of a single tensor.
  // Specified axis is not a split axis.
  int64_t dimValueSingle = (*singleTensorList)[0]->GetViewShape().GetDim(dimIds[1]);
  // tensorType[2] indicates whether to verify innerAxisdimId of tensorList1; if so, check if it's less than or equal to 65535.
  if (tensorType[2] == "true" && innerAxisdimId > -1) {
      int64_t dimValue = (*singleTensorList)[0]->GetViewShape().GetDim(innerAxisdimId);
      CHECK_COND(dimValue <= MAX_INNER_AXIS, ACLNN_ERR_PARAM_INVALID, "Dim %ld value of %s[0] should less or equal to 65535, but now is %ld.",
                 innerAxisdimId, tensorType[1].c_str(), dimValue);
    }
  uint64_t groupNum = longTensorList->Size();
  for (uint64_t i = 0; i < groupNum; i++) {
    int64_t dimValueLong = (*longTensorList)[i]->GetViewShape().GetDim(dimIds[0]);
    CHECK_COND(dimValueLong == dimValueSingle, ACLNN_ERR_PARAM_INVALID,
               "Dim %lu value of %s[%lu] %ld should be equal with dim %lu value of %s[0] %ld.",
               dimIds[0], tensorType[0].c_str(), i, dimValueLong,
               dimIds[1], tensorType[1].c_str(), dimValueSingle);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(const aclTensor *tensor, const std::string& tensorType, size_t idx) {
  bool isWeightTensor = tensorType == "weight";
  op::Format tensorFormat = tensor->GetStorageFormat();
  CHECK_COND(tensorFormat < Format::FORMAT_END, ACLNN_ERR_PARAM_INVALID, "Format of %s[%lu] %s is invalid.",
             tensorType.c_str(), idx, op::ToString(tensorFormat).GetString());
  if (isWeightTensor) {  // 310P weight need to be NZ
      CHECK_COND(!op::IsPrivateFormat(tensorFormat) || tensorFormat == Format::FORMAT_FRACTAL_NZ ||
                     tensorFormat == Format::FORMAT_FRACTAL_NZ_C0_16,
                 ACLNN_ERR_PARAM_INVALID, "Format of %s[%lu] %s is invalid.", tensorType.c_str(), idx,
                 op::ToString(tensorFormat).GetString());
  } else {
      CHECK_COND(!op::IsPrivateFormat(tensorFormat), ACLNN_ERR_PARAM_INVALID, "Format of %s[%lu] %s is invalid.",
                 tensorType.c_str(), idx, op::ToString(tensorFormat).GetString());
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckShapeDiffLengthTensorListSplitAxis(const aclTensorList *longTensorList,
                                                           const aclTensorList *singleTensorList,
                                                           const size_t dimIdxLongTensorList,
                                                           const size_t dimIdxSingleTensorList,
                                                           const std::vector<std::string>& tensorType) {
  // Check if the sum of values along a specified axis in a multi-tensor list equals
  //the corresponding axis value in a single-tensor list.
  // The specified axis is the split axis.
  int64_t dimValueSingle = (*singleTensorList)[0]->GetViewShape().GetDim(dimIdxSingleTensorList);
  uint64_t groupNum = longTensorList->Size();
  int64_t preOffset = 0;
  for (uint64_t i = 0; i < groupNum; i++) {
    int64_t dimValueLong = (*longTensorList)[i]->GetViewShape().GetDim(dimIdxLongTensorList);
    preOffset += dimValueLong;
  }
  CHECK_COND(preOffset == dimValueSingle, ACLNN_ERR_PARAM_INVALID,
             "Sum of dim %lu value of %s %ld should be equal with dim %lu value of %s[0] %ld.",
             dimIdxLongTensorList, tensorType[0].c_str(), preOffset,
             dimIdxSingleTensorList, tensorType[1].c_str(), dimValueSingle);
  return ACLNN_SUCCESS;
}

static aclnnStatus PreCheckGroupType(int64_t splitItem, int64_t groupType) {
  // Intercept currently unsupported groupType
  CHECK_COND(groupType != gmm::SPLIT_N, ACLNN_ERR_PARAM_INVALID, "Not support split n dim now, groupType can not be 1.");
  CHECK_COND(groupType == gmm::SPLIT_M || groupType == gmm::SPLIT_K || groupType == gmm::NO_SPLIT, ACLNN_ERR_PARAM_INVALID,
             "groupType only support -1/0/2 now, but given groupType is %ld", groupType);
  if (splitItem == X_SEPARATED || splitItem == NO_SEPARATED) {
    CHECK_COND(groupType != gmm::NO_SPLIT, ACLNN_ERR_PARAM_INVALID, "When splitItem is 2/3, groupType can not be -1.");
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckDimNumAndFormat(const gmm::GroupedMatmulParams &gmmParams, const aclTensorList *tensorList,
                                        const size_t expectedDimNum, const std::string& tensorType) {
  uint64_t tensorListLength = tensorList->Size();
  for (size_t i = 0; i < tensorListLength; ++i) {
    CHECK_COND((*tensorList)[i] != nullptr, ACLNN_ERR_PARAM_INVALID,
               "%s[%lu] is null, which is not supported.", tensorType.c_str(), i);
    CHECK_COND(CheckFormat((*tensorList)[i], tensorType, i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Invalid format.");
    size_t dimNum = (*tensorList)[i]->GetViewShape().GetDimNum();
    CHECK_COND(dimNum == expectedDimNum, ACLNN_ERR_PARAM_INVALID,
               "%s[%lu] dim num should be %lu in this case, but now is %lu.",
               tensorType.c_str(), i, expectedDimNum, dimNum);
    if (tensorType == "weight") {
      bool transpose = gmmParams.groupType == gmm::SPLIT_K ? CheckSpecialTranspose(tensorList) : gmm::IsTransposeLastTwoDims((*gmmParams.weight)[i]);
      CHECK_COND(transpose == gmmParams.transposeWeight, ACLNN_ERR_PARAM_INVALID,
                 "The transpose state must be the same for each tensor in weight.");
    }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckDimNumAndGroupListNoSplitAndFormat(const gmm::GroupedMatmulParams &gmmParams) {
  // When groupType is -1 and not V1 interface, grouplist be empty.
  if (gmmParams.apiVersion != gmm::GMMApiVersion::V1) {
    CHECK_COND(gmmParams.groupListOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "groupListOptional should be nullptr when groupType is -1.");
  }
  size_t tensorListLength = gmmParams.x->Size();
  // Check that the length of grouplist is consistent with x when grouplist is not empty.
  if (gmmParams.groupListOptional != nullptr) {
    CHECK_COND(gmmParams.groupListOptional->Size() == tensorListLength, ACLNN_ERR_PARAM_INVALID,
               "Size of groupListOptional %lu should be equal to size of x %lu.",
               gmmParams.groupListOptional->Size(), tensorListLength);
  }
  if (gmmParams.groupTensorOptional != nullptr) {
    CHECK_COND(gmmParams.groupTensorOptional->GetViewShape().GetDim(0) == static_cast<int64_t>(tensorListLength),
               ACLNN_ERR_PARAM_INVALID, "Size of groupListOptional(tensor) %ld should be equal to size of x %zu.",
               gmmParams.groupTensorOptional->GetViewShape().GetDim(0), tensorListLength);
  }
  int64_t preGoupList = 0;
  for (size_t i = 0; i < tensorListLength; ++i) {
    // Check dims
    CHECK_COND((*gmmParams.x)[i] != nullptr, ACLNN_ERR_PARAM_INVALID, "x[%lu] is null, which is not supported.", i);
    CHECK_COND((*gmmParams.weight)[i] != nullptr, ACLNN_ERR_PARAM_INVALID, "weight[%lu] is null, which is not supported.", i);
    CHECK_COND(gmm::IsTransposeLastTwoDims((*gmmParams.weight)[i]) == gmmParams.transposeWeight, ACLNN_ERR_PARAM_INVALID,
               "The transpose state must be the same for each tensor in weight.");
    CHECK_COND((*gmmParams.y)[i] != nullptr, ACLNN_ERR_PARAM_INVALID, "y[%lu] is null, which is not supported.", i);
    CHECK_COND(CheckFormat((*gmmParams.x)[i], "x", i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Invalid format.");
    CHECK_COND(CheckFormat((*gmmParams.weight)[i], "weight", i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Invalid format.");
    CHECK_COND(CheckFormat((*gmmParams.y)[i], "y", i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Invalid format.");
    size_t xDimNum = (*gmmParams.x)[i]->GetViewShape().GetDimNum();
    size_t weightDimNum = (*gmmParams.weight)[i]->GetViewShape().GetDimNum();
    size_t yDimNum = (*gmmParams.y)[i]->GetViewShape().GetDimNum();
    CHECK_COND(xDimNum <= gmm::MAX_FM_DIM && xDimNum >= gmm::MIN_FM_DIM, ACLNN_ERR_PARAM_INVALID,
               "x[%lu] dimNum is %lu , but only support 2-6.", i, xDimNum);
    CHECK_COND(weightDimNum == SEPARATED_WEIGHT_DIM, ACLNN_ERR_PARAM_INVALID,
               "weight[%lu] dimNum is %lu , but only support 2 when weight separated.", i, weightDimNum);
    CHECK_COND(xDimNum == yDimNum, ACLNN_ERR_PARAM_INVALID,
               "y[%lu] dimNum %lu should be equal with x[%lu] DimNum %lu.", i, yDimNum, i, xDimNum);
    // If not V1 interface and x dim > 2, grouplist be empty.
    if (xDimNum > gmm::MIN_FM_DIM) {
      CHECK_COND(gmmParams.groupListOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                 "groupListOptional should be nullptr when x, y both separated and dim num larger than 2.");
    }
    if (xDimNum == gmm::MIN_FM_DIM && gmmParams.groupListOptional != nullptr) {
      int64_t xMDimValue = (*gmmParams.x)[i]->GetViewShape().GetDim(0);
      std::string errorMessage = i == 0UL ? "groupListOptional[0]" :
        "groupListOptional[" + std::to_string(i) + "] - groupListOptional[" + std::to_string(i - 1UL) + "]";
      CHECK_COND(xMDimValue == (*gmmParams.groupListOptional)[i] - preGoupList, ACLNN_ERR_PARAM_INVALID,
                 "x[%lu] dim 0 value %ld should be equal to %s %ld.",
                 i, xMDimValue, errorMessage.c_str(), (*gmmParams.groupListOptional)[i] - preGoupList);
      preGoupList = (*gmmParams.groupListOptional)[i];
    }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckNotNull(const aclTensorList *x, const aclTensorList *weight, const aclTensorList *y) {
  CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
  CHECK_COND(x->Size() != 0, ACLNN_ERR_PARAM_INVALID, "x must not be empty tensorlist.");
  CHECK_COND(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weight must not be nullptr.");
  CHECK_COND(weight->Size() != 0, ACLNN_ERR_PARAM_INVALID, "weight must not be empty tensorlist.");
  CHECK_COND(y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "y must not be nullptr.");
  CHECK_COND(y->Size() != 0, ACLNN_ERR_PARAM_INVALID, "y must not be empty tensorlist.");
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckGroupListCommonIntArray(const gmm::GroupedMatmulParams &gmmParams, const bool isRequiredGroupList,
                                                const size_t groupNum, int64_t &groupListLastValue) {
  // Must pass groupList scenario, check groupList is not empty.
  CHECK_COND(gmmParams.groupListOptional != nullptr || !isRequiredGroupList, ACLNN_ERR_PARAM_NULLPTR,
             "groupListOptional required in this case, but get nullptr.");
  if (gmmParams.groupListOptional != nullptr) {
    // groupList must be an ascending sequence.
    uint64_t groupListSize = gmmParams.groupListOptional->Size();
    CHECK_COND(groupListSize <= MAX_GROUP_LIST_SIZE_ARRAY, ACLNN_ERR_PARAM_INVALID,
               "When groupList type is int array, size of groupList %lu should be less than or equal to %ld.",
               groupListSize, MAX_GROUP_LIST_SIZE_ARRAY);
    int64_t preGoupList = 0;
    for (size_t i = 0; i < groupListSize; i++) {
      CHECK_COND((*gmmParams.groupListOptional)[i] >= preGoupList, ACLNN_ERR_PARAM_INVALID,
                  "groupListOptional should be non-negative and incremental.");
      preGoupList = (*gmmParams.groupListOptional)[i];
    }
    // Check groupList length matches other tensor lists.
    CHECK_COND((groupListSize == groupNum && groupNum > 1) || groupNum == 1, ACLNN_ERR_PARAM_INVALID,
               "When groupList is not null, size of groupList %lu should be equal to groupNum %lu.",
               groupListSize, groupNum);
    groupListLastValue = preGoupList;
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckGroupListCommonTensor(const gmm::GroupedMatmulParams &gmmParams, const bool isRequiredGroupList,
                                              const size_t groupNum) {
  CHECK_COND(!(gmmParams.groupTensorOptional == nullptr && isRequiredGroupList), ACLNN_ERR_PARAM_INVALID,
             "groupListOptional(tensor) is required in this case, but get nullptr.");
  if (gmmParams.groupTensorOptional != nullptr) {
    int64_t groupListSize = gmmParams.groupTensorOptional->GetViewShape().GetDim(0);
    size_t groupListDimNum = gmmParams.groupTensorOptional->GetViewShape().GetDimNum();
    if (gmmParams.groupListType == gmm::GROUP_LIST_SPARSE_M) {
      CHECK_COND(groupListDimNum == 2, ACLNN_ERR_PARAM_INVALID,  // sparse, group list shape [e, 2]
        "When groupList type is 2, dim num of groupList should be 2, but now is %zu.", groupListDimNum);
    }
    CHECK_COND(groupListSize <= MAX_GROUP_LIST_SIZE_TENSOR, ACLNN_ERR_PARAM_INVALID,
               "When groupList type is tenosr, size of groupList %ld should be less than or equal to %ld.",
               groupListSize, MAX_GROUP_LIST_SIZE_TENSOR);
    CHECK_COND((groupListSize == static_cast<int64_t>(groupNum) && groupNum > 1) || groupNum == 1, ACLNN_ERR_PARAM_INVALID,
               "When groupList is not null, size of groupList(tensor) %ld should be equal to %s %lu with groupType %ld.",
               groupListSize, gmmParams.groupType == gmm::SPLIT_M ? "weight dim 0" : "y dim 0", groupNum,
               gmmParams.groupType);
    CHECK_COND(gmmParams.groupTensorOptional->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
               "Invalid dtype: Only int64 is supported for groupList.");
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckGroupListSplitK(const gmm::GroupedMatmulParams &gmmParams, const bool isRequiredGroupList,
                                        const bool xSeparated, const bool weightSeparated, const size_t groupNum) {
  int64_t groupListLastValue = 0;
  if (gmmParams.apiVersion == gmm::GMMApiVersion::WeightNz || gmmParams.apiVersion == gmm::GMMApiVersion::V5
      || gmmParams.apiVersion == gmm::GMMApiVersion::V4 || gmmParams.apiVersion == gmm::GMMApiVersion::V3) {
    CHECK_COND(CheckGroupListCommonTensor(gmmParams, isRequiredGroupList, groupNum) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "CheckGroupListCommonTensor failed in groupType 2.");
    return ACLNN_SUCCESS;
  }
  CHECK_COND(
    CheckGroupListCommonIntArray(gmmParams, isRequiredGroupList, groupNum, groupListLastValue) == ACLNN_SUCCESS,
    ACLNN_ERR_PARAM_INVALID, "CheckGroupListCommonIntArray failed.");
  if (gmmParams.groupListOptional != nullptr) {
    if (xSeparated) {
      int64_t preOffset = 0;
      // Check the increment in groupList matches x's k.
      for (size_t i = 0; i < groupNum; i++) {
        int64_t xKDimValue = (*gmmParams.x)[i]->GetViewShape().GetDim(1);
        std::string errorMessage = i == 0UL ? "groupListOptional[0]" :
          "groupListOptional[" + std::to_string(i) + "] - groupListOptional[" + std::to_string(i - 1UL) + "]";
        CHECK_COND(
          xKDimValue == (*gmmParams.groupListOptional)[i] - preOffset, ACLNN_ERR_PARAM_INVALID, "x[%lu] dim 1 value %ld should be equal to %s %ld.",
                   i, xKDimValue, errorMessage.c_str(), (*gmmParams.groupListOptional)[i] - preOffset);
        preOffset = (*gmmParams.groupListOptional)[i];
      }
    } else if (weightSeparated) {
      int64_t preOffset = 0;
      // Check the increment in groupList matches weight's k.
      for (size_t i = 0; i < groupNum; i++) {
        int64_t weightKDimValue = (*gmmParams.weight)[i]->GetViewShape().GetDim(0);
        std::string errorMessage = i == 0UL ? "groupListOptional[0]" :
          "groupListOptional[" + std::to_string(i) + "] - groupListOptional[" + std::to_string(i - 1UL) + "]";
        CHECK_COND(
          weightKDimValue == (*gmmParams.groupListOptional)[i] - preOffset, ACLNN_ERR_PARAM_INVALID, "weight[%lu] dim 0 %ld value should be equal to %s %ld.",
                   i, weightKDimValue, errorMessage.c_str(), (*gmmParams.groupListOptional)[i] - preOffset);
        preOffset = (*gmmParams.groupListOptional)[i];
      }
    } else {
      CHECK_COND(
        (*gmmParams.x)[0]->GetViewShape().GetDim(1) == groupListLastValue, ACLNN_ERR_PARAM_INVALID, "When splited axis is K, the last value of group list(%ld) must equal with x shape[1] (%ld).",
                 groupListLastValue, (*gmmParams.x)[0]->GetViewShape().GetDim(1));
    }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckGroupListSplitM(const gmm::GroupedMatmulParams &gmmParams, const bool isRequiredGroupList,
                                        const bool xSeparated, const bool ySeparated, const size_t groupNum) {
  int64_t groupListLastValue = 0;
  if (gmmParams.apiVersion == gmm::GMMApiVersion::WeightNz || gmmParams.apiVersion == gmm::GMMApiVersion::V5
      || gmmParams.apiVersion == gmm::GMMApiVersion::V4 || gmmParams.apiVersion == gmm::GMMApiVersion::V3) {
    CHECK_COND(CheckGroupListCommonTensor(gmmParams, isRequiredGroupList, groupNum) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "CheckGroupListCommonTensor failed in groupType 0.");
    return ACLNN_SUCCESS;
  }
  CHECK_COND(
    CheckGroupListCommonIntArray(gmmParams, isRequiredGroupList, groupNum, groupListLastValue) == ACLNN_SUCCESS,
    ACLNN_ERR_PARAM_INVALID, "CheckGroupListCommonIntArray failed!");
  if (gmmParams.groupListOptional != nullptr) {
    if (xSeparated) {
      int64_t preGoupList = 0;
      // Check the increment in groupList matches x's m.
      for (size_t i = 0; i < groupNum; i++) {
        int64_t xMDimValue = (*gmmParams.x)[i]->GetViewShape().GetDim(0);
        std::string errorMessage = i == 0UL ? "groupListOptional[0]" :
          "groupListOptional[" + std::to_string(i) + "] - groupListOptional[" + std::to_string(i - 1UL) + "]";
        CHECK_COND(
          xMDimValue == (*gmmParams.groupListOptional)[i] - preGoupList, ACLNN_ERR_PARAM_INVALID, "x[%lu] dim 0 value %ld should be equal to %s %ld.",
                   i, xMDimValue, errorMessage.c_str(), (*gmmParams.groupListOptional)[i] - preGoupList);
        preGoupList = (*gmmParams.groupListOptional)[i];
      }
    } else if (ySeparated) {
      int64_t preGoupList = 0;
      // Check the increment in groupList matches y's m.
      for (size_t i = 0; i < groupNum; i++) {
        int64_t yMDimValue = (*gmmParams.y)[i]->GetViewShape().GetDim(0);
        std::string errorMessage = i == 0UL ? "groupListOptional[0]" :
          "groupListOptional[" + std::to_string(i) + "] - groupListOptional[" + std::to_string(i - 1UL) + "]";
        CHECK_COND(
          yMDimValue == (*gmmParams.groupListOptional)[i] - preGoupList, ACLNN_ERR_PARAM_INVALID, "y[%lu] dim 0 value %ld should be equal to %s %ld.",
                   i, yMDimValue, errorMessage.c_str(), (*gmmParams.groupListOptional)[i] - preGoupList);
        preGoupList = (*gmmParams.groupListOptional)[i];
      }
    } else {
      CHECK_COND(
        (*gmmParams.x)[0]->GetViewShape().GetDim(0) == groupListLastValue, ACLNN_ERR_PARAM_INVALID, "When splited axis is M, the last value of group list(%ld) must equal with x shape[0] (%ld).",
                 groupListLastValue, (*gmmParams.x)[0]->GetViewShape().GetDim(0));
    }
  }
  return ACLNN_SUCCESS;
}

static uint64_t GetGroupSize(const gmm::GroupedMatmulParams &gmmParams) {
  // When X is already split, or in scenarios where splititem is 0 or 2, X input is pre-grouped,
  // and group size can be obtained from X.
  if (gmmParams.x->Size() > 1) {
    return gmmParams.x->Size();
  }
  if (gmmParams.weight->Size() > 1) {
    return gmmParams.weight->Size();
  }
  if (gmmParams.y->Size() > 1) {
    return gmmParams.y->Size();
  }
  if (gmmParams.groupListOptional != nullptr) {
    return gmmParams.groupListOptional->Size();
  }
  if (gmmParams.groupTensorOptional != nullptr) {
    return gmmParams.groupTensorOptional->GetViewShape().GetDim(0);
  }
  // If groupList is null, weight must provide split info for x, and it must be grouped into k.
  return 1;
}

static aclnnStatus CheckDimNumAndPerGroupNum(bool isAntiquantInt4, std::tuple<size_t, size_t> tensorDimNums,
  const gert::Shape& tensorShape, const std::string& tensorType, int64_t weightKDimValue) {
  size_t tensorDimNum = std::get<0>(tensorDimNums);
  size_t expectedDimNum = std::get<1>(tensorDimNums);  // 1: the sceond element
  if (isAntiquantInt4) {
    if (tensorDimNum == expectedDimNum) {
      int64_t perGroupNum = tensorShape.GetDim(tensorDimNum - 2);
      CHECK_COND(perGroupNum > 0 && weightKDimValue % perGroupNum == 0, ACLNN_ERR_PARAM_INVALID,
                 "perGroupNum must be larger than 0, and can evenly divided by K[%ld] in A16W4-pergroup case,"
                 " but now perGroupNum is %ld.", weightKDimValue, perGroupNum);
    } else {
      CHECK_COND(tensorDimNum == expectedDimNum - 1, ACLNN_ERR_PARAM_INVALID,
                 "%s Dim must be %zu in A16W4-perchannel case, but now is %zu.",
                 tensorType.c_str(), expectedDimNum - 1, tensorDimNum);
    }
  } else {
    CHECK_COND(tensorDimNum == expectedDimNum - 1, ACLNN_ERR_PARAM_INVALID,
               "%s Dim must be %zu, but now is %zu.",
               tensorType.c_str(), expectedDimNum - 1, tensorDimNum);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckOptionalTensorList(const gmm::GroupedMatmulParams &gmmParams, const aclTensorList *tensorList,
                                           const std::string& tensorType) {
  // Check bias, scale, antiquant scale, antiquant offset length, tensor's dims and shape.
  uint64_t numTotal = GetGroupSize(gmmParams);
  uint64_t tensorSize = tensorList->Size();
  uint64_t weightGroupedSize = gmmParams.weight->Size();
  auto w0Shape = (*gmmParams.weight)[0]->GetViewShape();
  uint64_t weightNDimIdx = w0Shape.GetDimNum() - 1;
  int64_t weightKDimValue = w0Shape.GetDim(w0Shape.GetDimNum() - 2);  // -2: k axis offset
  // Check tensorList length matches weight.
  CHECK_COND(tensorSize == weightGroupedSize, ACLNN_ERR_PARAM_INVALID, "%s size[%lu] must be "
             "equal with weight size[%lu].", tensorType.c_str(), tensorSize, weightGroupedSize);
  DataType w0Dtype = (*gmmParams.weight)[0]->GetDataType();
  bool isAntiquantInt4 = (w0Dtype == DataType::DT_INT4 && tensorType.find("antiquant") != std::string::npos);
  if (gmmParams.isSingleWeight) {
    // If weight is a single tensor, tensor must also be a single tensor following weight.
    // Check tensor dimensions must be 2.
    CHECK_COND((*tensorList)[0] != nullptr, ACLNN_ERR_PARAM_INVALID,
               "%s[0] must not be nullptr, but now is nullptr.", tensorType.c_str());
    auto tensor0Shape = (*tensorList)[0]->GetViewShape();
    size_t tensorDimNum = tensor0Shape.GetDimNum();
    // 3: shape is (E,G,N),G is the perGroupNum
    CHECK_COND(CheckDimNumAndPerGroupNum(isAntiquantInt4, {tensorDimNum, 3}, tensor0Shape, tensorType, weightKDimValue)
               == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckDimNumAndPerGroupNum failed.");
    // Check the first dimension, batch size must match the group size.
    uint64_t batchSize = tensor0Shape.GetDim(0);
    CHECK_COND(batchSize == numTotal, ACLNN_ERR_PARAM_INVALID, "%s batch size[%lu] should be euqal "
               "with groupList length[%lu].", tensorType.c_str(), batchSize, numTotal);
    // Check tensor’s Ndim must match weight’s Ndim.
    int64_t weightNDimValue = w0Shape.GetDim(weightNDimIdx);
    int64_t tensorNDimValue = tensor0Shape.GetDim(tensorDimNum - 1);
    CHECK_COND(tensorNDimValue == weightNDimValue, ACLNN_ERR_PARAM_INVALID,
               "NDim[%ld] of %s should be equal with NDim[%ld] of weight.",
               tensorNDimValue, tensorType.c_str(), weightNDimValue);
  } else {
    for (uint64_t i = 0; i < numTotal; i++) {
      CHECK_COND((*tensorList)[i] != nullptr, ACLNN_ERR_PARAM_INVALID,
                 "%s[%lu] must not be nullptr, but now is nullptr.", tensorType.c_str(), i);
      // If weight is not a single tensor, each tensor dimension must be 1.
      auto tensorShape = (*tensorList)[i]->GetViewShape();
      size_t tensorDimNum = tensorShape.GetDimNum();
      auto wShape = (*gmmParams.weight)[i]->GetViewShape();
      // 2: shape is (G,N), G is the perGroupNum
      CHECK_COND(CheckDimNumAndPerGroupNum(isAntiquantInt4, {tensorDimNum, 2}, tensorShape, tensorType,
                 wShape.GetDim(0)) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckDimNumAndPerGroupNum failed.");
      // Check the NDIm of each group’s tensor must match the NDim of the same group’s weight.
      int64_t weightNDimValue = wShape.GetDim(weightNDimIdx);
      int64_t tensorNDimValue = tensorShape.GetDim(tensorDimNum - 1);
      CHECK_COND(tensorNDimValue == weightNDimValue, ACLNN_ERR_PARAM_INVALID,
                 "NDim[%ld] of %s[%lu] should be equal with NDim[%ld] of weight[%lu].",
                 tensorNDimValue, tensorType.c_str(), i, weightNDimValue, i);
    }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckPerTokenScale(const gmm::GroupedMatmulParams &gmmParams) {
  // check pertoken scale lengh, tensor's dim and shape.
  uint64_t perTokenScaleSize = gmmParams.perTokenScaleOptional->Size();
  uint64_t xGroupedSize = gmmParams.x->Size();
  uint64_t weightGroupedSize = gmmParams.weight->Size();
  uint64_t yGroupedSize = gmmParams.y->Size();
  uint64_t xMDimIdx = 0;
  // check the length of pertoken scale matches x.
  if (xGroupedSize == 1UL && yGroupedSize == 1UL) {
    CHECK_COND(perTokenScaleSize == xGroupedSize && perTokenScaleSize == 1, ACLNN_ERR_PARAM_INVALID,
               "perTokenScaleOptional size[%zu] must be 1 and equal with x size[%zu].",
               perTokenScaleSize, xGroupedSize);
    CHECK_COND((*gmmParams.perTokenScaleOptional)[0] != nullptr, ACLNN_ERR_PARAM_INVALID,
               "perTokenScaleOptional[0] must not be nullptr, but now is nullptr.");
    // If x is a single tensor, pertoken scale must also be a single tensor following x.
    // Check tensor dimensions must be 1.
    size_t tensorDimNum = (*gmmParams.perTokenScaleOptional)[0]->GetViewShape().GetDimNum();
    CHECK_COND(tensorDimNum == 1, ACLNN_ERR_PARAM_INVALID,
               "perTokenScaleOptional dim num must be 1 when x is single tensor, but now is %zu.", tensorDimNum);
    // Check the shape size of pertoken scale must match x’s MDim.
    int64_t xMDimValue = (*gmmParams.x)[0]->GetViewShape().GetDim(xMDimIdx);
    int64_t tensorMDimValue = (*gmmParams.perTokenScaleOptional)[0]->GetViewShape().GetDim(tensorDimNum - 1UL);
    CHECK_COND(tensorMDimValue == xMDimValue, ACLNN_ERR_PARAM_INVALID,
               "MDim[%ld] of perTokenScaleOptional should be equal with MDim[%ld] of x.",
               tensorMDimValue, xMDimValue);
  } else {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "per-token quant case is only supported "
            "when x, weight and y are all single tensor, but now x size is %zu, weight size is %zu, y size is %zu",
            xGroupedSize, weightGroupedSize, yGroupedSize);
    return ACLNN_ERR_PARAM_INVALID;
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorListDataType(const aclTensorList *tensorList, const DataType dtype) {
  for (size_t i = 0; i < tensorList->Size(); i++) {
    const aclTensor* tensor = (*tensorList)[i];
    OP_CHECK_NULL(tensor, continue);
    OP_CHECK_DTYPE_NOT_MATCH(tensor, dtype, return ACLNN_ERR_PARAM_INVALID);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckMatmulDataType(const gmm::GroupedMatmulParams &gmmParams, const DataType xDtype,
                                       const DataType weightDtype, const DataType yDtype, const DataType biasDtype) {
  CHECK_COND(CheckTensorListDataType(gmmParams.x, xDtype) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "GMM: x dtype does not match with required dtype[%s].", op::ToString(xDtype).GetString());
  CHECK_COND(CheckTensorListDataType(gmmParams.weight, weightDtype) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "GMM: weight dtype does not match with required dtype[%s].", op::ToString(weightDtype).GetString());
  CHECK_COND(CheckTensorListDataType(gmmParams.y, yDtype) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "GMM: y dtype does not match with required dtype[%s].", op::ToString(yDtype).GetString());
  if (gmmParams.biasOptional != nullptr) {
    CHECK_COND(CheckTensorListDataType(gmmParams.biasOptional, biasDtype) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "GMM: bias dtype does not match with required dtype[%s].", op::ToString(biasDtype).GetString());
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckNonQuantMatmulDataType(const gmm::GroupedMatmulParams &gmmParams, const DataType weightDtype) {
  DataType biasDtype = gmmParams.xDtype == DataType::DT_BF16 ? DataType::DT_FLOAT : gmmParams.xDtype;
  CHECK_RET(CheckMatmulDataType(gmmParams, gmmParams.xDtype, weightDtype, gmmParams.xDtype, biasDtype) == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static aclnnStatus IsGmmQuantEmpty(const gmm::GroupedMatmulParams &gmmParams) {
  CHECK_RET(gmmParams.scaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(gmmParams.offsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(gmmParams.perTokenScaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static aclnnStatus IsGmmAntiQuantEmpty(const gmm::GroupedMatmulParams &gmmParams) {
  CHECK_RET(gmmParams.antiquantScaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(gmmParams.antiquantOffsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckNonQuant(const gmm::GroupedMatmulParams &gmmParams) {
  CHECK_COND(IsGmmQuantEmpty(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Detected nonquant, but quant inputs is not empty!");
  CHECK_COND(IsGmmAntiQuantEmpty(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Detected nonquant, but antiquant inputs is not empty!");
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckQuantParamsDtype(const gmm::GroupedMatmulParams &gmmParams, bool isPerTokenQuant) {
  DataType yDtype = (*gmmParams.y)[0]->GetDataType();
  for (size_t i = 0; i < gmmParams.scaleOptional->Size(); i++) {
    DataType scaleDtype = (*gmmParams.scaleOptional)[i]->GetDataType();
    if (isPerTokenQuant) {
      bool isOutputBF16 = scaleDtype == DataType::DT_BF16 && yDtype == DataType::DT_BF16;
      bool isOutputFloat16 = scaleDtype == DataType::DT_FLOAT && yDtype == DataType::DT_FLOAT16;
      CHECK_COND(isOutputBF16 || isOutputFloat16, ACLNN_ERR_PARAM_INVALID,
                 "per-token quant case only supports scale data type bfloat16 with output data type bfloat16,"
                 "or scale with data type float32 when output is float16,"
                 " but now scale[%zu] has data type %s and output has data type %s!",
                 i, op::ToString(scaleDtype).GetString(), op::ToString(yDtype).GetString());
    } else {
      bool isOutputInt8 = (scaleDtype == DataType::DT_INT64 || scaleDtype == DataType::DT_UINT64) &&
                          yDtype == DataType::DT_INT8;
      bool isOutputBF16 = scaleDtype == DataType::DT_BF16 && yDtype == DataType::DT_BF16;
      bool isOutputFP16 = scaleDtype == DataType::DT_FLOAT && yDtype == DataType::DT_FLOAT16;
      CHECK_COND(isOutputInt8 || isOutputBF16 || isOutputFP16, ACLNN_ERR_PARAM_INVALID,
                 "per-channel quant case only supports scale with data type int64/uint64 when output is int8, "
                 "or data type bfloat16 when output is bfloat16, "
                 "or data type float32 when output is float16, "
                 "but scale[%zu] has data type %s and output has data type %s!",
                 i, op::ToString(scaleDtype).GetString(), op::ToString(yDtype).GetString());
    }
  }
  if (isPerTokenQuant) {
    for (size_t i = 0; i < gmmParams.perTokenScaleOptional->Size(); i++) {
      DataType perTokenScaleDtype = (*gmmParams.perTokenScaleOptional)[i]->GetDataType();
      CHECK_COND(perTokenScaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                 "per-token quant case only support perTokenScale with data type float32, "
                 "but perTokenScale[%zu] has data type %s!", i, op::ToString(perTokenScaleDtype).GetString());
    }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckGroupedMatmulQuant(const gmm::GroupedMatmulParams &gmmParams) {
  bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
  CHECK_COND(!is310P, ACLNN_ERR_PARAM_INVALID,
             "GMM: quant cases do not support on Ascend310P.");
  CHECK_COND(gmmParams.groupType != gmm::SPLIT_K, ACLNN_ERR_PARAM_INVALID,
             "GMM: quant cases do not support splited axis is K.");
  CHECK_COND(gmmParams.offsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
             "GMM: offset must be nullptr in quant, but now is not nullptr.");
  CHECK_COND(gmmParams.scaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
             "GMM: scale must not be nullptr in quant, but now is nullptr.");
  CHECK_COND(CheckOptionalTensorList(gmmParams, gmmParams.scaleOptional, "scale") == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Invalid scale.");
  bool isPerTokenQuant = gmmParams.perTokenScaleOptional != nullptr;
  CHECK_COND(CheckQuantParamsDtype(gmmParams, isPerTokenQuant) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Check quant params data type failed!");
  if (isPerTokenQuant) {
    CHECK_COND(CheckPerTokenScale(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Check perTokenScale failed!");
  }
  CHECK_COND(IsGmmAntiQuantEmpty(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Detected quant, but antiquant inputs is not empty!");
  return ACLNN_SUCCESS;
}

static int64_t GetPergroupSize(const gmm::GroupedMatmulParams &gmmParams, size_t w0DimNum, const gert::Shape& wShape, const gert::Shape& shape) {
  int64_t pergroupSize = 0;
  size_t shapeDimNum = shape.GetDimNum();
  if (gmmParams.isSingleWeight && w0DimNum > SEPARATED_WEIGHT_DIM) {  // antiquant param shape (E, N), (E, G, N)
    if (shapeDimNum > SEPARATED_WEIGHT_DIM) {
      int64_t k = wShape.GetDim(1);
      pergroupSize = k / shape.GetDim(shapeDimNum - 2);  // 2: the last 2-th index
    }
  } else {  //  antiquant param shape (N), (G, N)
    if (shapeDimNum > 1UL) {
      int64_t k = wShape.GetDim(0);
      pergroupSize = k / shape.GetDim(shapeDimNum - 2);  // 2: the last 2-th index
    }
  }
  return pergroupSize;
}

static aclnnStatus CheckGroupedMatmulAntiQuant(const gmm::GroupedMatmulParams &gmmParams) {
  DataType weightDtype = (*gmmParams.weight)[0]->GetDataType();
  DataType xDtype = gmmParams.xDtype;
  string sXtype = gmm::DTYPE_STRING.at(xDtype);
  string sWtype = gmm::DTYPE_STRING.at(weightDtype);
  CHECK_COND(GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P, ACLNN_ERR_PARAM_INVALID,
             "GMM Xtype:%s Wtype:%s: antiquant cases do not support on Ascend310P.", sXtype.c_str(), sWtype.c_str());
  CHECK_COND(gmmParams.groupType != gmm::SPLIT_K, ACLNN_ERR_PARAM_INVALID,
             "GMM Xtype:%s Wtype:%s: antiquant cases do not support splited axis is k.", sXtype.c_str(), sWtype.c_str());
  CHECK_COND(gmmParams.antiquantScaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
             "GMM Xtype:%s Wtype:%s: antiquantScale must not be nullptr in antiquant, but now is nullptr.", sXtype.c_str(), sWtype.c_str());
  CHECK_COND(gmmParams.antiquantOffsetOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
             "GMM Xtype:%s Wtype:%s: antiquantOffset must not be nullptr in antiquant, but now is nullptr.", sXtype.c_str(), sWtype.c_str());
  // check the shape of antiquantScale and antiquantOffset
  CHECK_COND(CheckOptionalTensorList(gmmParams, gmmParams.antiquantScaleOptional, "antiquantScale") == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Invalid antiquantScale");
  CHECK_COND(CheckOptionalTensorList(gmmParams, gmmParams.antiquantOffsetOptional, "antiquantOffset") == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Invalid antiquantOffset");
  DataType w0Dtype = (*gmmParams.weight)[0]->GetDataType();
  // check perGroupNum
  bool isAntiquantInt4 = w0Dtype == DataType::DT_INT4;
  if (isAntiquantInt4) {
    auto antiquantScale0Shape = (*gmmParams.antiquantScaleOptional)[0]->GetViewShape();
    size_t antiquantScale0DimNum = antiquantScale0Shape.GetDimNum();
    auto w0Shape = (*gmmParams.weight)[0]->GetViewShape();
    size_t w0DimNum = w0Shape.GetDimNum();
    int64_t pergroupSize = GetPergroupSize(gmmParams, w0DimNum, w0Shape, antiquantScale0Shape);
    CHECK_COND(!gmmParams.transposeWeight || pergroupSize % 2 == 0, ACLNN_ERR_PARAM_INVALID,  // 2: a factor
               "pergroupSize should be even when weight is transposed in A16W4-pergroup case, but now is %ld", pergroupSize);
    for (size_t i = 0; i < gmmParams.antiquantScaleOptional->Size(); ++i) {
      auto antiquantScaleShape = (*gmmParams.antiquantScaleOptional)[i]->GetViewShape();
      auto antiquantOffsetShape = (*gmmParams.antiquantOffsetOptional)[i]->GetViewShape();
      size_t antiquantScaleDimNum = antiquantScaleShape.GetDimNum();
      size_t antiquantOffsetDimNum = antiquantOffsetShape.GetDimNum();
      CHECK_COND(antiquantScaleDimNum == antiquantScale0DimNum && antiquantScale0DimNum == antiquantOffsetDimNum,
                 ACLNN_ERR_PARAM_INVALID, "antiquantScale[%zu]'s dim num[%zu] is not equal with first tensor's dim"
                 " num[%zu] or antiquantOffset[%zu]'s dim num[%zu] is not equal with antiquantScale[0]'s dim num[%zu]",
                 i, antiquantScaleDimNum, antiquantScale0DimNum, i, antiquantOffsetDimNum, antiquantScale0DimNum);
      auto wShape = (*gmmParams.weight)[i]->GetViewShape();
      int64_t pergroupSizeOfScale = GetPergroupSize(gmmParams, w0DimNum, wShape, antiquantScaleShape);
      int64_t pergroupSizeOfOffset = GetPergroupSize(gmmParams, w0DimNum, wShape, antiquantOffsetShape);
      CHECK_COND(pergroupSizeOfScale == pergroupSize && pergroupSizeOfOffset == pergroupSize, ACLNN_ERR_PARAM_INVALID,
                 "antiquantScale[%zu]'s pergroup size[%ld] or antiquantOffset[%zu]'s pergroup size[%ld]"
                 "is not the required value[%ld]", i, pergroupSizeOfScale, i, pergroupSizeOfOffset, pergroupSize);
    }
  }
  CHECK_COND(CheckTensorListDataType(gmmParams.antiquantScaleOptional, gmmParams.xDtype) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "GMM: antiquantScale dtype does not match with x dtype[%s].",
             op::ToString(gmmParams.xDtype).GetString());
  CHECK_COND(CheckTensorListDataType(gmmParams.antiquantOffsetOptional, gmmParams.xDtype) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "GMM: antiquantOffset dtype does not match with x dtype[%s].",
             op::ToString(gmmParams.xDtype).GetString());
  CHECK_COND(IsGmmQuantEmpty(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Detected antiquant, but quant inputs is not empty!");
  return ACLNN_SUCCESS;
}
static aclnnStatus Check310PlatformForFunction(const gmm::GroupedMatmulParams &gmmParams, const DataType &weightDtype, bool isNoActivation) {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
    bool isAllInputFP16 = gmmParams.xDtype == DataType::DT_FLOAT16 && weightDtype == DataType::DT_FLOAT16;
    if (gmmParams.biasOptional != nullptr) {
      isAllInputFP16 = isAllInputFP16 && (*gmmParams.biasOptional)[0]->GetDataType() == DataType::DT_FLOAT16;
    }
    CHECK_COND(isAllInputFP16, ACLNN_ERR_PARAM_INVALID, "Only float16 is supported on Ascend310P platforms.");
    CHECK_COND(isNoActivation, ACLNN_ERR_PARAM_INVALID, "Activation is not supported on Ascend310P platforms.");
  }
  return ACLNN_SUCCESS;
}
static aclnnStatus CheckFunctionQuantParams(const gmm::GroupedMatmulParams &gmmParams) {
  DataType yDtypeOrg = (*gmmParams.y)[0]->GetDataType();
  for (size_t i = 0; i < gmmParams.y->Size(); i++) {
    const aclTensor* yTensor = (*gmmParams.y)[i];
    OP_CHECK_NULL(yTensor, continue);
    DataType yDtype = yTensor->GetDataType();
    CHECK_COND(yDtype == yDtypeOrg, ACLNN_ERR_PARAM_INVALID,
               "output tensorlist has different data type, y[0] data type is %s, and y[%zu] data type id %s.",
               op::ToString(yDtypeOrg).GetString(), i, op::ToString(yDtype).GetString());
    if (!(yDtype == DataType::DT_INT8 || yDtype == DataType::DT_BF16 || yDtype == DataType::DT_FLOAT16 || yDtype == DataType::DT_INT32)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expect yDtype is int8, int32, float16 or bfloat16 in quant case, "
              "but now y[%zu] dtype is %s", i, op::ToString(yDtype).GetString());
      return ACLNN_ERR_PARAM_INVALID;
    }
  }
  if (gmmParams.biasOptional != nullptr) {
    CHECK_COND(CheckTensorListDataType(gmmParams.biasOptional, DataType::DT_INT32) == ACLNN_SUCCESS ||
                   CheckTensorListDataType(gmmParams.biasOptional, DataType::DT_BF16) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "GMM: bias dtype does not match with required dtype int32 or bfloat16.");
  }
  if (yDtypeOrg == DataType::DT_INT32) {
    return ACLNN_SUCCESS;
  }
  CHECK_COND(CheckGroupedMatmulQuant(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "CheckGroupedMatmulQuant failed.");
  return ACLNN_SUCCESS;
}
static aclnnStatus CheckA8W4AsymQuantParamsRelationship(const gmm::GroupedMatmulParams &gmmParams) {
  const op::Shape &xShape = (*gmmParams.x)[0]->GetViewShape();
  const op::Shape &weightShape = (*gmmParams.weight)[0]->GetViewShape();
  const op::Shape &biasShape = (*gmmParams.biasOptional)[0]->GetViewShape();
  const op::Shape &scaleShape = (*gmmParams.scaleOptional)[0]->GetViewShape();
  const op::Shape &perTokenScaleShape = (*gmmParams.perTokenScaleOptional)[0]->GetViewShape();
  size_t biasDim = biasShape.GetDimNum();
  size_t scaleDim = scaleShape.GetDimNum();
  size_t perTokenScaleDim = perTokenScaleShape.GetDimNum();
  int64_t e = weightShape.GetDim(0);
  int64_t m = xShape.GetDim(0);
  int64_t n = weightShape.GetDim(WEIGHT_DIM_A8W4 - 1);
  CHECK_COND(biasDim == BIAS_DIM_A8W4 && biasShape.GetDim(0) == e && biasShape.GetDim(1) == n,
      ACLNN_ERR_PARAM_INVALID, "GMM Asymmetric Quant: bias shape is invalid, "
      "must be (e, n), the current shape is (%ld, %ld)", biasShape.GetDim(0), biasShape.GetDim(1));
  CHECK_COND(scaleDim == WEIGHT_DIM_A8W4 && scaleShape.GetDim(0) == e
      && scaleShape.GetDim(1) == 1 && scaleShape.GetDim(WEIGHT_DIM_A8W4 - 1) == n, ACLNN_ERR_PARAM_INVALID,
        "GMM Asymmetric Quant: scale shape is invalid, must be (e, 1, n), the current shape is (%ld, %ld, %ld)",
        scaleShape.GetDim(0), scaleShape.GetDim(1), scaleShape.GetDim(WEIGHT_DIM_A8W4 - 1));
  CHECK_COND(perTokenScaleDim == BIAS_DIM_A8W4 && perTokenScaleShape.GetDim(0) == m
      && perTokenScaleShape.GetDim(BIAS_DIM_A8W4 - 1) == 1, ACLNN_ERR_PARAM_INVALID,
      "GMM Asymmetric Quant: perTokenScale shape is invalid, must be (m, 1), the current shape is (%ld, %ld)",
      perTokenScaleShape.GetDim(0), perTokenScaleShape.GetDim(1));
  return ACLNN_SUCCESS;
}
static aclnnStatus CheckA8W4AsymQuantParams(const gmm::GroupedMatmulParams &gmmParams) {
  DataType yDtype = (*gmmParams.y)[0]->GetDataType();
  CHECK_COND(yDtype == DataType::DT_FLOAT16, ACLNN_ERR_PARAM_INVALID,
               "GMM Asymmetric Quant: output y dtype should be float16, current dtype is %s.",
               op::ToString(yDtype).GetString());
  DataType offsetDtype = (*gmmParams.offsetOptional)[0]->GetDataType();
  CHECK_COND(offsetDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "GMM Asymmetric Quant: offset dtype does not match with required dtype float32, current dtype is %s.",
               op::ToString(offsetDtype).GetString());
  CHECK_COND(gmmParams.biasOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "GMM Asymmetric Quant: bias must not be null");
  CHECK_COND(gmmParams.scaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "GMM Asymmetric Quant: scale must not be null");
  CHECK_COND(gmmParams.perTokenScaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "GMM Asymmetric Quant: perTokenScale must not be null");
  DataType biasDtype = (*gmmParams.biasOptional)[0]->GetDataType();
  CHECK_COND(biasDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "GMM Asymmetric Quant: bias dtype does not match with required dtype float32, current dtype is %s.",
               op::ToString(biasDtype).GetString());
  DataType scaleDtype = (*gmmParams.scaleOptional)[0]->GetDataType();
  CHECK_COND(scaleDtype == DataType::DT_UINT64, ACLNN_ERR_PARAM_INVALID,
               "GMM Asymmetric Quant: scale dtype does not match with required dtype uint64, current dtype is %s.",
               op::ToString(scaleDtype).GetString());
  DataType perTokenScaleDtype = (*gmmParams.perTokenScaleOptional)[0]->GetDataType();
  CHECK_COND(perTokenScaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "GMM Asymmetric Quant: perTokenScale dtype does not match with required dtype float32, current dtype is %s.",
               op::ToString(perTokenScaleDtype).GetString());
  CHECK_COND(gmmParams.antiquantScaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
            "GMM Asymmetric Quant: antiquantScale must be nullptr.");
  CHECK_COND(gmmParams.antiquantOffsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
            "GMM Asymmetric Quant: antiquantOffset must be nullptr.");
  CHECK_COND(CheckA8W4AsymQuantParamsRelationship(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckA8W4AsymQuantParamsRelationship failed.");
  return ACLNN_SUCCESS;
}

static bool isA8W8AsymmetricQuant(const gmm::GroupedMatmulParams &gmmParams) {
  if (gmmParams.offsetOptional == nullptr) {
    return false;
  }
  const op::Shape &offsetShape = (*gmmParams.offsetOptional)[0]->GetViewShape();
  int64_t offsetDim = offsetShape.GetDimNum();
  if (offsetDim != OFFSET_DIM_A8W4) {
    return false;
  }
  const op::Shape &weightShape = (*gmmParams.weight)[0]->GetViewShape();
  if (offsetShape.GetDim(0) == weightShape.GetDim(0) && offsetShape.GetDim(1) == 1
      && offsetShape.GetDim(OFFSET_DIM_A8W4 - 1) == weightShape.GetDim(OFFSET_DIM_A8W4 - 1)) {
      return true;
  }
  return false;
}
static aclnnStatus CheckA8W4QuantParams(const gmm::GroupedMatmulParams &gmmParams) {
  if (!isA8W8AsymmetricQuant(gmmParams)) {
      return ACLNN_SUCCESS;
  }
  CHECK_COND(CheckA8W4AsymQuantParams(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckA8W4AsymQuantParams failed.");
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckA4W4ParamsShape(const gmm::GroupedMatmulParams &gmmParams) {
  const op::Shape &scaleShape = (*gmmParams.scaleOptional)[0]->GetViewShape();
  int64_t scaleDimNum = scaleShape.GetDimNum();
  CHECK_COND(scaleDimNum == PER_CHANNEL_SCALE_DIM || scaleDimNum == PER_GROUP_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
             "GMM A4W4: scale dim only support 2/3 for perchannel/pergroup, but now is %ld.\n",
             scaleDimNum);
  int64_t n = scaleShape.GetDim(scaleDimNum - 1);
  CHECK_COND((n % static_cast<int64_t>(8)) == 0, ACLNN_ERR_PARAM_INVALID, // 8 : A4W4 N need 8 agligned
                 "A4W4 n axis should align with 16, but now is %ld", n);
  if (scaleDimNum == PER_GROUP_SCALE_DIM) {
    const op::Shape &inputShape = (*gmmParams.x)[0]->GetViewShape();
    int64_t k = inputShape.GetDim(1);
    int64_t kGroupNum =  scaleShape.GetDim(1);  // 1: pergroupe scale shape is [e, G, n]
    CHECK_COND(kGroupNum != 0 && (k % kGroupNum) == 0, ACLNN_ERR_PARAM_INVALID,
                 "Pergroup A4W4: while scale shape is [e, G, n], x shape is [m,k], k needs to be divisible by G,"
                 "but now k is %ld, G is %ld.\n", k, kGroupNum);
    int64_t pergroupNum = static_cast<int64_t>(k / kGroupNum);
    CHECK_COND((pergroupNum % 2) == 0, ACLNN_ERR_PARAM_INVALID, // 2: pergroupNum should be even number
                "Pergroup A4W4: while scale shape is [e, G, n], x shape is [m,k],"
                "pergroup num(k/G) needs to be divisible by 2, but now pergroupNum is %ld.\n", pergroupNum);
  }

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckA4W4QuantParams(const gmm::GroupedMatmulParams &gmmParams) {
  CHECK_COND(gmmParams.groupListType == 0 || gmmParams.groupListType == 1, ACLNN_ERR_PARAM_INVALID,
    "GMM A4W4: groupListType only support 0 or 1, but now is %ld.", gmmParams.groupListType);
  DataType yDtype = (*gmmParams.y)[0]->GetDataType();
  CHECK_COND(yDtype == DataType::DT_FLOAT16 || yDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
             "GMM A4W4: output y dtype should be float16 or bfloat16, current dtype is %s.",
             op::ToString(yDtype).GetString());
  CHECK_COND(gmmParams.offsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
             "GMM A4W4: offset must be null.");
  CHECK_COND(gmmParams.biasOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
             "GMM A4W4: bias must be null.");

  CHECK_COND(gmmParams.scaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
             "GMM A4W4: scale must not be null.");
  DataType scaleDtype = (*gmmParams.scaleOptional)[0]->GetDataType();
  CHECK_COND(scaleDtype == DataType::DT_UINT64, ACLNN_ERR_PARAM_INVALID,
             "GMM A4W4: scale dtype does not match with required dtype uint64, current dtype is %s.",
             op::ToString(scaleDtype).GetString());

  bool isPerTokenQuant = gmmParams.perTokenScaleOptional != nullptr;
  if (isPerTokenQuant) {
    DataType perTokenScaleDtype = (*gmmParams.perTokenScaleOptional)[0]->GetDataType();
    CHECK_COND(perTokenScaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "GMM A4W4: perTokenScale dtype does not match with required dtype float32, current dtype is %s.",
               op::ToString(perTokenScaleDtype).GetString());

    CHECK_COND(CheckPerTokenScale(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "GMM A4W4: Check perTokenScale failed!");
  }
  CHECK_COND(CheckA4W4ParamsShape(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckA4W4ParamsShape failed.");
  CHECK_COND(IsGmmAntiQuantEmpty(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "GMM A4W4: Detected quant, but antiquant inputs is not empty!");
  CHECK_COND(gmmParams.groupType == gmm::SPLIT_M && gmmParams.x->Size() == 1 && gmmParams.weight->Size() == 1
             && gmmParams.y->Size() == 1, ACLNN_ERR_PARAM_INVALID,
             "A4W4 only support split m, single x, single weight, single y.");
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckFunctionParams(const gmm::GroupedMatmulParams &gmmParams) {
  DataType weightDtype = (*gmmParams.weight)[0]->GetDataType();
  bool isNoActivation = gmmParams.activeType == GMMActType::GMM_ACT_TYPE_NONE;
  CHECK_COND(Check310PlatformForFunction(
    gmmParams, weightDtype, isNoActivation) == ACLNN_SUCCESS,
    ACLNN_ERR_PARAM_INVALID, "Check310PlatformForFunction failed.");
  if (gmmParams.xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT4) {
    CHECK_COND(CheckA8W4QuantParams(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckA8W4QuantParams failed.");
    return ACLNN_SUCCESS;
  }
  if (gmmParams.xDtype == DataType::DT_INT4 && weightDtype == DataType::DT_INT4) {
    CHECK_COND(CheckA4W4QuantParams(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckA4W4QuantParams failed.");
    return ACLNN_SUCCESS;
  }
  if ((gmmParams.xDtype == DataType::DT_BF16 || gmmParams.xDtype == DataType::DT_FLOAT16 ||
       gmmParams.xDtype == DataType::DT_FLOAT) && gmmParams.xDtype == weightDtype) {
    if (gmmParams.apiVersion == gmm::GMMApiVersion::V1) {
      CHECK_COND(gmmParams.xDtype != DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                 "aclnnGroupedMatmul does not support x or weight dtype float32.");
    }
    CheckNonQuantMatmulDataType(gmmParams, weightDtype);
    CHECK_COND(isNoActivation, ACLNN_ERR_PARAM_INVALID, "non quant case dose not support activation.");
    return CheckNonQuant(gmmParams);
  }
  if (gmmParams.xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
    // quant
    DataType yDtype = (*gmmParams.y)[0]->GetDataType();
    CHECK_COND(isNoActivation || yDtype != DataType::DT_INT8 || yDtype != DataType::DT_INT32,
               ACLNN_ERR_PARAM_INVALID, "quant case with output dtype int8 or int32 dose not support activation.");
    CHECK_COND(CheckFunctionQuantParams(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckFunctionQuantParams failed.");
    return ACLNN_SUCCESS;
  }
  if ((gmmParams.xDtype == DataType::DT_BF16 || gmmParams.xDtype == DataType::DT_FLOAT16)
      && (weightDtype == DataType::DT_INT8 || weightDtype == DataType::DT_INT4)) {
    // antiquant
    DataType biasDtype = gmmParams.xDtype == DataType::DT_BF16 ? DataType::DT_FLOAT: DataType::DT_FLOAT16;
    CHECK_RET(
      CheckMatmulDataType(gmmParams, gmmParams.xDtype, weightDtype, gmmParams.xDtype, biasDtype) == ACLNN_SUCCESS,
      ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(isNoActivation, ACLNN_ERR_PARAM_INVALID, "antiquant case dose not support activation.");
    return CheckGroupedMatmulAntiQuant(gmmParams);
  }
  OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GMM: there is no matching xDtype and weightDtype pattern. "
          "case with x dtype %s and weight dtype %s is not supported.",
          op::ToString(gmmParams.xDtype).GetString(), op::ToString(weightDtype).GetString());
  return ACLNN_ERR_PARAM_INVALID;
}

static aclnnStatus CheckWeightShapeInnerAxisEven(const aclTensorList *tensorList, const size_t weightSize,
                                                 const int64_t innerAxisDimId) {
  if ((*tensorList)[0]->GetDataType() == DataType::DT_INT4) {
    for (size_t i = 0; i < weightSize; ++i) {
      int64_t n = (*tensorList)[i]->GetViewShape().GetDim(innerAxisDimId);
      // 2: a even factor
      CHECK_COND(n % 2 == 0, ACLNN_ERR_PARAM_INVALID, "weight's inner axis size[%ld] is not even!", n);
    }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus SplitMSingleXSingleWeightSingleY(const gmm::GroupedMatmulParams &gmmParams) {
  static const std::vector<std::string> TENSOR_X_WEIGHT{"x", "weight", "true"};
  static const std::vector<std::string> TENSOR_X_Y{"x", "y", "false"};
  static const std::vector<std::string> TENSOR_WEIGHT_Y{"weight", "y", "true"};
  CHECK_COND(gmmParams.splitItem == X_SEPARATED || gmmParams.splitItem == NO_SEPARATED, ACLNN_ERR_PARAM_INVALID,
             "When y is not separated, splitItem should be 2/3, but current splitItem is %ld.", gmmParams.splitItem);
  // check dim
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.x, gmm::MIN_FM_DIM, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Dim num or format of tensor in tensor list x is invalid.");
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.weight, gmm::SPLIT_M_SINGLE_WEIGHT_DIM, "weight") == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Dim num or format of tensor in tensor list weight is invalid.");

  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.y, gmm::MIN_FM_DIM, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Dim num or format of tensor in tensor list y is invalid.");
  // check shape, x(m,k), weight(b,k,n),  y(m,n)
  int64_t innerAxisDimId = 1;  // x always is not transposed, check K axis
  CHECK_COND(CheckShapeSameLengthTensorList(gmmParams.x, gmmParams.weight, {1, 1}, innerAxisDimId, TENSOR_X_WEIGHT) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "k dim value of x and weight is not matched.");
  CHECK_COND(CheckShapeSameLengthTensorList(gmmParams.x, gmmParams.y, {0, 0}, -1, TENSOR_X_Y) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "m dim value of x and y is not matched.");
  innerAxisDimId = !gmmParams.transposeWeight ? 2 : -1;  // 2:N axis index of weight. If w is not transposed, check N asix; otherwise, check k axis, which can be skiped
  // 2:N axis index of weight.
  CHECK_COND(CheckShapeSameLengthTensorList(gmmParams.weight, gmmParams.y, {2, 1}, innerAxisDimId, TENSOR_WEIGHT_Y) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "n dim value of weight and y is not matched.");
  CHECK_COND(CheckWeightShapeInnerAxisEven(gmmParams.weight, gmmParams.weight->Size(),
             gmmParams.transposeWeight ? 1 : 2) == ACLNN_SUCCESS,  // 2: axis index
             ACLNN_ERR_PARAM_INVALID, "w inner axis size should be even when weight is int4 dtype.");
  // check groupList
  size_t batchSizeWeight = (*gmmParams.weight)[0]->GetViewShape().GetDim(0);
  CHECK_COND(CheckGroupListSplitM(gmmParams, true, false, false, batchSizeWeight) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Invalid groupList.");
  return ACLNN_SUCCESS;
}

static aclnnStatus SplitMSingleXSeparatedWeightSingleY(const gmm::GroupedMatmulParams &gmmParams) {
  size_t weightSize = gmmParams.weight->Size();
  static const std::vector<std::string> TENSOR_WEIGHT_X{"Weight", "x", "true"};
  static const std::vector<std::string> TENSOR_X_Y{"x", "y", "false"};
  static const std::vector<std::string> TENSOR_WEIGHT_Y{"Weight", "y", "true"};
  std::string errorMessage = gmmParams.apiVersion != gmm::GMMApiVersion::V2 ? "When splited axis is M" : "When groupType is 0";
  CHECK_COND(gmmParams.splitItem == X_SEPARATED || gmmParams.splitItem == NO_SEPARATED, ACLNN_ERR_PARAM_INVALID,
             "When y is not separated, splitItem should be 2/3, but current splitItem is %ld.", gmmParams.splitItem);
  // check dim
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.x, gmm::MIN_FM_DIM, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Dim num or format of tensor in tensor list x is invalid.");
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.weight, SEPARATED_WEIGHT_DIM, "weight") == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Dim num or format of tensor in tensor list weight is invalid.");
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.y, gmm::MIN_FM_DIM, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Dim num or format of tensor in tensor list y is invalid.");
  // check shape, x(m,k), weight(k,n), y(m,n)
  int64_t innerAxisDimId = 1;  // x always is not transposed, check K axis
  CHECK_COND(CheckShapeDiffLengthTensorList(gmmParams.weight, gmmParams.x, {0, 1}, innerAxisDimId, TENSOR_WEIGHT_X) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "k dim value of x and weight is not matched.");
  CHECK_COND(CheckShapeSameLengthTensorList(gmmParams.x, gmmParams.y, {0, 0}, -1, TENSOR_X_Y) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "m dim value of x and y is not matched.");
  innerAxisDimId = !gmmParams.transposeWeight ? 1 : -1;  // if w is not transposed, check N asix; otherwise, check k axis, which can be skiped
  CHECK_COND(CheckShapeDiffLengthTensorList(gmmParams.weight, gmmParams.y, {1, 1}, innerAxisDimId, TENSOR_WEIGHT_Y) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "n dim value of weight and y is not matched.");
  CHECK_COND(CheckWeightShapeInnerAxisEven(gmmParams.weight, weightSize, gmmParams.transposeWeight ? 0 : 1) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "w inner axis size should be even when weight is int4 dtype.");
  // check groupList
  CHECK_COND(CheckGroupListSplitM(gmmParams, true, false, false, weightSize) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Invalid groupList.");
  return ACLNN_SUCCESS;
}

static aclnnStatus SplitMSingleXSeparatedWeightSeparatedY(const gmm::GroupedMatmulParams &gmmParams) {
  size_t ySize = gmmParams.y->Size();
  size_t weightSize = gmmParams.weight->Size();
  static const std::vector<std::string> TENSOR_WEIGHT_X{"Weight", "x", "true"};
  static const std::vector<std::string> TENSOR_Y_X{"y", "x", "false"};
  static const std::vector<std::string> TENSOR_WEIGHT_Y{"Weight", "y", "true"};
  std::string errorMessage = gmmParams.apiVersion == gmm::GMMApiVersion::V1 ? "When splited axis is M" : "When groupType is 0";
  CHECK_COND(gmmParams.splitItem == X_Y_SEPARATED || gmmParams.splitItem == Y_SEPARATED, ACLNN_ERR_PARAM_INVALID,
             "When y is separated, splitItem should be 0/1, but current splitItem is %ld.", gmmParams.splitItem);
  CHECK_COND(ySize == weightSize, ACLNN_ERR_PARAM_INVALID,
             "When y and weight are separated, size of y %lu should equal to size of weight %lu.",
             ySize, weightSize);
  // check dim
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.x, gmm::MIN_FM_DIM, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Dim num or format of tensor in tensor list x is invalid.");
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.weight, SEPARATED_WEIGHT_DIM, "weight") == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Dim num or format of tensor in tensor list weight is invalid.");
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.y, gmm::MIN_FM_DIM, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Dim num or format of tensor in tensor list y is invalid.");
  // check shape, x(m,k), weight(k,n), y(m,n)
  int64_t innerAxisDimId = 1;  // x always is not transposed, check K axis
  CHECK_COND(CheckShapeDiffLengthTensorList(gmmParams.weight, gmmParams.x, {0, 1}, innerAxisDimId, TENSOR_WEIGHT_X) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "k dim value of x and weight is not matched.");
  CHECK_COND(CheckShapeDiffLengthTensorListSplitAxis(gmmParams.y, gmmParams.x, 0, 0, TENSOR_Y_X) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "m dim value of x and y is not matched.");
  innerAxisDimId = !gmmParams.transposeWeight ? 1 : -1;  // if w is not transposed, check N asix; otherwise, check K axis, which can be skiped
  CHECK_COND(CheckShapeSameLengthTensorList(gmmParams.weight, gmmParams.y, {1, 1}, innerAxisDimId, TENSOR_WEIGHT_Y) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "n dim value of weight and y is not matched.");
  CHECK_COND(CheckWeightShapeInnerAxisEven(gmmParams.weight, weightSize, gmmParams.transposeWeight ? 0 : 1) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "w inner axis size should be even when weight is int4 dtype.");
  // check groupList
  CHECK_COND(CheckGroupListSplitM(gmmParams, true, false, true, ySize) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Invalid groupList.");
  return ACLNN_SUCCESS;
}

static aclnnStatus SplitMSeparatedXSeparatedWeightSingleY(const gmm::GroupedMatmulParams &gmmParams) {
  size_t xSize = gmmParams.x->Size();
  size_t weightSize = gmmParams.weight->Size();
  static const std::vector<std::string> TENSOR_WEIGHT_X{"Weight", "x", "true"};
  static const std::vector<std::string> TENSOR_X_Y{"x", "y", "false"};
  static const std::vector<std::string> TENSOR_WEIGHT_Y{"Weight", "y", "true"};
  std::string errorMessage = gmmParams.apiVersion != gmm::GMMApiVersion::V2 ? "When splited axis is M" : "When groupType is 0";
  CHECK_COND(gmmParams.splitItem == X_SEPARATED || gmmParams.splitItem == NO_SEPARATED, ACLNN_ERR_PARAM_INVALID,
             "When y is not separated, splitItem should be 2/3, but current splitItem is %ld.", gmmParams.splitItem);
  CHECK_COND(xSize == weightSize, ACLNN_ERR_PARAM_INVALID,
             "When x and weight are separated, size of x %lu should equal to size of weight %lu.",
             xSize, weightSize);
  // check dim
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.x, gmm::MIN_FM_DIM, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Dim num or format of tensor in tensor list x is invalid.");
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.weight, SEPARATED_WEIGHT_DIM, "weight") == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Dim num or format of tensor in tensor list weight is invalid.");
  CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.y, gmm::MIN_FM_DIM, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Dim num or format of tensor in tensor list y is invalid.");
  // check shape, x(m,k), weight(k,n), y(m,n)
  int64_t innerAxisDimId = 0;  // 0: the index of weight's K axis. x always is not transposed, check K axis
  CHECK_COND(CheckShapeSameLengthTensorList(gmmParams.weight, gmmParams.x, {0, 1}, innerAxisDimId, TENSOR_WEIGHT_X) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "k dim value of x and weight is not matched.");
  CHECK_COND(CheckShapeDiffLengthTensorListSplitAxis(gmmParams.x, gmmParams.y, 0, 0, TENSOR_X_Y) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "m dim value of x and y is not matched.");
  innerAxisDimId = !gmmParams.transposeWeight ? 1 : -1;  // if w is not transposed, check N asix; otherwise, check k axis, which can be skiped
  CHECK_COND(CheckShapeDiffLengthTensorList(gmmParams.weight, gmmParams.y, {1, 1}, innerAxisDimId, TENSOR_WEIGHT_Y) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "n dim value of weight and y is not matched.");
  CHECK_COND(CheckWeightShapeInnerAxisEven(gmmParams.weight, weightSize, gmmParams.transposeWeight ? 0 : 1) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "w inner axis size should be even when weight is int4 dtype.");
  // check groupList
  CHECK_COND(CheckGroupListSplitM(gmmParams, false, true, false, xSize) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "Invalid groupList.");
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckCaseSplitM(const gmm::GroupedMatmulParams &gmmParams) {
  size_t xSize = gmmParams.x->Size();
  size_t ySize = gmmParams.y->Size();
  size_t weightSize = gmmParams.weight->Size();
  bool apiVersionFlag = gmmParams.apiVersion == gmm::GMMApiVersion::WeightNz || gmmParams.apiVersion == gmm::GMMApiVersion::V5
    || gmmParams.apiVersion == gmm::GMMApiVersion::V4 || gmmParams.apiVersion == gmm::GMMApiVersion::V3;
  if (xSize == 1UL && weightSize == 1UL && ySize == 1UL) {
    CHECK_COND(SplitMSingleXSingleWeightSingleY(gmmParams) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "Split m, single x, single weight, single y case failed.");
    return ACLNN_SUCCESS;
  }
  if (xSize == 1UL && weightSize > 1UL && ySize == 1UL) {
    CHECK_COND(SplitMSingleXSeparatedWeightSingleY(gmmParams) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "Split m, single x, separated weight, single y case failed.");
    return ACLNN_SUCCESS;
  }
  if (xSize == 1UL && weightSize > 1UL && ySize > 1UL) {
    CHECK_COND(!(apiVersionFlag),
               ACLNN_ERR_PARAM_INVALID,
               "When grouplist is tensor, split m, single x, separated weight, separated y cases do not support.");
    CHECK_COND(SplitMSingleXSeparatedWeightSeparatedY(gmmParams) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "Split m, single x, separated weight, separated y case failed.");
    return ACLNN_SUCCESS;
  }
  if (xSize > 1UL && weightSize > 1UL && ySize == 1UL) {
    CHECK_COND(SplitMSeparatedXSeparatedWeightSingleY(gmmParams) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "Split m, separated x, separated weight, single y case failed.");
    return ACLNN_SUCCESS;
  }
  std::string errorMessage = gmmParams.apiVersion != gmm::GMMApiVersion::V2 ? "When splited axis is M" : "When groupType is 0";
  if ((apiVersionFlag) && gmmParams.isSingleWeight) {
    errorMessage = "When groupType is 0";
  }
  std::string xStatus = xSize > 1UL ? "separated" : "not separated";
  std::string weightStatus = weightSize > 1UL ? "separated" : "not separated";
  std::string yStatus = ySize > 1UL ? "separated" : "not separated";
  OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s, current case with x %s, weight %s, y %s is not supported.",
          errorMessage.c_str(), xStatus.c_str(), weightStatus.c_str(), yStatus.c_str());
  return ACLNN_ERR_PARAM_INVALID;
}

static aclnnStatus CheckCaseSplitK(const gmm::GroupedMatmulParams &gmmParams) {
  static const std::vector<std::string> TENSOR_X_WEIGHT{"x", "weight", "true"};
  static const std::vector<std::string> TENSOR_X_Y{"x", "y", "false"};
  static const std::vector<std::string> TENSOR_WEIGHT_Y{"Weight", "y", "true"};
  size_t xSize = gmmParams.x->Size();
  size_t ySize = gmmParams.y->Size();
  size_t weightSize = gmmParams.weight->Size();
  if (xSize == 1UL) {
    // The left matrix must be transposed.
    CHECK_COND(gmmParams.transposeX, ACLNN_ERR_PARAM_INVALID,
               "When groupType is 2 and x is not separated, tensor in x should be transposed.");
    // check dim
    CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.x, gmm::MIN_FM_DIM, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Dim num or format of tensor in tensor list x is invalid.");
    CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.weight, gmm::SPLIT_K_SINGLE_WEIGHT_DIM, "weight") == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID,
               "Dim num or format of tensor in tensor list weight is invalid.");
    // 3:y is 3 Dims in single-tensor case when split K.
    if(weightSize == 1UL && ySize == 1UL) {
      CHECK_COND(CheckDimNumAndFormat(gmmParams, gmmParams.y, DIMS_THREE_FOR_GMM, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                "Dim num or format of tensor in tensor list y is invalid.");
      // check shape, x(m,k), weight(k,n), y(b,m,n)
      int64_t innerAxisDimId = 0;  // x always is transposed, check M axis

      CHECK_COND(CheckShapeSameLengthTensorList(gmmParams.x, gmmParams.weight, {1, 0}, innerAxisDimId, TENSOR_X_WEIGHT) == ACLNN_SUCCESS,
                ACLNN_ERR_PARAM_INVALID, "k dim value of x and weight is not matched.");
      CHECK_COND(CheckShapeSameLengthTensorList(gmmParams.x, gmmParams.y, {0, 1}, -1, TENSOR_X_Y) == ACLNN_SUCCESS,
                ACLNN_ERR_PARAM_INVALID, "m dim value of x and y is not matched.");
      innerAxisDimId = 1;  // w always is not transposed, check N axis
      // 2:N axis index of y
      CHECK_COND(CheckShapeSameLengthTensorList(gmmParams.weight, gmmParams.y, {1, 2}, innerAxisDimId, TENSOR_WEIGHT_Y) == ACLNN_SUCCESS,
                ACLNN_ERR_PARAM_INVALID, "n dim value of weight and y is not matched.");
      // check groupList
      size_t batchSizeY = (*gmmParams.y)[0]->GetViewShape().GetDim(0);
      CHECK_COND(CheckGroupListSplitK(gmmParams, true, false, false, batchSizeY) == ACLNN_SUCCESS,
                ACLNN_ERR_PARAM_INVALID, "Invalid groupList.");
    }
    return ACLNN_SUCCESS;
  }
  OP_LOGE(ACLNN_ERR_PARAM_INVALID,"When groupType is 2, only support case with unseparated x, y, "
          "but now x size is %lu, weight size is %lu, y size is %lu.", xSize, weightSize, ySize);
  return ACLNN_ERR_PARAM_INVALID;
}

static aclnnStatus CheckCaseNoSplit(const gmm::GroupedMatmulParams &gmmParams) {
  // When groupType is -1, splitItem mast be 0/1.
  CHECK_COND(gmmParams.splitItem == X_Y_SEPARATED || gmmParams.splitItem == Y_SEPARATED, ACLNN_ERR_PARAM_INVALID,
             "When y is separated, splitItem should be 0/1, but current splitItem is %ld.", gmmParams.splitItem);
  // 校验group num
  size_t xSize = gmmParams.x->Size();
  size_t ySize = gmmParams.y->Size();
  size_t weightSize = gmmParams.weight->Size();
  CHECK_COND(xSize == ySize, ACLNN_ERR_PARAM_INVALID,
             "When y is separated, size of x %lu should equal to size of y %lu.", xSize, ySize);
  CHECK_COND(xSize == weightSize, ACLNN_ERR_PARAM_INVALID,
             "When x and weight are separated, size of x %lu should equal to size of weight %lu.",
             xSize, weightSize);
  // check dim
  CHECK_COND(CheckDimNumAndGroupListNoSplitAndFormat(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Dim num or format of tensor in tensor lists or grouplist is invalid.");
  // check shape
  for (size_t i = 0; i < xSize; i++) {
    size_t xDimNum = (*gmmParams.x)[i]->GetViewShape().GetDimNum();
    // 2: Indicates validation up to the second last dimension, x and y must be equal in every dimension except the last one.
    for (size_t dimIdx = 0UL; dimIdx < xDimNum - 2UL; dimIdx++) {
      size_t xDimValue = (*gmmParams.x)[i]->GetViewShape().GetDim(dimIdx);
      size_t yDimValue = (*gmmParams.y)[i]->GetViewShape().GetDim(dimIdx);
      CHECK_COND(xDimValue == yDimValue, ACLNN_ERR_PARAM_INVALID,
                 "y[%lu] dim %lu value %lu should equal to x[%lu] dim %lu value %lu.",
                 i, dimIdx, xDimValue, i, dimIdx, yDimValue);
    }
    // check the inner dim of x is less than 65535
    size_t xKDimValue = (*gmmParams.x)[i]->GetViewShape().GetDim(xDimNum - 1UL);  // x always is not transposed
    CHECK_COND(xKDimValue <= MAX_INNER_AXIS, ACLNN_ERR_PARAM_INVALID,
               "x[%lu] dim %lu value %lu should less or equal to 65535.", i, xDimNum - 1, xKDimValue);
    size_t weightKDimValue = (*gmmParams.weight)[i]->GetViewShape().GetDim(0);
    CHECK_COND(xKDimValue == weightKDimValue, ACLNN_ERR_PARAM_INVALID,
               "x[%lu] dim %lu value %lu should equal to weight[%lu] dim 0 value %lu.",
               i, xDimNum - 1, xKDimValue, i, weightKDimValue);
    size_t weightNDimValue = (*gmmParams.weight)[i]->GetViewShape().GetDim(1);
    if (!gmmParams.transposeWeight) {  // if weight is not transposed, check N aisx; otherwise, check K axis, which can be skiped
      CHECK_COND(weightNDimValue <= MAX_INNER_AXIS, ACLNN_ERR_PARAM_INVALID,
                "w[%lu] dim %d value %lu should less or equal to 65535.", i, 1, weightNDimValue);
    }
    if ((*gmmParams.weight)[0]->GetDataType() == DataType::DT_INT4) {
      CHECK_COND(weightNDimValue % 2 == 0, ACLNN_ERR_PARAM_INVALID,  // 2: an even factor
                 "w[%lu] dim %d value %lu should be even when weight is int4 dtype.", i, 1, weightNDimValue);
    }
    // check y[n]=weight[n]
    size_t yNDimValue = (*gmmParams.y)[i]->GetViewShape().GetDim(xDimNum - 1UL);
    CHECK_COND(yNDimValue == weightNDimValue, ACLNN_ERR_PARAM_INVALID,
                 "y[%lu] dim %lu value %lu should equal to weight[%lu] dim 1 value %lu.",
                 i, xDimNum - 1, yNDimValue, i, weightNDimValue);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamDifferentGroupType(const gmm::GroupedMatmulParams &gmmParams) {
  CHECK_COND(!(gmmParams.transposeX && gmmParams.transposeWeight), ACLNN_ERR_PARAM_INVALID,
             "x and weight can not be transposed at the same time.");
  CHECK_COND((gmmParams.groupListOptional == nullptr || gmmParams.groupListOptional->Size() >= 1) &&
             (gmmParams.groupTensorOptional == nullptr || gmmParams.groupTensorOptional->GetViewShape().GetDim(0) >= 1),
             ACLNN_ERR_PARAM_INVALID, "size of groupList can not be 0."
             "If expected group num is 1, groupList should be nullptr.");
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P && gmmParams.transposeWeight) {
    CHECK_COND(gmmParams.groupType == gmm::SPLIT_M && gmmParams.x->Size() == 1 && gmmParams.weight->Size() == 1
               && gmmParams.y->Size() == 1, ACLNN_ERR_PARAM_INVALID,
               "When transpose weight, ASCEND310P only support split m, single x, single weight, single y.");
  }
  if (gmmParams.groupType == gmm::NO_SPLIT) {
    CHECK_COND(!gmmParams.transposeX, ACLNN_ERR_PARAM_INVALID,
               "When x, weight and y are all separated, x can not be transposed.");
    CHECK_COND(!(gmmParams.apiVersion == gmm::GMMApiVersion::V1 && gmmParams.transposeWeight), ACLNN_ERR_PARAM_INVALID,
               "in this version, when x, weight and y are all separated, weight can not be transposed.");
    CHECK_COND(CheckCaseNoSplit(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Invalid inputs!");
  } else if (gmmParams.groupType == gmm::SPLIT_M) {
    std::string errorMessage = gmmParams.apiVersion != gmm::GMMApiVersion::V2 && !gmmParams.isSingleWeight
                               ? "When splited axis is M" : "When groupType is 0";
    CHECK_COND(!gmmParams.transposeX, ACLNN_ERR_PARAM_INVALID,
               "%s, x can not be transposed.", errorMessage.c_str());
    CHECK_COND(CheckCaseSplitM(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Invalid inputs!");
  } else if (gmmParams.groupType == gmm::SPLIT_K) {
    CHECK_COND(gmmParams.biasOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "When groupType is 2, bias must be empty.");
    CHECK_COND(CheckCaseSplitK(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Invalid inputs!");
  }
  if (gmmParams.biasOptional != nullptr) {
    CHECK_COND(CheckOptionalTensorList(gmmParams, gmmParams.biasOptional, "bias") == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "Invalid bias!");
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckTuningConfig(const gmm::GroupedMatmulParams &gmmParams) {
  size_t tensorLength = gmmParams.x->Size();
  int64_t maxM = 0;
  for (size_t i = 0; i < tensorLength; ++i) {
    maxM = std::max(maxM, (*gmmParams.x)[i]->GetViewShape().GetDim(0));
  }
  if (gmmParams.tuningConfigOptional != nullptr && gmmParams.tuningConfigOptional->Size() > 0) {
    CHECK_COND((*gmmParams.tuningConfigOptional)[0] >= 0 && (*gmmParams.tuningConfigOptional)[0] <= maxM,
                ACLNN_ERR_PARAM_INVALID,
                "Invalid tuningConfigOptional (%ld)! It should be a non-negative num, and less than or equal to maxM: %ld",
                (*gmmParams.tuningConfigOptional)[0], maxM);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorListLength(const aclTensorList *tensorList) {
  size_t groupSize = 0;
  if (tensorList != nullptr) {
      groupSize = tensorList->Size();
  }
  CHECK_COND(
    groupSize <= MAX_GROUP_LIST_SIZE_ARRAY, ACLNN_ERR_PARAM_INVALID, "Length of tensorList should not exceed %ld, but actually got %lu.",
             MAX_GROUP_LIST_SIZE_ARRAY, groupSize);
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckGroupSize(const gmm::GroupedMatmulParams &gmmParams) {
  // Only groupSizes of necessary inputs will be checked here.
  // The groupSizes of optional inputs and output will be checked in subsequent steps.
  CHECK_COND(CheckTensorListLength(gmmParams.x) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Invalid length of tensorList x.");
  CHECK_COND(CheckTensorListLength(gmmParams.weight) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Invalid length of tensorList weight.");

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckUnusedParams(const gmm::GroupedMatmulParams &gmmParams) {
  // Check currently disabled parameters, delete accordingly when parameter functionality is supported.
  CHECK_COND(gmmParams.activationInputOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
             "activationInputOptional must be nullptr.");
  CHECK_COND(gmmParams.activationQuantScaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
             "activationQuantScaleOptional must be nullptr.");
  CHECK_COND(gmmParams.activationQuantOffsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
             "activationQuantOffsetOptional must be nullptr.");
  CHECK_COND(gmmParams.activationFeatureOutOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
             "activationFeatureOutOptional must be nullptr.");
  CHECK_COND(gmmParams.dynQuantScaleOutOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
             "dynQuantScaleOutOptional must be nullptr.");
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckParam(const gmm::GroupedMatmulParams &gmmParams) {
  CHECK_COND(CheckUnusedParams(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Invalid unused params.");
  CHECK_RET(CheckFunctionParams(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckParamDifferentGroupType(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckGroupSize(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckTuningConfig(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static void CheckOptionalTensorListEmpty(const aclTensorList *&tensorList) {
  if (tensorList != nullptr) {
    if (tensorList->Size() == 0) {
      tensorList = nullptr;
    } else if ((*tensorList)[0] == nullptr) {
      tensorList = nullptr;
    } else if (tensorList->Size() == 1) {
      op::Shape shape = (*tensorList)[0]->GetViewShape();
      if (shape.GetDimNum() == 1 && shape.GetDim(0) == 0) {
        tensorList = nullptr;
      }
    }
  }
}

static void ResetEmptyTensor(gmm::GroupedMatmulParams &gmmParams) {
  // set the empty tensor list to nullptr
  if (gmmParams.groupListOptional != nullptr && gmmParams.groupListOptional->Size() == 0) {
    gmmParams.groupListOptional = nullptr;
  }
  CheckOptionalTensorListEmpty(gmmParams.biasOptional);
  CheckOptionalTensorListEmpty(gmmParams.scaleOptional);
  CheckOptionalTensorListEmpty(gmmParams.offsetOptional);
  CheckOptionalTensorListEmpty(gmmParams.antiquantScaleOptional);
  CheckOptionalTensorListEmpty(gmmParams.antiquantOffsetOptional);
  CheckOptionalTensorListEmpty(gmmParams.perTokenScaleOptional);
  CheckOptionalTensorListEmpty(gmmParams.activationInputOptional);
  CheckOptionalTensorListEmpty(gmmParams.activationQuantScaleOptional);
  CheckOptionalTensorListEmpty(gmmParams.activationQuantOffsetOptional);
  CheckOptionalTensorListEmpty(gmmParams.activationFeatureOutOptional);
  CheckOptionalTensorListEmpty(gmmParams.dynQuantScaleOutOptional);
}

static void CreateEmptyTensor(const aclDataType dataType, const aclTensorList *&gmmTensorList,
                              aclTensorList *&tensorList, aclOpExecutor *executor) {
  // if tensor list is nullptr, convert tensorlist to a tensorlist containing a tensor with shape 0.
  if (gmmTensorList == nullptr) {
    FVector<aclTensor*> emptyTensors;
    aclTensor *emptyTensor = executor->AllocTensor({0}, static_cast<op::DataType>(dataType));
    emptyTensors.emplace_back(emptyTensor);
    tensorList = executor->AllocTensorList(emptyTensors.data(), emptyTensors.size());
    gmmTensorList = tensorList;
  }
}

static aclnnStatus DataContiguous(const aclTensorList *&tensors, aclOpExecutor *executor) {
    std::vector<const aclTensor *> tensorsVec;
    const aclTensor *contiguousTensor = nullptr;
    for (size_t i = 0; i < tensors->Size(); ++i) {
        const aclTensor *tensor = (*tensors)[i];
        contiguousTensor = l0op::Contiguous(tensor, executor);
        CHECK_RET(contiguousTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorsVec.push_back(contiguousTensor);
    }
    tensors = executor->AllocTensorList(tensorsVec.data(), tensorsVec.size());
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguousAndTransFormat(const aclTensor *tensor, const aclTensor *&reformatedTensor,
                                                const op::Format requiredFormat, aclOpExecutor *executor) {
  CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  if (!op::IsPrivateFormat(tensor->GetStorageFormat()) && tensor->GetStorageFormat() != op::Format::FORMAT_ND) {
    tensor = l0op::ReFormat(tensor, op::Format::FORMAT_ND, executor);
  }
  if (tensor == nullptr || tensor->GetViewShape().GetDimNum() == 1) {
    OP_LOGD("No need to do contiguous process");
    reformatedTensor = tensor;
  } else {
    reformatedTensor = l0op::Contiguous(tensor, executor);
  }
  CHECK_RET(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  reformatedTensor = l0op::TransData(reformatedTensor, requiredFormat, 1, executor);
  CHECK_RET(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static aclnnStatus TransWeightToNz(gmm::GroupedMatmulParams &gmmParams, aclOpExecutor *executor) {
  bool is310p = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
  if (is310p) {
    const aclTensorList *&weights = gmmParams.weight;
    size_t wLength = weights->Size();
    std::vector<const aclTensor*> reformatedWeightVec;
    const aclTensor* reformatedWeight = nullptr;
    // trans weight format
    for (size_t i(0); i < wLength; ++i) {
      const aclTensor* weight = (*weights)[i];
      op::Shape shape = weight->GetViewShape();
      // 2: When weight is transposed, n is the second last axis.
      int64_t n = gmmParams.transposeWeight ? shape.GetDim(shape.GetDimNum() - 2) : shape.GetDim(shape.GetDimNum() - 1);
      // 32: matmul api requires n axis aligning with 32 bytes
      CHECK_COND(n % static_cast<int64_t>(32 / std::max<int>(1, op::TypeSize(weight->GetDataType()))) == 0, ACLNN_ERR_PARAM_INVALID,
                 "output n axis should align with 32 Bytes, but now is %ld", n);
      aclnnStatus ret = DataContiguousAndTransFormat(weight, reformatedWeight,
                                                     Format::FORMAT_FRACTAL_NZ, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ret);
      reformatedWeightVec.push_back(reformatedWeight);
    }
    weights = executor->AllocTensorList(reformatedWeightVec.data(), reformatedWeightVec.size());
  } else {  // 910
    const aclTensorList *&weights = gmmParams.weight;
    const aclTensorList *&x = gmmParams.x;
    CHECK_COND((*x)[0] != nullptr, ACLNN_ERR_PARAM_INVALID, "The first tensor of x is nullptr!");
    auto xDtype = (*x)[0]->GetDataType();
    size_t wLength = weights->Size();
    for (size_t i(0); i < wLength; ++i) {
      const aclTensor* weight = (*weights)[i];
      if (weight->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        break;
      }
      size_t viewDimNum = weight->GetViewShape().GetDimNum();
      uint64_t k = gmmParams.transposeWeight ? weight->GetViewShape().GetDim(viewDimNum - 1)
                             : weight->GetViewShape().GetDim(viewDimNum - gmm::LAST_SECOND_DIM_INDEX);
      uint64_t n = gmmParams.transposeWeight ? weight->GetViewShape().GetDim(viewDimNum - gmm::LAST_SECOND_DIM_INDEX)
                             : weight->GetViewShape().GetDim(viewDimNum - 1);
      bool k_align = false;
      bool n_align = false;
      if (weight->GetDataType() == DataType::DT_INT8) {
          k_align = gmmParams.transposeWeight ? k % ALIGN_NZ_INT8_N == 0 : k % ALIGN_NZ_K == 0;
          n_align = gmmParams.transposeWeight ? n % ALIGN_NZ_K == 0 : n % ALIGN_NZ_INT8_N == 0;
      } else if (weight->GetDataType() == DataType::DT_BF16 || weight->GetDataType() == DataType::DT_FLOAT16 ||
                 (weight->GetDataType() == DataType::DT_FLOAT4_E2M1 &&
                  (xDtype == DataType::DT_FLOAT16 || xDtype == DataType::DT_BF16))) {
          k_align = k % ALIGN_NZ_K == 0;
          n_align = n % ALIGN_NZ_K == 0;
      } else if (weight->GetDataType() == DataType::DT_INT4) {
          k_align = gmmParams.transposeWeight ? k % ALIGN_NZ_INT4_N == 0 : k % ALIGN_NZ_K == 0;
          n_align = gmmParams.transposeWeight ? n % ALIGN_NZ_K == 0 : n % ALIGN_NZ_INT4_N == 0;
      }
      CHECK_COND(k_align == true && n_align == true, ACLNN_ERR_PARAM_INVALID,
                 "When weight(%s) format is FRACTAL_NZ, weight'shape(k[%lu], n[%lu]) should be divisible by the "
                 "following shape: INT8:[16, 32],BF16/FP16[16, 16],INT4[16, 64]). If the weight is transposed,"
                 "the k/n need to be reversed.", op::ToString(weight->GetDataType()).GetString(), k, n);
      continue;
    }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckZeroShape(gmm::GroupedMatmulParams &params, uint64_t *workspaceSize) {
    bool isEmpty = true;
    for (size_t i = 0; i < params.x->Size(); ++i) {
        if (!((*params.x)[i]->IsEmpty())) {
            isEmpty = false;
            break;
        }
    }
    if (isEmpty) {
        *workspaceSize = 0UL;
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static void SetParamsTensorEmpty(gmm::GroupedMatmulParams &params, aclOpExecutor *executor) {
  aclTensorList *emptyBiasList = nullptr;
  DataType weightDtype = (*(params.weight))[0]->GetDataType();
  if(params.xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT4) {  // A8W4 MSD
    CreateEmptyTensor(aclDataType::ACL_FLOAT, params.biasOptional, emptyBiasList, executor);
  }
  else { CreateEmptyTensor(gmm::BIAS_DTYPE.at(params.xDtype), params.biasOptional, emptyBiasList, executor); }
  aclTensorList *emptyScaleList = nullptr;
  CreateEmptyTensor(aclDataType::ACL_UINT64, params.scaleOptional, emptyScaleList, executor);

  aclTensorList *emptyOffsetList = nullptr;
  CreateEmptyTensor(aclDataType::ACL_FLOAT, params.offsetOptional, emptyOffsetList, executor);

  aclTensorList *emptyAntiquantScaleList = nullptr;
  CreateEmptyTensor(aclDataType::ACL_FLOAT16, params.antiquantScaleOptional, emptyAntiquantScaleList, executor);

  aclTensorList *emptyAntiquantOffsetList = nullptr;
  CreateEmptyTensor(aclDataType::ACL_FLOAT16, params.antiquantOffsetOptional, emptyAntiquantOffsetList, executor);

  aclTensorList *emptyPerTokenScaleList = nullptr;
  CreateEmptyTensor(aclDataType::ACL_FLOAT, params.perTokenScaleOptional, emptyPerTokenScaleList, executor);

  aclTensorList *emptyActivationInputList = nullptr;
  CreateEmptyTensor(aclDataType::ACL_FLOAT, params.activationInputOptional, emptyActivationInputList, executor);

  aclTensorList *emptyActivationQuantScaleList = nullptr;
  CreateEmptyTensor(aclDataType::ACL_FLOAT, params.activationQuantScaleOptional, emptyActivationQuantScaleList, executor);

  aclTensorList *emptyActivationQuantOffsetList = nullptr;
  CreateEmptyTensor(aclDataType::ACL_FLOAT, params.activationQuantOffsetOptional, emptyActivationQuantOffsetList, executor);

  aclTensorList *emptyActivationFeatureOutList = nullptr;
  CreateEmptyTensor(aclDataType::ACL_FLOAT, params.activationFeatureOutOptional, emptyActivationFeatureOutList, executor);

  aclTensorList *emptyDynQuantScaleOutList = nullptr;
  CreateEmptyTensor(aclDataType::ACL_FLOAT, params.dynQuantScaleOutOptional, emptyDynQuantScaleOutList, executor);
}

static aclnnStatus CheckOutputShape(const aclTensorList* l0Res, const aclTensorList* y) {
  CHECK_COND(l0Res->Size() == y->Size(), ACLNN_ERR_PARAM_INVALID, "Output tensor list length is not right.");
  for (size_t i = 0; i < y->Size(); ++i) {
    auto const &resShape = (*l0Res)[i]->GetViewShape();
    auto const &yShape = (*y)[i]->GetViewShape();
    if (resShape != yShape) {
      if (!(resShape.GetShapeSize() == 1 && yShape.GetShapeSize() == 1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensorlist Output %lu tensor's shape[%s] is not equal with infered output's shape[%s].", i,
                op::ToString(yShape).GetString(), op::ToString(resShape).GetString());
        return ACLNN_ERR_PARAM_INVALID;
      }
    }
  }
  return ACLNN_SUCCESS;
}

static void SetTransposedTensorListContiguous(gmm::GroupedMatmulParams &params, aclOpExecutor *executorPtr)
{
  bool isPerTileQuantMode = false;
  DataType weightDtype = (*params.weight)[0]->GetDataType();
  if (params.transposeX) {
    std::vector<aclTensor*> xTensorList;
    gmm::CreateContiguousTensorList(params.x, xTensorList, executorPtr);
    params.x = executorPtr->AllocTensorList(xTensorList.data(), xTensorList.size());
    if (params.perTokenScaleOptional != nullptr &&
        ((*params.perTokenScaleOptional)[0]->GetDataType() == DataType::DT_FLOAT8_E8M0 || isPerTileQuantMode)) {
      std::vector<aclTensor*> perTokenScaleTensorList;
      if (isPerTileQuantMode) {
        gmm::CreateContiguousTensorList(params.perTokenScaleOptional, perTokenScaleTensorList, executorPtr);
      } else {
        gmm::CreateContiguousTensorListForPertoken(params.perTokenScaleOptional, perTokenScaleTensorList, executorPtr);
      }
      params.perTokenScaleOptional = executorPtr->AllocTensorList(perTokenScaleTensorList.data(), perTokenScaleTensorList.size());
    }
  }
  if (params.transposeWeight) {
    std::vector<aclTensor *> weightTensorList;
    gmm::CreateContiguousTensorList(params.weight, weightTensorList, executorPtr);
    params.weight = executorPtr->AllocTensorList(weightTensorList.data(), weightTensorList.size());
    if (params.scaleOptional != nullptr) {
      std::vector<aclTensor *> scaleTensorList;
      if ((*params.scaleOptional)[0]->GetDataType() == DataType::DT_FLOAT8_E8M0) {
        gmm::CreateContiguousTensorListForMXTypeMScale(params.scaleOptional, scaleTensorList, executorPtr);
        params.scaleOptional = executorPtr->AllocTensorList(scaleTensorList.data(), scaleTensorList.size());
      } else if (isPerTileQuantMode) {
        gmm::CreateContiguousTensorList(params.scaleOptional, scaleTensorList, executorPtr);
        params.scaleOptional = executorPtr->AllocTensorList(scaleTensorList.data(), scaleTensorList.size());
      }
    }
  }
}

static aclnnStatus ParamsDataContiguous(gmm::GroupedMatmulParams &params, aclOpExecutor *executorPtr) {
  CHECK_COND(DataContiguous(params.x, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Contiguous x failed.");  // make x contiguous
  DataType xDtype = (*params.x)[0]->GetDataType();
  DataType weightDtype = (*params.weight)[0]->GetDataType();
  if (!(params.apiVersion == gmm::GMMApiVersion::WeightNz && (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16))) {
      CHECK_COND(DataContiguous(params.weight, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                 "Contiguous weight failed.");  // make w contiguous
  }
  CHECK_COND(DataContiguous(params.biasOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Contiguous biasOptional failed.");
  CHECK_COND(DataContiguous(params.scaleOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Contiguous scaleOptional failed.");
  CHECK_COND(DataContiguous(params.offsetOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Contiguous offsetOptional failed.");
  CHECK_COND(DataContiguous(params.antiquantScaleOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Contiguous antiquantScaleOptional failed.");
  CHECK_COND(DataContiguous(params.antiquantOffsetOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Contiguous antiquantOffsetOptional failed.");
  CHECK_COND(DataContiguous(params.perTokenScaleOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Contiguous perTokenScaleOptional failed.");
  if (params.groupTensorOptional != nullptr) {
    params.groupTensorOptional = l0op::Contiguous(params.groupTensorOptional, executorPtr);
    CHECK_COND(params.groupTensorOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "Contiguous groupTensorOptional failed.");
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightQuantGMMWeightNz(DataType x1Dtype, DataType weightDtype, DataType yDtype) {
    if (x1Dtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT4) {
        CHECK_COND(
            yDtype == DataType::DT_FLOAT16 || yDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
            "The dtypes of x[%s]-weight[%s]-y[%s] do not match with required dtype.The x-weight-y of the antiquant"
            "case[A8W4] only supports the following combinations: INT8-INT4-BF16,INT8-INT4-Fp16",
            op::ToString(x1Dtype).GetString(), op::ToString(weightDtype).GetString(), op::ToString(yDtype).GetString());
        return ACLNN_SUCCESS;
    } else if ((x1Dtype == DataType::DT_FLOAT16 || x1Dtype == DataType::DT_BF16) &&
               weightDtype == DataType::DT_FLOAT4_E2M1) {
        CHECK_COND(
            yDtype == DataType::DT_FLOAT16 || yDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
            "The dtypes of x[%s]-weight[%s]-y[%s] do not match with required dtype.The x-weight-y of the antiquant"
            "case[A16mxFp4] only supports the following combinations: Fp16-Fp4_e2m1-Fp16,BF16-Fp4_e2m1-BF16",
            op::ToString(x1Dtype).GetString(), op::ToString(weightDtype).GetString(), op::ToString(yDtype).GetString());
        return ACLNN_SUCCESS;
    }
    return ACLNN_ERR_PARAM_INVALID;
}

static aclnnStatus CheckQuantGMMWeightNz(DataType x1Dtype, DataType weightDtype, DataType yDtype) {
    if (x1Dtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
        CHECK_COND(
            yDtype == DataType::DT_FLOAT16 || yDtype == DataType::DT_BF16 || yDtype == DataType::DT_INT32,
            ACLNN_ERR_PARAM_INVALID,
            "The dtypes of x[%s]-weight[%s]-y[%s] do not match with required dtype.The x-weight-y of the quant case"
            "only supports the following combinations: INT8-INT8-BF16,INT8-INT8-Fp16,INT8-INT8-INT32",
            op::ToString(x1Dtype).GetString(), op::ToString(weightDtype).GetString(), op::ToString(yDtype).GetString());
        return ACLNN_SUCCESS;
    } else if (x1Dtype == DataType::DT_INT4 && weightDtype == DataType::DT_INT4) {
        CHECK_COND(
            yDtype == DataType::DT_FLOAT16 || yDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
            "The dtypes of x[%s]-weight[%s]-y[%s] do not match with required dtype.The x-weight-y of the antiquant"
            "case[A4W4] only supports the following combinations: INT4-INT4-BF16,INT4-INT4-Fp16",
            op::ToString(x1Dtype).GetString(), op::ToString(weightDtype).GetString(), op::ToString(yDtype).GetString());
        return ACLNN_SUCCESS;
    }
    return ACLNN_ERR_PARAM_INVALID;
}

static aclnnStatus CheckNoQuantGMMWeightNz(DataType x1Dtype, DataType weightDtype, DataType yDtype) {
    if (x1Dtype == DataType::DT_BF16 && weightDtype == DataType::DT_BF16) {
        CHECK_COND(
            yDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
            "The dtypes of x[%s]-weight[%s]-y[%s] do not match with required dtype."
            "The x-weight-y of the antiquant case[BF16] only supports the following combinations: BF16-BF16-BF16",
            op::ToString(x1Dtype).GetString(), op::ToString(weightDtype).GetString(), op::ToString(yDtype).GetString());
        return ACLNN_SUCCESS;
    } else if (x1Dtype == DataType::DT_FLOAT16 && weightDtype == DataType::DT_FLOAT16) {
        CHECK_COND(
            yDtype == DataType::DT_FLOAT16, ACLNN_ERR_PARAM_INVALID,
            "The dtypes of x[%s]-weight[%s]-y[%s] do not match with required dtype. The x-weight-y of the antiquant"
            "case[FLOAT16] only supports the following combinations: FLOAT16-FLOAT16-FLOAT16",
            op::ToString(x1Dtype).GetString(), op::ToString(weightDtype).GetString(), op::ToString(yDtype).GetString());
        return ACLNN_SUCCESS;
    }
    return ACLNN_ERR_PARAM_INVALID;
}

static aclnnStatus ParamsWeightNzDtype(gmm::GroupedMatmulParams &params) {
    DataType x1Dtype = params.xDtype;
    DataType weightDtype = (*params.weight)[0]->GetDataType();
    DataType yDtype = (*params.y)[0]->GetDataType();
    if (CheckWeightQuantGMMWeightNz(x1Dtype, weightDtype, yDtype) == ACLNN_SUCCESS ||
        CheckQuantGMMWeightNz(x1Dtype, weightDtype, yDtype) == ACLNN_SUCCESS ||
        CheckNoQuantGMMWeightNz(x1Dtype, weightDtype, yDtype) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The dtypes of x[%s]-weight[%s] do not match with required dtype."
            "Only supported x-weight: INT8-INT8,BF16-BF16,FP16-FP16,INT8-INT4, INT4-INT4,FP16/BF16-FP4_E2M1",
            op::ToString(x1Dtype).GetString(), op::ToString(weightDtype).GetString());
    return ACLNN_ERR_PARAM_INVALID;
}

static const aclTensor *SetTensorToNZFormat(const aclTensor *input, op::Shape &shape, aclOpExecutor *executor) {
    auto formatTensor = executor->CreateView(input, shape, input->GetViewOffset());
    formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetOriginalFormat(input->GetViewFormat());
    formatTensor->SetViewShape(input->GetViewShape());
    return formatTensor;
}

static aclnnStatus SetStorageShape(gmm::GroupedMatmulParams &params, op::Shape wqbmmNzShape) {
    DataType xDtype = (*params.x)[0]->GetDataType();
    DataType weightDtype = (*params.weight)[0]->GetDataType();
    return ACLNN_SUCCESS;
}

static aclnnStatus GetGMMResultByL0Api(gmm::GroupedMatmulParams &params, uint64_t *workspaceSize, aclOpExecutor **executor) {
  auto uniqueExecutor = CREATE_EXECUTOR();  // fixed writen style, create OpExecutor
  aclOpExecutor *executorPtr = uniqueExecutor.get();
  CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  if (params.xDtype != DataType::DT_INT4) { // A4W4 has no bias
    CHECK_COND(gmm::BIAS_DTYPE.find(params.xDtype) != gmm::BIAS_DTYPE.cend(), ACLNN_ERR_PARAM_INVALID,
    "GMM: Cannot find bias dtype match with xDtype[%s]", op::ToString(params.xDtype).GetString());
  }
  SetParamsTensorEmpty(params, executorPtr);  // create empty tensorLists
  SetTransposedTensorListContiguous(params, executorPtr);
  CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "ParamsDataContiguous failed.");
  if (CheckZeroShape(params, workspaceSize) != ACLNN_SUCCESS) {
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;}
  op::Shape wqbmmNzShape = (*params.weight)[0]->GetStorageShape();
  if (params.apiVersion == gmm::GMMApiVersion::WeightNz) {
      CHECK_COND(ParamsWeightNzDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "ParamsWeightNzDtype failed.");
      std::vector<const aclTensor *> tensorsVec;
      for (size_t i = 0; i < params.weight->Size(); ++i) {
          const aclTensor *tensor = (*params.weight)[i];
          op::Shape weightNzShape = tensor->GetViewShape();
          tensor = SetTensorToNZFormat(tensor, weightNzShape, executorPtr);
          tensorsVec.push_back(tensor);
      }
      params.weight = executorPtr->AllocTensorList(tensorsVec.data(), tensorsVec.size());
      SetStorageShape(params,wqbmmNzShape);
  }
  CHECK_COND(TransWeightToNz(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "TransWeightToNz failed.");
  if (params.groupListOptional != nullptr) {
    params.groupTensorOptional = uniqueExecutor->ConvertToTensor(params.groupListOptional, op::ToOpDataType(ACL_INT64));
  }
  auto perTokenScaleOptional = (*params.perTokenScaleOptional)[0]->IsEmpty() ? nullptr : (*params.perTokenScaleOptional)[0];
  // Invoke l0 operator GroupedMatmul for calculation.
  auto result = l0op::GroupedMatmul(params.x, params.weight, params.biasOptional, params.scaleOptional,
                  params.offsetOptional, params.antiquantScaleOptional, params.antiquantOffsetOptional,
                  params.groupTensorOptional, perTokenScaleOptional, params.splitItem,
                  (*params.y)[0]->GetDataType(), params.transposeWeight, params.transposeX, params.groupType,
                  params.groupListType, params.activeType, params.tuningConfigOptional, params.y->Size(), executorPtr);
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_COND(CheckOutputShape(result, params.y) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Check output shape failed.");
  // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
  for (size_t i(0); i < params.y->Size(); ++i) {
    auto viewCopyResult = l0op::ViewCopy((*result)[i], (*params.y)[i], executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  // Standard syntax, get the size of workspace needed during computation.
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

static int64_t CorrectSplitItem(const aclTensorList *x, const aclTensorList *y, int64_t splitItem) {
  int64_t splitItemCorrected = splitItem;
   // Adjust split item based on the range of split item and group type.
  if (splitItem == X_Y_SEPARATED || splitItem == Y_SEPARATED) {
    // If X and Y have the same size, the input X and Y must be grouped.
    splitItemCorrected = x->Size() == y->Size() ? X_Y_SEPARATED : Y_SEPARATED;
  }
  if (splitItem == X_SEPARATED || splitItem == NO_SEPARATED) {
    splitItemCorrected = x->Size() == 1 ? NO_SEPARATED : X_SEPARATED;
  }
  return splitItemCorrected;
}

static aclnnStatus CheckTransposeStatus(const aclTensorList *x, const aclTensorList *weight, bool &transposeX,
                                        bool &transposeWeight, int64_t groupType) {
  CHECK_COND((*x)[0] != nullptr, ACLNN_ERR_PARAM_INVALID, "x[0] is nullptr!");
  CHECK_COND((*weight)[0] != nullptr, ACLNN_ERR_PARAM_INVALID, "weight[0] is nullptr!");
  if (groupType == gmm::SPLIT_K) {
    transposeX = gmm::IsTransposeLastTwoDims((*x)[0]);  // check is transpose x
    transposeWeight = CheckSpecialTranspose(weight);
  } else if (groupType == gmm::SPLIT_M || groupType == gmm::NO_SPLIT) {
    transposeX = CheckSpecialTranspose(x);
    transposeWeight = gmm::IsTransposeLastTwoDims((*weight)[0]); // check is transpose w
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus aclnnGroupedMatmulGetWorkspaceSizeCommon(const aclTensorList *x, const aclTensorList *weight,
  const aclTensorList *biasOptional, const aclTensorList *scaleOptional, const aclTensorList *offsetOptional,
  const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional,
  const aclTensorList *perTokenScaleOptional, const aclIntArray *groupListOptional,
  const aclTensor *groupTensorOptional, const aclTensorList *activationInputOptional,
  const aclTensorList *activationQuantScaleOptional, const aclTensorList *activationQuantOffsetOptional,
  int64_t splitItem, int64_t groupType, int64_t groupListType, int64_t actType, aclIntArray *tuningConfigOptional, gmm::GMMApiVersion apiVersion,
  const aclTensorList *y, const aclTensorList *activationFeatureOutOptional,
  const aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor) {
  DataType xDtype = DataType::DT_UNDEFINED;
  for (size_t i = 0; i < x->Size(); ++i) {
    if ((*x)[i] != nullptr) {
      xDtype = (*x)[i]->GetDataType();
      break;
    }
  }
  bool isSingleWeight = (weight->Size() == 1 && groupType != gmm::NO_SPLIT);
  bool transposeX = false;
  bool transposeWeight = false;
  CHECK_COND(CheckTransposeStatus(x, weight, transposeX, transposeWeight, groupType) == ACLNN_SUCCESS,
             ACLNN_ERR_PARAM_INVALID, "CheckTransposeStatus failed!");
  gmm::GroupedMatmulParams gmmParams{x, weight, biasOptional, groupListOptional, groupTensorOptional, scaleOptional,
                                offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional,
                                activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                                splitItem, groupListType, actType, transposeWeight, transposeX, isSingleWeight,
                                apiVersion, groupType, tuningConfigOptional, y, activationFeatureOutOptional, dynQuantScaleOutOptional,
                                xDtype};
  if (gmmParams.scaleOptional != nullptr) {
      for (size_t i = 0; i < gmmParams.scaleOptional->Size(); i++) {
          if ((*gmmParams.scaleOptional)[i]->GetDataType() == DataType::DT_INT64) {
              (void)const_cast<aclTensor *>((*gmmParams.scaleOptional)[i])->SetDataType(op::DataType::DT_UINT64);
          }
      }
  }
  ResetEmptyTensor(gmmParams);  // make empty tensor/tensorList nullptr
  CHECK_RET(CheckParam(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  gmmParams.splitItem = CorrectSplitItem(x, y, splitItem);

  aclnnStatus ret = GetGMMResultByL0Api(gmmParams, workspaceSize, executor);

  return ret;
}
}

aclnnStatus CheckCommonParam(const aclTensorList *x , const aclTensorList *weight,
  const aclTensor *groupListOptional, int64_t splitItem, int64_t groupType, int64_t groupListType,
  int64_t actType, const aclTensorList *out) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  bool is310P = socVersion == SocVersion::ASCEND310P;
  bool supportedCaseOn310P = x->Size() == 1 && out->Size() == 1 && weight->Size() == 1 && groupType == 0;
  CHECK_COND((is310P && supportedCaseOn310P) || !is310P, ACLNN_ERR_PARAM_INVALID,
             "only surpport x, y, weight not separated case with groupType is 0 on ASCEND310P.");
  CHECK_COND(PreCheckGroupType(splitItem, groupType) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "PreCheckGroupType failed, groupType is invalid.");
  // sparse group list shape [e, 2]
  size_t validGroupTensorDimNum = (groupListType == gmm::GROUP_LIST_SPARSE_M) ? 2UL: 1UL;
  CHECK_COND(groupListOptional == nullptr || groupListOptional->GetViewShape().GetDimNum() == validGroupTensorDimNum,
             ACLNN_ERR_PARAM_INVALID, "When groupList type is tensor, groupList dim only support 1 or 2"
             "(only when groupListType is 2), but now groupList dim is %zu, groupListType is %ld.",
             groupListOptional->GetViewShape().GetDimNum(), groupListType);
  CHECK_COND(actType >= 0, ACLNN_ERR_PARAM_INVALID, "actType must be larger or equal to 0");
  if (actType != GMMActType::GMM_ACT_TYPE_NONE) {
    CHECK_COND(actType != GMMActType::GMM_ACT_TYPE_GELU_ERR_FUNC, ACLNN_ERR_PARAM_INVALID,
               "Activation function not support GELU_ERR_FUNC now.");
    CHECK_COND(actType < END_ACT_TYPE_ENUM, ACLNN_ERR_PARAM_INVALID,
               "Activation function only support RELU/GELU_TANH/FASTGELU/SILU.");
  }

  if (groupListType == gmm::GROUP_LIST_SPARSE_M) {
    CHECK_COND(socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93, ACLNN_ERR_PARAM_INVALID,
      "This platform not support groupListType is 2.");
    CHECK_COND(groupType == gmm::SPLIT_M, ACLNN_ERR_PARAM_INVALID,
      "When groupListType is 2 only support groupType 0, but get groupType %ld.", groupType);
  } else {
    CHECK_COND(groupListType == 0 || groupListType == 1, ACLNN_ERR_PARAM_INVALID, "groupListType shoule be 0 or 1.");
  }
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulWeightNzGetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight,
  const aclTensorList *biasOptional, const aclTensorList *scaleOptional, const aclTensorList *offsetOptional,
  const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional,
  const aclTensorList *perTokenScaleOptional, const aclTensor *groupListOptional,
  const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional,
  const aclTensorList *activationQuantOffsetOptional, int64_t splitItem, int64_t groupType, int64_t groupListType,
  int64_t actType, aclIntArray *tuningConfigOptional, int64_t quantGroupSize, aclTensorList *out, aclTensorList *activationFeatureOutOptional,
  aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor) {
  CHECK_COND(CheckNotNull(x, weight, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
             "one of required inputs is nullptr.");
  // Standard syntax, Check parameters.
  L2_DFX_PHASE_1(aclnnGroupedMatmulWeightNz,
                 DFX_IN(x, weight, biasOptional, scaleOptional, offsetOptional,
                        antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, activationInputOptional,
                        activationQuantScaleOptional, activationQuantOffsetOptional,
                        groupListOptional, splitItem, groupType, groupListType, actType, tuningConfigOptional),
                 DFX_OUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
  if ((*weight)[0]->GetDataType() == DataType::DT_INT32) {
    // convert weight from int32 to int4
    UnpackInt32ToInt4(weight, "weight");
  }
  if ((*x)[0]->GetDataType() == DataType::DT_INT32) {
    // convert x from int32 to int4
    UnpackInt32ToInt4(x, "x");
  }
  // aclnnGroupedMatmulWeightNz dont support split K dim.
  CHECK_COND(groupType != gmm::SPLIT_K, ACLNN_ERR_PARAM_INVALID, "Not support split k dim now, groupType can not be 2.");
  CHECK_COND(CheckCommonParam(x, weight, groupListOptional, splitItem, groupType, groupListType, actType, out)
             == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "one of required inputs does not meet the requirement.");
  (void)quantGroupSize;
  return aclnnGroupedMatmulGetWorkspaceSizeCommon(x, weight, biasOptional, scaleOptional, offsetOptional,
                                                  antiquantScaleOptional, antiquantOffsetOptional,
                                                  perTokenScaleOptional, nullptr, groupListOptional,
                                                  activationInputOptional, activationQuantScaleOptional,
                                                  activationQuantOffsetOptional, splitItem, groupType, groupListType,
                                                  actType, tuningConfigOptional, gmm::GMMApiVersion::WeightNz, out, activationFeatureOutOptional,
                                                  dynQuantScaleOutOptional, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulV5GetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight,
  const aclTensorList *biasOptional, const aclTensorList *scaleOptional, const aclTensorList *offsetOptional,
  const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional,
  const aclTensorList *perTokenScaleOptional, const aclTensor *groupListOptional,
  const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional,
  const aclTensorList *activationQuantOffsetOptional, int64_t splitItem, int64_t groupType, int64_t groupListType,
  int64_t actType, aclIntArray *tuningConfigOptional, aclTensorList *out, aclTensorList *activationFeatureOutOptional,
  aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor) {
  CHECK_COND(CheckNotNull(x, weight, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
             "one of required inputs is nullptr.");
  // Standard syntax, Check parameters.
  L2_DFX_PHASE_1(aclnnGroupedMatmulV5,
                 DFX_IN(x, weight, biasOptional, scaleOptional, offsetOptional,
                        antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, activationInputOptional,
                        activationQuantScaleOptional, activationQuantOffsetOptional,
                        groupListOptional, splitItem, groupType, groupListType, actType, tuningConfigOptional),
                 DFX_OUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
  CHECK_COND(weight->Size() != 0, ACLNN_ERR_PARAM_INVALID, "weight should not be null tensorlist ");
  if ((*weight)[0]->GetDataType() == DataType::DT_INT32) {
    // convert weight from int32 to int4
    UnpackInt32ToInt4(weight, "weight");
  }
  if ((*x)[0]->GetDataType() == DataType::DT_INT32) {
    // convert x from int32 to int4
    UnpackInt32ToInt4(x, "x");
  }
  CHECK_COND(CheckCommonParam(x, weight, groupListOptional, splitItem, groupType, groupListType, actType, out)
             == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "one of required inputs does not meet the requirement.");
  return aclnnGroupedMatmulGetWorkspaceSizeCommon(x, weight, biasOptional, scaleOptional, offsetOptional,
                                                  antiquantScaleOptional, antiquantOffsetOptional,
                                                  perTokenScaleOptional, nullptr, groupListOptional,
                                                  activationInputOptional, activationQuantScaleOptional,
                                                  activationQuantOffsetOptional, splitItem, groupType, groupListType,
                                                  actType, tuningConfigOptional, gmm::GMMApiVersion::V5, out, activationFeatureOutOptional,
                                                  dynQuantScaleOutOptional, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulV4GetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight,
  const aclTensorList *biasOptional, const aclTensorList *scaleOptional, const aclTensorList *offsetOptional,
  const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional,
  const aclTensorList *perTokenScaleOptional, const aclTensor *groupListOptional,
  const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional,
  const aclTensorList *activationQuantOffsetOptional, int64_t splitItem, int64_t groupType, int64_t groupListType,
  int64_t actType, aclTensorList *out, aclTensorList *activationFeatureOutOptional,
  aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor) {
  CHECK_COND(CheckNotNull(x, weight, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
             "one of required inputs is nullptr.");
  // Standard syntax, Check parameters.
  L2_DFX_PHASE_1(aclnnGroupedMatmulV4,
                 DFX_IN(x, weight, biasOptional, scaleOptional, offsetOptional,
                        antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, activationInputOptional,
                        activationQuantScaleOptional, activationQuantOffsetOptional,
                        groupListOptional, splitItem, groupType, groupListType, actType),
                 DFX_OUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
  if ((*weight)[0]->GetDataType() == DataType::DT_INT32) {
    // convert weight from int32 to int4
    UnpackInt32ToInt4(weight, "weight");
  }
  if ((*x)[0]->GetDataType() == DataType::DT_INT32) {
    // convert x from int32 to int4
    UnpackInt32ToInt4(x, "x");
  }
  CHECK_COND(CheckCommonParam(x, weight, groupListOptional, splitItem, groupType, groupListType, actType, out)
             == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "one of required inputs does not meet the requirement.");
  return aclnnGroupedMatmulGetWorkspaceSizeCommon(x, weight, biasOptional, scaleOptional, offsetOptional,
                                                  antiquantScaleOptional, antiquantOffsetOptional,
                                                  perTokenScaleOptional, nullptr, groupListOptional,
                                                  activationInputOptional, activationQuantScaleOptional,
                                                  activationQuantOffsetOptional, splitItem, groupType, groupListType,
                                                  actType, nullptr, gmm::GMMApiVersion::V4, out,
                                                  activationFeatureOutOptional, dynQuantScaleOutOptional, workspaceSize,
                                                  executor);
}

aclnnStatus aclnnGroupedMatmulV3GetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight,
  const aclTensorList *biasOptional, const aclTensorList *scaleOptional, const aclTensorList *offsetOptional,
  const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional,
  const aclTensor *groupListOptional, int64_t splitItem, int64_t groupType, const aclTensorList *y,
  uint64_t *workspaceSize, aclOpExecutor **executor) {
  CHECK_COND(CheckNotNull(x, weight, y) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
             "one of required inputs is nullptr.");
  // Standard syntax, Check parameters.
  L2_DFX_PHASE_1(aclnnGroupedMatmulV3,
                 DFX_IN(x, weight, biasOptional, scaleOptional, offsetOptional,
                   antiquantScaleOptional, antiquantOffsetOptional, groupListOptional,
                   splitItem, groupType),
                 DFX_OUT(y));
  bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
  bool supportedCaseOn310P = x->Size() == 1 && y->Size() == 1 && weight->Size() == 1 && groupType == 0;
  CHECK_COND((is310P && supportedCaseOn310P) || !is310P, ACLNN_ERR_PARAM_INVALID,
             "only surpport x, y, weight not separated case with groupType is 0 on ASCEND310P.");
  CHECK_COND(PreCheckGroupType(splitItem, groupType) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "PreCheckGroupType failed, groupType is invalid.");
  CHECK_COND(
    groupListOptional == nullptr || groupListOptional->GetViewShape().GetDimNum() == 1, ACLNN_ERR_PARAM_INVALID, "When groupList type is tensor, groupList dim only support 1, but now is %lu.",
             groupListOptional->GetViewShape().GetDimNum());
  return aclnnGroupedMatmulGetWorkspaceSizeCommon(x, weight, biasOptional, scaleOptional, offsetOptional,
                                                  antiquantScaleOptional, antiquantOffsetOptional, nullptr, nullptr,
                                                  groupListOptional, nullptr, nullptr, nullptr, splitItem, groupType,
                                                  0, 0, nullptr, gmm::GMMApiVersion::V3, y, nullptr, nullptr,
                                                  workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulV2GetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight,
  const aclTensorList *biasOptional, const aclTensorList *scaleOptional, const aclTensorList *offsetOptional,
  const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional,
  const aclIntArray *groupListOptional, int64_t splitItem, int64_t groupType, const aclTensorList *y,
  uint64_t *workspaceSize, aclOpExecutor **executor) {
  CHECK_COND(CheckNotNull(x, weight, y) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
             "one of required inputs is nullptr.");
  bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
  CHECK_COND(!is310P, ACLNN_ERR_PARAM_INVALID,
             "Only aclnnGroupedMatmulV3GetWorkspaceSize is supported on ASCEND310P.");
  // Standard syntax, Check parameters.
  L2_DFX_PHASE_1(aclnnGroupedMatmulV2,
                 DFX_IN(x, weight, biasOptional, scaleOptional, offsetOptional,
                   antiquantScaleOptional, antiquantOffsetOptional, groupListOptional,
                   splitItem, groupType),
                 DFX_OUT(y));
  CHECK_COND(PreCheckGroupType(splitItem, groupType) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "PreCheckGroupType failed, groupType is invalid.");
  return aclnnGroupedMatmulGetWorkspaceSizeCommon(x, weight, biasOptional, scaleOptional, offsetOptional,
                                                  antiquantScaleOptional, antiquantOffsetOptional, nullptr,
                                                  groupListOptional, nullptr, nullptr, nullptr, nullptr, splitItem,
                                                  groupType, 0, 0, nullptr, gmm::GMMApiVersion::V2, y, nullptr, nullptr,
                                                  workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulGetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight,
  const aclTensorList *biasOptional, const aclTensorList *scaleOptional, const aclTensorList *offsetOptional,
  const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional,
  const aclIntArray *groupListOptional, int64_t splitItem, const aclTensorList *y, uint64_t *workspaceSize,
  aclOpExecutor **executor) {
  CHECK_COND(CheckNotNull(x, weight, y) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
             "one of required inputs is nullptr.");
  bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
  CHECK_COND(!is310P, ACLNN_ERR_PARAM_INVALID,
             "Only aclnnGroupedMatmulV3GetWorkspaceSize is supported on ASCEND310P.");
  // Standard syntax, Check parameters.
  L2_DFX_PHASE_1(aclnnGroupedMatmul,
                 DFX_IN(x, weight, biasOptional, scaleOptional, offsetOptional,
                   antiquantScaleOptional, antiquantOffsetOptional, groupListOptional,
                   splitItem),
                 DFX_OUT(y));
  int64_t groupType = 0;
  // Support weight group size of 1 only when the overall group size is 1.
  if (weight->Size() == 1) {
    CHECK_COND(x->Size() == 1 && y->Size() == 1, ACLNN_ERR_PARAM_INVALID,
               "Only accept separated weight, but input weight is not separated.");
  }
  bool xYSeparated = (x->Size() > 1 && y->Size() > 1) ||
                     (x->Size() == 1 && y->Size() == 1 && weight->Size() == 1);
  // Group type is -1 only when both input X and Y are grouped case.
  if (xYSeparated) {
    groupType = -1L;
  }
  if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P) {
    bool isSingleWeight = (weight->Size() == 1) && !(x->Size() == 1 && xYSeparated);
    CHECK_COND(!isSingleWeight, ACLNN_ERR_PARAM_INVALID,
               "Only accept separated weight, but input weight is not separated.");
  }
  return aclnnGroupedMatmulGetWorkspaceSizeCommon(x, weight, biasOptional, scaleOptional, offsetOptional,
                                                  antiquantScaleOptional, antiquantOffsetOptional, nullptr,
                                                  groupListOptional, nullptr, nullptr, nullptr, nullptr, splitItem,
                                                  groupType, 0, 0, nullptr, gmm::GMMApiVersion::V1, y, nullptr, nullptr,
                                                  workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                               aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGroupedMatmul);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
             "This is an error in GMM launch aicore");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGroupedMatmulV2);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
             "This is an error in GMM launch aicore");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGroupedMatmulV3);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
             "This is an error in GMM launch aicore");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGroupedMatmulV4);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
             "This is an error in GMM launch aicore");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulV5(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
  aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGroupedMatmulV5);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
              "This is an error in GMM launch aicore");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
  aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGroupedMatmulWeightNz);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
              "This is an error in GMM launch aicore");
  return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
