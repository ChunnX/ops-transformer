/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_ERROR_CHECK_H__
#define MC2_ERROR_CHECK_H__

#include <string>
#include "op_common/log/log.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"

const int32_t MC2_NCHW_N_DIM = 0;
const int32_t MC2_NCHW_C_DIM = 1;
const int32_t MC2_NHWC_N_DIM = 0;
const int32_t MC2_NHWC_C_DIM = 3;

static inline bool Mc2IsNullptr(const char *opName, const aclTensor *tensor, const char *name) {
  if (tensor == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool Mc2IsNullptr(const char *opName, const aclTensorList *tensorList, const char *name) {
  if (tensorList == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool Mc2IsNullptr(const char *opName, const aclScalar *scalar, const char *name) {
  if (scalar == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool Mc2IsNullptr(const char *opName, const aclIntArray *intArr, const char *name) {
  if (intArr == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool Mc2IsNullptr(const char *opName, const aclBoolArray *boolArr, const char *name) {
  if (boolArr == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool Mc2IsNullptr(const char *opName, const aclFloatArray *floatArr, const char *name) {
  if (floatArr == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool Mc2CheckDims(const char *opName, const aclTensor *tensor) {
  const auto& xShape = tensor->GetViewShape();
  for(size_t i = 0; i < xShape.GetDimNum(); i++) {
    if (xShape.GetDim(i) > INT32_MAX) {
      std::string shapeStr = op::ToString(xShape).GetString();
      std::string reasonMsg = "All dimensions of the shape of this parameter must be within the range [0, " +
          std::to_string(INT32_MAX) + "].";
      OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "tensor",
          shapeStr.c_str(), reasonMsg.c_str());
      return false;
    }
  }
  return true;
}

static inline bool Mc2CheckReduceOutShape(const char *opName, const aclTensor *inferOut, const aclTensor *out)
{
  auto const &xShape = inferOut->GetViewShape();
  auto const &yShape = out->GetViewShape();
  if (xShape != yShape) {
    if (!(xShape.GetShapeSize() == 1 && yShape.GetShapeSize() == 1)) {
      std::string shapeMsg = op::ToString(yShape).GetString() + std::string(" and ") +
          op::ToString(xShape).GetString();
      OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, "out and inferOut",
          shapeMsg.c_str(),
          "The shapes of out and inferOut must be the same.");
      return false;
    }
  }
  return true;
}

static inline bool Mc2CheckNCDimValid(const char *opName, const aclTensor *self, const aclTensor *out) {
  auto format = self->GetStorageFormat();
  int64_t selfDimN = 0;
  int64_t selfDimC = 0;
  int64_t outDimN = 0;
  int64_t outDimC = 0;
  if (format == op::Format::FORMAT_NCHW) {
    selfDimN = self->GetViewShape().GetDim(MC2_NCHW_N_DIM);
    selfDimC = self->GetViewShape().GetDim(MC2_NCHW_C_DIM);
    outDimN = out->GetViewShape().GetDim(MC2_NCHW_N_DIM);
    outDimC = out->GetViewShape().GetDim(MC2_NCHW_C_DIM);
  } else if (format == op::Format::FORMAT_NHWC) {
    selfDimN = self->GetViewShape().GetDim(MC2_NHWC_N_DIM);
    selfDimC = self->GetViewShape().GetDim(MC2_NHWC_C_DIM);
    outDimN = out->GetViewShape().GetDim(MC2_NHWC_N_DIM);
    outDimC = out->GetViewShape().GetDim(MC2_NHWC_C_DIM);
  } else {
    std::string formatStr = Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(self->GetStorageFormat()))).c_str();
    OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName, "self and out",
        formatStr.c_str(),
        "The formats of self and out must be within the range [NCHW, NHWC].");
    return false;
  }
  if ((selfDimN != outDimN) || (selfDimC != outDimC)) {
    std::string dimMsg = std::to_string(selfDimN) + "/" + std::to_string(outDimN) +
        " and " + std::to_string(selfDimC) + "/" + std::to_string(outDimC);
    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, "self and out",
        dimMsg.c_str(),
        "The shape dims of self and out must be the same.");
    return false;
  }
  return true;
}


#define OP_MC2_CHECK_NULL(opName, param, retExpr) \
  if (Mc2IsNullptr(opName, param, #param)) { \
    retExpr; \
  }

#define OP_MC2_CHECK_DTYPE_NOT_SUPPORT(opName, tensor, supportList, retExpr) \
  if (!CheckType(tensor->GetDataType(), supportList)) { \
    std::string _mc2_dtype_str_ = op::ToString(tensor->GetDataType()).GetString(); \
    std::string _mc2_support_str_ = op::ToString(supportList).GetString(); \
    OP_LOGE_FOR_INVALID_DTYPE(opName, #tensor, _mc2_dtype_str_.c_str(), _mc2_support_str_.c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_DTYPE_NOT_MATCH(opName, tensor, expectedDtype, retExpr) \
  if (tensor->GetDataType() != expectedDtype) { \
    std::string _mc2_actual_dtype_ = op::ToString(tensor->GetDataType()).GetString(); \
    std::string _mc2_expected_dtype_ = op::ToString(expectedDtype).GetString(); \
    OP_LOGE_FOR_INVALID_DTYPE(opName, #tensor, _mc2_actual_dtype_.c_str(), _mc2_expected_dtype_.c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_DTYPE_NOT_SAME(opName, tensor1, tensor2, retExpr) \
  if (tensor1->GetDataType() != tensor2->GetDataType()) { \
    std::string _mc2_param_msg_ = std::string(#tensor1) + " and " + std::string(#tensor2); \
    std::string _mc2_dtype_msg_ = op::ToString(tensor1->GetDataType()).GetString() + \
        std::string(" and ") + op::ToString(tensor2->GetDataType()).GetString(); \
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, _mc2_param_msg_.c_str(), \
        _mc2_dtype_msg_.c_str(), \
        "The dtypes of tensor1 and tensor2 must be the same."); \
    retExpr; \
  }

#define OP_MC2_CHECK_RESULT_DTYPE_CAST_FAILED(opName, dtype, desiredDtype, retExpr) \
  if (!CanCast(dtype, desiredDtype)) { \
    std::string _mc2_cast_dtype_ = op::ToString(dtype).GetString(); \
    std::string _mc2_desired_dtype_ = op::ToString(desiredDtype).GetString(); \
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "result", \
        _mc2_cast_dtype_.c_str(), \
        (std::string("The dtype of result must be castable to the desired output type ") + _mc2_desired_dtype_).c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_BROADCAST(opName, tensor1, tensor2, retExpr) \
  if (!CheckBroadcastShape(tensor1->GetViewShape(), tensor2->GetViewShape())) { \
    std::string _mc2_shape_msg_ = op::ToString(tensor1->GetViewShape()).GetString() + \
        std::string(" and ") + op::ToString(tensor2->GetViewShape()).GetString(); \
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, #tensor1 " and " #tensor2, \
        _mc2_shape_msg_.c_str(), \
        (std::string("The shape of ") + std::string(#tensor1) + \
         std::string(" must be broadcastable to the shape of ") + std::string(#tensor2)).c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_BROADCAST_WITH_SHAPE(opName, tensor, shape, retExpr) \
  if (!CheckBroadcastShape(tensor->GetViewShape(), shape)) { \
    std::string _mc2_tensor_shape_ = op::ToString(tensor->GetViewShape()).GetString(); \
    std::string _mc2_target_shape_ = op::ToString(shape).GetString(); \
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, #tensor, \
        _mc2_tensor_shape_.c_str(), \
        (std::string("The shape of ") + std::string(#tensor) + \
         std::string(" must be broadcastable to the size ") + _mc2_target_shape_).c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_BROADCAST_AND_INFER_SHAPE(opName, tensor1, tensor2, retShape, retExpr) \
  if (!BroadcastInferShape(tensor1->GetViewShape(), tensor2->GetViewShape(), retShape)) { \
    std::string _mc2_shape_msg_ = op::ToString(tensor1->GetViewShape()).GetString() + \
        std::string(" and ") + op::ToString(tensor2->GetViewShape()).GetString(); \
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, #tensor1 " and " #tensor2, \
        _mc2_shape_msg_.c_str(), \
        (std::string("The shape of ") + std::string(#tensor1) + \
         std::string(" must be broadcastable to the shape of ") + std::string(#tensor2)).c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_SHAPE_NOT_EQUAL(opName, tensor1, tensor2, retExpr) \
  if (tensor1->GetViewShape() != tensor2->GetViewShape()) { \
    std::string _mc2_shape_msg_ = op::ToString(tensor1->GetViewShape()).GetString() + \
        std::string(" and ") + op::ToString(tensor2->GetViewShape()).GetString(); \
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, #tensor1 " and " #tensor2, \
        _mc2_shape_msg_.c_str(), \
        (std::string("The shapes of ") + std::string(#tensor1) + \
         std::string(" and ") + std::string(#tensor2) + std::string(" must be the same.")).c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(opName, tensor, shape, retExpr) \
  if (tensor->GetViewShape() != shape) { \
    std::string _mc2_actual_shape_ = op::ToString(tensor->GetViewShape()).GetString(); \
    std::string _mc2_expected_shape_ = op::ToString(shape).GetString(); \
    OP_LOGE_FOR_INVALID_SHAPE(opName, #tensor, \
        _mc2_actual_shape_.c_str(), _mc2_expected_shape_.c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_WRONG_DIMENSION(opName, tensor, expectedDimNum, retExpr) \
  if (tensor->GetViewShape().GetDimNum() != expectedDimNum) { \
    std::string _mc2_dim_str_ = std::to_string(tensor->GetViewShape().GetDimNum()) + "D"; \
    std::string _mc2_reason_msg_ = "The shape of " + std::string(#tensor) + \
        std::string(" must be ") + \
        std::to_string(static_cast<size_t>(expectedDimNum)) + std::string("D."); \
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, #tensor, \
        _mc2_dim_str_.c_str(), _mc2_reason_msg_.c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_MAX_DIM(opName, tensor, maxDim, retExpr) \
  if (tensor->GetViewShape().GetDimNum() > static_cast<size_t>(maxDim)) { \
    std::string _mc2_dim_str_ = std::to_string(tensor->GetViewShape().GetDimNum()) + "D"; \
    std::string _mc2_reason_msg_ = "The shape dim of " + std::string(#tensor) + \
        std::string(" must be less than or equal to ") + \
        std::to_string(static_cast<size_t>(maxDim)) + std::string("."); \
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, #tensor, \
        _mc2_dim_str_.c_str(), _mc2_reason_msg_.c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_MIN_DIM(opName, tensor, minDim, retExpr) \
  if (tensor->GetViewShape().GetDimNum() < static_cast<size_t>(minDim)) { \
    std::string _mc2_dim_str_ = std::to_string(tensor->GetViewShape().GetDimNum()) + "D"; \
    std::string _mc2_reason_msg_ = "The shape dim of " + std::string(#tensor) + \
        std::string(" must be greater than or equal to ") + \
        std::to_string(static_cast<size_t>(minDim)) + std::string("."); \
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, #tensor, \
        _mc2_dim_str_.c_str(), _mc2_reason_msg_.c_str()); \
    retExpr; \
  }

#define OP_MC2_CHECK_COMM_INPUT(opName, workspaceSize, executor) \
  if (workspaceSize == nullptr || executor == nullptr) { \
    OP_LOGE(opName, "The workspaceSize or executor is nullptr."); \
    return ACLNN_ERR_PARAM_NULLPTR; \
  }

#define OP_MC2_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(opName, cond, retExpr, errMsg, ...) \
  if (cond) { \
    OP_LOGE(opName, errMsg, ##__VA_ARGS__); \
    retExpr; \
  }

#define OP_MC2_CHECK_INFERSHAPE(opName, cond, retExpr, errMsg, ...) \
  if (cond) { \
    OP_LOGE(opName, errMsg, ##__VA_ARGS__); \
    retExpr; \
  }

#define OP_MC2_CHECK_TENSORLIST_SIZE_EQUAL(opName, tensorlist1, tensorlist2, retExpr)                              \
    if ((tensorlist1)->Size() != (tensorlist2)->Size()) {                                                           \
        std::string _mc2_num_msg_ = std::to_string((tensorlist1)->Size()) +                                         \
            std::string(" and ") + std::to_string((tensorlist2)->Size());                                           \
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(opName, #tensorlist1 " and " #tensorlist2,                       \
            _mc2_num_msg_.c_str(),                                                                                  \
            "The tensor nums in tensorlist1 and tensorlist2 must be the same.");                                    \
        retExpr;                                                                                                    \
    }

#endif
