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
 * \file mc2_log.h
 * \brief
 */

#ifndef MC2_LOG_H
#define MC2_LOG_H

#include <string>
#include <vector>

// Clear log/log.h macros that may have been introduced before this header.
#undef OpLogSub
#undef OpLogErrSub
#undef D_OP_LOGI
#undef D_OP_LOGW
#undef D_OP_LOGE
#undef D_OP_LOGD
#undef OP_LOGI
#undef OP_LOGW
#undef OP_LOGE_WITHOUT_REPORT
#undef OP_LOGE
#undef OP_LOGD

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"

// opdev/common_types.h pulls in opdev/op_log.h, whose OP_LOGE family uses
// aclnn errno as the first argument. MC2 host/tiling code uses opName first.
#undef OpLogSub
#undef OpLogErrSub
#undef D_OP_LOGI
#undef D_OP_LOGW
#undef D_OP_LOGE
#undef D_OP_LOGD
#undef OP_LOGI
#undef OP_LOGW
#undef OP_LOGE_WITHOUT_REPORT
#undef OP_LOGE
#undef OP_LOGD

#include "mc2_common_log.h"

#include "../op_kernel/mc2_tiling_struct.h"
#include "tiling/tiling_api.h"
#include "quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#include "mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"

#undef OpLogSub
#undef OpLogErrSub
#undef D_OP_LOGI
#undef D_OP_LOGW
#undef D_OP_LOGE
#undef D_OP_LOGD
#undef OP_LOGI
#undef OP_LOGW
#undef OP_LOGE_WITHOUT_REPORT
#undef OP_LOGE
#undef OP_LOGD

#define OpLogSub(moduleId, level, opInfo, fmt, ...)                                                                   \
    do {                                                                                                              \
        if (CheckLogLevel(static_cast<int>(moduleId), level) == 1) {                                                  \
            DlogRecord(                                                                                               \
                static_cast<int>(moduleId), level,                                                                    \
                "[%s:%d][%s]"                                                                                         \
                "[%s][%" PRIu64 "] OpName:[%s] " fmt,                                                                 \
                __FILE__, __LINE__, OP_SUBMOD_NAME, __FUNCTION__, Ops::Base::GetTid(), Ops::Base::GetSafeStr(opInfo), \
                ##__VA_ARGS__);                                                                                       \
        }                                                                                                             \
    } while (0)

#define OpLogErrSub(moduleId, level, opInfo, fmt, ...)                                                                \
    do {                                                                                                              \
        if (CheckLogLevel(static_cast<int>(moduleId), level) == 1) {                                                  \
            DlogRecord(                                                                                               \
                static_cast<int>(moduleId), level,                                                                    \
                "[%s:%d][%s]"                                                                                         \
                "[%s][%" PRIu64 "] OpName:[%s] " fmt,                                                                 \
                __FILE__, __LINE__, OP_SUBMOD_NAME, __FUNCTION__, Ops::Base::GetTid(), Ops::Base::GetSafeStr(opInfo), \
                ##__VA_ARGS__);                                                                                       \
        }                                                                                                             \
    } while (0)

#define D_OP_LOGI(opName, fmt, ...) OpLogSub(OP, DLOG_INFO, opName, fmt, ##__VA_ARGS__)
#define D_OP_LOGW(opName, fmt, ...) OpLogSub(OP, DLOG_WARN, opName, fmt, ##__VA_ARGS__)
#define D_OP_LOGE(opName, fmt, ...) OpLogErrSub(OP, DLOG_ERROR, opName, fmt, ##__VA_ARGS__)
#define D_OP_LOGD(opName, fmt, ...) OpLogSub(OP, DLOG_DEBUG, opName, fmt, ##__VA_ARGS__)

#define OP_LOGI(opName, ...) D_OP_LOGI(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGW(opName, ...) D_OP_LOGW(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGE_WITHOUT_REPORT(opName, ...) D_OP_LOGE(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGE(opName, ...)                           \
    do {                                               \
        OP_LOGE_WITHOUT_REPORT(opName, ##__VA_ARGS__); \
        REPORT_INNER_ERR_MSG("EZ9999", ##__VA_ARGS__); \
    } while (0)
#define OP_LOGD(opName, ...) D_OP_LOGD(Ops::Base::GetOpInfo(opName), __VA_ARGS__)

using namespace op;

template <typename T>
std::string ConcatString(const T &arg) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(T arg, Ts... arg_left) {
  std::ostringstream oss;
  oss << arg;
  oss << ConcatString(arg_left...);
  return oss.str();
}

/**
 * @brief 工具函数：将合法列表转为字符串动态生成
 * @param list: 有效值列表
 */
template <typename T>
std::string VectorToString(const std::vector<T> &list)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < list.size(); i++) {
        if (i > 0) {
          oss << ",";
        }
        oss << list[i];
    }
    oss << "]";
    return oss.str();
}

namespace Mc2Log {
void PrintMMV3TilingData(const std::string &opName, Mc2Tiling::MC2MatmulV3TilingData &tiling);

void PrintRCSTilingData(const std::string &opName, Mc2Tiling::RCSTiling& rcsTiling);
void PrintTileL2TilingData(const std::string &opName, Mc2Tiling::TileL2Tiling& tileL2Tiling);
void PrintMc2MsgData(const std::string &opName, Mc2Tiling::Mc2Msg& msg);
void PrintTCubeTilingData(const std::string &opName, ::TCubeTiling &tiling);
void PrintTCubeTilingWindowParam(const std::string &opName, DequantBmm::Mc2SlidingWindowParams &tiling);
void PrintTCubeTilingL2cache(const std::string &opName, DequantBmm::Mc2L2cacheTileParams &tiling);
void PrintTCubeTilingParams(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3DataParams &tiling);
void PrintMMV3TilingData(const std::string &opName, Mc2MatMulV3TilingData &tiling);
}  // namespace Mc2Log

inline const char *ConvertStringToCstr(const std::string &str) { return str.c_str(); }

#if !defined(__ANDROID__) && !defined(ANDROID)
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)              \
  do {                                                            \
    D_OP_LOGE(op_name, err_msg, ##__VA_ARGS__);                   \
    REPORT_INNER_ERR_MSG("E69999", "op[%s], " err_msg,            \
                         ConvertStringToCstr(Ops::Base::GetOpInfo(op_name)), \
                         ##__VA_ARGS__);                          \
  } while (0)

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)                  \
  do {                                                                         \
    OP_LOGE_WITHOUT_REPORT(op_name, "%s", ConvertStringToCstr(err_msg));                  \
    std::string errorStr = "E89999";                                           \
    REPORT_INNER_ERR_MSG(errorStr.c_str(), "%s",                               \
                         ConcatString("op[", op_name, "],", err_msg).c_str()); \
  } while (0)

#define OPS_REPORT_VECTOR_INNER_ERR(opName, ...) OPS_LOG_E(opName, ##__VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(opName, ...) OPS_LOG_E(opName, ##__VA_ARGS__)
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILING(op_name, err_msg, ...)     \
  do {                                                            \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);      \
    REPORT_INNER_ERR_MSG("E89999", "op[%s], " err_msg,            \
                         ConvertStringToCstr(Ops::Base::GetOpInfo(op_name)), \
                         ##__VA_ARGS__);                          \
  } while (0)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

#else
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)
#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName, err_msg)
#define OPS_REPORT_VECTOR_INNER_ERR(opName, ...)
#define OPS_REPORT_CUBE_INNER_ERR(opName, ...)
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILING(opName, err_msg, ...)
#define OP_TILING_CHECK(cond, log_func, expr)
}  // namespace optiling
#endif

const int32_t NCHW_N_DIM = 0;
const int32_t NCHW_C_DIM = 1;
const int32_t NHWC_N_DIM = 0;
const int32_t NHWC_C_DIM = 3;

static inline bool IsNullptr(const char *opName, const aclTensor *tensor, const char *name) {
  if (tensor == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool IsNullptr(const char *opName, const aclTensorList *tensorList, const char *name) {
  if (tensorList == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool IsNullptr(const char *opName, const aclScalar *scalar, const char *name) {
  if (scalar == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool IsNullptr(const char *opName, const aclIntArray *intArr, const char *name) {
  if (intArr == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool IsNullptr(const char *opName, const aclBoolArray *boolArr, const char *name) {
  if (boolArr == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool IsNullptr(const char *opName, const aclFloatArray *floatArr, const char *name) {
  if (floatArr == nullptr) {
    OP_LOGE_WITH_INVALID_INPUT(opName, name);
    return true;
  }
  return false;
}

static inline bool CheckDims(const char *opName, const aclTensor *tensor) {
  const auto& xShape = tensor->GetViewShape();
  for(size_t i = 0; i < xShape.GetDimNum(); i++) {
    if (xShape.GetDim(i) > INT32_MAX) {
      OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "tensor",
          std::to_string(xShape.GetDim(i)).c_str(),
          ("The value of tensor must be less than or equal to " + std::to_string(INT32_MAX)).c_str());
      return false;
    }
  }
  return true;
}

static inline bool CheckReduceOutShape(const char *opName, const aclTensor *inferOut, const aclTensor *out)
{
  auto const &xShape = inferOut->GetViewShape();
  auto const &yShape = out->GetViewShape();
  if (xShape != yShape) {
    if (!(xShape.GetShapeSize() == 1 && yShape.GetShapeSize() == 1)) {
      OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "out",
          op::ToString(out->GetViewShape()).GetString(),
          (std::string("The shape of out must be the same as shape ") +
          op::ToString(inferOut->GetViewShape()).GetString() + " of inferOut.").c_str());
      return false;
    }
  }
  return true;
}

static inline bool CheckNCDimValid(const char *opName, const aclTensor *self, const aclTensor *out) {
  auto format = self->GetStorageFormat();
  int64_t selfDimN = 0;
  int64_t selfDimC = 0;
  int64_t outDimN = 0;
  int64_t outDimC = 0;
  if (format == op::Format::FORMAT_NCHW) {
    selfDimN = self->GetViewShape().GetDim(NCHW_N_DIM);
    selfDimC = self->GetViewShape().GetDim(NCHW_C_DIM);
    outDimN = out->GetViewShape().GetDim(NCHW_N_DIM);
    outDimC = out->GetViewShape().GetDim(NCHW_C_DIM);
  } else if (format == op::Format::FORMAT_NHWC) {
    selfDimN = self->GetViewShape().GetDim(NHWC_N_DIM);
    selfDimC = self->GetViewShape().GetDim(NHWC_C_DIM);
    outDimN = out->GetViewShape().GetDim(NHWC_N_DIM);
    outDimC = out->GetViewShape().GetDim(NHWC_C_DIM);
  } else {
    OP_LOGE_FOR_INVALID_FORMAT(opName, "self",
        Ops::Base::ToString(format).c_str(), "NCHW, NHWC");
    return false;
  }
  if ((selfDimN != outDimN) || (selfDimC != outDimC)) {
    std::string dimMsg = std::to_string(selfDimN) + " and " + std::to_string(outDimN) +
        " (N), " + std::to_string(selfDimC) + " and " + std::to_string(outDimC) + " (C)";
    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, "self and out",
        dimMsg.c_str(), "The shape dims of self and out must be the same.");
    return false;
  }
  return true;
}

#define OP_CHECK_NULL(opName, param, retExpr) \
  if (IsNullptr(opName, param, #param)) { \
    retExpr; \
  }

#define OP_CHECK_DTYPE_NOT_SUPPORT(opName, tensor, supportList, retExpr) \
  if (!CheckType(tensor->GetDataType(), supportList)) { \
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, #tensor, \
        op::ToString(tensor->GetDataType()).GetString(), \
        ("The dtype of " + std::string(#tensor) + " must be within the range " + \
        op::ToString(supportList).GetString()).c_str()); \
    retExpr; \
  }

#define OP_CHECK_DTYPE_NOT_MATCH(opName, tensor, expectedDtype, retExpr) \
  if (tensor->GetDataType() != expectedDtype) { \
    OP_LOGE_FOR_INVALID_DTYPE(opName, #tensor, \
        op::ToString(tensor->GetDataType()).GetString(), \
        op::ToString(expectedDtype).GetString()); \
    retExpr; \
  }

#define OP_CHECK_DTYPE_NOT_SAME(opName, tensor1, tensor2, retExpr) \
  if (tensor1->GetDataType() != tensor2->GetDataType()) { \
    std::string _dtypes_msg_ = std::string(#tensor1) + " " + \
        op::ToString(tensor1->GetDataType()).GetString() + " and " + \
        std::string(#tensor2) + " " + op::ToString(tensor2->GetDataType()).GetString(); \
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, #tensor1 " and " #tensor2, \
        _dtypes_msg_.c_str(), \
        "The dtypes of " #tensor1 " and " #tensor2 " must be the same."); \
    retExpr; \
  }

#define OP_CHECK_RESULT_DTYPE_CAST_FAILED(opName, dtype, desiredDtype, retExpr); \
  if (!CanCast(dtype, desiredDtype)) { \
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "result", \
        op::ToString(dtype).GetString(), \
        ("The dtype of result must be " + op::ToString(desiredDtype).GetString()).c_str()); \
    retExpr; \
  }

#define OP_CHECK_BROADCAST(opName, tensor1, tensor2, retExpr) \
  if (!CheckBroadcastShape(tensor1->GetViewShape(), tensor2->GetViewShape())) { \
    std::string _shapes_msg_ = op::ToString(tensor1->GetViewShape()).GetString() + " and " + \
        op::ToString(tensor2->GetViewShape()).GetString(); \
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, #tensor1 " and " #tensor2, \
        _shapes_msg_.c_str(), \
        ("The shapes of " + std::string(#tensor1) + " and " + \
        std::string(#tensor2) + " must be the same.").c_str()); \
    retExpr; \
  }

#define OP_CHECK_BROADCAST_WITH_SHAPE(opName, tensor, shape, retExpr) \
  if (!CheckBroadcastShape(tensor->GetViewShape(), shape)) { \
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, #tensor, \
        op::ToString(tensor->GetViewShape()).GetString(), \
        ("The shape of " + std::string(#tensor) + " must be " + \
        op::ToString(shape).GetString()).c_str()); \
    retExpr; \
  }

#define OP_CHECK_BROADCAST_AND_INFER_SHAPE(opName, tensor1, tensor2, retShape, retExpr) \
  if (!BroadcastInferShape(tensor1->GetViewShape(), tensor2->GetViewShape(), retShape)) { \
    std::string _shapes_msg_ = op::ToString(tensor1->GetViewShape()).GetString() + " and " + \
        op::ToString(tensor2->GetViewShape()).GetString(); \
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, #tensor1 " and " #tensor2, \
        _shapes_msg_.c_str(), \
        ("The shapes of " + std::string(#tensor1) + " and " + \
        std::string(#tensor2) + " must be the same.").c_str()); \
    retExpr; \
  }

#define OP_CHECK_SHAPE_NOT_EQUAL(opName, tensor1, tensor2, retExpr) \
  if (tensor1->GetViewShape() != tensor2->GetViewShape()) { \
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, #tensor1, \
        op::ToString(tensor1->GetViewShape()).GetString(), \
        ("The shape of " + std::string(#tensor1) + " must be the same as shape " + \
        op::ToString(tensor2->GetViewShape()).GetString() + " of " + \
        std::string(#tensor2)).c_str()); \
    retExpr; \
  }

#define OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(opName, tensor, shape, retExpr) \
  if (tensor->GetViewShape() != shape) { \
    OP_LOGE_FOR_INVALID_SHAPE(opName, #tensor, \
        op::ToString(tensor->GetViewShape()).GetString(), \
        op::ToString(shape).GetString()); \
    retExpr; \
  }

#define OP_CHECK_WRONG_DIMENSION(opName, tensor, expectedDimNum, retExpr) \
  if (tensor->GetViewShape().GetDimNum() != expectedDimNum) { \
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, #tensor, \
        (std::to_string(tensor->GetViewShape().GetDimNum()) + "D").c_str(), \
        ("The shape of " + std::string(#tensor) + " must be " + \
        std::to_string(expectedDimNum) + "D.").c_str()); \
    retExpr; \
  }

#define OP_CHECK_MAX_DIM(opName, tensor, maxDim, retExpr) \
  if (tensor->GetViewShape().GetDimNum() > static_cast<size_t>(maxDim)) { \
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, #tensor, \
        (std::to_string(tensor->GetViewShape().GetDimNum()) + "D").c_str(), \
        ("The shape dim of " + std::string(#tensor) + " must be less than or equal to " + \
        std::to_string(static_cast<size_t>(maxDim))).c_str()); \
    retExpr; \
  }

#define OP_CHECK_MIN_DIM(opName, tensor, minDim, retExpr) \
  if (tensor->GetViewShape().GetDimNum() < static_cast<size_t>(minDim)) { \
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, #tensor, \
        (std::to_string(tensor->GetViewShape().GetDimNum()) + "D").c_str(), \
        ("The shape dim of " + std::string(#tensor) + " must be greater than or equal to " + \
        std::to_string(static_cast<size_t>(minDim))).c_str()); \
    retExpr; \
  }

#define OP_CHECK_COMM_INPUT(opName, workspaceSize, executor) \
  if (workspaceSize == nullptr || executor == nullptr) { \
    OP_LOGE(opName, "The workspaceSize or executor is nullptr."); \
    return ACLNN_ERR_PARAM_NULLPTR; \
  }

#define OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(opName, cond, retExpr, errMsg, ...) \
  if (cond) { \
    OP_LOGE(opName, errMsg, ##__VA_ARGS__); \
    retExpr; \
  }

#define OP_CHECK_INFERSHAPE(opName, cond, retExpr, errMsg, ...) \
  if (cond) { \
    OP_LOGE(opName, errMsg, ##__VA_ARGS__); \
    retExpr; \
  }

#define OP_CHECK_TENSORLIST_SIZE_EQUAL(opName, tensorlist1, tensorlist2, retExpr)                                 \
    if ((tensorlist1)->Size() != (tensorlist2)->Size()) {                                                         \
        std::string _nums_msg_ = std::to_string((tensorlist1)->Size()) + " and " +                                \
            std::to_string((tensorlist2)->Size());                                                                \
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(opName, #tensorlist1 " and " #tensorlist2,                     \
            _nums_msg_.c_str(),                                                                                   \
            "The tensor nums in " #tensorlist1 " and " #tensorlist2 " must be the same.");                        \
        retExpr;                                                                                                  \
    }

#endif
