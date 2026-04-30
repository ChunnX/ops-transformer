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
 * \file log.h
 * \brief
 */
#ifndef OP_COMMON_LOG_LOG_H
#define OP_COMMON_LOG_LOG_H

#include <string>
#include <sstream>
#include <cinttypes>
#include <unistd.h>
#include <sys/syscall.h>
#include "exe_graph/runtime/shape.h"
#include "base/err_msg.h"
#include "dlog_pub.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "op_common/op_host/util/opbase_export.h"
namespace Ops {
namespace Base {

#ifndef OP_SUBMOD_NAME
#define OP_SUBMOD_NAME "OPS_BASE"
#endif

static inline uint64_t GetTid()
{
    const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
    return tid;
}

template <class T>
typename std::enable_if<std::is_same<const char*, typename std::decay<T>::type>::value, const char*>::type GetSafeStr(
    T name)
{
    return name;
}

template <class T>
typename std::enable_if<std::is_same<char*, typename std::decay<T>::type>::value, const char*>::type GetSafeStr(T name)
{
    return name;
}

template <class T>
typename std::enable_if<std::is_same<std::string, typename std::decay<T>::type>::value, const char*>::type GetSafeStr(
    const T& name)
{
    return name.c_str();
}

inline const std::string& GetOpInfo(const std::string& str)
{
    return str;
}

inline const char* GetOpInfo(const char* str)
{
    return (str == nullptr) ? "nil" : str;
}

template <class T>
constexpr bool IsContextType()
{
    return !std::is_same<const char*, typename std::decay<T>::type>::value &&
           !std::is_same<char*, typename std::decay<T>::type>::value &&
           !std::is_same<std::string, typename std::decay<T>::type>::value;
}

template <class T>
typename std::enable_if<IsContextType<T>(), std::string>::type GetOpInfo(T context)
{
    if (context == nullptr) {
        return "nil:nil";
    }
    std::string opInfo = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
    opInfo += ":";
    opInfo += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
    return opInfo;
}

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
#define unlikely(x) __builtin_expect((x), 0)
#define likely(x) __builtin_expect((x), 1)

#define OP_LOGI(opName, ...) D_OP_LOGI(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGW(opName, ...) D_OP_LOGW(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGE_WITHOUT_REPORT(opName, ...) D_OP_LOGE(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGE(opName, ...)                           \
    do {                                               \
        OP_LOGE_WITHOUT_REPORT(opName, ##__VA_ARGS__); \
        REPORT_INNER_ERR_MSG("EZ9999", ##__VA_ARGS__); \
    } while (0)

/**
 * opName:string
 * index:int
 * incorrectShape:string
 * correctShape:string
 * errMessage:The [index]th input of operator [opName] has incorrect shape [incorrectShape]. The correct one should be
 * [correctShape].
 */
#define OP_LOGE_WITH_INVALID_INPUT_SHAPE(opName, index, incorrectShape, correctShape)                              \
    do {                                                                                                           \
        std::string _safe_opName_(opName);                                                                         \
        std::string _safe_incorrectShape_(incorrectShape);                                                         \
        std::string _safe_correctShape_(correctShape);                                                             \
        std::string index_str = std::to_string(index);                                                             \
        OP_LOGE_WITHOUT_REPORT(                                                                                    \
            _safe_opName_.c_str(),                                                                                 \
            "The %sth input of operator %s has incorrect shape [%s]. The correct one should be [%s].",             \
            index_str.c_str(), _safe_opName_.c_str(), _safe_incorrectShape_.c_str(), _safe_correctShape_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "index", "incorrect_shape", "correct_shape"};          \
        const std::vector<const char*> msgvalue = {                                                                \
            _safe_opName_.c_str(), index_str.c_str(), _safe_incorrectShape_.c_str(), _safe_correctShape_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0001", msgKey, msgvalue);                                                     \
    } while (0)

/**
 * opName:string
 * attrName:string
 * incorrectVal:string
 * correctVal:string
 * errMessage:Attribute [attrName] of operator [opName] has incorrect value [incorrectVal]. The correct one should be
 * [correctVal].
 */
#define OP_LOGE_WITH_INVALID_ATTR(opName, attrName, incorrectVal, correctVal)                                        \
    do {                                                                                                             \
        std::string _safe_opName_(opName);                                                                           \
        std::string _safe_attrName_(attrName);                                                                       \
        std::string _safe_incorrectVal_(incorrectVal);                                                               \
        std::string _safe_correctVal_(correctVal);                                                                   \
        OP_LOGE_WITHOUT_REPORT(                                                                                      \
            _safe_opName_.c_str(),                                                                                   \
            "Attribute %s of operator %s has incorrect value %s. The correct one should be %s.",                     \
            _safe_attrName_.c_str(), _safe_opName_.c_str(), _safe_incorrectVal_.c_str(), _safe_correctVal_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "attr_name", "incorrect_val", "correct_val"};            \
        const std::vector<const char*> msgvalue = {                                                                  \
            _safe_opName_.c_str(), _safe_attrName_.c_str(), _safe_incorrectVal_.c_str(), _safe_correctVal_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0002", msgKey, msgvalue);                                                       \
    } while (0)

/**
 * opName:string
 * attrName:string
 * incorrectSize:string
 * correctSize:string
 * errMessage:Attribute [attrName] of operator [opName] has incorrect size [incorrectSize]. The correct one should be
 * [correctSize].
 */
#define OP_LOGE_WITH_INVALID_ATTR_SIZE(opName, attrName, incorrectSize, correctSize)                                   \
    do {                                                                                                               \
        std::string _safe_opName_(opName);                                                                             \
        std::string _safe_attrName_(attrName);                                                                         \
        std::string _safe_incorrectSize_(incorrectSize);                                                               \
        std::string _safe_correctSize_(correctSize);                                                                   \
        OP_LOGE_WITHOUT_REPORT(                                                                                        \
            _safe_opName_.c_str(), "Attribute %s of operator %s has incorrect size %s. The correct one should be %s.", \
            _safe_attrName_.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(), _safe_correctSize_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "attr_name", "incorrect_size", "correct_size"};            \
        const std::vector<const char*> msgvalue = {                                                                    \
            _safe_opName_.c_str(), _safe_attrName_.c_str(), _safe_incorrectSize_.c_str(), _safe_correctSize_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0003", msgKey, msgvalue);                                                         \
    } while (0)

/**
 * opName:string
 * paramName:string
 * errMessage:Parameter [paramName] of operator [opName] is required, but it is empty.
 */
#define OP_LOGE_WITH_INVALID_INPUT(opName, paramName)                                                \
    do {                                                                                             \
        std::string _safe_opName_(opName);                                                           \
        std::string _safe_paramName_(paramName);                                                     \
        OP_LOGE_WITHOUT_REPORT(                                                                      \
            _safe_opName_.c_str(), "Parameter %s of operator %s is required, but it is empty.", \
            _safe_paramName_.c_str(), _safe_opName_.c_str());                                        \
        const std::vector<const char*> msgKey = {"op_name", "param_name"};                           \
        const std::vector<const char*> msgvalue = {_safe_opName_.c_str(), _safe_paramName_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0004", msgKey, msgvalue);                                       \
    } while (0)

/**
 * opName:string
 * index:int
 * incorrectSize:string
 * correctSize:string
 * errMessage:The [index]th input of operator [opName] has incorrect shape size [incorrectSize]. The correct one should
 * be [correctSize].
 */
#define OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE(opName, index, incorrectSize, correctSize)                          \
    do {                                                                                                         \
        std::string _safe_opName_(opName);                                                                       \
        std::string _safe_incorrectSize_(incorrectSize);                                                         \
        std::string _safe_correctSize_(correctSize);                                                             \
        std::string index_str = std::to_string(index);                                                           \
        OP_LOGE_WITHOUT_REPORT(                                                                                  \
            _safe_opName_.c_str(),                                                                               \
            "The %sth input of operator %s has incorrect shape size %s. The correct one should be %s.",          \
            index_str.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(), _safe_correctSize_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "index", "incorrect_size", "correct_size"};          \
        const std::vector<const char*> msgvalue = {                                                              \
            _safe_opName_.c_str(), index_str.c_str(), _safe_incorrectSize_.c_str(), _safe_correctSize_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0005", msgKey, msgvalue);                                                   \
    } while (0)

/**
 * opName:string
 * paramName:string
 * dataFormat:string
 * expectedFormatList:string
 * errMessage:Input parameter [paramName] of operator [opName] has incorrect format [dataFormat]. The correct one should
 * be [expectedFormatList].
 */
#define OP_LOGE_WITH_INVALID_INPUT_FORMAT(opName, paramName, dataFormat, expectedFormatList)                      \
    do {                                                                                                          \
        std::string _safe_opName_(opName);                                                                        \
        std::string _safe_paramName_(paramName);                                                                  \
        std::string _safe_dataFormat_(dataFormat);                                                                \
        std::string _safe_expectedFormatList_(expectedFormatList);                                                \
        OP_LOGE_WITHOUT_REPORT(                                                                                   \
            _safe_opName_.c_str(),                                                                                \
            "Input parameter %s of operator %s has incorrect format %s. The correct one should be %s.",           \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_dataFormat_.c_str(),                           \
            _safe_expectedFormatList_.c_str());                                                                   \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "data_format", "expected_format_list"}; \
        const std::vector<const char*> msgvalue = {                                                               \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_dataFormat_.c_str(),                           \
            _safe_expectedFormatList_.c_str()};                                                                   \
        REPORT_PREDEFINED_ERR_MSG("EZ0006", msgKey, msgvalue);                                                    \
    } while (0)

/**
 * opName:string
 * paramName:string
 * dataDtype:string
 * expectedDtypeList:string
 * errMessage:Input parameter [paramName] of operator [opName] has incorrect dtype [dataDtype]. The correct one should
 * be [expectedDtypeList].
 */
#define OP_LOGE_WITH_INVALID_INPUT_DTYPE(opName, paramName, dataDtype, expectedDtypeList)                       \
    do {                                                                                                        \
        std::string _safe_opName_(opName);                                                                      \
        std::string _safe_paramName_(paramName);                                                                \
        std::string _safe_dataDtype_(dataDtype);                                                                \
        std::string _safe_expectedDtypeList_(expectedDtypeList);                                                \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            _safe_opName_.c_str(),                                                                              \
            "Input parameter %s of operator %s has incorrect dtype %s. The correct one should be %s.",          \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_dataDtype_.c_str(),                          \
            _safe_expectedDtypeList_.c_str());                                                                  \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "data_dtype", "expected_dtype_list"}; \
        const std::vector<const char*> msgvalue = {                                                             \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_dataDtype_.c_str(),                          \
            _safe_expectedDtypeList_.c_str()};                                                                  \
        REPORT_PREDEFINED_ERR_MSG("EZ0007", msgKey, msgvalue);                                                  \
    } while (0)

// ============================================================================
// General Parameter Validation Macros (EZ0008-EZ0024)
// These macros do not distinguish between input/output and use generic "parameter" naming
// ============================================================================

/**
 * EZ0008: Parameter shape error (single parameter, with correct shape)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectShape:string - Actual shape, format like "[1,128,128]"
 * correctShape:string - Expected shape, format like "[1,256,256]"
 * errMessage: Parameter [paramName] of operator [opName] has incorrect shape [incorrectShape]. The correct one should
 * be [correctShape].
 */
#define OP_LOGE_FOR_INVALID_SHAPE(opName, paramName, incorrectShape, correctShape)                             \
    do {                                                                                                       \
        std::string _safe_opName_(opName);                                                                     \
        std::string _safe_paramName_(paramName);                                                               \
        std::string _safe_incorrectShape_(incorrectShape);                                                     \
        std::string _safe_correctShape_(correctShape);                                                         \
        OP_LOGE_WITHOUT_REPORT(                                                                                \
            _safe_opName_.c_str(),                                                                             \
            "Parameter %s of operator %s has incorrect shape [%s]. The correct one should be [%s].",           \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectShape_.c_str(),                    \
            _safe_correctShape_.c_str());                                                                      \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_shape", "correct_shape"}; \
        const std::vector<const char*> msgvalue = {                                                            \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectShape_.c_str(),                    \
            _safe_correctShape_.c_str()};                                                                      \
        REPORT_PREDEFINED_ERR_MSG("EZ0008", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0009: Parameter shape error (single parameter, with reason)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectShape:string - Actual incorrect shape, format like "[1,128]"
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of operator [opName] has incorrect shape [incorrectShape]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, paramName, incorrectShape, reason)                            \
    do {                                                                                                            \
        std::string _safe_opName_(opName);                                                                          \
        std::string _safe_paramName_(paramName);                                                                    \
        std::string _safe_incorrectShape_(incorrectShape);                                                          \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            _safe_opName_.c_str(), "Parameter %s of operator %s has incorrect shape [%s]. Reason: %s.",             \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectShape_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_shape", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                 \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectShape_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0009", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * EZ0010: Parameters shape error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectShapes:string - Actual incorrect shapes
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of operator [opName] have incorrect shapes [incorrectShapes]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, paramNames, incorrectShapes, reason)                           \
    do {                                                                                                              \
        std::string _safe_opName_(opName);                                                                            \
        std::string _safe_paramNames_(paramNames);                                                                    \
        std::string _safe_incorrectShapes_(incorrectShapes);                                                          \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_WITHOUT_REPORT(                                                                                       \
            _safe_opName_.c_str(), "Parameters %s of operator %s have incorrect shapes %s. Reason: %s.",              \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectShapes_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_shapes", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                   \
            _safe_opName_.c_str(), _safe_paramNames_.c_str(), _safe_incorrectShapes_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0010", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * EZ0011: Parameter shape dim error (single parameter, with correct dim)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectDim:string - Actual dimension, format like "2D" or "3D"
 * correctDim:string - Expected dimension, format like "4D"
 * errMessage: Parameter [paramName] of operator [opName] has incorrect shape dim [incorrectDim]. The correct one should
 * be [correctDim].
 */
#define OP_LOGE_FOR_INVALID_SHAPEDIM(opName, paramName, incorrectDim, correctDim)                                     \
    do {                                                                                                              \
        std::string _safe_opName_(opName);                                                                            \
        std::string _safe_paramName_(paramName);                                                                      \
        std::string _safe_incorrectDim_(incorrectDim);                                                                \
        std::string _safe_correctDim_(correctDim);                                                                    \
        OP_LOGE_WITHOUT_REPORT(                                                                                       \
            _safe_opName_.c_str(),                                                                                    \
            "Parameter %s of operator %s has incorrect shape dim %s. The correct one should be %s.",                  \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDim_.c_str(), _safe_correctDim_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_dim", "correct_dim"};            \
        const std::vector<const char*> msgvalue = {                                                                   \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectDim_.c_str(), _safe_correctDim_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0011", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * EZ0012: Parameter shape dim error (single parameter, with reason)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectDim:string - Actual incorrect dimension
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of operator [opName] has incorrect shape dim [incorrectDim]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, paramName, incorrectDim, reason)                         \
    do {                                                                                                          \
        std::string _safe_opName_(opName);                                                                        \
        std::string _safe_paramName_(paramName);                                                                  \
        std::string _safe_incorrectDim_(incorrectDim);                                                            \
        std::string _safe_reason_(reason);                                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                   \
            _safe_opName_.c_str(), "Parameter %s of operator %s has incorrect shape dim %s. Reason: %s.",         \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDim_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_dim", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                               \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectDim_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0012", msgKey, msgvalue);                                                    \
    } while (0)

/**
 * EZ0013: Parameters shape dim error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectDims:string - Actual incorrect dimensions
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of operator [opName] have incorrect shape dims [incorrectDims]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, paramNames, incorrectDims, reason)                        \
    do {                                                                                                            \
        std::string _safe_opName_(opName);                                                                          \
        std::string _safe_paramNames_(paramNames);                                                                  \
        std::string _safe_incorrectDims_(incorrectDims);                                                            \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            _safe_opName_.c_str(), "Parameters %s of operator %s have incorrect shape dims %s. Reason: %s.",        \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectDims_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_dims", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                 \
            _safe_opName_.c_str(), _safe_paramNames_.c_str(), _safe_incorrectDims_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0013", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * EZ0014: Parameter shape size error (single parameter, with correct size)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectSize:string - Actual size (total element count)
 * correctSize:string - Expected size (total element count)
 * errMessage: Parameter [paramName] of operator [opName] has incorrect shape size [incorrectSize]. The correct one
 * should be [correctSize].
 */
#define OP_LOGE_FOR_INVALID_SHAPESIZE(opName, paramName, incorrectSize, correctSize)                         \
    do {                                                                                                     \
        std::string _safe_opName_(opName);                                                                   \
        std::string _safe_paramName_(paramName);                                                             \
        std::string _safe_incorrectSize_(incorrectSize);                                                     \
        std::string _safe_correctSize_(correctSize);                                                         \
        OP_LOGE_WITHOUT_REPORT(                                                                              \
            _safe_opName_.c_str(),                                                                           \
            "Parameter %s of operator %s has incorrect shape size %s. The correct one should be %s.",        \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(),                   \
            _safe_correctSize_.c_str());                                                                     \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_size", "correct_size"}; \
        const std::vector<const char*> msgvalue = {                                                          \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectSize_.c_str(),                   \
            _safe_correctSize_.c_str()};                                                                     \
        REPORT_PREDEFINED_ERR_MSG("EZ0014", msgKey, msgvalue);                                               \
    } while (0)

/**
 * EZ0015: Parameter shape size error (single parameter, with reason)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectSize:string - Actual incorrect shape size
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of operator [opName] has incorrect shape size [incorrectSize]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, paramName, incorrectSize, reason)                        \
    do {                                                                                                           \
        std::string _safe_opName_(opName);                                                                         \
        std::string _safe_paramName_(paramName);                                                                   \
        std::string _safe_incorrectSize_(incorrectSize);                                                           \
        std::string _safe_reason_(reason);                                                                         \
        OP_LOGE_WITHOUT_REPORT(                                                                                    \
            _safe_opName_.c_str(), "Parameter %s of operator %s has incorrect shape size %s. Reason: %s.",         \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_size", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectSize_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0015", msgKey, msgvalue);                                                     \
    } while (0)

/**
 * EZ0016: Parameters shape size error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectSizes:string - Actual incorrect sizes
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of operator [opName] have incorrect shape sizes [incorrectSizes]. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName, paramNames, incorrectSizes, reason)                       \
    do {                                                                                                             \
        std::string _safe_opName_(opName);                                                                           \
        std::string _safe_paramNames_(paramNames);                                                                   \
        std::string _safe_incorrectSizes_(incorrectSizes);                                                           \
        std::string _safe_reason_(reason);                                                                           \
        OP_LOGE_WITHOUT_REPORT(                                                                                      \
            _safe_opName_.c_str(), "Parameters %s of operator %s have incorrect shape sizes %s. Reason: %s.",        \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectSizes_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_sizes", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                  \
            _safe_opName_.c_str(), _safe_paramNames_.c_str(), _safe_incorrectSizes_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0016", msgKey, msgvalue);                                                       \
    } while (0)

/**
 * EZ0017: Parameter format error (single parameter, with correct format)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectFormat:string - Actual format (e.g., "NHWC")
 * correctFormat:string - Expected format (e.g., "NCHW")
 * errMessage: Parameter [paramName] of operator [opName] has incorrect format [incorrectFormat]. The correct one should
 * be [correctFormat].
 */
#define OP_LOGE_FOR_INVALID_FORMAT(opName, paramName, incorrectFormat, correctFormat)                            \
    do {                                                                                                         \
        std::string _safe_opName_(opName);                                                                       \
        std::string _safe_paramName_(paramName);                                                                 \
        std::string _safe_incorrectFormat_(incorrectFormat);                                                     \
        std::string _safe_correctFormat_(correctFormat);                                                         \
        OP_LOGE_WITHOUT_REPORT(                                                                                  \
            _safe_opName_.c_str(),                                                                               \
            "Parameter %s of operator %s has incorrect format %s. The correct one should be %s.",                \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectFormat_.c_str(),                     \
            _safe_correctFormat_.c_str());                                                                       \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_format", "correct_format"}; \
        const std::vector<const char*> msgvalue = {                                                              \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectFormat_.c_str(),                     \
            _safe_correctFormat_.c_str()};                                                                       \
        REPORT_PREDEFINED_ERR_MSG("EZ0017", msgKey, msgvalue);                                                   \
    } while (0)

/**
 * EZ0018: Parameters format error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectFormats:string - Actual incorrect formats
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of operator [opName] have incorrect formats [incorrectFormats]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName, paramNames, incorrectFormats, reason)                          \
    do {                                                                                                               \
        std::string _safe_opName_(opName);                                                                             \
        std::string _safe_paramNames_(paramNames);                                                                     \
        std::string _safe_incorrectFormats_(incorrectFormats);                                                         \
        std::string _safe_reason_(reason);                                                                             \
        OP_LOGE_WITHOUT_REPORT(                                                                                        \
            _safe_opName_.c_str(), "Parameters %s of operator %s have incorrect formats %s. Reason: %s.",              \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectFormats_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_formats", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                    \
            _safe_opName_.c_str(), _safe_paramNames_.c_str(), _safe_incorrectFormats_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0018", msgKey, msgvalue);                                                         \
    } while (0)

/**
 * EZ0019: Parameter dtype error (single parameter, with correct dtype)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectDtype:string - Actual data type (e.g., "INT32")
 * correctDtype:string - Expected data type (e.g., "FLOAT16")
 * errMessage: Parameter [paramName] of operator [opName] has incorrect dtype [incorrectDtype]. The correct one should
 * be [correctDtype].
 */
#define OP_LOGE_FOR_INVALID_DTYPE(opName, paramName, incorrectDtype, correctDtype)                             \
    do {                                                                                                       \
        std::string _safe_opName_(opName);                                                                     \
        std::string _safe_paramName_(paramName);                                                               \
        std::string _safe_incorrectDtype_(incorrectDtype);                                                     \
        std::string _safe_correctDtype_(correctDtype);                                                         \
        OP_LOGE_WITHOUT_REPORT(                                                                                \
            _safe_opName_.c_str(),                                                                             \
            "Parameter %s of operator %s has incorrect dtype %s. The correct one should be %s.",               \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDtype_.c_str(),                    \
            _safe_correctDtype_.c_str());                                                                      \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_dtype", "correct_dtype"}; \
        const std::vector<const char*> msgvalue = {                                                            \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectDtype_.c_str(),                    \
            _safe_correctDtype_.c_str()};                                                                      \
        REPORT_PREDEFINED_ERR_MSG("EZ0019", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0020: Parameter dtype error (single parameter, with reason)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectDtype:string - Actual incorrect data type
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of operator [opName] has incorrect dtype [incorrectDtype]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, paramName, incorrectDtype, reason)                            \
    do {                                                                                                            \
        std::string _safe_opName_(opName);                                                                          \
        std::string _safe_paramName_(paramName);                                                                    \
        std::string _safe_incorrectDtype_(incorrectDtype);                                                          \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            _safe_opName_.c_str(), "Parameter %s of operator %s has incorrect dtype %s. Reason: %s.",               \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectDtype_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_dtype", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                 \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectDtype_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0020", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * EZ0021: Parameters dtype error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectDtypes:string - Actual incorrect data types
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of operator [opName] have incorrect dtypes [incorrectDtypes]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, paramNames, incorrectDtypes, reason)                           \
    do {                                                                                                              \
        std::string _safe_opName_(opName);                                                                            \
        std::string _safe_paramNames_(paramNames);                                                                    \
        std::string _safe_incorrectDtypes_(incorrectDtypes);                                                          \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_WITHOUT_REPORT(                                                                                       \
            _safe_opName_.c_str(), "Parameters %s of operator %s have incorrect dtypes %s. Reason: %s.",              \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectDtypes_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_dtypes", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                   \
            _safe_opName_.c_str(), _safe_paramNames_.c_str(), _safe_incorrectDtypes_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0021", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * EZ0022: Parameter tensor num error (single parameter, with correct num)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectNum:int64_t - Actual tensor count
 * correctNum:string - Expected tensor count
 * errMessage: Parameter [paramName] of operator [opName] has invalid tensor num [incorrectNum]. The correct one should
 * be [correctNum].
 */
#define OP_LOGE_FOR_INVALID_TENSORNUM(opName, paramName, incorrectNum, correctNum)                                \
    do {                                                                                                          \
        std::string _safe_opName_(opName);                                                                        \
        std::string _safe_paramName_(paramName);                                                                  \
        std::string _safe_correctNum_(correctNum);                                                                \
        OP_LOGE_WITHOUT_REPORT(                                                                                   \
            _safe_opName_.c_str(),                                                                                \
            "Parameter %s of operator %s has invalid tensor num %ld. The correct one should be %s.",              \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), static_cast<int64_t>(incorrectNum),                  \
            _safe_correctNum_.c_str());                                                                           \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_num", "correct_num"};        \
        std::string incorrectNumStr = std::to_string(static_cast<int64_t>(incorrectNum));                         \
        const std::vector<const char*> msgvalue = {                                                               \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), incorrectNumStr.c_str(), _safe_correctNum_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0022", msgKey, msgvalue);                                                    \
    } while (0)

/**
 * EZ0023: Parameters tensor num error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectNums:string - Actual incorrect tensor counts
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of operator [opName] have invalid tensor nums [incorrectNums]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(opName, paramNames, incorrectNums, reason)                       \
    do {                                                                                                            \
        std::string _safe_opName_(opName);                                                                          \
        std::string _safe_paramNames_(paramNames);                                                                  \
        std::string _safe_incorrectNums_(incorrectNums);                                                            \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            _safe_opName_.c_str(), "Parameters %s of operator %s have invalid tensor nums %s. Reason: %s.",         \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectNums_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_nums", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                 \
            _safe_opName_.c_str(), _safe_paramNames_.c_str(), _safe_incorrectNums_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0023", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * EZ0024: Parameter value error (single parameter, with correct value)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectValue:string - Actual incorrect value
 * correctValue:string - Expected correct value
 * errMessage: Parameter [paramName] of operator [opName] has incorrect value [incorrectValue]. The correct one should
 * be [correctValue].
 */
#define OP_LOGE_FOR_INVALID_VALUE(opName, paramName, incorrectValue, correctValue)                             \
    do {                                                                                                       \
        std::string _safe_opName_(opName);                                                                     \
        std::string _safe_paramName_(paramName);                                                               \
        std::string _safe_incorrectValue_(incorrectValue);                                                     \
        std::string _safe_correctValue_(correctValue);                                                         \
        OP_LOGE_WITHOUT_REPORT(                                                                                \
            _safe_opName_.c_str(),                                                                             \
            "Parameter %s of operator %s has incorrect value %s. The correct one should be %s.",               \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectValue_.c_str(),                    \
            _safe_correctValue_.c_str());                                                                      \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_value", "correct_value"}; \
        const std::vector<const char*> msgvalue = {                                                            \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectValue_.c_str(),                    \
            _safe_correctValue_.c_str()};                                                                      \
        REPORT_PREDEFINED_ERR_MSG("EZ0024", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * EZ0025: Parameter list size error (single parameter, with correct size)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectSize:string - Actual incorrect list size
 * correctSize:string - Expected correct list size
 * errMessage: Parameter [paramName] of operator [opName] has invalid list size [incorrectSize]. The correct one should
 * be [correctSize].
 */
#define OP_LOGE_FOR_INVALID_LISTSIZE(opName, paramName, incorrectSize, correctSize)                          \
    do {                                                                                                     \
        std::string _safe_opName_(opName);                                                                   \
        std::string _safe_paramName_(paramName);                                                             \
        std::string _safe_incorrectSize_(incorrectSize);                                                     \
        std::string _safe_correctSize_(correctSize);                                                         \
        OP_LOGE_WITHOUT_REPORT(                                                                              \
            _safe_opName_.c_str(),                                                                           \
            "Parameter %s of operator %s has invalid list size %s. The correct one should be %s.",           \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectSize_.c_str(),                   \
            _safe_correctSize_.c_str());                                                                     \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_size", "correct_size"}; \
        const std::vector<const char*> msgvalue = {                                                          \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectSize_.c_str(),                   \
            _safe_correctSize_.c_str()};                                                                     \
        REPORT_PREDEFINED_ERR_MSG("EZ0025", msgKey, msgvalue);                                               \
    } while (0)

/**
 * EZ0026: Parameter value error (single parameter, with reason)
 * opName:string - Operator name
 * paramName:string - Parameter name
 * incorrectValue:string - Actual incorrect value
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of operator [opName] has incorrect value [incorrectValue]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, paramName, incorrectValue, reason)                            \
    do {                                                                                                            \
        std::string _safe_opName_(opName);                                                                          \
        std::string _safe_paramName_(paramName);                                                                    \
        std::string _safe_incorrectValue_(incorrectValue);                                                          \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_WITHOUT_REPORT(                                                                                     \
            _safe_opName_.c_str(), "Parameter %s of operator %s has incorrect value %s. Reason: %s.",               \
            _safe_paramName_.c_str(), _safe_opName_.c_str(), _safe_incorrectValue_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_name", "incorrect_value", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                 \
            _safe_opName_.c_str(), _safe_paramName_.c_str(), _safe_incorrectValue_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0026", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * EZ0027: Parameters value error (multiple parameters, with reason)
 * opName:string - Operator name
 * paramNames:string - Parameter names
 * incorrectValues:string - Actual incorrect values
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of operator [opName] have incorrect values [incorrectValues]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, paramNames, incorrectValues, reason)                           \
    do {                                                                                                              \
        std::string _safe_opName_(opName);                                                                            \
        std::string _safe_paramNames_(paramNames);                                                                    \
        std::string _safe_incorrectValues_(incorrectValues);                                                          \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_WITHOUT_REPORT(                                                                                       \
            _safe_opName_.c_str(), "Parameters %s of operator %s have incorrect values %s. Reason: %s.",              \
            _safe_paramNames_.c_str(), _safe_opName_.c_str(), _safe_incorrectValues_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"op_name", "param_names", "incorrect_values", "reason"};             \
        const std::vector<const char*> msgvalue = {                                                                   \
            _safe_opName_.c_str(), _safe_paramNames_.c_str(), _safe_incorrectValues_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0027", msgKey, msgvalue);                                                        \
    } while (0)

#define OP_LOGD(opName, ...) D_OP_LOGD(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_CHECK_IF(condition, log, return_expr) \
    do {                                         \
        if (unlikely(condition)) {               \
            log;                                 \
            return_expr;                         \
        }                                        \
    } while (0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                           \
    do {                                                                                                   \
        if (unlikely((ptr) == nullptr)) {                                                                  \
            const char* name = (unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr)) ? \
                                   "nil" :                                                                 \
                                   (context)->GetNodeName();                                               \
            OP_LOGE(name, "%s is nullptr!", #ptr);                                                         \
            return ge::GRAPH_FAILED;                                                                       \
        }                                                                                                  \
    } while (0)
OPBASE_API std::string ToString(ge::DataType type);
OPBASE_API std::string ToString(ge::Format format);
OPBASE_API std::string ToString(const gert::Shape& shape);
OPBASE_API std::string ToString(const std::vector<const gert::Shape*>& v);
} // namespace Base
} // namespace Ops
#endif // OP_COMMON_LOG_LOG_H
