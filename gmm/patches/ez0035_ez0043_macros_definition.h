/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 
 * 新增错误码宏定义（EZ0035-EZ0043）
 * 用于规范化内部错误和特殊场景的错误上报
 * 
 * 文件位置：third_party/opbase/pkg_inc/op_common/log/log.h
 * 新增行号：791-948
 * 
 * 使用说明：
 * 1. 将此文件内容添加到 log.h 的 EZ0027 定义之后
 * 2. 或者直接替换 log.h 中对应位置的内容
 * 
 * 整改统计：
 * - EZ0035 (nullptr): 56个场景
 * - EZ0037 (tiling失败): 8个场景  
 * - EZ0039 (memcpy失败): 2个场景
 * - EZ0040 (Cube错误): 1个场景
 * - EZ0042 (内部检查): 56个场景
 * 总计：123个错误码场景
 */

#ifndef EZ0035_EZ0043_MACROS_H
#define EZ0035_EZ0043_MACROS_H

/**
 * EZ0035: Nullptr error
 * opName:string - Operator name
 * paramName:string - Parameter name (e.g. "x", "weight", "scale")
 * errMessage: Parameter [paramName] of operator [opName] is null.
 */
#define OP_LOGE_FOR_NULLPTR(opName, paramName)                                                       \
    do {                                                                                             \
        std::string _safe_opName_(opName);                                                           \
        std::string _safe_paramName_(paramName);                                                     \
        OP_LOGE_WITHOUT_REPORT(                                                                      \
            _safe_opName_.c_str(), "Parameter %s of operator %s is null.", _safe_paramName_.c_str(), \
            _safe_opName_.c_str());                                                                  \
        const std::vector<const char*> msgKey = {"op_name", "param_name"};                           \
        const std::vector<const char*> msgvalue = {_safe_opName_.c_str(), _safe_paramName_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0035", msgKey, msgvalue);                                       \
    } while (0)

/**
 * EZ0036: Tiling calculation failed
 * opName:string - Operator name
 * reason:string - Failure reason
 * errMessage: Op [opName] tiling failed. Reason: [reason].
 */
#define OP_LOGE_FOR_TILING_FAILED(opName, reason)                                                 \
    do {                                                                                          \
        std::string _safe_opName_(opName);                                                        \
        std::string _safe_reason_(reason);                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                   \
            _safe_opName_.c_str(), "Op [%s] tiling failed. Reason: %s.", _safe_opName_.c_str(),   \
            _safe_reason_.c_str());                                                               \
        const std::vector<const char*> msgKey = {"op_name", "reason"};                            \
        const std::vector<const char*> msgvalue = {_safe_opName_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0036", msgKey, msgvalue);                                    \
    } while (0)

/**
 * EZ0037: Validation failed
 * opName:string - Operator name
 * reason:string - Validation failure reason
 * errMessage: Op [opName] input validation failed. Reason: [reason].
 */
#define OP_LOGE_FOR_VALIDATION_FAILED(opName, reason)                                                     \
    do {                                                                                                  \
        std::string _safe_opName_(opName);                                                                \
        std::string _safe_reason_(reason);                                                                \
        OP_LOGE_WITHOUT_REPORT(                                                                           \
            _safe_opName_.c_str(), "Op [%s] input validation failed. Reason: %s.", _safe_opName_.c_str(), \
            _safe_reason_.c_str());                                                                       \
        const std::vector<const char*> msgKey = {"op_name", "reason"};                                    \
        const std::vector<const char*> msgvalue = {_safe_opName_.c_str(), _safe_reason_.c_str()};         \
        REPORT_PREDEFINED_ERR_MSG("EZ0037", msgKey, msgvalue);                                            \
    } while (0)

/**
 * EZ0038: Memory operation failed
 * opName:string - Operator name
 * retCode:int - Memory operation return code
 * errMessage: Memory copy failed in Op [opName]. Return code: [retCode].
 */
#define OP_LOGE_FOR_MEMORY_OP_FAILED(opName, retCode)                                                        \
    do {                                                                                                     \
        std::string _safe_opName_(opName);                                                                   \
        std::string _safe_retCode_(std::to_string(retCode));                                                 \
        OP_LOGE_WITHOUT_REPORT(                                                                              \
            _safe_opName_.c_str(), "Memory copy failed in Op [%s]. Return code: %s.", _safe_opName_.c_str(), \
            _safe_retCode_.c_str());                                                                         \
        const std::vector<const char*> msgKey = {"op_name", "return_code"};                                  \
        const std::vector<const char*> msgvalue = {_safe_opName_.c_str(), _safe_retCode_.c_str()};           \
        REPORT_PREDEFINED_ERR_MSG("EZ0038", msgKey, msgvalue);                                               \
    } while (0)

/**
 * EZ0039: Memory copy failed (deprecated, use EZ0038)
 * opName:string - Operator name
 * retCode:int - Memory copy return code
 * errMessage: Memory copy failed in Op [opName]. Return code: [retCode].
 */
#define OP_LOGE_FOR_MEMCPY_FAILED(opName, retCode) OP_LOGE_FOR_MEMORY_OP_FAILED(opName, retCode)

/**
 * EZ0040: Internal cube error
 * opName:string - Operator name
 * reason:string - Cube error reason
 * errMessage: Op [opName] cube inner error. Reason: [reason].
 */
#define OP_LOGE_FOR_CUBE_INNER_ERROR(opName, reason)                                               \
    do {                                                                                           \
        std::string _safe_opName_(opName);                                                         \
        std::string _safe_reason_(reason);                                                         \
        OP_LOGE_WITHOUT_REPORT(                                                                    \
            _safe_opName_.c_str(), "Op [%s] cube inner error. Reason: %s.", _safe_opName_.c_str(), \
            _safe_reason_.c_str());                                                                \
        const std::vector<const char*> msgKey = {"op_name", "reason"};                             \
        const std::vector<const char*> msgvalue = {_safe_opName_.c_str(), _safe_reason_.c_str()};  \
        REPORT_PREDEFINED_ERR_MSG("EZ0040", msgKey, msgvalue);                                     \
    } while (0)

/**
 * EZ0041: Function execution failed
 * opName:string - Operator name
 * functionName:string - Failed function name
 * errMessage: Function [functionName] execution failed in Op [opName].
 */
#define OP_LOGE_FOR_FUNCTION_FAILED(opName, functionName)                                                   \
    do {                                                                                                    \
        std::string _safe_opName_(opName);                                                                  \
        std::string _safe_functionName_(functionName);                                                      \
        OP_LOGE_WITHOUT_REPORT(                                                                             \
            _safe_opName_.c_str(), "Function %s execution failed in Op [%s].", _safe_functionName_.c_str(), \
            _safe_opName_.c_str());                                                                         \
        const std::vector<const char*> msgKey = {"op_name", "function_name"};                               \
        const std::vector<const char*> msgvalue = {_safe_opName_.c_str(), _safe_functionName_.c_str()};     \
        REPORT_PREDEFINED_ERR_MSG("EZ0041", msgKey, msgvalue);                                              \
    } while (0)

/**
 * EZ0042: Internal check failed
 * opName:string - Operator name
 * reason:string - Check failure reason
 * errMessage: Op [opName] internal check failed. Reason: [reason].
 */
#define OP_LOGE_FOR_INTERNAL_ERROR(opName, reason)                                                      \
    do {                                                                                                \
        std::string _safe_opName_(opName);                                                              \
        std::string _safe_reason_(reason);                                                              \
        OP_LOGE_WITHOUT_REPORT(                                                                         \
            _safe_opName_.c_str(), "Op [%s] internal check failed. Reason: %s.", _safe_opName_.c_str(), \
            _safe_reason_.c_str());                                                                     \
        const std::vector<const char*> msgKey = {"op_name", "reason"};                                  \
        const std::vector<const char*> msgvalue = {_safe_opName_.c_str(), _safe_reason_.c_str()};       \
        REPORT_PREDEFINED_ERR_MSG("EZ0042", msgKey, msgvalue);                                          \
    } while (0)

/**
 * EZ0043: Platform/context error
 * opName:string - Operator name
 * reason:string - Platform/context error reason
 * errMessage: Op [opName] platform error. Reason: [reason].
 */
#define OP_LOGE_FOR_PLATFORM_ERROR(opName, reason)                                                \
    do {                                                                                          \
        std::string _safe_opName_(opName);                                                        \
        std::string _safe_reason_(reason);                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                   \
            _safe_opName_.c_str(), "Op [%s] platform error. Reason: %s.", _safe_opName_.c_str(),  \
            _safe_reason_.c_str());                                                               \
        const std::vector<const char*> msgKey = {"op_name", "reason"};                            \
        const std::vector<const char*> msgvalue = {_safe_opName_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0043", msgKey, msgvalue);                                    \
    } while (0)

// Legacy compatibility macros (deprecated)
#define OP_LOGE_FOR_CHECK_FAILED(opName, reason) OP_LOGE_FOR_INTERNAL_ERROR(opName, reason)
#define OP_LOGE_FOR_TILING_FAIL(opName, reason) OP_LOGE_FOR_TILING_FAILED(opName, reason)
#define OP_LOGE_FOR_VALIDATION_FAIL(opName, reason) OP_LOGE_FOR_VALIDATION_FAILED(opName, reason)
#define OP_LOGE_FOR_MEMCPY_FAIL(opName, ret) OP_LOGE_FOR_MEMORY_OP_FAILED(opName, ret)
#define OP_LOGE_FOR_CUBE_INNER_ERR(opName, reason) OP_LOGE_FOR_CUBE_INNER_ERROR(opName, reason)

#endif // EZ0035_EZ0043_MACROS_H