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
 * \brief Simplified MC2 log header for NPUKernelBench
 */

#ifndef MC2_LOG_H
#define MC2_LOG_H

#include <stdarg.h>
#include <stdio.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>

#include "graph/ge_error_codes.h"

namespace {

#define unlikely(x) __builtin_expect((x), 0)

#define OP_LOGD(nodeName, fmt, ...) do { std::printf("[%s] ", nodeName); std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGI(nodeName, fmt, ...) do { std::printf("[%s] ", nodeName); std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE(nodeName, fmt, ...) do { std::printf("[%s] ERROR: ", nodeName); std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE_WITHOUT_REPORT(nodeName, fmt, ...) do { std::printf("[%s] ERROR: ", nodeName); std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define REPORT_INNER_ERR_MSG(errCode, fmt, ...) do { std::printf("[%s] ", errCode); std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

#define OPS_LOG_D(opName, ...) OP_LOGD(opName, __VA_ARGS__)
#define OPS_LOG_I(opName, ...) OP_LOGI(opName, __VA_ARGS__)
#define OPS_LOG_E(opName, ...) OP_LOGE(opName, __VA_ARGS__)

#define OP_LOGE_IF(condition, returnValue, opName, fmt, ...)                                                     \
    do {                                                                                                         \
        if (unlikely(condition)) {                                                                               \
            OP_LOGE(opName, fmt, ##__VA_ARGS__);                                                                 \
            return returnValue;                                                                                  \
        }                                                                                                        \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)         \
    if ((ptr) == nullptr)                                \
    {                                                    \
        std::printf("nullptr error!");                  \
        return ge::GRAPH_FAILED;                        \
    }

#define OPS_ERR_IF(COND, LOG_FUNC, EXPR) \
    if (__builtin_expect(COND, 0)) {    \
        LOG_FUNC;                       \
        EXPR;                           \
    }

#define OPS_CHECK(COND, LOG_FUNC, EXPR) \
    if (COND) {                         \
        LOG_FUNC;                       \
        EXPR;                           \
    }

}

inline std::vector<char> CreateErrorMsg(const char *format, ...) __attribute__((format(printf, 1, 2)));
inline std::vector<char> CreateErrorMsg();

inline std::vector<char> CreateErrorMsg(const char *format, ...) {
    va_list args;
    va_start(args, format);
    const size_t len = 4095;
    std::vector<char> msg(len + 1U, '\0');
    vsnprintf(msg.data(), len + 1U, format, args);
    va_end(args);
    return msg;
}

inline std::vector<char> CreateErrorMsg() { return {}; }

inline const char *ConvertStringToCstr(const std::string &str) { return str.c_str(); }

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

struct ErrorResult {
    operator bool() const { return false; }
    operator ge::graphStatus() const { return ge::GRAPH_PARAM_INVALID; }
    template <typename T>
    operator std::unique_ptr<T>() const { return nullptr; }
    template <typename T>
    operator std::shared_ptr<T>() const { return nullptr; }
    template <typename T>
    operator T *() const { return nullptr; }
    template <typename T>
    operator std::vector<std::shared_ptr<T>>() const { return {}; }
    template <typename T>
    operator std::vector<T>() const { return {}; }
    operator std::string() const { return ""; }
    template <typename T>
    operator T() const { return T(); }
};

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) \
    OP_LOGE(op_name, err_msg, ##__VA_ARGS__)

#endif // MC2_LOG_H