#pragma once
#include "log.h"
#include <string>
#include <sstream>
#include <type_traits>

#define CUBE_INNER_ERR_REPORT(opName, fmt, ...) do { std::printf("[%s] " fmt "\n", opName, ##__VA_ARGS__); } while(0)
#define VECTOR_INNER_ERR_REPORT_TILIING(opName, fmt, ...) do { std::printf("[%s] " fmt "\n", opName, ##__VA_ARGS__); } while(0)

#define OP_TILING_CHECK(condition, log, return_expr) \
    do {                                             \
        if (condition) {                              \
            log;                                      \
            return_expr;                              \
        }                                             \
    } while (0)

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                        \
        std::printf("nullptr error!");             \
        return ge::GRAPH_FAILED;                   \
    }

#define OPS_REPORT_CUBE_INNER_ERR(opName, fmt, ...)
#define OPS_REPORT_VECTOR_INNER_ERR(opName, fmt, ...)
#define REPORT_INNER_ERR(...)

#define OPS_LOG_D(...)
#define OPS_LOG_I(...)
#define OPS_LOG_W(...)
#define OPS_LOG_E(...)
#define OPS_LOG_EVENT(...)

#define OPS_ERR_IF(condition, log, return_expr) \
    do {                                         \
        if (condition) {                          \
            log;                                  \
            return_expr;                          \
        }                                         \
    } while (0)

#define OP_LOGW_IF(condition, opName, fmt, ...) \
    do { if (unlikely(condition)) { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } } while(0)

#define OP_LOGI_IF_RETURN(condition, returnValue, opName, fmt, ...) \
    do { if (unlikely(condition)) { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); return returnValue; } } while(0)

namespace Ops { namespace Base {
inline std::string GetSafeStr(const char* s) { return s ? std::string(s) : ""; }
inline const char* GetOpInfo(const char* opName) { return opName ? opName : ""; }
template <typename T>
std::string ToString(const T& shape) {
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}
}}
