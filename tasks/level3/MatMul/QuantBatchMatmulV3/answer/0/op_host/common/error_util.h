#pragma once
#include "log.h"
#include <string>

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

// OP_LOGE_IF / OP_LOGW_IF are defined in log.h with the correct 4-param signature:
//   OP_LOGE_IF(condition, ret, nodeName, fmt, ...)
// Do NOT redefine them here.
