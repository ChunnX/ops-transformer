#pragma once
#include <cstdio>
#include <cstdarg>
#include "dlog_pub.h"

// Log level definitions
#define DLOG_DEBUG 3
#define DLOG_INFO 4
#define DLOG_WARN 5
#define DLOG_ERROR 6

// Two-parameter form: OP_LOGD(nodeName, fmt, ...) - nodeName is ignored
#define OP_LOGD(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGI(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGW(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE_WITHOUT_REPORT(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do {                                         \
        if (unlikely(condition)) {               \
            log;                                 \
            return_expr;                         \
        }                                        \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)         \
    if ((ptr) == nullptr)                                 \
    {                                                     \
        std::printf("nullptr error!");                    \
        return ge::GRAPH_SUCCESS;                         \
    }
