#pragma once
#include <cstdio>
#include <cstdarg>

// Two-parameter form: OP_LOGD(nodeName, fmt, ...) - nodeName is ignored
#define OP_LOGD(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGI(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGW(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do {                                         \
        if (unlikely(condition)) {               \
            log;                                 \
            return_expr;                         \
        }                                        \
    } while (0)

#define OP_LOGE_WITHOUT_REPORT(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE_IF(condition, ret, nodeName, fmt, ...) \
    do { if (unlikely(condition)) { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); return ret; } } while(0)
#define OP_DBG_LEVEL 0
#define OP_LOG_IS_ON(level) (OP_DBG_LEVEL >= (level))
#define CheckLogLevel(lvl, submod) 0
#define DLOG_DEBUG 2
#define OP(moduleId) 0

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)         \
    if ((ptr) == nullptr)                                 \
    {                                                     \
        std::printf("nullptr error!");                    \
        return ge::GRAPH_SUCCESS;                         \
    }
