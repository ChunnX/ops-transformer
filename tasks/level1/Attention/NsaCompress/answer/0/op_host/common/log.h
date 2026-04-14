/**
 * Stub for msopgen / Bench 路径下缺失的 CANN `log/log.h`（见 `.codex/references/compile_debug.md` §5.4）。
 */
#pragma once

#include <cstdio>

namespace {
#define OP_LOGD(nodeName, fmt, ...) \
    do { \
        (void)(nodeName); \
        std::printf(fmt, ##__VA_ARGS__); \
        std::printf("\n"); \
    } while (0)
#define OP_LOGI(nodeName, fmt, ...) \
    do { \
        (void)(nodeName); \
        std::printf(fmt, ##__VA_ARGS__); \
        std::printf("\n"); \
    } while (0)
#define OP_LOGE(nodeName, fmt, ...) \
    do { \
        (void)(nodeName); \
        std::printf(fmt, ##__VA_ARGS__); \
        std::printf("\n"); \
    } while (0)
#define OP_LOGW(nodeName, fmt, ...) OP_LOGD(nodeName, fmt, ##__VA_ARGS__)

#ifndef OP
#define OP 0
#endif
#ifndef DLOG_DEBUG
#define DLOG_DEBUG 0
#endif
inline int32_t CheckLogLevel(int32_t, int32_t) { return 0; }

#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do { \
        if (unlikely(condition)) { \
            log; \
            return_expr; \
        } \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) { \
        std::printf("nullptr error!\n"); \
        return ge::GRAPH_FAILED; \
    }
} // namespace
