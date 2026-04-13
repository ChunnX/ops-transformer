/**
 * Bench：满足 tiling 中日志宏与 `OP_LOGD/OP_LOGI/OP_LOGE(context, fmt, ...)` 调用约定。
 */
#pragma once

#include <cstdio>

#define OP_LOGD(_ctx, fmt, ...) \
    do { \
        (void)(_ctx); \
        std::printf("[D] " fmt "\n", ##__VA_ARGS__); \
    } while (0)
#define OP_LOGI(_ctx, fmt, ...) \
    do { \
        (void)(_ctx); \
        std::printf("[I] " fmt "\n", ##__VA_ARGS__); \
    } while (0)
#define OP_LOGE(_ctx, fmt, ...) \
    do { \
        (void)(_ctx); \
        std::printf("[E] " fmt "\n", ##__VA_ARGS__); \
    } while (0)
