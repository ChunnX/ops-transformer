/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Bench：与 Posembedding 其它任务一致，供 infershape / tiling 使用。
 */
#pragma once

#include <cstdio>

namespace {
#define OP_LOGD(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while (0)
#define OP_LOGI(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while (0)
#define OP_LOGE(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while (0)

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
        std::printf("nullptr error!"); \
        return ge::GRAPH_FAILED; \
    }

} // namespace
