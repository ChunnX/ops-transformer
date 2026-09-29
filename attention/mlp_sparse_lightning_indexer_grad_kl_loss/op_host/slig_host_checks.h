// Diagnostic-only checks. Avoid an external opbase source checkout for this reproducer.
#pragma once
#include <cstdio>
#define OP_LOGE(context, ...) do { (void)(context); std::fprintf(stderr, "[SLIG] "); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#define OP_CHECK_IF(condition, message, action) do { if (condition) { message; action; } } while (0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, pointer) \
    OP_CHECK_IF((pointer) == nullptr, OP_LOGE(context, "null: %s", #pointer), return ge::GRAPH_FAILED)
