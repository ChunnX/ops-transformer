/**
 * 需在定义 OPS_REPORT_* 之前可见 OP_LOGE（见 ifa_bench_log.h）。
 */
#pragma once

#include "../ifa_bench_log.h"

#ifndef REPORT_INNER_ERR_MSG
#define REPORT_INNER_ERR_MSG(...) ((void)0)
#endif

#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) \
    do { \
        OP_LOGE(OPS_DESC, __VA_ARGS__); \
    } while (0)

#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...) \
    do { \
        OP_LOGE(OPS_DESC, __VA_ARGS__); \
    } while (0)
