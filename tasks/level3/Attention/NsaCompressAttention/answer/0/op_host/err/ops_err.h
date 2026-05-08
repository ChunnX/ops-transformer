/**
 * Bench 路径下 err/ops_err.h 桩，满足 NsaCompress tiling 中 OPS_REPORT_* 宏。
 */
#pragma once

/* 依赖编译单元先包含 common/log.h（见 *tiling*.cpp） */

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
