/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#define OPS_LOG_D(OPS_DESC, ...) ((void)0)
#define OPS_LOG_I(OPS_DESC, ...) ((void)0)
#define OPS_LOG_W(OPS_DESC, ...) ((void)0)
#define OPS_LOG_E(OPS_DESC, ...) ((void)0)
#define OPS_LOG_E_WITHOUT_REPORT(OPS_DESC, ...) ((void)0)
#define OPS_LOG_EVENT(OPS_DESC, ...) ((void)0)

#define OPS_LOG_FULL(LEVEL, OPS_DESC, ...) ((void)0)
#define OPS_LOG_D_FULL(OPS_DESC, ...) ((void)0)
#define OPS_LOG_I_FULL(OPS_DESC, ...) ((void)0)
#define OPS_LOG_W_FULL(OPS_DESC, ...) ((void)0)

#define OPS_LOG_D_IF(COND, OP_DESC, EXPR, ...) \
    do {                                       \
        if (COND) {                            \
            EXPR;                              \
        }                                      \
    } while (0)

#define OPS_LOG_I_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_D_IF(COND, OP_DESC, EXPR, ##__VA_ARGS__)
#define OPS_LOG_W_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_D_IF(COND, OP_DESC, EXPR, ##__VA_ARGS__)
#define OPS_LOG_E_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_D_IF(COND, OP_DESC, EXPR, ##__VA_ARGS__)
#define OPS_LOG_EVENT_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_D_IF(COND, OP_DESC, EXPR, ##__VA_ARGS__)

#define OPS_LOG_E_IF_NULL(OPS_DESC, PTR, EXPR) \
    do {                                       \
        if ((PTR) == nullptr) {                \
            EXPR;                              \
        }                                      \
    } while (0)

#define OPS_CHECK(COND, LOG_FUNC, EXPR) \
    do {                                \
        if (COND) {                     \
            LOG_FUNC;                   \
            EXPR;                       \
        }                               \
    } while (0)

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    do {                                        \
        if (unlikely((ptr) == nullptr)) {       \
            return ge::GRAPH_FAILED;            \
        }                                       \
    } while (0)
