/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This software is licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef MEGA_MOE_A8W4_TRACE_CONFIG_H
#define MEGA_MOE_A8W4_TRACE_CONFIG_H

#ifndef MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE
#define MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE 0
#endif

#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE != 0 && MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE != 1
#error "MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE must be 0 or 1"
#endif

namespace MegaMoeImpl {

constexpr unsigned long long A8W4_TRACE_OUTPUT_BYTES = 2ULL * 1024ULL * 1024ULL;
constexpr unsigned long long A8W4_TRACE_SCRATCH_BYTES = 50ULL * 1024ULL * 1024ULL;

} // namespace MegaMoeImpl

#endif
