/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_COMMON_INC_OP_LOG_H_
#define PTA_NPU_OP_API_COMMON_INC_OP_LOG_H_

#include <stdint.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include "op_errno.h"
#ifdef __GNUC__
#include <sys/syscall.h>
#include <unistd.h>
#endif

#define OPAPI_SUBMOD_NAME "NNOP"
#define OP_ID 63
#define LOG_FMT_IDX 3
#define LOG_AGR_IDX 4

namespace op {
class OpLog {
public:
    static uint64_t GetTid()
    {
#ifdef __GNUC__
        const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
        const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
        return tid;
    }
};
}

int CheckLogLevel(int32_t moduleId, int32_t logLevel);

#ifdef ACLNN_LOG_FMT_CHECK
void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...) __attribute__((format(printf,
                                                                            LOG_FMT_IDX, LOG_AGR_IDX)));
#else
void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...);
#endif

#define OP_LOG_DEBUG 0
#define OP_LOG_INFO 1
#define OP_LOG_WARN 2
#define OP_LOG_ERROR 3
#define OP_LOG_EVENT 0x10
#define RUN_LOG_MASK (0x01000000U)

constexpr const char *GetFileName(const char *path)
{
    const char *file = path;
    while (*path != '\0') {
        if (*path++ == '/') {
            file = path;
        }
    }
    return file;
}

inline std::string GetOpName()
{
    std::string opInfo = "OpName:[ Detection ] ";
    return opInfo;
}

#define DOplogSub(moduleId, submodule, level, fmt, ...)                                                                \
    do {                                                                                                               \
        if (CheckLogLevel(moduleId, level) == 1) {                                                                     \
            DlogRecord(moduleId, level, "[%s:%d][%s]" fmt, GetFileName(__FILE__), __LINE__, submodule, ##__VA_ARGS__); \
        }                                                                                                              \
    } while (false)

#define DDfxlogSub(moduleId, submodule, level, fmt, ...)                                                                                \
    do {                                                                                                                                \
        DlogRecord((moduleId | RUN_LOG_MASK), level, "[%s:%d][%s]" fmt, GetFileName(__FILE__), __LINE__, submodule, ##__VA_ARGS__); \
    } while (false)

#if defined(NNOPBASE_UT) || defined(NNOPBASE_ST)
#define OP_TEST_LOG(fmt, ...)                                                    \
    do {                                                                         \
        fprintf(stdout, "[OP_TEST] [tid: %lu][%s:%d] %s:" fmt "\n",              \
            op::OpLog::GetTid(), __FILE__, __LINE__, __FUNCTION, ##__VA_ARGS__); \
        fflush(stdout);                                                          \
    } while (0)
#define OP_LOGI(...) OP_TEST_LOG(__VA_ARGS__)
#define OP_LOGW(...) OP_TEST_LOG(__VA_ARGS__)
#define OP_LOGE(errno, ...) OP_TEST_LOG(__VA_ARGS__)
#define OP_LOGE_WITHOUT_REPORT(errno, ...) OP_TEST_LOG(__VA_ARGS__)
#define OP_LOGD(...) OP_TEST_LOG(__VA_ARGS__)
#define OP_EVENT(...) OP_TEST_LOG(__VA_ARGS__)
#define OP_DFX_LOGD(file, line, func, ...)
#define OP_DFX_LOGI(file, line, func, ...)
#define OP_DFX_LOGW(file, line, func, ...)
#define OP_DFX_LOGE(file, line, func, ...)
#elif !defined(__ANDROID__) && !defined(ANDROID)
#define OP_LOGI(...) D_OP_LOGI(GetOpName().c_str(), __VA_ARGS__)
#define OP_LOGW(...) D_OP_LOGW(GetOpName().c_str(), __VA_ARGS__)

#define OP_DFX_LOGD(...) D_OP_DFX_LOGD(GetOpName().c_str(), __VA_ARGS__)
#define OP_DFX_LOGI(...) D_OP_DFX_LOGI(GetOpName().c_str(), __VA_ARGS__)
#define OP_DFX_LOGW(...) D_OP_DFX_LOGW(GetOpName().c_str(), __VA_ARGS__)
#define OP_DFX_LOGE(...) D_OP_DFX_LOGW(GetOpName().c_str(), __VA_ARGS__)
#define OP_LOGE_WITHOUT_REPORT(errno, ...) D_OP_LOGE(GetOpName().c_str(), errno, __VA_ARGS__)
#define OP_LOGE(errno, ...)                         \
    do {                                            \
        OP_LOGE_WITHOUT_REPORT(errno, __VA_ARGS__); \
    } while (false)

#define OP_LOGD(...) D_OP_LOGD(GetOpName().c_str(), __VA_ARGS__)
#define OP_EVENT(...) D_OP_EVENT(GetOpName().c_str(), __VA_ARGS__)
#else
#define OP_LOGI(...)
#define OP_LOGW(...)
#define OP_LOGE_WITHOUT_REPORT(...)
#define OP_LOGE(...)
#define OP_LOGD(...)
#define OP_EVENT(...)
#endif

#define OpLogSub(moduleId, level, op_info, fmt, ...)                                                      \
    DOplogSub(static_cast<int32_t>(moduleId), OPAPI_SUBMOD_NAME, level, "[%s][%lu] %s" fmt, __FUNCTION__, \
            op::OpLog::GetTid(), op_info, ##__VA_ARGS__)

#define OpLogErrSub(moduleId, level, op_info, errno, fmt, ...)                                                      \
    DOplogSub(static_cast<int32_t>(moduleId), OPAPI_SUBMOD_NAME, level, "[%s][%lu] errno[%d] %s" fmt, __FUNCTION__, \
            op::OpLog::GetTid(), errno, op_info, ##__VA_ARGS__)

#define OpDfxLogSub(moduleId, level, op_info, fmt, ...)                                                    \
    DDfxlogSub(static_cast<int32_t>(moduleId), OPAPI_SUBMOD_NAME, level, "[%s][%lu] %s" fmt, __FUNCTION__, \
            op::OpLog::GetTid(), op_info, ##__VA_ARGS__)

#if !defined(__ANDROID__) && !defined(ANDROID)
#define D_OP_LOGI(opname, fmt, ...) OpLogSub(OP_ID, OP_LOG_INFO, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGW(opname, fmt, ...) OpLogSub(OP_ID, OP_LOG_WARN, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGE(opname, errno, fmt, ...) OpLogErrSub(OP_ID, OP_LOG_ERROR, opname, errno, fmt, ##__VA_ARGS__)
#define D_OP_LOGD(opname, fmt, ...) OpLogSub(OP_ID, OP_LOG_DEBUG, opname, fmt, ##__VA_ARGS__)
#define D_OP_EVENT(opname, fmt, ...) OpLogSub(OP_ID, OP_LOG_EVENT, opname, fmt, ##__VA_ARGS__)

#define D_OP_DFX_LOGI(opname, fmt, ...) \
OpDfxLogSub(OP_ID, OP_LOG_INFO, opname, fmt, ##__VA_ARGS__)
#define D_OP_DFX_LOGW(opname, fmt, ...) \
OpDfxLogSub(OP_ID, OP_LOG_WARN, opname, fmt, ##__VA_ARGS__)
#define D_OP_DFX_LOGE(opname, fmt, ...) \
OpDfxLogSub(OP_ID, OP_LOG_ERROR, opname, fmt, ##__VA_ARGS__)
#define D_OP_DFX_LOGD(opname, fmt, ...) \
OpDfxLogSub(OP_ID, OP_LOG_DEBUG, opname, fmt, ##__VA_ARGS__)

#else
#define D_OP_LOGI(opname, fmt, ...)
#define D_OP_LOGW(opname, fmt, ...)
#define D_OP_LOGE(opname, fmt, ...)
#define D_OP_LOGD(opname, fmt, ...)
#define D_OP_EVENT(opname, fmt, ...)
#endif

#define unlikely(x) __builtin_expect((x), 0)
#define likely(x) __builtin_expect((x), 1)

#endif //PTA_NPU_OP_API_COMMON_INC_OP_LOG_H_