/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FULLMESH_V2_PROFILER_DUMP_H
#define FULLMESH_V2_PROFILER_DUMP_H

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "acl/acl.h"

namespace FullMeshV2ProfilerDump {
constexpr aclError PROFILER_DUMP_ERROR = static_cast<aclError>(-1);
constexpr uint32_t PROFILER_MAX_CORE_NUM = 72U;
constexpr uint32_t PROFILER_MAX_EVENT_NUM = 32U;
constexpr uint32_t PROFILER_RECORD_SIZE = 16U;
constexpr uint32_t PROFILER_HEADER_SIZE = 64U;
constexpr uint32_t PROFILER_CORE_COUNT_ALIGN = 64U;
constexpr uint64_t PROFILER_BUFFER_SIZE = PROFILER_HEADER_SIZE +
    static_cast<uint64_t>(PROFILER_MAX_CORE_NUM) * PROFILER_CORE_COUNT_ALIGN +
    static_cast<uint64_t>(PROFILER_MAX_CORE_NUM) * PROFILER_MAX_EVENT_NUM * PROFILER_RECORD_SIZE;
constexpr uint32_t PROFILER_MAGIC = 0x50524F46U;

inline aclError Dump(void *profilerBuffer, uint64_t bufferSize, uint32_t rankId, const std::string &outputDir = ".")
{
    if (profilerBuffer == nullptr || bufferSize < PROFILER_BUFFER_SIZE) {
        return PROFILER_DUMP_ERROR;
    }
    std::vector<uint8_t> hostBuffer(PROFILER_BUFFER_SIZE);
    aclError ret = aclrtMemcpy(hostBuffer.data(), hostBuffer.size(), profilerBuffer, hostBuffer.size(),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    if (*reinterpret_cast<const uint32_t *>(hostBuffer.data()) != PROFILER_MAGIC) {
        return PROFILER_DUMP_ERROR;
    }
    std::string path = outputDir + "/fullmesh_v2_prof_rank" + std::to_string(rankId) + ".bin";
    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        return PROFILER_DUMP_ERROR;
    }
    output.write(reinterpret_cast<const char *>(hostBuffer.data()), hostBuffer.size());
    return output.good() ? ACL_SUCCESS : PROFILER_DUMP_ERROR;
}
} // namespace FullMeshV2ProfilerDump

#endif // FULLMESH_V2_PROFILER_DUMP_H
