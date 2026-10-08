// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once

#include <acl/acl.h>
#include "shmem.h"
#include <sstream>
#include <stdexcept>
#include <string>

inline void CheckAclStatus(aclError status, const char* expression, const char* file, int line)
{
    if (status == ACL_SUCCESS) {
        return;
    }
    std::ostringstream oss;
    oss << "ACL call failed: " << expression << ", status=" << status << ", location=" << file << ':' << line;
    throw std::runtime_error(oss.str());
}

inline void CheckShmemStatus(int status, const char* expression, const char* file, int line)
{
    if (status == ACLSHMEM_SUCCESS) {
        return;
    }
    std::ostringstream oss;
    oss << "CANN SHMEM call failed: " << expression << ", status=" << status << ", location=" << file << ':' << line;
    throw std::runtime_error(oss.str());
}

#define ASCEND_DEEPEP_CHECK_ACL(expr) CheckAclStatus((expr), #expr, __FILE__, __LINE__)
#define ASCEND_DEEPEP_CHECK_SHMEM(expr) CheckShmemStatus((expr), #expr, __FILE__, __LINE__)
