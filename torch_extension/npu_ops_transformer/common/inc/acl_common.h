/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file acl_common.h
 * \brief Runtime loading of ACL runtime library functions
 */

#ifndef NPU_OPS_TRANSFORMER_ACL_COMMON_H
#define NPU_OPS_TRANSFORMER_ACL_COMMON_H

#include "aclnn_common.h"
#include <dlfcn.h>
#include <c10/util/Exception.h>

using _AclrtGetSocName = const char *(*)(void);

static _AclrtGetSocName AclrtGetSocNameFunc = nullptr;

inline const char *GetAclRtLibName(void)
{
    return "libacl_rt.so";
}

template <typename T>
inline T GetAclRtFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetAclRtLibName());
    static T funcAddr = [&]() -> T {
        if (opApiHandler == nullptr) return nullptr;
        auto addr = dlsym(opApiHandler, apiName);
        if (addr == nullptr) {
            ASCEND_LOGW("dlsym %s from %s failed, error:%s", apiName, GetAclRtLibName(), dlerror());
        }
        return reinterpret_cast<T>(addr);
    }();
    return funcAddr;
}

inline void InitAclRtFunctions()
{
    AclrtGetSocNameFunc = GetAclRtFuncAddr<_AclrtGetSocName>("aclrtGetSocName");
    TORCH_CHECK(AclrtGetSocNameFunc != nullptr, "getAclrtGetSocName failed.");
}

#endif // NPU_OPS_TRANSFORMER_ACL_COMMON_H