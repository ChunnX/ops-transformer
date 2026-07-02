/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_api_loader.h
 * \brief Dynamic library loading (dlopen/dlsym) and operator API function address resolution.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_API_LOADER_H
#define CANN_OPS_TRANSFORMER_ACLNN_API_LOADER_H

#include "aclnn_types.h"
#include <dlfcn.h>
#include <sstream>
#include <cstdlib>
#include <climits>

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

inline const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline const char *GetTransformerOpApiLibName(void)
{
    return "libopapi_transformer.so";
}

inline const char *GetCustOpApiLibName(void)
{
    return "libcust_opapi.so";
}

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

inline std::vector<void *> GetCustOpApiHandlers()
{
    std::vector<void *> handlers;
    const char *env = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    if (env != nullptr) {
        std::string envStr(env);
        std::istringstream iss(envStr);
        std::string path;
        while (std::getline(iss, path, ':')) {
            if (path.empty()) {
                continue;
            }
            std::string soPathStr = path + "/op_api/lib/" + GetCustOpApiLibName();
            char soPath[PATH_MAX] = {0};
            if (realpath(soPathStr.c_str(), soPath) == nullptr) {
                ASCEND_LOGW("realpath failed for %s.", soPathStr.c_str());
                continue;
            }
            auto handler = dlopen(soPath, RTLD_LAZY);
            if (handler != nullptr) {
                handlers.push_back(handler);
            } else {
                ASCEND_LOGW("dlopen %s failed, error:%s.", soPath, dlerror());
            }
        }
    }
    if (handlers.empty()) {
        auto handler = GetOpApiLibHandler(GetCustOpApiLibName());
        if (handler != nullptr) {
            handlers.push_back(handler);
        }
    }
    return handlers;
}

inline void *GetOpApiFuncAddr(const char *apiName)
{
    static auto custHandlers = GetCustOpApiHandlers();
    for (auto handler : custHandlers) {
        auto funcAddr = GetOpApiFuncAddrInLib(handler, GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }

    static auto transformerOpApiHandler = GetOpApiLibHandler(GetTransformerOpApiLibName());
    if (transformerOpApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(transformerOpApiHandler, GetTransformerOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

#endif  // CANN_OPS_TRANSFORMER_ACLNN_API_LOADER_H