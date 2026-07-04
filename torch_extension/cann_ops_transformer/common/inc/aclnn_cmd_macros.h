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
 * \file aclnn_cmd_macros.h
 * \brief ACLNN_CMD macro definition — the core macro that ties together all sub-headers.
 *        Including this header provides everything needed for ACLNN_CMD usage.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_CMD_MACROS_H
#define CANN_OPS_TRANSFORMER_ACLNN_CMD_MACROS_H

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#if __has_include("torch_npu/csrc/flopcount/FlopCount.h")
#include "torch_npu/csrc/flopcount/FlopCount.h"
#endif

#include "aclnn_opapi_loader.h"
#include "aclnn_type_convert.h"
#include "aclnn_hash_utils.h"
#include "aclnn_release_utils.h"
#include "aclnn_device_utils.h"

#define ACLNN_CMD(aclnn_api, ...)                                                                            \
    do {                                                                                                     \
        auto device = DecodeDevice(__VA_ARGS__);                                                             \
        const c10::OptionalDeviceGuard device_guard(device);                                                 \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");        \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                      \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                          \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                      \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                               \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,                         \
            #aclnn_api,                                                                                      \
            " or ",                                                                                          \
            #aclnn_api "GetWorkspaceSize",                                                                   \
            " not in ",                                                                                      \
            GetOpApiLibName(),                                                                               \
            ", or ",                                                                                         \
            GetOpApiLibName(),                                                                               \
            ", or ",                                                                                         \
            GetTransformerOpApiLibName(),                                                                    \
            "not found.");                                                                                   \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                      \
        uint64_t workspace_size = 0;                                                                         \
        uint64_t *workspace_size_addr = &workspace_size;                                                     \
        aclOpExecutor *executor = nullptr;                                                                   \
        aclOpExecutor **executor_addr = &executor;                                                           \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);          \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);  \
        if (initMemFunc) {                                                                                   \
            initMemFunc(nullptr, false);                                                                     \
        }                                                                                                    \
        auto deterministic = at::globalContext().deterministicAlgorithms();                                  \
        aclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, deterministic ? 1 : 0);                \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);               \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);   \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());     \
        at::Tensor workspace_tensor;                                                              \
        void *workspace_addr = nullptr;                                                                      \
        if (workspace_size != 0) {                                                                           \
            at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());          \
            workspace_tensor = at::empty({workspace_size}, options.dtype(at::kByte));                        \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                          \
        }                                                                                                    \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor] () -> int { \
            typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream);                  \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                  \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());          \
            ReleaseConvertTypes(converted_params);                                                           \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                \
            if (releaseMemFunc) {                                                                            \
                releaseMemFunc(nullptr, false);                                                              \
            }                                                                                                \
            return api_ret;                                                                                  \
        };                                                                                                   \
        at_npu::native::OpCommand cmd;                                                                       \
        cmd.Name(#aclnn_api);                                                                                \
        cmd.SetCustomHandler(acl_call);                                                                      \
        cmd.Run();                                                                                           \
        if (unInitMemFunc) {                                                                                 \
            unInitMemFunc(nullptr, false);                                                                   \
        }                                                                                                    \ 
    } while (false)

#endif  // CANN_OPS_TRANSFORMER_ACLNN_CMD_MACROS_H
