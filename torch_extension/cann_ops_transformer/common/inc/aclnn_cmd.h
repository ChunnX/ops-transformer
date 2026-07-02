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
 * \file aclnn_cmd.h
 * \brief ACLNN_CMD macro, hash buffer utilities (MEMCPY_TO_BUF, AddParamToBuf, CalcHashId),
 *        and memory management type aliases for ACLNN operator execution.
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_CMD_H
#define CANN_OPS_TRANSFORMER_ACLNN_CMD_H

#include "aclnn_type_convert.h"

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;
extern thread_local char g_hashBuf[kHashBufSize];
extern thread_local int g_hashOffset;

#define MEMCPY_TO_BUF(data_expression, size_expression)                                    \
    if (g_hashOffset + (size_expression) > kHashBufSize) {                                 \
        g_hashOffset = kHashBufMaxSize;                                                    \
        return;                                                                            \
    }                                                                                      \
    auto ret = memcpy_s(g_hashBuf + g_hashOffset, size_expression, data_expression, size_expression); \
    TORCH_CHECK(ret == 0, "memcpy_s failed, error:", ret);                               \
    g_hashOffset += size_expression;

template <std::size_t N>
void AddParamToBuf(const std::array<bool, N> &value)
{
    MEMCPY_TO_BUF(value.data(), value.size() * sizeof(bool));
}

template <typename T>
void AddParamToBuf(const T &value)
{
    MEMCPY_TO_BUF(&value, sizeof(T));
}

void AddParamToBuf(const at::Tensor &);
void AddParamToBuf(const at::Scalar &);
void AddParamToBuf(const at::IntArrayRef &);
void AddParamToBuf(const at::ArrayRef<bool> &);
void AddParamToBuf(const at::TensorList &);
void AddParamToBuf(const c10::optional<at::Tensor> &);
void AddParamToBuf(const c10::optional<at::IntArrayRef> &);
void AddParamToBuf(const c10::optional<at::Scalar> &);
void AddParamToBuf(const at::ScalarType);
void AddParamToBuf(const string &);
void AddParamToBuf();

template <typename T, typename... Args>
void AddParamToBuf(const T &arg, Args &...args)
{
    AddParamToBuf(arg);
    AddParamToBuf(args...);
}

uint64_t CalcHashId();
using InitHugeMemThreadLocal = int (*)(void *, bool);
using UnInitHugeMemThreadLocal = void (*)(void *, bool);
using ReleaseHugeMem = void (*)(void *, bool);

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

#endif  // CANN_OPS_TRANSFORMER_ACLNN_CMD_H