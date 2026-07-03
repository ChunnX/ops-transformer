/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include <vector>
#include <array>
#include <cstdint>
#include <dlfcn.h>
#include "acl/acl.h"
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_sparse_flash_attention.h"
#include "../../../op_host/op_api/aclnn_sparse_flash_attention_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
using namespace std;
using namespace op;

namespace {
void DestroyAclTensor(aclTensor *tensor)
{
    Release(tensor);
}

class AclRuntimeGuard {
public:
    AclRuntimeGuard()
    {
        handle_ = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
        if (handle_ == nullptr) {
            return;
        }
        aclInit_ = reinterpret_cast<AclInitFunc>(dlsym(handle_, "aclInit"));
        aclFinalize_ = reinterpret_cast<AclFinalizeFunc>(dlsym(handle_, "aclFinalize"));
        aclrtSetDevice_ = reinterpret_cast<AclrtSetDeviceFunc>(dlsym(handle_, "aclrtSetDevice"));
        aclrtResetDevice_ = reinterpret_cast<AclrtResetDeviceFunc>(dlsym(handle_, "aclrtResetDevice"));
        aclrtCreateContext_ = reinterpret_cast<AclrtCreateContextFunc>(dlsym(handle_, "aclrtCreateContext"));
        aclrtDestroyContext_ = reinterpret_cast<AclrtDestroyContextFunc>(dlsym(handle_, "aclrtDestroyContext"));
        if (aclInit_ == nullptr || aclFinalize_ == nullptr || aclrtSetDevice_ == nullptr ||
            aclrtResetDevice_ == nullptr || aclrtCreateContext_ == nullptr || aclrtDestroyContext_ == nullptr) {
            return;
        }

        initRet_ = aclInit_(nullptr);
        if (initRet_ != ACL_SUCCESS) {
            return;
        }
        setDeviceRet_ = aclrtSetDevice_(deviceId_);
        if (setDeviceRet_ != ACL_SUCCESS) {
            return;
        }
        createContextRet_ = aclrtCreateContext_(&context_, deviceId_);
    }

    ~AclRuntimeGuard()
    {
        if (context_ != nullptr) {
            (void)aclrtDestroyContext_(context_);
        }
        if (setDeviceRet_ == ACL_SUCCESS) {
            (void)aclrtResetDevice_(deviceId_);
        }
        if (initRet_ == ACL_SUCCESS) {
            (void)aclFinalize_();
        }
        if (handle_ != nullptr) {
            (void)dlclose(handle_);
        }
    }

    aclError InitRet() const
    {
        return initRet_;
    }

    aclError SetDeviceRet() const
    {
        return setDeviceRet_;
    }

    aclError CreateContextRet() const
    {
        return createContextRet_;
    }

    bool Loaded() const
    {
        return aclInit_ != nullptr && aclFinalize_ != nullptr && aclrtSetDevice_ != nullptr &&
            aclrtResetDevice_ != nullptr && aclrtCreateContext_ != nullptr && aclrtDestroyContext_ != nullptr;
    }

private:
    using AclInitFunc = aclError (*)(const char *);
    using AclFinalizeFunc = aclError (*)();
    using AclrtSetDeviceFunc = aclError (*)(int32_t);
    using AclrtResetDeviceFunc = aclError (*)(int32_t);
    using AclrtCreateContextFunc = aclError (*)(aclrtContext *, int32_t);
    using AclrtDestroyContextFunc = aclError (*)(aclrtContext);

    void *handle_ = nullptr;
    AclInitFunc aclInit_ = nullptr;
    AclFinalizeFunc aclFinalize_ = nullptr;
    AclrtSetDeviceFunc aclrtSetDevice_ = nullptr;
    AclrtResetDeviceFunc aclrtResetDevice_ = nullptr;
    AclrtCreateContextFunc aclrtCreateContext_ = nullptr;
    AclrtDestroyContextFunc aclrtDestroyContext_ = nullptr;
    int32_t deviceId_ = 0;
    aclrtContext context_ = nullptr;
    aclError initRet_ = ACL_ERROR_INTERNAL_ERROR;
    aclError setDeviceRet_ = ACL_ERROR_INTERNAL_ERROR;
    aclError createContextRet_ = ACL_ERROR_INTERNAL_ERROR;
};
} // namespace

class sparse_flash_attention_opapi_ut : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "sparse_flash_attention_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "sparse_flash_attention_opapi_ut TearDown" << endl;
    }
};

// Missing lse outputs
TEST_F(sparse_flash_attention_opapi_ut, sparse_flash_attention_aclnn_0)
{
    char layoutQ[] = "BSND";
    char layoutKv[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto value = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndices = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto attentionOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGetWorkspaceSize(
        query.get(),
        key.get(),
        value.get(),
        sparseIndices.get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0.0416666666666667,
        64,
        layoutQ,
        layoutKv,
        3,
        INT64_MAX,
        INT64_MAX,
        0,
        true,
        attentionOut.get(),
        nullptr,
        nullptr,
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Holder fallback
TEST_F(sparse_flash_attention_opapi_ut, sparse_flash_attention_aclnn_1)
{
    char layoutQ[] = "BSND";
    char layoutKv[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto value = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndices = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto attentionOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGetWorkspaceSize(
        query.get(),
        nullptr,
        value.get(),
        sparseIndices.get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0.0416666666666667,
        64,
        layoutQ,
        layoutKv,
        3,
        INT64_MAX,
        INT64_MAX,
        0,
        false,
        attentionOut.get(),
        nullptr,
        nullptr,
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(sparse_flash_attention_opapi_ut, sparse_flash_attention_v2_aclnn_a3_not_supported)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    char layoutQ[] = "BSND";
    char layoutKv[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto value = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndices = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto attentionOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionV2GetWorkspaceSize(
        query.get(),
        key.get(),
        value.get(),
        sparseIndices.get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0.0416666666666667,
        64,
        layoutQ,
        layoutKv,
        3,
        INT64_MAX,
        INT64_MAX,
        0,
        false,
        attentionOut.get(),
        nullptr,
        nullptr,
        &workspaceSize,
        &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(sparse_flash_attention_opapi_ut, sparse_flash_attention_v2_aclnn_a5_with_sinks_success)
{
    AclRuntimeGuard aclGuard;
    ASSERT_TRUE(aclGuard.Loaded());
    ASSERT_EQ(aclGuard.InitRet(), ACL_SUCCESS);
    ASSERT_EQ(aclGuard.SetDeviceRet(), ACL_SUCCESS);
    ASSERT_EQ(aclGuard.CreateContextRet(), ACL_SUCCESS);

    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    char layoutQ[] = "TND";
    char layoutKv[] = "TND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2048, 1, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto value = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2048, 1, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndices = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 2048}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto actualSeqLengthsQuery = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto actualSeqLengthsKv = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto queryRope = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto keyRope = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2048, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sinks = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto attentionOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxMax = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto softmaxSum = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionV2GetWorkspaceSize(
        query.get(),
        key.get(),
        value.get(),
        sparseIndices.get(),
        nullptr,
        actualSeqLengthsQuery.get(),
        actualSeqLengthsKv.get(),
        queryRope.get(),
        keyRope.get(),
        sinks.get(),
        0.044194173824159216,
        1,
        layoutQ,
        layoutKv,
        3,
        INT64_MAX,
        INT64_MAX,
        2,
        true,
        attentionOut.get(),
        softmaxMax.get(),
        softmaxSum.get(),
        &workspaceSize,
        &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}
