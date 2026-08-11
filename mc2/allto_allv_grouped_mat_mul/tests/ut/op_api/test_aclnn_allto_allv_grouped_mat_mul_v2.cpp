/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../op_api/aclnn_allto_allv_grouped_mat_mul_v2.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include "gtest/gtest.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;

namespace AlltoAllvGroupedMatMulV2UT {
namespace {
constexpr int64_t EP_WORLD_SIZE = 2;
constexpr int64_t LOCAL_EXPERTS = 2;
constexpr int64_t M = 64;
constexpr int64_t K = 256;
constexpr int64_t N = 128;
void *const SHMEM_CONTEXT = reinterpret_cast<void *>(0x1000UL);

class AlltoAllvGroupedMatMulV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }
};

aclnnStatus RunGetWorkspace(const aclTensor *gmmX, const aclTensor *gmmWeight, void *shmemContext,
                            int64_t epWorldSize, const std::vector<int64_t> &sendCountsList,
                            const std::vector<int64_t> &recvCountsList, aclTensor *gmmY)
{
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                        INPUT(gmmX, gmmWeight, shmemContext, epWorldSize, sendCounts, recvCounts), OUTPUT(gmmY));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    return ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
}
} // namespace

TEST_F(AlltoAllvGroupedMatMulV2Test, ValidSourceCompatibleSignature)
{
    TensorDesc gmmX({M, K}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gmmWeight({LOCAL_EXPERTS, K, N}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gmmY({M, N}, ACL_BF16, ACL_FORMAT_ND);
    std::vector<int64_t> counts(EP_WORLD_SIZE * LOCAL_EXPERTS, M / (EP_WORLD_SIZE * LOCAL_EXPERTS));

    EXPECT_NE(RunGetWorkspace(gmmX, gmmWeight, SHMEM_CONTEXT, EP_WORLD_SIZE, counts, counts, gmmY),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AlltoAllvGroupedMatMulV2Test, RejectsNullRequiredParameters)
{
    TensorDesc gmmX({M, K}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gmmWeight({LOCAL_EXPERTS, K, N}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gmmY({M, N}, ACL_BF16, ACL_FORMAT_ND);
    std::vector<int64_t> counts(EP_WORLD_SIZE * LOCAL_EXPERTS, M / (EP_WORLD_SIZE * LOCAL_EXPERTS));

    EXPECT_EQ(RunGetWorkspace(nullptr, gmmWeight, SHMEM_CONTEXT, EP_WORLD_SIZE, counts, counts, gmmY),
              ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(RunGetWorkspace(gmmX, nullptr, SHMEM_CONTEXT, EP_WORLD_SIZE, counts, counts, gmmY),
              ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(RunGetWorkspace(gmmX, gmmWeight, nullptr, EP_WORLD_SIZE, counts, counts, gmmY),
              ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(RunGetWorkspace(gmmX, gmmWeight, SHMEM_CONTEXT, EP_WORLD_SIZE, counts, counts, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AlltoAllvGroupedMatMulV2Test, RejectsWorldSizeOutsideSourceRange)
{
    TensorDesc gmmX({M, K}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gmmWeight({LOCAL_EXPERTS, K, N}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gmmY({M, N}, ACL_BF16, ACL_FORMAT_ND);
    std::vector<int64_t> counts(EP_WORLD_SIZE * LOCAL_EXPERTS, M / (EP_WORLD_SIZE * LOCAL_EXPERTS));

    EXPECT_EQ(RunGetWorkspace(gmmX, gmmWeight, SHMEM_CONTEXT, 0, counts, counts, gmmY),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(RunGetWorkspace(gmmX, gmmWeight, SHMEM_CONTEXT, 9, counts, counts, gmmY),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AlltoAllvGroupedMatMulV2Test, RejectsCountLengthAndNegativeCount)
{
    TensorDesc gmmX({M, K}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gmmWeight({LOCAL_EXPERTS, K, N}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gmmY({M, N}, ACL_BF16, ACL_FORMAT_ND);
    std::vector<int64_t> validCounts(EP_WORLD_SIZE * LOCAL_EXPERTS, M / (EP_WORLD_SIZE * LOCAL_EXPERTS));
    std::vector<int64_t> shortCounts(validCounts.size() - 1, 1);
    std::vector<int64_t> negativeCounts = validCounts;
    negativeCounts[0] = -1;

    EXPECT_EQ(RunGetWorkspace(gmmX, gmmWeight, SHMEM_CONTEXT, EP_WORLD_SIZE, shortCounts, validCounts, gmmY),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(RunGetWorkspace(gmmX, gmmWeight, SHMEM_CONTEXT, EP_WORLD_SIZE, validCounts, shortCounts, gmmY),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(RunGetWorkspace(gmmX, gmmWeight, SHMEM_CONTEXT, EP_WORLD_SIZE, negativeCounts, validCounts, gmmY),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AlltoAllvGroupedMatMulV2Test, RejectsUnsupportedDtypeAndShape)
{
    TensorDesc fp16X({M, K}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bf16X({M, K}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc badRankX({1, M, K}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc weight({LOCAL_EXPERTS, K, N}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc badKWeight({LOCAL_EXPERTS, K / 2, N}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output({M, N}, ACL_BF16, ACL_FORMAT_ND);
    std::vector<int64_t> counts(EP_WORLD_SIZE * LOCAL_EXPERTS, M / (EP_WORLD_SIZE * LOCAL_EXPERTS));

    EXPECT_EQ(RunGetWorkspace(fp16X, weight, SHMEM_CONTEXT, EP_WORLD_SIZE, counts, counts, output),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(RunGetWorkspace(badRankX, weight, SHMEM_CONTEXT, EP_WORLD_SIZE, counts, counts, output),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(RunGetWorkspace(bf16X, badKWeight, SHMEM_CONTEXT, EP_WORLD_SIZE, counts, counts, output),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AlltoAllvGroupedMatMulV2Test, CoversSecondStageApi)
{
    EXPECT_NE(aclnnAlltoAllvGroupedMatMulV2(nullptr, 0, nullptr, nullptr), ACLNN_ERR_PARAM_INVALID);
}
} // namespace AlltoAllvGroupedMatMulV2UT
