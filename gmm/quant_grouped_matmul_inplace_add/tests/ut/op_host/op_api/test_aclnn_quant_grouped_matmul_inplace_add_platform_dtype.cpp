/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <memory>
#include "gtest/gtest.h"
#include "../../../../op_api/aclnn_quant_grouped_matmul_inplace_add.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {
using op::DataType;
using TensorPtr = std::unique_ptr<aclTensor, void (*)(aclTensor *)>;

class QGmmInplaceAddPlatformDtype : public testing::Test {
protected:
    void SetUp() override
    {
        savedSoc_ = op::GetCurrentPlatformInfo().GetSocVersion();
        savedArch_ = op::GetCurrentPlatformInfo().GetCurNpuArch();
    }
    void TearDown() override
    {
        op::SetPlatformSocVersion(savedSoc_);
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() != savedArch_) {
            op::SetPlatformNpuArch(savedArch_);
        }
    }
private:
    op::SocVersion savedSoc_;
    NpuArch savedArch_;
};

struct Hif4Inputs {
    TensorPtr x = TensorDesc({8, 64}, ACL_UINT8, ACL_FORMAT_ND, {1, 8}, 0, {64, 8}).ToAclType();
    TensorPtr weight = TensorDesc({64, 32}, ACL_UINT8, ACL_FORMAT_ND).ToAclType();
    TensorPtr perToken = TensorDesc({2, 8, 1}, ACL_UINT8, ACL_FORMAT_ND).ToAclType();
    TensorPtr scale = TensorDesc({2, 32, 1}, ACL_UINT8, ACL_FORMAT_ND).ToAclType();
    TensorPtr groups = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).Value(std::vector<int64_t>{64}).ToAclType();
    TensorPtr out = TensorDesc({1, 8, 32}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();

    Hif4Inputs()
    {
        // Descriptor-only API tests; no HIFLOAT4 data generation or kernel execution.
        x->SetDataType(DataType::DT_HIFLOAT4);
        weight->SetDataType(DataType::DT_HIFLOAT4);
        perToken->SetDataType(DataType::DT_HIFLOAT4_SCALE);
        scale->SetDataType(DataType::DT_HIFLOAT4_SCALE);
    }

    aclnnStatus Workspace(bool expectEmpty = false)
    {
        uint64_t workspaceSize = 12345;
        aclOpExecutor *executor = nullptr;
        auto ret = aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(
            x.get(), weight.get(), perToken.get(), scale.get(), groups.get(), out.get(),
            0, 0, &workspaceSize, &executor);
        if (ret != ACLNN_SUCCESS) {
            EXPECT_EQ(workspaceSize, 12345U);
            EXPECT_EQ(executor, nullptr);
        } else if (expectEmpty) {
            EXPECT_EQ(workspaceSize, 0U);
            EXPECT_NE(executor, nullptr);
        }
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
        return ret;
    }
};

TEST_F(QGmmInplaceAddPlatformDtype, Hif4RejectedOnA5)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    Hif4Inputs inputs;
    EXPECT_EQ(inputs.Workspace(), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(QGmmInplaceAddPlatformDtype, Hif4AcceptedOnA6)
{
    op::SetPlatformNpuArch(NpuArch::DAV_9201);
    ASSERT_EQ(op::GetCurrentPlatformInfo().GetCurNpuArch(), NpuArch::DAV_9201);
    Hif4Inputs inputs;
    EXPECT_EQ(inputs.Workspace(), ACLNN_SUCCESS);
}

TEST_F(QGmmInplaceAddPlatformDtype, A6RejectsMismatchedHif4Scales)
{
    op::SetPlatformNpuArch(NpuArch::DAV_9201);
    for (size_t index = 0; index < 2; ++index) {
        SCOPED_TRACE(index);
        Hif4Inputs inputs;
        auto *scale = index == 0 ? inputs.scale.get() : inputs.perToken.get();
        scale->SetDataType(DataType::DT_FLOAT8_E8M0);
        EXPECT_EQ(inputs.Workspace(), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(QGmmInplaceAddPlatformDtype, A5EmptyPathSkipsDtypeChecksAsBefore)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    for (size_t index = 0; index < 4; ++index) {
        SCOPED_TRACE(index);
        Hif4Inputs inputs;
        inputs.x->SetViewShape({0, 64});
        inputs.out->SetViewShape({1, 0, 32});
        const std::array<aclTensor *, 4> tensors = {
            inputs.x.get(), inputs.weight.get(), inputs.perToken.get(), inputs.scale.get()};
        for (auto *tensor : tensors) {
            tensor->SetDataType(DataType::DT_FLOAT8_E4M3FN);
        }
        tensors[index]->SetDataType(index < 2 ? DataType::DT_HIFLOAT4 : DataType::DT_HIFLOAT4_SCALE);
        // The existing empty-tensor fast path intentionally precedes CheckParams/CheckDtype.
        EXPECT_EQ(inputs.Workspace(true), ACLNN_SUCCESS);
    }
}

TEST_F(QGmmInplaceAddPlatformDtype, EmptyHif4PathIsPreservedOnBothPlatforms)
{
    for (auto arch : {NpuArch::DAV_3510, NpuArch::DAV_9201}) {
        op::SetPlatformNpuArch(arch);
        for (bool emptyM : {true, false}) {
            SCOPED_TRACE(testing::Message() << static_cast<int>(arch) << ":emptyM=" << emptyM);
            Hif4Inputs inputs;
            if (emptyM) {
                inputs.x->SetViewShape({0, 64});
                inputs.out->SetViewShape({1, 0, 32});
            } else {
                inputs.weight->SetViewShape({64, 0});
                inputs.out->SetViewShape({1, 8, 0});
            }
            EXPECT_EQ(inputs.Workspace(true), ACLNN_SUCCESS);
        }
    }
}
} // namespace
