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
#include "gtest/gtest.h"
#include "opdev/op_log.h"
#include "../../../../op_api/grouped_matmul_finalize_routing_950_checker.h"
#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_v3.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {
using op::DataType;
using TensorPtr = std::unique_ptr<aclTensor, void (*)(aclTensor *)>;

class GmmFinalizeRoutingPlatformDtype : public testing::Test {
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

struct MxInputs {
    TensorPtr x = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    TensorPtr weight = TensorDesc({2, 64, 32}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    TensorPtr scale = TensorDesc({2, 1, 32, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ToAclType();
    TensorPtr perToken = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ToAclType();
    TensorPtr groups = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    TensorPtr logit = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    TensorPtr rows = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    TensorPtr out = TensorDesc({8, 32}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();

    void SetTypes(DataType xType, DataType wType, DataType sType, DataType pType)
    {
        // Metadata-only checker tests: no tensor contents are generated or kernels executed.
        x->SetDataType(xType);
        weight->SetDataType(wType);
        scale->SetDataType(sType);
        perToken->SetDataType(pType);
        const int64_t factor = sType == DataType::DT_HIFLOAT4_SCALE ? 1 : 2;
        scale->SetViewShape({2, 1, 32, factor});
        perToken->SetViewShape({8, 1, factor});
    }

    GmmFinalizeRouting::GroupedMatmulParams Params()
    {
        return GmmFinalizeRouting::GroupedMatmulParamsBuilder::Create(x.get(), weight.get(), out.get())
            .SetScale(scale.get()).SetPertokenScale(perToken.get()).SetGroupList(groups.get())
            .SetLogit(logit.get()).SetRowIndex(rows.get()).Build();
    }

    aclnnStatus Check()
    {
        auto params = Params();
        GmmFinalizeRouting::AclnnGroupedMatmulFinalizeRoutingDAV3510Checker checker;
        return checker.CheckParams(params);
    }
};

TEST_F(GmmFinalizeRoutingPlatformDtype, PlatformAwareMxMembership)
{
    for (auto arch : {NpuArch::DAV_3510, NpuArch::DAV_9201}) {
        op::SetPlatformNpuArch(arch);
        ASSERT_EQ(op::GetCurrentPlatformInfo().GetCurNpuArch(), arch);
        EXPECT_TRUE(GmmFinalizeRouting::IsMxDataTypeSupported(DataType::DT_FLOAT4_E2M1));
        EXPECT_TRUE(GmmFinalizeRouting::IsMxScaleTypeSupported(DataType::DT_FLOAT8_E8M0));
        EXPECT_EQ(GmmFinalizeRouting::IsMxDataTypeSupported(DataType::DT_HIFLOAT4), arch == NpuArch::DAV_9201);
        EXPECT_EQ(GmmFinalizeRouting::IsMxScaleTypeSupported(DataType::DT_HIFLOAT4_SCALE), arch == NpuArch::DAV_9201);
    }
}

TEST_F(GmmFinalizeRoutingPlatformDtype, FullQuantDtypeMatrix)
{
    const DataType dataTypes[] = {DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2,
                                 DataType::DT_FLOAT4_E2M1, DataType::DT_HIFLOAT4};
    const DataType scaleTypes[] = {DataType::DT_FLOAT8_E8M0, DataType::DT_HIFLOAT4_SCALE};
    for (auto arch : {NpuArch::DAV_3510, NpuArch::DAV_9201}) {
        op::SetPlatformNpuArch(arch);
        for (auto x : dataTypes) {
            for (auto w : dataTypes) {
                for (auto scale : scaleTypes) {
                    for (auto perToken : scaleTypes) {
                        SCOPED_TRACE(testing::Message() << static_cast<int>(arch) << ":"
                            << static_cast<int>(x) << ":" << static_cast<int>(w) << ":"
                            << static_cast<int>(scale) << ":" << static_cast<int>(perToken));
                        const bool fp8Pair = (x == dataTypes[0] || x == dataTypes[1]) &&
                                             (w == dataTypes[0] || w == dataTypes[1]);
                        const bool fp4Pair = x == DataType::DT_FLOAT4_E2M1 && w == x;
                        const bool mxValid = (fp8Pair || fp4Pair) && scale == scaleTypes[0] &&
                                             perToken == scaleTypes[0];
                        const bool hif4Valid = arch == NpuArch::DAV_9201 && x == DataType::DT_HIFLOAT4 &&
                                               w == x && scale == scaleTypes[1] && perToken == scaleTypes[1];
                        MxInputs inputs;
                        inputs.SetTypes(x, w, scale, perToken);
                        EXPECT_EQ(inputs.Check(), mxValid || hif4Valid ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID);
                    }
                }
            }
        }
    }
}

TEST_F(GmmFinalizeRoutingPlatformDtype, A5Bf16MxOutputIsPreserved)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    MxInputs inputs;
    inputs.weight = TensorDesc({2, 64, 32}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ,
                               {}, 0, {2, 2, 2, 16, 32}).ToAclType();
    inputs.out->SetDataType(DataType::DT_BF16);
    inputs.rows->SetDataType(DataType::DT_INT32);
    auto params = inputs.Params();
    params.dtype = 2;
    GmmFinalizeRouting::AclnnGroupedMatmulFinalizeRoutingDAV3510Checker checker;
    EXPECT_EQ(checker.CheckParams(params), ACLNN_SUCCESS);
}

TEST_F(GmmFinalizeRoutingPlatformDtype, V3RejectsHif4OnA5)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    MxInputs inputs;
    inputs.SetTypes(DataType::DT_HIFLOAT4, DataType::DT_HIFLOAT4,
                    DataType::DT_HIFLOAT4_SCALE, DataType::DT_HIFLOAT4_SCALE);
    uint64_t workspaceSize = 12345;
    aclOpExecutor *executor = nullptr;
    EXPECT_EQ(aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(
        inputs.x.get(), inputs.weight.get(), inputs.scale.get(), nullptr, nullptr, nullptr, nullptr,
        inputs.perToken.get(), inputs.groups.get(), nullptr, inputs.logit.get(), inputs.rows.get(),
        0, 1.0F, 0, false, false, 0, nullptr, inputs.out.get(), &workspaceSize, &executor),
        ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspaceSize, 12345U);
    EXPECT_EQ(executor, nullptr);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
}
} // namespace
