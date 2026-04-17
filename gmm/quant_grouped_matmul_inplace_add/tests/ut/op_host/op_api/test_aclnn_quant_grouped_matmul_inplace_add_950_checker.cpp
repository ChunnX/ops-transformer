/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/quant_grouped_matmul_inplace_add_950_checker.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {

struct CheckerInputPack {
    std::unique_ptr<aclTensor, decltype(&Release)> x;
    std::unique_ptr<aclTensor, decltype(&Release)> weight;
    std::unique_ptr<aclTensor, decltype(&Release)> scale;
    std::unique_ptr<aclTensor, decltype(&Release)> perTokenScale;
    std::unique_ptr<aclTensor, decltype(&Release)> groupList;
    std::unique_ptr<aclTensor, decltype(&Release)> y;
    std::unique_ptr<aclTensorList, decltype(&Release)> offset;
    gmm::GroupedMatmulParamsBase<aclTensor> params;

    CheckerInputPack()
        : x(nullptr, Release),
          weight(nullptr, Release),
          scale(nullptr, Release),
          perTokenScale(nullptr, Release),
          groupList(nullptr, Release),
          y(nullptr, Release),
          offset(nullptr, Release)
    {}
};

CheckerInputPack BuildBasePack()
{
    CheckerInputPack pack;
    pack.x.reset(InferAclType(TensorDesc({96, 512}, ACL_HIFLOAT8, ACL_FORMAT_ND)));
    pack.weight.reset(InferAclType(TensorDesc({512, 128}, ACL_HIFLOAT8, ACL_FORMAT_ND)));
    pack.scale.reset(InferAclType(TensorDesc({4, 128}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.perTokenScale.reset(InferAclType(TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.groupList.reset(
        InferAclType(TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Value(std::vector<int64_t>{128, 256, 300, 512})));
    pack.y.reset(InferAclType(TensorDesc({4, 96, 128}, ACL_FLOAT, ACL_FORMAT_ND)));

    pack.params.x = pack.x.get();
    pack.params.weight = pack.weight.get();
    pack.params.scaleOptional = pack.scale.get();
    pack.params.perTokenScaleOptional = pack.perTokenScale.get();
    pack.params.groupTensorOptional = pack.groupList.get();
    pack.params.y = pack.y.get();
    pack.params.xDtype = op::DataType::DT_HIFLOAT8;
    pack.params.groupType = gmm::SPLIT_K;
    return pack;
}

aclnnStatus RunChecker(const gmm::GroupedMatmulParamsBase<aclTensor> &params)
{
    QGmmInPlaceAdd::AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<aclTensor> checker(params);
    checker.SetInputName("x1", "x2", "scale1Optional", "scale2", "groupList");
    return checker.CheckQuantGroupedMatmulInplaceAddDAV3510();
}

TEST(aclnn_qgmm_inplace_add_950_checker, valid_hifloat8_pass)
{
    auto pack = BuildBasePack();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_SUCCESS);
}

TEST(aclnn_qgmm_inplace_add_950_checker, group_list_over_1024_fail)
{
    auto pack = BuildBasePack();
    pack.groupList.reset(InferAclType(TensorDesc({1025}, ACL_INT64, ACL_FORMAT_ND)));
    pack.scale.reset(InferAclType(TensorDesc({1025, 128}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.perTokenScale.reset(InferAclType(TensorDesc({1025}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.y.reset(InferAclType(TensorDesc({1025, 96, 128}, ACL_FLOAT, ACL_FORMAT_ND)));

    pack.params.groupTensorOptional = pack.groupList.get();
    pack.params.scaleOptional = pack.scale.get();
    pack.params.perTokenScaleOptional = pack.perTokenScale.get();
    pack.params.y = pack.y.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, per_token_scale_null_fail)
{
    auto pack = BuildBasePack();
    pack.perTokenScale.reset(nullptr);
    pack.params.perTokenScaleOptional = nullptr;
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(aclnn_qgmm_inplace_add_950_checker, scale_null_fail)
{
    auto pack = BuildBasePack();
    pack.scale.reset(nullptr);
    pack.params.scaleOptional = nullptr;
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(aclnn_qgmm_inplace_add_950_checker, group_tensor_null_fail)
{
    auto pack = BuildBasePack();
    pack.groupList.reset(nullptr);
    pack.params.groupTensorOptional = nullptr;
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(aclnn_qgmm_inplace_add_950_checker, offset_not_null_fail)
{
    auto pack = BuildBasePack();
    pack.offset.reset(InferAclType(TensorListDesc(1, TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND))));
    pack.params.offsetOptional = pack.offset.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_x_format_fail)
{
    auto pack = BuildBasePack();
    pack.x.reset(InferAclType(TensorDesc({96, 512}, ACL_HIFLOAT8, ACL_FORMAT_FRACTAL_NZ)));
    pack.params.x = pack.x.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_y_dimnum_fail)
{
    auto pack = BuildBasePack();
    pack.y.reset(InferAclType(TensorDesc({4, 96}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.params.y = pack.y.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_y_group_dim_fail)
{
    auto pack = BuildBasePack();
    pack.y.reset(InferAclType(TensorDesc({5, 96, 128}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.params.y = pack.y.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_y_m_dim_fail)
{
    auto pack = BuildBasePack();
    pack.y.reset(InferAclType(TensorDesc({4, 95, 128}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.params.y = pack.y.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_y_n_dim_fail)
{
    auto pack = BuildBasePack();
    pack.y.reset(InferAclType(TensorDesc({4, 96, 127}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.params.y = pack.y.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_per_token_dtype_fail)
{
    auto pack = BuildBasePack();
    pack.perTokenScale.reset(InferAclType(TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND)));
    pack.params.perTokenScaleOptional = pack.perTokenScale.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_y_dtype_fail)
{
    auto pack = BuildBasePack();
    pack.y.reset(InferAclType(TensorDesc({4, 96, 128}, ACL_FLOAT16, ACL_FORMAT_ND)));
    pack.params.y = pack.y.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_scale_dim_fail)
{
    auto pack = BuildBasePack();
    pack.scale.reset(InferAclType(TensorDesc({4, 128, 1}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.params.scaleOptional = pack.scale.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_scale_g_dim_fail)
{
    auto pack = BuildBasePack();
    pack.scale.reset(InferAclType(TensorDesc({3, 128}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.params.scaleOptional = pack.scale.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_scale_n_dim_fail)
{
    auto pack = BuildBasePack();
    pack.scale.reset(InferAclType(TensorDesc({4, 127}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.params.scaleOptional = pack.scale.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, invalid_per_token_shape_dim2_fail)
{
    auto pack = BuildBasePack();
    pack.perTokenScale.reset(InferAclType(TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND)));
    pack.params.perTokenScaleOptional = pack.perTokenScale.get();
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

TEST(aclnn_qgmm_inplace_add_950_checker, unsupported_dtype_pair_fail)
{
    auto pack = BuildBasePack();
    pack.x.reset(InferAclType(TensorDesc({96, 512}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND)));
    pack.weight.reset(InferAclType(TensorDesc({512, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND)));
    pack.scale.reset(InferAclType(TensorDesc({12, 128, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));
    pack.perTokenScale.reset(InferAclType(TensorDesc({96, 12, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));

    pack.params.x = pack.x.get();
    pack.params.weight = pack.weight.get();
    pack.params.scaleOptional = pack.scale.get();
    pack.params.perTokenScaleOptional = pack.perTokenScale.get();
    pack.params.xDtype = op::DataType::DT_FLOAT8_E5M2;
    EXPECT_EQ(RunChecker(pack.params), ACLNN_ERR_PARAM_INVALID);
}

} // namespace
