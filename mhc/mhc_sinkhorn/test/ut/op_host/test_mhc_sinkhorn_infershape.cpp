/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

#define private public
#include "platform/platform_info.h"

class MhcSinkhorn : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcSinkhorn Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcSinkhorn Proto Test TearDown" << std::endl;
    }
};

TEST_F(MhcSinkhorn, MhcSinkhorn_normal_dims4)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara(
                                                      "MhcSinkhorn",
                                                      {
                                                        {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
                                                        {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(num_iters)},
                                                        {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_flag)},
                                                      }
                                                      );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 128, 4, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcSinkhorn, MhcSinkhorn_normal_dims3)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    gert::InfershapeContextPara infershapeContextPara("MhcSinkhorn",
                                                      {
                                                        {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
                                                        {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(num_iters)},
                                                        {"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(out_flag)},
                                                      }
                                                      );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 6, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

