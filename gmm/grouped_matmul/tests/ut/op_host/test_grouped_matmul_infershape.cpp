/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class GroupedMatmulInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulProto TearDown" << std::endl;
    }
};

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_0)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
        {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //weight
        {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
    }, 
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}