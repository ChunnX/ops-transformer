/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file test_grouped_matmul.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/grouped_matmul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "test_grouped_matmul_utils.h"

using namespace std;
using namespace ge;

class GroupedMatmulTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulTiling TearDown" << std::endl;
    }
};

TEST_F(GroupedMatmulTiling, test_tiling_fp16)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
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
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        GMM_TPL_FLOAT16, // D_T_A
        GMM_TPL_FLOAT16, // D_T_B
        GMM_TPL_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}