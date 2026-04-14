/**
聽* Copyright (c) 2025 Huawei Technologies Co., Ltd.
聽* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
聽* CANN Open Software License Agreement Version 2.0 (the "License").
聽* Please refer to the License for details. You may not use this file except in compliance with the License.
聽* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
聽* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
聽* See LICENSE in the root of the software repository for the full text of the License.
聽*/


/*!
 * \file test_grouped_matmul.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

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

// ANTIQUANT_A8W4
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_autotiling_1aic2aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096, 1})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "1 0 ";
    std::vector<size_t> expectWorkspaces = {67108864}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,250);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_autotiling_1aic2aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},  //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
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
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096, 1})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "1 0 ";
    std::vector<size_t> expectWorkspaces = {67108864}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,250);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_pergroup_antiqunt_1aic2aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERGROUP_ANTIQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {121634816}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_pergroup_antiqunt_1aic2aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},  //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
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
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERGROUP_ANTIQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {121634816}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_perchannel_antiqunt_1aic1aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERCHANNEL_ANTIQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33562624}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_perchannel_antiqunt_1aic1aiv)
{
    size_t M = 8192;
    size_t K = 2048;
    size_t N = 1024;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
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
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4096})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERCHANNEL_ANTIQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33562624}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_msd_vec_1aic2aiv)
{    
    size_t M = 1024;
    size_t K = 512;
    size_t N = 32768;
    size_t E = 256;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_VECTOR_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_msd_vec_1aic2aiv)
{
    size_t M = 1024;
    size_t K = 512;
    size_t N = 32768;
    size_t E = 256;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
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
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_VECTOR_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_msd_api_1aic2aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117440512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4obf16_msd_api_1aic1aiv)
{
    size_t M = 128;
    size_t K = 2048;
    size_t N = 1024;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117440512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_msd_api_1aic2aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
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
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117440512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_msd_api_1aic1aiv)
{
    size_t M = 128;
    size_t K = 2048;
    size_t N = 1024;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
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
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117440512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w4ofp16_msd_api_withoffset_1aic2aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                 //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},           //weight
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_UINT64, ge::FORMAT_ND},          //scale
                                                    {{{E, 1, N}, {E, 1, N}}, ge::DT_FLOAT, ge::FORMAT_ND},          //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                      //perTokenScale
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
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {117440512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// ANTIQUANT_A16W4
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_perchannel_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {83886080}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_perchannel_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {83886080}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_pergroup_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {83886080}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_pergroup_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {83886080}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_perchannel_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_perchannel_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_pergroup_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_pergroup_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_perchannel_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_perchannel_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4ofp16_pergroup_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w4obf16_pergroup_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, 256, N}, {E, 256, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}


// ANTIQUANT_A16W8
TEST_F(GroupedMatmulTiling, test_tiling_a16w8ofp16_antiquant_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
        DT_FLOAT16, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8obf16_antiquant_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_BF16, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {75497472}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8ofp16_msd_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
        DT_FLOAT16, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23134208}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8obf16_msd_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_BF16, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23134208}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8ofp16_antiquant_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
        DT_FLOAT16, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8obf16_antiquant_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_BF16, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a16w8ofp16_msd_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {27328512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8obf16_msd_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {27328512}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8ofp16_antiquant_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a16w8obf16_antiquant_transw_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// QUANT_A8W8O16
TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_notrans_1aic2aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8obf16_notrans_1aic2aiv)
{
    size_t M = 4096;
    size_t K = 2048;
    size_t N = 8192;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({256})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8obf16_notrans_1aic1aiv)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_notrans_1aic1aiv_static)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8obf16_notrans_1aic1aiv_static)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_notrans_1aic1aiv_sparse)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
TEST_F(GroupedMatmulTiling, test_tiling_a8w8obf16_notrans_1aic1aiv_sparse)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}



// QUANT_A8W8O16瀹氳酱绠楁硶
TEST_F(GroupedMatmulTiling, test_tiling_a8w8ofp16_fixed_axis)
{
    size_t M = 345;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({256, 0, -1})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        true //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {26669056}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}


// QUANT_A4W4
TEST_F(GroupedMatmulTiling, test_tiling_a4w4ofp16_notrans_1aic2aiv)
{
    size_t M = 350;
    size_t K = 1280;
    size_t N = 580;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
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
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4obf16_notrans_1aic2aiv)
{
    size_t M = 350;
    size_t K = 1280;
    size_t N = 580;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4ofp16_trans_1aic2aiv)
{
    size_t M = 350;
    size_t K = 1280;
    size_t N = 640;//must align to 64
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K/64, N/16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4obf16_trans_1aic2aiv)
{
    size_t M = 350;
    size_t K = 1280;
    size_t N = 640;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K/64, N/16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4ofp16_trans_dynamic_tiling_1aic2aiv)
{
    size_t M = 8000;
    size_t K = 1024;
    size_t N = 2048;//must align to 64
    size_t E = 80;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K/64, N/16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a4w4obf16_trans_dynamic_tiling_1aic2aiv)
{
    size_t M = 8000;
    size_t K = 1024;
    size_t N = 2048;
    size_t E = 80;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K/64, N/16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// QUANT_A4W4浼樺寲
TEST_F(GroupedMatmulTiling, test_tiling_a4w4ofp16_optimize)
{
    size_t M = 256;
    size_t K = 2048;
    size_t N = 7168;
    size_t E = 4;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT4, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K/64, N/16, 16, 64}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
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
        DT_INT4, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_2, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23068672}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// QUANT_A8W8O8
TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_notrans_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
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
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_transw_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_notrans_static_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                     //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
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
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_transw_static_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        1, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}


TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_notrans_sparse_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_transw_sparse_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT8, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {29360128}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}




// QUANT_A8W8O32
TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_notrans_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
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
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT32, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_transw_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT32, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_notrans_1aic_sparse)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT32, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_transw_1aic_sparse)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_INT32, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_INT8, // D_T_A
        DT_INT8, // D_T_B
        DT_INT32, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// 闈為噺鍖栧満鏅敤锟?
TEST_F(GroupedMatmulTiling, test_tiling_fp16_notrans_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
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
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_bf16_notrans_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_BF16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
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
        DT_BF16, // D_T_A
        DT_BF16, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_fp32_notrans_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND}
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
        DT_FLOAT, // D_T_A
        DT_FLOAT, // D_T_B
        DT_FLOAT, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_fp16_transw_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
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
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_bf16_transw_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_BF16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_BF16, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_fp32_transw_1aic)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT, // D_T_A
        DT_FLOAT, // D_T_B
        DT_FLOAT, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_fp16_transx_1aic1aiv)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        GMM_TPL_FLOAT16, // D_T_A
        GMM_TPL_FLOAT16, // D_T_B
        GMM_TPL_FLOAT16, // D_T_Y
        1, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_bf16_transx_1aic1aiv)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_BF16, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_BF16, // D_T_B
        DT_BF16, // D_T_Y
        1, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_fp32_transx_1aic1aiv)
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT, // D_T_A
        DT_FLOAT, // D_T_B
        DT_FLOAT, // D_T_Y
        1, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

namespace {

string GetExeDirPath()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    auto len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    string exePath(path, len);
    auto pos = exePath.find_last_of("\\/");
    return pos == string::npos ? string(".\\") : exePath.substr(0, pos + 1);
#else
    char path[4096] = {0};
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) {
        return "./";
    }
    path[len] = '\0';
    string exePath(path);
    auto pos = exePath.find_last_of('/');
    return pos == string::npos ? string("./") : exePath.substr(0, pos + 1);
#endif
}

void SplitStr2Vec(const string &input, const string &delimiter, vector<string> &output)
{
    const auto delimiterLen = delimiter.size();
    string::size_type currPos = 0;
    string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }
    if (currPos <= input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}

string Trim(string value)
{
    auto isNotSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), isNotSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), isNotSpace).base(), value.end());
    return value;
}

bool ParseBool(const string &value)
{
    string lower = value;
    transform(lower.begin(), lower.end(), lower.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return lower == "true" || lower == "1" || lower == "yes";
}

ge::DataType ParseDtype(const string &dtype)
{
    static const std::map<string, ge::DataType> dtypeMap = {
        {"FLOAT", ge::DT_FLOAT},
        {"FLOAT16", ge::DT_FLOAT16},
        {"BF16", ge::DT_BF16},
        {"INT8", ge::DT_INT8},
        {"INT32", ge::DT_INT32},
        {"INT64", ge::DT_INT64},
        {"UINT64", ge::DT_UINT64},
        {"INT4", ge::DT_INT4},
        {"UNDEFINED", ge::DT_UNDEFINED},
    };
    auto it = dtypeMap.find(dtype);
    return it == dtypeMap.end() ? ge::DT_UNDEFINED : it->second;
}

ge::Format ParseFormat(const string &format)
{
    static const std::map<string, ge::Format> formatMap = {
        {"ND", ge::FORMAT_ND},
        {"FRACTAL_NZ", ge::FORMAT_FRACTAL_NZ},
    };
    auto it = formatMap.find(format);
    return it == formatMap.end() ? ge::FORMAT_ND : it->second;
}

vector<int64_t> ParseDims(const string &value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> dimTokens;
    SplitStr2Vec(trimmed, ":", dimTokens);
    vector<int64_t> dims;
    for (const auto &token : dimTokens) {
        dims.emplace_back(stoll(Trim(token)));
    }
    return dims;
}

gert::Shape BuildShape(const vector<int64_t> &dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    return shape;
}

gert::StorageShape MakeStorageShape(const vector<int64_t> &originDims, const vector<int64_t> &storageDims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape() = BuildShape(originDims);
    shape.MutableStorageShape() = BuildShape(storageDims);
    return shape;
}

struct GroupedMatmulAscend950TensorDesc {
    vector<int64_t> originDims;
    vector<int64_t> storageDims;
    ge::DataType dtype = ge::DT_UNDEFINED;
    ge::Format format = ge::FORMAT_ND;
};

enum class GroupedMatmulAscend950ExpectType {
    TILING_KEY,
    GRAPH_FAILED,
};

struct GroupedMatmulAscend950Case {
    string caseName;
    bool enable = true;
    string prefix;
    GroupedMatmulAscend950ExpectType expectType = GroupedMatmulAscend950ExpectType::TILING_KEY;
    int64_t expectTilingKey = 0;
    uint32_t aicNum = 24;
    uint32_t aivNum = 48;
    uint64_t ubSize = 196608;
    uint64_t l1Size = 524288;
    uint64_t l2Size = 196608;
    uint64_t l0CSize = 131072;
    uint64_t l0ASize = 65536;
    uint64_t l0BSize = 65536;
    int64_t splitItem = 2;
    int64_t dtypeAttr = 0;
    bool transposeWeight = false;
    bool transposeX = false;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    int64_t tuningConfig = 0;
    vector<GroupedMatmulAscend950TensorDesc> inputs;
    GroupedMatmulAscend950TensorDesc output;

    void Run() const
    {
        optiling::GMMCompileInfo compileInfo = {
            aicNum,
            aivNum,
            ubSize,
            l1Size,
            l2Size,
            l0CSize,
            l0ASize,
            l0BSize,
            platform_ascendc::SocVersion::ASCEND950,
            NpuArch::DAV_3510,
        };
        gert::TilingContextPara tilingContextPara(
            "GroupedMatmul", BuildTensorDescs(inputs), {BuildTensorDesc(output)},
            {
                {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(splitItem)},
                {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtypeAttr)},
                {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
                {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
                {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
                {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(actType)},
                {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({tuningConfig})},
            },
            &compileInfo);

        if (expectType == GroupedMatmulAscend950ExpectType::GRAPH_FAILED) {
            ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
            return;
        }

        TilingInfo tilingInfo;
        ExecuteTiling(tilingContextPara, tilingInfo);
        EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    }

private:
    static gert::TilingContextPara::TensorDescription BuildTensorDesc(const GroupedMatmulAscend950TensorDesc &tensor)
    {
        return {MakeStorageShape(tensor.originDims, tensor.storageDims), tensor.dtype, tensor.format};
    }

    static vector<gert::TilingContextPara::TensorDescription> BuildTensorDescs(
        const vector<GroupedMatmulAscend950TensorDesc> &tensors)
    {
        vector<gert::TilingContextPara::TensorDescription> descs;
        descs.reserve(tensors.size());
        for (const auto &tensor : tensors) {
            descs.emplace_back(BuildTensorDesc(tensor));
        }
        return descs;
    }
};

vector<GroupedMatmulAscend950Case> LoadGroupedMatmulAscend950Cases()
{
    vector<GroupedMatmulAscend950Case> cases;
    string rootPath(GetExeDirPath() + "../../../../../");
    string csvPath(rootPath + "gmm/grouped_matmul/tests/ut/op_host/test_grouped_matmu_tiling.csv");
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << endl;
        return cases;
    }

    string line;
    while (getline(csvData, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "caseName" || items.size() < 61U) {
            continue;
        }

        size_t idx = 0;
        GroupedMatmulAscend950Case tc;
        tc.caseName = Trim(items[idx++]);
        tc.enable = ParseBool(Trim(items[idx++]));
        if (!tc.enable) {
            continue;
        }
        tc.prefix = Trim(items[idx++]);
        tc.expectType = ParseBool(Trim(items[idx++])) ? GroupedMatmulAscend950ExpectType::TILING_KEY
                                                      : GroupedMatmulAscend950ExpectType::GRAPH_FAILED;
        tc.expectTilingKey = stoll(Trim(items[idx++]));
        tc.aicNum = static_cast<uint32_t>(stoul(Trim(items[idx++])));
        tc.aivNum = static_cast<uint32_t>(stoul(Trim(items[idx++])));
        tc.ubSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l1Size = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l2Size = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l0CSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l0ASize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.l0BSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
        tc.splitItem = stoll(Trim(items[idx++]));
        tc.dtypeAttr = stoll(Trim(items[idx++]));
        tc.transposeWeight = ParseBool(Trim(items[idx++]));
        tc.transposeX = ParseBool(Trim(items[idx++]));
        tc.groupType = stoll(Trim(items[idx++]));
        tc.groupListType = stoll(Trim(items[idx++]));
        tc.actType = stoll(Trim(items[idx++]));
        tc.tuningConfig = stoll(Trim(items[idx++]));

        for (int tensorIdx = 0; tensorIdx < 9; ++tensorIdx) {
            tc.inputs.push_back({ParseDims(Trim(items[idx++])), ParseDims(Trim(items[idx++])),
                                 ParseDtype(Trim(items[idx++])), ParseFormat(Trim(items[idx++]))});
        }
        tc.output = {ParseDims(Trim(items[idx++])), ParseDims(Trim(items[idx++])),
                     ParseDtype(Trim(items[idx++])), ParseFormat(Trim(items[idx++]))};
        cases.push_back(tc);
    }
    return cases;
}

string MakeGroupedMatmulAscend950CaseName(const testing::TestParamInfo<GroupedMatmulAscend950Case> &info)
{
    string name = info.param.prefix;
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c) { return isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}

} // namespace

class GroupedMatmulAscend950Tiling : public testing::TestWithParam<GroupedMatmulAscend950Case> {
};

TEST_P(GroupedMatmulAscend950Tiling, generalTest)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GROUPED_MATMUL_ASCEND950, GroupedMatmulAscend950Tiling,
                         testing::ValuesIn(LoadGroupedMatmulAscend950Cases()),
                         MakeGroupedMatmulAscend950CaseName);
//鍏朵綑ut锛堝巻鍙插嚭杩囬棶棰樼殑琛ュ厖ut锟?

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
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
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        GMM_TPL_INT8, // D_T_A
        GMM_TPL_INT8, // D_T_B
        GMM_TPL_INT8, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_CUBE_ONLY, // AIV_AIC_RATIO
        false //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {18350080}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

TEST_F(GroupedMatmulTiling, test_tiling_illegal_group_num_size)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 1205;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_weightnz_tc_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 16L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_weightnz_kc_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 16L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_tc_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_kc_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_illegal_tt_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{E}, {E}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_illegal_acttype_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_illegal_xdtype_case)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_activation_tc_case_n_equal_1)
{
    size_t M = 190;
    size_t K = 15;
    size_t N = 1;
    size_t E = 1;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},         //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_kerneltype_1_corenum_check_fail_nonzero)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        50,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 16L;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_kerneltype_1_corenum_check_fail_zero)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        0,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 16L;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8_3510)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 1L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8_sparse_2d_grouplist_3510)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E, 2}, {E, 2}}, ge::DT_INT64, ge::FORMAT_ND},                //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 1L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_3510)
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
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{E, N}, {E, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
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
    int64_t expectTilingKey = 0L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_weightnz_notrans)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{E, N}, {E, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 0L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_weightnz_sparse_2d_grouplist)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{E, N}, {E, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E, 2}, {E, 2}}, ge::DT_INT64, ge::FORMAT_ND},                //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = 0L;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8_sparse_1d_grouplist_3510_error)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_weightnz_sparse_1d_grouplist_error)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{E, N}, {E, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8_sparse_3d_grouplist_3510_error)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E, 2, 2}, {E, 2, 2}}, ge::DT_INT64, ge::FORMAT_ND},          //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                },
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                },
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8_3510_bias_not_int32_error)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o32_3510_bias_not_int32_error)
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
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
                                                    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT32, ge::FORMAT_ND}
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
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_A8W8O8_3510_scale_not_perchannel_error)
{
    size_t M = 8;
    size_t K = 4096;
    size_t N = 1792;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, N, K}, {E, K, N}}, ge::DT_INT8, ge::FORMAT_ND},        //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, 1}, {E, 1}}, ge::DT_UINT64, ge::FORMAT_ND},               //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w8o8_weightnz_error)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 4;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND950,//ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},              //x
                                                    {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},   //weight
                                                    {{{M, N}, {M, N}}, ge::DT_INT32, ge::FORMAT_ND},                //bias
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                        //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_INT8, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulTiling, test_tiling_a8w4_950_aic_aiv_invalid_ratio)
{
    size_t M = 512;
    size_t K = 1024;
    size_t N = 1024;
    size_t E = 2;
    size_t quantGroupNum = K / 256;

    optiling::GMMCompileInfo compileInfo = {
        24,                                      // aicNum
        24,                                      // aivNum
        196608,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        131072,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                     // x
            {{{E, K, N}, {E, N/32, K/16, 16, 32}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},         // weight
            {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // bias
            {{{E, N}, {E, N}}, ge::DT_UINT64, ge::FORMAT_ND},                                   // scale  
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                            // offset
            {{{E, quantGroupNum, N}, {E, quantGroupNum, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // antiquantScale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                          // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                                          // groupList
            {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // perTokenScale
        },
        {// output info
         {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({256})},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}


