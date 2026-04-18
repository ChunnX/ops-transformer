/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/mhc_sinkhorn_backward_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MhcSinkhornBackwardTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcSinkhornBackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcSinkhornBackwardTiling TearDown" << std::endl;
    }
};

struct MhcSinkhornBackwardCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

// 测试4D输入（BSNN布局）+ n=4 + FLOAT32数据类型的正常Tiling计算场景
// 输入：grad_y=[1, 128, 4, 4], norm=[81920], sum=[20480]
// 输出：grad_input=[1, 128, 4, 4]
// totalLength = B*S = 1*128 = 128
// nAlignSize = 8
// numIters = 10
// norm.dim0 = 2 * numIters * totalLength * nSize * nAlignSize = 2 * 10 * 128 * 4 * 8 = 81920
// sum.dim0 = 2 * numIters * totalLength * nAlignSize = 2 * 10 * 128 * 8 = 20480
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_4d_fp32_success)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{81920}, {81920}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{20480}, {20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = ""; 
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试3D输入（TNN布局）+ n=6 + FLOAT32数据类型的正常Tiling计算场景
// 输入：grad_y=[1024, 6, 6], norm=[1474560], sum=[245760]
// 输出：grad_input=[1024, 6, 6]
// totalLength = T = 1024
// nAlignSize = 8
// numIters = 15
// norm.dim0 = 2 * numIters * totalLength * nSize * nAlignSize = 2 * 15 * 1024 * 6 * 8 = 1474560
// sum.dim0 = 2 * numIters * * totalLength * nAlignSize = 2 * 15 * 1024 * 8 = 245760
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_3d_fp32_success)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1474560}, {1474560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{245760}, {245760}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试4D输入（BSNN布局）+ n=8 + FLOAT32数据类型的正常Tiling计算场景
// 输入：grad_y=[2, 256, 8, 8], norm=[1638400], sum=[204800]
// 输出：grad_input=[2, 256, 8, 8]
// totalLength = B*S = 2*256 = 512
// nAlignSize = 8
// numIters = 25
// norm.dim0 = 2 * numIters * totalLength * nSize * nAlignSize = 2 * 25 * 512 * 8 * 8 = 1638400
// sum.dim0 = 2 * numIters * totalLength * nAlignSize = 2 * 25 * 512 * 8 = 204800
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_n8_4d_success)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{2, 256, 8, 8}, {2, 256, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1638400}, {1638400}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{204800}, {204800}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{2, 256, 8, 8}, {2, 256, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试数据类型校验：grad_y使用FLOAT16数据类型，应返回GRAPH_FAILED
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_dtype)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{163840}, {163840}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40960}, {40960}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试迭代次数校验：numIters=101超出范围[1,100]，应返回GRAPH_FAILED
// norm.dim0 = 829440, totalLength=128, nSize=4, nAlignSize=8
// numIters = 829440 / (2 * 128 * 4 * 8) = 101
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{829440}, {829440}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{207360}, {207360}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试迭代次数奇偶校验：norm的dim0=81921为奇数，应返回GRAPH_FAILED
// norm.dim0 = 81921, totalLength=128, nSize=4, nAlignSize=8
// numIters = 81921 / (2 * 128 * 4 * 8) = 10.00 (不是整数，校验失败)
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_odd_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{81921}, {81921}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{20480}, {20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试grad_y维度校验：grad_y为5D张量（非3D/4D），应返回GRAPH_FAILED
// totalLength = B*S = 1*2*128 = 256
// norm.dim0 = 2 * numIters * totalLength * nSize * nAlignSize = 2 * 10 * 256 * 4 * 8 = 163840
// sum.dim0 = 2 * numIters * totalLength * nAlignSize = 2 * 10 * 256 * 8 = 40960
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_grad_y_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum (invalid 5D shape)
            {{{1, 2, 128, 4, 4}, {1, 2, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{163840}, {163840}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40960}, {40960}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 2, 128, 4, 4}, {1, 2, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试输出shape校验：grad_input的shape与grad_y的shape不匹配，应返回GRAPH_FAILED
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_output_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{81920}, {81920}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{20480}, {20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input (invalid shape)
            {{{1, 128, 6, 6}, {1, 128, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试norm shape校验：norm的dim0应该等于2*numIters*totalLength*nSize*nAlignSize，应返回GRAPH_FAILED
// norm.dim0 = 81919, totalLength=128, nSize=4, nAlignSize=8
// numIters = 81919 / (2 * 128 * 4 * 8) = 10.00 (接近但不等于，校验失败)
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_norm_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{81919}, {81919}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{20480}, {20480}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试sum shape校验：sum的dim0应该等于2*numIters*totalLength*nAlignSize，应返回GRAPH_FAILED
// sum.dim0 = 20479, totalLength=128, nAlignSize=8
// numIters = 20479 / (2 * 128 * 8) = 10.00 (接近但不等于，校验失败)
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_sum_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{81920}, {81920}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{20479}, {20479}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试4D输入大batch场景：grad_y=[4, 256, 4, 4], norm=[655360], sum=[163840]
// totalLength = B*S = 4*256 = 1024
// numIters = 10
// nAlignSize = 8
// norm.dim0 = 2 * numIters * totalLength * nSize * nAlignSize = 2 * 10 * 1024 * 4 * 8 = 655360
// sum.dim0 = 2 * numIters * totalLength * nAlignSize = 2 * 10 * 1024 * 8 = 163840
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_4d_large_batch)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{4, 256, 4, 4}, {4, 256, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{655360}, {655360}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{163840}, {163840}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{4, 256, 4, 4}, {4, 256, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试3D输入大T场景：grad_y=[4096, 8, 8], norm=[13107200], sum=[1638400]
// totalLength = T = 4096
// numIters = 25
// nAlignSize = 8
// norm.dim0 = 2 * numIters * totalLength * nSize * nAlignSize = 2 * 25 * 4096 * 8 * 8 = 13107200
// sum.dim0 = 2 * numIters * totalLength * nAlignSize = 2 * 25 * 4096 * 8 = 1638400
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_3d_large_t)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{4096, 8, 8}, {4096, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{13107200}, {13107200}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1638400}, {1638400}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{4096, 8, 8}, {4096, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试最小迭代次数场景：norm的dim0=4096，numIters=1
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_min_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试最大迭代次数场景：norm的dim0=409600，numIters=100
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_max_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{409600}, {409600}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{102400}, {102400}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试3D输入n=4场景：grad_y=[512, 4, 4], norm=[327680], sum=[81920]
// totalLength = T = 512
// numIters = 10
// nAlignSize = 8
// norm.dim0 = 2 * numIters * totalLength * nSize * nAlignSize = 2 * 10 * 512 * 4 * 8 = 327680
// sum.dim0 = 2 * numIters * totalLength * nAlignSize = 2 * 10 * 512 * 8 = 81920
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_3d_n4)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{512, 4, 4}, {512, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{327680}, {327680}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{81920}, {81920}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{512, 4, 4}, {512, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}

// 测试4D输入n=6场景：grad_y=[2, 256, 6, 6], norm=[737280], sum=[122880]
// totalLength = B*S = 2*256 = 512
// numIters = 15
// nAlignSize = 8
// norm.dim0 = 2 * numIters * totalLength * nSize * nAlignSize = 2 * 15 * 512 * 6 * 8 = 737280
// sum.dim0 = 2 * numIters * totalLength * nAlignSize = 2 * 15 * 512 * 8 = 122880
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_4d_n6)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{2, 256, 6, 6}, {2, 256, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{737280}, {737280}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{122880}, {122880}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{2, 256, 6, 6}, {2, 256, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTarsDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTarsDataStr, expectWorkspaces);
}
