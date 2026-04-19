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
// 输入：grad_y=[1, 128, 4, 4], norm=[40, 2048], sum=[40, 512]
// 输出：grad_input=[1, 128, 4, 4]
// totalLength = B*S = 1*128 = 128
// numIters = norm.dim0 / 2 = 40 / 2 = 20
// nAlignSize = 16 (nSize=4, inputNumPerBlock=16)
// norm.dim1 = totalLength * nSize * nAlign = 128 * 4 * 16 = 8192
// sum.dim1 = totalLength * nAlign = 128 * 16 = 2048
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_4d_fp32_success)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 8192}, {40, 8192}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 2048}, {40, 2048}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "128 0 1 128 4 16 20 1 128 1 1 128 0 1 128 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试3D输入（TNN布局）+ n=6 + FLOAT32数据类型的正常Tiling计算场景
// 输入：grad_y=[1024, 6, 6], norm=[30, 36864], sum=[30, 6144]
// 输出：grad_input=[1024, 6, 6]
// totalLength = T = 1024
// numIters = norm.dim0 / 2 = 30 / 2 = 15
// nAlignSize = 8 (nSize=6, inputNumPerBlock=16)
// norm.dim1 = totalLength * nSize * nAlign = 1024 * 6 * 8 = 49152
// sum.dim1 = totalLength * nAlign = 1024 * 8 = 8192
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_3d_fp32_success)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{30, 49152}, {30, 49152}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{30, 8192}, {30, 8192}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1024, 6, 6}, {1024, 6, 6}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "1024 1024 0 0 6 6 8 15 16 64 1 1 64 0 40 1 40 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试4D输入（BSNN布局）+ n=8 + FLOAT32数据类型的正常Tiling计算场景
// 输入：grad_y=[2, 256, 8, 8], norm=[50, 32768], sum=[50, 4096]
// 输出：grad_input=[2, 256, 8, 8]
// totalLength = B*S = 2*256 = 512
// numIters = norm.dim0 / 2 = 50 / 2 = 25
// nAlignSize = 8 (nSize=8, inputNumPerBlock=16)
// norm.dim1 = totalLength * nSize * nAlign = 512 * 8 * 8 = 32768
// sum.dim1 = totalLength * nAlign = 512 * 8 = 4096
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_n8_4d_success)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{2, 256, 8, 8}, {2, 256, 8, 8}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{50, 32768}, {50, 32768}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{50, 4096}, {50, 4096}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{2, 256, 8, 8}, {2, 256, 8, 8}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "512 0 2 256 8 8 25 8 64 1 1 64 0 64 1 64 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试数据类型校验：grad_y使用FLOAT16数据类型，应返回GRAPH_FAILED
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_dtype)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{40, 8192}, {40, 8192}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 2048}, {40, 2048}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试迭代次数校验：norm的dim0=202，numIters=101超出范围[1,100]，应返回GRAPH_FAILED
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{202, 8192}, {202, 8192}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{202, 2048}, {202, 2048}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试迭代次数奇偶校验：norm的dim0=41为奇数，应返回GRAPH_FAILED
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_odd_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{41, 8192}, {41, 8192}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{41, 2048}, {41, 2048}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试grad_y维度校验：grad_y为5D张量（非3D/4D），应返回GRAPH_FAILED
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_grad_y_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum (invalid 5D shape)
            {{{1, 2, 128, 4, 4}, {1, 2, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 16384}, {40, 16384}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 4096}, {40, 4096}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 2, 128, 4, 4}, {1, 2, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试输出shape校验：grad_input的shape与grad_y的shape不匹配，应返回GRAPH_FAILED
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_output_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 8192}, {40, 8192}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 2048}, {40, 2048}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input (invalid shape)
            {{{1, 128, 6, 6}, {1, 128, 6, 6}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试norm shape校验：norm的dim1应该等于totalLength * nSize * nAlignSize，应返回GRAPH_FAILED
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_norm_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 8191}, {40, 8191}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 2048}, {40, 2048}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试sum shape校验：sum的dim1应该等于totalLength * nAlignSize，应返回GRAPH_FAILED
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_invalid_sum_shape)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 8192}, {40, 8192}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{40, 2047}, {40, 2047}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 128, 4, 4}, {1, 128, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试4D输入大batch场景：grad_y=[4, 256, 4, 4], norm=[20, 65536], sum=[20, 16384]
// totalLength = B*S = 4*256 = 1024
// numIters = norm.dim0 / 2 = 20 / 2 = 10
// nAlignSize = 16 (nSize=4, inputNumPerBlock=16)
// norm.dim1 = totalLength * nSize * nAlign = 1024 * 4 * 16 = 65536
// sum.dim1 = totalLength * nAlign = 1024 * 16 = 16384
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_4d_large_batch)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{4, 256, 4, 4}, {4, 256, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{20, 65536}, {20, 65536}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{20, 16384}, {20, 16384}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{4, 256, 4, 4}, {4, 256, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "1024 0 4 256 4 16 10 16 64 1 1 64 0 64 1 64 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试3D输入大T场景：grad_y=[4096, 8, 8], norm=[50, 262144], sum=[50, 32768]
// totalLength = T = 4096
// numIters = norm.dim0 / 2 = 50 / 2 = 25
// nAlignSize = 8 (nSize=8, inputNumPerBlock=16)
// norm.dim1 = totalLength * nSize * nAlign = 4096 * 8 * 8 = 262144
// sum.dim1 = totalLength * nAlign = 4096 * 8 = 32768
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_3d_large_t)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{4096, 8, 8}, {4096, 8, 8}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{50, 262144}, {50, 262144}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{50, 32768}, {50, 32768}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{4096, 8, 8}, {4096, 8, 8}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4096 4096 0 0 4096 8 8 25 64 64 1 1 64 0 64 1 64 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试最小迭代次数场景：norm的dim0=2，numIters=1
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_min_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{2, 4096}, {2, 4096}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "64 0 1 64 4 16 1 1 64 1 1 64 0 64 1 64 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试最大迭代次数场景：norm的dim0=200，numIters=100
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_max_num_iters)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{200, 4096}, {200, 4096}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{200, 1024}, {200, 1024}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{1, 64, 4, 4}, {1, 64, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "64 0 1 64 4 16 100 1 64 1 1 64 0 64 1 64 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试3D输入n=4场景：grad_y=[512, 4, 4], norm=[20, 32768], sum=[20, 8192]
// totalLength = T = 512
// numIters = norm.dim0 / 2 = 20 / 2 = 10
// nAlignSize = 16 (nSize=4, inputNumPerBlock=16)
// norm.dim1 = totalLength * nSize * nAlign = 512 * 4 * 16 = 32768
// sum.dim1 = totalLength * nAlign = 512 * 16 = 8192
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_3d_n4)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{512, 4, 4}, {512, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{20, 32768}, {20, 32768}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{20, 8192}, {20, 8192}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{512, 4, 4}, {512, 4, 4}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "512 512 0 0 512 4 16 10 8 64 1 1 64 0 64 1 64 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

// 测试4D输入n=6场景：grad_y=[2, 256, 6, 6], norm=[30, 24576], sum=[30, 4096]
// totalLength = B*S = 2*256 = 512
// numIters = norm.dim0 / 2 = 30 / 2 = 15
// nAlignSize = 8 (nSize=6, inputNumPerBlock=16)
// norm.dim1 = totalLength * nSize * nAlign = 512 * 6 * 8 = 24576
// sum.dim1 = totalLength * nAlign = 512 * 8 = 4096
TEST_F(MhcSinkhornBackwardTiling, test_mhc_sinkhorn_backward_4d_n6)
{
    MhcSinkhornBackwardCompileInfo compileInfo = {64, 64};
    gert::TilingContextPara tilingContextPara("MhcSinkhornBackward",
        {
            // input info: grad_y, norm, sum
            {{{2, 256, 6, 6}, {2, 256, 6, 6}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{30, 24576}, {30, 24576}}, ge::DT_FLOAT32, ge::FORMAT_ND},
            {{{30, 4096}, {30, 4096}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {
            // output info: grad_input
            {{{2, 256, 6, 6}, {2, 256, 6, 6}}, ge::DT_FLOAT32, ge::FORMAT_ND},
        },
        {});
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "512 0 2 256 6 8 15 8 64 1 1 64 0 64 1 64 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}
