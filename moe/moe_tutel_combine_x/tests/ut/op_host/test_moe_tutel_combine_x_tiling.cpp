/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_tutel_combine_x_tiling.cpp
 * \brief Tiling UT。覆盖三种 dtype、K/S/H 边界、H 超大触发 H 切分，以及各条 host 校验。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <limits>

#include "tiling/platform/platform_ascendc.h"
#include "tiling_case_executor.h"

namespace {
constexpr uint64_t TILING_KEY_MAIN = 10000;
} // namespace

class MoeTutelCombineXTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTutelCombineXTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTutelCombineXTiling TearDown" << std::endl;
    }
};

// 主路径 fp32：E=8, C=32, H=1024, K=2, S=512。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_float32)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{8, 32, 1024}, {8, 32, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 512}, {2, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 512}, {2, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 512}, {2, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{512, 1024}, {512, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_MAIN);
}

// 主路径 fp16。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_float16)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{16, 64, 2048}, {16, 64, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_MAIN);
}

// 主路径 bf16 + H 非 32B 对齐尾块（H=1023）。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_bfloat16_unaligned_h)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 8, 1023}, {4, 8, 1023}}, ge::DT_BF16, ge::FORMAT_ND},
                                                    {{{2, 33}, {2, 33}}, ge::DT_BF16, ge::FORMAT_ND},
                                                    {{{2, 33}, {2, 33}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 33}, {2, 33}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{33, 1023}, {33, 1023}}, ge::DT_BF16, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_MAIN);
}

// K=0：空归约轴，tiling 正常下发，kernel 写全 0。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_k_zero)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 8, 128}, {4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{0, 64}, {0, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{0, 64}, {0, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{0, 64}, {0, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_MAIN);
}

// S=0 空 tensor：派 1 核走空循环。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_empty_samples)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 8, 128}, {4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 0}, {2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 0}, {2, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 0}, {2, 0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{0, 128}, {0, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_MAIN);
}

// H=0 空 tensor。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_empty_hidden)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 8, 0}, {4, 8, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{16, 0}, {16, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_MAIN);
}

// C=0（每专家容量为 0）：locations < 0 恒假，全部路由无效，kernel 写全 0。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_empty_capacity)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 0, 128}, {4, 0, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{16, 128}, {16, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_MAIN);
}

// E=0（专家数为 0）：tiling 不拒收；E 不参与任何计算，越界索引由调用方契约保证。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_empty_expert)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{0, 8, 128}, {0, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{16, 128}, {16, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_MAIN);
}

// H 超大：单行放不进 UB，hFactor 被切小，仍走同一个 TilingKey。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_huge_hidden)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{2, 2, 262144}, {2, 2, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{4, 262144}, {4, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_MAIN);
}

// K 很大：kFactor 受 UB 限制被切小，仍能下发。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_large_k)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{64, 128, 4096}, {64, 128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                    {{{64, 128}, {64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                    {{{64, 128}, {64, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{64, 128}, {64, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, TILING_KEY_MAIN);
}

// indices 与 gates shape 不一致：拒绝。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_route_shape_mismatch)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 8, 128}, {4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{3, 64}, {3, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// locations 与 gates shape 不一致：拒绝。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_locations_shape_mismatch)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 8, 128}, {4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// y_grad rank 不是 3：拒绝。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_invalid_y_grad_rank)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 128}, {4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// gates 与 y_grad dtype 不一致：拒绝。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_dtype_mismatch)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 8, 128}, {4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// indices dtype 不是 int32：拒绝。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_invalid_indices_dtype)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 8, 128}, {4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// x_grad shape 与 [S, H] 不符：拒绝。
TEST_F(MoeTutelCombineXTiling, moe_tutel_combine_x_tiling_invalid_output_shape)
{
    gert::TilingContextPara optilingContextPara("MoeTutelCombineX",
                                                {
                                                    {{{4, 8, 128}, {4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                    {{{2, 64}, {2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{64, 256}, {64, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {}, nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
