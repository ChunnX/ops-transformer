/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "./test_ai_infra_sinkhorn_grad_fixture.h"

using namespace std;

// ==================== 正常测试用例 ====================

TEST_F(AiInfraSinkhornGradTiling, case_normal_3d)
{
    TestAiInfraSinkhornGradTNN("case_normal_3d", 100, 4, 3);
}

TEST_F(AiInfraSinkhornGradTiling, case_normal_4d)
{
    TestAiInfraSinkhornGradBSNN("case_normal_4d", 2, 50, 6, 5);
}

TEST_F(AiInfraSinkhornGradTiling, case_normal_small_n)
{
    TestAiInfraSinkhornGradTNN("case_normal_large_n", 50, 1, 28);
}

TEST_F(AiInfraSinkhornGradTiling, case_normal_large_n)
{
    TestAiInfraSinkhornGradBSNN("case_normal_large_n", 1, 20, 8, 10);
}

TEST_F(AiInfraSinkhornGradTiling, case_normal_small_num_iters)
{
    TestAiInfraSinkhornGradTNN("case_normal_large_n", 1024, 4, 1);
}

TEST_F(AiInfraSinkhornGradTiling, case_normal_large_num_iters)
{
    TestAiInfraSinkhornGradBSNN("case_normal_large_n", 2, 128, 4, 100);
}

TEST_F(AiInfraSinkhornGradTiling, case_normal_typical_network_tnn)
{
    TestAiInfraSinkhornGradTNN("case_normal_typical_network_tnn", 8192, 4);
}

TEST_F(AiInfraSinkhornGradTiling, case_normal_typical_network_bsnn)
{
    TestAiInfraSinkhornGradBSNN("case_normal_typical_network_bsnn", 2, 4096, 4);
}

// ==================== grad_output 相关错误测试用例 ====================

TEST_F(AiInfraSinkhornGradTiling, case_error_dim_not_3_4_grad_output)
{
    TestAiInfraSinkhornGradTNN("case_error_dim_not_3_4_grad_output", 100, 4, 3, ErrorType::GRAD_OUTPUT_DIM_NOT_3_4);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_shape_n_diff_grad_output)
{
    TestAiInfraSinkhornGradBSNN("case_error_shape_n_diff_grad_output", 2, 50, 4, 3, ErrorType::GRAD_OUTPUT_N_DIFF_BSNN);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_dtype_not_fp32_grad_output)
{
    TestAiInfraSinkhornGradTNN("case_error_dtype_not_fp32_grad_output", 100, 4, 3, ErrorType::GRAD_OUTPUT_DTYPE_NOT_FP32);
}

// ==================== norm_out 相关错误测试用例 ====================

TEST_F(AiInfraSinkhornGradTiling, case_error_dim_not_4_5_norm_out)
{
    TestAiInfraSinkhornGradBSNN("case_error_dim_not_4_5_norm_out", 2, 50, 4, 3, ErrorType::NORM_OUT_DIM_NOT_4_5_BSNN);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_dim_2i_not_even_norm_out)
{
    TestAiInfraSinkhornGradTNN("case_error_dim_2i_not_even_norm_out", 100, 4, 3, ErrorType::NORM_OUT_2I_NOT_EVEN);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_shape_n_diff_norm_out)
{
    TestAiInfraSinkhornGradBSNN("case_error_shape_n_diff_norm_out", 2, 50, 4, 3, ErrorType::NORM_OUT_N_DIFF);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_shape_bs_diff_norm_out_3d)
{
    TestAiInfraSinkhornGradTNN("case_error_shape_bs_diff_norm_out_3d", 100, 4, 3, ErrorType::NORM_OUT_BS_DIFF);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_shape_bs_diff_norm_out_4d)
{
    TestAiInfraSinkhornGradBSNN("case_error_shape_bs_diff_norm_out_4d", 2, 50, 4, 3, ErrorType::NORM_OUT_BS_DIFF);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_dtype_not_fp32_norm_out)
{
    TestAiInfraSinkhornGradTNN("case_error_dtype_not_fp32_norm_out", 100, 4, 3, ErrorType::NORM_OUT_DTYPE_NOT_FP32);
}

// ==================== sum_out 相关错误测试用例 ====================

TEST_F(AiInfraSinkhornGradTiling, case_error_dim_not_3_4_sum_out)
{
    TestAiInfraSinkhornGradBSNN("case_error_dim_not_3_4_sum_out", 2, 50, 4, 3, ErrorType::SUM_OUT_DIM_NOT_3_4_BSNN);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_dim_2i_not_even_sum_out)
{
    TestAiInfraSinkhornGradTNN("case_error_dim_2i_not_even_sum_out", 100, 4, 3, ErrorType::SUM_OUT_2I_NOT_EVEN);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_shape_n_diff_sum_out)
{
    TestAiInfraSinkhornGradTNN("case_error_shape_n_diff_sum_out", 100, 4, 3, ErrorType::SUM_OUT_N_DIFF);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_shape_bs_diff_sum_out_3d)
{
    TestAiInfraSinkhornGradTNN("case_error_shape_bs_diff_sum_out_3d", 100, 4, 3, ErrorType::SUM_OUT_BS_DIFF);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_shape_bs_diff_sum_out_4d)
{
    TestAiInfraSinkhornGradBSNN("case_error_shape_bs_diff_sum_out_4d", 2, 50, 4, 3, ErrorType::SUM_OUT_BS_DIFF);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_dtype_not_fp32_sum_out)
{
    TestAiInfraSinkhornGradTNN("case_error_dtype_not_fp32_sum_out", 100, 4, 3, ErrorType::SUM_OUT_DTYPE_NOT_FP32);
}

// ==================== grad_input 相关错误测试用例 ====================

TEST_F(AiInfraSinkhornGradTiling, case_error_dim_not_3_4_grad_input)
{
    TestAiInfraSinkhornGradTNN("case_error_dim_not_3_4_grad_input", 100, 4, 3, ErrorType::GRAD_INPUT_DIM_NOT_3_4);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_shape_diff_grad_input)
{
    TestAiInfraSinkhornGradBSNN("case_error_shape_diff_grad_input", 2, 50, 4, 3, ErrorType::GRAD_INPUT_SHAPE_DIFF);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_dtype_not_fp32_grad_input)
{
    TestAiInfraSinkhornGradTNN("case_error_dtype_not_fp32_grad_input", 100, 4, 3, ErrorType::GRAD_INPUT_DTYPE_NOT_FP32);
}

// ==================== 空 Tensor/Desc 测试用例 ====================
TEST_F(AiInfraSinkhornGradTiling, case_error_null_desc_grad_output)
{
    TestAiInfraSinkhornGradBSNN("case_error_null_desc_grad_output", 2, 50, 4, 3, ErrorType::GRAD_OUTPUT_NONE_DESC);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_null_tensor_norm_out)
{
    TestAiInfraSinkhornGradTNN("case_error_null_tensor_norm_out", 100, 4, 3, ErrorType::NORM_OUT_NONE);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_null_desc_norm_out)
{
    TestAiInfraSinkhornGradBSNN("case_error_null_desc_norm_out", 2, 50, 4, 3, ErrorType::NORM_OUT_NONE_DESC);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_null_tensor_sum_out)
{
    TestAiInfraSinkhornGradTNN("case_error_null_tensor_sum_out", 100, 4, 3, ErrorType::SUM_OUT_NONE);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_null_desc_sum_out)
{
    TestAiInfraSinkhornGradBSNN("case_error_null_desc_sum_out", 2, 50, 4, 3, ErrorType::SUM_OUT_NONE_DESC);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_null_desc_grad_input)
{
    TestAiInfraSinkhornGradBSNN("case_error_null_desc_grad_input", 2, 50, 4, 3, ErrorType::GRAD_INPUT_NONE_DESC);
}

// ==================== num_iters 取值范围 测试用例 ====================

TEST_F(AiInfraSinkhornGradTiling, case_error_num_iters_less_1)
{
    TestAiInfraSinkhornGradBSNN("case_error_num_iters_less_1", 2, 50, 4, 3, ErrorType::NORM_OUT_NUM_ITERS_LESS_1);
}

TEST_F(AiInfraSinkhornGradTiling, case_error_num_iters_greater_100)
{
    TestAiInfraSinkhornGradTNN("case_error_num_iters_greater_100", 250, 4, 3, ErrorType::NORM_OUT_NUM_ITERS_GREATER_100);
}
