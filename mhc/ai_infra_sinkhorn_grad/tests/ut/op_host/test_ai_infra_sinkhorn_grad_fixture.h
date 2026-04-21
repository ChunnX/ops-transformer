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
#include "../../../op_host/ai_infra_sinkhorn_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
// ==================== 枚举定义 ====================

// 错误类型枚举
enum class ErrorType {
    NONE,                                       // 正常用例

    // grad_output 相关错误
    GRAD_OUTPUT_DIM_NOT_3_4,                    // grad_output 维度不是 3 或 4
    GRAD_OUTPUT_N_DIFF_TNN,                     // grad_output 最后两个维度不相等 (TNN)
    GRAD_OUTPUT_N_DIFF_BSNN,                    // grad_output 最后两个维度不相等 (BSNN)
    GRAD_OUTPUT_DTYPE_NOT_FP32,                 // grad_output 数据类型不是 FP32

    // norm_out 相关错误
    NORM_OUT_DIM_NOT_4_5_TNN,                   // norm_out 维度不是 4 (TNN)
    NORM_OUT_DIM_NOT_4_5_BSNN,                  // norm_out 维度不是 5 (BSNN)
    NORM_OUT_2I_NOT_EVEN,                       // norm_out 第一个维度不是偶数
    NORM_OUT_N_DIFF,                            // norm_out 的 n 维度与 grad_output 不一致
    NORM_OUT_BS_DIFF,                           // norm_out 的 bs/t 维度与 grad_output 不一致
    NORM_OUT_DTYPE_NOT_FP32,                    // norm_out 数据类型不是 FP32

    // sum_out 相关错误
    SUM_OUT_DIM_NOT_3_4_TNN,                    // sum_out 维度不是 3 (TNN)
    SUM_OUT_DIM_NOT_3_4_BSNN,                   // sum_out 维度不是 4 (BSNN)
    SUM_OUT_2I_NOT_EVEN,                        // sum_out 第一个维度不是偶数
    SUM_OUT_N_DIFF,                             // sum_out 的 n 维度与 norm_out 不一致
    SUM_OUT_BS_DIFF,                            // sum_out 的 bs/t 维度与 grad_output 不一致
    SUM_OUT_DTYPE_NOT_FP32,                     // sum_out 数据类型不是 FP32

    // grad_input 相关错误
    GRAD_INPUT_DIM_NOT_3_4,                     // grad_input 维度不是 3 或 4
    GRAD_INPUT_SHAPE_DIFF,                      // grad_input 的 shape 与 grad_output 不一致
    GRAD_INPUT_DTYPE_NOT_FP32,                  // grad_input 数据类型不是 FP32

    // 空 Tensor/Desc 相关错误
    GRAD_OUTPUT_NONE,
    GRAD_OUTPUT_NONE_DESC,
    NORM_OUT_NONE,
    NORM_OUT_NONE_DESC,
    SUM_OUT_NONE,
    SUM_OUT_NONE_DESC,
    GRAD_INPUT_NONE,
    GRAD_INPUT_NONE_DESC,

    // numIters 范围
    NORM_OUT_NUM_ITERS_LESS_1,
    NORM_OUT_NUM_ITERS_GREATER_100
};

// ==================== 测试夹具类 ====================

class AiInfraSinkhornGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AiInfraSinkhornGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AiInfraSinkhornGradTiling TearDown" << std::endl;
    }

    /**
     * @brief 构造TNN模式(3D)的TilingContext
     */
    static gert::TilingContextPara BuildTilingContextTNN(uint32_t T, uint32_t n, uint32_t numIters
                                                          )
    {
        static optiling::AiInfraSinkhornGradCompileInfo compileInfo = {48, 48};
        return gert::TilingContextPara(
            "AiInfraSinkhornGrad",
            {
                {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},           // grad_output
                {{{2 * numIters, n, n, T}, {2 * numIters, n, n, T}}, ge::DT_FLOAT, ge::FORMAT_ND},  // norm_out
                {{{2 * numIters, n, T}, {2 * numIters, n, T}}, ge::DT_FLOAT, ge::FORMAT_ND}         // sum_out
            },
            {
                {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND}  // grad_input
            },
            {},
            &compileInfo
        );
    }

    /**
     * @brief 构造BSNN模式(4D)的TilingContext
     */
    static gert::TilingContextPara BuildTilingContextBSNN(uint32_t B, uint32_t S, uint32_t n, uint32_t numIters
                                                          )
    {
        static optiling::AiInfraSinkhornGradCompileInfo compileInfo = {48, 48};
        return gert::TilingContextPara(
            "AiInfraSinkhornGrad",
            {
                {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},           // grad_output
                {{{2 * numIters, n, n, B, S}, {2 * numIters, n, n, B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},  // norm_out
                {{{2 * numIters, n, B, S}, {2 * numIters, n, B, S}}, ge::DT_FLOAT, ge::FORMAT_ND}         // sum_out
            },
            {
                {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND}  // grad_input
            },
            {},
            &compileInfo
        );
    }

    /**
     * @brief 根据错误类型修改TilingContext
     */
    static void ApplyErrorModifications(gert::TilingContextPara& context, ErrorType errorType, bool isTshape,
                                        uint32_t n, uint32_t numIters, uint32_t T, uint32_t B, uint32_t S)
    {   
        switch (errorType) {
            case ErrorType::NONE:
                break;

            // grad_output 相关错误
            case ErrorType::GRAD_OUTPUT_DIM_NOT_3_4:
                context.inputTensorDesc_[0].shape_ = {{n, n}, {n, n}};
                break;
            case ErrorType::GRAD_OUTPUT_N_DIFF_TNN:
                context.inputTensorDesc_[0].shape_ = {{T, n, n + 1}, {T, n, n + 1}};
                break;
            case ErrorType::GRAD_OUTPUT_N_DIFF_BSNN:
                context.inputTensorDesc_[0].shape_ = {{B, S, n, n + 1}, {B, S, n, n + 1}};
                break;
            case ErrorType::GRAD_OUTPUT_DTYPE_NOT_FP32:
                context.inputTensorDesc_[0].dtype_ = ge::DT_FLOAT16;
                break;

            // norm_out 相关错误
            case ErrorType::NORM_OUT_DIM_NOT_4_5_TNN:
                context.inputTensorDesc_[1].shape_ = {{2 * numIters, n, n}, {2 * numIters, n, n}};

                break;
            case ErrorType::NORM_OUT_DIM_NOT_4_5_BSNN:

                context.inputTensorDesc_[1].shape_ = {{2 * numIters, n, n, B * S}, {2 * numIters, n, n, B * S}};
                break;
            case ErrorType::NORM_OUT_2I_NOT_EVEN:
                if (isTshape) {
                    context.inputTensorDesc_[1].shape_ = {{2 * numIters + 1, n, n, T}, {2 * numIters + 1, n, n, T}};
                    context.inputTensorDesc_[2].shape_ = {{2 * numIters + 1, n, T}, {2 * numIters + 1, n, T}};
                } else {
                    context.inputTensorDesc_[1].shape_ = {{2 * numIters + 1, n, n, B, S}, {2 * numIters + 1, n, n, B, S}};
                    context.inputTensorDesc_[2].shape_ = {{2 * numIters + 1, n, B, S}, {2 * numIters + 1, n, B, S}};
                }
                break;
            case ErrorType::NORM_OUT_N_DIFF:
                if (isTshape) {
                    context.inputTensorDesc_[1].shape_ = {{2 * numIters, n, n + 1, T}, {2 * numIters, n, n + 1, T}};
                } else {
                    context.inputTensorDesc_[1].shape_ = {{2 * numIters, n, n + 1, B, S}, {2 * numIters, n, n + 1, B, S}};
                }
                break;
            case ErrorType::NORM_OUT_BS_DIFF:
                context.inputTensorDesc_[1].shape_ = {{2 * numIters, n, n + 1, B + 10, S}, {2 * numIters, n, n + 1, B + 10, S}};
                break;
            case ErrorType::NORM_OUT_DTYPE_NOT_FP32:
                context.inputTensorDesc_[1].dtype_ = ge::DT_FLOAT16;
                break;

            // sum_out 相关错误
            case ErrorType::SUM_OUT_DIM_NOT_3_4_TNN:
                context.inputTensorDesc_[2].shape_ = {{2 * numIters, n}, {2 * numIters, n}};
                break;
            case ErrorType::SUM_OUT_DIM_NOT_3_4_BSNN:
                context.inputTensorDesc_[2].shape_ = {{2 * numIters, n, B * S}, {2 * numIters, n, B * S}};
                break;
            case ErrorType::SUM_OUT_2I_NOT_EVEN:
                if (isTshape) {
                    context.inputTensorDesc_[2].shape_ = {{2 * numIters + 1, n, T}, {2 * numIters + 1, n, T}};
                } else {
                    context.inputTensorDesc_[2].shape_ = {{2 * numIters + 1, n, B, S}, {2 * numIters + 1, n, B, S}};
                }
                break;
            case ErrorType::SUM_OUT_N_DIFF:
                context.inputTensorDesc_[2].shape_ = {{2 * numIters, n + 1, T}, {2 * numIters, n + 1, T}};
                break;
            case ErrorType::SUM_OUT_BS_DIFF:
                if (isTshape) {
                    context.inputTensorDesc_[2].shape_ = {{2 * numIters, n, T + 10}, {2 * numIters, n, T + 10}};
                } else {
                    context.inputTensorDesc_[2].shape_ = {{2 * numIters, n, B + 10, S}, {2 * numIters, n, B +10, S}};
                }
                break;
            case ErrorType::SUM_OUT_DTYPE_NOT_FP32:
                context.inputTensorDesc_[2].dtype_ = ge::DT_FLOAT16;
                break;

            // grad_input 相关错误
            case ErrorType::GRAD_INPUT_DIM_NOT_3_4:
                context.outputTensorDesc_[0].shape_ = {{n, n}, {n, n}};
                break;
            case ErrorType::GRAD_INPUT_SHAPE_DIFF:
                if (isTshape) {
                    context.outputTensorDesc_[0].shape_ = {{T + 10, n, n}, {T + 10, n, n}};
                } else {
                    context.outputTensorDesc_[0].shape_ = {{B + 10, S, n, n}, {B + 10, S, n, n}};
                }
                break;
            case ErrorType::GRAD_INPUT_DTYPE_NOT_FP32:
                context.outputTensorDesc_[0].dtype_ = ge::DT_FLOAT16;
                break;

            // 补充 空 tensor
            case ErrorType::GRAD_OUTPUT_NONE_DESC:
                context.inputTensorDesc_[0].shape_ = {{},{}};
                break;
            case ErrorType::NORM_OUT_NONE:
                context.inputTensorDesc_.pop_back();
                context.inputTensorDesc_.pop_back();
                break;
            case ErrorType::NORM_OUT_NONE_DESC:
                context.inputTensorDesc_[1].shape_ = {};
                break;
            case ErrorType::SUM_OUT_NONE:
                context.inputTensorDesc_.pop_back();
                break;
            case ErrorType::SUM_OUT_NONE_DESC:
                context.inputTensorDesc_[2].shape_ = {{},{}};
                break;
            case ErrorType::GRAD_INPUT_NONE_DESC:
                context.outputTensorDesc_[0].shape_ = {{},{}};
                break;

            // 补充 numIters 范围
            case ErrorType::NORM_OUT_NUM_ITERS_LESS_1:
                if (isTshape) {
                    context.inputTensorDesc_[1].shape_ = {{0, n, n, T}, {0, n, n, T}};
                    context.inputTensorDesc_[2].shape_ = {{0, n, T}, {0, n, T}};
                } else {
                    context.inputTensorDesc_[1].shape_ = {{0, n, n, B, S}, {0, n, n, B, S}};
                    context.inputTensorDesc_[2].shape_ = {{0, n, B, S}, {0, n, B, S}};
                }
                break;
                break;
            case ErrorType::NORM_OUT_NUM_ITERS_GREATER_100:
                if (isTshape) {
                    context.inputTensorDesc_[1].shape_ = {{202, n, n, T}, {202, n, n, T}};
                    context.inputTensorDesc_[2].shape_ = {{202, n, T}, {202, n, T}};
                } else {
                    context.inputTensorDesc_[1].shape_ = {{202, n, n, B, S}, {202, n, n, B, S}};
                    context.inputTensorDesc_[2].shape_ = {{202, n, B, S}, {202, n, B, S}};
                }
                break;
            default:
                break;
        }
    }

    /**
     * @brief TNN模式测试夹具
     */
    static void TestAiInfraSinkhornGradTNN(const string& caseName, uint32_t T, uint32_t n,
                                     uint32_t numIters = 20, ErrorType errorType = ErrorType::NONE)
    {
        optiling::AiInfraSinkhornGradCompileInfo compileInfo = {40, 40};
        gert::TilingContextPara tilingContextPara = BuildTilingContextTNN(T, n, numIters);

        if (errorType != ErrorType::NONE) {
            ApplyErrorModifications(tilingContextPara, errorType, true, n, numIters, T, 0, 0);
        }

        int64_t expectTilingKey = 0;
        string expectTilingDataStr = "";
        ge::graphStatus expectStatus = (errorType == ErrorType::NONE) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
        std::vector<size_t> expectWorkspaces = (errorType == ErrorType::NONE) ?
                                                std::vector<size_t>{16777216} : std::vector<size_t>{0};

        ExecuteTestCase(tilingContextPara, expectStatus, expectTilingKey,
                        expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int64_t>);
    }

    /**
     * @brief BSNN模式测试夹具
     */
    static void TestAiInfraSinkhornGradBSNN(const string& caseName, uint32_t B, uint32_t S, uint32_t n,
                                      uint32_t numIters = 20, ErrorType errorType = ErrorType::NONE)
    {
        gert::TilingContextPara tilingContextPara = BuildTilingContextBSNN(B, S, n, numIters);

        if (errorType != ErrorType::NONE) {
            ApplyErrorModifications(tilingContextPara, errorType, false, n, numIters, 0, B, S);
        }

        int64_t expectTilingKey = 0;
        string expectTilingDataStr = "";
        ge::graphStatus expectStatus = (errorType == ErrorType::NONE) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
        std::vector<size_t> expectWorkspaces = (errorType == ErrorType::NONE) ?
                                                std::vector<size_t>{16777216} : std::vector<size_t>{0};

        ExecuteTestCase(tilingContextPara, expectStatus, expectTilingKey,
                        expectTilingDataStr, expectWorkspaces, 0, TilingData2Str<int64_t>);
    }

    template <typename T>
    static string TilingData2Str(void* buf, size_t size) {
        string result;
        const T* data = reinterpret_cast<const T*>(buf);
        size_t len = size / sizeof(T);
        for (size_t i = 0; i < len; i++) {
            result += std::to_string(data[i]);
            result += " ";
        }
        return result;
    }
};
