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
 * \file test_moe_tutel_combine_gates_infershape.cpp
 * \brief Shape inference unit tests for MoeTutelCombineGates.
 *
 *        The shape contract under test: gates_grad.shape = indices.shape, with
 *        S = x.shape[0] = indices.shape[1] = locations.shape[1],
 *        H = x.shape[1] = y_grad.shape[2],
 *        B = y_grad.shape[0] = indices.shape[0] = locations.shape[0] and
 *        indices / locations holding exactly S*B elements.
 */

#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {

constexpr int64_t kDynamic = -1;

std::vector<gert::InfershapeContextPara::TensorDescription> MakeInputs(std::initializer_list<int64_t> xShape,
                                                                       std::initializer_list<int64_t> yShape,
                                                                       std::initializer_list<int64_t> idxShape,
                                                                       ge::DataType fcDtype = ge::DT_BF16,
                                                                       ge::Format format = ge::FORMAT_ND)
{
    return {
        {gert::StorageShape(xShape, xShape), fcDtype, format},
        {gert::StorageShape(yShape, yShape), fcDtype, format},
        {gert::StorageShape(idxShape, idxShape), ge::DT_INT32, format},
        {gert::StorageShape(idxShape, idxShape), ge::DT_INT32, format},
    };
}

std::vector<gert::InfershapeContextPara::TensorDescription> MakeOutputs(std::initializer_list<int64_t> shape)
{
    return {{gert::StorageShape(shape, shape), ge::DT_BF16, ge::FORMAT_ND}};
}

} // namespace

// ---------------------------------------------------------------------------
// Success: gates_grad.shape is copied from indices.shape.
// ---------------------------------------------------------------------------
TEST(MoeTutelCombineGatesInferShape, infershape_ok)
{
    gert::InfershapeContextPara para("MoeTutelCombineGates", MakeInputs({1024, 1024}, {8, 256, 1024}, {8, 1024}),
                                     MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{8, 1024}});
}

TEST(MoeTutelCombineGatesInferShape, infershape_ok_single_task)
{
    gert::InfershapeContextPara para("MoeTutelCombineGates", MakeInputs({1, 1}, {1, 1, 1}, {1, 1}), MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{1, 1}});
}

// Unknown dims are passed through, never guessed: the output simply mirrors the
// (also unknown) indices shape.
TEST(MoeTutelCombineGatesInferShape, infershape_ok_dynamic_dims)
{
    gert::InfershapeContextPara para(
        "MoeTutelCombineGates", MakeInputs({kDynamic, kDynamic}, {kDynamic, kDynamic, kDynamic}, {kDynamic, kDynamic}),
        MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{kDynamic, kDynamic}});
}

// S = 0 is a legal empty tensor; B still binds the leading dims.
TEST(MoeTutelCombineGatesInferShape, infershape_ok_empty_samples)
{
    gert::InfershapeContextPara para("MoeTutelCombineGates", MakeInputs({0, 8}, {2, 3, 8}, {2, 0}), MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{2, 0}});
}

// ---------------------------------------------------------------------------
// Rejection: rank, format, shape binding and element count.
//
// NOTE: the "a fully-known declared gates_grad shape that contradicts
// indices.shape is rejected instead of being overwritten" branch of
// InferShape4MoeTutelCombineGates cannot be driven from here: this framework
// feeds TensorDescription.shape_ only for INPUTS and leaves the output shape
// buffer untouched (InferShapeContextFaker::OutputShapes is a no-op), so
// GetOutputShape(0) is never a caller-declared shape.
// ---------------------------------------------------------------------------
TEST(MoeTutelCombineGatesInferShape, infershape_fail_rank_x)
{
    gert::InfershapeContextPara para("MoeTutelCombineGates", MakeInputs({8, 2, 1024}, {8, 256, 1024}, {8, 1024}),
                                     MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(MoeTutelCombineGatesInferShape, infershape_fail_rank_index)
{
    gert::InfershapeContextPara para("MoeTutelCombineGates", MakeInputs({1024, 1024}, {8, 256, 1024}, {8, 1024, 1}),
                                     MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(MoeTutelCombineGatesInferShape, infershape_fail_format_not_nd)
{
    gert::InfershapeContextPara para(
        "MoeTutelCombineGates", MakeInputs({1024, 1024}, {8, 256, 1024}, {8, 1024}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ),
        MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(MoeTutelCombineGatesInferShape, infershape_fail_hidden_binding)
{
    // y_grad.shape[2] = 512 contradicts x.shape[1] = 1024
    gert::InfershapeContextPara para("MoeTutelCombineGates", MakeInputs({1024, 1024}, {8, 256, 512}, {8, 1024}),
                                     MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(MoeTutelCombineGatesInferShape, infershape_fail_batch_binding)
{
    // indices.shape[0] = 4 contradicts y_grad.shape[0] = 8
    gert::InfershapeContextPara para("MoeTutelCombineGates", MakeInputs({1024, 1024}, {8, 256, 1024}, {4, 1024}),
                                     MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(MoeTutelCombineGatesInferShape, infershape_fail_samples_binding)
{
    // indices.shape[1] = 512 contradicts x.shape[0] = 1024
    gert::InfershapeContextPara para("MoeTutelCombineGates", MakeInputs({1024, 1024}, {8, 256, 1024}, {8, 512}),
                                     MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(MoeTutelCombineGatesInferShape, infershape_fail_element_count)
{
    // 8 x 1000 elements != S*B = 8 x 1024
    gert::InfershapeContextPara para("MoeTutelCombineGates", MakeInputs({1024, 1024}, {8, 256, 1024}, {8, 1000}),
                                     MakeOutputs({}));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
