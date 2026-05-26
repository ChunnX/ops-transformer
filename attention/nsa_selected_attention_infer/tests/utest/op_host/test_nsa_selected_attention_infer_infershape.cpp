/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_case_executor.h"
#include "nsai_tiling_ut_common.h"

using namespace nsai_tiling_ut;

class NsaSelectedAttentionInferProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaSelectedAttentionInferProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaSelectedAttentionInferProto TearDown" << std::endl;
    }
};

TEST_F(NsaSelectedAttentionInferProto, NsaSelectAttentionInfer_Case_q1_bsnd_infershape)
{
    CaseShapes shapes = BuildCaseShapes(LegacyCaseQ1Bsnd());
    gert::InfershapeContextPara para(
        "NsaSelectedAttentionInfer",
        {
            {MakeStorageShape(shapes.query), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapes.key), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapes.value), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT32, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_BOOL, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT32, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {MakeStorageShape({}), ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"selected_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"selected_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {30, 1, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(NsaSelectedAttentionInferProto, NsaSelectAttentionInfer_Case_q3_tnd_infershape)
{
    CaseShapes shapes = BuildCaseShapes(LegacyCaseQ3Tnd());
    gert::InfershapeContextPara para(
        "NsaSelectedAttentionInfer",
        {
            {MakeStorageShape(shapes.query), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapes.key), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapes.value), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT32, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_BOOL, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT32, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {MakeStorageShape({}), ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"selected_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"selected_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {90, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
