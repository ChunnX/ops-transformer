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
#include "arch22/ncai_tiling_ut_common.h"

using namespace ncai_tiling_ut;

class NsaCompressAttentionInferProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressAttentionInferProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressAttentionInferProto TearDown" << std::endl;
    }
};

TEST_F(NsaCompressAttentionInferProto, case_nsa_compress_attention_infer_0_infershape)
{
    CaseShapes shapes = BuildCaseShapes(LegacyCase0Tnd());
    gert::InfershapeContextPara para(
        "NsaCompressAttentionInfer",
        {
            {MakeStorageShape(shapes.query), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapes.key), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapes.value), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_BOOL, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT32, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {MakeStorageShape({}), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"numHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"selectBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"selectBlockCount", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"pageBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {20, 64, 128},
        {20, 4, 16},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(NsaCompressAttentionInferProto, case_nsa_compress_attention_infer_1_infershape)
{
    CaseShapes shapes = BuildCaseShapes(LegacyCase1Bsnd());
    gert::InfershapeContextPara para(
        "NsaCompressAttentionInfer",
        {
            {MakeStorageShape(shapes.query), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapes.key), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape(shapes.value), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_BOOL, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT32, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT64, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {MakeStorageShape({}), ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeStorageShape({}), ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"numHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"selectBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"selectBlockCount", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"pageBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {20, 1, 64, 128},
        {20, 1, 4, 16},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
