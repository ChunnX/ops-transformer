/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <gtest/gtest.h>
 #include <iostream>
 #include "infershape_context_faker.h"
 #include "base/registry/op_impl_space_registry_v2.h"
 #include <vector>

class RopeWithSinCosCache : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RopeWithSinCosCache Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RopeWithSinCosCache Proto Test TearDown" << std::endl;
    }
};

TEST_F(RopeWithSinCosCache, rope_with_sin_cos_cache_bf16_true)
{
    gert::StorageShape positionShape = {{48}, {48}};
    gert::StorageShape queryShape = {{48, 256}, {48, 256}};
    gert::StorageShape keyShape = {{48, 512}, {48, 512}};
    gert::StorageShape cosSinCacheShape = {{48, 128}, {48, 128}};
    std::vector<int64_t> mropeParams{0, 0, 0};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("RopeWithSinCosCache")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1, 1})
                      .InputShapes({&positionShape, &queryShape, &keyShape, &cosSinCacheShape})
                      .OutputShapes({&queryShape, &keyShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                           {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                           {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                           {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
                           {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                           {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                           {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RopeWithSinCosCache")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedQueryOutputShape = {48, 256};
    std::vector<int64_t> expectedKeyOutputShape = {48, 512};
    auto queryOutShape = context->GetOutputShape(0);
    auto keyOutShape = context->GetOutputShape(1);
    EXPECT_EQ(ops::ToVector(*queryOutShape), expectedQueryOutputShape);
    EXPECT_EQ(ops::ToVector(*keyOutShape), expectedKeyOutputShape);
}

TEST_F(RopeWithSinCosCache, rope_with_sin_cos_cache_fp16_true)
{
    gert::StorageShape positionShape = {{48}, {48}};
    gert::StorageShape queryShape = {{48, 256}, {48, 256}};
    gert::StorageShape keyShape = {{48, 512}, {48, 512}};
    gert::StorageShape cosSinCacheShape = {{48, 128}, {48, 128}};
    std::vector<int64_t> mropeParams{0, 0, 0};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("RopeWithSinCosCache")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1, 1})
                      .InputShapes({&positionShape, &queryShape, &keyShape, &cosSinCacheShape})
                      .OutputShapes({&queryShape, &keyShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                           {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                           {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                           {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
                           {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                           {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                           {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RopeWithSinCosCache")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedQueryOutputShape = {48, 256};
    std::vector<int64_t> expectedKeyOutputShape = {48, 512};
    auto queryOutShape = context->GetOutputShape(0);
    auto keyOutShape = context->GetOutputShape(1);
    EXPECT_EQ(ops::ToVector(*queryOutShape), expectedQueryOutputShape);
    EXPECT_EQ(ops::ToVector(*keyOutShape), expectedKeyOutputShape);
}

TEST_F(RopeWithSinCosCache, rope_with_sin_cos_cache_fp32_true)
{
    gert::StorageShape positionShape = {{48}, {48}};
    gert::StorageShape queryShape = {{48, 256}, {48, 256}};
    gert::StorageShape keyShape = {{48, 512}, {48, 512}};
    gert::StorageShape cosSinCacheShape = {{48, 128}, {48, 128}};
    vector<int64_t> mropeParams{0, 0, 0};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("RopeWithSinCosCache")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1, 1})
                      .InputShapes({&positionShape, &queryShape, &keyShape, &cosSinCacheShape})
                      .OutputShapes({&queryShape, &keyShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                           {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                           {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                           {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
                           {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                           {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
                           {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RopeWithSinCosCache")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedQueryOutputShape = {48, 256};
    std::vector<int64_t> expectedKeyOutputShape = {48, 512};
    auto queryOutShape = context->GetOutputShape(0);
    auto keyOutShape = context->GetOutputShape(1);
    EXPECT_EQ(ops::ToVector(*queryOutShape), expectedQueryOutputShape);
    EXPECT_EQ(ops::ToVector(*keyOutShape), expectedKeyOutputShape);
}