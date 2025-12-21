/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"

class GatherPaKvCache : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
        setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", true);
        std::cout << "GatherPaKvCache Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "0", true);
        setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", true);
        std::cout << "GatherPaKvCache Proto Test TearDown" << std::endl;
    }
};

TEST_F(GatherPaKvCache, GatherPaKvCache_infershape_case0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape keyCache_shape = {{30, 101, 128, 32}, {30, 101, 128, 32}};
    gert::StorageShape valueCache_shape = {{30, 127, 128, 32}, {30, 127, 128, 32}};
    gert::StorageShape blockTables_shape = {{36, 8}, {36, 8}};
    gert::StorageShape seqLens_shape = {{36}, {36}};
    gert::StorageShape key_shape = {{18933, 3232}, {18933, 3232}};
    gert::StorageShape value_shape = {{18933, 4064}, {18933, 4064}};
    gert::StorageShape seqOffset_shape = {{36}, {36}};
    gert::StorageShape keyOutShape;
    gert::StorageShape valueOutShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&keyCache_shape, &valueCache_shape, &blockTables_shape, &seqLens_shape, &key_shape,
                                    &value_shape, &seqOffset_shape})
                      .OutputShapes({&keyOutShape, &valueOutShape})
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                      .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"cacheMode", ge::AnyValue::CreateFrom<std::string>("PA_NZ")},
                                  {"isSeqLensCumsum", ge::AnyValue::CreateFrom<bool>(false)}})
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(ge::Shape2String(*output0), "[18933, 3232]");
    ASSERT_EQ(ge::Shape2String(*output1), "[18933, 4064]");
}

TEST_F(GatherPaKvCache, GatherPaKvCache_infershape_case1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape keyCache_shape = {{128, 128, 16, 144}, {128, 128, 16, 144}};
    gert::StorageShape valueCache_shape = {{128, 128, 16, 128}, {128, 128, 16, 128}};
    gert::StorageShape blockTables_shape = {{16, 12}, {16, 12}};
    gert::StorageShape seqLens_shape = {{16}, {16}};
    gert::StorageShape key_shape = {{8931, 16, 144}, {8931, 16, 144}};
    gert::StorageShape value_shape = {{8931, 16, 128}, {8931, 16, 128}};
    gert::StorageShape seqOffset_shape = {{16}, {16}};
    gert::StorageShape keyOutShape;
    gert::StorageShape valueOutShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&keyCache_shape, &valueCache_shape, &blockTables_shape, &seqLens_shape, &key_shape,
                                    &value_shape, &seqOffset_shape})
                      .OutputShapes({&keyOutShape, &valueOutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"cacheMode", ge::AnyValue::CreateFrom<std::string>("Norm")},
                                  {"isSeqLensCumsum", ge::AnyValue::CreateFrom<bool>(false)}})
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(ge::Shape2String(*output0), "[8931, 16, 144]");
    ASSERT_EQ(ge::Shape2String(*output1), "[8931, 16, 128]");
}

TEST_F(GatherPaKvCache, GatherPaKvCache_inferdtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType inputDtype = ge::DT_INT8;
    ge::DataType inputReDtype = ge::DT_INT32;
    ge::DataType outDtype = ge::DT_INT8;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(7)
                              .NodeIoNum(7, 2)
                              .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                              .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                              .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputDtype, &inputDtype, &inputReDtype, &inputReDtype, &inputDtype,
                                               &inputDtype, &inputReDtype})
                              .OutputDataTypes({&outDtype, &outDtype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(0), outDtype);
    EXPECT_EQ(context->GetOutputDataType(1), outDtype);
}


TEST_F(GatherPaKvCache, GatherPaKvCache_infershape_case_unknownrank)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape keyCache_shape = {{-2}, {30, 101, 128, 32}};
    gert::StorageShape valueCache_shape = {{-2}, {30, 127, 128, 32}};
    gert::StorageShape blockTables_shape = {{-2}, {36, 8}};
    gert::StorageShape seqLens_shape = {{-2}, {36}};
    gert::StorageShape key_shape = {{-2}, {18933, 3232}};
    gert::StorageShape value_shape = {{-2}, {18933, 4064}};
    gert::StorageShape seqOffset_shape = {{-2}, {36}};
    gert::StorageShape keyOutShape;
    gert::StorageShape valueOutShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&keyCache_shape, &valueCache_shape, &blockTables_shape, &seqLens_shape, &key_shape,
                                    &value_shape, &seqOffset_shape})
                      .OutputShapes({&keyOutShape, &valueOutShape})
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                      .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"cacheMode", ge::AnyValue::CreateFrom<std::string>("PA_NZ")},
                                  {"isSeqLensCumsum", ge::AnyValue::CreateFrom<bool>(false)}})
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(ge::Shape2String(*output0), "[-2]");
    ASSERT_EQ(ge::Shape2String(*output1), "[-2]");
}


TEST_F(GatherPaKvCache, GatherPaKvCache_infershape_case_unknownshape)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape keyCache_shape = {{-1, -1, -1, -1}, {128, 128, 16, 144}};
    gert::StorageShape valueCache_shape = {{-1, -1, -1, -1}, {128, 128, 16, 128}};
    gert::StorageShape blockTables_shape = {{-1, -1}, {16, 12}};
    gert::StorageShape seqLens_shape = {{-1}, {16}};
    gert::StorageShape key_shape = {{-1, -1, -1}, {8931, 16, 144}};
    gert::StorageShape value_shape = {{-1, -1, -1}, {8931, 16, 128}};
    gert::StorageShape seqOffset_shape = {{-1}, {16}};
    gert::StorageShape keyOutShape;
    gert::StorageShape valueOutShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&keyCache_shape, &valueCache_shape, &blockTables_shape, &seqLens_shape, &key_shape,
                                    &value_shape, &seqOffset_shape})
                      .OutputShapes({&keyOutShape, &valueOutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"cacheMode", ge::AnyValue::CreateFrom<std::string>("Norm")},
                                  {"isSeqLensCumsum", ge::AnyValue::CreateFrom<bool>(false)}})
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(ge::Shape2String(*output0), "[-1, -1, -1]");
    ASSERT_EQ(ge::Shape2String(*output1), "[-1, -1, -1]");
}


TEST_F(GatherPaKvCache, GatherPaKvCache_infershape_case_fail_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape keyCache_shape = {{128, 128, 16, 144}, {128, 128, 16, 144}};
    gert::StorageShape valueCache_shape = {{128, 128, 16, 128}, {128, 128, 16, 128}};
    gert::StorageShape blockTables_shape = {{16, 12}, {16, 12}};
    gert::StorageShape seqLens_shape = {{16}, {16}};
    gert::StorageShape key_shape = {{8931, 16, 144}, {8931, 16, 144}};
    gert::StorageShape value_shape = {{8931, 16, 128}, {8931, 16, 128}};
    gert::StorageShape seqOffset_shape = {{16}, {16}};
    gert::StorageShape keyOutShape;
    gert::StorageShape valueOutShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&keyCache_shape, &valueCache_shape, &blockTables_shape, &seqLens_shape, &key_shape,
                                    &value_shape, &seqOffset_shape})
                      .OutputShapes({&keyOutShape, &valueOutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"cacheMode", ge::AnyValue::CreateFrom<std::string>("Norm")},
                                  {"isSeqLensCumsum", ge::AnyValue::CreateFrom<bool>(true)}})
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(ge::Shape2String(*output0), "[8931, 16, 144]");
    ASSERT_EQ(ge::Shape2String(*output1), "[8931, 16, 128]");
}

TEST_F(GatherPaKvCache, GatherPaKvCache_infershape_case_fail_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape keyCache_shape = {{128, 128, 16, 144}, {128, 128, 16, 144}};
    gert::StorageShape valueCache_shape = {{128, 128, 16, 128}, {128, 128, 16, 128}};
    gert::StorageShape blockTables_shape = {{16, 12}, {16, 12}};
    gert::StorageShape seqLens_shape = {{1}, {1}};
    gert::StorageShape key_shape = {{8931, 16, 144}, {8931, 16, 144}};
    gert::StorageShape value_shape = {{8931, 16, 128}, {8931, 16, 128}};
    gert::StorageShape seqOffset_shape = {{16}, {16}};
    gert::StorageShape keyOutShape;
    gert::StorageShape valueOutShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&keyCache_shape, &valueCache_shape, &blockTables_shape, &seqLens_shape, &key_shape,
                                    &value_shape, &seqOffset_shape})
                      .OutputShapes({&keyOutShape, &valueOutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"cacheMode", ge::AnyValue::CreateFrom<std::string>("Norm")},
                                  {"isSeqLensCumsum", ge::AnyValue::CreateFrom<bool>(false)}})
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(ge::Shape2String(*output0), "[8931, 16, 144]");
    ASSERT_EQ(ge::Shape2String(*output1), "[8931, 16, 128]");
}

TEST_F(GatherPaKvCache, GatherPaKvCache_infershape_case_fail_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherPaKvCache")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape keyCache_shape = {{128, 128, 16, 144}, {128, 128, 16, 144}};
    gert::StorageShape valueCache_shape = {{100, 100, 16, 128}, {100, 100, 16, 128}};
    gert::StorageShape blockTables_shape = {{16, 12}, {16, 12}};
    gert::StorageShape seqLens_shape = {{16}, {16}};
    gert::StorageShape key_shape = {{8931, 16, 144}, {8931, 16, 144}};
    gert::StorageShape value_shape = {{8931, 16, 128}, {8931, 16, 128}};
    gert::StorageShape seqOffset_shape = {{16}, {16}};
    gert::StorageShape keyOutShape;
    gert::StorageShape valueOutShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&keyCache_shape, &valueCache_shape, &blockTables_shape, &seqLens_shape, &key_shape,
                                    &value_shape, &seqOffset_shape})
                      .OutputShapes({&keyOutShape, &valueOutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"cacheMode", ge::AnyValue::CreateFrom<std::string>("Norm")},
                                  {"isSeqLensCumsum", ge::AnyValue::CreateFrom<bool>(false)}})
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(ge::Shape2String(*output0), "[8931, 16, 144]");
    ASSERT_EQ(ge::Shape2String(*output1), "[8931, 16, 128]");
}