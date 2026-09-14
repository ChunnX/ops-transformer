/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Infer-shape and infer-dtype UT for GroupedMatmulSwigluQuantV3.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

namespace {

constexpr int64_t kM = 8;
constexpr int64_t kN = 256;

vector<gert::InfershapeContextPara::OpAttr> MakeAttrs(int64_t quantMode, int64_t quantDtype, bool transpose)
{
    return {
        {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
        {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantDtype)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transpose)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0})},
        {"activate_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"activate_left", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"swiglu_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"clamp_limit", Ops::Transformer::AnyValue::CreateFrom<float>(7.0F)},
        {"glu_alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.702F)},
        {"glu_bias", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
        {"axis", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"round_mode", Ops::Transformer::AnyValue::CreateFrom<string>("rint")},
        {"scale_alg", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dst_type_max", Ops::Transformer::AnyValue::CreateFrom<float>(0.0F)},
    };
}

gert::InfershapeContextPara MakeShapeContext(int64_t k, bool transpose, int64_t quantMode = 2,
                                             int64_t quantDtype = 0)
{
    const int64_t kScale = (k + 63) / 64;
    const vector<int64_t> weight = transpose ? vector<int64_t>{1, kN, k} : vector<int64_t>{1, k, kN};
    const vector<int64_t> weightScale =
        transpose ? vector<int64_t>{1, kN, kScale, 2} : vector<int64_t>{1, kScale, kN, 2};
    const vector<int64_t> weightStorage =
        transpose ? vector<int64_t>{1, (k + 31) / 32, (kN + 15) / 16, 16, 32}
                  : vector<int64_t>{1, (k + 15) / 16, (kN + 31) / 32, 16, 32};

    const vector<int64_t> xShape = {kM, k};
    const vector<int64_t> xScaleShape = {kM, kScale, 2};
    const vector<int64_t> groupListShape = {1};
    vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {ops::ut::MakeGertStorageShape(xShape), DT_FLOAT8_E4M3FN, FORMAT_ND},
        {ops::ut::MakeGertStorageShape(xScaleShape), DT_FLOAT8_E8M0, FORMAT_ND},
        {ops::ut::MakeGertStorageShape(groupListShape), DT_INT64, FORMAT_ND},
        {ops::ut::MakeGertStorageShape(weight, weightStorage), DT_FLOAT8_E4M3FN, FORMAT_FRACTAL_NZ},
        {ops::ut::MakeGertStorageShape(weightScale), DT_FLOAT8_E8M0, FORMAT_ND},
    };
    const vector<int64_t> emptyOutputShape = {0};
    vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {ops::ut::MakeGertStorageShape(emptyOutputShape), DT_FLOAT8_E4M3FN, FORMAT_ND},
        {ops::ut::MakeGertStorageShape(emptyOutputShape), DT_FLOAT8_E8M0, FORMAT_ND},
    };
    return gert::InfershapeContextPara("GroupedMatmulSwigluQuantV3", inputs, outputs,
                                       MakeAttrs(quantMode, quantDtype, transpose),
                                       {1, 1, 1, 1, 1, 0, 0, 0}, {1, 1});
}

TEST(GroupedMatmulSwigluQuantV3InferShape, InfersNzAndZnOutputShapes)
{
    for (bool transpose : {false, true}) {
        auto context = MakeShapeContext(128, transpose);
        ExecuteTestCase(context, GRAPH_SUCCESS, {{kM, kN / 2}, {kM, (kN / 2 + 63) / 64, 2}});
    }
}

TEST(GroupedMatmulSwigluQuantV3InferShape, InfersUnalignedKOutputShapes)
{
    auto context = MakeShapeContext(64, false);
    ExecuteTestCase(context, GRAPH_SUCCESS, {{kM, kN / 2}, {kM, (kN / 2 + 63) / 64, 2}});
}

TEST(GroupedMatmulSwigluQuantV3InferShape, RejectsNonMxQuantMode)
{
    auto context = MakeShapeContext(128, false, 0);
    ExecuteTestCase(context, GRAPH_FAILED);
}

TEST(GroupedMatmulSwigluQuantV3InferDataType, InfersOutputTypes)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto opImpl = registry->GetOpImpl("GroupedMatmulSwigluQuantV3");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    ge::DataType x = DT_FLOAT8_E4M3FN;
    ge::DataType xScale = DT_FLOAT8_E8M0;
    ge::DataType groupList = DT_INT64;
    ge::DataType weight = DT_FLOAT8_E4M3FN;
    ge::DataType weightScale = DT_FLOAT8_E8M0;
    ge::DataType optional = DT_FLOAT;
    vector<void *> inputTypes = {&x, &xScale, &groupList, &weight, &weightScale, &optional, &optional, &optional};
    // quant_dtype carries the output ge::DataType for this operator, rather
    // than the enum used by quant_mode.
    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType("GroupedMatmulSwigluQuantV3")
                      .NodeIoNum(8, 2)
                      .NodeOutputTd(0, FORMAT_ND, FORMAT_ND)
                      .NodeOutputTd(1, FORMAT_ND, FORMAT_ND)
                      .InputDataTypes(inputTypes)
                      .NodeAttrs({
                          {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                          {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                          {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                          {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(DT_FLOAT8_E4M3FN)},
                      })
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(context), GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), DT_FLOAT8_E4M3FN);
    EXPECT_EQ(context->GetOutputDataType(1), DT_FLOAT8_E8M0);
}

} // namespace
