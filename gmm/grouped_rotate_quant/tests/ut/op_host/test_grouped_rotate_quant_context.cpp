/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cstring>
#include "gtest/gtest.h"
#include "tiling_case_executor.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "../../../op_host/op_tiling/arch35/grouped_rotate_quant_tiling.h"

namespace {
using Value = Ops::Transformer::AnyValue;
template <class Para>
std::vector<typename Para::OpAttr> Attributes(int64_t groupType = 0, int64_t split = 1)
{
    return {
        {"split_item", Value::CreateFrom<int64_t>(split)},
        {"group_type", Value::CreateFrom<int64_t>(groupType)},
        {"group_list_type", Value::CreateFrom<int64_t>(0)},
        {"axis", Value::CreateFrom<int64_t>(-1)},
        {"round_mode", Value::CreateFrom<std::string>("rint")},
        {"scale_alg", Value::CreateFrom<int64_t>(0)},
        {"dst_type_max", Value::CreateFrom<float>(0)},
        {"trans", Value::CreateFrom<bool>(false)},
        {"dst_type", Value::CreateFrom<int64_t>(ge::DT_FLOAT8_E4M3FN)},
        {"tuning_config", Value::CreateFrom<std::vector<int64_t>>({})}};
}
optiling::GroupedRotateQuantCompileInfo CompileInfo()
{
    optiling::GroupedRotateQuantCompileInfo info;
    info.npuArch = static_cast<int32_t>(NpuArch::DAV_3510);
    info.hardware = {24, 48, 512 * 1024, 64 * 1024, 64 * 1024, 256 * 1024, 256 * 1024};
    return info;
}
gert::TilingContextPara GroupedCase(optiling::GroupedRotateQuantCompileInfo *info)
{
    // group_list has descriptor metadata but NO host-readable values.
    return gert::TilingContextPara("GroupedRotateQuant",
        {{{{13, 96}, {13, 96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{0, 96}, {0, 96}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{4, 96}, {4, 96}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{9, 96}, {9, 96}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{0, 2, 2}, {0, 2, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{4, 2, 2}, {4, 2, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{9, 2, 2}, {9, 2, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        Attributes<gert::TilingContextPara>(), {1, 1, 0, 1}, {3, 3},
        info, "Ascend950", 24, 256 * 1024, 128);
}
TEST(GroupedRotateQuantContext, TilingWithDeviceOnlyGroupListAndTwoDynamicOutputs)
{
    auto compileInfo = CompileInfo();
    const auto para = GroupedCase(&compileInfo);
    TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(para, result));
    EXPECT_EQ(result.tilingKey, 0);
    ASSERT_EQ(result.tilingDataSize, 128U);
    GroupedRotateQuant::TilingData data{};
    std::memcpy(&data, result.tilingData.get(), sizeof(data));
    EXPECT_EQ(data.runtimeM, 1U);
    EXPECT_EQ(data.yNum, 3U);
    EXPECT_EQ(data.hasAlpha, 0U);
    EXPECT_EQ(data.hasGroupList, 1U);
    EXPECT_EQ(data.usedCoreNum, 24U);
    EXPECT_EQ(result.workspaceSizes, (std::vector<int64_t>{16 * 1024 * 1024}));
}
TEST(GroupedRotateQuantContext, RejectSmallTilingBuffer)
{
    auto compileInfo = CompileInfo();
    auto para = GroupedCase(&compileInfo);
    para.tilingDataSize_ = 127;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
TEST(GroupedRotateQuantContext, RejectWrongScaleListCountAndDtype)
{
    auto compileInfo = CompileInfo();
    auto para = GroupedCase(&compileInfo);
    para.outputInstanceNum_ = {4, 2};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    para = GroupedCase(&compileInfo);
    para.outputTensorDesc_[4].dtype_ = ge::DT_FLOAT;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
TEST(GroupedRotateQuantContext, RejectWrongPlatformOrOutputCapacity)
{
    auto compileInfo = CompileInfo();
    auto para = GroupedCase(&compileInfo);
    compileInfo.npuArch = 0;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    compileInfo = CompileInfo();
    para.outputTensorDesc_[2].shape_ = {{10, 96}, {10, 96}};
    para.outputTensorDesc_[5].shape_ = {{10, 2, 2}, {10, 2, 2}};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
TEST(GroupedRotateQuantContext, InferSeparateOutputsWithoutReadingGroupList)
{
    using Para = gert::InfershapeContextPara;
    Para para("GroupedRotateQuant",
        {{{{13, 96}, {13, 96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        Attributes<Para>(), {1, 1, 0, 1}, {3, 3});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS,
        {{-1, 96}, {-1, 96}, {-1, 96}, {-1, 2, 2}, {-1, 2, 2}, {-1, 2, 2}});
}
TEST(GroupedRotateQuantContext, InferIndependentShapesWithSeparateScaleList)
{
    using Para = gert::InfershapeContextPara;
    Para para("GroupedRotateQuant",
        {{{{96}, {96}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 5, 128}, {2, 5, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{32, 32}, {32, 32}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        Attributes<Para>(-1, 0), {2, 1, 0, 0}, {2, 2});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{96}, {2, 5, 128}, {2, 2}, {2, 5, 2, 2}});
}
TEST(GroupedRotateQuantContext, InferDtypeUpdatesBothDynamicLists)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("GroupedRotateQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    ge::DataType xType = ge::DT_BF16;
    ge::DataType rotType = ge::DT_BF16;
    ge::DataType listType = ge::DT_INT64;
    std::vector<std::pair<std::string, Value>> attrs;
    for (const auto &attr : Attributes<gert::TilingContextPara>()) {
        attrs.emplace_back(attr.attrName_, attr.attr_);
    }
    auto holder = gert::InferDataTypeContextFaker()
        .SetOpType("GroupedRotateQuant")
        .IrInstanceNum({1, 1, 0, 1}, {3, 3})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes({&xType, &rotType, &listType})
        .NodeAttrs(attrs).Build();
    auto *context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(context->GetOutputDataType(i), ge::DT_FLOAT8_E4M3FN);
        EXPECT_EQ(context->GetOutputDataType(i + 3), ge::DT_FLOAT8_E8M0);
    }
}
} // namespace
