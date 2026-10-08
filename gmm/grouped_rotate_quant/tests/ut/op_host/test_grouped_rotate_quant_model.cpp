/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "gtest/gtest.h"
#include "../../../op_host/op_tiling/arch35/grouped_rotate_quant_planner.h"

namespace {
using namespace optiling::grq;
InputParams Independent()
{
    InputParams p;
    p.x = {{5, 128}, {3, 96}};
    p.rotation = {{64, 64}, {3, 32, 32}};
    p.yNum = p.scaleNum = 2;
    return p;
}
InputParams Grouped(int64_t splitItem = 3)
{
    InputParams p;
    p.attrs.groupType = 0;
    p.attrs.splitItem = splitItem;
    p.x = {{13, 96}};
    p.rotation = {{32, 32}};
    p.hasGroupList = true;
    p.groupList = {3};
    p.yNum = p.scaleNum = splitItem < 2 ? 3 : 1;
    return p;
}
Hardware Hardware950()
{
    return {24, 48, 512 * 1024, 64 * 1024, 64 * 1024, 256 * 1024, 256 * 1024};
}
bool Valid(InputParams &p, bool symbolic = false)
{
    std::string error;
    return ValidateInputs(p, symbolic, error);
}
bool Plan(InputParams &p, GroupedRotateQuant::TilingData &d, Hardware h = Hardware950())
{
    std::string error;
    return ValidateInputs(p, false, error) && BuildPlan(p, h, d, error);
}

TEST(GroupedRotateQuantModel, IndependentHeterogeneousShapesAndRotations)
{
    auto p = Independent();
    ASSERT_TRUE(Valid(p));
    EXPECT_EQ(p.totalM, 8);
    EXPECT_EQ(p.totalElements, 928);
    EXPECT_EQ(InferY(p), p.x);
    EXPECT_EQ(ScaleShape(p.x[1]), (Shape{3, 2, 2}));
    GroupedRotateQuant::TilingData d{};
    ASSERT_TRUE(Plan(p, d));
    EXPECT_EQ(d.tiles[0].mTile, 16U);
    EXPECT_EQ(d.tiles[1].mTile, 16U);
    EXPECT_EQ(d.tileCount, 4U); // one shared-rotation tile + three column-rotation tiles
    EXPECT_EQ(d.usedCoreNum, 4U);
}
TEST(GroupedRotateQuantModel, SingleInputSeparateOutputsRemainSymbolic)
{
    auto p = Grouped(1);
    ASSERT_TRUE(Valid(p));
    EXPECT_TRUE(p.runtimeM);
    EXPECT_EQ(InferY(p), (std::vector<Shape>{{-1, 96}, {-1, 96}, {-1, 96}}));
    std::string error;
    EXPECT_TRUE(ValidateOutputs(p, {{0, 96}, {4, 96}, {9, 96}}, {{0, 2, 2}, {4, 2, 2}, {9, 2, 2}}, error));
    EXPECT_FALSE(ValidateOutputs(p, {{0, 96}, {4, 96}, {10, 96}}, {{0, 2, 2}, {4, 2, 2}, {10, 2, 2}}, error));
}
TEST(GroupedRotateQuantModel, AllSplitModesAndNormalization)
{
    for (int64_t split = 0; split < 4; ++split) {
        auto p = Grouped(split);
        if (split == 0 || split == 2) {
            p.x = {{0, 96}, {4, 96}, {9, 96}};
        }
        GroupedRotateQuant::TilingData d{};
        ASSERT_TRUE(Plan(p, d));
        EXPECT_EQ(d.splitItem, split);
        EXPECT_EQ(d.runtimeM, split == 1 || split == 3);
        const auto y = InferY(p);
        EXPECT_EQ(y.size(), split < 2 ? 3U : 1U);
        if (split >= 2) {
            EXPECT_EQ(y[0], (Shape{13, 96}));
        }
    }
}
TEST(GroupedRotateQuantModel, SingleGroupDegenerateSplit)
{
    auto p = Grouped(1);
    p.groupList = {1};
    p.yNum = p.scaleNum = 1;
    GroupedRotateQuant::TilingData d{};
    ASSERT_TRUE(Plan(p, d));
    EXPECT_EQ(d.splitItem, 0U);
    EXPECT_EQ(InferY(p), p.x);
}
TEST(GroupedRotateQuantModel, MultiTensorGroupsMayOmitGroupList)
{
    auto p = Grouped(2);
    p.x = {{0, 96}, {4, 96}, {9, 96}};
    p.hasGroupList = false;
    GroupedRotateQuant::TilingData d{};
    ASSERT_TRUE(Plan(p, d));
    EXPECT_EQ(d.groupNum, 3U);
    EXPECT_EQ(d.hasGroupList, 0U);
    EXPECT_EQ(d.tileCount, 3U); // 0 + ceil(12/16) + ceil(27/16)
}
TEST(GroupedRotateQuantModel, HintNeverBecomesRuntimeShapeOrTileCount)
{
    auto p = Grouped();
    p.x = {{10000, 96}};
    GroupedRotateQuant::TilingData a{};
    GroupedRotateQuant::TilingData b{};
    ASSERT_TRUE(Plan(p, a));
    p.attrs.expectedM = 1;
    ASSERT_TRUE(Plan(p, b));
    EXPECT_NE(a.tiles[0].mTile, b.tiles[0].mTile);
    EXPECT_EQ(a.usedCoreNum, 24U);
    EXPECT_EQ(b.usedCoreNum, 24U);
    EXPECT_EQ(a.totalM, b.totalM);
    EXPECT_EQ(a.totalElements, b.totalElements);
    EXPECT_EQ(a.tileCount, 0U);
    EXPECT_EQ(b.tileCount, 0U);
}
TEST(GroupedRotateQuantModel, ThousandGroupsKeepConstantTilingSize)
{
    auto p = Grouped();
    p.groupList = {1024};
    p.rotation = std::vector<Shape>(1024, Shape{32, 32});
    GroupedRotateQuant::TilingData d{};
    ASSERT_TRUE(Plan(p, d));
    EXPECT_EQ(d.groupNum, 1024U);
    EXPECT_EQ(sizeof(d), 128U);
}
TEST(GroupedRotateQuantModel, WholeTileFitsEveryBuffer)
{
    auto p = Independent();
    p.x = {{100000, 128}, {100000, 128}, {100000, 128}};
    p.rotation = {{32, 32}, {64, 64}, {128, 128}};
    p.yNum = p.scaleNum = 3;
    auto h = Hardware950();
    GroupedRotateQuant::TilingData d{};
    ASSERT_TRUE(Plan(p, d, h));
    for (const auto &t : d.tiles) {
        const uint64_t m = t.mTile;
        const uint64_t r = t.rotationSize;
        EXPECT_EQ(m % 16, 0U);
        EXPECT_LE(m * r * 4, h.l0ASize);
        EXPECT_LE(r * r * 4, h.l0BSize);
        EXPECT_LE(m * r * 8, h.l0CSize);
        EXPECT_LE(m * r * 4 + r * r * 2, h.l1Size);
        EXPECT_LE(m * r, 16384U);
        EXPECT_LE(52736U + m * 32, h.ubSize);
    }
    EXPECT_EQ(d.tiles[0].mTile, 512U);
    EXPECT_EQ(d.tiles[1].mTile, 256U);
    EXPECT_EQ(d.tiles[2].mTile, 128U);
}
TEST(GroupedRotateQuantModel, HardwareLimitsAndTinyResourceRejection)
{
    auto p = Grouped();
    GroupedRotateQuant::TilingData d{};
    auto h = Hardware950();
    h.ubSize = EPILOGUE_FIXED_BYTES + 16 * 32;
    ASSERT_TRUE(Plan(p, d, h));
    EXPECT_EQ(d.tiles[0].mTile, 16U);
    h.ubSize--;
    EXPECT_FALSE(Plan(p, d, h));
    h = Hardware950();
    h.l1Size = 2048;
    EXPECT_FALSE(Plan(p, d, h));
    h = Hardware950();
    h.l0BSize = 4095;
    EXPECT_FALSE(Plan(p, d, h));
    h = Hardware950();
    h.aivNum = 24;
    EXPECT_FALSE(Plan(p, d, h));
    h.aicNum = 0;
    EXPECT_FALSE(Plan(p, d, h));
}
TEST(GroupedRotateQuantModel, EmptyInputsAndMixedEmptyGroups)
{
    auto p = Independent();
    p.x = {{0, 128}, {7, 0}};
    p.rotation = {{64, 64}, {0, 32, 32}};
    GroupedRotateQuant::TilingData d{};
    ASSERT_TRUE(Plan(p, d));
    EXPECT_EQ(d.totalElements, 0U);
    EXPECT_EQ(d.tileCount, 0U);
    EXPECT_EQ(d.usedCoreNum, 1U);
    p.x[0] = {3, 128};
    ASSERT_TRUE(Plan(p, d));
    EXPECT_EQ(d.tileCount, 1U);
}
TEST(GroupedRotateQuantModel, ScalarVectorAndHighRankInference)
{
    auto p = Independent();
    p.x = {{96}, {2, 1, 2, 1, 1, 3, 128}};
    p.rotation = {{32, 32}, {64, 64}};
    ASSERT_TRUE(Valid(p));
    EXPECT_EQ(p.totalM, 13);
    EXPECT_EQ(ScaleShape(p.x[0]), (Shape{2, 2}));
    EXPECT_EQ(ScaleShape(p.x[1]), (Shape{2, 1, 2, 1, 1, 3, 2, 2}));
    p.x[0] = {};
    EXPECT_FALSE(Valid(p));
}
TEST(GroupedRotateQuantModel, DynamicDimensionsArePreservedByInference)
{
    auto p = Independent();
    p.x = {{-1, 128}, {3, -1}};
    p.rotation = {{64, 64}, {-1, 32, 32}};
    ASSERT_TRUE(Valid(p, true));
    EXPECT_EQ(ScaleShape(p.x[1]), (Shape{3, -1, 2}));
    EXPECT_FALSE(Valid(p));
}
TEST(GroupedRotateQuantModel, RotationBatchIsColumnBlocksNotExpertCount)
{
    auto p = Grouped();
    p.rotation = {{3, 32, 32}};
    ASSERT_TRUE(Valid(p));
    p.groupList = {2};
    EXPECT_TRUE(Valid(p));
    p.rotation = {{2, 32, 32}};
    EXPECT_FALSE(Valid(p));
}
TEST(GroupedRotateQuantModel, UnsupportedGroupingAndMissingGroupList)
{
    auto p = Grouped();
    p.attrs.groupType = 2;
    EXPECT_FALSE(Valid(p));
    p.attrs.groupType = 1;
    EXPECT_FALSE(Valid(p));
    p.attrs.groupType = 0;
    p.attrs.groupListType = 2;
    EXPECT_FALSE(Valid(p));
    p.attrs.groupListType = 0;
    p.hasGroupList = false;
    EXPECT_FALSE(Valid(p));
}
TEST(GroupedRotateQuantModel, RejectListShapeAxisAndOutputErrors)
{
    auto p = Independent();
    p.scaleNum = 1;
    EXPECT_FALSE(Valid(p));
    p = Independent();
    p.attrs.axis = 0;
    EXPECT_FALSE(Valid(p));
    p = Independent();
    p.rotation[0] = {32, 64};
    EXPECT_FALSE(Valid(p));
    p = Independent();
    ASSERT_TRUE(Valid(p));
    std::string error;
    EXPECT_FALSE(ValidateOutputs(p, p.x, {{5, 2, 2}, {3, 3, 2}}, error));
}
TEST(GroupedRotateQuantModel, QuantizationAttributes)
{
    auto p = Independent();
    p.attrs.scaleAlg = 2;
    p.attrs.dstTypeMax = 7;
    p.attrs.roundMode = "floor";
    p.hasAlpha = true;
    p.alpha = {1};
    GroupedRotateQuant::TilingData d{};
    ASSERT_TRUE(Plan(p, d));
    EXPECT_EQ(d.roundMode, 1U);
    EXPECT_EQ(d.invDstTypeMax, 1.0f / 7);
    EXPECT_EQ(d.hasAlpha, 1U);
    p.attrs.outputType = OutputType::FP8_E4M3;
    EXPECT_FALSE(Valid(p));
    p.attrs.scaleAlg = 1;
    p.attrs.dstTypeMax = 0;
    p.attrs.roundMode = "rint";
    EXPECT_TRUE(Valid(p));
    p.attrs.dstTypeMax = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(Valid(p));
}
TEST(GroupedRotateQuantModel, OverflowAndZeroDimensions)
{
    auto p = Independent();
    p.x[0] = {std::numeric_limits<int64_t>::max(), 2, 128};
    EXPECT_FALSE(Valid(p));
    p.x[0] = {std::numeric_limits<int64_t>::max(), 2, 0, 128};
    EXPECT_TRUE(Valid(p));
    p = Grouped();
    p.attrs.expectedM = 14;
    EXPECT_FALSE(Valid(p));
}
} // namespace
