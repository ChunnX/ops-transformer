/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE file in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_roi_extractor_tiling.cpp
 * \brief Tiling UT: level-slot collection, shape/dtype/attr validation,
 * TilingKey (bit0 = max) and tiling-data fields.
 */
#include <gtest/gtest.h>
#include <iostream>

#include "../../../op_host/roi_extractor_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr uint64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024UL;
// Absent optional inputs are expressed as 0-dim storage shapes (framework faker convention).
const gert::StorageShape EMPTY_SHAPE = {{}, {}};

std::vector<float> DefaultScales(uint64_t levels)
{
    const std::vector<float> all = {0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f};
    return std::vector<float>(all.begin(), all.begin() + levels);
}

gert::TilingContextPara BuildCase(const std::vector<gert::StorageShape> &feats, ge::DataType dtype,
                                  const gert::StorageShape &roisShape, const gert::StorageShape &indexShape,
                                  int64_t pooledHeight, int64_t pooledWidth, const std::string &poolMode,
                                  int64_t sampleNum, bool aligned,
                                  const std::vector<gert::TilingContextPara::OpAttr> &extraAttrs = {})
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    for (const auto &f : feats) {
        inputs.push_back({f, dtype, ge::FORMAT_ND});
    }
    for (uint64_t i = feats.size(); i < 5; i++) {
        inputs.push_back({EMPTY_SHAPE, dtype, ge::FORMAT_ND});
    }
    inputs.push_back({roisShape, dtype, ge::FORMAT_ND});
    inputs.push_back({indexShape, ge::DT_INT32, ge::FORMAT_ND});

    int64_t roisNum = roisShape.GetStorageShape().GetDim(0);
    int64_t channels = feats.empty() ? 1 : feats[0].GetStorageShape().GetDim(1);
    gert::StorageShape yShape({roisNum, channels, pooledHeight, pooledWidth},
                              {roisNum, channels, pooledHeight, pooledWidth});
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.push_back({yShape, dtype, ge::FORMAT_ND});

    // extra attrs may override spatial_scale by value; index order must stay fixed
    std::vector<float> scales = DefaultScales(feats.size());
    for (const auto &a : extraAttrs) {
        if (a.attrName_ == "spatial_scale") {
            scales = *reinterpret_cast<const std::vector<float> *>(a.attr_.valuePtr_.get());
        }
    }
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"finest_scale", Ops::Transformer::AnyValue::CreateFrom<int64_t>(56)},
        {"roi_scale_factor", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
        {"spatial_scale", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>(scales)},
        {"pooled_height", Ops::Transformer::AnyValue::CreateFrom<int64_t>(pooledHeight)},
        {"pooled_width", Ops::Transformer::AnyValue::CreateFrom<int64_t>(pooledWidth)},
        {"sample_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sampleNum)},
        {"pool_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(poolMode)},
        {"aligned", Ops::Transformer::AnyValue::CreateFrom<bool>(aligned)},
    };

    static optiling::RoiExtractorCompileInfo compileInfo = {};
    compileInfo.coreNum = 50;
    compileInfo.ubSize = 262144;
    compileInfo.is_ascendc = true;
    return gert::TilingContextPara("RoiExtractor", inputs, outputs, attrs, &compileInfo, "Ascend950",
                                   compileInfo.coreNum, compileInfo.ubSize);
}
} // namespace

class RoiExtractorTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RoiExtractorTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RoiExtractorTiling TearDown" << std::endl;
    }
};

TEST_F(RoiExtractorTilingTest, FourLevelsFp32Avg)
{
    std::vector<gert::StorageShape> feats = {
        {{2, 256, 200, 336}, {2, 256, 200, 336}},
        {{2, 256, 100, 168}, {2, 256, 100, 168}},
        {{2, 256, 50, 84}, {2, 256, 50, 84}},
        {{2, 256, 25, 42}, {2, 256, 25, 42}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{8, 5}, {8, 5}}, EMPTY_SHAPE, 7, 7, "avg", 0, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "", {SYS_WORKSPACE_SIZE});
}

TEST_F(RoiExtractorTilingTest, FourLevelsFp16MaxKey)
{
    std::vector<gert::StorageShape> feats = {
        {{2, 64, 200, 336}, {2, 64, 200, 336}},
        {{2, 64, 100, 168}, {2, 64, 100, 168}},
        {{2, 64, 50, 84}, {2, 64, 50, 84}},
        {{2, 64, 25, 42}, {2, 64, 25, 42}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT16, {{16, 5}, {16, 5}}, EMPTY_SHAPE, 7, 7, "max", 0, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 1, "", {SYS_WORKSPACE_SIZE});
}

TEST_F(RoiExtractorTilingTest, SingleLevelDegenerate)
{
    std::vector<gert::StorageShape> feats = {
        {{1, 32, 64, 64}, {1, 32, 64, 64}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{3, 5}, {3, 5}}, EMPTY_SHAPE, 7, 7, "avg", 0, false);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "", {SYS_WORKSPACE_SIZE});
}

TEST_F(RoiExtractorTilingTest, RoiWithIndexAndNonSquarePool)
{
    std::vector<gert::StorageShape> feats = {
        {{2, 32, 64, 64}, {2, 32, 64, 64}},
        {{2, 32, 32, 32}, {2, 32, 32, 32}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{5, 4}, {5, 4}}, {{5}, {5}}, 7, 14, "avg", 2, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "", {SYS_WORKSPACE_SIZE});
}

TEST_F(RoiExtractorTilingTest, EmptyRois)
{
    std::vector<gert::StorageShape> feats = {
        {{2, 32, 64, 64}, {2, 32, 64, 64}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{0, 5}, {0, 5}}, EMPTY_SHAPE, 7, 7, "avg", 0, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "", {SYS_WORKSPACE_SIZE});
}

TEST_F(RoiExtractorTilingTest, FiveLevels)
{
    std::vector<gert::StorageShape> feats = {
        {{1, 16, 128, 128}, {1, 16, 128, 128}},
        {{1, 16, 64, 64}, {1, 16, 64, 64}},
        {{1, 16, 32, 32}, {1, 16, 32, 32}},
        {{1, 16, 16, 16}, {1, 16, 16, 16}},
        {{1, 16, 8, 8}, {1, 16, 8, 8}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{4, 5}, {4, 5}}, EMPTY_SHAPE, 7, 7, "avg", 0, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "", {SYS_WORKSPACE_SIZE});
}

TEST_F(RoiExtractorTilingTest, SlotGapRejected)
{
    // feats0 present, feats1 absent, feats2 present -> contiguous-prefix violation
    std::vector<gert::StorageShape> feats = {
        {{1, 8, 32, 32}, {1, 8, 32, 32}},
        EMPTY_SHAPE,
        {{1, 8, 8, 8}, {1, 8, 8, 8}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{2, 5}, {2, 5}}, EMPTY_SHAPE, 7, 7, "avg", 0, true);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(RoiExtractorTilingTest, LevelChannelMismatchRejected)
{
    std::vector<gert::StorageShape> feats = {
        {{1, 8, 32, 32}, {1, 8, 32, 32}},
        {{1, 16, 16, 16}, {1, 16, 16, 16}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{2, 5}, {2, 5}}, EMPTY_SHAPE, 7, 7, "avg", 0, true);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(RoiExtractorTilingTest, DtypeMismatchRejected)
{
    std::vector<gert::StorageShape> feats = {
        {{1, 8, 32, 32}, {1, 8, 32, 32}},
    };
    // feats fp32 vs rois fp16: override the rois slot (index 5) dtype after build
    auto para = BuildCase(feats, ge::DT_FLOAT, {{2, 5}, {2, 5}}, EMPTY_SHAPE, 7, 7, "avg", 0, true);
    para.inputTensorDesc_[5].dtype_ = ge::DT_FLOAT16;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(RoiExtractorTilingTest, BadRoisLastDimRejected)
{
    std::vector<gert::StorageShape> feats = {
        {{1, 8, 32, 32}, {1, 8, 32, 32}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{2, 6}, {2, 6}}, EMPTY_SHAPE, 7, 7, "avg", 0, true);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(RoiExtractorTilingTest, Roi4WithoutIndexRejected)
{
    std::vector<gert::StorageShape> feats = {
        {{1, 8, 32, 32}, {1, 8, 32, 32}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{2, 4}, {2, 4}}, EMPTY_SHAPE, 7, 7, "avg", 0, true);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(RoiExtractorTilingTest, BadSampleNumRejected)
{
    std::vector<gert::StorageShape> feats = {
        {{1, 8, 32, 32}, {1, 8, 32, 32}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{2, 5}, {2, 5}}, EMPTY_SHAPE, 7, 7, "avg", -1, true);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(RoiExtractorTilingTest, BadPoolModeRejected)
{
    std::vector<gert::StorageShape> feats = {
        {{1, 8, 32, 32}, {1, 8, 32, 32}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{2, 5}, {2, 5}}, EMPTY_SHAPE, 7, 7, "median", 0, true);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(RoiExtractorTilingTest, BadPooledSizeRejected)
{
    std::vector<gert::StorageShape> feats = {
        {{1, 8, 32, 32}, {1, 8, 32, 32}},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{2, 5}, {2, 5}}, EMPTY_SHAPE, 64, 7, "avg", 0, true);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(RoiExtractorTilingTest, SpatialScaleLengthMismatchRejected)
{
    std::vector<gert::StorageShape> feats = {
        {{1, 8, 32, 32}, {1, 8, 32, 32}},
        {{1, 8, 16, 16}, {1, 8, 16, 16}},
    };
    std::vector<gert::TilingContextPara::OpAttr> extra = {
        {"spatial_scale",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>(std::vector<float>{0.25f, 0.125f, 0.0625f})},
    };
    auto para = BuildCase(feats, ge::DT_FLOAT, {{2, 5}, {2, 5}}, EMPTY_SHAPE, 7, 7, "avg", 0, true, extra);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
