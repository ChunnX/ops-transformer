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
 * \file test_grouped_no_quant_arch35_attr_validation_tiling.cpp
 * \brief Unit tests for groupType and groupListType attribute validation in GroupedMatmul no-quant arch35 tiling.
 *
 * Covers three code paths modified in grouped_no_quant_matmul_tiling.cpp:
 *   1. Line 251: groupType must be in {-1, 0, 2}
 *      - Valid: NO_SPLIT(-1), SPLIT_M(0), SPLIT_K(2) → tiling succeeds
 *      - Invalid: groupType=1 → tiling fails
 *   2. Line 317: In no-quant case, groupListType must be in {0, 1}
 *      - Valid: CUMSUM(0), COUNT(1) → tiling succeeds
 *      - Invalid: SPARSE_M(2) → tiling fails
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

namespace {

gert::StorageShape MakeEmptyShape()
{
    return gert::StorageShape();
}

vector<gert::TilingContextPara::OpAttr> GetAttrs(int64_t splitItem, bool transposeX, bool transposeWeight,
                                                  int64_t groupType, int64_t groupListType)
{
    return {
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(splitItem)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    };
}

optiling::GMMCompileInfo MakeCompileInfo()
{
    return {
        32,                                 // aicNum
        64,                                 // aivNum
        262144,                             // ubSize
        524288,                             // l1Size
        134217728,                          // l2Size
        262144,                             // l0CSize
        65536,                              // l0ASize
        65536,                              // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // socVersion
        NpuArch::DAV_3510,                  // npuArch
    };
}

bool RunTiling(int64_t groupType, int64_t splitItem, int64_t groupListType)
{
    optiling::GMMCompileInfo compileInfo = MakeCompileInfo();

    const int64_t m = 256;
    const int64_t k = 1024;
    const int64_t n = 512;
    const int64_t groupNum = 2;

    gert::StorageShape xShape = ops::ut::MakeGertStorageShape(vector<int64_t>{m, k});
    gert::StorageShape weightShape = ops::ut::MakeGertStorageShape(vector<int64_t>{groupNum, k, n});
    gert::StorageShape groupListShape = ops::ut::MakeGertStorageShape(vector<int64_t>{groupNum});
    gert::StorageShape yShape = ops::ut::MakeGertStorageShape(vector<int64_t>{m, n});

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},
            {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        GetAttrs(splitItem, false, false, groupType, groupListType),
        {1U, 1U, 0U, 0U, 0U, 0U, 0U, 1U, 0U},
        {1U},
        &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo tilingInfo;
    return ExecuteTiling(tilingContextPara, tilingInfo);
}

} // namespace

class TestGroupedNoQuantArch35AttrValidation : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(TestGroupedNoQuantArch35AttrValidation, GroupTypeNoSplit_Valid)
{
    EXPECT_TRUE(RunTiling(-1, 0, 0));
}

TEST_F(TestGroupedNoQuantArch35AttrValidation, GroupTypeSplitM_Valid)
{
    EXPECT_TRUE(RunTiling(0, 3, 0));
}

TEST_F(TestGroupedNoQuantArch35AttrValidation, GroupTypeSplitK_Valid)
{
    EXPECT_TRUE(RunTiling(2, 3, 0));
}

TEST_F(TestGroupedNoQuantArch35AttrValidation, GroupTypeInvalid_Invalid)
{
    EXPECT_FALSE(RunTiling(1, 3, 0));
}

TEST_F(TestGroupedNoQuantArch35AttrValidation, GroupListTypeCumsum_Valid)
{
    EXPECT_TRUE(RunTiling(0, 3, 0));
}

TEST_F(TestGroupedNoQuantArch35AttrValidation, GroupListTypeCount_Valid)
{
    EXPECT_TRUE(RunTiling(0, 3, 1));
}

TEST_F(TestGroupedNoQuantArch35AttrValidation, GroupListTypeSparseM_Invalid)
{
    EXPECT_FALSE(RunTiling(0, 3, 2));
}
