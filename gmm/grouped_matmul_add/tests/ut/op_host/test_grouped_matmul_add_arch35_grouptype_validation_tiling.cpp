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
 * \file test_grouped_matmul_add_arch35_grouptype_validation_tiling.cpp
 * \brief Unit tests for groupType attribute validation in GroupedMatmulAdd arch35 tiling.
 *
 * Covers the code path modified in grouped_matmul_add_no_quant_tiling.cpp line 72:
 *   groupType must be in {2} (SPLIT_K only).
 *   - Valid: SPLIT_K(2) → tiling succeeds
 *   - Invalid: SPLIT_M(0) → tiling fails (caught at GetAttrs line 91)
 *
 * Note: Line 72 is protected by an earlier check at GetAttrs line 91 which validates
 * groupType == SPLIT_K before Init() reaches line 72. The invalid case therefore
 * fails at line 91, not line 72, but the observable behavior is the same:
 * invalid groupType causes tiling failure.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../../op_host/op_tiling/grouped_matmul_add_tiling.h"
#include "../../../op_host/op_tiling/arch35/grouped_matmul_add_compile_info.h"
#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

namespace {

vector<gert::TilingContextPara::OpAttr> GetAddAttrs(bool transposeX, bool transposeWeight, int64_t groupType,
                                                     int64_t groupListType)
{
    return {
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
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

bool RunTiling(int64_t groupType)
{
    optiling::GMMCompileInfo compileInfo = MakeCompileInfo();

    gert::StorageShape xShape = ops::ut::MakeGertStorageShape(vector<int64_t>{1280, 345});
    gert::StorageShape weightShape = ops::ut::MakeGertStorageShape(vector<int64_t>{1280, 567});
    gert::StorageShape groupListShape = ops::ut::MakeGertStorageShape(vector<int64_t>{2});
    gert::StorageShape yShape = ops::ut::MakeGertStorageShape(vector<int64_t>{690, 567});

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulAdd",
        {
            {xShape, ge::DT_BF16, ge::FORMAT_ND},
            {weightShape, ge::DT_BF16, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        GetAddAttrs(true, false, groupType, 0),
        &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo tilingInfo;
    return ExecuteTiling(tilingContextPara, tilingInfo);
}

} // namespace

class TestGroupedMatmulAddArch35GroupTypeValidation : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(TestGroupedMatmulAddArch35GroupTypeValidation, GroupTypeSplitK_Valid)
{
    EXPECT_TRUE(RunTiling(2));
}

TEST_F(TestGroupedMatmulAddArch35GroupTypeValidation, GroupTypeSplitM_Invalid)
{
    EXPECT_FALSE(RunTiling(0));
}
