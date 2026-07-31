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
#include <string>
#include <sstream>

#include "../../../op_host/dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

namespace {

std::string MakeSocInfo(uint32_t aicNum, uint32_t aivNum)
{
    std::ostringstream stream;
    stream << R"({"hardware_info": {)"
           << R"("BT_SIZE": 0, "load3d_constraints": "1", )"
           << R"("Intrinsic_fix_pipe_l0c2out": false, )"
           << R"("Intrinsic_data_move_l12ub": true, )"
           << R"("Intrinsic_data_move_l0c2ub": true, )"
           << R"("Intrinsic_data_move_out2l1_nd2nz": false, )"
           << R"("UB_SIZE": 262144, "L2_SIZE": 134217728, )"
           << R"("L1_SIZE": 524288, "L0A_SIZE": 65536, )"
           << R"("L0B_SIZE": 65536, "L0C_SIZE": 262144, )"
           << R"("cube_core_cnt": )" << aicNum << R"(, "vector_core_cnt": )" << aivNum
           << R"(, "socVersion": "Ascend950"}})";
    return stream.str();
}

gert::TilingContextPara MakeBsndCase(optiling::DenseLightningIndexerGradKLLossCompileInfo *compileInfo,
                                     uint32_t aicNum, bool withRope, int64_t n1 = 32)
{
    int64_t b = 1, s1 = 128, s2 = 256, n2 = n1, nidx1 = 8, nidx2 = 1;
    int64_t d = 128, dr = 64, g = n1 / n2;
    gert::TilingContextPara::TensorDescription queryRope =
        withRope ?
            gert::TilingContextPara::TensorDescription(
                gert::StorageShape({b, s1, n1, dr}, {b, s1, n1, dr}), ge::DT_FLOAT16, ge::FORMAT_ND) :
            gert::TilingContextPara::TensorDescription(
                gert::StorageShape({}, {}), ge::DT_UNDEFINED, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription keyRope =
        withRope ?
            gert::TilingContextPara::TensorDescription(
                gert::StorageShape({b, s2, n2, dr}, {b, s2, n2, dr}), ge::DT_FLOAT16, ge::FORMAT_ND) :
            gert::TilingContextPara::TensorDescription(
                gert::StorageShape({}, {}), ge::DT_UNDEFINED, ge::FORMAT_ND);

    return gert::TilingContextPara(
        "DenseLightningIndexerGradKLLoss",
        {
            {{{b, s1, n1, d}, {b, s1, n1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, s2, n2, d}, {b, s2, n2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, s1, nidx1, d}, {b, s1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, s2, nidx2, d}, {b, s2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, s1, nidx1}, {b, s1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, n2, s1, g}, {b, n2, s1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{b, n2, s1, g}, {b, n2, s1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{b, nidx2, s1}, {b, nidx2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{b, nidx2, s1}, {b, nidx2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            queryRope,
            keyRope,
            {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {{{b, s1, nidx1, d}, {b, s1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, s2, nidx2, d}, {b, s2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{b, s1, nidx1}, {b, s1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}
        },
        compileInfo, "Ascend950", aicNum * 2, 262144, 4096, MakeSocInfo(aicNum, aicNum * 2));
}

void CheckCoreTiling(const TilingInfo &tilingInfo, uint32_t expectedCoreNum, int64_t expectedTotalSize)
{
    ASSERT_EQ(tilingInfo.blockNum, expectedCoreNum);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(optiling::DenseLightningIndexerGradKLLossTilingData));
    auto tilingData =
        reinterpret_cast<const optiling::DenseLightningIndexerGradKLLossTilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->multiCoreParams.get_coreNum(), expectedCoreNum);
    EXPECT_EQ(tilingData->multiCoreParams.get_totalSize(), expectedTotalSize);
    for (uint32_t idx = 1; idx < expectedCoreNum; ++idx) {
        EXPECT_LE(tilingData->multiCoreParams.bS1Index[idx - 1],
                  tilingData->multiCoreParams.bS1Index[idx]);
    }
    for (uint32_t idx = expectedCoreNum; idx < optiling::MAX_CORE_NUM; ++idx) {
        EXPECT_EQ(tilingData->multiCoreParams.bS1Index[idx], expectedTotalSize);
    }
}

} // namespace

class DenseLightningIndexerGradKLLossTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "--- DenseLightningIndexerGradKLLossTiling UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "--- DenseLightningIndexerGradKLLossTiling UT TearDown ---" << std::endl;
    }
};

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_bsnd_case0)
{
    optiling::DenseLightningIndexerGradKLLossCompileInfo compileInfo;

    int64_t b = 1, s1 = 128, s2 = 256, n1 = 32, n2 = 32, nidx1 = 8, nidx2 = 1;
    int64_t d = 128, dr = 64, g = n1 / n2;

    int64_t actualSeqQueryData[1] = {s1};
    int64_t actualSeqKeyData[1] = {s2};

    gert::TilingContextPara tilingContextPara(
        "DenseLightningIndexerGradKLLoss",
        {
            {{ {b, s1, n1, d}, {b, s1, n1, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {b, s2, n2, d}, {b, s2, n2, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {b, s1, nidx1, d}, {b, s1, nidx1, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {b, s2, nidx2, d}, {b, s2, nidx2, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {b, s1, nidx1}, {b, s1, nidx1} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {b, n2, s1, g}, {b, n2, s1, g} }, ge::DT_FLOAT, ge::FORMAT_ND},
            {{ {b, n2, s1, g}, {b, n2, s1, g} }, ge::DT_FLOAT, ge::FORMAT_ND},
            {{ {b, nidx2, s1}, {b, nidx2, s1} }, ge::DT_FLOAT, ge::FORMAT_ND},
            {{ {b, nidx2, s1}, {b, nidx2, s1} }, ge::DT_FLOAT, ge::FORMAT_ND},
            {{ {b, s1, n1, dr}, {b, s1, n1, dr} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {b, s2, n2, dr}, {b, s2, n2, dr} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {b}, {b} }, ge::DT_INT64, ge::FORMAT_ND, true, (void*)actualSeqQueryData},
            {{ {b}, {b} }, ge::DT_INT64, ge::FORMAT_ND, true, (void*)actualSeqKeyData}
        },
        {
            {{ {b, s1, nidx1, d}, {b, s1, nidx1, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {b, s2, nidx2, d}, {b, s2, nidx2, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {b, s1, nidx1}, {b, s1, nidx1} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {1}, {1} }, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388)},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}
        },
        &compileInfo);

    uint64_t expectTilingKey = 1UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_tnd_case0)
{
    optiling::DenseLightningIndexerGradKLLossCompileInfo compileInfo;

    int64_t t1 = 128, t2 = 256, n1 = 64, n2 = 64, nidx1 = 16, nidx2 = 1;
    int64_t d = 128, dr = 64, g = n1 / n2;
    int64_t batch = 1;

    int64_t actualSeqQueryData[1] = {t1};
    int64_t actualSeqKeyData[1] = {t2};

    gert::TilingContextPara tilingContextPara(
        "DenseLightningIndexerGradKLLoss",
        {
            {{ {t1, n1, d}, {t1, n1, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {t2, n2, d}, {t2, n2, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {t1, nidx1, d}, {t1, nidx1, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {t2, nidx2, d}, {t2, nidx2, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {t1, nidx1}, {t1, nidx1} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {n2, t1, g}, {n2, t1, g} }, ge::DT_FLOAT, ge::FORMAT_ND},
            {{ {n2, t1, g}, {n2, t1, g} }, ge::DT_FLOAT, ge::FORMAT_ND},
            {{ {nidx2, t1}, {nidx2, t1} }, ge::DT_FLOAT, ge::FORMAT_ND},
            {{ {nidx2, t1}, {nidx2, t1} }, ge::DT_FLOAT, ge::FORMAT_ND},
            {{ {t1, n1, dr}, {t1, n1, dr} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {t2, n2, dr}, {t2, n2, dr} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {batch}, {batch} }, ge::DT_INT64, ge::FORMAT_ND, true, (void*)actualSeqQueryData},
            {{ {batch}, {batch} }, ge::DT_INT64, ge::FORMAT_ND, true, (void*)actualSeqKeyData}
        },
        {
            {{ {t1, nidx1, d}, {t1, nidx1, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {t2, nidx2, d}, {t2, nidx2, d} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {t1, nidx1}, {t1, nidx1} }, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{ {1}, {1} }, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388)},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}
        },
        &compileInfo);

    uint64_t expectTilingKey = 35UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_ascend950_28_aic_without_rope)
{
    optiling::DenseLightningIndexerGradKLLossCompileInfo compileInfo{};
    auto tilingContextPara = MakeBsndCase(&compileInfo, 28, false);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0UL);
    CheckCoreTiling(tilingInfo, 28, 128);
}

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_ascend950_32_aic_capacity)
{
    optiling::DenseLightningIndexerGradKLLossCompileInfo compileInfo{};
    auto tilingContextPara = MakeBsndCase(&compileInfo, 32, true);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    CheckCoreTiling(tilingInfo, 32, 128);
}

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_ascend950_36_aic_capacity)
{
    optiling::DenseLightningIndexerGradKLLossCompileInfo compileInfo{};
    auto tilingContextPara = MakeBsndCase(&compileInfo, 36, true);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    CheckCoreTiling(tilingInfo, 36, 128);
}

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_invalid_shape_returns_failure)
{
    optiling::DenseLightningIndexerGradKLLossCompileInfo compileInfo{};
    auto tilingContextPara = MakeBsndCase(&compileInfo, 28, false, 48);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}
