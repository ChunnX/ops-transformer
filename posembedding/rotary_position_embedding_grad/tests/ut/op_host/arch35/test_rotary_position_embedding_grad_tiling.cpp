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
#include "../../../../op_host/rotary_position_embedding_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class RotaryPositionEmbeddingGradTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RotaryPositionEmbeddingGradTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RotaryPositionEmbeddingGradTilingArch35 TearDown" << std::endl;
    }
};

TEST_F(RotaryPositionEmbeddingGradTilingArch35, RotaryPositionEmbeddingGradTiling_fp16_3d_bsd_broadcast_a5)
{
    optiling::RotaryPositionEmbeddingGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbeddingGrad",
                                              {
                                                  // input info
                                                  {{{8, 4096, 128}, {8, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4096, 128}, {1, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4096, 128}, {1, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8, 4096, 128}, {8, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{8, 4096, 128}, {8, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4096, 128}, {1, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 4096, 128}, {1, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend950", 40, 196608);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // cann-9.2.0 Ascend950 regbase tilingKey 
    ASSERT_EQ(tilingInfo.tilingKey, 17314107473ULL);
    ASSERT_EQ(tilingInfo.blockNum, 40);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    ASSERT_EQ(tilingInfo.workspaceSizes[0], 25165824);

    auto tilingData = reinterpret_cast<const int64_t *>(tilingInfo.tilingData.get());
    // ropeGradParams 起始下标 = reduceTiling 占用的 66 个 int64 之后
    EXPECT_EQ(tilingData[66], 8);    // B = T
    EXPECT_EQ(tilingData[67], 4096); // S = N
    EXPECT_EQ(tilingData[68], 128);  // D
    EXPECT_EQ(tilingData[69], 1);    // N = 1, TND 广播场景重映射为 BS1D
    EXPECT_EQ(tilingData[70], 8);    // blockNumB
    EXPECT_EQ(tilingData[71], 1);    // blockFactorB
    EXPECT_EQ(tilingData[72], 5);    // blockNumS
    EXPECT_EQ(tilingData[73], 820);  // blockFactorS
    EXPECT_EQ(tilingData[79], 40);   // usedCoreNum = blockNumB * blockNumS，40 核用满
    EXPECT_EQ(tilingData[80], 0);    // rotaryMode
    // rotaryXParams: b/s/n 同样为重映射后的值
    EXPECT_EQ(tilingData[97], 8);       // rotaryX B = T
    EXPECT_EQ(tilingData[98], 4096);    // rotaryX S = N
    EXPECT_EQ(tilingData[100], 1);      // rotaryX N = 1
    EXPECT_EQ(tilingData[101], 40);     // rotaryX usedCoreNum
    EXPECT_EQ(tilingData[106], 1);      // dCosFlag = 1, isEmptyDy = 0
    EXPECT_EQ(tilingData[107], 524288); // cosShapeSize = 4096 * 128
}
