// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <gtest/gtest.h>
#include <cstring>
#include <limits>

#include "../../../../op_host/arch22/compressor_v2_tiling_arch22.h"
#include "tiling_case_executor.h"

namespace {
const std::string SOC_INFO = R"({"hardware_info": {
    "BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "4096": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "vector_core_cnt": 40, "cube_core_cnt": 20, "socVersion": "Ascend910B"
}})";

class CompressorV2TilingArch22 : public testing::TestWithParam<int> {};

TEST_P(CompressorV2TilingArch22, ValidateContractAndSplitK)
{
    const int scenario = GetParam();
    const bool isTh = scenario == 2 || scenario == 4 || scenario == 14;
    const bool isEmpty = scenario == 3 || scenario == 4;
    const int64_t seq = isEmpty ? 0 : (scenario == 1 || scenario >= 16 ? 1 : 8);
    const int64_t hidden = scenario == 16 ? 4096 : 1024;
    const int64_t head = scenario == 17 ? 512 : 128;
    const int64_t capacity = scenario == 5 || scenario == 14 ? 10 : 11;
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "CompressorV2",
        {
            {{{2, seq, hidden}, {2, seq, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{head, hidden}, {head, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{head, hidden}, {head, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, capacity, 2 * head}, {2, capacity, 2 * head}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, (seq + 3) / 4, head}, {2, (seq + 3) / 4, head}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, capacity, 2 * head}, {2, capacity, 2 * head}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(scenario == 11 ? 3 : 4)},
            {"state_cache_stride_dim0",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(capacity * 2 * head - (scenario == 6 ? 1 : 0))},
        },
        &compileInfo, "Ascend910B", SOC_INFO, 4096);
    para.inputInstanceNum_ = {1, 1, 1, 1, 1, 0, 0, 0};
    para.outputInstanceNum_ = {1, 1};
    if (isTh) {
        para.inputTensorDesc_[0].shape_ = {{2 * seq, hidden}, {2 * seq, hidden}};
        para.inputTensorDesc_.push_back({{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND});
        para.inputInstanceNum_[5] = 1;
        const int64_t outputTokens = std::min(2 * seq, 2 * seq / 4 + 2);
        para.outputTensorDesc_[0].shape_ = {{outputTokens, head}, {outputTokens, head}};
    }
    if (scenario == 7) {
        para.inputTensorDesc_[3].dtype_ = ge::DT_FLOAT16;
    } else if (scenario == 8) {
        para.inputTensorDesc_[2].shape_ = {{256, hidden}, {256, hidden}};
    } else if (scenario == 9) {
        para.inputInstanceNum_[4] = 0;
        para.inputTensorDesc_.pop_back();
    } else if (scenario == 10) {
        para.inputTensorDesc_[4].shape_ = {{2, 1}, {2, 1}};
    } else if (scenario == 12) {
        para.inputTensorDesc_[0].shape_ = {{2, seq, 1025}, {2, seq, 1025}};
    } else if (scenario == 13) {
        para.outputTensorDesc_[0].dtype_ = ge::DT_FLOAT;
    }
    if (scenario >= 6 && scenario <= 13) {
        ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
        return;
    }
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const uint64_t templateId = isEmpty ? 0 : (!isTh && seq <= 4 ? 2 : 1);
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>((templateId << 5) | (isTh ? 1 : 0)));
    if (templateId == 2) {
        ASSERT_GE(info.tilingDataSize, sizeof(optiling::CompressorV2TilingData));
        optiling::CompressorV2TilingData data;
        std::memcpy(&data, info.tilingData.get(), sizeof(data));
        const auto &base = data.baseParams;
        EXPECT_EQ(base.kBaseNum, std::min<uint32_t>(hidden / 128, base.usedCoreNum / (head / 128)));
        uint32_t coveredK = 0;
        for (uint32_t split = 0; split < base.kBaseNum; ++split) {
            const auto &part = base.splitCoreParam[split * (head / 128)];
            EXPECT_EQ(part.kStart, coveredK);
            EXPECT_GT(part.kEnd, part.kStart);
            EXPECT_LE(part.kEnd, hidden);
            coveredK = part.kEnd;
        }
        EXPECT_EQ(coveredK, hidden);
    }
}

INSTANTIATE_TEST_SUITE_P(Contract, CompressorV2TilingArch22, testing::Range(0, 18),
                         ([](const testing::TestParamInfo<int> &info) {
                             const char *names[] = {"Normal",
                                                    "FullLoadSplitKBound",
                                                    "ThNormal",
                                                    "EmptyBsh",
                                                    "EmptyTh",
                                                    "BshShortRingCapacity",
                                                    "ShortStride",
                                                    "StateDtype",
                                                    "WeightShape",
                                                    "MissingTable",
                                                    "TableRank",
                                                    "InvalidRatio",
                                                    "HiddenAlignment",
                                                    "OutputDtype",
                                                    "ThNoValueCapacityCheck",
                                                    "ExactCapacity",
                                                    "SplitKUncapped",
                                                    "SplitKMultipleD"};
                             return names[info.param];
                         }));
} // namespace
