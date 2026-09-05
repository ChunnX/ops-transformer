/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <tuple>
#include <gtest/gtest.h>
#include "../test_quant_lightning_indexer_v2_utils.h"

namespace {
using ScaleCase = std::tuple<std::string, int64_t, std::string>;

qliv2_ut::CaseParam MakeScaleCase(const ScaleCase &test)
{
    const auto &layout = std::get<0>(test);
    const int64_t mode = std::get<1>(test);
    const bool tnd = layout == "TND" || layout == "TND_PA";
    const bool paged = layout == "PA_BBND" || layout == "TND_PA";
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.qShape = tnd ? std::vector<int64_t>{8, 2, 128} : std::vector<int64_t>{2, 4, 2, 128};
    p.kShape = tnd && !paged ? std::vector<int64_t>{16, 1, 128} : std::vector<int64_t>{2, 16, 1, 128};
    p.wShape = tnd ? std::vector<int64_t>{8, 2} : std::vector<int64_t>{2, 4, 2};
    p.outShape = tnd ? std::vector<int64_t>{8, 1, 4} : std::vector<int64_t>{2, 4, 1, 4};
    p.qScaleShape = p.qShape;
    p.kScaleShape = p.kShape;
    p.qScaleShape.pop_back();
    p.kScaleShape.pop_back();
    p.qType = mode == 2 ? ge::DT_INT8 : (mode == 4 ? ge::DT_HIFLOAT8 :
                                      (mode == 5 ? ge::DT_FLOAT4_E2M1 : ge::DT_FLOAT8_E4M3FN));
    p.kType = p.qType;
    p.wType = mode == 2 ? ge::DT_FLOAT16 : ge::DT_FLOAT;
    p.qScaleType = mode == 2 ? ge::DT_FLOAT16 : ge::DT_FLOAT;
    if (mode == 3 || mode == 5) {
        p.qScaleShape.insert(p.qScaleShape.end(), {2, 2});
        p.kScaleShape.insert(p.kScaleShape.end(), {2, 2});
        p.qScaleType = ge::DT_FLOAT8_E8M0;
    } else if (mode == 4) {
        p.qScaleShape = {1};
        p.kScaleShape = {1};
    }
    p.kScaleType = p.qScaleType;
    p.cuSeqQ = tnd ? std::vector<int64_t>{3} : std::vector<int64_t>{};
    p.cuSeqK = tnd && !paged ? std::vector<int64_t>{3} : std::vector<int64_t>{};
    p.sequsedQ = {2};
    p.sequsedK = {2};
    p.cmpResidual = {2};
    p.blockTable = paged ? std::vector<int64_t>{2, 1} : std::vector<int64_t>{};
    p.idxOffset = tnd ? std::vector<int64_t>{8, 1} : std::vector<int64_t>{2, 4, 1};
    p.layoutQ = tnd ? "TND" : "BSND";
    p.layoutK = paged ? "PA_BBND" : layout;
    p.topk = 4;
    p.quantMode = mode;
    p.maxSeqlenQ = 4;
    p.maskMode = 3;
    p.cmpRatio = 4;
    return p;
}
} // namespace

class QuantLightningIndexerV2ScaleContract : public testing::TestWithParam<ScaleCase> {};

TEST_P(QuantLightningIndexerV2ScaleContract, CheckScaleShape)
{
    auto p = MakeScaleCase(GetParam());
    const auto &variant = std::get<2>(GetParam());
    if (variant != "valid") {
        auto &shape = variant[0] == 'q' ? p.qScaleShape : p.kScaleShape;
        if (variant == "q_rank" || variant == "k_rank") {
            shape.push_back(1);
        } else {
            shape[0] += 1;
        }
    }
    qliv2_ut::RunTilingCase(p, variant == "valid" ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(Ascend950, QuantLightningIndexerV2ScaleContract,
    testing::Combine(testing::Values("BSND", "TND", "PA_BBND", "TND_PA"),
                     testing::Values(1, 2, 3, 4, 5),
                     testing::Values("q_rank", "k_rank", "q_axis", "k_axis", "valid")),
    [](const testing::TestParamInfo<ScaleCase> &info) {
        return std::get<0>(info.param) + "_mode" + std::to_string(std::get<1>(info.param)) + "_" +
               std::get<2>(info.param);
    });
