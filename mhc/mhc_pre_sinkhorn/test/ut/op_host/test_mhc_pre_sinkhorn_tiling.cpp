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
 * \file test_mhc_pre_sinkhorn_tiling.cpp
 * \brief MhcPreSinkhorn tiling test
 */
#include <iostream>

#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/mhc_pre_sinkhorn_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class MhcPreSinkhornTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreSinkhornTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreSinkhornTiling TearDown" << std::endl;
    }
};

TEST_F(MhcPreSinkhornTiling, test_tiling_3d_bf16_tilingkey_1)
{
    optiling::MhcPreSinkhornCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 64;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSinkhorn",
                                              {
                                                  {{{1024, 512}, {1024, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 2048}, {512, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 512}, {1024, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4}, {1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 4, 4}, {1024, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 4, 512}, {1024, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 4, 512}, {1024, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 4, 512}, {1024, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 4, 512}, {1024, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 4, 512}, {1024, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                                  {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                                  {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo,
                                              socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1024 4 512 512 64 16 16 1 1024 1 512 1 512 512 20 0.000001 0.000001 64 16 16 4 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, test_tiling_4d_bf16_tilingkey_1)
{
    optiling::MhcPreSinkhornCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 64;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSinkhorn",
                                              {
                                                  {{{1, 1024, 512}, {1, 1024, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 2048}, {512, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 1024, 512}, {1, 1024, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1024, 4}, {1, 1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1024, 4, 4}, {1, 1024, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1024, 4, 512}, {1, 1024, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1024, 4, 512}, {1, 1024, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1024, 4, 512}, {1, 1024, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1024, 4, 512}, {1, 1024, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1024, 4, 512}, {1, 1024, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                                  {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                                  {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo,
                                              socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1024 4 512 512 64 16 16 1 1024 1 512 1 512 512 20 0.000001 0.000001 64 16 16 4 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, test_tiling_n6_tilingkey_1)
{
    optiling::MhcPreSinkhornCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 64;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSinkhorn",
                                              {
                                                  {{{512, 1024}, {512, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512, 1024}, {512, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 6}, {512, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 6, 6}, {512, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 6, 1024}, {512, 6, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 6, 1024}, {512, 6, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 6, 1024}, {512, 6, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 6, 1024}, {512, 6, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 6, 1024}, {512, 6, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                                  {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                                  {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo,
                                              socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 1;
    string expectTilingData = "512 6 1024 1024 64 8 8 1 512 1 1024 1 1024 1024 20 0.000001 0.000001 64 8 8 6 8 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MhcPreSinkhornTiling, test_tiling_no_backward_tilingkey_1)
{
    optiling::MhcPreSinkhornCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 64;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSinkhorn",
                                              {
                                                  {{{256, 512}, {256, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 2048}, {512, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{256, 512}, {256, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 4}, {256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{256, 4, 4}, {256, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"hc_mult", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"num_iters", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                  {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                                  {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                                  {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo,
                                              socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 1;
    string expectTilingData = "256 4 512 512 64 16 16 1 256 1 512 1 512 512 20 0.000001 0.000001 64 16 16 4 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
