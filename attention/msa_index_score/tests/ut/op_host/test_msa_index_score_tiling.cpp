/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// 顶层（目录不带 arch）用例：executor 的平台信息全 mock，可在任意 soc 构建中执行。
// 以 socVersion="Ascend950" 驱动 DAV_3510 路径。
class MsaIndexScoreTilingCommon : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MsaIndexScoreTilingCommon SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MsaIndexScoreTilingCommon TearDown" << std::endl;
    }
};

namespace {
constexpr uint64_t MSA_TILING_KEY_FP16 = 1;
} // namespace

// mocked Ascend950，BBND fp16 非量化，sparse_mode=0（不传 scale / atten_mask）
TEST_F(MsaIndexScoreTilingCommon, MsaIndexScore_mocked_950_tiling_bbnd_fp16_success)
{
    struct MsaCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MsaIndexScore",
        {
            {{{32, 8, 128}, {32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // query
            {{{2, 128, 1, 128}, {2, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key BBND
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},                       // block_table
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // scale
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                // atten_mask
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                             // actual_seq_qlen
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                             // actual_seq_klen
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}                              // start_loc
        },
        {
            {{{8, 32, 16}, {8, 32, 16}}, ge::DT_FLOAT, ge::FORMAT_ND} // score
        },
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BBND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, MSA_TILING_KEY_FP16);
}

TEST_F(MsaIndexScoreTilingCommon, MsaIndexScore_mocked_950_tiling_missing_qlen_fail)
{
    struct MsaCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MsaIndexScore",
        {{{{32, 8, 128}, {32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 128, 1, 128}, {2, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // actual_seq_qlen absent
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{8, 32, 16}, {8, 32, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BBND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
