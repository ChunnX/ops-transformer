/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include "tiling_case_executor.h"

class KvCompressEpilogTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvCompressEpilogTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvCompressEpilogTiling TearDown" << std::endl;
    }
};

TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_e8m0)
{
    gert::TilingContextPara optilingContextPara("KvCompressEpilog",
        {
            {{{2048, 384}, {2048, 384}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        nullptr, "Ascend950"
    );
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// mode2: rope hifloat8 静态量化 + nope FLOAT4_E2M1 per-group 动态量化。
// d=256 -> nope=192, 可被 16/32/64 整除; tiling key 应为 2。
static gert::TilingContextPara MakeMode2Para(int64_t groupSize)
{
    return gert::TilingContextPara("KvCompressEpilog",
        {
            {{{2048, 384}, {2048, 384}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupSize)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(2.0f)},
        },
        nullptr, "Ascend950"
    );
}

TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_hif8_fp4_group64)
{
    auto para = MakeMode2Para(64);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);  // 单一 tiling key 0
}

TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_hif8_fp4_group32)
{
    auto para = MakeMode2Para(32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);  // 单一 tiling key 0
}

TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_hif8_fp4_group16)
{
    auto para = MakeMode2Para(16);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);  // 单一 tiling key 0
}

// mode2 仅支持 group_size in {16,32,64}, 其它取值应 tiling 失败。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_hif8_fp4_bad_group)
{
    auto para = MakeMode2Para(128);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

