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

TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_hifloat)
{
    gert::TilingContextPara optilingContextPara("KvCompressEpilog",
        {
            {{{2048, 256}, {2048, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        nullptr, "Ascend950"
    );
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}
