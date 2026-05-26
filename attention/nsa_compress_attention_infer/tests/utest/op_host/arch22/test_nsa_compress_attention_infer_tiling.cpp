/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "tiling_case_executor.h"
#include "ncai_tiling_ut_common.h"

using namespace ncai_tiling_ut;

class NsaCompressAttentionInferTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressAttentionInferTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressAttentionInferTiling TearDown" << std::endl;
    }
};

TEST_F(NsaCompressAttentionInferTiling, case_nsa_compress_attention_infer_0_tiling)
{
    CaseShapes shapes;
    auto para = BuildTilingContextPara(LegacyCase0Tnd(), shapes);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, LegacyCase0Tnd().expectTilingKey);
}

TEST_F(NsaCompressAttentionInferTiling, case_nsa_compress_attention_infer_1_tiling)
{
    CaseShapes shapes;
    auto para = BuildTilingContextPara(LegacyCase1Bsnd(), shapes);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, LegacyCase1Bsnd().expectTilingKey);
}

TEST_F(NsaCompressAttentionInferTiling, case_nsa_compress_attention_infer_aclgraph_tiling)
{
    CaseShapes shapes;
    auto para = BuildTilingContextPara(LegacyCaseAclGraphTnd(), shapes);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, LegacyCaseAclGraphTnd().expectTilingKey);
}

TEST_F(NsaCompressAttentionInferTiling, case_nsa_compress_attention_infer_mtp_tiling)
{
    CaseShapes shapes;
    auto para = BuildTilingContextPara(LegacyCaseMtpTnd(), shapes);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, LegacyCaseMtpTnd().expectTilingKey);
}
