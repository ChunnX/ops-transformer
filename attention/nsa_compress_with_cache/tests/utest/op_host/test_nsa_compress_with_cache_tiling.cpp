/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "tiling_case_executor.h"
#include "ncwc_tiling_ut_common.h"

using namespace ncwc_tiling_ut;

class NsaCompressWithCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressWithCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressWithCacheTiling TearDown" << std::endl;
    }
};

namespace {

void RunTilingFailureCase(const CaseParams& params)
{
    CaseShapes shapes;
    auto para = BuildTilingContextPara(params, shapes);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

}  // namespace

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_25_batch_size_too_large)
{
    CaseParams param;
    param.batchSize = 2049;
    param.headNum = 32;
    param.headDim = 64;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = ActSeqLenList32Batch();
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_26_input_too_large)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 256;
    param.headDim = 512;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.compressBlockSize = 64;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_27_empty_head_num)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 0;
    param.headDim = 128;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.compressBlockSize = 64;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_28_compress_block_size_misaligned)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 2;
    param.headDim = 128;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.compressBlockSize = 65;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_29_compress_stride_misaligned)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 3;
    param.headDim = 128;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.compressBlockSize = 64;
    param.compressStride = 15;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_30_stride_larger_than_block)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 4;
    param.headDim = 128;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.compressBlockSize = 16;
    param.compressStride = 32;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_31_invalid_act_seq_len_type)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 5;
    param.headDim = 128;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.actSeqLenType = 0;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_32_head_dim_misaligned)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 6;
    param.headDim = 127;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.actSeqLenType = 0;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_33_head_dim_smaller_than_head_num)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 32;
    param.headDim = 16;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.actSeqLenType = 0;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_34_unknown_layout)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 32;
    param.headDim = 16;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.layout = "UNKOWN";
    param.actSeqLenType = 0;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_35_unsupported_dtype)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 32;
    param.headDim = 16;
    param.dtype = ge::DT_FLOAT;
    param.actSeqLenList = {144, 192, 144, 144};
    param.actSeqLenType = 0;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_36_compress_block_size_too_large)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 32;
    param.headDim = 16;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.compressBlockSize = 128;
    param.actSeqLenType = 0;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_36_head_num_too_large)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 66;
    param.headDim = 16;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 1, 1};
    param.compressBlockSize = 32;
    param.actSeqLenType = 0;
    RunTilingFailureCase(param);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_Case_36_head_num_odd_and_large)
{
    CaseParams param;
    param.batchSize = 4;
    param.headNum = 51;
    param.headDim = 16;
    param.dtype = ge::DT_BF16;
    param.actSeqLenList = {144, 192, 144, 144};
    param.actSeqLenType = 0;
    RunTilingFailureCase(param);
}
