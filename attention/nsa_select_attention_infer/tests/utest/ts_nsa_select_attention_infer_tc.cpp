/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file ts_nsa_select_attention_infer_tc.cpp
 * \brief NsaSelectAttentionInfer用例.
 */

#include "ts_nsa_select_attention_infer.h"

namespace test_constants {
    constexpr uint32_t CASE1_B = 1;
    constexpr uint32_t CASE2_B = 30;
    constexpr uint32_t CASE1_S1 = 1;
    constexpr uint32_t CASE1_MTP_S1 = 3;
    constexpr int64_t CASE1_NUMTOKENS = 1;
    constexpr int64_t CASE2_NUMTOKENS = 30;
    constexpr int64_t CASE2_MTP_NUMTOKENS = 90;
    constexpr uint32_t CASE1_NUMHEADS = 1;
    constexpr uint32_t CASE1_HEADDIM = 192;
    constexpr uint32_t CASE1_MAXBLOCKNUMPERBATCH = 2;
    constexpr uint32_t CASE1_PAGEBLOCKSIZE = 128;
    constexpr uint32_t CASE1_S2 = 256;
    constexpr uint32_t CASE1_HEADDIMV = 128;
    constexpr uint32_t CASE1_NUMKVHEADS = 1;
    constexpr uint32_t CASE1_SELECTBLOCKSIZE = 128;
    constexpr uint32_t CASE1_SELECTBLOCKCOUNT = 2;
    constexpr uint32_t CASE1_SPARSEMODE = 0;
    constexpr float CASE1_SCALEVALUE = 1.0f; 
}

NsaSelectAttentionInferCase InitNormalCase(int64_t batchSize, int64_t qSeqSize, int64_t headSize, int64_t headDim,
                                int64_t maxBlockNumPerBatch, int64_t blockSize, int64_t seqSize, int64_t headDimV,
                                int64_t headSizeV, int64_t selectedBlockSize, int64_t selectedBlockCount, int64_t sparseMode, float scaleValue,
                                std::string inputLayout, ge::DataType optionalDataType, std::vector<int64_t> actQSeqLens, std::vector<int64_t> actKvSeqLens,
                                ge::graphStatus result, int64_t tilingKey, int64_t numtokens)
{
    NsaSelectAttentionInferCase cs;
        cs.mParam = {batchSize, qSeqSize, headSize, headDim, maxBlockNumPerBatch, blockSize, seqSize, headDimV, headSizeV,
                    selectedBlockSize, selectedBlockCount, sparseMode, scaleValue, inputLayout, optionalDataType, actQSeqLens, actKvSeqLens, numtokens};
        if (result == ge::GRAPH_SUCCESS) {
            cs.mOpInfo.mExp.mTilingKey = tilingKey;
            cs.mOpInfo.mCtr.mRunTiling = true;
            cs.mOpInfo.mCtr.mRunKernel = true;
        } else {
            cs.mOpInfo.mExp.mSuccess = false;
            cs.mOpInfo.mCtr.mRunTiling = true;
            cs.mOpInfo.mCtr.mRunKernel = false;
        }
        cs.Init();
        return cs;
}
void InitAndRunNormalCase(int64_t batchSize, int64_t qSeqSize, int64_t headSize, int64_t headDim,
                        int64_t maxBlockNumPerBatch, int64_t blockSize, int64_t seqSize, int64_t headDimV,
                        int64_t headSizeV, int64_t selectedBlockSize, int64_t selectedBlockCount, int64_t sparseMode, float scaleValue,
                        std::string inputLayout, ge::DataType optionalDataType, std::vector<int64_t> actQSeqLens, std::vector<int64_t> actKvSeqLens,
                        ge::graphStatus result, int64_t tilingKey, int64_t numtokens)
{  
    NsaSelectAttentionInferCase cs = InitNormalCase(batchSize, qSeqSize, headSize, headDim,
        maxBlockNumPerBatch, blockSize, seqSize, headDimV, headSizeV, selectedBlockSize, selectedBlockCount, sparseMode, scaleValue,
        inputLayout, optionalDataType, actQSeqLens, actKvSeqLens, result, tilingKey, numtokens);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

namespace tests {
    using namespace test_constants;
    TEST_F(Ts_NsaSelectAttentionInfer, nsa_select_attention_infer_base_basnd_q3_multi_batch_fp16)
    {   
        std::vector<int64_t> kvSeqlenData(CASE2_B, CASE1_S2);
        std::vector<int64_t> qSeqlenData(CASE2_B, CASE1_MTP_S1);
        InitAndRunNormalCase(CASE2_B, CASE1_S1, CASE1_NUMHEADS, CASE1_HEADDIM, CASE1_MAXBLOCKNUMPERBATCH, CASE1_PAGEBLOCKSIZE, CASE1_S2, CASE1_HEADDIMV, CASE1_NUMKVHEADS, CASE1_SELECTBLOCKSIZE, CASE1_SELECTBLOCKCOUNT, CASE1_SPARSEMODE, CASE1_SCALEVALUE, "TND",  ge::DT_FLOAT16, qSeqlenData, kvSeqlenData,
                                ge::GRAPH_SUCCESS, 1, CASE2_MTP_NUMTOKENS);
    }

    TEST_F(Ts_NsaSelectAttentionInfer, nsa_select_attention_infer_base_basnd_multi_batch_fp16)
    {   
        std::vector<int64_t> kvSeqlenData(CASE2_B, CASE1_S2);
        std::vector<int64_t> qSeqlenData(CASE2_B, CASE1_S1);
        InitAndRunNormalCase(CASE2_B, CASE1_S1, CASE1_NUMHEADS, CASE1_HEADDIM, CASE1_MAXBLOCKNUMPERBATCH, CASE1_PAGEBLOCKSIZE, CASE1_S2, CASE1_HEADDIMV, CASE1_NUMKVHEADS, CASE1_SELECTBLOCKSIZE, CASE1_SELECTBLOCKCOUNT, CASE1_SPARSEMODE, CASE1_SCALEVALUE, "BSND",  ge::DT_FLOAT16, qSeqlenData, kvSeqlenData,
                                ge::GRAPH_SUCCESS, 0, CASE2_NUMTOKENS);
    }

    TEST_F(Ts_NsaSelectAttentionInfer, nsa_select_attention_infer_base_bsh_single_batch_fp16)
    {   
        std::vector<int64_t> kvSeqlenData(CASE1_B, CASE1_S2);
        std::vector<int64_t> qSeqlenData(CASE1_B, CASE1_S1);
        InitAndRunNormalCase(CASE1_B, CASE1_S1, CASE1_NUMHEADS, CASE1_HEADDIM, CASE1_MAXBLOCKNUMPERBATCH, CASE1_PAGEBLOCKSIZE, CASE1_S2, CASE1_HEADDIMV, CASE1_NUMKVHEADS, CASE1_SELECTBLOCKSIZE, CASE1_SELECTBLOCKCOUNT, CASE1_SPARSEMODE, CASE1_SCALEVALUE, "BSH",  ge::DT_FLOAT16, qSeqlenData, kvSeqlenData,
                                ge::GRAPH_SUCCESS, 0, CASE1_NUMTOKENS);
    }

    TEST_F(Ts_NsaSelectAttentionInfer, nsa_select_attention_infer_base_tnd_single_batch_fp16)
    {
        std::vector<int64_t> kvSeqlenData(CASE1_B, CASE1_S2);
        std::vector<int64_t> qSeqlenData(CASE2_B, CASE1_S1);
        InitAndRunNormalCase(CASE1_B, CASE1_S1, CASE1_NUMHEADS, CASE1_HEADDIM, CASE1_MAXBLOCKNUMPERBATCH, CASE1_PAGEBLOCKSIZE, CASE1_S2, CASE1_HEADDIMV, CASE1_NUMKVHEADS, CASE1_SELECTBLOCKSIZE, CASE1_SELECTBLOCKCOUNT, CASE1_SPARSEMODE, CASE1_SCALEVALUE, "TND",  ge::DT_FLOAT16, qSeqlenData, kvSeqlenData,
                                ge::GRAPH_SUCCESS, 1, CASE1_NUMTOKENS);
    }
}