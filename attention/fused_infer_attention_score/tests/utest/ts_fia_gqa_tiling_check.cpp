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
 * \file ts_fia_tiling.cpp
 * \brief FusedInferAttentionScore用例-新UT框架.
 */

#include "ts_fia.h"

using CaseMode = ops::adv::tests::fia::CaseMode;
using CaseKvStorageMode = ops::adv::tests::fia::CaseKvStorageMode;

// // Softmax_Lse--------------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckLseExist_001) // 存在性校验
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    cs.mParam.softmax_lse_flag = 1;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.softmaxLse = Tensor("softmaxLse", {}, "BNSD", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckLseShape_002) // layout=TND, lse=(T,N1,1)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.softmax_lse_flag = 1;
    cs.mParam.layout = "TND";
    cs.mParam.actualSeqLength = {3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.softmaxLse = Tensor("softmaxLse", {2, 6, 2}, "TND", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckLseShape_003) // layout=NTD, lse=(T,N1,1)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.softmax_lse_flag = 1;
    cs.mParam.layout = "TND_NTD"; 
    cs.mParam.actualSeqLength = {3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 3, 512}, "NTD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.softmaxLse = Tensor("softmaxLse", {6, 2, 1}, "TND", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckLseShape_004) // layout=others, shape=[B,N1,S1,1]
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.softmax_lse_flag = 1;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 32, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 32, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 32, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 32, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.softmaxLse = Tensor("softmaxLse", {2, 6, 1, 1}, "TND", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckLseDType_006) // dtpye校验，仅支持FP32
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.softmax_lse_flag = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.softmaxLse = Tensor("softmaxLse", {2, 6, 1}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


// Mask-------------------------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_019) // sparsemod = 0, layout=S1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_021) // sparsemod = 0, layout=1S1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 2, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_022) // sparsemod = 0, layout=BS1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {3, 2, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_023) // sparsemod = 0, layout=11S1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 2, 1, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_024) // sparsemod = 0, layout=B1S1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {3, 2, 1, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_025) // sparsemod = 2, [2048, 2048]
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 2048}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_026) // sparsemod = 2, [1, 2048, 2048]
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2, 2048, 2048}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_027) // sparsemod = 2, [1, 1, 2048, 2048]
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 1, 2048, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

// PSE --------------------------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckPseShiftDType_028) // query_type=FP16, PSE_type=FP16
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {2, 1, 1, 1}, "BNSD", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckPseShiftDType_029) // query_type=BF16, PSE_type=BF16
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {2, 1, 1, 1}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckPseShiftShape_030) // BNS1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {3, 1, 1, 1}, "BNSD", ge::DataType::DT_BF16, ge::FORMAT_ND); // {3,64,2,2}
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckPseShiftShape_031) // 1NS1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {1, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1, 1, 1, 1}, "BNSD", ge::DataType::DT_BF16, ge::FORMAT_ND); // {1,64,2,2}
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


// attenmask-------------------------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckGqaAttentionMask_032) // GQA sparse mode = 2/3/4, 必须带mask
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "B", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaAttentionMask_033) // MLA D=128 sparse mode = 2/3/4, 必须带mask
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {3, 64, 64}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 64}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "B", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


// GQA PA------------------------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckGqaPageAttention_034) // 支持layout: BSH/BNSD/NZ
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.blockSize = 128;
    cs.mParam.layout = "BSND_NBSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 16, 64, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 3, 16, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckGqaPageAttention_035) // blocksize是16的倍数
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 129;
    cs.mParam.actualSeqLengthKV = {1};
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

// GQA 非特性------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureGqaLayout_039) // only supports BSH/BSND、BNSD、TND、NTD、BSH_BNSD、BSND_BNSD、BNSD_BSND、NTD_TND
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BSND_NBSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 3, 64, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureGqaNoQuantLayout_041) // BC, 只支持BSND/BSH、BNSD、TND
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "NTD";
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {64, 3, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 128, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 128, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 3, 512}, "NTD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureGqaNoQuantLayout_042) // TL, 只支持BSND/BSH、BNSD
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::TENSOR_LIST;
    cs.mParam.layout = "NTD";
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {64, 3, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {{3, 128, 512}, {3, 128, 512}}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {{3, 128, 512}, {3, 128, 512}}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 3, 512}, "NTD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {6, 6}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureGqaNoQuantShape_046) // actualSeqLengthsKVSize_ should <= 64 * 1024
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    std::vector<int64_t> temp(64 * 1024, 3);
    cs.mParam.actualSeqLength = temp;
    std::vector<int64_t> temp1(64 * 1024 + 1, 1);
    cs.mParam.actualSeqLengthKV = temp1;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}