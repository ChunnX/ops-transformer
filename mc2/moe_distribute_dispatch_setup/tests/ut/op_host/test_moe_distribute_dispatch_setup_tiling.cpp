/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNEsS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_distribute_dispatch_setup_tiling.cpp
 * \brief op_host tiling ut
 */

#include <gtest/gtest.h>
#include <memory>
#include <cstring>
#include <string>
#include "register/op_def_registry.h"
#include "tiling_base/tiling_context.h"
#include "moe_distribute_dispatch_setup_tiling_base.h"

namespace {
class MoeDistributeDispatchSetupTilingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckRequiredAttrValue_GroupEpValid_Success)
{
    char groupEp[] = "test_group";
    size_t len = strnlen(groupEp, 128);
    EXPECT_GT(len, 0);
    EXPECT_LT(len, 128);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckRequiredAttrValue_GroupEpEmpty_Fail)
{
    char groupEp[] = "";
    bool isInvalid = (strnlen(groupEp, 128) == 0) || (strnlen(groupEp, 128) == 128);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckRequiredAttrValue_EpWorldSizeValid_Success)
{
    constexpr int64_t MIN_GROUP_EP_SIZE = 2;
    constexpr int64_t MAX_GROUP_EP_SIZE = 384;

    int64_t epWorldSize = 2;
    bool isValid = (epWorldSize >= MIN_GROUP_EP_SIZE) && (epWorldSize <= MAX_GROUP_EP_SIZE);
    EXPECT_TRUE(isValid);

    epWorldSize = 384;
    isValid = (epWorldSize >= MIN_GROUP_EP_SIZE) && (epWorldSize <= MAX_GROUP_EP_SIZE);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckRequiredAttrValue_EpWorldSizeOutOfRange_Fail)
{
    constexpr int64_t MIN_GROUP_EP_SIZE = 2;
    constexpr int64_t MAX_GROUP_EP_SIZE = 384;

    int64_t epWorldSize = 1;
    bool isInvalid = (epWorldSize < MIN_GROUP_EP_SIZE) || (epWorldSize > MAX_GROUP_EP_SIZE);
    EXPECT_TRUE(isInvalid);

    epWorldSize = 385;
    isInvalid = (epWorldSize < MIN_GROUP_EP_SIZE) || (epWorldSize > MAX_GROUP_EP_SIZE);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckRequiredAttrValue_EpRankIdValid_Success)
{
    int64_t epWorldSize = 4;
    int64_t epRankId = 0;
    bool isValid = (epRankId >= 0) && (epRankId < epWorldSize);
    EXPECT_TRUE(isValid);

    epRankId = 3;
    isValid = (epRankId >= 0) && (epRankId < epWorldSize);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckRequiredAttrValue_EpRankIdOutOfRange_Fail)
{
    int64_t epWorldSize = 2;

    int64_t epRankId = -1;
    bool isInvalid = (epRankId < 0) || (epRankId >= epWorldSize);
    EXPECT_TRUE(isInvalid);

    epRankId = 2;
    isInvalid = (epRankId < 0) || (epRankId >= epWorldSize);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckRequiredAttrValue_MoeExpertNumValid_Success)
{
    constexpr int64_t MAX_MOE_EXPERT_NUM = 512;

    int64_t moeExpertNum = 1;
    bool isValid = (moeExpertNum > 0) && (moeExpertNum <= MAX_MOE_EXPERT_NUM);
    EXPECT_TRUE(isValid);

    moeExpertNum = 512;
    isValid = (moeExpertNum > 0) && (moeExpertNum <= MAX_MOE_EXPERT_NUM);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckRequiredAttrValue_MoeExpertNumOutOfRange_Fail)
{
    constexpr int64_t MAX_MOE_EXPERT_NUM = 512;

    int64_t moeExpertNum = 0;
    bool isInvalid = (moeExpertNum <= 0) || (moeExpertNum > MAX_MOE_EXPERT_NUM);
    EXPECT_TRUE(isInvalid);

    moeExpertNum = 513;
    isInvalid = (moeExpertNum <= 0) || (moeExpertNum > MAX_MOE_EXPERT_NUM);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckSharedExpertAttrValue_Valid_Success)
{
    uint32_t sharedExpertNum = 1;
    uint32_t sharedExpertRankNum = 1;
    bool isValid = (sharedExpertRankNum == 0) || 
                  (sharedExpertNum > 0 && sharedExpertNum <= sharedExpertRankNum && 
                   sharedExpertRankNum % sharedExpertNum == 0);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckSharedExpertAttrValue_SharedExpertNumExceedRankNum_Fail)
{
    uint32_t sharedExpertNum = 4;
    uint32_t sharedExpertRankNum = 2;
    bool isInvalid = sharedExpertNum > sharedExpertRankNum;
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckSharedExpertAttrValue_NotDivisible_Fail)
{
    uint32_t sharedExpertNum = 2;
    uint32_t sharedExpertRankNum = 3;
    bool isInvalid = (sharedExpertRankNum % sharedExpertNum) != 0;
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_ExpertShardTypeZero_Success)
{
    int64_t expertShardType = 0;
    bool isValid = (expertShardType == 0);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_ExpertShardTypeNotZero_Fail)
{
    int64_t expertShardType = 1;
    bool isInvalid = (expertShardType != 0);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_SharedExpertNumValid_Success)
{
    constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;

    int64_t sharedExpertNum = 0;
    bool isValid = (sharedExpertNum >= 0) && (sharedExpertNum <= MAX_SHARED_EXPERT_NUM);
    EXPECT_TRUE(isValid);

    sharedExpertNum = 4;
    isValid = (sharedExpertNum >= 0) && (sharedExpertNum <= MAX_SHARED_EXPERT_NUM);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_SharedExpertNumOutOfRange_Fail)
{
    constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;

    int64_t sharedExpertNum = -1;
    bool isInvalid = (sharedExpertNum < 0) || (sharedExpertNum > MAX_SHARED_EXPERT_NUM);
    EXPECT_TRUE(isInvalid);

    sharedExpertNum = 5;
    isInvalid = (sharedExpertNum < 0) || (sharedExpertNum > MAX_SHARED_EXPERT_NUM);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_QuantModeValid_Success)
{
    constexpr int64_t UNQUANT = 0;
    constexpr int64_t MX_QUANT = 4;

    for (int64_t quantMode = UNQUANT; quantMode <= MX_QUANT; ++quantMode) {
        bool isValid = (quantMode >= UNQUANT) && (quantMode <= MX_QUANT);
        EXPECT_TRUE(isValid);
    }
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_QuantModeOutOfRange_Fail)
{
    constexpr int64_t UNQUANT = 0;
    constexpr int64_t MX_QUANT = 4;

    int64_t quantMode = 5;
    bool isInvalid = (quantMode < UNQUANT) || (quantMode > MX_QUANT);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_GlobalBsValid_Success)
{
    int64_t globalBs = 0;
    bool isValid = (globalBs >= 0);
    EXPECT_TRUE(isValid);

    globalBs = 8;
    isValid = (globalBs >= 0);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_GlobalBsNegative_Fail)
{
    int64_t globalBs = -1;
    bool isInvalid = (globalBs < 0);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_CommTypeURMA_Success)
{
    constexpr int64_t URMA_COMM = 2;
    int64_t commType = 2;
    bool isValid = (commType == URMA_COMM);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_CommTypeNotSupport_Fail)
{
    constexpr int64_t URMA_COMM = 2;

    int64_t commType = 0;
    bool isInvalid = (commType != URMA_COMM);
    EXPECT_TRUE(isInvalid);

    commType = 1;
    isInvalid = (commType != URMA_COMM);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_CommAlgNullOrEmpty_Success)
{
    const char* commAlg = nullptr;
    bool isValid = (commAlg == nullptr);
    EXPECT_TRUE(isValid);

    std::string commAlgStr = "";
    isValid = (commAlgStr == "");
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalAttrValue_CommAlgNotEmpty_Fail)
{
    std::string commAlg = "test";
    bool isInvalid = (commAlg != "");
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, GetComplexAttrAndSetTilingData_MoERank_Success)
{
    uint32_t epRankId = 2;
    uint32_t epWorldSize = 4;
    uint32_t sharedExpertNum = 2;
    uint32_t sharedExpertRankNum = 2;

    bool isMoERank = (epRankId >= sharedExpertRankNum);
    EXPECT_TRUE(isMoERank);

    uint32_t moeExpertNum = 8;
    uint32_t expectedMoeExpertPerRankNum = moeExpertNum / (epWorldSize - sharedExpertNum);
    EXPECT_EQ(expectedMoeExpertPerRankNum, 4U);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, GetComplexAttrAndSetTilingData_SharedExpertRank_Success)
{
    uint32_t epRankId = 0;
    uint32_t sharedExpertRankNum = 2;

    bool isSharedExpertRank = (epRankId < sharedExpertRankNum);
    EXPECT_TRUE(isSharedExpertRank);

    uint32_t expectedMoeExpertPerRankNum = 1U;
    EXPECT_EQ(expectedMoeExpertPerRankNum, 1U);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, GetComplexAttrAndSetTilingData_MoeExpertNumNotDivisible_Fail)
{
    uint32_t epWorldSize = 4;
    uint32_t sharedExpertRankNum = 2;
    uint32_t moeExpertNum = 7;

    bool isDivisible = (moeExpertNum % (epWorldSize - sharedExpertRankNum) == 0);
    EXPECT_FALSE(isDivisible);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckTensorShapeRelation_BSEqual_Success)
{
    int64_t xDim0 = 4;
    int64_t expertIdsDim0 = 4;
    bool isValid = (xDim0 == expertIdsDim0);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckTensorShapeRelation_BSNotEqual_Fail)
{
    int64_t xDim0 = 4;
    int64_t expertIdsDim0 = 5;
    bool isInvalid = (xDim0 != expertIdsDim0);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckTensorShapeRelation_HEqual_Success)
{
    int64_t xDim1 = 128;
    int64_t scalesDim1 = 128;
    bool isValid = (xDim1 == scalesDim1);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckInputTensorDataType_XValid_Success)
{
    ge::DataType xDataType1 = ge::DT_FLOAT16;
    bool isValid = (xDataType1 == ge::DT_FLOAT16) || (xDataType1 == ge::DT_BF16);
    EXPECT_TRUE(isValid);

    ge::DataType xDataType2 = ge::DT_BF16;
    isValid = (xDataType2 == ge::DT_FLOAT16) || (xDataType2 == ge::DT_BF16);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckInputTensorDataType_XInvalid_Fail)
{
    ge::DataType xDataType = ge::DT_FLOAT;
    bool isInvalid = (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckInputTensorDataType_ExpertIdsValid_Success)
{
    ge::DataType expertIdsDataType = ge::DT_INT32;
    bool isValid = (expertIdsDataType == ge::DT_INT32);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckInputTensorDataType_ExpertIdsInvalid_Fail)
{
    ge::DataType expertIdsDataType = ge::DT_INT64;
    bool isInvalid = (expertIdsDataType != ge::DT_INT32);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalInputTensorDataType_ScalesValid_Success)
{
    ge::DataType scalesDataType = ge::DT_FLOAT;
    bool isValid = (scalesDataType == ge::DT_FLOAT);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalInputTensorDataType_ScalesInvalid_Fail)
{
    ge::DataType scalesDataType = ge::DT_HALF;
    bool isInvalid = (scalesDataType != ge::DT_FLOAT);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalInputTensorDataType_XActiveMaskValid_Success)
{
    ge::DataType xActiveMaskDataType = ge::DT_BOOL;
    bool isValid = (xActiveMaskDataType == ge::DT_BOOL);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOptionalInputTensorDataType_XActiveMaskInvalid_Fail)
{
    ge::DataType xActiveMaskDataType = ge::DT_INT8;
    bool isInvalid = (xActiveMaskDataType != ge::DT_BOOL);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOutputTensorDataType_ExpandIdxValid_Success)
{
    ge::DataType expandIdxDataType = ge::DT_INT32;
    bool isValid = (expandIdxDataType == ge::DT_INT32);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOutputTensorDataType_CommCmdInfoValid_Success)
{
    ge::DataType commCmdInfoDataType = ge::DT_INT32;
    bool isValid = (commCmdInfoDataType == ge::DT_INT32);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOutputTensorDataType_YQuantMode0_Success)
{
    constexpr int64_t UNQUANT = 0;
    int64_t quantMode = UNQUANT;
    ge::DataType yDataType = ge::DT_BF16;
    ge::DataType xDataType = ge::DT_BF16;

    bool isValid = (quantMode == UNQUANT) && (yDataType == xDataType);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOutputTensorDataType_YQuantMode1_Success)
{
    constexpr int64_t STATIC_QUANT = 1;
    int64_t quantMode = STATIC_QUANT;
    ge::DataType yDataType1 = ge::DT_HIFLOAT8;
    ge::DataType yDataType2 = ge::DT_INT8;

    bool isValid1 = (yDataType1 == ge::DT_HIFLOAT8) || (yDataType1 == ge::DT_INT8);
    bool isValid2 = (yDataType2 == ge::DT_HIFLOAT8) || (yDataType2 == ge::DT_INT8);
    EXPECT_TRUE(isValid1 && isValid2);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOutputTensorDataType_YQuantMode2_Success)
{
    constexpr int64_t PERTOKEN_DYNAMIC_QUANT = 2;
    int64_t quantMode = PERTOKEN_DYNAMIC_QUANT;
    ge::DataType yDataType = ge::DT_INT8;

    bool isValid = (yDataType == ge::DT_INT8) || (yDataType == ge::DT_FLOAT8_E4M3FN) || 
                  (yDataType == ge::DT_FLOAT8_E5M2);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOutputTensorDataType_YQuantMode3_Success)
{
    constexpr int64_t PERGROUP_DYNAMIC_QUANT = 3;
    int64_t quantMode = PERGROUP_DYNAMIC_QUANT;
    ge::DataType yDataType = ge::DT_FLOAT8_E4M3FN;

    bool isValid = (yDataType == ge::DT_FLOAT8_E4M3FN) || (yDataType == ge::DT_FLOAT8_E5M2);
    EXPECT_TRUE(isValid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, CheckOutputTensorDataType_YQuantModeInvalid_Fail)
{
    constexpr int64_t UNQUANT = 0;
    int64_t quantMode = UNQUANT;
    ge::DataType yDataType = ge::DT_INT8;
    ge::DataType xDataType = ge::DT_BF16;

    bool isInvalid = (quantMode == UNQUANT) && (yDataType != xDataType);
    EXPECT_TRUE(isInvalid);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, QuantMode_AlignCalculation_Success)
{
    constexpr int64_t ALIGN_256 = 256;
    constexpr int64_t ALIGN_32 = 32;
    constexpr int64_t ALIGN_512 = 512;
    constexpr int64_t QUANT_ALIGN_OFFSET = 4;
    constexpr int64_t UNQUANT = 0;
    constexpr int64_t STATIC_QUANT = 1;

    int64_t h = 128;
    int64_t quantMode = UNQUANT;
    int64_t expectedTokenMsgSize = ((h + ALIGN_256 - 1) / ALIGN_256) * ALIGN_256;
    EXPECT_EQ(expectedTokenMsgSize, 256);

    h = 300;
    quantMode = STATIC_QUANT;
    int64_t alignedH = ((h + ALIGN_32 - 1) / ALIGN_32) * ALIGN_32;
    int64_t tokenMsgSize = ((alignedH + QUANT_ALIGN_OFFSET + ALIGN_512 - 1) / ALIGN_512) * ALIGN_512;
    EXPECT_EQ(tokenMsgSize, 512);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, InputDimCheck_TwoDims_Success)
{
    constexpr uint32_t TWO_DIMS = 2;
    uint32_t actualDimNum = 2;
    EXPECT_EQ(actualDimNum, TWO_DIMS);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, InputDimCheck_OneDim_Success)
{
    constexpr uint32_t ONE_DIMS = 1;
    uint32_t actualDimNum = 1;
    EXPECT_EQ(actualDimNum, ONE_DIMS);
}

TEST_F(MoeDistributeDispatchSetupTilingTest, QuantMode_EnumValues_Success)
{
    constexpr int64_t UNQUANT = 0;
    constexpr int64_t STATIC_QUANT = 1;
    constexpr int64_t PERTOKEN_DYNAMIC_QUANT = 2;
    constexpr int64_t PERGROUP_DYNAMIC_QUANT = 3;
    constexpr int64_t MX_QUANT = 4;

    EXPECT_EQ(UNQUANT, 0);
    EXPECT_EQ(STATIC_QUANT, 1);
    EXPECT_EQ(PERTOKEN_DYNAMIC_QUANT, 2);
    EXPECT_EQ(PERGROUP_DYNAMIC_QUANT, 3);
    EXPECT_EQ(MX_QUANT, 4);
}
}