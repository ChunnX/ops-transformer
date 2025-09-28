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
 * \file ts_aclnn_fa_tc_level2_redline.cpp
 * \brief FlashAttentionScore / FlashAttentionScoreGrad ACLNN 测试用例.
 */

#include "ts_aclnn_fas.h"

TEST_P(Ts_Aclnn_Fas_WithParam_Ascend910B2, Tc_Level2_Redline)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_TRUE(case_->Run());
}

const auto Tc_Level2_Redline_Cases = ::testing::Values(

    AclnnFasCase("Test_001", true,                                       /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                        ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                   ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFaParam(2, 2, 1, 75, 75, 88,                       /* B,N2,G,S1,S2,D */
                              ge::DataType::DT_FLOAT16, LayoutType::TND, /* Dtype,Layout */
                              0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                              0, 0,                                      /* InnerPrecise,SparseMode */
                              PseShapeType::NONE,                        /* PseShapeType */
                              DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                              PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                              AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                              ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                              PrefixShapeType::NONE,                     /* PrefixShapeType */
                              {},                                        /* PrefixTensorData */
                              {25, 50},                                  /* ActualSeqQLenList */
                              {50, 50})                                  /* ActualSeqKVLenList */
                ),
    AclnnFasCase("Test_002", true,                                       /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                        ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                   ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFaParam(2, 2, 1, 75, 75, 88,                       /* B,N2,G,S1,S2,D */
                              ge::DataType::DT_FLOAT16, LayoutType::TND, /* Dtype,Layout */
                              0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                              0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                              PseShapeType::B_N1_S1_S2,                  /* PseShapeType */
                              DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                              PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                              AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                              ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                              PrefixShapeType::NONE,                     /* PrefixShapeType */
                              {},                                        /* PrefixTensorData */
                              {25, 50},                                  /* ActualSeqQLenList */
                              {50, 50})                                  /* ActualSeqKVLenList */
                ),
    AclnnFasCase("Test_003", true,                                       /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                        ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                   ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFaParam(2, 2, 1, 75, 75, 88,                       /* B,N2,G,S1,S2,D */
                              ge::DataType::DT_FLOAT16, LayoutType::BSH, /* Dtype,Layout */
                              0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                              0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                              PseShapeType::B_N1_S1_S2,                  /* PseShapeType */
                              DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                              PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                              AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                              ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                              PrefixShapeType::NONE)                     /* PrefixShapeType */
                )

);
INSTANTIATE_TEST_SUITE_P(Fas, Ts_Aclnn_Fas_WithParam_Ascend910B2, Tc_Level2_Redline_Cases);
