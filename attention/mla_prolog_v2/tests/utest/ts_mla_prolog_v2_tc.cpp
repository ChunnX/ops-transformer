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
 * \file ts_mla_prolog_v2_tc.cpp
 * \brief MlaPrologV2 正反向用例.
 */

#include "ts_mla_prolog_v2.h"

using Tensor = ops::adv::tests::utils::Tensor;

class Ts_MlaPrologV2_Ascend910B2_tc : public Ts_MlaPrologV2_WithParam_Ascend910B2 {};

TEST_P(Ts_MlaPrologV2_Ascend910B2_tc, Tc_Case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_MlaPrologV2_Case = ::testing::Values(
    // N
    MlaPrologV2Case("MlaPrologV2_Tc_00000", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // full kvCache quant
    MlaPrologV2Case("MlaPrologV2_Tc_00001", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {64, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1, 576}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV2Case("MlaPrologV2_Tc_00002", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {64, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1, 576}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // full non-kvCache quant
    MlaPrologV2Case("MlaPrologV2_Tc_00003", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {64, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1, 576}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {32, 2, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV2Case("MlaPrologV2_Tc_00004", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {64, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1, 576}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {32, 2, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // query datatype
    MlaPrologV2Case("MlaPrologV2_Tc_00005", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {32, 2, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* query */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // queryRope datatype
    MlaPrologV2Case("MlaPrologV2_Tc_00006", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* query */
                   {Tensor("queryRope", {32, 2, 64, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryRope */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // query dimNum
    MlaPrologV2Case("MlaPrologV2_Tc_00007", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {32}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* query */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // queryRope dimNum
    MlaPrologV2Case("MlaPrologV2_Tc_00008", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* query */
                   {Tensor("queryRope", {32}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryRope */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // kvCacheOut dimNum
    MlaPrologV2Case("MlaPrologV2_Tc_00009", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* query */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // krCacheOut dimNum
    MlaPrologV2Case("MlaPrologV2_Tc_00010", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {Tensor("krCacheOut", {16}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* krCacheOut */
                   {},                                             /* query */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // kvCacheOut shape
    MlaPrologV2Case("MlaPrologV2_Tc_00011", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {1024, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {Tensor("krCache", {1024, 128, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {1024, 16, 1, 513}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* query */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // krCacheOut shape
    MlaPrologV2Case("MlaPrologV2_Tc_00012", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {1024, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {Tensor("krCache", {1024, 128, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {1024, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {Tensor("krCacheOut", {1000, 128, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* krCacheOut */
                   {},                                             /* query */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // 强校验拦截
    // quantScaleCkv
    MlaPrologV2Case("MlaPrologV2_Tc_00013", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {64, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1, 576}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {1024, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // quantScaleCkr
    MlaPrologV2Case("MlaPrologV2_Tc_00014", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {64, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1, 576}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                   {Tensor("quantScaleCkr", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
	// 全量化kvcache量化
	// dequantScaleQNopeOutOptional=null
	MlaPrologV2Case("MlaPrologV2_Tc_00015", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(48, 16, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_NZ,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{Tensor("dequantScaleQNopeOut", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)})                                             /* dequantScaleQNopeOut */
		),
	MlaPrologV2Case("MlaPrologV2_Tc_00016", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{Tensor("dequantScaleQNopeOut", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)})                                             /* dequantScaleQNopeOut */
		),
	// 非交付场景拦截
	MlaPrologV2Case("MlaPrologV2_Tc_00017", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::BNSD,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{})                                             /* dequantScaleQNopeOut */
		),
	MlaPrologV2Case("MlaPrologV2_Tc_00018", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(48, 16, 7168, 1536, 512, 3, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{})                                             /* dequantScaleQNopeOut */
		),
	MlaPrologV2Case("MlaPrologV2_Tc_00019", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 129, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{})                                             /* dequantScaleQNopeOut */
		),
	MlaPrologV2Case("MlaPrologV2_Tc_00020", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(65537, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{})                                             /* dequantScaleQNopeOut */
		),
	MlaPrologV2Case("MlaPrologV2_Tc_00021", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(48, 17, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{})                                             /* dequantScaleQNopeOut */
		),
	MlaPrologV2Case("MlaPrologV2_Tc_00022", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 129, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{})                                             /* dequantScaleQNopeOut */
		),
	// format不符合要求
	MlaPrologV2Case("MlaPrologV2_Tc_00023", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{})                                             /* dequantScaleQNopeOut */
		),
	MlaPrologV2Case("MlaPrologV2_Tc_00024", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{Tensor("weightUqQr", {1536, 12544}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{})                                             /* dequantScaleQNopeOut */
		),
	MlaPrologV2Case("MlaPrologV2_Tc_00025", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{})                                             /* dequantScaleQNopeOut */
		),
    //全量化kvcache非量化
    //非交付场景拦截
    MlaPrologV2Case("MlaPrologV2_Tc_00026", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::BNSD,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV2Case("MlaPrologV2_Tc_00027", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(48, 16, 7168, 1536, 512, 3, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 4, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV2Case("MlaPrologV2_Tc_00028", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 129, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV2Case("MlaPrologV2_Tc_00029", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(65537, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV2Case("MlaPrologV2_Tc_00030", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(48, 17, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV2Case("MlaPrologV2_Tc_00031", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 129, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    //format不符合要求
    MlaPrologV2Case("MlaPrologV2_Tc_00032", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV2Case("MlaPrologV2_Tc_00033", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12544}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV2Case("MlaPrologV2_Tc_00034", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // B == 0
    MlaPrologV2Case("MlaPrologV2_Tc_00035", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(0, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 0,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    // S2 == 0
    MlaPrologV2Case("MlaPrologV2_Tc_00036", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(16, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   0, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)}, /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           )
);
INSTANTIATE_TEST_SUITE_P(MlaPrologV2, Ts_MlaPrologV2_Ascend910B2_tc, Tc_MlaPrologV2_Case);