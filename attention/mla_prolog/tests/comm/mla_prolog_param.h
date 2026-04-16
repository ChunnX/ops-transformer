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
 * \file mla_prolog_param.h
 * \brief MlaProlog 参数信息.
 */

#ifndef MLA_PROLOG_PARAM_H
#define MLA_PROLOG_PARAM_H

#include <cstdint>
#include <vector>
#include <map>
#include "graph/types.h"
#include "tests/utils/log.h"
#include "tests/utils/tensor.h"

namespace ops::adv::tests::MlaProlog {

class MlaPrologParam {
public:
    using Tensor = ops::adv::tests::utils::Tensor;

public:

    enum class CacheModeType {
        PA_BSND,
        PA_NZ,
        BNSD
    };

public:
    /* 设置参数 */
    int64_t T = 0;
    int64_t B = 0;
    int64_t S = 0;
    int64_t He = 0;
    int64_t Hcq = 0;
    int64_t Hckv = 0;
    int64_t N = 0;
    int64_t D = 0;
    int64_t Dr = 0;
    int64_t Skv = 0;
    int64_t Nkv = 0;
    int64_t BlockSize = 0;
    int64_t BlockNum = 0;
    CacheModeType cacheModeType = CacheModeType::PA_BSND;
    std::string cacheMode = "PA_BSND";
    float rmsnormEpsilonCq = 0.00001f;
    float rmsnormEpsilonCkv = 0.00001f;

    std::vector<Tensor> tokenXData = {};
    std::vector<Tensor> weightDqData = {};
    std::vector<Tensor> weightUqQrData = {};
    std::vector<Tensor> weightUkData = {};
    std::vector<Tensor> weightDkvKrData = {};
    std::vector<Tensor> rmsnormGammaCqData = {};
    std::vector<Tensor> rmsnormGammaCkvData = {};
    std::vector<Tensor> ropeSinData = {};
    std::vector<Tensor> ropeCosData = {};
    std::vector<Tensor> cacheIndexData = {};
    std::vector<Tensor> kvCacheData = {};
    std::vector<Tensor> krCacheData = {};

    std::vector<Tensor> dequantScaleXData = {};
    std::vector<Tensor> dequantScaleWDqData = {};
    std::vector<Tensor> dequantScaleWUqQrData = {};
    std::vector<Tensor> dequantScaleWDkvKrData = {};
    std::vector<Tensor> quantScaleCkvData = {};
    std::vector<Tensor> quantScaleCkrData = {};
    std::vector<Tensor> smoothScalesCqData = {};

    /* 输入输出 */
    std::map<std::string, Tensor> mTensorList;

public:
    MlaPrologParam() = default;
    MlaPrologParam(int64_t pB, int64_t pS, int64_t pHe, int64_t pHcq, int64_t pHckv, int64_t pN, int64_t pD, int64_t pDr, int64_t pSkv, int64_t pNkv, int64_t pBlockSize, int64_t pBlockNum,
        CacheModeType pCacheModeType, float pRmsnormEpsilonCq, float pRmsnormEpsilonCkv);

    // for TND
    MlaPrologParam(int64_t pT, int64_t pB, int64_t pS, int64_t pHe, int64_t pHcq, int64_t pHckv, int64_t pN, int64_t pD, int64_t pDr, int64_t pSkv, int64_t pNkv, int64_t pBlockSize, int64_t pBlockNum,
        CacheModeType pCacheModeType, float pRmsnormEpsilonCq, float pRmsnormEpsilonCkv);

    MlaPrologParam(int64_t pB, int64_t pS, int64_t pHe, int64_t pHcq, int64_t pHckv, int64_t pN, int64_t pD, int64_t pDr, int64_t pSkv, int64_t pNkv, int64_t pBlockSize, int64_t pBlockNum,
        CacheModeType pCacheModeType, float pRmsnormEpsilonCq, float pRmsnormEpsilonCkv,
        std::vector<Tensor> pTokenXData, std::vector<Tensor> pWeightDqData,
        std::vector<Tensor> pWeightUqQrData, std::vector<Tensor> pWeightUkData,
        std::vector<Tensor> pWeightDkvKrData, std::vector<Tensor> pRmsnormGammaCqData,
        std::vector<Tensor> pRmsnormGammaCkvData, std::vector<Tensor> pRopeSinData,
        std::vector<Tensor> pRopeCosData, std::vector<Tensor> pCacheIndexData,
        std::vector<Tensor> pKvCacheData, std::vector<Tensor> pKrCacheData,
        std::vector<Tensor> pDequantScaleXData, std::vector<Tensor> pDequantScaleWDqData,
        std::vector<Tensor> pDequantScaleWUqQrData, std::vector<Tensor> pDequantScaleWDkvKrData,
        std::vector<Tensor> pQuantScaleCkvData, std::vector<Tensor> pQuantScaleCkrData,
        std::vector<Tensor> pSmoothScalesCqData);
        
    // for TND
    MlaPrologParam(int64_t pT, int64_t pB, int64_t pS, int64_t pHe, int64_t pHcq, int64_t pHckv, int64_t pN, int64_t pD, int64_t pDr, int64_t pSkv, int64_t pNkv, int64_t pBlockSize, int64_t pBlockNum,
        CacheModeType pCacheModeType, float pRmsnormEpsilonCq, float pRmsnormEpsilonCkv,
        std::vector<Tensor> pTokenXData, std::vector<Tensor> pWeightDqData,
        std::vector<Tensor> pWeightUqQrData, std::vector<Tensor> pWeightUkData,
        std::vector<Tensor> pWeightDkvKrData, std::vector<Tensor> pRmsnormGammaCqData,
        std::vector<Tensor> pRmsnormGammaCkvData, std::vector<Tensor> pRopeSinData,
        std::vector<Tensor> pRopeCosData, std::vector<Tensor> pCacheIndexData,
        std::vector<Tensor> pKvCacheData, std::vector<Tensor> pKrCacheData,
        std::vector<Tensor> pDequantScaleXData, std::vector<Tensor> pDequantScaleWDqData,
        std::vector<Tensor> pDequantScaleWUqQrData, std::vector<Tensor> pDequantScaleWDkvKrData,
        std::vector<Tensor> pQuantScaleCkvData, std::vector<Tensor> pQuantScaleCkrData,
        std::vector<Tensor> pSmoothScalesCqData);

    virtual ~MlaPrologParam() = default;

    virtual bool Init();

    bool InitParam();
};

} // namespace ops::adv::tests::MlaProlog

#endif