/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_preprocess_case.h
 * \brief MlaPreprocess 测试用例.
 */

#pragma once
#include <vector>
#include <cstdint>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>

namespace ops::adv::tests::mla_preprocess {
class MlaPreprocessCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

public:

    class Param {
    public:
        int64_t tokenNum = 0;
        int64_t hiddenNum = 7168;
        int64_t headNum = 32;
        int64_t bloackNum = 192;
        int64_t bloackSize = 128;

        int64_t wdqDim = 128;
        int64_t qRopeDim = 0; 
        int64_t kRopeDim = 0;
        float epsilon = 1e-05f;
        int64_t qRotaryCoeff = 2;
        int64_t kRotaryCoeff = 2;
        bool transeposeWdq = true;
        bool transeposeWuq = true;
        bool transeposeWuk = true;
        int64_t cacheMode =  0;
        int64_t quantMode =  0;
        bool doRmsNorm = true;
        int64_t wdkvSplitCount = 1;
        ge::DataType inputDataType = ge::DataType::DT_FLOAT16;
        bool wdqkvFormatND = 0;
        bool wuqFormatND = 0;
        bool wukFormatND = 0;
        
        Param();
        Param(int64_t tokenNum_, int64_t hiddenNum_, int64_t headNum_, int64_t bloackNum_, int64_t bloackSize_,
              int64_t wdqDim_, int64_t qRopeDim_, int64_t kRopeDim_, float epsilon_, int64_t qRotaryCoeff_,
              int64_t kRotaryCoeff_, bool transeposeWdq_, bool transeposeWuq_, bool transeposeWuk_, int64_t cacheMode_,
              int64_t quantMode_, bool doRmsNorm_, int64_t wdkvSplitCount_, ge::DataType inputDataType_,
              bool wdqkvFormatND_, bool wuqFormatND_, bool wukFormatND_);
    };

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *actualSeqLengthsTensor = nullptr;
    };

    Tensor hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, descale1Gm, bias1Gm, gamma2Gm,
        beta2Gm, quantScale2Gm, quantOffset2Gm, wuqGm, descale2Gm, bias2Gm, gamma3Gm, cos1Gm, sin1Gm, wukGm, keycacheGm,
        keycacheRopeGm, slotMappingGm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2, keycacheOutGm2;

    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc MlaPreprocessTilingFunc = nullptr;
    MlaPreprocessCase();
    MlaPreprocessCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam& tilingParam);
};

} // namespace ops::adv::tests::mla_preprocess
