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
 * \file mla_prolog_case.h
 * \brief MlaProlog 测试用例.
 */

#ifndef MLA_PROLOG_CASE_H
#define MLA_PROLOG_CASE_H

#include <vector>
#include <utility>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "mla_prolog_param.h"


namespace ops::adv::tests::MlaProlog {

/**
 * 以下参数宏声明要与 MlaProlog Kernel 入口函数保持一致.
 */
#define MLA_PROLOG_KERNEL_PARAM                                                                                               \
    (uint8_t *tokenX, uint8_t *weightDq,    \
        uint8_t *weightUqQr, uint8_t *weightUk,    \
        uint8_t *weightDkvKr, uint8_t *rmsnormGammaCq,    \
        uint8_t *rmsnormGammaCkv, uint8_t *ropeSin,    \
        uint8_t *ropeCos, uint8_t *cacheIndex,    \
        uint8_t *kvCache, uint8_t *krCache,    \
        uint8_t *dequantScaleX, uint8_t *dequantScaleWDq,    \
        uint8_t *dequantScaleWUqQr, uint8_t *dequantScaleWDkvKr,    \
        uint8_t *quantScaleCkv, uint8_t *quantScaleCkr,    \
        uint8_t *smoothScalesCq,    \
        uint8_t *queryOut, uint8_t *queryRopeOut,    \
        uint8_t *kvCacheOut, uint8_t *krCacheOut,    \
        uint8_t *workspace, uint8_t *tiling)

/**
 * 算子 MlaProlog 参数
 */
class MlaPrologCase : public ops::adv::tests::utils::Case {
public:
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using MlaPrologParam = ops::adv::tests::MlaProlog::MlaPrologParam;

    typedef void(*MlaPrologKernelFunc) MLA_PROLOG_KERNEL_PARAM;

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *prefixTensor = nullptr;
        gert::Tensor *actSeqQLenTensor = nullptr;
        gert::Tensor *actSeqKVLenTensor = nullptr;
        gert::Tensor *qStartIdxTensor = nullptr;
        gert::Tensor *kvStartIdxTensor = nullptr;
    };

    /**
     * 执行 Tiling 前, 修改 TilingContext 回调函数.
     *
     * \attention: 一般用于异常用例.
     */
    typedef void (*PreTilingRunCbf)(DoTilingParam &tilingParam);

public:
    /* 算子控制信息 */
    OpInfo mOpInfo;
    Context mCtx;

    /* 输入/输出 参数 */
    MlaPrologParam mParam;

    gert::OpImplRegisterV2::TilingKernelFunc mMlaPrologOriginTilingFunc = nullptr;
    PreTilingRunCbf mPreTilingRunCbf = nullptr;

public:
    MlaPrologCase();
    MlaPrologCase(const char *name, bool enable, const char *dbgInfo, OpInfo prompt, MlaPrologParam param,
           int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);

    bool Run() override;
    bool DoOpTiling(DoTilingParam &tilingParam);

    static void PreTilingRunCbf_SetPlatformInfoNull(MlaPrologCase::DoTilingParam &tilingParam);

protected:
    std::string mMlaPrologOriginTilingFuncName;
    void *mMlaPrologKernelFunc = nullptr;

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitOpInfoCtx();
    bool InitOriginTilingFunc();
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::MlaProlog

#endif