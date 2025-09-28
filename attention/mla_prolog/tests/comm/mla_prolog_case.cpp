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
 * \file mla_prolog_case.cpp
 * \brief MlaProlog 测试用例.
 */

#include "mla_prolog_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/mlaprolog/tiling_data.h"
#include "tiling_base/tiling_templates_registry.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */
#ifdef SUPPORT_KERNEL
namespace ops::adv::tests::MlaProlog {
extern "C" __global__ __aicore__ void mla_prolog MLA_PROLOG_KERNEL_PARAM;
}
#endif

namespace optiling {
ASCENDC_EXTERN_C ge::graphStatus TilingMlaProlog(gert::TilingContext *context);
} // namespace optiling

using namespace Ops::Transformer::OpTiling;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;
using MlaPrologCase = ops::adv::tests::MlaProlog::MlaPrologCase;
using namespace ops::adv::tests::MlaProlog;
using TensorIntf = ops::adv::tests::utils::TensorIntf;

namespace {

ASCENDC_EXTERN_C ge::graphStatus MlaPrologTilingFuncStub(gert::TilingContext *context)
{
    auto *mlaPrologCase = static_cast<MlaPrologCase *>(Case::GetCurrentCase());
    if (mlaPrologCase != nullptr) {
        MlaPrologCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (mlaPrologCase->DoOpTiling(p)) {
            return p.ret;
        }
        return mlaPrologCase->mMlaPrologOriginTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool RunMlaProlog(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                       std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (MlaPrologCase::MlaPrologKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                inputs[0]->GetDevData(),  // tokenX
                inputs[1]->GetDevData(),  // weightDq
                inputs[2]->GetDevData(),  // weightUqQr
                inputs[3]->GetDevData(),  // weightUk
                inputs[4]->GetDevData(),  // weightDkvKr
                inputs[5]->GetDevData(),  // rmsnormGammaCq
                inputs[6]->GetDevData(),  // rmsnormGammaCkv
                inputs[7]->GetDevData(),  // ropeSin
                inputs[8]->GetDevData(),  // ropeCos
                inputs[9]->GetDevData(),  // cacheIndex
                inputs[10]->GetDevData(), // kvCache
                inputs[11]->GetDevData(), // krCache
                inputs[12]->GetDevData(), // dequantScaleX
                inputs[13]->GetDevData(), // dequantScaleWDq
                inputs[14]->GetDevData(), // dequantScaleWUqQr
                inputs[15]->GetDevData(), // dequantScaleWDkvKr
                inputs[16]->GetDevData(), // quantScaleCkv
                inputs[17]->GetDevData(), // quantScaleCkr
                inputs[18]->GetDevData(), // smoothScalesCq
                outputs[0]->GetDevData(), // queryOut
                outputs[1]->GetDevData(), // queryRopeOut
                outputs[2]->GetDevData(), // kvCacheOut
                outputs[3]->GetDevData(), // krCacheOut
                workspace, tilingData);
    return true;
}

} // namespace


MlaPrologCase::MlaPrologCase() : MlaPrologCase("Undefined", true, "", OpInfo(), MlaPrologParam(), kTilingTemplatePriority_Invalid)
{
}

MlaPrologCase::MlaPrologCase(const char *name, bool enable, const char *dbgInfo, OpInfo prompt,  MlaPrologParam param,
               int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, tilingTemplatePriority), mOpInfo(std::move(prompt)),
      mParam(std::move(param))
{
    mOpInfo.mName = "MlaProlog";

    mMlaPrologOriginTilingFuncName = "TilingMlaProlog";
#ifdef SUPPORT_KERNEL
    mMlaPrologKernelFunc = (void *)mla_prolog;
#endif
}

bool MlaPrologCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

bool MlaPrologCase::InitParam()
{
    if (!mParam.Init()) {
        return false;
    }
    return true;
}

bool MlaPrologCase::InitOpInfo()
{
    bool rst = this->InitOpInfoCtx();
    if (!rst) {
        return rst;
    }

    if (!this->InitOriginTilingFunc()) {
        return false;
    }
    IMPL_OP(MlaProlog).Tiling(MlaPrologTilingFuncStub);

    return true;
}

bool MlaPrologCase::InitOpInfoCtx()
{
    bool rst = mCtx.SetOpName(mOpInfo.mName.c_str());
    rst = rst && mCtx.SetDeterministic(mOpInfo.mCtr.mDeterministic);
    rst = rst && mCtx.SetInputs({&mParam.mTensorList["tokenX"], &mParam.mTensorList["weightDq"], &mParam.mTensorList["weightUqQr"], &mParam.mTensorList["weightUk"], &mParam.mTensorList["weightDkvKr"],
                                        &mParam.mTensorList["rmsnormGammaCq"], &mParam.mTensorList["rmsnormGammaCkv"], &mParam.mTensorList["ropeSin"], &mParam.mTensorList["ropeCos"],
                                        &mParam.mTensorList["cacheIndex"], &mParam.mTensorList["kvCache"], &mParam.mTensorList["krCache"], &mParam.mTensorList["dequantScaleX"], 
                                        &mParam.mTensorList["dequantScaleWDq"], &mParam.mTensorList["dequantScaleWUqQr"], &mParam.mTensorList["dequantScaleWDkvKr"], 
                                        &mParam.mTensorList["quantScaleCkv"], &mParam.mTensorList["quantScaleCkr"], &mParam.mTensorList["smoothScalesCq"]});
    rst = rst && mCtx.SetOutputs({&mParam.mTensorList["queryOut"], &mParam.mTensorList["queryRopeOut"], &mParam.mTensorList["kvCacheOut"], &mParam.mTensorList["krCacheOut"]});

    rst = rst && mCtx.SetAttrs({{"rmsnorm_epsilon_cq", mParam.rmsnormEpsilonCq},
                                       {"rmsnorm_epsilon_ckv", mParam.rmsnormEpsilonCkv},
                                       {"cache_mode", mParam.cacheMode}});
    rst = rst && mCtx.SetTilingDataMaxSize(2456); /* 2456 MlaProlog 最大 TilingData 大小 */
    rst = rst && mCtx.SetKernelRunCbf(RunMlaProlog);
    rst = rst && mCtx.SetKernelMainFunc((void *)mMlaPrologKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);
    return rst;
}

bool MlaPrologCase::InitOriginTilingFunc()
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    /* MlaProlog 需提供修改 TilingContext 功能 */
    /* MlaProlog 需提供按指定优先级调用 Tiling 模板功能 */
    mMlaPrologOriginTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym(mMlaPrologOriginTilingFuncName.c_str());
    if (mMlaPrologOriginTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, MlaProlog(%p)", mMlaPrologOriginTilingFunc);
        return false;
    }
    return true;
}

bool MlaPrologCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MlaPrologCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (mPreTilingRunCbf != nullptr) {
        mPreTilingRunCbf(tilingParam);
    }

    /* 按优先级 Tiling */
    auto priority = mTilingTemplatePriority;
    if (priority == Case::kTilingTemplatePriority_Invalid) {
        return false;
    }
    tilingParam.ret = Ops::Transformer::OpTiling::TilingRegistryNew::GetInstance().DoTilingImpl(tilingParam.ctx, {priority});
    return true;
}

void MlaPrologCase::PreTilingRunCbf_SetPlatformInfoNull(MlaPrologCase::DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return;
    }
    const auto compute_node_info = tilingParam.ctx->GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
        return;
    }
    /* PlatformInfo 位于 Inputs 和 Outputs 之后 */
    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum() + 1U;
    auto kernelContext = (gert::KernelContext *)tilingParam.ctx;
    kernelContext->GetContext()->values[index] = nullptr;
}
