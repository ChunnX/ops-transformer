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
 * \file grouped_matmul_swiglu_quant_case.cpp
 * \brief GroupedMatmulSwigluQuant 测试用例.
 */
#include "grouped_matmul_swiglu_quant_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define GMMSQ_KERNEL_PARAM                                                                                   \
    (__gm__ uint8_t * x, __gm__ uint8_t * weight, __gm__ uint8_t * weight_scale, __gm__ uint8_t * x_scale, \
     __gm__ uint8_t * group_list, __gm__ uint8_t * y, __gm__ uint8_t * y_scale,                            \
     __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)

typedef void(*GmmSwigluQuantKernelFunc) GMMSQ_KERNEL_PARAM;

extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant_fp32 GMMSQ_KERNEL_PARAM;

extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant_bf16 GMMSQ_KERNEL_PARAM;

extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant_fp16 GMMSQ_KERNEL_PARAM;

using namespace ops::adv::tests::grouped_matmul_swiglu_quant;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

enum class KernelInParams {
    X = 0,
    WEIGHT,
    WEIGHTSCALE,
    XSCALE,
    GROUPLIST
};

enum class KernelOutParams {
    Y = 0,
    YSCALE
};

bool RunGmmSwigluQuant(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                            std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (GmmSwigluQuantKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, 
                inputs[static_cast<int>(KernelInParams::X)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::WEIGHT)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::WEIGHTSCALE)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::XSCALE)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::GROUPLIST)]->GetDevData(),
                outputs[static_cast<int>(KernelOutParams::Y)]->GetDevData(),
                outputs[static_cast<int>(KernelOutParams::YSCALE)]->GetDevData(), 
                workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingGMMSwigluQuantStub(gert::TilingContext *context)
{
    auto *gmmSwigluQuantCase = static_cast<GmmSwigluQuantCase *>(Case::GetCurrentCase());
    if (gmmSwigluQuantCase != nullptr) {
        GmmSwigluQuantCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!gmmSwigluQuantCase->DoOpTiling(p)) {
            return p.ret;
        }
        return gmmSwigluQuantCase->GmmSwigluQuantTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool GmmSwigluQuantCase::InitParam()
{
    x = Tensor("x", {mParam.m, mParam.k}, "", ge::DataType::DT_INT8, ge::FORMAT_ND);
    weight = Tensor("weight", {mParam.e, mParam.n / 32, mParam.k / 16, 16, 32}, "", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ);
    weightScale = Tensor("weightScale", {mParam.e, mParam.n}, "", mParam.weightScaleDataType, ge::FORMAT_ND);
    xScale = Tensor("xScale", {mParam.m,}, "", mParam.xScaleDataType, ge::FORMAT_ND);
    groupList = Tensor("groupList", {mParam.e,}, "", ge::DataType::DT_INT64, ge::FORMAT_ND);
    return true;
}

bool GmmSwigluQuantCase::InitOpInfo()
{
    auto *gmmSwigluQuantKernelFunc = (void *)grouped_matmul_swiglu_quant_fp32;
    if (mParam.weightScaleDataType == ge::DataType::DT_FLOAT) {
        gmmSwigluQuantKernelFunc = (void *)grouped_matmul_swiglu_quant_fp32;
    } else if (mParam.weightScaleDataType == ge::DataType::DT_BF16) {
        gmmSwigluQuantKernelFunc = (void *)grouped_matmul_swiglu_quant_bf16;
    } else if (mParam.weightScaleDataType == ge::DataType::DT_FLOAT16) {
        gmmSwigluQuantKernelFunc = (void *)grouped_matmul_swiglu_quant_fp16;
    }

    bool rst = mCtx.SetOpName("GroupedMatmulSwigluQuant");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&x, &weight, &weightScale, &xScale, &groupList});
    rst = rst && mCtx.SetOutputs({&y, &yScale});
    rst = rst && mCtx.SetKernelRunCbf(RunGmmSwigluQuant);
    rst = rst && mCtx.SetTilingDataMaxSize(2984); // 2984 : max tilingDataLen
    rst = rst && mCtx.SetKernelMainFunc(gmmSwigluQuantKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    GmmSwigluQuantTilingFunc = reinterpret_cast<gert::OpImplRegisterV2::TilingKernelFunc>(platform->LoadOpTilingSoSym("TilingGMMSwigluQuant"));
    if (GmmSwigluQuantTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, GmmSwigluQuant(%p)", GmmSwigluQuantTilingFunc);
        return false;
    }
    IMPL_OP(GroupedMatmulSwigluQuant).Tiling(TilingGMMSwigluQuantStub);
    return rst;
}

bool GmmSwigluQuantCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool GmmSwigluQuantCase::Run()
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

GmmSwigluQuantCase::GmmSwigluQuantCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(opInfo)), mParam(std::move(param))
{
    this->mOpInfo.mName = "GroupedMatmulSwigluQuant";
}

GmmSwigluQuantCase::GmmSwigluQuantCase()
{
}

GmmSwigluQuantCase::Param::Param()
{
}

GmmSwigluQuantCase::Param::Param(int64_t E, int64_t M, int64_t K, int64_t N, ge::DataType WeightScaleDataType)
    : e(E), m(M), k(K), n(N), weightScaleDataType(WeightScaleDataType)
{
}

bool GmmSwigluQuantCase::DoOpTiling(DoTilingParam& tilingParam) {
  if (tilingParam.ctx == nullptr) {
    return false;
  }
  return true;
}