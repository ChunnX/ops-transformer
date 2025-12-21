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
 * \file mla_preprocess_v2_case.cpp
 * \brief MlaPreprocess 测试用例.
 */
#include "mla_preprocess_v2_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling_base/tiling_base.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define MLA_PREPROCESS_V2_KERNEL_PARAM                                                                                   \
    (__gm__ uint8_t * hiddenStateGm, __gm__ uint8_t * gamma1Gm, __gm__ uint8_t * beta1Gm, __gm__ uint8_t * quantScale1Gm, \
     __gm__ uint8_t * quantOffset1Gm, __gm__ uint8_t * wdqkvGm, __gm__ uint8_t * descale1Gm, __gm__ uint8_t * bias1Gm, \
     __gm__ uint8_t * gamma2Gm, __gm__ uint8_t * beta2Gm, __gm__ uint8_t * quantScale2Gm, __gm__ uint8_t * quantOffset2Gm, \
     __gm__ uint8_t * wuqGm, __gm__ uint8_t * descale2Gm, __gm__ uint8_t * bias2Gm, __gm__ uint8_t * gamma3Gm, \
     __gm__ uint8_t * sin1Gm, __gm__ uint8_t * cos1Gm, __gm__ uint8_t * wukGm, __gm__ uint8_t * keycacheGm, \
     __gm__ uint8_t * keycacheRopeGm, __gm__ uint8_t * slotMappingGm, __gm__ uint8_t * gmCtkvScale, __gm__ uint8_t * gmQnopeScale, \
     __gm__ uint8_t * qGm, __gm__ uint8_t * keycacheOutGm, __gm__ uint8_t * qGm2, __gm__ uint8_t * keycacheOutGm2, \
     __gm__ uint8_t * qDownGm, __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)

typedef void(*MlaPreprocessV2KernelFunc) MLA_PREPROCESS_V2_KERNEL_PARAM;

extern "C" __global__ __aicore__ void mla_preprocess_v2_fp32 MLA_PREPROCESS_V2_KERNEL_PARAM;

extern "C" __global__ __aicore__ void mla_preprocess_v2_bf16 MLA_PREPROCESS_V2_KERNEL_PARAM;

extern "C" __global__ __aicore__ void mla_preprocess_v2_fp16 MLA_PREPROCESS_V2_KERNEL_PARAM;

using namespace ops::adv::tests::mla_preprocess_v2;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

enum class KernelInParams {
    HIDDENSTATEGM = 0,
    GAMMA1GM,
    BETA1GM,
    QUANTSCALE1GM,
    QUANTOFFSET1GM,
    WDQKVGM,
    DESCALE1GM,
    BIAS1GM,
    GAMMA2GM,
    BETA2GM,
    QUANTSCALE2GM,
    QUANTOFFSET2GM,
    WUQGM,
    DESCALE2GM,
    BIAS2GM,
    GAMMA3GM,
    SIN1GM,
    COS1GM,
    WUKGM,
    KEYCACHEGM,
    KEYCACHEROPEGM,
    SLOTMAPPINGGM,
    GMCTKVSCALE,
    GMQNOPESCALE
};

enum class KernelOutParams {
    QGM = 0,
    KEYCACHEOUTGM,
    QGM2,
    KEYCACHEOUTGM2,
    QDOWNGM
};

bool RunMlaPreprocessV2(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                            std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (MlaPreprocessKernelV2Func)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, 
                inputs[static_cast<int>(KernelInParams::HIDDENSTATEGM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::GAMMA1GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::BETA1GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::QUANTSCALE1GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::QUANTOFFSET1GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::WDQKVGM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::DESCALE1GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::BIAS1GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::GAMMA2GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::BETA2GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::QUANTSCALE2GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::QUANTOFFSET2GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::WUQGM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::DESCALE2GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::BIAS2GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::GAMMA3GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::SIN1GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::COS1GM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::WUKGM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::KEYCACHEGM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::KEYCACHEROPEGM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::SLOTMAPPINGGM)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::GMCTKVSCALE)]->GetDevData(),
                inputs[static_cast<int>(KernelInParams::GMQNOPESCALE)]->GetDevData(),
                outputs[static_cast<int>(KernelOutParams::QGM)]->GetDevData(),
                outputs[static_cast<int>(KernelOutParams::KEYCACHEOUTGM)]->GetDevData(), 
                outputs[static_cast<int>(KernelOutParams::QGM2)]->GetDevData(), 
                outputs[static_cast<int>(KernelOutParams::KEYCACHEOUTGM2)]->GetDevData(),
                outputs[static_cast<int>(KernelOutParams::QDOWNGM)]->GetDevData(), 
                workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingMlaPreprocessV2Stub(gert::TilingContext *context)
{
    auto *mlaPreprocessV2Case = static_cast<MlaPreprocessV2Case *>(Case::GetCurrentCase());
    if (mlaPreprocessV2Case != nullptr) {
        MlaPreprocessV2Case::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!mlaPreprocessV2Case->DoOpTiling(p)) {
            return p.ret;
        }
        return mlaPreprocessV2Case->MlaPreprocessV2TilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool MlaPreprocessV2Case::InitParam()
{
    hiddenStateGm = Tensor("hiddenStateGm", {mParam.tokenNum, mParam.hiddenNum}, "", mParam.inputDataType, ge::FORMAT_ND);
    gamma1Gm = Tensor("gamma1Gm", {mParam.hiddenNum}, "", mParam.inputDataType, ge::FORMAT_ND);
    beta1Gm = Tensor("beta1Gm", {mParam.hiddenNum}, "", mParam.inputDataType, ge::FORMAT_ND);
    quantScale1Gm = Tensor("quantScale1Gm", {1}, "", mParam.inputDataType, ge::FORMAT_ND);
    quantOffset1Gm = Tensor("quantOffset1Gm", {1}, "", ge::DataType::DT_INT8, ge::FORMAT_ND);
    if (mParam.wdqkvFormatND){
        wdqkvGm = Tensor("wdqkvGm", {2112, mParam.hiddenNum}, "", ge::DataType::DT_INT8, ge::FORMAT_ND);
    } else {
        wdqkvGm = Tensor("wdqkvGm", {2112, mParam.hiddenNum}, "", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ);
    }
    descale1Gm = Tensor("descale1Gm", {2112}, "", ge::DataType::DT_INT64, ge::FORMAT_ND);
    bias1Gm = Tensor("bias1Gm", {2112}, "", ge::DataType::DT_INT32, ge::FORMAT_ND);
    gamma2Gm = Tensor("gamma2Gm", {1536}, "", mParam.inputDataType, ge::FORMAT_ND);
    beta2Gm = Tensor("beta2Gm", {1536}, "", mParam.inputDataType, ge::FORMAT_ND);
    quantScale2Gm = Tensor("quantScale2Gm", {1}, "", mParam.inputDataType, ge::FORMAT_ND);
    quantOffset2Gm = Tensor("quantOffset2Gm", {1}, "", ge::DataType::DT_INT8, ge::FORMAT_ND);
    if (mParam.wuqFormatND){
        wuqGm = Tensor("wuqGm", {mParam.headNum * 192, 1536}, "", ge::DataType::DT_INT8, ge::FORMAT_ND);
    } else {
        wuqGm = Tensor("wuqGm", {mParam.headNum * 192, 1536}, "", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ);
    }
    descale2Gm = Tensor("descale2Gm", {mParam.headNum * 192}, "", ge::DataType::DT_INT64, ge::FORMAT_ND);
    bias2Gm = Tensor("bias2Gm", {1, mParam.headNum * 192}, "", ge::DataType::DT_INT32, ge::FORMAT_ND);
    gamma3Gm = Tensor("gamma3Gm", {512}, "", mParam.inputDataType, ge::FORMAT_ND);
    cos1Gm = Tensor("cos1Gm", {mParam.tokenNum, 64}, "", mParam.inputDataType, ge::FORMAT_ND);
    sin1Gm = Tensor("sin1Gm", {mParam.tokenNum, 64}, "", mParam.inputDataType, ge::FORMAT_ND);
    if (mParam.wukFormatND){
        wukGm = Tensor("wukGm", {mParam.headNum, 128, 512}, "", mParam.inputDataType, ge::FORMAT_ND);
    } else {
        wukGm = Tensor("wukGm", {mParam.headNum, 128, 512}, "", mParam.inputDataType, ge::FORMAT_FRACTAL_NZ);
    }
    keycacheGm = Tensor("keycacheGm", {mParam.bloackNum, mParam.bloackSize, 1, 512}, "", mParam.inputDataType, ge::FORMAT_ND);
    keycacheRopeGm = Tensor("keycacheRopeGm", {mParam.bloackNum, mParam.bloackSize, 1, 64}, "", mParam.inputDataType, ge::FORMAT_ND);
    slotMappingGm = Tensor("slotMappingGm", {mParam.tokenNum}, "", ge::DataType::DT_INT32, ge::FORMAT_ND);
    gmCtkvScale = Tensor("gmCtkvScale", {1}, "", mParam.inputDataType, ge::FORMAT_ND);
    gmQnopeScale = Tensor("gmQnopeScale", {mParam.headNum}, "", mParam.inputDataType, ge::FORMAT_ND);

    qGm = Tensor("qGm", {mParam.tokenNum, mParam.headNum, 1, 512}, "", mParam.inputDataType, ge::FORMAT_ND);
    keycacheOutGm = Tensor("keycacheOutGm", {mParam.bloackNum, mParam.bloackSize, 1, 512}, "", mParam.inputDataType, ge::FORMAT_ND);
    qGm2 = Tensor("qGm2", {mParam.tokenNum, mParam.headNum, 1, 64}, "", mParam.inputDataType, ge::FORMAT_ND);
    keycacheOutGm2 = Tensor("keycacheOutGm2", {mParam.bloackNum, mParam.bloackSize, 1, 64}, "", mParam.inputDataType, ge::FORMAT_ND);
    qDownGM = Tensor("qDownGM", {mParam.tokenNum, 1536}, "", mParam.inputDataType, ge::FORMAT_ND);

    return true;
}

bool MlaPreprocessV2Case::InitOpInfo()
{
    // 定义参数的输入数据类型
    auto *mlaPreprocessV2KernelFunc = (void *)mla_preprocess_v2_fp16;
    if (mParam.inputDataType == ge::DataType::DT_BF16) {
        mlaPreprocessV2KernelFunc = (void *)mla_preprocess_v2_bf16;
    } else if (mParam.inputDataType == ge::DataType::DT_FLOAT16) {
        mlaPreprocessV2KernelFunc = (void *)mla_preprocess_v2_fp16;
    }

    bool rst = mCtx.SetOpName("MlaPreprocessV2");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&hiddenStateGm,  &gamma1Gm,       &beta1Gm,     &quantScale1Gm, &quantOffset1Gm,
                                 &wdqkvGm,        &descale1Gm,     &bias1Gm,     &gamma2Gm,      &beta2Gm,
                                 &quantScale2Gm,  &quantOffset2Gm, &wuqGm,       &descale2Gm,    &bias2Gm,
                                 &gamma3Gm,       &cos1Gm,         &sin1Gm,      &wukGm,         &keycacheGm,
                                 &keycacheRopeGm, &slotMappingGm,  &gmCtkvScale, &gmQnopeScale});
    rst = rst && mCtx.SetOutputs({&qGm, &keycacheOutGm, &qGm2, &keycacheOutGm2, &qDownGM});
    rst = rst && mCtx.SetAttrs({{"wdq_dim", mParam.wdqDim},
                                {"q_rope_dim", mParam.qRopeDim},
                                {"k_rope_dim", mParam.kRopeDim},
                                {"epsilon", mParam.epsilon},
                                {"q_rotary_coeff", mParam.qRotaryCoeff},
                                {"k_rotary_coeff", mParam.kRotaryCoeff},
                                {"transepose_wdq", mParam.transeposeWdq},
                                {"transepose_wuq", mParam.transeposeWuq},
                                {"transepose_wuk", mParam.transeposeWuk},
                                {"cache_mode", mParam.cacheMode},
                                {"quant_mode", mParam.quantMode},
                                {"do_rms_norm", mParam.doRmsNorm},
                                {"wdkv_split_count", mParam.wdkvSplitCount},
                                {"q_down_out_flag", mParam.qDownOutFlag}});
    rst = rst && mCtx.SetKernelRunCbf(RunMlaPreprocessV2);
    rst = rst && mCtx.SetKernelMainFunc(mlaPreprocessV2KernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    MlaPreprocessV2TilingFunc = reinterpret_cast<gert::OpImplRegisterV2::TilingKernelFunc>(platform->LoadOpTilingSoSym("TilingMLAPreprocessV2"));
    if (MlaPreprocessV2TilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, MlaPreprocessV2(%p)", MlaPreprocessV2TilingFunc);
        return false;
    }
    IMPL_OP(MlaPreprocessV2).Tiling(TilingMlaPreprocessV2Stub);
    return rst;
}

bool MlaPreprocessV2Case::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool MlaPreprocessV2Case::Run()
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

MlaPreprocessV2Case::MlaPreprocessV2Case(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(opInfo)), mParam(std::move(param))
{
    this->mOpInfo.mName = "MlaPreprocessV2";
}

MlaPreprocessV2Case::MlaPreprocessV2Case()
{
}

MlaPreprocessV2Case::Param::Param()
{
}
// 属性和数据类型都在这里初始化；
MlaPreprocessV2Case::Param::Param(int64_t tokenNum_, int64_t hiddenNum_, int64_t headNum_, int64_t bloackNum_,
                                int64_t bloackSize_, int64_t wdqDim_, int64_t qRopeDim_, int64_t kRopeDim_,
                                float epsilon_, int64_t qRotaryCoeff_, int64_t kRotaryCoeff_, bool transeposeWdq_,
                                bool transeposeWuq_, bool transeposeWuk_, int64_t cacheMode_, int64_t quantMode_,
                                bool doRmsNorm_, int64_t wdkvSplitCount_, int64_t qDownOutFlag_, ge::DataType inputDataType_,
                                bool wdqkvFormatND_, bool wuqFormatND_, bool wukFormatND_)
    : tokenNum(tokenNum_), hiddenNum(hiddenNum_), headNum(headNum_), bloackNum(bloackNum_), bloackSize(bloackSize_),
      wdqDim(wdqDim_), qRopeDim(qRopeDim_), kRopeDim(kRopeDim_), epsilon(epsilon_), qRotaryCoeff(qRotaryCoeff_),
      kRotaryCoeff(kRotaryCoeff_), transeposeWdq(transeposeWdq_), transeposeWuq(transeposeWuq_),
      transeposeWuk(transeposeWuk_), cacheMode(cacheMode_), quantMode(quantMode_), doRmsNorm(doRmsNorm_),
      wdkvSplitCount(wdkvSplitCount_), qDownOutFlag_(qDownOutFlag_), inputDataType(inputDataType_), wdqkvFormatND(wdqkvFormatND_),
      wuqFormatND(wuqFormatND_), wukFormatND(wukFormatND_)
{
}

bool MlaPreprocessV2Case::DoOpTiling(DoTilingParam& tilingParam) {
  if (tilingParam.ctx == nullptr) {
    return false;
  }
  return true;
}