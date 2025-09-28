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
 * \file nsa_select_attention_infer_case.cpp
 * \brief NsaSelectAttentionInfer 测试用例.
 */
#include "nsa_select_attention_infer_case.h"
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

#define NSA_SELECT_ATTENTION_INFER_KERNEL_PARAM                                                                                 \
    (GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR topkIndices, GM_ADDR attenMask,             \
    GM_ADDR blockTable, GM_ADDR actualQSeqLengths, GM_ADDR actualKVSeqLengths, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)

using NsaSelectAttentionInferKernelFunc = void(*) NSA_SELECT_ATTENTION_INFER_KERNEL_PARAM;

extern "C" __global__ __aicore__ void nsa_select_attention_infer NSA_SELECT_ATTENTION_INFER_KERNEL_PARAM;

using namespace ops::adv::tests::NsaSelectAttentionInfer;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunNsaSelectAttentionInfer(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                        std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (NsaSelectAttentionInferKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(), 
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(), 
                inputs[7]->GetDevData(), outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingNsaSelectAttentionInferStub(gert::TilingContext *context)
{
    auto *nsaSelectAttentionInferCase = static_cast<NsaSelectAttentionInferCase *>(Case::GetCurrentCase());
    if (nsaSelectAttentionInferCase != nullptr) {
    NsaSelectAttentionInferCase::DoTilingParam p;
    p.ctx = context;
    p.ret = ge::GRAPH_SUCCESS;
    p.actSeqSelQLenTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(6));
    p.actSeqSelKVLenTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(7));
    p.blocktableTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(5));
    p.topkIndicesTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(3));
    if (!nsaSelectAttentionInferCase->DoOpTiling(p)) {
        return p.ret;
    }
    return nsaSelectAttentionInferCase->nsaSelectAttentionInferTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool NsaSelectAttentionInferCase::InitParam()
{
    if (mParam.inputLayout == "BSND") {
        BSNDInitParam();
    } else if (mParam.inputLayout == "TND") {
        TNDInitParam();
    } else {
        BSHInitParam();
    }

    if (mParam.inputLayout == "TND") {
        topkIndices = Tensor("topkIndices", {mParam.numtokens, mParam.headSizeV, mParam.selectedBlockCount}, mParam.inputLayout.c_str(), ge::DT_INT32,
            ge::FORMAT_ND);
    } else {
        topkIndices = Tensor("topkIndices", {mParam.batchSize, mParam.qSeqSize, mParam.headSizeV, mParam.selectedBlockCount}, mParam.inputLayout.c_str(), ge::DT_INT32,
            ge::FORMAT_ND);
    }

    attenMask = Tensor("attenMask", {mParam.batchSize}, "1", ge::DT_BOOL, ge::FORMAT_ND); // not used now
    blocktable = Tensor("blocktable", {mParam.batchSize, mParam.maxBlockNumPerBatch}, mParam.inputLayout.c_str(), ge::DT_INT32, ge::FORMAT_ND);
    
    if (!mParam.actualSeqQLenList.empty()) {
        actualQSeqLengths = Tensor("actualQSeqLengths", {mParam.batchSize}, mParam.inputLayout.c_str(), ge::DataType::DT_INT64, ge::FORMAT_ND,
                                    Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        LOG_ERR("Currently, actualQSeqLengths must not be empty.");
    }

    TopkInitParam();

    if (!mParam.actualSeqKVLenList.empty()) {
        actualKvSeqLengths = Tensor("actualKvSeqLengths", {mParam.batchSize}, mParam.inputLayout.c_str(), ge::DataType::DT_INT64, ge::FORMAT_ND,
                                    Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        LOG_ERR("Currently, actualKvSeqLengths must not be empty.");
    }

    for (int blockId = 0; blockId < mParam.batchSize * mParam.maxBlockNumPerBatch; blockId++) {
        blocktableData.push_back(0);
    }

    if (!InitTensor(actualKvSeqLengths, mParam.actualSeqKVLenList)) {
        return false;
    }

    if (!InitTensor(actualQSeqLengths, mParam.actualSeqQLenList)) {
        return false;
    }

    if (!InitTensor(topkIndices, topkData)) {
        return false;
    }
    if (!InitTensor(blocktable, blocktableData)) {
        return false;
    }
    return true;
}

void NsaSelectAttentionInferCase::BSNDInitParam()
{
    query = Tensor("query", {mParam.batchSize, mParam.qSeqSize, mParam.headSize, mParam.headDim}, mParam.inputLayout.c_str(), mParam.optionalDataType,
        ge::FORMAT_ND);
    key = Tensor("key", {mParam.maxBlockNumPerBatch, mParam.blockSize, mParam.headSizeV, mParam.headDim}, mParam.inputLayout.c_str(), mParam.optionalDataType,
        ge::FORMAT_ND);
    value = Tensor("value", {mParam.maxBlockNumPerBatch, mParam.blockSize, mParam.headSizeV, mParam.headDimV}, mParam.inputLayout.c_str(), mParam.optionalDataType,
        ge::FORMAT_ND);
    attentionOut = Tensor("attentionOut", {mParam.batchSize, mParam.qSeqSize, mParam.headSize, mParam.headDimV}, mParam.inputLayout.c_str(),
        mParam.optionalDataType, ge::FORMAT_ND);
}

void NsaSelectAttentionInferCase::BSHInitParam()
{
    query = Tensor("query", {mParam.batchSize, mParam.qSeqSize, mParam.headSize*mParam.headDim}, mParam.inputLayout.c_str(), mParam.optionalDataType,
        ge::FORMAT_ND);
    key = Tensor("key", {mParam.maxBlockNumPerBatch, mParam.blockSize, mParam.headSizeV*mParam.headDim}, mParam.inputLayout.c_str(), mParam.optionalDataType,
        ge::FORMAT_ND);
    value = Tensor("value", {mParam.maxBlockNumPerBatch, mParam.blockSize, mParam.headSizeV*mParam.headDimV}, mParam.inputLayout.c_str(), mParam.optionalDataType,
        ge::FORMAT_ND);
    attentionOut = Tensor("attentionOut", {mParam.batchSize, mParam.qSeqSize, mParam.headSize*mParam.headDimV}, mParam.inputLayout.c_str(),
        mParam.optionalDataType, ge::FORMAT_ND);
}

void NsaSelectAttentionInferCase::TopkInitParam()
{
    if (mParam.inputLayout == "TND") {
        InitTopkDataTND();
    } else {
        InitTopkDataOtherLayout();
    }
}

void NsaSelectAttentionInferCase::InitTopkDataTND()
{
    for (int tokenId = 0; tokenId < mParam.numtokens; tokenId++) {
        InitTopkDataForHead();
    }
}

void NsaSelectAttentionInferCase::InitTopkDataForHead()
{
    for (int headSizeId = 0; headSizeId < mParam.headSizeV; headSizeId++) {
        InitTopkDataForSelectedBlockSize();
    }
}

void NsaSelectAttentionInferCase::InitTopkDataForSelectedBlockSize()
{
    for (int selectId = 0; selectId < mParam.selectedBlockCount; selectId++) {
        topkData.push_back(0);
    }
}

void NsaSelectAttentionInferCase::InitTopkDataOtherLayout()
{
    for (int batchId = 0; batchId < mParam.batchSize; batchId++) {
        InitTopkDataForS1();
    }
}

void NsaSelectAttentionInferCase::InitTopkDataForS1()
{
    for (int sId = 0; sId < mParam.qSeqSize; sId++) {
        InitTopkDataForHead();
    }
}

void NsaSelectAttentionInferCase::TNDInitParam()
{
    query = Tensor("query", {mParam.numtokens, mParam.headSize, mParam.headDim}, mParam.inputLayout.c_str(), mParam.optionalDataType,
        ge::FORMAT_ND);
    key = Tensor("key", {mParam.maxBlockNumPerBatch, mParam.blockSize, mParam.headSizeV * mParam.headDim}, mParam.inputLayout.c_str(), mParam.optionalDataType,
        ge::FORMAT_ND);
    value = Tensor("value", {mParam.maxBlockNumPerBatch, mParam.blockSize, mParam.headSizeV * mParam.headDimV}, mParam.inputLayout.c_str(), mParam.optionalDataType,
        ge::FORMAT_ND);
    attentionOut = Tensor("attentionOut", {mParam.numtokens, mParam.headSize , mParam.headDimV}, mParam.inputLayout.c_str(),
        mParam.optionalDataType, ge::FORMAT_ND);
}

bool NsaSelectAttentionInferCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("NsaSelectAttentionInfer");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&query, &key, &value, &topkIndices, &attenMask, &blocktable, &actualQSeqLengths, &actualKvSeqLengths});
    rst = rst && mCtx.SetOutputs({&attentionOut});
    rst = rst && mCtx.SetAttrs({{"input_layout", mParam.inputLayout},
                                {"num_heads", mParam.headSize},
                                {"num_key_value_heads", mParam.headSizeV},
                                {"selected_block_size", mParam.selectedBlockSize},
                                {"selected_block_count", mParam.selectedBlockCount},
                                {"block_size", mParam.blockSize},
                                {"scale_value", mParam.scaleValue},
                                {"sparse_mode", mParam.sparseMode}});
    rst = rst && mCtx.SetKernelRunCbf(RunNsaSelectAttentionInfer);
    rst = rst && mCtx.SetKernelMainFunc((void *)nsa_select_attention_infer);
    rst = rst && mOpInfo.SetContext(&mCtx);
    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }
    nsaSelectAttentionInferTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingNsaSelectAttentionInfer");

    if (nsaSelectAttentionInferTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, nsaSelectAttentionInfer(%p)", nsaSelectAttentionInferTilingFunc);
        return false;
    }
    
    IMPL_OP(NsaSelectAttentionInfer).Tiling(TilingNsaSelectAttentionInferStub);
    return rst;
}

bool NsaSelectAttentionInferCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool NsaSelectAttentionInferCase::Run()
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

NsaSelectAttentionInferCase::NsaSelectAttentionInferCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                        Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "NsaSelectAttentionInfer";
}

NsaSelectAttentionInferCase::NsaSelectAttentionInferCase()
{
}
NsaSelectAttentionInferCase::Param::Param()
{
}
NsaSelectAttentionInferCase::Param::Param(int64_t pBatchSize, int64_t pQSeqSize, int64_t pHeadSize, int64_t pHeadDim,
                                int64_t pMaxBlockNumPerBatch, int64_t pBlockSize, int64_t pSeqSize, int64_t pHeadDimV,
                                int64_t pHeadSizeV, int64_t pSelectedBlockSize, int64_t pSelectedBlockCount,
                                int64_t pSparseMode, float pScaleValue, std::string pInputLayout,
                                ge::DataType pOptionalDataType,std::vector<int64_t> pActualSeqQLenList, std::vector<int64_t> pActualSeqKVLenList,  int64_t pnumtokens)
    : batchSize(pBatchSize), qSeqSize(pQSeqSize), headSize(pHeadSize), headDim(pHeadDim), maxBlockNumPerBatch(pMaxBlockNumPerBatch),
    blockSize(pBlockSize), seqSize(pSeqSize), headDimV(pHeadDimV), headSizeV(pHeadSizeV), selectedBlockSize(pSelectedBlockSize), selectedBlockCount(pSelectedBlockCount), sparseMode(pSparseMode),
    scaleValue(pScaleValue), inputLayout(pInputLayout), optionalDataType(pOptionalDataType), actualSeqQLenList(std::move(pActualSeqQLenList)), actualSeqKVLenList(std::move(pActualSeqKVLenList)), numtokens(pnumtokens)
{
}

bool NsaSelectAttentionInferCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (tilingParam.actSeqSelQLenTensor != nullptr) {
        tilingParam.actSeqSelQLenTensor->SetData(gert::TensorData{mParam.actualSeqQLenList.data()});
    }
    if (tilingParam.actSeqSelKVLenTensor != nullptr) {
        tilingParam.actSeqSelKVLenTensor->SetData(gert::TensorData{mParam.actualSeqKVLenList.data()});
    }
    return true;
}