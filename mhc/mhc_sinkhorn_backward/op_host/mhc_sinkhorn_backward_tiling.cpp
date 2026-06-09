/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_sinkhorn_backward_tiling.cpp
 * \brief mhc_sinkhorn_backward_tiling
 */

#include <vector>
#include <sstream>
#include <iostream>
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "mhc_sinkhorn_backward_tiling.h"

using namespace AscendC;
namespace optiling {

const static int64_t GRAD_Y_INPUT_INDEX = 0;
const static int64_t NORM_INPUT_INDEX = 1;
const static int64_t SUM_INPUT_INDEX = 2;
const static int64_t GRAD_INPUT_OUTPUT_INDEX = 0;

const static int64_t GRAD_Y_INPUT_BS_DIMNUM = 4;
const static int64_t GRAD_Y_INPUT_T_DIMNUM = 3;
const static int64_t NORM_INPUT_DIMNUM = 1;
const static int64_t SUM_INPUT_DIMNUM = 1;
const static int64_t GRAD_INPUT_OUTPUT_BS_DIMNUM = 4;
const static int64_t GRAD_INPUT_OUTPUT_T_DIMNUM = 3;

const static int64_t MIN_PER_CORE_ELEMENTS = 32;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;

constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 0;

template <typename ShapeType>
static void PrintShapeInfo(const char *shapeName, const ShapeType *shape)
{
    if (shape == nullptr) {
        std::cout << "[MhcSinkhornBackward][Tiling][Shape] " << shapeName << " is nullptr" << std::endl;
        return;
    }
    std::cout << "[MhcSinkhornBackward][Tiling][Shape] " << shapeName << " dimNum=" << shape->GetDimNum()
              << " dims=[";
    for (size_t i = 0; i < shape->GetDimNum(); i++) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << shape->GetDim(i);
    }
    std::cout << "]" << std::endl;
}

ge::graphStatus MhcSinkhornBackwardTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();

    auto gradYShapePtr = context_->GetInputShape(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYShapePtr);
    gradYShape_ = &gradYShapePtr->GetStorageShape();
    
    auto normShapePtr = context_->GetInputShape(NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, normShapePtr);
    normShape_ = &normShapePtr->GetStorageShape();

    auto sumShapePtr = context_->GetInputShape(SUM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sumShapePtr);
    sumShape_ = &sumShapePtr->GetStorageShape();

    auto gradInputShapePtr = context_->GetOutputShape(GRAD_INPUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradInputShapePtr);
    gradInputShape_ = &gradInputShapePtr->GetStorageShape();

    std::cout << "\n[MhcSinkhornBackward][Tiling][GetShapeAttrsInfo] opName=" << opName_ << std::endl;
    PrintShapeInfo("grad_y", gradYShape_);
    PrintShapeInfo("norm", normShape_);
    PrintShapeInfo("sum", sumShape_);
    PrintShapeInfo("grad_input", gradInputShape_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const MhcSinkhornBackwardCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;

    std::cout << "[MhcSinkhornBackward][Tiling][GetPlatformInfo] totalCoreNum=" << totalCoreNum_
              << ", ubSize=" << ubSize_ << std::endl;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckDtype()
{
    auto gradYDesc = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYDesc);
    auto gradYDtype = gradYDesc->GetDataType();
    OP_CHECK_IF((gradYDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(opName_,
            "grad_y dtype %s error, only supports float32. please check.",
            ge::TypeUtils::DataTypeToSerialString(gradYDtype).c_str()),
        return ge::GRAPH_FAILED);

    inputDTypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(gradYDesc->GetDataType()));

    auto normDesc = context_->GetInputDesc(NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, normDesc);
    auto normDtype = normDesc->GetDataType();
    OP_CHECK_IF((normDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(opName_,
            "norm dtype %s error, only supports float32. please check.",
            ge::TypeUtils::DataTypeToSerialString(normDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto sumDesc = context_->GetInputDesc(SUM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sumDesc);
    auto sumDtype = sumDesc->GetDataType();
    OP_CHECK_IF((sumDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(opName_,
            "sum dtype %s error, only supports float32. please check.",
            ge::TypeUtils::DataTypeToSerialString(sumDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto gradInputDesc = context_->GetOutputDesc(GRAD_INPUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradInputDesc);
    auto gradInputDtype = gradInputDesc->GetDataType();
    OP_CHECK_IF((gradInputDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(opName_,
            "grad_input dtype %s error, only supports float32. please check.",
            ge::TypeUtils::DataTypeToSerialString(gradInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    std::cout << "[MhcSinkhornBackward][Tiling][CheckDtype] grad_y="
              << ge::TypeUtils::DataTypeToSerialString(gradYDtype)
              << ", norm=" << ge::TypeUtils::DataTypeToSerialString(normDtype)
              << ", sum=" << ge::TypeUtils::DataTypeToSerialString(sumDtype)
              << ", grad_input=" << ge::TypeUtils::DataTypeToSerialString(gradInputDtype)
              << ", inputDTypeSize=" << inputDTypeSize_ << std::endl;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckGradYShape()
{
    std::cout << "[MhcSinkhornBackward][Tiling][CheckGradYShape] isTShape=" << std::boolalpha << isTShape_
              << ", nSize=" << nSize_ << std::noboolalpha << std::endl;

    if (isTShape_) {
        std::cout << "[MhcSinkhornBackward][Tiling][CheckGradYShape] grad_y dim1="
                  << gradYShape_->GetDim(DIM_1) << ", dim2=" << gradYShape_->GetDim(DIM_2) << std::endl;
        if (gradYShape_->GetDim(DIM_2) != nSize_) {
            OP_LOGE(opName_, "The dim2 and dim3 of grad_y shape should be equal.");
            return ge::GRAPH_FAILED;
        }
    } else {
        std::cout << "[MhcSinkhornBackward][Tiling][CheckGradYShape] grad_y dim2="
                  << gradYShape_->GetDim(DIM_2) << ", dim3=" << gradYShape_->GetDim(DIM_3) << std::endl;
        if (gradYShape_->GetDim(DIM_3) != nSize_) {
            OP_LOGE(opName_, "The dim2 and dim3 of grad_y shape should be equal.");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckNormShape()
{
    int64_t expectNormDim0 = 2 * numIters_ * totalLength_ * nSize_ * nAlignSize_;
    std::cout << "[MhcSinkhornBackward][Tiling][CheckNormShape] actualNormDim0="
              << normShape_->GetDim(DIM_0) << ", expectNormDim0=" << expectNormDim0
              << ", numIters=" << numIters_ << ", totalLength=" << totalLength_
              << ", nSize=" << nSize_ << ", nAlignSize=" << nAlignSize_ << std::endl;

    if (normShape_->GetDim(DIM_0) != expectNormDim0) {
        OP_LOGE(opName_, "The norm shape is invalid, it should be {%ld}.", expectNormDim0);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckSumShape()
{
    int64_t expectSumDim0 = 2 * numIters_ * totalLength_ * nAlignSize_;
    std::cout << "[MhcSinkhornBackward][Tiling][CheckSumShape] actualSumDim0="
              << sumShape_->GetDim(DIM_0) << ", expectSumDim0=" << expectSumDim0
              << ", numIters=" << numIters_ << ", totalLength=" << totalLength_
              << ", nAlignSize=" << nAlignSize_ << std::endl;

    if (sumShape_->GetDim(DIM_0) != expectSumDim0) {
        OP_LOGE(opName_, "The sum shape is invalid, it should be {%ld}.", expectSumDim0);
        return ge::GRAPH_FAILED;
    }

    int64_t numIters = sumShape_->GetDim(DIM_0) / (2 * totalLength_ * nAlignSize_);
    std::cout << "[MhcSinkhornBackward][Tiling][CheckSumShape] numItersFromSum="
              << numIters << ", numIters=" << numIters_ << std::endl;

    if (numIters != numIters_) {
        OP_LOGE(opName_, "The numIters of Sum is invalid, it should be {%ld}.", numIters_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckGradInputShape()
{
    std::cout << "[MhcSinkhornBackward][Tiling][CheckGradInputShape] isTShape="
              << std::boolalpha << isTShape_ << std::noboolalpha << std::endl;

    if (isTShape_) {
        std::cout << "[MhcSinkhornBackward][Tiling][CheckGradInputShape] actual dimNum="
                  << gradInputShape_->GetDimNum() << ", expect dimNum=" << GRAD_INPUT_OUTPUT_T_DIMNUM
                  << ", actual=[" << gradInputShape_->GetDim(DIM_0) << ", "
                  << gradInputShape_->GetDim(DIM_1) << ", " << gradInputShape_->GetDim(DIM_2)
                  << "], expect=[" << tSize_ << ", " << nSize_ << ", " << nSize_ << "]" << std::endl;

        if (gradInputShape_->GetDimNum() != GRAD_INPUT_OUTPUT_T_DIMNUM) {
            OP_LOGE(opName_, "The grad_intput shape dim num is invalid");
            return ge::GRAPH_FAILED;
        }

        if (gradInputShape_->GetDim(DIM_0) != tSize_ ||
            gradInputShape_->GetDim(DIM_1) != nSize_ ||
            gradInputShape_->GetDim(DIM_2) != nSize_) {
            OP_LOGE(opName_, "The grad_input shape is invalid, it should be {%ld, %ld, %ld}.", tSize_,
                nSize_, nSize_);
            return ge::GRAPH_FAILED;
        }
    } else {
        std::cout << "[MhcSinkhornBackward][Tiling][CheckGradInputShape] actual dimNum="
                  << gradInputShape_->GetDimNum() << ", expect dimNum=" << GRAD_INPUT_OUTPUT_BS_DIMNUM
                  << ", actual=[" << gradInputShape_->GetDim(DIM_0) << ", "
                  << gradInputShape_->GetDim(DIM_1) << ", " << gradInputShape_->GetDim(DIM_2)
                  << ", " << gradInputShape_->GetDim(DIM_3) << "], expect=["
                  << bSize_ << ", " << sSize_ << ", " << nSize_ << ", " << nSize_ << "]" << std::endl;

        if (gradInputShape_->GetDimNum() != GRAD_INPUT_OUTPUT_BS_DIMNUM) {
            OP_LOGE(opName_, "The grad_intput shape dim num is invalid");
            return ge::GRAPH_FAILED;
        }

        if (gradInputShape_->GetDim(DIM_0) != bSize_ ||
            gradInputShape_->GetDim(DIM_1) != sSize_ ||
            gradInputShape_->GetDim(DIM_2) != nSize_ ||
            gradInputShape_->GetDim(DIM_3) != nSize_) {
            OP_LOGE(opName_, "The grad_input shape is invalid, it should be {%ld, %ld, %ld, %ld}.", bSize_,
                sSize_, nSize_, nSize_);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckShape()
{
    OP_CHECK_IF(gradYShape_ == nullptr,
        OP_LOGE(opName_, "grad_y shape is null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(normShape_ == nullptr,
        OP_LOGE(opName_, "norm shape is null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(sumShape_ == nullptr,
        OP_LOGE(opName_, "sum shape is null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gradInputShape_ == nullptr,
        OP_LOGE(opName_, "grad_intput shape is null"),
        return ge::GRAPH_FAILED);

    std::cout << "\n[MhcSinkhornBackward][Tiling][CheckShape] begin" << std::endl;
    PrintShapeInfo("grad_y", gradYShape_);
    PrintShapeInfo("norm", normShape_);
    PrintShapeInfo("sum", sumShape_);
    PrintShapeInfo("grad_input", gradInputShape_);
    
    if (normShape_->GetDimNum() != NORM_INPUT_DIMNUM) {
        OP_LOGE(opName_, "The norm shape dim num is invalid");
        return ge::GRAPH_FAILED;
    }
    if (sumShape_->GetDimNum() != SUM_INPUT_DIMNUM) {
        OP_LOGE(opName_, "The sum shape dim num is invalid");
        return ge::GRAPH_FAILED;
    }
    if (gradYShape_->GetDimNum() != GRAD_Y_INPUT_BS_DIMNUM &&
        gradYShape_->GetDimNum() != GRAD_Y_INPUT_T_DIMNUM) {
        OP_LOGE(opName_, "The grad_y shape dim num is invalid");
        return ge::GRAPH_FAILED;
    }

    isTShape_ = gradYShape_->GetDimNum() == GRAD_Y_INPUT_T_DIMNUM ? true : false;
    if (isTShape_) {
        tSize_ = gradYShape_->GetDim(DIM_0);
        nSize_ = gradYShape_->GetDim(DIM_1);
        totalLength_ = tSize_;
    } else {
        bSize_ = gradYShape_->GetDim(DIM_0);
        sSize_ = gradYShape_->GetDim(DIM_1);
        nSize_ = gradYShape_->GetDim(DIM_2);
        totalLength_ = bSize_ * sSize_;
    }

    std::cout << "[MhcSinkhornBackward][Tiling][CheckShape] isTShape=" << std::boolalpha << isTShape_
              << ", tSize=" << tSize_ << ", bSize=" << bSize_ << ", sSize=" << sSize_
              << ", nSize=" << nSize_ << ", totalLength=" << totalLength_ << std::noboolalpha << std::endl;

    if (nSize_ != 4 &&  nSize_ != 6 && nSize_ != 8) {
        OP_LOGE(opName_, "nSize of gradYShape is expected to be 4, 6 or 8, but the current value got %ld.",
            nSize_);
        return ge::GRAPH_FAILED;
    }

    int64_t blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t inputNumPerBlock = Ops::Base::FloorDiv(blockSize, inputDTypeSize_);
    nAlignSize_ = Ops::Base::CeilAlign(nSize_, inputNumPerBlock);

    std::cout << "[MhcSinkhornBackward][Tiling][CheckShape] blockSize=" << blockSize
              << ", inputDTypeSize=" << inputDTypeSize_
              << ", inputNumPerBlock=" << inputNumPerBlock
              << ", nAlignSize=" << nAlignSize_ << std::endl;

    int64_t normDim0 = normShape_->GetDim(DIM_0);
    if (normDim0 & 1) {
        OP_LOGE(opName_, "dim0 of norm is expected to be even, but the current value got %ld.",
            normDim0);
        return ge::GRAPH_FAILED;
    }
    numIters_ = normDim0 / (2 * totalLength_ * nSize_ * nAlignSize_);

    std::cout << "[MhcSinkhornBackward][Tiling][CheckShape] normDim0=" << normDim0
              << ", denominator=" << 2 * totalLength_ * nSize_ * nAlignSize_
              << ", numIters=" << numIters_ << std::endl;
    
    if (numIters_ > 100 || numIters_ < 1) {
        OP_LOGE(opName_, "num_Iters is expected to be [1, 100], but the current value got %ld.",
            numIters_);
        return ge::GRAPH_FAILED;
    }

    if (CheckGradYShape() != ge::GRAPH_SUCCESS ||
        CheckNormShape() != ge::GRAPH_SUCCESS ||
        CheckSumShape() != ge::GRAPH_SUCCESS ||
        CheckGradInputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    std::cout << "[MhcSinkhornBackward][Tiling][CheckShape] success" << std::endl;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::SplitCores()
{
    std::cout << "\n[MhcSinkhornBackward][Tiling][SplitCores] begin, totalLength=" << totalLength_
              << ", nSize=" << nSize_ << ", nAlignSize=" << nAlignSize_
              << ", inputDTypeSize=" << inputDTypeSize_
              << ", totalCoreNum=" << totalCoreNum_ << ", ubSize=" << ubSize_ << std::endl;

    auto splitInfo = BaseSplitCores(totalLength_, nSize_, inputDTypeSize_);

    needCoreNum_ = splitInfo.usedCoreNum;
    coreTSize_ = splitInfo.tNormCore;
    normCoreTLoops_ = splitInfo.tNormCoreLoop;
    normCorePerLoopTSize_ = splitInfo.tUbFactor;
    normCoreLastLoopTSize_ = splitInfo.tUbFactorTail;
    tailCoreTSize_ = splitInfo.tTailCore;
    tailCoreLoops_ = splitInfo.tTailCoreLoop;
    tailCorePerLoopTSize_ = splitInfo.tUbFactor;
    tailCoreLastLoopTSize_ = splitInfo.tUbTailTail;

    std::cout << "[MhcSinkhornBackward][Tiling][SplitCores] result:"
              << " needCoreNum=" << needCoreNum_
              << ", coreTSize=" << coreTSize_
              << ", normCoreTLoops=" << normCoreTLoops_
              << ", normCorePerLoopTSize=" << normCorePerLoopTSize_
              << ", normCoreLastLoopTSize=" << normCoreLastLoopTSize_
              << ", tailCoreTSize=" << tailCoreTSize_
              << ", tailCoreLoops=" << tailCoreLoops_
              << ", tailCorePerLoopTSize=" << tailCorePerLoopTSize_
              << ", tailCoreLastLoopTSize=" << tailCoreLastLoopTSize_ << std::endl;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::DoOpTiling()
{
    std::cout << "\n[MhcSinkhornBackward][Tiling][DoOpTiling] begin" << std::endl;

    auto ret = CheckDtype();
    if (ret != ge::GRAPH_SUCCESS) {
        std::cout << "[MhcSinkhornBackward][Tiling][DoOpTiling] CheckDtype failed" << std::endl;
        return ret;
    }

    ret = CheckShape();
    if (ret != ge::GRAPH_SUCCESS) {
        std::cout << "[MhcSinkhornBackward][Tiling][DoOpTiling] CheckShape failed" << std::endl;
        return ret;
    }

    ret = SplitCores();
    if (ret != ge::GRAPH_SUCCESS) {
        std::cout << "[MhcSinkhornBackward][Tiling][DoOpTiling] SplitCores failed" << std::endl;
        return ret;
    }

    SetTilingData();

    std::cout << "[MhcSinkhornBackward][Tiling][DoOpTiling] success" << std::endl;
    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornBackwardTiling::SetTilingData()
{
    MhcSinkhornBackwardTilingData *tilingData = context_->GetTilingData<MhcSinkhornBackwardTilingData>();
    tilingData->totalLength = totalLength_;
    tilingData->tSize = tSize_;
    tilingData->bSize = bSize_;
    tilingData->sSize = sSize_;
    tilingData->nSize = nSize_;
    tilingData->nAlignSize = nAlignSize_;
    tilingData->numIters = numIters_;
    tilingData->needCoreNum = needCoreNum_;
    tilingData->coreTSize = coreTSize_;
    tilingData->normCoreTLoops = normCoreTLoops_;
    tilingData->normCorePerLoopTSize = normCorePerLoopTSize_;
    tilingData->normCoreLastLoopTSize = normCoreLastLoopTSize_;
    tilingData->tailCoreTSize = tailCoreTSize_;
    tilingData->tailCoreLoops = tailCoreLoops_;
    tilingData->tailCorePerLoopTSize = tailCorePerLoopTSize_;
    tilingData->tailCoreLastLoopTSize = tailCoreLastLoopTSize_;

    std::cout << "[MhcSinkhornBackward][Tiling][SetTilingData]"
              << " totalLength=" << totalLength_
              << ", tSize=" << tSize_
              << ", bSize=" << bSize_
              << ", sSize=" << sSize_
              << ", nSize=" << nSize_
              << ", nAlignSize=" << nAlignSize_
              << ", numIters=" << numIters_
              << ", needCoreNum=" << needCoreNum_
              << ", coreTSize=" << coreTSize_
              << ", normCoreTLoops=" << normCoreTLoops_
              << ", normCorePerLoopTSize=" << normCorePerLoopTSize_
              << ", normCoreLastLoopTSize=" << normCoreLastLoopTSize_
              << ", tailCoreTSize=" << tailCoreTSize_
              << ", tailCoreLoops=" << tailCoreLoops_
              << ", tailCorePerLoopTSize=" << tailCorePerLoopTSize_
              << ", tailCoreLastLoopTSize=" << tailCoreLastLoopTSize_ << std::endl;
}

ge::graphStatus MhcSinkhornBackwardTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcSinkhornBackwardTiling::GetTilingKey() const
{
    uint64_t tilingKey = 1;
    uint64_t realTilingKey = GET_TPL_TILING_KEY(tilingKey);
    std::cout << "[MhcSinkhornBackward][Tiling][GetTilingKey] baseTilingKey=" << tilingKey
              << ", realTilingKey=" << realTilingKey << std::endl;
    return realTilingKey;
}

ge::graphStatus MhcSinkhornBackwardTiling::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = ASCENDC_TOOLS_WORKSPACE;
    std::cout << "[MhcSinkhornBackward][Tiling][GetWorkspaceSize] workspace[0]=" << workspaces[0] << std::endl;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::PostTiling()
{
    std::cout << "[MhcSinkhornBackward][Tiling][PostTiling] SetBlockDim=" << needCoreNum_
              << ", SetLocalMemorySize=" << ubSize_ << std::endl;

    context_->SetBlockDim(needCoreNum_);
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornBackwardTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "\n totalLength: " << totalLength_;
    info << "\n tSize: " << tSize_;
    info << "\n bSize: " << bSize_;
    info << "\n sSize: " << sSize_;
    info << "\n nSize: " << nSize_;
    info << "\n nAlignSize: " << nAlignSize_;
    info << "\n numIters: " << numIters_;
    info << "\n needCoreNum: " << needCoreNum_;
    info << "\n coreTSize: " << coreTSize_;
    info << "\n normCoreTLoops: " << normCoreTLoops_;
    info << "\n normCorePerLoopTSize: " << normCorePerLoopTSize_;
    info << "\n normCoreLastLoopTSize: " << normCoreLastLoopTSize_;
    info << "\n tailCoreTSize: " << tailCoreTSize_;
    info << "\n tailCoreLoops: " << tailCoreLoops_;
    info << "\n tailCorePerLoopTSize: " << tailCorePerLoopTSize_;
    info << "\n tailCoreLastLoopTSize: " << tailCoreLastLoopTSize_;
    OP_LOGI(opName_, "%s", info.str().c_str());
    std::cout << "[MhcSinkhornBackward][Tiling][DumpTilingInfo]" << info.str() << std::endl;
}

static ge::graphStatus TilingForMhcSinkhornBackward(gert::TilingContext *context)
{
    std::cout << "\n[MhcSinkhornBackward][TilingForMhcSinkhornBackward] begin" << std::endl;
    MhcSinkhornBackwardTiling tiling(context);
    auto ret = tiling.DoTiling();
    std::cout << "[MhcSinkhornBackward][TilingForMhcSinkhornBackward] end, ret=" << ret << std::endl;
    return ret;
}

static ge::graphStatus TilingPrepareForMhcSinkhornBackward([[maybe_unused]] gert::TilingParseContext *context)
{
    std::cout << "\n[MhcSinkhornBackward][TilingPrepare] begin" << std::endl;

    auto compileInfo = context->GetCompiledInfo<MhcSinkhornBackwardCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);

    std::cout << "[MhcSinkhornBackward][TilingPrepare] coreNum=" << compileInfo->coreNum
              << ", ubSize=" << compileInfo->ubSize << std::endl;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcSinkhornBackward)
    .Tiling(TilingForMhcSinkhornBackward)
    .TilingParse<MhcSinkhornBackwardCompileInfo>(TilingPrepareForMhcSinkhornBackward);
} // namespace optiling