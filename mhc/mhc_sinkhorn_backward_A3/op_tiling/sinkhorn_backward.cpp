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
 * \file ai_infra_sinkhorn_grad_tiling.cpp
 * \brief
 */

#include <cmath>
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_base.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "ai_infra_sinkhorn_grad_tiling.h"

namespace optiling {

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216;  // 预留16M
const static uint64_t TILING_KEY_GENERALIZED = 0;

const static int64_t GRAD_OUTPUT_INPUT_INDEX = 0;
const static int64_t NORM_OUT_INPUT_INDEX = 1;
const static int64_t SUM_OUT_INPUT_INDEX = 2;
const static int64_t GRAD_INPUT_OUTPUT_INDEX = 0;

const static int64_t GRAD_OUTPUT_INPUT_BS_DIMNUM = 4;
const static int64_t GRAD_OUTPUT_INPUT_T_DIMNUM = 3;
const static int64_t NORM_OUTPUT_INPUT_BS_DIMNUM = 5;
const static int64_t NORM_OUTPUT_INPUT_T_DIMNUM = 4;
const static int64_t SUM_OUTPUT_INPUT_BS_DIMNUM = 4;
const static int64_t SUM_OUTPUT_INPUT_T_DIMNUM = 3;
const static int64_t GRAD_INPUT_OUTPUT_BS_DIMNUM = 4;
const static int64_t GRAD_INPUT_OUTPUT_T_DIMNUM = 3;

const static int64_t AVALABLE_UB_SIZE = 190 * 1024;  // 最多用 190KB
const static int64_t DTYPE_SIZE = 4;  // flaot32
const static int64_t MIN_PER_CORE_ELEMENTS = 32;
const static int64_t TRANSPOSE_BUFFER_SIZE = 128;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;


class AiInfraSinkhornGradTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AiInfraSinkhornGradTilingBase(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~AiInfraSinkhornGradTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }

    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckOutputShape();
    ge::graphStatus CheckGradOutputShape();
    ge::graphStatus CheckNormOutShape();
    ge::graphStatus CheckSumOutShape();
    ge::graphStatus SplitCores();

    const gert::Shape *gradOutputShape_ = nullptr;
    const gert::Shape *normOutShape_ = nullptr;
    const gert::Shape *sumOutShape_ = nullptr;
    const gert::Shape *gradInputShape_ = nullptr;

    int64_t totalLength_ = 0;
    int64_t inputDtypeSize_ = 0;
    const char *opName_ = "";

    bool isTShape_ = false;

    int64_t tSize_ = 0;
    int64_t bSize_ = 0;
    int64_t sSize_ = 0;
    int64_t nSize_ = 0;
    int64_t numIters_ = 0;

    AiInfraSinkhornGradTilingData tilingData_;
};

ge::graphStatus AiInfraSinkhornGradTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "fail to get platform info"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();

    // 获取输入shape信息
    auto gradOutputShapePtr = context_->GetInputShape(GRAD_OUTPUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradOutputShapePtr);
    gradOutputShape_ = &gradOutputShapePtr->GetStorageShape();

    auto normOutShapePtr = context_->GetInputShape(NORM_OUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, normOutShapePtr);
    normOutShape_ = &normOutShapePtr->GetStorageShape();

    auto sumOutShapePtr = context_->GetInputShape(SUM_OUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sumOutShapePtr);
    sumOutShape_ = &sumOutShapePtr->GetStorageShape();

    // 获取输出shape
    auto gradInputShapePtr = context_->GetOutputShape(GRAD_INPUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradInputShapePtr);
    gradInputShape_ = &gradInputShapePtr->GetStorageShape();

    // 验证数据类型
    auto gradOutputDesc = context_->GetInputDesc(GRAD_OUTPUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradOutputDesc);
    auto gradOutputDtype = gradOutputDesc->GetDataType();
    OP_CHECK_IF((gradOutputDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(context_,
            "grad_output dtype %s error, only supports float32. please check.",
            ge::TypeUtils::DataTypeToSerialString(gradOutputDtype).c_str()),
        return ge::GRAPH_FAILED);

    inputDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(gradOutputDesc->GetDataType()));

    auto normOutDesc = context_->GetInputDesc(NORM_OUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, normOutDesc);
    auto normOutDtype = normOutDesc->GetDataType();
    OP_CHECK_IF((normOutDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(context_,
            "norm_Out dtype %s error, only supports float32. please check.",
            ge::TypeUtils::DataTypeToSerialString(normOutDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto sumOutDesc = context_->GetInputDesc(SUM_OUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sumOutDesc);
    auto sumOutDtype = sumOutDesc->GetDataType();
    OP_CHECK_IF((sumOutDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(context_,
            "sum_out dtype %s error, only supports float32. please check.",
            ge::TypeUtils::DataTypeToSerialString(sumOutDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto gradInputDesc = context_->GetOutputDesc(GRAD_INPUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradInputDesc);
    auto gradInputDtype = gradInputDesc->GetDataType();
    OP_CHECK_IF((gradInputDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(context_,
            "grad_input dtype %s error, only supports float32. please check.",
            ge::TypeUtils::DataTypeToSerialString(gradInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::CheckGradOutputShape()
{

    if (isTShape_) {
        if (gradOutputShape_->GetDim(DIM_2) != nSize_) {
            OP_LOGE(context_, "The dim2 and dim3 of grad_output shape should be equal.");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (gradOutputShape_->GetDim(DIM_3) != nSize_) {
            OP_LOGE(context_, "The dim2 and dim3 of grad_output shape should be equal.");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::CheckNormOutShape()
{

    if (isTShape_) {
        if (normOutShape_->GetDim(DIM_1) == nSize_ &&
            normOutShape_->GetDim(DIM_2) == nSize_ &&
            normOutShape_->GetDim(DIM_3) == tSize_) {
            return ge::GRAPH_SUCCESS;
        }

        OP_LOGE(context_, "The norm_out shape is invalid, it should be {%ld, %ld, %ld, %ld}.", numIters_,
            nSize_, nSize_, tSize_);
    } else {
        if (normOutShape_->GetDim(DIM_1) == nSize_ &&
            normOutShape_->GetDim(DIM_2) == nSize_ &&
            normOutShape_->GetDim(DIM_3) == bSize_ &&
            normOutShape_->GetDim(DIM_4) == sSize_) {
            return ge::GRAPH_SUCCESS;
        }

        OP_LOGE(context_, "The norm_out shape is invalid, it should be {%ld, %ld, %ld, %ld, %ld}.", numIters_,
            nSize_, nSize_, bSize_, sSize_);
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::CheckSumOutShape()
{

    if (isTShape_) {
        if (sumOutShape_->GetDim(DIM_0) == 2 * numIters_ &&
            sumOutShape_->GetDim(DIM_1) == nSize_ &&
            sumOutShape_->GetDim(DIM_2) == tSize_) {
            return ge::GRAPH_SUCCESS;
        }

        OP_LOGE(context_, "The sum_out shape is invalid, it should be {%ld, %ld, %ld}.", numIters_,
            nSize_, tSize_);
    } else {
        if (sumOutShape_->GetDim(DIM_0) == 2 * numIters_ &&
            sumOutShape_->GetDim(DIM_1) == nSize_ &&
            sumOutShape_->GetDim(DIM_2) == bSize_ &&
            sumOutShape_->GetDim(DIM_3) == sSize_) {
            return ge::GRAPH_SUCCESS;
        }

        OP_LOGE(context_, "The sum_out shape is invalid, it should be {%ld, %ld, %ld, %ld}.", numIters_,
            nSize_, bSize_, sSize_);
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::CheckInputShape()
{
    OP_CHECK_IF(gradOutputShape_ == nullptr,
        OP_LOGE(context_, "grad_output shape is null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(normOutShape_ == nullptr,
        OP_LOGE(context_, "norm_output shape is null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(sumOutShape_ == nullptr,
        OP_LOGE(context_, "sum_output shape is null"),
        return ge::GRAPH_FAILED);

    isTShape_ = gradOutputShape_->GetDimNum() == GRAD_OUTPUT_INPUT_T_DIMNUM ? true : false;
    if (isTShape_) {
        if (normOutShape_->GetDimNum() != NORM_OUTPUT_INPUT_T_DIMNUM ||
            sumOutShape_->GetDimNum() != SUM_OUTPUT_INPUT_T_DIMNUM) {
            OP_LOGE(context_, "Input shape dim num is invalid");
            return ge::GRAPH_FAILED;
        }
        tSize_ = gradOutputShape_->GetDim(DIM_0);
        nSize_ = gradOutputShape_->GetDim(DIM_1);
        totalLength_ = tSize_;
        
    } else {
        if (gradOutputShape_->GetDimNum() != GRAD_OUTPUT_INPUT_BS_DIMNUM ||
            normOutShape_->GetDimNum() != NORM_OUTPUT_INPUT_BS_DIMNUM ||
            sumOutShape_->GetDimNum() != SUM_OUTPUT_INPUT_BS_DIMNUM) {
            OP_LOGE(context_, "Input shape dim num is invalid");
            return ge::GRAPH_FAILED;
        }
        bSize_ = gradOutputShape_->GetDim(DIM_0);
        sSize_ = gradOutputShape_->GetDim(DIM_1);
        nSize_ = gradOutputShape_->GetDim(DIM_2);
        totalLength_ = bSize_ * sSize_;
    }
    int64_t normOutDim0 = normOutShape_->GetDim(DIM_0);
    if (normOutDim0 & 1) {
        OP_LOGE(context_, "dim0 of norm_Out is expected to be even, but the current value got %ld.",
            normOutDim0);
        return ge::GRAPH_FAILED;
    }
    numIters_ = normOutDim0 / 2;

    if (numIters_ > 100 || numIters_ < 1) {
        OP_LOGE(context_, "num_Iters is expected to be [1, 100], but the current value got %ld.",
            numIters_);
        return ge::GRAPH_FAILED;
    }

    if (CheckGradOutputShape() != ge::GRAPH_SUCCESS ||
        CheckNormOutShape() != ge::GRAPH_SUCCESS ||
        CheckSumOutShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    
    tilingData_.set_totalLength(totalLength_);
    tilingData_.set_n(nSize_);
    tilingData_.set_numIters(numIters_);

    OP_LOGI(context_, "total_length=%ld, n=%ld, numIters=%ld", totalLength_, nSize_, numIters_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::CheckOutputShape()
{
    OP_CHECK_IF(gradInputShape_ == nullptr,
        OP_LOGE(context_, "grad_intput shape is null"),
        return ge::GRAPH_FAILED);
    if (isTShape_) {
        if (gradInputShape_->GetDimNum() != GRAD_INPUT_OUTPUT_T_DIMNUM) {
            OP_LOGE(context_, "grad_intput shape dim num is invalid");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (gradInputShape_->GetDimNum() != GRAD_INPUT_OUTPUT_BS_DIMNUM) {
            OP_LOGE(context_, "grad_intput shape dim num is invalid");
            return ge::GRAPH_FAILED;
        }
    }

    if (isTShape_) {
        if (gradInputShape_->GetDim(DIM_0) == tSize_ &&
            gradInputShape_->GetDim(DIM_1) == nSize_ &&
            gradInputShape_->GetDim(DIM_2) == nSize_) {
            return ge::GRAPH_SUCCESS;
        }

        OP_LOGE(context_, "The grad_input shape is invalid, it should be {%ld, %ld, %ld}.", tSize_,
            nSize_, nSize_);
    } else {
        if (gradInputShape_->GetDim(DIM_0) == bSize_ &&
            gradInputShape_->GetDim(DIM_1) == sSize_ &&
            gradInputShape_->GetDim(DIM_2) == nSize_ &&
            gradInputShape_->GetDim(DIM_3) == nSize_) {
            return ge::GRAPH_SUCCESS;
        }

        OP_LOGE(context_, "The grad_input shape is invalid, it should be {%ld, %ld, %ld, %ld}.", bSize_,
            sSize_, nSize_, nSize_);
    }

    return ge::GRAPH_FAILED;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::SplitCores()
{
    // UB 分配计算：(7nnt + 3nt) * 4B < 可用UB大小
    // gradInQueue(1): tnn, normInQueue(2): 2tnn, sumInQueue(2): 2tn
    // gradOutQueue(1): tnn, tmpBuffer(1): 3tnn + tn
    // 总共: 7tnn + 3t*n

    int64_t maxTokensPerLoop = AVALABLE_UB_SIZE / (DTYPE_SIZE * (7 * nSize_ * nSize_ + 3 * nSize_));
    maxTokensPerLoop = maxTokensPerLoop & ~0x7;

    if (maxTokensPerLoop < TRANSPOSE_BUFFER_SIZE) {
        maxTokensPerLoop = (AVALABLE_UB_SIZE - DTYPE_SIZE * TRANSPOSE_BUFFER_SIZE * (2 * nSize_ * nSize_ + nSize_)) /
            (DTYPE_SIZE * (5 * nSize_ * nSize_ + 2 * nSize_));
        maxTokensPerLoop = maxTokensPerLoop & ~0x7;
    }

    OP_CHECK_IF(maxTokensPerLoop < 8,
        OP_LOGE(context_, "UB size is not enough for n=%ld", nSize_),
        return ge::GRAPH_FAILED);

    // 核间切分：在 T 维度上进行核间切分
    int64_t aivNum = static_cast<int64_t>(aicoreParams_.blockDim);
    // 初始按所有核平均分配
    int64_t perCoreElements = CeilDiv(totalLength_, aivNum);
    // 限核策略：当 perCoreElements 较小时（< 32），通过提高 perCoreElements 下限来减少使用的核数
    if (perCoreElements < MIN_PER_CORE_ELEMENTS && totalLength_ >= MIN_PER_CORE_ELEMENTS) {
        perCoreElements = MIN_PER_CORE_ELEMENTS;
    }

    // 核内切分：普通核
    int64_t perCoreLoops = CeilDiv(perCoreElements, maxTokensPerLoop);
    if (perCoreLoops == 1) {
        perCoreElements = (perCoreElements + 0x7) & ~0x7;
    }
    int64_t perCorePerLoopElements = perCoreElements < maxTokensPerLoop ? perCoreElements : maxTokensPerLoop;
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    int64_t needCoreNum = CeilDiv(totalLength_, perCoreElements);
    int64_t lastCoreElements = totalLength_ - perCoreElements * (needCoreNum - 1);

    // 核内切分：最后一个核
    int64_t lastCoreLoops = CeilDiv(lastCoreElements, maxTokensPerLoop);
    int64_t lastCorePerLoopElements = lastCoreElements < maxTokensPerLoop ? lastCoreElements : maxTokensPerLoop;
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    // 设置 tilingData
    tilingData_.set_needCoreNum(needCoreNum);
    tilingData_.set_perCoreElements(perCoreElements);
    tilingData_.set_lastCoreElements(lastCoreElements);
    tilingData_.set_perCoreLoops(perCoreLoops);
    tilingData_.set_perCorePerLoopElements(perCorePerLoopElements);
    tilingData_.set_perCoreLastLoopElements(perCoreLastLoopElements);
    tilingData_.set_lastCoreLoops(lastCoreLoops);
    tilingData_.set_lastCorePerLoopElements(lastCorePerLoopElements);
    tilingData_.set_lastCoreLastLoopElements(lastCoreLastLoopElements);

    OP_LOGI(context_, "Core splitting: needCoreNum=%ld, perCoreElements=%ld, lastCoreElements=%ld, "
        "maxTokensPerLoop=%ld, perCoreLoops=%ld, perCorePerLoopElements=%ld, perCoreLastLoopElements=%ld, "
        "lastCoreLoops=%ld, lastCorePerLoopElements=%ld, lastCoreLastLoopElements=%ld",
        needCoreNum, perCoreElements, lastCoreElements,
        maxTokensPerLoop, perCoreLoops, perCorePerLoopElements, perCoreLastLoopElements,
        lastCoreLoops, lastCorePerLoopElements, lastCoreLastLoopElements);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::DoOpTiling()
{
    auto ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
    return ret;
    }

    ret = CheckOutputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = SplitCores();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::GetWorkspaceSize()
{
    // 计算 workspace 大小
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraSinkhornGradTilingBase::PostTiling()
{
    context_->SetBlockDim(tilingData_.get_needCoreNum());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t AiInfraSinkhornGradTilingBase::GetTilingKey() const
{
    // 当前只有一个通用模板
    return TILING_KEY_GENERALIZED;
}

void AiInfraSinkhornGradTilingBase::Reset()
{
    opName_ = nullptr;
    return;
}

REGISTER_OPS_TILING_TEMPLATE(AiInfraSinkhornGrad, AiInfraSinkhornGradTilingBase, 2000);
} // namespace optiling
