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
 * \file attention_update_tiling.cpp
 * \brief
 */

#include "attention_update_tiling.h"

namespace optiling {
using namespace Ops::Base;

constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t LSE_DIM_NUM = 1;
constexpr uint64_t GO_DIM_NUM = 2;
constexpr uint64_t D_MIN = 8;
constexpr uint64_t D_MAX = 512;
constexpr uint64_t D_DIVIDE_8 = 8;
constexpr uint64_t ATTR_SP_MAX = 16;

constexpr uint64_t INPUT_LSE_INDEX = 0;
constexpr uint64_t INPUT_GO_INDEX = 1;
constexpr uint64_t ATTR_UPDATE_TYPE_INDEX = 0;
constexpr uint64_t ATTR_SP_INDEX = 1;
constexpr uint64_t OUTPUT_INDEX = 0;
constexpr uint64_t OUTPUT_LSE_M_INDEX = 1;

constexpr uint64_t ALL_TO_SP_MULTIPLIER = 2UL;
constexpr uint64_t NUM_2 = 2UL;
constexpr uint64_t TILING_KEY_EMPTY = 10000UL;
constexpr uint64_t TILING_KEY_INIT_VALUE = 20000UL;

constexpr uint64_t DOUBLE_BUFFER_NUM = 2UL;
constexpr uint64_t SYS_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);

bool AttentionUpdateTiling::IsCapable()
{
    return true;
}

ge::graphStatus AttentionUpdateTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionUpdateTiling::GetPlatformInfo()
{
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const DecodeUpdateCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        totalCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
        if (totalCoreNum_ == 0UL) {
            OP_LOGE(context_->GetNodeName(), "coreNum is 0");
            return ge::GRAPH_FAILED;
        }
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (ubSize == static_cast<uint64_t>(0)) {
            OP_LOGE(context_->GetNodeName(), "ubSize is 0");
            return ge::GRAPH_FAILED;
        }
        ubSize_ = static_cast<uint64_t>(ubSize);
    }

    return ge::GRAPH_SUCCESS;
}

// 检查输入数据类型
ge::graphStatus AttentionUpdateTiling::CheckInputDtype()
{
    if (goType_ != ge::DataType::DT_FLOAT && goType_ != ge::DataType::DT_FLOAT16 && goType_ != ge::DataType::DT_BF16) {
        std::string dtypeStr = ToString(goType_);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "go", dtypeStr.c_str(),
            "FLOAT, FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }

    if (lseType_ != ge::DataType::DT_FLOAT) {
        std::string dtypeStr = ToString(lseType_);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "lse", dtypeStr.c_str(), "FLOAT");
        return ge::GRAPH_FAILED;
    }

    for (uint64_t i = 0; i < sp_ * ALL_TO_SP_MULTIPLIER; i++) {
        if (i == 0UL || i == sp_) {
            continue;
        }
        auto currentDtype = context_->GetInputDesc(i)->GetDataType();
        if (i >= sp_) {
            if (goType_ != currentDtype) {
                std::string paramMsg = "go[" + std::to_string(i-sp_) + "] tensor";
                std::string currentDtypeStr = ToString(currentDtype);
                std::string reasonMsg = "All tensors in input go must have the same dtype, "
                    "but the dtype of go[" + std::to_string(i-sp_) + "] tensor is different from other dtypes " +
                    ToString(goType_);
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                    currentDtypeStr.c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        } else {
            if (lseType_ != currentDtype) {
                std::string paramMsg = "lse[" + std::to_string(i) + "] tensor";
                std::string currentDtypeStr = ToString(currentDtype);
                std::string reasonMsg = "All tensors in the lse must have the same dtype, "
                    "but the dtype of lse[" + std::to_string(i) + "] tensor is different from other dtypes " +
                    ToString(lseType_);
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                    currentDtypeStr.c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 检查输入数据维度，以及go和lse的第一维是否一致，以及go的第二维是否是8的倍数，且在[8,512]范围
ge::graphStatus AttentionUpdateTiling::CheckInputDim()
{
    uint64_t dimNum = 0;
    if (!(d_ >= D_MIN && d_ <= D_MAX && d_ % D_DIVIDE_8 == 0)) {
        std::string shapeStr = ToString(goShape_);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "go",
            shapeStr.c_str(),
            "The H dim of input go should be in the range of [8, 512] and divisible by 8, "
            "where H refers to the 1st dim");
        return ge::GRAPH_FAILED;
    }
    for (uint64_t i = 0; i < ALL_TO_SP_MULTIPLIER * sp_; i++) {
        dimNum = context_->GetInputShape(i)->GetOriginShape().GetDimNum();
        auto currentShape = context_->GetInputShape(i)->GetOriginShape();
        uint64_t currentBshSize = currentShape.GetDim(DIM_0);
        std::string paramMsg;
        if (i >= sp_) {
            uint64_t currentD = context_->GetInputShape(i)->GetOriginShape().GetDim(DIM_1);
            paramMsg = "go[" + std::to_string(i-sp_) + "] tensor";
            if (!(dimNum == GO_DIM_NUM)) {
                std::string dimNumStr = std::to_string(dimNum);
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), paramMsg.c_str(),
                    dimNumStr.c_str(), "2D");
                return ge::GRAPH_FAILED;
            }
            if (!(currentD == d_)) {
                std::string currentShapeStr = ToString(currentShape);
                std::string reasonMsg = "The H dims of all go tensors must be the same, "
                    "but the H dim of " + paramMsg + " is different from other H dims " +
                    ToString(goShape_) +", where H refers to the 1st dim";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                    currentShapeStr.c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        } else {
            paramMsg = "lse[" + std::to_string(i) + "] tensor";
            if (!(dimNum == LSE_DIM_NUM)) {
                std::string dimNumStr = std::to_string(dimNum);
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), paramMsg.c_str(),
                    dimNumStr.c_str(), "1D");
                return ge::GRAPH_FAILED;
            }
        }
        if (!(bshSize_ == currentBshSize)) {
            std::string currentShapeStr = ToString(currentShape);
            std::string reasonMsg = "The batch axis of all go and lse tensors should be the same, "
                "but the batch axis of " + paramMsg + " is different from other batch axes " +
                ToString(goShape_) + ", where batch refers to the 0th dim";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                currentShapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 检查输入参数
ge::graphStatus AttentionUpdateTiling::CheckInputParams()
{
    if (CheckInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDim() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 检查参数update_type
    if (!(updateType_ == 0 || updateType_ == 1)) {
        std::string updateTypeStr = std::to_string(updateType_);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "update_type",
            updateTypeStr.c_str(), "0 or 1");
        return ge::GRAPH_FAILED;
    }

    // 检查参数sp
    if (!(sp_ >= 1 && sp_ <= ATTR_SP_MAX)) {
        std::string spStr = std::to_string(sp_);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "sp",
            spStr.c_str(), "in the range of [1, 16]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionUpdateTiling::CheckOutputParams()
{
    auto outputDesc = context_->GetOutputDesc(OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto outputType = outputDesc->GetDataType();
    if (goType_ != outputType) {
        std::string dtypeMsg = ToString(goType_) + " and " + ToString(outputType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "go and output",
            dtypeMsg.c_str(), "The dtypes of parameter go and parameter output should be the same");
        return ge::GRAPH_FAILED;
    }

    if (updateType_ == 1) {
        auto outputLseMDesc = context_->GetOutputDesc(OUTPUT_LSE_M_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputLseMDesc);
        auto outputLseMType = outputLseMDesc->GetDataType();
        if (outputLseMType != ge::DataType::DT_FLOAT) {
            std::string dtypeStr = ToString(outputLseMType);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "lse_m",
                dtypeStr.c_str(),
                "The dtype of output lse_m should be FLOAT when the attr update_type is 1");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionUpdateTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("AttentionUpdate", "context is null"), return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    lseShape_ = context_->GetInputShape(INPUT_LSE_INDEX)->GetOriginShape();
    int64_t lseSizeNum = lseShape_.GetShapeSize();

    std::string reasonMsg = "此时的分支为： lseSizeNum = " + std::to_string(lseSizeNum) + " 。";

    // 从1到10 的分支
    if (lseSizeNum == 1){
        // EZ0008
        OP_LOGE_FOR_INVALID_SHAPE(
            context_->GetNodeName(), "lse", std::to_string(lseSizeNum).c_str(),
            reasonMsg.c_str());
    } else if (lseSizeNum == 2){
        // EZ0009
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "lse", std::to_string(lseSizeNum).c_str(),
            reasonMsg.c_str());
    } else if (lseSizeNum == 3){
        // EZ0010
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "lse, output", std::to_string(lseSizeNum).c_str(),
            reasonMsg.c_str());
    } else if (lseSizeNum == 4){
        // EZ0011
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "lse", std::to_string(lseSizeNum).c_str(),
            reasonMsg.c_str());
    } else if (lseSizeNum == 5){
        // EZ0012
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "lse", std::to_string(lseSizeNum).c_str(),
            reasonMsg.c_str());
    } else if (lseSizeNum == 6){
        // EZ0013
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "lse, output", std::to_string(lseSizeNum).c_str(),
            reasonMsg.c_str());
    } else if (lseSizeNum == 7){
        // EZ0014
        OP_LOGE_FOR_INVALID_SHAPESIZE(
            context_->GetNodeName(), "lse", std::to_string(lseSizeNum).c_str(),
            reasonMsg.c_str());
    } else if (lseSizeNum == 8){
        // EZ0015
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context_->GetNodeName(), "lse", std::to_string(lseSizeNum).c_str(),
            reasonMsg.c_str());
    } else if (lseSizeNum == 9){
        // EZ0016
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(), "lse, output", std::to_string(lseSizeNum).c_str(),
            reasonMsg.c_str());
    } else if (lseSizeNum == 10){
        // EZ0017
        OP_LOGE_FOR_INVALID_FORMAT(
            context_->GetNodeName(), "lse", std::to_string(lseSizeNum).c_str(),
            reasonMsg.c_str());
    }

    return ge::GRAPH_FAILED;
}

ge::graphStatus AttentionUpdateTiling::DoOpTiling()
{
    perCoreCount_ = Ops::Base::CeilDiv(bshSize_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(bshSize_, perCoreCount_);
    lastCoreCount_ = bshSize_ - (usedCoreNum_ - 1) * perCoreCount_;

    // 预留部分空间防止lse对齐搬入UB后占用空间变大
    ubSize_ = ubSize_ - sp_ * ubBlockSize_;
    uint64_t goBlockNum = ubBlockSize_ / goDtypeSize_;
    uint64_t dAlign = Ops::Base::CeilAlign(d_, goBlockNum);

    // 输入数据搬入UB需要开DB
    int64_t inputFactor = sp_ * sizeof(float) * DOUBLE_BUFFER_NUM + sp_ * dAlign * goDtypeSize_ * DOUBLE_BUFFER_NUM;

    // 中间计算需要提升精度
    int64_t calcFactor = sp_ * sizeof(float);

    // 计算结果搬出需要开DB
    int64_t outputFactor = DOUBLE_BUFFER_NUM * sizeof(float) + dAlign * goDtypeSize_ * DOUBLE_BUFFER_NUM;

    bshInLoop_ = ubSize_ / (inputFactor + calcFactor + outputFactor);
    uint64_t bshPerLoop =
        (ubSize_ - sp_ * bshInLoop_ * dAlign * goDtypeSize_ * DOUBLE_BUFFER_NUM -
         bshInLoop_ * dAlign * goDtypeSize_ * DOUBLE_BUFFER_NUM) /
        (sp_ * sizeof(float) * DOUBLE_BUFFER_NUM + sp_ * sizeof(float) + sizeof(float) * DOUBLE_BUFFER_NUM);
    uint64_t lseUbBlockCount = ubBlockSize_ / sizeof(float);
    if (bshPerLoop > lseUbBlockCount) {
        bshPerLoop = Ops::Base::FloorAlign(bshPerLoop, lseUbBlockCount);
    }

    bshInLoop_ = std::min(bshInLoop_, bshPerLoop);
    perCorePerLoopCount_ = bshPerLoop;

    perCoreLoops_ = Ops::Base::CeilDiv(perCoreCount_, bshPerLoop);
    lastCoreLoops_ = Ops::Base::CeilDiv(lastCoreCount_, bshPerLoop);
    perCoreLastLoopCount_ = perCoreCount_ - (perCoreLoops_ - 1) * bshPerLoop;
    lastCoreLastLoopCount_ = lastCoreCount_ - (lastCoreLoops_ - 1) * bshPerLoop;

    usedCoreNum_ = std::max(usedCoreNum_, static_cast<uint64_t>(1));
    return ge::GRAPH_SUCCESS;
}

uint64_t AttentionUpdateTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILING_KEY_INIT_VALUE;
    if (bshSize_ == 0) {
        tilingKey = TILING_KEY_EMPTY;
    } else {
        tilingKey = TILING_KEY_INIT_VALUE + updateType_;
    }
    return tilingKey;
}

ge::graphStatus AttentionUpdateTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionUpdateTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    context_->SetBlockDim(usedCoreNum_);

    // 输入参数
    tilingData_.set_sp(sp_);
    tilingData_.set_d(d_);

    // 分核数
    tilingData_.set_usedCoreNum(usedCoreNum_);

    // 单核切分大小
    tilingData_.set_perCoreCount(perCoreCount_);
    tilingData_.set_lastCoreCount(lastCoreCount_);

    // 设置内循环循环次数和切分大小
    tilingData_.set_perCoreLoops(perCoreLoops_);
    tilingData_.set_lastCoreLoops(lastCoreLoops_);

    tilingData_.set_perCorePerLoopCount(perCorePerLoopCount_);
    tilingData_.set_perCoreLastLoopCount(perCoreLastLoopCount_);
    tilingData_.set_lastCoreLastLoopCount(lastCoreLastLoopCount_);
    tilingData_.set_bshInLoop(bshInLoop_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void AttentionUpdateTiling::DumpTilingInfo()
{
    std::ostringstream info;
    // 输入参数
    info << "sp: " << sp_ << std::endl;
    info << "d: " << d_ << std::endl;
    info << "usedCoreNum: " << usedCoreNum_ << std::endl;

    info << "perCoreCount: " << perCoreCount_ << std::endl;
    info << "lastCoreCount: " << lastCoreCount_ << std::endl;
    info << "perCoreLoops: " << perCoreLoops_ << std::endl;
    info << "lastCoreLoops: " << lastCoreLoops_ << std::endl;

    // 内循环信息
    info << "perCorePerLoopCount: " << perCorePerLoopCount_ << std::endl;
    info << "perCoreLastLoopCount: " << perCoreLastLoopCount_ << std::endl;
    info << "lastCoreLastLoopCount: " << lastCoreLastLoopCount_ << std::endl;
    info << "bshInLoop: " << bshInLoop_ << std::endl;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(AttentionUpdate, AttentionUpdateTiling, 1);
} // namespace optiling
