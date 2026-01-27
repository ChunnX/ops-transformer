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
 * \file fused_infer_attention_score_tiling_check_consistency.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_check.h"

using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {
void FiaTilingCheck::SetFiaShapeCompare()
{
    queryShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.query.shape->GetStorageShape(),
        qLayout_, QUERY_NAME, opName_);
    keyShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(kCache_[0]->GetStorageShape(),
        kvLayout_, KEY_NAME, opName_);
    valueShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(vCache_[0]->GetStorageShape(),
        kvLayout_, VALUE_NAME, opName_);
    attenOutShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.attenOut.shape->GetStorageShape(),
        outLayout_, ATTEN_OUT_NAME, opName_);
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        queryRopeShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.queryRope.tensor->GetStorageShape(),
            qLayout_, QUERY_ROPE_NAME, opName_);
        keyRopeShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.keyRope.tensor->GetStorageShape(),
            kvLayout_, KEY_ROPE_NAME, opName_);
    }
}

ge::graphStatus FiaTilingCheck::CheckQAndQRopeDType() const
{
    if (opParamInfo_.query.desc->GetDataType() != inputQType_) {
        OP_LOGE(opName_, "%s's dtype is %s, it should be %s.",
            QUERY_NAME.c_str(),
            FusedDataTypeToSerialString(opParamInfo_.query.desc->GetDataType()).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str());
            return ge::GRAPH_FAILED;
    }
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        if (opParamInfo_.queryRope.desc->GetDataType() != inputQRopeType_) {
            OP_LOGE(opName_, "%s's dtype is %s, it should be %s.",
                QUERY_NAME.c_str(),
                FusedDataTypeToSerialString(opParamInfo_.queryRope.desc->GetDataType()).c_str(),
                FusedDataTypeToSerialString(inputQRopeType_).c_str());
                return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckQShape() const
{
    FiaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(bSize_);
    shapeParams.N = static_cast<int64_t>(n1Size_);
    shapeParams.S = static_cast<int64_t>(s1Size_);
    shapeParams.D = static_cast<int64_t>(qkHeadDim_);
    shapeParams.T = static_cast<int64_t>(qTSize_);
    return queryShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus FiaTilingCheck::CheckQRopeShape() const
{
    // rope分离模式时queryRope Tensor才存在
    if (ropeMode_ != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }

    FiaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(bSize_);
    shapeParams.N = static_cast<int64_t>(n1Size_);
    shapeParams.S = static_cast<int64_t>(s1Size_);
    shapeParams.D = static_cast<int64_t>(ropeHeadDim_);
    shapeParams.T = static_cast<int64_t>(qTSize_);
    return queryRopeShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus FiaTilingCheck::CheckQAndQRopeShape() const
{
    if (ge::GRAPH_SUCCESS != CheckQShape() ||
        ge::GRAPH_SUCCESS != CheckQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckQAndQRope() const
{
    if (ge::GRAPH_SUCCESS != CheckQAndQRopeDType() ||
        ge::GRAPH_SUCCESS != CheckQAndQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckKVDType() const
{
    if (opParamInfo_.key.desc->GetDataType() != inputKvType_) {
        OP_LOGE(opName_, "%s's dtype is %s, it should be %s.",
            KEY_NAME.c_str(),
            FusedDataTypeToSerialString(opParamInfo_.key.desc->GetDataType()).c_str(),
            FusedDataTypeToSerialString(inputKvType_).c_str());
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.value.desc->GetDataType() != inputKvType_) {
        OP_LOGE(opName_, "%s's dtype is %s, it should be %s.",
            VALUE_NAME.c_str(),
            FusedDataTypeToSerialString(opParamInfo_.value.desc->GetDataType()).c_str(),
            FusedDataTypeToSerialString(inputKvType_).c_str());
            return ge::GRAPH_FAILED;
    }
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        if (opParamInfo_.keyRope.desc->GetDataType() != inputKRopeType_) {
            OP_LOGE(opName_, "%s's dtype is %s, it should be %s.",
                KEY_ROPE_NAME.c_str(),
                FusedDataTypeToSerialString(opParamInfo_.keyRope.desc->GetDataType()).c_str(),
                FusedDataTypeToSerialString(inputKRopeType_).c_str());
                return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckKVShapeForBatchContinuous() const
{
    FiaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(bSize_);
    shapeParams.N = static_cast<int64_t>(n2Size_);
    shapeParams.S = s2Size_;
    shapeParams.D = static_cast<int64_t>(qkHeadDim_);
    shapeParams.T = static_cast<int64_t>(kTSize_);
    if (keyShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = static_cast<int64_t>(vHeadDim_);
    if (valueShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        shapeParams.D = static_cast<int64_t>(ropeHeadDim_);
        if (keyRopeShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckKVShape() const
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckKVShapeForBatchContinuous();
    }

    OP_LOGE(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus FiaTilingCheck::CheckKV() const
{
    if (ge::GRAPH_SUCCESS != CheckKVDType() ||
        ge::GRAPH_SUCCESS != CheckKVShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckAttenOut() const
{
    FiaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(bSize_);
    shapeParams.N = static_cast<int64_t>(n1Size_);
    shapeParams.S = static_cast<int64_t>(s1Size_);
    shapeParams.D = static_cast<int64_t>(vHeadDim_);
    shapeParams.T = static_cast<int64_t>(qTSize_);
    if (attenOutShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckActualSeqLensQ() const
{
    if ((opParamInfo_.actualSeqLengthsQ.tensor == nullptr) ||
        (opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr)) {
        return ge::GRAPH_SUCCESS;
    }

    if (qLayout_ == FiaLayout::TND) {
        if (actualSeqLengthsQSize_ != bSize_ && actualSeqLengthsQSize_ != 1U) {
            OP_LOGE(opName_, "%s shape size is %u, it should be equal to batch size(%u) or equal to 1.",
                ACTUAL_SEQ_Q_LEN_NAME.c_str(), actualSeqLengthsQSize_, bSize_);
            return ge::GRAPH_FAILED;
        }
    } else {
        if (actualSeqLengthsQSize_ < bSize_ && actualSeqLengthsQSize_ != 1U) {
            OP_LOGE(opName_, "%s shape size is %u, it should be bigger or equal to batch size(%u) or equal to 1.",
                ACTUAL_SEQ_Q_LEN_NAME.c_str(), actualSeqLengthsQSize_, bSize_);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckActualSeqLensKv() const
{
    if ((opParamInfo_.actualSeqLengths.tensor == nullptr) ||
        (opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr)) {
        return ge::GRAPH_SUCCESS;
    }

    if (kvLayout_ == FiaLayout::TND) {
        if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr &&
            opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() != nullptr &&
            actualSeqLengthsKvSize_ != actualSeqLengthsQSize_) {
            OP_LOGE(opName_, "%s shape size is %u, it should be equal to %s shape size(%u).",
                ACTUAL_SEQ_KV_LEN_NAME.c_str(), actualSeqLengthsKvSize_,
                ACTUAL_SEQ_Q_LEN_NAME.c_str(), actualSeqLengthsQSize_);
            return ge::GRAPH_FAILED;
        }
        if (actualSeqLengthsKvSize_ != bSize_ && actualSeqLengthsKvSize_ != 1U) {
            OP_LOGE(opName_, "%s shape size is %u, it should be equal to batch size(%u) or equal to 1.",
                ACTUAL_SEQ_KV_LEN_NAME.c_str(), actualSeqLengthsKvSize_, bSize_);
            return ge::GRAPH_FAILED;
        }
    } else {
        if (actualSeqLengthsKvSize_ < bSize_ && actualSeqLengthsKvSize_ != 1U) {
            OP_LOGE(opName_, "%s shape size is %u, it should be bigger or equal to batch size(%u) or equal to 1.",
                ACTUAL_SEQ_KV_LEN_NAME.c_str(), actualSeqLengthsKvSize_, bSize_);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckMultiParaConsistency()
{
    SetFiaShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensKv() ||
        ge::GRAPH_SUCCESS != CheckQAndQRope() ||
        ge::GRAPH_SUCCESS != CheckKV() ||
        ge::GRAPH_SUCCESS != CheckAttenOut()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
