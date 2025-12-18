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
 * \file fused_infer_attention_score_tiling_check_feature.cpp
 * \brief
 */

#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include <numeric>
#include <algorithm>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_check.h"

using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoQuantShape() const
{
    OP_CHECK_IF(qkHeadDim_ != 512U && qkHeadDim_ != 128U,
        OP_LOGE(opName_, "In %s situation, rope exsists, the K/V's head dim only support 128 and 512, but got %u",
            QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_),
        return ge::GRAPH_FAILED);

    if (vHeadDim_ == 512U) {
        OP_CHECK_IF(opParamInfo_.keyRope.tensor->GetStorageShape().GetShapeSize() == 0,
            OP_LOGE(opName_, "In %s situation, %s tensor should not be empty",
                RopeModeToSerialString(ropeMode_).c_str(), KEY_ROPE_NAME.c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(opParamInfo_.key.shape->GetStorageShape().GetShapeSize() == 0,
            OP_LOGE(opName_, "In %s situation, %s tensor should not be empty",
                RopeModeToSerialString(ropeMode_).c_str(), KEY_NAME.c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(n2Size_ != 1,
            OP_LOGE(opName_, "In %s situation, rope exsists and Q/K head dim = %u, %s should be 1, but got %u",
                QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_,
                KV_HEADS_NUM_NAME.c_str(), n2Size_), return ge::GRAPH_FAILED);

        std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
        OP_CHECK_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
            OP_LOGE(opName_, "In %s situation, rope exsists and Q/K head dim = %u, group num should be in 1, 2, 4, 8, 16, 32, 64, 128, but got %u",
                QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_, gSize_), return ge::GRAPH_FAILED);

        OP_CHECK_IF(qkHeadDim_ != vHeadDim_,
            OP_LOGE(opName_, "In %s situation, rope exsists, the Q/K's head dim(%u) should be equal to the value's head dim(%u)",
                QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_, vHeadDim_), return ge::GRAPH_FAILED);

        OP_CHECK_IF(ropeHeadDim_ != 64,
            OP_LOGE(opName_, "In %s situation, rope exsists and Q/K head dim = %u, the rope's head dim should be 64, but got %u",
                QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_, ropeHeadDim_), return ge::GRAPH_FAILED);
    } else if (vHeadDim_ == 128U) {
        return CheckFeatureGqaNoQuantShape();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoQuantLayout() const
{
    if (vHeadDim_ == 512U) {
        std::string layout = opParamInfo_.layOut;
        const std::vector<std::string> layoutSupportList = {
            "BSH", "BSND", "BNSD", "TND", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD"
        };
        OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
            OP_LOGE(opName_, "In %s situation, rope exsists and Q/K head dim = %u, layout only supports BSH, BSND, BNSD, TND, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_, layout.c_str()),
            return ge::GRAPH_FAILED);
    } else if (vHeadDim_ == 128U) {
        return CheckFeatureGqaNoQuantLayout();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureNoQuantDtype() const
{
    OP_CHECK_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OP_LOGE(opName_, "In %s situation, query dtype only support %s and %s, but got %s",
            QuantModeToSerialString(quantMode_).c_str(),
            FusedDataTypeToSerialString(ge::DT_BF16).c_str(), FusedDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(inputQType_ != inputKvType_,
        OP_LOGE(opName_, "In %s situation, K and V dtype(%s) must equal to query dtype(%s)",
            QuantModeToSerialString(quantMode_).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str(),
            FusedDataTypeToSerialString(inputKvType_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoQuantDtype() const
{
    if (CheckFeatureNoQuantDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((opParamInfo_.queryRope.desc->GetDataType() != opParamInfo_.query.desc->GetDataType()),
        OP_LOGE(opName_, "%s(%s) and %s(%s) must have same dType",
            QUERY_ROPE_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.queryRope.desc->GetDataType()).c_str(),
            QUERY_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.query.desc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((opParamInfo_.keyRope.desc->GetDataType() != opParamInfo_.key.desc->GetDataType()),
        OP_LOGE(opName_, "%s(%s) and %s(%s) must have same dType",
            KEY_ROPE_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.keyRope.desc->GetDataType()).c_str(),
            KEY_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.key.desc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureNoquantBlockSize() const
{
    constexpr int32_t BLOCK_SIZE_ALIGN_SIZE = 16;
    constexpr int32_t BLOCK_SIZE_MAX_SIZE = 1024;
    if (blockSize_ % BLOCK_SIZE_ALIGN_SIZE != 0) {
        OP_LOGE(opName_, "In %s situation, %s should aligned to 16, but got %d.",
            QuantModeToSerialString(quantMode_).c_str(), BLOCK_SIZE_NAME.c_str(), blockSize_);
            return ge::GRAPH_FAILED;
    }

    if (blockSize_ > BLOCK_SIZE_MAX_SIZE) {
        OP_LOGE(opName_, "In %s situation, %s should less equal than 1024, but got %d.",
            QuantModeToSerialString(quantMode_).c_str(), BLOCK_SIZE_NAME.c_str(), blockSize_);
            return ge::GRAPH_FAILED;
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && blockSize_ == 0) {
        OP_LOGE(opName_, "In %s and storage mode is page attention, %s should not be 0",
            QuantModeToSerialString(quantMode_).c_str(), BLOCK_SIZE_NAME.c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquantMask() const
{
    if(vHeadDim_ == 512U) {
        int32_t sparseMode = *opParamInfo_.sparseMode;
        if (sparseMode != SPARSE_MODE_NO_MASK && sparseMode != SPARSE_MODE_RIGHT_DOWN && sparseMode != SPARSE_MODE_BAND) {
            OP_LOGE(opName_,
                "In %s situation, rope exsists and query/key head dim = %u, %s only support 0/3/4, but got %d.",
                QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_, SPARSE_MODE_NAME.c_str(), sparseMode);
            return ge::GRAPH_FAILED;
        }

        if (!attenMaskFlag_) {
            if (sparseMode == SPARSE_MODE_NO_MASK) {
                return ge::GRAPH_SUCCESS;
            }
            if (sparseMode == SPARSE_MODE_RIGHT_DOWN || sparseMode == SPARSE_MODE_BAND) {
                OP_LOGE(opName_,
                    "In %s situation, rope exsists and query/key head dim = %u, when %s = 3/4, %s should not be null.",
                    QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_,
                    SPARSE_MODE_NAME.c_str(), ATTEN_MASK_NAME.c_str());
                return ge::GRAPH_FAILED;
            }
        }

        size_t maskDimNum = opParamInfo_.attenMask.tensor->GetStorageShape().GetDimNum();
        size_t maskDim0 = opParamInfo_.attenMask.tensor->GetStorageShape().GetDim(0);
        if (sparseMode == static_cast<int32_t>(SPARSE_MODE_NO_MASK) &&
            maskDimNum == DIM_NUM_TWO && s1Size_ == 1U && maskDim0 == static_cast<size_t>(bSize_)) {
            OP_CHECK_IF(qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD,
                    OP_LOGE(opName_, "In %s situation, rope exsists and Q/K head dim = %u, when %s layout is TND/NTD, %s layout BS2 is not supported.",
                        QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_, QUERY_NAME.c_str(), ATTEN_MASK_NAME.c_str()),
                return ge::GRAPH_FAILED);
        }
    } else if (vHeadDim_ == 128U) {
        return CheckFeatureGqaNoquantMask();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaSink() const
{
    // sink功能不支持MLA vD=512的场景
    OP_CHECK_IF((fiaInfo_.learnableSinkFlag == true) && (vHeadDim_ == HEAD_DIM_512),
        OP_LOGE(opName_, "In %s situation, rope exsists and value head dim is %u, %s is not supported.",
        QuantModeToSerialString(quantMode_).c_str(), vHeadDim_, LEARNABLE_SINK_NAME.c_str()),
        return ge::GRAPH_FAILED);

    return CheckFeatureGqaNoquantSink();
}

ge::graphStatus FiaTilingCheck::CheckFeatureNoquantUnsupported() const
{
    OP_CHECK_IF(fiaInfo_.outputType == ge::DT_INT8,
        OP_LOGE(opName_, "In %s situation, postquant is not supported.", QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);
 
    OP_CHECK_IF(fiaInfo_.pseShiftFlag,
        OP_LOGE(opName_, "In %s situation, pseshift is not supported.", QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);
 
    OP_CHECK_IF(fiaInfo_.qPaddingSizeFlag || fiaInfo_.kvPaddingSizeFlag,
        OP_LOGE(opName_, "In %s situation, left padding is not supported.", QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo_.sysPrefixFlag,
        OP_LOGE(opName_, "In %s situation, sys prifix is not supported.", QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED); 
 
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquantUnsupported() const
{
    if (vHeadDim_ == 512U) {
        if (CheckFeatureNoquantUnsupported() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
            OP_LOGE(opName_, "In %s situation, rope exsists and Q/K head dim = %u, the K/V's storage mode not support tensor list",
                QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_);
            return ge::GRAPH_FAILED;
        }
    } else if (vHeadDim_ == 128U) {
        return CheckFeatureGqaNoquantUnsupported();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquant()
{
    OP_CHECK_IF(socVersion_ == platform_ascendc::SocVersion::ASCEND310P,
        OP_LOGE(opName_, "In %s %s situation, Ascend310P is not supported",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantUnsupported() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoquantBlockSize() ||
        ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantMask() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaSink()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaAntiquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaFullquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMla()
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckFeatureMlaNoquant();
    } else if (quantMode_ == FiaQuantMode::ANTI_QUANT) {
        return CheckFeatureMlaAntiquant();
    } else if (quantMode_ == FiaQuantMode::FULL_QUANT) {
        return CheckFeatureMlaFullquant();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoquantUnsupported() const
{
    if (CheckFeatureNoquantUnsupported() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        const std::vector<std::string> layoutSupportList = {
            "BSH", "BSND", "BNSD", "BSND_BNSD",
        };
        std::string layout = opParamInfo_.layOut;
        OP_CHECK_IF((std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) && ropeMode_ != RopeMode::NO_ROPE,
            OP_LOGE(opName_, "In %s situation, tensor list is not supported.",
            QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoquantMask() const
{
    if (fiaInfo_.sparseMode == 0) {
        return ge::GRAPH_SUCCESS;
    }

    if (!attenMaskFlag_) {
        OP_LOGE(opName_,
            "In %s situation, when %s = 1/2/3/4, %s should not be null.",
            QuantModeToSerialString(quantMode_).c_str(),
            SPARSE_MODE_NAME.c_str(), ATTEN_MASK_NAME.c_str());
        return ge::GRAPH_FAILED;
    }

    size_t maskDimNum = opParamInfo_.attenMask.tensor->GetStorageShape().GetDimNum();
    int64_t maskDim0 = opParamInfo_.attenMask.tensor->GetStorageShape().GetDim(0);
    int32_t sparseMode = *opParamInfo_.sparseMode;

    if (sparseMode == SPARSE_MODE_NO_MASK && maskDimNum == DIM_NUM_TWO &&
        s1Size_ == 1U && maskDim0 == static_cast<int64_t>(bSize_)) {
        OP_CHECK_IF(qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD,
                OP_LOGE(opName_, "In %s situation, when %s layout is TND/NTD, %s layout BS2 is not supported.",
                    QuantModeToSerialString(quantMode_).c_str(), QUERY_NAME.c_str(), ATTEN_MASK_NAME.c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoquantSink() const
{
    if (fiaInfo_.learnableSinkFlag == false) {
        return ge::GRAPH_SUCCESS;
    }

    const std::vector<size_t> sinkDimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.learnableSink.tensor, sinkDimNumList, LEARNABLE_SINK_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sinkDim = opParamInfo_.learnableSink.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(sinkDim != fiaInfo_.n1Size,
        OP_LOGE(opName_, "learnable_sink enable, sink shape(%u) must be same equal queryN(%u)!", sinkDim, fiaInfo_.n1Size),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoQuantDtype() const
{
    return CheckFeatureNoQuantDtype();
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoQuantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSH", "BSND", "BNSD", "TND", "NTD", "BSH_BNSD", "BSND_BNSD", "BNSD_BSND", "NTD_TND",
    };
    std::string layout = opParamInfo_.layOut;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
        OP_LOGE(opName_, "In %s situation, layout only supports BSH, BSND, BNSD, TND, NTD, BSH_BNSD, BSND_BNSD, BNSD_BSND and NTD_TND, but got %s",
            QuantModeToSerialString(quantMode_).c_str(), layout.c_str()),
        return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        OP_CHECK_IF(kvLayout_ != FiaLayout::BSH && kvLayout_ != FiaLayout::BSND && kvLayout_ != FiaLayout::BNSD &&
            kvLayout_ != FiaLayout::TND && kvLayout_ != FiaLayout::NTD,
            OP_LOGE(opName_, "In %s situation, K/V's layout only support BSH, BSND, BNSD, TND and NTD in batch continuous scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), LayoutToSerialString(kvLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ != qLayout_,
            OP_LOGE(opName_, "In %s situation, K/V's layout and query's layout should be same in batch continuous scene.",
                QuantModeToSerialString(quantMode_).c_str()),
            return ge::GRAPH_FAILED);
    } else if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        OP_CHECK_IF(kvLayout_ != FiaLayout::BSH && kvLayout_ != FiaLayout::BSND && kvLayout_ != FiaLayout::BNSD,
            OP_LOGE(opName_, "In %s situation, K/V's layout only support BSH, BSND and BNSD in tensor list scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), LayoutToSerialString(kvLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ != qLayout_,
            OP_LOGE(opName_, "In %s situation, K/V's layout and query's layout should be same in tensor list scene.",
            QuantModeToSerialString(quantMode_).c_str()),
            return ge::GRAPH_FAILED);
    } else if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(kvLayout_ == FiaLayout::BnBsH && (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND &&
                        qLayout_ != FiaLayout::BNSD && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
            OP_LOGE(opName_, "In %s situation, the K/V's layout is BnBsH, %s layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ == FiaLayout::BnNBsD && (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND &&
                        qLayout_ != FiaLayout::BNSD && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
            OP_LOGE(opName_, "In %s situation, the K/V's layout is BnNBsD, %s layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ == FiaLayout::NZ && (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND &&
                        qLayout_ != FiaLayout::BNSD && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
            OP_LOGE(opName_, "In %s situation, the K/V's layout is BnNBsD, %s layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoQuantShape() const
{
    constexpr uint32_t MAX_ACTUAL_SEQ_LEN_BYTE = 64U * 1024U;
    constexpr uint32_t MAX_B_SIZE = 256U;

    OP_CHECK_IF(actualSeqLengthsQSize_ > MAX_ACTUAL_SEQ_LEN_BYTE,
    OP_LOGE(opName_, "In %s situation, actual sequence length q should be smaller or equal to 64K, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), actualSeqLengthsQSize_),
    return ge::GRAPH_FAILED);

    OP_CHECK_IF(actualSeqLengthsKvSize_ > MAX_ACTUAL_SEQ_LEN_BYTE,
    OP_LOGE(opName_, "In %s situation, actual sequence length kv should be smaller or equal to 64K, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), actualSeqLengthsKvSize_),
    return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        OP_CHECK_IF(bSize_ > MAX_B_SIZE,
            OP_LOGE(opName_, "In %s situation, batch size(%u) cannot be greater than %u in tensor list scene.",
                QuantModeToSerialString(quantMode_).c_str(), bSize_, MAX_B_SIZE),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoquant()
{
    OP_CHECK_IF(socVersion_ == platform_ascendc::SocVersion::ASCEND310P,
        OP_LOGE(opName_, "In %s %s situation, Ascend310P is not supported",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != CheckFeatureGqaNoquantUnsupported() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoquantBlockSize() ||
        ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckFeatureGqaNoquantMask() ||
        ge::GRAPH_SUCCESS != CheckFeatureGqaNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureGqaNoQuantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureGqaNoQuantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureGqaNoquantSink()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaAntiquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaFullquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqa()
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckFeatureGqaNoquant();
    } else if (quantMode_ == FiaQuantMode::ANTI_QUANT) {
        return CheckFeatureGqaAntiquant();
    } else if (quantMode_ == FiaQuantMode::FULL_QUANT) {
        return CheckFeatureGqaFullquant();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensExistence() const
{
    if ((qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD)) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
            OP_LOGE(opName_, "when %s's layout is %s, %s should not be null.", QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                ACTUAL_SEQ_Q_LEN_NAME.c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
            OP_LOGE(opName_, "when %s's layout is %s, %s should not be null.", QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                ACTUAL_SEQ_KV_LEN_NAME.c_str()),
            return ge::GRAPH_FAILED);

        if (!fiaInfo_.isMaxWorkspace) {
            OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr,
                OP_LOGE(opName_, "when %s's layout is %s, %s data should not be null.", QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_Q_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr,
                OP_LOGE(opName_, "when %s's layout is %s, %s data should not be null.", QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
        }
    } else {
        if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
            OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                OP_LOGE(opName_, "In page attention scene, %s should not be null.", ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
            if (!fiaInfo_.isMaxWorkspace) {
                OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr,
                    OP_LOGE(opName_, "In page attention scene, %s data should not be null.", ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                    return ge::GRAPH_FAILED);
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const FiaLayout &layout, const std::string &actualSeqLenName, const std::string &attrName)
{
    if (tensor == nullptr) {
        OP_LOGE(opName_, "when layout of %s is %s, %s must be provided.",
            attrName.c_str(), LayoutToSerialString(layout).c_str(), actualSeqLenName.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "%s shape size is %ld, it should be greater than 0.",
            actualSeqLenName.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensQData()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        qSize.push_back(s1Size_);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t *actualSeq = opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>();
    if (actualSeq == nullptr) {
        qSize.push_back(s1Size_);
        return ge::GRAPH_SUCCESS;
    }

    if (GetActualSeqLenSize(actualSeqLengthsQSize_, opParamInfo_.actualSeqLengthsQ.tensor,
        qLayout_, ACTUAL_SEQ_Q_LEN_NAME, QUERY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t loop = std::min(actualSeqLengthsQSize_, bSize_);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS1 = 0;
        if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
            OP_CHECK_IF(actualSeq[i] < 0,
                OP_LOGE(opName_, "when %s's layout is %s, %s[%u] should not be a negative number, but got %ld.",
                    QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, actualSeq[i]),
                    return ge::GRAPH_FAILED);

            OP_CHECK_IF(i > 0U && (actualSeq[i] < actualSeq[i - 1U]),
                OP_LOGE(opName_, "when %s's layout is %s, %s[%u](%ld) should not be less than %s[%u](%ld).",
                    QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, actualSeq[i],
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), (i - 1U), actualSeq[i - 1U]),
                    return ge::GRAPH_FAILED);

            tmpS1 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS1 = actualSeq[i];
        }
        if (tmpS1 > static_cast<int64_t>(s1Size_) || tmpS1 < 0) {
            OP_LOGE(opName_,
                "%s[%u] computed is %ld, it should be in range [0, Q_S(%u)].",
                ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, tmpS1, s1Size_);
            return ge::GRAPH_FAILED;
        }
        qSize.push_back(tmpS1);
    }

    OP_CHECK_IF((qLayout_ == FiaLayout::TND) && (qTSize_ != actualSeq[actualSeqLengthsQSize_ - 1]),
        OP_LOGE(opName_, "when %s's layout is %s, T(%u) should be equal to the last element of %s(%ld).",
            QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(), qTSize_, ACTUAL_SEQ_Q_LEN_NAME.c_str(),
            actualSeq[actualSeqLengthsQSize_ - 1]),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensKvData()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        kvSize.push_back(s2Size_);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t *actualSeq = opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>();
    if (actualSeq == nullptr) {
        kvSize.push_back(s2Size_);
        return ge::GRAPH_SUCCESS;
    }

    if(GetActualSeqLenSize(actualSeqLengthsKvSize_, opParamInfo_.actualSeqLengths.tensor,
        kvLayout_, ACTUAL_SEQ_KV_LEN_NAME, KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t loop = std::min(actualSeqLengthsKvSize_, bSize_);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS2 = 0;
        if (kvLayout_ == FiaLayout::TND || kvLayout_ == FiaLayout::NTD) {
            OP_CHECK_IF(actualSeq[i] < 0,
                OP_LOGE(opName_, "when kv's layout is %s, %s[%u] should not be a negative number, but got %ld.",
                    LayoutToSerialString(kvLayout_).c_str(),
                    ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, actualSeq[i]),
                    return ge::GRAPH_FAILED);

            OP_CHECK_IF(i > 0U && (actualSeq[i] < actualSeq[i - 1U]),
                OP_LOGE(opName_, "when kv's layout is %s, %s[%u](%ld) should not be less than %s[%u](%ld).",
                    LayoutToSerialString(kvLayout_).c_str(),
                    ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, actualSeq[i],
                    ACTUAL_SEQ_KV_LEN_NAME.c_str(), (i - 1U), actualSeq[i - 1U]),
                    return ge::GRAPH_FAILED);

            tmpS2 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS2 = actualSeq[i];
        }

        OP_CHECK_IF(tmpS2 < 0 || tmpS2 > s2Size_,
            OP_LOGE(opName_, "%s(%u) is %ld, it should be in range [0, KV_S(%ld)].",
                ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, tmpS2, s2Size_),
            return ge::GRAPH_FAILED);
        kvSize.push_back(tmpS2);
    }

    OP_CHECK_IF((kvLayout_ == FiaLayout::TND) && (kTSize_ != actualSeq[actualSeqLengthsKvSize_ - 1]),
        OP_LOGE(opName_, "when kv's layout is %s, T(%u) should be equal to the last element of %s(%ld).",
            LayoutToSerialString(kvLayout_).c_str(), kTSize_, ACTUAL_SEQ_KV_LEN_NAME.c_str(),
            actualSeq[actualSeqLengthsKvSize_ - 1]),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureInOutDtype() const
{
    const std::vector<std::pair<ge::DataType, ge::DataType>> inOutDtypePairSupported = {
        {ge::DT_INT8, ge::DT_INT8},
        {ge::DT_INT8, ge::DT_FLOAT16},
        {ge::DT_FLOAT16, ge::DT_INT8},
        {ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_BF16, ge::DT_BF16},
        {ge::DT_BF16, ge::DT_INT8},
        {ge::DT_INT8, ge::DT_INT8},
    };

    std::pair<ge::DataType, ge::DataType> inOutDtypePair = {inputQType_, outputType_};
    if (!VecContains(inOutDtypePairSupported, inOutDtypePair)) {
        OP_LOGE(opName_, "input dtype %d with output dtype %d is not currently supported.", static_cast<int32_t>(inputQType_),
                  static_cast<int32_t>(outputType_));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckFeatureActualSeqLensExistence() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLensQData() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLensKvData()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeature()
{
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        return CheckFeatureMla();
    } else {
        return CheckFeatureGqa();
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
