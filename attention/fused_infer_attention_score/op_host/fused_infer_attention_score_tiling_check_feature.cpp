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
    constexpr uint32_t MAX_B_SIZE = 65536U;
    constexpr uint32_t MAX_QT_BYTE = 1024U * 1024U;

    OP_CHECK_IF(opParamInfo_.keyRope.tensor->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(opName_, "In %s situation, %s tensor should not be empty",
            RopeModeToSerialString(ropeMode_).c_str(), KEY_ROPE_NAME.c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.key.shape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(opName_, "In %s situation, %s tensor should not be empty",
            RopeModeToSerialString(ropeMode_).c_str(), KEY_NAME.c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(bSize_ > MAX_B_SIZE,
        OP_LOGE(opName_, "batch size(%u) cannot be greater than %u.", bSize_, MAX_B_SIZE),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1,
        OP_LOGE(opName_, "In %s %s situation, %s should be 1, but got %u",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
            KV_HEADS_NUM_NAME.c_str(), n2Size_),
        return ge::GRAPH_FAILED);

    if (fiaInfo_.slidingFlag) {
        std::vector<uint32_t> gSizeSupportList = {128};
        OP_CHECK_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
            OP_LOGE(opName_, "In %s %s situation and sliding attention is enabled, group num should be in 128, but got %u",
                RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(), gSize_),
            return ge::GRAPH_FAILED);
    } else {
        std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
        OP_CHECK_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
            OP_LOGE(opName_, "In %s %s situation, group num should be in 1, 2, 4, 8, 16, 32, 64, 128, but got %u",
                RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(), gSize_),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(qkHeadDim_ != 512,
        OP_LOGE(opName_, "In %s %s situation, the query/key's head dim only support 512, but got %u",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qkHeadDim_ != vHeadDim_,
        OP_LOGE(opName_, "In %s %s situation, the query/key's head dim(%u) should be equal to the value's head dim(%u)",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(), qkHeadDim_, vHeadDim_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ropeHeadDim_ != 64,
        OP_LOGE(opName_, "In %s %s situation, the rope's head dim should be 64, but got %u",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(), ropeHeadDim_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qTSize_ > MAX_QT_BYTE / GetTypeSize(inputQType_),
        OP_LOGE(opName_, "In %s %s situation, query T should be smaller than %u / sizeof(query_dtype) = %u, but got %u",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(), MAX_QT_BYTE,
            MAX_QT_BYTE / GetTypeSize(inputQType_), qTSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoQuantLayout() const
{
    std::string layout = opParamInfo_.layOut;
    if (fiaInfo_.slidingFlag) {
        const std::vector<std::string> layoutSupportList = {"BSND"};
        OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
            OP_LOGE(opName_, "In %s %s situation and sliding attention is enabled, layout only supports BSND, but got %s",
                RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(), layout.c_str()),
            return ge::GRAPH_FAILED);
    } else {
        const std::vector<std::string> layoutSupportList = {
            "BSH", "BSND", "BNSD", "TND", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD"
        };
        OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
            OP_LOGE(opName_, "In %s %s situation, layout only supports BSH, BSND, BNSD, TND, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, but got %s",
                RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(), layout.c_str()),
            return ge::GRAPH_FAILED);
    }

    const std::vector<std::string> layoutSupportListNz = {
        "BSH", "BSND", "TND", "BSH_NBSD", "BSND_NBSD", "TND_NTD"
    };
    OP_CHECK_IF(kvLayout_ == FiaLayout::NZ &&
        std::find(layoutSupportListNz.begin(), layoutSupportListNz.end(), layout) == layoutSupportListNz.end(),
        OP_LOGE(opName_, "In %s %s situation and the key/value's layout is NZ, layout only supports BSH, BSND, TND, BSH_NBSD, BSND_NBSD, TND_NTD, but got %s",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(), layout.c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvLayout_ == FiaLayout::BnNBsD && (qLayout_ != FiaLayout::BNSD && qLayout_ != FiaLayout::TND),
        OP_LOGE(opName_, "In page attention scene, the key/value's layout is BnNBsD, %s layout must be BNSD or TND, but got %s",
            QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoQuantDtype() const
{
    OP_CHECK_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OP_LOGE(opName_, "In %s %s situation, query dtype only support %s and %s, but got %s",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
            FusedDataTypeToSerialString(ge::DT_BF16).c_str(), FusedDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);

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

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquantPa() const
{
    constexpr uint32_t MAX_BLOCK_SIZE = 512;
    constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;

    if (fiaInfo_.slidingFlag && kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        OP_LOGE(opName_, "In MLA NO_QUANT situation and sparse mode is not 4, "
            "the key/value's storage mode only supports page attention");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OP_LOGE(opName_, "when page attention is enabled, block_size(%d) should be in range (0, %u].",
        blockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);

    if (kvLayout_ == FiaLayout::NZ) {
        if (fiaInfo_.slidingFlag) {
            OP_LOGE(opName_, "In %s %s situation and sliding attention is enabled, only supports "
                "key/value's layout is ND, but now is NZ",
                RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str());
            return ge::GRAPH_FAILED;
        } else {
            OP_CHECK_IF(kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && blockSize_ != 128,
                OP_LOGE(opName_, "In %s %s situation and the key/value's layout is NZ, %s should be 128, but got %d",
                    RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
                    BLOCK_SIZE_NAME.c_str(), blockSize_),
                return ge::GRAPH_FAILED);
        }
    } else {
        if (fiaInfo_.slidingFlag) {
            OP_CHECK_IF(kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && blockSize_ != 64 && blockSize_ != 128,
                OP_LOGE(opName_, "In %s %s situation and the key/value's layout is ND, "
                    "and sliding attention is enabled, %s should be 64 or 128, but got %d",
                    RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
                    BLOCK_SIZE_NAME.c_str(), blockSize_),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && blockSize_ != 16 && blockSize_ != 128,
                OP_LOGE(opName_, "In %s %s situation and the key/value's layout is ND, %s should be 16 or 128, but got %d",
                    RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
                    BLOCK_SIZE_NAME.c_str(), blockSize_),
                return ge::GRAPH_FAILED);
        }
    }

    // gm到l1，copynd2nz的srcDValue最大支持65535
    if (qLayout_ == FiaLayout::BSH || qLayout_ == FiaLayout::BSND) {
        OP_CHECK_IF(n2Size_ * qkHeadDim_ > COPYND2NZ_SRC_STRIDE_LIMITATION,
            OP_LOGE(opName_,
                "In %s %s situation, When input kvcache layout is BSH, the N * D of kvcache is %u, exceeds the maximum limit (%u) of the datacopy instruction.",
                RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
                n2Size_ * qkHeadDim_, COPYND2NZ_SRC_STRIDE_LIMITATION),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquantMask() const
{
    if (qLayout_ == FiaLayout::TND) {
        OP_CHECK_IF(fiaInfo_.sparseMode != 3 && fiaInfo_.sparseMode != 0,
            OP_LOGE(opName_, "In %s %s situation, When %s layout is %s, %s must be 0 or 3, but got %d.",
                RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
                QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(), SPARSE_MODE_NAME.c_str(), fiaInfo_.sparseMode),
            return ge::GRAPH_FAILED);
    } else {
        if (s1Size_ > 1U) {
            OP_CHECK_IF(fiaInfo_.sparseMode != 3 && fiaInfo_.sparseMode != 4, // 3 : rightDown, 4 : band
                OP_LOGE(opName_,
                    "In %s %s situation, when %s layout is not TND and Q_S(%u) > 1, supports %s = 3 or 4, but got %d.",
                    RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
                    QUERY_NAME.c_str(), s1Size_, SPARSE_MODE_NAME.c_str(), fiaInfo_.sparseMode),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(fiaInfo_.sparseMode != 0 && fiaInfo_.sparseMode != 4,
                OP_LOGE(opName_,
                    "In %s %s situation, when %s layout is not TND and Q_S = %u, supports %s = 0 or 4, but got %d.",
                    RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
                    QUERY_NAME.c_str(), s1Size_, SPARSE_MODE_NAME.c_str(), fiaInfo_.sparseMode),
                return ge::GRAPH_FAILED);
        }
    }

    OP_CHECK_IF(fiaInfo_.sparseMode == 0 && attenMaskFlag_,
        OP_LOGE(opName_, "In %s %s situation, when %s = %d, %s must not exist.",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
            SPARSE_MODE_NAME.c_str(), fiaInfo_.sparseMode, ATTEN_MASK_NAME.c_str()),
        return ge::GRAPH_FAILED);  
    OP_CHECK_IF(fiaInfo_.sparseMode == 3 && (!attenMaskFlag_),
        OP_LOGE(opName_, "In %s %s situation, when %s = %d, %s must exist.",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str(),
            SPARSE_MODE_NAME.c_str(), fiaInfo_.sparseMode, ATTEN_MASK_NAME.c_str()),
        return ge::GRAPH_FAILED);  

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquantUnsupported() const
{
    OP_CHECK_IF(fiaInfo_.pseShiftFlag,
        OP_LOGE(opName_, "%s is not supported in MLA.", PSE_SHIFT_NAME.c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo_.sysPrefixFlag,
        OP_LOGE(opName_, "shared prefix is not supported in MLA."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo_.softmaxLseFlag,
        OP_LOGE(opName_, "%s output is not supported in MLA.", SOFTMAX_LSE_NAME.c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo_.outputType == ge::DT_INT8,
        OP_LOGE(opName_, "post_quant is not supported in MLA."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo_.kvPaddingSizeFlag,
        OP_LOGE(opName_, "%s is not supported in MLA.", KV_PADDING_SIZE_NAME.c_str()),
        return ge::GRAPH_FAILED);
    
    OP_CHECK_IF((fiaInfo_.innerPrecise != INNER_PRECISE_HIGH_PRECISION) &&
        (fiaInfo_.innerPrecise != INNER_PRECISE_HIGH_PERFORMANCE),
        OP_LOGE(opName_, "%s(%d) should be 0 or 1 in MLA", INNER_PRECISE_NAME.c_str(), fiaInfo_.innerPrecise),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquant() const
{
    OP_CHECK_IF(socVersion_ == platform_ascendc::SocVersion::ASCEND310P,
        OP_LOGE(opName_, "In %s %s situation, Ascend310P is not supported",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantUnsupported() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantPa() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantMask() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantShape()) {
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

ge::graphStatus FiaTilingCheck::CheckFeatureMla() const
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

ge::graphStatus FiaTilingCheck::CheckFeatureGqa() const
{
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
        OP_CHECK_IF(!fiaInfo_.slidingFlag && opParamInfo_.actualSeqLengthsQ.tensor != nullptr,
            OP_LOGE(opName_, "when %s's layout is %s, %s should be null.",
                QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(), ACTUAL_SEQ_Q_LEN_NAME.c_str()),
            return ge::GRAPH_FAILED);
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

    constexpr int64_t ACTUAL_SEQ_Q_LEN_MAX = 16;
    uint32_t loop = std::min(actualSeqLengthsQSize_, bSize_);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS1 = 0;
        if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
            tmpS1 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
            if ((tmpS1 < 0) || (tmpS1 > ACTUAL_SEQ_Q_LEN_MAX)) {
                OP_LOGE(opName_,
                    "%s(%u) computed is %ld, it should be in range [0, %ld].",
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, tmpS1, ACTUAL_SEQ_Q_LEN_MAX);
                return ge::GRAPH_FAILED;
            }
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
    if (ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens()) {
        return ge::GRAPH_FAILED;
    }

    if (ropeMode_ != RopeMode::NO_ROPE) {
        return CheckFeatureMla();
    } else {
        return CheckFeatureGqa();
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
