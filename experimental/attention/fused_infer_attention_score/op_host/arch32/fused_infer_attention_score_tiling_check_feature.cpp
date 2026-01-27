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
 * \file fused_infer_attention_score_tiling_check_feature.cpp
 * \brief
 */

#include <numeric>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_check.h"

using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus FiaTilingCheck::CheckFeatureNoQuantDtype() const
{
    OP_CHECK_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OP_LOGE(opName_, "In %s situation, query dtype only support %s and %s, but got %s",
            QuantModeToSerialString(quantMode_).c_str(),
            FusedDataTypeToSerialString(ge::DT_BF16).c_str(), FusedDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(inputQType_ != inputKvType_,
        OP_LOGE(opName_, "In %s situation, key and value dtype(%s) must equal to query dtype(%s)",
            QuantModeToSerialString(quantMode_).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str(),
            FusedDataTypeToSerialString(inputKvType_).c_str()),
        return ge::GRAPH_FAILED);

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
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
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquant()
{
    if (ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureAxisInfo() ||
        ge::GRAPH_SUCCESS != CheckFeatureHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMla()
{
    return CheckFeatureMlaNoquant();
}

ge::graphStatus FiaTilingCheck::CheckFeatureLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSH", "BSND", "BNSD", "TND", "NTD", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD", "NTD_TND", "BSH_BNSD", "BSND_BNSD", "BNSD_BSND"
    };
    std::string layout = opParamInfo_.layOut;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
        OP_LOGE(opName_, "In %s %s situation, layout only supports BSH, BSND, BNSD, TND, NTD, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, NTD_TND, BSH_BNSD, BSND_BNSD, BNSD_BSND, but got %s",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), layout.c_str()),
        return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        OP_CHECK_IF(kvLayout_ != FiaLayout::BSH && kvLayout_ != FiaLayout::BSND && kvLayout_ != FiaLayout::BNSD &&
            kvLayout_ != FiaLayout::TND && kvLayout_ != FiaLayout::NTD,
            OP_LOGE(opName_, "In %s %s situation, key/value's layout only support BSH, BSND, BNSD, TND and NTD in batch continuous scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), LayoutToSerialString(kvLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ != qLayout_,
            OP_LOGE(opName_, "In %s %s situation, key/value's layout and query's layout should be same in batch continuous scene.",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureAxisInfo() const
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

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus FiaTilingCheck::CheckFeatureHeadDim() const
{
    constexpr uint32_t MAX_HEAD_DIM = 512;
    constexpr uint32_t MAX_ROPE_DIM = 64;

    OP_CHECK_IF(vHeadDim_ > MAX_HEAD_DIM,
    OP_LOGE(opName_, "In %s situation, headDim of value should be smaller or equal to 512, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), vHeadDim_),
    return ge::GRAPH_FAILED);

    OP_CHECK_IF(ropeHeadDim_ > MAX_ROPE_DIM,
    OP_LOGE(opName_, "In %s situation, headDim of Rope should be smaller or equal to 64, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), ropeHeadDim_),
    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoquant()
{
    if (ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureAxisInfo() ||
        ge::GRAPH_SUCCESS != CheckFeatureHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqa()
{
    return CheckFeatureGqaNoquant();
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensExistence() const
{
    if ((qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD)) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
            OP_LOGE(opName_, "when %s's layout is %s, %s should not be null.", QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                ACTUAL_SEQ_Q_LEN_NAME.c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
            OP_LOGE(opName_, "when %s's layout is %s, %s should not be null.", KEY_NAME.c_str(), LayoutToSerialString(kvLayout_).c_str(),
                ACTUAL_SEQ_KV_LEN_NAME.c_str()),
            return ge::GRAPH_FAILED);

        if (!fiaInfo_.isMaxWorkspace) {
            OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr,
                OP_LOGE(opName_, "when %s's layout is %s, %s data should not be null.", QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_Q_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr,
                OP_LOGE(opName_, "when %s's layout is %s, %s data should not be null.", KEY_NAME.c_str(), LayoutToSerialString(kvLayout_).c_str(),
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
        {ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_BF16, ge::DT_BF16},
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
