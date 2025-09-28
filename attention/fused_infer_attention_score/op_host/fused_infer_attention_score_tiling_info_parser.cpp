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
 * \file fused_infer_attention_score_tiling_info_parser.cpp
 * \brief
 */

#include <map>
#include <string>
#include <utility>
#include <numeric>
#include <algorithm>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "fused_infer_attention_score_tiling_index.h"
#include "fused_infer_attention_score_tiling_info_parser.h"


using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus FiaInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE(opName_, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE(opName_, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.shape == nullptr, OP_LOGE(opName_, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.desc == nullptr, OP_LOGE(opName_, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.shape == nullptr, OP_LOGE(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.desc == nullptr, OP_LOGE(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.numHeads == nullptr, OP_LOGE(opName_, "attr numHeads is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.scaleValue == nullptr, OP_LOGE(opName_, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kvHeadNums == nullptr, OP_LOGE(opName_, "attr kvHeadNums is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layOut == nullptr, OP_LOGE(opName_, "attr layout is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.blockSize == nullptr, OP_LOGE(opName_, "attr blockSize is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.antiquantMode == nullptr, OP_LOGE(opName_, "attr antiquantMode is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.softmaxLseFlag == nullptr, OP_LOGE(opName_, "attr softmaxLseFlag is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.keyAntiquantMode == nullptr, OP_LOGE(opName_, "attr keyAntiquantMode is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.valueAntiquantMode == nullptr, OP_LOGE(opName_, "attr valueAntiquantMode is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.innerPrecise == nullptr, OP_LOGE(opName_, "attr innerPrecise is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr, OP_LOGE(opName_, "attr sparseMode is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.queryQuantMode == nullptr, OP_LOGE(opName_, "attr queryQuantMode is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetMaxWorkspaceFlag()
{
    if ((opParamInfo_.actualSeqLengths.tensor && !opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>()) || 
        (opParamInfo_.actualSeqLengthsQ.tensor && !opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>())) {
        isMaxWorkspace_ = true;
        OP_LOGI(opName_, "FIA tiling sink");
    } else {
        isMaxWorkspace_ = false;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const FiaLayout &layout, const std::string &actualSeqLenName, const std::string &attrName)
{
    if (tensor == nullptr) {
        OP_LOGE(opName_, "when %s's layout is %s, %s must be provided.",
            attrName.c_str(), LayoutToSerialString(layout).c_str(), actualSeqLenName.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "%s's shape size is %ld, it should be greater than 0.",
            actualSeqLenName.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_,
        ACTUAL_SEQ_Q_LEN_NAME, QUERY_NAME);
}

ge::graphStatus FiaInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND310P) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910B)) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d] is not support.", static_cast<int32_t>(socVersion_));
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.pseShift.tensor = context_->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    opParamInfo_.pseShift.desc = context_->GetOptionalInputDesc(PSE_SHIFT_INDEX);
    opParamInfo_.attenMask.tensor = context_->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    opParamInfo_.attenMask.desc = context_->GetOptionalInputDesc(ATTEN_MASK_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_KV_INDEX);
    opParamInfo_.deqScale1.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE1_INDEX);
    opParamInfo_.deqScale1.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE1_INDEX);
    opParamInfo_.quantScale1.tensor = context_->GetOptionalInputTensor(QUANT_SCALE1_INDEX);
    opParamInfo_.quantScale1.desc = context_->GetOptionalInputDesc(QUANT_SCALE1_INDEX);
    opParamInfo_.deqScale2.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE2_INDEX);
    opParamInfo_.deqScale2.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE2_INDEX);
    GetOptionalInputParaPostQuantInfo();
    opParamInfo_.antiquantScale.tensor = context_->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    opParamInfo_.antiquantScale.desc = context_->GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX);
    opParamInfo_.antiquantOffset.tensor = context_->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.antiquantOffset.desc = context_->GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    opParamInfo_.queryPaddingSize.tensor = context_->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    opParamInfo_.queryPaddingSize.desc = context_->GetOptionalInputDesc(QUERY_PADDING_SIZE_INDEX);
    opParamInfo_.kvPaddingSize.tensor = context_->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    opParamInfo_.kvPaddingSize.desc = context_->GetOptionalInputDesc(KV_PADDING_SIZE_INDEX);
    opParamInfo_.keyAntiquantScale.tensor = context_->GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.keyAntiquantScale.desc = context_->GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.keyAntiquantOffset.tensor = context_->GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.keyAntiquantOffset.desc = context_->GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.valueAntiquantScale.tensor = context_->GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.valueAntiquantScale.desc = context_->GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.valueAntiquantOffset.tensor = context_->GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.valueAntiquantOffset.desc = context_->GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX);
    GetOptionalInputParaPrefixInfo();
    GetOptionalInputParaRopeInfo();
    opParamInfo_.dequantScaleQuery.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX);
    opParamInfo_.dequantScaleQuery.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX);
}

void FiaInfoParser::GetOptionalInputParaPostQuantInfo()
{
    opParamInfo_.quantScale2.tensor = context_->GetOptionalInputTensor(QUANT_SCALE2_INDEX);
    opParamInfo_.quantScale2.desc = context_->GetOptionalInputDesc(QUANT_SCALE2_INDEX);
    opParamInfo_.quantOffset2.tensor = context_->GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
    opParamInfo_.quantOffset2.desc = context_->GetOptionalInputDesc(QUANT_OFFSET2_INDEX);
}

void FiaInfoParser::GetOptionalInputParaPrefixInfo()
{
    opParamInfo_.keySharedPrefix.tensor = context_->GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    opParamInfo_.keySharedPrefix.desc = context_->GetOptionalInputDesc(KEY_SHARED_PREFIX_INDEX);
    opParamInfo_.valueSharedPrefix.tensor = context_->GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    opParamInfo_.valueSharedPrefix.desc = context_->GetOptionalInputDesc(VALUE_SHARED_PREFIX_INDEX);
    opParamInfo_.actualSharedPrefixLen.tensor = context_->GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);
    opParamInfo_.actualSharedPrefixLen.desc = context_->GetOptionalInputDesc(ACTUAL_SHARED_PREFIX_LEN_INDEX);
}

void FiaInfoParser::GetOptionalInputParaRopeInfo()
{
    opParamInfo_.queryRope.tensor = context_->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    opParamInfo_.queryRope.desc = context_->GetOptionalInputDesc(QUERY_ROPE_INDEX);
    opParamInfo_.keyRope.tensor = context_->GetOptionalInputTensor(KEY_ROPE_INDEX);
    opParamInfo_.keyRope.desc = context_->GetOptionalInputDesc(KEY_ROPE_INDEX);
    opParamInfo_.keyRopeAntiquantScale.tensor = context_->GetOptionalInputTensor(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.keyRopeAntiquantScale.desc = context_->GetOptionalInputDesc(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
}

void FiaInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INDEX);
    GetOptionalInputParaInfo();
}

void FiaInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(ATTENTION_OUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(ATTENTION_OUT_INDEX);
    opParamInfo_.lseOut.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.lseOut.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus FiaInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    opParamInfo_.numHeads = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    opParamInfo_.layOut = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
    opParamInfo_.kvHeadNums = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    opParamInfo_.blockSize = attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_SIZE_INDEX);
    opParamInfo_.antiquantMode = attrs->GetAttrPointer<int64_t>(ANTIQUANT_MODE_INDEX);
    opParamInfo_.softmaxLseFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    opParamInfo_.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(KEY_ANTIQUANT_MODE_INDEX);
    opParamInfo_.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(VALUE_ANTIQUANT_MODE_INDEX);
    opParamInfo_.innerPrecise = attrs->GetAttrPointer<int32_t>(ATTR_INNER_PRECISE_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    opParamInfo_.queryQuantMode = attrs->GetAttrPointer<int64_t>(QUERY_QUANT_MODE_INDEX);
    opParamInfo_.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    opParamInfo_.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvCache()
{
    // 处理Key和value的TensorList场景
    kCache_.clear();
    uint32_t keyBIdx = 0;
    while ((context_->GetDynamicInputShape(KEY_INDEX, keyBIdx)) != nullptr) {
        kCache_.push_back(const_cast<gert::StorageShape *>(context_->GetDynamicInputShape(KEY_INDEX, keyBIdx)));
        keyBIdx++;
    }
    OP_CHECK_IF(keyBIdx == 0,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "tensor list of %s is empty.", KEY_NAME.c_str()),
        return ge::GRAPH_FAILED);
    kCache_.resize(keyBIdx);

    vCache_.clear();
    uint32_t valueBIdx = 0;
    while ((context_->GetDynamicInputShape(VALUE_INDEX, valueBIdx)) != nullptr) {
        vCache_.push_back(const_cast<gert::StorageShape *>(context_->GetDynamicInputShape(VALUE_INDEX, valueBIdx)));
        valueBIdx++;
    }
    vCache_.resize(valueBIdx);

    OP_CHECK_IF(kCache_.size() != vCache_.size(),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "tensor list of %s has %zu tensor, but tensor list of %s has %zu tensor, they should be equal.",
            KEY_NAME.c_str(), kCache_.size(), VALUE_NAME.c_str(), vCache_.size()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    if (opParamInfo_.queryRope.desc != nullptr) {
        inputQRopeType_ = opParamInfo_.queryRope.desc->GetDataType();
    }
    if (opParamInfo_.keyRope.desc != nullptr) {
        inputKRopeType_ = opParamInfo_.keyRope.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if ((qLayout_ == FiaLayout::TND) || (qLayout_ == FiaLayout::NTD)) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSH/BSND/BNSD
        if (queryShape_->CheckHasB(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        bSize_ = queryShape_->GetB();
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus FiaInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (queryShape_->HasT()) ? static_cast<uint32_t>(queryShape_->GetT()) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    if (queryShape_->CheckHasD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    qkHeadDim_ = static_cast<uint32_t>(queryShape_->GetD()); // 后面需要把qkHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND/NTD时, 以query的S维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if ((qLayout_ == FiaLayout::TND) || (qLayout_ == FiaLayout::NTD)) {
        const int64_t *actualSeqQ = opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>();
        if (actualSeqQ == nullptr) {
            if (queryShape_->CheckHasT(__func__) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            s1Size_ = static_cast<uint32_t>(queryShape_->GetT());
            return ge::GRAPH_SUCCESS;
        }

        uint32_t b = 0;
        if(GetActualSeqLenQSize(b) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        int64_t qActualSeqMax = 0;
        for (uint32_t i = 0; i < b; i++) {
            int64_t tmpS1 = (i == 0U) ? actualSeqQ[0] : (actualSeqQ[i] - actualSeqQ[i - 1U]);
            if (tmpS1 > qActualSeqMax) {
                qActualSeqMax = tmpS1;
            }
        }
        s1Size_ = static_cast<uint32_t>(qActualSeqMax);
    } else { // BSH/BSND/BNSD
        if (queryShape_->CheckHasS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s1Size_ = static_cast<uint32_t>(queryShape_->GetS());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvStorageMode()
{
    // kv存储模式基准值
    if (kCache_.size() > 1) {
        kvStorageMode_ = KvStorageMode::TENSOR_LIST;
    } else {
        if (opParamInfo_.blockTable.tensor != nullptr) {
            kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
        } else {
            kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvLayout()
{
    // kv Layout基准值
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        kvLayout_ = qLayout_;
    } else {
        uint32_t keyDimNum = kCache_[0]->GetStorageShape().GetDimNum();
        if (keyDimNum == 3U) {
            kvLayout_ = FiaLayout::BnBsH;
        } else if (keyDimNum == 4U) {
            kvLayout_ = FiaLayout::BnNBsD;
        } else if (keyDimNum == 5U) {
            kvLayout_ = FiaLayout::NZ;
        } else {
            OP_LOGE(opName_, "the first tensor of %s's tensor list is %u dim, only support 3/4/5.",
                KEY_NAME.c_str(), keyDimNum);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2SizeForBatchContinuous()
{
    if ((kvLayout_ == FiaLayout::TND) || (kvLayout_ == FiaLayout::NTD)) { // PFA场景下需要增补
        OP_LOGE(opName_, "%s's layout is %s, it is unsupported.", KEY_NAME.c_str(),
            LayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    } else { // BSH/BSND/BNSD
        if (keyShape_->CheckHasS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s2Size_ = keyShape_->GetS();
        kvListSeqLens_.push_back(s2Size_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2SizeForTensorList()
{
    if ((kvLayout_ == FiaLayout::TND) || (kvLayout_ == FiaLayout::NTD)) { // PFA场景下需要增补
        OP_LOGE(opName_, "%s's layout is %s, it is unsupported.", KEY_NAME.c_str(),
            LayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    } else { // BSH/BSND/BNSD
        if (keyShape_->CheckHasS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        s2Size_ = 0;
        for (uint32_t i = 0; i < kCache_.size(); i++) {
            auto keyShape= std::make_shared<FiaTilingShape>(kCache_[i]->GetStorageShape(),
                kvLayout_, KEY_NAME, opName_, n1Size_);
            if (keyShape->GetS() > s2Size_) {
                s2Size_ = keyShape->GetS();
            }
            if (keyShape->GetS() != keyShape_->GetS()) {
                isSameSeqAllKVTensor_ = false;
            }
            kvListSeqLens_.push_back(keyShape->GetS());
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetMaxBlockNumPerBatch()
{
    uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (dimNum != 2U) {
        OP_LOGE(opName_, "the dim num of %s is %u, it should be 2.", BLOCK_TABLE_NAME.c_str(), dimNum);
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetBlockSize()
{
    if (opParamInfo_.blockSize == nullptr) {
        OP_LOGE(opName_, "Attr %s not exist", BLOCK_SIZE_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    blockSize_ = *(opParamInfo_.blockSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = static_cast<int64_t>(maxBlockNumPerBatch_) * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 2、TENSOR_LIST时, 从kCache_的所有Tensor的S轴的最大值
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        return GetS2SizeForTensorList();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus FiaInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    if (valueShape_->CheckHasD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    vHeadDim_ = static_cast<uint32_t>(valueShape_->GetD()); // 后面需要把vHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetRopeMode()
{
    bool existSplitRopeTensor = ((opParamInfo_.queryRope.tensor != nullptr) && (opParamInfo_.queryRope.desc != nullptr));
    if (qkHeadDim_ < vHeadDim_) {
        OP_LOGE(opName_, "the query's head dim(%u) should be greater than or equal to the value's head dim(%u)",
            qkHeadDim_, vHeadDim_);
            return ge::GRAPH_FAILED;
    } else if (qkHeadDim_ > vHeadDim_) {
        if (existSplitRopeTensor) {
            OP_LOGE(opName_, "when %s exist, the query's head dim(%u) should be equal to the value's head dim(%u). ",
                QUERY_ROPE_NAME.c_str(), qkHeadDim_, vHeadDim_);
                return ge::GRAPH_FAILED;
        } else {
            ropeMode_ = RopeMode::ROPE_COMBINE;
        }
    } else {
        if (existSplitRopeTensor) {
            ropeMode_ = RopeMode::ROPE_SPLIT;
        } else {
            ropeMode_ = RopeMode::NO_ROPE;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetRopeHeadDim()
{
    if (ge::GRAPH_SUCCESS != GetRopeMode()) {
        return ge::GRAPH_FAILED;
    }
    if (ropeMode_ == RopeMode::NO_ROPE) {
        ropeHeadDim_ = 0U;
    } else if (ropeMode_ == RopeMode::ROPE_COMBINE) {
        ropeHeadDim_ = qkHeadDim_ - vHeadDim_;
    } else {
        queryRopeShape_ = std::make_shared<FiaTilingShape>(opParamInfo_.queryRope.tensor->GetStorageShape(),
            qLayout_, QUERY_ROPE_NAME, opName_, n1Size_);
        if (queryRopeShape_->CheckHasD(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        ropeHeadDim_ = static_cast<uint32_t>(queryRopeShape_->GetD());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // inputLayout: {qLayout, outLayout}
    const map<string, pair<FiaLayout, FiaLayout>> layoutMap = {
        {"BSH",         {FiaLayout::BSH,     FiaLayout::BSH }},
        {"BSND",        {FiaLayout::BSND,    FiaLayout::BSND}},
        {"BNSD",        {FiaLayout::BNSD,    FiaLayout::BNSD}},
        {"TND",         {FiaLayout::TND,     FiaLayout::TND }},
        {"BSH_NBSD",    {FiaLayout::BSH,     FiaLayout::NBSD}},
        {"BSND_NBSD",   {FiaLayout::BSND,    FiaLayout::NBSD}},
        {"BNSD_NBSD",   {FiaLayout::BNSD,    FiaLayout::NBSD}},
        {"TND_NTD",     {FiaLayout::TND,     FiaLayout::NTD }},
        {"NTD_TND",     {FiaLayout::NTD,     FiaLayout::TND }},
        {"BNSD_BSND",   {FiaLayout::BNSD,    FiaLayout::BSND}}
    };

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OP_LOGE(opName_, "input layout is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetN1Size()
{
    // 获取N1基准值
    int32_t numHeads = *(opParamInfo_.numHeads);
    if (numHeads <= 0) {
        OP_LOGE(opName_, "%s is %d, it should be greater than 0.",
            QUERY_HEADS_NUM_NAME.c_str(), numHeads);
        return ge::GRAPH_FAILED;
    }
    n1Size_ = static_cast<uint32_t>(numHeads);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetN2Size()
{
    // 获取N2基准值
    int32_t kvHeadNums = *(opParamInfo_.kvHeadNums);
    if (kvHeadNums < 0) {
        OP_LOGE(opName_, "%s is %d, it should be greater than 0.",
            KV_HEADS_NUM_NAME.c_str(), kvHeadNums);
        return ge::GRAPH_FAILED;
    }
    n2Size_ = (kvHeadNums == 0) ? n1Size_ : static_cast<uint32_t>(kvHeadNums);
    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::SetFiaShape()
{
    queryShape_ = std::make_shared<FiaTilingShape>(opParamInfo_.query.shape->GetStorageShape(),
        qLayout_, QUERY_NAME, opName_, n1Size_);
    keyShape_ = std::make_shared<FiaTilingShape>(kCache_[0]->GetStorageShape(),
        kvLayout_, KEY_NAME, opName_, n1Size_);
    valueShape_ = std::make_shared<FiaTilingShape>(vCache_[0]->GetStorageShape(),
        kvLayout_, VALUE_NAME, opName_, n2Size_);
}

ge::graphStatus FiaInfoParser::GetGSize()
{
    // 获取G基准值
    if (n1Size_ % n2Size_ != 0U) {
        OP_LOGE(opName_, "%s(%u) should be a multiple of %s(%u).", 
            QUERY_HEADS_NUM_NAME.c_str(), n1Size_,
            KV_HEADS_NUM_NAME.c_str(), n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetAttenMaskInfo()
{
    auto *maskTensor = opParamInfo_.attenMask.tensor;
    attenMaskFlag_ = (maskTensor != nullptr) && (maskTensor->GetStorageShape().GetShapeSize() != 0);
    // CHECK: 需要进一步梳理attenMaskSize计算逻辑
    if (attenMaskFlag_) {
        if (qLayout_ == FiaLayout::TND) {
            attenMaskSize_ = 0U;
        } else if (ropeMode_ != RopeMode::NO_ROPE && s1Size_ > 1U) {
            attenMaskSize_ = 0U;
        } else {
            attenMaskSize_ = maskTensor->GetStorageShape().GetDim(maskTensor->GetStorageShape().GetDimNum() - 1);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetPaddingSizeFlag()
{
    auto *paddingSizeTensor = opParamInfo_.kvPaddingSize.tensor;
    kvPaddingSizeFlag_ = (paddingSizeTensor != nullptr) && (paddingSizeTensor->GetStorageShape().GetShapeSize() != 0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetActualSeqInfo()
{
    maxActualseq_ = s2Size_;
    if (opParamInfo_.actualSeqLengths.tensor != nullptr && opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() != nullptr) {
        const int64_t *actualLenData = opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>();
        actualLenDims_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
        OP_LOGD(opName_, "data of actual_seq_lengths is not nullptr");
        uint32_t loop = ((actualLenDims_ == 1) && (kvListSeqLens_.size() == 1)) ? 1 : bSize_;
        loop = std::min(loop, actualLenDims_);
        for (uint32_t i = 0; i < loop; i++) {
            int64_t actLen = actualLenData[i];
            needInit_ = (actLen == 0);
            maxActualseq_ = maxActualseq_ < actLen ? actLen : maxActualseq_;
            if (actualLenData[i] != actualLenData[0]) {
                isSameActualseq_ = false;
                break;
            }
        }
    }

    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr && opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() != nullptr) {
        const int64_t *actualLenQData = opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>();
        actualLenQDims_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
        OP_LOGD(opName_, "data of actual_seq_lengthsQ is not nullptr");
        uint32_t loop = std::min(bSize_, actualLenQDims_);
        for (uint32_t i = 0; i < loop; i++) {
            if (actualLenQData[i] != static_cast<int64_t>(s1Size_)) {
                needInit_ = true;
                break;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetPreNextToken()
{
    if (!(*opParamInfo_.sparseMode == 4)) { // 4:band mode
        return ge::GRAPH_SUCCESS;
    }
    preToken_ = opParamInfo_.preToken == nullptr ? 0 : *opParamInfo_.preToken;
    nextToken_ = opParamInfo_.nextToken == nullptr ? 0 : *opParamInfo_.nextToken;
    if (preToken_ > SPARSE_MODE_INT_MAX) {
        preToken_ = SPARSE_MODE_INT_MAX;
    } else if (preToken_ < -(SPARSE_MODE_INT_MAX)) {
        preToken_ = -(SPARSE_MODE_INT_MAX);
    }
    if (nextToken_ > SPARSE_MODE_INT_MAX) {
        nextToken_ = SPARSE_MODE_INT_MAX;
    } else if (nextToken_ < -(SPARSE_MODE_INT_MAX)) {
        nextToken_ = -(SPARSE_MODE_INT_MAX);
    }
    return ge::GRAPH_SUCCESS;
}

TilingKeyLayout FiaInfoParser::MapStringToLayout(FiaLayout &layoutString) const
{
    const std::map<FiaLayout, TilingKeyLayout> layoutMap = {
        {FiaLayout::BSH, TilingKeyLayout::BSH_BSND},
        {FiaLayout::BSND, TilingKeyLayout::BSH_BSND},
        {FiaLayout::BNSD, TilingKeyLayout::BNSD},
        {FiaLayout::NZ, TilingKeyLayout::NZ},
        {FiaLayout::TND, TilingKeyLayout::TND},
        {FiaLayout::NBSD, TilingKeyLayout::NBSD},
        {FiaLayout::NTD, TilingKeyLayout::NTD},
        {FiaLayout::BnBsH, TilingKeyLayout::BSH_BSND},
        {FiaLayout::BnNBsD, TilingKeyLayout::BNSD},
    };

    auto it = layoutMap.find(layoutString);
    if (it != layoutMap.end()) {
        return it->second;
    }
    return TilingKeyLayout::BSH_BSND;
}

void FiaInfoParser::GenerateInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.opName = opName_;
    fiaInfo.platformInfo = platformInfo_;
    fiaInfo.opParamInfo = opParamInfo_;
    fiaInfo.socVersion = socVersion_;
    GenerateAxisInfo(fiaInfo);
    GenerateDtypeInfo(fiaInfo);
    fiaInfo.kvStorageMode = kvStorageMode_;
    fiaInfo.ropeMode = ropeMode_;
    fiaInfo.l2CacheSize = l2CacheSize_;

    fiaInfo.kCache = kCache_;
    fiaInfo.vCache = vCache_;

    fiaInfo.l2CacheOffFlag = false;
    fiaInfo.totalBlockNum = kCache_[0]->GetStorageShape().GetDim(0);
    fiaInfo.scaleValue = *opParamInfo_.scaleValue;
    fiaInfo.innerPrecise = *opParamInfo_.innerPrecise;
    fiaInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    fiaInfo.blockSize = blockSize_;
    fiaInfo.blockTypeSize = sizeof(float);
    fiaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    fiaInfo.actualLenQDims = actualLenQDims_;
    fiaInfo.actualLenDims = actualLenDims_;
    fiaInfo.maxActualseq = maxActualseq_;
    fiaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);
    fiaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    fiaInfo.isSameActualseq = isSameActualseq_;
    fiaInfo.kvListSeqLens = kvListSeqLens_;

    fiaInfo.attenMaskFlag = attenMaskFlag_;
    fiaInfo.attenMaskSize = attenMaskSize_;
    fiaInfo.sparseMode = *opParamInfo_.sparseMode;
    fiaInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS);
    fiaInfo.kvPaddingSizeFlag = kvPaddingSizeFlag_;
    fiaInfo.softmaxLseFlag = *opParamInfo_.softmaxLseFlag;
    fiaInfo.isMaxWorkspace = isMaxWorkspace_;
    fiaInfo.needInit = needInit_;
    fiaInfo.preToken = preToken_;
    fiaInfo.nextToken = nextToken_;
    fiaInfo.slidingFlag = (*opParamInfo_.sparseMode == 4); // 4:band mode

    fiaInfo.qLayout = qLayout_;
    fiaInfo.kvLayout = kvLayout_;
    fiaInfo.outLayout = outLayout_;
    fiaInfo.inputKvLayout = MapStringToLayout(kvLayout_);
    fiaInfo.inputLayout = MapStringToLayout(qLayout_);
    fiaInfo.outputLayout = MapStringToLayout(outLayout_);
}

void FiaInfoParser::GenerateAxisInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.bSize = bSize_;
    fiaInfo.n1Size = n1Size_;
    fiaInfo.n2Size = n2Size_;
    fiaInfo.s1Size = s1Size_;
    fiaInfo.s2Size = s2Size_;
    fiaInfo.gSize = gSize_;
    fiaInfo.qkHeadDim = qkHeadDim_;
    fiaInfo.vHeadDim = vHeadDim_;
    fiaInfo.ropeHeadDim = ropeHeadDim_;
    fiaInfo.qTSize = qTSize_;
}

void FiaInfoParser::GenerateDtypeInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.inputQType = inputQType_;
    fiaInfo.inputKvType = inputKvType_;
    fiaInfo.inputQRopeType = inputQRopeType_;
    fiaInfo.inputKRopeType = inputKRopeType_;
    fiaInfo.outputType = outputType_;
}

ge::graphStatus FiaInfoParser::Parse(FiaTilingInfo &fiaInfo)
{
    if (context_ == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != GetKvCache() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetKvStorageMode() ||
        ge::GRAPH_SUCCESS != GetKvLayout()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseAxisInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseFeatureInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(fiaInfo);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::ParseAxisInfo()
{
    if (ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size()) {
        return ge::GRAPH_FAILED;
    }
    SetFiaShape();
    if (ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetQkHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetRopeHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::ParseFeatureInfo()
{
    if (ge::GRAPH_SUCCESS != GetMaxWorkspaceFlag() ||
        ge::GRAPH_SUCCESS != GetAttenMaskInfo() ||
        ge::GRAPH_SUCCESS != GetPaddingSizeFlag() ||
        ge::GRAPH_SUCCESS != GetActualSeqInfo() ||
        ge::GRAPH_SUCCESS != GetPreNextToken()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
