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
 * \file fused_infer_attention_score_tiling_info_parser.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "../fused_infer_attention_score_tiling_index.h"
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
    OP_CHECK_IF(opParamInfo_.innerPrecise == nullptr, OP_LOGE(opName_, "attr innerPrecise is nullptr"),
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

ge::graphStatus FiaInfoParser::GetLegacyIfaFlag()
{
    uint32_t querySize = 0;
    std::string layout(opParamInfo_.layOut);
    if (layout == "BSH" || layout == "BSND" || layout == "BNSD") {
        if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        querySize = static_cast<uint32_t>(queryShape_->GetShapeS());
        if (querySize == 1U &&
            qkHeadDim_ == vHeadDim_ &&
            opParamInfo_.queryRope.tensor == nullptr &&
            opParamInfo_.keyRope.tensor == nullptr) {
            isLegacyIfa_ = true;
        }
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
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND910B) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d] is not support.", static_cast<int32_t>(socVersion_));
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_KV_INDEX);

    GetOptionalInputParaRopeInfo();
}


void FiaInfoParser::GetOptionalInputParaRopeInfo()
{
    opParamInfo_.queryRope.tensor = context_->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    opParamInfo_.queryRope.desc = context_->GetOptionalInputDesc(QUERY_ROPE_INDEX);
    opParamInfo_.keyRope.tensor = context_->GetOptionalInputTensor(KEY_ROPE_INDEX);
    opParamInfo_.keyRope.desc = context_->GetOptionalInputDesc(KEY_ROPE_INDEX);
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
}

ge::graphStatus FiaInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    // sparseMode,preToken,nextToken在GetUpdateInfo()中获取
    opParamInfo_.numHeads = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    opParamInfo_.layOut = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
    opParamInfo_.kvHeadNums = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    opParamInfo_.innerPrecise = attrs->GetAttrPointer<int32_t>(ATTR_INNER_PRECISE_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetUpdateInfo()
{
    if (isLegacyIfa_) {
        opParamInfo_.actualSeqLengthsQ.tensor = nullptr;
        opParamInfo_.actualSeqLengthsQ.desc = nullptr;
    } else {
        opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
        opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    }
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
        if (queryShape_->CheckHasShapeB(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        bSize_ = queryShape_->GetShapeB();
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus FiaInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (queryShape_->HasShapeT()) ? static_cast<uint32_t>(queryShape_->GetShapeT()) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKTSize()
{
    kTSize_ = (keyShape_->HasShapeT()) ? static_cast<uint32_t>(keyShape_->GetShapeT()) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    if (queryShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    qkHeadDim_ = static_cast<uint32_t>(queryShape_->GetShapeD()); // 后面需要把qkHeadDim_改成uint64
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
            if (queryShape_->CheckHasShapeT(__func__) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            s1Size_ = static_cast<uint32_t>(queryShape_->GetShapeT());
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
        if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s1Size_ = static_cast<uint32_t>(queryShape_->GetShapeS());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvStorageMode()
{
    // kv存储模式基准值
    if (kCache_.size() > 1) {
        OP_LOGE(opName_, "kv not support tensorlist!");
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvLayout()
{
    // kv Layout基准值
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        kvLayout_ = qLayout_;
    } 
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetMaxActualSeq(const gert::Tensor *actualSeqLensTensor,
    FiaLayout layout, int64_t &maxActualSeqLen)
{
    maxActualSeqLen = 0;
    const int64_t *actualLenData = actualSeqLensTensor->GetData<int64_t>();
    if (actualLenData != nullptr) {
        uint32_t loop = std::min(static_cast<uint32_t>(actualSeqLensTensor->GetShapeSize()), bSize_);
        for (uint32_t i = 0; i < loop; i++) {
            int64_t tmpS = 0;
            if (layout == FiaLayout::TND || layout == FiaLayout::NTD) {
                tmpS = (i == 0U) ? actualLenData[0] : (actualLenData[i] - actualLenData[i - 1U]);
            } else {
                tmpS = actualLenData[i];
            }
            maxActualSeqLen = std::max(maxActualSeqLen, tmpS);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2SizeFromActualSeqLens()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return GetMaxActualSeq(opParamInfo_.actualSeqLengths.tensor, kvLayout_, s2Size_);
}

ge::graphStatus FiaInfoParser::GetS2SizeForBatchContinuous()
{
    if ((kvLayout_ == FiaLayout::TND) || (kvLayout_ == FiaLayout::NTD)) {
        return GetS2SizeFromActualSeqLens();
    } else { // BSH/BSND/BNSD
        if (keyShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s2Size_ = keyShape_->GetShapeS();
        kvListSeqLens_.push_back(s2Size_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    return GetS2SizeForBatchContinuous();
}

ge::graphStatus FiaInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    if (valueShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    vHeadDim_ = static_cast<uint32_t>(valueShape_->GetShapeD()); // 后面需要把vHeadDim_改成uint64
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
        if (queryRopeShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        ropeHeadDim_ = static_cast<uint32_t>(queryRopeShape_->GetShapeD());
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
        {"BNSD_BSND",   {FiaLayout::BNSD,    FiaLayout::BSND}},
        {"BSND_BNSD",   {FiaLayout::BSND,    FiaLayout::BNSD}},
        {"BSH_BNSD",    {FiaLayout::BSH,     FiaLayout::BNSD}},
        {"NTD",         {FiaLayout::NTD,     FiaLayout::NTD}}
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

ge::graphStatus FiaInfoParser::GetActualSeqInfo()
{
    maxActualseq_ = 0;
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDims_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
        if ((kvLayout_ == FiaLayout::TND) || (kvLayout_ == FiaLayout::NTD)) {
            isAccumKVSeq_ = true;
        }
        if (opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() != nullptr) {
            const int64_t *actualLenData = opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>();
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
    } else {
        maxActualseq_ = s2Size_;
    }

    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
        const int64_t *actualLenQData = opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>();
        actualLenQDims_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
        if ((qLayout_ == FiaLayout::TND) || (qLayout_ == FiaLayout::NTD)) {
            isAccumQSeq_ = true;
        }
        if (opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() != nullptr) {
            uint32_t loop = std::min(bSize_, actualLenQDims_);
            for (uint32_t i = 0; i < loop; i++) {
                if (actualLenQData[i] != static_cast<int64_t>(s1Size_)) {
                    needInit_ = true;
                    break;
                }
            }
        }
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

void FiaInfoParser::GenerateFeatureInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.innerPrecise = *opParamInfo_.innerPrecise;
}
 
void FiaInfoParser::GenerateLayoutInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.qLayout = qLayout_;
    fiaInfo.kvLayout = kvLayout_;
    fiaInfo.outLayout = outLayout_;
    fiaInfo.inputKvLayout = MapStringToLayout(kvLayout_);
    fiaInfo.inputLayout = MapStringToLayout(qLayout_);
    fiaInfo.outputLayout = MapStringToLayout(outLayout_);
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
    fiaInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS);
    fiaInfo.ropeMode = ropeMode_;
    fiaInfo.l2CacheSize = l2CacheSize_;

    fiaInfo.kCache = kCache_;
    fiaInfo.vCache = vCache_;

    fiaInfo.totalOutputSize = opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize();

    fiaInfo.l2CacheOffFlag = false;
    fiaInfo.totalBlockNum = kCache_[0]->GetStorageShape().GetDim(0);
    fiaInfo.scaleValue = *opParamInfo_.scaleValue;
    fiaInfo.needInit = needInit_;

    fiaInfo.actualLenQDims = actualLenQDims_;
    fiaInfo.actualLenDims = actualLenDims_;
    fiaInfo.maxActualseq = maxActualseq_;
    fiaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);
    fiaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    fiaInfo.isSameActualseq = isSameActualseq_;
    fiaInfo.kvListSeqLens = kvListSeqLens_;

    fiaInfo.isAccumQSeq = isAccumQSeq_;
    fiaInfo.isAccumKVSeq = isAccumKVSeq_;

    GenerateFeatureInfo(fiaInfo);
    GenerateLayoutInfo(fiaInfo);
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
    fiaInfo.kTSize = kTSize_;
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
    if (ge::GRAPH_SUCCESS != GetQkHeadDim() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetLegacyIfaFlag() ||
        ge::GRAPH_SUCCESS != GetUpdateInfo() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetKTSize() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetRopeHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::ParseFeatureInfo()
{
    if (ge::GRAPH_SUCCESS != GetActualSeqInfo()){
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
