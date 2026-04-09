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
 * \file flash_attn_tiling_info_parser.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "flash_attn_tiling_index.h"
#include "flash_attn_tiling_info_parser.h"


using std::map;
using std::pair;
using std::string;
using namespace ge;
// using namespace AscendC;
namespace optiling {

ge::graphStatus FaInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE(opName_, "Shape of tensor query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE(opName_, "Desc of tensor query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE(opName_, "Shape of tensor key is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE(opName_, "Desc of tensor key is nullptr"),
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

ge::graphStatus FaInfoParser::CheckRequiredAttrExistence() const
{
    // OP_CHECK_IF(opParamInfo_.numHeads == nullptr, OP_LOGE(opName_, "attr numHeads is nullptr"), // TOUSE: numHeads属性
    //             return ge::GRAPH_FAILED); // TOUSE
    OP_CHECK_IF(opParamInfo_.softmaxScale == nullptr, OP_LOGE(opName_, "attr softmaxScale is nullptr"),
                return ge::GRAPH_FAILED);
    // OP_CHECK_IF(opParamInfo_.kvHeadNums == nullptr, OP_LOGE(opName_, "attr kvHeadNums is nullptr"), // TOUSE: kvHeadNums属性
    //             return ge::GRAPH_FAILED); // TOUSE
    // OP_CHECK_IF(opParamInfo_.layOut == nullptr, OP_LOGE(opName_, "attr layout is nullptr"), return ge::GRAPH_FAILED); // TOUSE: layout属性检查
    // OP_CHECK_IF(opParamInfo_.blockSize == nullptr, OP_LOGE(opName_, "attr blockSize is nullptr"), // TOUSE: blockSize属性
    //             return ge::GRAPH_FAILED); // TOUSE
    // OP_CHECK_IF(opParamInfo_.antiquantMode == nullptr, OP_LOGE(opName_, "attr antiquantMode is nullptr"), // TOUSE: antiquantMode属性
    //             return ge::GRAPH_FAILED); // TOUSE
    OP_CHECK_IF(opParamInfo_.softmaxLseFlag == nullptr, OP_LOGE(opName_, "attr softmaxLseFlag is nullptr"),
                return ge::GRAPH_FAILED);
    // OP_CHECK_IF(opParamInfo_.keyAntiquantMode == nullptr, OP_LOGE(opName_, "attr keyAntiquantMode is nullptr"), // TOUSE: keyAntiquantMode属性
    //             return ge::GRAPH_FAILED); // TOUSE
    // OP_CHECK_IF(opParamInfo_.valueAntiquantMode == nullptr, OP_LOGE(opName_, "attr valueAntiquantMode is nullptr"), // TOUSE: valueAntiquantMode属性
    //             return ge::GRAPH_FAILED); // TOUSE
    // OP_CHECK_IF(opParamInfo_.innerPrecise == nullptr, OP_LOGE(opName_, "attr innerPrecise is nullptr"), // TOUSE: innerPrecise属性
    //             return ge::GRAPH_FAILED); // TOUSE
    // OP_CHECK_IF(opParamInfo_.queryQuantMode == nullptr, OP_LOGE(opName_, "attr queryQuantMode is nullptr"), // TOUSE: queryQuantMode属性
    //             return ge::GRAPH_FAILED); // TOUSE
    // OP_CHECK_IF(opParamInfo_.pseType == nullptr, OP_LOGE(opName_, "attr pseType is nullptr"), // TOUSE: pseType属性
    //             return ge::GRAPH_FAILED); // TOUSE

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetEmptyTensorFlag()
{
    if ((opParamInfo_.query.shape->GetStorageShape().GetShapeSize() == 0 &&
         opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize() != 0) ||
        (opParamInfo_.query.shape->GetStorageShape().GetShapeSize() != 0 &&
         opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize() == 0)) {
        OP_LOGE(opName_,
                "query shape size is %llu byte, but attention Out shape size is %llu byte, they cannot be empty while "
                "the other is not",
                opParamInfo_.query.shape->GetStorageShape().GetShapeSize(),
                opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize());
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.query.shape->GetStorageShape().GetShapeSize() == 0 &&
        opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize() == 0) {
        emptyTensorFlag_ = true;
        return ge::GRAPH_SUCCESS;
    }
    if (*opParamInfo_.softmaxLseFlag) {
        if ((opParamInfo_.lseOut.shape == nullptr) ||
            (opParamInfo_.lseOut.shape->GetStorageShape().GetShapeSize() == 0)) {
            OP_LOGE(opName_, "lse Flag is %d, but lse shape size is 0 byte", *opParamInfo_.softmaxLseFlag);
            return ge::GRAPH_FAILED;
        }
    }
    // for (auto &kTensor : kCache_) {
    //     if (kTensor->GetStorageShape().GetShapeSize() != 0) {
    //         return ge::GRAPH_SUCCESS;
    //     }
    // }
    // for (auto &vTensor : vCache_) {
    //     if (vTensor->GetStorageShape().GetShapeSize() != 0) {
    //         return ge::GRAPH_SUCCESS;
    //     }
    // }
    emptyTensorFlag_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetMaxWorkspaceFlag()
{
    // if ((opParamInfo_.actualSeqLengths.tensor && !opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>()) ||
    //     (opParamInfo_.actualSeqLengthsQ.tensor && !opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>())) {
    //     isMaxWorkspace_ = true;
    //     OP_LOGI(opName_, "FA tiling sink");
    // } else {
        isMaxWorkspace_ = false;
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetLegacyIfaFlag()
{
    // uint32_t querySize = 0;
    // std::string layout(opParamInfo_.layOut);
    // if (layout == "BSH" || layout == "BSND" || layout == "BNSD") {
    //     if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
    //         return ge::GRAPH_FAILED;
    //     }
    //     querySize = static_cast<uint32_t>(queryShape_->GetShapeS());
    //     if (querySize == 1U && qkHeadDim_ == vHeadDim_ && opParamInfo_.queryRope.tensor == nullptr &&
    //         opParamInfo_.keyRope.tensor == nullptr) {
            isLegacyIfa_ = true;
    //     }
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    // if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
    //     OP_LOGE(opName_, "when %s's layout is %s, %s must be provided.", QUERY_NAME.c_str(),
    //             LayoutToSerialString(qLayout_).c_str(), ACTUAL_SEQ_Q_LEN_NAME.c_str());
    //     return ge::GRAPH_FAILED;
    // }
    // int64_t shapeSize = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
    // if (shapeSize <= 0) {
    //     OP_LOGE(opName_, "%s's shape size is %ld, it should be greater than 0.", ACTUAL_SEQ_Q_LEN_NAME.c_str(),
    //             shapeSize);
    //     return ge::GRAPH_FAILED;
    // }
    // size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("FlashAttn", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."),
                return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND310P) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (npuArch_ != NpuArch::DAV_3510) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_55)) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d]/NpuArch[%d] is not support.", static_cast<int32_t>(socVersion_), static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetOptionalInputParaInfo()
{
    // actualSeqLengthsQ和queryPaddingSize在GetUpdateInfo()中获取
    GetOptionalInputParaMaskInfo();
    GetOptionalInputParaActualSeqLengthInfo();
    GetOptionalInputParaPageAttentionInfo();
    GetOptionalInputParaPostQuantInfo();
    GetOptionalInputParaRopeInfo();
    GetOptionalInputParaPseInfo();
    GetOptionalInputParaLeftPaddingInfo();
    GetOptionalInputParaPrefixInfo();
    GetOptionalInputParaLearnableSinkInfo();
    GetOptionalInputParaQuantInfo();
}
void FaInfoParser::GetOptionalInputParaMaskInfo()
{
    // opParamInfo_.attenMask.tensor = context_->GetOptionalInputTensor(ATTEN_MASK_INDEX); // TOUSE: 获取attention mask tensor
    // opParamInfo_.attenMask.desc = context_->GetOptionalInputDesc(ATTEN_MASK_INDEX); // TOUSE: 获取attention mask desc
}

void FaInfoParser::GetOptionalInputParaActualSeqLengthInfo()
{
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_KV_INDEX);
}

void FaInfoParser::GetOptionalInputParaPageAttentionInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void FaInfoParser::GetOptionalInputParaPostQuantInfo()
{
    // opParamInfo_.quantScale2.tensor = context_->GetOptionalInputTensor(QUANT_SCALE2_INDEX); // TOUSE: 后量化scale
    // opParamInfo_.quantScale2.desc = context_->GetOptionalInputDesc(QUANT_SCALE2_INDEX); // TOUSE
    // opParamInfo_.quantOffset2.tensor = context_->GetOptionalInputTensor(QUANT_OFFSET2_INDEX); // TOUSE: 后量化offset
    // opParamInfo_.quantOffset2.desc = context_->GetOptionalInputDesc(QUANT_OFFSET2_INDEX); // TOUSE
}

void FaInfoParser::GetOptionalInputParaRopeInfo()
{
    // opParamInfo_.queryRope.tensor = context_->GetOptionalInputTensor(QUERY_ROPE_INDEX); // TOUSE: query rope tensor
    // opParamInfo_.queryRope.desc = context_->GetOptionalInputDesc(QUERY_ROPE_INDEX); // TOUSE
    // opParamInfo_.keyRope.tensor = context_->GetOptionalInputTensor(KEY_ROPE_INDEX); // TOUSE: key rope tensor
    // opParamInfo_.keyRope.desc = context_->GetOptionalInputDesc(KEY_ROPE_INDEX); // TOUSE
    // opParamInfo_.keyRopeAntiquantScale.tensor = context_->GetOptionalInputTensor(KEY_ROPE_ANTIQUANT_SCALE_INDEX); // TOUSE: key rope antiquant scale
    // opParamInfo_.keyRopeAntiquantScale.desc = context_->GetOptionalInputDesc(KEY_ROPE_ANTIQUANT_SCALE_INDEX); // TOUSE
}

void FaInfoParser::GetOptionalInputParaPseInfo()
{
    // opParamInfo_.pseShift.tensor = context_->GetOptionalInputTensor(PSE_SHIFT_INDEX); // TOUSE: PSE shift tensor
    // opParamInfo_.pseShift.desc = context_->GetOptionalInputDesc(PSE_SHIFT_INDEX); // TOUSE
    // opParamInfo_.qStartIdx.tensor = context_->GetOptionalInputTensor(Q_START_IDX_INDEX); // TOUSE: query start index
    // opParamInfo_.qStartIdx.desc = context_->GetOptionalInputDesc(Q_START_IDX_INDEX); // TOUSE
    // opParamInfo_.kvStartIdx.tensor = context_->GetOptionalInputTensor(KV_START_IDX_INDEX); // TOUSE: KV start index
    // opParamInfo_.kvStartIdx.desc = context_->GetOptionalInputDesc(KV_START_IDX_INDEX); // TOUSE
}

void FaInfoParser::GetOptionalInputParaLeftPaddingInfo()
{
    // opParamInfo_.kvPaddingSize.tensor = context_->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX); // TOUSE: KV padding size
    // opParamInfo_.kvPaddingSize.desc = context_->GetOptionalInputDesc(KV_PADDING_SIZE_INDEX); // TOUSE
}

void FaInfoParser::GetOptionalInputParaPrefixInfo()
{
    // opParamInfo_.keySharedPrefix.tensor = context_->GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX); // TOUSE: key shared prefix
    // opParamInfo_.keySharedPrefix.desc = context_->GetOptionalInputDesc(KEY_SHARED_PREFIX_INDEX); // TOUSE
    // opParamInfo_.valueSharedPrefix.tensor = context_->GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX); // TOUSE: value shared prefix
    // opParamInfo_.valueSharedPrefix.desc = context_->GetOptionalInputDesc(VALUE_SHARED_PREFIX_INDEX); // TOUSE
    // opParamInfo_.actualSharedPrefixLen.tensor = context_->GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX); // TOUSE: actual shared prefix length
    // opParamInfo_.actualSharedPrefixLen.desc = context_->GetOptionalInputDesc(ACTUAL_SHARED_PREFIX_LEN_INDEX); // TOUSE
}

void FaInfoParser::GetOptionalInputParaLearnableSinkInfo()
{
    // opParamInfo_.learnableSink.tensor = context_->GetOptionalInputTensor(LEARNABLE_SINK_INDEX); // TOUSE: learnable sink tensor
    // opParamInfo_.learnableSink.desc = context_->GetOptionalInputDesc(LEARNABLE_SINK_INDEX); // TOUSE
}

void FaInfoParser::GetOptionalInputParaQuantInfo()
{
    // opParamInfo_.deqScale1.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE1_INDEX); // TOUSE: dequant scale1
    // opParamInfo_.deqScale1.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE1_INDEX); // TOUSE
    // opParamInfo_.quantScale1.tensor = context_->GetOptionalInputTensor(QUANT_SCALE1_INDEX); // TOUSE: quant scale1
    // opParamInfo_.quantScale1.desc = context_->GetOptionalInputDesc(QUANT_SCALE1_INDEX); // TOUSE
    // opParamInfo_.deqScale2.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE2_INDEX); // TOUSE: dequant scale2
    // opParamInfo_.deqScale2.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE2_INDEX); // TOUSE
    // opParamInfo_.dequantScaleQuery.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX); // TOUSE: dequant scale query
    // opParamInfo_.dequantScaleQuery.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX); // TOUSE
    // opParamInfo_.antiquantScale.tensor = context_->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX); // TOUSE: antiquant scale
    // opParamInfo_.antiquantScale.desc = context_->GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX); // TOUSE
    // opParamInfo_.antiquantOffset.tensor = context_->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX); // TOUSE: antiquant offset
    // opParamInfo_.antiquantOffset.desc = context_->GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX); // TOUSE
    // opParamInfo_.keyAntiquantScale.tensor = context_->GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX); // TOUSE: key antiquant scale
    // opParamInfo_.keyAntiquantScale.desc = context_->GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX); // TOUSE
    // opParamInfo_.keyAntiquantOffset.tensor = context_->GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX); // TOUSE: key antiquant offset
    // opParamInfo_.keyAntiquantOffset.desc = context_->GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX); // TOUSE
    // opParamInfo_.valueAntiquantScale.tensor = context_->GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX); // TOUSE: value antiquant scale
    // opParamInfo_.valueAntiquantScale.desc = context_->GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX); // TOUSE
    // opParamInfo_.valueAntiquantOffset.tensor = context_->GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX); // TOUSE: value antiquant offset
    // opParamInfo_.valueAntiquantOffset.desc = context_->GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX); // TOUSE
}

void FaInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INDEX);
    GetOptionalInputParaInfo();
}

void FaInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(ATTENTION_OUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(ATTENTION_OUT_INDEX);
    opParamInfo_.lseOut.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.lseOut.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus FaInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    // 索引0: softmax_scale (Double) - 对应 scaleValue
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);

    // 索引1: mask_mode (Int)
    opParamInfo_.maskMode = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);

    // 索引2: win_left (Int) - 对应 preToken
    opParamInfo_.preToken = attrs->GetAttrPointer<int64_t>(ATTR_WIN_LEFT_INDEX);

    // 索引3: win_right (Int) - 对应 nextToken
    opParamInfo_.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_WIN_RIGHT_INDEX);

    // 索引4: max_seqlen_q (Int)
    opParamInfo_.maxSeqlenQ = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_Q_INDEX);

    // 索引5: max_seqlen_kv (Int)
    opParamInfo_.maxSeqlenKV = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_KV_INDEX);

    // 索引6: layout_q (String)
    opParamInfo_.layOut = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);

    // 索引7: layout_kv (String)
    opParamInfo_.layoutKV = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);

    // 索引8: layout_out (String)
    opParamInfo_.layoutOut = attrs->GetStr(ATTR_LAYOUT_OUT_INDEX);

    // 索引9: return_softmax_lse (Int) - 对应 softmaxLseFlag
    auto returnLsePtr = attrs->GetAttrPointer<int64_t>(ATTR_RETURN_LSE_INDEX);
    opParamInfo_.returnLseValue = returnLsePtr;
    static const bool true_val = true;
    static const bool false_val = false;
    opParamInfo_.softmaxLseFlag = (returnLsePtr != nullptr && *returnLsePtr != 0) ? &true_val : &false_val;

    // 索引10: deterministic (Int)
    opParamInfo_.deterministic = attrs->GetAttrPointer<int64_t>(ATTR_DETERMINISTIC_INDEX);

    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetUpdateInfo()
{
    auto attrs = context_->GetAttrs();
    static int32_t SPARSE_ZERO = 0U;
    static int64_t TOKEN_MAX = 2147483647;
    if (isLegacyIfa_ && (socVersion_ == platform_ascendc::SocVersion::ASCEND310P ||
                         socVersion_ == platform_ascendc::SocVersion::ASCEND910B)) {
        opParamInfo_.sparseMode = &SPARSE_ZERO;
        opParamInfo_.preToken = &TOKEN_MAX;
        opParamInfo_.nextToken = &TOKEN_MAX;
        // opParamInfo_.actualSeqLengthsQ.tensor = nullptr; // TOUSE: TND场景需要清空
        // opParamInfo_.actualSeqLengthsQ.desc = nullptr; // TOUSE
        // opParamInfo_.queryPaddingSize.tensor = nullptr; // TOUSE: 清空query padding
        // opParamInfo_.queryPaddingSize.desc = nullptr; // TOUSE
    } else {
        // opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX); // TOUSE: 获取sparseMode属性
        // opParamInfo_.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX); // TOUSE: 获取preToken属性
        // opParamInfo_.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX); // TOUSE: 获取nextToken属性
        // opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX); // TOUSE: 获取actualSeqLengthsQ
        // opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX); // TOUSE
        // opParamInfo_.queryPaddingSize.tensor = context_->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX); // TOUSE: 获取queryPaddingSize
        // opParamInfo_.queryPaddingSize.desc = context_->GetOptionalInputDesc(QUERY_PADDING_SIZE_INDEX); // TOUSE
    }
}

void FaInfoParser::GetPreNextToken()
{
    // 从输入读取参数值
    preToken_ = opParamInfo_.preToken == nullptr ? 0 : *opParamInfo_.preToken;
    nextToken_ = opParamInfo_.nextToken == nullptr ? 0 : *opParamInfo_.nextToken;

    // 特殊场景下需要更新值
    sparseMode_ = opParamInfo_.sparseMode == nullptr ? 0 : *opParamInfo_.sparseMode;
    if (sparseMode_ == SPARSE_MODE_NO_MASK) {
        // sparse_mode=0, 带mask, 启用left padding时, preToken和nextToken参数无效
        if (attenMaskFlag_ && (qPaddingSizeFlag_ || kvPaddingSizeFlag_)) {
            preToken_ = SPARSE_MODE_INT_MAX;
            nextToken_ = SPARSE_MODE_INT_MAX;
        }
    } else if (sparseMode_ == SPARSE_MODE_ALL_MASK) {
        preToken_ = SPARSE_MODE_INT_MAX;
        nextToken_ = SPARSE_MODE_INT_MAX;
    } else if (sparseMode_ == SPARSE_MODE_LEFT_UP || sparseMode_ == SPARSE_MODE_RIGHT_DOWN) {
    //     nextToken_ = 0;
    //     preToken_ = SPARSE_MODE_INT_MAX;
    // } else if (sparseMode_ == SPARSE_MODE_TREE) {
        nextToken_ = 0;
        preToken_ = SPARSE_MODE_INT_MAX;
    }

    // 边界场景需要更新值
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

    if (opParamInfo_.sparseMode != nullptr && sparseMode_ == SPARSE_MODE_NO_MASK) {
        // if (!attenMaskFlag_) {
            // sparse mode need process attention mask when empty tensor scenes as same.
            preToken_ = SPARSE_MODE_INT_MAX;
            nextToken_ = SPARSE_MODE_INT_MAX;
        // }
    }

    if (sparseMode_ == SPARSE_MODE_NO_MASK && qPaddingSizeFlag_) {
        // For scenes with sparse mode=0 and left padding, the attention mask part is fully calculated
        preToken_ = SPARSE_MODE_INT_MAX;
        nextToken_ = SPARSE_MODE_INT_MAX;
    }
}

ge::graphStatus FaInfoParser::GetKvCache()
{
    // // 处理Key和value的TensorList场景 // TOUSE: TensorList场景支持
    // kCache_.clear(); // TOUSE
    // uint32_t keyBIdx = 0; // TOUSE
    // while ((context_->GetDynamicInputShape(KEY_INDEX, keyBIdx)) != nullptr) { // TOUSE
    //     kCache_.push_back(const_cast<gert::StorageShape *>(context_->GetDynamicInputShape(KEY_INDEX, keyBIdx))); // TOUSE
    //     keyBIdx++; // TOUSE
    // } // TOUSE
    // OP_CHECK_IF(keyBIdx == 0, // TOUSE
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "tensor list of %s is empty.", KEY_NAME.c_str()), // TOUSE
    //             return ge::GRAPH_FAILED); // TOUSE
    // kCache_.resize(keyBIdx); // TOUSE

    // vCache_.clear(); // TOUSE
    // uint32_t valueBIdx = 0; // TOUSE
    // while ((context_->GetDynamicInputShape(VALUE_INDEX, valueBIdx)) != nullptr) { // TOUSE
    //     vCache_.push_back(const_cast<gert::StorageShape *>(context_->GetDynamicInputShape(VALUE_INDEX, valueBIdx))); // TOUSE
    //     valueBIdx++; // TOUSE
    // } // TOUSE
    // vCache_.resize(valueBIdx); // TOUSE

    // OP_CHECK_IF(kCache_.size() != vCache_.size(), // TOUSE
    //             OPS_REPORT_VECTOR_INNER_ERR( // TOUSE
    //                 context_->GetNodeName(), // TOUSE
    //                 "tensor list of %s has %zu tensor, but tensor list of %s has %zu tensor, they should be equal.", // TOUSE
    //                 KEY_NAME.c_str(), kCache_.size(), VALUE_NAME.c_str(), vCache_.size()), // TOUSE
    //             return ge::GRAPH_FAILED); // TOUSE

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    // if (opParamInfo_.queryRope.desc != nullptr) {
    //     inputQRopeType_ = opParamInfo_.queryRope.desc->GetDataType();
    // }
    // if (opParamInfo_.keyRope.desc != nullptr) {
    //     inputKRopeType_ = opParamInfo_.keyRope.desc->GetDataType();
    // }
}

ge::graphStatus FaInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if ((qLayout_ == FaLayout::TND) || (qLayout_ == FaLayout::NTD)) {
        if (isMaxWorkspace_) {
            bSize_ = 1;
            return ge::GRAPH_SUCCESS;
        }
        return GetActualSeqLenQSize(bSize_);
    } else { // BSH/BSND/BNSD
        if (queryShape_->CheckHasShapeB(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        bSize_ = queryShape_->GetShapeB();
        return ge::GRAPH_SUCCESS;
    }
}

void FaInfoParser::GetQueryTSize()
{
    // 获取query的T基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    queryTSize_ = (queryShape_->HasShapeT()) ? static_cast<uint32_t>(queryShape_->GetShapeT()) : 0;
}

void FaInfoParser::GetKeyTSize()
{
    keyTSize_ = (keyShape_->HasShapeT()) ? static_cast<uint32_t>(keyShape_->GetShapeT()) : 0;
}

ge::graphStatus FaInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    if (queryShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    qkHeadDim_ = static_cast<uint32_t>(queryShape_->GetShapeD()); // 后面需要把qkHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetS1Size()
{
    // 获取S1基准值
    // BSH/BSND/BNSD: 以query的S维度为基准
    if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s1Size_ = static_cast<uint32_t>(queryShape_->GetShapeS());
    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetKvStorageMode()
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
}

void FaInfoParser::GetQuantMode()
{
    if (inputQType_ != inputKvType_) {
        quantMode_ = FaQuantMode::ANTI_QUANT;
    } else if (inputQType_ == ge::DT_BF16 || inputQType_ == ge::DT_FLOAT16) {
        quantMode_ = FaQuantMode::NO_QUANT;
    } else {
        quantMode_ = FaQuantMode::FULL_QUANT;
    }
}

ge::graphStatus FaInfoParser::GetKvLayout()
{
    // kv Layout基准值
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        kvLayout_ = qLayout_;
    } else {
        // uint32_t keyDimNum = kCache_[0]->GetStorageShape().GetDimNum();
        uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
        if (keyDimNum == 3U) {
            kvLayout_ = FaLayout::BnBsH;
        } else if (keyDimNum == 4U) {
            kvLayout_ = FaLayout::BnNBsD;
        } else if (keyDimNum == 5U) {
            kvLayout_ = FaLayout::NZ;
        } else {
            OP_LOGE(opName_,
                    "when Page Attention enabled, the first tensor of %s's tensor list is %u dim, only support 3/4/5.",
                    KEY_NAME.c_str(), keyDimNum);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetMaxActualSeq(const gert::Tensor *actualSeqLensTensor, FaLayout layout,
                                               int64_t &maxActualSeqLen)
{
    maxActualSeqLen = 0;
    const int64_t *actualLenData = actualSeqLensTensor->GetData<int64_t>();
    if (actualLenData != nullptr) {
        uint32_t loop = std::min(static_cast<uint32_t>(actualSeqLensTensor->GetShapeSize()), bSize_);
        for (uint32_t i = 0; i < loop; i++) {
            int64_t tmpS = 0;
            if (layout == FaLayout::TND || layout == FaLayout::NTD) {
                tmpS = (i == 0U) ? actualLenData[0] : (actualLenData[i] - actualLenData[i - 1U]);
            } else {
                tmpS = actualLenData[i];
            }
            maxActualSeqLen = std::max(maxActualSeqLen, tmpS);
        }
    }
    // if (isMaxWorkspace_) {
    //     maxActualSeqLen = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetS2SizeFromActualSeqLens()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return GetMaxActualSeq(opParamInfo_.actualSeqLengths.tensor, kvLayout_, s2Size_);
}

ge::graphStatus FaInfoParser::GetS2SizeForBatchContinuous()
{
    if ((kvLayout_ == FaLayout::TND) || (kvLayout_ == FaLayout::NTD)) {
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

ge::graphStatus FaInfoParser::GetS2SizeForTensorList()
{
    if ((kvLayout_ == FaLayout::TND) || (kvLayout_ == FaLayout::NTD)) {
        return GetS2SizeFromActualSeqLens();
    } else { // BSH/BSND/BNSD
        if (keyShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        s2Size_ = 0;
        for (uint32_t i = 0; i < kCache_.size(); i++) {
            auto keyShape =
                std::make_shared<FaTilingShape>(kCache_[i]->GetStorageShape(), kvLayout_, KEY_NAME, opName_, n1Size_);
            if (keyShape->GetShapeS() > s2Size_) {
                s2Size_ = keyShape->GetShapeS();
            }
            if (keyShape->GetShapeS() != keyShape_->GetShapeS()) {
                isSameSeqAllKVTensor_ = false;
            }
            kvListSeqLens_.push_back(keyShape->GetShapeS());
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetMaxBlockNumPerBatch()
{
    if (antiQuantFlag_) {
        uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
        if (dimNum != 2U) {
            OP_LOGE(opName_, "the dim num of %s is %u, it should be 2.", BLOCK_TABLE_NAME.c_str(), dimNum);
            return ge::GRAPH_FAILED;
        }
        maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    } else {
        const gert::Tensor* actSeqLenKV = opParamInfo_.actualSeqLengths.tensor;
        int32_t actualSeqKVPerBatch = 0;
        int32_t blockNumPerBatch = 0;
        int64_t blockNumValid = 0;
        maxBlockNumPerBatch_ = 0;
        // if (actSeqLenKV->GetData<int64_t>() == nullptr) {
        //     return ge::GRAPH_SUCCESS;
        // }
        for (uint32_t i = 0; i < bSize_; i++) {
            actualSeqKVPerBatch = (actSeqLenKV->GetShapeSize() > 1) ?
                                      static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[i]) :
                                      static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[0]);
            blockNumPerBatch = (actualSeqKVPerBatch + blockSize_ - 1) / blockSize_;
            blockNumValid += blockNumPerBatch;
            if (blockNumPerBatch > maxBlockNumPerBatch_) {
                maxBlockNumPerBatch_ = blockNumPerBatch;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetBlockSize()
{
//     if (opParamInfo_.blockSize == nullptr) {
//         OP_LOGE(opName_, "Attr %s not exist", BLOCK_SIZE_NAME.c_str());
//         return ge::GRAPH_FAILED;
//     }
//     blockSize_ = *(opParamInfo_.blockSize);
//     return ge::GRAPH_SUCCESS;
// }

// ge::graphStatus FaInfoParser::GetBlockTableDim2()
// {
//     uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
//     if (dimNum != 2U) {
//         OP_LOGE(opName_, "the dim num of %s is %u, it should be 2.", BLOCK_TABLE_NAME.c_str(), dimNum);
//         return ge::GRAPH_FAILED;
//     }
//     blockTableDim2_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetS2SizeForPageAttention()
{
    // if (GetBlockSize() != ge::GRAPH_SUCCESS || GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS ||
    //     GetBlockTableDim2() != ge::GRAPH_SUCCESS) {
    //     return ge::GRAPH_FAILED;
    // }
    if (GetBlockSize() != ge::GRAPH_SUCCESS || GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // if (isMaxWorkspace_) {
    //     s2Size_ = static_cast<int64_t>(blockTableDim2_) * blockSize_;
    //     return ge::GRAPH_SUCCESS;
    // }
    s2Size_ = static_cast<int64_t>(maxBlockNumPerBatch_) * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetS2Size()
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

ge::graphStatus FaInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    if (valueShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    vHeadDim_ = static_cast<uint32_t>(valueShape_->GetShapeD()); // 后面需要把vHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetRopeMode()
{
    // bool existSplitRopeTensor =
    //     ((opParamInfo_.queryRope.tensor != nullptr) && (opParamInfo_.queryRope.desc != nullptr));
    // if (npuArch_ == NpuArch::DAV_3510) {
    //     if (existSplitRopeTensor) {
    //         ropeMode_ = RopeMode::ROPE_SPLIT;
    //     } else if (qkHeadDim_ == 192 && vHeadDim_ == 128) {
    //         ropeMode_ = RopeMode::ROPE_COMBINE;
    //     } else {
    //         ropeMode_ = RopeMode::NO_ROPE;
    //     }
    //     return ge::GRAPH_SUCCESS;
    // }
    // if (qkHeadDim_ < vHeadDim_) {
    //     OP_LOGE(opName_, "the query's head dim(%u) should be greater than or equal to the value's head dim(%u)",
    //             qkHeadDim_, vHeadDim_);
    //     return ge::GRAPH_FAILED;
    // } else if (qkHeadDim_ > vHeadDim_) {
    //     if (existSplitRopeTensor) {
    //         OP_LOGE(opName_, "when %s exist, the query's head dim(%u) should be equal to the value's head dim(%u). ",
    //                 QUERY_ROPE_NAME.c_str(), qkHeadDim_, vHeadDim_);
    //         return ge::GRAPH_FAILED;
    //     } else {
    //         ropeMode_ = RopeMode::ROPE_COMBINE;
    //     }
    // } else {
    //     if (existSplitRopeTensor) {
    //         ropeMode_ = RopeMode::ROPE_SPLIT;
    //     } else {
    //         ropeMode_ = RopeMode::NO_ROPE;
    //     }
    // }
    // return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetMlaMode()
{
    // if (ropeMode_ == RopeMode::NO_ROPE) {
    //     mlaMode_ = MlaMode::NO_MLA;
    // } else if (ropeMode_ == RopeMode::ROPE_COMBINE) {
    //     mlaMode_ = MlaMode::ROPE_COMBINE_D128;
    // } else {
    //     if (qkHeadDim_ == 512) {
    //         mlaMode_ = MlaMode::ROPE_SPLIT_D512;
    //     } else if (qkHeadDim_ == 128) {
    //         mlaMode_ = MlaMode::ROPE_SPLIT_D128;
    //     }
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetRopeHeadDim()
{
    // if (ge::GRAPH_SUCCESS != GetRopeMode() || ge::GRAPH_SUCCESS != GetMlaMode()) {
    //     return ge::GRAPH_FAILED;
    // }
    // if (ropeMode_ == RopeMode::NO_ROPE) {
    //     ropeHeadDim_ = 0U;
    // } else if (ropeMode_ == RopeMode::ROPE_COMBINE) {
    //     ropeHeadDim_ = qkHeadDim_ - vHeadDim_;
    // } else {
    //     queryRopeShape_ = std::make_shared<FaTilingShape>(opParamInfo_.queryRope.tensor->GetStorageShape(), qLayout_,
    //                                                        QUERY_ROPE_NAME, opName_, n1Size_);
    //     if (queryRopeShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
    //         return ge::GRAPH_FAILED;
    //     }
    //     ropeHeadDim_ = static_cast<uint32_t>(queryRopeShape_->GetShapeD());
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // inputLayout: {qLayout, outLayout}
    const map<string, pair<FaLayout, FaLayout>> layoutMap = {
        {"BSH", {FaLayout::BSH, FaLayout::BSH}},         {"BSND", {FaLayout::BSND, FaLayout::BSND}},
        {"BNSD", {FaLayout::BNSD, FaLayout::BNSD}},      {"TND", {FaLayout::TND, FaLayout::TND}},
        {"BSH_NBSD", {FaLayout::BSH, FaLayout::NBSD}},   {"BSND_NBSD", {FaLayout::BSND, FaLayout::NBSD}},
        {"BNSD_NBSD", {FaLayout::BNSD, FaLayout::NBSD}}, {"TND_NTD", {FaLayout::TND, FaLayout::NTD}},
        {"NTD_TND", {FaLayout::NTD, FaLayout::TND}},     {"BNSD_BSND", {FaLayout::BNSD, FaLayout::BSND}},
        {"BSND_BNSD", {FaLayout::BSND, FaLayout::BNSD}}, {"BSH_BNSD", {FaLayout::BSH, FaLayout::BNSD}},
        {"NTD", {FaLayout::NTD, FaLayout::NTD}}};

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

ge::graphStatus FaInfoParser::GetN1Size()
{
    // 从 Q 形状获取 N1 值
    if (queryShape_ != nullptr && queryShape_->HasShapeN()) {
        n1Size_ = static_cast<uint32_t>(queryShape_->GetShapeN());
    } else {
        OP_LOGE(opName_, "Failed to get N1 size from query shape.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetN2Size()
{
    // 从 K 形状获取 N2 值
    if (keyShape_ != nullptr && keyShape_->HasShapeN()) {
        n2Size_ = static_cast<uint32_t>(keyShape_->GetShapeN());
    } else {
        OP_LOGE(opName_, "Failed to get N2 size from key shape.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::SetFaShape()
{
    queryShape_ = std::make_shared<FaTilingShape>(opParamInfo_.query.shape->GetStorageShape(), qLayout_, QUERY_NAME,
                                                   opName_, n1Size_);
    // keyShape_ = std::make_shared<FaTilingShape>(kCache_[0]->GetStorageShape(), kvLayout_, KEY_NAME, opName_, n1Size_);
    // valueShape_ =
    //     std::make_shared<FaTilingShape>(vCache_[0]->GetStorageShape(), kvLayout_, VALUE_NAME, opName_, n2Size_);
    keyShape_ = std::make_shared<FaTilingShape>(opParamInfo_.key.shape->GetStorageShape(), kvLayout_, KEY_NAME, opName_, n1Size_);
    valueShape_ =
        std::make_shared<FaTilingShape>(opParamInfo_.value.shape->GetStorageShape(), kvLayout_, VALUE_NAME, opName_, n2Size_);
}

ge::graphStatus FaInfoParser::GetGSize()
{
    // 获取G基准值
    if (n1Size_ % n2Size_ != 0U) {
        OP_LOGE(opName_, "%s(%u) should be a multiple of %s(%u).", QUERY_HEADS_NUM_NAME.c_str(), n1Size_,
                KV_HEADS_NUM_NAME.c_str(), n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

// ge::graphStatus FaInfoParser::GetAttenMaskSparse9Info()
// {
//     // sparse9时，mask形状：TND时传入∑s1²，其余场景传入[B,S1,S1]
//     auto *maskTensor = opParamInfo_.attenMask.tensor;
//     uint32_t maskDimNum = maskTensor->GetStorageShape().GetDimNum();

//     // TND传入的mask
//     if (qLayout_ == FaLayout::TND || qLayout_ == FaLayout::NTD) {
//         if (maskDimNum == 1U) {
//             attenMaskBatchStride_ = 1;
//             attenMaskStride_ = 1;
//         } else {
//             OP_LOGE(opName_, "When layout is TND/NTD, Tree mask(%u) matrix dim only support 1.",
//                     *opParamInfo_.sparseMode);
//         }
//     } else {
//         if (maskDimNum == 3U) {
//             attenMaskBatchStride_ = maskTensor->GetStorageShape().GetDim(maskDimNum - 1) * \
//                                     maskTensor->GetStorageShape().GetDim(maskDimNum - 2);
//             attenMaskStride_ = maskTensor->GetStorageShape().GetDim(maskTensor->GetStorageShape().GetDimNum() - 1);
//         } else {
//             OP_LOGE(opName_, "When layout is not TND/NTD, Tree mask(%u) matrix dim only support 3.",
//                     *opParamInfo_.sparseMode);
//         }
//     }
//     return ge::GRAPH_SUCCESS;
// }

ge::graphStatus FaInfoParser::GetAntiQuantInfo()
{
    // antiQuantFlag_ = inputQType_ != inputKvType_;
    // if (opParamInfo_.keyAntiquantScale.tensor == nullptr || opParamInfo_.valueAntiquantScale.tensor == nullptr) {
    //     return ge::GRAPH_SUCCESS;
    // }
    // if (!antiQuantFlag_) {
    //     return ge::GRAPH_SUCCESS;
    // }
    // bool kPerChnVPerTokFlag_ =
    //      (inputKvType_ == ge::DT_INT4 || inputKvType_ == ge::DT_INT8) && 
    //      (*(opParamInfo_.keyAntiquantMode) == 0 && *(opParamInfo_.valueAntiquantMode) == 1);
    
    // auto tmpAntiquantMode = *(opParamInfo_.keyAntiquantMode);
    // auto tmpAntiquant = opParamInfo_.keyAntiquantScale.tensor->GetStorageShape();
    // if (opParamInfo_.keyAntiquantOffset.tensor != nullptr) {
    //     tmpAntiquant = opParamInfo_.keyAntiquantOffset.tensor->GetStorageShape();
    // }

    // if (kPerChnVPerTokFlag_) {
    //     tmpAntiquantMode = *(opParamInfo_.valueAntiquantMode);
    //     tmpAntiquant = opParamInfo_.valueAntiquantScale.tensor->GetStorageShape();
    //     if (opParamInfo_.valueAntiquantOffset.tensor != nullptr) {
    //         tmpAntiquant = opParamInfo_.valueAntiquantOffset.tensor->GetStorageShape();
    //     }
    // }

    // if (tmpAntiquantMode == 6) {
    //     if (systemPrefixFlag_) {
    //         if (tmpAntiquant.GetDimNum() != 5) {
    //             OP_LOGE(opName_, "The dimension(%lu) of antiquant is illegal, it should be 5 when per-token-group mode.", tmpAntiquant.GetDimNum());
    //         }
    //         antiquantParaSeqSize_ = tmpAntiquant.GetDim(3);
    //     }
    // } else if (tmpAntiquantMode == 1) {
    //     if (tmpAntiquant.GetDimNum() != 2 && tmpAntiquant.GetDimNum() != 3) {
    //         OP_LOGE(opName_, "The dimension(%lu) of antiquant is illegal, it should be 2 or 3 when per-token mode.", tmpAntiquant.GetDimNum());
    //     }
    //     antiquantParaSeqSize_ = tmpAntiquant.GetDimNum() == 3U ? tmpAntiquant.GetDim(2) : tmpAntiquant.GetDim(1);
    // } else if (tmpAntiquantMode == 3) {
    //     if (tmpAntiquant.GetDimNum() != 3) {
    //         OP_LOGE(opName_, "The dimension(%lu) of antiquant is illegal, it should be 3 when per-token-head mode.", tmpAntiquant.GetDimNum());
    //     }
    //     antiquantParaSeqSize_ = tmpAntiquant.GetDim(2);
    // }
    // return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetAttenMaskInfo()
{
    // // only bss & b1ss & bs need to calc attenMaskSize_ , attenMaskSize_ is uesed to calc batch offset
    // if (attenMaskFlag_) {
    //     if (*opParamInfo_.sparseMode == 9U) {
    //         return GetAttenMaskSparse9Info();
    //     }
    //     auto *maskTensor = opParamInfo_.attenMask.tensor;
    //     uint32_t maskDimNum = maskTensor->GetStorageShape().GetDimNum();
    //     if (maskDimNum == 2U) {
    //         if (s1Size_ == 1U) { // qs=1 仅支持BS
    //             attenMaskBatchStride_ = maskTensor->GetStorageShape().GetDim(1);
    //         } else { // qs > 1 仅支持SS
    //             attenMaskBatchStride_ = 0;
    //         }
    //     } else if (maskDimNum == 3U || maskDimNum == 4U) {
    //         if (maskTensor->GetStorageShape().GetDim(0) == bSize_) { // BSS B1SS BatchStride = S1*S2
    //             attenMaskBatchStride_ = maskTensor->GetStorageShape().GetDim(maskDimNum - 1) *
    //                                     maskTensor->GetStorageShape().GetDim(maskDimNum - 2);
    //         } else { // 1SS 11SS
    //             attenMaskBatchStride_ = 0;
    //         }
    //     } else {
    //         OP_LOGE(opName_, "mask matrix dim only support 2/3/4.");
    //     }
    //     if (*opParamInfo_.sparseMode == 0U || *opParamInfo_.sparseMode == 1U) {
    //         attenMaskStride_ = maskTensor->GetStorageShape().GetDim(maskTensor->GetStorageShape().GetDimNum() - 1);
    //     } else {
    //         attenMaskStride_ = 2048U; // compress mask
    //     }
    // }
    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetPaddingSizeFlag()
{
    // qPaddingSizeFlag_ = ((!isLegacyIfa_ || npuArch_ == NpuArch::DAV_3510) &&
    //             (opParamInfo_.queryPaddingSize.tensor != nullptr));

    // if (isLegacyIfa_) {
    //     if ((opParamInfo_.kvPaddingSize.tensor != nullptr &&
    //          opParamInfo_.kvPaddingSize.tensor->GetStorageShape().GetShapeSize() > 0) &&
    //         kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS && opParamInfo_.actualSeqLengths.tensor != nullptr) {
    //         kvPaddingSizeFlag_ = true;
    //     }
    // } else {
    //     if (opParamInfo_.kvPaddingSize.tensor != nullptr) {
    //         kvPaddingSizeFlag_ = true;
    //     }
    // }

    // if (antiQuantFlag_) {
    //     if (qPaddingSizeFlag_ || (kvPaddingSizeFlag_ && s1Size_ == 1)) {
    //         needInit_ = true;
    //     } 
    // } else {
    //     if (qPaddingSizeFlag_ || kvPaddingSizeFlag_) {
    //         needInit_ = true;
    //     }
    // }
}

void FaInfoParser::GetMaskFlag()
{
    // auto *maskTensor = opParamInfo_.attenMask.tensor;
    // attenMaskFlag_ = (maskTensor != nullptr) && (maskTensor->GetStorageShape().GetShapeSize() != 0);
}

ge::graphStatus FaInfoParser::GetPseShiftFlag()
{
    // const gert::StorageShape *pseShiftShape = context_->GetOptionalInputShape(PSE_SHIFT_INDEX);
    // if (opParamInfo_.pseType != nullptr) {
    //     pseType_ = *opParamInfo_.pseType;
    //     OP_CHECK_IF((pseType_ != static_cast<int64_t>(IfaPseType::PSE_OUTER_MUL_ADD_TYPE)) &&
    //                 (pseType_ != static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_TYPE)) &&
    //                 (pseType_ != static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_SQRT_TYPE)),
    //                 OP_LOGE(opName_, "PseType(%ld) is not support, pseType must be 0/2/3.", pseType_),
    //                 return ge::GRAPH_FAILED);
    // }
    // if (pseType_ == static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_TYPE) ||
    //     pseType_ == static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_SQRT_TYPE)){
    //     enableAlibiPse_ = true;
    // }
    // if ((pseShiftShape == nullptr) ||
    //     ((pseShiftShape != nullptr) && (pseShiftShape->GetStorageShape().GetShapeSize() == 0))) {
    //     pseShiftFlag_ = false;
    // } else if (pseType_ == static_cast<int64_t>(IfaPseType::PSE_OUTER_MUL_ADD_TYPE)){
    //     pseShiftFlag_ = true;
    //     pseShiftByBatch_ = pseShiftShape->GetStorageShape().GetDim(0);
    //     if (socVersion_ == platform_ascendc::SocVersion::ASCEND910B) {
    //         pseShiftByBatch_ = pseShiftShape->GetStorageShape().GetDim(0) != 1U ? 1 : 0;
    //     }
    //     pseShiftS1_ = pseShiftShape->GetStorageShape().GetDim(PSE_SHIFT_S1_INDEX);
    //     pseShiftS2_ = pseShiftShape->GetStorageShape().GetDim(PSE_SHIFT_S2_INDEX);
    // } else if (pseType_ == static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_TYPE) ||
    //     pseType_ == static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_SQRT_TYPE)){
    //     if (antiQuantFlag_) {
    //         pseShiftByBatch_ = pseShiftShape->GetStorageShape().GetDimNum() != 0 ? 1 : 0;
    //     } else {
    //         pseShiftByBatch_ = pseShiftShape->GetStorageShape().GetDimNum() == 1 ? 1 : 0;
    //     }
    // }
    // return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetSystemPrefix()
{
    // if (opParamInfo_.keySharedPrefix.tensor != nullptr && opParamInfo_.valueSharedPrefix.tensor != nullptr) {
    //     keyPrefixShape_ = std::make_shared<FaTilingShape>(opParamInfo_.keySharedPrefix.tensor->GetStorageShape(),
    //                                                        kvLayout_, KEY_SHARED_PREFIX_NAME, opName_, n2Size_);
    //     systemPrefixMaxLen_ = keyPrefixShape_->GetShapeS();
    //     systemPrefixFlag_ = true;
    //     systemPrefixLen_ = systemPrefixMaxLen_;
    //     if (opParamInfo_.actualSharedPrefixLen.tensor != nullptr &&
    //         opParamInfo_.actualSharedPrefixLen.tensor->GetStorageShape().GetShapeSize() != 0) {
    //         gert::Shape prefixShape{1};
    //         if (prefixShape != opParamInfo_.actualSharedPrefixLen.tensor->GetStorageShape()) {
    //             OP_LOGE(opName_, "System prefix is enabled but actualSharedPrefixLen shape is not {1}");
    //             return ge::GRAPH_FAILED;
    //         }
    //         if (opParamInfo_.actualSharedPrefixLen.tensor->GetData<int64_t>() != nullptr) {
    //             systemPrefixLen_ = opParamInfo_.actualSharedPrefixLen.tensor->GetData<int64_t>()[0];
    //         }
    //     }
    // }
    // return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetActualSeqInfo()
{
    maxActualseq_ = 0;
    if (isMaxWorkspace_) {
        maxActualseq_ = s2Size_;
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDims_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
        if ((kvLayout_ == FaLayout::TND) || (kvLayout_ == FaLayout::NTD)) {
            isAccumKVSeq_ = true;
        }
        if (opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() != nullptr) {
            const int64_t *actualLenData = opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>();
            uint32_t loop = ((actualLenDims_ == 1) && (kvListSeqLens_.size() == 1)) ? 1 : bSize_;
            loop = std::min(loop, actualLenDims_);
            for (uint32_t i = 0; i < loop; i++) {
                int64_t actLen = actualLenData[i];
                // if (antiQuantFlag_) {
                //     if (qLayout_ == FaLayout::TND && i > 0 && kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
                //         actLen -= actualLenData[i - 1];
                //     }
                //     if (s1Size_ == 1) {
                //         needInit_ = (actLen == 0);
                //     }
                // } else {
                //     needInit_ = (actLen == 0);
                //     if (qLayout_ == FaLayout::TND || qLayout_ == FaLayout::NTD) {
                //         continue;
                //     }
                //     if (actLen != s2Size_) {
                //         needInit_ = 1;
                //     }

                needInit_ = (actLen == 0);
                if (quantMode_ != FaQuantMode::ANTI_QUANT && (qLayout_ == FaLayout::TND || qLayout_ == FaLayout::NTD)) {
                    continue;
                }

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
        if ((qLayout_ == FaLayout::TND) || (qLayout_ == FaLayout::NTD)) {
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

        if ((quantMode_ == FaQuantMode::ANTI_QUANT) && (s1Size_ == 1)) {
            preToken_ = SPARSE_MODE_INT_MAX;
            nextToken_ = SPARSE_MODE_INT_MAX;
            if (qLayout_ != FaLayout::TND) {
                actualLenQDims_ = 0;
            }
            qPaddingSizeFlag_ = false;

            needInit_ = (actualLenQDims_ == 0); // add
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetPostQuantInfo()
{
    // ge::DataType outputType = opParamInfo_.attenOut.desc->GetDataType();
    // if (outputType != ge::DT_BF16 && outputType != ge::DT_FLOAT16) {
    //     isPostQuantEnable_ = true;
    //     if (opParamInfo_.quantScale2.tensor != nullptr && opParamInfo_.quantScale2.desc != nullptr) {
    //         isOutQuantPerChnOut_ = opParamInfo_.quantScale2.tensor->GetStorageShape().GetShapeSize() != 1;
    //         isOutQuantTypeBf16_ = opParamInfo_.quantScale2.desc->GetDataType() == DT_BF16;
    //     }
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetFullQuantMode()
{
    // if (quantMode_ == FaQuantMode::FULL_QUANT) {
    //     if (opParamInfo_.deqScale1.tensor != nullptr && opParamInfo_.deqScale1.desc != nullptr) {
    //         fullQuantMode_ = FaFullQuantMode::PER_TENSOR_FULL_QUANT;
    //     } else {
    //         fullQuantMode_ = FaFullQuantMode::PER_BLOCK_FULL_QUANT;
    //     }
    // }
    return ge::GRAPH_SUCCESS;
}

TilingKeyLayout FaInfoParser::MapStringToLayout(FaLayout &layoutString) const
{
    const std::map<FaLayout, TilingKeyLayout> layoutMap = {
        {FaLayout::BSH, TilingKeyLayout::BSH_BSND}, {FaLayout::BSND, TilingKeyLayout::BSH_BSND},
        {FaLayout::BNSD, TilingKeyLayout::BNSD},    {FaLayout::NZ, TilingKeyLayout::NZ},
        {FaLayout::TND, TilingKeyLayout::TND},      {FaLayout::NBSD, TilingKeyLayout::NBSD},
        {FaLayout::NTD, TilingKeyLayout::NTD},      {FaLayout::BnBsH, TilingKeyLayout::BSH_BSND},
        {FaLayout::BnNBsD, TilingKeyLayout::BNSD},
    };

    auto it = layoutMap.find(layoutString);
    if (it != layoutMap.end()) {
        return it->second;
    }
    return TilingKeyLayout::BSH_BSND;
}

void FaInfoParser::GenerateFeatureInfo(FaTilingInfo &faInfo)
{
    // empty tensor
    faInfo.emptyTensorFlag = emptyTensorFlag_;

    // pa
    faInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    faInfo.blockSize = blockSize_;
    faInfo.blockTypeSize = sizeof(float);

    // // inner precise
    // faInfo.innerPrecise = *opParamInfo_.innerPrecise;

    //pse shift
    faInfo.pseShiftFlag = pseShiftFlag_;
    faInfo.pseShiftByBatch = pseShiftByBatch_;
    faInfo.pseShiftS1 = pseShiftS1_;
    faInfo.pseShiftS2 = pseShiftS2_;
    faInfo.enableAlibiPse = enableAlibiPse_;

    // atten mask
    faInfo.attenMaskFlag = attenMaskFlag_;
    faInfo.attenMaskBatchStride = attenMaskBatchStride_;
    faInfo.attenMaskStride = attenMaskStride_;
    faInfo.sparseMode = sparseMode_;
    // // 4: only mla noquant & band mode suppport slidingFlag
    // faInfo.slidingFlag =
    //     (*opParamInfo_.sparseMode == 4) && (ropeMode_ == RopeMode::ROPE_SPLIT) && (qkHeadDim_ == 512U);
    faInfo.qPaddingSizeFlag = qPaddingSizeFlag_;
    faInfo.kvPaddingSizeFlag = kvPaddingSizeFlag_;
    faInfo.softmaxLseFlag = *opParamInfo_.softmaxLseFlag;
    faInfo.totalLseSize =
        (opParamInfo_.lseOut.shape == nullptr) ? 0 : opParamInfo_.lseOut.shape->GetStorageShape().GetShapeSize();
    faInfo.isMaxWorkspace = isMaxWorkspace_;
    faInfo.isLegacyIfa = isLegacyIfa_;
    faInfo.preToken = preToken_;
    faInfo.nextToken = nextToken_;
    // faInfo.learnableSinkFlag = (opParamInfo_.learnableSink.tensor != nullptr);
    faInfo.antiQuantFlag = antiQuantFlag_;
    faInfo.antiqSeqSize = antiquantParaSeqSize_;
    // faInfo.keyAntiquantMode = static_cast<uint32_t>(*opParamInfo_.keyAntiquantMode);
    // faInfo.valueAntiquantMode = static_cast<uint32_t>(*opParamInfo_.valueAntiquantMode);

    faInfo.isQKVDDifferent = (qkHeadDim_ != vHeadDim_) && !(qkHeadDim_ == 192U && vHeadDim_ == 128U);

    faInfo.sysPrefixFlag = systemPrefixFlag_;
    faInfo.systemPrefixLen = systemPrefixLen_;
    faInfo.systemPrefixMaxLen = systemPrefixMaxLen_;
    // postquant
    faInfo.isOutQuantPerChnOut = isOutQuantPerChnOut_;
    faInfo.isOutQuantTypeBf16 = isOutQuantTypeBf16_;
    faInfo.isOutQuantEnable = isPostQuantEnable_;
}

void FaInfoParser::GenerateLayoutInfo(FaTilingInfo &faInfo)
{
    faInfo.qLayout = qLayout_;
    faInfo.kvLayout = kvLayout_;
    faInfo.outLayout = outLayout_;
    faInfo.inputKvLayout = MapStringToLayout(kvLayout_);
    faInfo.inputLayout = MapStringToLayout(qLayout_);
    faInfo.outputLayout = MapStringToLayout(outLayout_);
}

void FaInfoParser::GenerateInfo(FaTilingInfo &faInfo)
{
    faInfo.opName = opName_;
    faInfo.platformInfo = platformInfo_;
    faInfo.opParamInfo = opParamInfo_;
    faInfo.socVersion = socVersion_;
    GenerateAxisInfo(faInfo);
    GenerateDtypeInfo(faInfo);
    faInfo.kvStorageMode = kvStorageMode_;
    faInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS);
    faInfo.ropeMode = ropeMode_;
    faInfo.mlaMode = mlaMode_;
    faInfo.quantMode = quantMode_;
    faInfo.fullQuantMode = fullQuantMode_;
    faInfo.l2CacheSize = l2CacheSize_;

    // faInfo.kCache = kCache_;
    // faInfo.vCache = vCache_;

    faInfo.totalOutputSize = opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize();

    faInfo.l2CacheOffFlag = false;
    // faInfo.totalBlockNum = kCache_[0]->GetStorageShape().GetDim(0);
    faInfo.scaleValue = *opParamInfo_.softmaxScale;
    faInfo.needInit = needInit_;
    faInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;
    // if (faInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
    //     faInfo.maxBlockNumPerBatch = blockTableDim2_;
    // }

    faInfo.actualLenQDims = actualLenQDims_;
    faInfo.actualLenDims = actualLenDims_;
    faInfo.maxActualseq = maxActualseq_;
    faInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);
    faInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    faInfo.isSameActualseq = isSameActualseq_;
    faInfo.kvListSeqLens = kvListSeqLens_;

    faInfo.isAccumQSeq = isAccumQSeq_;
    faInfo.isAccumKVSeq = isAccumKVSeq_;

    GenerateFeatureInfo(faInfo);
    GenerateLayoutInfo(faInfo);
}

void FaInfoParser::GenerateAxisInfo(FaTilingInfo &faInfo)
{
    faInfo.bSize = bSize_;
    faInfo.n1Size = n1Size_;
    faInfo.n2Size = n2Size_;
    faInfo.s1Size = s1Size_;
    faInfo.s2Size = s2Size_;
    faInfo.gSize = gSize_;
    faInfo.qkHeadDim = qkHeadDim_;
    faInfo.vHeadDim = vHeadDim_;
    faInfo.ropeHeadDim = ropeHeadDim_;
    faInfo.qTSize = queryTSize_;
    faInfo.kTSize = keyTSize_;
}

void FaInfoParser::GenerateDtypeInfo(FaTilingInfo &faInfo)
{
    faInfo.inputQType = inputQType_;
    faInfo.inputKvType = inputKvType_;
    faInfo.inputQRopeType = inputQRopeType_;
    faInfo.inputKRopeType = inputKRopeType_;
    faInfo.outputType = outputType_;
}

ge::graphStatus FaInfoParser::Parse(FaTilingInfo &faInfo)
{
    OP_LOGI(faInfo.opName, "enter FaInfoParser::Parse!");
    if (context_ == nullptr) {
        OP_LOGE(faInfo.opName, "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != GetKvCache() || ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }
    GetInOutDataType();
    GetKvStorageMode();
    GetQuantMode();
    if (ge::GRAPH_SUCCESS != GetQueryAndOutLayout() || ge::GRAPH_SUCCESS != GetKvLayout()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseAxisInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseFeatureInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(faInfo);
    OP_LOGI(faInfo.opName, "end FaInfoParser::Parse!");

    OP_LOGI(faInfo.opName, "b size is %d \n", faInfo.bSize);
    OP_LOGI(faInfo.opName, "n1 size is %d \n", faInfo.n1Size);
    OP_LOGI(faInfo.opName, "n2 size is %d \n", faInfo.n2Size);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::ParseAxisInfo()
{
    SetFaShape();
    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size()) {
        return ge::GRAPH_FAILED;
    }

    GetQueryTSize();

    if (ge::GRAPH_SUCCESS != GetQkHeadDim() || ge::GRAPH_SUCCESS != GetValueHeadDim() || 
        ge::GRAPH_SUCCESS != GetLegacyIfaFlag()) {
        return ge::GRAPH_FAILED;
    }

    // GetUpdateInfo();

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size()) {
        return ge::GRAPH_FAILED;
    }

    // if (emptyTensorFlag_) {
    //     return ge::GRAPH_SUCCESS;
    // }

    GetKeyTSize();
    
    if (ge::GRAPH_SUCCESS != GetGSize() || ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::ParseFeatureInfo()
{
    // GetMaskFlag();
    // GetPaddingSizeFlag();
    // GetPreNextToken();
    // if (ge::GRAPH_SUCCESS != GetAttenMaskInfo() || ge::GRAPH_SUCCESS != GetActualSeqInfo() ||
    //     ge::GRAPH_SUCCESS != GetSystemPrefix() || ge::GRAPH_SUCCESS != GetPseShiftFlag() ||
    //     ge::GRAPH_SUCCESS != GetPostQuantInfo() || ge::GRAPH_SUCCESS != GetFullQuantMode()) {
    //     return ge::GRAPH_FAILED;
    // }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling