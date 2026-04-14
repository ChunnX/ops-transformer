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

ge::graphStatus FaInfoParser::CheckRequiredAttrExistence() const // TODO 确认是否需要，确认可选函数
{
    OP_CHECK_IF(opParamInfo_.softmaxScale == nullptr, OP_LOGE(opName_, "attr softmaxScale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.maskMode == nullptr, OP_LOGE(opName_, "attr maskMode is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.maxSeqlenQ == nullptr, OP_LOGE(opName_, "attr maxSeqlenQ is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.maxSeqlenKV == nullptr, OP_LOGE(opName_, "attr maxSeqlenKV is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutQ == nullptr, OP_LOGE(opName_, "attr layoutQ is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr, OP_LOGE(opName_, "attr layoutKV is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutOut == nullptr, OP_LOGE(opName_, "attr layoutOut is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.returnSoftMaxLse == nullptr, OP_LOGE(opName_, "attr returnSoftMaxLse is nullptr"),
                return ge::GRAPH_FAILED);

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
    emptyTensorFlag_ = true;
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus FaInfoParser::GetActualSeqLenQSize(uint32_t &size) // TODO TND场景 获取cu
{
    // if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
    //     OP_LOGE(opName_, "when %s's layout is %s, %s must be provided.", QUERY_NAME.c_str(),
    //             LayoutToSerialString(layoutQ).c_str(), ACTUAL_SEQ_Q_LEN_NAME.c_str());
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
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910B) && (npuArch_ != NpuArch::DAV_3510) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_55)) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d]/NpuArch[%d] is not support.",
                                    static_cast<int32_t>(socVersion_), static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetOptionalInputParaInfo()
{
    GetOptionalInputParaActualSeqLengthInfo();
    GetOptionalInputParaPageAttentionInfo();
    GetOptionalInputParaPostQuantInfo();
    GetOptionalInputParaRopeInfo();
    GetOptionalInputParaPseInfo();
    GetOptionalInputParaLeftPaddingInfo();
    GetOptionalInputParaPrefixInfo();
    GetOptionalInputParaSinksInfo();
    GetOptionalInputParaQuantInfo();
}
void FaInfoParser::GetOptionalInputParaMaskInfo()
{
    // opParamInfo_.attenMask.tensor = context_->GetOptionalInputTensor(ATTEN_MASK_INDEX); // TOUSE: 获取attention mask tensor
    // opParamInfo_.attenMask.desc = context_->GetOptionalInputDesc(ATTEN_MASK_INDEX); // TOUSE: 获取attention mask desc
}

void FaInfoParser::GetOptionalInputParaActualSeqLengthInfo()
{
    opParamInfo_.cuSeqlensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensKv.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_KV_INDEX);
    opParamInfo_.cuSeqlensKv.desc = context_->GetOptionalInputDesc(CU_SEQLENS_KV_INDEX);
    opParamInfo_.sequsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedKv.tensor = context_->GetOptionalInputTensor(SEQUSED_KV_INDEX);
    opParamInfo_.sequsedKv.desc = context_->GetOptionalInputDesc(SEQUSED_KV_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = opParamInfo_.cuSeqlensQ.tensor;
    opParamInfo_.actualSeqLengthsQ.desc = opParamInfo_.cuSeqlensQ.desc;
    opParamInfo_.actualSeqLengths.tensor = opParamInfo_.cuSeqlensKv.tensor;
    opParamInfo_.actualSeqLengths.desc = opParamInfo_.cuSeqlensKv.desc;
}

void FaInfoParser::GetOptionalInputParaPageAttentionInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void FaInfoParser::GetOptionalInputParaSinksInfo()
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

void FaInfoParser::GetOptionalInputParaSinksInfo()
{
    opParamInfo_.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INDEX);
    opParamInfo_.sinks.desc = context_->GetOptionalInputDesc(SINKS_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
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
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX); // TODO名字对应

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
    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);

    // 索引7: layout_kv (String)
    opParamInfo_.layoutKV = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);

    // 索引8: layout_out (String)
    opParamInfo_.layoutOut = attrs->GetStr(ATTR_LAYOUT_OUT_INDEX);

    // 索引9: return_softmax_lse (Int) - 对应 softmaxLseFlag
    auto returnLsePtr = attrs->GetAttrPointer<int64_t>(ATTR_RETURN_LSE_INDEX);
    opParamInfo_.returnSoftMaxLse = returnLsePtr;
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
    static int64_t MASK_ZERO = 0U;
    static int64_t TOKEN_MAX = 2147483647;
    if (isLegacyIfa_ && (socVersion_ == platform_ascendc::SocVersion::ASCEND310P ||
                         socVersion_ == platform_ascendc::SocVersion::ASCEND910B)) {
        opParamInfo_.maskMode = &MASK_ZERO;
        opParamInfo_.preToken = &TOKEN_MAX;
        opParamInfo_.nextToken = &TOKEN_MAX;
        // opParamInfo_.actualSeqLengthsQ.tensor = nullptr; // TOUSE: TND场景需要清空
        // opParamInfo_.actualSeqLengthsQ.desc = nullptr; // TOUSE
        // opParamInfo_.queryPaddingSize.tensor = nullptr; // TOUSE: 清空query padding
        // opParamInfo_.queryPaddingSize.desc = nullptr; // TOUSE
    } else {
        // opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX); // TOUSE:
        // 获取sparseMode属性 opParamInfo_.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX); // TOUSE:
        // 获取preToken属性 opParamInfo_.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX); // TOUSE:
        // 获取nextToken属性 opParamInfo_.actualSeqLengthsQ.tensor =
        // context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX); // TOUSE: 获取actualSeqLengthsQ
        // opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX); // TOUSE
        // opParamInfo_.queryPaddingSize.tensor = context_->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX); // TOUSE:
        // 获取queryPaddingSize opParamInfo_.queryPaddingSize.desc =
        // context_->GetOptionalInputDesc(QUERY_PADDING_SIZE_INDEX); // TOUSE
    }
}

void FaInfoParser::GetPreNextToken()
{
    // 从输入读取参数值
    winLeft_ = opParamInfo_.preToken == nullptr ? 0 : *opParamInfo_.preToken;
    winRight_ = opParamInfo_.nextToken == nullptr ? 0 : *opParamInfo_.nextToken;

    // 特殊场景下需要更新值
    maskMode_ = opParamInfo_.maskMode == nullptr ? 0 : *opParamInfo_.maskMode;
    if (maskMode_ == MASK_MODE_ALL_MASK) {
            winLeft_ = MASK_MODE_INT_MAX;
            winRight_ = MASK_MODE_INT_MAX;
    } else if (maskMode_ == MASK_MODE_RIGHT_DOWN) {
        winRight_ = 0;
        winLeft_ = MASK_MODE_INT_MAX;
    }

    // 边界场景需要更新值
    if (winLeft_ == -1) {
        winLeft_ = MASK_MODE_INT_MAX;
    }
    if (winRight_ == -1) {
        winRight_ = MASK_MODE_INT_MAX;
    }
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
}

ge::graphStatus FaInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (layoutQ_ == FaLayout::TND) {
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

ge::graphStatus FaInfoParser::GetQkHeadDim() // TODO check/get函数区别在哪
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    if (queryShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    qkHeadDim_ = static_cast<uint32_t>(queryShape_->GetShapeD()); // 后面需要把qkHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetS1Size() // TODO TND场景补充
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
    if (opParamInfo_.blockTable.tensor != nullptr) {
        kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS; // TODO命名是否需要修改 相关
    }
}

ge::graphStatus FaInfoParser::GetKvLayout()  // 删除导致头数不对
{
    // kv Layout基准值
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        layoutKV_ = layoutQ_;
    } else {
        // uint32_t keyDimNum = kCache_[0]->GetStorageShape().GetDimNum();
        uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
        if (keyDimNum == 3U) {
            layoutKV_ = FaLayout::BnBsH;
        } else if (keyDimNum == 4U) {
            layoutKV_ = FaLayout::BnNBsD;
        } else if (keyDimNum == 5U) {
            layoutKV_ = FaLayout::NZ;
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
            if (layout == FaLayout::TND) {
                tmpS = (i == 0U) ? actualLenData[0] : (actualLenData[i] - actualLenData[i - 1U]);
            } else {
                tmpS = actualLenData[i];
            }
            maxActualSeqLen = std::max(maxActualSeqLen, tmpS);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetS2SizeFromActualSeqLens()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return GetMaxActualSeq(opParamInfo_.actualSeqLengths.tensor, layoutKV_, s2Size_);
}

ge::graphStatus FaInfoParser::GetS2SizeForBatchContinuous()
{
    if ((layoutKV_ == FaLayout::TND)) {
        return GetS2SizeFromActualSeqLens(); // TODO 直接获取入参，max_seqlen TND场景，未传入用T(非必传)
    } else {                                 // BSH/BSND/BNSD
        if (keyShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s2Size_ = keyShape_->GetShapeS();
        kvListSeqLens_.push_back(s2Size_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetMaxBlockNumPerBatch() // TODO PA场景S2 size 如何获取
{
    const gert::Tensor *actSeqLenKV = opParamInfo_.actualSeqLengths.tensor;
    int32_t actualSeqKVPerBatch = 0;
    int32_t blockNumPerBatch = 0;
    int64_t blockNumValid = 0;
    maxBlockNumPerBatch_ = 0;
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

ge::graphStatus FaInfoParser::GetQueryAndOutLayout()
{
    const std::vector<std::tuple<FaLayout, FaLayout, FaLayout>> layoutSupported = {
        // TODO是否保留
        {FaLayout::BNSD, FaLayout::BNSD, FaLayout::BNSD},    {FaLayout::BNSD, FaLayout::BNSD, FaLayout::BSND},
        {FaLayout::BNSD, FaLayout::PA_BBND, FaLayout::BNSD}, {FaLayout::BNSD, FaLayout::PA_BBND, FaLayout::BSND},
        {FaLayout::BNSD, FaLayout::PA_BNBD, FaLayout::BNSD}, {FaLayout::BNSD, FaLayout::PA_BNBD, FaLayout::BSND},
        {FaLayout::BNSD, FaLayout::PA_Nz, FaLayout::BNSD},   {FaLayout::BNSD, FaLayout::PA_Nz, FaLayout::BSND},

        {FaLayout::BSND, FaLayout::BSND, FaLayout::BSND},    {FaLayout::BSND, FaLayout::PA_BBND, FaLayout::BSND},
        {FaLayout::BSND, FaLayout::PA_BNBD, FaLayout::BSND}, {FaLayout::BSND, FaLayout::PA_Nz, FaLayout::BSND},

        {FaLayout::TND, FaLayout::TND, FaLayout::TND},       {FaLayout::TND, FaLayout::PA_BBND, FaLayout::TND},
        {FaLayout::TND, FaLayout::PA_BNBD, FaLayout::TND},   {FaLayout::TND, FaLayout::PA_Nz, FaLayout::TND},
    };
    layoutQ_ = layoutMap.at(opParamInfo_.layoutQ);
    layoutKV_ = layoutMap.at(opParamInfo_.layoutKV);
    layoutOut_ = layoutMap.at(opParamInfo_.layoutOut);

    std::tuple<FaLayout, FaLayout, FaLayout> inputLayoutTuple = {layoutQ_, layoutKV_, layoutOut_};

    // OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(inputLayoutTuple, layoutSupported),
    //         OP_LOGE(opName_, "InputLayout layoutQ(%s),  layoutKV(%s), layoutOut(%s) is not currently \n
    //         supported.", opParamInfo_.layoutQ, opParamInfo_.layoutKV, opParamInfo_.layoutOut), 
    //         return ge::GRAPH_FAILED);
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
    queryShape_ =
        std::make_shared<FaTilingShape>(opParamInfo_.query.shape->GetStorageShape(), layoutQ_, QUERY_NAME, opName_);
    keyShape_ =
        std::make_shared<FaTilingShape>(opParamInfo_.key.shape->GetStorageShape(), layoutKV_, KEY_NAME, opName_);
    valueShape_ =
        std::make_shared<FaTilingShape>(opParamInfo_.value.shape->GetStorageShape(), layoutKV_, VALUE_NAME, opName_);
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

ge::graphStatus FaInfoParser::GetActualSeqInfo()
{
    maxActualseq_ = 0;
    actualSeqLengthsQ_.clear();
    actualSeqLengthsKV_.clear();
    const gert::Tensor *actSeqLenQ = opParamInfo_.actualSeqLengthsQ.tensor;
    const gert::Tensor *actSeqLenKV = opParamInfo_.actualSeqLengths.tensor;
    const gert::Tensor *actSharedPrefixLen = opParamInfo_.actualSharedPrefixLen.tensor;
    uint32_t actSeqLenQDims = 0;
    uint32_t actSeqLenKVDims = 0;
    uint32_t actSharedPrefixLenDims = 0;
    actSeqLenQDims = (actSeqLenQ != nullptr) ? actSeqLenQ->GetShapeSize() : 0;
    actSeqLenKVDims = (actSeqLenKV != nullptr) ? actSeqLenKV->GetShapeSize() : 0;
    actSharedPrefixLenDims = (actSharedPrefixLen != nullptr) ? actSharedPrefixLen->GetShapeSize() : 0;
    bool actualSeqLenQFlag =
        !((actSeqLenQDims == 0) || (actSeqLenQ == nullptr) || (actSeqLenQ->GetData<int64_t>() == nullptr));
    bool actualSeqLenKVFlag =
        !((actSeqLenKVDims == 0) || (actSeqLenKV == nullptr) || (actSeqLenKV->GetData<int64_t>() == nullptr));
    bool actualSharedPrefixLenFlag = !((actSharedPrefixLenDims == 0) || (actSharedPrefixLen == nullptr) ||
                                       (actSharedPrefixLen->GetData<int64_t>() == nullptr));

    for (uint32_t bIdx = 0; bIdx < bSize_; bIdx++) {
        if (actualSeqLenQFlag) {
            int64_t actSeqLenQData = 0;
            if (actSeqLenQDims == 1) {
                actSeqLenQData = actSeqLenQ->GetData<int64_t>()[0];
            } else {
                if (layoutQ_ != FaLayout::TND) {
                    actSeqLenQData = actSeqLenQ->GetData<int64_t>()[bIdx];
                } else {
                    if (bIdx == 0) {
                        actSeqLenQData = actSeqLenQ->GetData<int64_t>()[bIdx];
                    } else {
                        actSeqLenQData =
                            actSeqLenQ->GetData<int64_t>()[bIdx] - actSeqLenQ->GetData<int64_t>()[bIdx - 1];
                    }
                }
            }
            actualSeqLengthsQ_.push_back(actSeqLenQData);
        } else {
            actualSeqLengthsQ_.push_back(s1Size_);
        }

        if (actualSeqLenKVFlag) {
            if (actSeqLenKVDims == 1) {
                actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[0]);
    } else {
                if (layoutKV_ != FaLayout::TND) {
                    actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx]);
                } else {
                    if (bIdx == 0) {
                        actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx]);
                    } else {
                        actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx] -
                                                      actSeqLenKV->GetData<int64_t>()[bIdx - 1]);
                    }
                }
            }
        } else {
            actualSeqLengthsKV_.push_back(s2Size_);
        }
    }

    // Store flags for later use (maybe not needed)
    actualLenQDims_ = actSeqLenQDims;
    actualLenDims_ = actSeqLenKVDims;
    isAccumQSeq_ = (layoutQ_ == FaLayout::TND);
    isAccumKVSeq_ = (layoutKV_ == FaLayout::TND);

    return ge::GRAPH_SUCCESS;
}

TilingKeyLayout FaInfoParser::MapStringToLayout(FaLayout &layoutString) const
{
    const std::map<FaLayout, TilingKeyLayout> layoutMap = {
        {FaLayout::BSH, TilingKeyLayout::BSH_BSND},   {FaLayout::BSND, TilingKeyLayout::BSH_BSND},
        {FaLayout::BNSD, TilingKeyLayout::BNSD},      {FaLayout::NZ, TilingKeyLayout::NZ},
        {FaLayout::TND, TilingKeyLayout::TND},        {FaLayout::NBSD, TilingKeyLayout::NBSD},
        {FaLayout::BnBsH, TilingKeyLayout::BSH_BSND}, {FaLayout::BnNBsD, TilingKeyLayout::BNSD},
    };

    auto it = layoutMap.find(layoutString);
    if (it != layoutMap.end()) {
        return it->second;
    }
    return TilingKeyLayout::BSH_BSND;
}

void FaInfoParser::GenerateFeatureInfo(FaTilingInfo &faInfo) // TODO 确认是否删除
{
    // empty tensor
    faInfo.emptyTensorFlag = emptyTensorFlag_;

    // pa
    faInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    faInfo.blockSize = blockSize_;
    faInfo.blockTypeSize = sizeof(float);

    // atten mask //TODO 确认是否删除
    faInfo.attenMaskFlag = attenMaskFlag_;
    faInfo.attenMaskBatchStride = attenMaskBatchStride_;
    faInfo.attenMaskStride = attenMaskStride_;
    faInfo.maskMode = maskMode_;

    faInfo.actualSeqLengthsQ = actualSeqLengthsQ_;
    faInfo.actualSeqLengthsKV = actualSeqLengthsKV_;
    // // 4: only mla noquant & band mode suppport slidingFlag
    // faInfo.slidingFlag =
    //     (*opParamInfo_.sparseMode == 4) && (ropeMode_ == RopeMode::ROPE_SPLIT) && (qkHeadDim_ == 512U);
    faInfo.qPaddingSizeFlag = qPaddingSizeFlag_;
    faInfo.kvPaddingSizeFlag = kvPaddingSizeFlag_;
    faInfo.softmaxLseFlag = *opParamInfo_.softmaxLseFlag;
    faInfo.totalLseSize =
        (opParamInfo_.lseOut.shape == nullptr) ? 0 : opParamInfo_.lseOut.shape->GetStorageShape().GetShapeSize();
    faInfo.isLegacyIfa = isLegacyIfa_;
    faInfo.preToken = preToken_;
    faInfo.nextToken = nextToken_;
    // faInfo.sinksFlag = (opParamInfo_.sinks.tensor != nullptr);
    faInfo.antiQuantFlag = antiQuantFlag_;
    faInfo.antiqSeqSize = antiquantParaSeqSize_;
    // faInfo.keyAntiquantMode = static_cast<uint32_t>(*opParamInfo_.keyAntiquantMode);
    // faInfo.valueAntiquantMode = static_cast<uint32_t>(*opParamInfo_.valueAntiquantMode);

    faInfo.isQKVDDifferent = (qkHeadDim_ != vHeadDim_) && !(qkHeadDim_ == 192U && vHeadDim_ == 128U);

    faInfo.sysPrefixFlag = systemPrefixFlag_;
    faInfo.systemPrefixLen = systemPrefixLen_;
    faInfo.systemPrefixMaxLen = systemPrefixMaxLen_;
}

void FaInfoParser::GenerateLayoutInfo(FaTilingInfo &faInfo)
{
    faInfo.qLayout = layoutQ_;
    faInfo.kvLayout = layoutKV_;
    faInfo.outLayout = layoutOut_;
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
    faInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS); // TODO 删除

    faInfo.totalOutputSize =
        opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize(); // TODO 确认空tensor场景是否需要

    // faInfo.totalBlockNum = kCache_[0]->GetStorageShape().GetDim(0);
    faInfo.softmaxScale = *opParamInfo_.softmaxScale;  // TODO 改名
    faInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_; // TODO PA 确认是否需要删除
    // faInfo.maxActualseq = maxActualseq_;               // TODO 确认是否需要

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
    faInfo.qTSize = queryTSize_;
    faInfo.kTSize = keyTSize_;
}

void FaInfoParser::GenerateDtypeInfo(FaTilingInfo &faInfo)
{
    faInfo.inputQType = inputQType_;
    faInfo.inputKvType = inputKvType_;
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
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }
    GetInOutDataType();
    GetKvStorageMode();

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
    SetFaShape(); // TODO n1Size_获取
    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size()) {
        return ge::GRAPH_FAILED;
    }

    GetQueryTSize();

    if (ge::GRAPH_SUCCESS != GetQkHeadDim() || ge::GRAPH_SUCCESS != GetValueHeadDim()) {
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

ge::graphStatus FaInfoParser::ParseFeatureInfo() // TODO FIA 加sink和lse
{
    GetPreNextToken();
    if (ge::GRAPH_SUCCESS != GetActualSeqInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling