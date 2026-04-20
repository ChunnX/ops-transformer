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
    OP_CHECK_IF(opParamInfo_.attnOut.shape == nullptr, OP_LOGE(opName_, "Shape of tensor output is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attnOut.desc == nullptr, OP_LOGE(opName_, "Desc of tensor output is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::CheckRequiredAttrExistence() const
{
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
         opParamInfo_.attnOut.shape->GetStorageShape().GetShapeSize() != 0) ||
        (opParamInfo_.lseOut.shape != nullptr && opParamInfo_.attnOut.shape->GetStorageShape().GetShapeSize() == 0)) {
        OP_LOGE(opName_,
                "query shape size is %llu byte, but attention Out shape size is %llu byte, they cannot be empty while "
                "the other is not",
                opParamInfo_.query.shape->GetStorageShape().GetShapeSize(),
                opParamInfo_.attnOut.shape->GetStorageShape().GetShapeSize());
    }
    if (opParamInfo_.lseOut.shape != nullptr && opParamInfo_.attnOut.shape->GetStorageShape().GetShapeSize() == 0) {
        // emptyTensorFlag_ = true;
        return ge::GRAPH_SUCCESS;
    }
    bool lseEnabled = (opParamInfo_.returnSoftMaxLse != nullptr && *opParamInfo_.returnSoftMaxLse != 0);
    if (lseEnabled) {
        if ((opParamInfo_.lseOut.shape == nullptr) ||
            (opParamInfo_.lseOut.shape->GetStorageShape().GetShapeSize() == 0)) {
            OP_LOGE(opName_, "return_softmax_lse is %d, but lse shape size is 0 byte",
                    opParamInfo_.returnSoftMaxLse != nullptr ? *opParamInfo_.returnSoftMaxLse : 0);
            return ge::GRAPH_FAILED;
        }
    }
    // emptyTensorFlag_ = true;
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus FaInfoParser::GetCuSeqLenQSize(uint32_t &size) // TODO TND场景 获取cu
{
    if (opParamInfo_.cuSeqlensQ.tensor == nullptr) {
        OP_LOGE(opName_, "when %s's layout is %s, %s must be provided.", QUERY_NAME.c_str(),
                LayoutToSerialString(layoutQ_).c_str(), CU_SEQLENS_Q_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = opParamInfo_.cuSeqlensQ.tensor->GetShapeSize();
    if (shapeSize <= 1) {
        OP_LOGE(opName_, "%s's shape size is %ld, it should be greater than 1.", CU_SEQLENS_Q_NAME.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize) - 1;
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
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_3510) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetOptionalInputParaInfo()
{
    GetOptionalInputParaSeqLengthInfo();
    GetOptionalInputParaPageAttentionInfo();
    GetOptionalInputParaSinksInfo();
}
void FaInfoParser::GetOptionalInputParaMaskInfo()
{
    opParamInfo_.attnMask.tensor = context_->GetOptionalInputTensor(ATTN_MASK_INDEX);
    opParamInfo_.attnMask.desc = context_->GetOptionalInputDesc(ATTN_MASK_INDEX);
}

void FaInfoParser::GetOptionalInputParaSeqLengthInfo()
{
    opParamInfo_.cuSeqlensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensKv.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_KV_INDEX);
    opParamInfo_.cuSeqlensKv.desc = context_->GetOptionalInputDesc(CU_SEQLENS_KV_INDEX);
    opParamInfo_.sequsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedKv.tensor = context_->GetOptionalInputTensor(SEQUSED_KV_INDEX);
    opParamInfo_.sequsedKv.desc = context_->GetOptionalInputDesc(SEQUSED_KV_INDEX);
}

void FaInfoParser::GetOptionalInputParaPageAttentionInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void FaInfoParser::GetOptionalInputParaSinksInfo()
{
    opParamInfo_.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INDEX);
    opParamInfo_.sinks.desc = context_->GetOptionalInputDesc(SINKS_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
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
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    opParamInfo_.lseOut.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.lseOut.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus FaInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    // 索引0: softmax_scale (Float)
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);

    // 索引1: mask_mode (Int)
    opParamInfo_.maskMode = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);

    // 索引2: win_left (Int)
    opParamInfo_.winLeft = attrs->GetAttrPointer<int64_t>(ATTR_WIN_LEFT_INDEX);

    // 索引3: win_right (Int)
    opParamInfo_.winRight = attrs->GetAttrPointer<int64_t>(ATTR_WIN_RIGHT_INDEX);

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

    // 索引9: return_softmax_lse (Int)
    opParamInfo_.returnSoftMaxLse = attrs->GetAttrPointer<int64_t>(ATTR_RETURN_LSE_INDEX);

    // 索引10: deterministic (Int)
    opParamInfo_.deterministic = attrs->GetAttrPointer<int64_t>(ATTR_DETERMINISTIC_INDEX);

    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetUpdateInfo()
{
    auto attrs = context_->GetAttrs();
    static int64_t MASK_ZERO = 0U;
    static int64_t TOKEN_MAX = 2147483647;
    // TODO wmk
    if (false) {
        opParamInfo_.maskMode = &MASK_ZERO;
        opParamInfo_.winLeft = &TOKEN_MAX;
        opParamInfo_.winRight = &TOKEN_MAX;
    } else {
        // opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
        // opParamInfo_.cuSeqlensQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
        // opParamInfo_.cuSeqlensQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
        // opParamInfo_.queryPaddingSize.tensor = context_->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
        // opParamInfo_.queryPaddingSize.desc = context_->GetOptionalInputDesc(QUERY_PADDING_SIZE_INDEX);
    }
}

void FaInfoParser::GetPreNextToken()
{
    // 从输入读取参数值
    winLeft_ = opParamInfo_.winLeft == nullptr ? 0 : *opParamInfo_.winLeft;
    winRight_ = opParamInfo_.winRight == nullptr ? 0 : *opParamInfo_.winRight;

    // 特殊场景下需要更新值
    maskMode_ = opParamInfo_.maskMode == nullptr ? 1 : *opParamInfo_.maskMode;

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
    outputType_ = opParamInfo_.attnOut.desc->GetDataType();
}

ge::graphStatus FaInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (layoutQ_ == FaLayout::TND) {
        return GetCuSeqLenQSize(bSize_);
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

ge::graphStatus FaInfoParser::GetS2SizeFromCuSeqLens()
{
    if (opParamInfo_.cuSeqlensKv.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return GetMaxActualSeq(opParamInfo_.cuSeqlensKv.tensor, layoutKV_, s2Size_);
}

ge::graphStatus FaInfoParser::GetS2SizeForBatchContinuous()
{
    // TODO TND 无法获取
    if ((layoutKV_ == FaLayout::TND)) {
        return GetS2SizeFromCuSeqLens(); // TODO 直接获取入参，max_seqlen TND场景，未传入用T(非必传)
    } else {                             // BSH/BSND/BNSD
        if (keyShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s2Size_ = keyShape_->GetShapeS();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetMaxBlockNumPerBatch() // TODO PA场景S2 size 如何获取
{
    const gert::Tensor *actSeqLenKV = opParamInfo_.cuSeqlensKv.tensor;
    if (actSeqLenKV == nullptr) {
        OP_LOGE(opName_, "cuSeqlensKv tensor is nullptr when computing maxBlockNumPerBatch");
        return ge::GRAPH_FAILED;
    }
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

ge::graphStatus FaInfoParser::GetInAndOutLayout()
{
    const std::vector<std::tuple<FaLayout, FaLayout, FaLayout>> layoutSupported = {
        {FaLayout::BNSD, FaLayout::BNSD, FaLayout::BNSD},    {FaLayout::BNSD, FaLayout::BNSD, FaLayout::BSND},
        {FaLayout::BNSD, FaLayout::PA_BBND, FaLayout::BNSD}, {FaLayout::BNSD, FaLayout::PA_BBND, FaLayout::BSND},
        {FaLayout::BNSD, FaLayout::PA_BNBD, FaLayout::BNSD}, {FaLayout::BNSD, FaLayout::PA_BNBD, FaLayout::BSND},
        {FaLayout::BNSD, FaLayout::PA_Nz, FaLayout::BNSD},   {FaLayout::BNSD, FaLayout::PA_Nz, FaLayout::BSND},
        {FaLayout::BSND, FaLayout::BSND, FaLayout::BSND},    {FaLayout::BSND, FaLayout::PA_BBND, FaLayout::BSND},
        {FaLayout::BSND, FaLayout::PA_BNBD, FaLayout::BSND}, {FaLayout::BSND, FaLayout::PA_Nz, FaLayout::BSND},
        {FaLayout::TND, FaLayout::TND, FaLayout::TND},       {FaLayout::TND, FaLayout::PA_BBND, FaLayout::TND},
        {FaLayout::TND, FaLayout::PA_BNBD, FaLayout::TND},   {FaLayout::TND, FaLayout::PA_Nz, FaLayout::TND},
    };
    auto itQ = layoutMap.find(opParamInfo_.layoutQ);
    if (itQ == layoutMap.end()) {
        OP_LOGE(opName_, "Invalid layoutQ: %s", opParamInfo_.layoutQ);
        return ge::GRAPH_FAILED;
    }
    layoutQ_ = itQ->second;

    auto itKV = layoutMap.find(opParamInfo_.layoutKV);
    if (itKV == layoutMap.end()) {
        OP_LOGE(opName_, "Invalid layoutKV: %s", opParamInfo_.layoutKV);
        return ge::GRAPH_FAILED;
    }
    layoutKV_ = itKV->second;

    auto itOut = layoutMap.find(opParamInfo_.layoutOut);
    if (itOut == layoutMap.end()) {
        OP_LOGE(opName_, "Invalid layoutOut: %s", opParamInfo_.layoutOut);
        return ge::GRAPH_FAILED;
    }
    layoutOut_ = itOut->second;

    std::tuple<FaLayout, FaLayout, FaLayout> inputLayoutTuple = {layoutQ_, layoutKV_, layoutOut_};

    if (std::find(layoutSupported.begin(), layoutSupported.end(), inputLayoutTuple) == layoutSupported.end()) {
        OP_LOGE(opName_, "InputLayout layoutQ(%s), layoutKV(%s), layoutOut(%s) is not currently supported.",
                opParamInfo_.layoutQ, opParamInfo_.layoutKV, opParamInfo_.layoutOut);
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
        OP_LOGE(opName_, "numHeads(%u) should be a multiple of numKeyValueHeads(%u).", n1Size_, n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetActualSeqInfo()
{
    // maxActualseq_ = 0;
    // actualSeqLengthsQ_.clear();
    // actualSeqLengthsKV_.clear();
    // const gert::Tensor *actSeqLenQ = opParamInfo_.cuSeqlensQ.tensor;
    // const gert::Tensor *actSeqLenKV = opParamInfo_.cuSeqlensKv.tensor;
    // uint32_t actSeqLenQDims = 0;
    // uint32_t actSeqLenKVDims = 0;
    // actSeqLenQDims = (actSeqLenQ != nullptr) ? actSeqLenQ->GetShapeSize() : 0;
    // actSeqLenKVDims = (actSeqLenKV != nullptr) ? actSeqLenKV->GetShapeSize() : 0;
    // bool actualSeqLenQFlag =
    //     !((actSeqLenQDims == 0) || (actSeqLenQ == nullptr) || (actSeqLenQ->GetData<int64_t>() == nullptr));
    // bool actualSeqLenKVFlag =
    //     !((actSeqLenKVDims == 0) || (actSeqLenKV == nullptr) || (actSeqLenKV->GetData<int64_t>() == nullptr));

    // for (uint32_t bIdx = 0; bIdx < bSize_; bIdx++) {
    //     if (actualSeqLenQFlag) {
    //         int64_t actSeqLenQData = 0;
    //         if (actSeqLenQDims == 1) {
    //             actSeqLenQData = actSeqLenQ->GetData<int64_t>()[0];
    //         } else {
    //             if (layoutQ_ != FaLayout::TND) {
    //                 actSeqLenQData = actSeqLenQ->GetData<int64_t>()[bIdx];
    //             } else {
    //                 if (bIdx == 0) {
    //                     actSeqLenQData = actSeqLenQ->GetData<int64_t>()[bIdx];
    //                 } else {
    //                     actSeqLenQData =
    //                         actSeqLenQ->GetData<int64_t>()[bIdx] - actSeqLenQ->GetData<int64_t>()[bIdx - 1];
    //                 }
    //             }
    //         }
    //         cuSeqlensQ_.push_back(actSeqLenQData);
    //     } else {
    //         cuSeqlensQ_.push_back(s1Size_);
    //     }

    //     if (actualSeqLenKVFlag) {
    //         if (actSeqLenKVDims == 1) {
    //             cuSeqlensKvKV_.push_back(actSeqLenKV->GetData<int64_t>()[0]);
    //         } else {
    //             if (layoutKV_ != FaLayout::TND) {
    //                 cuSeqlensKvKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx]);
    //             } else {
    //                 if (bIdx == 0) {
    //                     cuSeqlensKvKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx]);
    //                 } else {
    //                     cuSeqlensKvKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx] -
    //                                              actSeqLenKV->GetData<int64_t>()[bIdx - 1]);
    //                 }
    //             }
    //         }
    //     } else {
    //         cuSeqlensKvKV_.push_back(s2Size_);
    //     }
    // }

    // // Store flags for later use (maybe not needed)
    // actualLenQDims_ = actSeqLenQDims;
    // actualLenDims_ = actSeqLenKVDims;
    // isAccumQSeq_ = (layoutQ_ == FaLayout::TND);
    // isAccumKVSeq_ = (layoutKV_ == FaLayout::TND);

    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GenerateFeatureInfo(FaTilingInfo &faInfo) // TODO 确认是否删除
{
    // empty tensor
    // faInfo.emptyTensorFlag = emptyTensorFlag_;

    // pa
    faInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    faInfo.blockSize = blockSize_;
    faInfo.blockTypeSize = sizeof(float);

    // attn mask
    faInfo.attnMaskFlag = attnMaskFlag_;
    faInfo.maskMode = maskMode_;

    // faInfo.cuSeqlensQ = cuSeqlensQ_;
    // faInfo.cuSeqlensKvKV = cuSeqlensKvKV_;
    // // 4: only mla noquant & band mode suppport slidingFlag
    // faInfo.slidingFlag =
    //     (*opParamInfo_.sparseMode == 4) && (ropeMode_ == RopeMode::ROPE_SPLIT) && (qkHeadDim_ == 512U);
    faInfo.softmaxLseFlag = (opParamInfo_.returnSoftMaxLse != nullptr && *opParamInfo_.returnSoftMaxLse != 0);
    faInfo.totalLseSize =
        (opParamInfo_.lseOut.shape == nullptr) ? 0 : opParamInfo_.lseOut.shape->GetStorageShape().GetShapeSize();
    faInfo.winLeft = winLeft_;
    faInfo.winRight = winRight_;
    // faInfo.sinksFlag = (opParamInfo_.sinks.tensor != nullptr);
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
    GenerateAxisInfo(faInfo);
    GenerateDtypeInfo(faInfo);
    faInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS); // TODO 删除

    faInfo.totalOutputSize =
        opParamInfo_.attnOut.shape->GetStorageShape().GetShapeSize(); // TODO 确认空tensor场景是否需要

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

    if (ge::GRAPH_SUCCESS != GetInAndOutLayout()) {
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

    printf("===== FaTilingInfo Dump =====\n");
    printf("opName: %s\n", faInfo.opName);
    printf("bSize: %u, n1Size: %u, n2Size: %u\n", faInfo.bSize, faInfo.n1Size, faInfo.n2Size);
    printf("s1Size: %u, s2Size: %ld\n", faInfo.s1Size, faInfo.s2Size);
    printf("qkHeadDim: %u, vHeadDim: %u, gSize: %u\n", faInfo.qkHeadDim, faInfo.vHeadDim, faInfo.gSize);
    printf("qTSize: %u, kTSize: %u\n", faInfo.qTSize, faInfo.kTSize);
    printf("softmaxScale: %f\n", faInfo.softmaxScale);
    printf("numHeads: %u, kvHeadNums: %u\n", faInfo.numHeads, faInfo.kvHeadNums);
    printf("emptyTensorFlag: %d, totalOutputSize: %lu, totalLseSize: %lu\n", faInfo.emptyTensorFlag,
           faInfo.totalOutputSize, faInfo.totalLseSize);
    printf("pageAttentionFlag: %d, blockSize: %d\n", faInfo.pageAttentionFlag, faInfo.blockSize);
    printf("blockTypeSize: %u, maxBlockNumPerBatch: %u, totalBlockNum: %u\n", faInfo.blockTypeSize,
           faInfo.maxBlockNumPerBatch, faInfo.totalBlockNum);
    printf("seqUsedQDims: %u, cuSeqLenQDims: %u, maxSeqQ: %ld\n", faInfo.seqUsedQDims, faInfo.cuSeqLenQDims,
           faInfo.maxSeqQ);
    printf("seqUsedKvDims: %u, cuSeqLenKvDims: %u, maxSeqKv: %ld\n", faInfo.seqUsedKvDims, faInfo.cuSeqLenKvDims,
           faInfo.maxSeqKv);
    printf("attnMaskFlag: %d, isExistRowInvalid: %d, maskMode: %d\n", faInfo.attnMaskFlag, faInfo.isExistRowInvalid,
           faInfo.maskMode);
    printf("winLeft: %ld, winRight: %ld\n", faInfo.winLeft, faInfo.winRight);
    printf("batchContinuousFlag: %d, softmaxLseFlag: %d, isMaxWorkspace: %d\n", faInfo.batchContinuousFlag,
           faInfo.softmaxLseFlag, faInfo.isMaxWorkspace);
    printf("needInit: %d, slidingFlag: %d, sinksFlag: %d\n", faInfo.needInit, faInfo.slidingFlag, faInfo.sinksFlag);
    printf("inputQType: %d, inputKvType: %d, outputType: %d\n", static_cast<int>(faInfo.inputQType),
           static_cast<int>(faInfo.inputKvType), static_cast<int>(faInfo.outputType));
    printf("qLayout: %u, outLayout: %u, kvLayout: %u\n", static_cast<uint32_t>(faInfo.qLayout),
           static_cast<uint32_t>(faInfo.outLayout), static_cast<uint32_t>(faInfo.kvLayout));
    printf("===== FaTilingInfo Dump End =====\n");

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