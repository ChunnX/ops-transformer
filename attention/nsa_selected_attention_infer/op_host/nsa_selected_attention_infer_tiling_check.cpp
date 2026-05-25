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
 * \file nsa_selected_attention_infer_tiling_check.cpp
 * \brief
 */

#include <numeric>
#include <cstdio>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "nsa_selected_attention_infer_tiling.h"
#include "nsa_selected_attention_infer_tiling_base.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus NsaSelectTiling::CheckPABlockSize() const
{
    OP_CHECK_IF(
        blockSize_ == 0,
        ({ OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, "When Page Attention is enabled, input attribute blocksize can not be 0."); }),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((inputKvType_ == ge::DT_FLOAT16) || (inputKvType_ == ge::DT_BF16)) && (blockSize_ % 16 != 0),
        ({ OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, "When Page Attention is enabled, if kv cache dtype is float16/bfloat16, input attr blocksize should be 16 aligned"); }),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::CheckBaseInputsNull() const {
    // Check base input tensors
    OP_CHECK_IF(context_->query.shape == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "query"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->query.shape->GetStorageShape().GetShapeSize() == 0,
            ({ OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, "Tensor q is empty cause shapesize is 0."); }), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->query.desc == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "query"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->key.shape == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "k"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->key.desc == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "k"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->value.shape == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "value"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->value.desc == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "value"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->topkIndices.shape == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "topkindices"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->topkIndices.desc == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "topkindices"); }),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->actualKVSeqLengths.tensor == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "of"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->actualKVSeqLengths.desc == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "actualkvseqlengths"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->actualQSeqLengths.tensor == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "of"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->actualQSeqLengths.desc == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "actualqseqlengths"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->blockTable.tensor == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "of"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->blockTable.desc == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "blocktable"); }),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->attenOut.desc == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "output"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->attenOut.shape == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] parameter", context_->opName, "output"); }),
            return ge::GRAPH_FAILED);

    // Check base input attrs
    OP_CHECK_IF(context_->numHeads == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] attribute", context_->opName, "num_heads"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->scaleValue == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] attribute", context_->opName, "scale_value"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kvHeadNums == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] attribute", context_->opName, "kv_head_nums"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->layOut == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] attribute", context_->opName, "lay_out"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->blockSize == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] attribute", context_->opName, "block_size"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->selectedBlockSize == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] attribute", context_->opName, "selected_block_size"); }),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->selectedBlockCount == nullptr, ({ OP_LOGE(context_->opName, "The op [%s] missing [%s] attribute", context_->opName, "selected_block_count"); }),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus NsaSelectTiling::CheckInputFormatAndLimits() const
{
    OP_CHECK_IF(
        ((inputQType_ != inputKvType_)),
        ({ char reason[512] = {0}; OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, (snprintf(reason, sizeof(reason), "when input Q type %d is not equal to input KV type %d", inputQType_, inputKvType_) > 0) ? reason : "when input Q type %d is not equal to input KV type %d"); }),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (inputKvType_ == ge::DT_FLOAT16 || inputKvType_ == ge::DT_BF16) && (blockSize_ % 16 != 0),
        ({ char reason[512] = {0}; OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, (snprintf(reason, sizeof(reason), "blockSize=%ld, it need align to 16 when kv dtype is fp16/bf16.", blockSize_) > 0) ? reason : "blockSize=%ld, it need align to 16 when kv dtype is fp16/bf16."); }),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (batchSize_ > 3072),
        ({ char reason[512] = {0}; OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, (snprintf(reason, sizeof(reason), "batch size:%u cannot be greater than 3072.", batchSize_) > 0) ? reason : "batch size:%u cannot be greater than 3072."); }),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((blockSize_ != 64) && (blockSize_ != 128), ({ OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, "blockSize is not 64 or 128"); }), return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectedBlockSize_ % 16 != 0 || selectedBlockSize_ == 0, ({ char reason[512] = {0}; OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, (snprintf(reason, sizeof(reason), "selectedBlockSize %ld is not 16 aligned", selectedBlockSize_) > 0) ? reason : "selectedBlockSize %ld is not 16 aligned"); }), return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectedBlockSize_ > 128 , ({ char reason[512] = {0}; OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, (snprintf(reason, sizeof(reason), "selectedBlockSize %ld should not be greater than 128", selectedBlockSize_) > 0) ? reason : "selectedBlockSize %ld should not be greater than 128"); }), return ge::GRAPH_FAILED);
    OP_CHECK_IF((headDim_ != 192),
        ({ char reason[512] = {0}; OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, (snprintf(reason, sizeof(reason), "QK Head dim must == 192, but got: %u.", headDim_) > 0) ? reason : "QK Head dim must == 192, but got: %u."); }),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((headDim_ <= headDimV_),
        ({ char reason[512] = {0}; OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, (snprintf(reason, sizeof(reason), "headDim:%u cannot be smaller than headDimV:%u.", headDim_, headDimV_) > 0) ? reason : "headDim:%u cannot be smaller than headDimV:%u."); }),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((numKvHeads_ > 256),
        ({ char reason[512] = {0}; OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, (snprintf(reason, sizeof(reason), "numHead of key and value:%ld cannot be greater than 256.", numKvHeads_) > 0) ? reason : "numHead of key and value:%ld cannot be greater than 256."); }),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((headDimV_ != 128),
        ({ char reason[512] = {0}; OP_LOGE(context_->opName, "The op [%s] received bad params, the reason is: [%s]", context_->opName, (snprintf(reason, sizeof(reason), "V Head dim must == 128, but got: %u.", headDimV_) > 0) ? reason : "V Head dim must == 128, but got: %u."); }),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::CheckQKOutShape()
{
    const gert::StorageShape *queryShape = context_->query.shape;
    const gert::StorageShape *keyShape = context_->key.shape;
    const std::string inputLayoutStr = context_->layOut;

    auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
    auto dimOfK = keyShape->GetStorageShape().GetDimNum();
    auto dimOfOut = context_->attenOut.shape->GetStorageShape().GetDimNum();
    if (inputLayoutStr == "BSH") {
        OP_CHECK_IF(
            (dimOfQ != DIM_BSH) || (dimOfK != DIM_BSH) || (dimOfOut != DIM_BSH),
            ({ char reason[512] = {0}; OP_LOGE("[NsaSelectInfer]", "The op [%s] received bad params, the reason is: [%s]", "[NsaSelectInfer]", (snprintf(reason, sizeof(reason), "When input layout is BSH, the dimension should be 3, dimOfQ: %lu, dimOfK: %lu, dimOfOut: %lu", dimOfQ, dimOfK, dimOfOut) > 0) ? reason : "When input layout is BSH, the dimension should be 3, dimOfQ: %lu, dimOfK: %lu, dimOfOut: %lu"); }),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((queryShape->GetStorageShape().GetDim(1) < 1) || (queryShape->GetStorageShape().GetDim(1) > 4 ),
                ({ char reason[512] = {0}; OP_LOGE("[NsaSelectInfer]", "The op [%s] received bad params, the reason is: [%s]", "[NsaSelectInfer]", (snprintf(reason, sizeof(reason), "When input layout is BSH, the 2nd dimOfQ should be be greater than or equal to 1 and  less than 4, the 2nd dimOfQ: %ld", queryShape->GetStorageShape().GetDim(1)) > 0) ? reason : "When input layout is BSH, the 2nd dimOfQ should be be greater than or equal to 1 and  less than 4, the 2nd dimOfQ: %ld"); }),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(queryShape->GetStorageShape().GetDim(2) / numHeads_ !=
                    keyShape->GetStorageShape().GetDim(2) / numKvHeads_,
                ({ char reason[512] = {0}; OP_LOGE("[NsaSelectInfer]", "The op [%s] received bad params, the reason is: [%s]", "[NsaSelectInfer]", (snprintf(reason, sizeof(reason), "When input layout is BSH,the 3rd dimOfQ/numHeads(%ld) should be equal to the 3rd dimOfK/numKvHeads(%ld)", queryShape->GetStorageShape().GetDim(2) / numHeads_, keyShape->GetStorageShape().GetDim(2) / numKvHeads_) > 0) ? reason : "When input layout is BSH,the 3rd dimOfQ/numHeads(%ld) should be equal to the 3rd dimOfK/numKvHeads(%ld)"); }),
                return ge::GRAPH_FAILED);
    } else if (inputLayoutStr == "TND") {
        return TNDCheckQKOutShape();
    } else {
        OP_CHECK_IF(
            (dimOfQ != DIM_BNSD_OR_BNSD) || (dimOfK != DIM_BNSD_OR_BNSD) || (dimOfOut != DIM_BNSD_OR_BNSD),
            ({ char reason[512] = {0}; OP_LOGE("[NsaSelectInfer]", "The op [%s] received bad params, the reason is: [%s]", "[NsaSelectInfer]", (snprintf(reason, sizeof(reason), "When input layout is BNSD/BSND, the dim should be 4, 4th dimOfQ: %lu, 4th dimOfK: %lu, fourth dimOfOut: %lu", dimOfQ, dimOfK, dimOfOut) > 0) ? reason : "When input layout is BNSD/BSND, the dim should be 4, 4th dimOfQ: %lu, 4th dimOfK: %lu, fourth dimOfOut: %lu"); }),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            queryShape->GetStorageShape().GetDim(3) != keyShape->GetStorageShape().GetDim(3),
            ({ char reason[512] = {0}; OP_LOGE("[NsaSelectInfer]", "The op [%s] received bad params, the reason is: [%s]", "[NsaSelectInfer]", (snprintf(reason, sizeof(reason), "When input layout is BNSD/BSND, the 4th dimOfQ not be equal the 4th dimOfK, dimOfQ: %ld, dimOfK: %ld", queryShape->GetStorageShape().GetDim(3), keyShape->GetStorageShape().GetDim(3)) > 0) ? reason : "When input layout is BNSD/BSND, the 4th dimOfQ not be equal the 4th dimOfK, dimOfQ: %ld, dimOfK: %ld"); }),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::TNDCheckQKOutShape() const
{
    const gert::StorageShape *queryShape = context_->query.shape;
    const gert::StorageShape *keyShape = context_->key.shape;
    const std::string inputLayoutStr = context_->layOut;

    auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
    auto dimOfK = keyShape->GetStorageShape().GetDimNum();
    auto dimOfOut = context_->attenOut.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (dimOfQ != DIM_TND) || (dimOfK != DIM_TND) || (dimOfOut != DIM_TND),
        ({ char reason[512] = {0}; OP_LOGE("[NsaSelectInfer]", "The op [%s] received bad params, the reason is: [%s]", "[NsaSelectInfer]", (snprintf(reason, sizeof(reason), "When input layout is TND, the dim should be 3, dimOfQ: %lu,  dimOfK: %lu,  dimOfOut: %lu", dimOfQ, dimOfK, dimOfOut) > 0) ? reason : "When input layout is TND, the dim should be 3, dimOfQ: %lu,  dimOfK: %lu,  dimOfOut: %lu"); }),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        queryShape->GetStorageShape().GetDim(2) != keyShape->GetStorageShape().GetDim(2) / numKvHeads_,
        ({ char reason[512] = {0}; OP_LOGE("[NsaSelectInfer]", "The op [%s] received bad params, the reason is: [%s]", "[NsaSelectInfer]", (snprintf(reason, sizeof(reason), "When input layout is TND, the 3th dimOfQ should be equal the 3th dimOfK/numKvHeads, dimOfQ: %ld, dimOfK: %ld", queryShape->GetStorageShape().GetDim(2), keyShape->GetStorageShape().GetDim(2)) > 0) ? reason : "When input layout is TND, the 3th dimOfQ should be equal the 3th dimOfK/numKvHeads, dimOfQ: %ld, dimOfK: %ld"); }),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::CheckUbSpace()
{
    if (!CalcUbBmm()) {
        return false;
    }
    return true;
}

} // namespace optiling
 