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
 * \file fused_infer_attention_score_tiling_v3.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling_v3.h"
#include "fused_infer_attention_score_tiling_check.h"
#include "fused_infer_attention_score_tiling_info_parser.h"
#include "../../common/op_host/arch32/fia_tiling_nonquant_mla.h"
#include "../../common/op_host/arch32/fia_tiling_nonquant.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"

using namespace AscendC;
namespace optiling {
constexpr size_t DIM_NZ = 5;
constexpr uint32_t NZ_D1_IDX = 2;
constexpr uint32_t NZ_D0_IDX = 4;
constexpr uint32_t TND_NTD_D_IDX = 2;


FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV3(gert::TilingContext *context)
{
    FiaTilingInfo fiaInfo;
    FiaInfoParser fiaInfoParser(context);
    if (fiaInfoParser.Parse(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check函数只做校验，不能修改fiaInfo中的信息
    if (TilingCheck::Check(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo);
}

bool GetPaValueD(gert::TilingContext *context, int64_t &valueD)
{
    auto attrs = context->GetAttrs();
    int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
    int64_t numKvHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX));
    if (numKvHeads == 0) {
        numKvHeads = numHeads;
    }
    auto vStorageShape = context->GetInputShape(VALUE_INDEX)->GetStorageShape();
    if (vStorageShape.GetDimNum() == DIM_BSH) {
        valueD = vStorageShape.GetDim(BSH_H_IDX) / numKvHeads; // BnBsH
    } else if (vStorageShape.GetDimNum() == DIM_BNSD_OR_BSND) {
        valueD = vStorageShape.GetDim(BNSD_D_IDX); // BnNBsD
    } else if (vStorageShape.GetDimNum() == DIM_NZ) {
        valueD = vStorageShape.GetDim(NZ_D1_IDX) * vStorageShape.GetDim(NZ_D0_IDX); // NZ: BnND1BsD0
    } else {
        return false;
    }
    return true;
}

bool GetValueD(gert::TilingContext *context, int64_t &valueD)
{
    auto vShape = context->GetInputShape(VALUE_INDEX);
    if (vShape == nullptr) {
        return false;
    }
    auto vStorageShape = vShape->GetStorageShape();

    bool isPageAttention = context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr;
    if (isPageAttention) {
        return GetPaValueD(context, valueD);
    }

    auto attrs = context->GetAttrs();
    int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
    int64_t numKvHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX));
    if (numKvHeads == 0) {
        numKvHeads = numHeads;
    }
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BNSD_BSND" ||
        inputLayoutStr == "BSND_BNSD" ||
        inputLayoutStr == "BNSD_NBSD" ||
        inputLayoutStr == "BSND_NBSD" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND") {
        if (vStorageShape.GetDimNum() != DIM_BNSD_OR_BSND) {
            return false;
        }
        valueD = vStorageShape.GetDim(BNSD_D_IDX);
    } else if (inputLayoutStr == "BSH" ||
        inputLayoutStr == "BSH_NBSD" ||
        inputLayoutStr == "BSH_BNSD") {
        if (vStorageShape.GetDimNum() != DIM_BSH) {
            return false;
        }
        valueD = vStorageShape.GetDim(BSH_H_IDX) / numKvHeads;
    } else if (inputLayoutStr == "TND" ||
        inputLayoutStr == "NTD" ||
        inputLayoutStr == "TND_NTD" ||
        inputLayoutStr == "NTD_TND") {
        if (vStorageShape.GetDimNum() != DIM_TND) {
            return false;
        }
        valueD = vStorageShape.GetDim(TND_NTD_D_IDX);
    } else {
        return false;
    }

    return true;
}

bool GetQkvD(gert::TilingContext *context, int64_t &queryD, int64_t &queryRopeD, int64_t &valueD)
{
    auto qShape = context->GetInputShape(QUERY_INDEX);
    auto qRopeShape = context->GetOptionalInputShape(QUERY_ROPE_INDEX);
    if (qShape == nullptr) {
        return false;
    }
    auto qStorageShape = qShape->GetStorageShape();
    
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }

    int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BNSD_BSND" ||
        inputLayoutStr == "BSND_BNSD" ||
        inputLayoutStr == "BNSD_NBSD" ||
        inputLayoutStr == "BSND_NBSD" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND") {
        if (qStorageShape.GetDimNum() != DIM_BNSD_OR_BSND) {
            return false;
        }
        queryD = qStorageShape.GetDim(BNSD_D_IDX);
        if (qRopeShape != nullptr) {
            queryRopeD = qRopeShape->GetStorageShape().GetDim(BNSD_D_IDX);
        }
    } else if (inputLayoutStr == "BSH" ||
        inputLayoutStr == "BSH_NBSD" ||
        inputLayoutStr == "BSH_BNSD") {
        if (qStorageShape.GetDimNum() != DIM_BSH) {
            return false;
        }
        queryD = qStorageShape.GetDim(BSH_H_IDX) / numHeads;
        if (qRopeShape != nullptr) {
            queryRopeD = qRopeShape->GetStorageShape().GetDim(BSH_H_IDX) / numHeads;
        }
    } else if (inputLayoutStr == "TND" ||
        inputLayoutStr == "NTD" ||
        inputLayoutStr == "TND_NTD" ||
        inputLayoutStr == "NTD_TND") {
        if (qStorageShape.GetDimNum() != DIM_TND) {
            return false;
        }
        queryD = qStorageShape.GetDim(TND_NTD_D_IDX);
        if (qRopeShape != nullptr) {
            queryRopeD = qRopeShape->GetStorageShape().GetDim(TND_NTD_D_IDX);
        }
    } else {
        return false;
    }

    return GetValueD(context, valueD);
}

bool CheckGqaDSupport(gert::TilingContext *context)
{
    int64_t queryD = 0;
    int64_t queryRopeD = 0;
    int64_t valueD = 0;
    if (GetQkvD(context, queryD, queryRopeD, valueD) != true) {
        return false;
    }

    // D的组合(128+0,128)(64+0,64)(128+64,128)(192+0,128)
    if ((queryD  == 128 && queryRopeD  == 0 && valueD == 128) || // 128: gqa qkvD
        (queryD  == 64 && queryRopeD  == 0 && valueD == 64) || // 64: gqa qkvD
        (queryD  == 128 && queryRopeD  == 64 && valueD == 128) || // 128: gqa qkvD, 64: mla ropeD
        (queryD  == 192 && queryRopeD  == 0 && valueD == 128)) { // 192: gqa qkD, 128: gqa valueD
        return true;
    }

    return false;
}

bool CheckGqaInputLayoutSupport(gert::TilingContext *context)
{
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BNSD_BSND" ||
        inputLayoutStr == "BSND_BNSD" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND" ||
        inputLayoutStr == "BSH_BNSD" ||
        inputLayoutStr == "BSH" ||
        inputLayoutStr == "TND" ||
        inputLayoutStr == "NTD" ||
        inputLayoutStr == "NTD_TND") {
        return true;
    }

    return false;
}

bool IsEmptyTensor(gert::TilingContext *context)
{
    auto qShape = context->GetInputShape(QUERY_INDEX);
    if ((qShape != nullptr) && (qShape->GetStorageShape().GetShapeSize() == 0)) {
        return true;
    }

    auto attenoutShape = context->GetInputShape(ATTENTION_OUT_INDEX);
    if ((attenoutShape != nullptr) && (attenoutShape->GetStorageShape().GetShapeSize() == 0)) {
        return true;
    }

    bool softmaxLseFlag = context->GetAttrs()->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    if (softmaxLseFlag) {
        auto softmaxLseShape = context->GetInputShape(SOFTMAX_LSE_INDEX);
        if ((softmaxLseShape != nullptr) && (softmaxLseShape->GetStorageShape().GetShapeSize() == 0)) {
            return true;
        }
    }

    uint32_t keyBIdx = 0;
    while ((context->GetDynamicInputShape(KEY_INDEX, keyBIdx)) != nullptr) {
        const gert::StorageShape *keyShape =
            const_cast<gert::StorageShape *>(context->GetDynamicInputShape(KEY_INDEX, keyBIdx));
        if (keyShape->GetStorageShape().GetShapeSize() == 0) {
            return true;
        }
        keyBIdx++;
    }

    uint32_t valueBIdx = 0;
    while ((context->GetDynamicInputShape(VALUE_INDEX, valueBIdx)) != nullptr) {
        const gert::StorageShape *valueShape =
            const_cast<gert::StorageShape *>(context->GetDynamicInputShape(VALUE_INDEX, valueBIdx));
        if (valueShape->GetStorageShape().GetShapeSize() == 0) {
            return true;
        }
        valueBIdx++;
    }

    return false;
}

bool CheckGqaFeatureSupport(gert::TilingContext *context)
{
    auto pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    auto queryPaddingSize = context->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    auto kvPaddingSize = context->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    auto keySharedPrefix = context->GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    auto valueSharedPrefix = context->GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    auto actualSharedPrefixLen = context->GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);
    auto quantScale2 = context->GetOptionalInputTensor(QUANT_SCALE2_INDEX);
    auto quantOffset2 = context->GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
    if (pseShift != nullptr ||
        queryPaddingSize != nullptr ||
        kvPaddingSize != nullptr ||
        keySharedPrefix != nullptr ||
        valueSharedPrefix != nullptr ||
        actualSharedPrefixLen != nullptr ||
        quantScale2 != nullptr ||
        quantOffset2 != nullptr) {
        return false;
    }

    return true;
}

bool CheckGqaConstrain(gert::TilingContext *context)
{
    return false;
}


bool CheckMlaInputLayoutSupport(gert::TilingContext *context)
{
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BSH" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND" ||
        inputLayoutStr == "BNSD_NBSD" ||
        inputLayoutStr == "BSND_NBSD" ||
        inputLayoutStr == "BSH_NBSD" ||
        inputLayoutStr == "TND" ||
        inputLayoutStr == "TND_NTD") {
        return true;
    }

    return false;
}

bool CheckMlaDSupport(gert::TilingContext *context)
{
    int64_t queryD = 0;
    int64_t queryRopeD = 0;
    int64_t valueD = 0;
    if (GetQkvD(context, queryD, queryRopeD, valueD) != true) {
        return false;
    }

    if ((queryD  == 512 && queryRopeD  == 64 && valueD == 512)) { // 512: mla qkvD, 64: mla ropeD
        return true;
    }

    return false;
}

bool CheckMlaConstrain(gert::TilingContext *context)
{
    return false;
}

bool RouteToFia(gert::TilingContext *context)
{
    if ((context == nullptr) || context->GetAttrs() == nullptr ||
        (context->GetInputDesc(QUERY_INDEX) == nullptr) ||
        (context->GetInputDesc(KEY_INDEX) == nullptr)) {
        return false;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        return false;
    }

    ge::DataType qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    ge::DataType kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    bool isRopeSplit = (context->GetOptionalInputTensor(QUERY_ROPE_INDEX) != nullptr &&
        context->GetOptionalInputTensor(KEY_ROPE_INDEX) != nullptr);
    if (isRopeSplit) {
        // MLA非量化
        if ((qDataType == ge::DT_FLOAT16 || qDataType == ge::DT_BF16) && (qDataType == kDataType)) {
            if (CheckGqaConstrain(context)) {
                OP_LOGI(context->GetNodeName(), "FIA RopeSplit GQA No quant.");
                return true;
            }
            if (CheckMlaConstrain(context)) {
                OP_LOGI(context->GetNodeName(), "FIA RopeSplit MLA No quant.");
                return true;
            }
            return false;
        }
    } else {
        // GQA非量化
        if ((qDataType == ge::DT_FLOAT16 || qDataType == ge::DT_BF16) && (qDataType == kDataType)) {
            OP_LOGI(context->GetNodeName(), "FIA GQA No quant.");
            return CheckGqaConstrain(context);
        }
    }
    return false;
}
} // namespace optiling
