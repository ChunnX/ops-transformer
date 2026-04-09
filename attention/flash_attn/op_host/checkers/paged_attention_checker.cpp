/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file paged_attention_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "paged_attention_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// 公共校验函数
ge::graphStatus PagedAttentionChecker::CheckBlockTableExistence(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.opParamInfo.blockTable.tensor == nullptr,
        OP_LOGE(faInfo.opName,
                "When page attention enable, block table should not be null."),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeatureExistence(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF((faInfo.opParamInfo.queryPaddingSize.tensor != nullptr) ||
        (faInfo.opParamInfo.kvPaddingSize.tensor != nullptr),
        OP_LOGE(faInfo.opName,
            "When page attention enable, left padding is not supported."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((faInfo.opParamInfo.keySharedPrefix.tensor != nullptr) ||
        (faInfo.opParamInfo.valueSharedPrefix.tensor != nullptr),
        OP_LOGE(faInfo.opName,
            "When page attention enable, system prefix is not supported."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.isQKVDDifferent,
        OP_LOGE(faInfo.opName,
            "When page attention enable, the head dim of query and key should be equal."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSeqLengthKVExistence(const FaTilingInfo &faInfo)
{
    if (faInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(faInfo.opParamInfo.actualSeqLengths.tensor == nullptr ||
        faInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>() == nullptr ||
            faInfo.opParamInfo.actualSeqLengths.tensor->GetShapeSize() == 0,
        OP_LOGE(faInfo.opName,
            "When page attention enable, actualSeqLengths of KV should not be null."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int64_t PagedAttentionChecker::GetMaxBlockNumPerBatch(const FaTilingInfo &faInfo)
{
    const int32_t blockSize = faInfo.blockSize;
    const gert::Tensor* actSeqLenKV = faInfo.opParamInfo.actualSeqLengths.tensor;
    uint32_t actualSeqLengthsKVSize = static_cast<uint32_t>(actSeqLenKV->GetShapeSize());
    int64_t actualSeqKVPerBatch = 0;
    int64_t blockNumPerBatch = 0;
    int64_t maxBlockNumPerBatch = 0;
    uint32_t loop = std::min(actualSeqLengthsKVSize, faInfo.bSize);
    if (actSeqLenKV->GetData<int64_t>() == nullptr) {
        return 0;
    }
    for (uint32_t i = 0; i < loop; i++) {
        actualSeqKVPerBatch = actSeqLenKV->GetData<int64_t>()[i];
        blockNumPerBatch = (actualSeqKVPerBatch + blockSize - 1) / blockSize;
        if (blockNumPerBatch > maxBlockNumPerBatch) {
            maxBlockNumPerBatch = blockNumPerBatch;
        }
    }
    return maxBlockNumPerBatch;
}

// check mask shape
ge::graphStatus PagedAttentionChecker::CheckMaskShape(const FaTilingInfo &faInfo)
{
    if ((faInfo.sparseMode == SPARSE_MODE_NO_MASK || faInfo.sparseMode == SPARSE_MODE_ALL_MASK) &&
        faInfo.opParamInfo.attenMask.tensor != nullptr) {
        if (faInfo.opParamInfo.actualSeqLengths.tensor == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        const gert::Shape attenMaskShape = faInfo.opParamInfo.attenMask.tensor->GetStorageShape();
        int64_t maxBlockNumPerBatch = GetMaxBlockNumPerBatch(faInfo);
        uint32_t attenMaskDimNum = attenMaskShape.GetDimNum();
        OP_CHECK_IF(attenMaskDimNum > 0 &&
            (attenMaskShape.GetDim(attenMaskDimNum-1) < maxBlockNumPerBatch * faInfo.blockSize),
            OP_LOGE(faInfo.opName,
                "When page attention enable and attenMask enable (sparseMode = %u), the last dimension of "
                "attenMask(%ld) shoule be greater than maxBlockNumPerBatch(%ld) * blockSize(%d).",
                faInfo.sparseMode, attenMaskShape.GetDim(attenMaskDimNum-1), maxBlockNumPerBatch, faInfo.blockSize),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check blocktable dtype
ge::graphStatus PagedAttentionChecker::CheckBlockTableDtype(const FaTilingInfo &faInfo)
{
    if (faInfo.opParamInfo.blockTable.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::CompileTimeTensorDesc *blockTableDesc = faInfo.opParamInfo.blockTable.desc;
    OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(faInfo.opName, "When page attention enable, blockTable datatype only support INT32."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check pa cache shape
ge::graphStatus PagedAttentionChecker::CheckPACacheShape(const FaTilingInfo &faInfo, const gert::Shape tempShape,
    const std::string& inputName)
{
    uint32_t shapeDim = tempShape.GetDimNum();
    int64_t tempBlockSize = 0;
    int64_t tempH = 0;
    int64_t tempN = 0;
    int64_t tempD = 0;
    int64_t tempD0 = 0;
    int64_t tempD1 = 0;

    uint32_t compareD = 0;
    if (inputName == "key") {
        compareD = faInfo.qkHeadDim;
    } else if (inputName == "keyRope") {
        compareD = faInfo.ropeHeadDim;
    } else {
        compareD = faInfo.vHeadDim;
    }

    if (shapeDim == DIM_NUM_3) { // [blockNums, blockSize, H]
        tempBlockSize = tempShape.GetDim(DIM_NUM_1);
        tempH = tempShape.GetDim(DIM_NUM_2);
        OP_CHECK_IF(tempBlockSize != faInfo.blockSize,
            OP_LOGE(faInfo.opName, "When page attention enable, blocksize of %s(%u) should be %u",
                inputName.c_str(), tempBlockSize, faInfo.blockSize),
            return ge::GRAPH_FAILED);

        if (faInfo.inputKvType == ge::DT_INT4) {
            OP_CHECK_IF(tempH != faInfo.n2Size * compareD,
                OP_LOGE(faInfo.opName, "When page attention enable, if input kv dataType is INT32, "
                    "the axis H of %s(%u) should be %u, "
                    "if input kv dataType is INT4, the axis H of %s(%u) should be %u",
                    inputName.c_str(), tempH / NUM8, faInfo.n2Size * compareD / NUM8,
                    inputName.c_str(), tempH, faInfo.n2Size * compareD),
                return ge::GRAPH_FAILED);

            OP_CHECK_IF(tempH > H_LIMIT,
                OP_LOGE(faInfo.opName, "When page attention enable and layout is BSH, "
                    "if input kv dataType is INT32, the axis H of %s(%ld) should not > %u."
                    "if input kv dataType is INT4, the axis H of %s(%ld) should not > %u.",
                    inputName.c_str(), tempH / NUM8, H_LIMIT / NUM8, inputName.c_str(), tempH, H_LIMIT),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(tempH != faInfo.n2Size * compareD,
                OP_LOGE(faInfo.opName, "When page attention enable, the axis H(%u) of kvCache should be %u",
                    tempH, faInfo.n2Size * compareD),
                return ge::GRAPH_FAILED);

            OP_CHECK_IF(tempH > H_LIMIT,
                OP_LOGE(faInfo.opName, "When page attention enable and layout is BSH, "
                    "the axis H of %s(%ld) should not > %u.", inputName.c_str(), tempH, H_LIMIT),
                return ge::GRAPH_FAILED);
        }
    } else if (shapeDim == DIM_NUM_4) { // [blockNums, N, blockSize, D]
        tempN = tempShape.GetDim(DIM_NUM_1);
        tempBlockSize = tempShape.GetDim(DIM_NUM_2);
        tempD = tempShape.GetDim(DIM_NUM_3);

        OP_CHECK_IF(tempN != faInfo.n2Size,
            OP_LOGE(faInfo.opName, "When page attention enable, the axis N(%u) of kvCache should be %u",
                tempN, faInfo.n2Size),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(tempBlockSize != faInfo.blockSize,
            OP_LOGE(faInfo.opName, "When page attention enable, blocksize(%u) of kvCache should be %u",
                tempBlockSize, faInfo.blockSize),
            return ge::GRAPH_FAILED);
        
        if (faInfo.inputKvType == ge::DT_INT4) {
            OP_CHECK_IF(tempD != compareD,
                OP_LOGE(faInfo.opName, "When page attention enable, if input kv dataType is INT32, "
                    "the axis D of %s(%u) should be %u, "
                    "if input kv dataType is INT4, the axis D of %s(%u) should be %u",
                    inputName.c_str(), tempD / NUM8, compareD / NUM8,
                    inputName.c_str(), tempD, compareD),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(tempD != compareD,
                OP_LOGE(faInfo.opName, "When page attention enable, the axis D of %s(%u) should be %u",
                    inputName.c_str(), tempD, compareD),
                return ge::GRAPH_FAILED);
        }
    } else { // [blockNums, N, D1, blocksize, D0]
        tempN = tempShape.GetDim(DIM_NUM_1);
        tempD1 = tempShape.GetDim(DIM_NUM_2);
        tempBlockSize = tempShape.GetDim(DIM_NUM_3);
        tempD0 = tempShape.GetDim(DIM_NUM_4);
        const ge::DataType inputKvType = faInfo.inputKvType;
        uint32_t d0Size = NUM_16; // D0 = 16

        OP_CHECK_IF(tempN != faInfo.n2Size,
            OP_LOGE(faInfo.opName, "When page attention enable, the axis N of %s(%u) should be %u",
                inputName.c_str(), tempN, faInfo.n2Size),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(tempBlockSize != faInfo.blockSize,
            OP_LOGE(faInfo.opName, "When page attention enable, blocksize of %s(%u) should be %u",
                inputName.c_str(), tempBlockSize, faInfo.blockSize),
            return ge::GRAPH_FAILED);

        if (enableAntiQuant_) {
            d0Size = NUM_16;
            if (faInfo.inputKvType == ge::DT_INT4) {
                OP_CHECK_IF(tempD0 != d0Size,
                    OP_LOGE(faInfo.opName, "When PA_NZ enable, if input kv dataType is INT32, "
                        "the last dim (D0) of kvCache(%u) should be %u; "
                        "if input kv dataType is INT4, the last dim (D0) of %s(%u) should be %u",
                        inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size),
                    return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(tempD0 != d0Size,
                    OP_LOGE(faInfo.opName, "When PA_NZ enable, the last dim (D0) of %s(%u) should be %u",
                        inputName.c_str(), tempD0, d0Size),
                    return ge::GRAPH_FAILED);
            }
        } else {
            std::unordered_map<ge::DataType, float> typeSizeMap = {
                {ge::DT_FLOAT16, static_cast<float>(FLOAT16SIZE)},
                {ge::DT_BF16, static_cast<float>(BFLOAT16SIZE)},
                {ge::DT_INT8, static_cast<float>(INT8SIZE)},
                {ge::DT_HIFLOAT8, static_cast<float>(FLOAT8SIZE)},
                {ge::DT_FLOAT8_E4M3FN, static_cast<float>(FLOAT8SIZE)}};
            float dataTypeSizeValue = static_cast<float>(FLOAT16SIZE);
            auto inputTypeCheck = typeSizeMap.find(faInfo.inputKvType);
            if (inputTypeCheck != typeSizeMap.end()) {
                dataTypeSizeValue = inputTypeCheck->second;
            }

            if (enableFullQuant_ && inputName == "keyRope") {
                dataTypeSizeValue = static_cast<float>(BFLOAT16SIZE);
            }

            d0Size = BYTE_BLOCK / dataTypeSizeValue;
            OP_CHECK_IF(tempD0 != d0Size,
                OP_LOGE(faInfo.opName, "When PA_NZ enable, the last dim (D0) of %s(%u) should be %u",
                    inputName.c_str(), tempD0, d0Size),
                return ge::GRAPH_FAILED);
        }

        OP_CHECK_IF(tempD1 != compareD / d0Size,
            OP_LOGE(faInfo.opName, "When PA_NZ enable, the third dim (D1) of %s(%u) should be %u",
                inputName.c_str(), tempD1, compareD / d0Size),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check input query dtype
ge::graphStatus PagedAttentionChecker::CheckQDtypeSupport(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.inputQType == ge::DT_INT8 && faInfo.ropeMode != RopeMode::ROPE_SPLIT,
        OP_LOGE(faInfo.opName, "When page attention enable, the datatype(%s) of query should not be INT8 except for MLA fullquant.",
                DataTypeToSerialString(faInfo.inputQType).c_str()),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockTableShape(const FaTilingInfo &faInfo)
{
    if (faInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape blockTableShape = faInfo.opParamInfo.blockTable.tensor->GetStorageShape();

    // check dim num
    OP_CHECK_IF(blockTableShape.GetDimNum() != 2,
        OP_LOGE(faInfo.opName, "When page attention enable, the dim num(%zu) of blockTable should be 2.",
                blockTableShape.GetDimNum()),
            return ge::GRAPH_FAILED);

    // check blockTable each dim cannot be 0
    OP_CHECK_IF(blockTableShape.GetShapeSize() == 0,
        OP_LOGE(faInfo.opName, "When page attention enable, blockTable each dim can not be 0, now is [%ld, %ld].",
            blockTableShape.GetDim(0), blockTableShape.GetDim(1)),
        return ge::GRAPH_FAILED);
    
    if (faInfo.opParamInfo.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // 每个batch的 blockNum 小于 blocktable dim2
    int64_t maxBlockNumPerBatch = GetMaxBlockNumPerBatch(faInfo);

    OP_CHECK_IF(((blockTableShape.GetDim(0) != faInfo.bSize) || (blockTableShape.GetDim(1) < maxBlockNumPerBatch)),
        OP_LOGE(faInfo.opName,
            "When page attention enable, block table shape should be [%u, >=%ld], now is [%ld, %ld].",
            faInfo.bSize, maxBlockNumPerBatch, blockTableShape.GetDim(0), blockTableShape.GetDim(1)),
        return ge::GRAPH_FAILED);

    // check key cache shape
    if (ge::GRAPH_SUCCESS != CheckPACacheShape(faInfo, faInfo.opParamInfo.key.shape->GetStorageShape(), "key") ||
        ge::GRAPH_SUCCESS != CheckPACacheShape(faInfo, faInfo.opParamInfo.value.shape->GetStorageShape(), "value")) {
        return ge::GRAPH_FAILED;
    }

    // check rope cache shape
    if (faInfo.ropeMode == RopeMode::ROPE_SPLIT) {
        if (ge::GRAPH_SUCCESS != CheckPACacheShape(faInfo, faInfo.opParamInfo.keyRope.tensor->GetStorageShape(),
            "keyRope")) {
            return ge::GRAPH_FAILED;
        }
    }

    // warning: S2 <= 20M
    if (maxBlockNumPerBatch * faInfo.blockSize > S_LIMIT) {
        OP_LOGW(faInfo.opName,
            "When page attention enable, sequence length(%ld) of kv should <= 20M.",
            maxBlockNumPerBatch * faInfo.blockSize);
    }
    return ge::GRAPH_SUCCESS;
}

// check blocksize
ge::graphStatus PagedAttentionChecker::CheckBlockSize(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.opParamInfo.blockSize == nullptr,
        OP_LOGE(faInfo.opName,
            "When page attention enable, blockSize shuld not be null."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.blockSize == 0,
        OP_LOGE(faInfo.opName,
            "When page attention enable, blockSize should not be 0."),
        return ge::GRAPH_FAILED);

    if (enableNonQuant_) { // 非量化
        if (faInfo.ropeMode != RopeMode::NO_ROPE) { // MLA场景 [16, 1024]且16对齐
            OP_CHECK_IF(faInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT ||
                faInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 || faInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0,
                OP_LOGE(faInfo.opName,
                    "In no quant MLA scenario, when page attention enable, blockSize(%d) should be a multiple of "
                    "%u, and should be in range of [%u, %u].",
                    faInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_16, BLOCK_SIZE_ALIGN_SIZE_16, BLOCK_SIZE_MAX_FOR_NO_QUANT),
                return ge::GRAPH_FAILED);
        } else if (faInfo.qkHeadDim == NUM_64 || faInfo.qkHeadDim == NUM_128) { // GQA D =64/128场景 [16, 1024]且16对齐
            OP_CHECK_IF(faInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || 
                faInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 || faInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0,
                OP_LOGE(faInfo.opName,
                    "In no quant GQA(D = %u) scenario, when page attention enable, blockSize(%d) should be a multiple "
                    "of %u, and should be in range of [%u, %u].",
                    faInfo.qkHeadDim, faInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_16,
                    BLOCK_SIZE_ALIGN_SIZE_16, BLOCK_SIZE_MAX_FOR_NO_QUANT),
                return ge::GRAPH_FAILED);
        } else {
            // GQA D != 64/128, QS > 1 [128, 512]且128对齐
            OP_CHECK_IF((faInfo.s1Size > NUM1) && (faInfo.blockSize > BLOCK_SIZE_MAX || 
                faInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_128 || faInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_128 != 0),
                OP_LOGE(faInfo.opName,
                    "In no quant GQA (QS > 1) scenario, when page attention enable, blockSize(%d) should be a multiple "
                    "of %u, and should be in range of [%u, %u].",
                    faInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_128, BLOCK_SIZE_ALIGN_SIZE_128, BLOCK_SIZE_MAX),
                return ge::GRAPH_FAILED);
            // GQA D != 64/128, QS = 1 [16, 512]且16对齐
            OP_CHECK_IF((faInfo.s1Size == NUM1) && (faInfo.blockSize > BLOCK_SIZE_MAX || 
                faInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 || faInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0),
                OP_LOGE(faInfo.opName,
                    "In no quant GQA (QS = 1) scenario, when page attention enable, blockSize(%d) should be a multiple "
                    "of %u, and should be in range of [%u, %u].",
                    faInfo.s1Size, faInfo.blockSize, BLOCK_SIZE_ALIGN_SIZE_16,
                    BLOCK_SIZE_ALIGN_SIZE_16, BLOCK_SIZE_MAX),
                return ge::GRAPH_FAILED);
        }
    } else if (enableAntiQuant_) {
        std::unordered_map<ge::DataType, float> typeSizeMap = {{ge::DT_FLOAT16, static_cast<float>(FLOAT16SIZE)},
                                                               {ge::DT_BF16, static_cast<float>(BFLOAT16SIZE)},
                                                               {ge::DT_INT8, static_cast<float>(INT8SIZE)},
                                                               {ge::DT_HIFLOAT8, static_cast<float>(FLOAT8SIZE)},
                                                               {ge::DT_FLOAT8_E4M3FN, static_cast<float>(FLOAT8SIZE)},
                                                               {ge::DT_INT4, INT4SIZE},
                                                               {ge::DT_FLOAT4_E2M1, FLOAT4SIZE}};
        float dataTypeSizeValue = static_cast<float>(FLOAT16SIZE);
        auto inputTypeCheck = typeSizeMap.find(faInfo.inputKvType);
        if (inputTypeCheck != typeSizeMap.end()) {
            dataTypeSizeValue = inputTypeCheck->second;
        }
        uint32_t blockSizeAlign = static_cast<uint32_t>(BYTE_BLOCK / dataTypeSizeValue);

        // 伪量化, 与Dtype相关
        OP_CHECK_IF((faInfo.blockSize > BLOCK_SIZE_MAX || 
            faInfo.blockSize < blockSizeAlign || faInfo.blockSize % blockSizeAlign != 0),
            OP_LOGE(faInfo.opName,
                "In antiquant scenario, when page attention enable, if kvCache datatype is %s, "
                "blockSize(%d) should be a multiple of %u, and should be in range of [%u, %u].",
                DataTypeToSerialString(faInfo.inputKvType).c_str(),
                faInfo.blockSize, blockSizeAlign, blockSizeAlign, BLOCK_SIZE_MAX),
            return ge::GRAPH_FAILED);
    } else { // 全量化
        OP_CHECK_IF(faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 && faInfo.blockSize != NUM_128,
            OP_LOGE(faInfo.opName,
                "In MLA fullquant scenario, when page attention enable, blockSize(%d) should be 128.",
                faInfo.blockSize),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(faInfo.mlaMode == MlaMode::NO_MLA && (faInfo.blockSize > BLOCK_SIZE_MAX || 
            faInfo.blockSize < NUM_32 || faInfo.blockSize % NUM_32 != 0),
            OP_LOGE(faInfo.opName,
                "In per-tensor quant scenario, when page attention enable, "
                "blockSize(%d) should be a multiple of %u, and should be in range of [%u, %u].",
                faInfo.blockSize, NUM_32, NUM_32, BLOCK_SIZE_MAX),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckPADimNum(const FaTilingInfo &faInfo)
{
    if (!enableFullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const string inputLayout = faInfo.opParamInfo.layOut;
    const uint32_t dimNum = faInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    if (inputLayout == "BSH" || inputLayout == "BSND") {
        OP_CHECK_IF(dimNum == 4,
            OP_LOGE(faInfo.opName,
                "When page attention enable and input layout is %s, PA BnNBsD is not supported.", inputLayout.c_str()),
            return ge::GRAPH_FAILED);
    }
    // if (faInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
    //     OP_CHECK_IF(faInfo.kvLayout == FaLayout::BnBsH && (faInfo.qLayout != FaLayout::BSH && faInfo.qLayout != FaLayout::BSND &&
    //                     faInfo.qLayout != FaLayout::BNSD && faInfo.qLayout != FaLayout::TND && faInfo.qLayout != FaLayout::NTD),
    //         OP_LOGE(faInfo.opName, "In %s %s situation, the key/value's layout is BnBsH, query layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
    //             QuantModeToSerialString(faInfo.quantMode).c_str(), SituationToSerialString(faInfo.ropeMode).c_str(), LayoutToSerialString(faInfo.qLayout).c_str()),
    //         return ge::GRAPH_FAILED);

    //     OP_CHECK_IF(faInfo.kvLayout == FaLayout::BnNBsD && (faInfo.qLayout != FaLayout::BSH && faInfo.qLayout != FaLayout::BSND &&
    //                     faInfo.qLayout != FaLayout::BNSD && faInfo.qLayout != FaLayout::TND && faInfo.qLayout != FaLayout::NTD),
    //         OP_LOGE(faInfo.opName, "In %s %s situation, the key/value's layout is BnNBsD, query layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
    //             QuantModeToSerialString(faInfo.quantMode).c_str(), SituationToSerialString(faInfo.ropeMode).c_str(), LayoutToSerialString(faInfo.qLayout).c_str()),
    //         return ge::GRAPH_FAILED);

    //     OP_CHECK_IF(faInfo.kvLayout == FaLayout::NZ && (faInfo.qLayout != FaLayout::BSH && faInfo.qLayout != FaLayout::BSND &&
    //                     faInfo.qLayout != FaLayout::BNSD && faInfo.qLayout != FaLayout::TND && faInfo.qLayout != FaLayout::NTD),
    //         OP_LOGE(faInfo.opName, "In %s %s situation, the key/value's layout is BnNBsD, query layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
    //             QuantModeToSerialString(faInfo.quantMode).c_str(), SituationToSerialString(faInfo.ropeMode).c_str(), LayoutToSerialString(faInfo.qLayout).c_str()),
    //         return ge::GRAPH_FAILED);
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckBlockTableDtype(faInfo) ||
        ge::GRAPH_SUCCESS != CheckQDtypeSupport(faInfo)) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckBlockTableExistence(faInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureExistence(faInfo) ||
        ge::GRAPH_SUCCESS != CheckSeqLengthKVExistence(faInfo)) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckMaskShape(faInfo)) {
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckBlockSize(faInfo) ||
        ge::GRAPH_SUCCESS != CheckPADimNum(faInfo) ||
        ge::GRAPH_SUCCESS != CheckBlockTableShape(faInfo)) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling