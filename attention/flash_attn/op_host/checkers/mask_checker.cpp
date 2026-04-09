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
 * \file mask_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "mask_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// CheckSinglePara
ge::graphStatus MaskChecker::CheckDtypeAndFormat(const FaTilingInfo &faInfo)
{
    // AttentionMask data type must be int8/uint8/bool, and data format must be ND/NCHW/NHWC/NCDHW.
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.attenMask.desc, ATTEN_MASK_NAME)) {
        OP_LOGE(faInfo.opName, "AttentionMask data type must be int8/uint8/bool!");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.attenMask.desc, ATTEN_MASK_NAME)) {
        OP_LOGE(faInfo.opName, "Data format must be ND/NCHW/NHWC/NCDHW!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSparseMode(const FaTilingInfo &faInfo)
{
    // SparseMode only supports 0/1/2/3/4.
    const std::vector<int32_t> sparseModeList = {SPARSE_MODE_NO_MASK, SPARSE_MODE_ALL_MASK, SPARSE_MODE_LEFT_UP,
                                                 SPARSE_MODE_RIGHT_DOWN, SPARSE_MODE_BAND};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(faInfo.sparseMode, sparseModeList),
                OP_LOGE(faInfo.opName, "SparseMode only supports 0/1/2/3/4, but got %u", faInfo.sparseMode),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus MaskChecker::CheckNoQuantIFAMLA(const FaTilingInfo &faInfo)
{
    // For IFA MLA, input sparse mode only supports 0/3/4.
    enableIFAMLA = (faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512);
    if (enableIFAMLA) {
        OP_CHECK_IF(
            faInfo.sparseMode != SPARSE_MODE_NO_MASK && faInfo.sparseMode != SPARSE_MODE_RIGHT_DOWN &&
                faInfo.sparseMode != SPARSE_MODE_BAND,
            OP_LOGE(faInfo.opName, "Only support sparse(%d) 0/3/4 when ifa mla is enable!", faInfo.sparseMode),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFullQuantIFAMLA(const FaTilingInfo &faInfo)
{
    // For IFA MLA, input sparse mode %d, sparse 0 without mask is supported only when sequence length is 1,
    // and sparse 3 with mask or sparse 0 without mask is supported only when sequence length > 1.
    enableIFAMLA = (faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512);
    if (enableIFAMLA) {
        if (faInfo.s1Size == 1U) {
            OP_CHECK_IF(!((faInfo.sparseMode == SPARSE_MODE_NO_MASK) && (!faInfo.attenMaskFlag)),
                        OP_LOGE(faInfo.opName,
                                "Only support sparse 0 without mask when ifa mla and query's sequence length is 1, "
                                "input sparse mode is %d and there has%smask",
                                faInfo.sparseMode, faInfo.attenMaskFlag ? " " : " no "),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(faInfo.inputQType == ge::DT_FLOAT8_E4M3FN &&
                        !(((faInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) && (faInfo.attenMaskFlag)) ||
                          ((faInfo.sparseMode == SPARSE_MODE_NO_MASK) && (!faInfo.attenMaskFlag))),
                        OP_LOGE(faInfo.opName,
                                "Only support sparse 3 with mask, or sparse 0 without mask when ifa mla and "
                                "query's sequence length is > 1, and input datatype is FLOAT8_E4M3, "
                                "input sparse mode is %d and there has%smask",
                                faInfo.sparseMode, faInfo.attenMaskFlag ? " " : " no "),
                        return ge::GRAPH_FAILED);
            
            OP_CHECK_IF(faInfo.inputQType == ge::DT_INT8 &&
                        !((faInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) && (faInfo.attenMaskFlag)),
                        OP_LOGE(faInfo.opName,
                                "Only support sparse 3 with mask and "
                                "query's sequence length is > 1,and input datatype is INT8, "
                                "input sparse mode is %d and there has%smask",
                                faInfo.sparseMode, faInfo.attenMaskFlag ? " " : " no "),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckQKVDDifferent(const FaTilingInfo &faInfo)
{
    // Only support sparse mode 0/2/3 when query and key headdim is not equal to value headdim.
    OP_CHECK_IF(
        faInfo.isQKVDDifferent && faInfo.sparseMode != SPARSE_MODE_NO_MASK &&
            faInfo.sparseMode != SPARSE_MODE_LEFT_UP && faInfo.sparseMode != SPARSE_MODE_RIGHT_DOWN,
        OP_LOGE(faInfo.opName, "Not support sparse mode %d when query and key headdim is not equal to value headdim.",
                faInfo.sparseMode),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckPretokenAndNexttoken(const FaTilingInfo &faInfo)
{
    // In PFA mode, the values of pretoken and nexttoken must ensure the mask range remains valid.
    isIFAFlag = (enableAntiQuant_) && (faInfo.s1Size == 1);
    ge::DataType outputType = faInfo.opParamInfo.attenOut.desc->GetDataType();
    if (isIFAFlag) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((faInfo.nextToken * (-1)) > faInfo.preToken,
                OP_LOGE(faInfo.opName,
                        "Nexttoken line should be higher than pretoken line, preTokens = %ld, nextTokens = %ld.",
                        faInfo.preToken, faInfo.nextToken),
                return ge::GRAPH_FAILED);
    // Check the specific conditions that pretoken and nexttoken must satisfy under the band mode.
    OP_CHECK_IF(
        (faInfo.antiQuantFlag && faInfo.sparseMode == SPARSE_MODE_BAND && outputType == ge::DT_INT8 &&
         ((faInfo.preToken < 0) || faInfo.nextToken < 0)),
        OP_LOGE(faInfo.opName,
                "When output type is int8, sparse mode = 4, preTokens (%ld) or nextTokens (%ld) cannot be negative.",
                faInfo.preToken, faInfo.nextToken),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckIFADimAndShape(const FaTilingInfo &faInfo)
{
    uint32_t minAttenMaskSize = 0;
    std::string layoutStr(faInfo.opParamInfo.layOut);
    const gert::Tensor *maskShape = faInfo.opParamInfo.attenMask.tensor;
    size_t attenMaskDim = maskShape->GetStorageShape().GetDimNum();
    uint32_t attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
    uint32_t attenMaskSize = maskShape->GetStorageShape().GetDim(maskShape->GetStorageShape().GetDimNum() - 1);
    if (faInfo.pageAttentionFlag) {
        minAttenMaskSize = faInfo.s2Size;
    } else {
        minAttenMaskSize = faInfo.maxActualseq;
    }
    if (attenMaskDim == MASK_DIM_SS) {
        if (faInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B && (faInfo.sparseMode == SPARSE_MODE_NO_MASK || faInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
            if (faInfo.ropeMode == RopeMode::NO_ROPE ) {
                const std::vector<std::string> layoutSupportList = {
                    "BSH", "BSND", "BNSD", "BNSD_BSND",
                };
                OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutStr) == layoutSupportList.end(),
                    OP_LOGE(faInfo.opName,
                        "In gqa noquant situation, rope not exits and qkHeadDim = vHeadDim, when sparseMode = 0 or 1, "
                        "two dim mask only support for layout BSH,BSND,BNSD,BNSD_BSND, but got %s",
                        layoutStr.c_str()),
                    return ge::GRAPH_FAILED);
            } else {
                OP_LOGE(faInfo.opName,
                        "In gqa noquant situation, rope exits or qkHeadDim != vHeadDim, when sparseMode = 0 or 1, two dim mask is not supported.");
                return ge::GRAPH_FAILED;
            }
        } else {
            OP_LOGE(faInfo.opName,
                "The current dimension of the mask is 2. "
                "Please use 3D mask \[B,QS,KVS\]\/\[1,QS,KVS\] or 4D mask \[B,1,QS,KVS\]\/\[1,1,QS,KVS\].");
            return ge::GRAPH_FAILED;
        }
    } else {
        bool checkMask = (attenMaskSize >= minAttenMaskSize) &&
                    ((attenMaskBatch == faInfo.bSize) || (attenMaskBatch == DIM_NUM_1));
        OP_CHECK_IF(!checkMask,
                    OP_LOGE(faInfo.opName,
                            "AttenMask batch(%u) must be %u or 1, attenMask KV_S(%u) must be larger than or equal to"
                            "the second dimension of blockTable * blockSize(%u), please check",
                            attenMaskBatch, faInfo.bSize, attenMaskSize, minAttenMaskSize),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::GetMaskInfo(const FaTilingInfo &faInfo, MaskInfo &maskInfo)
{
    const gert::Tensor *maskShape = faInfo.opParamInfo.attenMask.tensor;
    size_t attenMaskDim = maskShape->GetStorageShape().GetDimNum();
    if (attenMaskDim == MASK_DIM_SS) {
       if ((faInfo.sparseMode == SPARSE_MODE_NO_MASK || faInfo.sparseMode == SPARSE_MODE_ALL_MASK) && faInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) {
            OP_LOGE(faInfo.opName,
                    "Attenmask does not support inputs with dim 2 when sparse mode = %u. "
                    "Please use 3D mask \[B,QS,KVS\]\/\[1,QS,KVS\] or 4D mask \[B,1,QS,KVS\]\/\[1,1,QS,KVS\].",
                    faInfo.sparseMode);
            return ge::GRAPH_FAILED;
        }
        else {
            if (faInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B && (faInfo.sparseMode == SPARSE_MODE_NO_MASK || faInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
                if (faInfo.ropeMode == RopeMode::NO_ROPE) {
                    const std::vector<std::string> layoutSupportList = {
                        "BSH", "BSND", "BNSD", "BNSD_BSND",
                    };
                    std::string layoutStr(faInfo.opParamInfo.layOut);
                    std::string layout = layoutStr;
                    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
                        OP_LOGE(faInfo.opName,
                            "In gqa no quant situation, rope not exits and qkHeadDim = vHeadDim, when sparseMode = 0 or 1, "
                            "two dim mask only support for layout BSH,BSND,BNSD,BNSD_BSND, but got %s",
                            layout.c_str()),
                        return ge::GRAPH_FAILED);
                } else {
                    OP_LOGE(faInfo.opName,
                            "In gqa no quant situation, rope exits or qkHeadDim != vHeadDim, when sparseMode = 0 or 1, two dim mask is not supported.");
                    return ge::GRAPH_FAILED;
                }
            }
        }
        maskInfo.attenMaskQSize = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
        maskInfo.attenMaskSize = maskShape->GetStorageShape().GetDim(DIM_NUM_1);
        maskInfo.strMaskShape = std::to_string(maskInfo.attenMaskQSize) + ", " + std::to_string(maskInfo.attenMaskSize);
    } else if (attenMaskDim == MASK_DIM_BSS) {
        maskInfo.attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
        maskInfo.attenMaskQSize = maskShape->GetStorageShape().GetDim(DIM_NUM_1);
        maskInfo.attenMaskSize =
            maskShape->GetStorageShape().GetDim(DIM_NUM_2);  // 2: When the dim is 3, the second dimension is S2.
        maskInfo.strMaskShape = std::to_string(maskInfo.attenMaskBatch) + ", " +
                                std::to_string(maskInfo.attenMaskQSize) + ", " + std::to_string(maskInfo.attenMaskSize);
    } else if (attenMaskDim == MASK_DIM_B1SS) {
        maskInfo.attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
        maskInfo.attenMaskN = maskShape->GetStorageShape().GetDim(DIM_NUM_1);
        maskInfo.attenMaskQSize =
            maskShape->GetStorageShape().GetDim(DIM_NUM_2);  // 2: When the dim is 4, the second dimension is S1.
        maskInfo.attenMaskSize = maskShape->GetStorageShape().GetDim(DIM_NUM_3);  // 3:The third dimension is S2.
        maskInfo.strMaskShape = std::to_string(maskInfo.attenMaskBatch) + ", " + std::to_string(maskInfo.attenMaskN) +
                                ", " + std::to_string(maskInfo.attenMaskQSize) + ", " +
                                std::to_string(maskInfo.attenMaskSize);
    } else {
        OP_LOGE(faInfo.opName, "AttenMask dim(%zu) must be 2 or 3 or 4!", attenMaskDim);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckDimAndShape(const FaTilingInfo &faInfo)
{
    // In PFA mode, the attenmask dimensions must be 2/3/4.
    // The allowed shape specifications for attenmask vary depending on the sparse mode.
    if ((!faInfo.attenMaskFlag) && (faInfo.sparseMode != SPARSE_MODE_NO_MASK)) {
        OP_LOGE(faInfo.opName, "when sparse_mode is %d, it not 0, atten_mask should not be null.",
                faInfo.sparseMode);
        return ge::GRAPH_FAILED;
    }
    if ((faInfo.isMaxWorkspace && faInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) || !faInfo.attenMaskFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (faInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B && (faInfo.sparseMode == SPARSE_MODE_NO_MASK || faInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
        if (faInfo.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }
    }
    isIFAFlag = (faInfo.antiQuantFlag) && (faInfo.s1Size == 1);
    if (isIFAFlag) {
        return CheckIFADimAndShape(faInfo);
    }

    MaskInfo maskInfo;
    if (GetMaskInfo(faInfo, maskInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    bool checkMask = false;
    if (faInfo.sparseMode == SPARSE_MODE_NO_MASK || faInfo.sparseMode == SPARSE_MODE_ALL_MASK) {
        checkMask = (maskInfo.attenMaskQSize >= faInfo.s1Size) && (maskInfo.attenMaskSize >= faInfo.s2Size) &&
                    (maskInfo.attenMaskBatch == NUM1 || maskInfo.attenMaskBatch == faInfo.bSize) &&
                    (static_cast<uint32_t>(maskInfo.attenMaskN) == NUM1);
    } else if ((faInfo.sparseMode == SPARSE_MODE_LEFT_UP) || (faInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
               (faInfo.sparseMode == SPARSE_MODE_BAND)) {
        checkMask = (maskInfo.attenMaskBatch == NUM1) && (static_cast<uint32_t>(maskInfo.attenMaskN) == NUM1) &&
                    (maskInfo.attenMaskQSize == SPARSE_OPTIMIZE_ATTENTION_SIZE) &&
                    (maskInfo.attenMaskSize == SPARSE_OPTIMIZE_ATTENTION_SIZE);
    }

    if (faInfo.sparseMode == SPARSE_MODE_NO_MASK || faInfo.sparseMode == SPARSE_MODE_ALL_MASK) {
        OP_CHECK_IF(
            !checkMask,
            OP_LOGE(faInfo.opName,
                    "attenMask batch(%u) must be 1 or %u, attenMask Q_S(%u) must be larger than or equal to sQ(%u),"
                    "attenMask KV_S(%u) must be larger than or equal to sK(%u), please check",
                    maskInfo.attenMaskBatch, faInfo.bSize, maskInfo.attenMaskQSize, faInfo.s1Size,
                    maskInfo.attenMaskSize, faInfo.s2Size),
            return ge::GRAPH_FAILED);
    }
    if (((faInfo.sparseMode == SPARSE_MODE_LEFT_UP) || (faInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
         (faInfo.sparseMode == SPARSE_MODE_BAND)) &&
        !checkMask) {
        OP_LOGE(faInfo.opName,
                "attenMask shape must be (2048, 2048) or (1, 2048, 2048) or (1, 1, 2048, 2048) when sparse mode = %u.",
                faInfo.sparseMode);
        OP_LOGE(faInfo.opName, "attenMask shape is (%s).", maskInfo.strMaskShape.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeAndFormat(faInfo) || ge::GRAPH_SUCCESS != CheckSparseMode(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckNoQuantIFAMLA(faInfo) || ge::GRAPH_SUCCESS != CheckQKVDDifferent(faInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFullQuantIFAMLA(faInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckPretokenAndNexttoken(faInfo) || ge::GRAPH_SUCCESS != CheckDimAndShape(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
