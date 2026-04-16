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
 * \file shape_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "shape_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// 公共校验函数
ge::graphStatus ShapeChecker::CheckInputFormat(const FaTilingInfo &faInfo)
{
    if (CheckFormatSupport(faInfo.opParamInfo.query.desc, "query") != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(faInfo.opParamInfo.key.desc, "key") != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(faInfo.opParamInfo.value.desc, "value") != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(faInfo.opParamInfo.attenOut.desc, "attentionOut") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckParaExistenceImpl(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.opParamInfo.query.desc == nullptr || faInfo.opParamInfo.query.shape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Query input is null pointer!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.key.desc == nullptr || faInfo.opParamInfo.key.shape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Key input is null pointer!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.value.desc == nullptr || faInfo.opParamInfo.value.shape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Value input is null pointer!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.attenOut.desc == nullptr || faInfo.opParamInfo.attenOut.shape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "AttentionOut is null pointer!"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckDtypeCommon(const gert::CompileTimeTensorDesc *desc,
    const std::string &name, std::map<std::string, std::vector<ge::DataType>> dataMap)
{
    if (desc != nullptr) {
        const auto& it = dataMap.find(name);
        OP_CHECK_IF(it == dataMap.end(),
            OP_LOGE("FA", "%s datatype support list should be specify in map", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(expectDtypeList.begin(), expectDtypeList.end(),
         desc->GetDataType()) == expectDtypeList.end(),
            "", // LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), //公共打印函数
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckPAKeyValue(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *keyShape = faInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = faInfo.opParamInfo.value.shape;
    uint32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    uint32_t keyBlockNum = faInfo.totalBlockNum;
    uint32_t keyHeadNum = faInfo.n2Size;
    uint32_t keyBlockSize = faInfo.blockSize;
    uint32_t keyHeadDim = faInfo.qkHeadDim;
    uint32_t keyH = faInfo.qkHeadDim * faInfo.n2Size;
    uint32_t keyD0 = 16;
    uint32_t keyD1 = faInfo.qkHeadDim / 16; // 当前PA NZ场景D 16

    uint32_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    uint32_t valueBlockNum = faInfo.totalBlockNum;
    uint32_t valueHeadNum = faInfo.n2Size;
    uint32_t valueBlockSize = faInfo.blockSize;
    uint32_t valueHeadDim = faInfo.vHeadDim;
    uint32_t valueH = faInfo.vHeadDim * faInfo.n2Size;
    uint32_t valueD0 = 16;
    uint32_t valueD1 = faInfo.vHeadDim / 16; // 当前PA NZ场景D 16
    OP_CHECK_IF(keyDimNum != valueDimNum,
                OP_LOGE(faInfo.opName,
                        "The dim num of key should be consistent with the dim num of value,"
                        "but current dim num of key is %zu, dim num of value is %zu.",
                        keyDimNum, valueDimNum),
                return ge::GRAPH_FAILED);

    if (keyDimNum == 3) {                    // BBH
        keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        keyBlockSize = keyShape->GetStorageShape().GetDim(1);
        keyH = keyShape->GetStorageShape().GetDim(2);
        valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        valueBlockSize = valueShape->GetStorageShape().GetDim(1);
        valueH = valueShape->GetStorageShape().GetDim(2);
    } else if (keyDimNum == 4) {            // BNBD
        keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        keyHeadNum = keyShape->GetStorageShape().GetDim(1);
        keyBlockSize = keyShape->GetStorageShape().GetDim(2);
        keyHeadDim = keyShape->GetStorageShape().GetDim(3);
        valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        valueHeadNum = valueShape->GetStorageShape().GetDim(1);
        valueBlockSize = valueShape->GetStorageShape().GetDim(2);
        valueHeadDim = valueShape->GetStorageShape().GetDim(3);
    } else if (keyDimNum == 5) {           // BND1BD0
        keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        keyHeadNum = keyShape->GetStorageShape().GetDim(1);
        keyD1 = keyShape->GetStorageShape().GetDim(2);
        keyBlockSize = keyShape->GetStorageShape().GetDim(3);
        keyD0 = keyShape->GetStorageShape().GetDim(4);

        valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        valueHeadNum = valueShape->GetStorageShape().GetDim(1);
        valueD1 = valueShape->GetStorageShape().GetDim(2);
        valueBlockSize = valueShape->GetStorageShape().GetDim(3);
        valueD0 = valueShape->GetStorageShape().GetDim(4);
    } else {
        OP_LOGE(faInfo.opName, "The dim num of key only support [%u, %u], but current is %zu.",
            INPUT_KV_SHAPE_MIN_DIMS, INPUT_KV_SHAPE_MAX_DIMS, keyDimNum);
        return ge::GRAPH_FAILED;
    }

    // if (faInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B && faInfo.kvLayout == FaLayout::NZ && faInfo.ropeMode == RopeMode::NO_ROPE) {
    //     const std::vector<std::int32_t> nzNoRopeDSupportList = {64, 128};
    //     if (std::find(nzNoRopeDSupportList.begin(), nzNoRopeDSupportList.end(), keyHeadDim) ==
    //             nzNoRopeDSupportList.end() ||
    //         std::find(nzNoRopeDSupportList.begin(), nzNoRopeDSupportList.end(), valueHeadDim) ==
    //             nzNoRopeDSupportList.end()) {
    //         OP_LOGE(faInfo.opName,
    //                 "In %s %s situation, when the dim of key&value is 5, and the headDim shared by query and key is "
    //                 "equal to that of value. The headDim of query|key|value should be 64 | "
    //                 "128, but got valueHeadDim:%u, queryHeadDim and keyHeadDim:%u",
    //                 QuantModeToSerialString(faInfo.quantMode).c_str(), SituationToSerialString(faInfo.ropeMode).c_str(), valueHeadDim,
    //                 keyHeadDim);
    //         return ge::GRAPH_FAILED;
    //     }
    // }

    OP_CHECK_IF((keyDimNum == 5) && ((keyBlockNum != valueBlockNum) || (keyHeadNum != valueHeadNum) || 
        ((keyD1 != valueD1) && (faInfo.mlaMode != MlaMode::ROPE_COMBINE_D128)) || (keyBlockSize != valueBlockSize) ||
        ((keyD0 != valueD0) && (faInfo.mlaMode != MlaMode::ROPE_COMBINE_D128))), 
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
            "The dim num of key and value are inconsistent when PA enable. key shape [%ld, %ld, %ld, %ld, %ld],"
            " value shape [%ld, %ld, %ld, %ld, %ld].",
            keyBlockNum, keyHeadNum, keyD1, keyBlockSize, keyD0, valueBlockNum, valueHeadNum, valueD1, valueBlockSize, valueD0),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((keyDimNum == 4) && ((keyBlockNum != valueBlockNum) || (keyHeadNum != valueHeadNum) || 
        (keyBlockSize != valueBlockSize) || ((keyHeadDim != valueHeadDim) && (faInfo.mlaMode != MlaMode::ROPE_COMBINE_D128))), 
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
            "the dim num of key and value are inconsistent when PA enable. key shape [%ld,%ld,%ld,%ld],"
            " value shape [%ld,%ld,%ld,%ld].",
            keyBlockNum, keyHeadNum, keyBlockSize, keyHeadDim, valueBlockNum, valueHeadNum, valueBlockSize, valueHeadDim),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((keyDimNum == 3) && ((keyBlockNum != valueBlockNum) || (keyBlockSize != valueBlockSize) || 
        ((keyH != valueH) && (faInfo.mlaMode != MlaMode::ROPE_COMBINE_D128))), OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
            "the dim num of key and value are inconsistent when PA enable. key shape [%ld,%ld,%ld], value shape [%ld,%ld,%ld].",
            keyBlockNum, keyBlockSize, keyH, valueBlockNum, valueBlockSize, valueH),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyHeadDim != faInfo.qkHeadDim,
                OP_LOGE(faInfo.opName, "The headDim of key is %u, it should be %u.", keyHeadDim, faInfo.qkHeadDim),
                ge::GRAPH_FAILED);
    OP_CHECK_IF(valueHeadDim != faInfo.vHeadDim,
                OP_LOGE(faInfo.opName, "The headDim of value is %u, it should be %u.", valueHeadDim, faInfo.vHeadDim),
                ge::GRAPH_FAILED);
    OP_CHECK_IF(keyH != faInfo.qkHeadDim * faInfo.n2Size,
                OP_LOGE(faInfo.opName, "The H of key is %u, it should be %u.", keyH,
                faInfo.qkHeadDim * faInfo.n2Size),
                ge::GRAPH_FAILED);
    OP_CHECK_IF(valueH != faInfo.vHeadDim * faInfo.n2Size,
                OP_LOGE(faInfo.opName, "The H of value is %u, it should be %u.",
                valueH, faInfo.qkHeadDim * faInfo.n2Size),
                ge::GRAPH_FAILED);
    OP_CHECK_IF(keyD0 != 16 || valueD0 != 16,
                OP_LOGE(faInfo.opName, "The D0 of key is %u, the D0 of value is %u, "
                        "they both should be 16", keyD0, valueD0),
                ge::GRAPH_FAILED);
    OP_CHECK_IF(keyD1 != faInfo.qkHeadDim  / 16,
                OP_LOGE(faInfo.opName, "The D1 of key is %u, it should be %u.", keyD1, faInfo.qkHeadDim / 16),
                ge::GRAPH_FAILED);
    OP_CHECK_IF(valueD1 != faInfo.vHeadDim / 16,
                OP_LOGE(faInfo.opName, "The D1 of value is %u, it should be %u.", valueD1, faInfo.vHeadDim / 16),
                ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool ShapeChecker::CheckEmptyTensorList(const FaTilingInfo &faInfo)
{
    for (int64_t tmpIdx = 0; tmpIdx < faInfo.kCache.size(); ++tmpIdx) {
        if (faInfo.kCache[tmpIdx]->GetStorageShape().GetShapeSize() != 0) {
            return false;
        }
        if (faInfo.vCache[tmpIdx]->GetStorageShape().GetShapeSize() != 0) {
            return false;
        }
    }
    return true;
}

bool ShapeChecker::CheckNormalTensorList(const FaTilingInfo &faInfo)
{
    std::string layoutStr(faInfo.opParamInfo.layoutOut);
    if (layoutStr == "BSH") { // check all H across batches and KVs are the same under BSH layout
        auto standardKH = faInfo.kCache[0]->GetStorageShape().GetDim(2);
        auto standardVH = faInfo.vCache[0]->GetStorageShape().GetDim(2);
        int64_t tmpNKv = (faInfo.n2Size != 0) ? faInfo.n2Size : faInfo.n1Size;
        int64_t keyRopeS = 0;
        if (faInfo.opParamInfo.keyRope.tensor != nullptr) {
            keyRopeS = faInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(1);
            OP_CHECK_IF(faInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0) != faInfo.kCache.size(),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Batch of Key(%ld) do NOT equal to Batch of KeyRope(%ld) under tensorlist mode!", 
                faInfo.kCache.size(), faInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0)),
                return false);
        }

        for (int64_t tmpIdx = 0; tmpIdx < faInfo.kCache.size(); ++tmpIdx) {
            // 2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist are the same.
            if (faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != standardKH) {
                OP_LOGE(faInfo.opName, "H of Key(%ld) in the %ld-th batch is different from H of Key(%ld) in the first batch under tensorlist mode!",
                    faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2), tmpIdx + 1, standardKH);
                return false;
            }
            if (faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2) != standardVH) {
                OP_LOGE(faInfo.opName, "H of Value(%ld) in the %ld-th batch is different from H of Value(%ld) in the first batch under tensorlist mode!",
                    faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2), tmpIdx + 1, standardVH);
                return false;
            }
            if (faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1)) { // k_s != v_s
                OP_LOGE(faInfo.opName, "S for Key(%ld) and Value(%ld) does NOT equal!", 
                    faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1), faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1));
                return false;
            }
            if (faInfo.opParamInfo.keyRope.tensor != nullptr) {
                if (faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != keyRopeS) { // k_s != krope_s
                    OP_LOGE(faInfo.opName, "S for Key(%ld) and keyRope(%ld) does NOT equal but they should!", 
                        faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1), keyRopeS);
                    return false;
                }
            }
        }
    } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND") { // check N and D, respectively, are the same
        // across batches and KVs under BNSD/BNSD_BSND
        auto standardN = faInfo.kCache[0]->GetStorageShape().GetDim(1);
        auto standardKD = faInfo.kCache[0]->GetStorageShape().GetDim(3);
        auto standardVD = faInfo.vCache[0]->GetStorageShape().GetDim(3);
        int64_t tmpNKv = (faInfo.n2Size != 0) ? faInfo.n2Size : faInfo.n1Size;
        int64_t keyRopeS = 0;

        if (faInfo.opParamInfo.keyRope.tensor != nullptr) {
            keyRopeS = faInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(2);
            OP_CHECK_IF(faInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0) != faInfo.kCache.size(),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Batch of Key(%ld) do NOT equal to Batch of KeyRope(%ld) under tensorlist mode!", 
                faInfo.kCache.size(), faInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0)),
                return false);
        }

        if (tmpNKv != standardN) {
            OP_LOGE(faInfo.opName, "N of Key(%ld) in the first batch is different from numKeyValueHeads(%ld)!", standardN, tmpNKv);
            return false;
        }

        for (int64_t tmpIdx = 0; tmpIdx < faInfo.kCache.size(); ++tmpIdx) {
            if ((faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != standardN) ||
                (faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1) != standardN)) {
                OP_LOGE(faInfo.opName, "N of Key(%ld) and Value(%ld) in the %ld-th batch is different from numKeyValueHeads(%ld) under tensorlist mode!",
                    faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1), faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1),
                    tmpIdx + 1, standardN);
                return false;
            }
            if (faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(3) != standardKD) {
                OP_LOGE(faInfo.opName, "D of Key(%ld) in the %ld-th batch is different from D of Key(%ld) in the first batch under tensorlist mode!",
                    faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(3), tmpIdx + 1, standardKD);
                return false;
            }
            if (faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(3) != standardVD) {
                OP_LOGE(faInfo.opName, "D of Value(%ld) in the %ld-th batch is different from D of Value(%ld) in the first batch under tensorlist mode!",
                    faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(3), tmpIdx + 1, standardVD);
                return false;
            }
            if (faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != // 2: Obtain the second dimension
                faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2)) { // 2: Obtain the second dimension
                OP_LOGE(faInfo.opName, "S from Key(%ld) and Value(%ld) does NOT equal but they should!",
                    faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2), faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2));
                return false;
            }
            if (faInfo.opParamInfo.keyRope.tensor != nullptr) {
                if (faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != keyRopeS) { // k_s != krope_s
                    OP_LOGE(faInfo.opName, "S for Key(%ld) and keyRope(%ld) does NOT equal but they should!", 
                        faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2), keyRopeS);
                    return false;
                }
            }
        }
    } else { // check N and D, respectively, are the same across batches and KVs under BSND
        auto standardN = faInfo.kCache[0]->GetStorageShape().GetDim(2);
        auto standardKD = faInfo.kCache[0]->GetStorageShape().GetDim(3);
        auto standardVD = faInfo.vCache[0]->GetStorageShape().GetDim(3);
        int64_t tmpNKv = (faInfo.n2Size != 0) ? faInfo.n2Size : faInfo.n1Size;
        int64_t keyRopeS = 0;

        if (faInfo.opParamInfo.keyRope.tensor != nullptr) {
            keyRopeS = faInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(1);
            OP_CHECK_IF(faInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0) != faInfo.kCache.size(),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Batch of Key(%ld) do NOT equal to Batch of KeyRope(%ld) under tensorlist mode!", 
                faInfo.kCache.size(), faInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0)),
                return false);
        }

        if (tmpNKv != standardN) {
            OP_LOGE(faInfo.opName, "N of Key(%ld) in the first batch is different from numKeyValueHeads(%ld)!", standardN, tmpNKv);
            return false;
        }

        for (int64_t tmpIdx = 0; tmpIdx < faInfo.kCache.size(); ++tmpIdx) {
            if ((faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != standardN) || // 2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist are the same.
                (faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2) != standardN)) { // 2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist are the same.
                OP_LOGE(faInfo.opName, "N of Key(%ld) and Value(%ld) in the %ld-th batch is different from numKeyValueHeads(%ld) under tensorlist mode!",
                    faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2), faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2),
                    tmpIdx + 1, standardN);
                return false;
            }
            if (faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(3) != standardKD) {
                OP_LOGE(faInfo.opName, "D of Key(%ld) in the %ld-th batch is different from D of Key(%ld) in the first batch under tensorlist mode!",
                    faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(3), tmpIdx + 1, standardKD);
                return false;
            }
            if (faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(3) != standardVD) {
                OP_LOGE(faInfo.opName, "D of Value(%ld) in the %ld-th batch is different from D of Value(%ld) in the first batch under tensorlist mode!",
                    faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(3), tmpIdx + 1, standardVD);
                return false;
            }
            if (faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1)) {
                OP_LOGE(faInfo.opName, "S from Key(%ld) and Value(%ld) does NOT equal but they should!",
                    faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1), faInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1));
                return false;
            }
            if (faInfo.opParamInfo.keyRope.tensor != nullptr) {
                if (faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != keyRopeS) { // k_s != krope_s
                    OP_LOGE(faInfo.opName, "S for Key(%ld) and keyRope(%ld) does NOT equal but they should!", 
                        faInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1), keyRopeS);
                    return false;
                }
            }
        }
    }
    return true;
}

ge::graphStatus ShapeChecker::CheckTensorList(const FaTilingInfo &faInfo)
{
    std::string layoutStr(faInfo.opParamInfo.layoutOut);
    OP_CHECK_IF((faInfo.opParamInfo.blockTable.tensor != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
            "When tensorlist is used, page attention is not supported!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((layoutStr == "TND" || layoutStr == "NTD" || layoutStr == "NTD_TND" || layoutStr == "TND_NTD"),
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
            "When tensorlist is used, layout TND/NTD/TND_NTD/NTD_TND is not supported!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((faInfo.bSize > B_LIMIT),
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
            "Batch of Query(%ld) do NOT larger than 65535 under tensorlist mode!", faInfo.bSize),
        return ge::GRAPH_FAILED);
    if (CheckEmptyTensorList(faInfo)) {
        return ge::GRAPH_SUCCESS;
    }
    if (!CheckNormalTensorList(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if ((CheckQueryKeyTensorlistConsistency(faInfo)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckMultiDtype(const FaTilingInfo &faInfo)
{
    const std::map<std::string, std::vector<ge::DataType>> QKVD_Different_MAP = {
        {"query",                  {ge::DT_FLOAT16, ge::DT_BF16}},
        {"key",                    {ge::DT_FLOAT16, ge::DT_BF16}},
        {"value",                  {ge::DT_FLOAT16, ge::DT_BF16}},
        {"attentionOut",           {ge::DT_FLOAT16, ge::DT_BF16}},
    };
    OP_CHECK_IF(faInfo.isQKVDDifferent &&
        ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.query.desc, "query", QKVD_Different_MAP),
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Query data type must be float16 or bf16 when query and "
            "key headdim is not equal to value headdim."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.isQKVDDifferent &&
        ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.attenOut.desc, "attentionOut", QKVD_Different_MAP),
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Output data type must be float16 or bf16 when query and "
            "key headdim is not equal to value headdim."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.isQKVDDifferent &&
        ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.key.desc, "key", QKVD_Different_MAP),
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Key data type must be float16 or bf16 when query and "
            "key headdim is not equal to value headdim."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.isQKVDDifferent &&
        ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.value.desc, "value", QKVD_Different_MAP),
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Value data type must be float16 or bf16 when query and "
            "key headdim is not equal to value headdim."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckAxis(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.bSize > B_LIMIT || faInfo.bSize <= 0,
                OP_LOGE(faInfo.opName, "The axis B only support (0, %u], the current is %u.",
                        B_LIMIT, faInfo.bSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s1Size < 0,
                OP_LOGE(faInfo.opName, "The axis S of query only support >=0, the current is %ld.",
                        faInfo.s1Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s2Size < 0,
                OP_LOGE(faInfo.opName, "The axis S of key and value only support >=0, the current is %ld.",
                        faInfo.s2Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.qkHeadDim > D_LIMIT || faInfo.qkHeadDim < 0,
                OP_LOGE(faInfo.opName, "The axis D of query and key only support (0, %u], the current is %u.",
                        D_LIMIT, faInfo.qkHeadDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.vHeadDim > D_LIMIT || faInfo.vHeadDim < 0,
                OP_LOGE(faInfo.opName, "The axis D of value only support (0, %u], the current is %u.",
                        D_LIMIT, faInfo.vHeadDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.isQKVDDifferent && (faInfo.qkHeadDim > 128),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Query headdim must smaller than 128 when query and key "
                    "headdim is not equal to value headdim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.isQKVDDifferent && (faInfo.vHeadDim > 128),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Value headdim must smaller than 128 when query and key "
                    "headdim is not equal to value headdim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.qLayout == FaLayout::TND && faInfo.qTSize < 0,
                OP_LOGE(faInfo.opName,
                "The axis T of query only support >0 when input layout is TND, the current is %u",
                        faInfo.qTSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.kvLayout == FaLayout::TND && faInfo.kTSize < 0,
                OP_LOGE(faInfo.opName,
                "The axis T of key only support >0 when input layout is TND, the current is %u.",
                        faInfo.kTSize),
                return ge::GRAPH_FAILED);
    if (faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        OP_CHECK_IF(faInfo.s1Size < 1,
                OP_LOGE(faInfo.opName, "input query's sequence length is %u, it should be "
                    ">=1 when enable ifa mla", faInfo.s1Size),
                return ge::GRAPH_FAILED);
        static const std::set<uint32_t> SUPPORT_G_IN_IFAMLA = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U}; // ifa mla场景g轴支持范围
        OP_CHECK_IF((SUPPORT_G_IN_IFAMLA.find(faInfo.gSize) == SUPPORT_G_IN_IFAMLA.end()),
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "The asix G should be in range of "
                "{1, 2, 4, 8, 16, 32, 64, 128} when enable ifa mla, the current is %u.",
                faInfo.n1Size / faInfo.n2Size),
            return ge::GRAPH_FAILED);
    }
    OP_LOGI(faInfo.opName, "The axis B(%u), qkD(%u), vD(%u), G(%u), qT(%u), kT(%u).",
            faInfo.bSize, faInfo.qkHeadDim, faInfo.vHeadDim, faInfo.gSize, faInfo.qTSize, faInfo.kTSize);
    return ge::GRAPH_SUCCESS;
}

void ShapeChecker::GetQueryDimAndOutDim(const gert::StorageShape* queryShape, const gert::StorageShape* outShape,
    const std::string &layoutStr, int64_t &tmpQueryDim, int64_t &outDim, uint32_t i) {
    if (layoutStr == "BNSD_BSND" || layoutStr == "BSND_BNSD") {
        if (i == 1) { // BNSD_BSND：query:N, output:S; BSND_BNSD：query:S, output:N
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 2) { // BNSD_BSND：query:S, output:N; BSND_BNSD：query:N, output:S
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        }
    } else if (layoutStr == "BSH_BNSD") {
        if (i == 2) { // BSH_BNSD：query:H, output:ND
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1) * outShape->GetStorageShape().GetDim(i - 1);
        }
    } else if (layoutStr == "BSND_NBSD") {
        if (i == 1) { // BSND_NBSD：query:S, output:B
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 2) { // BSND_NBSD：query:N, output:S
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 2);
        }
    } else if (layoutStr == "BNSD_NBSD") {
        if (i == 1) { // BNSD_NBSD：query:N, output:B
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        } else if (i == 2) { // BNSD_NBSD：query:S, output:S
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i);
        }
    } else if (layoutStr == "BSH_NBSD") {
        if (i == 2) { // BSH_NBSD：query:H, output:ND
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1) * outShape->GetStorageShape().GetDim(i - 2);
        }
    } else if (layoutStr == "NTD_TND" || layoutStr == "TND_NTD") {
        if (i == 0) { // query:N/T, output:T/N
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 1) { // 2 for current queryDimNum; Q:T/N, Output:N/T
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        }
    } else {
        tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
        outDim = outShape->GetStorageShape().GetDim(i);
    }
}

ge::graphStatus ShapeChecker::CheckQueryOutConsistency(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *queryShape = faInfo.opParamInfo.query.shape;
    const gert::StorageShape *attentionOutShape = faInfo.opParamInfo.attenOut.shape;
    size_t dimNumQ = queryShape->GetStorageShape().GetDimNum();
    size_t dimNumOut = attentionOutShape->GetStorageShape().GetDimNum();
    int64_t tmpQueryDim = 0;
    int64_t tmpOutDim = 0;
    std::string layoutStr(faInfo.opParamInfo.layoutOut);
    bool isLayoutShapeSupport = layoutStr == "BSH_BNSD" || layoutStr == "BSH_NBSD";
    OP_CHECK_IF(dimNumQ != dimNumOut && !isLayoutShapeSupport,
                OP_LOGE(faInfo.opName,
                        "The dim num of query should be consistent with the dim num of attentionOut,"
                        "but current dim num of query is %zu, dim num of attentionOut is %zu.",
                        dimNumQ, dimNumOut),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimNumQ < INPUT_Q_SHAPE_MIN_DIMS || dimNumQ > INPUT_Q_SHAPE_MAX_DIMS,
                OP_LOGE(faInfo.opName, "The dim num of query only support [%u, %u], but current is %zu.",
                        INPUT_Q_SHAPE_MIN_DIMS, INPUT_Q_SHAPE_MAX_DIMS, dimNumQ),
                return ge::GRAPH_FAILED);

    for (uint32_t queryDimIdx = 0; queryDimIdx < dimNumQ; ++queryDimIdx) {
        if ((queryDimIdx == dimNumQ - 1) && faInfo.mlaMode == MlaMode::ROPE_COMBINE_D128) {
            continue;
        }
        GetQueryDimAndOutDim(queryShape, attentionOutShape, layoutStr, tmpQueryDim, tmpOutDim, queryDimIdx);
        OP_CHECK_IF(!faInfo.isQKVDDifferent && (tmpQueryDim != tmpOutDim),
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                "tensor query shape (%ld) do not equal to tensor output shape(%ld) in dim %u for %s.",
                tmpQueryDim, tmpOutDim, queryDimIdx, layoutStr.c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(faInfo.isQKVDDifferent && (queryDimIdx != dimNumQ - 1) && (tmpQueryDim != tmpOutDim),
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                "tensor query shape (%ld) do not equal to tensor output shape(%ld) in dim %u for %s.",
                tmpQueryDim, tmpOutDim, queryDimIdx, layoutStr.c_str()),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckKeyValueConsistency(const FaTilingInfo &faInfo)
{
    if (faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *keyShape = faInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = faInfo.opParamInfo.value.shape;
    ge::DataType keyDataType = faInfo.opParamInfo.key.desc->GetDataType();
    ge::DataType valueDataType = faInfo.opParamInfo.value.desc->GetDataType();

    const size_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    const size_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((keyDataType != valueDataType),
                OP_LOGE(faInfo.opName, "The data type of key is %s,the data type of value is %s",
                    DataTypeToSerialString(keyDataType).c_str(), DataTypeToSerialString(valueDataType).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDimNum != valueDimNum,
                OP_LOGE(faInfo.opName,
                        "The dim num of key should be consistent with the dim num of value,"
                        "but current dim num of key is %zu, dim num of value is %zu.",
                        keyDimNum, valueDimNum),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(keyDimNum < INPUT_KV_SHAPE_MIN_DIMS || keyDimNum > INPUT_KV_SHAPE_MAX_DIMS,
                OP_LOGE(faInfo.opName, "The dim num of key only support [%u, %u], but current is %zu.",
                        INPUT_KV_SHAPE_MIN_DIMS, INPUT_KV_SHAPE_MAX_DIMS, keyDimNum),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < keyDimNum; ++i) {
        if ((i == keyDimNum - 1) && faInfo.mlaMode == MlaMode::ROPE_COMBINE_D128) {
            continue;
        }
        int64_t tmpKeyDim = keyShape->GetStorageShape().GetDim(i);
        int64_t tmpValueDim = valueShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(!faInfo.isQKVDDifferent && (tmpKeyDim != tmpValueDim),
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                "tensor key shape(%ld) do not equal to tensor value shape(%ld) in dim %u.", tmpKeyDim, tmpValueDim, i),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(faInfo.isQKVDDifferent && (i != keyDimNum - 1) && (tmpKeyDim != tmpValueDim),
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                "tensor key shape(%ld) do not equal to tensor value shape(%ld) in dim %u.", tmpKeyDim, tmpValueDim, i),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckQueryShape(const FaTilingInfo &faInfo)
{
    uint32_t attrN = faInfo.n1Size;
    const gert::StorageShape *queryShape = faInfo.opParamInfo.query.shape;
    uint32_t queryShapeHeadNum = attrN;
    size_t queryDim = queryShape->GetStorageShape().GetDimNum();
    int64_t queryH = 0;

    if (faInfo.qLayout == FaLayout::BNSD || faInfo.qLayout == FaLayout::TND) {
        queryShapeHeadNum = queryShape->GetStorageShape().GetDim(1);
    } else if (faInfo.qLayout == FaLayout::BSND) {
        queryShapeHeadNum = queryShape->GetStorageShape().GetDim(2);
    } else if (faInfo.qLayout == FaLayout::BSH) {
        queryH = queryShape->GetStorageShape().GetDim(2);
    } else if (faInfo.qLayout == FaLayout::NTD) {
        queryShapeHeadNum = queryShape->GetStorageShape().GetDim(0);
    }
    OP_CHECK_IF(attrN != queryShapeHeadNum,
                OP_LOGE(faInfo.opName, "The attr numHeads is %u, the axis N of query is %u, they should be equal.",
                        attrN, queryShapeHeadNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.qLayout == FaLayout::BSH && queryH % attrN != 0,
                OP_LOGE(faInfo.opName, "When layout is BSH, the axis H of query be an interger multiple of numHeads, "
                                        "but current H is %ld numHeads is %u.",
                        queryH, attrN),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((faInfo.qLayout == FaLayout::BNSD || faInfo.qLayout == FaLayout::BSND) && queryDim != 4U,
                OP_LOGE(faInfo.opName, "When layout is %s, query dim num is  %u, it should be 4.",
                        faInfo.opParamInfo.layoutOut, queryDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((faInfo.qLayout == FaLayout::BSH || faInfo.qLayout == FaLayout::TND) && queryDim != 3U,
                OP_LOGE(faInfo.opName, "When layout is %s, query dim num is  %u, it should be 3.",
                        faInfo.opParamInfo.layoutOut, queryDim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckKeyNHVaild(const FaTilingInfo &faInfo, const gert::Shape &keyShape)
{
    uint32_t attrKvN = faInfo.n2Size;
    size_t keyDim = keyShape.GetDimNum();
    uint32_t keyShapeHeadNum = attrKvN;
    int64_t keyH = 0;

    if (faInfo.kvLayout == FaLayout::BNSD || faInfo.kvLayout == FaLayout::TND) {
        keyShapeHeadNum = keyShape.GetDim(1);
    } else if (faInfo.kvLayout == FaLayout::BSND) {
        keyShapeHeadNum = keyShape.GetDim(2);
    } else if (faInfo.kvLayout == FaLayout::BSH) {
        keyH = keyShape.GetDim(2);
    }
    OP_CHECK_IF(attrKvN != keyShapeHeadNum,
                OP_LOGE(faInfo.opName,
                "The attr numKeyValueHeads is %u, the axis N of key is %u, they should be equal.",
                        attrKvN, keyShapeHeadNum),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(faInfo.kvLayout == FaLayout::BSH && keyH % attrKvN != 0,
                OP_LOGE(faInfo.opName, "When layout is BSH, the axis H of key be an interger multiple of numHeads, "
                                        "but current H is %ld numKeyValueHeads is %u.",
                        keyH, attrKvN),
                 return ge::GRAPH_FAILED);
    OP_CHECK_IF((faInfo.kvLayout == FaLayout::BNSD || faInfo.kvLayout == FaLayout::BSND) && keyDim != 4U,
                OP_LOGE(faInfo.opName, "When layout is %s, key dim num is  %u, it should be 4.",
                        faInfo.opParamInfo.layoutOut, keyDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((faInfo.kvLayout == FaLayout::BSH || faInfo.kvLayout == FaLayout::TND) && keyDim != 3U,
                OP_LOGE(faInfo.opName, "When layout is %s, key dim num is  %u, it should be 3.",
                        faInfo.opParamInfo.layoutOut, keyDim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckKeyDVaild(const FaTilingInfo &faInfo, const gert::Shape &keyShape)
{
    size_t keyDim = keyShape.GetDimNum();
    uint32_t keyHeadDim = 0;
    if (faInfo.kvLayout != FaLayout::BSH) {
        keyHeadDim = keyShape.GetDim(keyDim -1);
    } else {
        keyHeadDim = keyShape.GetDim(keyDim - 1) / faInfo.n2Size;
    }

    OP_CHECK_IF(keyHeadDim != faInfo.qkHeadDim,
                OP_LOGE(faInfo.opName, "The axis D of key is %u, the axis D of query is %u, they should be equal.",
                        keyHeadDim, faInfo.qkHeadDim),
                return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckKeyShape(const FaTilingInfo &faInfo)
{
    if (faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (faInfo.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckKeyNHVaild(faInfo, faInfo.opParamInfo.key.shape->GetStorageShape());
    } else {
        for (uint32_t i = 0; i < faInfo.kCache.size(); i++) {
            auto keyShape = faInfo.kCache[i]->GetStorageShape();
            if (CheckKeyNHVaild(faInfo, keyShape) != ge::GRAPH_SUCCESS) {
                OP_LOGE(faInfo.opName, "When kv storage mode is tensorlist, the input %d of key shape is invalid", i);
                return ge::GRAPH_FAILED;
            }
        }
    return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus ShapeChecker::CheckQueryKeyTensorlistConsistency(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.bSize != faInfo.kCache.size(),
        OP_LOGE(faInfo.opName,
        "The axis B of query is %u, the number of key is %zu when tensor list, they should be equal",
                faInfo.bSize, faInfo.kCache.size()),
        return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < faInfo.kCache.size(); i++) {
        auto keyShape = faInfo.kCache[i]->GetStorageShape();
        if (CheckKeyDVaild(faInfo, keyShape) != ge::GRAPH_SUCCESS) {
            OP_LOGE(faInfo.opName, "When kv storage mode is tensorlist, the input %d of key shape is invalid", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckQueryKeyConsistency(const FaTilingInfo &faInfo)
{
    if (faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }
    ge::DataType queryDataType = faInfo.opParamInfo.query.desc->GetDataType();
    ge::DataType keyDataType = faInfo.opParamInfo.key.desc->GetDataType();
    if (enableNonQuant_) {
        OP_CHECK_IF((queryDataType != keyDataType),
                    OP_LOGE(faInfo.opName, "The data type of query is %s,the data type of key is %s",
                        DataTypeToSerialString(queryDataType).c_str(), DataTypeToSerialString(keyDataType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    int64_t keyB = faInfo.opParamInfo.key.shape->GetStorageShape().GetDim(0);
    if (faInfo.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
        OP_CHECK_IF((faInfo.qLayout != FaLayout::TND && faInfo.qLayout != FaLayout::NTD) && faInfo.bSize != keyB,
                OP_LOGE(faInfo.opName, "The axis B of query is %u, the axis B of key is %ld, they should be equal",
                        faInfo.bSize, keyB),
                return ge::GRAPH_FAILED);
        return CheckKeyDVaild(faInfo, faInfo.opParamInfo.key.shape->GetStorageShape());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckMultiAttr(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.pageAttentionFlag && faInfo.isQKVDDifferent,
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Not support PA when query and key headdim is not equal to value headdim."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.pseShiftFlag && faInfo.isQKVDDifferent,
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Not support pse shift when query and key headdim is not equal to value headdim."),
        return false);
    OP_CHECK_IF(faInfo.enableAlibiPse && faInfo.isQKVDDifferent,
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Not support pse shift when query and key headdim is not equal to value headdim."),
        return false);

    if (faInfo.inputQType != ge::DT_FLOAT16) {
        OP_LOGW(faInfo.opName, "When query input is not fp16,innerPrecise will not take effect");
    }

    return ge::GRAPH_SUCCESS;
}

// enableNonQuant 相关校验函数
ge::graphStatus ShapeChecker::CheckNonQuantDataType(const FaTilingInfo &faInfo)
{
    const std::map<std::string, std::vector<ge::DataType>> NO_QUANT_MAP = {
        {"query",                  {ge::DT_FLOAT16, ge::DT_BF16}},
        {"key",                    {ge::DT_FLOAT16, ge::DT_BF16}},
        {"value",                  {ge::DT_FLOAT16, ge::DT_BF16}},
        {"attentionOut",           {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN}},
    };
    if (ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.query.desc, "query", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.key.desc, "key", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.value.desc, "value", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.attenOut.desc,
        "attentionOut", NO_QUANT_MAP)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckNonQuantAttr(const FaTilingInfo &faInfo)
{
    if (CheckNonQuantHeadNum(faInfo) != ge::GRAPH_SUCCESS ||
        CheckNonQuantInputLayout(faInfo) != ge::GRAPH_SUCCESS ||
        CheckNonQuantInnerPrecise(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckNonQuantHeadNum(const FaTilingInfo &faInfo)
{
    if ((faInfo.n1Size < 0) || (faInfo.n2Size < 0)) {
        OP_LOGE(faInfo.opName, "numHeads(%d) or numKeyValueHeads(%d) is negative!", faInfo.n1Size, faInfo.n2Size);
        return ge::GRAPH_FAILED;
    }

    if (faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) { // ifamla
        static const std::set<uint32_t> SUPPORT_NUM_HEAD_IN_IFAMLA = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U}; // ifa mla场景qN支持范围
        OP_CHECK_IF((SUPPORT_NUM_HEAD_IN_IFAMLA.find(faInfo.n1Size) == SUPPORT_NUM_HEAD_IN_IFAMLA.end()),
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Input query's heads num is %u, it should be in range of "
                "{1, 2, 4, 8, 16, 32, 64, 128} when enable ifa mla", faInfo.n1Size),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((faInfo.n2Size != 1U),
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Input key/value's heads num is %u, it should be 1 when enable "
                "ifa mla", faInfo.n2Size),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckNonQuantInputLayout(const FaTilingInfo &faInfo)
{
    // if (faInfo.opParamInfo.layoutOut == nullptr) {
    //     return ge::GRAPH_FAILED;
    // }
    // std::string inputLayout = faInfo.opParamInfo.layoutOut;
    // if (faInfo.ropeMode == RopeMode::NO_ROPE) {
    //     if (faInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) {
    //         const std::vector<std::string> INPUT_LAYOUT_LIST = {
    //             "BSH", "BSND", "BNSD", "TND", "NTD", "BSND_BNSD", "BSH_BNSD", "NTD_TND", "BNSD_BSND"
    //         };
    //         if (std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end()) {
    //             OP_LOGE(faInfo.opName, "When gqa noquant scenario is applied, the attr inputLayout only supports BSH, "
    //                 "BSND, BNSD, TND, NTD, BSND_BNSD, BSH_BNSD, NTD_TND, BNSD_BSND, but got %s", inputLayout.c_str());
    //             return ge::GRAPH_FAILED;
    //         }
    //     } else {
    //         const std::vector<std::string> INPUT_LAYOUT_LIST = {
    //             "BSH", "BSND", "BNSD", "TND", "NTD", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD", "NTD_TND", "BSH_BNSD", "BSND_BNSD", "BNSD_BSND"
    //         };
    //         OP_CHECK_IF(std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end(),
    //             OP_LOGE(faInfo.opName, "When gqa noquant scenario is applied, layout only supports BSH, BSND, BNSD, TND, NTD, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, NTD_TND, BSH_BNSD, BSND_BNSD, BNSD_BSND, but got %s",
    //                 inputLayout.c_str()),
    //             return ge::GRAPH_FAILED);
    //         const std::vector<std::string> noRopeLayoutSupportListA = {"BSH", "BSND", "BNSD"};
    //         const std::vector<std::string> noRopeLayoutSupportListB = {"BNSD_BSND"};
    //         const std::vector<std::string> noRopeLayoutSupportListC = {"NTD", "BSH_BNSD", "BSND_BNSD", "NTD_TND"};
    //         const std::vector<std::string> noRopeLayoutSupportListD = {"TND"};

    //         if (!faInfo.isLegacyIfa &&
    //             faInfo.vHeadDim % 16 != 0) { // 16: qkvD need 16 align when qs>1, in specific input layout
    //             OP_CHECK_IF(std::find(noRopeLayoutSupportListA.begin(), noRopeLayoutSupportListA.end(), inputLayout) !=
    //                             noRopeLayoutSupportListA.end(),
    //                         OP_LOGE(faInfo.opName,
    //                                 "In %s %s situation, when Qs>1 and input_layout is %s, headDim of query|key|value "
    //                                 "should be align to 16.",
    //                                 QuantModeToSerialString(faInfo.quantMode).c_str(),
    //                                 SituationToSerialString(faInfo.ropeMode).c_str(), inputLayout.c_str()),
    //                         return ge::GRAPH_FAILED);

    //             OP_CHECK_IF(std::find(noRopeLayoutSupportListB.begin(), noRopeLayoutSupportListB.end(), inputLayout) !=
    //                             noRopeLayoutSupportListB.end(),
    //                         OP_LOGE(faInfo.opName,
    //                                 "In %s %s situation, when Qs>1 and input_layout is %s, headDim of query|key|value "
    //                                 "should be align to 16.",
    //                                 QuantModeToSerialString(faInfo.quantMode).c_str(),
    //                                 SituationToSerialString(faInfo.ropeMode).c_str(), inputLayout.c_str()),
    //                         return ge::GRAPH_FAILED);
    //         }

    //         if (faInfo.isLegacyIfa &&
    //             (faInfo.vHeadDim != 64 &&
    //              faInfo.vHeadDim != 128)) { // 64: qkvD need 64 128: qkvD need 128 in specific input layout
    //             OP_CHECK_IF(std::find(noRopeLayoutSupportListB.begin(), noRopeLayoutSupportListB.end(), inputLayout) !=
    //                             noRopeLayoutSupportListB.end(),
    //                         OP_LOGE(faInfo.opName,
    //                                 "In %s %s situation, when Qs=1 and input_layout is BNSD_BSND, only query|key|value "
    //                                 "headDim = 64/128 are supported, but got %u",
    //                                 QuantModeToSerialString(faInfo.quantMode).c_str(),
    //                                 SituationToSerialString(faInfo.ropeMode).c_str(), faInfo.vHeadDim),
    //                         return ge::GRAPH_FAILED);
    //         }
    //         if (std::find(noRopeLayoutSupportListC.begin(), noRopeLayoutSupportListC.end(), inputLayout) !=
    //             noRopeLayoutSupportListC.end()) {
    //             OP_CHECK_IF(faInfo.vHeadDim != 64 &&
    //                             faInfo.vHeadDim !=
    //                                 128, // 64: qkvD (optional) 64 128: qkvD (optional) 128 in specific input layout
    //                         OP_LOGE(faInfo.opName,
    //                                 "In %s %s situation, when input_layout is NTD, BSH_BNSD, BSND_BNSD, NTD_TND, only "
    //                                 "query|key|value headDim = 64/128 are supported, but got %u",
    //                                 QuantModeToSerialString(faInfo.quantMode).c_str(),
    //                                 SituationToSerialString(faInfo.ropeMode).c_str(), faInfo.vHeadDim),
    //                         return ge::GRAPH_FAILED);
    //         }
    //         if (std::find(noRopeLayoutSupportListD.begin(), noRopeLayoutSupportListD.end(), inputLayout) !=
    //             noRopeLayoutSupportListD.end()) {
    //             OP_CHECK_IF(faInfo.vHeadDim != 64 && faInfo.vHeadDim != 128 &&
    //                             faInfo.vHeadDim != 192, // 64: qkvD (optional) 64, 128: qkvD (optional) 128, 192: qkvD
    //                                                      // (optional) 192 when input_layout=TND
    //                         OP_LOGE(faInfo.opName,
    //                                 "In %s %s situation, when input_layout is TND, only query|key|value headDim = "
    //                                 "64/128/192 are supported, but got %u",
    //                                 QuantModeToSerialString(faInfo.quantMode).c_str(),
    //                                 SituationToSerialString(faInfo.ropeMode).c_str(), faInfo.vHeadDim),
    //                         return ge::GRAPH_FAILED);
    //         }
    //     }
    // } else if (faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) { // decode mla
    //     const std::vector<std::string> INPUT_LAYOUT_LIST = {
    //         "BSH", "BSND", "BNSD", "TND", "BNSD_NBSD", "BSND_NBSD", "BSH_NBSD", "TND_NTD"
    //     };
    //     if (std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end()) {
    //         OP_LOGE(faInfo.opName, "When decode mla scenario is applied, the attr inputLayout only supports BSH, BSND, BNSD, "
    //             "TND ,BNSD_BSND, BSND_NBSD, BSH_NBSD, TND_NTD, but got %s", inputLayout.c_str());
    //         return ge::GRAPH_FAILED;
    //     }
    // } else { // prefill mla
    //     const std::vector<std::string> INPUT_LAYOUT_LIST = {
    //         "BSH", "BSND", "BNSD", "TND", "NTD", "BSND_BNSD", "BSH_BNSD", "NTD_TND", "BNSD_BSND"
    //     };
    //     if (std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end()) {
    //         OP_LOGE(faInfo.opName, "When prefill mla noquant scenario is applied, the attr inputLayout only supports BSH, "
    //             "BSND, BNSD, TND, NTD, BSND_BNSD, BSH_BNSD, NTD_TND, BNSD_BSND, but got %s", inputLayout.c_str());
    //         return ge::GRAPH_FAILED;
    //     }
    // }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckNonQuantInnerPrecise(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.innerPrecise > INNER_PRECISE_LIMIT || faInfo.innerPrecise < 0,
        OP_LOGE(faInfo.opName, "The attr innerPrecise only support [0, %u], the current is %u.",
                INNER_PRECISE_LIMIT, faInfo.innerPrecise),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

bool ShapeChecker::CheckTNDLayoutCrossover(const FaTilingInfo &faInfo)
{
    if (faInfo.inputLayout != TilingKeyLayout::TND) {
        return true;
    }

    std::string layoutStr(faInfo.opParamInfo.layoutOut);
    if (faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 && layoutStr == "TND_NTD") { // Decode MLA
        OP_CHECK_IF(faInfo.isOutQuantEnable,
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "In Decode MLA scenario, when layout is TND_NTD, post quant is not supported!"),
            return false);
    }

    OP_CHECK_IF(faInfo.pseShiftFlag,
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "When layout is TND, pse is not supported!"),
        return false);

    return true;
}

bool ShapeChecker::CheckNTDLayoutCrossover(const FaTilingInfo &faInfo)
{
    if (faInfo.inputLayout != TilingKeyLayout::NTD) {
        return true;
    }
    bool isGqa = enableNonQuant_ && !(faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) && !(faInfo.mlaMode == MlaMode::ROPE_SPLIT_D128) &&
        !(faInfo.mlaMode == MlaMode::ROPE_COMBINE_D128);
    std::string layoutStr(faInfo.opParamInfo.layoutOut);
    if (isGqa) { // GQA
        OP_CHECK_IF((faInfo.qkHeadDim != 64 && faInfo.qkHeadDim != 128),
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "In GQA scenario, when layout is %s, d size of query must be 64 or 128, but got d = %d.",
            layoutStr.c_str(), faInfo.qkHeadDim), return false);
    }
    if (isGqa) { // GQA
        OP_CHECK_IF(faInfo.isQKVDDifferent,
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "In GQA scenario, not support layout %s when query and key headdim is not equal to value headdim.",
            layoutStr.c_str()), return false);
    }

    OP_CHECK_IF(faInfo.pseShiftFlag,
        OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "When layout is %s, pse is not supported!", layoutStr.c_str()),
        return false);

    return true;
}

bool ShapeChecker::CheckTransposeLayoutCrossover(const FaTilingInfo &faInfo)
{
    std::string layoutStr(faInfo.opParamInfo.layoutOut);
    if (layoutStr != "BSH_BNSD" && layoutStr != "BSND_BNSD" && layoutStr != "BNSD_BSND") {
        return true;
    }
    bool isGqa = enableNonQuant_ && !(faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) && !(faInfo.mlaMode == MlaMode::ROPE_SPLIT_D128) &&
        !(faInfo.mlaMode == MlaMode::ROPE_COMBINE_D128);
    if (isGqa) { // GQA
        OP_CHECK_IF(faInfo.isQKVDDifferent,
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "In GQA scenario, not support layout %s when query and key headdim is not equal to value headdim.",
            layoutStr.c_str()), return false);
    }
    if (layoutStr == "BSH_BNSD" || layoutStr == "BSND_BNSD") {
        if (isGqa) { // GQA
            OP_CHECK_IF((faInfo.qkHeadDim != 64 && faInfo.qkHeadDim != 128),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "In GQA scenario, when layout is %s, d size of query must be 64 or 128, but got d = %d.",
                layoutStr.c_str(), faInfo.qkHeadDim), return false);
        }

        OP_CHECK_IF(faInfo.pseShiftFlag,
            OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "When layout is %s, pse is not supported!",
            layoutStr.c_str()), return false);
    } else if (layoutStr == "BNSD_BSND") {
        if (isGqa) { // GQA
            OP_CHECK_IF((faInfo.outputType == ge::DT_INT8 && faInfo.qkHeadDim % 32 != 0),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "In GQA scenario, when layout is %s and output dtype is int8, d size should be a multiple of %d, but got d = %d.",
                layoutStr.c_str(), 32, faInfo.qkHeadDim), return false);
            OP_CHECK_IF((faInfo.qkHeadDim % 16 != 0),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "In GQA scenario, when layout is %s, d size should be a multiple of %d, but got d = %d.",
                layoutStr.c_str(), 16, faInfo.qkHeadDim), return false);
        }
    }
    return true;
}

ge::graphStatus ShapeChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (enableNonQuant_) {
        if (CheckNonQuantDataType(faInfo) != ge::GRAPH_SUCCESS ||
            CheckInputFormat(faInfo) != ge::GRAPH_SUCCESS ||
            CheckNonQuantAttr(faInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (CheckParaExistenceImpl(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (enableNonQuant_) {
        if (faInfo.pageAttentionFlag) {
            if (CheckPAKeyValue(faInfo) != ge::GRAPH_SUCCESS) { // PA场景
                return ge::GRAPH_FAILED;
            }
        }
        ge::DataType queryDataType = faInfo.opParamInfo.query.desc->GetDataType();
        ge::DataType attenOutDataType = faInfo.opParamInfo.attenOut.desc->GetDataType();
        if (!faInfo.isOutQuantEnable) {
            OP_CHECK_IF((queryDataType != attenOutDataType),
                        OP_LOGE(faInfo.opName, "The data type of query is xxx,the data type of attentionOut is xxx"),
                        return ge::GRAPH_FAILED);
        }
        if (!CheckTNDLayoutCrossover(faInfo) || !CheckNTDLayoutCrossover(faInfo) || !CheckTransposeLayoutCrossover(faInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (enableNonQuant_) {
        if (CheckMultiDtype(faInfo) != ge::GRAPH_SUCCESS ||
            CheckAxis(faInfo) != ge::GRAPH_SUCCESS ||
            CheckQueryOutConsistency(faInfo) != ge::GRAPH_SUCCESS ||
            CheckKeyValueConsistency(faInfo) != ge::GRAPH_SUCCESS ||
            CheckQueryShape(faInfo) != ge::GRAPH_SUCCESS ||
            CheckKeyShape(faInfo) != ge::GRAPH_SUCCESS ||
            CheckQueryKeyConsistency(faInfo) != ge::GRAPH_SUCCESS ||
            CheckMultiAttr(faInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling