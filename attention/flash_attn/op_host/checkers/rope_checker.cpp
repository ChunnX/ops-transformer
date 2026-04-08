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
 * \file rope_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "rope_checker.h"

namespace optilingForBuildPKG {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// check rope dtype
ge::graphStatus RopeChecker::CheckRopeDtype(const FaTilingInfo &faInfo)
{
    if (faInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((faInfo.inputQRopeType != ge::DT_BF16 && faInfo.inputQRopeType != ge::DT_FLOAT16),
        OP_LOGE(faInfo.opName,
        "When rope exist, the datatype(%s) of queryRope only support BF16 or FLOAT16.",
        DataTypeToSerialString(faInfo.inputQRopeType).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((faInfo.inputKRopeType != ge::DT_BF16 && faInfo.inputKRopeType != ge::DT_FLOAT16),
        OP_LOGE(faInfo.opName,
        "When rope exist, the datatype(%s) of keyRope only support BF16 or FLOAT16.",
        DataTypeToSerialString(faInfo.inputKRopeType).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check rope DSize
ge::graphStatus RopeChecker::CheckRopeDSizeSupport(const FaTilingInfo &faInfo)
{
    if (faInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((faInfo.ropeHeadDim != NUM_64),
        OP_LOGE(faInfo.opName,
            "When rope exist, the d size of rope must be 64, but current is %u.", faInfo.ropeHeadDim),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckQDsizeSupport(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF((faInfo.qkHeadDim != 128 && faInfo.qkHeadDim != 512),
        OP_LOGE(faInfo.opName,
            "When rope exist, the d size of query and key must be 128 or 512, but current is %u.", faInfo.qkHeadDim),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckShapeSupport(const FaTilingInfo &faInfo)
{
    // check qk head dim and rope head dim
    if (ge::GRAPH_SUCCESS != CheckQDsizeSupport(faInfo)) {
            return ge::GRAPH_FAILED;
    }

    const gert::Shape queryShape = faInfo.opParamInfo.query.shape->GetStorageShape();
    const gert::Shape queryRopeShape = faInfo.opParamInfo.queryRope.tensor->GetStorageShape();
    const gert::Shape keyShape = faInfo.opParamInfo.key.shape->GetStorageShape();
    const gert::Shape keyRopeShape = faInfo.opParamInfo.keyRope.tensor->GetStorageShape();

    // check queryRope shape
    if (ge::GRAPH_SUCCESS != CheckQKAndQKRopeShapeConsistency(faInfo, queryShape, queryRopeShape, "query")) {
            return ge::GRAPH_FAILED;
    }
    // check keyRope shape
    if (ge::GRAPH_SUCCESS != CheckQKAndQKRopeShapeConsistency(faInfo, keyShape, keyRopeShape, "key") ||
        ge::GRAPH_SUCCESS != CheckPAKeyAndKeyRopeShapeConsistency(faInfo, keyShape, keyRopeShape) ||
        ge::GRAPH_SUCCESS != CheckTensorlistKeyAndKeyRopeShapeConsistency(faInfo)) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check rope dtype 与 query/key的数据类型一致
ge::graphStatus RopeChecker::CheckRopeDtypeConsistency(const FaTilingInfo &faInfo)
{
    if (enableNonQuant_) {
        OP_CHECK_IF((faInfo.inputQRopeType != faInfo.inputQType),
            OP_LOGE(faInfo.opName, "The datatype of query rope(%s) should be equal to query(%s).",
            DataTypeToSerialString(faInfo.inputQRopeType).c_str(), DataTypeToSerialString(faInfo.inputQType).c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((faInfo.inputKRopeType != faInfo.inputKvType),
            OP_LOGE(faInfo.opName, "The datatype of key rope(%s) should be equal to key(%s).",
            DataTypeToSerialString(faInfo.inputKRopeType).c_str(),
            DataTypeToSerialString(faInfo.inputKvType).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// 除D轴外 queryRope/keyRope其余轴需和query/key一致
ge::graphStatus RopeChecker::CheckQKAndQKRopeShapeConsistency(const FaTilingInfo &faInfo,
    const gert::Shape shape, const gert::Shape ropeShape, const std::string &inputName)
{
    if (inputName == "key" && faInfo.kvStorageMode != KvStorageMode::BATCH_CONTINUOUS) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string inputRopeName = inputName + "Rope";
    const uint32_t dimNum = shape.GetDimNum();
    const uint32_t ropeDimNum = ropeShape.GetDimNum();

    OP_CHECK_IF(dimNum != ropeDimNum,
        OP_LOGE(faInfo.opName,
            "The dim num of %s(%u) should be equal to %s(%u).",
            inputRopeName.c_str(), ropeDimNum, inputName.c_str(), dimNum),
        return ge::GRAPH_FAILED);
    
    for (uint32_t i = 0; i < dimNum - 1; i++) {
        OP_CHECK_IF(shape.GetDim(i) != ropeShape.GetDim(i),
            OP_LOGE(faInfo.opName,
                "The %uth dim of %s(%ld) should be equal to %s(%ld).",
                i, inputRopeName.c_str(), ropeShape.GetDim(i), inputName.c_str(), shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }

    if (faInfo.qLayout == FaLayout::BSH) {
        uint32_t tmpN = static_cast<uint32_t>(shape.GetDim(dimNum - 1) / faInfo.qkHeadDim);
        uint32_t ropeN = static_cast<uint32_t>(ropeShape.GetDim(ropeDimNum - 1) / faInfo.ropeHeadDim);
        OP_CHECK_IF(tmpN != ropeN,
            OP_LOGE(faInfo.opName,
                "The axis N of %s(%u) should be equal to %s(%u).",
                inputRopeName.c_str(), ropeN, inputName.c_str(), tmpN),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// PA场景 除D轴外 keyRope其余轴需和key一致
ge::graphStatus RopeChecker::CheckPAKeyAndKeyRopeShapeConsistency(const FaTilingInfo &faInfo,
    const gert::Shape &keyShape, const gert::Shape &keyRopeShape)
{
    if (faInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t keyDimNum = keyShape.GetDimNum();
    const uint32_t keyRopeDimNum = keyRopeShape.GetDimNum();
    OP_CHECK_IF(keyDimNum != keyRopeDimNum,
        OP_LOGE(faInfo.opName,
            "The dim num of keyRope(%u) should be equal to key(%u).",
            keyRopeDimNum, keyDimNum),
        return ge::GRAPH_FAILED);

    uint32_t keyBlockNum = 0;
    uint32_t keyRopeBlockNum = 0;
    uint32_t keyBlockSize = 0;
    uint32_t keyRopeBlockSize = 0;
    uint32_t keyN = 0;
    uint32_t keyRopeN = 0;

    if (keyDimNum == DIM_NUM_3) { // BBH
        keyBlockNum = keyShape.GetDim(DIM_NUM_0);
        keyBlockSize = keyShape.GetDim(DIM_NUM_1);
        keyN = keyShape.GetDim(DIM_NUM_2) / faInfo.qkHeadDim;
        keyRopeBlockNum = keyRopeShape.GetDim(DIM_NUM_0);
        keyRopeBlockSize = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_2) / faInfo.ropeHeadDim;
    } else if (keyDimNum == DIM_NUM_4) { // BNBD
        keyBlockNum = keyShape.GetDim(DIM_NUM_0);
        keyN = keyShape.GetDim(DIM_NUM_1);
        keyBlockSize = keyShape.GetDim(DIM_NUM_2);
        keyRopeBlockNum = keyRopeShape.GetDim(DIM_NUM_0);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeBlockSize = keyRopeShape.GetDim(DIM_NUM_2);
    } else { // BND1BD0
        keyBlockNum = keyShape.GetDim(DIM_NUM_0);
        keyN = keyShape.GetDim(DIM_NUM_1);
        keyBlockSize = keyShape.GetDim(DIM_NUM_3);
        keyRopeBlockNum = keyRopeShape.GetDim(DIM_NUM_0);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeBlockSize = keyRopeShape.GetDim(DIM_NUM_3);
    }

    OP_CHECK_IF(keyBlockNum != keyRopeBlockNum,
        OP_LOGE(faInfo.opName,
            "When page attention enable, the axis blockNum of keyRope(%u) should be equal to key(%u).",
            keyRopeBlockNum, keyBlockNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyBlockSize != keyRopeBlockSize,
        OP_LOGE(faInfo.opName,
            "When page attention enable, the axis blockSize of keyRope(%u) should be equal to key(%u).",
            keyRopeBlockSize, keyBlockSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyN != keyRopeN,
        OP_LOGE(faInfo.opName,
            "When page attention enable, the axis N of keyRope(%u) should be equal to key(%u).",
            keyRopeN, keyN),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// tensorlist keyRope和key的一致性校验
ge::graphStatus RopeChecker::CheckTensorlistKeyAndKeyRopeShapeConsistency(const FaTilingInfo &faInfo)
{
    if (faInfo.kvStorageMode != KvStorageMode::TENSOR_LIST) {
        return ge::GRAPH_SUCCESS;
    }

    if (faInfo.opParamInfo.keyRope.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    
    // b = len(kCache)
    const gert::Shape keyRopeShape = faInfo.opParamInfo.keyRope.tensor->GetStorageShape();
    OP_CHECK_IF(faInfo.kCache.size() != keyRopeShape.GetDim(DIM_NUM_0),
        OP_LOGE(faInfo.opName, "The axis B of keyRope(%u) should be equal to the tensor number of key(%u).",
                keyRopeShape.GetDim(DIM_NUM_0), faInfo.kCache.size()),
        return ge::GRAPH_FAILED);

    // k_s = krope_s
    const std::string inputLayout = faInfo.opParamInfo.layOut;
    int64_t keyRopeS = 0;
    if (inputLayout == "BNSD" || inputLayout == "BNSD_BSND") {
        keyRopeS = keyRopeShape.GetDim(DIM_NUM_2);
        for (uint32_t i = 0; i < faInfo.kCache.size(); i++) {
            const gert::Shape keyShape = faInfo.kCache[i]->GetStorageShape();
            OP_CHECK_IF(keyShape.GetDim(DIM_NUM_2) != keyRopeS,
                OP_LOGE(faInfo.opName, "The axis S of keyRope(%ld) should be equal to key(%ld).",
                    keyRopeS, keyShape.GetDim(DIM_NUM_2)),
                return ge::GRAPH_FAILED);
        }
    } else {
        keyRopeS = keyRopeShape.GetDim(DIM_NUM_1);
        for (uint32_t i = 0; i < faInfo.kCache.size(); i++) {
            const gert::Shape keyShape = faInfo.kCache[i]->GetStorageShape();
            OP_CHECK_IF(keyShape.GetDim(DIM_NUM_1) != keyRopeS,
                OP_LOGE(faInfo.opName, "The axis S of keyRope(%ld) should be equal to key(%ld).",
                    keyRopeS, keyShape.GetDim(DIM_NUM_1)),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckRopeExistence(const FaTilingInfo &faInfo)
{
    const gert::Tensor *queryRopeTensor = faInfo.opParamInfo.queryRope.tensor;
    const gert::Tensor *keyRopeTensor = faInfo.opParamInfo.keyRope.tensor;
    
    // 一个空 一个非空
    OP_CHECK_IF((queryRopeTensor == nullptr && keyRopeTensor != nullptr),
        OP_LOGE(faInfo.opName, "queryRope is null, but keyRope is not null, "
            "they should be consistent."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((keyRopeTensor == nullptr && queryRopeTensor != nullptr),
    OP_LOGE(faInfo.opName, "keyRope is null, but queryRope is not null, "
        "they should be consistent."),
    return ge::GRAPH_FAILED);

    // rope split时 rope必须都存在
    if (faInfo.ropeMode == RopeMode::ROPE_SPLIT) {
        OP_LOGI(faInfo.opName, "Rope mode is ROPE_SPLIT.");
        OP_CHECK_IF((queryRopeTensor == nullptr || keyRopeTensor == nullptr),
            OP_LOGE(faInfo.opName,
            "When rope exists, queryRope or keyRope should not be null."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckFeatureSupport(const FaTilingInfo &faInfo)
{
    if (faInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    // 不支持 prefix
    OP_CHECK_IF(faInfo.sysPrefixFlag,
        OP_LOGE(faInfo.opName, "When rope exists, system prefix is not supported."),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(faInfo.pseShiftFlag,
        OP_LOGE(faInfo.opName,
                "When rope exists, pse is not supported."),
            return ge::GRAPH_FAILED);
    
    // 不支持alibepse
    OP_CHECK_IF(faInfo.enableAlibiPse,
        OP_LOGE(faInfo.opName, "When rope exists, pseType = 2/3 is not supported."),
        return ge::GRAPH_FAILED);
    
    if (faInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        // 不支持左padding
        OP_CHECK_IF(faInfo.qPaddingSizeFlag || faInfo.kvPaddingSizeFlag,
            OP_LOGE(faInfo.opName, "When rope exsists, leftPadding is not supported."),
            return ge::GRAPH_FAILED);
        // 不支持tensorlist
        OP_CHECK_IF(faInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
            OP_LOGE(faInfo.opName, "When rope exsists, tensorlist is not supported."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckFeatureDecodeMLA(const FaTilingInfo &faInfo)
{
    if (faInfo.mlaMode != MlaMode::ROPE_SPLIT_D512) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(faInfo.qPaddingSizeFlag || faInfo.kvPaddingSizeFlag,
        OP_LOGE(faInfo.opName,
                "In the Decode MLA scenario, left padding is not supported."),
            return ge::GRAPH_FAILED);

     OP_CHECK_IF(faInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
        OP_LOGE(faInfo.opName,
                "In the Decode MLA scenario, tensorlist is not supported."),
            return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckFeatureAntiQuant(const FaTilingInfo &faInfo)
{
    // 不支持伪量化
    OP_CHECK_IF(faInfo.opParamInfo.queryRope.tensor != nullptr || faInfo.opParamInfo.keyRope.tensor != nullptr,
        OP_LOGE(faInfo.opName,
                "Rope is not supported in antiquant scenario."),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// MLA D=512时 N:1/2/4/8/16/32/64/128 全量化 S1：1-16
ge::graphStatus RopeChecker::CheckAxisSupport(const FaTilingInfo &faInfo)
{
    if (faInfo.mlaMode != MlaMode::ROPE_SPLIT_D512) {
        return ge::GRAPH_SUCCESS;
    }

    constexpr uint32_t maxQuerySeqLenForMLAFullquant = 16U;
    static const std::set<uint32_t> supportNumHeadForMLA = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U};

    OP_CHECK_IF((faInfo.s1Size < NUM1),
        OP_LOGE(faInfo.opName,
            "In the Decode MLA scenario, sequence length(%u) of query should be larger than 0.",
            faInfo.s1Size),
        return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(enableFullQuant_ && (faInfo.s1Size > maxQuerySeqLenForMLAFullquant),
        OP_LOGE(faInfo.opName,
            "In the Decode MLA fullquant scenario, sequence length(%u) of query should be in range of [1, %u].",
            faInfo.s1Size, maxQuerySeqLenForMLAFullquant),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((supportNumHeadForMLA.find(faInfo.n1Size) == supportNumHeadForMLA.end()),
        OP_LOGE(faInfo.opName,
            "In the Decode MLA scenario, "
            "the heads num(%u) of query should be in range of {1, 2, 4, 8, 16, 32, 64, 128}.", faInfo.n1Size),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((*faInfo.opParamInfo.kvHeadNums != NUM1),
        OP_LOGE(faInfo.opName,
            "In the Decode MLA scenario, the heads num(%d) of key should be 1.",
            *faInfo.opParamInfo.kvHeadNums),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckRopeDtype(faInfo) ||
        ge::GRAPH_SUCCESS != CheckRopeDSizeSupport(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckRopeExistence(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (faInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckFeatureSupport(faInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureDecodeMLA(faInfo)) {
            return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckShapeSupport(faInfo) ||
        ge::GRAPH_SUCCESS != CheckRopeDtypeConsistency(faInfo) ||
        ge::GRAPH_SUCCESS != CheckAxisSupport(faInfo)) {
            return ge::GRAPH_FAILED;
    }

    if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureAntiQuant(faInfo)) {
            return ge::GRAPH_FAILED;
        };
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (faInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optilingForBuildPKG
