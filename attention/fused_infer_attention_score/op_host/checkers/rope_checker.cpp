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
#include <sstream>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "rope_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// check rope dtype
ge::graphStatus RopeChecker::CheckRopeDtype(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((fiaInfo.inputQRopeType != ge::DT_BF16 && fiaInfo.inputQRopeType != ge::DT_FLOAT16),
        OP_LOGE(kFiaOpName, "In op [%s], the data type of [%s] is not supported, got [%s]", kFiaOpName, "tensor", DataTypeToSerialString(fiaInfo.inputQRopeType).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((fiaInfo.inputKRopeType != ge::DT_BF16 && fiaInfo.inputKRopeType != ge::DT_FLOAT16),
        OP_LOGE(kFiaOpName, "In op [%s], the data type of [%s] is not supported, got [%s]", kFiaOpName, "tensor", DataTypeToSerialString(fiaInfo.inputKRopeType).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check rope DSize
ge::graphStatus RopeChecker::CheckRopeDSizeSupport(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((fiaInfo.ropeHeadDim != NUM_64),
        ({ std::ostringstream oss; oss << "When rope exist, the d size of rope must be 64, but current is ";
        oss << fiaInfo.ropeHeadDim;
        oss << "."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckQDsizeSupport(const FiaTilingInfo &fiaInfo)
{
    if (enableFullQuant_ && fiaInfo.fullQuantMode == FiaFullQuantMode::MXFP8_FULL_QUANT) {
        OP_CHECK_IF((fiaInfo.qkHeadDim != NUM_64 && fiaInfo.qkHeadDim != NUM_128),
        ({ std::ostringstream oss; oss << "In MXFP8 fullquant scenario, when rope exist, the d size of query and key must be 64 or 128, but current is ";
        oss << fiaInfo.qkHeadDim;
        oss << "."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((fiaInfo.qkHeadDim != NUM_128 && fiaInfo.qkHeadDim != NUM_512),
        ({ std::ostringstream oss; oss << "When rope exist, the d size of query and key must be 128 or 512, but current is ";
        oss << fiaInfo.qkHeadDim;
        oss << "."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckShapeSupport(const FiaTilingInfo &fiaInfo)
{
    // check qk head dim and rope head dim
    if (ge::GRAPH_SUCCESS != CheckQDsizeSupport(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }

    const gert::Shape queryShape = fiaInfo.opParamInfo.query.shape->GetStorageShape();
    const gert::Shape queryRopeShape = fiaInfo.opParamInfo.queryRope.tensor->GetStorageShape();
    const gert::Shape keyShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
    const gert::Shape keyRopeShape = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape();

    // check queryRope shape
    if (ge::GRAPH_SUCCESS != CheckQKAndQKRopeShapeConsistency(fiaInfo, queryShape, queryRopeShape, "query")) {
            return ge::GRAPH_FAILED;
    }
    // check keyRope shape
    if (ge::GRAPH_SUCCESS != CheckQKAndQKRopeShapeConsistency(fiaInfo, keyShape, keyRopeShape, "key") ||
        ge::GRAPH_SUCCESS != CheckPAKeyAndKeyRopeShapeConsistency(fiaInfo, keyShape, keyRopeShape) ||
        ge::GRAPH_SUCCESS != CheckTensorlistKeyAndKeyRopeShapeConsistency(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check rope dtype 与 query/key的数据类型一致
ge::graphStatus RopeChecker::CheckRopeDtypeConsistency(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) {
        OP_CHECK_IF((fiaInfo.inputQRopeType != fiaInfo.inputQType),
            ({ std::ostringstream oss; oss << "The datatype of query rope(";
        oss << DataTypeToSerialString(fiaInfo.inputQRopeType).c_str();
        oss << ") should be equal to query(";
        oss << DataTypeToSerialString(fiaInfo.inputQType).c_str();
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((fiaInfo.inputKRopeType != fiaInfo.inputKvType),
            ({ std::ostringstream oss; oss << "The datatype of key rope(";
        oss << DataTypeToSerialString(fiaInfo.inputKRopeType).c_str();
        oss << ") should be equal to key(";
        oss << DataTypeToSerialString(fiaInfo.inputKvType).c_str();
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// 除D轴外 queryRope/keyRope其余轴需和query/key一致
ge::graphStatus RopeChecker::CheckQKAndQKRopeShapeConsistency(const FiaTilingInfo &fiaInfo,
    const gert::Shape shape, const gert::Shape ropeShape, const std::string &inputName)
{
    if (inputName == "key" && fiaInfo.kvStorageMode != KvStorageMode::BATCH_CONTINUOUS) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string inputRopeName = inputName + "Rope";
    const uint32_t dimNum = shape.GetDimNum();
    const uint32_t ropeDimNum = ropeShape.GetDimNum();

    OP_CHECK_IF(dimNum != ropeDimNum,
        ({ std::ostringstream oss; oss << "The dim num of ";
        oss << inputRopeName.c_str();
        oss << "(";
        oss << ropeDimNum;
        oss << ") should be equal to ";
        oss << inputName.c_str();
        oss << "(";
        oss << dimNum;
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    
    for (uint32_t i = 0; i < dimNum - 1; i++) {
        OP_CHECK_IF(shape.GetDim(i) != ropeShape.GetDim(i),
            ({ std::ostringstream oss; oss << "The ";
        oss << i;
        oss << "th dim of ";
        oss << inputRopeName.c_str();
        oss << "(";
        oss << ropeShape.GetDim(i);
        oss << ") should be equal to ";
        oss << inputName.c_str();
        oss << "(";
        oss << shape.GetDim(i);
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);
    }

    if (fiaInfo.qLayout == FiaLayout::BSH) {
        uint32_t tmpN = static_cast<uint32_t>(shape.GetDim(dimNum - 1) / fiaInfo.qkHeadDim);
        uint32_t ropeN = static_cast<uint32_t>(ropeShape.GetDim(ropeDimNum - 1) / fiaInfo.ropeHeadDim);
        OP_CHECK_IF(tmpN != ropeN,
            ({ std::ostringstream oss; oss << "The axis N of ";
        oss << inputRopeName.c_str();
        oss << "(";
        oss << ropeN;
        oss << ") should be equal to ";
        oss << inputName.c_str();
        oss << "(";
        oss << tmpN;
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// PA场景 除D轴外 keyRope其余轴需和key一致
ge::graphStatus RopeChecker::CheckPAKeyAndKeyRopeShapeConsistency(const FiaTilingInfo &fiaInfo,
    const gert::Shape &keyShape, const gert::Shape &keyRopeShape)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t keyDimNum = keyShape.GetDimNum();
    const uint32_t keyRopeDimNum = keyRopeShape.GetDimNum();
    OP_CHECK_IF(keyDimNum != keyRopeDimNum,
        ({ std::ostringstream oss; oss << "The dim num of keyRope(";
        oss << keyRopeDimNum;
        oss << ") should be equal to key(";
        oss << keyDimNum;
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
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
        keyN = keyShape.GetDim(DIM_NUM_2) / fiaInfo.qkHeadDim;
        keyRopeBlockNum = keyRopeShape.GetDim(DIM_NUM_0);
        keyRopeBlockSize = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_2) / fiaInfo.ropeHeadDim;
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
        ({ std::ostringstream oss; oss << "When page attention enable, the axis blockNum of keyRope(";
        oss << keyRopeBlockNum;
        oss << ") should be equal to key(";
        oss << keyBlockNum;
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyBlockSize != keyRopeBlockSize,
        ({ std::ostringstream oss; oss << "When page attention enable, the axis blockSize of keyRope(";
        oss << keyRopeBlockSize;
        oss << ") should be equal to key(";
        oss << keyBlockSize;
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyN != keyRopeN,
        ({ std::ostringstream oss; oss << "When page attention enable, the axis N of keyRope(";
        oss << keyRopeN;
        oss << ") should be equal to key(";
        oss << keyN;
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// tensorlist keyRope和key的一致性校验
ge::graphStatus RopeChecker::CheckTensorlistKeyAndKeyRopeShapeConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::TENSOR_LIST) {
        return ge::GRAPH_SUCCESS;
    }

    if (fiaInfo.opParamInfo.keyRope.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    
    // b = len(kCache)
    const gert::Shape keyRopeShape = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape();
    OP_CHECK_IF(fiaInfo.kCache.size() != keyRopeShape.GetDim(DIM_NUM_0),
        ({ std::ostringstream oss; oss << "The axis B of keyRope(";
        oss << keyRopeShape.GetDim(DIM_NUM_0);
        oss << ") should be equal to the tensor number of key(";
        oss << fiaInfo.kCache.size();
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);

    // k_s = krope_s
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;
    int64_t keyRopeS = 0;
    int64_t keyRopeN = 0;
    if (inputLayout == "BNSD" || inputLayout == "BNSD_BSND") {
        keyRopeS = keyRopeShape.GetDim(DIM_NUM_2);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_1);
        for (uint32_t i = 0; i < fiaInfo.kCache.size(); i++) {
            const gert::Shape keyShape = fiaInfo.kCache[i]->GetStorageShape();
            OP_CHECK_IF(keyShape.GetDim(DIM_NUM_2) != keyRopeS,
                ({ std::ostringstream oss; oss << "The axis S of keyRope(";
        oss << keyRopeS;
        oss << ") should be equal to key(";
        oss << keyShape.GetDim(DIM_NUM_2);
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(keyShape.GetDim(DIM_NUM_1) != keyRopeN,
                ({ std::ostringstream oss; oss << "The axis N of keyRope(";
        oss << keyRopeN;
        oss << ") should be equal to key(";
        oss << keyShape.GetDim(DIM_NUM_1);
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                return ge::GRAPH_FAILED);
        }
    } else if (inputLayout == "BSH") {
        keyRopeS = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_2) / fiaInfo.ropeHeadDim;
        for (uint32_t i = 0; i < fiaInfo.kCache.size(); i++) {
            const gert::Shape keyShape = fiaInfo.kCache[i]->GetStorageShape();
            OP_CHECK_IF(keyShape.GetDim(DIM_NUM_1) != keyRopeS,
                ({ std::ostringstream oss; oss << "The axis S of keyRope(";
        oss << keyRopeS;
        oss << ") should be equal to key(";
        oss << keyShape.GetDim(DIM_NUM_1);
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(keyShape.GetDim(DIM_NUM_2) / fiaInfo.qkHeadDim != keyRopeN,
                ({ std::ostringstream oss; oss << "The axis N of keyRope(";
        oss << keyRopeN;
        oss << ") should be equal to key(";
        oss << keyShape.GetDim(DIM_NUM_2) / fiaInfo.qkHeadDim;
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                return ge::GRAPH_FAILED);
        }
    } else if (inputLayout == "BSND") {
        keyRopeS = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_2);
        for (uint32_t i = 0; i < fiaInfo.kCache.size(); i++) {
            const gert::Shape keyShape = fiaInfo.kCache[i]->GetStorageShape();
            OP_CHECK_IF(keyShape.GetDim(DIM_NUM_1) != keyRopeS,
                ({ std::ostringstream oss; oss << "The axis S of keyRope(";
        oss << keyRopeS;
        oss << ") should be equal to key(";
        oss << keyShape.GetDim(DIM_NUM_1);
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(keyShape.GetDim(DIM_NUM_2) != keyRopeN,
                ({ std::ostringstream oss; oss << "The axis N of keyRope(";
        oss << keyRopeN;
        oss << ") should be equal to key(";
        oss << keyShape.GetDim(DIM_NUM_2);
        oss << ")."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckRopeExistence(const FiaTilingInfo &fiaInfo)
{
    const gert::Tensor *queryRopeTensor = fiaInfo.opParamInfo.queryRope.tensor;
    const gert::Tensor *keyRopeTensor = fiaInfo.opParamInfo.keyRope.tensor;
    
    // 一个空 一个非空
    OP_CHECK_IF((queryRopeTensor == nullptr && keyRopeTensor != nullptr),
        OP_LOGE(kFiaOpName, "The op [%s] missing [%s] parameter", kFiaOpName, "queryrope"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((keyRopeTensor == nullptr && queryRopeTensor != nullptr),
    OP_LOGE(kFiaOpName, "The op [%s] missing [%s] parameter", kFiaOpName, "keyrope"),
    return ge::GRAPH_FAILED);

    // rope split时 rope必须都存在
    if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
        OP_LOGI(fiaInfo.opName, "Rope mode is ROPE_SPLIT.");
        OP_CHECK_IF((queryRopeTensor == nullptr || keyRopeTensor == nullptr),
            ({ std::ostringstream oss; oss << "When rope exists, queryRope or keyRope should not be null."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckFeatureSupport(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    // 不支持 prefix
    OP_CHECK_IF(fiaInfo.sysPrefixFlag,
        ({ std::ostringstream oss; oss << "When rope exists, system prefix is not supported."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(fiaInfo.pseShiftFlag,
        ({ std::ostringstream oss; oss << "When rope exists, pse is not supported."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);
    
    // 不支持alibepse
    OP_CHECK_IF(fiaInfo.enableAlibiPse,
        ({ std::ostringstream oss; oss << "When rope exists, pseType = 2/3 is not supported."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    
    if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        // 不支持左padding
        OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
            ({ std::ostringstream oss; oss << "When rope exsists, leftPadding is not supported."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);
        // 不支持tensorlist
        OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
            ({ std::ostringstream oss; oss << "When rope exsists, tensorlist is not supported."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);
        // 不支持后量化
        OP_CHECK_IF(fiaInfo.isOutQuantEnable,
            ({ std::ostringstream oss; oss << "When rope exsists, postQuant is not supported."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckFeatureDecodeMLA(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
        ({ std::ostringstream oss; oss << "In the Decode MLA scenario, left padding is not supported."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);

     OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
        ({ std::ostringstream oss; oss << "In the Decode MLA scenario, tensorlist is not supported."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckFeatureAntiQuant(const FiaTilingInfo &fiaInfo)
{
    // 不支持伪量化
    OP_CHECK_IF(fiaInfo.opParamInfo.queryRope.tensor != nullptr || fiaInfo.opParamInfo.keyRope.tensor != nullptr,
        ({ std::ostringstream oss; oss << "Rope is not supported in antiquant scenario."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check QS size
ge::graphStatus RopeChecker::CheckQSSize(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512 || fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }

    // 全量化场景下 QS仅支持1-16
    constexpr uint32_t maxQuerySeqLenForMLAFullquant = 16U;
    

    OP_CHECK_IF((fiaInfo.s1Size < NUM1),
        ({ std::ostringstream oss; oss << "In the Decode MLA scenario, sequence length(";
        oss << fiaInfo.s1Size;
        oss << ") of query should be larger than 0."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(enableFullQuant_ && (fiaInfo.s1Size > maxQuerySeqLenForMLAFullquant),
        ({ std::ostringstream oss; oss << "In the Decode MLA fullquant scenario, sequence length(";
        oss << fiaInfo.s1Size;
        oss << ") of query should be in range of [1, ";
        oss << maxQuerySeqLenForMLAFullquant;
        oss << "]."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Decode MLA N1:1/2/4/8/16/32/64/128 N2:1
ge::graphStatus RopeChecker::CheckNSize(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512) {
        return ge::GRAPH_SUCCESS;
    }
    static const std::set<uint32_t> supportNumHeadForMLA = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U};
    OP_CHECK_IF((supportNumHeadForMLA.find(fiaInfo.n1Size) == supportNumHeadForMLA.end()),
        ({ std::ostringstream oss; oss << "In the Decode MLA scenario, the heads num(";
        oss << fiaInfo.n1Size;
        oss << ") of query should be in range of {1, 2, 4, 8, 16, 32, 64, 128}."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((*fiaInfo.opParamInfo.kvHeadNums != NUM1),
        ({ std::ostringstream oss; oss << "In the Decode MLA scenario, the heads num(";
        oss << *fiaInfo.opParamInfo.kvHeadNums;
        oss << ") of key should be 1."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckAxisSupport(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckQSSize(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckNSize(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckRopeDtype(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckRopeDSizeSupport(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckRopeExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckFeatureSupport(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureDecodeMLA(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckShapeSupport(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckRopeDtypeConsistency(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckAxisSupport(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }

    if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureAntiQuant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        };
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
