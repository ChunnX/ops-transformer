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
 * \file pse_checker.cpp
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
#include "pse_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

constexpr int32_t PSE_SHIFT_MAX = 1048576;  // 2^20
constexpr int32_t PSE_SHIFT_MIN = -1048576; // -2^20
constexpr int64_t PSE_OUTER_MUL_ADD_TYPE = 0;
constexpr int64_t PSE_OUTER_ADD_MUL_TYPE = 1;
constexpr int64_t PSE_INNER_MUL_ADD_TYPE = 2;
constexpr int64_t PSE_INNER_MUL_ADD_SQRT_TYPE = 3;
constexpr int64_t PSE_NONE_TYPE = 9;

// singlepara
ge::graphStatus PSEChecker::CheckPseType(const FiaTilingInfo &fiaInfo)
{
    // 校验pseType的合法性
    if (fiaInfo.opParamInfo.pseType == nullptr) {
        // 若pseType不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    int64_t pseType = *fiaInfo.opParamInfo.pseType;
    // pseType支持范围为0
    OP_CHECK_IF((pseType != PSE_OUTER_MUL_ADD_TYPE),
                ({ std::ostringstream oss; oss << "pseType(";
        oss << pseType;
        oss << ") is not supported. pseType must be 0."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckPseShiftDataType(const FiaTilingInfo &fiaInfo)
{
    // 校验pseShift的数据类型
    if (fiaInfo.opParamInfo.pseType == nullptr || fiaInfo.opParamInfo.pseShift.desc == nullptr) {
        // 若pseType或者pseShift的数据类型不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    int64_t pseType = *fiaInfo.opParamInfo.pseType;
    auto &pseShiftDesc = fiaInfo.opParamInfo.pseShift.desc;
    if (fiaInfo.enableAlibiPse) {
        // 使能alibi，pseShift的数据类型必须为FLOAT32
        OP_CHECK_IF((pseShiftDesc->GetDataType() != ge::DT_FLOAT),
                    ({ std::ostringstream oss; oss << "The datatype of pseShift(";
        oss << DataTypeToSerialString(pseShiftDesc->GetDataType()).c_str();
        oss << ") is not FLOAT32. The datatype of pseShift must be FLOAT32 when pseType is 2 or 3."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                    return ge::GRAPH_FAILED);
    } else {
        // 不使能alibi
        // query的datatype为FLOAT16、INT8时，pseShift的datatype应为FLOAT16
        if (fiaInfo.inputQType == ge::DT_FLOAT16 || fiaInfo.inputQType == ge::DT_INT8) {
            OP_CHECK_IF((pseShiftDesc->GetDataType() != ge::DT_FLOAT16),
                        ({ std::ostringstream oss; oss << "The datatype of pseShift(";
        oss << DataTypeToSerialString(pseShiftDesc->GetDataType()).c_str();
        oss << ") is not FLOAT16. The datatype of pseShift must be FLOAT16 when the datatype of query is FLOAT16 or INT8 and pseType is 0."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                        return ge::GRAPH_FAILED);
        }
        // query的datatype为BFLOAT16时，pseShift的datatype应为BFLOAT16
        if (fiaInfo.inputQType == ge::DT_BF16) {
            OP_CHECK_IF((pseShiftDesc->GetDataType() != ge::DT_BF16),
                        ({ std::ostringstream oss; oss << "The datatype of pseShift(";
        oss << DataTypeToSerialString(pseShiftDesc->GetDataType()).c_str();
        oss << ") is not BFLOAT16. The datatype of pseShift must be BFLOAT16 when the datatype of query is BFLOAT16 and pseType is 0."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckPseShiftShape(const FiaTilingInfo &fiaInfo)
{
    // 校验pseShift的shape
    auto &pseShiftTensor = fiaInfo.opParamInfo.pseShift.tensor;
    if (pseShiftTensor == nullptr) {
        // 若pseShiftTensor不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    auto pseShiftShape = pseShiftTensor->GetStorageShape();
    if (fiaInfo.enableAlibiPse) {
        // pseType为2或3时，使能alibi，pseShist的shape应为(N)
        uint32_t numHeads = *fiaInfo.opParamInfo.numHeads;
        gert::Shape expectedShapeN = gert::Shape({numHeads});
        OP_CHECK_IF((pseShiftShape != expectedShapeN),
                    ({ std::ostringstream oss; oss << "The shape of pseShift is not supported. The shape of pseShift must be [N]([";
        oss << numHeads;
        oss << "]) when pseType is 2 or 3."; OP_LOGE(kFiaOpName, "In op [%s], the shape of [%s] is not supported, got [%s]. Constraint: [%s]", kFiaOpName, "tensor", oss.str().c_str(), "see log"); }),
                    return ge::GRAPH_FAILED);
    } else {
        uint32_t batchSize = fiaInfo.bSize;
        uint32_t n1Size = fiaInfo.n1Size;
        uint32_t s1Size = fiaInfo.s1Size;
        int64_t s2Size = fiaInfo.s2Size;
        // pseShift的维度必须为4
        uint32_t pseShiftDimNum = pseShiftShape.GetDimNum();
        OP_CHECK_IF(pseShiftDimNum != DIM_NUM_4,
                    ({ std::ostringstream oss; oss << "The dimension number of pseShift must be 4 when pseType is 0, but the current dimensioin number of pseShift is ";
        oss << pseShiftDimNum;
        oss << "."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                    return ge::GRAPH_FAILED);
        uint32_t pseShiftBatch = pseShiftShape.GetDim(DIM_NUM_0);
        uint32_t pseShiftN = pseShiftShape.GetDim(DIM_NUM_1);
        uint32_t pseShiftS1 = pseShiftShape.GetDim(DIM_NUM_2);
        uint32_t pseShiftS2 = pseShiftShape.GetDim(DIM_NUM_3);
        uint32_t actualSharedPrefixLen = 0;
        if (fiaInfo.opParamInfo.actualSharedPrefixLen.tensor != nullptr &&
            fiaInfo.opParamInfo.actualSharedPrefixLen.tensor->GetStorageShape().GetShapeSize() != 0) {
            if (fiaInfo.opParamInfo.actualSharedPrefixLen.tensor->GetData<int64_t>() != nullptr) {
                actualSharedPrefixLen = fiaInfo.opParamInfo.actualSharedPrefixLen.tensor->GetData<int64_t>()[0];
            }
        }
        if (pseShiftS1 > 1) {
            // P_S1 > 1分支
            OP_CHECK_IF((pseShiftBatch != 1 && pseShiftBatch != batchSize) || (pseShiftN != n1Size) ||
                            (pseShiftS1 < s1Size) || (pseShiftS2 < s2Size + actualSharedPrefixLen),
                        ({ std::ostringstream oss; oss << "pseShift shape must be [1 or ";
        oss << batchSize;
        oss << ",";
        oss << n1Size;
        oss << ",>=";
        oss << s1Size;
        oss << ",>=";
        oss << s2Size + actualSharedPrefixLen;
        oss << "], but now is [";
        oss << pseShiftBatch;
        oss << ",";
        oss << pseShiftN;
        oss << ",";
        oss << pseShiftS1;
        oss << ",";
        oss << pseShiftS2;
        oss << "]."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                        return ge::GRAPH_FAILED);
        } else {
            // P_S1 = 1分支
            OP_CHECK_IF((pseShiftBatch != 1 && pseShiftBatch != batchSize) || (pseShiftN != n1Size) ||
                            (pseShiftS1 != 1) || (pseShiftS2 < s2Size + actualSharedPrefixLen),
                        ({ std::ostringstream oss; oss << "The shape of pseShift must be [1 or ";
        oss << batchSize;
        oss << ",";
        oss << n1Size;
        oss << ",1,>=";
        oss << s2Size + actualSharedPrefixLen;
        oss << "], but now is [";
        oss << pseShiftBatch;
        oss << ",";
        oss << pseShiftN;
        oss << ",";
        oss << pseShiftS1;
        oss << ",";
        oss << pseShiftS2;
        oss << "]."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

// existence
ge::graphStatus PSEChecker::CheckPseShiftExistence(const FiaTilingInfo &fiaInfo)
{
    // 校验pseShift的存在性
    if (fiaInfo.enableAlibiPse) {
        // pseType为2或3时，使能alibi，pseShift必须传
        OP_CHECK_IF((fiaInfo.opParamInfo.pseShift.tensor == nullptr),
                    ({ std::ostringstream oss; oss << "pseShift does not exist. pseShift must exist when pseType is 2 or 3."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                    return ge::GRAPH_FAILED);
    }
    // 校验pseShift的desc的存在性，若pseShift存在，其desc也必须存在
    if (fiaInfo.opParamInfo.pseShift.tensor != nullptr) {
        OP_CHECK_IF((fiaInfo.opParamInfo.pseShift.desc == nullptr),
                    ({ std::ostringstream oss; oss << "pseShift exists, but its datatype does not exist."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// feature
ge::graphStatus PSEChecker::CheckFeaturePA(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.opParamInfo.pseType == nullptr || fiaInfo.enableAlibiPse) {
        // 若不存在pseType或pseType = 2/3，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    // Page attention使能场景下，传入的PseShift的最后一维需要大于等于maxBlockNumPerSeq * blockSize
    if (*fiaInfo.opParamInfo.pseType != 0 && fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        uint32_t pseShiftS2 = fiaInfo.pseShiftS2;
        int32_t blockSize = fiaInfo.blockSize;
        uint32_t maxBlockNumPerBatch = fiaInfo.maxBlockNumPerBatch;
        OP_CHECK_IF(pseShiftS2 < maxBlockNumPerBatch * blockSize,
                    ({ std::ostringstream oss; oss << "The last dimension(";
        oss << pseShiftS2;
        oss << ") of pseShift is less than maxBlockNumPerBatch(";
        oss << maxBlockNumPerBatch;
        oss << ") * blockSize(";
        oss << blockSize;
        oss << "). The last dimension of pseShift should be larger than or equal to maxBlockNumPerBatch * blockSize when page attention is enabled."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckerFeatureCrossover(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    // 校验使能alibi时与其他特性交叉的约束
    if (fiaInfo.enableAlibiPse || fiaInfo.pseShiftFlag) {
        // D不等长时，不支持pse
        OP_CHECK_IF(fiaInfo.isQKVDDifferent,
                    ({ std::ostringstream oss; oss << "pseShift is not supported when query and key headdim is not equal to value headdim."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                    return ge::GRAPH_FAILED);
    }
    if (fiaInfo.enableAlibiPse) {
        // 使能alibi时，MLA不支持pse
        OP_CHECK_IF(fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128 || fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128 ||
                        fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512,
                    ({ std::ostringstream oss; oss << "MLA do not support pseShift."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }), return ge::GRAPH_FAILED);
    } else if (fiaInfo.pseShiftFlag) {
        if (fiaInfo.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }
        // 非alibi时，MLA，不支持pse
        OP_CHECK_IF(fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128 || fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128 ||
                        fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512,
                    ({ std::ostringstream oss; oss << "MLA do not support pseShift."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }), return ge::GRAPH_FAILED);
        // pse使能时，若inputLayout为BSH_BNSD/BSND_BNSD/TND/NTD/NTD_TND/TND_NTD，不支持pse
        OP_CHECK_IF(layoutStr == "BSH_BNSD" || layoutStr == "BSND_BNSD" || layoutStr == "TND" || layoutStr == "NTD" ||
                        layoutStr == "NTD_TND" || layoutStr == "TND_NTD",
                    ({ std::ostringstream oss; oss << "pse is not supported when the inputLayout is BSH_BNSD/BSND_BNSD/TND/NTD/NTD_TND/TND_NTD."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// multipara
ge::graphStatus PSEChecker::CheckAlibiStartIdx(const FiaTilingInfo &fiaInfo)
{
    // 使能alibi时，校验qStartIdx和kvStartIdx的取值范围，和kvStartIdx - qStartIdx的取值范围
    // qstartIdx的取值范围应满足[-2147483648, 2147483647]
    auto &qStartIdxTensor = fiaInfo.opParamInfo.qStartIdx.tensor;
    auto &kvStartIdxTensor = fiaInfo.opParamInfo.kvStartIdx.tensor;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    if (qStartIdxTensor != nullptr && qStartIdxTensor->GetShapeSize() >= 1) {
        const int64_t *qStartIdxs = qStartIdxTensor->GetData<int64_t>();
        if (qStartIdxs != nullptr) {
            qStartIdx = qStartIdxs[0];
            OP_CHECK_IF(qStartIdx > INT32_MAX || qStartIdx < INT32_MIN,
                        ({ std::ostringstream oss; oss << "qStartIdx(";
        oss << qStartIdx;
        oss << ") is not supported. qStartIdx should belongs to [-2147483648, 2147483647]."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                        return ge::GRAPH_FAILED);
        }
    }
    // kvStartIdx的取值范围应满足[-2147483648, 2147483647]
    if (kvStartIdxTensor != nullptr && kvStartIdxTensor->GetShapeSize() >= 1) {
        const int64_t *kvStartIdxs = kvStartIdxTensor->GetData<int64_t>();
        if (kvStartIdxs != nullptr) {
            kvStartIdx = kvStartIdxs[0];
            OP_CHECK_IF(kvStartIdx > INT32_MAX || kvStartIdx < INT32_MIN,
                        ({ std::ostringstream oss; oss << "kvStartIdx(";
        oss << kvStartIdx;
        oss << ") is not supported. kvStartIdx should belongs to [-2147483648, 2147483647]."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                        return ge::GRAPH_FAILED);
        }
    }
    // kvStartIdx - qStartIdx的取值范围应满足[-1048576, 1048576]
    OP_CHECK_IF((kvStartIdx - qStartIdx > PSE_SHIFT_MAX) || (kvStartIdx - qStartIdx) < PSE_SHIFT_MIN,
                ({ std::ostringstream oss; oss << "kvStartIdx - qStartIdx(";
        oss << kvStartIdx - qStartIdx;
        oss << ") is not supported. kvStartIdx - qStartIdx should belongs to [-1048576, 1048576]."; OP_LOGE(kFiaOpName, "The op [%s] received bad params, the reason is: [%s]", kFiaOpName, oss.str().c_str()); }),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckPseType(fiaInfo) || ge::GRAPH_SUCCESS != CheckPseShiftDataType(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckPseShiftShape(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckPseShiftExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeaturePA(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckerFeatureCrossover(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckAlibiStartIdx(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling