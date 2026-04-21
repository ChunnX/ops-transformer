/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file actual_seq_len_checker.cpp
 * \brief Checker for cu_seqlens_q/kv (B+1) and seqused_q/kv (B) parameters
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "seq_len_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

inline std::string FaLayoutToString(FaLayout layout)
{
    switch (layout) {
        case FaLayout::BSND:
            return "BSND";
        case FaLayout::BNSD:
            return "BNSD";
        case FaLayout::TND:
            return "TND";
        case FaLayout::PA_BBND:
            return "PA_BBND";
        case FaLayout::PA_BNBD:
            return "PA_BNBD";
        case FaLayout::PA_Nz:
            return "PA_Nz";
        default:
            return "UNKNOWN";
    }
}

ge::graphStatus ActualSeqLenChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    auto &sequsedQTensor = faInfo.opParamInfo.sequsedQ.tensor;
    if (sequsedQTensor != nullptr) {
        const gert::CompileTimeTensorDesc *sequsedQDesc = faInfo.opParamInfo.sequsedQ.desc;
        OP_CHECK_IF(sequsedQDesc != nullptr && sequsedQDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(faInfo.opName, "seqused_q dtype must be INT32, but got %s",
                            DataTypeToSerialString(sequsedQDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

        uint32_t sequsedQDimNum = sequsedQTensor->GetShapeSize();
        OP_CHECK_IF(sequsedQDimNum != faInfo.bSize,
                    OP_LOGE(faInfo.opName, "The length of seqused_q(%u) should be equal to batch(%u).", sequsedQDimNum,
                            faInfo.bSize),
                    return ge::GRAPH_FAILED);

        // TODO: tensor值校验需用户自行保证
        // 文档要求：seqused_q仅支持非负整数、seqused_q中的值需小于等于Q_S
        // 由于tiling侧无法获取tensor的具体数值，此处不做值校验
    }

    auto &sequsedKvTensor = faInfo.opParamInfo.sequsedKv.tensor;
    if (sequsedKvTensor != nullptr) {
        const gert::CompileTimeTensorDesc *sequsedKvDesc = faInfo.opParamInfo.sequsedKv.desc;
        OP_CHECK_IF(sequsedKvDesc != nullptr && sequsedKvDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(faInfo.opName, "seqused_kv dtype must be INT32, but got %s",
                            DataTypeToSerialString(sequsedKvDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

        uint32_t sequsedKvDimNum = sequsedKvTensor->GetShapeSize();
        OP_CHECK_IF(sequsedKvDimNum != faInfo.bSize,
                    OP_LOGE(faInfo.opName, "The length of seqused_kv(%u) should be equal to batch(%u).",
                            sequsedKvDimNum, faInfo.bSize),
                    return ge::GRAPH_FAILED);

        // TODO: tensor值校验需用户自行保证
        // 文档要求：seqused_kv仅支持非负整数、seqused_kv中的值需小于等于KV_S
        // 由于tiling侧无法获取tensor的具体数值，此处不做值校验
    }

    auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (cuSeqlensQTensor != nullptr) {
        const gert::CompileTimeTensorDesc *cuSeqlensQDesc = faInfo.opParamInfo.cuSeqlensQ.desc;
        OP_CHECK_IF(cuSeqlensQDesc != nullptr && cuSeqlensQDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(faInfo.opName, "cu_seqlens_q dtype must be INT32, but got %s",
                            DataTypeToSerialString(cuSeqlensQDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

        uint32_t cuSeqlensQDimNum = cuSeqlensQTensor->GetShapeSize();
        OP_CHECK_IF(cuSeqlensQDimNum != faInfo.bSize + 1,
                    OP_LOGE(faInfo.opName, "The length of cu_seqlens_q(%u) should be equal to batch + 1(%u).",
                            cuSeqlensQDimNum, faInfo.bSize + 1),
                    return ge::GRAPH_FAILED);

        // TODO: tensor值校验需用户自行保证
        // 文档要求：cu_seqlens_q值仅支持非负整数、其值应非递减排列、第一个元素为0、最后一个元素等于Q_T
        // 由于tiling侧无法获取tensor的具体数值，此处不做值校验
    }

    auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
    if (cuSeqlensKvTensor != nullptr) {
        const gert::CompileTimeTensorDesc *cuSeqlensKvDesc = faInfo.opParamInfo.cuSeqlensKv.desc;
        OP_CHECK_IF(cuSeqlensKvDesc != nullptr && cuSeqlensKvDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(faInfo.opName, "cu_seqlens_kv dtype must be INT32, but got %s",
                            DataTypeToSerialString(cuSeqlensKvDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

        uint32_t cuSeqlensKvDimNum = cuSeqlensKvTensor->GetShapeSize();
        OP_CHECK_IF(cuSeqlensKvDimNum != faInfo.bSize + 1,
                    OP_LOGE(faInfo.opName, "The length of cu_seqlens_kv(%u) should be equal to batch + 1(%u).",
                            cuSeqlensKvDimNum, faInfo.bSize + 1),
                    return ge::GRAPH_FAILED);

        // TODO: tensor值校验需用户自行保证
        // 文档要求：cu_seqlens_kv值仅支持非负整数、其值应非递减排列、第一个元素为0、最后一个元素等于KV_T
        // 由于tiling侧无法获取tensor的具体数值，此处不做值校验
    }

    auto &maxSeqlenQ = faInfo.opParamInfo.maxSeqlenQ;
    int64_t maxSeqlenQVal = *maxSeqlenQ; // 参数有默认值-1，不会为nullptr

    if (faInfo.qLayout != FaLayout::TND) {
        OP_CHECK_IF(maxSeqlenQVal != -1,
                    OP_LOGE(faInfo.opName,
                            "max_seqlen_q(%ld) is not supported when layout_q is %s, only supported in TND layout. "
                            "In non-TND layout, max_seqlen_q must be -1 (default).",
                            maxSeqlenQVal, FaLayoutToString(faInfo.qLayout).c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            maxSeqlenQVal != -1 && maxSeqlenQVal < 0,
            OP_LOGE(faInfo.opName, "max_seqlen_q(%ld) only supports -1 or non-negative integer.", maxSeqlenQVal),
            return ge::GRAPH_FAILED);
    }

    // TODO: tensor值校验需用户自行保证
    // 文档要求：当传入值为-1时，参数大小对齐T轴大小；当传入值为非负整数时，参数为该非负整数大小
    // 由于tiling侧无法获取tensor的具体数值，此处不做值与T轴大小的对齐校验

    auto &maxSeqlenKv = faInfo.opParamInfo.maxSeqlenKV;
    int64_t maxSeqlenKvVal = *maxSeqlenKv; // 参数有默认值-1，不会为nullptr

    if (faInfo.kvLayout != FaLayout::TND) {
        OP_CHECK_IF(maxSeqlenKvVal != -1,
                    OP_LOGE(faInfo.opName,
                            "max_seqlen_kv(%ld) is not supported when layout_kv is %s, only supported in TND layout. "
                            "In non-TND layout, max_seqlen_kv must be -1 (default).",
                            maxSeqlenKvVal, FaLayoutToString(faInfo.kvLayout).c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            maxSeqlenKvVal != -1 && maxSeqlenKvVal < 0,
            OP_LOGE(faInfo.opName, "max_seqlen_kv(%ld) only supports -1 or non-negative integer.", maxSeqlenKvVal),
            return ge::GRAPH_FAILED);
    }

    // TODO: tensor值校验需用户自行保证
    // 文档要求：当传入值为-1时，参数大小对齐T轴大小；当传入值为非负整数时，参数为该非负整数大小
    // 由于tiling侧无法获取tensor的具体数值，此处不做值与T轴大小的对齐校验

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (faInfo.pageAttentionFlag) {
        auto &sequsedKvTensor = faInfo.opParamInfo.sequsedKv.tensor;
        OP_CHECK_IF(sequsedKvTensor == nullptr,
                    OP_LOGE(faInfo.opName, "seqused_kv must be provided when PagedAttention is enabled."),
                    return ge::GRAPH_FAILED);
    }

    auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (faInfo.qLayout == FaLayout::TND) {
        OP_CHECK_IF(cuSeqlensQTensor == nullptr,
                    OP_LOGE(faInfo.opName, "cu_seqlens_q must be provided when layout_q is TND."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(cuSeqlensQTensor != nullptr,
                    OP_LOGE(faInfo.opName,
                            "cu_seqlens_q should not be provided when layout_q is %s, only supported in TND layout.",
                            FaLayoutToString(faInfo.qLayout).c_str()),
                    return ge::GRAPH_FAILED);
    }

    auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
    if (faInfo.kvLayout == FaLayout::TND) {
        OP_CHECK_IF(cuSeqlensKvTensor == nullptr,
                    OP_LOGE(faInfo.opName, "cu_seqlens_kv must be provided when layout_kv is TND."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(cuSeqlensKvTensor != nullptr,
                    OP_LOGE(faInfo.opName,
                            "cu_seqlens_kv should not be provided when layout_kv is %s, only supported in TND layout.",
                            FaLayoutToString(faInfo.kvLayout).c_str()),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    (void)faInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (cuSeqlensQTensor != nullptr && faInfo.qLayout == FaLayout::TND) {
        // TODO: tensor值校验需用户自行保证
        // 文档要求：cu_seqlens_q第一个元素为0、最后一个元素等于Q_T、其值应非递减排列
        // 由于tiling侧无法获取tensor的具体数值，此处不做值校验
    }

    auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
    if (cuSeqlensKvTensor != nullptr && faInfo.kvLayout == FaLayout::TND) {
        // TODO: tensor值校验需用户自行保证
        // 文档要求：cu_seqlens_kv第一个元素为0、最后一个元素等于KV_T、其值应非递减排列
        // 由于tiling侧无法获取tensor的具体数值，此处不做值校验
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling