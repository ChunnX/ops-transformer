/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file seq_len_checker.h
 * \brief
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
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus ActualSeqLenChecker::CheckSingleParaSequsedQ(const FaTilingInfo &faInfo)
{
    auto &sequsedQTensor = faInfo.opParamInfo.sequsedQ.tensor;
    if (sequsedQTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *sequsedQDesc = faInfo.opParamInfo.sequsedQ.desc;
    OP_CHECK_IF(sequsedQDesc != nullptr && sequsedQDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(faInfo.opName, "seqused_q dtype must be INT32, but got %s",
                        DataTypeToSerialString(sequsedQDesc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(sequsedQDesc, SEQUSED_Q_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sequsedQDimNum = sequsedQTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(sequsedQDimNum != 1, OP_LOGE(faInfo.opName, "seqused_q dim num must be 1, but got %u.", sequsedQDimNum),
                return ge::GRAPH_FAILED);

    uint32_t sequsedQShapeSize = sequsedQTensor->GetShapeSize();
    OP_CHECK_IF(
        sequsedQShapeSize != faInfo.bSize,
        OP_LOGE(faInfo.opName, "seqused_q shape(%u) should be equal to batch(%ld).", sequsedQShapeSize, faInfo.bSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaSequsedKv(const FaTilingInfo &faInfo)
{
    auto &sequsedKvTensor = faInfo.opParamInfo.sequsedKv.tensor;
    if (sequsedKvTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *sequsedKvDesc = faInfo.opParamInfo.sequsedKv.desc;
    OP_CHECK_IF(sequsedKvDesc != nullptr && sequsedKvDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(faInfo.opName, "seqused_kv dtype must be INT32, but got %s",
                        DataTypeToSerialString(sequsedKvDesc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(sequsedKvDesc, SEQUSED_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sequsedKvDimNum = sequsedKvTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(sequsedKvDimNum != 1,
                OP_LOGE(faInfo.opName, "seqused_kv dim num must be 1, but got %u.", sequsedKvDimNum),
                return ge::GRAPH_FAILED);

    uint32_t sequsedKvShapeSize = sequsedKvTensor->GetShapeSize();
    OP_CHECK_IF(
        sequsedKvShapeSize != faInfo.bSize,
        OP_LOGE(faInfo.opName, "seqused_kv shape(%u) should be equal to batch(%ld).", sequsedKvShapeSize, faInfo.bSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaCuSeqlensQ(const FaTilingInfo &faInfo)
{
    auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (cuSeqlensQTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *cuSeqlensQDesc = faInfo.opParamInfo.cuSeqlensQ.desc;
    OP_CHECK_IF(cuSeqlensQDesc != nullptr && cuSeqlensQDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(faInfo.opName, "cu_seqlens_q dtype must be INT32, but got %s",
                        DataTypeToSerialString(cuSeqlensQDesc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(cuSeqlensQDesc, CU_SEQLENS_Q_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t cuSeqlensQDimNum = cuSeqlensQTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(cuSeqlensQDimNum != 1,
                OP_LOGE(faInfo.opName, "cu_seqlens_q dim num must be 1, but got %u.", cuSeqlensQDimNum),
                return ge::GRAPH_FAILED);

    uint32_t cuSeqlensQShapeSize = cuSeqlensQTensor->GetShapeSize();
    OP_CHECK_IF(cuSeqlensQShapeSize != faInfo.bSize + 1,
                OP_LOGE(faInfo.opName, "cu_seqlens_q shape(%u) should be equal to batch + 1(%ld).", cuSeqlensQShapeSize,
                        faInfo.bSize + 1),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaCuSeqlensKv(const FaTilingInfo &faInfo)
{
    auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
    if (cuSeqlensKvTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *cuSeqlensKvDesc = faInfo.opParamInfo.cuSeqlensKv.desc;
    OP_CHECK_IF(cuSeqlensKvDesc != nullptr && cuSeqlensKvDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(faInfo.opName, "cu_seqlens_kv dtype must be INT32, but got %s",
                        DataTypeToSerialString(cuSeqlensKvDesc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(cuSeqlensKvDesc, CU_SEQLENS_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t cuSeqlensKvDimNum = cuSeqlensKvTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(cuSeqlensKvDimNum != 1,
                OP_LOGE(faInfo.opName, "cu_seqlens_kv dim num must be 1, but got %u.", cuSeqlensKvDimNum),
                return ge::GRAPH_FAILED);

    uint32_t cuSeqlensKvShapeSize = cuSeqlensKvTensor->GetShapeSize();
    OP_CHECK_IF(cuSeqlensKvShapeSize != faInfo.bSize + 1,
                OP_LOGE(faInfo.opName, "cu_seqlens_kv shape(%u) should be equal to batch + 1(%ld).",
                        cuSeqlensKvShapeSize, faInfo.bSize + 1),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaMaxSeqlenQ(const FaTilingInfo &faInfo)
{
    // if (faInfo.qLayout != FaLayout::TND) {
    //     OP_CHECK_IF(faInfo.maxSeqQ != -1,
    //                 OP_LOGE(faInfo.opName,
    //                         "max_seqlen_q(%ld) is not supported when layout_q is %s, only supported in TND layout. "
    //                         "In non-TND layout, max_seqlen_q must be -1 (default).",
    //                         faInfo.maxSeqQ, LayoutToSerialString(faInfo.qLayout).c_str()),
    //                 return ge::GRAPH_FAILED);
    // } else {
    //     OP_CHECK_IF(
    //         faInfo.maxSeqQ != -1 && faInfo.maxSeqQ < 0,
    //         OP_LOGE(faInfo.opName, "max_seqlen_q(%ld) only supports -1 or non-negative integer.", faInfo.maxSeqQ),
    //         return ge::GRAPH_FAILED);
    // }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaMaxSeqlenKv(const FaTilingInfo &faInfo)
{
    // if (faInfo.kvLayout != FaLayout::TND) {
    //     OP_CHECK_IF(faInfo.maxSeqKv != -1,
    //                 OP_LOGE(faInfo.opName,
    //                         "max_seqlen_kv(%ld) is not supported when layout_kv is %s, only supported in TND layout. "
    //                         "In non-TND layout, max_seqlen_kv must be -1 (default).",
    //                         faInfo.maxSeqKv, LayoutToSerialString(faInfo.kvLayout).c_str()),
    //                 return ge::GRAPH_FAILED);
    // } else {
    //     OP_CHECK_IF(
    //         faInfo.maxSeqKv != -1 && faInfo.maxSeqKv < 0,
    //         OP_LOGE(faInfo.opName, "max_seqlen_kv(%ld) only supports -1 or non-negative integer.", faInfo.maxSeqKv),
    //         return ge::GRAPH_FAILED);
    // }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (CheckSingleParaSequsedQ(faInfo) != ge::GRAPH_SUCCESS || CheckSingleParaSequsedKv(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensQ(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensKv(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenQ(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenKv(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
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
                            LayoutToSerialString(faInfo.qLayout).c_str()),
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
                            LayoutToSerialString(faInfo.kvLayout).c_str()),
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
        // 文档要求：cu_seqlens_q第一个元素为0、最后一个元素等于Q_T、其值应非递减排列
        // 由于tiling侧无法获取tensor的具体数值，此处不做值校验
    }

    auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
    if (cuSeqlensKvTensor != nullptr && faInfo.kvLayout == FaLayout::TND) {
        // 文档要求：cu_seqlens_kv第一个元素为0、最后一个元素等于KV_T、其值应非递减排列
        // 由于tiling侧无法获取tensor的具体数值，此处不做值校验
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling