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
#include "actual_seq_len_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus ActualSeqLenChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    auto &sequsedQTensor = faInfo.opParamInfo.sequsedQ.tensor;
    if (sequsedQTensor != nullptr) {
        uint32_t sequsedQDimNum = sequsedQTensor->GetShapeSize();
        OP_CHECK_IF(sequsedQDimNum != faInfo.bSize,
            OP_LOGE(faInfo.opName, "The length of seqused_q(%u) should be equal to batch(%u).",
                    sequsedQDimNum, faInfo.bSize),
            return ge::GRAPH_FAILED);

        for (uint32_t i = 0; i < sequsedQDimNum; i++) {
            int64_t seqLength = sequsedQTensor->GetData<int64_t>()[i];
            OP_CHECK_IF(seqLength < 0,
                OP_LOGE(faInfo.opName, "The value of seqused_q[%u](%ld) should be non-negative integer.",
                    i, seqLength),
                return ge::GRAPH_FAILED);
        }
    }

    auto &sequsedKvTensor = faInfo.opParamInfo.sequsedKv.tensor;
    if (sequsedKvTensor != nullptr) {
        uint32_t sequsedKvDimNum = sequsedKvTensor->GetShapeSize();
        OP_CHECK_IF(sequsedKvDimNum != faInfo.bSize,
            OP_LOGE(faInfo.opName, "The length of seqused_kv(%u) should be equal to batch(%u).",
                    sequsedKvDimNum, faInfo.bSize),
            return ge::GRAPH_FAILED);

        for (uint32_t i = 0; i < sequsedKvDimNum; i++) {
            int64_t seqLength = sequsedKvTensor->GetData<int64_t>()[i];
            OP_CHECK_IF(seqLength < 0,
                OP_LOGE(faInfo.opName, "The value of seqused_kv[%u](%ld) should be non-negative integer.",
                    i, seqLength),
                return ge::GRAPH_FAILED);
        }
    }

    auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (cuSeqlensQTensor != nullptr) {
        uint32_t cuSeqlensQDimNum = cuSeqlensQTensor->GetShapeSize();
        OP_CHECK_IF(cuSeqlensQDimNum != faInfo.bSize + 1,
            OP_LOGE(faInfo.opName, "The length of cu_seqlens_q(%u) should be equal to batch + 1(%u).",
                    cuSeqlensQDimNum, faInfo.bSize + 1),
            return ge::GRAPH_FAILED);

        for (uint32_t i = 0; i < cuSeqlensQDimNum; i++) {
            int64_t seqLength = cuSeqlensQTensor->GetData<int64_t>()[i];
            OP_CHECK_IF(seqLength < 0,
                OP_LOGE(faInfo.opName, "The value of cu_seqlens_q[%u](%ld) should be non-negative integer.",
                    i, seqLength),
                return ge::GRAPH_FAILED);
        }
    }

    auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
    if (cuSeqlensKvTensor != nullptr) {
        uint32_t cuSeqlensKvDimNum = cuSeqlensKvTensor->GetShapeSize();
        OP_CHECK_IF(cuSeqlensKvDimNum != faInfo.bSize + 1,
            OP_LOGE(faInfo.opName, "The length of cu_seqlens_kv(%u) should be equal to batch + 1(%u).",
                    cuSeqlensKvDimNum, faInfo.bSize + 1),
            return ge::GRAPH_FAILED);

        for (uint32_t i = 0; i < cuSeqlensKvDimNum; i++) {
            int64_t seqLength = cuSeqlensKvTensor->GetData<int64_t>()[i];
            OP_CHECK_IF(seqLength < 0,
                OP_LOGE(faInfo.opName, "The value of cu_seqlens_kv[%u](%ld) should be non-negative integer.",
                    i, seqLength),
                return ge::GRAPH_FAILED);
        }
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

    if (faInfo.qLayout == FaLayout::TND) {
        auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
        OP_CHECK_IF(cuSeqlensQTensor == nullptr,
            OP_LOGE(faInfo.opName, "cu_seqlens_q must be provided when layout_q is TND."),
            return ge::GRAPH_FAILED);
    }

    if (faInfo.kvLayout == FaLayout::TND) {
        auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
        OP_CHECK_IF(cuSeqlensKvTensor == nullptr,
            OP_LOGE(faInfo.opName, "cu_seqlens_kv must be provided when layout_kv is TND."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (cuSeqlensQTensor != nullptr && faInfo.qLayout == FaLayout::TND) {
        uint32_t cuSeqlensQDimNum = cuSeqlensQTensor->GetShapeSize();
        OP_CHECK_IF(cuSeqlensQTensor->GetData<int64_t>()[0] != 0,
            OP_LOGE(faInfo.opName, "The first element of cu_seqlens_q should be 0."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(cuSeqlensQTensor->GetData<int64_t>()[cuSeqlensQDimNum - 1] != faInfo.qTSize,
            OP_LOGE(faInfo.opName, "The last element of cu_seqlens_q(%ld) should be equal to T_Q(%u).",
                    cuSeqlensQTensor->GetData<int64_t>()[cuSeqlensQDimNum - 1], faInfo.qTSize),
            return ge::GRAPH_FAILED);

        for (uint32_t i = 1; i < cuSeqlensQDimNum; i++) {
            int64_t curSeqLength = cuSeqlensQTensor->GetData<int64_t>()[i];
            int64_t lastSeqLength = cuSeqlensQTensor->GetData<int64_t>()[i - 1];
            OP_CHECK_IF(curSeqLength < lastSeqLength,
                OP_LOGE(faInfo.opName, "cu_seqlens_q should be increasing, but element[%u](%ld) < element[%u](%ld).",
                    i, curSeqLength, i - 1, lastSeqLength),
                return ge::GRAPH_FAILED);
        }
    }

    auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
    if (cuSeqlensKvTensor != nullptr && faInfo.kvLayout == FaLayout::TND) {
        uint32_t cuSeqlensKvDimNum = cuSeqlensKvTensor->GetShapeSize();
        OP_CHECK_IF(cuSeqlensKvTensor->GetData<int64_t>()[0] != 0,
            OP_LOGE(faInfo.opName, "The first element of cu_seqlens_kv should be 0."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(cuSeqlensKvTensor->GetData<int64_t>()[cuSeqlensKvDimNum - 1] != faInfo.kTSize,
            OP_LOGE(faInfo.opName, "The last element of cu_seqlens_kv(%ld) should be equal to T_KV(%u).",
                    cuSeqlensKvTensor->GetData<int64_t>()[cuSeqlensKvDimNum - 1], faInfo.kTSize),
            return ge::GRAPH_FAILED);

        for (uint32_t i = 1; i < cuSeqlensKvDimNum; i++) {
            int64_t curSeqLength = cuSeqlensKvTensor->GetData<int64_t>()[i];
            int64_t lastSeqLength = cuSeqlensKvTensor->GetData<int64_t>()[i - 1];
            OP_CHECK_IF(curSeqLength < lastSeqLength,
                OP_LOGE(faInfo.opName, "cu_seqlens_kv should be increasing, but element[%u](%ld) < element[%u](%ld).",
                    i, curSeqLength, i - 1, lastSeqLength),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling