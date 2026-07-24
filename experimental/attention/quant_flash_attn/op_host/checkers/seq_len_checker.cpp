/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file seq_len_checker.cpp
 * \brief Checker for cu_seqlens_q/kv, seqused_q/kv, max_seqlen_q/kv (文档约束: SeqLengths参数组)
 */

#include <map>
#include <numeric>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../quant_flash_attn_tiling_info.h"
#include "seq_len_checker.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace Ops::Base;

// ============================================================================
// SinglePara
// 文档约束(单参数校验):
//   seqused_q/seqused_kv : tensor_type 支持 INT32；tensor_shape 为 (B,)；值仅支持非负整数
//                          seqused_q 值 ≤ Q_S, seqused_kv 值 ≤ KV_S
//   cu_seqlens_q/kv      : tensor_type 支持 INT32；tensor_shape 为 (B+1,)；值仅支持非负整数；
//                          非递减、首元素 0、尾元素等于 Q_T/KV_T
//   max_seqlen_q/kv      : data_type 支持 INT32；默认值为 -1
// 注: 值约束(非负、≤Q_S/KV_S、非递减、首0尾等于Q_T/KV_T)需读 tensor 数值,
//     tiling 阶段无法获取, 不校验(见文档声明 line 214)。
// ============================================================================

ge::graphStatus SeqLenChecker::CheckSingleParaSequsedQ(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // 文档约束: dtype=INT32, format=ND, shape=(B,)
    const gert::Tensor *tensor = qfaInfo.opParamInfo.sequsedQ.tensor;
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.sequsedQ.desc;
    if (tensor == nullptr || desc == nullptr) {
        return ge::GRAPH_SUCCESS; // 可选参数，存在性由 CheckParaExistence 负责
    }
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, SEQUSED_Q_NAME.c_str(),
                    DataTypeToSerialStr(desc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, SEQUSED_Q_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = tensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, SEQUSED_Q_NAME.c_str(),
                    (std::to_string(shape.GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != qfaInfo.bSize,
                OP_LOGE(qfaInfo.opName, "%s shape dim0(%ld) must be equal to B(%ld).",
                    SEQUSED_Q_NAME.c_str(), shape.GetDim(0), qfaInfo.bSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaSequsedKv(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // 文档约束: dtype=INT32, format=ND, shape=(B,)
    const gert::Tensor *tensor = qfaInfo.opParamInfo.sequsedKv.tensor;
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.sequsedKv.desc;
    if (tensor == nullptr || desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, SEQUSED_KV_NAME.c_str(),
                    DataTypeToSerialStr(desc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, SEQUSED_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = tensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, SEQUSED_KV_NAME.c_str(),
                    (std::to_string(shape.GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != qfaInfo.bSize,
                OP_LOGE(qfaInfo.opName, "%s shape dim0(%ld) must be equal to B(%ld).",
                    SEQUSED_KV_NAME.c_str(), shape.GetDim(0), qfaInfo.bSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaCuSeqlensQ(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // 文档约束: dtype=INT32, format=ND, shape=(B+1,)
    const gert::Tensor *tensor = qfaInfo.opParamInfo.cuSeqlensQ.tensor;
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.cuSeqlensQ.desc;
    if (tensor == nullptr || desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, CU_SEQLENS_Q_NAME.c_str(),
                    DataTypeToSerialStr(desc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, CU_SEQLENS_Q_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = tensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, CU_SEQLENS_Q_NAME.c_str(),
                    (std::to_string(shape.GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != qfaInfo.bSize + 1,
                OP_LOGE(qfaInfo.opName, "%s shape dim0(%ld) must be equal to B+1(%ld).",
                    CU_SEQLENS_Q_NAME.c_str(), shape.GetDim(0), qfaInfo.bSize + 1),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaCuSeqlensKv(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // 文档约束: dtype=INT32, format=ND, shape=(B+1,)
    const gert::Tensor *tensor = qfaInfo.opParamInfo.cuSeqlensKv.tensor;
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.cuSeqlensKv.desc;
    if (tensor == nullptr || desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, CU_SEQLENS_KV_NAME.c_str(),
                    DataTypeToSerialStr(desc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, CU_SEQLENS_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = tensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, CU_SEQLENS_KV_NAME.c_str(),
                    (std::to_string(shape.GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != qfaInfo.bSize + 1,
                OP_LOGE(qfaInfo.opName, "%s shape dim0(%ld) must be equal to B+1(%ld).",
                    CU_SEQLENS_KV_NAME.c_str(), shape.GetDim(0), qfaInfo.bSize + 1),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaMaxSeqlenQ(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // 文档约束: data_type 支持 INT32，默认值为 -1
    // 数值约束已移至特性交叉校验(CheckFeature): 非TND时与seqused_q至少传1个
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaMaxSeqlenKv(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // 文档约束: data_type 支持 INT32，默认值为 -1
    // 数值约束已移至特性交叉校验(CheckFeature): 非TND非PA时与seqused_kv至少传1个
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckSingleParaSequsedQ(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaSequsedKv(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensQ(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensKv(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenQ(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenKv(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence / Feature / MultiPara — 后续阶段补充
// ============================================================================

ge::graphStatus SeqLenChecker::CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // 文档约束(存在性校验列, rowspan=6): seqused_q/kv、cu_seqlens_q/kv、max_seqlen_q/kv 均为可选参数。
    // 各参数的存在性约束属于特性交叉校验(CheckFeature):
    //   - 当 layout_q/kv 为 TND 时, cu_seqlens_q/kv 必须传入; 不为 TND 时不支持传入
    //   - 当 layout_q/kv 不为 TND 时, seqused_q/kv 与 max_seqlen_q/kv 至少传入其中一个
    //   - PA 场景下 seqused_kv 必传(由 paged_attention_checker 负责)
    // 值约束(seqused[b] <= cu_seqlens[b+1]-cu_seqlens[b])属于值约束, tiling 阶段无法获取 tensor 数值, 不校验。
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — 非TND时 seqused 与 max_seqlen 至少传1组 (文档"特性交叉校验"列)
// 文档约束:
//   1. 当 layout_q 不为 TND 时, seqused_q 与 max_seqlen_q 至少传入其中一个
//   2. 当 layout_kv 不为 TND 且不为 PA 场景时, seqused_kv 与 max_seqlen_kv 至少传入其中一个
// 注: max_seqlen_q/kv 默认值为 -1, 表示未传入; seqused_q/kv 为 nullptr 表示未传入
//     PA 场景下 seqused_kv 必传由 paged_attention_checker 负责, 此处不重复
// ============================================================================

ge::graphStatus SeqLenChecker::CheckSequsedMaxSeqlenAtLeastOne(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // q 侧: layout_q 不为 TND 时, seqused_q 与 max_seqlen_q 至少传1个
    if (qfaInfo.layoutQ != FiaLayout::TND) {
        bool sequsedQExists = (qfaInfo.opParamInfo.sequsedQ.tensor != nullptr &&
                               qfaInfo.opParamInfo.sequsedQ.desc != nullptr);
        bool maxSeqQProvided = (qfaInfo.maxSeqLenQ >= 0);
        OP_CHECK_IF(!sequsedQExists && !maxSeqQProvided,
                    OP_LOGE(qfaInfo.opName,
                            "When layout_q is %s (not TND), at least one of seqused_q or max_seqlen_q "
                            "must be provided.",
                            LayoutToSerialStr(qfaInfo.layoutQ).c_str()),
                    return ge::GRAPH_FAILED);
    }

    // kv 侧: layout_kv 不为 TND 且不为 PA 场景时, seqused_kv 与 max_seqlen_kv 至少传1个
    bool isPaLayout = (qfaInfo.layoutKV == FiaLayout::BnBsH || qfaInfo.layoutKV == FiaLayout::BnNBsD ||
                       qfaInfo.layoutKV == FiaLayout::NZ);
    if (qfaInfo.layoutKV != FiaLayout::TND && !isPaLayout) {
        bool sequsedKvExists = (qfaInfo.opParamInfo.sequsedKv.tensor != nullptr &&
                                qfaInfo.opParamInfo.sequsedKv.desc != nullptr);
        bool maxSeqKvProvided = (qfaInfo.maxSeqLenKv >= 0);
        OP_CHECK_IF(!sequsedKvExists && !maxSeqKvProvided,
                    OP_LOGE(qfaInfo.opName,
                            "When layout_kv is %s (not TND and not PA), at least one of seqused_kv or "
                            "max_seqlen_kv must be provided.",
                            LayoutToSerialStr(qfaInfo.layoutKV).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckSequsedMaxSeqlenAtLeastOne(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // 文档约束(一致性校验列, rowspan=6): 全部标注为"无"
    //   - seqused_q / seqused_kv / cu_seqlens_q / cu_seqlens_kv / max_seqlen_q / max_seqlen_kv
    // 无一致性校验需要实现，使用基类默认实现(返回 GRAPH_SUCCESS)即可。
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
