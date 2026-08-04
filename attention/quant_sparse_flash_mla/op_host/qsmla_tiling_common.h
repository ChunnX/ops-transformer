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
 * \file qsmla_tiling_common.h
 * \brief QSMLA shared constants, enums, maps, and index definitions.
 */

#ifndef QSMLA_TILING_COMMON_H
#define QSMLA_TILING_COMMON_H

#include <map>
#include <string>
#include <vector>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"

namespace optiling {

// ---------- Tensor name constants ----------
const std::string ORI_BLOCK_TABLE_NAME = "ori_block_table";
const std::string CMP_BLOCK_TABLE_NAME = "cmp_block_table";
const std::string SINKS_NAME = "sinks";
const std::string QUERY_NAME = "query";
const std::string ORI_KV_NAME = "ori_kv";
const std::string CMP_KV_NAME = "cmp_kv";
const std::string ORI_SPARSE_INDICES_NAME = "ori_sparse_indices";
const std::string CMP_SPARSE_INDICES_NAME = "cmp_sparse_indices";
const std::string ATTEN_OUT_NAME = "attention_out";
const std::string SOFTMAX_LSE_NAME = "softmax_lse";
const std::string CU_SEQLENS_Q_NAME = "cu_seqlens_q";
const std::string CU_SEQLENS_ORI_KV_NAME = "cu_seqlens_ori_kv";
const std::string CU_SEQLENS_CMP_KV_NAME = "cu_seqlens_cmp_kv";
const std::string SEQUSED_Q_NAME = "seqused_q";
const std::string SEQUSED_ORI_KV_NAME = "seqused_ori_kv";
const std::string SEQUSED_CMP_KV_NAME = "seqused_cmp_kv";
const std::string CMP_RESIDUAL_KV_NAME = "cmp_residual_kv";
const std::string ORI_TOPK_LENGTH_NAME = "ori_topk_length";
const std::string CMP_TOPK_LENGTH_NAME = "cmp_topk_length";
const std::string METADATA_NAME = "metadata";

// ---------- Parameter info structs ----------
struct QSMLATilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct QSMLATilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

// ---------- Layout enum ----------
enum class QSMLALayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

// ---------- Axis enum ----------
enum class QSMLAAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3,
    T = 5,
    Bn = 6,
    Bs = 7
};

// ---------- Template mode enum ----------
enum class QSMLATemplateMode : uint32_t {
    SWA_TEMPLATE_MODE = 0,
    HCA_TEMPLATE_MODE = 1,
    CSA_TEMPLATE_MODE = 2,
    ORI_SPARSE_TEMPLATE_MODE = 3,
    ORI_CMP_SPARSE_TEMPLATE_MODE = 4
};

// ---------- Inputs Index ----------
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t ORI_KV_INDEX = 1;
constexpr uint32_t CMP_KV_INDEX = 2;
constexpr uint32_t Q_DESCALE = 3;
constexpr uint32_t ORI_KV_DESCALE = 4;
constexpr uint32_t CMP_KV_DESCALE = 5;
constexpr uint32_t ORI_SPARSE_INDICES_INDEX = 6;
constexpr uint32_t CMP_SPARSE_INDICES_INDEX = 7;
constexpr uint32_t ORI_BLOCK_TABLE_INDEX = 8;
constexpr uint32_t CMP_BLOCK_TABLE_INDEX = 9;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 10;
constexpr uint32_t CU_SEQLENS_ORI_KV_INDEX = 11;
constexpr uint32_t CU_SEQLENS_CMP_KV_INDEX = 12;
constexpr uint32_t SEQUSED_Q_INDEX = 13;
constexpr uint32_t SEQUSED_ORI_KV_INDEX = 14;
constexpr uint32_t SEQUSED_CMP_KV_INDEX = 15;
constexpr uint32_t CMP_RESIDUAL_KV_INDEX = 16;
constexpr uint32_t ORI_TOPK_LENGTH_INDEX = 17;
constexpr uint32_t CMP_TOPK_LENGTH_INDEX = 18;
constexpr uint32_t SINKS_INDEX = 19;
constexpr uint32_t METADATA_INDEX = 20;

// ---------- Outputs Index ----------
constexpr uint32_t ATTN_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

// ---------- Attributes Index ----------
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 0;
constexpr uint32_t ATTR_SOFTMAX_SCALE_INDEX = 1;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 2;
constexpr uint32_t ATTR_ORI_MASK_MODE_INDEX = 3;
constexpr uint32_t ATTR_CMP_MASK_MODE_INDEX = 4;
constexpr uint32_t ATTR_ORI_WIN_LEFT_INDEX = 5;
constexpr uint32_t ATTR_ORI_WIN_RIGHT_INDEX = 6;
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 7;
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 8;
constexpr uint32_t ATTR_TOPK_VALUE_MODE_INDEX = 9;
constexpr uint32_t ATTR_RETURN_SOFTMAX_LSE_INDEX = 10;

// ---------- Dim Index ----------
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_IDX_FOUR = 4;
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;

// ---------- Dim Num ----------
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

// ---------- Limits ----------
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t METADATA_SIZE = 1024;

// ---------- DataType to string map ----------
const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},
    {ge::DT_FLOAT, "DT_FLOAT"},
    {ge::DT_FLOAT16, "DT_FLOAT16"},
    {ge::DT_INT8, "DT_INT8"},
    {ge::DT_INT16, "DT_INT16"},
    {ge::DT_UINT16, "DT_UINT16"},
    {ge::DT_UINT8, "DT_UINT8"},
    {ge::DT_INT32, "DT_INT32"},
    {ge::DT_INT64, "DT_INT64"},
    {ge::DT_UINT32, "DT_UINT32"},
    {ge::DT_UINT64, "DT_UINT64"},
    {ge::DT_BOOL, "DT_BOOL"},
    {ge::DT_DOUBLE, "DT_DOUBLE"},
    {ge::DT_DUAL, "DT_DUAL"},
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},
    {ge::DT_QINT8, "DT_QINT8"},
    {ge::DT_QINT16, "DT_QINT16"},
    {ge::DT_QINT32, "DT_QINT32"},
    {ge::DT_QUINT8, "DT_QUINT8"},
    {ge::DT_QUINT16, "DT_QUINT16"},
    {ge::DT_RESOURCE, "DT_RESOURCE"},
    {ge::DT_STRING_REF, "DT_STRING_REF"},
    {ge::DT_STRING, "DT_STRING"},
    {ge::DT_VARIANT, "DT_VARIANT"},
    {ge::DT_BF16, "DT_BFLOAT16"},
    {ge::DT_INT4, "DT_INT4"},
    {ge::DT_UINT1, "DT_UINT1"},
    {ge::DT_INT2, "DT_INT2"},
    {ge::DT_UINT2, "DT_UINT2"},
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
};

// ---------- Layout-axis map ----------
const std::map<QSMLALayout, std::vector<QSMLAAxis>> QSMLA_LAYOUT_AXIS_MAP = {
    {QSMLALayout::BSND, {QSMLAAxis::B, QSMLAAxis::S, QSMLAAxis::N, QSMLAAxis::D}},
    {QSMLALayout::TND, {QSMLAAxis::T, QSMLAAxis::N, QSMLAAxis::D}},
    {QSMLALayout::PA_BBND, {QSMLAAxis::Bn, QSMLAAxis::Bs, QSMLAAxis::N, QSMLAAxis::D}},
};

// ---------- Layout-dim map ----------
const std::map<QSMLALayout, size_t> QSMLA_LAYOUT_DIM_MAP = {
    {QSMLALayout::BSND, DIM_NUM_FOUR},
    {QSMLALayout::TND, DIM_NUM_THREE},
    {QSMLALayout::PA_BBND, DIM_NUM_FOUR},
};

} // namespace optiling
#endif // QSMLA_TILING_COMMON_H
