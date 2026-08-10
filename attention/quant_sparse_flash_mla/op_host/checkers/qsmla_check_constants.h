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
 * \file qsmla_check_constants.h
 * \brief Additional shared constants for QSMLA checkers that are NOT already defined
 *        in qsmla_tiling_common.h. This file supplements the common header
 *        with: DTYPE_SUPPORT_MAP, FORMAT_SUPPORT_SET, mask mode enums, cmp_ratio
 *        limits, quant mode constant, layout string constants, and head num limits.
 *
 *        NOTE: String name constants (QUERY_NAME, ORI_KV_NAME, ...), DATATYPE_TO_STRING_MAP,
 *        QSMLA_LAYOUT_AXIS_MAP, QSMLA_LAYOUT_DIM_MAP, MAX_BLOCK_SIZE, METADATA_SIZE, etc.
 *        are already defined in qsmla_tiling_common.h and are NOT duplicated here.
 */

#ifndef QSMLA_CHECK_CONSTANTS_H
#define QSMLA_CHECK_CONSTANTS_H

#include <map>
#include <set>
#include <string>
#include <vector>
#include "graph/types.h"

namespace optiling {

// ---------- Descale tensor name constants (not in parent header) ----------
const std::string Q_DESCALE_NAME = "q_descale";
const std::string ORI_KV_DESCALE_NAME = "ori_kv_descale";
const std::string CMP_KV_DESCALE_NAME = "cmp_kv_descale";

// ---------- Mask mode enumerations ----------
constexpr uint32_t MASK_MODE_NO_MASK = 0;
constexpr uint32_t MASK_MODE_CAUSAL = 3;
constexpr uint32_t MASK_MODE_SLIDING_WINDOW = 4;
constexpr int64_t WIN_LEFT_RIGHT_UNLIMITED = -1;

// ---------- Quant mode ----------
constexpr int64_t QUANT_MODE_PER_TENSOR_HIFLOAT8 = 1;

// ---------- Topk value mode ----------
constexpr int64_t TOPK_VALUE_MODE_DEFAULT = 1;

// ---------- Cmp ratio limits ----------
constexpr uint32_t CMP_RATIO_MIN = 1;
constexpr uint32_t CMP_RATIO_MAX = 128;

// ---------- Head num limits ----------
constexpr uint32_t Q_HEAD_NUM_MIN = 1;
constexpr uint32_t Q_HEAD_NUM_MAX = 128;
constexpr uint32_t KV_HEAD_NUM_LIMIT = 1;
constexpr uint32_t HEAD_DIM_512 = 512;

// ---------- Batch size limit ----------
constexpr uint32_t B_LIMIT = 65536;

// ---------- Layout string constants ----------
const std::string LAYOUT_BSND = "BSND";
const std::string LAYOUT_TND = "TND";
const std::string LAYOUT_PA_BBND = "PA_BBND";

// ---------- Dtype support map per tensor name ----------
const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME, {ge::DT_HIFLOAT8}},           {ORI_KV_NAME, {ge::DT_HIFLOAT8}},
    {CMP_KV_NAME, {ge::DT_HIFLOAT8}},          {Q_DESCALE_NAME, {ge::DT_FLOAT}},
    {ORI_KV_DESCALE_NAME, {ge::DT_FLOAT}},     {CMP_KV_DESCALE_NAME, {ge::DT_FLOAT}},
    {ORI_SPARSE_INDICES_NAME, {ge::DT_INT32}}, {CMP_SPARSE_INDICES_NAME, {ge::DT_INT32}},
    {ORI_BLOCK_TABLE_NAME, {ge::DT_INT32}},    {CMP_BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_Q_NAME, {ge::DT_INT32}},       {CU_SEQLENS_ORI_KV_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_CMP_KV_NAME, {ge::DT_INT32}},  {SEQUSED_Q_NAME, {ge::DT_INT32}},
    {SEQUSED_ORI_KV_NAME, {ge::DT_INT32}},     {SEQUSED_CMP_KV_NAME, {ge::DT_INT32}},
    {CMP_RESIDUAL_KV_NAME, {ge::DT_INT32}},    {ORI_TOPK_LENGTH_NAME, {ge::DT_INT32}},
    {CMP_TOPK_LENGTH_NAME, {ge::DT_INT32}},    {SINKS_NAME, {ge::DT_FLOAT}},
    {METADATA_NAME, {ge::DT_INT32}},           {ATTEN_OUT_NAME, {ge::DT_BF16}},
    {SOFTMAX_LSE_NAME, {ge::DT_FLOAT}},
};

// ---------- Format support set ----------
const std::set<ge::Format> FORMAT_SUPPORT_SET = {ge::FORMAT_ND};

} // namespace optiling

#endif // QSMLA_CHECK_CONSTANTS_H
