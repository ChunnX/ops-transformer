/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_all_gather_matmul_v3_tiling.cpp
 * \brief host侧tiling ut
 */
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace AllGatherMatmulV3UT {

const std::string OP_NAME = "AllGatherMatmulV3";

constexpr int64_t MX_GROUP_SIZE_VAL = static_cast<int64_t>((1ULL << 32) | (1ULL << 16) | 32);

struct AllGatherMatmulV3TestParam {
    std::string caseName;
    std::initializer_list<int64_t> x1Shape;
    ge::DataType x1Dtype;
    ge::Format x1Format;
    std::initializer_list<int64_t> x2Shape;
    ge::DataType x2Dtype;
    ge::Format x2Format;
    std::initializer_list<int64_t> biasShape;
    ge::DataType biasDtype;
    ge::Format biasFormat;
    std::initializer_list<int64_t> x1ScaleShape;
    ge::DataType x1ScaleDtype;
    ge::Format x1ScaleFormat;
    std::initializer_list<int64_t> x2ScaleShape;
    ge::DataType x2ScaleDtype;
    ge::Format x2ScaleFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::string groupAttr;
    int64_t hcclBufferSizeAttr;
    bool isTransAAttr;
    bool isTransBAttr;
    int64_t rankSizeAttr;
    int64_t groupSizeAttr;
    int64_t yDtypeAttr;
    std::string commModeAttr;
    uint64_t rankNum;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;
};

inline std::ostream &operator<<(std::ostream &os, const AllGatherMatmulV3TestParam &param)
{
    return os << param.caseName;
}

static AllGatherMatmulV3TestParam g_testCases[] = {
    {"all_gather_matmul_v3_critical_case_e4m3_bf16_rank4",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e4m3_fp16_rank4",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_FLOAT16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_FLOAT16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e5m2_bf16_rank4",
     {2048, 4096}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e5m2_fp16_rank4",
     {2048, 4096}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_FLOAT16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_FLOAT16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e4m3_e5m2_bf16_rank4",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e5m2_e4m3_fp16_rank4",
     {2048, 4096}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_FLOAT16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_FLOAT16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_fp4_bf16_rank4",
     {2048, 4096}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_fp4_fp16_rank4",
     {2048, 4096}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_FLOAT16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_FLOAT16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e4m3_bf16_rank2",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {4096, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 2, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     2, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e4m3_bf16_rank8",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {16384, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 8, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     8, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e4m3_bf16_rank16",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {32768, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 16, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     16, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e4m3_fp16_rank4_m2061",
     {2061, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2061, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8244, 2560}, ge::DT_FLOAT16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_FLOAT16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e4m3_bf16_rank4_n6144",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {6144, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {6144}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {6144, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 6144}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e4m3_fp16_rank4_m8192",
     {8192, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {8192, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {32768, 2560}, ge::DT_FLOAT16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_FLOAT16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_critical_case_e4m3_bf16_rank4_m3072",
     {3072, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {3072, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {12288, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, MC2_TILING_DATA_RESERVED_LEN},
    {"all_gather_matmul_v3_abuse_case_invalid_x1_dtype",
     {2048, 4096}, ge::DT_INT8, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_invalid_x2_dtype",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_fp4_mismatch_x1",
     {2048, 4096}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_fp4_mismatch_x2",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_invalid_y_dtype",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_INT8, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_INT8), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_invalid_x1_scale_dtype",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_invalid_x2_scale_dtype",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_invalid_bias_dtype",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_x1_1d",
     {4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_x2_3d",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2, 2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_y_3d",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2, 4096, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_k_mismatch",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4032}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 63, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_k_not_divisible_64",
     {2048, 4033}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4033}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 63, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 63, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_k_exceeds_max",
     {2048, 65536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 65536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 1024, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 1024, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_invalid_rank_size",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {6144, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 3, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     3, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_is_trans_b_false",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, false, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_comm_mode_ai_cpu",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "ai_cpu",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_invalid_group_size",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, 0, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_x1_scale_wrong_shape",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {1024, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_x2_scale_wrong_shape",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {1280, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_y_shape_mismatch",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {4096, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_x1_format_nz",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_x1_scale_2d",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_bias_wrong_shape",
     {2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {1280}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_m_zero",
     {0, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {0, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {0, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"all_gather_matmul_v3_abuse_case_k_zero",
     {2048, 0}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560, 0}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
     {2560}, ge::DT_FLOAT, ge::FORMAT_ND,
     {2048, 0, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {2560, 0, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
     {8192, 2560}, ge::DT_BF16, ge::FORMAT_ND,
     "group", 200, false, true, 4, MX_GROUP_SIZE_VAL, static_cast<int64_t>(ge::DT_BF16), "urma",
     4, "3510",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
};

class AllGatherMatmulV3Arch35TilingTest : public testing::TestWithParam<AllGatherMatmulV3TestParam> {
protected:
    static void SetUpTestCase()
    {
        setenv("HCCL_BUFFSIZE", "300", 1);
        std::cout << "AllGatherMatmulV3Arch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        unsetenv("HCCL_BUFFSIZE");
        std::cout << "AllGatherMatmulV3Arch35TilingTest TearDown." << std::endl;
    }
};

static struct AllGatherMatmulV3CompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const AllGatherMatmulV3TestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    gert::StorageShape contextShape = {{100}, {100}};
    gert::StorageShape x1Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape x2Shape = {param.x2Shape, param.x2Shape};
    gert::StorageShape biasShape = {param.biasShape, param.biasShape};
    gert::StorageShape x1ScaleShape = {param.x1ScaleShape, param.x1ScaleShape};
    gert::StorageShape x2ScaleShape = {param.x2ScaleShape, param.x2ScaleShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    int64_t mPerRank = (param.x1Shape.size() >= 1) ? *(param.x1Shape.begin()) : 0;
    int64_t k = (param.x1Shape.size() >= 2) ? *(param.x1Shape.begin() + 1) : 0;
    std::initializer_list<int64_t> gatherOutShape = {static_cast<int64_t>(param.rankSizeAttr) * mPerRank, k};
    gert::StorageShape gatherOutStorageShape = {gatherOutShape, gatherOutShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {x1Shape, param.x1Dtype, param.x1Format},
         {x2Shape, param.x2Dtype, param.x2Format},
         {biasShape, param.biasDtype, param.biasFormat},
         {x1ScaleShape, param.x1ScaleDtype, param.x1ScaleFormat},
         {x2ScaleShape, param.x2ScaleDtype, param.x2ScaleFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat},
         {gatherOutStorageShape, param.x1Dtype, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.groupAttr)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hcclBufferSizeAttr)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.isTransAAttr)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.isTransBAttr)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rankSizeAttr)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSizeAttr)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.yDtypeAttr)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commModeAttr)}});
    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                   param.socVersion, 64, 262144, 8192);
}

static void ThreadFunction(const AllGatherMatmulV3TestParam *testCases, size_t caseNum, size_t threadIdx,
                           size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        auto param = testCases[idx];
        auto tilingContextPara = BuildTilingContextPara(param);
        ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                        param.expectWorkspaces, param.mc2TilingDataReservedLen);
    }
}

static void TestExecMultiThread(const AllGatherMatmulV3TestParam *testCases, size_t testCaseNum, size_t threadNum)
{
    if (threadNum == 0 || testCases == nullptr || testCaseNum == 0) {
        return;
    }
    std::vector<std::thread> threads(threadNum);
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
    }
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(AllGatherMatmulV3Arch35TilingTest, GeneralCasesTest)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey,
                       param.expectTilingData, param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

TEST_F(AllGatherMatmulV3Arch35TilingTest, GeneralCasesMultiThreadTest)
{
    GTEST_SKIP() << "Skipped due to incompatible mock environment for non-default rankNum cases";
}

INSTANTIATE_TEST_CASE_P(AllGatherMatmulV3TilingUT, AllGatherMatmulV3Arch35TilingTest,
                        testing::ValuesIn(g_testCases));

} // namespace AllGatherMatmulV3UT
