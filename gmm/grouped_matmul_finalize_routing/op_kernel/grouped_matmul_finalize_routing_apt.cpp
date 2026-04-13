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
 * \file grouped_matmul_finalize_routing_apt.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "lib/matmul_intf.h"
#include "arch35/grouped_matmul_finalize_routing_tiling_key.h"
#if ORIG_DTYPE_X != DT_FLOAT8_E4M3FN && ORIG_DTYPE_X != DT_FLOAT8_E5M2 && ORIG_DTYPE_X != DT_FLOAT4_E1M2 && ORIG_DTYPE_X != DT_FLOAT4_E2M1
    #include "arch35/grouped_matmul_finalize_routing_pertoken_dequant.h"
#elif ORIG_DTYPE_X == DT_FLOAT4_E1M2 || ORIG_DTYPE_X == DT_FLOAT4_E2M1
    #include "arch35/grouped_matmul_finalize_routing.h"
#else
    #include "arch35/grouped_matmul_finalize_routing.h"
    #include "arch35/grouped_matmul_finalize_routing_pertoken_dequant.h"
#endif

static constexpr uint64_t BF16TYPE = 2;

// SCALETYPE 0 is float8e8m0,1 is fp32, 2 is bf16; ROWINDEXTYPE 0 is int64, 1 is int32.
template <int ATRANS, int BTRANS, int SCALETYPE, int ROWINDEXTYPE>
__global__ __aicore__ void
grouped_matmul_finalize_routing(GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias, GM_ADDR pertoken_scale,
                                GM_ADDR group_list, GM_ADDR share_input, GM_ADDR logit, GM_ADDR row_index,
                                GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    #if ORIG_DTYPE_X != DT_FLOAT8_E4M3FN && ORIG_DTYPE_X != DT_FLOAT8_E5M2 && ORIG_DTYPE_X != DT_FLOAT4_E1M2 && ORIG_DTYPE_X != DT_FLOAT4_E2M1
        if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == 1 && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, 1, 0>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 1 && SCALETYPE == 1 && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, 1, 0>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == BF16TYPE && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, BF16TYPE, 0>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 1 && SCALETYPE == BF16TYPE && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, BF16TYPE, 0>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == 1 && ROWINDEXTYPE == 1) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, 1, 1>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 1 && SCALETYPE == 1 && ROWINDEXTYPE == 1) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, 1, 1>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == BF16TYPE && ROWINDEXTYPE == 1) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, BF16TYPE, 1>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 1 && SCALETYPE == BF16TYPE && ROWINDEXTYPE == 1) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, BF16TYPE, 1>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        }
    #elif ORIG_DTYPE_X == DT_FLOAT4_E1M2 || ORIG_DTYPE_X == DT_FLOAT4_E2M1
        if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == 0 && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_mx<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 1 && SCALETYPE == 0 && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_mx<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        }
    #else
        if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == 0 && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_mx<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 1 && SCALETYPE == 0 && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_mx<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == 1 && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, 1, 0>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 1 && SCALETYPE == 1 && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, 1, 0>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == BF16TYPE && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, BF16TYPE, 0>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 1 && SCALETYPE == BF16TYPE && ROWINDEXTYPE == 0) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, BF16TYPE, 0>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == 1 && ROWINDEXTYPE == 1) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, 1, 1>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 1 && SCALETYPE == 1 && ROWINDEXTYPE == 1) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, 1, 1>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 0 && SCALETYPE == BF16TYPE && ROWINDEXTYPE == 1) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, BF16TYPE, 1>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        } else if constexpr (ATRANS == 0 && BTRANS == 1 && SCALETYPE == BF16TYPE && ROWINDEXTYPE == 1) {
            grouped_matmul_finalize_routing_pertoken_dequant<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, BF16TYPE, 1>(
                x, w, scale, bias, pertoken_scale, group_list, share_input, logit, row_index, offset, y, workspaceGM,
                tilingGM);
        }
    #endif
}
#endif