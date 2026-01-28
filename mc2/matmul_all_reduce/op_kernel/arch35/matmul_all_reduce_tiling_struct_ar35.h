/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_all_reduce_tiling_struct
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_STRUCT_ARCH35_H
#define MATMUL_ALL_REDUCE_TILING_STRUCT_ARCH35_H

#include "kernel_tiling/kernel_tiling.h"
#include "../../common/inc/kernel/mc2_tiling_struct.h"
#include "../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#include "../../3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#include "../arch32/unquant_matmul_all_reduce_tiling_data.h"
#include "../arch32/quant_matmul_all_reduce_tiling_data.h"
#include "../arch32/weight_quant_matmul_all_reduce_tiling_data.h"
namespace Mc2Tiling {

struct WeightQuantMatmulAllReduceA5TilingData {
    uint32_t version;
    uint32_t hcommCnt;
    Mc2Tiling::MC2ServerCfg serverCfg;
    Mc2Tiling::MC2HcommCfg hcommCfg;
    Mc2Tiling::Mc2Msg msg;
    Mc2Tiling::RCSTiling param;
    Mc2WeightQuantBatchMatmulV2RegBaseTilingData tileRegBaseMmTiling;
    Mc2WeightQuantBatchMatmulV2RegBaseTilingData tailRegBaseMmTiling;
};

struct WeightQuantMatmulAllReduceA5Fp8TilingData {
    uint32_t version;
    uint32_t hcommCnt;
    Mc2Tiling::MC2ServerCfg serverCfg;
    Mc2Tiling::MC2HcommCfg hcommCfg;
    Mc2Tiling::Mc2Msg msg;
    Mc2Tiling::RCSTiling param;
    Mc2WeightQuantBatchMatmulV2ASTilingData tileMmASTiling;
    Mc2WeightQuantBatchMatmulV2ASTilingData tailMmASTiling;
};

struct QuantMatmulAllReduceTilingDataA5{
    uint32_t version;
    uint32_t hcommCnt;
    Mc2Tiling::MC2ServerCfg serverCfg;
    Mc2Tiling::MC2HcommCfg hcommCfg;
    Mc2Tiling::MC2HcommCfg hcommInt8Cfg;
    Mc2Tiling::Mc2Msg msg;
    Mc2Tiling::RCSTiling param;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams tilematmulTiling;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams tailmatmulTiling;
};

struct MatmulAllReduce910TilingDataA5{
    uint32_t version;
    uint32_t hcommCnt;
    Mc2Tiling::MC2ServerCfg serverCfg;
    Mc2Tiling::MC2HcommCfg hcommCfg;
    Mc2Tiling::Mc2Msg msg;
    Mc2Tiling::RCSTiling param;
    Mc2MatMulV3TilingData mC2Mmv3TileTilingData;
    Mc2MatMulV3TilingData mC2Mmv3TailTilingData;
};

}
#endif // MATMUL_ALL_REDUCE_TILING_STRUCT_ARCH35_H