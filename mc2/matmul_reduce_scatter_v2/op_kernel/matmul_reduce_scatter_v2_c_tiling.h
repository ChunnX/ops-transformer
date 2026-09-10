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
 * \file matmul_reduce_scatter_v2_c_tiling.h
 * \brief
 */
#ifndef _MATMUL_REDUCE_SCATTER_V2_C_TILING_H_
#define _MATMUL_REDUCE_SCATTER_V2_C_TILING_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/op_kernel/mc2_tiling_struct.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmv3_arch35_tiling_data.h"
#include "../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"

namespace Mc2Tiling {

constexpr uint32_t MAX_EXPERT_SIZE = 256U; // 最大通信域专家的数量
constexpr uint32_t MAX_EP_RANK_SIZE = 64U; // 最大通信域内卡的数量

// DFX 头部约定: dumpInfo 必须是 tiling 结构的第一个成员(offset 0)，
// 异常 dump 回调直接从 tiling blob 起始位置解析 DfxDumpInfo
struct MatmulReduceScatterV2TilingData {
#if MC2_DFX_ENABLE
    Utils::DfxDumpInfo dumpInfo; // DFX dump 元数据(workspace/peermem 布局)
#endif
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    RCSTiling param;
    uint32_t dataType;
    uint32_t debugMode;
    Mc2MatMulV3TilingData mC2Mmv3TileTilingData;
    Mc2MatMulV3TilingData mC2Mmv3TailTilingData;
};

struct QuantBatchMatmulV3ReduceScatterTilingData {
#if MC2_DFX_ENABLE
    Utils::DfxDumpInfo dumpInfo; // DFX dump 元数据(workspace/peermem 布局)
#endif
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    RCSTiling param;
    uint32_t dataType;
    uint32_t debugMode;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams quantBmmV3TileTiling;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams quantBmmV3TailTiling;
};
} // namespace Mc2Tiling
#endif
