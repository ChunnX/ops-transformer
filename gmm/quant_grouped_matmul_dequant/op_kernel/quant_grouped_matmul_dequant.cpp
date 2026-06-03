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
 * \file quant_grouped_matmul_dequant.cpp
 * \brief
 */

#include "quant_matmul_dequant_grouped.h"

namespace AscendC {
struct OptQuantMatmulDequantTilingData {
    uint32_t coreNum;
    uint32_t MCoreNum;
    uint32_t NCoreNum;
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t E;
    uint32_t originKAligned32;
    uint32_t originKAligned512;
    uint32_t fracM;
    uint32_t fracK;
    uint32_t fracN;
    uint32_t Mb;
    uint32_t Nb;
    uint32_t Kb;
    uint32_t kPasses;
    uint32_t wChunkKPasses;
    uint32_t phaseXChunk;
    uint32_t phaseXFullPairs;
    uint32_t phaseXTail;
    uint32_t maxOneTurnToken;
    uint32_t maxWeightColOneTurn;
    uint32_t perCoreIters;
    uint32_t remainderIters;
    uint32_t weightBlockN1;
    uint32_t nBlockSize;
    uint32_t flags;
    uint64_t ubKMask;
};
}  // namespace AscendC

#define QuantMatmulDequantTilingData OptQuantMatmulDequantTilingData
#include "../../quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant.h"
#include "../../quant_grouped_matmul_dequant/op_kernel/quant_grouped_matmul_dequant_staged_run.h"
#undef QuantMatmulDequantTilingData

namespace AscendC {
__aicore__ inline void BuildOptTilingData(const QuantMatmulDequantTilingData *__restrict src,
                                          OptQuantMatmulDequantTilingData &dst)
{
  dst.coreNum = src->optCoreNum;
  dst.MCoreNum = src->optMCoreNum;
  dst.NCoreNum = src->optNCoreNum;
  dst.M = src->optM;
  dst.K = src->optK;
  dst.N = src->optN;
  dst.E = src->optE;
  dst.originKAligned32 = src->optOriginKAligned32;
  dst.originKAligned512 = src->optOriginKAligned512;
  dst.fracM = src->optFracM;
  dst.fracK = src->optFracK;
  dst.fracN = src->optFracN;
  dst.Mb = src->optMb;
  dst.Nb = src->optNb;
  dst.Kb = src->optKb;
  dst.kPasses = src->optKPasses;
  dst.wChunkKPasses = src->optWChunkKPasses;
  dst.phaseXChunk = src->optPhaseXChunk;
  dst.phaseXFullPairs = src->optPhaseXFullPairs;
  dst.phaseXTail = src->optPhaseXTail;
  dst.maxOneTurnToken = src->optMaxOneTurnToken;
  dst.maxWeightColOneTurn = src->optMaxWeightColOneTurn;
  dst.perCoreIters = src->optPerCoreIters;
  dst.remainderIters = src->optRemainderIters;
  dst.weightBlockN1 = src->optWeightBlockN1;
  dst.nBlockSize = src->optNBlockSize;
  dst.flags = src->optFlags;
  dst.ubKMask = src->ubKMask;
}
}  // namespace AscendC

#define QGMMD_ORIGIN_OPT_DISPATCH(hasSmooth, scaleIsU64, smallM)                         \
  do {                                                                                   \
    QuantGroupedMatmulDequant<true, hasSmooth, scaleIsU64, false, smallM, false> op;     \
    op.Init(x, quantized_weight, weight_scale, group_list, bias, x_scale,                \
            x_offset, smooth_scale, y, usrworkspace, &optTiling, &tPipe);                \
    op.Process();                                                                        \
  } while (0)

#define QGMMD_ORIGIN_OPT_DISPATCH_SMOOTH(scaleIsU64, smallM)                            \
  do {                                                                                   \
    if (hasSmooth) {                                                                     \
      QGMMD_ORIGIN_OPT_DISPATCH(true, scaleIsU64, smallM);                               \
    } else {                                                                             \
      QGMMD_ORIGIN_OPT_DISPATCH(false, scaleIsU64, smallM);                              \
    }                                                                                    \
  } while (0)

#define QGMMD_ORIGIN_OPT_DISPATCH_SCALE(smallM)                                         \
  do {                                                                                   \
    if (scaleIsU64) {                                                                    \
      QGMMD_ORIGIN_OPT_DISPATCH_SMOOTH(true, smallM);                                    \
    } else {                                                                             \
      QGMMD_ORIGIN_OPT_DISPATCH_SMOOTH(false, smallM);                                   \
    }                                                                                    \
  } while (0)

extern "C" __global__ __aicore__ void quant_grouped_matmul_dequant(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR group_list,
                                                           GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale,
                                                           GM_ADDR y, GM_ADDR usrworkspace, GM_ADDR qmmTiling) {
  SetAtomicNone();
  GET_TILING_DATA(tiling_data, qmmTiling);
  QuantMatmulDequantTilingData* __restrict tilingData = const_cast<QuantMatmulDequantTilingData *> (&tiling_data);
  if (tilingData->perToken && tilingData->dynamicQuant) {
    OptQuantMatmulDequantTilingData optTiling;
    BuildOptTilingData(tilingData, optTiling);
    if (GetFlag(optTiling.flags, FLAG_BIT_USE_XZN_WORKSPACE)) {
      RunStagedB0(x, quantized_weight, weight_scale, group_list, bias, x_scale,
                  x_offset, smooth_scale, y, usrworkspace, &optTiling);
      SetMaskNorm();
      ResetMask();
      return;
    }
    const bool hasSmooth = GetFlag(optTiling.flags, FLAG_BIT_HAS_SMOOTH);
    const bool scaleIsU64 = GetFlag(optTiling.flags, FLAG_BIT_SCALE_IS_INT64);
    const bool smallM = GetFlag(optTiling.flags, FLAG_BIT_SMALL_M);
    TPipe tPipe;
    if (smallM) {
      QGMMD_ORIGIN_OPT_DISPATCH_SCALE(true);
    } else {
      QGMMD_ORIGIN_OPT_DISPATCH_SCALE(false);
    }
    SetMaskNorm();
    ResetMask();
    return;
  }
  if (TILING_KEY_IS(10000003)) {
    QuantMatmulDequantGrouped handle;
    handle.Init(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, y, usrworkspace, tilingData);
    handle.Process();
  } else if (TILING_KEY_IS(10000004)){
    QuantMatmulDequantGrouped handle;
    handle.SetSwift();
    handle.Init(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, y, usrworkspace, tilingData);
    handle.Process();
  }
  SetMaskNorm();
  ResetMask();
}
