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
 * \file compressor_grad_tiling_data.h
 * \brief CompressorGrad arch22 (A3 / ascend910_93) tiling data: union of the
 *        cast / scatter / gemm kernel init params plus workspace byte offsets.
 *        Plain struct shared by host tiling and kernel (mirrors compressor
 *        arch22 CompressorTilingData); named to avoid clashing with the pypto
 *        generated CompressorGradTiling and the legacy CompressorGradTilingData.
 */

#ifndef COMPRESSOR_GRAD_TILING_DATA_A3_H
#define COMPRESSOR_GRAD_TILING_DATA_A3_H

#include <cstdint>

namespace optiling {
struct CompressorGradTilingDataA3 {
    int64_t batchSize = 0;      // B
    int64_t seqSize = 0;        // S
    int64_t tLen = 0;           // T = B * S
    int64_t tPad = 0;           // T padded up to a multiple of 128
    int64_t hiddenSize = 0;     // H
    int64_t headDim = 0;        // D
    int64_t totalHeadDim = 0;   // C = coff * D
    int64_t cmpRatio = 0;       // cr
    int64_t coff = 0;           // 1 or 2
    int64_t cmpRowCnt = 0;      // N = coff * cr
    int64_t nblk = 0;           // S / cr (complete blocks)
    int64_t nblkIn = 0;         // d_cmp_kv block dim (>= nblk)
    int64_t dnPlanes = 0;       // dn workspace planes: fp16 HiLo = 2, bf16 = 1
    int64_t castCoreNum = 0;    // AIV cores used by the cast/zero phase
    int64_t scatterCoreNum = 0; // AIV cores used by the scatter phase
    int64_t blocksPerCore = 0;  // scatter blocks per AIV core
    int64_t segLen = 0;         // scatter D-segment length (64/32/16)
    int64_t gemmDxCores = 0;    // gDx: AIC cores assigned to the d_x branch
    // workspace byte offsets from the user workspace base
    int64_t xF16WsOff = 0;     // x cast to fp16, blocked-NZ, tPad * H elements
    int64_t wkvF16WsOff = 0;   // wkv cast to fp16, blocked-NZ, C * H elements
    int64_t wgateF16WsOff = 0; // wgate cast to fp16, blocked-NZ, C * H elements
    int64_t dnKvWsOff = 0;     // dn_kv fp16, dnPlanes * tPad * C elements
    int64_t dnScWsOff = 0;     // dn_sc fp16, dnPlanes * tPad * C elements
};
} // namespace optiling

#endif // COMPRESSOR_GRAD_TILING_DATA_A3_H
