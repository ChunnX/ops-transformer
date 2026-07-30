/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <kernel_operator.h>
#include "allto_all_matmul_apace_tiling_key.h"
#include "allto_all_matmul_apace_tiling_data.h"
#include "apace/kernel/all_to_all_quant_matmul/all_to_all_mx_matmul_udma_impl.h"

using namespace AscendC;
using namespace Apace;

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS, bool ISSMALLK, uint32_t COMMTYPE>
__global__ __aicore__ void allto_all_matmul_apace(GM_ADDR context, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                                                  GM_ADDR x1_scale, GM_ADDR x2_scale,
                                                  GM_ADDR comm_scale, GM_ADDR x1_offset, GM_ADDR x2_offset,
                                                  GM_ADDR y, GM_ADDR all2all_out,
                                                  GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)bias; (void)comm_scale; (void)x1_offset; (void)x2_offset; (void)all2all_out;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    REGISTER_TILING_DEFAULT(AlltoAllMatmulApaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulApaceTilingData, tilingData, tilingGM);

    if (AscendC::GetBlockIdx() == 0) {
        AscendC::printf("[APT_ADDR] c=%p x1=%p x2=%p s1=%p s2=%p y=%p ws=%p t=%p\n", context, x1, x2, x1_scale,
                        x2_scale, y, workspaceGM, tilingGM);
        auto &cm = tilingData.tilingData.commTilingData;
        auto &mm = tilingData.tilingData.tileQbmmTilingData;
        AscendC::printf("[APT_COMM] sz=%u cnt=%u tsz=%u tcn=%u nsz=%u\n", static_cast<uint32_t>(cm.splitAxisTileSize),
                        static_cast<uint32_t>(cm.splitAxisTileCnt), static_cast<uint32_t>(cm.splitAxisTailSize),
                        static_cast<uint32_t>(cm.splitAxisTailCnt), static_cast<uint32_t>(cm.nonSplitAxisSize));
        auto &scm = tilingData.tilingData.scaleCommTilingData;
        AscendC::printf("[APT_SCALE] sz=%u cnt=%u tsz=%u tcn=%u nsz=%u\n",
                        static_cast<uint32_t>(scm.splitAxisTileSize),
                        static_cast<uint32_t>(scm.splitAxisTileCnt),
                        static_cast<uint32_t>(scm.splitAxisTailSize),
                        static_cast<uint32_t>(scm.splitAxisTailCnt),
                        static_cast<uint32_t>(scm.nonSplitAxisSize));
        AscendC::printf("[APT_MM] m=%u n=%u k=%u bM=%u bN=%u bK=%u sKL1=%u\n", mm.m, mm.n, mm.k, mm.baseM, mm.baseN,
                        mm.baseK, mm.scaleKL1);
        AscendC::printf("[APT_MM2] mT=%u nT=%u mS=%u nS=%u mM=%u nM=%u cN=%u sK=%u nb=%u dL=%u loc=%u\n", mm.mTailTile,
                        mm.nTailTile, mm.mBaseTailSplitCnt, mm.nBaseTailSplitCnt, mm.mTailMain, mm.nTailMain,
                        mm.usedCoreNum, mm.stepK, mm.nBufferNum, mm.dbL0c, tilingData.tilingData.localMatmul);
        AscendC::printf("[APT_PM] precisionMode(loc)=%u\n", tilingData.tilingData.localMatmul);
    }

    __gm__ CommContext *hcommCtx = reinterpret_cast<__gm__ CommContext *>(context);

#if (ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN && ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN)
    using Impl = AllToAllQuantMatmulUdmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true>;
#elif (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2 && ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)
    using Impl = AllToAllQuantMatmulUdmaImpl<fp8_e5m2_t, fp8_e5m2_t, bfloat16_t, false, true>;
#elif (ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN && ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)
    using Impl = AllToAllQuantMatmulUdmaImpl<fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, false, true>;
#else
    using Impl = AllToAllQuantMatmulUdmaImpl<fp8_e5m2_t, fp8_e4m3fn_t, bfloat16_t, false, true>;
#endif

    Impl impl;
    impl.Init(hcommCtx, x1, x1_scale, x2, x2_scale, y, &tilingData.tilingData);
    impl.Run();
}
