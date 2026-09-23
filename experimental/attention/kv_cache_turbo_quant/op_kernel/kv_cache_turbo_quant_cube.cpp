/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_CUBE_ONLY)
extern "C" void kv_cache_turbo_quant_aic_object_is_built_with_ASCENDC_CUBE_ONLY() {}
#else
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

template <bool TransB>
class Fp32Matmul {
public:
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, float, TransB>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;

    Matmul<AType, BType, CType, BiasType> mm;
    GlobalTensor<float> aGm;
    GlobalTensor<float> bGm;
    GlobalTensor<float> cGm;
    TCubeTiling tiling;

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGm)
    {
        auto *src = reinterpret_cast<__gm__ uint8_t *>(tilingGm);
        auto *dst = reinterpret_cast<uint8_t *>(&tiling);
        for (uint32_t i = 0; i < sizeof(TCubeTiling); ++i) {
            dst[i] = src[i];
        }
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a), static_cast<uint32_t>(tiling.M) * tiling.Ka);
        const uint32_t bCount = TransB ? static_cast<uint32_t>(tiling.N) * tiling.Kb
                                       : static_cast<uint32_t>(tiling.Kb) * tiling.N;
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b), bCount);
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), static_cast<uint32_t>(tiling.M) * tiling.N);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        const int32_t coreNum = tiling.usedCoreNum > 0 ? tiling.usedCoreNum : 1;
        if (blockIdx >= coreNum) {
            return;
        }
        const int32_t mTotal = tiling.M;
        const int32_t n = tiling.N;
        const int32_t k = tiling.Ka;
        const int32_t base = mTotal / coreNum;
        const int32_t rem = mTotal % coreNum;
        const int32_t myM = base + (blockIdx < rem ? 1 : 0);
        const int32_t mOff = blockIdx * base + (blockIdx < rem ? blockIdx : rem);
        if (myM <= 0) {
            return;
        }
        mm.SetOrgShape(myM, n, k);
        mm.SetSingleShape(myM, n, k);
        mm.SetTensorA(aGm[mOff * k], false);
        mm.SetTensorB(bGm, TransB);
        mm.IterateAll(cGm[mOff * n]);
        mm.End();
    }
};

extern "C" __global__ __aicore__ void mm_fp32_transb(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    TPipe pipe;
    Fp32Matmul<true> op;
    op.Init(a, b, c, tilingGm);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm, &op.tiling);
    op.Process();
}

#endif
