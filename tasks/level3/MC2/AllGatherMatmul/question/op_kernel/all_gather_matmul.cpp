/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_matmul.cpp
 * \brief
 */
#include "basic_api/kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "all_gather_matmul_tiling.h"
#include "all_gather_matmul_tiling_key.h"
#if __CCE_AICORE__ == 310
#else
    #include "all_gather_matmul_full_mesh.h"
#endif

using namespace AscendC;
template <class T> struct BiasType {
    using type = float;
};
template <> struct BiasType<half> {
    using type = half;
};

template<bool IsFullMesh, bool IsNd2Nz, bool IsBias>
__global__ __aicore__ void all_gather_matmul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
}